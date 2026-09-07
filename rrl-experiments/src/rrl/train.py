"""
Unified training loop for all presets.

Handles: differential LRs (BERT vs head), linear warmup + cosine decay,
gradient accumulation over documents, early stopping on val macro-F1,
top-k checkpoint retention, and checkpoint AVERAGING (the val-test-gap
countermeasure: averaging the top-k best-val checkpoints removes the
"lucky checkpoint" selection bias).
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time

import torch

from .config import ExperimentConfig
from .data import Doc, add_context, load_all
from .encoders import SentenceEncoder
from .evaluate import (append_result, apply_structural_constraints,
                       macro_f1, per_class_report)
from .model import RRLModel


def set_seed(seed: int) -> None:
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _lr_lambda(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * prog))


def average_state_dicts(paths: list[str]) -> dict:
    avg = None
    for p in paths:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if avg is None:
            avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k in avg:
                avg[k] += sd[k].float()
    for k in avg:
        avg[k] /= len(paths)
    return avg


class Trainer:
    def __init__(self, cfg: ExperimentConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        set_seed(cfg.seed)

        from transformers import AutoTokenizer
        self.tag2idx = json.loads(open(cfg.tag2idx_path).read())
        self.skip_ids = {self.tag2idx[k] for k in ("<pad>", "<start>", "<end>")}
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.bert_dir)

        self.encoder = SentenceEncoder(
            cfg.bert_dir, pooling=cfg.pooling,
            n_fine_tune_layers=cfg.n_fine_tune_layers,
            bert_batch_size=cfg.bert_batch_size, lora=cfg.lora,
            lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout, freeze=cfg.freeze_bert,
            grad_checkpoint=cfg.grad_checkpoint,
            device=device,
        ).to(device)

        self.model = RRLModel(
            cfg, n_tags=len(self.tag2idx),
            sos=self.tag2idx["<start>"], eos=self.tag2idx["<end>"],
            pad=self.tag2idx["<pad>"], in_dim=self.encoder.out_dim,
            device=device,
        ).to(device)

        if cfg.init_ckpt:
            sd = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)
            missing = self.model.load_state_dict(sd.get("model", sd), strict=False)
            print(f"[init] warm start from {cfg.init_ckpt} ({missing})")

        self.run_dir = os.path.join(cfg.out_dir, cfg.name)
        os.makedirs(self.run_dir, exist_ok=True)
        json.dump(cfg.to_dict(), open(os.path.join(self.run_dir, "config.json"), "w"), indent=2)

    # ------------------------------------------------------------------
    def _doc_forward(self, doc: Doc):
        sents = add_context(doc.sentences, self.cfg.context)
        embs = self.encoder.encode(sents, self.tokenizer, self.cfg.max_length)
        return self.model.emissions_from_embeddings([embs])

    @torch.no_grad()
    def evaluate_split(self, docs: list[Doc], constraints: bool = False):
        self.encoder.eval(); self.model.eval()
        gold, pred = [], []
        for doc in docs:
            emissions, mask = self._doc_forward(doc)
            path = self.model.decode(emissions, mask)[0]
            if constraints:
                path = apply_structural_constraints(path, self.tag2idx)
            gold.append(doc.labels)
            pred.append(path[: len(doc.labels)])
        return gold, pred, macro_f1(gold, pred, self.skip_ids)

    # ------------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        splits = load_all(cfg, self.tag2idx)
        train_docs, val_docs, test_docs = splits["train"], splits["val"], splits["test"]
        print(f"[data] train={len(train_docs)} val={len(val_docs)} test={len(test_docs)} "
              f"| window={cfg.window} max_length={cfg.max_length} context={cfg.context}")

        # ---- imbalance toolkit -------------------------------------------
        if cfg.weight_scheme != "manual":
            from .imbalance import compute_class_weights
            w = compute_class_weights(train_docs, len(self.tag2idx),
                                      scheme=cfg.weight_scheme, beta=cfg.cb_beta,
                                      structural_ids=self.skip_ids)
            self.model.set_class_weights(w)
            named = {k: round(w[v], 3) for k, v in self.tag2idx.items()
                     if v not in self.skip_ids}
            print(f"[weights] scheme={cfg.weight_scheme}: {named}")
        if self.model.none_head is not None:
            self.model.none_id = self.tag2idx.get("None")
        rare_ids = {self.tag2idx[c] for c in cfg.rare_classes if c in self.tag2idx}
        if cfg.oversample_boost > 1.0:
            from .imbalance import oversample_docs
            sched = oversample_docs(train_docs, rare_ids, boost=cfg.oversample_boost)
            print(f"[oversample] boost={cfg.oversample_boost} rare={sorted(cfg.rare_classes)} "
                  f"-> epoch schedule {len(train_docs)} -> {len(sched)} docs")
        else:
            sched = list(train_docs)

        bert_params = [p for p in self.encoder.parameters() if p.requires_grad]
        head_params = list(self.model.parameters())
        opt = torch.optim.AdamW(
            [{"params": bert_params, "lr": cfg.lr_bert},
             {"params": head_params, "lr": cfg.lr_head}],
            weight_decay=cfg.weight_decay,
        )
        steps_per_epoch = max(1, len(sched) // cfg.grad_accum_docs)
        total = steps_per_epoch * cfg.epochs
        lr_sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: _lr_lambda(s, total, int(cfg.warmup_frac * total)))

        best, bad_epochs = -1.0, 0
        topk: list[tuple[float, str]] = []          # (val_f1, path)

        for epoch in range(1, cfg.epochs + 1):
            self.encoder.train(); self.model.train()
            random.shuffle(sched)
            t0, running = time.time(), 0.0
            opt.zero_grad()
            for i, doc in enumerate(sched):
                emissions, mask = self._doc_forward(doc)
                loss = self.model.loss(emissions, mask, [doc.labels]) / cfg.grad_accum_docs
                loss.backward()
                running += float(loss.item())
                if (i + 1) % cfg.grad_accum_docs == 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.model.parameters()), 5.0)
                    opt.step(); lr_sched.step(); opt.zero_grad()

            _, _, val_f1 = self.evaluate_split(val_docs)
            print(f"[epoch {epoch:02d}] loss={running:.2f} val_macroF1={val_f1:.4f} "
                  f"({time.time()-t0:.0f}s)")

            ck = os.path.join(self.run_dir, f"ckpt_e{epoch:02d}_f{val_f1:.4f}.pt")
            torch.save({"model": {**self.model.state_dict()},
                        "encoder": self.encoder.state_dict(),
                        "epoch": epoch, "val_f1": val_f1}, ck)
            topk.append((val_f1, ck))
            topk.sort(key=lambda t: t[0], reverse=True)
            for _, old in topk[cfg.keep_top_k:]:
                if os.path.exists(old):
                    os.remove(old)
            topk = topk[: cfg.keep_top_k]

            if val_f1 > best:
                best, bad_epochs = val_f1, 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f"[early stop] no val improvement for {cfg.patience} epochs")
                    break

        # ---- final evaluation: best single checkpoint --------------------
        best_f1, best_path = topk[0]
        sd = torch.load(best_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(sd["model"]); self.encoder.load_state_dict(sd["encoder"])
        self._final_eval(val_docs, test_docs, tag=cfg.name)

        # ---- averaged top-k ------------------------------------------------
        if cfg.average_top_k and len(topk) > 1:
            print(f"[avg] averaging top-{len(topk)} checkpoints "
                  f"(val F1s: {[round(f,4) for f,_ in topk]})")
            self.model.load_state_dict(average_state_dicts([p for _, p in topk]))
            # note: encoder stays at best-ckpt weights; head averaging is the
            # main gap-closer and avoids doubling checkpoint size for BERT.
            self._final_eval(val_docs, test_docs, tag=f"{cfg.name}+avg{len(topk)}")

        if cfg.apply_constraints:
            g, p, tf1 = self.evaluate_split(test_docs, constraints=True)
            print(f"[constraints] test macroF1 with structural pass: {tf1:.4f}")

    # ------------------------------------------------------------------
    def _final_eval(self, val_docs, test_docs, tag: str):
        vg, vp, vf1 = self.evaluate_split(val_docs)
        tg, tp, tf1 = self.evaluate_split(test_docs)
        gap = vf1 - tf1
        print(f"\n===== {tag} ===== val={vf1:.4f} test={tf1:.4f} gap={gap:+.4f}")
        rep = per_class_report(tg, tp, self.tag2idx)
        append_result(self.cfg.out_dir, tag, "test", rep, tf1, gap, self.cfg.to_dict())
