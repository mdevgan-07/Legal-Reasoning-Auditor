"""
Tier B two-stage pipeline.

Stage 1 (precompute): run the (fine-tuned or base) encoder ONCE over every
document and cache embeddings to disk. Clean-experiment mode: embeddings are
regenerated fresh from the checkpoint you specify — no reuse of old .pkl
caches whose provenance is unclear.

Stage 2 (train fast): with embeddings fixed, the sequence layer + CRF trains
with REAL document batches (16 docs at once instead of 1) for many epochs in
minutes, testing the hypothesis that the paper's 0.77 comes from a thoroughly
trained sequence layer over frozen embeddings rather than a better encoder.
"""

from __future__ import annotations

import json
import os
import random
import time

import torch

from .config import ExperimentConfig
from .data import Doc, add_context, load_all
from .encoders import SentenceEncoder
from .evaluate import append_result, macro_f1, per_class_report
from .model import RRLModel
from .train import average_state_dicts, set_seed


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute(cfg: ExperimentConfig, cache_dir: str,
               encoder_ckpt: str = "", device: str = "cuda") -> None:
    """Encode all splits once; save {split}.pt with lists of tensors + labels."""
    from transformers import AutoTokenizer

    os.makedirs(cache_dir, exist_ok=True)
    tag2idx = json.loads(open(cfg.tag2idx_path).read())
    tokenizer = AutoTokenizer.from_pretrained(cfg.bert_dir)

    enc = SentenceEncoder(cfg.bert_dir, pooling=cfg.pooling,
                          freeze=True, bert_batch_size=cfg.bert_batch_size,
                          device=device).to(device).eval()
    if encoder_ckpt:
        sd = torch.load(encoder_ckpt, map_location=device, weights_only=False)
        state = sd.get("encoder", sd.get("state_dict", sd))
        # tolerate old checkpoints where keys were bert.* under the full model
        filtered = {k.replace("bert.", "bert.", 1): v for k, v in state.items()
                    if k.startswith("bert.")}
        missing = enc.load_state_dict(filtered or state, strict=False)
        print(f"[stage1] loaded encoder weights from {encoder_ckpt} ({missing})")

    splits = load_all(cfg, tag2idx)
    meta = {"pooling": cfg.pooling, "context": cfg.context,
            "max_length": cfg.max_length, "window": cfg.window,
            "encoder_ckpt": encoder_ckpt, "out_dim": enc.out_dim}
    json.dump(meta, open(os.path.join(cache_dir, "meta.json"), "w"), indent=2)

    for split, docs in splits.items():
        items, t0 = [], time.time()
        for i, doc in enumerate(docs):
            sents = add_context(doc.sentences, cfg.context)
            emb = enc.encode(sents, tokenizer, cfg.max_length).cpu()
            items.append({"doc_id": doc.doc_id, "emb": emb, "labels": doc.labels})
            if (i + 1) % 200 == 0:
                print(f"  [{split}] {i+1}/{len(docs)} ({time.time()-t0:.0f}s)", flush=True)
        torch.save(items, os.path.join(cache_dir, f"{split}.pt"))
        print(f"[stage1] {split}: {len(items)} docs cached")


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

def train_stage2(cfg: ExperimentConfig, device: str = "cuda") -> None:
    set_seed(cfg.seed)
    tag2idx = json.loads(open(cfg.tag2idx_path).read())
    skip_ids = {tag2idx[k] for k in ("<pad>", "<start>", "<end>")}
    meta = json.loads(open(os.path.join(cfg.emb_cache_dir, "meta.json")).read())
    in_dim = meta["out_dim"]

    def load(split):
        return torch.load(os.path.join(cfg.emb_cache_dir, f"{split}.pt"),
                          map_location="cpu", weights_only=False)

    train_items, val_items, test_items = load("train"), load("val"), load("test")
    print(f"[stage2] cache={cfg.emb_cache_dir} in_dim={in_dim} "
          f"train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    model = RRLModel(cfg, n_tags=len(tag2idx), sos=tag2idx["<start>"],
                     eos=tag2idx["<end>"], pad=tag2idx["<pad>"],
                     in_dim=in_dim, device=device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.stage2_lr,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.stage2_epochs)

    run_dir = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(run_dir, exist_ok=True)

    @torch.no_grad()
    def eval_items(items):
        model.eval()
        gold, pred = [], []
        B = cfg.stage2_batch_docs
        for s in range(0, len(items), B):
            batch = items[s: s + B]
            embs = [b["emb"].to(device) for b in batch]
            emissions, mask = model.emissions_from_embeddings(embs)
            paths = model.decode(emissions, mask)
            for b, p in zip(batch, paths):
                gold.append(b["labels"]); pred.append(p[: len(b["labels"])])
        return gold, pred, macro_f1(gold, pred, skip_ids)

    best, topk, bad = -1.0, [], 0
    for epoch in range(1, cfg.stage2_epochs + 1):
        model.train(); random.shuffle(train_items)
        running, t0 = 0.0, time.time()
        B = cfg.stage2_batch_docs
        for s in range(0, len(train_items), B):
            batch = train_items[s: s + B]
            embs = [b["emb"].to(device) for b in batch]
            emissions, mask = model.emissions_from_embeddings(embs)
            loss = model.loss(emissions, mask, [b["labels"] for b in batch])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.item())
        sched.step()

        _, _, vf1 = eval_items(val_items)
        print(f"[s2 epoch {epoch:02d}] loss={running:.1f} val={vf1:.4f} "
              f"({time.time()-t0:.0f}s)")
        ck = os.path.join(run_dir, f"s2_e{epoch:02d}_f{vf1:.4f}.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "val_f1": vf1}, ck)
        topk.append((vf1, ck)); topk.sort(key=lambda t: t[0], reverse=True)
        for _, old in topk[cfg.keep_top_k:]:
            if os.path.exists(old):
                os.remove(old)
        topk = topk[: cfg.keep_top_k]
        if vf1 > best:
            best, bad = vf1, 0
        else:
            bad += 1
            if bad >= cfg.patience * 2:      # stage2 epochs are cheap; be patient
                print("[early stop]")
                break

    # best single + averaged
    for tag, state in [
        (cfg.name, torch.load(topk[0][1], map_location=device, weights_only=False)["model"]),
        (f"{cfg.name}+avg{len(topk)}",
         average_state_dicts([p for _, p in topk]) if len(topk) > 1 else None),
    ]:
        if state is None:
            continue
        model.load_state_dict(state)
        vg, vp, vf1 = eval_items(val_items)
        tg, tp, tf1 = eval_items(test_items)
        print(f"\n===== {tag} ===== val={vf1:.4f} test={tf1:.4f} gap={vf1-tf1:+.4f}")
        rep = per_class_report(tg, tp, tag2idx)
        append_result(cfg.out_dir, tag, "test", rep, tf1, vf1 - tf1, cfg.to_dict())
