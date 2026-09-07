"""
Live RRL inference — wraps YOUR trained FinetunedInLegalBERT_BiLSTM_CRF so
raw documents (from PDF ingestion) get real rhetorical-role labels.

Written against the exact class in
  /workspace/legal_capstone/code/FinetunedBERT/finetuned_model.py
whose forward(batch_docs, tokenizer, max_length) takes raw sentence lists and
returns Viterbi label-id paths, stashing emissions in self._emissions.

Usage in JupyterLab:

    from lra.rrl.live import load_rrl_labeler
    labeler = load_rrl_labeler()                      # loads ckpt once
    from lra.ingest.pdf_ingest import ingest_pdf
    j = ingest_pdf("judgment.pdf", labeler=labeler)   # real RRL labels

Confidence note: the CRF emits hard Viterbi paths (no marginals), so
per-sentence confidence is approximated as the softmax of the emission scores
at each position (transition contribution ignored). This is a reasonable
proxy: high emission confidence ⇒ the label didn't depend on transition
smoothing. Documented for your report's methods section.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

from ..schema import Role
from .adapter import role_from_id

_DEFAULTS = {
    "code_dir": "/workspace/legal_capstone/code/FinetunedBERT",
    "tag2idx": "/workspace/legal_capstone/saved_models/tag2idx.json",
    "bert_dir": "/workspace/legal_capstone/saved_models/InLegalBERT_safe",
    "ckpt": "/workspace/legal_capstone/saved_models/finetuned_bert_best.tar",
}

# class weights exactly as used in training (index-aligned with tag2idx)
_CLASS_WEIGHTS = [0.0, 0.0, 0.0, 0.4, 1.0, 5.0, 2.0, 2.5, 1.2, 4.0]


class LiveRRL:
    def __init__(
        self,
        code_dir: str = _DEFAULTS["code_dir"],
        tag2idx_path: str = _DEFAULTS["tag2idx"],
        bert_dir: str = _DEFAULTS["bert_dir"],
        ckpt_path: str = _DEFAULTS["ckpt"],
        device: str = "cuda",
        max_sentences: int = 200,
        max_length: int = 128,
    ):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_sentences = max_sentences
        self.max_length = max_length

        if code_dir not in sys.path:
            sys.path.append(code_dir)
        from finetuned_model import FinetunedInLegalBERT_BiLSTM_CRF  # your class

        self.tag2idx = json.loads(Path(tag2idx_path).read_text())
        self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)

        class_weights = torch.tensor(_CLASS_WEIGHTS, dtype=torch.float32).to(device)
        self.model = FinetunedInLegalBERT_BiLSTM_CRF(
            n_tags=len(self.tag2idx),
            sos_tag_idx=self.tag2idx["<start>"],
            eos_tag_idx=self.tag2idx["<end>"],
            pad_tag_idx=self.tag2idx["<pad>"],
            class_weights=class_weights,
            bert_model_name=bert_dir,
            hidden_dim=512,
            lstm_dropout=0.5,
            n_fine_tune_layers=12,
            bert_batch_size=16,
            device=device,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        print(f"[live-rrl] loaded epoch {ckpt.get('epoch','?')} "
              f"val_F1={ckpt.get('val_f1', float('nan')):.4f}")

    # ------------------------------------------------------------------
    def label(self, sentences: list[str]) -> list[tuple[Role, float]]:
        """sentences -> [(Role, confidence)], aligned 1:1 (input order kept)."""
        torch = self.torch
        sentences = [s if s.strip() else "." for s in sentences][: self.max_sentences]
        if not sentences:
            return []
        with torch.no_grad():
            paths = self.model([sentences], self.tokenizer, self.max_length)
        label_ids = paths[0]

        # emission-softmax confidence per position (see module docstring)
        emissions = self.model._emissions[0, : len(label_ids)]      # [N, n_tags]
        probs = torch.softmax(emissions, dim=-1)
        confs = [float(probs[i, lid]) for i, lid in enumerate(label_ids)]

        return [(role_from_id(int(lid)), c) for lid, c in zip(label_ids, confs)]


def load_rrl_labeler(**kw) -> Callable[[list[str]], list[tuple[Role, float]]]:
    """One-call factory: loads the model and returns a labeler callable that
    plugs directly into ingest_pdf(labeler=...) / ingest_text(labeler=...)."""
    live = LiveRRL(**kw)

    def labeler(sentences: list[str]) -> list[tuple[Role, float]]:
        out = live.label(sentences)
        # pad if the model truncated at max_sentences
        while len(out) < len(sentences):
            out.append((Role.NONE, 0.0))
        return out

    return labeler
