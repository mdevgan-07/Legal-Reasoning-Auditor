"""
RRL adapter: turn FinetunedInLegalBERT_BiLSTM_CRF predictions into the
`Judgment` schema the auditor consumes.

Two entry points:

1. from_prediction_file(...)  — offline: you already have label-id sequences
   (like finetuned_bert_predictions.json) PLUS the sentence texts. Use this to
   batch-convert your test set into Judgments for the benchmark.

2. RRLInference.predict(sentences) — online: wraps your loaded model to tag a
   fresh document end to end. Fill in the two marked hooks with your own
   inference code (you already have it from training/eval).

################################  ACTION REQUIRED  ############################
# Confirm this mapping against your training `label2id` dict. Label ids 3-9
# appear in the prediction file. If any pair is wrong, FIX IT HERE ONLY —
# every downstream stage reads from this single dict.
##############################################################################
"""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Judgment, Role, Sentence

# ---- CONFIRMED MAPPING (from training label2id) -------------------------
# 0=<pad> 1=<start> 2=<end> are CRF structural tags, never real roles.
LABEL_ID_TO_ROLE: dict[int, Role] = {
    3: Role.NONE,
    4: Role.FACTS,
    5: Role.ISSUE,
    6: Role.AOP,       # Arguments of Petitioner
    7: Role.AOR,       # Arguments of Respondent
    8: Role.REASONING,
    9: Role.DECISION,
}
# Structural CRF tags — if they ever appear in output, treat as None (skip).
STRUCTURAL_IDS = {0, 1, 2}
# --------------------------------------------------------------------------


def role_from_id(label_id: int) -> Role:
    if label_id not in LABEL_ID_TO_ROLE:
        return Role.NONE
    return LABEL_ID_TO_ROLE[label_id]


# ==========================================================================
# 1. Offline: prediction file (+ sentence texts) -> Judgments
# ==========================================================================

def from_prediction_file(
    predictions_path: str | Path,
    sentences_path: str | Path,
    use: str = "test_pred",
    confidences_path: str | Path | None = None,
) -> list[Judgment]:
    """
    predictions_path : JSON like {"test_gold": [[id,...],...], "test_pred": [...]}
    sentences_path   : JSON list-of-lists of raw sentence strings, aligned 1:1
                       with the docs/sentences in predictions_path. You must
                       export this alongside predictions (the model saw texts;
                       just dump them in the same order).
    use              : "test_pred" (model output) or "test_gold" (oracle RRL,
                       useful to measure the auditor in isolation from RRL error)
    confidences_path : optional JSON of per-sentence max-softmax floats, same
                       shape as predictions. Populates role_confidence.
    """
    pred_blob = json.loads(Path(predictions_path).read_text())
    label_docs = pred_blob[use]
    text_docs = json.loads(Path(sentences_path).read_text())
    conf_docs = json.loads(Path(confidences_path).read_text()) if confidences_path else None

    if len(label_docs) != len(text_docs):
        raise ValueError(
            f"doc count mismatch: {len(label_docs)} label docs vs {len(text_docs)} text docs"
        )

    judgments: list[Judgment] = []
    for di, (labels, texts) in enumerate(zip(label_docs, text_docs)):
        # labels may be padded to 200; trust the texts for the true length
        n = len(texts)
        labels = labels[:n]
        if len(labels) < n:
            raise ValueError(f"doc {di}: fewer labels ({len(labels)}) than sentences ({n})")
        confs = conf_docs[di][:n] if conf_docs else [1.0] * n

        sents = [
            Sentence(
                sid=f"s{si:03d}",
                text=str(t).strip(),
                role=role_from_id(int(lab)),
                role_confidence=float(cf),
            )
            for si, (lab, t, cf) in enumerate(zip(labels, texts, confs))
        ]
        judgments.append(Judgment(doc_id=f"test_doc_{di:04d}", sentences=sents))
    return judgments


# ==========================================================================
# 2. Online: wrap the live model
# ==========================================================================

class RRLInference:
    """Wraps FinetunedInLegalBERT_BiLSTM_CRF for tagging fresh documents.

    Fill the two hooks with the inference code you already have from eval.
    """

    def __init__(self, model, tokenizer, device: str = "cuda", max_sentences: int = 200):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_sentences = max_sentences
        self.model.to(device).eval()

    def _encode(self, sentences: list[str]):
        """HOOK 1 — sentence list -> per-sentence [CLS] embeddings [N,768].
        Paste your BERT-encoding loop (bert_batch_size=16, [CLS] token)."""
        raise NotImplementedError("paste your BERT encoding loop here")

    def _decode(self, embeddings):
        """HOOK 2 — embeddings -> (label_ids, confidences) via BiLSTM+CRF Viterbi.
        Paste your forward + crf.decode call. Return per-sentence max-marginal
        or softmax confidence if available, else a list of 1.0s."""
        raise NotImplementedError("paste your BiLSTM+CRF decode here")

    def predict(self, sentences: list[str], doc_id: str = "live_doc") -> Judgment:
        sentences = [s.strip() for s in sentences if s.strip()][: self.max_sentences]
        emb = self._encode(sentences)
        label_ids, confs = self._decode(emb)
        sents = [
            Sentence(sid=f"s{i:03d}", text=t, role=role_from_id(int(l)), role_confidence=float(c))
            for i, (t, l, c) in enumerate(zip(sentences, label_ids, confs))
        ]
        return Judgment(doc_id=doc_id, sentences=sents)


def sanity_check_mapping(predictions_path: str | Path) -> None:
    """Print per-role sentence counts under the current mapping so you can eyeball
    whether it's sane (e.g. Facts should dominate, Issue/Decision should be rare)."""
    from collections import Counter

    blob = json.loads(Path(predictions_path).read_text())
    c = Counter()
    for doc in blob["test_pred"]:
        c.update(role_from_id(int(x)).value for x in doc)
    print("Sentence count per role under current LABEL_ID_TO_ROLE:")
    for role, n in c.most_common():
        print(f"  {role:<28} {n}")
    print("\nSanity: Facts should be large; Issue & Decision should be small.")
    print("If this looks wrong, fix LABEL_ID_TO_ROLE in rrl_adapter.py.")
