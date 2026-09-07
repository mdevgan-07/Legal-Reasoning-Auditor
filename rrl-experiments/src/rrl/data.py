"""
Data loading for LegalSeg-format splits, with the windowing fix.

Format (matches your load_raw_split): directories train/ val/ test/ of .txt
files, one "sentence<TAB>label" per line.

Windowing — the key fix. Your old setup kept the FIRST 200 sentences of each
document, but Decision sentences live at the END of judgments: for every long
document you were deleting your rarest, most up-weighted class from both
training and evaluation. `head_tail` keeps the first n_head and last n_tail
sentences (BiLSTM/Transformer still sees a coherent, order-preserving
sequence; a boundary marker is unnecessary since the CRF learns transitions
from data). `full` keeps everything up to hard_cap — chunked BERT encoding
makes memory a non-issue.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .config import ExperimentConfig


@dataclass
class Doc:
    doc_id: str
    sentences: list[str]
    labels: list[int]          # label ids (tag2idx space)
    kept_idx: list[int]        # original indices kept after windowing


def load_split(split_dir: str, tag2idx: dict) -> list[Doc]:
    docs: list[Doc] = []
    files = sorted(f for f in os.listdir(split_dir) if f.endswith(".txt"))
    for fname in files:
        sents, labs = [], []
        with open(os.path.join(split_dir, fname), encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                sent, label = parts[0].strip(), parts[-1].strip()
                if label not in tag2idx or not sent:
                    continue
                sents.append(sent)
                labs.append(tag2idx[label])
        if sents:
            docs.append(Doc(doc_id=fname, sentences=sents, labels=labs,
                            kept_idx=list(range(len(sents)))))
    return docs


def apply_window(doc: Doc, cfg: ExperimentConfig) -> Doc:
    n = len(doc.sentences)
    if cfg.window == "truncate_head":
        keep = list(range(min(n, cfg.max_sents)))
    elif cfg.window == "head_tail":
        if n <= cfg.max_sents:
            keep = list(range(n))
        else:
            head = list(range(cfg.n_head))
            tail = list(range(n - cfg.n_tail, n))
            keep = head + tail                      # order preserved, no overlap (n > head+tail)
    elif cfg.window == "full":
        if n <= cfg.hard_cap:
            keep = list(range(n))
        else:
            # over-cap: head + tail within budget — never head-truncate,
            # Decision sentences at document end must survive
            n_tail = min(cfg.n_tail, cfg.hard_cap // 4)
            n_head = cfg.hard_cap - n_tail
            keep = list(range(n_head)) + list(range(n - n_tail, n))
    else:
        raise ValueError(f"unknown window strategy: {cfg.window}")
    return Doc(
        doc_id=doc.doc_id,
        sentences=[doc.sentences[i] for i in keep],
        labels=[doc.labels[i] for i in keep],
        kept_idx=keep,
    )


def add_context(sentences: list[str], k: int, sep: str = " [SEP] ") -> list[str]:
    """Context-augmented encoding: each sentence becomes
    [i-k .. i-1] SEP [i] SEP [i+1 .. i+k]. The paper's own ablation shows
    +9 F1 from two neighbors on plain InLegalBERT — complementary to the
    sequence layer, which mixes *embeddings*, not tokens."""
    if k <= 0:
        return sentences
    out = []
    n = len(sentences)
    for i in range(n):
        left = sentences[max(0, i - k): i]
        right = sentences[i + 1: min(n, i + 1 + k)]
        parts = []
        if left:
            parts.append(" ".join(left))
        parts.append(sentences[i])
        if right:
            parts.append(" ".join(right))
        out.append(sep.join(parts))
    return out


def load_all(cfg: ExperimentConfig, tag2idx: dict) -> dict[str, list[Doc]]:
    splits = {}
    for split in ("train", "val", "test"):
        raw = load_split(os.path.join(cfg.data_dir, split), tag2idx)
        splits[split] = [apply_window(d, cfg) for d in raw]
    return splits
