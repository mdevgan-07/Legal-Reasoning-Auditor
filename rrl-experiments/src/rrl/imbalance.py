"""
Class-imbalance handling.

IMPORTANT framing from the diagnostic: the classes dragging macro-F1 are
Reasoning (0.59) and None (0.66) — MAJORITY classes whose problem is mutual
confusion, not scarcity. Only Issue is a true rarity problem. Therefore:

  * effective-number weights + oversampling  -> target Issue (modest gains)
  * the aux None-vs-content head (model.py)  -> targets the Reasoning↔None
    boundary, the single largest error source (the real lever)

What is deliberately NOT here: sentence-level synthetic oversampling
(SMOTE-style). Sequence labeling depends on document structure — injecting
synthetic sentences or duplicating sentences in-place corrupts the very
transition statistics the CRF learns. Oversampling is done at DOCUMENT
granularity only.
"""

from __future__ import annotations

import math
from collections import Counter

from .data import Doc


def class_counts(docs: list[Doc], n_tags: int) -> list[int]:
    c = Counter()
    for d in docs:
        c.update(d.labels)
    return [c.get(i, 0) for i in range(n_tags)]


def compute_class_weights(
    docs: list[Doc],
    n_tags: int,
    scheme: str = "effective",
    beta: float = 0.9999,
    structural_ids: set[int] = frozenset({0, 1, 2}),
    normalize_mean_to: float = 1.5,
) -> list[float]:
    """
    scheme:
      manual        — caller keeps their hand-tuned weights (returns None-like)
      inverse_sqrt  — w_c = 1 / sqrt(n_c)   (mild)
      effective     — w_c = (1 - beta) / (1 - beta**n_c)   (Cui et al. 2019;
                      saturates for frequent classes, boosts rare ones without
                      the explosion plain inverse-frequency causes)
    Structural CRF tags (<pad>/<start>/<end>) get weight 0. Weights are
    rescaled so their mean over real classes equals `normalize_mean_to`,
    keeping the loss magnitude comparable to your original setup.
    """
    counts = class_counts(docs, n_tags)
    w = [0.0] * n_tags
    for i in range(n_tags):
        if i in structural_ids:
            continue
        n = max(1, counts[i])
        if scheme == "inverse_sqrt":
            w[i] = 1.0 / math.sqrt(n)
        elif scheme == "effective":
            w[i] = (1.0 - beta) / (1.0 - beta ** n)
        else:
            raise ValueError(f"unknown scheme: {scheme}")
    real = [w[i] for i in range(n_tags) if i not in structural_ids]
    mean = sum(real) / max(1, len(real))
    scale = normalize_mean_to / max(mean, 1e-12)
    return [x * scale for x in w]


def oversample_docs(
    docs: list[Doc],
    rare_ids: set[int],
    boost: float = 2.0,
    min_rare_sents: int = 1,
) -> list[Doc]:
    """Document-level oversampling: a doc containing >= min_rare_sents
    sentences of any rare class appears ceil(boost) times per epoch
    (fractional boost -> probabilistic extra copy handled by caller shuffle;
    we keep it deterministic: int(boost) copies + 1 more if frac>0.5).
    Returned list is a new schedule; Doc objects are shared (no copy cost)."""
    if boost <= 1.0:
        return list(docs)
    out: list[Doc] = []
    extra_int = int(boost) - 1
    extra_frac = boost - int(boost)
    for d in docs:
        out.append(d)
        n_rare = sum(1 for t in d.labels if t in rare_ids)
        if n_rare >= min_rare_sents:
            for _ in range(extra_int):
                out.append(d)
            if extra_frac > 0.5:
                out.append(d)
    return out
