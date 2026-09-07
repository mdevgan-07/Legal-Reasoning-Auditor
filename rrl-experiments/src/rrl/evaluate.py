"""
Evaluation utilities: per-class F1 report, val-test gap tracking, and a
shared results.csv so every experiment lands in one comparable table.
Also the post-hoc structural-constraint pass (Tier B3).
"""

from __future__ import annotations

import csv
import json
import os
from datetime import datetime

from sklearn.metrics import classification_report, f1_score


def macro_f1(gold_docs: list[list[int]], pred_docs: list[list[int]],
             skip_ids: set[int]) -> float:
    g = [t for d in gold_docs for t in d if t not in skip_ids]
    p = [t for d, gd in zip(pred_docs, gold_docs)
         for t, gt in zip(d, gd) if gt not in skip_ids]
    return f1_score(g, p, average="macro", zero_division=0)


def per_class_report(gold_docs, pred_docs, tag2idx) -> dict:
    idx2tag = {v: k for k, v in tag2idx.items()}
    skip = {"<pad>", "<start>", "<end>"}
    labels = [v for k, v in tag2idx.items() if k not in skip]
    names = [idx2tag[l] for l in labels]
    g = [t for d in gold_docs for t in d]
    p = [t for d in pred_docs for t in d]
    rep = classification_report(g, p, labels=labels, target_names=names,
                                digits=4, zero_division=0, output_dict=True)
    print(classification_report(g, p, labels=labels, target_names=names,
                                digits=4, zero_division=0))
    return rep


RESULT_FIELDS = [
    "timestamp", "name", "split", "macro_f1",
    "None", "Facts", "Issue",
    "Arguments of Petitioner", "Arguments of Respondent",
    "Reasoning", "Decision",
    "val_test_gap", "config",
]


def append_result(out_dir: str, name: str, split: str, report: dict,
                  macro: float, val_test_gap: float | None,
                  config: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "results.csv")
    exists = os.path.exists(path)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "name": name, "split": split, "macro_f1": round(macro, 4),
        "val_test_gap": round(val_test_gap, 4) if val_test_gap is not None else "",
        "config": json.dumps(config, separators=(",", ":"))[:800],
    }
    for cls in ["None", "Facts", "Issue", "Arguments of Petitioner",
                "Arguments of Respondent", "Reasoning", "Decision"]:
        row[cls] = round(report.get(cls, {}).get("f1-score", 0.0), 4)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(f"[results] appended {name}/{split} -> {path}")


# ---------------------------------------------------------------------------
# Tier B3: post-hoc structural constraints on Viterbi output.
# Conservative rules only — each is a documented, testable heuristic.
# ---------------------------------------------------------------------------

def apply_structural_constraints(pred: list[int], tag2idx: dict) -> list[int]:
    """Rules (all conservative):
    R1: an isolated single-sentence role sandwiched between two identical
        roles is smoothed to the neighbor role (fixes 1-token label noise),
        EXCEPT protected rare roles (Issue, Decision) which are never removed.
    R2: Decision before any Facts/Reasoning in the first 20% of the document
        is relabeled to the previous label (Decisions do not open judgments).
    """
    if not pred:
        return pred
    inv = {v: k for k, v in tag2idx.items()}
    protected = {tag2idx.get("Issue"), tag2idx.get("Decision")}
    out = list(pred)

    # R1 — median smoothing of singletons
    for i in range(1, len(out) - 1):
        if out[i - 1] == out[i + 1] and out[i] != out[i - 1] and out[i] not in protected:
            out[i] = out[i - 1]

    # R2 — no opening Decisions
    dec = tag2idx.get("Decision")
    cutoff = max(1, int(0.2 * len(out)))
    seen_body = False
    body = {tag2idx.get("Facts"), tag2idx.get("Reasoning")}
    for i in range(len(out)):
        if out[i] in body:
            seen_body = True
        if i < cutoff and out[i] == dec and not seen_body:
            out[i] = out[i - 1] if i > 0 else tag2idx.get("None", out[i])
    return out
