#!/usr/bin/env python3
"""
RUN THIS FIRST — no training needed. Diagnoses your EXISTING model from the
prediction file you already have, answering three questions:

1. Which classes drag the test macro-F1? (per-class report)
2. How much is the head-truncation costing? (position analysis: where do
   Decision/Issue sentences sit in documents, and how many fall in the tail
   that truncate_head deletes from long documents?)
3. Is the val-test gap concentrated in specific classes?

  python scripts/run_diagnostic.py \
      --pred /workspace/legal_capstone/saved_models/finetuned_bert_predictions.json \
      --tag2idx /workspace/legal_capstone/saved_models/tag2idx.json \
      --data-dir /workspace/legal_capstone/data/Hier_BiLSTM_CRF
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.metrics import classification_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag2idx", required=True)
    ap.add_argument("--data-dir", default=None,
                    help="if given, analyses untruncated docs for tail loss")
    args = ap.parse_args()

    tag2idx = json.loads(open(args.tag2idx).read())
    idx2tag = {v: k for k, v in tag2idx.items()}
    skip = {tag2idx[k] for k in ("<pad>", "<start>", "<end>")}
    labels = sorted(set(tag2idx.values()) - skip)
    names = [idx2tag[l] for l in labels]

    blob = json.loads(open(args.pred).read())
    gold, pred = blob["test_gold"], blob["test_pred"]

    # ---- 1. per-class test report -------------------------------------
    g = [t for d in gold for t in d if t not in skip]
    p = [t for dg, dp in zip(gold, pred) for tg, t in zip(dg, dp) if tg not in skip]
    print("=" * 70)
    print("1. PER-CLASS TEST PERFORMANCE (your current model)")
    print("=" * 70)
    print(classification_report(g, p, labels=labels, target_names=names,
                                digits=4, zero_division=0))

    # ---- 2. positional analysis / truncation cost -----------------------
    print("=" * 70)
    print("2. WHERE RARE CLASSES LIVE (gold labels, relative doc position)")
    print("=" * 70)
    pos_bins = defaultdict(lambda: Counter())
    for d in gold:
        n = len(d)
        for i, t in enumerate(d):
            if t in skip:
                continue
            decile = min(9, int(10 * i / max(1, n)))
            pos_bins[idx2tag[t]][decile] += 1
    for cls in ("Decision", "Issue", "Facts", "Reasoning"):
        c = pos_bins.get(cls, Counter())
        total = sum(c.values()) or 1
        bar = " ".join(f"{100*c.get(d,0)/total:4.0f}" for d in range(10))
        print(f"  {cls:<10} decile%: {bar}   (0=doc start ... 9=doc end)")
    print("  -> If Decision mass sits in deciles 8-9, head-truncation of long"
          "\n     docs deletes exactly those sentences.")

    if args.data_dir:
        test_dir = os.path.join(args.data_dir, "test")
        if os.path.isdir(test_dir):
            print("\n  Tail-loss estimate on untruncated test docs (cap=200):")
            lost = Counter(); total_long = 0
            for fname in sorted(os.listdir(test_dir)):
                if not fname.endswith(".txt"):
                    continue
                labs = []
                for line in open(os.path.join(test_dir, fname),
                                 encoding="utf-8", errors="ignore"):
                    if "\t" in line:
                        lab = line.strip().split("\t")[-1]
                        if lab in tag2idx:
                            labs.append(lab)
                if len(labs) > 200:
                    total_long += 1
                    for lab in labs[200:]:
                        lost[lab] += 1
            print(f"  docs >200 sentences: {total_long}")
            for cls, n in lost.most_common():
                print(f"    {cls:<26} {n} gold sentences DELETED by truncate_head")

    # ---- 3. hardest confusions ------------------------------------------
    print("\n" + "=" * 70)
    print("3. TOP CONFUSIONS (gold -> predicted)")
    print("=" * 70)
    conf = Counter()
    for dg, dp in zip(gold, pred):
        for tg, tp_ in zip(dg, dp):
            if tg not in skip and tg != tp_:
                conf[(idx2tag[tg], idx2tag.get(tp_, str(tp_)))] += 1
    for (a, b), n in conf.most_common(10):
        print(f"  {a:<26} -> {b:<26} {n}")


if __name__ == "__main__":
    main()
