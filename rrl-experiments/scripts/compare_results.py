#!/usr/bin/env python3
"""
Print the shared results table sorted by test macro-F1.

  python scripts/compare_results.py --out-dir /workspace/legal_capstone/rrl_experiments
"""
from __future__ import annotations

import argparse
import csv
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/legal_capstone/rrl_experiments")
    args = ap.parse_args()

    path = os.path.join(args.out_dir, "results.csv")
    if not os.path.exists(path):
        sys.exit(f"no results yet at {path}")

    rows = list(csv.DictReader(open(path)))
    rows.sort(key=lambda r: float(r["macro_f1"]), reverse=True)

    cols = ["name", "macro_f1", "val_test_gap", "Issue", "Decision",
            "Facts", "Reasoning", "None"]
    widths = {c: max(len(c), max((len(str(r.get(c, ""))) for r in rows), default=0))
              for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header); print("-" * len(header))
    for r in rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    print(f"\n(paper's Hier_BiLSTM-CRF benchmark: test macro-F1 0.77; "
          f"your previous best: 0.71)")


if __name__ == "__main__":
    main()
