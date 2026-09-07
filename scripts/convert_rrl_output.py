#!/usr/bin/env python3
"""
Convert RRL model output into benchmark-ready Judgment JSONs.

Prereqs (produced in JupyterLab):
  finetuned_bert_predictions.json   (you have this)
  test_sentences.json               (from export_sentences_JUPYTER.py)
  test_confidences.json             (optional)

Usage:
  python scripts/convert_rrl_output.py \
      --pred finetuned_bert_predictions.json \
      --texts test_sentences.json \
      --out data/real \
      [--conf test_confidences.json] [--use test_pred|test_gold]

Writes one Judgment per document into --out/. Point the benchmark at that dir:
  python scripts/run_benchmark.py --data data/real --backend hf --seeds 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.rrl.adapter import from_prediction_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--texts", required=True)
    ap.add_argument("--conf", default=None)
    ap.add_argument("--use", default="test_pred", choices=["test_pred", "test_gold"])
    ap.add_argument("--out", default="data/real")
    args = ap.parse_args()

    judgments = from_prediction_file(
        predictions_path=args.pred,
        sentences_path=args.texts,
        use=args.use,
        confidences_path=args.conf,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for j in judgments:
        j.save(out / f"{j.doc_id}.json")

    # quick corpus stats
    from collections import Counter
    roles = Counter()
    for j in judgments:
        roles.update(s.role.value for s in j.sentences)
    print(f"wrote {len(judgments)} judgments to {out}/ (source={args.use})")
    print("role distribution:", dict(roles.most_common()))
    auditable = sum(
        1 for j in judgments
        if any(s.role.value == "Facts" for s in j.sentences)
        and any(s.role.value in ("Reasoning", "Decision") for s in j.sentences)
    )
    print(f"auditable docs (have both Facts and Reasoning/Decision): {auditable}/{len(judgments)}")


if __name__ == "__main__":
    main()
