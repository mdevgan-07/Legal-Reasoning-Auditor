#!/usr/bin/env python3
"""
End-to-end benchmark: clean judgments -> inject errors -> audit -> score.

Usage:
  python scripts/run_benchmark.py --data data/ --backend heuristic --seeds 5
  python scripts/run_benchmark.py --data data/ --backend hf                # DeBERTa NLI
  python scripts/run_benchmark.py --data data/ --backend llm               # Anthropic judge

Each seed produces an independently perturbed copy of every clean judgment
(2 errors per copy by default), so N docs x S seeds = N*S benchmark items.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment
from lra.inject.rule_based import perturb
from lra.audit.auditor import Auditor, AuditConfig
from lra.audit.nli_backends import make_backend
from lra.evalx.metrics import score_pair, merge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="dir of clean *.json judgments")
    ap.add_argument("--backend", default="heuristic", help="heuristic | hf | llm")
    ap.add_argument("--model", default=None, help="override model name for hf/llm backends")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--errors-per-doc", type=int, default=2)
    ap.add_argument("--out", default="runs/latest", help="output dir for reports")
    args = ap.parse_args()

    kw = {"model_name": args.model} if (args.model and args.backend == "hf") else \
         {"model": args.model} if (args.model and args.backend == "llm") else {}
    backend = make_backend(args.backend, **kw)
    auditor = Auditor(backend, AuditConfig())

    clean_docs = [Judgment.load(p) for p in sorted(Path(args.data).glob("*.json"))]
    if not clean_docs:
        sys.exit(f"no *.json judgments found in {args.data}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scores = []
    for doc in clean_docs:
        clean_report = auditor.audit(doc)  # false-positive baseline
        for seed in range(args.seeds):
            pj = perturb(doc, n_errors=args.errors_per_doc, seed=seed,
                         target_roles=auditor.cfg.audit_roles)
            report = auditor.audit(pj)
            s = score_pair(pj, report, clean_report if seed == 0 else None)
            scores.append(s)

            item_dir = out / pj.doc_id
            item_dir.mkdir(parents=True, exist_ok=True)
            pj.save(item_dir / "perturbed.json")
            (item_dir / "audit_report.json").write_text(json.dumps(report.to_dict(), indent=2))

    total = merge(scores)
    print(f"\n=== LRA benchmark | backend={backend.name} | docs={len(clean_docs)} seeds={args.seeds} ===")
    print(total.summary())
    (out / "summary.txt").write_text(total.summary())
    print(f"\nartifacts written to {out}/")


if __name__ == "__main__":
    main()
