#!/usr/bin/env python3
"""
Benchmark the holistic LLM auditor and (optionally) compare to the pairwise
NLI auditor on the SAME injected documents.

Cost note: this makes ~1 API call per audited Decision per document per seed.
Start small with --limit. Example first run (cheap):
  python scripts/run_llm_benchmark.py --data data/real --limit 30 --seeds 1

Full comparison once you're happy:
  python scripts/run_llm_benchmark.py --data data/real --limit 150 --seeds 2 --compare-nli
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment, Role
from lra.inject.rule_based import perturb
from lra.audit.llm_auditor import LLMAuditor, LLMAuditConfig
from lra.audit.local_llm_auditor import LocalLLMAuditor, LocalLLMAuditConfig
from lra.audit.auditor import Auditor, AuditConfig
from lra.audit.nli_backends import make_backend
from lra.evalx.metrics import score_pair, merge


def _make_llm_auditor(args):
    if args.judge == "local":
        return LocalLLMAuditor(LocalLLMAuditConfig(model=args.model, load_4bit=args.load_4bit))
    return LLMAuditor(LLMAuditConfig(model=args.model))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real")
    ap.add_argument("--limit", type=int, default=30, help="max #documents (cost guard)")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--errors-per-doc", type=int, default=2)
    ap.add_argument("--judge", default="local", choices=["local", "api"],
                    help="local = open-source model on your GPU (no key); api = Anthropic")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct",
                    help="HF model id for --judge local, or Anthropic model for --judge api")
    ap.add_argument("--load-4bit", action="store_true", help="4-bit quantization (for 32B on 40GB)")
    ap.add_argument("--compare-nli", action="store_true",
                    help="also run the pairwise DeBERTa auditor on the same docs")
    ap.add_argument("--out", default="runs/llm")
    args = ap.parse_args()

    docs = [Judgment.load(p) for p in sorted(Path(args.data).glob("*.json"))][: args.limit]
    if not docs:
        sys.exit(f"no docs in {args.data}")

    llm = _make_llm_auditor(args)
    nli = Auditor(make_backend("hf"), AuditConfig()) if args.compare_nli else None

    llm_scores, nli_scores = [], []
    Path(args.out).mkdir(parents=True, exist_ok=True)

    for doc in docs:
        # clean-doc false-positive baselines
        llm_clean = llm.audit(doc)
        nli_clean = nli.audit(doc) if nli else None
        for seed in range(args.seeds):
            pj = perturb(doc, n_errors=args.errors_per_doc, seed=seed,
                         target_roles=(Role.DECISION,))
            lrep = llm.audit(pj)
            # adapt LLMAuditReport -> shape score_pair expects (has .flags)
            llm_scores.append(score_pair(pj, lrep, llm_clean if seed == 0 else None))
            if nli:
                nrep = nli.audit(pj)
                nli_scores.append(score_pair(pj, nrep, nli_clean if seed == 0 else None))

    llm_total = merge(llm_scores)
    tag = f"LOCAL {args.model}" if args.judge == "local" else f"API {args.model}"
    print(f"\n=== HOLISTIC LLM AUDITOR ({tag}) | docs={len(docs)} seeds={args.seeds} ===")
    print(llm_total.summary())
    (Path(args.out) / "llm_summary.txt").write_text(llm_total.summary())

    if nli:
        nli_total = merge(nli_scores)
        print("\n=== PAIRWISE DeBERTa NLI (same docs) ===")
        print(nli_total.summary())
        print("\n--- head to head ---")
        print(f"precision   LLM {llm_total.precision:.2f}  vs  NLI {nli_total.precision:.2f}")
        print(f"recall      LLM {llm_total.recall:.2f}  vs  NLI {nli_total.recall:.2f}")
        print(f"F1          LLM {llm_total.f1:.2f}  vs  NLI {nli_total.f1:.2f}")
        print(f"clean FP    LLM {llm_total.clean_fp}  vs  NLI {nli_total.clean_fp}")


if __name__ == "__main__":
    main()
