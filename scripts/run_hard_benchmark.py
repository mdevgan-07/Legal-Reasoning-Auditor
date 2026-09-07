#!/usr/bin/env python3
"""
HARD-TIER benchmark: subtle LLM-injected errors (overreach, ignored evidence,
burden-shift) audited by the holistic LLM auditor. One model instance serves
both roles (injector + auditor) — loaded once.

Report easy-tier (rule-based) vs hard-tier (this) detection side by side to
show the benchmark's difficulty gradient.

Usage:
  python scripts/run_hard_benchmark.py --data data/real_auditable --limit 40 --seeds 1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment, Role
from lra.inject.llm_based import perturb_subtle
from lra.audit.local_llm_auditor import LocalLLMAuditor, LocalLLMAuditConfig
from lra.evalx.metrics import score_pair, merge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_auditable")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--errors-per-doc", type=int, default=1)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    docs = [Judgment.load(p) for p in sorted(Path(args.data).glob("*.json"))][: args.limit]
    if not docs:
        sys.exit(f"no docs in {args.data}")

    print("loading model (serves as both injector and auditor)...", flush=True)
    auditor = LocalLLMAuditor(LLMcfg := LocalLLMAuditConfig(model=args.model))
    engine = auditor.engine

    scores, skipped = [], 0
    for di, doc in enumerate(docs):
        clean = auditor.audit(doc)
        for seed in range(args.seeds):
            pj = perturb_subtle(doc, engine, n_errors=args.errors_per_doc,
                                seed=seed, target_roles=(Role.DECISION,))
            if not pj.injections:          # injector found no candidate / parse fail
                skipped += 1
                continue
            scores.append(score_pair(pj, auditor.audit(pj),
                                     clean if seed == 0 else None))
        if (di + 1) % 5 == 0:
            t = merge(scores)
            print(f"[{di+1}/{len(docs)}] hard-tier: P={t.precision:.2f} "
                  f"R={t.recall:.2f} F1={t.f1:.2f} (skipped={skipped})", flush=True)

    total = merge(scores)
    print(f"\n=== HARD TIER (subtle LLM-injected errors) | "
          f"items={len(scores)} skipped={skipped} ===")
    print(total.summary())
    print("\nCompare with easy-tier (rule-based) numbers from "
          "run_llm_benchmark.py to report the difficulty gradient.")


if __name__ == "__main__":
    main()
