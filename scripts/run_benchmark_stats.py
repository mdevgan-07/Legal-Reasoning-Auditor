#!/usr/bin/env python3
"""
Statistically rigorous benchmark: multi-seed runs, bootstrap 95% CIs, and a
paired permutation test on per-document F1 between two auditors.

Usage (report-grade numbers):
  python scripts/run_benchmark_stats.py --data data/real_auditable \
      --limit 80 --seeds 5 --systems llm,nli

Outputs mean±CI per metric per system and p-value for the F1 difference.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment, Role
from lra.inject.rule_based import perturb
from lra.evalx.metrics import score_pair, merge


def doc_f1(s) -> float:
    p = s.tp_detect / max(1, s.tp_detect + s.fp)
    r = s.tp_detect / max(1, s.n_injections)
    return 2 * p * r / max(1e-9, p + r)


def bootstrap_ci(vals: list[float], n: int = 5000, alpha: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    vals = np.asarray(vals, dtype=float)
    boots = [float(np.mean(rng.choice(vals, size=len(vals), replace=True))) for _ in range(n)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(vals)), float(lo), float(hi)


def paired_permutation_p(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation test on mean difference."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    d = a - b
    obs = abs(float(np.mean(d)))
    count = 0
    for _ in range(n):
        signs = rng.choice([1, -1], size=len(d))
        if abs(float(np.mean(d * signs))) >= obs:
            count += 1
    return (count + 1) / (n + 1)


def build_auditor(name: str, model: str):
    if name == "llm":
        from lra.audit.local_llm_auditor import LocalLLMAuditor, LocalLLMAuditConfig
        a = LocalLLMAuditor(LocalLLMAuditConfig(model=model))
        return a, a.engine
    if name == "nli":
        from lra.audit.auditor import Auditor, AuditConfig
        from lra.audit.nli_backends import make_backend
        return Auditor(make_backend("hf"), AuditConfig()), None
    if name == "heuristic":
        from lra.audit.auditor import Auditor, AuditConfig
        from lra.audit.nli_backends import make_backend
        return Auditor(make_backend("heuristic"), AuditConfig()), None
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/real_auditable")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--errors-per-doc", type=int, default=2)
    ap.add_argument("--systems", default="llm,nli", help="comma list: llm,nli,heuristic")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    args = ap.parse_args()

    docs = [Judgment.load(p) for p in sorted(Path(args.data).glob("*.json"))][: args.limit]
    if not docs:
        sys.exit(f"no docs in {args.data}")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]

    # per-system: list of per-(doc,seed) scores, and per-doc mean F1 for pairing
    per_item: dict[str, list] = {s: [] for s in systems}
    per_doc_f1: dict[str, list[float]] = {s: [] for s in systems}

    auditors = {}
    for s in systems:
        print(f"loading auditor: {s}", flush=True)
        auditors[s], _ = build_auditor(s, args.model)

    for di, doc in enumerate(docs):
        doc_scores = {s: [] for s in systems}
        cleans = {s: auditors[s].audit(doc) for s in systems}
        for seed in range(args.seeds):
            pj = perturb(doc, n_errors=args.errors_per_doc, seed=seed,
                         target_roles=(Role.DECISION,))
            for s in systems:
                sc = score_pair(pj, auditors[s].audit(pj),
                                cleans[s] if seed == 0 else None)
                per_item[s].append(sc)
                doc_scores[s].append(doc_f1(sc))
        for s in systems:
            per_doc_f1[s].append(float(np.mean(doc_scores[s])))
        if (di + 1) % 10 == 0:
            print(f"[{di+1}/{len(docs)}]", flush=True)

    print(f"\n=== STATISTICAL SUMMARY | docs={len(docs)} seeds={args.seeds} "
          f"({len(docs)*args.seeds} items/system) ===")
    for s in systems:
        tot = merge(per_item[s])
        f1s = per_doc_f1[s]
        m, lo, hi = bootstrap_ci(f1s)
        print(f"\n[{s}]")
        print(f"  pooled: P={tot.precision:.3f} R={tot.recall:.3f} "
              f"F1={tot.f1:.3f} cleanFP={tot.clean_fp}")
        print(f"  per-doc F1: mean={m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    if len(systems) >= 2:
        a, b = systems[0], systems[1]
        p = paired_permutation_p(per_doc_f1[a], per_doc_f1[b])
        da = np.mean(per_doc_f1[a]) - np.mean(per_doc_f1[b])
        print(f"\npaired permutation test ({a} vs {b}): "
              f"ΔF1={da:+.3f}, p={p:.4f} "
              f"({'significant at 0.05' if p < 0.05 else 'not significant'})")


if __name__ == "__main__":
    main()
