#!/usr/bin/env python3
"""
DEMO SPINE — the full pipeline in one command:

  PDF  ->  ingest (extract/clean/split)  ->  RRL labeling  ->
  logical audit (LLM or NLI)  ->  RAG statutory verification  ->
  Grounds of Appeal (formal + explanatory)

Usage:
  # everything on, LLM auditor, heuristic RRL fallback:
  python scripts/demo_e2e.py --pdf my_judgment.pdf --auditor llm

  # fast demo without any model (heuristic auditor + fallback RRL):
  python scripts/demo_e2e.py --pdf my_judgment.pdf --auditor heuristic

  # from an already-segmented judgment JSON (skips ingest+RRL):
  python scripts/demo_e2e.py --json data/real_auditable/test_doc_0001.json --auditor llm

Outputs into --out (default demo_out/): the segmented judgment, the audit
report with all flags, and both Grounds drafts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment
from lra.rag.verifier import StatuteVerifier, StatuteVerifierConfig
from lra.generate.grounds import GroundsGenerator, GroundsConfig


def stage(msg: str) -> None:
    print(f"\n{'='*8} {msg} {'='*8}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="raw judgment PDF")
    src.add_argument("--json", help="pre-segmented Judgment JSON")
    ap.add_argument("--auditor", default="llm", choices=["llm", "nli", "heuristic"])
    ap.add_argument("--rrl", default="fallback", choices=["fallback", "live"],
                    help="live = your trained InLegalBERT+BiLSTM+CRF; fallback = keyword heuristic")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--case-title", default="State v. [Accused]")
    ap.add_argument("--kb", default="data/statutes/ipc_bns.json")
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # ---- 1. INGEST ---------------------------------------------------------
    if args.pdf:
        stage("1/5 INGEST: PDF -> sentences -> roles")
        from lra.ingest.pdf_ingest import ingest_pdf
        labeler = None
        if args.rrl == "live":
            from lra.rrl.live import load_rrl_labeler
            print("  loading trained RRL model (InLegalBERT+BiLSTM+CRF)...", flush=True)
            labeler = load_rrl_labeler()
        j = ingest_pdf(args.pdf, labeler=labeler)
        print(f"  pages: {j.meta.get('n_pages')} | sentences: {len(j.sentences)} "
              f"| labeler: {j.meta.get('labeler')}")
        if j.meta.get("labeler") == "heuristic_fallback":
            print("  NOTE: using keyword-fallback RRL. Plug your trained model "
                  "via ingest_pdf(labeler=...) for real labels.")
    else:
        stage("1/5 INGEST: loading pre-segmented judgment")
        j = Judgment.load(args.json)
        print(f"  sentences: {len(j.sentences)}")

    from collections import Counter
    roles = Counter(s.role.value for s in j.sentences)
    print(f"  role distribution: {dict(roles.most_common())}")
    j.save(out / "1_segmented_judgment.json")

    # ---- 2. LOGICAL AUDIT --------------------------------------------------
    stage(f"2/5 LOGICAL AUDIT ({args.auditor})")
    engine = None
    if args.auditor == "llm":
        from lra.audit.local_llm_auditor import LocalLLMAuditor, LocalLLMAuditConfig
        auditor = LocalLLMAuditor(LocalLLMAuditConfig(model=args.model))
        report = auditor.audit(j)
        engine = auditor.engine
        flags = list(report.flags)
    else:
        from lra.audit.auditor import Auditor, AuditConfig
        from lra.audit.nli_backends import make_backend
        backend = "hf" if args.auditor == "nli" else "heuristic"
        report = Auditor(make_backend(backend), AuditConfig()).audit(j)
        flags = list(report.flags)
    print(f"  logical flags: {len(flags)}")
    for f in flags:
        print(f"    [{f.flag_type.value} @ {f.hypothesis_sid} conf {f.score:.2f}] "
              f"{f.rationale[:100]}")

    # ---- 3. STATUTORY VERIFICATION (RAG) ------------------------------------
    stage("3/5 STATUTORY VERIFICATION (RAG vs Bare-Act KB)")
    verifier = StatuteVerifier(StatuteVerifierConfig(kb_path=args.kb))
    stat_flags = verifier.verify(j)
    print(f"  statutory flags: {len(stat_flags)}")
    for f in stat_flags:
        print(f"    [{f.flag_type.value} @ {f.hypothesis_sid}] {f.rationale[:110]}")
    flags.extend(stat_flags)

    (out / "2_audit_report.json").write_text(
        json.dumps({"doc_id": j.doc_id, "flags": [f.to_dict() for f in flags]}, indent=2)
    )

    # ---- 4-5. GROUNDS OF APPEAL (both modes) -------------------------------
    for i, mode in enumerate(("formal", "explanatory"), start=4):
        stage(f"{i}/5 GROUNDS OF APPEAL — {mode} mode")
        gen = GroundsGenerator(
            GroundsConfig(mode=mode, case_title=args.case_title), engine=engine
        )
        draft = gen.generate(j, flags)
        path = out / f"grounds_{mode}.md"
        path.write_text(draft, encoding="utf-8")
        n = draft.count("### GROUND")
        print(f"  {n} ground(s) drafted -> {path}")

    stage("DONE")
    print(f"  all artifacts in {out}/")


if __name__ == "__main__":
    main()
