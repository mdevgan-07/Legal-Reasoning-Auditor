#!/usr/bin/env python3
"""
End-to-end: judgment JSON -> audit -> Grounds of Appeal draft (Markdown).

Examples:
  # lawyer-facing formal draft, LLM auditor (best recall), no polish:
  python scripts/draft_grounds.py --doc data/real_auditable/test_doc_0003.json \
      --auditor llm --mode formal --out grounds_0003.md

  # student-facing explanatory draft with the fast NLI auditor:
  python scripts/draft_grounds.py --doc data/sample_judgment.json \
      --auditor nli --mode explanatory --out grounds_sample.md

  # add --polish to have the local LLM smooth the prose (reuses the same
  # loaded model; strict guardrails prevent added content).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lra.schema import Judgment
from lra.generate.grounds import GroundsGenerator, GroundsConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True, help="judgment JSON (RRL-segmented)")
    ap.add_argument("--auditor", default="nli", choices=["nli", "llm"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--mode", default="formal", choices=["formal", "explanatory"])
    ap.add_argument("--polish", action="store_true", help="LLM prose polish (llm auditor only)")
    ap.add_argument("--case-title", default="State v. [Accused]")
    ap.add_argument("--out", default=None, help="output .md path (default: grounds_<docid>.md)")
    args = ap.parse_args()

    j = Judgment.load(args.doc)

    engine = None
    if args.auditor == "llm":
        from lra.audit.local_llm_auditor import LocalLLMAuditor, LocalLLMAuditConfig
        auditor = LocalLLMAuditor(LocalLLMAuditConfig(model=args.model))
        report = auditor.audit(j)
        engine = auditor.engine
    else:
        from lra.audit.auditor import Auditor, AuditConfig
        from lra.audit.nli_backends import make_backend
        auditor = Auditor(make_backend("hf"), AuditConfig())
        report = auditor.audit(j)

    gen = GroundsGenerator(
        GroundsConfig(mode=args.mode, case_title=args.case_title,
                      use_llm_polish=(args.polish and engine is not None)),
        engine=engine,
    )
    draft = gen.generate(j, report.flags)

    out = Path(args.out or f"grounds_{j.doc_id}.md")
    out.write_text(draft)
    print(f"flags detected: {len(report.flags)}")
    print(f"grounds drafted -> {out}")


if __name__ == "__main__":
    main()
