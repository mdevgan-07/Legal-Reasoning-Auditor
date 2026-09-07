"""
Holistic LLM auditor.

Unlike the pairwise NLI auditor (retrieve top-k fact sentences, run MNLI on
each pair), this shows the model the ENTIRE established record at once and asks
the legal-reasoning question directly: does the record support this holding,
contradict it, or stay silent? This is what pairwise MNLI structurally cannot
do — it can't aggregate a 40-sentence fact section, and it reads quantity/
polarity mismatches on otherwise-similar sentences as entailment.

Returns the same AuditFlag objects as the pairwise auditor, so the benchmark
scorer and the downstream Grounds-of-Appeal generator consume both identically.

Requires ANTHROPIC_API_KEY. Batches one API call per audited Decision (plus one
per statute check if enabled), so cost scales with #Decisions, not #pairs.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

from ..schema import AuditFlag, ErrorType, Judgment, Role

_SECTION_RE = re.compile(r"[Ss]ection\s+(\d{2,3}[A-Z]?)")

_SYSTEM = """You are a meticulous appellate law clerk auditing a trial court judgment for reversible errors. You compare the court's final findings against its own established factual record. You are precise and conservative: you only flag a finding when the record genuinely fails to support it, not merely when the wording differs. A judge legitimately rejecting evidence (e.g. disbelieving a witness, rejecting an alibi) is NOT an error."""

_USER = """Below is the ESTABLISHED FACTUAL RECORD of a trial judgment, followed by ONE FINDING/HOLDING made by the court.

=== ESTABLISHED FACTUAL RECORD ===
{facts}

=== FINDING / HOLDING TO AUDIT ===
{decision}

Audit this finding against the record. Classify it as exactly one of:
- "supported": the record contains facts that reasonably support this finding.
- "contradicted": the record contains facts that directly conflict with this finding (a possible perverse finding / reversible error).
- "unsupported": the finding relies on evidence, witnesses, or facts that do NOT appear anywhere in the record (a possible reliance on material outside the record).
- "neutral": the finding is a legal statement, ruling, or sentence that doesn't make an auditable factual claim.

Respond in strict JSON, no prose outside it:
{{"label": "<one of the four>", "confidence": <0.0-1.0>, "conflicting_facts": "<brief quote or paraphrase of the specific record facts that conflict, or empty>", "reason": "<one sentence>"}}"""


@dataclass
class LLMAuditConfig:
    model: str = "claude-sonnet-4-6"
    min_role_confidence: float = 0.5
    audit_roles: tuple = (Role.DECISION,)
    flag_labels: tuple = ("contradicted", "unsupported")  # which map to flags
    min_confidence: float = 0.6
    max_facts_chars: int = 12000   # truncate very long records to control tokens
    enable_statute_check: bool = True


@dataclass
class LLMAuditReport:
    doc_id: str
    flags: list[AuditFlag] = field(default_factory=list)
    n_hypotheses: int = 0
    n_calls: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_hypotheses": self.n_hypotheses,
            "n_calls": self.n_calls,
            "flags": [f.to_dict() for f in self.flags],
        }


_LABEL_TO_ERRTYPE = {
    "contradicted": ErrorType.FACT_CONTRADICTION,
    "unsupported": ErrorType.FABRICATED_EVIDENCE,
}


class LLMAuditor:
    def __init__(self, config: LLMAuditConfig | None = None):
        import anthropic

        self.cfg = config or LLMAuditConfig()
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def _judge(self, facts: str, decision: str) -> dict:
        msg = self.client.messages.create(
            model=self.cfg.model,
            max_tokens=250,
            system=_SYSTEM,
            messages=[{"role": "user", "content": _USER.format(facts=facts, decision=decision)}],
        )
        raw = msg.content[0].text.strip()
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        try:
            return json.loads(raw)
        except Exception:
            # be conservative on parse failure: treat as neutral
            return {"label": "neutral", "confidence": 0.0, "conflicting_facts": "", "reason": "parse_error"}

    def audit(self, j: Judgment) -> LLMAuditReport:
        cfg = self.cfg
        rep = LLMAuditReport(doc_id=j.doc_id)

        facts_sents = [s for s in j.sentences if s.role == Role.FACTS
                       and s.role_confidence >= cfg.min_role_confidence]
        decisions = [s for s in j.sentences if s.role in cfg.audit_roles
                     and s.role_confidence >= cfg.min_role_confidence]
        rep.n_hypotheses = len(decisions)
        if not facts_sents or not decisions:
            return rep

        facts_block = "\n".join(f"- {s.text}" for s in facts_sents)[: cfg.max_facts_chars]

        for d in decisions:
            verdict = self._judge(facts_block, d.text)
            rep.n_calls += 1
            label = str(verdict.get("label", "neutral")).lower()
            conf = float(verdict.get("confidence", 0.0) or 0.0)
            if label in cfg.flag_labels and conf >= cfg.min_confidence:
                rep.flags.append(
                    AuditFlag(
                        flag_type=_LABEL_TO_ERRTYPE.get(label, ErrorType.FACT_CONTRADICTION),
                        hypothesis_sid=d.sid,
                        premise_sids=[],
                        score=round(conf, 3),
                        rationale=(
                            f"[LLM audit: {label}] {verdict.get('reason','')} "
                            f"Conflicting record: {verdict.get('conflicting_facts','')}".strip()
                        ),
                    )
                )

        # statute check reuses the cheap internal-consistency rule (no API call)
        if cfg.enable_statute_check:
            hyp_sids = {d.sid for d in decisions}
            elsewhere = set()
            for s in j.sentences:
                if s.sid not in hyp_sids:
                    elsewhere.update(_SECTION_RE.findall(s.text))
            from collections import Counter
            hyp_secs = Counter(sec for d in decisions for sec in _SECTION_RE.findall(d.text))
            for d in decisions:
                for sec in _SECTION_RE.findall(d.text):
                    if sec not in elsewhere and hyp_secs[sec] <= 1:
                        rep.flags.append(
                            AuditFlag(
                                flag_type=ErrorType.STATUTE_MISCITE,
                                hypothesis_sid=d.sid,
                                premise_sids=[],
                                score=0.5,
                                rationale=f"Section {sec} appears only in the holding, nowhere else in the judgment — possible miscitation (verify against Bare Acts).",
                            )
                        )
        return rep
