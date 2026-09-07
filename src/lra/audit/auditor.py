"""
The inter-role logical auditor.

Pipeline per judgment:
  1. Split sentences into PREMISE side (Facts) and HYPOTHESIS side
     (Reasoning + Decision), skipping low-RRL-confidence sentences.
  2. For each hypothesis, retrieve top-k premise sentences by TF-IDF
     cosine similarity (the aggregation strategy: NLI models need short
     premise-hypothesis pairs, not a 40-paragraph premise).
  3. Run pairwise NLI via the pluggable backend.
  4. Aggregate:
       - CONTRADICTION flag if any retrieved pair scores contradiction
         above `contradiction_threshold`.
       - UNSUPPORTED (fabricated-evidence candidate) flag if the hypothesis
         makes an evidentiary claim (mentions exhibits/witnesses/recoveries)
         but its best premise similarity is below `support_sim_threshold`
         — i.e. the record contains nothing it could rest on.
  5. Separately, a lightweight statute-consistency check flags
     section citations in conclusions that never appear in Facts/Issue
     (placeholder until the RAG module owns statutory verification).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schema import AuditFlag, CONCLUSION_ROLES, ErrorType, Judgment, PREMISE_ROLES, Role
from .nli_backends import NLIBackend

_EVIDENCE_MARKERS = re.compile(
    r"\b(Ext\.?\s?[A-Z]?-?\d+|PW-?\d+|DW-?\d+|recover(y|ed)|seiz(ure|ed)|"
    r"exhibit|footage|forensic|ballistic|confession)\b",
    re.I,
)
_SECTION_RE = re.compile(r"[Ss]ection\s+(\d{2,3}[A-Z]?)")


@dataclass
class AuditConfig:
    # Retrieve more premises so the corroboration vote below has a real sample.
    top_k_premises: int = 8
    # Contradiction gate.
    contradiction_threshold: float = 0.80
    contradiction_margin: float = 0.40    # p(contra) - p(entail) must exceed this
    # NEW: only flag if the contradiction is corroborated — the best premise
    # says contradiction AND fewer than `max_entailing_premises` of the other
    # retrieved premises say entailment (a real perverse finding conflicts with
    # the record broadly, not with one cherry-picked sentence).
    min_contradicting_premises: int = 2
    max_entailing_premises: int = 0
    # NEW: which conclusion roles to audit. Decision-only removes the dominant
    # false-positive source: Reasoning sentences legitimately reject evidence
    # and thus "contradict" isolated facts by design. Set to include Reasoning
    # only for the ablation in your report.
    audit_roles: tuple = (Role.DECISION,)
    # unsupported-evidence + statute checks
    support_sim_threshold: float = 0.06
    min_role_confidence: float = 0.5
    enable_statute_check: bool = True


@dataclass
class AuditReport:
    doc_id: str
    flags: list[AuditFlag] = field(default_factory=list)
    n_hypotheses: int = 0
    n_pairs_scored: int = 0

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "n_hypotheses": self.n_hypotheses,
            "n_pairs_scored": self.n_pairs_scored,
            "flags": [f.to_dict() for f in self.flags],
        }


class Auditor:
    def __init__(self, backend: NLIBackend, config: AuditConfig | None = None):
        self.backend = backend
        self.cfg = config or AuditConfig()

    # ------------------------------------------------------------------
    def audit(self, j: Judgment) -> AuditReport:
        cfg = self.cfg
        report = AuditReport(doc_id=j.doc_id)

        premises = [
            s for s in j.by_role(PREMISE_ROLES) if s.role_confidence >= cfg.min_role_confidence
        ]
        hypotheses = [
            s for s in j.sentences
            if s.role in cfg.audit_roles and s.role_confidence >= cfg.min_role_confidence
        ]
        report.n_hypotheses = len(hypotheses)
        if not premises or not hypotheses:
            return report

        # ---- retrieval ------------------------------------------------
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        P = vec.fit_transform([p.text for p in premises])
        H = vec.transform([h.text for h in hypotheses])
        sims = cosine_similarity(H, P)  # [n_hyp, n_prem]

        # ---- pairwise NLI ----------------------------------------------
        pair_index: list[tuple[int, int]] = []  # (hyp_idx, prem_idx)
        pairs: list[tuple[str, str]] = []
        for hi, h in enumerate(hypotheses):
            order = sims[hi].argsort()[::-1][: cfg.top_k_premises]
            for pi in order:
                if sims[hi, pi] <= 0:
                    continue
                pair_index.append((hi, pi))
                pairs.append((premises[pi].text, h.text))

        results = self.backend.predict_batch(pairs) if pairs else []
        report.n_pairs_scored = len(results)

        # ---- aggregation: contradictions (corroborated) -----------------
        # Collect every retrieved premise's (contra, entail) per hypothesis.
        by_hyp: dict[int, list[tuple[int, float, float]]] = {}
        for (hi, pi), r in zip(pair_index, results):
            by_hyp.setdefault(hi, []).append((pi, r.contradiction, r.entailment))

        for hi, scored in by_hyp.items():
            # count how many retrieved premises strongly say contradiction vs entailment
            contra_hits = [
                (pi, c, e) for pi, c, e in scored
                if c >= cfg.contradiction_threshold and (c - e) >= cfg.contradiction_margin
            ]
            entail_hits = [pi for pi, c, e in scored if e >= 0.80]

            # A perverse finding conflicts with the record *broadly*: require
            # multiple corroborating contradictions AND essentially no premise
            # that entails the finding (which would signal the model is just
            # confused by an isolated pair).
            if len(contra_hits) >= cfg.min_contradicting_premises and \
               len(entail_hits) <= cfg.max_entailing_premises:
                contra_hits.sort(key=lambda t: t[1], reverse=True)
                best_pi, best_c, best_e = contra_hits[0]
                report.flags.append(
                    AuditFlag(
                        flag_type=ErrorType.FACT_CONTRADICTION,
                        hypothesis_sid=hypotheses[hi].sid,
                        premise_sids=[premises[best_pi].sid],
                        score=round(float(best_c), 3),
                        rationale=(
                            "NLI backend rates this finding as contradicting the "
                            f"factual record (p_contradiction={best_c:.2f}). "
                            f'Finding: "{hypotheses[hi].text[:120]}..." vs '
                            f'record: "{premises[best_pi].text[:120]}..."'
                        ),
                    )
                )

        # ---- aggregation: unsupported evidentiary claims -----------------
        for hi, h in enumerate(hypotheses):
            if not _EVIDENCE_MARKERS.search(h.text):
                continue
            max_sim = float(sims[hi].max()) if sims.shape[1] else 0.0
            if max_sim < cfg.support_sim_threshold:
                report.flags.append(
                    AuditFlag(
                        flag_type=ErrorType.FABRICATED_EVIDENCE,
                        hypothesis_sid=h.sid,
                        premise_sids=[],
                        score=round(1.0 - max_sim, 3),
                        rationale=(
                            "Finding relies on evidentiary material with no anchor in the "
                            f"factual record (best premise similarity {max_sim:.2f}). "
                            "Candidate reliance on evidence outside the record."
                        ),
                    )
                )

        # ---- statute-consistency placeholder ------------------------------
        if cfg.enable_statute_check:
            # A section is "orphaned" only if it appears in a conclusion and
            # NOWHERE else in the entire judgment (facts, args, OR other
            # reasoning). Judges routinely introduce the operative section number
            # only at the reasoning stage, so restricting to record-roles alone
            # produced large numbers of false miscite flags.
            elsewhere_secs = set()
            hyp_sids = {h.sid for h in hypotheses}
            for s in j.sentences:
                if s.sid in hyp_sids:
                    continue
                elsewhere_secs.update(_SECTION_RE.findall(s.text))
            # also treat sections cited by >1 conclusion as internally consistent
            all_hyp_secs = Counter(
                sec for h in hypotheses for sec in _SECTION_RE.findall(h.text)
            )
            for h in hypotheses:
                for sec in _SECTION_RE.findall(h.text):
                    if sec not in elsewhere_secs and all_hyp_secs[sec] <= 1:
                        report.flags.append(
                            AuditFlag(
                                flag_type=ErrorType.STATUTE_MISCITE,
                                hypothesis_sid=h.sid,
                                premise_sids=[],
                                score=0.5,
                                rationale=(
                                    f"Section {sec} cited in the conclusion never appears in the "
                                    "charge/facts/arguments — possible miscitation. "
                                    "(To be verified against Bare Acts by the RAG module.)"
                                ),
                            )
                        )
        return report
