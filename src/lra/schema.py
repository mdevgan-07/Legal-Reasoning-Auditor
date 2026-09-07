"""
Core data schema for the Legal Reasoning Auditor (LRA).

This is the contract between the RRL module (upstream) and the
auditing/benchmark modules (this package). The RRL module should emit
JSON that deserialises into `Judgment`.

Label set follows LegalSeg (Nigam et al., NAACL 2025), 7 roles.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


class Role(str, Enum):
    FACTS = "Facts"
    ISSUE = "Issue"
    AOP = "ArgumentsOfPetitioner"
    AOR = "ArgumentsOfRespondent"
    REASONING = "Reasoning"
    DECISION = "Decision"
    NONE = "None"


# Roles treated as the judge's own findings/conclusions (hypothesis side).
CONCLUSION_ROLES = {Role.REASONING, Role.DECISION}
# Roles treated as the established record (premise side).
PREMISE_ROLES = {Role.FACTS}


class ErrorType(str, Enum):
    """Taxonomy of injectable / detectable reversible-error candidates."""

    FACT_CONTRADICTION = "fact_contradiction"      # ruling negates an established fact
    ENTITY_SWAP = "entity_swap"                    # finding attributed to wrong party/witness
    QUANTITY_CORRUPTION = "quantity_corruption"    # numbers/dates in ruling clash with record
    STATUTE_MISCITE = "statute_miscite"            # wrong / repealed section cited
    STATUTE_REPEALED = "statute_repealed"          # cites repealed code (e.g. IPC post-BNS)
    STATUTE_MISMATCH = "statute_mismatch"          # section's offence doesn't match found facts
    SUBTLE_REASONING = "subtle_reasoning"          # LLM-injected: overreach/ignored evidence/burden
    FABRICATED_EVIDENCE = "fabricated_evidence"    # ruling relies on evidence absent from record
    NONE = "none"


@dataclass
class Sentence:
    sid: str                 # stable id, e.g. "s017"
    text: str
    role: Role
    # Confidence from the RRL classifier (1.0 for gold labels). The auditor
    # can refuse to audit low-confidence segments to avoid cascading errors.
    role_confidence: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "Sentence":
        return Sentence(
            sid=d["sid"],
            text=d["text"],
            role=Role(d["role"]),
            role_confidence=float(d.get("role_confidence", 1.0)),
        )


@dataclass
class InjectionRecord:
    """Ground truth for one injected error (used only by the benchmark)."""

    error_type: ErrorType
    target_sid: str                  # sentence that was corrupted / inserted
    original_text: Optional[str]     # None if the sentence was inserted
    corrupted_text: str
    # sids of premise sentences the corruption conflicts with (may be empty
    # for FABRICATED_EVIDENCE, where the point is that no premise exists).
    conflicting_sids: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["error_type"] = self.error_type.value
        return d

    @staticmethod
    def from_dict(d: dict) -> "InjectionRecord":
        return InjectionRecord(
            error_type=ErrorType(d["error_type"]),
            target_sid=d["target_sid"],
            original_text=d.get("original_text"),
            corrupted_text=d["corrupted_text"],
            conflicting_sids=list(d.get("conflicting_sids", [])),
            note=d.get("note", ""),
        )


@dataclass
class Judgment:
    doc_id: str
    sentences: list[Sentence]
    meta: dict = field(default_factory=dict)
    # Present only on benchmark (perturbed) copies.
    injections: list[InjectionRecord] = field(default_factory=list)

    # ---- convenience -----------------------------------------------------
    def by_role(self, roles: set[Role]) -> list[Sentence]:
        return [s for s in self.sentences if s.role in roles]

    def get(self, sid: str) -> Sentence:
        for s in self.sentences:
            if s.sid == sid:
                return s
        raise KeyError(sid)

    # ---- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "meta": self.meta,
            "sentences": [s.to_dict() for s in self.sentences],
            "injections": [i.to_dict() for i in self.injections],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "Judgment":
        d = json.loads(Path(path).read_text())
        return Judgment(
            doc_id=d["doc_id"],
            meta=d.get("meta", {}),
            sentences=[Sentence.from_dict(s) for s in d["sentences"]],
            injections=[InjectionRecord.from_dict(i) for i in d.get("injections", [])],
        )


@dataclass
class AuditFlag:
    """One candidate reversible error surfaced by the auditor."""

    flag_type: ErrorType
    hypothesis_sid: str              # the Reasoning/Decision sentence flagged
    premise_sids: list[str]          # supporting evidence for the flag
    score: float                     # backend-dependent confidence in [0, 1]
    rationale: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flag_type"] = self.flag_type.value
        return d
