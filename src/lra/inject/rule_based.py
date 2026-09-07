"""
Rule-based error injection.

Takes a clean, RRL-segmented judgment and produces a perturbed copy with
ground-truth `InjectionRecord`s. Deterministic given a seed, dependency-free.

Each injector returns None when it cannot apply cleanly to a document, so
the driver can skip it rather than produce a degenerate perturbation.

Design notes
------------
* Corruptions are applied to CONCLUSION sentences (Reasoning/Decision) so the
  premise record stays intact — mirroring the real audit setting where the
  trial record is taken as given and the judge's findings are interrogated.
* FABRICATED_EVIDENCE instead *inserts* a new Decision-side sentence that
  relies on an exhibit/witness never present in the record.
* STATUTE_MISCITE swaps a cited section for a plausible-but-wrong one and is
  the hook for the (future) RAG statutory-verification module.
"""

from __future__ import annotations

import random
import re
from typing import Callable, Optional

from ..schema import (
    CONCLUSION_ROLES,
    ErrorType,
    InjectionRecord,
    Judgment,
    Role,
    Sentence,
)

# --------------------------------------------------------------------------
# Lexical resources
# --------------------------------------------------------------------------

NEGATION_FLIPS: list[tuple[str, str]] = [
    (r"\bwas not\b", "was"),
    (r"\bwere not\b", "were"),
    (r"\bdid not\b", "did"),
    (r"\bhas not\b", "has"),
    (r"\bhad not\b", "had"),
    (r"\bno evidence\b", "clear evidence"),
    (r"\bnot proved\b", "proved"),
    (r"\bnot established\b", "established"),
    (r"\bfailed to prove\b", "succeeded in proving"),
    (r"\bnot recovered\b", "recovered"),
    (r"\bnot present\b", "present"),
    (r"\bnot guilty\b", "guilty"),
]
# and the reverse direction
NEGATION_INSERTS: list[tuple[str, str]] = [
    (r"\bwas (?!not\b)", "was not "),
    (r"\bwere (?!not\b)", "were not "),
    (r"\bhas been\b", "has not been"),
    (r"\bstands proved\b", "stands disproved"),
    (r"\bis established\b", "is not established"),
    (r"\bwas recovered\b", "was not recovered"),
    (r"\bwas present\b", "was not present"),
    (r"\bproved beyond reasonable doubt\b", "not proved beyond reasonable doubt"),
]

FABRICATED_TEMPLATES = [
    "The recovery of the blood-stained dagger (Ext. P-99) from the residence of "
    "the accused conclusively links the accused to the offence.",
    "The testimony of the ballistic expert (PW-19) establishes beyond doubt that "
    "the cartridges matched the weapon of the accused.",
    "The CCTV footage placed on record as Ext. P-88 clearly shows the accused at "
    "the scene of occurrence at the relevant time.",
    "The confessional statement of the accused recorded before the Magistrate "
    "(Ext. P-77) leaves no room for doubt as to his guilt.",
]

# IPC -> plausible wrong section (same neighbourhood, different offence) for
# the miscite injector. Deliberately *wrong*, so a citator/RAG check or an
# NLI check against the described offence should fire.
SECTION_SWAPS = {
    "302": "304",   # murder -> culpable homicide not amounting to murder
    "376": "354",   # rape -> assault on woman
    "420": "406",   # cheating -> criminal breach of trust
    "379": "411",   # theft -> receiving stolen property
    "307": "324",   # attempt to murder -> voluntarily causing hurt
    "498A": "494",
    "34": "149",
}

_NUM_RE = re.compile(r"\b(\d{1,2})\b")
_SECTION_RE = re.compile(r"[Ss]ection\s+(\d{2,3}[A-Z]?)")
_WITNESS_RE = re.compile(r"\bPW-?(\d{1,2})\b")


# --------------------------------------------------------------------------
# Individual injectors: (judgment, rng) -> Optional[(Sentence-ish edits, record)]
# --------------------------------------------------------------------------

def _conclusion_candidates(j: Judgment, roles=None) -> list[Sentence]:
    if roles is None:
        stashed = j.meta.get("_inject_roles") if hasattr(j, "meta") else None
        if stashed:
            roles = {Role(v) for v in stashed}
        else:
            roles = CONCLUSION_ROLES
    return [s for s in j.sentences if s.role in roles]


def inject_fact_contradiction(j: Judgment, rng: random.Random) -> Optional[InjectionRecord]:
    """Flip a polarity marker in a conclusion sentence so it now contradicts
    the factual record it was originally consistent with."""
    cands = _conclusion_candidates(j)
    rng.shuffle(cands)
    for sent in cands:
        for patterns in (NEGATION_FLIPS, NEGATION_INSERTS):
            for pat, repl in patterns:
                if re.search(pat, sent.text):
                    new_text = re.sub(pat, repl, sent.text, count=1)
                    if new_text == sent.text:
                        continue
                    rec = InjectionRecord(
                        error_type=ErrorType.FACT_CONTRADICTION,
                        target_sid=sent.sid,
                        original_text=sent.text,
                        corrupted_text=new_text,
                        conflicting_sids=_related_fact_sids(j, sent),
                        note=f"polarity flip via /{pat}/",
                    )
                    sent.text = new_text
                    return rec
    return None


def inject_entity_swap(j: Judgment, rng: random.Random) -> Optional[InjectionRecord]:
    """Reattribute a finding to a different witness (PW-n -> PW-m)."""
    cands = [s for s in _conclusion_candidates(j) if _WITNESS_RE.search(s.text)]
    if not cands:
        return None
    sent = rng.choice(cands)
    m = _WITNESS_RE.search(sent.text)
    old_n = int(m.group(1))
    new_n = old_n + rng.choice([3, 4, 5, 7])
    new_text = sent.text[: m.start()] + f"PW-{new_n}" + sent.text[m.end():]
    rec = InjectionRecord(
        error_type=ErrorType.ENTITY_SWAP,
        target_sid=sent.sid,
        original_text=sent.text,
        corrupted_text=new_text,
        conflicting_sids=_related_fact_sids(j, sent),
        note=f"PW-{old_n} -> PW-{new_n}",
    )
    sent.text = new_text
    return rec


def inject_quantity_corruption(j: Judgment, rng: random.Random) -> Optional[InjectionRecord]:
    """Corrupt a small integer (counts, dates-of-month) in a conclusion."""
    cands = [s for s in _conclusion_candidates(j) if _NUM_RE.search(s.text)]
    if not cands:
        return None
    sent = rng.choice(cands)
    m = _NUM_RE.search(sent.text)
    old = int(m.group(1))
    new = old + rng.choice([1, 2, 3, 10])
    new_text = sent.text[: m.start()] + str(new) + sent.text[m.end():]
    rec = InjectionRecord(
        error_type=ErrorType.QUANTITY_CORRUPTION,
        target_sid=sent.sid,
        original_text=sent.text,
        corrupted_text=new_text,
        conflicting_sids=_related_fact_sids(j, sent),
        note=f"{old} -> {new}",
    )
    sent.text = new_text
    return rec


def inject_statute_miscite(j: Judgment, rng: random.Random) -> Optional[InjectionRecord]:
    """Swap a cited section for a wrong neighbouring section."""
    cands = [s for s in _conclusion_candidates(j) if _SECTION_RE.search(s.text)]
    if not cands:
        return None
    sent = rng.choice(cands)
    m = _SECTION_RE.search(sent.text)
    old_sec = m.group(1)
    new_sec = SECTION_SWAPS.get(old_sec)
    if new_sec is None:
        # generic perturbation: +2
        digits = re.match(r"(\d+)", old_sec).group(1)
        new_sec = str(int(digits) + 2)
    new_text = sent.text[: m.start()] + f"Section {new_sec}" + sent.text[m.end():]
    rec = InjectionRecord(
        error_type=ErrorType.STATUTE_MISCITE,
        target_sid=sent.sid,
        original_text=sent.text,
        corrupted_text=new_text,
        conflicting_sids=[],
        note=f"Section {old_sec} -> Section {new_sec}",
    )
    sent.text = new_text
    return rec


def inject_fabricated_evidence(j: Judgment, rng: random.Random) -> Optional[InjectionRecord]:
    """Insert a Decision-side sentence relying on evidence absent from the record."""
    text = rng.choice(FABRICATED_TEMPLATES)
    # place it just before the last Decision sentence (or append)
    idx = len(j.sentences)
    for i in range(len(j.sentences) - 1, -1, -1):
        if j.sentences[i].role == Role.DECISION:
            idx = i
            break
    sid = f"s{len(j.sentences):03d}_inj"
    # Insert as a Decision-role sentence so it falls within the default audit
    # scope (Decision-only). If you widen audit_roles to include Reasoning,
    # this still works.
    new_sent = Sentence(sid=sid, text=text, role=Role.DECISION, role_confidence=1.0)
    j.sentences.insert(idx, new_sent)
    return InjectionRecord(
        error_type=ErrorType.FABRICATED_EVIDENCE,
        target_sid=sid,
        original_text=None,
        corrupted_text=text,
        conflicting_sids=[],
        note="inserted reliance on non-existent exhibit/witness",
    )


def _related_fact_sids(j: Judgment, sent: Sentence, k: int = 3) -> list[str]:
    """Cheap lexical-overlap heuristic to record which fact sentences the
    corruption most plausibly conflicts with (ground-truth bookkeeping only —
    the auditor never sees this)."""
    words = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", sent.text)}
    scored = []
    for f in j.sentences:
        if f.role != Role.FACTS:
            continue
        fw = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", f.text)}
        scored.append((len(words & fw), f.sid))
    scored.sort(reverse=True)
    return [sid for score, sid in scored[:k] if score > 0]


INJECTORS: dict[ErrorType, Callable] = {
    ErrorType.FACT_CONTRADICTION: inject_fact_contradiction,
    ErrorType.ENTITY_SWAP: inject_entity_swap,
    ErrorType.QUANTITY_CORRUPTION: inject_quantity_corruption,
    ErrorType.STATUTE_MISCITE: inject_statute_miscite,
    ErrorType.FABRICATED_EVIDENCE: inject_fabricated_evidence,
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def perturb(
    clean: Judgment,
    error_types: list[ErrorType] | None = None,
    n_errors: int = 1,
    seed: int = 13,
    target_roles: tuple | None = None,
) -> Judgment:
    """Return a deep-copied, perturbed judgment carrying its InjectionRecords.

    target_roles : restrict corruptions to these conclusion roles so the
                   benchmark matches the auditor's scope. Pass the SAME value
                   as AuditConfig.audit_roles (default: Decision only) to keep
                   recall meaningful — injecting into roles the auditor never
                   inspects would spuriously depress recall.
    """
    import copy

    rng = random.Random(seed)
    j = copy.deepcopy(clean)
    j.doc_id = f"{clean.doc_id}__perturbed_seed{seed}"
    j.injections = []

    # Stash target roles where the candidate helper can read them.
    j.meta = dict(j.meta)
    j.meta["_inject_roles"] = [r.value for r in target_roles] if target_roles else None

    pool = list(error_types or INJECTORS.keys())
    rng.shuffle(pool)

    applied = 0
    for et in pool:
        if applied >= n_errors:
            break
        rec = INJECTORS[et](j, rng)
        if rec is not None:
            j.injections.append(rec)
            applied += 1
    j.meta.pop("_inject_roles", None)
    return j
