"""
LLM-based subtle error injection — the "hard tier" of the benchmark.

Rule-based injections (rule_based.py) create surface errors: polarity flips,
number swaps. Real reversible errors are subtler. This injector uses a local
LLM to rewrite a Decision/Reasoning sentence so that it commits one of:

  overreach        — the conclusion asserts more than the findings support
                     (e.g. facts support presence at scene; conclusion asserts
                     commission of the act beyond what was found).
  ignored_evidence — the conclusion is rewritten to disregard exculpatory
                     material present in the record (e.g. concludes guilt
                     while the record contains an unrebutted alibi the
                     original sentence had addressed).
  burden_shift     — the conclusion is rephrased to place the burden of proof
                     on the accused ("the accused failed to prove his
                     innocence") contrary to the presumption of innocence.

All three are tagged ErrorType.SUBTLE_REASONING with the sub-type in the
injection note, so the benchmark can report easy-tier vs hard-tier detection
separately.

The injector reuses any engine exposing .generate(list[str]) -> list[str]
(the LocalLLMAuditor's engine works directly — pass it in to avoid a second
model load).
"""

from __future__ import annotations

import json
import random
import re

from ..schema import ErrorType, InjectionRecord, Judgment, Role

_SUBTYPES = ("overreach", "ignored_evidence", "burden_shift")

_PROMPT = """You are creating evaluation data for a legal-error-detection system. Below is the factual record of a judgment and ONE conclusion sentence from it.

FACTUAL RECORD (for context):
{facts}

ORIGINAL CONCLUSION SENTENCE:
"{sentence}"

Rewrite the conclusion sentence to introduce this specific subtle legal flaw: {instruction}

STRICT RULES:
- Keep the same general topic, parties, and style; change as little wording as possible.
- The flaw must be SUBTLE: no crude negation flips, no changed numbers, no new exhibits or witnesses.
- The rewritten sentence must still read like something a judge could plausibly write.
- Output STRICT JSON only: {{"rewritten": "<the sentence>", "flaw_summary": "<one line describing the flaw you introduced>"}}"""

_INSTRUCTIONS = {
    "overreach": (
        "make the conclusion assert MORE than the record supports — e.g. if the "
        "record supports presence or opportunity, conclude actual commission or "
        "intent that was never found as fact"
    ),
    "ignored_evidence": (
        "make the conclusion silently disregard a piece of favourable/exculpatory "
        "material that appears in the record, reaching a one-sided conclusion "
        "that a full reading of the record would not support"
    ),
    "burden_shift": (
        "rephrase so that the burden of proof is placed on the accused — e.g. "
        "'the accused has failed to establish his innocence / disprove the "
        "prosecution version' — contrary to the presumption of innocence"
    ),
}


def _parse_json(raw: str) -> dict | None:
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class LLMInjector:
    def __init__(self, engine):
        """engine: .generate(list[str]) -> list[str] (LocalLLMAuditor.engine)."""
        self.engine = engine

    def inject(
        self,
        j: Judgment,
        subtype: str | None = None,
        rng: random.Random | None = None,
        target_roles: tuple = (Role.DECISION,),
    ) -> InjectionRecord | None:
        """Corrupt one conclusion sentence in-place with a subtle flaw.
        Returns the ground-truth record, or None if no candidate / parse fail."""
        rng = rng or random.Random()
        subtype = subtype or rng.choice(_SUBTYPES)

        cands = [s for s in j.sentences if s.role in target_roles and len(s.text.split()) >= 8]
        if not cands:
            return None
        sent = rng.choice(cands)

        facts = "\n".join(
            f"- {s.text}" for s in j.sentences if s.role == Role.FACTS
        )[:8000] or "(no separate fact section)"

        prompt = _PROMPT.format(
            facts=facts, sentence=sent.text, instruction=_INSTRUCTIONS[subtype]
        )
        out = self.engine.generate([prompt])[0]
        parsed = _parse_json(out)
        if not parsed or not parsed.get("rewritten"):
            return None
        new_text = str(parsed["rewritten"]).strip()
        if new_text == sent.text or len(new_text.split()) < 5:
            return None

        rec = InjectionRecord(
            error_type=ErrorType.SUBTLE_REASONING,
            target_sid=sent.sid,
            original_text=sent.text,
            corrupted_text=new_text,
            conflicting_sids=[],
            note=f"subtype={subtype}: {parsed.get('flaw_summary','')}",
        )
        sent.text = new_text
        return rec


def perturb_subtle(
    clean: Judgment,
    engine,
    n_errors: int = 1,
    seed: int = 13,
    target_roles: tuple = (Role.DECISION,),
) -> Judgment:
    """Deep-copied judgment with n subtle LLM-injected errors + ground truth."""
    import copy

    rng = random.Random(seed)
    j = copy.deepcopy(clean)
    j.doc_id = f"{clean.doc_id}__subtle_seed{seed}"
    j.injections = []
    inj = LLMInjector(engine)
    subtypes = list(_SUBTYPES)
    rng.shuffle(subtypes)
    for k in range(n_errors):
        rec = inj.inject(j, subtype=subtypes[k % len(subtypes)], rng=rng,
                         target_roles=target_roles)
        if rec is not None:
            j.injections.append(rec)
    return j
