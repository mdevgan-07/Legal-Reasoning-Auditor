"""
Grounds of Appeal generator — the final stage of the LRA pipeline.

Consumes:  a Judgment + an audit report's flags (from either the pairwise NLI
           auditor or the holistic LLM auditor — both emit AuditFlag).
Produces:  a structured "Grounds of Appeal" draft in Markdown.

Hallucination guardrail (proposal objective 4): the generator is fed ONLY the
flagged errors and the exact judgment sentences they reference. It cannot
introduce grounds that were not detected upstream, because nothing else enters
the prompt. A deterministic template path exists as well, which uses no model
at all.

Verbosity toggle:
  mode="formal"       — court-style draft for a practicing lawyer to edit.
  mode="explanatory"  — teaching mode: each ground includes WHY it is a
                        candidate reversible error and what to verify,
                        for law students / clinical education.

Two rendering engines:
  * template (default) — deterministic, dependency-free, always available.
  * llm                — polishes each ground's prose with the local model
                         (pass the LocalLLMAuditor's engine to reuse the
                         already-loaded model; no second model load).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ..schema import AuditFlag, ErrorType, Judgment

# --------------------------------------------------------------------------
# Legal framing per error type
# --------------------------------------------------------------------------

_FRAMING = {
    ErrorType.FACT_CONTRADICTION: {
        "title": "Perverse Finding Contrary to the Record",
        "formal": (
            "the learned trial court returned a finding which is contrary to and "
            "unsupported by the evidence on record, rendering the finding perverse "
            "and unsustainable in law"
        ),
        "explain": (
            "A 'perverse finding' is one that no reasonable court could have reached "
            "on the evidence actually recorded. Appellate courts can set aside such "
            "findings. Verify: read the flagged holding against the cited record "
            "facts — does the conclusion actually follow?"
        ),
    },
    ErrorType.FABRICATED_EVIDENCE: {
        "title": "Reliance on Material Outside the Record",
        "formal": (
            "the learned trial court placed reliance upon material, testimony, or "
            "exhibits which do not form part of the record of the case, in violation "
            "of the fundamental principle that a judgment must rest solely upon "
            "evidence duly brought on record"
        ),
        "explain": (
            "A court may only decide on evidence formally on record. If the holding "
            "invokes an exhibit, witness, or fact that appears nowhere in the "
            "established record, that reliance is itself a ground of appeal. Verify: "
            "search the full judgment and trial record for the material referenced — "
            "the auditor found no anchor for it in the fact section."
        ),
    },
    ErrorType.STATUTE_MISCITE: {
        "title": "Misapplication / Miscitation of Statutory Provision",
        "formal": (
            "the learned trial court invoked a statutory provision which finds no "
            "foundation in the charge, the pleadings, or the record, amounting to a "
            "misapplication of law apparent on the face of the record"
        ),
        "explain": (
            "The operative section in the holding does not appear anywhere else in "
            "the judgment — not in the charge, facts, or arguments. This may be a "
            "typographical slip or a genuine misapplication. Verify: check the "
            "charge-sheet and the Bare Act — was the accused ever charged under "
            "this provision, and does its ingredients match the found facts?"
        ),
    },
    ErrorType.STATUTE_REPEALED: {
        "title": "Conviction Under a Repealed Statutory Provision",
        "formal": (
            "the learned trial court convicted the accused under a provision of "
            "the Indian Penal Code notwithstanding its repeal by the Bharatiya "
            "Nyaya Sanhita, 2023, without adverting to the applicable provision "
            "of the successor code, vitiating the conviction and sentence"
        ),
        "explain": (
            "The IPC stands repealed w.e.f. 01-07-2024 (replaced by the BNS). If "
            "the offence date falls after the repeal, the charge and conviction "
            "must be under the BNS provision. Verify: the date of the offence "
            "(not the judgment) determines which code applies — check whether the "
            "citation is a substantive error or a permissible reference for "
            "offences pre-dating the repeal."
        ),
    },
    ErrorType.STATUTE_MISMATCH: {
        "title": "Conviction Under a Provision Whose Ingredients Are Not Made Out",
        "formal": (
            "the ingredients of the statutory provision invoked by the learned "
            "trial court are not made out on the facts found, and the conviction "
            "thereunder is unsustainable in law"
        ),
        "explain": (
            "Every offence has specific ingredients that the found facts must "
            "satisfy. The auditor measured weak alignment between this section's "
            "ingredients and the judgment's factual findings. Verify: list the "
            "section's ingredients from the Bare Act and check each against the "
            "findings — if any essential ingredient is unfound, the conviction "
            "under this section cannot stand."
        ),
    },
    ErrorType.SUBTLE_REASONING: {
        "title": "Error in the Process of Reasoning",
        "formal": (
            "the process of reasoning adopted by the learned trial court is "
            "vitiated by a material legal infirmity — the conclusion travels "
            "beyond the findings, disregards material on record, or misplaces "
            "the burden of proof — rendering the finding unsustainable"
        ),
        "explain": (
            "The flagged conclusion commits a subtle reasoning error: asserting "
            "more than the findings support, ignoring favourable material, or "
            "shifting the burden of proof to the accused. Verify: trace the "
            "conclusion back to the specific findings — does each step follow?"
        ),
    },
    ErrorType.QUANTITY_CORRUPTION: {
        "title": "Finding Inconsistent with Recorded Particulars",
        "formal": (
            "the particulars recited in the impugned finding are at variance with "
            "the particulars established on record, disclosing non-application of "
            "mind to the evidence"
        ),
        "explain": (
            "Numbers, dates, or counts in the holding don't match the record. Small "
            "discrepancies can matter (dates fix alibis; counts fix charges). "
            "Verify: compare the specific figures in the holding against the "
            "fact section and documentary exhibits."
        ),
    },
    ErrorType.ENTITY_SWAP: {
        "title": "Misattribution of Testimony or Role",
        "formal": (
            "the impugned finding attributes testimony or conduct to a person other "
            "than the witness or party borne out by the record, vitiating the "
            "appreciation of evidence"
        ),
        "explain": (
            "The holding attributes evidence to the wrong witness or party. "
            "Verify: check the deposition index — who actually said what?"
        ),
    },
}

_DEFAULT_FRAMING = {
    "title": "Error Apparent on the Face of the Record",
    "formal": "the impugned finding suffers from an error apparent on the face of the record",
    "explain": "Review the flagged finding against the record and classify the error.",
}


@dataclass
class GroundsConfig:
    mode: str = "formal"            # "formal" | "explanatory"
    case_title: str = "State v. [Accused]"
    appellant: str = "[Appellant]"
    court: str = "[Appellate Court]"
    use_llm_polish: bool = False    # if True, pass engine to polish prose
    max_grounds: int = 12           # cap; take highest-scored flags first


# --------------------------------------------------------------------------
# Core generator
# --------------------------------------------------------------------------

class GroundsGenerator:
    def __init__(self, config: GroundsConfig | None = None, engine=None):
        """engine: optional text-generation engine exposing .generate(list[str])
        -> list[str] (the LocalLLMAuditor's engine works directly). Only used
        when config.use_llm_polish is True."""
        self.cfg = config or GroundsConfig()
        self.engine = engine

    # -- public API ---------------------------------------------------------
    def generate(self, j: Judgment, flags: list[AuditFlag]) -> str:
        cfg = self.cfg
        flags = self._merge_flags(flags)
        flags = sorted(flags, key=lambda f: f.score, reverse=True)[: cfg.max_grounds]
        if not flags:
            return self._render_no_grounds(j)

        grounds = [self._render_ground(j, f, idx + 1) for idx, f in enumerate(flags)]

        if cfg.use_llm_polish and self.engine is not None:
            grounds = self._polish(grounds)

        return self._assemble(j, grounds, flags)

    @staticmethod
    def _merge_flags(flags: list[AuditFlag]) -> list[AuditFlag]:
        """Merge multiple flags of the same type on the same sentence into one
        ground (e.g. two miscited sections in a single holding), combining
        rationales and premise references."""
        merged: dict[tuple, AuditFlag] = {}
        for f in flags:
            key = (f.hypothesis_sid, f.flag_type)
            if key not in merged:
                merged[key] = AuditFlag(
                    flag_type=f.flag_type,
                    hypothesis_sid=f.hypothesis_sid,
                    premise_sids=list(f.premise_sids),
                    score=f.score,
                    rationale=f.rationale,
                )
            else:
                m = merged[key]
                m.score = max(m.score, f.score)
                for sid in f.premise_sids:
                    if sid not in m.premise_sids:
                        m.premise_sids.append(sid)
                if f.rationale and f.rationale not in m.rationale:
                    m.rationale = f"{m.rationale} Further: {f.rationale}"
        return list(merged.values())

    # -- rendering ------------------------------------------------------------
    def _render_ground(self, j: Judgment, f: AuditFlag, n: int) -> str:
        cfg = self.cfg
        fr = _FRAMING.get(f.flag_type, _DEFAULT_FRAMING)
        try:
            finding = j.get(f.hypothesis_sid).text
        except KeyError:
            finding = "[flagged finding text unavailable]"
        record = []
        for sid in f.premise_sids:
            try:
                record.append(j.get(sid).text)
            except KeyError:
                pass

        lines = [f"### GROUND {n}: {fr['title']}", ""]
        lines.append(
            f"**That** {fr['formal']}; in particular, the finding that:"
            if cfg.mode == "formal"
            else f"**Candidate error** ({f.flag_type.value}, auditor confidence {f.score:.2f}):"
        )
        lines.append("")
        lines.append(f"> \u201c{finding}\u201d  \n> *(judgment, segment {f.hypothesis_sid})*")
        lines.append("")

        if record:
            lines.append(
                "is contrary to the record, which establishes:"
                if cfg.mode == "formal"
                else "The established record states:"
            )
            lines.append("")
            for r_sid, r_text in zip(f.premise_sids, record):
                lines.append(f"> \u201c{r_text}\u201d *(segment {r_sid})*")
            lines.append("")

        if f.rationale:
            lines.append(f"*Auditor's note:* {f.rationale}")
            lines.append("")

        if cfg.mode == "explanatory":
            lines.append(f"**Why this can be a ground of appeal:** {fr['explain']}")
            lines.append("")

        return "\n".join(lines)

    def _assemble(self, j: Judgment, grounds: list[str], flags: list[AuditFlag]) -> str:
        cfg = self.cfg
        head = [
            f"# GROUNDS OF APPEAL (DRAFT)",
            "",
            f"**In the {cfg.court}**  ",
            f"**{cfg.case_title}** — Appeal against judgment `{j.doc_id}`  ",
            f"**Drafted:** {date.today().isoformat()} · **Mode:** {cfg.mode} · "
            f"**Grounds:** {len(grounds)} (from {len(flags)} verified auditor flags)",
            "",
            "---",
            "",
            "The Appellant most respectfully submits the following grounds, amongst others, "
            "each of which is without prejudice to the rest:"
            if cfg.mode == "formal"
            else "The auditor detected the following candidate reversible errors. Each must be "
                 "verified against the full record before inclusion in a filed appeal:",
            "",
        ]
        tail = [
            "---",
            "",
            "**PRAYER**: In light of the above grounds, the Appellant prays that the "
            "impugned judgment be set aside and such further relief granted as this "
            "Hon'ble Court deems fit."
            if cfg.mode == "formal"
            else "**Next steps**: verify each ground against the certified copy of the record; "
                 "discard any ground that does not survive verification; develop surviving "
                 "grounds with supporting precedent.",
            "",
            "> *Auto-generated draft. Every ground derives exclusively from auditor-verified "
            "flags on the judgment text; no ground has been added beyond the detected errors. "
            "This draft requires review by a qualified advocate before any use in proceedings.*",
        ]
        return "\n".join(head + grounds + tail)

    def _render_no_grounds(self, j: Judgment) -> str:
        return (
            f"# GROUNDS OF APPEAL (DRAFT)\n\n"
            f"Judgment `{j.doc_id}`: the auditor detected **no flaggable errors** at the "
            f"current thresholds. No grounds have been drafted — the generator does not "
            f"invent grounds beyond verified flags.\n"
        )

    # -- optional LLM polish -----------------------------------------------
    _POLISH_PROMPT = (
        "Rewrite the following draft ground of appeal in polished formal Indian "
        "appellate drafting style. STRICT RULES: do not add any new facts, "
        "exhibits, witnesses, sections, or claims not present in the draft; do not "
        "remove the quoted passages or segment references; keep it under 180 words. "
        "Output only the rewritten ground in Markdown.\n\nDRAFT:\n{draft}"
    )

    def _polish(self, grounds: list[str]) -> list[str]:
        prompts = [self._POLISH_PROMPT.format(draft=g) for g in grounds]
        try:
            outs = self.engine.generate(prompts)
            # guardrail: if the model dropped the quote or segment ref, keep original
            polished = []
            for orig, new in zip(grounds, outs):
                ok = ("\u201c" in new or ">" in new) and ("segment" in new or "s0" in new)
                polished.append(new.strip() if ok and len(new.strip()) > 80 else orig)
            return polished
        except Exception:
            return grounds
