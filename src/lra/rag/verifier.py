"""
RAG statutory verification (proposal Objective 3).

Verifies statute citations in a judgment's conclusions against a statutory
knowledge base, producing three classes of flags:

  STATUTE_REPEALED  — the judgment cites the IPC for an offence committed /
                      judgment delivered after 01-07-2024, when the Bharatiya
                      Nyaya Sanhita (BNS) replaced it. The flag carries the
                      correct BNS equivalent.
  STATUTE_MISMATCH  — the cited section's statutory ingredients do not align
                      with the facts established in the judgment (e.g. facts
                      describe theft but the conviction cites the cheating
                      section). Alignment is measured by TF-IDF similarity
                      between the section's ingredient summary and the
                      judgment's Facts/Issue text, with an LLM upgrade path.
  STATUTE_MISCITE   — the cited section does not exist in the knowledge base
                      at all (possible typographical or substantive error).

Retrieval is lexical (TF-IDF) by default so it runs anywhere; pass a
sentence-embedding model name to upgrade retrieval quality.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..schema import AuditFlag, ErrorType, Judgment, Role

_SECTION_RE = re.compile(
    r"[Ss]ection[s]?\s+(\d{2,3}[A-Z]?)(?:\s*(?:read with|r/w|/)\s*[Ss]?e?c?t?i?o?n?s?\s*(\d{2,3}[A-Z]?))?"
)
_CODE_RE = re.compile(r"\b(IPC|Indian Penal Code|BNS|Bharatiya Nyaya Sanhita)\b", re.I)


@dataclass
class StatuteVerifierConfig:
    kb_path: str = "data/statutes/ipc_bns.json"
    # If the judgment date is on/after this and it cites IPC -> repealed flag.
    ipc_repeal_date: str = "2024-07-01"
    # Alignment: similarity between section ingredients and facts below this
    # => mismatch candidate. Kept low: this is a "verify ingredients" nudge.
    mismatch_threshold: float = 0.05
    check_roles: tuple = (Role.DECISION, Role.REASONING)


class StatuteVerifier:
    def __init__(self, config: StatuteVerifierConfig | None = None):
        self.cfg = config or StatuteVerifierConfig()
        kb = json.loads(Path(self.cfg.kb_path).read_text())
        self.ipc: dict[str, dict] = kb["ipc"]
        self.repeal = date.fromisoformat(
            kb.get("meta", {}).get("ipc_repeal_date", self.cfg.ipc_repeal_date)
        )

    # ------------------------------------------------------------------
    def _extract_citations(self, j: Judgment) -> list[tuple[str, str, str]]:
        """Return (section, code_hint, sid) for every section cited in the
        conclusion roles. code_hint is 'IPC', 'BNS', or '' if unstated."""
        out = []
        for s in j.sentences:
            if s.role not in self.cfg.check_roles:
                continue
            codes = _CODE_RE.findall(s.text)
            code = "IPC" if any("ipc" in c.lower() or "penal" in c.lower() for c in codes) else \
                   "BNS" if any("bns" in c.lower() or "sanhita" in c.lower() for c in codes) else ""
            for m in _SECTION_RE.finditer(s.text):
                for g in (m.group(1), m.group(2)):
                    if g:
                        out.append((g, code, s.sid))
        return out

    def _judgment_date(self, j: Judgment) -> date | None:
        # try meta first, then scan text for a dd.mm.yyyy / yyyy pattern
        d = j.meta.get("judgment_date")
        if d:
            try:
                return date.fromisoformat(str(d))
            except ValueError:
                pass
        for s in j.sentences[:10]:
            m = re.search(r"(\d{1,2})[./-](\d{1,2})[./-](20\d{2})", s.text)
            if m:
                try:
                    return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
                except ValueError:
                    continue
        return None

    # ------------------------------------------------------------------
    def verify(self, j: Judgment) -> list[AuditFlag]:
        cfg = self.cfg
        flags: list[AuditFlag] = []
        cites = self._extract_citations(j)
        if not cites:
            return flags

        jd = self._judgment_date(j)
        facts_text = " ".join(
            s.text for s in j.sentences if s.role in (Role.FACTS, Role.ISSUE)
        )

        seen: set[tuple[str, str]] = set()
        for sec, code, sid in cites:
            if (sec, sid) in seen:
                continue
            seen.add((sec, sid))
            entry = self.ipc.get(sec)

            # 1) unknown section --------------------------------------------------
            if entry is None:
                flags.append(AuditFlag(
                    flag_type=ErrorType.STATUTE_MISCITE,
                    hypothesis_sid=sid, premise_sids=[], score=0.6,
                    rationale=(f"Section {sec} not found in the statutory knowledge "
                               f"base (IPC core offences). Possible miscitation — "
                               f"verify against the Bare Act."),
                ))
                continue

            # 2) repealed-code check ---------------------------------------------
            cites_ipc = (code == "IPC") or (code == "" and sec in self.ipc)
            if cites_ipc and jd is not None and jd >= self.repeal:
                flags.append(AuditFlag(
                    flag_type=ErrorType.STATUTE_REPEALED,
                    hypothesis_sid=sid, premise_sids=[], score=0.9,
                    rationale=(f"Judgment dated {jd.isoformat()} cites IPC Section "
                               f"{sec} ('{entry['offence']}'), but the IPC stands "
                               f"repealed w.e.f. {self.repeal.isoformat()}. The "
                               f"corresponding provision is BNS Section "
                               f"{entry['bns']}. Verify which code governs the "
                               f"offence date and correct the citation."),
                ))

            # 3) offence-fact alignment -------------------------------------------
            if facts_text.strip():
                vec = TfidfVectorizer(stop_words="english")
                try:
                    M = vec.fit_transform([entry["ingredients"], facts_text])
                    sim = float(cosine_similarity(M[0], M[1])[0][0])
                except ValueError:
                    sim = 0.0
                if sim < cfg.mismatch_threshold:
                    flags.append(AuditFlag(
                        flag_type=ErrorType.STATUTE_MISMATCH,
                        hypothesis_sid=sid, premise_sids=[], score=round(1 - sim, 3),
                        rationale=(f"Section {sec} ('{entry['offence']}') requires: "
                                   f"{entry['ingredients']}. The judgment's factual "
                                   f"record shows weak alignment with these "
                                   f"ingredients (similarity {sim:.2f}) — verify "
                                   f"that the found facts satisfy the section's "
                                   f"ingredients."),
                    ))
        return flags
