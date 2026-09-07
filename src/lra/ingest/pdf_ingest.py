"""
PDF ingestion — the front of the pipeline (raw judgment PDF -> Judgment).

Steps:
  1. Extract text per page with PyMuPDF.
  2. Clean: strip page numbers, running headers/footers (lines repeating on
     most pages), and hyphenation across line breaks.
  3. Sentence-split with a legal-aware splitter (protects 'Sec.', 'Rs.',
     'No.', 'vs.', numbered clauses, and initials from false splits).
  4. Build a Judgment with roles assigned by a labeler:
       - plug your trained RRL model via `labeler=` (callable:
         list[str] -> list[tuple[Role, float]]), OR
       - fall back to a transparent keyword heuristic (clearly marked
         low-confidence) so the demo runs end-to-end without the model.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from ..schema import Judgment, Role, Sentence

# --------------------------------------------------------------------------
# 1-2. Extraction + cleaning
# --------------------------------------------------------------------------

def extract_pdf_text(path: str | Path) -> list[str]:
    """Return cleaned text per page."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages = [p.get_text("text") for p in doc]
    doc.close()
    return _strip_repeating_lines(pages)


def _strip_repeating_lines(pages: list[str]) -> list[str]:
    """Remove running headers/footers: lines that repeat on >=60% of pages,
    plus bare page numbers."""
    if not pages:
        return pages
    counts: Counter = Counter()
    per_page_lines = []
    for p in pages:
        lines = [l.strip() for l in p.splitlines()]
        per_page_lines.append(lines)
        for l in set(l for l in lines if l):
            counts[l] += 1
    threshold = max(2, int(0.6 * len(pages)))
    repeating = {l for l, c in counts.items() if c >= threshold and len(l) < 80}

    cleaned = []
    for lines in per_page_lines:
        keep = []
        for l in lines:
            if not l or l in repeating:
                continue
            if re.fullmatch(r"[-–—\s]*\d{1,4}[-–—\s]*", l):   # bare page number
                continue
            keep.append(l)
        cleaned.append("\n".join(keep))
    return cleaned


def _dehyphenate(text: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# --------------------------------------------------------------------------
# 3. Legal-aware sentence splitting
# --------------------------------------------------------------------------

_ABBREV = [
    "No", "Nos", "Sec", "Secs", "Art", "Arts", "vs", "v", "Rs", "Dr", "Mr",
    "Mrs", "Ms", "Sh", "Smt", "Hon", "Ld", "Adv", "P.W", "D.W", "Ext", "Exh",
    "i.e", "e.g", "etc", "Co", "Ltd", "Pvt", "Anr", "Ors", "Cr", "Crl", "Civ",
]
_PROTECT = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.", re.I)


def split_sentences(text: str) -> list[str]:
    text = _dehyphenate(text)
    text = re.sub(r"\s+", " ", text).strip()
    # protect abbreviation periods with a placeholder
    text = _PROTECT.sub(lambda m: m.group(1) + "\u2024", text)
    # protect decimal numbers and section refs like 313 Cr.P.C / 15,402.50
    text = re.sub(r"(\d)\.(\d)", "\\1\u2024\\2", text)
    # split on sentence enders followed by space + capital/quote/digit-paren
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\u201c\"(\d])", text)
    out = []
    for p in parts:
        p = p.replace("\u2024", ".").strip()
        if len(p.split()) >= 3:          # drop fragments
            out.append(p)
    return out


# --------------------------------------------------------------------------
# 4. Role labeling
# --------------------------------------------------------------------------

# Fallback keyword labeler — transparent, low-confidence. Replace with your
# trained RRL model for real use; this exists so the demo runs standalone.
_RULES: list[tuple[Role, re.Pattern]] = [
    (Role.ISSUE, re.compile(r"\b(question for (determination|consideration)|point[s]? for determination|whether the)\b", re.I)),
    (Role.DECISION, re.compile(r"\b(hereby (convicted|acquitted|sentenced|allowed|dismissed|modified)|is (convicted|acquitted) (under|of)|appeal (is|stands) (allowed|dismissed)|sentenced to|prosecution has proved its case)\b", re.I)),
    (Role.AOP, re.compile(r"\b(learned (public prosecutor|counsel for the (State|prosecution|petitioner|appellant)))\b.*\b(argued|submitted|contended)\b", re.I)),
    (Role.AOR, re.compile(r"\b(learned (defence counsel|counsel for the (accused|respondent|defence)))\b.*\b(argued|submitted|contended)\b", re.I)),
    (Role.REASONING, re.compile(r"\b(in (my|our) (considered )?(view|opinion)|it is (true|settled|well settled)|I (find|hold|am of the)|the testimony of .{0,40}(is|stands)|the plea of|stands proved|not fatal)\b", re.I)),
    (Role.FACTS, re.compile(r"\b(deposed|stated that|case of the prosecution|post[- ]mortem|FIR|investigating officer|recovered|witness(ed)?|on the night of|examined as)\b", re.I)),
]


def heuristic_labeler(sentences: list[str]) -> list[tuple[Role, float]]:
    out = []
    for s in sentences:
        role, conf = Role.NONE, 0.55
        for r, pat in _RULES:
            if pat.search(s):
                role, conf = r, 0.55   # deliberately low: heuristic
                break
        out.append((role, conf))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def ingest_pdf(
    path: str | Path,
    doc_id: Optional[str] = None,
    labeler: Optional[Callable[[list[str]], list[tuple[Role, float]]]] = None,
    max_sentences: int = 200,
) -> Judgment:
    """PDF -> Judgment. Pass labeler=<your RRL model wrapper> for real labels;
    defaults to the keyword heuristic (role_confidence 0.55, marked in meta)."""
    pages = extract_pdf_text(path)
    sentences: list[str] = []
    for pg in pages:
        sentences.extend(split_sentences(pg))
    sentences = sentences[:max_sentences]

    used_fallback = labeler is None
    labeler = labeler or heuristic_labeler
    labels = labeler(sentences)

    sents = [
        Sentence(sid=f"s{i:03d}", text=t, role=r, role_confidence=c)
        for i, (t, (r, c)) in enumerate(zip(sentences, labels))
    ]
    return Judgment(
        doc_id=doc_id or Path(path).stem,
        sentences=sents,
        meta={
            "source_pdf": str(path),
            "labeler": "heuristic_fallback" if used_fallback else "rrl_model",
            "n_pages": len(pages),
        },
    )


def ingest_text(
    raw_text: str,
    doc_id: str = "pasted_text",
    labeler: Optional[Callable] = None,
    max_sentences: int = 200,
) -> Judgment:
    """Same pipeline for raw pasted text (demo convenience)."""
    sentences = split_sentences(raw_text)[:max_sentences]
    used_fallback = labeler is None
    labeler = labeler or heuristic_labeler
    labels = labeler(sentences)
    sents = [
        Sentence(sid=f"s{i:03d}", text=t, role=r, role_confidence=c)
        for i, (t, (r, c)) in enumerate(zip(sentences, labels))
    ]
    return Judgment(doc_id=doc_id, sentences=sents,
                    meta={"labeler": "heuristic_fallback" if used_fallback else "rrl_model"})
