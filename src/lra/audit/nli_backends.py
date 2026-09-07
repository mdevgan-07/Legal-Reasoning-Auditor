"""
Pluggable NLI backends.

Every backend maps (premise, hypothesis) -> NLIResult with probabilities for
entailment / neutral / contradiction. Swap backends without touching the
auditor logic.

Backends
--------
1. HFCrossEncoderNLI  — DeBERTa-v3 NLI cross-encoder via `transformers`.
                        Recommended for real runs (needs internet/HF weights).
2. LLMJudgeNLI        — routes the pair to an LLM (Anthropic API) with a
                        legal-entailment prompt; useful where surface-level
                        NLI is too shallow. Needs ANTHROPIC_API_KEY.
3. HeuristicNLI       — dependency-free lexical heuristic. NOT for research
                        claims; exists so the pipeline is testable end-to-end
                        anywhere (CI, this container, a laptop on a train).
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class NLIResult:
    entailment: float
    neutral: float
    contradiction: float

    @property
    def label(self) -> str:
        triples = [
            ("entailment", self.entailment),
            ("neutral", self.neutral),
            ("contradiction", self.contradiction),
        ]
        return max(triples, key=lambda t: t[1])[0]


class NLIBackend(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> NLIResult: ...

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        return [self.predict(p, h) for p, h in pairs]


# --------------------------------------------------------------------------
# 1. HuggingFace cross-encoder (real backend)
# --------------------------------------------------------------------------

class HFCrossEncoderNLI(NLIBackend):
    """
    Wraps a cross-encoder NLI model. Good defaults:
      - "cross-encoder/nli-deberta-v3-base"        (fast)
      - "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" (stronger)
    """

    name = "hf-cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-base", device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        # map label ids -> canonical names, robust to model-specific ordering
        self.id2label = {i: l.lower() for i, l in self.model.config.id2label.items()}

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        torch = self._torch
        results: list[NLIResult] = []
        B = 16
        for i in range(0, len(pairs), B):
            chunk = pairs[i : i + B]
            enc = self.tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                max_length=512,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for row in probs:
                d = {self.id2label[k]: v for k, v in enumerate(row)}
                results.append(
                    NLIResult(
                        entailment=d.get("entailment", 0.0),
                        neutral=d.get("neutral", 0.0),
                        contradiction=d.get("contradiction", 0.0),
                    )
                )
        return results

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        return self.predict_batch([(premise, hypothesis)])[0]


# --------------------------------------------------------------------------
# 2. LLM judge (Anthropic API)
# --------------------------------------------------------------------------

_LLM_PROMPT = """You are auditing a trial court judgment for internal logical consistency.

PREMISE (from the established factual record of the judgment):
\"\"\"{premise}\"\"\"

HYPOTHESIS (a finding/conclusion by the trial judge):
\"\"\"{hypothesis}\"\"\"

Question: given ONLY the premise, is the hypothesis entailed, contradicted, or neither (neutral)? Judge legal-factual consistency (parties, witnesses, quantities, polarity of findings), not writing style.

Respond with EXACTLY one word: entailment, contradiction, or neutral."""


class LLMJudgeNLI(NLIBackend):
    name = "llm-judge"

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # pip install anthropic

        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=5,
            messages=[{"role": "user", "content": _LLM_PROMPT.format(premise=premise, hypothesis=hypothesis)}],
        )
        word = msg.content[0].text.strip().lower()
        soft = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
        soft[word if word in soft else "neutral"] = 1.0
        return NLIResult(**soft)


# --------------------------------------------------------------------------
# 3. Heuristic mock (testing / plumbing only)
# --------------------------------------------------------------------------

_NEG_MARKERS = re.compile(
    r"\b(not|no|never|dis(proved|established)|failed to|without)\b", re.I
)
_NUM = re.compile(r"\b\d+[A-Z]?\b")
_PW = re.compile(r"\bPW-?\d+\b", re.I)


class HeuristicNLI(NLIBackend):
    """Lexical-overlap + polarity/number/witness mismatch heuristic."""

    name = "heuristic"

    @staticmethod
    def _content_words(t: str) -> set[str]:
        return {w.lower() for w in re.findall(r"[A-Za-z]{4,}", t)}

    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        pw, hw = self._content_words(premise), self._content_words(hypothesis)
        overlap = len(pw & hw) / max(1, min(len(pw), len(hw)))

        if overlap < 0.25:
            return NLIResult(0.05, 0.9, 0.05)

        # polarity mismatch on shared content
        p_neg = bool(_NEG_MARKERS.search(premise))
        h_neg = bool(_NEG_MARKERS.search(hypothesis))
        polarity_clash = p_neg != h_neg

        # witness / number mismatch on otherwise-similar sentences
        p_nums, h_nums = set(_NUM.findall(premise)), set(_NUM.findall(hypothesis))
        num_clash = bool(p_nums and h_nums and not (p_nums & h_nums) and overlap > 0.45)
        p_pws, h_pws = set(map(str.upper, _PW.findall(premise))), set(map(str.upper, _PW.findall(hypothesis)))
        pw_clash = bool(p_pws and h_pws and not (p_pws & h_pws) and overlap > 0.45)

        if polarity_clash or num_clash or pw_clash:
            c = 0.6 + 0.3 * overlap
            return NLIResult(entailment=0.05, neutral=1 - c - 0.05, contradiction=c)
        if overlap > 0.6:
            return NLIResult(entailment=0.7, neutral=0.27, contradiction=0.03)
        return NLIResult(entailment=0.2, neutral=0.75, contradiction=0.05)


def make_backend(name: str, **kw) -> NLIBackend:
    name = name.lower()
    if name in ("hf", "hf-cross-encoder", "deberta"):
        return HFCrossEncoderNLI(**kw)
    if name in ("llm", "llm-judge", "claude"):
        return LLMJudgeNLI(**kw)
    if name in ("heuristic", "mock"):
        return HeuristicNLI()
    raise ValueError(f"unknown NLI backend: {name}")
