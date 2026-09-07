"""
Local (open-source) LLM auditor — no API key, runs on your own GPU.

Mirrors LLMAuditor exactly (same holistic prompt, same JSON contract, same
AuditFlag output), so the benchmark runner and scorer treat it identically.
The ONLY difference from the API version is which model produces the judgment.

Two execution paths, auto-selected:
  * vLLM   — if `vllm` imports, uses it for fast batched generation.
  * transformers — otherwise falls back to HF generate() (guaranteed to work
                   since you already run transformers for DeBERTa). Slower, but
                   fine for a few hundred documents.

Default model: Qwen/Qwen2.5-14B-Instruct (~28GB bf16, fits 40GB with headroom).
Swap via LocalLLMAuditConfig.model:
  - "Qwen/Qwen2.5-7B-Instruct"   faster, ~15GB
  - "Qwen/Qwen2.5-32B-Instruct"  stronger, needs 4-bit (set load_4bit=True)
  - "meta-llama/Llama-3.1-8B-Instruct"  alternative
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from ..schema import AuditFlag, ErrorType, Judgment, Role
# reuse the SAME prompt strings as the API auditor for comparability
from .llm_auditor import _SYSTEM, _USER, _LABEL_TO_ERRTYPE, _SECTION_RE, LLMAuditReport


@dataclass
class LocalLLMAuditConfig:
    model: str = "Qwen/Qwen2.5-14B-Instruct"
    load_4bit: bool = False          # set True for 32B on 40GB
    max_new_tokens: int = 200
    min_role_confidence: float = 0.5
    audit_roles: tuple = (Role.DECISION,)
    flag_labels: tuple = ("contradicted", "unsupported")
    min_confidence: float = 0.6
    max_facts_chars: int = 12000
    enable_statute_check: bool = True
    batch_size: int = 8              # vLLM batches; transformers loops


# --------------------------------------------------------------------------
# Backend wrappers
# --------------------------------------------------------------------------

class _VLLMEngine:
    def __init__(self, model: str, load_4bit: bool):
        from vllm import LLM, SamplingParams
        kw = dict(model=model, dtype="bfloat16", gpu_memory_utilization=0.85,
                  max_model_len=8192, trust_remote_code=True)
        if load_4bit:
            kw["quantization"] = "bitsandbytes"
            kw["load_format"] = "bitsandbytes"
        self.llm = LLM(**kw)
        self.SamplingParams = SamplingParams
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    def generate(self, prompts: list[str]) -> list[str]:
        chats = [
            self.tok.apply_chat_template(
                [{"role": "system", "content": _SYSTEM},
                 {"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True,
            )
            for p in prompts
        ]
        sp = self.SamplingParams(temperature=0.0, max_tokens=200)
        outs = self.llm.generate(chats, sp)
        return [o.outputs[0].text for o in outs]


class _HFEngine:
    def __init__(self, model: str, load_4bit: bool, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        kw = dict(torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
            )
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()

    def generate(self, prompts: list[str]) -> list[str]:
        torch = self.torch
        results = []
        for p in prompts:   # simple loop; docs-per-run is modest
            msgs = [{"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": p}]
            text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            enc = self.tok(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc, max_new_tokens=self.max_new_tokens,
                    do_sample=False, temperature=None, top_p=None,
                    pad_token_id=self.tok.eos_token_id,
                )
            gen = out[0][enc["input_ids"].shape[1]:]
            results.append(self.tok.decode(gen, skip_special_tokens=True))
        return results


def _make_engine(cfg: LocalLLMAuditConfig):
    try:
        import vllm  # noqa: F401
        print(f"[local-llm] using vLLM backend for {cfg.model}")
        return _VLLMEngine(cfg.model, cfg.load_4bit)
    except Exception as e:
        print(f"[local-llm] vLLM unavailable ({type(e).__name__}); using transformers backend")
        return _HFEngine(cfg.model, cfg.load_4bit, cfg.max_new_tokens)


# --------------------------------------------------------------------------
# Auditor
# --------------------------------------------------------------------------

_DISPOSITION_RE = re.compile(
    r"\b(appeal|petition|application|revision|suit)\s+(is\s+)?"
    r"(hereby\s+)?(allowed|dismissed|disposed|succeeds?|rejected|granted)\b",
    re.I,
)


def _is_bare_disposition(text: str) -> bool:
    """True for procedural dispositions ('the appeal is allowed') that carry no
    auditable factual claim, so an 'unsupported' verdict on them is a false
    positive. A disposition that also cites concrete particulars (amounts,
    sections, witnesses, exhibits) is NOT bare and remains auditable."""
    t = text.strip()
    if not _DISPOSITION_RE.search(t):
        return False
    has_particulars = re.search(r"Rs\.?\s*\d|Section\s+\d|PW-?\d|Ext|\bacre\b", t)
    if has_particulars:
        return False
    return len(t.split()) <= 40


def _parse(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    # grab the first {...} block if the model added stray text
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if m:
        raw = m.group(0)
    try:
        return json.loads(raw)
    except Exception:
        return {"label": "neutral", "confidence": 0.0, "conflicting_facts": "", "reason": "parse_error"}


class LocalLLMAuditor:
    def __init__(self, config: LocalLLMAuditConfig | None = None):
        self.cfg = config or LocalLLMAuditConfig()
        self.engine = _make_engine(self.cfg)

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
        prompts = [_USER.format(facts=facts_block, decision=d.text) for d in decisions]
        outputs = self.engine.generate(prompts)
        rep.n_calls += len(prompts)

        for d, raw in zip(decisions, outputs):
            verdict = _parse(raw)
            label = str(verdict.get("label", "neutral")).lower()
            conf = float(verdict.get("confidence", 0.0) or 0.0)
            # A bare procedural disposition ("the appeal is allowed", "appeal
            # dismissed", "petition succeeds") is a conclusion of law, not a
            # factual finding — it legitimately has no anchor in the record, so
            # an "unsupported" verdict on it is a false positive. Suppress it.
            if label == "unsupported" and _is_bare_disposition(d.text):
                continue
            if label in cfg.flag_labels and conf >= cfg.min_confidence:
                rep.flags.append(
                    AuditFlag(
                        flag_type=_LABEL_TO_ERRTYPE.get(label, ErrorType.FACT_CONTRADICTION),
                        hypothesis_sid=d.sid,
                        premise_sids=[],
                        score=round(conf, 3),
                        rationale=(
                            f"[local-LLM: {label}] {verdict.get('reason','')} "
                            f"Conflicting record: {verdict.get('conflicting_facts','')}".strip()
                        ),
                    )
                )

        if cfg.enable_statute_check:
            hyp_sids = {d.sid for d in decisions}
            elsewhere = set()
            for s in j.sentences:
                if s.sid not in hyp_sids:
                    elsewhere.update(_SECTION_RE.findall(s.text))
            hyp_secs = Counter(sec for d in decisions for sec in _SECTION_RE.findall(d.text))
            for d in decisions:
                for sec in _SECTION_RE.findall(d.text):
                    if sec not in elsewhere and hyp_secs[sec] <= 1:
                        rep.flags.append(
                            AuditFlag(
                                flag_type=ErrorType.STATUTE_MISCITE,
                                hypothesis_sid=d.sid, premise_sids=[], score=0.5,
                                rationale=f"Section {sec} appears only in the holding, nowhere else — possible miscitation (verify against Bare Acts).",
                            )
                        )
        return rep
