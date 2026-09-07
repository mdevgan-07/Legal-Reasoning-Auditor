"""
Benchmark scoring: match auditor flags against injected ground truth.

Matching rule
-------------
A flag is a TRUE POSITIVE for an injection if it targets the same sentence
(`hypothesis_sid == injection.target_sid`). Two granularities are reported:

  * detection    — did *any* flag land on the corrupted sentence?
  * typed        — did a flag of the *correct error type* land on it?

Flags on clean sentences count as false positives; unmatched injections are
false negatives. We also report FP-rate on the *clean* copy of each document
(auditors that scream at everything are useless to a lawyer).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schema import Judgment
from ..audit.auditor import AuditReport


@dataclass
class BenchScore:
    tp_detect: int = 0
    tp_typed: int = 0
    fn: int = 0
    fp: int = 0
    clean_fp: int = 0
    n_injections: int = 0
    n_flags: int = 0
    per_type: dict = field(default_factory=dict)  # error_type -> [hit, total]

    # ------------------------------------------------------------------
    @property
    def precision(self) -> float:
        return self.tp_detect / max(1, self.tp_detect + self.fp)

    @property
    def recall(self) -> float:
        return self.tp_detect / max(1, self.n_injections)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / max(1e-9, p + r)

    @property
    def typed_recall(self) -> float:
        return self.tp_typed / max(1, self.n_injections)

    def summary(self) -> str:
        lines = [
            f"injections={self.n_injections}  flags={self.n_flags}",
            f"detection  P={self.precision:.2f}  R={self.recall:.2f}  F1={self.f1:.2f}",
            f"typed recall={self.typed_recall:.2f}   clean-doc false flags={self.clean_fp}",
            "per-type recall:",
        ]
        for et, (hit, tot) in sorted(self.per_type.items()):
            lines.append(f"  {et:<22} {hit}/{tot}")
        return "\n".join(lines)


def score_pair(perturbed: Judgment, report: AuditReport, clean_report: AuditReport | None = None) -> BenchScore:
    s = BenchScore()
    s.n_injections = len(perturbed.injections)
    s.n_flags = len(report.flags)

    target_sids = {inj.target_sid: inj for inj in perturbed.injections}
    matched: set[str] = set()

    for flag in report.flags:
        inj = target_sids.get(flag.hypothesis_sid)
        if inj is None:
            s.fp += 1
            continue
        if flag.hypothesis_sid not in matched:
            s.tp_detect += 1
            matched.add(flag.hypothesis_sid)
        if flag.flag_type == inj.error_type:
            s.tp_typed += 1 if f"typed::{flag.hypothesis_sid}" not in matched else 0
            matched.add(f"typed::{flag.hypothesis_sid}")

    s.fn = s.n_injections - s.tp_detect

    for inj in perturbed.injections:
        hit, tot = s.per_type.get(inj.error_type.value, (0, 0))
        s.per_type[inj.error_type.value] = (hit + (1 if inj.target_sid in matched else 0), tot + 1)

    if clean_report is not None:
        s.clean_fp = len(clean_report.flags)
    return s


def merge(scores: list[BenchScore]) -> BenchScore:
    total = BenchScore()
    for x in scores:
        total.tp_detect += x.tp_detect
        total.tp_typed += x.tp_typed
        total.fn += x.fn
        total.fp += x.fp
        total.clean_fp += x.clean_fp
        total.n_injections += x.n_injections
        total.n_flags += x.n_flags
        for k, (h, t) in x.per_type.items():
            H, T = total.per_type.get(k, (0, 0))
            total.per_type[k] = (H + h, T + t)
    return total
