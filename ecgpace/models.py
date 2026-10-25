"""Typed information exchanged among the frozen model, search policy and tools."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import math
import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
OUTCOMES = ("support", "oppose", "uncertain", "unavailable")


def _finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


@dataclass
class ECGRecord:
    record_id: str
    fs: float
    signals: dict[str, np.ndarray]
    # Fiducial annotations are expressed in sample indices, not milliseconds.
    annotations: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    image_path: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id or self.fs <= 0:
            raise ValueError("record_id and positive sampling frequency are required")
        if set(self.signals) != set(LEADS):
            raise ValueError(f"Expected 12 canonical leads: {LEADS}")
        arrays = {k: np.asarray(v, dtype=float) for k, v in self.signals.items()}
        sizes = {len(v) for v in arrays.values()}
        if len(sizes) != 1 or not sizes or next(iter(sizes)) < 3:
            raise ValueError("All leads must contain equal-length, nonempty 1D signals")
        if any(v.ndim != 1 for v in arrays.values()):
            raise ValueError("All ECG leads must be 1D")
        self.signals = arrays

    @property
    def duration_seconds(self) -> float:
        return len(self.signals["II"]) / self.fs


@dataclass(frozen=True)
class Action:
    id: str
    kind: str
    lead: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    cost: float = 1.0
    hypotheses: tuple[str, ...] = ()
    # Optional policy-side decision rule; applied only AFTER the action returns.
    support_rule: dict[str, Any] | None = None
    criterion_key: str | None = None

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "Action":
        data = dict(item)
        data["hypotheses"] = tuple(data.get("hypotheses", []))
        return cls(**data)

    def __post_init__(self) -> None:
        if not self.id or not self.kind or self.cost < 0:
            raise ValueError("Action needs id/kind and a nonnegative cost")
        if self.lead is not None and self.lead not in LEADS:
            raise ValueError(f"Unknown lead: {self.lead}")
        if self.start_s is not None and self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("Action time window must be increasing")


@dataclass
class Evidence:
    id: str
    action_id: str
    kind: str
    outcome: str
    observed: dict[str, Any]
    reliability: float
    accepted: bool
    source: str
    lead: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    units: dict[str, str] = field(default_factory=dict)
    reason: str = ""
    patient_observation: bool = True

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"Invalid outcome: {self.outcome}")
        if not 0 <= self.reliability <= 1:
            raise ValueError("reliability must be in [0, 1]")


@dataclass
class InitialInterpretation:
    report: str
    # Beliefs are raw log odds (log P(present) - log P(absent)).
    beliefs: dict[str, float]

    def __post_init__(self) -> None:
        self.beliefs = {k: _finite(v, f"belief[{k}]") for k, v in self.beliefs.items()}
        if not self.beliefs:
            raise ValueError("An initial interpretation requires diagnostic hypotheses")


@dataclass(frozen=True)
class ReportEdit:
    old_text: str
    new_text: str
    evidence_ids: tuple[str, ...]
    expected_benefit: float

    @classmethod
    def from_dict(cls, item: dict[str, Any]) -> "ReportEdit":
        return cls(
            old_text=item["old_text"], new_text=item["new_text"],
            evidence_ids=tuple(item.get("evidence_ids", [])),
            expected_benefit=float(item.get("expected_benefit", 0)),
        )


@dataclass
class RunResult:
    record_id: str
    initial_report: str
    final_report: str
    initial_beliefs: dict[str, float]
    final_beliefs: dict[str, float]
    evidence: list[Evidence]
    trajectory: list[dict[str, Any]]
    revisions: list[dict[str, Any]]
    stop_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
