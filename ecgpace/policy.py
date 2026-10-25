"""Prospective action valuation without access to unexecuted measurements."""
from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Iterable
from .models import Action, OUTCOMES


@dataclass(frozen=True)
class OutcomeEstimate:
    probability: float
    expected_quality_gain: float


class EmpiricalPolicy:
    """An offline-fitted outcome and quality model, indexed by action ID.

    Expected gain is evaluated prospectively:
      V(a) = sum_z P(z | state, a) DeltaQ(state, a, z) - lambda Cost(a) - mu Risk(a).
    The currently released tabular backend conditions on action ID; stronger
    estimators can replace `outcomes` while retaining the same interface.
    """

    def __init__(self, parameters: dict[str, Any]):
        self.parameters = parameters

    @classmethod
    def from_json(cls, path: str | Path) -> "EmpiricalPolicy":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def outcomes(self, action: Action, beliefs: dict[str, float], evidence: list) -> dict[str, OutcomeEstimate]:
        del beliefs, evidence  # This tabular estimator is action-conditional only.
        entry = self.parameters.get("actions", {}).get(action.id)
        if entry is None:
            return {z: OutcomeEstimate(0.25, 0.0) for z in OUTCOMES}
        values = entry.get("outcomes", {})
        distribution = {}
        for z in OUTCOMES:
            point = values.get(z, {})
            distribution[z] = OutcomeEstimate(
                probability=float(point.get("probability", 0)),
                expected_quality_gain=float(point.get("mean_delta_quality", 0)),
            )
        total = sum(v.probability for v in distribution.values())
        if total <= 0:
            raise ValueError(f"Policy action {action.id} has invalid outcome distribution")
        return {k: OutcomeEstimate(v.probability / total, v.expected_quality_gain)
                for k, v in distribution.items()}

    def unreliable_risk(self, action: Action) -> float:
        return max(0.0, min(1.0, float(
            self.parameters.get("actions", {}).get(action.id, {}).get("unreliability", 1.0)
        )))

    def score(self, action: Action, beliefs: dict[str, float], evidence: list,
              cost_weight: float, reliability_weight: float) -> dict[str, float]:
        preds = self.outcomes(action, beliefs, evidence)
        gain = sum(p.probability * p.expected_quality_gain for p in preds.values())
        cost_penalty = cost_weight * action.cost
        reliability_penalty = reliability_weight * self.unreliable_risk(action)
        return {
            "expected_gain": gain, "cost_penalty": cost_penalty,
            "reliability_penalty": reliability_penalty,
            "value": gain - cost_penalty - reliability_penalty,
        }


def fit_empirical_policy(rows: Iterable[dict[str, Any]], smoothing: float = 1.0) -> dict:
    """Fit P(z|action) and E[DeltaQ|action,z] from offline training rows.

    Required columns: action_id, outcome, delta_quality, unreliable (bool).
    `delta_quality` is normalized to the same quality scale as thresholds.
    """
    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["outcome"] not in OUTCOMES:
            raise ValueError(f"Unknown outcome {row['outcome']}")
        delta = float(row["delta_quality"])
        if not -1 <= delta <= 1:
            raise ValueError("delta_quality must lie in [-1, 1]")
        grouped[str(row["action_id"])].append(row)
    result = {"format_version": 1, "estimator": "action_conditional_empirical", "actions": {}}
    for action_id, observations in sorted(grouped.items()):
        n = len(observations)
        estimates = {}
        for outcome in OUTCOMES:
            subset = [r for r in observations if r["outcome"] == outcome]
            estimates[outcome] = {
                "probability": (len(subset) + smoothing) / (n + smoothing * len(OUTCOMES)),
                "mean_delta_quality": (sum(float(r["delta_quality"]) for r in subset) / len(subset))
                if subset else 0.0,
                "count": len(subset),
            }
        result["actions"][action_id] = {
            "outcomes": estimates,
            "unreliability": sum(bool(r["unreliable"]) for r in observations) / n,
            "count": n,
        }
    return result
