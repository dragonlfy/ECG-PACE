"""Deterministic diagnosis-label evaluation and paired uncertainty estimates."""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np


def diagnosis_scores(predicted: Sequence[set[str]], reference: Sequence[set[str]]) -> dict[str, float]:
    if not predicted or len(predicted) != len(reference):
        raise ValueError("Predicted and reference arrays need equal nonzero lengths")
    tp = fp = fn = 0
    for p, r in zip(predicted, reference):
        tp += len(p & r)
        fp += len(p - r)
        fn += len(r - p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def paired_bootstrap_ci(a: Sequence[float], b: Sequence[float], *,
                        samples: int = 10000, confidence: float = 0.95,
                        seed: int = 42) -> dict[str, float]:
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("Paired score vectors require equal length and at least two elements")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Paired score vectors must be finite")
    if samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap settings")
    differences = x - y
    rng = np.random.default_rng(seed)
    # Iterate to avoid materializing a 10k x 2k array for large evaluations.
    estimates = np.empty(samples)
    for i in range(samples):
        index = rng.integers(0, len(differences), size=len(differences))
        estimates[i] = differences[index].mean()
    alpha = (1 - confidence) / 2
    return {"mean_difference": float(differences.mean()),
            "lower": float(np.quantile(estimates, alpha)),
            "upper": float(np.quantile(estimates, 1 - alpha)),
            "confidence": confidence, "resamples": samples}
