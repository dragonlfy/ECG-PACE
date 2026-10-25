"""Deterministic generated waveform example, not a clinical ECG or test-set sample."""
from __future__ import annotations

import numpy as np
from .models import Action, ECGRecord, LEADS
from .policy import EmpiricalPolicy


def example_record() -> ECGRecord:
    fs = 250
    t = np.arange(0, 6, 1 / fs)
    peaks = np.array([0.6, 1.5, 2.4, 2.98, 4.05, 4.95])
    # Gaussian pulses are signal fixtures for testing the code paths.
    base = 0.01 * np.sin(2 * np.pi * 1.1 * t)
    for p in peaks:
        base += np.exp(-0.5 * ((t - p) / 0.017) ** 2)
        base += 0.12 * np.exp(-0.5 * ((t - p + 0.16) / 0.04) ** 2)
        base += 0.20 * np.exp(-0.5 * ((t - p - 0.22) / 0.07) ** 2)
    scales = {lead: 0.9 if lead in ("I", "II", "V1") else 0.65 for lead in LEADS}
    signals = {lead: base * scale for lead, scale in scales.items()}
    r_samples = [int(round(p * fs)) for p in peaks]
    annotation = {
        "r_peaks": r_samples,
        "qrs_onsets": [p - 10 for p in r_samples],
        "qrs_offsets": [p + 13 for p in r_samples],
        "p_peaks": [p - 40 for p in r_samples],
    }
    return ECGRecord(
        record_id="generated-example-001", fs=fs, signals=signals,
        annotations={lead: dict(annotation) for lead in LEADS},
    )


def example_actions() -> list[Action]:
    return [
        Action(id="rr_ii", kind="rr_intervals", lead="II", cost=0.15,
               hypotheses=("premature_atrial_contractions",),
               support_rule={"feature": "max_rr_change_ms", "op": "gt", "threshold": 180}),
        Action(id="qrs_v1", kind="qrs_duration", lead="V1", cost=0.25,
               hypotheses=("wide_qrs",),
               support_rule={"feature": "qrs_ms", "op": "ge", "threshold": 120}),
        Action(id="p_v1", kind="p_wave_presence", lead="V1", cost=0.25,
               hypotheses=("premature_atrial_contractions",)),
    ]


def example_policy() -> EmpiricalPolicy:
    # Small, hand-authored action expectations for the generated example only.
    # These are not trained parameters or measurements of diagnostic quality.
    weights = {"rr_ii": (0.39, 0.04), "qrs_v1": (0.28, 0.10), "p_v1": (0.09, 0.10)}
    actions = {}
    for action_id, (gain, risk) in weights.items():
        actions[action_id] = {
            "unreliability": risk,
            "outcomes": {
                "support": {"probability": 0.45, "mean_delta_quality": gain},
                "oppose": {"probability": 0.45, "mean_delta_quality": gain},
                "uncertain": {"probability": 0.05, "mean_delta_quality": gain},
                "unavailable": {"probability": 0.05, "mean_delta_quality": gain},
            },
        }
    return EmpiricalPolicy({"format_version": 1, "actions": actions})
