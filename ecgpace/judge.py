"""Aggregate structured ECG judge output using manuscript score conventions."""
from __future__ import annotations

import math

CRITERIA = (
    "DiagnosisAccuracy", "AnalysisCompleteness", "AnalysisRelevance",
    "LeadEvidenceValidity", "GroundedECGUnderstanding", "EvidenceBasedReasoning",
    "RealisticDiagnosticProcess",
)


def aggregate_judge_record(payload: dict) -> dict[str, float]:
    """Convert item-level judge scores to seven metrics and Diagnosis Credit.

    The judge is external. This function neither fabricates items nor makes
    clinical decisions; it validates the supplied object and aggregates it.
    """
    if set(payload) != set(CRITERIA):
        raise ValueError("Judge response must contain exactly seven required criteria")
    values = {}
    for name in CRITERIA:
        entries = payload[name]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Judge criterion {name} must be a nonempty array")
        scores = []
        for item in entries:
            if not isinstance(item, dict) or "Score" not in item:
                raise ValueError(f"Invalid judge item for {name}")
            score = float(item["Score"])
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite judge score for {name}")
            if name == "AnalysisCompleteness":
                valid = score in (0, 1)
            elif name in CRITERIA[:4]:
                valid = score in (0, 1, 2)
            else:
                valid = 0 <= score <= 100
            if not valid:
                raise ValueError(f"Out-of-range judge score for {name}")
            scores.append(score)
        values[name] = scores
    diagnoses = values["DiagnosisAccuracy"]
    return {
        "DA": 100.0 * sum(s > 0 for s in diagnoses) / len(diagnoses),
        "Credit": 100.0 * sum(diagnoses) / (2 * len(diagnoses)),
        "AC": sum(values["AnalysisCompleteness"]),
        "AR": sum(values["AnalysisRelevance"]),
        "LEV": sum(values["LeadEvidenceValidity"]),
        "GEU": sum(values["GroundedECGUnderstanding"]) / len(values["GroundedECGUnderstanding"]),
        "EBR": sum(values["EvidenceBasedReasoning"]) / len(values["EvidenceBasedReasoning"]),
        "RDP": sum(values["RealisticDiagnosticProcess"]) / len(values["RealisticDiagnosticProcess"]),
    }
