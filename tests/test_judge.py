import pytest
from ecgpace.judge import CRITERIA, aggregate_judge_record


def test_judge_aggregation():
    obj = {k: [{"Score": 80, "Explanation": ""}] for k in CRITERIA}
    obj["DiagnosisAccuracy"] = [{"Score": 2}, {"Score": 1}, {"Score": 0}]
    obj["AnalysisCompleteness"] = [{"Score": 1}, {"Score": 0}]
    obj["AnalysisRelevance"] = [{"Score": 2}, {"Score": 1}]
    obj["LeadEvidenceValidity"] = [{"Score": 1}, {"Score": 1}]
    scores = aggregate_judge_record(obj)
    assert scores["DA"] == pytest.approx(100 * 2/3)
    assert scores["Credit"] == 50
    assert scores["AC"] == 1
    assert scores["AR"] == 3
    assert scores["LEV"] == 2
    assert scores["GEU"] == scores["EBR"] == scores["RDP"] == 80


def test_judge_rejects_unexpected_schema():
    with pytest.raises(ValueError):
        aggregate_judge_record({"DiagnosisAccuracy": [{"Score": 2}]})
