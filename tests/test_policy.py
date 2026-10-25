import pytest
from ecgpace.models import Action
from ecgpace.policy import EmpiricalPolicy, fit_empirical_policy


def test_fitted_outcome_probabilities_sum_to_one():
    rows = [
        {"action_id": "a", "outcome": "support", "delta_quality": .4, "unreliable": False},
        {"action_id": "a", "outcome": "oppose", "delta_quality": .1, "unreliable": True},
    ]
    fitted = fit_empirical_policy(rows)
    policy = EmpiricalPolicy(fitted)
    predictions = policy.outcomes(Action("a", "rr_intervals", lead="II"), {}, [])
    assert sum(p.probability for p in predictions.values()) == pytest.approx(1)
    assert policy.unreliable_risk(Action("a", "rr_intervals", lead="II")) == .5


def test_missing_policy_actions_have_conservative_value():
    policy = EmpiricalPolicy({"actions": {}})
    score = policy.score(Action("not_fitted", "rr_intervals", lead="II"), {}, [], .1, .2)
    assert score["value"] < 0


def test_training_rejects_invalid_rows():
    with pytest.raises(ValueError):
        fit_empirical_policy([{"action_id": "x", "outcome": "bad", "delta_quality": 0, "unreliable": False}])
    with pytest.raises(ValueError):
        fit_empirical_policy([{"action_id": "x", "outcome": "support", "delta_quality": 2, "unreliable": False}])
