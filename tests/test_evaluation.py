import pytest
from ecgpace.evaluation import diagnosis_scores, paired_bootstrap_ci


def test_diagnosis_scores():
    scores = diagnosis_scores([{"pac", "rbbb"}, {"normal"}], [{"pac"}, {"rbbb"}])
    assert scores["tp"] == 1
    assert scores["fp"] == 2
    assert scores["fn"] == 1
    assert scores["precision"] == pytest.approx(1/3)
    assert scores["recall"] == pytest.approx(1/2)


def test_paired_ci_is_deterministic():
    a = [0.9, .8, .6, .7, .9]
    b = [.7, .8, .5, .6, .8]
    ci = paired_bootstrap_ci(a, b, samples=1000, seed=5)
    assert ci["mean_difference"] == pytest.approx(.1)
    assert ci["lower"] <= ci["mean_difference"] <= ci["upper"]
    assert paired_bootstrap_ci(a, b, samples=1000, seed=5) == ci
