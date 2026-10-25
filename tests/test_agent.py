import numpy as np
from ecgpace.agent import PACEAgent
from ecgpace.backbone import FixtureBackbone
from ecgpace.config import PACEConfig
from ecgpace.demo import example_actions, example_policy, example_record
from ecgpace.models import Action, Evidence, ReportEdit
from ecgpace.policy import EmpiricalPolicy
from ecgpace.tools import EvidenceTools


def build_agent(**kwargs):
    config = kwargs.get("config", PACEConfig())
    return PACEAgent(
        FixtureBackbone(), kwargs.get("policy", example_policy()),
        EvidenceTools(minimum_reliability=config.minimum_evidence_reliability),
        kwargs.get("actions", example_actions()), config,
    )


def test_full_demo_run_has_supported_revision():
    result = build_agent().run(example_record())
    assert result.record_id.startswith("generated-")
    assert [t["action"] for t in result.trajectory] == ["rr_ii", "qrs_v1"]
    assert result.stop_reason == "insufficient_expected_value"
    assert "Premature beats are present" in result.final_report
    assert "QRS appears narrow." in result.final_report
    assert result.revisions[0]["status"] == "applied"
    assert result.final_beliefs["premature_atrial_contractions"] > result.initial_beliefs["premature_atrial_contractions"]


def test_missing_fiducials_become_unavailable():
    record = example_record()
    record.annotations = {}
    observed = EvidenceTools().execute(record, example_actions()[0])
    assert observed.outcome == "unavailable"
    assert observed.accepted is False
    assert observed.reliability == 0


def test_evidence_maintains_units_and_traceability():
    record = example_record()
    ev = EvidenceTools().execute(record, example_actions()[0])
    assert ev.units["mean_rr_ms"] == "ms"
    assert ev.lead == "II" and ev.patient_observation
    assert 0 <= ev.start_s < ev.end_s <= record.duration_seconds
    assert ev.observed["max_rr_change_ms"] > 180


def test_threshold_stops_before_any_acquisition():
    config = PACEConfig(stop_threshold=100)
    result = build_agent(config=config).run(example_record())
    assert not result.evidence
    assert result.final_report == result.initial_report
    assert result.stop_reason == "insufficient_expected_value"


def test_revisions_must_be_unique_and_supported():
    agent = build_agent()
    valid = Evidence("ev:good", "good", "rr_intervals", "support", {},
                     0.95, True, "test", lead="II")
    edits = [
        ReportEdit("abc", "xyz", ("ev:absent",), 1),
        ReportEdit("abc", "xyz", ("ev:good",), 0.01),
        ReportEdit("not found", "xyz", ("ev:good",), 0.50),
        ReportEdit("abc", "xyz", ("ev:good",), 0.50),
    ]
    final, attempts = agent._apply_edits("abc and def", edits, [valid])
    assert final == "xyz and def"
    assert [v["status"] for v in attempts] == [
        "missing_validated_patient_evidence",
        "below_revision_threshold",
        "original_text_not_uniquely_matched",
        "applied",
    ]


def test_reference_criteria_do_not_change_patient_beliefs():
    criteria = {"atrial_ectopy": {"text": "reference checklist", "version": "test"}}
    action = Action(id="reference", kind="clinical_criterion", criterion_key="atrial_ectopy")
    obs = EvidenceTools(criteria).execute(example_record(), action)
    assert obs.accepted and not obs.patient_observation
    # A reference-only run cannot trigger a selective edit.
    values = {"actions": {"reference": {"unreliability": 0, "outcomes": {
        "uncertain": {"probability": 1, "mean_delta_quality": .5}}}}}
    agent = PACEAgent(FixtureBackbone(), EmpiricalPolicy(values), EvidenceTools(criteria), [action])
    result = agent.run(example_record())
    assert result.final_beliefs == result.initial_beliefs
    assert result.final_report == result.initial_report


def test_invalid_evidence_does_not_modify_beliefs():
    record = example_record()
    record.signals["II"][:] = np.nan
    action = example_actions()[0]
    agent = build_agent(actions=[action])
    result = agent.run(record)
    assert result.evidence[0].outcome == "unavailable"
    assert result.final_beliefs == result.initial_beliefs


def test_action_selection_does_not_need_unexecuted_waveform():
    policy = example_policy()
    actions = example_actions()
    values1 = [policy.score(a, {}, [], .1, .2) for a in actions]
    values2 = [policy.score(a, {"x": 10.0}, [object()], .1, .2) for a in actions]
    assert values1 == values2


def test_frontal_axis_reports_traceable_feature():
    action = Action(id="axis", kind="frontal_axis", cost=0.2)
    evidence = EvidenceTools().execute(example_record(), action)
    assert evidence.accepted
    assert evidence.units["qrs_axis_deg"] == "degree"
    assert -180 <= evidence.observed["qrs_axis_deg"] <= 180
