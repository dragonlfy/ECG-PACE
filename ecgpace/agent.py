"""The ECG-PACE sequential control loop and evidence-gated revision."""
from __future__ import annotations

from dataclasses import asdict
import math
from .backbone import FrozenBackbone
from .config import PACEConfig
from .models import Action, ECGRecord, Evidence, ReportEdit, RunResult
from .policy import EmpiricalPolicy
from .tools import EvidenceTools


class PACEAgent:
    def __init__(self, backbone: FrozenBackbone, policy: EmpiricalPolicy,
                 tools: EvidenceTools, catalog: list[Action],
                 config: PACEConfig | None = None):
        self.backbone = backbone
        self.policy = policy
        self.tools = tools
        self.catalog = catalog
        self.config = config or PACEConfig()
        if len({a.id for a in catalog}) != len(catalog):
            raise ValueError("Action IDs must be unique")

    def _candidates(self, beliefs: dict[str, float], evidence: list[Evidence]) -> list[Action]:
        done = {e.action_id for e in evidence}
        return [a for a in self.catalog if a.id not in done and
                (not a.hypotheses or any(h in beliefs for h in a.hypotheses))]

    def run(self, record: ECGRecord) -> RunResult:
        initial = self.backbone.interpret(record)
        beliefs = dict(initial.beliefs)
        evidence: list[Evidence] = []
        trajectory: list[dict] = []
        stop_reason = "budget_exhausted"

        for step in range(self.config.max_actions):
            choices = self._candidates(beliefs, evidence)
            if not choices:
                stop_reason = "no_remaining_action"
                break
            scored = [(action, self.policy.score(
                action, beliefs, evidence,
                cost_weight=self.config.cost_weight,
                reliability_weight=self.config.reliability_weight,
            )) for action in choices]
            # Stable tie-breaking ensures that a fixed policy gives a fixed trace.
            scored.sort(key=lambda pair: (-pair[1]["value"], pair[0].id))
            chosen, selection = scored[0]
            if selection["value"] <= self.config.stop_threshold:
                stop_reason = "insufficient_expected_value"
                break
            observed = self.tools.execute(record, chosen)
            evidence.append(observed)
            before = dict(beliefs)
            if observed.accepted and observed.patient_observation:
                suggested = self.backbone.conditioned_beliefs(
                    record, initial.report, dict(beliefs), [e for e in evidence
                    if e.accepted and e.patient_observation],
                )
                # Apply reliability blending only to this step's newly observed
                # evidence. Earlier observations are context, not new updates.
                if set(suggested) != set(beliefs):
                    raise ValueError("Backbone belief query changed the hypothesis set")
                for label, current in beliefs.items():
                    target = float(suggested[label])
                    if not math.isfinite(target):
                        raise ValueError("Nonfinite evidence-conditioned model belief")
                    beliefs[label] = (1 - observed.reliability) * current + observed.reliability * target
            trajectory.append({
                "step": step + 1,
                "action": chosen.id,
                "action_value": selection,
                "available_action_values": {a.id: metrics["value"] for a, metrics in scored},
                "evidence_id": observed.id,
                "outcome": observed.outcome,
                "accepted": observed.accepted,
                "beliefs_before": before,
                "beliefs_after": dict(beliefs),
            })
        else:
            stop_reason = "budget_exhausted"

        observed_patient_evidence = [e for e in evidence if e.accepted and e.patient_observation]
        proposed = self.backbone.propose_edits(
            record, initial.report, dict(beliefs), observed_patient_evidence,
        ) if observed_patient_evidence else []
        final_report, revisions = self._apply_edits(initial.report, proposed, observed_patient_evidence)
        return RunResult(
            record_id=record.record_id, initial_report=initial.report, final_report=final_report,
            initial_beliefs=initial.beliefs, final_beliefs=beliefs,
            evidence=evidence, trajectory=trajectory, revisions=revisions,
            stop_reason=stop_reason,
        )

    def _apply_edits(self, initial_report: str, proposals: list[ReportEdit],
                     observed_patient_evidence: list[Evidence]) -> tuple[str, list[dict]]:
        approved = {e.id: e for e in observed_patient_evidence}
        report = initial_report
        records = []
        for edit in proposals:
            linked = list(edit.evidence_ids)
            reason = "applied"
            if edit.expected_benefit < self.config.revision_threshold:
                reason = "below_revision_threshold"
            elif not linked or any(eid not in approved for eid in linked):
                reason = "missing_validated_patient_evidence"
            elif not edit.old_text or edit.old_text == edit.new_text:
                reason = "invalid_or_unchanged_edit"
            elif report.count(edit.old_text) != 1:
                reason = "original_text_not_uniquely_matched"
            if reason == "applied":
                report = report.replace(edit.old_text, edit.new_text, 1)
            records.append({**asdict(edit), "status": reason})
        return report, records
