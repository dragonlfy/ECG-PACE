"""Waveform measurements with explicit provenance and quality checks.

Measurements use supplied fiducial sample annotations. No unvalidated
onset/offset detector is silently assumed. Feature limits are research-side
configuration values and should not be treated as autonomous diagnoses.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .models import Action, ECGRecord, Evidence


@dataclass
class EvidenceTools:
    clinical_criteria: dict[str, dict] | None = None
    minimum_reliability: float = 0.5

    def execute(self, record: ECGRecord, action: Action) -> Evidence:
        if action.kind == "clinical_criterion":
            entry = (self.clinical_criteria or {}).get(action.criterion_key or "")
            return Evidence(
                id=f"ev:{action.id}", action_id=action.id, kind=action.kind,
                outcome="uncertain" if entry else "unavailable",
                observed=entry if entry else {}, reliability=1.0 if entry else 0.0,
                accepted=bool(entry), source=f"criteria:{action.criterion_key}",
                reason="Reference information; not an observation of this ECG",
                patient_observation=False,
            )
        try:
            return self._measure(record, action)
        except (KeyError, IndexError, ValueError, ArithmeticError) as exc:
            return self._unavailable(action, f"Measurement unavailable: {exc}")

    @staticmethod
    def _unavailable(action: Action, reason: str) -> Evidence:
        return Evidence(
            id=f"ev:{action.id}", action_id=action.id, kind=action.kind,
            outcome="unavailable", observed={}, reliability=0.0, accepted=False,
            source="ecg:raw_signal", lead=action.lead,
            start_s=action.start_s, end_s=action.end_s, reason=reason,
        )

    def _measure(self, record: ECGRecord, action: Action) -> Evidence:
        if action.kind == "frontal_axis":
            return self._frontal_axis(record, action)
        if not action.lead:
            raise ValueError("Patient-specific measurement requires a lead")
        data = record.signals[action.lead]
        start = max(0, int((action.start_s or 0) * record.fs))
        end = min(len(data), int((action.end_s or record.duration_seconds) * record.fs))
        if start >= end or end - start < 3:
            raise ValueError("Empty or too short acquisition window")
        window = data[start:end]
        finite_fraction = float(np.isfinite(window).mean())
        if finite_fraction < 0.98:
            raise ValueError("Insufficient finite waveform samples")
        # Signal sanity only. This factor is not a clinical SQI estimate.
        dynamic_range = float(np.ptp(window))
        quality = 0.95 if 0.01 <= dynamic_range <= 10.0 else 0.30
        annotations = record.annotations.get(action.lead, {})

        def events(name: str) -> np.ndarray:
            result = np.asarray(annotations.get(name, []), dtype=int)
            return result[(result >= start) & (result < end)]

        values: dict[str, float | int | str] = {}
        units: dict[str, str] = {}
        if action.kind == "rr_intervals":
            peaks = events("r_peaks")
            if len(peaks) < 3:
                raise ValueError("At least three R peaks required")
            intervals = np.diff(peaks) * 1000.0 / record.fs
            values = {
                "mean_rr_ms": float(np.mean(intervals)),
                "min_rr_ms": float(np.min(intervals)),
                "max_rr_ms": float(np.max(intervals)),
                "max_rr_change_ms": float(np.max(np.abs(np.diff(intervals)))) if len(intervals) > 1 else 0.0,
                "num_rr_intervals": int(len(intervals)),
            }
            units = {k: "ms" for k in values if k.endswith("_ms")}
        elif action.kind in {"qrs_duration", "pr_interval", "qt_interval"}:
            names = {
                "qrs_duration": ("qrs_onsets", "qrs_offsets"),
                "pr_interval": ("p_onsets", "qrs_onsets"),
                "qt_interval": ("qrs_onsets", "t_offsets"),
            }
            onsets = events(names[action.kind][0])
            offsets = events(names[action.kind][1])
            # Pair each onset with the closest subsequent offset; cap span.
            spans = []
            for onset in onsets:
                candidates = offsets[(offsets > onset) & (offsets < onset + int(record.fs * 0.65))]
                if len(candidates):
                    spans.append((candidates[0] - onset) * 1000.0 / record.fs)
            if not spans:
                raise ValueError(f"Missing usable fiducials for {action.kind}")
            key = {"qrs_duration": "qrs_ms", "pr_interval": "pr_ms", "qt_interval": "qt_ms"}[action.kind]
            values[key] = float(np.median(spans))
            values["num_beats"] = len(spans)
            units[key] = "ms"
        elif action.kind == "qtc_interval":
            # Bazett correction is provided as one explicit configurable baseline.
            q_on, t_end, r_peaks = events("qrs_onsets"), events("t_offsets"), events("r_peaks")
            if len(r_peaks) < 2:
                raise ValueError("At least two R peaks required for QTc")
            rr_s = float(np.median(np.diff(r_peaks)) / record.fs)
            if rr_s <= 0:
                raise ValueError("Invalid RR duration")
            qt_ms = []
            for q in q_on:
                next_t = t_end[(t_end > q) & (t_end < q + int(record.fs * 0.65))]
                if len(next_t):
                    qt_ms.append((next_t[0] - q) * 1000.0 / record.fs)
            if not qt_ms:
                raise ValueError("No delineated QT intervals")
            values = {"qtc_bazett_ms": float(np.median(qt_ms) / math.sqrt(rr_s)), "rr_s": rr_s}
            units = {"qtc_bazett_ms": "ms", "rr_s": "s"}
        elif action.kind == "p_wave_presence":
            peaks = events("r_peaks")
            p_peaks = events("p_peaks")
            if len(peaks) < 2:
                raise ValueError("At least two R peaks required")
            if "p_peaks" not in annotations:
                raise ValueError("P-wave annotations unavailable")
            num_with_p = sum(np.any((p_peaks >= r - int(record.fs * 0.28)) &
                                      (p_peaks < r - int(record.fs * 0.04))) for r in peaks)
            values = {"preceding_p_fraction": float(num_with_p / len(peaks)), "num_beats": len(peaks)}
            units = {"preceding_p_fraction": "fraction"}
        elif action.kind == "waveform_summary":
            values = {
                "min_mv": float(np.min(window)), "max_mv": float(np.max(window)),
                "rms_mv": float(np.sqrt(np.mean(window ** 2))),
            }
            units = {k: "mV" for k in values}
        elif action.kind == "st_level":
            offsets = events("qrs_offsets")
            onsets = events("qrs_onsets")
            if len(offsets) < 1 or len(onsets) < 1:
                raise ValueError("QRS fiducials required for ST measurement")
            shifts = []
            for q in offsets:
                st_center = q + int(0.06 * record.fs)
                recent = onsets[onsets < q]
                if not len(recent):
                    continue
                baseline_center = recent[-1] - int(0.055 * record.fs)
                if baseline_center < 0 or st_center >= len(data):
                    continue
                shifts.append(float(data[st_center] - data[baseline_center]))
            if not shifts:
                raise ValueError("No valid ST and baseline samples")
            values = {"median_st_shift_mv": float(np.median(shifts)), "num_beats": len(shifts)}
            units = {"median_st_shift_mv": "mV"}
        else:
            raise ValueError(f"Unsupported measurement tool: {action.kind}")

        outcome = "uncertain"
        if action.support_rule:
            rule = action.support_rule
            feature = str(rule["feature"])
            op = rule["op"]
            cutoff = float(rule["threshold"])
            if feature not in values:
                raise ValueError(f"Support rule refers to absent feature: {feature}")
            actual = float(values[feature])
            predicates = {
                "gt": actual > cutoff, "ge": actual >= cutoff,
                "lt": actual < cutoff, "le": actual <= cutoff,
            }
            if op not in predicates:
                raise ValueError(f"Unsupported support comparison: {op}")
            outcome = "support" if predicates[op] else "oppose"

        accepted = quality >= self.minimum_reliability
        return Evidence(
            id=f"ev:{action.id}", action_id=action.id, kind=action.kind,
            outcome=outcome if accepted else "unavailable", observed=values,
            reliability=quality if accepted else 0.0, accepted=accepted,
            source="ecg:raw_signal+fiducials" if annotations else "ecg:raw_signal",
            lead=action.lead, start_s=start / record.fs, end_s=end / record.fs,
            units=units, reason="" if accepted else "Signal sanity check failed",
        )

    def _frontal_axis(self, record: ECGRecord, action: Action) -> Evidence:
        """Approximate frontal-plane QRS axis from baseline-corrected I/aVF area.

        The returned angle is a research feature; electrode layout, fiducials,
        baseline and amplitude calibration require independent verification.
        """
        start = max(0, int((action.start_s or 0) * record.fs))
        end = min(len(record.signals["I"]), int((action.end_s or record.duration_seconds) * record.fs))
        areas = {}
        for lead in ("I", "aVF"):
            x = record.signals[lead]
            tags = record.annotations.get(lead, {})
            left = tags.get("qrs_onsets", [])
            right = tags.get("qrs_offsets", [])
            if not left or not right:
                raise ValueError(f"Missing QRS boundaries in {lead}")
            currents = []
            for first in left:
                if first < start or first >= end:
                    continue
                later = [v for v in right if first < v <= end and v - first < record.fs * 0.3]
                if not later:
                    continue
                last = min(later)
                segment = x[first:last]
                base_start = max(start, first - int(0.12 * record.fs))
                base_end = max(base_start + 1, first - int(0.04 * record.fs))
                baseline = float(np.median(x[base_start:base_end]))
                if not np.isfinite(segment).all() or not np.isfinite(baseline):
                    continue
                currents.append(float(np.sum(segment - baseline) / record.fs))
            if not currents:
                raise ValueError(f"No delineated QRS complexes in {lead}")
            areas[lead] = float(np.median(currents))
        if abs(areas["I"]) + abs(areas["aVF"]) < 1e-7:
            raise ValueError("Net QRS areas too small for axis estimation")
        angle = float(np.degrees(np.arctan2(areas["aVF"], areas["I"])))
        return Evidence(
            id=f"ev:{action.id}", action_id=action.id, kind=action.kind,
            outcome="uncertain", observed={"qrs_axis_deg": angle,
            "qrs_area_i_mv_s": areas["I"], "qrs_area_avf_mv_s": areas["aVF"]},
            reliability=0.85, accepted=0.85 >= self.minimum_reliability,
            source="ecg:raw_signal+fiducials", start_s=start / record.fs,
            end_s=end / record.fs, units={"qrs_axis_deg": "degree",
            "qrs_area_i_mv_s": "mV*s", "qrs_area_avf_mv_s": "mV*s"},
            reason="Approximate vector from QRS net areas; verify signal calibration",
        )
