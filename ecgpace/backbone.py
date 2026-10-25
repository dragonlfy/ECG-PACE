"""Frozen inference interfaces. No model weights are modified by this package."""
from __future__ import annotations

from abc import ABC, abstractmethod
import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse

from .models import ECGRecord, Evidence, InitialInterpretation, ReportEdit


class FrozenBackbone(ABC):
    @abstractmethod
    def interpret(self, record: ECGRecord) -> InitialInterpretation:
        """Generate an initial report and diagnosis-level log odds."""

    @abstractmethod
    def conditioned_beliefs(self, record: ECGRecord, initial_report: str,
                            current: dict[str, float], evidence: list[Evidence]) -> dict[str, float]:
        """Infer beliefs conditioned only on evidence already observed."""

    @abstractmethod
    def propose_edits(self, record: ECGRecord, initial_report: str,
                      beliefs: dict[str, float], evidence: list[Evidence]) -> list[ReportEdit]:
        """Suggest narrowly scoped, evidence-linked report edits."""


class FixtureBackbone(FrozenBackbone):
    """Deterministic teaching backend for generated signals; not a clinical model."""

    def interpret(self, record: ECGRecord) -> InitialInterpretation:
        del record
        return InitialInterpretation(
            report="Sinus rhythm.\nNo significant premature beats.\nQRS appears narrow.",
            beliefs={"premature_atrial_contractions": -0.8, "wide_qrs": -1.0},
        )

    def conditioned_beliefs(self, record: ECGRecord, initial_report: str,
                            current: dict[str, float], evidence: list[Evidence]) -> dict[str, float]:
        del record, initial_report
        updated = dict(current)
        # Scripted example effect, based strictly on acquired measurements.
        for ev in evidence:
            if ev.kind == "rr_intervals" and "max_rr_change_ms" in ev.observed:
                updated["premature_atrial_contractions"] = (
                    1.3 if ev.observed["max_rr_change_ms"] > 180 else -1.3
                )
            if ev.kind == "qrs_duration" and "qrs_ms" in ev.observed:
                updated["wide_qrs"] = 1.2 if ev.observed["qrs_ms"] >= 120 else -1.2
        return updated

    def propose_edits(self, record: ECGRecord, initial_report: str,
                      beliefs: dict[str, float], evidence: list[Evidence]) -> list[ReportEdit]:
        del record, initial_report
        result = []
        rr_evidence = [e for e in evidence if e.kind == "rr_intervals" and e.accepted]
        if rr_evidence and beliefs.get("premature_atrial_contractions", 0) > 0:
            result.append(ReportEdit(
                old_text="No significant premature beats.",
                new_text="Premature beats are present; targeted atrial evaluation is recommended.",
                evidence_ids=(rr_evidence[-1].id,), expected_benefit=0.25,
            ))
        return result


class HTTPBackbone(FrozenBackbone):
    """HTTP JSON bridge for a separately hosted, frozen multimodal ECG model.

    POST /interpret: {record: {...}} -> {report: str, beliefs: {diagnosis: log_odds}}
    POST /beliefs: {record, initial_report, current_beliefs, evidence}
                   -> {beliefs: {diagnosis: log_odds}}
    POST /edits: {record, initial_report, beliefs, evidence}
                 -> {edits: [{old_text, new_text, evidence_ids, expected_benefit}]}

    Only loopback hosts are accepted unless allow_remote=True is explicitly set.
    """

    def __init__(self, base_url: str, *, allow_remote: bool = False, timeout: int = 120):
        parsed = urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("base_url must be an http(s) URL")
        if not allow_remote and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("Remote ECG transfer requires explicit allow_remote=True")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @staticmethod
    def _record_payload(record: ECGRecord) -> dict:
        # Image file paths are local to the caller; avoid transmitting or
        # claiming to transmit image bytes implicitly.
        return {
            "record_id": record.record_id, "fs": record.fs,
            "signals_mv": {lead: values.tolist() for lead, values in record.signals.items()},
            "annotations": record.annotations,
            "image_path": record.image_path,
        }

    def _post(self, route: str, payload: dict) -> dict:
        content = json.dumps(payload, allow_nan=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/{route}", data=content,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.load(response)
        except (HTTPError, URLError) as exc:
            raise ConnectionError(f"Frozen backbone endpoint failed at {route}: {exc}") from exc
        if not isinstance(result, dict):
            raise TypeError("Frozen backbone must return a JSON object")
        return result

    def interpret(self, record: ECGRecord) -> InitialInterpretation:
        result = self._post("interpret", {"record": self._record_payload(record)})
        return InitialInterpretation(report=str(result["report"]), beliefs=result["beliefs"])

    def conditioned_beliefs(self, record: ECGRecord, initial_report: str,
                            current: dict[str, float], evidence: list[Evidence]) -> dict[str, float]:
        from dataclasses import asdict
        result = self._post("beliefs", {
            "record": self._record_payload(record), "initial_report": initial_report,
            "current_beliefs": current, "evidence": [asdict(e) for e in evidence],
        })
        return {str(k): float(v) for k, v in result["beliefs"].items()}

    def propose_edits(self, record: ECGRecord, initial_report: str,
                      beliefs: dict[str, float], evidence: list[Evidence]) -> list[ReportEdit]:
        from dataclasses import asdict
        result = self._post("edits", {
            "record": self._record_payload(record), "initial_report": initial_report,
            "beliefs": beliefs, "evidence": [asdict(e) for e in evidence],
        })
        return [ReportEdit.from_dict(v) for v in result.get("edits", [])]
