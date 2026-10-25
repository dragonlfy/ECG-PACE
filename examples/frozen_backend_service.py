"""Minimal local HTTP bridge showing the provider request/response contract.

This serves the deterministic FixtureBackbone only. Replace it with a model
service that actually runs ECG-R1 to perform research inference.
"""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

from ecgpace.backbone import FixtureBackbone
from ecgpace.models import ECGRecord, Evidence, LEADS


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 10_000_000:
                raise ValueError("Request payload must be at most 10 MB")
            body = json.loads(self.rfile.read(size))
            raw = body["record"]
            record = ECGRecord(raw["record_id"], raw["fs"],
                               {k: np.asarray(raw["signals_mv"][k]) for k in LEADS},
                               annotations=raw.get("annotations", {}),
                               image_path=raw.get("image_path"))
            model = FixtureBackbone()
            if self.path == "/interpret":
                initial = model.interpret(record)
                result = {"report": initial.report, "beliefs": initial.beliefs}
            elif self.path == "/beliefs":
                ev = [Evidence(**e) for e in body["evidence"]]
                result = {"beliefs": model.conditioned_beliefs(record, body["initial_report"],
                            body["current_beliefs"], ev)}
            elif self.path == "/edits":
                from dataclasses import asdict
                ev = [Evidence(**e) for e in body["evidence"]]
                edits = model.propose_edits(record, body["initial_report"], body["beliefs"], ev)
                result = {"edits": [asdict(e) for e in edits]}
            else:
                self.send_error(404, "Unknown endpoint")
                return
            data = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_error(400, str(exc))


if __name__ == "__main__":
    print("Example bridge listening on http://127.0.0.1:8765")
    HTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
