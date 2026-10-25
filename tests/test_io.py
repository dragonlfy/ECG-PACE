import json
import numpy as np
import pytest
from ecgpace.demo import example_record
from ecgpace.io import load_catalog, load_record
from ecgpace.models import LEADS


def test_roundtrip_record(tmp_path):
    record = example_record()
    np.savez(tmp_path / "wave.npz", **record.signals)
    (tmp_path / "meta.json").write_text(json.dumps({
        "record_id": record.record_id, "fs": record.fs, "annotations": record.annotations,
    }))
    restored = load_record(tmp_path / "wave.npz", tmp_path / "meta.json")
    assert restored.record_id == record.record_id
    assert list(restored.signals) == list(LEADS)


def test_record_rejects_missing_leads():
    record = example_record()
    del record.signals["II"]
    from ecgpace.models import ECGRecord
    with pytest.raises(ValueError):
        ECGRecord("bad", fs=250, signals=record.signals)


def test_duplicate_catalog_ids_rejected(tmp_path):
    path = tmp_path / "actions.json"
    path.write_text(json.dumps([{"id": "x", "kind": "rr_intervals"}] * 2))
    with pytest.raises(ValueError):
        load_catalog(path)
