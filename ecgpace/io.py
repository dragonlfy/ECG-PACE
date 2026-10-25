"""File loading and safe result persistence."""
from __future__ import annotations

import json
from pathlib import Path
from .models import ECGRecord, Action, RunResult, LEADS


def load_record(npz_path: str | Path, metadata_path: str | Path) -> ECGRecord:
    """Read a twelve-lead .npz and sidecar JSON. Voltage must be in millivolts."""
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as obj:
        missing = set(LEADS) - set(obj.files)
        if missing:
            raise ValueError(f"Missing ECG leads in {npz_path}: {sorted(missing)}")
        signals = {lead: obj[lead].copy() for lead in LEADS}
    meta = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return ECGRecord(
        record_id=str(meta["record_id"]), fs=float(meta["fs"]),
        signals=signals, annotations=meta.get("annotations", {}),
        image_path=meta.get("image_path"),
    )


def load_catalog(path: str | Path) -> list[Action]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = [Action.from_dict(v) for v in items]
    if len({a.id for a in actions}) != len(actions):
        raise ValueError("Duplicate action IDs in catalog")
    return actions


def save_result(result: RunResult, path: str | Path) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
