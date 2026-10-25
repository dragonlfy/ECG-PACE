"""Command-line interface for offline policy fitting, inference and inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .agent import PACEAgent
from .backbone import FixtureBackbone, HTTPBackbone
from .config import PACEConfig
from .io import load_catalog, load_record, save_result
from .policy import EmpiricalPolicy, fit_empirical_policy
from .tools import EvidenceTools


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="ecg-pace", description="Hypothesis-guided ECG evidence acquisition"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run a complete trace on generated ECG-like signals")
    demo.add_argument("--out", default="outputs/demo_trace.json")
    export = commands.add_parser("export-example", help="Save generated 12-lead signals and sidecar JSON")
    export.add_argument("--directory", default="outputs/generated_record")
    run = commands.add_parser("run", help="Run with an ECG record and frozen-backbone provider")
    run.add_argument("--npz", required=True)
    run.add_argument("--metadata", required=True)
    run.add_argument("--catalog", required=True)
    run.add_argument("--policy", required=True)
    run.add_argument("--criteria", default=None)
    run.add_argument("--endpoint", required=True, help="Frozen model bridge root URL")
    run.add_argument("--allow-remote", action="store_true", default=False)
    run.add_argument("--max-actions", type=int, default=4)
    run.add_argument("--out", default="outputs/run_trace.json")
    fit = commands.add_parser("fit-policy", help="Fit an empirical action-value policy")
    fit.add_argument("--input", required=True, help="Training JSONL rows")
    fit.add_argument("--out", required=True)
    fit.add_argument("--smoothing", type=float, default=1)
    args = parser.parse_args(argv)

    if args.command == "fit-policy":
        rows = [json.loads(line) for line in Path(args.input).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        parameters = fit_empirical_policy(rows, smoothing=args.smoothing)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(parameters['actions'])} action distributions to {args.out}")
        return

    if args.command == "export-example":
        import numpy as np
        from .demo import example_record
        rec = example_record()
        directory = Path(args.directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / "record.npz", **rec.signals)
        (directory / "record.json").write_text(json.dumps({
            "record_id": rec.record_id, "fs": rec.fs, "annotations": rec.annotations,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"Saved generated ECG inputs in {directory}")
        return

    if args.command == "demo":
        from .demo import example_record, example_actions, example_policy
        record, catalog, policy, backbone = (
            example_record(), example_actions(), example_policy(), FixtureBackbone()
        )
        criteria = {}
        config = PACEConfig()
    else:
        record = load_record(args.npz, args.metadata)
        catalog = load_catalog(args.catalog)
        policy = EmpiricalPolicy.from_json(args.policy)
        backbone = HTTPBackbone(args.endpoint, allow_remote=args.allow_remote)
        criteria = json.loads(Path(args.criteria).read_text(encoding="utf-8")) if args.criteria else {}
        config = PACEConfig(max_actions=args.max_actions)
    result = PACEAgent(
        backbone=backbone, policy=policy,
        tools=EvidenceTools(criteria, minimum_reliability=config.minimum_evidence_reliability),
        catalog=catalog, config=config,
    ).run(record)
    save_result(result, args.out)
    print(json.dumps({
        "record_id": result.record_id,
        "stop_reason": result.stop_reason,
        "executed_actions": [t["action"] for t in result.trajectory],
        "initial_report": result.initial_report,
        "final_report": result.final_report,
        "trace_path": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
