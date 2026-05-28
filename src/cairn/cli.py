from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from .compiler import compile_plan
from .jsonutil import load_json, write_json
from .mapping import build_rank_map
from .simulator import simulate_plan
from .validation import (
    ValidationReport,
    validate_bundle,
    validate_checkpoint,
    validate_dataset,
    validate_model,
    validate_topology,
    validate_training,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"cairn: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cairn", description="Static AI training planner")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate one spec file or a full bundle")
    validate.add_argument("path", nargs="?", type=Path, help="single JSON spec path")
    validate.add_argument("--kind", choices=("model", "training", "topology", "dataset", "checkpoint"), help="spec kind")
    add_bundle_args(validate)
    validate.add_argument("--out", type=Path, help="write validation report")
    validate.set_defaults(func=cmd_validate)

    map_cmd = sub.add_parser("map", help="generate deterministic rank map")
    map_cmd.add_argument("--topology", required=True, type=Path)
    map_cmd.add_argument("--training", required=True, type=Path)
    map_cmd.add_argument("--out", type=Path)
    map_cmd.set_defaults(func=cmd_map)

    compile_cmd = sub.add_parser("compile", help="compile a static training plan")
    add_bundle_args(compile_cmd, required=True)
    compile_cmd.add_argument("--out", required=True, type=Path)
    compile_cmd.set_defaults(func=cmd_compile)

    simulate = sub.add_parser("simulate", help="simulate a compiled plan")
    simulate.add_argument("plan_dir", type=Path)
    simulate.add_argument("--failure")
    simulate.add_argument("--out", type=Path)
    simulate.set_defaults(func=cmd_simulate)

    report = sub.add_parser("report", help="print a compact JSON report summary")
    report.add_argument("path", type=Path)
    report.set_defaults(func=cmd_report)

    return parser


def add_bundle_args(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument("--model", required=required, type=Path)
    parser.add_argument("--training", required=required, type=Path)
    parser.add_argument("--topology", required=required, type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--checkpoint", type=Path)


def cmd_validate(args: argparse.Namespace) -> int:
    if args.model or args.training or args.topology:
        if not (args.model and args.training and args.topology):
            raise ValueError("bundle validation requires --model, --training, and --topology")
        model = load_json(args.model)
        training = load_json(args.training)
        topology = load_json(args.topology)
        dataset = load_json(args.dataset) if args.dataset else None
        checkpoint = load_json(args.checkpoint) if args.checkpoint else None
        report = validate_bundle(
            model=model,
            training=training,
            topology=topology,
            dataset=dataset,
            checkpoint=checkpoint,
        )
    else:
        if args.path is None or args.kind is None:
            raise ValueError("single-file validation requires PATH and --kind")
        report = validate_single(args.kind, load_json(args.path))

    output_report(report, args.out)
    return 0 if not report.errors else 1


def validate_single(kind: str, value: dict[str, Any]) -> ValidationReport:
    validators = {
        "model": validate_model,
        "training": validate_training,
        "topology": validate_topology,
        "dataset": validate_dataset,
        "checkpoint": validate_checkpoint,
    }
    return validators[kind](value)


def cmd_map(args: argparse.Namespace) -> int:
    topology = load_json(args.topology)
    training = load_json(args.training)
    report = validate_bundle(
        model=minimal_model_for_mapping(training),
        training=training,
        topology=topology,
    )
    errors = [message for message in report.errors if not message.startswith("model has")]
    if errors:
        report.errors = errors
        report.raise_for_errors()
    rank_map = build_rank_map(topology, training)
    if args.out:
        write_json(args.out, rank_map)
    else:
        output_dict(rank_map)
    return 0


def cmd_compile(args: argparse.Namespace) -> int:
    manifest = compile_plan(
        model=load_json(args.model),
        training=load_json(args.training),
        topology=load_json(args.topology),
        dataset=load_json(args.dataset) if args.dataset else None,
        checkpoint=load_json(args.checkpoint) if args.checkpoint else None,
        out_dir=args.out,
    )
    print(f"compiled plan {manifest['plan_id']} to {args.out}")
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    report = simulate_plan(args.plan_dir, failure=args.failure, out_path=args.out)
    if args.out:
        print(f"wrote simulation report to {args.out}")
    else:
        output_dict(report)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    value = load_json(args.path)
    status = value.get("status")
    if status:
        print(f"status: {status}")
        for key in ("errors", "warnings"):
            items = value.get(key) or []
            print(f"{key}: {len(items)}")
            for item in items:
                print(f"  - {item}")
        return 0 if status == "ok" else 1

    print(f"plan_id: {value.get('plan_id', 'unknown')}")
    for key in (
        "world_size",
        "pipeline_bubble_ratio",
        "pipeline_stage_imbalance",
        "memory_high_water_bytes_per_rank",
        "total_communication_bytes_per_step",
    ):
        if key in value:
            print(f"{key}: {value[key]}")
    return 0


def output_report(report: ValidationReport, out: Path | None) -> None:
    if out:
        write_json(out, report.to_dict())
    else:
        output_dict(report.to_dict())


def output_dict(value: dict[str, Any]) -> None:
    import json

    print(json.dumps(value, indent=2, sort_keys=True))


def minimal_model_for_mapping(training: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "family": "gpt",
        "layers": max(1, training["parallelism"]["pipeline"]),
        "hidden_size": 8,
        "attention_heads": 1,
        "sequence_length": 8,
        "vocab_size": 32,
        "ffn_hidden_size": 32,
    }
