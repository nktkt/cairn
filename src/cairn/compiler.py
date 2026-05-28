from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

from .jsonutil import canonical_json, write_json
from .mapping import assign_layers, build_comm_groups, build_rank_map
from .op_registry import registry_metadata, validate_op
from .validation import estimate_memory_bytes, required_world_size, validate_bundle


STREAMS = {
    "compute_hi": 0,
    "compute_lo": 1,
    "comm_tp": 2,
    "comm_pp": 3,
    "comm_dp": 4,
    "io": 5,
}

OP_CLASS_IDS = {
    "compute": 0,
    "communication": 1,
    "io": 2,
}

BINARY_PLAN_MAGIC = b"CAIRNPLN"
BINARY_PLAN_VERSION = 1
BINARY_PLAN_HEADER = struct.Struct("<8sII64s64sIIIIIQQ")
BINARY_PLAN_SEGMENT = struct.Struct("<64sQQ")
BINARY_PLAN_OP = struct.Struct("<II48s32sQ")


def compile_plan(
    *,
    model: dict[str, Any],
    training: dict[str, Any],
    topology: dict[str, Any],
    dataset: dict[str, Any] | None,
    checkpoint: dict[str, Any] | None,
    out_dir: Path,
) -> dict[str, Any]:
    validation = validate_bundle(
        model=model,
        training=training,
        topology=topology,
        dataset=dataset,
        checkpoint=checkpoint,
    )
    validation.raise_for_errors()

    rank_map = build_rank_map(topology, training)
    comm_groups = build_comm_groups(training)
    layer_assignments = assign_layers(model, training)
    memory = build_memory_layout(model, training)
    op_registry = registry_metadata()

    compiler_inputs = {
        "model": model,
        "training": training,
        "topology": topology,
        "dataset": dataset,
        "checkpoint": checkpoint,
        "rank_map": rank_map,
        "comm_groups": comm_groups,
        "layer_assignments": layer_assignments,
        "memory": memory,
        "op_registry": op_registry,
    }
    plan_id = hashlib.sha256(canonical_json(compiler_inputs)).hexdigest()

    manifest = {
        "version": 1,
        "plan_id": plan_id,
        "op_registry_version": op_registry["version"],
        "op_registry_sha256": op_registry["sha256"],
        "world_size": required_world_size(training),
        "model": {
            "family": model["family"],
            "layers": model["layers"],
            "hidden_size": model["hidden_size"],
            "sequence_length": model["sequence_length"],
        },
        "training": {
            "precision": training["precision"],
            "microbatch_size": training["microbatch_size"],
            "gradient_accumulation_steps": training["gradient_accumulation_steps"],
            "parallelism": training["parallelism"],
        },
        "artifacts": {
            "rank_map": "rank-map.json",
            "comm_groups": "comm-groups.json",
            "layer_assignments": "layer-assignments.json",
            "memory_layout": "memory-layout.json",
            "validation_report": "validation-report.json",
            "rank_plans_dir": "ranks",
            "rank_binary_plans_dir": "ranks-bin",
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    ranks_dir = out_dir / "ranks"
    rank_binary_dir = out_dir / "ranks-bin"
    ranks_dir.mkdir(parents=True, exist_ok=True)
    rank_binary_dir.mkdir(parents=True, exist_ok=True)

    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "rank-map.json", rank_map)
    write_json(out_dir / "comm-groups.json", comm_groups)
    write_json(out_dir / "layer-assignments.json", {"version": 1, "assignments": layer_assignments})
    write_json(out_dir / "memory-layout.json", memory)
    write_json(out_dir / "validation-report.json", validation.to_dict())

    active_ranks = [rank for rank in rank_map["ranks"] if rank.get("active")]
    for rank in active_ranks:
        rank_plan = build_rank_plan(
            plan_id=plan_id,
            rank=rank,
            model=model,
            training=training,
            layer_assignments=layer_assignments,
            memory=memory,
            op_registry=op_registry,
        )
        write_json(ranks_dir / f"rank_{rank['global_rank']:06d}.json", rank_plan)
        write_binary_rank_plan(rank_binary_dir / f"rank_{rank['global_rank']:06d}.cairn", rank_plan)

    return manifest


def build_memory_layout(model: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    estimated = estimate_memory_bytes(model, training)
    budget = training["memory_budget_bytes_per_gpu"]
    segments = [
        ("params_shard", 0.24),
        ("gradients_shard", 0.24),
        ("optimizer_state_shard", 0.24),
        ("activation_ring", 0.16),
        ("comm_scratch", 0.06),
        ("attention_workspace", 0.04),
        ("rng_and_metrics", 0.02),
    ]
    offset = 0
    segment_output = []
    for name, fraction in segments:
        size = align(int(estimated * fraction), 256)
        segment_output.append({"name": name, "offset": offset, "nbytes": size})
        offset += size

    return {
        "version": 1,
        "estimated_bytes_per_rank": estimated,
        "budget_bytes_per_rank": budget,
        "max_live_bytes_by_rank": estimated,
        "segments": segment_output,
    }


def align(value: int, boundary: int) -> int:
    return ((value + boundary - 1) // boundary) * boundary


def build_rank_plan(
    *,
    plan_id: str,
    rank: dict[str, Any],
    model: dict[str, Any],
    training: dict[str, Any],
    layer_assignments: list[dict[str, Any]],
    memory: dict[str, Any],
    op_registry: dict[str, Any],
) -> dict[str, Any]:
    pp = rank["pipeline_index"]
    tensor = training["parallelism"]["tensor"]
    data = training["parallelism"]["data"]
    assignment = layer_assignments[pp]
    ops: list[dict[str, Any]] = []

    op_id = 0
    for layer in range(assignment["layer_start"], assignment["layer_end"]):
        ops.append(op(op_id, "rmsnorm", "compute_lo", layer=layer, bytes=model["hidden_size"] * 2))
        op_id += 1
        ops.append(op(op_id, "attention_fwd", "compute_hi", layer=layer, bytes=activation_bytes(model, training)))
        op_id += 1
        if tensor > 1:
            ops.append(op(op_id, "all_reduce", "comm_tp", group="tensor", layer=layer, bytes=activation_bytes(model, training)))
            op_id += 1
        ops.append(op(op_id, "mlp_fwd", "compute_hi", layer=layer, bytes=activation_bytes(model, training)))
        op_id += 1

    if training["parallelism"]["pipeline"] > 1:
        if pp < training["parallelism"]["pipeline"] - 1:
            ops.append(op(op_id, "pipe_send_activation", "comm_pp", group="pipeline", bytes=activation_bytes(model, training)))
            op_id += 1
        if pp > 0:
            ops.append(op(op_id, "pipe_recv_activation_grad", "comm_pp", group="pipeline", bytes=activation_bytes(model, training)))
            op_id += 1

    for layer in range(assignment["layer_end"] - 1, assignment["layer_start"] - 1, -1):
        ops.append(op(op_id, "mlp_bwd", "compute_hi", layer=layer, bytes=activation_bytes(model, training)))
        op_id += 1
        ops.append(op(op_id, "attention_bwd", "compute_hi", layer=layer, bytes=activation_bytes(model, training)))
        op_id += 1
        if data > 1:
            ops.append(op(op_id, "reduce_scatter", "comm_dp", group="data", layer=layer, bytes=gradient_bytes(model, training)))
            op_id += 1

    ops.append(op(op_id, "optimizer", "compute_lo", bytes=memory["estimated_bytes_per_rank"] // 3))
    op_id += 1
    ops.append(op(op_id, "checkpoint_stage", "io", bytes=memory["estimated_bytes_per_rank"] // 2))

    return {
        "version": 1,
        "plan_id": plan_id,
        "op_registry_version": op_registry["version"],
        "op_registry_sha256": op_registry["sha256"],
        "world_size": (
            training["parallelism"]["tensor"]
            * training["parallelism"]["pipeline"]
            * training["parallelism"]["data"]
            * training["parallelism"].get("context", 1)
            * training["parallelism"].get("expert", 1)
        ),
        "microbatch_size": training["microbatch_size"],
        "rank": rank,
        "streams": STREAMS,
        "layer_assignment": assignment,
        "memory": {
            "estimated_bytes_per_rank": memory["estimated_bytes_per_rank"],
            "segments": memory["segments"],
        },
        "ops": ops,
    }


def op(op_id: int, kind: str, stream: str, **kwargs: Any) -> dict[str, Any]:
    klass = validate_op(kind, stream)
    return {
        "op_id": op_id,
        "kind": kind,
        "op_class": klass,
        "stream": stream,
        **kwargs,
    }


def activation_bytes(model: dict[str, Any], training: dict[str, Any]) -> int:
    dtype_size = 1 if training["precision"] == "fp8" else 2
    return model["hidden_size"] * model["sequence_length"] * training["microbatch_size"] * dtype_size


def gradient_bytes(model: dict[str, Any], training: dict[str, Any]) -> int:
    dtype_size = 1 if training["precision"] == "fp8" else 2
    return model["hidden_size"] * model["hidden_size"] * dtype_size // max(1, training["parallelism"]["tensor"])


def write_binary_rank_plan(path: Path, rank_plan: dict[str, Any]) -> None:
    segments = rank_plan["memory"]["segments"]
    ops = rank_plan["ops"]
    arena_bytes = arena_bytes_from_segments(segments)
    chunks = [
        BINARY_PLAN_HEADER.pack(
            BINARY_PLAN_MAGIC,
            BINARY_PLAN_VERSION,
            rank_plan["op_registry_version"],
            fixed_bytes(rank_plan["op_registry_sha256"], 64),
            fixed_bytes(rank_plan["plan_id"], 64),
            rank_plan["world_size"],
            rank_plan["rank"]["global_rank"],
            rank_plan["microbatch_size"],
            len(ops),
            len(segments),
            rank_plan["memory"]["estimated_bytes_per_rank"],
            arena_bytes,
        )
    ]
    for segment in segments:
        chunks.append(
            BINARY_PLAN_SEGMENT.pack(
                fixed_bytes(segment["name"], 64),
                segment["offset"],
                segment["nbytes"],
            )
        )
    for item in ops:
        chunks.append(
            BINARY_PLAN_OP.pack(
                item["op_id"],
                OP_CLASS_IDS[item["op_class"]],
                fixed_bytes(item["kind"], 48),
                fixed_bytes(item["stream"], 32),
                item.get("bytes", 0),
            )
        )
    path.write_bytes(b"".join(chunks))


def fixed_bytes(value: str, size: int) -> bytes:
    encoded = value.encode("ascii")
    if len(encoded) > size:
        raise ValueError(f"value is too long for fixed field of {size} bytes: {value}")
    return encoded + (b"\0" * (size - len(encoded)))


def arena_bytes_from_segments(segments: list[dict[str, Any]]) -> int:
    arena_bytes = 0
    for segment in segments:
        arena_bytes = max(arena_bytes, segment["offset"] + segment["nbytes"])
    return arena_bytes
