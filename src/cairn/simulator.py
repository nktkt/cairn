from __future__ import annotations

from pathlib import Path
from typing import Any

from .jsonutil import load_json, write_json


def simulate_plan(plan_dir: Path, *, failure: str | None = None, out_path: Path | None = None) -> dict[str, Any]:
    manifest = load_json(plan_dir / "manifest.json")
    memory = load_json(plan_dir / "memory-layout.json")
    layer_assignments = load_json(plan_dir / "layer-assignments.json")["assignments"]
    comm_groups = load_json(plan_dir / "comm-groups.json")

    p = manifest["training"]["parallelism"]
    pp = p["pipeline"]
    microbatches = manifest["training"]["gradient_accumulation_steps"]
    bubble = (pp - 1) / (microbatches + pp - 1) if pp > 1 else 0.0

    rank_files = sorted((plan_dir / "ranks").glob("rank_*.json"))
    op_counts: dict[str, int] = {}
    stream_counts: dict[str, int] = {}
    total_comm_bytes = 0
    total_checkpoint_bytes = 0
    tensor_count_by_rank: dict[str, int] = {}
    dependency_count_by_rank: dict[str, int] = {}
    for rank_file in rank_files:
        rank_plan = load_json(rank_file)
        tensor_count_by_rank[str(rank_plan["rank"]["global_rank"])] = len(rank_plan.get("tensors", []))
        dependency_count_by_rank[str(rank_plan["rank"]["global_rank"])] = sum(
            len(item.get("deps", [])) for item in rank_plan["ops"]
        )
        for item in rank_plan["ops"]:
            op_counts[item["kind"]] = op_counts.get(item["kind"], 0) + 1
            stream_counts[item["stream"]] = stream_counts.get(item["stream"], 0) + 1
            if item["stream"].startswith("comm"):
                total_comm_bytes += int(item.get("bytes", 0))
            if item["kind"] == "checkpoint_stage":
                total_checkpoint_bytes += int(item.get("bytes", 0))

    stage_layers = [stage["layer_count"] for stage in layer_assignments]
    max_layers = max(stage_layers) if stage_layers else 0
    min_layers = min(stage_layers) if stage_layers else 0
    imbalance = (max_layers - min_layers) / max_layers if max_layers else 0.0

    report = {
        "version": 1,
        "plan_id": manifest["plan_id"],
        "world_size": manifest["world_size"],
        "pipeline_bubble_ratio": round(bubble, 6),
        "pipeline_stage_imbalance": round(imbalance, 6),
        "stage_layer_counts": stage_layers,
        "op_counts": op_counts,
        "stream_counts": stream_counts,
        "tensor_count_by_rank": tensor_count_by_rank,
        "dependency_count_by_rank": dependency_count_by_rank,
        "memory_high_water_bytes_per_rank": memory["max_live_bytes_by_rank"],
        "total_communication_bytes_per_step": total_comm_bytes,
        "checkpoint_stage_bytes_per_step": total_checkpoint_bytes,
        "communicator_group_counts": {key: len(value) for key, value in comm_groups["groups"].items()},
        "failure_injection": simulate_failure(failure, manifest["world_size"]) if failure else None,
    }

    if out_path is not None:
        write_json(out_path, report)
    return report


def simulate_failure(failure: str, world_size: int) -> dict[str, Any]:
    if not failure.startswith("rank:"):
        return {
            "status": "unsupported",
            "input": failure,
            "message": "only rank:<id> failure injection is supported",
        }
    try:
        rank = int(failure.split(":", 1)[1])
    except ValueError:
        return {"status": "invalid", "input": failure, "message": "rank id is not an integer"}
    if rank < 0 or rank >= world_size:
        return {"status": "invalid", "input": failure, "message": "rank id is outside world size"}
    return {
        "status": "restart_required",
        "failed_rank": rank,
        "recovery_policy": "stop job group, restore latest complete checkpoint, restart with same plan_id",
    }
