from __future__ import annotations

from typing import Any

from .validation import required_world_size


def build_physical_ranks(topology: dict[str, Any]) -> list[dict[str, Any]]:
    ranks: list[dict[str, Any]] = []
    rails = topology.get("rails", 1)

    for rack_index, rack in enumerate(topology["racks"]):
        gpu_in_rack = 0
        for tray_index, tray in enumerate(rack["trays"]):
            for gpu_in_tray in range(tray["gpus"]):
                rank = len(ranks)
                ranks.append(
                    {
                        "global_rank": rank,
                        "rack_index": rack_index,
                        "rack_id": rack["id"],
                        "tray_index": tray_index,
                        "tray_id": tray["id"],
                        "gpu_in_tray": gpu_in_tray,
                        "gpu_in_rack": gpu_in_rack,
                        "nic_id": gpu_in_rack,
                        "rail_id": rank % rails,
                        "failure_domain": rack["id"],
                    }
                )
                gpu_in_rack += 1
    return ranks


def logical_rank(dp: int, pp: int, tp: int, pipeline: int, tensor: int) -> int:
    return ((dp * pipeline) + pp) * tensor + tp


def build_rank_map(topology: dict[str, Any], training: dict[str, Any]) -> dict[str, Any]:
    physical = build_physical_ranks(topology)
    required = required_world_size(training)
    parallelism = training["parallelism"]
    tensor = parallelism["tensor"]
    pipeline = parallelism["pipeline"]
    data = parallelism["data"]

    ranks: list[dict[str, Any]] = []
    for dp in range(data):
        for pp in range(pipeline):
            for tp in range(tensor):
                logical = logical_rank(dp, pp, tp, pipeline, tensor)
                phys = physical[logical]
                ranks.append(
                    {
                        **phys,
                        "logical_rank": logical,
                        "data_index": dp,
                        "pipeline_index": pp,
                        "tensor_index": tp,
                        "active": True,
                    }
                )

    reserved = physical[required:]
    for rank in reserved:
        ranks.append({**rank, "active": False})

    return {
        "version": 1,
        "world_size": required,
        "physical_gpu_count": len(physical),
        "parallelism": {
            "tensor": tensor,
            "pipeline": pipeline,
            "data": data,
            "sequence": parallelism.get("sequence", 1),
            "context": parallelism.get("context", 1),
            "expert": parallelism.get("expert", 1),
        },
        "ranks": ranks,
    }


def build_comm_groups(training: dict[str, Any]) -> dict[str, Any]:
    p = training["parallelism"]
    tensor = p["tensor"]
    pipeline = p["pipeline"]
    data = p["data"]

    tp_groups = []
    pp_groups = []
    dp_groups = []

    for dp in range(data):
        for pp in range(pipeline):
            ranks = [logical_rank(dp, pp, tp, pipeline, tensor) for tp in range(tensor)]
            tp_groups.append({"kind": "tensor", "data_index": dp, "pipeline_index": pp, "ranks": ranks})

    for dp in range(data):
        for tp in range(tensor):
            ranks = [logical_rank(dp, pp, tp, pipeline, tensor) for pp in range(pipeline)]
            pp_groups.append({"kind": "pipeline", "data_index": dp, "tensor_index": tp, "ranks": ranks})

    for pp in range(pipeline):
        for tp in range(tensor):
            ranks = [logical_rank(dp, pp, tp, pipeline, tensor) for dp in range(data)]
            dp_groups.append({"kind": "data", "pipeline_index": pp, "tensor_index": tp, "ranks": ranks})

    return {
        "version": 1,
        "groups": {
            "tensor": tp_groups,
            "pipeline": pp_groups,
            "data": dp_groups,
        },
    }


def assign_layers(model: dict[str, Any], training: dict[str, Any]) -> list[dict[str, Any]]:
    layers = model["layers"]
    pipeline = training["parallelism"]["pipeline"]
    assignments: list[dict[str, Any]] = []
    base = layers // pipeline
    remainder = layers % pipeline
    start = 0
    for stage in range(pipeline):
        count = base + (1 if stage < remainder else 0)
        end = start + count
        assignments.append(
            {
                "pipeline_index": stage,
                "layer_start": start,
                "layer_end": end,
                "layer_count": count,
            }
        )
        start = end
    return assignments
