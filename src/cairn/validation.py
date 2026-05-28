from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationReport:
    status: str = "ok"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)
        self.status = "error"

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def extend(self, other: "ValidationReport") -> None:
        for message in other.errors:
            self.error(message)
        self.warnings.extend(other.warnings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            joined = "\n".join(f"- {message}" for message in self.errors)
            raise ValueError(f"validation failed:\n{joined}")


def _expect_int(report: ValidationReport, obj: dict[str, Any], key: str, *, minimum: int = 1) -> int | None:
    value = obj.get(key)
    if not isinstance(value, int):
        report.error(f"{key}: expected integer")
        return None
    if value < minimum:
        report.error(f"{key}: expected >= {minimum}, got {value}")
        return None
    return value


def validate_topology(topology: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()

    if topology.get("version") != 1:
        report.error("topology.version: expected 1")
    if not isinstance(topology.get("name"), str) or not topology["name"]:
        report.error("topology.name: expected non-empty string")

    racks = topology.get("racks")
    if not isinstance(racks, list) or not racks:
        report.error("topology.racks: expected non-empty list")
        return report

    seen_racks: set[str] = set()
    total_gpus = 0
    for rack_index, rack in enumerate(racks):
        if not isinstance(rack, dict):
            report.error(f"topology.racks[{rack_index}]: expected object")
            continue
        rack_id = rack.get("id")
        if not isinstance(rack_id, str) or not rack_id:
            report.error(f"topology.racks[{rack_index}].id: expected non-empty string")
        elif rack_id in seen_racks:
            report.error(f"topology.racks[{rack_index}].id: duplicate rack id {rack_id}")
        else:
            seen_racks.add(rack_id)

        trays = rack.get("trays")
        if not isinstance(trays, list) or not trays:
            report.error(f"topology.racks[{rack_index}].trays: expected non-empty list")
            continue

        seen_trays: set[str] = set()
        for tray_index, tray in enumerate(trays):
            if not isinstance(tray, dict):
                report.error(f"topology.racks[{rack_index}].trays[{tray_index}]: expected object")
                continue
            tray_id = tray.get("id")
            if not isinstance(tray_id, str) or not tray_id:
                report.error(f"topology.racks[{rack_index}].trays[{tray_index}].id: expected non-empty string")
            elif tray_id in seen_trays:
                report.error(f"topology.racks[{rack_index}].trays[{tray_index}].id: duplicate tray id {tray_id}")
            else:
                seen_trays.add(tray_id)
            gpus = tray.get("gpus")
            if not isinstance(gpus, int) or gpus < 1:
                report.error(f"topology.racks[{rack_index}].trays[{tray_index}].gpus: expected integer >= 1")
            else:
                total_gpus += gpus

    rails = topology.get("rails", 1)
    if not isinstance(rails, int) or rails < 1:
        report.error("topology.rails: expected integer >= 1")

    if total_gpus == 0:
        report.error("topology: no GPUs found")
    return report


def validate_model(model: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()

    if model.get("version") != 1:
        report.error("model.version: expected 1")
    if model.get("family") != "gpt":
        report.error("model.family: only 'gpt' is supported in this compiler MVP")

    layers = _expect_int(report, model, "layers")
    hidden = _expect_int(report, model, "hidden_size")
    heads = _expect_int(report, model, "attention_heads")
    _expect_int(report, model, "sequence_length")
    _expect_int(report, model, "vocab_size")
    ffn = _expect_int(report, model, "ffn_hidden_size")

    if hidden is not None and heads is not None and hidden % heads != 0:
        report.error("model.hidden_size must be divisible by model.attention_heads")
    if hidden is not None and ffn is not None and ffn < hidden:
        report.warn("model.ffn_hidden_size is smaller than hidden_size; this is unusual")
    if layers is not None and layers < 2:
        report.warn("model.layers < 2; useful only for smoke tests")

    return report


def validate_training(training: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()

    if training.get("version") != 1:
        report.error("training.version: expected 1")

    parallelism = training.get("parallelism")
    if not isinstance(parallelism, dict):
        report.error("training.parallelism: expected object")
        return report

    for key in ("tensor", "pipeline", "data"):
        value = parallelism.get(key)
        if not isinstance(value, int) or value < 1:
            report.error(f"training.parallelism.{key}: expected integer >= 1")
    for key in ("sequence", "context", "expert"):
        value = parallelism.get(key, 1)
        if not isinstance(value, int) or value < 1:
            report.error(f"training.parallelism.{key}: expected integer >= 1")

    _expect_int(report, training, "microbatch_size")
    _expect_int(report, training, "gradient_accumulation_steps")
    _expect_int(report, training, "memory_budget_bytes_per_gpu", minimum=1024)

    precision = training.get("precision")
    if precision not in ("bf16", "fp16", "fp8"):
        report.error("training.precision: expected one of bf16, fp16, fp8")

    optimizer = training.get("optimizer")
    if not isinstance(optimizer, dict):
        report.error("training.optimizer: expected object")
    elif optimizer.get("type") not in ("adamw", "sgd"):
        report.error("training.optimizer.type: expected adamw or sgd")

    return report


def validate_dataset(dataset: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()

    if dataset.get("version") != 1:
        report.error("dataset.version: expected 1")
    if not isinstance(dataset.get("name"), str) or not dataset["name"]:
        report.error("dataset.name: expected non-empty string")
    if dataset.get("format") != "fixed-token-binary":
        report.error("dataset.format: expected fixed-token-binary")
    if dataset.get("token_dtype", "uint32") not in ("uint32", "uint16", "uint8"):
        report.error("dataset.token_dtype: expected one of uint32, uint16, uint8")
    shards = dataset.get("shards")
    if not isinstance(shards, list) or not shards:
        report.error("dataset.shards: expected non-empty list")
    else:
        for index, shard in enumerate(shards):
            if not isinstance(shard, dict):
                report.error(f"dataset.shards[{index}]: expected object")
                continue
            if not isinstance(shard.get("path"), str) or not shard["path"]:
                report.error(f"dataset.shards[{index}].path: expected non-empty string")
            tokens = shard.get("tokens")
            if not isinstance(tokens, int) or tokens < 1:
                report.error(f"dataset.shards[{index}].tokens: expected integer >= 1")

    return report


def validate_checkpoint(checkpoint: dict[str, Any]) -> ValidationReport:
    report = ValidationReport()

    if checkpoint.get("version") != 1:
        report.error("checkpoint.version: expected 1")
    interval = checkpoint.get("interval_steps")
    if not isinstance(interval, int) or interval < 1:
        report.error("checkpoint.interval_steps: expected integer >= 1")
    local_path = checkpoint.get("local_path")
    if not isinstance(local_path, str) or not local_path:
        report.error("checkpoint.local_path: expected non-empty string")
    if "remote_path" in checkpoint and not isinstance(checkpoint["remote_path"], str):
        report.error("checkpoint.remote_path: expected string when present")

    return report


def validate_bundle(
    *,
    model: dict[str, Any],
    training: dict[str, Any],
    topology: dict[str, Any],
    dataset: dict[str, Any] | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> ValidationReport:
    report = ValidationReport()
    report.extend(validate_model(model))
    report.extend(validate_training(training))
    report.extend(validate_topology(topology))
    if dataset is not None:
        report.extend(validate_dataset(dataset))
    if checkpoint is not None:
        report.extend(validate_checkpoint(checkpoint))

    if report.errors:
        return report

    world_size = topology_gpu_count(topology)
    required = required_world_size(training)
    if world_size < required:
        report.error(f"topology has {world_size} GPUs but training parallelism requires {required}")
    elif world_size > required:
        report.warn(f"topology has {world_size} GPUs; only first {required} ranks are used by this plan")

    layers = model["layers"]
    pp = training["parallelism"]["pipeline"]
    if layers < pp:
        report.error(f"model has {layers} layers but pipeline parallelism requires {pp} stages")

    estimated_memory = estimate_memory_bytes(model, training)
    budget = training["memory_budget_bytes_per_gpu"]
    if estimated_memory > budget:
        report.error(
            f"estimated per-rank memory {estimated_memory} exceeds memory_budget_bytes_per_gpu {budget}"
        )

    return report


def topology_gpu_count(topology: dict[str, Any]) -> int:
    total = 0
    for rack in topology.get("racks", []):
        for tray in rack.get("trays", []):
            total += int(tray.get("gpus", 0))
    return total


def required_world_size(training: dict[str, Any]) -> int:
    parallelism = training["parallelism"]
    return (
        parallelism["tensor"]
        * parallelism["pipeline"]
        * parallelism["data"]
        * parallelism.get("context", 1)
        * parallelism.get("expert", 1)
    )


def dtype_size_bytes(precision: str) -> int:
    if precision == "fp8":
        return 1
    return 2


def estimate_memory_bytes(model: dict[str, Any], training: dict[str, Any]) -> int:
    dtype_size = dtype_size_bytes(training["precision"])
    hidden = model["hidden_size"]
    ffn = model["ffn_hidden_size"]
    layers = model["layers"]
    seq = model["sequence_length"]
    microbatch = training["microbatch_size"]
    tp = training["parallelism"]["tensor"]
    dp = training["parallelism"]["data"]

    qkv = 3 * hidden * hidden
    out = hidden * hidden
    mlp = 2 * hidden * ffn
    layer_params = qkv + out + mlp
    total_params = layers * layer_params + model["vocab_size"] * hidden

    param_shard = total_params * dtype_size // max(1, tp * dp)
    gradients = param_shard
    optimizer = param_shard * 4 if training["optimizer"]["type"] == "adamw" else param_shard
    activation = microbatch * seq * hidden * dtype_size * 6
    comm = hidden * seq * dtype_size * 2
    safety = 64 * 1024 * 1024
    return param_shard + gradients + optimizer + activation + comm + safety
