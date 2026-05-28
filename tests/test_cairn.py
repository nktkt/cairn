from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cairn.compiler import (
    BINARY_PLAN_HEADER,
    BINARY_PLAN_MAGIC,
    BINARY_PLAN_OP,
    BINARY_PLAN_REF,
    BINARY_PLAN_SEGMENT,
    BINARY_PLAN_TENSOR,
    BINARY_PLAN_VERSION,
    compile_plan,
)
from cairn.jsonutil import load_json
from cairn.mapping import build_comm_groups, build_rank_map
from cairn.op_registry import OP_REGISTRY_SHA256, OP_REGISTRY_VERSION, supported_op_kinds
from cairn.simulator import simulate_plan
from cairn.validation import validate_bundle


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "small"


def load_example(name: str) -> dict:
    return load_json(EXAMPLE / f"{name}.json")


def digest_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(item.read_bytes())
    return digest.hexdigest()


class ValidationTests(unittest.TestCase):
    def test_small_bundle_validates(self) -> None:
        report = validate_bundle(
            model=load_example("model"),
            training=load_example("training"),
            topology=load_example("topology"),
            dataset=load_example("dataset"),
            checkpoint=load_example("checkpoint"),
        )
        self.assertEqual(report.status, "ok")
        self.assertEqual(report.errors, [])

    def test_invalid_world_size_fails(self) -> None:
        topology = load_example("topology")
        topology["racks"][0]["trays"][1]["gpus"] = 1
        report = validate_bundle(
            model=load_example("model"),
            training=load_example("training"),
            topology=topology,
        )
        self.assertEqual(report.status, "error")
        self.assertTrue(any("parallelism requires" in message for message in report.errors))


class MappingTests(unittest.TestCase):
    def test_rank_map_and_groups_are_deterministic(self) -> None:
        topology = load_example("topology")
        training = load_example("training")
        self.assertEqual(build_rank_map(topology, training), build_rank_map(topology, training))
        groups = build_comm_groups(training)
        self.assertEqual(len(groups["groups"]["tensor"]), 4)
        self.assertEqual(len(groups["groups"]["pipeline"]), 4)
        self.assertEqual(len(groups["groups"]["data"]), 4)


class CompileTests(unittest.TestCase):
    def test_op_registry_generated_files_are_current(self) -> None:
        result = subprocess.run(
            [sys.executable, "tools/generate_op_registry.py", "--check"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)

    def test_compile_is_deterministic(self) -> None:
        model = load_example("model")
        training = load_example("training")
        topology = load_example("topology")
        dataset = load_example("dataset")
        checkpoint = load_example("checkpoint")
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            compile_plan(
                model=model,
                training=training,
                topology=topology,
                dataset=dataset,
                checkpoint=checkpoint,
                out_dir=Path(a),
            )
            compile_plan(
                model=model,
                training=training,
                topology=topology,
                dataset=dataset,
                checkpoint=checkpoint,
                out_dir=Path(b),
            )
            self.assertEqual(digest_tree(Path(a)), digest_tree(Path(b)))

    def test_compiled_plan_records_op_registry(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            compile_plan(
                model=load_example("model"),
                training=load_example("training"),
                topology=load_example("topology"),
                dataset=load_example("dataset"),
                checkpoint=load_example("checkpoint"),
                out_dir=Path(out),
            )
            manifest = load_json(Path(out) / "manifest.json")
            rank_plan = load_json(Path(out) / "ranks" / "rank_000000.json")
            binary_plan = Path(out) / "ranks-bin" / "rank_000000.cairn"
            self.assertEqual(manifest["op_registry_version"], OP_REGISTRY_VERSION)
            self.assertEqual(manifest["op_registry_sha256"], OP_REGISTRY_SHA256)
            self.assertEqual(manifest["artifacts"]["rank_binary_plans_dir"], "ranks-bin")
            self.assertEqual(rank_plan["op_registry_sha256"], OP_REGISTRY_SHA256)
            kinds = {op["kind"] for op in rank_plan["ops"]}
            self.assertLessEqual(kinds, supported_op_kinds())
            self.assertTrue(all("op_class" in op for op in rank_plan["ops"]))
            self.assertTrue(binary_plan.exists())
            tensors_by_name = {tensor["name"]: tensor for tensor in rank_plan["tensors"]}
            self.assertIn("input_tokens", tensors_by_name)
            self.assertIn("activation_slot_0", tensors_by_name)
            input_tokens = tensors_by_name["input_tokens"]
            activation_slot = tensors_by_name["activation_slot_0"]
            self.assertLessEqual(input_tokens["offset"] + input_tokens["nbytes"], activation_slot["offset"])
            binary_header = BINARY_PLAN_HEADER.unpack(binary_plan.read_bytes()[: BINARY_PLAN_HEADER.size])
            (
                magic,
                binary_version,
                registry_version,
                registry_sha,
                plan_id,
                world_size,
                global_rank,
                microbatch_size,
                op_count,
                segment_count,
                tensor_count,
                dependency_count,
                tensor_ref_count,
                estimated_memory_bytes,
                arena_bytes,
            ) = binary_header
            self.assertEqual(magic, BINARY_PLAN_MAGIC)
            self.assertEqual(binary_version, BINARY_PLAN_VERSION)
            self.assertEqual(registry_version, OP_REGISTRY_VERSION)
            self.assertEqual(registry_sha.decode("ascii").rstrip("\0"), OP_REGISTRY_SHA256)
            self.assertEqual(plan_id.decode("ascii").rstrip("\0"), manifest["plan_id"])
            self.assertEqual(world_size, manifest["world_size"])
            self.assertEqual(global_rank, 0)
            self.assertEqual(microbatch_size, rank_plan["microbatch_size"])
            self.assertEqual(op_count, len(rank_plan["ops"]))
            self.assertEqual(segment_count, len(rank_plan["memory"]["segments"]))
            self.assertEqual(tensor_count, len(rank_plan["tensors"]))
            self.assertEqual(dependency_count, sum(len(op["deps"]) for op in rank_plan["ops"]))
            self.assertEqual(
                tensor_ref_count,
                sum(len(op["input_tensors"]) + len(op["output_tensors"]) for op in rank_plan["ops"]),
            )
            self.assertEqual(estimated_memory_bytes, rank_plan["memory"]["estimated_bytes_per_rank"])
            self.assertGreaterEqual(arena_bytes, estimated_memory_bytes)
            self.assertEqual(rank_plan["ops"][0]["deps"], [])
            self.assertTrue(all(op["tick"] == op["op_id"] for op in rank_plan["ops"]))
            self.assertTrue(all(op["input_tensors"] for op in rank_plan["ops"]))
            self.assertTrue(all(op["output_tensors"] for op in rank_plan["ops"] if op["op_class"] != "io"))
            self.assertEqual(
                binary_plan.stat().st_size,
                BINARY_PLAN_HEADER.size
                + (segment_count * BINARY_PLAN_SEGMENT.size)
                + (tensor_count * BINARY_PLAN_TENSOR.size)
                + (op_count * BINARY_PLAN_OP.size)
                + ((dependency_count + tensor_ref_count) * BINARY_PLAN_REF.size),
            )

    def test_simulator_reports_pipeline_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            compile_plan(
                model=load_example("model"),
                training=load_example("training"),
                topology=load_example("topology"),
                dataset=load_example("dataset"),
                checkpoint=load_example("checkpoint"),
                out_dir=Path(out),
            )
            report = simulate_plan(Path(out), failure="rank:3")
            self.assertEqual(report["world_size"], 8)
            self.assertGreater(report["pipeline_bubble_ratio"], 0)
            self.assertEqual(report["tensor_count_by_rank"]["0"], 9)
            self.assertEqual(report["dependency_count_by_rank"]["0"], 16)
            self.assertEqual(report["failure_injection"]["status"], "restart_required")


class CliTests(unittest.TestCase):
    def test_cli_validate_and_compile(self) -> None:
        with tempfile.TemporaryDirectory() as out:
            validation = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cairn",
                    "validate",
                    "--model",
                    str(EXAMPLE / "model.json"),
                    "--training",
                    str(EXAMPLE / "training.json"),
                    "--topology",
                    str(EXAMPLE / "topology.json"),
                    "--dataset",
                    str(EXAMPLE / "dataset.json"),
                    "--checkpoint",
                    str(EXAMPLE / "checkpoint.json"),
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(validation.returncode, 0, validation.stderr + validation.stdout)

            compiled = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cairn",
                    "compile",
                    "--model",
                    str(EXAMPLE / "model.json"),
                    "--training",
                    str(EXAMPLE / "training.json"),
                    "--topology",
                    str(EXAMPLE / "topology.json"),
                    "--dataset",
                    str(EXAMPLE / "dataset.json"),
                    "--checkpoint",
                    str(EXAMPLE / "checkpoint.json"),
                    "--out",
                    out,
                ],
                cwd=ROOT,
                env={"PYTHONPATH": str(ROOT / "src")},
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(compiled.returncode, 0, compiled.stderr + compiled.stdout)
            manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["world_size"], 8)


if __name__ == "__main__":
    unittest.main()
