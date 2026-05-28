from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cairn.compiler import compile_plan
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
            self.assertEqual(manifest["op_registry_version"], OP_REGISTRY_VERSION)
            self.assertEqual(manifest["op_registry_sha256"], OP_REGISTRY_SHA256)
            self.assertEqual(rank_plan["op_registry_sha256"], OP_REGISTRY_SHA256)
            kinds = {op["kind"] for op in rank_plan["ops"]}
            self.assertLessEqual(kinds, supported_op_kinds())
            self.assertTrue(all("op_class" in op for op in rank_plan["ops"]))

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
