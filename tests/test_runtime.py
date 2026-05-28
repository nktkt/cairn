from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import json
from pathlib import Path

from cairn.compiler import compile_plan
from cairn.jsonutil import load_json


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "small"


def load_example(name: str) -> dict:
    return load_json(EXAMPLE / f"{name}.json")


@unittest.skipIf(shutil.which("cc") is None, "cc compiler is not available")
class RuntimeSmokeTests(unittest.TestCase):
    def test_c_runtime_smoke_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            plan_dir = temp_path / "plan"
            binary = temp_path / "runtime-smoke"
            restore_binary = temp_path / "runtime-restore"
            checkpoint = temp_path / "checkpoint"
            compile_plan(
                model=load_example("model"),
                training=load_example("training"),
                topology=load_example("topology"),
                dataset=load_example("dataset"),
                checkpoint=load_example("checkpoint"),
                out_dir=plan_dir,
            )
            compile_result = subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT / "runtime" / "include"),
                    str(ROOT / "runtime" / "src" / "cairn.c"),
                    str(ROOT / "runtime" / "examples" / "smoke.c"),
                    "-o",
                    str(binary),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            restore_compile_result = subprocess.run(
                [
                    "cc",
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-I",
                    str(ROOT / "runtime" / "include"),
                    str(ROOT / "runtime" / "src" / "cairn.c"),
                    str(ROOT / "runtime" / "examples" / "restore.c"),
                    "-o",
                    str(restore_binary),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(restore_compile_result.returncode, 0, restore_compile_result.stderr)

            run_result = subprocess.run(
                [str(binary), str(plan_dir / "ranks" / "rank_000000.json"), "8", str(checkpoint)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(run_result.returncode, 0, run_result.stderr)
            self.assertIn("cairn runtime smoke ok", run_result.stdout)
            self.assertIn("ops=17", run_result.stdout)
            self.assertIn("segments=7", run_result.stdout)
            self.assertTrue((checkpoint / "latest.json").exists())
            latest_data = load_json(checkpoint / "latest.json")
            self.assertTrue(latest_data["complete"])
            manifest_path = checkpoint / latest_data["manifest_path"]
            self.assertTrue(manifest_path.exists())
            manifest_data = load_json(manifest_path)
            self.assertTrue(manifest_data["complete"])
            self.assertEqual(manifest_data["rank_count"], 1)
            rank_shard = manifest_path.parent / manifest_data["rank_shard_path"]
            self.assertTrue(rank_shard.exists())
            checkpoint_data = load_json(rank_shard)
            self.assertEqual(checkpoint_data["step"], 1)
            self.assertEqual(checkpoint_data["ops_executed"], 17)
            self.assertEqual(checkpoint_data["memory_segment_count"], 7)
            self.assertGreaterEqual(checkpoint_data["arena_bytes"], checkpoint_data["estimated_memory_bytes"])
            self.assertGreater(checkpoint_data["compute_ops"], 0)
            self.assertGreater(checkpoint_data["communication_ops"], 0)
            self.assertGreater(checkpoint_data["io_ops"], 0)
            self.assertGreater(checkpoint_data["compute_bytes"], 0)
            self.assertGreater(checkpoint_data["communication_bytes"], 0)
            self.assertGreater(checkpoint_data["io_bytes"], 0)
            restore_result = subprocess.run(
                [str(restore_binary), str(plan_dir / "ranks" / "rank_000000.json"), "8", str(checkpoint)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(restore_result.returncode, 0, restore_result.stderr)
            self.assertIn("cairn restore smoke ok", restore_result.stdout)

            incomplete_checkpoint = temp_path / "incomplete-checkpoint"
            shutil.copytree(checkpoint, incomplete_checkpoint)
            incomplete_latest = load_json(incomplete_checkpoint / "latest.json")
            incomplete_latest["complete"] = False
            (incomplete_checkpoint / "latest.json").write_text(
                json.dumps(incomplete_latest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            incomplete_result = subprocess.run(
                [str(restore_binary), str(plan_dir / "ranks" / "rank_000000.json"), "8", str(incomplete_checkpoint)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(incomplete_result.returncode, 0)
            self.assertIn("incomplete", incomplete_result.stderr)

            incomplete_manifest_checkpoint = temp_path / "incomplete-manifest-checkpoint"
            shutil.copytree(checkpoint, incomplete_manifest_checkpoint)
            incomplete_manifest_latest = load_json(incomplete_manifest_checkpoint / "latest.json")
            incomplete_manifest_path = incomplete_manifest_checkpoint / incomplete_manifest_latest["manifest_path"]
            incomplete_manifest = load_json(incomplete_manifest_path)
            incomplete_manifest["complete"] = False
            incomplete_manifest_path.write_text(
                json.dumps(incomplete_manifest, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            incomplete_manifest_result = subprocess.run(
                [
                    str(restore_binary),
                    str(plan_dir / "ranks" / "rank_000000.json"),
                    "8",
                    str(incomplete_manifest_checkpoint),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(incomplete_manifest_result.returncode, 0)
            self.assertIn("incomplete", incomplete_manifest_result.stderr)

            mismatch_result = subprocess.run(
                [str(binary), str(plan_dir / "ranks" / "rank_000000.json"), "7", str(temp_path / "bad-checkpoint")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(mismatch_result.returncode, 0)
            self.assertIn("world_size does not match", mismatch_result.stderr)

            bad_plan = temp_path / "bad-rank-plan.json"
            bad_plan_data = load_json(plan_dir / "ranks" / "rank_000000.json")
            bad_plan_data["memory"]["segments"][1]["offset"] = 1
            bad_plan.write_text(json.dumps(bad_plan_data, indent=2, sort_keys=True), encoding="utf-8")
            bad_memory_result = subprocess.run(
                [str(binary), str(bad_plan), "8", str(temp_path / "bad-memory-checkpoint")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_memory_result.returncode, 0)
            self.assertIn("memory segments", bad_memory_result.stderr)

            bad_kind = temp_path / "bad-kind-plan.json"
            bad_kind_data = load_json(plan_dir / "ranks" / "rank_000000.json")
            bad_kind_data["ops"][0]["kind"] = "unknown_kernel"
            bad_kind.write_text(json.dumps(bad_kind_data, indent=2, sort_keys=True), encoding="utf-8")
            bad_kind_result = subprocess.run(
                [str(binary), str(bad_kind), "8", str(temp_path / "bad-kind-checkpoint")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_kind_result.returncode, 0)
            self.assertIn("op table", bad_kind_result.stderr)

            bad_stream = temp_path / "bad-stream-plan.json"
            bad_stream_data = load_json(plan_dir / "ranks" / "rank_000000.json")
            bad_stream_data["ops"][0]["stream"] = "comm_dp"
            bad_stream.write_text(json.dumps(bad_stream_data, indent=2, sort_keys=True), encoding="utf-8")
            bad_stream_result = subprocess.run(
                [str(binary), str(bad_stream), "8", str(temp_path / "bad-stream-checkpoint")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_stream_result.returncode, 0)
            self.assertIn("op table", bad_stream_result.stderr)

            bad_registry = temp_path / "bad-registry-plan.json"
            bad_registry_data = load_json(plan_dir / "ranks" / "rank_000000.json")
            bad_registry_data["op_registry_sha256"] = "0" * 64
            bad_registry.write_text(json.dumps(bad_registry_data, indent=2, sort_keys=True), encoding="utf-8")
            bad_registry_result = subprocess.run(
                [str(binary), str(bad_registry), "8", str(temp_path / "bad-registry-checkpoint")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(bad_registry_result.returncode, 0)
            self.assertIn("op registry hash", bad_registry_result.stderr)


if __name__ == "__main__":
    unittest.main()
