from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
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
            checkpoint = temp_path / "checkpoint.json"
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
            self.assertTrue(checkpoint.exists())
            checkpoint_data = load_json(checkpoint)
            self.assertEqual(checkpoint_data["step"], 1)
            self.assertEqual(checkpoint_data["ops_executed"], 17)
            self.assertGreater(checkpoint_data["communication_bytes"], 0)
            self.assertGreater(checkpoint_data["io_bytes"], 0)

            mismatch_result = subprocess.run(
                [str(binary), str(plan_dir / "ranks" / "rank_000000.json"), "7", str(temp_path / "bad.json")],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertNotEqual(mismatch_result.returncode, 0)
            self.assertIn("world_size does not match", mismatch_result.stderr)


if __name__ == "__main__":
    unittest.main()
