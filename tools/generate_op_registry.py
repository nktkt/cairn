#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "registry" / "op_registry.json"
C_HEADER_PATH = ROOT / "runtime" / "include" / "cairn" / "op_registry.h"
PY_MODULE_PATH = ROOT / "src" / "cairn" / "op_registry.py"


CLASS_TO_C = {
    "compute": "CAIRN_OP_CLASS_COMPUTE",
    "communication": "CAIRN_OP_CLASS_COMMUNICATION",
    "io": "CAIRN_OP_CLASS_IO",
}


def load_registry() -> dict[str, Any]:
    with REGISTRY_PATH.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    return registry


def validate_registry(registry: dict[str, Any]) -> None:
    if registry.get("version") != 1:
        raise SystemExit("registry.version must be 1")
    streams = registry.get("streams")
    ops = registry.get("ops")
    if not isinstance(streams, dict) or not streams:
        raise SystemExit("registry.streams must be a non-empty object")
    if not isinstance(ops, list) or not ops:
        raise SystemExit("registry.ops must be a non-empty list")

    seen: set[str] = set()
    for stream, stream_class in streams.items():
        if stream_class not in CLASS_TO_C:
            raise SystemExit(f"unsupported stream class for {stream}: {stream_class}")

    for op in ops:
        kind = op.get("kind")
        op_class = op.get("class")
        default_stream = op.get("default_stream")
        if not isinstance(kind, str) or not kind:
            raise SystemExit("op.kind must be a non-empty string")
        if kind in seen:
            raise SystemExit(f"duplicate op kind: {kind}")
        seen.add(kind)
        if op_class not in CLASS_TO_C:
            raise SystemExit(f"unsupported op class for {kind}: {op_class}")
        if default_stream not in streams:
            raise SystemExit(f"default stream for {kind} is not declared: {default_stream}")
        if streams[default_stream] != op_class:
            raise SystemExit(f"default stream class mismatch for {kind}")


def registry_hash(registry: dict[str, Any]) -> str:
    encoded = json.dumps(registry, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_c_header(registry: dict[str, Any]) -> str:
    digest = registry_hash(registry)
    lines = [
        "/* Generated from registry/op_registry.json. Do not edit by hand. */",
        "#ifndef CAIRN_OP_REGISTRY_H",
        "#define CAIRN_OP_REGISTRY_H",
        "",
        "#include <stddef.h>",
        "",
        f"#define CAIRN_OP_REGISTRY_VERSION {registry['version']}",
        f"#define CAIRN_OP_REGISTRY_SHA256 \"{digest}\"",
        "",
        "typedef enum {",
        "    CAIRN_OP_CLASS_COMPUTE = 0,",
        "    CAIRN_OP_CLASS_COMMUNICATION = 1,",
        "    CAIRN_OP_CLASS_IO = 2",
        "} cairn_op_class_t;",
        "",
        "typedef struct {",
        "    const char *kind;",
        "    cairn_op_class_t op_class;",
        "} cairn_executor_desc_t;",
        "",
        "static const cairn_executor_desc_t CAIRN_EXECUTORS[] = {",
    ]
    ops = registry["ops"]
    for index, op in enumerate(ops):
        suffix = "," if index < len(ops) - 1 else ""
        lines.append(f"    {{\"{op['kind']}\", {CLASS_TO_C[op['class']]}}}{suffix}")
    lines.extend(
        [
            "};",
            "",
            "static const size_t CAIRN_EXECUTOR_COUNT = sizeof(CAIRN_EXECUTORS) / sizeof(CAIRN_EXECUTORS[0]);",
            "",
            "#endif",
            "",
        ]
    )
    return "\n".join(lines)


def render_python_module(registry: dict[str, Any]) -> str:
    digest = registry_hash(registry)
    registry_literal = json.dumps(registry, indent=4, sort_keys=True)
    return f'''"""Generated from registry/op_registry.json. Do not edit by hand."""

from __future__ import annotations

from typing import Any


OP_REGISTRY: dict[str, Any] = {registry_literal}
OP_REGISTRY_VERSION = {registry["version"]}
OP_REGISTRY_SHA256 = "{digest}"


def registry_metadata() -> dict[str, Any]:
    return {{
        "version": OP_REGISTRY_VERSION,
        "sha256": OP_REGISTRY_SHA256,
    }}


def op_class(kind: str) -> str:
    for op in OP_REGISTRY["ops"]:
        if op["kind"] == kind:
            return op["class"]
    raise ValueError(f"unsupported op kind: {{kind}}")


def stream_class(stream: str) -> str:
    try:
        return OP_REGISTRY["streams"][stream]
    except KeyError as exc:
        raise ValueError(f"unsupported stream: {{stream}}") from exc


def validate_op(kind: str, stream: str) -> str:
    expected = op_class(kind)
    actual = stream_class(stream)
    if expected != actual:
        raise ValueError(f"op {{kind}} requires {{expected}} stream, got {{stream}}/{{actual}}")
    return expected


def supported_op_kinds() -> set[str]:
    return {{op["kind"] for op in OP_REGISTRY["ops"]}}
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="fail if generated files are stale")
    args = parser.parse_args()

    registry = load_registry()
    outputs = {
        C_HEADER_PATH: render_c_header(registry),
        PY_MODULE_PATH: render_python_module(registry),
    }

    if args.check:
        stale = []
        for path, expected in outputs.items():
            actual = path.read_text(encoding="utf-8") if path.exists() else ""
            if actual != expected:
                stale.append(str(path.relative_to(ROOT)))
        if stale:
            raise SystemExit("stale generated op registry files: " + ", ".join(stale))
        return 0

    for path, contents in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
