"""Generated from registry/op_registry.json. Do not edit by hand."""

from __future__ import annotations

from typing import Any


OP_REGISTRY: dict[str, Any] = {
    "ops": [
        {
            "class": "compute",
            "default_stream": "compute_lo",
            "kind": "rmsnorm"
        },
        {
            "class": "compute",
            "default_stream": "compute_hi",
            "kind": "attention_fwd"
        },
        {
            "class": "compute",
            "default_stream": "compute_hi",
            "kind": "attention_bwd"
        },
        {
            "class": "compute",
            "default_stream": "compute_hi",
            "kind": "mlp_fwd"
        },
        {
            "class": "compute",
            "default_stream": "compute_hi",
            "kind": "mlp_bwd"
        },
        {
            "class": "compute",
            "default_stream": "compute_lo",
            "kind": "optimizer"
        },
        {
            "class": "communication",
            "default_stream": "comm_tp",
            "kind": "all_reduce"
        },
        {
            "class": "communication",
            "default_stream": "comm_dp",
            "kind": "reduce_scatter"
        },
        {
            "class": "communication",
            "default_stream": "comm_dp",
            "kind": "all_gather"
        },
        {
            "class": "communication",
            "default_stream": "comm_pp",
            "kind": "pipe_send_activation"
        },
        {
            "class": "communication",
            "default_stream": "comm_pp",
            "kind": "pipe_recv_activation_grad"
        },
        {
            "class": "io",
            "default_stream": "io",
            "kind": "checkpoint_stage"
        }
    ],
    "streams": {
        "comm_dp": "communication",
        "comm_pp": "communication",
        "comm_tp": "communication",
        "compute_hi": "compute",
        "compute_lo": "compute",
        "io": "io"
    },
    "version": 1
}
OP_REGISTRY_VERSION = 1
OP_REGISTRY_SHA256 = "8fbcf67fae9818f861e337b822663cd501a6357c0fa00b854c6f9c1878241511"


def registry_metadata() -> dict[str, Any]:
    return {
        "version": OP_REGISTRY_VERSION,
        "sha256": OP_REGISTRY_SHA256,
    }


def op_class(kind: str) -> str:
    for op in OP_REGISTRY["ops"]:
        if op["kind"] == kind:
            return op["class"]
    raise ValueError(f"unsupported op kind: {kind}")


def stream_class(stream: str) -> str:
    try:
        return OP_REGISTRY["streams"][stream]
    except KeyError as exc:
        raise ValueError(f"unsupported stream: {stream}") from exc


def validate_op(kind: str, stream: str) -> str:
    expected = op_class(kind)
    actual = stream_class(stream)
    if expected != actual:
        raise ValueError(f"op {kind} requires {expected} stream, got {stream}/{actual}")
    return expected


def supported_op_kinds() -> set[str]:
    return {op["kind"] for op in OP_REGISTRY["ops"]}
