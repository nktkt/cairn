# ADR 0002: Deterministic Binary Rank Plan Artifact

## Status

Accepted

## Context

Cairn emits JSON rank plans that are useful for review, tests, and debugging. The C runtime also needs a compact artifact that it can load without depending on JSON parsing in the runtime hot path.

The binary artifact must remain deterministic, self-describing enough for compatibility checks, and tied to the shared op registry used by the compiler and runtime.

## Decision

The compiler emits two rank-plan forms:

- `ranks/rank_XXXXXX.json` for inspection and debugging.
- `ranks-bin/rank_XXXXXX.cairn` for C runtime loading.

The `.cairn` rank plan uses a fixed little-endian format. Format version 2 contains:

- magic bytes and binary format version
- op registry version and SHA-256
- plan ID
- world size, global rank, microbatch size
- op count and memory segment count
- tensor count, dependency reference count, and tensor reference count
- estimated memory bytes and arena bytes
- fixed-width memory segment records
- fixed-width tensor records
- fixed-width op records
- dependency reference records
- input/output tensor reference records

The C runtime rejects binary plans when the magic, binary version, op registry metadata, world size, global rank, memory layout, tensor placement, op class, stream, dependency graph, tensor references, or file size does not match expectations.

## Consequences

This gives Cairn a stable runtime-facing artifact while preserving readable JSON for humans. Future format changes must increment the binary format version and keep compatibility decisions explicit.
