# ADR 0001: Build the Static Plan Compiler Before the GPU Runtime

Status: accepted

## Context

Cairn targets fixed-shape, topology-aware AI training workloads. The long-term design includes a C runtime, static memory arenas, CUDA Graph replay, NCCL/NVSHMEM communication, checkpointing, and telemetry.

Building the GPU runtime first would create runtime complexity before the project can prove that plans are valid, deterministic, inspectable, and useful.

## Decision

Cairn will implement the static validation, rank mapping, plan compiler, and simulator before implementing the C GPU runtime.

The early product surface is:

- `cairn validate`
- `cairn map`
- `cairn compile`
- `cairn simulate`
- `cairn report`

## Consequences

Positive:

- Invalid distributed-training plans fail before expensive runtime work.
- Deterministic artifacts can be tested in CI.
- Runtime requirements become concrete through generated plans.
- Users can inspect memory, communication, and pipeline schedules early.

Negative:

- Cairn is not yet a trainer.
- Some runtime constraints will need adjustment once real GPU execution exists.
- The planner must avoid encoding assumptions that the C runtime cannot honor.

## Validation

This decision is validated by tests that check:

- valid and invalid bundles
- deterministic rank mapping
- deterministic plan compilation
- simulator output
- CLI behavior
