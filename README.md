# Cairn

This repository contains an early implementation and design draft for a C-based, fixed-shape AI training stack intended for very large GPU clusters.

The project does not run frontier-scale training yet. Today it provides a deterministic planning CLI that validates fixed training specs, maps ranks onto a topology, compiles inspectable per-rank plan artifacts, and simulates the resulting schedule.

## Core Idea

The stack is not meant to recreate JAX, PyTorch, XLA, or a general-purpose machine learning framework. Its goal is narrower:

- fixed transformer model family
- fixed tensor shapes
- fixed precision policy
- fixed optimizer
- fixed physical topology
- fixed parallelism plan
- fixed memory layout
- fixed communication schedule
- fixed checkpoint layout

The main thesis is that a dedicated runtime can remove dynamic framework decisions from the training hot path and map computation directly onto the physical machine.

## Architecture

The design is centered on two components:

1. An offline plan compiler that turns model, topology, parallelism, memory, and checkpoint inputs into rank-local execution plans.
2. A thin C runtime that executes those plans using static memory arenas, CUDA Graphs, cuBLASLt, cuDNN, NCCL, NVSHMEM, checkpoint I/O, and telemetry.

The runtime should not infer graph structure, allocate memory, compile kernels, or choose communication groups during steady-state training.

## Current Implementation

The current implementation includes:

- `cairn validate` for model, training, topology, dataset, and checkpoint specs
- `cairn map` for deterministic physical/logical rank mapping
- `cairn compile` for deterministic static plan generation
- `cairn simulate` for pipeline, memory, communication, checkpoint, and failure-injection reports
- `cairn report` for compact report summaries
- a C ABI/runtime dry-run skeleton that loads rank plans, validates memory segments, reserves arena metadata, rejects unsupported op kinds, executes op tables, records stats, and writes checkpoint manifests
- small example specs under [examples/small](examples/small)
- unit tests and GitHub Actions CI

## Quick Start

Run the test suite:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Validate the example bundle:

```bash
PYTHONPATH=src python -m cairn validate \
  --model examples/small/model.json \
  --training examples/small/training.json \
  --topology examples/small/topology.json \
  --dataset examples/small/dataset.json \
  --checkpoint examples/small/checkpoint.json
```

Compile and simulate a plan:

```bash
PYTHONPATH=src python -m cairn compile \
  --model examples/small/model.json \
  --training examples/small/training.json \
  --topology examples/small/topology.json \
  --dataset examples/small/dataset.json \
  --checkpoint examples/small/checkpoint.json \
  --out build/plan

PYTHONPATH=src python -m cairn simulate build/plan \
  --failure rank:3 \
  --out build/simulation-report.json

PYTHONPATH=src python -m cairn report build/simulation-report.json
```

Build and run the C runtime smoke binary:

```bash
cc -std=c11 -Wall -Wextra -Werror \
  -I runtime/include \
  runtime/src/cairn.c \
  runtime/examples/smoke.c \
  -o /tmp/cairn-runtime-smoke

/tmp/cairn-runtime-smoke \
  build/plan/ranks/rank_000000.json \
  8 \
  /tmp/cairn-checkpoints
```

The restore smoke binary verifies that a checkpoint root can be restored through `latest.json`:

```bash
cc -std=c11 -Wall -Wextra -Werror \
  -I runtime/include \
  runtime/src/cairn.c \
  runtime/examples/restore.c \
  -o /tmp/cairn-runtime-restore

/tmp/cairn-runtime-restore \
  build/plan/ranks/rank_000000.json \
  8 \
  /tmp/cairn-checkpoints
```

By default the runtime reserves arena metadata only. Set `CAIRN_ALLOCATE_HOST_ARENA=1` to allocate the full host arena for small local smoke plans.

The smoke runtime writes checkpoint artifacts in a shard-like layout:

```text
/tmp/cairn-checkpoints/
  latest.json
  step_000000001/
    manifest.json
    ranks/
      rank_000000.json
```

## What Is Included

The design covers:

- rank and topology mapping
- tensor, pipeline, data, sequence, context, and expert parallelism
- static op tables
- GPU memory arena planning
- kernel strategy
- NCCL and NVSHMEM communication
- static 1F1B pipeline scheduling
- offline plan compiler responsibilities
- dataset loading
- checkpoint and restart
- telemetry and profiling
- correctness and baseline comparison
- phased implementation milestones
- risk register

## Main Document

See the full design here:

- [docs/ai-training-stack-design.md](docs/ai-training-stack-design.md)

See the long-horizon product roadmap here:

- [ROADMAP.md](ROADMAP.md)

## Scope

This is a planner with a C runtime skeleton, not a GPU training runtime yet. The next practical implementation milestone is a single-GPU C trainer with a static memory arena, fixed model math, checkpoint support, and trace output. Distributed execution should only follow after correctness is established against a reference implementation.

## License

MIT. See [LICENSE](LICENSE).
