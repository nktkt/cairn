# Static AI Training Stack Design

This repository contains a design draft for a C-based, fixed-shape AI training stack intended for very large GPU clusters.

The project does not run frontier-scale training. Instead, it documents how to build a specialized training runtime that treats a known model, known tensor shapes, and a known cluster topology as a static execution target.

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

## Scope

This is a design repository, not a runnable trainer. The first practical implementation milestone would be a single-GPU C trainer with a static memory arena, fixed model math, checkpoint support, and trace output. Distributed execution should only follow after correctness is established against a reference implementation.

## License

No license has been selected yet.
