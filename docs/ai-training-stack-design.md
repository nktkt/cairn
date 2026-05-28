# Static AI Training Stack Design

Date: 2026-05-28
Status: design draft
Target: fixed-shape distributed training runtime for very large GPU clusters

## 1. Purpose

This document defines a practical design for building a C-based AI training stack without actually running frontier-scale training in this environment.

The goal is not to recreate JAX, PyTorch, XLA, or a general machine learning framework. The goal is to build a specialized training runtime that assumes:

- fixed model family
- fixed tensor shapes
- fixed precision policy
- fixed optimizer
- fixed cluster topology
- fixed parallelism plan
- fixed memory layout
- fixed communication schedule
- fixed checkpoint layout

The performance thesis is that a static, topology-aware trainer can beat a general framework for a narrow production workload by removing dynamic decisions from the hot path and by mapping the workload directly onto the physical machine.

## 2. Non-Goals

The first version must avoid broad framework work.

Non-goals:

- no general automatic differentiation engine
- no dynamic shape support
- no Python execution in the training hot path
- no generic operator dispatcher
- no user-facing NumPy-compatible API
- no automatic model import from arbitrary frameworks
- no attempt to replace cuBLASLt, cuDNN, NCCL, or NVSHMEM at the beginning
- no full elastic training in V1

V1 should be a dedicated training machine for one known transformer architecture and one known deployment topology.

## 3. Hardware Assumptions

The design uses the following planning assumptions. These must be revalidated against vendor documentation before implementation and deployment.

- GB300 NVL72-style rack-scale systems
- 72 GPUs per rack
- 36 Grace CPUs per rack
- rack-local NVLink/NVSwitch domain
- 800 Gb/s class scale-out network per GPU through ConnectX-8 class NICs
- 220,000 GPU target interpreted as 3,056 racks, or 220,032 addressable GPU ranks
- surplus 32 ranks reserved for spare, smoke test, or non-training control functions

The implementation must not hard-code product names throughout the runtime. Product-specific data belongs in the topology database and generated plan files.

## 4. Core Architecture

The stack has two major pieces:

1. Offline plan compiler
2. Thin C training runtime

```text
+-------------------------------------------------------------+
|                    offline plan compiler                   |
|                                                             |
| model spec + topology + parallel config + kernel catalog    |
|                         |                                   |
|                         v                                   |
| plan.rankNNNNNN.bin + tensor table + comm groups + layout   |
+-------------------------------------------------------------+
                          |
                          v
+-------------------------------------------------------------+
|                      C training runtime                     |
|                                                             |
| launcher | rank init | memory arena | scheduler | telemetry |
|                                                             |
| CUDA Driver API | CUDA Graphs | cuBLASLt | cuDNN | NCCL     |
| NVSHMEM | checkpoint I/O | dataset reader                  |
+-------------------------------------------------------------+
```

The plan compiler performs all expensive reasoning. The runtime executes a rank-local instruction stream.

Runtime rules:

- no graph compilation during training
- no dynamic memory allocation after initialization
- no rank placement decisions after initialization
- no collective shape inference at runtime
- no runtime model graph traversal
- no optimizer graph construction during training

## 5. Repository Layout

The implementation should use a layout like this:

```text
cai/
  include/
    cai.h
    cai_plan.h
    cai_topology.h
    cai_tensor.h
    cai_comm.h
    cai_checkpoint.h
    cai_telemetry.h

  src/
    runtime/
      main.c
      context.c
      rank.c
      scheduler.c
      cuda_driver.c
      cuda_graph.c

    topology/
      topology_db.c
      rank_map.c
      rail_map.c
      group_builder.c

    memory/
      arena.c
      tensor_table.c
      lifetime.c
      activation_pool.c
      comm_pool.c

    comm/
      nccl_groups.c
      nvshmem_runtime.c
      pipeline_p2p.c
      dp_shards.c
      moe_exchange.c

    model/
      gpt_forward.c
      gpt_backward.c
      optimizer.c
      loss.c

    io/
      dataset.c
      shard_reader.c
      checkpoint_writer.c
      checkpoint_reader.c
      manifest.c

    telemetry/
      event_trace.c
      gpu_metrics.c
      nic_metrics.c
      goodput.c

  kernels/
    rmsnorm.cu
    rope.cu
    swiglu.cu
    attention_fwd.cu
    attention_bwd.cu
    adamw.cu
    moe_dispatch.cu
    moe_combine.cu
    pack.cu

  tools/
    plan_compiler/
    topology_linter/
    kernel_bench/
    jax_compare/
    trace_viewer/

  tests/
    unit/
    integration/
    scale_sim/
```

The host runtime can be C. GPU kernels may be CUDA C++ compiled to cubin/PTX and loaded from C through the CUDA Driver API.

## 6. Public C ABI

The C ABI should be small and stable.

```c
typedef struct cai_context cai_context_t;
typedef struct cai_batch cai_batch_t;

typedef struct {
    uint32_t global_rank;
    uint32_t world_size;
    uint32_t local_rank;
    const char *plan_path;
    const char *topology_path;
    const char *checkpoint_path;
    const char *dataset_manifest_path;
} cai_init_desc_t;

int cai_init(cai_context_t **ctx, const cai_init_desc_t *desc);
int cai_load_plan(cai_context_t *ctx, const char *path);
int cai_load_checkpoint(cai_context_t *ctx, const char *path);
int cai_next_batch(cai_context_t *ctx, cai_batch_t *batch);
int cai_train_step(cai_context_t *ctx, const cai_batch_t *batch);
int cai_should_checkpoint(cai_context_t *ctx);
int cai_save_checkpoint(cai_context_t *ctx, const char *tag);
int cai_finalize(cai_context_t *ctx);
```

The runtime must return explicit error codes and record enough context for postmortem debugging. Fatal distributed mismatches should fail before the training loop starts.

## 7. Rank and Topology Model

The rank mapper represents physical topology separately from model parallel groups.

```c
#define CAI_GPUS_PER_RACK 72
#define CAI_TRAYS_PER_RACK 18
#define CAI_GPUS_PER_TRAY 4

typedef struct {
    uint32_t global_rank;
    uint16_t rack_id;
    uint8_t tray_id;
    uint8_t gpu_in_tray;
    uint8_t gpu_in_rack;
    uint8_t nic_id;
    uint8_t rail_id;
    uint16_t failure_domain;
} cai_phys_id_t;

static inline uint32_t cai_rank_of(uint16_t rack, uint8_t tray, uint8_t gpu) {
    return ((uint32_t)rack * CAI_GPUS_PER_RACK)
         + ((uint32_t)tray * CAI_GPUS_PER_TRAY)
         + gpu;
}
```

Topology hierarchy:

```text
L0: GPU
L1: tray, typically 4 GPUs
L2: rack, typically 72 GPUs
L3: pod, many racks
L4: full cluster
```

Placement policy:

- keep tensor parallel and sequence parallel inside a rack when possible
- keep latency-sensitive expert routing local to a rack or pod when possible
- use cross-rack links mainly for pipeline activation traffic, data-parallel reduce-scatter/all-gather, and MoE exchange
- build communicator groups from the resolved topology, not from contiguous rank IDs alone

## 8. Parallelism Strategy

The stack should support these parallel dimensions:

- TP: tensor parallel
- PP: pipeline parallel
- DP: data parallel
- SP: sequence parallel
- CP: context parallel, only when long context requires it
- EP: expert parallel, only for MoE models
- FSDP/ZeRO-style sharding for weights, gradients, and optimizer state

Basic sizing:

```text
total_gpus = TP * PP * DP * CP * EP_effective

global_batch =
    microbatch_size
  * gradient_accumulation_steps
  * DP
```

Pipeline bubble estimate:

```text
bubble_ratio = (PP - 1) / (num_microbatches + PP - 1)
```

The plan compiler must reject configurations where the expected bubble ratio, stage imbalance, or communication pressure exceeds configured limits.

## 9. Execution Model

Each rank is one process bound to one GPU.

```text
1 process = 1 global rank
1 rank = 1 CUDA context
1 rank = 1 static memory arena
1 rank = 1 precomputed schedule
1 rank = 1 telemetry ring buffer
```

Each rank owns separate streams:

```c
typedef struct {
    cudaStream_t compute_hi;
    cudaStream_t compute_lo;
    cudaStream_t comm_tp;
    cudaStream_t comm_pp;
    cudaStream_t comm_dp;
    cudaStream_t comm_ep;
    cudaStream_t io;
} cai_streams_t;
```

Target execution structure:

```text
graph_0: warmup forward pipeline
graph_1: steady 1F1B pipeline
graph_2: drain backward pipeline
graph_3: optimizer and sharded weight update
graph_4: asynchronous checkpoint staging
```

The steady-state training step should primarily replay CUDA Graphs or execute a compact static op table.

## 10. Static Op Table

The op table is the rank-local instruction stream produced by the plan compiler.

```c
typedef enum {
    CAI_OP_GEMM,
    CAI_OP_ATTENTION_FWD,
    CAI_OP_ATTENTION_BWD,
    CAI_OP_RMSNORM,
    CAI_OP_ROPE,
    CAI_OP_SWIGLU,
    CAI_OP_CROSS_ENTROPY,
    CAI_OP_REDUCE_SCATTER,
    CAI_OP_ALL_GATHER,
    CAI_OP_ALL_REDUCE,
    CAI_OP_PIPE_SEND,
    CAI_OP_PIPE_RECV,
    CAI_OP_MOE_DISPATCH,
    CAI_OP_MOE_COMBINE,
    CAI_OP_OPTIMIZER,
    CAI_OP_EVENT_RECORD,
    CAI_OP_EVENT_WAIT
} cai_op_kind_t;

typedef struct {
    uint16_t kind;
    uint16_t stream_id;
    uint32_t input0;
    uint32_t input1;
    uint32_t input2;
    uint32_t output0;
    uint32_t aux;
    uint64_t nbytes;
    uint32_t dep_first;
    uint32_t dep_count;
} cai_op_t;
```

The runtime must not infer dependencies. It only validates plan hashes, tensor bounds, communicator IDs, and stream handles before execution.

## 11. Memory Design

All GPU memory is allocated at initialization.

Arena segments:

- parameter shard
- gradient shard
- optimizer state shard
- activation ring buffer
- recomputation workspace
- pipeline send buffers
- pipeline receive buffers
- tensor-parallel collective scratch
- data-parallel collective scratch
- expert routing scratch
- attention workspace
- RNG state
- loss and metrics buffer

Tensor descriptors use arena offsets.

```c
typedef struct {
    uint64_t offset;
    uint64_t nbytes;
    uint32_t shape_id;
    uint16_t dtype;
    uint16_t flags;
} cai_tensor_desc_t;
```

Required memory optimizations:

- activation recomputation
- microbatch ring buffers
- static tensor lifetime analysis
- double or triple buffering for communication
- optimizer state sharding
- shared attention workspace
- no malloc/free during training

The plan compiler must produce a memory certificate:

```text
max_live_bytes_by_rank
arena_segment_offsets
tensor_lifetime_intervals
workspace_reuse_map
alignment_constraints
```

## 12. Kernel Strategy

V1 should use vendor libraries for hard primitives and hand-write only high-value glue kernels.

Use libraries first:

- large GEMM through cuBLASLt
- fused attention through cuDNN or an established attention kernel path
- synchronized collectives through NCCL
- GPU-initiated fine-grained communication through NVSHMEM
- CUDA Graphs for repeated schedules

Write custom kernels early:

- RMSNorm forward and backward
- RoPE
- SwiGLU or GeGLU pointwise fusion
- bias/dropout/add fusion when needed
- cross entropy
- AdamW or selected optimizer update
- activation pack/unpack
- MoE dispatch/combine
- pipeline transfer packing

Replace later only after profiling:

- QKV projection plus RoPE fusion
- attention megakernel
- MLP epilogue fusion
- optimizer plus gradient scaling fusion
- MoE exchange plus dispatch fusion
- GPU-triggered pipeline send/receive paths

## 13. Communication Design

Communication primitives:

```text
TP:
  all-reduce
  reduce-scatter
  all-gather

DP/FSDP:
  gradient reduce-scatter
  weight all-gather

PP:
  activation send/recv
  activation-gradient send/recv

EP/MoE:
  token dispatch all-to-all
  token combine all-to-all

metadata:
  small all-reduce
  broadcast
```

NCCL is the default for large synchronized collectives. NVSHMEM is reserved for communication patterns where GPU-initiated operations reduce CPU orchestration or launch overhead.

The plan compiler must verify:

- every collective has the same participants on every rank in the group
- count and datatype match across ranks
- communicator ordering cannot deadlock
- no rank participates in conflicting blocking operations
- pipeline send/receive pairs are balanced
- MoE token capacity and routing buffers are bounded

## 14. Pipeline Scheduler

V1 uses static 1F1B pipeline scheduling.

Each stage needs:

- previous stage rank group
- next stage rank group
- microbatch ID
- activation buffer ID
- gradient buffer ID
- parameter version

Conceptual schedule:

```c
for (int tick = 0; tick < total_ticks; ++tick) {
    if (should_run_forward(stage, tick)) {
        run_forward_microbatch(ctx, forward_mb(stage, tick));
        send_activation(ctx, next_stage, forward_mb(stage, tick));
    }

    if (should_run_backward(stage, tick)) {
        recv_activation_grad(ctx, next_stage, backward_mb(stage, tick));
        run_backward_microbatch(ctx, backward_mb(stage, tick));
        send_activation_grad(ctx, prev_stage, backward_mb(stage, tick));
    }

    progress_gradient_reduce_scatter(ctx);
}
```

The actual runtime should not execute this branchy loop. The plan compiler expands it into op tables and graph capture boundaries.

## 15. Offline Plan Compiler

Inputs:

```text
model.yaml
parallelism.yaml
topology.yaml or topology.bin
kernel_catalog.json
memory_budget.yaml
checkpoint_policy.yaml
dataset_manifest.json
```

Outputs:

```text
plan.rank000000.bin
plan.rank000001.bin
...
tensor_table.bin
shape_table.bin
comm_groups.bin
checkpoint_layout.bin
topology.resolved.bin
validation_report.json
```

Compiler responsibilities:

1. Parse model and training config.
2. Expand forward and backward graph.
3. Assign layers to pipeline stages.
4. Build tensor-parallel groups from topology.
5. Build data-parallel and sharding groups.
6. Place experts if MoE is enabled.
7. Select kernels and library calls.
8. Calculate tensor lifetimes.
9. Allocate memory arena offsets.
10. Generate stream/event dependencies.
11. Generate NCCL communicator groups.
12. Generate NVSHMEM symmetric heap layout.
13. Generate checkpoint shard layout.
14. Validate collective shapes across ranks.
15. Validate rank and topology hashes.
16. Emit per-rank op tables.

The compiler must be deterministic. Same inputs must produce byte-identical plans.

## 16. Dataset Pipeline

The training runtime should not depend on a Python data pipeline.

Dataset format:

```text
dataset/
  manifest.json
  index.bin
  shard_000000.bin
  shard_000001.bin
  ...
```

Data loader responsibilities:

- deterministic shard assignment
- mmap or direct I/O where appropriate
- pinned host prefetch buffers
- asynchronous host-to-device copy
- local NVMe cache support
- sample cursor checkpointing
- RNG seed checkpointing
- duplicate and gap detection

The dataset reader should be independent from the model runtime except for fixed batch shape and dtype.

## 17. Checkpoint Design

Checkpointing is a V1 feature. At 220k GPU scale, failures are normal operations.

Checkpoint layout:

```text
checkpoint/
  step_000123456/
    manifest.bin
    topology_hash
    plan_hash
    dataset_cursor/
    rng_state/
    params/
      rank_000000.bin
      rank_000001.bin
    optimizer/
      rank_000000.bin
      rank_000001.bin
```

Required properties:

- asynchronous checkpoint staging
- rank-local write first
- remote durable storage later
- atomic manifest publish
- two-phase checkpoint commit
- topology hash validation
- plan hash validation
- optimizer shard validation
- RNG and data cursor recovery
- spare-rank restoration path

V1 recovery policy:

```text
fault detected
  -> stop affected job group
  -> replace failed rank if spare exists
  -> reload latest complete checkpoint
  -> restart from known step
```

Full elastic training can wait until a later version.

## 18. Telemetry and Profiling

The stack must make performance claims measurable.

Required metrics:

- global tokens/sec
- tokens/sec/GPU
- training goodput
- model FLOPs utilization
- pipeline bubble ratio
- stage imbalance
- GPU idle percentage
- HBM bandwidth estimate
- NVLink utilization where available
- NIC utilization
- NCCL latency histogram
- NVSHMEM latency histogram
- checkpoint pause time
- data loader stall time
- failure recovery time
- loss equivalence against baseline

Rank-local trace event:

```c
typedef struct {
    uint64_t step;
    uint32_t rank;
    uint16_t op_id;
    uint16_t stream_id;
    uint64_t start_ns;
    uint64_t end_ns;
    uint64_t bytes;
    uint32_t flags;
} cai_trace_event_t;
```

Tracing must be low overhead. The runtime should write rank-local ring buffers and aggregate them by rack or pod out of band.

## 19. Baseline and Correctness

The comparison target must be fair.

Baseline requirements:

- same model
- same tokenizer and data
- same sequence length
- same precision policy
- same optimizer and schedule
- same global batch or matched convergence regime
- same checkpoint cadence
- same failure injection policy when measuring goodput

Correctness gates:

- forward output matches reference within dtype tolerance
- gradient matches reference on small models
- optimizer update matches reference
- loss curve equivalence on a small run
- deterministic checkpoint restore at the same step
- communicator mismatch tests fail before launch
- corrupted plan files fail validation

Performance gates:

- single GPU does not regress badly against a reference kernel stack
- one rack demonstrates correct TP and DP groups
- multi-rack run demonstrates PP overlap
- checkpoint does not dominate step time
- telemetry explains remaining stalls

## 20. Implementation Phases

### Phase A: Specification and Golden Reference

Deliverables:

- model spec
- optimizer spec
- precision spec
- dataset format
- reference implementation in JAX or PyTorch
- tolerance policy
- profiling report for baseline

Exit criteria:

- forward, backward, and optimizer math are fixed
- target shapes are fixed
- initial bottleneck hypothesis is measurable

### Phase B: Single-GPU C Trainer

Deliverables:

- C runtime skeleton
- static memory arena
- dataset reader
- checkpoint writer and reader
- selected custom kernels
- cuBLASLt or cuDNN integration
- CUDA Graph replay for steady step

Exit criteria:

- one-GPU loss matches reference
- no allocation in the training loop
- trace output shows per-op timing

### Phase C: Tray and Rack Scale

Deliverables:

- rank mapper
- topology linter
- NCCL communicator generation
- TP groups
- DP groups
- rack-local checkpoint shards

Exit criteria:

- 4, 8, and 72 GPU tests pass
- collective validation catches mismatches before execution
- rack-local TP beats or matches reference implementation for the target model

### Phase D: Multi-Rack Pipeline

Deliverables:

- pipeline send/recv
- static 1F1B schedule
- activation ring buffer
- cross-rack telemetry
- failure injection at rank and link level

Exit criteria:

- predicted and observed bubble ratios are close
- stage imbalance is visible
- communication overlaps compute
- checkpoint restore works after failure injection

### Phase E: Pod Scale

Deliverables:

- hierarchical launcher
- pod-local dataset cache
- pod-level telemetry aggregation
- gradient reduce-scatter/all-gather
- MoE routing if needed
- spare-rank restoration

Exit criteria:

- pod goodput improves over baseline
- failures recover from checkpoint
- stragglers are detected
- checkpoint staging does not dominate wall-clock

### Phase F: Full-Scale Plan Generation

Deliverables:

- full topology map
- per-rank plan files
- global validation report
- smoke test plan
- benchmark report template

Exit criteria:

- all plan hashes and topology hashes validate
- all collective groups pass static checks
- all memory budgets fit
- small-step smoke test completes
- benchmark report includes goodput, MFU, stalls, checkpoint, and recovery data

## 21. Risk Register

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Building a general framework by accident | V1 never ships | Keep the model, shapes, and optimizer fixed |
| Collective mismatch | Deadlock or corruption | Static group and shape validation |
| Pipeline stage imbalance | Poor utilization | Compiler-driven layer assignment and telemetry |
| Checkpoint too slow | Low wall-clock goodput | Async local staging and two-phase commit |
| Custom kernels underperform libraries | Wasted work | Use libraries first, replace only profiled bottlenecks |
| Fault recovery is bolted on late | Unusable at scale | Treat checkpoint/restart as V1 scope |
| Baseline comparison is unfair | Invalid performance claim | Define baseline protocol before optimization |
| Topology assumptions drift | Bad placement | Version topology data and validate hashes |

## 22. Minimal Runtime Skeleton

```c
int main(int argc, char **argv) {
    cai_init_desc_t desc = {
        .global_rank = read_env_u32("RANK"),
        .world_size = read_env_u32("WORLD_SIZE"),
        .local_rank = read_env_u32("LOCAL_RANK"),
        .plan_path = getenv("CAI_PLAN"),
        .topology_path = getenv("CAI_TOPOLOGY"),
        .checkpoint_path = getenv("CAI_CKPT"),
        .dataset_manifest_path = getenv("CAI_DATASET")
    };

    cai_context_t *ctx = NULL;

    if (cai_init(&ctx, &desc) != 0) {
        return 1;
    }

    if (cai_load_plan(ctx, desc.plan_path) != 0) {
        return 2;
    }

    if (desc.checkpoint_path != NULL) {
        if (cai_load_checkpoint(ctx, desc.checkpoint_path) != 0) {
            return 3;
        }
    }

    for (;;) {
        cai_batch_t batch;

        if (cai_next_batch(ctx, &batch) != 0) {
            break;
        }

        if (cai_train_step(ctx, &batch) != 0) {
            cai_save_checkpoint(ctx, "emergency");
            cai_finalize(ctx);
            return 4;
        }

        if (cai_should_checkpoint(ctx)) {
            if (cai_save_checkpoint(ctx, "periodic") != 0) {
                cai_finalize(ctx);
                return 5;
            }
        }
    }

    cai_finalize(ctx);
    return 0;
}
```

## 23. What V1 Should Prove

V1 should prove three things:

1. A fixed transformer training step can be executed by a static C runtime with no hot-path allocation or graph construction.
2. A topology-aware plan compiler can produce correct rank-local schedules for TP, PP, DP, and checkpointing.
3. The stack can measure goodput rigorously enough to justify or reject further low-level optimization.

The milestone is not "10x faster than JAX" as a generic claim. The milestone is a defensible report showing where the dedicated stack wins, where it does not, and what the next bottleneck is.

## 24. Reference Links

- NVIDIA GB300 NVL72: https://www.nvidia.com/en-us/data-center/gb300-nvl72/
- NVIDIA NVL72 AI Factory reference architecture: https://docs.nvidia.com/enterprise-reference-architectures/nvl72-ai-factory/latest/components.html
- NVIDIA NCCL documentation: https://docs.nvidia.com/deeplearning/nccl/user-guide/index.html
- NVIDIA NVSHMEM documentation: https://docs.nvidia.com/nvshmem/api/introduction.html
- CUDA Graphs programming guide: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- JAX distributed arrays and automatic parallelization: https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html
- JAX shard_map manual parallelism: https://docs.jax.dev/en/latest/notebooks/shard_map.html
