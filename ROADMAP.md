# Cairn Roadmap

Status: long-horizon product roadmap
Audience: contributors, maintainers, infrastructure engineers, and future users

## Product Direction

Cairn should become a scalable product for building, validating, and operating static AI training plans on large GPU clusters.

The long-term product is not a general machine learning framework. Cairn should be a compiler-runtime system for known training workloads where the model, tensor shapes, topology, communication schedule, memory layout, and checkpoint layout can be planned ahead of time.

The product should grow in this order:

1. Reliable design and validation tools.
2. A small deterministic plan compiler.
3. A single-GPU and single-node runtime.
4. Rack-scale distributed execution.
5. Multi-rack orchestration and telemetry.
6. Production operations, failure recovery, and ecosystem integrations.

## North Star

Cairn should let an infrastructure team describe a fixed training workload and a physical GPU topology, then produce a validated, deterministic execution plan that can be inspected, simulated, benchmarked, and eventually executed by a thin C runtime.

The core product promise:

- plans are deterministic
- failures are explainable
- memory use is statically bounded
- communication groups are validated before execution
- performance claims are measured with goodput, not only peak throughput
- deployment scales by hierarchy: GPU, tray, rack, pod, cluster

## Product Principles

- Build the planner before the runtime becomes complex.
- Validate everything possible before launch.
- Treat checkpoint and restart as first-class product features.
- Optimize only after a baseline and trace prove the bottleneck.
- Keep the runtime small enough to audit.
- Prefer explicit schemas over implicit conventions.
- Make simulation and dry-run modes useful before real hardware is available.
- Scale by stable interfaces, not by expanding one monolithic binary.

## Target Users

Primary users:

- AI infrastructure engineers
- distributed training platform teams
- GPU cluster operators
- performance engineers

Secondary users:

- model training researchers who need predictable large-scale runs
- compiler/runtime engineers
- vendors validating topology-aware training patterns

Non-target users for early versions:

- casual Python users
- small-model experimentation workflows
- dynamic research notebooks
- arbitrary model training without fixed shapes

## Product Shape

Cairn should eventually be composed of these product layers:

```text
spec layer
  model.yaml
  training.yaml
  topology.yaml
  checkpoint.yaml
  dataset.yaml

compiler layer
  graph expansion
  parallelism placement
  memory planner
  communication planner
  validation engine
  per-rank plan emitter

simulation layer
  dry-run planner
  memory simulation
  communication schedule simulation
  failure injection
  performance estimates

runtime layer
  C ABI
  static memory arena
  CUDA Graph replay
  NCCL/NVSHMEM adapters
  checkpoint I/O
  telemetry events

operations layer
  launcher integration
  dashboards
  trace viewer
  run manifests
  restart tooling
  benchmark reports
```

Each layer should have stable file formats and command-line boundaries so the project can grow without turning into a single fragile tool.

## Phase 0: Product Foundation

Goal: turn the design into a maintainable public project.

Deliverables:

- public repository structure
- license decision
- contribution guide
- code of conduct if community contributions are expected
- architecture decision records
- roadmap and issue templates
- schema directory for future config files
- design examples for small, medium, and large topologies

Quality gates:

- every public document has a clear scope
- no undocumented claims of runtime performance
- repository can onboard a new contributor without private context
- naming is consistent across README, design, and future code

Product outcome:

- Cairn is understandable as a product, not only as a technical note.

## Phase 1: Schemas and Static Validation

Goal: make invalid plans fail before runtime exists.

Deliverables:

- versioned schemas for model, training, topology, dataset, and checkpoint specs
- topology linter
- rank mapping generator
- communicator group validator
- tensor shape validator
- memory budget checker
- deterministic plan ID and hash format
- human-readable validation reports

CLI shape:

```text
cairn validate topology.yaml
cairn validate training.yaml
cairn map --topology topology.yaml --parallelism parallelism.yaml
cairn report validation-report.json
```

Quality gates:

- schema changes are versioned
- invalid collective groups are rejected
- impossible memory budgets are rejected
- rank maps are deterministic
- reports explain what failed and where

Product outcome:

- Cairn is useful even before it can train, because it can catch expensive distributed-training mistakes early.

## Phase 2: Plan Compiler MVP

Goal: compile a small fixed transformer training workload into a rank-local static plan.

Deliverables:

- minimal model graph format
- forward and backward graph expansion for a fixed transformer block
- pipeline stage assignment
- tensor-parallel group assignment
- data-parallel group assignment
- static op table format
- tensor table format
- memory arena layout
- checkpoint layout
- deterministic binary and JSON debug outputs

CLI shape:

```text
cairn compile \
  --model model.yaml \
  --training training.yaml \
  --topology topology.yaml \
  --out build/plan
```

Quality gates:

- same inputs produce byte-identical plans
- every emitted tensor has a bounded lifetime
- every op references valid tensors and streams
- every collective has matching participants, dtype, and count
- debug output is readable enough for review

Product outcome:

- Cairn becomes a real compiler product with inspectable artifacts.

## Phase 3: Simulator and Trace Model

Goal: let users understand a plan before running it on expensive hardware.

Deliverables:

- op schedule simulator
- pipeline bubble estimator
- stage imbalance report
- memory pressure timeline
- communication volume report
- checkpoint write volume report
- failure injection model
- synthetic trace output

CLI shape:

```text
cairn simulate build/plan
cairn simulate build/plan --failure rank:1042
cairn trace view trace.jsonl
```

Quality gates:

- simulator catches deadlock-prone schedules
- predicted memory high-water marks match compiler output
- pipeline utilization is visible by stage
- simulation output is stable in CI

Product outcome:

- Cairn can be adopted as a planning and review tool before runtime deployment.

## Phase 4: Single-GPU Runtime

Goal: execute a fixed training step locally with static memory and trace output.

Deliverables:

- C runtime skeleton
- stable C ABI
- static GPU memory arena
- dataset reader for fixed binary shards
- checkpoint writer and reader
- CUDA Driver API loading path
- CUDA Graph capture or replay path
- baseline custom kernels for simple fused ops
- reference comparison against JAX or PyTorch

Quality gates:

- no allocation inside the steady training loop
- forward output matches reference within tolerance
- gradient output matches reference on small models
- optimizer update matches reference
- checkpoint restore reproduces the same next step
- trace output covers every op in the step

Product outcome:

- Cairn becomes executable, but remains narrow and auditable.

## Phase 5: Single-Node and Rack-Local Distributed Runtime

Goal: prove static planning across multiple GPUs before multi-rack complexity.

Deliverables:

- NCCL communicator initialization from compiler output
- tensor-parallel collectives
- data-parallel reduce-scatter and all-gather
- rack-local rank mapping
- multi-process launch scripts
- rack-local telemetry aggregation
- rack-local checkpoint shards

Quality gates:

- 2, 4, 8, and rack-sized simulations pass the same validation suite
- collective mismatch tests fail before runtime execution
- loss curve matches the single-GPU or reference baseline where comparable
- trace output shows compute and communication overlap
- checkpoint restore works across all ranks

Product outcome:

- Cairn can be tested on real multi-GPU systems without needing full cluster scale.

## Phase 6: Multi-Rack Pipeline Execution

Goal: support pipeline-parallel execution across racks or pods.

Deliverables:

- static 1F1B pipeline scheduler
- activation send/receive plan
- activation gradient send/receive plan
- microbatch ring buffers
- cross-rack communication telemetry
- pipeline stage balancing report
- restart from checkpoint after injected failure

Quality gates:

- observed pipeline bubble ratio is close to simulation
- stage imbalance is reported clearly
- send/receive pairs are statically balanced
- checkpoint and restart work after rank failure injection
- goodput report includes data loading, checkpointing, and recovery costs

Product outcome:

- Cairn graduates from a runtime experiment to a distributed training operations product.

## Phase 7: Product-Grade Operations

Goal: make Cairn usable by teams that operate clusters repeatedly.

Deliverables:

- run manifest format
- run directory standard
- resumable job launcher integration
- health checks
- telemetry exporters
- trace viewer
- benchmark report generator
- hardware compatibility matrix
- release notes and upgrade guide

Operational artifacts:

```text
runs/
  run_2026_05_28_001/
    manifest.json
    plan/
    logs/
    traces/
    checkpoints/
    reports/
```

Quality gates:

- failed runs leave enough evidence for postmortem analysis
- reports distinguish peak throughput from training goodput
- plan, topology, checkpoint, and binary versions are recorded together
- users can reproduce a run configuration from artifacts

Product outcome:

- Cairn becomes repeatable infrastructure, not a one-off training script.

## Phase 8: Extensibility and Ecosystem

Goal: allow controlled expansion without losing determinism.

Deliverables:

- plugin boundary for topology providers
- plugin boundary for kernel catalogs
- plugin boundary for launcher integrations
- model-family adapters for a small number of fixed architectures
- import tools from restricted reference formats
- public compatibility tests
- stable artifact versioning

Expansion policy:

- add one model family at a time
- add one topology class at a time
- add one runtime backend at a time
- require deterministic plans for every supported combination
- require reference tests before performance work

Quality gates:

- plugins cannot bypass validation
- artifact formats remain backward compatible within a major version
- unsupported model shapes fail explicitly
- compatibility matrix is published for every release

Product outcome:

- Cairn can grow as a product while preserving its core premise: static, inspectable, validated execution.

## Phase 9: Enterprise and Large-Scale Readiness

Goal: support teams that need governance, reproducibility, and operational controls.

Deliverables:

- signed plan artifacts
- signed release artifacts
- SBOM generation
- audit logs for plan compilation
- policy checks for topology and checkpoint locations
- secrets-free run manifests
- role-aware operational tooling
- long-term support release branch policy

Quality gates:

- production artifacts can be verified
- release provenance is inspectable
- plan compilation is reproducible
- security-sensitive data is not written to traces or reports
- upgrades have documented migration paths

Product outcome:

- Cairn becomes credible as infrastructure software in serious environments.

## Phase 10: Frontier-Scale Validation

Goal: validate the architecture at the largest available scale without relying on unmeasured claims.

Deliverables:

- hierarchical topology planner
- pod-level plan compilation
- full-cluster dry-run validation
- synthetic full-scale schedule simulation
- staged rollout procedure
- benchmark methodology
- failure recovery report
- cost and goodput analysis

Quality gates:

- full plans compile deterministically
- communication groups pass static validation
- memory budgets fit on every rank class
- checkpoint/restart is measured
- goodput includes failure recovery and checkpoint overhead
- baseline comparison uses the same model, data, precision, and convergence target

Product outcome:

- Cairn can make defensible scale claims backed by data.

## Cross-Cutting Tracks

### Correctness

The correctness track must move ahead of performance work.

Milestones:

- reference math tests
- deterministic plan tests
- tensor lifetime tests
- collective validation tests
- checkpoint restore tests
- failure injection tests
- numerical tolerance policy

### Performance

Performance work should start only after traces identify the bottleneck.

Milestones:

- single-step trace format
- op-level timing
- communication histograms
- memory high-water reports
- pipeline utilization reports
- goodput report template
- baseline comparison protocol

### Developer Experience

Cairn needs a narrow but polished interface.

Milestones:

- consistent CLI
- readable validation errors
- examples that run without special hardware
- dry-run mode
- compact debug artifacts
- contributor documentation
- architecture decision records

### Operations

Operations should be designed early because large-scale training fails in operational details.

Milestones:

- run manifests
- restart tooling
- health checks
- log and trace retention policy
- dashboard exports
- postmortem templates
- compatibility matrix

### Security and Governance

Security should be added before broad adoption.

Milestones:

- license selection
- dependency policy
- SBOM
- signed artifacts
- secret redaction
- vulnerability reporting policy
- release provenance

## Product Metrics

Cairn should track product metrics separately from training performance.

Adoption metrics:

- number of valid example plans
- number of supported topology classes
- number of supported model families
- number of reproducible benchmark reports
- time to diagnose validation failure

Runtime metrics:

- tokens/sec/GPU
- global tokens/sec
- training goodput
- model FLOPs utilization
- pipeline bubble ratio
- stage imbalance
- checkpoint pause time
- recovery time objective

Quality metrics:

- test coverage for validators
- deterministic output checks
- compatibility matrix pass rate
- artifact format migration success
- documented known limitations

## Near-Term Backlog

The next concrete work should be:

1. Select a license.
2. Add contribution and issue templates.
3. Create `schemas/` with draft YAML schemas.
4. Add `examples/small/` with a toy topology and fixed transformer config.
5. Implement a topology linter.
6. Implement deterministic rank mapping.
7. Emit a JSON validation report.
8. Add CI for schema and example validation.
9. Create the first architecture decision record.
10. Define the binary plan artifact format.

## Strategic Non-Goals

These should remain out of scope until the core product is stable:

- arbitrary dynamic model graphs
- full Python framework compatibility
- general-purpose automatic differentiation
- handwritten replacements for all vendor libraries
- cloud-hosted control plane before local tooling is mature
- broad model zoo support before one model family is deeply reliable
- performance marketing before benchmark methodology is fixed

## Release Strategy

Suggested release ladder:

```text
v0.1  documentation, schemas, examples
v0.2  topology validation and rank mapping
v0.3  plan compiler MVP
v0.4  simulator and report generation
v0.5  single-GPU runtime prototype
v0.6  multi-GPU local runtime
v0.7  rack-scale execution path
v0.8  checkpoint/restart hardening
v0.9  production operations preview
v1.0  stable schemas, stable artifacts, validated runtime path
```

Version 1.0 should mean that Cairn can compile, validate, execute, checkpoint, restore, and report on a fixed supported workload with documented limitations. It should not mean broad framework parity.

## Final Direction

Cairn should scale as a product by making the hard parts explicit:

- what model is supported
- what topology is supported
- what plan was generated
- what memory is required
- what communication happens
- what checkpoint can restore
- what performance was measured

The winning product shape is a deterministic compiler-runtime system with strong validation and operational evidence. That is the path from design document to scalable infrastructure product.
