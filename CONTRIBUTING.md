# Contributing to Cairn

Cairn is an early-stage compiler/runtime project. Contributions should preserve determinism, inspectability, and static validation.

## Development Setup

Run commands from the repository root:

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

## Contribution Rules

- Keep generated plans deterministic for identical inputs.
- Add tests for validators, compiler output, and simulator behavior.
- Do not add runtime performance claims without benchmark artifacts.
- Prefer explicit config fields over inferred behavior.
- Keep GPU runtime work behind stable plan artifacts and C ABI boundaries.
- Do not bypass validation in plugins, examples, or tests.

## Pull Request Checklist

- Unit tests pass.
- Example validation still passes.
- Documentation reflects user-facing behavior changes.
- Artifact format changes are versioned or explicitly documented.
- New claims are backed by test output, generated artifacts, or benchmark data.
