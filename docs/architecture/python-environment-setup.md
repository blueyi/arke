# Arke Python Environment Setup

This document defines the **one-click Python environment setup** for Arke so a brand-new machine can bootstrap the project consistently.

## Goals

- Support a **fresh developer environment** with one command
- Support **CPU-only**, **GPU/dev**, and **benchmark** installation profiles
- Keep the project editable (`pip install -e .`) for normal development
- Make the virtual environment path configurable
- Provide deterministic verification steps after installation

## Supported Profiles

Arke now provides a single bootstrap entry point:

```bash
scripts/bootstrap_env.sh [cpu-dev|gpu-dev|bench]
```

Profiles:

- `cpu-dev` — editable install with dev dependencies only
- `gpu-dev` — editable install with dev + GPU dependencies
- `bench` — editable install with dev + GPU + benchmark stack

## Quick Start

### Option A: Makefile shortcuts

```bash
make setup-cpu
make setup-gpu
make setup-bench
```

Default setup:

```bash
make setup
```

This maps to `setup-gpu`.

### Option B: Direct bootstrap script

```bash
scripts/bootstrap_env.sh cpu-dev
scripts/bootstrap_env.sh gpu-dev
scripts/bootstrap_env.sh bench
```

## Configure a Custom venv Path

By default, Arke uses:

```bash
.venv
```

You can override it with `ARKE_VENV`:

```bash
ARKE_VENV=~/.venvs/arke scripts/bootstrap_env.sh gpu-dev
```

Or with `make`:

```bash
make setup-gpu VENV=~/.venvs/arke
```

## Configure the Python Interpreter

By default, the bootstrap script uses `python3`.

Override it with `ARKE_PYTHON`:

```bash
ARKE_PYTHON=python3.10 scripts/bootstrap_env.sh gpu-dev
```

## What the Bootstrap Script Does

The bootstrap flow is:

1. Resolve repository root
2. Create a **fresh virtual environment**
3. Upgrade `pip`, `setuptools`, `wheel`
4. Install the selected dependency profile
5. Verify core imports
6. For GPU/bench profiles, attempt Torch/CUDA verification
7. Print recommended next commands

## Verification Commands

After setup:

```bash
source .venv/bin/activate
python -m pytest tests/test_parser.py -q
python -m benchmarks --layer L1
```

If you use a custom venv:

```bash
source ~/.venvs/arke/bin/activate
~/.venvs/arke/bin/python -m pytest tests/test_parser.py -q
```

## Dependency Sources

Arke dependency layers are split as follows:

- `pyproject.toml`
  - base runtime dependencies
  - optional `dev`
  - optional `gpu`
- `requirements-benchmark.txt`
  - benchmark-only extras such as baseline libraries

This keeps standard development installation lightweight while still supporting a full benchmark environment.

## Recommended Team Usage

### Normal development

```bash
make setup-cpu
```

### GPU/compiler development

```bash
make setup-gpu
```

### Benchmark reproduction / Stage evaluation

```bash
make setup-bench
```

## Notes

- The bootstrap script **recreates** the target venv path from scratch.
- If a benchmark dependency requires a platform-specific wheel or CUDA toolchain, install logs should be preserved for debugging.
- `bench` profile is the recommended baseline for Stage 7 / benchmark validation work.
