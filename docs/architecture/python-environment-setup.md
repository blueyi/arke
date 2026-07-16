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
- `mlir-gpu` — editable install with dev + GPU + MLIR GPU dependencies (Phase 3)
- `bench` — editable install with dev + GPU + benchmark stack

## Quick Start

### Option A: Makefile shortcuts

```bash
make setup-cpu
make setup-gpu
make setup-mlir
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

If the user does **not** specify a virtual environment path, Arke automatically creates the environment in the **project root** as:

```bash
.venv
```

This path is the default for both:

- `scripts/bootstrap_env.sh`
- `make setup-*`

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

## Reusing a Pre-existing venv

If a usable virtual environment already exists on the machine (for example
a shared one at `~/.venvs/arke` previously created by `make setup-gpu` or
`make setup-bench`), you do **not** need to re-run the bootstrap script.
You can use it directly:

```bash
~/.venvs/arke/bin/python -c "import arke, torch; print(torch.__version__, torch.cuda.is_available())"
~/.venvs/arke/bin/python -m pytest tests/ -q
ARKE_GPU_TESTS=1 ~/.venvs/arke/bin/python -m pytest tests/test_benchmark_correctness_probe.py -q
```

Only re-bootstrap (`make setup-gpu VENV=~/.venvs/arke`) if imports fail
or a required dependency (e.g. `torch`, `triton`, `arke` editable install)
is missing or outdated. This is the recommended default for day-to-day
development and CI-style local runs on machines that already have the
environment provisioned.

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
  - optional `mlir-gpu` (Phase 3 MLIR GPU path)
  - optional `agent` (live-LLM optimization driver)
- `requirements-benchmark.txt`
  - benchmark-only extras such as baseline libraries

This keeps standard development installation lightweight while still supporting a full benchmark environment.

## MLIR & LLVM Toolchain Setup (Phase 3 + Phase 5)

Phase 3 (Arke → MLIR Dialect) and Phase 5 (Arke → LLVM IR) require the **LLVM 20** toolchain. This provides:
- **Phase 3:** `mlir-opt`, `mlir-translate`, `mlir-runner` for MLIR GPU kernel compilation
- **Phase 5:** `llc` (LLVM IR → PTX), plus `ptxas` (PTX → cubin) from the CUDA toolkit

The toolchain is installed **user-local** — no root privileges needed.

### LLVM Version Policy

Arke defaults to **LLVM 20** (aligned with MLIR 20 / Triton 3.2 / PyTorch 2.6). This is enforced at three levels:

| Level | Mechanism | Override |
|---|---|---|
| Bootstrap | `ARKE_LLVM_VERSION` env var | `ARKE_LLVM_VERSION=18 make setup-gpu` |
| Runtime | `ARKE_LLC` env var | `ARKE_LLC=/path/to/llc-20 arke run --backend llvm ...` |
| Code | `_find_llc()` priority chain | `ARKE_LLC` → `MLIR_HOME/bin/llc` → `~/opt/mlir20` → PATH |

> **⚠️ Do NOT use system LLVM 18 for Arke.** The nvptx64 codegen in LLVM 18 may produce PTX incompatible with the CUDA 13.x ptxas, and MLIR 20 dialect compatibility requires LLVM 20.

### Prerequisites

- NVIDIA GPU with CUDA 12.x+
- Python `cuda-python` package (installed by `pip install -e ".[mlir-gpu]"`)
- LLVM 20 CLI tools: `llc`, `opt` (Phase 5), `mlir-opt`, `mlir-translate`, `mlir-runner` (Phase 3)

### One-Click Setup

```bash
# Default: installs LLVM 20 automatically
make setup-gpu          # GPU/dev profile
make setup-mlir         # MLIR+GPU profile (Phase 3+5)

# Override LLVM version (not recommended):
make setup-gpu LLVM_VERSION=18
```

Or directly:

```bash
ARKE_LLVM_VERSION=20 scripts/bootstrap_env.sh gpu-dev
```

The bootstrap script:
1. Downloads `llvm-20` deb via `apt-get download` (no root)
2. Extracts to `~/opt/mlir20/root/` via `dpkg-deb -x`
3. Generates/updates `~/opt/mlir20/env.sh` with all LLVM 20 paths
4. Injects `source ~/opt/mlir20/env.sh` into the venv's `activate` script

### Manual LLVM 20 Install (if bootstrap fails)

On Ubuntu/WSL2, MLIR 20 tools can be installed user-local from the official LLVM
packages:

```bash
# Download the Ubuntu LLVM 20, mlir-20-tools and libmlir-20 deb packages
apt download llvm-20 mlir-20-tools libmlir-20

# Extract to user-local directory (no root needed)
mkdir -p ~/opt/mlir20/root
dpkg-deb -x llvm-20_*.deb ~/opt/mlir20/root
dpkg-deb -x mlir-20-tools_*.deb ~/opt/mlir20/root
dpkg-deb -x libmlir-20_*.deb ~/opt/mlir20/root
```

Create an activation script at `~/opt/mlir20/env.sh`:

```bash
cat > ~/opt/mlir20/env.sh << 'EOF'
export PATH="$HOME/opt/mlir20/usr/lib/llvm-20/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/opt/mlir20/usr/lib/llvm-20/lib:${LD_LIBRARY_PATH:-}"
EOF
```

Then source it in your shell:

```bash
source ~/opt/mlir20/env.sh
mlir-opt --version  # verify
```

### MLIR GPU Pipeline (How It Works)

The Phase 3 MLIR GPU pipeline:

```text
SemanticIR → MLIR (linalg/gpu/scf) → mlir-opt lowering → PTX (via NVPTX)
  → cubin (via ptxas) → CUDA driver API launch (via cuda-python)
```

- `mlir-opt` lowers high-level MLIR to NVVM IR and PTX
- `ptxas` assembles PTX to GPU binary (cubin)
- `cuda-python` loads cubin and launches kernels via the CUDA driver API
- Transcendental functions (exp, tanh, etc.) link via `libdevice.10.bc`

The libdevice bitcode is typically found at:
```
/usr/local/cuda/nvvm/libdevice/libdevice.10.bc
```

### Verification

```bash
source ~/opt/mlir20/env.sh
source ~/.venvs/arke/bin/activate  # or your venv

# CPU correctness test
python -m pytest tests/backend/test_mlir_ops_p3s2.py -q

# GPU correctness test
python -m pytest tests/backend/test_mlir_gpu_elementwise_p3s2.py -q

# Full benchmark (MLIR-GPU vs cuBLAS/cuDNN)
python -m benchmarks.bench_mlir_gpu
```

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

## Git Ignore Behavior

The default project-local environment directory is:

```bash
.venv/
```

This path must remain ignored by git so local environments are never committed.
The repository `.gitignore` already includes `.venv/`.

## Repository Git Push Defaults

For repository-local git defaults, Arke provides:

```bash
make git-setup-defaults
# or
scripts/git_setup_defaults.sh
```

This configures the current repository to:

- prefer pushing the current branch by default
- auto-setup upstream for new branches when possible
- provide helpful aliases:
  - `git pub` → `git push -u origin HEAD`
  - `git sync` → `git pull --rebase --autostash && git push`

This does **not** force global user git behavior; it only sets repo-local defaults.

## Notes

- The bootstrap script **recreates** the target venv path from scratch.
- If a benchmark dependency requires a platform-specific wheel or CUDA toolchain, install logs should be preserved for debugging.
- `bench` profile is the recommended baseline for Stage 7 / benchmark validation work.
