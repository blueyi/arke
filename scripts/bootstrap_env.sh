#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ARKE_VENV:-$ROOT_DIR/.venv}"
PYTHON_BIN="${ARKE_PYTHON:-python3}"
PROFILE="${1:-gpu-dev}"

# ── LLVM version config ──────────────────────────────────────
# Default: LLVM 20 (aligned with MLIR 20 / Triton 3.2 / PyTorch 2.6).
# Override version:  ARKE_LLVM_VERSION=20 scripts/bootstrap_env.sh gpu-dev
# Source build:      ARKE_LLVM_SRC=/path/to/llvm-project scripts/bootstrap_env.sh gpu-dev
#   (builds LLVM from source with NVPTX + MLIR enabled, installs to ARKE_LLVM_HOME)
LLVM_VERSION="${ARKE_LLVM_VERSION:-20}"
LLVM_INSTALL_DIR="${ARKE_LLVM_HOME:-$HOME/opt/mlir20/root}"
LLVM_SRC="${ARKE_LLVM_SRC:-}"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_env.sh [cpu-dev|gpu-dev|mlir-gpu|bench]

Profiles:
  cpu-dev   Create a fresh venv and install editable Arke + dev deps
  gpu-dev   Create a fresh venv and install editable Arke + dev + GPU deps
  mlir-gpu  Create a fresh venv and install editable Arke + dev + GPU + MLIR deps (Phase 3+5)
  bench     Create a fresh venv and install editable Arke + benchmark stack

Environment variables:
  ARKE_VENV          Override venv path (default: ./.venv)
  ARKE_PYTHON        Override Python executable (default: python3)
  ARKE_LLVM_VERSION  LLVM version to use (default: 20)
  ARKE_LLVM_HOME     Override LLVM install prefix (default: ~/opt/mlir20/root)
  ARKE_LLVM_SRC      Path to llvm-project source tree — build from source
                     (cmake + ninja required; builds with NVPTX + MLIR enabled)
EOF
}

if [[ "$PROFILE" == "-h" || "$PROFILE" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

case "$PROFILE" in
  cpu-dev|gpu-dev|mlir-gpu|bench) ;;
  *)
    echo "error: unknown profile: $PROFILE" >&2
    usage
    exit 1
    ;;
esac

echo "==> Root: $ROOT_DIR"
echo "==> Python: $PYTHON_BIN"
echo "==> Venv: $VENV_DIR"
echo "==> Profile: $PROFILE"
echo "==> LLVM version: $LLVM_VERSION"

rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/pip" install --upgrade pip setuptools wheel

case "$PROFILE" in
  cpu-dev)
    "$VENV_DIR/bin/pip" install -e ".[dev]"
    ;;
  gpu-dev)
    "$VENV_DIR/bin/pip" install -e ".[dev,gpu]"
    ;;
  mlir-gpu)
    "$VENV_DIR/bin/pip" install -e ".[dev,gpu,mlir-gpu]"
    ;;
  bench)
    "$VENV_DIR/bin/pip" install -e ".[dev,gpu]"
    "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements-benchmark.txt"
    ;;
esac

# ── LLVM/MLIR toolchain setup (gpu-dev, mlir-gpu, bench) ─────
if [[ "$PROFILE" != "cpu-dev" ]]; then
  _setup_llvm_toolchain() {
    local llvm_ver="$1"
    local install_dir="$2"
    local llvm_bin_dir="$install_dir/usr/lib/llvm-${llvm_ver}/bin"
    local env_sh="$HOME/opt/mlir20/env.sh"

    echo "==> Checking LLVM ${llvm_ver} toolchain"

    # Check if llc is already installed at expected location
    if [[ -f "$llvm_bin_dir/llc" ]]; then
      echo "    llc found: $llvm_bin_dir/llc"
      "$llvm_bin_dir/llc" --version 2>&1 | head -1
    elif [[ -n "$LLVM_SRC" ]]; then
      # ── Build from source ──────────────────────────────────
      echo "    Building LLVM from source: $LLVM_SRC"
      if [[ ! -d "$LLVM_SRC/llvm" ]]; then
        echo "error: ARKE_LLVM_SRC=$LLVM_SRC does not contain llvm/ subdirectory" >&2
        echo "       Expected: a llvm-project checkout (git clone https://github.com/llvm/llvm-project.git)" >&2
        exit 1
      fi
      if ! command -v cmake >/dev/null 2>&1; then
        echo "error: cmake not found — required for LLVM source build" >&2
        exit 1
      fi

      local build_dir="$LLVM_SRC/build-arke"
      local cmake_gen="Unix Makefiles"
      if command -v ninja >/dev/null 2>&1; then
        cmake_gen="Ninja"
      fi

      local install_prefix="$install_dir/usr/lib/llvm-${llvm_ver}"
      mkdir -p "$build_dir"

      echo "    cmake generator: $cmake_gen"
      echo "    install prefix:  $install_prefix"
      echo "    This may take 30-60 minutes..."

      cmake -S "$LLVM_SRC/llvm" -B "$build_dir" \
        -G "$cmake_gen" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$install_prefix" \
        -DLLVM_ENABLE_PROJECTS="mlir;clang" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX" \
        -DLLVM_INSTALL_UTILS=ON \
        -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
        -DLLVM_ENABLE_ASSERTIONS=OFF \
        -DLLVM_ENABLE_RTTI=ON

      cmake --build "$build_dir" --target install -- -j"$(nproc)"

      echo "    ✅ LLVM ${llvm_ver} built and installed to $install_prefix"
    else
      echo "    llc-${llvm_ver} not found at $llvm_bin_dir"
      echo "    Attempting to download and extract llvm-${llvm_ver}..."

      # Try apt download + dpkg-deb extract (no root needed)
      if command -v apt-get >/dev/null 2>&1; then
        local tmp_dir
        tmp_dir=$(mktemp -d)
        pushd "$tmp_dir" >/dev/null
        if apt-get download "llvm-${llvm_ver}" 2>/dev/null; then
          local deb_file
          deb_file=$(ls llvm-${llvm_ver}*.deb 2>/dev/null | head -1)
          if [[ -n "$deb_file" ]]; then
            dpkg-deb -x "$deb_file" "$install_dir"
            echo "    ✅ llvm-${llvm_ver} extracted to $install_dir"
          fi
        else
          echo "    ⚠️  apt-get download llvm-${llvm_ver} failed."
          echo "    Manual install: apt download llvm-${llvm_ver} && dpkg-deb -x llvm-${llvm_ver}*.deb $install_dir"
        fi
        popd >/dev/null
        rm -rf "$tmp_dir"
      else
        echo "    ⚠️  apt-get not available. Install llvm-${llvm_ver} manually."
        echo "    Then set ARKE_LLC=$llvm_bin_dir/llc"
      fi
    fi

    # Check MLIR tools
    if [[ -f "$llvm_bin_dir/mlir-opt" ]]; then
      echo "    mlir-opt found: $llvm_bin_dir/mlir-opt"
    elif [[ "$PROFILE" == "mlir-gpu" ]]; then
      echo "    ⚠️  mlir-opt not found. Install mlir-${llvm_ver}-tools:"
      echo "       apt download mlir-${llvm_ver}-tools libmlir-${llvm_ver}"
      echo "       dpkg-deb -x mlir-${llvm_ver}-tools*.deb $install_dir"
      echo "       dpkg-deb -x libmlir-${llvm_ver}*.deb $install_dir"
    fi

    # Write/update env.sh if it doesn't exist or is outdated
    if [[ ! -f "$env_sh" ]] || ! grep -q "LLVM.*${llvm_ver}" "$env_sh" 2>/dev/null; then
      echo "==> Generating $env_sh for LLVM ${llvm_ver}"
      mkdir -p "$(dirname "$env_sh")"
      cat > "$env_sh" <<ENVEOF
#!/usr/bin/env bash
# Arke LLVM/MLIR toolchain environment (auto-generated by bootstrap_env.sh).
# LLVM version: ${llvm_ver} (aligned with Triton 3.2 / PyTorch 2.6)
# Source this to activate: source ~/opt/mlir20/env.sh
export MLIR_HOME="$install_dir/usr/lib/llvm-${llvm_ver}"
export PATH="\$MLIR_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$MLIR_HOME/lib:\${LD_LIBRARY_PATH:-}"
# MLIR runner libs (Phase 3 MLIR GPU):
export ARKE_MLIR_RUNNER_UTILS="\$MLIR_HOME/lib/libmlir_runner_utils.so.${llvm_ver}.1"
export ARKE_MLIR_C_RUNNER_UTILS="\$MLIR_HOME/lib/libmlir_c_runner_utils.so.${llvm_ver}.1"
export ARKE_MLIR_OPT="\$MLIR_HOME/bin/mlir-opt"
export ARKE_MLIR_TRANSLATE="\$MLIR_HOME/bin/mlir-translate"
export ARKE_MLIR_CPU_RUNNER="\$MLIR_HOME/bin/mlir-runner"
# Phase 5 LLVM-IR backend:
export ARKE_LLC="\$MLIR_HOME/bin/llc"
# GPU (NVPTX):
if [ -d /usr/local/cuda/bin ]; then export PATH="/usr/local/cuda/bin:\$PATH"; fi
export ARKE_GPU_CHIP="\${ARKE_GPU_CHIP:-sm_86}"
ENVEOF
    fi

    # Inject source env.sh into venv activate script
    local activate="$VENV_DIR/bin/activate"
    if [[ -f "$activate" ]] && ! grep -q "mlir20/env.sh" "$activate" 2>/dev/null; then
      echo "==> Injecting LLVM/MLIR env into venv activate"
      cat >> "$activate" <<ACTEOF

# ── Arke LLVM/MLIR toolchain (auto-injected by bootstrap_env.sh) ──
# LLVM ${llvm_ver}, aligned with Triton 3.2 / PyTorch 2.6
if [ -f "\$HOME/opt/mlir20/env.sh" ]; then
    source "\$HOME/opt/mlir20/env.sh"
fi
export GEMS_VENDOR="\${GEMS_VENDOR:-nvidia}"
ACTEOF
    fi
  }

  _setup_llvm_toolchain "$LLVM_VERSION" "$LLVM_INSTALL_DIR"
fi

echo "==> Verifying environment"
"$VENV_DIR/bin/python" - <<'PY'
import importlib
import sys

mods = ["arke", "numpy", "lark", "jinja2", "click", "rich"]
missing = []
for name in mods:
    try:
        importlib.import_module(name)
    except Exception as e:
        missing.append(f"{name}: {e}")
if missing:
    print("Missing modules:")
    for item in missing:
        print(f"  - {item}")
    sys.exit(1)
print("Core imports: OK")
PY

if [[ "$PROFILE" != "cpu-dev" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"Torch verification skipped/failed: {e}")
PY
fi

if [[ "$PROFILE" == "mlir-gpu" ]]; then
  "$VENV_DIR/bin/python" - <<'PY'
try:
    import cuda.cuda
    print("cuda-python: OK")
except Exception as e:
    print(f"cuda-python verification failed: {e}")
try:
    import subprocess
    r = subprocess.run(["mlir-opt", "--version"], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"mlir-opt: {r.stdout.strip()}")
    else:
        print("mlir-opt: NOT FOUND in PATH (source ~/opt/mlir20/env.sh first)")
except FileNotFoundError:
    print("mlir-opt: NOT FOUND in PATH (source ~/opt/mlir20/env.sh first)")
PY
fi

# Verify LLVM toolchain for non-cpu profiles
if [[ "$PROFILE" != "cpu-dev" ]]; then
  echo "==> Verifying LLVM toolchain"
  "$VENV_DIR/bin/python" - <<PY
import os, subprocess, shutil

# Check llc
llc = os.environ.get("ARKE_LLC") or shutil.which("llc")
if llc and os.path.isfile(llc):
    r = subprocess.run([llc, "--version"], capture_output=True, text=True)
    ver_line = r.stdout.split("\\n")[0] if r.stdout else "(unknown)"
    print(f"llc: {llc} ({ver_line})")
    if "${LLVM_VERSION}" not in ver_line:
        print(f"  ⚠️  Expected LLVM ${LLVM_VERSION}, got: {ver_line}")
else:
    print("llc: NOT FOUND — Phase 5 LLVM-IR backend will not work")
    print("  Fix: source ~/opt/mlir20/env.sh or set ARKE_LLC")

# Check ptxas
ptxas = shutil.which("ptxas")
if ptxas:
    print(f"ptxas: {ptxas}")
else:
    print("ptxas: NOT FOUND — add /usr/local/cuda/bin to PATH")
PY
fi

cat <<EOF

Environment ready.
Activate it with:
  source "$VENV_DIR/bin/activate"

LLVM/MLIR toolchain: LLVM $LLVM_VERSION (auto-sourced on venv activate)
To override LLVM version (not recommended — default 20 is aligned with MLIR/Triton):
  ARKE_LLVM_VERSION=<ver> scripts/bootstrap_env.sh $PROFILE

Recommended next steps:
  $VENV_DIR/bin/python -m pytest tests/test_parser.py -q
  $VENV_DIR/bin/python -m pytest tests/backend/test_llvm_backend.py -q
EOF
