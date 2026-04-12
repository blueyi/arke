#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ARKE_VENV:-$ROOT_DIR/.venv}"
PYTHON_BIN="${ARKE_PYTHON:-python3}"
PROFILE="${1:-gpu-dev}"

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_env.sh [cpu-dev|gpu-dev|bench]

Profiles:
  cpu-dev  Create a fresh venv and install editable Arke + dev deps
  gpu-dev  Create a fresh venv and install editable Arke + dev + GPU deps
  bench    Create a fresh venv and install editable Arke + benchmark stack

Environment variables:
  ARKE_VENV    Override venv path (default: ./.venv)
  ARKE_PYTHON  Override Python executable used to create the venv (default: python3)
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
  cpu-dev|gpu-dev|bench) ;;
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
  bench)
    "$VENV_DIR/bin/pip" install -e ".[dev,gpu]"
    "$VENV_DIR/bin/pip" install -r "$ROOT_DIR/requirements-benchmark.txt"
    ;;
esac

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

cat <<EOF

Environment ready.
Activate it with:
  source "$VENV_DIR/bin/activate"

Recommended next steps:
  $VENV_DIR/bin/python -m pytest tests/test_parser.py -q
  $VENV_DIR/bin/python -m benchmarks --layer L1
EOF
