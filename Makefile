.PHONY: help setup install dev test test-gpu lint format check clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## One-click environment setup (venv + deps + GPU verification)
	@echo "==> Creating virtual environment..."
	python3 -m venv $(VENV)
	@echo "==> Installing PyTorch + Triton + Arke..."
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo "==> Verifying GPU..."
	@$(PYTHON) -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
	@$(PYTHON) -c "import triton; print(f'  Triton: {triton.__version__}')"
	@echo "==> Running smoke test..."
	$(PYTHON) -m pytest tests/ -q --tb=line -x 2>/dev/null && echo "  Tests: PASS" || echo "  Tests: FAIL (run 'make test' for details)"
	@echo ""
	@echo "Setup complete. Activate with: source $(VENV)/bin/activate"

install:  ## Install arke package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests (CPU only)
	pytest tests/ -v --tb=short

test-gpu:  ## Run tests including GPU correctness tests
	ARKE_GPU_TESTS=1 pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check arke/ tests/

format:  ## Format code
	ruff format arke/ tests/

check:  ## Run all checks (lint + type check + test)
	ruff check arke/ tests/
	mypy arke/ --ignore-missing-imports
	pytest tests/ -v --tb=short

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
