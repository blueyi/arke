.PHONY: help setup setup-cpu setup-gpu setup-bench install dev test test-gpu lint format check bench clean

VENV ?= .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup: setup-gpu  ## Default one-click environment setup (GPU/dev profile)

setup-cpu:  ## Create a fresh CPU/dev environment in $(VENV)
	ARKE_VENV=$(VENV) scripts/bootstrap_env.sh cpu-dev

setup-gpu:  ## Create a fresh GPU/dev environment in $(VENV)
	ARKE_VENV=$(VENV) scripts/bootstrap_env.sh gpu-dev

setup-bench:  ## Create a fresh benchmark environment in $(VENV)
	ARKE_VENV=$(VENV) scripts/bootstrap_env.sh bench

install:  ## Install arke package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests (CPU only)
	pytest tests/ -v --tb=short

test-gpu:  ## Run tests including GPU correctness tests
	ARKE_GPU_TESTS=1 pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check arke/ tests/ benchmarks/

format:  ## Format code
	ruff format arke/ tests/ benchmarks/

check:  ## Run all checks (lint + type check + test)
	ruff check arke/ tests/ benchmarks/
	mypy arke/ --ignore-missing-imports
	pytest tests/ -v --tb=short

bench:  ## Run full benchmark suite (L1 + L2 + L3)
	python -m benchmarks --all

bench-l1:  ## Run L1 single operator benchmarks
	python -m benchmarks --layer L1

bench-report:  ## Generate benchmark report from existing results
	python -m benchmarks --report

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
