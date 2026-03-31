.PHONY: help install dev test lint format check clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install arke package
	pip install -e .

dev:  ## Install with dev dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v --tb=short

lint:  ## Run linter
	ruff check arke/ arkec/ tests/

format:  ## Format code
	ruff format arke/ arkec/ tests/

check:  ## Run all checks (lint + type check + test)
	ruff check arke/ arkec/ tests/
	mypy arke/ arkec/
	pytest tests/ -v --tb=short

clean:  ## Clean build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
