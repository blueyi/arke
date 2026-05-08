# Arke Test Layout

This directory contains all automated tests, organized by *test scope* first and by *phase/stage ownership* where that improves navigation.

## Layout

- `conftest.py`
  - Shared pytest fixtures and configuration.
- `benchmark/`
  - Stage-agnostic benchmark infrastructure tests.
  - Covers reusable benchmark layers such as dashboards, artifact contracts, runners, CLI, status, memory policy, and baseline comparison logic.
- `phase1/`
  - Tests tied to Phase 1 delivery contracts.
  - `stage7/` contains Stage 7 / Gate G7 specific tests and artifact-contract checks.
- repository-root `test_*.py`
  - Cross-cutting tests that are not yet stage-owned or are shared compiler / IR / agent regression tests.
  - These should stay here unless a clearer phase/stage or subsystem home emerges.

## Organization Rules

### 1. Put phase/stage-owned tests under `tests/phase*/stage*/`
Use this for tests whose meaning is defined by a roadmap checkpoint or gate contract.

Examples:
- Gate-specific artifact contracts
- Stage-only language/IR closure tests
- Stage dashboards or reports that exist to satisfy a specific gate

### 2. Put reusable benchmark infrastructure tests under `tests/benchmark/`
Use this for tests of benchmark machinery that should survive stage changes.

Examples:
- generic dashboard synthesis
- artifact tree validators
- benchmark CLI behavior
- benchmark status summaries
- baseline resolution / advice / memory policy helpers

### 3. Keep core compiler / IR / parser / backend regressions at top level unless a better subsystem folder is introduced
These tests often cut across multiple phases and are currently easiest to discover at the root.

Examples:
- parser
- semantic IR
- strategy IR
- MLIR backend
- pipeline
- agent tools

## Naming Conventions

- Generic benchmark tests:
  - `tests/benchmark/test_<feature>.py`
- Phase/stage tests:
  - `tests/phase1/stage7/test_<stage_contract>.py`
- Cross-cutting regressions:
  - `tests/test_<subsystem>.py`

## Current Intentional Split

- `tests/benchmark/test_dashboard.py`
  - Verifies the standardized benchmark dashboard builder in `benchmarks/dashboard.py`.
- `tests/phase1/stage7/test_stage7_dashboard.py`
  - Verifies the Stage 7 dashboard wrapper around the generic dashboard.
- `tests/phase1/stage7/test_track6_contract.py`
  - Verifies Stage 7 Track 6 artifact presence in the committed result tree.

## Test Organization Guidance

When adding new tests:
1. Ask whether the behavior is stage-specific or reusable.
2. If reusable across stages/phases, prefer `tests/benchmark/` or another subsystem folder.
3. If the behavior exists only because a gate requires it, place it under the relevant `tests/phase*/stage*/` tree.
4. If uncertain, prefer the more generic location and add a thin stage wrapper test only when needed.
