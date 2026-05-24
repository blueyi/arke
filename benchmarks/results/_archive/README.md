# benchmarks/results/_archive/

This directory holds **superseded benchmark result snapshots** that are no longer the
authoritative source for any active Gate, but are preserved for reproducibility audit
and historical comparison.

## Why archive instead of delete?

Per Arke project policy (AGENTS.md §"Arke 工作流偏好"):

- We **don't delete ops, shapes, or evidence rows** even when superseded.
- Old runs may be needed to:
  - Diff regressions across Stage transitions
  - Reconstruct decision context when reviewing a past commit
  - Audit baseline-runner methodology changes

## Layout

Each subdirectory is named `<stage>-<track>-<level>-<original-timestamp>/` and
contains the verbatim original output tree (CSV / JSON / log).

| Subdirectory | Source | Reason archived |
|:---|:---|:---|
| `stage8-track4-l3-2026-05-17_015001/` | Stage 8 Track 4 L3 run on 2026-05-17 | Superseded by post-D8-F3 trajectory v1.0 schema (commit `dd4f7c7`); old run uses pre-v1.0 trajectory layout |

## Adding to this archive

```bash
mv benchmarks/results/<path>/<run-dir> benchmarks/results/_archive/<descriptive-name>/
# Then update this README's table.
# Commit with prefix: chore(archive): ...
```

## NOT to be used for

- Active Gate evidence (use `benchmarks/results/phase1/stage<N>/...` instead)
- CI baselines (those live in `benchmarks/baselines/`)
- Regression detection runs (re-run fresh)
