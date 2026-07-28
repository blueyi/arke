# Convergence curves — KESTREL K-H5.2

## Purpose

Answer the 2026-07-27 audit §一.3 gap: **how efficient is Arke's optimization
loop's convergence?** Each CSV projects a live-agent `arke run` trajectory
into the sequence `(iteration, current_ratio, best_so_far_ratio)`, one row
per `compile_and_profile` call. See `arke/agent/convergence.py` for the
extraction contract (module docstring).

## First batch (2026-07-28) — matmul / softmax / flash_attention

| Op | Shape | Iterations | Correct | best_so_far ratio | Notes |
|:---|:---|:---:|:---:|:---:|:---|
| matmul | 512×512×512 | 2 | 2/2 | 0.38 → **1.28** | LLM found a config beating default 1.28× |
| softmax | 512×1024 | 4 | 4/4 | **1.0** (flat) | LLM tried 3 alternatives, none beat default (one 0.29× regression) |
| flash_attention | 2×4×512×64 | 3 | 1/3 | (none) → **1.0** | 2/3 correctness failures — honest attention weakness (informs K-ATT) |

## Reproduction

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python -m arke.cli run \
    --kernel matmul --shape 512,512,512 \
    --backend builtin --model yunwu/claude-sonnet-4-6 \
    --max-turns 15 --timeout 600 \
    --output /tmp/arke_conv_matmul
# writes /tmp/arke_conv_matmul/{convergence.csv, trajectory.json, state.json}
```

CSV always lands at `<output_dir>/convergence.csv`; override with
`--emit-convergence-csv PATH`.

## Columns (stable contract, see `arke.agent.convergence.CONVERGENCE_COLUMNS`)

`iteration, step, tool, backend, success, correct, max_diff, latency_ms,
baseline_ratio, vs_default, meas_spread, current_ratio, best_so_far_ratio`

- `current_ratio` prefers `1 / vs_default` (gate criterion) then falls back to
  `baseline_ratio`. Higher is better.
- `best_so_far_ratio` advances **only** on `success && correct == True`;
  failures / incorrect results still emit a row (audit trail) with the running
  best unchanged.

## Live-run observations

- Answers "how efficient is the loop" with **honest data**: matmul converged
  in 2 iters; softmax spent 4 turns bouncing around the default; flash_attention
  suffered 2 correctness failures — the same weakness K-ATT is scoped to fix.
- Auto-routing (`compile_and_profile` picks `llvm` backend when L3 decisions
  are applied) is visible per-iter — see softmax rows (all `llvm`) vs matmul
  (`triton`).

## Related

- K-H5.2 in `docs/audit/kestrel-backlog.md`
- Convergence extractor: `arke/agent/convergence.py`
- Unit tests: `tests/agent/test_convergence.py` (16 tests)
- CLI flag: `arke run --emit-convergence-csv PATH` (defaults to
  `<output_dir>/convergence.csv`)
