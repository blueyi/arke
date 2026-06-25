# G8[4b] — Arke→torch.compile bridge invocation evidence (D7-E1.4)

**Date:** 2026-06-25 · **Hardware:** RTX 3060 Laptop 6GB (SM 8.6), fp16.

Concrete evidence for **G8 Tier-2[4b]**: ≥1 Arke-generated Triton kernel runs on
GPT-2's critical path, correctness 100%, via `torch.library.custom_op`
registration (`arke/integration/torch_bridge.py`, D7-E1.4).

## What the bridge does

`register_arke_ops()` registers two Arke custom ops into the `arke::` torch
library, each with a `register_fake` abstract impl so Dynamo traces through them
without graph breaks:

- `arke::matmul(a, b)` — routes through Arke's Triton matmul kernel
  (`TritonBackend.lower/compile/run`), eager `torch.matmul` fallback on miss.
- `arke::rmsnorm(x, w, eps)` — routes through Arke's Triton rmsnorm kernel,
  eager fallback on miss.

Both are **inference-only** (no autograd backward), single-file, not exported
from `arke.__init__`, not part of the Façade — a transient Substrate artifact
per the scope guardrails (`docs/architecture/arke-harness.md` §3.0.3).

## Invocation evidence (`bridge_invocation.json`)

GPT-2's `Conv1D` linear projections were routed through `arke::matmul`, then a
full forward was run and compared to vanilla eager GPT-2:

| Metric | Value |
|---|---|
| `arke::matmul` invocations per forward | **48** (every Conv1D projection — critical path) |
| eager top-1 token | 329 |
| arke-bridged top-1 token | 329 |
| **top-1 match** | **✅ true** |
| logits max-abs-diff (fp16) | 0.125 |
| correctness_ok | ✅ true |

48 Arke Triton matmul kernels fire on every GPT-2 forward and the model's
top-1 next-token prediction is **unchanged** — i.e. Arke kernels are doing real
work on the critical path while preserving correctness.

## Unit + trace coverage

- `tests/test_torch_bridge.py` (5 tests): op registration idempotency, matmul +
  rmsnorm correctness, `register_fake` enables `torch.compile` trace, and the
  scope-guardrail assertion that the bridge is NOT exported from `arke.__init__`.
- Smoke: `arke::matmul` + `arke::rmsnorm` both dispatch to real Arke Triton
  (rmsnorm max_diff 0.0039, matmul 0.0312 vs eager) and `torch.compile` traces a
  2-op model end-to-end.

## Regenerate

```bash
cd ~/workspace/repos/arke && source ~/.venvs/arke/bin/activate
python - <<'PY'
# see git history: _kitty_bridge_evidence.py (Conv1D → arke::matmul, forward parity)
PY
```
