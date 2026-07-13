#!/usr/bin/env bash
# Batch live-LLM autotuning across multiple ops → accumulate RL corpus.
# Each run drives claude-sonnet-4-6 through the frozen Façade on one op/shape,
# emits a mineable trajectory, and the whole batch is merged into one RL dataset.
set -uo pipefail

cd ~/workspace/repos/arke
source ~/.env.rc 2>/dev/null || true
# yunwu token is shared; ~/.env.rc only exports ANTHROPIC_API_KEY. Use it as the
# /v1 OpenAI-compatible key (clean endpoint — avoids Claude-Code ctx injection).
export ARKE_LLM_API_KEY="${OPENAI_API_KEY:-$ANTHROPIC_API_KEY}"
export ARKE_LLM_BASE_URL="https://yunwu.ai/v1"
export ARKE_LLM_PROTOCOL="openai"
export ARKE_LLM_MODEL="claude-sonnet-4-6"
export PATH=/usr/local/cuda/bin:$PATH
source ~/.venvs/arke/bin/activate

OUTROOT="benchmarks/results/phase4/live"
MODEL="yunwu/claude-sonnet-4-6"

# (op, shape, out-subdir) — a spread across tiers, biased to under-target ops.
run_one () {
  local op="$1" shape="$2" dir="$3"
  echo "=== [$(date +%H:%M:%S)] autotune $op ($shape) ==="
  timeout 400 python -m benchmarks.live.run_live_optimize \
    --op "$op" --shape "$shape" --model "$MODEL" \
    --max-turns 12 --timeout 150 --out "$OUTROOT/$dir" 2>&1 | tail -3
}

run_one matmul   "1024,1024,1024" matmul_1024
run_one matmul   "256,256,256"    matmul_256
run_one softmax  "64,4096"        softmax_4096
run_one layernorm "64,2048"       layernorm_2048
run_one silu     "1024,1024"      silu_1024
run_one add      "512,512"        add_512

echo "=== batch complete; mining RL corpus ==="
python - <<'PY'
import glob, json
from arke.learn.rl_dataset import build_rl_dataset, extract_rl_samples, reward_histogram
trajs = sorted(glob.glob("benchmarks/results/phase4/live/*/trajectory.jsonl"))
counts = build_rl_dataset(trajs, "benchmarks/results/phase4/live/rl_corpus.jsonl")
print("RL corpus:", counts)
# Reward distribution across all step samples
allsteps = []
for t in trajs:
    s, tj = extract_rl_samples(t)
    allsteps += s
print("reward histogram (steps):", reward_histogram(allsteps))
print("trajectories mined:", len(trajs))
PY
