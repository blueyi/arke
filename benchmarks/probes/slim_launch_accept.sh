#!/usr/bin/env bash
# slim-launch acceptance: bench_l1 median-of-3 for the reduce family.
set -u
cd ~/workspace/repos/arke
source ~/.venvs/arke/bin/activate
OPS="softmax reduce_sum reduce_mean layernorm"
for run in 1 2 3; do
  for op in $OPS; do
    echo "=== RUN $run op=$op ==="
    python -m benchmarks.bench_l1 --op "$op" --tier 2 --no-resume --force-restart >/dev/null 2>&1
    python - "$op" "$run" <<'EOF'
import csv, math, sys
op, run = sys.argv[1], sys.argv[2]
p = f"benchmarks/results/phase1/stage6/trackg6/l1/perf_{op}.csv"
lat = {}
for r in csv.DictReader(open(p)):
    if r["operator"] != op or r["status"] != "ok":
        continue
    try:
        lat.setdefault(r["shape_tag"], {})[r["baseline"]] = float(r["latency_us"])
    except ValueError:
        pass
ratios = []
for tag, d in lat.items():
    if "Arke" in d and "FlagGems" in d:
        ratios.append((tag, d["FlagGems"] / d["Arke"]))
if ratios:
    g = math.exp(sum(math.log(x) for _, x in ratios) / len(ratios))
    detail = " ".join(f"{t}={x:.3f}" for t, x in ratios)
    print(f"RESULT run={run} op={op} geomean={g:.4f} n={len(ratios)} :: {detail}")
else:
    print(f"RESULT run={run} op={op} NO-DATA")
EOF
  done
done
echo "DONE"
