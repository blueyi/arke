#!/usr/bin/env bash
# batch_matmul slim-launch acceptance: bench_l1 median-of-3.
set -u
cd ~/workspace/repos/arke
source ~/.venvs/arke/bin/activate
for run in 1 2 3; do
  echo "=== RUN $run op=batch_matmul ==="
  python -m benchmarks.bench_l1 --op batch_matmul --tier 2 --no-resume --force-restart >/dev/null 2>&1
  python - "$run" <<'EOF'
import csv, math, sys
run = sys.argv[1]
p = "benchmarks/results/phase1/stage6/trackg6/l1/perf_batch_matmul.csv"
lat = {}
for r in csv.DictReader(open(p)):
    if r["status"] != "ok":
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
    print(f"RESULT run={run} op=batch_matmul geomean={g:.4f} n={len(ratios)} :: {detail}")
else:
    print(f"RESULT run={run} op=batch_matmul NO-DATA")
EOF
done
echo "DONE"
