# Gate Archive Structure

Gate results are archived at each Stage exit and major milestone.
Each archive is a self-contained snapshot for reproducibility.

## Directory Layout

```
benchmarks/results/gates/
└── G{N}/                              # One directory per gate (final exit state)
    ├── meta.json                       # Gate metadata (commit, tag, command, timestamp)
    ├── summary.json                    # Gate pass/fail summary with per-criterion detail
    ├── inputs/                         # Engineering inputs
    │   ├── shapes.json                 # Shape definitions used
    │   └── hardware.json               # Hardware profile
    ├── sources/                        # Arke source artifacts
    │   ├── ak/                         # .ak source files
    │   │   ├── 01_matmul.ak
    │   │   └── ...
    │   ├── ir/                         # Semantic + Strategy IR snapshots
    │   │   ├── matmul_1024_1024.json
    │   │   └── ...
    │   └── triton/                     # Generated Triton source code
    │       ├── matmul_1024_1024.py
    │       └── ...
    ├── accuracy/                       # Per-shape accuracy results
    │   ├── matmul_accuracy.csv
    │   ├── softmax_accuracy.csv
    │   ├── elementwise_accuracy.csv
    │   └── layernorm_accuracy.csv
    └── performance/                    # Per-shape performance results
        ├── matmul_perf.csv
        ├── softmax_perf.csv
        ├── elementwise_perf.csv
        └── layernorm_perf.csv
```

## CSV Format

### accuracy/*.csv
```
shape_tag,op,M,N,K,arke_matches_ref,max_abs_diff,max_rel_diff,status
```

### performance/*.csv
```
shape_tag,op,M,N,K,arke_us,baseline_us,baseline_name,ratio,trials,warmup,reps
```

## meta.json
```json
{
  "gate": "G2",
  "stage": "Stage 1",
  "timestamp": "2026-04-03T09:00:00+08:00",
  "commit": "845c3fb",
  "tag": "stage1-g2-exit",
  "tier": 2,
  "command": "python -m benchmarks.gate G2 --tier 2 --archive",
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "cuda": "12.4",
    "pytorch": "2.6.0+cu124",
    "triton": "3.2.0"
  }
}
```

## Rules
- Gate directory is **overwritten** on re-run (only final exit state kept)
- Each archive includes reproducibility info (commit, command)
- Tag format: `stage{N}-g{M}-exit` (e.g., `stage1-g2-exit`)
