# Gate Archive Structure

Gate results are archived at each Stage exit and major milestone.
Each archive is a self-contained snapshot for reproducibility.

## Directory Layout

```
benchmarks/results/gates/
├── ARCHIVE.md                         # This file
├── phase1/                            # Phase 1 archives
│   ├── G0/                            # Gate 0 (final exit state)
│   │   ├── meta.json
│   │   ├── summary.json
│   │   ├── inputs/
│   │   │   ├── shapes.json
│   │   │   └── hardware.json
│   │   └── sources/
│   │       └── ak/
│   ├── G1/
│   │   ├── ...
│   │   └── sources/
│   │       ├── ak/
│   │       ├── ir/                    # Representative IR snapshots
│   │       └── triton/                # Generated Triton source
│   └── G2/
│       ├── ...
│       ├── accuracy/                  # Per-shape accuracy CSV
│       │   ├── matmul_accuracy.csv
│       │   ├── softmax_accuracy.csv
│       │   ├── elementwise_accuracy.csv
│       │   └── layernorm_accuracy.csv
│       └── performance/               # Per-shape perf CSV (multi-trial median)
│           ├── matmul_perf.csv
│           ├── softmax_perf.csv
│           ├── elementwise_perf.csv
│           └── layernorm_perf.csv
├── phase2/                            # Phase 2 archives (future)
│   ├── G3/
│   ├── G4/
│   └── G5/
└── ...
```

## Usage

```bash
# Archive with explicit stage
python -m benchmarks.gate G0 G1 G2 --tier 2 --stage phase1 --archive

# Default stage is "phase1"
python -m benchmarks.gate G2 --tier 2 --archive

# Future stages
python -m benchmarks.gate G3 G4 --tier 3 --stage phase2 --archive
```

## CSV Format

### accuracy/*.csv
```
shape_tag,op,M,N,K,matches_ref,max_abs_diff,status
```

### performance/*.csv
```
shape_tag,op,M,N,K,arke_us,baseline_us,baseline_name,ratio,trials,warmup,reps
```

## meta.json
```json
{
  "gate": "G2",
  "stage": "phase1",
  "timestamp": "2026-04-03T09:00:00+08:00",
  "commit": "845c3fb",
  "tag": "phase1-g2-exit",
  "tier": 2,
  "command": "python -m benchmarks.gate G2 --tier 2 --stage phase1 --archive",
  "hardware": {
    "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "cuda": "12.4",
    "pytorch": "2.6.0+cu124",
    "triton": "3.2.0"
  }
}
```

## Rules

- Gate directory is **overwritten** on re-run (only final exit state kept per stage)
- Each archive includes reproducibility info (commit, command)
- Stages are independent — phase2 G3 doesn't overwrite phase1 G2
- Tag format suggestion: `stage{N}-g{M}-exit` (e.g., `phase1-g2-exit`)
