# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unified CLI for the Arke Benchmark System.

Entry points:
    arke bench                              # BL2 default
    arke bench --bl 5                       # Complete suite
    arke bench report {run_id}              # Generate report
    arke bench diff {run_id_1} {run_id_2}   # Compare runs (stub)
    arke bench history --op matmul          # Performance trend (stub)
    python -m benchmarks                    # Equivalent to arke bench

See docs/benchmark/benchmark-protocol.md for full specification.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ── BL → Default expansion table ────────────────────────────────────────

BL_DEFAULTS: dict[int, dict] = {
    1: {"ot_range": (0, 2), "st_range": (1, 1), "layers": ["L1"]},
    2: {"ot_range": (0, 2), "st_range": (1, 2), "layers": ["L1"]},
    3: {"ot_range": (0, 2), "st_range": (1, 3), "layers": ["L1"]},
    4: {"ot_range": (0, 4), "st_range": (1, 2), "layers": ["L1", "L2"]},
    5: {"ot_range": (0, 4), "st_range": (1, 4), "layers": ["L1", "L2"]},
    6: {"ot_range": (0, 4), "st_range": (1, 4), "layers": ["L1", "L2", "L3"]},
}

# ── OT → Operators (sourced from benchmark-ops.md via op_registry) ──────
from benchmarks.op_registry import OT_OPS  # noqa: E402


def _parse_int_list(s: str) -> list[int]:
    """Parse comma-separated integers."""
    return [int(x.strip()) for x in s.split(",")]


def _parse_str_list(s: str) -> list[str]:
    """Parse comma-separated strings."""
    return [x.strip() for x in s.split(",")]


# ── Config / Provenance ─────────────────────────────────────────────────

def _generate_run_id(bl: int = 2, layer: str = "L1") -> str:
    """Generate a run ID based on Gate/BL/Layer.
    
    Format: phase1/stage{N}/g{N}_bl{BL}_{LAYER}
    Example: phase1/stage6/g6_bl4_l1
    """
    # Map BL to Gate
    bl_to_gate = {1: 2, 2: 2, 3: 3, 4: 6, 5: 7, 6: 9}
    gate = bl_to_gate.get(bl, 6)
    
    # Map Gate to Stage
    gate_to_stage = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
    stage = gate_to_stage.get(gate, 6)
    
    return f"phase1/stage{stage}/g{gate}_bl{bl}_{layer.lower()}"


def _write_config(run_dir: Path, config: dict) -> None:
    """Write config.json for provenance."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    logger.info(f"Config: {run_dir / 'config.json'}")


def _write_hardware(run_dir: Path) -> None:
    """Write hardware.json."""
    try:
        from benchmarks.hardware import collect_hardware_info
        hw = collect_hardware_info()
        (run_dir / "hardware.json").write_text(json.dumps(hw, indent=2))
    except Exception as e:
        logger.warning(f"Could not collect hardware info: {e}")


def _merge_perf_all(run_dir: Path) -> None:
    """Merge all per-op CSVs into PERF_ALL.csv."""
    all_rows: list[dict] = []
    for csv_file in sorted(run_dir.rglob("perf_*.csv")):
        try:
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
        except Exception as e:
            logger.warning(f"Could not read {csv_file}: {e}")

    if all_rows:
        all_fields = list(all_rows[0].keys())
        out = run_dir / "PERF_ALL.csv"
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"Merged {len(all_rows)} rows → {out}")


def _write_summary(run_dir: Path) -> None:
    """Generate summary.json with geomean scores."""
    import math

    perf_all = run_dir / "PERF_ALL.csv"
    if not perf_all.exists():
        return

    ratios_by_op: dict[str, list[float]] = {}
    try:
        with open(perf_all) as f:
            for row in csv.DictReader(f):
                op = row.get("operator", "unknown")
                r = row.get("ratio_vs_baseline")
                if r and r not in ("", "N/A", "None"):
                    try:
                        val = float(r)
                        if val > 0:
                            ratios_by_op.setdefault(op, []).append(val)
                    except ValueError:
                        pass
    except Exception as e:
        logger.warning(f"Could not parse PERF_ALL.csv: {e}")
        return

    def _geomean(vals: list[float]) -> float:
        if not vals:
            return 0.0
        return math.exp(sum(math.log(v) for v in vals) / len(vals))

    op_scores = {op: round(_geomean(rs), 4) for op, rs in ratios_by_op.items()}
    all_ratios = [r for rs in ratios_by_op.values() for r in rs]
    overall = round(_geomean(all_ratios), 4) if all_ratios else 0.0

    summary = {
        "overall_geomean": overall,
        "op_scores": op_scores,
        "total_shapes": sum(len(rs) for rs in ratios_by_op.values()),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"Summary: overall geomean = {overall}")


# ── Resolve benchmark config ────────────────────────────────────────────

def resolve_config(args: argparse.Namespace) -> dict:
    """Resolve BL/OT/ST/Layer/Op from CLI args into concrete config."""
    # Resolve BL
    if getattr(args, "all", False):
        bl = 6
    elif args.bl is not None:
        bl = args.bl
    elif any([args.ot, args.st, args.layer, args.op]):
        bl = None  # User specified overrides directly
    else:
        bl = 2  # Default

    # BL defaults
    if bl is not None:
        defaults = BL_DEFAULTS.get(bl, BL_DEFAULTS[2])
        ot_min, ot_max = defaults["ot_range"]
        st_min, st_max = defaults["st_range"]
        layers = list(defaults["layers"])
    else:
        ot_min, ot_max = 0, 4
        st_min, st_max = 1, 4
        layers = ["L1"]

    # Override: layer
    if args.layer:
        layer_str = args.layer.upper()
        layers = [layer_str]
        # Validation: L3 ≡ BL6
        if layer_str == "L3":
            bl = 6
            ot_min, ot_max = 0, 4
            st_min, st_max = 1, 4
        # L2 requires OT3+
        if layer_str == "L2" and ot_max < 3:
            ot_max = 4

    # Override: OT
    if args.ot:
        ot_tiers = _parse_int_list(args.ot)
    else:
        ot_tiers = list(range(ot_min, ot_max + 1))

    # Override: ST
    if args.st:
        st_tiers = _parse_int_list(args.st)
    else:
        st_tiers = list(range(st_min, st_max + 1))

    # Resolve operators
    if args.op:
        ops = _parse_str_list(args.op)
    else:
        ops = []
        for ot in ot_tiers:
            ops.extend(OT_OPS.get(ot, []))

    max_st = max(st_tiers) if st_tiers else 2

    # Shapes filter
    shapes = _parse_str_list(args.shapes) if args.shapes else None

    # Baselines filter
    baselines = _parse_str_list(args.baselines) if args.baselines else None

    return {
        "bl": bl,
        "ot_tiers": ot_tiers,
        "st_tiers": st_tiers,
        "max_st": max_st,
        "layers": layers,
        "ops": ops,
        "shapes": shapes,
        "baselines": baselines,
        "model": getattr(args, "model", None),
        "warmup": args.warmup,
        "reps": args.reps,
        "seq_len": getattr(args, "seq_len", None),
    }


# ── Layer runners ────────────────────────────────────────────────────────

def _run_l1(config: dict, output_dir: str) -> None:
    """Run L1 single operator benchmarks."""
    from benchmarks.bench_l1 import ALL_OPS, run_l1

    supported = set(ALL_OPS)
    runnable = [op for op in config["ops"] if op in supported]
    skipped = [op for op in config["ops"] if op not in supported]

    if skipped:
        logger.warning(f"L1: skipping unsupported ops: {skipped}")
    if not runnable:
        logger.warning("L1: no runnable ops for current config")
        return

    logger.info(f"Running L1: ops={runnable}, max_st={config['max_st']}")
    run_l1(
        ops=runnable,
        output_dir=output_dir,
        warmup=config["warmup"],
        reps=config["reps"],
        tier=config["max_st"],
        shape_tags=config["shapes"],
    )


def _run_l2(config: dict, output_dir: str) -> None:
    """Run L2 fused operator benchmarks."""
    from benchmarks.bench_l2 import ALL_FUSED_OPS, run_l2

    logger.info(f"Running L2: {ALL_FUSED_OPS}")
    run_l2(
        ops=ALL_FUSED_OPS,
        output_dir=output_dir,
        warmup=config["warmup"],
        reps=config["reps"],
        shape_tags=config["shapes"],
    )


def _run_l3(config: dict, output_dir: str) -> None:
    """Run L3 E2E model benchmarks."""
    from benchmarks.bench_l3 import DEFAULT_SEQ_LENS, run_l3

    if config["seq_len"]:
        seq_lens = [int(s.strip()) for s in config["seq_len"].split(",")]
    else:
        seq_lens = DEFAULT_SEQ_LENS

    logger.info(f"Running L3: seq_lens={seq_lens}")
    run_l3(seq_lens=seq_lens, output_dir=output_dir)


# ── Subcommands: report / diff / history ─────────────────────────────────

def _cmd_report(args: argparse.Namespace) -> None:
    """Generate report from results."""
    from benchmarks.report import generate_report

    results_dir = Path(args.results_dir)
    if args.run_id:
        results_dir = results_dir / args.run_id

    report = generate_report(
        results_dir=results_dir,
        output_path=results_dir / "report.md",
    )
    print(report)


def _cmd_diff(args: argparse.Namespace) -> None:
    """Compare two benchmark runs."""
    print(f"arke bench diff: comparing {args.run_id_1} vs {args.run_id_2}")
    print("Not yet implemented. See benchmark-protocol.md §CLI for spec.")
    sys.exit(0)


def _cmd_history(args: argparse.Namespace) -> None:
    """Show performance trend for an operator/shape."""
    print(f"arke bench history: op={args.op}")
    print("Not yet implemented. See benchmark-protocol.md §CLI for spec.")
    sys.exit(0)


# ── Main CLI parser ──────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Build the benchmark CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="arke bench",
        description="Arke Benchmark System — BL/OT/ST/Layer classification",
    )
    sub = parser.add_subparsers(dest="subcmd")

    # --- report subcommand ---
    p_report = sub.add_parser("report", help="Generate report from results")
    p_report.add_argument("run_id", nargs="?", default=None, help="Run ID (default: latest)")
    p_report.add_argument("--results-dir", default="benchmarks/results")

    # --- diff subcommand ---
    p_diff = sub.add_parser("diff", help="Compare two benchmark runs")
    p_diff.add_argument("run_id_1", help="First run ID")
    p_diff.add_argument("run_id_2", help="Second run ID")
    p_diff.add_argument("--results-dir", default="benchmarks/results")

    # --- history subcommand ---
    p_history = sub.add_parser("history", help="Performance trend")
    p_history.add_argument("--op", required=True, help="Operator name")
    p_history.add_argument("--shape", default=None, help="Shape tag filter")
    p_history.add_argument("--results-dir", default="benchmarks/results")

    # --- run options (default when no subcommand) ---
    parser.add_argument("--bl", type=int, default=None,
                        help="Benchmark Level (1-6, default: 2)")
    parser.add_argument("--ot", type=str, default=None,
                        help="Operator Tier filter (0-4, comma-separated)")
    parser.add_argument("--st", type=str, default=None,
                        help="Shape Tier filter (1-4, comma-separated)")
    parser.add_argument("--layer", type=str, default=None,
                        help="Evaluation Layer (L1, L2, L3)")
    parser.add_argument("--op", type=str, default=None,
                        help="Specific operator(s) (comma-separated)")
    parser.add_argument("--shapes", type=str, default=None,
                        help="Specific shape tags (comma-separated)")
    parser.add_argument("--baselines", type=str, default=None,
                        help="Baseline methods (comma-separated, or 'all')")
    parser.add_argument("--model", type=str, default=None,
                        help="Model for L3/BL6 (e.g. gpt2)")
    parser.add_argument("--all", action="store_true",
                        help="Run all layers (equivalent to --bl 6)")
    parser.add_argument("--report", action="store_true",
                        help="Generate report (legacy, use 'arke bench report')")
    parser.add_argument("--results-dir", type=str, default="benchmarks/results")
    parser.add_argument("--output", type=str, default="benchmarks/results")
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--seq-len", type=str, default=None,
                        help="Seq lengths for L3 (comma-separated)")
    parser.add_argument("-v", "--verbose", action="store_true")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Route subcommands
    if args.subcmd == "report":
        _cmd_report(args)
        return
    if args.subcmd == "diff":
        _cmd_diff(args)
        return
    if args.subcmd == "history":
        _cmd_history(args)
        return

    # Legacy --report flag
    if getattr(args, "report", False):
        from benchmarks.report import generate_report

        report = generate_report(
            results_dir=Path(args.results_dir),
            output_path=Path(args.results_dir) / "report.md",
        )
        print(report)
        return

    # No subcommand and no run args → show help
    if not any([
        args.bl, args.ot, args.st, args.layer, args.op,
        getattr(args, "all", False),
    ]):
        # Default to BL2
        pass

    # Resolve config
    config = resolve_config(args)

    logger.info(
        f"Benchmark config: BL={config['bl']} OT={config['ot_tiers']} "
        f"ST={config['st_tiers']} Layers={config['layers']} Ops={config['ops']}"
    )

    # Create run directory
    run_id = _generate_run_id(bl=config["bl"], layer=config["layers"][0])
    run_dir = Path(args.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write provenance
    _write_config(run_dir, {
        "run_id": run_id,
        "bl": config["bl"],
        "ot_tiers": config["ot_tiers"],
        "st_tiers": config["st_tiers"],
        "layers": config["layers"],
        "ops": config["ops"],
        "shapes": config["shapes"],
        "baselines": config["baselines"],
        "warmup": config["warmup"],
        "reps": config["reps"],
        "timestamp": datetime.datetime.now().isoformat(),
    })
    _write_hardware(run_dir)

    # Execute layers
    for layer in config["layers"]:
        if layer == "L1":
            _run_l1(config, args.output)
        elif layer == "L2":
            _run_l2(config, args.output)
        elif layer == "L3":
            _run_l3(config, args.output)
        else:
            logger.error(f"Unknown layer: {layer}")
            sys.exit(1)

    # Post-processing
    _merge_perf_all(run_dir)
    _write_summary(run_dir)

    # Generate report
    try:
        from benchmarks.report import generate_report

        generate_report(
            results_dir=Path(args.output),
            output_path=run_dir / "report.md",
        )
        logger.info(f"Report: {run_dir / 'report.md'}")
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")

    logger.info(f"Benchmark complete. Results: {run_dir}")


if __name__ == "__main__":
    main()
