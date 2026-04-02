# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Report generator — reads CSVs from results/ and produces report.md.

Includes scoring ratios vs P0 (vendor) and P1 (expert Triton) baselines.

Usage:
    python -m benchmarks.report
    python -m benchmarks.report --results-dir benchmarks/results
    python -m benchmarks.report --output benchmarks/report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)


# ── CSV reading ─────────────────────────────────────────────


def _find_latest_dir(base: Path, layer: str) -> Path | None:
    """Find the latest timestamped directory for a layer."""
    layer_dir = base / layer
    if not layer_dir.is_dir():
        return None
    subdirs = sorted(
        [d for d in layer_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    return subdirs[0] if subdirs else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_l1_results(results_dir: Path) -> list[dict[str, str]]:
    """Read all L1 CSVs from the latest run."""
    latest = _find_latest_dir(results_dir, "L1")
    if latest is None:
        return []
    rows: list[dict[str, str]] = []
    for csv_path in sorted(latest.glob("*_results.csv")):
        rows.extend(_read_csv(csv_path))
    return rows


def _read_l2_results(results_dir: Path) -> list[dict[str, str]]:
    """Read all L2 CSVs from the latest run."""
    latest = _find_latest_dir(results_dir, "L2")
    if latest is None:
        return []
    rows: list[dict[str, str]] = []
    for csv_path in sorted(latest.glob("*_results.csv")):
        rows.extend(_read_csv(csv_path))
    return rows


def _read_l3_results(results_dir: Path) -> list[dict[str, str]]:
    """Read L3 CSV from the latest run."""
    latest = _find_latest_dir(results_dir, "L3")
    if latest is None:
        return []
    csv_path = latest / "gpt2_results.csv"
    if csv_path.exists():
        return _read_csv(csv_path)
    return []


def _read_sources(results_dir: Path, layer: str) -> dict:
    """Read sources.json from the latest run."""
    latest = _find_latest_dir(results_dir, layer)
    if latest is None:
        return {}
    src_path = latest / "sources.json"
    if src_path.exists():
        with open(src_path) as f:
            return json.load(f)
    return {}


# ── Scoring ─────────────────────────────────────────────────


def _compute_l1_scores(rows: list[dict[str, str]]) -> dict:
    """Compute per-op scores: ratio of baseline latency / Arke latency.

    Returns dict[op][shape_tag] = {baseline: ratio, ...}
    and aggregate scores.
    """
    # Group: op → shape_tag → baseline → latency
    grouped: dict[str, dict[str, dict[str, float]]] = {}
    for r in rows:
        op = r["op"]
        tag = r["shape_tag"]
        baseline = r.get("baseline", r.get("approach", "unknown"))
        try:
            lat = float(r["latency_us"])
        except (ValueError, KeyError):
            continue
        grouped.setdefault(op, {}).setdefault(tag, {})[baseline] = lat

    scores: dict[str, dict[str, dict[str, float]]] = {}
    op_averages: dict[str, dict[str, float]] = {}

    for op, shapes in grouped.items():
        scores[op] = {}
        ratios_by_baseline: dict[str, list[float]] = {}

        for tag, baselines in shapes.items():
            scores[op][tag] = {}
            for bl_name, bl_lat in baselines.items():
                # For each other baseline, compute ratio: bl_lat / this_lat
                for other_name, other_lat in baselines.items():
                    if other_name != bl_name and other_lat > 0:
                        ratio = bl_lat / other_lat
                        key = f"{bl_name}_vs_{other_name}"
                        scores[op][tag][key] = ratio

            # vs_P0 and vs_P1 ratios for each baseline
            p0_names = {"cuBLAS/cuDNN"}
            p1_names = {"FlagGems", "Liger-Kernel"}

            p0_lat = None
            for n in p0_names:
                if n in baselines:
                    p0_lat = baselines[n]
                    break

            p1_lat = None
            for n in p1_names:
                if n in baselines:
                    p1_lat = baselines[n]
                    break

            for bl_name, bl_lat in baselines.items():
                if p0_lat and p0_lat > 0:
                    r = p0_lat / bl_lat
                    scores[op][tag][f"{bl_name}_vs_P0"] = r
                    ratios_by_baseline.setdefault(f"{bl_name}_vs_P0", []).append(r)
                if p1_lat and p1_lat > 0:
                    r = p1_lat / bl_lat
                    scores[op][tag][f"{bl_name}_vs_P1"] = r
                    ratios_by_baseline.setdefault(f"{bl_name}_vs_P1", []).append(r)

        # Geometric mean of ratios per baseline pair
        op_averages[op] = {}
        for key, vals in ratios_by_baseline.items():
            if vals:
                geo = math.exp(sum(math.log(v) for v in vals if v > 0) / len(vals))
                op_averages[op][key] = geo

    return {"per_shape": scores, "averages": op_averages}


# ── Report generation ───────────────────────────────────────


def generate_report(
    results_dir: Path,
    output_path: Path | None = None,
) -> str:
    """Generate a markdown report from benchmark results."""
    lines: list[str] = []

    lines.append("# Arke Benchmark Report")
    lines.append("")
    lines.append(
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    lines.append("")

    # ── Hardware ──
    hw_data = None
    for layer in ("L1", "L2", "L3"):
        latest = _find_latest_dir(results_dir, layer)
        if latest:
            hw_path = latest / "hardware.json"
            if hw_path.exists():
                with open(hw_path) as f:
                    hw_data = json.load(f)
                break

    if hw_data:
        lines.append("## Hardware")
        lines.append("")
        lines.append(f"- **GPU:** {hw_data.get('gpu_name', 'N/A')}")
        lines.append(
            f"- **GPU Memory:** {hw_data.get('gpu_memory_mb', 0)} MB"
        )
        lines.append(f"- **CUDA:** {hw_data.get('cuda_version', 'N/A')}")
        lines.append(
            f"- **PyTorch:** {hw_data.get('torch_version', 'N/A')}"
        )
        lines.append(
            f"- **Triton:** {hw_data.get('triton_version', 'N/A')}"
        )
        lines.append("")

    # ── L1 Results ──
    l1_rows = _read_l1_results(results_dir)
    if l1_rows:
        lines.append("## L1: Single Operator Results")
        lines.append("")
        lines.extend(_format_l1_table(l1_rows))
        lines.append("")

        # Scoring
        l1_scores = _compute_l1_scores(l1_rows)
        if l1_scores["averages"]:
            lines.append("### L1 Scoring (geometric mean ratios)")
            lines.append("")
            lines.append("| Op | Metric | Ratio |")
            lines.append("|:---|:-------|------:|")
            for op, metrics in sorted(l1_scores["averages"].items()):
                for metric, val in sorted(metrics.items()):
                    indicator = "🟢" if val >= 0.9 else "🟡" if val >= 0.8 else "🔴"
                    lines.append(
                        f"| {op} | {metric} | {indicator} {val:.3f} |"
                    )
            lines.append("")

    # ── L1 Sources ──
    l1_sources = _read_sources(results_dir, "L1")
    if l1_sources:
        lines.append("### L1 Baseline Sources")
        lines.append("")
        for name, info in l1_sources.items():
            src = info.get("source", "N/A")
            pri = info.get("priority", "?")
            lines.append(f"- **{name}** ({pri}): {src}")
        lines.append("")

    # ── L2 Results ──
    l2_rows = _read_l2_results(results_dir)
    if l2_rows:
        lines.append("## L2: Fused Operator Results")
        lines.append("")
        lines.extend(_format_l2_table(l2_rows))
        lines.append("")

    # ── L3 Results ──
    l3_rows = _read_l3_results(results_dir)
    if l3_rows:
        lines.append("## L3: E2E Model Results (GPT-2 Small)")
        lines.append("")
        lines.extend(_format_l3_table(l3_rows))
        lines.append("")

        # L3 scoring
        lines.extend(_format_l3_scoring(l3_rows))
        lines.append("")

    # ── Summary ──
    lines.append("## Summary")
    lines.append("")
    if l1_rows:
        ops = sorted({r["op"] for r in l1_rows})
        baselines = sorted({r.get("baseline", "?") for r in l1_rows})
        lines.append(
            f"- **L1:** {len(ops)} operators, "
            f"{len(baselines)} baselines, "
            f"{len(l1_rows)} measurements"
        )
    if l2_rows:
        fused_ops = sorted({r["op"] for r in l2_rows})
        lines.append(
            f"- **L2:** {len(fused_ops)} fused ops, "
            f"{len(l2_rows)} measurements"
        )
    if l3_rows:
        seq_lens = sorted({int(r["seq_len"]) for r in l3_rows})
        lines.append(
            f"- **L3:** GPT-2 Small, seq_lens={seq_lens}, "
            f"{len(l3_rows)} measurements"
        )
    lines.append("")
    lines.append("---")
    lines.append("*Report generated by Arke Benchmark System*")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report)
        logger.info(f"Report saved to: {output_path}")

    return report


# ── Table formatters ────────────────────────────────────────


def _format_l1_table(rows: list[dict[str, str]]) -> list[str]:
    """Format L1 results as markdown tables grouped by op."""
    lines: list[str] = []

    # Group by op
    ops: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        ops.setdefault(r["op"], []).append(r)

    for op, op_rows in sorted(ops.items()):
        lines.append(f"### {op}")
        lines.append("")

        # Collect baselines
        baselines = list(dict.fromkeys(r["baseline"] for r in op_rows))

        lines.append(
            "| Shape | " + " | ".join(baselines) + " |"
        )
        lines.append(
            "|:---" + "|---:" * len(baselines) + "|"
        )

        # Group by shape
        shapes: dict[str, dict[str, str]] = {}
        for r in op_rows:
            tag = r["shape_tag"]
            bl = r["baseline"]
            lat = r["latency_us"]
            shapes.setdefault(tag, {})[bl] = lat

        for tag, bl_lats in shapes.items():
            cells = []
            for bl in baselines:
                if bl in bl_lats:
                    cells.append(f"{bl_lats[bl]} μs")
                else:
                    cells.append("N/A")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")

        lines.append("")

    return lines


def _format_l2_table(rows: list[dict[str, str]]) -> list[str]:
    """Format L2 results as markdown tables grouped by op."""
    lines: list[str] = []

    ops: dict[str, list[dict[str, str]]] = {}
    for r in rows:
        ops.setdefault(r["op"], []).append(r)

    for op, op_rows in sorted(ops.items()):
        lines.append(f"### {op}")
        lines.append("")

        approaches = list(dict.fromkeys(r["approach"] for r in op_rows))

        lines.append(
            "| Shape | " + " | ".join(approaches) + " |"
        )
        lines.append(
            "|:---" + "|---:" * len(approaches) + "|"
        )

        shapes: dict[str, dict[str, str]] = {}
        for r in op_rows:
            tag = r["shape_tag"]
            approach = r["approach"]
            lat = r["latency_us"]
            shapes.setdefault(tag, {})[approach] = lat

        for tag, appr_lats in shapes.items():
            cells = []
            for a in approaches:
                if a in appr_lats:
                    cells.append(f"{appr_lats[a]} μs")
                else:
                    cells.append("N/A")
            lines.append(f"| {tag} | " + " | ".join(cells) + " |")

        lines.append("")

    return lines


def _format_l3_table(rows: list[dict[str, str]]) -> list[str]:
    """Format L3 results as a markdown table."""
    lines: list[str] = []

    lines.append(
        "| SeqLen | Mode | Mean (ms) | Min (ms) | Memory (MB) | "
        "Correct | Top-1 Match |"
    )
    lines.append("|---:|:---|---:|---:|---:|:---:|:---:|")

    for r in rows:
        correct = "✅" if r.get("correct", "False") == "True" else "❌"
        top1 = "✅" if r.get("top1_match", "False") == "True" else "❌"
        lines.append(
            f"| {r['seq_len']} | {r['mode']} | {r['mean_ms']} | "
            f"{r['min_ms']} | {r['peak_memory_mb']} | {correct} | {top1} |"
        )

    return lines


def _format_l3_scoring(rows: list[dict[str, str]]) -> list[str]:
    """Compute and format L3 scoring: Arke vs eager ratios."""
    lines: list[str] = []
    lines.append("### L3 Scoring")
    lines.append("")

    # Group by seq_len
    by_seq: dict[int, dict[str, float]] = {}
    for r in rows:
        seq = int(r["seq_len"])
        mode = r["mode"]
        try:
            mean_ms = float(r["mean_ms"])
        except (ValueError, KeyError):
            continue
        by_seq.setdefault(seq, {})[mode] = mean_ms

    lines.append("| SeqLen | eager (ms) | compile (ms) | arke (ms) | "
                  "arke/eager | compile/eager |")
    lines.append("|---:|---:|---:|---:|---:|---:|")

    for seq in sorted(by_seq):
        modes = by_seq[seq]
        eager = modes.get("eager", 0)
        compile_ms = modes.get("torch.compile")
        arke_ms = modes.get("arke")

        eager_str = f"{eager:.2f}"
        comp_str = f"{compile_ms:.2f}" if compile_ms else "N/A"
        arke_str = f"{arke_ms:.2f}" if arke_ms else "N/A"

        arke_ratio = f"{arke_ms / eager:.2f}x" if arke_ms and eager > 0 else "N/A"
        comp_ratio = (
            f"{compile_ms / eager:.2f}x" if compile_ms and eager > 0 else "N/A"
        )

        lines.append(
            f"| {seq} | {eager_str} | {comp_str} | {arke_str} | "
            f"{arke_ratio} | {comp_ratio} |"
        )

    return lines


# ── CLI ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Arke Benchmark Report"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="benchmarks/results",
        help="Path to results directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/report.md",
        help="Output report path",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    report = generate_report(
        results_dir=Path(args.results_dir),
        output_path=Path(args.output),
    )
    print(report)


if __name__ == "__main__":
    main()
