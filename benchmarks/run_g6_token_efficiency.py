#!/usr/bin/env python3
"""G6.4 Token Efficiency Benchmark.

Measures: lines(.ak kernel+strategy) vs lines(generated Triton kernel)
Goal: .ak code lines < Triton code lines at comparable performance.

Output: benchmarks/results/phase1/g6_token_efficiency.csv + .md summary
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from arke.pipeline import ArkePipeline


def count_lines(text: str, exclude_blanks: bool = True, exclude_comments: bool = True,
                comment_prefix: str = "#") -> int:
    """Count non-blank, non-comment lines."""
    lines = text.splitlines()
    count = 0
    for line in lines:
        stripped = line.strip()
        if not stripped and exclude_blanks:
            continue
        if stripped.startswith(comment_prefix) and exclude_comments:
            continue
        count += 1
    return count


def count_ak_lines(text: str) -> tuple[int, int]:
    """Return (total_lines, code_lines) for .ak source.

    Code lines = non-blank + non-comment (// prefix).
    """
    lines = text.splitlines()
    total = len(lines)
    code = sum(
        1 for l in lines
        if l.strip() and not l.strip().startswith("//")
    )
    return total, code


def analyze_ak_file(ak_path: str) -> dict:
    """Run pipeline + collect token efficiency metrics."""
    name = os.path.basename(ak_path)
    ak_src = open(ak_path).read()
    ak_total, ak_code = count_ak_lines(ak_src)

    # Count kernel-only vs kernel+strategy
    lines = ak_src.splitlines()
    in_strategy = False
    kernel_lines = []
    strategy_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("strategy ") or in_strategy:
            in_strategy = True
            if not stripped.startswith("//"):
                strategy_lines.append(line)
        elif not stripped.startswith("//") and stripped:
            kernel_lines.append(line)

    ak_kernel_code = len(kernel_lines)
    ak_strategy_code = len(strategy_lines)

    # Run pipeline with codegen=True
    result = ArkePipeline.from_ak_file(ak_path, target_hw="nvidia_ampere", codegen=True)

    triton_src = result.codegen_source or ""
    triton_total, _ = count_ak_lines(triton_src)  # use same counter
    triton_code = count_lines(triton_src, comment_prefix="#")

    ratio = ak_code / triton_code if triton_code > 0 else 0.0
    passes_g6 = ak_code < triton_code  # G6.4 criterion: .ak < Triton

    n_decisions = len(result.strategy_ir.get("decisions", []))
    source = result.strategy_source

    return {
        "file": name,
        "ak_total_lines": ak_total,
        "ak_code_lines": ak_code,
        "ak_kernel_lines": ak_kernel_code,
        "ak_strategy_lines": ak_strategy_code,
        "triton_total_lines": triton_total,
        "triton_code_lines": triton_code,
        "ratio_ak_to_triton": round(ratio, 3),
        "passes_g6_criterion": passes_g6,
        "strategy_decisions": n_decisions,
        "strategy_source": source,
        "correct": result.correct,
    }


def main():
    ak_files = sorted(glob.glob("docs/examples/operators/*.ak"))
    print(f"G6.4 Token Efficiency Benchmark — {len(ak_files)} operators")
    print("=" * 72)

    rows = []
    for f in ak_files:
        try:
            row = analyze_ak_file(f)
            rows.append(row)
            status = "✅" if row["passes_g6_criterion"] else "❌"
            print(
                f"  {status} {row['file']:<28} "
                f".ak={row['ak_code_lines']:3d}  triton={row['triton_code_lines']:3d}  "
                f"ratio={row['ratio_ak_to_triton']:.3f}  d={row['strategy_decisions']}"
            )
        except Exception as e:
            print(f"  ERR {f}: {e}")

    # Summary
    passes = sum(1 for r in rows if r["passes_g6_criterion"])
    total = len(rows)
    avg_ratio = sum(r["ratio_ak_to_triton"] for r in rows) / total if total else 0
    print()
    print(f"Pass: {passes}/{total}  avg ratio(ak/triton)={avg_ratio:.3f}")
    print(f"G6.4 criterion: .ak code lines < Triton code lines")

    # Write CSV
    out_dir = Path("benchmarks/results/phase1")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "g6_token_efficiency.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV: {csv_path}")

    # Write Markdown summary
    md_path = out_dir / "g6_token_efficiency.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# G6.4 Token Efficiency Report\n\n")
        f.write(f"**Pass rate:** {passes}/{total}  ")
        f.write(f"**Avg ratio (.ak/Triton):** {avg_ratio:.3f}\n\n")
        f.write("| File | .ak lines | Triton lines | Ratio | Pass |\n")
        f.write("|------|----------:|-------------:|------:|:----:|\n")
        for r in rows:
            icon = "✅" if r["passes_g6_criterion"] else "❌"
            f.write(
                f"| {r['file']} | {r['ak_code_lines']} | {r['triton_code_lines']} "
                f"| {r['ratio_ak_to_triton']:.3f} | {icon} |\n"
            )
        f.write("\n**G6.4 criterion:** `.ak` code lines < generated Triton code lines\n")
    print(f"MD:  {md_path}")

    return 0 if passes == total else 1


if __name__ == "__main__":
    sys.exit(main())
