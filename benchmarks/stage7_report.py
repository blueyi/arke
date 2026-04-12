# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def collect_stage7_operator_shape_stats(base_dir: str | Path) -> dict:
    base = Path(base_dir)
    report: dict[str, dict] = {}

    for perf_all in sorted(base.glob("**/PERF_ALL.csv")):
        layer = perf_all.parent.name
        layer_stats: dict[str, dict] = defaultdict(lambda: {
            "shape_tags": set(),
            "rows": 0,
            "status_counts": defaultdict(int),
        })
        for row in csv.DictReader(perf_all.open()):
            op = row.get("operator", "unknown")
            shape_tag = row.get("shape_tag", "unknown")
            status = row.get("status", "ok")
            stat = layer_stats[op]
            stat["shape_tags"].add(shape_tag)
            stat["rows"] += 1
            stat["status_counts"][status] += 1

        report[layer] = {}
        for op, stat in sorted(layer_stats.items()):
            report[layer][op] = {
                "shape_count": len(stat["shape_tags"]),
                "shape_tags": sorted(stat["shape_tags"]),
                "rows": stat["rows"],
                "status_counts": dict(sorted(stat["status_counts"].items())),
            }

    return report


def write_stage7_operator_shape_stats(base_dir: str | Path, out_path: str | Path | None = None) -> Path:
    base = Path(base_dir)
    out = Path(out_path) if out_path is not None else base / "stage7_operator_shape_stats.json"
    out.write_text(json.dumps(collect_stage7_operator_shape_stats(base), indent=2))
    return out
