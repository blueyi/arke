# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""``python -m benchmarks status`` CLI for resume-aware progress reporting."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from benchmarks import progress as _progress

L1_KEY_FIELDS = ("op", "shape_tag", "baseline")
L2_KEY_FIELDS = ("op", "shape_tag", "approach")


def _layer_key_fields(layer: str) -> tuple[str, ...]:
    if layer.lower() == "l1":
        return L1_KEY_FIELDS
    if layer.lower() == "l2":
        return L2_KEY_FIELDS
    return ("op", "shape_tag")


def _resolve_layer_dir(
    output_root: Path,
    phase: int,
    stage: int,
    track: int | str,
    layer: str,
) -> Path:
    canonical = _progress.normalize_output_root(
        output_root, phase=phase, stage=stage, track=track, layer=layer.lower()
    )
    return (
        canonical
        / f"phase{phase}"
        / f"stage{stage}"
        / f"track{track}"
        / layer.lower()
    )


def _read_recent_events(layer_dir: Path, limit: int = 10) -> list[dict]:
    log_path = layer_dir / _progress.PROGRESS_LOG_NAME
    if not log_path.exists():
        return []
    events: list[dict] = []
    try:
        with log_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return events[-limit:]


def _human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    if seconds < 86400:
        return f"{seconds/3600:.1f}h"
    return f"{seconds/86400:.1f}d"


def render_status(layer_dir: Path, layer: str, *, recent: int = 10) -> str:
    if not layer_dir.exists():
        return f"[no benchmark output yet at {layer_dir}]"

    key_fields = _layer_key_fields(layer)

    # Per-op summary across CSVs.
    csvs = sorted(layer_dir.glob("*_results.csv"))
    if not csvs:
        return f"[no *_results.csv under {layer_dir}]"

    lines: list[str] = []
    lines.append(f"== Benchmark status: {layer_dir} ==")

    config_path = layer_dir / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            lines.append(
                f"  config ts={cfg.get('timestamp')}  "
                f"ops={len(cfg.get('ops') or [])}  "
                f"warmup={cfg.get('warmup')} reps={cfg.get('reps')} "
                f"tier={cfg.get('tier')}"
            )
            lines.append(f"  fingerprint={_progress.compute_fingerprint(cfg)}")
        except (OSError, json.JSONDecodeError):
            pass

    lock = _progress.lock_status(layer_dir)
    if lock is None:
        lines.append("  lock: <none>  (no live benchmark process)")
    else:
        live = "ALIVE" if lock["alive"] else "stale"
        age = _human_duration(time.time() - float(lock.get("started_at") or 0))
        lines.append(
            f"  lock: pid={lock['pid']} {live} host={lock.get('host','')} "
            f"started={age} ago"
        )

    total_ok = total_perm = total_retry = total_other = total_rows = 0
    lines.append("")
    lines.append(f"  {'op':28s} {'rows':>6s} {'ok':>6s} {'perm':>6s} {'retry':>6s} {'other':>6s}")
    lines.append("  " + "-" * 66)
    for csv_path in csvs:
        op = csv_path.stem.removesuffix("_results")
        summary = _progress.summarize_csv(csv_path, key_fields)
        total_ok += summary["ok"]
        total_perm += summary["permanent_failure"]
        total_retry += summary["retryable_failure"]
        total_other += summary["other"]
        total_rows += summary["rows"]
        lines.append(
            f"  {op:28s} {summary['rows']:>6d} "
            f"{summary['ok']:>6d} {summary['permanent_failure']:>6d} "
            f"{summary['retryable_failure']:>6d} {summary['other']:>6d}"
        )
    lines.append("  " + "-" * 66)
    lines.append(
        f"  {'TOTAL':28s} {total_rows:>6d} "
        f"{total_ok:>6d} {total_perm:>6d} {total_retry:>6d} {total_other:>6d}"
    )

    events = _read_recent_events(layer_dir, limit=recent)
    if events:
        lines.append("")
        lines.append(f"  recent events (last {len(events)}):")
        now = time.time()
        for ev in events:
            ts = float(ev.get("ts", 0))
            age = _human_duration(now - ts) if ts else "?"
            kind = ev.get("event", "?")
            extra: list[str] = []
            for k in ("op", "shape_tag", "baseline", "approach", "status", "skipped", "new", "total"):
                if k in ev:
                    extra.append(f"{k}={ev[k]}")
            lines.append(f"    -{age:>6s}  {kind:14s} {' '.join(extra)}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks status",
        description="Show benchmark progress / resume status.",
    )
    parser.add_argument(
        "--output", default="benchmarks/results",
        help="Benchmark output root (defaults to benchmarks/results)",
    )
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument(
        "--track", default="6",
        help="Track id (int or string, e.g. 6, g6, 1)",
    )
    parser.add_argument(
        "--layer", choices=["l1", "l2", "L1", "L2"], default="l2",
    )
    parser.add_argument("--recent", type=int, default=10)
    parser.add_argument("--json", action="store_true",
        help="Emit machine-readable JSON instead of the text report.")
    args = parser.parse_args(argv)

    layer = args.layer.lower()
    track: int | str
    try:
        track = int(args.track)
    except ValueError:
        track = args.track

    layer_dir = _resolve_layer_dir(
        Path(args.output), args.phase, args.stage, track, layer
    )

    if args.json:
        key_fields = _layer_key_fields(layer)
        per_op = {}
        for csv_path in sorted(layer_dir.glob("*_results.csv")):
            op = csv_path.stem.removesuffix("_results")
            per_op[op] = _progress.summarize_csv(csv_path, key_fields)
        payload = {
            "layer_dir": str(layer_dir),
            "layer": layer,
            "lock": _progress.lock_status(layer_dir),
            "per_op": per_op,
            "recent_events": _read_recent_events(layer_dir, limit=args.recent),
        }
        print(json.dumps(payload, indent=2, default=str))
        return 0

    print(render_status(layer_dir, layer, recent=args.recent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
