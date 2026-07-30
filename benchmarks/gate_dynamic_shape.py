# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""D2 soft gate for the dynamic-shape cliff track.

Frozen-layer status (Leon approvals)
------------------------------------
* 2026-07-29: D1 measure-only approved (track ships curves, no pass/fail).
* 2026-07-30: **"D推进D2并完成依赖"** — D2 soft gate approved. The threshold
  below is therefore a Leon-approved frozen parameter:

      SAME_SPEC_GEOMEAN_LIMIT = 5.0

Architecture
------------
``benchmarks/dynamic_shape.py`` is the measurement track and is hard-guarded
against ever baking in a pass/fail threshold
(``test_no_gate_threshold_in_module``). This module is the *consumer* that
wraps the track and applies the D2 semantics — exactly the split the track's
design contract prescribes. Keep thresholds HERE, never in the track.

D2 semantics (per op)
---------------------
1. **same_spec_geomean <= 5.0** — shapes predicted warm (their ``spec_key``
   was already seen in the sweep) must be compile-free on first call; the
   geomean of their cliff ratios staying under 5x asserts the bucket / memo
   machinery works. 5x (not ~1x) leaves room for host-side launch jitter and
   GPU clock spin-up on small shapes, which cross-run variance data shows can
   push individual warm first-calls to a few x (see
   docs/benchmark/dynamic-shape-cliff.md §5 variance table).
2. **new-spec prediction consistency** — the recorded ``new_spec`` flags must
   equal a recount from first-occurrence of ``spec_key`` over the rows, and
   ``n_new_spec``/``n_same_spec`` in summary.json must match the rows. This
   catches accidental despecialization two ways: (a) a launcher/template
   change that adds an unpredicted specialization axis shows up as
   predicted-warm shapes paying compile-scale cliffs -> check 1 trips; (b) a
   spec_key regression that stops predicting compiles shows up as a
   count/flag mismatch -> this check trips.

Usage
-----
    python -m benchmarks.gate_dynamic_shape <run_dir> [<run_dir> ...]

where each ``run_dir`` is a ``benchmarks.dynamic_shape --out`` directory
(contains ``summary.json`` + ``<op>_cliff.csv``). Multiple run dirs are
evaluated independently; the gate verdict is per (run, op) and the overall
exit code is 0 only if every (run, op) passes.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Frozen parameter — Leon-approved 2026-07-30 ("D推进D2并完成依赖").
# Do NOT change without project-lead approval.
SAME_SPEC_GEOMEAN_LIMIT = 5.0


@dataclass
class OpVerdict:
    op: str
    passed: bool
    same_spec_geomean: float
    n_new_spec: int
    n_same_spec: int
    reasons: list[str] = field(default_factory=list)


def _geomean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v) and v > 0]
    if not clean:
        return float("nan")
    return math.exp(sum(math.log(v) for v in clean) / len(clean))


def evaluate_op(rows: list[dict], summary: dict | None = None) -> OpVerdict:
    """Apply D2 semantics to one op's cliff rows (dicts from <op>_cliff.csv).

    ``summary`` is the op's entry from summary.json when available; its
    n_new_spec / n_same_spec are cross-checked against the rows.
    """
    op = rows[0]["op"] if rows else "?"
    reasons: list[str] = []

    ok_rows = [r for r in rows if r.get("status", "ok") == "ok"]

    # ── Check 2a: recorded new_spec flags == first-occurrence recount ──────
    seen: set[str] = set()
    recount_flags: list[bool] = []
    for r in rows:
        spec = r["spec_key"]
        recount_flags.append(spec not in seen)
        seen.add(spec)
    recorded_flags = [str(r["new_spec"]).lower() == "true" for r in rows]
    if recorded_flags != recount_flags:
        bad = [i for i, (a, b) in enumerate(zip(recorded_flags, recount_flags)) if a != b]
        reasons.append(
            f"new_spec flags disagree with spec_key first-occurrence recount "
            f"at row indices {bad[:5]} (despecialization or spec_key regression)"
        )

    new_rows = [r for r, flag in zip(rows, recorded_flags) if flag and r in ok_rows]
    same_rows = [r for r, flag in zip(rows, recorded_flags) if not flag and r in ok_rows]

    # ── Check 2b: summary counts match the rows ────────────────────────────
    if summary is not None:
        if int(summary.get("n_new_spec", -1)) != len(new_rows):
            reasons.append(
                f"summary n_new_spec={summary.get('n_new_spec')} != rows {len(new_rows)}"
            )
        if int(summary.get("n_same_spec", -1)) != len(same_rows):
            reasons.append(
                f"summary n_same_spec={summary.get('n_same_spec')} != rows {len(same_rows)}"
            )

    # ── Check 1: same-spec (predicted warm) geomean under the frozen limit ─
    same_geo = _geomean([float(r["cliff_ratio"]) for r in same_rows])
    if same_rows and math.isfinite(same_geo) and same_geo > SAME_SPEC_GEOMEAN_LIMIT:
        reasons.append(
            f"same_spec_geomean {same_geo:.2f} > {SAME_SPEC_GEOMEAN_LIMIT} "
            f"(predicted-warm shapes are paying compile-scale cliffs)"
        )
    if same_rows and not math.isfinite(same_geo):
        reasons.append("same_spec_geomean not finite (bad rows)")

    return OpVerdict(
        op=op,
        passed=not reasons,
        same_spec_geomean=same_geo,
        n_new_spec=len(new_rows),
        n_same_spec=len(same_rows),
        reasons=reasons,
    )


def evaluate_run(run_dir: Path) -> list[OpVerdict]:
    """Evaluate every <op>_cliff.csv in a dynamic_shape run directory."""
    run_dir = Path(run_dir)
    summary_all: dict = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary_all = json.loads(summary_path.read_text(encoding="utf-8")).get("ops", {})

    verdicts: list[OpVerdict] = []
    for csv_path in sorted(run_dir.glob("*_cliff.csv")):
        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            verdicts.append(OpVerdict(op=csv_path.stem.replace("_cliff", ""),
                                      passed=False,
                                      same_spec_geomean=float("nan"),
                                      n_new_spec=0, n_same_spec=0,
                                      reasons=["empty cliff csv"]))
            continue
        op = rows[0]["op"]
        verdicts.append(evaluate_op(rows, summary_all.get(op)))
    return verdicts


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    all_pass = True
    for run in args:
        verdicts = evaluate_run(Path(run))
        if not verdicts:
            print(f"[gate-D2] {run}: no *_cliff.csv found — FAIL")
            all_pass = False
            continue
        for v in verdicts:
            mark = "PASS" if v.passed else "FAIL"
            print(f"[gate-D2] {run} :: {v.op:<10} {mark}  "
                  f"same_spec_geomean={v.same_spec_geomean:.2f} "
                  f"(limit {SAME_SPEC_GEOMEAN_LIMIT}) "
                  f"new={v.n_new_spec} same={v.n_same_spec}")
            for r in v.reasons:
                print(f"          ! {r}")
            all_pass = all_pass and v.passed
    print(f"[gate-D2] overall: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
