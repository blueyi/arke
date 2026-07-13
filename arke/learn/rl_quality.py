# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Arke Learn — D3/M3: RL corpus quality gates.

Leon-approved 2026-07-13: M3 ensures the corpus mined by M1/M2 is
quality-checked before it feeds M4 (SFT export) or M5 (RL fine-tune).

Four checks, each returns pass/fail + diagnostics:
  1. **Schema sanity** — every sample is valid JSON with required fields.
  2. **Deduplication** — trajectories keyed by (op, shape_hash, decisions_hash)
     are deduplicated; identical re-runs don't inflate the corpus.
  3. **Reward distribution** — at least *min_beat* trajectories have
     final_reward >= 2 (beat-baseline signal); a corpus that is 100% reward=1
     has no RL gradient.
  4. **Tier coverage** — each requested op-tier has at least *min_per_tier*
     trajectories; an under-covered tier means the policy won't generalize.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Schema validation ──────────────────────────────────────────────
_STEP_REQUIRED = {"type", "op", "action", "reward"}
_TRAJ_REQUIRED = {"type", "op", "decisions", "final_reward"}


def _validate_sample(sample: dict[str, Any]) -> str | None:
    """Return an error string if the sample fails schema sanity, else None."""
    stype = sample.get("type")
    if stype == "step":
        missing = _STEP_REQUIRED - set(sample)
        if missing:
            return f"step missing fields: {missing}"
        if not isinstance(sample.get("reward"), (int, float)):
            return f"step reward is not numeric: {sample.get('reward')!r}"
    elif stype == "trajectory":
        missing = _TRAJ_REQUIRED - set(sample)
        if missing:
            return f"trajectory missing fields: {missing}"
        if not isinstance(sample.get("decisions"), list):
            return f"trajectory decisions is not a list"
        if not isinstance(sample.get("final_reward"), (int, float)):
            return f"trajectory final_reward not numeric"
    else:
        return f"unknown sample type: {stype!r}"
    return None


# ── Deduplication ──────────────────────────────────────────────────
def _traj_key(sample: dict[str, Any]) -> str:
    """Deterministic dedup key for a trajectory sample."""
    op = sample.get("op", "")
    shape = json.dumps(sample.get("shape", {}), sort_keys=True)
    decisions = json.dumps(sample.get("decisions", []), sort_keys=True, default=str)
    raw = f"{op}|{shape}|{decisions}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def deduplicate(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate trajectory samples by (op, shape, decisions).

    Step samples are kept as-is (they inherit dedup from their parent traj).
    Returns (deduped_samples, n_removed).
    """
    seen_keys: set[str] = set()
    out: list[dict[str, Any]] = []
    removed = 0
    for s in samples:
        if s.get("type") == "trajectory":
            key = _traj_key(s)
            if key in seen_keys:
                removed += 1
                continue
            seen_keys.add(key)
        out.append(s)
    return out, removed


# ── Quality report ─────────────────────────────────────────────────
@dataclass
class QualityCheck:
    name: str
    passed: bool
    detail: str = ""

@dataclass
class QualityReport:
    """Result of running all M3 quality gates on a corpus."""
    checks: list[QualityCheck] = field(default_factory=list)
    total_samples: int = 0
    total_steps: int = 0
    total_trajectories: int = 0
    dedup_removed: int = 0
    reward_histogram: dict[int, int] = field(default_factory=dict)
    tier_coverage: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"RL Corpus Quality Gate: {status}"]
        lines.append(f"  samples: {self.total_samples} "
                     f"({self.total_steps} step + {self.total_trajectories} traj)")
        lines.append(f"  dedup removed: {self.dedup_removed}")
        lines.append(f"  reward histogram: {self.reward_histogram}")
        lines.append(f"  tier coverage: {self.tier_coverage}")
        for c in self.checks:
            mark = "✅" if c.passed else "❌"
            lines.append(f"  {mark} {c.name}: {c.detail}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "total_samples": self.total_samples,
            "total_steps": self.total_steps,
            "total_trajectories": self.total_trajectories,
            "dedup_removed": self.dedup_removed,
            "reward_histogram": self.reward_histogram,
            "tier_coverage": self.tier_coverage,
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                        for c in self.checks],
        }


def quality_gate(
    corpus_path: str | Path,
    *,
    min_beat: int = 1,
    min_per_tier: int = 0,
    tiers: list[str] | None = None,
) -> QualityReport:
    """Run all M3 quality gates on a corpus JSONL file.

    Parameters
    ----------
    corpus_path
        Path to the RL corpus JSONL (produced by ``build_rl_dataset``).
    min_beat
        Minimum number of trajectories with ``final_reward >= 2`` (beat-
        baseline signal). Default 1 — at least one trajectory must beat.
    min_per_tier
        Minimum trajectories per op-tier. Default 0 (disabled).
    tiers
        Op-tier names to check coverage for (e.g. ``['matmul', 'softmax']``).
        If None, all ops in the corpus are checked.

    Returns
    -------
    QualityReport
        With ``passed`` True only if ALL checks pass.
    """
    corpus_path = Path(corpus_path)
    report = QualityReport()

    # Load samples
    if not corpus_path.exists():
        report.checks.append(QualityCheck("file_exists", False, f"{corpus_path} not found"))
        return report

    samples: list[dict[str, Any]] = []
    schema_errors: list[str] = []
    for i, line in enumerate(corpus_path.read_text().splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            s = json.loads(line)
        except json.JSONDecodeError:
            schema_errors.append(f"line {i+1}: invalid JSON")
            continue
        err = _validate_sample(s)
        if err:
            schema_errors.append(f"line {i+1}: {err}")
        samples.append(s)

    # 1. Schema sanity
    if schema_errors:
        report.checks.append(QualityCheck(
            "schema_sanity", False,
            f"{len(schema_errors)} error(s): {'; '.join(schema_errors[:5])}"))
    else:
        report.checks.append(QualityCheck(
            "schema_sanity", True, f"{len(samples)} samples, all valid"))

    # Partition
    steps = [s for s in samples if s.get("type") == "step"]
    trajs = [s for s in samples if s.get("type") == "trajectory"]

    # 2. Deduplication
    deduped_trajs, n_removed = deduplicate(trajs)
    report.dedup_removed = n_removed
    report.checks.append(QualityCheck(
        "deduplication", True,
        f"{len(trajs)} trajectories → {len(deduped_trajs)} unique "
        f"({n_removed} duplicates removed)"))

    # 3. Reward distribution
    reward_hist: dict[int, int] = {}
    for t in deduped_trajs:
        r = int(t.get("final_reward", 0))
        reward_hist[r] = reward_hist.get(r, 0) + 1
    report.reward_histogram = reward_hist
    n_beat = sum(v for k, v in reward_hist.items() if k >= 2)
    beat_pass = n_beat >= min_beat
    report.checks.append(QualityCheck(
        "reward_distribution", beat_pass,
        f"{n_beat} beat-baseline (reward≥2) vs min={min_beat}; "
        f"histogram={reward_hist}"))

    # 4. Tier coverage
    tier_counts: dict[str, int] = {}
    for t in deduped_trajs:
        op = t.get("op", "unknown")
        tier_counts[op] = tier_counts.get(op, 0) + 1
    report.tier_coverage = tier_counts
    check_tiers = tiers or list(tier_counts.keys())
    under = {t: tier_counts.get(t, 0) for t in check_tiers
             if tier_counts.get(t, 0) < min_per_tier}
    if under and min_per_tier > 0:
        report.checks.append(QualityCheck(
            "tier_coverage", False,
            f"under min_per_tier={min_per_tier}: {under}"))
    else:
        report.checks.append(QualityCheck(
            "tier_coverage", True,
            f"{len(tier_counts)} ops covered: {tier_counts}"))

    report.total_samples = len(samples)
    report.total_steps = len(steps)
    report.total_trajectories = len(deduped_trajs)

    return report


__all__ = [
    "quality_gate", "QualityReport", "QualityCheck",
    "deduplicate", "_validate_sample",
]
