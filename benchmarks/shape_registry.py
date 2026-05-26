# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Parse benchmark-shapes.md to extract the canonical shape catalog.

This module is the **single source of truth** bridge for shapes: it reads
all shape tables from ``docs/benchmark/benchmark-shapes.md`` and
exposes structured data that ``benchmarks/shapes.py`` and all test files
consume.

If benchmark-shapes.md is edited (shapes added/removed/modified),
every downstream consumer picks up the change automatically at import time.
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Locate the document ──────────────────────────────────────────────────

_THIS_DIR = Path(__file__).resolve().parent          # benchmarks/
_REPO_ROOT = _THIS_DIR.parent                         # arke repo root
_SHAPES_MD = _REPO_ROOT / "docs" / "benchmark" / "benchmark-shapes.md"

# ── Helpers ──────────────────────────────────────────────────────────────

_DASH_CHARS = {"—", "–", "-", "—"}  # em-dash, en-dash, hyphen, Unicode em-dash


def _coerce_value(val: str) -> int | float | str | None:
    """Convert a table cell value to int, float, str, or None."""
    val = val.strip()
    # Blank or dash-only → None
    if not val or val in _DASH_CHARS or val == "":
        return None
    # Strip backticks (tag names)
    if val.startswith("`") and val.endswith("`"):
        val = val[1:-1]
    # Strip warning emoji prefix
    val = val.lstrip("⚠️ ").strip()
    # Try int
    try:
        return int(val)
    except ValueError:
        pass
    # Try float
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _normalize_table_key(heading: str) -> str:
    """Derive a table key from a section heading.

    Examples:
        '## OT0 — Elementwise Shapes' → 'elementwise'
        '### Softmax: `softmax(X:[M,N]) → Y:[M,N]`' → 'softmax'
        '### TopK: ...' → 'topk'
        '### LayerNorm / RMSNorm / RMSNorm-Residual: ...' → 'layernorm'
        '### Reduce Sum / Max / Mean: ...' → 'reduce'
        '### SiLU-and-Mul / GELU-and-Mul: ...' → 'silu_and_mul'
        '### Batch Matmul: ...' → 'batch_matmul'
        '### Grouped Matmul: ...' → 'grouped_matmul'
        '### Fused Linear Cross Entropy: ...' → 'fused_linear_cross_entropy'
        '### Cross Entropy: ...' → 'cross_entropy'
        '### Quantize per Token: ...' → 'quantize_per_token'
        '### Dequantize per Channel: ...' → 'dequantize_per_channel'
        '### Paged Attention: ...' → 'paged_attention'
        '### Flash Attention: ...' → 'flash_attention'
        '### Grouped Query Attention (GQA): ...' → 'grouped_query_attention'
        '### Multi-Latent Attention (MLA): ...' → 'multi_latent_attention'
        '### Cross Attention: ...' → 'cross_attention'
    """
    # Strip markdown heading markers
    h = re.sub(r"^#+\s*", "", heading).strip()

    # For OT-level headings: ## OT0 — Elementwise Shapes
    ot_match = re.match(r"OT\d+\s*[—–-]\s*(.+?)(?:\s+Shapes?)?\s*$", h)
    if ot_match:
        return ot_match.group(1).strip().lower()

    # For sub-section headings: ### OpName: signature
    # Take everything before `:` or `(`
    h = re.split(r"[:(]", h, maxsplit=1)[0].strip()

    # Specific multi-word normalisations
    _KNOWN_NAMES = {
        "softmax": "softmax",
        "layernorm": "layernorm",
        "layernorm / rmsnorm / rmsnorm-residual": "layernorm",
        "reduce sum / max / mean": "reduce",
        "topk": "topk",
        "cumsum": "cumsum",
        "matmul": "matmul",
        "batch matmul": "batch_matmul",
        "grouped matmul": "grouped_matmul",
        "transpose": "transpose",
        "concat": "concat",
        "split": "split",
        "gather": "gather",
        "scatter": "scatter",
        "embedding": "embedding",
        "permute": "permute",
        "silu-and-mul / gelu-and-mul": "silu_and_mul",
        "silu-and-mul / gelu_and_mul": "silu_and_mul",
        "silu_and_mul / gelu_and_mul": "silu_and_mul",
        "swiglu / geglu": "silu_and_mul",  # legacy heading before D8-X1 C1/C2 rename
        "rope": "rope",
        "cross entropy": "cross_entropy",
        "fused linear cross entropy": "fused_linear_cross_entropy",
        "quantize per token": "quantize_per_token",
        "dequantize per channel": "dequantize_per_channel",
        "flash attention": "flash_attention",
        "grouped query attention": "grouped_query_attention",
        "multi-latent attention": "multi_latent_attention",
        "cross attention": "cross_attention",
        "paged attention": "paged_attention",
    }

    h_lower = h.lower().strip()
    # Remove parenthetical like "(GQA)" or "(MLA)" or "DeepSeek-V2/V3"
    h_clean = re.sub(r"\s*\(.*?\)", "", h_lower).strip()
    # Remove trailing model info like ": deepseek-v2/v3"
    h_clean = re.split(r":\s*", h_clean, maxsplit=1)[0].strip()

    if h_clean in _KNOWN_NAMES:
        return _KNOWN_NAMES[h_clean]

    # Fallback: lowercase, replace spaces/hyphens with underscore
    return re.sub(r"[\s-]+", "_", h_clean).lower()


def _parse_table(lines: list[str]) -> tuple[list[str], list[dict]]:
    """Parse a markdown table from a list of lines.

    Returns (headers, rows) where each row is a dict keyed by lowercase
    header names.
    """
    # Find header line (contains |)
    header_line = lines[0]
    headers_raw = [c.strip() for c in header_line.split("|")]
    # Filter empty from leading/trailing |
    headers = [h for h in headers_raw if h]

    # Normalise header names
    norm_headers = []
    for h in headers:
        # Clean markdown formatting
        h_clean = h.replace("*", "").strip()
        # Special: N₁ → n1, N₂ → n2, N_total → n_total
        h_clean = h_clean.replace("₁", "1").replace("₂", "2")
        # Remove parenthetical like (input)
        h_clean = re.sub(r"\s*\(.*?\)", "", h_clean)
        norm_headers.append(h_clean.lower().strip())

    # Skip separator line (|:---|...)
    data_lines = lines[2:]  # skip header + separator

    rows = []
    for line in data_lines:
        line = line.strip()
        if not line or not line.startswith("|"):
            break
        cells = [c.strip() for c in line.split("|")]
        cells = [c for c in cells if c or cells.index(c) not in (0,)]
        # Filter leading/trailing empty from split
        cells_clean = []
        raw_cells = line.split("|")
        # raw_cells[0] is '' (before first |), raw_cells[-1] is '' (after last |)
        for c in raw_cells[1:-1]:
            cells_clean.append(c.strip())

        row = {}
        for i, hdr in enumerate(norm_headers):
            if i < len(cells_clean):
                row[hdr] = _coerce_value(cells_clean[i])
            else:
                row[hdr] = None
        rows.append(row)

    return norm_headers, rows


# ── Main parser ──────────────────────────────────────────────────────────

def parse_shapes_md(path: Path | str | None = None) -> dict[str, list[dict]]:
    """Parse benchmark-shapes.md and return ``{table_name: [row_dicts]}``.

    Parameters
    ----------
    path : Path or str, optional
        Override path to benchmark-shapes.md. Defaults to the repo copy.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from table name to list of row dicts.
    """
    md_path = Path(path) if path is not None else _SHAPES_MD
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    tables: dict[str, list[dict]] = {}

    # Track current section context
    current_ot_key: str | None = None      # e.g. "elementwise"
    current_op_key: str | None = None      # e.g. "softmax"
    current_sub_key: str | None = None     # e.g. "st4"
    in_bl6 = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect BL6 section — skip it entirely (not shape tables for benchmarking)
        if stripped.startswith("## BL6") or stripped.startswith("## BL6 "):
            in_bl6 = True
            i += 1
            continue

        # Detect new ## section (OT-level)
        if stripped.startswith("## ") and not in_bl6:
            in_bl6 = False
            current_ot_key = _normalize_table_key(stripped)
            current_op_key = None
            current_sub_key = None
            i += 1
            continue

        # Skip BL6 content
        if in_bl6:
            i += 1
            continue

        # Detect ### section (operator-level)
        if stripped.startswith("### ") and not stripped.startswith("####"):
            current_op_key = _normalize_table_key(stripped)
            current_sub_key = None
            i += 1
            continue

        # Detect #### section (ST4 or sub-sections)
        if stripped.startswith("#### "):
            sub = stripped.lstrip("#").strip().lower()
            if "st4" in sub:
                current_sub_key = "st4"
            else:
                current_sub_key = sub
            i += 1
            continue

        # Detect table start: header line followed by separator
        if (
            stripped.startswith("|")
            and i + 1 < len(lines)
            and re.match(r"\s*\|[\s:|-]+\|\s*$", lines[i + 1])
        ):
            # Collect table lines
            table_lines = [stripped, lines[i + 1].strip()]
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                table_lines.append(lines[j].strip())
                j += 1

            # Parse the table
            _hdrs, rows = _parse_table(table_lines)

            if not rows:
                i = j
                continue

            # Determine table key
            if current_op_key:
                base_key = current_op_key
            elif current_ot_key:
                base_key = current_ot_key
            else:
                base_key = f"table_{i}"

            # For ST4 sub-tables, append _st4 suffix
            if current_sub_key == "st4":
                table_key = f"{base_key}_st4"
            else:
                table_key = base_key

            # If key already exists, this is an additional table in the same section
            # (e.g. ST1-ST3 followed by ST4). Merge or use suffix.
            if table_key in tables:
                # Check if it's genuinely a duplicate key — use a numbered suffix
                suffix = 2
                while f"{table_key}_{suffix}" in tables:
                    suffix += 1
                table_key = f"{table_key}_{suffix}"

            tables[table_key] = rows
            i = j
            continue

        i += 1

    if not tables:
        raise ValueError(f"No shape tables found in {md_path}")

    # Filter out metadata tables that aren't shape definitions
    _METADATA_TABLES = {"shape_tier", "reference_llm_architecture_parameters"}
    tables = {k: v for k, v in tables.items() if k not in _METADATA_TABLES}

    return tables


def _extract_tier(row: dict) -> int | None:
    """Extract tier from a row dict, checking 'tier' key."""
    tier_val = row.get("tier")
    if tier_val is not None:
        try:
            return int(tier_val)
        except (ValueError, TypeError):
            pass
    return None


def _all_tags(tables: dict[str, list[dict]]) -> list[str]:
    """Collect all shape tags from all tables."""
    tags = []
    for rows in tables.values():
        for row in rows:
            tag = row.get("tag")
            if tag is not None:
                tags.append(str(tag))
    return tags


def _total_shapes(tables: dict[str, list[dict]]) -> int:
    """Count total number of shapes across all tables."""
    return sum(len(rows) for rows in tables.values())


def _shapes_by_tier(tables: dict[str, list[dict]]) -> dict[int, list[dict]]:
    """Group all shapes by their Shape Tier (ST1-4)."""
    by_tier: dict[int, list[dict]] = {}
    for table_name, rows in tables.items():
        for row in rows:
            tier = _extract_tier(row)
            if tier is None:
                # ST4 sub-tables don't have a Tier column; infer from table name
                if "_st4" in table_name:
                    tier = 4
                else:
                    tier = 0  # unknown
            by_tier.setdefault(tier, []).append(row)
    return by_tier


# ── Module-level singletons (computed once at import) ────────────────────

SHAPE_TABLES: dict[str, list[dict]] = parse_shapes_md()
"""Canonical table_name → shape rows, parsed from benchmark-shapes.md."""

ALL_SHAPE_TAGS: list[str] = _all_tags(SHAPE_TABLES)
"""Flat list of all shape tag names."""

TOTAL_SHAPES: int = _total_shapes(SHAPE_TABLES)
"""Total number of shapes in the catalog."""

SHAPES_BY_TIER: dict[int, list[dict]] = _shapes_by_tier(SHAPE_TABLES)
"""All shapes grouped by Shape Tier (ST1-4)."""


# ── Mapping from op name to registry table keys ─────────────────────────

# This maps each of the 45 operators to one or more table keys in SHAPE_TABLES.
# Multiple ops can share a table (e.g. all 12 elementwise ops share "elementwise").
_OP_TO_TABLE_KEYS: dict[str, list[str]] = {
    # OT0 — Elementwise (12 ops share one table)
    "relu": ["elementwise"],
    "gelu": ["elementwise"],
    "silu": ["elementwise"],
    "tanh": ["elementwise"],
    "sigmoid": ["elementwise"],
    "add": ["elementwise"],
    "mul": ["elementwise"],
    "where_": ["elementwise"],
    "cast": ["elementwise"],
    "neg": ["elementwise"],
    "exp": ["elementwise"],
    "rsqrt": ["elementwise"],
    # OT1 — Reduction
    "softmax": ["softmax", "softmax_st4"],
    "layernorm": ["layernorm", "layernorm_st4"],
    "rmsnorm": ["layernorm", "layernorm_st4"],
    "rmsnorm_residual": ["layernorm", "layernorm_st4"],
    "reduce_sum": ["reduce"],
    "reduce_max": ["reduce"],
    "reduce_mean": ["reduce"],
    "argmax": ["reduce"],
    "topk": ["topk"],
    "cumsum": ["cumsum"],
    # OT2 — Data Movement & Dense
    "matmul": ["matmul", "matmul_st4"],
    "batch_matmul": ["batch_matmul"],
    "grouped_matmul": ["grouped_matmul"],
    "transpose": ["transpose"],
    "concat": ["concat"],
    "split": ["split"],
    "gather": ["gather"],
    "scatter": ["scatter"],
    "embedding": ["embedding"],
    "permute": ["permute"],
    "copy_": ["elementwise"],  # copy_ reuses elementwise shapes
    # OT3 — Fused Compound
    "silu_and_mul": ["silu_and_mul"],
    "gelu_and_mul": ["silu_and_mul"],
    "rope": ["rope"],
    "cross_entropy": ["cross_entropy"],
    "fused_linear_cross_entropy": ["fused_linear_cross_entropy"],
    "quantize_per_token": ["quantize_per_token"],
    "dequantize_per_channel": ["dequantize_per_channel"],
    # OT4 — Attention
    "flash_attention": ["flash_attention"],
    "grouped_query_attention": ["grouped_query_attention"],
    "multi_latent_attention": ["multi_latent_attention"],
    "cross_attention": ["cross_attention"],
    "paged_attention": ["paged_attention"],
}


def get_table_keys_for_op(op: str) -> list[str]:
    """Return the SHAPE_TABLES keys that provide shapes for a given op."""
    return _OP_TO_TABLE_KEYS.get(op, [])


def get_registry_shapes_for_op(op: str) -> list[dict]:
    """Return all registry shape dicts for an operator (merged from all its tables)."""
    keys = get_table_keys_for_op(op)
    shapes = []
    for k in keys:
        shapes.extend(SHAPE_TABLES.get(k, []))
    return shapes
