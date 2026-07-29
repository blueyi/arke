# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Regression gate for Tensor Core dtype discipline in Triton templates.

Root cause this guards against (K-ATT FA-v2, 2026-07-29): loading a `tl.dot`
operand as fp32 via `.to(tl.float32)` forces the matmul onto the CUDA cores'
FFMA path and leaves the sm_86 Tensor Cores idle. FA-v2 proved the fix
(fp16 dot + `out_dtype=tl.float32` accumulate) is worth geomean 0.496->0.846.
The same anti-pattern was later found lurking in mla.py.j2 and fixed.

This test statically scans every rendered Triton template and fails if any
`tl.dot` operand traces back to a `.to(tl.float32)` cast, so the knowledge
lives as a gate instead of tribal memory. It also enforces the positive
convention: every `tl.dot` should carry `out_dtype=tl.float32` (fp32
accumulation), which is how you get both speed AND accuracy.

Legitimate fp32 usage (accumulators m_i/l_i/o_acc, softmax math) is NOT
flagged — only casts feeding a dot operand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

TEMPLATE_DIR = (
    Path(__file__).resolve().parents[2]
    / "arke" / "backend" / "triton_templates"
)

# Minimal Jinja context to render each template to valid Python source.
_RENDER_CTX = {
    "kernel_name": "k",
    "causal": True,
    "gqa_groups": 1,
    "op_variant": "flash",
    "dtype": "float16",
    "BLOCK": 64,
}


def _dot_templates() -> list[Path]:
    return sorted(
        p for p in TEMPLATE_DIR.glob("*.py.j2")
        if "tl.dot" in p.read_text(encoding="utf-8")
    )


def _render(path: Path) -> str:
    from jinja2 import Template
    return Template(path.read_text(encoding="utf-8")).render(**_RENDER_CTX)


# Names that are legitimately fp32 (accumulators / softmax scratch) — a cast
# to these is never a TC violation even if the name later appears near a dot.
_FP32_OK_TARGETS = {"acc", "o_acc", "m_i", "l_i", "m_new", "alpha", "p_scale"}


def _dot_operand_names(src: str) -> set[str]:
    """Collect bare identifiers passed positionally to tl.dot(...)."""
    names: set[str] = set()
    for m in re.finditer(r"tl\.dot\(([^)]*)\)", src):
        args = m.group(1)
        # first two positional args are the matmul operands
        for arg in args.split(",")[:2]:
            arg = arg.strip()
            # strip common wrappers: tl.trans(x), x.to(...)
            base = re.sub(r"^tl\.trans\(", "", arg).rstrip(")")
            base = re.sub(r"\.to\([^)]*\)$", "", base)
            idm = re.match(r"[A-Za-z_]\w*", base)
            if idm:
                names.add(idm.group(0))
    return names


@pytest.mark.parametrize("tpl", _dot_templates(), ids=lambda p: p.name)
def test_no_fp32_cast_on_dot_operands(tpl: Path) -> None:
    """No tl.dot operand may be loaded/cast as fp32 (kills Tensor Core)."""
    src = _render(tpl)
    dot_operands = _dot_operand_names(src)

    # Find `NAME = tl.load(...).to(tl.float32)` or `NAME = ....to(tl.float32)`
    # where NAME feeds a dot — that's the FFMA-forcing anti-pattern.
    violations = []
    for line in src.splitlines():
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=.*\.to\(tl\.float32\)", line)
        if not m:
            continue
        target = m.group(1)
        if target in _FP32_OK_TARGETS:
            continue
        if target in dot_operands:
            violations.append(f"  {target!r} <- .to(tl.float32) feeds tl.dot: {line.strip()}")

    assert not violations, (
        f"{tpl.name}: Tensor Core dtype violation — a tl.dot operand is cast "
        f"to fp32 (forces FFMA, TC idle). Load it as fp16 and use "
        f"tl.dot(..., out_dtype=tl.float32) instead.\n" + "\n".join(violations)
    )


def _dot_calls(src: str) -> list[tuple[str, str]]:
    """Return (assign_target, full_arg_string) for every tl.dot(...) call,
    with balanced-paren arg extraction (handles nested tl.trans(...), .to(...),
    out_dtype=...). assign_target is the LHS name if the dot is on an
    assignment/augmented-assignment line, else ''.
    """
    flat = re.sub(r"#[^\n]*", "", src)       # strip line comments first
    flat = re.sub(r"\s+", " ", flat)
    out: list[tuple[str, str]] = []
    for m in re.finditer(r"tl\.dot\(", flat):
        i = m.end()
        depth = 1
        j = i
        while j < len(flat) and depth:
            if flat[j] == "(":
                depth += 1
            elif flat[j] == ")":
                depth -= 1
            j += 1
        args = flat[i:j - 1]
        # look back for `NAME =` / `NAME +=` immediately before tl.dot
        prefix = flat[max(0, m.start() - 40):m.start()]
        tm = re.search(r"([A-Za-z_]\w*)\s*\+?=\s*$", prefix)
        out.append((tm.group(1) if tm else "", args))
    return out


@pytest.mark.parametrize("tpl", _dot_templates(), ids=lambda p: p.name)
def test_dot_accumulates_in_fp32(tpl: Path) -> None:
    """Every tl.dot must accumulate in fp32 — either via out_dtype=tl.float32
    OR by feeding a pre-declared fp32 accumulator (acc/o_acc = tl.zeros(...,
    dtype=tl.float32)). fp16 accumulation loses precision on long reductions.
    Checks the *effect*, so both the attention style (out_dtype) and the
    matmul style (fp32 acc buffer) pass.
    """
    src = _render(tpl)
    fp32_accs = set(
        re.findall(r"([A-Za-z_]\w*)\s*=\s*tl\.zeros\(.*?tl\.float32", src)
    )
    bad = []
    for target, args in _dot_calls(src):
        if "out_dtype" in args:
            continue
        if target and target in fp32_accs:
            continue
        bad.append(args.strip()[:50])
    assert not bad, (
        f"{tpl.name}: tl.dot neither uses out_dtype=tl.float32 nor targets an "
        f"fp32 accumulator (fp16 accumulate loses precision):\n  "
        + "\n  ".join(bad)
    )


def test_gate_covers_all_dot_templates() -> None:
    """Sanity: the gate must actually be scanning templates (not vacuous)."""
    tpls = _dot_templates()
    assert len(tpls) >= 5, f"expected >=5 tl.dot templates, found {len(tpls)}"
    names = {p.name for p in tpls}
    assert {"flash_attention.py.j2", "matmul.py.j2", "mla.py.j2"} <= names
