# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""P5: LLM-direct baseline — LLM writes Triton kernel directly.

This baseline prompts an LLM to write a complete Triton kernel from
scratch (no Arke IR, no tool-use, no multi-turn optimisation).  It is
the "best single-shot attempt" the LLM can produce.

In **offline** mode the runner is never called; gate G4 uses it only
when ``--live`` is passed.
"""

from __future__ import annotations

import importlib
import logging
import re
import tempfile
import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from benchmarks.baselines.base import BaselineRunner, register_baseline

logger = logging.getLogger(__name__)

# ── Prompt templates ────────────────────────────────────────

MATMUL_PROMPT = textwrap.dedent("""\
    Write a complete Triton GPU kernel for matrix multiplication.

    Specifications:
    - Input A: [{M}, {K}] float16
    - Input B: [{K}, {N}] float16
    - Output C: [{M}, {N}] float16
    - Use tiling for shared memory efficiency
    - Include the Python wrapper function

    Return ONLY the complete Python code (no explanation).
    The code must define a function `triton_matmul(A, B) -> C`.
""")


def _extract_python_code(response: str) -> str:
    """Extract Python code from an LLM response.

    Handles:
      1. Fenced code blocks (```python ... ``` or ``` ... ```)
      2. Raw code (no fences)
    """
    # Try fenced block first
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: treat entire response as code
    return response.strip()


def _compile_and_load(code: str, tag: str) -> Any:
    """Write *code* to a temp file and import it as a module.

    Returns the loaded module or raises on import/syntax error.
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="arke_llm_"))
    module_file = tmp_dir / f"llm_kernel_{tag}.py"
    module_file.write_text(code)

    spec = importlib.util.spec_from_file_location(
        f"llm_kernel_{tag}", str(module_file),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {module_file}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@register_baseline
class LLMDirectRunner(BaselineRunner):
    """P5: LLM-direct — single-shot Triton kernel generation.

    The runner holds prompt templates and code-extraction logic.
    Actual LLM calls happen only in **live** mode (not offline).
    """

    def __init__(self) -> None:
        self._token_usage: dict[str, int] = {"in": 0, "out": 0}
        # Cache of compiled callables: (op, M, N, K) → fn
        self._kernel_cache: dict[tuple[str, int, int, int], Callable] = {}

    # ── BaselineRunner ABC ──────────────────────────────────

    @property
    def name(self) -> str:
        return "LLM-direct"

    @property
    def priority(self) -> int:
        return 5

    @property
    def source(self) -> str:
        return "LLM single-shot generation (no Arke IR) | provider varies"

    @property
    def available(self) -> bool:
        # In offline mode we never actually call the LLM, so the runner
        # is always "available" as a class.  Gate G4 decides whether to
        # invoke it based on --live.
        return True

    def supports(self, op: str) -> bool:
        return op in ("matmul",)

    def get_fn(
        self,
        op: str,
        M: int,
        N: int,
        K: int = 0,
        dtype: torch.dtype = torch.float16,
    ) -> Callable[[], torch.Tensor] | None:
        """Return a zero-arg callable wrapping the LLM-generated kernel.

        This method is only called in **live** mode.  It generates the
        kernel via ``generate_kernel``, compiles it, and wraps it into
        a zero-arg callable suitable for ``bench_fn``.

        Returns None if generation or compilation fails.
        """
        key = (op, M, N, K)
        fn = self._kernel_cache.get(key)
        if fn is not None:
            return fn

        try:
            code, usage = self.generate_kernel(op, M, N, K, dtype)
        except NotImplementedError:
            return None

        self._token_usage["in"] += usage.get("in", 0)
        self._token_usage["out"] += usage.get("out", 0)

        try:
            mod = _compile_and_load(code, f"{op}_{M}_{N}_{K}")
        except Exception as e:
            logger.warning("LLM-direct compile failed for %s: %s", key, e)
            return None

        entry = getattr(mod, "triton_matmul", None)
        if entry is None:
            logger.warning("LLM-direct: no triton_matmul() in generated code")
            return None

        # Pre-allocate inputs
        A = torch.randn(M, K, device="cuda", dtype=dtype)
        B = torch.randn(K, N, device="cuda", dtype=dtype)

        def _call(a: torch.Tensor = A, b: torch.Tensor = B) -> torch.Tensor:
            return entry(a, b)

        self._kernel_cache[key] = _call
        return _call

    # ── LLM interaction (live-mode only) ────────────────────

    def generate_kernel(
        self,
        op: str,
        M: int,
        N: int,
        K: int,
        dtype: torch.dtype = torch.float16,
    ) -> tuple[str, dict[str, int]]:
        """Call LLM to generate Triton code.

        Returns ``(code_str, {"in": prompt_tokens, "out": completion_tokens})``.

        Raises ``NotImplementedError`` when no LLM backend is configured
        (i.e. offline mode).
        """
        raise NotImplementedError(
            "LLM-direct generation requires --live mode with a configured "
            "LLM provider.  In offline mode, G4 uses optimistic assumptions "
            "for LLM-direct results."
        )

    # ── Helpers ─────────────────────────────────────────────

    @property
    def token_usage(self) -> dict[str, int]:
        """Accumulated token usage across all generate_kernel calls."""
        return dict(self._token_usage)

    @staticmethod
    def build_prompt(op: str, M: int, N: int, K: int) -> str:
        """Build the LLM prompt for a given op and shape.

        Useful for dry-run / token-counting without an actual LLM call.
        """
        if op == "matmul":
            return MATMUL_PROMPT.format(M=M, N=N, K=K)
        raise ValueError(f"No prompt template for op '{op}'")

    @staticmethod
    def extract_code(response: str) -> str:
        """Extract Python code from an LLM response string."""
        return _extract_python_code(response)
