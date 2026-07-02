# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""MLIRBackend — Phase 3 (P3-S1) ArkeBackend implementation.

Implements the 4-method ``ArkeBackend`` protocol (``lower`` / ``compile`` /
``run`` / ``supports_op``) via the user-local MLIR 18 CLI toolchain
(``mlir-opt`` + ``mlir-cpu-runner``), lowering the standard ``linalg`` +
``memref`` dialects to LLVM and JIT-executing on CPU.

Correctness strategy (P3-S1, no python bindings available on this host):
  The MLIR CPU JIT entry point (``mlir-cpu-runner -e main``) cannot take
  external tensor arguments, so ``run()`` synthesizes a self-contained
  ``@main`` harness that materializes the caller's input tensors as MLIR
  ``memref.global`` constants, invokes the emitted kernel, and prints the
  result via ``printMemrefF32``. The printed values are parsed back into a
  numpy array and returned — bit-comparable against a torch reference.

  This is the standard bindings-free MLIR correctness path; the GPU path
  (``gpu`` dialect → PTX) and an in-process ExecutionEngine path are P3-S1
  follow-ups (see docs/roadmap/plan.md Phase 3).

Toolchain discovery: honors ARKE_MLIR_OPT / ARKE_MLIR_CPU_RUNNER /
ARKE_MLIR_RUNNER_UTILS / ARKE_MLIR_C_RUNNER_UTILS env vars (set by
``~/opt/mlir18/env.sh``); falls back to PATH lookup.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from arke.backend.mlir_emitter import (
    EmittedKernel,
    emit_kernel,
    mlir_dtype,
    memref_type,
)
from arke.backend.protocol import BackendArtifact, CompiledKernel
from arke.ir.graph import IRGraph


# ── toolchain discovery ────────────────────────────────────────

def _tool(env_var: str, exe: str) -> str | None:
    p = os.environ.get(env_var)
    if p and os.path.exists(p):
        return p
    return shutil.which(exe)


def mlir_toolchain_available() -> bool:
    return bool(_tool("ARKE_MLIR_OPT", "mlir-opt")
                and _tool("ARKE_MLIR_CPU_RUNNER", "mlir-cpu-runner"))


# Standard CPU lowering pipeline: linalg(memref) → loops → LLVM.
_LOWER_PASSES = [
    "-convert-linalg-to-loops",
    "-convert-scf-to-cf",
    "-convert-cf-to-llvm",
    "-convert-func-to-llvm",
    "-finalize-memref-to-llvm",
    "-convert-arith-to-llvm",
    "-reconcile-unrealized-casts",
]


class MLIRBackend:
    """Arke MLIR backend (P3-S1: linalg → CPU JIT via mlir-cpu-runner)."""

    name = "mlir"

    def __init__(self) -> None:
        self.mlir_opt = _tool("ARKE_MLIR_OPT", "mlir-opt")
        self.cpu_runner = _tool("ARKE_MLIR_CPU_RUNNER", "mlir-cpu-runner")
        self.runner_utils = _tool("ARKE_MLIR_RUNNER_UTILS", "libmlir_runner_utils.so")
        self.c_runner_utils = _tool(
            "ARKE_MLIR_C_RUNNER_UTILS", "libmlir_c_runner_utils.so"
        )

    # ── ArkeBackend protocol ───────────────────────────────────

    def supports_op(self, op_name: str) -> bool:
        from arke.backend.mlir_emitter import SUPPORTED_OPS
        return op_name in SUPPORTED_OPS

    def lower(self, graph: IRGraph) -> BackendArtifact:
        """Generate executable MLIR text from the IR graph."""
        emitted = emit_kernel(graph)
        return BackendArtifact(
            source_code=emitted.mlir_text,
            backend_name=self.name,
            op_name=graph.nodes[0].op if graph.nodes else "",
            metadata={"emitted": emitted, "graph_name": graph.name},
        )

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Verify the kernel lowers cleanly to LLVM (mlir-opt dry run).

        The actual JIT is deferred to run() (which builds a data-bound @main
        harness), but compiling here catches lowering errors early and proves
        the linalg→LLVM pipeline succeeds — the substance of the P3-S1 gate.
        """
        if not self.mlir_opt:
            return CompiledKernel.fail("mlir-opt not found (source ~/opt/mlir18/env.sh)")
        try:
            llvm_ir = self._lower_to_llvm(artifact.source_code)
        except subprocess.CalledProcessError as e:
            return CompiledKernel.fail(f"mlir-opt lowering failed: {e.stderr}")
        return CompiledKernel.ok(
            fn=None,
            backend_name=self.name,
            emitted=artifact.metadata["emitted"],
            llvm_dialect=llvm_ir,
            graph_name=artifact.metadata.get("graph_name", ""),
        )

    def run(self, kernel: CompiledKernel, inputs: dict[str, Any]) -> dict[str, Any]:
        """JIT-execute on CPU with concrete inputs; return {result_name: ndarray}."""
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")
        emitted: EmittedKernel = kernel.metadata["emitted"]
        np_inputs = {k: _to_numpy(v) for k, v in inputs.items()}
        harness = self._build_main_harness(emitted, np_inputs)
        result = self._jit_execute(harness, emitted)
        return {emitted.result_name: result}

    # ── internals ──────────────────────────────────────────────

    def _lower_to_llvm(self, mlir_text: str) -> str:
        cmd = [self.mlir_opt, *_LOWER_PASSES]
        proc = subprocess.run(
            cmd, input=mlir_text, capture_output=True, text=True, check=True
        )
        return proc.stdout

    def _build_main_harness(
        self, emitted: EmittedKernel, np_inputs: dict[str, np.ndarray]
    ) -> str:
        """Wrap the kernel with a @main that binds concrete f32 inputs and prints.

        P3-S1 restricts execution correctness to f32 (printMemrefF32); other
        dtypes lower/compile fine but their JIT print path lands in a follow-up.
        """
        if emitted.result_dtype != "float32":
            raise NotImplementedError(
                f"MLIR JIT run (P3-S1): f32 only, got {emitted.result_dtype}"
            )
        elem = mlir_dtype(emitted.result_dtype)
        globals_lines: list[str] = []
        alloc_lines: list[str] = []

        # Emit each input as a memref.global constant + a get_global in main.
        for idx, (name, shape, dtype) in enumerate(
            zip(emitted.arg_names, emitted.arg_shapes, emitted.arg_dtypes)
        ):
            arr = np_inputs[name].astype(np.float32)
            gname = f"@__arke_in{idx}"
            mtype = memref_type(shape, dtype)
            globals_lines.append(
                f"  memref.global \"private\" constant {gname} : {mtype} = "
                f"dense<{_nest_literal(arr)}>"
            )
            alloc_lines.append(f"    %in{idx} = memref.get_global {gname} : {mtype}")

        # Output buffer for the kernel (dest-passing arg).
        res_ty = memref_type(emitted.result_shape, emitted.result_dtype)
        alloc_lines.append(f"    %out = memref.alloc() : {res_ty}")

        in_args = ", ".join(f"%in{i}" for i in range(len(emitted.arg_names)))
        call_args = f"{in_args}, %out" if in_args else "%out"
        all_tys = ", ".join(
            memref_type(s, d) for s, d in zip(emitted.arg_shapes, emitted.arg_dtypes)
        )
        call_sig = f"{all_tys}, {res_ty}" if all_tys else res_ty

        main = [
            emitted.mlir_text.rstrip().removesuffix("}").rstrip(),  # drop closing module brace
            "",
            *globals_lines,
            "  func.func private @printMemrefF32(memref<*xf32>)",
            "  func.func @main() {",
            *alloc_lines,
            f"    func.call @{emitted.kernel_name}({call_args}) : ({call_sig}) -> ()",
            f"    %cast = memref.cast %out : {res_ty} to memref<*xf32>",
            "    func.call @printMemrefF32(%cast) : (memref<*xf32>) -> ()",
            "    return",
            "  }",
            "}",
        ]
        return "\n".join(main)

    def _jit_execute(self, harness_mlir: str, emitted: EmittedKernel) -> np.ndarray:
        llvm = self._lower_to_llvm(harness_mlir)
        cmd = [
            self.cpu_runner, "-e", "main", "-entry-point-result=void",
            f"-shared-libs={self.runner_utils}",
            f"-shared-libs={self.c_runner_utils}",
        ]
        proc = subprocess.run(
            cmd, input=llvm, capture_output=True, text=True, check=True
        )
        return _parse_printmemref(proc.stdout, emitted.result_shape)


# ── helpers: numpy <-> MLIR text ───────────────────────────────

def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    # torch tensor
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _fmt_f32(x: float) -> str:
    return repr(float(x))


def _nest_literal(arr: np.ndarray) -> str:
    """Recursively render nested [[..],[..]] literal matching arr shape."""
    if arr.ndim == 0:
        return _fmt_f32(float(arr))
    if arr.ndim == 1:
        return "[" + ", ".join(_fmt_f32(float(x)) for x in arr) + "]"
    return "[" + ", ".join(_nest_literal(sub) for sub in arr) + "]"


_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?")


def _parse_printmemref(stdout: str, shape: list[int]) -> np.ndarray:
    """Parse mlir-cpu-runner printMemrefF32 output into an ndarray.

    Output format (from printMemrefF32):
      Unranked Memref base@ = 0x.. rank = 2 ... data =
      [[6,   6],
       [6,   6]]
    We locate the 'data =' marker and parse every float after it.
    """
    idx = stdout.find("data =")
    if idx == -1:
        raise RuntimeError(f"printMemref output missing 'data =':\n{stdout}")
    tail = stdout[idx + len("data ="):]
    nums = [float(m.group()) for m in _NUM_RE.finditer(tail)]
    expected = int(np.prod(shape)) if shape else 1
    if len(nums) < expected:
        raise RuntimeError(
            f"printMemref parse: expected {expected} values, got {len(nums)}:\n{stdout}"
        )
    return np.array(nums[:expected], dtype=np.float32).reshape(shape)
