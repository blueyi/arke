# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""MLIR GPU runtime — PTX generation + driver-API launch (Phase 3, P3-S1 GPU).

The Ubuntu ``mlir-18-tools`` package is built **with** the NVPTX LLVM target
(so ``gpu-module-to-binary`` can emit PTX/cubin) but **without** the CUDA JIT
runner (`libmlir_cuda_runtime.so` / `mlir-cuda-runner` are absent). So instead
of MLIR's own GPU JIT, we:

  1. Lower a single-kernel ``gpu.module`` to **PTX text** via
     ``mlir-opt -convert-scf-to-cf -convert-gpu-to-nvvm
     -gpu-module-to-binary=format=isa`` (extracting the ``assembly = "..."``
     string from the emitted ``#gpu.object``).
  2. Load the PTX with the CUDA **driver API** (``cuda-python``:
     ``cuModuleLoadData`` JIT-compiles PTX in-driver — no ``ptxas`` byte
     wrangling) and launch it, marshaling MLIR's unpacked-memref ABI by hand.

This gives a bindings-free GPU correctness path on hosts that have a CUDA
driver + ``cuda-python`` but no MLIR CUDA runner. Verified on RTX 3060
(SM 8.6, CUDA driver 13.x): matmul bit-correct vs numpy.

MLIR unpacked-memref calling convention (per memref arg):
    rank-0: alloc_ptr, aligned_ptr, offset                       (3 scalars)
    rank-N: alloc_ptr, aligned_ptr, offset, sizes[N], strides[N] (3 + 2N)
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np


# ── toolchain ──────────────────────────────────────────────────

def _tool(env_var: str, exe: str) -> str | None:
    p = os.environ.get(env_var)
    if p and os.path.exists(p):
        return p
    return shutil.which(exe)


def _cuda_python_available() -> bool:
    try:
        import cuda.bindings.driver  # noqa: F401
        return True
    except Exception:
        return False


def gpu_toolchain_available() -> bool:
    """True iff we can emit PTX (mlir-opt) *and* launch it (cuda-python + driver)."""
    if not _tool("ARKE_MLIR_OPT", "mlir-opt"):
        return False
    if not _cuda_python_available():
        return False
    try:
        from cuda.bindings import driver
        err, = _as_tuple(driver.cuInit(0))
        if err != driver.CUresult.CUDA_SUCCESS:
            return False
        err, n = _as_tuple(driver.cuDeviceGetCount())
        return err == driver.CUresult.CUDA_SUCCESS and n > 0
    except Exception:
        return False


def _as_tuple(ret: Any) -> tuple:
    return ret if isinstance(ret, tuple) else (ret,)


# ── PTX extraction ─────────────────────────────────────────────

_ASM_RE = re.compile(r'assembly = "((?:[^"\\]|\\.)*)"')

# libdevice.bc — CUDA's math library (bitcode). Linked into the gpu binary so
# transcendentals (math.exp/tanh/… → __nv_* libdevice calls) resolve to native
# PTX (ex2.approx etc.); without it the driver rejects the PTX (INVALID_PTX).
def _find_libdevice() -> str | None:
    for p in (
        os.environ.get("ARKE_LIBDEVICE"),
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
    ):
        if p and os.path.exists(p):
            return p
    import glob
    hits = sorted(glob.glob("/usr/local/cuda*/nvvm/libdevice/libdevice.*.bc"))
    return hits[0] if hits else None


# PTX-lowering passes: scf→cf (unroll the K-loop control flow), gpu→nvvm,
# then serialize the gpu.module to PTX text (format=isa). libdevice linked via -l.
def _ptx_passes() -> list[str]:
    libdev = _find_libdevice()
    fmt = "format=isa"
    if libdev:
        fmt = f"format=isa l={libdev}"
    return [
        "-convert-scf-to-cf",
        "-convert-gpu-to-nvvm",
        f"-gpu-module-to-binary={fmt}",
    ]


def _mlir_unescape(s: str) -> bytes:
    """Decode an MLIR string literal (``\\\\``, ``\\"``, ``\\HH`` hex escapes)."""
    out = bytearray()
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            n = s[i + 1]
            if n == "\\":
                out.append(0x5C); i += 2; continue
            if n == '"':
                out.append(0x22); i += 2; continue
            out.append(int(s[i + 1:i + 3], 16)); i += 3; continue
        out.append(ord(c)); i += 1
    return bytes(out)


def mlir_gpu_to_ptx(gpu_mlir: str, mlir_opt: str | None = None) -> str:
    """Lower a single-kernel gpu.module MLIR string to PTX text."""
    tool = mlir_opt or _tool("ARKE_MLIR_OPT", "mlir-opt")
    if not tool:
        raise RuntimeError("mlir-opt not found (source ~/opt/mlir20/env.sh)")
    proc = subprocess.run(
        [tool, *_ptx_passes()], input=gpu_mlir,
        capture_output=True, text=True, check=True,
    )
    m = _ASM_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"no PTX assembly in gpu-module-to-binary output:\n{proc.stdout[:500]}")
    return _mlir_unescape(m.group(1)).decode("utf-8", "replace")


# ── CUDA driver launch ─────────────────────────────────────────

@dataclass
class GPUBuffer:
    dptr: Any
    nbytes: int
    shape: tuple[int, ...]


class CudaLauncher:
    """Thin RAII wrapper over the CUDA driver API for PTX launch."""

    def __init__(self) -> None:
        from cuda.bindings import driver
        self.driver = driver
        self._chk(driver.cuInit(0))
        self.dev = self._chk(driver.cuDeviceGet(0))
        self.ctx = self._chk(driver.cuCtxCreate(driver.CUctxCreateParams(), 0, self.dev))
        self._allocs: list[Any] = []
        self._modules: list[Any] = []

    def _chk(self, ret: Any) -> Any:
        t = _as_tuple(ret)
        err = t[0]
        if err != self.driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
        rest = t[1:]
        if len(rest) == 1:
            return rest[0]
        return rest

    def load_ptx(self, ptx: str, entry: str) -> Any:
        mod = self._chk(self.driver.cuModuleLoadData(ptx.encode()))
        self._modules.append(mod)
        return self._chk(self.driver.cuModuleGetFunction(mod, entry.encode()))

    def to_device(self, arr: np.ndarray) -> GPUBuffer:
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        dptr = self._chk(self.driver.cuMemAlloc(arr.nbytes))
        self._allocs.append(dptr)
        self._chk(self.driver.cuMemcpyHtoD(dptr, arr.ctypes.data, arr.nbytes))
        return GPUBuffer(dptr, arr.nbytes, arr.shape)

    def alloc_output(self, shape: tuple[int, ...]) -> GPUBuffer:
        nbytes = int(np.prod(shape)) * 4
        dptr = self._chk(self.driver.cuMemAlloc(nbytes))
        self._allocs.append(dptr)
        return GPUBuffer(dptr, nbytes, tuple(shape))

    def from_device(self, buf: GPUBuffer) -> np.ndarray:
        host = np.empty(buf.shape, dtype=np.float32)
        self._chk(self.driver.cuMemcpyDtoH(host.ctypes.data, buf.dptr, buf.nbytes))
        return host

    @staticmethod
    def _memref_args(buf: GPUBuffer) -> list[np.ndarray]:
        """Marshal a memref to MLIR's unpacked ABI: alloc, align, offset, sizes, strides."""
        u64 = lambda v: np.array([int(v)], dtype=np.uint64)   # noqa: E731
        i64 = lambda v: np.array([int(v)], dtype=np.int64)    # noqa: E731
        args = [u64(int(buf.dptr)), u64(int(buf.dptr)), i64(0)]
        # row-major strides
        shape = buf.shape
        strides = []
        acc = 1
        for d in reversed(shape):
            strides.append(acc)
            acc *= d
        strides = list(reversed(strides))
        args += [i64(s) for s in shape]
        args += [i64(s) for s in strides]
        return args

    def launch(self, fn: Any, grid: tuple[int, int, int],
               block: tuple[int, int, int], buffers: list[GPUBuffer]) -> None:
        arg_arrays: list[np.ndarray] = []
        for b in buffers:
            arg_arrays += self._memref_args(b)
        arg_ptrs = np.array([a.ctypes.data for a in arg_arrays], dtype=np.uint64)
        self._chk(self.driver.cuLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            0, 0, arg_ptrs.ctypes.data, 0,
        ))
        self._chk(self.driver.cuCtxSynchronize())

    def close(self) -> None:
        for dptr in self._allocs:
            self.driver.cuMemFree(dptr)
        for mod in self._modules:
            self.driver.cuModuleUnload(mod)
        if self.ctx is not None:
            self.driver.cuCtxDestroy(self.ctx)
        self._allocs.clear()
        self._modules.clear()
        self.ctx = None

    def __enter__(self) -> "CudaLauncher":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ── ArkeBackend implementation (GPU) ───────────────────────────

class MLIRGPUBackend:
    """Arke MLIR GPU backend (P3-S1 GPU): gpu.module → PTX → CUDA driver launch.

    Implements the 4-method ArkeBackend protocol. ``lower`` emits a single-kernel
    gpu.module; ``compile`` lowers it to PTX text; ``run`` JIT-loads the PTX via
    the CUDA driver API and launches it, returning the result as a numpy array.

    This is the NVIDIA leg of Phase 3's multi-hardware-via-MLIR story
    (Thesis L2 / P3-S_FINAL): the same SemanticIR that the CPU MLIR backend
    lowers to linalg is lowered here to the gpu dialect and executed on PTX.
    """

    name = "mlir-gpu"

    def __init__(self, chip: str = "sm_86") -> None:
        self.chip = chip
        self.mlir_opt = _tool("ARKE_MLIR_OPT", "mlir-opt")

    def supports_op(self, op_name: str) -> bool:
        from arke.backend.mlir_emitter import GPU_ELEMENTWISE_OPS
        return op_name == "matmul" or op_name in GPU_ELEMENTWISE_OPS

    def lower(self, graph: Any) -> Any:
        from arke.backend.mlir_emitter import (
            emit_gpu_matmul, emit_gpu_elementwise, GPU_ELEMENTWISE_OPS,
        )
        from arke.backend.protocol import BackendArtifact
        op = graph.nodes[0].op if graph.nodes else ""
        if op in GPU_ELEMENTWISE_OPS:
            emitted = emit_gpu_elementwise(graph, chip=self.chip)
        else:
            emitted = emit_gpu_matmul(graph, chip=self.chip)
        return BackendArtifact(
            source_code=emitted.mlir_text,
            backend_name=self.name,
            op_name=op,
            metadata={"emitted": emitted},
        )

    def compile(self, artifact: Any) -> Any:
        from arke.backend.protocol import CompiledKernel
        if not self.mlir_opt:
            return CompiledKernel.fail("mlir-opt not found (source ~/opt/mlir20/env.sh)")
        try:
            ptx = mlir_gpu_to_ptx(artifact.source_code, self.mlir_opt)
        except subprocess.CalledProcessError as e:
            return CompiledKernel.fail(f"gpu→PTX lowering failed: {e.stderr}")
        except RuntimeError as e:
            return CompiledKernel.fail(str(e))
        return CompiledKernel.ok(
            fn=None,
            backend_name=self.name,
            emitted=artifact.metadata["emitted"],
            ptx=ptx,
        )

    def run(self, kernel: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")
        emitted = kernel.metadata["emitted"]
        ptx = kernel.metadata["ptx"]
        np_inputs = {k: _to_numpy(v) for k, v in inputs.items()}
        with CudaLauncher() as cu:
            fn = cu.load_ptx(ptx, emitted.kernel_name)
            bufs: list[GPUBuffer] = []
            out_buf: GPUBuffer | None = None
            for name in emitted.buffer_order:
                if name == emitted.result_name:
                    out_buf = cu.alloc_output(tuple(emitted.result_shape))
                    bufs.append(out_buf)
                else:
                    bufs.append(cu.to_device(np_inputs[name]))
            assert out_buf is not None
            cu.launch(fn, emitted.grid, emitted.block, bufs)
            result = cu.from_device(out_buf)
        return {emitted.result_name: result}


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
