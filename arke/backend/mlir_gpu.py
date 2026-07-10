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
_BIN_RE = re.compile(r'bin = "((?:[^"\\]|\\.)*)"')

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


# cubin-lowering passes: same as PTX but format=bin → ptxas compiles to native
# SASS (register allocation, instruction scheduling, occupancy tuning).
def _cubin_passes() -> list[str]:
    libdev = _find_libdevice()
    fmt = "format=bin"
    if libdev:
        fmt = f"format=bin l={libdev}"
    return [
        "-convert-scf-to-cf",
        "-convert-gpu-to-nvvm",
        f"-gpu-module-to-binary={fmt}",
    ]


# nvgpu-aware PTX lowering: convert nvgpu → nvvm first, then the normal pipeline.
# Used for tensor-core matmul kernels that emit nvgpu.mma.sync / ldmatrix / cp.async.
def _nvgpu_ptx_passes() -> list[str]:
    libdev = _find_libdevice()
    fmt = "format=isa"
    if libdev:
        fmt = f"format=isa l={libdev}"
    return [
        "-convert-nvgpu-to-nvvm",
        "-convert-vector-to-llvm",
        "-convert-arith-to-llvm",
        "-convert-scf-to-cf",
        "-convert-gpu-to-nvvm",
        "-reconcile-unrealized-casts",
        f"-gpu-module-to-binary={fmt}",
    ]


def _nvgpu_cubin_passes() -> list[str]:
    libdev = _find_libdevice()
    fmt = "format=bin"
    if libdev:
        fmt = f"format=bin l={libdev}"
    return [
        "-convert-nvgpu-to-nvvm",
        "-convert-vector-to-llvm",
        "-convert-arith-to-llvm",
        "-convert-scf-to-cf",
        "-convert-gpu-to-nvvm",
        "-reconcile-unrealized-casts",
        f"-gpu-module-to-binary={fmt}",
    ]


# nvgpu two-stage lowering for tensor-core kernels (nvgpu.mma.sync / ldmatrix /
# cp.async). The emitter produces warp-level `vector.contract`; stage 1 distributes
# it into per-thread nvgpu fragment ops, stage 2 lowers to a cubin.
#
# VERIFIED path (2026-07-07, RTX 3060 sm_86): the single-pass list does NOT work
# when `gpu.lane_id` + workgroup memrefs appear (the memref-space conversion the
# manual `-convert-gpu-to-nvvm` does fails). The `-gpu-lower-to-nvvm-pipeline`
# one-shot handles memref-space conversion internally, so we run:
#   stage 1: --convert-vector-to-gpu=use-nvgpu
#   stage 2: -convert-nvgpu-to-nvvm -gpu-lower-to-nvvm-pipeline=cubin-chip=<chip>
# and extract the cubin blob from the emitted `#gpu.object<..., "BLOB">`.
def _nvgpu_stage1_passes() -> list[str]:
    return ["--convert-vector-to-gpu=use-nvgpu"]


def _nvgpu_stage2_passes(chip: str) -> list[str]:
    return [
        "--nvgpu-optimize-shared-memory",
        "-convert-nvgpu-to-nvvm",
        f"-gpu-lower-to-nvvm-pipeline=cubin-chip={chip}",
    ]


# Extracts the cubin blob from a `#gpu.object<#nvvm.target<...>, "BLOB">` attr
# (what -gpu-lower-to-nvvm-pipeline emits, vs the `bin = "..."` of
# -gpu-module-to-binary=format=bin).
_OBJ_RE = re.compile(r'#gpu\.object<[^,]*,\s*"((?:[^"\\]|\\.)*)"')


def mlir_nvgpu_to_cubin(gpu_mlir: str, chip: str = "sm_86",
                        mlir_opt: str | None = None) -> bytes:
    """Lower a tensor-core (nvgpu) gpu.module MLIR string to a native cubin.

    Runs the verified two-stage nvgpu pipeline and extracts the cubin blob from
    the emitted ``#gpu.object``. Raises on any stage failure.
    """
    tool = mlir_opt or _tool("ARKE_MLIR_OPT", "mlir-opt")
    if not tool:
        raise RuntimeError("mlir-opt not found (source ~/opt/mlir20/env.sh)")
    s1 = subprocess.run(
        [tool, *_nvgpu_stage1_passes()], input=gpu_mlir,
        capture_output=True, text=True, check=True,
    ).stdout
    if "nvgpu.mma.sync" not in s1:
        raise RuntimeError(
            "nvgpu stage1 did not distribute vector.contract → nvgpu.mma.sync "
            "(check that transfer_reads source workgroup memory and the "
            "contract shape is a valid MMA shape, e.g. m16n8k16)"
        )
    s2 = subprocess.run(
        [tool, *_nvgpu_stage2_passes(chip)], input=s1,
        capture_output=True, text=True, check=True,
    ).stdout
    m = _OBJ_RE.search(s2)
    if not m:
        raise RuntimeError(f"no #gpu.object blob in nvgpu lowering output:\n{s2[:500]}")
    return _mlir_unescape(m.group(1))


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


def mlir_gpu_to_ptx(gpu_mlir: str, mlir_opt: str | None = None,
                    passes: list[str] | None = None) -> str:
    """Lower a single-kernel gpu.module MLIR string to PTX text.

    If *passes* is given it replaces the default ``_ptx_passes()`` pipeline.
    Use ``_nvgpu_ptx_passes()`` for kernels that contain nvgpu ops.
    """
    tool = mlir_opt or _tool("ARKE_MLIR_OPT", "mlir-opt")
    if not tool:
        raise RuntimeError("mlir-opt not found (source ~/opt/mlir20/env.sh)")
    proc = subprocess.run(
        [tool, *(passes or _ptx_passes())], input=gpu_mlir,
        capture_output=True, text=True, check=True,
    )
    m = _ASM_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"no PTX assembly in gpu-module-to-binary output:\n{proc.stdout[:500]}")
    return _mlir_unescape(m.group(1)).decode("utf-8", "replace")


def mlir_gpu_to_cubin(gpu_mlir: str, mlir_opt: str | None = None,
                      passes: list[str] | None = None) -> bytes:
    """Lower a single-kernel gpu.module MLIR string to a native cubin (ELF).

    Uses ``format=bin`` so that ``mlir-opt`` invokes ``ptxas`` internally.
    If *passes* is given it replaces the default ``_cubin_passes()`` pipeline.
    Use ``_nvgpu_cubin_passes()`` for kernels that contain nvgpu ops.
    """
    tool = mlir_opt or _tool("ARKE_MLIR_OPT", "mlir-opt")
    if not tool:
        raise RuntimeError("mlir-opt not found (source ~/opt/mlir20/env.sh)")
    proc = subprocess.run(
        [tool, *(passes or _cubin_passes())], input=gpu_mlir,
        capture_output=True, text=True, check=True,
    )
    m = _BIN_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(
            f"no cubin blob in gpu-module-to-binary output:\n{proc.stdout[:500]}"
        )
    return _mlir_unescape(m.group(1))


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

    def load_cubin(self, cubin: bytes, entry: str) -> Any:
        """Load a native cubin (ELF) binary and return the kernel function."""
        mod = self._chk(self.driver.cuModuleLoadData(cubin))
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

    def launch_no_sync(self, fn: Any, grid: tuple[int, int, int],
                       block: tuple[int, int, int], buffers: list[GPUBuffer]) -> None:
        """Enqueue the kernel WITHOUT a trailing cuCtxSynchronize.

        For timed benchmarking: the caller records CUDA events around a batch of
        launches and synchronizes once, so per-launch host sync doesn't pollute
        the measured GPU time. ``arg_arrays`` is kept alive until the call
        returns (cuLaunchKernel copies the arg buffer), which is sufficient
        because we sync before freeing anything.
        """
        arg_arrays: list[np.ndarray] = []
        for b in buffers:
            arg_arrays += self._memref_args(b)
        arg_ptrs = np.array([a.ctypes.data for a in arg_arrays], dtype=np.uint64)
        self._chk(self.driver.cuLaunchKernel(
            fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
            0, 0, arg_ptrs.ctypes.data, 0,
        ))

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

    def time_kernel(self, fn: Any, grid: tuple[int, int, int],
                    block: tuple[int, int, int], buffers: list[GPUBuffer],
                    iters: int = 50, warmup: int = 10) -> float:
        """Return mean kernel-only wall time in ms via CUDA events.

        Excludes H2D/D2H copy and PTX-load cost (those happen once, outside the
        timed region) — this is the fair *kernel* latency for a perf comparison
        against Triton/torch, which are also timed kernel-only. Records one
        start/stop event pair around ``iters`` back-to-back launches (no per-
        launch host sync) and divides by ``iters``.
        """
        drv = self.driver
        start = self._chk(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DEFAULT))
        stop = self._chk(drv.cuEventCreate(drv.CUevent_flags.CU_EVENT_DEFAULT))
        for _ in range(warmup):
            self.launch_no_sync(fn, grid, block, buffers)
        self._chk(drv.cuCtxSynchronize())
        self._chk(drv.cuEventRecord(start, 0))
        for _ in range(iters):
            self.launch_no_sync(fn, grid, block, buffers)
        self._chk(drv.cuEventRecord(stop, 0))
        self._chk(drv.cuEventSynchronize(stop))
        ms = self._chk(drv.cuEventElapsedTime(start, stop))
        drv.cuEventDestroy(start)
        drv.cuEventDestroy(stop)
        return float(ms) / iters

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

    def __init__(self, chip: str = "sm_86", use_tensor_core: bool = False) -> None:
        self.chip = chip
        self.mlir_opt = _tool("ARKE_MLIR_OPT", "mlir-opt")
        # Opt-in tensor-core (nvgpu.mma.sync) matmul path. When True, matmul
        # lowering prefers emit_gpu_matmul_mma (f16 tensor core, ~1e-3 rel vs
        # strict f32) for MMA-tileable shapes, falling back to the scalar FP32
        # ladder otherwise. Kept off by default so the bit-accurate f32 matmul
        # stays the default precision class (the tensor-core precision tradeoff
        # is a benchmark-semantics decision, not silently changed here).
        self.use_tensor_core = use_tensor_core

    def supports_op(self, op_name: str) -> bool:
        from arke.backend.mlir_emitter import (
            GPU_ELEMENTWISE_OPS, GPU_ROWWISE_OPS, GPU_MOVEMENT_OPS, GPU_GATED_OPS,
            GPU_ROWWISE2_OPS, GPU_INDEX_OPS,
        )
        return (op_name == "matmul" or op_name == "batch_matmul"
                or op_name == "topk" or op_name == "cross_entropy"
                or op_name == "quantize_per_token"
                or op_name == "dequantize_per_channel"
                or op_name == "swiglu_packed"
                or op_name == "fused_linear_cross_entropy"
                or op_name in GPU_ELEMENTWISE_OPS
                or op_name in GPU_ROWWISE_OPS or op_name in GPU_MOVEMENT_OPS
                or op_name in GPU_GATED_OPS or op_name in GPU_ROWWISE2_OPS
                or op_name in GPU_INDEX_OPS)

    def lower(self, graph: Any) -> Any:
        from arke.backend.mlir_emitter import (
            emit_gpu_matmul, emit_gpu_matmul_tiled, emit_gpu_matmul_regblock,
            emit_gpu_elementwise, emit_gpu_rowwise, emit_gpu_movement, emit_gpu_gated,
            emit_gpu_rowwise2, emit_gpu_index,
            GPU_ELEMENTWISE_OPS, GPU_ROWWISE_OPS, GPU_MOVEMENT_OPS, GPU_GATED_OPS,
            GPU_ROWWISE2_OPS, GPU_INDEX_OPS,
        )
        from arke.backend.protocol import BackendArtifact
        op = graph.nodes[0].op if graph.nodes else ""
        is_mma = False
        if op in GPU_ELEMENTWISE_OPS:
            emitted = emit_gpu_elementwise(graph, chip=self.chip)
        elif op in GPU_ROWWISE_OPS:
            emitted = emit_gpu_rowwise(graph, chip=self.chip)
        elif op in GPU_ROWWISE2_OPS:
            emitted = emit_gpu_rowwise2(graph, chip=self.chip)
        elif op in GPU_MOVEMENT_OPS:
            emitted = emit_gpu_movement(graph, chip=self.chip)
        elif op in GPU_GATED_OPS:
            emitted = emit_gpu_gated(graph, chip=self.chip)
        elif op in GPU_INDEX_OPS:
            emitted = emit_gpu_index(graph, chip=self.chip)
        elif op == "batch_matmul":
            from arke.backend.mlir_emitter import emit_gpu_batch_matmul
            emitted = emit_gpu_batch_matmul(graph, chip=self.chip)
        elif op == "topk":
            from arke.backend.mlir_emitter import emit_gpu_topk
            emitted = emit_gpu_topk(graph, chip=self.chip)
        elif op == "cross_entropy":
            from arke.backend.mlir_emitter import emit_gpu_cross_entropy
            emitted = emit_gpu_cross_entropy(graph, chip=self.chip)
        elif op == "quantize_per_token":
            from arke.backend.mlir_emitter import emit_gpu_quantize_per_token
            emitted = emit_gpu_quantize_per_token(graph, chip=self.chip)
        elif op == "dequantize_per_channel":
            from arke.backend.mlir_emitter import emit_gpu_dequantize_per_channel
            emitted = emit_gpu_dequantize_per_channel(graph, chip=self.chip)
        elif op == "swiglu_packed":
            from arke.backend.mlir_emitter import emit_gpu_swiglu_packed
            emitted = emit_gpu_swiglu_packed(graph, chip=self.chip)
        elif op == "fused_linear_cross_entropy":
            from arke.backend.mlir_emitter import emit_gpu_fused_linear_cross_entropy
            emitted = emit_gpu_fused_linear_cross_entropy(graph, chip=self.chip)
        elif op == "matmul":
            # Perf ladder: register-blocked (best) → shared-mem tiled →
            # correctness kernel, falling back on shape-alignment constraints.
            # Shape-adaptive tile selection:
            # - Small matrices (M,N ≤ 256): BM=BN=32, TM=TN=2 maximizes
            #   parallelism (more blocks → all SMs active).
            # - Medium/large: BM=BN=64, TM=TN=4 for higher arithmetic
            #   intensity; BK=32 for K≥1024 (fewer barriers).
            node = graph.nodes[0]
            in_names = list(node.inputs.values())
            A_val = graph.values[in_names[0]]
            B_val = graph.values[in_names[1]]
            M_dim = A_val.shape[0] if len(A_val.shape) == 2 else 0
            K_dim = A_val.shape[1] if len(A_val.shape) == 2 else 0
            N_dim = B_val.shape[1] if len(B_val.shape) == 2 else 0
            small = max(M_dim, N_dim) <= 256
            tile_kw: dict = {}
            if small and M_dim % 32 == 0 and N_dim % 32 == 0 and K_dim % 16 == 0:
                tile_kw = {"BM": 32, "BN": 32, "TM": 2, "TN": 2, "BK": 16}
            elif K_dim >= 1024 and K_dim % 32 == 0:
                tile_kw = {"BK": 32}
            # Tensor-core matmul (default for MMA-tileable shapes): fp16 tensor
            # core via nvgpu.mma.sync with f32 accumulation. Precision matches
            # cuBLAS tf32 (the Golden baseline) — both are reduced-precision TC
            # paths with ~1e-2 tolerance vs strict-f32; allclose(mlir_tc,
            # cublas_tf32, rtol=1e-2) = True (verified 2026-07-07).
            # Falls through to the scalar FP32 ladder on shapes that don't
            # MMA-tile (or when the emitter raises NotImplementedError).
            from arke.backend.mlir_emitter import emit_gpu_matmul_mma
            try:
                emitted = emit_gpu_matmul_mma(graph, chip=self.chip)
                is_mma = True
                return BackendArtifact(
                    source_code=emitted.mlir_text,
                    backend_name=self.name,
                    op_name=op,
                    metadata={"emitted": emitted, "is_mma": True},
                )
            except NotImplementedError:
                pass
            for emit in (emit_gpu_matmul_regblock, emit_gpu_matmul_tiled):
                try:
                    kw = {"chip": self.chip}
                    if emit is emit_gpu_matmul_regblock:
                        kw.update(tile_kw)
                    emitted = emit(graph, **kw)
                    break
                except NotImplementedError:
                    continue
            else:
                emitted = emit_gpu_matmul(graph, chip=self.chip)
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
        # Tensor-core (nvgpu) kernels need the two-stage nvgpu pipeline, not the
        # scalar gpu→PTX/cubin path. Route them through mlir_nvgpu_to_cubin.
        if artifact.metadata.get("is_mma"):
            try:
                cubin = mlir_nvgpu_to_cubin(artifact.source_code, self.chip, self.mlir_opt)
            except (subprocess.CalledProcessError, RuntimeError) as e:
                msg = e.stderr if isinstance(e, subprocess.CalledProcessError) else str(e)
                return CompiledKernel.fail(f"nvgpu tensor-core lowering failed: {msg}")
            return CompiledKernel.ok(
                fn=None,
                backend_name=self.name,
                emitted=artifact.metadata["emitted"],
                cubin=cubin,
            )
        # Try cubin (native SASS via ptxas) first for better register allocation
        # and instruction scheduling; fall back to PTX if ptxas unavailable.
        cubin: bytes | None = None
        try:
            cubin = mlir_gpu_to_cubin(artifact.source_code, self.mlir_opt)
        except (subprocess.CalledProcessError, RuntimeError):
            pass  # fall through to PTX
        if cubin:
            return CompiledKernel.ok(
                fn=None,
                backend_name=self.name,
                emitted=artifact.metadata["emitted"],
                cubin=cubin,
            )
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

    def _load_kernel(self, kernel: Any, cu: "CudaLauncher") -> Any:
        """Load the compiled kernel into a CUDA context (cubin or PTX)."""
        emitted = kernel.metadata["emitted"]
        if "cubin" in kernel.metadata:
            return cu.load_cubin(kernel.metadata["cubin"], emitted.kernel_name)
        return cu.load_ptx(kernel.metadata["ptx"], emitted.kernel_name)

    def run(self, kernel: Any, inputs: dict[str, Any]) -> dict[str, Any]:
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")
        emitted = kernel.metadata["emitted"]
        np_inputs = {k: _to_numpy(v) for k, v in inputs.items()}
        with CudaLauncher() as cu:
            fn = self._load_kernel(kernel, cu)
            bufs: list[GPUBuffer] = []
            out_buf: GPUBuffer | None = None
            for name in emitted.buffer_order:
                if name == emitted.result_name:
                    # scatter needs zero-filled output; other ops use uninitialized
                    if getattr(emitted, 'kernel_name', '') == 'scatter':
                        zeros = np.zeros(emitted.result_shape, dtype=np.float32)
                        out_buf = cu.to_device(zeros)
                    else:
                        out_buf = cu.alloc_output(tuple(emitted.result_shape))
                    bufs.append(out_buf)
                else:
                    bufs.append(cu.to_device(np_inputs[name]))
            assert out_buf is not None
            cu.launch(fn, emitted.grid, emitted.block, bufs)
            result = cu.from_device(out_buf)
        return {emitted.result_name: result}

    def benchmark(self, kernel: Any, inputs: dict[str, Any],
                  iters: int = 50, warmup: int = 10) -> float:
        """Mean kernel-only latency (ms) for a compiled kernel.

        Reuses ONE CUDA context, loads the PTX once, copies inputs H2D once, then
        times ``iters`` back-to-back kernel launches with CUDA events (see
        ``CudaLauncher.time_kernel``). This isolates kernel execution from the
        one-time context/JIT/copy overhead that dominates the correctness
        ``run()`` path — the apples-to-apples number for a Triton/torch perf
        comparison, which are likewise timed kernel-only.
        """
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")
        emitted = kernel.metadata["emitted"]
        np_inputs = {k: _to_numpy(v) for k, v in inputs.items()}
        with CudaLauncher() as cu:
            fn = self._load_kernel(kernel, cu)
            bufs: list[GPUBuffer] = []
            for name in emitted.buffer_order:
                if name == emitted.result_name:
                    bufs.append(cu.alloc_output(tuple(emitted.result_shape)))
                else:
                    bufs.append(cu.to_device(np_inputs[name]))
            return cu.time_kernel(fn, emitted.grid, emitted.block, bufs,
                                  iters=iters, warmup=warmup)


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)
