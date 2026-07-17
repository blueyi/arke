# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""LLVMBackend -- Phase 5 (P5-S1) ArkeBackend implementation.

Generates LLVM IR text from IRGraph, compiles to PTX via ``llc``, then to
cubin via ``ptxas``, and executes via ``cuda.bindings.driver`` (CUDA driver API).

Pipeline:  IRGraph -> LLVM IR (.ll) -> llc -> PTX (.ptx) -> ptxas -> cubin -> driver launch

Design ref: docs/architecture/arke-compiler-infrastructure.md
Extension seam: arke/backend/protocol.py (ArkeBackend + BackendRegistry)
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np

from arke.backend.cuda_c_backend import CudaCKernel, _as_tuple, _ir_dtype_to_numpy
from arke.backend.llvm_emitter import emit_llvm_ir_matmul, LLVM_EMITTERS
from arke.backend.protocol import BackendArtifact, CompiledKernel
from arke.ir.graph import IRGraph


# -- toolchain discovery --

def _find_llc() -> str | None:
    """Find llc binary — prefer LLVM 20 (MLIR_HOME) over system LLVM."""
    # 1. Explicit env override
    p = os.environ.get("ARKE_LLC")
    if p and os.path.isfile(p):
        return p
    # 2. MLIR_HOME/bin/llc (LLVM 20, aligned with Triton 3.2 / PyTorch 2.6)
    mlir_home = os.environ.get("MLIR_HOME", "")
    if mlir_home:
        candidate = os.path.join(mlir_home, "bin", "llc")
        if os.path.isfile(candidate):
            return candidate
    # 3. Well-known LLVM 20 paths (source build + deb extract)
    for candidate in [
        os.path.expanduser("~/opt/llvm20-src/usr/lib/llvm-20/bin/llc"),
        os.path.expanduser("~/opt/mlir20/root/usr/lib/llvm-20/bin/llc"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    # 4. PATH fallback
    return shutil.which("llc")


def _find_llvm_link() -> str | None:
    """Find llvm-link binary — same search order as llc."""
    p = os.environ.get("ARKE_LLVM_LINK")
    if p and os.path.isfile(p):
        return p
    mlir_home = os.environ.get("MLIR_HOME", "")
    if mlir_home:
        candidate = os.path.join(mlir_home, "bin", "llvm-link")
        if os.path.isfile(candidate):
            return candidate
    for candidate in [
        os.path.expanduser("~/opt/llvm20-src/usr/lib/llvm-20/bin/llvm-link"),
        os.path.expanduser("~/opt/mlir20/root/usr/lib/llvm-20/bin/llvm-link"),
    ]:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("llvm-link")


def _find_libdevice() -> str | None:
    """Find NVIDIA libdevice.10.bc bitcode library."""
    p = os.environ.get("ARKE_LIBDEVICE")
    if p and os.path.isfile(p):
        return p
    for candidate in [
        "/usr/local/cuda/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda-13.2/nvvm/libdevice/libdevice.10.bc",
        "/usr/local/cuda-12.4/nvvm/libdevice/libdevice.10.bc",
    ]:
        if os.path.isfile(candidate):
            return candidate
    # Try via CUDA_HOME
    cuda_home = os.environ.get("CUDA_HOME", "")
    if cuda_home:
        c = os.path.join(cuda_home, "nvvm", "libdevice", "libdevice.10.bc")
        if os.path.isfile(c):
            return c
    return None


def _find_ptxas() -> str | None:
    """Find ptxas binary."""
    p = os.environ.get("ARKE_PTXAS")
    if p and os.path.isfile(p):
        return p
    for candidate in ["/usr/local/cuda/bin/ptxas", "/usr/local/cuda-13.2/bin/ptxas"]:
        if os.path.isfile(candidate):
            return candidate
    return shutil.which("ptxas")


def llvm_toolchain_available() -> bool:
    """True iff llc + ptxas + CUDA driver API are all available."""
    if not _find_llc() or not _find_ptxas():
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


# -- Cached execution context --

class _CachedModule:
    """Pre-loaded CUDA module + pre-allocated GPU buffers for zero-overhead re-execution.

    Lifecycle:
        1. Created by LLVMBackend.prepare(kernel)
        2. Reused across run_fast() calls — no module load, no alloc, no free
        3. Freed explicitly via release() or when LLVMBackend.release_all() is called

    The module holds:
        - Loaded CUmodule + CUfunction (from cubin)
        - Pre-allocated GPU device pointers for all inputs + output
        - Pre-built kernel argument array (pointer layout doesn't change)
        - Grid/block dimensions
    """

    __slots__ = (
        "driver", "module", "function", "emitted",
        "gpu_ptrs", "_alloc_list", "arg_buffers", "arg_ptrs_array",
        "gx", "gy", "gz", "bx", "by", "bz", "shared_mem",
        "np_dtypes", "_released",
    )

    def __init__(
        self,
        driver: Any,
        module: Any,
        function: Any,
        emitted: CudaCKernel,
        gpu_ptrs: dict[str, int],
        alloc_list: list[int],
        arg_buffers: list[np.ndarray],
        arg_ptrs_array: np.ndarray,
    ) -> None:
        self.driver = driver
        self.module = module
        self.function = function
        self.emitted = emitted
        self.gpu_ptrs = gpu_ptrs
        self._alloc_list = alloc_list
        self.arg_buffers = arg_buffers
        self.arg_ptrs_array = arg_ptrs_array
        self.gx, self.gy, self.gz = emitted.grid
        self.bx, self.by, self.bz = emitted.block
        self.shared_mem = emitted.shared_mem
        self.np_dtypes = {
            name: _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
            for name in emitted.param_names
        }
        self._released = False

    def release(self) -> None:
        """Free GPU memory and unload module."""
        if self._released:
            return
        self._released = True
        for dptr in self._alloc_list:
            self.driver.cuMemFree(dptr)
        self.driver.cuModuleUnload(self.module)

    def __del__(self) -> None:
        self.release()


# -- Backend --

class LLVMBackend:
    """ArkeBackend that generates LLVM IR, compiles via llc+ptxas, runs via CUDA driver.

    Execution modes:
        1. run(kernel, inputs)      — legacy: load + alloc + H2D + launch + D2H + free per call
        2. prepare(kernel, shapes)  — pre-load module + pre-alloc GPU buffers (once)
           run_fast(cached, inputs) — only H2D + launch + sync + D2H (zero module/alloc overhead)
           release(cached)          — free resources when done
    """

    name = "llvm"

    def __init__(self, chip: str = "sm_86") -> None:
        self.chip = chip
        self.llc = _find_llc()
        self.ptxas = _find_ptxas()
        self._cache_dir = os.path.join(
            tempfile.gettempdir(), "arke_llvm_cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)
        self._cached_modules: list[_CachedModule] = []

    def supports_op(self, op_name: str) -> bool:
        """P5-S2: all 46 ops supported."""
        return op_name in LLVM_EMITTERS

    def lower(self, graph: IRGraph) -> BackendArtifact:
        """Generate LLVM IR source from IRGraph."""
        node = graph.nodes[0]
        op = node.op

        if op not in LLVM_EMITTERS:
            raise ValueError(f"LLVM backend does not support op {op!r}")

        emitter = LLVM_EMITTERS[op]
        emitted = emitter(graph, chip=self.chip)
        return BackendArtifact(
            source_code=emitted.source,
            backend_name=self.name,
            op_name=op,
            metadata={"emitted": emitted, "graph_name": graph.name},
        )

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Compile LLVM IR -> PTX (via llc) -> cubin (via ptxas).

        If the IR references libdevice functions (__nv_expf, __nv_logf, etc.),
        links libdevice.10.bc before compilation via llvm-link.
        """
        if not self.llc:
            return CompiledKernel.fail("llc not found. Install LLVM with nvptx64 support.")
        if not self.ptxas:
            return CompiledKernel.fail(
                "ptxas not found. Set ARKE_PTXAS or add /usr/local/cuda/bin to PATH."
            )

        emitted: CudaCKernel = artifact.metadata["emitted"]

        # Hash for caching
        src_hash = hashlib.sha256(artifact.source_code.encode()).hexdigest()[:16]
        ll_path = os.path.join(self._cache_dir, f"{emitted.kernel_name}_{src_hash}.ll")
        linked_path = ll_path.replace(".ll", "_linked.ll")
        ptx_path = ll_path.replace(".ll", ".ptx")
        cubin_path = ll_path.replace(".ll", ".cubin")

        if not os.path.exists(cubin_path):
            # Write .ll
            with open(ll_path, "w") as f:
                f.write(artifact.source_code)

            sm = self.chip.replace("sm_", "")

            # Step 0: Link libdevice if needed (resolves __nv_expf etc.)
            needs_libdevice = "__nv_" in artifact.source_code
            compile_input = ll_path

            if needs_libdevice:
                libdevice = _find_libdevice()
                llvm_link = _find_llvm_link()
                if libdevice and llvm_link:
                    cmd_link = [
                        llvm_link,
                        ll_path,
                        libdevice,
                        "-o", linked_path,
                        "--internalize",
                    ]
                    try:
                        subprocess.run(cmd_link, capture_output=True, text=True,
                                       check=True, timeout=60)
                        compile_input = linked_path
                    except subprocess.CalledProcessError as e:
                        return CompiledKernel.fail(f"llvm-link failed:\n{e.stderr}")
                    except subprocess.TimeoutExpired:
                        return CompiledKernel.fail("llvm-link timed out (60s)")
                elif not libdevice:
                    return CompiledKernel.fail(
                        "libdevice.10.bc not found. Install CUDA toolkit."
                    )

            # Step 1: llc -> PTX
            cmd_llc = [
                self.llc,
                "-march=nvptx64",
                f"-mcpu=sm_{sm}",
                "-O2",
                "-o", ptx_path,
                compile_input,
            ]
            try:
                subprocess.run(cmd_llc, capture_output=True, text=True, check=True, timeout=120)
            except subprocess.CalledProcessError as e:
                return CompiledKernel.fail(f"llc compilation failed:\n{e.stderr}")
            except subprocess.TimeoutExpired:
                return CompiledKernel.fail("llc compilation timed out (120s)")

            # Step 2: ptxas -> cubin
            cmd_ptxas = [
                self.ptxas,
                f"--gpu-name=sm_{sm}",
                "-O3",
                "-o", cubin_path,
                ptx_path,
            ]
            try:
                subprocess.run(cmd_ptxas, capture_output=True, text=True, check=True, timeout=120)
            except subprocess.CalledProcessError as e:
                return CompiledKernel.fail(f"ptxas assembly failed:\n{e.stderr}")
            except subprocess.TimeoutExpired:
                return CompiledKernel.fail("ptxas assembly timed out (120s)")

        # Read cubin
        with open(cubin_path, "rb") as f:
            cubin = f.read()

        return CompiledKernel.ok(
            fn=None,
            backend_name=self.name,
            cubin=cubin,
            cubin_path=cubin_path,
            emitted=emitted,
        )

    def run(self, kernel: CompiledKernel, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute compiled LLVM-IR kernel via cuda.bindings.driver."""
        if not kernel.success:
            raise RuntimeError(f"Cannot run failed kernel: {kernel.error}")

        emitted: CudaCKernel = kernel.metadata["emitted"]
        cubin: bytes = kernel.metadata["cubin"]

        return self._run_driver(emitted, cubin, inputs)

    def _run_driver(
        self, emitted: CudaCKernel, cubin: bytes, inputs: dict[str, Any]
    ) -> dict[str, Any]:
        """Load cubin via CUDA driver API, launch kernel, read back result.

        This is functionally identical to CudaCBackend._run_driver.
        """
        from cuda.bindings import driver

        # Ensure CUDA initialized
        _as_tuple(driver.cuInit(0))

        # Get/create context (reuse torch's if available)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
        except ImportError:
            pass

        err, ctx = _as_tuple(driver.cuCtxGetCurrent())
        if err != driver.CUresult.CUDA_SUCCESS or int(ctx) == 0:
            dev = self._chk(driver, driver.cuDeviceGet(0))
            ctx = self._chk(driver, driver.cuCtxCreate(
                driver.CUctxCreateParams(), 0, dev
            ))

        # Load module
        mod = self._chk(driver, driver.cuModuleLoadData(cubin))
        func = self._chk(driver, driver.cuModuleGetFunction(
            mod, emitted.kernel_name.encode()
        ))

        # Convert inputs to numpy + allocate GPU memory
        np_inputs: dict[str, np.ndarray] = {}
        for name in emitted.param_names:
            if name == emitted.output_name:
                continue
            val = inputs.get(name)
            if val is None:
                raise ValueError(f"Missing input: {name}")
            try:
                import torch as _torch
                if isinstance(val, _torch.Tensor):
                    val = val.detach().cpu().numpy()
            except ImportError:
                pass
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
            np_inputs[name] = np.ascontiguousarray(val, dtype=np_dtype)

        # GPU alloc + H2D
        gpu_ptrs: dict[str, int] = {}
        allocs: list[int] = []

        for name in emitted.param_names:
            if name == emitted.output_name:
                out_shape = emitted.shapes[name]
                np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
                nbytes = int(np.prod(out_shape)) * np_dtype.itemsize
                dptr = self._chk(driver, driver.cuMemAlloc(nbytes))
                gpu_ptrs[name] = int(dptr)
                allocs.append(int(dptr))
            else:
                arr = np_inputs[name]
                dptr = self._chk(driver, driver.cuMemAlloc(arr.nbytes))
                self._chk(driver, driver.cuMemcpyHtoD(dptr, arr.ctypes.data, arr.nbytes))
                gpu_ptrs[name] = int(dptr)
                allocs.append(int(dptr))

        # Build kernel args
        arg_buffers: list[np.ndarray] = []
        for arg_type, arg_val in emitted.kernel_args:
            if arg_type == "ptr":
                arg_buffers.append(np.array([gpu_ptrs[arg_val]], dtype=np.uint64))
            elif arg_type == "int":
                arg_buffers.append(np.array([arg_val], dtype=np.int32))
            elif arg_type == "float":
                arg_buffers.append(np.array([arg_val], dtype=np.float32))

        arg_ptrs = np.array([a.ctypes.data for a in arg_buffers], dtype=np.uint64)

        gx, gy, gz = emitted.grid
        bx, by, bz = emitted.block

        self._chk(driver, driver.cuLaunchKernel(
            func,
            gx, gy, gz,
            bx, by, bz,
            emitted.shared_mem,
            0,  # default stream
            arg_ptrs.ctypes.data,
            0,
        ))
        self._chk(driver, driver.cuCtxSynchronize())

        # D2H: read output
        out_shape = emitted.shapes[emitted.output_name]
        np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(emitted.output_name, "float32"))
        result = np.empty(out_shape, dtype=np_dtype)
        nbytes = result.nbytes
        self._chk(driver, driver.cuMemcpyDtoH(
            result.ctypes.data, gpu_ptrs[emitted.output_name], nbytes
        ))

        # Free GPU memory
        for dptr in allocs:
            driver.cuMemFree(dptr)
        driver.cuModuleUnload(mod)

        return {emitted.output_name: result}

    # -- Cached execution API (P5-S3: eliminate 40-57× E2E overhead) --

    def prepare(self, kernel: CompiledKernel) -> _CachedModule:
        """Pre-load cubin module and pre-allocate GPU buffers.

        Call once per kernel shape. Returns a _CachedModule that can be reused
        across many run_fast() calls with zero module-load and alloc overhead.

        Usage:
            cached = backend.prepare(compiled_kernel)
            for batch in data:
                result = backend.run_fast(cached, batch)
            cached.release()
        """
        if not kernel.success:
            raise RuntimeError(f"Cannot prepare failed kernel: {kernel.error}")

        from cuda.bindings import driver as drv

        emitted: CudaCKernel = kernel.metadata["emitted"]
        cubin: bytes = kernel.metadata["cubin"]

        # Ensure CUDA context
        _as_tuple(drv.cuInit(0))
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.init()
        except ImportError:
            pass

        err, ctx = _as_tuple(drv.cuCtxGetCurrent())
        if err != drv.CUresult.CUDA_SUCCESS or int(ctx) == 0:
            dev = self._chk(drv, drv.cuDeviceGet(0))
            ctx = self._chk(drv, drv.cuCtxCreate(
                drv.CUctxCreateParams(), 0, dev
            ))

        # Load module + function (once)
        mod = self._chk(drv, drv.cuModuleLoadData(cubin))
        func = self._chk(drv, drv.cuModuleGetFunction(
            mod, emitted.kernel_name.encode()
        ))

        # Pre-allocate GPU memory for all params (inputs + output)
        gpu_ptrs: dict[str, int] = {}
        alloc_list: list[int] = []

        for name in emitted.param_names:
            shape = emitted.shapes[name]
            np_dtype = _ir_dtype_to_numpy(emitted.dtypes.get(name, "float32"))
            nbytes = int(np.prod(shape)) * np_dtype.itemsize
            dptr = self._chk(drv, drv.cuMemAlloc(nbytes))
            gpu_ptrs[name] = int(dptr)
            alloc_list.append(int(dptr))

        # Build kernel arg array (once — pointers don't change between runs)
        arg_buffers: list[np.ndarray] = []
        for arg_type, arg_val in emitted.kernel_args:
            if arg_type == "ptr":
                arg_buffers.append(np.array([gpu_ptrs[arg_val]], dtype=np.uint64))
            elif arg_type == "int":
                arg_buffers.append(np.array([arg_val], dtype=np.int32))
            elif arg_type == "float":
                arg_buffers.append(np.array([arg_val], dtype=np.float32))

        arg_ptrs_array = np.array([a.ctypes.data for a in arg_buffers], dtype=np.uint64)

        cached = _CachedModule(
            driver=drv,
            module=mod,
            function=func,
            emitted=emitted,
            gpu_ptrs=gpu_ptrs,
            alloc_list=alloc_list,
            arg_buffers=arg_buffers,
            arg_ptrs_array=arg_ptrs_array,
        )
        self._cached_modules.append(cached)
        return cached

    def run_fast(self, cached: _CachedModule, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute kernel using pre-loaded module + pre-allocated buffers.

        Only performs: H2D copy → kernel launch → sync → D2H copy.
        No module load, no GPU alloc/free — that's all in prepare().

        Returns dict with output numpy array.
        """
        drv = cached.driver
        emitted = cached.emitted

        # H2D: copy input data to pre-allocated GPU buffers
        for name in emitted.param_names:
            if name == emitted.output_name:
                continue
            val = inputs.get(name)
            if val is None:
                raise ValueError(f"Missing input: {name}")
            try:
                import torch as _torch
                if isinstance(val, _torch.Tensor):
                    val = val.detach().cpu().numpy()
            except ImportError:
                pass
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            arr = np.ascontiguousarray(val, dtype=cached.np_dtypes[name])
            self._chk(drv, drv.cuMemcpyHtoD(
                cached.gpu_ptrs[name], arr.ctypes.data, arr.nbytes
            ))

        # Launch kernel
        self._chk(drv, drv.cuLaunchKernel(
            cached.function,
            cached.gx, cached.gy, cached.gz,
            cached.bx, cached.by, cached.bz,
            cached.shared_mem,
            0,  # default stream
            cached.arg_ptrs_array.ctypes.data,
            0,
        ))
        self._chk(drv, drv.cuCtxSynchronize())

        # D2H: read output
        out_shape = emitted.shapes[emitted.output_name]
        np_dtype = cached.np_dtypes[emitted.output_name]
        result = np.empty(out_shape, dtype=np_dtype)
        self._chk(drv, drv.cuMemcpyDtoH(
            result.ctypes.data, cached.gpu_ptrs[emitted.output_name], result.nbytes
        ))

        return {emitted.output_name: result}

    def run_fast_no_copy(self, cached: _CachedModule) -> None:
        """Launch kernel on already-resident GPU data — kernel-only timing.

        Does NOT copy H2D or D2H. Use for measuring pure kernel latency.
        Caller must have already done H2D via prepare() or run_fast().
        """
        drv = cached.driver
        self._chk(drv, drv.cuLaunchKernel(
            cached.function,
            cached.gx, cached.gy, cached.gz,
            cached.bx, cached.by, cached.bz,
            cached.shared_mem,
            0,
            cached.arg_ptrs_array.ctypes.data,
            0,
        ))
        self._chk(drv, drv.cuCtxSynchronize())

    def benchmark_cached(self, cached: _CachedModule,
                         iters: int = 100, warmup: int = 30) -> float:
        """Mean kernel-only latency (ms) via CUDA events.

        Apples-to-apples with CudaCBackend.benchmark: `iters` back-to-back
        kernel launches inside a single CUDA-event region with ONE sync,
        rather than sync-per-launch. Caller must have done H2D via prepare()
        or run_fast() so GPU data is resident.
        """
        drv = cached.driver

        err, start = _as_tuple(drv.cuEventCreate(0))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuEventCreate failed: {err}")
        err, stop = _as_tuple(drv.cuEventCreate(0))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuEventCreate failed: {err}")

        def _launch():
            self._chk(drv, drv.cuLaunchKernel(
                cached.function,
                cached.gx, cached.gy, cached.gz,
                cached.bx, cached.by, cached.bz,
                cached.shared_mem, 0,
                cached.arg_ptrs_array.ctypes.data, 0,
            ))

        for _ in range(warmup):
            _launch()
        self._chk(drv, drv.cuCtxSynchronize())

        self._chk(drv, drv.cuEventRecord(start, 0))
        for _ in range(iters):
            _launch()
        self._chk(drv, drv.cuEventRecord(stop, 0))
        self._chk(drv, drv.cuEventSynchronize(stop))
        elapsed_ms = self._chk(drv, drv.cuEventElapsedTime(start, stop))

        drv.cuEventDestroy(start)
        drv.cuEventDestroy(stop)
        return float(elapsed_ms) / iters

    def release(self, cached: _CachedModule) -> None:
        """Release a cached module's GPU resources."""
        cached.release()
        if cached in self._cached_modules:
            self._cached_modules.remove(cached)

    def release_all(self) -> None:
        """Release all cached modules."""
        for cm in self._cached_modules:
            cm.release()
        self._cached_modules.clear()

    @staticmethod
    def _chk(driver: Any, ret: Any) -> Any:
        t = _as_tuple(ret)
        err = t[0]
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
        rest = t[1:]
        return rest[0] if len(rest) == 1 else rest
