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
from arke.backend.llvm_emitter import emit_llvm_ir_matmul
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


# -- Backend --

class LLVMBackend:
    """ArkeBackend that generates LLVM IR, compiles via llc+ptxas, runs via CUDA driver."""

    name = "llvm"

    def __init__(self, chip: str = "sm_86") -> None:
        self.chip = chip
        self.llc = _find_llc()
        self.ptxas = _find_ptxas()
        self._cache_dir = os.path.join(
            tempfile.gettempdir(), "arke_llvm_cache"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    def supports_op(self, op_name: str) -> bool:
        """P5-S1: only matmul."""
        return op_name == "matmul"

    def lower(self, graph: IRGraph) -> BackendArtifact:
        """Generate LLVM IR source from IRGraph."""
        node = graph.nodes[0]
        op = node.op

        if op == "matmul":
            emitted = emit_llvm_ir_matmul(graph, chip=self.chip)
        else:
            raise ValueError(f"LLVM backend does not support op {op!r} yet (P5-S1: matmul only)")

        return BackendArtifact(
            source_code=emitted.source,
            backend_name=self.name,
            op_name=op,
            metadata={"emitted": emitted, "graph_name": graph.name},
        )

    def compile(self, artifact: BackendArtifact) -> CompiledKernel:
        """Compile LLVM IR -> PTX (via llc) -> cubin (via ptxas)."""
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
        ptx_path = ll_path.replace(".ll", ".ptx")
        cubin_path = ll_path.replace(".ll", ".cubin")

        if not os.path.exists(cubin_path):
            # Write .ll
            with open(ll_path, "w") as f:
                f.write(artifact.source_code)

            sm = self.chip.replace("sm_", "")

            # Step 1: llc -> PTX
            cmd_llc = [
                self.llc,
                "-march=nvptx64",
                f"-mcpu=sm_{sm}",
                "-o", ptx_path,
                ll_path,
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

    @staticmethod
    def _chk(driver: Any, ret: Any) -> Any:
        t = _as_tuple(ret)
        err = t[0]
        if err != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA driver error: {err}")
        rest = t[1:]
        return rest[0] if len(rest) == 1 else rest
