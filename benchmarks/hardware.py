# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Hardware info collection for benchmark context."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class HardwareInfo:
    """GPU and system hardware info."""

    gpu_name: str = ""
    gpu_memory_mb: int = 0
    gpu_sm_count: int = 0
    gpu_compute_capability: str = ""
    cuda_version: str = ""
    driver_version: str = ""
    triton_version: str = ""
    torch_version: str = ""
    python_version: str = ""
    os: str = ""
    cpu: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def collect_hardware_info() -> HardwareInfo:
    """Collect hardware info from the current environment."""
    info = HardwareInfo()
    info.python_version = platform.python_version()
    info.os = f"{platform.system()} {platform.release()}"

    try:
        info.cpu = platform.processor() or "unknown"
    except Exception:
        info.cpu = "unknown"

    try:
        import torch

        info.torch_version = torch.__version__
        if torch.cuda.is_available():
            info.gpu_name = torch.cuda.get_device_name(0)
            info.gpu_memory_mb = (
                torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
            )
            info.gpu_sm_count = (
                torch.cuda.get_device_properties(0).multi_processor_count
            )
            cap = torch.cuda.get_device_capability(0)
            info.gpu_compute_capability = f"{cap[0]}.{cap[1]}"
            info.cuda_version = torch.version.cuda or ""
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info.driver_version = result.stdout.strip()
    except Exception:
        pass

    try:
        import triton

        info.triton_version = triton.__version__
    except Exception:
        pass

    return info
