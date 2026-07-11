# Copyright 2026 Arke Contributors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for GPU kernel tuning policy (arke/backend/gpu_tuning.py)."""

import pytest

from arke.backend.gpu_tuning import (
    DEFAULT_GPU,
    GPUProfile,
    MMA_DEFAULT,
    MMAConfig,
    elementwise_block_size,
    matmul_mma_config,
    rowwise_block_size,
    select_kernel_family,
)


class TestRowwiseBlockSize:
    """Verify block-size selection heuristic for rowwise ops."""

    def test_small_D_uses_256(self):
        assert rowwise_block_size(512) == 256
        assert rowwise_block_size(1024) == 256
        assert rowwise_block_size(2048) == 256

    def test_large_D_uses_512(self):
        assert rowwise_block_size(4096) == 512
        assert rowwise_block_size(8192) == 512
        assert rowwise_block_size(16384) == 512

    def test_boundary(self):
        assert rowwise_block_size(4095) == 256
        assert rowwise_block_size(4096) == 512

    def test_custom_hw_profile(self):
        """Policy function accepts a hardware profile for future extensibility."""
        hw = GPUProfile(chip="sm_90", num_sm=128)
        # Same logic for now, but proves the interface works
        assert rowwise_block_size(4096, hw=hw) == 512

    def test_result_is_power_of_2(self):
        for D in [64, 256, 512, 1024, 2048, 4096, 8192]:
            bs = rowwise_block_size(D)
            assert bs & (bs - 1) == 0, f"block_size={bs} not power of 2"
            assert bs <= 1024


class TestElementwiseBlockSize:
    def test_default_256(self):
        assert elementwise_block_size(1024 * 1024) == 256
        assert elementwise_block_size(512 * 512) == 256


class TestMMAConfig:
    def test_default_config_properties(self):
        cfg = MMA_DEFAULT
        assert cfg.BM == 64   # 2*2*16
        assert cfg.BN == 128  # 2*4*16
        assert cfg.threads_per_block == 128  # 2*2*32

    def test_tileable_shape(self):
        cfg = matmul_mma_config(1024, 1024, 1024)
        assert cfg is not None
        assert cfg == MMA_DEFAULT

    def test_untileable_shape_returns_none(self):
        # 300 % 64 != 0
        cfg = matmul_mma_config(300, 300, 300)
        assert cfg is None

    def test_boundary_shapes(self):
        assert matmul_mma_config(64, 128, 16) is not None
        assert matmul_mma_config(63, 128, 16) is None


class TestKernelFamilySelection:
    def test_elementwise_ops(self):
        assert select_kernel_family("relu", [512, 512]) == "elementwise"
        assert select_kernel_family("gelu", [1024, 1024]) == "elementwise"

    def test_rowwise_ops(self):
        assert select_kernel_family("softmax", [512, 4096]) == "rowwise"
        assert select_kernel_family("layernorm", [256, 1024]) == "rowwise"

    def test_matmul(self):
        assert select_kernel_family("matmul", [1024, 1024]) == "matmul_mma"

    def test_attention(self):
        assert select_kernel_family("flash_attention", [1, 8, 64, 128]) == "attention"


class TestGPUProfile:
    def test_default_profile(self):
        assert DEFAULT_GPU.chip == "sm_86"
        assert DEFAULT_GPU.num_sm == 30
        assert DEFAULT_GPU.warp_size == 32
        assert DEFAULT_GPU.max_warps_per_block == 32  # 1024/32
