"""Tests for Arke elementwise Triton kernels."""
import pytest
import torch

from arke.integration.kernel_cache import KernelCache


@pytest.fixture
def cache():
    return KernelCache()


class TestElementwiseKernels:
    """Test elementwise kernel correctness."""

    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    @pytest.mark.parametrize("shape", [(128, 768), (4096, 4096), (127, 769)])
    def test_correctness(self, cache, activation, shape):
        """Compare Arke elementwise output with PyTorch reference."""
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)

        # Arke output
        if activation == "relu":
            arke_out = cache.relu(x)
            ref = torch.nn.functional.relu(x)
        elif activation == "gelu":
            arke_out = cache.gelu(x)
            ref = torch.nn.functional.gelu(x)
        elif activation == "silu":
            arke_out = cache.silu(x)
            ref = torch.nn.functional.silu(x)

        torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.parametrize("activation", ["relu", "gelu", "silu"])
    def test_non_aligned_shape(self, cache, activation):
        """Test with non-power-of-2 shapes."""
        x = torch.randn(333, 777, device="cuda", dtype=torch.float16)
        if activation == "relu":
            arke_out = cache.relu(x)
            ref = torch.nn.functional.relu(x)
        elif activation == "gelu":
            arke_out = cache.gelu(x)
            ref = torch.nn.functional.gelu(x)
        elif activation == "silu":
            arke_out = cache.silu(x)
            ref = torch.nn.functional.silu(x)
        torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
