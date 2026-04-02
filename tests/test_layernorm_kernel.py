"""Tests for Arke LayerNorm/RMSNorm Triton kernels."""
import pytest
import torch

from arke.integration.kernel_cache import KernelCache


@pytest.fixture
def cache():
    return KernelCache()


class TestLayerNormKernel:
    @pytest.mark.parametrize("shape", [(128, 768), (2048, 4096), (127, 769)])
    def test_layernorm_correctness(self, cache, shape):
        M, N = shape
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.ones(N, device="cuda", dtype=torch.float16)
        bias = torch.zeros(N, device="cuda", dtype=torch.float16)

        arke_out = cache.layernorm(x, weight, bias)
        ref = torch.nn.functional.layer_norm(x.float(), [N], weight.float(), bias.float()).half()

        torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)

    @pytest.mark.parametrize("shape", [(128, 768), (2048, 4096), (127, 769)])
    def test_rmsnorm_correctness(self, cache, shape):
        M, N = shape
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.ones(N, device="cuda", dtype=torch.float16)

        arke_out = cache.rmsnorm(x, weight)

        # Manual RMSNorm reference
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32 ** 2, dim=-1, keepdim=True) + 1e-5)
        ref = (x_f32 / rms * weight.float()).half()

        torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)

    def test_layernorm_with_bias(self, cache):
        M, N = 128, 768
        x = torch.randn(M, N, device="cuda", dtype=torch.float16)
        weight = torch.randn(N, device="cuda", dtype=torch.float16)
        bias = torch.randn(N, device="cuda", dtype=torch.float16)

        arke_out = cache.layernorm(x, weight, bias)
        ref = torch.nn.functional.layer_norm(x.float(), [N], weight.float(), bias.float()).half()

        torch.testing.assert_close(arke_out, ref, atol=0.1, rtol=0.05)
