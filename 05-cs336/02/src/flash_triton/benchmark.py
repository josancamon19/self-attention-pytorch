import math
import torch

import triton
import triton.language as tl

from src.flash_torch import dummy_attention  # , FlashForward as FlashPytorch
from src.flash_triton.flash import FlashAttention
import os
import pdb

# os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
torch.manual_seed(42)


def get_q_k_v(n_heads=16, seq_length=16384, head_dim=64, dtype=torch.float32):
    q, k, v = torch.randn(
        3,
        n_heads,
        seq_length,
        head_dim,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    return q, k, v


def flash_benchmarking():
    q, k, v = get_q_k_v(dtype=torch.bfloat16)
    flash = FlashAttention.apply
    # flash = torch.compile(FlashAttention.apply)  # Causing issues with TMA implementation :/
    # flash_torch = torch.compile(FlashPytorch.apply)
    # flash_torch = FlashPytorch.apply
    dummy_compiled_fn = torch.compile(dummy_attention)

    def benchmark(fn):
        def wrap():
            o = fn(q, k, v, True)
            loss = o.sum()
            loss.backward()

        return wrap

    results = triton.testing.do_bench(benchmark(flash), rep=10000, warmup=1000)
    print("flash_triton:", results)
    # results = triton.testing.do_bench(benchmark(dummy_attention), rep=10000, warmup=100)
    # print("dummy:", results)
    # results = triton.testing.do_bench(benchmark(dummy_compiled_fn), rep=10000, warmup=100)
    # print("dummy_compiled:", results)


def verify_correctness(dtype=torch.float32):
    test_configs = [
        # (n_heads, seq_length, d_head, is_causal, desc)
        # (4, 128, 64, False, "small non-causal"),
        # (4, 128, 64, True, "small causal"),
        # (8, 512, 64, False, "medium non-causal"),
        # (8, 512, 64, True, "medium causal"),
        # (12, 1024, 64, False, "large non-causal"),
        # (12, 1024, 64, True, "large causal"),
        (16, 16384, 64, False, "huge causal"),
    ]
    for n_heads, seq_length, d_head, is_causal, desc in test_configs:
        q, k, v = get_q_k_v(n_heads, seq_length, d_head, dtype)

        # Clone tensors for both implementations
        q_flash = q.clone().detach().requires_grad_(True)
        k_flash = k.clone().detach().requires_grad_(True)
        v_flash = v.clone().detach().requires_grad_(True)

        q_dummy = q.clone().detach().requires_grad_(True)
        k_dummy = k.clone().detach().requires_grad_(True)
        v_dummy = v.clone().detach().requires_grad_(True)

        output_flash = FlashAttention.apply(
            q_flash, k_flash, v_flash, is_causal)
        output_dummy = dummy_attention(q_dummy, k_dummy, v_dummy, is_causal)

        # Check forward pass
        max_diff = torch.max(torch.abs(output_flash - output_dummy)).item()
        rtol = 5e-3  # Relative tolerance
        atol = 1e-3  # Absolute tolerance
        rtol_backward = 1e-2  # More lenient for gradients
        atol_backward = 5e-3  # More lenient for gradients
        forward_match = torch.allclose(
            output_flash, output_dummy, rtol=rtol, atol=atol)
        grad_out = torch.randn_like(output_flash)

        loss_flash = (output_flash * grad_out).sum()
        loss_dummy = (output_dummy * grad_out).sum()

        loss_flash.backward()
        loss_dummy.backward()

        # Check gradients
        q_grad_diff = torch.max(torch.abs(q_flash.grad - q_dummy.grad)).item()
        k_grad_diff = torch.max(torch.abs(k_flash.grad - k_dummy.grad)).item()
        v_grad_diff = torch.max(torch.abs(v_flash.grad - v_dummy.grad)).item()

        q_grad_match = torch.allclose(
            q_flash.grad, q_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        k_grad_match = torch.allclose(
            k_flash.grad, k_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        v_grad_match = torch.allclose(
            v_flash.grad, v_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        print(f"Test case: {desc}")
        print(f"  Shape: ({n_heads}, {seq_length}, {d_head})")
        print(
            f"  Forward pass: {'✓ PASSED' if forward_match else '✗ FAILED'} (max diff: {max_diff:.2e})")
        print(f"  Backward pass:")
        print(
            f"    Q grad: {'✓ PASSED' if q_grad_match else '✗ FAILED'} (max diff: {q_grad_diff:.2e})")
        print(
            f"    K grad: {'✓ PASSED' if k_grad_match else '✗ FAILED'} (max diff: {k_grad_diff:.2e})")
        print(
            f"    V grad: {'✓ PASSED' if v_grad_match else '✗ FAILED'} (max diff: {v_grad_diff:.2e})")
        print()


if __name__ == "__main__":
    # verify_correctness(dtype=torch.bfloat16)
    flash_benchmarking()
    # results
    # - backward implementation atomic, from 28 to 27
    # - backward 2 passes, from 27 to 20
    # - autotuning backward passes, 20 17
    # - skip all future masked, 15-12-10
    # - TMA from 10.25 to 10.005 (painful and almost nothing), had to remove torch.compile
    # TODO:
    # - Separate the non-masked tiles from the tile diagonals, computing the first without ever comparing indices, and the second with a single comparison
    # - Use Persistent Matmul?
    # - base2 ops instead of exp

    # TODO: navigate through backward algo, and write it on your own words
