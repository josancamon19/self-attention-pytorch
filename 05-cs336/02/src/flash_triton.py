import math
from sympy.assumptions.ask_generated import Q
import torch

import triton
import triton.language as tl

from src.flash_torch import dummy_attention  # , FlashForward as FlashPytorch
import os
import pdb

# os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
torch.manual_seed(42)

def get_cuda_autotune_config():
    def config_item(q, k, ns, nw, mr= None):
        config = triton.Config({"BLOCK_SIZE_Q": q, "BLOCK_SIZE_K": k}, num_stages=ns, num_warps=nw)
        if mr is not None:
            config.maxnreg = mr
        return config

    return [
        config_item(128, 256, 3, 8),
        config_item(64, 256, 4, 4),
        config_item(64, 64, 3, 4), # best
        config_item(128, 128, 4, 4),
        config_item(128, 64, 4, 4),
        config_item(64, 128, 4, 4),
        config_item(128, 32, 4, 4),
        config_item(64, 32, 5, 2),
        config_item(32, 64, 5, 2),
        config_item(128, 64, 5, 8),
        config_item(256, 64, 5, 8),
    ]


# Triton autotuning for function flash finished after 740.29s;
# best config selected: BLOCK_SIZE_Q: 64, BLOCK_SIZE_K: 64, num_warps: 4, num_ctas: 1, num_stages: 4,
# num_buffers_warp_spec: 0, num_consumer_groups: 0, reg_dec_producer: 0, reg_inc_consumer: 0, maxnreg: None;
@triton.autotune(configs=get_cuda_autotune_config(), key=["q", "k", "v"])
@triton.jit
def flash(
    q_ptr,
    k_ptr,
    v_ptr,
    l_ptr,
    o_ptr,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    q_block_id = tl.program_id(0)  # q_tile
    bh = tl.program_id(1)  # batch * head
    
    # Calculate base offset for this batch/head
    base_offset = bh * seq_length * head_dim
    
    # Create block pointer for Q
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),  # row-major layout
        offsets=(q_block_id * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0)  # row-major
    )
    
    # Load Q tile using block pointer
    q_tile = tl.load(q_block_ptr, boundary_check=(0, 1))
    
    # Initialize accumulators
    mi = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    li = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    oi = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)
    
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # Loop over K/V blocks
    for k_block_id in range(0, tl.cdiv(seq_length, BLOCK_SIZE_K)):
        # Create block pointers for K and V
        k_block_ptr = tl.make_block_ptr(
            base=k_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(k_block_id * BLOCK_SIZE_K, 0),
            block_shape=(BLOCK_SIZE_K, head_dim),
            order=(1, 0)
        )
        
        v_block_ptr = tl.make_block_ptr(
            base=v_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(k_block_id * BLOCK_SIZE_K, 0),
            block_shape=(BLOCK_SIZE_K, head_dim),
            order=(1, 0)
        )
        
        # Load K and V tiles
        k_tile = tl.load(k_block_ptr, boundary_check=(0, 1))
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))
        
        # Compute attention scores
        attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
        
        # Apply causal mask if needed
        if is_causal:
            q_indices = q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
            k_indices = k_block_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
            causal_mask = q_indices >= k_indices
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))
        
        # Flash attention online softmax update
        rowmax = tl.max(attn_scores, axis=1)
        
        prev_mi = mi
        mi = tl.maximum(mi, rowmax)
        pj = tl.exp(attn_scores - mi[:, None])
        # pj_sum = tl.sum(pj, axis=-1).to(tl.float32)
        
        rescale_factor = tl.exp(prev_mi - mi)
        li = rescale_factor * li + tl.sum(pj, axis=-1)
        oi = rescale_factor[:, None] * oi + tl.dot(pj.to(v_tile.dtype), v_tile)
    
    # Final normalization
    oi = oi / li[:, None]
    
    # Store output using block pointer
    o_block_ptr = tl.make_block_ptr(
        base=o_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(q_block_id * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0)
    )
    tl.store(o_block_ptr, oi.to(q_tile.dtype), boundary_check=(0, 1))
    
    # Store L (log sum exp)
    l_offset = bh * seq_length + q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    l_mask = (q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < seq_length
    li = mi + tl.log(li)
    tl.store(l_ptr + l_offset, li, mask=l_mask)

@triton.autotune(configs=get_cuda_autotune_config(), key=["q", "k", "v"])
@triton.jit
def flash_backward(
    grad_out_ptr,
    D_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    grad_q_ptr,
    grad_k_ptr,
    grad_v_ptr,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    k_block_id = tl.program_id(0)  # k_tile    
    bh = tl.program_id(1)  # batch * head
    
    # Base offset for this batch/head
    base_offset = bh * seq_length * head_dim
    
    # Create block pointers for K and V (fixed for this k_block)
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(k_block_id * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, head_dim),
        order=(1, 0)
    )
    
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(k_block_id * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, head_dim),
        order=(1, 0)
    )
    
    # Load K and V tiles (fixed for this kernel instance)
    k_tile = tl.load(k_block_ptr, boundary_check=(0, 1))
    v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))
    
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # Accumulation tiles in float32
    grad_v = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    grad_k = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    
    # Loop over Q blocks
    for q_block_id in range(0, tl.cdiv(seq_length, BLOCK_SIZE_Q)):
        # Create block pointers for Q-related tensors
        q_block_ptr = tl.make_block_ptr(
            base=q_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(q_block_id * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, head_dim),
            order=(1, 0)
        )
        
        o_block_ptr = tl.make_block_ptr(
            base=o_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(q_block_id * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, head_dim),
            order=(1, 0)
        )
        
        grad_out_block_ptr = tl.make_block_ptr(
            base=grad_out_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(q_block_id * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, head_dim),
            order=(1, 0)
        )
        
        grad_q_block_ptr = tl.make_block_ptr(
            base=grad_q_ptr + base_offset,
            shape=(seq_length, head_dim),
            strides=(head_dim, 1),
            offsets=(q_block_id * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, head_dim),
            order=(1, 0)
        )
        
        # Load Q-related tiles
        q_tile = tl.load(q_block_ptr, boundary_check=(0, 1))
        o_tile = tl.load(o_block_ptr, boundary_check=(0, 1))
        grad_out_tile = tl.load(grad_out_block_ptr, boundary_check=(0, 1))
        
        one_dim_offset = bh * seq_length + q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
        one_dim_mask = (q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < seq_length
        l_tile = tl.load(l_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]  # Shape: (BLOCK_SIZE_Q, 1)
        D_tile = tl.load(D_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]  # Shape: (BLOCK_SIZE_Q, 1)
        
        # Compute attention scores
        s = tl.dot(q_tile, tl.trans(k_tile)) * scale
        
        if is_causal:
            q_indices = q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
            k_indices = k_block_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
            causal_mask = q_indices >= k_indices
            s = tl.where(causal_mask, s, float("-inf"))
        
        p = tl.exp(s - l_tile)
        
        # Compute gradients
        grad_v += tl.dot(tl.trans(p.to(grad_out_tile.dtype)), grad_out_tile).to(tl.float32)
        grad_p = tl.dot(grad_out_tile, tl.trans(v_tile))
        grad_s = p * (grad_p - D_tile) * scale
        
        # Update Q gradients using atomic add
        grad_q_update = tl.dot(grad_s, k_tile.to(tl.float32))
        
        # For atomic operations, we need to use manual pointers, not block pointers
        q_offset = q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
        d_offset = tl.arange(0, head_dim)[None, :]
        grad_q_ptrs = grad_q_ptr + base_offset + q_offset * head_dim + d_offset
        grad_q_mask = (q_offset < seq_length) & (d_offset < head_dim)
        
        tl.atomic_add(grad_q_ptrs, grad_q_update.to(grad_out_tile.dtype), mask=grad_q_mask)
        
        # Accumulate K gradients
        grad_k += tl.dot(tl.trans(grad_s.to(q_tile.dtype)), q_tile).to(tl.float32)
    
    # Store final K and V gradients using block pointers
    grad_k_block_ptr = tl.make_block_ptr(
        base=grad_k_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(k_block_id * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, head_dim),
        order=(1, 0)
    )
    
    grad_v_block_ptr = tl.make_block_ptr(
        base=grad_v_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(k_block_id * BLOCK_SIZE_K, 0),
        block_shape=(BLOCK_SIZE_K, head_dim),
        order=(1, 0)
    )
    
    tl.store(grad_k_block_ptr, grad_k.to(k_tile.dtype), boundary_check=(0, 1))
    tl.store(grad_v_block_ptr, grad_v.to(v_tile.dtype), boundary_check=(0, 1))


class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        debug: bool = False,
    ):
        has_batch_size = len(q.shape) == 4
        if has_batch_size:
            batch_size, num_heads, seq_length, d = q.shape
        else:
            num_heads, seq_length, d = q.shape
            batch_size = 1

        batch_heads = batch_size * num_heads

        l = torch.empty(
            (batch_heads, seq_length),
            device=q.device,
            dtype=q.dtype,
        )
        o = torch.empty(
            (batch_heads, seq_length, d),
            device=q.device,
            dtype=q.dtype,
        )

        reshape = lambda m: m.reshape((batch_heads, seq_length, d))  # noqa: E731

        q_tile_size, k_tile_size, num_warps, num_stages = 64, 64, 4, 3
        # q_tile_size, k_tile_size, num_warps, num_stages = 64, 64, 4, 4
        # grid = (triton.cdiv(seq_length, q_tile_size), batch_heads)

        grid = lambda meta: (triton.cdiv(seq_length, meta["BLOCK_SIZE_Q"]), batch_heads)  # noqa: E731
        if debug:
            print("grid:", grid)
            print(f"q shape: {q.shape}")
            print(f"k shape: {k.shape}")
            print(f"v shape: {v.shape}")
            print(f"l shape: {l.shape}")
            print(f"o shape: {o.shape}")

        flash[grid](
            reshape(q),
            reshape(k),
            reshape(v),
            l,
            o,
            seq_length=seq_length,
            head_dim=d,
            is_causal=is_causal,
            # BLOCK_SIZE_Q=q_tile_size,
            # BLOCK_SIZE_K=k_tile_size,
            # num_warps=num_warps,
            # num_stages=num_stages,
            # num_ctas=num_ctas, # not include, worst results.
        )
        if not has_batch_size:
            o = o.squeeze(0)
            l = l.squeeze(0)
            o = o.reshape(num_heads, seq_length, d)
        else:
            o = o.reshape(batch_size, num_heads, seq_length, d)
            l = l.reshape(batch_size, num_heads, seq_length)

        if debug:
            print(f"output shape: {o.shape}")
        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        # pdb.set_trace()
        return o

    # @staticmethod
    # def backward(ctx, grad_out):
    #     q, k, v, o, l = ctx.saved_tensors
    #     D = torch.sum(o * grad_out, dim=-1)
    #     scale = 1.0 / math.sqrt(k.shape[-1])

    #     s = q @ k.transpose(-2, -1) * scale

    #     if ctx.is_causal:
    #         causal_mask = torch.tril(torch.ones_like(s, dtype=torch.bool))
    #         s = s.masked_fill(~causal_mask, float("-inf"))

    #     p = torch.exp(s - l.unsqueeze(-1))
    #     grad_v = p.transpose(-2, -1) @ grad_out
    #     grad_p = grad_out @ v.transpose(-2, -1)
    #     grad_s = p * (grad_p - D.unsqueeze(-1))
    #     grad_q = grad_s @ k * scale
    #     grad_k = grad_s.transpose(-2, -1) @ q * scale

    #     return grad_q, grad_k, grad_v, None, None, None, None

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, l = ctx.saved_tensors
        D = torch.sum(o * grad_out, dim=-1)
        grad_out = grad_out.contiguous()
        
        has_batch_size = len(q.shape) == 4
        if has_batch_size:
            batch_size, num_heads, seq_length, d = q.shape
        else:
            num_heads, seq_length, d = q.shape
            batch_size = 1

        batch_heads = batch_size * num_heads

        shape = (batch_heads, seq_length, d)
        grad_q = torch.zeros(shape, device=q.device, dtype=torch.float32)  # atomic ops
        grad_k = torch.empty(shape, device=k.device, dtype=q.dtype)  # fp32 for accumulation
        grad_v = torch.empty(shape, device=v.device, dtype=q.dtype)  # fp32 for accumulation

        # BLOCK_SIZE_Q, BLOCK_SIZE_K = 64, 64
        # grid = (triton.cdiv(seq_length, BLOCK_SIZE_K), batch_heads)
        grid = lambda meta: (triton.cdiv(seq_length, meta['BLOCK_SIZE_K']), batch_heads) # noqa
        reshape = lambda m: m.reshape((batch_heads, seq_length, d)) # noqa

        # pdb.set_trace()
        flash_backward[grid](
            reshape(grad_out),
            D.reshape(batch_heads, seq_length),
            reshape(q),
            reshape(k),
            reshape(v),
            reshape(o),
            l.reshape(batch_heads, seq_length),
            grad_q,
            grad_k,
            grad_v,
            seq_length,
            d,
            ctx.is_causal,
            # BLOCK_SIZE_Q,
            # BLOCK_SIZE_K,
        )
        
        # Reshape gradients to match input shapes
        if has_batch_size:
            grad_q = grad_q.reshape(batch_size, num_heads, seq_length, d)
            grad_k = grad_k.reshape(batch_size, num_heads, seq_length, d)
            grad_v = grad_v.reshape(batch_size, num_heads, seq_length, d)
        
        return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), None, None, None, None


def get_q_k_v(n_heads = 16, seq_length = 16384, head_dim=64, dtype=torch.float32):
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
    flash = torch.compile(FlashAttention.apply)
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


def verify_correctness(dtype = torch.float32):
    test_configs = [
        # (n_heads, seq_length, d_head, is_causal, desc)
        (4, 128, 64, False, "small non-causal"),
        (4, 128, 64, True, "small causal"),
        (8, 512, 64, False, "medium non-causal"),
        (8, 512, 64, True, "medium causal"),
        (12, 1024, 64, False, "large non-causal"),
        (12, 1024, 64, True, "large causal"),
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
        
        output_flash = FlashAttention.apply(q_flash, k_flash, v_flash, is_causal)
        output_dummy = dummy_attention(q_dummy, k_dummy, v_dummy, is_causal)
        
        # Check forward pass
        max_diff = torch.max(torch.abs(output_flash - output_dummy)).item()
        rtol = 5e-3  # Relative tolerance
        atol = 1e-3  # Absolute tolerance
        rtol_backward = 1e-2  # More lenient for gradients
        atol_backward = 5e-3  # More lenient for gradients
        forward_match = torch.allclose(output_flash, output_dummy, rtol=rtol, atol=atol)
        grad_out = torch.randn_like(output_flash)
        
        loss_flash = (output_flash * grad_out).sum()
        loss_dummy = (output_dummy * grad_out).sum()
        
        loss_flash.backward()
        loss_dummy.backward()
        
        # Check gradients
        q_grad_diff = torch.max(torch.abs(q_flash.grad - q_dummy.grad)).item()
        k_grad_diff = torch.max(torch.abs(k_flash.grad - k_dummy.grad)).item()
        v_grad_diff = torch.max(torch.abs(v_flash.grad - v_dummy.grad)).item()
        
        q_grad_match = torch.allclose(q_flash.grad, q_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        k_grad_match = torch.allclose(k_flash.grad, k_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        v_grad_match = torch.allclose(v_flash.grad, v_dummy.grad, rtol=rtol_backward, atol=atol_backward)
        print(f"Test case: {desc}")
        print(f"  Shape: ({n_heads}, {seq_length}, {d_head})")
        print(f"  Forward pass: {'✓ PASSED' if forward_match else '✗ FAILED'} (max diff: {max_diff:.2e})")
        print(f"  Backward pass:")
        print(f"    Q grad: {'✓ PASSED' if q_grad_match else '✗ FAILED'} (max diff: {q_grad_diff:.2e})")
        print(f"    K grad: {'✓ PASSED' if k_grad_match else '✗ FAILED'} (max diff: {k_grad_diff:.2e})")
        print(f"    V grad: {'✓ PASSED' if v_grad_match else '✗ FAILED'} (max diff: {v_grad_diff:.2e})")
        print()

if __name__ == "__main__":
    # flashattn = FlashAttention.apply
    # q, k, v = get_q_k_v(dtype=torch.bfloat16)
    # output = flashattn(q, k, v, True)
    # loss = output.sum()
    # loss.backward()
    # verify_correctness(dtype=torch.bfloat16)
    flash_benchmarking()
    # fix why is so off on bfloat16 correctness, and closer to float32 ✅
    # TODO: what's happening to so many dtype converdsions?
    # TODO: verify stupid improvements with atomic stuff
    # TODO: base2 ops instead of exp
    # TODO: don't do atomic ops
    # TODO: navigate through backward algo, and write it on your own words
