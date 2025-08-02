import math
from sympy.assumptions.ask_generated import Q
import torch

import triton
import triton.language as tl

from src.flash_torch import dummy_attention  # , FlashForward as FlashPytorch
import os
# import pdb

# os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cuda_autotune_config():
    def config_item(q, k, ns, nw, mr= None):
        config = triton.Config({"BLOCK_SIZE_Q": q, "BLOCK_SIZE_K": k}, num_stages=ns, num_warps=nw)
        if mr is not None:
            config.maxnreg = mr
        return config

    # configs = []
    # for q in [64, 128, 256, 512]:
    #     for k in [64, 128, 256, 512]:
    #         for ns in [3, 4, 5, 6, 7]:  # , 4, 5, 6, 7
    #             for nw in [4, 8]:  # must be power of 2 # 16, 32? caused register allocation failed
    #                 # for mr in [None, 128, 160, 192]:  # maxnreg
    #                 configs.append(config_item(q, k, ns, nw, None))
    # print(len(configs))  # 192 configs, took 6.5 minutes
    # return configs

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
# @triton.autotune(configs=get_cuda_autotune_config(), key=["q", "k", "v"])
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
    q = tl.program_id(0)  # q_tile
    bh = tl.program_id(1)  # batch * head

    q_start = q * BLOCK_SIZE_Q
    qm_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    qn_offset = tl.arange(0, head_dim)[None, :]

    q_ptrs = q_ptr + bh * seq_length * head_dim + qm_offset * head_dim + qn_offset
    q_mask = (qm_offset < seq_length) & (qn_offset < head_dim)

    q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

    mi = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    li = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    oi = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(float(head_dim))

    for j in range(0, tl.cdiv(seq_length, BLOCK_SIZE_K)):
        k_start = j * BLOCK_SIZE_K
        # this would affect once you launch more than # of SMs so they can move on to other processes
        kvm_offset = k_start + tl.arange(0, BLOCK_SIZE_K)[:, None]
        kvn_offset = tl.arange(0, head_dim)[None, :]

        k_ptrs = k_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset
        v_ptrs = v_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset

        kv_mask = (kvm_offset < seq_length) & (kvn_offset < head_dim)
        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # tl.device_print("k_tile", k_tile)  # shape?
        # tl.device_print("v_tile", v_tile)

        # attn_scores = tl.dot(q_tile.to(tl.float16), tl.trans(k_tile).to(tl.float16)) * scale
        attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

        if is_causal:
            q_indices = qm_offset  # [BLOCK_SIZE_Q, 1]
            k_indices = k_start + tl.arange(0, BLOCK_SIZE_K)[None, :]  # [1, BLOCK_SIZE_K]
            causal_mask = q_indices >= k_indices  # True where attention is allowed
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))
        # tl.device_print("attn_scores", attn_scores)

        rowmax = tl.max(attn_scores, axis=1)
        prev_mi = mi
        mi = tl.maximum(mi, rowmax)
        pj = tl.exp(attn_scores - mi[:, None])
        li = tl.exp(prev_mi - mi) * li + tl.sum(pj, axis=-1)
        # oi = tl.exp(prev_mi - mi)[:, None] * oi + tl.dot(pj.to(tl.float16), v_tile.to(tl.float16))
        # oi = tl.exp(prev_mi - mi)[:, None] * oi + tl.dot(pj, v_tile.to(tl.float32))
        oi = tl.exp(prev_mi - mi)[:, None] * oi + tl.dot(pj.to(v_tile.dtype), v_tile)

    # tl.device_print("oi_before_div", oi)
    oi = oi / li[:, None]

    li = mi + tl.log(li)

    om_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)[:, None]
    on_offset = tl.arange(0, head_dim)[None, :]
    o_ptrs = o_ptr + bh * seq_length * head_dim + om_offset * head_dim + on_offset
    o_mask = (om_offset < seq_length) & (on_offset < head_dim)
    tl.store(o_ptrs, oi, mask=o_mask)

    l_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)
    l_ptrs = l_ptr + bh * seq_length + l_offset
    l_mask = l_offset < seq_length
    tl.store(l_ptrs, li, mask=l_mask)

# @triton.autotune(configs=get_cuda_autotune_config(), key=["q", "k", "v"])
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
    k = tl.program_id(0)  # k_tile    
    bh = tl.program_id(1)  # batch * head

    # this would affect once you launch more than # of SMs so they can move on to other processes
    k_start = k * BLOCK_SIZE_K
    kvm_offset = k_start + tl.arange(0, BLOCK_SIZE_K)[:, None]
    kvn_offset = tl.arange(0, head_dim)[None, :]

    k_ptrs = k_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset
    v_ptrs = v_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset

    kv_mask = (kvm_offset < seq_length) & (kvn_offset < head_dim)
    k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
    v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)
    
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # tl.device_print("k_start:", k_start)
    # tl.device_print("scale:", scale)
    # tl.device_print("k_tile:", k_tile)
    # tl.device_print("v_tile:", v_tile)
    # tl.device_print("kv_mask:", kv_mask)
    
    grad_v = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    grad_k = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    
    for j in range(0, tl.cdiv(seq_length, BLOCK_SIZE_Q)):
        q_start = j * BLOCK_SIZE_Q
        qm_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)[:, None]
        qn_offset = tl.arange(0, head_dim)[None, :]

        q_ptrs = q_ptr + bh * seq_length * head_dim + qm_offset * head_dim + qn_offset
        o_ptrs = o_ptr + bh * seq_length * head_dim + qm_offset * head_dim + qn_offset
        grad_out_ptrs = grad_out_ptr + bh * seq_length * head_dim + qm_offset * head_dim + qn_offset
        grad_q_ptrs = grad_q_ptr + bh * seq_length * head_dim + qm_offset * head_dim + qn_offset

        q_mask = (qm_offset < seq_length) & (qn_offset < head_dim)

        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)
        o_tile = tl.load(o_ptrs, mask=q_mask, other=0.0)
        grad_out_tile = tl.load(grad_out_ptrs, mask=q_mask, other=0.0)
        
        # Load L and D tiles - they are 1D tensors
        l_offset = q_start + tl.arange(0, BLOCK_SIZE_Q)
        l_mask = l_offset < seq_length
        
        l_ptrs = l_ptr + bh * seq_length + l_offset
        D_ptrs = D_ptr + bh * seq_length + l_offset
        
        l_tile = tl.load(l_ptrs, mask=l_mask, other=0.0)[:, None]  # Shape: (BLOCK_SIZE_Q, 1)
        D_tile = tl.load(D_ptrs, mask=l_mask, other=0.0)[:, None]  # Shape: (BLOCK_SIZE_Q, 1)

        s = tl.dot(q_tile, tl.trans(k_tile)) * scale
        
        if is_causal:
            q_indices = qm_offset  # [BLOCK_SIZE_Q, 1]
            k_indices = k_start + tl.arange(0, BLOCK_SIZE_K)[None, :]  # [1, BLOCK_SIZE_K]
            causal_mask = q_indices >= k_indices
            s = tl.where(causal_mask, s, float("-inf"))
        
        p = tl.exp(s - l_tile)
        
        grad_v += tl.dot(tl.trans(p.to(grad_out_tile.dtype)), grad_out_tile)
        grad_p = tl.dot(grad_out_tile, tl.trans(v_tile))
        grad_s = p * (grad_p - D_tile) * scale
        
        grad_q_update = tl.dot(grad_s, k_tile.to(tl.float32)).to(tl.float32) 
        tl.atomic_add(grad_q_ptrs, grad_q_update, mask=q_mask)
        
        grad_k += tl.dot(tl.trans(grad_s.to(q_tile.dtype)), q_tile) 
    
    grad_k_ptrs = grad_k_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset
    grad_v_ptrs = grad_v_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset

    tl.store(grad_k_ptrs, grad_k, mask=kv_mask)
    tl.store(grad_v_ptrs, grad_v, mask=kv_mask)


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
        grid = (triton.cdiv(seq_length, q_tile_size), batch_heads)

        # grid = lambda meta: (triton.cdiv(seq_length, meta["BLOCK_SIZE_Q"]), batch_heads)  # noqa: E731
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
            BLOCK_SIZE_Q=q_tile_size,
            BLOCK_SIZE_K=k_tile_size,
            num_warps=num_warps,
            num_stages=num_stages,
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
        grad_k = torch.empty(shape, device=k.device, dtype=torch.float32)  # fp32 for accumulation
        grad_v = torch.empty(shape, device=v.device, dtype=torch.float32)  # fp32 for accumulation

        BLOCK_SIZE_Q, BLOCK_SIZE_K = 64, 64
        grid = (triton.cdiv(seq_length, BLOCK_SIZE_K), batch_heads)
        # grid = lambda meta: (triton.cdiv(seq_length, meta['BLOCK_SIZE_K']), batch_heads) # noqa
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
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
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
    results = triton.testing.do_bench(benchmark(dummy_attention), rep=10000, warmup=100)
    print("dummy:", results)
    results = triton.testing.do_bench(benchmark(dummy_compiled_fn), rep=10000, warmup=100)
    print("dummy_compiled:", results)


def verify_correctness(is_causal: bool = True):
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
        q, k, v = get_q_k_v(n_heads, seq_length, d_head, torch.float32)
        
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
    # verify_correctness()
    flash_benchmarking()
    # TODO: fix why is so off on bfloat16 correctness, and closer to float32
    # TODO: what's happening to so many dtype converdsions?
    # TODO: verify stupid improvements with atomic stuff
    # TODO: base2 ops instead of exp
    # TODO: don't do atomic ops
    # TODO: navigate through backward algo, and write it on your own words
