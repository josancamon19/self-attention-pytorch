import math
import torch

import triton
import triton.language as tl

from src.flash_torch import dummy_attention
import os

# os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cuda_autotune_config():
    def config_item(q, k, ns, nw):
        return triton.Config({"BLOCK_SIZE_Q": q, "BLOCK_SIZE_K": k}, num_stages=ns, num_warps=nw)

    configs = []
    for q in [32, 64, 128, 256]:
        for k in [32, 64, 128, 256]:
            for ns in [3, 4, 5, 6, 7]:
                for nw in [2, 4, 8, 16]:  # must be multiple of 2
                    # for nw in [1, 2, 4, 6, 8, 12, 16, 20, 24, 32]:
                    configs.append(config_item(q, k, ns, nw))
    print(len(configs))  # 192 configs, took 6.5 minutes
    return configs

    # return [
    #     config_item(128, 256, 3, 8),
    #     config_item(64, 256, 4, 4),
    #     config_item(64, 64, 3, 4), # best
    #     config_item(128, 128, 4, 4),
    #     config_item(128, 64, 4, 4),
    #     config_item(64, 128, 4, 4),
    #     config_item(128, 32, 4, 4),
    #     config_item(64, 32, 5, 2),
    #     config_item(32, 64, 5, 2),
    #     config_item(128, 64, 5, 8),
    #     config_item(256, 64, 5, 8),
    # ]


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
    if is_causal and q_start >= seq_length:
        return
    
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
        kvm_offset = j * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
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
            k_indices = j * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]  # [1, BLOCK_SIZE_K]
            causal_mask = q_indices >= k_indices  # True where attention is allowed
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))
        # tl.device_print("attn_scores", attn_scores)

        rowmax = tl.max(attn_scores, axis=1)
        prev_mi = mi
        mi = tl.maximum(mi, rowmax)
        pj = tl.exp(attn_scores - mi[:, None])
        li = tl.exp(prev_mi - mi) * li + tl.sum(pj, axis=-1)
        # oi = tl.exp(prev_mi - mi)[:, None] * oi + tl.dot(pj.to(tl.float16), v_tile.to(tl.float16))
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

        l = torch.zeros(
            (batch_heads, seq_length),
            device=q.device,
            dtype=q.dtype,
        )
        o = torch.zeros(
            (batch_heads, seq_length, d),
            device=q.device,
            dtype=q.dtype,
        )

        reshape = lambda m: m.reshape((batch_heads, seq_length, d))  # noqa: E731

        q_tile_size, k_tile_size, num_warps, num_stages = 64, 64, 4, 3
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

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, l = ctx.saved_tensors
        D = torch.sum(o * grad_out, dim=-1)
        scale = 1.0 / math.sqrt(k.shape[-1])

        s = q @ k.transpose(-2, -1) * scale

        if ctx.is_causal:
            causal_mask = torch.tril(torch.ones_like(s, dtype=torch.bool))
            s = s.masked_fill(~causal_mask, float("-inf"))

        p = torch.exp(s - l.unsqueeze(-1))
        grad_v = p.transpose(-2, -1) @ grad_out
        grad_p = grad_out @ v.transpose(-2, -1)
        grad_s = p * (grad_p - D.unsqueeze(-1))
        grad_q = grad_s @ k * scale
        grad_k = grad_s.transpose(-2, -1) @ q * scale

        return grad_q, grad_k, grad_v, None, None, None, None


def get_q_k_v():
    n_heads = 16
    d_head = 64
    seq_length = 16384
    q, k, v = torch.randn(
        3,
        n_heads,
        seq_length,
        d_head,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    return q, k, v


def flash_benchmarking():
    q, k, v = get_q_k_v()
    flash = torch.compile(FlashAttention.apply)

    def flash_forward_backward():
        o = flash(q, k, v, True)
        loss = o.sum()
        loss.backward()

    def dummy_forward_backward():
        o = dummy_attention(q, k, v, True)
        loss = o.sum()
        loss.backward()

    results = triton.testing.do_bench(flash_forward_backward, rep=10000, warmup=1000)
    print(results)


if __name__ == "__main__":
    # flashattn = FlashAttention.apply
    # q, k, v = get_q_k_v()
    # output = flashattn(q, k, v, True)
    # loss = output.sum()
    # loss.backward()

    flash_benchmarking()
