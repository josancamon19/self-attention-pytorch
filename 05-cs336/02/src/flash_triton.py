import torch

import triton
import triton.language as tl

from src.flash_torch import dummy_attention
# import os
# import pdb

# os.environ["TRITON_INTERPRET"] = "1"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@triton.jit
def flash(
    q_ptr,
    k_ptr,
    v_ptr,
    l_ptr,
    o_ptr,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
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
        kvm_offset = j * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[:, None]
        kvn_offset = tl.arange(0, head_dim)[None, :]

        k_ptrs = k_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset
        v_ptrs = v_ptr + bh * seq_length * head_dim + kvm_offset * head_dim + kvn_offset

        kv_mask = (kvm_offset < seq_length) & (kvn_offset < head_dim)
        k_tile = tl.load(k_ptrs, mask=kv_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=kv_mask, other=0.0)

        # tl.device_print("k_tile", k_tile)  # shape?
        # tl.device_print("v_tile", v_tile)

        attn_scores = tl.dot(q_tile.to(tl.float16), tl.trans(k_tile).to(tl.float16)) * scale
        # tl.device_print("attn_scores", attn_scores)

        rowmax = tl.max(attn_scores, axis=1)
        prev_mi = mi
        mi = tl.maximum(mi, rowmax)
        pj = tl.exp(attn_scores - mi[:, None])
        li = tl.exp(prev_mi - mi) * li + tl.sum(pj, axis=-1)
        oi = tl.exp(prev_mi - mi)[:, None] * oi + tl.dot(pj.to(tl.float16), v_tile.to(tl.float16))

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


class FlashForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
        q_tile_size: int = 16,
        k_tile_size: int = 16,
    ):
        batch_size, num_heads, seq_length, d = q.shape
        batch_heads = batch_size * num_heads

        reshape = lambda m: m.reshape((batch_heads, seq_length, d))  # noqa: E731
        q, k, v = reshape(q), reshape(k), reshape(v)
        # print(q[0, :q_tile_size, :])

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

        grid = (triton.cdiv(seq_length, q_tile_size), batch_heads)
        print("grid:", grid)
        print(f"q shape: {q.shape}")
        print(f"k shape: {k.shape}")
        print(f"v shape: {v.shape}")
        print(f"l shape: {l.shape}")
        print(f"o shape: {o.shape}")
        flash[grid](
            q,
            k,
            v,
            l,
            o,
            seq_length=seq_length,
            head_dim=d,
            BLOCK_SIZE_Q=q_tile_size,
            BLOCK_SIZE_K=k_tile_size,
        )
        ctx.save_for_backward(l)
        output = o.reshape(batch_size, num_heads, seq_length, d)
        print(f"output shape: {output.shape}")
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


def get_q_k_v():
    embedding_dim = 128
    head_size = 32
    num_heads = embedding_dim // head_size
    batch_size = 2
    seq_length = 128
    dims = (batch_size, num_heads, seq_length, head_size)
    print("dims:", dims)
    get = lambda: torch.rand(dims, device=device, dtype=torch.float16, requires_grad=True)  # noqa: E731
    return get(), get(), get()


if __name__ == "__main__":
    flashattn = FlashForward.apply
    q, k, v = get_q_k_v()
    output = flashattn(q, k, v, False, 16, 16)
    dummy = dummy_attention(q, k, v)
    print("output", output)
    print("output", dummy)
    print("torch.allclose(output, dummy)", torch.allclose(output, dummy, rtol=0.0001))
