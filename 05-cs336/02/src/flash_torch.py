import math
import timeit
import torch

# import os
import triton
import triton.language as tl
# import pdb

# os.environ["TRITON_INTERPRET"] = "1"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== FlashAttention Forward Torch Autograd. ===


# TODO: benchmark this, vs compiled, vs yours flash
# - see each one of them in nsight sys


def dummy_attention(q, k, v, is_causal=False):
    # start = timeit.default_timer()
    scale = 1.0 / math.sqrt(k.shape[-1])
    attn_scores = q @ k.transpose(-2, -1) * scale

    if is_causal:
        mask = torch.tril(torch.ones_like(attn_scores, dtype=torch.bool))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

    attn_weights = torch.softmax(attn_scores, dim=-1)
    output = attn_weights @ v
    # torch.cuda.synchronize()
    # print(f"dummy_attention took {timeit.default_timer() - start} seconds")
    return output


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
        # Is so strange that this works
        # attn_scores should never be saved

        # The key insight: softmax can be computed incrementally if you track the right statistics.
        # mi, running maximum
        # li, running sum of exponentials
        # oi, running weighted sum, tracks sum(exp(scores - max) * values) numerator (rescaled when mi changes)
        # the rescaling ensures all exp are relative to same value
        has_batch_size = q.shape == 4
        if has_batch_size:
            batch_size, num_heads, seq_length, d = q.shape
        else:
            num_heads, seq_length, d = q.shape
            batch_size = 1
        # head_dim = d # reminder

        scale = 1 / math.sqrt(d)

        tiles_count = triton.cdiv(seq_length, q_tile_size)
        tiles_q = torch.chunk(q, tiles_count, dim=-2)  # seq_length split
        tiles_k = torch.chunk(k, tiles_count, dim=-2)
        tiles_v = torch.chunk(v, tiles_count, dim=-2)

        tiles_o = []
        tiles_l = []
        for i, tq in enumerate(tiles_q):
            oi = torch.zeros(
                (batch_size, num_heads, q_tile_size, d),
                device=q.device,
                dtype=q.dtype,
            )
            li = torch.zeros(
                (batch_size, num_heads, q_tile_size),
                device=q.device,
                dtype=q.dtype,
            )
            mi = torch.full(
                (batch_size, num_heads, q_tile_size),
                float("-inf"),
                device=q.device,
                dtype=q.dtype,
            )

            for j, kv in enumerate(zip(tiles_k, tiles_v)):
                tk, tv = kv

                attn_scores = tq @ tk.transpose(-2, -1) * scale
                rowmax_values = torch.max(attn_scores, dim=-1).values

                prev_mi = mi
                mi = torch.maximum(prev_mi, rowmax_values)  # element wise max
                pj = torch.exp(attn_scores - mi.unsqueeze(-1))
                li = torch.exp(prev_mi - mi) * li + torch.sum(pj, dim=-1)  # or -2?

                rescale_factor = torch.exp(prev_mi - mi)
                oi = rescale_factor.unsqueeze(-1) * oi + pj @ tv

            # TODO: understand better `diag` notation, how it translates
            oi = oi / li.unsqueeze(-1)
            li = mi + torch.log(li)

            tiles_o.append(oi)
            tiles_l.append(li)

        O = torch.cat(tiles_o, dim=-2)  # Shape: [batch, heads, seq_len, d]
        L = torch.cat(tiles_l, dim=-1)  # Shape: [batch, heads, seq_len]
        if not has_batch_size:
            O = O.squeeze(0)
            L = L.squeeze(0)

        ctx.save_for_backward(q, k, v, O, L)
        ctx.is_causal = is_causal
        # ctx.save_for_backward(L)
        return O

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
        # pdb.set_trace()
        grad_q = grad_s @ k * scale
        grad_k = grad_s.transpose(-2, -1) @ q * scale

        return grad_q, grad_k, grad_v, None, None, None, None


def get_q_k_v():
    dims = (batch_size, num_heads, seq_length, head_size)
    print("dims:", dims)
    get = lambda: torch.rand(dims, device=device, dtype=torch.float32, requires_grad=True)  # noqa: E731
    return get(), get(), get()


if __name__ == "__main__":
    embedding_dim = 128
    head_size = 32
    num_heads = embedding_dim // head_size
    batch_size = 1
    seq_length = 128
    q, k, v = get_q_k_v()

    # x = batch_size, seq_length, embedding_dim
    # Q = embedding_dim, num_heads * head_size (embedding_dim)
    # x @ Q = batch_size, seq_length, embedding_dim (num_heads, head_size)
    # transpose ~ batch_size, num_heads, seq_length, head_size

    flashattn = FlashForward.apply
    flashattn_compiled = torch.compile(FlashForward.apply)
    output2, _ = flashattn_compiled(q, k, v, False, 16, 16)

    start = timeit.default_timer()
    output, _ = flashattn(q, k, v, False, 16, 16)
    end1 = timeit.default_timer()
    torch.cuda.synchronize()
    output2, _ = flashattn_compiled(q, k, v, False, 16, 16)

    end2 = timeit.default_timer()
    torch.cuda.synchronize()
    print(f"flashattn time: {end1 - start:.6f} seconds")
    print(f"flashattn_compiled time: {end2 - end1:.6f} seconds")

    dummy_output = dummy_attention(q, k, v)
    print(torch.allclose(dummy_output, output))
