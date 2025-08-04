import torch
from src.flash_triton.triton_forward import flash_forward
from src.flash_triton.triton_backward_2 import flash_backward_pass2_grad_kv, flash_backward_pass1_grad_q
import triton


# Set up TMA allocator for Triton 3.4.0 TMA functionality
def alloc_fn(size: int, alignment: int, stream):
    return torch.empty(size, device="cuda", dtype=torch.int8)


triton.set_allocator(alloc_fn)


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
        # has_batch_size = len(q.shape) == 4
        # if has_batch_size:
        #     batch_size, num_heads, seq_length, d = q.shape
        # else:
        num_heads, seq_length, d = q.shape
        batch_size = 1

        batch_heads = batch_size * num_heads

        l = torch.empty((batch_heads, seq_length), device=q.device, dtype=q.dtype)
        o = torch.empty_like(q)

        # def reshape(m): return m.reshape((batch_heads, seq_length, d))  # noqa: E731

        # q_tile_size, k_tile_size, num_warps, num_stages = 64, 64, 4, 3
        # autotune value performs worst (?)
        # q_tile_size, k_tile_size, num_warps, num_stages = 128, 64, 4, 4
        # grid = (triton.cdiv(seq_length, q_tile_size), batch_heads)
        grid = lambda meta: (triton.cdiv(seq_length, meta["BLOCK_SIZE_Q"]), batch_heads)  # noqa: E731

        # flash_forward[grid](
        flash_forward[grid](
            # reshape(q),
            # reshape(k),
            # reshape(v),
            q,
            k,
            v,
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
            # WARP_SPECIALIZE=False,  # TMA + warp specialization causes race conditions
        )
        # if has_batch_size:
        #     o = o.reshape(batch_size, num_heads, seq_length, d)
        #     l = l.reshape(batch_size, num_heads, seq_length)

        ctx.save_for_backward(q, k, v, o, l)
        ctx.is_causal = is_causal
        # pdb.set_trace()
        return o

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, o, l = ctx.saved_tensors
        D = torch.sum(o * grad_out, dim=-1)
        grad_out = grad_out.contiguous()

        # has_batch_size = len(q.shape) == 4
        # if has_batch_size:
        #     batch_size, num_heads, seq_length, d = q.shape
        # else:
        num_heads, seq_length, d = q.shape
        batch_size = 1
        batch_heads = batch_size * num_heads

        # Initialize ALL gradients
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)

        # def reshape(m): return m.reshape((batch_heads, seq_length, d))
        def reshape(m):
            return m

        # BLOCK_SIZE_Q, BLOCK_SIZE_K, num_warps, num_stages = 256, 64, 8, 3  # 3 or 5
        # grid = (triton.cdiv(seq_length, BLOCK_SIZE_Q), batch_heads)
        grid = lambda meta: (triton.cdiv(seq_length, meta["BLOCK_SIZE_Q"]), batch_heads)  # noqa
        flash_backward_pass1_grad_q[grid](
            reshape(grad_out),
            D.reshape(batch_heads, seq_length),
            reshape(q),
            reshape(k),
            reshape(v),
            reshape(o),
            l.reshape(batch_heads, seq_length),
            grad_q,  # Only grad_q computed
            seq_length,
            d,
            ctx.is_causal,
            # BLOCK_SIZE_Q,
            # BLOCK_SIZE_K,
            # num_warps=num_warps,
            # num_stages=num_stages,
        )

        # PASS 2: Compute grad_k and grad_v (parallelize over K blocks)
        # BLOCK_SIZE_Q, BLOCK_SIZE_K, num_warps, num_stages = 64, 64, 4, 3
        # grid = (triton.cdiv(seq_length, BLOCK_SIZE_K), batch_heads)
        grid = lambda meta: (triton.cdiv(seq_length, meta["BLOCK_SIZE_K"]), batch_heads)  # noqa

        flash_backward_pass2_grad_kv[grid](
            reshape(grad_out),
            D.reshape(batch_heads, seq_length),
            reshape(q),
            reshape(k),
            reshape(v),
            reshape(o),
            l.reshape(batch_heads, seq_length),
            grad_k,  # grad_k computed
            grad_v,  # grad_v computed
            seq_length,
            d,
            ctx.is_causal,
            # BLOCK_SIZE_Q,
            # BLOCK_SIZE_K,
            # num_warps=num_warps,
            # num_stages=num_stages,
        )

        # Reshape gradients to match input shapes
        # if has_batch_size:
        #     grad_q = grad_q.reshape(batch_size, num_heads, seq_length, d)
        #     grad_k = grad_k.reshape(batch_size, num_heads, seq_length, d)
        #     grad_v = grad_v.reshape(batch_size, num_heads, seq_length, d)

        return grad_q, grad_k, grad_v, None, None, None, None
