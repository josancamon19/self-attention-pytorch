
import torch
from src.flash_triton.triton_forward import flash_forward
from src.flash_triton.triton_backward import flash_backward
from src.flash_triton.triton_backward_2 import flash_backward_pass2_grad_kv, flash_backward_pass1_grad_q
import triton

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

        flash_forward[grid](
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


    # @staticmethod
    # def backward(ctx, grad_out):
    #     q, k, v, o, l = ctx.saved_tensors
    #     D = torch.sum(o * grad_out, dim=-1)
    #     grad_out = grad_out.contiguous()
        
    #     has_batch_size = len(q.shape) == 4
    #     if has_batch_size:
    #         batch_size, num_heads, seq_length, d = q.shape
    #     else:
    #         num_heads, seq_length, d = q.shape
    #         batch_size = 1

    #     batch_heads = batch_size * num_heads

    #     shape = (batch_heads, seq_length, d)
    #     grad_q = torch.zeros(shape, device=q.device, dtype=torch.float32)  # atomic ops
    #     grad_k = torch.empty(shape, device=k.device, dtype=q.dtype)  # fp32 for accumulation
    #     grad_v = torch.empty(shape, device=v.device, dtype=q.dtype)  # fp32 for accumulation

    #     BLOCK_SIZE_Q, BLOCK_SIZE_K = 64, 64
    #     grid = (triton.cdiv(seq_length, BLOCK_SIZE_K), batch_heads)
    #     # grid = lambda meta: (triton.cdiv(seq_length, meta['BLOCK_SIZE_K']), batch_heads) # noqa
    #     reshape = lambda m: m.reshape((batch_heads, seq_length, d)) # noqa

    #     # pdb.set_trace()
    #     flash_backward[grid](
    #         reshape(grad_out),
    #         D.reshape(batch_heads, seq_length),
    #         reshape(q),
    #         reshape(k),
    #         reshape(v),
    #         reshape(o),
    #         l.reshape(batch_heads, seq_length),
    #         grad_q,
    #         grad_k,
    #         grad_v,
    #         seq_length,
    #         d,
    #         ctx.is_causal,
    #         BLOCK_SIZE_Q,
    #         BLOCK_SIZE_K,
    #     )
        
    #     # Reshape gradients to match input shapes
    #     if has_batch_size:
    #         grad_q = grad_q.reshape(batch_size, num_heads, seq_length, d)
    #         grad_k = grad_k.reshape(batch_size, num_heads, seq_length, d)
    #         grad_v = grad_v.reshape(batch_size, num_heads, seq_length, d)
        
    #     return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype), None, None, None, None
    
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
        
        # Initialize ALL gradients
        grad_q = torch.empty(shape, device=q.device, dtype=q.dtype)
        grad_k = torch.empty(shape, device=k.device, dtype=q.dtype)
        grad_v = torch.empty(shape, device=v.device, dtype=q.dtype)

        reshape = lambda m: m.reshape((batch_heads, seq_length, d))

        # PASS 1: Compute grad_q (parallelize over Q blocks)
        BLOCK_SIZE_Q, BLOCK_SIZE_K = 64, 64
        grid = (triton.cdiv(seq_length, BLOCK_SIZE_Q), batch_heads)
        # grid = lambda meta: (triton.cdiv(seq_length, meta['BLOCK_SIZE_Q']), batch_heads)
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
            BLOCK_SIZE_Q,
            BLOCK_SIZE_K,
        )
        
        # PASS 2: Compute grad_k and grad_v (parallelize over K blocks)
        # grid_k = lambda meta: (triton.cdiv(seq_length, meta['BLOCK_SIZE_K']), batch_heads)
        grid = (triton.cdiv(seq_length, BLOCK_SIZE_K), batch_heads)
        flash_backward_pass2_grad_kv[grid](
            reshape(grad_out),
            D.reshape(batch_heads, seq_length),
            reshape(q),
            reshape(k),
            reshape(v),
            reshape(o),
            l.reshape(batch_heads, seq_length),
            grad_k,   # grad_k computed
            grad_v,   # grad_v computed
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
        
        return grad_q, grad_k, grad_v, None, None, None, None