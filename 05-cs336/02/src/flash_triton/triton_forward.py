import torch
import triton
import triton.language as tl
import os
from src.flash_triton.shared import get_cuda_autotune_config

torch.manual_seed(42)


def supports_tma():
    """Check if current GPU supports TMA (H100+)"""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


def supports_warp_specialize():
    """Check if current GPU supports warp specialization"""
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


HAS_TMA = supports_tma() and hasattr(tl, "make_tensor_descriptor")

# TMA-optimized Flash Attention Forward Pass
# @triton.autotune(configs=get_cuda_autotune_config(), key=["seq_length", "head_dim"])
@triton.jit
def flash_forward_tma(
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
    """
    TMA-optimized Flash Attention forward pass for H100+ GPUs.
    Uses tensor descriptors for efficient memory access patterns.
    """
    q_block_id = tl.program_id(0)  # q_tile
    bh = tl.program_id(1)  # batch * head

    # Calculate base offset for this batch/head
    base_offset = bh * seq_length * head_dim

    # Create TMA tensor descriptors for efficient memory access
    q_desc = tl.make_tensor_descriptor(
        q_ptr + base_offset,
        shape=[seq_length, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SIZE_Q, head_dim],
    )

    k_desc = tl.make_tensor_descriptor(
        k_ptr + base_offset,
        shape=[seq_length, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SIZE_K, head_dim],
    )

    v_desc = tl.make_tensor_descriptor(
        v_ptr + base_offset,
        shape=[seq_length, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SIZE_K, head_dim],
    )

    o_desc = tl.make_tensor_descriptor(
        o_ptr + base_offset,
        shape=[seq_length, head_dim],
        strides=[head_dim, 1],
        block_shape=[BLOCK_SIZE_Q, head_dim],
    )

    # Load Q tile using TMA descriptor
    q_offset = q_block_id * BLOCK_SIZE_Q
    q_tile = q_desc.load([q_offset, 0])

    # Initialize accumulators
    mi = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    li = tl.zeros((BLOCK_SIZE_Q,), dtype=tl.float32)
    oi = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)

    scale = 1.0 / tl.sqrt(float(head_dim))
    scale *= 1.44269504  # 1/log(2) for base-2 exponentials

    # Optimized causal attention: separate past, diagonal, and future blocks
    k_tiles = tl.cdiv(seq_length, BLOCK_SIZE_K)
    
    if is_causal:
        # Phase 1: Process past blocks (no masking needed)
        max_past_block = q_block_id
        for k_block_id in range(0, max_past_block):
            k_offset = k_block_id * BLOCK_SIZE_K
            k_tile = k_desc.load([k_offset, 0])
            v_tile = v_desc.load([k_offset, 0])

            # Compute attention scores (no masking)
            attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
            rowmax = tl.max(attn_scores, axis=1)

            prev_mi = mi
            mi = tl.maximum(mi, rowmax)
            pj = tl.math.exp2(attn_scores - mi[:, None])

            rescale_factor = tl.math.exp2(prev_mi - mi)
            li = rescale_factor * li + tl.sum(pj, axis=-1)
            oi = rescale_factor[:, None] * oi + \
                tl.dot(pj.to(v_tile.dtype), v_tile)

        # Phase 2: Process diagonal block (with masking)
        if q_block_id < k_tiles:
            k_offset = q_block_id * BLOCK_SIZE_K
            k_tile = k_desc.load([k_offset, 0])
            v_tile = v_desc.load([k_offset, 0])

            # Compute attention scores
            attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Apply causal mask (single comparison)
            q_indices = q_block_id * BLOCK_SIZE_Q + \
                tl.arange(0, BLOCK_SIZE_Q)[:, None]
            k_indices = q_block_id * BLOCK_SIZE_K + \
                tl.arange(0, BLOCK_SIZE_K)[None, :]
            causal_mask = q_indices >= k_indices
            attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))

            rowmax = tl.max(attn_scores, axis=1)

            prev_mi = mi
            mi = tl.maximum(mi, rowmax)
            pj = tl.math.exp2(attn_scores - mi[:, None])

            rescale_factor = tl.math.exp2(prev_mi - mi)
            li = rescale_factor * li + tl.sum(pj, axis=-1)
            oi = rescale_factor[:, None] * oi + \
                tl.dot(pj.to(v_tile.dtype), v_tile)

        # Phase 3: Future blocks are implicitly skipped (no loop needed)

    else:
        # Non-causal: process all blocks without masking
        for k_block_id in tl.range(k_tiles, warp_specialize=False):
            k_offset = k_block_id * BLOCK_SIZE_K
            k_tile = k_desc.load([k_offset, 0])
            v_tile = v_desc.load([k_offset, 0])

            # Compute attention scores (no masking)
            attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale
            rowmax = tl.max(attn_scores, axis=1)

            prev_mi = mi
            mi = tl.maximum(mi, rowmax)
            pj = tl.math.exp2(attn_scores - mi[:, None])

            rescale_factor = tl.math.exp2(prev_mi - mi)
            li = rescale_factor * li + tl.sum(pj, axis=-1)
            oi = rescale_factor[:, None] * oi + \
                tl.dot(pj.to(v_tile.dtype), v_tile)

    # Final normalization
    oi = oi / li[:, None]

    # Store output using TMA descriptor
    o_desc.store([q_offset, 0], oi.to(q_tile.dtype))

    # Store logsumexp values (fallback to regular store for simplicity)
    l_offset = bh * seq_length + q_block_id * \
        BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    l_mask = (q_block_id * BLOCK_SIZE_Q +
              tl.arange(0, BLOCK_SIZE_Q)) < seq_length
    li = mi + tl.math.log2(li)
    tl.store(l_ptr + l_offset, li, mask=l_mask)
