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


@triton.autotune(configs=get_cuda_autotune_config(), key=["seq_length", "head_dim"])
@triton.jit
def flash_forward(
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
    scale *= 1.44269504  # 1/log(2) for base-2 exponentials

    # Loop over K/V blocks
    for k_block_id in range(0, tl.cdiv(seq_length, BLOCK_SIZE_K)):

        # Determine if this block should be processed (causal optimization)
        should_process = True
        needs_masking = False

        if is_causal:
            q_start = q_block_id * BLOCK_SIZE_Q
            q_end = (q_block_id + 1) * BLOCK_SIZE_Q - 1
            k_start = k_block_id * BLOCK_SIZE_K
            k_end = (k_block_id + 1) * BLOCK_SIZE_K - 1

            # Skip if entire K block is "future" (all positions would be masked)
            should_process = q_end >= k_start
            # Need masking if there's partial overlap (diagonal blocks)
            needs_masking = should_process and (q_start < k_end)

        if should_process:
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
            if needs_masking:
                q_indices = q_block_id * BLOCK_SIZE_Q + \
                    tl.arange(0, BLOCK_SIZE_Q)[:, None]
                k_indices = k_block_id * BLOCK_SIZE_K + \
                    tl.arange(0, BLOCK_SIZE_K)[None, :]
                causal_mask = q_indices >= k_indices
                attn_scores = tl.where(causal_mask, attn_scores, float("-inf"))

            rowmax = tl.max(attn_scores, axis=1)

            prev_mi = mi
            mi = tl.maximum(mi, rowmax)
            pj = tl.math.exp2(attn_scores - mi[:, None])
            # pj_sum = tl.sum(pj, axis=-1).to(tl.float32)

            rescale_factor = tl.math.exp2(prev_mi - mi)
            li = rescale_factor * li + tl.sum(pj, axis=-1)
            oi = rescale_factor[:, None] * oi + \
                tl.dot(pj.to(v_tile.dtype), v_tile)

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

    l_offset = bh * seq_length + q_block_id * \
        BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    l_mask = (q_block_id * BLOCK_SIZE_Q +
              tl.arange(0, BLOCK_SIZE_Q)) < seq_length
    li = mi + tl.math.log2(li)
    tl.store(l_ptr + l_offset, li, mask=l_mask)


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

    # Loop over K/V blocks
    k_tiles = tl.cdiv(seq_length, BLOCK_SIZE_K)

    for k_block_id in tl.range(k_tiles, warp_specialize=False):
        # Determine if this block should be processed (causal optimization)
        should_process = True
        needs_masking = False

        if is_causal:
            q_start = q_block_id * BLOCK_SIZE_Q
            q_end = (q_block_id + 1) * BLOCK_SIZE_Q - 1
            k_start = k_block_id * BLOCK_SIZE_K
            k_end = (k_block_id + 1) * BLOCK_SIZE_K - 1

            # Skip if entire K block is "future" (all positions would be masked)
            should_process = q_end >= k_start
            # Need masking if there's partial overlap (diagonal blocks)
            needs_masking = should_process and (q_start < k_end)

        if should_process:
            # Load K and V tiles using TMA descriptors
            k_offset = k_block_id * BLOCK_SIZE_K
            k_tile = k_desc.load([k_offset, 0])
            v_tile = v_desc.load([k_offset, 0])

            # Compute attention scores
            attn_scores = tl.dot(q_tile, tl.trans(k_tile)) * scale

            # Apply causal mask if needed
            if needs_masking:
                q_indices = q_block_id * BLOCK_SIZE_Q + \
                    tl.arange(0, BLOCK_SIZE_Q)[:, None]
                k_indices = k_block_id * BLOCK_SIZE_K + \
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
