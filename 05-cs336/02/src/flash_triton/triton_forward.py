import torch
import triton
import triton.language as tl
import os

torch.manual_seed(42)

# @triton.autotune(configs=get_cuda_autotune_config(), key=["q", "k", "v"])
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
    
    l_offset = bh * seq_length + q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    l_mask = (q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < seq_length
    li = mi + tl.log(li)
    tl.store(l_ptr + l_offset, li, mask=l_mask)
    