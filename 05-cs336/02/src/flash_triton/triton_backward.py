import math

import triton
import triton.language as tl

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