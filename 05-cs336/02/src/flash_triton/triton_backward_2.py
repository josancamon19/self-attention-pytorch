import triton
import triton.language as tl

# Lots of help from claude tbh, mostly
from src.flash_triton.shared import get_cuda_autotune_config

# @triton.autotune(configs=get_cuda_autotune_config(), key=["seq_length", "head_dim"])
@triton.jit
def flash_backward_pass1_grad_q(
    grad_out_ptr,
    D_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    grad_q_ptr,  # Only compute grad_q in this pass
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    q_block_id = tl.program_id(0)  # NOW PARALLELIZE OVER Q-BLOCKS
    bh = tl.program_id(1)  # batch * head
    
    base_offset = bh * seq_length * head_dim
    
    # Load Q block (fixed for this kernel instance)
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(q_block_id * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0)
    )
    
    # Load other Q-related tensors
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
    
    q_tile = tl.load(q_block_ptr, boundary_check=(0, 1))
    o_tile = tl.load(o_block_ptr, boundary_check=(0, 1))
    grad_out_tile = tl.load(grad_out_block_ptr, boundary_check=(0, 1))
    
    # Load 1D tensors
    one_dim_offset = bh * seq_length + q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    one_dim_mask = (q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < seq_length
    l_tile = tl.load(l_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]
    D_tile = tl.load(D_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]
    
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # Initialize grad_q accumulator
    grad_q = tl.zeros((BLOCK_SIZE_Q, head_dim), dtype=tl.float32)
    
    # Loop over K blocks (no atomic operations needed!)
    for k_block_id in range(0, tl.cdiv(seq_length, BLOCK_SIZE_K)):
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
            # Load K and V blocks
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
            
            k_tile = tl.load(k_block_ptr, boundary_check=(0, 1))
            v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))
            
            # Compute attention scores
            s = tl.dot(q_tile, tl.trans(k_tile)) * scale
            
            # Apply causal mask
            if needs_masking:
                q_indices = q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
                k_indices = k_block_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
                causal_mask = q_indices >= k_indices
                s = tl.where(causal_mask, s, float("-inf"))
            
            p = tl.exp(s - l_tile)
            
            # Compute grad_s and accumulate grad_q
            grad_p = tl.dot(grad_out_tile, tl.trans(v_tile))
            grad_s = p * (grad_p - D_tile) * scale
            
            # Accumulate grad_q (NO ATOMICS!)
            grad_q += tl.dot(grad_s, k_tile.to(tl.float32))
    
    # Store grad_q 
    grad_q_block_ptr = tl.make_block_ptr(
        base=grad_q_ptr + base_offset,
        shape=(seq_length, head_dim),
        strides=(head_dim, 1),
        offsets=(q_block_id * BLOCK_SIZE_Q, 0),
        block_shape=(BLOCK_SIZE_Q, head_dim),
        order=(1, 0)
    )
    
    tl.store(grad_q_block_ptr, grad_q.to(q_tile.dtype), boundary_check=(0, 1))
    
# @triton.autotune(configs=get_cuda_autotune_config(), key=["seq_length", "head_dim"])  
@triton.jit
def flash_backward_pass2_grad_kv(
    grad_out_ptr,
    D_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    l_ptr,
    grad_k_ptr,  # Only compute grad_k and grad_v
    grad_v_ptr,
    seq_length: tl.constexpr,
    head_dim: tl.constexpr,
    is_causal: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    k_block_id = tl.program_id(0)  # BACK TO K-BLOCK PARALLELIZATION
    bh = tl.program_id(1)  # batch * head
    
    base_offset = bh * seq_length * head_dim
    
    # Load K and V tiles (fixed for this kernel instance)
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
    
    k_tile = tl.load(k_block_ptr, boundary_check=(0, 1))
    v_tile = tl.load(v_block_ptr, boundary_check=(0, 1))
    
    scale = 1.0 / tl.sqrt(float(head_dim))
    
    # Initialize grad_k and grad_v accumulators
    grad_k = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    grad_v = tl.zeros((BLOCK_SIZE_K, head_dim), dtype=tl.float32)
    
    # Loop over Q blocks
    for q_block_id in range(0, tl.cdiv(seq_length, BLOCK_SIZE_Q)):
        should_process = True
        needs_masking = False
        
        if is_causal:
            q_start = q_block_id * BLOCK_SIZE_Q
            q_end = (q_block_id + 1) * BLOCK_SIZE_Q - 1
            k_start = k_block_id * BLOCK_SIZE_K
            k_end = (k_block_id + 1) * BLOCK_SIZE_K - 1
            
            # Skip if entire Q block is "future" relative to K block
            should_process = q_end >= k_start
            # Need masking if there's partial overlap (diagonal blocks)
            needs_masking = should_process and (q_start < k_end)
        
        if should_process:
            # Load Q-related tensors (same as before)
            q_block_ptr = tl.make_block_ptr(
                base=q_ptr + base_offset,
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
            
            q_tile = tl.load(q_block_ptr, boundary_check=(0, 1))
            grad_out_tile = tl.load(grad_out_block_ptr, boundary_check=(0, 1))
            
            # Load 1D tensors
            one_dim_offset = bh * seq_length + q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
            one_dim_mask = (q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)) < seq_length
            l_tile = tl.load(l_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]
            D_tile = tl.load(D_ptr + one_dim_offset, mask=one_dim_mask, other=0.0)[:, None]
            
            # Compute attention scores
            s = tl.dot(q_tile, tl.trans(k_tile)) * scale
            
            # Apply causal mask
            if needs_masking:
                q_indices = q_block_id * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)[:, None]
                k_indices = k_block_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)[None, :]
                causal_mask = q_indices >= k_indices
                s = tl.where(causal_mask, s, float("-inf"))
            
            p = tl.exp(s - l_tile)
            
            # Compute gradients and accumulate grad_k, grad_v
            grad_v += tl.dot(tl.trans(p.to(grad_out_tile.dtype)), grad_out_tile).to(tl.float32)
            grad_p = tl.dot(grad_out_tile, tl.trans(v_tile))
            grad_s = p * (grad_p - D_tile) * scale
            grad_k += tl.dot(tl.trans(grad_s.to(q_tile.dtype)), q_tile).to(tl.float32)
    
    # Store grad_k and grad_v (same as before)
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