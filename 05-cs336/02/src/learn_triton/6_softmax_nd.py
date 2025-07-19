import triton
import triton.language as tl
import torch

import pdb
import os

# os.environ["TRITON_INTERPRET"] = "1"
# torch.manual_seed(42)


@triton.jit
def softmax_max_kernel(
    x_ptr,
    max_term_ptr,
    dim_size,
    dim_stride,
    BLOCK_SIZE: tl.constexpr,
):
    slice_id = tl.program_id(axis=0)  # which softmax are we doing
    chunk_id = tl.program_id(axis=1)  # which block of that softmax is this?

    offset = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < dim_size

    ptrs = x_ptr + slice_id * (dim_size * dim_stride) + offset * dim_stride
    x_block = tl.load(ptrs, mask=mask, other=float("-inf"))

    # pdb.set_trace()

    tl.atomic_max(max_term_ptr + slice_id, tl.max(x_block))


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    dim_size,
    dim_stride,
    max_term_ptr,
    div_term_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    slice_id = tl.program_id(axis=0)
    chunk_id = tl.program_id(axis=1)
    offset = chunk_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < dim_size

    # TODO: not fully clear this access here
    ptrs = x_ptr + slice_id * (offset * dim_stride) + offset * dim_stride
    x_block = tl.load(ptrs, mask=mask)
    
    # TODO: continue it later, exhausted, mental bandwidth is off

    numerator = tl.exp(x_block - tl.load(max_term_ptr + slice_id))
    tl.store(out_ptr + offset, numerator, mask=mask)
    tl.atomic_add(div_term_ptr, tl.sum(numerator))


@triton.jit
def softmax_normalize_kernel(
    out_ptr,
    global_sum_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(axis=0)
    offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements

    global_sum = tl.load(global_sum_ptr)
    values = tl.load(out_ptr + offset, mask=mask)
    tl.store(out_ptr + offset, values / global_sum, mask=mask)


def softmax(x: torch.Tensor, dim: int = -1):
    N = x.shape[dim]  # each softmax op elements
    stride = x.stride(dim)
    slices = x.numel() // N  # independent softmax ops

    BLOCK_SIZE = 32
    chunks_per_slice = triton.cdiv(N, BLOCK_SIZE)
    grid = (slices, chunks_per_slice)

    output = torch.empty_like(x, device=x.device, dtype=x.dtype)

    # div_term = torch.zeros((slices,), device=x.device, dtype=x.dtype)
    max_term = torch.full((slices,), float("-inf"), device=x.device, dtype=x.dtype)

    softmax_max_kernel[grid](x, max_term, N, stride, BLOCK_SIZE=BLOCK_SIZE)

    # softmax_kernel[grid](x, output, max_term, div_term, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    # softmax_normalize_kernel[grid](output, div_term, x.numel(), BLOCK_SIZE=BLOCK_SIZE)

    return max_term


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones((512, 256), device=device, dtype=torch.float32)
    x[0, 0] = 3
    x[128, 0] = 5
    output = softmax(x)
    # print(x)
    # print(output)
    # print(torch.sum(output))
    print(output)
    print(output.shape)
