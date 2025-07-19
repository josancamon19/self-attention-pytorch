import triton
import triton.language as tl
import torch

# import pdb
# import os
# os.environ["TRITON_INTERPRET"] = "1"
torch.manual_seed(42)

# wrong math intuition, but interesting
# @triton.jit
# def softmax_kernel(
#     x_ptr,
#     out_ptr,
#     num_elements,
#     BLOCK_SIZE: tl.constexpr,
#     BLOCK_SIZE_K: tl.constexpr,
# ):
#     # tl.exp(x)
#     i = tl.program_id(axis=0)
#     offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = offset < num_elements

#     x_block = tl.load(x_ptr + offset, mask=mask)
#     numerator = tl.exp(x_block)  # exp goes only gpu, no pdb

#     out_ptrs = out_ptr + offset
#     tl.store(out_ptrs, tl.load(out_ptrs, mask=mask) * numerator, mask=mask)

#     # now I could do the operation of dividing here, but I need div_term to divide the whole thing
#     # I could initialize output with 1, and then mul * numerator, but divide div_term for all
#     # 1/3/2 == 1/6, nvm but you need 1/(3+2) fuck.

#     div_term = tl.sum(numerator)
#     for k in range(0, num_elements, BLOCK_SIZE_K):
#         k_offset = k + tl.arange(0, BLOCK_SIZE_K)
#         out_ptrs = out_ptr + k_offset
#         k_mask = k_offset < num_elements
#         tl.store(out_ptrs, tl.div_rn(tl.load(out_ptrs, mask=k_mask), div_term), mask=k_mask)


@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    div_term_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    offset = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements

    x_block = tl.load(x_ptr + offset, mask=mask)
    numerator = tl.exp(x_block)
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


def softmax(x: torch.Tensor):
    # let's assume it's 1D for now
    N = x.shape[0]
    BLOCK_SIZE = 32
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    output = torch.empty_like(x, device=x.device, dtype=x.dtype)
    div_term = torch.zeros(1, device=x.device, dtype=x.dtype)
    softmax_kernel[grid](x, output, div_term, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    softmax_normalize_kernel[grid](output, div_term, x.numel(), BLOCK_SIZE=BLOCK_SIZE)
    return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randint(0, 10, (1024,), device=device, dtype=torch.float32)
    output = softmax(x)
    # print(x)
    # print(output)
    print(torch.sum(output))
