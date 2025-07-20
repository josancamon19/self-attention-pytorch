import torch
import triton
import triton.language as tl
# import pdb
# import os

# os.environ["TRITON_INTERPRET"] = "1"

# x_block_ptr = tl.make_block_ptr(
#     x_ptr,
#     shape=(num_elements,),
#     strides=(1,),
#     offsets=(i * BLOCK_SIZE),
#     block_shape=(BLOCK_SIZE,),
#     order=(0,),
# )
# x_block = tl.load(x_block_ptr, boundary_check=(0,))


@triton.jit
def weighted_sum_kernel(
    x_ptr,
    stride_m,
    stride_n,
    weight_ptr,
    output_ptr,
    m,
    n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    offset_m = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offset_m < m
    mask_n = offset_n < n
    mask = mask_m[:, None] & mask_n[None, :]

    x_ptrs = x_ptr + offset_m[:, None] * stride_m + offset_n[None, :] * stride_n
    x_block = tl.load(x_ptrs, mask=mask)
    w_block = tl.load(weight_ptr + offset_n, mask=mask_n)
    tl.atomic_add(output_ptr + offset_m, tl.sum(x_block * w_block, axis=1))


def weighted_sum(x, weight):
    output_dims = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    m, n = x.shape  # m x n

    assert len(weight.shape) == 1 and weight.shape[0] == n
    assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

    output = torch.zeros(m, device=x.device, dtype=x.dtype)
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8

    stride_m, stride_n = x.stride(0), x.stride(1)
    grid = (triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))

    weighted_sum_kernel[grid](
        x,
        stride_m,
        stride_n,
        weight,
        output,
        m,
        n,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
    )
    return output.view(output_dims)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randint(0, 10, (100, 1000, 14000), device=device, dtype=torch.float32)
    weight = torch.rand((14000,), device=device, dtype=torch.float32)
    output = weighted_sum(x, weight)
    torch_output = torch.sum(x * weight, axis=-1)
    print("output ≈ torch_output", torch.allclose(output, torch_output))
    # print("output", output)
    # print("output", torch_output)
