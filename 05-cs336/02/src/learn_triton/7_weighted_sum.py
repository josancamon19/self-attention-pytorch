import torch
import triton
import triton.language as tl
# import pdb
# import os

# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def weighted_sum_kernel(
    x_ptr,
    weight_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(axis=0)

    x_block_ptr = tl.make_block_ptr(
        x_ptr,
        shape=(num_elements,),
        strides=(1,),
        offsets=(i * BLOCK_SIZE),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    x_block = tl.load(x_block_ptr, boundary_check=(0,))

    offset = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    w_block = tl.load(weight_ptr + offset, mask=mask)
    tl.atomic_add(output_ptr, tl.sum(x_block * w_block))


def weighted_sum(x, weight):
    BLOCK_SIZE = 32
    N = x.numel()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    output = torch.zeros(1, device=x.device, dtype=x.dtype)
    weighted_sum_kernel[grid](x, weight, output, N, BLOCK_SIZE)
    return output


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.ones((8,), device=device, dtype=torch.float32)
    weight = torch.rand((8,), device=device, dtype=torch.float32)
    output = weighted_sum(x, weight)
    print("output", output)
    print("output", torch.sum(x * weight))
