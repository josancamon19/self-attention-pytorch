import torch

import triton
import triton.language as tl


@triton.jit
def add_2d_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    stride_am,
    stride_an,
    stride_bm,
    stride_bn,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)
    # stride_am, row for a
    # stride_an, col for a
    # 2 block sizes, M and N cols

    # TODO: debug each item, shape, strides values

    offsets_m = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)

    offsets_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]  # combine both masks True and True
    a = tl.load(a_ptr + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an, mask=mask)
    b = tl.load(b_ptr + offsets_m[:, None] * stride_bm + offsets_n[None, :] * stride_bn, mask=mask)

    out = a + b
    tl.store(out_ptr + offsets_m[:, None] * stride_om + offsets_n[None, :] * stride_on, out, mask=mask)


def add_2d(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape
    M, N = a.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))  # noqa: E731

    out = torch.empty_like(a)
    print(f"Shape: M={M}, N={N}")
    print("a.stride:", a.stride())
    print("b.stride:", b.stride())
    add_2d_kernel[grid](
        a,
        b,
        out,
        M,
        N,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
        BLOCK_M=128,  # how to determine? start with 128, fit in shared memory, multiple of warp size
        BLOCK_N=128,  # TODO: triton.autotune to find block sizes, Marcel mentioned smth related
    )
    return out
