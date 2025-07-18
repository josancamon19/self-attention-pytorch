import torch

import triton
import triton.language as tl
import pdb
import os

os.environ["TRITON_INTERPRET"] = "1"


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
    # each kernel launch, gives you a tile size, (i, j) to start from, given tile size (block sizes)
    row_idx = tl.program_id(axis=0)
    col_idx = tl.program_id(axis=1)

    # not sure how useful, tons of logs, gotta use pdb instead
    # tl.device_print("add_2d_kernel row:", row_idx)
    # tl.device_print("add_2d_kernel col:", col_idx)
    # tl.device_print("num_programs:", tl.num_programs(axis=0), tl.num_programs(axis=1))
    # tl.device_print("M, N, stride_am:", M, N, stride_am)

    # stride_am, row for a
    # stride_an, col for a
    # 2 block sizes, M and N cols

    offsets_m = row_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = col_idx * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offsets_m < M
    mask_n = offsets_n < N
    mask = mask_m[:, None] & mask_n[None, :]  # combine both masks True and True

    a = tl.load(a_ptr + offsets_m[:, None] * stride_am + offsets_n[None, :] * stride_an, mask=mask)
    b = tl.load(b_ptr + offsets_m[:, None] * stride_bm + offsets_n[None, :] * stride_bn, mask=mask)

    out = a + b
    # pdb.set_trace()
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
        BLOCK_M=32,  # how to determine? start with 128, fit in shared memory, multiple of warp size
        BLOCK_N=64,  # TODO: triton.autotune to find block sizes, Marcel mentioned smth related
    )
    return out


if __name__ == "__main__":
    # a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")
    # b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device="cuda")
    a = torch.rand(128, 256, device="cuda")
    b = torch.rand(128, 256, device="cuda")
    add_2d(a, b)
