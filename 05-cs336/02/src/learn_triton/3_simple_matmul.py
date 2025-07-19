import torch

import triton
import triton.language as tl

# os.environ["TRITON_INTERPRET"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Simple squared NxN * NxN


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    N,
    stride_m,  # 4 to new row
    stride_n,  # 1 to new col
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    offset_m = i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offset_n = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # C[i,j] = Σ(k=0 to N-1) A[i,k] * B[k,j]
    # - for block 0,0
    # - if BLOCK_SIZE = 8
    # -- would be generating results for 0,0 - 8,8
    # -- for that needs to read, 0,0, up to 8, N (rows)
    # -- and also 0,0 up to N,8 (cols)
    # -- we make those reads slicing, cause reading the whole row/col at once might oto much

    accum = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k in range(0, N, BLOCK_SIZE):
        # a_ptrs = A_ptr + (offset_m[:, None] * stride_m) +
        offset_k = k + tl.arange(0, BLOCK_SIZE)

        # TODO: [:, None] broadcasting review, very annoying but relevant here
        a_row_idx = offset_m[:, None]
        a_col_idx = offset_k[None, :]

        b_row_idx = offset_k[:, None]
        b_col_idx = offset_n[None, :]

        A_ptrs = A_ptr + (a_row_idx * stride_m) + (a_col_idx * stride_n)
        A_mask = (a_row_idx < N) & (a_col_idx < N)

        B_ptrs = B_ptr + (b_row_idx * stride_m) + (b_col_idx * stride_n)
        B_mask = (b_row_idx < N) & (b_col_idx < N)

        A_block = tl.load(A_ptrs, mask=A_mask)
        B_block = tl.load(B_ptrs, mask=B_mask)
        # ~~ Was running this on an nvidia T4, and apprently tl.dot doesn't support tl.float32 wtf
        partial = tl.dot(A_block.to(tl.float16), B_block.to(tl.float16))
        accum += partial
    out_ptrs = out_ptr + (offset_m[:, None] * stride_m) + (offset_n[None, :] * stride_n)
    out_mask = (offset_m[:, None] < N) & (offset_n[None, :] < N)
    tl.store(out_ptrs, accum, mask=out_mask)


def matmul(A: torch.Tensor, B: torch.Tensor):
    BLOCK_SIZE = 16  # tl.dot requires min a block size 16
    N = A.shape[0]
    assert A.shape == B.shape == (N, N)
    grid = (triton.cdiv(N, BLOCK_SIZE), triton.cdiv(N, BLOCK_SIZE))
    print("matmul grid:", grid)

    output = torch.empty_like(A).to(A.device, dtype=A.dtype)

    matmul_kernel[grid](
        A,
        B,
        output,
        N,
        A.stride(0),
        A.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output


if __name__ == "__main__":
    A = torch.randint(0, 10, (20, 20), device=device, dtype=torch.float32)
    B = torch.randint(0, 10, (20, 20), device=device, dtype=torch.float32)
    # B = torch.arange(0, 144, device=device, dtype=torch.float32).reshape((12, 12))
    output = matmul(A, B)
    print(output == A @ B)
