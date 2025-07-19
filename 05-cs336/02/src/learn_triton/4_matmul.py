import pdb
import torch

import triton
import triton.language as tl

# import os
# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def matmul_kernel(
    A_ptr,
    B_ptr,
    out_ptr,
    M,
    K,
    N,
    am_stride,
    ak_stride,
    bk_stride,
    bn_stride,
    om_stride,
    on_stride,
    # THIS IS IN TERMS OF OUTPUT MATRIX
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)

    m_offset = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    accum = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        k_offset = k + tl.arange(0, BLOCK_SIZE_K)

        a_row_idx = m_offset[:, None]
        a_col_idx = k_offset[None, :]
        A_ptrs = A_ptr + (a_row_idx * am_stride) + (a_col_idx * ak_stride)
        A_mask = (a_row_idx < M) & (a_col_idx < K)
        A_block = tl.load(A_ptrs, mask=A_mask)

        b_row_idx = k_offset[:, None]
        b_col_idx = n_offset[None, :]
        B_ptrs = B_ptr + (b_row_idx * bk_stride) + (b_col_idx * bn_stride)
        B_mask = (b_row_idx < K) & (b_col_idx < N)
        B_block = tl.load(B_ptrs, mask=B_mask)

        accum += tl.dot(A_block.to(tl.float16), B_block.to(tl.float16))
        # tl.device_print("accum", accum)
        # pdb.set_trace()

    out_ptrs = out_ptr + (m_offset[:, None] * om_stride) + (n_offset[None, :] * on_stride)
    out_mask = (m_offset[:, None] < M) & (n_offset[None, :] < N)
    tl.store(out_ptrs, accum, mask=out_mask)


def matmul(A: torch.Tensor, B: torch.Tensor):
    (M, K, N) = *A.shape, B.shape[1]
    print(A.shape, B.shape)
    result = torch.empty((M, N), device=A.device, dtype=A.dtype)
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32

    # smaller debugging values
    # BLOCK_SIZE_M = 8
    # BLOCK_SIZE_N = 4
    # BLOCK_SIZE_K = 2

    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))

    matmul_kernel[grid](
        A,
        B,
        result,
        M,
        K,
        N,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        result.stride(0),
        result.stride(1),
        BLOCK_SIZE_M,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
    )
    return result


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],
        line_arg="source",
        line_vals=["triton", "torch"],
        line_names=["Triton", "Torch"],
        styles=[("green", "-"), ("blue", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance",
        args={"test": [1, 2]},
    )
)
def benchmark(M, N, K, source, test):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if source == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a @ b, quantiles=quantiles)
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)  # noqa
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # M, K, N = 128, 512, 1024
    # A = torch.rand((M, K), device=device, dtype=torch.float16)  # 12, 4
    # B = torch.rand((K, N), device=device, dtype=torch.float16)  # 4, 8
    # output = matmul(A, B)
    # print(output.shape)
    # print(output == A @ B)

    benchmark.run(show_plots=True, print_data=True, save_path="./")
