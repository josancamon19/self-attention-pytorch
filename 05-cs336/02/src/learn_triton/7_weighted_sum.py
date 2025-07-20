import torch
import triton
import triton.language as tl
import os
# import pdb

# os.environ["TRITON_INTERPRET"] = "1"
os.environ["TRITON_PRINT_AUTOTUNING"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "1"
torch._logging.set_logs(output_code=True, kernel_code=True, schedule=True, fusion=True)
# TODO: debug this more

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cuda_autotune_config():
    def config_item(m, n, ns, nw):
        return triton.Config({"BLOCK_SIZE_M": m, "BLOCK_SIZE_N": n})  # , num_stages=ns, num_warps=nw

    return [
        config_item(128, 256, 3, 8),
        config_item(64, 256, 4, 4),
        config_item(128, 128, 4, 4),
        config_item(128, 64, 4, 4),
        config_item(64, 128, 4, 4),
        config_item(128, 32, 4, 4),
        config_item(64, 32, 5, 2),
        config_item(32, 64, 5, 2),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["m", "n"], reset_to_zero=["output_ptr"])
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
    tl.atomic_add(output_ptr + offset_m, tl.sum(x_block * w_block, axis=1), mask=mask_m)


def weighted_sum(x, weight, ctx=None):
    output_dims = x.shape[:-1]
    x = x.reshape(-1, x.shape[-1])
    m, n = x.shape  # m x n

    assert len(weight.shape) == 1 and weight.shape[0] == n
    assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

    if ctx:
        ctx.save_for_backward(x, weight)

    output = torch.zeros(m, device=x.device, dtype=x.dtype)

    stride_m, stride_n = x.stride(0), x.stride(1)
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]), triton.cdiv(n, meta["BLOCK_SIZE_N"]))  # noqa: E731

    weighted_sum_kernel[grid](x, stride_m, stride_n, weight, output, m, n)
    return output.view(output_dims)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["m", "n"],
        x_vals=[128 * i for i in range(2, 48)],
        line_arg="source",
        line_vals=["triton", "torch", "compile"],
        line_names=["Triton", "Torch", "Compile"],
        styles=[("green", "-"), ("blue", "-"), ("red", "-")],
        ylabel="TFLOPS",
        plot_name="weighted-sum",
        args={},
    )
)
def benchmark(m, n, source):
    x = torch.randint(0, 10, (m, n), device=device, dtype=torch.float32)
    weight = torch.rand((n,), device=device, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if source == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.sum(x * weight, axis=-1), quantiles=quantiles)
    elif source == "compile":
        compiled_fn = torch.compile(lambda x, w: torch.sum(x * w, axis=-1))
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled_fn(x, weight), quantiles=quantiles)
    else:  # triton
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: weighted_sum(x, weight), quantiles=quantiles)

    perf = lambda ms: 1 * m * n * 1e-12 / (ms * 1e-3)  # noqa
    return perf(ms), perf(max_ms), perf(min_ms)


def unit_test():
    compiled_fn = torch.compile(lambda x, w: torch.sum(x * w, axis=-1))
    for i in range(12):
        x = torch.randint(0, 10, (100 * (i + 1), 2**i), device=device, dtype=torch.float32)
        weight = torch.rand((2**i,), device=device, dtype=torch.float32)
        output = weighted_sum(x, weight)
        torch_output = compiled_fn(x, weight)
        same = torch.allclose(output, torch_output)
        print("output ≈ torch_output", same)
        if not same:
            break


if __name__ == "__main__":
    unit_test()
    # benchmark.run(show_plots=True, print_data=True, save_path="./")
