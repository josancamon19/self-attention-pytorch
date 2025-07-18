import torch

import triton
import triton.language as tl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# os.environ["TORCH_INTERPRET"] = "1" cpu, not macos lol


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x, device=device)
    assert x.device == device and y.device == device and output.device == device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_kernel():
    torch.manual_seed(0)
    size = 768324
    x = torch.rand(size, device=device)
    y = torch.rand(size, device=device)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f"The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}")


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],  # Argument names to use as an x-axis for the plot.
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg="provider",  # Argument name whose value corresponds to a different line in the plot.
        line_vals=["triton", "torch"],  # Possible values for `line_arg`.
        line_names=["Triton", "Torch"],  # Label name for the lines.
        styles=[("blue", "-"), ("green", "-")],  # Line styles.
        ylabel="GB/s",  # Label name for the y-axis.
        plot_name="vector-add-performance",  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device=device, dtype=torch.float32)
    y = torch.rand(size, device=device, dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == "triton":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa: E731
    return gbps(ms), gbps(max_ms), gbps(min_ms)


@triton.jit
def subtract_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr, offsets, mask=mask)
    y = tl.load(y_ptr, offsets, mask=mask)

    tl.store(out_ptr, y - x, mask=mask)


def sub(x: torch.Tensor, y: torch.Tensor):
    assert x.device == y.device
    assert x.shape == y.shape
    # TODO: how's torch broadcasting happening then?
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = (triton.cdiv(n_elements / 1024),)
    subtract_kernel[grid](x, y, output, n_elements)
    return output  # TODO: the kernel was launched async, why do we have a value of this


if __name__ == "__main__":
    test_kernel()
    # benchmark.run(print_data=True, show_plots=True)
