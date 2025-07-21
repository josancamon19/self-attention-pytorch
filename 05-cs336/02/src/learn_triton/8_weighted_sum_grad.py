import torch
import importlib.util
import os
import triton
import triton.language as tl


torch.manual_seed(42)

# wanted to keep naming int_
spec = importlib.util.spec_from_file_location(
    "weighted_sum_module", os.path.join(os.path.dirname(__file__), "7_weighted_sum.py")
)
weighted_sum_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighted_sum_module)
weighted_sum = weighted_sum_module.weighted_sum
weighted_sum_kernel = weighted_sum_module.weighted_sum_kernel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# import pdb  # noqa: E402

# os.environ["TRITON_INTERPRET"] = "1"


@triton.jit
def weighted_sum_backward_kernel(
    x_ptr,
    weight_ptr,
    m,  # xm
    n,  # xn, wn
    stride_xm,  # stride_gradxm as well
    stride_xn,  # stride_gradxn, stride_w as well
    grad_out_ptr,  # received grad
    grad_x_ptr,
    partial_grad_weight_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    i = tl.program_id(0)
    j = tl.program_id(1)

    offset_m = i * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_n = j * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    mask_m = offset_m < m
    mask_n = offset_n < n
    mask = mask_m[:, None] & mask_n[None, :]

    # grad_x[i,j] = grad_out[i] * weight[j]
    grad_out_block = tl.load(grad_out_ptr + offset_m, mask=mask_m)
    weight_block = tl.load(weight_ptr + offset_n, mask=mask_n)

    grad_x_block = grad_out_block[:, None] * weight_block[None, :]
    grad_x_ptrs = grad_x_ptr + offset_m[:, None] * stride_xm + offset_n[None, :] * stride_xn
    # pdb.set_trace()
    # TODO: agh there's a fucking bug, pointers and block and mask seem correct, but still fails, why?
    tl.store(grad_x_ptrs, grad_x_block, mask=mask)


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        return weighted_sum(x, weight, ctx)

    @staticmethod
    def backward(ctx, grad_out):
        print("grad_out:", grad_out, grad_out.shape)
        # grad_out shape = x.shape[:-1]
        # - whatever the output of forward() shape
        # - this means if you change any of the values of the output of forward, how much whatever you compute changes
        # - in case of a sum, if you incr 1, will change 1
        # - in case of .mean(), if you incr 1 will change 1/numel()

        # given this
        # - find the computation that resulted in forward output i,j
        # - output = forward()
        # - for computing output[i], we multiplied x[i] * weight, then summed
        # -- if I incr 1 for x[i, j], how does output[i] changes? "d(output[i]) / d(x[i,j])"
        # --- remember grad_out already knows how each of the item affects
        # --- d(output[i])/d(x[i,j]) = weight[j]
        # --- By chain rule: grad_x[i,j] = grad_out[i] * weight[j]
        # -- if I incr 1 for weight[j], d(output[i]) / d(weight[j])
        # --- d(output[i])/d(weight[j]) = x[i,j]
        # --- weight[j] affects all output ~ need to sum contributions from all batch items
        # --- grad_weight[j] = sum_i(grad_out[i] * x[i,j]) for all i

        x, weight = ctx.saved_tensors
        m, n = x.shape

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty(
            (triton.cdiv(m, 32), n),
            device=x.device,
            dtype=x.dtype,
        )

        BLOCK_SIZE_M = BLOCK_SIZE_N = 16
        grid = (triton.cdiv(m, BLOCK_SIZE_M), triton.cdiv(n, BLOCK_SIZE_N))
        weighted_sum_backward_kernel[grid](
            x,
            weight,
            m,
            n,
            x.stride(0),
            x.stride(1),
            grad_out,
            grad_x,
            partial_grad_weight,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
        )
        print(grad_x)
        grad_weight = partial_grad_weight.sum(axis=0)

        # 2
        return grad_x, grad_weight


def test():
    # Test data
    x = torch.randint(0, 10, (4, 10), device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.rand((10,), device=device, dtype=torch.float32, requires_grad=True)

    # Test 1: Your custom implementation
    f_weightedsum = WeightedSumFunc.apply
    output_custom = f_weightedsum(x, weight)
    loss_custom = output_custom.sum()
    loss_custom.backward()

    # Save your gradients
    grad_x_custom = x.grad.clone()
    grad_weight_custom = weight.grad.clone()

    # Clear gradients for next test
    x.grad = None
    weight.grad = None

    # Test 2: PyTorch reference implementation
    output_torch = torch.sum(x * weight, dim=-1)  # Same as weighted sum
    loss_torch = output_torch.sum()
    loss_torch.backward()

    # Compare gradients
    grad_x_torch = x.grad
    grad_weight_torch = weight.grad

    print("=== Gradient Comparison ===")
    print(f"grad_x match: {torch.allclose(grad_x_custom, grad_x_torch, atol=1e-5)}")
    print(f"grad_weight match: {torch.allclose(grad_weight_custom, grad_weight_torch, atol=1e-5)}")

    if not torch.allclose(grad_x_custom, grad_x_torch, atol=1e-5):
        print("grad_x difference:")
        print("Custom:", grad_x_custom)
        print("PyTorch:", grad_x_torch)
        print("Max diff:", torch.max(torch.abs(grad_x_custom - grad_x_torch)))


if __name__ == "__main__":
    test()
