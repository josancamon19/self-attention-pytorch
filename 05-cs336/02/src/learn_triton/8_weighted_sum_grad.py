import torch
import importlib.util
import os
import triton
import pdb

# wanted to keep naming int_
spec = importlib.util.spec_from_file_location(
    "weighted_sum_module", os.path.join(os.path.dirname(__file__), "7_weighted_sum.py")
)
weighted_sum_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weighted_sum_module)
weighted_sum = weighted_sum_module.weighted_sum
weighted_sum_kernel = weighted_sum_module.weighted_sum_kernel


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        return weighted_sum(x, weight, ctx)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        m, n = x.shape

        grad_x = torch.empty_like(x)
        partial_grad_weight = torch.empty(
            (triton.cdiv(m, 32), n),
            device=x.device,
            dtype=x.dtype,
        )

        grad_weight = partial_grad_weight.sum(axis=0)

        return grad_x, grad_weight


f_weightedsum = WeightedSumFunc.apply
