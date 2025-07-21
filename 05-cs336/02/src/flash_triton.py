import torch
import importlib.util
import os
import triton
import triton.language as tl
import pdb

os.environ["TRITON_INTERPRET"] = "1"

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== FlashAttention Forward Torch Autograd. ===


class FlashForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal: bool):
        # produce Output O and logsumexp value L
        pass

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


if __name__ == "__main__":
    pass