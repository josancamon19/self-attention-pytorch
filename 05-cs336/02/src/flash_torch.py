import math
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
    def forward(ctx, q, k, v, is_causal: bool):
        # produce Output O and logsumexp value L
        print("q.shape", q.shape)
        scale = 1.0 / math.sqrt(k.shape[-1])
        attn_scores = torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1)
        print("attn_scores.shape", attn_scores.shape)
        output = attn_scores @ v
        print("output.shape", output.shape)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()


if __name__ == "__main__":
    embedding_dim = 128
    head_size = 32
    num_heads = embedding_dim // head_size
    batch_size = 1
    seq_length = 100

    # x = batch_size, seq_length, embedding_dim
    # Q = embedding_dim, num_heads * head_size (embedding_dim)
    # x @ Q = batch_size, seq_length, embedding_dim (num_heads, head_size)
    # transpose ~ batch_size, num_heads, seq_length, head_size

    dims = (batch_size, num_heads, seq_length, head_size)
    print("dims:", dims)
    get = lambda: torch.rand(dims, device=device, dtype=torch.float32, requires_grad=True)  # noqa: E731
    q, k, v = get(), get(), get()

    flashattn = FlashForward.apply
    flashattn(q, k, v, False)
