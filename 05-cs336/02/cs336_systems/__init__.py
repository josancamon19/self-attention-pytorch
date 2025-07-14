from enum import Enum
import importlib.metadata
import timeit
from cs336_basics.model import BasicsTransformerLM
import torch

__version__ = importlib.metadata.version("cs336-systems")

# Notes

# - For all models, we’ll use a vocabulary size of 10,000 and a batch size of 4, with varying context lengths.
# - automate constructing tables for your writeup in code pandas.DataFrame.to_latex()
# - model sizes table
# - we'll be testing precision changes, swapping layers, have your script enable this variations through cli ✅
# - suggests using `sbatch or submitit on Slurm`, why not ray? for quick iteration
# 1. timing fwd backward passes

# When you call a CUDA kernel,
# such as when you invoke torch.matmul, the function call returns control to your code without waiting for
# the matrix multiplication to finish. In this way, the CPU can continue running while the GPU computes the
# matrix multiplication. (start_time - time.time() doesn't work at all)
# torch.cuda.synchronize() to wait for all GPU kernels to complete

vocab_size = 10000
batch_size = 4


def get_model(seq_length, d_model, num_layers, num_heads, dff, rope_theta=10000):
    return BasicsTransformerLM(
        vocab_size,
        seq_length,
        d_model,
        num_layers,
        num_heads,
        dff,
        rope_theta,
    )


def get_random_batch(seq_length: int, padding_size: int = None):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    if not padding_size:
        return input_ids

    # assert padding_size >= seq_length
    # padding_mask = torch.zeros_like(input_ids)


def run_warmup(model: torch.nn.Module, n_steps: int = 100):
    for _ in range(n_steps):
        model(get_random_batch(), None)


class MeasurementType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"  # backward includes both


def measure(
    model: torch.nn.Module,
    n_steps: int,
    type: MeasurementType = MeasurementType.BACKWARD,
) -> float:
    start = timeit.default_timer()
    for _ in range(n_steps):
        output = model(get_random_batch(), None)
        if type == MeasurementType.BACKWARD:
            # this part might be wrong
            loss = torch.nn.functional.cross_entropy(output, get_random_batch())
            loss.backward()
        torch.cuda.synchronize()
    total = timeit.default_timer() - start
    print("[measure] n_steps:", n_steps, type)
    return total
