import timeit
from cs336_basics.model import BasicsTransformerLM
import torch
from enum import Enum
import torch.cuda.nvtx as nvtx

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
seq_length = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = [
    {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"name": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]


def get_model(seq_length, d_model, num_layers, num_heads, dff, rope_theta=10000):
    return BasicsTransformerLM(
        vocab_size,
        seq_length,
        d_model,
        num_layers,
        num_heads,
        dff,
        rope_theta,
    ).to(device)


def get_random_batch(seq_length: int, padding_size: int = 0):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    if not padding_size:
        return input_ids

    # assert padding_size >= seq_length
    # padding_mask = torch.zeros_like(input_ids)


@nvtx.range("warmup steps")
def run_warmup(model: torch.nn.Module, n_steps: int = 10):
    for _ in range(n_steps):
        output = model(get_random_batch(seq_length))
        output.sum().backward()

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MeasurementType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"  # backward includes both


def profile_with_torch_profiler(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        if torch.cuda.is_available()
        else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        fn()
    print(
        prof.key_averages().table(
            sort_by="self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total", row_limit=10
        )
    )


@nvtx.range("measurement")
def measure(
    model: torch.nn.Module,
    n_steps: int,
    type: MeasurementType = MeasurementType.BACKWARD,
) -> tuple[float, float]:
    times = []
    for _ in range(n_steps):
        start = timeit.default_timer()
        output = model(get_random_batch(seq_length))
        if type == MeasurementType.BACKWARD:
            with nvtx.range("BasicTransformerLM.backward"):
                output.sum().backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(timeit.default_timer() - start)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def precision_checks():
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.bfloat16)
    print(s)
    
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)


if __name__ == "__main__":
    # TODO: different sequence lengths
    # TODO: different datatypes
    precision_checks()
    # for config in model_configs:
    #     model = get_model(
    #         seq_length=seq_length,
    #         d_model=config["d_model"],
    #         num_layers=config["num_layers"],
    #         num_heads=config["num_heads"],
    #         dff=config["d_ff"],
    #         rope_theta=config.get("rope_theta", 10000),
    #     )
    #     run_warmup(model, n_steps=5)
    #     avg, std = measure(model, n_steps=10)
    #     print(f"Config: {config}, Avg Time: {avg:.4f} ± {std:.4f} seconds")
    #     break
