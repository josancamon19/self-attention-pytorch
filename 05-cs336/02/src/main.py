import math
import timeit

from sympy import SeqAdd
from cs336_basics.model import BasicsTransformerLM, Embedding
from cs336_basics.nn_utils import cross_entropy
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_configs = [
    {"name": "small", "d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    {"name": "medium", "d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    {"name": "large", "d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    {"name": "xl", "d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    {"name": "2.7B", "d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
]

target_seq_lengths = [128, 256, 512, 1024]
dtypes = [torch.float32, torch.float16, torch.bfloat16]


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


def get_random_batch(seq_length: int, padding_size: int = 0, batch_size: int = 4):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
    if not padding_size:
        return input_ids


@nvtx.range("warmup steps")
def run_warmup(model: torch.nn.Module, n_steps: int = 10, seq_length: int = 512):
    input_ids = get_random_batch(seq_length)
    for _ in range(n_steps):
        output = model(input_ids)
        loss = cross_entropy(output, torch.ones(output.shape[:-1], dtype=torch.int64, device=device))
        loss.backward()
        # loss computation, this is only backward for that idx, not the whole graph

    if torch.cuda.is_available():
        torch.cuda.synchronize()


class MeasurementType(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"  # backward includes both
    FULL = "full"  # + loss + optimizer


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
    seq_length: int = 512,
    use_autocast: bool = False,
    target_dtype: torch.dtype = torch.float32,
) -> tuple[float, float]:
    times = []
    optimizer = torch.optim.AdamW(model.parameters())

    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=use_autocast):
        for _ in range(n_steps):
            start = timeit.default_timer()
            output = model(get_random_batch(seq_length))
            with nvtx.range("BasicTransformerLM.backward"):
                if type == MeasurementType.BACKWARD or type == MeasurementType.FULL:
                    loss = cross_entropy(output, torch.ones(output.shape[:-1], dtype=torch.int64, device=device))
                    del output
                    loss.backward()

            if type == MeasurementType.FULL:
                with nvtx.range("optimizer.step"):
                    optimizer.step()

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


class LogType(Enum):
    SEQ_LENGTH = "seq_length"
    DTYPE = "dtype"


def log(type: LogType):
    dtype: torch.dtype = torch.float32
    seq_length: int = 512

    torch.cuda.memory._record_memory_history(max_entries=1000000)
    for value in target_seq_lengths if type == LogType.SEQ_LENGTH else dtypes:
        if type == LogType.SEQ_LENGTH:
            seq_length = value
        else:
            dtype = value

        for config in model_configs:
            torch.cuda.synchronize()
            model = get_model(
                seq_length=seq_length,
                d_model=config["d_model"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
                dff=config["d_ff"],
                rope_theta=config.get("rope_theta", 10000),
            )
            run_warmup(model, n_steps=5, seq_length=seq_length)
            avg, std = measure(
                model,
                n_steps=10,
                seq_length=seq_length,
                use_autocast=type == LogType.DTYPE,
                target_dtype=dtype,
            )
            print(f"Config: {config}, Avg Time: {avg:.4f} ± {std:.4f} seconds")
            # break
        break

    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    torch.cuda.memory._record_memory_history(enabled=None)


class SingleHeadAttention(torch.nn.Module):
    def __init__(self, embedding_dim, head_dim):
        super().__init__()
        self.QKV = torch.nn.Parameter(torch.ones((embedding_dim, head_dim * 3), device=device))
        self.scale = 1 / math.sqrt(head_dim)

    @nvtx.range("SHA.forward")
    def forward(self, x):
        with nvtx.range("x @ QKV"):
            qkv = x @ self.QKV
        q, k, v = qkv.chunk(3, dim=-1)
        with nvtx.range("q @ k.T / sqrt(d_k)"):
            attention_scores = q @ k.transpose(-2, -1) * self.scale
        with nvtx.range("softmax"):
            attention_weights = torch.softmax(attention_scores, dim=-1)
        with nvtx.range("attn_weights @ v"):
            output = attention_weights @ v
        return output


def attention(use_compile: bool = False):
    batch_size = 8
    seq_length = 512
    embedding_dim = 768
    head_dim = [16, 32, 64, 128]
    # torch.cuda.memory._record_memory_history(max_entries=1000000)
    x = get_random_batch(seq_length, batch_size=batch_size)
    # embedding = Embedding(vocab_size, embedding_dim).to(device)
    # x_emb = embedding(x).detach()  # remove from computation graph
    x_emb = torch.randn((*x.shape, embedding_dim), requires_grad=True, device=device)
    passes = 100

    for dim in head_dim:
        attn = SingleHeadAttention(embedding_dim, dim)
        if use_compile:
            attn = torch.compile(attn)

        # warmup
        for _ in range(5):
            output = attn(x_emb)
            loss = cross_entropy(output, torch.ones(output.shape[:-1], dtype=torch.int64, device=device))
            loss.backward()
            attn.zero_grad()

        torch.cuda.synchronize()

        # measure
        start = timeit.default_timer()
        for _ in range(passes):
            output = attn(x_emb)
            with nvtx.range("backward"):
                loss = cross_entropy(output, torch.ones(output.shape[:-1], dtype=torch.int64, device=device))
                loss.backward()
            
            attn.zero_grad()
            torch.cuda.synchronize()

        print(f"Head dim: {dim}, Time taken: {timeit.default_timer() - start:.4f} seconds")
        break


if __name__ == "__main__":
    # log_baselines()
    # precision_checks()
    # log(LogType.SEQ_LENGTH)
    attention(True)
