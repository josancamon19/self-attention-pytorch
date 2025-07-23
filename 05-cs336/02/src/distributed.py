import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
from cs336_basics.model import BasicsTransformerLM


def setup(rank: int, world_size: int, backend: str = "gloo"):  # gpu_id, gpu_counts
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def test(rank, world_size):
    setup(rank, world_size, "gloo")
    data = torch.randint(0, 10, (10,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")
    # For timing results, mean measurements from diff ranks, using all-gather


def _benchmark(rank, world_size, backend, device_type, reduce_data_size_mb):
    setup(rank, world_size, backend)

    # 1MB = 1024kb, 1kb = 1024 bytes = (fp32 1 item 4 bytes), 32 items
    elements_per_mb = (1024 * 1024) // 4
    total_elements = elements_per_mb * reduce_data_size_mb
    tensor = torch.randint(0, 10, (1024, total_elements // 1024), dtype=torch.float32, device=f"{device_type}:{rank}")

    start = timeit.default_timer()
    dist.all_reduce(tensor, async_op=False)
    if device_type == "cuda":
        torch.cuda.synchronize()
    seconds = timeit.default_timer() - start
    print(f"_benchmark rank={rank}, seconds={seconds}")
    # TODO: Deliverable: Plot(s) and/or table(s) comparing the various settings,
    # with 2-3 sentences of commentary about your results and thoughts about how the various factors interact.


def benchmark(
    backend: str = "gloo",  # nccl
    device_type: str = "cpu",  # cuda
    reduce_data_size_mb: int = 1,
    processes: int = 2,  # 4, 5
):
    mp.spawn(
        _benchmark,
        args=(processes, backend, device_type, reduce_data_size_mb),
        nprocs=processes,
        join=True,
    )


# ======= 2.2 Naive DDP ========


class ToyModel(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.l1 = torch.nn.Linear(d_in, d_in * 4)
        self.relu = torch.nn.functional.relu
        self.l2 = torch.nn.Linear(d_in * 4, d_out)

    def forward(self, x):
        return self.l2(self.relu(self.l1(x.to(torch.float32))))


import enum  # noqa: E402


class OptimizationType(enum.Enum):
    NONE = "none"
    FLATTEN = "flatten"
    OVERLAPPING = "overlapping"


def train_ddp(
    rank: int,
    world_size: int,
    get_model: callable,
    _: int,
    input_dimensions: tuple,
    input_dtype: torch.dtype,
    optimization_type: OptimizationType = OptimizationType.NONE,
    backend: str = "gloo",
):
    setup(rank, world_size, backend)
    model = get_model()
    model.train()

    rand_inp = torch.randint(0, 10, input_dimensions, dtype=input_dtype)
    start = timeit.default_timer()

    output = model(rand_inp)
    loss = output.sum()

    hook_handles = []

    def hook(p: torch.nn.Parameter):
        if p.grad is None:
            return
        if backend == "gloo":
            # p.grad /= world_size # TODO: extra dumb comp for later
            hook_handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True))
        else:
            hook_handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))

    if optimization_type == OptimizationType.OVERLAPPING:
        for p in model.parameters():
            p.register_post_accumulate_grad_hook(hook)

    loss.backward()
    # print(f"rank={rank}, loss={loss.item()}")

    start_sync = timeit.default_timer()
    if optimization_type == OptimizationType.FLATTEN:
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        flatten = torch._utils._flatten_dense_tensors(grads)
        if backend == "gloo":
            dist.all_reduce(flatten, op=dist.ReduceOp.SUM, async_op=False)
            flatten /= world_size
        else:
            dist.all_reduce(flatten, op=dist.ReduceOp.AVG, async_op=False)

        torch._utils._unflatten_dense_tensors(flatten, grads)

    elif optimization_type == OptimizationType.OVERLAPPING:
        for handle in hook_handles:
            handle.wait()
        hook_handles.clear()

    else:
        for p in model.parameters():
            if p.grad is None:
                continue

            if backend == "gloo":
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=False)
                p.grad /= world_size
            else:
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=False)

    if torch.cuda.is_available() == "cuda":
        torch.cuda.synchronize()

    sync_time = timeit.default_timer() - start_sync
    if rank == 0:
        print("syncing gradients took:", sync_time, "seconds")

    dist.all_reduce(loss, op=dist.ReduceOp.SUM, async_op=False)
    loss /= world_size
    if rank == 0:
        print("avg loss:", loss)
        total = timeit.default_timer() - start
        print("total training time:", total, "sync took", sync_time / total * 100, "%")

    # optimizer step


# model_fn = lambda: ToyModel(16, 4)  # noqa: E731
def model_fn():
    return BasicsTransformerLM(10000, 256, 512, 6, 8, 704, 10000)


# ======= 2.3 Improving Naive DDP ========

# above limitations
# - separate all reduce for each tensor
# - backward pass can be computed + sent in parallel, overlapping, we can start comm while it's computing others

# Deliverable: The measured time per training iteration and time spent communicating gradients
# under distributed data parallel training with a single batched all-reduce call. 1-2 sentences comparing
# the results when batching vs. individually communicating gradients.

# TODO: make multiple passes to get a better idea
# TODO: check sync timing computation correctness when overlapping
# TODO: fix gloo division when overlapping.
# TODO: test in actual GPU's


if __name__ == "__main__":
    # benchmark("gloo", "cpu", 1000, 4)

    world_size = 4
    batch_size = 64

    # inp_dimensions = (batch_size // world_size, 16)

    inp_dimensions = (batch_size // world_size, 256)

    mp.spawn(
        train_ddp,
        args=(
            world_size,
            model_fn,
            batch_size,
            inp_dimensions,
            torch.long,
            OptimizationType.OVERLAPPING,
            "gloo",
        ),
        nprocs=world_size,
        join=True,
    )
