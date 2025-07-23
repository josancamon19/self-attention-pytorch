import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit


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
