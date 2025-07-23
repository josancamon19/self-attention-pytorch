import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
from cs336_basics.model import BasicsTransformerLM
import enum  # noqa: E402
from src.distributed.basics import setup, ToyModel
import torch.cuda.nvtx as nvtx


class OptimizationType(enum.Enum):
    NONE = "none"
    FLATTEN = "flatten"
    OVERLAPPING = "overlapping"
    OVERLAPPING_BUCKETED = "bucketed"


def get_model(seq_length: int = 512):
    # XL dimensions
    vocab_size, seq_length, d_model, num_layers, num_heads, dff, rope_t = 10000, 512, 1600, 48, 25, 6400, 10000
    return BasicsTransformerLM(vocab_size, seq_length, d_model, num_layers, num_heads, dff, rope_t)


def get_toy_model():
    return ToyModel(16, 4)


def train_ddp(
    rank: int,
    world_size: int,
    get_model: callable,
    input_dimensions: tuple,
    input_dtype: torch.dtype,
    optimization_type: OptimizationType = OptimizationType.NONE,
    backend: str = "nccl",
    bucket_size_mb=1,
):
    setup(rank, world_size, backend)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else f"cpu:{rank}")

    model = get_model()
    model = model.to(device)
    model.train()

    # ==== broadcast sync model weights ====
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # ==== warmup ====
    rand_inp = torch.randint(0, 10, input_dimensions, dtype=input_dtype, device=device)
    for _ in range(10):  # warmup
        output = model(rand_inp)
        loss = output.sum()
        loss.backward()
        model.zero_grad()

    torch.cuda.synchronize()
    start = timeit.default_timer()

    # ==== setup overlapping buckets ====
    buckets = []

    if optimization_type == OptimizationType.OVERLAPPING_BUCKETED:
        curr_bucket = []
        curr_size_bytes = 0
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        params = list(reversed(model.parameters()))
        for param in params:
            param_size = param.numel() * param.element_size()  # Not hard-coded 4

            # Check if adding this param would exceed limit
            if curr_bucket and curr_size_bytes + param_size > bucket_size_bytes:
                buckets.append(curr_bucket)
                curr_bucket = [param]
                curr_size_bytes = param_size
            else:
                curr_bucket.append(param)
                curr_size_bytes += param_size

        if curr_bucket:
            buckets.append(curr_bucket)

    param_to_bucket = {}
    for bucket_idx, bucket in enumerate(buckets):
        for param in bucket:
            param_to_bucket[param] = bucket_idx

    # ==== setup overlapping comm hooks ====
    hook_handles = []

    buckets_items_ready = {i: 0 for i in range(len(buckets))}

    def hook(p: torch.nn.Parameter):
        if p.grad is not None:
            if optimization_type == OptimizationType.OVERLAPPING:
                hook_handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))

            # ==== overlapping buckets specifics ====
            elif optimization_type == OptimizationType.OVERLAPPING_BUCKETED:
                bucket_idx = param_to_bucket[p]
                buckets_items_ready[bucket_idx] += 1
                if buckets_items_ready[bucket_idx] == len(buckets[bucket_idx]):
                    bucket_grads = [p.grad for p in buckets[bucket_idx]]
                    flatten = torch._utils._flatten_dense_tensors(bucket_grads)
                    hook_handles.append(dist.all_reduce(flatten, op=dist.ReduceOp.AVG, async_op=True))
                    # TODO: store bucket_grads as well so you can reassign later

    if optimization_type in [OptimizationType.OVERLAPPING, OptimizationType.OVERLAPPING_BUCKETED]:
        [p.register_post_accumulate_grad_hook(hook) for p in model.parameters()]

    # ==== model(x) + loss compute ====

    output = model(rand_inp)
    loss = output.sum()
    loss.backward()

    # ==== communication by optimization chosen ====

    if optimization_type == OptimizationType.FLATTEN:
        params_with_grads = [p for p in model.parameters() if p.grad is not None]
        grads = [p.grad for p in params_with_grads]

        flatten = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flatten, op=dist.ReduceOp.AVG, async_op=False)
        unflattened_grads = torch._utils._unflatten_dense_tensors(flatten, grads)

        for param, new_grad in zip(params_with_grads, unflattened_grads):
            param.grad = new_grad

    elif optimization_type == OptimizationType.OVERLAPPING:
        [handle.wait() for handle in hook_handles]
        hook_handles.clear()

    elif optimization_type == OptimizationType.OVERLAPPING_BUCKETED:
        [handle.wait() for handle in hook_handles]
        hook_handles.clear()
        # TODO: take bucketed results, and reassign

    else:
        [
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=False)
            for p in model.parameters()
            if p.grad is not None
        ]

    # ==== finish benchmarking ====

    torch.cuda.synchronize()
    total = timeit.default_timer() - start

    if rank == 0:
        print(f"total training time: {total:.2f}")

    # print(list(model.parameters())[0].grad.sum()) # verification (both should be same)
    dist.destroy_process_group()


# TODO: profile with nsight systems


if __name__ == "__main__":
    gpu_count = torch.cuda.device_count()
    for optimization_type in OptimizationType:
        print(optimization_type)
        with nvtx.range(f"Optimization: {optimization_type.value}"):
            mp.spawn(
                train_ddp,
                args=(gpu_count, get_model, (16 // gpu_count, 512), torch.long, optimization_type),
                nprocs=gpu_count,
                join=True,
            )
