import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
from cs336_basics.model import BasicsTransformerLM
import enum  # noqa: E402
from src.distributed.basics import setup
import torch.cuda.nvtx as nvtx
import math


batch_size = 16
seq_length = 512


def get_model():
    # XL dimensions
    vocab_size, d_model, num_layers, num_heads, dff, rope_t = 10000, 1600, 48, 25, 6400, 10000
    return BasicsTransformerLM(vocab_size, seq_length, d_model, num_layers, num_heads, dff, rope_t)



def train_zero1(rank: int, world_size: int):
    setup(rank, world_size, "nccl")
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else f"cpu:{rank}")

    model = get_model()
    model = model.to(device)
    model.train()

    # ==== broadcast sync model weights ====
    for param in model.parameters():
        dist.broadcast(param.data, src=0)
    
    # ==== warmup ====
    local_batch_size = batch_size // world_size
    rand_inp = torch.randint(0, 10, (local_batch_size, seq_length), dtype=torch.long, device=device)
    for _ in range(10):  # warmup
        output = model(rand_inp)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    start = timeit.default_timer()
        
    # ==== setup overlapping comm hooks ====
    hook_handles = []
    def hook(p: torch.nn.Parameter):
        if p.grad is not None:
            hook_handles.append(dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, async_op=True))
    [p.register_post_accumulate_grad_hook(hook) for p in model.parameters()]
    
    # ==== create optimizer sharded ====
    params = list(model.parameters())
    params_per_rank = math.ceil(len(params) / world_size)
    rank_params = params[rank * params_per_rank : min((rank + 1) * params_per_rank, len(params))]
    optimizer = torch.optim.AdamW(rank_params)

    # ==== forward/backward comp ====
    with nvtx.range(f"forward/backward comp"):
        output = model(rand_inp)
        loss = output.sum()
        loss.backward()
    
    # ==== wait for gradients sync ====
    [handle.wait() for handle in hook_handles]
    hook_handles.clear()
    
    # ==== optimizer step + sync model weights ====
    with nvtx.range(f"Optimizer step"):
        optimizer.step()
    
    for i, param in enumerate(params): 
        # suboptimal ofc, can be better (flattened), just world_size comms.
        param_rank = i // params_per_rank
        dist.broadcast(param.data, src=param_rank)
    
    # ==== final timer logs ====
    torch.cuda.synchronize()
    total = timeit.default_timer() - start

    print(list(model.parameters())[0].sum()) # verification (for all ranks should be same)
    
    if rank == 0:
        print(f"total training time: {total:.2f}")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    with nvtx.range(f"ZERO Stage 1"):
        mp.spawn(train_zero1, args=(world_size,), nprocs=world_size, join=True)
