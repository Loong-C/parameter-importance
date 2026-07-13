#!/usr/bin/env python3
import os
import torch
import torch.distributed as dist

dist.init_process_group("nccl")
rank = dist.get_rank()
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
x = torch.tensor([float(rank + 1)], device=f"cuda:{local_rank}", dtype=torch.bfloat16)
dist.all_reduce(x)
expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
if x.item() != expected:
    raise RuntimeError(f"rank {rank}: all_reduce={x.item()}, expected={expected}")
if rank == 0:
    print(f"NCCL_OK world_size={dist.get_world_size()} sum={x.item()}")
dist.destroy_process_group()
