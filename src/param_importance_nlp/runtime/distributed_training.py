"""Torch DDP 进程组、设备与训练 reducer 的显式生命周期。

本模块不负责启动子进程；正式命令由 ``torchrun`` 设置 ``RANK/WORLD_SIZE`` 等
环境变量，每个进程调用 :meth:`TorchDDPTrainingExecutor.from_environment`。
模型仍包装为 ``DistributedDataParallel`` 以获得参数初始广播和标准故障边界，
但训练引擎把 microbatch forward/backward 放在 ``no_sync`` 中，再用
``TorchDistributedReducer`` 对 global mean 和重要性充分量各归约一次。
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import torch

from .reducers import TorchDistributedReducer


@dataclass(frozen=True, slots=True)
class DistributedLaunchSpec:
    """一个由 launcher 冻结的 rank 身份。"""

    backend: str
    rank: int
    local_rank: int
    world_size: int
    init_method: str = "env://"

    def __post_init__(self) -> None:
        if self.backend not in {"gloo", "nccl"}:
            raise ValueError("DISTRIBUTED_BACKEND_UNSUPPORTED")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("DISTRIBUTED_RANK_WORLD_SIZE_INVALID")
        if self.local_rank < 0 or not self.init_method:
            raise ValueError("DISTRIBUTED_LOCAL_RANK_OR_INIT_METHOD_INVALID")


class TorchDDPTrainingExecutor:
    """拥有 process group 的上下文；退出时只销毁由自己初始化的 group。"""

    def __init__(self, spec: DistributedLaunchSpec) -> None:
        import torch.distributed as dist

        self.spec = spec
        if not dist.is_available():
            raise RuntimeError("DISTRIBUTED_TORCH_NOT_AVAILABLE")
        if spec.backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL_REQUIRES_CUDA")
            if spec.local_rank >= torch.cuda.device_count():
                raise ValueError("DISTRIBUTED_LOCAL_RANK_OUT_OF_RANGE")
            torch.cuda.set_device(spec.local_rank)
            self.device = torch.device("cuda", spec.local_rank)
        else:
            self.device = torch.device("cpu")
        self._owns_group = not dist.is_initialized()
        if self._owns_group:
            dist.init_process_group(
                backend=spec.backend,
                init_method=spec.init_method,
                rank=spec.rank,
                world_size=spec.world_size,
            )
        else:
            if str(dist.get_backend()) != spec.backend:
                raise ValueError("DISTRIBUTED_EXISTING_BACKEND_MISMATCH")
            if int(dist.get_rank()) != spec.rank or int(dist.get_world_size()) != spec.world_size:
                raise ValueError("DISTRIBUTED_EXISTING_IDENTITY_MISMATCH")
        self.reducer = TorchDistributedReducer(integer_device=self.device)

    @classmethod
    def from_environment(cls, *, backend: str) -> "TorchDDPTrainingExecutor":
        """严格读取 torchrun 环境；缺字段时不猜测单进程成功。"""

        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(f"TORCHRUN_ENVIRONMENT_MISSING:{','.join(missing)}")
        try:
            spec = DistributedLaunchSpec(
                backend=backend,
                rank=int(os.environ["RANK"]),
                local_rank=int(os.environ["LOCAL_RANK"]),
                world_size=int(os.environ["WORLD_SIZE"]),
            )
        except ValueError as exc:
            raise ValueError("TORCHRUN_ENVIRONMENT_INVALID") from exc
        return cls(spec)

    def wrap_model(self, module: torch.nn.Module, **kwargs: Any) -> torch.nn.Module:
        """将模型搬运到 rank device 并包装 DDP；optimizer 必须在此后构造。"""

        from torch.nn.parallel import DistributedDataParallel

        module = module.to(self.device)
        if self.spec.backend == "nccl":
            return DistributedDataParallel(
                module,
                device_ids=[self.spec.local_rank],
                output_device=self.spec.local_rank,
                **kwargs,
            )
        return DistributedDataParallel(module, **kwargs)

    def barrier(self) -> None:
        import torch.distributed as dist

        dist.barrier()

    def close(self) -> None:
        import torch.distributed as dist

        if self._owns_group and dist.is_initialized():
            dist.destroy_process_group()

    def __enter__(self) -> "TorchDDPTrainingExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = ["DistributedLaunchSpec", "TorchDDPTrainingExecutor"]
