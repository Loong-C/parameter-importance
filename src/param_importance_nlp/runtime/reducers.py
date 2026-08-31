"""充分统计量归约接口。

本机 ``LocalReducer`` 与 Gloo 测试只验证归约代数；它们不能把 CUDA/NCCL Gate
标成通过。``TorchDistributedReducer`` 不硬编码 backend、world size 或设备编号，
这些身份必须由已解析配置和运行时进程组提供。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, TypeVar


TensorT = TypeVar("TensorT")


@dataclass(frozen=True, slots=True)
class ReducerCapabilities:
    """运行时真实归约能力；它不是 formal Gate 结论。"""

    backend: str
    world_size: int
    device_type: str


class Reducer(Protocol[TensorT]):
    def sum_tensors(self, values: Mapping[str, TensorT]) -> dict[str, TensorT]: ...

    def sum_int(self, value: int) -> int: ...


class LocalReducer:
    """world size 为 1 的无副作用归约器。"""

    capabilities = ReducerCapabilities("local", 1, "cpu")

    def sum_tensors(self, values: Mapping[str, TensorT]) -> dict[str, TensorT]:
        return {name: value for name, value in values.items()}

    def sum_int(self, value: int) -> int:
        return int(value)


class TorchDistributedReducer:
    """对已初始化 Torch 进程组执行 sum all-reduce。"""

    def __init__(
        self,
        process_group: object | None = None,
        *,
        integer_device: object | None = None,
    ) -> None:
        import torch
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("DISTRIBUTED_PROCESS_GROUP_NOT_INITIALIZED")
        self.process_group = process_group
        self.backend = str(dist.get_backend(process_group))
        self.world_size = int(dist.get_world_size(process_group))
        if self.backend == "gloo":
            self.integer_device = torch.device("cpu")
        elif self.backend == "nccl":
            if integer_device is None:
                raise ValueError("NCCL_INTEGER_DEVICE_REQUIRED")
            self.integer_device = torch.device(integer_device)
            if self.integer_device.type != "cuda":
                raise ValueError("NCCL_REQUIRES_CUDA_DEVICE")
        else:
            raise ValueError(f"DISTRIBUTED_BACKEND_UNSUPPORTED:{self.backend}")
        self.capabilities = ReducerCapabilities(
            self.backend, self.world_size, self.integer_device.type
        )

    def sum_tensors(self, values: Mapping[str, TensorT]) -> dict[str, TensorT]:
        import torch.distributed as dist

        result: dict[str, TensorT] = {}
        for name in sorted(values):
            tensor = values[name]
            if self.backend == "gloo" and tensor.device.type != "cpu":  # type: ignore[attr-defined]
                raise ValueError("GLOO_LOCAL_PROFILE_REQUIRES_CPU_TENSOR")
            if self.backend == "nccl" and tensor.device.type != "cuda":  # type: ignore[attr-defined]
                raise ValueError("NCCL_REQUIRES_CUDA_TENSOR")
            reduced = tensor.clone()  # type: ignore[attr-defined]
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=self.process_group)
            result[name] = reduced
        return result

    def sum_int(self, value: int) -> int:
        import torch
        import torch.distributed as dist

        tensor = torch.tensor([value], dtype=torch.int64, device=self.integer_device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=self.process_group)
        return int(tensor.item())


__all__ = [
    "LocalReducer",
    "Reducer",
    "ReducerCapabilities",
    "TorchDistributedReducer",
]
