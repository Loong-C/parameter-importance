"""与参数注册表同构的有序张量容器。"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any

import torch

from .errors import NumericalError, TensorMapError
from .registry import ParameterRegistry


class TensorMap(Mapping[str, torch.Tensor]):
    """保存一组无隐式广播、顺序稳定的参数张量。

    构造函数同时兼容 ``TensorMap(values, registry=registry)`` 与历史友好的
    ``TensorMap(registry, values)``。有注册表时只接收 eligible canonical names，
    alias 输入会先归一化；无注册表时保留传入 mapping 的迭代顺序，适合低维解析
    fixture。所有算术操作都会先验证名称和 shape 完全一致。
    """

    def __init__(
        self,
        values_or_registry: Mapping[str, torch.Tensor] | ParameterRegistry,
        values: Mapping[str, torch.Tensor] | None = None,
        *,
        registry: ParameterRegistry | None = None,
        clone: bool = False,
        require_finite: bool = True,
    ) -> None:
        if isinstance(values_or_registry, ParameterRegistry):
            if registry is not None:
                raise TensorMapError("registry 不能同时以位置参数和关键字参数传入")
            registry = values_or_registry
            if values is None:
                raise TensorMapError("TensorMap(registry, values) 缺少 values")
            source = values
        else:
            if values is not None:
                raise TensorMapError("无 registry 的 TensorMap 只接受一个 mapping 位置参数")
            source = values_or_registry
        if not source:
            raise TensorMapError("TensorMap 不能为空")

        normalized: dict[str, torch.Tensor] = {}
        if registry is not None:
            for raw_name, tensor in source.items():
                canonical = registry.canonical_name(raw_name)
                record = registry.record(canonical)
                if not record.eligible:
                    raise TensorMapError(f"参数 {canonical!r} 不在 eligible set")
                if canonical in normalized:
                    raise TensorMapError(f"alias 导致参数 {canonical!r} 被重复提供")
                normalized[canonical] = tensor
            expected = registry.eligible_names
            actual = tuple(normalized)
            if set(actual) != set(expected):
                missing = sorted(set(expected) - set(actual))
                extra = sorted(set(actual) - set(expected))
                raise TensorMapError(f"TensorMap 坐标不完整: missing={missing}, extra={extra}")
            ordered = {name: normalized[name] for name in expected}
        else:
            ordered = dict(source)

        for name, tensor in ordered.items():
            if not isinstance(name, str) or not name:
                raise TensorMapError("TensorMap 名称必须是非空字符串")
            if not isinstance(tensor, torch.Tensor):
                raise TensorMapError(f"{name!r} 的值不是 torch.Tensor")
            if tensor.layout is not torch.strided:
                raise TensorMapError(f"{name!r} 必须是稠密 strided 张量")
            if registry is not None:
                registry.validate_gradient(name, tensor)
            if require_finite and not bool(torch.isfinite(tensor).all()):
                raise NumericalError(f"TensorMap[{name!r}] 含 NaN/Inf")

        self.registry = registry
        self._values = MappingProxyType(
            {name: tensor.detach().clone() if clone else tensor for name, tensor in ordered.items()}
        )
        self._shapes = {name: tuple(tensor.shape) for name, tensor in self._values.items()}

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, torch.Tensor],
        *,
        registry: ParameterRegistry | None = None,
        clone: bool = False,
        require_finite: bool = True,
    ) -> "TensorMap":
        return cls(values, registry=registry, clone=clone, require_finite=require_finite)

    @classmethod
    def zeros_like(
        cls,
        other: "TensorMap",
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "TensorMap":
        return cls(
            {
                name: torch.zeros_like(tensor, dtype=dtype or tensor.dtype, device=device or tensor.device)
                for name, tensor in other.items()
            },
            registry=other.registry,
        )

    def __getitem__(self, name: str) -> torch.Tensor:
        if self.registry is not None:
            name = self.registry.canonical_name(name)
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    @property
    def registry_hash(self) -> str | None:
        return None if self.registry is None else self.registry.coordinate_registry_hash

    def clone(self, *, detach: bool = True) -> "TensorMap":
        values: dict[str, torch.Tensor] = {}
        for name, tensor in self.items():
            cloned = tensor.detach().clone() if detach else tensor.clone()
            # 路径端点通常来自 requires_grad Parameter。detach 的目的只是切断旧图，
            # 不是静默冻结新路径节点，因此在可微 dtype 上恢复原 requires_grad 标志。
            if detach and (cloned.is_floating_point() or cloned.is_complex()):
                cloned.requires_grad_(tensor.requires_grad)
            values[name] = cloned
        return TensorMap(values, registry=self.registry, require_finite=False)

    def to(
        self,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "TensorMap":
        """执行显式 dtype/device 转换；调用方必须自行记录转换语义。"""

        return TensorMap(
            {name: value.to(dtype=dtype, device=device) for name, value in self.items()},
            registry=self.registry,
            require_finite=False,
        )

    def map(self, function: Callable[[torch.Tensor], torch.Tensor]) -> "TensorMap":
        return TensorMap(
            {name: function(value) for name, value in self.items()},
            registry=self.registry,
            require_finite=False,
        )

    def zip_map(
        self,
        other: "TensorMap",
        function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> "TensorMap":
        self.assert_compatible(other, require_dtype_device=True)
        return TensorMap(
            {name: function(self[name], other[name]) for name in self},
            registry=self.registry or other.registry,
            require_finite=False,
        )

    def assert_compatible(self, other: "TensorMap", *, require_dtype_device: bool = False) -> None:
        if tuple(self) != tuple(other):
            raise TensorMapError(f"TensorMap 名称或顺序不一致: {tuple(self)} != {tuple(other)}")
        if (self.registry_hash is None) != (other.registry_hash is None):
            raise TensorMapError("一个 TensorMap 已绑定 registry、另一个未绑定，不能混算")
        if self.registry_hash and self.registry_hash != other.registry_hash:
            raise TensorMapError("TensorMap coordinate_registry_hash 不一致")
        for name in self:
            if self._shapes[name] != other._shapes[name]:
                raise TensorMapError(f"TensorMap[{name!r}] shape 不一致")
            if require_dtype_device and (
                self[name].dtype != other[name].dtype or self[name].device != other[name].device
            ):
                raise TensorMapError(
                    f"TensorMap[{name!r}] 二元运算前必须显式统一 dtype/device"
                )

    def assert_finite(self) -> None:
        for name, value in self.items():
            if not bool(torch.isfinite(value).all()):
                raise NumericalError(f"TensorMap[{name!r}] 含 NaN/Inf")

    def total_numel(self) -> int:
        return sum(value.numel() for value in self.values())

    def scalar_sum(self, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
        parts = [value.to(dtype=dtype).sum() for value in self.values()]
        return torch.stack(parts).sum()

    def flatten(self, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        values = [value.reshape(-1).to(dtype=dtype or value.dtype) for value in self.values()]
        if not values:
            raise TensorMapError("空 TensorMap 无法展平")
        common = values[0].dtype
        if any(value.dtype != common for value in values):
            raise TensorMapError("混合 dtype 展平必须显式指定 dtype")
        return torch.cat(values)

    def to_dict(self, *, clone: bool = False) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().clone() if clone else value
            for name, value in self.items()
        }

    def __add__(self, other: "TensorMap") -> "TensorMap":
        return self.zip_map(other, torch.add)

    def __sub__(self, other: "TensorMap") -> "TensorMap":
        return self.zip_map(other, torch.sub)

    def __mul__(self, scalar_or_map: float | int | torch.Tensor | "TensorMap") -> "TensorMap":
        if isinstance(scalar_or_map, TensorMap):
            return self.zip_map(scalar_or_map, torch.mul)
        return self.map(lambda value: value * scalar_or_map)

    def __rmul__(self, scalar_or_map: float | int | torch.Tensor | "TensorMap") -> "TensorMap":
        return self.__mul__(scalar_or_map)

    def __truediv__(self, scalar: float | int | torch.Tensor) -> "TensorMap":
        return self.map(lambda value: value / scalar)

    def __neg__(self) -> "TensorMap":
        return self.map(torch.neg)
