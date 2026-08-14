"""Stage 1 参数重要性状态容器与严格序列化边界。

``ImportanceState`` 只负责坐标同构的状态存放和 wire contract，不负责决定某个
估计器应如何更新这些量。临时 S1/S2 可以被清空并复用；长期量只能通过显式标记
为已完成的 optimizer step 的提交入口写入，避免半步状态被误发布。
"""

from __future__ import annotations

from collections.abc import Mapping
import sys
from pathlib import Path
from typing import Any

import torch

from .errors import CoreContractError, RegistryError, TensorMapError
from .registry import ParameterRegistry
from .tensors import TensorMap


STATE_SCHEMA_VERSION = "stage1-importance-state-v1"
TEMPORARY_SLOTS = ("s1", "s2")
LONG_TERM_SLOTS = (
    "signed",
    "positive",
    "negative_mass",
    "absolute",
    "raw",
    "data_movement",
    "net_data_movement",
    "magnitude",
)
ACTUAL_UPDATE_SLOT = "actual_update_raw_importance"


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _dtype_from_name(value: Any) -> torch.dtype:
    if not isinstance(value, str) or not value:
        raise CoreContractError("state accumulation_dtype 必须是非空字符串")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise CoreContractError(f"不支持的 state accumulation_dtype: {value!r}")
    return dtype


def _clone_mapping(value: TensorMap, *, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().to(device=device).clone()
        for name, tensor in value.items()
    }


class ImportanceState:
    """绑定一个 :class:`ParameterRegistry` 的 FP32/FP64 状态容器。

    默认状态只包含计划中的核心长期命名空间；``include_actual_update=True`` 时
    额外保留实际更新驱动的 raw importance。所有 slot 都必须覆盖完整 eligible
    坐标集，且不会发生隐式广播或自动补零。
    """

    def __init__(
        self,
        registry: ParameterRegistry,
        *,
        accumulation_dtype: torch.dtype = torch.float32,
        include_actual_update: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        if not isinstance(registry, ParameterRegistry):
            raise RegistryError("ImportanceState 必须绑定 ParameterRegistry")
        if not registry.eligible_records:
            raise CoreContractError("ImportanceState 不能绑定空 eligible registry")
        if accumulation_dtype not in {torch.float32, torch.float64}:
            raise CoreContractError("ImportanceState accumulation_dtype 必须为 FP32 或 FP64")
        self.registry = registry
        self.accumulation_dtype = accumulation_dtype
        self.device = self._resolve_device(registry, device)
        self._slot_names = TEMPORARY_SLOTS + LONG_TERM_SLOTS + (
            (ACTUAL_UPDATE_SLOT,) if include_actual_update else ()
        )
        template = {
            record.canonical_name: torch.zeros(
                record.shape,
                dtype=accumulation_dtype,
                device=self.device,
            )
            for record in registry.eligible_records
        }
        self._slots = {
            slot_name: TensorMap(template, registry=registry, clone=True)
            for slot_name in self._slot_names
        }

    @staticmethod
    def _resolve_device(
        registry: ParameterRegistry,
        device: torch.device | str | None,
    ) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            first = registry.parameter(registry.eligible_names[0])
        except RegistryError:
            return torch.device("cpu")
        return first.device

    @property
    def registry_hash(self) -> str:
        return self.registry.coordinate_registry_hash

    @property
    def slot_names(self) -> tuple[str, ...]:
        return self._slot_names

    def slot(self, name: str) -> TensorMap:
        """返回某个 slot 的副本，调用方不能绕过提交边界原地修改状态。"""

        try:
            return self._slots[name].clone()
        except KeyError as exc:
            raise CoreContractError(f"未知或未启用的 state slot: {name!r}") from exc

    def _candidate(
        self,
        slot_name: str,
        values: TensorMap | Mapping[str, torch.Tensor],
    ) -> TensorMap:
        if slot_name not in self._slots:
            raise CoreContractError(f"未知或未启用的 state slot: {slot_name!r}")
        candidate = values if isinstance(values, TensorMap) else TensorMap(
            values,
            registry=self.registry,
        )
        if candidate.registry_hash != self.registry_hash:
            raise TensorMapError("state slot 的 coordinate_registry_hash 不一致")
        for name, tensor in candidate.items():
            if tensor.dtype != self.accumulation_dtype:
                raise CoreContractError(
                    f"state slot {slot_name!r} 的 {name!r} dtype 必须为 "
                    f"{_dtype_name(self.accumulation_dtype)}"
                )
        return TensorMap(
            _clone_mapping(candidate, device=self.device),
            registry=self.registry,
        )

    def set_slot(
        self,
        name: str,
        values: TensorMap | Mapping[str, torch.Tensor],
    ) -> None:
        """严格替换一个 slot；shape、坐标、dtype 或有限性不符时不改变状态。"""

        candidate = self._candidate(name, values)
        for coordinate in self._slots[name]:
            self._slots[name][coordinate].copy_(candidate[coordinate])

    def reset_temporary(self) -> None:
        """清零并复用 S1/S2 工作区，不创建新的坐标结构。"""

        for slot_name in TEMPORARY_SLOTS:
            for tensor in self._slots[slot_name].values():
                tensor.zero_()

    release_temporary = reset_temporary

    def commit_long_term(
        self,
        updates: Mapping[str, TensorMap | Mapping[str, torch.Tensor]],
        *,
        step_completed: bool,
    ) -> None:
        """原子提交一个完整 optimizer step 产生的长期量。

        ``step_completed`` 必须显式为 ``True``；失败时所有 slot 保持原状。
        """

        if step_completed is not True:
            raise CoreContractError("长期状态只能在完整 optimizer step 后提交")
        if not isinstance(updates, Mapping) or not updates:
            raise CoreContractError("commit_long_term 需要非空 slot mapping")
        if any(name in TEMPORARY_SLOTS for name in updates):
            raise CoreContractError("S1/S2 只能通过临时工作区入口写入")
        staged = {
            name: self._candidate(name, values)
            for name, values in updates.items()
        }
        for name, candidate in staged.items():
            for coordinate in self._slots[name]:
                self._slots[name][coordinate].copy_(candidate[coordinate])

    def schema_manifest(self) -> dict[str, Any]:
        """返回 metadata-only state schema，便于报告和下游预检。"""

        fields = [
            {
                "canonical_name": record.canonical_name,
                "shape": list(record.shape),
                "numel": record.numel,
            }
            for record in self.registry.eligible_records
        ]
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "registry_hash": self.registry_hash,
            "accumulation_dtype": _dtype_name(self.accumulation_dtype),
            "byte_order": sys.byteorder,
            "slot_names": list(self.slot_names),
            "slots": {
                name: {
                    "dtype": _dtype_name(self.accumulation_dtype),
                    "fields": fields,
                }
                for name in self.slot_names
            },
        }

    def state_dict(self) -> dict[str, Any]:
        """返回 raw tensor bundle 可编码的严格状态树。"""

        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "registry_hash": self.registry_hash,
            "accumulation_dtype": _dtype_name(self.accumulation_dtype),
            "slot_names": list(self.slot_names),
            "slots": {
                slot_name: self._slots[slot_name].to_dict(clone=True)
                for slot_name in self.slot_names
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """严格恢复状态；全部校验通过前不触碰当前内存状态。"""

        required = {
            "schema_version",
            "registry_hash",
            "accumulation_dtype",
            "slot_names",
            "slots",
        }
        if not isinstance(state, Mapping) or set(state) != required:
            raise CoreContractError("ImportanceState state_dict 字段集合无效")
        if state["schema_version"] != STATE_SCHEMA_VERSION:
            raise CoreContractError("ImportanceState schema_version 不受支持")
        if state["registry_hash"] != self.registry_hash:
            raise CoreContractError("ImportanceState registry_hash 不一致")
        if state["accumulation_dtype"] != _dtype_name(self.accumulation_dtype):
            raise CoreContractError("ImportanceState accumulation_dtype 不一致")
        slot_names = state["slot_names"]
        if slot_names != list(self.slot_names):
            raise CoreContractError("ImportanceState slot_names 不一致")
        slots = state["slots"]
        if not isinstance(slots, Mapping) or set(slots) != set(self.slot_names):
            raise CoreContractError("ImportanceState slots 集合不一致")
        staged = {
            name: self._candidate(name, values)
            for name, values in slots.items()
        }
        for name, candidate in staged.items():
            for coordinate in self._slots[name]:
                self._slots[name][coordinate].copy_(candidate[coordinate])

    def save_bundle(self, path: str | Path) -> Any:
        """使用既有 raw-tensor-bundle codec 发布不可覆盖状态对象。"""

        from ..runtime.tensor_bundle import publish_tensor_bundle

        return publish_tensor_bundle(path, self.state_dict())

    @classmethod
    def load_bundle(
        cls,
        path: str | Path,
        registry: ParameterRegistry,
        *,
        device: torch.device | str | None = None,
    ) -> tuple["ImportanceState", Any]:
        """先验证 bundle，再按 manifest schema 构造并恢复状态。"""

        from ..runtime.tensor_bundle import load_tensor_bundle

        state_dict, identity = load_tensor_bundle(path)
        if not isinstance(state_dict, Mapping):
            raise CoreContractError("ImportanceState bundle 顶层必须是 object")
        dtype = _dtype_from_name(state_dict.get("accumulation_dtype"))
        slot_names = state_dict.get("slot_names")
        if not isinstance(slot_names, list):
            raise CoreContractError("ImportanceState bundle.slot_names 必须是数组")
        state = cls(
            registry,
            accumulation_dtype=dtype,
            include_actual_update=ACTUAL_UPDATE_SLOT in slot_names,
            device=device,
        )
        state.load_state_dict(state_dict)
        return state, identity


__all__ = [
    "ACTUAL_UPDATE_SLOT",
    "ImportanceState",
    "LONG_TERM_SLOTS",
    "STATE_SCHEMA_VERSION",
    "TEMPORARY_SLOTS",
]
