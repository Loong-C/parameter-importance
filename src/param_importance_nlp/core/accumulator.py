"""训练轨迹重要性的长期累计器。

累计器将单步 signed score 分解为非负的 positive/negative mass。公开 ``signed``
与 ``absolute`` 由这两个基础量派生，而不是独立浮点累计，因此每个坐标始终满足：

``signed == positive - negative_mass``
``absolute == positive + negative_mass``

movement 也分清“逐步绝对路程”和“首尾净位移”：data movement 是数据驱动更新
绝对值之和，net data movement 是数据更新有符号和的绝对值，total endpoint
movement 是实际完整更新（可含 weight decay）有符号和的绝对值。
"""

from __future__ import annotations

from typing import Mapping

import torch

from .errors import CoreContractError
from .tensors import TensorMap


def _copy_into(destination: TensorMap, source: TensorMap) -> None:
    destination.assert_compatible(source)
    for name in destination:
        destination[name].copy_(source[name].to(dtype=destination[name].dtype))


class ImportanceAccumulator:
    """与一个冻结坐标集合同形的 FP32/FP64 长期统计状态。"""

    def __init__(
        self,
        template: TensorMap,
        *,
        accumulation_dtype: torch.dtype = torch.float32,
    ) -> None:
        if accumulation_dtype not in {torch.float32, torch.float64}:
            raise CoreContractError("长期累计 dtype 必须为 FP32 或 FP64")
        self.accumulation_dtype = accumulation_dtype
        self._positive = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._negative = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._raw = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._data_movement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._data_displacement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._total_movement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._total_displacement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._magnitude = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self.successful_steps = 0
        self.skipped_steps = 0

    @property
    def positive(self) -> TensorMap:
        return self._positive.clone()

    @property
    def negative_mass(self) -> TensorMap:
        return self._negative.clone()

    @property
    def signed(self) -> TensorMap:
        return self._positive - self._negative

    @property
    def absolute(self) -> TensorMap:
        return self._positive + self._negative

    @property
    def raw(self) -> TensorMap:
        return self._raw.clone()

    @property
    def data_movement(self) -> TensorMap:
        return self._data_movement.clone()

    @property
    def net_data_movement(self) -> TensorMap:
        return self._data_displacement.map(torch.abs)

    @property
    def total_movement(self) -> TensorMap:
        return self._total_movement.clone()

    @property
    def total_endpoint_movement(self) -> TensorMap:
        return self._total_displacement.map(torch.abs)

    @property
    def magnitude(self) -> TensorMap:
        return self._magnitude.clone()

    def add_step(
        self,
        contribution: TensorMap,
        *,
        raw: TensorMap | None = None,
        data_update: TensorMap | None = None,
        total_update: TensorMap | None = None,
        current_parameters: TensorMap | None = None,
    ) -> None:
        """原子式提交一个已成功 optimizer step 的长期统计。

        方法先校验全部候选输入，再原地提交；任一输入非有限或坐标不一致时不会
        产生部分累计。skip step 应调用 :meth:`record_skip`，不得传入零贡献伪装。
        """

        candidates = [contribution]
        candidates.extend(
            value
            for value in (raw, data_update, total_update, current_parameters)
            if value is not None
        )
        for candidate in candidates:
            self._positive.assert_compatible(candidate)
            candidate.assert_finite()
        converted = contribution.to(dtype=self.accumulation_dtype)
        converted_raw = None if raw is None else raw.to(dtype=self.accumulation_dtype)
        converted_data = None if data_update is None else data_update.to(dtype=self.accumulation_dtype)
        converted_total = None if total_update is None else total_update.to(dtype=self.accumulation_dtype)
        converted_parameters = (
            None if current_parameters is None else current_parameters.to(dtype=self.accumulation_dtype)
        )
        converted_candidates = [converted]
        converted_candidates.extend(
            value
            for value in (converted_raw, converted_data, converted_total, converted_parameters)
            if value is not None
        )
        for candidate in converted_candidates:
            candidate.assert_finite()
        if converted_raw is not None:
            for name, value in converted_raw.items():
                if bool((value < 0).any()):
                    raise CoreContractError(f"raw importance {name!r} 不得为负")

        for name, value in converted.items():
            self._positive[name].add_(value.clamp_min(0))
            self._negative[name].add_((-value).clamp_min(0))
        if converted_raw is not None:
            for name, value in converted_raw.items():
                self._raw[name].add_(value)
        if converted_data is not None:
            for name, value in converted_data.items():
                self._data_movement[name].add_(value.abs())
                self._data_displacement[name].add_(value)
        if converted_total is not None:
            for name, value in converted_total.items():
                self._total_movement[name].add_(value.abs())
                self._total_displacement[name].add_(value)
        if converted_parameters is not None:
            for name, value in converted_parameters.items():
                self._magnitude[name].copy_(value.abs())
        self.successful_steps += 1

    def set_magnitude(self, parameters: TensorMap) -> None:
        """用当前参数绝对值刷新静态 magnitude 基线。"""

        self._magnitude.assert_compatible(parameters)
        parameters.assert_finite()
        converted = parameters.to(dtype=self.accumulation_dtype)
        converted.assert_finite()
        for name, value in converted.items():
            self._magnitude[name].copy_(value.abs())

    def record_skip(self) -> None:
        """记录一次未执行 optimizer update 的 attempt，不改变任何张量状态。"""

        self.skipped_steps += 1

    def validate_invariants(self, *, atol: float = 0.0, rtol: float = 0.0) -> None:
        """逐坐标检查四视图恒等式和所有非负质量。"""

        signed = self.signed
        absolute = self.absolute
        for name in self._positive:
            if bool((self._positive[name] < 0).any()) or bool((self._negative[name] < 0).any()):
                raise CoreContractError(f"{name!r} 的 positive/negative mass 出现负值")
            if not torch.allclose(
                signed[name], self._positive[name] - self._negative[name], atol=atol, rtol=rtol
            ):
                raise CoreContractError(f"{name!r} 违反 signed 恒等式")
            if not torch.allclose(
                absolute[name], self._positive[name] + self._negative[name], atol=atol, rtol=rtol
            ):
                raise CoreContractError(f"{name!r} 违反 absolute 恒等式")

    def state_dict(self) -> dict[str, object]:
        """返回只含 primitive/TensorMap 的安全状态，不使用 pickle 对象图。"""

        return {
            "version": 1,
            "accumulation_dtype": str(self.accumulation_dtype),
            "successful_steps": self.successful_steps,
            "skipped_steps": self.skipped_steps,
            "positive": self._positive.to_dict(clone=True),
            "negative_mass": self._negative.to_dict(clone=True),
            "raw": self._raw.to_dict(clone=True),
            "data_movement": self._data_movement.to_dict(clone=True),
            "data_displacement": self._data_displacement.to_dict(clone=True),
            "total_movement": self._total_movement.to_dict(clone=True),
            "total_displacement": self._total_displacement.to_dict(clone=True),
            "magnitude": self._magnitude.to_dict(clone=True),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """严格恢复累计状态；缺字段、版本漂移或坐标不符均失败。"""

        if state.get("version") != 1:
            raise CoreContractError("不支持的 ImportanceAccumulator state version")
        expected_dtype = str(self.accumulation_dtype)
        if state.get("accumulation_dtype") != expected_dtype:
            raise CoreContractError("累计状态 dtype 与当前合同不一致")
        destinations = {
            "positive": self._positive,
            "negative_mass": self._negative,
            "raw": self._raw,
            "data_movement": self._data_movement,
            "data_displacement": self._data_displacement,
            "total_movement": self._total_movement,
            "total_displacement": self._total_displacement,
            "magnitude": self._magnitude,
        }
        staged: dict[str, TensorMap] = {}
        for key, destination in destinations.items():
            raw_mapping = state.get(key)
            if not isinstance(raw_mapping, Mapping):
                raise CoreContractError(f"累计状态缺少张量 mapping: {key}")
            staged[key] = TensorMap(raw_mapping, registry=destination.registry)
            destination.assert_compatible(staged[key])
            for name, value in staged[key].items():
                if value.dtype != self.accumulation_dtype:
                    raise CoreContractError(
                        f"累计状态 {key}.{name} dtype 与 accumulation dtype 不一致"
                    )
        successful_steps = state.get("successful_steps")
        skipped_steps = state.get("skipped_steps")
        if not isinstance(successful_steps, int) or successful_steps < 0:
            raise CoreContractError("successful_steps 非法")
        if not isinstance(skipped_steps, int) or skipped_steps < 0:
            raise CoreContractError("skipped_steps 非法")
        for key, destination in destinations.items():
            _copy_into(destination, staged[key])
        self.successful_steps = successful_steps
        self.skipped_steps = skipped_steps
        self.validate_invariants()
