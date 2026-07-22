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
        self._raw_clipped = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._data_movement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._data_displacement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._total_movement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._total_displacement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._weight_decay_movement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._weight_decay_displacement = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._magnitude = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._initial_parameters = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._last_parameters = TensorMap.zeros_like(template, dtype=accumulation_dtype)
        self._has_initial_parameters = False
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
    def raw_clipped(self) -> TensorMap:
        """同批 clip factor 后的 plug-in raw 在线分数。"""

        return self._raw_clipped.clone()

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
    def weight_decay_movement(self) -> TensorMap:
        return self._weight_decay_movement.clone()

    @property
    def net_weight_decay_movement(self) -> TensorMap:
        return self._weight_decay_displacement.map(torch.abs)

    @property
    def magnitude(self) -> TensorMap:
        return self._magnitude.clone()

    @property
    def attempted_steps(self) -> int:
        """成功 commit 与明确 skip 的 attempt 总数。"""

        return self.successful_steps + self.skipped_steps

    def set_initial_parameters(self, parameters: TensorMap) -> None:
        """冻结训练起点；只允许在第一个成功 step 之前设置一次。"""

        if self.successful_steps or self._has_initial_parameters:
            raise CoreContractError("初始参数端点只能在训练开始前设置一次")
        self._initial_parameters.assert_compatible(parameters)
        parameters.assert_finite()
        converted = parameters.to(dtype=self.accumulation_dtype)
        _copy_into(self._initial_parameters, converted)
        _copy_into(self._last_parameters, converted)
        self._has_initial_parameters = True
        self.set_magnitude(parameters)

    def add_step(
        self,
        contribution: TensorMap,
        *,
        raw: TensorMap | None = None,
        raw_clipped: TensorMap | None = None,
        data_update: TensorMap | None = None,
        total_update: TensorMap | None = None,
        weight_decay_update: TensorMap | None = None,
        current_parameters: TensorMap | None = None,
    ) -> None:
        """原子式提交一个已成功 optimizer step 的长期统计。

        方法先校验全部候选输入，再原地提交；任一输入非有限或坐标不一致时不会
        产生部分累计。skip step 应调用 :meth:`record_skip`，不得传入零贡献伪装。
        """

        candidates = [contribution]
        candidates.extend(
            value
            for value in (
                raw,
                raw_clipped,
                data_update,
                total_update,
                weight_decay_update,
                current_parameters,
            )
            if value is not None
        )
        for candidate in candidates:
            self._positive.assert_compatible(candidate)
            candidate.assert_finite()
        converted = contribution.to(dtype=self.accumulation_dtype)
        converted_raw = None if raw is None else raw.to(dtype=self.accumulation_dtype)
        converted_raw_clipped = (
            converted_raw
            if raw_clipped is None and converted_raw is not None
            else (
                None
                if raw_clipped is None
                else raw_clipped.to(dtype=self.accumulation_dtype)
            )
        )
        converted_data = None if data_update is None else data_update.to(dtype=self.accumulation_dtype)
        converted_total = None if total_update is None else total_update.to(dtype=self.accumulation_dtype)
        converted_decay = (
            None
            if weight_decay_update is None
            else weight_decay_update.to(dtype=self.accumulation_dtype)
        )
        converted_parameters = (
            None if current_parameters is None else current_parameters.to(dtype=self.accumulation_dtype)
        )
        converted_candidates = [converted]
        converted_candidates.extend(
            value
            for value in (
                converted_raw,
                converted_raw_clipped,
                converted_data,
                converted_total,
                converted_decay,
                converted_parameters,
            )
            if value is not None
        )
        for candidate in converted_candidates:
            candidate.assert_finite()
        for field_name, raw_value in (
            ("raw", converted_raw),
            ("raw_clipped", converted_raw_clipped),
        ):
            if raw_value is not None:
                for name, value in raw_value.items():
                    if bool((value < 0).any()):
                        raise CoreContractError(f"{field_name} importance {name!r} 不得为负")

        for name, value in converted.items():
            self._positive[name].add_(value.clamp_min(0))
            self._negative[name].add_((-value).clamp_min(0))
        if converted_raw is not None:
            for name, value in converted_raw.items():
                self._raw[name].add_(value)
        if converted_raw_clipped is not None:
            for name, value in converted_raw_clipped.items():
                self._raw_clipped[name].add_(value)
        if converted_data is not None:
            for name, value in converted_data.items():
                self._data_movement[name].add_(value.abs())
                self._data_displacement[name].add_(value)
        if converted_total is not None:
            for name, value in converted_total.items():
                self._total_movement[name].add_(value.abs())
                self._total_displacement[name].add_(value)
        if converted_decay is not None:
            for name, value in converted_decay.items():
                self._weight_decay_movement[name].add_(value.abs())
                self._weight_decay_displacement[name].add_(value)
        if converted_parameters is not None:
            for name, value in converted_parameters.items():
                self._magnitude[name].copy_(value.abs())
                self._last_parameters[name].copy_(value)
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
        # 首尾参数直接相减与逐 step delta 累加的浮点运算顺序不同，FP32 下不能
        # 要求逐位相等。容差仅用于一致性报错，不会生成或修饰任何公开统计量。
        endpoint_atol = atol
        endpoint_rtol = rtol
        if self._has_initial_parameters and atol == 0.0 and rtol == 0.0:
            epsilon = torch.finfo(self.accumulation_dtype).eps
            endpoint_atol = epsilon * max(1, self.successful_steps) * 4
            endpoint_rtol = epsilon * 4
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
            if self._has_initial_parameters and not torch.allclose(
                self._last_parameters[name] - self._initial_parameters[name],
                self._total_displacement[name],
                atol=endpoint_atol,
                rtol=endpoint_rtol,
            ):
                raise CoreContractError(f"{name!r} 的累计总位移与首尾端点不一致")

    def delta_since(self, previous: "ImportanceAccumulator") -> dict[str, object]:
        """返回两个 commit 边界之间的视图增量，不修改任一累计器。"""

        self._positive.assert_compatible(previous._positive)
        if self.accumulation_dtype != previous.accumulation_dtype:
            raise CoreContractError("不同累计 dtype 不能计算区间增量")
        if self.successful_steps < previous.successful_steps or self.skipped_steps < previous.skipped_steps:
            raise CoreContractError("previous 计数不能晚于当前累计器")
        return {
            "successful_steps": self.successful_steps - previous.successful_steps,
            "skipped_steps": self.skipped_steps - previous.skipped_steps,
            "signed": self.signed - previous.signed,
            "absolute": self.absolute - previous.absolute,
            "raw": self.raw - previous.raw,
            "raw_clipped": self.raw_clipped - previous.raw_clipped,
            "data_movement": self.data_movement - previous.data_movement,
            "total_movement": self.total_movement - previous.total_movement,
            "weight_decay_movement": self.weight_decay_movement - previous.weight_decay_movement,
        }

    def state_dict(self) -> dict[str, object]:
        """返回只含 primitive/TensorMap 的安全状态，不使用 pickle 对象图。"""

        return {
            "version": 2,
            "accumulation_dtype": str(self.accumulation_dtype),
            "successful_steps": self.successful_steps,
            "skipped_steps": self.skipped_steps,
            "positive": self._positive.to_dict(clone=True),
            "negative_mass": self._negative.to_dict(clone=True),
            "raw": self._raw.to_dict(clone=True),
            "raw_clipped": self._raw_clipped.to_dict(clone=True),
            "data_movement": self._data_movement.to_dict(clone=True),
            "data_displacement": self._data_displacement.to_dict(clone=True),
            "total_movement": self._total_movement.to_dict(clone=True),
            "total_displacement": self._total_displacement.to_dict(clone=True),
            "weight_decay_movement": self._weight_decay_movement.to_dict(clone=True),
            "weight_decay_displacement": self._weight_decay_displacement.to_dict(clone=True),
            "magnitude": self._magnitude.to_dict(clone=True),
            "initial_parameters": self._initial_parameters.to_dict(clone=True),
            "last_parameters": self._last_parameters.to_dict(clone=True),
            "has_initial_parameters": self._has_initial_parameters,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        """严格恢复累计状态，并对 0.3.x 的 v1 状态做无损可知字段迁移。

        v1 没有 clipped-raw、weight-decay 分解和有符号参数端点；这些量不能从旧
        字段反推，因此迁移时明确置零并保持 ``has_initial_parameters=false``。
        已存在的 signed/raw/data/total/magnitude 与计数逐位保留，随后写出统一 v2。
        """

        version = state.get("version")
        if version not in {1, 2}:
            raise CoreContractError("不支持的 ImportanceAccumulator state version")
        normalized: dict[str, object] = dict(state)
        if version == 1:
            expected_v1 = {
                "version", "accumulation_dtype", "successful_steps", "skipped_steps",
                "positive", "negative_mass", "raw", "data_movement",
                "data_displacement", "total_movement", "total_displacement", "magnitude",
            }
            if set(normalized) != expected_v1:
                raise CoreContractError("ImportanceAccumulator v1 字段集合无效")
            zero_state = {
                name: torch.zeros_like(value) for name, value in self._positive.items()
            }
            normalized.update(
                {
                    "version": 2,
                    "raw_clipped": {name: value.clone() for name, value in zero_state.items()},
                    "weight_decay_movement": {
                        name: value.clone() for name, value in zero_state.items()
                    },
                    "weight_decay_displacement": {
                        name: value.clone() for name, value in zero_state.items()
                    },
                    "initial_parameters": {
                        name: value.clone() for name, value in zero_state.items()
                    },
                    "last_parameters": {
                        name: value.clone() for name, value in zero_state.items()
                    },
                    "has_initial_parameters": False,
                }
            )
        state = normalized
        expected_dtype = str(self.accumulation_dtype)
        if state.get("accumulation_dtype") != expected_dtype:
            raise CoreContractError("累计状态 dtype 与当前合同不一致")
        destinations = {
            "positive": self._positive,
            "negative_mass": self._negative,
            "raw": self._raw,
            "raw_clipped": self._raw_clipped,
            "data_movement": self._data_movement,
            "data_displacement": self._data_displacement,
            "total_movement": self._total_movement,
            "total_displacement": self._total_displacement,
            "weight_decay_movement": self._weight_decay_movement,
            "weight_decay_displacement": self._weight_decay_displacement,
            "magnitude": self._magnitude,
            "initial_parameters": self._initial_parameters,
            "last_parameters": self._last_parameters,
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
        if (
            isinstance(successful_steps, bool)
            or not isinstance(successful_steps, int)
            or successful_steps < 0
        ):
            raise CoreContractError("successful_steps 非法")
        if (
            isinstance(skipped_steps, bool)
            or not isinstance(skipped_steps, int)
            or skipped_steps < 0
        ):
            raise CoreContractError("skipped_steps 非法")
        has_initial = state.get("has_initial_parameters")
        if type(has_initial) is not bool:
            raise CoreContractError("has_initial_parameters 非法")
        for key, destination in destinations.items():
            _copy_into(destination, staged[key])
        self.successful_steps = successful_steps
        self.skipped_steps = skipped_steps
        self._has_initial_parameters = has_initial
        self.validate_invariants()
