"""无需模型与外部资产的确定性合成梯度 provider。

该实现把一个有限的 ``sample_id -> parameter gradients`` 表视作冻结经验分布。
同一个 sample ID 总是返回完全相同的单样本梯度；不同 draw 偶然命中同一
sample ID 时会按出现次数重复计入。这一点专门用于验证 Stage 2 的有放回
抽样、draw 身份和碰撞策略。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np

from .protocols import GradientBatch


def _draw_sample_id(draw: object) -> Hashable:
    """从 draw 对象或直接 ID 中取得 sample ID。

    Sampling 层的 ``Draw`` 暴露 ``sample_id`` 属性；为了让 provider 也能在
    很小的单元测试中独立使用，这里同时接受直接传入的 hashable ID。
    """

    return getattr(draw, "sample_id", draw)  # type: ignore[no-any-return]


def _canonical_table_digest(
    table: Mapping[Hashable, Mapping[str, np.ndarray]],
    weights: Mapping[Hashable, float],
) -> str:
    """摘要梯度表及逐样本权重。

    权重会改变 provider 返回的批平均梯度，因此它属于固定状态身份。旧实现
    只摘要梯度数组，会让同一张量表在更换权重后仍得到相同 ``state_digest``；
    这会破坏 Stage 2 的只读状态证明。
    """

    digest = hashlib.sha256()
    for sample_id in sorted(table, key=lambda value: str(value)):
        encoded_id = json.dumps(
            sample_id,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        digest.update(len(encoded_id).to_bytes(8, "big"))
        digest.update(encoded_id)
        encoded_weight = float(weights[sample_id]).hex().encode("ascii")
        digest.update(len(encoded_weight).to_bytes(8, "big"))
        digest.update(encoded_weight)
        for name in sorted(table[sample_id]):
            array = np.ascontiguousarray(table[sample_id][name])
            digest.update(name.encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(array.shape).encode("ascii"))
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(slots=True)
class SyntheticGradientProvider:
    """由有限梯度表实现的 :class:`FixedStateGradientProvider`。

    Parameters
    ----------
    sample_gradients:
        两层映射 ``sample_id -> parameter_name -> gradient``。构造时会深拷贝
        成只读 NumPy 数组，防止调用方之后修改原数组造成静默状态漂移。
    fixed_state_id:
        合成 checkpoint 的逻辑身份。该值与表内容共同进入状态摘要。
    statistical_weights:
        可选的逐 sample 正权重。缺省时每个 draw 权重为 1；如果同一 sample
        ID 被抽中两次，它的权重也按两次累计。
    statistical_unit, weight_unit, sampling_design:
        写入每个 :class:`GradientBatch` 的统计语义。直接构造 provider 时必须
        显式填写，避免测试 fixture 无意中依赖 runner 的硬编码默认值。
    weights_exogenous, common_mean_assumption:
        加权 U 的两项无偏性前提。provider 只忠实传递声明；传入 ``False``
        完全合法，但一旦实际进入非等权 U 路径，Stage 2 runner 会 fail-closed。
    """

    _table: dict[Hashable, dict[str, np.ndarray]]
    fixed_state_id: str
    _weights: dict[Hashable, float]
    _registry_hash: str
    _parameter_names: tuple[str, ...]
    _state_digest: str
    statistical_unit: str
    weight_unit: str
    sampling_design: str
    weights_exogenous: bool
    common_mean_assumption: bool

    def __init__(
        self,
        sample_gradients: Mapping[Hashable, Mapping[str, object]],
        *,
        fixed_state_id: str = "synthetic-state-v1",
        statistical_weights: Mapping[Hashable, float] | None = None,
        statistical_unit: str,
        weight_unit: str,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> None:
        if not sample_gradients:
            raise ValueError("sample_gradients 不能为空")
        first = next(iter(sample_gradients.values()))
        names = tuple(sorted(first))
        if not names:
            raise ValueError("每个样本至少需要一个参数梯度")

        table: dict[Hashable, dict[str, np.ndarray]] = {}
        shapes: dict[str, tuple[int, ...]] = {}
        for sample_id, gradients in sample_gradients.items():
            if tuple(sorted(gradients)) != names:
                raise ValueError("所有样本必须具有完全相同的参数名集合")
            copied: dict[str, np.ndarray] = {}
            for name in names:
                array = np.array(gradients[name], dtype=np.float64, copy=True)
                if not np.all(np.isfinite(array)):
                    raise ValueError(f"参数 {name!r} 的合成梯度包含非有限值")
                if name in shapes and shapes[name] != array.shape:
                    raise ValueError(f"参数 {name!r} 在不同样本中的 shape 不一致")
                shapes[name] = array.shape
                array.setflags(write=False)
                copied[name] = array
            table[sample_id] = copied

        weights = {
            sample_id: float(
                1.0 if statistical_weights is None else statistical_weights[sample_id]
            )
            for sample_id in table
        }
        if set(weights) != set(table) or any(
            not np.isfinite(value) or value <= 0 for value in weights.values()
        ):
            raise ValueError("statistical_weights 必须覆盖全部样本且严格为正有限数")
        for field_name, value in (
            ("statistical_unit", statistical_unit),
            ("weight_unit", weight_unit),
            ("sampling_design", sampling_design),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} 必须是非空字符串")
        if type(weights_exogenous) is not bool:
            raise TypeError("weights_exogenous 必须是显式 bool")
        if type(common_mean_assumption) is not bool:
            raise TypeError("common_mean_assumption 必须是显式 bool")

        registry_payload = [
            {"name": name, "shape": list(shapes[name]), "dtype": "float64"}
            for name in names
        ]
        registry_hash = hashlib.sha256(
            json.dumps(
                registry_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        table_hash = _canonical_table_digest(table, weights)
        sampling_contract = json.dumps(
            {
                "common_mean_assumption": common_mean_assumption,
                "sampling_design": sampling_design,
                "statistical_unit": statistical_unit,
                "weight_unit": weight_unit,
                "weights_exogenous": weights_exogenous,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        state_hash = hashlib.sha256(
            (
                f"{fixed_state_id}\0{registry_hash}\0{table_hash}\0"
                f"{sampling_contract}"
            ).encode("utf-8")
        ).hexdigest()

        self._table = table
        self.fixed_state_id = fixed_state_id
        self._weights = weights
        self._registry_hash = registry_hash
        self._parameter_names = names
        self._state_digest = state_hash
        self.statistical_unit = statistical_unit
        self.weight_unit = weight_unit
        self.sampling_design = sampling_design
        self.weights_exogenous = weights_exogenous
        self.common_mean_assumption = common_mean_assumption

    @classmethod
    def from_location_scale(
        cls,
        *,
        parameter_shapes: Mapping[str, tuple[int, ...]],
        sample_count: int,
        mean: float = 0.0,
        noise_scale: float = 1.0,
        seed: int = 0,
        fixed_state_id: str = "synthetic-normal-v1",
    ) -> "SyntheticGradientProvider":
        """构造可复现的有限高斯梯度表。

        这里的随机性只在构造时发生；表构造完成后即成为固定经验分布，所以
        后续相同 draw manifest 的结果逐字节确定。该便捷入口只用于本机 fixture，
        不能生成 formal reference 或正式 EstimatorDecision。
        """

        if sample_count <= 0:
            raise ValueError("sample_count 必须严格为正")
        if noise_scale < 0 or not np.isfinite(noise_scale):
            raise ValueError("noise_scale 必须是非负有限数")
        rng = np.random.default_rng(seed)
        table: dict[int, dict[str, np.ndarray]] = {}
        for sample_id in range(sample_count):
            table[sample_id] = {
                name: rng.normal(mean, noise_scale, size=shape)
                for name, shape in sorted(parameter_shapes.items())
            }
        # 该便捷构造器专用于仓库内的均匀、有放回 synthetic fixture。这里在
        # provider 边界明确写出五项假设，而不是让 Stage 2 runner 看到等权后
        # 自行猜测统计语义。真实 adapter 必须按自身数据管线另行声明。
        return cls(
            table,
            fixed_state_id=fixed_state_id,
            statistical_unit="synthetic_draw_group_mean",
            weight_unit="synthetic_draw_count",
            sampling_design="uniform_with_replacement_disjoint_draw_groups",
            weights_exogenous=True,
            common_mean_assumption=True,
        )

    @property
    def registry_hash(self) -> str:
        """返回合成参数注册表摘要。"""

        return self._registry_hash

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """返回稳定排序的参数名。"""

        return self._parameter_names

    @property
    def sample_ids(self) -> tuple[Hashable, ...]:
        """返回有限经验分布中可抽取的 sample IDs。"""

        return tuple(self._table)

    def state_digest(self) -> str:
        """返回固定梯度表摘要；读取梯度不会改变它。"""

        return self._state_digest

    def assert_unchanged(self, expected_digest: str) -> None:
        """验证 provider 未发生状态漂移。"""

        if self._state_digest != expected_digest:
            raise RuntimeError("synthetic provider 固定状态摘要发生漂移")

    def gradient(self, draws: Sequence[object]) -> GradientBatch:
        """按原始 draw 次数计算加权平均梯度。

        返回数组是新分配的可写数组，因此调用方可以安全地交给张量适配器；
        provider 内部的只读表不会泄漏给调用方。
        """

        if not draws:
            raise ValueError("draws 不能为空")
        sample_ids = tuple(_draw_sample_id(draw) for draw in draws)
        missing = [sample_id for sample_id in sample_ids if sample_id not in self._table]
        if missing:
            raise KeyError(f"draw 引用了经验分布之外的 sample ID: {missing[0]!r}")

        total_weight = float(sum(self._weights[sample_id] for sample_id in sample_ids))
        means: dict[str, np.ndarray] = {}
        for name in self._parameter_names:
            accumulator = np.zeros_like(self._table[sample_ids[0]][name], dtype=np.float64)
            for sample_id in sample_ids:
                accumulator += self._weights[sample_id] * self._table[sample_id][name]
            means[name] = accumulator / total_weight
        return GradientBatch(
            gradients=means,
            statistical_weight=total_weight,
            statistical_unit=self.statistical_unit,
            weight_unit=self.weight_unit,
            sampling_design=self.sampling_design,
            weights_exogenous=self.weights_exogenous,
            common_mean_assumption=self.common_mean_assumption,
            sample_ids=sample_ids,
        )
