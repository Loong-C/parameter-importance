"""Stage 2 有放回抽样、嵌套 microbatch 与 B/M 选择合同。

本模块只生成身份和编排，不读取数据、不计算模型梯度。五条随机流使用独立
seed namespace；draw ID 表示一次抽样事件，因此必须唯一，而 sample ID
表示经验分布中的有限单元，允许因有放回抽样发生碰撞。任何调用方都不得
为了让样本集合“不重叠”而去重或重抽。
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Hashable, Iterable, Literal, Mapping, Sequence

from ..contracts.immutable import freeze_json_mapping, thaw_json_value


STREAM_NAMES = (
    "reference_sizing",
    "reference_A",
    "reference_B",
    "pilot",
    "confirmatory",
)
"""Stage 2 冻结的五条独立随机流名称。"""

CANDIDATE_BATCH_SIZES = (32, 64, 128, 256)
CANDIDATE_MICROBATCH_COUNTS = (2, 4, 8, 16, 32)
MICROBATCH_SELECTION_ORDER = (32, 16, 8, 4)

StreamName = Literal[
    "reference_sizing", "reference_A", "reference_B", "pilot", "confirmatory"
]


class FormalDecisionBlocked(RuntimeError):
    """本机 fixture 代码被要求发布 formal 决策时抛出的硬错误。"""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class SamplingUniverse:
    r"""有限经验分布 \(\mathcal F\) 的版本化身份。

    ``sample_ids`` 必须唯一，因为每个 ID 应唯一定位一个不重叠统计单元；
    重复只允许出现在后续 draw manifest 中。当前核心仅允许均匀、有放回抽样，
    这样 raw、double 与 U 才共享同一个明确 estimand。
    """

    universe_id: str
    sample_ids: tuple[Hashable, ...]
    version: str = "sampling-universe-v1"
    sampling_design: str = "uniform_with_replacement"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.universe_id, str) or not self.universe_id:
            raise ValueError("universe_id 不能为空")
        if self.version != "sampling-universe-v1":
            raise ValueError(f"不支持的 sampling universe 版本: {self.version!r}")
        if self.sampling_design != "uniform_with_replacement":
            raise ValueError("当前只支持 uniform_with_replacement")
        if not self.sample_ids:
            raise ValueError("sample_ids 不能为空")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("SamplingUniverse metadata 必须是 object")
        try:
            unique_count = len(set(self.sample_ids))
        except TypeError as exc:
            raise TypeError("sample ID 必须可哈希") from exc
        if unique_count != len(self.sample_ids):
            raise ValueError("SamplingUniverse 中的 sample ID 必须唯一")
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="SamplingUniverse.metadata"),
        )
        # 在构造时验证所有字段可以进入 canonical manifest；不接受 repr-only 对象。
        _canonical_json(self.to_manifest())

    def to_manifest(self) -> dict[str, object]:
        """返回不含运行态数据的 canonical manifest。"""

        return {
            "version": self.version,
            "universe_id": self.universe_id,
            "sampling_design": self.sampling_design,
            "sample_ids": list(self.sample_ids),
            "metadata": thaw_json_value(self.metadata),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> "SamplingUniverse":
        """严格加载 sampling universe；未知字段和宽松数值转换均失败。"""

        required = {
            "version",
            "universe_id",
            "sampling_design",
            "sample_ids",
            "metadata",
        }
        if set(value) != required:
            raise ValueError(
                "SamplingUniverse 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        if not isinstance(value["universe_id"], str):
            raise TypeError("SamplingUniverse.universe_id 必须是字符串")
        if not isinstance(value["sample_ids"], list):
            raise TypeError("SamplingUniverse.sample_ids 必须是数组")
        if not isinstance(value["metadata"], Mapping):
            raise TypeError("SamplingUniverse.metadata 必须是 object")
        return cls(
            universe_id=value["universe_id"],
            sample_ids=tuple(value["sample_ids"]),
            version=str(value["version"]),
            sampling_design=str(value["sampling_design"]),
            metadata=value["metadata"],
        )

    @property
    def digest(self) -> str:
        """返回经验分布身份摘要。"""

        return _sha256_json(self.to_manifest())


@dataclass(frozen=True, slots=True)
class Draw:
    """一次有放回抽样事件。

    ``draw_id`` 与 stream/position 一一对应；``sample_id`` 可以在同一或不同
    stream 中重复。``position`` 是 stream 内从零开始的绝对位置，不因分片
    或恢复而重新编号。
    """

    draw_id: str
    stream: StreamName
    position: int
    sample_id: Hashable
    algorithm_version: str

    def to_manifest(self) -> dict[str, object]:
        return {
            "draw_id": self.draw_id,
            "stream": self.stream,
            "position": self.position,
            "sample_id": self.sample_id,
            "algorithm_version": self.algorithm_version,
        }


@dataclass(frozen=True, slots=True)
class SamplingPlan:
    """从一个冻结经验分布派生五条独立 draw streams。

    每条 stream 的整数 seed 必须互异。实现用 stream-specific seed 初始化
    Python 的 MT19937，并以 ``randrange`` 做无偏有限整数抽样；读取后续分片
    时会从 stream 起点重放到 ``start``，用确定性换取明确、可审计的恢复语义。
    """

    universe: SamplingUniverse
    stream_seeds: Mapping[str, int]
    algorithm_version: str = "stage2-draws-python-randrange-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.stream_seeds, Mapping):
            raise TypeError("stream_seeds 必须是 object")
        if set(self.stream_seeds) != set(STREAM_NAMES):
            missing = sorted(set(STREAM_NAMES) - set(self.stream_seeds))
            extra = sorted(set(self.stream_seeds) - set(STREAM_NAMES))
            raise ValueError(f"stream seed 集合错误；missing={missing}, extra={extra}")
        seeds = tuple(self.stream_seeds[name] for name in STREAM_NAMES)
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise TypeError("全部 stream seed 必须是整数且不能是 bool")
        if len(set(seeds)) != len(seeds):
            raise ValueError("五条 stream 必须使用互不相同的 seed namespace")
        if self.algorithm_version != "stage2-draws-python-randrange-v1":
            raise ValueError(f"不支持的 draw 算法版本: {self.algorithm_version!r}")
        object.__setattr__(
            self,
            "stream_seeds",
            freeze_json_mapping(self.stream_seeds, field="SamplingPlan.stream_seeds"),
        )

    def _stream_seed(self, stream: StreamName) -> int:
        payload = {
            "universe_digest": self.universe.digest,
            "stream": stream,
            "seed": self.stream_seeds[stream],
            "algorithm_version": self.algorithm_version,
        }
        return int.from_bytes(hashlib.sha256(_canonical_json(payload)).digest(), "big")

    def draws(self, stream: StreamName, count: int, *, start: int = 0) -> tuple[Draw, ...]:
        """确定性生成 ``[start, start + count)`` 的 draw manifest。

        重复调用相同区间会得到逐字段相同的结果，因而失败重试不需要生成新
        样本。sample ID 碰撞原样保留。
        """

        if stream not in STREAM_NAMES:
            raise ValueError(f"未知 stream: {stream!r}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count 必须是非负整数")
        if isinstance(start, bool) or not isinstance(start, int) or start < 0:
            raise ValueError("start 必须是非负整数")
        rng = random.Random(self._stream_seed(stream))
        result: list[Draw] = []
        size = len(self.universe.sample_ids)
        for position in range(start + count):
            sample_id = self.universe.sample_ids[rng.randrange(size)]
            if position < start:
                continue
            identity = {
                "universe_digest": self.universe.digest,
                "stream": stream,
                "position": position,
                "algorithm_version": self.algorithm_version,
            }
            draw_id = f"{stream}:{position:012d}:{_sha256_json(identity)[:16]}"
            result.append(
                Draw(
                    draw_id=draw_id,
                    stream=stream,
                    position=position,
                    sample_id=sample_id,
                    algorithm_version=self.algorithm_version,
                )
            )
        return tuple(result)

    @property
    def digest(self) -> str:
        """返回 universe、seed namespaces 与算法版本的联合摘要。"""

        return _sha256_json(
            {
                "universe_digest": self.universe.digest,
                "stream_seeds": {
                    name: self.stream_seeds[name] for name in STREAM_NAMES
                },
                "algorithm_version": self.algorithm_version,
            }
        )

    def to_dict(self) -> dict[str, object]:
        """返回 hash 绑定的 Stage 2 公共 wire artifact。"""

        return {
            "schema_version": "sampling-plan-v1",
            "universe": self.universe.to_manifest(),
            "universe_hash": self.universe.digest,
            "stream_seeds": {
                name: self.stream_seeds[name] for name in STREAM_NAMES
            },
            "algorithm_version": self.algorithm_version,
            "plan_hash": self.digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SamplingPlan":
        """严格加载并重算 universe/plan 两层摘要。"""

        required = {
            "schema_version",
            "universe",
            "universe_hash",
            "stream_seeds",
            "algorithm_version",
            "plan_hash",
        }
        if set(value) != required:
            raise ValueError(
                "SamplingPlan 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        if value["schema_version"] != "sampling-plan-v1":
            raise ValueError("不支持的 SamplingPlan schema")
        if not isinstance(value["universe"], Mapping):
            raise TypeError("SamplingPlan.universe 必须是 object")
        if not isinstance(value["stream_seeds"], Mapping):
            raise TypeError("SamplingPlan.stream_seeds 必须是 object")
        universe = SamplingUniverse.from_manifest(value["universe"])
        if value["universe_hash"] != universe.digest:
            raise ValueError("SamplingPlan universe_hash 与 universe 内容不一致")
        seeds: dict[str, int] = {}
        for name, seed in value["stream_seeds"].items():
            if not isinstance(name, str):
                raise TypeError("SamplingPlan stream seed 名必须是字符串")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("SamplingPlan stream seed 必须是整数且不能是 bool")
            seeds[name] = seed
        plan = cls(
            universe=universe,
            stream_seeds=seeds,
            algorithm_version=str(value["algorithm_version"]),
        )
        if value["plan_hash"] != plan.digest:
            raise ValueError("SamplingPlan plan_hash 与完整内容不一致")
        return plan


@dataclass(frozen=True, slots=True)
class RepetitionMapping:
    """一次 paired repetition 的嵌套分组与 double 两半映射。

    基础 draw 顺序不可修改。对任意 ``M``，第 ``j`` 组是长度 ``B/M`` 的
    相邻区间；所以较粗分组总是由相邻的细分组合并得到。double 固定使用前
    后两个 draw-ID 半区，而不是按 sample ID 去重后重新划分。
    """

    repetition_id: str
    draws: tuple[Draw, ...]
    m_values: tuple[int, ...]
    mapping_version: str = "stage2-nested-adjacent-v1"

    def __post_init__(self) -> None:
        batch_size = len(self.draws)
        if not self.repetition_id:
            raise ValueError("repetition_id 不能为空")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("double 等预算映射要求 B 为不小于 2 的偶数")
        if len({draw.draw_id for draw in self.draws}) != batch_size:
            raise ValueError("同一 repetition 的 draw ID 必须唯一")
        if len({(draw.stream, draw.position) for draw in self.draws}) != batch_size:
            raise ValueError("同一 stream position 不得在 repetition 中重复")
        if len({draw.stream for draw in self.draws}) != 1:
            raise ValueError("一个 repetition 的全部 draws 必须来自同一冻结 stream")
        if len({draw.algorithm_version for draw in self.draws}) != 1:
            raise ValueError("一个 repetition 不能混用不同 draw 算法版本")
        positions = [draw.position for draw in self.draws]
        if positions != list(range(positions[0], positions[0] + batch_size)):
            raise ValueError("repetition draws 必须保持连续、递增的 stream position")
        if not self.m_values:
            raise ValueError("m_values 不能为空")
        normalized = tuple(sorted(set(self.m_values)))
        if normalized != self.m_values:
            raise ValueError("m_values 必须去重并按升序提供")
        if any(m < 2 or batch_size % m for m in self.m_values):
            raise ValueError("每个 M 必须不小于 2 且整除 B")
        largest = max(self.m_values)
        if any(largest % m for m in self.m_values):
            raise ValueError("嵌套分组要求每个较小 M 整除 M_max")
        if self.mapping_version != "stage2-nested-adjacent-v1":
            raise ValueError(f"不支持的 mapping 版本: {self.mapping_version!r}")

    @classmethod
    def create(
        cls,
        *,
        repetition_id: str,
        draws: Sequence[Draw],
        m_values: Iterable[int] = CANDIDATE_MICROBATCH_COUNTS,
    ) -> "RepetitionMapping":
        """从冻结 draw 顺序创建嵌套映射。"""

        values = tuple(sorted(set(m_values)))
        return cls(repetition_id=repetition_id, draws=tuple(draws), m_values=values)

    @property
    def batch_size(self) -> int:
        return len(self.draws)

    def groups(self, microbatch_count: int) -> tuple[tuple[Draw, ...], ...]:
        """返回指定 M 的相邻、等大小统计 microbatches。"""

        if microbatch_count not in self.m_values:
            raise ValueError(f"M={microbatch_count} 不在冻结 m_values 中")
        width = self.batch_size // microbatch_count
        return tuple(
            self.draws[index : index + width]
            for index in range(0, self.batch_size, width)
        )

    @property
    def double_halves(self) -> tuple[tuple[Draw, ...], tuple[Draw, ...]]:
        """返回等总预算 double estimator 的前后两个 draw-ID 半区。"""

        half = self.batch_size // 2
        return self.draws[:half], self.draws[half:]

    @property
    def sample_collision_count(self) -> int:
        """返回重复 sample 命中数；该数是诊断字段，不是错误。"""

        return self.batch_size - len({draw.sample_id for draw in self.draws})

    def to_manifest(self) -> dict[str, object]:
        return {
            "mapping_version": self.mapping_version,
            "repetition_id": self.repetition_id,
            "batch_size": self.batch_size,
            "m_values": list(self.m_values),
            "draws": [draw.to_manifest() for draw in self.draws],
            "double_half_draw_ids": [
                [draw.draw_id for draw in half] for half in self.double_halves
            ],
            "sample_collision_count": self.sample_collision_count,
        }

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_manifest())


@dataclass(frozen=True, slots=True)
class PilotObservation:
    """盲化 B/M 扫描可以读取的一行 pilot 观测。

    该类型故意没有 bias、方法均值、NMSE、相关性或显著性字段，避免选择器
    根据科学结果挑选参数。``anchors_runnable``/``finite`` 表示六个冻结 anchor
    的合取结果。
    """

    batch_size: int
    microbatch_count: int
    anchors_runnable: bool
    finite: bool
    aggregation_overhead_ratio: float
    r_required: int
    resource_within_budget: bool

    def __post_init__(self) -> None:
        if self.batch_size not in CANDIDATE_BATCH_SIZES:
            raise ValueError("batch_size 不在冻结候选集合中")
        if self.microbatch_count not in CANDIDATE_MICROBATCH_COUNTS:
            raise ValueError("microbatch_count 不在冻结候选集合中")
        if not math.isfinite(self.aggregation_overhead_ratio):
            raise ValueError("aggregation_overhead_ratio 必须有限")
        if self.r_required <= 0:
            raise ValueError("r_required 必须严格为正")


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """一个候选在确定性扫描中的机器可读判定。"""

    batch_size: int
    microbatch_count: int
    eligible_for_m: bool
    eligible_for_pair: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrimaryPairDecision:
    """本机 fixture 的 B/M 选择结果。

    即便状态为 ``FIXTURE_SELECTED``，``formal_eligible`` 也恒为 ``False``；
    正式 B/M/R 必须由服务器 pilot、Gate 与冻结 artifact 共同产生。
    """

    status: Literal["FIXTURE_SELECTED", "BLOCKED"]
    batch_size: int | None
    microbatch_count: int | None
    scope: Literal["local_fixture"]
    formal_eligible: bool
    evaluations: tuple[CandidateEvaluation, ...]
    reason: str | None = None


def select_primary_pair(
    observations: Sequence[PilotObservation],
    *,
    r_max: int,
    scope: str = "local_fixture",
    overhead_limit: float = 0.25,
) -> PrimaryPairDecision:
    """按预注册顺序选择唯一的本机 fixture B/M pair。

    算法先对每个 B 按 ``[32,16,8,4]`` 选第一个满足整除、数值、anchor
    可运行与聚合开销条件的 M；再按 B 升序选择第一个同时满足最坏 R 与资源
    预算的 pair。M=2 仅用于 double 等价检查，不参与主 M 选择。

    本函数拒绝 ``formal`` scope，防止本机 synthetic 数据意外冻结正式矩阵。
    """

    if scope != "local_fixture":
        raise FormalDecisionBlocked(
            "本机选择器只能生成 local_fixture 决策；formal B/M 仍为 UNFROZEN/BLOCKED"
        )
    if r_max <= 0:
        raise ValueError("r_max 必须严格为正")
    keyed: dict[tuple[int, int], PilotObservation] = {}
    for observation in observations:
        key = (observation.batch_size, observation.microbatch_count)
        if key in keyed:
            raise ValueError(f"重复 pilot candidate: B={key[0]}, M={key[1]}")
        keyed[key] = observation

    evaluations: list[CandidateEvaluation] = []
    candidates_by_b: dict[int, PilotObservation] = {}
    for batch_size in CANDIDATE_BATCH_SIZES:
        chosen_for_b = False
        for microbatch_count in MICROBATCH_SELECTION_ORDER:
            reasons: list[str] = []
            observation = keyed.get((batch_size, microbatch_count))
            if batch_size % microbatch_count:
                reasons.append("M_DOES_NOT_DIVIDE_B")
            if observation is None:
                reasons.append("OBSERVATION_MISSING")
            else:
                if not observation.anchors_runnable:
                    reasons.append("ANCHOR_NOT_RUNNABLE")
                if not observation.finite:
                    reasons.append("NON_FINITE")
                if observation.aggregation_overhead_ratio > overhead_limit:
                    reasons.append("AGGREGATION_OVERHEAD_EXCEEDED")
            eligible_for_m = not reasons and not chosen_for_b
            eligible_for_pair = False
            if eligible_for_m and observation is not None:
                chosen_for_b = True
                candidates_by_b[batch_size] = observation
                pair_reasons: list[str] = []
                if observation.r_required > r_max:
                    pair_reasons.append("R_REQUIRED_EXCEEDED")
                if not observation.resource_within_budget:
                    pair_reasons.append("RESOURCE_BUDGET_EXCEEDED")
                reasons.extend(pair_reasons)
                eligible_for_pair = not pair_reasons
            elif not reasons and chosen_for_b:
                reasons.append("LOWER_PRIORITY_THAN_SELECTED_M")
            evaluations.append(
                CandidateEvaluation(
                    batch_size=batch_size,
                    microbatch_count=microbatch_count,
                    eligible_for_m=eligible_for_m,
                    eligible_for_pair=eligible_for_pair,
                    reasons=tuple(reasons),
                )
            )

    for batch_size in CANDIDATE_BATCH_SIZES:
        observation = candidates_by_b.get(batch_size)
        if observation is None:
            continue
        if observation.r_required <= r_max and observation.resource_within_budget:
            return PrimaryPairDecision(
                status="FIXTURE_SELECTED",
                batch_size=batch_size,
                microbatch_count=observation.microbatch_count,
                scope="local_fixture",
                formal_eligible=False,
                evaluations=tuple(evaluations),
            )
    return PrimaryPairDecision(
        status="BLOCKED",
        batch_size=None,
        microbatch_count=None,
        scope="local_fixture",
        formal_eligible=False,
        evaluations=tuple(evaluations),
        reason="NO_PREREGISTERED_PAIR_SATISFIES_FIXTURE_CONSTRAINTS",
    )
