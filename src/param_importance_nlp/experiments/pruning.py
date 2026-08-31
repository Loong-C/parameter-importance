"""Stage 7 剪枝研究矩阵编排。

坐标选择、alias 感知与可恢复参数上下文由 ``core.pruning`` 负责；本模块只把
importance artifact、方向、scope、比例和随机 mask seed 编译为不可歧义的
运行单元。每个非随机条件都必须绑定实际可用 artifact，不能用方法名替代数据。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..contracts.immutable import freeze_json_mapping


IMPORTANCE_METHODS = (
    "magnitude",
    "movement",
    "raw",
    "u",
    "double",
    "empirical_fisher",
    "si",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True, slots=True)
class ImportanceSourceSpec:
    """一份可用于剪枝排序的重要性 artifact 身份。"""

    method: str
    artifact_id: str
    artifact_hash: str
    coordinate_registry_hash: str
    score_view: str = "absolute"
    available: bool = True
    scope: str = "local_fixture"
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in IMPORTANCE_METHODS:
            raise ValueError(f"未知 importance method: {self.method!r}")
        if not self.artifact_id:
            raise ValueError("artifact_id 不能为空")
        if not _valid_sha256(self.artifact_hash) or not _valid_sha256(
            self.coordinate_registry_hash
        ):
            raise ValueError("artifact_hash 与 coordinate_registry_hash 必须是 SHA-256")
        if self.score_view not in {"signed", "positive", "absolute"}:
            raise ValueError("score_view 必须是 signed、positive 或 absolute")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("source scope 只能是 local_fixture 或 formal")
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PruningRunSpec:
    """一个可重放剪枝条件的完整身份。"""

    run_id: str
    method: str
    direction: str
    pruning_scope: str
    ratio: float
    mask_seed: int | None
    source_artifact_hash: str | None
    coordinate_registry_hash: str
    tie_breaker: str = "canonical_coordinate_id"

    def __post_init__(self) -> None:
        if self.method not in (*IMPORTANCE_METHODS, "random"):
            raise ValueError("未知剪枝 method")
        if self.direction not in {"high", "low", "random"}:
            raise ValueError("direction 必须是 high、low 或 random")
        if self.pruning_scope not in {"global", "layer_balanced"}:
            raise ValueError("pruning_scope 必须是 global 或 layer_balanced")
        if not math.isfinite(self.ratio) or not 0 <= self.ratio <= 1:
            raise ValueError("ratio 必须在 [0,1] 内")
        if self.direction == "random":
            if self.mask_seed is None or self.source_artifact_hash is not None:
                raise ValueError("random 条件必须有 mask_seed 且不能伪造 source artifact")
        else:
            if self.mask_seed is not None or self.source_artifact_hash is None:
                raise ValueError("非随机条件必须绑定 source artifact 且不能带 mask_seed")
        if not _valid_sha256(self.coordinate_registry_hash):
            raise ValueError("coordinate_registry_hash 必须是 SHA-256")
        if self.tie_breaker != "canonical_coordinate_id":
            raise ValueError("并列值只能使用 canonical coordinate ID 决胜")


@dataclass(frozen=True, slots=True)
class PruningStudySpec:
    """固定比例网格和随机重复的剪枝研究合同。"""

    study_id: str
    sources: tuple[ImportanceSourceSpec, ...]
    ratios: tuple[float, ...]
    pruning_scopes: tuple[str, ...] = ("global", "layer_balanced")
    random_mask_seeds: tuple[int, ...] = (0, 1, 2)
    run_intent: str = "local_fixture"
    frozen: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", tuple(self.sources))
        object.__setattr__(self, "ratios", tuple(float(ratio) for ratio in self.ratios))
        object.__setattr__(self, "pruning_scopes", tuple(self.pruning_scopes))
        object.__setattr__(self, "random_mask_seeds", tuple(self.random_mask_seeds))
        if not self.study_id:
            raise ValueError("study_id 不能为空")
        if not self.sources or not self.ratios:
            raise ValueError("sources 与 ratios 不能为空")
        if tuple(sorted(set(self.ratios))) != self.ratios:
            raise ValueError("ratios 必须唯一并按升序冻结")
        if any(not math.isfinite(ratio) or not 0 <= ratio <= 1 for ratio in self.ratios):
            raise ValueError("全部 pruning ratio 必须位于 [0,1]")
        if not self.frozen:
            raise ValueError("执行层只消费 frozen PruningStudySpec")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise ValueError("run_intent 只能是 local_fixture 或 formal")
        if set(self.pruning_scopes) - {"global", "layer_balanced"}:
            raise ValueError("存在未知 pruning scope")
        if len(set(self.random_mask_seeds)) != len(self.random_mask_seeds):
            raise ValueError("random mask seeds 必须唯一")
        if any(not source.available for source in self.sources):
            missing = [source.method for source in self.sources if not source.available]
            raise ValueError(f"重要性 artifact 不可用: {missing}")
        registry_hashes = {source.coordinate_registry_hash for source in self.sources}
        if len(registry_hashes) != 1:
            raise ValueError("全部剪枝 source 必须共享同一 coordinate registry")
        if self.run_intent == "formal" and any(
            source.scope != "formal" for source in self.sources
        ):
            raise ValueError("formal 剪枝矩阵不能消费 local_fixture source")

    @property
    def coordinate_registry_hash(self) -> str:
        return self.sources[0].coordinate_registry_hash

    def compile(self) -> tuple[PruningRunSpec, ...]:
        """按 method/direction/scope/ratio/seed 稳定顺序编译运行单元。"""

        runs: list[PruningRunSpec] = []
        for source in sorted(self.sources, key=lambda item: (item.method, item.artifact_hash)):
            for direction in ("high", "low"):
                for pruning_scope in sorted(self.pruning_scopes):
                    for ratio in self.ratios:
                        payload = {
                            "study_id": self.study_id,
                            "method": source.method,
                            "direction": direction,
                            "scope": pruning_scope,
                            "ratio_hex": ratio.hex(),
                            "source": source.artifact_hash,
                            "registry": self.coordinate_registry_hash,
                        }
                        runs.append(
                            PruningRunSpec(
                                run_id=f"prune-{_digest(payload)[:20]}",
                                method=source.method,
                                direction=direction,
                                pruning_scope=pruning_scope,
                                ratio=ratio,
                                mask_seed=None,
                                source_artifact_hash=source.artifact_hash,
                                coordinate_registry_hash=self.coordinate_registry_hash,
                            )
                        )
        for pruning_scope in sorted(self.pruning_scopes):
            for ratio in self.ratios:
                for seed in sorted(self.random_mask_seeds):
                    payload = {
                        "study_id": self.study_id,
                        "method": "random",
                        "scope": pruning_scope,
                        "ratio_hex": ratio.hex(),
                        "seed": seed,
                        "registry": self.coordinate_registry_hash,
                    }
                    runs.append(
                        PruningRunSpec(
                            run_id=f"prune-{_digest(payload)[:20]}",
                            method="random",
                            direction="random",
                            pruning_scope=pruning_scope,
                            ratio=ratio,
                            mask_seed=seed,
                            source_artifact_hash=None,
                            coordinate_registry_hash=self.coordinate_registry_hash,
                        )
                    )
        if len({run.run_id for run in runs}) != len(runs):
            raise RuntimeError("剪枝运行身份发生哈希碰撞")
        return tuple(runs)
