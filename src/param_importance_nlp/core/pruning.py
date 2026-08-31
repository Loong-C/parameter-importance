"""基于 canonical coordinate ID 的可恢复非破坏性剪枝。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Literal, Mapping

import torch

from .errors import CoreContractError
from .tensors import TensorMap


@dataclass(frozen=True, slots=True)
class PruningPlan:
    """一次固定比例剪枝的纯选择合同。"""

    ratio: float
    strategy: Literal["high", "low", "random"]
    scope: Literal["global", "layer_balanced"] = "global"
    seed: int = 0
    score_view: Literal["signed", "positive", "absolute"] = "absolute"

    def __post_init__(self) -> None:
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, (int, float)):
            raise CoreContractError("pruning ratio 必须是数值且不能是 bool")
        normalized_ratio = float(self.ratio)
        if not math.isfinite(normalized_ratio) or not 0 <= normalized_ratio <= 1:
            raise CoreContractError("pruning ratio 必须位于 [0,1]")
        if self.strategy not in {"high", "low", "random"}:
            raise CoreContractError("strategy 必须为 high/low/random")
        if self.scope not in {"global", "layer_balanced"}:
            raise CoreContractError("scope 必须为 global/layer_balanced")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise CoreContractError("pruning seed 必须为整数")
        if self.score_view not in {"signed", "positive", "absolute"}:
            raise CoreContractError("score_view 必须为 signed/positive/absolute")
        object.__setattr__(self, "ratio", normalized_ratio)

    @property
    def digest(self) -> str:
        """返回纯选择合同的 canonical SHA-256；不包含运行时参数值。"""

        payload = {
            "schema_version": "pruning-plan-v1",
            "ratio": self.ratio,
            "strategy": self.strategy,
            "scope": self.scope,
            "seed": self.seed,
            "score_view": self.score_view,
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, object]:
        """返回严格、hash 绑定的 Stage 7 pruning plan artifact。"""

        return {
            "schema_version": "pruning-plan-v1",
            "ratio": self.ratio,
            "strategy": self.strategy,
            "scope": self.scope,
            "seed": self.seed,
            "score_view": self.score_view,
            "plan_hash": self.digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PruningPlan":
        """严格加载 plan；拒绝未知字段、bool 数值和摘要漂移。"""

        required = {
            "schema_version",
            "ratio",
            "strategy",
            "scope",
            "seed",
            "score_view",
            "plan_hash",
        }
        if set(value) != required:
            raise CoreContractError(
                "PruningPlan 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        if value["schema_version"] != "pruning-plan-v1":
            raise CoreContractError("不支持的 PruningPlan schema")
        ratio = value["ratio"]
        seed = value["seed"]
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise CoreContractError("PruningPlan ratio 必须是数值且不能是 bool")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise CoreContractError("PruningPlan seed 必须是整数且不能是 bool")
        for name in ("strategy", "scope", "score_view", "plan_hash"):
            if not isinstance(value[name], str):
                raise CoreContractError(f"PruningPlan {name} 必须是字符串")
        plan = cls(
            ratio=float(ratio),
            strategy=value["strategy"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            seed=seed,
            score_view=value["score_view"],  # type: ignore[arg-type]
        )
        if value["plan_hash"] != plan.digest:
            raise CoreContractError("PruningPlan plan_hash 与完整内容不一致")
        return plan


@dataclass(frozen=True, slots=True)
class CoordinateSelection:
    """按 canonical tensor name 分组的扁平坐标索引。"""

    indices: Mapping[str, tuple[int, ...]]
    selected_count: int
    eligible_count: int
    coordinate_ids: tuple[str, ...]


def canonical_coordinate_id(name: str, flat_index: int) -> str:
    """构造不依赖设备与 dtype 的稳定标量坐标 ID。"""

    return f"{name}#{flat_index:012d}"


def _rank_group(
    coordinates: list[tuple[float, str, str, int]],
    count: int,
    strategy: str,
    seed: int,
) -> list[tuple[float, str, str, int]]:
    if count == 0:
        return []
    if strategy == "high":
        return sorted(coordinates, key=lambda row: (-row[0], row[1]))[:count]
    if strategy == "low":
        return sorted(coordinates, key=lambda row: (row[0], row[1]))[:count]
    # random 使用稳定 canonical ID 排序后再抽样，避免 mapping 遍历顺序改变 RNG 结果。
    ordered = sorted(coordinates, key=lambda row: row[1])
    generator = random.Random(seed)
    chosen_indices = sorted(generator.sample(range(len(ordered)), count))
    return [ordered[index] for index in chosen_indices]


def select_pruned_coordinates(scores: TensorMap, plan: PruningPlan) -> CoordinateSelection:
    """按 plan 选择坐标；并列值永远由 canonical ID 决胜。"""

    scores.assert_finite()
    coordinates: list[tuple[float, str, str, int]] = []
    group_by_name: dict[str, str] = {}
    for name, tensor in scores.items():
        if scores.registry is not None:
            group = scores.registry.record(name).tags.get("layer", name)
        else:
            group = name.rsplit(".", 1)[0] if "." in name else name
        group_by_name[name] = group
        for index, value in enumerate(tensor.detach().to(torch.float64).reshape(-1).tolist()):
            coordinate_id = canonical_coordinate_id(name, index)
            coordinates.append((float(value), coordinate_id, name, index))
    eligible_count = len(coordinates)
    selected_count = math.floor(plan.ratio * eligible_count)

    if plan.scope == "global":
        selected = _rank_group(coordinates, selected_count, plan.strategy, plan.seed)
    else:
        groups: dict[str, list[tuple[float, str, str, int]]] = {}
        for row in coordinates:
            groups.setdefault(group_by_name[row[2]], []).append(row)
        base_counts = {group: math.floor(plan.ratio * len(rows)) for group, rows in groups.items()}
        remaining = selected_count - sum(base_counts.values())
        # Hamilton 最大余数法保证层平衡的总 K 与 global 合同完全一致。
        remainder_order = sorted(
            groups,
            key=lambda group: (-(plan.ratio * len(groups[group]) - base_counts[group]), group),
        )
        for group in remainder_order[:remaining]:
            base_counts[group] += 1
        selected = []
        for group in sorted(groups):
            group_seed_bytes = f"{plan.seed}:{group}".encode("utf-8")
            group_seed = int.from_bytes(hashlib.sha256(group_seed_bytes).digest()[:8], "big")
            selected.extend(
                _rank_group(groups[group], base_counts[group], plan.strategy, group_seed)
            )

    selected = sorted(selected, key=lambda row: row[1])
    indices: dict[str, list[int]] = {name: [] for name in scores}
    for _, _, name, index in selected:
        indices[name].append(index)
    return CoordinateSelection(
        indices={name: tuple(values) for name, values in indices.items()},
        selected_count=len(selected),
        eligible_count=eligible_count,
        coordinate_ids=tuple(row[1] for row in selected),
    )


class PruningContext:
    """进入时将选中坐标置零、退出时逐字节恢复参数的上下文管理器。"""

    def __init__(self, parameters: TensorMap, selection: CoordinateSelection) -> None:
        if tuple(parameters) != tuple(selection.indices):
            raise CoreContractError("PruningContext selection 与参数名称/顺序不一致")
        self.parameters = parameters
        self.selection = selection
        self._backup: dict[str, torch.Tensor] | None = None

    def __enter__(self) -> "PruningContext":
        if self._backup is not None:
            raise CoreContractError("同一个 PruningContext 不能重入")
        total_coordinates = sum(value.numel() for value in self.parameters.values())
        selected_ids: list[str] = []
        for name, indices in self.selection.indices.items():
            if len(indices) != len(set(indices)):
                raise CoreContractError(f"参数 {name!r} 的 pruning index 重复")
            for index in indices:
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or not 0 <= index < self.parameters[name].numel()
                ):
                    raise CoreContractError(f"参数 {name!r} 的 pruning index 越界")
                selected_ids.append(canonical_coordinate_id(name, index))
        if self.selection.eligible_count != total_coordinates:
            raise CoreContractError("PruningContext eligible_count 与参数坐标数不一致")
        if self.selection.selected_count != len(selected_ids):
            raise CoreContractError("PruningContext selected_count 与索引数不一致")
        if self.selection.coordinate_ids != tuple(sorted(selected_ids)):
            raise CoreContractError("PruningContext coordinate_ids 与选择索引不一致")
        self._backup = self.parameters.to_dict(clone=True)
        try:
            with torch.no_grad():
                for name, indices in self.selection.indices.items():
                    if not indices:
                        continue
                    flat = self.parameters[name].reshape(-1)
                    flat[torch.tensor(indices, dtype=torch.long, device=flat.device)] = 0
        except BaseException:
            # ``__enter__`` 抛错时 Python 不会调用 ``__exit__``；因此这里必须自行
            # 回滚，保证任何设备/索引异常都不会遗留半剪枝参数。
            with torch.no_grad():
                for name, original in self._backup.items():
                    self.parameters[name].copy_(original)
            self._backup = None
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        assert self._backup is not None
        with torch.no_grad():
            for name, original in self._backup.items():
                self.parameters[name].copy_(original)
        self._backup = None
