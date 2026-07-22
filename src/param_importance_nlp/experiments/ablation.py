"""Stage 8 单因素消融矩阵编译器。

矩阵只包含一个 baseline 和“每次只改变一个叶字段”的子单元，不生成全因子
笛卡尔积。每个子单元都显式指向 baseline lineage，并从独立 seed namespace
派生随机种子；执行器只能消费 ``frozen=True`` 的矩阵。
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _deep_freeze(value: object) -> object:
    """递归冻结 JSON-like 配置，避免嵌套修改与 config_hash 分离。"""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _get_path(config: Mapping[str, object], path: tuple[str, ...]) -> object:
    current: object = config
    for component in path:
        if not isinstance(current, Mapping) or component not in current:
            raise KeyError(".".join(path))
        current = current[component]
    return current


def _set_path(config: dict[str, object], path: tuple[str, ...], value: object) -> None:
    current = config
    for component in path[:-1]:
        child = current.get(component)
        if not isinstance(child, dict):
            raise KeyError(".".join(path))
        current = child
    current[path[-1]] = copy.deepcopy(value)


def _leaf_differences(
    left: object,
    right: object,
    prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences: set[tuple[str, ...]] = set()
        keys = set(left).union(right)
        for key in keys:
            component = str(key)
            if key not in left or key not in right:
                differences.add((*prefix, component))
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], (*prefix, component))
                )
        return differences
    return set() if left == right else {prefix}


@dataclass(frozen=True, slots=True)
class AblationFactor:
    """一个可消融的叶配置字段及其冻结候选。"""

    name: str
    config_path: tuple[str, ...]
    baseline_value: object
    alternatives: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.config_path or any(not part for part in self.config_path):
            raise ValueError("factor name 与 config_path 不能为空")
        if not self.alternatives:
            raise ValueError("AblationFactor 至少需要一个 alternative")
        encoded = {_canonical_json(value) for value in self.alternatives}
        if len(encoded) != len(self.alternatives):
            raise ValueError("alternatives 不得重复")
        baseline = _canonical_json(self.baseline_value)
        if baseline in encoded:
            raise ValueError("alternative 不能与 baseline_value 相同")


@dataclass(frozen=True, slots=True)
class AblationCell:
    """一个 baseline 或单因素子实验单元。"""

    cell_id: str
    parent_cell_id: str | None
    changed_factor: str | None
    changed_path: tuple[str, ...] | None
    config: Mapping[str, object]
    config_hash: str
    seed: int

    def __post_init__(self) -> None:
        if not self.cell_id or self.seed < 0:
            raise ValueError("cell_id 不能为空且 seed 不能为负")
        if self.parent_cell_id is None:
            if self.changed_factor is not None or self.changed_path is not None:
                raise ValueError("baseline cell 不能声明 changed_factor")
        elif self.changed_factor is None or self.changed_path is None:
            raise ValueError("子 cell 必须声明 changed_factor 与 changed_path")
        if _digest(self.config) != self.config_hash:
            raise ValueError("AblationCell config_hash 与 config 内容不一致")
        object.__setattr__(self, "config", _deep_freeze(copy.deepcopy(dict(self.config))))

    def to_dict(self) -> dict[str, object]:
        """返回包含完整单因素配置的可审计 wire object。"""

        return {
            "cell_id": self.cell_id,
            "parent_cell_id": self.parent_cell_id,
            "changed_factor": self.changed_factor,
            "changed_path": (
                None if self.changed_path is None else list(self.changed_path)
            ),
            "config": _json_ready(self.config),
            "config_hash": self.config_hash,
            "seed": self.seed,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationCell":
        """严格加载 cell，并重新计算 config hash。"""

        required = {
            "cell_id",
            "parent_cell_id",
            "changed_factor",
            "changed_path",
            "config",
            "config_hash",
            "seed",
        }
        if set(value) != required:
            raise ValueError(
                "AblationCell 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        for name in ("cell_id", "config_hash"):
            if not isinstance(value[name], str):
                raise TypeError(f"AblationCell {name} 必须是字符串")
        for name in ("parent_cell_id", "changed_factor"):
            if value[name] is not None and not isinstance(value[name], str):
                raise TypeError(f"AblationCell {name} 必须是字符串或 null")
        changed_path = value["changed_path"]
        if changed_path is not None and (
            not isinstance(changed_path, list)
            or not changed_path
            or not all(isinstance(item, str) and item for item in changed_path)
        ):
            raise TypeError("AblationCell changed_path 必须是非空字符串数组或 null")
        if not isinstance(value["config"], Mapping):
            raise TypeError("AblationCell config 必须是 object")
        seed = value["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("AblationCell seed 必须是整数且不能是 bool")
        return cls(
            cell_id=value["cell_id"],
            parent_cell_id=value["parent_cell_id"],
            changed_factor=value["changed_factor"],
            changed_path=(
                None if changed_path is None else tuple(changed_path)
            ),
            config=value["config"],
            config_hash=value["config_hash"],
            seed=seed,
        )


@dataclass(frozen=True, slots=True)
class AblationMatrix:
    """冻结的 baseline + 单因素子单元矩阵。"""

    matrix_id: str
    baseline_cell_id: str
    cells: tuple[AblationCell, ...]
    seed_namespace: str
    frozen: bool = True
    schema_version: str = "ablation-matrix-v1"

    def __post_init__(self) -> None:
        if not self.matrix_id or not self.seed_namespace:
            raise ValueError("matrix_id 与 seed_namespace 不能为空")
        if self.schema_version != "ablation-matrix-v1":
            raise ValueError("不支持的 AblationMatrix 版本")
        if not self.frozen:
            raise ValueError("执行层只接受 frozen AblationMatrix")
        if not self.cells or len({cell.cell_id for cell in self.cells}) != len(self.cells):
            raise ValueError("cells 不能为空且 cell_id 必须唯一")
        baseline = [cell for cell in self.cells if cell.cell_id == self.baseline_cell_id]
        if len(baseline) != 1 or baseline[0].parent_cell_id is not None:
            raise ValueError("baseline_cell_id 必须唯一指向根 cell")
        for cell in self.cells:
            if cell.cell_id == self.baseline_cell_id:
                continue
            if cell.parent_cell_id != self.baseline_cell_id:
                raise ValueError("每个消融子 cell 必须直接继承 baseline")
            differences = _leaf_differences(baseline[0].config, cell.config)
            if differences != {cell.changed_path}:
                raise ValueError(
                    f"cell {cell.cell_id!r} 不是严格单因素变化: differences={differences}"
                )

    @classmethod
    def compile(
        cls,
        *,
        matrix_id: str,
        base_config: Mapping[str, object],
        factors: Sequence[AblationFactor],
        base_seed: int,
        seed_namespace: str = "stage8-ablation-v1",
    ) -> "AblationMatrix":
        """确定性编译 baseline 与每个 alternative 的单因素单元。"""

        if base_seed < 0:
            raise ValueError("base_seed 不能为负")
        if not factors:
            raise ValueError("factors 不能为空")
        if len({factor.name for factor in factors}) != len(factors):
            raise ValueError("factor name 必须唯一")
        if len({factor.config_path for factor in factors}) != len(factors):
            raise ValueError("不同 factor 不能指向同一个 config path")
        base = copy.deepcopy(dict(base_config))
        for factor in factors:
            actual = _get_path(base, factor.config_path)
            if _canonical_json(actual) != _canonical_json(factor.baseline_value):
                raise ValueError(
                    f"factor {factor.name!r} 的 baseline_value 与 base_config 不一致"
                )
        base_hash = _digest(base)
        baseline_id = f"ablation-baseline-{base_hash[:16]}"
        cells: list[AblationCell] = [
            AblationCell(
                cell_id=baseline_id,
                parent_cell_id=None,
                changed_factor=None,
                changed_path=None,
                config=base,
                config_hash=base_hash,
                seed=_derive_seed(base_seed, seed_namespace, "baseline", base_hash),
            )
        ]
        for factor in sorted(factors, key=lambda item: item.name):
            for alternative in sorted(factor.alternatives, key=_canonical_json):
                config = copy.deepcopy(base)
                _set_path(config, factor.config_path, alternative)
                differences = _leaf_differences(base, config)
                if differences != {factor.config_path}:
                    raise ValueError(f"factor {factor.name!r} 不是单一叶字段变化")
                config_hash = _digest(config)
                identity = {
                    "matrix_id": matrix_id,
                    "parent": baseline_id,
                    "factor": factor.name,
                    "path": list(factor.config_path),
                    "value": alternative,
                    "config_hash": config_hash,
                }
                cell_id = f"ablation-{_digest(identity)[:20]}"
                cells.append(
                    AblationCell(
                        cell_id=cell_id,
                        parent_cell_id=baseline_id,
                        changed_factor=factor.name,
                        changed_path=factor.config_path,
                        config=config,
                        config_hash=config_hash,
                        seed=_derive_seed(
                            base_seed,
                            seed_namespace,
                            factor.name,
                            _digest(alternative),
                        ),
                    )
                )
        return cls(
            matrix_id=matrix_id,
            baseline_cell_id=baseline_id,
            cells=tuple(cells),
            seed_namespace=seed_namespace,
        )

    @property
    def digest(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "matrix_id": self.matrix_id,
                "baseline_cell_id": self.baseline_cell_id,
                "seed_namespace": self.seed_namespace,
                "frozen": self.frozen,
                "cells": [
                    {
                        "cell_id": cell.cell_id,
                        "parent_cell_id": cell.parent_cell_id,
                        "changed_factor": cell.changed_factor,
                        "changed_path": cell.changed_path,
                        "config_hash": cell.config_hash,
                        "seed": cell.seed,
                    }
                    for cell in self.cells
                ],
            }
        )

    def to_dict(self) -> dict[str, object]:
        """返回冻结矩阵；完整 cell 配置由各自 config_hash 绑定。"""

        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "baseline_cell_id": self.baseline_cell_id,
            "cells": [cell.to_dict() for cell in self.cells],
            "seed_namespace": self.seed_namespace,
            "frozen": self.frozen,
            "matrix_hash": self.digest,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationMatrix":
        """严格加载并复验单因素差异、父子 lineage 与矩阵摘要。"""

        required = {
            "schema_version",
            "matrix_id",
            "baseline_cell_id",
            "cells",
            "seed_namespace",
            "frozen",
            "matrix_hash",
        }
        if set(value) != required:
            raise ValueError(
                "AblationMatrix 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        for name in (
            "schema_version",
            "matrix_id",
            "baseline_cell_id",
            "seed_namespace",
            "matrix_hash",
        ):
            if not isinstance(value[name], str):
                raise TypeError(f"AblationMatrix {name} 必须是字符串")
        if not isinstance(value["frozen"], bool):
            raise TypeError("AblationMatrix frozen 必须是布尔值")
        if not isinstance(value["cells"], list) or not all(
            isinstance(item, Mapping) for item in value["cells"]
        ):
            raise TypeError("AblationMatrix cells 必须是 object 数组")
        matrix = cls(
            matrix_id=value["matrix_id"],
            baseline_cell_id=value["baseline_cell_id"],
            cells=tuple(AblationCell.from_mapping(item) for item in value["cells"]),
            seed_namespace=value["seed_namespace"],
            frozen=value["frozen"],
            schema_version=value["schema_version"],
        )
        if value["matrix_hash"] != matrix.digest:
            raise ValueError("AblationMatrix matrix_hash 与完整矩阵内容不一致")
        return matrix


def _derive_seed(base_seed: int, namespace: str, *parts: str) -> int:
    payload = {
        "base_seed": base_seed,
        "namespace": namespace,
        "parts": list(parts),
    }
    return int.from_bytes(hashlib.sha256(_canonical_json(payload)).digest()[:8], "big")
