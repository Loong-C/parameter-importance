"""Stage 8 冻结单因素矩阵的可恢复执行器与确定性 reducer。

编译矩阵只证明“配置每次改变一个叶字段”；本模块进一步要求每个 cell 在执行前
通过严格配置验证，并把 baseline result hash 写入所有子结果。这样一个子实验不能
在换过 baseline 后继续冒充同一比较。昂贵 cell 使用与 Stage 7 相同的两阶段 JSON
commit store，重启后只恢复完整提交，orphan object 不会被当作成功。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from ..contracts.config import ResolvedConfig
from ..contracts.config_v2 import CONFIG_V2_SCHEMA_VERSION, ResolvedConfigV2
from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.jsonio import canonical_json_hash
from .ablation import AblationCell, AblationMatrix
from .pruning_runner import (
    CanonicalResultStore,
    EvaluationOutcome,
    _finite_metrics,
    _require_hash,
    _require_id,
)


@runtime_checkable
class AblationCellExecutor(Protocol):
    """执行一个严格 resolved cell 的外部训练/评测适配器协议。"""

    def execute(
        self,
        cell: AblationCell,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> EvaluationOutcome: ...


def _default_config_validator(
    value: Mapping[str, object],
) -> ResolvedConfig | ResolvedConfigV2:
    """把完整 cell 重新解析为严格 v1/v2 配置。

    消融矩阵中的 v2 配置刻意不保存 ``config_hash/full_hash``：二者是由配置
    内容派生的字段，而不是允许单独消融的科学变量。矩阵改变一个叶字段后，本函数
    会重新计算这两个摘要；若调用者传入的是完整 wire object，则仍使用
    :meth:`ResolvedConfigV2.from_mapping` 校验原摘要，绝不静默忽略陈旧 hash。
    """

    thawed = thaw_json_value(value)
    if not isinstance(thawed, dict):  # pragma: no cover - Mapping 输入的防御性断言
        raise TypeError("ablation cell config 必须解冻为 JSON object")
    mapping = thawed
    if mapping.get("schema_version") != CONFIG_V2_SCHEMA_VERSION:
        return ResolvedConfig.from_mapping(mapping)
    has_config_hash = "config_hash" in mapping
    has_full_hash = "full_hash" in mapping
    if has_config_hash is not has_full_hash:
        raise ValueError("resolved-config-v2 的两个派生 hash 必须同时存在或同时省略")
    if has_config_hash:
        return ResolvedConfigV2.from_mapping(mapping)
    # ``ResolvedConfigV2`` 构造器接收不含派生摘要的严格 payload；内部仍会完成
    # 全字段、跨字段、task catalog 与 formal/tiny 隔离验证。
    return ResolvedConfigV2(mapping)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AblationCellResult:
    """一个 baseline 或单因素 cell 的 hash 绑定结果。"""

    result_id: str
    matrix_id: str
    matrix_hash: str
    cell_id: str
    parent_cell_id: str | None
    parent_result_hash: str | None
    changed_factor: str | None
    changed_path: tuple[str, ...] | None
    config_hash: str
    seed: int
    executor_id: str
    metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    metric_directions: Mapping[str, str]
    deltas: Mapping[str, float]
    directed_effects: Mapping[str, float]
    scope: str
    formal_eligible: bool
    evidence_hash: str | None
    artifact_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "ablation-cell-result-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation-cell-result-v1":
            raise ValueError("AblationCellResult schema_version 不受支持")
        for name in ("result_id", "matrix_id", "cell_id", "executor_id"):
            _require_id(getattr(self, name), field_name=name)
        for name in ("matrix_hash", "config_hash", "artifact_hash"):
            _require_hash(getattr(self, name), field_name=name)
        if self.parent_result_hash is not None:
            _require_hash(self.parent_result_hash, field_name="parent_result_hash")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("ablation result scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("ablation result formal_eligible 与 scope 不一致")
        if self.scope == "formal":
            _require_hash(self.evidence_hash, field_name="evidence_hash")
        elif self.evidence_hash is not None:
            raise ValueError("fixture cell 不得携带 formal evidence")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("ablation seed 必须是非负整数")
        if self.parent_cell_id is None:
            if any(
                item is not None
                for item in (
                    self.parent_result_hash,
                    self.changed_factor,
                    self.changed_path,
                )
            ):
                raise ValueError("baseline cell 不能声明 parent/change")
        elif (
            self.parent_result_hash is None
            or self.changed_factor is None
            or self.changed_path is None
        ):
            raise ValueError("子 cell 必须绑定 parent result 与 changed factor/path")
        elif not self.changed_path or any(
            not isinstance(component, str) or not component
            for component in self.changed_path
        ):
            raise ValueError("changed_path 必须是非空字符串路径")
        metrics = _finite_metrics(self.metrics)
        baseline = _finite_metrics(self.baseline_metrics)
        deltas = _finite_metrics(self.deltas)
        effects = _finite_metrics(self.directed_effects)
        directions = dict(self.metric_directions)
        if not (
            set(metrics)
            == set(baseline)
            == set(deltas)
            == set(effects)
            == set(directions)
        ):
            raise ValueError("ablation metric 键集合不一致")
        if any(
            direction not in {"higher_is_better", "lower_is_better"}
            for direction in directions.values()
        ):
            raise ValueError("ablation metric direction 非法")
        expected_deltas = {
            name: metrics[name] - baseline[name] for name in metrics
        }
        expected_effects = {
            name: (
                expected_deltas[name]
                if directions[name] == "higher_is_better"
                else -expected_deltas[name]
            )
            for name in metrics
        }
        if dict(deltas) != expected_deltas or dict(effects) != expected_effects:
            raise ValueError("ablation delta/directed_effect 与 metrics 不一致")
        if self.parent_cell_id is None and (
            dict(metrics) != dict(baseline)
            or any(value != 0.0 for value in deltas.values())
        ):
            raise ValueError("ablation baseline metrics/deltas 不自洽")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "deltas", deltas)
        object.__setattr__(self, "directed_effects", effects)
        object.__setattr__(self, "metric_directions", MappingProxyType(directions))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("AblationCellResult artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "matrix_id": self.matrix_id,
            "matrix_hash": self.matrix_hash,
            "cell_id": self.cell_id,
            "parent_cell_id": self.parent_cell_id,
            "parent_result_hash": self.parent_result_hash,
            "changed_factor": self.changed_factor,
            "changed_path": None if self.changed_path is None else list(self.changed_path),
            "config_hash": self.config_hash,
            "seed": self.seed,
            "executor_id": self.executor_id,
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "metric_directions": dict(self.metric_directions),
            "deltas": dict(self.deltas),
            "directed_effects": dict(self.directed_effects),
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "evidence_hash": self.evidence_hash,
            "metadata": thaw_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(cls, **values: object) -> "AblationCellResult":
        wire_values = dict(values)
        if isinstance(wire_values.get("changed_path"), tuple):
            wire_values["changed_path"] = list(wire_values["changed_path"])
        payload = {"schema_version": "ablation-cell-result-v1", **wire_values}
        return cls(**values, artifact_hash=canonical_json_hash(payload))  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationCellResult":
        required = set(cls.__dataclass_fields__)
        if set(value) != required:
            raise ValueError("ABLATION_CELL_RESULT_FIELDS_MISMATCH")
        if value["schema_version"] != "ablation-cell-result-v1":
            raise ValueError("ABLATION_CELL_RESULT_SCHEMA_MISMATCH")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        changed_path = kwargs.get("changed_path")
        if changed_path is not None:
            if not isinstance(changed_path, list) or not all(
                isinstance(item, str) for item in changed_path
            ):
                raise TypeError("changed_path 必须是字符串数组或 null")
            kwargs["changed_path"] = tuple(changed_path)
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AblationStudyResult:
    """完整矩阵 reducer 输出，可直接转换为 Stage 9 源表行。"""

    matrix_id: str
    matrix_hash: str
    baseline_result_hash: str
    cell_result_hashes: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]
    scope: str
    formal_eligible: bool
    artifact_hash: str
    schema_version: str = "ablation-study-result-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation-study-result-v1":
            raise ValueError("AblationStudyResult schema_version 不受支持")
        _require_id(self.matrix_id, field_name="matrix_id")
        for value in (
            self.matrix_hash,
            self.baseline_result_hash,
            self.artifact_hash,
            *self.cell_result_hashes,
        ):
            _require_hash(value, field_name="ablation study hash")
        if not self.rows or not self.cell_result_hashes:
            raise ValueError("AblationStudyResult rows/hash 不能为空")
        if len(set(self.cell_result_hashes)) != len(self.cell_result_hashes):
            raise ValueError("AblationStudyResult cell hash 重复")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("AblationStudyResult scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("AblationStudyResult formal scope 不一致")
        frozen_rows = tuple(freeze_json_mapping(row) for row in self.rows)
        object.__setattr__(self, "rows", frozen_rows)
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("AblationStudyResult hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "matrix_hash": self.matrix_hash,
            "baseline_result_hash": self.baseline_result_hash,
            "cell_result_hashes": list(self.cell_result_hashes),
            "rows": [thaw_json_value(row) for row in self.rows],
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationStudyResult":
        """严格加载消融 reducer 的冻结行表。"""

        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("ABLATION_STUDY_RESULT_FIELDS_MISMATCH")
        hashes, rows = value["cell_result_hashes"], value["rows"]
        if not isinstance(hashes, list):
            raise TypeError("cell_result_hashes 必须是 array")
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise TypeError("AblationStudyResult rows 必须是 object array")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        kwargs["cell_result_hashes"] = tuple(hashes)
        kwargs["rows"] = tuple(rows)
        return cls(**kwargs, schema_version=value["schema_version"])  # type: ignore[arg-type]


class AblationResultReducer:
    """严格检查矩阵覆盖与 baseline lineage 后生成确定性行表。"""

    def reduce(
        self,
        matrix: AblationMatrix,
        results: Sequence[AblationCellResult],
    ) -> AblationStudyResult:
        by_cell = {result.cell_id: result for result in results}
        if len(by_cell) != len(results):
            raise ValueError("ablation results 包含重复 cell")
        expected = {cell.cell_id for cell in matrix.cells}
        if set(by_cell) != expected:
            raise ValueError("ablation results 未精确覆盖 frozen matrix")
        matrix_hash = matrix.digest
        if any(
            result.matrix_id != matrix.matrix_id or result.matrix_hash != matrix_hash
            for result in results
        ):
            raise ValueError("ablation result matrix identity 不一致")
        baseline = by_cell[matrix.baseline_cell_id]
        if baseline.parent_cell_id is not None:
            raise ValueError("baseline result 不是根")
        scopes = {result.scope for result in results}
        executors = {result.executor_id for result in results}
        if len(scopes) != 1 or len(executors) != 1:
            raise ValueError("ablation matrix 混合了 scope 或 executor")
        rows: list[Mapping[str, object]] = []
        for cell in sorted(matrix.cells, key=lambda item: item.cell_id):
            result = by_cell[cell.cell_id]
            if result.config_hash != cell.config_hash or result.seed != cell.seed:
                raise ValueError("cell result 与 frozen cell config/seed 不一致")
            if cell.cell_id != matrix.baseline_cell_id:
                if (
                    result.parent_cell_id != matrix.baseline_cell_id
                    or result.parent_result_hash != baseline.artifact_hash
                ):
                    raise ValueError("ablation child 未绑定当前 baseline result")
                if (
                    dict(result.baseline_metrics) != dict(baseline.metrics)
                    or dict(result.metric_directions)
                    != dict(baseline.metric_directions)
                ):
                    raise ValueError("ablation child 与 baseline metric lineage 不一致")
            for metric_name in sorted(result.metrics):
                rows.append(
                    {
                        "matrix_id": matrix.matrix_id,
                        "matrix_hash": matrix_hash,
                        "cell_id": cell.cell_id,
                        "parent_cell_id": cell.parent_cell_id,
                        "changed_factor": cell.changed_factor,
                        "changed_path": (
                            None if cell.changed_path is None else ".".join(cell.changed_path)
                        ),
                        "config_hash": cell.config_hash,
                        "seed": cell.seed,
                        "metric": metric_name,
                        "value": result.metrics[metric_name],
                        "baseline_value": result.baseline_metrics[metric_name],
                        "delta": result.deltas[metric_name],
                        "directed_effect": result.directed_effects[metric_name],
                        "direction": result.metric_directions[metric_name],
                        "cell_result_hash": result.artifact_hash,
                        "scope": result.scope,
                        "formal_eligible": result.formal_eligible,
                    }
                )
        ordered_results = tuple(by_cell[cell.cell_id] for cell in sorted(matrix.cells, key=lambda c: c.cell_id))
        payload = {
            "schema_version": "ablation-study-result-v1",
            "matrix_id": matrix.matrix_id,
            "matrix_hash": matrix_hash,
            "baseline_result_hash": baseline.artifact_hash,
            "cell_result_hashes": [result.artifact_hash for result in ordered_results],
            "rows": rows,
            "scope": next(iter(scopes)),
            "formal_eligible": next(iter(scopes)) == "formal",
        }
        return AblationStudyResult(
            matrix_id=matrix.matrix_id,
            matrix_hash=matrix_hash,
            baseline_result_hash=baseline.artifact_hash,
            cell_result_hashes=tuple(result.artifact_hash for result in ordered_results),
            rows=tuple(rows),
            scope=next(iter(scopes)),
            formal_eligible=next(iter(scopes)) == "formal",
            artifact_hash=canonical_json_hash(payload),
        )


class AblationMatrixRunner:
    """按 baseline-first 顺序执行或恢复一个冻结消融矩阵。"""

    def __init__(
        self,
        matrix: AblationMatrix,
        *,
        executor: AblationCellExecutor,
        result_root: str | Path,
        run_intent: str,
        config_validator: Callable[[Mapping[str, object]], object] | None = None,
        formal_authorization_hash: str | None = None,
    ) -> None:
        if not isinstance(executor, AblationCellExecutor):
            raise TypeError("executor 未实现 AblationCellExecutor")
        if run_intent not in {"local_fixture", "formal"}:
            raise ValueError("run_intent 只能是 local_fixture/formal")
        self.matrix = matrix
        self.executor = executor
        self.store = CanonicalResultStore(result_root)
        self.run_intent = run_intent
        self.matrix_hash = matrix.digest
        self.validator = config_validator or _default_config_validator
        if run_intent == "formal":
            self.formal_authorization_hash = _require_hash(
                formal_authorization_hash, field_name="formal_authorization_hash"
            )
        else:
            if formal_authorization_hash is not None:
                raise ValueError("fixture matrix 不得携带 formal authorization")
            self.formal_authorization_hash = None
        # 在执行任何 cell 前一次性严格验证全部配置，避免矩阵中途才发现非法项。
        resolved_configs: dict[str, object] = {}
        for cell in matrix.cells:
            resolved = self.validator(cell.config)
            if run_intent == "formal":
                if isinstance(resolved, ResolvedConfigV2):
                    if resolved.run_intent != "formal" or not resolved.formal_eligible:
                        raise ValueError("formal ablation cell 不是正式 resolved-config-v2")
                elif isinstance(resolved, ResolvedConfig):
                    identity = resolved.section("identity")
                    if (
                        identity["run_intent"] != "formal"
                        or identity["formal_eligible"] is not True
                    ):
                        raise ValueError("formal ablation cell 不是正式 resolved-config-v1")
                else:
                    raise TypeError(
                        "formal ablation 必须由严格 ResolvedConfig/ResolvedConfigV2 validator 返回"
                    )
            resolved_configs[cell.cell_id] = resolved
        self.resolved_configs = resolved_configs

    def _validate_outcome(self, outcome: EvaluationOutcome) -> None:
        if outcome.scope != self.run_intent:
            raise ValueError("ablation outcome scope 与 runner 不一致")
        if self.run_intent == "formal":
            _require_hash(outcome.evidence_hash, field_name="outcome evidence_hash")

    def _existing(self) -> dict[str, AblationCellResult]:
        results: dict[str, AblationCellResult] = {}
        for value in self.store.restore():
            result = AblationCellResult.from_mapping(value)
            if result.matrix_id != self.matrix.matrix_id or result.matrix_hash != self.matrix_hash:
                raise ValueError("ablation store 混入其他 matrix")
            if result.cell_id in results:
                raise ValueError("ablation store 出现重复 cell_id")
            results[result.cell_id] = result
        return results

    def _execute_cell(
        self,
        cell: AblationCell,
        baseline: AblationCellResult | None,
    ) -> AblationCellResult:
        outcome = self.executor.execute(
            cell,
            resolved_config=self.resolved_configs[cell.cell_id],
            context={
                "matrix_id": self.matrix.matrix_id,
                "matrix_hash": self.matrix_hash,
                "baseline_cell_id": self.matrix.baseline_cell_id,
                "parent_result_hash": None if baseline is None else baseline.artifact_hash,
                "formal_authorization_hash": self.formal_authorization_hash,
            },
        )
        self._validate_outcome(outcome)
        if baseline is None:
            baseline_metrics = dict(outcome.metrics)
            deltas = {name: 0.0 for name in outcome.metrics}
            effects = dict(deltas)
        else:
            if outcome.evaluator_id != baseline.executor_id:
                raise ValueError("ablation child 与 baseline executor 不一致")
            if dict(outcome.metric_directions) != dict(baseline.metric_directions):
                raise ValueError("ablation child 与 baseline metric directions 不一致")
            if set(outcome.metrics) != set(baseline.metrics):
                raise ValueError("ablation child 与 baseline metric 集合不一致")
            baseline_metrics = dict(baseline.metrics)
            deltas = {
                name: outcome.metrics[name] - baseline.metrics[name]
                for name in outcome.metrics
            }
            effects = {
                name: (
                    deltas[name]
                    if outcome.metric_directions[name] == "higher_is_better"
                    else -deltas[name]
                )
                for name in outcome.metrics
            }
        result_id = f"ablation-{canonical_json_hash({'matrix': self.matrix_hash, 'cell': cell.cell_id})[:24]}"
        result = AblationCellResult.create(
            result_id=result_id,
            matrix_id=self.matrix.matrix_id,
            matrix_hash=self.matrix_hash,
            cell_id=cell.cell_id,
            parent_cell_id=cell.parent_cell_id,
            parent_result_hash=None if baseline is None else baseline.artifact_hash,
            changed_factor=cell.changed_factor,
            changed_path=cell.changed_path,
            config_hash=cell.config_hash,
            seed=cell.seed,
            executor_id=outcome.evaluator_id,
            metrics=dict(outcome.metrics),
            baseline_metrics=baseline_metrics,
            metric_directions=dict(outcome.metric_directions),
            deltas=deltas,
            directed_effects=effects,
            scope=outcome.scope,
            formal_eligible=outcome.formal_eligible,
            evidence_hash=outcome.evidence_hash,
            metadata={"outcome": thaw_json_value(outcome.metadata)},
        )
        self.store.publish(result.result_id, result.to_dict())
        return result

    def run(self) -> AblationStudyResult:
        existing = self._existing()
        baseline_cell = next(
            cell for cell in self.matrix.cells if cell.cell_id == self.matrix.baseline_cell_id
        )
        baseline = existing.get(baseline_cell.cell_id)
        if baseline is None:
            baseline = self._execute_cell(baseline_cell, None)
        results = {baseline.cell_id: baseline}
        for cell in sorted(self.matrix.cells, key=lambda item: item.cell_id):
            if cell.cell_id == baseline.cell_id:
                continue
            result = existing.get(cell.cell_id)
            if result is None:
                result = self._execute_cell(cell, baseline)
            results[cell.cell_id] = result
        return AblationResultReducer().reduce(self.matrix, tuple(results.values()))


__all__ = [
    "AblationCellExecutor",
    "AblationCellResult",
    "AblationMatrixRunner",
    "AblationResultReducer",
    "AblationStudyResult",
]
