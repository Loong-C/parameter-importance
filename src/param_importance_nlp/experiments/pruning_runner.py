"""Stage 7 可恢复剪枝评测执行器。

本模块把 :mod:`experiments.pruning` 产生的运行单元真正连接到参数张量和任务
评估器。执行器本身不认识 Hugging Face，也不假设“指标越大越好”：评估器必须
为每个标量声明方向，执行器据此把性能变化统一转换为“越大表示损伤越严重”。

恢复协议采用不可变 result object + 独立权威 commit：object 单独存在只算 orphan，
只有 commit 与 object 都通过 canonical hash 复核后才会被恢复。已提交单元再次运行
时不会调用 evaluator，从而适合昂贵的语言模型/下游评测。formal 路径必须同时有
正式 study source、授权 hash，以及 evaluator 返回的正式证据；任何 fixture 结果都
不能通过参数开关升级为正式结果。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from ..core.pruning import PruningContext, PruningPlan, select_pruned_coordinates
from ..core.tensors import TensorMap
from .pruning import (
    IMPORTANCE_METHODS,
    ImportanceSourceSpec,
    PruningRunSpec,
    PruningStudySpec,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,191}$")


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _require_id(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} 不是安全 ID")
    return value


def _finite_metrics(values: Mapping[str, object]) -> Mapping[str, float]:
    if not values:
        raise ValueError("evaluation metrics 不能为空")
    normalized: dict[str, float] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            raise TypeError("metric name 必须是非空字符串")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"metric {name!r} 必须是数值")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"metric {name!r} 包含非有限数")
        normalized[name] = number
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """一次未剪枝或剪枝后模型评估的结构化结果。

    ``metric_directions`` 的值只能是 ``higher_is_better`` 或
    ``lower_is_better``。formal outcome 必须携带由外部资产/Gate 体系产生的
    ``evidence_hash``；本类只验证身份，不自行伪造正式证据。
    """

    evaluator_id: str
    metrics: Mapping[str, float]
    metric_directions: Mapping[str, str]
    scope: str
    formal_eligible: bool
    evidence_hash: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.evaluator_id, field_name="evaluator_id")
        metrics = _finite_metrics(self.metrics)
        directions = dict(self.metric_directions)
        if set(directions) != set(metrics):
            raise ValueError("metric_directions 必须与 metrics 具有相同键集合")
        if any(
            value not in {"higher_is_better", "lower_is_better"}
            for value in directions.values()
        ):
            raise ValueError("metric direction 只能是 higher_is_better/lower_is_better")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("outcome scope 只能是 local_fixture/formal")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("formal_eligible 必须严格等于 (scope == 'formal')")
        if self.scope == "formal":
            _require_hash(self.evidence_hash, field_name="formal evidence_hash")
        elif self.evidence_hash is not None:
            raise ValueError("local_fixture outcome 不得携带 formal evidence_hash")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "metric_directions", MappingProxyType(directions))
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))


@runtime_checkable
class PruningEvaluator(Protocol):
    """真实任务评估适配器协议；实现可以封装 LM 或分类任务。"""

    def evaluate(
        self,
        parameters: TensorMap,
        *,
        run_id: str,
        context: Mapping[str, object],
    ) -> EvaluationOutcome: ...


def pruning_study_hash(study: PruningStudySpec) -> str:
    """计算包含 score view 和全部编译单元的稳定研究身份。"""

    payload = {
        "study_id": study.study_id,
        "run_intent": study.run_intent,
        "frozen": study.frozen,
        "ratios": list(study.ratios),
        "pruning_scopes": list(study.pruning_scopes),
        "random_mask_seeds": list(study.random_mask_seeds),
        "sources": [
            {
                "method": source.method,
                "artifact_id": source.artifact_id,
                "artifact_hash": source.artifact_hash,
                "coordinate_registry_hash": source.coordinate_registry_hash,
                "score_view": source.score_view,
                "scope": source.scope,
                "available": source.available,
                "metadata": thaw_json_value(source.metadata),
            }
            for source in sorted(
                study.sources,
                key=lambda item: (item.method, item.artifact_hash, item.score_view),
            )
        ],
        "runs": [
            {
                "run_id": run.run_id,
                "method": run.method,
                "direction": run.direction,
                "pruning_scope": run.pruning_scope,
                "ratio": run.ratio,
                "mask_seed": run.mask_seed,
                "source_artifact_hash": run.source_artifact_hash,
                "coordinate_registry_hash": run.coordinate_registry_hash,
                "tie_breaker": run.tie_breaker,
            }
            for run in study.compile()
        ],
    }
    return canonical_json_hash(payload)


@dataclass(frozen=True, slots=True)
class PruningEvaluationResult:
    """一个 baseline 或剪枝运行单元的不可变结果 artifact。"""

    result_id: str
    study_id: str
    study_hash: str
    run_id: str
    model_checkpoint_hash: str
    coordinate_registry_hash: str
    evaluator_id: str
    method: str
    direction: str
    pruning_scope: str
    ratio: float
    mask_seed: int | None
    source_artifact_hash: str | None
    score_view: str | None
    selected_count: int
    eligible_count: int
    metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    damage: Mapping[str, float]
    metric_directions: Mapping[str, str]
    scope: str
    formal_eligible: bool
    evidence_hash: str | None
    artifact_hash: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "pruning-evaluation-result-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pruning-evaluation-result-v1":
            raise ValueError("PruningEvaluationResult schema_version 不受支持")
        for name in ("result_id", "study_id", "run_id", "evaluator_id"):
            _require_id(getattr(self, name), field_name=name)
        for name in (
            "study_hash",
            "model_checkpoint_hash",
            "coordinate_registry_hash",
            "artifact_hash",
        ):
            _require_hash(getattr(self, name), field_name=name)
        if self.source_artifact_hash is not None:
            _require_hash(self.source_artifact_hash, field_name="source_artifact_hash")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("result scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("result formal_eligible 与 scope 不一致")
        if self.scope == "formal":
            _require_hash(self.evidence_hash, field_name="evidence_hash")
        elif self.evidence_hash is not None:
            raise ValueError("fixture result 不得携带 formal evidence")
        if (
            isinstance(self.selected_count, bool)
            or not isinstance(self.selected_count, int)
            or isinstance(self.eligible_count, bool)
            or not isinstance(self.eligible_count, int)
            or self.selected_count < 0
            or self.eligible_count < 0
            or self.selected_count > self.eligible_count
        ):
            raise ValueError("selected/eligible count 非法")
        if not math.isfinite(self.ratio) or not 0 <= self.ratio <= 1:
            raise ValueError("ratio 必须位于 [0,1]")
        metrics = _finite_metrics(self.metrics)
        baseline = _finite_metrics(self.baseline_metrics)
        damage = _finite_metrics(self.damage)
        directions = MappingProxyType(dict(self.metric_directions))
        if not (set(metrics) == set(baseline) == set(damage) == set(directions)):
            raise ValueError("metrics/baseline/damage/directions 键集合不一致")
        if any(
            value not in {"higher_is_better", "lower_is_better"}
            for value in directions.values()
        ):
            raise ValueError("metric direction 非法")
        if self.method == "baseline":
            if (
                self.direction != "none"
                or self.pruning_scope != "none"
                or self.ratio != 0.0
                or self.mask_seed is not None
                or self.source_artifact_hash is not None
                or self.score_view is not None
                or self.selected_count != 0
                or dict(metrics) != dict(baseline)
                or any(value != 0.0 for value in damage.values())
            ):
                raise ValueError("baseline pruning result 字段不自洽")
        else:
            if self.method not in (*IMPORTANCE_METHODS, "random"):
                raise ValueError("pruning result method 非法")
            if self.direction not in {"high", "low", "random"}:
                raise ValueError("pruning result direction 非法")
            if self.pruning_scope not in {"global", "layer_balanced"}:
                raise ValueError("pruning result scope 非法")
            if self.direction == "random":
                if (
                    self.method != "random"
                    or isinstance(self.mask_seed, bool)
                    or not isinstance(self.mask_seed, int)
                    or self.mask_seed < 0
                    or self.source_artifact_hash is not None
                    or self.score_view is not None
                ):
                    raise ValueError("random pruning result 字段不自洽")
            elif (
                self.method in {"baseline", "random"}
                or self.mask_seed is not None
                or self.source_artifact_hash is None
                or self.score_view not in {"signed", "positive", "absolute"}
            ):
                raise ValueError("非随机 pruning result 必须绑定 score source/view")
            expected_damage = {
                name: (
                    baseline[name] - metrics[name]
                    if directions[name] == "higher_is_better"
                    else metrics[name] - baseline[name]
                )
                for name in metrics
            }
            if dict(damage) != expected_damage:
                raise ValueError("pruning damage 与 metric direction 不一致")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "baseline_metrics", baseline)
        object.__setattr__(self, "damage", damage)
        object.__setattr__(self, "metric_directions", directions)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("PruningEvaluationResult artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "result_id": self.result_id,
            "study_id": self.study_id,
            "study_hash": self.study_hash,
            "run_id": self.run_id,
            "model_checkpoint_hash": self.model_checkpoint_hash,
            "coordinate_registry_hash": self.coordinate_registry_hash,
            "evaluator_id": self.evaluator_id,
            "method": self.method,
            "direction": self.direction,
            "pruning_scope": self.pruning_scope,
            "ratio": self.ratio,
            "mask_seed": self.mask_seed,
            "source_artifact_hash": self.source_artifact_hash,
            "score_view": self.score_view,
            "selected_count": self.selected_count,
            "eligible_count": self.eligible_count,
            "metrics": dict(self.metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "damage": dict(self.damage),
            "metric_directions": dict(self.metric_directions),
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "evidence_hash": self.evidence_hash,
            "metadata": thaw_json_value(self.metadata),
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(cls, **values: object) -> "PruningEvaluationResult":
        payload = {
            "schema_version": "pruning-evaluation-result-v1",
            **values,
        }
        return cls(**values, artifact_hash=canonical_json_hash(payload))  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PruningEvaluationResult":
        required = set(cls.__dataclass_fields__) - {"schema_version"}
        required.add("schema_version")
        if set(value) != required:
            raise ValueError("PRUNING_RESULT_FIELDS_MISMATCH")
        if value["schema_version"] != "pruning-evaluation-result-v1":
            raise ValueError("PRUNING_RESULT_SCHEMA_MISMATCH")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        return cls(**kwargs)  # type: ignore[arg-type]


class CanonicalResultStore:
    """小型 JSON 结果的两阶段、可恢复 commit store。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)

    def _paths(self, result_id: str) -> tuple[Path, Path]:
        safe = _require_id(result_id, field_name="result_id")
        return self.objects / f"{safe}.json", self.commits / f"{safe}.json"

    @staticmethod
    def _validate_object(value: object) -> dict[str, object]:
        if not isinstance(value, dict):
            raise ValueError("RESULT_OBJECT_NOT_MAPPING")
        artifact_hash = value.get("artifact_hash")
        _require_hash(artifact_hash, field_name="artifact_hash")
        payload = dict(value)
        payload.pop("artifact_hash")
        if canonical_json_hash(payload) != artifact_hash:
            raise ValueError("RESULT_OBJECT_HASH_MISMATCH")
        return value

    def publish(self, result_id: str, value: Mapping[str, object]) -> bool:
        object_path, commit_path = self._paths(result_id)
        normalized = self._validate_object(dict(value))
        if commit_path.exists():
            existing = self.load(result_id)
            if existing != normalized:
                raise ValueError("RESULT_IDEMPOTENCE_CONFLICT")
            return False
        if object_path.exists():
            existing_object = self._validate_object(load_canonical_json(object_path))
            if existing_object != normalized:
                raise ValueError("RESULT_ORPHAN_OBJECT_CONFLICT")
        else:
            write_canonical_json(object_path, normalized)
        verified = self._validate_object(load_canonical_json(object_path))
        object_hash = canonical_json_hash(verified)
        commit_payload = {
            "schema_version": "canonical-result-commit-v1",
            "result_id": result_id,
            "object_file": object_path.name,
            "object_hash": object_hash,
            "artifact_hash": verified["artifact_hash"],
        }
        commit = commit_payload | {"commit_hash": canonical_json_hash(commit_payload)}
        write_canonical_json(commit_path, commit)
        self.load(result_id)
        return True

    def load(self, result_id: str) -> dict[str, object]:
        object_path, commit_path = self._paths(result_id)
        commit = load_canonical_json(commit_path)
        if not isinstance(commit, dict) or set(commit) != {
            "schema_version",
            "result_id",
            "object_file",
            "object_hash",
            "artifact_hash",
            "commit_hash",
        }:
            raise ValueError("RESULT_COMMIT_FIELDS_MISMATCH")
        if commit["schema_version"] != "canonical-result-commit-v1":
            raise ValueError("RESULT_COMMIT_SCHEMA_MISMATCH")
        payload = dict(commit)
        observed_commit_hash = payload.pop("commit_hash")
        if canonical_json_hash(payload) != observed_commit_hash:
            raise ValueError("RESULT_COMMIT_HASH_MISMATCH")
        if commit["result_id"] != result_id or commit["object_file"] != object_path.name:
            raise ValueError("RESULT_COMMIT_IDENTITY_MISMATCH")
        value = self._validate_object(load_canonical_json(object_path))
        if canonical_json_hash(value) != commit["object_hash"]:
            raise ValueError("RESULT_COMMIT_OBJECT_HASH_MISMATCH")
        if value["artifact_hash"] != commit["artifact_hash"]:
            raise ValueError("RESULT_COMMIT_ARTIFACT_HASH_MISMATCH")
        return value

    def restore(self) -> tuple[dict[str, object], ...]:
        return tuple(self.load(path.stem) for path in sorted(self.commits.glob("*.json")))

    def reconcile(self) -> dict[str, object]:
        valid: list[str] = []
        invalid: list[dict[str, str]] = []
        for path in sorted(self.commits.glob("*.json")):
            try:
                self.load(path.stem)
            except Exception as exc:
                invalid.append({"result_id": path.stem, "reason": str(exc)})
            else:
                valid.append(path.stem)
        committed_objects = {f"{name}.json" for name in valid}
        return {
            "schema_version": "canonical-result-reconcile-v1",
            "valid_result_ids": valid,
            "invalid_commits": invalid,
            "orphan_objects": sorted(
                path.name for path in self.objects.glob("*.json") if path.name not in committed_objects
            ),
        }


@dataclass(frozen=True, slots=True)
class PruningStudyResult:
    """按 run ID 排序的完整剪枝研究结果。"""

    study_id: str
    study_hash: str
    result_hashes: tuple[str, ...]
    result_ids: tuple[str, ...]
    scope: str
    formal_eligible: bool
    artifact_hash: str
    schema_version: str = "pruning-study-result-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "pruning-study-result-v1":
            raise ValueError("PruningStudyResult schema_version 不受支持")
        _require_id(self.study_id, field_name="study_id")
        _require_hash(self.study_hash, field_name="study_hash")
        if len(self.result_hashes) != len(self.result_ids) or not self.result_ids:
            raise ValueError("study result ids/hashes 数量不一致或为空")
        if len(set(self.result_ids)) != len(self.result_ids):
            raise ValueError("study result IDs 重复")
        for value in self.result_hashes:
            _require_hash(value, field_name="result_hash")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("study result scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("study formal_eligible 与 scope 不一致")
        _require_hash(self.artifact_hash, field_name="artifact_hash")
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("PruningStudyResult hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "study_hash": self.study_hash,
            "result_ids": list(self.result_ids),
            "result_hashes": list(self.result_hashes),
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "PruningStudyResult":
        """严格加载剪枝研究汇总 artifact。"""

        required = set(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("PRUNING_STUDY_RESULT_FIELDS_MISMATCH")
        result_ids, result_hashes = value["result_ids"], value["result_hashes"]
        if not isinstance(result_ids, list) or not isinstance(result_hashes, list):
            raise TypeError("result_ids/result_hashes 必须是 array")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        kwargs["result_ids"] = tuple(result_ids)
        kwargs["result_hashes"] = tuple(result_hashes)
        return cls(**kwargs, schema_version=value["schema_version"])  # type: ignore[arg-type]


class PruningStudyRunner:
    """执行并恢复冻结的 :class:`PruningStudySpec`。"""

    def __init__(
        self,
        study: PruningStudySpec,
        *,
        parameters: TensorMap,
        scores_by_artifact_hash: Mapping[str, TensorMap],
        evaluator: PruningEvaluator,
        result_root: str | Path,
        model_checkpoint_hash: str,
        coordinate_registry_hash: str,
        formal_authorization_hash: str | None = None,
    ) -> None:
        if not isinstance(evaluator, PruningEvaluator):
            raise TypeError("evaluator 未实现 PruningEvaluator")
        self.study = study
        self.parameters = parameters
        self.scores = dict(scores_by_artifact_hash)
        self.evaluator = evaluator
        self.store = CanonicalResultStore(result_root)
        self.model_checkpoint_hash = _require_hash(
            model_checkpoint_hash, field_name="model_checkpoint_hash"
        )
        self.coordinate_registry_hash = _require_hash(
            coordinate_registry_hash, field_name="coordinate_registry_hash"
        )
        if coordinate_registry_hash != study.coordinate_registry_hash:
            raise ValueError("runner coordinate registry 与 pruning study 不一致")
        if study.run_intent == "formal":
            self.formal_authorization_hash = _require_hash(
                formal_authorization_hash, field_name="formal_authorization_hash"
            )
        else:
            if formal_authorization_hash is not None:
                raise ValueError("local fixture runner 不得携带 formal authorization")
            self.formal_authorization_hash = None
        expected_sources = {source.artifact_hash for source in study.sources}
        if set(self.scores) != expected_sources:
            raise ValueError("scores_by_artifact_hash 必须精确覆盖全部 study sources")
        for score in self.scores.values():
            parameters.assert_compatible(score)
            score.assert_finite()
        self.study_hash = pruning_study_hash(study)

    def _validate_outcome(self, outcome: EvaluationOutcome) -> None:
        expected_scope = self.study.run_intent
        if outcome.scope != expected_scope:
            raise ValueError("evaluator outcome scope 与 study run_intent 不一致")
        if expected_scope == "formal":
            assert self.formal_authorization_hash is not None
            _require_hash(outcome.evidence_hash, field_name="outcome evidence_hash")

    def _existing(self) -> dict[str, PruningEvaluationResult]:
        restored: dict[str, PruningEvaluationResult] = {}
        expected_runs = {run.run_id: run for run in self.study.compile()}
        baseline_id = f"baseline-{self.study_hash[:20]}"
        for value in self.store.restore():
            result = PruningEvaluationResult.from_mapping(value)
            if result.study_hash != self.study_hash:
                raise ValueError("result store 混入其他 pruning study")
            if result.study_id != self.study.study_id:
                raise ValueError("result store study_id 不一致")
            if result.model_checkpoint_hash != self.model_checkpoint_hash:
                raise ValueError("result store checkpoint hash 不一致")
            if result.coordinate_registry_hash != self.coordinate_registry_hash:
                raise ValueError("result store coordinate registry 不一致")
            if result.scope != self.study.run_intent:
                raise ValueError("result store scope 不一致")
            if result.run_id != baseline_id:
                run = expected_runs.get(result.run_id)
                if run is None:
                    raise ValueError("result store 包含未编译 pruning run")
                source = self._source_for(run)
                if (
                    result.method != run.method
                    or result.direction != run.direction
                    or result.pruning_scope != run.pruning_scope
                    or result.ratio != run.ratio
                    or result.mask_seed != run.mask_seed
                    or result.source_artifact_hash != run.source_artifact_hash
                    or result.score_view != (None if source is None else source.score_view)
                ):
                    raise ValueError("restored pruning result 与 frozen run spec 不一致")
            if result.run_id in restored:
                raise ValueError("result store 出现重复 run_id")
            restored[result.run_id] = result
        return restored

    def _evaluate_baseline(self) -> PruningEvaluationResult:
        run_id = f"baseline-{self.study_hash[:20]}"
        outcome = self.evaluator.evaluate(
            self.parameters,
            run_id=run_id,
            context={
                "study_id": self.study.study_id,
                "study_hash": self.study_hash,
                "condition": "baseline",
            },
        )
        self._validate_outcome(outcome)
        zero_damage = {name: 0.0 for name in outcome.metrics}
        result = PruningEvaluationResult.create(
            result_id=run_id,
            study_id=self.study.study_id,
            study_hash=self.study_hash,
            run_id=run_id,
            model_checkpoint_hash=self.model_checkpoint_hash,
            coordinate_registry_hash=self.coordinate_registry_hash,
            evaluator_id=outcome.evaluator_id,
            method="baseline",
            direction="none",
            pruning_scope="none",
            ratio=0.0,
            mask_seed=None,
            source_artifact_hash=None,
            score_view=None,
            selected_count=0,
            eligible_count=sum(value.numel() for value in self.parameters.values()),
            metrics=dict(outcome.metrics),
            baseline_metrics=dict(outcome.metrics),
            damage=zero_damage,
            metric_directions=dict(outcome.metric_directions),
            scope=outcome.scope,
            formal_eligible=outcome.formal_eligible,
            evidence_hash=outcome.evidence_hash,
            metadata={"outcome": thaw_json_value(outcome.metadata)},
        )
        self.store.publish(result.result_id, result.to_dict())
        return result

    def _source_for(self, run: PruningRunSpec) -> ImportanceSourceSpec | None:
        if run.source_artifact_hash is None:
            return None
        matches = [
            source
            for source in self.study.sources
            if source.artifact_hash == run.source_artifact_hash and source.method == run.method
        ]
        if len(matches) != 1:
            raise ValueError("pruning run 无法唯一解析 importance source/score view")
        return matches[0]

    def _run_one(
        self,
        run: PruningRunSpec,
        baseline: PruningEvaluationResult,
    ) -> PruningEvaluationResult:
        source = self._source_for(run)
        score = (
            TensorMap.zeros_like(self.parameters, dtype=torch.float64)
            if source is None
            else self.scores[source.artifact_hash]
        )
        plan = PruningPlan(
            ratio=run.ratio,
            strategy=run.direction,
            scope=run.pruning_scope,
            seed=0 if run.mask_seed is None else run.mask_seed,
            score_view="absolute" if source is None else source.score_view,
        )
        selection = select_pruned_coordinates(score, plan)
        context = {
            "study_id": self.study.study_id,
            "study_hash": self.study_hash,
            "method": run.method,
            "direction": run.direction,
            "pruning_scope": run.pruning_scope,
            "ratio": run.ratio,
            "mask_seed": run.mask_seed,
            "source_artifact_hash": run.source_artifact_hash,
            "score_view": None if source is None else source.score_view,
            "selected_count": selection.selected_count,
            "eligible_count": selection.eligible_count,
        }
        with PruningContext(self.parameters, selection):
            outcome = self.evaluator.evaluate(
                self.parameters,
                run_id=run.run_id,
                context=context,
            )
        self._validate_outcome(outcome)
        if outcome.evaluator_id != baseline.evaluator_id:
            raise ValueError("baseline 与 pruning run evaluator_id 不一致")
        if dict(outcome.metric_directions) != dict(baseline.metric_directions):
            raise ValueError("baseline 与 pruning run metric directions 不一致")
        if set(outcome.metrics) != set(baseline.metrics):
            raise ValueError("baseline 与 pruning run metric 集合不一致")
        damage = {
            name: (
                baseline.metrics[name] - outcome.metrics[name]
                if outcome.metric_directions[name] == "higher_is_better"
                else outcome.metrics[name] - baseline.metrics[name]
            )
            for name in outcome.metrics
        }
        result = PruningEvaluationResult.create(
            result_id=run.run_id,
            study_id=self.study.study_id,
            study_hash=self.study_hash,
            run_id=run.run_id,
            model_checkpoint_hash=self.model_checkpoint_hash,
            coordinate_registry_hash=self.coordinate_registry_hash,
            evaluator_id=outcome.evaluator_id,
            method=run.method,
            direction=run.direction,
            pruning_scope=run.pruning_scope,
            ratio=run.ratio,
            mask_seed=run.mask_seed,
            source_artifact_hash=run.source_artifact_hash,
            score_view=None if source is None else source.score_view,
            selected_count=selection.selected_count,
            eligible_count=selection.eligible_count,
            metrics=dict(outcome.metrics),
            baseline_metrics=dict(baseline.metrics),
            damage=damage,
            metric_directions=dict(outcome.metric_directions),
            scope=outcome.scope,
            formal_eligible=outcome.formal_eligible,
            evidence_hash=outcome.evidence_hash,
            metadata={
                "coordinate_ids_hash": canonical_json_hash(list(selection.coordinate_ids)),
                "outcome": thaw_json_value(outcome.metadata),
            },
        )
        self.store.publish(result.result_id, result.to_dict())
        return result

    def run(self) -> PruningStudyResult:
        existing = self._existing()
        baseline_id = f"baseline-{self.study_hash[:20]}"
        baseline = existing.get(baseline_id) or self._evaluate_baseline()
        results: dict[str, PruningEvaluationResult] = {baseline.run_id: baseline}
        for run in self.study.compile():
            result = existing.get(run.run_id)
            if result is None:
                result = self._run_one(run, baseline)
            elif (
                result.evaluator_id != baseline.evaluator_id
                or dict(result.metric_directions) != dict(baseline.metric_directions)
                or dict(result.baseline_metrics) != dict(baseline.metrics)
            ):
                raise ValueError("restored pruning result 与 baseline lineage 不一致")
            results[result.run_id] = result
        ordered = tuple(results[name] for name in sorted(results))
        payload = {
            "schema_version": "pruning-study-result-v1",
            "study_id": self.study.study_id,
            "study_hash": self.study_hash,
            "result_ids": [result.result_id for result in ordered],
            "result_hashes": [result.artifact_hash for result in ordered],
            "scope": self.study.run_intent,
            "formal_eligible": self.study.run_intent == "formal",
        }
        return PruningStudyResult(
            study_id=self.study.study_id,
            study_hash=self.study_hash,
            result_ids=tuple(result.result_id for result in ordered),
            result_hashes=tuple(result.artifact_hash for result in ordered),
            scope=self.study.run_intent,
            formal_eligible=self.study.run_intent == "formal",
            artifact_hash=canonical_json_hash(payload),
        )


__all__ = [
    "CanonicalResultStore",
    "EvaluationOutcome",
    "PruningEvaluationResult",
    "PruningEvaluator",
    "PruningStudyResult",
    "PruningStudyRunner",
    "pruning_study_hash",
]
