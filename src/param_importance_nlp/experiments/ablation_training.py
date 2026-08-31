"""Stage 8 消融矩阵的真实训练 cell 执行核心。

``ablation_runner`` 已经负责单因素矩阵验证、baseline lineage、结果 delta 与小型
JSON 两阶段提交；本模块在其前面补上此前缺失的“真正训练”边界：

* 每个 cell 必须由 builder 构造一个 ``TrainingEngine`` 兼容运行时，不能直接提交
  手工 metrics；
* 训练完成并存在权威 checkpoint 后，才对训练后模型执行只读评估；
* metrics 与 training evidence 分别采用不可变 object + 独立 commit；最后才允许
  ``AblationMatrixRunner`` 发布 ``ablation-cell-result-v1``；
* 若进程在 training evidence 已提交、cell result 尚未提交之间退出，resume 会严格
  复核 evidence/metrics/checkpoint 引用并复用，不会重复训练；训练过程内部的中断
  则由 builder 使用传入的 ``cell_root`` 和自身 checkpoint store 恢复；
* baseline 结果仍由既有 reducer 复用，子 cell 同时获得 baseline 的 checkpoint、
  metrics 与 evidence 引用，供正式 builder 做输入 lineage 审计。

本模块不解析 Hugging Face 资产，也不把缺资产伪装成成功。formal builder 可以在
构造模型/数据时抛出上层 ``TaskBlockedError``；这里仅要求 formal 配置、授权 hash
和训练 outcome 维持同一 scope。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from ..contracts.immutable import freeze_json_mapping, thaw_json_value
from ..contracts.jsonio import canonical_json_hash
from ..providers.training import (
    ClassificationEvaluator,
    ModelAdapter,
    TaskEvaluator,
    TrainingMicrobatch,
)
from ..runtime.checkpoint import CheckpointStore
from ..runtime.training import TrainingEngine, TrainingRunResult, TrainingRunSpec
from .ablation import AblationCell, AblationMatrix
from .ablation_runner import (
    AblationCellResult,
    AblationMatrixRunner,
    AblationStudyResult,
)
from .pruning_runner import (
    CanonicalResultStore,
    EvaluationOutcome,
    _finite_metrics,
    _require_hash,
    _require_id,
)


def _logical_ref(value: str, *, field_name: str) -> str:
    """验证 workspace 相对 POSIX 引用；禁止绝对路径、反斜杠与 ``..``。"""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field_name} 必须是 POSIX workspace 相对引用")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field_name} 发生路径逃逸")
    return path.as_posix()


def _module_state_hash(model: ModelAdapter) -> str:
    """计算参数与 buffer 的逐字节摘要，用于证明 evaluator 没有修改模型。"""

    digest = hashlib.sha256()
    state = model.module.state_dict()
    if not state:
        raise ValueError("ABLATION_TRAINING_MODEL_STATE_EMPTY")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(canonical_json_hash(list(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@runtime_checkable
class AblationTrainingEngine(Protocol):
    """cell runner 需要的最小训练引擎协议；``TrainingEngine`` 原生满足。"""

    model: ModelAdapter

    def run(self) -> TrainingRunResult: ...


@dataclass(frozen=True, slots=True)
class AblationTrainingCellRuntime:
    """builder 为单个 cell 创建的全新训练与评估资源。

    Args:
        engine: 真正执行 optimizer step/checkpoint 的训练引擎。
        evaluator: 对训练后 ``engine.model`` 做只读评估的任务 evaluator。
        evaluation_microbatches: 冻结评估面板；不能为空。
        metric_directions: 每个预期 metric 的方向。键集合在评估后再次严格比对。
        checkpoint_ref_resolver: 把训练结果中的 checkpoint ID 转为 workspace 逻辑
            commit 引用；不得返回目录或绝对路径。
        seed: builder 实际消费的 seed，必须等于冻结 cell seed。
        runtime_id: 模型/数据/训练 adapter 的版本化身份。
    """

    engine: AblationTrainingEngine
    evaluator: TaskEvaluator
    evaluation_microbatches: tuple[TrainingMicrobatch, ...]
    metric_directions: Mapping[str, str]
    checkpoint_ref_resolver: Callable[[str], str]
    seed: int
    runtime_id: str
    run_intent: str

    def __post_init__(self) -> None:
        if not isinstance(self.engine, AblationTrainingEngine):
            raise TypeError("cell runtime engine 不满足 AblationTrainingEngine")
        if not isinstance(self.evaluator, TaskEvaluator):
            raise TypeError("cell runtime evaluator 不满足 TaskEvaluator")
        if not self.evaluation_microbatches or not all(
            isinstance(item, TrainingMicrobatch)
            for item in self.evaluation_microbatches
        ):
            raise ValueError("evaluation_microbatches 必须是非空冻结 batch")
        directions = dict(self.metric_directions)
        if not directions or any(
            not isinstance(name, str)
            or not name
            or direction not in {"higher_is_better", "lower_is_better"}
            for name, direction in directions.items()
        ):
            raise ValueError("metric_directions 非法")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("cell runtime seed 必须是非负整数")
        _require_id(self.runtime_id, field_name="runtime_id")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise ValueError("cell runtime run_intent 非法")
        if not callable(self.checkpoint_ref_resolver):
            raise TypeError("checkpoint_ref_resolver 必须可调用")
        object.__setattr__(self, "evaluation_microbatches", tuple(self.evaluation_microbatches))
        object.__setattr__(self, "metric_directions", MappingProxyType(directions))


@runtime_checkable
class AblationTrainingCellBuilder(Protocol):
    """从冻结 cell/config 创建训练资源；不得返回预先填写的 metrics。"""

    def __call__(
        self,
        cell: AblationCell,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> AblationTrainingCellRuntime: ...


@dataclass(frozen=True, slots=True)
class AblationCellTrainingEvidence:
    """一个完成训练 cell 的 checkpoint/metrics lineage 证据。"""

    matrix_id: str
    matrix_hash: str
    cell_id: str
    config_hash: str
    seed: int
    runtime_id: str
    training_run_id: str
    training_result_hash: str
    checkpoint_refs: tuple[str, ...]
    metrics_ref: str
    metrics_hash: str
    parent_evidence_hash: str | None
    scope: str
    formal_eligible: bool
    formal_authorization_hash: str | None
    artifact_hash: str
    schema_version: str = "ablation-cell-training-evidence-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation-cell-training-evidence-v1":
            raise ValueError("ABLATION_TRAINING_EVIDENCE_SCHEMA_MISMATCH")
        for name in ("matrix_id", "cell_id", "runtime_id", "training_run_id"):
            _require_id(getattr(self, name), field_name=name)
        for name in (
            "matrix_hash",
            "config_hash",
            "training_result_hash",
            "metrics_hash",
            "artifact_hash",
        ):
            _require_hash(getattr(self, name), field_name=name)
        if self.parent_evidence_hash is not None:
            _require_hash(self.parent_evidence_hash, field_name="parent_evidence_hash")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("training evidence seed 非法")
        refs = tuple(
            _logical_ref(value, field_name=f"checkpoint_refs[{index}]")
            for index, value in enumerate(self.checkpoint_refs)
        )
        if not refs or len(refs) != len(set(refs)):
            raise ValueError("training evidence checkpoint_refs 不能为空或重复")
        object.__setattr__(self, "checkpoint_refs", refs)
        object.__setattr__(
            self,
            "metrics_ref",
            _logical_ref(self.metrics_ref, field_name="metrics_ref"),
        )
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("training evidence scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("training evidence formal_eligible 与 scope 不一致")
        if self.scope == "formal":
            _require_hash(
                self.formal_authorization_hash,
                field_name="formal_authorization_hash",
            )
        elif self.formal_authorization_hash is not None:
            raise ValueError("fixture evidence 不得携带 formal authorization")
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("AblationCellTrainingEvidence artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "matrix_hash": self.matrix_hash,
            "cell_id": self.cell_id,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "runtime_id": self.runtime_id,
            "training_run_id": self.training_run_id,
            "training_result_hash": self.training_result_hash,
            "checkpoint_refs": list(self.checkpoint_refs),
            "metrics_ref": self.metrics_ref,
            "metrics_hash": self.metrics_hash,
            "parent_evidence_hash": self.parent_evidence_hash,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "formal_authorization_hash": self.formal_authorization_hash,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(cls, **values: object) -> "AblationCellTrainingEvidence":
        payload = {
            "schema_version": "ablation-cell-training-evidence-v1",
            **values,
        }
        if isinstance(payload.get("checkpoint_refs"), tuple):
            payload["checkpoint_refs"] = list(payload["checkpoint_refs"])
        return cls(**values, artifact_hash=canonical_json_hash(payload))  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationCellTrainingEvidence":
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("ABLATION_TRAINING_EVIDENCE_FIELDS_MISMATCH")
        refs = value["checkpoint_refs"]
        if not isinstance(refs, list):
            raise TypeError("checkpoint_refs 必须是 array")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        kwargs["checkpoint_refs"] = tuple(refs)
        return cls(**kwargs, schema_version=value["schema_version"])  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class AblationCellEvidenceManifest:
    """供 Stage 8 task adapter 消费的完整 cell evidence 索引。

    ``result_ref`` 指向权威 ``ablation-cell-result-v1`` commit。该结果的 metadata
    又绑定本类上方的 training evidence；后者显式列出 checkpoint 与 metrics refs，
    因而 manifest 中的 inline metrics 只是可复核缓存，不是人工数据入口。
    """

    matrix_hash: str
    checkpoint_artifact_hash: str
    scope: str
    formal_eligible: bool
    cells: tuple[Mapping[str, object], ...]
    artifact_hash: str
    schema_version: str = "ablation-cell-evidence-manifest-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation-cell-evidence-manifest-v1":
            raise ValueError("ABLATION_CELL_EVIDENCE_MANIFEST_SCHEMA_MISMATCH")
        _require_hash(self.matrix_hash, field_name="matrix_hash")
        _require_hash(
            self.checkpoint_artifact_hash,
            field_name="checkpoint_artifact_hash",
        )
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("manifest scope 非法")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("manifest formal_eligible 与 scope 不一致")
        if not self.cells:
            raise ValueError("manifest cells 不能为空")
        normalized: list[Mapping[str, object]] = []
        seen: set[str] = set()
        fields = {
            "cell_id",
            "config_hash",
            "metrics",
            "metric_directions",
            "evidence_hash",
            "result_ref",
        }
        for row in self.cells:
            if not isinstance(row, Mapping) or set(row) != fields:
                raise ValueError("ABLATION_CELL_EVIDENCE_ROW_FIELDS_MISMATCH")
            cell_id = _require_id(row["cell_id"], field_name="cell_id")
            if cell_id in seen:
                raise ValueError("manifest cell_id 重复")
            seen.add(cell_id)
            _require_hash(row["config_hash"], field_name="config_hash")
            _require_hash(row["evidence_hash"], field_name="evidence_hash")
            metrics = _finite_metrics(row["metrics"])  # type: ignore[arg-type]
            directions = dict(row["metric_directions"])  # type: ignore[arg-type]
            if set(metrics) != set(directions) or any(
                item not in {"higher_is_better", "lower_is_better"}
                for item in directions.values()
            ):
                raise ValueError("manifest metrics/directions 不一致")
            normalized.append(
                freeze_json_mapping(
                    {
                        "cell_id": cell_id,
                        "config_hash": row["config_hash"],
                        "metrics": dict(metrics),
                        "metric_directions": directions,
                        "evidence_hash": row["evidence_hash"],
                        "result_ref": _logical_ref(
                            row["result_ref"],  # type: ignore[arg-type]
                            field_name="result_ref",
                        ),
                    }
                )
            )
        object.__setattr__(self, "cells", tuple(normalized))
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("AblationCellEvidenceManifest artifact_hash 不匹配")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matrix_hash": self.matrix_hash,
            "checkpoint_artifact_hash": self.checkpoint_artifact_hash,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "cells": [thaw_json_value(row) for row in self.cells],
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def create(cls, **values: object) -> "AblationCellEvidenceManifest":
        payload = {
            "schema_version": "ablation-cell-evidence-manifest-v1",
            **values,
        }
        if isinstance(payload.get("cells"), tuple):
            payload["cells"] = [thaw_json_value(item) for item in payload["cells"]]
        return cls(**values, artifact_hash=canonical_json_hash(payload))  # type: ignore[arg-type]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationCellEvidenceManifest":
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("ABLATION_CELL_EVIDENCE_MANIFEST_FIELDS_MISMATCH")
        cells = value["cells"]
        if not isinstance(cells, list) or not all(isinstance(item, Mapping) for item in cells):
            raise TypeError("manifest cells 必须是 object array")
        kwargs = dict(value)
        kwargs.pop("schema_version")
        kwargs["cells"] = tuple(cells)
        return cls(**kwargs, schema_version=value["schema_version"])  # type: ignore[arg-type]


class TrainingAblationCellExecutor:
    """把真实训练引擎适配成既有 ``AblationCellExecutor`` 协议。"""

    evaluator_id = "stage8-training-cell-v1"

    def __init__(
        self,
        *,
        matrix: AblationMatrix,
        builder: AblationTrainingCellBuilder,
        result_root: str | Path,
        artifact_ref_prefix: str,
        run_intent: str,
    ) -> None:
        if not callable(builder):
            raise TypeError("AblationTrainingCellBuilder 必须可调用")
        if run_intent not in {"local_fixture", "formal"}:
            raise ValueError("run_intent 非法")
        self.matrix = matrix
        self.builder = builder
        self.root = Path(result_root)
        self.metrics_store = CanonicalResultStore(self.root / "metrics")
        self.evidence_store = CanonicalResultStore(self.root / "evidence")
        self.artifact_ref_prefix = _logical_ref(
            artifact_ref_prefix,
            field_name="artifact_ref_prefix",
        )
        self.run_intent = run_intent

    @staticmethod
    def _metrics_id(cell: AblationCell) -> str:
        return f"metrics-{cell.cell_id}"

    @staticmethod
    def _evidence_id(cell: AblationCell) -> str:
        return f"evidence-{cell.cell_id}"

    def _commit_ref(self, section: str, result_id: str) -> str:
        return f"{self.artifact_ref_prefix}/{section}/commits/{result_id}.json"

    def _parent_evidence(self, cell: AblationCell) -> AblationCellTrainingEvidence | None:
        if cell.parent_cell_id is None:
            return None
        baseline = next(
            item for item in self.matrix.cells if item.cell_id == cell.parent_cell_id
        )
        return AblationCellTrainingEvidence.from_mapping(
            self.evidence_store.load(self._evidence_id(baseline))
        )

    def _load_committed(
        self,
        cell: AblationCell,
        *,
        parent: AblationCellTrainingEvidence | None,
        formal_authorization_hash: str | None,
    ) -> EvaluationOutcome | None:
        evidence_id = self._evidence_id(cell)
        commit = self.evidence_store.commits / f"{evidence_id}.json"
        if not commit.exists():
            return None
        evidence = AblationCellTrainingEvidence.from_mapping(
            self.evidence_store.load(evidence_id)
        )
        expected_parent = None if parent is None else parent.artifact_hash
        if (
            evidence.matrix_id != self.matrix.matrix_id
            or evidence.matrix_hash != self.matrix.digest
            or evidence.cell_id != cell.cell_id
            or evidence.config_hash != cell.config_hash
            or evidence.seed != cell.seed
            or evidence.parent_evidence_hash != expected_parent
            or evidence.scope != self.run_intent
            or evidence.formal_authorization_hash != formal_authorization_hash
        ):
            raise ValueError("ABLATION_COMMITTED_TRAINING_EVIDENCE_IDENTITY_DRIFT")
        metrics_id = self._metrics_id(cell)
        metrics_value = self.metrics_store.load(metrics_id)
        expected_metrics_ref = self._commit_ref("metrics", metrics_id)
        if (
            evidence.metrics_ref != expected_metrics_ref
            or metrics_value.get("artifact_hash") != evidence.metrics_hash
            or metrics_value.get("cell_id") != cell.cell_id
            or metrics_value.get("training_result_hash")
            != evidence.training_result_hash
        ):
            raise ValueError("ABLATION_COMMITTED_METRICS_BINDING_DRIFT")
        metrics = metrics_value.get("metrics")
        directions = metrics_value.get("metric_directions")
        if not isinstance(metrics, Mapping) or not isinstance(directions, Mapping):
            raise ValueError("ABLATION_COMMITTED_METRICS_FIELDS_INVALID")
        return EvaluationOutcome(
            evaluator_id=self.evaluator_id,
            metrics=metrics,  # type: ignore[arg-type]
            metric_directions=directions,  # type: ignore[arg-type]
            scope=self.run_intent,
            formal_eligible=self.run_intent == "formal",
            evidence_hash=(
                formal_authorization_hash if self.run_intent == "formal" else None
            ),
            metadata={
                "training_evidence_hash": evidence.artifact_hash,
                "training_evidence_ref": self._commit_ref("evidence", evidence_id),
                "training_result_hash": evidence.training_result_hash,
                "checkpoint_refs": list(evidence.checkpoint_refs),
                "metrics_ref": evidence.metrics_ref,
                "resumed_training_evidence": True,
            },
        )

    def execute(
        self,
        cell: AblationCell,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> EvaluationOutcome:
        """执行或恢复一个 cell；不接受调用方传入的 metrics。"""

        formal_hash = context.get("formal_authorization_hash")
        if self.run_intent == "formal":
            formal_hash = _require_hash(
                formal_hash,
                field_name="formal_authorization_hash",
            )
        elif formal_hash is not None:
            raise ValueError("fixture cell 不得携带 formal authorization")
        parent = self._parent_evidence(cell)
        restored = self._load_committed(
            cell,
            parent=parent,
            formal_authorization_hash=formal_hash,  # type: ignore[arg-type]
        )
        if restored is not None:
            return restored

        cell_root = self.root / "cells" / cell.cell_id
        builder_context: dict[str, object] = {
            **dict(context),
            "cell_root": cell_root,
            "cell_artifact_ref_prefix": (
                f"{self.artifact_ref_prefix}/cells/{cell.cell_id}/checkpoints"
            ),
            "cell_seed": cell.seed,
            "cell_config_hash": cell.config_hash,
            "parent_training_evidence_hash": (
                None if parent is None else parent.artifact_hash
            ),
            "parent_checkpoint_refs": (
                [] if parent is None else list(parent.checkpoint_refs)
            ),
            "parent_metrics_ref": None if parent is None else parent.metrics_ref,
        }
        runtime = self.builder(
            cell,
            resolved_config=resolved_config,
            context=MappingProxyType(builder_context),
        )
        if not isinstance(runtime, AblationTrainingCellRuntime):
            raise TypeError("ABLATION_TRAINING_BUILDER_RESULT_INVALID")
        if runtime.seed != cell.seed:
            raise ValueError("ABLATION_TRAINING_BUILDER_SEED_MISMATCH")
        if runtime.run_intent != self.run_intent:
            raise ValueError("ABLATION_TRAINING_BUILDER_SCOPE_MISMATCH")
        training = runtime.engine.run()
        if not isinstance(training, TrainingRunResult):
            raise TypeError("ABLATION_TRAINING_ENGINE_RESULT_INVALID")
        if training.status != "COMPLETE":
            # 不发布 metrics/evidence/cell result；builder 可在下次调用时从 cell_root
            # 中的权威训练 checkpoint 继续。
            raise RuntimeError(f"ABLATION_TRAINING_CELL_INCOMPLETE:{cell.cell_id}")
        training_wire = training.to_dict()
        training_hash = training_wire["artifact_hash"]
        _require_hash(training_hash, field_name="training_result_hash")
        checkpoint_ids = tuple(training.checkpoint_ids)
        if (
            not checkpoint_ids
            or len(checkpoint_ids) != len(set(checkpoint_ids))
            or training.state.last_checkpoint_id != checkpoint_ids[-1]
        ):
            raise RuntimeError("ABLATION_TRAINING_CHECKPOINT_SET_INVALID")
        checkpoint_refs = tuple(
            _logical_ref(
                runtime.checkpoint_ref_resolver(checkpoint_id),
                field_name=f"checkpoint_ref[{index}]",
            )
            for index, checkpoint_id in enumerate(checkpoint_ids)
        )

        before = _module_state_hash(runtime.engine.model)
        metrics = _finite_metrics(
            runtime.evaluator.evaluate(
                runtime.engine.model,
                runtime.evaluation_microbatches,
            )
        )
        after = _module_state_hash(runtime.engine.model)
        if before != after:
            raise RuntimeError("ABLATION_TRAINING_EVALUATOR_MUTATED_MODEL")
        directions = dict(runtime.metric_directions)
        if set(metrics) != set(directions):
            raise ValueError("ABLATION_TRAINING_METRIC_DIRECTION_KEYS_MISMATCH")

        metrics_id = self._metrics_id(cell)
        metrics_payload: dict[str, object] = {
            "schema_version": "ablation-cell-training-metrics-v1",
            "matrix_hash": self.matrix.digest,
            "cell_id": cell.cell_id,
            "config_hash": cell.config_hash,
            "seed": cell.seed,
            "training_result_hash": training_hash,
            "metrics": dict(metrics),
            "metric_directions": directions,
            "scope": self.run_intent,
        }
        metrics_payload["artifact_hash"] = canonical_json_hash(metrics_payload)
        self.metrics_store.publish(metrics_id, metrics_payload)
        metrics_ref = self._commit_ref("metrics", metrics_id)

        evidence = AblationCellTrainingEvidence.create(
            matrix_id=self.matrix.matrix_id,
            matrix_hash=self.matrix.digest,
            cell_id=cell.cell_id,
            config_hash=cell.config_hash,
            seed=cell.seed,
            runtime_id=runtime.runtime_id,
            training_run_id=training.run_id,
            training_result_hash=training_hash,
            checkpoint_refs=checkpoint_refs,
            metrics_ref=metrics_ref,
            metrics_hash=metrics_payload["artifact_hash"],
            parent_evidence_hash=None if parent is None else parent.artifact_hash,
            scope=self.run_intent,
            formal_eligible=self.run_intent == "formal",
            formal_authorization_hash=formal_hash,
        )
        evidence_id = self._evidence_id(cell)
        self.evidence_store.publish(evidence_id, evidence.to_dict())
        return EvaluationOutcome(
            evaluator_id=self.evaluator_id,
            metrics=dict(metrics),
            metric_directions=directions,
            scope=self.run_intent,
            formal_eligible=self.run_intent == "formal",
            evidence_hash=formal_hash if self.run_intent == "formal" else None,
            metadata={
                "training_evidence_hash": evidence.artifact_hash,
                "training_evidence_ref": self._commit_ref("evidence", evidence_id),
                "training_result_hash": training_hash,
                "checkpoint_refs": list(checkpoint_refs),
                "metrics_ref": metrics_ref,
                "resumed_training_evidence": False,
            },
        )

    def evidence_for(self, cell: AblationCell) -> AblationCellTrainingEvidence:
        """读取一个已经提交的 cell training evidence。"""

        return AblationCellTrainingEvidence.from_mapping(
            self.evidence_store.load(self._evidence_id(cell))
        )


@dataclass(frozen=True, slots=True)
class AblationTrainingStudyOutput:
    """真实训练矩阵的 reducer 与 manifest 交付对象。"""

    study_result: AblationStudyResult
    evidence_manifest: AblationCellEvidenceManifest
    evidence_manifest_ref: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_manifest_ref",
            _logical_ref(
                self.evidence_manifest_ref,
                field_name="evidence_manifest_ref",
            ),
        )
        if self.study_result.matrix_hash != self.evidence_manifest.matrix_hash:
            raise ValueError("training study 与 evidence manifest matrix 不一致")

    def to_dict(self) -> dict[str, object]:
        payload = {
            "schema_version": "ablation-training-study-output-v1",
            "study_result": self.study_result.to_dict(),
            "evidence_manifest": self.evidence_manifest.to_dict(),
            "evidence_manifest_ref": self.evidence_manifest_ref,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        return payload


class AblationStudyRunner:
    """baseline-first 执行完整冻结矩阵，并发布可供 Stage789 接线的 manifest。"""

    def __init__(
        self,
        matrix: AblationMatrix,
        *,
        builder: AblationTrainingCellBuilder,
        result_root: str | Path,
        artifact_ref_prefix: str,
        run_intent: str,
        source_checkpoint_artifact_hash: str,
        config_validator: Callable[[Mapping[str, object]], object] | None = None,
        formal_authorization_hash: str | None = None,
    ) -> None:
        self.matrix = matrix
        self.root = Path(result_root)
        self.prefix = _logical_ref(
            artifact_ref_prefix,
            field_name="artifact_ref_prefix",
        )
        self.source_checkpoint_artifact_hash = _require_hash(
            source_checkpoint_artifact_hash,
            field_name="source_checkpoint_artifact_hash",
        )
        self.executor = TrainingAblationCellExecutor(
            matrix=matrix,
            builder=builder,
            result_root=self.root / "training",
            artifact_ref_prefix=f"{self.prefix}/training",
            run_intent=run_intent,
        )
        self.matrix_runner = AblationMatrixRunner(
            matrix,
            executor=self.executor,
            result_root=self.root / "results",
            run_intent=run_intent,
            config_validator=config_validator,
            formal_authorization_hash=formal_authorization_hash,
        )
        self.manifest_store = CanonicalResultStore(self.root / "manifest")
        self.run_intent = run_intent

    def _result_ref(self, result: AblationCellResult) -> str:
        return f"{self.prefix}/results/commits/{result.result_id}.json"

    def run(self) -> AblationTrainingStudyOutput:
        """执行/恢复所有 cell；仅完整覆盖矩阵后发布 evidence manifest。"""

        study = self.matrix_runner.run()
        results = {
            result.cell_id: result
            for result in (
                AblationCellResult.from_mapping(value)
                for value in self.matrix_runner.store.restore()
            )
        }
        if set(results) != {cell.cell_id for cell in self.matrix.cells}:
            raise RuntimeError("ABLATION_TRAINING_RESULT_COVERAGE_INCOMPLETE")
        rows: list[Mapping[str, object]] = []
        for cell in sorted(self.matrix.cells, key=lambda item: item.cell_id):
            result = results[cell.cell_id]
            evidence = self.executor.evidence_for(cell)
            metadata = thaw_json_value(result.metadata)
            outcome = metadata.get("outcome") if isinstance(metadata, Mapping) else None
            if (
                not isinstance(outcome, Mapping)
                or outcome.get("training_evidence_hash") != evidence.artifact_hash
                or outcome.get("training_result_hash") != evidence.training_result_hash
                or outcome.get("metrics_ref") != evidence.metrics_ref
                or tuple(outcome.get("checkpoint_refs", ())) != evidence.checkpoint_refs
            ):
                raise ValueError("ABLATION_CELL_RESULT_TRAINING_EVIDENCE_DRIFT")
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "config_hash": cell.config_hash,
                    "metrics": dict(result.metrics),
                    "metric_directions": dict(result.metric_directions),
                    "evidence_hash": evidence.artifact_hash,
                    "result_ref": self._result_ref(result),
                }
            )
        manifest = AblationCellEvidenceManifest.create(
            matrix_hash=self.matrix.digest,
            checkpoint_artifact_hash=self.source_checkpoint_artifact_hash,
            scope=self.run_intent,
            formal_eligible=self.run_intent == "formal",
            cells=tuple(rows),
        )
        manifest_id = f"manifest-{self.matrix.digest[:24]}"
        self.manifest_store.publish(manifest_id, manifest.to_dict())
        return AblationTrainingStudyOutput(
            study,
            manifest,
            f"{self.prefix}/manifest/commits/{manifest_id}.json",
        )


class _DeviceBatchCursor:
    """在 cursor 消费边界显式搬运 batch，并原样委托可恢复状态。"""

    def __init__(self, source: object, device: torch.device) -> None:
        self.source = source
        self.device = device

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]:
        batches = self.source.next_microbatches()  # type: ignore[attr-defined]
        return tuple(batch.to(self.device) for batch in batches)

    def state_dict(self) -> Mapping[str, object]:
        return self.source.state_dict()  # type: ignore[attr-defined,no-any-return]

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.source.load_state_dict(state)  # type: ignore[attr-defined]


class _SelectedMetricEvaluator:
    """只发布配置声明的指标，防止 provider 的附加指标改变 artifact schema。"""

    def __init__(self, delegate: TaskEvaluator, names: Sequence[str]) -> None:
        self.delegate = delegate
        self.names = tuple(names)

    def evaluate(
        self,
        model: ModelAdapter,
        microbatches: Sequence[TrainingMicrobatch],
    ) -> Mapping[str, float]:
        values = self.delegate.evaluate(model, microbatches)
        missing = [name for name in self.names if name not in values]
        if missing:
            raise ValueError(
                "ABLATION_CONFIGURED_EVALUATION_METRIC_MISSING:" + ",".join(missing)
            )
        return {name: float(values[name]) for name in self.names}


class ConfiguredAblationTrainingBuilder:
    """从 ``ResolvedConfigV2`` 与本地 provider manifest 构造真实 cell。

    该 builder 是 Stage 8 正式执行的默认代码路径：上层只需传入任务运行时已经
    验证的环境快照，不需要编写或注入 Python factory。模型、数据与 tokenizer
    仍由现有 ``_training_resources`` 严格离线加载；资产或可选依赖缺失会转换成
    ``TaskBlockedError``，由统一任务运行时记录为结构化 ``BLOCKED``。
    """

    def __init__(self, workspace_root: str | Path, *, environment: object | None = None) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.environment = environment

    def __call__(
        self,
        cell: AblationCell,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> AblationTrainingCellRuntime:
        # 延迟导入避免 experiments 包初始化时与默认 task runner 形成环。
        from ..assets import AssetManifestError
        from ..contracts.config_v2 import ResolvedConfigV2
        from ..contracts.seed import SeedPlan
        from ..providers import configure_batch_cursor
        from ..providers.optional import DependencyUnavailable
        from ..runtime.task_runtime import (
            BlockerCode,
            TaskBlockedError,
            TaskBlocker,
            TaskExecutionRequest,
            TaskRuntimeEnvironment,
        )
        from ..runtime.training import TrainingState, install_training_rng
        from ..runtime.training_factory import (
            build_grad_scaler,
            build_optimizer,
            build_scheduler,
        )
        from .task_runners import _training_resources

        if not isinstance(resolved_config, ResolvedConfigV2):
            raise TypeError("configured ablation builder 要求 ResolvedConfigV2")
        if resolved_config.task_definition.stage != 8:
            raise ValueError("CONFIGURED_ABLATION_TASK_MUST_BE_STAGE8")
        cell_root = context.get("cell_root")
        if not isinstance(cell_root, Path):
            raise TypeError("configured ablation builder 缺少 cell_root")

        base = resolved_config.base_config
        distributed = base.section("distributed")
        if int(distributed["world_size"]) != 1:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    "stage8_cell_world_size_1",
                    "当前消融 cell builder 只接受单进程 cell；矩阵可由外层调度并行执行",
                    False,
                )
            )
        runtime_device = torch.device(str(base.section("runtime")["device"]))
        if runtime_device.type == "cuda" and not torch.cuda.is_available():
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.DEVICE_UNAVAILABLE,
                    "cuda",
                    "配置要求 CUDA，但当前进程没有可用 CUDA device",
                    True,
                )
            )

        environment = self.environment
        if environment is None:
            decision_ref = base.section("importance")["estimator_decision_ref"]
            environment = TaskRuntimeEnvironment(estimator_decision_ref=decision_ref)  # type: ignore[arg-type]
        if not isinstance(environment, TaskRuntimeEnvironment):
            raise TypeError("configured ablation environment 类型非法")
        request = TaskExecutionRequest(
            resolved_config,
            resolved_config.task_definition,
            environment,
        )
        try:
            resources = _training_resources(
                request,
                self.workspace_root,
                rank=0,
                world_size=1,
            )
        except DependencyUnavailable as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.DEPENDENCY_UNAVAILABLE,
                    error.dependency,
                    str(error),
                    True,
                )
            ) from error
        except (FileNotFoundError, AssetManifestError) as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "offline_ablation_assets",
                    f"离线消融资产不可用：{type(error).__name__}: {error}",
                    True,
                )
            ) from error

        training = resolved_config.section("training")
        evaluation = resolved_config.section("evaluation")
        execution = resolved_config.section("execution")
        scheduler_options = resolved_config.section("scheduler")
        precision_runtime = resolved_config.section("precision_runtime")
        optimizer_runtime = resolved_config.section("optimizer_runtime")
        checkpoint_schedule = resolved_config.section("checkpoint_schedule")
        data_loader = resolved_config.section("data_loader")
        assert all(
            isinstance(value, dict)
            for value in (
                training,
                evaluation,
                execution,
                scheduler_options,
                precision_runtime,
                optimizer_runtime,
                checkpoint_schedule,
                data_loader,
            )
        )
        max_steps = training["max_steps"]
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("CONFIGURED_ABLATION_REQUIRES_TRAINING_MAX_STEPS")
        if not bool(evaluation["enabled"]):
            raise ValueError("CONFIGURED_ABLATION_REQUIRES_EVALUATION")
        if resources.evaluation_dataset is None or resources.evaluator is None:
            raise RuntimeError("CONFIGURED_ABLATION_EVALUATION_RESOURCES_MISSING")

        seed_plan = install_training_rng(cell.seed, rank=0, world_size=1)
        resources.model.module.to(runtime_device)
        optimizer = build_optimizer(
            resources.model.module.parameters(),
            base.section("optimizer"),
            optimizer_runtime,
        )
        scheduler = build_scheduler(optimizer, scheduler_options)
        scaler = build_grad_scaler(
            precision_runtime,
            device_type=runtime_device.type,
        )
        importance = base.section("importance")
        estimator_name = importance["estimator_name"]
        if estimator_name == "weighted_u":
            estimator_name = "u"
        if estimator_name not in {"raw", "u", "double"}:
            raise ValueError("CONFIGURED_ABLATION_ESTIMATOR_UNSUPPORTED")
        decision_hash: str | None = None
        decision_gate: str | None = None
        if resolved_config.run_intent == "formal":
            reference = environment.estimator_decision_ref
            if reference is None:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
                        "estimator_decision",
                        "formal 消融训练缺少已审核 EstimatorDecision 引用",
                        True,
                    )
                )
            # 完整 decision/Gate 资格已由 TaskRuntime preflight 验证；训练 checkpoint
            # 仍绑定该不可变输入文件的 canonical hash，防止恢复时替换决策。
            from ..contracts.jsonio import load_canonical_json

            try:
                decision_value = load_canonical_json(self.workspace_root / reference)
            except FileNotFoundError as error:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
                        "estimator_decision",
                        str(error),
                        True,
                        (reference,),
                    )
                ) from error
            decision_hash = canonical_json_hash(decision_value)
            decision_gate = "PASS"

        autocast_dtype = (
            str(precision_runtime["autocast_dtype"])
            if bool(precision_runtime["autocast_enabled"])
            else "none"
        )
        logging = base.section("logging")
        data = base.section("data")
        asset_fingerprint = canonical_json_hash(
            {
                "builder": "configured-ablation-training-v1",
                "task_id": resolved_config.task_id,
                "provider": resolved_config.section("providers"),
                "asset_evidence": list(resources.asset_evidence),
            }
        )
        spec = TrainingRunSpec(
            run_id=f"stage8-{cell.cell_id}",
            run_intent=resolved_config.run_intent,
            max_steps=max_steps,
            max_attempts=max_steps + int(execution["max_attempts"]) - 1,
            importance_enabled=True,
            estimator_name=str(estimator_name),
            accumulation_dtype=str(base.section("precision")["statistic_dtype"]),
            max_grad_norm=training["gradient_clip_max_norm"],  # type: ignore[arg-type]
            autocast_dtype=autocast_dtype,
            checkpoint_every_steps=int(checkpoint_schedule["segments"][0]["every_steps"]),  # type: ignore[index]
            log_every_steps=int(logging["log_every_steps"]),
            weights_exogenous=bool(data["weights_exogenous"]),
            common_mean_assumption=bool(data["common_mean_assumption"]),
            estimator_decision_hash=decision_hash,
            estimator_gate_status=decision_gate,
            metadata={
                "cell_id": cell.cell_id,
                "cell_config_hash": cell.config_hash,
                "cell_seed": cell.seed,
                "seed_plan_hash": seed_plan.artifact_hash,
                "asset_fingerprint": asset_fingerprint,
            },
            checkpoint_segments=tuple(
                dict(segment) for segment in checkpoint_schedule["segments"]  # type: ignore[arg-type]
            ),
        )
        checkpoint_store = CheckpointStore(cell_root / "checkpoints")
        source_cursor = resources.dataset.cursor(  # type: ignore[attr-defined]
            seed=seed_plan.seed_for("sampler"), rank=0, world_size=1
        )
        cursor = configure_batch_cursor(
            _DeviceBatchCursor(source_cursor, runtime_device),
            num_workers=int(data_loader["num_workers"]),
            prefetch_factor=data_loader["prefetch_factor"],  # type: ignore[arg-type]
            persistent_workers=bool(data_loader["persistent_workers"]),
        )
        engine = TrainingEngine(
            spec=spec,
            model=resources.model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=cursor,
            checkpoint_store=checkpoint_store,
            experiment_id="stage8-ablation",
            attempt_id="attempt-0000",
            session_id=f"{cell.cell_id}-session",
        )
        if checkpoint_store.discover():
            engine.resume_latest()
            engine.state = TrainingState(
                engine.state.global_step,
                engine.state.attempt_index,
                engine.state.skipped_steps,
                0,
                engine.state.last_checkpoint_id,
            )

        evaluation_seed = SeedPlan.from_master_seed(cell.seed).derive_subseed(
            "sampler", "evaluation"
        )
        eval_cursor = resources.evaluation_dataset.cursor(  # type: ignore[attr-defined]
            seed=evaluation_seed, rank=0, world_size=1
        )
        evaluation_batches: list[TrainingMicrobatch] = []
        for _ in range(int(evaluation["max_batches"] or 1)):
            try:
                evaluation_batches.extend(
                    batch.to(runtime_device) for batch in eval_cursor.next_microbatches()
                )
            except StopIteration:
                break
        if not evaluation_batches:
            raise ValueError("CONFIGURED_ABLATION_EVALUATION_PANEL_EMPTY")
        requested_metrics = tuple(str(name) for name in evaluation["metrics"])
        directions = {
            name: (
                "lower_is_better"
                if name in {"loss", "perplexity", "pile_loss", "pile_perplexity"}
                else "higher_is_better"
            )
            for name in requested_metrics
        }
        prefix = _logical_ref(
            str(context["cell_artifact_ref_prefix"]),
            field_name="cell_checkpoint_prefix",
        )
        return AblationTrainingCellRuntime(
            engine=engine,
            evaluator=_SelectedMetricEvaluator(resources.evaluator, requested_metrics),  # type: ignore[arg-type]
            evaluation_microbatches=tuple(evaluation_batches),
            metric_directions=directions,
            checkpoint_ref_resolver=lambda checkpoint_id: (
                f"{prefix}/commits/{checkpoint_id}.json"
            ),
            seed=cell.seed,
            runtime_id=f"configured-ablation-{asset_fingerprint[:24]}",
            run_intent=resolved_config.run_intent,
        )


class TinyAblationTrainingBuilder:
    """本机 CPU 的真实 ``TrainingEngine`` cell builder。

    支持的声明字段仅为 ``optimizer.learning_rate``、可选
    ``training.max_steps``。它用于 run-ready fixture/测试，不会被 formal 入口自动
    选择；formal 必须注入真实离线模型与数据 builder。
    """

    def __call__(
        self,
        cell: AblationCell,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> AblationTrainingCellRuntime:
        from ..providers.tiny import build_tiny_training_fixture

        if not isinstance(resolved_config, Mapping):
            raise TypeError("tiny ablation config 必须是 mapping")
        optimizer_config = resolved_config.get("optimizer")
        training_config = resolved_config.get("training", {})
        if not isinstance(optimizer_config, Mapping) or not isinstance(
            training_config,
            Mapping,
        ):
            raise TypeError("tiny ablation optimizer/training config 非法")
        learning_rate = optimizer_config.get("learning_rate")
        max_steps = training_config.get("max_steps", 2)
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or float(learning_rate) <= 0
        ):
            raise ValueError("tiny ablation learning_rate 必须为正数")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("tiny ablation max_steps 必须为正整数")
        cell_root = context.get("cell_root")
        if not isinstance(cell_root, Path):
            raise TypeError("tiny ablation builder 缺少 cell_root")
        fixture = build_tiny_training_fixture(
            task_type="sequence_classification",
            seed=cell.seed,
            steps=max_steps,
            microbatches_per_step=2,
            microbatch_size=2,
        )
        evaluation = build_tiny_training_fixture(
            task_type="sequence_classification",
            seed=cell.seed + 1,
            steps=1,
            microbatches_per_step=2,
            microbatch_size=2,
        )
        optimizer = torch.optim.SGD(
            fixture.model.module.parameters(),
            lr=float(learning_rate),
        )
        checkpoint_root = cell_root / "checkpoints"
        checkpoint_store = CheckpointStore(checkpoint_root)
        run_id = f"stage8-{cell.cell_id}"
        engine = TrainingEngine(
            spec=TrainingRunSpec(
                run_id,
                "local_fixture",
                max_steps=max_steps,
                max_attempts=max_steps,
                checkpoint_every_steps=1,
                importance_enabled=False,
                estimator_name="u",
                weights_exogenous=True,
                common_mean_assumption=True,
            ),
            model=fixture.model,
            optimizer=optimizer,
            cursor=fixture.dataset.cursor(seed=cell.seed),
            checkpoint_store=checkpoint_store,
        )
        if any(checkpoint_store.commits.glob("*.json")):
            engine.resume_latest()
        evaluation_batches = tuple(
            batch for step in evaluation.dataset.steps for batch in step
        )
        prefix = context.get("cell_artifact_ref_prefix")
        if prefix is None:
            # executor 的 cell_root 是本机真实目录；checkpoint ref 仍必须是逻辑引用，
            # 使用 cell ID 下的稳定相对布局，不暴露绝对临时路径。
            prefix = f"cells/{cell.cell_id}/checkpoints"
        logical_prefix = _logical_ref(str(prefix), field_name="cell_checkpoint_prefix")
        return AblationTrainingCellRuntime(
            engine=engine,
            evaluator=ClassificationEvaluator(),
            evaluation_microbatches=evaluation_batches,
            metric_directions={
                "loss": "lower_is_better",
                "accuracy": "higher_is_better",
            },
            checkpoint_ref_resolver=lambda checkpoint_id: (
                f"{logical_prefix}/commits/{checkpoint_id}.json"
            ),
            seed=cell.seed,
            runtime_id="tiny-training-engine-v1",
            run_intent="local_fixture",
        )


__all__ = [
    "AblationCellEvidenceManifest",
    "AblationCellTrainingEvidence",
    "AblationStudyRunner",
    "AblationTrainingCellBuilder",
    "AblationTrainingCellRuntime",
    "AblationTrainingEngine",
    "AblationTrainingStudyOutput",
    "ConfiguredAblationTrainingBuilder",
    "TinyAblationTrainingBuilder",
    "TrainingAblationCellExecutor",
]
