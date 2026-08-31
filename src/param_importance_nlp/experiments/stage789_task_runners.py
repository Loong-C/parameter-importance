"""Stage 7--9 核心到统一 :class:`TaskRuntime` 的复合适配层。

本模块只负责“任务目录协议”和已有科学核心之间的连接，不重新实现剪枝、
消融或统计公式。设计上有四条边界：

1. ``local_fixture`` 可以使用内置的小型、确定性数据，也可以读取调用方提供的
   canonical input ref；一旦声明了 ref，损坏或语义不匹配就立即失败，绝不静默
   换回内置数据。
2. ``formal`` 从不使用内置 fixture。剪枝和消融执行必须由工厂注入真实资源
   loader；分析任务必须读取 hash-bound、``formal_eligible=true`` 的源产物。
3. 所有公开输出统一经 :class:`TaskArtifactStore` 发布，且产物种类及顺序严格
   等于 task catalog 的 ``artifact_kinds``。
4. ANALYSIS/STATISTICS/REPORTING 等 runner kind 也服务较早阶段，因此复合 runner
   只消费本模块明确声明的 task ID；其余 task 委托同 kind fallback，未配置
   fallback 时给出结构化 ``TaskBlockedError``。

真实模型对象、数据加载器和 evaluator 不能安全序列化进 JSON，所以 formal
Stage 7/8 通过 ``PruningResourceLoader`` / ``AblationResourceLoader`` 注入。loader
返回值仍必须绑定 checkpoint、importance/matrix、coordinate registry 和当前
运行环境授权 hash，本模块会在启动核心 runner 前再次核对。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import torch

from ..analysis.charts import ChartArtifact, ChartSpec
from ..analysis.metrics import MetricResult, damage_auc
from ..analysis.pipeline import (
    AnalysisBundle,
    AnalysisBundleBuilder,
    BoundSourceTable,
    CompositeFigureArtifact,
    CrossStageSourceBuilder,
    StageArtifactRows,
    TableArtifact,
    TableSpec,
    grouped_statistics,
    paired_statistics_with_holm,
    analysis_producer_source_hash,
    render_composite_figure_set,
    render_table,
)
from ..analysis.report import AnalysisReport, AnalysisReportBuilder, FrozenSourceTable
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.task_catalog import RecoveryMode, RunnerKind
from ..atomic import sha256_file
from ..core.baselines import produce_baseline_scores
from ..core.tensors import TensorMap
from ..providers.training import ModelAdapter, TaskEvaluator, TrainingMicrobatch
from ..runtime.task_artifacts import TaskArtifactStore
from ..runtime.task_runtime import (
    BlockerCode,
    TaskBlockedError,
    TaskBlocker,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunner,
    TaskRuntime,
)
from ..runtime.tensor_bundle import load_tensor_bundle, publish_tensor_bundle
from .ablation import AblationFactor, AblationMatrix
from .ablation_runner import (
    AblationCellExecutor,
    AblationMatrixRunner,
    AblationStudyResult,
    _default_config_validator,
)
from .ablation_training import (
    AblationStudyRunner as TrainingAblationStudyRunner,
    AblationTrainingCellBuilder,
    ConfiguredAblationTrainingBuilder,
    TinyAblationTrainingBuilder,
)
from .pruning import IMPORTANCE_METHODS, ImportanceSourceSpec, PruningStudySpec
from .pruning_runner import (
    EvaluationOutcome,
    PruningEvaluator,
    PruningStudyRunner,
    pruning_study_hash,
)
from .task_runners import _training_resources
from .artifact_lineage import (
    LoadedInputArtifact,
    configured_input_refs as _configured_input_refs,
    load_configured_inputs as _load_configured_inputs,
    load_input_artifact as _load_input_artifact,
    safe_workspace_path as _safe_workspace_path,
    validate_analysis_lineage,
)
from .stage7_training_source import (
    Stage7TrainingSource,
    load_checkpoint_state_from_commit,
    load_route_training_source,
)


_HANDLED_BY_KIND: Mapping[RunnerKind, frozenset[str]] = {
    RunnerKind.CONTRACT: frozenset({"stage8.freeze"}),
    RunnerKind.PRUNING: frozenset(
        {
            "stage4.pruning_validation",
            "stage7.functional_pruning_validation",
            "stage7.matrix",
            "stage7.evaluate",
        }
    ),
    RunnerKind.ABLATION: frozenset(
        {"stage8.ablation_and_robustness", "stage8.execute"}
    ),
    RunnerKind.ANALYSIS: frozenset(
        {
            "stage8.recommend",
            "stage9.ingest",
            "stage9.analysis_visualization_reporting",
        }
    ),
    RunnerKind.STATISTICS: frozenset(
        {"stage7.reduce", "stage8.reduce", "stage9.statistics"}
    ),
    RunnerKind.REPORTING: frozenset(
        {
            "stage7.report",
            "stage8.report",
            "stage9.tables",
            "stage9.charts",
            "stage9.report",
        }
    ),
    RunnerKind.DELIVERY: frozenset({"stage9.bundle"}),
    RunnerKind.TEST_MATRIX: frozenset({"stage9.replay"}),
}


def _json_mapping(value: Mapping[str, object]) -> dict[str, JSONValue]:
    """把冻结 mapping 转成 canonical writer 可接收的 JSON object。"""

    thawed = thaw_json_value(value)
    if not isinstance(thawed, dict):  # pragma: no cover - Mapping 输入的防御性分支
        raise TypeError("STAGE789_PAYLOAD_NOT_JSON_OBJECT")
    return thawed  # type: ignore[return-value]


def _versioned_task_payload(
    artifact_kind: str,
    value: Mapping[str, object],
) -> dict[str, JSONValue]:
    """为 Stage 7--9 的公开任务载荷补齐稳定的机器版本。

    大多数核心结果（例如 ``PruningStudyResult``、``AblationStudyResult``）
    自身已经携带专用 ``schema_version``。少数 task 输出只是把多个专用结果聚合在
    同一个顶层 mapping 中；这类包装层过去没有版本，因而会被统一
    :class:`TaskArtifactStore` 的 fail-closed 契约拒绝。

    本函数只在顶层版本缺失时补充版本，不重命名、移动或包裹既有字段，所以既有
    消费者仍可直接读取 ``study_result``、``results`` 等字段。版本由 catalog 中冻结的
    ``artifact_kind`` 确定，同一种产物跨运行保持不变：
    ``stage789-<artifact-kind>-task-payload-v1``。
    """

    payload = _json_mapping(value)
    if "schema_version" not in payload:
        payload["schema_version"] = (
            f"stage789-{artifact_kind.replace('_', '-')}-task-payload-v1"
        )
    return payload


def _require_formal_inputs(
    request: TaskExecutionRequest,
    documents: Sequence[LoadedInputArtifact],
    *,
    requirement: str,
) -> None:
    if request.config.run_intent != "formal":
        return
    if not documents:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                requirement,
                "formal 任务缺少 hash-bound 输入引用",
                True,
            )
        )
    ineligible = tuple(document.ref for document in documents if not document.formal_eligible)
    if ineligible:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.GATE_NOT_READY,
                requirement,
                "formal 任务引用了 local_fixture 或未取得正式资格的输入",
                True,
                ineligible,
            )
        )


@dataclass(frozen=True, slots=True)
class PruningTaskResources:
    """工厂注入给 Stage 7 的真实运行资源。

    ``formal_authorization_hash`` 必须等于当前 ``TaskRuntimeEnvironment`` 的 hash，
    防止一个已经离开原 Gate/能力快照的 evaluator 被复用到另一次 formal 执行。
    """

    study: PruningStudySpec
    parameters: TensorMap
    scores_by_artifact_hash: Mapping[str, TensorMap]
    evaluator: PruningEvaluator
    model_checkpoint_hash: str
    coordinate_registry_hash: str
    formal_authorization_hash: str | None = None
    checkpoint_input_hash: str | None = None
    importance_input_hashes: tuple[str, ...] = ()
    matrix_input_hash: str | None = None


@dataclass(frozen=True, slots=True)
class AblationTaskResources:
    """工厂注入给 Stage 8 的冻结矩阵和真实 cell executor。"""

    matrix: AblationMatrix
    executor: AblationCellExecutor
    config_validator: Callable[[Mapping[str, object]], object]
    formal_authorization_hash: str | None = None
    checkpoint_input_hash: str | None = None
    matrix_input_hash: str | None = None
    training_builder: AblationTrainingCellBuilder | None = None
    source_checkpoint_artifact_hash: str | None = None


@runtime_checkable
class PruningResourceLoader(Protocol):
    def __call__(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> PruningTaskResources: ...


@runtime_checkable
class AblationResourceLoader(Protocol):
    def __call__(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> AblationTaskResources: ...


class _TinyPruningEvaluator:
    """只供 local fixture 使用的确定性 CPU evaluator。"""

    def evaluate(
        self,
        parameters: TensorMap,
        *,
        run_id: str,
        context: Mapping[str, object],
    ) -> EvaluationOutcome:
        # 平方范数模拟“保留的任务效用”；剪掉任意非零坐标都会降低该量。
        utility = float(
            sum(torch.sum(tensor.detach().to(torch.float64).square()).item()
                for tensor in parameters.values())
        )
        return EvaluationOutcome(
            evaluator_id="stage7-tiny-quadratic",
            metrics={"utility": utility},
            metric_directions={"utility": "higher_is_better"},
            scope="local_fixture",
            formal_eligible=False,
            metadata={"fixture": True, "run_id": run_id},
        )


class _TinyAblationExecutor:
    """从 cell 配置的唯一改动计算可重复分数，仅供 local fixture。"""

    def execute(
        self,
        cell: object,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> EvaluationOutcome:
        # fixture 只验证矩阵执行、父子 lineage 与归约语义；不把任意配置值解释为
        # 真实任务质量。根 cell 固定为 1，所有单因素 child 固定为 0.8。
        score = 1.0 if getattr(cell, "parent_cell_id") is None else 0.8
        return EvaluationOutcome(
            evaluator_id="stage8-tiny-cell",
            metrics={"quality": score},
            metric_directions={"quality": "higher_is_better"},
            scope="local_fixture",
            formal_eligible=False,
            metadata={"fixture": True},
        )


class _TorchTaskPruningEvaluator:
    """把 provider 的只读任务评估器接到可恢复剪枝执行器。

    ``parameters`` 与 ``model.module`` 的 Parameter 共享存储；
    :class:`PruningContext` 因而会在 evaluate 期间真实地把选中坐标置零，并在返回
    后恢复。evidence hash 同时绑定训练 task-output、当前环境和本次运行上下文。
    """

    def __init__(
        self,
        model: ModelAdapter,
        evaluator: TaskEvaluator,
        batches: Sequence[TrainingMicrobatch],
        *,
        source_artifact_hash: str,
        environment_hash: str,
        scope: str,
    ) -> None:
        if not batches:
            raise ValueError("STAGE7_EVALUATION_BATCHES_EMPTY")
        self.model = model
        self.evaluator = evaluator
        self.batches = tuple(batches)
        self.source_artifact_hash = source_artifact_hash
        self.environment_hash = environment_hash
        self.scope = scope

    def evaluate(
        self,
        parameters: TensorMap,
        *,
        run_id: str,
        context: Mapping[str, object],
    ) -> EvaluationOutcome:
        # 先确认传入 TensorMap 仍引用当前 model 的同一批坐标；避免 evaluator 忽略
        # PruningContext 实际修改的是另一份 clone。
        named = dict(self.model.module.named_parameters())
        if tuple(parameters) != tuple(name for name in named if name in parameters):
            raise ValueError("STAGE7_EVALUATOR_PARAMETER_ORDER_MISMATCH")
        if any(parameters[name].data_ptr() != named[name].data_ptr() for name in parameters):
            raise ValueError("STAGE7_EVALUATOR_PARAMETER_STORAGE_MISMATCH")
        metrics = dict(self.evaluator.evaluate(self.model, self.batches))
        # 通用任务 evaluator 的 loss/perplexity 越低越好，其余任务质量指标越高越好。
        directions = {
            name: (
                "lower_is_better"
                if name.casefold() in {"loss", "validation_loss", "perplexity"}
                else "higher_is_better"
            )
            for name in metrics
        }
        evidence = canonical_json_hash(
            {
                "schema_version": "stage7-evaluation-evidence-v1",
                "source_artifact_hash": self.source_artifact_hash,
                "environment_hash": self.environment_hash,
                "run_id": run_id,
                "context": thaw_json_value(context),
                "metrics": metrics,
            }
        )
        return EvaluationOutcome(
            evaluator_id="stage7-provider-task-evaluator-v1",
            metrics=metrics,
            metric_directions=directions,
            scope=self.scope,
            formal_eligible=self.scope == "formal",
            evidence_hash=evidence if self.scope == "formal" else None,
            metadata={
                "source_artifact_hash": self.source_artifact_hash,
                "evaluation_batch_ids": [batch.batch_id for batch in self.batches],
            },
        )


class _ArtifactAblationExecutor:
    """消费 hash-bound cell 证据的默认消融 executor。

    训练型 ablation cell 可以由 Stage 4--6 正式任务分别执行；Stage 8 只需在
    manifest 中声明 ``cell_id/config_hash/metrics`` 和证据 hash，无需再编写 Python
    glue。这里严格拒绝缺 cell、config 漂移和重复 cell。
    """

    def __init__(
        self,
        cells: Mapping[str, Mapping[str, object]],
        *,
        scope: str,
    ) -> None:
        self.cells = dict(cells)
        self.scope = scope

    def execute(
        self,
        cell: object,
        *,
        resolved_config: object,
        context: Mapping[str, object],
    ) -> EvaluationOutcome:
        cell_id = str(getattr(cell, "cell_id"))
        config_hash = str(getattr(cell, "config_hash"))
        evidence = self.cells.get(cell_id)
        if evidence is None:
            raise ValueError(f"STAGE8_CELL_EVIDENCE_MISSING:{cell_id}")
        if evidence.get("config_hash") != config_hash:
            raise ValueError(f"STAGE8_CELL_EVIDENCE_CONFIG_HASH_MISMATCH:{cell_id}")
        metrics = evidence.get("metrics")
        directions = evidence.get("metric_directions")
        evidence_hash = evidence.get("evidence_hash")
        if not isinstance(metrics, Mapping) or not isinstance(directions, Mapping):
            raise TypeError("STAGE8_CELL_EVIDENCE_METRICS_INVALID")
        return EvaluationOutcome(
            evaluator_id="stage8-hash-bound-cell-evidence-v1",
            metrics={str(name): float(value) for name, value in metrics.items()},
            metric_directions={str(name): str(value) for name, value in directions.items()},
            scope=self.scope,
            formal_eligible=self.scope == "formal",
            evidence_hash=str(evidence_hash) if self.scope == "formal" else None,
            metadata={"cell_id": cell_id, "source": "hash_bound_cell_manifest"},
        )


def _training_payload_document(
    inputs: Sequence[LoadedInputArtifact],
) -> tuple[LoadedInputArtifact, Mapping[str, object]]:
    matches: list[tuple[LoadedInputArtifact, Mapping[str, object]]] = []
    for document in inputs:
        payload = document.value
        if (
            isinstance(payload.get("training_result"), Mapping)
            and isinstance(payload.get("checkpoint_commits"), list)
            and isinstance(payload.get("importance_tensor_bundle_ref"), str)
            and isinstance(payload.get("importance_tensor_bundle_hash"), str)
        ):
            matches.append((document, payload))
    if len(matches) != 1:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "hash_bound_training_checkpoint_and_importance",
                "Stage 7 必须恰好引用一个同时绑定 checkpoint commit 与 importance bundle 的训练 task-output",
                True,
                tuple(item.ref for item in inputs),
            )
        )
    return matches[0]


def _estimator_family(training_result: Mapping[str, object]) -> str:
    records = training_result.get("records")
    if not isinstance(records, list):
        raise ValueError("STAGE7_TRAINING_RECORDS_INVALID")
    names = [
        str(item["estimator_name"])
        for item in records
        if isinstance(item, Mapping) and isinstance(item.get("estimator_name"), str)
    ]
    if not names:
        raise ValueError("STAGE7_TRAINING_ESTIMATOR_MISSING")
    name = names[-1].casefold()
    if "double" in name:
        return "double"
    if "_u" in name or name == "u" or "weighted_u" in name:
        return "u"
    if "raw" in name:
        return "raw"
    raise ValueError(f"STAGE7_TRAINING_ESTIMATOR_UNKNOWN:{names[-1]}")


def _evaluation_batches(
    request: TaskExecutionRequest,
    resources: object,
) -> tuple[TrainingMicrobatch, ...]:
    dataset = getattr(resources, "evaluation_dataset", None)
    evaluator = getattr(resources, "evaluator", None)
    if dataset is None or evaluator is None:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage7_evaluation_dataset",
                "Stage 7 正式剪枝要求 evaluation.enabled=true 且评估数据资产可用",
                True,
            )
        )
    identity = request.config.base_config.section("identity")
    evaluation = request.config.section("evaluation")
    runtime = request.config.base_config.section("runtime")
    assert isinstance(identity, dict) and isinstance(evaluation, dict) and isinstance(runtime, dict)
    cursor = dataset.cursor(seed=int(identity["master_seed"]), rank=0, world_size=1)
    batches: list[TrainingMicrobatch] = []
    maximum = int(evaluation["max_batches"] or 1)
    for _ in range(maximum):
        try:
            batches.extend(cursor.next_microbatches())
        except StopIteration:
            break
    if not batches:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage7_evaluation_batches",
                "Stage 7 评估数据游标为空",
                True,
            )
        )
    return tuple(batch.to(str(runtime["device"])) for batch in batches)


def _per_sequence_gradients(
    model: ModelAdapter,
    batches: Sequence[TrainingMicrobatch],
    parameter_names: Sequence[str],
) -> tuple[tuple[TensorMap, ...], tuple[float, ...]]:
    """按输入序列/分类样本生成 empirical Fisher 的显式统计单位梯度。"""

    named = dict(model.module.named_parameters())
    parameters = tuple(named[name] for name in parameter_names)
    gradients: list[TensorMap] = []
    weights: list[float] = []
    was_training = model.module.training
    model.module.eval()
    try:
        for batch in batches:
            batch_size = int(next(iter(batch.payload.values())).shape[0])
            for index in range(batch_size):
                unit = TrainingMicrobatch(
                    batch_id=f"{batch.batch_id}:unit-{index:06d}",
                    payload={name: tensor[index : index + 1] for name, tensor in batch.payload.items()},
                    sample_ids=(batch.sample_ids[index],),
                    metadata={"parent_batch_id": batch.batch_id},
                )
                loss = model.loss(unit)
                values = torch.autograd.grad(
                    loss.mean_loss,
                    parameters,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=True,
                )
                gradients.append(
                    TensorMap(
                        {
                            name: (
                                torch.zeros_like(parameter)
                                if gradient is None
                                else gradient.detach()
                            )
                            for name, parameter, gradient in zip(
                                parameter_names, parameters, values, strict=True
                            )
                        }
                    )
                )
                weights.append(float(loss.effective_count))
    finally:
        model.module.train(was_training)
    return tuple(gradients), tuple(weights)


def _tiny_pruning_resources(request: TaskExecutionRequest) -> PruningTaskResources:
    registry_hash = canonical_json_hash(
        {"schema_version": "stage7-tiny-registry-v1", "coordinates": ["layer.weight"]}
    )
    source_hash = canonical_json_hash(
        {"schema_version": "stage7-tiny-importance-v1", "fixture_id": "stage7-core-v1"}
    )
    source = ImportanceSourceSpec(
        method="magnitude",
        artifact_id="tiny-magnitude",
        artifact_hash=source_hash,
        coordinate_registry_hash=registry_hash,
        score_view="absolute",
        scope="local_fixture",
        metadata={"fixture": True},
    )
    study = PruningStudySpec(
        study_id="tiny-pruning-core-v1",
        sources=(source,),
        ratios=(0.25, 0.5),
        pruning_scopes=("global",),
        random_mask_seeds=(17,),
        run_intent="local_fixture",
    )
    parameters = TensorMap(
        {"layer.weight": torch.tensor([1.0, -2.0, 3.0, 0.5], dtype=torch.float64)}
    )
    scores = TensorMap(
        {"layer.weight": torch.tensor([1.0, 2.0, 3.0, 0.5], dtype=torch.float64)}
    )
    return PruningTaskResources(
        study=study,
        parameters=parameters,
        scores_by_artifact_hash={source_hash: scores},
        evaluator=_TinyPruningEvaluator(),
        model_checkpoint_hash=canonical_json_hash(
            {"schema_version": "stage7-tiny-checkpoint-v1", "fixture_id": "stage7-core-v1"}
        ),
        coordinate_registry_hash=registry_hash,
    )


def _default_pruning_resources(
    request: TaskExecutionRequest,
    inputs: tuple[LoadedInputArtifact, ...],
    *,
    workspace_root: Path,
    output_root: Path,
) -> PruningTaskResources:
    """无需 Python 注入地从训练 task-output 重建 Stage 7 正式资源。"""

    route_documents = [
        document
        for document in inputs
        if document.value.get("schema_version") == "stage456-route-execution-v1"
    ]
    route_groups = {
        str(document.value.get("route_result", {}).get("artifact_hash")): document
        for document in route_documents
        if isinstance(document.value.get("route_result"), Mapping)
    }
    if route_documents and len(route_groups) != 1:
        raise ValueError("STAGE7_MULTIPLE_DISTINCT_ROUTE_EXECUTIONS")
    if route_groups:
        source = load_route_training_source(
            workspace_root,
            next(iter(route_groups.values())),
        )
    else:
        training_document, payload = _training_payload_document(inputs)
        training_result = payload["training_result"]
        assert isinstance(training_result, Mapping)
        state_value = training_result.get("state")
        if not isinstance(state_value, Mapping) or not isinstance(
            state_value.get("last_checkpoint_id"), str
        ):
            raise ValueError("STAGE7_TRAINING_LAST_CHECKPOINT_MISSING")
        checkpoint_id = str(state_value["last_checkpoint_id"])
        raw_commits = payload["checkpoint_commits"]
        assert isinstance(raw_commits, list)
        matching = [
            item
            for item in raw_commits
            if isinstance(item, Mapping) and item.get("checkpoint_id") == checkpoint_id
        ]
        if len(matching) != 1:
            raise ValueError("STAGE7_LAST_CHECKPOINT_COMMIT_NOT_UNIQUE")
        checkpoint_entry = matching[0]
        if set(checkpoint_entry) != {
            "checkpoint_id",
            "commit_ref",
            "commit_identity_sha256",
            "bundle_manifest_sha256",
        }:
            raise ValueError("STAGE7_CHECKPOINT_ENTRY_FIELDS_MISMATCH")
        checkpoint_state, _checkpoint_commit = load_checkpoint_state_from_commit(
            workspace_root,
            commit_ref=str(checkpoint_entry["commit_ref"]),
            expected_identity_sha256=str(
                checkpoint_entry["commit_identity_sha256"]
            ),
            expected_bundle_manifest_sha256=str(
                checkpoint_entry["bundle_manifest_sha256"]
            ),
        )

        bundle_ref = str(payload["importance_tensor_bundle_ref"])
        bundle_state, bundle = load_tensor_bundle(
            _safe_workspace_path(
                workspace_root,
                bundle_ref,
                field="importance_tensor_bundle_ref",
            )
        )
        if bundle.manifest_sha256 != payload["importance_tensor_bundle_hash"]:
            raise ValueError("STAGE7_IMPORTANCE_BUNDLE_HASH_MISMATCH")
        if not isinstance(bundle_state, Mapping) or set(bundle_state) != {
            "metadata",
            "accumulator",
        }:
            raise ValueError("STAGE7_IMPORTANCE_BUNDLE_STATE_INVALID")
        accumulator = bundle_state["accumulator"]
        if not isinstance(accumulator, Mapping):
            raise TypeError("STAGE7_IMPORTANCE_ACCUMULATOR_NOT_OBJECT")
        source = Stage7TrainingSource(
            document=training_document,
            training_result=training_result,
            checkpoint_state=checkpoint_state,
            checkpoint_identity_hash=str(
                checkpoint_entry["commit_identity_sha256"]
            ),
            importance_state_hash=bundle.manifest_sha256,
            accumulator=accumulator,
        )

    training_document = source.document
    training_result = source.training_result
    checkpoint_state = source.checkpoint_state
    accumulator = source.accumulator
    # 后续循环会创建多个 ImportanceSourceSpec；保留训练源的独立名字，避免循环变量
    # 覆盖 checkpoint/importance 的权威身份。
    training_source = source
    registry_hash = checkpoint_state.get("registry_hash")
    if not isinstance(registry_hash, str) or len(registry_hash) != 64:
        raise ValueError("STAGE7_CHECKPOINT_REGISTRY_HASH_INVALID")
    if training_result.get("registry_hash") != registry_hash:
        raise ValueError("STAGE7_TRAINING_CHECKPOINT_REGISTRY_MISMATCH")

    # provider 层只允许本地已有资产；缺 dependency/manifest 时转换成可重试 BLOCKED，
    # 绝不让正式 CLI 因未实现 loader 或隐式下载而崩溃。
    try:
        resources = _training_resources(request, workspace_root, rank=0, world_size=1)
    except TaskBlockedError:
        raise
    except Exception as error:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage7_model_and_data_assets",
                f"无法从严格离线 provider 构建剪枝模型/数据：{type(error).__name__}",
                True,
                (training_document.ref,),
            )
        ) from error
    model = resources.model
    model_state = checkpoint_state.get("model")
    if not isinstance(model_state, Mapping):
        raise ValueError("STAGE7_CHECKPOINT_MODEL_STATE_INVALID")
    if model_state and all(str(name).startswith("module.") for name in model_state):
        model_state = {
            str(name)[len("module.") :]: tensor for name, tensor in model_state.items()
        }
    model.module.load_state_dict(model_state, strict=True)  # type: ignore[arg-type]
    runtime = request.config.base_config.section("runtime")
    assert isinstance(runtime, dict)
    model.module.to(str(runtime["device"]))

    template_mapping = accumulator.get("magnitude")
    if not isinstance(template_mapping, Mapping) or not template_mapping:
        raise ValueError("STAGE7_ACCUMULATOR_MAGNITUDE_INVALID")
    parameter_names = tuple(str(name) for name in template_mapping)
    named_parameters = dict(model.module.named_parameters())
    if set(parameter_names) - set(named_parameters):
        raise ValueError("STAGE7_MODEL_MISSING_ACCUMULATOR_COORDINATES")
    parameters = TensorMap({name: named_parameters[name] for name in parameter_names})
    batches = _evaluation_batches(request, resources)
    per_unit_gradients, per_unit_weights = _per_sequence_gradients(
        model, batches, parameter_names
    )
    optimizer = request.config.base_config.section("optimizer")
    importance = request.config.base_config.section("importance")
    assert isinstance(optimizer, dict) and isinstance(importance, dict)
    production = produce_baseline_scores(
        parameters,
        accumulator,
        estimator_name=_estimator_family(training_result),
        per_unit_gradients=per_unit_gradients,
        per_unit_weights=per_unit_weights,
        # 当训练路线恰好是无动量、无裁剪 SGD 时，训练累计的 raw 与端点移动
        # 足以精确恢复 SI；AdamW/动量/裁剪路线则保持 fail-closed，不能用近似值
        # 冒充真实 SI。显式历史生产器仍可在这些路线中提供 SI artifact。
        optimizer_type=str(optimizer["type"]),
        optimizer_momentum=float(optimizer["momentum"]),
        clip_mode=str(importance["clip_mode"]),
    )

    bundle_root = output_root / "core" / "importance-baselines"
    source_by_method: dict[str, ImportanceSourceSpec] = {}
    score_by_hash: dict[str, TensorMap] = {}
    for method in IMPORTANCE_METHODS:
        score = production.scores.get(method)
        if score is None:
            continue
        path = bundle_root / method
        state = {
            "schema_version": "importance-baseline-tensor-v1",
            "method": method,
            "coordinate_registry_hash": registry_hash,
            "source_training_artifact_hash": training_document.artifact_hash,
            "source_checkpoint_identity_hash": training_source.checkpoint_identity_hash,
            "source_importance_state_hash": training_source.importance_state_hash,
            "score": score.to_dict(clone=True),
            "producer_metadata": dict(production.metadata),
        }
        if path.exists():
            restored, identity = load_tensor_bundle(path)
            # 已存在目录只允许是同一安全状态；否则要求人工清理冲突输出。
            if not isinstance(restored, Mapping) or restored.get("method") != method:
                raise ValueError("STAGE7_BASELINE_BUNDLE_CONFLICT")
        else:
            identity = publish_tensor_bundle(path, state)
        logical_ref = path.relative_to(workspace_root).as_posix()
        score_view = "signed" if method in {"u", "double"} else "absolute"
        source = ImportanceSourceSpec(
            method=method,
            artifact_id=f"{method}-{identity.manifest_sha256[:16]}",
            artifact_hash=identity.manifest_sha256,
            coordinate_registry_hash=registry_hash,
            score_view=score_view,
            scope=request.config.run_intent,
            metadata={
                "tensor_bundle_ref": logical_ref,
                "producer": "stage7.default-baseline-producer.v1",
                "source_training_artifact_hash": training_document.artifact_hash,
            },
        )
        source_by_method[method] = source
        score_by_hash[source.artifact_hash] = score

    referenced: list[tuple[LoadedInputArtifact, PruningStudySpec]] = []
    for document in inputs:
        for mapping in _find_mappings(document.value):
            if mapping.get("schema_version") == "pruning-study-spec-v1":
                referenced.append((document, _study_from_mapping(mapping)))
    if referenced:
        unique = {pruning_study_hash(study): (document, study) for document, study in referenced}
        if len(unique) != 1:
            raise ValueError("STAGE7_MULTIPLE_PRUNING_STUDIES")
        matrix_document, study = next(iter(unique.values()))
        if study.run_intent != request.config.run_intent:
            raise ValueError("STAGE7_REFERENCED_STUDY_SCOPE_MISMATCH")
        missing = [
            source.method
            for source in study.sources
            if source.method not in source_by_method
            or source_by_method[source.method].artifact_hash != source.artifact_hash
        ]
        if missing:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "frozen_pruning_importance_sources",
                    f"冻结 pruning matrix 的 source 无法由当前训练 artifact 重建：{missing}",
                    True,
                    (matrix_document.ref, training_document.ref),
                )
            )
        scores = {source.artifact_hash: score_by_hash[source.artifact_hash] for source in study.sources}
        matrix_input_hash = matrix_document.artifact_hash
    else:
        pruning = request.config.base_config.section("pruning")
        assert isinstance(pruning, dict)
        if not pruning["enabled"] or not pruning["ratios"]:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "pruning_ratio_grid",
                    "正式 Stage 7 matrix 要求 config.pruning.enabled=true 且 ratios 非空",
                    True,
                )
            )
        study = PruningStudySpec(
            study_id=f"stage7-{request.config.config_hash[:20]}",
            sources=tuple(source_by_method[name] for name in IMPORTANCE_METHODS if name in source_by_method),
            ratios=tuple(float(value) for value in pruning["ratios"]),  # type: ignore[arg-type]
            pruning_scopes=(str(pruning["scope"]),),
            random_mask_seeds=tuple(range(int(pruning["random_repetitions"]))),
            run_intent=request.config.run_intent,
        )
        scores = score_by_hash
        matrix_input_hash = None
    evaluator = _TorchTaskPruningEvaluator(
        model,
        resources.evaluator,  # type: ignore[arg-type]
        batches,
        source_artifact_hash=training_document.artifact_hash,
        environment_hash=request.environment.environment_hash,
        scope=request.config.run_intent,
    )
    return PruningTaskResources(
        study=study,
        parameters=parameters,
        scores_by_artifact_hash=scores,
        evaluator=evaluator,
        model_checkpoint_hash=training_source.checkpoint_identity_hash,
        coordinate_registry_hash=registry_hash,
        formal_authorization_hash=(
            request.environment.environment_hash
            if request.config.run_intent == "formal"
            else None
        ),
        checkpoint_input_hash=training_document.artifact_hash,
        importance_input_hashes=(training_document.artifact_hash,),
        matrix_input_hash=matrix_input_hash,
    )


def _tiny_ablation_resources(request: TaskExecutionRequest) -> AblationTaskResources:
    matrix = AblationMatrix.compile(
        matrix_id="tiny-stage8-core-v1",
        base_config={
            "optimizer": {"learning_rate": 0.08},
            "training": {"max_steps": 1},
        },
        factors=(
            AblationFactor(
                name="learning_rate",
                config_path=("optimizer", "learning_rate"),
                baseline_value=0.08,
                alternatives=(0.16,),
            ),
        ),
        base_seed=23,
    )
    return AblationTaskResources(
        matrix=matrix,
        executor=_TinyAblationExecutor(),
        config_validator=lambda value: dict(value),
        training_builder=TinyAblationTrainingBuilder(),
        source_checkpoint_artifact_hash=canonical_json_hash(
            {"schema_version": "tiny-ablation-initialization-v1", "seed": 23}
        ),
    )


def _default_ablation_resources(
    request: TaskExecutionRequest,
    inputs: tuple[LoadedInputArtifact, ...],
) -> AblationTaskResources:
    """从冻结 matrix 与 cell-evidence manifest 构建无需代码注入的 executor。"""

    matrix = _ablation_matrix_from_documents(inputs)
    matrix_documents = [
        document
        for document in inputs
        if any(
            mapping.get("schema_version") == "ablation-matrix-v1"
            and AblationMatrix.from_mapping(mapping).digest == matrix.digest
            for mapping in _find_mappings(document.value)
        )
    ]
    if len(matrix_documents) != 1:
        raise ValueError("STAGE8_MATRIX_DOCUMENT_NOT_UNIQUE")
    manifests: list[tuple[LoadedInputArtifact, Mapping[str, object]]] = []
    for document in inputs:
        for mapping in _find_mappings(document.value):
            if mapping.get("schema_version") == "ablation-cell-evidence-manifest-v1":
                manifests.append((document, mapping))
    if len(manifests) != 1:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "hash_bound_ablation_cell_evidence",
                "Stage 8 execute 要求恰好一个 ablation-cell-evidence-manifest-v1；可由各 cell 的正式训练结果声明式汇总",
                True,
                tuple(item.ref for item in inputs),
            )
        )
    evidence_document, manifest = manifests[0]
    expected_fields = {
        "schema_version",
        "matrix_hash",
        "checkpoint_artifact_hash",
        "scope",
        "formal_eligible",
        "cells",
        "artifact_hash",
    }
    if set(manifest) != expected_fields:
        raise ValueError("STAGE8_CELL_EVIDENCE_MANIFEST_FIELDS_MISMATCH")
    declared_hash = manifest["artifact_hash"]
    without_hash = dict(manifest)
    without_hash.pop("artifact_hash")
    if canonical_json_hash(without_hash) != declared_hash:
        raise ValueError("STAGE8_CELL_EVIDENCE_MANIFEST_HASH_MISMATCH")
    if manifest["matrix_hash"] != matrix.digest:
        raise ValueError("STAGE8_CELL_EVIDENCE_MATRIX_HASH_MISMATCH")
    if manifest["scope"] != request.config.run_intent or manifest["formal_eligible"] is not (
        request.config.run_intent == "formal"
    ):
        raise ValueError("STAGE8_CELL_EVIDENCE_SCOPE_MISMATCH")
    checkpoint_hash = manifest["checkpoint_artifact_hash"]
    checkpoint_documents = [item for item in inputs if item.artifact_hash == checkpoint_hash]
    if len(checkpoint_documents) != 1:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "hash_bound_ablation_checkpoint",
                "cell evidence 引用的 checkpoint task-output 不在当前输入集合",
                True,
                (evidence_document.ref,),
            )
        )
    raw_cells = manifest["cells"]
    if not isinstance(raw_cells, list):
        raise TypeError("STAGE8_CELL_EVIDENCE_CELLS_NOT_ARRAY")
    cells: dict[str, Mapping[str, object]] = {}
    required_cell_fields = {
        "cell_id",
        "config_hash",
        "metrics",
        "metric_directions",
        "evidence_hash",
        "result_ref",
    }
    for item in raw_cells:
        if not isinstance(item, Mapping) or set(item) != required_cell_fields:
            raise ValueError("STAGE8_CELL_EVIDENCE_FIELDS_MISMATCH")
        cell_id = item["cell_id"]
        if not isinstance(cell_id, str) or cell_id in cells:
            raise ValueError("STAGE8_CELL_EVIDENCE_ID_INVALID_OR_DUPLICATE")
        if not isinstance(item["evidence_hash"], str) or len(str(item["evidence_hash"])) != 64:
            raise ValueError("STAGE8_CELL_EVIDENCE_HASH_INVALID")
        cells[cell_id] = item
    expected_cells = {cell.cell_id: cell.config_hash for cell in matrix.cells}
    if set(cells) != set(expected_cells) or any(
        cells[cell_id].get("config_hash") != config_hash
        for cell_id, config_hash in expected_cells.items()
    ):
        raise ValueError("STAGE8_CELL_EVIDENCE_MATRIX_COVERAGE_MISMATCH")
    return AblationTaskResources(
        matrix=matrix,
        executor=_ArtifactAblationExecutor(cells, scope=request.config.run_intent),
        config_validator=lambda value: dict(value),
        formal_authorization_hash=(
            request.environment.environment_hash
            if request.config.run_intent == "formal"
            else None
        ),
        checkpoint_input_hash=checkpoint_documents[0].artifact_hash,
        matrix_input_hash=matrix_documents[0].artifact_hash,
    )


def _study_to_dict(study: PruningStudySpec) -> dict[str, JSONValue]:
    return {
        "schema_version": "pruning-study-spec-v1",
        "study_id": study.study_id,
        "sources": [
            {
                "method": source.method,
                "artifact_id": source.artifact_id,
                "artifact_hash": source.artifact_hash,
                "coordinate_registry_hash": source.coordinate_registry_hash,
                "score_view": source.score_view,
                "available": source.available,
                "scope": source.scope,
                "metadata": thaw_json_value(source.metadata),
            }
            for source in study.sources
        ],
        "ratios": list(study.ratios),
        "pruning_scopes": list(study.pruning_scopes),
        "random_mask_seeds": list(study.random_mask_seeds),
        "run_intent": study.run_intent,
        "frozen": study.frozen,
        "study_hash": pruning_study_hash(study),
    }


def _study_from_mapping(value: Mapping[str, object]) -> PruningStudySpec:
    """严格重载本模块发布的 ``pruning-study-spec-v1``。"""

    expected = {
        "schema_version",
        "study_id",
        "sources",
        "ratios",
        "pruning_scopes",
        "random_mask_seeds",
        "run_intent",
        "frozen",
        "study_hash",
    }
    if set(value) != expected or value.get("schema_version") != "pruning-study-spec-v1":
        raise ValueError("STAGE7_PRUNING_STUDY_FIELDS_OR_SCHEMA_MISMATCH")
    if type(value.get("frozen")) is not bool:
        raise TypeError("STAGE7_PRUNING_STUDY_FROZEN_NOT_BOOLEAN")
    if any(
        not isinstance(value.get(field), str) or not value.get(field)
        for field in ("study_id", "run_intent", "study_hash")
    ):
        raise TypeError("STAGE7_PRUNING_STUDY_STRING_FIELD_INVALID")
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        raise TypeError("STAGE7_PRUNING_STUDY_SOURCES_NOT_ARRAY")
    source_expected = {
        "method",
        "artifact_id",
        "artifact_hash",
        "coordinate_registry_hash",
        "score_view",
        "available",
        "scope",
        "metadata",
    }
    parsed_sources: list[ImportanceSourceSpec] = []
    for item in sources:
        if not isinstance(item, Mapping) or set(item) != source_expected:
            raise ValueError("STAGE7_PRUNING_SOURCE_FIELDS_MISMATCH")
        if type(item.get("available")) is not bool:
            raise TypeError("STAGE7_PRUNING_SOURCE_AVAILABLE_NOT_BOOLEAN")
        metadata = item.get("metadata")
        if not isinstance(metadata, Mapping):
            raise TypeError("STAGE7_PRUNING_SOURCE_METADATA_NOT_OBJECT")
        parsed_sources.append(
            ImportanceSourceSpec(
                method=item["method"],  # type: ignore[arg-type]
                artifact_id=item["artifact_id"],  # type: ignore[arg-type]
                artifact_hash=item["artifact_hash"],  # type: ignore[arg-type]
                coordinate_registry_hash=item["coordinate_registry_hash"],  # type: ignore[arg-type]
                score_view=item["score_view"],  # type: ignore[arg-type]
                available=item["available"],  # type: ignore[arg-type]
                scope=item["scope"],  # type: ignore[arg-type]
                metadata=metadata,
            )
        )
    ratios = value.get("ratios")
    scopes = value.get("pruning_scopes")
    seeds = value.get("random_mask_seeds")
    if not isinstance(ratios, list) or not isinstance(scopes, list) or not isinstance(seeds, list):
        raise TypeError("STAGE7_PRUNING_STUDY_ARRAY_FIELD_INVALID")
    if (
        any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in ratios)
        or any(not isinstance(item, str) or not item for item in scopes)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in seeds)
    ):
        raise TypeError("STAGE7_PRUNING_STUDY_ARRAY_ITEM_INVALID")
    study = PruningStudySpec(
        study_id=value["study_id"],  # type: ignore[arg-type]
        sources=tuple(parsed_sources),
        ratios=tuple(ratios),  # type: ignore[arg-type]
        pruning_scopes=tuple(scopes),  # type: ignore[arg-type]
        random_mask_seeds=tuple(seeds),  # type: ignore[arg-type]
        run_intent=value["run_intent"],  # type: ignore[arg-type]
        frozen=value["frozen"],  # type: ignore[arg-type]
    )
    if value.get("study_hash") != pruning_study_hash(study):
        raise ValueError("STAGE7_PRUNING_STUDY_HASH_MISMATCH")
    return study


def _run_to_dict(run: object) -> dict[str, JSONValue]:
    return {
        "run_id": str(getattr(run, "run_id")),
        "method": str(getattr(run, "method")),
        "direction": str(getattr(run, "direction")),
        "pruning_scope": str(getattr(run, "pruning_scope")),
        "ratio": float(getattr(run, "ratio")),
        "mask_seed": getattr(run, "mask_seed"),
        "source_artifact_hash": getattr(run, "source_artifact_hash"),
        "coordinate_registry_hash": str(getattr(run, "coordinate_registry_hash")),
        "tie_breaker": str(getattr(run, "tie_breaker")),
    }


def _formal_loader_blocker(kind: str) -> TaskBlockedError:
    return TaskBlockedError(
        TaskBlocker(
            BlockerCode.CAPABILITY_UNAVAILABLE,
            f"formal_{kind}_resource_loader",
            f"formal {kind} 必须由工厂注入真实、hash-bound 资源 loader",
            True,
        )
    )


def _validate_pruning_resources(
    request: TaskExecutionRequest,
    resources: PruningTaskResources,
    inputs: Sequence[LoadedInputArtifact],
) -> None:
    if resources.study.run_intent != request.config.run_intent:
        raise ValueError("STAGE7_RESOURCE_SCOPE_MISMATCH")
    if resources.study.coordinate_registry_hash != resources.coordinate_registry_hash:
        raise ValueError("STAGE7_RESOURCE_REGISTRY_HASH_MISMATCH")
    expected_sources = {source.artifact_hash for source in resources.study.sources}
    if set(resources.scores_by_artifact_hash) != expected_sources:
        raise ValueError("STAGE7_RESOURCE_IMPORTANCE_SET_MISMATCH")
    if request.config.run_intent == "formal":
        if resources.formal_authorization_hash != request.environment.environment_hash:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.GATE_NOT_READY,
                    "formal_pruning_authorization",
                    "formal pruning 资源未绑定当前 Gate/能力环境快照",
                    True,
                )
            )
        available_hashes = {item.artifact_hash for item in inputs}
        if resources.checkpoint_input_hash not in available_hashes:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "hash_bound_checkpoint",
                    "formal pruning loader 未声明由当前输入中的 checkpoint artifact 构建",
                    True,
                    tuple(item.ref for item in inputs),
                )
            )
        importance_hashes = tuple(resources.importance_input_hashes)
        if (
            not importance_hashes
            or len(importance_hashes) != len(set(importance_hashes))
            or not set(importance_hashes).issubset(available_hashes)
        ):
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
                    "hash_bound_importance",
                    "formal pruning loader 未绑定当前输入中的 importance artifact 集合",
                    True,
                    tuple(item.ref for item in inputs),
                )
            )
        if (
            request.task.task_id == "stage7.evaluate"
            and resources.matrix_input_hash not in available_hashes
        ):
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "hash_bound_pruning_matrix",
                    "formal pruning evaluation 未绑定当前输入中的冻结 matrix artifact",
                    True,
                    tuple(item.ref for item in inputs),
                )
            )


def _validate_ablation_resources(
    request: TaskExecutionRequest,
    resources: AblationTaskResources,
    inputs: Sequence[LoadedInputArtifact],
) -> None:
    if request.config.run_intent == "formal" and (
        resources.formal_authorization_hash != request.environment.environment_hash
    ):
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.GATE_NOT_READY,
                "formal_ablation_authorization",
                "formal ablation 资源未绑定当前 Gate/能力环境快照",
                True,
            )
        )
    if request.config.run_intent == "formal":
        available_hashes = {item.artifact_hash for item in inputs}
        if resources.checkpoint_input_hash not in available_hashes:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "hash_bound_checkpoint",
                    "formal ablation loader 未绑定当前输入中的 checkpoint artifact",
                    True,
                    tuple(item.ref for item in inputs),
                )
            )
        if resources.matrix_input_hash not in available_hashes:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "hash_bound_ablation_matrix",
                    "formal ablation loader 未绑定当前输入中的冻结 matrix artifact",
                    True,
                    tuple(item.ref for item in inputs),
                )
            )


def _tiny_stage_rows(*, scope: str = "local_fixture") -> StageArtifactRows:
    rows = [
        {"condition": "baseline", "replicate": 0, "pair_id": "0", "metric_name": "utility", "ratio": 0.0, "value": 1.00, "damage": 0.00},
        {"condition": "baseline", "replicate": 1, "pair_id": "1", "metric_name": "utility", "ratio": 0.0, "value": 0.98, "damage": 0.02},
        {"condition": "candidate", "replicate": 0, "pair_id": "0", "metric_name": "utility", "ratio": 0.5, "value": 0.82, "damage": 0.18},
        {"condition": "candidate", "replicate": 1, "pair_id": "1", "metric_name": "utility", "ratio": 0.5, "value": 0.80, "damage": 0.20},
    ]
    source_hash = canonical_json_hash(
        {"schema_version": "stage9-tiny-source-v1", "rows": rows}
    )
    return StageArtifactRows.create(
        artifact_id="stage9-tiny-fixture",
        source_schema_version="stage9-tiny-source-v1",
        source_artifact_hash=source_hash,
        stage=9,
        adapter_id="stage9.tiny_rows.v1",
        rows=rows,
        scope=scope,
        formal_eligible=scope == "formal",
        metadata={"fixture": scope == "local_fixture"},
    )


def _find_mappings(value: object) -> Sequence[Mapping[str, object]]:
    found: list[Mapping[str, object]] = []
    if isinstance(value, Mapping):
        found.append(value)
        for child in value.values():
            found.extend(_find_mappings(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_mappings(child))
    return found


def _analysis_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    default_condition: str,
    default_metric: str,
) -> tuple[Mapping[str, object], ...]:
    """为异构 Stage 0--8 行补齐 Stage 9 的共同分析键，不覆盖原始列。"""

    normalized: list[Mapping[str, object]] = []
    for index, raw in enumerate(rows):
        row = dict(raw)
        numeric_candidates = [
            name
            for name in (
                "value",
                "damage",
                "directed_effect",
                "mean_loss",
                "mean",
                "mse",
                "mae",
                "bias",
                "conservative_error",
                "completeness_absolute_residual",
            )
            if isinstance(row.get(name), (int, float))
            and not isinstance(row.get(name), bool)
        ]
        if "value" not in row:
            if not numeric_candidates:
                # 纯 lineage/index artifact 没有科学数值，不能伪造 0；这类 mapping
                # 仍由输入 hash 审计，但不进入数值源表。
                continue
            row["value"] = float(row[numeric_candidates[0]])
        if not isinstance(row["value"], (int, float)) or isinstance(row["value"], bool):
            continue
        row.setdefault(
            "condition",
            row.get("method")
            or row.get("phase_type")
            or row.get("estimator")
            or default_condition,
        )
        row.setdefault("metric_name", row.get("metric") or (numeric_candidates[0] if numeric_candidates else default_metric))
        row.setdefault("replicate", index)
        row.setdefault(
            "pair_id",
            str(
                row.get("pair_id")
                or row.get("seed")
                or row.get("global_step")
                or row.get("replicate")
            ),
        )
        normalized.append(row)
    return tuple(normalized)


def _verify_nested_artifact_hash(value: Mapping[str, object]) -> None:
    declared = value.get("artifact_hash")
    if declared is None:
        return
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("STAGE9_NESTED_ARTIFACT_HASH_INVALID")
    payload = dict(value)
    payload.pop("artifact_hash")
    if canonical_json_hash(payload) != declared:
        raise ValueError("STAGE9_NESTED_ARTIFACT_HASH_MISMATCH")


def _rows_from_documents(
    documents: Sequence[LoadedInputArtifact],
    *,
    scope: str,
) -> tuple[StageArtifactRows, ...]:
    """直接适配 Stage 0--8 的 task-output envelope，不要求临时转换脚本。

    外层 commit/object hash 已由 :func:`_load_input_artifact` 复核；每个内层带
    ``artifact_hash`` 的对象还会再次校验。生成的 source artifact 始终把外层
    ``document.artifact_hash`` 作为父身份，因此把 payload 从另一个 commit 搬来会
    改变 Stage 9 lineage。
    """

    rows_artifacts: list[StageArtifactRows] = []
    for document_index, document in enumerate(documents):
        emitted: set[tuple[str, str]] = set()

        def add(
            *,
            schema: str,
            adapter: str,
            stage: int,
            rows: Sequence[Mapping[str, object]],
            condition: str,
            metric: str,
        ) -> None:
            normalized = _analysis_rows(
                rows, default_condition=condition, default_metric=metric
            )
            if not normalized:
                return
            fingerprint = canonical_json_hash([dict(row) for row in normalized])
            key = (adapter, fingerprint)
            if key in emitted:
                return
            emitted.add(key)
            rows_artifacts.append(
                StageArtifactRows.create(
                    artifact_id=f"input-{document.artifact_hash[:12]}-{len(emitted):03d}",
                    source_schema_version=schema,
                    source_artifact_hash=document.artifact_hash,
                    stage=stage,
                    adapter_id=adapter,
                    rows=normalized,
                    scope=scope,
                    formal_eligible=scope == "formal",
                    metadata={
                        "input_ref": document.ref,
                        "input_formal_eligible": document.formal_eligible,
                    },
                )
            )

        for mapping in _find_mappings(document.value):
            schema = mapping.get("schema_version")
            if not isinstance(schema, str):
                # FrozenSourceTable 的 schema_version 是业务版本，靠精确字段识别。
                continue
            if schema == "stage-artifact-rows-v1":
                artifact = StageArtifactRows.from_mapping(mapping)
                if (
                    document.artifact_hash == artifact.artifact_hash
                    and artifact.scope == scope
                    and artifact.formal_eligible is (scope == "formal")
                ):
                    # direct canonical StageArtifactRows 已经把自身 hash 作为权威父
                    # 身份；保持原对象可兼容既有 lineage。嵌套在 task envelope 中
                    # 时则走下方 adapter，把 envelope hash 加入父链。
                    rows_artifacts.append(artifact)
                    emitted.add(("stage9.direct_stage_rows.v1", artifact.artifact_hash))
                    continue
                add(
                    schema=artifact.source_schema_version,
                    adapter="stage9.stage_rows_envelope.v1",
                    stage=artifact.stage,
                    rows=artifact.rows,
                    condition=f"stage{artifact.stage}",
                    metric="value",
                )
                continue
            if schema == "bound-frozen-source-table-v1":
                source = BoundSourceTable.from_mapping(mapping)
                add(
                    schema=source.table.schema_version,
                    adapter="stage9.bound_source_passthrough.v1",
                    stage=9,
                    rows=source.table.rows,
                    condition=source.table.name,
                    metric="value",
                )
                continue
            if set(mapping) == {"name", "schema_version", "rows", "content_hash", "frozen"}:
                table = FrozenSourceTable.from_mapping(mapping)
                add(
                    schema=table.schema_version,
                    adapter="stage9.frozen_source_table.v1",
                    stage=max(0, min(9, int(document.value.get("stage", 9)))),
                    rows=table.rows,
                    condition=table.name,
                    metric="value",
                )
                continue
            if schema == "ablation-study-result-v1":
                study = AblationStudyResult.from_mapping(mapping)
                add(
                    schema=study.schema_version,
                    adapter="stage9.ablation_study.v1",
                    stage=8,
                    rows=study.rows,
                    condition="ablation",
                    metric="directed_effect",
                )
                continue
            if schema == "training-run-result-v1":
                _verify_nested_artifact_hash(mapping)
                records = mapping.get("records")
                if not isinstance(records, list) or not all(isinstance(row, Mapping) for row in records):
                    raise TypeError("STAGE9_TRAINING_RECORDS_INVALID")
                add(
                    schema=schema,
                    adapter="stage9.training_run_result.v1",
                    stage=int(document.value.get("stage", 1)),
                    rows=[
                        {**dict(row), "run_id": mapping.get("run_id"), "value": row.get("mean_loss")}
                        for row in records
                        if row.get("mean_loss") is not None
                    ],
                    condition=str(mapping.get("run_id", "training")),
                    metric="mean_loss",
                )
                continue
            if schema == "importance-trajectory-v1":
                _verify_nested_artifact_hash(mapping)
                points = mapping.get("points")
                if not isinstance(points, list):
                    raise TypeError("STAGE9_IMPORTANCE_TRAJECTORY_POINTS_INVALID")
                trajectory_rows: list[Mapping[str, object]] = []
                for point in points:
                    if not isinstance(point, Mapping) or not isinstance(point.get("snapshot"), Mapping):
                        raise TypeError("STAGE9_IMPORTANCE_TRAJECTORY_POINT_INVALID")
                    snapshot = point["snapshot"]
                    summaries = snapshot.get("scalar_summaries")
                    if not isinstance(summaries, Mapping):
                        raise TypeError("STAGE9_IMPORTANCE_SUMMARIES_INVALID")
                    for name, value in summaries.items():
                        trajectory_rows.append(
                            {
                                "global_step": point.get("global_step"),
                                "checkpoint_id": point.get("checkpoint_id"),
                                "metric_name": str(name),
                                "value": value,
                                "condition": str(name),
                                "pair_id": str(point.get("global_step")),
                            }
                        )
                add(
                    schema=schema,
                    adapter="stage9.importance_trajectory.v1",
                    stage=int(document.value.get("stage", 5)),
                    rows=trajectory_rows,
                    condition="importance",
                    metric="importance_mass",
                )
                continue
            if schema == "stage2-paired-wave-summary-v1":
                _verify_nested_artifact_hash(mapping)
                stats = mapping.get("method_statistics")
                if not isinstance(stats, Mapping):
                    raise TypeError("STAGE9_STAGE2_METHOD_STATISTICS_INVALID")
                rows = [
                    {
                        "condition": str(method),
                        "metric_name": str(metric_name),
                        "value": metric_value,
                        "pair_id": str(method),
                    }
                    for method, values in stats.items()
                    if isinstance(values, Mapping)
                    for metric_name, metric_value in values.items()
                    if isinstance(metric_value, (int, float)) and not isinstance(metric_value, bool)
                ]
                add(
                    schema=schema,
                    adapter="stage9.stage2_estimator_summary.v1",
                    stage=2,
                    rows=rows,
                    condition="estimator",
                    metric="estimator_metric",
                )
                continue
            if schema == "path-integral-result-v1":
                _verify_nested_artifact_hash(mapping)
                metric_fields = (
                    "loss_drop",
                    "completeness_absolute_residual",
                    "completeness_relative_residual",
                    "completeness_l1_scaled_residual",
                    "unique_gradient_evaluations",
                )
                rows = [
                    {
                        "condition": str(
                            mapping.get("rule", {}).get("name", "path")
                            if isinstance(mapping.get("rule"), Mapping)
                            else "path"
                        ),
                        "metric_name": name,
                        "value": mapping[name],
                        "pair_id": str(mapping.get("path_identity_hash")),
                    }
                    for name in metric_fields
                    if isinstance(mapping.get(name), (int, float))
                    and not isinstance(mapping.get(name), bool)
                ]
                add(
                    schema=schema,
                    adapter="stage9.stage3_path_integral.v1",
                    stage=3,
                    rows=rows,
                    condition="path",
                    metric="path_metric",
                )
                continue
            if schema == "stage456-route-execution-v1":
                route_rows: list[Mapping[str, object]] = []
                training_metrics = mapping.get("training_metrics")
                if isinstance(training_metrics, list):
                    for row in training_metrics:
                        if not isinstance(row, Mapping):
                            raise TypeError("STAGE9_ROUTE_TRAINING_METRIC_INVALID")
                        for name in ("mean_training_loss", "final_training_loss"):
                            if isinstance(row.get(name), (int, float)):
                                route_rows.append({**dict(row), "metric_name": name, "value": row[name]})
                evaluation_metrics = mapping.get("evaluation_metrics")
                if isinstance(evaluation_metrics, list):
                    for row in evaluation_metrics:
                        if not isinstance(row, Mapping) or not isinstance(row.get("metrics"), Mapping):
                            raise TypeError("STAGE9_ROUTE_EVALUATION_METRIC_INVALID")
                        for name, value in row["metrics"].items():
                            if isinstance(value, (int, float)) and not isinstance(value, bool):
                                route_rows.append({**{k: v for k, v in row.items() if k != "metrics"}, "metric_name": str(name), "value": value})
                add(
                    schema=schema,
                    adapter="stage9.stage456_route_execution.v1",
                    stage=int(document.value.get("stage", 6)),
                    rows=route_rows,
                    condition="route",
                    metric="route_metric",
                )
                continue
        # checkpoint/authorization 等非数值输入已参与 source_refs 与 input hash 校验，
        # 但不会被强行解释成数据行。
    if not rows_artifacts:
        raise ValueError("STAGE789_INPUT_SET_HAS_NO_RECOGNIZED_ROWS")
    return tuple(rows_artifacts)


def _pruning_study_from_documents(
    documents: Sequence[LoadedInputArtifact],
) -> PruningStudySpec:
    studies: dict[str, PruningStudySpec] = {}
    for document in documents:
        for mapping in _find_mappings(document.value):
            if mapping.get("schema_version") == "pruning-study-spec-v1":
                study = _study_from_mapping(mapping)
                studies[pruning_study_hash(study)] = study
    if len(studies) != 1:
        raise ValueError("STAGE7_INPUT_MUST_CONTAIN_EXACTLY_ONE_PRUNING_STUDY")
    return next(iter(studies.values()))


def _ablation_matrix_from_documents(
    documents: Sequence[LoadedInputArtifact],
) -> AblationMatrix:
    matrices: dict[str, AblationMatrix] = {}
    for document in documents:
        for mapping in _find_mappings(document.value):
            if mapping.get("schema_version") == "ablation-matrix-v1":
                matrix = AblationMatrix.from_mapping(mapping)
                matrices[matrix.digest] = matrix
    if len(matrices) != 1:
        raise ValueError("STAGE8_INPUT_MUST_CONTAIN_EXACTLY_ONE_ABLATION_MATRIX")
    return next(iter(matrices.values()))


@dataclass(frozen=True, slots=True)
class AblationMatrixDeclaration:
    """可经严格 JSON 输入声明、随后由核心编译器冻结的 Stage 8 矩阵。"""

    matrix_id: str
    base_config: Mapping[str, object]
    factors: tuple[AblationFactor, ...]
    base_seed: int
    seed_namespace: str
    scope: str
    formal_eligible: bool
    artifact_hash: str
    schema_version: str = "ablation-matrix-declaration-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "ablation-matrix-declaration-v1":
            raise ValueError("STAGE8_DECLARATION_SCHEMA_MISMATCH")
        if (
            not isinstance(self.matrix_id, str)
            or not self.matrix_id
            or not isinstance(self.seed_namespace, str)
            or not self.seed_namespace
            or not self.factors
        ):
            raise ValueError("STAGE8_DECLARATION_ID_OR_FACTORS_EMPTY")
        if not isinstance(self.base_config, Mapping) or not self.base_config:
            raise TypeError("STAGE8_DECLARATION_BASE_CONFIG_NOT_OBJECT")
        if any(not isinstance(factor, AblationFactor) for factor in self.factors):
            raise TypeError("STAGE8_DECLARATION_FACTOR_TYPE_INVALID")
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int) or self.base_seed < 0:
            raise ValueError("STAGE8_DECLARATION_BASE_SEED_INVALID")
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("STAGE8_DECLARATION_SCOPE_INVALID")
        if type(self.formal_eligible) is not bool:
            raise TypeError("STAGE8_DECLARATION_FORMAL_ELIGIBILITY_NOT_BOOLEAN")
        if self.formal_eligible is not (self.scope == "formal"):
            raise ValueError("STAGE8_DECLARATION_FORMAL_ELIGIBILITY_MISMATCH")
        if canonical_json_hash(self._payload_without_hash()) != self.artifact_hash:
            raise ValueError("STAGE8_DECLARATION_HASH_MISMATCH")

    def _payload_without_hash(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "matrix_id": self.matrix_id,
            "base_config": thaw_json_value(self.base_config),
            "factors": [
                {
                    "name": factor.name,
                    "config_path": list(factor.config_path),
                    "baseline_value": thaw_json_value(factor.baseline_value),
                    "alternatives": [thaw_json_value(item) for item in factor.alternatives],
                }
                for factor in self.factors
            ],
            "base_seed": self.base_seed,
            "seed_namespace": self.seed_namespace,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
        }

    def to_dict(self) -> dict[str, object]:
        return self._payload_without_hash() | {"artifact_hash": self.artifact_hash}

    def compile(self) -> AblationMatrix:
        return AblationMatrix.compile(
            matrix_id=self.matrix_id,
            base_config=self.base_config,
            factors=self.factors,
            base_seed=self.base_seed,
            seed_namespace=self.seed_namespace,
        )

    @classmethod
    def create(
        cls,
        *,
        matrix_id: str,
        base_config: Mapping[str, object],
        factors: Sequence[AblationFactor],
        base_seed: int,
        seed_namespace: str,
        scope: str,
    ) -> "AblationMatrixDeclaration":
        """从无 hash 声明生成 canonical manifest，供 CLI/builder 使用。"""

        values = {
            "schema_version": "ablation-matrix-declaration-v1",
            "matrix_id": matrix_id,
            "base_config": thaw_json_value(base_config),
            "factors": [
                {
                    "name": factor.name,
                    "config_path": list(factor.config_path),
                    "baseline_value": thaw_json_value(factor.baseline_value),
                    "alternatives": [thaw_json_value(item) for item in factor.alternatives],
                }
                for factor in factors
            ],
            "base_seed": base_seed,
            "seed_namespace": seed_namespace,
            "scope": scope,
            "formal_eligible": scope == "formal",
        }
        return cls(
            matrix_id=matrix_id,
            base_config=base_config,
            factors=tuple(factors),
            base_seed=base_seed,
            seed_namespace=seed_namespace,
            scope=scope,
            formal_eligible=scope == "formal",
            artifact_hash=canonical_json_hash(values),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "AblationMatrixDeclaration":
        expected = {
            "schema_version",
            "matrix_id",
            "base_config",
            "factors",
            "base_seed",
            "seed_namespace",
            "scope",
            "formal_eligible",
            "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("STAGE8_DECLARATION_FIELDS_MISMATCH")
        base_config = value.get("base_config")
        raw_factors = value.get("factors")
        if not isinstance(base_config, Mapping):
            raise TypeError("STAGE8_DECLARATION_BASE_CONFIG_NOT_OBJECT")
        if not isinstance(raw_factors, list) or not raw_factors:
            raise TypeError("STAGE8_DECLARATION_FACTORS_NOT_ARRAY")
        factors: list[AblationFactor] = []
        factor_fields = {"name", "config_path", "baseline_value", "alternatives"}
        for item in raw_factors:
            if not isinstance(item, Mapping) or set(item) != factor_fields:
                raise ValueError("STAGE8_DECLARATION_FACTOR_FIELDS_MISMATCH")
            path = item.get("config_path")
            alternatives = item.get("alternatives")
            if (
                not isinstance(path, list)
                or not path
                or not all(isinstance(part, str) and part for part in path)
                or not isinstance(alternatives, list)
                or not alternatives
            ):
                raise TypeError("STAGE8_DECLARATION_FACTOR_ARRAY_INVALID")
            factors.append(
                AblationFactor(
                    name=item["name"],  # type: ignore[arg-type]
                    config_path=tuple(path),
                    baseline_value=item["baseline_value"],
                    alternatives=tuple(alternatives),
                )
            )
        return cls(
            matrix_id=value["matrix_id"],  # type: ignore[arg-type]
            base_config=base_config,
            factors=tuple(factors),
            base_seed=value["base_seed"],  # type: ignore[arg-type]
            seed_namespace=value["seed_namespace"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            artifact_hash=value["artifact_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


def _ablation_declaration_from_documents(
    documents: Sequence[LoadedInputArtifact],
) -> AblationMatrixDeclaration | None:
    declarations: dict[str, AblationMatrixDeclaration] = {}
    for document in documents:
        for mapping in _find_mappings(document.value):
            if mapping.get("schema_version") == "ablation-matrix-declaration-v1":
                declaration = AblationMatrixDeclaration.from_mapping(mapping)
                declarations[declaration.artifact_hash] = declaration
    if len(declarations) > 1:
        raise ValueError("STAGE8_INPUT_CONTAINS_MULTIPLE_MATRIX_DECLARATIONS")
    return None if not declarations else next(iter(declarations.values()))


def _build_source_table(
    request: TaskExecutionRequest,
    documents: Sequence[LoadedInputArtifact],
) -> BoundSourceTable:
    if documents:
        rows_artifacts = _rows_from_documents(documents, scope=request.config.run_intent)
    else:
        if request.config.run_intent == "formal":
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "formal_source_table",
                    "formal analysis 缺少 hash-bound frozen source",
                    True,
                )
            )
        rows_artifacts = (_tiny_stage_rows(),)
    builder = CrossStageSourceBuilder(
        scope=request.config.run_intent,
        formal_authorization_hash=(
            request.environment.environment_hash
            if request.config.run_intent == "formal"
            else None
        ),
    )
    for artifact in rows_artifacts:
        builder.add(artifact)
    source = builder.build(
        name="stage9_frozen_source",
        schema_version="stage9-source-table-v1",
    )
    authorization_hash = (
        request.environment.environment_hash
        if request.config.run_intent == "formal"
        else None
    )
    validate_analysis_lineage(
        documents,
        rebuild_source=lambda ancestors: _rebuild_lineage_source(
            ancestors,
            scope=request.config.run_intent,
            formal_authorization_hash=authorization_hash,
        ),
        find_mappings=_find_mappings,
    )
    return source


def _rebuild_lineage_source(
    documents: Sequence[LoadedInputArtifact],
    *,
    scope: str,
    formal_authorization_hash: str | None,
) -> BoundSourceTable:
    """只从一组祖先 task artifact 重建报告生成时使用的冻结源。

    这里故意不读取报告或推荐对象中的标量结果。报告只是 claim，祖先 task-output
    中的逐行证据才是 source。因而即使有人手工改写 ``best_observed_mean`` 或报告
    Markdown，也无法让该数字进入 Stage 9 ETL。
    """

    rows = _rows_from_documents(documents, scope=scope)
    builder = CrossStageSourceBuilder(
        scope=scope,
        formal_authorization_hash=formal_authorization_hash,
    )
    for artifact in rows:
        builder.add(artifact)
    return builder.build(
        name="stage9_frozen_source",
        schema_version="stage9-source-table-v1",
    )


def _numeric_column(source: BoundSourceTable) -> str:
    for preferred in ("damage", "damage_auc", "directed_effect", "value", "mean"):
        if preferred in source.table.columns and all(
            isinstance(row[preferred], (int, float)) and not isinstance(row[preferred], bool)
            for row in source.table.rows
        ):
            return preferred
    for column in source.table.columns:
        if all(
            isinstance(row[column], (int, float)) and not isinstance(row[column], bool)
            for row in source.table.rows
        ):
            return column
    raise ValueError("STAGE9_SOURCE_HAS_NO_NUMERIC_COLUMN")


def _statistics_table(
    source: BoundSourceTable,
    *,
    confidence: float = 0.95,
) -> BoundSourceTable:
    value_column = _numeric_column(source)
    group_columns = tuple(
        column
        for column in ("metric_name", "condition")
        if column in source.table.columns and column != value_column
    )
    return grouped_statistics(
        source,
        group_columns=group_columns,
        value_column=value_column,
        confidence=confidence,
        output_name="stage9_statistics",
    )


def _paired_statistics_table(
    source: BoundSourceTable,
    *,
    confidence: float,
) -> BoundSourceTable:
    value_column = _numeric_column(source)
    required = {"condition", "pair_id", value_column}
    if not required.issubset(source.table.columns):
        raise ValueError("STAGE9_PAIRED_SOURCE_COLUMNS_MISSING")
    conditions = {row["condition"] for row in source.table.rows}
    if "baseline" not in conditions:
        raise ValueError("STAGE9_PAIRED_BASELINE_CONDITION_MISSING")
    groups = ("metric_name",) if "metric_name" in source.table.columns else ()
    return paired_statistics_with_holm(
        source,
        group_columns=groups,
        pair_id_column="pair_id",
        condition_column="condition",
        value_column=value_column,
        baseline_condition="baseline",
        confidence=confidence,
        familywise_alpha=1.0 - confidence,
        output_name="stage9_paired_statistics_holm",
    )


def _damage_auc_source_table(source: BoundSourceTable) -> BoundSourceTable:
    required = {"method", "direction", "pruning_scope", "metric", "ratio", "damage"}
    if not required.issubset(source.table.columns):
        raise ValueError("STAGE7_DAMAGE_AUC_SOURCE_COLUMNS_MISSING")
    grouped: dict[tuple[str, str, str, str], dict[float, list[float]]] = {}
    for row in source.table.rows:
        key = (
            str(row["method"]),
            str(row["direction"]),
            str(row["pruning_scope"]),
            str(row["metric"]),
        )
        ratio = float(row["ratio"])
        damage = float(row["damage"])
        grouped.setdefault(key, {}).setdefault(ratio, []).append(damage)
    rows: list[Mapping[str, object]] = []
    for key in sorted(grouped):
        ratios = sorted(grouped[key])
        damages = [sum(grouped[key][ratio]) / len(grouped[key][ratio]) for ratio in ratios]
        result = damage_auc(ratios, damages)
        rows.append(
            {
                "method": key[0],
                "direction": key[1],
                "pruning_scope": key[2],
                "metric": key[3],
                "defined": result.defined,
                "damage_auc": result.value,
                "reason": result.reason,
                "ratio_count": len(ratios),
            }
        )
    table = FrozenSourceTable.from_rows(
        name="stage7_damage_auc",
        schema_version="stage7-damage-auc-table-v1",
        rows=rows,
    )
    return BoundSourceTable.create(
        table,
        role="derived_damage_auc",
        scope=source.scope,
        parent_artifact_hashes=(source.artifact_hash,),
        derivation_id="stage7.damage_auc.v1",
    )


def _robustness_source_table(
    source: BoundSourceTable,
    statistics: BoundSourceTable,
) -> BoundSourceTable:
    rows = [
        {
            **dict(row),
            "quality_status": "DEFINED" if row.get("ci_defined") is True else "UNDEFINED",
        }
        for row in statistics.table.rows
    ]
    table = FrozenSourceTable.from_rows(
        name="stage8_robustness",
        schema_version="stage8-robustness-table-v1",
        rows=rows,
    )
    return BoundSourceTable.create(
        table,
        role="derived_robustness",
        scope=source.scope,
        parent_artifact_hashes=(source.artifact_hash, statistics.artifact_hash),
        derivation_id="stage8.robustness_summary.v1",
    )


def _build_report(source: BoundSourceTable, *, report_id: str) -> AnalysisReport:
    column = _numeric_column(source)
    values = [float(row[column]) for row in source.table.rows]
    result = MetricResult(True, sum(values) / len(values), metadata={"n": len(values)})
    builder = AnalysisReportBuilder(report_id=report_id)
    builder.add_source(source.table)
    builder.add_metric(
        f"mean_{column}",
        result,
        source=source.table,
        derivation_id="stage9.arithmetic_mean.v1",
        input_columns=(column,),
    )
    return builder.build(metadata={"source_artifact_hash": source.artifact_hash})


def _build_chart(source: BoundSourceTable, *, chart_id: str) -> ChartArtifact:
    y_column = _numeric_column(source)
    candidates = [
        column
        for column in ("ratio", "condition", "source_row_index", "replicate")
        if column in source.table.columns and column != y_column
    ]
    if not candidates:
        candidates = [column for column in source.table.columns if column != y_column]
    if not candidates:
        raise ValueError("STAGE9_CHART_REQUIRES_DISTINCT_X_COLUMN")
    x_column = candidates[0]
    spec = ChartSpec.from_table(
        source.table,
        chart_id=chart_id,
        chart_type="line" if x_column in {"ratio", "source_row_index", "replicate"} else "bar",
        x_column=x_column,
        y_columns=(y_column,),
        sort_columns=(x_column,),
    )
    return ChartArtifact.from_spec(spec)


def _composite_figure_source(
    source: BoundSourceTable,
) -> tuple[BoundSourceTable, str, str, str, str, bool]:
    """为严格矩形的组合图准备可追溯输入表。

    ``render_composite_figure_set`` 有意要求 ``condition × pair`` 是完整矩形；
    这个约束对配对实验很重要，但 Stage 9 的跨阶段 ETL 还会收到剪枝与消融等
    非配对、行数不同的结果。此时不能补零，也不能为了画图截断某个实验分支。

    若原表已经满足矩形契约，本函数原样返回。否则，它创建一个只用于可视化的
    行视图：所有观测共享一个明确的 ``all_source_rows`` 条件，每条原始行获得唯一
    的顺序 pair；数值来自原表中冻结的数值列。派生表的每一行都嵌入完整原始行及
    其 hash，因此既不丢行也不伪造数值；派生表又以原 ``BoundSourceTable`` hash 为
    唯一 parent，从而保留 ETL lineage。

    返回值依次为派生/原始表、value/condition/pair/x 列名，以及是否执行了矩形化。
    """

    value_column = _numeric_column(source)
    condition_column = "condition"
    pair_column = "replicate"
    required = {value_column, condition_column, pair_column}
    if required.issubset(source.table.columns):
        conditions: set[str] = set()
        pairs: set[str] = set()
        cells: set[tuple[str, str]] = set()
        duplicated = False
        for row in source.table.rows:
            condition_key = canonical_json_hash(
                {"value": thaw_json_value(row[condition_column])}
            )
            pair_key = canonical_json_hash(
                {"value": thaw_json_value(row[pair_column])}
            )
            key = (condition_key, pair_key)
            if key in cells:
                duplicated = True
                break
            conditions.add(condition_key)
            pairs.add(pair_key)
            cells.add(key)
        if not duplicated and len(cells) == len(conditions) * len(pairs):
            x_column = "ratio" if "ratio" in source.table.columns else pair_column
            return (
                source,
                value_column,
                condition_column,
                pair_column,
                x_column,
                False,
            )

    derived_rows: list[Mapping[str, object]] = []
    for row_index, frozen_row in enumerate(source.table.rows):
        source_row = thaw_json_value(frozen_row)
        if not isinstance(source_row, dict):  # pragma: no cover - FrozenSourceTable 保证
            raise TypeError("STAGE9_FIGURE_SOURCE_ROW_NOT_OBJECT")
        derived_rows.append(
            {
                "figure_condition": "all_source_rows",
                "figure_pair": row_index,
                "figure_value": float(frozen_row[value_column]),
                "figure_x": row_index,
                "source_row": source_row,
                "source_row_hash": canonical_json_hash(source_row),
            }
        )
    table = FrozenSourceTable.from_rows(
        name="stage9_composite_figure_rows",
        schema_version="stage9-composite-figure-source-v1",
        rows=derived_rows,
    )
    derived = BoundSourceTable.create(
        table,
        role="derived_rectangular_figure_source",
        scope=source.scope,
        parent_artifact_hashes=(source.artifact_hash,),
        derivation_id=f"stage9.rectangular_row_view.{value_column}.v1",
    )
    return (
        derived,
        "figure_value",
        "figure_condition",
        "figure_pair",
        "figure_x",
        True,
    )


def _build_table(source: BoundSourceTable) -> tuple[TableSpec, TableArtifact]:
    # 限制 fixture payload 尺寸；列选择仍完全由 source 的 canonical 列集决定。
    spec = TableSpec(
        table_artifact_hash=source.artifact_hash,
        columns=source.table.columns[:12],
        formats=("csv", "markdown"),
        caption="Stage 9 frozen-source reconstruction",
    )
    return spec, render_table(source, spec)


def _build_bundle(
    request: TaskExecutionRequest,
    source: BoundSourceTable,
) -> tuple[AnalysisBundle, AnalysisReport, ChartArtifact, TableArtifact]:
    report = _build_report(source, report_id=f"stage9-{request.config.config_hash[:16]}")
    chart = _build_chart(source, chart_id=f"stage9-chart-{request.config.config_hash[:12]}")
    _, table = _build_table(source)
    builder = AnalysisBundleBuilder(
        bundle_id=f"stage9-bundle-{request.config.config_hash[:12]}",
        scope=request.config.run_intent,
        formal_authorization_hash=(
            request.environment.environment_hash
            if request.config.run_intent == "formal"
            else None
        ),
    )
    builder.add_table(source)
    builder.add_report(report)
    builder.add_chart(chart)
    builder.add_rendered_table(table)
    return builder.build(), report, chart, table


@dataclass(slots=True)
class Stage789CompositeTaskRunner:
    """按 task ID 分派 Stage 7--9，并为共享 runner kind 保留 fallback。"""

    runner_kind: RunnerKind
    workspace_root: Path
    fallback: TaskRunner | None = None
    pruning_resource_loader: PruningResourceLoader | None = None
    ablation_resource_loader: AblationResourceLoader | None = None

    def __post_init__(self) -> None:
        if self.runner_kind not in _HANDLED_BY_KIND:
            raise ValueError(f"STAGE789_RUNNER_KIND_UNSUPPORTED:{self.runner_kind.value}")
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.fallback is not None and self.fallback.runner_kind is not self.runner_kind:
            raise ValueError("STAGE789_FALLBACK_KIND_MISMATCH")

    @property
    def handled_task_ids(self) -> frozenset[str]:
        return _HANDLED_BY_KIND[self.runner_kind]

    def _store(self, request: TaskExecutionRequest) -> TaskArtifactStore:
        artifacts = request.config.section("artifacts")
        if not isinstance(artifacts, dict):  # pragma: no cover
            raise TypeError("STAGE789_ARTIFACTS_NOT_OBJECT")
        return TaskArtifactStore(self.workspace_root, str(artifacts["output_dir"]))

    def _completed(self, request: TaskExecutionRequest) -> TaskRunResult | None:
        refs = self._store(request).discover_complete(
            task_id=request.task.task_id,
            config_hash=request.config.config_hash,
            artifact_kinds=request.task.artifact_kinds,
            formal_eligible=request.config.run_intent == "formal",
        )
        if refs is None:
            return None
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            message="stage7-9 task restored from authoritative commits",
            metadata={"restored": True},
        )

    def _authorize_partial_resume(self, request: TaskExecutionRequest) -> None:
        """把 ``task resume`` 引用绑定到 Stage 7/8 的权威 cell commits。

        完整任务 commit 由 :meth:`_completed` 幂等读取；这里仅处理尚未聚合完成的
        pruning/ablation 单元。普通 ``task run`` 不得根据输出目录残留静默续跑，
        而 ``resume_ref`` 必须位于当前任务输出根内并覆盖全部已发现 commit。
        """

        if request.task.recovery_mode is not RecoveryMode.RESUME_SHARDS:
            return
        store = self._store(request)
        commit_paths = tuple(
            sorted(
                (
                    *(
                        store.root / "core" / "pruning" / "commits"
                    ).glob("*.json"),
                    *(
                        store.root / "core" / "ablation" / "commits"
                    ).glob("*.json"),
                    *(
                        store.root / "core" / "ablation-training"
                    ).glob("**/commits/*.json"),
                ),
                key=lambda path: path.as_posix(),
            )
        )
        recovery = request.config.section("recovery")
        assert isinstance(recovery, dict)
        raw_ref = recovery["resume_ref"]
        if not commit_paths:
            if raw_ref is not None:
                raise FileNotFoundError(
                    "STAGE789_RESUME_REF_HAS_NO_AUTHORITATIVE_CELL_COMMITS"
                )
            return
        if raw_ref is None:
            refs = ",".join(
                path.relative_to(self.workspace_root).as_posix()
                for path in commit_paths
            )
            raise RuntimeError(f"STAGE789_RESUME_REF_REQUIRED:{refs}")
        resume_path = _safe_workspace_path(
            self.workspace_root,
            str(raw_ref),
            field="recovery.resume_ref",
        ).resolve()
        try:
            resume_path.relative_to(store.root.resolve())
        except ValueError as error:
            raise ValueError("STAGE789_RESUME_REF_OUTSIDE_TASK_OUTPUT") from error
        if not resume_path.exists():
            raise FileNotFoundError("STAGE789_RESUME_REF_NOT_FOUND")
        if not all(
            path.resolve() == resume_path
            or path.resolve().is_relative_to(resume_path)
            for path in commit_paths
        ):
            raise ValueError("STAGE789_RESUME_REF_DOES_NOT_COVER_ALL_CELL_COMMITS")

    def _publish(
        self,
        request: TaskExecutionRequest,
        payloads: Mapping[str, Mapping[str, object]],
        *,
        source_refs: tuple[str, ...],
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> TaskRunResult:
        if tuple(payloads) != request.task.artifact_kinds:
            raise ValueError("STAGE789_OUTPUT_KINDS_OR_ORDER_MISMATCH")
        store = self._store(request)
        refs: dict[str, str] = {}
        for kind, payload in payloads.items():
            published = store.publish(
                task_id=request.task.task_id,
                artifact_kind=kind,
                config_hash=request.config.config_hash,
                run_intent=request.config.run_intent,
                payload=_versioned_task_payload(kind, payload),
                formal_eligible=request.config.run_intent == "formal",
                source_refs=source_refs,
            )
            refs[kind] = published.commit_ref
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            message="stage7-9 core task completed",
            metadata={} if metadata is None else metadata,
        )

    def _pruning_resources(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> PruningTaskResources:
        if request.config.run_intent == "formal":
            _require_formal_inputs(request, inputs, requirement="checkpoint_and_importance")
            resources = (
                self.pruning_resource_loader(request, inputs)
                if self.pruning_resource_loader is not None
                else _default_pruning_resources(
                    request,
                    inputs,
                    workspace_root=self.workspace_root,
                    output_root=self._store(request).root,
                )
            )
        elif self.pruning_resource_loader is not None and inputs:
            resources = self.pruning_resource_loader(request, inputs)
        elif inputs and any(
            isinstance(document.value.get("checkpoint_commits"), list)
            or document.value.get("schema_version") == "stage456-route-execution-v1"
            for document in inputs
        ):
            resources = _default_pruning_resources(
                request,
                inputs,
                workspace_root=self.workspace_root,
                output_root=self._store(request).root,
            )
        elif inputs:
            resources = _tiny_pruning_resources(request)
            referenced_study = _pruning_study_from_documents(inputs)
            if pruning_study_hash(referenced_study) != pruning_study_hash(resources.study):
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.CAPABILITY_UNAVAILABLE,
                        "local_pruning_resource_loader",
                        "输入 pruning study 不是内置 tiny 合同，必须注入资源 loader",
                        False,
                        tuple(item.ref for item in inputs),
                    )
                )
        else:
            resources = _tiny_pruning_resources(request)
        _validate_pruning_resources(request, resources, inputs)
        return resources

    def _ablation_resources(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> AblationTaskResources:
        has_evidence_manifest = any(
            mapping.get("schema_version")
            == "ablation-cell-evidence-manifest-v1"
            for document in inputs
            for mapping in _find_mappings(document.value)
        )
        if request.task.task_id == "stage8.execute" and has_evidence_manifest:
            # 兼容已经完成外部 cell 训练的权威 evidence manifest：本任务只做严格
            # 复核与 deterministic reduce，不得重新启动 Tiny/Configured 训练。
            _require_formal_inputs(
                request,
                inputs,
                requirement="checkpoint_and_ablation_matrix",
            )
            resources = _default_ablation_resources(request, inputs)
            _validate_ablation_resources(request, resources, inputs)
            return resources
        if request.task.task_id == "stage8.execute":
            matrix = _ablation_matrix_from_documents(inputs)
            matrix_documents = [
                document
                for document in inputs
                if any(
                    mapping.get("schema_version") == "ablation-matrix-v1"
                    and AblationMatrix.from_mapping(mapping).digest == matrix.digest
                    for mapping in _find_mappings(document.value)
                )
            ]
            if len(matrix_documents) != 1:
                raise ValueError("STAGE8_MATRIX_DOCUMENT_NOT_UNIQUE")
            if request.config.run_intent == "formal":
                _require_formal_inputs(
                    request,
                    inputs,
                    requirement="checkpoint_and_ablation_matrix",
                )
                if self.ablation_resource_loader is not None:
                    resources = self.ablation_resource_loader(request, inputs)
                else:
                    checkpoint_documents = [
                        document
                        for document in inputs
                        if document is not matrix_documents[0]
                        and document.value.get("schema_version")
                        in {
                            "stage456-route-execution-v1",
                            "stage01-training-task-output-v1",
                        }
                    ]
                    if len(checkpoint_documents) != 1:
                        raise TaskBlockedError(
                            TaskBlocker(
                                BlockerCode.ASSET_UNAVAILABLE,
                                "hash_bound_ablation_checkpoint",
                                "formal Stage 8 execute 要求唯一训练路线/checkpoint 祖先",
                                True,
                                tuple(item.ref for item in inputs),
                            )
                        )
                    checkpoint = checkpoint_documents[0]
                    resources = AblationTaskResources(
                        matrix=matrix,
                        executor=_TinyAblationExecutor(),
                        config_validator=_default_config_validator,
                        formal_authorization_hash=request.environment.environment_hash,
                        checkpoint_input_hash=checkpoint.artifact_hash,
                        matrix_input_hash=matrix_documents[0].artifact_hash,
                        training_builder=ConfiguredAblationTrainingBuilder(
                            self.workspace_root,
                            environment=request.environment,
                        ),
                        source_checkpoint_artifact_hash=checkpoint.artifact_hash,
                    )
            else:
                resources = AblationTaskResources(
                    matrix=matrix,
                    executor=_TinyAblationExecutor(),
                    config_validator=lambda value: dict(value),
                    matrix_input_hash=matrix_documents[0].artifact_hash,
                    training_builder=TinyAblationTrainingBuilder(),
                    source_checkpoint_artifact_hash=canonical_json_hash(
                        {
                            "schema_version": "tiny-ablation-initialization-v1",
                            "matrix_hash": matrix.digest,
                        }
                    ),
                )
            _validate_ablation_resources(request, resources, inputs)
            return resources
        if request.config.run_intent == "formal":
            _require_formal_inputs(request, inputs, requirement="checkpoint_and_ablation_matrix")
            resources = (
                self.ablation_resource_loader(request, inputs)
                if self.ablation_resource_loader is not None
                else _default_ablation_resources(request, inputs)
            )
        elif self.ablation_resource_loader is not None and inputs:
            resources = self.ablation_resource_loader(request, inputs)
        elif inputs and any(
            mapping.get("schema_version") == "ablation-cell-evidence-manifest-v1"
            for document in inputs
            for mapping in _find_mappings(document.value)
        ):
            resources = _default_ablation_resources(request, inputs)
        elif inputs:
            matrix = _ablation_matrix_from_documents(inputs)
            resources = AblationTaskResources(
                matrix=matrix,
                executor=_TinyAblationExecutor(),
                config_validator=lambda value: dict(value),
                training_builder=TinyAblationTrainingBuilder(),
                source_checkpoint_artifact_hash=canonical_json_hash(
                    {
                        "schema_version": "tiny-ablation-initialization-v1",
                        "matrix_hash": matrix.digest,
                    }
                ),
            )
        else:
            resources = _tiny_ablation_resources(request)
        _validate_ablation_resources(request, resources, inputs)
        return resources

    def _run_ablation_freeze(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> tuple[dict[str, Mapping[str, object]], Mapping[str, JSONValue]]:
        """编译或复核冻结矩阵，并显式发布单因素验证证据。"""

        _require_formal_inputs(request, inputs, requirement="ablation_matrix_declaration")
        declaration = _ablation_declaration_from_documents(inputs) if inputs else None
        matrices: dict[str, AblationMatrix] = {}
        for document in inputs:
            for mapping in _find_mappings(document.value):
                if mapping.get("schema_version") == "ablation-matrix-v1":
                    matrix = AblationMatrix.from_mapping(mapping)
                    matrices[matrix.digest] = matrix
        if len(matrices) > 1:
            raise ValueError("STAGE8_FREEZE_MULTIPLE_DISTINCT_MATRICES")
        matrix = None if not matrices else next(iter(matrices.values()))
        if declaration is not None:
            if declaration.scope != request.config.run_intent:
                raise ValueError("STAGE8_DECLARATION_REQUEST_SCOPE_MISMATCH")
            compiled = declaration.compile()
            if matrix is not None and matrix.digest != compiled.digest:
                raise ValueError("STAGE8_DECLARATION_COMPILED_MATRIX_MISMATCH")
            matrix = compiled
        if matrix is None:
            if inputs:
                raise ValueError("STAGE8_FREEZE_INPUT_HAS_NO_MATRIX_OR_DECLARATION")
            if request.config.run_intent == "formal":  # 已由 _require_formal_inputs 阻塞
                raise AssertionError("formal freeze unexpectedly reached tiny fallback")
            matrix = _tiny_ablation_resources(request).matrix

        validations = [
            {
                "cell_id": cell.cell_id,
                "parent_cell_id": cell.parent_cell_id,
                "changed_factor": cell.changed_factor,
                "changed_path": (
                    None if cell.changed_path is None else list(cell.changed_path)
                ),
                "config_hash": cell.config_hash,
                "seed": cell.seed,
                "expected_leaf_difference_count": (
                    0 if cell.cell_id == matrix.baseline_cell_id else 1
                ),
                "valid": True,
            }
            for cell in sorted(matrix.cells, key=lambda item: item.cell_id)
        ]
        formal = request.config.run_intent == "formal"
        source_hashes = [item.artifact_hash for item in inputs]
        return (
            {
                "ablation_matrix": matrix.to_dict(),
                "single_factor_validation": {
                    "schema_version": "ablation-single-factor-validation-v1",
                    "matrix_id": matrix.matrix_id,
                    "matrix_hash": matrix.digest,
                    "baseline_cell_id": matrix.baseline_cell_id,
                    "all_valid": all(bool(item["valid"]) for item in validations),
                    "cells": validations,
                    "validation_rule": "baseline_or_exactly_one_leaf_difference",
                },
                "matrix_freeze": {
                    "schema_version": "ablation-matrix-freeze-v1",
                    "matrix_id": matrix.matrix_id,
                    "matrix_hash": matrix.digest,
                    "frozen": matrix.frozen,
                    "scope": request.config.run_intent,
                    "formal_eligible": formal,
                    "formal_authorization_hash": (
                        request.environment.environment_hash if formal else None
                    ),
                    "estimator_decision_ref": (
                        request.environment.estimator_decision_ref if formal else None
                    ),
                    "source_artifact_hashes": source_hashes,
                    "declaration_artifact_hash": (
                        None if declaration is None else declaration.artifact_hash
                    ),
                },
            },
            {
                "matrix_hash": matrix.digest,
                "cell_count": len(matrix.cells),
                "compiled_from_declaration": declaration is not None,
            },
        )

    def _run_pruning(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> tuple[dict[str, Mapping[str, object]], Mapping[str, JSONValue]]:
        resources = self._pruning_resources(request, inputs)
        study_dict = _study_to_dict(resources.study)
        plans = [_run_to_dict(run) for run in resources.study.compile()]
        if request.task.task_id == "stage7.matrix":
            return (
                {
                    "pruning_matrix": study_dict,
                    "pruning_plans": {
                        "schema_version": "pruning-plans-v1",
                        "study_hash": pruning_study_hash(resources.study),
                        "plans": plans,
                    },
                    "mask_manifest": {
                        "schema_version": "pruning-mask-manifest-v1",
                        "study_hash": pruning_study_hash(resources.study),
                        "run_ids": [plan["run_id"] for plan in plans],
                        "tie_breaker": "canonical_coordinate_id",
                        "coordinate_registry_hash": resources.coordinate_registry_hash,
                    },
                },
                {"study_hash": pruning_study_hash(resources.study), "executed": False},
            )
        result_root = self._store(request).root / "core" / "pruning"
        runner = PruningStudyRunner(
            resources.study,
            parameters=resources.parameters,
            scores_by_artifact_hash=resources.scores_by_artifact_hash,
            evaluator=resources.evaluator,
            result_root=result_root,
            model_checkpoint_hash=resources.model_checkpoint_hash,
            coordinate_registry_hash=resources.coordinate_registry_hash,
            formal_authorization_hash=resources.formal_authorization_hash,
        )
        study_result = runner.run()
        result_values = runner.store.restore()
        rows = [
            {
                "method": value["method"],
                "direction": value["direction"],
                "pruning_scope": value["pruning_scope"],
                "ratio": value["ratio"],
                "metric": metric,
                "value": value["metrics"][metric],
                "damage": value["damage"][metric],
                "run_id": value["run_id"],
            }
            for value in result_values
            for metric in sorted(value["metrics"])
            if value["method"] != "baseline"
        ]
        row_artifact = StageArtifactRows.create(
            artifact_id=f"stage{request.task.stage}-pruning-{study_result.study_id}",
            source_schema_version=study_result.schema_version,
            source_artifact_hash=study_result.artifact_hash,
            stage=request.task.stage,
            adapter_id="stage789.pruning_results.v1",
            rows=rows,
            scope=study_result.scope,
            formal_eligible=study_result.formal_eligible,
        )
        if request.task.task_id == "stage7.evaluate":
            return (
                {
                    "pruning_evaluation_results": {
                        "study_result": study_result.to_dict(),
                        "results": result_values,
                        "stage_rows": row_artifact.to_dict(),
                    },
                    "damage_curves": {
                        "schema_version": "pruning-damage-curves-v1",
                        "study_hash": study_result.study_hash,
                        "rows": rows,
                        "source_artifact_hash": row_artifact.artifact_hash,
                    },
                },
                {"study_hash": study_result.study_hash, "executed": True},
            )
        # 兼容任务一次完成计划、执行、简化 AUC 与报告摘要。
        auc_rows: list[dict[str, object]] = []
        groups: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
        for row in rows:
            key = (
                str(row["method"]),
                str(row["direction"]),
                str(row["pruning_scope"]),
                str(row["metric"]),
            )
            groups.setdefault(key, []).append(row)
        for key in sorted(groups):
            group = sorted(groups[key], key=lambda row: float(row["ratio"]))
            auc = damage_auc(
                [float(row["ratio"]) for row in group],
                [float(row["damage"]) for row in group],
            )
            auc_rows.append(
                {
                    "method": key[0],
                    "direction": key[1],
                    "pruning_scope": key[2],
                    "metric": key[3],
                    "defined": auc.defined,
                    "value": auc.value,
                    "reason": auc.reason,
                }
            )
        if request.task.task_id == "stage4.pruning_validation":
            return (
                {
                    "pruning_plan": {"study": study_dict, "plans": plans},
                    "pruning_results": {
                        "study_result": study_result.to_dict(),
                        "results": result_values,
                        "stage_rows": row_artifact.to_dict(),
                    },
                    "damage_auc_table": {
                        "schema_version": "pruning-damage-auc-table-v1",
                        "study_hash": study_result.study_hash,
                        "rows": auc_rows,
                        "scope": request.config.run_intent,
                    },
                },
                {"study_hash": study_result.study_hash, "executed": True},
            )
        return (
            {
                "pruning_plan": {"study": study_dict, "plans": plans},
                "pruning_results": {
                    "study_result": study_result.to_dict(),
                    "results": result_values,
                    "stage_rows": row_artifact.to_dict(),
                },
                "damage_auc_table": {
                    "schema_version": "pruning-damage-auc-table-v1",
                    "study_hash": study_result.study_hash,
                    "rows": auc_rows,
                },
                "stage_report": {
                    "schema_version": "stage7-core-report-v1",
                    "scope": request.config.run_intent,
                    "study_hash": study_result.study_hash,
                    "run_count": len(result_values),
                    "claim_status": "fixture_only" if request.config.run_intent == "local_fixture" else "formal_evidence",
                },
            },
            {"study_hash": study_result.study_hash, "executed": True},
        )

    def _run_ablation(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> tuple[dict[str, Mapping[str, object]], Mapping[str, JSONValue]]:
        resources = self._ablation_resources(request, inputs)
        if request.task.task_id == "stage8.execute" and resources.training_builder is not None:
            source_hash = resources.source_checkpoint_artifact_hash
            if not isinstance(source_hash, str) or len(source_hash) != 64:
                raise ValueError("STAGE8_SOURCE_CHECKPOINT_HASH_INVALID")
            result_root = self._store(request).root / "core" / "ablation-training"
            try:
                prefix = result_root.resolve().relative_to(
                    self.workspace_root.resolve()
                ).as_posix()
            except ValueError as error:
                raise ValueError("STAGE8_RESULT_ROOT_ESCAPES_WORKSPACE") from error
            training_runner = TrainingAblationStudyRunner(
                resources.matrix,
                builder=resources.training_builder,
                result_root=result_root,
                artifact_ref_prefix=prefix,
                run_intent=request.config.run_intent,
                source_checkpoint_artifact_hash=source_hash,
                config_validator=resources.config_validator,
                formal_authorization_hash=resources.formal_authorization_hash,
            )
            output = training_runner.run()
            result = output.study_result
            return (
                {
                    "ablation_cell_results": result.to_dict(),
                    "cell_lineage_manifest": output.evidence_manifest.to_dict(),
                },
                {
                    "matrix_hash": result.matrix_hash,
                    "cell_count": len(result.cell_result_hashes),
                    "training_evidence_manifest_hash": (
                        output.evidence_manifest.artifact_hash
                    ),
                },
            )
        runner = AblationMatrixRunner(
            resources.matrix,
            executor=resources.executor,
            result_root=self._store(request).root / "core" / "ablation",
            run_intent=request.config.run_intent,
            config_validator=resources.config_validator,
            formal_authorization_hash=resources.formal_authorization_hash,
        )
        result = runner.run()
        if request.task.task_id == "stage8.execute":
            return (
                {
                    "ablation_cell_results": result.to_dict(),
                    "cell_lineage_manifest": {
                        "schema_version": "ablation-cell-lineage-v1",
                        "matrix_id": result.matrix_id,
                        "matrix_hash": result.matrix_hash,
                        "baseline_result_hash": result.baseline_result_hash,
                        "cell_result_hashes": list(result.cell_result_hashes),
                    },
                },
                {
                    "matrix_hash": result.matrix_hash,
                    "cell_count": len(result.cell_result_hashes),
                    "reduced_from_existing_evidence": True,
                },
            )
        effects = [float(row["directed_effect"]) for row in result.rows]
        return (
            {
                "ablation_matrix": resources.matrix.to_dict(),
                "ablation_results": result.to_dict(),
                "robustness_report": {
                    "schema_version": "ablation-robustness-report-v1",
                    "matrix_hash": result.matrix_hash,
                    "minimum_directed_effect": min(effects),
                    "maximum_directed_effect": max(effects),
                    "scope": result.scope,
                },
                "stage_report": {
                    "schema_version": "stage8-core-report-v1",
                    "matrix_hash": result.matrix_hash,
                    "cell_count": len(result.cell_result_hashes),
                    "claim_status": "fixture_only" if result.scope == "local_fixture" else "formal_evidence",
                },
            },
            {"matrix_hash": result.matrix_hash, "cell_count": len(result.cell_result_hashes)},
        )

    def _run_analysis_family(
        self,
        request: TaskExecutionRequest,
        inputs: tuple[LoadedInputArtifact, ...],
    ) -> tuple[dict[str, Mapping[str, object]], Mapping[str, JSONValue]]:
        _require_formal_inputs(request, inputs, requirement="frozen_source")
        source = _build_source_table(request, inputs)
        task_id = request.task.task_id
        if task_id == "stage9.ingest":
            return (
                {
                    "frozen_source_table": source.to_dict(),
                    "source_lineage_manifest": {
                        "schema_version": "stage9-source-lineage-v1",
                        "source_artifact_hash": source.artifact_hash,
                        "parent_artifact_hashes": list(source.parent_artifact_hashes),
                        "input_refs": [item.ref for item in inputs],
                    },
                    "ingest_report": {
                        "schema_version": "stage9-ingest-report-v1",
                        "row_count": len(source.table.rows),
                        "column_count": len(source.table.columns),
                        "scope": source.scope,
                    },
                },
                {"source_artifact_hash": source.artifact_hash},
            )
        if task_id in {"stage7.reduce", "stage8.reduce", "stage9.statistics"}:
            analysis_config = request.config.base_config.section("analysis")
            assert isinstance(analysis_config, dict)
            confidence = float(analysis_config["confidence_level"])
            statistics = _statistics_table(source, confidence=confidence)
            undefined = [
                dict(row)
                for row in statistics.table.rows
                if row.get("ci_defined") is False
            ]
            if task_id == "stage7.reduce":
                auc_table = _damage_auc_source_table(source)
                return (
                    {
                        "pruning_summary_table": statistics.to_dict(),
                        "damage_auc_table": auc_table.to_dict(),
                        "confidence_intervals": {
                            "schema_version": "confidence-intervals-v1",
                            "source_artifact_hash": statistics.artifact_hash,
                            "rows": [dict(row) for row in statistics.table.rows],
                        },
                    },
                    {"source_artifact_hash": source.artifact_hash},
                )
            if task_id == "stage8.reduce":
                robustness = _robustness_source_table(source, statistics)
                return (
                    {
                        "ablation_summary_table": statistics.to_dict(),
                        "robustness_table": robustness.to_dict(),
                        "quality_gates": {
                            "schema_version": "stage8-quality-gates-v1",
                            "gate_status": "NOT_RUN",
                            "local_validation_status": "PASS" if not undefined else "FAIL",
                            "undefined_row_count": len(undefined),
                            "source_artifact_hash": robustness.artifact_hash,
                        },
                    },
                    {"source_artifact_hash": source.artifact_hash},
                )
            names = request.task.artifact_kinds
            try:
                paired = _paired_statistics_table(source, confidence=confidence)
                paired_payload: Mapping[str, object] = paired.to_dict()
                paired_undefined = [
                    dict(row)
                    for row in paired.table.rows
                    if row.get("comparison_defined") is False
                ]
            except ValueError as error:
                # 不满足配对设计时必须显式 undefined，不能退化成非配对检验。
                paired = None
                paired_payload = {
                    "schema_version": "paired-statistics-unavailable-v1",
                    "defined": False,
                    "reason": str(error),
                    "source_artifact_hash": source.artifact_hash,
                }
                paired_undefined = [dict(paired_payload)]
            return (
                {
                    names[0]: {
                        "schema_version": "stage9-statistics-result-v1",
                        "grouped_statistics": statistics.to_dict(),
                        "paired_statistics_holm": paired_payload,
                        "source_artifact_hash": source.artifact_hash,
                    },
                    names[1]: {
                        "schema_version": "confidence-intervals-v1",
                        "source_artifact_hash": statistics.artifact_hash,
                        "rows": [dict(row) for row in statistics.table.rows],
                        "paired_rows": (
                            [] if paired is None else [dict(row) for row in paired.table.rows]
                        ),
                    },
                    names[2]: {
                        "schema_version": "undefined-metric-report-v1",
                        "source_artifact_hash": statistics.artifact_hash,
                        "undefined_rows": [*undefined, *paired_undefined],
                        "all_defined": not undefined and not paired_undefined,
                    },
                },
                {
                    "source_artifact_hash": source.artifact_hash,
                    "paired_statistics_executed": paired is not None,
                    "multiplicity_method": "holm",
                },
            )
        if task_id == "stage8.recommend":
            stats = _statistics_table(source)
            means = [float(row["mean"]) for row in stats.table.rows]
            return (
                {
                    "configuration_recommendation": {
                        "schema_version": "configuration-recommendation-v1",
                        "source_artifact_hash": stats.artifact_hash,
                        "status": "fixture_only" if source.scope == "local_fixture" else "evidence_bound",
                        "best_observed_mean": max(means),
                        "statistics_table": stats.to_dict(),
                    },
                    "applicability_report": {
                        "schema_version": "applicability-report-v1",
                        "source_artifact_hash": source.artifact_hash,
                        "row_count": len(source.table.rows),
                    },
                    "limitation_table": {
                        "schema_version": "limitation-table-v1",
                        "source_artifact_hash": source.artifact_hash,
                        "rows": [{"code": "observed_inputs_only", "active": True}],
                    },
                },
                {"source_artifact_hash": source.artifact_hash},
            )
        if task_id == "stage9.tables":
            spec, table = _build_table(source)
            return (
                {
                    "table_specs": {
                        "schema_version": "stage9-table-spec-publication-v1",
                        "spec": spec.to_dict(),
                        "producer_source_hash": analysis_producer_source_hash(),
                    },
                    "table_artifacts": {
                        "schema_version": "stage9-table-artifact-publication-v1",
                        "artifact": table.to_dict(),
                        "producer_source_hash": analysis_producer_source_hash(),
                    },
                },
                {"source_artifact_hash": source.artifact_hash},
            )
        if task_id == "stage9.charts":
            chart = _build_chart(source, chart_id=f"stage9-chart-{request.config.config_hash[:12]}")
            render_root = self._store(request).root / "rendered" / "stage9-figures"
            (
                figure_source,
                figure_value_column,
                figure_condition_column,
                figure_pair_column,
                figure_x_column,
                rectangularization_applied,
            ) = _composite_figure_source(source)
            composite = render_composite_figure_set(
                figure_source,
                render_root,
                value_column=figure_value_column,
                condition_column=figure_condition_column,
                pair_column=figure_pair_column,
                x_column=figure_x_column,
            )
            file_refs = [
                (render_root / f"{kind}.png").relative_to(self.workspace_root).as_posix()
                for kind in ("heatmap", "errorbar", "facet")
            ]
            return (
                {
                    "chart_specs": {
                        "schema_version": "stage9-chart-spec-publication-v1",
                        "canonical_chart_spec": chart.spec.to_dict(),
                        "composite_spec_hash": composite.spec_hash,
                        "producer_source_hash": composite.producer_source_hash,
                        "original_frozen_source": source.to_dict(),
                        "visualization_source": figure_source.to_dict(),
                        "rectangularization_applied": rectangularization_applied,
                    },
                    "chart_artifacts": {
                        "schema_version": "stage9-chart-artifact-publication-v1",
                        "canonical_chart_artifact": chart.to_dict(),
                        "composite_figure_artifact": composite.to_dict(),
                        "original_source_artifact_hash": source.artifact_hash,
                        "visualization_source_artifact_hash": (
                            figure_source.artifact_hash
                        ),
                        "file_refs": file_refs,
                    },
                },
                {
                    "source_artifact_hash": source.artifact_hash,
                    "composite_figure_hash": composite.artifact_hash,
                },
            )
        if task_id in {"stage7.report", "stage8.report", "stage9.report"}:
            report = _build_report(source, report_id=f"{task_id}-{request.config.config_hash[:12]}")
            chart = _build_chart(source, chart_id=f"{task_id}-chart")
            if task_id == "stage9.report":
                return (
                    {
                        "analysis_report": report.to_dict(),
                        "claim_evidence_index": {
                            "schema_version": "claim-evidence-index-v1",
                            "report_hash": report.report_hash,
                            "source_hashes": [item.content_hash for item in report.source_artifacts],
                        },
                    },
                    {"report_hash": report.report_hash},
                )
            return (
                {
                    "stage_report": report.to_dict(),
                    "chart_artifacts": chart.to_dict(),
                    "gate_summary": {
                        "schema_version": "stage-report-gate-summary-v1",
                        "gate_status": "NOT_RUN",
                        "local_validation_status": "PASS",
                        "report_hash": report.report_hash,
                    },
                },
                {"report_hash": report.report_hash},
            )
        bundle, report, chart, _ = _build_bundle(request, source)
        if task_id == "stage9.analysis_visualization_reporting":
            return (
                {
                    "frozen_source_table": source.to_dict(),
                    "analysis_report": report.to_dict(),
                    "chart_artifacts": chart.to_dict(),
                    "reproduction_manifest": bundle.to_dict(),
                },
                {"bundle_hash": bundle.artifact_hash},
            )
        if task_id == "stage9.bundle":
            bundle_path = bundle.publish(self._store(request).root / "analysis-bundle")
            relative = bundle_path.resolve().relative_to(self.workspace_root).as_posix()
            inventory = {
                "tables": len(bundle.tables),
                "reports": len(bundle.reports),
                "charts": len(bundle.chart_artifacts),
                "rendered_tables": len(bundle.table_artifacts),
            }
            return (
                {
                    "reproduction_manifest": bundle.to_dict(),
                    "delivery_manifest": {
                        "schema_version": "stage9-delivery-manifest-v1",
                        "bundle_hash": bundle.artifact_hash,
                        "bundle_ref": relative,
                        "scope": bundle.scope,
                    },
                    "artifact_inventory": {
                        "schema_version": "stage9-artifact-inventory-v1",
                        "bundle_hash": bundle.artifact_hash,
                        **inventory,
                    },
                },
                {"bundle_hash": bundle.artifact_hash},
            )
        if task_id == "stage9.replay":
            replay_bundle, _, _, _ = _build_bundle(request, source)
            replay_root = self._store(request).root / "deterministic-replay"
            first_path = bundle.publish(replay_root / "first-clean-build")
            second_path = replay_bundle.publish(replay_root / "second-clean-build")
            first_wire = load_canonical_json(first_path)
            second_wire = load_canonical_json(second_path)
            if not isinstance(first_wire, Mapping) or not isinstance(second_wire, Mapping):
                raise ValueError("STAGE9_REPLAY_BUNDLE_ROOT_NOT_OBJECT")
            first_reloaded = AnalysisBundle.from_mapping(first_wire)
            second_reloaded = AnalysisBundle.from_mapping(second_wire)
            matched = (
                replay_bundle.artifact_hash
                == bundle.artifact_hash
                == first_reloaded.artifact_hash
                == second_reloaded.artifact_hash
                and canonical_json_hash(first_wire) == canonical_json_hash(second_wire)
            )
            if not matched:  # pragma: no cover - 只在确定性合同被破坏时触发
                raise RuntimeError("STAGE9_REPLAY_HASH_MISMATCH")
            first_ref = first_path.resolve().relative_to(self.workspace_root).as_posix()
            second_ref = second_path.resolve().relative_to(self.workspace_root).as_posix()
            return (
                {
                    "replay_report": {
                        "schema_version": "stage9-replay-report-v1",
                        "matched": matched,
                        "first_hash": bundle.artifact_hash,
                        "second_hash": replay_bundle.artifact_hash,
                        "first_manifest_ref": first_ref,
                        "second_manifest_ref": second_ref,
                    },
                    "hash_comparison": {
                        "schema_version": "stage9-hash-comparison-v1",
                        "config_hash": request.config.config_hash,
                        "artifact_hashes_equal": matched,
                        "canonical_manifest_hash": canonical_json_hash(first_wire),
                    },
                    "gate_summary": {
                        "schema_version": "stage9-replay-gate-summary-v1",
                        "gate_status": "NOT_RUN",
                        "local_validation_status": "PASS" if matched else "FAIL",
                    },
                },
                {"bundle_hash": bundle.artifact_hash, "replay_match": matched},
            )
        raise ValueError(f"STAGE789_ANALYSIS_TASK_UNHANDLED:{task_id}")

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        if request.task.runner_kind is not self.runner_kind:
            raise ValueError("STAGE789_REQUEST_RUNNER_KIND_MISMATCH")
        if request.task.task_id not in self.handled_task_ids:
            if self.fallback is not None:
                return self.fallback.run(request)
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    f"fallback:{self.runner_kind.value}",
                    f"复合 runner 不处理 {request.task.task_id}，且未配置同 kind fallback",
                    False,
                )
            )
        completed = self._completed(request)
        if completed is not None:
            return completed
        self._authorize_partial_resume(request)
        try:
            inputs = _load_configured_inputs(request, self.workspace_root)
        except FileNotFoundError as error:
            if request.config.run_intent != "formal":
                raise
            refs = _configured_input_refs(request)
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "referenced_input_artifact",
                    f"formal 输入引用当前不可发现：{error}",
                    True,
                    refs,
                )
            ) from error
        if self.runner_kind is RunnerKind.CONTRACT:
            payloads, metadata = self._run_ablation_freeze(request, inputs)
        elif self.runner_kind is RunnerKind.PRUNING:
            payloads, metadata = self._run_pruning(request, inputs)
        elif self.runner_kind is RunnerKind.ABLATION:
            payloads, metadata = self._run_ablation(request, inputs)
        else:
            payloads, metadata = self._run_analysis_family(request, inputs)
        source_refs = [item.ref for item in inputs]
        if request.config.run_intent == "formal":
            if request.environment.estimator_decision_ref is not None:
                source_refs.append(request.environment.estimator_decision_ref)
            source_refs.extend(
                request.environment.evidence_refs[key]
                for key in sorted(request.environment.evidence_refs)
            )
        return self._publish(
            request,
            payloads,
            source_refs=tuple(dict.fromkeys(source_refs)),
            metadata=metadata,
        )


def build_stage789_composite_runners(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
    pruning_resource_loader: PruningResourceLoader | None = None,
    ablation_resource_loader: AblationResourceLoader | None = None,
    include_delivery_and_replay: bool = True,
) -> tuple[Stage789CompositeTaskRunner, ...]:
    """构建供统一 runtime 工厂注册的复合 runner 集合。

    工厂应先注册本函数返回值，再注册尚未覆盖的 runner kind。共享 kind 的旧实现
    通过 ``fallbacks`` 传入，而不是在 :class:`TaskRuntime` 中进行不安全覆盖。
    """

    fallback_map = dict(fallbacks or {})
    kinds = [
        RunnerKind.CONTRACT,
        RunnerKind.PRUNING,
        RunnerKind.ABLATION,
        RunnerKind.ANALYSIS,
        RunnerKind.STATISTICS,
        RunnerKind.REPORTING,
    ]
    if include_delivery_and_replay:
        kinds.extend((RunnerKind.DELIVERY, RunnerKind.TEST_MATRIX))
    return tuple(
        Stage789CompositeTaskRunner(
            runner_kind=kind,
            workspace_root=Path(workspace_root),
            fallback=fallback_map.get(kind),
            pruning_resource_loader=pruning_resource_loader,
            ablation_resource_loader=ablation_resource_loader,
        )
        for kind in kinds
    )


def build_stage789_runner_overrides(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
    pruning_resource_loader: PruningResourceLoader | None = None,
    ablation_resource_loader: AblationResourceLoader | None = None,
    include_delivery_and_replay: bool = True,
) -> Mapping[RunnerKind, Stage789CompositeTaskRunner]:
    """返回便于默认 runtime 工厂逐层组合的只读 ``RunnerKind -> runner`` 映射。

    该接口与 Stage 4--6 的 override 工厂形状一致；调用方可以把上一层组合 runner
    作为 ``fallbacks`` 传入，再一次性注册本映射的 values。
    """

    runners = build_stage789_composite_runners(
        workspace_root,
        fallbacks=fallbacks,
        pruning_resource_loader=pruning_resource_loader,
        ablation_resource_loader=ablation_resource_loader,
        include_delivery_and_replay=include_delivery_and_replay,
    )
    return MappingProxyType({runner.runner_kind: runner for runner in runners})


def register_stage789_task_runners(
    runtime: TaskRuntime,
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
    pruning_resource_loader: PruningResourceLoader | None = None,
    ablation_resource_loader: AblationResourceLoader | None = None,
    include_delivery_and_replay: bool = True,
) -> tuple[Stage789CompositeTaskRunner, ...]:
    """注册复合 runner 并返回同一组实例，便于工厂测试其分派范围。"""

    runners = build_stage789_composite_runners(
        workspace_root,
        fallbacks=fallbacks,
        pruning_resource_loader=pruning_resource_loader,
        ablation_resource_loader=ablation_resource_loader,
        include_delivery_and_replay=include_delivery_and_replay,
    )
    for runner in runners:
        runtime.register(runner)
    return runners


__all__ = [
    "AblationMatrixDeclaration",
    "AblationResourceLoader",
    "AblationTaskResources",
    "LoadedInputArtifact",
    "PruningResourceLoader",
    "PruningTaskResources",
    "Stage789CompositeTaskRunner",
    "build_stage789_composite_runners",
    "build_stage789_runner_overrides",
    "register_stage789_task_runners",
]
