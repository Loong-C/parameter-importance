"""Stage 0--9 任务目录的默认执行器注册表。

这里把严格 ``ResolvedConfigV2`` 真正接到训练与通用任务生命周期。所有 runner
共享三条硬规则：

1. 输出只能位于配置声明的 workspace 相对目录；
2. 先发布内容寻址对象，再发布独立权威 commit；
3. formal 不会回退到 tiny/synthetic，外部依赖或资产不足时抛出结构化 blocker。

训练 runner 已执行真实 Torch autograd、optimizer、scheduler、在线重要性、事件和
完整 checkpoint。其余合同/审计/派生型任务使用 ``CatalogTaskRunner``：它会验证
全部输入引用并发布可重放的执行证据，而不是生成临时脚本。Stage 2/3、剪枝、消融
和 Stage 9 的专用 runner 会在同一注册工厂中覆盖相应 ``RunnerKind``。
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import platform
import sys
from typing import Iterator, Mapping, Sequence

import torch

from ..assets import AssetManifestError, resolve_ready_asset
from ..atomic import sha256_file
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.seed import SeedPlan
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from ..core.estimators import double_sample_importance, equal_u_importance, raw_importance
from ..core.sufficient_statistics import EqualSufficientStatistics
from ..core.tensors import TensorMap
from ..providers import (
    CausalLMEvaluator,
    ClassificationEvaluator,
    HuggingFaceTaskMetricEvaluator,
    OfflineHuggingFaceModelAdapter,
    PretokenizedGlueDatasetAdapter,
    PretokenizedPileDatasetAdapter,
    TinyTrainingFixture,
    TorchModelAdapter,
    build_tiny_training_fixture,
    configure_batch_cursor,
)
from ..providers.optional import DependencyUnavailable
from ..runtime.checkpoint import CheckpointStore
from ..runtime.events import JsonlEventSink, read_event_stream
from ..runtime.task_artifacts import TaskArtifactStore
from ..runtime.telemetry import ResourceProfile, ResourceSampler
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
from ..runtime.training import (
    TrainingEngine,
    TrainingRunResult,
    TrainingRunSpec,
    TrainingState,
    install_training_rng,
)
from ..runtime.training_factory import build_grad_scaler, build_optimizer, build_scheduler
from .training_endpoints import TrainingEndpointObserver


def _logical_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"TASK_RUNNER_LOGICAL_PATH_INVALID:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"TASK_RUNNER_PATH_ESCAPE:{field}")
    return path


def _resolve_workspace_path(root: Path, value: str, *, field: str) -> Path:
    logical = _logical_path(value, field=field)
    target = root.joinpath(*logical.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"TASK_RUNNER_PATH_ESCAPE:{field}") from error
    return target


def _artifact_store(request: TaskExecutionRequest, root: Path) -> TaskArtifactStore:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    return TaskArtifactStore(root, str(artifacts["output_dir"]))


def _checkpoint_result_ref(artifact_refs: Mapping[str, str]) -> str | None:
    for kind, reference in artifact_refs.items():
        if "checkpoint" in kind or kind == "training_state_manifest":
            return reference
    return None


def _semantic_event_stream_hash(path: Path) -> str:
    """计算不含采集时间/随机 event ID 的事件语义摘要。

    原始 JSONL 仍保留 ``event_id`` 与 ``occurred_at``，便于运维审计；它们不是训练
    estimand，也不能让两个相同 fixture 的功能 artifact hash 漂移。序号、lineage、
    类型和 payload 全部进入摘要，所以删除、重排或篡改真正事件内容仍会被发现。
    """

    events = read_event_stream(path)
    return canonical_json_hash(
        {
            "schema_version": "semantic-event-stream-v1",
            "events": [
                {
                    "schema_version": event.schema_version,
                    "experiment_id": event.experiment_id,
                    "run_id": event.run_id,
                    "attempt_id": event.attempt_id,
                    "session_id": event.session_id,
                    "rank": event.rank,
                    "event_type": event.event_type,
                    "sequence": event.sequence,
                    "payload": event.payload,
                }
                for event in events
            ],
        }
    )


def _completed_result(
    request: TaskExecutionRequest,
    store: TaskArtifactStore,
) -> TaskRunResult | None:
    refs = store.discover_complete(
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
        checkpoint_ref=_checkpoint_result_ref(refs),
        message="task completed",
        metadata={"execution_contract": "run-ready-v1"},
    )


def _input_evidence(request: TaskExecutionRequest, root: Path) -> tuple[list[JSONValue], tuple[str, ...]]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    refs = tuple(str(item) for item in orchestration["input_result_refs"])
    evidence: list[JSONValue] = []
    for reference in refs:
        path = _resolve_workspace_path(root, reference, field="input_result_refs")
        if path.is_dir():
            _, identity = load_tensor_bundle(path)
            digest = identity.manifest_sha256
            kind = "tensor_bundle"
        elif path.suffix.casefold() == ".json":
            value = load_canonical_json(path)
            digest = canonical_json_hash(value)
            kind = str(value.get("schema_version", "json")) if isinstance(value, dict) else "json"
        elif path.is_file():
            digest = sha256_file(path)
            kind = "hash_bound_file"
        else:
            raise FileNotFoundError(f"TASK_INPUT_ARTIFACT_NOT_FOUND:{reference}")
        evidence.append({"ref": reference, "sha256": digest, "kind": kind})
    return evidence, refs


@dataclass(frozen=True, slots=True)
class _EndpointCapturePlan:
    """训练 CLI 可直接消费的版本化 endpoint 选择计划。"""

    reference: str | None
    selected_steps: frozenset[int]
    include_checkpoint_steps: bool
    scope: str
    formal_eligible: bool
    qualification_evidence_hash: str | None
    probe_plan_ref: str | None


def _endpoint_capture_plan(
    request: TaskExecutionRequest,
    root: Path,
    spec: TrainingRunSpec,
) -> _EndpointCapturePlan | None:
    """从 ``input_result_refs`` 加载唯一的 hash-bound endpoint 计划。

    若 base config 只打开 ``path_integration`` 而未提供计划，本机 fixture 会按
    checkpoint 边界自动选择；formal 则 fail-closed，防止临时改 Python 代码挑点。
    """

    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for reference in orchestration["input_result_refs"]:
        path = _resolve_workspace_path(root, str(reference), field="input_result_refs")
        if path.suffix.casefold() != ".json":
            continue
        value = load_canonical_json(path)
        if isinstance(value, Mapping) and value.get("schema_version") == (
            "training-endpoint-capture-plan-v1"
        ):
            candidates.append((str(reference), value))
    base_path = request.config.base_config.section("path_integration")
    assert isinstance(base_path, dict)
    if not candidates:
        if not bool(base_path["enabled"]):
            return None
        if request.config.run_intent == "formal":
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "training_endpoint_capture_plan",
                    "formal 路径审计必须在 input_result_refs 中绑定版本化 endpoint 选择计划",
                    False,
                )
            )
        selected = frozenset(
            step for step in range(1, spec.max_steps + 1) if spec.should_checkpoint(step)
        )
        if not selected:
            selected = frozenset({spec.max_steps})
        return _EndpointCapturePlan(
            None, selected, True, "local_fixture", False, None, None
        )
    if len(candidates) != 1:
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_NOT_UNIQUE")
    reference, value = candidates[0]
    expected = {
        "schema_version",
        "plan_id",
        "selected_steps",
        "include_checkpoint_steps",
        "scope",
        "formal_eligible",
        "qualification_evidence_hash",
        "probe_plan_ref",
        "artifact_hash",
    }
    if set(value) != expected:
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_FIELDS_MISMATCH")
    declared = value["artifact_hash"]
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_HASH_MISMATCH")
    raw_steps = value["selected_steps"]
    if not isinstance(raw_steps, list) or any(
        isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= spec.max_steps
        for step in raw_steps
    ):
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_STEPS_INVALID")
    if len(set(raw_steps)) != len(raw_steps):
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_STEPS_DUPLICATE")
    include = value["include_checkpoint_steps"]
    scope = value["scope"]
    formal = value["formal_eligible"]
    evidence_hash = value["qualification_evidence_hash"]
    if type(include) is not bool or type(formal) is not bool:
        raise TypeError("TRAINING_ENDPOINT_CAPTURE_PLAN_BOOLEAN_INVALID")
    if scope != request.config.run_intent or formal != (scope == "formal"):
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_SCOPE_MISMATCH")
    selected = set(raw_steps)
    if include:
        selected.update(
            step for step in range(1, spec.max_steps + 1) if spec.should_checkpoint(step)
        )
    if not selected:
        raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_EMPTY")
    if request.config.run_intent == "formal":
        evidence_ref = request.environment.evidence_refs.get("formal_execution")
        if evidence_ref is None:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "formal_execution_evidence",
                    "formal endpoint 捕获计划缺少 runtime FormalExecutionEvidence 引用",
                    True,
                    (reference,),
                )
            )
        evidence_value = load_canonical_json(
            _resolve_workspace_path(root, evidence_ref, field="formal_execution")
        )
        if not isinstance(evidence_value, Mapping):
            raise ValueError("FORMAL_EXECUTION_EVIDENCE_ROOT_INVALID")
        evidence = FormalExecutionEvidence.from_mapping(evidence_value)
        evidence.require_for_stage(3)
        if evidence_hash != evidence.artifact_hash:
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_EVIDENCE_HASH_MISMATCH")
    elif evidence_hash is not None:
        raise ValueError("LOCAL_ENDPOINT_PLAN_CANNOT_CARRY_FORMAL_EVIDENCE")
    probe_ref = value["probe_plan_ref"]
    if probe_ref is not None:
        _resolve_workspace_path(root, str(probe_ref), field="probe_plan_ref")
    return _EndpointCapturePlan(
        reference,
        frozenset(selected),
        include,
        str(scope),
        formal,
        None if evidence_hash is None else str(evidence_hash),
        None if probe_ref is None else str(probe_ref),
    )


def _publish_payloads(
    request: TaskExecutionRequest,
    store: TaskArtifactStore,
    payloads: Mapping[str, Mapping[str, JSONValue]],
    *,
    source_refs: tuple[str, ...] = (),
) -> Mapping[str, str]:
    if tuple(payloads) != request.task.artifact_kinds:
        raise ValueError("TASK_RUNNER_PAYLOAD_KIND_ORDER_MISMATCH")
    refs: dict[str, str] = {}
    for kind, payload in payloads.items():
        published = store.publish(
            task_id=request.task.task_id,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent=request.config.run_intent,
            payload=payload,
            formal_eligible=request.config.run_intent == "formal",
            source_refs=source_refs,
        )
        refs[kind] = published.commit_ref
    return refs


def _deterministic_core_probe() -> Mapping[str, JSONValue]:
    """执行可手算的 raw/U/double 探针，证明核心导入不是空壳。"""

    samples = [
        TensorMap({"weight": torch.tensor([1.0, -1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([3.0, 1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([2.0, 0.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([4.0, 2.0], dtype=torch.float64)}),
    ]
    statistics = EqualSufficientStatistics.from_samples(
        samples, accumulation_dtype=torch.float64
    )
    first = EqualSufficientStatistics.from_samples(
        samples[:2], accumulation_dtype=torch.float64
    ).mean_gradient
    second = EqualSufficientStatistics.from_samples(
        samples[2:], accumulation_dtype=torch.float64
    ).mean_gradient
    raw = raw_importance(statistics.mean_gradient)
    u = equal_u_importance(statistics)
    double = double_sample_importance(first, second)
    return {
        "sample_count": len(samples),
        "raw_sum": float(raw.scalar_sum(dtype=torch.float64).item()),
        "u_sum": float(u.scalar_sum(dtype=torch.float64).item()),
        "double_sum": float(double.scalar_sum(dtype=torch.float64).item()),
        "dtype": "float64",
    }


@dataclass(slots=True)
class CatalogTaskRunner(TaskRunner):
    """审计、合同、派生和交付类任务的确定性执行器。"""

    runner_kind: RunnerKind
    workspace_root: Path

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        store = _artifact_store(request, self.workspace_root)
        existing = _completed_result(request, store)
        if existing is not None:
            return existing
        inputs, source_refs = _input_evidence(request, self.workspace_root)
        common: dict[str, JSONValue] = {
            "schema_version": "catalog-task-output-v1",
            "task_definition_hash": canonical_json_hash(request.task.to_dict()),
            "catalog_hash": DEFAULT_TASK_CATALOG.catalog_hash,
            "config_full_hash": request.config.full_hash,
            "runner_kind": self.runner_kind.value,
            "input_evidence": inputs,
            "environment": {
                "python": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": sys.platform,
                "torch": torch.__version__,
            },
            "core_probe": _deterministic_core_probe(),
            "checks": [
                "strict_config_validated",
                "catalog_binding_validated",
                "input_hashes_validated",
                "canonical_two_phase_publish",
            ],
        }
        payloads: dict[str, Mapping[str, JSONValue]] = {}
        for kind in request.task.artifact_kinds:
            payloads[kind] = {
                **common,
                "artifact_role": kind,
                "local_validation_status": (
                    "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
                ),
                # Gate 是独立审核决策；runner 只能发布候选证据，不能替审核人
                # 自动把 formal Gate 写成 PASS。
                "gate_status": "NOT_RUN" if "gate" in kind else None,
            }
        refs = _publish_payloads(request, store, payloads, source_refs=source_refs)
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            checkpoint_ref=_checkpoint_result_ref(refs),
            message="task completed",
            metadata={"execution_contract": "run-ready-v1"},
        )


@dataclass(frozen=True, slots=True)
class _TrainingResources:
    model: TorchModelAdapter
    dataset: object
    evaluation_dataset: object | None
    evaluator: object | None
    task_name: str
    asset_evidence: tuple[Mapping[str, JSONValue], ...]


def _training_resources(
    request: TaskExecutionRequest,
    root: Path,
    *,
    rank: int,
    world_size: int,
) -> _TrainingResources:
    providers = request.config.section("providers")
    training = request.config.section("training")
    batching = request.config.base_config.section("batching")
    data = request.config.base_config.section("data")
    identity = request.config.base_config.section("identity")
    evaluation = request.config.section("evaluation")
    assert (
        isinstance(providers, dict)
        and isinstance(training, dict)
        and isinstance(evaluation, dict)
    )
    max_steps = training["max_steps"]
    if not isinstance(max_steps, int):
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.CAPABILITY_UNAVAILABLE,
                "epoch_dataset_adapter",
                "当前 provider 必须把训练长度解析为 training.max_steps",
                False,
            )
        )
    task_type = str(providers["task_type"])
    task_name = str(providers["task_name"])
    if providers["kind"] == "tiny":
        if request.config.run_intent != "local_fixture":
            raise RuntimeError("FORMAL_TRAINING_MUST_NOT_USE_TINY_PROVIDER")
        if task_type not in {"causal_lm", "sequence_classification"}:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    "tiny_task_type",
                    f"tiny 训练只支持 causal_lm/sequence_classification，收到 {task_type}",
                    False,
                )
            )
        microbatch_size = int(batching["microbatch_size"])
        microbatches_per_step = int(batching["accumulation_steps"]) * (
            int(batching["per_device_batch_size"]) // microbatch_size
        )
        seed_plan = SeedPlan.from_master_seed(
            int(identity["master_seed"]), world_size=world_size
        )
        model_fixture: TinyTrainingFixture = build_tiny_training_fixture(
            task_type=task_type,
            seed=seed_plan.seed_for("model_init"),
            # InMemoryDatasetAdapter 按 rank 对 step 序列切片，因此先为所有 rank
            # 生成交错计划，保证每个 rank 都恰好拥有 max_steps 个 optimizer step。
            steps=max_steps * world_size,
            microbatches_per_step=microbatches_per_step,
            microbatch_size=microbatch_size,
            sequence_length=int(data["sequence_length"]),
            num_labels=int(providers["num_labels"] or 3),
        )
        data_fixture = build_tiny_training_fixture(
            task_type=task_type,
            seed=seed_plan.seed_for("sampler"),
            steps=max_steps * world_size,
            microbatches_per_step=microbatches_per_step,
            microbatch_size=microbatch_size,
            sequence_length=int(data["sequence_length"]),
            num_labels=int(providers["num_labels"] or 3),
        )
        fixture = TinyTrainingFixture(model_fixture.model, data_fixture.dataset)
        evaluator = (
            CausalLMEvaluator()
            if task_type == "causal_lm"
            else ClassificationEvaluator()
        )
        return _TrainingResources(
            fixture.model,
            fixture.dataset,
            fixture.dataset if bool(evaluation["enabled"]) else None,
            evaluator if bool(evaluation["enabled"]) else None,
            task_name,
            (),
        )

    manifest_fields = (
        ("model_manifest_ref", "model_root_ref"),
        ("data_manifest_ref", "data_root_ref"),
        ("tokenizer_manifest_ref", "tokenizer_root_ref"),
    )
    resolved_assets = []
    evidence: list[Mapping[str, JSONValue]] = []
    for manifest_field, root_field in manifest_fields:
        manifest_ref = str(providers[manifest_field])
        root_ref = str(providers[root_field])
        manifest_path = _resolve_workspace_path(root, manifest_ref, field=manifest_field)
        asset_root = _resolve_workspace_path(root, root_ref, field=root_field)
        asset = resolve_ready_asset(manifest_path, asset_root)
        resolved_assets.append(asset)
        evidence.append(
            {
                "manifest_ref": manifest_ref,
                "root_ref": root_ref,
                "asset_id": asset.asset_id,
                "file_count": len(asset.files),
            }
        )
    model_asset, data_asset, _tokenizer_asset = resolved_assets
    model = OfflineHuggingFaceModelAdapter.from_local_directory(
        model_asset.root,
        task_type=task_type,
        num_labels=providers["num_labels"],  # type: ignore[arg-type]
    )
    if task_name == "fixture":
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.CAPABILITY_UNAVAILABLE,
                "offline_task_name",
                "offline_hf 训练必须显式选择 pile/sst2/mnli/rte，而不能沿用 fixture",
                False,
            )
        )
    microbatches_per_step = int(batching["accumulation_steps"]) * (
        int(batching["per_device_batch_size"])
        // int(batching["microbatch_size"])
    )

    def build_dataset(*, split: str, microbatch_size: int, micros: int) -> object:
        if task_name == "pile":
            return PretokenizedPileDatasetAdapter(
                data_asset.root,
                dataset_id=data_asset.asset_id,
                split=split,
                microbatch_size=microbatch_size,
                microbatches_per_step=micros,
                sampling_design=str(data["sampling_design"]),
                weights_exogenous=bool(data["weights_exogenous"]),
                common_mean_assumption=bool(data["common_mean_assumption"]),
                allowed_root=data_asset.root,
            )
        return PretokenizedGlueDatasetAdapter(
            data_asset.root,
            task_name=task_name,
            split=split,
            dataset_id=data_asset.asset_id,
            microbatch_size=microbatch_size,
            microbatches_per_step=micros,
            allowed_root=data_asset.root,
        )

    dataset = build_dataset(
        split=str(data["split"]),
        microbatch_size=int(batching["microbatch_size"]),
        micros=microbatches_per_step,
    )
    evaluation_dataset = None
    evaluator = None
    if bool(evaluation["enabled"]):
        evaluation_dataset = build_dataset(
            split=str(evaluation["split"]),
            microbatch_size=int(evaluation["batch_size"]),
            micros=1,
        )
        evaluator = HuggingFaceTaskMetricEvaluator(
            task_name, split=str(evaluation["split"])
        )
    evidence.append(
        {
            "provider_task_name": task_name,
            "training_split": str(data["split"]),
            "evaluation_split": evaluation["split"],
            "dataset_state_digest": getattr(dataset, "state_digest")(),
        }
    )
    return _TrainingResources(
        model,
        dataset,
        evaluation_dataset,
        evaluator,
        task_name,
        tuple(evidence),
    )


def _decision_hash(request: TaskExecutionRequest, root: Path) -> tuple[str | None, str | None]:
    if request.config.run_intent == "local_fixture":
        return None, None
    # Stage 0/1 在产生 Stage 2 EstimatorDecision 之前就必须能够运行；只有任务目录
    # 明确声明消费 decision 的 Stage 3+ 执行路径才读取它。v1 配置中的
    # require_decision_for_formal 是全局安全默认值，不能反过来制造时间循环依赖。
    if not request.task.execution_policy.requires_estimator_decision:
        return None, None
    reference = request.environment.estimator_decision_ref
    if reference is None:
        raise RuntimeError("FORMAL_ESTIMATOR_DECISION_PREFLIGHT_BYPASSED")
    value = load_canonical_json(_resolve_workspace_path(root, reference, field="decision_ref"))
    if not isinstance(value, dict):
        raise ValueError("ESTIMATOR_DECISION_ROOT_INVALID")
    digest = value.get("artifact_hash") or value.get("result_hash")
    if not isinstance(digest, str) or len(digest) != 64:
        # task-output commit 绑定的实际对象 hash 也可作为已审核 decision 身份。
        digest = value.get("artifact_hash")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("ESTIMATOR_DECISION_HASH_MISSING")
    return digest, "PASS"


def _evaluation_store(
    request: TaskExecutionRequest,
    root: Path,
    *,
    global_step: int,
) -> TaskArtifactStore:
    """为单个 evaluation 边界建立独立两阶段提交目录。"""

    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    output_dir = str(artifacts["output_dir"])
    return TaskArtifactStore(
        root,
        f"{output_dir}/evaluation/step-{global_step:08d}",
    )


def _load_evaluation_record(
    request: TaskExecutionRequest,
    root: Path,
    *,
    global_step: int,
) -> Mapping[str, JSONValue] | None:
    """只通过权威 commit 读取已完成 evaluation；孤立对象不算完成。"""

    store = _evaluation_store(request, root, global_step=global_step)
    refs = store.discover_complete(
        task_id=request.task.task_id,
        config_hash=request.config.config_hash,
        artifact_kinds=("metrics",),
        formal_eligible=request.config.run_intent == "formal",
    )
    if refs is None:
        return None
    published = store.load_commit(refs["metrics"])
    value = load_canonical_json(
        _resolve_workspace_path(root, published.object_ref, field="evaluation_object_ref")
    )
    if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
        raise ValueError("EVALUATION_OBJECT_PAYLOAD_INVALID")
    return dict(value["payload"])


def _run_evaluation(
    request: TaskExecutionRequest,
    root: Path,
    resources: _TrainingResources,
    *,
    global_step: int,
    rank: int,
) -> Mapping[str, JSONValue]:
    """在固定 validation panel 上执行只读评估并按 step 幂等提交。"""

    existing = _load_evaluation_record(
        request, root, global_step=global_step
    )
    if existing is not None:
        return existing
    if resources.evaluation_dataset is None or resources.evaluator is None:
        raise RuntimeError("EVALUATION_RESOURCES_NOT_CONFIGURED")
    evaluation = request.config.section("evaluation")
    identity = request.config.base_config.section("identity")
    assert isinstance(evaluation, dict)
    max_batches = int(evaluation["max_batches"] or 1)
    # 每个 DDP rank 都在同一个完整 validation panel 上只读 forward。这样无需把
    # rank 局部平均再错误地平均一次，也避免可变 token count 的加权歧义。
    evaluation_seed = SeedPlan.from_master_seed(
        int(identity["master_seed"])
    ).derive_subseed("sampler", "evaluation")
    cursor = resources.evaluation_dataset.cursor(  # type: ignore[attr-defined]
        seed=evaluation_seed,
        rank=0,
        world_size=1,
    )
    microbatches = []
    for _ in range(max_batches):
        try:
            microbatches.extend(cursor.next_microbatches())
        except StopIteration:
            break
    if not microbatches:
        raise ValueError("EVALUATION_PANEL_EMPTY")
    values = resources.evaluator.evaluate(resources.model, tuple(microbatches))  # type: ignore[attr-defined]
    requested = tuple(str(name) for name in evaluation["metrics"])
    missing = [name for name in requested if name not in values]
    if missing:
        raise ValueError(f"EVALUATION_METRIC_NOT_PRODUCED:{','.join(missing)}")
    metrics: dict[str, JSONValue] = {name: float(values[name]) for name in requested}
    payload: dict[str, JSONValue] = {
        "schema_version": "training-evaluation-record-v1",
        "global_step": global_step,
        "split": str(evaluation["split"]),
        "task_name": resources.task_name,
        "metrics": metrics,
        "panel_batch_ids": [batch.batch_id for batch in microbatches],
        "panel_sample_ids_hash": canonical_json_hash(
            [sample_id for batch in microbatches for sample_id in batch.sample_ids]
        ),
    }
    # 只有 rank 0 发布；其他 rank 仍执行同一只读 forward，以满足 DDP forward 的
    # collective 约束。barrier 由外层 executor 在训练 task 完成边界统一处理。
    if rank == 0:
        _evaluation_store(request, root, global_step=global_step).publish(
            task_id=request.task.task_id,
            artifact_kind="metrics",
            config_hash=request.config.config_hash,
            run_intent=request.config.run_intent,
            payload=payload,
            formal_eligible=request.config.run_intent == "formal",
        )
    return payload


@contextmanager
def _deterministic_algorithms(enabled: bool) -> Iterator[None]:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(enabled)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _synchronize_profile_device(device_name: str, *, enabled: bool) -> None:
    """仅在配置显式要求时同步 CUDA；CPU 不制造伪同步事件。"""

    if not enabled:
        return
    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PROFILE_CUDA_SYNCHRONIZE_WITHOUT_CUDA")
        torch.cuda.synchronize(device)


def _resource_window_path(
    output_root: Path,
    *,
    rank: int,
    repetition: int,
) -> Path:
    return (
        output_root
        / "resource-profiles"
        / f"rank-{rank:04d}"
        / "commits"
        / f"window-{repetition:04d}.json"
    )


def _validate_resource_window(value: object) -> Mapping[str, JSONValue]:
    """严格复核可恢复 profiling 窗口，不接受无 hash 的手工结果。"""

    expected = {
        "schema_version",
        "repetition",
        "start_step",
        "end_step",
        "requested",
        "profile",
        "communication",
        "artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("TRAINING_RESOURCE_WINDOW_FIELDS_INVALID")
    if value.get("schema_version") != "training-resource-window-v1":
        raise ValueError("TRAINING_RESOURCE_WINDOW_VERSION_INVALID")
    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise ValueError("TRAINING_RESOURCE_WINDOW_HASH_MISMATCH")
    profile = value.get("profile")
    if not isinstance(profile, Mapping) or profile.get("schema_version") != (
        "training-resource-profile-v1"
    ):
        raise ValueError("TRAINING_RESOURCE_PROFILE_INVALID")
    profile_declared = profile.get("profile_hash")
    profile_body = {key: item for key, item in profile.items() if key != "profile_hash"}
    if not isinstance(profile_declared, str) or profile_declared != canonical_json_hash(
        profile_body
    ):
        raise ValueError("TRAINING_RESOURCE_PROFILE_HASH_MISMATCH")
    return dict(value)  # type: ignore[return-value]


def _publish_resource_window(
    commit_path: Path,
    *,
    workspace_root: Path,
    repetition: int,
    start_step: int,
    end_step: int,
    requested: Mapping[str, JSONValue],
    profile: ResourceProfile,
    capture_throughput: bool,
    capture_communication: bool,
) -> Mapping[str, JSONValue]:
    """用“内容寻址对象 + 独立 commit”发布一个不可变测量窗口。"""

    from ..contracts.jsonio import write_canonical_json

    profile_payload = profile.to_dict()
    if not capture_throughput:
        profile_payload["steps_per_second"] = None
        profile_payload["units_per_second"] = None
        profile_payload["profile_hash"] = canonical_json_hash(
            {key: item for key, item in profile_payload.items() if key != "profile_hash"}
        )
    communication: dict[str, JSONValue] = {
        "requested": capture_communication,
        "defined": False,
        "bytes": None,
        "reason": (
            "backend_exact_communication_counter_unavailable"
            if capture_communication
            else "not_requested"
        ),
    }
    body: dict[str, JSONValue] = {
        "schema_version": "training-resource-window-v1",
        "repetition": repetition,
        "start_step": start_step,
        "end_step": end_step,
        "requested": dict(requested),
        "profile": profile_payload,
        "communication": communication,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    object_dir = commit_path.parent.parent / "objects"
    object_path = object_dir / f"{body['artifact_hash']}.json"
    if object_path.exists():
        if _validate_resource_window(load_canonical_json(object_path)) != body:
            raise RuntimeError("TRAINING_RESOURCE_WINDOW_OBJECT_DRIFT")
    else:
        write_canonical_json(object_path, body)
    object_ref = object_path.resolve().relative_to(workspace_root).as_posix()
    commit_body: dict[str, JSONValue] = {
        "schema_version": "training-resource-window-commit-v1",
        "repetition": repetition,
        "artifact_hash": str(body["artifact_hash"]),
        "object_ref": object_ref,
    }
    commit_body["commit_hash"] = canonical_json_hash(commit_body)
    if commit_path.exists():
        existing_commit = load_canonical_json(commit_path)
        if existing_commit != commit_body:
            raise FileExistsError("TRAINING_RESOURCE_WINDOW_COMMIT_CONFLICT")
    else:
        write_canonical_json(commit_path, commit_body)
    return _load_resource_window_commit(commit_path, workspace_root=workspace_root)


def _load_resource_window_commit(
    commit_path: Path,
    *,
    workspace_root: Path,
) -> Mapping[str, JSONValue]:
    """沿权威 commit 发现 profiling object，并重算两层 hash。"""

    commit = load_canonical_json(commit_path)
    expected = {
        "schema_version", "repetition", "artifact_hash", "object_ref", "commit_hash"
    }
    if not isinstance(commit, Mapping) or set(commit) != expected:
        raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_FIELDS_INVALID")
    if commit.get("schema_version") != "training-resource-window-commit-v1":
        raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_VERSION_INVALID")
    declared_commit = commit.get("commit_hash")
    commit_without_hash = {
        key: item for key, item in commit.items() if key != "commit_hash"
    }
    if not isinstance(declared_commit, str) or declared_commit != canonical_json_hash(
        commit_without_hash
    ):
        raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_HASH_MISMATCH")
    object_ref = commit.get("object_ref")
    if not isinstance(object_ref, str):
        raise TypeError("TRAINING_RESOURCE_WINDOW_OBJECT_REF_INVALID")
    object_path = _resolve_workspace_path(
        workspace_root, object_ref, field="resource_profile.object_ref"
    )
    if object_path.parent.name != "objects":
        raise ValueError("TRAINING_RESOURCE_WINDOW_OBJECT_LAYOUT_INVALID")
    body = _validate_resource_window(load_canonical_json(object_path))
    if body.get("artifact_hash") != commit.get("artifact_hash"):
        raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_OBJECT_MISMATCH")
    return body


@dataclass(slots=True)
class TrainingTaskRunner(TaskRunner):
    """配置驱动的单进程真实训练执行器。"""

    workspace_root: Path
    runner_kind: RunnerKind = RunnerKind.TRAINING

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    def _run_training(
        self,
        request: TaskExecutionRequest,
        *,
        rank: int = 0,
        world_size: int = 1,
        reducer: object | None = None,
        resources: _TrainingResources | None = None,
        wrapped_model: TorchModelAdapter | None = None,
    ) -> tuple[
        TrainingRunResult,
        TrainingEngine,
        Path,
        tuple[Mapping[str, JSONValue], ...],
        tuple[Mapping[str, JSONValue], ...],
        tuple[Mapping[str, JSONValue], ...],
    ]:
        training = request.config.section("training")
        scheduler_options = request.config.section("scheduler")
        precision = request.config.section("precision_runtime")
        optimizer_runtime = request.config.section("optimizer_runtime")
        checkpoint_schedule = request.config.section("checkpoint_schedule")
        evaluation = request.config.section("evaluation")
        profiling = request.config.section("profiling")
        data_loader = request.config.section("data_loader")
        base = request.config.base_config
        base_optimizer = base.section("optimizer")
        base_importance = base.section("importance")
        base_data = base.section("data")
        base_logging = base.section("logging")
        identity = base.section("identity")
        assert all(
            isinstance(value, dict)
            for value in (
                training, scheduler_options, precision, optimizer_runtime,
                checkpoint_schedule, evaluation, profiling, data_loader,
            )
        )
        max_steps = training["max_steps"]
        if not isinstance(max_steps, int):
            raise ValueError("TRAINING_TASK_REQUIRES_MAX_STEPS")
        seed_plan = install_training_rng(
            int(identity["master_seed"]),
            rank=rank,
            world_size=world_size,
        )
        if resources is None:
            try:
                resources = _training_resources(
                    request, self.workspace_root, rank=rank, world_size=world_size
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
                        "offline_training_assets",
                        f"离线训练资产不可用：{type(error).__name__}: {error}",
                        True,
                    )
                ) from error
        model = resources.model if wrapped_model is None else wrapped_model
        if wrapped_model is not None:
            resources = _TrainingResources(
                model,
                resources.dataset,
                resources.evaluation_dataset,
                resources.evaluator,
                resources.task_name,
                resources.asset_evidence,
            )
        runtime_device = str(base.section("runtime")["device"])
        if wrapped_model is None:
            model.module.to(torch.device(runtime_device))
        optimizer = build_optimizer(
            model.module.parameters(), base_optimizer, optimizer_runtime
        )
        scheduler = build_scheduler(optimizer, scheduler_options)
        scaler = build_grad_scaler(precision, device_type=runtime_device)
        decision_hash, decision_gate = _decision_hash(request, self.workspace_root)
        autocast_dtype = (
            str(precision["autocast_dtype"])
            if bool(precision["autocast_enabled"])
            else "none"
        )
        execution = request.config.section("execution")
        assert isinstance(execution, dict)
        spec = TrainingRunSpec(
            run_id=f"{request.task.task_id.replace('.', '-')}-rank-{rank:04d}",
            run_intent=request.config.run_intent,
            max_steps=max_steps,
            max_attempts=max_steps + int(execution["max_attempts"]) - 1,
            importance_enabled=True,
            estimator_name=str(base_importance["estimator_name"]),
            accumulation_dtype=str(base.section("precision")["statistic_dtype"]),
            max_grad_norm=training["gradient_clip_max_norm"],  # type: ignore[arg-type]
            autocast_dtype=autocast_dtype,
            checkpoint_every_steps=int(checkpoint_schedule["segments"][0]["every_steps"]),  # type: ignore[index]
            log_every_steps=int(base_logging["log_every_steps"]),
            weights_exogenous=bool(base_data["weights_exogenous"]),
            common_mean_assumption=bool(base_data["common_mean_assumption"]),
            estimator_decision_hash=decision_hash,
            estimator_gate_status=decision_gate,
            metadata={
                "task_id": request.task.task_id,
                "config_hash": request.config.config_hash,
                "seed_plan_hash": seed_plan.artifact_hash,
                "rank": rank,
                "world_size": world_size,
                "data_loader": {
                    "num_workers": data_loader["num_workers"],
                    "prefetch_factor": data_loader["prefetch_factor"],
                    "persistent_workers": data_loader["persistent_workers"],
                    "drop_last": data_loader["drop_last"],
                    "cursor_policy": data_loader["cursor_policy"],
                },
            },
            checkpoint_segments=tuple(
                dict(segment) for segment in checkpoint_schedule["segments"]  # type: ignore[arg-type]
            ),
        )
        artifacts = request.config.section("artifacts")
        assert isinstance(artifacts, dict)
        output_root = _resolve_workspace_path(
            self.workspace_root, str(artifacts["output_dir"]), field="output_dir"
        )
        checkpoint_root = output_root / "checkpoints" / f"rank-{rank:04d}"
        checkpoint_store = CheckpointStore(checkpoint_root)
        events_dir = output_root / "events"
        events_dir.mkdir(parents=True, exist_ok=True)
        recovery = request.config.section("recovery")
        assert isinstance(recovery, dict)
        resume_ref = recovery["resume_ref"]
        committed_checkpoints = checkpoint_store.discover()
        resume_checkpoint_id: str | None = None
        resume_requested = resume_ref is not None
        if not resume_requested and committed_checkpoints:
            # ``task run`` 不能根据输出目录内容猜测恢复点；这既可能把旧实验误接到
            # 新 run，也会让命令语义依赖残留文件。调用方必须在 v2 配置中显式绑定
            # 一个权威 commit/root，再使用 ``task resume``。
            raise RuntimeError("TRAINING_RESUME_REF_REQUIRED_FOR_EXISTING_CHECKPOINTS")
        if resume_requested:
            resume_path = _resolve_workspace_path(
                self.workspace_root,
                str(resume_ref),
                field="recovery.resume_ref",
            )
            checkpoints_root = checkpoint_root.parent.resolve()
            if resume_path in {checkpoint_root.resolve(), checkpoints_root}:
                # DDP 统一绑定 ``.../checkpoints``；每个 rank 只在自己的安全 store
                # 内解析 latest，不读取其他 rank 的状态。
                resume_checkpoint_id = None
            elif (
                resume_path.parent.resolve() == checkpoint_store.commits.resolve()
                and resume_path.suffix.casefold() == ".json"
            ):
                resume_checkpoint_id = resume_path.stem
            elif resume_path == (checkpoint_root / "latest.json").resolve():
                resume_checkpoint_id = None
            else:
                raise ValueError("TRAINING_RESUME_REF_OUTSIDE_RANK_CHECKPOINT_STORE")
            if not committed_checkpoints:
                raise FileNotFoundError("TRAINING_RESUME_REF_HAS_NO_COMMITTED_CHECKPOINT")
        previous_sessions = sorted(events_dir.glob(f"rank-{rank:04d}-session-*.jsonl"))
        session_index = len(previous_sessions)
        events_path = events_dir / f"rank-{rank:04d}-session-{session_index:04d}.jsonl"
        training_cursor = configure_batch_cursor(
            resources.dataset.cursor(
                seed=seed_plan.seed_for("sampler"),
                rank=rank,
                world_size=world_size,
            ),
            num_workers=int(data_loader["num_workers"]),
            prefetch_factor=data_loader["prefetch_factor"],  # type: ignore[arg-type]
            persistent_workers=bool(data_loader["persistent_workers"]),
        )
        engine = TrainingEngine(
            spec=spec,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            reducer=reducer,  # type: ignore[arg-type]
            cursor=training_cursor,
            checkpoint_store=checkpoint_store,
            event_sink=None,
            experiment_id=f"stage-{request.task.stage}-experiment",
            attempt_id="attempt-0000",
            session_id=f"session-rank-{rank:04d}-{session_index:04d}",
            rank=rank,
        )
        endpoint_plan = _endpoint_capture_plan(
            request, self.workspace_root, spec
        )
        endpoint_observer: TrainingEndpointObserver | None = None
        if endpoint_plan is not None:
            endpoint_observer = TrainingEndpointObserver(
                source_run_id=spec.run_id,
                parameter_registry_hash=engine.registry.coordinate_registry_hash,
                selected_steps=endpoint_plan.selected_steps,
                output_root=(
                    output_root / "stage3-endpoints" / f"rank-{rank:04d}"
                ),
                workspace_root=self.workspace_root,
                scope=endpoint_plan.scope,
                formal_eligible=endpoint_plan.formal_eligible,
                qualification_evidence_hash=(
                    endpoint_plan.qualification_evidence_hash
                ),
            )
            endpoint_observer.bind_engine(engine)
            engine.register_observer(endpoint_observer)
        if resume_requested:
            if resume_checkpoint_id is None:
                engine.resume_latest()
            else:
                engine.resume_checkpoint(resume_checkpoint_id)
            # 每个 fresh-process resume 使用新的 session JSONL，序号从零开始；
            # checkpoint 中仍保留上一 session 最后一个已提交序号，二者不会拼成
            # 一个看似连续但实际跨进程的流。
            engine.state = TrainingState(
                engine.state.global_step,
                engine.state.attempt_index,
                engine.state.skipped_steps,
                0,
                engine.state.last_checkpoint_id,
            )
            if endpoint_observer is not None:
                missing_past = sorted(
                    step
                    for step in endpoint_plan.selected_steps
                    if step <= engine.state.global_step
                    and step not in endpoint_observer.captured_steps
                )
                if missing_past:
                    raise RuntimeError(
                        "TRAINING_ENDPOINT_COMMIT_MISSING_BEFORE_RESUME_POINT:"
                        + ",".join(str(step) for step in missing_past)
                    )
        evaluation_records: list[Mapping[str, JSONValue]] = []
        evaluation_enabled = bool(evaluation["enabled"])
        if evaluation_enabled:
            interval = int(
                evaluation["every_steps"]
                or training["validation_every_steps"]
                or max_steps
            )
            evaluation_steps = list(range(interval, max_steps + 1, interval))
            if not evaluation_steps or evaluation_steps[-1] != max_steps:
                evaluation_steps.append(max_steps)
            for step in evaluation_steps:
                if step > engine.state.global_step:
                    continue
                record = _load_evaluation_record(
                    request, self.workspace_root, global_step=step
                )
                if record is None:
                    if step < engine.state.global_step:
                        raise RuntimeError(
                            f"EVALUATION_COMMIT_MISSING_BEFORE_RESUME_POINT:{step}"
                        )
                    record = _run_evaluation(
                        request,
                        self.workspace_root,
                        resources,
                        global_step=step,
                        rank=rank,
                    )
                evaluation_records.append(record)
        else:
            evaluation_steps = []

        profiling_enabled = bool(profiling["enabled"])
        profile_windows: list[tuple[int, int, int, Path]] = []
        resource_profiles: list[Mapping[str, JSONValue]] = []
        requested_profile: dict[str, JSONValue] = {
            "capture_memory": bool(profiling["capture_memory"]),
            "capture_throughput": bool(profiling["capture_throughput"]),
            "capture_communication": bool(profiling["capture_communication"]),
            "synchronize_device": bool(profiling["synchronize_device"]),
        }
        if profiling_enabled:
            warmup = int(profiling["warmup_steps"])
            measure = int(profiling["measure_steps"])
            repetitions = int(profiling["repetitions"])
            for repetition in range(repetitions):
                start = warmup + repetition * measure
                end = start + measure
                commit_path = _resource_window_path(
                    output_root, rank=rank, repetition=repetition
                )
                profile_windows.append((repetition, start, end, commit_path))
                if engine.state.global_step >= end:
                    if not commit_path.exists():
                        raise RuntimeError(
                            "TRAINING_RESOURCE_WINDOW_COMMIT_MISSING_BEFORE_RESUME_POINT:"
                            f"{repetition}"
                        )
                    loaded_profile = _load_resource_window_commit(
                        commit_path, workspace_root=self.workspace_root
                    )
                    if (
                        loaded_profile.get("repetition") != repetition
                        or loaded_profile.get("start_step") != start
                        or loaded_profile.get("end_step") != end
                        or loaded_profile.get("requested") != requested_profile
                    ):
                        raise ValueError("TRAINING_RESOURCE_WINDOW_IDENTITY_DRIFT")
                    resource_profiles.append(loaded_profile)
                elif start < engine.state.global_step < end:
                    # 墙钟时间不能跨进程续接。用户必须显式选择窗口起点处的
                    # checkpoint；模型恢复能力不受影响，但 profiling 证据拒绝伪造。
                    raise RuntimeError(
                        "TRAINING_PROFILE_RESUME_MUST_USE_WINDOW_BOUNDARY:"
                        f"step={engine.state.global_step}:window={start}-{end}"
                    )

        starts = {start: item for item in profile_windows for start in (item[1],)}
        ends = {end: item for item in profile_windows for end in (item[2],)}
        boundaries = sorted(
            {
                max_steps,
                *evaluation_steps,
                *(item[1] for item in profile_windows),
                *(item[2] for item in profile_windows),
            }
        )
        result: TrainingRunResult | None = None
        sampler: ResourceSampler | None = None
        active_window: tuple[int, int, int, Path] | None = None

        def start_profile_at(step: int) -> None:
            nonlocal sampler, active_window
            window = starts.get(step)
            if window is None or step < engine.state.global_step:
                return
            if window[3].exists():
                return
            if sampler is not None:
                raise RuntimeError("TRAINING_RESOURCE_WINDOWS_OVERLAP")
            _synchronize_profile_device(
                runtime_device, enabled=bool(profiling["synchronize_device"])
            )
            sampler = ResourceSampler(capture_memory=bool(profiling["capture_memory"]))
            sampler.start()
            active_window = window

        def finish_profile_at(
            step: int,
            current_result: TrainingRunResult,
        ) -> None:
            nonlocal sampler, active_window
            window = ends.get(step)
            if window is None or window[3].exists():
                return
            if sampler is None or active_window != window:
                raise RuntimeError("TRAINING_RESOURCE_WINDOW_SAMPLER_STATE_INVALID")
            _synchronize_profile_device(
                runtime_device, enabled=bool(profiling["synchronize_device"])
            )
            effective_units = sum(
                record.effective_count
                for record in current_result.records
                if record.status == "COMMITTED" and window[1] < record.global_step <= window[2]
            )
            profile = sampler.stop(
                completed_steps=window[2] - window[1],
                effective_units=effective_units,
            )
            resource_profiles.append(
                _publish_resource_window(
                    window[3],
                    workspace_root=self.workspace_root,
                    repetition=window[0],
                    start_step=window[1],
                    end_step=window[2],
                    requested=requested_profile,
                    profile=profile,
                    capture_throughput=bool(profiling["capture_throughput"]),
                    capture_communication=bool(profiling["capture_communication"]),
                )
            )
            sampler = None
            active_window = None

        with JsonlEventSink(events_path) as sink:
            engine.event_sink = sink
            with _deterministic_algorithms(bool(training["deterministic_algorithms"])):
                # 统一的边界循环同时处理 evaluation 与 profiling，确保 validation
                # forward 永远不被计入训练吞吐窗口。
                start_profile_at(engine.state.global_step)
                for step in boundaries:
                    if step <= engine.state.global_step:
                        continue
                    result = engine.run(
                        resume=False,
                        until_step=step if step < max_steps else None,
                    )
                    if engine.state.global_step != step:
                        break
                    finish_profile_at(step, result)
                    if evaluation_enabled and step in evaluation_steps:
                        evaluation_records.append(
                            _run_evaluation(
                                request,
                                self.workspace_root,
                                resources,
                                global_step=step,
                                rank=rank,
                            )
                        )
                    start_profile_at(step)
                if result is None:
                    # checkpoint 已经处于 max_steps、但 task 级 commit 尚未发布。
                    result = engine.run(resume=False)
        if sampler is not None:
            raise RuntimeError("TRAINING_RESOURCE_WINDOW_DID_NOT_REACH_COMMIT_BOUNDARY")
        assert result is not None
        if result.status != "COMPLETE":
            raise RuntimeError(f"TRAINING_TASK_NOT_COMPLETE:{result.status}")
        close_cursor = getattr(training_cursor, "close", None)
        if callable(close_cursor):
            close_cursor()
        return (
            result,
            engine,
            events_path,
            resources.asset_evidence,
            tuple(evaluation_records),
            tuple(sorted(resource_profiles, key=lambda item: int(item["repetition"]))),
        )

    def _result_payloads(
        self,
        request: TaskExecutionRequest,
        result: TrainingRunResult,
        engine: TrainingEngine,
        events_path: Path,
        asset_evidence: tuple[Mapping[str, JSONValue], ...],
        evaluation_records: tuple[Mapping[str, JSONValue], ...],
        resource_profiles: tuple[Mapping[str, JSONValue], ...],
    ) -> Mapping[str, Mapping[str, JSONValue]]:
        output: dict[str, Mapping[str, JSONValue]] = {}
        snapshot = None if result.importance_snapshot is None else result.importance_snapshot.to_dict()
        tensor_bundle_ref: str | None = None
        tensor_bundle_hash: str | None = None
        if engine.tracker is not None:
            artifacts = request.config.section("artifacts")
            assert isinstance(artifacts, dict)
            output_root = _resolve_workspace_path(
                self.workspace_root, str(artifacts["output_dir"]), field="output_dir"
            )
            bundle_path = output_root / "tensor-bundles" / "importance-final"
            if bundle_path.exists():
                _, identity = load_tensor_bundle(bundle_path)
            else:
                identity = publish_tensor_bundle(
                    bundle_path,
                    {
                        "metadata": {
                            "task_id": request.task.task_id,
                            "config_hash": request.config.config_hash,
                            "scope": request.config.run_intent,
                        },
                        "accumulator": engine.tracker.accumulator.state_dict(),
                    },
                )
            tensor_bundle_ref = bundle_path.relative_to(self.workspace_root).as_posix()
            tensor_bundle_hash = identity.manifest_sha256
        event_files = sorted(events_path.parent.glob(f"rank-{engine.rank:04d}-session-*.jsonl"))
        event_streams: list[JSONValue] = [
            {
                "ref": path.relative_to(self.workspace_root).as_posix(),
                "semantic_sha256": _semantic_event_stream_hash(path),
            }
            for path in event_files
        ]
        checkpoint_commits: list[JSONValue] = []
        if engine.checkpoint_store is not None:
            # task-output 必须直接绑定物理 checkpoint commit；仅记录逻辑 ID 无法让
            # Stage 7 在 fresh process 中证明自己加载的是训练产生的同一对象。
            for checkpoint_id in dict.fromkeys(result.checkpoint_ids):
                commit_path = engine.checkpoint_store.commits / f"{checkpoint_id}.json"
                resolved = commit_path.resolve()
                try:
                    reference = resolved.relative_to(self.workspace_root).as_posix()
                except ValueError as error:  # pragma: no cover - store 构造已限制在 workspace
                    raise ValueError("TRAINING_CHECKPOINT_COMMIT_ESCAPES_WORKSPACE") from error
                # load 会同时复核 commit 字段、对象 manifest 和张量文件。commit 的
                # ``committed_at`` 是运维诊断，不属于恢复身份；功能 artifact 只绑定
                # 去除该字段后的语义 hash 及 tensor bundle manifest hash。
                _state, verified_commit = engine.checkpoint_store.load(checkpoint_id)
                commit_value = load_canonical_json(resolved)
                if not isinstance(commit_value, dict):
                    raise ValueError("TRAINING_CHECKPOINT_COMMIT_NOT_OBJECT")
                identity = {
                    key: value
                    for key, value in commit_value.items()
                    if key != "committed_at"
                }
                checkpoint_commits.append(
                    {
                        "checkpoint_id": checkpoint_id,
                        "commit_ref": reference,
                        "commit_identity_sha256": canonical_json_hash(identity),
                        "bundle_manifest_sha256": verified_commit.manifest_sha256,
                    }
                )
        common: dict[str, JSONValue] = {
            "schema_version": "training-task-output-v1",
            "training_result": result.to_dict(),
            "event_stream_ref": events_path.relative_to(self.workspace_root).as_posix(),
            "event_stream_semantic_sha256": _semantic_event_stream_hash(events_path),
            "event_streams": event_streams,
            "checkpoint_commits": checkpoint_commits,
            "importance_snapshot": snapshot,
            "importance_tensor_bundle_ref": tensor_bundle_ref,
            "importance_tensor_bundle_hash": tensor_bundle_hash,
            "asset_evidence": [dict(item) for item in asset_evidence],
            "evaluation_records": [dict(item) for item in evaluation_records],
            "resource_profiles": [dict(item) for item in resource_profiles],
            "endpoint_bundles": [
                bundle.to_dict()
                for observer in engine.observers
                if isinstance(observer, TrainingEndpointObserver)
                for bundle in observer.bundles
            ],
            "gate_status": "NOT_RUN",
            "local_validation_status": (
                "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
            ),
        }
        for kind in request.task.artifact_kinds:
            output[kind] = {**common, "artifact_role": kind}
        return output

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        store = _artifact_store(request, self.workspace_root)
        existing = _completed_result(request, store)
        if existing is not None:
            return existing
        launcher = request.config.section("launcher")
        assert isinstance(launcher, dict)
        if int(launcher["world_size"]) != 1:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    "distributed_launcher",
                    "多 rank 配置必须使用 distributed_training runner",
                    False,
                )
            )
        try:
            (
                result,
                engine,
                events_path,
                assets,
                evaluations,
                resource_profiles,
            ) = self._run_training(request)
        except (AssetManifestError, FileNotFoundError) as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "offline_training_assets",
                    f"离线训练资产缺失或未通过 manifest 校验：{type(error).__name__}",
                    True,
                )
            ) from error
        payloads = self._result_payloads(
            request,
            result,
            engine,
            events_path,
            assets,
            evaluations,
            resource_profiles,
        )
        _, source_refs = _input_evidence(request, self.workspace_root)
        refs = _publish_payloads(request, store, payloads, source_refs=source_refs)
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            checkpoint_ref=_checkpoint_result_ref(refs),
            message="task completed",
            metadata={"execution_contract": "run-ready-v1"},
        )


@dataclass(slots=True)
class DistributedTrainingTaskRunner(TrainingTaskRunner):
    """torchrun/DDP 训练入口；world-size 1 则执行本机语义 fixture。"""

    runner_kind: RunnerKind = RunnerKind.DISTRIBUTED_TRAINING

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        launcher = request.config.section("launcher")
        assert isinstance(launcher, dict)
        world_size = int(launcher["world_size"])
        if world_size == 1:
            # 本机 fixture 仍运行真实训练，只是不声称验证了多 rank 通信。
            result = super().run(request)
            return result
        from ..runtime.distributed_training import TorchDDPTrainingExecutor
        from ..providers.training import TorchModelAdapter

        executor = TorchDDPTrainingExecutor.from_environment(backend=str(launcher["backend"]))
        if executor.spec.world_size != world_size:
            executor.close()
            raise ValueError("DISTRIBUTED_LAUNCHER_WORLD_SIZE_MISMATCH")
        rank = executor.spec.rank
        store = _artifact_store(request, self.workspace_root)
        try:
            if rank == 0:
                existing = _completed_result(request, store)
                if existing is not None:
                    executor.barrier()
                    return existing
            resources = _training_resources(
                request, self.workspace_root, rank=rank, world_size=world_size
            )
            wrapped = executor.wrap_model(resources.model.module)
            adapter = TorchModelAdapter(wrapped, task_type=resources.model.task_type)
            (
                result,
                engine,
                events_path,
                assets,
                evaluations,
                resource_profiles,
            ) = self._run_training(
                request,
                rank=rank,
                world_size=world_size,
                reducer=executor.reducer,
                resources=resources,
                wrapped_model=adapter,
            )
            executor.barrier()
            if rank == 0:
                payloads = self._result_payloads(
                    request,
                    result,
                    engine,
                    events_path,
                    assets,
                    evaluations,
                    resource_profiles,
                )
                # rank-0 结果绑定所有 rank 的事件和 checkpoint commit 集合。
                output_root = _resolve_workspace_path(
                    self.workspace_root,
                    str(request.config.section("artifacts")["output_dir"]),  # type: ignore[index]
                    field="output_dir",
                )
                rank_event_refs = [
                    path.relative_to(self.workspace_root).as_posix()
                    for item in range(world_size)
                    for path in sorted(
                        (output_root / "events").glob(
                            f"rank-{item:04d}-session-*.jsonl"
                        )
                    )
                ]
                for payload in payloads.values():
                    assert isinstance(payload, dict)
                    payload["distributed"] = {
                        "backend": launcher["backend"],
                        "world_size": world_size,
                        "rank_event_refs": rank_event_refs,
                    }
                _, source_refs = _input_evidence(request, self.workspace_root)
                refs = _publish_payloads(
                    request, store, payloads, source_refs=source_refs
                )
            executor.barrier()
            if rank != 0:
                discovered = store.discover_complete(
                    task_id=request.task.task_id,
                    config_hash=request.config.config_hash,
                    artifact_kinds=request.task.artifact_kinds,
                    formal_eligible=request.config.run_intent == "formal",
                )
                if discovered is None:
                    raise RuntimeError("DISTRIBUTED_RANK0_ARTIFACTS_MISSING")
                refs = discovered
            return TaskRunResult.passed(
                request,
                artifact_refs=refs,
                checkpoint_ref=_checkpoint_result_ref(refs),
                message="task completed",
                metadata={"execution_contract": "run-ready-v1"},
            )
        except (AssetManifestError, FileNotFoundError) as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "offline_training_assets",
                    f"离线分布式训练资产缺失或未通过 manifest 校验：{type(error).__name__}",
                    True,
                )
            ) from error
        finally:
            executor.close()


@dataclass(slots=True)
class RouteTrainingTaskRunner(TrainingTaskRunner):
    """Stage 4--6 route task 的执行入口。

    当前 task 原子单元执行一条 phase；route DAG 的父子约束由配置中的冻结
    ``route_spec_ref`` 和输入结果 commit 共同绑定。多 phase 路线可由 task runtime
    逐 phase resume，不需要临时训练脚本。
    """

    runner_kind: RunnerKind = RunnerKind.ROUTE_TRAINING


def build_default_task_runtime(
    workspace_root: str | Path | None = None,
    *,
    catalog=DEFAULT_TASK_CATALOG,
) -> TaskRuntime:
    """构造覆盖目录中全部 ``RunnerKind`` 的默认运行时。"""

    root = Path.cwd().resolve() if workspace_root is None else Path(workspace_root).resolve()
    runtime = TaskRuntime(catalog=catalog, workspace_root=root)
    # 先为每种 kind 构造可委托的保底 runner，再逐层组合专用实现。TaskRuntime
    # 禁止“先注册再覆盖”，因此最终只把组合完成的唯一实例注册进去。
    runners: dict[RunnerKind, TaskRunner] = {
        kind: CatalogTaskRunner(kind, root) for kind in RunnerKind
    }
    runners[RunnerKind.TRAINING] = TrainingTaskRunner(root)
    runners[RunnerKind.DISTRIBUTED_TRAINING] = DistributedTrainingTaskRunner(root)
    runners[RunnerKind.ROUTE_TRAINING] = RouteTrainingTaskRunner(root)

    from .stage01_task_runners import build_stage01_runner_overrides
    from .stage23_task_runners import build_stage23_runner_overrides
    from .stage456_task_runners import build_stage456_runner_overrides
    from .stage789_task_runners import build_stage789_runner_overrides

    runners.update(build_stage01_runner_overrides(root, fallbacks=runners))
    runners.update(build_stage23_runner_overrides(root, fallbacks=runners))
    # Stage 4--6 和 Stage 7--9 共享 STATISTICS/ANALYSIS/REPORTING 等 kind。
    # 后一层只处理自己的 task_id，未命中时沿 fallback 链回到前一层。
    runners.update(build_stage456_runner_overrides(root, fallbacks=runners))
    runners.update(build_stage789_runner_overrides(root, fallbacks=runners))
    for kind in RunnerKind:
        runtime.register(runners[kind])
    return runtime


# 公共合同别名：统一任务 runner 的协议就是 StageRunner。
StageRunner = TaskRunner


__all__ = [
    "CatalogTaskRunner",
    "DistributedTrainingTaskRunner",
    "RouteTrainingTaskRunner",
    "StageRunner",
    "TrainingTaskRunner",
    "build_default_task_runtime",
]
