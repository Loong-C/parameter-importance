"""Stage 4--6 训练路线与派生任务的专用 task runner 适配层。

本模块把三个已经冻结的执行边界连成一条完整链路：

``TrainingRouteSpec -> TrainingRouteRunner -> TrainingEngine``

其中路线规范负责科学 lineage，phase runner 负责逐 phase 的不可变结果提交，训练
引擎负责 optimizer-step/checkpoint 的事务恢复。任务级输出最后再由
``TaskArtifactStore`` 以内容寻址对象加独立 commit 发布。因此进程无论在 step、phase
还是任务产物边界退出，fresh process 都只会从最近一个权威 commit 继续。

正式运行有两道额外防线：

* 在线重要性只能消费与路线内嵌对象逐字段一致、且被 Stage 2 Gate 证据引用的
  ``EstimatorDecision``；
* 路径审计只能消费 ``QUALIFIED`` 的 Stage 3 quadrature recommendation，并要求
  独立 Gate 的 hash 和 evidence ref 同时匹配。

本机 fixture 使用 tiny Torch 模型真正执行 pretrain/direct/finetune DAG，但其所有
产物始终是 ``formal_eligible=false``。本模块不访问网络、不猜测服务器路径，也不
把缺失资产降级成 synthetic formal 结果。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import fmean, stdev
from types import MappingProxyType
from typing import Mapping, Sequence

import torch

from ..assets import resolve_ready_asset
from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
)
from ..contracts.seed import SeedPlan
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import RunnerKind
from ..core.pruning import PruningContext, PruningPlan, select_pruned_coordinates
from ..core.tensors import TensorMap
from ..providers.tiny import TinyTrainingFixture, build_tiny_training_fixture
from ..providers.training import (
    CausalLMEvaluator,
    ClassificationEvaluator,
    OfflineHuggingFaceModelAdapter,
    PretokenizedJsonlDatasetAdapter,
    TorchModelAdapter,
    TrainingMicrobatch,
    configure_batch_cursor,
)
from ..runtime.checkpoint import CheckpointStore
from ..runtime.events import JsonlEventSink
from ..runtime.task_artifacts import TaskArtifactStore
from ..runtime.task_runtime import (
    BlockerCode,
    TaskBlockedError,
    TaskBlocker,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunner,
)
from ..runtime.telemetry import (
    ResourceSampler,
    ResourceWindowStore,
    synchronize_profile_device,
)
from ..runtime.training import (
    TrainingEngine,
    TrainingRunResult,
    TrainingRunSpec,
    TrainingState,
    install_training_rng,
)
from ..runtime.training_factory import build_grad_scaler, build_optimizer, build_scheduler
from .routes import TrainingPhaseSpec, TrainingRouteSpec, validate_comparable_routes
from .stage2 import EstimatorDecision
from .stage3 import QuadratureDecision
from .stage6_evaluation import (
    audit_with_hash as _stage6_audit_with_hash,
    build_route_matrix as _build_stage6_route_matrix,
    compare_paired_metrics as _compare_stage6_paired_metrics,
    evidences_from_matrix as _stage6_evidences_from_matrix,
    importance_reuse_tables as _stage6_importance_reuse_tables,
    paired_route_evaluations as _stage6_paired_route_evaluations,
    route_matrix_from_inputs as _stage6_route_matrix_from_inputs,
    unique_role_data as _stage6_unique_role_data,
    unique_route_audit as _stage6_unique_route_audit,
    verify_route_evaluation_checkpoints as _stage6_verify_checkpoints,
)
from .training_routes import (
    TrainingPhaseRuntime,
    TrainingRouteResult,
    TrainingRouteRunner,
)


_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ROUTE_TASK_IDS = frozenset(
    {
        "stage4.minimal_complete_loop",
        "stage4.pretrain",
        "stage4.direct_supervised",
        "stage4.finetune",
        "stage5.formal_pretraining",
        "stage5.pretrain",
        "stage6.training_route_comparison",
    }
)
_TASK_IDS_BY_KIND: Mapping[RunnerKind, frozenset[str]] = MappingProxyType(
    {
        RunnerKind.ROUTE_TRAINING: _ROUTE_TASK_IDS,
        RunnerKind.CONTRACT: frozenset({"stage4.route", "stage6.route_matrix"}),
        RunnerKind.STATISTICS: frozenset(
            {
                "stage4.importance_trajectory",
                "stage5.importance_trajectory",
                "stage6.compare",
            }
        ),
        RunnerKind.VALIDATION: frozenset({"stage6.evaluate"}),
        RunnerKind.ANALYSIS: frozenset(
            {"stage5.checkpoint_analysis", "stage6.importance_reuse"}
        ),
        RunnerKind.REPORTING: frozenset(
            {"stage4.report", "stage5.report", "stage6.report"}
        ),
    }
)


def _logical_path(value: object, *, field_name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE456_LOGICAL_PATH_INVALID:{field_name}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"STAGE456_PATH_ESCAPE:{field_name}")
    return path


def _resolve(root: Path, value: object, *, field_name: str) -> Path:
    logical = _logical_path(value, field_name=field_name)
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"STAGE456_PATH_ESCAPE:{field_name}") from error
    return candidate


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"STAGE456_SHA256_INVALID:{field_name}")
    return value


def _tensor_tree_hash(values: Mapping[str, object]) -> str:
    """计算不依赖 device 的 tensor mapping 摘要，用于实际初始化一致性检查。"""

    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"STAGE456_STATE_VALUE_NOT_TENSOR:{name}")
        tensor = value.detach().to(device="cpu").contiguous()
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _ResolvedJson:
    """已经验证 object/commit 关系的本地 JSON 输入。"""

    ref: str
    source_hash: str
    value: Mapping[str, object]


def _load_json_input(root: Path, reference: str) -> _ResolvedJson:
    """严格读取普通 artifact 或 ``task-output-commit-v1``。

    task commit 的 ``object_ref`` 会再次做 workspace 边界检查，并重新计算对象内的
    ``artifact_hash``；因此调用方拿到的不是“一个存在的 JSON 文件”，而是一份可
    消费的权威输入。
    """

    logical = _logical_path(reference, field_name="input_ref").as_posix()
    path = _resolve(root, logical, field_name="input_ref")
    if not path.is_file():
        raise FileNotFoundError(f"STAGE456_INPUT_NOT_FOUND:{logical}")
    value = load_canonical_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"STAGE456_INPUT_ROOT_NOT_OBJECT:{logical}")
    if value.get("schema_version") != "task-output-commit-v1":
        declared = value.get("artifact_hash")
        if declared is not None:
            payload = dict(value)
            payload.pop("artifact_hash")
            if declared != canonical_json_hash(payload):
                raise ValueError(f"STAGE456_INPUT_ARTIFACT_HASH_MISMATCH:{logical}")
            digest = _require_hash(declared, field_name="artifact_hash")
        else:
            digest = canonical_json_hash(value)
        return _ResolvedJson(logical, digest, value)

    expected = {
        "schema_version",
        "task_id",
        "artifact_kind",
        "config_hash",
        "artifact_hash",
        "object_ref",
        "formal_eligible",
    }
    if set(value) != expected:
        raise ValueError("STAGE456_TASK_COMMIT_FIELDS_MISMATCH")
    object_ref = _logical_path(value["object_ref"], field_name="object_ref").as_posix()
    object_path = _resolve(root, object_ref, field_name="object_ref")
    object_value = load_canonical_json(object_path)
    if not isinstance(object_value, dict):
        raise ValueError("STAGE456_TASK_OBJECT_ROOT_NOT_OBJECT")
    object_payload = dict(object_value)
    object_hash = object_payload.pop("artifact_hash", None)
    if object_hash != canonical_json_hash(object_payload):
        raise ValueError("STAGE456_TASK_OBJECT_HASH_MISMATCH")
    for field_name in (
        "task_id",
        "artifact_kind",
        "config_hash",
        "formal_eligible",
        "artifact_hash",
    ):
        if value[field_name] != object_value.get(field_name):
            raise ValueError(f"STAGE456_TASK_COMMIT_OBJECT_MISMATCH:{field_name}")
    return _ResolvedJson(logical, str(object_hash), object_value)


def _unwrap_payload(value: Mapping[str, object]) -> Mapping[str, object]:
    if value.get("schema_version") != "task-output-artifact-v1":
        return value
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("STAGE456_TASK_OUTPUT_PAYLOAD_NOT_OBJECT")
    return payload


def _validate_input_scope(value: Mapping[str, object], *, run_intent: str) -> None:
    """阻止 fixture/formal 输入在派生链路中相互冒充。"""

    expected_formal = run_intent == "formal"
    if value.get("schema_version") == "task-output-artifact-v1":
        if value.get("formal_eligible") is not expected_formal:
            raise ValueError("STAGE456_INPUT_TASK_ARTIFACT_SCOPE_MISMATCH")
    payload = _unwrap_payload(value)
    if isinstance(payload.get("formal_eligible"), bool):
        if payload["formal_eligible"] is not expected_formal:
            raise ValueError("STAGE456_INPUT_FORMAL_ELIGIBILITY_MISMATCH")
        return
    if isinstance(payload.get("scope"), str):
        if payload["scope"] != run_intent:
            raise ValueError("STAGE456_INPUT_SCOPE_MISMATCH")
        return
    if payload.get("schema_version") == "training-route-v1":
        if payload.get("run_intent") != run_intent:
            raise ValueError("STAGE456_INPUT_ROUTE_SCOPE_MISMATCH")
        return
    # TaskArtifact 外壳已经提供资格字段；直接 JSON 则必须自行声明 scope，避免
    # formal 派生任务把一个只有 hash、没有资格语义的手工表当作正式输入。
    if value.get("schema_version") != "task-output-artifact-v1":
        raise ValueError("STAGE456_DIRECT_INPUT_MISSING_SCOPE_QUALIFICATION")


def _route_candidate(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = _unwrap_payload(value)
    if payload.get("schema_version") == "training-route-v1":
        return payload
    for field_name in ("route_spec", "training_route"):
        candidate = payload.get(field_name)
        if isinstance(candidate, Mapping) and candidate.get("schema_version") == "training-route-v1":
            return candidate
    raise ValueError("STAGE456_INPUT_DOES_NOT_CONTAIN_TRAINING_ROUTE")


def _blocked_missing(
    code: BlockerCode,
    requirement: str,
    message: str,
    *,
    evidence_refs: tuple[str, ...] = (),
) -> TaskBlockedError:
    return TaskBlockedError(
        TaskBlocker(code, requirement, message, True, evidence_refs)
    )


def _load_route(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[TrainingRouteSpec, str, str]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    reference = orchestration["route_spec_ref"]
    if reference is None:
        raise _blocked_missing(
            BlockerCode.CONTRACT_UNFROZEN,
            "route_spec_ref",
            "Stage 4--6 路线任务缺少 hash-bound TrainingRouteSpec",
        )
    try:
        resolved = _load_json_input(root, str(reference))
        route = TrainingRouteSpec.from_mapping(_route_candidate(resolved.value))
    except FileNotFoundError as error:
        raise _blocked_missing(
            BlockerCode.ASSET_UNAVAILABLE,
            "route_spec_ref",
            str(error),
            evidence_refs=(str(reference),),
        ) from error
    if route.run_intent != request.config.run_intent:
        raise ValueError("STAGE456_ROUTE_RUN_INTENT_MISMATCH")
    _validate_route_shape_for_task(request.task.task_id, route)
    return route, str(reference), resolved.source_hash


def _validate_route_shape_for_task(task_id: str, route: TrainingRouteSpec) -> None:
    types = {phase.phase_type for phase in route.phases}
    required: set[str]
    if task_id in {"stage4.pretrain", "stage5.pretrain", "stage5.formal_pretraining"}:
        required = {"pretrain"}
    elif task_id == "stage4.direct_supervised":
        required = {"direct_supervised"}
    elif task_id == "stage4.finetune":
        required = {"pretrain", "finetune"}
    elif task_id in {
        "stage4.minimal_complete_loop",
        "stage6.training_route_comparison",
    }:
        required = {"pretrain", "direct_supervised", "finetune"}
    else:
        return
    missing = required - types
    if missing:
        raise ValueError(f"STAGE456_ROUTE_PHASE_TYPES_MISSING:{sorted(missing)}")


def _extract_direct_mapping(
    value: Mapping[str, object],
    *,
    schema_version: str,
    field_names: Sequence[str],
) -> Mapping[str, object]:
    payload = _unwrap_payload(value)
    if payload.get("schema_version") == schema_version:
        return payload
    for name in field_names:
        candidate = payload.get(name)
        if isinstance(candidate, Mapping) and candidate.get("schema_version") == schema_version:
            return candidate
    raise ValueError(f"STAGE456_INPUT_SCHEMA_NOT_FOUND:{schema_version}")


def _validate_estimator_binding(
    request: TaskExecutionRequest,
    route: TrainingRouteSpec,
    root: Path,
) -> Mapping[str, JSONValue]:
    enabled = any(phase.importance_enabled for phase in route.phases)
    if not enabled:
        return {
            "enabled": False,
            "decision_hash": None,
            "gate_hash": None,
            "qualified": False,
        }
    decision = route.estimator_decision
    if not isinstance(decision, EstimatorDecision):
        raise ValueError("STAGE456_ROUTE_ESTIMATOR_DECISION_NOT_NORMALIZED")
    if request.config.run_intent == "local_fixture":
        return {
            "enabled": True,
            "decision_hash": decision.artifact_hash,
            "gate_hash": None,
            "qualified": False,
        }

    reference = request.environment.estimator_decision_ref
    if reference is None:
        raise _blocked_missing(
            BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
            "estimator_decision",
            "formal 路线缺少独立 EstimatorDecision 引用",
        )
    try:
        resolved = _load_json_input(root, reference)
        direct = _extract_direct_mapping(
            resolved.value,
            schema_version="estimator-decision-v1",
            field_names=("estimator_decision", "decision"),
        )
        external = EstimatorDecision.from_mapping(direct)
    except FileNotFoundError as error:
        raise _blocked_missing(
            BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE,
            "estimator_decision",
            str(error),
            evidence_refs=(reference,),
        ) from error
    if external.to_dict() != decision.to_dict():
        raise ValueError("STAGE456_ROUTE_AND_EXTERNAL_ESTIMATOR_DECISION_MISMATCH")
    if decision.artifact_ref != reference:
        raise ValueError("STAGE456_ESTIMATOR_DECISION_REF_MISMATCH")
    gate = route.estimator_gate
    if not isinstance(gate, GateRecord):
        raise ValueError("STAGE456_FORMAL_ESTIMATOR_GATE_MISSING")
    if gate.gate_id not in request.environment.passed_gate_ids:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            gate.gate_id,
            "Stage 2 estimator Gate 未出现在已验证环境快照中",
            evidence_refs=(reference,),
        )
    if reference not in gate.evidence_refs:
        raise ValueError("STAGE456_ESTIMATOR_GATE_DOES_NOT_BIND_DECISION_REF")
    if gate.effective_status() not in {
        GateStatus.PASS,
        GateStatus.CONDITIONALLY_ACCEPTED,
    }:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            gate.gate_id,
            "Stage 2 estimator Gate 当前不可接受或已经过期",
            evidence_refs=(reference,),
        )
    return {
        "enabled": True,
        "decision_hash": decision.artifact_hash,
        "decision_source_hash": resolved.source_hash,
        "gate_hash": gate.artifact_hash,
        "gate_id": gate.gate_id,
        "qualified": True,
    }


def _validate_quadrature_binding(
    request: TaskExecutionRequest,
    root: Path,
) -> Mapping[str, JSONValue]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    reference = orchestration["quadrature_decision_ref"]
    if reference is None:
        if request.config.run_intent == "formal":
            raise _blocked_missing(
                BlockerCode.CONTRACT_UNFROZEN,
                "quadrature_decision",
                "formal Stage 4--6 路线缺少 Stage 3 quadrature recommendation",
            )
        return {
            "enabled": False,
            "decision_hash": None,
            "qualified": False,
        }
    try:
        resolved = _load_json_input(root, str(reference))
    except FileNotFoundError as error:
        raise _blocked_missing(
            BlockerCode.ASSET_UNAVAILABLE,
            "quadrature_decision",
            str(error),
            evidence_refs=(str(reference),),
        ) from error
    payload = _unwrap_payload(resolved.value)
    direct: Mapping[str, object]
    if payload.get("schema_version") in {
        "quadrature-decision-v1",
        "stage3-quadrature-recommendation-v1",
    }:
        direct = payload
    else:
        candidate = payload.get("quadrature_decision") or payload.get(
            "quadrature_recommendation"
        )
        if not isinstance(candidate, Mapping):
            raise ValueError("STAGE456_QUADRATURE_DECISION_NOT_FOUND")
        direct = candidate

    if direct.get("schema_version") == "quadrature-decision-v1":
        decision = QuadratureDecision.from_mapping(direct)
        if request.config.run_intent == "formal":
            raise _blocked_missing(
                BlockerCode.CONTRACT_UNFROZEN,
                "quadrature_decision",
                "fixture QuadratureDecision 不能解锁 formal 路线",
                evidence_refs=(str(reference),),
            )
        return {
            "enabled": True,
            "decision_hash": decision.artifact_hash,
            "default_rule": decision.default_rule,
            "qualified": False,
        }

    expected = {
        "schema_version",
        "recommendation_id",
        "status",
        "default_rule",
        "fallback_rule",
        "passing_rules",
        "required_unit_ids",
        "thresholds",
        "thresholds_hash",
        "execution_evidence_hash",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "reasons",
        "artifact_hash",
    }
    if set(direct) != expected:
        raise ValueError("STAGE456_QUADRATURE_RECOMMENDATION_FIELDS_MISMATCH")
    body = dict(direct)
    artifact_hash = body.pop("artifact_hash")
    if artifact_hash != canonical_json_hash(body):
        raise ValueError("STAGE456_QUADRATURE_RECOMMENDATION_HASH_MISMATCH")
    if request.config.run_intent == "local_fixture":
        if direct["scope"] != "local_fixture" or direct["formal_eligible"] is not False:
            raise ValueError("STAGE456_FIXTURE_QUADRATURE_SCOPE_MISMATCH")
        return {
            "enabled": True,
            "decision_hash": str(artifact_hash),
            "default_rule": direct["default_rule"],  # type: ignore[dict-item]
            "qualified": False,
        }

    if (
        direct["scope"] != "formal"
        or direct["formal_eligible"] is not True
        or direct["status"] != "QUALIFIED"
    ):
        raise _blocked_missing(
            BlockerCode.CONTRACT_UNFROZEN,
            "quadrature_decision",
            "Stage 3 recommendation 尚未被正式 Gate 资格化",
            evidence_refs=(str(reference),),
        )
    gate_hash = _require_hash(
        direct["qualification_gate_hash"], field_name="qualification_gate_hash"
    )
    gate_reference = request.environment.evidence_refs.get("quadrature_gate")
    if gate_reference is None:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            "quadrature_gate",
            "formal 路线缺少独立 Stage 3 GateRecord 引用",
            evidence_refs=(str(reference),),
        )
    try:
        gate_input = _load_json_input(root, gate_reference)
        gate_mapping = _extract_direct_mapping(
            gate_input.value,
            schema_version="gate-record-v1",
            field_names=("gate_record", "gate"),
        )
        gate = GateRecord.from_mapping(dict(gate_mapping))
    except FileNotFoundError as error:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            "quadrature_gate",
            str(error),
            evidence_refs=(gate_reference,),
        ) from error
    if gate.stage != 3 or gate.artifact_hash != gate_hash:
        raise ValueError("STAGE456_QUADRATURE_GATE_HASH_OR_STAGE_MISMATCH")
    if gate.gate_id not in request.environment.passed_gate_ids:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            gate.gate_id,
            "Stage 3 Gate 未出现在已验证环境快照中",
            evidence_refs=(gate_reference,),
        )
    if str(reference) not in gate.evidence_refs:
        raise ValueError("STAGE456_QUADRATURE_GATE_DOES_NOT_BIND_RECOMMENDATION")
    if gate.effective_status() not in {
        GateStatus.PASS,
        GateStatus.CONDITIONALLY_ACCEPTED,
    }:
        raise _blocked_missing(
            BlockerCode.GATE_NOT_READY,
            gate.gate_id,
            "Stage 3 quadrature Gate 当前不可接受或已经过期",
            evidence_refs=(gate_reference,),
        )
    return {
        "enabled": True,
        "decision_hash": str(artifact_hash),
        "default_rule": direct["default_rule"],  # type: ignore[dict-item]
        "gate_hash": gate.artifact_hash,
        "gate_id": gate.gate_id,
        "qualified": True,
    }


@dataclass(frozen=True, slots=True)
class _PhaseOptions:
    """从 route metadata 解析出的少量、严格 phase 运行差异。"""

    max_steps: int
    task_type: str
    num_labels: int
    data_seed_offset: int
    learning_rate: float | None
    training_split: str
    evaluation_split: str

    @classmethod
    def from_request(
        cls,
        request: TaskExecutionRequest,
        phase: TrainingPhaseSpec,
    ) -> "_PhaseOptions":
        training = request.config.section("training")
        providers = request.config.section("providers")
        data = request.config.base_config.section("data")
        assert isinstance(training, dict) and isinstance(providers, dict)
        raw = phase.metadata.get("runtime", {})
        if not isinstance(raw, Mapping):
            raise TypeError("STAGE456_PHASE_RUNTIME_METADATA_NOT_OBJECT")
        allowed = {
            "max_steps",
            "task_type",
            "num_labels",
            "data_seed_offset",
            "learning_rate",
            "training_split",
            "evaluation_split",
        }
        if set(raw) - allowed:
            raise ValueError(
                f"STAGE456_PHASE_RUNTIME_UNKNOWN_FIELDS:{sorted(set(raw)-allowed)}"
            )
        max_steps = raw.get("max_steps", training["max_steps"])
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise _blocked_missing(
                BlockerCode.CAPABILITY_UNAVAILABLE,
                "phase_max_steps",
                "路线执行必须把 epoch/资产长度解析为严格正整数 max_steps",
            )
        task_type = raw.get("task_type", providers["task_type"])
        if task_type not in {"causal_lm", "sequence_classification"}:
            raise ValueError("STAGE456_PHASE_TASK_TYPE_UNSUPPORTED")
        num_labels = raw.get("num_labels", providers["num_labels"] or 3)
        if isinstance(num_labels, bool) or not isinstance(num_labels, int) or num_labels < 2:
            raise ValueError("STAGE456_PHASE_NUM_LABELS_INVALID")
        seed_offset = raw.get("data_seed_offset", 0)
        if isinstance(seed_offset, bool) or not isinstance(seed_offset, int):
            raise ValueError("STAGE456_PHASE_SEED_OFFSET_INVALID")
        learning_rate = raw.get("learning_rate")
        if learning_rate is not None and (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(float(learning_rate))
            or float(learning_rate) <= 0
        ):
            raise ValueError("STAGE456_PHASE_LEARNING_RATE_INVALID")
        training_split = raw.get("training_split", data["split"])
        evaluation = request.config.section("evaluation")
        assert isinstance(evaluation, dict)
        evaluation_split = raw.get(
            "evaluation_split", evaluation["split"] or training_split
        )
        if not isinstance(training_split, str) or not training_split:
            raise ValueError("STAGE456_PHASE_TRAINING_SPLIT_INVALID")
        if not isinstance(evaluation_split, str) or not evaluation_split:
            raise ValueError("STAGE456_PHASE_EVALUATION_SPLIT_INVALID")
        return cls(
            max_steps,
            str(task_type),
            num_labels,
            seed_offset,
            None if learning_rate is None else float(learning_rate),
            training_split,
            evaluation_split,
        )


@dataclass(frozen=True, slots=True)
class _PhaseResources:
    model: TorchModelAdapter
    dataset: object
    options: _PhaseOptions
    evidence: tuple[Mapping[str, JSONValue], ...]


class _RouteTrainingEngine:
    """为一个 route phase 增加确定性边界与可恢复资源测量。

    ``TrainingRouteRunner`` 只要求 phase engine 暴露 ``run()`` 并返回
    ``TrainingRunResult``。这里保留真正的 ``TrainingEngine`` 作为唯一状态机，
    仅把一次完整运行切在 profiling 窗口边界；因此 optimizer、scheduler、数据
    游标和重要性事务语义不会复制到第二套实现。

    墙钟时间不能跨进程拼接。若恢复点落在窗口内部，执行器明确拒绝并要求选择
    窗口边界 checkpoint；已经越过的窗口必须存在独立权威 commit。
    """

    def __init__(
        self,
        engine: TrainingEngine,
        *,
        profiling: Mapping[str, object],
        store: ResourceWindowStore,
        runtime_device: str,
        deterministic_algorithms: bool,
        evaluation_steps: Sequence[int],
        published: list[Mapping[str, JSONValue]],
    ) -> None:
        self.engine = engine
        self.profiling = profiling
        self.store = store
        self.runtime_device = runtime_device
        self.deterministic_algorithms = deterministic_algorithms
        self.evaluation_steps = tuple(evaluation_steps)
        self.published = published

    @property
    def state(self) -> TrainingState:
        return self.engine.state

    def run(self) -> TrainingRunResult:
        """执行到 phase 终点，并在每个测量窗口完成两阶段提交。"""

        previous_determinism = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(self.deterministic_algorithms)
        try:
            return self._run_segmented()
        finally:
            torch.use_deterministic_algorithms(previous_determinism)

    def _run_segmented(self) -> TrainingRunResult:
        """统一处理 profiling/评估 checkpoint 边界。"""

        if not bool(self.profiling["enabled"]):
            result: TrainingRunResult | None = None
            for boundary in sorted({self.engine.spec.max_steps, *self.evaluation_steps}):
                if boundary <= self.engine.state.global_step:
                    continue
                result = self.engine.run(
                    resume=False,
                    until_step=(
                        boundary if boundary < self.engine.spec.max_steps else None
                    ),
                )
                if self.engine.state.global_step != boundary:
                    break
                self._checkpoint_evaluation_boundary(boundary)
            return result if result is not None else self.engine.run(resume=False)
        return self._run_profiled()

    def _checkpoint_evaluation_boundary(self, step: int) -> None:
        if step not in self.evaluation_steps:
            return
        expected = f"{self.engine.spec.run_id}-step-{step:08d}"
        if self.engine.state.last_checkpoint_id != expected:
            self.engine.save_checkpoint()

    def _run_profiled(self) -> TrainingRunResult:
        warmup = int(self.profiling["warmup_steps"])
        measure = int(self.profiling["measure_steps"])
        repetitions = int(self.profiling["repetitions"])
        maximum = self.engine.spec.max_steps
        if warmup + measure * repetitions > maximum:
            raise ValueError(
                "STAGE456_PROFILE_WINDOWS_EXCEED_PHASE_MAX_STEPS:"
                f"required={warmup + measure * repetitions}:max_steps={maximum}"
            )
        requested: dict[str, JSONValue] = {
            "capture_memory": bool(self.profiling["capture_memory"]),
            "capture_throughput": bool(self.profiling["capture_throughput"]),
            "capture_communication": bool(self.profiling["capture_communication"]),
            "synchronize_device": bool(self.profiling["synchronize_device"]),
        }
        windows: list[tuple[int, int, int]] = []
        self.published.clear()
        for repetition in range(repetitions):
            start = warmup + repetition * measure
            end = start + measure
            windows.append((repetition, start, end))
            commit_path = self.store.commit_path(repetition)
            if self.engine.state.global_step >= end:
                if not commit_path.exists():
                    raise RuntimeError(
                        "STAGE456_RESOURCE_WINDOW_COMMIT_MISSING_BEFORE_RESUME_POINT:"
                        f"{repetition}"
                    )
                loaded = self.store.load(repetition)
                if (
                    loaded.get("start_step") != start
                    or loaded.get("end_step") != end
                    or loaded.get("requested") != requested
                ):
                    raise ValueError("STAGE456_RESOURCE_WINDOW_IDENTITY_DRIFT")
                self.published.append(loaded)
            elif start < self.engine.state.global_step < end:
                raise RuntimeError(
                    "STAGE456_PROFILE_RESUME_MUST_USE_WINDOW_BOUNDARY:"
                    f"step={self.engine.state.global_step}:window={start}-{end}"
                )

        starts = {start: (repetition, start, end) for repetition, start, end in windows}
        ends = {end: (repetition, start, end) for repetition, start, end in windows}
        boundaries = sorted({maximum, *starts, *ends, *self.evaluation_steps})
        sampler: ResourceSampler | None = None
        active: tuple[int, int, int] | None = None
        result: TrainingRunResult | None = None

        def start_at(step: int) -> None:
            nonlocal sampler, active
            window = starts.get(step)
            if window is None or self.store.commit_path(window[0]).exists():
                return
            if sampler is not None:
                raise RuntimeError("STAGE456_RESOURCE_WINDOWS_OVERLAP")
            synchronize_profile_device(
                self.runtime_device,
                enabled=bool(self.profiling["synchronize_device"]),
            )
            sampler = ResourceSampler(
                capture_memory=bool(self.profiling["capture_memory"])
            )
            sampler.start()
            active = window

        def finish_at(step: int, current: TrainingRunResult) -> None:
            nonlocal sampler, active
            window = ends.get(step)
            if window is None or self.store.commit_path(window[0]).exists():
                return
            if sampler is None or active != window:
                raise RuntimeError("STAGE456_RESOURCE_WINDOW_SAMPLER_STATE_INVALID")
            synchronize_profile_device(
                self.runtime_device,
                enabled=bool(self.profiling["synchronize_device"]),
            )
            effective_units = sum(
                record.effective_count
                for record in current.records
                if record.status == "COMMITTED"
                and window[1] < record.global_step <= window[2]
            )
            profile = sampler.stop(
                completed_steps=window[2] - window[1],
                effective_units=effective_units,
            )
            self.published.append(
                self.store.publish(
                    repetition=window[0],
                    start_step=window[1],
                    end_step=window[2],
                    requested=requested,
                    profile=profile,
                    capture_throughput=bool(self.profiling["capture_throughput"]),
                    capture_communication=bool(
                        self.profiling["capture_communication"]
                    ),
                )
            )
            sampler = None
            active = None

        start_at(self.engine.state.global_step)
        for boundary in boundaries:
            if boundary <= self.engine.state.global_step:
                continue
            result = self.engine.run(
                resume=False,
                until_step=boundary if boundary < maximum else None,
            )
            if self.engine.state.global_step != boundary:
                break
            finish_at(boundary, result)
            self._checkpoint_evaluation_boundary(boundary)
            start_at(boundary)
        if result is None:
            result = self.engine.run(resume=False)
        if sampler is not None:
            raise RuntimeError("STAGE456_RESOURCE_WINDOW_DID_NOT_REACH_COMMIT_BOUNDARY")
        self.published.sort(key=lambda item: int(item["repetition"]))
        return result


class _RoutePhaseBuilder:
    """为每个 phase 构造全新 optimizer，并在 finetune 前只载入父模型状态。"""

    def __init__(
        self,
        request: TaskExecutionRequest,
        route: TrainingRouteSpec,
        *,
        workspace_root: Path,
        execution_root: Path,
        rank: int = 0,
        world_size: int = 1,
        distributed_executor: object | None = None,
        resume_requested: bool = False,
    ) -> None:
        self.request = request
        self.route = route
        self.workspace_root = workspace_root
        self.execution_root = execution_root
        self.rank = rank
        self.world_size = world_size
        self.distributed_executor = distributed_executor
        self.resume_requested = resume_requested
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("STAGE456_RANK_WORLD_SIZE_INVALID")
        if (distributed_executor is None) != (world_size == 1):
            raise ValueError("STAGE456_DISTRIBUTED_EXECUTOR_WORLD_SIZE_MISMATCH")
        rank_root = execution_root / f"rank-{rank:04d}"
        self.phase_checkpoint_root = rank_root / "phase-checkpoints"
        self.phase_event_root = rank_root / "events"
        self.phase_checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.phase_event_root.mkdir(parents=True, exist_ok=True)
        self.base_initialization_hashes: dict[str, str] = {}
        self.asset_evidence: dict[str, tuple[Mapping[str, JSONValue], ...]] = {}
        self.resource_profiles: dict[
            str, list[Mapping[str, JSONValue]]
        ] = {}
        # 评估频率属于 resolved config，而 checkpoint 行来自训练后的权威 commit。
        # 在 builder 内保存每个 phase 的确定边界，使派生阶段只评估合同要求的
        # checkpoint；训练自身因常规保存策略产生的额外 checkpoint 不会被误当成
        # 一次正式评估。
        self.evaluation_steps: dict[str, tuple[int, ...]] = {}
        self._sinks: list[JsonlEventSink] = []
        self._cursors: list[object] = []

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()
        self._sinks.clear()
        for cursor in self._cursors:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()
        self._cursors.clear()

    def _provider_options(self, phase: TrainingPhaseSpec) -> Mapping[str, object]:
        base = self.request.config.section("providers")
        assert isinstance(base, dict)
        raw = phase.metadata.get("provider", {})
        if not isinstance(raw, Mapping):
            raise TypeError("STAGE456_PHASE_PROVIDER_METADATA_NOT_OBJECT")
        if set(raw) - set(base):
            raise ValueError(
                f"STAGE456_PHASE_PROVIDER_UNKNOWN_FIELDS:{sorted(set(raw)-set(base))}"
            )
        return {**base, **dict(raw)}

    @staticmethod
    def _select_dataset_file(asset: object, split: str) -> Path:
        matches = [
            item.path
            for item in tuple(getattr(asset, "files"))
            if item.path.suffix.casefold() in {".jsonl", ".json"}
            and (item.role == split or split.casefold() in item.relative_path.casefold())
        ]
        if len(matches) != 1:
            raise ValueError(
                f"STAGE456_DATASET_SPLIT_FILE_NOT_UNIQUE:{split}:count={len(matches)}"
            )
        return matches[0]

    def resources(
        self,
        phase: TrainingPhaseSpec,
        *,
        evaluation: bool = False,
    ) -> _PhaseResources:
        options = _PhaseOptions.from_request(self.request, phase)
        providers = self._provider_options(phase)
        batching = self.request.config.base_config.section("batching")
        identity = self.request.config.base_config.section("identity")
        data = self.request.config.base_config.section("data")
        microbatch_size = int(batching["microbatch_size"])
        micros_per_step = int(batching["accumulation_steps"]) * (
            int(batching["per_device_batch_size"]) // microbatch_size
        )
        master_seed = int(identity["master_seed"])
        seed_plan = SeedPlan.from_master_seed(master_seed, world_size=self.world_size)
        data_seed = seed_plan.derive_subseed(
            "sampler",
            "route_phase",
            options.data_seed_offset,
            "evaluation" if evaluation else "training",
        )
        if providers["kind"] == "tiny":
            if self.request.config.run_intent != "local_fixture":
                raise ValueError("STAGE456_FORMAL_ROUTE_CANNOT_USE_TINY_PROVIDER")
            evaluation_options = self.request.config.section("evaluation")
            assert isinstance(evaluation_options, dict)
            fixture_steps = (
                int(evaluation_options["max_batches"] or 1)
                if evaluation
                else options.max_steps * self.world_size
            )
            model_fixture: TinyTrainingFixture = build_tiny_training_fixture(
                task_type=options.task_type,
                # 模型初始化只绑定 base seed；phase 差异只能改变数据 seed。这样直接
                # 监督与预训练根节点的“同一初始化”是实际 tensor hash 相等，而非
                # 仅比较 route 中的一段字符串。
                seed=seed_plan.seed_for("model_init"),
                steps=fixture_steps,
                microbatches_per_step=micros_per_step,
                microbatch_size=microbatch_size,
                sequence_length=int(data["sequence_length"]),
                num_labels=options.num_labels,
            )
            # tiny builder 的一个 seed 同时决定模型和数据；分两次构造再组合，确保
            # model_init 与 sampler 域真正隔离。两次构造都通过 fork_rng，不改变
            # 即将被 TrainingEngine checkpoint 的全局 runtime RNG。
            data_fixture = build_tiny_training_fixture(
                task_type=options.task_type,
                seed=data_seed,
                steps=fixture_steps,
                microbatches_per_step=micros_per_step,
                microbatch_size=microbatch_size,
                sequence_length=int(data["sequence_length"]),
                num_labels=options.num_labels,
            )
            return _PhaseResources(
                model_fixture.model,
                data_fixture.dataset,
                options,
                (),
            )

        if providers["kind"] != "offline_hf":
            raise ValueError("STAGE456_PROVIDER_KIND_UNSUPPORTED")
        pairs = (
            ("model_manifest_ref", "model_root_ref"),
            ("data_manifest_ref", "data_root_ref"),
            ("tokenizer_manifest_ref", "tokenizer_root_ref"),
        )
        assets = []
        evidence: list[Mapping[str, JSONValue]] = []
        for manifest_field, root_field in pairs:
            manifest_ref = providers[manifest_field]
            root_ref = providers[root_field]
            if manifest_ref is None or root_ref is None:
                raise _blocked_missing(
                    BlockerCode.ASSET_UNAVAILABLE,
                    manifest_field,
                    f"formal phase 缺少 {manifest_field}/{root_field}",
                )
            try:
                asset = resolve_ready_asset(
                    _resolve(self.workspace_root, manifest_ref, field_name=manifest_field),
                    _resolve(self.workspace_root, root_ref, field_name=root_field),
                )
            except FileNotFoundError as error:
                raise _blocked_missing(
                    BlockerCode.ASSET_UNAVAILABLE,
                    manifest_field,
                    str(error),
                    evidence_refs=(str(manifest_ref),),
                ) from error
            assets.append(asset)
            evidence.append(
                {
                    "manifest_ref": str(manifest_ref),
                    "asset_id": asset.asset_id,
                    "file_count": len(asset.files),
                }
            )
        model_asset, data_asset, _tokenizer_asset = assets
        if model_asset.asset_id != phase.model_asset_id:
            raise ValueError("STAGE456_ROUTE_MODEL_ASSET_ID_MISMATCH")
        if data_asset.asset_id != phase.dataset_asset_id:
            raise ValueError("STAGE456_ROUTE_DATASET_ASSET_ID_MISMATCH")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(master_seed)
            model = OfflineHuggingFaceModelAdapter.from_local_directory(
                model_asset.root,
                task_type=options.task_type,
                num_labels=options.num_labels,
            )
        split = options.evaluation_split if evaluation else options.training_split
        source = self._select_dataset_file(data_asset, split)
        dataset = PretokenizedJsonlDatasetAdapter(
            source,
            dataset_id=data_asset.asset_id,
            microbatch_size=microbatch_size,
            microbatches_per_step=micros_per_step,
            tensor_fields=("input_ids", "attention_mask", "labels"),
        )
        return _PhaseResources(model, dataset, options, tuple(evidence))

    def _load_parent_model(
        self,
        phase: TrainingPhaseSpec,
        parent_result: Mapping[str, object],
        model: TorchModelAdapter,
    ) -> str:
        parent_id = phase.parent_phase_id
        assert parent_id is not None
        physical = parent_result.get("physical_checkpoint_id")
        if not isinstance(physical, str) or not physical:
            raise ValueError("STAGE456_PARENT_PHYSICAL_CHECKPOINT_MISSING")
        store = CheckpointStore(self.phase_checkpoint_root / parent_id)
        state, commit = store.load(physical)
        if not isinstance(state, Mapping) or not isinstance(state.get("model"), Mapping):
            raise ValueError("STAGE456_PARENT_CHECKPOINT_MODEL_STATE_INVALID")
        parent_model = state["model"]
        assert isinstance(parent_model, Mapping)
        # DDP checkpoint 的 key 带 ``module.``；新 phase 在包装 DDP 之前恢复父模型，
        # 因而必须显式、全量去除这一层前缀，禁止混合 key 静默加载。
        if parent_model and all(str(name).startswith("module.") for name in parent_model):
            parent_model = {
                str(name)[len("module.") :]: tensor
                for name, tensor in parent_model.items()
            }
        model.module.load_state_dict(parent_model, strict=True)  # type: ignore[arg-type]
        return commit.manifest_sha256

    def __call__(
        self,
        phase: TrainingPhaseSpec,
        parent_result: Mapping[str, object] | None,
    ) -> TrainingPhaseRuntime:
        resources = self.resources(phase)
        model = resources.model
        model_state_hash = _tensor_tree_hash(model.module.state_dict())
        if parent_result is None:
            observed = self.base_initialization_hashes.setdefault(
                phase.base_initialization_id, model_state_hash
            )
            if observed != model_state_hash:
                raise ValueError("STAGE456_ACTUAL_BASE_INITIALIZATION_MISMATCH")
            consumed_checkpoint = None
            parent_hash = None
        else:
            if phase.input_checkpoint_id is None:
                raise ValueError("STAGE456_CHILD_INPUT_CHECKPOINT_ID_MISSING")
            self._load_parent_model(phase, parent_result, model)
            consumed_checkpoint = phase.input_checkpoint_id
            parent_hash = parent_result.get("artifact_hash")
            if not isinstance(parent_hash, str) or _HASH_RE.fullmatch(parent_hash) is None:
                raise ValueError("STAGE456_PARENT_PHASE_RESULT_HASH_INVALID")

        device_name = str(self.request.config.base_config.section("runtime")["device"])
        if self.distributed_executor is None:
            model.module.to(torch.device(device_name))
        else:
            wrapped = self.distributed_executor.wrap_model(model.module)  # type: ignore[union-attr]
            model = TorchModelAdapter(wrapped, task_type=resources.options.task_type)
        optimizer_config = dict(self.request.config.base_config.section("optimizer"))
        if resources.options.learning_rate is not None:
            optimizer_config["learning_rate"] = resources.options.learning_rate
        optimizer_runtime = self.request.config.section("optimizer_runtime")
        scheduler_options = dict(self.request.config.section("scheduler"))
        precision = self.request.config.section("precision_runtime")
        assert isinstance(optimizer_runtime, dict) and isinstance(precision, dict)
        optimizer = build_optimizer(
            model.module.parameters(), optimizer_config, optimizer_runtime
        )
        if scheduler_options["kind"] != "none" and scheduler_options["total_steps"] is None:
            scheduler_options["total_steps"] = resources.options.max_steps
        scheduler = build_scheduler(optimizer, scheduler_options)
        scaler = build_grad_scaler(precision, device_type=device_name)

        decision = self.route.estimator_decision
        estimator = "u"
        decision_hash: str | None = None
        decision_gate_status: str | None = None
        if phase.importance_enabled:
            if not isinstance(decision, EstimatorDecision):
                raise ValueError("STAGE456_PHASE_IMPORTANCE_DECISION_MISSING")
            estimator = "u" if decision.selected_estimator == "weighted_u" else str(
                decision.selected_estimator
            )
            if self.request.config.run_intent == "formal":
                decision_hash = decision.artifact_hash
                decision_gate_status = decision.gate_status
        execution = self.request.config.section("execution")
        training_options = self.request.config.section("training")
        profiling = self.request.config.section("profiling")
        evaluation = self.request.config.section("evaluation")
        checkpoint_schedule = self.request.config.section("checkpoint_schedule")
        base_importance = self.request.config.base_config.section("importance")
        base_data = self.request.config.base_config.section("data")
        base_logging = self.request.config.base_config.section("logging")
        identity = self.request.config.base_config.section("identity")
        assert isinstance(execution, dict)
        assert all(
            isinstance(item, dict)
            for item in (
                training_options,
                profiling,
                evaluation,
                checkpoint_schedule,
            )
        )
        if bool(evaluation["enabled"]):  # type: ignore[index]
            interval = int(
                evaluation["every_steps"]  # type: ignore[index]
                or training_options["validation_every_steps"]  # type: ignore[index]
                or resources.options.max_steps
            )
            evaluation_steps = list(
                range(interval, resources.options.max_steps + 1, interval)
            )
            if not evaluation_steps or evaluation_steps[-1] != resources.options.max_steps:
                evaluation_steps.append(resources.options.max_steps)
        else:
            evaluation_steps = []
        self.evaluation_steps[phase.phase_id] = tuple(evaluation_steps)
        seed_plan = install_training_rng(
            int(identity["master_seed"]),  # type: ignore[index]
            rank=self.rank,
            world_size=self.world_size,
        )
        autocast_dtype = (
            str(precision["autocast_dtype"])
            if bool(precision["autocast_enabled"])
            else "none"
        )
        spec = TrainingRunSpec(
            run_id=f"{self.route.route_id}-{phase.phase_id}",
            run_intent=self.request.config.run_intent,
            max_steps=resources.options.max_steps,
            max_attempts=resources.options.max_steps + int(execution["max_attempts"]) - 1,
            importance_enabled=phase.importance_enabled,
            estimator_name=estimator,
            accumulation_dtype=str(
                self.request.config.base_config.section("precision")["statistic_dtype"]
            ),
            max_grad_norm=training_options["gradient_clip_max_norm"],  # type: ignore[arg-type]
            autocast_dtype=autocast_dtype,
            checkpoint_every_steps=phase.checkpoint_frequency_steps,
            log_every_steps=int(base_logging["log_every_steps"]),
            weights_exogenous=bool(base_data["weights_exogenous"]),
            common_mean_assumption=bool(base_data["common_mean_assumption"]),
            estimator_decision_hash=decision_hash,
            estimator_gate_status=decision_gate_status,
            metadata={
                "route_id": self.route.route_id,
                "route_lineage_hash": self.route.lineage_hash,
                "phase_id": phase.phase_id,
                "logical_output_checkpoint_id": phase.output_checkpoint_id,
                "base_initialization_id": phase.base_initialization_id,
                "seed_plan_hash": seed_plan.artifact_hash,
                "task_estimator_contract": base_importance["estimator_name"],
                "route_declared_checkpoint_frequency_steps": (
                    phase.checkpoint_frequency_steps
                ),
                "evaluation_steps": evaluation_steps,
            },
            # v2 的分段计划是运行时权威合同；route phase 中的旧频率保留为
            # checkpoint_every_steps fallback 和 lineage 元数据。评估边界若不落在
            # 分段保存点，包装器会显式补一个事务 checkpoint。
            checkpoint_segments=tuple(
                dict(item)
                for item in checkpoint_schedule["segments"]  # type: ignore[index,union-attr]
            ),
        )
        checkpoint_store = CheckpointStore(self.phase_checkpoint_root / phase.phase_id)
        session_index = len(tuple(self.phase_event_root.glob(f"{phase.phase_id}-*.jsonl")))
        event_path = self.phase_event_root / f"{phase.phase_id}-{session_index:04d}.jsonl"
        sink = JsonlEventSink(event_path)
        self._sinks.append(sink)
        data_loader = self.request.config.section("data_loader")
        assert isinstance(data_loader, dict)
        training_cursor = configure_batch_cursor(
            resources.dataset.cursor(
                seed=seed_plan.seed_for("sampler"),
                rank=self.rank,
                world_size=self.world_size,
            ),
            num_workers=int(data_loader["num_workers"]),
            prefetch_factor=data_loader["prefetch_factor"],  # type: ignore[arg-type]
            persistent_workers=bool(data_loader["persistent_workers"]),
        )
        self._cursors.append(training_cursor)
        engine = TrainingEngine(
            spec=spec,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            cursor=training_cursor,
            reducer=(
                None
                if self.distributed_executor is None
                else self.distributed_executor.reducer  # type: ignore[union-attr]
            ),
            checkpoint_store=checkpoint_store,
            event_sink=sink,
            experiment_id=f"stage-{self.request.task.stage}-route",
            attempt_id="attempt-0000",
            session_id=f"{phase.phase_id}-session-{session_index:04d}",
            rank=self.rank,
        )
        committed_checkpoints = checkpoint_store.discover()
        if committed_checkpoints and not self.resume_requested:
            raise RuntimeError("TRAINING_RESUME_REF_REQUIRED_FOR_EXISTING_CHECKPOINTS")
        if self.resume_requested and committed_checkpoints:
            engine.resume_latest()
            # 每个 fresh process 使用独立 JSONL session，故事件序号从零重新开始；
            # optimizer/cursor/importance 等状态仍完整来自 checkpoint。
            engine.state = TrainingState(
                engine.state.global_step,
                engine.state.attempt_index,
                engine.state.skipped_steps,
                0,
                engine.state.last_checkpoint_id,
            )
        phase_profiles = self.resource_profiles.setdefault(phase.phase_id, [])
        profiled_engine = _RouteTrainingEngine(
            engine,
            profiling=profiling,
            store=ResourceWindowStore(
                self.workspace_root,
                self.execution_root
                / f"rank-{self.rank:04d}"
                / "resource-profiles"
                / phase.phase_id,
            ),
            runtime_device=device_name,
            deterministic_algorithms=bool(
                training_options["deterministic_algorithms"]
            ),
            evaluation_steps=evaluation_steps,
            published=phase_profiles,
        )
        self.asset_evidence[phase.phase_id] = resources.evidence
        return TrainingPhaseRuntime(  # type: ignore[arg-type]
            profiled_engine,
            consumed_checkpoint,
            parent_hash,
        )

    def _checkpoint_rows(
        self,
        phase: TrainingPhaseSpec,
    ) -> tuple[Mapping[str, JSONValue], ...]:
        store = CheckpointStore(self.phase_checkpoint_root / phase.phase_id)
        rows: list[Mapping[str, JSONValue]] = []
        for commit in store.discover():
            state, _ = store.load(commit.checkpoint_id)
            if not isinstance(state, Mapping):
                raise ValueError("STAGE456_CHECKPOINT_STATE_NOT_OBJECT")
            importance = state.get("importance")
            training_state = state.get("training_state")
            if not isinstance(training_state, Mapping) or not isinstance(
                training_state.get("global_step"), int
            ):
                raise ValueError("STAGE456_CHECKPOINT_TRAINING_STATE_INVALID")
            summaries = _importance_state_summary(importance)
            registry_hash = _require_hash(
                state.get("registry_hash"),
                field_name=f"checkpoint_registry_hash:{phase.phase_id}",
            )
            rows.append(
                {
                    "route_id": self.route.route_id,
                    "phase_id": phase.phase_id,
                    "phase_type": phase.phase_type,
                    "checkpoint_id": commit.checkpoint_id,
                    "checkpoint_manifest_sha256": commit.manifest_sha256,
                    "generation": commit.generation,
                    "parent_checkpoint_id": commit.parent_checkpoint_id,
                    "global_step": int(training_state["global_step"]),
                    "coordinate_registry_hash": registry_hash,
                    **summaries,
                }
            )
        return tuple(rows)

    def phase_model_and_batches(
        self,
        phase: TrainingPhaseSpec,
        phase_result: Mapping[str, object],
    ) -> tuple[TorchModelAdapter, tuple[TrainingMicrobatch, ...], Mapping[str, object]]:
        resources = self.resources(phase, evaluation=True)
        physical = phase_result.get("physical_checkpoint_id")
        if not isinstance(physical, str):
            raise ValueError("STAGE456_PHASE_PHYSICAL_CHECKPOINT_MISSING")
        state, _commit = CheckpointStore(
            self.phase_checkpoint_root / phase.phase_id
        ).load(physical)
        if not isinstance(state, Mapping) or not isinstance(state.get("model"), Mapping):
            raise ValueError("STAGE456_PHASE_MODEL_CHECKPOINT_INVALID")
        model_state = state["model"]
        assert isinstance(model_state, Mapping)
        if model_state and all(str(name).startswith("module.") for name in model_state):
            model_state = {
                str(name)[len("module.") :]: tensor
                for name, tensor in model_state.items()
            }
        resources.model.module.load_state_dict(model_state, strict=True)  # type: ignore[arg-type]
        identity = self.request.config.base_config.section("identity")
        evaluation_seed = SeedPlan.from_master_seed(
            int(identity["master_seed"])  # type: ignore[index]
        ).derive_subseed(
            "sampler",
            "route_evaluation",
            resources.options.data_seed_offset,
        )
        cursor = resources.dataset.cursor(
            seed=evaluation_seed,
            rank=0,
            world_size=1,
        )
        evaluation = self.request.config.section("evaluation")
        assert isinstance(evaluation, dict)
        max_batches = int(evaluation["max_batches"] or 1)
        batches: list[TrainingMicrobatch] = []
        for _ in range(max_batches):
            try:
                batches.extend(cursor.next_microbatches())
            except StopIteration:
                break
        if not batches:
            raise ValueError("STAGE456_EVALUATION_DATA_EMPTY")
        device = torch.device(
            str(self.request.config.base_config.section("runtime")["device"])
        )
        resources.model.module.to(device)
        return resources.model, tuple(batch.to(device) for batch in batches), state


def _tensor_mapping_sum(value: object) -> float:
    if not isinstance(value, Mapping):
        raise TypeError("STAGE456_IMPORTANCE_VIEW_NOT_MAPPING")
    result = 0.0
    for tensor in value.values():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("STAGE456_IMPORTANCE_VIEW_VALUE_NOT_TENSOR")
        result += float(tensor.detach().to(torch.float64).sum().item())
    if not math.isfinite(result):
        raise ValueError("STAGE456_IMPORTANCE_VIEW_NONFINITE")
    return result


def _importance_state_summary(value: object) -> Mapping[str, JSONValue]:
    if value is None:
        return {
            "importance_enabled": False,
            "successful_steps": 0,
            "skipped_steps": 0,
            "view_sums": {},
            "topk_parameter_names": [],
            "topk_mass": None,
        }
    if not isinstance(value, Mapping):
        raise TypeError("STAGE456_IMPORTANCE_STATE_NOT_OBJECT")
    positive = value.get("positive")
    negative = value.get("negative_mass")
    if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
        raise ValueError("STAGE456_IMPORTANCE_STATE_VIEWS_MISSING")
    if tuple(sorted(positive)) != tuple(sorted(negative)):
        raise ValueError("STAGE456_IMPORTANCE_STATE_COORDINATE_MISMATCH")
    per_parameter: list[tuple[float, str]] = []
    total_absolute = 0.0
    signed_total = 0.0
    for name in sorted(positive):
        pos = positive[name]
        neg = negative[name]
        if not isinstance(pos, torch.Tensor) or not isinstance(neg, torch.Tensor):
            raise TypeError("STAGE456_IMPORTANCE_COORDINATE_NOT_TENSOR")
        absolute = float((pos + neg).detach().to(torch.float64).sum().item())
        signed = float((pos - neg).detach().to(torch.float64).sum().item())
        per_parameter.append((absolute, str(name)))
        total_absolute += absolute
        signed_total += signed
    count = max(1, math.ceil(0.1 * len(per_parameter)))
    ranked = sorted(per_parameter, key=lambda item: (-item[0], item[1]))
    top = ranked[:count]
    top_mass: float | None = None
    if total_absolute > 0:
        top_mass = sum(value for value, _ in top) / total_absolute
    views: dict[str, float] = {
        "positive": _tensor_mapping_sum(positive),
        "negative_mass": _tensor_mapping_sum(negative),
        "absolute": total_absolute,
        "signed": signed_total,
    }
    for name in (
        "raw",
        "raw_clipped",
        "data_movement",
        "total_movement",
        "weight_decay_movement",
        "magnitude",
    ):
        if name in value:
            views[name] = _tensor_mapping_sum(value[name])
    return {
        "importance_enabled": True,
        "successful_steps": int(value.get("successful_steps", 0)),
        "skipped_steps": int(value.get("skipped_steps", 0)),
        "view_sums": views,
        "topk_parameter_names": [name for _, name in top],
        "topk_mass": top_mass,
    }


def _layer_module_summary(
    route_id: str,
    phase: TrainingPhaseSpec,
    importance: object,
    *,
    coordinate_registry_hash: str,
) -> tuple[Mapping[str, JSONValue], ...]:
    if not isinstance(importance, Mapping):
        return ()
    positive = importance.get("positive")
    negative = importance.get("negative_mass")
    if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
        return ()
    groups: dict[tuple[str, str], dict[str, float]] = {}
    for name in sorted(positive):
        pos = positive[name]
        neg = negative[name]
        if not isinstance(pos, torch.Tensor) or not isinstance(neg, torch.Tensor):
            raise TypeError("STAGE456_LAYER_IMPORTANCE_NOT_TENSOR")
        module = str(name).rsplit(".", 1)[0] if "." in str(name) else str(name)
        layer = str(name).split(".", 1)[0]
        bucket = groups.setdefault(
            (layer, module),
            {"positive": 0.0, "negative_mass": 0.0, "signed": 0.0, "absolute": 0.0},
        )
        pos_sum = float(pos.detach().to(torch.float64).sum().item())
        neg_sum = float(neg.detach().to(torch.float64).sum().item())
        bucket["positive"] += pos_sum
        bucket["negative_mass"] += neg_sum
        bucket["signed"] += pos_sum - neg_sum
        bucket["absolute"] += pos_sum + neg_sum
    return tuple(
        {
            "route_id": route_id,
            "phase_id": phase.phase_id,
            "phase_type": phase.phase_type,
            "coordinate_registry_hash": coordinate_registry_hash,
            "layer": layer,
            "module": module,
            **groups[(layer, module)],
        }
        for layer, module in sorted(groups)
    )


def _evaluate(
    model: TorchModelAdapter,
    batches: Sequence[TrainingMicrobatch],
) -> Mapping[str, float]:
    evaluator = (
        CausalLMEvaluator()
        if model.task_type == "causal_lm"
        else ClassificationEvaluator()
    )
    return evaluator.evaluate(model, batches)


def _topk_jaccard(left: Sequence[str], right: Sequence[str]) -> Mapping[str, JSONValue]:
    lhs, rhs = set(left), set(right)
    union = lhs | rhs
    if not union:
        return {"defined": False, "reason": "both_topk_sets_empty", "value": None}
    return {
        "defined": True,
        "reason": None,
        "value": len(lhs & rhs) / len(union),
    }


@dataclass(frozen=True, slots=True)
class _ExecutedRoute:
    route: TrainingRouteSpec
    route_result: TrainingRouteResult
    common_payload: Mapping[str, JSONValue]
    source_refs: tuple[str, ...]


class _SynchronizedTrainingRouteRunner(TrainingRouteRunner):
    """在每个 phase commit 前后加入 rank 屏障的 DDP 路线执行器。

    路线 phase 的权威 commit 不能由某个 rank 提前发布，否则 fresh-process resume
    可能让该 rank 跳过 collective，而其他 rank 仍在 backward。这里先比较所有
    rank 的 phase-commit 状态：全有则共同跳过、全无则共同训练；状态不对称时
    fail-closed，避免以死锁或不一致参数继续。正常训练只有在所有 rank 都完成后
    才各自发布本 rank 的 phase commit。
    """

    def __init__(self, *args: object, executor: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.executor = executor

    def run(self, *, resume: bool = True) -> TrainingRouteResult:
        completed: dict[str, Mapping[str, object]] = {}
        world_size = int(self.executor.spec.world_size)  # type: ignore[union-attr]
        for phase_id in self.route.topological_order:
            phase = self.route.phase(phase_id)
            existing = self._load_committed(phase) if resume else None
            committed_count = self.executor.reducer.sum_int(  # type: ignore[union-attr]
                1 if existing is not None else 0
            )
            if committed_count == world_size:
                assert existing is not None
                completed[phase_id] = existing
                continue
            if committed_count != 0:
                raise RuntimeError(
                    f"STAGE456_DISTRIBUTED_PHASE_COMMIT_ASYMMETRY:{phase_id}:"
                    f"{committed_count}/{world_size}"
                )
            parent = (
                None
                if phase.parent_phase_id is None
                else completed.get(phase.parent_phase_id)
            )
            if phase.parent_phase_id is not None and parent is None:
                raise RuntimeError("TRAINING_ROUTE_PARENT_NOT_COMPLETE")
            runtime = self.builder(phase, parent)
            if not isinstance(runtime, TrainingPhaseRuntime):
                raise TypeError("TRAINING_ROUTE_BUILDER_RESULT_INVALID")
            if phase.input_checkpoint_id != runtime.consumed_checkpoint_id:
                raise ValueError("TRAINING_ROUTE_CONSUMED_CHECKPOINT_MISMATCH")
            expected_parent_hash = None if parent is None else parent.get("artifact_hash")
            if expected_parent_hash != runtime.consumed_parent_result_hash:
                raise ValueError("TRAINING_ROUTE_CONSUMED_PARENT_HASH_MISMATCH")
            result = runtime.engine.run()
            complete_count = self.executor.reducer.sum_int(  # type: ignore[union-attr]
                1 if result.status == "COMPLETE" else 0
            )
            if complete_count != world_size:
                return TrainingRouteResult(
                    self.route.route_id,
                    self.route.lineage_hash,
                    "PARTIAL",
                    completed,
                )
            if result.state.last_checkpoint_id is None:
                raise RuntimeError("TRAINING_ROUTE_PHASE_MISSING_OUTPUT_CHECKPOINT")
            self.executor.barrier()  # type: ignore[union-attr]
            completed[phase_id] = self._publish_phase(phase, result)
            self.executor.barrier()  # type: ignore[union-attr]
        return TrainingRouteResult(
            self.route.route_id,
            self.route.lineage_hash,
            "COMPLETE",
            completed,
        )


def _run_route(
    request: TaskExecutionRequest,
    root: Path,
    *,
    rank: int = 0,
    world_size: int = 1,
    distributed_executor: object | None = None,
    build_payload: bool = True,
) -> _ExecutedRoute:
    route, route_ref, route_source_hash = _load_route(request, root)
    estimator_audit = _validate_estimator_binding(request, route, root)
    quadrature_audit = _validate_quadrature_binding(request, root)
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    output_root = _resolve(root, artifacts["output_dir"], field_name="output_dir")
    execution_root = output_root / "route-execution" / route.lineage_hash
    recovery = request.config.section("recovery")
    assert isinstance(recovery, dict)
    resume_ref = recovery["resume_ref"]
    resume_requested = resume_ref is not None
    if resume_requested:
        resume_path = _resolve(root, resume_ref, field_name="recovery.resume_ref")
        try:
            resume_path.relative_to(execution_root.resolve())
        except ValueError as error:
            raise ValueError("STAGE456_RESUME_REF_OUTSIDE_ROUTE_EXECUTION") from error
        if not execution_root.exists():
            raise FileNotFoundError("STAGE456_RESUME_REF_HAS_NO_ROUTE_EXECUTION")
    builder = _RoutePhaseBuilder(
        request,
        route,
        workspace_root=root,
        execution_root=execution_root,
        rank=rank,
        world_size=world_size,
        distributed_executor=distributed_executor,
        resume_requested=resume_requested,
    )
    try:
        runner_class = (
            TrainingRouteRunner
            if distributed_executor is None
            else _SynchronizedTrainingRouteRunner
        )
        runner_kwargs = (
            {}
            if distributed_executor is None
            else {"executor": distributed_executor}
        )
        result = runner_class(
            route,
            builder,
            result_root=execution_root / f"rank-{rank:04d}" / "phase-results",
            **runner_kwargs,
        ).run(resume=resume_requested)
    finally:
        builder.close()
    if result.status != "COMPLETE":
        raise RuntimeError(f"STAGE456_ROUTE_NOT_COMPLETE:{result.status}")
    if not build_payload:
        return _ExecutedRoute(route, result, {}, (route_ref,))

    lineage: list[Mapping[str, JSONValue]] = []
    training_metrics: list[Mapping[str, JSONValue]] = []
    evaluation_metrics: list[Mapping[str, JSONValue]] = []
    trajectory: list[Mapping[str, JSONValue]] = []
    layers: list[Mapping[str, JSONValue]] = []
    topk_sets: list[Mapping[str, JSONValue]] = []
    phase_states: dict[str, Mapping[str, object]] = {}
    phase_models: dict[str, tuple[TorchModelAdapter, tuple[TrainingMicrobatch, ...]]] = {}
    for phase_id in route.topological_order:
        phase = route.phase(phase_id)
        phase_result = result.phase_results[phase_id]
        physical = phase_result.get("physical_checkpoint_id")
        if not isinstance(physical, str):
            raise ValueError("STAGE456_ROUTE_PHASE_PHYSICAL_CHECKPOINT_INVALID")
        store = CheckpointStore(builder.phase_checkpoint_root / phase_id)
        state, commit = store.load(physical)
        if not isinstance(state, Mapping):
            raise ValueError("STAGE456_ROUTE_PHASE_CHECKPOINT_STATE_INVALID")
        phase_states[phase_id] = state
        training_result = phase_result.get("training_result")
        if not isinstance(training_result, Mapping):
            raise ValueError("STAGE456_ROUTE_TRAINING_RESULT_INVALID")
        records = training_result.get("records")
        if not isinstance(records, list):
            raise ValueError("STAGE456_ROUTE_RECORDS_INVALID")
        committed_losses = [
            float(item["mean_loss"])
            for item in records
            if isinstance(item, Mapping) and item.get("status") == "COMMITTED"
        ]
        physical_commit_path = store.commits / f"{physical}.json"
        physical_commit_value = load_canonical_json(physical_commit_path)
        if not isinstance(physical_commit_value, dict):
            raise ValueError("STAGE456_PHYSICAL_CHECKPOINT_COMMIT_NOT_OBJECT")
        # ``committed_at`` 是发布时钟，不属于逻辑 checkpoint 身份；排除后才能让
        # fresh-process 重放得到相同 identity，同时仍由 commit 文件本身保留审计时间。
        physical_commit_identity = {
            key: value
            for key, value in physical_commit_value.items()
            if key != "committed_at"
        }
        lineage.append(
            {
                "route_id": route.route_id,
                "route_lineage_hash": route.lineage_hash,
                "phase_id": phase_id,
                "phase_type": phase.phase_type,
                "parent_phase_id": phase.parent_phase_id,
                "base_initialization_id": phase.base_initialization_id,
                "base_initialization_tensor_hash": builder.base_initialization_hashes.get(
                    phase.base_initialization_id
                ),
                "logical_input_checkpoint_id": phase.input_checkpoint_id,
                "logical_output_checkpoint_id": phase.output_checkpoint_id,
                "physical_checkpoint_id": physical,
                "physical_checkpoint_commit_ref": _relative(
                    root,
                    physical_commit_path,
                ),
                # 两个摘要分别绑定权威 commit 对象与其指向的安全 TensorBundle。
                # 下游不能只凭 checkpoint ID 或可变目录名认定可恢复状态。
                "commit_identity_sha256": canonical_json_hash(
                    physical_commit_identity
                ),
                "bundle_manifest_sha256": commit.manifest_sha256,
                "physical_checkpoint_manifest_sha256": commit.manifest_sha256,
                "phase_result_hash": phase_result.get("artifact_hash"),
            }
        )
        training_metrics.append(
            {
                "route_id": route.route_id,
                "phase_id": phase_id,
                "phase_type": phase.phase_type,
                "successful_steps": len(committed_losses),
                "skipped_steps": sum(
                    1
                    for item in records
                    if isinstance(item, Mapping) and item.get("status") == "SKIPPED"
                ),
                "mean_training_loss": (
                    None if not committed_losses else fmean(committed_losses)
                ),
                "final_training_loss": (
                    None if not committed_losses else committed_losses[-1]
                ),
            }
        )
        model, batches, loaded_state = builder.phase_model_and_batches(phase, phase_result)
        phase_models[phase_id] = (model, batches)
        rows = builder._checkpoint_rows(phase)
        trajectory.extend(rows)
        evaluation_config = request.config.section("evaluation")
        assert isinstance(evaluation_config, dict)
        if bool(evaluation_config["enabled"]):
            requested_steps = set(builder.evaluation_steps.get(phase_id, ()))
            evaluated_steps: set[int] = set()
            for row in rows:
                step = int(row["global_step"])
                if step not in requested_steps:
                    continue
                checkpoint_id = str(row["checkpoint_id"])
                checkpoint_state, checkpoint_commit = store.load(checkpoint_id)
                if not isinstance(checkpoint_state, Mapping) or not isinstance(
                    checkpoint_state.get("model"), Mapping
                ):
                    raise ValueError("STAGE456_EVALUATION_CHECKPOINT_MODEL_INVALID")
                model_state = checkpoint_state["model"]
                assert isinstance(model_state, Mapping)
                if model_state and all(
                    str(name).startswith("module.") for name in model_state
                ):
                    model_state = {
                        str(name)[len("module.") :]: tensor
                        for name, tensor in model_state.items()
                    }
                model.module.load_state_dict(model_state, strict=True)  # type: ignore[arg-type]
                computed_metrics = dict(_evaluate(model, batches))
                requested_metrics = tuple(
                    str(name) for name in evaluation_config["metrics"]
                )
                missing_metrics = set(requested_metrics) - set(computed_metrics)
                if missing_metrics:
                    raise ValueError(
                        "STAGE456_EVALUATOR_METRICS_MISSING:"
                        + ",".join(sorted(missing_metrics))
                    )
                sample_ids = [
                    sample_id
                    for batch in batches
                    for sample_id in batch.sample_ids
                ]
                metric_values = {
                    name: computed_metrics[name] for name in requested_metrics
                }
                panel_hash = canonical_json_hash(sample_ids)
                evaluation_contract = {
                    "schema_version": "stage456-evaluation-contract-v1",
                    "task_id": phase.task_id,
                    "split": str(evaluation_config["split"]),
                    "metric_names": sorted(requested_metrics),
                    "sample_count": len(sample_ids),
                    "panel_sample_ids_hash": panel_hash,
                }
                row: dict[str, JSONValue] = {
                    "schema_version": "stage456-route-evaluation-v1",
                    "route_id": route.route_id,
                    "route_lineage_hash": route.lineage_hash,
                    "phase_id": phase_id,
                    "phase_type": phase.phase_type,
                    "task_id": phase.task_id,
                    "global_step": step,
                    "split": str(evaluation_config["split"]),
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_manifest_sha256": checkpoint_commit.manifest_sha256,
                    "evaluator_id": (
                        "causal-lm-loss-perplexity-v1"
                        if model.task_type == "causal_lm"
                        else "classification-loss-accuracy-v1"
                    ),
                    "metrics": metric_values,
                    "sample_count": len(sample_ids),
                    "panel_sample_ids_hash": panel_hash,
                    "evaluation_contract_hash": canonical_json_hash(
                        evaluation_contract
                    ),
                }
                # record hash 绑定实际 checkpoint、固定 panel 与 metric 值；Stage 6
                # evaluate 会复算它并重新打开 checkpoint commit，而不是信任摘要表。
                row["record_hash"] = canonical_json_hash(row)
                evaluation_metrics.append(row)
                evaluated_steps.add(step)
            missing_steps = requested_steps - evaluated_steps
            if missing_steps:
                raise RuntimeError(
                    "STAGE456_EVALUATION_CHECKPOINTS_MISSING:"
                    + ",".join(str(step) for step in sorted(missing_steps))
                )
            # 剪枝探针必须观察 phase 最终参数，而不是最后一次遍历碰巧加载的状态。
            # 当前合同总会把 max_steps 纳入评估边界；仍显式恢复终态，以免未来调整
            # 评估策略后引入隐蔽的顺序依赖。
            final_model_state = loaded_state.get("model")
            if not isinstance(final_model_state, Mapping):
                raise ValueError("STAGE456_PHASE_FINAL_MODEL_STATE_INVALID")
            if final_model_state and all(
                str(name).startswith("module.") for name in final_model_state
            ):
                final_model_state = {
                    str(name)[len("module.") :]: tensor
                    for name, tensor in final_model_state.items()
                }
            model.module.load_state_dict(final_model_state, strict=True)  # type: ignore[arg-type]
        registry_hash = _require_hash(
            loaded_state.get("registry_hash"),
            field_name=f"checkpoint_registry_hash:{phase_id}",
        )
        final_summary = _importance_state_summary(loaded_state.get("importance"))
        topk_sets.append(
            {
                "route_id": route.route_id,
                "phase_id": phase_id,
                "phase_type": phase.phase_type,
                "coordinate_registry_hash": registry_hash,
                "parameter_names": final_summary["topk_parameter_names"],
                "topk_mass": final_summary["topk_mass"],
            }
        )
        layers.extend(
            _layer_module_summary(
                route.route_id,
                phase,
                loaded_state.get("importance"),
                coordinate_registry_hash=registry_hash,
            )
        )

    by_type = {
        str(row["phase_type"]): row
        for row in evaluation_metrics
        if row["phase_type"] in {"direct_supervised", "finetune"}
    }
    comparison: Mapping[str, JSONValue]
    if set(by_type) == {"direct_supervised", "finetune"}:
        direct_metrics = by_type["direct_supervised"]["metrics"]
        finetune_metrics = by_type["finetune"]["metrics"]
        assert isinstance(direct_metrics, Mapping) and isinstance(finetune_metrics, Mapping)
        shared = sorted(set(direct_metrics) & set(finetune_metrics))
        direct_top = next(
            row["parameter_names"]
            for row in topk_sets
            if row["phase_type"] == "direct_supervised"
        )
        finetune_top = next(
            row["parameter_names"]
            for row in topk_sets
            if row["phase_type"] == "finetune"
        )
        comparison = {
            "defined": True,
            "reason": None,
            "metric_deltas_finetune_minus_direct": {
                name: float(finetune_metrics[name]) - float(direct_metrics[name])
                for name in shared
            },
            "topk_jaccard": _topk_jaccard(direct_top, finetune_top),  # type: ignore[arg-type]
        }
    else:
        comparison = {
            "defined": False,
            "reason": "route_missing_direct_or_finetune_phase",
            "metric_deltas_finetune_minus_direct": {},
            "topk_jaccard": {
                "defined": False,
                "reason": "route_missing_direct_or_finetune_phase",
                "value": None,
            },
        }

    pruning_results = _minimal_pruning_probe(route, phase_models, phase_states)
    event_refs = sorted(
        _relative(root, path)
        for path in execution_root.glob("rank-*/events/*.jsonl")
    )
    common: dict[str, JSONValue] = {
        "schema_version": "stage456-route-execution-v1",
        "route_spec": route.to_dict(),  # type: ignore[dict-item]
        "route_source_hash": route_source_hash,
        "route_result": result.to_dict(),  # type: ignore[dict-item]
        "checkpoint_lineage": lineage,
        "training_metrics": training_metrics,
        "evaluation_metrics": evaluation_metrics,
        "importance_trajectory": trajectory,
        "layer_module_summary": layers,
        "topk_sets": topk_sets,
        "route_comparison": comparison,
        "pruning_results": pruning_results,
        "estimator_decision_audit": dict(estimator_audit),
        "quadrature_decision_audit": dict(quadrature_audit),
        "event_stream_refs": event_refs,
        "resource_profiles": [
            {
                "phase_id": phase_id,
                "window": dict(window),
            }
            for phase_id, windows in sorted(builder.resource_profiles.items())
            for window in sorted(windows, key=lambda item: int(item["repetition"]))
        ],
        "distributed_execution": {
            "world_size": world_size,
            "result_rank": rank,
            "backend": (
                "local"
                if distributed_executor is None
                else distributed_executor.spec.backend  # type: ignore[union-attr]
            ),
        },
        "asset_evidence": {
            phase_id: [dict(item) for item in rows]
            for phase_id, rows in sorted(builder.asset_evidence.items())
        },
        "local_validation_status": (
            "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
        ),
        "gate_status": "NOT_RUN",
    }
    # 同一路线会按 catalog artifact role 发布多个 envelope。独立的 execution hash
    # 排除 role/data 视图，只绑定一次真实训练、checkpoint、评估和重要性证据；Stage 6
    # matrix 可据此去重并拒绝同一 route lineage 下内容不一致的多个 task artifact。
    common["route_execution_hash"] = canonical_json_hash(common)
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    refs = [route_ref]
    refs.extend(str(item) for item in orchestration["input_result_refs"])
    quadrature_ref = orchestration["quadrature_decision_ref"]
    if quadrature_ref is not None:
        refs.append(str(quadrature_ref))
    if request.environment.estimator_decision_ref is not None:
        refs.append(request.environment.estimator_decision_ref)
    source_refs = tuple(dict.fromkeys(refs))
    return _ExecutedRoute(route, result, common, source_refs)


def _minimal_pruning_probe(
    route: TrainingRouteSpec,
    phase_models: Mapping[str, tuple[TorchModelAdapter, tuple[TrainingMicrobatch, ...]]],
    phase_states: Mapping[str, Mapping[str, object]],
) -> list[Mapping[str, JSONValue]]:
    """对最终监督 phase 做一个真实、非破坏性的 tiny 闭环剪枝探针。"""

    candidates = [
        phase
        for phase in route.phases
        if phase.phase_type in {"finetune", "direct_supervised"}
        and phase.importance_enabled
    ]
    if not candidates:
        return []
    phase = sorted(candidates, key=lambda item: (item.phase_type != "finetune", item.phase_id))[0]
    model, batches = phase_models[phase.phase_id]
    state = phase_states[phase.phase_id]
    importance = state.get("importance")
    if not isinstance(importance, Mapping):
        return []
    positive = importance.get("positive")
    negative = importance.get("negative_mass")
    if not isinstance(positive, Mapping) or not isinstance(negative, Mapping):
        return []
    named = dict(model.module.named_parameters())
    resolved_parameters: dict[str, torch.Tensor] = {}
    for importance_name in sorted(set(positive) & set(negative)):
        model_name = str(importance_name)
        if model_name not in named and model_name.startswith("module."):
            model_name = model_name[len("module.") :]
        if model_name in named:
            resolved_parameters[str(importance_name)] = named[model_name]
    names = tuple(resolved_parameters)
    if not names:
        return []
    parameters = TensorMap({name: resolved_parameters[name] for name in names})
    scores = TensorMap(
        {
            name: positive[name].detach().to(torch.float64)
            + negative[name].detach().to(torch.float64)
            for name in names
        }
    )
    baseline = dict(_evaluate(model, batches))
    directions = {
        name: ("lower_is_better" if name in {"loss", "perplexity"} else "higher_is_better")
        for name in baseline
    }
    rows: list[Mapping[str, JSONValue]] = [
        {
            "phase_id": phase.phase_id,
            "strategy": "baseline",
            "ratio": 0.0,
            "selected_count": 0,
            "eligible_count": sum(value.numel() for value in parameters.values()),
            "metrics": baseline,
            "damage": {name: 0.0 for name in baseline},
        }
    ]
    for strategy in ("high", "low", "random"):
        plan = PruningPlan(0.1, strategy, "global", 17, "absolute")
        selection = select_pruned_coordinates(scores, plan)
        with PruningContext(parameters, selection):
            metrics = dict(_evaluate(model, batches))
        damage = {
            name: (
                metrics[name] - baseline[name]
                if directions[name] == "lower_is_better"
                else baseline[name] - metrics[name]
            )
            for name in baseline
        }
        rows.append(
            {
                "phase_id": phase.phase_id,
                "strategy": strategy,
                "ratio": plan.ratio,
                "selected_count": selection.selected_count,
                "eligible_count": selection.eligible_count,
                "metrics": metrics,
                "damage": damage,
                "selection_hash": canonical_json_hash(list(selection.coordinate_ids)),
            }
        )
    return rows


def _artifact_store(request: TaskExecutionRequest, root: Path) -> TaskArtifactStore:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    return TaskArtifactStore(root, str(artifacts["output_dir"]))


def _completed(
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
        checkpoint_ref=refs.get("checkpoint_lineage") or refs.get("training_route"),
        message="Stage 4--6 task already committed",
        metadata={"runner": "stage456-v1"},
    )


def _publish(
    request: TaskExecutionRequest,
    store: TaskArtifactStore,
    payloads: Mapping[str, Mapping[str, JSONValue]],
    *,
    source_refs: tuple[str, ...],
) -> TaskRunResult:
    if tuple(payloads) != request.task.artifact_kinds:
        raise ValueError("STAGE456_ARTIFACT_KIND_ORDER_MISMATCH")
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
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        checkpoint_ref=refs.get("checkpoint_lineage") or refs.get("training_route"),
        message="Stage 4--6 task already committed",
        metadata={"runner": "stage456-v1"},
    )


def _route_payloads(
    request: TaskExecutionRequest,
    execution: _ExecutedRoute,
) -> Mapping[str, Mapping[str, JSONValue]]:
    payloads: dict[str, Mapping[str, JSONValue]] = {}
    for kind in request.task.artifact_kinds:
        payloads[kind] = {
            **dict(execution.common_payload),
            "artifact_role": kind,
            "data": _artifact_role_data(kind, execution.common_payload),
        }
    return payloads


def _artifact_role_data(
    kind: str,
    common: Mapping[str, JSONValue],
) -> JSONValue:
    mapping = {
        "training_route": "route_spec",
        "training_routes": "route_spec",
        "checkpoint_lineage": "checkpoint_lineage",
        "training_metrics": "training_metrics",
        "importance_trajectory": "importance_trajectory",
        "importance_trajectory_table": "importance_trajectory",
        "layer_module_summary": "layer_module_summary",
        "route_comparison_table": "route_comparison",
        "importance_reuse_report": "route_comparison",
        "pruning_results": "pruning_results",
    }
    source = mapping.get(kind)
    if source is not None:
        return common.get(source)
    if kind == "stage_report":
        return {
            "route_id": common["route_spec"]["route_id"],  # type: ignore[index]
            "route_lineage_hash": common["route_spec"]["lineage_hash"],  # type: ignore[index]
            "gate_status": "NOT_RUN",
            "local_validation_status": common["local_validation_status"],
        }
    return {"source_schema_version": common["schema_version"]}


def _source_inputs(
    request: TaskExecutionRequest,
    root: Path,
    *,
    include_matrix: bool = True,
    include_route_spec: bool = True,
) -> tuple[tuple[_ResolvedJson, ...], tuple[str, ...]]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    refs = [str(item) for item in orchestration["input_result_refs"]]
    if include_route_spec and orchestration["route_spec_ref"] is not None:
        refs.insert(0, str(orchestration["route_spec_ref"]))
    if include_matrix and orchestration["matrix_ref"] is not None:
        refs.insert(0, str(orchestration["matrix_ref"]))
    paired = orchestration["paired_design"]
    if isinstance(paired, Mapping) and paired.get("enabled") is True:
        mapping_ref = paired.get("mapping_ref")
        if mapping_ref is None:
            raise ValueError("STAGE456_PAIRED_MAPPING_REF_MISSING")
        refs.insert(0, str(mapping_ref))
    if not refs:
        raise _blocked_missing(
            BlockerCode.ASSET_UNAVAILABLE,
            "input_result_refs",
            "派生任务必须声明至少一份 hash-bound 上游产物",
        )
    values: list[_ResolvedJson] = []
    for ref in dict.fromkeys(refs):
        try:
            resolved = _load_json_input(root, ref)
            _validate_input_scope(resolved.value, run_intent=request.config.run_intent)
            values.append(resolved)
        except FileNotFoundError as error:
            raise _blocked_missing(
                BlockerCode.ASSET_UNAVAILABLE,
                "input_result_refs",
                str(error),
                evidence_refs=(ref,),
            ) from error
    return tuple(values), tuple(dict.fromkeys(refs))


def _find_lists(
    inputs: Sequence[_ResolvedJson],
    field_name: str,
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    role_aliases = {
        "importance_trajectory": {
            "importance_trajectory",
            "importance_trajectory_table",
        },
        "evaluation_metrics": {"evaluation_metrics", "evaluation_results"},
        "layer_module_summary": {"layer_module_summary"},
        "topk_sets": {"topk_sets"},
    }
    accepted_roles = role_aliases.get(field_name, {field_name})
    for item in inputs:
        payload = _unwrap_payload(item.value)
        value = payload.get(field_name)
        if isinstance(value, list):
            rows.extend(row for row in value if isinstance(row, Mapping))
        data = payload.get("data")
        if isinstance(data, list) and payload.get("artifact_role") in accepted_roles:
            rows.extend(row for row in data if isinstance(row, Mapping))
    # artifact role 间可能复制同一 common payload；canonical row hash 去重。
    # 排序必须先使用科学身份。若直接按整行 hash 排序，checkpoint 的诊断字段即使
    # 不改变 estimand，也可能让 pretrain/finetune 的行顺序互换，进而污染下游表。
    unique = {canonical_json_hash(dict(row)): dict(row) for row in rows}
    identity_fields = (
        "route_id",
        "phase_id",
        "phase_type",
        "task_id",
        "checkpoint_id",
        "global_step",
        "layer",
        "module",
        "metric",
        "strategy",
        "scope",
        "condition",
        "replicate",
    )

    def sort_key(item: tuple[str, Mapping[str, object]]) -> tuple[str, ...]:
        digest, row = item
        return tuple(
            json.dumps(row.get(field), ensure_ascii=False, sort_keys=True)
            for field in identity_fields
        ) + (digest,)

    return [row for _digest, row in sorted(unique.items(), key=sort_key)]


def _find_mappings(
    inputs: Sequence[_ResolvedJson],
    field_name: str,
) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    for item in inputs:
        payload = _unwrap_payload(item.value)
        value = payload.get(field_name)
        if isinstance(value, Mapping):
            rows.append(dict(value))
    unique = {canonical_json_hash(dict(row)): dict(row) for row in rows}
    return [unique[key] for key in sorted(unique)]


def _normal_ci(values: Sequence[float]) -> Mapping[str, JSONValue]:
    if not values:
        return {"defined": False, "reason": "empty_sample", "n": 0}
    mean = fmean(values)
    if len(values) == 1:
        return {
            "defined": False,
            "reason": "n_less_than_2",
            "n": 1,
            "mean": mean,
        }
    standard_error = stdev(values) / math.sqrt(len(values))
    return {
        "defined": True,
        "reason": None,
        "n": len(values),
        "mean": mean,
        "method": "normal_approximation_95",
        "lower": mean - 1.96 * standard_error,
        "upper": mean + 1.96 * standard_error,
    }


@dataclass(frozen=True, slots=True)
class _RouteExecutionEvidence:
    """一条由 task commit 发现并完成结构复核的真实路线执行证据。"""

    item: _ResolvedJson
    payload: Mapping[str, object]
    route: TrainingRouteSpec
    route_role: str
    terminal_phase_id: str
    terminal_evaluations: tuple[Mapping[str, object], ...]
    coordinate_registry_hash: str


def _route_execution_core_hash(payload: Mapping[str, object]) -> str:
    """复算与 artifact role 无关的路线执行身份。"""

    core = {
        key: value
        for key, value in payload.items()
        if key not in {"artifact_role", "data", "route_execution_hash"}
    }
    return canonical_json_hash(core)


def _route_role(route: TrainingRouteSpec) -> tuple[str, str]:
    supervised = [
        phase
        for phase in route.phases
        if phase.phase_type in {"direct_supervised", "finetune"}
    ]
    if len(supervised) != 1:
        raise ValueError(
            "STAGE6_ROUTE_MUST_HAVE_EXACTLY_ONE_TERMINAL_SUPERVISED_PHASE:"
            f"{route.route_id}"
        )
    phase = supervised[0]
    if phase.phase_type == "direct_supervised":
        if any(item.phase_type == "pretrain" for item in route.phases):
            raise ValueError("STAGE6_DIRECT_ROUTE_CANNOT_CONTAIN_PRETRAIN")
        return "direct_supervised", phase.phase_id
    if not any(item.phase_type == "pretrain" for item in route.phases):
        raise ValueError("STAGE6_FINETUNE_ROUTE_REQUIRES_PRETRAIN")
    return "pretrain_finetune", phase.phase_id


def _validate_evaluation_record(
    row: Mapping[str, object],
    *,
    route: TrainingRouteSpec,
) -> None:
    expected = {
        "schema_version",
        "route_id",
        "route_lineage_hash",
        "phase_id",
        "phase_type",
        "task_id",
        "global_step",
        "split",
        "checkpoint_id",
        "checkpoint_manifest_sha256",
        "evaluator_id",
        "metrics",
        "sample_count",
        "panel_sample_ids_hash",
        "evaluation_contract_hash",
        "record_hash",
    }
    if set(row) != expected or row.get("schema_version") != (
        "stage456-route-evaluation-v1"
    ):
        raise ValueError("STAGE6_EVALUATION_RECORD_FIELDS_OR_VERSION_INVALID")
    if row.get("route_id") != route.route_id or row.get(
        "route_lineage_hash"
    ) != route.lineage_hash:
        raise ValueError("STAGE6_EVALUATION_ROUTE_LINEAGE_MISMATCH")
    phase_id = row.get("phase_id")
    if not isinstance(phase_id, str):
        raise TypeError("STAGE6_EVALUATION_PHASE_ID_INVALID")
    phase = route.phase(phase_id)
    if row.get("phase_type") != phase.phase_type or row.get("task_id") != phase.task_id:
        raise ValueError("STAGE6_EVALUATION_PHASE_CONTRACT_MISMATCH")
    if (
        isinstance(row.get("global_step"), bool)
        or not isinstance(row.get("global_step"), int)
        or int(row["global_step"]) <= 0
        or isinstance(row.get("sample_count"), bool)
        or not isinstance(row.get("sample_count"), int)
        or int(row["sample_count"]) <= 0
    ):
        raise ValueError("STAGE6_EVALUATION_COUNT_INVALID")
    for name in ("split", "checkpoint_id", "evaluator_id"):
        if not isinstance(row.get(name), str) or not row[name]:
            raise TypeError(f"STAGE6_EVALUATION_STRING_INVALID:{name}")
    for name in (
        "checkpoint_manifest_sha256",
        "panel_sample_ids_hash",
        "evaluation_contract_hash",
        "record_hash",
    ):
        _require_hash(row.get(name), field_name=name)
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        raise TypeError("STAGE6_EVALUATION_METRICS_INVALID")
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("STAGE6_EVALUATION_METRIC_NONFINITE_OR_INVALID")
    contract = {
        "schema_version": "stage456-evaluation-contract-v1",
        "task_id": row["task_id"],
        "split": row["split"],
        "metric_names": sorted(str(name) for name in metrics),
        "sample_count": row["sample_count"],
        "panel_sample_ids_hash": row["panel_sample_ids_hash"],
    }
    if row["evaluation_contract_hash"] != canonical_json_hash(contract):
        raise ValueError("STAGE6_EVALUATION_CONTRACT_HASH_MISMATCH")
    without_hash = dict(row)
    declared = without_hash.pop("record_hash")
    if declared != canonical_json_hash(without_hash):
        raise ValueError("STAGE6_EVALUATION_RECORD_HASH_MISMATCH")


def _load_route_execution_evidence(item: _ResolvedJson) -> _RouteExecutionEvidence:
    envelope = item.value
    if envelope.get("schema_version") != "task-output-artifact-v1":
        raise ValueError("STAGE6_ROUTE_MATRIX_REQUIRES_TASK_OUTPUT_ARTIFACT")
    if envelope.get("task_id") not in _ROUTE_TASK_IDS:
        raise ValueError("STAGE6_ROUTE_EXECUTION_PRODUCER_INVALID")
    payload = _unwrap_payload(envelope)
    if payload.get("schema_version") != "stage456-route-execution-v1":
        raise ValueError("STAGE6_INPUT_NOT_ROUTE_EXECUTION")
    declared_execution_hash = _require_hash(
        payload.get("route_execution_hash"), field_name="route_execution_hash"
    )
    if declared_execution_hash != _route_execution_core_hash(payload):
        raise ValueError("STAGE6_ROUTE_EXECUTION_HASH_MISMATCH")
    route = TrainingRouteSpec.from_mapping(_route_candidate(envelope))
    route_result = payload.get("route_result")
    if not isinstance(route_result, Mapping):
        raise TypeError("STAGE6_ROUTE_RESULT_NOT_OBJECT")
    result_without_hash = dict(route_result)
    result_hash = result_without_hash.pop("artifact_hash", None)
    if result_hash != canonical_json_hash(result_without_hash):
        raise ValueError("STAGE6_ROUTE_RESULT_HASH_MISMATCH")
    if (
        route_result.get("status") != "COMPLETE"
        or route_result.get("route_id") != route.route_id
        or route_result.get("route_lineage_hash") != route.lineage_hash
    ):
        raise ValueError("STAGE6_ROUTE_RESULT_NOT_COMPLETE_OR_LINEAGE_MISMATCH")
    route_role, terminal_phase_id = _route_role(route)
    raw_evaluations = payload.get("evaluation_metrics")
    if not isinstance(raw_evaluations, list):
        raise TypeError("STAGE6_ROUTE_EVALUATIONS_NOT_ARRAY")
    evaluations: list[Mapping[str, object]] = []
    for raw in raw_evaluations:
        if not isinstance(raw, Mapping):
            raise TypeError("STAGE6_ROUTE_EVALUATION_NOT_OBJECT")
        _validate_evaluation_record(raw, route=route)
        if raw["phase_id"] == terminal_phase_id:
            evaluations.append(dict(raw))
    if not evaluations:
        raise ValueError("STAGE6_ROUTE_TERMINAL_EVALUATIONS_EMPTY")
    raw_topk = payload.get("topk_sets")
    if not isinstance(raw_topk, list):
        raise TypeError("STAGE6_ROUTE_TOPK_NOT_ARRAY")
    terminal_topk = [
        row
        for row in raw_topk
        if isinstance(row, Mapping) and row.get("phase_id") == terminal_phase_id
    ]
    if len(terminal_topk) != 1:
        raise ValueError("STAGE6_ROUTE_TERMINAL_TOPK_NOT_UNIQUE")
    registry_hash = _require_hash(
        terminal_topk[0].get("coordinate_registry_hash"),
        field_name="coordinate_registry_hash",
    )
    return _RouteExecutionEvidence(
        item=item,
        payload=payload,
        route=route,
        route_role=route_role,
        terminal_phase_id=terminal_phase_id,
        terminal_evaluations=tuple(
            sorted(
                evaluations,
                key=lambda row: (int(row["global_step"]), str(row["record_hash"])),
            )
        ),
        coordinate_registry_hash=registry_hash,
    )


def _route_execution_source(
    evidence: _RouteExecutionEvidence,
) -> dict[str, JSONValue]:
    envelope = evidence.item.value
    checkpoint_lineage = evidence.payload.get("checkpoint_lineage")
    if not isinstance(checkpoint_lineage, list):
        raise TypeError("STAGE6_CHECKPOINT_LINEAGE_NOT_ARRAY")
    base_hashes = {
        row.get("base_initialization_tensor_hash")
        for row in checkpoint_lineage
        if isinstance(row, Mapping)
        and row.get("phase_type") in {"pretrain", "direct_supervised"}
    }
    if len(base_hashes) != 1:
        raise ValueError("STAGE6_BASE_INITIALIZATION_TENSOR_HASH_NOT_UNIQUE")
    base_tensor_hash = _require_hash(
        next(iter(base_hashes)), field_name="base_initialization_tensor_hash"
    )
    return {
        "schema_version": "stage6-route-execution-source-v1",
        "route_id": evidence.route.route_id,
        "route_lineage_hash": evidence.route.lineage_hash,
        "route_role": evidence.route_role,
        "terminal_phase_id": evidence.terminal_phase_id,
        "task_id": str(envelope["task_id"]),
        "artifact_kind": str(envelope["artifact_kind"]),
        "config_hash": str(envelope["config_hash"]),
        "artifact_ref": evidence.item.ref,
        "artifact_hash": evidence.item.source_hash,
        "route_execution_hash": str(evidence.payload["route_execution_hash"]),
        "base_initialization_tensor_hash": base_tensor_hash,
        "coordinate_registry_hash": evidence.coordinate_registry_hash,
        "checkpoint_lineage_hash": canonical_json_hash(checkpoint_lineage),
        "evaluation_record_hashes": [
            str(row["record_hash"]) for row in evidence.terminal_evaluations
        ],
        "importance_trajectory_hash": canonical_json_hash(
            evidence.payload.get("importance_trajectory")
        ),
        "topk_sets_hash": canonical_json_hash(evidence.payload.get("topk_sets")),
        "layer_module_summary_hash": canonical_json_hash(
            evidence.payload.get("layer_module_summary")
        ),
    }


def _route_matrix_payload(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, JSONValue], tuple[str, ...]]:
    inputs, refs = _source_inputs(
        request,
        root,
        include_matrix=False,
        include_route_spec=False,
    )
    # 真实路线矩阵的结构审计属于独立纯核心；runner 只负责解析 task 输入。
    return (
        _build_stage6_route_matrix(
            inputs,
            refs=refs,
            run_intent=request.config.run_intent,
            allowed_task_ids=_ROUTE_TASK_IDS,
        ),
        refs,
    )

    # 以下旧实现暂留作版本迁移对照；上方独立核心是唯一可达执行路径。
    grouped: dict[str, list[_RouteExecutionEvidence]] = {}
    for item in inputs:
        evidence = _load_route_execution_evidence(item)
        grouped.setdefault(evidence.route.lineage_hash, []).append(evidence)
    if len(grouped) != 2:
        raise _blocked_missing(
            BlockerCode.ASSET_UNAVAILABLE,
            "comparable_route_executions",
            "Stage 6 route matrix 必须绑定恰好两条已完成、hash-bound 的路线执行产物",
            evidence_refs=refs,
        )

    selected: list[_RouteExecutionEvidence] = []
    artifact_priority = {"checkpoint_lineage": 0, "training_metrics": 1}
    for route_hash in sorted(grouped):
        candidates = grouped[route_hash]
        execution_hashes = {str(item.payload["route_execution_hash"]) for item in candidates}
        if len(execution_hashes) != 1:
            raise ValueError("STAGE6_DUPLICATE_ROUTE_EXECUTION_CONTENT_DRIFT")
        selected.append(
            min(
                candidates,
                key=lambda item: (
                    artifact_priority.get(
                        str(item.item.value.get("artifact_kind")), 100
                    ),
                    item.item.ref,
                ),
            )
        )
    roles = {item.route_role for item in selected}
    if roles != {"direct_supervised", "pretrain_finetune"}:
        raise ValueError("STAGE6_ROUTE_MATRIX_ROLE_PAIR_INVALID")
    routes = [item.route for item in selected]
    initialization_id = validate_comparable_routes(routes)
    sources = [_route_execution_source(item) for item in selected]
    base_hashes = {str(item["base_initialization_tensor_hash"]) for item in sources}
    if len(base_hashes) != 1:
        raise ValueError("STAGE6_ROUTE_MATRIX_BASE_TENSOR_MISMATCH")
    registry_hashes = {str(item["coordinate_registry_hash"]) for item in sources}
    if len(registry_hashes) != 1:
        raise ValueError("STAGE6_ROUTE_MATRIX_REGISTRY_MISMATCH")
    contract_sets = {
        item.route_role: {
            str(row["evaluation_contract_hash"])
            for row in item.terminal_evaluations
        }
        for item in selected
    }
    if contract_sets["direct_supervised"] != contract_sets["pretrain_finetune"]:
        raise ValueError("STAGE6_ROUTE_MATRIX_EVALUATION_PANEL_OR_METRICS_MISMATCH")
    route_hashes = sorted(grouped)
    matrix_body: dict[str, JSONValue] = {
        "schema_version": "stage456-route-matrix-v1",
        "matrix_id": f"route-matrix-{canonical_json_hash(route_hashes)[:16]}",
        "base_initialization_id": initialization_id,
        "base_initialization_tensor_hash": next(iter(base_hashes)),
        "coordinate_registry_hash": next(iter(registry_hashes)),
        "route_lineage_hashes": route_hashes,
        "route_refs": [str(item["artifact_ref"]) for item in sources],
        "route_execution_sources": sources,
        "evaluation_contract_hashes": sorted(
            contract_sets["direct_supervised"]
        ),
        "paired_by": [
            "task_id",
            "global_step",
            "split",
            "panel_sample_ids_hash",
            "base_initialization_id",
        ],
        "scope": request.config.run_intent,
        "formal_eligible": request.config.run_intent == "formal",
    }
    matrix_body["artifact_hash"] = canonical_json_hash(matrix_body)
    return matrix_body, refs


def _route_matrix_from_inputs(
    inputs: Sequence[_ResolvedJson],
    *,
    run_intent: str,
) -> Mapping[str, object]:
    matrices: dict[str, Mapping[str, object]] = {}
    for item in inputs:
        payload = _unwrap_payload(item.value)
        candidate = payload.get("route_matrix")
        if not isinstance(candidate, Mapping):
            continue
        declared = candidate.get("artifact_hash")
        without_hash = dict(candidate)
        without_hash.pop("artifact_hash", None)
        if declared != canonical_json_hash(without_hash):
            raise ValueError("STAGE6_ROUTE_MATRIX_HASH_MISMATCH")
        matrices[str(declared)] = dict(candidate)
    if len(matrices) != 1:
        raise ValueError("STAGE6_ROUTE_MATRIX_NOT_UNIQUE")
    matrix = next(iter(matrices.values()))
    if (
        matrix.get("schema_version") != "stage456-route-matrix-v1"
        or matrix.get("scope") != run_intent
        or matrix.get("formal_eligible") is not (run_intent == "formal")
    ):
        raise ValueError("STAGE6_ROUTE_MATRIX_SCOPE_OR_VERSION_MISMATCH")
    sources = matrix.get("route_execution_sources")
    if not isinstance(sources, list) or len(sources) != 2:
        raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_COUNT_INVALID")
    return matrix


def _evidences_from_matrix(
    matrix: Mapping[str, object],
    *,
    root: Path,
) -> tuple[_RouteExecutionEvidence, ...]:
    raw_sources = matrix.get("route_execution_sources")
    assert isinstance(raw_sources, list)
    evidences: list[_RouteExecutionEvidence] = []
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise TypeError("STAGE6_ROUTE_MATRIX_SOURCE_NOT_OBJECT")
        reference = source.get("artifact_ref")
        if not isinstance(reference, str):
            raise TypeError("STAGE6_ROUTE_MATRIX_SOURCE_REF_INVALID")
        item = _load_json_input(root, reference)
        if item.source_hash != source.get("artifact_hash"):
            raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_ARTIFACT_HASH_MISMATCH")
        evidence = _load_route_execution_evidence(item)
        if _route_execution_source(evidence) != dict(source):
            raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_LINEAGE_DRIFT")
        evidences.append(evidence)
    if {item.route_role for item in evidences} != {
        "direct_supervised",
        "pretrain_finetune",
    }:
        raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_ROLES_INVALID")
    return tuple(sorted(evidences, key=lambda item: item.route_role))


def _verify_route_evaluation_checkpoints(
    evidence: _RouteExecutionEvidence,
    *,
    root: Path,
) -> tuple[Mapping[str, JSONValue], ...]:
    """重新打开每个评估 checkpoint，复核 commit、bundle 与 registry。"""

    envelope_path = _resolve(root, evidence.item.ref, field_name="route_execution_ref")
    task_output_root = envelope_path.parent.parent
    distributed = evidence.payload.get("distributed_execution")
    if not isinstance(distributed, Mapping) or not isinstance(
        distributed.get("result_rank"), int
    ):
        raise ValueError("STAGE6_ROUTE_DISTRIBUTED_IDENTITY_INVALID")
    rank = int(distributed["result_rank"])
    verified: list[Mapping[str, JSONValue]] = []
    for row in evidence.terminal_evaluations:
        phase_id = str(row["phase_id"])
        checkpoint_root = (
            task_output_root
            / "route-execution"
            / evidence.route.lineage_hash
            / f"rank-{rank:04d}"
            / "phase-checkpoints"
            / phase_id
        )
        if not checkpoint_root.is_dir():
            raise _blocked_missing(
                BlockerCode.ASSET_UNAVAILABLE,
                "route_evaluation_checkpoint",
                f"评估 checkpoint 根目录不可发现：{checkpoint_root}",
                evidence_refs=(evidence.item.ref,),
            )
        try:
            state, commit = CheckpointStore(checkpoint_root).load(
                str(row["checkpoint_id"])
            )
        except FileNotFoundError as error:
            raise _blocked_missing(
                BlockerCode.ASSET_UNAVAILABLE,
                "route_evaluation_checkpoint",
                str(error),
                evidence_refs=(evidence.item.ref,),
            ) from error
        if commit.manifest_sha256 != row["checkpoint_manifest_sha256"]:
            raise ValueError("STAGE6_EVALUATION_CHECKPOINT_MANIFEST_MISMATCH")
        if not isinstance(state, Mapping) or state.get(
            "registry_hash"
        ) != evidence.coordinate_registry_hash:
            raise ValueError("STAGE6_EVALUATION_CHECKPOINT_REGISTRY_MISMATCH")
        verified.append(
            {
                "route_id": evidence.route.route_id,
                "route_role": evidence.route_role,
                "phase_id": phase_id,
                "checkpoint_id": str(row["checkpoint_id"]),
                "checkpoint_manifest_sha256": commit.manifest_sha256,
                "evaluation_record_hash": str(row["record_hash"]),
                "panel_sample_ids_hash": str(row["panel_sample_ids_hash"]),
                "evaluation_contract_hash": str(row["evaluation_contract_hash"]),
                "metrics_hash": canonical_json_hash(row["metrics"]),
                "verified": True,
            }
        )
    return tuple(verified)


def _paired_route_evaluations(
    evidences: Sequence[_RouteExecutionEvidence],
) -> tuple[list[Mapping[str, JSONValue]], list[Mapping[str, JSONValue]]]:
    verified_rows: list[Mapping[str, JSONValue]] = []
    grouped: dict[
        tuple[object, int, str, str], dict[str, Mapping[str, object]]
    ] = {}
    for evidence in evidences:
        for row in evidence.terminal_evaluations:
            verified_rows.append(
                {
                    "schema_version": "stage6-verified-route-evaluation-v1",
                    "route_role": evidence.route_role,
                    "source_artifact_hash": evidence.item.source_hash,
                    "source_route_execution_hash": str(
                        evidence.payload["route_execution_hash"]
                    ),
                    "checkpoint_verified": True,
                    "evaluation_record": dict(row),
                }
            )
            key = (
                row["task_id"],
                int(row["global_step"]),
                str(row["split"]),
                str(row["evaluation_contract_hash"]),
            )
            role_rows = grouped.setdefault(key, {})
            if evidence.route_role in role_rows:
                raise ValueError("STAGE6_EVALUATION_PAIR_ROLE_DUPLICATE")
            role_rows[evidence.route_role] = row
    paired: list[Mapping[str, JSONValue]] = []
    for key in sorted(grouped, key=lambda value: tuple(str(item) for item in value)):
        role_rows = grouped[key]
        if set(role_rows) != {"direct_supervised", "pretrain_finetune"}:
            raise ValueError("STAGE6_EVALUATION_PAIR_INCOMPLETE")
        direct = role_rows["direct_supervised"]
        finetune = role_rows["pretrain_finetune"]
        direct_metrics = direct["metrics"]
        finetune_metrics = finetune["metrics"]
        assert isinstance(direct_metrics, Mapping) and isinstance(finetune_metrics, Mapping)
        if set(direct_metrics) != set(finetune_metrics):
            raise ValueError("STAGE6_EVALUATION_PAIR_METRIC_SET_MISMATCH")
        if direct["panel_sample_ids_hash"] != finetune["panel_sample_ids_hash"]:
            raise ValueError("STAGE6_EVALUATION_PAIR_PANEL_MISMATCH")
        for metric_name in sorted(direct_metrics):
            body: dict[str, JSONValue] = {
                "schema_version": "stage6-paired-route-metric-v1",
                "task_id": direct["task_id"],
                "global_step": direct["global_step"],
                "split": direct["split"],
                "panel_sample_ids_hash": direct["panel_sample_ids_hash"],
                "evaluation_contract_hash": direct["evaluation_contract_hash"],
                "metric_name": str(metric_name),
                "direct_route_id": direct["route_id"],
                "direct_checkpoint_id": direct["checkpoint_id"],
                "direct_record_hash": direct["record_hash"],
                "direct_value": float(direct_metrics[metric_name]),
                "finetune_route_id": finetune["route_id"],
                "finetune_checkpoint_id": finetune["checkpoint_id"],
                "finetune_record_hash": finetune["record_hash"],
                "finetune_value": float(finetune_metrics[metric_name]),
                "delta_finetune_minus_direct": (
                    float(finetune_metrics[metric_name])
                    - float(direct_metrics[metric_name])
                ),
            }
            body["pair_hash"] = canonical_json_hash(body)
            paired.append(body)
    if not paired:
        raise ValueError("STAGE6_PAIRED_ROUTE_METRICS_EMPTY")
    return verified_rows, paired


def _audit_with_hash(value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    body = dict(value)
    body.pop("audit_hash", None)
    body["audit_hash"] = canonical_json_hash(body)
    return body


def _validate_stage6_audit(value: Mapping[str, object]) -> Mapping[str, JSONValue]:
    body = dict(value)
    declared = body.pop("audit_hash", None)
    if (
        value.get("schema_version") != "stage6-route-lineage-audit-v1"
        or declared != canonical_json_hash(body)
    ):
        raise ValueError("STAGE6_ROUTE_AUDIT_INVALID")
    return dict(value)  # type: ignore[return-value]


def _unique_route_audit(inputs: Sequence[_ResolvedJson]) -> Mapping[str, JSONValue]:
    audits: dict[str, Mapping[str, JSONValue]] = {}
    for item in inputs:
        payload = _unwrap_payload(item.value)
        candidate = payload.get("route_audit")
        if not isinstance(candidate, Mapping):
            continue
        audit = _validate_stage6_audit(candidate)
        audits[str(audit["audit_hash"])] = audit
    if len(audits) != 1:
        raise ValueError("STAGE6_ROUTE_AUDIT_NOT_UNIQUE")
    return next(iter(audits.values()))


def _unique_role_data(
    inputs: Sequence[_ResolvedJson],
    role: str,
    *,
    expected_type: type,
) -> object:
    values: list[object] = []
    for item in inputs:
        payload = _unwrap_payload(item.value)
        if payload.get("artifact_role") == role:
            values.append(payload.get("data"))
    if len(values) != 1 or not isinstance(values[0], expected_type):
        raise ValueError(f"STAGE6_ARTIFACT_ROLE_DATA_INVALID:{role}")
    return values[0]


def _stage6_payloads(
    request: TaskExecutionRequest,
    *,
    inputs: Sequence[_ResolvedJson],
    route_audit: Mapping[str, JSONValue],
    data_by_kind: Mapping[str, JSONValue],
) -> Mapping[str, Mapping[str, JSONValue]]:
    source_hashes = sorted({item.source_hash for item in inputs})
    payloads: dict[str, Mapping[str, JSONValue]] = {}
    for kind in request.task.artifact_kinds:
        if kind not in data_by_kind:
            raise ValueError(f"STAGE6_DERIVED_ARTIFACT_ROLE_UNHANDLED:{kind}")
        payloads[kind] = {
            "schema_version": "stage456-derived-artifact-v1",
            "artifact_role": kind,
            "source_hashes": source_hashes,
            "derivation_id": f"stage456-{request.task.task_id}-v2",
            "route_audit": dict(route_audit),
            "data": data_by_kind[kind],
            "local_validation_status": (
                "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
            ),
            "gate_status": "NOT_RUN" if "gate" in kind else None,
        }
    return payloads


def _stage6_evaluate_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs, refs = _source_inputs(
        request,
        root,
        include_matrix=False,
        include_route_spec=False,
    )
    matrix = _stage6_route_matrix_from_inputs(
        inputs, run_intent=request.config.run_intent
    )
    evidences = _stage6_evidences_from_matrix(
        matrix,
        load_artifact=lambda reference: _load_json_input(root, reference),
        allowed_task_ids=_ROUTE_TASK_IDS,
    )
    verification_rows = [
        row
        for evidence in evidences
        for row in _stage6_verify_checkpoints(evidence, workspace_root=root)
    ]
    evaluations, paired = _stage6_paired_route_evaluations(evidences)
    audit = _stage6_audit_with_hash(
        {
            "schema_version": "stage6-route-lineage-audit-v1",
            "audit_stage": "evaluate",
            "matrix_hash": str(matrix["artifact_hash"]),
            "route_execution_sources": matrix["route_execution_sources"],
            "coordinate_registry_hash": str(matrix["coordinate_registry_hash"]),
            "evaluation_contract_hashes": matrix["evaluation_contract_hashes"],
            "checkpoint_verification_hash": canonical_json_hash(verification_rows),
            "verified_checkpoint_count": len(verification_rows),
            "paired_metric_count": len(paired),
        }
    )
    data_by_kind: Mapping[str, JSONValue] = {
        "evaluation_results": evaluations,
        "paired_route_metrics": paired,
    }
    return (
        _stage6_payloads(
            request,
            inputs=inputs,
            route_audit=audit,
            data_by_kind=data_by_kind,
        ),
        refs,
    )


def _stage6_compare_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs, refs = _source_inputs(
        request,
        root,
        include_matrix=False,
        include_route_spec=False,
    )
    parent_audit = _stage6_unique_route_audit(inputs)
    paired = _stage6_unique_role_data(
        inputs,
        "paired_route_metrics",
        expected_type=list,
    )
    assert isinstance(paired, list)
    comparison_rows, confidence = _compare_stage6_paired_metrics(paired)
    audit = _stage6_audit_with_hash(
        {
            key: value
            for key, value in parent_audit.items()
            if key not in {"audit_hash", "audit_stage"}
        }
        | {
            "schema_version": "stage6-route-lineage-audit-v1",
            "audit_stage": "compare",
            "parent_audit_hash": str(parent_audit["audit_hash"]),
            "comparison_table": comparison_rows,
            "comparison_table_hash": canonical_json_hash(comparison_rows),
        }
    )
    data_by_kind: Mapping[str, JSONValue] = {
        "route_comparison_table": comparison_rows,
        "confidence_intervals": confidence,
        "quality_gates": {
            "schema_version": "stage6-comparison-quality-gates-v1",
            "gate_status": "NOT_RUN",
            "reason": "scientific thresholds require independent review",
            "all_pairs_complete": True,
            "defined_metric_count": sum(
                1 for value in confidence.values() if value.get("defined") is True
            ),
        },
    }
    return (
        _stage6_payloads(
            request,
            inputs=inputs,
            route_audit=audit,
            data_by_kind=data_by_kind,
        ),
        refs,
    )


def _stage6_reuse_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs, refs = _source_inputs(
        request,
        root,
        include_matrix=False,
        include_route_spec=False,
    )
    parent_audit = _stage6_unique_route_audit(inputs)
    sources = parent_audit.get("route_execution_sources")
    if not isinstance(sources, list):
        raise ValueError("STAGE6_REUSE_ROUTE_SOURCES_MISSING")
    evidences = _stage6_evidences_from_matrix(
        {"route_execution_sources": sources},
        load_artifact=lambda reference: _load_json_input(root, reference),
        allowed_task_ids=_ROUTE_TASK_IDS,
    )
    trajectory_rows, overlap, layer_differences = _stage6_importance_reuse_tables(
        evidences,
        coordinate_registry_hash=str(parent_audit["coordinate_registry_hash"]),
    )
    audit = _stage6_audit_with_hash(
        {
            key: value
            for key, value in parent_audit.items()
            if key not in {"audit_hash", "audit_stage"}
        }
        | {
            "schema_version": "stage6-route-lineage-audit-v1",
            "audit_stage": "importance_reuse",
            "parent_audit_hash": str(parent_audit["audit_hash"]),
            "importance_reuse_hash": canonical_json_hash(
                {
                    "trajectory": trajectory_rows,
                    "overlap": overlap,
                    "layers": layer_differences,
                }
            ),
        }
    )
    data_by_kind: Mapping[str, JSONValue] = {
        "importance_reuse_table": trajectory_rows,
        "topk_overlap_table": [overlap],
        "layer_module_difference": layer_differences,
    }
    return (
        _stage6_payloads(
            request,
            inputs=inputs,
            route_audit=audit,
            data_by_kind=data_by_kind,
        ),
        refs,
    )

    # 旧的内联实现不可达；保留到迁移窗口结束，避免本波次扩大差异。
    trajectory_rows: list[Mapping[str, JSONValue]] = []
    terminal_topk: dict[str, Mapping[str, object]] = {}
    terminal_layers: dict[str, dict[tuple[str, str], Mapping[str, object]]] = {}
    for evidence in evidences:
        raw_trajectory = evidence.payload.get("importance_trajectory")
        raw_topk = evidence.payload.get("topk_sets")
        raw_layers = evidence.payload.get("layer_module_summary")
        if not isinstance(raw_trajectory, list) or not isinstance(raw_topk, list) or not isinstance(raw_layers, list):
            raise TypeError("STAGE6_REUSE_IMPORTANCE_PAYLOAD_INVALID")
        for row in raw_trajectory:
            if isinstance(row, Mapping) and row.get("phase_id") == evidence.terminal_phase_id:
                if row.get("coordinate_registry_hash") != evidence.coordinate_registry_hash:
                    raise ValueError("STAGE6_REUSE_TRAJECTORY_REGISTRY_MISMATCH")
                trajectory_rows.append({**dict(row), "route_role": evidence.route_role})
        topk_rows = [
            row
            for row in raw_topk
            if isinstance(row, Mapping) and row.get("phase_id") == evidence.terminal_phase_id
        ]
        if len(topk_rows) != 1 or topk_rows[0].get(
            "coordinate_registry_hash"
        ) != evidence.coordinate_registry_hash:
            raise ValueError("STAGE6_REUSE_TOPK_REGISTRY_OR_CARDINALITY_MISMATCH")
        terminal_topk[evidence.route_role] = topk_rows[0]
        layer_rows = {
            (str(row["layer"]), str(row["module"])): row
            for row in raw_layers
            if isinstance(row, Mapping)
            and row.get("phase_id") == evidence.terminal_phase_id
            and row.get("coordinate_registry_hash") == evidence.coordinate_registry_hash
        }
        terminal_layers[evidence.route_role] = layer_rows
    if not trajectory_rows or set(terminal_topk) != {
        "direct_supervised",
        "pretrain_finetune",
    }:
        raise ValueError("STAGE6_REUSE_TERMINAL_IMPORTANCE_EMPTY")
    direct_topk = terminal_topk["direct_supervised"]
    finetune_topk = terminal_topk["pretrain_finetune"]
    overlap = {
        "schema_version": "stage6-topk-overlap-v1",
        "coordinate_registry_hash": str(parent_audit["coordinate_registry_hash"]),
        "direct_phase_id": direct_topk["phase_id"],
        "finetune_phase_id": finetune_topk["phase_id"],
        "jaccard": _topk_jaccard(
            direct_topk.get("parameter_names", []),  # type: ignore[arg-type]
            finetune_topk.get("parameter_names", []),  # type: ignore[arg-type]
        ),
    }
    direct_layers = terminal_layers["direct_supervised"]
    finetune_layers = terminal_layers["pretrain_finetune"]
    if set(direct_layers) != set(finetune_layers):
        raise ValueError("STAGE6_REUSE_LAYER_KEYS_MISMATCH")
    layer_differences: list[Mapping[str, JSONValue]] = []
    for key in sorted(direct_layers):
        direct = direct_layers[key]
        finetune = finetune_layers[key]
        row: dict[str, JSONValue] = {
            "schema_version": "stage6-layer-module-difference-v1",
            "coordinate_registry_hash": str(parent_audit["coordinate_registry_hash"]),
            "layer": key[0],
            "module": key[1],
        }
        for view in ("positive", "negative_mass", "signed", "absolute"):
            direct_value = float(direct[view])
            finetune_value = float(finetune[view])
            row[f"direct_{view}"] = direct_value
            row[f"finetune_{view}"] = finetune_value
            row[f"delta_{view}"] = finetune_value - direct_value
        layer_differences.append(row)
    audit = _audit_with_hash(
        {
            key: value
            for key, value in parent_audit.items()
            if key not in {"audit_hash", "audit_stage"}
        }
        | {
            "schema_version": "stage6-route-lineage-audit-v1",
            "audit_stage": "importance_reuse",
            "parent_audit_hash": str(parent_audit["audit_hash"]),
            "importance_reuse_hash": canonical_json_hash(
                {
                    "trajectory": trajectory_rows,
                    "overlap": overlap,
                    "layers": layer_differences,
                }
            ),
        }
    )
    data_by_kind: Mapping[str, JSONValue] = {
        "importance_reuse_table": trajectory_rows,
        "topk_overlap_table": [overlap],
        "layer_module_difference": layer_differences,
    }
    return (
        _stage6_payloads(
            request,
            inputs=inputs,
            route_audit=audit,
            data_by_kind=data_by_kind,
        ),
        refs,
    )


def _stage6_report_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs, refs = _source_inputs(
        request,
        root,
        include_matrix=False,
        include_route_spec=False,
    )
    audit = _stage6_unique_route_audit(inputs)
    reuse_rows = _stage6_unique_role_data(
        inputs, "importance_reuse_table", expected_type=list
    )
    overlaps = _stage6_unique_role_data(
        inputs, "topk_overlap_table", expected_type=list
    )
    layer_rows = _stage6_unique_role_data(
        inputs, "layer_module_difference", expected_type=list
    )
    assert isinstance(reuse_rows, list) and isinstance(overlaps, list) and isinstance(layer_rows, list)
    comparison = audit.get("comparison_table")
    if not isinstance(comparison, list) or not comparison:
        raise ValueError("STAGE6_REPORT_COMPARISON_TABLE_MISSING")
    source_hashes = sorted({item.source_hash for item in inputs})
    data_by_kind: Mapping[str, JSONValue] = {
        "stage_report": {
            "schema_version": "stage6-route-comparison-report-v1",
            "matrix_hash": str(audit["matrix_hash"]),
            "route_count": len(audit["route_execution_sources"]),  # type: ignore[arg-type]
            "paired_metric_count": int(audit["paired_metric_count"]),
            "comparison_rows": comparison,
            "importance_checkpoint_count": len(reuse_rows),
            "layer_difference_count": len(layer_rows),
            "source_hashes": source_hashes,
            "gate_status": "NOT_RUN",
            "local_validation_status": (
                "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
            ),
        },
        "chart_artifacts": {
            "schema_version": "stage6-chart-source-index-v1",
            "source_hashes": source_hashes,
            "chart_specs": [
                "paired_route_metric_delta",
                "importance_mass_by_checkpoint",
                "topk_overlap",
                "layer_module_difference",
            ],
            "data_hashes": {
                "comparison": canonical_json_hash(comparison),
                "importance_reuse": canonical_json_hash(reuse_rows),
                "topk_overlap": canonical_json_hash(overlaps),
                "layer_difference": canonical_json_hash(layer_rows),
            },
            "manually_edited_numbers": False,
        },
        "gate_summary": {
            "schema_version": "stage6-report-gate-summary-v1",
            "gate_status": "NOT_RUN",
            "reason": "runner publishes evidence; reviewer owns Gate decision",
            "route_audit_hash": str(audit["audit_hash"]),
        },
    }
    return (
        _stage6_payloads(
            request,
            inputs=inputs,
            route_audit=audit,
            data_by_kind=data_by_kind,
        ),
        refs,
    )


def _derived_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    if request.task.task_id == "stage4.route":
        route, route_ref, source_hash = _load_route(request, root)
        estimator = _validate_estimator_binding(request, route, root)
        quadrature = _validate_quadrature_binding(request, root)
        common: dict[str, JSONValue] = {
            "schema_version": "stage456-route-contract-v1",
            "route_spec": route.to_dict(),  # type: ignore[dict-item]
            "route_source_hash": source_hash,
            "checks": [
                "route_hash_verified",
                "phase_dag_acyclic",
                "shared_base_initialization_declared",
                "finetune_parent_lineage_verified",
                "estimator_decision_verified",
                "quadrature_decision_verified",
            ],
            "estimator_decision_audit": dict(estimator),
            "quadrature_decision_audit": dict(quadrature),
            "local_validation_status": (
                "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
            ),
            "gate_status": "NOT_RUN",
        }
        payloads = {
            kind: {**common, "artifact_role": kind, "data": common.get("route_spec")}
            for kind in request.task.artifact_kinds
        }
        refs = [route_ref]
        orchestration = request.config.section("orchestration")
        assert isinstance(orchestration, dict)
        if orchestration["quadrature_decision_ref"] is not None:
            refs.append(str(orchestration["quadrature_decision_ref"]))
        if request.environment.estimator_decision_ref is not None:
            refs.append(request.environment.estimator_decision_ref)
        return payloads, tuple(dict.fromkeys(refs))
    if request.task.task_id == "stage6.route_matrix":
        matrix, refs = _route_matrix_payload(request, root)
        payloads = {
            kind: {
                "schema_version": "stage456-route-matrix-task-v1",
                "artifact_role": kind,
                "route_matrix": dict(matrix),
                "data": (
                    matrix["route_execution_sources"]
                    if kind == "training_routes"
                    else dict(matrix)
                ),
                "local_validation_status": (
                    "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
                ),
                "gate_status": "NOT_RUN",
            }
            for kind in request.task.artifact_kinds
        }
        return payloads, refs

    if request.task.task_id == "stage6.evaluate":
        return _stage6_evaluate_payloads(request, root)
    if request.task.task_id == "stage6.compare":
        return _stage6_compare_payloads(request, root)
    if request.task.task_id == "stage6.importance_reuse":
        return _stage6_reuse_payloads(request, root)
    if request.task.task_id == "stage6.report":
        return _stage6_report_payloads(request, root)

    route_audit: Mapping[str, JSONValue] | None = None
    inputs, refs = _source_inputs(request, root)
    trajectory = _find_lists(inputs, "importance_trajectory")
    layers = _find_lists(inputs, "layer_module_summary")
    evaluations = _find_lists(inputs, "evaluation_metrics")
    topk = _find_lists(inputs, "topk_sets")
    comparisons = _find_mappings(inputs, "route_comparison")
    source_hashes = [item.source_hash for item in inputs]

    if request.task.task_id in {
        "stage4.importance_trajectory",
        "stage5.importance_trajectory",
    }:
        stability: list[Mapping[str, JSONValue]] = []
        ordered_topk = sorted(
            topk,
            key=lambda row: (
                str(row.get("route_id")),
                str(row.get("phase_id")),
            ),
        )
        for left, right in zip(ordered_topk, ordered_topk[1:]):
            stability.append(
                {
                    "left_phase_id": left.get("phase_id"),
                    "right_phase_id": right.get("phase_id"),
                    "jaccard": _topk_jaccard(
                        left.get("parameter_names", []),  # type: ignore[arg-type]
                        right.get("parameter_names", []),  # type: ignore[arg-type]
                    ),
                }
            )
        concentration = [
            {
                "route_id": row.get("route_id"),
                "phase_id": row.get("phase_id"),
                "checkpoint_id": row.get("checkpoint_id"),
                "topk_mass": row.get("topk_mass"),
            }
            for row in trajectory
        ]
        data_by_kind: Mapping[str, JSONValue] = {
            "importance_trajectory_table": trajectory,
            "layer_module_summary": layers,
            "concentration_table": concentration,
            "topk_stability_table": stability,
        }
    elif request.task.task_id == "stage5.checkpoint_analysis":
        data_by_kind = {
            "checkpoint_analysis_table": trajectory,
            "layer_module_summary": layers,
            "heatmap_sources": [
                {
                    "route_id": row.get("route_id"),
                    "phase_id": row.get("phase_id"),
                    "layer": row.get("layer"),
                    "module": row.get("module"),
                    "absolute": row.get("absolute"),
                }
                for row in layers
            ],
        }
    else:  # Stage 4/5 reporting；Stage 6 已由上方严格 lineage 分支处理。
        data_by_kind = {
            "stage_report": {
                "source_hashes": sorted(source_hashes),
                "training_route_count": len(
                    {str(row.get("route_id")) for row in evaluations}
                ),
                "evaluation_row_count": len(evaluations),
                "importance_checkpoint_count": len(trajectory),
                "gate_status": "NOT_RUN",
                "local_validation_status": (
                    "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
                ),
            },
            "chart_artifacts": {
                "source_hashes": sorted(source_hashes),
                "chart_specs": [
                    "training_loss_by_phase",
                    "importance_mass_by_checkpoint",
                    "route_metric_comparison",
                ],
                "manually_edited_numbers": False,
            },
            "gate_summary": {
                "gate_status": "NOT_RUN",
                "reason": "runner publishes evidence; reviewer owns Gate decision",
            },
        }

    payloads: dict[str, Mapping[str, JSONValue]] = {}
    for kind in request.task.artifact_kinds:
        if kind not in data_by_kind:
            raise ValueError(f"STAGE456_DERIVED_ARTIFACT_ROLE_UNHANDLED:{kind}")
        payloads[kind] = {
            "schema_version": "stage456-derived-artifact-v1",
            "artifact_role": kind,
            "source_hashes": sorted(source_hashes),
            "derivation_id": f"stage456-{request.task.task_id}-v1",
            "route_audit": None if route_audit is None else dict(route_audit),
            "data": data_by_kind[kind],
            "local_validation_status": (
                "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
            ),
            "gate_status": "NOT_RUN" if "gate" in kind else None,
        }
    return payloads, refs


@dataclass(slots=True)
class Stage456TaskRunner(TaskRunner):
    """按 ``RunnerKind`` 包装 Stage 4--6 专用逻辑，并可组合已有 fallback。"""

    runner_kind: RunnerKind
    workspace_root: Path
    fallback: TaskRunner | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.runner_kind not in _TASK_IDS_BY_KIND:
            raise ValueError(f"STAGE456_RUNNER_KIND_UNSUPPORTED:{self.runner_kind.value}")
        if self.fallback is not None and self.fallback.runner_kind is not self.runner_kind:
            raise ValueError("STAGE456_FALLBACK_RUNNER_KIND_MISMATCH")

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        supported = _TASK_IDS_BY_KIND[self.runner_kind]
        if request.task.task_id not in supported:
            if self.fallback is not None:
                return self.fallback.run(request)
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.RUNNER_UNAVAILABLE,
                    request.task.task_id,
                    f"Stage456TaskRunner 不负责 {request.task.task_id}",
                    False,
                )
            )
        store = _artifact_store(request, self.workspace_root)
        existing = _completed(request, store)
        if existing is not None:
            return existing
        if self.runner_kind is RunnerKind.ROUTE_TRAINING:
            launcher = request.config.section("launcher")
            assert isinstance(launcher, dict)
            world_size = int(launcher["world_size"])
            if world_size == 1:
                execution = _run_route(request, self.workspace_root)
                return _publish(
                    request,
                    store,
                    _route_payloads(request, execution),
                    source_refs=execution.source_refs,
                )

            # 多 rank 路线由 torchrun 启动同一命令。每个 rank 拥有独立 checkpoint
            # 与 phase-result commit；模型梯度/finite/U 充分量由 TrainingEngine 的
            # reducer 归约。只有 rank 0 发布任务级聚合产物。
            try:
                from ..runtime.distributed_training import TorchDDPTrainingExecutor

                executor = TorchDDPTrainingExecutor.from_environment(
                    backend=str(launcher["backend"])
                )
            except RuntimeError as error:
                raise _blocked_missing(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    "torchrun_environment",
                    str(error),
                ) from error
            try:
                if executor.spec.world_size != world_size:
                    raise ValueError("STAGE456_TORCHRUN_WORLD_SIZE_MISMATCH")
                execution = _run_route(
                    request,
                    self.workspace_root,
                    rank=executor.spec.rank,
                    world_size=executor.spec.world_size,
                    distributed_executor=executor,
                    build_payload=executor.spec.rank == 0,
                )
                executor.barrier()
                published: TaskRunResult | None = None
                publication_error: BaseException | None = None
                if executor.spec.rank == 0:
                    try:
                        published = _publish(
                            request,
                            store,
                            _route_payloads(request, execution),
                            source_refs=execution.source_refs,
                        )
                    except BaseException as error:  # 先通知其他 rank，再由 rank 0 重抛。
                        publication_error = error
                successful_publishers = executor.reducer.sum_int(
                    0 if publication_error is not None else 1
                )
                if successful_publishers != world_size:
                    if publication_error is not None:
                        raise publication_error
                    raise RuntimeError("STAGE456_DISTRIBUTED_RANK0_PUBLISH_FAILED")
                if executor.spec.rank == 0:
                    assert published is not None
                    return published
                discovered = _completed(request, store)
                if discovered is None:
                    raise RuntimeError("STAGE456_DISTRIBUTED_TASK_COMMIT_MISSING")
                return discovered
            finally:
                executor.close()
        payloads, source_refs = _derived_payloads(request, self.workspace_root)
        return _publish(
            request,
            store,
            payloads,
            source_refs=source_refs,
        )


def build_stage456_runner_overrides(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
) -> Mapping[RunnerKind, Stage456TaskRunner]:
    """构造供默认 runtime 工厂覆盖的组合 runner 映射。

    ``TaskRuntime`` 有意禁止同一 ``RunnerKind`` 被静默重复注册。因此统一工厂应先
    构造原 runner，再把它作为 ``fallback`` 传入本函数，并注册返回的组合对象。
    Stage 4--6 之外的任务会原样委托，不会因为本适配层而失去原执行路径。
    """

    root = Path(workspace_root).resolve()
    fallback_map = {} if fallbacks is None else dict(fallbacks)
    return MappingProxyType(
        {
            kind: Stage456TaskRunner(kind, root, fallback_map.get(kind))
            for kind in _TASK_IDS_BY_KIND
        }
    )


__all__ = [
    "Stage456TaskRunner",
    "build_stage456_runner_overrides",
]
