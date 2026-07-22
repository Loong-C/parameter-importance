"""Stage 6 路线比较的纯证据核心。

本模块刻意不负责 task catalog、配置解析或 artifact 发布；这些编排职责留在
``stage456_task_runners``。这里仅处理四类可独立测试的严格语义：

1. 从已提交的 route-execution task artifact 复核训练、checkpoint、评估与参数
   registry lineage；
2. 由一条 direct-supervised 与一条 pretrain→finetune 执行证据冻结比较矩阵；
3. 重新打开实际 checkpoint bundle，并把固定 panel 上的 metric 记录组成配对行；
4. 从已验证的配对行与重要性轨迹确定性生成比较/复用源表。

所有 hash 均覆盖 canonical JSON。缺失的已声明 checkpoint 属于可重试外部状态，
返回结构化 ``TaskBlockedError``；内容 hash、路线身份或统计单位漂移则 fail-closed。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path, PurePosixPath
from statistics import fmean, stdev
from typing import Callable, Mapping, Protocol, Sequence

from ..contracts.jsonio import JSONValue, canonical_json_hash
from ..runtime.checkpoint import CheckpointStore
from ..runtime.task_runtime import BlockerCode, TaskBlockedError, TaskBlocker
from .routes import TrainingRouteSpec, validate_comparable_routes


class ArtifactInput(Protocol):
    """runner 已验证 commit/object 关系后的最小只读输入协议。"""

    ref: str
    source_hash: str
    value: Mapping[str, object]


LoadArtifact = Callable[[str], ArtifactInput]


@dataclass(frozen=True, slots=True)
class RouteExecutionEvidence:
    """一条由 task commit 发现并完成结构复核的真实路线执行证据。"""

    item: ArtifactInput
    payload: Mapping[str, object]
    route: TrainingRouteSpec
    route_role: str
    terminal_phase_id: str
    terminal_evaluations: tuple[Mapping[str, object], ...]
    coordinate_registry_hash: str


def _require_hash(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"STAGE6_SHA256_INVALID:{field_name}")
    return value


def _resolve(root: Path, value: object, *, field_name: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE6_LOGICAL_PATH_INVALID:{field_name}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"STAGE6_PATH_ESCAPE:{field_name}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"STAGE6_PATH_ESCAPE:{field_name}") from error
    return candidate


def _blocked_checkpoint(message: str, *, evidence_ref: str) -> TaskBlockedError:
    return TaskBlockedError(
        TaskBlocker(
            BlockerCode.ASSET_UNAVAILABLE,
            "route_evaluation_checkpoint",
            message,
            True,
            (evidence_ref,),
        )
    )


def _unwrap_payload(value: Mapping[str, object]) -> Mapping[str, object]:
    if value.get("schema_version") != "task-output-artifact-v1":
        return value
    payload = value.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("STAGE6_TASK_OUTPUT_PAYLOAD_NOT_OBJECT")
    return payload


def _route_candidate(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = _unwrap_payload(value)
    if payload.get("schema_version") == "training-route-v1":
        return payload
    candidate = payload.get("route_spec")
    if isinstance(candidate, Mapping) and candidate.get("schema_version") == (
        "training-route-v1"
    ):
        return candidate
    raise ValueError("STAGE6_INPUT_DOES_NOT_CONTAIN_TRAINING_ROUTE")


def route_execution_core_hash(payload: Mapping[str, object]) -> str:
    """复算与 artifact role/data 视图无关的路线执行身份。"""

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


def load_route_execution_evidence(
    item: ArtifactInput,
    *,
    allowed_task_ids: frozenset[str],
) -> RouteExecutionEvidence:
    envelope = item.value
    if envelope.get("schema_version") != "task-output-artifact-v1":
        raise ValueError("STAGE6_ROUTE_MATRIX_REQUIRES_TASK_OUTPUT_ARTIFACT")
    if envelope.get("task_id") not in allowed_task_ids:
        raise ValueError("STAGE6_ROUTE_EXECUTION_PRODUCER_INVALID")
    payload = _unwrap_payload(envelope)
    if payload.get("schema_version") != "stage456-route-execution-v1":
        raise ValueError("STAGE6_INPUT_NOT_ROUTE_EXECUTION")
    declared_execution_hash = _require_hash(
        payload.get("route_execution_hash"), field_name="route_execution_hash"
    )
    if declared_execution_hash != route_execution_core_hash(payload):
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
    return RouteExecutionEvidence(
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


def route_execution_source(evidence: RouteExecutionEvidence) -> dict[str, JSONValue]:
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


def build_route_matrix(
    inputs: Sequence[ArtifactInput],
    *,
    refs: Sequence[str],
    run_intent: str,
    allowed_task_ids: frozenset[str],
) -> Mapping[str, JSONValue]:
    grouped: dict[str, list[RouteExecutionEvidence]] = {}
    for item in inputs:
        evidence = load_route_execution_evidence(
            item, allowed_task_ids=allowed_task_ids
        )
        grouped.setdefault(evidence.route.lineage_hash, []).append(evidence)
    if len(grouped) != 2:
        raise TaskBlockedError(
            TaskBlocker(
                BlockerCode.ASSET_UNAVAILABLE,
                "comparable_route_executions",
                "Stage 6 route matrix 必须绑定恰好两条已完成、hash-bound 的路线执行产物",
                True,
                tuple(refs),
            )
        )
    selected: list[RouteExecutionEvidence] = []
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
    if {item.route_role for item in selected} != {
        "direct_supervised",
        "pretrain_finetune",
    }:
        raise ValueError("STAGE6_ROUTE_MATRIX_ROLE_PAIR_INVALID")
    initialization_id = validate_comparable_routes([item.route for item in selected])
    sources = [route_execution_source(item) for item in selected]
    base_hashes = {str(item["base_initialization_tensor_hash"]) for item in sources}
    registry_hashes = {str(item["coordinate_registry_hash"]) for item in sources}
    if len(base_hashes) != 1:
        raise ValueError("STAGE6_ROUTE_MATRIX_BASE_TENSOR_MISMATCH")
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
    matrix: dict[str, JSONValue] = {
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
        "scope": run_intent,
        "formal_eligible": run_intent == "formal",
    }
    matrix["artifact_hash"] = canonical_json_hash(matrix)
    return matrix


def route_matrix_from_inputs(
    inputs: Sequence[ArtifactInput],
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


def evidences_from_matrix(
    matrix: Mapping[str, object],
    *,
    load_artifact: LoadArtifact,
    allowed_task_ids: frozenset[str],
) -> tuple[RouteExecutionEvidence, ...]:
    raw_sources = matrix.get("route_execution_sources")
    if not isinstance(raw_sources, list):
        raise ValueError("STAGE6_ROUTE_MATRIX_SOURCES_MISSING")
    evidences: list[RouteExecutionEvidence] = []
    for source in raw_sources:
        if not isinstance(source, Mapping) or not isinstance(
            source.get("artifact_ref"), str
        ):
            raise TypeError("STAGE6_ROUTE_MATRIX_SOURCE_INVALID")
        item = load_artifact(str(source["artifact_ref"]))
        if item.source_hash != source.get("artifact_hash"):
            raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_ARTIFACT_HASH_MISMATCH")
        evidence = load_route_execution_evidence(
            item, allowed_task_ids=allowed_task_ids
        )
        if route_execution_source(evidence) != dict(source):
            raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_LINEAGE_DRIFT")
        evidences.append(evidence)
    if {item.route_role for item in evidences} != {
        "direct_supervised",
        "pretrain_finetune",
    }:
        raise ValueError("STAGE6_ROUTE_MATRIX_SOURCE_ROLES_INVALID")
    return tuple(sorted(evidences, key=lambda item: item.route_role))


def verify_route_evaluation_checkpoints(
    evidence: RouteExecutionEvidence,
    *,
    workspace_root: Path,
) -> tuple[Mapping[str, JSONValue], ...]:
    """重新打开每个评估 checkpoint，复核 commit、bundle 与 registry。"""

    envelope_path = _resolve(
        workspace_root, evidence.item.ref, field_name="route_execution_ref"
    )
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
            raise _blocked_checkpoint(
                f"评估 checkpoint 根目录不可发现：{checkpoint_root}",
                evidence_ref=evidence.item.ref,
            )
        try:
            state, commit = CheckpointStore(checkpoint_root).load(
                str(row["checkpoint_id"])
            )
        except FileNotFoundError as error:
            raise _blocked_checkpoint(
                str(error), evidence_ref=evidence.item.ref
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


def paired_route_evaluations(
    evidences: Sequence[RouteExecutionEvidence],
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
        if not isinstance(direct_metrics, Mapping) or not isinstance(
            finetune_metrics, Mapping
        ):
            raise TypeError("STAGE6_EVALUATION_PAIR_METRICS_INVALID")
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


def compare_paired_metrics(
    paired: Sequence[object],
) -> tuple[list[Mapping[str, JSONValue]], Mapping[str, Mapping[str, JSONValue]]]:
    metric_values: dict[str, list[float]] = {}
    metric_pair_hashes: dict[str, list[str]] = {}
    for raw in paired:
        if not isinstance(raw, Mapping):
            raise TypeError("STAGE6_PAIRED_METRIC_NOT_OBJECT")
        without_hash = dict(raw)
        declared = without_hash.pop("pair_hash", None)
        if declared != canonical_json_hash(without_hash):
            raise ValueError("STAGE6_PAIRED_METRIC_HASH_MISMATCH")
        metric_name = raw.get("metric_name")
        delta = raw.get("delta_finetune_minus_direct")
        if (
            not isinstance(metric_name, str)
            or isinstance(delta, bool)
            or not isinstance(delta, (int, float))
            or not math.isfinite(float(delta))
        ):
            raise ValueError("STAGE6_PAIRED_METRIC_VALUE_INVALID")
        metric_values.setdefault(metric_name, []).append(float(delta))
        metric_pair_hashes.setdefault(metric_name, []).append(str(declared))
    if not metric_values:
        raise ValueError("STAGE6_COMPARE_HAS_NO_PAIRED_METRICS")
    rows: list[Mapping[str, JSONValue]] = []
    confidence: dict[str, Mapping[str, JSONValue]] = {}
    for metric_name in sorted(metric_values):
        values = metric_values[metric_name]
        interval = _normal_ci(values)
        confidence[metric_name] = interval
        rows.append(
            {
                "schema_version": "stage6-route-comparison-row-v1",
                "metric_name": metric_name,
                "pair_count": len(values),
                "mean_delta_finetune_minus_direct": fmean(values),
                "confidence_interval": dict(interval),
                "paired_source_hash": canonical_json_hash(
                    sorted(metric_pair_hashes[metric_name])
                ),
            }
        )
    return rows, confidence


def importance_reuse_tables(
    evidences: Sequence[RouteExecutionEvidence],
    *,
    coordinate_registry_hash: str,
) -> tuple[
    list[Mapping[str, JSONValue]],
    list[Mapping[str, JSONValue]],
    list[Mapping[str, JSONValue]],
]:
    trajectory_rows: list[Mapping[str, JSONValue]] = []
    terminal_topk: dict[str, Mapping[str, object]] = {}
    terminal_layers: dict[str, dict[tuple[str, str], Mapping[str, object]]] = {}
    for evidence in evidences:
        raw_trajectory = evidence.payload.get("importance_trajectory")
        raw_topk = evidence.payload.get("topk_sets")
        raw_layers = evidence.payload.get("layer_module_summary")
        if not isinstance(raw_trajectory, list) or not isinstance(
            raw_topk, list
        ) or not isinstance(raw_layers, list):
            raise TypeError("STAGE6_REUSE_IMPORTANCE_PAYLOAD_INVALID")
        for row in raw_trajectory:
            if isinstance(row, Mapping) and row.get(
                "phase_id"
            ) == evidence.terminal_phase_id:
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
        terminal_layers[evidence.route_role] = {
            (str(row["layer"]), str(row["module"])): row
            for row in raw_layers
            if isinstance(row, Mapping)
            and row.get("phase_id") == evidence.terminal_phase_id
            and row.get("coordinate_registry_hash") == evidence.coordinate_registry_hash
        }
    if not trajectory_rows or set(terminal_topk) != {
        "direct_supervised",
        "pretrain_finetune",
    }:
        raise ValueError("STAGE6_REUSE_TERMINAL_IMPORTANCE_EMPTY")
    direct_topk = terminal_topk["direct_supervised"]
    finetune_topk = terminal_topk["pretrain_finetune"]
    direct_names = set(str(name) for name in direct_topk.get("parameter_names", []))
    finetune_names = set(str(name) for name in finetune_topk.get("parameter_names", []))
    union = direct_names | finetune_names
    overlap: Mapping[str, JSONValue] = {
        "schema_version": "stage6-topk-overlap-v1",
        "coordinate_registry_hash": coordinate_registry_hash,
        "direct_phase_id": str(direct_topk["phase_id"]),
        "finetune_phase_id": str(finetune_topk["phase_id"]),
        "jaccard": {
            "defined": bool(union),
            "reason": None if union else "both_topk_sets_empty",
            "value": None if not union else len(direct_names & finetune_names) / len(union),
        },
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
            "coordinate_registry_hash": coordinate_registry_hash,
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
    return trajectory_rows, [overlap], layer_differences


def audit_with_hash(value: Mapping[str, JSONValue]) -> Mapping[str, JSONValue]:
    body = dict(value)
    body.pop("audit_hash", None)
    body["audit_hash"] = canonical_json_hash(body)
    return body


def validate_stage6_audit(value: Mapping[str, object]) -> Mapping[str, JSONValue]:
    body = dict(value)
    declared = body.pop("audit_hash", None)
    if (
        value.get("schema_version") != "stage6-route-lineage-audit-v1"
        or declared != canonical_json_hash(body)
    ):
        raise ValueError("STAGE6_ROUTE_AUDIT_INVALID")
    return dict(value)  # type: ignore[return-value]


def unique_route_audit(inputs: Sequence[ArtifactInput]) -> Mapping[str, JSONValue]:
    audits: dict[str, Mapping[str, JSONValue]] = {}
    for item in inputs:
        payload = _unwrap_payload(item.value)
        candidate = payload.get("route_audit")
        if not isinstance(candidate, Mapping):
            continue
        audit = validate_stage6_audit(candidate)
        audits[str(audit["audit_hash"])] = audit
    if len(audits) != 1:
        raise ValueError("STAGE6_ROUTE_AUDIT_NOT_UNIQUE")
    return next(iter(audits.values()))


def unique_role_data(
    inputs: Sequence[ArtifactInput],
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


__all__ = [
    "ArtifactInput",
    "RouteExecutionEvidence",
    "audit_with_hash",
    "build_route_matrix",
    "compare_paired_metrics",
    "evidences_from_matrix",
    "importance_reuse_tables",
    "paired_route_evaluations",
    "route_execution_core_hash",
    "route_matrix_from_inputs",
    "unique_role_data",
    "unique_route_audit",
    "validate_stage6_audit",
    "verify_route_evaluation_checkpoints",
]
