"""Stage 7 从权威训练/路线产物恢复模型与重要性状态。

剪枝任务不能从 task ID、目录名或“最新文件”猜测模型。上游必须在功能 payload 中
冻结 workspace 相对 checkpoint commit 引用、排除墙钟字段的稳定 commit identity，
以及安全 TensorBundle manifest。这里依次复核这三层后才返回状态：

``route task commit -> checkpoint commit -> immutable TensorBundle``。

物理 commit 的 ``committed_at`` 仍由 :class:`CheckpointStore` 校验和读取，但不会进入
跨干净根比较的科学身份。未知 schema、多个同深终点、禁用重要性的终点与任一 hash
漂移都会 fail-closed。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..runtime.tensor_bundle import load_tensor_bundle
from .artifact_lineage import (
    LoadedInputArtifact,
    load_input_artifact,
    safe_workspace_path,
)
from .routes import TrainingRouteSpec


@dataclass(frozen=True, slots=True)
class Stage7TrainingSource:
    """Stage 7 所需的训练终点、模型 checkpoint 与在线累计器统一视图。"""

    document: LoadedInputArtifact
    training_result: Mapping[str, object]
    checkpoint_state: Mapping[str, object]
    checkpoint_identity_hash: str
    importance_state_hash: str
    accumulator: Mapping[str, object]


def validate_embedded_artifact_hash(
    value: Mapping[str, object],
    *,
    error_code: str,
) -> None:
    """复算带 ``artifact_hash`` 的嵌套 canonical 对象。"""

    declared = value.get("artifact_hash")
    if not isinstance(declared, str):
        raise ValueError(f"{error_code}_MISSING")
    payload = dict(value)
    payload.pop("artifact_hash")
    if canonical_json_hash(payload) != declared:
        raise ValueError(f"{error_code}_MISMATCH")


def load_checkpoint_state_from_commit(
    root: Path,
    *,
    commit_ref: str,
    expected_identity_sha256: str,
    expected_bundle_manifest_sha256: str,
) -> tuple[Mapping[str, object], LoadedInputArtifact]:
    """复核稳定 commit identity，并从安全 bundle 加载训练状态。

    Args:
        root: workspace 物理根。
        commit_ref: workspace 相对 runtime checkpoint commit 引用。
        expected_identity_sha256: 对 commit 去除 ``committed_at`` 后的 canonical hash。
        expected_bundle_manifest_sha256: 上游功能 payload 冻结的 bundle manifest。
    """

    commit_path = safe_workspace_path(root, commit_ref, field="checkpoint_commit_ref")
    commit_wire = load_canonical_json(commit_path)
    if not isinstance(commit_wire, dict):
        raise ValueError("STAGE7_CHECKPOINT_COMMIT_NOT_OBJECT")
    identity = {
        key: value for key, value in commit_wire.items() if key != "committed_at"
    }
    if canonical_json_hash(identity) != expected_identity_sha256:
        raise ValueError("STAGE7_CHECKPOINT_COMMIT_IDENTITY_HASH_MISMATCH")
    loaded = load_input_artifact(root, commit_ref)
    if loaded.value.get("schema_version") != "runtime.checkpoint-commit.v1":
        raise ValueError("STAGE7_CHECKPOINT_REF_NOT_RUNTIME_COMMIT")
    object_relative = loaded.value.get("object_relative_path")
    if not isinstance(object_relative, str):
        raise TypeError("STAGE7_CHECKPOINT_OBJECT_REF_NOT_STRING")
    bundle_root = safe_workspace_path(
        commit_path.parent.parent.resolve(),
        object_relative,
        field="checkpoint_object_relative_path",
    )
    state, bundle = load_tensor_bundle(bundle_root)
    if bundle.manifest_sha256 != loaded.value.get("bundle_manifest_sha256"):
        raise ValueError("STAGE7_CHECKPOINT_OBJECT_MANIFEST_MISMATCH")
    if bundle.manifest_sha256 != expected_bundle_manifest_sha256:
        raise ValueError("STAGE7_CHECKPOINT_DECLARED_BUNDLE_HASH_MISMATCH")
    if not isinstance(state, Mapping) or state.get(
        "schema_version"
    ) != "training-checkpoint-state-v1":
        raise ValueError("STAGE7_CHECKPOINT_STATE_SCHEMA_MISMATCH")
    return state, loaded


def _phase_depth(route: TrainingRouteSpec, phase_id: str) -> int:
    """计算 phase 到根的边数，并防御性拒绝循环。"""

    seen: set[str] = set()
    current = route.phase(phase_id)
    value = 0
    while current.parent_phase_id is not None:
        if current.phase_id in seen:
            raise ValueError("STAGE7_ROUTE_PHASE_CYCLE")
        seen.add(current.phase_id)
        value += 1
        current = route.phase(current.parent_phase_id)
    return value


def _select_terminal_phase(route: TrainingRouteSpec) -> str:
    """选择唯一最深终点；并列时不猜测 direct/finetune 的优先级。"""

    parent_ids = {
        phase.parent_phase_id
        for phase in route.phases
        if phase.parent_phase_id is not None
    }
    terminal_ids = [
        phase.phase_id for phase in route.phases if phase.phase_id not in parent_ids
    ]
    maximum_depth = max(_phase_depth(route, phase_id) for phase_id in terminal_ids)
    selected = [
        phase_id
        for phase_id in terminal_ids
        if _phase_depth(route, phase_id) == maximum_depth
    ]
    if len(selected) != 1:
        raise ValueError(
            "STAGE7_ROUTE_TERMINAL_PHASE_AMBIGUOUS:" + ",".join(sorted(selected))
        )
    return selected[0]


def _lineage_checkpoint_identity(
    row: Mapping[str, object],
) -> tuple[str, str, str]:
    """读取 Stage456 冻结的稳定 checkpoint 三元组。

    旧字段 ``physical_checkpoint_manifest_sha256`` 仅作为一致性副本接受；稳定消费
    必须存在新的 ``bundle_manifest_sha256`` 与 ``commit_identity_sha256``。
    """

    commit_ref = row.get("physical_checkpoint_commit_ref")
    identity_hash = row.get("commit_identity_sha256")
    manifest_hash = row.get("bundle_manifest_sha256")
    if not isinstance(commit_ref, str):
        raise ValueError("STAGE7_ROUTE_CHECKPOINT_COMMIT_REF_INVALID")
    for field_name, value in (
        ("COMMIT_IDENTITY", identity_hash),
        ("BUNDLE_MANIFEST", manifest_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"STAGE7_ROUTE_{field_name}_HASH_INVALID")
    legacy_manifest = row.get("physical_checkpoint_manifest_sha256")
    if legacy_manifest is not None and legacy_manifest != manifest_hash:
        raise ValueError("STAGE7_ROUTE_CHECKPOINT_MANIFEST_ALIAS_MISMATCH")
    return commit_ref, identity_hash, manifest_hash


def load_route_training_source(
    root: Path,
    document: LoadedInputArtifact,
) -> Stage7TrainingSource:
    """沿 Stage 4--6 route lineage 恢复唯一最深终点 checkpoint。"""

    payload = document.value
    if payload.get("schema_version") != "stage456-route-execution-v1":
        raise ValueError("STAGE7_ROUTE_SOURCE_SCHEMA_MISMATCH")
    route_wire = payload.get("route_spec")
    result_wire = payload.get("route_result")
    lineage_wire = payload.get("checkpoint_lineage")
    if not isinstance(route_wire, Mapping):
        raise TypeError("STAGE7_ROUTE_SPEC_NOT_OBJECT")
    if not isinstance(result_wire, Mapping):
        raise TypeError("STAGE7_ROUTE_RESULT_NOT_OBJECT")
    if not isinstance(lineage_wire, list) or not all(
        isinstance(item, Mapping) for item in lineage_wire
    ):
        raise TypeError("STAGE7_ROUTE_CHECKPOINT_LINEAGE_NOT_OBJECT_ARRAY")

    route = TrainingRouteSpec.from_mapping(route_wire)
    validate_embedded_artifact_hash(
        result_wire,
        error_code="STAGE7_ROUTE_RESULT_HASH",
    )
    if (
        result_wire.get("schema_version") != "training-route-result-v1"
        or result_wire.get("status") != "COMPLETE"
        or result_wire.get("route_id") != route.route_id
        or result_wire.get("route_lineage_hash") != route.lineage_hash
    ):
        raise ValueError("STAGE7_ROUTE_RESULT_IDENTITY_MISMATCH")
    phase_results = result_wire.get("phase_results")
    if not isinstance(phase_results, Mapping):
        raise TypeError("STAGE7_ROUTE_PHASE_RESULTS_NOT_OBJECT")

    phase_id = _select_terminal_phase(route)
    phase = route.phase(phase_id)
    if not phase.importance_enabled:
        raise ValueError("STAGE7_ROUTE_TERMINAL_IMPORTANCE_DISABLED")
    phase_result = phase_results.get(phase_id)
    if not isinstance(phase_result, Mapping):
        raise ValueError("STAGE7_ROUTE_TERMINAL_RESULT_MISSING")
    validate_embedded_artifact_hash(
        phase_result,
        error_code="STAGE7_ROUTE_PHASE_RESULT_HASH",
    )
    training_result = phase_result.get("training_result")
    if not isinstance(training_result, Mapping):
        raise TypeError("STAGE7_ROUTE_TRAINING_RESULT_NOT_OBJECT")
    validate_embedded_artifact_hash(
        training_result,
        error_code="STAGE7_ROUTE_TRAINING_RESULT_HASH",
    )
    checkpoint_id = phase_result.get("physical_checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError("STAGE7_ROUTE_PHYSICAL_CHECKPOINT_ID_INVALID")
    state_wire = training_result.get("state")
    if not isinstance(state_wire, Mapping) or state_wire.get(
        "last_checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("STAGE7_ROUTE_TRAINING_LAST_CHECKPOINT_MISMATCH")

    matches: Sequence[Mapping[str, object]] = tuple(
        item
        for item in lineage_wire
        if item.get("phase_id") == phase_id
        and item.get("physical_checkpoint_id") == checkpoint_id
        and item.get("route_lineage_hash") == route.lineage_hash
    )
    if len(matches) != 1:
        raise ValueError("STAGE7_ROUTE_CHECKPOINT_LINEAGE_NOT_UNIQUE")
    commit_ref, identity_hash, manifest_hash = _lineage_checkpoint_identity(matches[0])
    checkpoint_state, checkpoint_document = load_checkpoint_state_from_commit(
        root,
        commit_ref=commit_ref,
        expected_identity_sha256=identity_hash,
        expected_bundle_manifest_sha256=manifest_hash,
    )
    if checkpoint_document.value.get("checkpoint_id") != checkpoint_id:
        raise ValueError("STAGE7_ROUTE_CHECKPOINT_ID_MISMATCH")
    accumulator = checkpoint_state.get("importance")
    if not isinstance(accumulator, Mapping):
        raise ValueError("STAGE7_ROUTE_CHECKPOINT_IMPORTANCE_STATE_MISSING")
    return Stage7TrainingSource(
        document=document,
        training_result=training_result,
        checkpoint_state=checkpoint_state,
        checkpoint_identity_hash=identity_hash,
        importance_state_hash=manifest_hash,
        accumulator=accumulator,
    )


__all__ = [
    "Stage7TrainingSource",
    "load_checkpoint_state_from_commit",
    "load_route_training_source",
    "validate_embedded_artifact_hash",
]
