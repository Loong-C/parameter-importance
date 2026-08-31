"""Stage 7--9 的 hash-bound 输入与分析 claim lineage。

本模块只处理“某个逻辑引用究竟指向什么、它声明了哪些权威祖先”这类跨阶段
问题，不包含剪枝、消融或统计公式。集中这层契约有三个目的：

* task runner 不扫描目录、不猜测相邻文件，只沿已进入 artifact hash 的
  ``source_refs`` 递归；
* runtime checkpoint、task-output commit 与 direct canonical artifact 使用各自严格
  字段集合，legacy/损坏对象不能混入 formal 输入；
* Stage 8/9 报告中的数字只是 claim。只有从祖先闭包重新构造出的冻结源与报告
  source binding 完全一致时，后续 ETL 才接受该报告。

所有路径都是 workspace 相对 POSIX 引用。函数不会联网，也不会把绝对物理根写入
科学 artifact。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from ..analysis.pipeline import BoundSourceTable
from ..analysis.report import AnalysisReport
from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..runtime.task_runtime import TaskExecutionRequest
from ..runtime.tensor_bundle import load_tensor_bundle
from .ablation import AblationMatrix


@dataclass(frozen=True, slots=True)
class LoadedInputArtifact:
    """一个已经通过 canonical/hash 检查的输入引用。

    ``source_refs`` 仅来自经过复核的 ``task-output-artifact-v1`` envelope；direct
    artifact 不得自行获得递归祖先权限。
    """

    ref: str
    value: Mapping[str, object]
    artifact_hash: str
    formal_eligible: bool
    source_refs: tuple[str, ...] = ()


def safe_workspace_path(root: Path, logical_ref: str, *, field: str) -> Path:
    """解析 POSIX 逻辑引用，并确保最终路径仍位于 ``root`` 内。"""

    if not isinstance(logical_ref, str) or not logical_ref or "\\" in logical_ref:
        raise ValueError(f"STAGE789_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(logical_ref)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"STAGE789_LOGICAL_REF_ESCAPE:{field}")
    resolved_root = root.resolve()
    path = resolved_root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"STAGE789_LOGICAL_REF_ESCAPE:{field}") from error
    return path


def load_input_artifact(root: Path, ref: str) -> LoadedInputArtifact:
    """加载 direct canonical artifact 或 ``TaskArtifactStore`` 权威 commit。

    runtime checkpoint 会同时读取安全 TensorBundle，以确认 commit 中的 manifest
    摘要确实对应磁盘对象；这里不会反序列化 pickle 或未知 Python 对象。
    """

    path = safe_workspace_path(root, ref, field="input_ref")
    value = load_canonical_json(path)
    if not isinstance(value, dict):
        raise ValueError("STAGE789_INPUT_ROOT_NOT_OBJECT")

    if value.get("schema_version") == "runtime.checkpoint-commit.v1":
        expected = {
            "schema_version",
            "checkpoint_id",
            "generation",
            "object_relative_path",
            "bundle_manifest_sha256",
            "parent_checkpoint_id",
            "metadata",
            "committed_at",
        }
        if set(value) != expected:
            raise ValueError("STAGE789_CHECKPOINT_COMMIT_FIELDS_MISMATCH")
        object_relative = value.get("object_relative_path")
        checkpoint_id = value.get("checkpoint_id")
        if (
            not isinstance(object_relative, str)
            or not isinstance(checkpoint_id, str)
            or object_relative != f"objects/{checkpoint_id}"
        ):
            raise ValueError("STAGE789_CHECKPOINT_OBJECT_IDENTITY_MISMATCH")
        store_root = path.parent.parent.resolve()
        bundle_root = safe_workspace_path(
            store_root,
            object_relative,
            field="checkpoint_object_relative_path",
        )
        state, bundle = load_tensor_bundle(bundle_root)
        if bundle.manifest_sha256 != value.get("bundle_manifest_sha256"):
            raise ValueError("STAGE789_CHECKPOINT_BUNDLE_HASH_MISMATCH")
        run_spec = state.get("run_spec") if isinstance(state, Mapping) else None
        return LoadedInputArtifact(
            ref=ref,
            value=value,
            artifact_hash=canonical_json_hash(value),
            formal_eligible=(
                isinstance(run_spec, Mapping)
                and run_spec.get("run_intent") == "formal"
            ),
        )

    if value.get("schema_version") == "task-output-commit-v1":
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
            raise ValueError("STAGE789_INPUT_COMMIT_FIELDS_MISMATCH")
        object_ref = value.get("object_ref")
        if not isinstance(object_ref, str):
            raise TypeError("STAGE789_INPUT_OBJECT_REF_NOT_STRING")
        object_path = safe_workspace_path(root, object_ref, field="object_ref")
        object_value = load_canonical_json(object_path)
        if not isinstance(object_value, dict):
            raise ValueError("STAGE789_INPUT_OBJECT_ROOT_NOT_OBJECT")
        object_expected = {
            "schema_version",
            "task_id",
            "artifact_kind",
            "config_hash",
            "run_intent",
            "formal_eligible",
            "source_refs",
            "payload",
            "artifact_hash",
        }
        if set(object_value) != object_expected:
            raise ValueError("STAGE789_INPUT_OBJECT_FIELDS_MISMATCH")
        if object_value.get("schema_version") != "task-output-artifact-v1":
            raise ValueError("STAGE789_INPUT_OBJECT_SCHEMA_MISMATCH")
        if object_value.get("run_intent") not in {"local_fixture", "formal"}:
            raise ValueError("STAGE789_INPUT_OBJECT_RUN_INTENT_INVALID")
        if type(object_value.get("formal_eligible")) is not bool or (
            object_value["formal_eligible"]
            is not (object_value["run_intent"] == "formal")
        ):
            raise ValueError("STAGE789_INPUT_OBJECT_FORMAL_ELIGIBILITY_MISMATCH")
        source_refs = object_value.get("source_refs")
        if not isinstance(source_refs, list) or len(source_refs) != len(set(source_refs)):
            raise ValueError("STAGE789_INPUT_OBJECT_SOURCE_REFS_INVALID")
        for index, source_ref in enumerate(source_refs):
            if not isinstance(source_ref, str):
                raise TypeError("STAGE789_INPUT_OBJECT_SOURCE_REF_NOT_STRING")
            safe_workspace_path(root, source_ref, field=f"source_refs[{index}]")
        without_hash = dict(object_value)
        declared_hash = without_hash.pop("artifact_hash")
        if canonical_json_hash(without_hash) != declared_hash:
            raise ValueError("STAGE789_INPUT_OBJECT_HASH_MISMATCH")
        for identity_field in (
            "task_id",
            "artifact_kind",
            "config_hash",
            "artifact_hash",
            "formal_eligible",
        ):
            if value[identity_field] != object_value[identity_field]:
                raise ValueError(
                    f"STAGE789_INPUT_COMMIT_OBJECT_MISMATCH:{identity_field}"
                )
        payload = object_value["payload"]
        if not isinstance(payload, Mapping):
            raise TypeError("STAGE789_INPUT_PAYLOAD_NOT_OBJECT")
        return LoadedInputArtifact(
            ref=ref,
            value=dict(payload),
            artifact_hash=str(declared_hash),
            formal_eligible=bool(object_value["formal_eligible"]),
            source_refs=tuple(str(item) for item in source_refs),
        )

    if value.get("schema_version") == "ablation-matrix-v1":
        matrix = AblationMatrix.from_mapping(value)
        return LoadedInputArtifact(
            ref=ref,
            value=value,
            artifact_hash=matrix.digest,
            # matrix 合同本身没有 scope；正式资格必须来自外层 task commit。
            formal_eligible=False,
        )

    declared_hash = value.get("artifact_hash")
    if not isinstance(declared_hash, str):
        raise ValueError("STAGE789_DIRECT_INPUT_MISSING_ARTIFACT_HASH")
    without_hash = dict(value)
    without_hash.pop("artifact_hash")
    if canonical_json_hash(without_hash) != declared_hash:
        raise ValueError("STAGE789_DIRECT_INPUT_HASH_MISMATCH")
    return LoadedInputArtifact(
        ref=ref,
        value=value,
        artifact_hash=declared_hash,
        formal_eligible=value.get("formal_eligible") is True,
    )


def configured_input_refs(request: TaskExecutionRequest) -> tuple[str, ...]:
    """按配置顺序返回去重后的普通输入与 matrix 引用。"""

    orchestration = request.config.section("orchestration")
    if not isinstance(orchestration, dict):  # pragma: no cover - v2 已严格验证
        raise TypeError("STAGE789_ORCHESTRATION_NOT_OBJECT")
    raw_refs = orchestration.get("input_result_refs", [])
    if not isinstance(raw_refs, list):
        raise TypeError("STAGE789_INPUT_RESULT_REFS_NOT_ARRAY")
    refs = [str(item) for item in raw_refs]
    matrix_ref = orchestration.get("matrix_ref")
    if matrix_ref is not None:
        refs.append(str(matrix_ref))
    return tuple(dict.fromkeys(refs))


def load_configured_inputs(
    request: TaskExecutionRequest,
    workspace_root: Path,
) -> tuple[LoadedInputArtifact, ...]:
    """加载显式输入及 task-output 声明的完整上游 lineage。

    DFS 保留配置与 ``source_refs`` 顺序；循环、路径逃逸、hash 漂移都立即失败。
    """

    loaded_by_ref: dict[str, LoadedInputArtifact] = {}
    visiting: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visiting:
            raise ValueError(f"STAGE789_INPUT_LINEAGE_CYCLE:{ref}")
        if ref in loaded_by_ref:
            return
        visiting.add(ref)
        document = load_input_artifact(workspace_root, ref)
        loaded_by_ref[ref] = document
        for source_ref in document.source_refs:
            visit(source_ref)
        visiting.remove(ref)

    for configured_ref in configured_input_refs(request):
        visit(configured_ref)
    return tuple(loaded_by_ref.values())


def lineage_descendants(
    document: LoadedInputArtifact,
    *,
    index: Mapping[str, LoadedInputArtifact],
) -> tuple[LoadedInputArtifact, ...]:
    """按 task runtime 的 DFS 顺序返回一个 envelope 的完整祖先闭包。"""

    ordered: list[LoadedInputArtifact] = []
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(ref: str) -> None:
        if ref in visiting:
            raise ValueError(f"STAGE9_REPORT_LINEAGE_CYCLE:{ref}")
        if ref in visited:
            return
        ancestor = index.get(ref)
        if ancestor is None:
            raise ValueError(f"STAGE9_REPORT_LINEAGE_SOURCE_MISSING:{ref}")
        visiting.add(ref)
        visited.add(ref)
        ordered.append(ancestor)
        for parent_ref in ancestor.source_refs:
            visit(parent_ref)
        visiting.remove(ref)

    for source_ref in document.source_refs:
        visit(source_ref)
    return tuple(ordered)


def validate_analysis_lineage(
    documents: Sequence[LoadedInputArtifact],
    *,
    rebuild_source: Callable[[Sequence[LoadedInputArtifact]], BoundSourceTable],
    find_mappings: Callable[[object], Sequence[Mapping[str, object]]],
) -> None:
    """复核报告/推荐对象与其祖先闭包重建出的冻结源完全一致。

    ``rebuild_source`` 由分析 adapter 注入，因此本模块不依赖具体 ETL 行公式；它只
    负责 claim-to-source 的身份约束。
    """

    index = {document.ref: document for document in documents}
    if len(index) != len(documents):
        raise ValueError("STAGE9_INPUT_REF_DUPLICATE")

    for document in documents:
        mappings = find_mappings(document.value)
        reports = [
            AnalysisReport.from_mapping(mapping)
            for mapping in mappings
            if mapping.get("schema_version") == "analysis-report-v1"
        ]
        recommendations = [
            mapping
            for mapping in mappings
            if mapping.get("schema_version") == "configuration-recommendation-v1"
        ]
        if not reports and not recommendations:
            continue
        ancestors = lineage_descendants(document, index=index)
        if not ancestors:
            raise ValueError("STAGE9_ANALYSIS_CLAIM_HAS_NO_SOURCE_REFS")
        rebuilt = rebuild_source(ancestors)

        for report in reports:
            if len(report.source_artifacts) != 1:
                raise ValueError("STAGE9_REPORT_SOURCE_CARDINALITY_INVALID")
            declared = report.source_artifacts[0]
            if (
                declared.name != rebuilt.table.name
                or declared.schema_version != rebuilt.table.schema_version
                or declared.content_hash != rebuilt.table.content_hash
                or declared.row_count != len(rebuilt.table.rows)
                or not declared.frozen
            ):
                raise ValueError("STAGE9_REPORT_REBUILT_SOURCE_MISMATCH")

        for recommendation in recommendations:
            expected = {
                "schema_version",
                "source_artifact_hash",
                "status",
                "best_observed_mean",
                "statistics_table",
            }
            if set(recommendation) != expected:
                raise ValueError("STAGE9_RECOMMENDATION_FIELDS_MISMATCH")
            statistics_value = recommendation.get("statistics_table")
            if not isinstance(statistics_value, Mapping):
                raise TypeError("STAGE9_RECOMMENDATION_TABLE_NOT_OBJECT")
            statistics = BoundSourceTable.from_mapping(statistics_value)
            if recommendation.get("source_artifact_hash") != statistics.artifact_hash:
                raise ValueError("STAGE9_RECOMMENDATION_TABLE_HASH_MISMATCH")
            if rebuilt.artifact_hash not in statistics.parent_artifact_hashes:
                raise ValueError("STAGE9_RECOMMENDATION_PARENT_SOURCE_MISMATCH")


__all__ = [
    "LoadedInputArtifact",
    "configured_input_refs",
    "lineage_descendants",
    "load_configured_inputs",
    "load_input_artifact",
    "safe_workspace_path",
    "validate_analysis_lineage",
]
