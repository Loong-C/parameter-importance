"""Stage 2/3 专用任务运行时适配器。

本模块把 :mod:`stage2_formal`、:mod:`stage3_formal` 中已经冻结的研究核心接到
统一 ``TaskRuntime``。它刻意独立于 ``task_runners.py``，以便默认工厂只需在注册
通用 runner 之前调用 :func:`register_stage23_runners`，而不必让科学实现反向依赖
CLI 或训练入口。

设计边界
--------
* ``local_fixture`` 运行真实但缩小的 FP64 reference、paired estimator、端点捕获和
  路径求积，不把预制 JSON 当成计算结果；
* 每个核心 runner 的 shard/level commit 位于任务输出目录的 ``resume/`` 下，进程
  中断后以权威 commit 恢复；任务级 JSON 又统一交给 :class:`TaskArtifactStore`
  执行不可变对象加独立 commit 的发布；
* formal 在读取任何梯度前必须加载 ``FormalExecutionEvidence``、核对 prerequisite
  Gate、验证本地 asset manifest，并构造 ``TorchFixedStateGradientProvider``。
  缺少任一条件都会抛出 ``TaskBlockedError``，绝不会降级到 synthetic provider；
* 墙钟计时、此次恢复命中数等机器相关诊断不进入任务 artifact。科学产物仅保存
  draw/规则/向量/成本单位等确定内容，从而保证全新执行与恢复执行得到同一 hash；
* Gate artifact 只写 ``NOT_RUN``。任务运行成功表示编排单元成功，不代表 formal
  Gate 已通过，更不会把本机验证改写成 ``PASS`` Gate。
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import subprocess
from time import perf_counter

import numpy as np
import torch

from ..capacity import (
    StorageBudget,
    estimate_experiment_storage,
    estimate_parameter_statistics_bytes,
)
from ..contracts.artifacts import validate_reference_result_artifact
from ..contracts.errors import DependencyUnavailable, FormalRunRejected
from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from ..contracts.immutable import thaw_json_value
from ..contracts.stage1_handoff import (
    Stage1ExitEvidence,
    Stage1HandoffError,
    validate_stage1_exit_evidence,
)
from ..contracts.stage0_handoff import (
    Stage0HandoffError,
    Stage0HandoffEvidence,
    validate_stage0_handoff,
)
from ..contracts.seed import SeedPlan
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.freeze import ContractFreeze
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG, RecoveryMode, RunnerKind
from ..analysis import (
    AnalysisReportBuilder,
    ChartArtifact,
    ChartSpec,
    FrozenSourceTable,
    bias as analysis_bias,
    mae as analysis_mae,
    mse as analysis_mse,
    pearson as analysis_pearson,
    spearman as analysis_spearman,
    top_k_overlap as analysis_top_k_overlap,
)
from ..core.quadrature import (
    PathIntegralResult,
    PathSpec,
    composite_left_rule,
    composite_midpoint_rule,
    composite_right_rule,
    composite_simpson_rule,
    composite_trapezoid_rule,
    default_quadrature_rules,
    gauss_legendre_rule,
    integrate_scalar_function,
    midpoint_rule,
    simpson_rule,
    trapezoid_rule,
)
from ..core.registry import ParameterRegistry
from ..core.tensors import TensorMap
from ..g3_runtime_assets import (
    FormalG3RuntimeAssets,
    G3RuntimeAssetError,
    formal_pile_route,
    reject_legacy_provider_paths,
)
from ..providers import (
    FixedStateGradientProvider,
    OfflineHuggingFaceModelAdapter,
    PythiaMMapFrozenSampleResolver,
    PretokenizedGlueDatasetAdapter,
    SyntheticGradientProvider,
    TorchFixedStateGradientProvider,
)
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from ..runtime.task_runtime import (
    BlockerCode,
    TaskBlockedError,
    TaskBlocker,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunner,
    TaskRuntime,
    TaskRuntimeError,
)
from ..runtime.tensor_bundle import (
    TensorBundle,
    load_tensor_bundle,
    publish_tensor_bundle,
)
from ..runtime.training_factory import build_optimizer
from .sampling import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    DrawStreamManifest,
    MICROBATCH_SELECTION_ORDER,
    PrimaryPairDecision,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    STREAM_NAMES,
)
from .stage2_assets import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    DataFile,
    DataRangeManifest,
    validate_formal_asset_identity,
)
from .stage2 import PairedEstimatorRunner, build_fixture_estimator_decision
from .preregistration import (
    ABSOLUTE_FLOORS,
    PREREGISTRATION_SCHEMA_VERSION,
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
    validate_stage2_preregistration,
)
from .stage2_formal import (
    ANCHOR_IDS,
    AnchorPilotResult,
    CostSemantics,
    freeze_fixture_matrix,
    run_artificial_distribution_calibration,
    FormalExperimentPlan,
    OneShotReferencePlan,
    OneShotReferenceRunner,
    PilotCellObservation,
    PilotThresholds,
    RecoverablePairedWaveRunner,
    ReferenceSizingPlan,
    Stage2RecommendationEngine,
    StreamingReferenceSizer,
    _ReferenceSnapshotStore,
    _ReferenceShardStore,
    _moments_from_shards,
    _draw_digest,
)
from .stage2_g23_contracts import (
    boundary_digest,
    generator_boundary,
    source_manifest_for_refs,
    validate_external_manifest,
    validate_sizing_plan_contract,
    validate_weighting_contract,
)
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage3 import (
    EndpointState,
    NodeCacheKey,
    PathAnalysisRunner,
    ProbeSpec,
    build_fixture_quadrature_decision,
)
from .training_endpoints import validate_endpoint_state_bundle
from .stage3_formal import (
    EndpointCaptureCoordinator,
    EndpointCaptureRequest,
    ProbePanel,
    ProbePanelEntry,
    QuadratureObservation,
    QuadratureRecommendation,
    QuadratureRecommendationEngine,
    QuadratureThresholds,
    PersistentNodeGradientCache,
    ReferenceRefinementRunner,
    ReferenceRuleLevel,
    SafeTensorTreeCodec,
)


_STAGE2_REFERENCE_TASK = "stage2.04_reference_target"
_STAGE2_ESTIMATOR_TASKS = frozenset(
    {"stage2.05_paired_estimator_runner", "stage2.07_main_sweep"}
)
_STAGE2_PILOT_TASK = "stage2.06_pilot_and_matrix_freeze"
_FORMAL_SELECTED_CHECKPOINT_TASKS = frozenset(
    {
        _STAGE2_REFERENCE_TASK,
        "stage2.05_paired_estimator_runner",
        _STAGE2_PILOT_TASK,
        "stage2.07_main_sweep",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STAGE3_ENDPOINT_TASK = "stage3.03_endpoint_and_probe_pipeline"
_STAGE3_REFERENCE_TASK = "stage3.05_reference_integral_and_precision"
_STAGE3_PILOT_TASK = "stage3.06_pilot_and_threshold_freeze"
_STAGE3_MATRIX_TASK = "stage3.07_formal_experiment_matrix"

_STAGE23_TASK_ORDER = (
    "stage2.01_scope_hypotheses_and_preregistration",
    "stage2.02_stage1_handoff_and_fixed_state_contract",
    "stage2.03_assets_checkpoints_and_sampling",
    _STAGE2_REFERENCE_TASK,
    "stage2.05_paired_estimator_runner",
    _STAGE2_PILOT_TASK,
    "stage2.07_main_sweep",
    "stage2.08_statistics_and_robustness",
    "stage2.09_cost_and_system_validation",
    "stage2.10_visualization_reporting_and_decision",
    "stage2.11_delivery_and_exit_gate",
    "stage3.01_prerequisites_and_scope",
    "stage3.02_math_and_metric_contract",
    _STAGE3_ENDPOINT_TASK,
    "stage3.04_quadrature_engine_and_unit_tests",
    _STAGE3_REFERENCE_TASK,
    _STAGE3_PILOT_TASK,
    _STAGE3_MATRIX_TASK,
    "stage3.08_error_analysis_and_stability",
    "stage3.09_cost_and_method_selection",
    "stage3.10_reports_visualizations_and_handoff",
)

# Freeze the Stage 2 plan's direct DAG edges.  Stage 2.02 and 2.03 are
# independent consumers of 2.01; later tasks consume only the listed direct
# predecessors, rather than accidentally serializing the whole experiment.
_REQUIRED_PREDECESSORS: Mapping[str, tuple[str, ...]] = {
    # Stage 2 enters only from the completed formal Stage 1.11 delivery set.
    # Local fixture callers may still use the isolated S2.1 contract helper;
    # ``_predecessor_context`` keeps that explicit test-only exception while
    # formal execution always consumes all four Stage 1.11 commits.
    "stage2.01_scope_hypotheses_and_preregistration": (
        "stage1.11_reporting_and_exit_gate",
    ),
    "stage2.02_stage1_handoff_and_fixed_state_contract": (
        "stage2.01_scope_hypotheses_and_preregistration",
    ),
    "stage2.03_assets_checkpoints_and_sampling": (
        "stage2.01_scope_hypotheses_and_preregistration",
    ),
    "stage2.04_reference_target": (
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    ),
    "stage2.05_paired_estimator_runner": (
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    ),
    "stage2.06_pilot_and_matrix_freeze": (
        "stage2.04_reference_target",
        "stage2.05_paired_estimator_runner",
    ),
    "stage2.07_main_sweep": ("stage2.06_pilot_and_matrix_freeze",),
    "stage2.08_statistics_and_robustness": ("stage2.07_main_sweep",),
    "stage2.09_cost_and_system_validation": ("stage2.07_main_sweep",),
    "stage2.10_visualization_reporting_and_decision": (
        "stage2.08_statistics_and_robustness",
        "stage2.09_cost_and_system_validation",
    ),
    "stage2.11_delivery_and_exit_gate": (
        "stage2.10_visualization_reporting_and_decision",
    ),
}
# Stage 3 edges are intentionally left on their existing order; this task only
# corrects the Stage 2 mapping required by plan/stage2/README.md.
_REQUIRED_PREDECESSORS = {
    **_REQUIRED_PREDECESSORS,
    **{
        task_id: (_STAGE23_TASK_ORDER[index - 1],)
        for index, task_id in enumerate(_STAGE23_TASK_ORDER)
        if task_id.startswith("stage3.")
    },
}

# S2.5 is allowed to run against the development handoff in local fixtures,
# but its formal route consumes the independently qualified G2.3 reference.
# Keep this expansion formal-only so a fixture cannot masquerade as a formal
# reference and the historical development DAG remains executable.
_FORMAL_S25_PREDECESSORS = (
    "stage2.02_stage1_handoff_and_fixed_state_contract",
    "stage2.03_assets_checkpoints_and_sampling",
    "stage2.04_reference_target",
)
# S2.3 is a DAG sibling of the S2.2 handoff: its only task predecessor is the
# frozen S2.1 preregistration.  Stage 1/formal asset evidence is supplied by
# the environment and is intentionally not smuggled in through S2.2.
_REQUIRED_PREDECESSORS = {
    **_REQUIRED_PREDECESSORS,
    "stage2.03_assets_checkpoints_and_sampling": (
        "stage2.01_scope_hypotheses_and_preregistration",
    ),
}


def _logical_path(value: str, *, field: str) -> PurePosixPath:
    """验证 workspace 相对逻辑路径，不接受反斜杠、绝对路径或 ``..``。"""

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE23_TASK_LOGICAL_PATH_INVALID:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"STAGE23_TASK_PATH_ESCAPE:{field}")
    return path


def _workspace_path(root: Path, value: str, *, field: str) -> Path:
    logical = _logical_path(value, field=field)
    if root.is_symlink():
        raise ValueError(f"STAGE23_TASK_SYMLINK_FORBIDDEN:{field}")
    current = root
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"STAGE23_TASK_SYMLINK_FORBIDDEN:{field}")
    target = root.joinpath(*logical.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"STAGE23_TASK_PATH_ESCAPE:{field}") from error
    return target


def _artifact_store(request: TaskExecutionRequest, root: Path) -> TaskArtifactStore:
    section = request.config.section("artifacts")
    assert isinstance(section, dict)
    return TaskArtifactStore(root, str(section["output_dir"]))


def _completed_result(
    request: TaskExecutionRequest,
    store: TaskArtifactStore,
) -> TaskRunResult | None:
    """只从完整 task commits 恢复；孤立对象或部分 commit 不代表任务完成。"""

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
        # 首次执行与 commit 恢复必须产生同一 TaskRunResult 身份；恢复命中属于运行期
        # 诊断，不能通过 message/metadata 污染结果 hash。
        message="stage2/3 specialized task completed",
        metadata={"execution_contract": "stage23-specialized-v1"},
    )


def _authoritative_partial_paths(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Path, ...]:
    """发现尚未形成完整任务结果的权威恢复边界。

    ``TaskArtifactStore`` 的顶层 ``commits/`` 可能只发布了部分预期产物；Stage 2
    的 repetition/block-pair 与 Stage 3 的 refinement/node cache 则把独立 commit
    放在输出目录更深层的 ``commits/``。这些 commit 都代表已经发生的可恢复状态
    推进，普通 ``task run`` 不能在调用方不知情的情况下消费它们。

    仅有内容寻址 object 或 TensorBundle manifest 不属于权威 commit：进程若在两
    阶段发布的第一阶段退出，fresh run 可以核对内容后幂等重建 commit。
    """

    if request.task.recovery_mode not in {
        RecoveryMode.RESUME_CHECKPOINT,
        RecoveryMode.RESUME_SHARDS,
        RecoveryMode.RECONCILE_STATE,
    }:
        # restart_idempotent/rebuild_derived 的目录合同明确要求从配置或冻结源重建；
        # 已发布的相同 task artifact 可由 publish() 校验并复用，不应伪装成续跑。
        return ()

    expected_kinds = set(request.task.artifact_kinds)
    expected_formal = request.config.run_intent == "formal"
    paths: list[Path] = []
    for path in sorted(store.commits.glob("*.json")):
        if path.stem not in expected_kinds:
            raise ValueError(f"STAGE23_UNEXPECTED_TASK_COMMIT:{path.name}")
        published = store.load_commit(path.resolve().relative_to(root).as_posix())
        if (
            published.task_id != request.task.task_id
            or published.config_hash != request.config.config_hash
            or published.formal_eligible != expected_formal
        ):
            raise ValueError(
                f"STAGE23_PARTIAL_TASK_COMMIT_IDENTITY_DRIFT:{path.name}"
            )
        paths.append(path.resolve())

    # 顶层 task commit 已在上面完成身份复核；这里只收集核心 runner 发布的 shard、
    # refinement 与未来 node-cache commit。目录名是协议的一部分，不能把 object
    # 目录中的普通 JSON 误判成恢复边界。
    for path in sorted(store.root.rglob("commits/*.json")):
        resolved = path.resolve()
        if resolved.parent == store.commits.resolve():
            continue
        paths.append(resolved)
    return tuple(dict.fromkeys(paths))


def _authorize_partial_resume(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    partial_paths: Sequence[Path],
) -> None:
    """把配置中的 ``resume_ref`` 与实际权威恢复状态绑定。

    完整 task commit 已由 :func:`_completed_result` 提前处理，不需要伪装成恢复。
    对不完整状态，``task resume`` 必须提供位于当前任务输出目录内、且能指向至少
    一个现存权威 commit（或其祖先目录）的引用；反之 ``task run`` 明确失败。
    """

    recovery = request.config.section("recovery")
    assert isinstance(recovery, dict)
    raw_resume_ref = recovery["resume_ref"]
    if not partial_paths:
        if raw_resume_ref is not None:
            raise FileNotFoundError("STAGE23_RESUME_REF_HAS_NO_AUTHORITATIVE_STATE")
        return
    if raw_resume_ref is None:
        refs = ",".join(path.relative_to(root).as_posix() for path in partial_paths)
        raise RuntimeError(f"STAGE23_RESUME_REF_REQUIRED:{refs}")

    resume_path = _workspace_path(
        root,
        str(raw_resume_ref),
        field="recovery.resume_ref",
    ).resolve()
    try:
        resume_path.relative_to(store.root.resolve())
    except ValueError as error:
        raise ValueError("STAGE23_RESUME_REF_OUTSIDE_TASK_OUTPUT") from error
    if not resume_path.exists():
        raise FileNotFoundError("STAGE23_RESUME_REF_NOT_FOUND")
    if not any(
        path == resume_path or path.is_relative_to(resume_path)
        for path in partial_paths
    ):
        raise ValueError("STAGE23_RESUME_REF_DOES_NOT_BIND_AUTHORITATIVE_STATE")


def _source_refs(request: TaskExecutionRequest, extra: Sequence[str] = ()) -> tuple[str, ...]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    ordered = [str(item) for item in orchestration["input_result_refs"]]
    ordered.extend(extra)
    # source_refs 是 lineage，不是 multiset；保持首次出现顺序并避免 store 的重复拒绝。
    return tuple(dict.fromkeys(ordered))


def _formal_environment_source_refs(request: TaskExecutionRequest) -> tuple[str, ...]:
    """Return immutable environment evidence refs for formal handoff lineage.

    S2.2's direct predecessors remain the three S2.1 commits.  The formal
    fixed-state handoff is additionally bound to the exact G2.1/environment
    snapshot that authorized the provider; these refs are appended by the
    specialized handler rather than smuggled into the predecessor set.
    """

    if request.config.run_intent != "formal":
        return ()
    evidence_refs = request.environment.evidence_refs
    return tuple(dict.fromkeys(str(ref) for ref in evidence_refs.values()))


def _publish_payloads(
    request: TaskExecutionRequest,
    store: TaskArtifactStore,
    payloads: Mapping[str, Mapping[str, JSONValue]],
    *,
    source_refs: tuple[str, ...] = (),
) -> Mapping[str, str]:
    if tuple(payloads) != request.task.artifact_kinds:
        raise ValueError("STAGE23_PAYLOAD_KIND_ORDER_MISMATCH")
    references: dict[str, str] = {}
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
        references[kind] = published.commit_ref
    return references


def _gate_candidate(request: TaskExecutionRequest) -> dict[str, JSONValue]:
    """生成待独立审核的 Gate 记录，绝不把 runner/local validation 当作 Gate PASS。"""

    gate_ids = list(request.task.formal_eligibility.required_gate_ids)
    return {
        "schema_version": "stage23-task-gate-candidate-v1",
        "task_id": request.task.task_id,
        "gate_ids": gate_ids,
        "gate_status": "NOT_RUN",
        "local_validation_status": (
            "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
        ),
        "formal_eligible": False,
        "reason": "formal_gate_requires_independent_review",
    }


def _publish_or_load_bundle(path: Path, state: Mapping[str, object]) -> TensorBundle:
    """幂等发布小型 tensor bundle，并在复用时核对完整状态树。"""

    normalized = _plain_state_tree(state)
    assert isinstance(normalized, dict)
    if path.exists():
        restored, bundle = load_tensor_bundle(path)
        if not isinstance(restored, Mapping) or _tree_digest(restored) != _tree_digest(
            normalized
        ):
            raise ValueError("STAGE23_EXISTING_TENSOR_BUNDLE_DRIFT")
        return bundle
    return publish_tensor_bundle(path, normalized)


def _plain_state_tree(value: object) -> object:
    """把 MappingProxyType 等只读视图降为安全 bundle 支持的普通状态树。"""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("STAGE23_STATE_TREE_KEY_NOT_STRING")
        return {str(key): _plain_state_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_state_tree(item) for item in value]
    if isinstance(value, list):
        return [_plain_state_tree(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return value


def _tree_digest(value: object) -> str:
    """仅用于本模块幂等核对的 primitive/tensor tree 摘要。"""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(canonical_json_hash(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                if not isinstance(key, str):
                    raise TypeError("STAGE23_STATE_TREE_KEY_NOT_STRING")
                digest.update(key.encode("utf-8"))
                digest.update(b"\0")
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
            return
        digest.update(canonical_json_hash(item).encode("ascii"))

    visit(value)
    return digest.hexdigest()


def _as_numpy_vector(value: Mapping[str, object]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, item in sorted(value.items()):
        if hasattr(item, "detach"):
            item = item.detach()  # type: ignore[union-attr]
        if hasattr(item, "cpu"):
            item = item.cpu()  # type: ignore[union-attr]
        if hasattr(item, "numpy"):
            item = item.numpy()  # type: ignore[union-attr]
        array = np.array(item, dtype=np.float64, copy=True, order="C")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"STAGE23_VECTOR_NONFINITE:{name}")
        result[str(name)] = array
    if not result:
        raise ValueError("STAGE23_VECTOR_EMPTY")
    return result


def _vector_digest(value: Mapping[str, object]) -> str:
    """与 Stage 2/3 formal 核心一致的坐标向量 SHA-256。"""

    digest = hashlib.sha256()
    for name, array in _as_numpy_vector(value).items():
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _flatten(value: Mapping[str, object]) -> np.ndarray:
    arrays = _as_numpy_vector(value)
    return np.concatenate([arrays[name].reshape(-1) for name in arrays])


@dataclass(frozen=True, slots=True)
class _ProviderContext:
    provider: FixedStateGradientProvider
    sample_ids: tuple[Hashable, ...]
    evidence: FormalExecutionEvidence
    provider_kind: str
    asset_manifest_hashes: tuple[str, ...]
    # Formal contexts must carry the validated immutable S1.11 closure.  The
    # local fixture deliberately leaves this empty and can never be promoted.
    stage1_exit: Stage1ExitEvidence | None = None
    # Formal contexts must also carry the current Stage 0 infrastructure
    # binding.  A historical/blocked manifest is retained for diagnostics but
    # can never unlock a formal provider.
    stage0_handoff: Stage0HandoffEvidence | None = None
    # Test/in-memory adapters predate G3 provenance projection.  The formal
    # provider still supplies the full tuple explicitly; a missing value here
    # means "no external asset provenance", never an inferred formal claim.
    asset_provenance: tuple[Mapping[str, JSONValue], ...] = ()
    # Stage 2.04--2.07 formal providers additionally bind the selected S2.3
    # checkpoint.  G3 assets remain in ``asset_provenance`` as the runtime
    # authorization/provenance boundary; this field records the actual model
    # root loaded for gradients.
    checkpoint_identity: Mapping[str, JSONValue] | None = None

    def to_payload(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "provider_kind": self.provider_kind,
            "fixed_state_id": self.provider.fixed_state_id,
            "provider_state_digest": self.provider.state_digest(),
            "registry_hash": self.provider.registry_hash,
            "parameter_names": list(self.provider.parameter_names),
            "sample_universe_size": len(self.sample_ids),
            "execution_evidence_hash": self.evidence.artifact_hash,
            "stage1_g1_exit": (
                self.stage1_exit.to_dict() if self.stage1_exit is not None else None
            ),
            "stage0_handoff": (
                self.stage0_handoff.to_dict()
                if self.stage0_handoff is not None
                else None
            ),
            "asset_manifest_hashes": list(self.asset_manifest_hashes),
            "asset_provenance": [dict(item) for item in self.asset_provenance],
            "weighting_assumptions": {
                "statistical_unit": self.provider.statistical_unit,
                "weight_unit": self.provider.weight_unit,
                "sampling_design": self.provider.sampling_design,
                "weights_exogenous": self.provider.weights_exogenous,
                "common_mean_assumption": self.provider.common_mean_assumption,
            },
        }
        if self.checkpoint_identity is not None:
            payload["checkpoint_identity"] = dict(self.checkpoint_identity)
        return payload


def _local_provider(request: TaskExecutionRequest) -> _ProviderContext:
    """构造确定性的有限经验梯度分布；只允许 local_fixture 调用。"""

    if request.config.run_intent != "local_fixture":
        raise RuntimeError("SYNTHETIC_PROVIDER_FOR_FORMAL_FORBIDDEN")
    identity = request.config.base_config.section("identity")
    seed_plan = SeedPlan.from_master_seed(int(identity["master_seed"]))
    # task_id、输出目录和当前 Stage 都不属于固定模型状态。若直接使用 resolved
    # config hash，不同链路任务会在数值相同的情况下得到不同 fixed_state_id，进而
    # 掩盖错误交接。这里仅绑定真正决定 fixture 梯度分布的静态配置与 seed。
    fixed_state_binding = canonical_json_hash(
        {
            "master_seed": int(identity["master_seed"]),
            "model": request.config.base_config.section("model"),
            "data": request.config.base_config.section("data"),
            "loss": request.config.base_config.section("loss"),
            "optimizer": request.config.base_config.section("optimizer"),
        }
    )
    provider = SyntheticGradientProvider.from_location_scale(
        parameter_shapes={"layer.bias": (2,), "layer.weight": (4,)},
        sample_count=32,
        mean=0.4,
        noise_scale=0.2,
        seed=seed_plan.seed_for("importance_sampling"),
        fixed_state_id=f"fixture-{fixed_state_binding[:16]}",
    )
    evidence = FormalExecutionEvidence(
        "local_fixture",
        metadata={
            "task_id": request.task.task_id,
            "config_hash": request.config.config_hash,
            "seed_plan_hash": seed_plan.artifact_hash,
        },
    )
    return _ProviderContext(
        provider=provider,
        sample_ids=provider.sample_ids,
        evidence=evidence,
        provider_kind="synthetic_local_fixture",
        asset_manifest_hashes=(),
        asset_provenance=(),
    )


def _document_hash(path: Path) -> str:
    value = load_canonical_json(path)
    if isinstance(value, Mapping):
        declared = value.get("artifact_hash")
        if isinstance(declared, str) and len(declared) == 64:
            payload = {key: item for key, item in value.items() if key != "artifact_hash"}
            if canonical_json_hash(payload) != declared:
                raise ValueError(f"STAGE23_DOCUMENT_ARTIFACT_HASH_MISMATCH:{path}")
            return declared
    return canonical_json_hash(value)


def _formal_contract_freeze_hash(
    root: Path,
    reference: str,
    *,
    stage: int,
) -> str:
    """读取 formal contract freeze，严格区分 TaskArtifact 与旧裸文档。"""

    path = _workspace_path(root, reference, field="contract_freeze")
    value = load_canonical_json(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != (
        "task-output-commit-v1"
    ):
        # Historical handoff documents predate the TaskArtifact envelope and remain
        # readable only through the narrowly scoped legacy hash path.
        return _document_hash(path)

    # Once the root advertises a TaskArtifact commit, every validation below is
    # mandatory.  In particular, never fall back to hashing the commit envelope
    # when its object, kind, or payload is malformed.
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    if loaded.identity.artifact_kind != "contract_freeze":
        raise ValueError(
            "CONTRACT_FREEZE_COMMIT_ARTIFACT_KIND_INVALID:"
            f"{loaded.identity.artifact_kind}"
        )
    payload = loaded.payload
    if payload.get("schema_version") != "contract-freeze-v1":
        raise ValueError("CONTRACT_FREEZE_COMMIT_PAYLOAD_SCHEMA_INVALID")
    freeze = ContractFreeze.from_mapping(dict(payload))
    if freeze.stage != stage or not freeze.formal_eligible:
        raise ValueError("CONTRACT_FREEZE_COMMIT_NOT_FORMAL_FOR_STAGE")
    payload_hash = payload.get("artifact_hash")
    if not isinstance(payload_hash, str):
        raise ValueError("CONTRACT_FREEZE_COMMIT_PAYLOAD_HASH_MISSING")
    # ContractFreeze.from_mapping already binds this field to the payload; return
    # the payload hash (rather than the enclosing TaskArtifact hash) to compare
    # against FormalExecutionEvidence.contract_freeze_hash.
    return payload_hash


def _blocked(
    code: BlockerCode,
    requirement: str,
    message: str,
    *,
    retryable: bool = True,
    evidence_refs: tuple[str, ...] = (),
) -> TaskBlockedError:
    return TaskBlockedError(
        TaskBlocker(code, requirement, message, retryable, evidence_refs)
    )


@dataclass(frozen=True, slots=True)
class _BoundInputArtifact:
    """一个已从权威 task commit 完整复核的前序产物。"""

    task_id: str
    artifact_kind: str
    artifact_hash: str
    config_hash: str
    run_intent: str
    formal_eligible: bool
    commit_ref: str
    source_refs: tuple[str, ...]
    payload: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _PredecessorContext:
    """当前任务的直接前驱证据集。

    ``artifacts`` 的顺序严格跟随任务目录中的 ``artifact_kinds``。这既固定了
    lineage hash，也防止调用方只传前驱的一个“好看结果”而漏掉 Gate、完整性或
    shard 报告。
    """

    predecessor_task_ids: tuple[str, ...]
    artifacts: tuple[_BoundInputArtifact, ...]
    auxiliary_refs: tuple[str, ...]

    @property
    def references(self) -> tuple[str, ...]:
        return tuple(item.commit_ref for item in self.artifacts) + self.auxiliary_refs

    @property
    def binding_hash(self) -> str:
        return canonical_json_hash(
            {
                "predecessor_task_ids": list(self.predecessor_task_ids),
                "artifacts": [
                    {
                        "task_id": item.task_id,
                        "artifact_kind": item.artifact_kind,
                        "artifact_hash": item.artifact_hash,
                        "config_hash": item.config_hash,
                        "run_intent": item.run_intent,
                        "formal_eligible": item.formal_eligible,
                        "commit_ref": item.commit_ref,
                    }
                    for item in self.artifacts
                ],
                "auxiliary_refs": list(self.auxiliary_refs),
            }
        )

    def payload(self, artifact_kind: str) -> Mapping[str, object]:
        matches = [
            item.payload for item in self.artifacts if item.artifact_kind == artifact_kind
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"STAGE23_PREDECESSOR_PAYLOAD_NOT_UNIQUE:{artifact_kind}:{len(matches)}"
            )
        return matches[0]

    def payload_for(self, task_id: str, artifact_kind: str) -> Mapping[str, object]:
        """Return one payload bound to both its producer task and artifact kind."""

        matches = [
            item.payload
            for item in self.artifacts
            if item.task_id == task_id and item.artifact_kind == artifact_kind
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "STAGE23_PREDECESSOR_PAYLOAD_NOT_UNIQUE:"
                f"{task_id}:{artifact_kind}:{len(matches)}"
            )
        return matches[0]


def _load_bound_task_input(
    store: TaskArtifactStore,
    reference: str,
) -> _BoundInputArtifact:
    """从 commit 走完对象发现路径，并额外读取 run_intent/payload。"""

    published = store.load_commit(reference)
    body = load_canonical_json(
        _workspace_path(store.workspace_root, published.object_ref, field="object_ref")
    )
    if not isinstance(body, Mapping):  # pragma: no cover - load_commit 已覆盖
        raise ValueError("STAGE23_INPUT_OBJECT_ROOT_NOT_MAPPING")
    expected = {
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
    if set(body) != expected or body.get("schema_version") != "task-output-artifact-v1":
        raise ValueError("STAGE23_INPUT_OBJECT_FIELDS_INVALID")
    payload = body.get("payload")
    if not isinstance(payload, Mapping):
        raise ValueError("STAGE23_INPUT_PAYLOAD_NOT_MAPPING")
    source_refs = body.get("source_refs")
    if not isinstance(source_refs, (list, tuple)) or not all(
        isinstance(value, str) and value for value in source_refs
    ):
        raise ValueError("STAGE23_INPUT_SOURCE_REFS_INVALID")
    run_intent = body.get("run_intent")
    if run_intent not in {"local_fixture", "formal"}:
        raise ValueError("STAGE23_INPUT_RUN_INTENT_INVALID")
    return _BoundInputArtifact(
        task_id=published.task_id,
        artifact_kind=published.artifact_kind,
        artifact_hash=published.artifact_hash,
        config_hash=published.config_hash,
        run_intent=str(run_intent),
        formal_eligible=published.formal_eligible,
        commit_ref=published.commit_ref,
        source_refs=tuple(source_refs),
        payload=dict(payload),
    )


def _predecessor_context(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> _PredecessorContext:
    """严格装载当前 Stage 2/3 任务的完整直接前驱产物集。

    非 task 文档只允许作为带 ``artifact_hash`` 的辅助计划输入；它们不能替代
    canonical 前驱。任一 commit/object/hash、scope 或 artifact 集不完整都转换为
    结构化 ``BLOCKED``，formal 绝不回退到 fixture。
    """

    expected_tasks = _REQUIRED_PREDECESSORS.get(request.task.task_id)
    if expected_tasks is None:
        raise TaskRuntimeError(
            f"STAGE23_PREDECESSOR_CONTRACT_MISSING:{request.task.task_id}"
        )
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    raw_refs = tuple(str(item) for item in orchestration["input_result_refs"])
    if len(raw_refs) != len(set(raw_refs)):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "predecessor_artifacts",
            "input_result_refs 含重复引用，无法形成唯一 lineage",
            retryable=False,
        )

    if (
        request.config.run_intent == "local_fixture"
        and request.task.task_id == "stage2.01_scope_hypotheses_and_preregistration"
        and not raw_refs
        and expected_tasks == ("stage1.11_reporting_and_exit_gate",)
    ):
        expected_tasks = ()
    elif (
        request.config.run_intent == "formal"
        and request.task.task_id == "stage2.02_stage1_handoff_and_fixed_state_contract"
    ):
        # The reviewed G2.1 config carries the complete Stage 1 handoff (S1.10
        # and S1.11) alongside the three S2.1 commits.  They are direct
        # immutable inputs for this audit, while the catalog still names S2.1
        # as the scientific predecessor.  Keep the expanded set formal-only;
        # fixture chains retain the narrow legacy contract.
        expected_tasks = (
            "stage2.01_scope_hypotheses_and_preregistration",
            "stage1.10_checkpoint_resume_and_artifacts",
            "stage1.11_reporting_and_exit_gate",
        )
    elif (
        request.config.run_intent == "formal"
        and request.task.task_id == "stage2.05_paired_estimator_runner"
    ):
        expected_tasks = _FORMAL_S25_PREDECESSORS

    grouped: dict[str, dict[str, _BoundInputArtifact]] = {}
    auxiliaries: list[str] = []
    for reference in raw_refs:
        try:
            value = load_canonical_json(
                _workspace_path(root, reference, field="input_result_refs")
            )
            if isinstance(value, Mapping) and value.get("schema_version") == (
                "task-output-commit-v1"
            ):
                item = _load_bound_task_input(store, reference)
                if item.run_intent != request.config.run_intent:
                    raise ValueError("STAGE23_INPUT_RUN_INTENT_MISMATCH")
                if item.formal_eligible != (request.config.run_intent == "formal"):
                    raise ValueError("STAGE23_INPUT_FORMAL_ELIGIBILITY_MISMATCH")
                if item.task_id not in expected_tasks:
                    raise ValueError(
                        f"STAGE23_UNEXPECTED_PREDECESSOR_TASK:{item.task_id}"
                    )
                by_kind = grouped.setdefault(item.task_id, {})
                if item.artifact_kind in by_kind:
                    raise ValueError(
                        f"STAGE23_DUPLICATE_PREDECESSOR_KIND:{item.artifact_kind}"
                    )
                by_kind[item.artifact_kind] = item
                continue

            # formal sizing/matrix 等外部计划可以并列输入，但必须自己携带可复算 hash。
            if not isinstance(value, Mapping):
                raise ValueError("STAGE23_AUXILIARY_INPUT_ROOT_NOT_MAPPING")
            declared = value.get("artifact_hash")
            if not isinstance(declared, str) or canonical_json_hash(
                {key: item for key, item in value.items() if key != "artifact_hash"}
            ) != declared:
                raise ValueError("STAGE23_AUXILIARY_INPUT_NOT_HASH_BOUND")
            auxiliaries.append(reference)
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "predecessor_artifacts",
                f"前序引用不可验证：{reference}: {type(error).__name__}: {error}",
                retryable=False,
                evidence_refs=(reference,),
            ) from error

    ordered: list[_BoundInputArtifact] = []
    for predecessor_id in expected_tasks:
        definition = DEFAULT_TASK_CATALOG.get(predecessor_id)
        observed = grouped.get(predecessor_id, {})
        missing = [kind for kind in definition.artifact_kinds if kind not in observed]
        extra = sorted(set(observed) - set(definition.artifact_kinds))
        if missing or extra:
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                f"complete_predecessor:{predecessor_id}",
                f"前序 artifact 集不完整：missing={missing}, extra={extra}",
                evidence_refs=raw_refs,
            )
        ordered.extend(observed[kind] for kind in definition.artifact_kinds)

    return _PredecessorContext(expected_tasks, tuple(ordered), tuple(auxiliaries))


def validate_formal_s203_payloads(
    payloads: Mapping[str, Mapping[str, object]],
    *,
    expected_preregistration_hash: str | None = None,
    expected_upstream_binding_hash: str | None = None,
) -> None:
    """Shared canonical validation for the four S2.3 formal payloads.

    The producer calls this before returning its payloads and the G2.2 adapter
    calls it after commit discovery.  Keeping the aggregate replay checks here
    prevents the consumer from inventing a second S2.3 wire contract.
    """
    kinds = ("sampling_plan", "draw_manifest", "asset_resolution", "gate_record")
    if tuple(payloads) != kinds:
        raise ValueError("STAGE23_S203_PAYLOAD_KIND_ORDER_MISMATCH")
    asset = payloads["asset_resolution"]
    if set(asset) != {
        "schema_version", "provider", "stage2_asset_manifest",
        "preregistration_contract_hash", "upstream_binding_hash", "formal_eligible",
    } or asset.get("schema_version") != "stage2-task-asset-resolution-v1" or asset.get("formal_eligible") is not False:
        raise ValueError("STAGE23_S203_ASSET_PAYLOAD_INVALID")
    manifest_value = asset.get("stage2_asset_manifest")
    if not isinstance(manifest_value, Mapping):
        raise ValueError("STAGE23_S203_ASSET_MANIFEST_NOT_OBJECT")
    manifest = AssetResolutionManifest.from_mapping(manifest_value)
    validate_formal_asset_identity(manifest)
    provider = asset.get("provider")
    if not isinstance(provider, Mapping) or provider.get("provider_kind") != "offline_hf_stage2_asset_manifest" or provider.get("asset_resolution_hash") != manifest.digest or provider.get("data_range_hash") != manifest.data_range.digest:
        raise ValueError("STAGE23_S203_PROVIDER_BINDING_INVALID")
    if expected_preregistration_hash is not None and asset.get("preregistration_contract_hash") != expected_preregistration_hash:
        raise ValueError("STAGE23_S203_PREREGISTRATION_HASH_MISMATCH")
    if expected_upstream_binding_hash is not None and asset.get("upstream_binding_hash") != expected_upstream_binding_hash:
        raise ValueError("STAGE23_S203_UPSTREAM_BINDING_HASH_MISMATCH")
    plan = SamplingPlan.from_mapping(payloads["sampling_plan"])
    if tuple(plan.universe.sample_ids) != tuple(range(manifest.data_range.sample_id_min, manifest.data_range.sample_id_max_exclusive)):
        raise ValueError("STAGE23_S203_SAMPLING_UNIVERSE_MISMATCH")
    draw = payloads["draw_manifest"]
    required = {"schema_version", "sampling_plan_hash", "draws", "draw_count_by_stream", "stream_manifests", "draw_id_unique", "sample_id_collisions_allowed", "replay_hash", "nested_mapping", "nested_mapping_hash"}
    if set(draw) != required or draw.get("schema_version") != "stage2-task-draw-manifest-v1" or draw.get("sampling_plan_hash") != plan.digest:
        raise ValueError("STAGE23_S203_DRAW_PAYLOAD_INVALID")
    streams = draw.get("stream_manifests")
    if not isinstance(streams, Mapping) or set(streams) != set(STREAM_NAMES):
        raise ValueError("STAGE23_S203_STREAM_SET_INVALID")
    parsed = {name: DrawStreamManifest.from_manifest(value) for name, value in streams.items() if isinstance(value, Mapping)}
    if set(parsed) != set(STREAM_NAMES) or any(item.sampling_plan_hash != plan.digest for item in parsed.values()) or any(len(item.draws) != 4 for item in parsed.values()):
        raise ValueError("STAGE23_S203_STREAM_INVALID")
    rows = draw.get("draws")
    expected_rows = [item.to_manifest() for name in STREAM_NAMES for item in parsed[name].draws]
    if rows != expected_rows or draw.get("draw_count_by_stream") != {name: 4 for name in STREAM_NAMES} or draw.get("replay_hash") != canonical_json_hash(rows):
        raise ValueError("STAGE23_S203_DRAW_REPLAY_INVALID")
    mapping_value = draw.get("nested_mapping")
    if not isinstance(mapping_value, Mapping):
        raise ValueError("STAGE23_S203_MAPPING_NOT_OBJECT")
    mapping = RepetitionMapping.from_manifest(mapping_value)
    if mapping.digest != draw.get("nested_mapping_hash") or mapping.repetition_id != "stage2-formal-sampling-nested-fixture" or mapping.m_values != (2, 4, 8) or tuple(item.to_manifest() for item in mapping.draws) != tuple(item.to_manifest() for item in plan.draws("pilot", 8)):
        raise ValueError("STAGE23_S203_MAPPING_REPLAY_INVALID")
    candidate = payloads["gate_record"]
    if candidate != {
        "schema_version": "stage23-task-gate-candidate-v1",
        "task_id": "stage2.03_assets_checkpoints_and_sampling",
        "gate_ids": ["stage2.G2.1"],
        "gate_status": "NOT_RUN",
        "local_validation_status": "NOT_RUN",
        "formal_eligible": False,
        "reason": "formal_gate_requires_independent_review",
    }:
        raise ValueError("STAGE23_S203_GATE_CANDIDATE_INVALID")


def validate_formal_s203_task_artifacts(
    workspace_root: str | Path,
    refs: Mapping[str, str],
) -> tuple[dict[str, LoadedTaskArtifact], str]:
    """Discover the complete S2.3 formal envelope through the shared loader."""
    kinds = ("sampling_plan", "draw_manifest", "asset_resolution", "gate_record")
    if tuple(refs) != kinds:
        raise ValueError("STAGE23_S203_FORMAL_ARTIFACTS_INCOMPLETE")
    loaded = {
        kind: load_committed_task_artifact(workspace_root, refs[kind], require_formal=True)
        for kind in kinds
    }
    if any(item.identity.task_id != "stage2.03_assets_checkpoints_and_sampling" or item.identity.artifact_kind != kind for kind, item in loaded.items()):
        raise ValueError("STAGE23_S203_FORMAL_ARTIFACT_IDENTITY_INVALID")
    hashes = {item.identity.config_hash for item in loaded.values()}
    if len(hashes) != 1:
        raise ValueError("STAGE23_S203_FORMAL_CONFIG_HASH_MISMATCH")
    sources = {item.source_refs for item in loaded.values()}
    if len(sources) != 1 or not next(iter(sources), ()):
        raise ValueError("STAGE23_S203_FORMAL_SOURCE_REFS_INVALID")
    validate_formal_s203_payloads({kind: item.payload for kind, item in loaded.items()})
    return loaded, next(iter(hashes))


def _formal_execution_evidence(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[FormalExecutionEvidence, str]:
    reference = request.environment.evidence_refs.get("formal_execution")
    if reference is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_execution_evidence",
            "formal Stage 2/3 缺少 FormalExecutionEvidence commit 引用",
        )
    try:
        value = load_canonical_json(
            _workspace_path(root, reference, field="formal_execution")
        )
        if not isinstance(value, Mapping):
            raise ValueError("FORMAL_EXECUTION_EVIDENCE_ROOT_NOT_OBJECT")
        evidence = FormalExecutionEvidence.from_mapping(value)
        evidence.require_for_stage(request.task.stage)
    except (FileNotFoundError, ValueError, TypeError, FormalRunRejected) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_execution_evidence",
            f"FormalExecutionEvidence 不可用：{type(error).__name__}: {error}",
            evidence_refs=(reference,),
        ) from error

    stage1_ref = request.environment.evidence_refs.get("stage1_g1_exit")
    if not isinstance(stage1_ref, str) or not stage1_ref:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage1_g1_exit",
            "formal Stage 2/3 必须绑定 Stage 1 G1-EXIT 正式 index，不能消费 Git fixture",
            evidence_refs=(reference,),
        )
    try:
        validate_stage1_exit_evidence(root, stage1_ref)
    except (Stage1HandoffError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage1_g1_exit",
            f"Stage 1 G1-EXIT 正式 handoff 不可接受：{type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(stage1_ref, reference),
        ) from error

    stage0_ref = request.environment.evidence_refs.get("stage0_handoff")
    if not isinstance(stage0_ref, str) or not stage0_ref:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage0_handoff",
            "formal Stage 2/3 必须绑定 Stage 0 handoff manifest",
            evidence_refs=(reference, stage1_ref),
        )
    try:
        # ``require_ready`` is deliberate: a complete historical role/hash
        # index is not a current hardware or persistence authorization.
        validate_stage0_handoff(root, stage0_ref, require_ready=True)
    except (Stage0HandoffError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage0_handoff",
            f"Stage 0 handoff 不可作为当前 formal authority：{type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(stage0_ref, stage1_ref, reference),
        ) from error

    required_gates = set(request.task.formal_eligibility.required_gate_ids)
    evidence_gates = {gate.gate_id for gate in evidence.prerequisite_gates}
    missing = sorted(required_gates - evidence_gates)
    if missing:
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            missing[0],
            f"FormalExecutionEvidence 未绑定任务所需 Gate：{missing}",
            evidence_refs=(reference,),
        )
    stage1_gates = tuple(
        gate for gate in evidence.prerequisite_gates if gate.gate_id == "stage1.G1-EXIT"
    )
    if len(stage1_gates) != 1 or stage1_ref not in stage1_gates[0].evidence_refs:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage1_g1_exit",
            "FormalExecutionEvidence 的 stage1.G1-EXIT 未绑定同一正式 index ref",
            retryable=False,
            evidence_refs=(stage1_ref, reference),
        )
    runtime_gates = set(request.environment.passed_gate_ids)
    if not required_gates.issubset(runtime_gates):
        missing_runtime = sorted(required_gates - runtime_gates)
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            missing_runtime[0],
            f"runtime environment 未确认 Gate：{missing_runtime}",
            evidence_refs=(reference,),
        )

    freeze_ref = request.environment.evidence_refs.get("contract_freeze")
    if freeze_ref is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "contract_freeze",
            "formal execution 必须绑定可重读的 contract freeze artifact",
            evidence_refs=(reference,),
        )
    try:
        observed = _formal_contract_freeze_hash(
            root,
            freeze_ref,
            stage=request.task.stage,
        )
    except Exception as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "contract_freeze",
            f"contract freeze artifact 不可读：{error}",
            evidence_refs=(freeze_ref,),
        ) from error
    if observed != evidence.contract_freeze_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "contract_freeze",
            "FormalExecutionEvidence 的 contract_freeze_hash 与文件不一致",
            retryable=False,
            evidence_refs=(freeze_ref, reference),
        )
    return evidence, reference


def _load_formal_parameter_registry(
    root: Path,
    reference: str,
    model: torch.nn.Module,
) -> ParameterRegistry:
    """Reload the hash-bound S2.3 coordinate registry for a live model."""

    registry_task = load_committed_task_artifact(
        root, reference, require_formal=True
    )
    if registry_task.identity.artifact_kind != "parameter_registry":
        raise ValueError("FORMAL_PARAMETER_REGISTRY_TASK_KIND_INVALID")
    binding = registry_task.payload
    source_ref = binding.get("source_s203_manifest_ref")
    source_sha256 = binding.get("source_s203_manifest_sha256")
    declared_registry_hash = binding.get("registry_hash")
    if not all(
        isinstance(value, str) and value
        for value in (source_ref, source_sha256, declared_registry_hash)
    ):
        raise ValueError("FORMAL_PARAMETER_REGISTRY_SOURCE_BINDING_INVALID")
    source_path = _workspace_path(
        root, source_ref, field="source_s203_manifest_ref"
    )
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("FORMAL_PARAMETER_REGISTRY_SOURCE_HASH_INVALID")
    source_manifest = load_canonical_json(source_path)
    if not isinstance(source_manifest, Mapping):
        raise ValueError("FORMAL_PARAMETER_REGISTRY_SOURCE_MANIFEST_INVALID")
    registry_manifest = source_manifest.get("registry")
    if not isinstance(registry_manifest, Mapping):
        raise ValueError("FORMAL_PARAMETER_REGISTRY_MANIFEST_MISSING")
    try:
        named_parameters = dict(model.named_parameters(remove_duplicate=False))
        registry = ParameterRegistry.from_manifest(
            registry_manifest, parameters=named_parameters
        )
    except Exception as error:
        raise ValueError("FORMAL_PARAMETER_REGISTRY_MANIFEST_INVALID") from error
    if registry.coordinate_registry_hash != declared_registry_hash:
        raise ValueError("FORMAL_PARAMETER_REGISTRY_COORDINATE_HASH_MISMATCH")
    return registry


@dataclass(frozen=True, slots=True)
class _FormalCheckpointBinding:
    """The exact S2.3 model root consumed by a formal Stage 2 provider."""

    model_id: str
    training_stage: str
    checkpoint_id: str
    revision: str
    root_ref: str
    manifest_ref: str
    manifest_sha256: str
    registry_hash: str
    config_hash: str
    root: Path

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "model_id": self.model_id,
            "training_stage": self.training_stage,
            "checkpoint_id": self.checkpoint_id,
            "revision": self.revision,
            "root_ref": self.root_ref,
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "registry_hash": self.registry_hash,
            "config_hash": self.config_hash,
        }


def _formal_external_payload(
    request: TaskExecutionRequest,
    root: Path,
    *,
    environment_key: str,
    expected_kind: str,
) -> LoadedTaskArtifact:
    """Load one formal materializer TaskArtifact with source-byte binding."""

    reference = request.environment.evidence_refs.get(environment_key)
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"FORMAL_CHECKPOINT_{environment_key.upper()}_REF_REQUIRED")
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    if loaded.identity.artifact_kind != expected_kind:
        raise ValueError(
            f"FORMAL_CHECKPOINT_{environment_key.upper()}_KIND_INVALID"
        )
    validate_external_manifest(loaded, root, expected_kind=expected_kind)
    if not isinstance(loaded.payload, Mapping):
        raise ValueError(f"FORMAL_CHECKPOINT_{environment_key.upper()}_PAYLOAD_INVALID")
    return loaded


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _validate_formal_checkpoint_root(
    root: Path,
    *,
    root_ref: str,
    manifest_ref: str,
    manifest_sha256: str,
    manifest: Mapping[str, object],
) -> Path:
    """Verify selected-root bytes before allowing Transformers to load them."""

    model_root = _workspace_path(root, root_ref, field="checkpoint_root_ref")
    manifest_path = _workspace_path(root, manifest_ref, field="checkpoint_manifest_ref")
    if not model_root.is_dir() or manifest_path.parent != model_root:
        raise ValueError("FORMAL_CHECKPOINT_ROOT_MANIFEST_PARENT_MISMATCH")
    if manifest_path.name != "model-manifest.json":
        raise ValueError("FORMAL_CHECKPOINT_MANIFEST_NAME_INVALID")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != manifest_sha256:
        raise ValueError("FORMAL_CHECKPOINT_SOURCE_MANIFEST_HASH_INVALID")
    if _SHA256_RE.fullmatch(manifest_sha256) is None:
        raise ValueError("FORMAL_CHECKPOINT_SOURCE_MANIFEST_SHA256_INVALID")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("FORMAL_CHECKPOINT_MANIFEST_FILES_INVALID")
    expected: dict[str, tuple[int, str]] = {}
    for item in files:
        if not isinstance(item, Mapping) or set(item) != {
            "name", "official_lfs_sha256", "sha256", "size_bytes"
        }:
            raise ValueError("FORMAL_CHECKPOINT_MANIFEST_FILE_SCHEMA_INVALID")
        name = item["name"]
        digest = item["sha256"]
        size = item["size_bytes"]
        official_lfs_sha256 = item["official_lfs_sha256"]
        if (
            not isinstance(name, str)
            or not name
            or name in expected
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or _SHA256_RE.fullmatch(digest) is None
            or (
                official_lfs_sha256 is not None
                and (
                    not isinstance(official_lfs_sha256, str)
                    or _SHA256_RE.fullmatch(official_lfs_sha256) is None
                )
            )
            or (
                name.endswith(".safetensors")
                and (
                    not isinstance(official_lfs_sha256, str)
                    or _SHA256_RE.fullmatch(official_lfs_sha256) is None
                )
            )
        ):
            raise ValueError("FORMAL_CHECKPOINT_MANIFEST_FILE_IDENTITY_INVALID")
        file_path = _workspace_path(
            root, f"{root_ref}/{name}", field="checkpoint_file"
        )
        if not file_path.is_file() or file_path.is_symlink():
            raise ValueError("FORMAL_CHECKPOINT_FILE_MISSING_OR_SYMLINK")
        expected[name] = (size, digest)
    for name, expected_identity in expected.items():
        if _sha256_file(model_root / name) != expected_identity:
            raise ValueError(f"FORMAL_CHECKPOINT_FILE_BYTES_INVALID:{name}")
    sidecar_path = model_root / "SHA256SUMS"
    if not sidecar_path.is_file() or sidecar_path.is_symlink():
        raise ValueError("FORMAL_CHECKPOINT_SIDECAR_MISSING_OR_SYMLINK")
    expected_sidecar = [
        (name, digest) for name, (_size, digest) in expected.items()
    ] + [("model-manifest.json", manifest_sha256)]
    observed_sidecar: list[tuple[str, str]] = []
    try:
        for line in sidecar_path.read_text(encoding="ascii").splitlines():
            if not line or "  " not in line:
                raise ValueError
            digest, name = line.split("  ", 1)
            if name.startswith("*"):
                name = name[1:]
            observed_sidecar.append((name, digest))
    except (OSError, UnicodeError, ValueError) as error:
        raise ValueError("FORMAL_CHECKPOINT_SIDECAR_INVALID") from error
    if observed_sidecar != [(name, digest) for name, digest in expected_sidecar]:
        raise ValueError("FORMAL_CHECKPOINT_SIDECAR_MISMATCH")
    actual_files: set[str] = set()
    for path in model_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("FORMAL_CHECKPOINT_ROOT_SYMLINK_REJECTED")
        if path.is_file():
            actual_files.add(path.relative_to(model_root).as_posix())
        elif not path.is_dir():
            raise ValueError("FORMAL_CHECKPOINT_ROOT_NONREGULAR_ENTRY")
    allowed_files = set(expected) | {"model-manifest.json", "SHA256SUMS"}
    if actual_files != allowed_files:
        raise ValueError("FORMAL_CHECKPOINT_ROOT_FILE_SET_INVALID")
    return model_root


def _load_formal_selected_checkpoint(
    request: TaskExecutionRequest,
    root: Path,
) -> _FormalCheckpointBinding | None:
    """Resolve and cross-check the selected S2.3 root for Stage 2.04--2.07."""

    if request.task.task_id not in _FORMAL_SELECTED_CHECKPOINT_TASKS:
        return None
    checkpoint_artifact = _formal_external_payload(
        request, root, environment_key="stage2_checkpoint_manifest", expected_kind="checkpoint_manifest"
    )
    six_cell_artifact = _formal_external_payload(
        request, root, environment_key="stage2_s23_six_cell_manifest", expected_kind="six_cell_manifest"
    )
    resolved_artifact = _formal_external_payload(
        request, root, environment_key="stage2_resolved_config", expected_kind="resolved_config"
    )
    registry_artifact = _formal_external_payload(
        request, root, environment_key="stage2_parameter_registry", expected_kind="parameter_registry"
    )
    source_artifacts = (
        checkpoint_artifact,
        six_cell_artifact,
        resolved_artifact,
        registry_artifact,
    )
    if any(
        item.identity.task_id != _STAGE2_REFERENCE_TASK
        or item.identity.formal_eligible is not True
        for item in source_artifacts
    ):
        raise ValueError("FORMAL_CHECKPOINT_SOURCE_PRODUCER_IDENTITY_INVALID")
    checkpoint = checkpoint_artifact.payload
    six_cell = six_cell_artifact.payload
    resolved = resolved_artifact.payload
    registry = registry_artifact.payload
    required = {
        "schema_version", "checkpoint_id", "model_id", "revision",
        "checkpoint_manifest", "source_manifest_ref", "source_manifest_sha256",
    }
    if set(checkpoint) != required or checkpoint["schema_version"] != "checkpoint-manifest-v1":
        raise ValueError("FORMAL_CHECKPOINT_BINDING_FIELDS_INVALID")
    model_id = checkpoint["model_id"]
    checkpoint_id = checkpoint["checkpoint_id"]
    revision = checkpoint["revision"]
    manifest_ref = checkpoint["source_manifest_ref"]
    manifest_sha256 = checkpoint["source_manifest_sha256"]
    manifest = checkpoint["checkpoint_manifest"]
    if not all(
        isinstance(value, str) and value
        for value in (model_id, checkpoint_id, revision, manifest_ref, manifest_sha256)
    ) or not isinstance(manifest, Mapping):
        raise ValueError("FORMAL_CHECKPOINT_BINDING_IDENTITY_INVALID")
    source_config_hash = resolved.get("config_hash")
    if not isinstance(source_config_hash, str) or _SHA256_RE.fullmatch(source_config_hash) is None:
        raise ValueError("FORMAL_CHECKPOINT_RESOLVED_CONFIG_HASH_INVALID")
    if any(
        item.identity.config_hash != source_config_hash
        for item in source_artifacts
    ):
        raise ValueError("FORMAL_CHECKPOINT_SOURCE_ARTIFACT_CONFIG_HASH_MISMATCH")
    # Downstream S2.5--2.7 envelopes may derive a new execution hash while
    # retaining the S2.4 resolved-config/checkpoint/registry identity.
    if (
        request.task.task_id == _STAGE2_REFERENCE_TASK
        and request.config.config_hash != source_config_hash
    ):
        raise ValueError("FORMAL_CHECKPOINT_SOURCE_CONFIG_HASH_MISMATCH")
    if canonical_json_hash(dict(manifest)) != manifest_sha256:
        raise ValueError("FORMAL_CHECKPOINT_MANIFEST_PAYLOAD_HASH_INVALID")
    if manifest.get("schema") != "parameter-importance-model-manifest-v1":
        raise ValueError("FORMAL_CHECKPOINT_MANIFEST_SCHEMA_INVALID")
    if manifest.get("revision") != revision:
        raise ValueError("FORMAL_CHECKPOINT_MANIFEST_REVISION_MISMATCH")

    base_model = request.config.base_config.section("model")
    base_identity = request.config.base_config.section("identity")
    if not isinstance(base_model, Mapping) or not isinstance(base_identity, Mapping):
        raise ValueError("FORMAL_CHECKPOINT_CONFIG_SECTIONS_INVALID")
    if (
        base_model.get("architecture") != model_id
        or base_model.get("asset_id") != PurePosixPath(manifest_ref).parent.name
        or base_model.get("revision") != revision
        or base_model.get("initialization_id") != checkpoint_id
        or base_identity.get("input_checkpoint_id") != checkpoint_id
    ):
        raise ValueError("FORMAL_CHECKPOINT_CONFIG_IDENTITY_MISMATCH")
    resolved_base = resolved.get("base_config")
    if not isinstance(resolved_base, Mapping):
        raise ValueError("FORMAL_CHECKPOINT_RESOLVED_CONFIG_BASE_MISSING")
    resolved_model = resolved_base.get("model")
    resolved_identity = resolved_base.get("identity")
    if not isinstance(resolved_model, Mapping) or not isinstance(resolved_identity, Mapping):
        raise ValueError("FORMAL_CHECKPOINT_RESOLVED_CONFIG_SECTIONS_INVALID")
    for section, expected in ((resolved_model, base_model), (resolved_identity, base_identity)):
        for field in ("architecture", "asset_id", "revision", "initialization_id", "input_checkpoint_id"):
            if field in expected and section.get(field) != expected.get(field):
                raise ValueError("FORMAL_CHECKPOINT_RESOLVED_CONFIG_IDENTITY_MISMATCH")

    rows = six_cell.get("checkpoints")
    if six_cell.get("schema_version") != "stage2-s204-six-cell-manifest-v1" or six_cell.get("status") != "READY" or six_cell.get("scope") != "formal" or not isinstance(rows, list):
        raise ValueError("FORMAL_CHECKPOINT_SIX_CELL_MANIFEST_INVALID")
    if len(rows) != len(EXPECTED_CELL_IDS) or tuple(
        row.get("cell_id") if isinstance(row, Mapping) else None for row in rows
    ) != EXPECTED_CELL_IDS:
        raise ValueError("FORMAL_CHECKPOINT_SIX_CELL_CELL_ORDER_INVALID")
    matches = [
        row for row in rows
        if isinstance(row, Mapping) and row.get("checkpoint_id") == checkpoint_id
    ]
    if len(matches) != 1:
        raise ValueError("FORMAL_CHECKPOINT_SIX_CELL_CHECKPOINT_NOT_UNIQUE")
    row = matches[0]
    row_fields = ("model_id", "training_stage", "checkpoint_root_ref", "checkpoint_manifest_ref", "checkpoint_revision", "checkpoint_hash", "registry_hash", "config_hash")
    if any(not isinstance(row.get(field), str) or not row.get(field) for field in row_fields):
        raise ValueError("FORMAL_CHECKPOINT_SIX_CELL_IDENTITY_INVALID")
    model_root_ref = PurePosixPath(manifest_ref).parent.as_posix()
    if (
        row["model_id"] != model_id
        or row["checkpoint_root_ref"] != model_root_ref
        or row["checkpoint_manifest_ref"] != manifest_ref
        or row["checkpoint_revision"] != revision
        or row["checkpoint_hash"] != manifest_sha256
        or row["config_hash"] != source_config_hash
    ):
        raise ValueError("FORMAL_CHECKPOINT_SIX_CELL_IDENTITY_MISMATCH")
    registry_hash = row["registry_hash"]
    if (
        registry.get("schema_version") != "stage2-parameter-registry-artifact-v1"
        or registry.get("status") != "READY"
        or registry.get("scope") != "formal"
        or registry.get("checkpoint_id") != checkpoint_id
        or registry.get("model_id") != model_id
        or registry.get("training_stage") != row["training_stage"]
        or registry.get("registry_hash") != registry_hash
    ):
        raise ValueError("FORMAL_CHECKPOINT_REGISTRY_IDENTITY_MISMATCH")
    selected_root = _validate_formal_checkpoint_root(
        root,
        root_ref=model_root_ref,
        manifest_ref=manifest_ref,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )
    return _FormalCheckpointBinding(
        model_id=str(model_id),
        training_stage=str(row["training_stage"]),
        checkpoint_id=str(checkpoint_id),
        revision=str(revision),
        root_ref=model_root_ref,
        manifest_ref=str(manifest_ref),
        manifest_sha256=str(manifest_sha256),
        registry_hash=str(registry_hash),
        config_hash=str(source_config_hash),
        root=selected_root,
    )


def _formal_provider(request: TaskExecutionRequest, root: Path) -> _ProviderContext:
    """验证离线资产并构造真实 Torch fixed-state provider。

    此函数没有 synthetic fallback。HF 依赖仍由 provider 的延迟导入负责；本机未安装
    ``transformers``/``datasets`` 时会转换为结构化 blocker。
    """

    if request.config.run_intent != "formal":
        raise RuntimeError("FORMAL_PROVIDER_REQUIRES_FORMAL_INTENT")
    evidence, evidence_ref = _formal_execution_evidence(request, root)
    stage1_ref = request.environment.evidence_refs["stage1_g1_exit"]
    stage0_ref = request.environment.evidence_refs["stage0_handoff"]
    try:
        stage1_exit = validate_stage1_exit_evidence(root, stage1_ref)
    except (Stage1HandoffError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        # Keep this second read deliberate: the provider context stores the
        # exact hashes it consumed, so later handoff payloads cannot infer them
        # from a mutable request mapping.
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage1_g1_exit",
            f"Stage 1 G1-EXIT 正式 handoff 不可接受：{type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(stage1_ref, evidence_ref),
        ) from error
    try:
        stage0_handoff = validate_stage0_handoff(root, stage0_ref, require_ready=True)
    except (Stage0HandoffError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage0_handoff",
            f"Stage 0 handoff 不可作为当前 formal authority：{type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(stage0_ref, stage1_ref, evidence_ref),
        ) from error
    providers = request.config.section("providers")
    if not isinstance(providers, dict) or providers.get("kind") != "offline_hf":
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "offline_hf_fixed_state_provider",
            "formal Stage 2/3 禁止 tiny/synthetic provider",
            retryable=False,
            evidence_refs=(evidence_ref,),
        )
    reject_legacy_provider_paths(providers)

    resolution_ref = request.environment.evidence_refs.get("g3_resolution")
    selected_checkpoint: _FormalCheckpointBinding | None = None
    try:
        runtime_assets = FormalG3RuntimeAssets.from_request(request, root)
        base = request.config.base_config
        model_config = base.section("model")
        data = base.section("data")
        runtime = base.section("runtime")
        identity = base.section("identity")
        assert all(
            isinstance(value, dict)
            for value in (model_config, data, runtime, identity)
        )
        task_type = str(providers["task_type"])
        expected_data_kind = (
            "pile" if task_type == "causal_lm" else "glue_derived"
        )
        # G3 qualifies the immutable model initialization (step 0).  The
        # selected S2.3 checkpoint identity remains in the config and in the
        # six-cell lineage, but is not a G3 logical asset alias.  Project the
        # runtime load to the qualified base model instead of inventing a
        # trained-checkpoint G3 entry.
        architecture = model_config.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise G3RuntimeAssetError("G3_RUNTIME_MODEL_ARCHITECTURE_REQUIRED")
        model_asset = runtime_assets.resolve(
            f"{architecture}-step0", expected_kind="model"
        )
        data_asset = runtime_assets.resolve(
            str(data["asset_id"]), expected_kind=expected_data_kind
        )
        tokenizer_asset = runtime_assets.resolve(
            str(model_config["tokenizer_asset_id"]), expected_kind="tokenizer"
        )
        # G3 continues to authorize the base model/data/tokenizer assets.  The
        # selected S2.3 checkpoint is a separate hash-bound input and is loaded
        # only for the four formal Stage 2 provider consumers.
        selected_checkpoint = _load_formal_selected_checkpoint(request, root)
    except (FileNotFoundError, OSError, G3RuntimeAssetError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "qualified_g3_runtime_assets",
            f"qualified G3 runtime assets unavailable: {type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(resolution_ref,) if isinstance(resolution_ref, str) else (),
        ) from error

    qualified_assets = (model_asset, data_asset, tokenizer_asset)
    runtime_lineage_sha256 = runtime_assets.runtime_lineage_sha256(
        *qualified_assets
    )
    manifest_hashes = [item.ready_manifest_sha256 for item in qualified_assets]
    manifest_refs = [item.manifest_ref for item in qualified_assets]
    if not set(manifest_hashes).issubset(evidence.asset_manifest_hashes):
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "asset_manifest_hashes",
            "FormalExecutionEvidence 未覆盖配置所绑定的模型/数据/tokenizer manifest",
            retryable=False,
            evidence_refs=(evidence_ref, *manifest_refs),
        )

    try:
        model_root = (
            model_asset.resolved.root
            if selected_checkpoint is None
            else selected_checkpoint.root
        )
        model = OfflineHuggingFaceModelAdapter.from_local_directory(
            model_root,
            task_type=task_type,
            num_labels=providers["num_labels"],  # type: ignore[arg-type]
            # Frozen-state main gradients are preregistered in FP32; only
            # reference/statistic accumulation is FP64.  Coordinate registry
            # hashes intentionally exclude dtype, so this binds the existing
            # S2.3 coordinates without rerunning qualification.
            torch_dtype=torch.float32,
        )
        model.module.to(torch.device(str(runtime["device"])))
        model.module.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.module.parameters()):
            raise ValueError("FORMAL_MAIN_GRADIENT_MODEL_DTYPE_NOT_FLOAT32")
        if task_type == "causal_lm":
            if data_asset.storage_kind != "pythia_mmap_shards":
                raise G3RuntimeAssetError(
                    "G3_RUNTIME_STAGE23_PILE_STORAGE_KIND_INVALID"
                )
            stage = int(identity["stage"])
            route = formal_pile_route(
                stage=stage,
                evaluation=False,
                declared_sampling_design=str(data["sampling_design"]),
                configured_split=str(data["split"]),
            )
            start, stop = runtime_assets.pile_split_interval(data_asset, route.split)
            runtime_assets.validate_pile_budget(
                stage=stage,
                split=route.split,
                requested_records=stop - start,
            )
            resolver = PythiaMMapFrozenSampleResolver(
                runtime_assets.pythia_dataset(data_asset, split=route.split),
                asset_id=data_asset.resolved.asset_id,
                ready_manifest_sha256=data_asset.ready_manifest_sha256,
                qualification_sha256=data_asset.qualification_artifact_hash,
                g3_resolution_artifact_hash=(
                    runtime_assets.resolution_artifact_hash
                ),
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=runtime_lineage_sha256,
                split_start=start,
                split_stop=stop,
                sampling_design=str(data["sampling_design"]),
                weights_exogenous=bool(data["weights_exogenous"]),
                common_mean_assumption=bool(data["common_mean_assumption"]),
            )
        elif task_type == "sequence_classification":
            if data_asset.storage_kind != "hf_load_from_disk":
                raise G3RuntimeAssetError(
                    "G3_RUNTIME_STAGE23_GLUE_STORAGE_KIND_INVALID"
                )
            glue_task = data_asset.require_glue_route(
                task_name=str(providers["task_name"]),
                split=str(data["split"]),
            )
            resolver = PretokenizedGlueDatasetAdapter(
                data_asset.resolved.root,
                task_name=glue_task,
                split=str(data["split"]),
                dataset_id=data_asset.resolved.asset_id,
                microbatch_size=1,
                microbatches_per_step=1,
                expected_asset_hash=data_asset.directory_content_sha256,
                allowed_root=data_asset.resolved.root,
                g3_resolution_artifact_hash=(
                    runtime_assets.resolution_artifact_hash
                ),
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=runtime_lineage_sha256,
            )
        else:
            raise ValueError("FORMAL_FIXED_STATE_TASK_TYPE_UNSUPPORTED")
        fixed_binding = {
            "g3_resolution_artifact_hash": runtime_assets.resolution_artifact_hash,
            "assets": [item.provenance() for item in qualified_assets],
            "selected_checkpoint": (
                selected_checkpoint.to_payload()
                if selected_checkpoint is not None
                else None
            ),
        }
        fixed_id = f"offline-{canonical_json_hash(fixed_binding)[:24]}"
        # The coordinate registry is an upstream S2.3 authority, not a
        # property to derive from the current model.  The per-cell auxiliary
        # TaskArtifact binds the selected S2.3 source manifest and its raw
        # bytes; reload both before binding current Parameter objects.
        registry_ref = request.environment.evidence_refs.get(
            "stage2_parameter_registry"
        )
        if not isinstance(registry_ref, str) or not registry_ref:
            raise ValueError("FORMAL_PARAMETER_REGISTRY_REF_REQUIRED")
        registry = _load_formal_parameter_registry(
            root, registry_ref, model.module
        )
        if (
            selected_checkpoint is not None
            and registry.coordinate_registry_hash
            != selected_checkpoint.registry_hash
        ):
            raise ValueError("FORMAL_CHECKPOINT_REGISTRY_MODEL_HASH_MISMATCH")
        provider = TorchFixedStateGradientProvider(
            model,
            resolver,
            fixed_state_id=fixed_id,
            registry=registry,
            output_dtype=torch.float32,
            gradient_chunk_size=1,
            enable_formal_batched=True,
            formal_batch_chunk_size=4,
        )
    except DependencyUnavailable as error:
        raise _blocked(
            BlockerCode.DEPENDENCY_UNAVAILABLE,
            error.dependency,
            str(error),
            evidence_refs=tuple(manifest_refs),
        ) from error
    except (
        FileNotFoundError,
        G3RuntimeAssetError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as error:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "offline_hf_fixed_state_provider",
            f"无法构造离线 fixed-state provider：{type(error).__name__}: {error}",
            evidence_refs=tuple(manifest_refs),
        ) from error

    return _ProviderContext(
        provider=provider,
        sample_ids=tuple(resolver.sample_ids),
        evidence=evidence,
        provider_kind="offline_hf_torch_fixed_state",
        asset_manifest_hashes=tuple(manifest_hashes),
        asset_provenance=tuple(item.provenance() for item in qualified_assets),
        stage1_exit=stage1_exit,
        stage0_handoff=stage0_handoff,
        checkpoint_identity=(
            selected_checkpoint.to_payload()
            if selected_checkpoint is not None
            else None
        ),
    )


def _provider_context(request: TaskExecutionRequest, root: Path) -> _ProviderContext:
    return (
        _local_provider(request)
        if request.config.run_intent == "local_fixture"
        else _formal_provider(request, root)
    )


def _sampling_plan(request: TaskExecutionRequest, context: _ProviderContext) -> SamplingPlan:
    return _sampling_plan_for_ids(
        request,
        context.sample_ids,
        universe_id=f"{context.provider.fixed_state_id}-universe",
        metadata={
            "registry_hash": context.provider.registry_hash,
            "provider_state_digest": context.provider.state_digest(),
        },
    )


def _sampling_plan_provider_compatible(
    upstream: SamplingPlan,
    provider_plan: SamplingPlan,
) -> bool:
    """Check the immutable algorithm/seed boundary for a provider projection.

    S2.3 freezes a logical sampling universe, while the later G3 fixed-state
    provider exposes the qualified physical universe.  Their universe IDs and
    ordered sample IDs are therefore intentionally different.  The immutable
    handoff boundary is the draw algorithm, stream seed namespace, and
    sampling design; the caller must explicitly project the upstream streams
    onto the provider universe before requesting any gradients.
    """

    return (
        upstream.algorithm_version == provider_plan.algorithm_version
        and dict(upstream.stream_seeds) == dict(provider_plan.stream_seeds)
        and upstream.universe.sampling_design
        == provider_plan.universe.sampling_design
        and bool(provider_plan.universe.sample_ids)
    )


def _project_sampling_plan_to_provider(
    upstream: SamplingPlan,
    provider_plan: SamplingPlan,
) -> SamplingPlan:
    """Project committed S2.3 seeds onto the qualified G3 sample universe.

    The S2.3 artifact remains an immutable source ref.  Only its stream seed
    namespace and draw algorithm cross the S2.2/S2.4 boundary; physical sample
    IDs come from the fixed-state provider so every draw is resolvable.  The
    upstream hash is retained in projection metadata for audit/replay.
    """

    if not _sampling_plan_provider_compatible(upstream, provider_plan):
        raise ValueError("STAGE2_SAMPLING_PROVIDER_PROJECTION_INCOMPATIBLE")
    metadata = dict(provider_plan.universe.metadata)
    metadata.update(
        {
            "projection_schema": "stage2-sampling-provider-projection-v1",
            "upstream_sampling_plan_hash": upstream.digest,
        }
    )
    return SamplingPlan(
        SamplingUniverse(
            provider_plan.universe.universe_id,
            provider_plan.universe.sample_ids,
            metadata=metadata,
        ),
        upstream.stream_seeds,
        algorithm_version=upstream.algorithm_version,
    )


def _sampling_plan_for_ids(
    request: TaskExecutionRequest,
    sample_ids: Sequence[Hashable],
    *,
    universe_id: str,
    metadata: Mapping[str, object],
) -> SamplingPlan:
    """Build the frozen stream plan without requiring a gradient provider.

    Formal S2.3 validates the six-cell asset resolution independently of the
    later fixed-state gradient provider, so it still needs a deterministic
    empirical-universe plan while no model is queried.
    """

    identity = request.config.base_config.section("identity")
    seed_plan = SeedPlan.from_master_seed(int(identity["master_seed"]))
    universe = SamplingUniverse(
        universe_id=universe_id,
        sample_ids=tuple(sample_ids),
        metadata=metadata,
    )
    return SamplingPlan(
        universe=universe,
        stream_seeds={name: seed_plan.seed_for(name) for name in STREAM_NAMES},
    )


def _formal_stage2_asset_manifest(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[AssetResolutionManifest, str]:
    """Load the independently published formal S2.3 asset resolution.

    S2.3 must not synthesize a fixture matrix under ``formal``.  The manifest
    is supplied through environment evidence so this task remains parallel to
    the S2.2 handoff and can be audited without reading gradients.
    """

    reference = request.environment.evidence_refs.get("stage2_asset_resolution")
    if reference is None:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage2_asset_resolution",
            "formal S2.3 requires an environment-bound asset resolution manifest",
        )
    try:
        value = load_canonical_json(_workspace_path(root, reference, field="stage2_asset_resolution"))
        if not isinstance(value, Mapping):
            raise ValueError("stage2 asset resolution must be an object")
        manifest = AssetResolutionManifest.from_mapping(value)
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage2_asset_resolution",
            f"formal S2.3 asset resolution is unreadable: {type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(reference,),
        ) from error
    if manifest.scope != "formal" or manifest.status != "READY":
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage2_asset_resolution",
            "formal S2.3 requires a READY formal six-cell asset matrix",
            retryable=False,
            evidence_refs=(reference,),
        )
    try:
        validate_formal_asset_identity(manifest)
    except ValueError as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_asset_identity",
            f"formal S2.3 asset identity drifted: {error}",
            retryable=False,
            evidence_refs=(reference,),
        ) from error
    return manifest, reference


def _run_formal_stage2_assets_and_sampling(
    request: TaskExecutionRequest,
    root: Path,
    inputs: _PredecessorContext,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """Publish the formal S2.3 candidate from the independent asset evidence."""

    assets, _asset_reference = _formal_stage2_asset_manifest(request, root)
    data_range = assets.data_range
    sample_ids = tuple(range(data_range.sample_id_min, data_range.sample_id_max_exclusive))
    sampling = _sampling_plan_for_ids(
        request,
        sample_ids,
        universe_id=f"{assets.digest}-universe",
        metadata={
            "asset_resolution_hash": assets.digest,
            "data_range_hash": data_range.digest,
            "sampling_design": data_range.sampling_design,
            "upstream_binding_hash": inputs.binding_hash,
        },
    )
    if SamplingPlan.from_mapping(sampling.to_dict()).digest != sampling.digest:
        raise RuntimeError("STAGE2_FORMAL_SAMPLING_PLAN_ROUNDTRIP_DRIFT")
    stream_manifests = {
        stream: sampling.draw_manifest(stream, 4).to_manifest()
        for stream in STREAM_NAMES
    }
    draw_rows = [
        draw.to_manifest()
        for stream in STREAM_NAMES
        for draw in sampling.draws(stream, 4)  # type: ignore[arg-type]
    ]
    if len({str(row["draw_id"]) for row in draw_rows}) != len(draw_rows):
        raise RuntimeError("STAGE2_FORMAL_DRAW_ID_COLLISION")
    nested_mapping = RepetitionMapping.create(
        repetition_id="stage2-formal-sampling-nested-fixture",
        draws=sampling.draws("pilot", 8),
        m_values=(2, 4, 8),
    )
    provider_payload: dict[str, JSONValue] = {
        "provider_kind": "offline_hf_stage2_asset_manifest",
        "asset_resolution_hash": assets.digest,
        "data_range_hash": data_range.digest,
        "sample_universe_size": len(sample_ids),
        "sampling_design": data_range.sampling_design,
        "asset_manifest_hashes": [assets.digest],
        "asset_provenance": [
            {
                "model_id": record.model_id,
                "training_stage": record.training_stage,
                "revision": record.revision,
                "load_evidence_ref": record.load_evidence_ref,
            }
            for record in assets.checkpoints
        ],
    }
    payloads: dict[str, Mapping[str, JSONValue]] = {
            "sampling_plan": sampling.to_dict(),  # type: ignore[dict-item]
            "draw_manifest": {
                "schema_version": "stage2-task-draw-manifest-v1",
                "sampling_plan_hash": sampling.digest,
                "draws": draw_rows,  # type: ignore[dict-item]
                "draw_count_by_stream": {name: 4 for name in STREAM_NAMES},
                "stream_manifests": stream_manifests,
                "draw_id_unique": True,
                "sample_id_collisions_allowed": True,
                "replay_hash": canonical_json_hash(draw_rows),
                "nested_mapping": nested_mapping.to_dict(),
                "nested_mapping_hash": nested_mapping.digest,
            },
            "asset_resolution": {
                "schema_version": "stage2-task-asset-resolution-v1",
                "provider": provider_payload,
                "stage2_asset_manifest": assets.to_dict(),
                "preregistration_contract_hash": canonical_json_hash(
                    inputs.payload("preregistration")
                ),
                "upstream_binding_hash": inputs.binding_hash,
                "formal_eligible": False,
            },
            "gate_record": _gate_candidate(request),
    }
    validate_formal_s203_payloads(
        payloads,
        expected_preregistration_hash=canonical_json_hash(inputs.payload("preregistration")),
        expected_upstream_binding_hash=inputs.binding_hash,
    )
    # The resolved config's input_result_refs are the complete canonical
    # predecessor lineage.  The environment-bound asset manifest is already
    # embedded and hash-bound in ``asset_resolution``; it is evidence, not an
    # implicit source_ref.  Keeping source_refs identical to the config avoids
    # consumers silently deleting or adding references during Gate review.
    return payloads, inputs.references


def _stage2_source_identity() -> tuple[str, str | None, str | None]:
    """Resolve the reproducibility bindings without writing to the workspace.

    Local fixture tests execute with a temporary artifact root, so the repository
    identity comes from this source module's worktree rather than that output root.
    Stage1 formal provenance is supplied by the immutable DATA_ROOT bridge; the
    tracked ``reports/...local_fixture`` file is intentionally never used.
    """

    repository_root = Path(__file__).resolve().parents[3]
    try:
        producer_commit = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        producer_commit = "unresolved-local-source"

    def file_hash(relative: str) -> str | None:
        path = repository_root / Path(relative)
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()

    return (producer_commit, file_hash("docs/mathematics.md"), None)


def _stage1_formal_bridge_identity(
    root: Path,
    inputs: _PredecessorContext,
) -> Mapping[str, JSONValue] | None:
    """Load and verify the canonical S1.11 manifest from formal predecessor commits.

    Formal S1.11 TaskArtifact envelopes carry the complete source closure but do
    not carry the historical ``stage1-11-bridge-evidence.json`` compatibility
    ref.  The manifest is therefore discovered from the common output root of
    the four already-verified commit refs and returned byte-for-byte as the
    evaluator's ``bridge_payload``.
    """

    expected_kinds = (
        "stage_report",
        "requirements_matrix",
        "gate_summary",
        "delivery_manifest",
    )
    if tuple(item.artifact_kind for item in inputs.artifacts) != expected_kinds:
        raise ValueError("STAGE2_STAGE1_MANIFEST_ARTIFACT_SET_INVALID")
    output_roots: set[str] = set()
    for item in inputs.artifacts:
        commit_path = PurePosixPath(item.commit_ref)
        if len(commit_path.parts) < 3 or commit_path.parent.name != "commits":
            raise ValueError("STAGE2_STAGE1_MANIFEST_OUTPUT_ROOT_INVALID")
        output_roots.add(commit_path.parent.parent.as_posix())
    if len(output_roots) != 1:
        raise ValueError("STAGE2_STAGE1_MANIFEST_OUTPUT_ROOT_NOT_COMMON")
    output_root = next(iter(output_roots))
    manifest_ref = f"{output_root}/manifest.json"
    value = load_canonical_json(
        _workspace_path(root, manifest_ref, field="stage1_manifest")
    )
    if not isinstance(value, Mapping):
        raise ValueError("STAGE2_STAGE1_MANIFEST_NOT_OBJECT")

    def is_sha256(candidate: object) -> bool:
        return (
            isinstance(candidate, str)
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        )

    manifest_hash = value.get("artifact_hash")
    if not is_sha256(manifest_hash) or canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ) != manifest_hash:
        raise ValueError("STAGE2_STAGE1_MANIFEST_SELF_HASH_INVALID")
    if (
        value.get("schema_version") != "stage1-s1-11-task-artifact-manifest-v2"
        or value.get("status") != "PASS"
        or value.get("run_intent") != "formal"
        or value.get("formal_eligible") is not True
        or value.get("task_id") != "stage1.11_reporting_and_exit_gate"
        or value.get("gate_id") != "G1-EXIT"
    ):
        raise ValueError("STAGE2_STAGE1_MANIFEST_IDENTITY_INVALID")

    config_hashes = {item.config_hash for item in inputs.artifacts}
    if len(config_hashes) != 1:
        raise ValueError("STAGE2_STAGE1_MANIFEST_CONFIG_IDENTITY_INVALID")
    config_hash = next(iter(config_hashes))
    config_ref = value.get("config_ref")
    if config_ref != f"{output_root}/producer-config.json" or value.get(
        "config_hash"
    ) != config_hash:
        raise ValueError("STAGE2_STAGE1_MANIFEST_CONFIG_INVALID")
    config = load_canonical_json(
        _workspace_path(root, str(config_ref), field="stage1_manifest_config")
    )
    if (
        not isinstance(config, Mapping)
        or config.get("config_hash") != config_hash
        or canonical_json_hash(
            {key: item for key, item in config.items() if key != "config_hash"}
        )
        != config_hash
    ):
        raise ValueError("STAGE2_STAGE1_MANIFEST_CONFIG_SELF_HASH_INVALID")

    expected_commit_refs = {
        item.artifact_kind: item.commit_ref for item in inputs.artifacts
    }
    expected_envelope_hashes = {
        item.artifact_kind: item.artifact_hash for item in inputs.artifacts
    }
    if value.get("commit_refs") != expected_commit_refs:
        raise ValueError("STAGE2_STAGE1_MANIFEST_COMMIT_REFS_INVALID")
    if value.get("commit_artifact_hashes") != expected_envelope_hashes:
        raise ValueError("STAGE2_STAGE1_MANIFEST_ENVELOPE_HASHES_INVALID")
    return dict(value)


def _formal_stage1_report_artifact_hash(inputs: _PredecessorContext) -> str:
    """Return the authoritative envelope hash for the formal Stage 1 report.

    ``_BoundInputArtifact.artifact_hash`` is copied from the verified task commit;
    it is intentionally distinct from a payload field named ``artifact_hash``.
    The latter is a business-payload hash and is not a valid Stage 1 provenance
    binding for the formal evaluator.
    """

    matches = [
        item
        for item in inputs.artifacts
        if item.artifact_kind == "stage_report"
    ]
    if len(matches) != 1:
        raise ValueError("STAGE2_STAGE1_REPORT_ARTIFACT_NOT_UNIQUE")
    report = matches[0]
    if report.run_intent != "formal" or report.formal_eligible is not True:
        raise ValueError("STAGE2_STAGE1_REPORT_FORMAL_ARTIFACT_REQUIRED")
    return report.artifact_hash


def _run_stage2_contract(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """冻结 Stage 2 可在无模型资产条件下确定的 estimand 与选择边界。"""

    inputs = _predecessor_context(request, root, store)
    identity = request.config.base_config.section("identity")
    sampling_config = request.config.base_config.section("sampling")
    frozen_candidates = {
        "candidate_batch_sizes": CANDIDATE_BATCH_SIZES,
        "candidate_microbatch_counts": CANDIDATE_MICROBATCH_COUNTS,
        "microbatch_preference": MICROBATCH_SELECTION_ORDER,
    }
    for field_name, expected in frozen_candidates.items():
        if tuple(sampling_config[field_name]) != expected:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                f"sampling.{field_name}",
                f"resolved config 与冻结候选集合不一致：expected={list(expected)}",
                retryable=False,
            )
    seed_plan = SeedPlan.from_master_seed(int(identity["master_seed"]))
    producer_commit, mathematics_hash, stage1_report_hash = _stage2_source_identity()
    stage1_handoff = (
        _stage1_formal_bridge_identity(root, inputs)
        if request.config.run_intent == "formal"
        else None
    )
    formal_stage1_report_hash = (
        _formal_stage1_report_artifact_hash(inputs)
        if request.config.run_intent == "formal"
        else None
    )
    preregistration = build_stage2_preregistration(
        seed_plan_hash=seed_plan.artifact_hash,
        producer_commit=producer_commit,
        mathematics_hash=mathematics_hash,
        # Formal provenance binds the verified Stage 1 task-artifact envelope;
        # the tracked report remains local-draft compatibility only.
        stage1_report_hash=(
            formal_stage1_report_hash
            if request.config.run_intent == "formal"
            else stage1_report_hash
        ),
        upstream_binding_hash=inputs.binding_hash,
        stage1_handoff=stage1_handoff,
        scope=request.config.run_intent,
    )
    validate_stage2_preregistration(preregistration)
    hypothesis = build_stage2_hypothesis_contract(
        preregistration,
        upstream_binding_hash=inputs.binding_hash,
    )
    gate = _gate_candidate(request)
    gate.update(
        {
            "gate_id": "stage2.G2.0",
            "preregistration_hash": preregistration["preregistration_hash"],
            "hypothesis_contract_hash": hypothesis["hypothesis_contract_hash"],
            "quality_gate_status": "NOT_RUN",
            "sample_generation_status": "FORBIDDEN_UNTIL_COMMITTED",
        }
    )
    return (
        {
            "preregistration": preregistration,
            "hypothesis_contract": hypothesis,
            "gate_record": gate,
        },
        inputs.references,
    )


def _run_stage2_handoff_audit(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """复核预注册产物，并用真实 provider 状态摘要冻结 fixed-state 交接。"""

    inputs = _predecessor_context(request, root, store)
    preregistration = inputs.payload("preregistration")
    hypothesis = inputs.payload("hypothesis_contract")
    if tuple(preregistration.get("sampling_stream_names", ())) != STREAM_NAMES:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_sampling_stream_contract",
            "预注册 artifact 的五条 sampling stream 与冻结合同不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    if hypothesis.get("statistical_unit") != "independent_repetition":
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_statistical_unit",
            "预注册 artifact 未冻结 independent repetition 统计单位",
            retryable=False,
            evidence_refs=inputs.references,
        )
    context = _provider_context(request, root)
    if request.config.run_intent == "formal":
        if context.stage1_exit is None:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "stage1_g1_exit",
                "formal fixed-state context 缺少已验证的 Stage 1 G1-EXIT handoff",
                retryable=False,
                evidence_refs=inputs.references,
            )
        if context.stage0_handoff is None:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "stage0_handoff",
                "formal fixed-state context 缺少已验证的 Stage 0 handoff",
                retryable=False,
                evidence_refs=inputs.references,
            )
        # ``_formal_provider`` 已完成 offline_hf 类型、三类资产 manifest、
        # FormalExecutionEvidence 与模型状态装载验证。这里再次核对返回上下文，既
        # 防止未来 adapter 漂移，也保证测试替身不能偷偷换成 synthetic provider。
        try:
            context.evidence.require_for_stage(2)
        except FormalRunRejected as error:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "formal_fixed_state_handoff_evidence",
                f"formal fixed-state provider 证据不可接受：{error}",
                retryable=False,
                evidence_refs=inputs.references,
            ) from error
        stage1_ref = request.environment.evidence_refs.get("stage1_g1_exit")
        stage1_gates = tuple(
            gate
            for gate in context.evidence.prerequisite_gates
            if gate.gate_id == "stage1.G1-EXIT"
        )
        if not isinstance(stage1_ref, str) or len(stage1_gates) != 1 or stage1_ref not in stage1_gates[0].evidence_refs:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "stage1_g1_exit",
                "formal fixed-state context 的 G1-EXIT 未绑定同一正式 index ref",
                retryable=False,
                evidence_refs=inputs.references,
            )
        if (
            context.provider_kind != "offline_hf_torch_fixed_state"
            or not context.asset_manifest_hashes
            or not set(context.asset_manifest_hashes).issubset(
                context.evidence.asset_manifest_hashes
            )
        ):
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                "offline_hf_fixed_state_provider",
                "formal handoff 只接受 evidence-bound offline_hf fixed-state provider",
                retryable=False,
                evidence_refs=inputs.references,
            )
    sampling = _sampling_plan(request, context)
    invariant_mapping = RepetitionMapping.create(
        repetition_id="stage2-handoff-invariant",
        draws=sampling.draws("pilot", 8),
        m_values=(2, 4),
    )
    before_digest = context.provider.state_digest()
    invariant = PairedEstimatorRunner(context.provider, m2_tolerance=1e-10).run(
        invariant_mapping
    )
    after_digest = context.provider.state_digest()
    if before_digest != after_digest:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "fixed_state_state_digest",
            "fixed-state gradient queries 改变了 provider/model state digest",
            retryable=False,
            evidence_refs=inputs.references,
        )
    negative_u_count = int(
        np.count_nonzero(_flatten(invariant.u_by_m[4]) < 0)
    )
    candidate_status = (
        "FORMAL_CANDIDATE"
        if request.config.run_intent == "formal"
        else "FIXTURE_VALIDATED"
    )
    handoff: dict[str, JSONValue] = {
        "schema_version": "stage2-task-handoff-manifest-v1",
        "source_task_id": inputs.predecessor_task_ids[0],
        "source_artifact_hashes": [item.artifact_hash for item in inputs.artifacts],
        "upstream_binding_hash": inputs.binding_hash,
        "provider_state_digest": context.provider.state_digest(),
        "registry_hash": context.provider.registry_hash,
        "scope": request.config.run_intent,
        "status": candidate_status,
        "stage1_g1_exit": (
            context.stage1_exit.to_dict() if context.stage1_exit is not None else None
        ),
        "stage0_handoff": (
            context.stage0_handoff.to_dict()
            if context.stage0_handoff is not None
            else None
        ),
        # handoff invariant 成功不等于本阶段 Gate 已通过；任务 envelope 会保留 formal
        # 执行身份，而科学 artifact 仍等待独立审核。
        "formal_eligible": False,
    }
    fixed_state: dict[str, JSONValue] = {
        "schema_version": "stage2-task-fixed-state-contract-v1",
        "contract_version": "stage2-fixed-state-contract-v1",
        "fixed_state_id": context.provider.fixed_state_id,
        "provider_state_digest": context.provider.state_digest(),
        "registry_hash": context.provider.registry_hash,
        "parameter_names": list(context.provider.parameter_names),
        "weighting_assumptions": context.to_payload()["weighting_assumptions"],
        "mutation_policy": "read_only_gradient_queries",
        "formula_contract": {
            "formula_version": "stage2-fixed-state-local-gradient-square-v1",
            "raw": "mean_gradient**2",
            "double": "mean_gradient_A*mean_gradient_B",
            "equal_weight_u": "(S1**2-S2)/(M*(M-1))",
            "weighted_u": "(G1**2-G2)/(N1**2-N2)",
            "signed_u_preserved": True,
            "clamp_min_zero": False,
        },
        "provider_api_contract": {
            "api_version": "fixed-state-gradient-provider-v1",
            "gradient_method": "gradient(draws)",
            "state_digest_method": "state_digest()",
            "registry_hash_property": "registry_hash",
            "parameter_names_property": "parameter_names",
            "output_dtype_property": "output_dtype",
            "gradient_chunk_size_property": "gradient_chunk_size",
            "gradient_chunk_size": getattr(context.provider, "gradient_chunk_size", None),
            "formal_batched_execution_property": "enable_formal_batched",
            "formal_batched_execution": bool(
                getattr(context.provider, "enable_formal_batched", False)
            ),
            "formal_batch_chunk_size_property": "formal_batch_chunk_size",
            "formal_batch_chunk_size": getattr(
                context.provider, "formal_batch_chunk_size", None
            ),
            "mutation_policy": "read_only_gradient_queries",
        },
        "state_contract": {
            "model_mode": "eval",
            "optimizer_step": "forbidden",
            "scheduler_step": "forbidden",
            "gradient_clipping": "disabled",
            "main_gradient_dtype": "float32",
            "provider_output_dtype": str(
                getattr(context.provider, "output_dtype", "not_exposed")
            ).replace("torch.", ""),
            "reference_accumulation_dtype": "float64",
            "loss_reduction": "mean_effective_target_token",
            "sampling_rng": "advances_by_manifest",
            "worker_rng": "advances_by_manifest",
        },
        "stage1_g1_exit": (
            context.stage1_exit.to_dict() if context.stage1_exit is not None else None
        ),
        "stage0_handoff": (
            context.stage0_handoff.to_dict()
            if context.stage0_handoff is not None
            else None
        ),
        "status": candidate_status,
        "validation_evidence": {
            "mapping_hash": invariant_mapping.digest,
            "provider_state_before": before_digest,
            "provider_state_after": after_digest,
            "state_unchanged": before_digest == after_digest,
            "m2_double_max_abs_error": invariant.m2_double_max_abs_error,
            "negative_u_coordinate_count": negative_u_count,
            "unclipped_u_preserved": True,
            "result_hash": invariant.digest,
        },
        "formal_eligible": False,
    }
    return (
        {
            "handoff_manifest": handoff,
            "fixed_state_contract": fixed_state,
            "gate_record": _gate_candidate(request),
        },
        tuple(dict.fromkeys((*inputs.references, *_formal_environment_source_refs(request)))),
    )


def _run_stage2_assets_and_sampling(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """解析 provider，并实际生成五条可重放 draw stream 的小型 manifest。"""

    inputs = _predecessor_context(request, root, store)
    # Formal S2.3 is a DAG sibling of S2.2 and consumes only S2.1 plus the
    # independently published asset evidence; it must never construct a
    # fixture matrix or bind to a handoff fixed-state contract.
    if request.config.run_intent == "formal":
        return _run_formal_stage2_assets_and_sampling(request, root, inputs)

    context = _provider_context(request, root)
    sampling = _sampling_plan(request, context)
    # 公共 loader round-trip 与同区间重放是抽样合同的一部分；两者都在发布前执行。
    if SamplingPlan.from_mapping(sampling.to_dict()).digest != sampling.digest:
        raise RuntimeError("STAGE2_SAMPLING_PLAN_ROUNDTRIP_DRIFT")
    for stream in STREAM_NAMES:
        if sampling.draws(stream, 4) != sampling.draws(stream, 4):  # type: ignore[arg-type]
            raise RuntimeError(f"STAGE2_DRAW_REPLAY_DRIFT:{stream}")
    nested_mapping = RepetitionMapping.create(
        repetition_id="stage2-sampling-nested-fixture",
        draws=sampling.draws("pilot", 8),
        m_values=(2, 4, 8),
    )
    stream_manifests = {
        stream: sampling.draw_manifest(stream, 4).to_manifest()
        for stream in STREAM_NAMES
    }
    draw_rows = [
        draw.to_manifest()
        for stream in STREAM_NAMES
        for draw in sampling.draws(stream, 4)  # type: ignore[arg-type]
    ]
    draw_ids = [str(row["draw_id"]) for row in draw_rows]
    if len(draw_ids) != len(set(draw_ids)):
        raise RuntimeError("STAGE2_DRAW_ID_COLLISION")
    # The fixture matrix exercises the same six-cell/checkpoint and data-range
    # wire contract as formal S2.3 while remaining explicitly synthetic.  It is
    # never accepted by the formal provider; formal assets must provide their
    # own immutable revisions and offline-load evidence.
    producer_commit, _, _ = _stage2_source_identity()
    fixture_revision = "0" * 40
    fixture_files = (
        CheckpointFile("model.safetensors", 1, "1" * 64, "weights"),
        CheckpointFile("config.json", 1, "2" * 64, "config"),
        CheckpointFile("tokenizer.json", 1, "4" * 64, "tokenizer"),
    )
    fixture_checkpoints = tuple(
        CheckpointRecord(
            model_id=model,
            training_stage=stage,
            checkpoint_id=f"fixture-{model}-{stage}",
            training_step={"initialization": 0, "early": 1, "mid_late": 50}[stage],
            total_training_steps=100,
            target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
            repository=f"fixture/{model}",
            revision=fixture_revision,
            root_ref=f"fixture/models/{model}/{stage}",
            state="ready",
            files=fixture_files,
            manifest_ref=f"fixture/manifests/{model}-{stage}.json",
            manifest_sha256="3" * 64,
            parameter_registry_hash=context.provider.registry_hash,
            config_sha256="2" * 64,
            tokenizer_sha256="4" * 64,
            load_status="passed",
            load_evidence_ref=f"fixture/evidence/{model}-{stage}.json",
            load_evidence_sha256="5" * 64,
        )
        for model in ("pythia-14m", "pythia-31m-deduped")
        for stage in ("initialization", "early", "mid_late")
    )
    fixture_data_range = DataRangeManifest(
        dataset_id="fixture-data-range",
        revision=fixture_revision,
        manifest_ref="fixture/manifests/data-range.json",
        manifest_sha256="6" * 64,
        files=(
            # The local provider does not read these paths.  They identify the
            # same two-file shape as the formal Pile allowlist.
            # Sizes/hashes are intentionally fixture values.
            DataFile("document-00000-of-00020.bin", 1, "7" * 64, "token_shard"),
            DataFile("document.idx", 1, "8" * 64, "index"),
        ),
        sample_id_max_exclusive=len(context.sample_ids),
    )
    fixture_assets = AssetResolutionManifest(
        scope="local_fixture",
        checkpoints=fixture_checkpoints,
        data_range=fixture_data_range,
        producer_commit=producer_commit,
        execution_commit=producer_commit,
    )
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "sampling_plan": sampling.to_dict(),  # type: ignore[dict-item]
        "draw_manifest": {
            "schema_version": "stage2-task-draw-manifest-v1",
            "sampling_plan_hash": sampling.digest,
            "draws": draw_rows,  # type: ignore[dict-item]
            "draw_count_by_stream": {name: 4 for name in STREAM_NAMES},
            "stream_manifests": stream_manifests,
            "draw_id_unique": True,
            "sample_id_collisions_allowed": True,
            "replay_hash": canonical_json_hash(draw_rows),
            "nested_mapping": nested_mapping.to_dict(),
            "nested_mapping_hash": nested_mapping.digest,
        },
        "asset_resolution": {
            "schema_version": "stage2-task-asset-resolution-v1",
            "provider": context.to_payload(),
            "stage2_asset_manifest": fixture_assets.to_dict(),
            "preregistration_contract_hash": canonical_json_hash(
                inputs.payload("preregistration")
            ),
            "upstream_binding_hash": inputs.binding_hash,
            "formal_eligible": False,
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, inputs.references


def _formal_input_document(
    request: TaskExecutionRequest,
    root: Path,
    *,
    schema_version: str,
    requirement: str,
) -> tuple[Mapping[str, object], str]:
    """从环境证据或兼容 input refs 唯一选择 formal plan。

    ``TaskRuntime`` 会把 ``orchestration.input_result_refs`` 严格解释为任务目录中的
    前驱 commit，因此正式辅助计划应通过 ``environment.evidence_refs`` 传入。保留
    对旧 input refs 的只读扫描，是为了让既有直接 runner 测试和历史配置能给出
    明确的歧义/缺失错误；正式 preflight 不会因此放宽前驱 commit 合同。
    """

    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    environment_ref = request.environment.evidence_refs.get(requirement)
    if environment_ref is not None:
        try:
            value = load_canonical_json(
                _workspace_path(root, environment_ref, field=requirement)
            )
        except (FileNotFoundError, ValueError) as error:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                requirement,
                f"正式辅助计划不可读：{type(error).__name__}: {error}",
                evidence_refs=(environment_ref,),
            ) from error
        if not isinstance(value, Mapping) or value.get("schema_version") != schema_version:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                requirement,
                f"环境证据不是所需 {schema_version}",
                retryable=False,
                evidence_refs=(environment_ref,),
            )
        return value, environment_ref

    matches: list[tuple[Mapping[str, object], str]] = []
    for reference in dict.fromkeys(
        str(item) for item in orchestration["input_result_refs"]
    ):
        try:
            value = load_canonical_json(
                _workspace_path(root, reference, field=requirement)
            )
        except (FileNotFoundError, ValueError):
            continue
        if isinstance(value, Mapping) and value.get("schema_version") == schema_version:
            matches.append((value, reference))
    if len(matches) != 1:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            f"formal 任务需要唯一 {schema_version}，当前找到 {len(matches)} 个",
            retryable=len(matches) == 0,
            evidence_refs=tuple(reference for _, reference in matches),
        )
    return matches[0]


def _stage2_reference_plan(
    request: TaskExecutionRequest,
    root: Path,
    context: _ProviderContext,
    authoritative: Mapping[str, object] | None = None,
) -> tuple[ReferenceSizingPlan, tuple[str, ...]]:
    if request.config.run_intent == "local_fixture":
        return (
            ReferenceSizingPlan(
                reference_id=f"reference-{request.config.config_hash[:16]}",
                candidate_sample_counts=(4, 8),
                block_size=2,
                convergence_tolerance=1e6,
                required_consecutive=1,
                execution=context.evidence,
            ),
            (),
        )
    if authoritative is not None:
        value = authoritative
        reference = str(request.environment.evidence_refs.get("stage2_reference_sizing_plan"))
    else:
        value, reference = _formal_input_document(
            request,
            root,
            schema_version="stage2-reference-sizing-plan-v1",
            requirement="formal_reference_sizing_plan",
        )
    if value.get("execution_evidence_hash") != context.evidence.artifact_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_reference_sizing_plan",
            "reference sizing plan 未绑定当前 FormalExecutionEvidence",
            retryable=False,
            evidence_refs=(reference,),
        )
    plan = ReferenceSizingPlan(
        reference_id=str(value["reference_id"]),
        candidate_sample_counts=tuple(value["candidate_sample_counts"]),  # type: ignore[arg-type]
        block_size=int(value["block_size"]),
        convergence_tolerance=float(value["convergence_tolerance"]),
        required_consecutive=int(value["required_consecutive"]),
        execution=context.evidence,
    )
    if value.get("artifact_hash") != plan.artifact_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_reference_sizing_plan",
            "reference sizing plan hash 漂移",
            retryable=False,
            evidence_refs=(reference,),
        )
    return plan, (reference,)


def _stable_reference_artifact(
    *,
    reference_id: str,
    result: object,
    block_size: int,
    bundle_ref: str,
    bundle_hash: str,
    metadata_extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """发布与恢复路径无关的 ``reference-result-v1``。

    formal core result 中的 ``resumed_from_block_pairs`` 是运行诊断，不属于 estimand，
    因此这里不把它间接带入共享 reference 身份。
    """

    bias = getattr(result, "bias_reference")
    cross = getattr(result, "cross_reference")
    ranking = getattr(result, "ranking_reference")
    selected = getattr(result, "selected_sample_count_per_stream", None)
    processed = getattr(result, "processed_sample_count_per_stream")
    scope = str(getattr(result, "scope", "local_fixture"))
    metadata: dict[str, object] = {
        "candidate_status": str(getattr(result, "status")),
        "converged": bool(getattr(result, "converged", True)),
        "weighting_assumptions": dict(getattr(result, "weighting_assumptions", {})),
        "qualification_gate_hash": None,
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    payload: dict[str, object] = {
        "schema_version": "reference-result-v1",
        "reference_id": reference_id,
        "bias_reference_hash": _vector_digest(bias),
        "cross_reference_hash": _vector_digest(cross),
        "ranking_reference_hash": _vector_digest(ranking),
        "sample_count_a": int(selected or processed),
        "sample_count_b": int(selected or processed),
        "block_size": block_size,
        "registry_hash": str(getattr(result, "registry_hash")),
        "scope": scope,
        "formal_eligible": False,
        "metadata": metadata,
        "tensor_bundle_ref": bundle_ref,
        "tensor_bundle_manifest_hash": bundle_hash,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    validate_reference_result_artifact(payload)
    return payload


def _reference_identity_hash(value: Mapping[str, object]) -> str:
    """Hash an identity object without allowing a caller-supplied digest."""

    return canonical_json_hash(dict(value))


def _reference_tokenizer_identity(
    external_tokenizer: object,
    *,
    formal: bool,
    expected_asset_id: object,
    expected_checkpoint_id: object,
    evidence_refs: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build the tokenizer identity from the v1/v2 boundary contract.

    ``tokenizer`` is not a v1 config section: its selector is
    ``model.tokenizer_asset_id``.  A formal S2.4 run additionally consumes the
    hash-bound tokenizer manifest published by S2.3; it must carry the selected
    checkpoint identity before the reference artifact is published.  The
    manifest's ``asset_id`` is a content identity and is therefore not compared
    directly with the logical selector name.
    """

    if not formal:
        identity = (
            {"asset_id": expected_asset_id}
            if isinstance(expected_asset_id, str) and expected_asset_id
            else {}
        )
        return {**identity, "identity_hash": _reference_identity_hash(identity)}

    requirement = "stage2_tokenizer_manifest"
    if not isinstance(expected_asset_id, str) or not expected_asset_id:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal reference config must bind model.tokenizer_asset_id",
            retryable=False,
            evidence_refs=evidence_refs,
        )
    if not isinstance(expected_checkpoint_id, str) or not expected_checkpoint_id:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal reference checkpoint identity is required before tokenizer binding",
            retryable=False,
            evidence_refs=evidence_refs,
        )
    if not isinstance(external_tokenizer, Mapping):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal reference requires the hash-bound tokenizer manifest",
            retryable=False,
            evidence_refs=evidence_refs,
        )
    if external_tokenizer.get("schema_version") != "tokenizer-manifest-v1":
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal tokenizer manifest schema is invalid",
            retryable=False,
            evidence_refs=evidence_refs,
        )
    fields = ("asset_id", "revision", "checkpoint_id")
    identity = {key: external_tokenizer.get(key) for key in fields}
    missing = [
        key
        for key, value in identity.items()
        if not isinstance(value, str) or not value.strip()
    ]
    if missing:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal tokenizer manifest identity fields are missing or invalid: "
            + ",".join(missing),
            retryable=False,
            evidence_refs=evidence_refs,
        )
    if identity["checkpoint_id"] != expected_checkpoint_id:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            requirement,
            "formal tokenizer manifest checkpoint_id does not match the selected checkpoint",
            retryable=False,
            evidence_refs=evidence_refs,
        )
    return {**identity, "identity_hash": _reference_identity_hash(identity)}


def _reference_parameter_registry(
    request: TaskExecutionRequest,
    root: Path,
    context: _ProviderContext,
    authoritative: Mapping[str, object] | None = None,
) -> Mapping[str, object] | None:
    """Load the canonical parameter grouping used by every G2.3 endpoint.

    Formal execution must receive this artifact from the frozen Stage 1/S2.2
    handoff.  A local fixture may derive its tiny registry from the fixed
    provider only because the fixture provider itself publishes the explicit
    mapping; the evaluator never guesses groups from parameter names.
    """

    if authoritative is not None:
        value = authoritative
        reference = str(request.environment.evidence_refs.get("stage2_parameter_registry"))
        groups = value.get("parameter_groups") if isinstance(value, Mapping) else None
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != "stage2-parameter-registry-artifact-v1"
            or value.get("registry_hash") != context.provider.registry_hash
            or not isinstance(groups, Mapping)
            or not isinstance(value.get("artifact_hash"), str)
            or value.get("artifact_hash") != canonical_json_hash({k: v for k, v in value.items() if k != "artifact_hash"})
        ):
            return None
        return value
    reference = request.environment.evidence_refs.get("stage2_parameter_registry")
    if reference is not None:
        try:
            value = load_canonical_json(_workspace_path(
                root,
                reference,
                field="stage2_parameter_registry",
            ))
        except (OSError, TypeError, ValueError):
            # The task runner's formal root is not carried by the context; the
            # caller will report the missing artifact as an explicit blocker.
            return None
        if not isinstance(value, Mapping):
            return None
        if value.get("schema_version") != "stage2-parameter-registry-artifact-v1":
            return None
        if value.get("registry_hash") != context.provider.registry_hash:
            return None
        groups = value.get("parameter_groups")
        if not isinstance(groups, Mapping):
            return None
        return value
    if request.config.run_intent != "local_fixture":
        return None
    groups: dict[str, object] = {}
    for name in context.provider.parameter_names:
        parts = str(name).split(".")
        groups[str(name)] = {
            "layer": ".".join(parts[:2]) if len(parts) > 1 else parts[0],
            "module": parts[0],
        }
    body: dict[str, object] = {
        "schema_version": "stage2-parameter-registry-artifact-v1",
        "registry_hash": context.provider.registry_hash,
        "parameter_groups": groups,
        "source": "local_fixed_state_provider_registry",
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _reference_formula_contract(
    request: TaskExecutionRequest,
    authoritative: Mapping[str, object] | None,
) -> tuple[Mapping[str, object], str]:
    """Validate the S2.1 formula/floor contract without consuming numbers."""

    if authoritative is None:
        if request.config.run_intent != "local_fixture":
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "stage2_preregistration",
                "formal reference requires the hash-bound S2.1 formula contract",
                retryable=False,
            )
        body: dict[str, object] = {
            "schema_version": PREREGISTRATION_SCHEMA_VERSION,
            "scope": "local_fixture",
            "equivalence_and_precision": {
                "scientific_margin_formula": "max(0.10*Delta_c_e(B),0.01*S_c_e)",
                "signal_definition": "S_model=max(abs(sum(a_k)),tau_model); S_q=max(sum_g(abs(sum_g(a_k))),tau_q)",
                "noise_definition": "Delta_model=abs(sum_k(d_k(B))); Delta_q=sum_g(abs(sum_g(d_k(B))))",
                "sizing_vectors": {"a": "mu_sizing^2", "d": "sigma_squared_over_B"},
                "group_registry": "canonical_non_overlapping_layer_and_module_registry",
                "absolute_floors": dict(ABSOLUTE_FLOORS),
            },
        }
        return dict(body, preregistration_hash=canonical_json_hash(body)), canonical_json_hash(body)

    value = authoritative
    if value.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION or value.get("scope") != "formal":
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "S2.1 formula contract must be formal", retryable=False)
    supplied = value.get("preregistration_hash")
    if not isinstance(supplied, str) or supplied != canonical_json_hash({key: item for key, item in value.items() if key != "preregistration_hash"}):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "S2.1 preregistration hash is not content-bound", retryable=False)
    precision = value.get("equivalence_and_precision")
    if not isinstance(precision, Mapping):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "S2.1 equivalence_and_precision contract is missing", retryable=False)
    expected_formulas = {
        "scientific_margin_formula": "max(0.10*Delta_c_e(B),0.01*S_c_e)",
        "signal_definition": "S_model=max(abs(sum(a_k)),tau_model); S_q=max(sum_g(abs(sum_g(a_k))),tau_q)",
        "noise_definition": "Delta_model=abs(sum_k(d_k(B))); Delta_q=sum_g(abs(sum_g(d_k(B))))",
        "sizing_vectors": {"a": "mu_sizing^2", "d": "sigma_squared_over_B"},
        "group_registry": "canonical_non_overlapping_layer_and_module_registry",
    }
    if any(precision.get(key) != expected for key, expected in expected_formulas.items()):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "S2.1 formula contract drifted", retryable=False)
    floors = precision.get("absolute_floors")
    if not isinstance(floors, Mapping) or set(floors) != set(ABSOLUTE_FLOORS):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "S2.1 absolute floors are missing or changed", retryable=False)
    for name, expected in ABSOLUTE_FLOORS.items():
        actual = floors.get(name)
        if isinstance(actual, bool) or not isinstance(actual, (int, float)) or not math.isfinite(float(actual)) or float(actual) != float(expected) or float(actual) <= 0:
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", f"S2.1 absolute floor invalid: {name}", retryable=False)
    return value, supplied


def _weighted_sequence_variance_from_shards(
    store: _ReferenceShardStore,
    refs: Sequence[Mapping[str, object]],
    assumptions: Mapping[str, object],
    block_size: int,
) -> Mapping[str, np.ndarray]:
    """Compute sequence variance in two streaming passes over immutable shards."""

    if block_size <= 0 or len(refs) < 2:
        raise ValueError("STAGE2_SIZING_VARIANCE_REQUIRES_TWO_BLOCKS")
    moments = _moments_from_shards(store, refs, assumptions)
    mean = moments.mean()
    variance = {name: np.zeros_like(value, dtype=np.float64) for name, value in mean.items()}
    for ref in refs:
        vector, weight, _ = store.load(ref)
        for name in variance:
            variance[name] += float(weight) * np.square(vector[name] - mean[name])
    return {name: value * float(block_size) / float(moments.n1) for name, value in variance.items()}


def _sizing_groups(
    parameter_registry: Mapping[str, object],
    names: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    groups = parameter_registry.get("parameter_groups")
    if not isinstance(groups, Mapping) or set(str(name) for name in groups) != set(names):
        raise ValueError("STAGE2_SIZING_PARAMETER_REGISTRY_SET_MISMATCH")
    layer: dict[str, list[str]] = {}
    module: dict[str, list[str]] = {}
    for name in names:
        entry = groups.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"STAGE2_SIZING_PARAMETER_GROUP_MISSING:{name}")
        layer_name, module_name = entry.get("layer"), entry.get("module")
        if not isinstance(layer_name, str) or not layer_name or not isinstance(module_name, str) or not module_name:
            raise ValueError(f"STAGE2_SIZING_PARAMETER_GROUP_INVALID:{name}")
        layer.setdefault(layer_name, []).append(name)
        module.setdefault(module_name, []).append(name)
    return {key: tuple(value) for key, value in sorted(layer.items())}, {key: tuple(value) for key, value in sorted(module.items())}


def _sizing_delta_sci(signal: float, noise: float) -> float:
    """Apply the frozen sizing margin formula to one endpoint and one B.

    Keeping this as a small pure operation makes the numeric contract
    independently testable.  The caller still derives ``signal`` and
    ``noise`` from raw sizing shards; this helper never accepts a caller
    supplied margin.
    """

    signal_f, noise_f = float(signal), float(noise)
    if not math.isfinite(signal_f) or not math.isfinite(noise_f) or signal_f <= 0.0 or noise_f <= 0.0:
        raise ValueError("STAGE2_SIZING_MARGIN_SCALES_MUST_BE_FINITE_POSITIVE")
    delta = max(0.10 * noise_f, 0.01 * signal_f)
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("STAGE2_SIZING_MARGIN_INVALID")
    return delta


def _derive_sizing_delta_sci(
    *,
    root: Path,
    sizing_root: Path,
    plan: ReferenceSizingPlan,
    parameter_registry: Mapping[str, object],
    formula_contract: Mapping[str, object],
    formula_contract_hash: str,
    provider: FixedStateGradientProvider,
    sizing_result_hash: str,
) -> Mapping[str, object]:
    """Derive and freeze all candidate margins from sizing shards before A/B."""

    commits = sorted((sizing_root / "commits").glob("*.json"))
    if not commits:
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing commits missing", retryable=False)
    assumptions = {
        "statistical_unit": provider.statistical_unit,
        "weight_unit": provider.weight_unit,
        "sampling_design": provider.sampling_design,
        "weights_exogenous": provider.weights_exogenous,
        "common_mean_assumption": provider.common_mean_assumption,
    }
    shard_store = _ReferenceShardStore(sizing_root)
    state_by_count: dict[int, tuple[Mapping[str, object], Mapping[str, object]]] = {}
    for commit_path in commits:
        commit = load_canonical_json(commit_path)
        if not isinstance(commit, Mapping):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing commit invalid", retryable=False)
        object_ref = commit.get("object_ref")
        if not isinstance(object_ref, str):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing object ref missing", retryable=False)
        state, bundle = load_tensor_bundle(sizing_root / object_ref)
        if not isinstance(state, Mapping) or state.get("schema_version") != "stage2-reference-progress-state-v1" or bundle.manifest_sha256 != commit.get("object_manifest_hash"):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing state invalid", retryable=False)
        count = int(state.get("processed_block_pairs", 0)) * plan.block_size
        refs = state.get("shard_refs_a")
        if count <= 0 or not isinstance(refs, list) or len(refs) != int(state.get("processed_block_pairs", 0)) or state.get("shard_refs_b") not in ([], None):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing shard prefix invalid", retryable=False)
        for ref in refs:
            if not isinstance(ref, Mapping):
                raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing shard ref invalid", retryable=False)
            shard_store.load(ref)
        state_by_count[count] = (state, {"commit": commit, "refs": refs})
    if any(count not in state_by_count for count in plan.candidate_sample_counts):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_sizing", "sizing shards do not cover every candidate B", retryable=False)
    latest_refs = state_by_count[plan.candidate_sample_counts[-1]][1]["refs"]
    assert isinstance(latest_refs, list) and latest_refs
    first_vector, _, _ = shard_store.load(latest_refs[0])
    names = tuple(sorted(first_vector))
    layer_groups, module_groups = _sizing_groups(parameter_registry, names)
    precision = formula_contract.get("equivalence_and_precision")
    if not isinstance(precision, Mapping) or not isinstance(precision.get("absolute_floors"), Mapping):
        raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "absolute floors missing", retryable=False)
    floors = precision["absolute_floors"]
    endpoints = ("model_total", "layer", "module")
    delta_by_endpoint: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    signal_by_endpoint: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    noise_by_endpoint: dict[str, dict[str, float]] = {endpoint: {} for endpoint in endpoints}
    nodes: list[dict[str, object]] = []
    for count in plan.candidate_sample_counts:
        state, raw = state_by_count[count]
        refs = raw["refs"]
        assert isinstance(refs, list)
        moments = _moments_from_shards(shard_store, refs, assumptions)
        mean = moments.mean()
        sigma2 = _weighted_sequence_variance_from_shards(shard_store, refs, assumptions, plan.block_size)
        a = {name: np.square(mean[name]) for name in names}
        model_s = max(abs(float(sum(np.sum(value) for value in a.values()))), float(floors["tau_model"]))
        model_d = abs(float(sum(np.sum(value) for value in sigma2.values()))) / float(count)
        layer_a = [float(sum(np.sum(a[name]) for name in group)) for group in layer_groups.values()]
        layer_d = [float(sum(np.sum(sigma2[name]) for name in group)) / float(count) for group in layer_groups.values()]
        module_a = [float(sum(np.sum(a[name]) for name in group)) for group in module_groups.values()]
        module_d = [float(sum(np.sum(sigma2[name]) for name in group)) / float(count) for group in module_groups.values()]
        endpoint_values = {
            "model_total": (max(abs(model_s), float(floors["tau_model"])), abs(model_d)),
            "layer": (max(float(sum(abs(value) for value in layer_a)), float(floors["tau_layer"])), float(sum(abs(value) for value in layer_d))),
            "module": (max(float(sum(abs(value) for value in module_a)), float(floors["tau_module"])), float(sum(abs(value) for value in module_d))),
        }
        for endpoint, (signal, noise) in endpoint_values.items():
            try:
                delta = _sizing_delta_sci(signal, noise)
            except (TypeError, ValueError, OverflowError) as error:
                raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_delta_sci", f"invalid sizing margin: {endpoint}/{count}", retryable=False)
            signal_by_endpoint[endpoint][str(count)] = signal
            noise_by_endpoint[endpoint][str(count)] = noise
            delta_by_endpoint[endpoint][str(count)] = delta
        commit = raw["commit"]
        assert isinstance(commit, Mapping)
        nodes.append({
            "sample_count": count,
            "state_digest": commit.get("state_digest"),
            "shard_refs_hash": canonical_json_hash([
                {"shard_hash": ref.get("shard_hash"), "manifest_hash": ref.get("manifest_hash"), "weight": ref.get("weight")}
                for ref in refs if isinstance(ref, Mapping)
            ]),
            "mean_hash": _vector_digest(mean),
            "sequence_variance_hash": _vector_digest(sigma2),
        })
    body: dict[str, object] = {
        "schema_version": "stage2-reference-delta-sci-v2",
        "source_kind": "reference_sizing_raw_shards",
        "formula_contract_hash": formula_contract_hash,
        "formula_version": "stage2-reference-sizing-margin-v1",
        "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
        "absolute_floors": dict(floors),
        "reference_id": plan.reference_id,
        "sizing_result_hash": sizing_result_hash,
        "sizing_plan_hash": plan.artifact_hash,
        "candidate_sample_counts": list(plan.candidate_sample_counts),
        "delta_sci_by_endpoint": delta_by_endpoint,
        "signal_scale_by_endpoint": signal_by_endpoint,
        "noise_scale_by_endpoint": noise_by_endpoint,
        "sizing_nodes": nodes,
        "registry_hash": provider.registry_hash,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    derived_dir = sizing_root / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    derived_path = derived_dir / f"{body['artifact_hash']}.json"
    if derived_path.exists():
        if load_canonical_json(derived_path) != body:
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_reference_delta_sci", "sizing-derived artifact drift", retryable=False)
    else:
        write_canonical_json(derived_path, body)
    source_ref = derived_path.relative_to(root).as_posix()
    published = dict(body)
    published["source_ref"] = source_ref
    published["source_hash"] = body["artifact_hash"]
    published["source_artifact_hash"] = body["artifact_hash"]
    return published


def _reference_six_cell_manifest(
    inputs: _PredecessorContext,
    context: _ProviderContext,
    authoritative: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Project the validated S2.3 six-cell asset matrix into S2.4 output."""

    try:
        asset_resolution = authoritative if authoritative is not None else inputs.payload("asset_resolution")
    except (KeyError, ValueError, TypeError):
        return {
            "schema_version": "stage2-s204-six-cell-manifest-v1",
            "status": "MISSING",
            "checkpoints": [],
        }
    if isinstance(asset_resolution, Mapping) and asset_resolution.get("schema_version") == "stage2-s204-six-cell-manifest-v1":
        return dict(asset_resolution)
    manifest = asset_resolution.get("stage2_asset_manifest") if isinstance(asset_resolution, Mapping) else None
    if not isinstance(manifest, Mapping):
        return {
            "schema_version": "stage2-s204-six-cell-manifest-v1",
            "status": "MISSING",
            "checkpoints": [],
        }
    raw = manifest.get("checkpoints")
    rows: list[Mapping[str, object]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, Mapping):
                continue
            rows.append({
                "cell_id": f"{item.get('model_id')}:{item.get('training_stage')}",
                "model_id": item.get("model_id"),
                "training_stage": item.get("training_stage"),
                "checkpoint_id": item.get("checkpoint_id"),
                "checkpoint_hash": item.get("manifest_sha256"),
                "checkpoint_revision": item.get("revision"),
                "registry_hash": item.get("parameter_registry_hash"),
                "config_hash": item.get("config_sha256"),
            })
    # Keep the S2.3 order frozen (model-major, then initialization/early/
    # mid_late); the evaluator derives this order from the manifest and never
    # accepts a caller supplied permutation.
    model_order = {"pythia-14m": 0, "pythia-31m-deduped": 1}
    stage_order = {"initialization": 0, "early": 1, "mid_late": 2}
    rows.sort(
        key=lambda item: (
            model_order.get(str(item.get("model_id")), 99),
            stage_order.get(str(item.get("training_stage")), 99),
        )
    )
    data_range = manifest.get("data_range")
    body: dict[str, object] = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY" if len(rows) == 6 else "MISSING",
        "scope": manifest.get("scope"),
        "asset_resolution_hash": manifest.get("asset_resolution_hash"),
        "asset_producer_commit": manifest.get("producer_commit"),
        "asset_execution_commit": manifest.get("execution_commit"),
        "checkpoints": rows,
        "data": dict(data_range) if isinstance(data_range, Mapping) else None,
        "registry_hash": context.provider.registry_hash,
    }
    if isinstance(data_range, Mapping):
        body["data_range_hash"] = data_range.get("data_range_hash")
    else:
        body["data_range_hash"] = None
    body["manifest_hash"] = canonical_json_hash(body)
    return body


def _trusted_stage2_provenance(*, require_clean: bool) -> Mapping[str, object]:
    """Bind producer output to an actual repository object and source bytes."""

    repository_root = Path(__file__).resolve().parents[3]
    try:
        def git(*arguments: str) -> str:
            return subprocess.run(
                ["git", "-C", str(repository_root), *arguments],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        head = git("rev-parse", "HEAD")
        tree = git("rev-parse", "HEAD^{tree}")
        git("cat-file", "-e", f"{head}^{{commit}}")
        status = git("status", "--porcelain", "--untracked-files=no")
        tracked_clean = status == ""
        source_paths = (
            "src/param_importance_nlp/experiments/stage2_formal.py",
            "src/param_importance_nlp/experiments/stage23_task_runners.py",
            "src/param_importance_nlp/experiments/stage2_g23_evaluator.py",
            "ops/stage2/evaluate_s204_g23.py",
        )
        source_bytes: list[dict[str, object]] = []
        for relative in source_paths:
            path = repository_root / Path(relative)
            if not path.is_file():
                raise RuntimeError(f"STAGE2_SOURCE_MISSING:{relative}")
            source_bytes.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "git_blob": git("hash-object", relative),
                }
            )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("STAGE2_TRUSTED_REPOSITORY_UNAVAILABLE") from error
    if require_clean and not tracked_clean:
        raise RuntimeError("STAGE2_PRODUCER_REPOSITORY_DIRTY")
    return {
        "schema_version": "stage2-reference-producer-provenance-v2",
        "repository_root_name": repository_root.name,
        "head_commit": head,
        "head_tree": tree,
        "tracked_clean": tracked_clean,
        "source_bytes": source_bytes,
        "provenance_hash": canonical_json_hash(
            {
                "head_commit": head,
                "head_tree": tree,
                "tracked_clean": tracked_clean,
                "source_bytes": source_bytes,
            }
        ),
    }


def _actual_sampling_state(
    sampling: SamplingPlan,
    stream: str,
    count: int,
) -> Mapping[str, object]:
    """Capture real Python generator states and verify the frozen draws."""

    if stream not in STREAM_NAMES or count < 0:
        raise ValueError("STAGE2_SAMPLING_STATE_ARGUMENT_INVALID")
    return generator_boundary(sampling, stream, count)


def _available_ram_bytes() -> int | None:
    try:
        import ctypes

        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("sullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullAvailPhys)
    except (AttributeError, OSError, TypeError):
        pass
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        if "MemAvailable" in values:
            return values["MemAvailable"]
    except (OSError, ValueError, IndexError):
        pass
    return None


def _reference_capacity_preflight(
    provider: FixedStateGradientProvider,
    plan: ReferenceSizingPlan,
    output_root: Path,
    *,
    model_manifest: Mapping[str, object] | None = None,
    stable_identity: bool = False,
) -> Mapping[str, object]:
    """Fail closed on actual model-size, RAM, and single-copy shard capacity."""

    parameter_count: int | None = None
    parameters = getattr(provider, "_parameters", None)
    if isinstance(parameters, Mapping):
        values = list(parameters.values())
        if values and all(hasattr(value, "numel") for value in values):
            parameter_count = sum(int(value.numel()) for value in values)
    if parameter_count is None and isinstance(model_manifest, Mapping):
        candidate = model_manifest.get("parameter_count")
        if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
            parameter_count = candidate
    if parameter_count is None:
        table = getattr(provider, "_table", None)
        if isinstance(table, Mapping) and table:
            first = next(iter(table.values()))
            if isinstance(first, Mapping):
                parameter_count = sum(int(np.asarray(value).size) for value in first.values())
    if parameter_count is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_capacity_preflight",
            "formal capacity preflight requires actual parameter_count from model/provider",
            retryable=False,
        )
    max_blocks = int(plan.candidate_sample_counts[-1] // plan.block_size)
    # Sizing and one-shot are retained until the task result is published.  A
    # block is one FP64 vector plus a small manifest; moments are two FP64
    # vectors per stream per commit.  No B/parameter/sample reduction is used.
    shard_bytes = max_blocks * 2 * parameter_count * 8
    snapshot_moment_bytes = max_blocks * 4 * parameter_count * 8
    estimated_disk = int((shard_bytes + snapshot_moment_bytes) * 1.20 + 64 * 1024**2)
    free_disk = shutil.disk_usage(output_root).free
    available_ram = _available_ram_bytes()
    peak_ram = int(3 * parameter_count * 8 + 64 * 1024**2)
    capacity = {
        "schema_version": "stage2-reference-capacity-preflight-v1",
        "parameter_count": parameter_count,
        "candidate_max_sample_count_per_stream": int(plan.candidate_sample_counts[-1]),
        "block_size": plan.block_size,
        "max_block_count_per_stream": max_blocks,
        "single_copy_shard_bytes": shard_bytes,
        "snapshot_moment_bytes": snapshot_moment_bytes,
        "estimated_disk_bytes": estimated_disk,
        "free_disk_bytes": None if stable_identity else int(free_disk),
        "peak_ram_bytes": peak_ram,
        "available_ram_bytes": None if stable_identity else available_ram,
        "disk_ok": free_disk >= estimated_disk,
        "ram_ok": available_ram is not None and available_ram >= peak_ram,
        "fail_closed_if_unknown": True,
    }
    capacity["artifact_hash"] = canonical_json_hash(capacity)
    if not capacity["disk_ok"] or not capacity["ram_ok"]:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_capacity_preflight",
            f"reference capacity insufficient: disk_ok={capacity['disk_ok']} ram_ok={capacity['ram_ok']}",
            retryable=True,
        )
    return capacity


def _load_reference_external_lineage(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[Mapping[str, object], Mapping[str, Mapping[str, object]]]:
    """Load all formal S2.3/materializer refs as authoritative TaskArtifacts.

    The keys are intentionally explicit.  This prevents a producer from
    recursively searching arbitrary predecessor JSON and accidentally binding
    to a similarly named plan, delta, or manifest.
    """

    if request.config.run_intent != "formal":
        return {}, {}
    key_to_kind = {
        "s23_asset_resolution": ("stage2_s23_asset_resolution", "asset_resolution"),
        "s23_six_cell_manifest": ("stage2_s23_six_cell_manifest", "six_cell_manifest"),
        "resolved_config": ("stage2_resolved_config", "resolved_config"),
        "checkpoint_manifest": ("stage2_checkpoint_manifest", "checkpoint_manifest"),
        "model_manifest": ("stage2_model_manifest", "model_manifest"),
        "data_manifest": ("stage2_data_manifest", "data_manifest"),
        "tokenizer_manifest": ("stage2_tokenizer_manifest", "tokenizer_manifest"),
        "parameter_registry": ("stage2_parameter_registry", "parameter_registry"),
        # S2.1 freezes the formula and native-unit floors.  Numeric
        # ``delta_sci_by_B`` is deliberately *not* an external input: S2.4
        # derives it from the immutable sizing shards before creating A/B.
        "preregistration": ("stage2_preregistration", "preregistration"),
        "sizing_plan": ("stage2_reference_sizing_plan", "reference_sizing_plan"),
    }
    lineage: dict[str, object] = {}
    payloads: dict[str, Mapping[str, object]] = {}
    for name, (environment_key, expected_kind) in key_to_kind.items():
        reference = request.environment.evidence_refs.get(environment_key)
        if not isinstance(reference, str) or not reference:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                environment_key,
                f"formal S2.4 requires explicit TaskArtifact ref: {environment_key}",
                retryable=False,
            )
        _workspace_path(root, reference, field=environment_key)
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, ValueError, TypeError) as error:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                environment_key,
                f"external TaskArtifact invalid: {type(error).__name__}",
                retryable=False,
                evidence_refs=(reference,),
            ) from error
        _workspace_path(root, loaded.identity.object_ref, field=f"{environment_key}.object_ref")
        if loaded.identity.artifact_kind != expected_kind:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                environment_key,
                f"external artifact kind mismatch: expected {expected_kind}",
                retryable=False,
                evidence_refs=(reference,),
            )
        if not isinstance(loaded.payload, Mapping):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, environment_key, "external payload object required", retryable=False, evidence_refs=(reference,))
        try:
            source_manifest = validate_external_manifest(
                loaded,
                root,
                expected_kind=expected_kind,
            )
        except (OSError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                environment_key,
                f"external source manifest invalid: {type(error).__name__}",
                retryable=False,
                evidence_refs=(reference,),
            ) from error
        payload = dict(loaded.payload)
        payloads[name] = payload
        lineage[name] = {
            "commit_ref": reference,
            "artifact_kind": loaded.identity.artifact_kind,
            "artifact_hash": loaded.identity.artifact_hash,
            "config_hash": loaded.identity.config_hash,
            "task_id": loaded.identity.task_id,
            "formal_eligible": loaded.identity.formal_eligible,
            "payload_hash": canonical_json_hash(payload),
            "source_refs": list(loaded.source_refs),
            "source_manifest": source_manifest,
        }
    resolved = payloads.get("resolved_config")
    resolved_lineage = lineage.get("resolved_config")
    if not isinstance(resolved, Mapping) or not isinstance(resolved_lineage, Mapping):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_resolved_config",
            "resolved formal config lineage missing",
            retryable=False,
        )
    if (
        resolved.get("task_id") != request.task.task_id
        or resolved.get("config_hash") != request.config.config_hash
        or resolved_lineage.get("config_hash") != request.config.config_hash
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_resolved_config",
            "resolved formal config is not bound to the current TaskRuntime config",
            retryable=False,
        )
    return lineage, payloads


def _reference_numeric_diagnostics(
    *,
    final_root: Path,
    result: OneShotReferenceResult,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Recompute U from raw committed blocks with long-double accumulation.

    The returned first object is JSON metadata; the second is stored in the
    tensor bundle because it contains the signed vectors.
    """

    commits = sorted((final_root / "commits").glob("*.json"))
    if not commits:
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_BLOCK_COMMITS_MISSING")
    latest = load_canonical_json(commits[-1])
    if not isinstance(latest, Mapping):
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_LATEST_COMMIT_INVALID")
    state, bundle = load_tensor_bundle(final_root / str(latest["object_ref"]))
    if not isinstance(state, Mapping):
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_BLOCK_STATE_INVALID")
    raw_a, raw_b = state.get("shard_refs_a"), state.get("shard_refs_b")
    if not isinstance(raw_a, list) or not isinstance(raw_b, list) or not raw_a or len(raw_a) != len(raw_b):
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_SHARD_REFS_MISSING")
    shard_store = _ReferenceShardStore(final_root)
    raw_digest_rows: list[Mapping[str, object]] = []
    names: tuple[str, ...] | None = None
    first_sums: dict[str, np.ndarray] = {}
    second_sums: dict[str, np.ndarray] = {}
    n1 = np.longdouble(0)
    n2 = np.longdouble(0)
    for ref in list(raw_a) + list(raw_b):
        if not isinstance(ref, Mapping):
            raise RuntimeError("STAGE2_REFERENCE_NUMERIC_SHARD_REF_INVALID")
        vector, weight, _ = shard_store.load(ref)
        current_names = tuple(vector)
        if names is None:
            names = current_names
            first_sums = {name: np.zeros_like(vector[name], dtype=np.longdouble) for name in names}
            second_sums = {name: np.zeros_like(vector[name], dtype=np.longdouble) for name in names}
        if current_names != names:
            raise RuntimeError("STAGE2_REFERENCE_NUMERIC_PARAMETER_SET_DRIFT")
        long_weight = np.longdouble(weight)
        if not np.isfinite(long_weight) or long_weight <= 0:
            raise RuntimeError("STAGE2_REFERENCE_NUMERIC_BLOCK_WEIGHTS_INVALID")
        n1 += long_weight
        n2 += long_weight * long_weight
        raw_digest_rows.append({"vector_hash": _vector_digest(vector), "weight": float(weight)})
        for name in names:
            value = np.asarray(vector[name], dtype=np.longdouble)
            first_sums[name] += long_weight * value
            second_sums[name] += long_weight * long_weight * value * value
    if names is None:
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_SHARDS_EMPTY")
    denominator = n1 * n1 - n2
    if denominator <= 0:
        raise RuntimeError("STAGE2_REFERENCE_NUMERIC_U_DENOMINATOR_INVALID")
    high: dict[str, np.ndarray] = {}
    accumulated: dict[str, np.ndarray] = {}
    for name in names:
        high[name] = np.asarray((first_sums[name] * first_sums[name] - second_sums[name]) / denominator, dtype=np.float64)
        accumulated[name] = np.asarray(result.bias_reference[name], dtype=np.float64)
    raw_block_digest = canonical_json_hash(raw_digest_rows)
    numeric_vectors = {
        "high_precision": high,
        "accumulated": accumulated,
    }
    metadata: dict[str, object] = {
        "schema_version": "stage2-reference-numerical-diagnostics-v1",
        "recompute_method": "longdouble_pairwise_u_from_content_addressed_shards",
        "raw_block_digest": raw_block_digest,
        "raw_block_count_a": len(raw_a),
        "raw_block_count_b": len(raw_b),
        "high_precision_hash": _vector_digest(high),
        "accumulated_hash": _vector_digest(accumulated),
        "max_abs_error": max(float(np.max(np.abs(high[name] - accumulated[name]))) for name in names),
        "resume_latest_commit_ref": commits[-1].relative_to(final_root).as_posix(),
        "resume_latest_commit_hash": str(latest.get("artifact_hash")),
        "resume_latest_manifest_hash": str(bundle.manifest_sha256),
    }
    metadata["artifact_hash"] = canonical_json_hash(metadata)
    return metadata, numeric_vectors


def _run_stage2_reference(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    context = _provider_context(request, root)
    try:
        upstream_sampling = SamplingPlan.from_mapping(inputs.payload("sampling_plan"))
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_sampling_plan",
            f"前序 sampling_plan 无法严格加载：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    provider_sampling = _sampling_plan(request, context)
    if not _sampling_plan_provider_compatible(upstream_sampling, provider_sampling):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_sampling_plan",
            "前序 sampling_plan 与当前 fixed-state provider/seed 不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    external_lineage: Mapping[str, object] = {}
    external_payloads: Mapping[str, Mapping[str, object]] = {}
    if request.config.run_intent == "formal":
        external_lineage, external_payloads = _load_reference_external_lineage(request, root)
    plan, plan_refs = _stage2_reference_plan(
        request,
        root,
        context,
        external_payloads.get("sizing_plan"),
    )
    capacity_preflight = _reference_capacity_preflight(
        context.provider,
        plan,
        store.root,
        model_manifest=external_payloads.get("model_manifest"),
        stable_identity=request.config.run_intent == "local_fixture",
    )
    sampling = _project_sampling_plan_to_provider(upstream_sampling, provider_sampling)
    provider_state_before = context.provider.state_digest()
    six_cell_manifest = _reference_six_cell_manifest(
        inputs,
        context,
        external_payloads.get("s23_six_cell_manifest"),
    )
    if request.config.run_intent == "formal" and (
        six_cell_manifest.get("status") != "READY"
        or six_cell_manifest.get("scope") != "formal"
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_s204_six_cell_manifest",
            "formal reference requires the validated S2.3 six-cell manifest",
            retryable=False,
            evidence_refs=inputs.references,
        )
    formula_contract, formula_contract_hash = _reference_formula_contract(
        request,
        external_payloads.get("preregistration"),
    )
    if request.config.run_intent == "formal":
        lineage_formula = external_lineage.get("preregistration")
        if not isinstance(lineage_formula, Mapping) or not isinstance(lineage_formula.get("artifact_hash"), str):
            raise _blocked(BlockerCode.CONTRACT_UNFROZEN, "stage2_preregistration", "formula contract TaskArtifact lineage missing", retryable=False)
        formula_contract_hash = str(lineage_formula["artifact_hash"])
    parameter_registry = _reference_parameter_registry(
        request,
        root,
        context,
        external_payloads.get("parameter_registry"),
    )
    if request.config.run_intent == "formal" and parameter_registry is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_parameter_registry",
            "formal reference requires the hash-bound parameter registry artifact",
            retryable=False,
            evidence_refs=tuple(
                [request.environment.evidence_refs["stage2_parameter_registry"]]
                if "stage2_parameter_registry" in request.environment.evidence_refs
                else inputs.references
            ),
        )
    maximum = plan.candidate_sample_counts[-1]
    sizing_draws = sampling.draws("reference_sizing", maximum)
    sizing_rng_state = _actual_sampling_state(sampling, "reference_sizing", maximum)
    sizing_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_sizing", index * plan.block_size)
        for index in range(maximum // plan.block_size + 1)
    )
    result = StreamingReferenceSizer(context.provider).run(
        plan,
        # Keep the old positional API available for direct callers, while the
        # task runner always uses the independent sizing stream.
        draws_a=(),
        draws_b=(),
        draws_sizing=sizing_draws,
        artifact_root=store.root / "resume" / "reference-sizing",
        rng_boundaries=sizing_rng_boundaries,
        require_rng_boundaries=request.config.run_intent == "formal",
    )
    if not result.converged or result.selected_sample_count_per_stream is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_reference_sizing",
            "reference sizing 未达到预注册收敛条件，禁止创建 one-shot A/B",
            retryable=False,
            evidence_refs=inputs.references,
        )
    final_count = result.selected_sample_count_per_stream
    validate_sizing_plan_contract(
        plan.to_dict(),
        selected_sample_count=final_count,
        field="formal_reference_sizing_plan",
    )
    sizing_draw_hash = _draw_digest(sizing_draws)
    sizing_identity_hash = canonical_json_hash(
        {
            "plan_hash": plan.artifact_hash,
            "provider_state_digest": result.provider_state_digest,
            "registry_hash": result.registry_hash,
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_stream": "reference_sizing",
        }
    )
    # Freeze the sizing-derived numeric margin before *any* final A/B draw is
    # materialized.  This source is built only from the sizing shard prefix;
    # it is never included in the final A/B estimand.
    delta_sci = _derive_sizing_delta_sci(
        root=root,
        sizing_root=store.root / "resume" / "reference-sizing",
        plan=plan,
        parameter_registry=parameter_registry,
        formula_contract=formula_contract,
        formula_contract_hash=(
            str(external_lineage.get("preregistration", {}).get("artifact_hash"))
            if isinstance(external_lineage.get("preregistration"), Mapping)
            else formula_contract_hash
        ),
        provider=context.provider,
        sizing_result_hash=result.scientific_artifact_hash,
    )
    one_shot_plan = OneShotReferencePlan(
        reference_id=plan.reference_id,
        sizing_result_hash=result.scientific_artifact_hash,
        sample_count_per_stream=final_count,
        block_size=plan.block_size,
    )
    # The final draw manifests are created only after the sizing-derived
    # margin has been atomically published.
    final_a = sampling.draws("reference_A", final_count)
    final_b = sampling.draws("reference_B", final_count)
    final_a_rng_state = _actual_sampling_state(sampling, "reference_A", final_count)
    final_b_rng_state = _actual_sampling_state(sampling, "reference_B", final_count)
    final_a_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_A", index * plan.block_size)
        for index in range(final_count // plan.block_size + 1)
    )
    final_b_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_B", index * plan.block_size)
        for index in range(final_count // plan.block_size + 1)
    )
    one_shot = OneShotReferenceRunner(context.provider).run(
        one_shot_plan,
        draws_a=final_a,
        draws_b=final_b,
        sizing_draws=sizing_draws,
        artifact_root=store.root / "resume" / "reference-final",
        rng_boundaries_a=final_a_rng_boundaries,
        rng_boundaries_b=final_b_rng_boundaries,
        require_rng_boundaries=request.config.run_intent == "formal",
    )
    if one_shot.status != "COMPLETE":
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_reference_one_shot",
            f"one-shot A/B 未完成：{one_shot.status}",
            retryable=False,
            evidence_refs=inputs.references,
        )
    provider_state_after = context.provider.state_digest()
    replay = OneShotReferenceRunner(context.provider).run(
        one_shot_plan,
        draws_a=final_a,
        draws_b=final_b,
        sizing_draws=sizing_draws,
        artifact_root=store.root / "resume" / "reference-final",
        rng_boundaries_a=final_a_rng_boundaries,
        rng_boundaries_b=final_b_rng_boundaries,
        require_rng_boundaries=request.config.run_intent == "formal",
    )
    if replay.status != "COMPLETE" or replay.artifact_hash != one_shot.artifact_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_reference_resume_replay",
            "one-shot resume replay did not reproduce the complete artifact",
            retryable=False,
            evidence_refs=inputs.references,
        )
    numerical_metadata, numerical_vectors = _reference_numeric_diagnostics(
        final_root=store.root / "resume" / "reference-final",
        result=one_shot,
    )
    if parameter_registry is None:
        parameter_registry = {
            "schema_version": "stage2-parameter-registry-artifact-v1",
            "status": "MISSING",
            "registry_hash": context.provider.registry_hash,
            "parameter_groups": None,
        }
    config_identity = {
        "config_hash": request.config.config_hash,
        "task_id": request.task.task_id,
    }
    # Scientific identity belongs to the immutable v1 base config.  The v2
    # execution envelope intentionally exposes only execution sections at the
    # top level, so asking it for ``model``/``data`` would be an accidental
    # KeyError in the producer rather than a published diagnostic.
    model_section = request.config.base_config.section("model")
    data_section = request.config.base_config.section("data")
    checkpoint_section = request.config.base_config.section("identity")
    model_identity = dict(model_section) if isinstance(model_section, Mapping) else {}
    data_identity = dict(data_section) if isinstance(data_section, Mapping) else {}
    identity_section = dict(checkpoint_section) if isinstance(checkpoint_section, Mapping) else {}
    data_identity["data_range_hash"] = six_cell_manifest.get("data_range_hash")
    manifest_rows = six_cell_manifest.get("checkpoints")
    current_checkpoint_id = identity_section.get("input_checkpoint_id")
    current_manifest_row: Mapping[str, object] | None = None
    if isinstance(manifest_rows, list) and current_checkpoint_id is not None:
        matches = [
            item for item in manifest_rows
            if isinstance(item, Mapping) and item.get("checkpoint_id") == current_checkpoint_id
        ]
        if len(matches) == 1:
            current_manifest_row = matches[0]
    if request.config.run_intent == "formal" and current_manifest_row is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_s204_checkpoint_identity",
            "formal reference config checkpoint is not one of the six validated S2.3 cells",
            retryable=False,
            evidence_refs=inputs.references,
        )
    checkpoint_identity: dict[str, object] = {
        "checkpoint_id": identity_section.get("input_checkpoint_id"),
        "checkpoint_revision": model_identity.get("revision"),
        "checkpoint_asset_id": model_identity.get("asset_id"),
        "model_id": model_identity.get("asset_id"),
        "training_stage": None,
        "checkpoint_hash": None,
    }
    if current_manifest_row is not None:
        config_identity["checkpoint_config_hash"] = current_manifest_row.get("config_hash")
        checkpoint_identity.update(
            {
                "cell_id": current_manifest_row.get("cell_id"),
                "model_id": current_manifest_row.get("model_id"),
                "training_stage": current_manifest_row.get("training_stage"),
                "checkpoint_revision": current_manifest_row.get("checkpoint_revision"),
                "checkpoint_hash": current_manifest_row.get("checkpoint_hash"),
                "registry_hash": current_manifest_row.get("registry_hash"),
                "config_hash": current_manifest_row.get("config_hash"),
            }
        )
    checkpoint_identity["identity_hash"] = _reference_identity_hash(checkpoint_identity)
    config_identity["identity_hash"] = _reference_identity_hash(config_identity)
    model_identity["identity_hash"] = _reference_identity_hash(model_identity)
    data_identity["identity_hash"] = _reference_identity_hash(data_identity)
    tokenizer_identity = _reference_tokenizer_identity(
        external_payloads.get("tokenizer_manifest"),
        formal=request.config.run_intent == "formal",
        expected_asset_id=model_identity.get("tokenizer_asset_id"),
        expected_checkpoint_id=checkpoint_identity.get("checkpoint_id"),
        evidence_refs=inputs.references,
    )
    registry_identity = {
        "registry_hash": context.provider.registry_hash,
        "parameter_registry_artifact_hash": parameter_registry.get("artifact_hash"),
    }
    registry_identity["identity_hash"] = _reference_identity_hash(registry_identity)
    sizing_plan = plan.to_dict()
    sizing_plan_hash = plan.artifact_hash
    final_commits = sorted((store.root / "resume" / "reference-final" / "commits").glob("*.json"))
    replay_commit_ref = (
        final_commits[-1]
        .relative_to(store.root / "resume" / "reference-final")
        .as_posix()
        if final_commits
        else None
    )
    replay_commit = load_canonical_json(final_commits[-1]) if final_commits else {}
    replay_diagnostic = {
        "schema_version": "stage2-reference-resume-replay-v1",
        "artifact_ref": replay_commit_ref,
        "artifact_hash": replay_commit.get("artifact_hash"),
        "state_digest": replay_commit.get("state_digest"),
        "object_manifest_hash": replay_commit.get("object_manifest_hash"),
        "source_one_shot_result_hash": one_shot.artifact_hash,
        "replayed_one_shot_result_hash": replay.artifact_hash,
        "sizing_result_identity_hash": canonical_json_hash(
            {
                "sizing_result_hash": one_shot_plan.sizing_result_hash,
                "provider_state_digest": one_shot.provider_state_digest,
                "registry_hash": one_shot.registry_hash,
                "stream_a_draw_hash": one_shot.stream_a_draw_hash,
                "stream_b_draw_hash": one_shot.stream_b_draw_hash,
            }
        ),
    }
    replay_diagnostic["replay_hash"] = canonical_json_hash(replay_diagnostic)
    rng_before_state = {
        "sampling_plan_hash": sampling.digest,
        "streams": {
            "reference_sizing": sizing_rng_state["state_before"],
            "reference_A": final_a_rng_state["state_before"],
            "reference_B": final_b_rng_state["state_before"],
        },
    }
    rng_after_state = {
        "sampling_plan_hash": sampling.digest,
        "streams": {
            "reference_sizing": sizing_rng_state["state_after"],
            "reference_A": final_a_rng_state["state_after"],
            "reference_B": final_b_rng_state["state_after"],
        },
    }
    rng_before = canonical_json_hash(rng_before_state)
    rng_after = canonical_json_hash(rng_after_state)
    producer_provenance = _trusted_stage2_provenance(
        require_clean=request.config.run_intent == "formal"
    )
    producer_commit = str(producer_provenance["head_commit"])
    bundle_path = store.root / "tensor-bundles" / "reference-final"
    bundle = _publish_or_load_bundle(
        bundle_path,
        {
            "bias_reference": one_shot.bias_reference,
            "cross_reference": one_shot.cross_reference,
            "ranking_reference": one_shot.ranking_reference,
            "uncertainty": {
                "bias_variance": one_shot.uncertainty.bias_variance,
                "cross_variance": one_shot.uncertainty.cross_variance,
                "ranking_variance": one_shot.uncertainty.ranking_variance,
            },
            "sequence_variance": one_shot.sequence_variance,
            "numerical_diagnostics": {
                "schema_version": "stage2-reference-numerical-diagnostics-v1",
                "raw_block_digest": numerical_metadata["raw_block_digest"],
                "high_precision": numerical_vectors["high_precision"],
                "accumulated": numerical_vectors["accumulated"],
                "high_precision_hash": numerical_metadata["high_precision_hash"],
                "accumulated_hash": numerical_metadata["accumulated_hash"],
            },
        },
    )
    bundle_ref = bundle_path.relative_to(root).as_posix()
    reference = _stable_reference_artifact(
        reference_id=plan.reference_id,
        result=one_shot,
        block_size=plan.block_size,
        bundle_ref=bundle_ref,
        bundle_hash=bundle.manifest_sha256,
        metadata_extra={
            "one_shot": True,
            "sizing_result_hash": result.scientific_artifact_hash,
            "sizing_stream": "reference_sizing",
            "final_streams": ["reference_A", "reference_B"],
            "sizing_sample_count_per_stream": result.processed_sample_count_per_stream,
            "final_sample_count_per_stream": one_shot.processed_sample_count_per_stream,
            "uncertainty": one_shot.uncertainty.to_dict(),
            "sequence_variance_hash": _vector_digest(one_shot.sequence_variance),
        },
    )
    convergence: dict[str, JSONValue] = {
        "schema_version": "stage2-reference-convergence-report-v1",
        "plan": plan.to_dict(),  # type: ignore[dict-item]
        "status": result.status,
        "converged": result.converged,
        "selected_sample_count_per_stream": result.selected_sample_count_per_stream,
        "processed_sample_count_per_stream": result.processed_sample_count_per_stream,
        "points": [point.to_dict() for point in result.points],  # type: ignore[list-item]
        "sizing_result_hash": result.scientific_artifact_hash,
        "one_shot_plan": one_shot_plan.to_dict(),
        "one_shot_result": one_shot.to_dict(),
        "sizing_stream": "reference_sizing",
        "final_streams": ["reference_A", "reference_B"],
        "final_sample_count_per_stream": one_shot.processed_sample_count_per_stream,
        "reference_uncertainty": one_shot.uncertainty.to_dict(),
        "provider": context.to_payload(),
        "sampling_plan_hash": sampling.digest,
        "recovery_semantics": "authoritative_block_pair_commits",
        "reference_protocol": "authoritative_sizing_and_one_shot_block_pair_commits",
        "diagnostics_schema_version": "stage2-reference-producer-diagnostics-v1",
        "stage2_reference_producer_commit": producer_commit,
        "producer_provenance": producer_provenance,
        "external_lineage": external_lineage,
        "formal_scope": request.config.run_intent,
        "cell_id": checkpoint_identity.get("cell_id"),
        "six_cell_manifest": six_cell_manifest,
        "six_cell_manifest_hash": six_cell_manifest.get("manifest_hash"),
        "sizing_plan": sizing_plan,
        "sizing_plan_artifact_hash": sizing_plan_hash,
        "sizing_draw_hash": sizing_draw_hash,
        "sizing_identity_hash": sizing_identity_hash,
        "formula_contract": formula_contract,
        "formula_contract_hash": formula_contract_hash,
        "candidate_delta_sci": delta_sci,
        "candidate_delta_sci_source": delta_sci.get("source_ref"),
        "candidate_delta_sci_source_hash": delta_sci.get("source_hash"),
        "config_identity": config_identity,
        "model_identity": model_identity,
        "data_identity": data_identity,
        "tokenizer_identity": tokenizer_identity,
        "checkpoint_identity": checkpoint_identity,
        "registry_identity": registry_identity,
        "parameter_registry_artifact": parameter_registry,
        "numerical_diagnostics": numerical_metadata,
        "state_invariance": {
            "model_state_before_hash": provider_state_before,
            "model_state_after_hash": provider_state_after,
            "rng_state_before_hash": rng_before,
            "rng_state_after_hash": rng_after,
            "rng_state_before": rng_before_state,
            "rng_state_after": rng_after_state,
        },
        "draw_artifacts": {
            "reference_sizing": {
                "sampling_plan": sampling.to_dict(),
                "manifest": sampling.draw_manifest("reference_sizing", maximum).to_manifest(),
                "actual_state": sizing_rng_state,
            },
            "reference_A": {
                "sampling_plan": sampling.to_dict(),
                "manifest": sampling.draw_manifest("reference_A", final_count).to_manifest(),
                "actual_state": final_a_rng_state,
            },
            "reference_B": {
                "sampling_plan": sampling.to_dict(),
                "manifest": sampling.draw_manifest("reference_B", final_count).to_manifest(),
                "actual_state": final_b_rng_state,
            },
        },
        "resume_replay": replay_diagnostic,
        "sizing_result_identity_hash": replay_diagnostic["sizing_result_identity_hash"],
        "numerical_floor": 1.0e-12,
        "capacity_preflight": capacity_preflight,
        "formal_eligible": False,
    }
    convergence["reference_producer_diagnostics_hash"] = canonical_json_hash(
        {key: value for key, value in convergence.items() if key != "reference_producer_diagnostics_hash"}
    )
    payload_by_kind: dict[str, Mapping[str, JSONValue]] = {
        "reference_result": reference,  # type: ignore[dict-item]
        "reference_convergence_report": convergence,
        "gate_record": _gate_candidate(request),
    }
    external_refs = tuple(
        str(item.get("commit_ref"))
        for item in external_lineage.values()
        if isinstance(item, Mapping) and isinstance(item.get("commit_ref"), str)
    )
    return payload_by_kind, tuple(dict.fromkeys((*_source_refs(request, plan_refs), *external_refs)))


def _exact_importance_reference(context: _ProviderContext) -> Mapping[str, np.ndarray]:
    full = context.provider.gradient(context.sample_ids)
    means = _as_numpy_vector(full.gradients)
    return {name: np.square(value) for name, value in means.items()}


_FIXTURE_PAIRED_REPETITIONS = 2
_FIXTURE_PAIRED_BATCH_SIZE = 8
_FIXTURE_PAIRED_M_VALUES = (2, 4)


def _paired_mappings(
    sampling: SamplingPlan,
    *,
    stream: str,
    plan: FormalExperimentPlan | None,
) -> tuple[RepetitionMapping, ...]:
    """按冻结计划生成 repetition mapping；fixture 常量只存在于本机分支。

    正式路径的 B/M/R 必须全部来自 :class:`FormalExperimentPlan`。这里不提供正式
    默认值，避免配置缺失时静默退回本机的 ``8/2/4`` 缩小规模。
    """

    if plan is None:
        repetitions = _FIXTURE_PAIRED_REPETITIONS
        batch_size = _FIXTURE_PAIRED_BATCH_SIZE
        m_values = _FIXTURE_PAIRED_M_VALUES
    else:
        if stream != plan.stream:
            raise ValueError("FORMAL_EXPERIMENT_PLAN_STREAM_DRIFT")
        repetitions = plan.repetitions
        batch_size = plan.batch_size
        m_values = plan.microbatch_counts
    draws = sampling.draws(stream, repetitions * batch_size)  # type: ignore[arg-type]
    return tuple(
        RepetitionMapping.create(
            repetition_id=f"rep-{index:04d}",
            draws=draws[index * batch_size : (index + 1) * batch_size],
            m_values=m_values,
        )
        for index in range(repetitions)
    )


def _stable_wave_payload(summary: object) -> dict[str, JSONValue]:
    """Return the canonical paired-wave schema with timing scrubbed.

    Wall-clock values remain in the resumable unit state for diagnostics, but
    are intentionally omitted from the task artifact identity.  The payload
    itself is still a ``stage2-paired-wave-summary-v1`` artifact so downstream
    consumers can validate it through the shared Stage 2 contract instead of
    guessing fields from a task-specific JSON shape.
    """

    costs: dict[str, JSONValue] = {}
    for name, raw in sorted(getattr(summary, "cost_statistics").items()):
        values = dict(raw)
        costs[name] = {
            "defined": bool(values["defined"]),
            "sample_budget": values.get("sample_budget"),
            "statistical_weight": values.get("statistical_weight"),
            "gradient_evaluations": values["gradient_evaluations"],
            "formula_seconds": None,
            "wall_seconds": None,
            "reason": (
                "local_timing_excluded_from_canonical_artifact"
                if values["defined"]
                else values["reason"]
            ),
        }
    payload: dict[str, JSONValue] = {
        "schema_version": "stage2-paired-wave-summary-v1",
        "wave_id": str(getattr(summary, "wave_id")),
        "registry_hash": str(getattr(summary, "registry_hash")),
        "reference_hash": str(getattr(summary, "reference_hash")),
        "reference_hashes": dict(getattr(summary, "reference_hashes")),
        "expected_unit_ids": list(getattr(summary, "expected_unit_ids")),
        "completed_unit_ids": list(getattr(summary, "completed_unit_ids")),
        "complete": bool(getattr(summary, "complete")),
        "status": str(getattr(summary, "status")),
        "method_statistics": {
            name: dict(values)
            for name, values in sorted(getattr(summary, "method_statistics").items())
        },
        "reference_statistics": {
            name: {method: dict(values) for method, values in methods.items()}
            for name, methods in sorted(getattr(summary, "reference_statistics").items())
        },
        "microbatch_diagnostics": [
            dict(item) for item in getattr(summary, "microbatch_diagnostics")
        ],
        "replay_evidence": thaw_json_value(getattr(summary, "replay_evidence")),
        "cost_statistics": costs,
        "scope": str(getattr(summary, "scope")),
        "formal_eligible": False,
        "qualification_gate_hash": None,
        "resumed_unit_count": int(getattr(summary, "resumed_unit_count")),
        "weighting_assumptions": dict(getattr(summary, "weighting_assumptions")),
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def _require_formal_experiment_plan(
    request: TaskExecutionRequest,
    root: Path,
    *,
    context: _ProviderContext,
    sampling: SamplingPlan,
    inputs: _PredecessorContext,
) -> tuple[FormalExperimentPlan | None, tuple[str, ...]]:
    """严格装载并绑定当前 Stage 2 正式 B/M/R 计划。

    计划同时绑定任务 ID、固定抽样计划、正式执行证据以及完整直接前驱 commit 集。
    任一身份漂移都返回结构化 ``BLOCKED``，不会使用 fixture B/M/R 继续执行。
    """

    if request.config.run_intent == "local_fixture":
        return None, ()
    value, reference = _formal_input_document(
        request,
        root,
        schema_version="stage2-formal-experiment-plan-v1",
        requirement="formal_stage2_experiment_plan",
    )
    try:
        plan = FormalExperimentPlan.from_mapping(value)
    except (TypeError, ValueError, FormalRunRejected) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage2_experiment_plan",
            f"formal B/M/R plan 无法严格加载：{type(error).__name__}: {error}",
            retryable=False,
            evidence_refs=(reference,),
        ) from error
    expected_refs = tuple(sorted(item.commit_ref for item in inputs.artifacts))
    mismatches: list[str] = []
    if plan.task_id != request.task.task_id:
        mismatches.append("task_id")
    if plan.sampling_plan_hash != sampling.digest:
        mismatches.append("sampling_plan_hash")
    if plan.execution_evidence_hash != context.evidence.artifact_hash:
        mismatches.append("execution_evidence_hash")
    if plan.source_artifact_refs != expected_refs:
        mismatches.append("source_artifact_refs")
    if mismatches:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage2_experiment_plan",
            f"formal B/M/R plan 与当前执行身份不一致：{mismatches}",
            retryable=False,
            evidence_refs=(reference, *expected_refs),
        )
    return plan, (reference,)


def _formal_s204_reference_views(
    inputs: _PredecessorContext,
    root: Path,
    context: _ProviderContext,
) -> dict[str, Mapping[str, np.ndarray]]:
    """Load exactly the qualified S2.4/G2.3 reference, never an auxiliary file."""

    matches = [
        item
        for item in inputs.artifacts
        if item.task_id == "stage2.04_reference_target"
        and item.artifact_kind == "reference_result"
    ]
    if len(matches) != 1:
        raise ValueError("S25_FORMAL_REFERENCE_RESULT_NOT_UNIQUE")
    item = matches[0]
    if item.run_intent != "formal" or item.formal_eligible is not True:
        raise ValueError("S25_FORMAL_REFERENCE_COMMIT_NOT_FORMAL")
    manifest = item.payload
    validate_reference_result_artifact(manifest)
    if manifest.get("scope") != "formal" or manifest.get("formal_eligible") is not False:
        raise ValueError("S25_FORMAL_REFERENCE_SCOPE_INVALID")
    if manifest.get("registry_hash") != context.provider.registry_hash:
        raise ValueError("S25_FORMAL_REFERENCE_REGISTRY_MISMATCH")

    try:
        gate = inputs.payload_for("stage2.04_reference_target", "gate_record")
    except RuntimeError as error:
        raise ValueError("S25_FORMAL_REFERENCE_G23_GATE_NOT_UNIQUE") from error
    if gate.get("gate_id") != "stage2.G2.3" or gate.get("gate_status") != "PASS":
        raise ValueError("S25_FORMAL_REFERENCE_G23_NOT_PASS")
    bundle_ref = str(manifest["tensor_bundle_ref"])
    state, bundle = load_tensor_bundle(
        _workspace_path(root, bundle_ref, field="reference_result.tensor_bundle_ref")
    )
    if bundle.manifest_sha256 != manifest.get("tensor_bundle_manifest_hash"):
        raise ValueError("S25_FORMAL_REFERENCE_BUNDLE_HASH_MISMATCH")
    if not isinstance(state, Mapping) or set(state) != {
        "bias_reference", "cross_reference", "ranking_reference"
    }:
        raise ValueError("S25_FORMAL_REFERENCE_BUNDLE_VIEWS_INVALID")
    views = {
        "bias": _as_numpy_vector(state["bias_reference"]),  # type: ignore[arg-type]
        "cross": _as_numpy_vector(state["cross_reference"]),  # type: ignore[arg-type]
        "ranking": _as_numpy_vector(state["ranking_reference"]),  # type: ignore[arg-type]
    }
    expected_hashes = {
        "bias": manifest.get("bias_reference_hash"),
        "cross": manifest.get("cross_reference_hash"),
        "ranking": manifest.get("ranking_reference_hash"),
    }
    for name, view in views.items():
        if _vector_digest(view) != expected_hashes[name]:
            raise ValueError(f"S25_FORMAL_REFERENCE_{name.upper()}_HASH_MISMATCH")
    return views


def _run_stage2_estimator(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    context = _provider_context(request, root)
    sampling = _sampling_plan(request, context)
    experiment_plan, plan_refs = _require_formal_experiment_plan(
        request,
        root,
        context=context,
        sampling=sampling,
        inputs=inputs,
    )
    stream = (
        "pilot"
        if request.task.task_id == "stage2.05_paired_estimator_runner"
        else "confirmatory"
    )
    mappings = _paired_mappings(sampling, stream=stream, plan=experiment_plan)
    if request.task.task_id == "stage2.05_paired_estimator_runner":
        reference_views: Mapping[str, Mapping[str, np.ndarray]]
        try:
            if request.config.run_intent == "formal":
                reference_views = _formal_s204_reference_views(inputs, root, context)
                reference = reference_views["bias"]
                reference_hash = _vector_digest(reference)
            else:
                reference_manifest = inputs.payload("reference_result")
                raise RuntimeError("FIXTURE_REFERENCE_DIRECT_PATH")
        except RuntimeError as error:
            if request.config.run_intent != "local_fixture":
                raise _blocked(
                    BlockerCode.ASSET_UNAVAILABLE,
                    "stage2_reference_tensor_bundle",
                    "formal S2.5 requires an independently published reference_result",
                    retryable=False,
                    evidence_refs=inputs.references,
                ) from error
            # The plan DAG intentionally runs S2.5 from S2.2+S2.3.  The local
            # provider can derive the exact fixture anchor directly; formal
            # execution remains fail-closed until S2.4 publishes its bundle.
            reference = _exact_importance_reference(context)
            reference_views = {name: reference for name in ("bias", "cross", "ranking")}
            reference_hash = _vector_digest(reference)
        except (KeyError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage2_reference_tensor_bundle",
                f"前序 formal reference_result 无法恢复：{error}",
                retryable=False,
                evidence_refs=inputs.references,
            ) from error
    else:
        reference = _exact_importance_reference(context)
    summary = RecoverablePairedWaveRunner(
        context.provider,
        execution=context.evidence,
    ).run(
        wave_id=(
            f"wave-{request.config.config_hash[:16]}"
            if experiment_plan is None
            else experiment_plan.wave_id
        ),
        mappings=mappings,
        reference=reference,
        reference_hash=(
            reference_hash
            if request.task.task_id == "stage2.05_paired_estimator_runner"
            else _vector_digest(reference)
        ),
        references=(
            reference_views
            if request.task.task_id == "stage2.05_paired_estimator_runner"
            else None
        ),
        artifact_root=store.root / "resume" / "paired-wave",
    )
    stable = _stable_wave_payload(summary)
    commits = sorted(
        path.name
        for path in (store.root / "resume" / "paired-wave" / "commits").glob("*.json")
    )
    committed_shards: list[dict[str, JSONValue]] = []
    for commit_name in commits:
        commit = load_canonical_json(
            store.root / "resume" / "paired-wave" / "commits" / commit_name
        )
        if not isinstance(commit, Mapping):
            raise ValueError("S25_COMMIT_NOT_MAPPING")
        committed_shards.append(
            {
                "unit_id": str(commit["unit_id"]),
                "attempt_id": str(commit["attempt_id"]),
                "input_hash": str(commit["input_hash"]),
                "scientific_digest": str(commit["scientific_digest"]),
                "object_manifest_hash": str(commit["object_manifest_hash"]),
            }
        )
    shard_payload: dict[str, JSONValue] = {
        "schema_version": "stage2-task-sufficient-stat-shards-v1",
        "sampling_plan_hash": sampling.digest,
        "experiment_plan_hash": (
            None if experiment_plan is None else experiment_plan.artifact_hash
        ),
        "mapping_hashes": [mapping.digest for mapping in mappings],
        "committed_units": commits,
        "committed_shards": committed_shards,
        "expected_unit_count": len(mappings),
        "complete": bool(summary.complete),
        "recovery_semantics": "immutable_tensor_bundle_plus_authoritative_commit",
        "reference_hashes": dict(summary.reference_hashes),
        "reference_statistics": {
            name: {method: dict(values) for method, values in methods.items()}
            for name, methods in summary.reference_statistics.items()
        },
        "microbatch_diagnostics": [
            dict(item) for item in summary.microbatch_diagnostics
        ],
        "replay_evidence": thaw_json_value(summary.replay_evidence),
        "failure_evidence_dir": "resume/paired-wave/failures",
        "formal_eligible": False,
    }
    if request.task.task_id == "stage2.05_paired_estimator_runner":
        payloads = {
            "paired_runner_report": {
                **stable,
                "provider": context.to_payload(),
                "sampling_plan": sampling.to_dict(),
                "experiment_plan_hash": (
                    None if experiment_plan is None else experiment_plan.artifact_hash
                ),
            },
            "sufficient_stat_shards": shard_payload,
            "gate_record": _gate_candidate(request),
        }
    else:
        pilot_report = inputs.payload("pilot_report")
        matrix = inputs.payload("frozen_experiment_matrix")
        recommendation = pilot_report.get("recommendation")
        expected_status = (
            "FIXTURE_RECOMMENDATION"
            if request.config.run_intent == "local_fixture"
            else "FORMAL_CANDIDATE"
        )
        if not isinstance(recommendation, Mapping) or recommendation.get(
            "status"
        ) != expected_status:
            raise _blocked(
                BlockerCode.CONTRACT_UNFROZEN,
                "stage2_estimator_recommendation",
                "确认性实验缺少前序可用 estimator recommendation",
                evidence_refs=inputs.references,
            )
        if experiment_plan is not None:
            primary_m = experiment_plan.microbatch_counts[0]
            matrix_m = matrix.get("microbatch_counts")
            selected = recommendation.get("selected_estimator")
            formal_bindings_valid = (
                recommendation.get("scope") == "formal"
                and recommendation.get("execution_evidence_hash")
                == context.evidence.artifact_hash
                and recommendation.get("batch_size") == experiment_plan.batch_size
                and recommendation.get("repetitions") == experiment_plan.repetitions
                and matrix.get("scope") == "formal"
                and matrix.get("formal_freeze_status") == "FROZEN_CANDIDATE"
                and matrix.get("sampling_plan_hash") == sampling.digest
                and matrix.get("batch_sizes") == [experiment_plan.batch_size]
                and matrix.get("repetitions") == experiment_plan.repetitions
                and isinstance(matrix_m, list)
                and primary_m in matrix_m
                and (
                    selected == "double"
                    or recommendation.get("microbatch_count") == primary_m
                )
            )
            if not formal_bindings_valid:
                raise _blocked(
                    BlockerCode.CONTRACT_UNFROZEN,
                    "stage2_frozen_primary_matrix",
                    "确认性 B/M/R 与前序 pilot recommendation/matrix 不一致",
                    retryable=False,
                    evidence_refs=inputs.references,
                )
        payloads = {
            "confirmatory_results": {
                **stable,
                "provider": context.to_payload(),
                "sampling_plan_hash": sampling.digest,
                "pilot_recommendation": dict(recommendation),
                "experiment_matrix_hash": canonical_json_hash(matrix),
                "experiment_plan_hash": (
                    None
                    if experiment_plan is None
                    else experiment_plan.artifact_hash
                ),
                "upstream_binding_hash": inputs.binding_hash,
            },
            "sufficient_stat_shards": shard_payload,
            "completeness_report": {
                "schema_version": "stage2-task-completeness-report-v1",
                "expected_unit_ids": list(summary.expected_unit_ids),
                "completed_unit_ids": list(summary.completed_unit_ids),
                "complete": bool(summary.complete),
                "formal_eligible": False,
            },
        }
    return payloads, _source_refs(request, plan_refs)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    if np.all(left_ranks == left_ranks[0]) or np.all(right_ranks == right_ranks[0]):
        raise ValueError("SPEARMAN_UNDEFINED_CONSTANT_VECTOR")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def _topk_overlap(left: np.ndarray, right: np.ndarray, *, fraction: float = 0.5) -> float:
    count = max(1, math.ceil(left.size * fraction))
    indices = np.arange(left.size)
    left_top = set(np.lexsort((indices, -np.abs(left)))[:count].tolist())
    right_top = set(np.lexsort((indices, -np.abs(right)))[:count].tolist())
    return len(left_top.intersection(right_top)) / count


def _pilot_observations(
    context: _ProviderContext,
    mappings: Sequence[RepetitionMapping],
    reference: Mapping[str, object],
    *,
    cell_id: str,
    u_microbatch_count: int,
) -> tuple[PilotCellObservation, ...]:
    """从真实 paired vectors 计算缩小 pilot 指标，而不是写死推荐结论。"""

    runner = PairedEstimatorRunner(context.provider)
    method_vectors: dict[str, list[np.ndarray]] = {"u": [], "double": []}
    for mapping in mappings:
        result = runner.run(mapping)
        method_vectors["u"].append(
            _flatten(result.vectors[f"u_m{u_microbatch_count}"])
        )
        method_vectors["double"].append(_flatten(result.vectors["double"]))
    target = _flatten(reference)
    target_scale = float(np.square(target).mean())
    if target_scale <= 0:
        raise ValueError("PILOT_REFERENCE_SCALE_ZERO")
    observations: list[PilotCellObservation] = []
    for estimator, vectors in method_vectors.items():
        matrix = np.stack(vectors)
        mean = matrix.mean(axis=0)
        coordinate_errors = matrix - target[None, :]
        scalar_errors = coordinate_errors.mean(axis=1)
        center = float(scalar_errors.mean())
        half_width = (
            0.0
            if len(scalar_errors) < 2
            else float(2.0 * scalar_errors.std(ddof=1) / math.sqrt(len(scalar_errors)))
        )
        observations.append(
            PilotCellObservation(
                cell_id=cell_id,
                estimator=estimator,
                batch_size=mappings[0].batch_size,
                microbatch_count=(u_microbatch_count if estimator == "u" else 2),
                repetitions=len(mappings),
                bias_interval_low=center - half_width,
                bias_interval_high=center + half_width,
                corrected_nmse_ratio=float(np.square(coordinate_errors).mean() / target_scale),
                spearman=_spearman(mean, target),
                topk_overlap=_topk_overlap(mean, target),
                # paired runner 的 scientific equal-sample 口径下两种 estimator 共用同一梯度池。
                online_cost_ratio=1.0,
            )
        )
    return tuple(observations)


def _run_stage2_pilot(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    paired_report = inputs.payload("paired_runner_report")
    if paired_report.get("complete") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "complete_paired_runner_report",
            "pilot 只能消费完整的前序 paired wave",
            evidence_refs=inputs.references,
        )
    context = _provider_context(request, root)
    sampling = _sampling_plan(request, context)
    experiment_plan, plan_refs = _require_formal_experiment_plan(
        request,
        root,
        context=context,
        sampling=sampling,
        inputs=inputs,
    )
    mappings = _paired_mappings(
        sampling,
        stream="pilot",
        plan=experiment_plan,
    )
    reference = _exact_importance_reference(context)
    summary = RecoverablePairedWaveRunner(
        context.provider,
        execution=context.evidence,
    ).run(
        wave_id=(
            f"pilot-{request.config.config_hash[:16]}"
            if experiment_plan is None
            else experiment_plan.wave_id
        ),
        mappings=mappings,
        reference=reference,
        reference_hash=_vector_digest(reference),
        artifact_root=store.root / "resume" / "pilot-wave",
    )
    cell_id = "fixture-anchor" if experiment_plan is None else experiment_plan.cell_id
    primary_m = (
        _FIXTURE_PAIRED_M_VALUES[-1]
        if experiment_plan is None
        else max(experiment_plan.microbatch_counts)
    )
    observations = _pilot_observations(
        context,
        mappings,
        reference,
        cell_id=cell_id,
        u_microbatch_count=primary_m,
    )
    if experiment_plan is None:
        thresholds = PilotThresholds(
            bias_margin=10.0,
            max_corrected_nmse_ratio=1e6,
            min_spearman=-1.0,
            min_topk_overlap=0.0,
            max_online_cost_ratio=2.0,
        )
    else:
        assert experiment_plan.pilot_thresholds is not None
        thresholds = PilotThresholds(**dict(experiment_plan.pilot_thresholds))  # type: ignore[arg-type]
    recommendation = Stage2RecommendationEngine().recommend(
        recommendation_id=(
            f"fixture-estimator-{request.config.config_hash[:16]}"
            if experiment_plan is None
            else experiment_plan.plan_id
        ),
        observations=observations,
        required_cells=(cell_id,),
        thresholds=thresholds,
        execution=context.evidence,
    )
    # S2.6 contract layer is deliberately kept separate from the existing
    # fixture recommendation.  A local synthetic provider is one anchor only;
    # therefore the blind six-anchor scan remains BLOCKED/UNFROZEN and cannot
    # be mistaken for a formal matrix.  The calibration itself is useful local
    # evidence and does not consume confirmatory draws.
    calibration_rng = np.random.default_rng(206)
    calibration = run_artificial_distribution_calibration(
        calibration_rng.normal(size=(64, 2)),
        batch_size=32,
        m_values=(2, 4, 8, 16, 32),
    )
    matrix_contract = freeze_fixture_matrix(
        (),
        required_anchor_ids=ANCHOR_IDS,
        r_max=1000,
        cost_semantics=CostSemantics(
            **{
                name: {"defined": False, "reason": "local_fixture_timing_not_formal"}
                for name in (
                    "scientific_equal_sample_cost",
                    "isolated_estimator_cost",
                    "online_training_incremental_cost",
                )
            },
            cost_io_quiescent=False,
        ),
    )
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "pilot_report": {
            "schema_version": "stage2-task-pilot-report-v1",
            "wave": _stable_wave_payload(summary),
            "observations": [
                {
                    "cell_id": item.cell_id,
                    "estimator": item.estimator,
                    "batch_size": item.batch_size,
                    "microbatch_count": item.microbatch_count,
                    "repetitions": item.repetitions,
                    "bias_interval_low": item.bias_interval_low,
                    "bias_interval_high": item.bias_interval_high,
                    "corrected_nmse_ratio": item.corrected_nmse_ratio,
                    "spearman": item.spearman,
                    "topk_overlap": item.topk_overlap,
                    "online_cost_ratio": item.online_cost_ratio,
                    "quality_complete": item.quality_complete,
                }
                for item in observations
            ],
            "recommendation": recommendation.to_dict(),
            "provider": context.to_payload(),
            "experiment_plan_hash": (
                None if experiment_plan is None else experiment_plan.artifact_hash
            ),
            "development_wave_hash": canonical_json_hash(paired_report),
            "upstream_binding_hash": inputs.binding_hash,
            "artificial_calibration": calibration.to_dict(),
            "matrix_freeze_contract": matrix_contract.to_dict(),
            "matrix_freeze_status": matrix_contract.status,
            "formal_matrix_generated": False,
            "formal_block_reason": "G2.3_PENDING_EXTERNAL_AUTHORIZATION",
        },
        "frozen_experiment_matrix": {
            "schema_version": (
                "stage2-task-fixture-matrix-v1"
                if experiment_plan is None
                else "stage2-task-experiment-matrix-v1"
            ),
            "scope": request.config.run_intent,
            "formal_eligible": False,
            "batch_sizes": [mappings[0].batch_size],
            "microbatch_counts": list(mappings[0].m_values),
            "repetitions": len(mappings),
            "sampling_plan_hash": sampling.digest,
            "experiment_plan_hash": (
                None if experiment_plan is None else experiment_plan.artifact_hash
            ),
            "mapping_hashes": [mapping.digest for mapping in mappings],
            "formal_freeze_status": (
                "UNFROZEN"
                if experiment_plan is None
                else "FROZEN_CANDIDATE"
            ),
            "matrix_freeze_contract": matrix_contract.to_dict(),
            "confirmatory_mapping_status": "NOT_GENERATED",
            "formal_block_reason": "G2.3_PENDING_EXTERNAL_AUTHORIZATION",
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, _source_refs(request, plan_refs)


def _analysis_metric_payload(result: object) -> dict[str, JSONValue]:
    """把 analysis.MetricResult 转为保留 undefined reason 的 JSON。"""

    return {
        "defined": bool(getattr(result, "defined")),
        "value": getattr(result, "value"),
        "reason": getattr(result, "reason"),
        "metadata": dict(getattr(result, "metadata")),
    }


def _stage2_statistics_table(confirmatory: Mapping[str, object]) -> FrozenSourceTable:
    statistics = confirmatory.get("method_statistics")
    recommendation = confirmatory.get("pilot_recommendation")
    if not isinstance(statistics, Mapping) or not isinstance(recommendation, Mapping):
        raise ValueError("STAGE2_CONFIRMATORY_STATISTICS_OR_RECOMMENDATION_MISSING")
    rows: list[dict[str, object]] = []
    for method, raw in sorted(statistics.items()):
        if not isinstance(method, str) or not isinstance(raw, Mapping):
            raise TypeError("STAGE2_METHOD_STATISTIC_ROW_INVALID")
        required = {
            "repetitions",
            "coordinate_count",
            "bias",
            "absolute_bias",
            "variance",
            "mse",
            "mae",
            "negative_fraction",
            "positive_mass",
            "negative_mass",
        }
        if set(raw) != required:
            raise ValueError(f"STAGE2_METHOD_STATISTIC_FIELDS_INVALID:{method}")
        rows.append(
            {
                "method": method,
                **{name: raw[name] for name in sorted(required)},
                "pilot_selected_estimator": recommendation.get("selected_estimator"),
                "batch_size": recommendation.get("batch_size"),
                "microbatch_count": recommendation.get("microbatch_count"),
                "pilot_repetitions": recommendation.get("repetitions"),
            }
        )
    return FrozenSourceTable.from_rows(
        name="stage2_confirmatory_statistics",
        schema_version="stage2-confirmatory-statistics-table-v1",
        rows=rows,
    )


def _run_stage2_statistics(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """从确认性 paired 结果重建冻结统计表并执行稳健性指标。"""

    inputs = _predecessor_context(request, root, store)
    confirmatory = inputs.payload("confirmatory_results")
    completeness = inputs.payload("completeness_report")
    if completeness.get("complete") is not True or confirmatory.get("complete") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "complete_confirmatory_results",
            "统计任务拒绝不完整的确认性 repetition 集",
            evidence_refs=inputs.references,
        )
    try:
        table = _stage2_statistics_table(confirmatory)
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "confirmatory_statistics_schema",
            f"确认性统计字段无效：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    rows = [dict(row) for row in table.rows]
    methods = [str(row["method"]) for row in rows]
    biases = np.asarray([float(row["bias"]) for row in rows], dtype=np.float64)
    mses = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)
    maes = np.asarray([float(row["mae"]) for row in rows], dtype=np.float64)
    quality_metrics = {
        "pooled_method_bias": _analysis_metric_payload(
            analysis_bias(biases, np.zeros_like(biases))
        ),
        "mse_mae_pearson": _analysis_metric_payload(analysis_pearson(mses, maes)),
        "mse_mae_spearman": _analysis_metric_payload(analysis_spearman(mses, maes)),
        "best_half_overlap": _analysis_metric_payload(
            analysis_top_k_overlap(
                -mses,
                -maes,
                max(1, len(methods) // 2),
                canonical_ids=methods,
            )
        ),
    }
    recommendation = confirmatory["pilot_recommendation"]
    assert isinstance(recommendation, Mapping)
    selected_family = str(recommendation.get("selected_estimator"))
    selected_method = (
        f"u_m{recommendation.get('microbatch_count')}"
        if selected_family == "u"
        else "double"
    )
    hypothesis_decisions: dict[str, JSONValue] = {
        "schema_version": "stage2-task-hypothesis-decisions-v1",
        "scope": request.config.run_intent,
        "selected_candidate_method": selected_method,
        "selected_candidate_present": selected_method in methods,
        "decision_status": "FIXTURE_EVIDENCE_ONLY",
        "formal_decision_status": "UNFROZEN",
        "source_table_hash": table.content_hash,
        "upstream_binding_hash": inputs.binding_hash,
    }
    quality_gates: dict[str, JSONValue] = {
        "schema_version": "stage2-task-quality-gates-v1",
        "metrics": quality_metrics,
        "all_values_finite": all(
            math.isfinite(float(row[name]))
            for row in rows
            for name in ("bias", "variance", "mse", "mae")
        ),
        "cost_statistics": confirmatory.get("cost_statistics"),
        "pilot_recommendation": dict(recommendation),
        "gate_status": "NOT_RUN",
        "local_validation_status": "PASS",
    }
    return (
        {
            "frozen_source_table": table.to_dict(),  # type: ignore[dict-item]
            "quality_gates": quality_gates,
            "hypothesis_decisions": hypothesis_decisions,
        },
        inputs.references,
    )


def _run_stage2_capacity(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """核对三种成本语义，并保留本机未测 wall/system 能力的显式状态。"""

    inputs = _predecessor_context(request, root, store)
    confirmatory = inputs.payload("confirmatory_results")
    shards = inputs.payload("sufficient_stat_shards")
    if confirmatory.get("complete") is not True or shards.get("complete") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "complete_confirmatory_results",
            "S2.9 requires complete S2.7 confirmatory results and sufficient-stat shards",
            evidence_refs=inputs.references,
        )
    try:
        statistics = _stage2_statistics_table(confirmatory)
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_confirmatory_statistics_table",
            f"S2.7 confirmatory statistics are invalid: {error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    costs = confirmatory.get("cost_statistics")
    if not isinstance(costs, Mapping) or set(costs) != {
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    }:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_cost_semantics",
            "前序质量报告未完整记录三种成本口径",
            retryable=False,
            evidence_refs=inputs.references,
        )
    scientific = costs["scientific_equal_sample_cost"]
    if not isinstance(scientific, Mapping) or scientific.get("defined") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "scientific_equal_sample_cost",
            "本机 fixture 缺少可定义的等样本梯度调用成本",
            evidence_refs=inputs.references,
        )
    gradient_evaluations = scientific.get("gradient_evaluations")
    cost_rows = [
        {
            **dict(row),
            "scientific_gradient_evaluations": gradient_evaluations,
            "isolated_estimator_cost_defined": bool(
                isinstance(costs["isolated_estimator_cost"], Mapping)
                and costs["isolated_estimator_cost"].get("defined")
            ),
            "online_incremental_cost_defined": bool(
                isinstance(costs["online_training_incremental_cost"], Mapping)
                and costs["online_training_incremental_cost"].get("defined")
            ),
        }
        for row in statistics.rows
    ]
    cost_table = FrozenSourceTable.from_rows(
        name="stage2_cost_accuracy",
        schema_version="stage2-cost-accuracy-table-v1",
        rows=cost_rows,
    )
    parameter_count = max(int(row["coordinate_count"]) for row in statistics.rows)
    statistics_bytes = estimate_parameter_statistics_bytes(
        parameter_count,
        resident_fp32_buffers=3,
        transient_fp32_buffers=2,
    )
    experiment_bytes = estimate_experiment_storage(
        parameter_count=parameter_count,
        retained_checkpoints=1,
        resident_fp32_buffers=3,
        seed_count=1,
        parallel_runs=1,
        logs_and_reports_per_run=1024 * 1024,
    )
    storage_budget = StorageBudget.from_expected(
        "stage2-local-fixture",
        experiment_bytes,
    )
    recommendation = confirmatory.get("pilot_recommendation")
    if not isinstance(recommendation, Mapping):
        raise ValueError("STAGE2_CAPACITY_RECOMMENDATION_MISSING")
    system_report: dict[str, JSONValue] = {
        "schema_version": "stage2-task-system-validation-v1",
        "source_statistics_hash": statistics.content_hash,
        "source_cost_table_hash": cost_table.content_hash,
        "cost_semantics": {name: dict(value) for name, value in costs.items()},  # type: ignore[union-attr]
        "fixture_estimator_inputs": dict(recommendation),
        "deterministic_capacity_estimate": {
            "parameter_count": parameter_count,
            "parameter_statistics_bytes": statistics_bytes,
            "experiment_storage_bytes": experiment_bytes,
            "storage_budget": storage_budget.as_dict(),
            "instantaneous_free_space_excluded_from_artifact": True,
        },
        "cpu_fixture_replay": "PASS",
        "cuda_validation": "NOT_RUN",
        "nccl_validation": "NOT_RUN",
        "formal_system_gate": "NOT_RUN",
        "upstream_binding_hash": inputs.binding_hash,
    }
    return (
        {
            "cost_table": cost_table.to_dict(),  # type: ignore[dict-item]
            "system_validation_report": system_report,
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _run_stage2_reporting(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """只从冻结成本/误差表构建报告、图表 spec 与 fixture decision。"""

    inputs = _predecessor_context(request, root, store)
    if request.config.run_intent == "formal":
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            "stage2.G2.7b",
            "正式 EstimatorDecision 必须由独立 Gate 资格化；runner 不得自签 PASS",
            evidence_refs=inputs.references,
        )
    try:
        table = FrozenSourceTable.from_mapping(inputs.payload("cost_table"))
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_cost_table",
            f"成本源表不可用：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    system = inputs.payload("system_validation_report")
    decision_inputs = system.get("fixture_estimator_inputs")
    if not isinstance(decision_inputs, Mapping):
        raise ValueError("STAGE2_FIXTURE_DECISION_INPUTS_MISSING")
    selected = str(decision_inputs.get("selected_estimator"))
    pair = PrimaryPairDecision(
        status="FIXTURE_SELECTED",
        batch_size=int(decision_inputs["batch_size"]),
        microbatch_count=int(decision_inputs["microbatch_count"]),
        scope="local_fixture",
        formal_eligible=False,
        evaluations=(),
    )
    decision = build_fixture_estimator_decision(
        pair,
        selected_estimator=selected,
        repetitions=int(decision_inputs["repetitions"]),
    )
    rows = [dict(row) for row in table.rows]
    mse_values = np.asarray([float(row["mse"]) for row in rows], dtype=np.float64)
    mae_values = np.asarray([float(row["mae"]) for row in rows], dtype=np.float64)
    builder = AnalysisReportBuilder(
        report_id=f"stage2-fixture-{table.content_hash[:16]}"
    )
    builder.add_source(table)
    builder.add_metric(
        "mean_method_mse",
        analysis_bias(mse_values, np.zeros_like(mse_values)),
        source=table,
        derivation_id="stage2.mean-method-mse.v1",
        input_columns=("mse",),
    )
    builder.add_metric(
        "mse_mae_spearman",
        analysis_spearman(mse_values, mae_values),
        source=table,
        derivation_id="stage2.mse-mae-spearman.v1",
        input_columns=("mse", "mae"),
    )
    report = builder.build(
        metadata={
            "scope": "local_fixture",
            "formal_eligible": False,
            "estimator_decision_hash": decision.artifact_hash,
        }
    )
    spec = ChartSpec.from_table(
        table,
        chart_id=f"stage2-mse-{table.content_hash[:12]}",
        chart_type="bar",
        x_column="method",
        y_columns=("mse", "mae"),
        sort_columns=("method",),
    )
    chart = ChartArtifact.from_spec(spec)
    return (
        {
            "analysis_report": report.to_dict(),  # type: ignore[dict-item]
            "chart_artifacts": {
                "schema_version": "stage2-task-chart-artifacts-v1",
                "source_table_hash": table.content_hash,
                "artifacts": [chart.to_dict()],  # type: ignore[list-item]
                "manual_numeric_edits_allowed": False,
            },
            "estimator_decision": decision.to_dict(),  # type: ignore[dict-item]
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _run_stage2_delivery(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """复核 Stage 2 发布集合；外部 Git/服务器同步保持显式 BLOCKED。"""

    inputs = _predecessor_context(request, root, store)
    decision = inputs.payload("estimator_decision")
    report = inputs.payload("analysis_report")
    charts = inputs.payload("chart_artifacts")
    try:
        # 两个公共 loader 会重新计算各自内容 hash，而不是相信上游摘要字段。
        from .stage2 import EstimatorDecision
        from ..analysis import AnalysisReport

        restored_decision = EstimatorDecision.from_mapping(decision)
        restored_report = AnalysisReport.from_mapping(report)
        chart_values = charts.get("artifacts")
        if not isinstance(chart_values, list) or not chart_values:
            raise TypeError("STAGE2_DELIVERY_CHARTS_NOT_NONEMPTY_ARRAY")
        restored_charts = [
            ChartArtifact.from_mapping(value)
            for value in chart_values
            if isinstance(value, Mapping)
        ]
        if len(restored_charts) != len(chart_values):
            raise TypeError("STAGE2_DELIVERY_CHART_ITEM_NOT_MAPPING")
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_delivery_inputs",
            f"Stage 2 报告或 decision 无法严格重放：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    inventory = [
        {
            "artifact_kind": item.artifact_kind,
            "artifact_hash": item.artifact_hash,
            "commit_ref": item.commit_ref,
        }
        for item in inputs.artifacts
    ]
    return (
        {
            "delivery_manifest": {
                "schema_version": "stage2-task-delivery-manifest-v1",
                "scope": request.config.run_intent,
                "artifacts": inventory,
                "decision_hash": restored_decision.artifact_hash,
                "estimator_decision": restored_decision.to_dict(),
                "report_hash": restored_report.report_hash,
                "chart_artifact_hashes": [
                    chart.artifact_hash for chart in restored_charts
                ],
                "formal_stage_complete": False,
            },
            "replay_report": {
                "schema_version": "stage2-task-replay-report-v1",
                "status": "PASS",
                "verified_commit_count": len(inputs.artifacts),
                "upstream_binding_hash": inputs.binding_hash,
            },
            "gate_summary": {
                "schema_version": "stage2-task-gate-summary-v1",
                "stage2.G2.7b": "NOT_RUN",
                "formal_exit_gate": "NOT_RUN",
                "local_validation_status": "PASS",
            },
            "sync_report": {
                "schema_version": "stage2-task-sync-report-v1",
                "github": "BLOCKED",
                "server": "BLOCKED",
                "reason": "server_unreachable_and_network_operations_out_of_scope",
            },
        },
        inputs.references,
    )


def _run_stage3_prerequisites(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """验证 Stage 2 交付和 estimator decision，再冻结 Stage 3 fixture 范围。"""

    inputs = _predecessor_context(request, root, store)
    delivery = inputs.payload("delivery_manifest")
    replay = inputs.payload("replay_report")
    embedded = delivery.get("estimator_decision")
    if not isinstance(embedded, Mapping) or replay.get("status") != "PASS":
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage2_delivery_replay",
            "Stage 3 缺少已严格重放的 Stage 2 decision 交付",
            evidence_refs=inputs.references,
        )
    try:
        from .stage2 import EstimatorDecision

        decision = EstimatorDecision.from_mapping(embedded)
        if request.config.run_intent == "formal":
            decision.require_formal()
    except (TypeError, ValueError, RuntimeError) as error:
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            "stage2_estimator_decision",
            f"EstimatorDecision 不满足 Stage 3 入口：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    prerequisite: dict[str, JSONValue] = {
        "schema_version": "stage3-task-prerequisite-report-v1",
        "stage2_delivery_hash": canonical_json_hash(delivery),
        "stage2_replay_hash": canonical_json_hash(replay),
        "estimator_decision_hash": decision.artifact_hash,
        "estimator_scope": decision.scope,
        "local_validation_status": "PASS",
        "formal_gate_status": "NOT_RUN",
        "upstream_binding_hash": inputs.binding_hash,
    }
    scope_freeze: dict[str, JSONValue] = {
        "schema_version": "stage3-task-scope-freeze-v1",
        "path_family": "linear_parameter_endpoint",
        "probe_roles": ["pilot", "formal", "replay"],
        "required_views": ["signed", "positive", "negative_mass", "absolute"],
        "reference_policy": "two_independent_rule_families_with_continuous_refinement",
        "formal_default_rule_status": "UNFROZEN",
        "formal_probe_count_status": "UNFROZEN",
        "formal_node_budget_status": "UNFROZEN",
        "scope": request.config.run_intent,
        "formal_eligible": False,
    }
    return (
        {
            "prerequisite_report": prerequisite,
            "scope_freeze": scope_freeze,
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _run_stage3_contract(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """冻结路径数学、完备性残差和求积规则注册表身份。"""

    inputs = _predecessor_context(request, root, store)
    prerequisites = inputs.payload("prerequisite_report")
    scope = inputs.payload("scope_freeze")
    if prerequisites.get("local_validation_status") != "PASS":
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            "stage3_prerequisite_report",
            "Stage 3 数学合同不能消费未通过本机重放的前置报告",
            evidence_refs=inputs.references,
        )
    rules = default_quadrature_rules()
    rule_manifest = {name: rule.to_dict() for name, rule in sorted(rules.items())}
    path_math: dict[str, JSONValue] = {
        "schema_version": "stage3-task-path-math-contract-v1",
        "path": "theta(alpha)=theta_pre+alpha*(theta_post-theta_pre)",
        "signed_contribution": "delta_theta*integral_0^1 gradient(theta(alpha)) d_alpha",
        "identities": [
            "signed=positive-negative_mass",
            "absolute=positive+negative_mass",
        ],
        "parameter_post_state_distinct_from_attempt_commit_state": True,
        "quadrature_weight_dtype": "float64",
        "rule_registry_hash": canonical_json_hash(rule_manifest),
        "scope_freeze_hash": canonical_json_hash(scope),
    }
    metric_contract: dict[str, JSONValue] = {
        "schema_version": "stage3-task-metric-contract-v1",
        "metrics": [
            "normalized_l1_error",
            "completeness_absolute_residual",
            "completeness_relative_residual",
            "completeness_l1_scaled_residual",
            "pearson",
            "spearman",
            "top_k_overlap",
        ],
        "undefined_policy": "defined_false_with_reason_no_epsilon",
        "reference_policy": scope["reference_policy"],
        "registered_rules": rule_manifest,  # type: ignore[dict-item]
        "upstream_binding_hash": inputs.binding_hash,
    }
    return (
        {
            "path_math_contract": path_math,
            "metric_contract": metric_contract,
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _sha(label: object) -> str:
    return canonical_json_hash(label)


@dataclass(slots=True)
class _FixtureEndpointAdapter:
    """严格模拟真实 step 的三个状态边界，并执行独立 replay 校验。"""

    pre: EndpointState
    parameter_post: EndpointState
    attempt_commit: EndpointState
    buffer_hash: str
    delta_hash: str
    phase: str = "pre"

    def capture_pre_state(self) -> EndpointState:
        if self.phase != "pre":
            raise RuntimeError("FIXTURE_ENDPOINT_PRE_CAPTURE_ORDER")
        return self.pre

    def apply_optimizer_update(self) -> None:
        if self.phase != "pre":
            raise RuntimeError("FIXTURE_ENDPOINT_OPTIMIZER_ORDER")
        self.phase = "parameter_post"

    def capture_parameter_post_state(self) -> EndpointState:
        if self.phase != "parameter_post":
            raise RuntimeError("FIXTURE_ENDPOINT_POST_CAPTURE_ORDER")
        return self.parameter_post

    def advance_attempt_commit(self) -> None:
        if self.phase != "parameter_post":
            raise RuntimeError("FIXTURE_ENDPOINT_COMMIT_ORDER")
        self.phase = "attempt_commit"

    def capture_attempt_commit_state(self) -> EndpointState:
        if self.phase != "attempt_commit":
            raise RuntimeError("FIXTURE_ENDPOINT_COMMIT_CAPTURE_ORDER")
        return self.attempt_commit

    def full_update_delta_hash(self) -> str:
        return self.delta_hash

    def probe_buffer_snapshot_hash(self) -> str:
        return self.buffer_hash

    def verify_replay(self, record: object) -> bool:
        return (
            self.phase == "attempt_commit"
            and getattr(record, "pre_state") == self.pre
            and getattr(record, "parameter_post_state") == self.parameter_post
            and getattr(record, "attempt_commit_state") == self.attempt_commit
        )

    def restore_pre_state(self) -> None:
        self.phase = "pre"


@dataclass(frozen=True, slots=True)
class _PathStateController:
    """把 fixture 常量摘要或 formal provider 摘要接到只读路径保护器。"""

    digest_fn: Callable[[], str]
    restore_fn: Callable[[], None]

    def digest(self) -> str:
        value = self.digest_fn()
        if not isinstance(value, str) or not value:
            raise ValueError("STAGE3_PATH_STATE_DIGEST_EMPTY")
        return value

    def restore(self) -> None:
        # formal provider 的每次节点/端点调用本身都会在 finally 中恢复完整快照；
        # controller 仍保留该窄接口，以便未来训练运行时传入真正的外层 restore。
        self.restore_fn()


@dataclass(frozen=True, slots=True)
class _PathContext:
    """同一份 local/formal 路径执行上下文。

    formal 与 fixture 只在状态/gradient provider 的来源上不同；求积、reference、
    pilot 和矩阵必须共享同一调用面，避免 formal 分支再次退回解析 fixture。
    """

    registry: ParameterRegistry
    path: PathSpec
    endpoint: object
    panel: ProbePanel
    primary_probe: ProbeSpec
    pre_bundle_path: Path
    pre_bundle: TensorBundle
    post_bundle_path: Path
    post_bundle: TensorBundle
    execution: FormalExecutionEvidence
    gradient_fn: Callable[[float, TensorMap], TensorMap]
    loss_fn: Callable[[TensorMap], torch.Tensor]
    unit_id: str
    state_controller: _PathStateController
    node_cache: PersistentNodeGradientCache
    node_cache_root_ref: str

    @property
    def precision(self) -> str:
        return (
            "float64"
            if self.path.accumulation_dtype == torch.float64
            else "float32"
        )

    def node_key(self, alpha: float) -> NodeCacheKey:
        """用路径、精度、坐标 registry 与 probe loss 合同构造唯一节点键。"""

        return NodeCacheKey(
            path_unit_id=self.unit_id,
            alpha=alpha,
            precision=self.precision,
            parameter_registry_hash=self.registry.coordinate_registry_hash,
            loss_contract_hash=self.primary_probe.loss_contract_hash,
        )

    def integrate(self, rule: object) -> PathIntegralResult:
        """经公共只读事务与持久节点缓存执行一次真实求积。"""

        evaluation = PathAnalysisRunner(node_cache=self.node_cache).run_bound(
            unit_id=self.unit_id,
            precision=self.precision,
            parameter_registry_hash=self.registry.coordinate_registry_hash,
            loss_contract_hash=self.primary_probe.loss_contract_hash,
            path_spec=self.path,
            rule=rule,
            gradient_callback=self.gradient_fn,
            loss_callback=self.loss_fn,
            state_controller=self.state_controller,
            scope=self.execution.run_intent,
            formal_eligible=self.execution.formal_eligible,
        )
        result = evaluation.result
        if not isinstance(result, PathIntegralResult):
            raise TypeError("STAGE3_PATH_INTEGRATOR_RESULT_INVALID")
        return result

    def cache_evidence(self, rules: Sequence[object]) -> dict[str, JSONValue]:
        """生成与规则集合绑定、fresh/resume 一致的节点缓存证据。

        这里故意不输出本次 cache hit/miss 数，因为中断发生在哪个规则之后会改变
        这些计数。产物改为列出每条规则请求的完整 NodeCacheKey、跨规则重复键、
        对应权威 commit 和 reconciliation；只要最终科学状态相同，证据 hash 就相同。
        """

        requests_by_rule: dict[str, dict[str, JSONValue]] = {}
        keys_by_digest: dict[str, NodeCacheKey] = {}
        key_rule_counts: dict[str, int] = {}
        for rule in rules:
            rule_hash = getattr(rule, "artifact_hash", None)
            rule_name = getattr(rule, "name", None)
            nodes = getattr(rule, "nodes", None)
            if (
                not isinstance(rule_hash, str)
                or not isinstance(rule_name, str)
                or not isinstance(nodes, torch.Tensor)
            ):
                raise TypeError("STAGE3_CACHE_EVIDENCE_RULE_INVALID")
            # 同一个冻结规则若被上层重复列出，只作为一个规则身份计数，避免制造
            # 虚假的“跨规则复用”；规则内部重复节点同样先规范化。
            if rule_hash in requests_by_rule:
                continue
            rule_keys = {
                self.node_key(float(alpha))
                for alpha in nodes.detach().cpu().to(torch.float64).tolist()
            }
            key_digests = sorted(key.digest for key in rule_keys)
            requests_by_rule[rule_hash] = {
                "rule_name": rule_name,
                "rule_hash": rule_hash,
                "node_key_digests": key_digests,
            }
            for key in rule_keys:
                keys_by_digest[key.digest] = key
                key_rule_counts[key.digest] = key_rule_counts.get(key.digest, 0) + 1

        commit_evidence = self.node_cache.commit_evidence(
            tuple(keys_by_digest[digest] for digest in sorted(keys_by_digest))
        )
        if commit_evidence["all_requested_keys_committed"] is not True:
            raise RuntimeError("STAGE3_NODE_CACHE_COMMIT_INCOMPLETE")
        shared = sorted(
            digest for digest, count in key_rule_counts.items() if count > 1
        )
        payload: dict[str, JSONValue] = {
            "schema_version": "stage3-path-node-cache-evidence-v1",
            "cache_root_ref": self.node_cache_root_ref,
            "path_unit_id": self.unit_id,
            "precision": self.precision,
            "parameter_registry_hash": self.registry.coordinate_registry_hash,
            "loss_contract_hash": self.primary_probe.loss_contract_hash,
            "rule_requests": [
                requests_by_rule[digest] for digest in sorted(requests_by_rule)
            ],
            "cross_rule_reused_key_digests": shared,
            "cross_rule_reused_key_count": len(shared),
            "commit_evidence": commit_evidence,  # type: ignore[dict-item]
        }
        payload["evidence_hash"] = canonical_json_hash(payload)
        return payload


def _fixture_path_scope_hash(request: TaskExecutionRequest) -> str:
    """排除 task/output 等编排字段后的 Stage 3 fixture 路径身份。"""

    identity = request.config.base_config.section("identity")
    return canonical_json_hash(
        {
            "master_seed": identity["master_seed"],
            "model": request.config.base_config.section("model"),
            "loss": request.config.base_config.section("loss"),
            "path_integration": request.config.base_config.section("path_integration"),
        }
    )


def _stage3_node_cache(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    unit_id: str,
    registry: ParameterRegistry,
) -> tuple[PersistentNodeGradientCache, str]:
    """构造同一路径单元跨 task/process 共用的安全节点缓存。

    正式 ``ResolvedConfig v2`` 与仓库 fixture 都声明 ``runtime.cache_root``，因此
    reference、pilot 和 matrix 会落在同一缓存根。仅为兼容直接调用内部 helper 的
    窄测试，在缺少 runtime 分区时回退到当前 task 的 ``resume`` 目录；该回退不会
    被正式 CLI 配置触发。
    """

    workspace = root.resolve()
    runtime: object
    try:
        runtime = request.config.base_config.section("runtime")
    except (AttributeError, KeyError, TypeError):
        runtime = None
    cache_value = runtime.get("cache_root") if isinstance(runtime, Mapping) else None
    if isinstance(cache_value, str):
        cache_base = _workspace_path(
            workspace,
            cache_value,
            field="runtime.cache_root",
        )
    else:
        cache_base = (store.root / "resume" / "node-gradient-cache").resolve()
    cache_root = (cache_base / "stage3-node-gradients" / unit_id).resolve()
    try:
        cache_root_ref = cache_root.relative_to(workspace).as_posix()
    except ValueError as error:
        raise ValueError("STAGE3_NODE_CACHE_PATH_ESCAPE") from error
    cache = PersistentNodeGradientCache(
        cache_root,
        codec=SafeTensorTreeCodec(registry=registry),
    )
    return cache, cache_root_ref


def _endpoint_state(
    artifact_id: str,
    artifact_hash: str,
    *,
    parameter_hash: str,
    buffer_hash: str,
    optimizer_hash: str,
    control_tag: str,
) -> EndpointState:
    return EndpointState(
        artifact_id=artifact_id,
        artifact_hash=artifact_hash,
        parameter_hash=parameter_hash,
        buffer_hash=buffer_hash,
        optimizer_hash=optimizer_hash,
        scheduler_hash=_sha([control_tag, "scheduler"]),
        scaler_hash=_sha([control_tag, "scaler"]),
        rng_hash=_sha([control_tag, "rng"]),
        data_cursor_hash=_sha([control_tag, "cursor"]),
        model_mode_hash=_sha([control_tag, "model-mode"]),
    )


def _fixture_path_context(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> _PathContext:
    if request.config.run_intent != "local_fixture":
        return _formal_path_context(request, root, store)
    path_scope_hash = _fixture_path_scope_hash(request)

    module = torch.nn.Module()
    module.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    )
    module.register_parameter(
        "bias", torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    )
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    registry = ParameterRegistry.from_model(module, optimizer)
    pre = TensorMap(
        {
            "weight": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "bias": torch.tensor([0.0], dtype=torch.float64),
        },
        registry=registry,
    )
    post = TensorMap(
        {
            "weight": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "bias": torch.tensor([1.0], dtype=torch.float64),
        },
        registry=registry,
    )
    bundle_root = store.root / "tensor-bundles"
    pre_path = bundle_root / "path-pre"
    post_path = bundle_root / "path-post"
    commit_path = bundle_root / "attempt-commit"
    pre_bundle = _publish_or_load_bundle(pre_path, {"parameters": pre.to_dict(clone=True)})
    post_bundle = _publish_or_load_bundle(post_path, {"parameters": post.to_dict(clone=True)})
    commit_bundle = _publish_or_load_bundle(
        commit_path,
        {
            "parameters": post.to_dict(clone=True),
            "control": {"scheduler_step": 1, "rng_epoch": 1, "cursor": 1},
        },
    )
    buffer_hash = _sha(["fixture", "buffers", request.config.config_hash])
    optimizer_pre = _sha(["fixture", "optimizer", "pre"])
    optimizer_post = _sha(["fixture", "optimizer", "post"])
    pre_state = _endpoint_state(
        "fixture-pre",
        pre_bundle.manifest_sha256,
        parameter_hash=_vector_digest(pre),
        buffer_hash=buffer_hash,
        optimizer_hash=optimizer_pre,
        control_tag="pre",
    )
    post_state = _endpoint_state(
        "fixture-parameter-post",
        post_bundle.manifest_sha256,
        parameter_hash=_vector_digest(post),
        buffer_hash=buffer_hash,
        optimizer_hash=optimizer_post,
        control_tag="pre",
    )
    attempt_state = _endpoint_state(
        "fixture-attempt-commit",
        commit_bundle.manifest_sha256,
        parameter_hash=_vector_digest(post),
        buffer_hash=buffer_hash,
        optimizer_hash=optimizer_post,
        control_tag="committed",
    )
    execution = FormalExecutionEvidence(
        "local_fixture",
        metadata={
            "fixture_path_scope_hash": path_scope_hash,
            "evidence_role": "stage3_shared_analytic_path",
        },
    )
    captured = EndpointCaptureCoordinator().capture(
        EndpointCaptureRequest(
            path_state_id=f"path-state-{path_scope_hash[:16]}",
            source_run_id=f"source-{path_scope_hash[:16]}",
            optimizer_step=1,
            parameter_registry_hash=registry.coordinate_registry_hash,
            update_sample_ids=("update-0", "update-1"),
            execution=execution,
            metadata={"fixture": True},
        ),
        _FixtureEndpointAdapter(
            pre_state,
            post_state,
            attempt_state,
            buffer_hash,
            _vector_digest({name: post[name] - pre[name] for name in pre}),
        ),
    )
    loss_hash = canonical_json_hash(request.config.base_config.section("loss"))
    probes = (
        ProbePanelEntry(
            "pilot",
            ProbeSpec("probe-pilot", ("probe-0",), _sha("probe-0"), loss_hash),
        ),
        ProbePanelEntry(
            "formal",
            ProbeSpec("probe-formal", ("probe-1",), _sha("probe-1"), loss_hash),
        ),
        ProbePanelEntry(
            "replay",
            ProbeSpec("probe-replay", ("probe-2",), _sha("probe-2"), loss_hash),
        ),
    )
    panel = ProbePanel.build(
        panel_id=f"probe-panel-{path_scope_hash[:16]}",
        endpoint=captured.record,
        entries=probes,
        execution=execution,
    )
    path = PathSpec(
        pre,
        post,
        path_id=f"fixture-path-{path_scope_hash[:16]}",
        probe_id=probes[0].probe.probe_id,
        loss_id=f"fixture-loss-{loss_hash[:16]}",
        accumulation_dtype=torch.float64,
    )
    target = TensorMap(
        {
            "weight": torch.tensor([2.0, 4.0], dtype=torch.float64),
            "bias": torch.tensor([3.0], dtype=torch.float64),
        },
        registry=registry,
    )

    def fixture_gradient(_alpha: float, state: TensorMap) -> TensorMap:
        return state.to(dtype=torch.float64) - target

    # 旧 fixture 链跨 task 共享同一个解析路径单元；不能让每个 task 自己的
    # config/output hash 通过模拟 endpoint 状态渗入 pilot/matrix recommendation。
    unit_id = "fixture-path-unit"
    node_cache, node_cache_root_ref = _stage3_node_cache(
        request,
        root,
        store,
        unit_id=unit_id,
        registry=registry,
    )
    state_digest = canonical_json_hash(
        {
            "fixture_path_identity_hash": path.identity_hash,
            "registry_hash": registry.coordinate_registry_hash,
        }
    )
    return _PathContext(
        registry=registry,
        path=path,
        endpoint=captured,
        panel=panel,
        primary_probe=probes[0].probe,
        pre_bundle_path=pre_path,
        pre_bundle=pre_bundle,
        post_bundle_path=post_path,
        post_bundle=post_bundle,
        execution=execution,
        gradient_fn=fixture_gradient,
        loss_fn=_fixture_loss,
        unit_id=unit_id,
        state_controller=_PathStateController(
            digest_fn=lambda: state_digest,
            restore_fn=lambda: None,
        ),
        node_cache=node_cache,
        node_cache_root_ref=node_cache_root_ref,
    )


@dataclass(frozen=True, slots=True)
class _LoadedTrainingEndpoint:
    """从训练 endpoint 权威 commit 复核出的只读对象。"""

    record: object
    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _endpoint_state_from_wire(value: object) -> EndpointState:
    if not isinstance(value, Mapping):
        raise TypeError("STAGE3_ENDPOINT_STATE_NOT_OBJECT")
    expected = {
        "artifact_id", "artifact_hash", "parameter_hash", "buffer_hash",
        "optimizer_hash", "scheduler_hash", "scaler_hash", "rng_hash",
        "data_cursor_hash", "model_mode_hash",
    }
    if set(value) != expected:
        raise ValueError("STAGE3_ENDPOINT_STATE_FIELDS_MISMATCH")
    return EndpointState(**dict(value))  # type: ignore[arg-type]


def _endpoint_record_from_wire(value: object):
    """重建 EndpointRecord，并复算 digest，拒绝只改 JSON 字段的伪端点。"""

    from .stage3 import EndpointRecord

    if not isinstance(value, Mapping):
        raise TypeError("STAGE3_ENDPOINT_RECORD_NOT_OBJECT")
    expected = {
        "path_state_id", "source_run_id", "optimizer_step",
        "parameter_registry_hash", "pre_state", "parameter_post_state",
        "attempt_commit_state", "attempt_commit_parent_hash",
        "probe_buffer_snapshot_hash", "full_update_delta_hash",
        "update_sample_ids", "replay_verified", "metadata", "endpoint_digest",
    }
    if set(value) != expected:
        raise ValueError("STAGE3_ENDPOINT_RECORD_FIELDS_MISMATCH")
    sample_ids = value["update_sample_ids"]
    metadata = value["metadata"]
    if not isinstance(sample_ids, list) or not isinstance(metadata, Mapping):
        raise TypeError("STAGE3_ENDPOINT_RECORD_ARRAY_OR_METADATA_INVALID")
    record = EndpointRecord(
        path_state_id=value["path_state_id"],  # type: ignore[arg-type]
        source_run_id=value["source_run_id"],  # type: ignore[arg-type]
        optimizer_step=value["optimizer_step"],  # type: ignore[arg-type]
        parameter_registry_hash=value["parameter_registry_hash"],  # type: ignore[arg-type]
        pre_state=_endpoint_state_from_wire(value["pre_state"]),
        parameter_post_state=_endpoint_state_from_wire(value["parameter_post_state"]),
        attempt_commit_state=_endpoint_state_from_wire(value["attempt_commit_state"]),
        attempt_commit_parent_hash=value["attempt_commit_parent_hash"],  # type: ignore[arg-type]
        probe_buffer_snapshot_hash=value["probe_buffer_snapshot_hash"],  # type: ignore[arg-type]
        full_update_delta_hash=value["full_update_delta_hash"],  # type: ignore[arg-type]
        update_sample_ids=tuple(sample_ids),
        replay_verified=value["replay_verified"],  # type: ignore[arg-type]
        metadata=metadata,
    )
    if value["endpoint_digest"] != record.digest:
        raise ValueError("STAGE3_ENDPOINT_RECORD_DIGEST_MISMATCH")
    return record


def _load_formal_endpoint_and_probe_plan(
    request: TaskExecutionRequest,
    root: Path,
    evidence: FormalExecutionEvidence,
) -> tuple[
    _LoadedTrainingEndpoint,
    Mapping[str, object],
    str,
    Mapping[str, Mapping[str, str]],
]:
    """从 input refs 选择唯一匹配的 endpoint commit 与 probe plan。"""

    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    endpoint_commits: list[tuple[str, Mapping[str, object]]] = []
    probe_plans: list[tuple[str, Mapping[str, object]]] = []
    for raw_ref in orchestration["input_result_refs"]:
        reference = str(raw_ref)
        try:
            value = load_canonical_json(
                _workspace_path(root, reference, field="input_result_refs")
            )
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage3_endpoint_probe_assets",
                f"endpoint/probe 输入不可读：{reference}: {error}",
                evidence_refs=(reference,),
            ) from error
        if not isinstance(value, Mapping):
            continue
        if value.get("schema_version") == "endpoint-commit-v1":
            endpoint_commits.append((reference, value))
        elif value.get("schema_version") == "stage3-probe-plan-v1":
            probe_plans.append((reference, value))
    if not endpoint_commits:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "training_endpoint_commit",
            "formal Stage 3 缺少训练阶段发布的 endpoint-commit-v1",
            evidence_refs=tuple(str(item) for item in orchestration["input_result_refs"]),
        )
    if len(probe_plans) != 1:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage3_probe_plan",
            f"formal Stage 3 要求唯一 probe plan，当前数量={len(probe_plans)}",
            evidence_refs=tuple(str(item) for item in orchestration["input_result_refs"]),
        )
    probe_ref, probe_plan = probe_plans[0]
    declared_probe_hash = probe_plan.get("artifact_hash")
    probe_body = {
        key: item for key, item in probe_plan.items() if key != "artifact_hash"
    }
    if declared_probe_hash != canonical_json_hash(probe_body):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_probe_plan",
            "probe plan artifact_hash 不可复算",
            retryable=False,
            evidence_refs=(probe_ref,),
        )
    if (
        probe_plan.get("scope") != "formal"
        or probe_plan.get("formal_eligible") is not True
        or probe_plan.get("execution_evidence_hash") != evidence.artifact_hash
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_probe_plan_qualification",
            "probe plan 未绑定当前 formal execution evidence",
            retryable=False,
            evidence_refs=(probe_ref,),
        )
    matching = [
        item
        for item in endpoint_commits
        if item[1].get("endpoint_digest") == probe_plan.get("endpoint_digest")
    ]
    if len(matching) != 1:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "training_endpoint_commit",
            f"与 probe plan 匹配的 endpoint 数量必须为 1，当前={len(matching)}",
            evidence_refs=tuple(item[0] for item in endpoint_commits) + (probe_ref,),
        )
    endpoint_ref, commit = matching[0]
    declared_commit_hash = commit.get("artifact_hash")
    commit_body = {key: item for key, item in commit.items() if key != "artifact_hash"}
    if declared_commit_hash != canonical_json_hash(commit_body):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "training_endpoint_commit",
            "endpoint commit artifact_hash 不可复算",
            retryable=False,
            evidence_refs=(endpoint_ref,),
        )
    if (
        commit.get("scope") != "formal"
        or commit.get("formal_eligible") is not True
        or commit.get("qualification_evidence_hash") != evidence.artifact_hash
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "training_endpoint_qualification",
            "endpoint commit 未由当前 formal execution evidence 资格化",
            retryable=False,
            evidence_refs=(endpoint_ref,),
        )
    object_ref = commit.get("object_ref")
    if not isinstance(object_ref, str):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "training_endpoint_object",
            "endpoint commit 缺少对象引用",
            retryable=False,
            evidence_refs=(endpoint_ref,),
        )
    try:
        object_value = load_canonical_json(
            _workspace_path(root, object_ref, field="endpoint.object_ref")
        )
        if not isinstance(object_value, Mapping):
            raise TypeError("endpoint object 不是 object")
        if canonical_json_hash(object_value) != commit.get("object_sha256"):
            raise ValueError("endpoint object hash 与 commit 不一致")
        if object_value.get("artifact_hash") != canonical_json_hash(
            {key: item for key, item in object_value.items() if key != "artifact_hash"}
        ):
            raise ValueError("endpoint object artifact_hash 不可复算")
        record = _endpoint_record_from_wire(object_value.get("record"))
        if record.digest != commit.get("endpoint_digest"):
            raise ValueError("endpoint record 与 commit digest 不一致")
        raw_bundles = object_value.get("state_bundles")
        if not isinstance(raw_bundles, Mapping) or set(raw_bundles) != {
            "pre", "parameter_post", "attempt_commit"
        }:
            raise ValueError("endpoint state bundle 集不完整")
        bundle_refs: dict[str, Mapping[str, str]] = {}
        for phase, raw in raw_bundles.items():
            if not isinstance(raw, Mapping) or set(raw) != {"ref", "manifest_sha256"}:
                raise ValueError(f"endpoint state bundle 引用无效：{phase}")
            if not all(isinstance(raw[field], str) for field in raw):
                raise TypeError(f"endpoint state bundle 字段不是字符串：{phase}")
            bundle_refs[str(phase)] = {
                "ref": str(raw["ref"]),
                "manifest_sha256": str(raw["manifest_sha256"]),
            }
    except (FileNotFoundError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "training_endpoint_object",
            f"endpoint 对象或状态引用不可验证：{error}",
            retryable=False,
            evidence_refs=(endpoint_ref, object_ref),
        ) from error
    return (
        _LoadedTrainingEndpoint(record, object_value),
        probe_plan,
        probe_ref,
        bundle_refs,
    )


def _normalize_ddp_names(
    values: Mapping[str, object],
    expected_names: Sequence[str],
    *,
    field: str,
) -> dict[str, object]:
    """显式适配 DDP ``module.`` 前缀；其他名称漂移一律拒绝。"""

    if set(values) == set(expected_names):
        return {name: values[name] for name in expected_names}
    stripped: dict[str, object] = {}
    has_ddp_root = "module" in values
    for name, item in values.items():
        # named_modules() 同时包含 DDP wrapper 的 ``""`` 与被包模型的
        # ``"module"``。路径求值只需要后者；二者模式应相同，wrapper root 不
        # 属于离线 provider 的模块结构。
        if has_ddp_root and name == "":
            continue
        normalized = (
            "" if name == "module" else name.removeprefix("module.")
        )
        if normalized in stripped:
            raise ValueError(f"{field} 的 DDP 名称规范化发生碰撞")
        stripped[normalized] = item
    # DDP 的 wrapper root ``""`` 会被有意丢弃，所以不能再拿规范化后的长度与
    # 原 mapping 比较；最终名称集合完全相等已经同时覆盖缺项、额外项与碰撞。
    if set(stripped) != set(expected_names):
        raise ValueError(f"{field} 与 provider 坐标不一致")
    return {name: stripped[name] for name in expected_names}


def _formal_path_context(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> _PathContext:
    """构造真实 endpoint + probe + offline HF fixed-state 路径上下文。"""

    provider_context = _formal_provider(request, root)
    evidence = provider_context.evidence.require_for_stage(3)
    endpoint, probe_plan, probe_ref, bundle_refs = _load_formal_endpoint_and_probe_plan(
        request, root, evidence
    )
    record = endpoint.record
    if not isinstance(provider_context.provider, TorchFixedStateGradientProvider):
        raise _blocked(
            BlockerCode.CAPABILITY_UNAVAILABLE,
            "torch_fixed_state_path_provider",
            "formal Stage 3 路径执行要求 TorchFixedStateGradientProvider",
            retryable=False,
        )
    provider = provider_context.provider
    loaded_states: dict[str, Mapping[str, object]] = {}
    loaded_bundles: dict[str, tuple[Path, TensorBundle]] = {}
    for phase, binding in bundle_refs.items():
        try:
            path = _workspace_path(root, binding["ref"], field=f"endpoint.{phase}")
            state, bundle = load_tensor_bundle(path)
            if bundle.manifest_sha256 != binding["manifest_sha256"]:
                raise ValueError("manifest hash 与 endpoint object 不一致")
            expected_state = getattr(
                record,
                "pre_state" if phase == "pre" else (
                    "parameter_post_state" if phase == "parameter_post" else "attempt_commit_state"
                ),
            )
            if bundle.manifest_sha256 != expected_state.artifact_hash:
                raise ValueError("bundle 未绑定 EndpointState artifact_hash")
            if not isinstance(state, Mapping):
                raise TypeError("bundle root 不是 object")
            validate_endpoint_state_bundle(state, expected_state)
            loaded_states[phase] = state
            loaded_bundles[phase] = (path, bundle)
        except (FileNotFoundError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                f"endpoint_state_bundle:{phase}",
                f"端点状态 bundle 不可验证：{error}",
                retryable=False,
                evidence_refs=(binding["ref"],),
            ) from error

    base_optimizer = request.config.base_config.section("optimizer")
    optimizer_runtime = request.config.section("optimizer_runtime")
    assert isinstance(base_optimizer, dict) and isinstance(optimizer_runtime, dict)
    try:
        registry_optimizer = build_optimizer(
            provider.model_adapter.module.parameters(),
            base_optimizer,
            optimizer_runtime,
        )
        registry = ParameterRegistry.from_model(
            provider.model_adapter.module, registry_optimizer
        )
        pre_raw = loaded_states["pre"].get("parameters")
        post_raw = loaded_states["parameter_post"].get("parameters")
        buffers_raw = loaded_states["pre"].get("buffers")
        modes_raw = loaded_states["pre"].get("model_modes")
        if not all(
            isinstance(item, Mapping)
            for item in (pre_raw, post_raw, buffers_raw, modes_raw)
        ):
            raise TypeError("endpoint bundle 缺少 parameters/buffers/model_modes")
        pre_values = _normalize_ddp_names(
            pre_raw, registry.eligible_names, field="pre parameters"  # type: ignore[arg-type]
        )
        post_values = _normalize_ddp_names(
            post_raw, registry.eligible_names, field="post parameters"  # type: ignore[arg-type]
        )
        expected_buffers = tuple(
            name
            for name, _value in provider.model_adapter.module.named_buffers(
                remove_duplicate=True
            )
        )
        buffer_values = _normalize_ddp_names(
            buffers_raw, expected_buffers, field="buffers"  # type: ignore[arg-type]
        )
        expected_modes = tuple(
            name for name, _module in provider.model_adapter.module.named_modules()
        )
        mode_values = _normalize_ddp_names(
            modes_raw, expected_modes, field="model modes"  # type: ignore[arg-type]
        )
        if not all(isinstance(value, torch.Tensor) for value in pre_values.values()):
            raise TypeError("pre parameters 含非 tensor")
        if not all(isinstance(value, torch.Tensor) for value in post_values.values()):
            raise TypeError("post parameters 含非 tensor")
        if not all(isinstance(value, torch.Tensor) for value in buffer_values.values()):
            raise TypeError("buffers 含非 tensor")
        if not all(type(value) is bool for value in mode_values.values()):
            raise TypeError("model modes 含非 bool")
        pre = TensorMap(pre_values, registry=registry)  # type: ignore[arg-type]
        post = TensorMap(post_values, registry=registry)  # type: ignore[arg-type]
    except (TypeError, ValueError, RuntimeError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "endpoint_provider_coordinate_binding",
            f"endpoint 与离线 provider 坐标无法绑定：{error}",
            retryable=False,
            evidence_refs=tuple(binding["ref"] for binding in bundle_refs.values()),
        ) from error

    entries_raw = probe_plan.get("entries")
    if not isinstance(entries_raw, list):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_probe_plan",
            "probe plan entries 不是数组",
            retryable=False,
            evidence_refs=(probe_ref,),
        )
    try:
        entries: list[ProbePanelEntry] = []
        available = set(provider_context.sample_ids)
        for raw in entries_raw:
            if not isinstance(raw, Mapping) or set(raw) != {
                "role", "probe_id", "sample_ids", "content_hash",
                "loss_contract_hash", "effective_weight_unit", "metadata",
            }:
                raise ValueError("probe entry 字段集合无效")
            sample_ids = raw["sample_ids"]
            if not isinstance(sample_ids, list) or not set(sample_ids).issubset(available):
                raise ValueError("probe sample IDs 不属于冻结 provider universe")
            entries.append(
                ProbePanelEntry(
                    str(raw["role"]),
                    ProbeSpec(
                        str(raw["probe_id"]),
                        tuple(sample_ids),
                        str(raw["content_hash"]),
                        str(raw["loss_contract_hash"]),
                        str(raw["effective_weight_unit"]),
                        raw["metadata"],  # type: ignore[arg-type]
                    ),
                )
            )
        panel = ProbePanel.build(
            panel_id=str(probe_plan["panel_id"]),
            endpoint=record,
            entries=entries,
            execution=evidence,
            minimum_formal_probes=int(probe_plan["minimum_formal_probes"]),
        )
        qualification_gate = next(
            gate for gate in evidence.prerequisite_gates if gate.gate_id == "stage3.G3-1"
        )
        panel = panel.qualify(
            execution=evidence,
            gate=qualification_gate,
            artifact_ref=probe_ref,
        )
    except (StopIteration, TypeError, ValueError, FormalRunRejected) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_probe_panel_binding",
            f"probe plan 无法形成正式 panel：{error}",
            retryable=False,
            evidence_refs=(probe_ref,),
        ) from error
    primary = next((entry.probe for entry in panel.entries if entry.role == "formal"), None)
    if primary is None:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_formal_probe",
            "probe panel 缺少 formal role",
            retryable=False,
            evidence_refs=(probe_ref,),
        )
    draws = primary.sample_ids
    frozen_buffers = {
        name: value for name, value in buffer_values.items()  # type: ignore[assignment]
    }
    frozen_modes = {name: bool(value) for name, value in mode_values.items()}

    def gradient_at(_alpha: float, state: TensorMap) -> TensorMap:
        batch = provider.gradient_at_parameter_state(
            state.to_dict(clone=True),
            draws,
            buffers=frozen_buffers,  # type: ignore[arg-type]
            model_modes=frozen_modes,
        )
        return TensorMap(
            {
                name: batch.gradients[name].detach().cpu().to(torch.float64)
                for name in registry.eligible_names
            },
            registry=registry,
        )

    def loss_at(state: TensorMap) -> torch.Tensor:
        batch = provider.gradient_at_parameter_state(
            state.to_dict(clone=True),
            draws,
            buffers=frozen_buffers,  # type: ignore[arg-type]
            model_modes=frozen_modes,
        )
        if batch.loss is None:
            raise RuntimeError("FORMAL_PATH_PROVIDER_LOSS_MISSING")
        return torch.tensor(float(batch.loss), dtype=torch.float64)

    precision = request.config.base_config.section("precision")
    accumulation = (
        torch.float64
        if precision["path_accumulation_dtype"] == "float64"
        else torch.float32
    )
    path = PathSpec(
        pre,
        post,
        path_id=f"formal-path-{record.digest[:20]}",
        probe_id=primary.probe_id,
        loss_id=f"formal-loss-{primary.loss_contract_hash[:20]}",
        accumulation_dtype=accumulation,
    )
    unit_hash = canonical_json_hash(
        {
            "endpoint_digest": record.digest,
            "probe_digest": primary.digest,
            "path_identity_hash": path.identity_hash,
            "execution_evidence_hash": evidence.artifact_hash,
        }
    )
    unit_id = f"path-unit-{unit_hash[:20]}"
    node_cache, node_cache_root_ref = _stage3_node_cache(
        request,
        root,
        store,
        unit_id=unit_id,
        registry=registry,
    )
    return _PathContext(
        registry=registry,
        path=path,
        endpoint=endpoint,
        panel=panel,
        primary_probe=primary,
        pre_bundle_path=loaded_bundles["pre"][0],
        pre_bundle=loaded_bundles["pre"][1],
        post_bundle_path=loaded_bundles["parameter_post"][0],
        post_bundle=loaded_bundles["parameter_post"][1],
        execution=evidence,
        gradient_fn=gradient_at,
        loss_fn=loss_at,
        unit_id=unit_id,
        state_controller=_PathStateController(
            digest_fn=provider.state_digest,
            # ``gradient_at_parameter_state`` 已对参数、buffer、mode、RNG 和已有
            # gradient 做 finally 恢复；若它仍泄漏状态，ReadOnlyPathContext 会
            # fail-closed。这里不重复触碰 provider 的私有快照实现。
            restore_fn=lambda: None,
        ),
        node_cache=node_cache,
        node_cache_root_ref=node_cache_root_ref,
    )


def _fixture_loss(state: TensorMap) -> torch.Tensor:
    # L(theta)=1/2 ||theta-target||^2，其梯度沿线性路径是 alpha 的一次多项式。
    target = TensorMap(
        {
            "weight": torch.tensor([2.0, 4.0], dtype=torch.float64),
            "bias": torch.tensor([3.0], dtype=torch.float64),
        },
        registry=state.registry,
    )
    difference = state - target
    return 0.5 * sum(torch.square(value).sum() for value in difference.values())


def _path_result_payload(result: PathIntegralResult) -> dict[str, JSONValue]:
    return {
        "rule": result.rule.to_dict(),
        "path_identity_hash": result.path_identity_hash,
        "signed": {
            name: result.signed[name].detach().cpu().to(torch.float64).tolist()
            for name in result.signed
        },
        "positive": {
            name: result.positive[name].detach().cpu().to(torch.float64).tolist()
            for name in result.positive
        },
        "negative_mass": {
            name: result.negative_mass[name].detach().cpu().to(torch.float64).tolist()
            for name in result.negative_mass
        },
        "absolute": {
            name: result.absolute[name].detach().cpu().to(torch.float64).tolist()
            for name in result.absolute
        },
        "endpoint_loss_pre": result.endpoint_loss_pre,
        "endpoint_loss_post": result.endpoint_loss_post,
        "loss_drop": result.loss_drop,
        "completeness_absolute_residual": result.completeness_absolute_residual,
        "completeness_relative_residual": result.completeness_relative_residual,
        "completeness_l1_scaled_residual": result.completeness_l1_scaled_residual,
        "unique_gradient_evaluations": result.unique_gradient_evaluations,
    }


def _run_stage3_endpoint(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    metric_contract = inputs.payload("metric_contract")
    if metric_contract.get("undefined_policy") != (
        "defined_false_with_reason_no_epsilon"
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_metric_contract",
            "端点管线拒绝未冻结退化统计语义的合同",
            retryable=False,
            evidence_refs=inputs.references,
        )
    context = _fixture_path_context(request, root, store)
    path_artifact = context.path.to_artifact(
        parameter_pre_bundle_ref=context.pre_bundle_path.relative_to(root).as_posix(),
        parameter_pre_bundle_manifest_hash=context.pre_bundle.manifest_sha256,
        parameter_post_bundle_ref=context.post_bundle_path.relative_to(root).as_posix(),
        parameter_post_bundle_manifest_hash=context.post_bundle.manifest_sha256,
    )
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "path_spec": path_artifact,  # type: ignore[dict-item]
        "probe_manifest": context.panel.to_dict(),  # type: ignore[dict-item]
        "state_restoration_report": {
            "schema_version": "stage3-task-state-restoration-report-v1",
            "endpoint": context.endpoint.to_dict(),  # type: ignore[union-attr]
            "replay_verified": True,
            "parameter_post_is_attempt_commit": False,
            "failure_restore_boundary": "pre_state",
            "scope": request.config.run_intent,
            "formal_eligible": (
                request.config.run_intent == "formal"
                and context.execution.formal_eligible
                and context.panel.qualification.formal_eligible
            ),
            "execution_evidence_hash": context.execution.artifact_hash,
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, inputs.references


def _run_stage3_quadrature_validation(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """对规则注册表执行解析多项式积分和复合规则加密检查。"""

    inputs = _predecessor_context(request, root, store)
    path_spec = inputs.payload("path_spec")
    context = _fixture_path_context(request, root, store)
    if path_spec.get("path_identity_hash") != context.path.identity_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_path_identity",
            "前序 PathSpec 与当前无副作用 fixture 路径身份不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    rules: dict[str, object] = dict(default_quadrature_rules())
    rules.update(
        {
            "composite_left_4": composite_left_rule(4),
            "composite_right_4": composite_right_rule(4),
            "composite_midpoint_4": composite_midpoint_rule(4),
            "composite_trapezoid_4": composite_trapezoid_rule(4),
            "composite_simpson_4": composite_simpson_rule(4),
            "gauss_legendre_4": gauss_legendre_rule(4),
        }
    )
    rows: list[dict[str, object]] = []
    for name, raw_rule in sorted(rules.items()):
        rule = raw_rule  # 保留下面公式中的 rule 名称，避免依赖实现类私有字段。
        degree_errors: list[float] = []
        for degree in range(4):
            observed = integrate_scalar_function(
                lambda alpha, degree=degree: alpha**degree,
                rule,  # type: ignore[arg-type]
            )
            degree_errors.append(abs(observed - 1.0 / (degree + 1)))
        rows.append(
            {
                "rule_name": name,
                "rule_hash": getattr(rule, "artifact_hash"),
                "unique_nodes": getattr(rule, "unique_gradient_evaluations"),
                "polynomial_degree_0_to_3_max_abs_error": max(degree_errors),
                "finite": all(math.isfinite(value) for value in degree_errors),
            }
        )
    # 理论阶 fixture：同一光滑函数在 n/2n 网格上的误差必须实际计算并保存；这里不
    # 通过阈值把规则伪装成正式默认，只验证核心能稳定重放。
    refinement_rows: list[dict[str, object]] = []
    exact_exp = math.e - 1.0
    for family, factory in (
        ("left", composite_left_rule),
        ("right", composite_right_rule),
        ("midpoint", composite_midpoint_rule),
        ("trapezoid", composite_trapezoid_rule),
        ("simpson", composite_simpson_rule),
    ):
        coarse = abs(integrate_scalar_function(math.exp, factory(4)) - exact_exp)
        fine = abs(integrate_scalar_function(math.exp, factory(8)) - exact_exp)
        refinement_rows.append(
            {
                "family": family,
                "coarse_abs_error": coarse,
                "fine_abs_error": fine,
                "error_decreased": fine < coarse,
                "empirical_ratio": None if fine == 0 else coarse / fine,
            }
        )
    passed = all(bool(row["finite"]) for row in rows) and all(
        bool(row["error_decreased"]) for row in refinement_rows
    )
    return (
        {
            "quadrature_rules": {
                "schema_version": "stage3-task-quadrature-rules-v1",
                "rules": {
                    name: raw_rule.to_dict()  # type: ignore[union-attr]
                    for name, raw_rule in sorted(rules.items())
                },
                "registry_hash": canonical_json_hash(
                    {
                        name: raw_rule.to_dict()  # type: ignore[union-attr]
                        for name, raw_rule in sorted(rules.items())
                    }
                ),
            },
            "analytic_validation_report": {
                "schema_version": "stage3-task-analytic-validation-report-v1",
                "path_identity_hash": context.path.identity_hash,
                "polynomial_rows": rows,  # type: ignore[dict-item]
                "refinement_rows": refinement_rows,  # type: ignore[dict-item]
                "passed": passed,
                "formal_gate_status": "NOT_RUN",
                "local_validation_status": "PASS" if passed else "FAIL",
                "upstream_binding_hash": inputs.binding_hash,
            },
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _stage3_reference(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[_PathContext, object, tuple[object, ...]]:
    context = _fixture_path_context(request, root, store)
    levels = (
        ReferenceRuleLevel("gauss_legendre", 0, gauss_legendre_rule(2)),
        ReferenceRuleLevel("gauss_legendre", 1, gauss_legendre_rule(4)),
        ReferenceRuleLevel("composite_simpson", 0, composite_simpson_rule(2)),
        ReferenceRuleLevel("composite_simpson", 1, composite_simpson_rule(4)),
    )
    result = ReferenceRefinementRunner().run(
        unit_id=context.unit_id,
        levels=levels,
        evaluator=context.integrate,
        artifact_root=store.root / "resume" / "path-reference",
        tolerance=1e-12,
        required_consecutive=1,
        primary_family="gauss_legendre",
        execution=context.execution,
    )
    return context, result, tuple(level.rule for level in levels)


def _run_stage3_reference(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    validation = inputs.payload("analytic_validation_report")
    if validation.get("passed") is not True:
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            "stage3_analytic_quadrature_validation",
            "reference refinement 拒绝未通过解析 fixture 的规则实现",
            evidence_refs=inputs.references,
        )
    context, result, reference_rules = _stage3_reference(request, root, store)
    if validation.get("path_identity_hash") != context.path.identity_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_path_identity",
            "解析验证与 reference refinement 的路径身份不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    contribution_path = store.root / "tensor-bundles" / "path-reference"
    contribution_bundle = _publish_or_load_bundle(
        contribution_path,
        {"reference_contribution": result.reference_contribution},
    )
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "path_integral_reference": {
            "schema_version": "stage3-task-path-integral-reference-v1",
            "refinement": result.to_dict(),
            "contribution_bundle_ref": contribution_path.relative_to(root).as_posix(),
            "contribution_bundle_manifest_hash": contribution_bundle.manifest_sha256,
            "path_identity_hash": context.path.identity_hash,
            "node_gradient_cache": context.cache_evidence(reference_rules),
            "scope": context.execution.run_intent,
            "formal_eligible": False,
            "execution_evidence_hash": context.execution.artifact_hash,
        },
        "precision_budget": {
            "schema_version": "stage3-task-precision-budget-v1",
            "gradient_dtype": "float64",
            "quadrature_weight_dtype": "float64",
            "path_accumulation_dtype": "float64",
            "two_independent_rule_families": True,
            "continuous_refinement": True,
            "selected_level": result.selected_level,
            "conservative_error": result.conservative_error,
            "formal_threshold_status": "UNFROZEN",
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, inputs.references


def _quadrature_rule_from_name(name: str) -> object:
    """由冻结名称重建规则，不允许调用方传任意代码或 pickle 对象。"""

    defaults = default_quadrature_rules()
    if name in defaults:
        return defaults[name]
    composite_factories = {
        "composite_left": composite_left_rule,
        "composite_right": composite_right_rule,
        "composite_midpoint": composite_midpoint_rule,
        "composite_trapezoid": composite_trapezoid_rule,
        "composite_simpson": composite_simpson_rule,
    }
    for prefix, factory in composite_factories.items():
        marker = f"{prefix}_"
        if name.startswith(marker):
            raw = name[len(marker) :]
            if not raw.isdecimal() or int(raw) <= 0:
                break
            return factory(int(raw))
    marker = "gauss_legendre_"
    if name.startswith(marker):
        raw = name[len(marker) :]
        if raw.isdecimal() and int(raw) > 0:
            return gauss_legendre_rule(int(raw))
    raise ValueError(f"STAGE3_QUADRATURE_RULE_NAME_UNSUPPORTED:{name}")


def _load_path_reference_vector(
    reference: Mapping[str, object],
    root: Path,
) -> Mapping[str, object]:
    """沿 reference artifact 的安全 bundle 读取独立贡献向量。"""

    bundle_ref = reference.get("contribution_bundle_ref")
    expected_hash = reference.get("contribution_bundle_manifest_hash")
    if not isinstance(bundle_ref, str) or not isinstance(expected_hash, str):
        raise ValueError("STAGE3_REFERENCE_BUNDLE_BINDING_MISSING")
    state, bundle = load_tensor_bundle(
        _workspace_path(root, bundle_ref, field="path_reference_bundle")
    )
    if bundle.manifest_sha256 != expected_hash or not isinstance(state, Mapping):
        raise ValueError("STAGE3_REFERENCE_BUNDLE_HASH_OR_ROOT_INVALID")
    contribution = state.get("reference_contribution")
    if not isinstance(contribution, Mapping) or not contribution:
        raise ValueError("STAGE3_REFERENCE_CONTRIBUTION_MISSING")
    return contribution


def _formal_stage3_pilot_plan(
    request: TaskExecutionRequest,
    root: Path,
    context: _PathContext,
) -> tuple[QuadratureThresholds, tuple[str, ...], tuple[str, ...], str]:
    """严格加载预注册的 formal 候选规则、阈值与路径单元集合。"""

    value, reference = _formal_input_document(
        request,
        root,
        schema_version="stage3-formal-pilot-plan-v1",
        requirement="formal_stage3_pilot_plan",
    )
    expected = {
        "schema_version",
        "plan_id",
        "scope",
        "candidate_rules",
        "required_unit_ids",
        "thresholds",
        "execution_evidence_hash",
        "formal_eligible",
        "artifact_hash",
    }
    if set(value) != expected:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage3_pilot_plan",
            "formal pilot plan 字段集合不匹配",
            retryable=False,
            evidence_refs=(reference,),
        )
    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if declared != canonical_json_hash(body):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage3_pilot_plan",
            "formal pilot plan artifact_hash 不可复算",
            retryable=False,
            evidence_refs=(reference,),
        )
    candidates = value.get("candidate_rules")
    units = value.get("required_unit_ids")
    thresholds_raw = value.get("thresholds")
    try:
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(isinstance(item, str) for item in candidates)
            or len(set(candidates)) != len(candidates)
        ):
            raise ValueError("candidate_rules 必须为非空无重复字符串数组")
        if (
            not isinstance(units, list)
            or not units
            or not all(isinstance(item, str) for item in units)
            or len(set(units)) != len(units)
        ):
            raise ValueError("required_unit_ids 必须为非空无重复字符串数组")
        if not isinstance(thresholds_raw, Mapping) or set(thresholds_raw) != {
            "max_normalized_l1_error",
            "max_completeness_absolute_residual",
            "min_spearman",
            "min_topk_overlap",
            "max_unique_nodes",
        }:
            raise ValueError("thresholds 字段集合无效")
        thresholds = QuadratureThresholds(**dict(thresholds_raw))  # type: ignore[arg-type]
        for name in candidates:
            _quadrature_rule_from_name(name)
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage3_pilot_plan",
            f"formal pilot plan 无法严格加载：{error}",
            retryable=False,
            evidence_refs=(reference,),
        ) from error
    if (
        value.get("scope") != "formal"
        or value.get("formal_eligible") is not True
        or value.get("execution_evidence_hash") != context.execution.artifact_hash
        or context.unit_id not in units
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "formal_stage3_pilot_plan_qualification",
            "formal pilot plan 未绑定当前 execution evidence/path unit",
            retryable=False,
            evidence_refs=(reference,),
        )
    return thresholds, tuple(candidates), tuple(units), reference


def _quadrature_observations(
    context: _PathContext,
    *,
    candidate_rule_names: Sequence[str] = ("midpoint", "trapezoid", "simpson"),
    reference_contribution: Mapping[str, object] | None = None,
) -> tuple[tuple[QuadratureObservation, ...], Mapping[str, PathIntegralResult]]:
    if reference_contribution is None:
        reference_contribution = context.integrate(gauss_legendre_rule(4)).signed
    reference_vector = _flatten(reference_contribution)
    reference_l1 = float(np.abs(reference_vector).sum())
    if reference_l1 <= 0:
        raise ValueError("STAGE3_REFERENCE_ZERO_L1")
    results: dict[str, PathIntegralResult] = {}
    observations: list[QuadratureObservation] = []
    for rule_name in candidate_rule_names:
        rule = _quadrature_rule_from_name(rule_name)
        # 执行真实积分；计时只用于运行期诊断，不进入 canonical artifact。
        started = perf_counter()
        result = context.integrate(rule)
        _elapsed = perf_counter() - started
        vector = _flatten(result.signed)
        residual = result.completeness_absolute_residual
        if residual is None:
            raise ValueError("STAGE3_COMPLETENESS_RESIDUAL_UNDEFINED")
        results[rule_name] = result
        observations.append(
            QuadratureObservation(
                unit_id=context.unit_id,
                rule_name=rule_name,
                unique_nodes=result.unique_gradient_evaluations,
                normalized_l1_error=float(np.abs(vector - reference_vector).sum() / reference_l1),
                completeness_absolute_residual=float(residual),
                spearman=_spearman(vector, reference_vector),
                topk_overlap=_topk_overlap(vector, reference_vector),
                # 推荐器在 node count 相同后才比较本字段；fixture 使用确定性的节点成本单位，
                # payload 会明确声明它不是墙钟秒数。
                wall_seconds=float(result.unique_gradient_evaluations),
            )
        )
    return tuple(observations), results


def _stage3_recommendation(
    context: _PathContext,
    *,
    thresholds: QuadratureThresholds | None = None,
    candidate_rule_names: Sequence[str] = ("midpoint", "trapezoid", "simpson"),
    reference_contribution: Mapping[str, object] | None = None,
    required_unit_ids: Sequence[str] | None = None,
) -> tuple[
    QuadratureRecommendation,
    tuple[QuadratureObservation, ...],
    Mapping[str, PathIntegralResult],
    QuadratureThresholds,
]:
    observations, results = _quadrature_observations(
        context,
        candidate_rule_names=candidate_rule_names,
        reference_contribution=reference_contribution,
    )
    if thresholds is None:
        if context.execution.run_intent != "local_fixture":
            raise RuntimeError("FORMAL_STAGE3_THRESHOLDS_MUST_BE_DECLARATIVE")
        thresholds = QuadratureThresholds(
            max_normalized_l1_error=1e-12,
            max_completeness_absolute_residual=1e-12,
            min_spearman=0.999999,
            min_topk_overlap=1.0,
            max_unique_nodes=5,
        )
    recommendation = QuadratureRecommendationEngine().recommend(
        recommendation_id=f"quadrature-{context.path.identity_hash[:16]}",
        observations=observations,
        required_unit_ids=(
            (context.unit_id,) if required_unit_ids is None else required_unit_ids
        ),
        thresholds=thresholds,
        execution=context.execution,
    )
    return recommendation, observations, results, thresholds


def _observation_payload(item: QuadratureObservation) -> dict[str, JSONValue]:
    return {
        "unit_id": item.unit_id,
        "rule_name": item.rule_name,
        "unique_nodes": item.unique_nodes,
        "normalized_l1_error": item.normalized_l1_error,
        "completeness_absolute_residual": item.completeness_absolute_residual,
        "spearman": item.spearman,
        "topk_overlap": item.topk_overlap,
        "deterministic_node_cost_units": item.wall_seconds,
    }


def _run_stage3_pilot(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    reference = inputs.payload("path_integral_reference")
    refinement = reference.get("refinement")
    if not isinstance(refinement, Mapping) or refinement.get("converged") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage3_converged_reference",
            "quadrature pilot 必须消费已收敛的跨规则 reference",
            evidence_refs=inputs.references,
        )
    context = _fixture_path_context(request, root, store)
    if reference.get("path_identity_hash") != context.path.identity_hash:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_path_identity",
            "reference artifact 与 pilot 路径身份不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    plan_ref: str | None = None
    reference_contribution = _load_path_reference_vector(reference, root)
    if request.config.run_intent == "formal":
        thresholds, candidate_rules, required_units, plan_ref = (
            _formal_stage3_pilot_plan(request, root, context)
        )
    else:
        thresholds = None
        candidate_rules = ("midpoint", "trapezoid", "simpson")
        required_units = (context.unit_id,)
    recommendation, observations, _results, thresholds = _stage3_recommendation(
        context,
        thresholds=thresholds,
        candidate_rule_names=candidate_rules,
        reference_contribution=reference_contribution,
        required_unit_ids=required_units,
    )
    node_cache_evidence = context.cache_evidence(
        tuple(_quadrature_rule_from_name(name) for name in candidate_rules)
    )
    scope = context.execution.run_intent
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "quadrature_pilot_report": {
            "schema_version": "stage3-task-quadrature-pilot-report-v1",
            "observations": [_observation_payload(item) for item in observations],
            "cost_semantics": "deterministic_unique_node_units_not_wall_clock",
            "recommendation": recommendation.to_dict(),
            "path_identity_hash": context.path.identity_hash,
            "reference_artifact_hash": canonical_json_hash(reference),
            "reference_bundle_ref": reference.get("contribution_bundle_ref"),
            "reference_bundle_manifest_hash": reference.get(
                "contribution_bundle_manifest_hash"
            ),
            "upstream_binding_hash": inputs.binding_hash,
            "node_gradient_cache": node_cache_evidence,
            "scope": scope,
            "formal_eligible": False,
            "pilot_plan_ref": plan_ref,
        },
        "threshold_freeze": {
            "schema_version": "stage3-task-threshold-freeze-v1",
            "thresholds": thresholds.to_dict(),
            "thresholds_hash": thresholds.artifact_hash,
            "scope": scope,
            "formal_eligible": False,
            "formal_freeze_status": (
                "PENDING_GATE_REVIEW" if scope == "formal" else "UNFROZEN"
            ),
            "execution_evidence_hash": context.execution.artifact_hash,
            "pilot_plan_ref": plan_ref,
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, inputs.references


def _run_stage3_matrix(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    inputs = _predecessor_context(request, root, store)
    pilot = inputs.payload("quadrature_pilot_report")
    threshold_freeze = inputs.payload("threshold_freeze")
    upstream_recommendation = pilot.get("recommendation")
    if not isinstance(upstream_recommendation, Mapping):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_quadrature_recommendation",
            "正式矩阵 fixture 缺少前序 quadrature recommendation",
            evidence_refs=inputs.references,
        )
    context = _fixture_path_context(request, root, store)
    if threshold_freeze.get("thresholds_hash") != upstream_recommendation.get(
        "thresholds_hash"
    ):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_quadrature_recommendation",
            "前序 recommendation 与 threshold freeze 不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    try:
        recommendation = QuadratureRecommendation.from_mapping(
            upstream_recommendation
        )
    except (TypeError, ValueError, FormalRunRejected) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_quadrature_recommendation",
            f"前序 recommendation 无法严格加载：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    if request.config.run_intent == "formal":
        try:
            gate = next(
                item
                for item in context.execution.prerequisite_gates
                if item.gate_id == "stage3.G3-5"
            )
            recommendation = recommendation.qualify(
                execution=context.execution,
                gate=gate,
            )
        except (StopIteration, FormalRunRejected) as error:
            raise _blocked(
                BlockerCode.GATE_NOT_READY,
                "stage3.G3-5",
                f"正式矩阵无法资格化 pilot recommendation：{error}",
                evidence_refs=inputs.references,
            ) from error
    observations_raw = pilot.get("observations")
    if not isinstance(observations_raw, list) or not observations_raw:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_quadrature_observations",
            "pilot report 缺少候选规则观测",
            retryable=False,
            evidence_refs=inputs.references,
        )
    candidate_names = tuple(
        dict.fromkeys(
            str(item["rule_name"])
            for item in observations_raw
            if isinstance(item, Mapping) and isinstance(item.get("rule_name"), str)
        )
    )
    if len(candidate_names) != len(observations_raw):
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_quadrature_observations",
            "pilot 候选规则缺项或重复",
            retryable=False,
            evidence_refs=inputs.references,
        )
    reference_binding = {
        "contribution_bundle_ref": pilot.get("reference_bundle_ref"),
        "contribution_bundle_manifest_hash": pilot.get(
            "reference_bundle_manifest_hash"
        ),
    }
    reference_contribution = _load_path_reference_vector(reference_binding, root)
    observations, results = _quadrature_observations(
        context,
        candidate_rule_names=candidate_names,
        reference_contribution=reference_contribution,
    )
    node_cache_evidence = context.cache_evidence(
        tuple(_quadrature_rule_from_name(name) for name in candidate_names)
    )
    selected = recommendation.default_rule
    if selected is None:
        raise RuntimeError("STAGE3_RECOMMENDATION_BLOCKED")
    if selected not in results:
        raise ValueError("STAGE3_SELECTED_RULE_NOT_IN_FROZEN_CANDIDATES")
    result = results[selected]
    # 结果表中的 independent reference 直接来自 Stage 3.05 跨规则连续加密产物，
    # 不允许候选方法重新以自身充当零误差 reference。
    independent_reference = {
        "rule": {"name": "stage3.05_cross_family_refinement"},
        "path_identity_hash": context.path.identity_hash,
        "signed": {
            name: (
                value.detach().cpu().to(torch.float64).tolist()
                if isinstance(value, torch.Tensor)
                else np.asarray(value, dtype=np.float64).tolist()
            )
            for name, value in reference_contribution.items()
        },
        "positive": {},
        "negative_mass": {},
        "absolute": {},
        "endpoint_loss_pre": None,
        "endpoint_loss_post": None,
        "loss_drop": None,
        "completeness_absolute_residual": None,
        "completeness_relative_residual": None,
        "completeness_l1_scaled_residual": None,
        "node_losses": [],
        "unique_gradient_evaluations": None,
        "reference_artifact_hash": pilot.get("reference_artifact_hash"),
    }
    scope = context.execution.run_intent
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "formal_path_results": {
            "schema_version": "stage3-task-path-results-v1",
            "scope": scope,
            "formal_eligible": (
                scope == "formal" and recommendation.qualification.formal_eligible
            ),
            "selected_rule": selected,
            "quadrature_recommendation": recommendation.to_dict(),
            "result": _path_result_payload(result),
            "candidate_results": {
                name: _path_result_payload(candidate)
                for name, candidate in sorted(results.items())
            },
            "node_gradient_cache": node_cache_evidence,
            "independent_reference": independent_reference,
            "upstream_binding_hash": inputs.binding_hash,
        },
        "completeness_report": {
            "schema_version": "stage3-task-completeness-report-v1",
            "path_identity_hash": context.path.identity_hash,
            "candidate_count": len(observations),
            "selected_rule": selected,
            "absolute_residual": result.completeness_absolute_residual,
            "relative_residual": result.completeness_relative_residual,
            "l1_scaled_residual": result.completeness_l1_scaled_residual,
            "defined": result.completeness_absolute_residual is not None,
            "node_gradient_cache_evidence_hash": node_cache_evidence[
                "evidence_hash"
            ],
        },
        "gate_record": _gate_candidate(request),
    }
    return payloads, inputs.references


def _path_wire_vector(
    path_result: Mapping[str, object],
) -> tuple[np.ndarray, tuple[str, ...]]:
    signed = path_result.get("signed")
    if not isinstance(signed, Mapping) or not signed:
        raise ValueError("STAGE3_PATH_RESULT_SIGNED_MISSING")
    arrays: list[np.ndarray] = []
    coordinate_ids: list[str] = []
    for name, values in sorted(signed.items()):
        if not isinstance(name, str):
            raise TypeError("STAGE3_PATH_PARAMETER_NAME_NOT_STRING")
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError(f"STAGE3_PATH_VECTOR_INVALID:{name}")
        arrays.append(array)
        coordinate_ids.extend(f"{name}[{index}]" for index in range(array.size))
    return np.concatenate(arrays), tuple(coordinate_ids)


def _run_stage3_statistics(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """用独立 reference 计算逐规则误差、相关性、top-k 与完备性表。"""

    inputs = _predecessor_context(request, root, store)
    formal_results = inputs.payload("formal_path_results")
    completeness = inputs.payload("completeness_report")
    candidates = formal_results.get("candidate_results")
    reference_wire = formal_results.get("independent_reference")
    if not isinstance(candidates, Mapping) or not isinstance(reference_wire, Mapping):
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage3_candidate_and_reference_results",
            "误差分析缺少候选结果或独立 reference",
            retryable=False,
            evidence_refs=inputs.references,
        )
    try:
        reference, coordinate_ids = _path_wire_vector(reference_wire)
        rows: list[dict[str, object]] = []
        for rule_name, raw in sorted(candidates.items()):
            if not isinstance(rule_name, str) or not isinstance(raw, Mapping):
                raise TypeError("STAGE3_CANDIDATE_RESULT_INVALID")
            candidate, observed_ids = _path_wire_vector(raw)
            if observed_ids != coordinate_ids:
                raise ValueError("STAGE3_CANDIDATE_COORDINATE_ID_DRIFT")
            denominator = float(np.abs(reference).sum())
            normalized_l1 = (
                None
                if denominator == 0
                else float(np.abs(candidate - reference).sum() / denominator)
            )
            rule = raw.get("rule")
            if not isinstance(rule, Mapping):
                raise TypeError("STAGE3_CANDIDATE_RULE_MISSING")
            pearson = analysis_pearson(candidate, reference)
            spearman = analysis_spearman(candidate, reference)
            overlap = analysis_top_k_overlap(
                np.abs(candidate),
                np.abs(reference),
                max(1, candidate.size // 2),
                canonical_ids=coordinate_ids,
            )
            rows.append(
                {
                    "rule_name": rule_name,
                    "rule_hash": rule.get("artifact_hash"),
                    "unique_nodes": raw.get("unique_gradient_evaluations"),
                    "normalized_l1_error": normalized_l1,
                    "mae": analysis_mae(candidate, reference).value,
                    "mse": analysis_mse(candidate, reference).value,
                    "pearson_defined": pearson.defined,
                    "pearson": pearson.value,
                    "pearson_reason": pearson.reason,
                    "spearman_defined": spearman.defined,
                    "spearman": spearman.value,
                    "spearman_reason": spearman.reason,
                    "topk_defined": overlap.defined,
                    "topk_overlap": overlap.value,
                    "topk_reason": overlap.reason,
                    "completeness_absolute_residual": raw.get(
                        "completeness_absolute_residual"
                    ),
                    "path_identity_hash": raw.get("path_identity_hash"),
                }
            )
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_path_result_schema",
            f"路径结果无法确定性统计：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    table = FrozenSourceTable.from_rows(
        name="stage3_path_error",
        schema_version="stage3-path-error-table-v1",
        rows=rows,
    )
    finite_rows = all(
        row["normalized_l1_error"] is not None
        and math.isfinite(float(row["normalized_l1_error"]))
        and row["completeness_absolute_residual"] is not None
        and math.isfinite(float(row["completeness_absolute_residual"]))
        for row in rows
    )
    selected = formal_results.get("selected_rule")
    selected_rows = [row for row in rows if row["rule_name"] == selected]
    if len(selected_rows) != 1 or completeness.get("defined") is not True:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "stage3_selected_rule_completeness",
            "选中规则缺少唯一误差行或完备性结果",
            evidence_refs=inputs.references,
        )
    stability: dict[str, JSONValue] = {
        "schema_version": "stage3-task-stability-report-v1",
        "selected_rule": selected,
        "selected_rule_metrics": selected_rows[0],  # type: ignore[dict-item]
        "all_rows_finite": finite_rows,
        "coordinate_count": len(coordinate_ids),
        "coordinate_ids_hash": canonical_json_hash(list(coordinate_ids)),
        "source_table_hash": table.content_hash,
        "formal_gate_status": "NOT_RUN",
        "local_validation_status": "PASS" if finite_rows else "FAIL",
        "upstream_binding_hash": inputs.binding_hash,
    }
    return (
        {
            "path_error_table": {
                "schema_version": "stage3-task-path-error-table-wrapper-v1",
                "source_table": table.to_dict(),
                "reference_path_identity_hash": reference_wire.get(
                    "path_identity_hash"
                ),
            },
            "stability_report": stability,
            "frozen_source_table": table.to_dict(),  # type: ignore[dict-item]
        },
        inputs.references,
    )


def _run_stage3_analysis(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """从冻结误差/成本表运行唯一的求积推荐器并发布 fixture decision。"""

    inputs = _predecessor_context(request, root, store)
    if request.config.run_intent == "formal":
        raise _blocked(
            BlockerCode.GATE_NOT_READY,
            "stage3.G3-7",
            "formal QuadratureDecision 需要独立 Gate；分析 runner 不得自签正式方法",
            evidence_refs=inputs.references,
        )
    try:
        source = FrozenSourceTable.from_mapping(inputs.payload("frozen_source_table"))
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_frozen_error_table",
            f"冻结路径误差表不可用：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    observations: list[QuadratureObservation] = []
    for raw in source.rows:
        if (
            raw["normalized_l1_error"] is None
            or raw["spearman"] is None
            or raw["topk_overlap"] is None
            or raw["completeness_absolute_residual"] is None
        ):
            continue
        observations.append(
            QuadratureObservation(
                unit_id="fixture-path-unit",
                rule_name=str(raw["rule_name"]),
                unique_nodes=int(raw["unique_nodes"]),
                normalized_l1_error=float(raw["normalized_l1_error"]),
                completeness_absolute_residual=float(
                    raw["completeness_absolute_residual"]
                ),
                spearman=float(raw["spearman"]),
                topk_overlap=float(raw["topk_overlap"]),
                wall_seconds=float(raw["unique_nodes"]),
            )
        )
    if not observations:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "defined_quadrature_observations",
            "所有候选规则统计量均未定义，无法选择方法",
            evidence_refs=inputs.references,
        )
    thresholds = QuadratureThresholds(
        max_normalized_l1_error=1e-12,
        max_completeness_absolute_residual=1e-12,
        min_spearman=0.999999,
        min_topk_overlap=1.0,
        max_unique_nodes=5,
    )
    execution = FormalExecutionEvidence(
        "local_fixture",
        metadata={"frozen_source_table_hash": source.content_hash},
    )
    recommendation = QuadratureRecommendationEngine().recommend(
        recommendation_id=f"stage3-analysis-{source.content_hash[:16]}",
        observations=tuple(observations),
        required_unit_ids=("fixture-path-unit",),
        thresholds=thresholds,
        execution=execution,
    )
    decision = build_fixture_quadrature_decision(
        passing_rules_by_cost=recommendation.passing_rules,
        fallback_rule=recommendation.fallback_rule,
    )
    ordered_rows = sorted(
        (dict(row) for row in source.rows),
        key=lambda row: (int(row["unique_nodes"]), str(row["rule_name"])),
    )
    cost_table = FrozenSourceTable.from_rows(
        name="stage3_cost_accuracy",
        schema_version="stage3-cost-accuracy-table-v1",
        rows=ordered_rows,
    )
    return (
        {
            "cost_accuracy_table": cost_table.to_dict(),  # type: ignore[dict-item]
            "quadrature_decision": decision.to_dict(),  # type: ignore[dict-item]
            "gate_record": _gate_candidate(request),
        },
        inputs.references,
    )


def _run_stage3_reporting(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """从 hash 绑定成本表重建 Stage 3 报告、图表与后续交接。"""

    inputs = _predecessor_context(request, root, store)
    try:
        table = FrozenSourceTable.from_mapping(inputs.payload("cost_accuracy_table"))
        from .stage3 import QuadratureDecision

        decision = QuadratureDecision.from_mapping(inputs.payload("quadrature_decision"))
    except (KeyError, TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage3_reporting_inputs",
            f"Stage 3 决策或成本表不可重放：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    errors = np.asarray(
        [float(row["normalized_l1_error"]) for row in table.rows], dtype=np.float64
    )
    nodes = np.asarray([float(row["unique_nodes"]) for row in table.rows], dtype=np.float64)
    builder = AnalysisReportBuilder(
        report_id=f"stage3-fixture-{table.content_hash[:16]}"
    )
    builder.add_source(table)
    builder.add_metric(
        "mean_normalized_l1_error",
        analysis_bias(errors, np.zeros_like(errors)),
        source=table,
        derivation_id="stage3.mean-normalized-l1-error.v1",
        input_columns=("normalized_l1_error",),
    )
    builder.add_metric(
        "nodes_error_pearson",
        analysis_pearson(nodes, errors),
        source=table,
        derivation_id="stage3.nodes-error-pearson.v1",
        input_columns=("unique_nodes", "normalized_l1_error"),
    )
    report = builder.build(
        metadata={
            "scope": request.config.run_intent,
            "formal_eligible": False,
            "quadrature_decision_hash": decision.artifact_hash,
        }
    )
    spec = ChartSpec.from_table(
        table,
        chart_id=f"stage3-cost-error-{table.content_hash[:12]}",
        chart_type="scatter",
        x_column="unique_nodes",
        y_columns=("normalized_l1_error",),
        sort_columns=("unique_nodes", "rule_name"),
    )
    chart = ChartArtifact.from_spec(spec)
    return (
        {
            "analysis_report": report.to_dict(),  # type: ignore[dict-item]
            "chart_artifacts": {
                "schema_version": "stage3-task-chart-artifacts-v1",
                "source_table_hash": table.content_hash,
                "artifacts": [chart.to_dict()],  # type: ignore[list-item]
                "manual_numeric_edits_allowed": False,
            },
            "handoff_manifest": {
                "schema_version": "stage3-task-handoff-manifest-v1",
                "quadrature_decision": decision.to_dict(),
                "source_table_hash": table.content_hash,
                "cost_semantics": "deterministic_unique_node_units_not_wall_clock",
                "report_hash": report.report_hash,
                "formal_stage_complete": False,
            },
            "gate_summary": {
                "schema_version": "stage3-task-gate-summary-v1",
                "stage3.G3-7": "NOT_RUN",
                "formal_exit_gate": "NOT_RUN",
                "local_validation_status": "PASS",
            },
        },
        inputs.references,
    )


@dataclass(slots=True)
class _Stage23Runner(TaskRunner):
    """同一 RunnerKind 下按 canonical task_id 分派的薄适配器。"""

    runner_kind: RunnerKind
    workspace_root: Path
    handlers: Mapping[
        str,
        Callable[
            [TaskExecutionRequest, Path, TaskArtifactStore],
            tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]],
        ],
    ]
    fallback: TaskRunner | None = None

    def __post_init__(self) -> None:
        self.workspace_root = self.workspace_root.resolve()

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        if request.task.runner_kind is not self.runner_kind:
            raise TaskRuntimeError("STAGE23_RUNNER_KIND_DISPATCH_MISMATCH")
        handler = self.handlers.get(request.task.task_id)
        if handler is None:
            if self.fallback is not None:
                return self.fallback.run(request)
            raise TaskRuntimeError(
                f"STAGE23_RUNNER_TASK_ID_UNSUPPORTED:{request.task.task_id}"
            )
        store = _artifact_store(request, self.workspace_root)
        # formal 先核对执行证据，确保即便输出目录中残留旧 commit，也不能在本次
        # environment 缺 Gate/freeze 时绕过资格检查。随后完整验证直接前驱。
        if request.config.run_intent == "formal":
            _formal_execution_evidence(request, self.workspace_root)
        _predecessor_context(request, self.workspace_root, store)
        completed = _completed_result(request, store)
        if completed is not None:
            return completed
        _authorize_partial_resume(
            request,
            self.workspace_root,
            store,
            _authoritative_partial_paths(request, self.workspace_root, store),
        )
        payloads, source_refs = handler(request, self.workspace_root, store)
        references = _publish_payloads(
            request,
            store,
            payloads,
            source_refs=source_refs,
        )
        metadata: dict[str, JSONValue] = {
            "execution_contract": "stage23-specialized-v1"
        }
        if request.task.task_id == _STAGE2_REFERENCE_TASK and request.config.run_intent == "formal":
            convergence = payloads.get("reference_convergence_report", {})
            if isinstance(convergence, Mapping):
                bindings = {
                    key: convergence.get(key)
                    for key in (
                        "stage2_reference_producer_commit",
                        "producer_provenance",
                        "config_identity",
                        "checkpoint_identity",
                        "registry_identity",
                        "model_identity",
                        "data_identity",
                        "tokenizer_identity",
                        "external_lineage",
                    )
                    if convergence.get(key) is not None
                }
                metadata["identity_bindings"] = bindings  # type: ignore[assignment]
        checkpoint_ref = None
        if request.config.run_intent == "formal":
            convergence = payloads.get("reference_convergence_report", {})
            lineage = convergence.get("external_lineage") if isinstance(convergence, Mapping) else None
            checkpoint = lineage.get("checkpoint_manifest") if isinstance(lineage, Mapping) else None
            if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("commit_ref"), str):
                checkpoint_ref = checkpoint["commit_ref"]
        return TaskRunResult.passed(
            request,
            artifact_refs=references,
            checkpoint_ref=checkpoint_ref,
            message="stage2/3 specialized task completed",
            metadata=metadata,
        )


def _fallback_map(
    fallbacks: Mapping[RunnerKind, TaskRunner] | Iterable[TaskRunner],
) -> dict[RunnerKind, TaskRunner]:
    if isinstance(fallbacks, Mapping):
        normalized = dict(fallbacks)
        for kind, runner in normalized.items():
            if not isinstance(kind, RunnerKind) or runner.runner_kind is not kind:
                raise TypeError("STAGE23_FALLBACK_MAPPING_KIND_MISMATCH")
        return normalized
    normalized: dict[RunnerKind, TaskRunner] = {}
    for runner in fallbacks:
        if not isinstance(getattr(runner, "runner_kind", None), RunnerKind) or not callable(
            getattr(runner, "run", None)
        ):
            raise TypeError("STAGE23_FALLBACK_NOT_TASK_RUNNER")
        if runner.runner_kind in normalized:
            raise ValueError(
                f"STAGE23_DUPLICATE_FALLBACK_KIND:{runner.runner_kind.value}"
            )
        normalized[runner.runner_kind] = runner
    return normalized


def build_stage23_runner_overrides(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | Iterable[TaskRunner] = (),
) -> Mapping[RunnerKind, TaskRunner]:
    """构造 Stage 2/3 专派 runner，并按 kind 组合已有通用 fallback。

    ``TaskRuntime`` 每个 ``RunnerKind`` 只能注册一次；而 CONTRACT/STATISTICS 等
    kind 同时服务其他 Stage。因此本工厂返回单层复合 runner：Stage 2/3 canonical
    task_id 走本模块的科学 handler，其余 task_id 原样转发给调用方提供的 fallback。
    未与 Stage 2/3 重叠的 fallback 也会保留在返回映射中，调用方可以直接
    ``runners.update(...)`` 完成组合。
    """

    root = Path(workspace_root).resolve()
    fallback_by_kind = _fallback_map(fallbacks)
    handlers_by_kind: dict[
        RunnerKind,
        dict[
            str,
            Callable[
                [TaskExecutionRequest, Path, TaskArtifactStore],
                tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]],
            ],
        ],
    ] = {
        RunnerKind.CONTRACT: {
            "stage2.01_scope_hypotheses_and_preregistration": _run_stage2_contract,
            "stage3.02_math_and_metric_contract": _run_stage3_contract,
        },
        RunnerKind.AUDIT: {
            "stage2.02_stage1_handoff_and_fixed_state_contract": _run_stage2_handoff_audit,
            "stage3.01_prerequisites_and_scope": _run_stage3_prerequisites,
        },
        RunnerKind.ASSET: {
            "stage2.03_assets_checkpoints_and_sampling": _run_stage2_assets_and_sampling,
        },
        RunnerKind.REFERENCE: {
            _STAGE2_REFERENCE_TASK: _run_stage2_reference,
            _STAGE3_REFERENCE_TASK: _run_stage3_reference,
        },
        RunnerKind.ESTIMATOR_EXPERIMENT: {
            task_id: _run_stage2_estimator for task_id in _STAGE2_ESTIMATOR_TASKS
        },
        RunnerKind.PILOT: {
            _STAGE2_PILOT_TASK: _run_stage2_pilot,
            _STAGE3_PILOT_TASK: _run_stage3_pilot,
        },
        RunnerKind.PATH_INTEGRATION: {
            _STAGE3_ENDPOINT_TASK: _run_stage3_endpoint,
            _STAGE3_MATRIX_TASK: _run_stage3_matrix,
        },
        RunnerKind.STATISTICS: {
            "stage2.08_statistics_and_robustness": _run_stage2_statistics,
            "stage3.08_error_analysis_and_stability": _run_stage3_statistics,
        },
        RunnerKind.CAPACITY: {
            "stage2.09_cost_and_system_validation": _run_stage2_capacity,
        },
        RunnerKind.REPORTING: {
            "stage2.10_visualization_reporting_and_decision": _run_stage2_reporting,
            "stage3.10_reports_visualizations_and_handoff": _run_stage3_reporting,
        },
        RunnerKind.DELIVERY: {
            "stage2.11_delivery_and_exit_gate": _run_stage2_delivery,
        },
        RunnerKind.VALIDATION: {
            "stage3.04_quadrature_engine_and_unit_tests": _run_stage3_quadrature_validation,
        },
        RunnerKind.ANALYSIS: {
            "stage3.09_cost_and_method_selection": _run_stage3_analysis,
        },
    }
    combined: dict[RunnerKind, TaskRunner] = {
        kind: _Stage23Runner(
            kind,
            root,
            handlers,
            fallback=fallback_by_kind.pop(kind, None),
        )
        for kind, handlers in handlers_by_kind.items()
    }
    combined.update(fallback_by_kind)
    return combined


def stage23_runners(workspace_root: str | Path) -> tuple[TaskRunner, ...]:
    """构造覆盖全部 Stage 2/3 canonical task_id 的专派 runner。"""

    return tuple(build_stage23_runner_overrides(workspace_root).values())


def register_stage23_runners(
    runtime: TaskRuntime,
    workspace_root: str | Path,
) -> TaskRuntime:
    """向空缺的全部 Stage 2/3 ``RunnerKind`` 注册专用实现。

    ``TaskRuntime`` 明确禁止静默覆盖，因此默认运行时工厂应在注册 generic runner
    之前调用本函数。若顺序错误，本函数保留原 runner 并立即失败。
    """

    if not isinstance(runtime, TaskRuntime):
        raise TypeError("runtime 必须是 TaskRuntime")
    runners = stage23_runners(workspace_root)
    occupied = {runner.runner_kind for runner in runners}.intersection(
        runtime.registered_kinds
    )
    if occupied:
        raise TaskRuntimeError(
            "STAGE23_RUNNER_KIND_ALREADY_REGISTERED:"
            + ",".join(sorted(kind.value for kind in occupied))
        )
    for runner in runners:
        runtime.register(runner)
    return runtime


__all__ = [
    "build_stage23_runner_overrides",
    "register_stage23_runners",
    "stage23_runners",
]
