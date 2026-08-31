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
from time import perf_counter

import numpy as np
import torch

from ..assets import AssetManifestError, resolve_ready_asset
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
)
from ..contracts.seed import SeedPlan
from ..contracts.stage23 import FormalExecutionEvidence
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
from ..providers import (
    FixedStateGradientProvider,
    OfflineHuggingFaceModelAdapter,
    PretokenizedGlueDatasetAdapter,
    PretokenizedPileDatasetAdapter,
    SyntheticGradientProvider,
    TorchFixedStateGradientProvider,
)
from ..runtime.task_artifacts import TaskArtifactStore
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
    MICROBATCH_SELECTION_ORDER,
    PrimaryPairDecision,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    STREAM_NAMES,
)
from .stage2 import PairedEstimatorRunner, build_fixture_estimator_decision
from .stage2_formal import (
    FormalExperimentPlan,
    PilotCellObservation,
    PilotThresholds,
    RecoverablePairedWaveRunner,
    ReferenceSizingPlan,
    Stage2RecommendationEngine,
    StreamingReferenceSizer,
)
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

# 线性链并不意味着未来不能增加 DAG 分支；这里冻结的是当前计划中每个任务必须
# 完整消费的直接前驱。值采用 tuple，后续扩展为多前驱时无需改变验证器协议。
_REQUIRED_PREDECESSORS: Mapping[str, tuple[str, ...]] = {
    task_id: (() if index == 0 else (_STAGE23_TASK_ORDER[index - 1],))
    for index, task_id in enumerate(_STAGE23_TASK_ORDER)
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

    def to_payload(self) -> dict[str, JSONValue]:
        return {
            "provider_kind": self.provider_kind,
            "fixed_state_id": self.provider.fixed_state_id,
            "provider_state_digest": self.provider.state_digest(),
            "registry_hash": self.provider.registry_hash,
            "parameter_names": list(self.provider.parameter_names),
            "sample_universe_size": len(self.sample_ids),
            "execution_evidence_hash": self.evidence.artifact_hash,
            "asset_manifest_hashes": list(self.asset_manifest_hashes),
            "weighting_assumptions": {
                "statistical_unit": self.provider.statistical_unit,
                "weight_unit": self.provider.weight_unit,
                "sampling_design": self.provider.sampling_design,
                "weights_exogenous": self.provider.weights_exogenous,
                "common_mean_assumption": self.provider.common_mean_assumption,
            },
        }


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

    # Stage2.01 是本链入口；它可以绑定额外预注册附件，但不能伪造前驱 task。
    return _PredecessorContext(expected_tasks, tuple(ordered), tuple(auxiliaries))


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
        observed = _document_hash(
            _workspace_path(root, freeze_ref, field="contract_freeze")
        )
    except (FileNotFoundError, ValueError) as error:
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


def _infer_glue_task(asset_id: str) -> str:
    normalized = asset_id.lower().replace("_", "-")
    matches = [name for name in ("sst-2", "mnli", "rte") if name in normalized]
    if len(matches) != 1:
        raise ValueError("FORMAL_GLUE_TASK_NOT_UNIQUE_IN_DATA_ASSET_ID")
    return matches[0]


def _formal_provider(request: TaskExecutionRequest, root: Path) -> _ProviderContext:
    """验证离线资产并构造真实 Torch fixed-state provider。

    此函数没有 synthetic fallback。HF 依赖仍由 provider 的延迟导入负责；本机未安装
    ``transformers``/``datasets`` 时会转换为结构化 blocker。
    """

    if request.config.run_intent != "formal":
        raise RuntimeError("FORMAL_PROVIDER_REQUIRES_FORMAL_INTENT")
    evidence, evidence_ref = _formal_execution_evidence(request, root)
    providers = request.config.section("providers")
    if not isinstance(providers, dict) or providers.get("kind") != "offline_hf":
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "offline_hf_fixed_state_provider",
            "formal Stage 2/3 禁止 tiny/synthetic provider",
            retryable=False,
            evidence_refs=(evidence_ref,),
        )

    manifest_fields = (
        ("model_manifest_ref", "model_root_ref"),
        ("data_manifest_ref", "data_root_ref"),
        ("tokenizer_manifest_ref", "tokenizer_root_ref"),
    )
    resolved_assets: list[object] = []
    manifest_hashes: list[str] = []
    manifest_refs: list[str] = []
    try:
        for manifest_field, root_field in manifest_fields:
            manifest_ref = str(providers[manifest_field])
            root_ref = str(providers[root_field])
            manifest_path = _workspace_path(root, manifest_ref, field=manifest_field)
            asset_root = _workspace_path(root, root_ref, field=root_field)
            manifest_hashes.append(_document_hash(manifest_path))
            manifest_refs.append(manifest_ref)
            resolved_assets.append(resolve_ready_asset(manifest_path, asset_root))
    except (FileNotFoundError, AssetManifestError, ValueError) as error:
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "offline_hf_assets",
            f"offline_hf 资产未就绪或校验失败：{type(error).__name__}: {error}",
            evidence_refs=tuple(manifest_refs),
        ) from error

    if not set(manifest_hashes).issubset(evidence.asset_manifest_hashes):
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "asset_manifest_hashes",
            "FormalExecutionEvidence 未覆盖配置所绑定的模型/数据/tokenizer manifest",
            retryable=False,
            evidence_refs=(evidence_ref, *manifest_refs),
        )

    model_asset, data_asset, _tokenizer_asset = resolved_assets
    base = request.config.base_config
    data = base.section("data")
    runtime = base.section("runtime")
    task_type = str(providers["task_type"])
    try:
        model = OfflineHuggingFaceModelAdapter.from_local_directory(
            getattr(model_asset, "root"),
            task_type=task_type,
            num_labels=providers["num_labels"],  # type: ignore[arg-type]
            torch_dtype=torch.float64,
        )
        model.module.to(torch.device(str(runtime["device"])))
        if task_type == "causal_lm":
            resolver = PretokenizedPileDatasetAdapter(
                getattr(data_asset, "root"),
                dataset_id=str(data["asset_id"]),
                split=str(data["split"]),
                microbatch_size=1,
                microbatches_per_step=1,
                sampling_design=str(data["sampling_design"]),
                weights_exogenous=bool(data["weights_exogenous"]),
                common_mean_assumption=bool(data["common_mean_assumption"]),
                allowed_root=getattr(data_asset, "root"),
            )
        elif task_type == "sequence_classification":
            resolver = PretokenizedGlueDatasetAdapter(
                getattr(data_asset, "root"),
                task_name=_infer_glue_task(str(data["asset_id"])),
                split=str(data["split"]),
                dataset_id=str(data["asset_id"]),
                microbatch_size=1,
                microbatches_per_step=1,
                allowed_root=getattr(data_asset, "root"),
            )
        else:
            raise ValueError("FORMAL_FIXED_STATE_TASK_TYPE_UNSUPPORTED")
        fixed_id = f"offline-{canonical_json_hash(manifest_hashes)[:24]}"
        provider = TorchFixedStateGradientProvider(
            model,
            resolver,
            fixed_state_id=fixed_id,
            output_dtype=torch.float64,
        )
    except DependencyUnavailable as error:
        raise _blocked(
            BlockerCode.DEPENDENCY_UNAVAILABLE,
            error.dependency,
            str(error),
            evidence_refs=tuple(manifest_refs),
        ) from error
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
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
    )


def _provider_context(request: TaskExecutionRequest, root: Path) -> _ProviderContext:
    return (
        _local_provider(request)
        if request.config.run_intent == "local_fixture"
        else _formal_provider(request, root)
    )


def _sampling_plan(request: TaskExecutionRequest, context: _ProviderContext) -> SamplingPlan:
    identity = request.config.base_config.section("identity")
    seed_plan = SeedPlan.from_master_seed(int(identity["master_seed"]))
    universe = SamplingUniverse(
        universe_id=f"{context.provider.fixed_state_id}-universe",
        sample_ids=context.sample_ids,
        metadata={
            "registry_hash": context.provider.registry_hash,
            "provider_state_digest": context.provider.state_digest(),
        },
    )
    return SamplingPlan(
        universe=universe,
        stream_seeds={name: seed_plan.seed_for(name) for name in STREAM_NAMES},
    )


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
    preregistration: dict[str, JSONValue] = {
        "schema_version": "stage2-task-preregistration-v1",
        "scope": request.config.run_intent,
        "estimand": "fixed_state_per_coordinate_gradient_mean_square",
        "candidate_batch_sizes": list(CANDIDATE_BATCH_SIZES),
        "candidate_microbatch_counts": list(CANDIDATE_MICROBATCH_COUNTS),
        "microbatch_selection_order": list(MICROBATCH_SELECTION_ORDER),
        "sampling_stream_names": list(STREAM_NAMES),
        "seed_plan_hash": seed_plan.artifact_hash,
        "formal_primary_values_status": "UNFROZEN",
        "formal_eligible": False,
    }
    hypothesis: dict[str, JSONValue] = {
        "schema_version": "stage2-task-hypothesis-contract-v1",
        "primary_comparison": "u_vs_double_under_paired_equal_sample_budget",
        "null_hypotheses": [
            "mean_signed_bias_interval_contains_zero",
            "candidate_corrected_nmse_not_better_than_double",
        ],
        "statistical_unit": "independent_repetition",
        "weight_contract_required": [
            "statistical_unit",
            "weight_unit",
            "sampling_design",
            "weights_exogenous",
            "common_mean_assumption",
        ],
        "multiplicity_policy": "preregistered_family_no_posthoc_promotion",
        "upstream_binding_hash": inputs.binding_hash,
    }
    return (
        {
            "preregistration": preregistration,
            "hypothesis_contract": hypothesis,
            "gate_record": _gate_candidate(request),
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
        # handoff invariant 成功不等于本阶段 Gate 已通过；任务 envelope 会保留 formal
        # 执行身份，而科学 artifact 仍等待独立审核。
        "formal_eligible": False,
    }
    fixed_state: dict[str, JSONValue] = {
        "schema_version": "stage2-task-fixed-state-contract-v1",
        "fixed_state_id": context.provider.fixed_state_id,
        "provider_state_digest": context.provider.state_digest(),
        "registry_hash": context.provider.registry_hash,
        "parameter_names": list(context.provider.parameter_names),
        "weighting_assumptions": context.to_payload()["weighting_assumptions"],
        "mutation_policy": "read_only_gradient_queries",
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
        inputs.references,
    )


def _run_stage2_assets_and_sampling(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
) -> tuple[Mapping[str, Mapping[str, JSONValue]], tuple[str, ...]]:
    """解析 provider，并实际生成五条可重放 draw stream 的小型 manifest。"""

    inputs = _predecessor_context(request, root, store)
    fixed_state = inputs.payload("fixed_state_contract")
    context = _provider_context(request, root)
    if (
        fixed_state.get("fixed_state_id") != context.provider.fixed_state_id
        or fixed_state.get("provider_state_digest") != context.provider.state_digest()
        or fixed_state.get("registry_hash") != context.provider.registry_hash
    ):
        raise _blocked(
            BlockerCode.ASSET_UNAVAILABLE,
            "fixed_state_provider_binding",
            "当前 provider 与前序 fixed_state_contract 身份不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
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
    draw_rows = [
        draw.to_manifest()
        for stream in STREAM_NAMES
        for draw in sampling.draws(stream, 4)  # type: ignore[arg-type]
    ]
    draw_ids = [str(row["draw_id"]) for row in draw_rows]
    if len(draw_ids) != len(set(draw_ids)):
        raise RuntimeError("STAGE2_DRAW_ID_COLLISION")
    payloads: dict[str, Mapping[str, JSONValue]] = {
        "sampling_plan": sampling.to_dict(),  # type: ignore[dict-item]
        "draw_manifest": {
            "schema_version": "stage2-task-draw-manifest-v1",
            "sampling_plan_hash": sampling.digest,
            "draws": draw_rows,  # type: ignore[dict-item]
            "draw_count_by_stream": {name: 4 for name in STREAM_NAMES},
            "draw_id_unique": True,
            "sample_id_collisions_allowed": True,
            "replay_hash": canonical_json_hash(draw_rows),
            "nested_mapping": nested_mapping.to_manifest(),
            "nested_mapping_hash": nested_mapping.digest,
        },
        "asset_resolution": {
            "schema_version": "stage2-task-asset-resolution-v1",
            "provider": context.to_payload(),
            "fixed_state_contract_hash": canonical_json_hash(fixed_state),
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
) -> dict[str, object]:
    """发布与恢复路径无关的 ``reference-result-v1``。

    formal core result 中的 ``resumed_from_block_pairs`` 是运行诊断，不属于 estimand，
    因此这里不把它间接带入共享 reference 身份。
    """

    bias = getattr(result, "bias_reference")
    cross = getattr(result, "cross_reference")
    ranking = getattr(result, "ranking_reference")
    selected = getattr(result, "selected_sample_count_per_stream")
    processed = getattr(result, "processed_sample_count_per_stream")
    scope = str(getattr(result, "scope"))
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
        "metadata": {
            "candidate_status": str(getattr(result, "status")),
            "converged": bool(getattr(result, "converged")),
            "weighting_assumptions": dict(getattr(result, "weighting_assumptions")),
            "qualification_gate_hash": None,
        },
        "tensor_bundle_ref": bundle_ref,
        "tensor_bundle_manifest_hash": bundle_hash,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    validate_reference_result_artifact(payload)
    return payload


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
    if upstream_sampling.digest != _sampling_plan(request, context).digest:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_sampling_plan",
            "前序 sampling_plan 与当前 fixed-state provider/seed 不一致",
            retryable=False,
            evidence_refs=inputs.references,
        )
    plan, plan_refs = _stage2_reference_plan(request, root, context)
    sampling = upstream_sampling
    maximum = plan.candidate_sample_counts[-1]
    result = StreamingReferenceSizer(context.provider).run(
        plan,
        draws_a=sampling.draws("reference_A", maximum),
        draws_b=sampling.draws("reference_B", maximum),
        artifact_root=store.root / "resume" / "reference-sizing",
    )
    bundle_path = store.root / "tensor-bundles" / "reference-final"
    bundle = _publish_or_load_bundle(
        bundle_path,
        {
            "bias_reference": result.bias_reference,
            "cross_reference": result.cross_reference,
            "ranking_reference": result.ranking_reference,
        },
    )
    bundle_ref = bundle_path.relative_to(root).as_posix()
    reference = _stable_reference_artifact(
        reference_id=plan.reference_id,
        result=result,
        block_size=plan.block_size,
        bundle_ref=bundle_ref,
        bundle_hash=bundle.manifest_sha256,
    )
    convergence: dict[str, JSONValue] = {
        "schema_version": "stage2-reference-convergence-report-v1",
        "plan": plan.to_dict(),  # type: ignore[dict-item]
        "status": result.status,
        "converged": result.converged,
        "selected_sample_count_per_stream": result.selected_sample_count_per_stream,
        "processed_sample_count_per_stream": result.processed_sample_count_per_stream,
        "points": [point.to_dict() for point in result.points],  # type: ignore[list-item]
        "provider": context.to_payload(),
        "sampling_plan_hash": sampling.digest,
        "recovery_semantics": "authoritative_block_pair_commits",
        "formal_eligible": False,
    }
    payload_by_kind: dict[str, Mapping[str, JSONValue]] = {
        "reference_result": reference,  # type: ignore[dict-item]
        "reference_convergence_report": convergence,
        "gate_record": _gate_candidate(request),
    }
    return payload_by_kind, _source_refs(request, plan_refs)


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
    """保留科学统计和三类成本定义，排除不可复现的 wall/formula seconds。"""

    costs: dict[str, JSONValue] = {}
    for name, raw in sorted(getattr(summary, "cost_statistics").items()):
        values = dict(raw)
        costs[name] = {
            "defined": bool(values["defined"]),
            "gradient_evaluations": values["gradient_evaluations"],
            "formula_seconds": None,
            "wall_seconds": None,
            "reason": (
                "local_timing_excluded_from_canonical_artifact"
                if values["defined"]
                else values["reason"]
            ),
        }
    return {
        "schema_version": "stage2-task-wave-summary-v1",
        "wave_id": str(getattr(summary, "wave_id")),
        "registry_hash": str(getattr(summary, "registry_hash")),
        "reference_hash": str(getattr(summary, "reference_hash")),
        "expected_unit_ids": list(getattr(summary, "expected_unit_ids")),
        "completed_unit_ids": list(getattr(summary, "completed_unit_ids")),
        "complete": bool(getattr(summary, "complete")),
        "status": str(getattr(summary, "status")),
        "method_statistics": {
            name: dict(values)
            for name, values in sorted(getattr(summary, "method_statistics").items())
        },
        "cost_statistics": costs,
        "weighting_assumptions": dict(getattr(summary, "weighting_assumptions")),
        "scope": str(getattr(summary, "scope")),
        "formal_eligible": False,
        "recovery_semantics": "authoritative_repetition_commits",
    }


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
        reference_manifest = inputs.payload("reference_result")
        try:
            bundle_ref = str(reference_manifest["tensor_bundle_ref"])
            reference_state, bundle = load_tensor_bundle(
                _workspace_path(root, bundle_ref, field="reference_tensor_bundle")
            )
            if bundle.manifest_sha256 != reference_manifest.get(
                "tensor_bundle_manifest_hash"
            ):
                raise ValueError("REFERENCE_TENSOR_BUNDLE_HASH_MISMATCH")
            if not isinstance(reference_state, Mapping):
                raise ValueError("REFERENCE_TENSOR_BUNDLE_ROOT_NOT_MAPPING")
            reference = _as_numpy_vector(reference_state["bias_reference"])  # type: ignore[arg-type]
            if _vector_digest(reference) != reference_manifest.get(
                "bias_reference_hash"
            ):
                raise ValueError("REFERENCE_VECTOR_HASH_MISMATCH")
        except (KeyError, TypeError, ValueError) as error:
            raise _blocked(
                BlockerCode.ASSET_UNAVAILABLE,
                "stage2_reference_tensor_bundle",
                f"前序 reference_result 无法恢复：{error}",
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
        reference_hash=_vector_digest(reference),
        artifact_root=store.root / "resume" / "paired-wave",
    )
    stable = _stable_wave_payload(summary)
    commits = sorted(
        path.name
        for path in (store.root / "resume" / "paired-wave" / "commits").glob("*.json")
    )
    shard_payload: dict[str, JSONValue] = {
        "schema_version": "stage2-task-sufficient-stat-shards-v1",
        "sampling_plan_hash": sampling.digest,
        "experiment_plan_hash": (
            None if experiment_plan is None else experiment_plan.artifact_hash
        ),
        "mapping_hashes": [mapping.digest for mapping in mappings],
        "committed_units": commits,
        "expected_unit_count": len(mappings),
        "complete": bool(summary.complete),
        "recovery_semantics": "immutable_tensor_bundle_plus_authoritative_commit",
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
    try:
        statistics = FrozenSourceTable.from_mapping(inputs.payload("frozen_source_table"))
    except (TypeError, ValueError) as error:
        raise _blocked(
            BlockerCode.CONTRACT_UNFROZEN,
            "stage2_frozen_statistics_table",
            f"冻结统计源表不可用：{error}",
            retryable=False,
            evidence_refs=inputs.references,
        ) from error
    quality = inputs.payload("quality_gates")
    costs = quality.get("cost_statistics")
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
    recommendation = quality.get("pilot_recommendation")
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
        return TaskRunResult.passed(
            request,
            artifact_refs=references,
            message="stage2/3 specialized task completed",
            metadata={"execution_contract": "stage23-specialized-v1"},
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
