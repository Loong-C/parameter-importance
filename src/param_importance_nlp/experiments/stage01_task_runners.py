"""Stage 0/1 非训练任务的专用复合 task runner。

本模块不把审计、存储、数学验证和交付任务降级成同一个 ``core_probe``。每个
canonical task ID 都会调用其负责层的真实 API，并发布带任务语义的证据：例如
checkpoint 任务会真的执行两阶段发布、发现、恢复和 tombstone-only retention；
Stage 1 estimator 任务会分别计算生产核与独立 FP64 oracle。

这里的 tiny 数值只承担本机合同验证，产物始终把 formal Gate 写为 ``NOT_RUN``。
formal 命令所需服务器、设备、资产或上游 Gate 仍由统一 ``TaskRuntime`` preflight
计算为结构化 ``BLOCKED``，本模块既不探测网络，也不伪造这些能力。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import platform
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping

import torch

from ..asset_layout import load_stage0_asset_layout
from ..asset_requirements import load_stage0_asset_requirements
from ..assets import (
    AssetActorRole,
    AssetState,
    AssetType,
    build_manifest,
    load_asset_manifest,
    transition_manifest,
    validate_g3_qualification,
    verify_only,
)
from ..atomic import atomic_write_bytes, sha256_file
from ..capacity import (
    estimate_checkpoint_bytes,
    estimate_experiment_storage,
    estimate_parameter_statistics_bytes,
)
from ..contracts.identity import RunIdentity, derive_experiment_id
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.freeze import ContractFreeze
from ..contracts.runtime_evidence import RuntimeCapabilityEvidence
from ..contracts.seed import SeedPlan
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import RunnerKind
from ..core.estimators import (
    double_sample_importance,
    equal_u_importance,
    raw_importance,
)
from ..core.losses import causal_lm_loss, sequence_classification_loss
from ..core.oracles import (
    compare_tensor_maps_fp64,
    fp64_double_sample_oracle,
    fp64_equal_u_oracle,
    fp64_mean_gradient_oracle,
    fp64_raw_oracle,
)
from ..core.registry import ParameterRegistry
from ..core.sufficient_statistics import EqualSufficientStatistics
from ..core.tensors import TensorMap
from ..runtime.checkpoint import CheckpointRetentionPolicy, CheckpointStore
from ..runtime.events import (
    EventRecord,
    EventType,
    JsonlEventSink,
    canonical_optimizer_steps,
    read_event_stream,
)
from ..runtime.gradients import GradientAttempt
from ..runtime.lineage import AttemptDisposition, LineageStore
from ..runtime.optimizer import OptimizerBridge, compute_global_clip_factor
from ..runtime.task_artifacts import (
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
)
from ..runtime.tensor_bundle import load_tensor_bundle
from ..g3_gate import (
    GATE_IDS as G3_GATE_IDS,
    evaluate_stage0_g3,
    validate_stage0_g3_resolution,
)
from ..storage import (
    REQUIRED_DIRECTORIES,
    StorageLayout,
    is_within,
    require_data_root,
)


_HANDLED_BY_KIND: Mapping[RunnerKind, frozenset[str]] = MappingProxyType(
    {
        RunnerKind.AUDIT: frozenset({"stage0.01_baseline_and_safety"}),
        RunnerKind.STORAGE: frozenset({"stage0.02_storage_and_layout"}),
        RunnerKind.ENVIRONMENT: frozenset({"stage0.03_runtime_and_dependencies"}),
        RunnerKind.ASSET: frozenset({"stage0.04_assets_and_manifests"}),
        RunnerKind.CONTRACT: frozenset(
            {"stage0.05_config_run_identity_and_seeds", "stage1.01_entry_and_contract"}
        ),
        RunnerKind.OBSERVABILITY: frozenset({"stage0.08_logging_and_tracking"}),
        RunnerKind.CHECKPOINT: frozenset(
            {"stage0.09_checkpoint_and_resume", "stage1.10_checkpoint_resume_and_artifacts"}
        ),
        RunnerKind.CAPACITY: frozenset({"stage0.10_capacity_and_operations"}),
        RunnerKind.TEST_MATRIX: frozenset({"stage0.11_test_quality_and_replay"}),
        RunnerKind.DELIVERY: frozenset({"stage0.12_delivery_and_sync"}),
        RunnerKind.REGISTRY: frozenset({"stage1.02_architecture_and_parameter_registry"}),
        RunnerKind.ORACLE: frozenset({"stage1.03_fixtures_and_oracles"}),
        RunnerKind.VALIDATION: frozenset(
            {
                "stage1.04_loss_and_gradient_scale",
                "stage1.09_precision_clipping_and_optimizer_boundaries",
            }
        ),
        RunnerKind.ESTIMATOR: frozenset({"stage1.05_estimators"}),
        RunnerKind.REPORTING: frozenset({"stage1.11_reporting_and_exit_gate"}),
    }
)

STAGE01_HANDLED_TASK_IDS = frozenset(
    task_id for task_ids in _HANDLED_BY_KIND.values() for task_id in task_ids
)

# 这些任务的 formal 结论依赖服务器、资产、真实训练或交付现场，不能重跑下面的
# tiny/local 探针后发布成 formal 产物。它们的正式路径只消费已经通过两阶段协议
# 发布的上游/能力证据，并生成 hash-bound 的证据投影。
_FORMAL_EVIDENCE_ONLY_TASKS = frozenset(
    {
        "stage0.01_baseline_and_safety",
        "stage0.02_storage_and_layout",
        "stage0.03_runtime_and_dependencies",
        "stage0.10_capacity_and_operations",
        "stage0.11_test_quality_and_replay",
        "stage0.12_delivery_and_sync",
        "stage1.09_precision_clipping_and_optimizer_boundaries",
        "stage1.10_checkpoint_resume_and_artifacts",
        "stage1.11_reporting_and_exit_gate",
    }
)

_G3_TASK_ID = "stage0.04_assets_and_manifests"
_G4_TASK_ID = "stage0.05_config_run_identity_and_seeds"
_G3_REQUIREMENTS_REF = "configs/stage0/g3-asset-requirements-v1.json"
_G3_LAYOUT_REF = "configs/stage0/g3-asset-layout-v1.json"
_G3_MANIFEST_SCHEMA_VERSION = "stage0-g3-asset-manifest-index-v1"
_G3_AUDIT_SCHEMA_VERSION = "stage0-g3-asset-audit-v1"
_G3_CRITICAL_SOURCE_REFS = (
    _G3_REQUIREMENTS_REF,
    _G3_LAYOUT_REF,
    "configs/stage0/g3-download-plan-v1.json",
    "ops/stage0/attest_g3_materialization.py",
    "ops/stage0/materialize_and_publish_g3.py",
    "ops/stage0/verify_g3_assets.py",
    "schemas/stage0/asset-layout-v1.json",
    "schemas/stage0/asset-requirements-v1.json",
    "schemas/stage0/download-plan-v1.json",
    "schemas/stage0-asset-manifest-v1.json",
    "schemas/stage0-g3-acquisition-report-v1.json",
    "schemas/stage0-g3-verify-only-report-v1.json",
    "src/param_importance_nlp/asset_acquisition.py",
    "src/param_importance_nlp/asset_download_plan.py",
    "src/param_importance_nlp/asset_layout.py",
    "src/param_importance_nlp/asset_requirements.py",
    "src/param_importance_nlp/assets.py",
    "src/param_importance_nlp/atomic.py",
    "src/param_importance_nlp/contracts/__init__.py",
    "src/param_importance_nlp/contracts/jsonio.py",
    "src/param_importance_nlp/data/pythia_mmap.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/g3_asset_publication.py",
    "src/param_importance_nlp/g3_gate.py",
    "src/param_importance_nlp/g3_lifecycle_evidence.py",
    "src/param_importance_nlp/g3_semantic_evidence.py",
    "src/param_importance_nlp/glue_builder.py",
    "src/param_importance_nlp/providers/optional.py",
    "src/param_importance_nlp/runtime/task_artifacts.py",
)
_G3_MODULE_ORIGINS = (
    (
        "param_importance_nlp.asset_acquisition",
        "src/param_importance_nlp/asset_acquisition.py",
    ),
    (
        "param_importance_nlp.asset_download_plan",
        "src/param_importance_nlp/asset_download_plan.py",
    ),
    ("param_importance_nlp.asset_layout", "src/param_importance_nlp/asset_layout.py"),
    (
        "param_importance_nlp.asset_requirements",
        "src/param_importance_nlp/asset_requirements.py",
    ),
    ("param_importance_nlp.assets", "src/param_importance_nlp/assets.py"),
    ("param_importance_nlp.atomic", "src/param_importance_nlp/atomic.py"),
    (
        "param_importance_nlp.contracts",
        "src/param_importance_nlp/contracts/__init__.py",
    ),
    (
        "param_importance_nlp.contracts.jsonio",
        "src/param_importance_nlp/contracts/jsonio.py",
    ),
    (
        "param_importance_nlp.data.pythia_mmap",
        "src/param_importance_nlp/data/pythia_mmap.py",
    ),
    (
        "param_importance_nlp.g3_asset_publication",
        "src/param_importance_nlp/g3_asset_publication.py",
    ),
    ("param_importance_nlp.g3_gate", "src/param_importance_nlp/g3_gate.py"),
    (
        "param_importance_nlp.g3_semantic_evidence",
        "src/param_importance_nlp/g3_semantic_evidence.py",
    ),
    (
        "param_importance_nlp.g3_lifecycle_evidence",
        "src/param_importance_nlp/g3_lifecycle_evidence.py",
    ),
    (
        "param_importance_nlp.glue_builder",
        "src/param_importance_nlp/glue_builder.py",
    ),
    (
        "param_importance_nlp.providers.optional",
        "src/param_importance_nlp/providers/optional.py",
    ),
    (
        "param_importance_nlp.runtime.task_artifacts",
        "src/param_importance_nlp/runtime/task_artifacts.py",
    ),
)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_G3_OUTPUT_SCHEMAS: Mapping[str, str] = MappingProxyType(
    {
        "asset_manifest": _G3_MANIFEST_SCHEMA_VERSION,
        "asset_audit": _G3_AUDIT_SCHEMA_VERSION,
        "asset_resolution": "stage0-g3-resolution-audit-v1",
    }
)


@dataclass(frozen=True, slots=True)
class _G3SourceBinding:
    source_root: Path
    head_commit: str
    requirements_path: Path
    requirements_file_sha256: str
    layout_path: Path
    layout_file_sha256: str

    def payload(self, *, producer_git_commit: str) -> dict[str, JSONValue]:
        return {
            "producer_git_commit": producer_git_commit,
            "requirements_ref": _G3_REQUIREMENTS_REF,
            "requirements_file_sha256": self.requirements_file_sha256,
            "layout_ref": _G3_LAYOUT_REF,
            "layout_file_sha256": self.layout_file_sha256,
            "critical_source_refs": list(_G3_CRITICAL_SOURCE_REFS),
        }


def _assert_g3_module_origins(source_root: Path) -> None:
    expected_current = source_root.joinpath(
        *PurePosixPath(
            "src/param_importance_nlp/experiments/stage01_task_runners.py"
        ).parts
    ).resolve(strict=True)
    if Path(__file__).resolve(strict=True) != expected_current:
        raise ValueError(
            "STAGE0_G3_IMPORTED_MODULE_ORIGIN_MISMATCH:stage01_task_runners"
        )
    for module_name, reference in _G3_MODULE_ORIGINS:
        module = importlib.import_module(module_name)
        raw_origin = getattr(module, "__file__", None)
        if not isinstance(raw_origin, str) or not raw_origin:
            raise ValueError(
                f"STAGE0_G3_IMPORTED_MODULE_ORIGIN_MISSING:{module_name}"
            )
        expected = source_root.joinpath(*PurePosixPath(reference).parts).resolve(
            strict=True
        )
        if Path(raw_origin).resolve(strict=True) != expected:
            raise ValueError(
                f"STAGE0_G3_IMPORTED_MODULE_ORIGIN_MISMATCH:{module_name}"
            )


def _assert_g3_critical_source_paths(source_root: Path) -> None:
    for reference in _G3_CRITICAL_SOURCE_REFS:
        current = source_root
        for part in PurePosixPath(reference).parts:
            current = current / part
            if _is_link_like(current):
                raise ValueError("STAGE0_G3_CRITICAL_SOURCE_LINK_FORBIDDEN")
        if not current.is_file():
            raise ValueError("STAGE0_G3_CRITICAL_SOURCE_INVALID")


def _git_constrained_g3_sources() -> _G3SourceBinding:
    """Capture one clean, tracked current view of every critical G3 source."""

    source_root = Path(__file__).resolve().parents[3]
    command_prefix = (
        "git",
        "-c",
        f"safe.directory={source_root.as_posix()}",
        "-C",
        str(source_root),
    )
    try:
        top_level = subprocess.run(
            [*command_prefix, "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if Path(top_level).resolve() != source_root:
            raise ValueError("STAGE0_G3_SOURCE_GIT_ROOT_MISMATCH")
        head_commit = subprocess.run(
            [*command_prefix, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if _GIT_COMMIT_RE.fullmatch(head_commit) is None:
            raise ValueError("STAGE0_G3_SOURCE_HEAD_INVALID")
        subprocess.run(
            [
                *command_prefix,
                "ls-files",
                "--error-unmatch",
                "--",
                *_G3_CRITICAL_SOURCE_REFS,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        dirty = subprocess.run(
            [
                *command_prefix,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                *_G3_CRITICAL_SOURCE_REFS,
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("STAGE0_G3_SOURCE_NOT_GIT_CONSTRAINED") from error
    if dirty.strip():
        raise ValueError("STAGE0_G3_CRITICAL_SOURCE_DIRTY")

    _assert_g3_critical_source_paths(source_root)
    _assert_g3_module_origins(source_root)
    requirements_path = source_root.joinpath(
        *PurePosixPath(_G3_REQUIREMENTS_REF).parts
    )
    layout_path = source_root.joinpath(*PurePosixPath(_G3_LAYOUT_REF).parts)
    return _G3SourceBinding(
        source_root=source_root,
        head_commit=head_commit,
        requirements_path=requirements_path,
        requirements_file_sha256=sha256_file(requirements_path),
        layout_path=layout_path,
        layout_file_sha256=sha256_file(layout_path),
    )


def _run_g3_git(
    source_root: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={source_root.as_posix()}",
                "-C",
                str(source_root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("STAGE0_G3_SOURCE_NOT_GIT_CONSTRAINED") from error


def _assert_g3_producer_commit_compatible(
    binding: _G3SourceBinding,
    producer_commit: str,
) -> None:
    """Accept later unrelated commits while rejecting any critical-source drift."""

    if _GIT_COMMIT_RE.fullmatch(producer_commit) is None:
        raise ValueError("STAGE0_G3_ASSET_GENERATOR_COMMIT_INVALID")
    current_head = _run_g3_git(binding.source_root, "rev-parse", "HEAD")
    if (
        current_head.returncode != 0
        or current_head.stdout.strip() != binding.head_commit
    ):
        raise ValueError("STAGE0_G3_SOURCE_BINDING_DRIFTED")
    verified = _run_g3_git(
        binding.source_root,
        "rev-parse",
        "--verify",
        f"{producer_commit}^{{commit}}",
    )
    if verified.returncode != 0 or verified.stdout.strip() != producer_commit:
        raise ValueError("STAGE0_G3_ASSET_GENERATOR_COMMIT_UNKNOWN")
    ancestor = _run_g3_git(
        binding.source_root,
        "merge-base",
        "--is-ancestor",
        producer_commit,
        binding.head_commit,
    )
    if ancestor.returncode == 1:
        raise ValueError("STAGE0_G3_ASSET_GENERATOR_COMMIT_NOT_ANCESTOR")
    if ancestor.returncode != 0:
        raise ValueError("STAGE0_G3_SOURCE_NOT_GIT_CONSTRAINED")

    for reference in _G3_CRITICAL_SOURCE_REFS:
        present = _run_g3_git(
            binding.source_root,
            "cat-file",
            "-e",
            f"{producer_commit}:{reference}",
        )
        if present.returncode != 0:
            raise ValueError(
                f"STAGE0_G3_ASSET_GENERATOR_SOURCE_REF_ABSENT:{reference}"
            )

    committed = _run_g3_git(
        binding.source_root,
        "diff",
        "--quiet",
        f"{producer_commit}..{binding.head_commit}",
        "--",
        *_G3_CRITICAL_SOURCE_REFS,
    )
    if committed.returncode == 1:
        raise ValueError("STAGE0_G3_CRITICAL_SOURCE_DRIFT")
    if committed.returncode != 0:
        raise ValueError("STAGE0_G3_SOURCE_NOT_GIT_CONSTRAINED")

    # Re-check both views here as well as in ``_git_constrained_g3_sources``.
    # This keeps the producer-compatibility predicate independently fail-closed
    # when it is used directly and closes the gap between the earlier capture
    # and the later producer comparison.
    for scope in ((), ("--cached",)):
        dirty = _run_g3_git(
            binding.source_root,
            "diff",
            *scope,
            "--quiet",
            "--",
            *_G3_CRITICAL_SOURCE_REFS,
        )
        if dirty.returncode == 1:
            raise ValueError("STAGE0_G3_CRITICAL_SOURCE_DIRTY")
        if dirty.returncode != 0:
            raise ValueError("STAGE0_G3_SOURCE_NOT_GIT_CONSTRAINED")


def _formal_g3_roots(workspace_root: Path) -> tuple[_G3SourceBinding, Path]:
    """Bind formal assets to the explicit DATA_ROOT and sources to Git."""

    workspace = workspace_root.resolve()
    data_root = require_data_root().resolve()
    if data_root != workspace:
        raise ValueError("STAGE0_G3_DATA_ROOT_WORKSPACE_MISMATCH")

    source_root = Path(__file__).resolve().parents[3]
    if is_within(workspace, source_root) or is_within(source_root, workspace):
        raise ValueError("STAGE0_G3_WORKSPACE_OVERLAPS_SOURCE_ROOT")
    return _git_constrained_g3_sources(), data_root


def _is_link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except (FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _g3_evidence_file(data_root: Path, reference: str) -> Path:
    logical = _logical(reference, field="g3_evidence_ref")
    current = data_root
    for part in logical.parts:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise ValueError("STAGE0_G3_EVIDENCE_LINK_FORBIDDEN")
    if not current.is_file():
        raise FileNotFoundError("STAGE0_G3_EVIDENCE_FILE_MISSING")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(data_root)
    except ValueError as error:
        raise ValueError("STAGE0_G3_EVIDENCE_PATH_ESCAPE") from error
    return resolved


@dataclass(frozen=True, slots=True)
class _G3EvidenceIdentity:
    checked_at: str
    producer_git_commit: str


def _stable_g3_evidence_identity(
    binding: _G3SourceBinding,
    data_root: Path,
) -> _G3EvidenceIdentity:
    """Derive the unique timestamp and producer commit admitted by all assets."""

    requirements = load_stage0_asset_requirements(binding.requirements_path)
    layout = load_stage0_asset_layout(
        binding.layout_path,
        requirements=requirements,
    )
    checked_at_values: set[str] = set()
    producer_commits: set[str] = set()
    for entry in layout["entries"]:
        manifest = load_asset_manifest(
            _g3_evidence_file(data_root, entry["manifest_ref"])
        )
        raw_qualification = load_canonical_json(
            _g3_evidence_file(data_root, entry["qualification_ref"])
        )
        if not isinstance(raw_qualification, Mapping):
            raise ValueError("STAGE0_G3_QUALIFICATION_ROOT_INVALID")
        qualification = dict(raw_qualification)
        validate_g3_qualification(qualification)
        manifest_commit = manifest.get("generator_git_commit")
        qualification_commit = qualification.get("generator_git_commit")
        if (
            not isinstance(manifest_commit, str)
            or _GIT_COMMIT_RE.fullmatch(manifest_commit) is None
            or qualification_commit != manifest_commit
        ):
            raise ValueError("STAGE0_G3_ASSET_GENERATOR_COMMIT_AMBIGUOUS")
        producer_commits.add(manifest_commit)
        checked_at = qualification.get("checked_at")
        if not isinstance(checked_at, str):
            raise ValueError("STAGE0_G3_QUALIFICATION_CHECKED_AT_INVALID")
        checked_at_values.add(checked_at)
    if len(checked_at_values) != 1:
        raise ValueError("STAGE0_G3_QUALIFICATION_CHECKED_AT_NOT_UNIQUE")
    if len(producer_commits) != 1:
        raise ValueError("STAGE0_G3_ASSET_GENERATOR_COMMIT_AMBIGUOUS")
    return _G3EvidenceIdentity(
        checked_at=next(iter(checked_at_values)),
        producer_git_commit=next(iter(producer_commits)),
    )


def _stable_g3_checked_at(
    binding: _G3SourceBinding,
    data_root: Path,
) -> str:
    """Return the replay-stable qualification timestamp."""

    return _stable_g3_evidence_identity(binding, data_root).checked_at


def _evaluate_current_formal_g3(
    binding: _G3SourceBinding,
    data_root: Path,
) -> tuple[dict[str, Any], str]:
    """Replay qualified resolution and re-check source/asset bindings."""

    evidence_identity = _stable_g3_evidence_identity(binding, data_root)
    _assert_g3_producer_commit_compatible(
        binding,
        evidence_identity.producer_git_commit,
    )
    resolution = evaluate_stage0_g3(
        binding.requirements_path,
        binding.layout_path,
        data_root,
        checked_at=evidence_identity.checked_at,
    )
    _require_formal_g3_pass(resolution)
    if resolution.get("checked_at") != evidence_identity.checked_at:
        raise ValueError("STAGE0_G3_RESOLUTION_CHECKED_AT_DRIFT")
    if _stable_g3_evidence_identity(binding, data_root) != evidence_identity:
        raise ValueError("STAGE0_G3_ASSET_BINDING_DRIFTED")
    if _git_constrained_g3_sources() != binding:
        raise ValueError("STAGE0_G3_SOURCE_BINDING_DRIFTED")
    return resolution, evidence_identity.producer_git_commit


def _g3_payload_with_hash(payload: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    value = dict(payload)
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_formal_g3_pass(resolution: Mapping[str, object]) -> None:
    validate_stage0_g3_resolution(resolution)
    raw_entries = resolution.get("entries")
    raw_gates = resolution.get("gates")
    if not isinstance(raw_entries, list) or len(raw_entries) != 13:
        raise ValueError("STAGE0_G3_RESOLUTION_ENTRY_COUNT_INVALID")
    if not isinstance(raw_gates, list) or len(raw_gates) != len(G3_GATE_IDS):
        raise ValueError("STAGE0_G3_RESOLUTION_GATE_COUNT_INVALID")
    if resolution.get("status") != "PASS" or tuple(
        gate.get("gate_id") if isinstance(gate, Mapping) else None
        for gate in raw_gates
    ) != G3_GATE_IDS:
        raise ValueError("STAGE0_G3_AGGREGATE_BLOCKED")
    if any(
        not isinstance(gate, Mapping) or gate.get("status") != "PASS"
        for gate in raw_gates
    ):
        raise ValueError("STAGE0_G3_SUBGATE_BLOCKED")

    logical_names: set[str] = set()
    acquisition_bindings: set[tuple[str, str]] = set()
    for entry in raw_entries:
        if not isinstance(entry, Mapping) or entry.get("status") != "PASS":
            raise ValueError("STAGE0_G3_ENTRY_BLOCKED")
        logical_name = entry.get("logical_name")
        checks = entry.get("checks")
        reasons = entry.get("reasons")
        if (
            not isinstance(logical_name, str)
            or logical_name in logical_names
            or not isinstance(checks, Mapping)
            or not checks
            or any(value is not True for value in checks.values())
            or reasons != []
        ):
            raise ValueError("STAGE0_G3_ENTRY_EVIDENCE_INVALID")
        logical_names.add(logical_name)
        for field in (
            "asset_id",
            "candidate_id",
            "candidate_sha256",
            "ready_manifest_sha256",
            "qualification_artifact_hash",
            "acquisition_sha256",
            "verification_sha256",
            "semantic_evidence_sha256",
            "semantic_evidence_artifact_hash",
        ):
            if not _is_sha256(entry.get(field)):
                raise ValueError(f"STAGE0_G3_ENTRY_DIGEST_INVALID:{field}")
        for field in (
            "manifest_ref",
            "candidate_ref",
            "qualification_ref",
            "acquisition_ref",
            "verification_ref",
            "semantic_evidence_ref",
        ):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise ValueError(f"STAGE0_G3_ENTRY_REF_INVALID:{field}")
        acquisition_ref = str(entry["acquisition_ref"])
        acquisition_sha256 = str(entry["acquisition_sha256"])
        if acquisition_ref != (
            "manifests/evidence/g3/acquisition/"
            f"{acquisition_sha256}.json"
        ):
            raise ValueError("STAGE0_G3_ENTRY_ACQUISITION_BINDING_INVALID")
        acquisition_bindings.add((acquisition_ref, acquisition_sha256))
    if len(acquisition_bindings) != 1:
        raise ValueError("STAGE0_G3_ACQUISITION_REPORT_NOT_UNIQUE")


def _formal_g3_payloads(
    resolution: Mapping[str, object],
    *,
    source_binding: _G3SourceBinding,
    producer_git_commit: str,
) -> dict[str, dict[str, JSONValue]]:
    """Project one validated resolution into three non-overlapping artifacts."""

    _require_formal_g3_pass(resolution)
    entries = resolution["entries"]
    gates = resolution["gates"]
    assert isinstance(entries, list) and isinstance(gates, list)

    manifest_entries: list[JSONValue] = []
    audit_entries: list[JSONValue] = []
    for raw_entry in entries:
        assert isinstance(raw_entry, Mapping)
        manifest_entries.append(
            {
                "logical_name": raw_entry["logical_name"],
                "kind": raw_entry["kind"],
                "requirement_name": raw_entry["requirement_name"],
                "manifest_ref": raw_entry["manifest_ref"],
                "asset_id": raw_entry["asset_id"],
                "candidate_id": raw_entry["candidate_id"],
                "candidate_ref": raw_entry["candidate_ref"],
                "candidate_sha256": raw_entry["candidate_sha256"],
                "ready_manifest_sha256": raw_entry["ready_manifest_sha256"],
            }
        )
        audit_entries.append(
            {
                "logical_name": raw_entry["logical_name"],
                "kind": raw_entry["kind"],
                "status": raw_entry["status"],
                "checks": raw_entry["checks"],
                "reasons": raw_entry["reasons"],
                "qualification_ref": raw_entry["qualification_ref"],
                "qualification_artifact_hash": raw_entry[
                    "qualification_artifact_hash"
                ],
                "acquisition_ref": raw_entry["acquisition_ref"],
                "acquisition_sha256": raw_entry["acquisition_sha256"],
                "verification_ref": raw_entry["verification_ref"],
                "verification_sha256": raw_entry["verification_sha256"],
                "semantic_evidence_ref": raw_entry["semantic_evidence_ref"],
                "semantic_evidence_sha256": raw_entry[
                    "semantic_evidence_sha256"
                ],
                "semantic_evidence_artifact_hash": raw_entry[
                    "semantic_evidence_artifact_hash"
                ],
            }
        )

    shared: dict[str, JSONValue] = {
        "status": "PASS",
        "requirements_ref": resolution["requirements_ref"],
        "requirements_artifact_hash": resolution["requirements_artifact_hash"],
        "layout_artifact_hash": resolution["layout_artifact_hash"],
        "source_binding": source_binding.payload(
            producer_git_commit=producer_git_commit
        ),
    }
    asset_manifest = _g3_payload_with_hash(
        {
            "schema_version": _G3_MANIFEST_SCHEMA_VERSION,
            **shared,
            "entry_count": len(manifest_entries),
            "entries": manifest_entries,
        }
    )
    asset_audit = _g3_payload_with_hash(
        {
            "schema_version": _G3_AUDIT_SCHEMA_VERSION,
            **shared,
            "entry_count": len(audit_entries),
            "entries": audit_entries,
            "gates": gates,
        }
    )
    return {
        "asset_manifest": asset_manifest,
        "asset_audit": asset_audit,
        "asset_resolution": dict(resolution),
    }


def _formal_g3_source_refs(
    source_binding: _G3SourceBinding,
    resolution: Mapping[str, object],
    input_source_refs: tuple[str, ...],
    *,
    producer_git_commit: str,
) -> tuple[str, ...]:
    """Bind task envelopes to the producer and every replayed evidence object."""

    _require_formal_g3_pass(resolution)
    _assert_g3_producer_commit_compatible(source_binding, producer_git_commit)
    refs = list(input_source_refs)
    expected_git_refs = {
        f"git-source/{producer_git_commit}/{reference}"
        for reference in _G3_CRITICAL_SOURCE_REFS
    }
    refs.extend(
        f"git-source/{producer_git_commit}/{reference}"
        for reference in _G3_CRITICAL_SOURCE_REFS
    )
    requirements_ref = resolution.get("requirements_ref")
    if not isinstance(requirements_ref, str) or not requirements_ref:
        raise ValueError("STAGE0_G3_REQUIREMENTS_REF_INVALID")
    refs.append(requirements_ref)
    entries = resolution.get("entries")
    if not isinstance(entries, list):
        raise ValueError("STAGE0_G3_RESOLUTION_ENTRIES_INVALID")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("STAGE0_G3_RESOLUTION_ENTRY_INVALID")
        for field in (
            "manifest_ref",
            "asset_root_ref",
            "candidate_ref",
            "qualification_ref",
            "acquisition_ref",
            "verification_ref",
            "semantic_evidence_ref",
        ):
            reference = entry.get(field)
            if not isinstance(reference, str) or not reference:
                raise ValueError(f"STAGE0_G3_ENTRY_REF_INVALID:{field}")
            refs.append(reference)
    gates = resolution.get("gates")
    if not isinstance(gates, list):
        raise ValueError("STAGE0_G3_RESOLUTION_GATES_INVALID")
    for gate in gates:
        if not isinstance(gate, Mapping):
            raise ValueError("STAGE0_G3_RESOLUTION_GATE_INVALID")
        evidence_refs = gate.get("evidence_refs")
        if not isinstance(evidence_refs, list):
            raise ValueError("STAGE0_G3_GATE_EVIDENCE_REFS_INVALID")
        for reference in evidence_refs:
            if not isinstance(reference, str) or not reference:
                raise ValueError("STAGE0_G3_GATE_EVIDENCE_REF_INVALID")
            refs.append(reference)
    result = tuple(dict.fromkeys(refs))
    observed_git_refs = {
        reference for reference in result if reference.startswith("git-source/")
    }
    if observed_git_refs != expected_git_refs:
        raise ValueError("STAGE0_G3_SOURCE_REFS_NOT_EXACT")
    return result


def _restore_formal_g3_outputs(
    root: Path,
    references: Mapping[str, str],
    *,
    input_source_refs: tuple[str, ...],
) -> Mapping[str, object]:
    source_binding, data_root = _formal_g3_roots(root)
    # A complete old task commit is only a cache candidate.  Re-evaluate all
    # qualified assets before reading its PASS payload for restoration.
    resolution, producer_git_commit = _evaluate_current_formal_g3(
        source_binding,
        data_root,
    )
    expected = _formal_g3_payloads(
        resolution,
        source_binding=source_binding,
        producer_git_commit=producer_git_commit,
    )
    expected_sources = _formal_g3_source_refs(
        source_binding,
        resolution,
        input_source_refs,
        producer_git_commit=producer_git_commit,
    )
    loaded = {
        kind: load_committed_task_artifact(root, references[kind], require_formal=True)
        for kind in _G3_OUTPUT_SCHEMAS
    }
    for kind, expected_schema in _G3_OUTPUT_SCHEMAS.items():
        if loaded[kind].payload.get("schema_version") != expected_schema:
            raise ValueError(f"STAGE0_G3_RESTORE_SCHEMA_INVALID:{kind}")
    for kind in _G3_OUTPUT_SCHEMAS:
        if loaded[kind].source_refs != expected_sources:
            raise ValueError(f"STAGE0_G3_RESTORE_SOURCE_REFS_DRIFT:{kind}")
        if dict(loaded[kind].payload) != expected[kind]:
            raise ValueError(f"STAGE0_G3_RESTORE_PAYLOAD_DRIFT:{kind}")
    return resolution


def _logical(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"STAGE01_LOGICAL_PATH_INVALID:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"STAGE01_PATH_ESCAPE:{field}")
    return path


def _resolve(root: Path, value: str, *, field: str) -> Path:
    path = _logical(value, field=field)
    target = root.joinpath(*path.parts).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"STAGE01_PATH_ESCAPE:{field}") from error
    return target


def _store(request: TaskExecutionRequest, root: Path) -> TaskArtifactStore:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    return TaskArtifactStore(root, str(artifacts["output_dir"]))


def _input_evidence(
    request: TaskExecutionRequest, root: Path
) -> tuple[list[JSONValue], tuple[str, ...]]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    refs = tuple(str(item) for item in orchestration["input_result_refs"])
    evidence: list[JSONValue] = []
    formal_identities: dict[tuple[str, str], str] = {}
    for ref in refs:
        if request.config.run_intent == "formal":
            try:
                loaded = load_committed_task_artifact(root, ref, require_formal=True)
            except Exception as error:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_commit",
                        f"正式输入 commit 无法验证：{ref} ({type(error).__name__})",
                        True,
                        (ref,),
                    )
                ) from error
            identity = loaded.identity
            key = (identity.task_id, identity.artifact_kind)
            if key in formal_identities:
                raise TaskBlockedError(
                    TaskBlocker(
                        BlockerCode.ASSET_UNAVAILABLE,
                        "formal_input_duplicate",
                        f"正式输入身份重复：{identity.task_id}:{identity.artifact_kind}",
                        False,
                        (formal_identities[key], ref),
                    )
                )
            formal_identities[key] = ref
            evidence.append(
                {
                    "ref": ref,
                    "sha256": identity.artifact_hash,
                    "kind": "task-output-commit-v1",
                    "producer_task_id": identity.task_id,
                    "artifact_kind": identity.artifact_kind,
                    "formal_eligible": True,
                }
            )
            continue
        path = _resolve(root, ref, field="input_result_refs")
        if path.is_dir():
            _, identity = load_tensor_bundle(path)
            digest = identity.manifest_sha256
            kind = "tensor_bundle"
        elif path.suffix.casefold() == ".json":
            value = load_canonical_json(path)
            digest = canonical_json_hash(value)
            kind = (
                str(value.get("schema_version", "json"))
                if isinstance(value, dict)
                else "json"
            )
        elif path.is_file():
            digest = sha256_file(path)
            kind = "hash_bound_file"
        else:
            raise FileNotFoundError(f"STAGE01_INPUT_NOT_FOUND:{ref}")
        evidence.append({"ref": ref, "sha256": digest, "kind": kind})
    if request.config.run_intent == "formal":
        blockers: list[TaskBlocker] = []
        for contract in request.task.input_artifacts:
            if not contract.required:
                continue
            for artifact_kind in contract.artifact_kinds:
                if not any(
                    producer in contract.producer_task_ids and kind == artifact_kind
                    for producer, kind in formal_identities
                ):
                    blockers.append(
                        TaskBlocker(
                            BlockerCode.ASSET_UNAVAILABLE,
                            f"input:{contract.input_id}:{artifact_kind}",
                            f"缺少正式上游 commit：{contract.input_id}/{artifact_kind}",
                            True,
                        )
                    )
        if blockers:
            raise TaskBlockedError(*blockers)
    return evidence, refs


def _formal_payload(
    root: Path,
    reference: str,
    schema_version: str,
) -> Mapping[str, object]:
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    payload = loaded.payload
    if payload.get("schema_version") == schema_version:
        return payload
    candidates = [
        value
        for value in payload.values()
        if isinstance(value, Mapping) and value.get("schema_version") == schema_version
    ]
    if len(candidates) != 1:
        raise ValueError(f"STAGE01_FORMAL_PAYLOAD_NOT_UNIQUE:{schema_version}")
    return candidates[0]


def _formal_guard(request: TaskExecutionRequest, root: Path) -> None:
    """直接调用 runner 时也保留与统一 preflight 相同的外部条件防线。"""

    if request.config.run_intent != "formal":
        return
    policy = request.task.formal_eligibility
    blockers: list[TaskBlocker] = []
    for stage in policy.required_contract_stages:
        reference = request.environment.evidence_refs.get(f"contract_stage_{stage}")
        valid = False
        if stage in request.environment.frozen_contract_stages and reference is not None:
            try:
                freeze = ContractFreeze.from_mapping(
                    dict(_formal_payload(root, reference, "contract-freeze-v1"))
                )
                valid = freeze.stage == stage and freeze.formal_eligible
            except Exception:
                valid = False
        if not valid:
            blockers.append(
                TaskBlocker(
                    BlockerCode.CONTRACT_UNFROZEN,
                    f"stage{stage}",
                    f"Stage {stage} 缺少可验证的 formal ContractFreeze commit",
                    True,
                    (() if reference is None else (reference,)),
                )
            )
    for gate_id in policy.required_gate_ids:
        key = "gate_" + "".join(
            character if character.isalnum() else "_"
            for character in gate_id.casefold()
        ).strip("_")
        reference = request.environment.evidence_refs.get(key)
        valid = False
        if gate_id in request.environment.passed_gate_ids and reference is not None:
            try:
                gate = GateRecord.from_mapping(
                    dict(_formal_payload(root, reference, "gate-record-v1"))
                )
                valid = gate.gate_id == gate_id and gate.status is GateStatus.PASS
            except Exception:
                valid = False
        if not valid:
            blockers.append(
                TaskBlocker(
                    BlockerCode.GATE_NOT_READY,
                    gate_id,
                    f"前置 Gate 缺少可验证的 PASS commit：{gate_id}",
                    True,
                    (() if reference is None else (reference,)),
                )
            )
    for capability in sorted(policy.required_capabilities):
        reference = request.environment.evidence_refs.get(f"capability_{capability}")
        valid = False
        if capability in request.environment.capabilities and reference is not None:
            try:
                item = RuntimeCapabilityEvidence.from_mapping(
                    _formal_payload(
                        root,
                        reference,
                        "runtime-capability-evidence-v1",
                    )
                )
                valid = item.capability == capability and item.verified
            except Exception:
                valid = False
        if valid:
            continue
        code = (
            BlockerCode.SERVER_UNREACHABLE
            if capability == "server"
            else BlockerCode.DEVICE_UNAVAILABLE
            if capability in {"cuda", "nccl"}
            else BlockerCode.ASSET_UNAVAILABLE
            if capability in {"model_assets", "data_assets", "tokenizer_assets"}
            else BlockerCode.DEPENDENCY_UNAVAILABLE
            if capability == "wheelhouse"
            else BlockerCode.CAPABILITY_UNAVAILABLE
        )
        blockers.append(
            TaskBlocker(
                code,
                capability,
                f"formal 外部条件缺少可验证的 capability commit：{capability}",
                True,
                (() if reference is None else (reference,)),
            )
        )
    if blockers:
        raise TaskBlockedError(*blockers)


def _formal_external_evidence(
    request: TaskExecutionRequest,
    root: Path,
    inputs: list[JSONValue],
) -> tuple[Mapping[str, JSONValue], tuple[str, ...]]:
    """把已核验的 formal 输入投影为 Stage 0/1 证据，不运行本机 fixture。"""

    refs: list[str] = []
    environment_items: list[JSONValue] = []
    for key, reference in sorted(request.environment.evidence_refs.items()):
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except Exception as error:
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.ASSET_UNAVAILABLE,
                    f"formal_evidence:{key}",
                    f"正式环境证据 commit 无法验证：{key}",
                    True,
                    (reference,),
                )
            ) from error
        refs.append(reference)
        environment_items.append(
            {
                "key": key,
                "ref": reference,
                "artifact_hash": loaded.identity.artifact_hash,
                "producer_task_id": loaded.identity.task_id,
                "artifact_kind": loaded.identity.artifact_kind,
            }
        )
    source_refs = tuple(dict.fromkeys(
        [str(item["ref"]) for item in inputs if isinstance(item, dict)] + refs
    ))
    return (
        {
            "evidence_type": "formal_committed_evidence",
            "task_id": request.task.task_id,
            "execution_mode": "formal_evidence_projection",
            "input_commits": inputs,
            "environment_commits": environment_items,
            "local_fixture_executed": False,
            "formal_gate_automatically_passed": False,
        },
        source_refs,
    )


def _run_formal_g3_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: tuple[str, ...],
) -> TaskRunResult:
    try:
        source_binding, data_root = _formal_g3_roots(root)
        resolution, producer_git_commit = _evaluate_current_formal_g3(
            source_binding,
            data_root,
        )
        payloads = _formal_g3_payloads(
            resolution,
            source_binding=source_binding,
            producer_git_commit=producer_git_commit,
        )
        bound_source_refs = _formal_g3_source_refs(
            source_binding,
            resolution,
            source_refs,
            producer_git_commit=producer_git_commit,
        )
    except Exception as error:
        raise RuntimeError(
            f"STAGE0_G3_FORMAL_EVALUATION_FAILED:{type(error).__name__}"
        ) from error

    refs: dict[str, str] = {}
    for artifact_kind in request.task.artifact_kinds:
        published = store.publish(
            task_id=request.task.task_id,
            artifact_kind=artifact_kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payloads[artifact_kind],
            formal_eligible=True,
            source_refs=bound_source_refs,
        )
        refs[artifact_kind] = published.commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        message="Stage 0 G3 assets resolved and all five sub-gates passed",
        metadata={
            "stage01_specialized": True,
            "g3_resolution_artifact_hash": resolution["artifact_hash"],
            "g3_gate_ids": list(G3_GATE_IDS),
        },
    )


def _tensor_values(value: TensorMap) -> dict[str, JSONValue]:
    return {
        name: tensor.detach().to(device="cpu", dtype=torch.float64).reshape(-1).tolist()
        for name, tensor in value.items()
    }


def _gradient_samples() -> list[TensorMap]:
    return [
        TensorMap({"weight": torch.tensor([1.0, -1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([3.0, 1.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([2.0, 0.0], dtype=torch.float64)}),
        TensorMap({"weight": torch.tensor([4.0, 2.0], dtype=torch.float64)}),
    ]


def _baseline_evidence(root: Path) -> Mapping[str, JSONValue]:
    source_root = root
    if not (source_root / "Agent").is_dir() or not (source_root / "plan").is_dir():
        source_root = Path(__file__).resolve().parents[3]
    hashes: dict[str, JSONValue] = {}
    for directory in ("Agent", "plan"):
        base = source_root / directory
        for path in sorted(base.rglob("*.md")):
            hashes[path.relative_to(source_root).as_posix()] = sha256_file(path)
    return {
        "evidence_type": "baseline_and_safety",
        "source_hashes": hashes,
        "source_count": len(hashes),
        "remote_execution_attempted": False,
        "server_state": "BLOCKED:server_unreachable",
        "sensitive_payload_policy": "runtime.events.reject_known_secret_patterns",
    }


def _storage_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    sandbox = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "storage-layout"
    sandbox.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_DIRECTORIES:
        (sandbox / name).mkdir(exist_ok=True)
    layout = StorageLayout(sandbox)
    escape_rejected = False
    try:
        layout.path("runs", "..", "outside")
    except ValueError:
        escape_rejected = True
    return {
        "evidence_type": "storage_layout",
        "required_directories": list(REQUIRED_DIRECTORIES),
        "validation_failures": layout.validate(require_writable=True),
        "escape_rejected": escape_rejected,
        "output_within_workspace": is_within(sandbox, root),
        "persistence_semantics": "immutable_object_plus_authoritative_commit",
    }


def _environment_evidence() -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "runtime_environment",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": sys.platform,
        "torch": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "dependency_mode": "local_installed_only_no_download",
        "optional_ml_dependencies": "lazy_import",
    }


def _asset_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    providers = request.config.section("providers")
    artifacts = request.config.section("artifacts")
    assert isinstance(providers, dict)
    assert isinstance(artifacts, dict)
    asset_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "asset-fixture"
    asset_root.mkdir(parents=True, exist_ok=True)
    asset_file = asset_root / "fixture.bin"
    content = b"stage01-safe-local-asset\n"
    if not asset_file.exists():
        atomic_write_bytes(asset_file, content)
    elif asset_file.read_bytes() != content:
        raise ValueError("STAGE01_ASSET_FIXTURE_DRIFT")
    descriptor = {
        "path": "fixture.bin",
        "size_bytes": len(content),
        "sha256": sha256_file(asset_file),
        "role": "contract_fixture",
    }
    manifest = build_manifest(
        asset_type=AssetType.SOURCE,
        name="stage01-contract-fixture",
        source="workspace-generated-fixture",
        revision="fixture-v1",
        files=(descriptor,),
        actor="stage01-runner",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref=None,
        generator_version="0.4.0",
        metadata={"source_kind": "contract_fixture", "license": "internal-test-only"},
        created_at="2026-01-01T00:00:00Z",
    )
    for state, role, minute in (
        (AssetState.DOWNLOADED, AssetActorRole.FETCHER, 1),
        (AssetState.VERIFIED, AssetActorRole.VERIFIER, 2),
        (AssetState.READY, AssetActorRole.GATE, 3),
    ):
        manifest = transition_manifest(
            manifest,
            state,
            actor="stage01-runner",
            actor_role=role,
            evidence_ref=(None if state is AssetState.DOWNLOADED else f"fixture/{state.value}.json"),
            summary=f"fixture {state.value}",
            at=f"2026-01-01T00:0{minute}:00Z",
        )
    verification = verify_only(manifest, asset_root)
    manifest_fields = sorted(key for key in providers if key.endswith("manifest_ref"))
    return {
        "evidence_type": "asset_manifest_boundary",
        "provider_kind": str(providers["kind"]),
        "manifest_fields": manifest_fields,
        "configured_manifest_refs": {
            key: providers[key] for key in manifest_fields if providers[key] is not None
        },
        "download_attempted": False,
        "fixture_asset_id": manifest["asset_id"],
        "fixture_asset_state": manifest["state"],
        "fixture_verification": verification,
        "legacy_bom_importer_separate": True,
        "canonical_json_requires_utf8_no_bom": True,
    }


def _identity_seed_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    identity = request.config.base_config.section("identity")
    model = request.config.base_config.section("model")
    assert isinstance(identity, dict) and isinstance(model, dict)
    master_seed = int(identity["master_seed"])
    seed_plan = SeedPlan.from_master_seed(master_seed, world_size=1)
    experiment_id = derive_experiment_id(
        stage=request.task.stage,
        task=request.task.task_id,
        model_identity=str(model["asset_id"]),
        route=str(identity["route"]),
        master_seed=master_seed,
        config_hash=request.config.config_hash,
    )
    run = RunIdentity.create(
        experiment_id=experiment_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        collision_code="00000001",
    )
    resumed = run.next_attempt(
        started_at=datetime(2026, 1, 1, 0, 1, tzinfo=timezone.utc),
        input_checkpoint_id="fixture-checkpoint-0001",
    )
    return {
        "evidence_type": "config_identity_seed",
        "config_hash": request.config.config_hash,
        "config_full_hash": request.config.full_hash,
        "run_identity": run.to_dict(),
        "resume_attempt": resumed.to_dict(),
        "seed_plan": seed_plan.to_dict(),
        "seed_domains_unique": len(set(seed_plan.domains.values()))
        == len(seed_plan.domains),
    }


def _observability_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    evidence_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "observability"
    evidence_root.mkdir(parents=True, exist_ok=True)
    event_path = evidence_root / "events.jsonl"
    if not event_path.exists():
        with JsonlEventSink(event_path) as sink:
            for sequence, step in enumerate((0, 1)):
                sink.append(
                    EventRecord.create(
                        experiment_id="stage0-observability",
                        run_id="fixture-run",
                        attempt_id="attempt-0001",
                        session_id="session-0001",
                        rank=0,
                        event_type=EventType.OPTIMIZER_STEP,
                        sequence=sequence,
                        event_id=f"stage0-event-{sequence:04d}",
                        occurred_at=f"2026-01-01T00:00:0{sequence}+00:00",
                        payload={
                            "global_step": step,
                            "microstep_count": 1,
                            "sample_count": 1,
                            "effective_token_count": 1,
                            "mean_loss": 1.0 / (step + 1),
                            "global_gradient_norm": 0.5,
                            "learning_rates_post_step": {"group_0000": 0.1},
                        },
                    ),
                    critical=True,
                )
    events = read_event_stream(event_path)
    canonical = canonical_optimizer_steps((events,))

    lineage = LineageStore(evidence_root / "lineage.json", run_id="fixture-run")
    if not lineage.records():
        lineage.register("attempt-0001", parent_attempt_id=None, reason="initial")
        lineage.register("attempt-0002", parent_attempt_id="attempt-0001", reason="retry")
        lineage.mark_orphan("attempt-0002", reason="incomplete")
        lineage.select_canonical(
            "attempt-0001", reason="complete event stream", evidence_hash=sha256_file(event_path)
        )
    dispositions = {
        item.attempt_id: item.disposition.value for item in lineage.records()
    }
    return {
        "evidence_type": "event_and_lineage",
        "event_count": len(events),
        "event_stream_sha256": sha256_file(event_path),
        "canonical_optimizer_steps": [int(item.payload["global_step"]) for item in canonical],
        "lineage_dispositions": dispositions,
        "single_writer": True,
        "expected_dispositions_present": sorted(
            {AttemptDisposition.CANONICAL.value, AttemptDisposition.ORPHAN.value}
        ),
    }


def _checkpoint_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    checkpoint_root = _resolve(root, str(artifacts["output_dir"]), field="output_dir") / "checkpoint-core"
    store = CheckpointStore(checkpoint_root)
    if not (store.commits / "state-0001.json").exists():
        store.publish(
            "state-0001",
            {"parameter": torch.tensor([1.0, 2.0]), "step": 1},
            generation=1,
            metadata={"boundary": "parameter_post_state"},
        )
    if not (store.commits / "state-0002.json").exists():
        state, _ = store.load("state-0001")
        store.publish(
            "state-0002",
            {"parameter": state["parameter"] + 2.0, "step": 2},
            generation=2,
            metadata={"boundary": "attempt_commit_state"},
            parent_checkpoint_id="state-0001",
        )
    resumed, commit = store.load("state-0002")
    uninterrupted = torch.tensor([3.0, 4.0])
    equivalent = bool(torch.equal(resumed["parameter"], uninterrupted))
    active_before = [item.checkpoint_id for item in store.discover()]
    if not (store.tombstones / "state-0001.json").exists():
        selection = store.select_retention(CheckpointRetentionPolicy(keep_latest=1))
        application = store.apply_retention(selection, reason="stage01 fixture retention")
        newly_tombstoned = list(application.newly_tombstoned)
    else:
        newly_tombstoned = []
    recovered_again, _ = CheckpointStore(checkpoint_root).load("state-0002")
    reconciliation = store.reconcile()
    return {
        "evidence_type": "checkpoint_resume",
        "checkpoint_id": commit.checkpoint_id,
        "generation": commit.generation,
        "bundle_manifest_sha256": commit.manifest_sha256,
        "active_before_retention": active_before,
        "active_after_retention": [item.checkpoint_id for item in store.discover()],
        "newly_tombstoned": newly_tombstoned,
        "objects_deleted": 0,
        "resume_equivalent": equivalent
        and bool(torch.equal(recovered_again["parameter"], uninterrupted)),
        "reconcile_invalid": reconciliation["invalid"],
        "two_phase_commit": True,
    }


def _capacity_evidence() -> Mapping[str, JSONValue]:
    parameter_count = 1024
    checkpoint_bytes = estimate_checkpoint_bytes(parameter_count)
    statistics_bytes = estimate_parameter_statistics_bytes(
        parameter_count, resident_fp32_buffers=3
    )
    total = estimate_experiment_storage(
        parameter_count=parameter_count,
        retained_checkpoints=2,
        resident_fp32_buffers=3,
        seed_count=2,
        parallel_runs=1,
        logs_and_reports_per_run=4096,
    )
    return {
        "evidence_type": "capacity_and_operations",
        "parameter_count": parameter_count,
        "checkpoint_bytes": checkpoint_bytes,
        "statistics_bytes": statistics_bytes,
        "experiment_storage_bytes": total,
        "actual_server_capacity": "BLOCKED:server_unreachable",
        "launcher_invoked": False,
    }


def _replay_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    payload = {
        "task_id": request.task.task_id,
        "config_hash": request.config.config_hash,
        "catalog_contract": request.task.to_dict(),
    }
    first = canonical_json_hash(payload)
    second = canonical_json_hash(dict(payload))
    return {
        "evidence_type": "deterministic_replay",
        "first_hash": first,
        "second_hash": second,
        "hashes_equal": first == second,
        "formal_gate_status": "NOT_RUN",
        "server_matrix_status": "BLOCKED:server_unreachable",
    }


def _delivery_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "delivery_and_sync",
        "expected_artifact_kinds": list(request.task.artifact_kinds),
        "immutable_publish_required": True,
        "worklog_language": "zh-CN",
        "local_delivery_ready": True,
        "github_push_status": "NOT_RUN",
        "server_sync_status": "BLOCKED:server_unreachable",
    }


def _stage1_contract_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    # 测试可把产物 workspace 指向临时目录；合同源文件仍必须来自已安装源码所属仓库，
    # 不能因为输出根不同就悄悄生成另一份数学合同。
    contract_root = root
    if not (contract_root / "docs" / "mathematics.md").is_file():
        contract_root = Path(__file__).resolve().parents[3]
    math_path = contract_root / "docs" / "mathematics.md"
    plan_path = contract_root / "plan" / "stage1" / "01_entry_and_contract.md"
    return {
        "evidence_type": "stage1_math_contract",
        "contract_hashes": {
            "docs/mathematics.md": sha256_file(math_path),
            "plan/stage1/01_entry_and_contract.md": sha256_file(plan_path),
        },
        "frozen_formulas": {
            "raw": "mean_gradient**2",
            "equal_u": "(S1**2-S2)/(M*(M-1))",
            "double": "mean_gradient_A*mean_gradient_B",
        },
        "dynamic_learning_rate_in_coordinate_hash": False,
        "same_batch_clipped_u_claim": "plugin_no_strict_unbiasedness",
        "task_definition_hash": canonical_json_hash(request.task.to_dict()),
    }


class _AliasModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.weight_alias = self.weight
        self.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))


def _registry_evidence() -> Mapping[str, JSONValue]:
    model = _AliasModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, foreach=False)
    registry = ParameterRegistry.from_model(model, optimizer)
    records = [
        {
            "canonical_name": record.canonical_name,
            "aliases": list(record.aliases),
            "shape": list(record.shape),
            "order": record.order,
            "eligible": record.eligible,
            "group_id": record.group_id,
        }
        for record in registry
    ]
    return {
        "evidence_type": "parameter_registry",
        "records": records,
        "eligible_names": list(registry.eligible_names),
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "optimizer_contract_hash": registry.optimizer_contract_hash,
        "runtime_layout_hash": registry.runtime_layout_hash,
        "alias_resolves_to": registry.canonical_name("weight_alias"),
    }


def _oracle_evidence() -> Mapping[str, JSONValue]:
    samples = _gradient_samples()
    mean = fp64_mean_gradient_oracle(samples)
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    production = equal_u_importance(statistics)
    oracle = fp64_equal_u_oracle(samples)
    comparison = compare_tensor_maps_fp64(production, oracle, natural_scale=4.0)
    return {
        "evidence_type": "fp64_fixture_oracle",
        "sample_count": len(samples),
        "mean_gradient": _tensor_values(mean),
        "u_oracle": _tensor_values(oracle),
        "comparison": comparison.to_dict(),
        "fixture_scope": "local_fixture",
    }


def _loss_scale_evidence() -> Mapping[str, JSONValue]:
    lm_logits = torch.tensor(
        [[[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [0.0, -1.0, 3.0]]],
        dtype=torch.float64,
    )
    lm_labels = torch.tensor([[0, 1, 2]])
    lm = causal_lm_loss(lm_logits, lm_labels, torch.tensor([[1, 1, 1]]))
    cls = sequence_classification_loss(
        torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float64),
        torch.tensor([0, 1, -100]),
    )
    merged = cls.merge(
        sequence_classification_loss(
            torch.tensor([[1.0, 0.0]], dtype=torch.float64), torch.tensor([0])
        )
    )
    return {
        "evidence_type": "loss_and_gradient_scale",
        "causal_lm": {
            "effective_count": lm.effective_count,
            "statistical_unit": lm.statistical_unit,
            "numerator": float(lm.loss_numerator.item()),
        },
        "classification": {
            "effective_count": cls.effective_count,
            "statistical_unit": cls.statistical_unit,
            "numerator": float(cls.loss_numerator.item()),
        },
        "merged_effective_count": merged.effective_count,
        "merge_uses_numerator_denominator": True,
    }


def _estimator_evidence() -> Mapping[str, JSONValue]:
    samples = _gradient_samples()
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    mean_a = fp64_mean_gradient_oracle(samples[:2])
    mean_b = fp64_mean_gradient_oracle(samples[2:])
    raw = raw_importance(statistics.mean_gradient)
    double = double_sample_importance(mean_a, mean_b)
    u = equal_u_importance(statistics)
    comparisons = {
        "raw": compare_tensor_maps_fp64(
            raw, fp64_raw_oracle(statistics.mean_gradient), natural_scale=4.0
        ).to_dict(),
        "double": compare_tensor_maps_fp64(
            double, fp64_double_sample_oracle(mean_a, mean_b), natural_scale=4.0
        ).to_dict(),
        "u": compare_tensor_maps_fp64(
            u, fp64_equal_u_oracle(samples), natural_scale=4.0
        ).to_dict(),
    }
    return {
        "evidence_type": "importance_estimators",
        "raw": _tensor_values(raw),
        "double": _tensor_values(double),
        "u": _tensor_values(u),
        "comparisons": comparisons,
        "u_can_be_negative": any(value < 0 for values in _tensor_values(u).values() for value in values),
        "unclipped_u_claim": "unbiased_fixed_state_under_declared_sampling_assumptions",
        "same_batch_clipped_u_claim": "plugin_same_batch_clip_no_strict_unbiasedness",
    }


def _numeric_boundary_evidence() -> Mapping[str, JSONValue]:
    attempt = GradientAttempt.capture(
        {"weight": torch.tensor([6.0, 8.0]), "unused": None},
        gradient_scale=2.0,
        scaled=True,
    ).unscale().check_finite().clip(2.5)
    skipped = GradientAttempt.capture(
        {"weight": torch.tensor([float("inf")])}, scaled=False
    ).check_finite()
    norm, factor = compute_global_clip_factor({"weight": torch.tensor([3.0, 4.0])}, 2.5)

    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW(
        [parameter], lr=0.1, weight_decay=0.2, foreach=False, fused=False
    )
    parameter.grad = torch.tensor([0.5, -0.25])
    outcome = OptimizerBridge({"weight": parameter}, optimizer).step()
    decomposition_error = (
        outcome.total_delta["weight"]
        - outcome.data_delta["weight"]
        - outcome.weight_decay_delta["weight"]
    ).abs().max()
    gradient_lifecycle: dict[str, JSONValue] = {
        "phase": attempt.phase.value,
        "gradient_scale": attempt.gradient_scale,
        "missing_names": list(attempt.missing_names),
        "global_norm": attempt.global_norm,
        "clip_factor": attempt.clip_factor,
        "skip_reason": attempt.skip_reason,
    }
    return {
        "evidence_type": "numeric_optimizer_boundary",
        "gradient_lifecycle": gradient_lifecycle,
        "nonfinite_phase": skipped.phase.value,
        "nonfinite_skip_reason": skipped.skip_reason,
        "global_norm": norm,
        "global_clip_factor": factor,
        "adamw_learning_rate": outcome.learning_rates["weight"],
        "adamw_decomposition_max_error": float(decomposition_error.item()),
        "foreach": False,
        "fused": False,
    }


def _report_evidence(request: TaskExecutionRequest) -> Mapping[str, JSONValue]:
    return {
        "evidence_type": "stage1_exit_report",
        "requirements": [
            "contract",
            "parameter_registry",
            "fp64_oracle",
            "loss_gradient_scale",
            "estimators",
            "training_integration",
            "ddp",
            "numeric_boundaries",
            "checkpoint_resume",
        ],
        "formal_gate_status": "NOT_RUN",
        "local_evidence_only": request.config.run_intent == "local_fixture",
        "server_validation": "BLOCKED:server_unreachable",
        "manual_numbers_allowed": False,
    }


def _task_evidence(request: TaskExecutionRequest, root: Path) -> Mapping[str, JSONValue]:
    task_id = request.task.task_id
    if task_id == "stage0.01_baseline_and_safety":
        return _baseline_evidence(root)
    if task_id == "stage0.02_storage_and_layout":
        return _storage_evidence(request, root)
    if task_id == "stage0.03_runtime_and_dependencies":
        return _environment_evidence()
    if task_id == "stage0.04_assets_and_manifests":
        return _asset_evidence(request, root)
    if task_id == "stage0.05_config_run_identity_and_seeds":
        return _identity_seed_evidence(request)
    if task_id == "stage0.08_logging_and_tracking":
        return _observability_evidence(request, root)
    if task_id in {"stage0.09_checkpoint_and_resume", "stage1.10_checkpoint_resume_and_artifacts"}:
        return _checkpoint_evidence(request, root)
    if task_id == "stage0.10_capacity_and_operations":
        return _capacity_evidence()
    if task_id == "stage0.11_test_quality_and_replay":
        return _replay_evidence(request)
    if task_id == "stage0.12_delivery_and_sync":
        return _delivery_evidence(request)
    if task_id == "stage1.01_entry_and_contract":
        return _stage1_contract_evidence(request, root)
    if task_id == "stage1.02_architecture_and_parameter_registry":
        return _registry_evidence()
    if task_id == "stage1.03_fixtures_and_oracles":
        return _oracle_evidence()
    if task_id == "stage1.04_loss_and_gradient_scale":
        return _loss_scale_evidence()
    if task_id == "stage1.05_estimators":
        return _estimator_evidence()
    if task_id == "stage1.09_precision_clipping_and_optimizer_boundaries":
        return _numeric_boundary_evidence()
    if task_id == "stage1.11_reporting_and_exit_gate":
        return _report_evidence(request)
    raise ValueError(f"STAGE01_TASK_UNHANDLED:{task_id}")


def _role_evidence(task_id: str, artifact_kind: str) -> Mapping[str, JSONValue]:
    """为同一任务中的不同产物声明不同消费语义，禁止复制同一 payload 冒充分区。"""

    if "gate" in artifact_kind:
        return {"role": "gate_candidate", "gate_status": "NOT_RUN", "decision_authority": "external_reviewer"}
    if "report" in artifact_kind or "summary" in artifact_kind:
        return {"role": "derived_report", "source": "task_core_evidence", "manual_numeric_edits": False}
    if "manifest" in artifact_kind or "registry" in artifact_kind or "plan" in artifact_kind:
        return {"role": "frozen_manifest", "canonical_order": True, "identity_bound": True}
    if "checkpoint" in artifact_kind or "state" in artifact_kind:
        return {"role": "recovery_boundary", "authoritative_commit_required": True, "directory_rename_is_commit": False}
    if "event" in artifact_kind or "lineage" in artifact_kind:
        return {"role": "append_only_runtime_evidence", "single_writer": True, "canonical_selection_required": True}
    if "sync" in artifact_kind or "delivery" in artifact_kind or artifact_kind == "worklog":
        return {"role": "delivery_record", "local_only": True, "server_sync_claimed": False}
    return {"role": "task_specific_evidence", "artifact_contract": f"{task_id}:{artifact_kind}"}


@dataclass(slots=True)
class Stage01CompositeTaskRunner(TaskRunner):
    """同一 ``RunnerKind`` 下按 task ID 分派 Stage 0/1 专用逻辑。"""

    runner_kind: RunnerKind
    workspace_root: Path
    fallback: TaskRunner | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    @property
    def handled_task_ids(self) -> frozenset[str]:
        return _HANDLED_BY_KIND.get(self.runner_kind, frozenset())

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        if request.task.runner_kind is not self.runner_kind:
            raise ValueError("STAGE01_RUNNER_KIND_MISMATCH")
        if request.task.task_id not in self.handled_task_ids:
            if self.fallback is not None:
                return self.fallback.run(request)
            raise TaskBlockedError(
                TaskBlocker(
                    BlockerCode.CAPABILITY_UNAVAILABLE,
                    f"fallback:{self.runner_kind.value}",
                    f"Stage 0/1 composite 不处理任务 {request.task.task_id}",
                    False,
                )
            )
        _formal_guard(request, self.workspace_root)
        if (
            request.config.run_intent == "formal"
            and request.task.task_id == _G3_TASK_ID
        ):
            # Fail before TaskArtifactStore creates any output directory when
            # the CLI/runtime accidentally points formal execution at source.
            _formal_g3_roots(self.workspace_root)
        store = _store(request, self.workspace_root)
        existing = store.discover_complete(
            task_id=request.task.task_id,
            config_hash=request.config.config_hash,
            artifact_kinds=request.task.artifact_kinds,
            formal_eligible=request.config.run_intent == "formal",
        )
        if existing is not None:
            if (
                request.config.run_intent == "formal"
                and request.task.task_id == _G3_TASK_ID
            ):
                _, restore_input_source_refs = _input_evidence(
                    request,
                    self.workspace_root,
                )
                resolution = _restore_formal_g3_outputs(
                    self.workspace_root,
                    existing,
                    input_source_refs=restore_input_source_refs,
                )
                return TaskRunResult.passed(
                    request,
                    artifact_refs=existing,
                    message="Stage 0 G3 assets restored from validated formal commits",
                    metadata={
                        "stage01_specialized": True,
                        "restored": True,
                        "g3_resolution_artifact_hash": resolution[
                            "artifact_hash"
                        ],
                        "g3_gate_ids": list(G3_GATE_IDS),
                    },
                )
            if (
                request.config.run_intent == "formal"
                and request.task.task_id == _G4_TASK_ID
            ):
                from ..stage0_g4 import validate_formal_g4_outputs

                gate = validate_formal_g4_outputs(
                    request,
                    self.workspace_root,
                    existing,
                )
                return TaskRunResult.passed(
                    request,
                    artifact_refs=existing,
                    message="Stage 0 G4 restored from revalidated formal commits",
                    metadata={
                        "stage0_g4_specialized": True,
                        "restored": True,
                        "gate_id": gate.gate_id,
                    },
                )
            if (
                request.config.run_intent == "formal"
                and request.task.task_id == "stage0.08_logging_and_tracking"
            ):
                from ..stage0_g7 import validate_formal_g7_outputs

                gate = validate_formal_g7_outputs(
                    request,
                    self.workspace_root,
                    existing,
                )
                return TaskRunResult.passed(
                    request,
                    artifact_refs=existing,
                    message="Stage 0 S0.8 restored from revalidated formal commits",
                    metadata={
                        "stage0_g7_logging_specialized": True,
                        "restored": True,
                        "component_gate_id": gate.gate_id,
                    },
                )
            if (
                request.config.run_intent == "formal"
                and request.task.task_id == "stage0.09_checkpoint_and_resume"
            ):
                from ..stage0_g7_recovery import validate_formal_g7_recovery_outputs

                gate = validate_formal_g7_recovery_outputs(
                    request,
                    self.workspace_root,
                    existing,
                )
                return TaskRunResult.passed(
                    request,
                    artifact_refs=existing,
                    checkpoint_ref=existing["checkpoint_commit"],
                    message="Stage 0 S0.9 restored from revalidated formal commits",
                    metadata={
                        "stage0_g7_recovery_specialized": True,
                        "restored": True,
                        "gate_id": gate.gate_id,
                    },
                )
            if (
                request.config.run_intent == "formal"
                and request.task.task_id in _FORMAL_EVIDENCE_ONLY_TASKS
            ):
                for reference in existing.values():
                    loaded = load_committed_task_artifact(
                        self.workspace_root,
                        reference,
                        require_formal=True,
                    )
                    core = loaded.payload.get("core_evidence")
                    if not isinstance(core, Mapping) or (
                        core.get("evidence_type") != "formal_committed_evidence"
                        or core.get("local_fixture_executed") is not False
                    ):
                        raise TaskBlockedError(
                            TaskBlocker(
                                BlockerCode.CONTRACT_UNFROZEN,
                                "formal_stage01_output_scope",
                                "已有输出不是 formal evidence-only 路径，请使用新的输出目录",
                                False,
                                (reference,),
                            )
                        )
            return TaskRunResult.passed(
                request,
                artifact_refs=existing,
                checkpoint_ref=next(
                    (ref for kind, ref in existing.items() if "checkpoint" in kind or "state" in kind),
                    None,
                ),
                message="stage0-1 task restored from authoritative commits",
                metadata={"stage01_specialized": True, "restored": True},
            )

        inputs, source_refs = _input_evidence(request, self.workspace_root)
        if (
            request.config.run_intent == "formal"
            and request.task.task_id == _G3_TASK_ID
        ):
            return _run_formal_g3_task(
                request,
                self.workspace_root,
                store,
                source_refs=source_refs,
            )
        if (
            request.config.run_intent == "formal"
            and request.task.task_id == _G4_TASK_ID
        ):
            from ..stage0_g4 import run_formal_g4_task

            return run_formal_g4_task(
                request,
                self.workspace_root,
                store,
                source_refs=source_refs,
            )
        if (
            request.config.run_intent == "formal"
            and request.task.task_id == "stage0.08_logging_and_tracking"
        ):
            from ..stage0_g7 import run_formal_g7_task

            return run_formal_g7_task(
                request,
                self.workspace_root,
                store,
                source_refs=source_refs,
            )
        if (
            request.config.run_intent == "formal"
            and request.task.task_id == "stage0.09_checkpoint_and_resume"
        ):
            from ..stage0_g7_recovery import run_formal_g7_recovery_task

            return run_formal_g7_recovery_task(
                request,
                self.workspace_root,
                store,
                source_refs=source_refs,
            )
        if (
            request.config.run_intent == "formal"
            and request.task.task_id in _FORMAL_EVIDENCE_ONLY_TASKS
        ):
            formal_evidence, formal_sources = _formal_external_evidence(
                request,
                self.workspace_root,
                inputs,
            )
            evidence = dict(formal_evidence)
            source_refs = formal_sources
        else:
            evidence = dict(_task_evidence(request, self.workspace_root))
        evidence_hash = canonical_json_hash(evidence)
        refs: dict[str, str] = {}
        for artifact_kind in request.task.artifact_kinds:
            payload: dict[str, JSONValue] = {
                "schema_version": "stage01-task-evidence-v1",
                "task_id": request.task.task_id,
                "artifact_role": artifact_kind,
                "scope": request.config.run_intent,
                "local_validation_status": (
                    "PASS" if request.config.run_intent == "local_fixture" else "NOT_RUN"
                ),
                "gate_status": "NOT_RUN",
                "config_hash": request.config.config_hash,
                "task_definition_hash": canonical_json_hash(request.task.to_dict()),
                "input_evidence": inputs,
                "core_evidence_hash": evidence_hash,
                "core_evidence": evidence,
                "role_evidence": dict(_role_evidence(request.task.task_id, artifact_kind)),
            }
            published = store.publish(
                task_id=request.task.task_id,
                artifact_kind=artifact_kind,
                config_hash=request.config.config_hash,
                run_intent=request.config.run_intent,
                payload=payload,
                formal_eligible=request.config.run_intent == "formal",
                source_refs=source_refs,
            )
            refs[artifact_kind] = published.commit_ref
        return TaskRunResult.passed(
            request,
            artifact_refs=refs,
            checkpoint_ref=next(
                (ref for kind, ref in refs.items() if "checkpoint" in kind or "state" in kind),
                None,
            ),
            message="stage0-1 specialized core task completed",
            metadata={"stage01_specialized": True, "core_evidence_hash": evidence_hash},
        )


def build_stage01_runner_overrides(
    workspace_root: str | Path,
    *,
    fallbacks: Mapping[RunnerKind, TaskRunner] | None = None,
) -> Mapping[RunnerKind, Stage01CompositeTaskRunner]:
    """构造 ``RunnerKind -> composite`` 覆盖映射，供统一 runtime 工厂逐层组合。

    未命中 Stage 0/1 task ID 时会转交同 kind 的 ``fallback``；因此 CONTRACT、
    VALIDATION、REPORTING 等共享 kind 能继续被 Stage 2--9 的专用 runner 消费。
    """

    root = Path(workspace_root).resolve()
    fallback_map = dict(fallbacks or {})
    return MappingProxyType(
        {
            kind: Stage01CompositeTaskRunner(kind, root, fallback_map.get(kind))
            for kind in _HANDLED_BY_KIND
        }
    )


__all__ = [
    "STAGE01_HANDLED_TASK_IDS",
    "Stage01CompositeTaskRunner",
    "build_stage01_runner_overrides",
]
