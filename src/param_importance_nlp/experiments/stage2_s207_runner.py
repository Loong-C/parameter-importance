"""Production S2.7 worker and detached queue.

This module is intentionally separate from the shared Stage 2 task runner.  It
is the execution adapter for the already frozen S2.7 plan: it reads the
S2.6/G2.4b mapping, constructs one real fixed-state provider for a selected
S2.3 checkpoint, and executes each repetition exactly once through
``RecoverablePairedWaveRunner``.  The worker never generates a draw, performs a
statistical reduction, or decides whether an estimator is useful.

The durable boundary is an attempt marker followed by one terminal raw-unit
record.  A committed paired-wave object can be recovered into a raw-unit
record; an abandoned attempt is converted to an explicit failure on recovery,
never silently retried.  The full reducer is deliberately the strict reducer
from :mod:`stage2_s207_formal` and is called only after all six cell waves have
terminal unit records.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..core.registry import ParameterRegistry
from ..g3_runtime_assets import FormalG3RuntimeAssets, G3RuntimeAssetError, formal_pile_route
from ..providers import (
    FixedStateGradientProvider,
    OfflineHuggingFaceModelAdapter,
    PythiaMMapFrozenSampleResolver,
    PretokenizedGlueDatasetAdapter,
    TorchFixedStateGradientProvider,
)
from ..runtime.task_artifacts import load_committed_task_artifact
from ..runtime.task_runtime import TaskExecutionRequest, TaskRuntimeEnvironment
from ..runtime.tensor_bundle import load_tensor_bundle
from ..experiments.stage2_formal import RecoverablePairedWaveRunner, _vector_digest
from ..experiments.sampling import RepetitionMapping
from .stage2_s204_ids import EXPECTED_CELL_IDS
from .stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_PCI,
    S27CellPlan,
    S27G25Blocked,
    S27MappingUnit,
    S27Plan,
    S27RawUnit,
    S27StatusStore,
    StrictG25Reducer,
    validate_gpu_inventory,
)


S27_EXECUTION_SCHEMA = "stage2-s27-production-execution-v1"
S27_ATTEMPT_SCHEMA = "stage2-s27-unit-attempt-v1"
S27_WAVE_SEAL_SCHEMA = "stage2-s27-wave-seal-v1"
S27_LAUNCH_SCHEMA = "stage2-s27-detached-launch-v1"
S27_LAUNCH_STATUS_SCHEMA = "stage2-s27-launch-status-v1"
S27_RAW_RESULT_SCHEMA = "stage2-s27-raw-result-v1"
S27_DEFAULT_MAX_ATTEMPTS = 1
S27_DEFAULT_M2_TOLERANCE = 1e-10
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,159}$")


class S27ExecutionBlocked(RuntimeError):
    """Raised when a production input or recovery boundary is not safe."""


class S27ProviderFactory(Protocol):
    def __call__(
        self,
        cell: S27CellPlan,
        *,
        data_root: Path,
        checkpoint_root_ref: str,
        registry_ref: str,
        request: TaskExecutionRequest | None = None,
    ) -> "S27ProviderContext": ...


@dataclass(frozen=True, slots=True)
class S27ProviderContext:
    """The only provider state accepted by a production cell worker."""

    provider: FixedStateGradientProvider
    execution: FormalExecutionEvidence
    checkpoint_root_ref: str
    checkpoint_hash: str
    registry_ref: str
    registry_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.provider, FixedStateGradientProvider):
            raise TypeError("S27_PROVIDER_MUST_IMPLEMENT_FIXED_STATE_CONTRACT")
        if not isinstance(self.execution, FormalExecutionEvidence):
            raise TypeError("S27_FORMAL_EXECUTION_EVIDENCE_REQUIRED")
        if not _SHA256.fullmatch(self.checkpoint_hash):
            raise ValueError("S27_CHECKPOINT_HASH_INVALID")
        if not _SHA256.fullmatch(self.registry_hash):
            raise ValueError("S27_REGISTRY_HASH_INVALID")


@dataclass(frozen=True, slots=True)
class S27RetryPolicy:
    """Frozen retry policy.

    A confirmatory unit has one attempt.  An OOM retry is permitted only if an
    upstream frozen artifact explicitly names that unit and the caller supplies
    a smaller-forward provider; this module does not invent such a provider.
    Consequently the default policy is intentionally no-retry.
    """

    max_attempts: int = S27_DEFAULT_MAX_ATTEMPTS
    pre_registered_oom_unit_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or self.max_attempts != 1:
            raise S27ExecutionBlocked("S27_RETRY_POLICY_MUST_BE_SINGLE_ATTEMPT")
        if len(set(self.pre_registered_oom_unit_ids)) != len(self.pre_registered_oom_unit_ids):
            raise S27ExecutionBlocked("S27_RETRY_POLICY_DUPLICATE_UNIT")
        if self.pre_registered_oom_unit_ids:
            # There is deliberately no implicit smaller-forward implementation.
            raise S27ExecutionBlocked("S27_OOM_RETRY_REQUIRES_EXPLICIT_SMALLER_FORWARD_ADAPTER")


def _safe_ref(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S27ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE")
    current = root
    if root.is_symlink():
        raise S27ExecutionBlocked(f"{field}:SYMLINK_ROOT")
    for part in logical.parts:
        current = current / part
        if current.is_symlink():
            raise S27ExecutionBlocked(f"{field}:SYMLINK_COMPONENT")
    target = (root.joinpath(*logical.parts)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    return target


def _relative_ref(root: Path, path: Path, *, field: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise S27ExecutionBlocked(f"{field}:OUTSIDE_DATA_ROOT") from error
    if not relative or ".." in PurePosixPath(relative).parts:
        raise S27ExecutionBlocked(f"{field}:INVALID_RELATIVE_REFERENCE")
    return relative


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_payload(value: Mapping[str, object], *, field: str) -> tuple[dict[str, object], str]:
    body = {str(key): item for key, item in value.items() if key != "artifact_hash"}
    try:
        digest = canonical_json_hash(body)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"{field}:NOT_CANONICAL_JSON") from error
    declared = value.get("artifact_hash")
    if declared != digest:
        raise S27ExecutionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return dict(value), digest


def _load_payload(root: Path, reference: str, *, field: str) -> tuple[dict[str, object], str, str]:
    """Load either a direct canonical payload or a formal task commit."""

    path = _safe_ref(root, reference, field=field)
    raw = load_canonical_json(path)
    if not isinstance(raw, Mapping):
        raise S27ExecutionBlocked(f"{field}:OBJECT_REQUIRED")
    if raw.get("schema_version") == "task-output-commit-v1":
        try:
            loaded = load_committed_task_artifact(root, reference, require_formal=True)
        except (OSError, TypeError, ValueError) as error:
            raise S27ExecutionBlocked(f"{field}:FORMAL_TASK_COMMIT_INVALID") from error
        payload = loaded.payload
        if not isinstance(payload, Mapping):
            raise S27ExecutionBlocked(f"{field}:TASK_PAYLOAD_INVALID")
        return dict(payload), loaded.identity.artifact_hash, reference
    payload, digest = _canonical_payload(raw, field=field)
    return payload, digest, reference


def load_s27_plan(data_root: str | Path, plan_ref: str) -> S27Plan:
    root = Path(data_root).resolve()
    payload, _, _ = _load_payload(root, plan_ref, field="s27_plan")
    try:
        return S27Plan.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_PLAN_INVALID:{error}") from error


def load_s27_frozen_mappings(
    data_root: str | Path,
    plan: S27Plan,
    *,
    cell_id: str | None = None,
) -> dict[str, RepetitionMapping]:
    """Read the exact S2.6 mappings; this function never calls a draw API."""

    root = Path(data_root).resolve()
    payload, digest, _ = _load_payload(root, plan.frozen_inputs.mapping_ref, field="s27_mapping")
    if digest != plan.frozen_inputs.mapping_hash:
        raise S27ExecutionBlocked("S27_MAPPING_HASH_DRIFT")
    if payload.get("schema_version") != "stage2-formal-confirmatory-mapping-v1":
        raise S27ExecutionBlocked("S27_MAPPING_SCHEMA_INVALID")
    if payload.get("scope") != "formal" or payload.get("stream") != "confirmatory" or payload.get("formal_eligible") is not True or payload.get("complete") is not True or payload.get("draw_id_unique") is not True:
        raise S27ExecutionBlocked("S27_MAPPING_FORMAL_SCOPE_INVALID")
    if payload.get("freeze_hash") != plan.frozen_inputs.matrix_hash or payload.get("qualification_gate_hash") != plan.frozen_inputs.g24b_gate_hash or payload.get("sampling_plan_hash") != plan.frozen_inputs.sampling_plan_hash:
        raise S27ExecutionBlocked("S27_MAPPING_FREEZE_LINEAGE_INVALID")
    result: dict[str, RepetitionMapping] = {}
    observed_draw_ids: set[str] = set()
    observed_positions: set[tuple[str, int]] = set()
    expected_units = {
        unit.unit_id: unit
        for unit in plan.frozen_inputs.units
        if cell_id is None or unit.cell_id == cell_id
    }
    cells = payload.get("cells")
    if not isinstance(cells, list):
        raise S27ExecutionBlocked("S27_MAPPING_CELLS_REQUIRED")
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise S27ExecutionBlocked("S27_MAPPING_CELL_NOT_OBJECT")
        anchor = cell.get("anchor_id")
        if not isinstance(anchor, str) or (cell_id is not None and anchor != cell_id):
            if cell_id is None:
                raise S27ExecutionBlocked("S27_MAPPING_ANCHOR_INVALID")
            continue
        rows = cell.get("mappings")
        if not isinstance(rows, list):
            raise S27ExecutionBlocked(f"S27_MAPPING_ROWS_REQUIRED:{anchor}")
        for raw in rows:
            if not isinstance(raw, Mapping):
                raise S27ExecutionBlocked(f"S27_MAPPING_ROW_NOT_OBJECT:{anchor}")
            try:
                mapping = RepetitionMapping.from_manifest(raw)
            except (TypeError, ValueError) as error:
                raise S27ExecutionBlocked(f"S27_MAPPING_REPETITION_INVALID:{anchor}") from error
            unit_id = f"{anchor}::{mapping.repetition_id}"
            expected = expected_units.get(unit_id)
            if expected is None:
                raise S27ExecutionBlocked(f"S27_MAPPING_UNEXPECTED_UNIT:{unit_id}")
            if mapping.digest != expected.mapping_hash:
                raise S27ExecutionBlocked(f"S27_MAPPING_UNIT_HASH_DRIFT:{unit_id}")
            if mapping.batch_size != plan.frozen_inputs.batch_size or mapping.m_values != (2, plan.frozen_inputs.microbatch_count):
                raise S27ExecutionBlocked(f"S27_MAPPING_B_M_DRIFT:{unit_id}")
            for draw in mapping.draws:
                position = (str(draw.stream), int(draw.position))
                if draw.draw_id in observed_draw_ids or position in observed_positions:
                    raise S27ExecutionBlocked(f"S27_MAPPING_GLOBAL_DRAW_COLLISION:{unit_id}")
                observed_draw_ids.add(str(draw.draw_id))
                observed_positions.add(position)
            if tuple(draw.draw_id for draw in mapping.draws) != expected.draw_ids or tuple(draw.sample_id for draw in mapping.draws) != expected.sample_ids:
                raise S27ExecutionBlocked(f"S27_MAPPING_DRAW_SAMPLE_DRIFT:{unit_id}")
            if unit_id in result:
                raise S27ExecutionBlocked(f"S27_MAPPING_DUPLICATE_UNIT:{unit_id}")
            result[unit_id] = mapping
    if set(result) != set(expected_units):
        missing = sorted(set(expected_units) - set(result))
        raise S27ExecutionBlocked(f"S27_MAPPING_MISSING_UNIT:{','.join(missing[:8])}")
    return result


def load_s27_reference_views(
    data_root: str | Path,
    cell: S27CellPlan,
    *,
    expected_registry_hash: str | None = None,
    reference_output_root_ref: str | None = None,
) -> dict[str, Mapping[str, object]]:
    """Load the independent S2.4 reference bundle and its G2.3 PASS gate."""

    root = Path(data_root).resolve()
    reference, reference_digest, reference_ref = _load_payload(root, cell.reference_ref, field=f"reference.{cell.cell_id}")
    if reference_digest != cell.reference_hash:
        raise S27ExecutionBlocked(f"S27_REFERENCE_ARTIFACT_HASH_MISMATCH:{cell.cell_id}")
    try:
        gate_payload, gate_hash, _ = _load_payload(root, cell.reference_gate_ref, field=f"reference_gate.{cell.cell_id}")
        gate = GateRecord.from_mapping(dict(gate_payload))
    except (OSError, TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_INVALID:{cell.cell_id}") from error
    if gate_hash != cell.reference_gate_hash or gate.gate_id != "stage2.G2.3" or gate.status is not GateStatus.PASS:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_NOT_PASS:{cell.cell_id}")
    if cell.reference_ref not in gate.evidence_refs:
        raise S27ExecutionBlocked(f"S27_REFERENCE_GATE_DOES_NOT_BIND_REFERENCE:{cell.cell_id}")
    if reference.get("schema_version") != "reference-result-v1" or reference.get("scope") != "formal":
        raise S27ExecutionBlocked(f"S27_REFERENCE_SCOPE_INVALID:{cell.cell_id}")
    if expected_registry_hash is not None and reference.get("registry_hash") != expected_registry_hash:
        raise S27ExecutionBlocked(f"S27_REFERENCE_REGISTRY_MISMATCH:{cell.cell_id}")
    bundle_ref = reference.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str):
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_REF_MISSING:{cell.cell_id}")
    bundle_candidates: list[Path] = []
    try:
        bundle_candidates.append(_safe_ref(root, bundle_ref, field=f"reference_bundle.{cell.cell_id}"))
    except S27ExecutionBlocked:
        pass
    if reference_output_root_ref is not None:
        _safe_ref(root, reference_output_root_ref, field=f"reference_output_root.{cell.cell_id}")
        logical_bundle = PurePosixPath(reference_output_root_ref) / PurePosixPath(bundle_ref)
        bundle_candidates.append(
            _safe_ref(root, logical_bundle.as_posix(), field=f"reference_bundle.{cell.cell_id}.output_root")
        )
    reference_path = _safe_ref(root, reference_ref, field=f"reference.{cell.cell_id}")
    try:
        logical_parent = PurePosixPath(reference_ref).parent / PurePosixPath(bundle_ref)
        bundle_candidates.append(
            _safe_ref(root, logical_parent.as_posix(), field=f"reference_bundle.{cell.cell_id}.parent")
        )
    except (ValueError, S27ExecutionBlocked):
        pass
    bundle_path = next(
        (
            candidate
            for candidate in bundle_candidates
            if candidate.is_dir() and not candidate.is_symlink()
        ),
        None,
    )
    if bundle_path is None:
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_MISSING:{cell.cell_id}")
    try:
        state, bundle = load_tensor_bundle(bundle_path)
    except (OSError, TypeError, ValueError) as error:
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_INVALID:{cell.cell_id}") from error
    if bundle.manifest_sha256 != reference.get("tensor_bundle_manifest_hash") or not isinstance(state, Mapping):
        raise S27ExecutionBlocked(f"S27_REFERENCE_BUNDLE_HASH_MISMATCH:{cell.cell_id}")
    expected_names = {"bias_reference", "cross_reference", "ranking_reference"}
    if not expected_names.issubset(set(state)):
        raise S27ExecutionBlocked(f"S27_REFERENCE_VIEWS_INVALID:{cell.cell_id}")
    views: dict[str, Mapping[str, object]] = {}
    for short, long_name in (("bias", "bias_reference"), ("cross", "cross_reference"), ("ranking", "ranking_reference")):
        value = state[long_name]
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_REFERENCE_VIEW_NOT_MAPPING:{cell.cell_id}:{short}")
        declared = reference.get(f"{short}_reference_hash")
        if declared != _vector_digest(value):
            raise S27ExecutionBlocked(f"S27_REFERENCE_VIEW_HASH_MISMATCH:{cell.cell_id}:{short}")
        views[short] = value
    return views


@dataclass(frozen=True, slots=True)
class S27MaterializedCellInput:
    """S2.4 materialization row consumed by the S2.7 worker."""

    cell_id: str
    config_ref: str
    environment_ref: str
    checkpoint_root_ref: str
    registry_ref: str
    reference_output_root_ref: str = ""

    def __post_init__(self) -> None:
        if self.cell_id not in EXPECTED_CELL_IDS:
            raise ValueError("S27_MATERIALIZED_CELL_UNKNOWN")
        for field, value in (("config_ref", self.config_ref), ("environment_ref", self.environment_ref), ("checkpoint_root_ref", self.checkpoint_root_ref), ("registry_ref", self.registry_ref)):
            if not isinstance(value, str) or not value or "\\" in value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts:
                raise ValueError(f"S27_MATERIALIZED_{field.upper()}_INVALID")
        if self.reference_output_root_ref and ("\\" in self.reference_output_root_ref or PurePosixPath(self.reference_output_root_ref).is_absolute() or ".." in PurePosixPath(self.reference_output_root_ref).parts):
            raise ValueError("S27_MATERIALIZED_REFERENCE_OUTPUT_ROOT_REF_INVALID")


def load_s27_materialized_inputs(data_root: str | Path, index_ref: str) -> dict[str, S27MaterializedCellInput]:
    root = Path(data_root).resolve()
    index_path = _safe_ref(root, index_ref, field="s27_materialization_index")
    raw_index = load_canonical_json(index_path)
    if not isinstance(raw_index, Mapping):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_OBJECT_REQUIRED")
    if raw_index.get("schema_version") == "task-output-commit-v1":
        payload, _, _ = _load_payload(root, index_ref, field="s27_materialization_index")
    else:
        declared_index_hash = raw_index.get("index_hash")
        index_body = {key: item for key, item in raw_index.items() if key != "index_hash"}
        if declared_index_hash != canonical_json_hash(index_body):
            raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_HASH_INVALID")
        payload = dict(raw_index)
    if payload.get("schema_version") != "stage2-s204-six-cell-materialization-index-v1":
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_SCHEMA_INVALID")
    rows = payload.get("cells")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_CELL_IDS):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_INDEX_SIX_CELL_REQUIRED")
    manifest_ref = payload.get("six_cell_manifest_ref")
    if not isinstance(manifest_ref, str):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_MANIFEST_REF_MISSING")
    manifest_path = _safe_ref(root, manifest_ref, field="s27_six_cell_manifest")
    raw_manifest = load_canonical_json(manifest_path)
    if not isinstance(raw_manifest, Mapping):
        raise S27ExecutionBlocked("S27_SIX_CELL_MANIFEST_OBJECT_REQUIRED")
    if raw_manifest.get("schema_version") == "task-output-commit-v1":
        manifest, _, _ = _load_payload(root, manifest_ref, field="s27_six_cell_manifest")
    else:
        declared_manifest_hash = raw_manifest.get("manifest_hash")
        manifest_body = {key: item for key, item in raw_manifest.items() if key != "manifest_hash"}
        if declared_manifest_hash != canonical_json_hash(manifest_body):
            raise S27ExecutionBlocked("S27_SIX_CELL_MANIFEST_HASH_INVALID")
        manifest = dict(raw_manifest)
    checkpoint_rows = manifest.get("checkpoints")
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != len(EXPECTED_CELL_IDS):
        raise S27ExecutionBlocked("S27_SIX_CELL_CHECKPOINT_ROWS_INVALID")
    checkpoint_by_cell: dict[str, Mapping[str, object]] = {}
    for checkpoint in checkpoint_rows:
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("cell_id"), str):
            raise S27ExecutionBlocked("S27_SIX_CELL_CHECKPOINT_ROW_INVALID")
        checkpoint_by_cell[str(checkpoint["cell_id"])] = checkpoint
    result: dict[str, S27MaterializedCellInput] = {}
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise S27ExecutionBlocked("S27_MATERIALIZATION_ROW_INVALID")
        cell_id = raw.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in result:
            raise S27ExecutionBlocked("S27_MATERIALIZATION_CELL_SET_INVALID")
        checkpoint = checkpoint_by_cell.get(cell_id)
        if checkpoint is None or not isinstance(checkpoint.get("checkpoint_root_ref"), str):
            raise S27ExecutionBlocked(f"S27_CHECKPOINT_ROOT_REF_MISSING:{cell_id}")
        config_ref = raw.get("config_ref")
        if not isinstance(config_ref, str):
            raise S27ExecutionBlocked(f"S27_CONFIG_REF_MISSING:{cell_id}")
        environment_ref = raw.get("environment_ref")
        if not isinstance(environment_ref, str) or not environment_ref:
            raise S27ExecutionBlocked(f"S27_ENVIRONMENT_REF_MISSING:{cell_id}")
        registry_ref = raw.get("registry_ref", raw.get("parameter_registry_ref"))
        if not isinstance(registry_ref, str) or not registry_ref:
            raise S27ExecutionBlocked(f"S27_REGISTRY_REF_MISSING:{cell_id}")
        config_path = _safe_ref(root, config_ref, field=f"config.{cell_id}")
        config_value = load_canonical_json(config_path)
        if not isinstance(config_value, Mapping):
            raise S27ExecutionBlocked(f"S27_CONFIG_OBJECT_REQUIRED:{cell_id}")
        artifacts = config_value.get("artifacts")
        if not isinstance(artifacts, Mapping) or not isinstance(artifacts.get("output_dir"), str):
            raise S27ExecutionBlocked(f"S27_CONFIG_OUTPUT_DIR_MISSING:{cell_id}")
        # The six-cell manifest is the authority for the selected checkpoint
        # root; the index itself must not be allowed to point to an unbound root.
        result[cell_id] = S27MaterializedCellInput(
            cell_id=cell_id,
            config_ref=config_ref,
            environment_ref=environment_ref,
            checkpoint_root_ref=str(checkpoint["checkpoint_root_ref"]),
            registry_ref=registry_ref,
            reference_output_root_ref=str(artifacts["output_dir"]),
        )
    if tuple(result) != EXPECTED_CELL_IDS:
        raise S27ExecutionBlocked("S27_MATERIALIZATION_CELL_ORDER_INVALID")
    return result


def _load_checkpoint_root(
    root: Path,
    cell: S27CellPlan,
    materialized: S27MaterializedCellInput,
) -> Path:
    manifest_path = _safe_ref(root, cell.checkpoint_ref, field=f"checkpoint.{cell.cell_id}.manifest")
    manifest_value = load_canonical_json(manifest_path) if manifest_path.is_file() else None
    if not isinstance(manifest_value, Mapping) or canonical_json_hash(dict(manifest_value)) != cell.checkpoint_hash:
        raise S27ExecutionBlocked(f"S27_CHECKPOINT_MANIFEST_HASH_MISMATCH:{cell.cell_id}")
    checkpoint_root = _safe_ref(root, materialized.checkpoint_root_ref, field=f"checkpoint.{cell.cell_id}.root")
    if not checkpoint_root.is_dir():
        raise S27ExecutionBlocked(f"S27_CHECKPOINT_ROOT_MISSING:{cell.cell_id}")
    return checkpoint_root


def build_s27_torch_provider(
    cell: S27CellPlan,
    *,
    data_root: Path,
    checkpoint_root_ref: str,
    registry_ref: str,
    request: TaskExecutionRequest,
) -> S27ProviderContext:
    """Build the real selected-checkpoint Torch provider.

    ``FormalG3RuntimeAssets`` still authorizes the immutable data/tokenizer
    route and the G3 lineage.  The model itself is loaded from the hash-bound
    S2.3 checkpoint root, never from the step-0 alias used by the shared legacy
    adapter.  This function is intentionally lazy about optional HF imports so
    control-plane tests remain CPU-only.
    """

    if request.config.run_intent != "formal" or request.task.stage != 2:
        raise S27ExecutionBlocked("S27_FORMAL_REQUEST_REQUIRED")
    from .stage23_task_runners import (
        _formal_execution_evidence,
        _load_formal_parameter_registry,
        _load_formal_selected_checkpoint,
    )
    try:
        evidence, _ = _formal_execution_evidence(request, data_root)
        providers = request.config.section("providers")
        base = request.config.base_config
        model_config = base.section("model")
        data = base.section("data")
        runtime = base.section("runtime")
        identity = base.section("identity")
        if not all(isinstance(value, Mapping) for value in (providers, model_config, data, runtime, identity)):
            raise ValueError("S27_REQUEST_SECTIONS_INVALID")
        if providers.get("kind") != "offline_hf":
            raise S27ExecutionBlocked("S27_OFFLINE_HF_PROVIDER_REQUIRED")
        task_type = str(providers["task_type"])
        runtime_assets = FormalG3RuntimeAssets.from_request(request, data_root)
        architecture = model_config.get("architecture")
        if not isinstance(architecture, str) or not architecture:
            raise ValueError("S27_MODEL_ARCHITECTURE_REQUIRED")
        model_asset = runtime_assets.resolve(f"{architecture}-step0", expected_kind="model")
        expected_data_kind = "pile" if task_type == "causal_lm" else "glue_derived"
        data_asset = runtime_assets.resolve(str(data["asset_id"]), expected_kind=expected_data_kind)
        tokenizer_asset = runtime_assets.resolve(str(model_config["tokenizer_asset_id"]), expected_kind="tokenizer")
        lineage = runtime_assets.runtime_lineage_sha256(model_asset, data_asset, tokenizer_asset)
        if not set(item.ready_manifest_sha256 for item in (model_asset, data_asset, tokenizer_asset)).issubset(evidence.asset_manifest_hashes):
            raise S27ExecutionBlocked("S27_G3_ASSET_MANIFEST_LINEAGE_MISMATCH")
        # Reuse the shared S2.4 strict selected-checkpoint binding.  It
        # verifies the source manifest bytes, exact file inventory and every
        # model.safetensors digest before Transformers sees the directory.
        selected_checkpoint = _load_formal_selected_checkpoint(request, data_root)
        if selected_checkpoint is None:
            raise S27ExecutionBlocked("S27_FORMAL_SELECTED_CHECKPOINT_REQUIRED")
        if (
            selected_checkpoint.checkpoint_id != cell.checkpoint_id
            or selected_checkpoint.manifest_sha256 != cell.checkpoint_hash
            or selected_checkpoint.root_ref != checkpoint_root_ref
        ):
            raise S27ExecutionBlocked("S27_SELECTED_CHECKPOINT_LINEAGE_MISMATCH")
        checkpoint_root = selected_checkpoint.root
        model = OfflineHuggingFaceModelAdapter.from_local_directory(
            checkpoint_root,
            task_type=task_type,
            num_labels=providers.get("num_labels"),
            torch_dtype=__import__("torch").float32,
        )
        torch = __import__("torch")
        model.module.to(torch.device(str(runtime["device"])))
        model.module.eval()
        if any(parameter.dtype != torch.float32 for parameter in model.module.parameters()):
            raise ValueError("S27_MODEL_DTYPE_NOT_FLOAT32")
        if task_type == "causal_lm":
            if data_asset.storage_kind != "pythia_mmap_shards":
                raise G3RuntimeAssetError("S27_PILE_STORAGE_KIND_INVALID")
            route = formal_pile_route(
                stage=int(identity["stage"]),
                evaluation=False,
                declared_sampling_design=str(data["sampling_design"]),
                configured_split=str(data["split"]),
            )
            start, stop = runtime_assets.pile_split_interval(data_asset, route.split)
            runtime_assets.validate_pile_budget(stage=int(identity["stage"]), split=route.split, requested_records=stop - start)
            resolver = PythiaMMapFrozenSampleResolver(
                runtime_assets.pythia_dataset(data_asset, split=route.split),
                asset_id=data_asset.resolved.asset_id,
                ready_manifest_sha256=data_asset.ready_manifest_sha256,
                qualification_sha256=data_asset.qualification_artifact_hash,
                g3_resolution_artifact_hash=runtime_assets.resolution_artifact_hash,
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=lineage,
                split_start=start,
                split_stop=stop,
                sampling_design=str(data["sampling_design"]),
                weights_exogenous=bool(data["weights_exogenous"]),
                common_mean_assumption=bool(data["common_mean_assumption"]),
            )
        elif task_type == "sequence_classification":
            if data_asset.storage_kind != "hf_load_from_disk":
                raise G3RuntimeAssetError("S27_GLUE_STORAGE_KIND_INVALID")
            glue_task = data_asset.require_glue_route(task_name=str(providers["task_name"]), split=str(data["split"]))
            resolver = PretokenizedGlueDatasetAdapter(
                data_asset.resolved.root,
                task_name=glue_task,
                split=str(data["split"]),
                dataset_id=data_asset.resolved.asset_id,
                microbatch_size=1,
                microbatches_per_step=1,
                expected_asset_hash=data_asset.directory_content_sha256,
                allowed_root=data_asset.resolved.root,
                g3_resolution_artifact_hash=runtime_assets.resolution_artifact_hash,
                g3_source_commit=runtime_assets.source_git_commit,
                g3_runtime_lineage_sha256=lineage,
            )
        else:
            raise ValueError("S27_TASK_TYPE_UNSUPPORTED")
        registry = _load_formal_parameter_registry(data_root, registry_ref, model.module)
        if registry.get("registry_hash") != selected_checkpoint.registry_hash:
            raise S27ExecutionBlocked("S27_SELECTED_CHECKPOINT_REGISTRY_MISMATCH")
        provider = TorchFixedStateGradientProvider(
            model,
            resolver,
            fixed_state_id=f"s27-{cell.checkpoint_id}-{cell.checkpoint_hash[:16]}",
            registry=registry,
            output_dtype=torch.float32,
            gradient_chunk_size=1,
            enable_formal_batched=True,
            formal_batch_chunk_size=4,
        )
        return S27ProviderContext(
            provider=provider,
            execution=evidence,
            checkpoint_root_ref=checkpoint_root_ref,
            checkpoint_hash=cell.checkpoint_hash,
            registry_ref=registry_ref,
            registry_hash=provider.registry_hash,
        )
    except S27ExecutionBlocked:
        raise
    except Exception as error:
        raise S27ExecutionBlocked(f"S27_TORCH_PROVIDER_BIND_FAILED:{cell.cell_id}:{type(error).__name__}:{error}") from error


def _unit_file_name(unit_id: str) -> str:
    return hashlib.sha256(unit_id.encode("utf-8")).hexdigest() + ".json"


def _read_json_if_exists(path: Path) -> Mapping[str, object] | None:
    if not path.exists():
        return None
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise S27ExecutionBlocked(f"S27_JSON_OBJECT_REQUIRED:{path.name}")
    return value


def _write_once(path: Path, value: Mapping[str, object], *, field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = load_canonical_json(path)
        if existing != dict(value):
            raise S27ExecutionBlocked(f"{field}:OUTPUT_CONFLICT")
        return
    write_canonical_json(path, dict(value))


def _state_from_committed_wave(wave_root: Path, unit_id: str) -> tuple[Mapping[str, object], Mapping[str, object]] | None:
    commit_path = wave_root / "commits" / f"{unit_id}.json"
    if not commit_path.exists():
        return None
    commit = load_canonical_json(commit_path)
    if not isinstance(commit, Mapping):
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_INVALID:{unit_id}")
    if commit.get("artifact_hash") != canonical_json_hash({key: item for key, item in commit.items() if key != "artifact_hash"}):
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_HASH_INVALID:{unit_id}")
    relative = commit.get("object_ref")
    if not isinstance(relative, str) or PurePosixPath(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise S27ExecutionBlocked(f"S27_WAVE_OBJECT_REF_INVALID:{unit_id}")
    state, bundle = load_tensor_bundle((wave_root / relative).resolve())
    if not isinstance(state, Mapping):
        raise S27ExecutionBlocked(f"S27_WAVE_STATE_INVALID:{unit_id}")
    if bundle.manifest_sha256 != commit.get("object_manifest_hash") or commit.get("unit_id") != unit_id:
        raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_BINDING_INVALID:{unit_id}")
    return state, commit


def _cost_from_state(state: Mapping[str, object]) -> dict[str, object]:
    wall = state.get("wall_seconds")
    formula = state.get("formula_seconds")
    gradients = state.get("gradient_evaluations")
    valid = isinstance(wall, (int, float)) and not isinstance(wall, bool) and float(wall) >= 0 and isinstance(formula, (int, float)) and not isinstance(formula, bool) and float(formula) >= 0 and isinstance(gradients, int) and not isinstance(gradients, bool) and gradients >= 0
    return {
        "valid": bool(valid),
        "wall_seconds": wall,
        "formula_seconds": formula,
        "gradient_seconds": state.get("gradient_seconds"),
        "gradient_evaluations": gradients,
        "peak_memory_bytes": state.get("peak_memory_bytes"),
        "shared_gradient_pool": True,
    }


class _GradientAuditProxy:
    """Capture the immutable base pool for an observed mean-gradient audit."""

    def __init__(self, delegate: FixedStateGradientProvider) -> None:
        self._delegate = delegate
        self._batches: list[tuple[tuple[object, ...], object]] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self._delegate, name)

    @property
    def registry_hash(self) -> str:
        return self._delegate.registry_hash

    @property
    def fixed_state_id(self) -> str:
        return self._delegate.fixed_state_id

    @property
    def statistical_unit(self) -> str:
        return self._delegate.statistical_unit

    @property
    def weight_unit(self) -> str:
        return self._delegate.weight_unit

    @property
    def sampling_design(self) -> str:
        return self._delegate.sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._delegate.weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._delegate.common_mean_assumption

    def state_digest(self) -> str:
        return self._delegate.state_digest()

    def assert_unchanged(self, expected_digest: str) -> None:
        self._delegate.assert_unchanged(expected_digest)

    def begin(self) -> None:
        self._batches.clear()

    def gradient(self, draws: Sequence[object]) -> object:
        batch = self._delegate.gradient(draws)
        self._batches.append((tuple(draws), batch))
        return batch

    @staticmethod
    def _weighted_mean(batches: Sequence[object]) -> dict[str, np.ndarray]:
        if not batches:
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_POOL_EMPTY")
        vectors = [getattr(batch, "gradients") for batch in batches]
        weights = [float(getattr(batch, "statistical_weight")) for batch in batches]
        return _GradientAuditProxy._weighted_vector_mean(vectors, weights)

    @staticmethod
    def _weighted_vector_mean(
        vectors: Sequence[Mapping[str, object]], weights: Sequence[float]
    ) -> dict[str, np.ndarray]:
        if not vectors or len(vectors) != len(weights):
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_VECTOR_POOL_INVALID")
        names = tuple(vectors[0].keys())
        total_weight = float(sum(weights))
        if not np.isfinite(total_weight) or total_weight <= 0:
            raise S27ExecutionBlocked("S27_MEAN_GRADIENT_WEIGHT_INVALID")
        result: dict[str, np.ndarray] = {}
        for name in names:
            value = np.zeros_like(np.asarray(vectors[0][name]), dtype=np.float64)
            for gradients, weight in zip(vectors, weights):
                if tuple(gradients.keys()) != names:
                    raise S27ExecutionBlocked("S27_MEAN_GRADIENT_PARAMETER_SET_DRIFT")
                value += np.asarray(gradients[name], dtype=np.float64) * weight
            result[name] = value / total_weight
        return result

    def finish(self, mapping: RepetitionMapping, *, tolerance: float) -> dict[str, object]:
        expected_count = max(mapping.m_values)
        if len(self._batches) != expected_count:
            return {"passed": False, "max_abs_error": None, "observed_batch_count": len(self._batches), "expected_batch_count": expected_count}
        batches = [batch for _, batch in self._batches]
        full = self._weighted_mean(batches)
        # Re-form every frozen M from the exact live base pool.  This is an
        # observed regrouping audit (not a metadata assertion): the same
        # GradientBatch values that fed raw/double/U are grouped at each M,
        # reduced with their true weights, and compared with the full mean.
        max_m = len(batches)
        per_m: dict[str, dict[str, object]] = {}
        errors: list[float] = []
        for microbatch_count in tuple(mapping.m_values):
            if microbatch_count <= 0 or max_m % microbatch_count:
                per_m[str(microbatch_count)] = {
                    "passed": False,
                    "max_abs_error": None,
                    "reason": "base_pool_not_divisible_by_m",
                }
                continue
            merge_width = max_m // microbatch_count
            grouped_vectors: list[Mapping[str, object]] = []
            grouped_weights: list[float] = []
            for start in range(0, max_m, merge_width):
                group = batches[start : start + merge_width]
                grouped_vectors.append(self._weighted_mean(group))
                grouped_weights.append(
                    float(sum(float(getattr(batch, "statistical_weight")) for batch in group))
                )
            regrouped = self._weighted_vector_mean(grouped_vectors, grouped_weights)
            m_errors = [
                float(np.max(np.abs(full[name] - regrouped[name])))
                for name in full
            ]
            m_error = float(max(m_errors, default=0.0))
            finite_m = bool(np.isfinite(m_error))
            if finite_m:
                errors.append(m_error)
            per_m[str(microbatch_count)] = {
                "passed": bool(finite_m and m_error <= tolerance),
                "max_abs_error": m_error if finite_m else None,
                "merge_width": merge_width,
                "group_count": microbatch_count,
            }
        error = float(max(errors, default=float("nan")))
        finite = bool(np.isfinite(error))
        all_m_passed = bool(per_m) and all(item.get("passed") is True for item in per_m.values())
        return {
            "passed": bool(all_m_passed and finite and error <= tolerance),
            "max_abs_error": error if finite else None,
            "observed_batch_count": len(self._batches),
            "expected_batch_count": expected_count,
            "per_m": per_m,
            "identity": "weighted_full_mean_equals_regrouped_nested_m_means",
            "tolerance": tolerance,
        }


def _success_record(
    plan: S27Plan,
    cell: S27CellPlan,
    unit: S27MappingUnit,
    state: Mapping[str, object],
    commit: Mapping[str, object],
    *,
    artifact_root: Path,
    data_root: Path,
    attempt_id: str,
    mean_gradient_audit: Mapping[str, object],
) -> S27RawUnit:
    vectors = state.get("vectors")
    if not isinstance(vectors, Mapping):
        raise S27ExecutionBlocked(f"S27_RAW_VECTORS_INVALID:{unit.unit_id}")
    if state.get("input_hash") != unit.mapping_hash or commit.get("input_hash") != unit.mapping_hash:
        raise S27ExecutionBlocked(f"S27_RAW_MAPPING_BINDING_INVALID:{unit.unit_id}")
    expected_methods = {"raw", "double", "u_m2", f"u_m{unit.microbatch_count}"}
    if set(vectors) != expected_methods:
        raise S27ExecutionBlocked(f"S27_RAW_METHOD_SET_INVALID:{unit.unit_id}")
    for method, vector in vectors.items():
        if not isinstance(vector, Mapping) or any(not np.all(np.isfinite(np.asarray(value))) for value in vector.values()):
            raise S27ExecutionBlocked(f"S27_RAW_NONFINITE_VECTOR:{unit.unit_id}:{method}")
    # ``object_ref`` is relative to the paired-wave artifact root, not the
    # enclosing S2.7 run root.  Binding the manifest to that exact immutable
    # object keeps the sealed raw manifest recoverable and prevents a
    # misleading path that happens to hash correctly but cannot be reopened.
    raw_ref = _relative_ref(data_root, (artifact_root / str(commit["object_ref"])).resolve(), field=f"raw_artifact.{unit.unit_id}")
    raw_hash = commit.get("object_manifest_hash")
    if not isinstance(raw_hash, str) or not _SHA256.fullmatch(raw_hash):
        raise S27ExecutionBlocked(f"S27_RAW_OBJECT_HASH_INVALID:{unit.unit_id}")
    state_digest = state.get("state_digest")
    state_after = state.get("state_digest_after")
    if state_digest != state_after:
        raise S27ExecutionBlocked(f"S27_PROVIDER_STATE_DRIFT:{unit.unit_id}")
    assumptions = state.get("weighting_assumptions")
    mean_consistent = mean_gradient_audit.get("passed") is True
    metrics: dict[str, object] = {
        "finite": True,
        "raw_double_shared_gradient_pool": True,
        "nested_m_shared_gradient_pool": True,
        "mean_gradient_audit": dict(mean_gradient_audit),
        "inner_attempt_id": state.get("attempt_id"),
    }
    return S27RawUnit(
        unit_id=unit.unit_id,
        cell_id=unit.cell_id,
        repetition_id=unit.repetition_id,
        status="SUCCESS",
        attempt_id=attempt_id,
        matrix_hash=plan.frozen_inputs.matrix_hash,
        mapping_hash=unit.mapping_hash,
        sampling_plan_hash=plan.frozen_inputs.sampling_plan_hash,
        checkpoint_hash=cell.checkpoint_hash,
        reference_hash=cell.reference_hash,
        batch_size=unit.batch_size,
        microbatch_count=unit.microbatch_count,
        draw_ids=unit.draw_ids,
        sample_ids=unit.sample_ids,
        raw_artifact_ref=raw_ref,
        raw_artifact_hash=raw_hash,
        metrics=metrics,
        methods=tuple(sorted(expected_methods)),
        m2_identity_max_abs=float(state.get("m2_double_max_abs_error")),
        mean_gradient_consistent=bool(mean_consistent),
        clamp_applied=False,
        clip_mode="none",
        cost=_cost_from_state(state),
    )


def _failure_record(
    plan: S27Plan,
    cell: S27CellPlan,
    unit: S27MappingUnit,
    *,
    attempt_id: str,
    run_root: Path,
    data_root: Path,
    code: str,
    reason: str,
) -> S27RawUnit:
    if not _SAFE_COMPONENT.fullmatch(code):
        code = "S27_WORKER_FAILURE"
    reason = " ".join(str(reason).split())
    if not reason:
        reason = "unspecified production unit failure"
    body: dict[str, object] = {
        "schema_version": S27_RAW_RESULT_SCHEMA,
        "run_id": run_root.name,
        "unit_id": unit.unit_id,
        "cell_id": unit.cell_id,
        "repetition_id": unit.repetition_id,
        "status": "FAILED",
        "attempt_id": attempt_id,
        "failure_code": code,
        "failure_reason": reason,
        "matrix_hash": plan.frozen_inputs.matrix_hash,
        "mapping_hash": unit.mapping_hash,
        "sampling_plan_hash": plan.frozen_inputs.sampling_plan_hash,
        "checkpoint_hash": cell.checkpoint_hash,
        "reference_hash": cell.reference_hash,
        "draw_ids": list(unit.draw_ids),
        "sample_ids": list(unit.sample_ids),
    }
    body["artifact_hash"] = canonical_json_hash(body)
    artifact_path = run_root / "raw-artifacts" / f"{_unit_file_name(unit.unit_id)[:-5]}-failure.json"
    _write_once(artifact_path, body, field=f"S27_FAILURE_ARTIFACT:{unit.unit_id}")
    return S27RawUnit(
        unit_id=unit.unit_id,
        cell_id=unit.cell_id,
        repetition_id=unit.repetition_id,
        status="FAILED",
        attempt_id=attempt_id,
        matrix_hash=plan.frozen_inputs.matrix_hash,
        mapping_hash=unit.mapping_hash,
        sampling_plan_hash=plan.frozen_inputs.sampling_plan_hash,
        checkpoint_hash=cell.checkpoint_hash,
        reference_hash=cell.reference_hash,
        batch_size=unit.batch_size,
        microbatch_count=unit.microbatch_count,
        draw_ids=unit.draw_ids,
        sample_ids=unit.sample_ids,
        raw_artifact_ref=_relative_ref(data_root, artifact_path, field=f"S27_FAILURE_REF:{unit.unit_id}"),
        raw_artifact_hash=str(body["artifact_hash"]),
        metrics={"finite": False, "failed": True},
        methods=("raw", "double", "u_m2", f"u_m{unit.microbatch_count}"),
        m2_identity_max_abs=None,
        mean_gradient_consistent=False,
        clamp_applied=False,
        clip_mode="none",
        cost={"valid": False, "reason": "unit_failed"},
        failure_code=code,
        failure_reason=reason,
    )


def _attempt_payload(unit: S27MappingUnit, *, attempt_id: str, mapping_hash: str, status: str, code: str | None = None, reason: str | None = None) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": S27_ATTEMPT_SCHEMA,
        "unit_id": unit.unit_id,
        "mapping_hash": mapping_hash,
        "attempt_id": attempt_id,
        "status": status,
        "started_at": _now(),
        "failure_code": code,
        "failure_reason": reason,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


class S27ProductionWorker:
    """Run one frozen cell on one approved GPU and publish raw terminal units."""

    def __init__(
        self,
        *,
        plan: S27Plan,
        run_id: str,
        cell_id: str,
        gpu_uuid: str,
        data_root: str | Path,
        run_root: str | Path,
        provider_factory: S27ProviderFactory | None = None,
        request: TaskExecutionRequest | None = None,
        materialized_input: S27MaterializedCellInput | None = None,
        retry_policy: S27RetryPolicy | None = None,
        m2_tolerance: float = S27_DEFAULT_M2_TOLERANCE,
    ) -> None:
        self.plan = plan
        self.run_id = run_id
        self.cell = next((item for item in plan.cells if item.cell_id == cell_id), None)
        if self.cell is None:
            raise S27ExecutionBlocked(f"S27_UNKNOWN_CELL:{cell_id}")
        if gpu_uuid not in APPROVED_GPU_UUIDS or gpu_uuid != self.cell.assigned_gpu_uuid:
            raise S27ExecutionBlocked(f"S27_GPU_ASSIGNMENT_INVALID:{cell_id}")
        self.gpu_uuid = gpu_uuid
        self.data_root = Path(data_root).resolve()
        self.run_root = Path(run_root).resolve()
        self.provider_factory = provider_factory or build_s27_torch_provider
        self.request = request
        self.materialized_input = materialized_input
        self.retry_policy = retry_policy or S27RetryPolicy()
        if not isinstance(m2_tolerance, (int, float)) or isinstance(m2_tolerance, bool) or float(m2_tolerance) < 0:
            raise ValueError("S27_M2_TOLERANCE_INVALID")
        self.m2_tolerance = float(m2_tolerance)
        self.wave_root = self.run_root / "waves" / self.cell.cell_id.replace(":", "__")
        self.raw_root = self.run_root / "raw-units"
        self.attempt_root = self.run_root / "attempts"

    def _unit_path(self, unit_id: str) -> Path:
        return self.raw_root / _unit_file_name(unit_id)

    def _attempt_path(self, unit_id: str) -> Path:
        return self.attempt_root / _unit_file_name(unit_id)

    def _terminal_attempt_path(self, unit_id: str) -> Path:
        return self.attempt_root / ("terminal-" + _unit_file_name(unit_id))

    def _validate_attempt_ledger(self, record: S27RawUnit) -> Mapping[str, object]:
        attempt = _read_json_if_exists(self._attempt_path(record.unit_id))
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
        ):
            raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_BINDING_INVALID:{record.unit_id}")
        return attempt

    def _terminal_payload(self, record: S27RawUnit) -> dict[str, object]:
        body: dict[str, object] = {
            "schema_version": S27_ATTEMPT_SCHEMA,
            "unit_id": record.unit_id,
            "mapping_hash": record.mapping_hash,
            "attempt_id": record.attempt_id,
            "status": record.status,
            "record_artifact_hash": record.artifact_hash,
            "finished_at": _now(),
        }
        body["artifact_hash"] = canonical_json_hash(body)
        return body

    def _publish_terminal(self, record: S27RawUnit) -> None:
        _write_once(
            self._terminal_attempt_path(record.unit_id),
            self._terminal_payload(record),
            field=f"S27_TERMINAL_ATTEMPT:{record.unit_id}",
        )

    def _validate_terminal_attempt(self, record: S27RawUnit) -> None:
        self._validate_attempt_ledger(record)
        terminal = _read_json_if_exists(self._terminal_attempt_path(record.unit_id))
        if (
            terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_TERMINAL_ATTEMPT_BINDING_MISSING:{record.unit_id}")

    def _load_existing_records(self, reducer: StrictG25Reducer, mappings: Mapping[str, RepetitionMapping]) -> set[str]:
        done: set[str] = set()
        for unit in self.plan.frozen_inputs.units:
            if unit.cell_id != self.cell.cell_id:
                continue
            path = self._unit_path(unit.unit_id)
            raw = _read_json_if_exists(path)
            if raw is not None:
                record = S27RawUnit.from_mapping(raw)
                self._validate_attempt_ledger(record)
                if _read_json_if_exists(self._terminal_attempt_path(record.unit_id)) is None:
                    # A crash after the append-only raw record but before its
                    # terminal marker is recoverable: the raw record is the
                    # completion marker, and the outer RUNNING ledger binds it
                    # to the sole attempt.  Publish the missing marker without
                    # changing the raw object.
                    self._publish_terminal(record)
                self._validate_terminal_attempt(record)
                reducer.add(record)
                done.add(unit.unit_id)
                continue
            attempt = _read_json_if_exists(self._attempt_path(unit.unit_id))
            if attempt is not None and attempt.get("status") == "RUNNING":
                if attempt.get("artifact_hash") != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"}) or attempt.get("mapping_hash") != unit.mapping_hash:
                    raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_HASH_INVALID:{unit.unit_id}")
                # A process that was detached and killed has crossed the only
                # retry boundary.  It is a failure in the fixed denominator.
                failed = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=str(attempt.get("attempt_id", "attempt-recovery")),
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_INTERRUPTED_ATTEMPT",
                    reason="previous worker left a non-terminal attempt; recovery is fail-closed",
                )
                self._publish_record(reducer, failed)
                done.add(unit.unit_id)
        return done

    def _publish_record(self, reducer: StrictG25Reducer, record: S27RawUnit) -> None:
        path = self._unit_path(record.unit_id)
        _write_once(path, record.to_dict(), field=f"S27_RAW_UNIT:{record.unit_id}")
        reducer.add(record)
        self._publish_terminal(record)

    def run(self) -> dict[str, object]:
        if self.gpu_uuid not in APPROVED_GPU_UUIDS:
            raise S27ExecutionBlocked("S27_APPROVED_GPU_REQUIRED")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None and visible != self.gpu_uuid:
            raise S27ExecutionBlocked("S27_CUDA_VISIBLE_DEVICES_ASSIGNMENT_DRIFT")
        if self.plan.frozen_inputs.mapping_ref is None:
            raise S27ExecutionBlocked("S27_FROZEN_MAPPING_REQUIRED")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.attempt_root.mkdir(parents=True, exist_ok=True)
        mappings = load_s27_frozen_mappings(self.data_root, self.plan, cell_id=self.cell.cell_id)
        reducer = StrictG25Reducer(self.plan, run_id=self.run_id)
        done = self._load_existing_records(reducer, mappings)
        cell_inputs = self.materialized_input
        context: S27ProviderContext | None = None
        provider_error: Exception | None = None
        try:
            if cell_inputs is None:
                raise S27ExecutionBlocked("S27_MATERIALIZED_CELL_INPUT_REQUIRED")
            _load_checkpoint_root(self.data_root, self.cell, cell_inputs)
            context = self.provider_factory(
                self.cell,
                data_root=self.data_root,
                checkpoint_root_ref=cell_inputs.checkpoint_root_ref,
                registry_ref=cell_inputs.registry_ref,
                request=self.request,
            )
            if context.checkpoint_hash != self.cell.checkpoint_hash:
                raise S27ExecutionBlocked("S27_PROVIDER_CHECKPOINT_HASH_DRIFT")
        except Exception as error:
            provider_error = error
        if context is None:
            # Provider construction failure still consumes every expected unit
            # as an explicit failed denominator row; no cell is silently skipped.
            for unit in self.plan.frozen_inputs.units:
                if unit.cell_id != self.cell.cell_id or unit.unit_id in done:
                    continue
                provider_attempt_id = f"attempt-provider-{time.time_ns()}"
                _write_once(
                    self._attempt_path(unit.unit_id),
                    _attempt_payload(unit, attempt_id=provider_attempt_id, mapping_hash=unit.mapping_hash, status="RUNNING"),
                    field=f"S27_ATTEMPT:{unit.unit_id}",
                )
                failed = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=provider_attempt_id,
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_PROVIDER_BIND_FAILED",
                    reason=f"{type(provider_error).__name__}:{provider_error}",
                )
                self._publish_record(reducer, failed)
            self._seal_wave(reducer)
            return {"status": "FAILED", "cell_id": self.cell.cell_id, "failed_units": len(mappings), "failure_code": "S27_PROVIDER_BIND_FAILED"}

        reference_views = load_s27_reference_views(
            self.data_root,
            self.cell,
            expected_registry_hash=context.provider.registry_hash,
            reference_output_root_ref=(cell_inputs.reference_output_root_ref or None),
        )
        reference = reference_views["bias"]
        audit_provider = _GradientAuditProxy(context.provider)
        wave_runner = RecoverablePairedWaveRunner(audit_provider, execution=context.execution, m2_tolerance=self.m2_tolerance)
        for unit in self.plan.frozen_inputs.units:
            if unit.cell_id != self.cell.cell_id or unit.unit_id in done:
                continue
            mapping = mappings[unit.unit_id]
            attempt_path = self._attempt_path(unit.unit_id)
            existing_attempt = _read_json_if_exists(attempt_path)
            if existing_attempt is not None:
                if existing_attempt.get("mapping_hash") != unit.mapping_hash or existing_attempt.get("status") != "RUNNING":
                    raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_CONFLICT:{unit.unit_id}")
                # The only legal pre-existing state was handled by recovery.
                raise S27ExecutionBlocked(f"S27_NONTERMINAL_ATTEMPT_NOT_RECOVERED:{unit.unit_id}")
            attempt_id = f"attempt-{time.time_ns()}"
            _write_once(attempt_path, _attempt_payload(unit, attempt_id=attempt_id, mapping_hash=unit.mapping_hash, status="RUNNING"), field=f"S27_ATTEMPT:{unit.unit_id}")
            try:
                audit_provider.begin()
                unit_wave_root = self.wave_root / "paired" / _unit_file_name(unit.unit_id).removesuffix(".json")
                wave_runner.run(
                    wave_id=f"s27-{self.cell.cell_id.replace(':', '-')}",
                    mappings=(mapping,),
                    reference=reference,
                    reference_hash=_vector_digest(reference),
                    references=reference_views,
                    artifact_root=unit_wave_root,
                    max_new_units=1,
                )
                committed = _state_from_committed_wave(unit_wave_root, mapping.repetition_id)
                if committed is None:
                    raise S27ExecutionBlocked(f"S27_WAVE_COMMIT_MISSING:{unit.unit_id}")
                state, commit = committed
                if state.get("registry_hash") != context.provider.registry_hash:
                    raise S27ExecutionBlocked(f"S27_WAVE_REGISTRY_HASH_DRIFT:{unit.unit_id}")
                audit = audit_provider.finish(mapping, tolerance=self.m2_tolerance)
                record = _success_record(
                    self.plan,
                    self.cell,
                    unit,
                    state,
                    commit,
                    artifact_root=unit_wave_root,
                    data_root=self.data_root,
                    attempt_id=attempt_id,
                    mean_gradient_audit=audit,
                )
            except Exception as error:
                record = _failure_record(
                    self.plan,
                    self.cell,
                    unit,
                    attempt_id=attempt_id,
                    run_root=self.run_root,
                    data_root=self.data_root,
                    code="S27_UNIT_EXECUTION_FAILED",
                    reason=f"{type(error).__name__}:{error}",
                )
            self._publish_record(reducer, record)
        return self._seal_wave(reducer)

    def _seal_wave(self, reducer: StrictG25Reducer) -> dict[str, object]:
        expected = [unit.unit_id for unit in self.plan.frozen_inputs.units if unit.cell_id == self.cell.cell_id]
        records = [record for record in reducer.records if record.cell_id == self.cell.cell_id]
        if {record.unit_id for record in records} != set(expected):
            raise S27ExecutionBlocked(f"S27_WAVE_EXPECTED_UNITS_MISSING:{self.cell.cell_id}")
        descriptors = [{"unit_id": record.unit_id, "status": record.status, "attempt_id": record.attempt_id, "unit_artifact_hash": record.artifact_hash} for record in records]
        body: dict[str, object] = {
            "schema_version": S27_WAVE_SEAL_SCHEMA,
            "run_id": self.run_id,
            "plan_hash": self.plan.artifact_hash,
            "cell_id": self.cell.cell_id,
            "gpu_uuid": self.gpu_uuid,
            "expected_unit_count": len(expected),
            "completed_unit_count": len(records),
            "failed_unit_count": sum(record.status == "FAILED" for record in records),
            "units": descriptors,
            "sealed": True,
            "checked_at": _now(),
        }
        body["artifact_hash"] = canonical_json_hash(body)
        path = self.run_root / "wave-seals" / f"{self.cell.cell_id.replace(':', '__')}.json"
        _write_once(path, body, field=f"S27_WAVE_SEAL:{self.cell.cell_id}")
        return {"status": "SEALED", "cell_id": self.cell.cell_id, "wave_seal_ref": _relative_ref(self.data_root, path, field="S27_WAVE_SEAL_REF"), "expected_units": len(expected), "failed_units": body["failed_unit_count"]}


def load_s27_raw_records(data_root: str | Path, plan: S27Plan, run_root: str | Path, *, run_id: str) -> StrictG25Reducer:
    root = Path(data_root).resolve()
    reducer = StrictG25Reducer(plan, run_id=run_id)
    record_root = Path(run_root).resolve() / "raw-units"
    if not record_root.is_dir():
        raise S27ExecutionBlocked("S27_RAW_UNIT_DIRECTORY_MISSING")
    for path in sorted(record_root.glob("*.json")):
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_RAW_UNIT_NOT_OBJECT:{path.name}")
        record = S27RawUnit.from_mapping(value)
        artifact_path = _safe_ref(root, record.raw_artifact_ref, field=f"S27_RAW_ARTIFACT:{record.unit_id}")
        if record.status == "SUCCESS":
            if not artifact_path.is_dir() or artifact_path.is_symlink():
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_DIRECTORY_MISSING:{record.unit_id}")
            try:
                state, bundle = load_tensor_bundle(artifact_path)
            except (OSError, TypeError, ValueError) as error:
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_INVALID:{record.unit_id}") from error
            if (
                not isinstance(state, Mapping)
                or state.get("schema_version") != "stage2-wave-unit-state-v1"
                or state.get("unit_id") != record.repetition_id
                or not isinstance(state.get("vectors"), Mapping)
                or bundle.manifest_sha256 != record.raw_artifact_hash
            ):
                raise S27ExecutionBlocked(f"S27_RAW_SUCCESS_ARTIFACT_HASH_INVALID:{record.unit_id}")
        else:
            if not artifact_path.is_file() or artifact_path.is_symlink():
                raise S27ExecutionBlocked(f"S27_RAW_FAILURE_ARTIFACT_MISSING:{record.unit_id}")
            failure_payload = load_canonical_json(artifact_path)
            if (
                not isinstance(failure_payload, Mapping)
                or failure_payload.get("artifact_hash") != record.raw_artifact_hash
                or canonical_json_hash({key: item for key, item in failure_payload.items() if key != "artifact_hash"}) != record.raw_artifact_hash
                or failure_payload.get("unit_id") != record.unit_id
            ):
                raise S27ExecutionBlocked(f"S27_RAW_FAILURE_ARTIFACT_HASH_INVALID:{record.unit_id}")
        attempt_path = record_root.parent / "attempts" / _unit_file_name(record.unit_id)
        terminal_path = record_root.parent / "attempts" / ("terminal-" + _unit_file_name(record.unit_id))
        attempt = _read_json_if_exists(attempt_path)
        terminal = _read_json_if_exists(terminal_path)
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
            or terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_ATTEMPT_LEDGER_BINDING_INVALID:{record.unit_id}")
        reducer.add(record)
    return reducer


def seal_s27_run(data_root: str | Path, plan: S27Plan, run_root: str | Path, *, run_id: str) -> dict[str, object]:
    """Seal only the complete denominator; no statistics are computed here."""

    reducer = load_s27_raw_records(data_root, plan, run_root, run_id=run_id)
    for cell in plan.cells:
        seal = Path(run_root).resolve() / "wave-seals" / f"{cell.cell_id.replace(':', '__')}.json"
        if not seal.is_file():
            raise S27ExecutionBlocked(f"S27_WAVE_SEAL_MISSING:{cell.cell_id}")
    return reducer.seal(Path(run_root).resolve() / "sealed")


def validate_s27_quality_wave(
    data_root: str | Path,
    plan: S27Plan,
    run_root: str | Path,
    *,
    cell_id: str,
) -> dict[str, object]:
    """Validate the first 14M quality wave before any later child starts.

    This is an operational conjunction only.  It checks every terminal raw
    unit, finite/m2/observed-mean-gradient/cost evidence and raw artifact
    persistence.  It never reads or compares estimator means, ranking, bias,
    or any superiority metric.
    """

    if cell_id != EXPECTED_CELL_IDS[0]:
        raise S27ExecutionBlocked("S27_QUALITY_WAVE_MUST_BE_FIRST_14M_INITIALIZATION")
    root = Path(data_root).resolve()
    run = Path(run_root).resolve()
    expected = [unit for unit in plan.frozen_inputs.units if unit.cell_id == cell_id]
    found: list[S27RawUnit] = []
    for unit in expected:
        path = run / "raw-units" / _unit_file_name(unit.unit_id)
        if not path.is_file():
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_UNIT_MISSING:{unit.unit_id}")
        value = load_canonical_json(path)
        if not isinstance(value, Mapping):
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_UNIT_INVALID:{unit.unit_id}")
        record = S27RawUnit.from_mapping(value)
        if record.status != "SUCCESS":
            raise S27ExecutionBlocked(f"S27_QUALITY_UNIT_FAILED:{unit.unit_id}")
        if record.mean_gradient_consistent is not True or record.m2_identity_max_abs is None or record.m2_identity_max_abs > 1e-12 or record.clamp_applied or record.clip_mode != "none" or record.cost.get("valid") is not True:
            raise S27ExecutionBlocked(f"S27_QUALITY_INTEGRITY_FAILED:{unit.unit_id}")
        audit = record.metrics.get("mean_gradient_audit")
        if not isinstance(audit, Mapping) or audit.get("passed") is not True:
            raise S27ExecutionBlocked(f"S27_QUALITY_MEAN_GRADIENT_AUDIT_FAILED:{unit.unit_id}")
        per_m = audit.get("per_m")
        expected_m = {str(value) for value in (2, plan.frozen_inputs.microbatch_count)}
        if (
            not isinstance(per_m, Mapping)
            or set(per_m) != expected_m
            or any(not isinstance(per_m[value], Mapping) or per_m[value].get("passed") is not True for value in expected_m)
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_MEAN_GRADIENT_PER_M_AUDIT_FAILED:{unit.unit_id}")
        attempt = _read_json_if_exists(run / "attempts" / _unit_file_name(unit.unit_id))
        terminal = _read_json_if_exists(run / "attempts" / ("terminal-" + _unit_file_name(unit.unit_id)))
        if (
            attempt is None
            or attempt.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in attempt.items() if key != "artifact_hash"})
            or attempt.get("unit_id") != record.unit_id
            or attempt.get("mapping_hash") != record.mapping_hash
            or attempt.get("attempt_id") != record.attempt_id
            or attempt.get("status") != "RUNNING"
            or terminal is None
            or terminal.get("artifact_hash")
            != canonical_json_hash({key: item for key, item in terminal.items() if key != "artifact_hash"})
            or terminal.get("unit_id") != record.unit_id
            or terminal.get("mapping_hash") != record.mapping_hash
            or terminal.get("attempt_id") != record.attempt_id
            or terminal.get("status") != record.status
            or terminal.get("record_artifact_hash") != record.artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_ATTEMPT_LEDGER_INVALID:{unit.unit_id}")
        artifact = _safe_ref(root, record.raw_artifact_ref, field=f"S27_QUALITY_RAW_ARTIFACT:{unit.unit_id}")
        if not artifact.is_dir() or artifact.is_symlink():
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_MISSING:{unit.unit_id}")
        try:
            state, bundle = load_tensor_bundle(artifact)
        except (OSError, TypeError, ValueError) as error:
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_INVALID:{unit.unit_id}") from error
        if (
            not isinstance(state, Mapping)
            or state.get("schema_version") != "stage2-wave-unit-state-v1"
            or state.get("unit_id") != record.repetition_id
            or bundle.manifest_sha256 != record.raw_artifact_hash
        ):
            raise S27ExecutionBlocked(f"S27_QUALITY_RAW_ARTIFACT_HASH_INVALID:{unit.unit_id}")
        found.append(record)
    return {
        "status": "QUALITY_PASS",
        "cell_id": cell_id,
        "expected_unit_count": len(expected),
        "completed_unit_count": len(found),
        "failed_unit_count": 0,
        "finite_outputs": True,
        "m2_identity": True,
        "mean_gradient_audit": True,
        "cost_profiler": True,
        "raw_artifacts_persisted": True,
        "statistical_conclusions": False,
    }


def normalized_gpu_inventory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Validate the four approved UUIDs and excluded PCI before launching."""

    normalized = []
    for row in rows:
        uuid = row.get("uuid")
        pci = row.get("pci_bus_id", row.get("pci"))
        if isinstance(uuid, str) and isinstance(pci, str):
            normalized.append({"uuid": uuid, "pci_bus_id": pci, **{str(k): v for k, v in row.items()}})
    return validate_gpu_inventory(normalized)


def nvidia_smi_inventory() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S27ExecutionBlocked(f"S27_GPU_INVENTORY_UNAVAILABLE:{type(error).__name__}") from error
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        rows.append({"uuid": parts[0], "pci_bus_id": parts[1], "memory_used_mib": parts[2] if len(parts) > 2 else None, "utilization_gpu": parts[3] if len(parts) > 3 else None})
    if not rows:
        raise S27ExecutionBlocked("S27_GPU_INVENTORY_EMPTY")
    return rows


@dataclass(frozen=True, slots=True)
class S27SubprocessSpec:
    cell_id: str
    gpu_uuid: str
    command: tuple[str, ...]
    log_path: Path


def build_s27_worker_command(
    *,
    python: str | Path,
    launcher_script: str | Path,
    data_root: str | Path,
    plan_ref: str,
    run_root: str,
    run_id: str,
    cell_id: str,
    gpu_uuid: str,
    materialization_index_ref: str,
    execution_evidence_ref: str,
) -> tuple[str, ...]:
    if gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S27ExecutionBlocked("S27_WORKER_COMMAND_UNAPPROVED_GPU")
    return (
        str(python),
        str(launcher_script),
        "--worker",
        "--data-root",
        str(data_root),
        "--plan-ref",
        plan_ref,
        "--run-root",
        run_root,
        "--run-id",
        run_id,
        "--cell-id",
        cell_id,
        "--gpu-uuid",
        gpu_uuid,
        "--materialization-index-ref",
        materialization_index_ref,
        "--execution-evidence-ref",
        execution_evidence_ref,
    )


class S27DetachedLauncher:
    """Four-slot dynamic queue for the six frozen waves."""

    def __init__(
        self,
        *,
        data_root: str | Path,
        plan_ref: str,
        run_root: str | Path,
        run_id: str,
        python: str | Path,
        launcher_script: str | Path,
        materialization_index_ref: str,
        execution_evidence_ref: str,
        approved_inventory: Sequence[Mapping[str, object]],
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.plan = load_s27_plan(self.data_root, plan_ref)
        self.plan_ref = plan_ref
        self.run_root = Path(run_root).resolve()
        self.run_id = run_id
        self.python = str(python)
        self.launcher_script = str(launcher_script)
        self.materialization_index_ref = materialization_index_ref
        self.execution_evidence_ref = execution_evidence_ref
        normalized_gpu_inventory(approved_inventory)
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.run_root.is_relative_to(self.data_root) is False:
            raise S27ExecutionBlocked("S27_RUN_ROOT_OUTSIDE_DATA_ROOT")

    def _status_path(self) -> Path:
        return self.run_root / "launcher-status.json"

    def _publish_status(self, status: str, *, waves: Mapping[str, Mapping[str, object]], reason: str | None = None) -> None:
        body: dict[str, object] = {
            "schema_version": S27_LAUNCH_STATUS_SCHEMA,
            "run_id": self.run_id,
            "plan_ref": self.plan_ref,
            "plan_hash": self.plan.artifact_hash,
            "status": status,
            "wave_order": list(EXPECTED_CELL_IDS),
            "waves": {key: dict(value) for key, value in sorted(waves.items())},
            "updated_at": _now(),
            "reason": reason,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        _write_once(self._status_path(), body, field="S27_LAUNCH_STATUS") if status in {"PREPARED", "SEALED"} else write_canonical_json(self._status_path(), body)

    def execute(self) -> dict[str, object]:
        waves: dict[str, Mapping[str, object]] = {}
        if self._status_path().exists():
            raise S27ExecutionBlocked("S27_LAUNCH_STATUS_EXISTS_REQUIRES_EXPLICIT_RECOVERY_AUDIT")
        self._publish_status("PREPARED", waves=waves)
        # The first 14M initialization wave is an explicit quality barrier.  No
        # 14M early/mid_late or 31M process is even created until its complete
        # operational conjunction is sealed.
        quality_cell = self.plan.cells[0]
        queue = deque((quality_cell,))
        # The six cells are checkpoint waves, not independent jobs.  The
        # frozen protocol requires initialization -> early -> mid_late for
        # 14M, followed by the same three stages for 31M.  Keep a dynamic
        # approved-GPU queue for detached/recovery semantics, but release only
        # the next canonical wave after its predecessor has sealed.  This
        # prevents a 31M child from starting while an earlier 14M wave is
        # incomplete and makes the ordering auditable in launcher-status.json.
        next_wave_index = 1
        running: dict[Future[tuple[str, int, str]], S27CellPlan] = {}
        # The queue uses each cell's frozen assignment; a card is returned only
        # after its child exits, so repeated GPU0/GPU1 cells cannot overlap.
        free = deque(APPROVED_GPU_UUIDS)
        quality_passed = False
        with ThreadPoolExecutor(max_workers=len(APPROVED_GPU_UUIDS)) as pool:
            while queue or running:
                while queue and free:
                    cell = queue[0]
                    if cell.assigned_gpu_uuid not in free:
                        # Preserve frozen assignment and wait for that card.
                        break
                    queue.popleft()
                    gpu = cell.assigned_gpu_uuid
                    free.remove(gpu)
                    spec = build_s27_worker_command(
                        python=self.python,
                        launcher_script=self.launcher_script,
                        data_root=self.data_root,
                        plan_ref=self.plan_ref,
                        run_root=self.run_root,
                        run_id=self.run_id,
                        cell_id=cell.cell_id,
                        gpu_uuid=gpu,
                        materialization_index_ref=self.materialization_index_ref,
                        execution_evidence_ref=self.execution_evidence_ref,
                    )
                    log_path = self.run_root / "logs" / f"{cell.cell_id.replace(':', '__')}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    env = os.environ.copy()
                    env.update({"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu, "NVIDIA_VISIBLE_DEVICES": gpu})
                    def _launch(spec: tuple[str, ...] = spec, log_path: Path = log_path, env: Mapping[str, str] = env, cell_id: str = cell.cell_id, gpu: str = gpu) -> tuple[str, int, str]:
                        with log_path.open("ab") as handle:
                            proc = subprocess.Popen(spec, cwd=self.launcher_script and str(Path(self.launcher_script).resolve().parents[2]), env=dict(env), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                            code = proc.wait()
                        return cell_id, int(code), _relative_ref(self.data_root, log_path, field="S27_LOG_REF")
                    future = pool.submit(_launch)
                    running[future] = cell
                    self._publish_status("RUNNING", waves=waves)
                if not running:
                    if queue:
                        raise S27ExecutionBlocked("S27_DYNAMIC_QUEUE_FROZEN_GPU_ASSIGNMENT_DEADLOCK")
                    break
                done_futures, _ = wait(tuple(running), return_when=FIRST_COMPLETED)
                for future in done_futures:
                    cell = running.pop(future)
                    free.append(cell.assigned_gpu_uuid)
                    try:
                        cell_id, code, log_ref = future.result()
                    except Exception as error:
                        result = {
                            "cell_id": cell.cell_id,
                            "gpu_uuid": cell.assigned_gpu_uuid,
                            "returncode": None,
                            "status": "FAILED",
                            "error": f"{type(error).__name__}:{error}",
                        }
                        waves[cell.cell_id] = result
                        self._publish_status("FAILED", waves=waves, reason=f"S27_CHILD_LAUNCH_FAILED:{cell.cell_id}")
                        raise S27ExecutionBlocked(f"S27_CHILD_LAUNCH_FAILED:{cell.cell_id}") from error
                    wave_seal = self.run_root / "wave-seals" / f"{cell.cell_id.replace(':', '__')}.json"
                    result: dict[str, object] = {"cell_id": cell_id, "gpu_uuid": cell.assigned_gpu_uuid, "returncode": code, "status": "COMPLETE" if code == 0 and wave_seal.is_file() else "FAILED", "log_ref": log_ref}
                    waves[cell_id] = result
                    self._publish_status("RUNNING", waves=waves)
                    if code != 0 or not wave_seal.is_file():
                        self._publish_status("FAILED", waves=waves, reason=f"S27_WORKER_FAILED:{cell_id}:{code}")
                        raise S27ExecutionBlocked(f"S27_WORKER_FAILED:{cell_id}:{code}")
                    if cell is quality_cell:
                        try:
                            quality = validate_s27_quality_wave(self.data_root, self.plan, self.run_root, cell_id=cell.cell_id)
                        except Exception as error:
                            self._publish_status("FAILED", waves=waves, reason=f"S27_QUALITY_WAVE_BLOCKED:{error}")
                            raise
                        waves[cell_id] = {**dict(result), "quality": quality}
                        quality_passed = True
                        if next_wave_index < len(self.plan.cells):
                            queue.append(self.plan.cells[next_wave_index])
                            next_wave_index += 1
                        self._publish_status("RUNNING", waves=waves)
                    else:
                        try:
                            reducer = load_s27_raw_records(self.data_root, self.plan, self.run_root, run_id=self.run_id)
                            failed = sum(record.status == "FAILED" for record in reducer.records)
                            fraction = failed / self.plan.frozen_inputs.completion_denominator
                        except Exception as error:
                            self._publish_status("FAILED", waves=waves, reason=f"S27_FAILURE_ACCOUNTING_BLOCKED:{error}")
                            raise
                        if fraction > self.plan.frozen_inputs.max_failure_fraction:
                            self._publish_status("FAILED", waves=waves, reason="S27_FAILURE_FRACTION_EXCEEDED")
                            raise S27ExecutionBlocked("S27_FAILURE_FRACTION_EXCEEDED")
                        if next_wave_index < len(self.plan.cells):
                            queue.append(self.plan.cells[next_wave_index])
                            next_wave_index += 1
        if set(waves) != set(EXPECTED_CELL_IDS):
            self._publish_status("FAILED", waves=waves, reason="S27_QUEUE_INCOMPLETE")
            raise S27ExecutionBlocked("S27_QUEUE_INCOMPLETE")
        if not quality_passed:
            self._publish_status("FAILED", waves=waves, reason="S27_QUALITY_WAVE_NOT_PASSED")
            raise S27ExecutionBlocked("S27_QUALITY_WAVE_NOT_PASSED")
        try:
            sealed = seal_s27_run(self.data_root, self.plan, self.run_root, run_id=self.run_id)
        except Exception as error:
            self._publish_status("FAILED", waves=waves, reason=f"{type(error).__name__}:{error}")
            raise
        result = {"status": "SEALED", "run_id": self.run_id, "plan_hash": self.plan.artifact_hash, "waves": waves, "sealed": {"manifest_hash": sealed["manifest"]["artifact_hash"], "gate_ref": sealed["gate_ref"]}}
        self._publish_status("SEALED", waves=waves)
        return result


__all__ = [
    "S27_ATTEMPT_SCHEMA",
    "S27_DEFAULT_MAX_ATTEMPTS",
    "S27_DEFAULT_M2_TOLERANCE",
    "S27DetachedLauncher",
    "S27ExecutionBlocked",
    "S27MaterializedCellInput",
    "S27ProductionWorker",
    "S27ProviderContext",
    "S27RetryPolicy",
    "S27SubprocessSpec",
    "S27_WAVE_SEAL_SCHEMA",
    "build_s27_torch_provider",
    "build_s27_worker_command",
    "load_s27_frozen_mappings",
    "load_s27_materialized_inputs",
    "load_s27_plan",
    "load_s27_raw_records",
    "load_s27_reference_views",
    "normalized_gpu_inventory",
    "nvidia_smi_inventory",
    "seal_s27_run",
    "validate_s27_quality_wave",
]
