#!/usr/bin/env python3
"""Materialize the formal S2.4 inputs and six-cell runtime contract.

The normal commit-materialization path is deliberately control-plane only: it
never downloads an asset or starts a task.  The explicit raw bootstrap path is
different by design: it retains raw inputs as nonformal candidates, then calls
the existing formal TaskRuntime S2.1 -> S2.2/S2.3 DAG to produce authoritative
commits.  That DAG performs only the short prerequisite checks needed to emit
the task contracts; it is not a training launch.

The source manifest is intentionally explicit.  In particular, no source is
discovered by globbing and no missing formal record is replaced by a fixture,
source-code hash, or a local validation report.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
from typing import Any, Final

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from param_importance_nlp.contracts import (  # noqa: E402
    ContractFreeze,
    FormalExecutionEvidence,
    ResolvedConfig,
    RuntimeCapabilityEvidence,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.status import GateRecord, GateStatus  # noqa: E402
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG  # noqa: E402
from param_importance_nlp.contracts.stage0_handoff import (  # noqa: E402
    validate_stage0_handoff,
)
from param_importance_nlp.contracts.stage1_handoff import (  # noqa: E402
    STAGE1_G1_EXIT_GATE_ID,
    STAGE1_G1_EXIT_TASK_ID,
    validate_stage1_exit_evidence,
)
from param_importance_nlp.contracts.g21_formal_handoff import (  # noqa: E402
    ALLOWED_DEVICES as G21_ALLOWED_DEVICES,
    EXCLUDED_PCI as G21_EXCLUDED_PCI,
    EXCLUDED_UUID as G21_EXCLUDED_UUID,
    load_g21_formal_handoff,
)
from param_importance_nlp.experiments.stage2_assets import (  # noqa: E402
    AssetResolutionManifest,
    CheckpointRecord,
    validate_formal_asset_identity,
)
from param_importance_nlp.experiments.stage2_s204_ids import (  # noqa: E402
    EXPECTED_CELL_IDS,
    canonical_cell_id,
    cell_path_component,
)
from param_importance_nlp.experiments.stage2_registry_qualification import (  # noqa: E402
    ASSET_RESOLUTION_AMENDMENT_SCHEMA,
    load_asset_resolution_input,
)
from param_importance_nlp.core.registry import ParameterRegistry  # noqa: E402
from param_importance_nlp.experiments.stage2_formal import (  # noqa: E402
    ReferenceSizingPlan,
)
from param_importance_nlp.experiments.preregistration import (  # noqa: E402
    validate_stage2_preregistration,
)
from param_importance_nlp.experiments.sampling import (  # noqa: E402
    DrawStreamManifest,
    SamplingPlan,
    STREAM_NAMES,
)
from param_importance_nlp.g3_runtime_assets import (  # noqa: E402
    FormalG3RuntimeAssets,
)
from param_importance_nlp.runtime import (  # noqa: E402
    TaskArtifactStore,
    TaskRunStatus,
    TaskRuntime,
    TaskRuntimeEnvironment,
    publish_canonical_immutable,
)
from param_importance_nlp.experiments.stage23_task_runners import (  # noqa: E402
    register_stage23_runners,
)
from param_importance_nlp.runtime.task_artifacts import (  # noqa: E402
    LoadedTaskArtifact,
    load_committed_task_artifact,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2  # noqa: E402


S204_TASK_ID: Final = "stage2.04_reference_target"
S204_SCHEMA: Final = "stage2-reference-sizing-plan-v1"
# G2.1 owns the PCI/UUID identity.  A current nvidia-smi index is resolved by
# the launcher at execution time and must never be materialized here.
EXCLUDED_PCI: Final = G21_EXCLUDED_PCI
EXCLUDED_UUID: Final = G21_EXCLUDED_UUID
DEFAULT_CANDIDATES: Final = (512, 1024, 2048, 4096)
DEFAULT_BLOCK_SIZE: Final = 32
DEFAULT_TOKENIZER_ASSET_ID: Final = "pythia-tokenizer"
DEFAULT_DATA_ASSET_ID: Final = "pile-selected-prefix"
S204_SIX_CELL_SCHEMA: Final = "stage2-s204-six-cell-manifest-v1"
S204_REGISTRY_SCHEMA: Final = "stage2-parameter-registry-artifact-v1"
S204_DELTA_SCHEMA: Final = "stage2-reference-delta-sci-v1"
S204_DELTA_PLAN_SCHEMA: Final = "stage2-reference-delta-sci-plan-v1"
# S2.2 task commits fill the existing r7 task-output namespace.  The resolved
# config already fixes that output directory and is never regenerated here.
S204_S22_CANONICAL_OUTPUT_DIR: Final = "evidence/stage2/s204/materialized-task-inputs-r7"
# New r8 control objects (environment/evidence/summary) are append-only and
# deliberately separate from the historical r7 G2.0/G2.1 extension files.
S204_S22_CONTROL_OUTPUT_DIR: Final = "evidence/stage2/s204/formal-s22-r8"
S204_S22_COMMIT_OUTPUT_DIR: Final = f"{S204_S22_CANONICAL_OUTPUT_DIR}/task-outputs/stage2-02"
S204_S22_CONFIG_REF: Final = f"{S204_S22_CANONICAL_OUTPUT_DIR}/configs/generated/stage2/stage2-02-resolved-config-v2.json"
# G3-v5 lineage is append-only: the historical g20/g21 extension files are
# retained and never overwritten by the S2.2 formal producer.
S22_G3_FORMAL_EXECUTION_G20_REF: Final = f"{S204_S22_CONTROL_OUTPUT_DIR}/formal-execution-g20-g3-v5.json"
S22_G3_FORMAL_EXECUTION_G21_REF: Final = f"{S204_S22_CONTROL_OUTPUT_DIR}/formal-execution-g21-g3-v5.json"
S22_G3_FORMAL_ENVIRONMENT_REF: Final = f"{S204_S22_CONTROL_OUTPUT_DIR}/environments/stage2-02-g3-v5.json"
S22_G3_READY_MANIFEST_COUNT: Final = 13
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_LOGICAL = re.compile(r"^[^\\/][^\\]*$")

TASK_INPUTS: Final = {
    "stage2.01_scope_hypotheses_and_preregistration": (
        "preregistration",
        "hypothesis_contract",
        "gate_record",
    ),
    "stage2.02_stage1_handoff_and_fixed_state_contract": (
        "handoff_manifest",
        "fixed_state_contract",
        "gate_record",
    ),
    "stage2.03_assets_checkpoints_and_sampling": (
        "sampling_plan",
        "draw_manifest",
        "asset_resolution",
        "gate_record",
    ),
}

STAGE1_TASK_ID: Final = "stage1.11_reporting_and_exit_gate"
STAGE1_10_TASK_ID: Final = "stage1.10_checkpoint_resume_and_artifacts"
STAGE1_10_TASK_INPUTS: Final = (
    "training_state_manifest",
    "resume_equivalence_report",
    "gate_record",
)
STAGE1_TASK_INPUTS: Final = (
    "stage_report",
    "requirements_matrix",
    "gate_summary",
    "delivery_manifest",
)
STAGE1_PAYLOAD_SCHEMAS: Final = {
    "stage_report": "stage1-s1-11-stage-report-v1",
    "requirements_matrix": "stage1-s1-11-requirements-matrix-v1",
    "gate_summary": "stage1-s1-11-gate-summary-v1",
    "delivery_manifest": "stage1-s1-11-delivery-manifest-v1",
}

_PAYLOAD_SCHEMAS: Final = {
    "preregistration": "stage2-preregistration-v1",
    "hypothesis_contract": "stage2-hypothesis-contract-v1",
    "handoff_manifest": "stage2-task-handoff-manifest-v1",
    "fixed_state_contract": "stage2-task-fixed-state-contract-v1",
    "sampling_plan": "sampling-plan-v1",
    "draw_manifest": "stage2-task-draw-manifest-v1",
    "asset_resolution": "stage2-task-asset-resolution-v1",
    # Gate candidates are intentionally NOT_RUN in runner outputs.  Formal
    # envelope qualification is separate from this scientific Gate candidate.
    "gate_record": "stage23-task-gate-candidate-v1",
}


class S204MaterializationError(RuntimeError):
    """A required formal source or identity is absent/invalid."""


class FormalDAGResult(dict[str, dict[str, str]]):
    """Mapping-compatible DAG result carrying the final evidence snapshot."""

    def __init__(
        self,
        refs_by_task: Mapping[str, Mapping[str, str]],
        *,
        final_evidence: FormalExecutionEvidence,
        final_evidence_ref: str,
        bridge_gate_refs: Mapping[str, str],
        authoritative_asset_ref: str,
        stage1_bridge_ref: str | None = None,
        stage1_bridge_config_ref: str | None = None,
        stage1_10_refs: Mapping[str, str] | None = None,
        stage1_11_refs: Mapping[str, str] | None = None,
        blocked_reason: str | None = None,
    ) -> None:
        super().__init__({task: dict(refs) for task, refs in refs_by_task.items()})
        self.final_evidence = final_evidence
        self.final_evidence_ref = final_evidence_ref
        self.bridge_gate_refs = dict(bridge_gate_refs)
        self.authoritative_asset_ref = authoritative_asset_ref
        self.stage1_bridge_ref = stage1_bridge_ref
        self.stage1_bridge_config_ref = stage1_bridge_config_ref
        self.stage1_10_refs = dict(stage1_10_refs or {})
        self.stage1_11_refs = dict(stage1_11_refs or {})
        self.blocked_reason = blocked_reason


def _error(code: str, detail: object = "") -> S204MaterializationError:
    suffix = f":{detail}" if detail != "" else ""
    return S204MaterializationError(f"S204_{code}{suffix}")


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _error("SHA256_INVALID", field)
    return value


def _logical(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _error("LOGICAL_REF_INVALID", field)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _error("LOGICAL_REF_INVALID", field)
    return path.as_posix()


def _safe_relative(data_root: Path, ref: str, field: str) -> Path:
    logical = _logical(ref, field)
    path = (data_root / Path(*PurePosixPath(logical).parts)).resolve()
    try:
        path.relative_to(data_root.resolve())
    except ValueError as error:
        raise _error("REF_ESCAPES_DATA_ROOT", field) from error
    return path


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("OBJECT_REQUIRED", field)
    return value


def _load_mapping(data_root: Path, ref: str, field: str) -> Mapping[str, Any]:
    path = _safe_relative(data_root, ref, field)
    try:
        return _mapping(load_canonical_json(path), field)
    except (OSError, TypeError, ValueError) as error:
        raise _error("SOURCE_UNREADABLE", f"{field}:{ref}") from error


def _load_raw_json_mapping(
    data_root: Path,
    ref: str,
    field: str,
    *,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    """Load a producer's ordinary JSON manifest with byte-hash binding.

    S2.3's legacy ``prefix_coverage.json`` is intentionally an ordinary JSON
    file (and contains absolute audit paths), not a canonical control-plane
    object.  Its declared ``manifest_sha256`` is the raw byte SHA, so it must
    not be forced through ``load_canonical_json`` or re-hashed as canonical
    JSON.
    """

    path = _safe_relative(data_root, ref, field)
    try:
        raw = path.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != _sha(
            expected_sha256, f"{field}.sha256"
        ):
            raise _error("SOURCE_HASH_MISMATCH", field)
        value = json.loads(raw.decode("utf-8"))
    except S204MaterializationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise _error("SOURCE_UNREADABLE", f"{field}:{ref}") from error
    return _mapping(value, field)


def _load_source_manifest(data_root: Path, value: str | Path) -> Mapping[str, Any]:
    """Load the CLI source manifest from DATA_ROOT or an explicitly contained path."""

    candidate = Path(value)
    if candidate.is_absolute():
        path = candidate.resolve()
        try:
            path.relative_to(data_root.resolve())
        except ValueError as error:
            raise _error("SOURCES_OUTSIDE_DATA_ROOT", str(value)) from error
        try:
            return _mapping(load_canonical_json(path), "sources")
        except (OSError, TypeError, ValueError) as error:
            raise _error("SOURCE_UNREADABLE", str(value)) from error
    return _load_mapping(data_root, candidate.as_posix(), "sources")


def _config_path(data_root: Path, value: str | Path) -> Path:
    """Resolve a base config from DATA_ROOT or the checked-in repository layer."""

    candidate = Path(value)
    if candidate.is_absolute():
        path = candidate.resolve()
    else:
        logical = _logical(candidate.as_posix(), "base_config")
        data_candidate = (data_root / Path(*PurePosixPath(logical).parts)).resolve()
        repo_candidate = (ROOT / Path(*PurePosixPath(logical).parts)).resolve()
        path = data_candidate if data_candidate.exists() else repo_candidate
    allowed_roots = (data_root.resolve(), ROOT.resolve())
    if not any(path == allowed or allowed in path.parents for allowed in allowed_roots):
        raise _error("BASE_CONFIG_OUTSIDE_ALLOWED_ROOT", str(value))
    return path


def _reject_nonformal_ref(ref: str, field: str) -> None:
    """Reject obvious tracked/source/fixture authorities before loading.

    ``require_formal=True`` is the authoritative scope check.  This extra
    path policy prevents a formal-looking copied JSON under a tracked fixture
    path from becoming a formal source by accident.
    """

    parts = set(PurePosixPath(ref).parts)
    # Generated formal v2 configs are themselves content-addressed evidence;
    # only checked-in config layers remain forbidden as formal source refs.
    generated_config = (
        "configs" in parts
        and "generated" in parts
        and "stage2" in parts
    )
    forbidden = parts.intersection({"fixtures", "src", "ops", "configs"})
    if forbidden and not (forbidden == {"configs"} and generated_config):
        raise _error("FIXTURE_OR_CODE_REF_FORBIDDEN", f"{field}:{ref}")


def _source_ref(value: object, field: str) -> str:
    ref = _logical(value, field)
    _reject_nonformal_ref(ref, field)
    return ref


def _task_refs(raw: object, *, task_id: str) -> dict[str, str]:
    expected = TASK_INPUTS[task_id]
    if isinstance(raw, Mapping):
        if set(raw) != set(expected):
            raise _error("TASK_INPUT_SET_INVALID", task_id)
        return {kind: _source_ref(raw[kind], f"{task_id}.{kind}") for kind in expected}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        if len(raw) != len(expected):
            raise _error("TASK_INPUT_COUNT_INVALID", task_id)
        return {
            kind: _source_ref(ref, f"{task_id}.{kind}")
            for kind, ref in zip(expected, raw, strict=True)
        }
    raise _error("TASK_INPUT_MAPPING_REQUIRED", task_id)


def _validate_payload(kind: str, payload: Mapping[str, Any]) -> None:
    expected = _PAYLOAD_SCHEMAS[kind]
    if payload.get("schema_version") != expected:
        raise _error("PAYLOAD_SCHEMA_MISMATCH", f"{kind}:{payload.get('schema_version')!r}")
    # A formal envelope may carry a scientific candidate (whose nested
    # formal_eligible is deliberately false), but it may never carry a local
    # scope or a synthetic provider.
    if payload.get("scope") == "local_fixture":
        raise _error("LOCAL_SCOPE_PAYLOAD_FORBIDDEN", kind)
    provider = payload.get("provider")
    if isinstance(provider, Mapping) and str(provider.get("provider_kind", "")).startswith(
        ("synthetic", "local")
    ):
        raise _error("SYNTHETIC_PROVIDER_FORBIDDEN", kind)


def _load_formal_source(
    data_root: Path,
    ref: str,
    *,
    task_id: str,
    artifact_kind: str,
) -> LoadedTaskArtifact:
    try:
        loaded = load_committed_task_artifact(
            data_root,
            ref,
            require_formal=True,
        )
    except Exception as error:
        raise _error("FORMAL_COMMIT_REQUIRED", f"{task_id}.{artifact_kind}:{ref}") from error
    if loaded.identity.task_id != task_id or loaded.identity.artifact_kind != artifact_kind:
        raise _error("TASK_ARTIFACT_IDENTITY_MISMATCH", f"{task_id}.{artifact_kind}")
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise _error("FORMAL_SCOPE_REQUIRED", f"{task_id}.{artifact_kind}")
    for source_ref in loaded.source_refs:
        _reject_nonformal_ref(source_ref, f"{task_id}.{artifact_kind}.source_ref")
    _validate_payload(artifact_kind, loaded.payload)
    return loaded


def _assert_source_file(root: Path, ref: str, field: str, *, sha256: str | None = None) -> str:
    """Require a source object to exist and, when supplied, match its byte hash."""

    path = _safe_relative(root, ref, field)
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise _error("SOURCE_UNREADABLE", f"{field}:{ref}") from error
    if sha256 is not None and digest != _sha(sha256, f"{field}.sha256"):
        raise _error("SOURCE_HASH_MISMATCH", field)
    return digest


def _stage1_task_refs(source: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the actual four S1.11 TaskRuntime commits.

    The released S1.11 index/role files are evidence, not task-output commits.
    This loader intentionally has no import/bridge path: a role path, source
    file, fixture, or code hash can never satisfy a Stage 2 predecessor.
    """

    raw = source.get("stage1_11_task_outputs")
    if isinstance(raw, Mapping) and STAGE1_TASK_ID in raw:
        raw = raw[STAGE1_TASK_ID]
    if not isinstance(raw, Mapping) or set(raw) != set(STAGE1_TASK_INPUTS):
        raise _error("STAGE1_11_FORMAL_COMMITS_REQUIRED")
    return {
        kind: _source_ref(raw[kind], f"stage1_11_task_outputs.{kind}")
        for kind in STAGE1_TASK_INPUTS
    }


def _stage1_10_task_refs(source: Mapping[str, Any]) -> Mapping[str, str]:
    """Return the actual three S1.10 TaskRuntime commits.

    S1.10 is not a direct S2.1 predecessor in the task catalog, but its group
    success and commit identities are part of the formal Stage 1 handoff used
    by the G2.1 adapter.  It is therefore loaded and bound, never inferred.
    """

    raw = source.get("stage1_10_task_outputs")
    if isinstance(raw, Mapping) and STAGE1_10_TASK_ID in raw:
        raw = raw[STAGE1_10_TASK_ID]
    if not isinstance(raw, Mapping) or set(raw) != set(STAGE1_10_TASK_INPUTS):
        raise _error("STAGE1_10_FORMAL_COMMITS_REQUIRED")
    return {
        kind: _source_ref(raw[kind], f"stage1_10_task_outputs.{kind}")
        for kind in STAGE1_10_TASK_INPUTS
    }


def _load_stage1_commit_group(
    root: Path,
    refs: Mapping[str, str],
    *,
    task_id: str,
    kinds: Sequence[str],
) -> dict[str, LoadedTaskArtifact]:
    """Load one completed formal TaskRuntime group without promotion."""

    loaded: dict[str, LoadedTaskArtifact] = {}
    for kind in kinds:
        ref = _source_ref(refs[kind], f"{task_id}.{kind}")
        try:
            item = load_committed_task_artifact(root, ref, require_formal=True)
        except Exception as error:
            raise _error("STAGE1_FORMAL_COMMIT_REQUIRED", f"{task_id}.{kind}") from error
        if item.identity.task_id != task_id or item.identity.artifact_kind != kind:
            raise _error("STAGE1_FORMAL_COMMIT_IDENTITY_MISMATCH", f"{task_id}.{kind}")
        if item.run_intent != "formal" or item.identity.formal_eligible is not True:
            raise _error("STAGE1_FORMAL_SCOPE_REQUIRED", f"{task_id}.{kind}")
        for source_ref in item.source_refs:
            _reject_nonformal_ref(source_ref, f"{task_id}.{kind}.source_ref")
        payload = item.payload
        # Explicit candidate/failure statuses are never a successful group.
        status_values = {
            str(payload.get(name)).upper()
            for name in ("status", "gate_status", "exit_verdict", "overall_status")
            if payload.get(name) is not None
        }
        if status_values.intersection({"NOT_RUN", "BLOCKED", "FAIL", "FAILED", "FORMAL_CANDIDATE"}):
            raise _error("STAGE1_GROUP_NOT_SUCCESS", f"{task_id}.{kind}")
        loaded[kind] = item
    config_hashes = {item.identity.config_hash for item in loaded.values()}
    if len(config_hashes) != 1:
        raise _error("STAGE1_GROUP_CONFIG_IDENTITY_MISMATCH", task_id)
    source_sets = {item.source_refs for item in loaded.values()}
    if len(source_sets) != 1:
        raise _error("STAGE1_GROUP_SOURCE_CLOSURE_MISMATCH", task_id)
    return loaded


def _load_stage1_formal_commits(
    root: Path,
    source: Mapping[str, Any],
    *,
    stage1_index_ref: str,
    base_config_ref: str,
    output_dir: str,
) -> tuple[dict[str, LoadedTaskArtifact], Any, str | None, str | None]:
    """Load the actual S1.10/S1.11 groups; never bridge role documents."""

    try:
        stage1_exit = validate_stage1_exit_evidence(root, stage1_index_ref)
    except Exception as error:
        raise _error("STAGE1_EXIT_REQUIRED", stage1_index_ref) from error
    s110_refs = _stage1_10_task_refs(source)
    s111_refs = _stage1_task_refs(source)
    _load_stage1_commit_group(
        root,
        s110_refs,
        task_id=STAGE1_10_TASK_ID,
        kinds=STAGE1_10_TASK_INPUTS,
    )
    loaded = _load_stage1_commit_group(
        root,
        s111_refs,
        task_id=STAGE1_TASK_ID,
        kinds=STAGE1_TASK_INPUTS,
    )
    # The tuple shape is kept for callers from the previous API.  None is
    # intentional: S1.11 commits and their producer config are authoritative;
    # no Stage2 bridge config/evidence is manufactured here.
    return loaded, stage1_exit, None, None


_STAGE1_ROLES: Final = (
    "formal_observation",
    "stage_report",
    "delivery_manifest",
    "gate_summary",
    "requirements_matrix",
    "validation",
    "replay_validation",
)


def _stage1_role_ref_map(root: Path, index_ref: str) -> dict[str, str]:
    """Resolve every S1.11 role ref relative to the validated index."""

    index = _load_mapping(root, index_ref, "stage1_g1_exit")
    role_refs = _mapping(index.get("role_refs"), "stage1.role_refs")
    base = PurePosixPath(index_ref).parent
    resolved: dict[str, str] = {}
    for role in _STAGE1_ROLES:
        role_ref = role_refs.get(role)
        if not isinstance(role_ref, str):
            raise _error("STAGE1_ROLE_REF_MISSING", role)
        absolute = _safe_relative(
            root,
            f"{base.as_posix()}/{role_ref}",
            f"stage1.role.{role}",
        )
        if not absolute.is_file():
            raise _error("STAGE1_ROLE_REF_MISSING", role)
        resolved[role] = absolute.relative_to(root).as_posix()
    return resolved


def _stage1_role_refs(root: Path, index_ref: str) -> tuple[str, ...]:
    """Resolve every S1.11 role ref for bridge evidence lineage."""

    resolved = _stage1_role_ref_map(root, index_ref)
    return tuple(resolved[role] for role in _STAGE1_ROLES)


def _validate_stage1_bridge(
    root: Path,
    *,
    index_ref: str,
    bridge_ref: str,
    config_ref: str,
) -> None:
    """Re-read the S1.11 role-to-config bridge and verify every binding."""

    index = _source_ref(index_ref, "stage1_g1_exit")
    bridge = _source_ref(bridge_ref, "stage1_bridge")
    config = _source_ref(config_ref, "stage1_bridge_config")
    value = _load_mapping(root, bridge, "stage1_bridge")
    if value.get("schema_version") != "stage2-s204-stage1-bridge-v1" or value.get("status") != "PASS":
        raise _error("STAGE1_BRIDGE_EVIDENCE_INVALID")
    if value.get("formal_eligible") is not True or value.get("task_id") != STAGE1_TASK_ID:
        raise _error("STAGE1_BRIDGE_SCOPE_INVALID")
    if value.get("gate_id") != f"stage1.{STAGE1_G1_EXIT_GATE_ID}" or value.get("index_ref") != index:
        raise _error("STAGE1_BRIDGE_INDEX_BINDING_INVALID")
    declared_source_refs = value.get("source_refs")
    expected_roles = _stage1_role_ref_map(root, index)
    if value.get("role_refs") != expected_roles:
        raise _error("STAGE1_BRIDGE_ROLE_REF_BINDING_INVALID")
    if not isinstance(declared_source_refs, list) or set(declared_source_refs) != {index, *expected_roles.values()}:
        raise _error("STAGE1_BRIDGE_SOURCE_BINDING_INVALID")
    try:
        exit_evidence = validate_stage1_exit_evidence(root, index)
        role_sha = dict(exit_evidence.role_sha256)
    except Exception as error:
        raise _error("STAGE1_BRIDGE_STAGE1_INVALID") from error
    if value.get("index_sha256") != exit_evidence.index_sha256 or value.get("index_artifact_hash") != exit_evidence.index_artifact_hash:
        raise _error("STAGE1_BRIDGE_INDEX_HASH_INVALID")
    if value.get("role_sha256") != role_sha or value.get("execution_commit") != exit_evidence.execution_commit:
        raise _error("STAGE1_BRIDGE_ROLE_HASH_INVALID")
    payload_hashes = value.get("payload_hashes")
    if not isinstance(payload_hashes, Mapping) or set(payload_hashes) != set(STAGE1_TASK_INPUTS):
        raise _error("STAGE1_BRIDGE_PAYLOAD_HASH_SET_INVALID")
    for kind in STAGE1_TASK_INPUTS:
        payload = _load_mapping(root, expected_roles[kind], f"stage1.role.{kind}")
        if payload_hashes[kind] != canonical_json_hash(dict(payload)):
            raise _error("STAGE1_BRIDGE_PAYLOAD_HASH_INVALID", kind)
    if value.get("bridge_config_ref") != config:
        raise _error("STAGE1_BRIDGE_CONFIG_REF_INVALID")
    try:
        resolved = ResolvedConfigV2.from_mapping(_load_mapping(root, config, "stage1_bridge_config"))
    except Exception as error:
        raise _error("STAGE1_BRIDGE_CONFIG_INVALID") from error
    if value.get("bridge_config_hash") != resolved.config_hash or value.get("bridge_config_full_hash") != resolved.full_hash:
        raise _error("STAGE1_BRIDGE_CONFIG_HASH_INVALID")
    declared_artifact_hash = value.get("artifact_hash")
    if not isinstance(declared_artifact_hash, str) or canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ) != declared_artifact_hash:
        raise _error("STAGE1_BRIDGE_ARTIFACT_HASH_INVALID")


def _publish_authoritative_asset_manifest(
    root: Path,
    *,
    source: Mapping[str, Any],
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
    output_dir: str,
) -> tuple[str, AssetResolutionManifest]:
    """Load an already-published formal S2.3 six-checkpoint manifest.

    S2.3 is the canonical producer for this object.  This function is only a
    strict reader: it does not unwrap a task candidate, copy an asset payload,
    infer G3 aliases, or republish a manifest under a new authority.
    """

    value: Mapping[str, Any] | None = None
    source_ref: str | None = None
    for key in ("stage2_asset_resolution_manifest", "stage2_asset_resolution"):
        candidate = source.get(key)
        if isinstance(candidate, str):
            source_ref = _source_ref(candidate, key)
            value = _load_mapping(root, source_ref, key)
            break
    if value is None and raw_refs is not None:
        raise _error("CANDIDATE_ASSET_INPUTS_FORBIDDEN")
    if value is None or source_ref is None:
        raise _error("FORMAL_ASSET_MANIFEST_REQUIRED")
    declared_sha = source.get("stage2_asset_resolution_sha256")
    source_sha256 = _assert_source_file(
        root,
        source_ref,
        "stage2_asset_resolution",
        sha256=(str(declared_sha) if declared_sha is not None else None),
    )
    payload = value
    if value.get("schema_version") == "stage2-task-asset-resolution-v1":
        # A TaskRuntime S2.3 output is a candidate payload, not an asset
        # authority.  Only the existing direct manifest can seed the runner.
        raise _error("CANDIDATE_ASSET_INPUTS_FORBIDDEN", source_ref)
    if value.get("schema_version") == ASSET_RESOLUTION_AMENDMENT_SCHEMA:
        # Registry qualification owns the amendment contract.  In particular,
        # it verifies the immutable parent and all six qualification refs and
        # hashes before exposing the materialized v1 manifest.  Do not copy or
        # interpret any nested field locally: the loader is the sole authority.
        try:
            payload = load_asset_resolution_input(
                _safe_relative(root, source_ref, "stage2_asset_resolution"),
                root=root,
                data_root=root,
            )
        except Exception as error:
            raise _error("FORMAL_ASSET_MANIFEST_INVALID", source_ref) from error
    try:
        manifest = AssetResolutionManifest.from_mapping(dict(payload))
        validate_formal_asset_identity(manifest)
    except Exception as error:
        raise _error("FORMAL_ASSET_MANIFEST_INVALID", source_ref) from error
    if manifest.scope != "formal" or manifest.status != "READY":
        raise _error("FORMAL_ASSET_MANIFEST_NOT_READY", source_ref)
    if value.get("schema_version") != ASSET_RESOLUTION_AMENDMENT_SCHEMA:
        return source_ref, manifest

    # TaskRuntime's S2.3 runner intentionally accepts only a direct v1
    # AssetResolutionManifest.  Publish the loader's verified materialization
    # under the append-only run root and retain the amendment's ref/hash in a
    # separate lineage object rather than weakening that runner contract.
    authoritative_ref = PurePosixPath(output_dir, "asset-resolution-manifest.json").as_posix()
    authoritative_path = _safe_relative(root, authoritative_ref, "asset_resolution_output")
    publish_canonical_immutable(authoritative_path, manifest.to_dict())
    if _load_mapping(root, authoritative_ref, "asset_resolution_output") != manifest.to_dict():
        raise _error("FORMAL_ASSET_MANIFEST_ROUND_TRIP_DRIFT", authoritative_ref)
    lineage_ref = PurePosixPath(output_dir, "asset-resolution-input-lineage.json").as_posix()
    lineage: dict[str, Any] = {
        "schema_version": "stage2-s204-asset-resolution-input-lineage-v1",
        "input_ref": source_ref,
        "input_sha256": source_sha256,
        "input_schema_version": ASSET_RESOLUTION_AMENDMENT_SCHEMA,
        "materialized_asset_resolution_ref": authoritative_ref,
        "materialized_asset_resolution_hash": manifest.digest,
    }
    lineage["lineage_hash"] = canonical_json_hash(lineage)
    publish_canonical_immutable(
        _safe_relative(root, lineage_ref, "asset_resolution_input_lineage_output"),
        lineage,
    )
    if _load_mapping(root, lineage_ref, "asset_resolution_input_lineage") != lineage:
        raise _error("FORMAL_ASSET_MANIFEST_LINEAGE_ROUND_TRIP_DRIFT", lineage_ref)
    return authoritative_ref, manifest


def _find_schema_payload(value: object, schema_version: str) -> tuple[Mapping[str, Any], ...]:
    """Return nested payloads with one exact schema, without accepting paths."""

    found: list[Mapping[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if node.get("schema_version") == schema_version:
                found.append(node)
                return
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(value)
    return tuple(found)


def _load_formal_contract_freeze(
    root: Path,
    ref: str | None,
    *,
    stage: int,
) -> tuple[str, ContractFreeze]:
    """Load one formal ContractFreeze commit for the TaskRuntime environment."""

    if ref is None:
        raise _error("CONTRACT_STAGE_REF_REQUIRED", stage)
    logical_ref = _source_ref(ref, f"contract_stage_{stage}")
    try:
        loaded = load_committed_task_artifact(root, logical_ref, require_formal=True)
    except Exception as error:
        raise _error("CONTRACT_STAGE_FORMAL_COMMIT_REQUIRED", f"stage{stage}:{logical_ref}") from error
    for source_ref in loaded.source_refs:
        _reject_nonformal_ref(source_ref, f"contract_stage_{stage}.source_ref")
    matches = _find_schema_payload(loaded.payload, "contract-freeze-v1")
    if len(matches) != 1:
        raise _error("CONTRACT_STAGE_PAYLOAD_NOT_UNIQUE", f"stage{stage}:{logical_ref}")
    try:
        freeze = ContractFreeze.from_mapping(dict(matches[0]))
    except Exception as error:
        raise _error("CONTRACT_STAGE_PAYLOAD_INVALID", f"stage{stage}:{logical_ref}") from error
    if freeze.stage != stage or not freeze.formal_eligible:
        raise _error("CONTRACT_STAGE_IDENTITY_MISMATCH", f"stage{stage}:{logical_ref}")
    return logical_ref, freeze


def _publish_contract_document(
    root: Path,
    freeze: ContractFreeze,
    *,
    output_ref: str,
) -> str:
    """Publish the payload document used by stage23's direct freeze loader."""

    target_ref = PurePosixPath(output_ref).as_posix()
    target = _safe_relative(root, target_ref, "contract_freeze_document_output")
    publish_canonical_immutable(target, freeze.to_dict())
    try:
        reread = ContractFreeze.from_mapping(load_canonical_json(target))
    except Exception as error:
        raise _error("CONTRACT_FREEZE_DOCUMENT_ROUND_TRIP_DRIFT", target_ref) from error
    if reread.artifact_hash != freeze.artifact_hash:
        raise _error("CONTRACT_FREEZE_DOCUMENT_HASH_DRIFT", target_ref)
    return target_ref


def materialize_formal_task_inputs(
    data_root: str | Path,
    source_refs: Mapping[str, Mapping[str, str] | Sequence[str]],
    *,
    output_dir: str = "evidence/stage2/s204/materialized-task-inputs",
) -> dict[str, dict[str, str]]:
    """Load and content-address republish all S2.1/S2.2/S2.3 formal commits.

    Every source is validated before the first destination write, so a missing
    predecessor cannot leave a partially materialized formal set.  Returned
    refs are ordered by the catalog artifact order and are directly usable as
    S2.4 ``orchestration.input_result_refs``.
    """

    root = Path(data_root).resolve()
    expected_tasks = tuple(TASK_INPUTS)
    if set(source_refs) != set(expected_tasks):
        raise _error("TASK_SET_INVALID", sorted(set(source_refs) ^ set(expected_tasks)))
    loaded: dict[str, dict[str, LoadedTaskArtifact]] = {}
    for task_id in expected_tasks:
        refs = _task_refs(source_refs[task_id], task_id=task_id)
        task_loaded = {
            kind: _load_formal_source(
                root,
                ref,
                task_id=task_id,
                artifact_kind=kind,
            )
            for kind, ref in refs.items()
        }
        config_hashes = {item.identity.config_hash for item in task_loaded.values()}
        if len(config_hashes) != 1:
            raise _error("TASK_CONFIG_IDENTITY_MISMATCH", task_id)
        loaded[task_id] = task_loaded

    stores = {
        task_id: TaskArtifactStore(root, f"{output_dir}/{task_id.replace('.', '-')}")
        for task_id in expected_tasks
    }
    result: dict[str, dict[str, str]] = {}
    for task_id in expected_tasks:
        task_result: dict[str, str] = {}
        config_hash = next(iter({item.identity.config_hash for item in loaded[task_id].values()}))
        for kind in TASK_INPUTS[task_id]:
            source = loaded[task_id][kind]
            published = stores[task_id].publish(
                task_id=task_id,
                artifact_kind=kind,
                config_hash=config_hash,
                run_intent="formal",
                payload=source.payload,
                formal_eligible=True,
                source_refs=source.source_refs,
            )
            # A destination commit is not allowed to change the immutable
            # payload identity.  Different envelope/source refs are expected;
            # the content-addressed payload itself must be exactly retained.
            round_trip = load_committed_task_artifact(root, published.commit_ref, require_formal=True)
            if round_trip.payload != source.payload:
                raise _error("PUBLISHED_PAYLOAD_DRIFT", f"{task_id}.{kind}")
            task_result[kind] = published.commit_ref
        result[task_id] = task_result
    return result


def _raw_task_specs(raw: object, *, task_id: str) -> dict[str, Mapping[str, Any]]:
    """Normalize explicit raw bootstrap declarations.

    Raw bootstrap is intentionally a different input mode from commit
    materialization.  Each item must name a DATA_ROOT JSON payload and the
    formal config hash which produced it; no hash is inferred from source code
    or payload bytes.
    """

    expected = TASK_INPUTS[task_id]
    if not isinstance(raw, Mapping) or set(raw) != set(expected):
        raise _error("RAW_TASK_INPUT_SET_INVALID", task_id)
    result: dict[str, Mapping[str, Any]] = {}
    for kind in expected:
        item = raw[kind]
        if isinstance(item, str):
            raise _error("RAW_CONFIG_HASH_REQUIRED", f"{task_id}.{kind}")
        if not isinstance(item, Mapping) or set(item) - {"ref", "config_hash", "source_refs"}:
            raise _error("RAW_TASK_SPEC_INVALID", f"{task_id}.{kind}")
        if not isinstance(item.get("ref"), str) or not isinstance(item.get("config_hash"), str):
            raise _error("RAW_TASK_SPEC_INVALID", f"{task_id}.{kind}")
        _sha(item["config_hash"], f"{task_id}.{kind}.config_hash")
        refs = item.get("source_refs", ())
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes, bytearray)):
            raise _error("RAW_SOURCE_REFS_INVALID", f"{task_id}.{kind}")
        normalized = dict(item)
        normalized["ref"] = _source_ref(item["ref"], f"raw.{task_id}.{kind}.ref")
        normalized["source_refs"] = tuple(
            _source_ref(ref, f"raw.{task_id}.{kind}.source_refs") for ref in refs
        )
        if task_id != "stage2.01_scope_hypotheses_and_preregistration" and not normalized["source_refs"]:
            raise _error("RAW_SOURCE_REFS_REQUIRED", f"{task_id}.{kind}")
        result[kind] = normalized
    return result


def _validate_raw_payload(
    task_id: str, artifact_kind: str, payload: Mapping[str, Any]
) -> None:
    """Apply the existing semantic validators to one uncommitted raw payload."""

    _validate_payload(artifact_kind, payload)
    if artifact_kind == "preregistration":
        validate_stage2_preregistration(payload)
    elif artifact_kind == "hypothesis_contract":
        supplied = payload.get("hypothesis_contract_hash")
        body = {key: value for key, value in payload.items() if key != "hypothesis_contract_hash"}
        if supplied != canonical_json_hash(body):
            raise _error("RAW_HYPOTHESIS_HASH_INVALID", task_id)
    elif artifact_kind == "handoff_manifest":
        if payload.get("scope") != "formal" or payload.get("status") != "FORMAL_CANDIDATE" or payload.get("formal_eligible") is not False:
            raise _error("RAW_HANDOFF_SCOPE_INVALID", task_id)
        for field in ("upstream_binding_hash", "provider_state_digest", "registry_hash"):
            _sha(payload.get(field), f"{task_id}.{artifact_kind}.{field}")
    elif artifact_kind == "fixed_state_contract":
        if payload.get("status") != "FORMAL_CANDIDATE" or payload.get("formal_eligible") is not False:
            raise _error("RAW_FIXED_STATE_SCOPE_INVALID", task_id)
        for field in ("provider_state_digest", "registry_hash"):
            _sha(payload.get(field), f"{task_id}.{artifact_kind}.{field}")
        if payload.get("mutation_policy") != "read_only_gradient_queries":
            raise _error("RAW_FIXED_STATE_MUTATION_POLICY_INVALID", task_id)
    elif artifact_kind == "sampling_plan":
        SamplingPlan.from_mapping(payload)
    elif artifact_kind == "draw_manifest":
        draws = payload.get("draws")
        streams = payload.get("stream_manifests")
        if not isinstance(draws, list) or not isinstance(streams, Mapping) or set(streams) != set(STREAM_NAMES):
            raise _error("RAW_DRAW_MANIFEST_INVALID", task_id)
        for stream, value in streams.items():
            if not isinstance(value, Mapping):
                raise _error("RAW_DRAW_STREAM_INVALID", f"{task_id}:{stream}")
            DrawStreamManifest.from_mapping(dict(value))
        if payload.get("draw_id_unique") is not True or payload.get("replay_hash") != canonical_json_hash(draws):
            raise _error("RAW_DRAW_REPLAY_INVALID", task_id)
    elif artifact_kind == "asset_resolution":
        _formal_s23_asset_manifest(payload, field=f"raw.{task_id}.{artifact_kind}")
    elif artifact_kind == "gate_record":
        if payload.get("gate_status") != "NOT_RUN" or payload.get("formal_eligible") is not False:
            raise _error("RAW_GATE_CANDIDATE_INVALID", task_id)


def bootstrap_formal_task_inputs(
    data_root: str | Path,
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_dir: str = "evidence/stage2/s204/materialized-task-inputs",
) -> dict[str, dict[str, str]]:
    """Reject raw/candidate bootstrap.

    A prior implementation published ``local_fixture`` candidate commits and
    then fed those paths to a formal DAG.  That made a caller-controlled JSON
    look like a producer result.  The only supported bootstrap now is the
    canonical TaskRuntime producer invoked by :func:`execute_formal_predecessor_dag`.
    """

    del data_root, raw_refs, output_dir
    raise _error("CANDIDATE_BOOTSTRAP_FORBIDDEN")


def _publish_candidate_asset_manifest(
    root: Path,
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_dir: str,
) -> str:
    del root, raw_refs, output_dir
    raise _error("CANDIDATE_ASSET_PROMOTION_FORBIDDEN")


def _build_formal_predecessor_environment(
    root: Path,
    *,
    formal_execution_ref: str,
    source: Mapping[str, Any],
    candidate_asset_ref: str,
) -> TaskRuntimeEnvironment:
    """Build only the evidence snapshot needed by the S2.1--S2.3 DAG."""

    formal_value = _load_mapping(root, formal_execution_ref, "formal_execution")
    try:
        evidence = FormalExecutionEvidence.from_mapping(formal_value)
        evidence.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_INVALID", formal_execution_ref) from error

    stage0_ref = _source_ref(source.get("stage0_handoff"), "stage0_handoff")
    stage1_ref = _source_ref(source.get("stage1_g1_exit"), "stage1_g1_exit")
    try:
        validate_stage0_handoff(root, stage0_ref, require_ready=True)
        validate_stage1_exit_evidence(root, stage1_ref)
    except Exception as error:
        raise _error("STAGE0_STAGE1_READY_REQUIRED") from error

    contract_refs: dict[int, str] = {}
    freezes: dict[int, ContractFreeze] = {}
    stage2_ref = source.get("contract_stage_2", source.get("contract_freeze"))
    for stage, ref in (
        (0, source.get("contract_stage_0")),
        (1, source.get("contract_stage_1")),
        (2, stage2_ref),
    ):
        loaded_ref, freeze = _load_formal_contract_freeze(root, ref, stage=stage)
        contract_refs[stage] = loaded_ref
        freezes[stage] = freeze
    if freezes[2].artifact_hash != evidence.contract_freeze_hash:
        raise _error("CONTRACT_FREEZE_IDENTITY_MISMATCH", contract_refs[2])
    contract_document_ref = _publish_contract_document(
        root,
        freezes[2],
        output_ref=f"{candidate_asset_ref.rsplit('/', 1)[0]}/contract-freeze-stage2.json",
    )

    g3_ref = _source_ref(source.get("g3_resolution"), "g3_resolution")
    try:
        g3 = load_committed_task_artifact(root, g3_ref, require_formal=True)
        if g3.identity.artifact_kind != "asset_resolution":
            raise ValueError("wrong G3 artifact kind")
        FormalG3RuntimeAssets.load(root, g3_ref)
    except Exception as error:
        raise _error("G3_FORMAL_COMMIT_REQUIRED", g3_ref) from error

    gpu_ref, allowed_devices = _load_gpu_health_identity(
        root,
        source.get("g21_handoff", source.get("gpu_health")),
        expected_stage1_ref=stage1_ref,
    )
    gates = _load_formal_gate_refs(
        root,
        _mapping(source.get("gate_refs"), "gate_refs"),
        required=("stage1.G1-EXIT",),
    )
    capability_values = {
        name: _load_capability(
            root,
            _mapping(source.get("capability_refs"), "capability_refs").get(name),
            name,
        )
        for name in ("server", "cuda", "model_assets", "data_assets")
    }
    cuda_metadata = capability_values["cuda"][1].metadata
    runtime_devices = cuda_metadata.get("allowed_devices")
    if runtime_devices is not None:
        runtime_pairs = tuple(
            (str(item.get("pci_bus_id")), str(item.get("uuid")))
            for item in runtime_devices
            if isinstance(item, Mapping)
        )
        if runtime_pairs != allowed_devices:
            raise _error("GPU_RUNTIME_INVENTORY_DRIFT")
    runtime_excluded = cuda_metadata.get("excluded_device")
    if runtime_excluded is not None and (
        not isinstance(runtime_excluded, Mapping)
        or str(runtime_excluded.get("pci_bus_id")) != EXCLUDED_PCI
        or str(runtime_excluded.get("uuid")) != EXCLUDED_UUID
    ):
        raise _error("GPU_RUNTIME_EXCLUDED_IDENTITY_DRIFT")

    evidence_refs: dict[str, str] = {
        "formal_execution": formal_execution_ref,
        "stage0_handoff": stage0_ref,
        "stage1_g1_exit": stage1_ref,
        "contract_stage_0": contract_refs[0],
        "contract_stage_1": contract_refs[1],
        "contract_stage_2": contract_refs[2],
        "contract_freeze": contract_document_ref,
        "g3_resolution": g3_ref,
        "stage2_asset_resolution": candidate_asset_ref,
        "gpu_health": gpu_ref,
    }
    evidence_refs.update({f"gate_{key.replace('.', '_').replace('-', '_').lower()}": ref for key, ref in gates.items()})
    evidence_refs.update({f"capability_{key}": value[0] for key, value in capability_values.items()})
    return TaskRuntimeEnvironment(
        capabilities=frozenset(capability_values),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset(gates),
        evidence_refs=evidence_refs,
    )


def _publish_bridge_config(
    root: Path,
    *,
    base_config_ref: str,
    task_id: str,
    input_refs: Sequence[str],
    output_ref: str,
) -> tuple[ResolvedConfigV2, str]:
    """Persist a dedicated v2 config for an evidence bridge.

    Historical Stage1 role documents do not carry a config hash.  The bridge
    therefore has its own fully resolved identity; it records the role/index
    hashes as evidence instead of pretending that those documents were
    produced by this config.
    """

    config = _formal_dag_config(
        root,
        base_config_ref=base_config_ref,
        task_id=task_id,
        input_refs=tuple(dict.fromkeys(input_refs)),
        output_dir=f"{output_ref.rsplit('/', 1)[0]}/outputs",
    )
    target = _safe_relative(root, output_ref, "bridge_config_output")
    publish_canonical_immutable(target, config.to_dict())
    try:
        reread = ResolvedConfigV2.from_mapping(load_canonical_json(target))
    except Exception as error:
        raise _error("BRIDGE_CONFIG_ROUND_TRIP_DRIFT", output_ref) from error
    if reread.config_hash != config.config_hash or reread.full_hash != config.full_hash:
        raise _error("BRIDGE_CONFIG_IDENTITY_DRIFT", output_ref)
    return config, PurePosixPath(output_ref).as_posix()


def _publish_bridge_gate(
    root: Path,
    *,
    gate_id: str,
    producer_task_id: str,
    config: ResolvedConfigV2,
    config_ref: str,
    upstream_refs: Sequence[str],
    measured: Mapping[str, Any],
    output_dir: str,
) -> tuple[str, GateRecord]:
    """Publish an independently derived formal GateRecord task commit."""

    refs = tuple(dict.fromkeys((config_ref, *upstream_refs)))
    for ref in refs:
        _assert_source_file(root, ref, f"bridge.{gate_id}.source_ref")
    gate = GateRecord(
        gate_id=gate_id,
        stage=2,
        status=GateStatus.PASS,
        checked_at=datetime.now(timezone.utc).isoformat(),
        measured=dict(measured),
        threshold={"evaluator": "stage2-s204-independent-bridge-v1"},
        evidence_refs=refs,
        reasons=("independent_evidence_revalidated",),
    )
    store = TaskArtifactStore(root, output_dir)
    published = store.publish(
        task_id=producer_task_id,
        artifact_kind="gate_record",
        config_hash=config.config_hash,
        run_intent="formal",
        payload=gate.to_dict(),
        formal_eligible=True,
        source_refs=refs,
    )
    loaded = load_committed_task_artifact(root, published.commit_ref, require_formal=True)
    if loaded.identity.task_id != producer_task_id or loaded.identity.artifact_kind != "gate_record":
        raise _error("BRIDGE_GATE_IDENTITY_DRIFT", gate_id)
    try:
        reread = GateRecord.from_mapping(dict(loaded.payload))
    except Exception as error:
        raise _error("BRIDGE_GATE_PAYLOAD_INVALID", gate_id) from error
    if reread.gate_id != gate_id or reread.status is not GateStatus.PASS:
        raise _error("BRIDGE_GATE_NOT_PASS", gate_id)
    if set(refs) - set(loaded.source_refs) or set(refs) - set(reread.evidence_refs):
        raise _error("BRIDGE_GATE_LINEAGE_DRIFT", gate_id)
    return published.commit_ref, reread


def _extend_formal_execution(
    root: Path,
    *,
    evidence_ref: str,
    gate: GateRecord,
    asset_hashes: Sequence[str] = (),
    destination: str,
) -> tuple[FormalExecutionEvidence, str]:
    """Append one derived Gate and asset identity to the execution evidence."""

    value = _load_mapping(root, evidence_ref, "formal_execution")
    try:
        previous = FormalExecutionEvidence.from_mapping(value)
        previous.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_INVALID", evidence_ref) from error
    duplicate = gate.gate_id in {item.gate_id for item in previous.prerequisite_gates}
    if duplicate:
        raise _error("FORMAL_EXECUTION_GATE_DUPLICATE", gate.gate_id)
    hashes = tuple(dict.fromkeys((*previous.asset_manifest_hashes, *asset_hashes)))
    try:
        extended = FormalExecutionEvidence(
            run_intent="formal",
            contract_freeze_hash=previous.contract_freeze_hash,
            asset_manifest_hashes=hashes,
            prerequisite_gates=(*previous.prerequisite_gates, gate),
            metadata=dict(previous.metadata),
        )
    except Exception as error:
        raise _error("FORMAL_EXECUTION_EXTENSION_INVALID", gate.gate_id) from error
    target = _safe_relative(root, destination, "formal_execution_extension_output")
    # A rerun of the canonical producer must be safe after a process restart.
    # Existing extension files are accepted only when they are byte-for-byte
    # the extension implied by the same predecessor and GateRecord; an edited
    # file remains a hard identity failure.
    if target.exists():
        try:
            existing = FormalExecutionEvidence.from_mapping(load_canonical_json(target))
            existing.require_for_stage(2)
        except Exception as error:
            raise _error("FORMAL_EXECUTION_EXTENSION_DRIFT", destination) from error
        if existing.artifact_hash != extended.artifact_hash:
            raise _error("FORMAL_EXECUTION_EXTENSION_DRIFT", destination)
        return existing, PurePosixPath(destination).as_posix()
    publish_canonical_immutable(target, extended.to_dict())
    reread = FormalExecutionEvidence.from_mapping(load_canonical_json(target))
    reread.require_for_stage(2)
    if reread.artifact_hash != extended.artifact_hash:
        raise _error("FORMAL_EXECUTION_EXTENSION_DRIFT", destination)
    return reread, PurePosixPath(destination).as_posix()


def _build_phase_environment(
    root: Path,
    *,
    source: Mapping[str, Any],
    formal_execution_ref: str,
    stage0_ref: str,
    stage1_ref: str,
    contract_stage_refs: Mapping[int, str],
    g3_ref: str,
    stage2_asset_ref: str,
    gate_refs: Mapping[str, str],
    phase_gate_ids: Sequence[str],
    capability_refs: Mapping[str, Any],
    g21_ref: str,
    output_ref: str,
    stage1_bridge_ref: str | None = None,
    stage1_bridge_config_ref: str | None = None,
) -> TaskRuntimeEnvironment:
    """Construct one phase snapshot with current Stage0/1/GPU/G3 identities."""

    formal_value = _load_mapping(root, formal_execution_ref, "formal_execution")
    try:
        evidence = FormalExecutionEvidence.from_mapping(formal_value)
        evidence.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_INVALID", formal_execution_ref) from error
    try:
        validate_stage0_handoff(root, stage0_ref, require_ready=True)
        validate_stage1_exit_evidence(root, stage1_ref)
    except Exception as error:
        raise _error("STAGE0_STAGE1_READY_REQUIRED") from error
    stage1_exit = validate_stage1_exit_evidence(root, stage1_ref)
    contract_refs: dict[int, str] = {}
    freezes: dict[int, ContractFreeze] = {}
    for stage in (0, 1, 2):
        ref, freeze = _load_formal_contract_freeze(
            root,
            contract_stage_refs.get(stage),
            stage=stage,
        )
        contract_refs[stage] = ref
        freezes[stage] = freeze
    if freezes[2].artifact_hash != evidence.contract_freeze_hash:
        raise _error("CONTRACT_FREEZE_IDENTITY_MISMATCH", contract_refs[2])
    contract_doc_ref = _publish_contract_document(
        root,
        freezes[2],
        output_ref=PurePosixPath(output_ref).with_name("contract-freeze-stage2.json").as_posix(),
    )
    g3_source = _source_ref(g3_ref, "g3_resolution")
    try:
        g3_loaded = load_committed_task_artifact(root, g3_source, require_formal=True)
        if g3_loaded.identity.task_id != "stage0.04_assets_and_manifests" or g3_loaded.identity.artifact_kind != "asset_resolution":
            raise ValueError("wrong G3 identity")
        FormalG3RuntimeAssets.load(root, g3_source)
    except Exception as error:
        raise _error("G3_RUNTIME_ASSETS_INVALID", g3_source) from error
    asset_ref = _source_ref(stage2_asset_ref, "stage2_asset_resolution")
    try:
        raw_asset = _load_formal_or_direct_payload(
            root,
            asset_ref,
            field="stage2_asset_resolution",
            artifact_kind="asset_resolution",
        )
        if raw_asset.get("schema_version") == "stage2-task-asset-resolution-v1":
            asset = _formal_s23_asset_manifest(raw_asset, field=asset_ref)
        else:
            asset = AssetResolutionManifest.from_mapping(raw_asset)
            validate_formal_asset_identity(asset)
    except Exception as error:
        raise _error("S23_ASSET_RESOLUTION_INVALID", asset_ref) from error
    gpu_ref, allowed_devices = _load_gpu_health_identity(
        root,
        g21_ref,
        expected_stage1_ref=stage1_ref,
    )
    binding_ref = PurePosixPath(output_ref).with_name("gpu-health-binding.json").as_posix()
    binding_payload = _gpu_health_binding_payload(gpu_ref, allowed_devices)
    publish_canonical_immutable(_safe_relative(root, binding_ref, "gpu_binding_output"), binding_payload)
    capability_values = {
        name: _load_capability(root, _mapping(capability_refs, "capability_refs").get(name), name)
        for name in ("server", "cuda", "model_assets", "data_assets")
    }
    cuda_metadata = capability_values["cuda"][1].metadata
    runtime_devices = cuda_metadata.get("allowed_devices")
    if runtime_devices is not None:
        runtime_pairs = tuple(
            (str(item.get("pci_bus_id")), str(item.get("uuid")))
            for item in runtime_devices
            if isinstance(item, Mapping)
        )
        if runtime_pairs != allowed_devices:
            raise _error("GPU_RUNTIME_INVENTORY_DRIFT")
    runtime_excluded = cuda_metadata.get("excluded_device")
    if runtime_excluded is not None and (
        not isinstance(runtime_excluded, Mapping)
        or str(runtime_excluded.get("pci_bus_id")) != EXCLUDED_PCI
        or str(runtime_excluded.get("uuid")) != EXCLUDED_UUID
    ):
        raise _error("GPU_RUNTIME_EXCLUDED_IDENTITY_DRIFT")
    if set(gate_refs) != set(phase_gate_ids):
        raise _error("PHASE_GATE_REF_SET_INVALID", output_ref)
    expected_gate_tasks = {
        "stage1.G1-EXIT": STAGE1_TASK_ID,
        "stage2.G2.0": "stage2.01_scope_hypotheses_and_preregistration",
        "stage2.G2.1": "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.G2.2": "stage2.03_assets_checkpoints_and_sampling",
    }
    selected_gates = _load_formal_gate_refs(
        root,
        {str(key): str(value) for key, value in gate_refs.items()},
        required=tuple(phase_gate_ids),
        expected_task=expected_gate_tasks,
        # G1-EXIT is a Stage 1 gate summary, not a Stage 2 TaskArtifact gate
        # commit.  Stage 2 adapter gates are the only records with the formal
        # TaskArtifact ``gate_record`` kind here.
        expected_artifact_kind={
            key: "gate_record"
            for key in expected_gate_tasks
            if key.startswith("stage2.")
        },
    )
    evidence_refs: dict[str, str] = {
        "formal_execution": formal_execution_ref,
        "stage0_handoff": stage0_ref,
        "stage1_g1_exit": stage1_ref,
        "contract_stage_0": contract_refs[0],
        "contract_stage_1": contract_refs[1],
        "contract_stage_2": contract_refs[2],
        "contract_freeze": contract_doc_ref,
        "g3_resolution": g3_source,
        "stage2_asset_resolution": asset_ref,
        "gpu_health": gpu_ref,
        "gpu_health_binding": binding_ref,
    }
    if stage1_bridge_ref is not None:
        bridge_ref = _source_ref(stage1_bridge_ref, "stage1_bridge")
        if stage1_bridge_config_ref is None:
            raise _error("STAGE1_BRIDGE_CONFIG_REQUIRED")
        _validate_stage1_bridge(
            root,
            index_ref=stage1_ref,
            bridge_ref=bridge_ref,
            config_ref=stage1_bridge_config_ref,
        )
        evidence_refs["stage1_11_bridge"] = bridge_ref
    if stage1_bridge_config_ref is not None:
        config_ref = _source_ref(stage1_bridge_config_ref, "stage1_bridge_config")
        _assert_source_file(root, config_ref, "stage1_bridge_config")
        try:
            ResolvedConfigV2.from_mapping(_load_mapping(root, config_ref, "stage1_bridge_config"))
        except Exception as error:
            raise _error("STAGE1_BRIDGE_CONFIG_INVALID") from error
        evidence_refs["stage1_11_bridge_config"] = config_ref
    evidence_refs.update({f"gate_{key.replace('.', '_').replace('-', '_').lower()}": value for key, value in selected_gates.items()})
    evidence_refs.update({f"capability_{key}": value[0] for key, value in capability_values.items()})
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(capability_values),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset(phase_gate_ids),
        evidence_refs=evidence_refs,
    )
    target = _safe_relative(root, output_ref, "phase_environment_output")
    publish_canonical_immutable(target, environment.to_dict())
    reread = TaskRuntimeEnvironment.from_mapping(load_canonical_json(target))
    if reread.environment_hash != environment.environment_hash:
        raise _error("ENVIRONMENT_ROUND_TRIP_DRIFT", output_ref)
    return reread


def _validate_s22_g3_resolution(root: Path, g3_ref: str) -> LoadedTaskArtifact:
    """Validate the one Stage 0 G3 resolution consumed by formal S2.2."""

    normalized = _source_ref(g3_ref, "g3_resolution")
    try:
        loaded = load_committed_task_artifact(root, normalized, require_formal=True)
        if (
            loaded.identity.task_id != "stage0.04_assets_and_manifests"
            or loaded.identity.artifact_kind != "asset_resolution"
            or loaded.identity.formal_eligible is not True
        ):
            raise ValueError("S22_G3_RESOLUTION_IDENTITY_INVALID")
        FormalG3RuntimeAssets.load(root, normalized)
    except Exception as error:
        raise _error("S22_G3_RESOLUTION_INVALID", normalized) from error
    return loaded


def _s22_g3_ready_manifest_hashes(root: Path, g3_ref: str) -> tuple[str, ...]:
    """Load G3 once and bind every qualified manifest digest to S2.2 evidence.

    The S2.2 fixed-state provider resolves only three assets, but its formal
    execution evidence must carry the complete G3 resolution lineage.  The
    resolution loader has already validated the entry shape and each digest;
    this additional check keeps the S2.2 boundary explicit and rejects a
    truncated or duplicate entry list before any evidence extension is
    published.
    """

    normalized = _source_ref(g3_ref, "g3_resolution")
    try:
        runtime_assets = FormalG3RuntimeAssets.load(root, normalized)
        raw_entries = runtime_assets.resolution.get("entries")
        if (
            not isinstance(raw_entries, list)
            or len(raw_entries) != S22_G3_READY_MANIFEST_COUNT
        ):
            raise ValueError("G3 ready manifest entry count must be exactly 13")
        hashes: list[str] = []
        for index, raw_entry in enumerate(raw_entries):
            if not isinstance(raw_entry, Mapping):
                raise ValueError(f"G3 resolution entry {index} is not an object")
            digest = raw_entry.get("ready_manifest_sha256")
            if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise ValueError(f"G3 resolution entry {index} has invalid manifest hash")
            hashes.append(digest)
        if len(set(hashes)) != S22_G3_READY_MANIFEST_COUNT:
            raise ValueError("G3 ready manifest hashes must be unique")
        return tuple(hashes)
    except Exception as error:
        raise _error("S22_G3_MANIFEST_HASHES_INVALID", normalized) from error


def _build_s22_formal_environment(
    root: Path,
    *,
    formal_execution_ref: str,
    stage0_ref: str,
    stage1_ref: str,
    contract_stage_refs: Mapping[int, str],
    g3_ref: str,
    gate_refs: Mapping[str, str],
    g1_ref: str,
    g21_handoff_ref: str,
    output_ref: str,
) -> TaskRuntimeEnvironment:
    """Narrow S2.2 audit environment with the independent Stage 0 G3 input.

    G3 is the real upstream provider asset authority for the S2.2 fixed-state
    audit.  It is intentionally kept separate from S2.3/G2.2 assets and
    runtime capability evidence.
    """

    evidence = FormalExecutionEvidence.from_mapping(
        _load_mapping(root, _source_ref(formal_execution_ref, "formal_execution"), "formal_execution")
    )
    evidence.require_for_stage(2)
    stage0 = _source_ref(stage0_ref, "stage0_handoff")
    stage1 = _source_ref(stage1_ref, "stage1_g1_exit")
    validate_stage0_handoff(root, stage0, require_ready=True)
    validate_stage1_exit_evidence(root, stage1)
    contracts: dict[int, str] = {}
    freezes: dict[int, ContractFreeze] = {}
    for stage in (0, 1, 2):
        contracts[stage], freezes[stage] = _load_formal_contract_freeze(
            root, contract_stage_refs.get(stage), stage=stage
        )
    if freezes[2].artifact_hash != evidence.contract_freeze_hash:
        raise _error("CONTRACT_FREEZE_IDENTITY_MISMATCH", contracts[2])
    _validate_s22_g3_resolution(root, g3_ref)
    external_g21 = _source_ref(g21_handoff_ref, "g21_handoff")
    gpu_ref, allowed_devices = _load_gpu_health_identity(root, external_g21, expected_stage1_ref=stage1)
    binding_ref = PurePosixPath(output_ref).with_name("gpu-health-binding.json").as_posix()
    binding = _gpu_health_binding_payload(gpu_ref, allowed_devices)
    publish_canonical_immutable(_safe_relative(root, binding_ref, "gpu_binding_output"), binding)
    selected = _load_formal_gate_refs(
        root,
        {"stage1.G1-EXIT": _source_ref(g1_ref, "gate.stage1.G1-EXIT"), **{str(k): str(v) for k, v in gate_refs.items()}},
        required=("stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"),
        expected_task={
            "stage1.G1-EXIT": STAGE1_TASK_ID,
            "stage2.G2.0": "stage2.01_scope_hypotheses_and_preregistration",
            "stage2.G2.1": "stage2.02_stage1_handoff_and_fixed_state_contract",
        },
        expected_artifact_kind={"stage2.G2.0": "gate_record", "stage2.G2.1": "gate_record"},
    )
    evidence_refs = {
        "formal_execution": _source_ref(formal_execution_ref, "formal_execution"),
        "stage0_handoff": stage0,
        "stage1_g1_exit": stage1,
        "contract_stage_0": contracts[0],
        "contract_stage_1": contracts[1],
        "contract_stage_2": contracts[2],
        "contract_freeze": _publish_contract_document(
            root, freezes[2], output_ref=PurePosixPath(output_ref).with_name("contract-freeze-stage2.json").as_posix()
        ),
        "g3_resolution": _source_ref(g3_ref, "g3_resolution"),
        "gpu_health": gpu_ref,
        "gpu_health_binding": binding_ref,
    }
    evidence_refs.update({f"gate_{key.replace('.', '_').replace('-', '_').lower()}": ref for key, ref in selected.items()})
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}),
        evidence_refs=evidence_refs,
    )
    target = _safe_relative(root, output_ref, "s22_environment_output")
    publish_canonical_immutable(target, environment.to_dict())
    return TaskRuntimeEnvironment.from_mapping(load_canonical_json(target))


def _formal_dag_config(
    root: Path,
    *,
    base_config_ref: str,
    task_id: str,
    input_refs: Sequence[str],
    output_dir: str,
) -> ResolvedConfigV2:
    """Resolve a formal S2 predecessor config through the CLI-compatible path."""

    base_path = _config_path(root, base_config_ref)
    try:
        from param_importance_nlp.cli import _load_mapping as cli_load_mapping

        defaults_path = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
        base = ResolvedConfig.resolve(cli_load_mapping(defaults_path), cli_load_mapping(base_path))
    except Exception as error:
        raise _error("BASE_CONFIG_INVALID", base_config_ref) from error
    if base.section("identity").get("run_intent") != "formal":
        raise _error("BASE_CONFIG_FORMAL_REQUIRED")
    value = base.to_dict()
    identity = value["identity"]
    identity.update({
        "task": task_id,
        "stage": 2,
        "run_intent": "formal",
        "formal_eligible": True,
    })
    task = DEFAULT_TASK_CATALOG.get(task_id)
    return ResolvedConfigV2.resolve(
        value,
        task_id=task_id,
        overrides={
            "providers": {
                "kind": "offline_hf",
                "task_type": "causal_lm",
                "task_name": "pile",
                "num_labels": None,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            },
            "orchestration": {"input_result_refs": list(input_refs)},
            "execution": {"runner_kind": task.runner_kind.value, "dry_run": False, "fail_on_blocked": True},
            "artifacts": {
                "output_dir": output_dir,
                "required_kinds": list(task.artifact_kinds),
                "publish_partial": False,
            },
            "recovery": {
                "mode": task.recovery_mode.value,
                "resume_ref": None,
                "safe_boundary": task.safe_boundary.value,
                "max_restarts": 0,
            },
        },
    )


def _load_task_result_set(
    root: Path,
    refs: Mapping[str, str],
    *,
    task_id: str,
) -> dict[str, LoadedTaskArtifact]:
    expected = TASK_INPUTS[task_id]
    if set(refs) != set(expected):
        raise _error("FORMAL_DAG_OUTPUT_SET_INVALID", task_id)
    loaded: dict[str, LoadedTaskArtifact] = {}
    for kind in expected:
        loaded[kind] = _load_formal_source(
            root,
            _source_ref(refs[kind], f"formal_dag.{task_id}.{kind}"),
            task_id=task_id,
            artifact_kind=kind,
        )
    configs = {item.identity.config_hash for item in loaded.values()}
    if len(configs) != 1:
        raise _error("FORMAL_DAG_CONFIG_IDENTITY_MISMATCH", task_id)
    return loaded


def _publish_resolved_config(
    root: Path,
    config: ResolvedConfigV2,
    *,
    output_ref: str,
) -> str:
    """Persist one exact runtime config before an independent evaluator reads it."""

    target_ref = PurePosixPath(output_ref).as_posix()
    target = _safe_relative(root, target_ref, "resolved_config_output")
    publish_canonical_immutable(target, config.to_dict())
    try:
        reread = ResolvedConfigV2.from_mapping(load_canonical_json(target))
    except Exception as error:
        raise _error("RESOLVED_CONFIG_ROUND_TRIP_DRIFT", target_ref) from error
    if reread.task_id != config.task_id or reread.config_hash != config.config_hash or reread.full_hash != config.full_hash:
        raise _error("RESOLVED_CONFIG_IDENTITY_DRIFT", target_ref)
    return target_ref


def _validate_formal_s22_task_group(
    root: Path,
    refs: Mapping[str, str],
    *,
    config_ref: str,
    environment_ref: str,
    required_lineage: Sequence[str],
    expected_input_refs: Sequence[str] | None = None,
    g21_ref: str | None = None,
) -> dict[str, LoadedTaskArtifact]:
    """Validate the producer-owned S2.2 commit set and its phase snapshot.

    This is deliberately a postcondition of the producer, not a convenience
    loader for caller-supplied commits.  In particular every S2.2 commit must
    carry the exact environment evidence refs (including G2.1) and the direct
    S2.1 lineage.  That makes an old/partial directory fail closed even when
    the task output directory happens to contain all three filenames.
    """

    task_id = "stage2.02_stage1_handoff_and_fixed_state_contract"
    # The resolved config is producer output under the evidence namespace;
    # unlike upstream source refs it is not subject to the tracked ``configs/``
    # path ban (and is checked by its own ResolvedConfigV2 hash).
    normalized_config_ref = _logical(config_ref, "s22.config_ref")
    try:
        config = ResolvedConfigV2.from_mapping(
            _load_mapping(root, normalized_config_ref, "s22.resolved_config")
        )
    except Exception as error:
        raise _error("S22_RESOLVED_CONFIG_INVALID") from error
    artifacts = config.section("artifacts")
    orchestration = config.section("orchestration")
    if (
        config.task_id != task_id
        or config.run_intent != "formal"
        or config.formal_eligible is not True
        or artifacts.get("output_dir") != S204_S22_COMMIT_OUTPUT_DIR
        or tuple(artifacts.get("required_kinds", ())) != TASK_INPUTS[task_id]
    ):
        raise _error("S22_RESOLVED_CONFIG_SCOPE_INVALID")
    if expected_input_refs is not None and tuple(orchestration.get("input_result_refs", ())) != tuple(expected_input_refs):
        raise _error("S22_RESOLVED_CONFIG_INPUT_BINDING_INVALID")
    try:
        environment = TaskRuntimeEnvironment.from_mapping(
            _load_mapping(root, _source_ref(environment_ref, "s22.environment_ref"), "s22.environment")
        )
    except Exception as error:
        raise _error("S22_ENVIRONMENT_INVALID") from error
    required_gates = {"stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"}
    if set(environment.passed_gate_ids) != required_gates:
        raise _error("S22_ENVIRONMENT_GATE_SET_INVALID")
    if set(environment.capabilities):
        raise _error("S22_ENVIRONMENT_CAPABILITIES_FORBIDDEN")
    g3_entries = [
        (key, str(ref))
        for key, ref in environment.evidence_refs.items()
        if key == "g3_resolution"
    ]
    if len(g3_entries) != 1:
        raise _error("S22_G3_RESOLUTION_REQUIRED")
    _validate_s22_g3_resolution(root, g3_entries[0][1])
    forbidden_tokens = (
        "g2.2",
        "stage2.03",
        "stage2-03",
        "s2.3",
        "capabilit",
        "model_assets",
        "data_assets",
        "stage0.04",
    )
    if any(
        key != "g3_resolution"
        and not (
            key == "formal_execution"
            and PurePosixPath(str(ref)).as_posix() == S22_G3_FORMAL_EXECUTION_G21_REF
        )
        and (
            "g3" in str(ref).casefold()
            or any(token in f"{key}={ref}".casefold() for token in forbidden_tokens)
        )
        for key, ref in environment.evidence_refs.items()
    ):
        raise _error("S22_ENVIRONMENT_DOWNSTREAM_LINEAGE_FORBIDDEN")
    environment_refs = tuple(dict.fromkeys(str(ref) for ref in environment.evidence_refs.values()))
    required_refs = set(str(ref) for ref in required_lineage) | set(environment_refs)
    canonical_prefix = f"{S204_S22_CANONICAL_OUTPUT_DIR}/task-outputs/stage2-02/commits"
    expected_refs = {
        kind: f"{canonical_prefix}/{kind}.json"
        for kind in TASK_INPUTS[task_id]
    }
    if dict(refs) != expected_refs:
        raise _error("S22_FORMAL_COMMIT_NAMESPACE_INVALID")
    try:
        loaded = _load_task_result_set(root, refs, task_id=task_id)
    except S204MaterializationError:
        raise
    except Exception as error:
        raise _error("S22_FORMAL_COMMITS_REQUIRED") from error
    for kind, item in loaded.items():
        if item.identity.config_hash != config.config_hash:
            raise _error("S22_CONFIG_IDENTITY_MISMATCH", kind)
        if not required_refs.issubset(item.source_refs):
            raise _error("S22_LINEAGE_BINDING_MISMATCH", kind)
        if kind == "handoff_manifest":
            if item.payload.get("scope") != "formal" or item.payload.get("status") != "FORMAL_CANDIDATE" or item.payload.get("formal_eligible") is not False:
                raise _error("S22_CANDIDATE_PAYLOAD_INVALID", kind)
        elif kind == "fixed_state_contract":
            optional_scope = item.payload.get("scope")
            if (
                item.payload.get("schema_version") != _PAYLOAD_SCHEMAS[kind]
                or (optional_scope is not None and optional_scope != "formal")
                or item.payload.get("status") != "FORMAL_CANDIDATE"
                or item.payload.get("formal_eligible") is not False
                or item.payload.get("contract_version") != "stage2-fixed-state-contract-v1"
                or not isinstance(item.payload.get("fixed_state_id"), str)
                or not item.payload["fixed_state_id"]
            ):
                raise _error("S22_CANDIDATE_PAYLOAD_INVALID", kind)
        elif item.payload.get("schema_version") != _PAYLOAD_SCHEMAS[kind] or item.payload.get("formal_eligible") is not False:
            raise _error("S22_GATE_CANDIDATE_INVALID")
    if g21_ref is not None:
        try:
            g21 = load_committed_task_artifact(
                root, _source_ref(g21_ref, "s22.g21_ref"), require_formal=True
            )
        except Exception as error:
            raise _error("S22_G21_GATE_INVALID") from error
        if (
            g21.identity.task_id
            != "stage2.02_stage1_handoff_and_fixed_state_contract"
            or g21.identity.artifact_kind != "gate_record"
            or g21.identity.config_hash != config.config_hash
        ):
            raise _error("S22_CONFIG_GATE_HASH_MISMATCH")
    return loaded


def produce_formal_s22_task_outputs(
    data_root: str | Path,
    *,
    source: Mapping[str, Any],
    s21_refs: Mapping[str, str],
    g20_ref: str,
    g20_gate: GateRecord,
    g21_ref: str,
    g21_gate: GateRecord,
    g21_resolved_config_ref: str,
    formal_execution_ref: str,
    base_config_ref: str,
    stage0_ref: str,
    stage1_ref: str,
    contract_refs: Mapping[int, str],
    g3_ref: str,
    stage1_10_refs: Mapping[str, str],
    stage1_11_refs: Mapping[str, str],
    g1_ref: str,
    output_dir: str,
) -> tuple[dict[str, str], FormalExecutionEvidence, str, str, str]:
    """Produce the formal S2.2 group through the existing TaskRuntime handler.

    The function owns the canonical output location and all inputs needed to
    build the environment.  It accepts no S2.2 artifact refs, so an absent
    server directory is produced by the reviewed runner rather than promoted
    from a caller's JSON.  The returned tuple is ``refs, evidence,
    evidence_ref, config_ref, environment_ref``.
    """

    root = Path(data_root).resolve()
    if PurePosixPath(output_dir).as_posix() != S204_S22_CONTROL_OUTPUT_DIR:
        raise _error("S22_OUTPUT_DIR_NOT_CANONICAL", output_dir)
    task_id = "stage2.02_stage1_handoff_and_fixed_state_contract"
    _load_task_result_set(root, s21_refs, task_id="stage2.01_scope_hypotheses_and_preregistration")
    s21_lineage = tuple(s21_refs[kind] for kind in TASK_INPUTS["stage2.01_scope_hypotheses_and_preregistration"])
    # Re-read the external adapter commits and the hardware report before any
    # S2.2 output can be published.  G2.1 is bound to Stage 1/S2.1 upstream;
    # the S2.2 group is downstream and is never used as its source.
    for gate_id, ref, gate in (("stage2.G2.0", g20_ref, g20_gate), ("stage2.G2.1", g21_ref, g21_gate)):
        if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
            raise _error("S22_ADAPTER_GATE_INVALID", gate_id)
        _source_ref(ref, f"s22.{gate_id}.ref")
    external_g21 = _source_ref(source.get("g21_handoff"), "g21_handoff")
    _load_gpu_health_identity(root, external_g21, expected_stage1_ref=stage1_ref)
    g3_manifest_hashes = _s22_g3_ready_manifest_hashes(root, g3_ref)
    evidence, phase_evidence_ref = _extend_formal_execution(
        root,
        evidence_ref=formal_execution_ref,
        gate=g20_gate,
        asset_hashes=g3_manifest_hashes,
        destination=S22_G3_FORMAL_EXECUTION_G20_REF,
    )
    evidence, phase_evidence_ref = _extend_formal_execution(
        root,
        evidence_ref=phase_evidence_ref,
        gate=g21_gate,
        destination=S22_G3_FORMAL_EXECUTION_G21_REF,
    )
    del base_config_ref
    config_ref = _logical(g21_resolved_config_ref, "g21_resolved_config")
    if config_ref != S204_S22_CONFIG_REF:
        raise _error("S22_RESOLVED_CONFIG_REF_NOT_CANONICAL")
    try:
        config = ResolvedConfigV2.from_mapping(_load_mapping(root, config_ref, "g21_resolved_config"))
    except Exception as error:
        raise _error("S22_RESOLVED_CONFIG_INVALID") from error
    orchestration = config.section("orchestration")
    artifacts = config.section("artifacts")
    if (
        config.task_id != task_id
        or config.run_intent != "formal"
        or config.formal_eligible is not True
        or artifacts.get("output_dir") != S204_S22_COMMIT_OUTPUT_DIR
        or tuple(artifacts.get("required_kinds", ())) != TASK_INPUTS[task_id]
    ):
        raise _error("S22_RESOLVED_CONFIG_SCOPE_INVALID")
    try:
        g21_loaded = load_committed_task_artifact(root, _source_ref(g21_ref, "s22.g21_ref"), require_formal=True)
    except Exception as error:
        raise _error("S22_G21_GATE_INVALID") from error
    if g21_loaded.identity.config_hash != config.config_hash:
        raise _error("S22_CONFIG_GATE_HASH_MISMATCH")
    expected_config_inputs = tuple(
        (*s21_lineage, *stage1_10_refs.values(), *stage1_11_refs.values())
    )
    if tuple(orchestration.get("input_result_refs", ())) != expected_config_inputs:
        raise _error("S22_RESOLVED_CONFIG_INPUT_BINDING_INVALID")
    environment_ref = S22_G3_FORMAL_ENVIRONMENT_REF
    environment = _build_s22_formal_environment(
        root,
        formal_execution_ref=phase_evidence_ref,
        stage0_ref=stage0_ref,
        stage1_ref=stage1_ref,
        contract_stage_refs=contract_refs,
        g3_ref=g3_ref,
        gate_refs={"stage2.G2.0": g20_ref, "stage2.G2.1": g21_ref},
        g1_ref=g1_ref,
        g21_handoff_ref=external_g21,
        output_ref=environment_ref,
    )
    runtime = TaskRuntime(workspace_root=root)
    register_stage23_runners(runtime, root)
    result = runtime.execute(config, environment=environment)
    if result.status is not TaskRunStatus.PASS or not result.formal_eligible:
        raise _error("S22_TASK_NOT_PASS", result.status.value)
    refs = dict(result.artifact_refs)
    _validate_formal_s22_task_group(
        root,
        refs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        required_lineage=(*s21_lineage, g20_ref, g21_ref),
        expected_input_refs=(*s21_lineage, *stage1_10_refs.values(), *stage1_11_refs.values()),
        g21_ref=g21_ref,
    )
    return refs, evidence, phase_evidence_ref, config_ref, environment_ref


def ensure_formal_s22_task_outputs(
    data_root: str | Path,
    *,
    predecessor_refs: dict[str, dict[str, str]],
    output_dir: str,
    producer_kwargs: Mapping[str, Any],
) -> tuple[dict[str, str], FormalExecutionEvidence, str, str, str]:
    """Ensure one complete, producer-owned S2.2 output set exists.

    ``predecessor_refs`` may omit S2.2 (the normal absent→produce case), but
    a partial set is never repaired.  Existing refs are accepted only from the
    canonical output namespace and after the same lineage/config/environment
    postcondition used by the producer.
    """

    root = Path(data_root).resolve()
    task_id = "stage2.02_stage1_handoff_and_fixed_state_contract"
    raw = predecessor_refs.get(task_id)
    canonical_dir = PurePosixPath(output_dir).as_posix()
    if canonical_dir != S204_S22_CONTROL_OUTPUT_DIR:
        raise _error("S22_OUTPUT_DIR_NOT_CANONICAL", canonical_dir)
    config_ref = S204_S22_CONFIG_REF
    environment_ref = S22_G3_FORMAL_ENVIRONMENT_REF
    evidence_ref = S22_G3_FORMAL_EXECUTION_G21_REF
    if raw is None:
        refs, evidence, evidence_ref, config_ref, environment_ref = produce_formal_s22_task_outputs(
            root,
            output_dir=canonical_dir,
            **dict(producer_kwargs),
        )
        predecessor_refs[task_id] = dict(refs)
        return refs, evidence, evidence_ref, config_ref, environment_ref
    if set(raw) != set(TASK_INPUTS[task_id]):
        raise _error("S204_PREDECESSOR_ARTIFACT_SET_INVALID", task_id)
    normalized = {
        kind: _source_ref(raw[kind], f"predecessor.{task_id}.{kind}")
        for kind in TASK_INPUTS[task_id]
    }
    expected_refs = {
        kind: f"{S204_S22_COMMIT_OUTPUT_DIR}/commits/{kind}.json"
        for kind in TASK_INPUTS[task_id]
    }
    if normalized != expected_refs:
        raise _error("S22_FORMAL_COMMIT_NAMESPACE_INVALID")
    _validate_formal_s22_task_group(
        root,
        normalized,
        config_ref=config_ref,
        environment_ref=environment_ref,
        required_lineage=tuple(producer_kwargs["s21_refs"].values())
        + (str(producer_kwargs["g20_ref"]), str(producer_kwargs["g21_ref"])),
        expected_input_refs=(
            tuple(producer_kwargs["expected_input_refs"])
            if "expected_input_refs" in producer_kwargs
            else (
                (
                    *tuple(producer_kwargs["s21_refs"].values()),
                    *tuple(producer_kwargs.get("stage1_10_refs", {}).values()),
                    *tuple(producer_kwargs.get("stage1_11_refs", {}).values()),
                )
                if producer_kwargs.get("stage1_10_refs")
                or producer_kwargs.get("stage1_11_refs")
                else None
            )
        ),
        g21_ref=str(producer_kwargs["g21_ref"]),
    )
    evidence = FormalExecutionEvidence.from_mapping(
        _load_mapping(root, evidence_ref, "s22.formal_execution")
    )
    predecessor_refs[task_id] = normalized
    return normalized, evidence, evidence_ref, config_ref, environment_ref


def _derive_g20_gate(
    root: Path,
    *,
    s21_refs: Mapping[str, str],
    stage1_refs: Mapping[str, str],
    stage1_exit: Any,
    formal_execution_ref: str,
    base_config_ref: str,
    s21_config_ref: str,
    output_dir: str,
) -> tuple[str, GateRecord, str | None]:
    """Consume the independent G2.0 evaluator's formal GateRecord commit.

    The evaluator owns all preregistration/hypothesis semantics.  This module
    only supplies the verified S2.1/Stage1 lineage and performs the final
    content-addressed loader checks; it never upgrades the runner candidate.
    """

    # Keep this import lazy: the evaluator is an independently reviewed Stage
    # 2.1 component and is merged alongside this materializer.  Do not fall
    # back to a local formula or to a runner candidate when that component is
    # absent.
    persisted_config_ref = _source_ref(s21_config_ref, "g20.s21_config_ref")
    try:
        persisted_config = ResolvedConfigV2.from_mapping(
            _load_mapping(root, persisted_config_ref, "g20.s21_config")
        )
    except Exception as error:
        raise _error("G20_RESOLVED_CONFIG_INVALID") from error
    if persisted_config.task_id != "stage2.01_scope_hypotheses_and_preregistration" or persisted_config.run_intent != "formal" or persisted_config.formal_eligible is not True:
        raise _error("G20_RESOLVED_CONFIG_SCOPE_INVALID")
    if any(item.identity.config_hash != persisted_config.config_hash for item in _load_task_result_set(
        root,
        s21_refs,
        task_id="stage2.01_scope_hypotheses_and_preregistration",
    ).values()):
        raise _error("G20_RESOLVED_CONFIG_IDENTITY_MISMATCH")

    try:
        from param_importance_nlp.experiments.stage2_g20_evaluator import (
            evaluate_formal_g20,
        )
    except ImportError as error:
        raise _error("G20_INDEPENDENT_EVALUATOR_REQUIRED") from error
    try:
        # This is the intentionally narrow API.  The evaluator re-reads the
        # three formal S2.1 commits and owns the frozen preregistration /
        # hypothesis validators; no caller status, metrics, or gate envelope
        # is accepted here.
        # The first reviewed evaluator release exposed ``output_root`` and
        # derived repository identity from its own source.  The hardened
        # release additionally requires an explicit repository root and the
        # persisted S2.1 ResolvedConfigV2.  Select only these known keyword
        # names from the callable signature; never pass caller status/metrics
        # or fall back to a runner candidate.
        parameters = inspect.signature(evaluate_formal_g20).parameters
        evaluator_kwargs: dict[str, Any] = {}
        if "output_root" in parameters:
            evaluator_kwargs["output_root"] = f"{output_dir}/bridges/stage2-g2-0"
        elif "output_dir" in parameters:
            evaluator_kwargs["output_dir"] = f"{output_dir}/bridges/stage2-g2-0"
        else:
            raise _error("G20_EVALUATOR_OUTPUT_BOUNDARY_REQUIRED")
        if "repository_root" in parameters:
            evaluator_kwargs["repository_root"] = ROOT
        if "resolved_config_ref" in parameters:
            evaluator_kwargs["resolved_config_ref"] = persisted_config_ref
        elif "evaluation_config_ref" in parameters:
            evaluator_kwargs["evaluation_config_ref"] = persisted_config_ref
        result = evaluate_formal_g20(root, dict(s21_refs), **evaluator_kwargs)
    except Exception as error:
        raise _error("G20_INDEPENDENT_EVALUATOR_FAILED", type(error).__name__) from error
    if not isinstance(result, Mapping):
        raise _error("G20_EVALUATOR_RESULT_INVALID")
    if result.get("status") != GateStatus.PASS.value or result.get("formal_eligible") is not True:
        raise _error("G20_EVALUATOR_NOT_PASS")
    gate_ref = _source_ref(result.get("commit_ref"), "g20.evaluator.commit_ref")
    loaded = load_committed_task_artifact(root, gate_ref, require_formal=True)
    if loaded.identity.task_id != "stage2.01_scope_hypotheses_and_preregistration" or loaded.identity.artifact_kind != "gate_record":
        raise _error("G20_GATE_IDENTITY_MISMATCH")
    try:
        gate = GateRecord.from_mapping(dict(loaded.payload))
    except Exception as error:
        raise _error("G20_GATE_PAYLOAD_INVALID") from error
    if gate.gate_id != "stage2.G2.0" or gate.status is not GateStatus.PASS:
        raise _error("G20_GATE_NOT_PASS")
    # The evaluator's committed envelope must be exactly bound to the S2.1
    # result set.  Stage0/Stage1/current evidence remain bound by the
    # phase environment and FormalExecutionEvidence; they are not fabricated
    # into the evaluator's own source_refs after the fact.
    expected_upstream = tuple(
        s21_refs[kind]
        for kind in TASK_INPUTS["stage2.01_scope_hypotheses_and_preregistration"]
    )
    if tuple(loaded.source_refs) != expected_upstream or tuple(gate.evidence_refs) != expected_upstream:
        raise _error("G20_GATE_LINEAGE_MISMATCH")
    if isinstance(result.get("envelope_artifact_hash"), str) and result["envelope_artifact_hash"] != loaded.identity.artifact_hash:
        raise _error("G20_ENVELOPE_HASH_MISMATCH")
    if not isinstance(gate.measured, Mapping) or gate.measured.get("config_hash") != loaded.identity.config_hash:
        raise _error("G20_GATE_CONFIG_IDENTITY_MISMATCH")
    # The S2.1 runner's ResolvedConfigV2 is already the config identity carried
    # by all three commits.  No synthetic bridge config is attached to this
    # independent evaluator result; G2.1/G2.2 adapters create their own
    # dedicated bridge configs when adapting custom authority reports.
    return gate_ref, gate, None


def _derive_g21_gate(
    root: Path,
    *,
    s22_refs: Mapping[str, str],
    stage1_refs: Mapping[str, str],
    stage1_ref: str,
    stage1_bridge_ref: str,
    stage1_config_ref: str,
    stage1_exit: Any,
    g21_ref: str,
    formal_execution_ref: str,
    base_config_ref: str,
    output_dir: str,
) -> tuple[str, GateRecord, str]:
    loaded = _load_task_result_set(
        root,
        s22_refs,
        task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
    )
    handoff = loaded["handoff_manifest"].payload
    fixed = loaded["fixed_state_contract"].payload
    for name, payload in (("handoff", handoff), ("fixed", fixed)):
        if payload.get("status") != "FORMAL_CANDIDATE" or payload.get("formal_eligible") is not False:
            raise _error("G21_RUNNER_CANDIDATE_INVALID", name)
    handoff_stage1 = handoff.get("stage1_g1_exit")
    if not isinstance(handoff_stage1, Mapping) or handoff_stage1.get("index_ref") != stage1_ref:
        raise _error("G21_S22_STAGE1_BINDING_INVALID")
    authoritative = _load_authoritative_g21(root, g21_ref, expected_stage1_ref=stage1_ref)
    source_refs = tuple(
        dict.fromkeys(
            (
                formal_execution_ref,
                g21_ref,
                stage1_ref,
                stage1_bridge_ref,
                stage1_config_ref,
                *_stage1_role_refs(root, stage1_ref),
                *stage1_refs.values(),
                *s22_refs.values(),
            )
        )
    )
    config, config_ref = _publish_bridge_config(
        root,
        base_config_ref=base_config_ref,
        task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
        input_refs=source_refs,
        output_ref=f"{output_dir}/bridges/g21-resolved-config-v2.json",
    )
    stage1_payload_hashes: dict[str, str] = {}
    for kind, ref in stage1_refs.items():
        stage1_payload = load_committed_task_artifact(root, ref, require_formal=True).payload
        stage1_payload_hashes[kind] = canonical_json_hash(dict(stage1_payload))
    measured = {
        "bridge_config_hash": config.config_hash,
        "bridge_config_full_hash": config.full_hash,
        "stage1_bridge_evidence_ref": stage1_bridge_ref,
        "stage1_bridge_config_ref": stage1_config_ref,
        "execution_commit": stage1_exit.execution_commit,
        "stage1_index_artifact_hash": stage1_exit.index_artifact_hash,
        "stage1_role_sha256": dict(stage1_exit.role_sha256),
        "stage1_task_payload_hashes": stage1_payload_hashes,
        "g21_authoritative_payload_hash": canonical_json_hash(dict(authoritative)),
        "s22_payload_hashes": {kind: canonical_json_hash(dict(item.payload)) for kind, item in loaded.items()},
        "s22_artifact_hashes": {kind: item.identity.artifact_hash for kind, item in loaded.items()},
        "fixed_state_digest": fixed.get("provider_state_digest"),
        "registry_hash": fixed.get("registry_hash"),
    }
    gate_ref, gate = _publish_bridge_gate(
        root,
        gate_id="stage2.G2.1",
        producer_task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
        config=config,
        config_ref=config_ref,
        upstream_refs=source_refs,
        measured=measured,
        output_dir=f"{output_dir}/bridges/stage2-g2-1",
    )
    return gate_ref, gate, config_ref


def _load_authoritative_g22(
    root: Path,
    ref: str,
    *,
    manifest: AssetResolutionManifest,
    expected_sha256: str | None = None,
) -> Mapping[str, Any]:
    value = _load_mapping(root, ref, "g22_asset_evidence")
    _assert_source_file(root, ref, "g22_asset_evidence", sha256=expected_sha256)
    status = value.get("status", value.get("gate_status"))
    if status != "PASS" or "BLOCKED" in str(value.get("gate_status", "")):
        raise _error("G22_AUTHORITATIVE_EVIDENCE_NOT_PASS")
    schema = value.get("schema_version")
    schema_text = "" if not isinstance(schema, str) else schema.casefold()
    gate_text = str(value.get("gate_id", "")).casefold()
    if not ("g2.2" in schema_text or "g22" in schema_text or "g2.2" in gate_text or "g22" in gate_text):
        raise _error("G22_AUTHORITATIVE_SCHEMA_INVALID")
    if isinstance(value.get("artifact_hash"), str):
        declared = _sha(value["artifact_hash"], "g22_asset_evidence.artifact_hash")
        if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != declared:
            raise _error("G22_AUTHORITATIVE_HASH_INVALID")
    asset_hits = _find_mappings(
        value,
        predicate=lambda item: any(
            key in item and item.get(key) == manifest.digest
            for key in ("asset_resolution_hash", "asset_manifest_hash", "asset_digest", "manifest_digest")
        ),
    )
    data_hits = _find_mappings(
        value,
        predicate=lambda item: any(
            key in item and item.get(key) == manifest.data_range.digest
            for key in ("data_range_hash", "data_manifest_hash", "data_digest")
        ),
    )
    if not asset_hits or not data_hits:
        raise _error("G22_ASSET_DATA_HASH_BINDING_INVALID")
    return value


def _derive_g22_gate(
    root: Path,
    *,
    s23_refs: Mapping[str, str],
    stage2_asset_ref: str,
    g22_ref: str,
    g22_sha256: str | None,
    stage1_ref: str,
    stage1_bridge_ref: str,
    stage1_config_ref: str,
    formal_execution_ref: str,
    base_config_ref: str,
    output_dir: str,
) -> tuple[str, GateRecord, str]:
    loaded = _load_task_result_set(
        root,
        s23_refs,
        task_id="stage2.03_assets_checkpoints_and_sampling",
    )
    asset_payload = loaded["asset_resolution"].payload
    manifest = _formal_s23_asset_manifest(asset_payload, field="s23.asset_resolution")
    authoritative = _load_authoritative_g22(
        root,
        g22_ref,
        manifest=manifest,
        expected_sha256=g22_sha256,
    )
    source_refs = tuple(
        dict.fromkeys(
            (
                formal_execution_ref,
                g22_ref,
                stage2_asset_ref,
                stage1_ref,
                stage1_bridge_ref,
                stage1_config_ref,
                *s23_refs.values(),
            )
        )
    )
    config, config_ref = _publish_bridge_config(
        root,
        base_config_ref=base_config_ref,
        task_id="stage2.03_assets_checkpoints_and_sampling",
        input_refs=source_refs,
        output_ref=f"{output_dir}/bridges/g22-resolved-config-v2.json",
    )
    measured = {
        "bridge_config_hash": config.config_hash,
        "bridge_config_full_hash": config.full_hash,
        "stage1_g1_exit_ref": stage1_ref,
        "stage1_bridge_evidence_ref": stage1_bridge_ref,
        "stage1_bridge_config_ref": stage1_config_ref,
        "g22_authoritative_payload_hash": canonical_json_hash(dict(authoritative)),
        "s23_payload_hashes": {kind: canonical_json_hash(dict(item.payload)) for kind, item in loaded.items()},
        "s23_artifact_hashes": {kind: item.identity.artifact_hash for kind, item in loaded.items()},
        "asset_resolution_hash": manifest.digest,
        "data_range_hash": manifest.data_range.digest,
        "checkpoint_manifest_hashes": {
            record.checkpoint_id: record.manifest_sha256 for record in manifest.checkpoints
        },
    }
    gate_ref, gate = _publish_bridge_gate(
        root,
        gate_id="stage2.G2.2",
        producer_task_id="stage2.03_assets_checkpoints_and_sampling",
        config=config,
        config_ref=config_ref,
        upstream_refs=source_refs,
        measured=measured,
        output_dir=f"{output_dir}/bridges/stage2-g2-2",
    )
    return gate_ref, gate, config_ref


def _adapter_gate_ref(source: Mapping[str, Any], name: str) -> str:
    """Read an already-published adapter commit, never a caller GateRecord."""

    candidates = (
        f"{name}_adapter_output",
        f"{name}_adapter",
        f"{name}_gate",
        name,
    )
    raw: object = None
    for key in candidates:
        if key in source:
            raw = source[key]
            break
    if isinstance(raw, Mapping):
        raw = raw.get("commit_ref")
    if not isinstance(raw, str):
        raise _error(f"{name.upper()}_ADAPTER_OUTPUT_REQUIRED")
    return _source_ref(raw, f"{name}.adapter.commit_ref")


def _load_adapter_gate(
    root: Path,
    source: Mapping[str, Any],
    *,
    name: str,
    gate_id: str,
    task_id: str,
    expected_refs: Sequence[str],
) -> tuple[str, GateRecord]:
    """Reload and bind one independent G2.0/G2.1 adapter output."""

    ref = _adapter_gate_ref(source, name)
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except Exception as error:
        raise _error(f"{name.upper()}_ADAPTER_COMMIT_INVALID", ref) from error
    if loaded.identity.task_id != task_id or loaded.identity.artifact_kind != "gate_record":
        raise _error(f"{name.upper()}_ADAPTER_COMMIT_IDENTITY_INVALID")
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise _error(f"{name.upper()}_ADAPTER_FORMAL_REQUIRED")
    try:
        gate = GateRecord.from_mapping(dict(loaded.payload))
    except Exception as error:
        raise _error(f"{name.upper()}_ADAPTER_GATE_INVALID") from error
    if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
        raise _error(f"{name.upper()}_ADAPTER_GATE_NOT_PASS")
    observed = set(loaded.source_refs) | set(gate.evidence_refs)
    if not set(expected_refs).issubset(observed):
        raise _error(f"{name.upper()}_ADAPTER_LINEAGE_MISMATCH")
    return ref, gate


def _build_narrow_formal_environment(
    root: Path,
    *,
    formal_execution_ref: str,
    stage0_ref: str,
    stage1_ref: str,
    contract_refs: Mapping[int, str],
    stage1_10_refs: Mapping[str, str],
    stage1_11_refs: Mapping[str, str],
    asset_ref: str,
    g1_ref: str | None,
    gate_refs: Mapping[str, str],
    required_gate_ids: Sequence[str],
    output_ref: str,
    capability_refs: Mapping[str, Any] | None = None,
    asset_input_ref: str | None = None,
    asset_input_lineage_ref: str | None = None,
) -> TaskRuntimeEnvironment:
    """Build the S2.1/S2.3 preflight snapshot without G2.2/G2.3 work.

    The environment binds contract stages 0/1/2, the actual Stage 1 commit
    groups, the direct six-checkpoint manifest, and adapter Gate commits.  It
    deliberately does not probe devices or construct any G3 alias.
    """

    evidence = FormalExecutionEvidence.from_mapping(
        _load_mapping(root, _source_ref(formal_execution_ref, "formal_execution"), "formal_execution")
    )
    evidence.require_for_stage(2)
    stage0 = _source_ref(stage0_ref, "stage0_handoff")
    stage1 = _source_ref(stage1_ref, "stage1_g1_exit")
    try:
        validate_stage0_handoff(root, stage0, require_ready=True)
        validate_stage1_exit_evidence(root, stage1)
    except Exception as error:
        raise _error("STAGE0_STAGE1_READY_REQUIRED") from error
    freezes: dict[int, ContractFreeze] = {}
    normalized_contracts: dict[int, str] = {}
    for stage in (0, 1, 2):
        ref, freeze = _load_formal_contract_freeze(root, contract_refs.get(stage), stage=stage)
        normalized_contracts[stage] = ref
        freezes[stage] = freeze
    if evidence.contract_freeze_hash != freezes[2].artifact_hash:
        raise _error("CONTRACT_FREEZE_IDENTITY_MISMATCH", normalized_contracts[2])
    asset_value = _load_mapping(root, asset_ref, "stage2_asset_resolution")
    try:
        asset = AssetResolutionManifest.from_mapping(dict(asset_value))
        validate_formal_asset_identity(asset)
    except Exception as error:
        raise _error("FORMAL_ASSET_MANIFEST_INVALID", asset_ref) from error
    if asset.scope != "formal" or asset.status != "READY":
        raise _error("FORMAL_ASSET_MANIFEST_NOT_READY", asset_ref)
    all_gate_refs = dict(gate_refs)
    if g1_ref is not None:
        all_gate_refs["stage1.G1-EXIT"] = _source_ref(g1_ref, "gate.stage1.G1-EXIT")
    normalized_gates = _load_formal_gate_refs(
        root,
        all_gate_refs,
        required=tuple(all_gate_refs),
        expected_task={
            "stage2.G2.0": "stage2.01_scope_hypotheses_and_preregistration",
            "stage2.G2.1": "stage2.02_stage1_handoff_and_fixed_state_contract",
        },
        expected_artifact_kind={key: "gate_record" for key in ("stage2.G2.0", "stage2.G2.1")},
    )
    evidence_refs: dict[str, str] = {
        "formal_execution": _source_ref(formal_execution_ref, "formal_execution"),
        "stage0_handoff": stage0,
        "stage1_g1_exit": stage1,
        "contract_stage_0": normalized_contracts[0],
        "contract_stage_1": normalized_contracts[1],
        "contract_stage_2": normalized_contracts[2],
        "contract_freeze": normalized_contracts[2],
        "stage2_asset_resolution": _source_ref(asset_ref, "stage2_asset_resolution"),
    }
    if asset_input_ref is not None:
        evidence_refs["stage2_asset_resolution_input"] = _source_ref(
            asset_input_ref,
            "stage2_asset_resolution_input",
        )
    if asset_input_lineage_ref is not None:
        evidence_refs["stage2_asset_resolution_input_lineage"] = _source_ref(
            asset_input_lineage_ref,
            "stage2_asset_resolution_input_lineage",
        )
    evidence_refs.update({f"stage1_10_{kind}": _source_ref(ref, f"stage1_10.{kind}") for kind, ref in stage1_10_refs.items()})
    evidence_refs.update({f"stage1_11_{kind}": _source_ref(ref, f"stage1_11.{kind}") for kind, ref in stage1_11_refs.items()})
    evidence_refs.update({f"gate_{key.replace('.', '_').replace('-', '_').lower()}": ref for key, ref in normalized_gates.items()})
    normalized_capabilities: dict[str, str] = {}
    if capability_refs is not None:
        required_capabilities = {"server", "cuda", "model_assets", "data_assets"}
        if set(capability_refs) != required_capabilities:
            raise _error("CAPABILITY_REF_SET_INVALID", sorted(set(capability_refs) ^ required_capabilities))
        loaded_capabilities = {
            capability: _load_capability(root, capability_refs[capability], capability)
            for capability in sorted(required_capabilities)
        }
        normalized_capabilities = {
            capability: loaded[0]
            for capability, loaded in loaded_capabilities.items()
        }
        evidence_refs.update(
            {f"capability_{key}": ref for key, ref in normalized_capabilities.items()}
        )
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(normalized_capabilities),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset(required_gate_ids),
        evidence_refs=evidence_refs,
    )
    target = _safe_relative(root, output_ref, "formal_environment_output")
    publish_canonical_immutable(target, environment.to_dict())
    reread = TaskRuntimeEnvironment.from_mapping(load_canonical_json(target))
    if reread.environment_hash != environment.environment_hash:
        raise _error("ENVIRONMENT_ROUND_TRIP_DRIFT", output_ref)
    return reread


def execute_formal_predecessor_dag(
    data_root: str | Path,
    *,
    source: Mapping[str, Any],
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
    formal_execution_ref: str,
    base_config_ref: str,
    output_dir: str,
) -> FormalDAGResult:
    """Produce only the canonical S2.1 and S2.3 formal task commits.

    G2.0/G2.1 are independent adapter authorities.  Their existing commits
    are consumed after S2.1 and before S2.3; if either is absent this function
    fails closed instead of signing a candidate or running S2.2/G2.2.
    """

    if raw_refs is not None:
        raise _error("CANDIDATE_INPUTS_FORBIDDEN")
    root = Path(data_root).resolve()
    stage1_ref = _source_ref(source.get("stage1_g1_exit"), "stage1_g1_exit")
    stage1_10 = _stage1_10_task_refs(source)
    stage1_11 = _stage1_task_refs(source)
    stage1_commits, _stage1_exit, _, _ = _load_stage1_formal_commits(
        root,
        source,
        stage1_index_ref=stage1_ref,
        base_config_ref=base_config_ref,
        output_dir=output_dir,
    )
    stage0_ref = _source_ref(source.get("stage0_handoff"), "stage0_handoff")
    contract_refs = {
        0: _source_ref(source.get("contract_stage_0"), "contract_stage_0"),
        1: _source_ref(source.get("contract_stage_1"), "contract_stage_1"),
        2: _source_ref(source.get("contract_stage_2", source.get("contract_freeze")), "contract_stage_2"),
    }
    asset_input_ref: str | None = None
    for key in ("stage2_asset_resolution_manifest", "stage2_asset_resolution"):
        candidate = source.get(key)
        if isinstance(candidate, str):
            asset_input_ref = _source_ref(candidate, key)
            break
    asset_ref, asset_manifest = _publish_authoritative_asset_manifest(
        root,
        source=source,
        raw_refs=None,
        output_dir=output_dir,
    )
    if asset_ref == asset_input_ref:
        asset_input_ref = None
    asset_input_lineage_ref = (
        PurePosixPath(output_dir, "asset-resolution-input-lineage.json").as_posix()
        if asset_input_ref is not None
        else None
    )
    formal_value = _load_mapping(root, _source_ref(formal_execution_ref, "formal_execution"), "formal_execution")
    evidence = FormalExecutionEvidence.from_mapping(formal_value)
    evidence.require_for_stage(2)
    gate_source = _mapping(source.get("gate_refs"), "gate_refs")
    g1_ref = _source_ref(gate_source.get("stage1.G1-EXIT"), "gate.stage1.G1-EXIT")
    runtime = TaskRuntime(workspace_root=root)
    register_stage23_runners(runtime, root)
    refs_by_task: dict[str, dict[str, str]] = {}
    s21_task = "stage2.01_scope_hypotheses_and_preregistration"
    s21_config = _formal_dag_config(
        root,
        base_config_ref=base_config_ref,
        task_id=s21_task,
        input_refs=tuple(stage1_11.values()),
        output_dir=f"{output_dir}/task-outputs/stage2-01",
    )
    s21_config_ref = _publish_resolved_config(
        root,
        s21_config,
        output_ref=f"{output_dir}/configs/stage2-01/resolved-config-v2.json",
    )
    env1 = _build_narrow_formal_environment(
        root,
        formal_execution_ref=formal_execution_ref,
        stage0_ref=stage0_ref,
        stage1_ref=stage1_ref,
        contract_refs=contract_refs,
        stage1_10_refs=stage1_10,
        stage1_11_refs=stage1_11,
        asset_ref=asset_ref,
        g1_ref=g1_ref,
        gate_refs={},
        required_gate_ids=("stage1.G1-EXIT",),
        output_ref=f"{output_dir}/environments/stage2-01.json",
        asset_input_ref=asset_input_ref,
        asset_input_lineage_ref=asset_input_lineage_ref,
    )
    result = runtime.execute(s21_config, environment=env1)
    if result.status is not TaskRunStatus.PASS or not result.formal_eligible:
        raise _error("FORMAL_DAG_TASK_NOT_PASS", f"{s21_task}:{result.status.value}")
    refs_by_task[s21_task] = dict(result.artifact_refs)

    g20_ref, g20_gate = _load_adapter_gate(
        root,
        source,
        name="g20",
        gate_id="stage2.G2.0",
        task_id=s21_task,
        expected_refs=tuple(refs_by_task[s21_task].values()),
    )
    g21_ref, _g21_gate = _load_adapter_gate(
        root,
        source,
        name="g21",
        gate_id="stage2.G2.1",
        task_id="stage2.02_stage1_handoff_and_fixed_state_contract",
        expected_refs=tuple((*refs_by_task[s21_task].values(), *stage1_10.values(), *stage1_11.values())),
    )
    # Produce S2.2 through the reviewed TaskRuntime handler.  G2.1 remains
    # bound to S2.1 + Stage 1 upstream; the newly produced S2.2 commits bind
    # both adapter gates and the exact environment snapshot instead.
    s22_refs, evidence, phase_evidence_ref, _s22_config_ref, _s22_environment_ref = produce_formal_s22_task_outputs(
        root,
        source=source,
        s21_refs=refs_by_task[s21_task],
        g20_ref=g20_ref,
        g20_gate=g20_gate,
        g21_ref=g21_ref,
        g21_gate=_g21_gate,
        g21_resolved_config_ref=_source_ref(source.get("g21_resolved_config"), "g21_resolved_config"),
        formal_execution_ref=formal_execution_ref,
        base_config_ref=base_config_ref,
        stage0_ref=stage0_ref,
        stage1_ref=stage1_ref,
        contract_refs=contract_refs,
        g3_ref=_source_ref(source.get("g3_resolution"), "g3_resolution"),
        stage1_10_refs=stage1_10,
        stage1_11_refs=stage1_11,
        g1_ref=g1_ref,
        output_dir=S204_S22_CONTROL_OUTPUT_DIR,
    )
    refs_by_task["stage2.02_stage1_handoff_and_fixed_state_contract"] = s22_refs
    s23_task = "stage2.03_assets_checkpoints_and_sampling"
    s23_config = _formal_dag_config(
        root,
        base_config_ref=base_config_ref,
        task_id=s23_task,
        input_refs=tuple(refs_by_task[s21_task].values()),
        output_dir=f"{output_dir}/task-outputs/stage2-03",
    )
    _publish_resolved_config(
        root,
        s23_config,
        output_ref=f"{output_dir}/configs/stage2-03/resolved-config-v2.json",
    )
    env3 = _build_narrow_formal_environment(
        root,
        formal_execution_ref=phase_evidence_ref,
        stage0_ref=stage0_ref,
        stage1_ref=stage1_ref,
        contract_refs=contract_refs,
        stage1_10_refs=stage1_10,
        stage1_11_refs=stage1_11,
        asset_ref=asset_ref,
        g1_ref=g1_ref,
        gate_refs={"stage2.G2.0": g20_ref, "stage2.G2.1": g21_ref},
        required_gate_ids=("stage2.G2.1",),
        output_ref=f"{output_dir}/environments/stage2-03.json",
        capability_refs=_mapping(source.get("capability_refs"), "capability_refs"),
        asset_input_ref=asset_input_ref,
        asset_input_lineage_ref=asset_input_lineage_ref,
    )
    result = runtime.execute(s23_config, environment=env3)
    if result.status is not TaskRunStatus.PASS or not result.formal_eligible:
        raise _error("FORMAL_DAG_TASK_NOT_PASS", f"{s23_task}:{result.status.value}")
    refs_by_task[s23_task] = dict(result.artifact_refs)
    return FormalDAGResult(
        refs_by_task,
        final_evidence=evidence,
        final_evidence_ref=phase_evidence_ref,
        bridge_gate_refs={"stage2.G2.0": g20_ref, "stage2.G2.1": g21_ref},
        authoritative_asset_ref=asset_ref,
        stage1_bridge_ref=None,
        stage1_bridge_config_ref=None,
        stage1_10_refs=stage1_10,
        stage1_11_refs=stage1_11,
    )


def _publish_formal_execution(
    data_root: Path,
    source_ref: str,
    *,
    destination: str,
) -> tuple[FormalExecutionEvidence, str]:
    ref = _source_ref(source_ref, "formal_execution")
    value = _load_mapping(data_root, ref, "formal_execution")
    try:
        evidence = FormalExecutionEvidence.from_mapping(value)
        evidence.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_INVALID", ref) from error
    target = _safe_relative(data_root, destination, "formal_execution_destination")
    publish_canonical_immutable(target, evidence.to_dict())
    try:
        reread = FormalExecutionEvidence.from_mapping(load_canonical_json(target))
        reread.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_ROUND_TRIP_DRIFT", destination) from error
    return evidence, PurePosixPath(destination).as_posix()


def _load_capability(
    data_root: Path, ref: str, capability: str
) -> tuple[str, RuntimeCapabilityEvidence]:
    source = _source_ref(ref, f"capability.{capability}")
    try:
        loaded = load_committed_task_artifact(data_root, source, require_formal=True)
    except Exception as error:
        raise _error("CAPABILITY_FORMAL_COMMIT_REQUIRED", capability) from error
    for source_ref in loaded.source_refs:
        _reject_nonformal_ref(source_ref, f"capability.{capability}.source_ref")

    # Capability producers use both the direct RuntimeCapabilityEvidence payload
    # and a small task envelope containing it.  Accept only one exact, hash-bound
    # evidence object; never infer capability from an arbitrary metadata field.
    candidates: list[Mapping[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if node.get("schema_version") == "runtime-capability-evidence-v1":
                candidates.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(loaded.payload)
    if len(candidates) != 1:
        raise _error("CAPABILITY_EVIDENCE_INVALID", capability)
    try:
        evidence = RuntimeCapabilityEvidence.from_mapping(candidates[0])
    except Exception as error:
        raise _error("CAPABILITY_EVIDENCE_INVALID", capability) from error
    if evidence.capability != capability or not evidence.verified:
        raise _error("CAPABILITY_NOT_VERIFIED", capability)
    return source, evidence


def _find_mappings(value: object, *, predicate: Any) -> tuple[Mapping[str, Any], ...]:
    found: list[Mapping[str, Any]] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            if predicate(node):
                found.append(node)
            for child in node.values():
                visit(child)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for child in node:
                visit(child)

    visit(value)
    return tuple(found)


def _load_authoritative_g21(
    root: Path,
    g21_handoff_ref: str,
    *,
    expected_stage1_ref: str | None = None,
) -> Mapping[str, Any]:
    """Load either the canonical handoff or the dated custom PASS report.

    The custom server report is an adapter input only.  Its status, task/gate
    identity, Stage1 index identity and raw GPU smoke are checked before any
    values are projected into a GateRecord.
    """

    path = _safe_relative(root, g21_handoff_ref, "g21_handoff")
    try:
        gpu_handoff = load_g21_formal_handoff(path, data_root=root)
    except Exception:
        value = _load_mapping(root, g21_handoff_ref, "g21_handoff")
        if (
            value.get("schema_version") != "stage2-s2.2-g2.1-handoff-evidence-v1"
            or value.get("status") != "PASS"
            or value.get("task_id") != "stage2.02_stage1_handoff_and_fixed_state_contract"
        ):
            raise _error("GPU_HEALTH_EVIDENCE_INVALID", g21_handoff_ref)
        stage1_values = _find_mappings(
            value,
            predicate=lambda item: isinstance(item.get("index_ref"), str)
            and isinstance(item.get("index_artifact_hash"), str),
        )
        if len(stage1_values) != 1:
            raise _error("G21_STAGE1_BINDING_INVALID", g21_handoff_ref)
        if expected_stage1_ref is not None and stage1_values[0].get("index_ref") != expected_stage1_ref:
            raise _error("G21_STAGE1_REF_MISMATCH", g21_handoff_ref)
        gpu_handoff = value

    smoke = gpu_handoff.get("current_gpu_smoke")
    if not isinstance(smoke, Mapping):
        candidates = _find_mappings(
            gpu_handoff,
            predicate=lambda item: item.get("schema_version") == "stage2-s202-current-gpu-smoke-v1"
            and isinstance(item.get("ref"), str)
            and isinstance(item.get("sha256"), str),
        )
        if len(candidates) != 1:
            raise _error("GPU_SMOKE_SUMMARY_INVALID", g21_handoff_ref)
        smoke = candidates[0]
    if expected_stage1_ref is not None:
        stage1_values = _find_mappings(
            gpu_handoff,
            predicate=lambda item: isinstance(item.get("ref"), str)
            and item.get("ref") == expected_stage1_ref,
        )
        if not stage1_values:
            raise _error("G21_STAGE1_REF_MISMATCH", g21_handoff_ref)
    allowed_devices = tuple(
        (str(item.get("pci_bus_id")), str(item.get("uuid")))
        for item in smoke.get("allowed_devices", [])
        if isinstance(item, Mapping)
    )
    if allowed_devices != G21_ALLOWED_DEVICES:
        raise _error("GPU_ALLOWED_IDENTITY_DRIFT", g21_handoff_ref)
    if smoke.get("excluded_pci_bus_ids") != [G21_EXCLUDED_PCI]:
        raise _error("GPU_EXCLUDED_PCI_DRIFT", g21_handoff_ref)
    smoke_ref = _source_ref(smoke.get("ref"), "g21_handoff.current_gpu_smoke.ref")
    report_sha = _sha(smoke.get("sha256"), "g21_current_gpu_smoke.sha256")
    _assert_source_file(root, smoke_ref, "g21_current_gpu_smoke", sha256=report_sha)
    # The dated external G2.1 handoff is a custom adapter envelope.  Its raw
    # smoke report is hash-bound above, but historical reports are valid JSON
    # that is not emitted by the repository canonical serializer.  Keep the
    # canonical loader strict everywhere else and parse only this already
    # hash-verified external report directly.
    report_path = _safe_relative(root, smoke_ref, "g21_current_gpu_smoke")
    try:
        report = _mapping(json.loads(report_path.read_text(encoding="utf-8")), "g21_current_gpu_smoke")
    except (OSError, UnicodeError, TypeError, ValueError) as error:
        raise _error("SOURCE_UNREADABLE", f"g21_current_gpu_smoke:{smoke_ref}") from error
    if report.get("schema_version") != "stage2-s202-current-gpu-smoke-v1" or report.get("status") != "PASS":
        raise _error("GPU_SMOKE_REPORT_INVALID", smoke_ref)
    if report.get("excluded_pci_bus_ids") != [EXCLUDED_PCI]:
        raise _error("GPU_SMOKE_REPORT_EXCLUSION_INVALID", smoke_ref)
    excluded = report.get("excluded_device")
    if (
        not isinstance(excluded, Mapping)
        or str(excluded.get("pci_bus_id", "")).casefold() != EXCLUDED_PCI.casefold()
        or str(excluded.get("uuid", "")).casefold() != EXCLUDED_UUID.casefold()
        or excluded.get("scheduled") is not False
    ):
        raise _error("GPU_SMOKE_REPORT_EXCLUDED_IDENTITY_INVALID", smoke_ref)
    report_devices = tuple(
        (str(item.get("pci_bus_id")), str(item.get("uuid")))
        for item in report.get("allowed_devices", [])
        if isinstance(item, Mapping)
    )
    if report_devices != allowed_devices:
        raise _error("GPU_SMOKE_RUNTIME_MAPPING_DRIFT", smoke_ref)
    return gpu_handoff


def _load_gpu_health_identity(
    root: Path,
    g21_handoff_ref: str | None,
    *,
    expected_stage1_ref: str | None = None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Validate current smoke and retain PCI+UUID identity, never only index."""

    if g21_handoff_ref is None:
        raise _error("GPU_HEALTH_EVIDENCE_REQUIRED")
    gpu_ref = _source_ref(g21_handoff_ref, "g21_handoff")
    try:
        gpu_handoff = _load_authoritative_g21(
            root,
            gpu_ref,
            expected_stage1_ref=expected_stage1_ref,
        )
    except S204MaterializationError:
        raise
    except Exception as error:
        raise _error("GPU_HEALTH_EVIDENCE_INVALID", gpu_ref) from error
    smoke = gpu_handoff.get("current_gpu_smoke")
    if not isinstance(smoke, Mapping):
        candidates = _find_mappings(
            gpu_handoff,
            predicate=lambda item: item.get("schema_version") == "stage2-s202-current-gpu-smoke-v1",
        )
        if len(candidates) != 1:
            raise _error("GPU_SMOKE_SUMMARY_INVALID", gpu_ref)
        smoke = candidates[0]
    allowed_devices = tuple(
        (str(item.get("pci_bus_id")), str(item.get("uuid")))
        for item in smoke.get("allowed_devices", [])
        if isinstance(item, Mapping)
    )
    return gpu_ref, allowed_devices


def _gpu_health_binding_payload(
    source_ref: str,
    allowed_devices: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Build a binding containing only stable PCI/UUID identities.

    Physical nvidia-smi indices are intentionally not persisted.  They are
    ephemeral selectors and must be resolved against a fresh inventory by the
    formal launcher immediately before a cell starts.
    """

    payload: dict[str, Any] = {
        "schema_version": "stage2-s204-gpu-health-binding-v1",
        "source_ref": source_ref,
        "excluded": {"pci_bus_id": EXCLUDED_PCI, "uuid": EXCLUDED_UUID},
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid}
            for pci, uuid in allowed_devices
        ],
    }
    payload["binding_hash"] = canonical_json_hash(payload)
    return payload


def _load_formal_gate_refs(
    root: Path,
    gate_refs: Mapping[str, str],
    *,
    required: Sequence[str],
    expected_task: Mapping[str, str] | None = None,
    expected_artifact_kind: Mapping[str, str] | None = None,
    expected_upstream_refs: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Load exactly one PASS GateRecord per formal TaskRuntime gate."""

    if not set(required).issubset(gate_refs):
        raise _error("GATE_REF_SET_INVALID", sorted(set(required) - set(gate_refs)))
    normalized: dict[str, str] = {}
    for gate_id in required:
        gate_ref = _source_ref(gate_refs[gate_id], f"gate.{gate_id}")
        try:
            loaded = load_committed_task_artifact(root, gate_ref, require_formal=True)
        except Exception as error:
            raise _error("GATE_FORMAL_COMMIT_REQUIRED", gate_id) from error
        if expected_task is not None and gate_id in expected_task and loaded.identity.task_id != expected_task[gate_id]:
            raise _error("GATE_TASK_IDENTITY_MISMATCH", gate_id)
        if expected_artifact_kind is not None and gate_id in expected_artifact_kind and loaded.identity.artifact_kind != expected_artifact_kind[gate_id]:
            raise _error("GATE_ARTIFACT_KIND_MISMATCH", gate_id)
        for source_ref in loaded.source_refs:
            _reject_nonformal_ref(source_ref, f"gate.{gate_id}.source_ref")
        found: list[GateRecord] = []

        def visit(node: object) -> None:
            if isinstance(node, Mapping):
                if node.get("schema_version") == "gate-record-v1":
                    try:
                        gate = GateRecord.from_mapping(dict(node))
                    except Exception:
                        gate = None
                    if gate is not None and gate.gate_id == gate_id:
                        found.append(gate)
                    return
                for child in node.values():
                    visit(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
                for child in node:
                    visit(child)

        visit(loaded.payload)
        if len(found) != 1 or found[0].status is not GateStatus.PASS:
            raise _error("GATE_NOT_PASS", gate_id)
        if expected_upstream_refs is not None:
            expected = tuple(expected_upstream_refs.get(gate_id, ()))
            if not set(expected).issubset(set(found[0].evidence_refs)):
                raise _error("GATE_UPSTREAM_BINDING_MISMATCH", gate_id)
        normalized[gate_id] = gate_ref
    return normalized


def build_formal_runtime_environment(
    data_root: str | Path,
    *,
    formal_execution_ref: str,
    stage0_handoff_ref: str,
    stage1_g1_exit_ref: str,
    contract_freeze_ref: str,
    g3_resolution_ref: str,
    stage2_asset_resolution_ref: str,
    gate_refs: Mapping[str, str],
    capability_refs: Mapping[str, str],
    contract_stage_0_ref: str | None = None,
    contract_stage_1_ref: str | None = None,
    contract_stage_2_ref: str | None = None,
    g21_handoff_ref: str | None = None,
    stage1_bridge_ref: str | None = None,
    stage1_bridge_config_ref: str | None = None,
    reference_sizing_plan_ref: str | None = None,
    g22_asset_evidence_ref: str | None = None,
    stage2_parameter_registry_ref: str | None = None,
    stage2_reference_delta_sci_ref: str | None = None,
    cell_id: str | None = None,
    six_cell_manifest_ref: str | None = None,
    delta_phase: str = "pre_sizing",
    output_ref: str = "evidence/stage2/s204/runtime-environment.json",
) -> tuple[TaskRuntimeEnvironment, str]:
    """Build a rereadable environment bound to current formal evidence.

    Stage 0/Stage 1 validators are reused verbatim.  G3 and Stage 2.3 asset
    resolution are loaded from formal task commits and semantically checked;
    no path or hash supplied by a caller is accepted as a substitute for the
    published object.
    """

    root = Path(data_root).resolve()
    formal_ref = _source_ref(formal_execution_ref, "formal_execution")
    formal_value = _load_mapping(root, formal_ref, "formal_execution")
    try:
        evidence = FormalExecutionEvidence.from_mapping(formal_value)
        evidence.require_for_stage(2)
    except Exception as error:
        raise _error("FORMAL_EXECUTION_INVALID", formal_ref) from error

    # G2.1 is the current hardware identity authority.  Bind the full handoff
    # and its raw smoke report (including PCI+UUID pairs), not a numeric index
    # copied from one inventory.
    gpu_ref, allowed_devices = _load_gpu_health_identity(
        root,
        g21_handoff_ref,
        expected_stage1_ref=stage1_g1_exit_ref,
    )
    # Store a dedicated binding object so a consumer can re-read both stable
    # hardware identities.  Current scheduler indices are intentionally absent:
    # they are ephemeral selectors resolved against live nvidia-smi output.
    binding_ref = PurePosixPath(output_ref).with_name("gpu-health-binding.json").as_posix()
    binding_payload = _gpu_health_binding_payload(gpu_ref, allowed_devices)
    binding_target = _safe_relative(root, binding_ref, "gpu_binding_output")
    publish_canonical_immutable(binding_target, binding_payload)
    binding_round_trip = _load_mapping(root, binding_ref, "gpu_binding")
    if binding_round_trip != binding_payload:
        raise _error("GPU_BINDING_ROUND_TRIP_DRIFT")

    stage0_ref = _source_ref(stage0_handoff_ref, "stage0_handoff")
    try:
        validate_stage0_handoff(root, stage0_ref, require_ready=True)
    except Exception as error:
        raise _error("STAGE0_READY_REQUIRED", stage0_ref) from error
    stage1_ref = _source_ref(stage1_g1_exit_ref, "stage1_g1_exit")
    try:
        validate_stage1_exit_evidence(root, stage1_ref)
    except Exception as error:
        raise _error("STAGE1_EXIT_REQUIRED", stage1_ref) from error

    freeze_ref = _source_ref(contract_freeze_ref, "contract_freeze")
    stage2_contract_ref = contract_stage_2_ref or freeze_ref
    if contract_stage_2_ref is not None and _source_ref(contract_stage_2_ref, "contract_stage_2") != freeze_ref:
        raise _error("CONTRACT_STAGE_2_REF_MISMATCH")
    contract_refs: dict[int, str] = {}
    freezes: dict[int, ContractFreeze] = {}
    for stage, stage_ref in (
        (0, contract_stage_0_ref),
        (1, contract_stage_1_ref),
        (2, stage2_contract_ref),
    ):
        loaded_ref, loaded_freeze = _load_formal_contract_freeze(root, stage_ref, stage=stage)
        contract_refs[stage] = loaded_ref
        freezes[stage] = loaded_freeze
    if freezes[2].artifact_hash != evidence.contract_freeze_hash:
        raise _error("CONTRACT_FREEZE_IDENTITY_MISMATCH", contract_refs[2])
    contract_document_ref = _publish_contract_document(
        root,
        freezes[2],
        output_ref=PurePosixPath(output_ref).with_name("contract-freeze-stage2.json").as_posix(),
    )

    g3_source = _source_ref(g3_resolution_ref, "g3_resolution")
    try:
        g3 = load_committed_task_artifact(root, g3_source, require_formal=True)
    except Exception as error:
        raise _error("G3_FORMAL_COMMIT_REQUIRED", g3_source) from error
    if g3.identity.artifact_kind != "asset_resolution":
        raise _error("G3_ARTIFACT_KIND_MISMATCH", g3_source)
    try:
        # Reuse the authoritative G3 loader so a formal-looking task envelope
        # cannot stand in for qualified model/data/tokenizer manifests.
        FormalG3RuntimeAssets.load(root, g3_source)
    except Exception as error:
        raise _error("G3_RUNTIME_ASSETS_INVALID", g3_source) from error

    asset_source = _source_ref(stage2_asset_resolution_ref, "stage2_asset_resolution")
    try:
        raw_asset = _load_formal_or_direct_payload(
            root,
            asset_source,
            field="stage2_asset_resolution",
            artifact_kind="asset_resolution",
        )
        if raw_asset.get("schema_version") == "stage2-task-asset-resolution-v1":
            asset_manifest = _formal_s23_asset_manifest(raw_asset, field=asset_source)
        else:
            asset_manifest = AssetResolutionManifest.from_mapping(raw_asset)
            validate_formal_asset_identity(asset_manifest)
    except Exception as error:
        raise _error("S23_ASSET_RESOLUTION_INVALID", asset_source) from error
    # G2.2's formal gate binds the canonical S2.3 asset commit directly.  Its
    # TaskArtifact source_refs are S2.1 lineage and are intentionally not
    # duplicated in the G2.2 gate evidence_refs.
    asset_gate_refs: tuple[str, ...] = (asset_source,)

    # Formal S2.4 consumes the complete adapter chain.  A lone G2.2 record is
    # insufficient because it cannot prove the fixed-state handoff or the
    # preregistration contract that produced the S2.3 asset set.
    required_gates = {"stage2.G2.0", "stage2.G2.1", "stage2.G2.2"}
    optional_gates = {"stage1.G1-EXIT"}
    if not required_gates.issubset(gate_refs) or set(gate_refs) - required_gates - optional_gates:
        raise _error("GATE_REF_SET_INVALID", sorted(set(gate_refs) ^ required_gates))
    normalized_gate_refs = _load_formal_gate_refs(
        root,
        {str(key): str(value) for key, value in gate_refs.items()},
        required=tuple(gate_refs),
        expected_task={
            "stage2.G2.0": "stage2.01_scope_hypotheses_and_preregistration",
            "stage2.G2.1": "stage2.02_stage1_handoff_and_fixed_state_contract",
            "stage2.G2.2": "stage2.03_assets_checkpoints_and_sampling",
        },
        expected_artifact_kind={
            "stage2.G2.0": "gate_record",
            "stage2.G2.1": "gate_record",
            "stage2.G2.2": "gate_record",
        },
        expected_upstream_refs={"stage2.G2.2": asset_gate_refs},
    )

    required_capabilities = {"server", "cuda", "model_assets", "data_assets"}
    if set(capability_refs) != required_capabilities:
        raise _error("CAPABILITY_REF_SET_INVALID", sorted(set(capability_refs) ^ required_capabilities))
    loaded_capabilities = {
        capability: _load_capability(root, ref, capability)
        for capability, ref in capability_refs.items()
    }
    # When the CUDA probe publishes a runtime inventory, require it to agree
    # with the current G2.1 PCI+UUID map.  Missing optional inventory metadata
    # is tolerated for non-GPU capability probes; a present but drifting map is
    # always a hard block.
    cuda_metadata = loaded_capabilities["cuda"][1].metadata
    runtime_devices = cuda_metadata.get("allowed_devices")
    if runtime_devices is not None:
        if not isinstance(runtime_devices, Sequence) or isinstance(runtime_devices, (str, bytes, bytearray)):
            raise _error("GPU_RUNTIME_INVENTORY_INVALID")
        runtime_pairs = tuple(
            (str(item.get("pci_bus_id")), str(item.get("uuid")))
            for item in runtime_devices
            if isinstance(item, Mapping)
        )
        if runtime_pairs != allowed_devices:
            raise _error("GPU_RUNTIME_INVENTORY_DRIFT")
    runtime_excluded = cuda_metadata.get("excluded_device")
    if runtime_excluded is not None:
        if not isinstance(runtime_excluded, Mapping) or str(runtime_excluded.get("pci_bus_id")) != EXCLUDED_PCI or str(runtime_excluded.get("uuid")) != EXCLUDED_UUID:
            raise _error("GPU_RUNTIME_EXCLUDED_IDENTITY_DRIFT")
    normalized_capabilities = {
        capability: item[0] for capability, item in loaded_capabilities.items()
    }

    if reference_sizing_plan_ref is None:
        raise _error("SIZING_PLAN_REF_REQUIRED")
    sizing_ref = _source_ref(reference_sizing_plan_ref, "formal_reference_sizing_plan")
    sizing_value = _load_mapping(root, sizing_ref, "formal_reference_sizing_plan")
    try:
        sizing_execution_hash = sizing_value.get("execution_evidence_hash")
        sizing_plan = ReferenceSizingPlan(
            reference_id=str(sizing_value["reference_id"]),
            candidate_sample_counts=tuple(sizing_value["candidate_sample_counts"]),  # type: ignore[arg-type]
            block_size=int(sizing_value["block_size"]),
            convergence_tolerance=float(sizing_value["convergence_tolerance"]),
            required_consecutive=int(sizing_value["required_consecutive"]),
            execution=evidence,
        )
    except Exception as error:
        raise _error("SIZING_PLAN_INVALID", sizing_ref) from error
    if (
        sizing_value.get("schema_version") != S204_SCHEMA
        or sizing_execution_hash != evidence.artifact_hash
        or sizing_value.get("artifact_hash") != sizing_plan.artifact_hash
    ):
        raise _error("SIZING_PLAN_IDENTITY_MISMATCH", sizing_ref)

    if (stage2_parameter_registry_ref is None) != (stage2_reference_delta_sci_ref is None):
        raise _error("CELL_FORMAL_EVIDENCE_SET_INVALID")
    cell_registry_ref: str | None = None
    cell_delta_ref: str | None = None
    if stage2_parameter_registry_ref is not None and stage2_reference_delta_sci_ref is not None:
        if cell_id is None:
            raise _error("CELL_ID_REQUIRED_FOR_FORMAL_EVIDENCE")
        if cell_id not in EXPECTED_CELL_IDS:
            raise _error("CELL_ID_INVALID", cell_id)
        cell_registry_ref = _source_ref(stage2_parameter_registry_ref, "stage2_parameter_registry")
        cell_delta_ref = _source_ref(stage2_reference_delta_sci_ref, "stage2_reference_delta_sci")
        registry_value = _load_formal_or_direct_payload(
            root,
            cell_registry_ref,
            field="stage2_parameter_registry",
            artifact_kind="parameter_registry",
        )
        if registry_value.get("schema_version") != S204_REGISTRY_SCHEMA or registry_value.get("status") != "READY" or registry_value.get("cell_id") != cell_id:
            raise _error("PARAMETER_REGISTRY_FORMAL_REQUIRED", cell_id)
        if registry_value.get("artifact_hash") != canonical_json_hash(
            {key: item for key, item in registry_value.items() if key != "artifact_hash"}
        ):
            raise _error("PARAMETER_REGISTRY_HASH_INVALID", cell_id)
        _validate_delta_sci_artifact(
            root,
            cell_delta_ref,
            cell_id=cell_id,
            allow_pre_sizing_plan=delta_phase == "pre_sizing",
        )

    evidence_refs: dict[str, str] = {
        "formal_execution": formal_ref,
        "stage0_handoff": stage0_ref,
        "stage1_g1_exit": stage1_ref,
        # TaskRuntime uses contract_stage_2 (formal commit); stage23's
        # FormalExecutionEvidence loader consumes this payload document and
        # compares its ContractFreeze artifact hash.
        "contract_freeze": contract_document_ref,
        "contract_stage_0": contract_refs[0],
        "contract_stage_1": contract_refs[1],
        "contract_stage_2": contract_refs[2],
        "g3_resolution": g3_source,
        "stage2_asset_resolution": asset_source,
        "gpu_health": gpu_ref,
        "gpu_health_binding": binding_ref,
        # This exact key is consumed by stage23_task_runners._formal_input_document;
        # keeping one authoritative ref prevents an input scan from accepting a
        # second sizing plan accidentally placed in predecessor commits.
        "formal_reference_sizing_plan": sizing_ref,
    }
    if g22_asset_evidence_ref is not None:
        evidence_refs["g22_asset_evidence"] = _source_ref(g22_asset_evidence_ref, "g22_asset_evidence")
    if stage1_bridge_ref is not None:
        bridge_ref = _source_ref(stage1_bridge_ref, "stage1_bridge")
        if stage1_bridge_config_ref is None:
            raise _error("STAGE1_BRIDGE_CONFIG_REQUIRED")
        _validate_stage1_bridge(
            root,
            index_ref=stage1_ref,
            bridge_ref=bridge_ref,
            config_ref=stage1_bridge_config_ref,
        )
        evidence_refs["stage1_11_bridge"] = bridge_ref
    if stage1_bridge_config_ref is not None:
        bridge_config_ref = _source_ref(stage1_bridge_config_ref, "stage1_bridge_config")
        evidence_refs["stage1_11_bridge_config"] = bridge_config_ref
    evidence_refs.update({
        f"gate_{key.replace('.', '_').replace('-', '_').lower()}": value
        for key, value in normalized_gate_refs.items()
    })
    if cell_registry_ref is not None and cell_delta_ref is not None:
        evidence_refs["stage2_parameter_registry"] = cell_registry_ref
        evidence_refs["stage2_reference_delta_sci"] = cell_delta_ref
    if six_cell_manifest_ref is not None:
        evidence_refs["stage2_s204_six_cell_manifest"] = _source_ref(
            six_cell_manifest_ref, "six_cell_manifest"
        )
    evidence_refs.update({f"capability_{key}": value for key, value in normalized_capabilities.items()})
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(required_capabilities),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset(normalized_gate_refs),
        evidence_refs=evidence_refs,
    )
    target = _safe_relative(root, output_ref, "environment_output")
    publish_canonical_immutable(target, environment.to_dict())
    reread = TaskRuntimeEnvironment.from_mapping(load_canonical_json(target))
    if reread.environment_hash != environment.environment_hash:
        raise _error("ENVIRONMENT_ROUND_TRIP_DRIFT")
    return environment, PurePosixPath(output_ref).as_posix()


def publish_reference_sizing_plan(
    data_root: str | Path,
    *,
    formal_execution: FormalExecutionEvidence,
    reference_id: str = "stage2-s204-reference",
    candidate_sample_counts: Sequence[int] = DEFAULT_CANDIDATES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    convergence_tolerance: float = 0.02,
    # The frozen Stage 2 plan and existing formal runner both use one
    # adjacent convergence transition.  Raising this locally would change
    # the scientific stop rule without a preregistered basis.
    required_consecutive: int = 1,
    output_ref: str = "evidence/stage2/s204/reference-sizing-plan.json",
) -> tuple[ReferenceSizingPlan, str]:
    """Create and immutably publish the sizing-only S2.4 plan."""

    if formal_execution.run_intent != "formal":
        raise _error("FORMAL_EXECUTION_REQUIRED_FOR_SIZING")
    if required_consecutive != 1:
        raise _error("SIZING_REQUIRED_CONSECUTIVE_FROZEN", required_consecutive)
    plan = ReferenceSizingPlan(
        reference_id=reference_id,
        candidate_sample_counts=tuple(candidate_sample_counts),
        block_size=block_size,
        convergence_tolerance=convergence_tolerance,
        required_consecutive=required_consecutive,
        execution=formal_execution,
    )
    # Validate the exact wire contract before publication.  This also checks
    # that the plan is not accidentally written as a fixture/schema variant.
    from param_importance_nlp.contracts.stage23 import validate_stage23_artifact

    validate_stage23_artifact(plan.to_dict())
    root = Path(data_root).resolve()
    target = _safe_relative(root, output_ref, "sizing_plan_output")
    publish_canonical_immutable(target, plan.to_dict())
    reread = _load_mapping(root, PurePosixPath(output_ref).as_posix(), "sizing_plan")
    if reread.get("schema_version") != S204_SCHEMA or reread.get("artifact_hash") != plan.artifact_hash:
        raise _error("SIZING_PLAN_ROUND_TRIP_DRIFT")
    return plan, PurePosixPath(output_ref).as_posix()


def publish_per_cell_sizing_plans(
    data_root: str | Path,
    *,
    formal_execution: FormalExecutionEvidence,
    candidate_sample_counts: Sequence[int] = DEFAULT_CANDIDATES,
    block_size: int = DEFAULT_BLOCK_SIZE,
    convergence_tolerance: float = 0.02,
    output_dir: str = "evidence/stage2/s204/reference-sizing",
) -> dict[str, str]:
    """Publish six independent sizing studies with stable cell identities."""

    root = Path(data_root).resolve()
    refs: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        component = _cell_path_component(cell_id)
        reference_id = f"stage2-s204-{component}-sizing"
        output_ref = f"{output_dir}/{component}/reference-sizing-plan.json"
        _plan, ref = publish_reference_sizing_plan(
            root,
            formal_execution=formal_execution,
            reference_id=reference_id,
            candidate_sample_counts=candidate_sample_counts,
            block_size=block_size,
            convergence_tolerance=convergence_tolerance,
            required_consecutive=1,
            output_ref=output_ref,
        )
        refs[cell_id] = ref
    if len(set(refs.values())) != len(EXPECTED_CELL_IDS):
        raise _error("SIX_CELL_SIZING_REF_NOT_UNIQUE")
    return refs


def _validate_delta_sci_artifact(
    root: Path,
    ref: str,
    *,
    cell_id: str,
    candidate_sample_counts: Sequence[int] = DEFAULT_CANDIDATES,
    allow_pre_sizing_plan: bool = False,
) -> Mapping[str, Any]:
    value = _load_formal_or_direct_payload(
        root,
        ref,
        field="stage2_reference_delta_sci",
        artifact_kind="reference_delta_sci",
    )
    if value.get("schema_version") == S204_DELTA_PLAN_SCHEMA:
        if not allow_pre_sizing_plan:
            raise _error("REFERENCE_DELTA_SCI_NUMERIC_REQUIRED", cell_id)
        if value.get("status") != "READY" or value.get("scope") != "formal" or value.get("phase") != "pre_sizing":
            raise _error("REFERENCE_DELTA_SCI_PLAN_FORMAL_REQUIRED", cell_id)
        if value.get("cell_id") != cell_id:
            raise _error("REFERENCE_DELTA_SCI_CELL_MISMATCH", cell_id)
        counts = value.get("candidate_sample_counts")
        if not isinstance(counts, list) or tuple(counts) != tuple(candidate_sample_counts):
            raise _error("REFERENCE_DELTA_SCI_PLAN_CANDIDATES_INVALID", cell_id)
        refs = value.get("source_contract_refs")
        hashes = value.get("source_contract_artifact_hashes")
        if not isinstance(refs, list) or not refs or not isinstance(hashes, list) or len(refs) != len(hashes):
            raise _error("REFERENCE_DELTA_SCI_SOURCE_BINDING_INVALID", cell_id)
        for raw_ref, raw_hash in zip(refs, hashes, strict=True):
            source_ref = _source_ref(raw_ref, f"delta_sci_plan.{cell_id}.source_ref")
            loaded = load_committed_task_artifact(root, source_ref, require_formal=True)
            if loaded.identity.artifact_hash != _sha(raw_hash, f"delta_sci_plan.{cell_id}.source_hash"):
                raise _error("REFERENCE_DELTA_SCI_SOURCE_BINDING_INVALID", cell_id)
        declared = value.get("artifact_hash")
        if not isinstance(declared, str) or canonical_json_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        ) != declared:
            raise _error("REFERENCE_DELTA_SCI_HASH_INVALID", cell_id)
        if "delta_sci_by_B" in value:
            raise _error("REFERENCE_DELTA_SCI_PLAN_MUST_NOT_CONTAIN_NUMBERS", cell_id)
        return value
    if value.get("schema_version") != S204_DELTA_SCHEMA or value.get("status") != "READY" or value.get("scope") != "formal":
        raise _error("REFERENCE_DELTA_SCI_FORMAL_REQUIRED", cell_id)
    if value.get("cell_id") != cell_id:
        raise _error("REFERENCE_DELTA_SCI_CELL_MISMATCH", cell_id)
    values = value.get("delta_sci_by_B")
    if not isinstance(values, Mapping) or any(
        str(count) not in values
        or isinstance(values[str(count)], bool)
        or not isinstance(values[str(count)], (int, float))
        or not math.isfinite(float(values[str(count)]))
        or float(values[str(count)]) <= 0
        for count in candidate_sample_counts
    ):
        raise _error("REFERENCE_DELTA_SCI_CANDIDATE_COVERAGE", cell_id)
    refs = value.get("source_contract_refs")
    hashes = value.get("source_contract_artifact_hashes")
    if not isinstance(refs, list) or not refs or not isinstance(hashes, list) or len(refs) != len(hashes):
        raise _error("REFERENCE_DELTA_SCI_SOURCE_BINDING_INVALID", cell_id)
    for raw_ref, raw_hash in zip(refs, hashes, strict=True):
        source_ref = _source_ref(raw_ref, f"delta_sci.{cell_id}.source_ref")
        loaded = load_committed_task_artifact(root, source_ref, require_formal=True)
        if loaded.identity.artifact_hash != _sha(raw_hash, f"delta_sci.{cell_id}.source_hash"):
            raise _error("REFERENCE_DELTA_SCI_SOURCE_BINDING_INVALID", cell_id)
    declared = value.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ) != declared:
        raise _error("REFERENCE_DELTA_SCI_HASH_INVALID", cell_id)
    return value


def _validate_sizing_ref(
    root: Path,
    ref: str,
    *,
    formal_execution_hash: str,
    cell_id: str,
) -> Mapping[str, Any]:
    value = _load_formal_or_direct_payload(
        root,
        ref,
        field="formal_reference_sizing_plan",
        artifact_kind="reference_sizing_plan",
    )
    if value.get("schema_version") != S204_SCHEMA or value.get("execution_evidence_hash") != formal_execution_hash:
        raise _error("SIZING_PLAN_IDENTITY_MISMATCH", cell_id)
    if value.get("required_consecutive") != 1:
        raise _error("SIZING_REQUIRED_CONSECUTIVE_FROZEN", cell_id)
    counts = value.get("candidate_sample_counts")
    if not isinstance(counts, list) or len(counts) < 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in counts
    ) or tuple(sorted(set(counts))) != tuple(counts):
        raise _error("SIZING_PLAN_INVALID", cell_id)
    if not isinstance(value.get("reference_id"), str) or not value.get("reference_id"):
        raise _error("SIZING_PLAN_INVALID", cell_id)
    if value.get("artifact_hash") != canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ):
        raise _error("SIZING_PLAN_HASH_INVALID", cell_id)
    return value


def _publish_auxiliary_task_artifact(
    root: Path,
    *,
    artifact_kind: str,
    payload: Mapping[str, Any],
    config_hash: str,
    source_refs: Sequence[str],
    output_dir: str,
) -> str:
    """Bridge one validated direct auxiliary object into a formal TaskArtifact."""

    refs = tuple(dict.fromkeys(_source_ref(ref, "auxiliary.source_ref") for ref in source_refs))
    store = TaskArtifactStore(root, output_dir)
    published = store.publish(
        task_id=S204_TASK_ID,
        artifact_kind=artifact_kind,
        config_hash=config_hash,
        run_intent="formal",
        payload=dict(payload),
        formal_eligible=True,
        source_refs=refs,
    )
    loaded = load_committed_task_artifact(root, published.commit_ref, require_formal=True)
    if loaded.identity.artifact_kind != artifact_kind or loaded.payload != payload:
        raise _error("AUXILIARY_TASK_ARTIFACT_DRIFT", artifact_kind)
    return published.commit_ref


def _load_direct_manifest_payload(root: Path, ref: str, field: str) -> Mapping[str, Any]:
    value = _load_mapping(root, ref, field)
    if value.get("schema_version") == "stage2-task-asset-resolution-v1":
        nested = value.get("stage2_asset_manifest")
        if not isinstance(nested, Mapping):
            raise _error("DIRECT_MANIFEST_INVALID", field)
        return nested
    return value


def _load_formal_or_direct_payload(
    root: Path,
    ref: str,
    *,
    field: str,
    artifact_kind: str,
) -> Mapping[str, Any]:
    """Read a direct artifact or its formal TaskArtifact envelope exactly once."""

    value = _load_mapping(root, ref, field)
    if value.get("schema_version") != "task-output-commit-v1":
        return value
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except Exception as error:
        raise _error("FORMAL_AUXILIARY_COMMIT_REQUIRED", field) from error
    if loaded.identity.artifact_kind != artifact_kind:
        raise _error("FORMAL_AUXILIARY_KIND_MISMATCH", field)
    return loaded.payload


def publish_per_cell_runtime_environments(
    data_root: str | Path,
    *,
    base_environment_ref: str,
    sizing_refs: Mapping[str, str],
    registry_refs: Mapping[str, str],
    delta_refs: Mapping[str, str],
    output_dir: str = "evidence/stage2/s204/environments",
    six_cell_manifest_ref: str | None = None,
    configs: Mapping[str, ResolvedConfigV2] | None = None,
    config_refs: Mapping[str, str] | None = None,
    s23_asset_task_ref: str | None = None,
    asset_manifest_ref: str | None = None,
    g3_resolution_ref: str | None = None,
    tokenizer_asset_id: str = DEFAULT_TOKENIZER_ASSET_ID,
    data_asset_id: str = DEFAULT_DATA_ASSET_ID,
    delta_phase: str = "pre_sizing",
) -> tuple[dict[str, TaskRuntimeEnvironment], dict[str, str]]:
    """Derive six rereadable environments from one verified final snapshot.

    Each environment has exactly one sizing, registry and delta source under
    the keys consumed by the formal Stage2.04 runner.  The global environment
    is never reused as a launch target; it is only the already-verified base
    evidence snapshot from which these six identities are derived.
    """

    root = Path(data_root).resolve()
    if delta_phase not in {"pre_sizing", "post_sizing"}:
        raise _error("DELTA_PHASE_INVALID", delta_phase)
    base_ref = _source_ref(base_environment_ref, "base_environment")
    try:
        base = TaskRuntimeEnvironment.from_mapping(_load_mapping(root, base_ref, "base_environment"))
    except Exception as error:
        raise _error("BASE_ENVIRONMENT_INVALID") from error
    if base.frozen_contract_stages != frozenset({0, 1, 2}) or "stage2.G2.2" not in base.passed_gate_ids:
        raise _error("BASE_ENVIRONMENT_FORMAL_CHAIN_INVALID")
    sizing = _cell_ref_map(sizing_refs, field="sizing_refs", required=True, root=root)
    registries = _cell_ref_map(registry_refs, field="registry_refs", required=True, root=root)
    deltas = _cell_ref_map(delta_refs, field="delta_refs", required=True, root=root)
    evidence_ref = base.evidence_refs.get("formal_execution")
    if evidence_ref is None:
        raise _error("BASE_ENVIRONMENT_FORMAL_EXECUTION_MISSING")
    evidence = FormalExecutionEvidence.from_mapping(_load_mapping(root, evidence_ref, "formal_execution"))
    formal_external_lineage = all(
        item is not None
        for item in (
            configs,
            config_refs,
            six_cell_manifest_ref,
            s23_asset_task_ref,
            asset_manifest_ref,
            g3_resolution_ref,
        )
    )
    manifest: AssetResolutionManifest | None = None
    g3_assets: FormalG3RuntimeAssets | None = None
    if formal_external_lineage:
        assert asset_manifest_ref is not None and g3_resolution_ref is not None
        assert s23_asset_task_ref is not None
        try:
            s23_loaded = load_committed_task_artifact(root, _source_ref(s23_asset_task_ref, "s23_asset_task"), require_formal=True)
            if s23_loaded.identity.artifact_kind != "asset_resolution":
                raise ValueError("S2.3 asset task kind mismatch")
            manifest = _formal_s23_asset_manifest(s23_loaded.payload, field="s23_asset_task")
            g3_assets = FormalG3RuntimeAssets.load(root, _source_ref(g3_resolution_ref, "g3_resolution"))
        except Exception as error:
            raise _error("FORMAL_EXTERNAL_LINEAGE_INVALID") from error
        if tuple(config_refs or {}) != EXPECTED_CELL_IDS:
            raise _error("CONFIG_REF_SET_INVALID")
    environments: dict[str, TaskRuntimeEnvironment] = {}
    refs: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        sizing_value = _validate_sizing_ref(
            root,
            sizing[cell_id],
            formal_execution_hash=evidence.artifact_hash,
            cell_id=cell_id,
        )
        _validate_delta_sci_artifact(
            root,
            deltas[cell_id],
            cell_id=cell_id,
            allow_pre_sizing_plan=delta_phase == "pre_sizing",
        )
        registry_value = _load_formal_or_direct_payload(
            root,
            registries[cell_id],
            field="stage2_parameter_registry",
            artifact_kind="parameter_registry",
        )
        if registry_value.get("schema_version") != S204_REGISTRY_SCHEMA or registry_value.get("status") != "READY":
            raise _error("PARAMETER_REGISTRY_FORMAL_REQUIRED", cell_id)
        if registry_value.get("cell_id") != cell_id:
            raise _error("PARAMETER_REGISTRY_CELL_MISMATCH", cell_id)
        groups = registry_value.get("parameter_groups")
        if not isinstance(groups, Mapping) or not groups:
            raise _error("PARAMETER_REGISTRY_GROUPS_REQUIRED", cell_id)
        if (
            configs is not None
            and registry_value.get("config_hash") is not None
            and registry_value.get("config_hash") != configs[cell_id].config_hash
        ):
            raise _error("PARAMETER_REGISTRY_CONFIG_MISMATCH", cell_id)
        declared_registry_hash = registry_value.get("artifact_hash")
        if not isinstance(declared_registry_hash, str) or canonical_json_hash(
            {key: item for key, item in registry_value.items() if key != "artifact_hash"}
        ) != declared_registry_hash:
            raise _error("PARAMETER_REGISTRY_HASH_INVALID", cell_id)

        auxiliary: dict[str, str] = {}
        if formal_external_lineage:
            assert manifest is not None and g3_assets is not None
            assert configs is not None and config_refs is not None
            assert six_cell_manifest_ref is not None and s23_asset_task_ref is not None
            checkpoint = next(item for item in manifest.checkpoints if _cell_id(item) == cell_id)
            config = configs[cell_id]
            config_ref = _source_ref(config_refs[cell_id], f"config.{cell_id}")
            auxiliary["resolved_config"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="resolved_config",
                payload=config.to_dict(),
                config_hash=config.config_hash,
                source_refs=(config_ref,),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/resolved-config",
            )
            auxiliary["reference_sizing_plan"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="reference_sizing_plan",
                payload=sizing_value,
                config_hash=config.config_hash,
                source_refs=(sizing[cell_id],),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/sizing-plan",
            )
            auxiliary["parameter_registry"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="parameter_registry",
                payload=registry_value,
                config_hash=config.config_hash,
                source_refs=(registries[cell_id],),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/registry",
            )
            delta_value = _validate_delta_sci_artifact(
                root,
                deltas[cell_id],
                cell_id=cell_id,
                allow_pre_sizing_plan=delta_phase == "pre_sizing",
            )
            auxiliary["reference_delta_sci"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="reference_delta_sci",
                payload=delta_value,
                config_hash=config.config_hash,
                source_refs=(deltas[cell_id], *tuple(delta_value["source_contract_refs"])),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/delta-sci",
            )
            auxiliary["six_cell_manifest"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="six_cell_manifest",
                payload=_load_mapping(root, six_cell_manifest_ref, "six_cell_manifest"),
                config_hash=config.config_hash,
                source_refs=(six_cell_manifest_ref, asset_manifest_ref),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/six-cell-manifest",
            )
            checkpoint_payload = _load_mapping(root, checkpoint.manifest_ref, f"checkpoint.{cell_id}.manifest")
            auxiliary["checkpoint_manifest"] = _publish_auxiliary_task_artifact(
                root,
                artifact_kind="checkpoint_manifest",
                payload=checkpoint_payload,
                config_hash=config.config_hash,
                source_refs=(checkpoint.manifest_ref,),
                output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/checkpoint-manifest",
            )
            model_asset = g3_assets.resolve(f"{checkpoint.model_id}-step0", expected_kind="model")
            tokenizer_asset = g3_assets.resolve(tokenizer_asset_id, expected_kind="tokenizer")
            data_asset = g3_assets.resolve(data_asset_id, expected_kind="pile")
            for kind, asset, key in (
                ("model_manifest", model_asset, "model_manifest"),
                ("tokenizer_manifest", tokenizer_asset, "tokenizer_manifest"),
                ("data_manifest", data_asset, "data_manifest"),
            ):
                payload = _load_direct_manifest_payload(root, asset.manifest_ref, f"{cell_id}.{kind}")
                auxiliary[key] = _publish_auxiliary_task_artifact(
                    root,
                    artifact_kind=kind,
                    payload=payload,
                    config_hash=config.config_hash,
                    source_refs=(asset.manifest_ref, g3_resolution_ref),
                    output_dir=f"{output_dir}/auxiliary/{_cell_path_component(cell_id)}/{kind}",
                )
            auxiliary["s23_asset_resolution"] = _source_ref(s23_asset_task_ref, "s23_asset_task")
        evidence_refs = dict(base.evidence_refs)
        evidence_refs.pop("formal_reference_sizing_plan", None)
        evidence_refs.pop("stage2_parameter_registry", None)
        evidence_refs.pop("stage2_reference_delta_sci", None)
        evidence_refs["formal_reference_sizing_plan"] = auxiliary.get("reference_sizing_plan", sizing[cell_id])
        evidence_refs["stage2_parameter_registry"] = auxiliary.get("parameter_registry", registries[cell_id])
        evidence_refs["stage2_reference_delta_sci"] = auxiliary.get("reference_delta_sci", deltas[cell_id])
        if six_cell_manifest_ref is not None:
            evidence_refs["stage2_s204_six_cell_manifest"] = _source_ref(
                six_cell_manifest_ref, "six_cell_manifest"
            )
        if formal_external_lineage:
            evidence_refs.update(
                {
                    "stage2_s23_asset_resolution": auxiliary["s23_asset_resolution"],
                    "stage2_s23_six_cell_manifest": auxiliary["six_cell_manifest"],
                    "stage2_resolved_config": auxiliary["resolved_config"],
                    "stage2_checkpoint_manifest": auxiliary["checkpoint_manifest"],
                    "stage2_model_manifest": auxiliary["model_manifest"],
                    "stage2_data_manifest": auxiliary["data_manifest"],
                    "stage2_tokenizer_manifest": auxiliary["tokenizer_manifest"],
                    "stage2_parameter_registry": auxiliary["parameter_registry"],
                    "stage2_reference_delta_sci": auxiliary["reference_delta_sci"],
                    "stage2_reference_sizing_plan": auxiliary["reference_sizing_plan"],
                }
            )
        environment = TaskRuntimeEnvironment(
            capabilities=base.capabilities,
            frozen_contract_stages=frozenset({0, 1, 2}),
            passed_gate_ids=base.passed_gate_ids,
            estimator_decision_ref=base.estimator_decision_ref,
            evidence_refs=evidence_refs,
        )
        output_ref = f"{output_dir}/{_cell_path_component(cell_id)}.json"
        publish_canonical_immutable(_safe_relative(root, output_ref, "cell_environment_output"), environment.to_dict())
        reread = TaskRuntimeEnvironment.from_mapping(_load_mapping(root, output_ref, "cell_environment"))
        if reread.environment_hash != environment.environment_hash:
            raise _error("ENVIRONMENT_ROUND_TRIP_DRIFT", cell_id)
        environments[cell_id] = reread
        refs[cell_id] = PurePosixPath(output_ref).as_posix()
    if len({env.environment_hash for env in environments.values()}) != len(EXPECTED_CELL_IDS):
        raise _error("SIX_CELL_ENVIRONMENT_NOT_UNIQUE")
    return environments, refs


def _asset_id_from_root(root_ref: str) -> str:
    path = PurePosixPath(root_ref)
    if not path.parts:
        raise _error("CHECKPOINT_ROOT_INVALID")
    return path.parts[-1]


def _formal_s23_asset_manifest(
    payload: Mapping[str, Any], *, field: str
) -> AssetResolutionManifest:
    """Extract the S2.3 manifest from its task-output wrapper."""

    if payload.get("schema_version") != "stage2-task-asset-resolution-v1":
        raise _error("S23_ASSET_PAYLOAD_SCHEMA_INVALID", field)
    nested = payload.get("stage2_asset_manifest")
    if not isinstance(nested, Mapping):
        raise _error("S23_ASSET_MANIFEST_MISSING", field)
    try:
        manifest = AssetResolutionManifest.from_mapping(dict(nested))
        validate_formal_asset_identity(manifest)
    except Exception as error:
        raise _error("S23_ASSET_MANIFEST_INVALID", field) from error
    if payload.get("formal_eligible") is not False:
        raise _error("S23_ASSET_CANDIDATE_SCOPE_INVALID", field)
    return manifest


def _g3_resolved_file_inventory(
    asset: Any, *, field: str
) -> dict[str, tuple[int, str]]:
    """Return the exact file inventory verified by a qualified G3 asset.

    ``FormalG3RuntimeAssets.resolve`` has already performed the physical size
    and SHA-256 checks.  Consume that resolved inventory rather than treating
    ``ready_manifest_sha256`` as a file digest.
    """

    resolved = getattr(asset, "resolved", None)
    raw_files = getattr(resolved, "files", None)
    if not isinstance(raw_files, tuple) or not raw_files:
        raise _error("G3_MANIFEST_FILE_INVENTORY_INVALID", field)
    inventory: dict[str, tuple[int, str]] = {}
    for index, raw in enumerate(raw_files):
        normalized = getattr(raw, "relative_path", None)
        size_value = getattr(raw, "size_bytes", None)
        digest = getattr(raw, "sha256", None)
        try:
            normalized = _logical(normalized, f"{field}[{index}].path")
            digest = _sha(digest, f"{field}[{index}].sha256")
        except S204MaterializationError as error:
            raise _error("G3_MANIFEST_FILE_DESCRIPTOR_INVALID", f"{field}[{index}]") from error
        if isinstance(size_value, bool) or not isinstance(size_value, int) or size_value < 0:
            raise _error("G3_MANIFEST_FILE_DESCRIPTOR_INVALID", f"{field}[{index}].size_bytes")
        if normalized in inventory:
            raise _error("G3_MANIFEST_FILE_PATH_DUPLICATE", f"{field}:{normalized}")
        inventory[normalized] = (size_value, digest)
    return inventory


def _match_g3_file(
    inventory: Mapping[str, tuple[int, str]],
    path: str,
    expected: tuple[int, str],
    *,
    field: str,
) -> None:
    """Match by relative path, with a unique basename fallback for manifests."""

    observed = inventory.get(path)
    if observed is None:
        basename = PurePosixPath(path).name
        candidates = [value for key, value in inventory.items() if PurePosixPath(key).name == basename]
        if len(candidates) != 1:
            raise _error("G3_FILE_PATH_MISSING", field)
        observed = candidates[0]
    if observed != expected:
        raise _error("G3_FILE_HASH_MISMATCH", field)


def _checkpoint_inventory(
    checkpoint: CheckpointRecord, *, roles: set[str] | None = None
) -> dict[str, tuple[int, str]]:
    selected = checkpoint.files if roles is None else tuple(
        item for item in checkpoint.files if item.role in roles
    )
    return {item.path: (item.size_bytes, item.sha256) for item in selected}


def _load_checkpoint_tokenizer_identity(
    root: Path,
    checkpoint: CheckpointRecord,
    *,
    cell_id: str,
) -> tuple[str, int, int, str]:
    """Load one checkpoint tokenizer offline and return semantic identity."""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise _error("G3_TOKENIZER_RUNTIME_UNAVAILABLE", cell_id) from error
    tokenizer_root = _safe_relative(root, checkpoint.root_ref, f"checkpoint.{cell_id}.root_ref")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_root), local_files_only=True, trust_remote_code=False
        )
        vocab = tokenizer.get_vocab()
        if not isinstance(vocab, Mapping) or any(
            not isinstance(key, str) or type(value) is not int for key, value in vocab.items()
        ):
            raise TypeError("tokenizer vocabulary mapping is invalid")
        observed = (
            type(tokenizer).__name__,
            int(tokenizer.vocab_size),
            int(len(tokenizer)),
            canonical_json_hash(vocab),
        )
    except Exception as error:
        raise _error("G3_TOKENIZER_OFFLINE_LOAD_FAILED", cell_id) from error
    return observed


def _verify_checkpoint_tokenizer_files_on_disk(
    root: Path,
    checkpoint: CheckpointRecord,
    *,
    cell_id: str,
) -> None:
    """Retain S2.3 tokenizer file identity while permitting semantic equivalence."""

    tokenizer_files = [item for item in checkpoint.files if item.role == "tokenizer"]
    if len(tokenizer_files) != 1:
        raise _error("G3_TOKENIZER_FILE_REQUIRED", cell_id)
    for item in checkpoint.files:
        if item.role not in {"tokenizer", "tokenizer_model", "special_tokens", "tokenizer_config"}:
            continue
        path = _safe_relative(root, f"{checkpoint.root_ref}/{item.path}", f"checkpoint.{cell_id}.{item.path}")
        try:
            if path.stat().st_size != item.size_bytes:
                raise _error("G3_CHECKPOINT_FILE_SIZE_MISMATCH", f"{cell_id}:{item.path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        except S204MaterializationError:
            raise
        except (OSError, ValueError) as error:
            raise _error("G3_CHECKPOINT_FILE_UNREADABLE", f"{cell_id}:{item.path}") from error
        if digest.hexdigest() != item.sha256:
            raise _error("G3_CHECKPOINT_FILE_HASH_MISMATCH", f"{cell_id}:{item.path}")


def _validate_g3_checkpoint_binding(
    root: Path,
    *,
    checkpoint: CheckpointRecord,
    base_checkpoint: CheckpointRecord,
    model_asset: Any,
    tokenizer_asset: Any,
    cell_id: str,
) -> None:
    """Bind each S2.4 checkpoint to the G3-qualified base model safely.

    The G3 model entry is a step-0 asset.  Early/mid-late S2.3 rows therefore
    bind through their model's initialization row, not by pretending G3
    published those later checkpoint manifests.  Tokenizer file hashes are
    checked against the approved checkpoint root; the independently-versioned
    G3 tokenizer manifest remains a separately bound runtime identity.
    """

    if model_asset.asset_root_ref != base_checkpoint.root_ref:
        raise _error(
            "G3_MODEL_ROOT_MISMATCH",
            f"{cell_id}:{model_asset.asset_root_ref}!={base_checkpoint.root_ref}",
        )
    if model_asset.logical_name != _asset_id_from_root(base_checkpoint.root_ref):
        raise _error("G3_MODEL_ASSET_ID_MISMATCH", cell_id)
    if model_asset.resolved.revision != base_checkpoint.revision:
        raise _error(
            "G3_MODEL_REVISION_MISMATCH",
            f"{cell_id}:{model_asset.resolved.revision}!={base_checkpoint.revision}",
        )
    model_inventory = _g3_resolved_file_inventory(
        model_asset, field=f"g3.model.{model_asset.logical_name}.files"
    )
    expected_model = _checkpoint_inventory(base_checkpoint, roles={"config", "weights"})
    if len(model_inventory) != len(expected_model):
        raise _error("G3_MODEL_FILE_INVENTORY_MISMATCH", cell_id)
    for path, expected in expected_model.items():
        _match_g3_file(model_inventory, path, expected, field=f"{cell_id}:model:{path}")

    # The generic G3 tokenizer is deliberately not compared by ready-manifest
    # hash or by tokenizer.json bytes.  It is independently qualified and its
    # semantic metadata is the cross-version contract.  The exact S2.3
    # tokenizer files remain bound to every checkpoint root below.
    base_tokenizer = _checkpoint_inventory(
        base_checkpoint,
        roles={"tokenizer", "tokenizer_model", "special_tokens", "tokenizer_config"},
    )
    current_tokenizer = _checkpoint_inventory(
        checkpoint,
        roles={"tokenizer", "tokenizer_model", "special_tokens", "tokenizer_config"},
    )
    if not base_tokenizer or current_tokenizer != base_tokenizer:
        raise _error("G3_TOKENIZER_FILE_INVENTORY_MISMATCH", cell_id)
    tokenizer_inventory = _g3_resolved_file_inventory(
        tokenizer_asset, field=f"g3.tokenizer.{tokenizer_asset.logical_name}.files"
    )
    if not any(PurePosixPath(path).name == "tokenizer.json" for path in tokenizer_inventory):
        raise _error("G3_TOKENIZER_FILE_REQUIRED", cell_id)
    metadata = tokenizer_asset.manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise _error("G3_TOKENIZER_METADATA_INVALID", cell_id)
    expected_semantics = (
        metadata.get("tokenizer_class"),
        metadata.get("vocab_size"),
        metadata.get("token_count_with_added_tokens"),
        metadata.get("vocab_mapping_sha256"),
    )
    if (
        not isinstance(expected_semantics[0], str)
        or isinstance(expected_semantics[1], bool)
        or not isinstance(expected_semantics[1], int)
        or isinstance(expected_semantics[2], bool)
        or not isinstance(expected_semantics[2], int)
    ):
        raise _error("G3_TOKENIZER_METADATA_INVALID", cell_id)
    try:
        _sha(expected_semantics[3], "g3.tokenizer.vocab_mapping_sha256")
    except S204MaterializationError as error:
        raise _error("G3_TOKENIZER_METADATA_INVALID", cell_id) from error
    observed_semantics = _load_checkpoint_tokenizer_identity(root, checkpoint, cell_id=cell_id)
    if observed_semantics != expected_semantics:
        raise _error("G3_TOKENIZER_SEMANTIC_MISMATCH", cell_id)
    _verify_checkpoint_tokenizer_files_on_disk(root, checkpoint, cell_id=cell_id)

    # Force the separate tokenizer asset to be resolved and retained in the
    # caller's auxiliary evidence.  Its ready-manifest hash is intentionally
    # not compared with a checkpoint file SHA.
    if tokenizer_asset.kind != "tokenizer" or not tokenizer_asset.manifest_ref:
        raise _error("G3_TOKENIZER_ASSET_INVALID", cell_id)


def _validate_g3_data_binding(
    data: Any,
    data_asset: Any,
    *,
    cell_id: str,
) -> None:
    """Require exact revision and two-file S2.3/G3 pile inventory equality."""

    g3_revision = data_asset.resolved.revision
    if g3_revision != data.revision:
        raise _error("G3_DATA_REVISION_MISMATCH", f"{cell_id}:{g3_revision}!={data.revision}")
    expected = {item.path: (item.size_bytes, item.sha256) for item in data.files}
    observed = _g3_resolved_file_inventory(
        data_asset, field=f"g3.data.{data_asset.logical_name}.files"
    )
    if len(observed) != len(expected):
        raise _error("G3_DATA_FILE_INVENTORY_MISMATCH", cell_id)
    for path, descriptor in expected.items():
        _match_g3_file(observed, path, descriptor, field=f"{cell_id}:data:{path}")


def _validate_g3_s23_runtime_equivalence(
    root: Path,
    *,
    checkpoint: CheckpointRecord,
    base_checkpoint: CheckpointRecord,
    data: Any,
    model_asset: Any,
    tokenizer_asset: Any,
    data_asset: Any,
    cell_id: str,
) -> None:
    """Fail-closed S2.3↔G3 equivalence for one formal S2.4 cell."""

    _validate_g3_checkpoint_binding(
        root,
        checkpoint=checkpoint,
        base_checkpoint=base_checkpoint,
        model_asset=model_asset,
        tokenizer_asset=tokenizer_asset,
        cell_id=cell_id,
    )
    _validate_g3_data_binding(data, data_asset, cell_id=cell_id)


def _checkpoint_identity(
    checkpoint: CheckpointRecord,
    *,
    tokenizer_asset_id: str,
    data_asset_id: str,
    data_revision: str,
) -> dict[str, Any]:
    if not checkpoint.ready or checkpoint.revision is None:
        raise _error("CHECKPOINT_NOT_READY", checkpoint.checkpoint_id)
    root_asset_id = _asset_id_from_root(checkpoint.root_ref)
    expected_root_prefix = f"{checkpoint.model_id}-step{checkpoint.training_step}"
    if root_asset_id != expected_root_prefix and not root_asset_id.startswith(
        f"{expected_root_prefix}-"
    ):
        raise _error(
            "CHECKPOINT_MODEL_ASSET_ID_MISMATCH",
            f"{checkpoint.checkpoint_id}:{root_asset_id}!={expected_root_prefix}",
        )
    if checkpoint.manifest_sha256 is None or checkpoint.tokenizer_sha256 is None:
        raise _error("CHECKPOINT_MANIFEST_IDENTITY_MISSING", checkpoint.checkpoint_id)
    return {
        "model_asset_id": root_asset_id,
        "model_id": checkpoint.model_id,
        "training_stage": checkpoint.training_stage,
        "checkpoint_id": checkpoint.checkpoint_id,
        "model_revision": checkpoint.revision,
        "model_manifest_sha256": checkpoint.manifest_sha256,
        "parameter_registry_hash": checkpoint.parameter_registry_hash,
        "tokenizer_asset_id": tokenizer_asset_id,
        "tokenizer_manifest_sha256": checkpoint.tokenizer_sha256,
        "data_asset_id": data_asset_id,
        "data_revision": data_revision,
    }


def _cell_id(checkpoint: CheckpointRecord) -> str:
    """Return the one canonical G2.3 cell spelling.

    Colons are part of the contract (``model:stage``).  Filesystem names use
    :func:`_cell_path_component` below; callers must never silently replace the
    contract ID with a filename-safe variant.
    """

    return canonical_cell_id(checkpoint.model_id, checkpoint.training_stage)


def _cell_path_component(cell_id: str) -> str:
    try:
        return cell_path_component(cell_id)
    except ValueError as error:
        raise _error("CELL_ID_INVALID", cell_id) from error


def _expected_cell_checkpoints(
    manifest: AssetResolutionManifest,
) -> tuple[CheckpointRecord, ...]:
    """Require S2.3's six checkpoint rows in the fixed G2.3 order."""

    rows = tuple(manifest.checkpoints)
    ids = tuple(_cell_id(row) for row in rows)
    if ids != EXPECTED_CELL_IDS or len(set(ids)) != len(EXPECTED_CELL_IDS):
        raise _error("SIX_CELL_IDENTITY_NOT_UNIQUE", ids)
    if any(not row.ready for row in rows):
        raise _error("SIX_CELL_CHECKPOINT_NOT_READY")
    return rows


def _cell_ref_map(
    value: Mapping[str, Any] | None,
    *,
    field: str,
    required: bool,
    root: Path | None = None,
) -> dict[str, str]:
    """Normalize an explicit per-cell ref map without name-based discovery."""

    if value is None:
        if required:
            raise _error(f"{field.upper()}_REQUIRED")
        return {}
    if not isinstance(value, Mapping) or set(value) != set(EXPECTED_CELL_IDS):
        raise _error(f"{field.upper()}_SET_INVALID")
    result: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        raw = value[cell_id]
        if isinstance(raw, Mapping):
            if set(raw) - {"ref", "sha256"} or not isinstance(raw.get("ref"), str):
                raise _error(f"{field.upper()}_SPEC_INVALID", cell_id)
            declared_sha = raw.get("sha256")
            raw = raw["ref"]
            if declared_sha is not None:
                if root is None:
                    raise _error(f"{field.upper()}_HASH_ROOT_REQUIRED", cell_id)
                _assert_source_file(
                    root,
                    _source_ref(raw, f"{field}.{cell_id}"),
                    f"{field}.{cell_id}",
                    sha256=str(declared_sha),
                )
        result[cell_id] = _source_ref(raw, f"{field}.{cell_id}")
    if len(set(result.values())) != len(result):
        raise _error(f"{field.upper()}_MUST_BE_PER_CELL")
    return result


def _load_parameter_registry_artifact(
    root: Path,
    ref: str,
    *,
    cell_id: str,
    checkpoint: CheckpointRecord,
    config_hash: str | None = None,
    require_config_hash: bool = True,
) -> Mapping[str, Any]:
    """Load a producer-published registry and bind it to this checkpoint.

    The registry is intentionally a direct artifact rather than a TaskArtifact
    wrapper.  It must carry its own content hash and explicit model/checkpoint
    identity; deriving groups from a model name or parameter names is not a
    formal fallback.
    """

    value = _load_formal_or_direct_payload(
        root,
        ref,
        field="stage2_parameter_registry",
        artifact_kind="parameter_registry",
    )
    if value.get("schema_version") != S204_REGISTRY_SCHEMA or value.get("status") != "READY":
        raise _error("PARAMETER_REGISTRY_FORMAL_REQUIRED", cell_id)
    if value.get("scope", "formal") != "formal":
        raise _error("PARAMETER_REGISTRY_SCOPE_INVALID", cell_id)
    expected_hash = checkpoint.parameter_registry_hash
    if expected_hash is None or value.get("registry_hash") != expected_hash:
        raise _error("PARAMETER_REGISTRY_CHECKPOINT_MISMATCH", cell_id)
    groups = value.get("parameter_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise _error("PARAMETER_REGISTRY_GROUPS_REQUIRED", cell_id)
    for name, group in groups.items():
        if not isinstance(name, str) or not isinstance(group, Mapping):
            raise _error("PARAMETER_REGISTRY_GROUP_INVALID", cell_id)
        if not isinstance(group.get("layer"), str) or not group.get("layer"):
            raise _error("PARAMETER_REGISTRY_LAYER_REQUIRED", cell_id)
        if not isinstance(group.get("module"), str) or not group.get("module"):
            raise _error("PARAMETER_REGISTRY_MODULE_REQUIRED", cell_id)
    identity = value.get("identity")
    direct_identity = {
        "cell_id": value.get("cell_id"),
        "checkpoint_id": value.get("checkpoint_id"),
        "model_id": value.get("model_id"),
        "training_stage": value.get("training_stage"),
        "config_hash": value.get("config_hash"),
    }
    if isinstance(identity, Mapping):
        for key in direct_identity:
            if direct_identity[key] is None:
                direct_identity[key] = identity.get(key)
    expected_identity = {
        "cell_id": cell_id,
        "checkpoint_id": checkpoint.checkpoint_id,
        "model_id": checkpoint.model_id,
        "training_stage": checkpoint.training_stage,
    }
    for key, expected in expected_identity.items():
        if direct_identity.get(key) != expected:
            raise _error("PARAMETER_REGISTRY_IDENTITY_MISMATCH", f"{cell_id}:{key}")
    if config_hash is not None:
        declared_config = direct_identity.get("config_hash")
        if declared_config is not None and declared_config != config_hash:
            raise _error("PARAMETER_REGISTRY_CONFIG_MISMATCH", cell_id)
        if require_config_hash and declared_config is None:
            raise _error("PARAMETER_REGISTRY_CONFIG_REQUIRED", cell_id)
    declared = value.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    ) != declared:
        raise _error("PARAMETER_REGISTRY_HASH_INVALID", cell_id)
    return value


def publish_per_cell_delta_sci_plans(
    data_root: str | Path,
    *,
    s21_refs: Mapping[str, str],
    candidate_sample_counts: Sequence[int] = DEFAULT_CANDIDATES,
    cell_ids: Sequence[str] = EXPECTED_CELL_IDS,
    output_dir: str = "evidence/stage2/s204/reference-delta-sci-plans",
) -> dict[str, str]:
    """Publish pre-sizing formula plans without inventing numeric margins.

    Numeric ``delta_sci(B)`` values are intentionally absent.  The Stage 2.4
    runner derives the v2 artifact from committed sizing shards; this plan is
    only the immutable pre-sizing contract and candidate ladder.
    """

    root = Path(data_root).resolve()
    if tuple(cell_ids) != EXPECTED_CELL_IDS:
        raise _error("SIX_CELL_ID_SET_INVALID")
    required = ("preregistration", "hypothesis_contract")
    if set(s21_refs) != set(required):
        raise _error("REFERENCE_DELTA_SCI_S21_INPUTS_REQUIRED")
    contracts: list[tuple[str, LoadedTaskArtifact]] = []
    for kind in required:
        ref = _source_ref(s21_refs[kind], f"s21.{kind}")
        loaded = _load_formal_source(
            root,
            ref,
            task_id="stage2.01_scope_hypotheses_and_preregistration",
            artifact_kind=kind,
        )
        contracts.append((ref, loaded))
    preregistration = contracts[0][1].payload
    precision = preregistration.get("equivalence_and_precision")
    if not isinstance(precision, Mapping) or not precision:
        raise _error("REFERENCE_DELTA_SCI_FORMULA_CONTRACT_REQUIRED", "preregistration")
    preregistration_hash = preregistration.get("preregistration_hash")
    hypothesis_hash = contracts[1][1].payload.get("preregistration_hash")
    if not isinstance(preregistration_hash, str) or hypothesis_hash != preregistration_hash:
        raise _error("REFERENCE_DELTA_SCI_HYPOTHESIS_PREREGISTRATION_MISMATCH")
    formula_contract = dict(precision)
    formula_hash = canonical_json_hash(formula_contract)
    counts = tuple(int(item) for item in candidate_sample_counts)
    if tuple(sorted(set(counts))) != counts or any(item <= 0 for item in counts):
        raise _error("REFERENCE_DELTA_SCI_CANDIDATES_INVALID")
    refs: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        payload: dict[str, Any] = {
            "schema_version": S204_DELTA_PLAN_SCHEMA,
            "status": "READY",
            "scope": "formal",
            "phase": "pre_sizing",
            "cell_id": cell_id,
            "candidate_sample_counts": list(counts),
            "formula_contract": formula_contract,
            "formula_contract_hash": formula_hash,
            "source_contract_refs": [item[0] for item in contracts],
            "source_contract_artifact_hashes": [item[1].identity.artifact_hash for item in contracts],
            "numeric_delta_source": "stage2-reference-sizing-raw-shards-v2-after-sizing-commit",
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        ref = f"{output_dir}/{_cell_path_component(cell_id)}.json"
        publish_canonical_immutable(_safe_relative(root, ref, f"delta_sci_plan_output.{cell_id}"), payload)
        if _load_mapping(root, ref, f"delta_sci_plan.{cell_id}") != payload:
            raise _error("REFERENCE_DELTA_SCI_PLAN_ROUND_TRIP_DRIFT", cell_id)
        refs[cell_id] = PurePosixPath(ref).as_posix()
    return refs


def publish_formal_registry_artifacts(
    data_root: str | Path,
    *,
    asset_manifest_ref: str,
    g22_gate_ref: str,
    output_dir: str = "evidence/stage2/s204/parameter-registry",
) -> dict[str, str]:
    """Convert the validated S2.3/r6 registry source artifacts for S2.4.

    The S2.3 producer's ``stage2-parameter-registry-manifest-v1`` is the
    authoritative source.  This function only projects its already validated
    ``ParameterRegistry`` records into the narrower S2.4 artifact consumed by
    the reference runner; it never infers coordinates from model names.
    The conversion is deliberately not config-bound: the per-cell auxiliary
    TaskArtifact published by ``publish_per_cell_runtime_environments`` binds
    the resulting registry to the resolved config hash after config creation.
    """

    root = Path(data_root).resolve()
    asset_ref = _source_ref(asset_manifest_ref, "stage2_asset_resolution")
    try:
        asset_task = load_committed_task_artifact(root, asset_ref, require_formal=True)
        if asset_task.identity.task_id != "stage2.03_assets_checkpoints_and_sampling" or asset_task.identity.artifact_kind != "asset_resolution":
            raise ValueError("S2.3 asset TaskArtifact identity mismatch")
        manifest = _formal_s23_asset_manifest(asset_task.payload, field=asset_ref)
    except Exception as error:
        raise _error("S23_ASSET_COMMIT_IDENTITY_MISMATCH", asset_ref) from error

    gate_ref = _source_ref(g22_gate_ref, "g22_gate")
    try:
        gate_task = load_committed_task_artifact(root, gate_ref, require_formal=True)
        if gate_task.identity.task_id != "stage2.03_assets_checkpoints_and_sampling" or gate_task.identity.artifact_kind != "gate_record":
            raise ValueError("G2.2 adapter task identity mismatch")
        gate = GateRecord.from_mapping(dict(gate_task.payload))
    except Exception as error:
        raise _error("G22_FORMAL_COMMIT_REQUIRED", gate_ref) from error
    if gate.gate_id != "stage2.G2.2" or gate.status is not GateStatus.PASS:
        raise _error("G22_GATE_NOT_PASS", gate_ref)
    if not any(ref in set(gate.evidence_refs) for ref in (asset_ref, *asset_task.source_refs)):
        raise _error("G22_ASSET_UPSTREAM_BINDING_MISMATCH", gate_ref)

    # Reuse the reviewed r6 validator.  It checks the fixed amendment parent,
    # registry-index hash, all six source-manifest hashes, and every
    # checkpoint/registry cross-binding before this projection is written.
    try:
        from param_importance_nlp.experiments import stage2_g22_adapter as g22

        parent_value = _load_mapping(root, g22.ASSET_REF, "g22_parent_asset")
        parent = AssetResolutionManifest.from_mapping(parent_value)
        materialized, bindings, _ = g22._validate_amendment(root, parent)
        if materialized.digest != manifest.digest:
            raise ValueError("G2.2 amendment asset digest differs from S2.3 task")
        rows = g22._validate_formal_registry_index(root, materialized, bindings)
    except Exception as error:
        raise _error("FORMAL_R6_REGISTRY_VALIDATION_FAILED", type(error).__name__) from error

    by_cell = {str(row["cell_id"]): row for row in rows}
    if tuple(by_cell) != EXPECTED_CELL_IDS:
        raise _error("FORMAL_R6_REGISTRY_CELL_SET_INVALID")
    refs: dict[str, str] = {}
    for checkpoint in manifest.checkpoints:
        cell_id = _cell_id(checkpoint)
        source = by_cell.get(cell_id)
        if source is None:
            raise _error("FORMAL_R6_REGISTRY_CELL_MISSING", cell_id)
        source_ref = _source_ref(source["ref"], f"registry_source.{cell_id}")
        source_value = _load_mapping(root, source_ref, f"registry_source.{cell_id}")
        try:
            registry = ParameterRegistry.from_manifest(source_value["registry"])
        except Exception as error:
            raise _error("FORMAL_R6_REGISTRY_PAYLOAD_INVALID", cell_id) from error
        groups: dict[str, dict[str, str]] = {}
        for record in registry.eligible_records:
            layer = record.tags.get("layer")
            module = record.tags.get("module")
            if not isinstance(layer, str) or not layer or not isinstance(module, str) or not module:
                raise _error("FORMAL_R6_REGISTRY_GROUP_TAGS_REQUIRED", cell_id)
            groups[record.canonical_name] = {"layer": layer, "module": module}
        if set(groups) != set(registry.eligible_names):
            raise _error("FORMAL_R6_REGISTRY_ELIGIBLE_SET_INVALID", cell_id)
        payload: dict[str, Any] = {
            "schema_version": S204_REGISTRY_SCHEMA,
            "status": "READY",
            "scope": "formal",
            "cell_id": cell_id,
            "checkpoint_id": checkpoint.checkpoint_id,
            "model_id": checkpoint.model_id,
            "training_stage": checkpoint.training_stage,
            "registry_hash": str(source["registry_hash"]),
            "parameter_groups": groups,
            "source_s203_manifest_ref": source_ref,
            "source_s203_manifest_sha256": str(source["sha256"]),
            "source_asset_resolution_hash": manifest.digest,
            "source_g22_gate_ref": gate_ref,
            "source_g22_gate_artifact_hash": gate_task.identity.artifact_hash,
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        ref = f"{output_dir}/{_cell_path_component(cell_id)}.json"
        publish_canonical_immutable(_safe_relative(root, ref, f"registry_output.{cell_id}"), payload)
        if _load_mapping(root, ref, f"registry_output.{cell_id}") != payload:
            raise _error("FORMAL_R6_REGISTRY_ROUND_TRIP_DRIFT", cell_id)
        refs[cell_id] = PurePosixPath(ref).as_posix()
    if len(refs) != len(EXPECTED_CELL_IDS):
        raise _error("FORMAL_R6_REGISTRY_CELL_COUNT_INVALID")
    return refs


def _extract_frozen_delta_sci(
    root: Path,
    s21_refs: Mapping[str, str],
    candidate_counts: Sequence[int],
) -> tuple[dict[str, float], tuple[str, ...], tuple[str, ...]]:
    """Extract explicit numeric ``delta_sci(B)`` from the frozen S2.1 pair.

    The current S2.1 builder stores the scientific formula, not numeric
    margins.  In that case this function raises a stable missing-input error;
    it never evaluates the formula, uses a floor, or accepts a caller scalar.
    """

    required_order = ("preregistration", "hypothesis_contract")
    expected = set(required_order)
    if set(s21_refs) != expected and not expected.issubset(s21_refs):
        raise _error("REFERENCE_DELTA_SCI_S21_INPUTS_REQUIRED")
    loaded: dict[str, LoadedTaskArtifact] = {}
    for kind in required_order:
        loaded[kind] = _load_formal_source(
            root,
            _source_ref(s21_refs[kind], f"s21.{kind}"),
            task_id="stage2.01_scope_hypotheses_and_preregistration",
            artifact_kind=kind,
        )
    found: dict[str, tuple[dict[str, float], str]] = {}

    def visit(node: object, path: str) -> None:
        if isinstance(node, Mapping):
            candidate: object | None = None
            if "delta_sci_by_B" in node:
                candidate = node.get("delta_sci_by_B")
            elif isinstance(node.get("delta_sci"), Mapping):
                delta = node["delta_sci"]
                if isinstance(delta, Mapping):
                    candidate = delta.get("delta_sci_by_B", delta.get("by_B"))
            if candidate is not None:
                if not isinstance(candidate, Mapping):
                    raise _error("REFERENCE_DELTA_SCI_CONTRACT_INVALID", path)
                values: dict[str, float] = {}
                for count in candidate_counts:
                    raw = candidate.get(str(count), candidate.get(count))
                    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)) or float(raw) <= 0:
                        raise _error("REFERENCE_DELTA_SCI_CONTRACT_INVALID", f"{path}:{count}")
                    values[str(count)] = float(raw)
                found[canonical_json_hash(values)] = (values, path)
            for key, child in node.items():
                visit(child, f"{path}.{key}")
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
            for index, child in enumerate(node):
                visit(child, f"{path}[{index}]")

    for kind in required_order:
        item = loaded[kind]
        visit(item.payload, f"s21.{kind}")
    if len(found) != 1:
        raise _error("REFERENCE_DELTA_SCI_REQUIRED")
    values, _ = next(iter(found.values()))
    return values, tuple(item.identity.artifact_hash for item in loaded.values()), tuple(
        _source_ref(s21_refs[kind], f"s21.{kind}") for kind in ("preregistration", "hypothesis_contract")
    )


def publish_per_cell_delta_sci(
    data_root: str | Path,
    *,
    s21_refs: Mapping[str, str],
    candidate_sample_counts: Sequence[int] = DEFAULT_CANDIDATES,
    cell_ids: Sequence[str] = EXPECTED_CELL_IDS,
    output_dir: str = "evidence/stage2/s204/reference-delta-sci",
) -> dict[str, str]:
    """Publish one hash-bound delta source for each canonical G2.3 cell."""

    root = Path(data_root).resolve()
    if tuple(cell_ids) != EXPECTED_CELL_IDS:
        raise _error("SIX_CELL_ID_SET_INVALID")
    values, contract_hashes, contract_refs = _extract_frozen_delta_sci(
        root, s21_refs, candidate_sample_counts
    )
    refs: dict[str, str] = {}
    for cell_id in EXPECTED_CELL_IDS:
        payload: dict[str, Any] = {
            "schema_version": S204_DELTA_SCHEMA,
            "status": "READY",
            "scope": "formal",
            "cell_id": cell_id,
            "delta_sci_by_B": dict(values),
            "source_contract_refs": list(contract_refs),
            "source_contract_artifact_hashes": list(contract_hashes),
        }
        payload["artifact_hash"] = canonical_json_hash(payload)
        ref = f"{output_dir}/{_cell_path_component(cell_id)}.json"
        publish_canonical_immutable(_safe_relative(root, ref, "delta_sci_output"), payload)
        reread = _load_mapping(root, ref, "stage2_reference_delta_sci")
        if reread != payload:
            raise _error("REFERENCE_DELTA_SCI_ROUND_TRIP_DRIFT", cell_id)
        refs[cell_id] = PurePosixPath(ref).as_posix()
    return refs


def generate_six_cell_configs(
    data_root: str | Path,
    *,
    asset_manifest_ref: str,
    predecessor_refs: Mapping[str, Mapping[str, str]],
    base_config_ref: str = "configs/run-ready/layers/formal-stage2-estimator.yaml",
    g3_resolution_ref: str | None = None,
    output_dir: str = "configs/generated/stage2/s204",
    mode: str = "fresh",
    resume_refs: Mapping[str, str] | None = None,
    tokenizer_asset_id: str = DEFAULT_TOKENIZER_ASSET_ID,
    data_asset_id: str = DEFAULT_DATA_ASSET_ID,
    parameter_registry_refs: Mapping[str, str] | None = None,
    delta_sci_refs: Mapping[str, str] | None = None,
    delta_phase: str = "pre_sizing",
) -> dict[str, ResolvedConfigV2]:
    """Generate one unique, identity-bound v2 config per formal checkpoint.

    ``mode=fresh`` emits configs with ``recovery.resume_ref=null``.  ``mode=resume``
    requires an explicit per-cell logical committed boundary; it never guesses
    a checkpoint or reuses another cell's output.
    """

    if mode not in {"fresh", "resume"}:
        raise _error("CONFIG_MODE_INVALID", mode)
    if delta_phase not in {"pre_sizing", "post_sizing"}:
        raise _error("DELTA_PHASE_INVALID", delta_phase)
    root = Path(data_root).resolve()
    output_root_ref = _logical(output_dir, "config_output_dir")
    manifest_ref = _source_ref(asset_manifest_ref, "stage2_asset_resolution")
    loaded = load_committed_task_artifact(root, manifest_ref, require_formal=True)
    if loaded.identity.task_id != "stage2.03_assets_checkpoints_and_sampling" or loaded.identity.artifact_kind != "asset_resolution":
        raise _error("S23_ASSET_COMMIT_IDENTITY_MISMATCH", manifest_ref)
    manifest = _formal_s23_asset_manifest(loaded.payload, field=manifest_ref)
    g3_assets: FormalG3RuntimeAssets | None = None
    if g3_resolution_ref is not None:
        g3_ref = _source_ref(g3_resolution_ref, "g3_resolution")
        try:
            g3_commit = load_committed_task_artifact(root, g3_ref, require_formal=True)
            if g3_commit.identity.artifact_kind != "asset_resolution":
                raise ValueError("wrong G3 artifact kind")
            g3_assets = FormalG3RuntimeAssets.load(root, g3_ref)
        except Exception as error:
            raise _error("G3_RUNTIME_ASSETS_INVALID", g3_ref) from error
    if set(predecessor_refs) != set(TASK_INPUTS):
        raise _error("PREDECESSOR_TASK_SET_INVALID")
    direct: list[str] = []
    for task_id in (
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    ):
        expected = TASK_INPUTS[task_id]
        refs = predecessor_refs[task_id]
        if set(refs) != set(expected):
            raise _error("PREDECESSOR_ARTIFACT_SET_INVALID", task_id)
        for kind in expected:
            ref = _source_ref(refs[kind], f"predecessor.{task_id}.{kind}")
            item = _load_formal_source(root, ref, task_id=task_id, artifact_kind=kind)
            direct.append(ref)
    # S2.1 is required as a formal materialization input, but is not a direct
    # S2.4 DAG predecessor.  It is checked here to prevent a partial upstream
    # set from being silently hidden behind S2.2/S2.3 references.
    s201 = predecessor_refs["stage2.01_scope_hypotheses_and_preregistration"]
    if set(s201) != set(TASK_INPUTS["stage2.01_scope_hypotheses_and_preregistration"]):
        raise _error("S21_ARTIFACT_SET_INVALID")
    for kind in TASK_INPUTS["stage2.01_scope_hypotheses_and_preregistration"]:
        _load_formal_source(
            root,
            _source_ref(s201[kind], f"predecessor.stage2.01.{kind}"),
            task_id="stage2.01_scope_hypotheses_and_preregistration",
            artifact_kind=kind,
        )

    # The real S2.4 producer must expose one authoritative registry and one
    # frozen delta source per canonical cell.  Local fixture config tests do
    # not pass G3 and remain intentionally limited to config identity checks.
    formal_cell_inputs = g3_assets is not None
    registry_map = _cell_ref_map(
        parameter_registry_refs,
        field="parameter_registry_refs",
        required=formal_cell_inputs,
        root=root,
    )
    if delta_sci_refs is None and formal_cell_inputs:
        if delta_phase == "pre_sizing":
            delta_map = publish_per_cell_delta_sci_plans(
                root,
                s21_refs=s201,
                candidate_sample_counts=DEFAULT_CANDIDATES,
                output_dir=f"{output_root_ref}/reference-delta-sci-plans",
            )
        else:
            raise _error("POST_SIZING_DELTA_REF_REQUIRED")
    else:
        delta_map = _cell_ref_map(
            delta_sci_refs,
            field="delta_sci_refs",
            required=formal_cell_inputs,
            root=root,
        )

    base_path = _config_path(root, base_config_ref)
    try:
        from param_importance_nlp.cli import _load_mapping as cli_load_mapping

        # The checked-in run-ready layer is intentionally a partial science
        # layer.  Resolve it through the same defaults+layer path as the CLI;
        # fixture defaults provide schema defaults only and are never used as a
        # formal asset/provider identity.
        defaults_path = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
        base = ResolvedConfig.resolve(
            cli_load_mapping(defaults_path),
            cli_load_mapping(base_path),
        )
    except Exception as error:
        raise _error("BASE_CONFIG_INVALID", base_config_ref) from error
    if base.section("identity").get("run_intent") != "formal":
        raise _error("BASE_CONFIG_FORMAL_REQUIRED")
    data = manifest.data_range
    checkpoints = _expected_cell_checkpoints(manifest)
    base_checkpoints = {
        item.model_id: item
        for item in checkpoints
        if item.training_step == 0 and item.training_stage == "initialization"
    }
    if set(base_checkpoints) != {item.model_id for item in checkpoints}:
        raise _error("G3_BASE_CHECKPOINT_SET_INVALID")
    expected_cell_ids = set(EXPECTED_CELL_IDS)
    if mode == "resume":
        if resume_refs is None:
            raise _error("RESUME_REF_REQUIRED")
        if set(resume_refs) != expected_cell_ids:
            raise _error("RESUME_REF_SET_INVALID")
    elif resume_refs:
        raise _error("FRESH_CONFIG_CANNOT_CARRY_RESUME_REF")
    cells: dict[str, ResolvedConfigV2] = {}
    for checkpoint in checkpoints:
        identity = _checkpoint_identity(
            checkpoint,
            tokenizer_asset_id=tokenizer_asset_id,
            data_asset_id=data_asset_id,
            data_revision=data.revision,
        )
        cell_id = _cell_id(checkpoint)
        if g3_assets is not None:
            try:
                # G3 qualifies only the base model.  Checkpoint step manifests
                # are authoritative S2.3 outputs and must never be looked up as
                # if G3 had published step-1000/71000 entries.
                base_model_id = f"{checkpoint.model_id}-step0"
                model_asset = g3_assets.resolve(base_model_id, expected_kind="model")
                tokenizer_asset = g3_assets.resolve(tokenizer_asset_id, expected_kind="tokenizer")
                data_asset = g3_assets.resolve(data_asset_id, expected_kind="pile")
            except Exception as error:
                raise _error("G3_CONFIG_ASSET_ID_UNRESOLVED", cell_id) from error
            _validate_g3_s23_runtime_equivalence(
                root,
                checkpoint=checkpoint,
                base_checkpoint=base_checkpoints[checkpoint.model_id],
                model_asset=model_asset,
                tokenizer_asset=tokenizer_asset,
                data=data,
                data_asset=data_asset,
                cell_id=cell_id,
            )
            checkpoint_root = _safe_relative(root, checkpoint.root_ref, f"checkpoint.{cell_id}.root_ref")
            if not checkpoint_root.is_dir():
                raise _error("CHECKPOINT_ROOT_MISSING", cell_id)
            checkpoint_manifest = _load_mapping(root, checkpoint.manifest_ref, f"checkpoint.{cell_id}.manifest")
            if canonical_json_hash(dict(checkpoint_manifest)) != checkpoint.manifest_sha256:
                raise _error("CHECKPOINT_MANIFEST_HASH_MISMATCH", cell_id)
            _load_raw_json_mapping(
                root,
                data.manifest_ref,
                "data.manifest",
                expected_sha256=data.manifest_sha256,
            )
        if cell_id in cells:
            raise _error("CELL_ID_DUPLICATE", cell_id)
        if mode == "resume":
            assert resume_refs is not None
            resume_ref = _logical(resume_refs[cell_id], f"resume_ref.{cell_id}")
            if not resume_ref.startswith(f"{output_root_ref}/{_cell_path_component(cell_id)}/"):
                raise _error("RESUME_REF_CELL_MISMATCH", cell_id)
        else:
            resume_ref = None
        base_value = base.to_dict()
        model = base_value["model"]
        model.update(
            {
                "asset_id": identity["model_asset_id"],
                "revision": identity["model_revision"],
                "initialization_id": identity["checkpoint_id"],
                "architecture": identity["model_id"],
                "tokenizer_asset_id": tokenizer_asset_id,
            }
        )
        data_value = base_value["data"]
        data_value.update(
            {
                "asset_id": data_asset_id,
                "revision": data.revision,
                "sequence_length": data.input_sequence_length,
            }
        )
        # G2.3 derives checkpoint identity from this field.  Leaving it null
        # makes a config look unique while severing the S2.3 checkpoint link.
        base_value["identity"]["input_checkpoint_id"] = checkpoint.checkpoint_id
        base_value["identity"]["task"] = S204_TASK_ID
        base_value["identity"]["stage"] = 2
        base_value["identity"]["run_intent"] = "formal"
        base_value["identity"]["formal_eligible"] = True
        output_ref = f"{output_root_ref}/{_cell_path_component(cell_id)}"
        overrides: dict[str, Any] = {
            "providers": {
                "kind": "offline_hf",
                "task_type": "causal_lm",
                "task_name": "pile",
                "num_labels": None,
                "local_files_only": True,
                "trust_remote_code": False,
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            },
            "orchestration": {"input_result_refs": direct},
            "artifacts": {
                "output_dir": output_ref,
                "required_kinds": ["reference_result", "reference_convergence_report", "gate_record"],
                "publish_partial": False,
            },
            "recovery": {
                "mode": "resume_shards",
                "resume_ref": resume_ref,
                "safe_boundary": "shard_commit",
                "max_restarts": 0,
            },
            "execution": {"runner_kind": "reference", "dry_run": False, "fail_on_blocked": True},
        }
        config = ResolvedConfigV2.resolve(base_value, task_id=S204_TASK_ID, overrides=overrides)
        if formal_cell_inputs:
            _load_parameter_registry_artifact(
                root,
                registry_map[cell_id],
                cell_id=cell_id,
                checkpoint=checkpoint,
                config_hash=config.config_hash,
                require_config_hash=False,
            )
            _validate_delta_sci_artifact(
                root,
                delta_map[cell_id],
                cell_id=cell_id,
                allow_pre_sizing_plan=delta_phase == "pre_sizing",
            )
        cells[cell_id] = config
    if len(cells) != 6 or tuple(cells) != EXPECTED_CELL_IDS or len({item.config_hash for item in cells.values()}) != 6:
        raise _error("SIX_CELL_IDENTITY_NOT_UNIQUE")
    return cells


def write_six_cell_configs(
    configs: Mapping[str, ResolvedConfigV2],
    data_root: str | Path,
    *,
    output_dir: str = "configs/generated/stage2/s204",
    mode: str = "fresh",
) -> dict[str, str]:
    """Publish the generated configs as immutable canonical JSON objects."""

    root = Path(data_root).resolve()
    refs: dict[str, str] = {}
    for cell_id, config in configs.items():
        target_ref = f"{output_dir}/{mode}/{_cell_path_component(cell_id)}.json"
        target = _safe_relative(root, target_ref, "config_output")
        publish_canonical_immutable(target, config.to_dict())
        reread = ResolvedConfigV2.from_mapping(load_canonical_json(target))
        if reread.full_hash != config.full_hash:
            raise _error("CONFIG_ROUND_TRIP_DRIFT", cell_id)
        refs[cell_id] = PurePosixPath(target_ref).as_posix()
    return refs


def publish_six_cell_manifest(
    data_root: str | Path,
    *,
    asset_manifest_ref: str,
    configs: Mapping[str, ResolvedConfigV2],
    registry_refs: Mapping[str, str],
    output_ref: str = "evidence/stage2/s204/six-cell-manifest.json",
) -> str:
    """Publish the S2.3 checkpoint-root projection consumed by G2.3.

    The projection copies no checkpoint path or model identity from G3.  It
    binds each row to the exact S2.3 checkpoint manifest and the resolved v2
    config generated for that row, while retaining a distinct registry source
    for every cell.
    """

    root = Path(data_root).resolve()
    asset_ref = _source_ref(asset_manifest_ref, "stage2_asset_resolution")
    loaded = load_committed_task_artifact(root, asset_ref, require_formal=True)
    manifest = _formal_s23_asset_manifest(loaded.payload, field=asset_ref)
    checkpoints = _expected_cell_checkpoints(manifest)
    if tuple(configs) != EXPECTED_CELL_IDS:
        raise _error("SIX_CELL_CONFIG_ORDER_INVALID")
    refs = _cell_ref_map(registry_refs, field="registry_refs", required=True, root=root)
    rows: list[dict[str, Any]] = []
    registry_hashes: dict[str, str] = {}
    for checkpoint in checkpoints:
        cell_id = _cell_id(checkpoint)
        config = configs[cell_id]
        registry = _load_parameter_registry_artifact(
            root,
            refs[cell_id],
            cell_id=cell_id,
            checkpoint=checkpoint,
            config_hash=config.config_hash,
            require_config_hash=False,
        )
        registry_hash = _sha(registry.get("registry_hash"), f"registry.{cell_id}")
        registry_hashes[cell_id] = registry_hash
        rows.append(
            {
                "cell_id": cell_id,
                "model_id": checkpoint.model_id,
                "training_stage": checkpoint.training_stage,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_hash": _sha(checkpoint.manifest_sha256, f"checkpoint.{cell_id}.manifest_sha256"),
                "checkpoint_revision": checkpoint.revision,
                "registry_hash": registry_hash,
                "config_hash": config.config_hash,
                "checkpoint_root_ref": checkpoint.root_ref,
                "checkpoint_manifest_ref": checkpoint.manifest_ref,
                "parameter_registry_ref": refs[cell_id],
            }
        )
    body: dict[str, Any] = {
        "schema_version": S204_SIX_CELL_SCHEMA,
        "status": "READY",
        "scope": "formal",
        "asset_resolution_hash": manifest.digest,
        "asset_producer_commit": manifest.producer_commit,
        "asset_execution_commit": manifest.execution_commit,
        "checkpoints": rows,
        "data": manifest.data_range.to_dict(),
        "data_range_hash": manifest.data_range.digest,
        # Keep both the legacy common hash and the explicit per-cell map.  A
        # producer with model-specific registries must not collapse them into a
        # guessed shared value; consumers can require the map when present.
        "registry_hash": (
            next(iter(set(registry_hashes.values())))
            if len(set(registry_hashes.values())) == 1
            else canonical_json_hash(registry_hashes)
        ),
        "registry_hashes_by_cell": registry_hashes,
    }
    body["manifest_hash"] = canonical_json_hash(body)
    publish_canonical_immutable(_safe_relative(root, output_ref, "six_cell_manifest_output"), body)
    reread = _load_mapping(root, output_ref, "six_cell_manifest")
    if reread != body:
        raise _error("SIX_CELL_MANIFEST_ROUND_TRIP_DRIFT")
    return PurePosixPath(output_ref).as_posix()


def publish_six_cell_materialization_index(
    data_root: str | Path,
    *,
    config_refs: Mapping[str, str],
    environment_refs: Mapping[str, str],
    sizing_refs: Mapping[str, str],
    registry_refs: Mapping[str, str],
    delta_refs: Mapping[str, str],
    six_cell_manifest_ref: str,
    output_ref: str,
    mode: str,
) -> str:
    """Write the explicit per-cell handoff index used by launch tooling."""

    root = Path(data_root).resolve()
    maps = {
        "config_ref": _cell_ref_map(config_refs, field="config_refs", required=True, root=root),
        "environment_ref": _cell_ref_map(environment_refs, field="environment_refs", required=True, root=root),
        "sizing_ref": _cell_ref_map(sizing_refs, field="sizing_refs", required=True, root=root),
        "registry_ref": _cell_ref_map(registry_refs, field="registry_refs", required=True, root=root),
        "delta_ref": _cell_ref_map(delta_refs, field="delta_refs", required=True, root=root),
    }
    manifest_ref = _source_ref(six_cell_manifest_ref, "six_cell_manifest")
    rows = [
        {"cell_id": cell_id, **{key: value[cell_id] for key, value in maps.items()}}
        for cell_id in EXPECTED_CELL_IDS
    ]
    payload: dict[str, Any] = {
        "schema_version": "stage2-s204-six-cell-materialization-index-v1",
        "scope": "formal",
        "mode": mode,
        "six_cell_manifest_ref": manifest_ref,
        "cells": rows,
    }
    payload["index_hash"] = canonical_json_hash(payload)
    publish_canonical_immutable(_safe_relative(root, output_ref, "six_cell_index_output"), payload)
    if _load_mapping(root, output_ref, "six_cell_index") != payload:
        raise _error("SIX_CELL_INDEX_ROUND_TRIP_DRIFT")
    return PurePosixPath(output_ref).as_posix()


def _parse_key_refs(raw: Sequence[str], field: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise _error("KEY_REF_INVALID", field)
        key, value = item.split("=", 1)
        if not key or key in result:
            raise _error("KEY_REF_DUPLICATE", field)
        result[key] = value
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True, help="canonical source manifest JSON")
    parser.add_argument("--base-config", default="configs/run-ready/layers/formal-stage2-estimator.yaml")
    parser.add_argument("--output-dir", default=S204_S22_CANONICAL_OUTPUT_DIR)
    parser.add_argument("--config-output-dir", default="configs/generated/stage2/s204")
    parser.add_argument("--mode", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--resume-ref", action="append", default=[], metavar="CELL=REF")
    parser.add_argument("--environment-output", default="evidence/stage2/s204/runtime-environment.json")
    parser.add_argument("--sizing-output", default="evidence/stage2/s204/reference-sizing-plan.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.data_root.resolve()
        source = dict(_load_source_manifest(root, args.sources))
        # Caller-supplied task outputs and raw/candidate envelopes are never
        # accepted as formal input.  The canonical producer below is the only
        # path that emits S2.1/S2.3 commits.
        if source.get("task_outputs") is not None:
            raise _error("CALLER_FORMAL_TASK_OUTPUTS_FORBIDDEN")
        if source.get("raw_task_outputs") is not None:
            raise _error("CANDIDATE_INPUTS_FORBIDDEN")
        formal_ref = _source_ref(source["formal_execution"], "sources.formal_execution")
        evidence, evidence_ref = _publish_formal_execution(
            root,
            formal_ref,
            destination=str(source.get("materialized_formal_execution", "evidence/stage2/s204/formal-execution.json")),
        )
        materialized = execute_formal_predecessor_dag(
            root,
            source=source,
            raw_refs=None,
            formal_execution_ref=evidence_ref,
            base_config_ref=args.base_config,
            output_dir=args.output_dir,
        )
        if not isinstance(materialized, FormalDAGResult):
            raise _error("FORMAL_DAG_RESULT_REQUIRED")
        evidence = materialized.final_evidence
        evidence_ref = materialized.final_evidence_ref
        summary = {
            "schema_version": "stage2-s204-materialization-summary-v1",
            "formal_eligible": True,
            "task_output_refs": materialized,
            "adapter_gate_refs": materialized.bridge_gate_refs,
            "stage1_10_task_output_refs": materialized.stage1_10_refs,
            "stage1_11_task_output_refs": materialized.stage1_11_refs,
            "formal_execution_ref": evidence_ref,
            "asset_manifest_ref": materialized.authoritative_asset_ref,
            "mode": args.mode,
        }
        summary_ref = f"{args.output_dir}/materialization-summary.json"
        publish_canonical_immutable(_safe_relative(root, summary_ref, "summary_output"), summary)
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except KeyError as error:
        print(f"S2.4 materialization blocked: S204_SOURCE_FIELD_REQUIRED:{error.args[0]}", file=sys.stderr)
        return 3
    except (OSError, TypeError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"S2.4 materialization blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXCLUDED_PCI",
    "EXCLUDED_UUID",
    "EXPECTED_CELL_IDS",
    "FormalDAGResult",
    "S204MaterializationError",
    "S204_S22_CANONICAL_OUTPUT_DIR",
    "S204_S22_CONTROL_OUTPUT_DIR",
    "S204_S22_COMMIT_OUTPUT_DIR",
    "S204_S22_CONFIG_REF",
    "S22_G3_FORMAL_ENVIRONMENT_REF",
    "S22_G3_FORMAL_EXECUTION_G20_REF",
    "S22_G3_FORMAL_EXECUTION_G21_REF",
    "STAGE1_10_TASK_ID",
    "STAGE1_10_TASK_INPUTS",
    "STAGE1_TASK_ID",
    "STAGE1_TASK_INPUTS",
    "TASK_INPUTS",
    "build_formal_runtime_environment",
    "bootstrap_formal_task_inputs",
    "execute_formal_predecessor_dag",
    "ensure_formal_s22_task_outputs",
    "generate_six_cell_configs",
    "main",
    "materialize_formal_task_inputs",
    "publish_per_cell_delta_sci",
    "publish_per_cell_delta_sci_plans",
    "publish_formal_registry_artifacts",
    "publish_per_cell_runtime_environments",
    "publish_per_cell_sizing_plans",
    "publish_reference_sizing_plan",
    "publish_six_cell_manifest",
    "publish_six_cell_materialization_index",
    "produce_formal_s22_task_outputs",
    "write_six_cell_configs",
]
