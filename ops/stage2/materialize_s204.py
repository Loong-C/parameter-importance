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
import json
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
    validate_stage1_exit_evidence,
)
from param_importance_nlp.contracts.g21_formal_handoff import (  # noqa: E402
    ALLOWED_DEVICES as G21_ALLOWED_DEVICES,
    EXCLUDED_PCI as G21_EXCLUDED_PCI,
    load_g21_formal_handoff,
)
from param_importance_nlp.experiments.stage2_assets import (  # noqa: E402
    AssetResolutionManifest,
    CheckpointRecord,
    validate_formal_asset_identity,
)
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
EXCLUDED_GPU_INDEX: Final = "1"
EXCLUDED_PCI: Final = "0000:50:00.0"
EXCLUDED_UUID: Final = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
ALLOWED_GPU_INDICES: Final = frozenset({"0", "2", "3", "4"})
DEFAULT_CANDIDATES: Final = (512, 1024, 2048, 4096)
DEFAULT_BLOCK_SIZE: Final = 32
DEFAULT_TOKENIZER_ASSET_ID: Final = "pythia-tokenizer"
DEFAULT_DATA_ASSET_ID: Final = "pile-selected-prefix"
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
    if parts.intersection({"fixtures", "src", "ops", "configs"}):
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
    """Import raw manifests as explicit nonformal candidates.

    This helper never creates a formal predecessor.  It is used by the DAG
    bootstrap path to retain auditable raw inputs while the existing formal
    TaskRuntime runners recompute and publish authoritative commits.
    """

    root = Path(data_root).resolve()
    if set(raw_refs) != set(TASK_INPUTS):
        raise _error("RAW_TASK_SET_INVALID")
    loaded: dict[str, dict[str, tuple[Mapping[str, Any], str, tuple[str, ...]]]] = {}
    for task_id in TASK_INPUTS:
        specs = _raw_task_specs(raw_refs[task_id], task_id=task_id)
        values: dict[str, tuple[Mapping[str, Any], str, tuple[str, ...]]] = {}
        for kind in TASK_INPUTS[task_id]:
            spec = specs[kind]
            payload = _load_mapping(root, spec["ref"], f"raw.{task_id}.{kind}")
            _validate_raw_payload(task_id, kind, payload)
            values[kind] = (payload, spec["config_hash"], tuple(spec["source_refs"]))
        if len({item[1] for item in values.values()}) != 1:
            raise _error("RAW_TASK_CONFIG_IDENTITY_MISMATCH", task_id)
        loaded[task_id] = values

    result: dict[str, dict[str, str]] = {}
    stores = {
        task_id: TaskArtifactStore(root, f"{output_dir}/candidates/{task_id.replace('.', '-')}")
        for task_id in TASK_INPUTS
    }
    for task_id in TASK_INPUTS:
        task_result: dict[str, str] = {}
        for kind in TASK_INPUTS[task_id]:
            payload, config_hash, source_refs = loaded[task_id][kind]
            published = stores[task_id].publish(
                task_id=task_id,
                artifact_kind=kind,
                config_hash=config_hash,
                # Raw manifests are scientific candidates only.  The formal
                # envelope is reserved for the TaskRuntime DAG below, which
                # executes the existing S2.1 -> S2.2/S2.3 runners.
                run_intent="local_fixture",
                payload=payload,
                formal_eligible=False,
                source_refs=source_refs,
            )
            reread = load_committed_task_artifact(root, published.commit_ref, require_formal=False)
            if reread.payload != payload:
                raise _error("RAW_BOOTSTRAP_PAYLOAD_DRIFT", f"{task_id}.{kind}")
            task_result[kind] = published.commit_ref
        result[task_id] = task_result
    return result


def _publish_candidate_asset_manifest(
    root: Path,
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_dir: str,
) -> str:
    """Expose the raw S2.3 asset candidate to the formal asset runner."""

    task_id = "stage2.03_assets_checkpoints_and_sampling"
    try:
        spec = raw_refs[task_id]["asset_resolution"]
    except (KeyError, TypeError) as error:
        raise _error("RAW_ASSET_CANDIDATE_REQUIRED") from error
    if not isinstance(spec, Mapping) or not isinstance(spec.get("ref"), str):
        raise _error("RAW_ASSET_CANDIDATE_REQUIRED")
    raw_ref = _source_ref(spec["ref"], "raw.stage2.03.asset_resolution.ref")
    payload = _load_mapping(root, raw_ref, "raw.stage2.03.asset_resolution")
    manifest = _formal_s23_asset_manifest(payload, field=raw_ref)
    target_ref = f"{output_dir}/candidates/stage2-asset-resolution-manifest.json"
    target = _safe_relative(root, target_ref, "raw_asset_candidate_output")
    publish_canonical_immutable(target, manifest.to_dict())
    try:
        reread = AssetResolutionManifest.from_mapping(load_canonical_json(target))
        validate_formal_asset_identity(reread)
    except Exception as error:
        raise _error("RAW_ASSET_CANDIDATE_ROUND_TRIP_DRIFT", target_ref) from error
    return PurePosixPath(target_ref).as_posix()


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
    )
    gates = _load_formal_gate_refs(
        root,
        _mapping(source.get("gate_refs"), "gate_refs"),
        required=("stage1.G1-EXIT", "stage2.G2.0", "stage2.G2.1"),
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


def execute_formal_predecessor_dag(
    data_root: str | Path,
    *,
    source: Mapping[str, Any],
    raw_refs: Mapping[str, Mapping[str, Mapping[str, Any]]],
    formal_execution_ref: str,
    base_config_ref: str,
    output_dir: str,
) -> dict[str, dict[str, str]]:
    """Run the existing formal S2.1 -> S2.2/S2.3 TaskRuntime DAG.

    Raw payloads remain candidate evidence; only these formal runner outputs are
    returned as S2.4 predecessors.
    """

    root = Path(data_root).resolve()
    candidate_asset_ref = _publish_candidate_asset_manifest(root, raw_refs, output_dir=output_dir)
    environment = _build_formal_predecessor_environment(
        root,
        formal_execution_ref=formal_execution_ref,
        source=source,
        candidate_asset_ref=candidate_asset_ref,
    )
    runtime = TaskRuntime(workspace_root=root)
    register_stage23_runners(runtime, root)
    refs_by_task: dict[str, dict[str, str]] = {}
    task_order = (
        "stage2.01_scope_hypotheses_and_preregistration",
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    )
    for task_id in task_order:
        if task_id == "stage2.01_scope_hypotheses_and_preregistration":
            input_refs: tuple[str, ...] = ()
        else:
            input_refs = tuple(refs_by_task["stage2.01_scope_hypotheses_and_preregistration"].values())
        config = _formal_dag_config(
            root,
            base_config_ref=base_config_ref,
            task_id=task_id,
            input_refs=input_refs,
            output_dir=f"{output_dir}/formal-dag/{task_id.replace('.', '-')}",
        )
        result = runtime.execute(config, environment=environment)
        if result.status is not TaskRunStatus.PASS or not result.formal_eligible:
            raise _error("FORMAL_DAG_TASK_NOT_PASS", f"{task_id}:{result.status.value}")
        refs_by_task[task_id] = dict(result.artifact_refs)
    return refs_by_task


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


def _load_gpu_health_identity(
    root: Path,
    g21_handoff_ref: str | None,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Validate G2.1 plus its raw smoke report as one PCI+UUID authority."""

    if g21_handoff_ref is None:
        raise _error("GPU_HEALTH_EVIDENCE_REQUIRED")
    gpu_ref = _source_ref(g21_handoff_ref, "g21_handoff")
    try:
        gpu_handoff = load_g21_formal_handoff(
            _safe_relative(root, gpu_ref, "g21_handoff"),
            data_root=root,
        )
    except Exception as error:
        raise _error("GPU_HEALTH_EVIDENCE_INVALID", gpu_ref) from error
    smoke = gpu_handoff.get("current_gpu_smoke")
    if not isinstance(smoke, Mapping):
        raise _error("GPU_SMOKE_SUMMARY_INVALID", gpu_ref)
    allowed_devices = tuple(
        (str(item.get("pci_bus_id")), str(item.get("uuid")))
        for item in smoke.get("allowed_devices", [])
        if isinstance(item, Mapping)
    )
    if allowed_devices != G21_ALLOWED_DEVICES:
        raise _error("GPU_ALLOWED_IDENTITY_DRIFT", gpu_ref)
    if smoke.get("excluded_pci_bus_ids") != [G21_EXCLUDED_PCI]:
        raise _error("GPU_EXCLUDED_PCI_DRIFT", gpu_ref)

    # The handoff schema intentionally contains only the raw report ref/sha;
    # ``excluded_device`` lives in that report.  Re-read it and bind the bad
    # card's PCI+UUID instead of trusting a copied index or an optional summary
    # field.
    smoke_ref = _source_ref(smoke.get("ref"), "g21_handoff.current_gpu_smoke.ref")
    report = _load_mapping(root, smoke_ref, "g21_current_gpu_smoke")
    if report.get("schema_version") != "stage2-s202-current-gpu-smoke-v1" or report.get("status") != "PASS":
        raise _error("GPU_SMOKE_REPORT_INVALID", smoke_ref)
    if report.get("excluded_pci_bus_ids") != [EXCLUDED_PCI]:
        raise _error("GPU_SMOKE_REPORT_EXCLUSION_INVALID", smoke_ref)
    excluded = report.get("excluded_device")
    if (
        not isinstance(excluded, Mapping)
        or str(excluded.get("index")) != EXCLUDED_GPU_INDEX
        or str(excluded.get("pci_bus_id")) != EXCLUDED_PCI
        or str(excluded.get("uuid")) != EXCLUDED_UUID
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
    return gpu_ref, allowed_devices


def _load_formal_gate_refs(
    root: Path,
    gate_refs: Mapping[str, str],
    *,
    required: Sequence[str],
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
    reference_sizing_plan_ref: str | None = None,
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
    gpu_ref, allowed_devices = _load_gpu_health_identity(root, g21_handoff_ref)
    if set(ALLOWED_GPU_INDICES).intersection({EXCLUDED_GPU_INDEX}):
        raise _error("GPU_ALLOWED_INDEX_INCLUDES_EXCLUDED")

    # Store a dedicated binding object so a consumer can re-read both stable
    # hardware identities and the scheduler indices.  The indices are merely a
    # launch allow-list; PCI+UUID pairs remain the identity authority.
    binding_ref = PurePosixPath(output_ref).with_name("gpu-health-binding.json").as_posix()
    binding_payload = {
        "schema_version": "stage2-s204-gpu-health-binding-v1",
        "source_ref": gpu_ref,
        "excluded": {
            "index": int(EXCLUDED_GPU_INDEX),
            "pci_bus_id": EXCLUDED_PCI,
            "uuid": EXCLUDED_UUID,
        },
        "allowed_indices": [int(item) for item in sorted(ALLOWED_GPU_INDICES)],
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid}
            for pci, uuid in allowed_devices
        ],
    }
    binding_payload["binding_hash"] = canonical_json_hash(binding_payload)
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
        asset_commit = load_committed_task_artifact(root, asset_source, require_formal=True)
        if asset_commit.identity.task_id != "stage2.03_assets_checkpoints_and_sampling" or asset_commit.identity.artifact_kind != "asset_resolution":
            raise ValueError("wrong S2.3 asset resolution commit")
        asset_manifest = _formal_s23_asset_manifest(asset_commit.payload, field=asset_source)
    except Exception as error:
        raise _error("S23_ASSET_RESOLUTION_INVALID", asset_source) from error

    required_gates = {"stage2.G2.2"}
    optional_gates = {"stage1.G1-EXIT"}
    if not required_gates.issubset(gate_refs) or set(gate_refs) - required_gates - optional_gates:
        raise _error("GATE_REF_SET_INVALID", sorted(set(gate_refs) ^ required_gates))
    normalized_gate_refs: dict[str, str] = {}
    for gate_id, ref in gate_refs.items():
        gate_ref = _source_ref(ref, f"gate.{gate_id}")
        try:
            loaded = load_committed_task_artifact(root, gate_ref, require_formal=True)
        except Exception as error:
            raise _error("GATE_FORMAL_COMMIT_REQUIRED", gate_id) from error
        for source_ref in loaded.source_refs:
            _reject_nonformal_ref(source_ref, f"gate.{gate_id}.source_ref")
        # A gate may be nested in a formal task payload; TaskRuntime performs
        # the same exact-one GateRecord check at preflight time.
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
        normalized_gate_refs[gate_id] = gate_ref

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
        "gate_stage2_g2_2": normalized_gate_refs["stage2.G2.2"],
        "gpu_health": gpu_ref,
        "gpu_health_binding": binding_ref,
        # This exact key is consumed by stage23_task_runners._formal_input_document;
        # keeping one authoritative ref prevents an input scan from accepting a
        # second sizing plan accidentally placed in predecessor commits.
        "formal_reference_sizing_plan": sizing_ref,
    }
    if "stage1.G1-EXIT" in normalized_gate_refs:
        evidence_refs["gate_stage1_g1_exit"] = normalized_gate_refs["stage1.G1-EXIT"]
    evidence_refs.update({f"capability_{key}": value for key, value in normalized_capabilities.items()})
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(required_capabilities),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage2.G2.2"}),
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


def _checkpoint_identity(
    checkpoint: CheckpointRecord,
    *,
    tokenizer_asset_id: str,
    data_asset_id: str,
    data_revision: str,
) -> dict[str, Any]:
    if not checkpoint.ready or checkpoint.revision is None:
        raise _error("CHECKPOINT_NOT_READY", checkpoint.checkpoint_id)
    model_asset_id = f"{checkpoint.model_id}-step{checkpoint.training_step}"
    root_asset_id = _asset_id_from_root(checkpoint.root_ref)
    if root_asset_id != model_asset_id:
        raise _error(
            "CHECKPOINT_MODEL_ASSET_ID_MISMATCH",
            f"{checkpoint.checkpoint_id}:{root_asset_id}!={model_asset_id}",
        )
    if checkpoint.manifest_sha256 is None or checkpoint.tokenizer_sha256 is None:
        raise _error("CHECKPOINT_MANIFEST_IDENTITY_MISSING", checkpoint.checkpoint_id)
    return {
        "model_asset_id": model_asset_id,
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
) -> dict[str, ResolvedConfigV2]:
    """Generate one unique, identity-bound v2 config per formal checkpoint.

    ``mode=fresh`` emits configs with ``recovery.resume_ref=null``.  ``mode=resume``
    requires an explicit per-cell logical committed boundary; it never guesses
    a checkpoint or reuses another cell's output.
    """

    if mode not in {"fresh", "resume"}:
        raise _error("CONFIG_MODE_INVALID", mode)
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
    cells: dict[str, ResolvedConfigV2] = {}
    for checkpoint in manifest.checkpoints:
        identity = _checkpoint_identity(
            checkpoint,
            tokenizer_asset_id=tokenizer_asset_id,
            data_asset_id=data_asset_id,
            data_revision=data.revision,
        )
        if g3_assets is not None:
            try:
                model_asset = g3_assets.resolve(identity["model_asset_id"], expected_kind="model")
                tokenizer_asset = g3_assets.resolve(tokenizer_asset_id, expected_kind="tokenizer")
                data_asset = g3_assets.resolve(data_asset_id, expected_kind="pile")
            except Exception as error:
                raise _error("G3_CONFIG_ASSET_ID_UNRESOLVED", cell_id) from error
            if (
                model_asset.manifest_ref != checkpoint.manifest_ref
                or model_asset.ready_manifest_sha256 != checkpoint.manifest_sha256
                or tokenizer_asset.ready_manifest_sha256 != checkpoint.tokenizer_sha256
                or data_asset.manifest_ref != data.manifest_ref
                or data_asset.ready_manifest_sha256 != data.manifest_sha256
            ):
                raise _error("G3_CONFIG_MANIFEST_IDENTITY_MISMATCH", cell_id)
        cell_id = f"{checkpoint.model_id}-{checkpoint.training_stage}"
        if cell_id in cells:
            raise _error("CELL_ID_DUPLICATE", cell_id)
        if mode == "resume":
            if resume_refs is None or cell_id not in resume_refs:
                raise _error("RESUME_REF_REQUIRED", cell_id)
            resume_ref = _logical(resume_refs[cell_id], f"resume_ref.{cell_id}")
            if not resume_ref.startswith(f"{output_root_ref}/{cell_id}/"):
                raise _error("RESUME_REF_CELL_MISMATCH", cell_id)
        else:
            if resume_refs is not None and cell_id in resume_refs:
                raise _error("FRESH_CONFIG_CANNOT_CARRY_RESUME_REF", cell_id)
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
        base_value["identity"]["task"] = S204_TASK_ID
        base_value["identity"]["stage"] = 2
        base_value["identity"]["run_intent"] = "formal"
        base_value["identity"]["formal_eligible"] = True
        output_ref = f"{output_root_ref}/{cell_id}"
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
        cells[cell_id] = config
    if len(cells) != 6 or len({item.config_hash for item in cells.values()}) != 6:
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
        target_ref = f"{output_dir}/{mode}/{cell_id}.json"
        target = _safe_relative(root, target_ref, "config_output")
        publish_canonical_immutable(target, config.to_dict())
        reread = ResolvedConfigV2.from_mapping(load_canonical_json(target))
        if reread.full_hash != config.full_hash:
            raise _error("CONFIG_ROUND_TRIP_DRIFT", cell_id)
        refs[cell_id] = PurePosixPath(target_ref).as_posix()
    return refs


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
    parser.add_argument("--output-dir", default="evidence/stage2/s204/materialized-task-inputs")
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
        raw_sources: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None
        candidate_refs: dict[str, dict[str, str]] | None = None
        if source.get("task_outputs") is not None:
            task_sources = _mapping(source.get("task_outputs"), "sources.task_outputs")
            materialized = materialize_formal_task_inputs(root, task_sources, output_dir=args.output_dir)
            bootstrap_mode = False
        elif source.get("raw_task_outputs") is not None:
            raw_sources = _mapping(source.get("raw_task_outputs"), "sources.raw_task_outputs")
            candidate_refs = bootstrap_formal_task_inputs(root, raw_sources, output_dir=args.output_dir)
            materialized = {}
            bootstrap_mode = True
        else:
            raise _error("FORMAL_TASK_OUTPUTS_REQUIRED", "task_outputs or raw_task_outputs")
        formal_ref = _source_ref(source["formal_execution"], "sources.formal_execution")
        evidence, evidence_ref = _publish_formal_execution(
            root,
            formal_ref,
            destination=str(source.get("materialized_formal_execution", "evidence/stage2/s204/formal-execution.json")),
        )
        if raw_sources is not None:
            materialized = execute_formal_predecessor_dag(
                root,
                source=source,
                raw_refs=raw_sources,
                formal_execution_ref=evidence_ref,
                base_config_ref=args.base_config,
                output_dir=args.output_dir,
            )
        gate_refs = _mapping(source.get("gate_refs"), "sources.gate_refs")
        capability_refs = _mapping(source.get("capability_refs"), "sources.capability_refs")
        # Publish the sizing document first so the environment can bind and
        # reread the exact object under the runner's canonical evidence key.
        _, sizing_ref = publish_reference_sizing_plan(
            root,
            formal_execution=evidence,
            output_ref=args.sizing_output,
        )
        environment, environment_ref = build_formal_runtime_environment(
            root,
            formal_execution_ref=evidence_ref,
            stage0_handoff_ref=str(source["stage0_handoff"]),
            stage1_g1_exit_ref=str(source["stage1_g1_exit"]),
            contract_freeze_ref=str(source["contract_freeze"]),
            g3_resolution_ref=str(source["g3_resolution"]),
            stage2_asset_resolution_ref=materialized["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"],
            gate_refs={str(k): str(v) for k, v in gate_refs.items()},
            capability_refs={str(k): str(v) for k, v in capability_refs.items()},
            contract_stage_0_ref=(None if source.get("contract_stage_0") is None else str(source["contract_stage_0"])),
            contract_stage_1_ref=(None if source.get("contract_stage_1") is None else str(source["contract_stage_1"])),
            contract_stage_2_ref=(None if source.get("contract_stage_2") is None else str(source["contract_stage_2"])),
            g21_handoff_ref=str(source.get("g21_handoff", source.get("gpu_health", ""))),
            reference_sizing_plan_ref=sizing_ref,
            output_ref=args.environment_output,
        )
        resume_refs = _parse_key_refs(args.resume_ref, "resume_ref") if args.mode == "resume" else None
        configs = generate_six_cell_configs(
            root,
            asset_manifest_ref=materialized["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"],
            predecessor_refs=materialized,
            base_config_ref=args.base_config,
            g3_resolution_ref=str(source["g3_resolution"]),
            output_dir=args.config_output_dir,
            mode=args.mode,
            resume_refs=resume_refs,
        )
        config_refs = write_six_cell_configs(configs, root, output_dir=args.config_output_dir, mode=args.mode)
        summary = {
            "schema_version": "stage2-s204-materialization-summary-v1",
            "formal_eligible": False,
            "excluded_gpu": {"index": 1, "pci": EXCLUDED_PCI, "allowed_indices": sorted(ALLOWED_GPU_INDICES)},
            "task_output_refs": materialized,
            "candidate_task_output_refs": candidate_refs,
            "runtime_environment_ref": environment_ref,
            "environment_hash": environment.environment_hash,
            "formal_execution_ref": evidence_ref,
            "sizing_plan_ref": sizing_ref,
            "sizing_plan_schema": S204_SCHEMA,
            "config_refs": config_refs,
            "cell_count": len(config_refs),
            "mode": args.mode,
            "bootstrap_mode": bootstrap_mode,
        }
        summary_ref = f"{args.config_output_dir}/{args.mode}/materialization-summary.json"
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
    "ALLOWED_GPU_INDICES",
    "EXCLUDED_PCI",
    "EXCLUDED_UUID",
    "S204MaterializationError",
    "TASK_INPUTS",
    "build_formal_runtime_environment",
    "bootstrap_formal_task_inputs",
    "execute_formal_predecessor_dag",
    "generate_six_cell_configs",
    "main",
    "materialize_formal_task_inputs",
    "publish_reference_sizing_plan",
    "write_six_cell_configs",
]
