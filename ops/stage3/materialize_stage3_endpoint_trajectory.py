"""Materialize the Stage 3 real endpoint-trajectory control plane.

This module is deliberately a *control-plane* compiler.  It reads only explicit
JSON sources and publishes four immutable JSON objects:

* a ``training-endpoint-capture-plan-v1``;
* a strict ``ResolvedConfigV2`` for ``stage3.03``;
* a hash-bound ``TaskRuntimeEnvironment`` derived from the published G3-0/G3-1
  environments; and
* a materialization receipt describing the command that may later run the real
  trajectory producer.

It never imports a training runner, probes CUDA, starts a process, or creates an
endpoint/result artifact.  In particular, Stage 2 output references are kept in
the environment provenance.  ``stage3.03`` has no catalog input contracts, so
putting these references in ``orchestration.input_result_refs`` would make the
runtime reject an otherwise valid trajectory before its own strict loader gets
to inspect the capture plan.

The source contract is intentionally explicit: ``model``/``seed``/``max_steps``
are not inferred from a path or a free-form run name, and every selected step has
an exact ``model``/``seed``/``stage`` entry.  Pilot and formal scopes share the
same real training semantics but are never upgraded into one another: scope,
eligibility, probe scope, output identity, and the published environment all
remain bound to the requested path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from ops.stage3.materialize_stage3_fanout import (
    FORBIDDEN_RE,
    _base_v1,
    _hash,
    _logical,
    _merge_section,
    _no_forbidden,
)
from ops.stage3.run_stage3_formal import (
    EXPECTED_STAGE2_RUN_ID,
    Stage3OrchestratorError,
    _fail,
    _resolve_ref,
    validate_stage2_identity,
)
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    ResolvedConfigV2,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.runtime import TaskRuntimeEnvironment, publish_canonical_immutable


SOURCE_SCHEMA = "stage3-endpoint-trajectory-materialization-source-v1"
RECEIPT_SCHEMA = "stage3-endpoint-trajectory-materialization-receipt-v1"
TASK_ID = "stage3.03_endpoint_and_probe_pipeline"
SCOPES = frozenset({"pilot", "formal"})
MODELS = frozenset({"14M", "31M"})
STAGES = frozenset({"early", "middle", "late"})
UPDATE_SPLIT = "train"
UPDATE_SAMPLING_DESIGN = "without_replacement_frozen_epoch"
UPDATE_SAMPLER = "without_replacement"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def _error(code: str, detail: object | None = None) -> Stage3OrchestratorError:
    return _fail(code, detail)


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _error("ENDPOINT_MATERIALIZATION_MAPPING_REQUIRED", field)
    return value


def _strict_hash(value: object, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _error("ENDPOINT_MATERIALIZATION_HASH_INVALID", field)
    return value


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise _error("ENDPOINT_MATERIALIZATION_ID_INVALID", field)
    return value


def _logical_output(value: object, field: str) -> str:
    """Validate a data-root-relative output reference and return its wire form."""

    return _logical(value, field)


def _load_object(path: Path, field: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise _error("ENDPOINT_MATERIALIZATION_INPUT_NOT_FOUND", field)
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_INPUT_INVALID", field) from error
    if not isinstance(value, Mapping):
        raise _error("ENDPOINT_MATERIALIZATION_INPUT_NOT_OBJECT", field)
    return value


def _relative_to(path: Path, root: Path, field: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise _error("ENDPOINT_MATERIALIZATION_REFERENCE_OUTSIDE_ROOT", field) from error


def _payload_from_ref(
    reference: object,
    *,
    roots: tuple[Path, ...],
    workspace_root: Path,
    data_root: Path,
    field: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = _resolve_ref(reference, roots=roots, field=field)
    value = _load_object(path, field)
    if value.get("schema_version") != "task-output-commit-v1":
        return path, value
    # Some published G3/G31 inputs are task commits rather than bare payloads.
    # Re-load through the strict commit loader so an object path cannot masquerade
    # as a commit or silently bypass its envelope identity.
    try:
        from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact

        if path.is_relative_to(data_root.resolve()):
            root = data_root
        elif path.is_relative_to(workspace_root.resolve()):
            root = workspace_root
        else:  # pragma: no cover - _resolve_ref already enforces this boundary
            raise ValueError("commit root unavailable")
        logical = _relative_to(path, root, field)
        loaded = load_committed_task_artifact(root, logical)
    except (OSError, TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_COMMIT_INVALID", field) from error
    if not isinstance(loaded.payload, Mapping):  # pragma: no cover - loader contract
        raise _error("ENDPOINT_MATERIALIZATION_COMMIT_PAYLOAD_INVALID", field)
    return path, loaded.payload


def _write_immutable(path: Path, value: Mapping[str, Any], field: str) -> None:
    """Publish a canonical object without ever overwriting a different object."""

    try:
        publish_canonical_immutable(path, value)
    except Exception as error:
        raise _error("ENDPOINT_MATERIALIZATION_IMMUTABLE_CONFLICT", field) from error


def _validate_stage2_outputs(
    refs: object,
    *,
    workspace_root: Path,
    data_root: Path,
) -> tuple[str, ...]:
    if (
        not isinstance(refs, list)
        or not refs
        or any(not isinstance(item, str) or not item for item in refs)
        or len(refs) != len(set(refs))
    ):
        raise _error("ENDPOINT_MATERIALIZATION_STAGE2_OUTPUT_REFS_INVALID")
    normalized: list[str] = []
    for index, reference in enumerate(refs):
        _no_forbidden(reference, f"stage2_output_refs[{index}]")
        path, _ = _payload_from_ref(
            reference,
            roots=(data_root, workspace_root),
            workspace_root=workspace_root,
            data_root=data_root,
            field=f"stage2_output_refs[{index}]",
        )
        normalized.append(
            _relative_to(path, data_root if path.is_relative_to(data_root) else workspace_root, f"stage2_output_refs[{index}]")
        )
    return tuple(normalized)


def _validate_step_metadata(
    source: Mapping[str, Any],
) -> tuple[tuple[int, ...], dict[str, dict[str, object]]]:
    model = source.get("model")
    if model not in MODELS:
        raise _error("ENDPOINT_MATERIALIZATION_MODEL_INVALID")
    seed = source.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise _error("ENDPOINT_MATERIALIZATION_SEED_INVALID")
    max_steps = source.get("max_steps")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise _error("ENDPOINT_MATERIALIZATION_MAX_STEPS_INVALID")
    selected = source.get("selected_steps")
    if (
        not isinstance(selected, list)
        or not selected
        or any(
            isinstance(step, bool)
            or not isinstance(step, int)
            or not 1 <= step <= max_steps
            for step in selected
        )
        or len(selected) != len(set(selected))
        or selected != sorted(selected)
    ):
        raise _error("ENDPOINT_MATERIALIZATION_SELECTED_STEPS_INVALID")
    raw_metadata = source.get("endpoint_metadata")
    if not isinstance(raw_metadata, Mapping) or not raw_metadata:
        raise _error("ENDPOINT_MATERIALIZATION_ENDPOINT_METADATA_REQUIRED")
    if {int(key) for key in raw_metadata if isinstance(key, str) and key.isdecimal()} != set(selected):
        raise _error("ENDPOINT_MATERIALIZATION_METADATA_COVERAGE_INVALID")
    metadata: dict[str, dict[str, object]] = {}
    for raw_step, item in raw_metadata.items():
        if not isinstance(raw_step, str) or not raw_step.isdecimal() or int(raw_step) <= 0:
            raise _error("ENDPOINT_MATERIALIZATION_METADATA_STEP_INVALID")
        if not isinstance(item, Mapping) or set(item) != {"model", "seed", "stage"}:
            raise _error("ENDPOINT_MATERIALIZATION_METADATA_FIELDS_INVALID")
        if item["model"] != model or item["seed"] != seed or item["stage"] not in STAGES:
            raise _error("ENDPOINT_MATERIALIZATION_METADATA_IDENTITY_INVALID", raw_step)
        metadata[str(int(raw_step))] = {
            "model": model,
            "seed": seed,
            "stage": item["stage"],
        }
    if set(metadata) != {str(step) for step in selected}:
        raise _error("ENDPOINT_MATERIALIZATION_METADATA_COVERAGE_INVALID")
    return tuple(selected), metadata


def _validate_probe_plan(
    reference: object,
    *,
    scope: str,
    workspace_root: Path,
    data_root: Path,
) -> str | None:
    if reference is None:
        return None
    _logical(reference, "probe_plan_ref")
    path, value = _payload_from_ref(
        reference,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="probe_plan_ref",
    )
    if value.get("schema_version") != "stage3-probe-plan-v1":
        raise _error("ENDPOINT_MATERIALIZATION_PROBE_PLAN_SCHEMA_INVALID")
    if value.get("scope") != scope or value.get("formal_eligible") is not (scope == "formal"):
        raise _error("ENDPOINT_MATERIALIZATION_PROBE_SCOPE_MISMATCH")
    declared = value.get("artifact_hash")
    if declared != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
        raise _error("ENDPOINT_MATERIALIZATION_PROBE_PLAN_HASH_INVALID")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise _error("ENDPOINT_MATERIALIZATION_PROBE_PLAN_ENTRIES_INVALID")
    if scope == "formal" and len(entries) < 3:
        raise _error("ENDPOINT_MATERIALIZATION_FORMAL_PROBE_COVERAGE_INVALID")
    if scope == "pilot" and len(entries) < 2:
        raise _error("ENDPOINT_MATERIALIZATION_PILOT_PROBE_COVERAGE_INVALID")
    return _relative_to(path, data_root if path.is_relative_to(data_root) else workspace_root, "probe_plan_ref")


def _environment_ref(
    environment: TaskRuntimeEnvironment,
    *keys: str,
) -> str | None:
    for key in keys:
        value = environment.evidence_refs.get(key)
        if value is not None:
            return value
    return None


def _gate_from_ref(
    reference: str,
    *,
    expected_gate_id: str,
    workspace_root: Path,
    data_root: Path,
) -> GateRecord:
    _, payload = _payload_from_ref(
        reference,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field=f"{expected_gate_id}.ref",
    )
    try:
        gate = GateRecord.from_mapping(dict(payload))
    except (TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_GATE_INVALID", expected_gate_id) from error
    if gate.gate_id != expected_gate_id or gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
        raise _error("ENDPOINT_MATERIALIZATION_GATE_NOT_PASS", expected_gate_id)
    return gate


def _validate_environments(
    source: Mapping[str, Any],
    *,
    workspace_root: Path,
    data_root: Path,
) -> tuple[TaskRuntimeEnvironment, TaskRuntimeEnvironment, str, GateRecord, GateRecord, FormalExecutionEvidence, str, str]:
    g30_ref = source.get("g30_environment_ref")
    g31_ref = source.get("g31_environment_ref")
    _logical(g30_ref, "g30_environment_ref")
    _logical(g31_ref, "g31_environment_ref")
    g30_path, g30_value = _payload_from_ref(
        g30_ref,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="g30_environment_ref",
    )
    g31_path, g31_value = _payload_from_ref(
        g31_ref,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="g31_environment_ref",
    )
    try:
        g30 = TaskRuntimeEnvironment.from_mapping(g30_value)
        g31 = TaskRuntimeEnvironment.from_mapping(g31_value)
    except (TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_ENVIRONMENT_INVALID") from error
    if "stage3.G3-0" not in g30.passed_gate_ids:
        raise _error("ENDPOINT_MATERIALIZATION_G30_ENVIRONMENT_GATE_MISSING")
    if not {"stage3.G3-0", "stage3.G3-1"}.issubset(g31.passed_gate_ids):
        raise _error("ENDPOINT_MATERIALIZATION_G31_ENVIRONMENT_GATE_MISSING")
    decision_ref = _environment_ref(g30, "stage3_scope_decision")
    g30_gate_ref = _environment_ref(g30, "stage3_g30_gate", "gate_stage3_g3_0")
    g31_decision_ref = _environment_ref(g31, "stage3_scope_decision")
    g31_g30_gate_ref = _environment_ref(g31, "stage3_g30_gate", "gate_stage3_g3_0")
    g31_gate_ref = _environment_ref(g31, "gate_stage3_g3_1", "stage3_g31_gate")
    execution_ref = _environment_ref(g31, "formal_execution")
    if not all(isinstance(item, str) and item for item in (decision_ref, g30_gate_ref, g31_gate_ref, execution_ref)):
        raise _error("ENDPOINT_MATERIALIZATION_ENVIRONMENT_EVIDENCE_MISSING")
    if decision_ref != g31_decision_ref or g30_gate_ref != g31_g30_gate_ref:
        raise _error("ENDPOINT_MATERIALIZATION_G30_ENVIRONMENT_BINDING_CONFLICT")
    assert isinstance(decision_ref, str) and isinstance(g30_gate_ref, str)
    assert isinstance(g31_gate_ref, str) and isinstance(execution_ref, str)
    decision_path, decision = _payload_from_ref(
        decision_ref,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="stage3_scope_decision",
    )
    g30_gate = _gate_from_ref(
        g30_gate_ref,
        expected_gate_id="stage3.G3-0",
        workspace_root=workspace_root,
        data_root=data_root,
    )
    g31_gate = _gate_from_ref(
        g31_gate_ref,
        expected_gate_id="stage3.G3-1",
        workspace_root=workspace_root,
        data_root=data_root,
    )
    try:
        from param_importance_nlp.contracts.stage3_scope import validate_stage3_scope_authority

        validate_stage3_scope_authority(decision, g30_gate, decision_ref=decision_ref)
    except (TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_G30_AUTHORITY_INVALID") from error
    _, execution_payload = _payload_from_ref(
        execution_ref,
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="formal_execution",
    )
    try:
        execution = FormalExecutionEvidence.from_mapping(dict(execution_payload))
        execution.require_for_stage(3)
    except (TypeError, ValueError) as error:
        raise _error("ENDPOINT_MATERIALIZATION_FORMAL_EXECUTION_INVALID") from error
    if not any(gate.gate_id == "stage3.G3-1" and gate.status is GateStatus.PASS for gate in execution.prerequisite_gates):
        raise _error("ENDPOINT_MATERIALIZATION_FORMAL_EXECUTION_G31_MISSING")
    if g31.environment_hash == g30.environment_hash:
        raise _error("ENDPOINT_MATERIALIZATION_G30_G31_ENVIRONMENT_NOT_DISTINCT")
    return (
        g30,
        g31,
        execution_ref,
        g30_gate,
        g31_gate,
        execution,
        _relative_to(decision_path, data_root if decision_path.is_relative_to(data_root) else workspace_root, "stage3_scope_decision"),
        _relative_to(g30_path, data_root if g30_path.is_relative_to(data_root) else workspace_root, "g30_environment_ref"),
    )


def _model_is_bound(base: Mapping[str, Any], model: str) -> bool:
    model_section = base.get("model")
    if not isinstance(model_section, Mapping):
        return False
    needle = model.casefold().replace("-", "").replace("_", "")
    return any(
        isinstance(value, str)
        and needle in value.casefold().replace("-", "").replace("_", "")
        for value in model_section.values()
    )


def _output_path(reference: str, *, data_root: Path, field: str) -> Path:
    path = _resolve_ref(reference, roots=(data_root,), field=field)
    try:
        path.relative_to(data_root.resolve())
    except ValueError as error:  # pragma: no cover - _resolve_ref enforces this
        raise _error("ENDPOINT_MATERIALIZATION_OUTPUT_OUTSIDE_DATA_ROOT", field) from error
    return path


def materialize(
    source: Mapping[str, Any],
    *,
    workspace_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    """Compile a real pilot/formal trajectory plan without executing it."""

    workspace_root = Path(workspace_root).resolve()
    data_root = Path(data_root).resolve()
    expected = {
        "schema_version",
        "scope",
        "plan_id",
        "model",
        "seed",
        "max_steps",
        "selected_steps",
        "include_checkpoint_steps",
        "endpoint_metadata",
        "probe_plan_ref",
        "base_config_ref",
        "stage2_authority_ref",
        "stage2_output_refs",
        "g30_environment_ref",
        "g31_environment_ref",
        "artifact_output_dir",
        "cache_root",
        "capture_plan_ref",
        "config_ref",
        "environment_ref",
        "materialization_receipt_ref",
        "trajectory_receipt_ref",
    }
    if not isinstance(source, Mapping) or set(source) != expected or source.get("schema_version") != SOURCE_SCHEMA:
        raise _error("ENDPOINT_MATERIALIZATION_SOURCE_FIELDS_INVALID")
    _no_forbidden(source, "source")
    scope = source.get("scope")
    if scope not in SCOPES:
        raise _error("ENDPOINT_MATERIALIZATION_SCOPE_INVALID")
    plan_id = _safe_id(source.get("plan_id"), "plan_id")
    selected_steps, endpoint_metadata = _validate_step_metadata(source)
    if type(source.get("include_checkpoint_steps")) is not bool:
        raise _error("ENDPOINT_MATERIALIZATION_CHECKPOINT_FLAG_INVALID")
    logical_inputs = {
        field: _logical(source.get(field), field)
        for field in (
            "base_config_ref",
            "stage2_authority_ref",
            "g30_environment_ref",
            "g31_environment_ref",
            "artifact_output_dir",
            "cache_root",
            "capture_plan_ref",
            "config_ref",
            "environment_ref",
            "materialization_receipt_ref",
            "trajectory_receipt_ref",
        )
    }
    probe_plan_ref = _validate_probe_plan(
        source.get("probe_plan_ref"),
        scope=str(scope),
        workspace_root=workspace_root,
        data_root=data_root,
    )
    stage2_output_refs = _validate_stage2_outputs(
        source.get("stage2_output_refs"),
        workspace_root=workspace_root,
        data_root=data_root,
    )
    output_names = (
        "capture_plan_ref",
        "config_ref",
        "environment_ref",
        "materialization_receipt_ref",
        "trajectory_receipt_ref",
    )
    output_paths = {field: _output_path(logical_inputs[field], data_root=data_root, field=field) for field in output_names}
    if len(set(output_paths.values())) != len(output_paths):
        raise _error("ENDPOINT_MATERIALIZATION_OUTPUT_REF_COLLISION")
    _, stage2_authority = _payload_from_ref(
        source["stage2_authority_ref"],
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="stage2_authority_ref",
    )
    _no_forbidden(stage2_authority, "stage2_authority")
    validate_stage2_identity(stage2_authority)
    _, base_value = _payload_from_ref(
        source["base_config_ref"],
        roots=(data_root, workspace_root),
        workspace_root=workspace_root,
        data_root=data_root,
        field="base_config_ref",
    )
    _no_forbidden(base_value, "base_config")
    base = _base_v1(base_value)
    if not _model_is_bound(base, str(source["model"])):
        raise _error("ENDPOINT_MATERIALIZATION_MODEL_BASE_CONFIG_MISMATCH")
    identity = base.get("identity")
    runtime = base.get("runtime")
    model_section = base.get("model")
    data = base.get("data")
    importance = base.get("importance")
    path_integration = base.get("path_integration")
    if not all(isinstance(item, dict) for item in (identity, runtime, model_section, data, importance, path_integration)):
        raise _error("ENDPOINT_MATERIALIZATION_BASE_SECTIONS_INVALID")
    seed = int(source["seed"])
    identity.update(
        {
            "stage": 3,
            "task": TASK_ID,
            "route": "path_integration",
            "master_seed": seed,
            "run_intent": "formal",
            "formal_eligible": True,
            "input_run_id": EXPECTED_STAGE2_RUN_ID,
        }
    )
    runtime.update(
        {
            "device": "cuda",
            "allow_dirty_worktree": False,
            "offline": True,
            "cache_root": logical_inputs["cache_root"],
            "output_root": str(PurePosixPath(logical_inputs["artifact_output_dir"]).parent),
            "temp_root": str(PurePosixPath(logical_inputs["cache_root"]) / "tmp"),
        }
    )
    data.update(
        {
            "split": UPDATE_SPLIT,
            "sampler": UPDATE_SAMPLER,
            "sampling_design": UPDATE_SAMPLING_DESIGN,
        }
    )
    importance.update(
        {
            "estimator_name": "u",
            "clip_mode": "none",
            "require_decision_for_formal": True,
        }
    )
    path_integration.update(
        {
            "enabled": True,
            "probe_count": 2 if scope == "pilot" else 3,
            "default_rule": "simpson",
            "fallback_rule": "gauss_legendre_8",
            "node_budget": 16,
            "thresholds_ref": "plans/stage3-thresholds.json",
        }
    )
    g30, g31, execution_ref, g30_gate, g31_gate, execution, decision_ref, _ = _validate_environments(
        source,
        workspace_root=workspace_root,
        data_root=data_root,
    )
    formal_hash = execution.artifact_hash
    capture_plan: dict[str, Any] = {
        "schema_version": "training-endpoint-capture-plan-v1",
        "plan_id": plan_id,
        "selected_steps": list(selected_steps),
        "include_checkpoint_steps": bool(source["include_checkpoint_steps"]),
        "scope": scope,
        "formal_eligible": scope == "formal",
        "qualification_evidence_hash": formal_hash if scope == "formal" else None,
        "probe_plan_ref": probe_plan_ref,
        "endpoint_metadata": endpoint_metadata,
    }
    capture_plan["artifact_hash"] = canonical_json_hash(capture_plan)
    _write_immutable(output_paths["capture_plan_ref"], capture_plan, "capture_plan_ref")

    # Keep S3.03's catalog input list empty; its own trajectory loader reads
    # these exact refs from the environment's auxiliary evidence map.
    overrides: dict[str, Any] = {
        "training": {"max_steps": int(source["max_steps"])},
        "providers": {
            "kind": "offline_hf",
            "model_manifest_ref": None,
            "model_root_ref": None,
            "data_manifest_ref": None,
            "data_root_ref": None,
            "tokenizer_manifest_ref": None,
            "tokenizer_root_ref": None,
            "task_type": "causal_lm",
            "task_name": "pile",
            "num_labels": None,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "orchestration": {"route_spec_ref": None, "input_result_refs": []},
        "recovery": {"resume_ref": None},
        "artifacts": {"output_dir": logical_inputs["artifact_output_dir"], "publish_partial": False},
    }
    resolved = ResolvedConfigV2.resolve(base, task_id=TASK_ID, overrides=overrides)
    providers = resolved.section("providers")
    if not isinstance(providers, Mapping) or providers.get("kind") != "offline_hf" or providers.get("local_files_only") is not True or providers.get("trust_remote_code") is not False:
        raise _error("ENDPOINT_MATERIALIZATION_OFFLINE_PROVIDER_REQUIRED")
    resolved_data = resolved.base_config.section("data")
    if not isinstance(resolved_data, Mapping) or resolved_data.get("split") != UPDATE_SPLIT or resolved_data.get("sampling_design") != UPDATE_SAMPLING_DESIGN or resolved_data.get("sampler") != UPDATE_SAMPLER:
        raise _error("ENDPOINT_MATERIALIZATION_UPDATE_ROUTE_INVALID")
    _write_immutable(output_paths["config_ref"], resolved.to_dict(), "config_ref")

    evidence_refs: dict[str, str] = dict(g31.evidence_refs)
    for key, value in g30.evidence_refs.items():
        if key in evidence_refs and evidence_refs[key] != value:
            raise _error("ENDPOINT_MATERIALIZATION_ENVIRONMENT_EVIDENCE_CONFLICT", key)
        evidence_refs[key] = value
    evidence_refs.update(
        {
            "stage3_endpoint_capture_plan": logical_inputs["capture_plan_ref"],
            "g30_environment": logical_inputs["g30_environment_ref"],
            "g31_environment": logical_inputs["g31_environment_ref"],
            "stage2_authority": logical_inputs["stage2_authority_ref"],
        }
    )
    for index, reference in enumerate(stage2_output_refs):
        evidence_refs[f"stage2_output_{index}"] = reference
    estimator_ref = g31.estimator_decision_ref or g30.estimator_decision_ref
    environment = TaskRuntimeEnvironment(
        capabilities=g30.capabilities | g31.capabilities,
        frozen_contract_stages=g30.frozen_contract_stages | g31.frozen_contract_stages,
        passed_gate_ids=g30.passed_gate_ids | g31.passed_gate_ids,
        estimator_decision_ref=estimator_ref,
        evidence_refs=evidence_refs,
    )
    _write_immutable(output_paths["environment_ref"], environment.to_dict(), "environment_ref")

    receipt_id = f"stage3-endpoint-trajectory-{scope}-{plan_id}"
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "receipt_id": receipt_id,
        "task_id": TASK_ID,
        "scope": scope,
        "formal_eligible": scope == "formal",
        "status": "MATERIALIZED",
        "model": source["model"],
        "seed": seed,
        "max_steps": source["max_steps"],
        "selected_steps": list(selected_steps),
        "endpoint_metadata": endpoint_metadata,
        "capture_plan_ref": logical_inputs["capture_plan_ref"],
        "capture_plan_hash": capture_plan["artifact_hash"],
        "config_ref": logical_inputs["config_ref"],
        "config_hash": resolved.config_hash,
        "environment_ref": logical_inputs["environment_ref"],
        "environment_hash": environment.environment_hash,
        "stage2_authority_ref": _logical(source["stage2_authority_ref"], "stage2_authority_ref"),
        "stage2_output_refs": list(stage2_output_refs),
        "g30_environment_ref": logical_inputs["g30_environment_ref"],
        "g31_environment_ref": logical_inputs["g31_environment_ref"],
        "g30_gate_hash": g30_gate.artifact_hash,
        "g31_gate_hash": g31_gate.artifact_hash,
        "formal_execution_ref": execution_ref,
        "formal_execution_hash": formal_hash,
        "trajectory_receipt_ref": logical_inputs["trajectory_receipt_ref"],
        "output_refs": {
            "capture_plan": logical_inputs["capture_plan_ref"],
            "config": logical_inputs["config_ref"],
            "environment": logical_inputs["environment_ref"],
            "trajectory_receipt": logical_inputs["trajectory_receipt_ref"],
        },
        "command": [
            "{python}",
            "-m",
            "param_importance_nlp",
            "task",
            "stage3-trajectory",
            "--config",
            "{config}",
            "--environment",
            "{environment}",
            "--receipt",
            "{trajectory_receipt}",
        ],
    }
    receipt["artifact_hash"] = canonical_json_hash(receipt)
    _write_immutable(output_paths["materialization_receipt_ref"], receipt, "materialization_receipt_ref")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        source_path = arguments.source.resolve()
        workspace_root = arguments.workspace_root.resolve()
        data_root = arguments.data_root.resolve()
        try:
            source_path.relative_to(data_root)
        except ValueError:
            source_path.relative_to(workspace_root)
        source = _load_object(source_path, "source")
        value = materialize(source, workspace_root=workspace_root, data_root=data_root)
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["RECEIPT_SCHEMA", "SOURCE_SCHEMA", "TASK_ID", "materialize", "main"]
