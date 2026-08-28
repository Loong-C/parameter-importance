"""Production Stage 3 endpoint/probe unit-index construction.

This module is intentionally a read-only consumer of published artifacts.  It
does not run a model, generate a probe, or manufacture an endpoint.  A unit
index can only be built from canonical ``endpoint-commit-v1`` files whose
backing ``endpoint-record-v1`` object and replay/state identities are intact,
and canonical Stage 3 probe-plan/panel files.  The resulting index is a
declaration of work, not a result artifact.

The matrix shape is deliberately stricter than the generic Stage 3 protocol:
pilot is exactly six 14M endpoints with two probes each (12 units), while
formal is exactly 24 14M plus 9 31M endpoints with three probes each (99
units).  Fixture and synthetic artifacts are rejected in both scopes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, TypeAlias

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json


PRODUCTION_PLAN_SCHEMA = "stage3-production-unit-index-v1"
ENDPOINT_COMMIT_SCHEMA = "endpoint-commit-v1"
ENDPOINT_OBJECT_SCHEMA = "endpoint-record-v1"
PROBE_PLAN_SCHEMAS = frozenset({"stage3-probe-plan-v1", "stage3-probe-panel-v1"})
PILOT_SCOPE = "pilot"
FORMAL_SCOPE = "formal"
STAGES = ("early", "middle", "late")
# These are the frozen formal training seeds from the Stage 3 preregistration.
# They are part of the production-index contract, not examples.  In
# particular, accepting the old 0/1 placeholders would make a formally
# published 14M/31M matrix point at the wrong training trajectories.
FORMAL_MODEL_SEEDS: Mapping[str, frozenset[int]] = {
    "14M": frozenset({4301, 4302}),
    "31M": frozenset({5301}),
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")
_FORBIDDEN_RE = re.compile(r"(?:fixture|synthetic)", re.IGNORECASE)

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
SourceInput: TypeAlias = str | Path | Mapping[str, Any] | Sequence[object]


class ProductionPlanError(ValueError):
    """Raised when a production source is missing, stale, or inconsistent."""


def _fail(code: str, detail: object | None = None) -> ProductionPlanError:
    if detail is None:
        return ProductionPlanError(code)
    return ProductionPlanError(f"{code}:{detail}")


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _fail("HASH_INVALID", field)
    return value


def _id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise _fail("IDENTIFIER_INVALID", field)
    return value


def _no_forbidden(value: object, *, field: str = "source") -> None:
    """Reject fixture/synthetic labels recursively, including mapping keys."""

    if isinstance(value, str):
        if _FORBIDDEN_RE.search(value):
            raise _fail("FIXTURE_OR_SYNTHETIC_REJECTED", field)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _FORBIDDEN_RE.search(key):
                raise _fail("FIXTURE_OR_SYNTHETIC_REJECTED", f"{field}.{key}")
            _no_forbidden(child, field=field)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _no_forbidden(child, field=field)


def _legacy_stage3_hash(value: object) -> str:
    """Hash used by ``stage3.EndpointRecord`` and ``EndpointState``.

    Stage 3 identity hashes predate ``canonical_json_hash`` and intentionally
    omit its trailing LF.  Repeating the small encoding here keeps this module
    lightweight while still checking the producer's actual record digest.
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_identity_hash(state: Mapping[str, object]) -> str:
    return _legacy_stage3_hash(
        {
            "artifact_id": state["artifact_id"],
            "artifact_hash": state["artifact_hash"],
            "parameter_hash": state["parameter_hash"],
            "buffer_hash": state["buffer_hash"],
            "optimizer_hash": state["optimizer_hash"],
            "scheduler_hash": state["scheduler_hash"],
            "scaler_hash": state["scaler_hash"],
            "rng_hash": state["rng_hash"],
            "data_cursor_hash": state["data_cursor_hash"],
            "model_mode_hash": state["model_mode_hash"],
        }
    )


def _probe_digest(entry: Mapping[str, object]) -> str:
    return _legacy_stage3_hash(
        {
            "probe_id": entry["probe_id"],
            "sample_ids": entry["sample_ids"],
            "content_hash": entry["content_hash"],
            "loss_contract_hash": entry["loss_contract_hash"],
            "effective_weight_unit": entry["effective_weight_unit"],
        }
    )


def _record_digest(record: Mapping[str, object]) -> str:
    return _legacy_stage3_hash(
        {
            "path_state_id": record["path_state_id"],
            "source_run_id": record["source_run_id"],
            "optimizer_step": record["optimizer_step"],
            "parameter_registry_hash": record["parameter_registry_hash"],
            "pre_state": _state_identity_hash(record["pre_state"]),  # type: ignore[arg-type]
            "parameter_post_state": _state_identity_hash(record["parameter_post_state"]),  # type: ignore[arg-type]
            "attempt_commit_state": _state_identity_hash(record["attempt_commit_state"]),  # type: ignore[arg-type]
            "attempt_commit_parent_hash": record["attempt_commit_parent_hash"],
            "probe_buffer_snapshot_hash": record["probe_buffer_snapshot_hash"],
            "full_update_delta_hash": record["full_update_delta_hash"],
            "update_sample_ids": record["update_sample_ids"],
            "replay_verified": record["replay_verified"],
        }
    )


def _logical_ref(path: Path, workspace_root: Path | None) -> str:
    resolved = path.resolve()
    if workspace_root is not None:
        try:
            return resolved.relative_to(workspace_root).as_posix()
        except ValueError as error:
            raise _fail("REFERENCE_OUTSIDE_WORKSPACE", resolved) from error
    return resolved.as_posix()


def _safe_reference(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail("REFERENCE_INVALID", field)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise _fail("REFERENCE_PATH_ESCAPE", field)
    return value


class _Source:
    __slots__ = ("ref", "path", "inline")

    def __init__(self, ref: str, path: Path | None, inline: Mapping[str, Any] | None = None) -> None:
        self.ref = ref
        self.path = path
        self.inline = inline


def _source_from_path(value: str | Path, *, workspace_root: Path | None) -> _Source:
    path = Path(value)
    if not path.is_absolute() and workspace_root is not None:
        path = workspace_root.joinpath(path)
    path = path.resolve()
    return _Source(_logical_ref(path, workspace_root), path)


def _source_items(value: SourceInput, *, workspace_root: Path | None) -> list[_Source]:
    """Flatten path(s), directories, or ref -> path mappings.

    Inline payloads are intentionally not accepted.  Hash-bound refs must point
    at immutable files, otherwise a caller could supply a dictionary that was
    never committed by the endpoint/probe producer.
    """

    if isinstance(value, (str, Path)):
        source = _source_from_path(value, workspace_root=workspace_root)
        if source.path is None:
            raise _fail("SOURCE_PATH_MISSING", source.ref)
        if source.path.is_dir():
            preferred = source.path / "commits"
            directory = preferred if preferred.is_dir() else source.path
            paths = sorted(directory.rglob("*.json"))
            return [_Source(_logical_ref(path, workspace_root), path) for path in paths]
        return [source]
    if isinstance(value, Mapping):
        if "schema_version" in value:
            raise _fail("SOURCE_REF_REQUIRED")
        result: list[_Source] = []
        for ref, child in value.items():
            if not isinstance(ref, str) or not isinstance(child, (str, Path)):
                raise _fail("SOURCE_MAPPING_MUST_MAP_REF_TO_PATH")
            source = _source_from_path(child, workspace_root=workspace_root)
            # Keep the caller's logical ref, because it is the provenance field
            # that downstream consumers will use.
            result.append(_Source(_safe_reference(ref, field="source_ref"), source.path))
        return result
    if isinstance(value, (str, bytes)):
        raise _fail("SOURCE_SEQUENCE_INVALID")
    try:
        values = tuple(value)
    except TypeError as error:
        raise _fail("SOURCE_SEQUENCE_INVALID") from error
    result = []
    for child in values:
        if not isinstance(child, (str, Path)):
            raise _fail("SOURCE_SEQUENCE_ITEMS_MUST_BE_PATHS")
        result.extend(_source_items(child, workspace_root=workspace_root))
    return result


def _read_source(source: _Source) -> Mapping[str, object]:
    if source.path is None or not source.path.is_file():
        raise _fail("SOURCE_NOT_FOUND", source.ref)
    try:
        value = load_canonical_json(source.path)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("SOURCE_NOT_CANONICAL", source.ref) from error
    if not isinstance(value, Mapping):
        raise _fail("SOURCE_ROOT_NOT_OBJECT", source.ref)
    return value


def _resolve_object_path(commit_path: Path, object_ref: str, workspace_root: Path | None) -> Path:
    _safe_reference(object_ref, field="object_ref")
    relative = Path(object_ref)
    candidates: list[Path] = []
    if workspace_root is not None:
        candidates.append(workspace_root.joinpath(*relative.parts))
    # TrainingEndpointObserver emits objects/<id>.json while object_ref is
    # relative to the endpoint root (the parent of ``commits``).
    candidates.append(commit_path.parent.parent.joinpath(*relative.parts))
    candidates.append(commit_path.parent.joinpath(*relative.parts))
    for candidate in candidates:
        resolved = candidate.resolve()
        if workspace_root is not None:
            try:
                resolved.relative_to(workspace_root)
            except ValueError:
                continue
        if resolved.is_file():
            return resolved
    raise _fail("ENDPOINT_OBJECT_NOT_FOUND", object_ref)


def _required_state(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("ENDPOINT_STATE_INVALID", field)
    required = {
        "artifact_id", "artifact_hash", "parameter_hash", "buffer_hash",
        "optimizer_hash", "scheduler_hash", "scaler_hash", "rng_hash",
        "data_cursor_hash", "model_mode_hash",
    }
    if set(value) != required:
        raise _fail("ENDPOINT_STATE_FIELDS_MISMATCH", field)
    if not isinstance(value["artifact_id"], str) or not value["artifact_id"]:
        raise _fail("ENDPOINT_STATE_ARTIFACT_ID_INVALID", field)
    for key in required - {"artifact_id"}:
        _hash(value[key], field=f"{field}.{key}")
    return value


def _metadata_value(record: Mapping[str, object], commit: Mapping[str, object], keys: Sequence[str]) -> object:
    metadata = record.get("metadata")
    containers: tuple[Mapping[str, object], ...] = tuple(
        item for item in (metadata, record, commit) if isinstance(item, Mapping)
    )
    for container in containers:
        for key in keys:
            if key in container:
                return container[key]
    return None


@dataclass(frozen=True, slots=True)
class EndpointCommitIdentity:
    """Validated identity extracted from one committed endpoint."""

    ref: str
    artifact_hash: str
    endpoint_id: str
    endpoint_digest: str
    model: str
    seed: int
    stage: str
    update_sample_ids: tuple[str | int, ...]
    qualification_evidence_hash: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint_ref": self.ref,
            "endpoint_hash": self.artifact_hash,
            "endpoint_id": self.endpoint_id,
            "endpoint_digest": self.endpoint_digest,
            "model": self.model,
            "seed": self.seed,
            "stage": self.stage,
            "update_sample_ids": list(self.update_sample_ids),
            "qualification_evidence_hash": self.qualification_evidence_hash,
        }


def _load_endpoint(source: _Source, *, scope: str, workspace_root: Path | None) -> EndpointCommitIdentity:
    commit = _read_source(source)
    _no_forbidden(commit, field=source.ref)
    expected = {
        "schema_version", "endpoint_id", "optimizer_step", "endpoint_digest",
        "object_ref", "object_sha256", "scope", "formal_eligible",
        "qualification_evidence_hash", "artifact_hash",
    }
    if set(commit) != expected or commit.get("schema_version") != ENDPOINT_COMMIT_SCHEMA:
        raise _fail("ENDPOINT_COMMIT_FIELDS_MISMATCH", source.ref)
    body = {key: value for key, value in commit.items() if key != "artifact_hash"}
    artifact_hash = _hash(commit["artifact_hash"], field="endpoint.artifact_hash")
    if artifact_hash != canonical_json_hash(body):
        raise _fail("ENDPOINT_COMMIT_HASH_MISMATCH", source.ref)
    endpoint_id = _id(commit["endpoint_id"], field="endpoint_id")
    endpoint_digest = _hash(commit["endpoint_digest"], field="endpoint_digest")
    object_ref = _safe_reference(commit["object_ref"], field="object_ref")
    object_sha = _hash(commit["object_sha256"], field="object_sha256")
    formal_eligible = commit["formal_eligible"]
    if type(formal_eligible) is not bool:
        raise _fail("ENDPOINT_FORMAL_ELIGIBILITY_INVALID", source.ref)
    commit_scope = commit["scope"]
    if not isinstance(commit_scope, str) or commit_scope == "local_fixture" or _FORBIDDEN_RE.search(commit_scope):
        raise _fail("ENDPOINT_SCOPE_REJECTED", source.ref)
    if scope == FORMAL_SCOPE:
        if commit_scope != FORMAL_SCOPE or formal_eligible is not True:
            raise _fail("FORMAL_ENDPOINT_NOT_QUALIFIED", source.ref)
    elif scope == PILOT_SCOPE:
        if formal_eligible is True or commit_scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
            raise _fail("PILOT_ENDPOINT_SCOPE_MISMATCH", source.ref)
    else:
        raise _fail("PRODUCTION_SCOPE_INVALID", scope)
    evidence = commit["qualification_evidence_hash"]
    if formal_eligible:
        evidence = _hash(evidence, field="qualification_evidence_hash")
    elif evidence is not None:
        raise _fail("UNQUALIFIED_ENDPOINT_CARRIES_EVIDENCE", source.ref)

    if source.path is None:
        raise _fail("ENDPOINT_COMMIT_MUST_BE_FILE", source.ref)
    object_path = _resolve_object_path(source.path, object_ref, workspace_root)
    try:
        obj = load_canonical_json(object_path)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("ENDPOINT_OBJECT_NOT_CANONICAL", object_ref) from error
    if not isinstance(obj, Mapping):
        raise _fail("ENDPOINT_OBJECT_ROOT_INVALID", object_ref)
    if canonical_json_hash(obj) != object_sha:
        raise _fail("ENDPOINT_OBJECT_HASH_MISMATCH", source.ref)
    if obj.get("schema_version") != ENDPOINT_OBJECT_SCHEMA:
        raise _fail("ENDPOINT_OBJECT_SCHEMA_MISMATCH", source.ref)
    declared_object_hash = obj.get("artifact_hash")
    if not isinstance(declared_object_hash, str) or declared_object_hash != canonical_json_hash(
        {key: value for key, value in obj.items() if key != "artifact_hash"}
    ):
        raise _fail("ENDPOINT_OBJECT_ARTIFACT_HASH_MISMATCH", source.ref)
    object_expected = {"schema_version", "scope", "formal_eligible", "qualification_evidence_hash", "record", "state_bundles", "artifact_hash"}
    if set(obj) != object_expected:
        raise _fail("ENDPOINT_OBJECT_FIELDS_MISMATCH", source.ref)
    if obj["scope"] != commit_scope or obj["formal_eligible"] != formal_eligible or obj["qualification_evidence_hash"] != evidence:
        raise _fail("ENDPOINT_OBJECT_SCOPE_DRIFT", source.ref)

    record = obj["record"]
    if not isinstance(record, Mapping):
        raise _fail("ENDPOINT_RECORD_ROOT_INVALID", source.ref)
    record_expected = {
        "path_state_id", "source_run_id", "optimizer_step", "parameter_registry_hash",
        "pre_state", "parameter_post_state", "attempt_commit_state", "attempt_commit_parent_hash",
        "probe_buffer_snapshot_hash", "full_update_delta_hash", "update_sample_ids",
        "replay_verified", "metadata", "endpoint_digest",
    }
    if set(record) != record_expected:
        raise _fail("ENDPOINT_RECORD_FIELDS_MISMATCH", source.ref)
    if record["path_state_id"] != endpoint_id:
        raise _fail("ENDPOINT_RECORD_ID_MISMATCH", source.ref)
    if not isinstance(record["source_run_id"], str) or not record["source_run_id"]:
        raise _fail("ENDPOINT_SOURCE_RUN_INVALID", source.ref)
    if isinstance(record["optimizer_step"], bool) or not isinstance(record["optimizer_step"], int) or record["optimizer_step"] <= 0:
        raise _fail("ENDPOINT_STEP_INVALID", source.ref)
    if record["optimizer_step"] != commit["optimizer_step"]:
        raise _fail("ENDPOINT_STEP_DRIFT", source.ref)
    _hash(record["parameter_registry_hash"], field="parameter_registry_hash")
    if record["replay_verified"] is not True:
        raise _fail("ENDPOINT_REPLAY_NOT_VERIFIED", source.ref)
    if not isinstance(record["metadata"], Mapping):
        raise _fail("ENDPOINT_METADATA_INVALID", source.ref)
    for state_name in ("pre_state", "parameter_post_state", "attempt_commit_state"):
        _required_state(record[state_name], field=state_name)
    pre = record["pre_state"]
    post = record["parameter_post_state"]
    attempt = record["attempt_commit_state"]
    assert isinstance(pre, Mapping) and isinstance(post, Mapping) and isinstance(attempt, Mapping)
    if pre["buffer_hash"] != post["buffer_hash"] or post["buffer_hash"] != record["probe_buffer_snapshot_hash"]:
        raise _fail("ENDPOINT_BUFFER_DRIFT", source.ref)
    if post["parameter_hash"] != attempt["parameter_hash"] or post["buffer_hash"] != attempt["buffer_hash"] or post["optimizer_hash"] != attempt["optimizer_hash"]:
        raise _fail("ENDPOINT_COMMIT_STATE_DRIFT", source.ref)
    if post["artifact_id"] == attempt["artifact_id"] or record["attempt_commit_parent_hash"] != post["artifact_hash"]:
        raise _fail("ENDPOINT_POST_COMMIT_BINDING_INVALID", source.ref)
    if pre["parameter_hash"] == post["parameter_hash"]:
        raise _fail("ENDPOINT_ZERO_DELTA_REJECTED", source.ref)
    for field in ("attempt_commit_parent_hash", "probe_buffer_snapshot_hash", "full_update_delta_hash", "endpoint_digest"):
        _hash(record[field], field=field)
    update_ids = record["update_sample_ids"]
    if not isinstance(update_ids, list) or not update_ids:
        raise _fail("ENDPOINT_UPDATE_SAMPLE_IDS_INVALID", source.ref)
    if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in update_ids) or len(set(update_ids)) != len(update_ids):
        raise _fail("ENDPOINT_UPDATE_SAMPLE_IDS_INVALID", source.ref)
    if record["endpoint_digest"] != endpoint_digest or _record_digest(record) != endpoint_digest:
        raise _fail("ENDPOINT_RECORD_DIGEST_MISMATCH", source.ref)
    raw_bundles = obj["state_bundles"]
    if not isinstance(raw_bundles, Mapping) or set(raw_bundles) != {"pre", "parameter_post", "attempt_commit"}:
        raise _fail("ENDPOINT_STATE_BUNDLES_INCOMPLETE", source.ref)
    for phase, reference in raw_bundles.items():
        if not isinstance(reference, Mapping) or set(reference) != {"ref", "manifest_sha256"}:
            raise _fail("ENDPOINT_STATE_BUNDLE_REFERENCE_INVALID", phase)
        _safe_reference(reference["ref"], field=f"state_bundles.{phase}.ref")
        _hash(reference["manifest_sha256"], field=f"state_bundles.{phase}.manifest_sha256")
        state = record["pre_state" if phase == "pre" else phase + "_state"]
        assert isinstance(state, Mapping)
        if reference["manifest_sha256"] != state["artifact_hash"]:
            raise _fail("ENDPOINT_STATE_BUNDLE_HASH_MISMATCH", phase)

    model_raw = _metadata_value(record, commit, ("model", "model_size"))
    stage_raw = _metadata_value(record, commit, ("stage", "training_stage"))
    seed_raw = _metadata_value(record, commit, ("seed", "training_seed", "master_seed"))
    if model_raw is None or stage_raw is None or seed_raw is None:
        raise _fail("ENDPOINT_METADATA_MISSING", source.ref)
    model = _id(model_raw, field="model")
    stage = _id(stage_raw, field="stage")
    if stage not in STAGES:
        raise _fail("ENDPOINT_STAGE_INVALID", source.ref)
    if isinstance(seed_raw, bool) or not isinstance(seed_raw, int):
        raise _fail("ENDPOINT_SEED_INVALID", source.ref)
    return EndpointCommitIdentity(
        ref=source.ref,
        artifact_hash=artifact_hash,
        endpoint_id=endpoint_id,
        endpoint_digest=endpoint_digest,
        model=model,
        seed=seed_raw,
        stage=stage,
        update_sample_ids=tuple(update_ids),
        qualification_evidence_hash=evidence if isinstance(evidence, str) else None,
    )


@dataclass(frozen=True, slots=True)
class ProbePlanIdentity:
    ref: str
    artifact_hash: str
    endpoint_digest: str
    execution_evidence_hash: str
    entries: tuple[Mapping[str, object], ...]


def _load_probe_plan(source: _Source, *, scope: str) -> ProbePlanIdentity:
    plan = _read_source(source)
    _no_forbidden(plan, field=source.ref)
    schema = plan.get("schema_version")
    if schema not in PROBE_PLAN_SCHEMAS:
        raise _fail("PROBE_PLAN_SCHEMA_INVALID", source.ref)
    expected = {
        "schema_version", "panel_id", "endpoint_digest", "entries",
        "minimum_formal_probes", "execution_evidence_hash", "scope",
        "formal_eligible", "artifact_hash",
    }
    if schema == "stage3-probe-panel-v1":
        expected.add("qualification_gate_hash")
    if set(plan) != expected:
        raise _fail("PROBE_PLAN_FIELDS_MISMATCH", source.ref)
    artifact_hash = _hash(plan["artifact_hash"], field="probe_plan.artifact_hash")
    if artifact_hash != canonical_json_hash({key: value for key, value in plan.items() if key != "artifact_hash"}):
        raise _fail("PROBE_PLAN_HASH_MISMATCH", source.ref)
    _id(plan["panel_id"], field="panel_id")
    endpoint_digest = _hash(plan["endpoint_digest"], field="probe_plan.endpoint_digest")
    execution_hash = _hash(plan["execution_evidence_hash"], field="probe_plan.execution_evidence_hash")
    plan_scope = plan["scope"]
    formal_eligible = plan["formal_eligible"]
    if type(formal_eligible) is not bool:
        raise _fail("PROBE_PLAN_FORMAL_ELIGIBILITY_INVALID", source.ref)
    if scope == FORMAL_SCOPE and (plan_scope != FORMAL_SCOPE or formal_eligible is not True):
        raise _fail("FORMAL_PROBE_PLAN_NOT_QUALIFIED", source.ref)
    if scope == PILOT_SCOPE and (formal_eligible is True or plan_scope not in {PILOT_SCOPE, FORMAL_SCOPE}):
        raise _fail("PILOT_PROBE_PLAN_SCOPE_MISMATCH", source.ref)
    if schema == "stage3-probe-panel-v1":
        qualification = plan["qualification_gate_hash"]
        if formal_eligible is True:
            _hash(qualification, field="probe_plan.qualification_gate_hash")
        elif qualification is not None:
            raise _fail("PROBE_PLAN_QUALIFICATION_GATE_INVALID", source.ref)
    entries_raw = plan["entries"]
    if not isinstance(entries_raw, list) or not entries_raw:
        raise _fail("PROBE_PLAN_ENTRIES_EMPTY", source.ref)
    normalized: list[Mapping[str, object]] = []
    probe_ids: set[str] = set()
    seen_samples: set[str | int] = set()
    expected_entry_fields = {"role", "probe_id", "sample_ids", "content_hash", "loss_contract_hash", "effective_weight_unit", "metadata"}
    for index, raw in enumerate(entries_raw):
        if not isinstance(raw, Mapping):
            raise _fail("PROBE_ENTRY_INVALID", f"{source.ref}:{index}")
        allowed = expected_entry_fields | {"probe_digest"}
        if set(raw) - allowed or not expected_entry_fields.issubset(raw):
            raise _fail("PROBE_ENTRY_FIELDS_MISMATCH", f"{source.ref}:{index}")
        role = raw["role"]
        expected_role = "formal" if scope == FORMAL_SCOPE else "pilot"
        if role != expected_role:
            raise _fail("PROBE_SCOPE_MIXED", f"{source.ref}:{index}")
        probe_id = _id(raw["probe_id"], field="probe_id")
        if probe_id in probe_ids:
            raise _fail("PROBE_ID_DUPLICATE", probe_id)
        probe_ids.add(probe_id)
        sample_ids = raw["sample_ids"]
        if not isinstance(sample_ids, list) or not sample_ids:
            raise _fail("PROBE_SAMPLE_IDS_INVALID", probe_id)
        if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise _fail("PROBE_SAMPLE_IDS_INVALID", probe_id)
        if seen_samples.intersection(sample_ids):
            raise _fail("PROBE_SAMPLE_ID_DUPLICATE", probe_id)
        seen_samples.update(sample_ids)
        _hash(raw["content_hash"], field=f"{probe_id}.content_hash")
        _hash(raw["loss_contract_hash"], field=f"{probe_id}.loss_contract_hash")
        if not isinstance(raw["effective_weight_unit"], str) or not raw["effective_weight_unit"]:
            raise _fail("PROBE_WEIGHT_UNIT_INVALID", probe_id)
        if not isinstance(raw["metadata"], Mapping):
            raise _fail("PROBE_METADATA_INVALID", probe_id)
        digest = _probe_digest(raw)
        if "probe_digest" in raw and raw["probe_digest"] != digest:
            raise _fail("PROBE_DIGEST_MISMATCH", probe_id)
        normalized.append(raw)
    minimum = plan["minimum_formal_probes"]
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
        raise _fail("PROBE_MINIMUM_INVALID", source.ref)
    if scope == FORMAL_SCOPE and (minimum < 3 or len(normalized) != 3):
        raise _fail("FORMAL_PROBE_COUNT_INVALID", source.ref)
    if scope == PILOT_SCOPE and len(normalized) != 2:
        raise _fail("PILOT_PROBE_COUNT_INVALID", source.ref)
    return ProbePlanIdentity(source.ref, artifact_hash, endpoint_digest, execution_hash, tuple(normalized))


@dataclass(frozen=True, slots=True)
class ProductionUnit:
    """One hash-bound ``endpoint × probe`` work unit."""

    path_unit_id: str
    scope: str
    model: str
    seed: int
    stage: str
    update: str
    update_sample_ids: tuple[str | int, ...]
    probe: str
    probe_sample_ids: tuple[str | int, ...]
    endpoint_ref: str
    endpoint_hash: str
    endpoint_digest: str
    probe_ref: str
    probe_hash: str
    probe_digest: str
    probe_content_hash: str
    loss_contract_hash: str
    effective_weight_unit: str

    def __post_init__(self) -> None:
        _id(self.path_unit_id, field="path_unit_id")
        if self.scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
            raise _fail("UNIT_SCOPE_INVALID")
        _id(self.model, field="model")
        _id(self.stage, field="stage")
        _id(self.update, field="update")
        _id(self.probe, field="probe")
        for field_name in ("endpoint_hash", "endpoint_digest", "probe_hash", "probe_digest", "probe_content_hash", "loss_contract_hash"):
            _hash(getattr(self, field_name), field=field_name)
        if not self.update_sample_ids or not self.probe_sample_ids:
            raise _fail("UNIT_SAMPLE_IDS_EMPTY", self.path_unit_id)
        for field_name in ("update_sample_ids", "probe_sample_ids"):
            values = getattr(self, field_name)
            if any(isinstance(item, bool) or not isinstance(item, (str, int)) for item in values):
                raise _fail("UNIT_SAMPLE_IDS_INVALID", self.path_unit_id)
            if len(set(values)) != len(values):
                raise _fail("UNIT_SAMPLE_IDS_DUPLICATE", self.path_unit_id)
        if set(self.update_sample_ids).intersection(self.probe_sample_ids):
            raise _fail("UNIT_UPDATE_PROBE_OVERLAP", self.path_unit_id)
        object.__setattr__(self, "update_sample_ids", tuple(self.update_sample_ids))
        object.__setattr__(self, "probe_sample_ids", tuple(self.probe_sample_ids))

    @property
    def unit_id(self) -> str:
        return self.path_unit_id

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "path_unit_id": self.path_unit_id,
            "scope": self.scope,
            "model": self.model,
            "seed": self.seed,
            "stage": self.stage,
            "update": self.update,
            "update_sample_ids": list(self.update_sample_ids),
            "probe": self.probe,
            "probe_sample_ids": list(self.probe_sample_ids),
            "endpoint_ref": self.endpoint_ref,
            "endpoint_hash": self.endpoint_hash,
            "endpoint_digest": self.endpoint_digest,
            "probe_ref": self.probe_ref,
            "probe_hash": self.probe_hash,
            "probe_digest": self.probe_digest,
            "probe_content_hash": self.probe_content_hash,
            "loss_contract_hash": self.loss_contract_hash,
            "effective_weight_unit": self.effective_weight_unit,
        }

    @property
    def scientific_identity_hash(self) -> str:
        """Stable identity for one endpoint×probe declaration.

        This is deliberately computed from the complete hash-bound unit row;
        execution diagnostics and observations cannot alter it.
        """

        return canonical_json_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class ProductionUnitIndex:
    """Immutable, hash-bound pilot or formal unit index."""

    index_id: str
    scope: str
    units: tuple[ProductionUnit, ...]

    def __post_init__(self) -> None:
        _id(self.index_id, field="index_id")
        if self.scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
            raise _fail("PRODUCTION_SCOPE_INVALID", self.scope)
        if not self.units:
            raise _fail("UNIT_INDEX_EMPTY")
        expected = 12 if self.scope == PILOT_SCOPE else 99
        if len(self.units) != expected:
            raise _fail("UNIT_COVERAGE_INVALID", f"expected={expected},actual={len(self.units)}")
        if any(unit.scope != self.scope for unit in self.units):
            raise _fail("UNIT_SCOPE_MIXED")
        unit_ids = tuple(unit.path_unit_id for unit in self.units)
        if len(set(unit_ids)) != len(unit_ids):
            raise _fail("PATH_UNIT_ID_DUPLICATE")
        if len({unit.probe for unit in self.units}) != len(self.units):
            raise _fail("PROBE_ID_DUPLICATE")
        for unit in self.units:
            if unit.path_unit_id != _unit_path_id(
                {
                    key: value
                    for key, value in unit.to_dict().items()
                    if key != "path_unit_id"
                }
            ):
                raise _fail("PATH_UNIT_ID_HASH_MISMATCH", unit.path_unit_id)
        endpoint_groups: dict[str, list[ProductionUnit]] = {}
        for unit in self.units:
            endpoint_groups.setdefault(unit.endpoint_digest, []).append(unit)
        expected_probes = 2 if self.scope == PILOT_SCOPE else 3
        if len(endpoint_groups) != (6 if self.scope == PILOT_SCOPE else 33):
            raise _fail("UNIT_ENDPOINT_COVERAGE_INVALID")
        if any(len(group) != expected_probes for group in endpoint_groups.values()):
            raise _fail("UNIT_PROBE_COVERAGE_INVALID")
        if self.scope == FORMAL_SCOPE:
            by_model: dict[str, set[str]] = {}
            by_model_seed_stage: dict[tuple[str, int, str], set[str]] = {}
            for unit in self.units:
                by_model.setdefault(unit.model, set()).add(unit.endpoint_digest)
                by_model_seed_stage.setdefault(
                    (unit.model, unit.seed, unit.stage), set()
                ).add(unit.endpoint_digest)
            if {key: len(value) for key, value in by_model.items()} != {
                "14M": 24,
                "31M": 9,
            }:
                raise _fail("FORMAL_MODEL_ENDPOINT_COVERAGE_INVALID")
            expected = {
                ("14M", seed, stage): 4
                for seed in FORMAL_MODEL_SEEDS["14M"]
                for stage in STAGES
            } | {
                ("31M", seed, stage): 3
                for seed in FORMAL_MODEL_SEEDS["31M"]
                for stage in STAGES
            }
            if {
                key: len(value) for key, value in by_model_seed_stage.items()
            } != expected:
                raise _fail("FORMAL_MODEL_SEED_STAGE_COVERAGE_INVALID")
        else:
            if {unit.model for unit in self.units} != {"14M"}:
                raise _fail("PILOT_MODEL_COVERAGE_INVALID")
            if len({unit.seed for unit in self.units}) != 1:
                raise _fail("PILOT_SEED_COVERAGE_INVALID")
            for stage in STAGES:
                if len(
                    {
                        unit.endpoint_digest
                        for unit in self.units
                        if unit.stage == stage
                    }
                ) != 2:
                    raise _fail("PILOT_STAGE_ENDPOINT_COVERAGE_INVALID", stage)

    @property
    def formal_eligible(self) -> bool:
        return self.scope == FORMAL_SCOPE

    @property
    def endpoint_count(self) -> int:
        return len({unit.endpoint_digest for unit in self.units})

    @property
    def probe_count(self) -> int:
        return len({(unit.endpoint_digest, unit.probe) for unit in self.units})

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def payload_dict(self) -> dict[str, object]:
        by_model: dict[str, set[str]] = {}
        by_model_seed_stage: dict[str, set[str]] = {}
        for unit in self.units:
            by_model.setdefault(unit.model, set()).add(unit.endpoint_digest)
            key = f"{unit.model}:{unit.seed}:{unit.stage}"
            by_model_seed_stage.setdefault(key, set()).add(unit.endpoint_digest)
        return {
            "schema_version": PRODUCTION_PLAN_SCHEMA,
            "index_id": self.index_id,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "endpoint_count": self.endpoint_count,
            "probe_count": self.probe_count,
            "unit_count": self.unit_count,
            "units": [unit.to_dict() for unit in self.units],
            "coverage": {
                "by_model": {key: len(value) for key, value in sorted(by_model.items())},
                "by_model_seed_stage": {
                    key: len(value) for key, value in sorted(by_model_seed_stage.items())
                },
            },
        }

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def unit_strata(self) -> dict[str, dict[str, str]]:
        """Derive the only admissible formal-plan strata from this index."""

        return {
            unit.path_unit_id: {
                "model": unit.model,
                "stage": unit.stage,
                "update": unit.update,
                "probe": unit.probe,
            }
            for unit in self.units
        }

    def unit(self, path_unit_id: str) -> ProductionUnit:
        matches = tuple(
            item for item in self.units if item.path_unit_id == path_unit_id
        )
        if len(matches) != 1:
            raise _fail("PATH_UNIT_ID_UNKNOWN", path_unit_id)
        return matches[0]


def _unit_path_id(payload: Mapping[str, object]) -> str:
    return f"path-unit-{canonical_json_hash(payload)}"


def path_unit_id_for_payload(payload: Mapping[str, object]) -> str:
    """Public canonical path-unit-id algorithm shared by runners and builders."""

    return _unit_path_id(
        {
            key: value for key, value in payload.items() if key != "path_unit_id"
        }
    )


def load_production_unit_index(
    path: str | Path,
    *,
    workspace_root: str | Path | None = None,
    expected_scope: str = FORMAL_SCOPE,
) -> ProductionUnitIndex:
    """Load and fully revalidate an immutable production unit index.

    Formal execution must consume this file rather than reconstructing IDs or
    strata from a task-local endpoint/probe panel.  ``workspace_root`` is only
    a path safety boundary; the index itself remains content-hash bound.
    """

    if expected_scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
        raise _fail("PRODUCTION_SCOPE_INVALID", expected_scope)
    target = Path(path)
    if not target.is_absolute() and workspace_root is not None:
        target = Path(workspace_root).joinpath(target)
    target = target.resolve()
    root = None if workspace_root is None else Path(workspace_root).resolve()
    if root is not None:
        try:
            target.relative_to(root)
        except ValueError as error:
            raise _fail("REFERENCE_OUTSIDE_WORKSPACE", target) from error
    try:
        value = load_canonical_json(target)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("UNIT_INDEX_NOT_CANONICAL", target) from error
    if not isinstance(value, Mapping):
        raise _fail("UNIT_INDEX_ROOT_NOT_OBJECT", target)
    _no_forbidden(value, field="production_unit_index")
    expected = {
        "schema_version", "index_id", "scope", "formal_eligible",
        "endpoint_count", "probe_count", "unit_count", "units", "coverage",
        "artifact_hash",
    }
    if set(value) != expected or value.get("schema_version") != PRODUCTION_PLAN_SCHEMA:
        raise _fail("UNIT_INDEX_FIELDS_MISMATCH", target)
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    artifact_hash = _hash(value["artifact_hash"], field="unit_index.artifact_hash")
    if artifact_hash != canonical_json_hash(body):
        raise _fail("UNIT_INDEX_HASH_MISMATCH", target)
    scope = value["scope"]
    if scope != expected_scope:
        raise _fail("UNIT_INDEX_SCOPE_MISMATCH", target)
    raw_units = value["units"]
    if not isinstance(raw_units, list):
        raise _fail("UNIT_INDEX_UNITS_INVALID", target)
    units: list[ProductionUnit] = []
    for index, raw in enumerate(raw_units):
        if not isinstance(raw, Mapping):
            raise _fail("UNIT_INDEX_UNIT_INVALID", index)
        try:
            _safe_reference(raw["endpoint_ref"], field="unit.endpoint_ref")
            _safe_reference(raw["probe_ref"], field="unit.probe_ref")
            units.append(
                ProductionUnit(
                    path_unit_id=str(raw["path_unit_id"]),
                    scope=str(raw["scope"]),
                    model=str(raw["model"]),
                    seed=int(raw["seed"]),
                    stage=str(raw["stage"]),
                    update=str(raw["update"]),
                    update_sample_ids=tuple(raw["update_sample_ids"]),  # type: ignore[arg-type]
                    probe=str(raw["probe"]),
                    probe_sample_ids=tuple(raw["probe_sample_ids"]),  # type: ignore[arg-type]
                    endpoint_ref=str(raw["endpoint_ref"]),
                    endpoint_hash=str(raw["endpoint_hash"]),
                    endpoint_digest=str(raw["endpoint_digest"]),
                    probe_ref=str(raw["probe_ref"]),
                    probe_hash=str(raw["probe_hash"]),
                    probe_digest=str(raw["probe_digest"]),
                    probe_content_hash=str(raw["probe_content_hash"]),
                    loss_contract_hash=str(raw["loss_contract_hash"]),
                    effective_weight_unit=str(raw["effective_weight_unit"]),
                )
            )
        except (KeyError, TypeError, ValueError) as error:
            raise _fail("UNIT_INDEX_UNIT_INVALID", index) from error
        if set(raw) != set(units[-1].to_dict()):
            raise _fail("UNIT_INDEX_UNIT_FIELDS_MISMATCH", index)
        if dict(raw) != units[-1].to_dict():
            raise _fail("UNIT_INDEX_UNIT_NORMALIZATION_MISMATCH", index)
    index = ProductionUnitIndex(
        index_id=str(value["index_id"]),
        scope=str(scope),
        units=tuple(units),
    )
    if index.to_dict() != dict(value):
        raise _fail("UNIT_INDEX_PAYLOAD_MISMATCH", target)
    return index


def _validate_endpoint_coverage(endpoints: Sequence[EndpointCommitIdentity], *, scope: str) -> None:
    if scope == PILOT_SCOPE:
        if len(endpoints) != 6 or {item.model for item in endpoints} != {"14M"} or len({item.seed for item in endpoints}) != 1:
            raise _fail("PILOT_ENDPOINT_COVERAGE_INVALID")
        for stage in STAGES:
            selected = [item for item in endpoints if item.stage == stage]
            if len(selected) != 2:
                raise _fail("PILOT_STAGE_ENDPOINT_COUNT_INVALID", stage)
        return
    if len(endpoints) != 33 or {item.model for item in endpoints} != {"14M", "31M"}:
        raise _fail("FORMAL_ENDPOINT_COVERAGE_INVALID")
    for model, expected_seeds, per_stage in (
        ("14M", FORMAL_MODEL_SEEDS["14M"], 4),
        ("31M", FORMAL_MODEL_SEEDS["31M"], 3),
    ):
        selected_model = [item for item in endpoints if item.model == model]
        observed_seeds = {item.seed for item in selected_model}
        if observed_seeds != set(expected_seeds):
            raise _fail("FORMAL_SEED_COVERAGE_INVALID", model)
        for seed in expected_seeds:
            for stage in STAGES:
                count = sum(item.seed == seed and item.stage == stage for item in selected_model)
                if count != per_stage:
                    raise _fail("FORMAL_STAGE_ENDPOINT_COUNT_INVALID", f"{model}:{seed}:{stage}")


def build_production_unit_index(
    endpoint_commits: SourceInput,
    probe_plans: SourceInput,
    *,
    scope: str,
    workspace_root: str | Path | None = None,
    index_id: str | None = None,
) -> ProductionUnitIndex:
    """Build a strict pilot/formal index from committed endpoint/probe files.

    ``endpoint_commits`` may be a commit file, an endpoint directory, a list of
    files, or a ``logical_ref -> file`` mapping.  ``probe_plans`` accepts the
    same forms.  All files must be canonical JSON; inline mappings are rejected
    so the output always carries a real file reference and hash.
    """

    if scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
        raise _fail("PRODUCTION_SCOPE_INVALID", scope)
    root = None if workspace_root is None else Path(workspace_root).resolve()
    endpoint_sources = _source_items(endpoint_commits, workspace_root=root)
    probe_sources = _source_items(probe_plans, workspace_root=root)
    if not endpoint_sources:
        raise _fail("ENDPOINT_COMMITS_MISSING")
    if not probe_sources:
        raise _fail("PROBE_PLANS_MISSING")
    endpoints = tuple(_load_endpoint(source, scope=scope, workspace_root=root) for source in endpoint_sources)
    if len({item.endpoint_id for item in endpoints}) != len(endpoints) or len({item.endpoint_digest for item in endpoints}) != len(endpoints):
        raise _fail("ENDPOINT_DUPLICATE")
    update_samples: set[str | int] = set()
    for endpoint in endpoints:
        overlap = update_samples.intersection(endpoint.update_sample_ids)
        if overlap:
            raise _fail("UPDATE_SAMPLE_DUPLICATE", endpoint.endpoint_id)
        update_samples.update(endpoint.update_sample_ids)
    _validate_endpoint_coverage(endpoints, scope=scope)
    plans = tuple(_load_probe_plan(source, scope=scope) for source in probe_sources)
    if len({plan.ref for plan in plans}) != len(plans) or len({plan.endpoint_digest for plan in plans}) != len(plans):
        raise _fail("PROBE_PLAN_DUPLICATE")
    endpoint_by_digest = {item.endpoint_digest: item for item in endpoints}
    if set(endpoint_by_digest) != {plan.endpoint_digest for plan in plans}:
        raise _fail("ENDPOINT_PROBE_COVERAGE_MISMATCH")
    evidence = {item.qualification_evidence_hash for item in endpoints if item.qualification_evidence_hash is not None}
    plan_evidence = {item.execution_evidence_hash for item in plans}
    if scope == FORMAL_SCOPE and (len(evidence) != 1 or plan_evidence != evidence):
        raise _fail("FORMAL_EVIDENCE_BINDING_INVALID")
    probe_samples: set[str | int] = set()
    probe_ids_global: set[str] = set()
    units: list[ProductionUnit] = []
    for plan in plans:
        endpoint = endpoint_by_digest[plan.endpoint_digest]
        for entry in plan.entries:
            sample_ids = tuple(entry["sample_ids"])  # type: ignore[arg-type]
            if set(endpoint.update_sample_ids).intersection(sample_ids):
                raise _fail("UPDATE_PROBE_OVERLAP", endpoint.endpoint_id)
            if probe_samples.intersection(sample_ids):
                raise _fail("PROBE_SAMPLE_DUPLICATE", str(entry["probe_id"]))
            probe_samples.update(sample_ids)
            probe_id = str(entry["probe_id"])
            if probe_id in probe_ids_global:
                raise _fail("PROBE_ID_DUPLICATE", probe_id)
            probe_ids_global.add(probe_id)
            probe_digest = str(entry.get("probe_digest") or _probe_digest(entry))
            payload: dict[str, object] = {
                "scope": scope,
                "model": endpoint.model,
                "seed": endpoint.seed,
                "stage": endpoint.stage,
                "update": endpoint.endpoint_id,
                "update_sample_ids": list(endpoint.update_sample_ids),
                "probe": probe_id,
                "probe_sample_ids": list(sample_ids),
                "endpoint_ref": endpoint.ref,
                "endpoint_hash": endpoint.artifact_hash,
                "endpoint_digest": endpoint.endpoint_digest,
                "probe_ref": plan.ref,
                "probe_hash": plan.artifact_hash,
                "probe_digest": probe_digest,
                "probe_content_hash": entry["content_hash"],
                "loss_contract_hash": entry["loss_contract_hash"],
                "effective_weight_unit": entry["effective_weight_unit"],
            }
            units.append(ProductionUnit(path_unit_id=_unit_path_id(payload), **payload))  # type: ignore[arg-type]
    units.sort(key=lambda item: (item.model, item.seed, item.stage, item.update, item.probe))
    expected_units = 12 if scope == PILOT_SCOPE else 99
    if len(units) != expected_units:
        raise _fail("UNIT_COVERAGE_INVALID", f"expected={expected_units},actual={len(units)}")
    if index_id is None:
        index_id = f"stage3-{scope}-production-v1"
    return ProductionUnitIndex(index_id=index_id, scope=scope, units=tuple(units))


def write_production_unit_index(path: str | Path, index: ProductionUnitIndex) -> Path:
    """Publish an index as canonical JSON without overwriting a different one."""

    if not isinstance(index, ProductionUnitIndex):
        raise TypeError("index 必须是 ProductionUnitIndex")
    target = Path(path)
    if target.exists():
        existing = load_canonical_json(target)
        if existing != index.to_dict():
            raise _fail("UNIT_INDEX_IMMUTABLE_CONFLICT", target)
        return target
    write_canonical_json(target, index.to_dict())
    return target


# Naming aliases keep callers aligned with the Stage 3 plan vocabulary.
build_stage3_production_unit_index = build_production_unit_index
build_stage3_unit_index = build_production_unit_index
build_production_plan = build_production_unit_index
build_stage3_production_plan = build_production_unit_index
build_unit_index = build_production_unit_index


__all__ = [
    "ENDPOINT_COMMIT_SCHEMA",
    "ENDPOINT_OBJECT_SCHEMA",
    "FORMAL_MODEL_SEEDS",
    "FORMAL_SCOPE",
    "PILOT_SCOPE",
    "PROBE_PLAN_SCHEMAS",
    "PRODUCTION_PLAN_SCHEMA",
    "EndpointCommitIdentity",
    "ProbePlanIdentity",
    "ProductionPlanError",
    "ProductionUnit",
    "ProductionUnitIndex",
    "build_production_plan",
    "build_production_unit_index",
    "build_stage3_production_plan",
    "build_stage3_production_unit_index",
    "build_stage3_unit_index",
    "build_unit_index",
    "load_production_unit_index",
    "path_unit_id_for_payload",
    "write_production_unit_index",
]
