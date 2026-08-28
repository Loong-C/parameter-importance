"""Strict real Stage 3 endpoint trajectory production.

The normal ``stage3.03`` task has a historical path/probe artifact contract.  This
module is the execution-side producer for that task: it consumes the frozen capture
plan, runs the real :class:`TrainingTaskRunner` engine, and publishes only a separate
trajectory completion receipt.  In particular, it never calls ``TaskArtifactStore``
for the stage3.03 top-level artifact set.

The receipt is intentionally downstream of endpoint replay verification.  A run which
trains successfully but misses one planned endpoint, has stale model/seed/stage
metadata, or cannot prove its G3-0/G3-1 inputs fails closed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path, PurePosixPath
import re
from typing import Any

from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.stage3_scope import (
    validate_stage3_scope_authority,
    validate_stage3_scope_decision,
)
from ..contracts.status import GateRecord, GateStatus
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from ..contracts.stage23 import FormalExecutionEvidence
from ..runtime.task_artifacts import load_committed_task_artifact
from ..runtime.task_lifecycle import publish_canonical_immutable
from ..runtime.task_runtime import TaskExecutionRequest
from .stage3_production_plan import FORMAL_SCOPE, PILOT_SCOPE
from .task_runners import TrainingTaskRunner
from .training_endpoints import TrainingEndpointObserver


STAGE3_ENDPOINT_TASK_ID = "stage3.03_endpoint_and_probe_pipeline"
TRAJECTORY_RECEIPT_SCHEMA = "stage3-trajectory-completion-receipt-v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_MODEL_NAMES = frozenset({"14M", "31M"})
_STAGES = frozenset({"early", "middle", "late"})


class Stage3TrajectoryError(ValueError):
    """A strict trajectory precondition, capture, or receipt error."""


def _fail(code: str, detail: object | None = None) -> Stage3TrajectoryError:
    return Stage3TrajectoryError(code if detail is None else f"{code}:{detail}")


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _fail("HASH_INVALID", field)
    return value


def _logical_path(root: Path, value: str | Path, *, field: str) -> Path:
    raw = str(value)
    if not raw or "\\" in raw:
        raise _fail("REFERENCE_INVALID", field)
    candidate = Path(raw)
    if candidate.is_absolute():
        resolved = candidate.resolve()
    else:
        logical = PurePosixPath(raw)
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise _fail("REFERENCE_PATH_ESCAPE", field)
        resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise _fail("REFERENCE_OUTSIDE_WORKSPACE", field) from error
    return resolved


def _logical_ref(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise _fail("RECEIPT_OUTSIDE_WORKSPACE", path) from error


def _mapping_payload(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    payload = value.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return value


def _load_input_payloads(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    orchestration = request.config.section("orchestration")
    if not isinstance(orchestration, Mapping):
        raise _fail("ORCHESTRATION_SECTION_INVALID")
    refs = orchestration.get("input_result_refs")
    if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
        raise _fail("CAPTURE_PLAN_INPUT_REFS_MISSING")
    auxiliary_refs = tuple(
        reference
        for key in (
            "stage3_endpoint_capture_plan",
            "stage3_scope_decision",
            "stage3_g30_gate",
        )
        if isinstance(
            (reference := request.environment.evidence_refs.get(key)), str
        )
        and reference
    )
    loaded: list[tuple[str, Mapping[str, object]]] = []
    for raw_ref in dict.fromkeys((*refs, *auxiliary_refs)):
        if not isinstance(raw_ref, str):
            raise _fail("INPUT_REFERENCE_INVALID")
        path = _logical_path(root, raw_ref, field="input_result_refs")
        if not path.is_file():
            raise _fail("INPUT_ARTIFACT_NOT_FOUND", raw_ref)
        value = load_canonical_json(path)
        payload = _mapping_payload(value)
        if payload is None:
            continue
        if value.get("schema_version") == "task-output-commit-v1":
            try:
                loaded_artifact = load_committed_task_artifact(root, raw_ref)
            except (OSError, TypeError, ValueError) as error:
                raise _fail("INPUT_TASK_ARTIFACT_INVALID", raw_ref) from error
            loaded.append((raw_ref, loaded_artifact.payload))
        else:
            loaded.append((raw_ref, payload))
    return tuple(loaded)


def _load_one_payload(root: Path, reference: str) -> Mapping[str, object]:
    path = _logical_path(root, reference, field="artifact_ref")
    value = load_canonical_json(path)
    payload = _mapping_payload(value)
    if payload is None:
        raise _fail("ARTIFACT_ROOT_INVALID", reference)
    if value.get("schema_version") == "task-output-commit-v1":
        try:
            loaded = load_committed_task_artifact(root, reference)
        except (OSError, TypeError, ValueError) as error:
            raise _fail("TASK_ARTIFACT_INVALID", reference) from error
        return loaded.payload
    return payload


def _capture_plan(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[str, Mapping[str, object], frozenset[int], dict[int, Mapping[str, object]]]:
    candidates: list[tuple[str, Mapping[str, object]]] = []
    for ref, payload in _load_input_payloads(request, root):
        if payload.get("schema_version") == "training-endpoint-capture-plan-v1":
            candidates.append((ref, payload))
    if len(candidates) != 1:
        raise _fail("CAPTURE_PLAN_NOT_UNIQUE")
    reference, value = candidates[0]
    expected = {
        "schema_version", "plan_id", "selected_steps", "include_checkpoint_steps",
        "scope", "formal_eligible", "qualification_evidence_hash", "probe_plan_ref",
        "endpoint_metadata", "artifact_hash",
    }
    if set(value) != expected:
        raise _fail("CAPTURE_PLAN_FIELDS_MISMATCH")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if value.get("artifact_hash") != canonical_json_hash(body):
        raise _fail("CAPTURE_PLAN_HASH_MISMATCH")
    scope = value.get("scope")
    if scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
        raise _fail("REAL_CAPTURE_SCOPE_REQUIRED")
    formal_eligible = value.get("formal_eligible")
    if formal_eligible is not (scope == FORMAL_SCOPE):
        raise _fail("CAPTURE_PLAN_FORMAL_ELIGIBILITY_MISMATCH")
    if scope == PILOT_SCOPE and value.get("qualification_evidence_hash") is not None:
        raise _fail("PILOT_CAPTURE_PLAN_CANNOT_CARRY_FORMAL_EVIDENCE")
    if scope == FORMAL_SCOPE and not isinstance(value.get("qualification_evidence_hash"), str):
        raise _fail("FORMAL_CAPTURE_PLAN_EVIDENCE_REQUIRED")
    raw_steps = value.get("selected_steps")
    training = request.config.section("training")
    if not isinstance(training, Mapping) or not isinstance(training.get("max_steps"), int):
        raise _fail("TRAINING_MAX_STEPS_MISSING")
    max_steps = int(training["max_steps"])
    if (
        not isinstance(raw_steps, list)
        or not raw_steps
        or any(isinstance(step, bool) or not isinstance(step, int) or not 1 <= step <= max_steps for step in raw_steps)
        or len(set(raw_steps)) != len(raw_steps)
    ):
        raise _fail("CAPTURE_PLAN_STEPS_INVALID")
    raw_metadata = value.get("endpoint_metadata")
    if not isinstance(raw_metadata, Mapping):
        raise _fail("CAPTURE_PLAN_ENDPOINT_METADATA_REQUIRED")
    metadata: dict[int, Mapping[str, object]] = {}
    for key, item in raw_metadata.items():
        if not isinstance(key, str) or not key.isdigit() or int(key) <= 0:
            raise _fail("CAPTURE_PLAN_METADATA_STEP_INVALID")
        if not isinstance(item, Mapping) or set(item) != {"model", "seed", "stage"}:
            raise _fail("CAPTURE_PLAN_METADATA_FIELDS_MISMATCH")
        if item["model"] not in _MODEL_NAMES:
            raise _fail("CAPTURE_PLAN_METADATA_MODEL_INVALID")
        if isinstance(item["seed"], bool) or not isinstance(item["seed"], int) or item["seed"] < 0:
            raise _fail("CAPTURE_PLAN_METADATA_SEED_INVALID")
        if item["stage"] not in _STAGES:
            raise _fail("CAPTURE_PLAN_METADATA_STAGE_INVALID")
        step = int(key)
        if step > max_steps:
            raise _fail("CAPTURE_PLAN_METADATA_STEP_INVALID")
        metadata[step] = {"model": item["model"], "seed": item["seed"], "stage": item["stage"]}
    selected = frozenset(int(step) for step in raw_steps)
    # ``include_checkpoint_steps`` may add boundaries known only to the runner.
    # Requiring at least all explicitly selected steps here prevents silently
    # shrinking a frozen plan; task_runners performs the final expanded check.
    if not selected.issubset(metadata):
        raise _fail("CAPTURE_PLAN_METADATA_PLAN_COVERAGE_DRIFT")
    probe_ref = value.get("probe_plan_ref")
    if probe_ref is not None:
        probe_path = _logical_path(root, probe_ref, field="probe_plan_ref")
        if not probe_path.is_file():
            raise _fail("PROBE_PLAN_NOT_FOUND", probe_ref)
    return reference, value, selected, metadata


def _formal_evidence(
    request: TaskExecutionRequest,
    root: Path,
    *,
    scope: str,
) -> tuple[FormalExecutionEvidence | None, str | None, dict[str, GateRecord]]:
    evidence_ref = request.environment.evidence_refs.get("formal_execution")
    if evidence_ref is None:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_REQUIRED")
    try:
        payload = _load_one_payload(root, evidence_ref)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_INVALID") from error
    try:
        evidence = FormalExecutionEvidence.from_mapping(payload)
    except (TypeError, ValueError) as error:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_INVALID") from error
    if evidence.run_intent != "formal":
        raise _fail("FORMAL_EXECUTION_EVIDENCE_NOT_FORMAL")
    try:
        evidence.require_for_stage(3)
    except (TypeError, ValueError) as error:
        raise _fail("FORMAL_EXECUTION_EVIDENCE_NOT_ELIGIBLE") from error
    gates = {gate.gate_id: gate for gate in evidence.prerequisite_gates}
    for gate_id in ("stage3.G3-0", "stage3.G3-1"):
        gate = gates.get(gate_id)
        if gate is None or gate.effective_status() is not GateStatus.PASS:
            raise _fail("REQUIRED_STAGE3_GATE_MISSING", gate_id)
    return evidence, evidence_ref, gates


def _validate_scope_authority(
    request: TaskExecutionRequest,
    root: Path,
    payloads: tuple[tuple[str, Mapping[str, object]], ...],
    *,
    scope: str,
) -> tuple[str | None, str | None]:
    decisions = [(ref, p) for ref, p in payloads if p.get("schema_version") == "stage3-g30-user-scope-decision-v1"]
    gates: list[tuple[str, GateRecord]] = []
    for ref, payload in payloads:
        if payload.get("schema_version") != "gate-record-v1":
            continue
        try:
            gate = GateRecord.from_mapping(dict(payload))
        except (TypeError, ValueError) as error:
            raise _fail("GATE_RECORD_INVALID", ref) from error
        if gate.gate_id == "stage3.G3-0":
            gates.append((ref, gate))
    if len(decisions) != 1 or len(gates) != 1:
        raise _fail("G30_SCOPE_AUTHORITY_MISSING")
    decision_ref, decision = decisions[0]
    gate_ref, gate = gates[0]
    try:
        validate_stage3_scope_authority(decision, gate, decision_ref=decision_ref)
    except (TypeError, ValueError) as error:
        raise _fail("G30_SCOPE_AUTHORITY_INVALID") from error
    return decision_ref, gate.artifact_hash


def _validate_estimator_authority(
    payloads: tuple[tuple[str, Mapping[str, object]], ...],
    *,
    request: TaskExecutionRequest | None = None,
    root: Path | None = None,
) -> str:
    """Require the frozen Stage 2 U-32/B=32 authority, without relabeling it."""

    candidates: list[tuple[str, Mapping[str, object]]] = []
    for ref, payload in payloads:
        if payload.get("schema_version") == "stage3-g30-user-scope-decision-v1":
            candidates.append((ref, payload.get("accepted_stage_inputs", {})))  # type: ignore[arg-type]
        elif payload.get("schema_version") in {
            "estimator-decision-v1",
            "stage2-estimator-recommendation-v1",
            "stage2-estimator-decision-v1",
        }:
            candidates.append((ref, payload))
    if request is not None and root is not None:
        environment_ref = request.environment.estimator_decision_ref
        if environment_ref is not None:
            candidates.append((environment_ref, _load_one_payload(root, environment_ref)))
    for ref, candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        stage2 = candidate.get("stage2") if isinstance(candidate.get("stage2"), Mapping) else candidate
        if not isinstance(stage2, Mapping):
            continue
        estimator = stage2.get("default_estimator", stage2.get("selected_estimator", stage2.get("estimator")))
        batch_size = stage2.get("batch_size", stage2.get("reference_batch_size"))
        if str(estimator).upper() in {"U", "U-32", "EQUAL_U", "EQUAL-U"} and batch_size == 32:
            return ref
    raise _fail("ESTIMATOR_U32_B32_AUTHORITY_MISSING")


def _validate_offline_assets(request: TaskExecutionRequest) -> None:
    providers = request.config.section("providers")
    if not isinstance(providers, Mapping):
        raise _fail("OFFLINE_HF_PROVIDER_REQUIRED")
    if providers.get("kind") != "offline_hf" or providers.get("local_files_only") is not True or providers.get("trust_remote_code") is not False:
        raise _fail("OFFLINE_HF_PROVIDER_REQUIRED")


def _endpoint_records(
    root: Path,
    observer: object,
    *,
    scope: str,
    metadata_by_step: Mapping[int, Mapping[str, object]],
) -> tuple[tuple[int, str, str, Mapping[str, object]], ...]:
    bundles = getattr(observer, "bundles", ())
    if not isinstance(bundles, Sequence):
        raise _fail("ENDPOINT_BUNDLES_UNAVAILABLE")
    result: list[tuple[int, str, str, Mapping[str, object]]] = []
    for bundle in bundles:
        commit_ref = getattr(bundle, "commit_ref", None)
        if not isinstance(commit_ref, str):
            raise _fail("ENDPOINT_COMMIT_REF_MISSING")
        commit_path = _logical_path(root, commit_ref, field="endpoint_commit_ref")
        commit = load_canonical_json(commit_path)
        if not isinstance(commit, Mapping) or commit.get("schema_version") != "endpoint-commit-v1":
            raise _fail("ENDPOINT_COMMIT_INVALID", commit_ref)
        commit_body = {key: item for key, item in commit.items() if key != "artifact_hash"}
        if commit.get("artifact_hash") != canonical_json_hash(commit_body):
            raise _fail("ENDPOINT_COMMIT_HASH_MISMATCH", commit_ref)
        if commit.get("scope") != scope or commit.get("formal_eligible") is not (scope == FORMAL_SCOPE):
            raise _fail("ENDPOINT_SCOPE_DRIFT", commit_ref)
        object_ref = commit.get("object_ref")
        if not isinstance(object_ref, str):
            raise _fail("ENDPOINT_OBJECT_REF_MISSING", commit_ref)
        object_path = _logical_path(root, object_ref, field="endpoint_object_ref")
        value = load_canonical_json(object_path)
        if not isinstance(value, Mapping) or value.get("schema_version") != "endpoint-record-v1":
            raise _fail("ENDPOINT_OBJECT_INVALID", object_ref)
        if value.get("artifact_hash") != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
            raise _fail("ENDPOINT_OBJECT_HASH_MISMATCH", object_ref)
        record = value.get("record")
        if not isinstance(record, Mapping) or record.get("replay_verified") is not True:
            raise _fail("ENDPOINT_REPLAY_NOT_VERIFIED", object_ref)
        step = commit.get("optimizer_step")
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise _fail("ENDPOINT_STEP_INVALID", commit_ref)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping):
            raise _fail("ENDPOINT_METADATA_MISSING", object_ref)
        planned = metadata_by_step.get(step)
        if planned is None or any(metadata.get(key) != item for key, item in planned.items()):
            raise _fail("ENDPOINT_METADATA_PLAN_DRIFT", object_ref)
        diagnostics = metadata.get("step_diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise _fail("ENDPOINT_STEP_DIAGNOSTICS_MISSING", object_ref)
        required_diagnostics = {
            "raw_gradient_norm", "global_gradient_norm", "applied_optimizer_gradient_norm",
            "optimizer_gradient_norm", "clip_factor", "total_update_l2_norm",
            "data_update_l2_norm", "decay_update_l2_norm", "total_update_delta_hash",
            "data_update_delta_hash", "decay_update_delta_hash", "learning_rates",
            "learning_rate_identity", "optimizer_step_called",
        }
        if not required_diagnostics.issubset(diagnostics):
            raise _fail("ENDPOINT_STEP_DIAGNOSTICS_INCOMPLETE", object_ref)
        if record.get("full_update_delta_hash") != diagnostics.get("total_update_delta_hash"):
            raise _fail("ENDPOINT_TOTAL_DELTA_DIAGNOSTIC_MISMATCH", object_ref)
        result.append((step, commit_ref, str(commit.get("endpoint_digest")), metadata))
    return tuple(sorted(result, key=lambda row: row[0]))


@dataclass(frozen=True, slots=True)
class Stage3TrajectoryReceipt:
    """Immutable completion receipt for a replay-verified real trajectory."""

    receipt_id: str
    task_id: str
    config_hash: str
    purpose_scope: str
    formal_eligible: bool
    capture_plan_ref: str
    capture_plan_hash: str
    training_run_id: str
    selected_steps: tuple[int, ...]
    endpoint_commit_refs: tuple[str, ...]
    endpoint_digests: tuple[str, ...]
    replay_verified_steps: tuple[int, ...]
    estimator_authority_ref: str
    formal_execution_ref: str | None = None
    g30_scope_decision_ref: str | None = None
    g30_gate_hash: str | None = None
    g31_gate_hash: str | None = None
    schema_version: str = TRAJECTORY_RECEIPT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TRAJECTORY_RECEIPT_SCHEMA:
            raise _fail("TRAJECTORY_RECEIPT_SCHEMA_INVALID")
        if not isinstance(self.receipt_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", self.receipt_id
        ):
            raise _fail("TRAJECTORY_RECEIPT_ID_INVALID")
        if self.task_id != STAGE3_ENDPOINT_TASK_ID:
            raise _fail("TRAJECTORY_RECEIPT_TASK_INVALID")
        _hash(self.config_hash, field="config_hash")
        _hash(self.capture_plan_hash, field="capture_plan_hash")
        if self.purpose_scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
            raise _fail("TRAJECTORY_RECEIPT_SCOPE_INVALID")
        if type(self.formal_eligible) is not bool or self.formal_eligible != (self.purpose_scope == FORMAL_SCOPE):
            raise _fail("TRAJECTORY_RECEIPT_ELIGIBILITY_INVALID")
        if not self.capture_plan_ref or not self.training_run_id or not self.estimator_authority_ref:
            raise _fail("TRAJECTORY_RECEIPT_REFERENCE_MISSING")
        steps = tuple(self.selected_steps)
        replay_steps = tuple(self.replay_verified_steps)
        if not steps or any(isinstance(step, bool) or not isinstance(step, int) or step <= 0 for step in steps):
            raise _fail("TRAJECTORY_RECEIPT_STEPS_INVALID")
        if len(set(steps)) != len(steps) or replay_steps != steps:
            raise _fail("TRAJECTORY_RECEIPT_REPLAY_COVERAGE_INVALID")
        refs = tuple(self.endpoint_commit_refs)
        digests = tuple(self.endpoint_digests)
        if len(refs) != len(steps) or len(digests) != len(steps) or any(not isinstance(ref, str) or not ref for ref in refs):
            raise _fail("TRAJECTORY_RECEIPT_ENDPOINT_COVERAGE_INVALID")
        for digest in digests:
            _hash(digest, field="endpoint_digest")
        for field in ("g30_gate_hash", "g31_gate_hash"):
            value = getattr(self, field)
            if value is not None:
                _hash(value, field=field)
        if self.formal_eligible and (
            not isinstance(self.formal_execution_ref, str)
            or not isinstance(self.g30_scope_decision_ref, str)
            or self.g30_gate_hash is None
            or self.g31_gate_hash is None
        ):
            raise _fail("FORMAL_TRAJECTORY_RECEIPT_AUTHORITY_MISSING")
        object.__setattr__(self, "selected_steps", steps)
        object.__setattr__(self, "replay_verified_steps", replay_steps)
        object.__setattr__(self, "endpoint_commit_refs", refs)
        object.__setattr__(self, "endpoint_digests", digests)

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "purpose_scope": self.purpose_scope,
            "formal_eligible": self.formal_eligible,
            "capture_plan_ref": self.capture_plan_ref,
            "capture_plan_hash": self.capture_plan_hash,
            "training_run_id": self.training_run_id,
            "selected_steps": list(self.selected_steps),
            "endpoint_commit_refs": list(self.endpoint_commit_refs),
            "endpoint_digests": list(self.endpoint_digests),
            "replay_verified_steps": list(self.replay_verified_steps),
            "estimator_authority_ref": self.estimator_authority_ref,
            "formal_execution_ref": self.formal_execution_ref,
            "g30_scope_decision_ref": self.g30_scope_decision_ref,
            "g30_gate_hash": self.g30_gate_hash,
            "g31_gate_hash": self.g31_gate_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        return self._payload() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3TrajectoryReceipt":
        expected = {
            "schema_version", "receipt_id", "task_id", "config_hash", "purpose_scope",
            "formal_eligible", "capture_plan_ref", "capture_plan_hash", "training_run_id",
            "selected_steps", "endpoint_commit_refs", "endpoint_digests", "replay_verified_steps",
            "estimator_authority_ref", "formal_execution_ref", "g30_scope_decision_ref",
            "g30_gate_hash", "g31_gate_hash", "artifact_hash",
        }
        if set(value) != expected or value.get("schema_version") != TRAJECTORY_RECEIPT_SCHEMA:
            raise _fail("TRAJECTORY_RECEIPT_FIELDS_MISMATCH")
        receipt = cls(
            receipt_id=str(value["receipt_id"]), task_id=str(value["task_id"]),
            config_hash=str(value["config_hash"]), purpose_scope=str(value["purpose_scope"]),
            formal_eligible=value["formal_eligible"], capture_plan_ref=str(value["capture_plan_ref"]),
            capture_plan_hash=str(value["capture_plan_hash"]), training_run_id=str(value["training_run_id"]),
            selected_steps=tuple(value["selected_steps"]), endpoint_commit_refs=tuple(value["endpoint_commit_refs"]),
            endpoint_digests=tuple(value["endpoint_digests"]), replay_verified_steps=tuple(value["replay_verified_steps"]),
            estimator_authority_ref=str(value["estimator_authority_ref"]),
            formal_execution_ref=value["formal_execution_ref"],
            g30_scope_decision_ref=value["g30_scope_decision_ref"], g30_gate_hash=value["g30_gate_hash"],
            g31_gate_hash=value["g31_gate_hash"],
        )
        if value.get("artifact_hash") != receipt.artifact_hash:
            raise _fail("TRAJECTORY_RECEIPT_HASH_MISMATCH")
        return receipt


@dataclass(slots=True)
class Stage3TrajectoryProducer:
    """Run real stage3.03 training and publish only a trajectory receipt."""

    workspace_root: Path
    training_runner: TrainingTaskRunner | None = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()

    def run(
        self,
        request: TaskExecutionRequest,
        *,
        receipt_path: str | Path | None = None,
    ) -> Stage3TrajectoryReceipt:
        if request.task.task_id != STAGE3_ENDPOINT_TASK_ID:
            raise _fail("STAGE3_TRAJECTORY_TASK_REQUIRED")
        task = DEFAULT_TASK_CATALOG.get(STAGE3_ENDPOINT_TASK_ID)
        if (
            request.task.task_id != task.task_id
            or request.task.stage != 3
            or request.task.runner_kind is not RunnerKind.PATH_INTEGRATION
        ):
            raise _fail("STAGE3_TRAJECTORY_TASK_INVALID")
        _validate_offline_assets(request)
        plan_ref, plan, selected, metadata = _capture_plan(request, self.workspace_root)
        scope = str(plan["scope"])
        if scope == FORMAL_SCOPE and request.config.run_intent != "formal":
            raise _fail("FORMAL_TRAJECTORY_REQUIRES_FORMAL_RUN_INTENT")
        if scope == PILOT_SCOPE and request.config.run_intent not in {"local_fixture", "formal"}:
            raise _fail("PILOT_TRAJECTORY_RUN_INTENT_INVALID")
        evidence, evidence_ref, evidence_gates = _formal_evidence(request, self.workspace_root, scope=scope)
        payloads = _load_input_payloads(request, self.workspace_root)
        scope_decision_ref, g30_gate_hash = _validate_scope_authority(
            request, self.workspace_root, payloads, scope=scope
        )
        estimator_ref = _validate_estimator_authority(
            payloads, request=request, root=self.workspace_root
        )
        runner = self.training_runner or TrainingTaskRunner(self.workspace_root)
        # Deliberately call the lower-level real training entry point.  Calling
        # runner.run() would publish the stage3.03 top-level TaskArtifacts.
        training_result, engine, _events_path, _assets, _evals, _profiles = runner._run_training(request)
        if getattr(training_result, "status", None) != "COMPLETE":
            raise _fail("TRAINING_NOT_COMPLETE")
        observers = tuple(getattr(engine, "observers", ()))
        endpoint_observers = tuple(
            observer
            for observer in observers
            if isinstance(observer, TrainingEndpointObserver)
            or observer.__class__.__name__ == "TrainingEndpointObserver"
        )
        if len(endpoint_observers) != 1:
            raise _fail("TRAINING_ENDPOINT_OBSERVER_MISSING")
        rows = _endpoint_records(
            self.workspace_root,
            endpoint_observers[0],
            scope=scope,
            metadata_by_step=metadata,
        )
        expected_steps = set(selected)
        # Include-checkpoint expansion is reflected by the observer's selected plan;
        # coverage must still be exact at completion.
        actual_steps = {row[0] for row in rows}
        if not expected_steps.issubset(actual_steps) or len(actual_steps) != len(rows):
            raise _fail("ENDPOINT_CAPTURE_COVERAGE_DRIFT")
        if set(metadata) != actual_steps:
            raise _fail("ENDPOINT_CAPTURE_METADATA_COVERAGE_DRIFT")
        receipt_seed = canonical_json_hash(
            {
                "task_id": request.task.task_id,
                "config_hash": request.config.config_hash,
                "scope": scope,
                "plan_hash": plan["artifact_hash"],
                "endpoint_digests": [row[2] for row in rows],
            }
        )
        receipt = Stage3TrajectoryReceipt(
            receipt_id=f"stage3-trajectory-{receipt_seed[:32]}",
            task_id=request.task.task_id,
            config_hash=request.config.config_hash,
            purpose_scope=scope,
            formal_eligible=scope == FORMAL_SCOPE,
            capture_plan_ref=plan_ref,
            capture_plan_hash=str(plan["artifact_hash"]),
            training_run_id=str(training_result.run_id),
            selected_steps=tuple(row[0] for row in rows),
            endpoint_commit_refs=tuple(row[1] for row in rows),
            endpoint_digests=tuple(row[2] for row in rows),
            replay_verified_steps=tuple(row[0] for row in rows),
            estimator_authority_ref=estimator_ref,
            formal_execution_ref=evidence_ref,
            g30_scope_decision_ref=scope_decision_ref,
            g30_gate_hash=g30_gate_hash,
            g31_gate_hash=(evidence_gates.get("stage3.G3-1").artifact_hash if evidence_gates.get("stage3.G3-1") else None),
        )
        target = (
            _logical_path(self.workspace_root, receipt_path, field="receipt_path")
            if receipt_path is not None
            else self.workspace_root / "stage3-trajectory" / "receipts" / f"{receipt.receipt_id}.json"
        )
        publish_canonical_immutable(target, receipt.to_dict())
        return receipt


def run_stage3_trajectory(
    request: TaskExecutionRequest,
    *,
    workspace_root: str | Path,
    receipt_path: str | Path | None = None,
    training_runner: TrainingTaskRunner | None = None,
) -> Stage3TrajectoryReceipt:
    return Stage3TrajectoryProducer(Path(workspace_root), training_runner).run(
        request, receipt_path=receipt_path
    )


produce_stage3_trajectory = run_stage3_trajectory


def load_stage3_trajectory_receipt(path: str | Path) -> Stage3TrajectoryReceipt:
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise _fail("TRAJECTORY_RECEIPT_ROOT_INVALID")
    return Stage3TrajectoryReceipt.from_mapping(value)


validate_stage3_trajectory_receipt = load_stage3_trajectory_receipt
Stage3EndpointTrajectoryProducer = Stage3TrajectoryProducer
TrainingEndpointTrajectoryProducer = Stage3TrajectoryProducer
Stage3TrajectoryCompletionReceipt = Stage3TrajectoryReceipt


__all__ = [
    "STAGE3_ENDPOINT_TASK_ID",
    "TRAJECTORY_RECEIPT_SCHEMA",
    "Stage3TrajectoryError",
    "Stage3TrajectoryProducer",
    "Stage3TrajectoryReceipt",
    "Stage3EndpointTrajectoryProducer",
    "TrainingEndpointTrajectoryProducer",
    "Stage3TrajectoryCompletionReceipt",
    "load_stage3_trajectory_receipt",
    "produce_stage3_trajectory",
    "run_stage3_trajectory",
    "validate_stage3_trajectory_receipt",
]
