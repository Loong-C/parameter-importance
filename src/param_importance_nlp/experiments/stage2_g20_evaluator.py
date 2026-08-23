"""Independent, fail-closed qualification of the Stage 2 G2.0 Gate.

``stage2.01_scope_hypotheses_and_preregistration`` deliberately emits a
``stage23-task-gate-candidate-v1`` object whose status is ``NOT_RUN``.  That
candidate is useful lineage evidence, but it is not a scientific Gate.  This
module is the separate consumer: it accepts only the three immutable formal
task commits, re-reads their payloads, re-runs the frozen S2.1 validators and
publishes a real :class:`~param_importance_nlp.contracts.status.GateRecord`.

The public entry point takes references only.  It has no status, metric,
threshold, or ``formal_eligible`` override, so callers cannot self-sign a
PASS or supply numbers that become Gate evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import re
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping, Sequence

from ..contracts.jsonio import JSONValue, canonical_json_hash
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    PublishedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from .preregistration import (
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
    validate_stage2_hypothesis_contract,
    validate_stage2_preregistration,
)


SCHEMA_VERSION = "stage2-g20-evaluation-v1"
GATE_ID = "stage2.G2.0"
TASK_ID = "stage2.01_scope_hypotheses_and_preregistration"
ARTIFACT_KINDS: tuple[str, ...] = (
    "preregistration",
    "hypothesis_contract",
    "gate_record",
)
PLAN_PATH = "plan/stage2/01_scope_hypotheses_and_preregistration.md"
MATHEMATICS_PATH = "docs/mathematics.md"
STAGE1_REPORT_PATH = (
    "reports/stage1/cpu-evidence-20260814-s12-r2/"
    "stage1.11_reporting_and_exit_gate/stage_report.json"
)
FROZEN_PLAN_SHA256 = "9af31ee0b82cbb0526817c7bb741ca708e33f96e7be3dda65d54b188e942fe12"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# This is an evaluator identity, not a caller-controlled scientific threshold.
# It binds the output envelope to the evaluator's frozen source/plan policy even
# when no ResolvedConfig file is available to the materializer.
EVALUATION_CONFIG: Mapping[str, JSONValue] = {
    "schema_version": SCHEMA_VERSION,
    "gate_id": GATE_ID,
    "task_id": TASK_ID,
    "artifact_kinds": list(ARTIFACT_KINDS),
    "plan_path": PLAN_PATH,
    "plan_sha256": FROZEN_PLAN_SHA256,
    "mathematics_path": MATHEMATICS_PATH,
    "stage1_report_path": STAGE1_REPORT_PATH,
    "runner_candidate_rule": "candidate_must_be_NOT_RUN_and_formal_eligible_false",
    "decision_rule": "PASS_only_after_rebuilt_preregistration_and_hypothesis_contract_match",
}
EVALUATION_CONFIG_HASH = canonical_json_hash(EVALUATION_CONFIG)


class G20Blocked(ValueError):
    """An unavailable, malformed, tampered, or unsafe formal input."""


class G20Failed(ValueError):
    """A readable formal input that violates the frozen S2.1 contract."""


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise G20Blocked(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: object, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise G20Blocked(f"{field}:PRODUCER_COMMIT_REQUIRED")
    return value


def _logical_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G20Blocked(f"{field}:LOGICAL_PATH_REQUIRED")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G20Blocked(f"{field}:PATH_ESCAPE")
    return parsed.as_posix()


def _reject_symlink_chain(root: Path, logical: str, field: str) -> None:
    current = root
    try:
        for part in PurePosixPath(logical).parts:
            current = current / part
            if current.is_symlink():
                raise G20Blocked(f"{field}:SYMLINK_FORBIDDEN")
    except OSError as error:
        raise G20Blocked(f"{field}:UNREADABLE") from error


def _resolve(root: Path, value: object, field: str) -> Path:
    logical = _logical_path(value, field)
    _reject_symlink_chain(root, logical, field)
    target = (root / Path(logical)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise G20Blocked(f"{field}:PATH_ESCAPE") from error
    return target


def _file_sha256(path: Path, field: str) -> str:
    try:
        if not path.is_file():
            raise G20Blocked(f"{field}:FILE_REQUIRED")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except G20Blocked:
        raise
    except (OSError, ValueError) as error:
        raise G20Blocked(f"{field}:UNREADABLE") from error


def _normalise_refs(value: object) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        if set(value) != set(ARTIFACT_KINDS):
            raise G20Blocked("artifact_refs:EXACT_THREE_KINDS_REQUIRED")
        raw = [value[kind] for kind in ARTIFACT_KINDS]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = list(value)
    else:
        raise G20Blocked("artifact_refs:ARRAY_OR_KIND_MAPPING_REQUIRED")
    if len(raw) != len(ARTIFACT_KINDS):
        raise G20Blocked("artifact_refs:EXACTLY_THREE_REQUIRED")
    refs = tuple(_logical_path(item, f"artifact_refs[{index}]") for index, item in enumerate(raw))
    if len(set(refs)) != len(refs):
        raise G20Blocked("artifact_refs:DUPLICATE_COMMIT_REF")
    return refs


def _load_source_refs(root: Path, refs: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(_logical_path(item, f"source_refs[{index}]") for index, item in enumerate(refs))
    if not normalised:
        raise G20Blocked("source_refs:REQUIRED")
    if len(normalised) != len(set(normalised)):
        raise G20Blocked("source_refs:DUPLICATE")
    for index, reference in enumerate(normalised):
        path = _resolve(root, reference, f"source_refs[{index}]")
        if not path.is_file():
            raise G20Blocked(f"source_refs[{index}]:FILE_REQUIRED")
    return normalised


def _load_committed(root: Path, ref: str) -> LoadedTaskArtifact:
    _resolve(root, ref, "artifact_commit_ref")
    try:
        return load_committed_task_artifact(root, ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise G20Blocked(f"artifact_commit_ref:INVALID:{type(error).__name__}") from error


@dataclass(frozen=True, slots=True)
class _LoadedSet:
    refs_by_kind: Mapping[str, str]
    artifacts_by_kind: Mapping[str, LoadedTaskArtifact]
    config_hash: str
    source_refs: tuple[str, ...]


def _load_and_bind(root: Path, references: object) -> _LoadedSet:
    refs = _normalise_refs(references)
    loaded: list[tuple[str, LoadedTaskArtifact, str]] = []
    for ref in refs:
        item = _load_committed(root, ref)
        identity = item.identity
        if identity.task_id != TASK_ID:
            raise G20Blocked("artifact_identity:WRONG_TASK_ID")
        if identity.artifact_kind not in ARTIFACT_KINDS:
            raise G20Blocked("artifact_identity:WRONG_ARTIFACT_KIND")
        if item.run_intent != "formal" or identity.formal_eligible is not True:
            raise G20Blocked("artifact_identity:FORMAL_ENVELOPE_REQUIRED")
        loaded.append((identity.artifact_kind, item, ref))
    if {kind for kind, _, _ in loaded} != set(ARTIFACT_KINDS):
        raise G20Blocked("artifact_identity:COMPLETE_ARTIFACT_SET_REQUIRED")
    if len({kind for kind, _, _ in loaded}) != len(ARTIFACT_KINDS):
        raise G20Blocked("artifact_identity:DUPLICATE_ARTIFACT_KIND")
    configs = {item.identity.config_hash for _, item, _ in loaded}
    if len(configs) != 1:
        raise G20Blocked("artifact_identity:MIXED_CONFIG_HASH")
    config_hash = next(iter(configs))
    if _SHA256.fullmatch(config_hash) is None:
        raise G20Blocked("artifact_identity:CONFIG_HASH_INVALID")
    source_sets = {item.source_refs for _, item, _ in loaded}
    if len(source_sets) != 1:
        raise G20Blocked("artifact_identity:SOURCE_REFS_NOT_COMMON")
    source_refs = _load_source_refs(root, next(iter(source_sets)))
    refs_by_kind = {kind: ref for kind, _, ref in loaded}
    artifacts_by_kind = {kind: item for kind, item, _ in loaded}
    return _LoadedSet(refs_by_kind, artifacts_by_kind, config_hash, source_refs)


def _validate_document_bindings(root: Path, prereg: Mapping[str, Any]) -> None:
    provenance = prereg.get("provenance")
    if not isinstance(provenance, Mapping):
        raise G20Blocked("preregistration:PROVENANCE_REQUIRED")
    producer = _commit(provenance.get("producer_commit"), "provenance.producer_commit")
    if not producer:
        raise G20Blocked("provenance.producer_commit:EMPTY")
    seed = _sha(provenance.get("seed_plan_hash"), "provenance.seed_plan_hash")
    upstream = _sha(provenance.get("upstream_binding_hash"), "provenance.upstream_binding_hash")
    mathematics_hash = _sha(provenance.get("mathematics_hash"), "provenance.mathematics_hash")
    stage1_hash = _sha(provenance.get("stage1_report_hash"), "provenance.stage1_report_hash")
    if provenance.get("mathematics_path") != MATHEMATICS_PATH:
        raise G20Blocked("provenance.mathematics_path:FROZEN_PATH_MISMATCH")
    if provenance.get("stage1_report_path") != STAGE1_REPORT_PATH:
        raise G20Blocked("provenance.stage1_report_path:FROZEN_PATH_MISMATCH")
    if _file_sha256(_resolve(root, MATHEMATICS_PATH, "mathematics_path"), "mathematics_path") != mathematics_hash:
        raise G20Blocked("provenance.mathematics_hash:CONTENT_MISMATCH")
    if _file_sha256(_resolve(root, STAGE1_REPORT_PATH, "stage1_report_path"), "stage1_report_path") != stage1_hash:
        raise G20Blocked("provenance.stage1_report_hash:CONTENT_MISMATCH")
    if _file_sha256(_resolve(root, PLAN_PATH, "plan_path"), "plan_path") != FROZEN_PLAN_SHA256:
        raise G20Blocked("plan:CONTENT_MISMATCH")
    # Keep these accesses explicit: a future builder change must not silently
    # stop binding one of the required provenance fields.
    _ = seed, upstream


def _validate_preregistration(root: Path, value: Mapping[str, Any]) -> None:
    try:
        validate_stage2_preregistration(value)
    except (TypeError, ValueError) as error:
        raise G20Failed(f"preregistration:VALIDATOR_REJECTED:{error}") from error
    if value.get("scope") != "formal" or value.get("formal_eligible") is not False:
        raise G20Failed("preregistration:FORMAL_SCOPE_REQUIRED")
    _validate_document_bindings(root, value)
    provenance = value["provenance"]
    assert isinstance(provenance, Mapping)
    try:
        rebuilt = build_stage2_preregistration(
            seed_plan_hash=str(provenance["seed_plan_hash"]),
            producer_commit=str(provenance["producer_commit"]),
            mathematics_hash=str(provenance["mathematics_hash"]),
            stage1_report_hash=str(provenance["stage1_report_hash"]),
            upstream_binding_hash=str(provenance["upstream_binding_hash"]),
            scope="formal",
        )
    except (TypeError, ValueError) as error:
        raise G20Failed(f"preregistration:REBUILD_FAILED:{error}") from error
    if dict(value) != rebuilt:
        raise G20Failed("preregistration:FROZEN_CONTENT_MISMATCH")


def _validate_hypothesis(value: Mapping[str, Any], preregistration: Mapping[str, Any]) -> None:
    try:
        validate_stage2_hypothesis_contract(value, preregistration=preregistration)
    except (TypeError, ValueError) as error:
        raise G20Failed(f"hypothesis_contract:VALIDATOR_REJECTED:{error}") from error
    provenance = preregistration.get("provenance")
    assert isinstance(provenance, Mapping)
    expected = build_stage2_hypothesis_contract(
        preregistration,
        upstream_binding_hash=str(provenance["upstream_binding_hash"]),
    )
    if dict(value) != expected:
        raise G20Failed("hypothesis_contract:FROZEN_CONTENT_MISMATCH")


def _validate_runner_candidate(value: Mapping[str, Any], preregistration: Mapping[str, Any], hypothesis: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version", "task_id", "gate_ids", "gate_status",
        "local_validation_status", "formal_eligible", "reason", "gate_id",
        "preregistration_hash", "hypothesis_contract_hash", "quality_gate_status",
        "sample_generation_status",
    }
    if set(value) != expected_fields:
        raise G20Failed("gate_candidate:FIELDS_INVALID")
    if value.get("schema_version") != "stage23-task-gate-candidate-v1":
        raise G20Failed("gate_candidate:SCHEMA_INVALID")
    if value.get("task_id") != TASK_ID or value.get("gate_id") != GATE_ID:
        raise G20Failed("gate_candidate:IDENTITY_MISMATCH")
    if value.get("gate_ids") != ["stage1.G1-EXIT"]:
        raise G20Failed("gate_candidate:REQUIRED_GATES_MISMATCH")
    # This is the only role of the runner candidate.  A candidate PASS (or a
    # caller-authored metric hidden in an extra field) is never accepted.
    if value.get("gate_status") != "NOT_RUN":
        raise G20Failed("gate_candidate:SELF_SIGNED_STATUS_REJECTED")
    if value.get("local_validation_status") != "NOT_RUN":
        raise G20Failed("gate_candidate:LOCAL_VALIDATION_NOT_ALLOWED")
    if value.get("formal_eligible") is not False:
        raise G20Failed("gate_candidate:FORMAL_ELIGIBLE_MUST_BE_FALSE")
    if value.get("reason") != "formal_gate_requires_independent_review":
        raise G20Failed("gate_candidate:REASON_MISMATCH")
    if value.get("quality_gate_status") != "NOT_RUN" or value.get("sample_generation_status") != "FORBIDDEN_UNTIL_COMMITTED":
        raise G20Failed("gate_candidate:RUNNER_NUMERIC_OR_QUALITY_CLAIM_REJECTED")
    if value.get("preregistration_hash") != preregistration.get("preregistration_hash"):
        raise G20Failed("gate_candidate:PREREGISTRATION_HASH_MISMATCH")
    if value.get("hypothesis_contract_hash") != hypothesis.get("hypothesis_contract_hash"):
        raise G20Failed("gate_candidate:HYPOTHESIS_HASH_MISMATCH")


def _evaluator_provenance() -> tuple[str, str]:
    source_path = Path(__file__).resolve()
    source_hash = _file_sha256(source_path, "evaluator_source")
    repository_root = source_path.parents[3]
    try:
        producer = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        raise G20Blocked("evaluator_producer_commit:UNRESOLVED") from error
    return _commit(producer, "evaluator_producer_commit"), source_hash


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_evidence_refs(references: object) -> tuple[str, ...]:
    try:
        refs = _normalise_refs(references)
    except G20Blocked:
        return ()
    return tuple(refs)


def _make_gate(
    *,
    status: GateStatus,
    reasons: Sequence[str],
    evidence_refs: Sequence[str],
    loaded: _LoadedSet | None,
    evaluator_commit: str | None,
    source_hash: str | None,
) -> GateRecord:
    config_hash = None if loaded is None else loaded.config_hash
    prereg_hash = None
    hypothesis_hash = None
    candidate_status = None
    if loaded is not None:
        prereg_hash = loaded.artifacts_by_kind["preregistration"].payload.get("preregistration_hash")
        hypothesis_hash = loaded.artifacts_by_kind["hypothesis_contract"].payload.get("hypothesis_contract_hash")
        candidate_status = loaded.artifacts_by_kind["gate_record"].payload.get("gate_status")
    measured: dict[str, JSONValue] = {
        "input_artifact_count": 3 if loaded is not None else 0,
        "task_id": TASK_ID,
        "artifact_kinds": list(ARTIFACT_KINDS),
        "config_hash": config_hash,
        "preregistration_hash": prereg_hash,
        "hypothesis_contract_hash": hypothesis_hash,
        "runner_candidate_status": candidate_status,
        "runner_candidate_role": "non_self_signed_lineage_only",
        "evaluator": {
            "producer_commit": evaluator_commit,
            "source_sha256": source_hash,
            "evaluation_config_hash": EVALUATION_CONFIG_HASH,
            "schema_version": SCHEMA_VERSION,
        },
    }
    threshold: dict[str, JSONValue] = {
        "contract": "frozen_s2_1_validators_and_document_hash_bindings",
        "required_status": "PASS",
        "runner_candidate": "NOT_RUN_only",
        "scientific_numbers": "not_supplied_by_caller_or_runner_candidate",
    }
    return GateRecord(
        gate_id=GATE_ID,
        stage=2,
        status=status,
        checked_at=_timestamp(),
        measured=measured,
        threshold=threshold,
        evidence_refs=tuple(evidence_refs),
        reasons=tuple(reasons),
    )


def _output_logical_dir(root: Path, output_root: str | Path | None, output_dir: str | Path) -> str:
    if output_root is None:
        candidate = Path(output_dir)
        if candidate.is_absolute():
            candidate = candidate.resolve()
            try:
                base = PurePosixPath(candidate.relative_to(root).as_posix())
            except ValueError as error:
                raise G20Blocked("output_dir:PATH_ESCAPE") from error
            if base.is_absolute() or any(part in {"", ".", ".."} for part in base.parts):
                raise G20Blocked("output_dir:PATH_ESCAPE")
        else:
            base = PurePosixPath(_logical_path(str(output_dir), "output_dir"))
    else:
        candidate = Path(output_root)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            base = PurePosixPath(candidate.relative_to(root).as_posix())
        except ValueError as error:
            raise G20Blocked("output_root:PATH_ESCAPE") from error
        if base.is_absolute() or any(part in {"", ".", ".."} for part in base.parts):
            raise G20Blocked("output_root:PATH_ESCAPE")
    return base.as_posix()


def _publish(
    root: Path,
    *,
    base_dir: str,
    attempt_id: str,
    gate: GateRecord,
    loaded: _LoadedSet,
) -> PublishedTaskArtifact:
    store = TaskArtifactStore(root, f"{base_dir}/g2.0-attempts/{attempt_id}")
    return store.publish(
        task_id=TASK_ID,
        artifact_kind="gate_record",
        config_hash=loaded.config_hash,
        run_intent="formal",
        payload=gate.to_dict(),
        formal_eligible=True,
        source_refs=tuple(loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS),
    )


def evaluate_formal_g20(
    workspace_root: str | Path,
    artifact_refs: Sequence[str] | Mapping[str, str],
    *,
    output_root: str | Path | None = None,
    output_dir: str | Path = "runs/stage2-g20-evaluation",
) -> dict[str, JSONValue]:
    """Evaluate and publish G2.0 from exactly three formal task commits.

    The returned mapping always contains ``gate_record``.  For a complete
    formal input set it also contains the immutable envelope ``commit_ref``;
    malformed/unsafe input is returned as ``BLOCKED`` without publishing any
    misleading formal envelope.  No caller value can set status or metrics.
    """

    root = Path(workspace_root).resolve()
    loaded: _LoadedSet | None = None
    evaluator_commit: str | None = None
    source_hash: str | None = None
    status = GateStatus.BLOCKED
    reasons: list[str] = []
    try:
        evaluator_commit, source_hash = _evaluator_provenance()
        loaded = _load_and_bind(root, artifact_refs)
        prereg = loaded.artifacts_by_kind["preregistration"].payload
        hypothesis = loaded.artifacts_by_kind["hypothesis_contract"].payload
        candidate = loaded.artifacts_by_kind["gate_record"].payload
        _validate_preregistration(root, prereg)
        _validate_hypothesis(hypothesis, prereg)
        _validate_runner_candidate(candidate, prereg, hypothesis)
        status = GateStatus.PASS
    except G20Failed as error:
        status = GateStatus.FAIL
        reasons.append(str(error))
    except (G20Blocked, OSError, TypeError, ValueError, KeyError) as error:
        status = GateStatus.BLOCKED
        reasons.append(str(error))
    if not reasons and status is not GateStatus.PASS:
        reasons.append("G2.0 formal qualification did not complete")
    evidence_refs = (
        tuple(loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS)
        if loaded is not None
        else _safe_evidence_refs(artifact_refs)
    )
    try:
        gate = _make_gate(
            status=status,
            reasons=reasons,
            evidence_refs=evidence_refs,
            loaded=loaded,
            evaluator_commit=evaluator_commit,
            source_hash=source_hash,
        )
    except Exception as error:
        # GateRecord itself is the final contract boundary.  If provenance is
        # somehow unavailable, return a valid BLOCKED record rather than leak a
        # partially constructed/fake PASS.
        gate = GateRecord(
            gate_id=GATE_ID,
            stage=2,
            status=GateStatus.BLOCKED,
            checked_at=_timestamp(),
            measured={"evaluator": {"evaluation_config_hash": EVALUATION_CONFIG_HASH}},
            threshold={"contract": "frozen_s2_1_validators"},
            evidence_refs=(),
            reasons=(f"G2.0 GateRecord construction failed closed: {type(error).__name__}",),
        )
        status = GateStatus.BLOCKED
        loaded = None
    published: PublishedTaskArtifact | None = None
    if loaded is not None:
        try:
            base_dir = _output_logical_dir(root, output_root, output_dir)
            attempt_id = canonical_json_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "artifact_refs": [loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS],
                    "config_hash": loaded.config_hash,
                    "evaluation_config_hash": EVALUATION_CONFIG_HASH,
                }
            )
            published = _publish(
                root,
                base_dir=base_dir,
                attempt_id=attempt_id,
                gate=gate,
                loaded=loaded,
            )
        except (G20Blocked, OSError, TypeError, ValueError, RuntimeError) as error:
            # A qualification result is not complete until the formal envelope
            # is committed and re-discoverable.  Publishing failure therefore
            # downgrades even a calculated PASS to BLOCKED.
            status = GateStatus.BLOCKED
            reasons = [*gate.reasons, f"formal envelope publish failed: {type(error).__name__}"]
            gate = _make_gate(
                status=status,
                reasons=reasons,
                evidence_refs=evidence_refs,
                loaded=loaded,
                evaluator_commit=evaluator_commit,
                source_hash=source_hash,
            )
            published = None
    result: dict[str, JSONValue] = {
        "schema_version": SCHEMA_VERSION,
        "gate_record": gate.to_dict(),
        "status": gate.status.value,
        "formal_eligible": gate.status is GateStatus.PASS and published is not None,
        "evaluation_config_hash": EVALUATION_CONFIG_HASH,
        "source_refs": list(evidence_refs),
        "commit_ref": None if published is None else published.commit_ref,
        "envelope_artifact_hash": None if published is None else published.artifact_hash,
    }
    return result


def evaluate_g20(
    workspace_root: str | Path,
    artifact_refs: Sequence[str] | Mapping[str, str],
    **kwargs: object,
) -> dict[str, JSONValue]:
    """Short compatibility alias used by materializers."""

    return evaluate_formal_g20(workspace_root, artifact_refs, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "ARTIFACT_KINDS",
    "EVALUATION_CONFIG",
    "EVALUATION_CONFIG_HASH",
    "FROZEN_PLAN_SHA256",
    "G20Blocked",
    "G20Failed",
    "GATE_ID",
    "SCHEMA_VERSION",
    "TASK_ID",
    "evaluate_formal_g20",
    "evaluate_g20",
]
