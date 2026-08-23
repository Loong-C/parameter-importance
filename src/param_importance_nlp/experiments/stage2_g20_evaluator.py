"""Independent, fail-closed qualification of the Stage 2 G2.0 Gate.

The S2.1 runner publishes a ``stage23-task-gate-candidate-v1`` object.  That
object is lineage only: it is required to remain ``NOT_RUN`` and can never
sign the scientific Gate.  This module is the independent consumer of the
three immutable S2.1 task commits.

There are deliberately two roots in the public API:

* ``data_root`` contains TaskArtifact commits/objects, the persisted
  ``ResolvedConfigV2`` reference, and evaluator output;
* ``repository_root`` contains the frozen plan, mathematical/report sources,
  and the evaluator/validator/runner source used for identity binding.

No caller-provided status, metric, threshold, or formal flag is accepted.
Every path is logical and symlink-free, and a formal envelope is published only
after the complete input/source/config chain has been reloaded and validated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..contracts.seed import SeedPlan
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


SCHEMA_VERSION = "stage2-g20-evaluation-v2"
GATE_ID = "stage2.G2.0"
TASK_ID = "stage2.01_scope_hypotheses_and_preregistration"
ARTIFACT_KINDS: tuple[str, ...] = (
    "preregistration",
    "hypothesis_contract",
    "gate_record",
)

STAGE1_TASK_ID = "stage1.11_reporting_and_exit_gate"
STAGE1_GATE_ID = "G1-EXIT"
STAGE1_ARTIFACT_KINDS: tuple[str, ...] = (
    "stage_report",
    "requirements_matrix",
    "gate_summary",
    "delivery_manifest",
)
STAGE1_PAYLOAD_SCHEMAS = {
    "stage_report": "stage1-s1-11-stage-report-v1",
    "requirements_matrix": "stage1-s1-11-requirements-matrix-v1",
    "gate_summary": "stage1-s1-11-gate-summary-v1",
    "delivery_manifest": "stage1-s1-11-delivery-manifest-v1",
}
STAGE1_LEGACY_PAYLOAD_SCHEMA = "stage01-task-evidence-v1"

PLAN_PATH = "plan/stage2/01_scope_hypotheses_and_preregistration.md"
MATHEMATICS_PATH = "docs/mathematics.md"
STAGE1_REPORT_PATH = (
    "reports/stage1/cpu-evidence-20260814-s12-r2/"
    "stage1.11_reporting_and_exit_gate/stage_report.json"
)
FROZEN_PLAN_SHA256 = "9af31ee0b82cbb0526817c7bb741ca708e33f96e7be3dda65d54b188e942fe12"

EVALUATOR_SOURCE_PATH = "src/param_importance_nlp/experiments/stage2_g20_evaluator.py"
PREREGISTRATION_SOURCE_PATH = "src/param_importance_nlp/experiments/preregistration.py"
RUNNER_SOURCE_PATH = "src/param_importance_nlp/experiments/stage23_task_runners.py"
REPOSITORY_SOURCE_PATHS: tuple[str, ...] = (
    EVALUATOR_SOURCE_PATH,
    PREREGISTRATION_SOURCE_PATH,
    RUNNER_SOURCE_PATH,
    PLAN_PATH,
    MATHEMATICS_PATH,
    STAGE1_REPORT_PATH,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")

# Kept as a public policy template for callers that imported it in the first
# evaluator release.  The actual hash used by a Gate is produced by
# ``_repository_identity`` and includes all source bytes and their HEAD.
EVALUATION_CONFIG: Mapping[str, JSONValue] = {
    "schema_version": SCHEMA_VERSION,
    "gate_id": GATE_ID,
    "task_id": TASK_ID,
    "artifact_kinds": list(ARTIFACT_KINDS),
    "stage1_task_id": STAGE1_TASK_ID,
    "stage1_artifact_kinds": list(STAGE1_ARTIFACT_KINDS),
    "plan_path": PLAN_PATH,
    "frozen_plan_sha256": FROZEN_PLAN_SHA256,
    "mathematics_path": MATHEMATICS_PATH,
    "stage1_report_path": STAGE1_REPORT_PATH,
    "runner_candidate_rule": "candidate_must_be_NOT_RUN_and_formal_eligible_false",
    "decision_rule": "PASS_only_after_rebuilt_preregistration_and_hypothesis_contract_match",
    "resolved_config_rule": "persisted_resolved_config_v2_must_match_all_three_s21_commits",
    "source_rule": "four_formal_stage1_11_task_commits_with_content_and_bridge_evidence",
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
        raise G20Blocked(f"{field}:COMMIT_REQUIRED")
    return value


def _logical_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G20Blocked(f"{field}:LOGICAL_PATH_REQUIRED")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise G20Blocked(f"{field}:PATH_ESCAPE")
    return parsed.as_posix()


def _check_symlink_path(path: Path, field: str) -> None:
    """Reject a symlink at any existing component, including the root."""

    try:
        absolute = path.absolute()
        current = Path(absolute.anchor)
        for part in absolute.parts[1:] if absolute.anchor else absolute.parts:
            current = current / part
            if current.is_symlink():
                raise G20Blocked(f"{field}:SYMLINK_FORBIDDEN")
    except G20Blocked:
        raise
    except OSError as error:
        raise G20Blocked(f"{field}:UNREADABLE") from error


def _root(value: str | Path, field: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    candidate = candidate.absolute()
    _check_symlink_path(candidate, field)
    if not candidate.is_dir():
        raise G20Blocked(f"{field}:DIRECTORY_REQUIRED")
    return candidate


def _reject_symlink_chain(root: Path, logical: str, field: str) -> None:
    _check_symlink_path(root, field)
    current = root
    try:
        for part in PurePosixPath(logical).parts:
            current = current / part
            if current.is_symlink():
                raise G20Blocked(f"{field}:SYMLINK_FORBIDDEN")
    except G20Blocked:
        raise
    except OSError as error:
        raise G20Blocked(f"{field}:UNREADABLE") from error


def _resolve(root: Path, value: object, field: str) -> Path:
    logical = _logical_path(value, field)
    _reject_symlink_chain(root, logical, field)
    candidate = root.joinpath(*PurePosixPath(logical).parts)
    target = candidate.resolve(strict=False)
    try:
        target.relative_to(root.resolve(strict=False))
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


def _safe_evidence_refs(references: object) -> tuple[str, ...]:
    try:
        return _normalise_refs(references)
    except G20Blocked:
        return ()


def _load_committed(root: Path, ref: str) -> LoadedTaskArtifact:
    _resolve(root, ref, "artifact_commit_ref")
    try:
        item = load_committed_task_artifact(root, ref, require_formal=True)
        # ``load_committed_task_artifact`` verifies hashes and layout, but its
        # historical implementation follows symlinks.  The evaluator's
        # stricter boundary rejects both commit and object chains first.
        _resolve(root, item.identity.object_ref, "artifact_object_ref")
        return item
    except G20Blocked:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise G20Blocked(f"artifact_commit_ref:INVALID:{type(error).__name__}") from error


@dataclass(frozen=True, slots=True)
class _Stage1SourceSet:
    refs_by_kind: Mapping[str, str]
    artifacts_by_kind: Mapping[str, LoadedTaskArtifact]
    config_hash: str
    binding_hash: str


@dataclass(frozen=True, slots=True)
class _LoadedSet:
    refs_by_kind: Mapping[str, str]
    artifacts_by_kind: Mapping[str, LoadedTaskArtifact]
    config_hash: str
    source_refs: tuple[str, ...]
    stage1: _Stage1SourceSet


def _payload_status(payload: Mapping[str, JSONValue]) -> object:
    return payload.get("status", payload.get("gate_status"))


def _validate_stage1_payload(kind: str, item: LoadedTaskArtifact) -> None:
    payload = item.payload
    schema = payload.get("schema_version")
    if schema == STAGE1_LEGACY_PAYLOAD_SCHEMA:
        if payload.get("scope") != "formal":
            raise G20Blocked(f"stage1_source:{kind}:FORMAL_SCOPE_REQUIRED")
        if payload.get("task_id") != STAGE1_TASK_ID:
            raise G20Blocked(f"stage1_source:{kind}:WRONG_TASK_ID")
        if _payload_status(payload) != "PASS":
            raise G20Blocked(f"stage1_source:{kind}:STATUS_NOT_PASS")
    elif schema == STAGE1_PAYLOAD_SCHEMAS[kind]:
        if payload.get("task_id") != STAGE1_TASK_ID or payload.get("gate_id") != STAGE1_GATE_ID:
            raise G20Blocked(f"stage1_source:{kind}:IDENTITY_INVALID")
        if _payload_status(payload) != "PASS":
            raise G20Blocked(f"stage1_source:{kind}:STATUS_NOT_PASS")
    else:
        raise G20Blocked(f"stage1_source:{kind}:PAYLOAD_SCHEMA_INVALID")

    declared_config = payload.get("config_hash")
    if declared_config is not None and declared_config != item.identity.config_hash:
        raise G20Blocked(f"stage1_source:{kind}:CONFIG_HASH_MISMATCH")
    task_definition_hash = payload.get("task_definition_hash")
    if task_definition_hash is not None:
        _sha(task_definition_hash, f"stage1_source:{kind}.task_definition_hash")
    core = payload.get("core_evidence")
    core_hash = payload.get("core_evidence_hash")
    if core is not None:
        if not isinstance(core, Mapping) or not isinstance(core_hash, str):
            raise G20Blocked(f"stage1_source:{kind}:CORE_EVIDENCE_BRIDGE_MISSING")
        if canonical_json_hash(core) != core_hash:
            raise G20Blocked(f"stage1_source:{kind}:CORE_EVIDENCE_BRIDGE_MISMATCH")

    if kind == "gate_summary" and "unresolved_failure_count" in payload:
        if payload.get("unresolved_failure_count") != 0:
            raise G20Blocked("stage1_source:gate_summary:UNRESOLVED_FAILURES")
    if kind == "requirements_matrix" and "rows" in payload:
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != 28:
            raise G20Blocked("stage1_source:requirements_matrix:ROWS_INVALID")
        expected = [f"S1.11-R{index:02d}" for index in range(1, 29)]
        observed: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping) or row.get("status") != "PASS":
                raise G20Blocked("stage1_source:requirements_matrix:NOT_ALL_PASS")
            requirement_id = row.get("requirement_id")
            if not isinstance(requirement_id, str):
                raise G20Blocked("stage1_source:requirements_matrix:REQUIREMENT_ID_INVALID")
            observed.append(requirement_id)
        if observed != expected:
            raise G20Blocked("stage1_source:requirements_matrix:ORDER_INVALID")


def _stage1_binding_hash(artifacts: Mapping[str, LoadedTaskArtifact], refs: Mapping[str, str]) -> str:
    return canonical_json_hash(
        {
            "predecessor_task_ids": [STAGE1_TASK_ID],
            "artifacts": [
                {
                    "task_id": artifacts[kind].identity.task_id,
                    "artifact_kind": artifacts[kind].identity.artifact_kind,
                    "artifact_hash": artifacts[kind].identity.artifact_hash,
                    "config_hash": artifacts[kind].identity.config_hash,
                    "run_intent": artifacts[kind].run_intent,
                    "formal_eligible": artifacts[kind].identity.formal_eligible,
                    "commit_ref": refs[kind],
                }
                for kind in STAGE1_ARTIFACT_KINDS
            ],
            "auxiliary_refs": [],
        }
    )


def _load_stage1_sources(data_root: Path, refs: Sequence[str]) -> _Stage1SourceSet:
    if len(refs) != len(STAGE1_ARTIFACT_KINDS):
        raise G20Blocked("source_refs:EXACTLY_FOUR_STAGE1_11_COMMITS_REQUIRED")
    loaded: list[tuple[str, str, LoadedTaskArtifact]] = []
    for index, ref in enumerate(refs):
        reference = _logical_path(ref, f"source_refs[{index}]")
        item = _load_committed(data_root, reference)
        if item.identity.task_id != STAGE1_TASK_ID:
            raise G20Blocked("source_refs:WRONG_TASK_ID")
        kind = item.identity.artifact_kind
        if kind not in STAGE1_ARTIFACT_KINDS:
            raise G20Blocked("source_refs:WRONG_ARTIFACT_KIND")
        if kind in {entry[0] for entry in loaded}:
            raise G20Blocked("source_refs:DUPLICATE_ARTIFACT_KIND")
        if item.run_intent != "formal" or item.identity.formal_eligible is not True:
            raise G20Blocked("source_refs:FORMAL_ENVELOPE_REQUIRED")
        _validate_stage1_payload(kind, item)
        loaded.append((kind, reference, item))
    if {kind for kind, _, _ in loaded} != set(STAGE1_ARTIFACT_KINDS):
        raise G20Blocked("source_refs:COMPLETE_STAGE1_11_SET_REQUIRED")
    configs = {item.identity.config_hash for _, _, item in loaded}
    if len(configs) != 1 or _SHA256.fullmatch(next(iter(configs))) is None:
        raise G20Blocked("source_refs:MIXED_OR_INVALID_CONFIG_HASH")
    refs_by_kind = {kind: ref for kind, ref, _ in loaded}
    artifacts_by_kind = {kind: item for kind, _, item in loaded}
    # A Stage 1 report family is a bridge, not four unrelated PASS-looking
    # documents: all members must carry the same core evidence/task definition
    # where those fields are present.
    core_hashes = {
        item.payload.get("core_evidence_hash")
        for item in artifacts_by_kind.values()
        if item.payload.get("core_evidence_hash") is not None
    }
    if len(core_hashes) > 1:
        raise G20Blocked("source_refs:CORE_EVIDENCE_BRIDGE_MISMATCH")
    task_definition_hashes = {
        item.payload.get("task_definition_hash")
        for item in artifacts_by_kind.values()
        if item.payload.get("task_definition_hash") is not None
    }
    if len(task_definition_hashes) > 1:
        raise G20Blocked("source_refs:TASK_DEFINITION_BRIDGE_MISMATCH")
    return _Stage1SourceSet(
        refs_by_kind=refs_by_kind,
        artifacts_by_kind=artifacts_by_kind,
        config_hash=next(iter(configs)),
        binding_hash=_stage1_binding_hash(artifacts_by_kind, refs_by_kind),
    )


def _load_and_bind(data_root: Path, references: object) -> _LoadedSet:
    refs = _normalise_refs(references)
    loaded: list[tuple[str, LoadedTaskArtifact, str]] = []
    for ref in refs:
        item = _load_committed(data_root, ref)
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
    source_refs = tuple(
        _logical_path(item, f"source_refs[{index}]")
        for index, item in enumerate(next(iter(source_sets)))
    )
    stage1 = _load_stage1_sources(data_root, source_refs)
    refs_by_kind = {kind: ref for kind, _, ref in loaded}
    artifacts_by_kind = {kind: item for kind, item, _ in loaded}
    return _LoadedSet(refs_by_kind, artifacts_by_kind, config_hash, source_refs, stage1)


@dataclass(frozen=True, slots=True)
class _ResolvedConfigBinding:
    ref: str
    config: ResolvedConfigV2

    @property
    def config_hash(self) -> str:
        return self.config.config_hash

    @property
    def full_hash(self) -> str:
        return self.config.full_hash


def _load_resolved_config(data_root: Path, reference: object) -> _ResolvedConfigBinding:
    ref = _logical_path(reference, "resolved_config_ref")
    path = _resolve(data_root, ref, "resolved_config_ref")
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise G20Blocked(f"resolved_config_ref:INVALID:{type(error).__name__}") from error
    if not isinstance(value, Mapping):
        raise G20Blocked("resolved_config_ref:OBJECT_REQUIRED")
    # A config may itself be persisted as a formal TaskArtifact.  Direct JSON
    # is also accepted, but only when it parses as the strict V2 wire object.
    envelope_config_hash: str | None = None
    if value.get("schema_version") == "task-output-commit-v1":
        item = _load_committed(data_root, ref)
        payload = item.payload
        if item.identity.task_id != TASK_ID or item.identity.artifact_kind not in {"resolved_config", "config"}:
            raise G20Blocked("resolved_config_ref:WRONG_TASK_OR_KIND")
        if payload.get("schema_version") != "resolved-config-v2":
            raise G20Blocked("resolved_config_ref:V2_PAYLOAD_REQUIRED")
        envelope_config_hash = item.identity.config_hash
        value = payload
    try:
        config = ResolvedConfigV2.from_mapping(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, KeyError) as error:
        raise G20Blocked(f"resolved_config_ref:V2_INVALID:{type(error).__name__}") from error
    if config.task_id != TASK_ID:
        raise G20Blocked("resolved_config_ref:WRONG_TASK_ID")
    if config.run_intent != "formal" or config.formal_eligible is not True:
        raise G20Blocked("resolved_config_ref:FORMAL_CONFIG_REQUIRED")
    if envelope_config_hash is not None and envelope_config_hash != config.config_hash:
        raise G20Blocked("resolved_config_ref:ENVELOPE_CONFIG_HASH_MISMATCH")
    return _ResolvedConfigBinding(ref, config)


@dataclass(frozen=True, slots=True)
class _RepositoryIdentity:
    root: Path
    head: str
    source_hashes: Mapping[str, str]
    evaluation_config: Mapping[str, JSONValue]
    evaluation_config_hash: str
    validation_error: str | None


def _git_text(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise G20Blocked("repository_git:COMMAND_FAILED") from error
    return result.stdout.strip()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise G20Blocked("repository_git:COMMAND_FAILED") from error
    return result.stdout


def _repository_identity(repository_root: Path) -> _RepositoryIdentity:
    root = _root(repository_root, "repository_root")
    head = _commit(_git_text(root, "rev-parse", "HEAD"), "repository_head")
    try:
        _git_text(root, "cat-file", "-e", f"{head}^{{commit}}")
    except G20Blocked as error:
        raise G20Blocked("repository_head:GIT_OBJECT_MISSING") from error

    hashes: dict[str, str] = {}
    errors: list[str] = []
    for relative in REPOSITORY_SOURCE_PATHS:
        try:
            path = _resolve(root, relative, f"repository_source:{relative}")
            hashes[relative] = _file_sha256(path, f"repository_source:{relative}")
            committed = _git_bytes(root, "show", f"{head}:{relative}")
            # Git's Windows checkout may normalize LF blobs to CRLF.  Compare
            # canonical line endings for the byte identity check; the separate
            # ``git diff --quiet`` below still catches every content change.
            if committed.replace(b"\r\n", b"\n") != path.read_bytes().replace(b"\r\n", b"\n"):
                errors.append(f"repository_source:{relative}:WORKTREE_DRIFT")
            _git_text(root, "ls-files", "--error-unmatch", "--", relative)
        except G20Blocked as error:
            errors.append(str(error))

    # The code executing this function must be the same source family that the
    # trusted repository identity describes.  This closes the otherwise subtle
    # case where a caller points ``repository_root`` at a clean clone while an
    # older/newer installed evaluator module is actually running.
    runtime_sources = {
        EVALUATOR_SOURCE_PATH: Path(__file__).resolve(),
        PREREGISTRATION_SOURCE_PATH: Path(__file__).resolve().with_name("preregistration.py"),
        RUNNER_SOURCE_PATH: Path(__file__).resolve().with_name("stage23_task_runners.py"),
    }
    for relative, runtime_path in runtime_sources.items():
        try:
            repository_path = _resolve(root, relative, f"repository_source:{relative}")
            runtime_bytes = runtime_path.read_bytes()
            if runtime_bytes.replace(b"\r\n", b"\n") != repository_path.read_bytes().replace(b"\r\n", b"\n"):
                errors.append(f"runtime_source:{relative}:EXECUTED_BYTES_MISMATCH")
        except (OSError, G20Blocked) as error:
            errors.append(f"runtime_source:{relative}:UNREADABLE:{type(error).__name__}")

    if hashes.get(PLAN_PATH) != FROZEN_PLAN_SHA256:
        errors.append("plan:CONTENT_MISMATCH")
    try:
        diff = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *REPOSITORY_SOURCE_PATHS],
            capture_output=True,
            timeout=15,
        )
        if diff.returncode == 1:
            errors.append("repository_source:TRACKED_FILES_DIRTY")
        elif diff.returncode != 0:
            errors.append("repository_source:DIFF_CHECK_FAILED")
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"repository_source:DIFF_CHECK_FAILED:{type(error).__name__}")

    # A complete evaluation identity is only made from real 64-hex source
    # digests.  Missing files therefore produce a blocked binding rather than a
    # placeholder digest that can accidentally pass a test fixture.
    if set(hashes) != set(REPOSITORY_SOURCE_PATHS) or any(_SHA256.fullmatch(value) is None for value in hashes.values()):
        raise G20Blocked("repository_source:COMPLETE_HASH_SET_REQUIRED")
    evaluation: dict[str, JSONValue] = {
        **dict(EVALUATION_CONFIG),
        "producer_commit": head,
        "source_hashes": dict(hashes),
    }
    evaluation_hash = canonical_json_hash(evaluation)
    return _RepositoryIdentity(root, head, hashes, evaluation, evaluation_hash, ";".join(errors) if errors else None)


def _validate_document_bindings(
    preregistration: Mapping[str, Any],
    *,
    repository: _RepositoryIdentity,
    stage1_binding: _Stage1SourceSet,
    config: _ResolvedConfigBinding,
) -> None:
    provenance = preregistration.get("provenance")
    if not isinstance(provenance, Mapping):
        raise G20Blocked("preregistration:PROVENANCE_REQUIRED")
    producer = _commit(provenance.get("producer_commit"), "provenance.producer_commit")
    if producer != repository.head:
        raise G20Blocked("provenance.producer_commit:TRUSTED_HEAD_MISMATCH")
    seed_hash = _sha(provenance.get("seed_plan_hash"), "provenance.seed_plan_hash")
    try:
        identity = config.config.base_config.section("identity")
        expected_seed_hash = SeedPlan.from_master_seed(int(identity["master_seed"])).artifact_hash  # type: ignore[index]
    except (KeyError, TypeError, ValueError) as error:
        raise G20Blocked("resolved_config_ref:MASTER_SEED_INVALID") from error
    if seed_hash != expected_seed_hash:
        raise G20Blocked("provenance.seed_plan_hash:RESOLVED_CONFIG_MISMATCH")
    upstream = _sha(provenance.get("upstream_binding_hash"), "provenance.upstream_binding_hash")
    if upstream != stage1_binding.binding_hash:
        raise G20Blocked("provenance.upstream_binding_hash:SOURCE_BINDING_MISMATCH")
    mathematics_hash = _sha(provenance.get("mathematics_hash"), "provenance.mathematics_hash")
    stage1_hash = _sha(provenance.get("stage1_report_hash"), "provenance.stage1_report_hash")
    if provenance.get("mathematics_path") != MATHEMATICS_PATH:
        raise G20Blocked("provenance.mathematics_path:FROZEN_PATH_MISMATCH")
    if provenance.get("stage1_report_path") != STAGE1_REPORT_PATH:
        raise G20Blocked("provenance.stage1_report_path:FROZEN_PATH_MISMATCH")
    if mathematics_hash != repository.source_hashes[MATHEMATICS_PATH]:
        raise G20Blocked("provenance.mathematics_hash:CONTENT_MISMATCH")
    if stage1_hash != repository.source_hashes[STAGE1_REPORT_PATH]:
        raise G20Blocked("provenance.stage1_report_hash:CONTENT_MISMATCH")


def _validate_preregistration(
    value: Mapping[str, Any],
    *,
    repository: _RepositoryIdentity,
    stage1_binding: _Stage1SourceSet,
    config: _ResolvedConfigBinding,
) -> None:
    try:
        validate_stage2_preregistration(value)
    except (TypeError, ValueError) as error:
        raise G20Failed(f"preregistration:VALIDATOR_REJECTED:{error}") from error
    if value.get("scope") != "formal" or value.get("formal_eligible") is not False:
        raise G20Failed("preregistration:FORMAL_SCOPE_REQUIRED")
    _validate_document_bindings(
        value,
        repository=repository,
        stage1_binding=stage1_binding,
        config=config,
    )
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


def _validate_runner_candidate(
    value: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    hypothesis: Mapping[str, Any],
) -> None:
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


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _make_gate(
    *,
    status: GateStatus,
    reasons: Sequence[str],
    evidence_refs: Sequence[str],
    loaded: _LoadedSet | None,
    repository: _RepositoryIdentity | None,
    config: _ResolvedConfigBinding | None,
) -> GateRecord:
    candidate: Mapping[str, JSONValue] = {}
    prereg_hash: str | None = None
    hypothesis_hash: str | None = None
    config_hash = loaded.config_hash if loaded is not None else None
    source_identity: list[JSONValue] = []
    stage1_identity: list[JSONValue] = []
    if loaded is not None:
        prereg_hash = loaded.artifacts_by_kind["preregistration"].payload.get("preregistration_hash")
        hypothesis_hash = loaded.artifacts_by_kind["hypothesis_contract"].payload.get("hypothesis_contract_hash")
        candidate = loaded.artifacts_by_kind["gate_record"].payload
        source_identity = [
            {
                "artifact_kind": kind,
                "commit_ref": loaded.refs_by_kind[kind],
                "artifact_hash": loaded.artifacts_by_kind[kind].identity.artifact_hash,
                "config_hash": loaded.artifacts_by_kind[kind].identity.config_hash,
                "run_intent": loaded.artifacts_by_kind[kind].run_intent,
                "formal_eligible": loaded.artifacts_by_kind[kind].identity.formal_eligible,
            }
            for kind in ARTIFACT_KINDS
        ]
        stage1_identity = [
            {
                "artifact_kind": kind,
                "commit_ref": loaded.stage1.refs_by_kind[kind],
                "artifact_hash": loaded.stage1.artifacts_by_kind[kind].identity.artifact_hash,
                "config_hash": loaded.stage1.artifacts_by_kind[kind].identity.config_hash,
            }
            for kind in STAGE1_ARTIFACT_KINDS
        ]
    measured: dict[str, JSONValue] = {
        "input_artifact_count": 3 if loaded is not None else 0,
        "task_id": TASK_ID,
        "artifact_kinds": list(ARTIFACT_KINDS),
        "config_hash": config_hash,
        "preregistration_hash": prereg_hash,
        "hypothesis_contract_hash": hypothesis_hash,
        "runner_candidate_status": candidate.get("gate_status"),
        "runner_candidate_role": "non_self_signed_lineage_only",
        "source_artifacts": source_identity,
        "stage1_source_artifacts": stage1_identity,
        "resolved_config": (
            None
            if config is None
            else {
                "ref": config.ref,
                "task_id": config.config.task_id,
                "run_intent": config.config.run_intent,
                "formal_eligible": config.config.formal_eligible,
                "config_hash": config.config_hash,
                "full_hash": config.full_hash,
            }
        ),
        "evaluator": {
            "producer_commit": None if repository is None else repository.head,
            "source_sha256": None if repository is None else repository.source_hashes.get(EVALUATOR_SOURCE_PATH),
            "source_hashes": None if repository is None else dict(repository.source_hashes),
            "evaluation_config": None if repository is None else dict(repository.evaluation_config),
            "evaluation_config_hash": None if repository is None else repository.evaluation_config_hash,
            "repository_validation": None if repository is None else repository.validation_error,
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


def _output_logical_dir(data_root: Path, output_dir: str | Path) -> str:
    candidate = Path(output_dir)
    if candidate.is_absolute():
        absolute = candidate.absolute()
        _check_symlink_path(absolute, "output_dir")
        resolved_root = data_root.resolve(strict=False)
        resolved = absolute.resolve(strict=False)
        try:
            logical = PurePosixPath(resolved.relative_to(resolved_root).as_posix())
        except ValueError as error:
            raise G20Blocked("output_dir:PATH_ESCAPE") from error
        if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
            raise G20Blocked("output_dir:PATH_ESCAPE")
        value = logical.as_posix()
    else:
        if isinstance(output_dir, Path):
            value = _logical_path(PurePosixPath(*candidate.parts).as_posix(), "output_dir")
        else:
            value = _logical_path(str(output_dir), "output_dir")
    _reject_symlink_chain(data_root, value, "output_dir")
    return value


def _attempt_id(
    *,
    base_dir: str,
    loaded: _LoadedSet,
    config: _ResolvedConfigBinding,
    repository: _RepositoryIdentity,
) -> str:
    return canonical_json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "artifact_refs": [loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS],
            "config_hash": loaded.config_hash,
            "resolved_config_ref": config.ref,
            "resolved_config_full_hash": config.full_hash,
            "evaluation_config_hash": repository.evaluation_config_hash,
            "output_dir": base_dir,
        }
    )


def _reuse_existing(
    data_root: Path,
    commit_ref: str,
    *,
    expected_source_refs: Sequence[str],
    repository: _RepositoryIdentity,
    config: _ResolvedConfigBinding,
) -> tuple[GateRecord, LoadedTaskArtifact]:
    _resolve(data_root, commit_ref, "output_commit_ref")
    output = _load_committed(data_root, commit_ref)
    if output.identity.task_id != TASK_ID or output.identity.artifact_kind != "gate_record":
        raise G20Blocked("output_commit:IDENTITY_DRIFT")
    if output.identity.config_hash != config.config_hash:
        raise G20Blocked("output_commit:CONFIG_DRIFT")
    if output.source_refs != tuple(expected_source_refs):
        raise G20Blocked("output_commit:SOURCE_REFS_DRIFT")
    try:
        gate = GateRecord.from_mapping(dict(output.payload))
    except (TypeError, ValueError) as error:
        raise G20Blocked("output_commit:GATE_RECORD_INVALID") from error
    measured = gate.measured
    if isinstance(measured, Mapping):
        evaluator = measured.get("evaluator")
        if isinstance(evaluator, Mapping) and evaluator.get("evaluation_config_hash") != repository.evaluation_config_hash:
            raise G20Blocked("output_commit:EVALUATION_CONFIG_DRIFT")
        resolved = measured.get("resolved_config")
        if isinstance(resolved, Mapping) and resolved.get("full_hash") != config.full_hash:
            raise G20Blocked("output_commit:RESOLVED_CONFIG_DRIFT")
    return gate, output


def _publish(
    data_root: Path,
    *,
    base_dir: str,
    attempt_id: str,
    gate: GateRecord,
    loaded: _LoadedSet,
) -> PublishedTaskArtifact:
    attempt_dir = f"{base_dir}/g2.0-attempts/{attempt_id}"
    _reject_symlink_chain(data_root, attempt_dir, "output_attempt")
    store = TaskArtifactStore(data_root, attempt_dir)
    published = store.publish(
        task_id=TASK_ID,
        artifact_kind="gate_record",
        config_hash=loaded.config_hash,
        run_intent="formal",
        payload=gate.to_dict(),
        formal_eligible=True,
        source_refs=tuple(loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS),
    )
    _resolve(data_root, published.object_ref, "output_object_ref")
    return published


def _result(
    gate: GateRecord,
    *,
    published: PublishedTaskArtifact | None,
    repository: _RepositoryIdentity | None,
    config: _ResolvedConfigBinding | None,
    source_refs: Sequence[str],
) -> dict[str, JSONValue]:
    return {
        "schema_version": SCHEMA_VERSION,
        "gate_record": gate.to_dict(),
        "status": gate.status.value,
        "formal_eligible": gate.status is GateStatus.PASS and published is not None,
        "evaluation_config_hash": None if repository is None else repository.evaluation_config_hash,
        "source_refs": list(source_refs),
        "resolved_config_ref": None if config is None else config.ref,
        "resolved_config_full_hash": None if config is None else config.full_hash,
        "commit_ref": None if published is None else published.commit_ref,
        "envelope_artifact_hash": None if published is None else published.artifact_hash,
        "provenance": {
            "repository_head": None if repository is None else repository.head,
            "evaluation_config_hash": None if repository is None else repository.evaluation_config_hash,
            "resolved_config_ref": None if config is None else config.ref,
            "resolved_config_full_hash": None if config is None else config.full_hash,
        },
    }


def evaluate_formal_g20(
    data_root: str | Path,
    artifact_refs: Sequence[str] | Mapping[str, str],
    output_dir: str | Path = "runs/stage2-g20-evaluation",
    *,
    repository_root: str | Path,
    resolved_config_ref: str | None = None,
    evaluation_config_ref: str | None = None,
) -> dict[str, JSONValue]:
    """Evaluate and publish G2.0 from three formal S2.1 task commits.

    ``resolved_config_ref`` is a data-root-relative persisted V2 wire object or
    a formal ``resolved_config`` TaskArtifact commit.  The function returns a
    GateRecord-shaped result even for blocked input; only a fully loaded input
    set with a valid output boundary receives an immutable envelope.
    """

    if resolved_config_ref is None:
        resolved_config_ref = evaluation_config_ref
    elif evaluation_config_ref is not None and evaluation_config_ref != resolved_config_ref:
        resolved_config_ref = None
    loaded: _LoadedSet | None = None
    repository: _RepositoryIdentity | None = None
    config: _ResolvedConfigBinding | None = None
    status = GateStatus.BLOCKED
    reasons: list[str] = []
    try:
        try:
            repository = _repository_identity(_root(repository_root, "repository_root"))
        except G20Blocked as error:
            reasons.append(str(error))
        data = _root(data_root, "data_root")
        try:
            loaded = _load_and_bind(data, artifact_refs)
        except G20Failed as error:
            status = GateStatus.FAIL
            reasons.append(str(error))
        except (G20Blocked, OSError, TypeError, ValueError, KeyError) as error:
            reasons.append(str(error))
        if loaded is not None:
            try:
                config = _load_resolved_config(data, resolved_config_ref)
                if config.config_hash != loaded.config_hash:
                    raise G20Blocked("resolved_config_ref:CONFIG_HASH_MISMATCH")
            except (G20Blocked, OSError, TypeError, ValueError, KeyError) as error:
                reasons.append(str(error))
            if repository is None:
                pass
            elif repository.validation_error is not None:
                reasons.append(repository.validation_error)
            elif config is not None:
                try:
                    prereg = loaded.artifacts_by_kind["preregistration"].payload
                    hypothesis = loaded.artifacts_by_kind["hypothesis_contract"].payload
                    candidate = loaded.artifacts_by_kind["gate_record"].payload
                    _validate_preregistration(
                        prereg,
                        repository=repository,
                        stage1_binding=loaded.stage1,
                        config=config,
                    )
                    _validate_hypothesis(hypothesis, prereg)
                    _validate_runner_candidate(candidate, prereg, hypothesis)
                    status = GateStatus.PASS
                except G20Failed as error:
                    status = GateStatus.FAIL
                    reasons.append(str(error))
                except (G20Blocked, OSError, TypeError, ValueError, KeyError) as error:
                    reasons.append(str(error))
    except (G20Blocked, OSError, TypeError, ValueError, KeyError) as error:
        reasons.append(str(error))

    if status is not GateStatus.FAIL and reasons:
        status = GateStatus.BLOCKED
    if not reasons and status is not GateStatus.PASS:
        reasons.append("G2.0 formal qualification did not complete")
    evidence_refs = (
        tuple(loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS)
        if loaded is not None
        else _safe_evidence_refs(artifact_refs)
    )

    # A config/reference failure is not assigned a deterministic attempt ID;
    # do not overwrite an existing result with an unbound BLOCKED envelope.
    published: PublishedTaskArtifact | None = None
    if loaded is not None and repository is not None and config is not None:
        try:
            data = _root(data_root, "data_root")
            base_dir = _output_logical_dir(data, output_dir)
            attempt = _attempt_id(base_dir=base_dir, loaded=loaded, config=config, repository=repository)
            commit_ref = f"{base_dir}/g2.0-attempts/{attempt}/commits/gate_record.json"
            _reject_symlink_chain(data, commit_ref, "output_commit_ref")
            if (data / Path(*PurePosixPath(commit_ref).parts)).exists():
                existing_gate, existing = _reuse_existing(
                    data,
                    commit_ref,
                    expected_source_refs=tuple(loaded.refs_by_kind[kind] for kind in ARTIFACT_KINDS),
                    repository=repository,
                    config=config,
                )
                return _result(
                    existing_gate,
                    published=existing.identity,
                    repository=repository,
                    config=config,
                    source_refs=evidence_refs,
                )
            gate = _make_gate(
                status=status,
                reasons=reasons,
                evidence_refs=evidence_refs,
                loaded=loaded,
                repository=repository,
                config=config,
            )
            published = _publish(
                data,
                base_dir=base_dir,
                attempt_id=attempt,
                gate=gate,
                loaded=loaded,
            )
            return _result(gate, published=published, repository=repository, config=config, source_refs=evidence_refs)
        except (G20Blocked, OSError, TypeError, ValueError, RuntimeError) as error:
            reasons.append(f"formal envelope publish failed: {type(error).__name__}:{error}")
            status = GateStatus.BLOCKED

    try:
        gate = _make_gate(
            status=status,
            reasons=reasons,
            evidence_refs=evidence_refs,
            loaded=loaded,
            repository=repository,
            config=config,
        )
    except Exception as error:
        gate = GateRecord(
            gate_id=GATE_ID,
            stage=2,
            status=GateStatus.BLOCKED,
            checked_at=_timestamp(),
            measured={"evaluator": {"evaluation_config_hash": None}},
            threshold={"contract": "frozen_s2_1_validators"},
            evidence_refs=(),
            reasons=(f"G2.0 GateRecord construction failed closed: {type(error).__name__}",),
        )
    return _result(gate, published=None, repository=repository, config=config, source_refs=evidence_refs)


def evaluate_g20(
    data_root: str | Path,
    artifact_refs: Sequence[str] | Mapping[str, str],
    output_dir: str | Path = "runs/stage2-g20-evaluation",
    **kwargs: object,
) -> dict[str, JSONValue]:
    """Narrow compatibility alias used by materializers."""

    return evaluate_formal_g20(data_root, artifact_refs, output_dir, **kwargs)  # type: ignore[arg-type]


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
