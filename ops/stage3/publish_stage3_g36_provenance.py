#!/usr/bin/env python3
"""Publish clean formal provenance for the S3.08 -> independent G3-6 boundary.

The publisher derives the exact G3-6 evaluator source set from committed
S3.08/S3.07 artifacts.  It validates the complete streaming aggregate before
writing anything and refuses dirty or detached Git worktrees.  The raw source
document and its task-artifact commit are immutable and idempotent.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[2]
    for _candidate in (_repository_root, _repository_root / "src"):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

from ops.stage3.run_stage3_formal import _canonical_hash, _load_json
from param_importance_nlp.analysis.report import FrozenSourceTable
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    ProvenanceRecord,
    ProvenanceStatus,
    ResolvedConfigV2,
    RunIdentity,
    SeedPlan,
    derive_experiment_id,
)
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash
from param_importance_nlp.experiments.stage3_g36_publisher import (
    _evaluation_sources,
    _load_streaming_coverage,
    _plan_shape,
    _resolve_streaming_plan_ref,
    _scope_authority,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
    publish_canonical_immutable,
)


SCHEMA_VERSION = "stage3-g36-provenance-publication-v1"
S308_TASK_ID = "stage3.08_error_analysis_and_stability"
PROVENANCE_TASK_ID = "stage3.formal_provenance_authority"
S308_OUTPUT_KINDS = frozenset(
    {"path_error_table", "stability_report", "frozen_source_table"}
)
TIMED_EXECUTION_SCHEMA = "stage3-s308-timed-execution-v2"
MATERIALIZATION_RECEIPT_SCHEMA = "stage3-task-materialization-receipt-v1"


class Stage3ProvenancePublicationError(ValueError):
    """Raised when the provenance source cannot be proven from formal inputs."""


def _fail(code: str, detail: object | None = None) -> Stage3ProvenancePublicationError:
    suffix = "" if detail is None else f":{detail}"
    return Stage3ProvenancePublicationError(f"{code}{suffix}")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("STAGE3_PROVENANCE_MAPPING_REQUIRED", field)
    return value


def _required_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("STAGE3_PROVENANCE_STRING_REQUIRED", field)
    return value


def _logical_input(path: Path, root: Path, *, field: str) -> tuple[Path, str]:
    resolved_root = root.resolve()
    candidate = path if path.is_absolute() else resolved_root / path
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as error:
        raise _fail("STAGE3_PROVENANCE_PATH_OUTSIDE_DATA_ROOT", field) from error
    if resolved_root.is_symlink():
        raise _fail("STAGE3_PROVENANCE_DATA_ROOT_SYMLINK", field)
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise _fail("STAGE3_PROVENANCE_PATH_SYMLINK", field)
    if not resolved.is_file():
        raise _fail("STAGE3_PROVENANCE_INPUT_MISSING", field)
    return resolved, PurePosixPath(*relative.parts).as_posix()


def _logical_output(
    path: Path,
    root: Path,
    repository_root: Path,
    *,
    field: str,
) -> tuple[Path, str]:
    resolved_root = root.resolve()
    absolute = Path(path if path.is_absolute() else resolved_root / path).absolute()
    try:
        relative = absolute.relative_to(resolved_root)
    except ValueError as error:
        raise _fail("STAGE3_PROVENANCE_OUTPUT_OUTSIDE_DATA_ROOT", field) from error
    try:
        absolute.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise _fail("STAGE3_PROVENANCE_OUTPUT_INSIDE_GIT_WORKTREE", field)
    current = resolved_root
    for part in relative.parts[:-1]:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _fail("STAGE3_PROVENANCE_OUTPUT_SYMLINK", field)
    return absolute, PurePosixPath(*relative.parts).as_posix()


def _authority_output_dir(value: str, root: Path, repository_root: Path) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise _fail("STAGE3_PROVENANCE_AUTHORITY_DIR_INVALID")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise _fail("STAGE3_PROVENANCE_AUTHORITY_DIR_INVALID")
    target = root.resolve().joinpath(*logical.parts)
    try:
        target.relative_to(repository_root.resolve())
    except ValueError:
        pass
    else:
        raise _fail("STAGE3_PROVENANCE_AUTHORITY_INSIDE_GIT_WORKTREE")
    current = root.resolve()
    for part in logical.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise _fail("STAGE3_PROVENANCE_AUTHORITY_DIR_SYMLINK")
    return logical.as_posix()


def _git_command(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise _fail("STAGE3_PROVENANCE_GIT_COMMAND_FAILED", " ".join(arguments)) from error
    return completed.stdout.strip()


def _git_snapshot(repository_root: Path, expected_commit: str | None) -> tuple[str, str]:
    root = repository_root.resolve()
    commit = _git_command(root, "rev-parse", "--verify", "HEAD")
    branch = _git_command(root, "rev-parse", "--abbrev-ref", "HEAD")
    dirty = _git_command(root, "status", "--porcelain=v1", "--untracked-files=all")
    if len(commit) not in {40, 64} or any(char not in "0123456789abcdef" for char in commit):
        raise _fail("STAGE3_PROVENANCE_GIT_COMMIT_INVALID")
    if branch == "HEAD" or not branch:
        raise _fail("STAGE3_PROVENANCE_GIT_DETACHED")
    if dirty:
        raise _fail("STAGE3_PROVENANCE_GIT_DIRTY")
    if expected_commit is not None and commit != expected_commit:
        raise _fail("STAGE3_PROVENANCE_GIT_COMMIT_DRIFT")
    return commit, branch


def _confirm_git_snapshot(
    repository_root: Path,
    *,
    expected_commit: str,
    expected_branch: str,
) -> tuple[str, str]:
    """Fail if either component of a previously clean Git identity drifted."""

    commit, branch = _git_snapshot(repository_root, expected_commit)
    if branch != expected_branch:
        raise _fail("STAGE3_PROVENANCE_GIT_BRANCH_DRIFT")
    return commit, branch


def _formal_artifact(
    root: Path,
    reference: str,
    *,
    field: str,
    kinds: set[str],
):
    try:
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("STAGE3_PROVENANCE_FORMAL_COMMIT_INVALID", field) from error
    if loaded.identity.artifact_kind not in kinds:
        raise _fail("STAGE3_PROVENANCE_ARTIFACT_KIND_INVALID", field)
    return loaded


def _environment_ref(environment: TaskRuntimeEnvironment, key: str) -> str:
    return _required_string(environment.evidence_refs.get(key), field=f"environment.{key}")


def _parse_time(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise _fail("STAGE3_PROVENANCE_TIME_INVALID", field) from error
    if parsed.tzinfo is None:
        raise _fail("STAGE3_PROVENANCE_TIMEZONE_REQUIRED", field)
    return parsed


def _apply_timed_execution_receipt(
    arguments: argparse.Namespace,
    root: Path,
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Bind optional CLI inputs to one immutable S3.08 timing receipt."""

    receipt_argument = getattr(arguments, "timed_execution_receipt", None)
    if receipt_argument is None:
        required = ("config", "environment", "task_result", "started_at", "ended_at")
        missing = [name for name in required if getattr(arguments, name, None) is None]
        if missing:
            raise _fail("STAGE3_PROVENANCE_EXECUTION_INPUTS_REQUIRED", ",".join(missing))
        return None, None
    receipt_path, receipt_ref = _logical_input(
        receipt_argument, root, field="timed_execution_receipt"
    )
    receipt = _mapping(
        _load_json(receipt_path), field="timed_execution_receipt"
    )
    expected = {
        "schema_version", "status", "scope", "formal_eligible", "launch_hash",
        "task_id", "materialization_receipt_ref", "materialization_receipt_hash",
        "config_ref", "config_hash", "environment_ref", "environment_hash",
        "result_ref", "result_hash", "artifact_refs", "artifact_hashes",
        "git_commit", "git_branch", "started_at", "ended_at", "ended_at_source",
        "recovered", "handoff_audit_hash", "receipt_ref", "receipt_hash",
    }
    supplied_hash = receipt.get("receipt_hash")
    ended_at_source = receipt.get("ended_at_source")
    recovered = receipt.get("recovered")
    if (
        set(receipt) != expected
        or receipt.get("schema_version") != TIMED_EXECUTION_SCHEMA
        or receipt.get("status") != "PASS"
        or receipt.get("scope") != "formal"
        or receipt.get("formal_eligible") is not True
        or receipt.get("task_id") != S308_TASK_ID
        or receipt.get("receipt_ref") != receipt_ref
        or ended_at_source not in {"wrapper_post_wait", "result_mtime_recovery"}
        or type(recovered) is not bool
        or (ended_at_source == "result_mtime_recovery") != recovered
        or supplied_hash
        != _canonical_hash(
            {key: item for key, item in receipt.items() if key != "receipt_hash"}
        )
    ):
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_INVALID")
    for field in (
        "launch_hash", "materialization_receipt_hash", "config_hash",
        "environment_hash", "result_hash", "handoff_audit_hash",
    ):
        value = receipt.get(field)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_HASH_INVALID", field)
    git_commit = receipt.get("git_commit")
    if (
        not isinstance(git_commit, str)
        or len(git_commit) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in git_commit)
        or not isinstance(receipt.get("git_branch"), str)
        or not receipt["git_branch"]
    ):
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_GIT_INVALID")
    artifacts = _mapping(receipt.get("artifact_refs"), field="timed.artifact_refs")
    hashes = _mapping(receipt.get("artifact_hashes"), field="timed.artifact_hashes")
    if set(artifacts) != S308_OUTPUT_KINDS or set(hashes) != S308_OUTPUT_KINDS:
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_OUTPUT_SET_INVALID")
    for kind in S308_OUTPUT_KINDS:
        _required_string(artifacts.get(kind), field=f"timed.artifact_refs.{kind}")
        value = hashes.get(kind)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_HASH_INVALID", kind)
    materialization_ref = _required_string(
        receipt.get("materialization_receipt_ref"),
        field="timed.materialization_receipt_ref",
    )
    materialization_path, normalized_materialization_ref = _logical_input(
        Path(materialization_ref), root, field="timed.materialization_receipt_ref"
    )
    materialization = _mapping(
        _load_json(materialization_path), field="timed.materialization_receipt"
    )
    materialization_fields = {
        "schema_version", "task_id", "config_hash", "config_ref", "result_ref",
        "artifact_output_dir", "authority_output_dir", "output_refs",
        "evidence_refs", "external_gate_ref", "command", "artifact_hash",
    }
    expected_command = [
        "{python}", "-m", "param_importance_nlp", "task", "run",
        "--config", "{config}", "--environment", "{environment}",
        "--result", "{result}",
    ]
    if (
        normalized_materialization_ref != materialization_ref
        or set(materialization) != materialization_fields
        or materialization.get("schema_version") != MATERIALIZATION_RECEIPT_SCHEMA
        or materialization.get("task_id") != S308_TASK_ID
        or materialization.get("config_ref") != receipt.get("config_ref")
        or materialization.get("config_hash") != receipt.get("config_hash")
        or materialization.get("result_ref") != receipt.get("result_ref")
        or materialization.get("output_refs") != dict(artifacts)
        or materialization.get("external_gate_ref") is not None
        or materialization.get("command") != expected_command
        or materialization.get("artifact_hash")
        != receipt.get("materialization_receipt_hash")
        or materialization.get("artifact_hash")
        != _canonical_hash(
            {
                key: item
                for key, item in materialization.items()
                if key != "artifact_hash"
            }
        )
    ):
        raise _fail("STAGE3_PROVENANCE_MATERIALIZATION_RECEIPT_INVALID")
    started_at = _required_string(receipt.get("started_at"), field="timed.started_at")
    ended_at = _required_string(receipt.get("ended_at"), field="timed.ended_at")
    if _parse_time(ended_at, field="timed.ended_at") < _parse_time(
        started_at, field="timed.started_at"
    ):
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_TIME_ORDER_INVALID")

    for attribute, field in (
        ("config", "config_ref"),
        ("environment", "environment_ref"),
        ("task_result", "result_ref"),
    ):
        bound_path, _ = _logical_input(
            Path(_required_string(receipt.get(field), field=f"timed.{field}")),
            root,
            field=f"timed.{field}",
        )
        supplied = getattr(arguments, attribute, None)
        if supplied is not None:
            supplied_path, _ = _logical_input(supplied, root, field=attribute)
            if supplied_path != bound_path:
                raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_INPUT_MISMATCH", attribute)
        setattr(arguments, attribute, bound_path)
    for attribute in ("started_at", "ended_at"):
        supplied = getattr(arguments, attribute, None)
        bound = receipt[attribute]
        if supplied is not None and supplied != bound:
            raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_INPUT_MISMATCH", attribute)
        setattr(arguments, attribute, bound)
    supplied_commit = getattr(arguments, "expected_git_commit", None)
    if supplied_commit is not None and supplied_commit != git_commit:
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_INPUT_MISMATCH", "git_commit")
    arguments.expected_git_commit = git_commit
    return receipt, receipt_ref


def _device_mapping(config: ResolvedConfigV2) -> tuple[str, ...]:
    runtime = _mapping(config.base_config.section("runtime"), field="base_config.runtime")
    distributed = _mapping(
        config.base_config.section("distributed"), field="base_config.distributed"
    )
    device = runtime.get("device")
    if device == "cpu":
        return ("cpu",)
    if device != "cuda":
        raise _fail("STAGE3_PROVENANCE_DEVICE_INVALID", device)
    identifiers = distributed.get("device_ids")
    if (
        not isinstance(identifiers, list)
        or not identifiers
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in identifiers
        )
        or len(set(identifiers)) != len(identifiers)
    ):
        raise _fail("STAGE3_PROVENANCE_DEVICE_IDS_INVALID")
    return tuple(f"cuda:{item}" for item in identifiers)


def _build_record(
    *,
    config: ResolvedConfigV2,
    environment: TaskRuntimeEnvironment,
    config_ref: str,
    evaluator_sources: tuple[str, ...],
    git_commit: str,
    git_branch: str,
    started_at: str,
    ended_at: str,
) -> ProvenanceRecord:
    started = _parse_time(started_at, field="started_at")
    ended = _parse_time(ended_at, field="ended_at")
    if ended < started:
        raise _fail("STAGE3_PROVENANCE_TIME_ORDER_INVALID")
    identity = _mapping(config.base_config.section("identity"), field="base_config.identity")
    model = _mapping(config.base_config.section("model"), field="base_config.model")
    runtime = _mapping(config.base_config.section("runtime"), field="base_config.runtime")
    launcher = _mapping(config.section("launcher"), field="launcher")
    master_seed = identity.get("master_seed")
    world_size = launcher.get("world_size")
    if (
        isinstance(master_seed, bool)
        or not isinstance(master_seed, int)
        or isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise _fail("STAGE3_PROVENANCE_SEED_OR_WORLD_SIZE_INVALID")
    model_identity = _required_string(
        identity.get("input_checkpoint_id") or model.get("initialization_id"),
        field="model_identity",
    )
    route = _required_string(identity.get("route"), field="identity.route")
    experiment_id = derive_experiment_id(
        stage=3,
        task=config.task_id,
        model_identity=model_identity,
        route=route,
        master_seed=master_seed,
        config_hash=config.config_hash,
    )
    collision_code = canonical_json_hash(
        {
            "schema_version": "stage3-provenance-collision-v1",
            "config_hash": config.config_hash,
            "git_commit": git_commit,
            "started_at": started_at,
        }
    )[:8]
    run_identity = RunIdentity.create(
        experiment_id=experiment_id,
        created_at=started,
        collision_code=collision_code,
        parent_experiment_id=identity.get("parent_experiment_id"),
        input_run_id=identity.get("input_run_id"),
        input_checkpoint_id=identity.get("input_checkpoint_id"),
    )
    seed_plan = SeedPlan.from_master_seed(master_seed, world_size=world_size)
    return ProvenanceRecord(
        identity=run_identity,
        config_hash=config.config_hash,
        resolved_config_ref=config_ref,
        seed_plan_hash=seed_plan.artifact_hash,
        git_commit=git_commit,
        git_branch=git_branch,
        worktree_clean=True,
        environment_id=_required_string(
            runtime.get("environment_id"), field="runtime.environment_id"
        ),
        hardware_snapshot_ref=_environment_ref(environment, "gpu_health"),
        device_mapping=_device_mapping(config),
        model_manifest_id=_environment_ref(environment, "stage2_model_manifest"),
        tokenizer_manifest_id=_environment_ref(
            environment, "stage2_tokenizer_manifest"
        ),
        data_manifest_id=_environment_ref(environment, "stage2_data_manifest"),
        started_at=started_at,
        ended_at=ended_at,
        status=ProvenanceStatus.COMPLETED,
        scope="formal",
        artifact_refs=evaluator_sources,
    )


def publish_provenance(arguments: argparse.Namespace) -> Mapping[str, Any]:
    root = arguments.data_root.resolve()
    repository_root = arguments.repository_root.resolve()
    try:
        repository_root.relative_to(root)
    except ValueError as error:
        raise _fail("STAGE3_PROVENANCE_REPOSITORY_OUTSIDE_DATA_ROOT") from error
    if not repository_root.is_dir() or root.is_symlink():
        raise _fail("STAGE3_PROVENANCE_REPOSITORY_OR_DATA_ROOT_INVALID")
    timed_receipt, timed_receipt_ref = _apply_timed_execution_receipt(
        arguments, root
    )
    config_path, config_ref = _logical_input(arguments.config, root, field="config")
    environment_path, environment_ref = _logical_input(
        arguments.environment, root, field="environment"
    )
    result_path, result_ref = _logical_input(arguments.task_result, root, field="task_result")
    source_path, source_ref = _logical_output(
        arguments.source_output,
        root,
        repository_root,
        field="source_output",
    )
    receipt_path = None
    receipt_ref = None
    if arguments.receipt is not None:
        receipt_path, receipt_ref = _logical_output(
            arguments.receipt, root, repository_root, field="receipt"
        )
        if receipt_path == source_path:
            raise _fail("STAGE3_PROVENANCE_OUTPUT_COLLISION")
    authority_output_dir = _authority_output_dir(
        arguments.authority_output_dir, root, repository_root
    )

    git_commit, git_branch = _git_snapshot(repository_root, arguments.expected_git_commit)
    config = ResolvedConfigV2.from_mapping(_load_json(config_path))
    environment = TaskRuntimeEnvironment.from_mapping(_load_json(environment_path))
    result = TaskRunResult.from_mapping(_load_json(result_path)).to_dict()
    if (
        config.task_id != S308_TASK_ID
        or config.run_intent != "formal"
        or result.get("task_id") != S308_TASK_ID
        or result.get("config_hash") != config.config_hash
        or result.get("status") != "PASS"
        or result.get("formal_eligible") is not True
    ):
        raise _fail("STAGE3_PROVENANCE_S308_RESULT_INVALID")
    if timed_receipt is not None and (
        timed_receipt.get("config_hash") != config.config_hash
        or timed_receipt.get("environment_hash") != environment.environment_hash
        or timed_receipt.get("result_hash") != result.get("result_hash")
        or timed_receipt.get("git_branch") != git_branch
    ):
        raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_IDENTITY_MISMATCH")
    output_refs = _mapping(result.get("artifact_refs"), field="task_result.artifact_refs")
    if set(output_refs) != S308_OUTPUT_KINDS:
        raise _fail("STAGE3_PROVENANCE_S308_OUTPUT_SET_INVALID")
    if timed_receipt is not None:
        timed_refs = _mapping(
            timed_receipt.get("artifact_refs"), field="timed.artifact_refs"
        )
        timed_hashes = _mapping(
            timed_receipt.get("artifact_hashes"), field="timed.artifact_hashes"
        )
        if dict(output_refs) != dict(timed_refs):
            raise _fail("STAGE3_PROVENANCE_TIMED_RECEIPT_OUTPUT_REF_MISMATCH")
        for kind, reference in output_refs.items():
            loaded = _formal_artifact(
                root, str(reference), field=f"timed.{kind}", kinds={kind}
            )
            if loaded.identity.artifact_hash != timed_hashes.get(kind):
                raise _fail(
                    "STAGE3_PROVENANCE_TIMED_RECEIPT_OUTPUT_HASH_MISMATCH", kind
                )
    table_ref = _required_string(
        output_refs.get("frozen_source_table"), field="frozen_source_table_ref"
    )

    orchestration = _mapping(config.section("orchestration"), field="orchestration")
    predecessor_refs = orchestration.get("input_result_refs")
    if (
        not isinstance(predecessor_refs, list)
        or len(predecessor_refs) != 2
        or any(not isinstance(item, str) or not item for item in predecessor_refs)
    ):
        raise _fail("STAGE3_PROVENANCE_S307_PREDECESSORS_INVALID")
    predecessors = [
        _formal_artifact(
            root,
            reference,
            field=f"s307_predecessor[{index}]",
            kinds={"formal_path_results", "completeness_report"},
        )
        for index, reference in enumerate(predecessor_refs)
    ]
    if {item.identity.artifact_kind for item in predecessors} != {
        "formal_path_results",
        "completeness_report",
    }:
        raise _fail("STAGE3_PROVENANCE_S307_PREDECESSOR_SET_INVALID")
    formal_path = next(
        item for item in predecessors if item.identity.artifact_kind == "formal_path_results"
    )
    formal = _mapping(formal_path.payload, field="formal_path_results")

    table_artifact = _formal_artifact(
        root, table_ref, field="frozen_source_table", kinds={"frozen_source_table"}
    )
    plan_ref = _environment_ref(environment, "formal_stage3_matrix_plan")
    execution_ref = _environment_ref(environment, "formal_execution")
    decision_ref = _environment_ref(environment, "stage3_scope_decision")
    scope_gate_ref = _environment_ref(environment, "gate_stage3_g3_0")
    plan_artifact = _formal_artifact(
        root, plan_ref, field="formal_plan", kinds={"formal_plan", "stage3_formal_plan"}
    )
    execution_artifact = _formal_artifact(
        root,
        execution_ref,
        field="execution_evidence",
        kinds={"formal_execution_evidence", "execution_evidence"},
    )
    decision_artifact = _formal_artifact(
        root,
        decision_ref,
        field="scope_decision",
        kinds={"scope_authority", "stage3_scope_authority"},
    )
    scope_gate_artifact = _formal_artifact(
        root, scope_gate_ref, field="scope_gate", kinds={"gate_record"}
    )
    if any(
        item.identity.config_hash != config.config_hash
        for item in (table_artifact, plan_artifact)
    ):
        raise _fail("STAGE3_PROVENANCE_S308_CONFIG_BINDING_INVALID")
    table = FrozenSourceTable.from_mapping(dict(table_artifact.payload))
    execution = FormalExecutionEvidence.from_mapping(dict(execution_artifact.payload))
    execution.require_for_stage(3)
    plan = _mapping(plan_artifact.payload, field="formal_plan")
    units, rules = _plan_shape(plan, execution)
    _scope_authority(
        decision_payload=_mapping(decision_artifact.payload, field="scope_decision"),
        gate_payload=_mapping(scope_gate_artifact.payload, field="scope_gate"),
        decision_ref=decision_ref,
        execution=execution,
    )
    raw_plan_ref = _required_string(formal.get("formal_plan_ref"), field="raw_plan_ref")
    resolved_streaming_plan_ref = _resolve_streaming_plan_ref(
        root,
        plan_artifact=plan_artifact,
        formal_plan_ref=plan_ref,
        streaming_formal_plan_ref=raw_plan_ref,
    )
    streaming_ref = _required_string(
        formal.get("streaming_aggregate_ref"), field="streaming_coverage_ref"
    )
    streaming_hash = _required_string(
        formal.get("streaming_aggregate_hash"), field="streaming_coverage_hash"
    )
    binding_hash = _required_string(
        formal.get("reference_binding_hash"), field="reference_binding_hash"
    )
    _load_streaming_coverage(
        root,
        streaming_ref,
        expected_units=units,
        expected_rules=rules,
        expected_execution_hash=execution.artifact_hash,
        expected_plan_ref=resolved_streaming_plan_ref,
        expected_plan_hash=str(plan["artifact_hash"]),
        expected_index_ref=plan.get("production_unit_index_ref"),
        expected_index_hash=plan.get("production_unit_index_hash"),
        expected_binding_hash=binding_hash,
        declared_hash=streaming_hash,
    )
    evaluator_sources = _evaluation_sources(
        frozen_ref=table_ref,
        execution_ref=execution_ref,
        plan_ref=plan_ref,
        decision_ref=decision_ref,
        scope_gate_ref=scope_gate_ref,
        execution=execution,
        table=table,
        streaming_coverage_ref=streaming_ref,
        streaming_formal_plan_ref=resolved_streaming_plan_ref,
    )
    provenance = _build_record(
        config=config,
        environment=environment,
        config_ref=config_ref,
        evaluator_sources=evaluator_sources,
        git_commit=git_commit,
        git_branch=git_branch,
        started_at=arguments.started_at,
        ended_at=arguments.ended_at,
    )
    # Deep streaming validation may take long enough for the repository to
    # change after the initial snapshot.  Recheck immediately before the first
    # immutable write so a stale Git identity cannot leave partial authority.
    _confirm_git_snapshot(
        repository_root, expected_commit=git_commit, expected_branch=git_branch
    )
    publish_canonical_immutable(source_path, provenance.to_dict())
    envelope_sources = tuple(
        dict.fromkeys(
            (
                source_ref,
                *((timed_receipt_ref,) if timed_receipt_ref is not None else ()),
                config_ref,
                environment_ref,
                result_ref,
                *predecessor_refs,
                *evaluator_sources,
            )
        )
    )
    published = TaskArtifactStore(root, authority_output_dir).publish(
        task_id=PROVENANCE_TASK_ID,
        artifact_kind="provenance_record",
        config_hash=config.config_hash,
        run_intent="formal",
        payload=provenance.to_dict(),
        formal_eligible=True,
        source_refs=envelope_sources,
    )
    final_commit, final_branch = _confirm_git_snapshot(
        repository_root, expected_commit=git_commit, expected_branch=git_branch
    )
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "formal_eligible": True,
        "config_hash": config.config_hash,
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "s308_result_ref": result_ref,
        "s308_result_hash": result["result_hash"],
        "git_commit": final_commit,
        "git_branch": final_branch,
        "authority_output_dir": authority_output_dir,
        "provenance_source_ref": source_ref,
        "provenance_ref": published.commit_ref,
        "provenance_hash": provenance.artifact_hash,
        "provenance_envelope_hash": published.artifact_hash,
        "evaluator_source_count": len(evaluator_sources),
        "evaluator_sources_hash": canonical_json_hash(list(evaluator_sources)),
        "streaming_coverage_ref": streaming_ref,
        "streaming_coverage_hash": streaming_hash,
    }
    if timed_receipt is not None and timed_receipt_ref is not None:
        receipt["timed_execution_receipt_ref"] = timed_receipt_ref
        receipt["timed_execution_receipt_hash"] = timed_receipt["receipt_hash"]
    if receipt_ref is not None:
        receipt["receipt_ref"] = receipt_ref
    receipt["receipt_hash"] = canonical_json_hash(receipt)
    if receipt_path is not None:
        publish_canonical_immutable(receipt_path, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--task-result", type=Path)
    parser.add_argument("--started-at")
    parser.add_argument("--ended-at")
    parser.add_argument("--timed-execution-receipt", type=Path)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--authority-output-dir", required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--expected-git-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    receipt = publish_provenance(_parser().parse_args(argv))
    print(canonical_json_bytes(receipt).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "SCHEMA_VERSION",
    "Stage3ProvenancePublicationError",
    "_build_record",
    "_confirm_git_snapshot",
    "_device_mapping",
    "_git_snapshot",
    "_apply_timed_execution_receipt",
    "main",
    "publish_provenance",
]
