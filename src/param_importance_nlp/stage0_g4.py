"""Formal Stage 0 G4 configuration, identity, seed and provenance gate."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from typing import Any, Mapping

from .atomic import sha256_file
from .contracts import (
    ConfigContractError,
    GateRecord,
    GateStatus,
    ProvenanceRecord,
    ProvenanceStatus,
    ResolvedConfig,
    ResolvedConfigV2,
    RunIdentity,
    SeedPlan,
    canonical_json_hash,
    derive_experiment_id,
    diff_configs,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .experiments import build_default_task_runtime
from .g3_gate import validate_stage0_g3_resolution
from .lifecycle import RunDirectory
from .runtime import (
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g3_formalization import (
    Stage0G3FormalState,
    load_stage0_g3_formal_state,
)
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)
from .storage import StorageLayout


TASK_ID = "stage0.05_config_run_identity_and_seeds"
_G3_TASK_ID = "stage0.04_assets_and_manifests"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_CRITICAL_SOURCE_REFS = (
    "src/param_importance_nlp/contracts/config.py",
    "src/param_importance_nlp/contracts/config_v2.py",
    "src/param_importance_nlp/contracts/identity.py",
    "src/param_importance_nlp/contracts/provenance.py",
    "src/param_importance_nlp/contracts/seed.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/lifecycle.py",
    "src/param_importance_nlp/stage0_g4.py",
    "src/param_importance_nlp/stage0_gate.py",
    "schemas/stage0-gate-report-v1.json",
    "schemas/stage0-g4-provenance-evidence-v1.json",
    "schemas/stage0-g4-formalization-index-v1.json",
)
_PROVENANCE_ENVELOPE_FIELDS = {
    "schema_version",
    "status",
    "checked_at",
    "generator_git_commit",
    "environment_hash",
    "validation_report",
    "provenance_record",
    "gate_record",
    "gate_report",
    "artifact_hash",
}


class Stage0G4Error(RuntimeError):
    """G4 cannot be promoted to a formal PASS."""


@dataclass(frozen=True, slots=True)
class G4SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage0G4FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G4FormalState:
    """Strict, replay-validated handoff consumed by S0.6 and later tasks."""

    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _capture_source() -> G4SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    top = _git(repository, "rev-parse", "--show-toplevel")
    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    tracked = _git(
        repository,
        "ls-files",
        "--error-unmatch",
        "--",
        *_CRITICAL_SOURCE_REFS,
    )
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if any(item.returncode != 0 for item in (top, head, branch, tracked, status)):
        raise Stage0G4Error("G4_SOURCE_GIT_PROBE_FAILED")
    if Path(top.stdout.strip()).resolve() != repository:
        raise Stage0G4Error("G4_SOURCE_GIT_ROOT_MISMATCH")
    commit = head.stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch.stdout.strip():
        raise Stage0G4Error("G4_SOURCE_GIT_IDENTITY_INVALID")
    if status.stdout.strip():
        raise Stage0G4Error("G4_FORMAL_SOURCE_DIRTY")
    return G4SourceBinding(repository, commit, branch.stdout.strip())


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G4Error(f"G4_OBJECT_INVALID:{field}")
    return dict(value)


def _payload_with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _parse_time(value: object, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise Stage0G4Error(f"G4_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G4Error(f"G4_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Stage0G4Error(f"G4_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _load_inputs(
    request: TaskExecutionRequest,
    root: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    orchestration = request.config.section("orchestration")
    assert isinstance(orchestration, dict)
    references = tuple(str(item) for item in orchestration["input_result_refs"])
    loaded_by_kind: dict[str, Any] = {}
    refs_by_kind: dict[str, str] = {}
    for reference in references:
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
        if loaded.identity.task_id != _G3_TASK_ID:
            raise Stage0G4Error("G4_INPUT_PRODUCER_INVALID")
        kind = loaded.identity.artifact_kind
        if kind in loaded_by_kind:
            raise Stage0G4Error("G4_INPUT_KIND_DUPLICATE")
        loaded_by_kind[kind] = loaded
        refs_by_kind[kind] = reference
    if set(loaded_by_kind) != {"asset_manifest", "asset_audit", "asset_resolution"}:
        raise Stage0G4Error("G4_INPUT_SET_INVALID")
    resolution = _mapping(
        loaded_by_kind["asset_resolution"].payload, field="g3_resolution"
    )
    try:
        validate_stage0_g3_resolution(resolution)
    except Exception as error:
        raise Stage0G4Error("G4_G3_RESOLUTION_INVALID") from error
    if resolution.get("status") != "PASS":
        raise Stage0G4Error("G4_G3_RESOLUTION_NOT_PASS")
    if request.environment.evidence_refs.get("g3_resolution") != refs_by_kind["asset_resolution"]:
        raise Stage0G4Error("G4_G3_ENVIRONMENT_RESOLUTION_DRIFT")
    return resolution, refs_by_kind


def _gate_from_environment(
    environment: TaskRuntimeEnvironment,
    root: Path,
    gate_id: str,
) -> tuple[GateRecord, str]:
    key = "gate_" + re.sub(r"[^a-z0-9]+", "_", gate_id.casefold()).strip("_")
    reference = environment.evidence_refs.get(key)
    if reference is None:
        raise Stage0G4Error(f"G4_ENVIRONMENT_GATE_MISSING:{gate_id}")
    loaded = load_committed_task_artifact(root, reference, require_formal=True)
    payload = loaded.payload
    if payload.get("schema_version") == "gate-record-v1":
        candidate = payload
    else:
        candidates = [
            item
            for item in payload.values()
            if isinstance(item, Mapping) and item.get("schema_version") == "gate-record-v1"
        ]
        if len(candidates) != 1:
            raise Stage0G4Error(f"G4_ENVIRONMENT_GATE_PAYLOAD_INVALID:{gate_id}")
        candidate = candidates[0]
    gate = GateRecord.from_mapping(dict(candidate))
    if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
        raise Stage0G4Error(f"G4_ENVIRONMENT_GATE_NOT_PASS:{gate_id}")
    return gate, reference


def _entry(resolution: Mapping[str, Any], logical_name: str, kind: str) -> dict[str, Any]:
    entries = resolution.get("entries")
    if not isinstance(entries, list):
        raise Stage0G4Error("G4_G3_ENTRIES_INVALID")
    matches = [
        _mapping(item, field=f"g3.entries.{logical_name}")
        for item in entries
        if isinstance(item, Mapping) and item.get("logical_name") == logical_name
    ]
    if len(matches) != 1 or matches[0].get("kind") != kind or matches[0].get("status") != "PASS":
        raise Stage0G4Error(f"G4_G3_ASSET_MISSING:{logical_name}")
    return matches[0]


def _expect_rejected(callable_: Any, *, check: str) -> None:
    try:
        callable_()
    except (ConfigContractError, TypeError, ValueError):
        return
    raise Stage0G4Error(f"G4_NEGATIVE_CONTRACT_ACCEPTED:{check}")


def _reverse_mappings(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _reverse_mappings(item)
            for key, item in reversed(tuple(value.items()))
        }
    if isinstance(value, list):
        return [_reverse_mappings(item) for item in value]
    return value


def _contract_validation(
    request: TaskExecutionRequest,
    root: Path,
    *,
    seed_plan: SeedPlan,
    run: RunIdentity,
) -> tuple[dict[str, bool], dict[str, JSONValue]]:
    config = request.config
    base = config.base_config
    base_value = base.to_dict()

    unknown = deepcopy(base_value)
    unknown["identity"]["undeclared_parameter"] = 1
    _expect_rejected(
        lambda: ResolvedConfig.from_mapping(unknown), check="unknown_v1_field"
    )
    wrong_type = deepcopy(base_value)
    wrong_type["identity"]["master_seed"] = True
    _expect_rejected(
        lambda: ResolvedConfig.from_mapping(wrong_type), check="wrong_scalar_type"
    )
    invalid_batch = deepcopy(base_value)
    invalid_batch["batching"]["global_batch_size"] += 1
    _expect_rejected(
        lambda: ResolvedConfig.from_mapping(invalid_batch), check="batch_relation"
    )
    temporary_asset = deepcopy(base_value)
    temporary_asset["model"]["asset_id"] = "models/pythia.part"
    _expect_rejected(
        lambda: ResolvedConfig.from_mapping(temporary_asset), check="temporary_asset"
    )
    unknown_v2 = config.to_dict()
    unknown_v2["undeclared_section"] = {}
    _expect_rejected(
        lambda: ResolvedConfigV2.from_mapping(unknown_v2), check="unknown_v2_field"
    )

    reordered = ResolvedConfigV2.from_mapping(_reverse_mappings(config.to_dict()))
    if reordered.config_hash != config.config_hash or reordered.full_hash != config.full_hash:
        raise Stage0G4Error("G4_CONFIG_ORDER_NOT_CANONICAL")
    optimizer = base.section("optimizer")
    changed_learning_rate = float(optimizer["learning_rate"]) + 0.001
    changed = ResolvedConfig.resolve(
        base.to_dict(), {"optimizer": {"learning_rate": changed_learning_rate}}
    )
    differences = diff_configs(base, changed)
    if (
        changed.config_hash == base.config_hash
        or len(differences) != 1
        or differences[0].path != "optimizer.learning_rate"
        or not differences[0].semantic
    ):
        raise Stage0G4Error("G4_CONFIG_SEMANTIC_DIFF_INVALID")

    same_plan = SeedPlan.from_master_seed(seed_plan.master_seed, world_size=4)
    if same_plan.to_dict() != seed_plan.to_dict():
        raise Stage0G4Error("G4_SEED_PLAN_NOT_DETERMINISTIC")
    all_seeds = list(seed_plan.domains.values()) + list(seed_plan.rank_training.values())
    if len(all_seeds) != len(set(all_seeds)) or len(seed_plan.rank_training) != 4:
        raise Stage0G4Error("G4_SEED_DOMAINS_COLLIDE")
    second = RunIdentity.create(
        experiment_id=run.experiment_id,
        created_at=_parse_time(run.run_created_at, field="run.run_created_at"),
        collision_code=hashlib.sha256((run.run_id + ":repeat").encode()).hexdigest()[:8],
    )
    if second.run_id == run.run_id:
        raise Stage0G4Error("G4_INDEPENDENT_RUN_ID_COLLISION")
    resumed = run.next_attempt(
        started_at=_parse_time(run.attempt_started_at, field="run.attempt_started_at")
        + timedelta(seconds=1),
        input_checkpoint_id="checkpoints/g4-contract-resume-fixture",
    )
    if (
        resumed.run_id != run.run_id
        or resumed.attempt_id != 2
        or resumed.session_id == run.session_id
    ):
        raise Stage0G4Error("G4_RESUME_IDENTITY_INVALID")

    temporary_root = root / "tmp"
    if not temporary_root.is_dir():
        raise Stage0G4Error("G4_DATA_ROOT_TMP_MISSING")
    with tempfile.TemporaryDirectory(prefix="stage0-g4-identity-", dir=temporary_root) as temp:
        validation_root = Path(temp)
        (validation_root / "runs").mkdir()
        layout = StorageLayout(validation_root)
        RunDirectory.create(layout, run.run_id)
        try:
            RunDirectory.create(layout, run.run_id)
        except FileExistsError:
            no_clobber = True
        else:  # pragma: no cover - contract violation
            no_clobber = False
    if not no_clobber:
        raise Stage0G4Error("G4_RUN_DIRECTORY_CLOBBERED")

    distributed = base.section("distributed")
    runtime = base.section("runtime")
    batching = base.section("batching")
    launcher = config.section("launcher")
    assert isinstance(launcher, dict)
    explicit_ddp = (
        distributed["world_size"] == 4
        and distributed["backend"] == "nccl"
        and distributed["device_ids"] == [0, 1, 2, 3]
        and runtime["device"] == "cuda"
        and launcher["kind"] == "torchrun"
        and launcher["backend"] == "nccl"
        and launcher["world_size"] == 4
    )
    batch_relation = batching["global_batch_size"] == (
        distributed["world_size"]
        * batching["per_device_batch_size"]
        * batching["accumulation_steps"]
    )
    if not explicit_ddp or not batch_relation:
        raise Stage0G4Error("G4_FORMAL_CONFIG_RUNTIME_MATRIX_INVALID")

    checks = {
        "strict_unknown_and_type_rejection": True,
        "cross_field_batch_and_asset_rejection": True,
        "canonical_hash_and_semantic_diff": True,
        "explicit_four_rank_mapping": explicit_ddp,
        "run_no_clobber_and_resume_identity": no_clobber,
        "seed_domain_and_rank_independence": True,
    }
    observations: dict[str, JSONValue] = {
        "config_hash": config.config_hash,
        "config_full_hash": config.full_hash,
        "semantic_difference": differences[0].to_dict(),
        "master_seed": seed_plan.master_seed,
        "seed_plan_hash": seed_plan.artifact_hash,
        "seed_domain_count": len(seed_plan.domains),
        "rank_training_seeds": dict(seed_plan.rank_training),
        "independent_run_id": second.run_id,
        "resume_attempt_id": resumed.attempt_id,
        "world_size": 4,
        "device_ids": [0, 1, 2, 3],
    }
    return checks, observations


def _publish_g4(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: tuple[str, ...],
) -> TaskRunResult:
    source = _capture_source()
    resolution, input_refs = _load_inputs(request, root)
    g0g, hardware_ref = _gate_from_environment(
        request.environment, root, "stage0.G0-G"
    )
    g2, _ = _gate_from_environment(request.environment, root, "stage0.G2")
    g0_measured = _mapping(g0g.measured, field="g0g.measured")
    g2_measured = _mapping(g2.measured, field="g2.measured")
    gpu_uuids = g0_measured.get("allowed_gpu_uuids")
    if (
        not isinstance(gpu_uuids, list)
        or len(gpu_uuids) != 4
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in gpu_uuids)
    ):
        raise Stage0G4Error("G4_HARDWARE_MAPPING_INVALID")
    environment_id = g2_measured.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        raise Stage0G4Error("G4_ENVIRONMENT_ID_INVALID")

    model = _entry(resolution, "pythia-14m-step0", "model")
    tokenizer = _entry(resolution, "pythia-tokenizer", "tokenizer")
    data = _entry(resolution, "pile-selected-prefix", "pile")
    base = request.config.base_config
    identity_config = base.section("identity")
    model_config = base.section("model")
    data_config = base.section("data")
    runtime_config = base.section("runtime")
    if (
        model_config["asset_id"] != model["logical_name"]
        or model_config["tokenizer_asset_id"] != tokenizer["logical_name"]
        or data_config["asset_id"] != data["logical_name"]
        or runtime_config["environment_id"] != environment_id
    ):
        raise Stage0G4Error("G4_CONFIG_PROVENANCE_BINDING_INVALID")
    master_seed = int(identity_config["master_seed"])
    seed_plan = SeedPlan.from_master_seed(master_seed, world_size=4)
    experiment_id = derive_experiment_id(
        stage=0,
        task=TASK_ID,
        model_identity=str(model["asset_id"]),
        route=str(identity_config["route"]),
        master_seed=master_seed,
        config_hash=request.config.config_hash,
    )
    created_at = _parse_time(resolution.get("checked_at"), field="g3.checked_at")
    collision = hashlib.sha256(
        (request.config.config_hash + request.environment.environment_hash).encode()
    ).hexdigest()[:8]
    run = RunIdentity.create(
        experiment_id=experiment_id,
        created_at=created_at,
        collision_code=collision,
    )
    checks, observations = _contract_validation(
        request, root, seed_plan=seed_plan, run=run
    )

    refs: dict[str, str] = {}
    refs["resolved_config"] = store.publish(
        task_id=TASK_ID,
        artifact_kind="resolved_config",
        config_hash=request.config.config_hash,
        run_intent="formal",
        payload=request.config.to_dict(),
        formal_eligible=True,
        source_refs=source_refs,
    ).commit_ref
    refs["run_identity"] = store.publish(
        task_id=TASK_ID,
        artifact_kind="run_identity",
        config_hash=request.config.config_hash,
        run_intent="formal",
        payload=run.to_dict(),
        formal_eligible=True,
        source_refs=(refs["resolved_config"], *source_refs),
    ).commit_ref
    refs["seed_plan"] = store.publish(
        task_id=TASK_ID,
        artifact_kind="seed_plan",
        config_hash=request.config.config_hash,
        run_intent="formal",
        payload=seed_plan.to_dict(),
        formal_eligible=True,
        source_refs=(refs["resolved_config"], *source_refs),
    ).commit_ref

    ended_at = _timestamp(created_at + timedelta(seconds=1))
    provenance = ProvenanceRecord(
        identity=run,
        config_hash=request.config.config_hash,
        resolved_config_ref=refs["resolved_config"],
        seed_plan_hash=seed_plan.artifact_hash,
        git_commit=source.git_commit,
        git_branch=source.git_branch,
        worktree_clean=True,
        scope="formal",
        environment_id=environment_id,
        hardware_snapshot_ref=hardware_ref,
        device_mapping=tuple(gpu_uuids),
        model_manifest_id=str(model["asset_id"]),
        tokenizer_manifest_id=str(tokenizer["asset_id"]),
        data_manifest_id=str(data["asset_id"]),
        started_at=run.attempt_started_at,
        ended_at=ended_at,
        status=ProvenanceStatus.COMPLETED,
        artifact_refs=tuple(
            dict.fromkeys(
                (*input_refs.values(), refs["resolved_config"], refs["run_identity"], refs["seed_plan"])
            )
        ),
    )
    ProvenanceRecord.from_mapping(provenance.to_dict())
    checks["complete_clean_provenance_roundtrip"] = True
    observations.update(
        {
            "environment_id": environment_id,
            "gpu_uuids": list(gpu_uuids),
            "model_asset_id": model["asset_id"],
            "tokenizer_asset_id": tokenizer["asset_id"],
            "data_asset_id": data["asset_id"],
            "provenance_artifact_hash": provenance.artifact_hash,
        }
    )
    validation = _payload_with_hash(
        {
            "schema_version": "stage0-g4-validation-report-v1",
            "status": "PASS",
            "checked_at": ended_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "checks": dict(sorted(checks.items())),
            "observations": observations,
        }
    )
    evidence_refs = tuple(input_refs.values()) + (
        refs["resolved_config"],
        refs["run_identity"],
        refs["seed_plan"],
    )
    report = Stage0GateReport(
        gate_id="stage0.G4",
        generated_at=ended_at,
        generator_git_commit=source.git_commit,
        environment_id=environment_id,
        config_hashes={"stage0.05": request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(root / PurePosixPath(reference)),
                schema_version="task-output-commit-v1",
            )
            for reference in evidence_refs
        ),
        checks=tuple(
            Stage0GateCheck(
                check_id=f"stage0.G4-{index + 1}",
                check_class=(
                    Stage0CheckClass.SAFETY
                    if name in {"run_no_clobber_and_resume_identity", "complete_clean_provenance_roundtrip"}
                    else Stage0CheckClass.CORRECTNESS
                ),
                status=Stage0CheckStatus.PASS,
                summary=name.replace("_", " "),
                measurements={"passed": passed},
                evidence_refs=evidence_refs,
            )
            for index, (name, passed) in enumerate(sorted(checks.items()))
        ),
    )
    gate = GateRecord(
        gate_id="stage0.G4",
        stage=0,
        status=GateStatus.PASS,
        checked_at=ended_at,
        measured={"checks": dict(sorted(checks.items()))},
        threshold={"required": "all correctness and safety checks PASS"},
        evidence_refs=evidence_refs,
    )
    envelope = _payload_with_hash(
        {
            "schema_version": "stage0-g4-provenance-evidence-v1",
            "status": "PASS",
            "checked_at": ended_at,
            "generator_git_commit": source.git_commit,
            "environment_hash": request.environment.environment_hash,
            "validation_report": validation,
            "provenance_record": provenance.to_dict(),
            "gate_record": gate.to_dict(),
            "gate_report": report.to_dict(),
        }
    )
    critical_refs = tuple(
        f"git-source/{source.git_commit}/{reference}"
        for reference in _CRITICAL_SOURCE_REFS
    )
    refs["provenance_record"] = store.publish(
        task_id=TASK_ID,
        artifact_kind="provenance_record",
        config_hash=request.config.config_hash,
        run_intent="formal",
        payload=envelope,
        formal_eligible=True,
        source_refs=tuple(dict.fromkeys((*evidence_refs, *critical_refs))),
    ).commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        message="Stage 0 G4 strict config, identity, seed and provenance contracts passed",
        metadata={
            "stage0_g4_specialized": True,
            "gate_id": "stage0.G4",
            "run_id": run.run_id,
            "seed_plan_hash": seed_plan.artifact_hash,
        },
    )


def run_formal_g4_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: tuple[str, ...],
) -> TaskRunResult:
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G4Error("G4_FORMAL_REQUEST_REQUIRED")
    return _publish_g4(request, Path(root).resolve(), store, source_refs=source_refs)


def validate_formal_g4_outputs(
    request: TaskExecutionRequest,
    root: Path,
    references: Mapping[str, str],
) -> GateRecord:
    if tuple(references) != (
        "resolved_config",
        "run_identity",
        "seed_plan",
        "provenance_record",
    ):
        raise Stage0G4Error("G4_RESTORE_OUTPUT_ORDER_INVALID")
    loaded = {
        kind: load_committed_task_artifact(root, reference, require_formal=True)
        for kind, reference in references.items()
    }
    config = ResolvedConfigV2.from_mapping(loaded["resolved_config"].payload)
    identity = RunIdentity.from_mapping(dict(loaded["run_identity"].payload))
    seed = SeedPlan.from_mapping(dict(loaded["seed_plan"].payload))
    envelope = _mapping(loaded["provenance_record"].payload, field="g4.envelope")
    if set(envelope) != _PROVENANCE_ENVELOPE_FIELDS:
        raise Stage0G4Error("G4_RESTORE_ENVELOPE_FIELDS_INVALID")
    claimed = envelope.get("artifact_hash")
    payload = dict(envelope)
    payload.pop("artifact_hash")
    if claimed != canonical_json_hash(payload):
        raise Stage0G4Error("G4_RESTORE_ENVELOPE_HASH_MISMATCH")
    provenance = ProvenanceRecord.from_mapping(
        _mapping(envelope.get("provenance_record"), field="g4.provenance")
    )
    gate = GateRecord.from_mapping(
        _mapping(envelope.get("gate_record"), field="g4.gate")
    )
    Stage0GateReport.from_mapping(
        _mapping(envelope.get("gate_report"), field="g4.gate_report")
    )
    validation = _mapping(envelope.get("validation_report"), field="g4.validation")
    validation_payload = dict(validation)
    validation_hash = validation_payload.pop("artifact_hash", None)
    checks = validation.get("checks")
    source = _capture_source()
    if (
        config.to_dict() != request.config.to_dict()
        or provenance.identity != identity
        or provenance.seed_plan_hash != seed.artifact_hash
        or provenance.config_hash != config.config_hash
        or provenance.resolved_config_ref != references["resolved_config"]
        or not provenance.formal_eligible
        or provenance.status is not ProvenanceStatus.COMPLETED
        or provenance.git_commit != source.git_commit
        or provenance.git_branch != source.git_branch
        or gate.gate_id != "stage0.G4"
        or gate.status is not GateStatus.PASS
        or envelope.get("status") != "PASS"
        or envelope.get("environment_hash") != request.environment.environment_hash
        or validation.get("status") != "PASS"
        or validation_hash != canonical_json_hash(validation_payload)
        or not isinstance(checks, Mapping)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise Stage0G4Error("G4_RESTORE_EVIDENCE_DRIFT")
    return gate


def _load_environment_gate_value(
    state: Stage0G3FormalState,
    root: Path,
    gate_id: str,
) -> GateRecord:
    return _gate_from_environment(state.environment, root, gate_id)[0]


def build_stage0_g4_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G3FormalState,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    resolution_loaded = load_committed_task_artifact(
        root, state.task_output_refs["asset_resolution"], require_formal=True
    )
    resolution = _mapping(resolution_loaded.payload, field="g3_resolution")
    model = _entry(resolution, "pythia-14m-step0", "model")
    tokenizer = _entry(resolution, "pythia-tokenizer", "tokenizer")
    data = _entry(resolution, "pile-selected-prefix", "pile")
    g2 = _load_environment_gate_value(state, root, "stage0.G2")
    g2_measured = _mapping(g2.measured, field="g2.measured")
    environment_id = g2_measured.get("environment_id")
    if not isinstance(environment_id, str) or not environment_id:
        raise Stage0G4Error("G4_BUILD_ENVIRONMENT_ID_INVALID")
    output_dir = f"evidence/stage0/tasks/05-{state.resolution_artifact_hash}"
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=output_dir,
        base_overrides={
            "identity": {
                "route": f"stage0-g4-{state.resolution_artifact_hash[:12]}",
            },
            "runtime": {
                "environment_id": environment_id,
                "device": "cuda",
                "dependency_profile": "stage0-g2-formal",
                "output_root": "runs/stage0",
                "temp_root": "tmp/stage0",
                "cache_root": "cache/stage0",
            },
            "model": {
                "asset_id": model["logical_name"],
                "revision": model["ready_manifest_sha256"],
                "tokenizer_asset_id": tokenizer["logical_name"],
                "initialization_id": model["candidate_id"],
                "architecture": "gpt-neox",
            },
            "data": {
                "asset_id": data["logical_name"],
                "revision": data["ready_manifest_sha256"],
                "split": "train",
                "sequence_length": 2048,
                "sampler": "with_replacement",
                "sampling_design": "iid_with_replacement",
                "statistical_unit": "sequence",
                "weight_unit": "effective_token",
                "weights_exogenous": True,
                "common_mean_assumption": True,
            },
            "loss": {
                "task_type": "causal_lm",
                "weighting": "effective_token",
            },
            "batching": {
                "per_device_batch_size": 1,
                "global_batch_size": 4,
                "microbatch_size": 1,
                "accumulation_steps": 1,
                "no_sync": False,
            },
            "distributed": {
                "world_size": 4,
                "backend": "nccl",
                "device_ids": [0, 1, 2, 3],
                "timeout_seconds": 180,
            },
        },
        v2_overrides={
            "providers": {
                "kind": "tiny",
                "task_type": "causal_lm",
                "task_name": "pile",
            },
            "launcher": {
                "kind": "torchrun",
                "backend": "nccl",
                "world_size": 4,
                "init_method": "env",
                "init_ref": None,
                "rendezvous_id": f"stage0-g4-{state.resolution_artifact_hash[:16]}",
                "max_restarts": 0,
            },
        },
    )


def execute_stage0_g4(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g3_index_ref: str,
) -> Stage0G4FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g3_formal_state(
        data_root=root,
        index_ref=g3_index_ref,
        expected_git_commit=binding.git_commit,
    )
    config = build_stage0_g4_config(binding=binding, data_root=root, state=state)
    formal_dir = f"evidence/stage0/g4-formal/{state.resolution_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    write_canonical_json(root / config_ref, config.to_dict())
    result = build_default_task_runtime(root).execute(
        config, environment=state.environment
    )
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G4Error(
            f"G4_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}"
        )
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=state.environment,
    )
    gate = validate_formal_g4_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g4_resolved_config": outputs["resolved_config"],
            "g4_run_identity": outputs["run_identity"],
            "g4_seed_plan": outputs["seed_plan"],
            "g4_provenance": outputs["provenance_record"],
            "gate_stage0_g4": outputs["provenance_record"],
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids | frozenset({gate.gate_id}),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    write_canonical_json(root / environment_ref, environment.to_dict())
    index_payload: dict[str, JSONValue] = {
        "schema_version": "stage0-g4-formalization-index-v1",
        "generator_git_commit": binding.git_commit,
        "checked_at": gate.checked_at,
        "g3_index_ref": state.index_ref,
        "g3_index_sha256": state.index_sha256,
        "g3_resolution_artifact_hash": state.resolution_artifact_hash,
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "task_output_refs": outputs,
        "gate_ref": outputs["provenance_record"],
        "environment_ref": environment_ref,
        "environment_hash": environment.environment_hash,
        "next_task_id": "stage0.06_single_gpu_smoke",
        "next_input_refs": list(outputs.values()),
    }
    index_payload["artifact_hash"] = canonical_json_hash(index_payload)
    index_ref = f"{formal_dir}/index.json"
    write_canonical_json(root / index_ref, index_payload)
    return Stage0G4FormalizationResult(
        environment=environment,
        task_output_refs=outputs,
        config_ref=config_ref,
        environment_ref=environment_ref,
        index_ref=index_ref,
    )


def _logical_ref_path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage0G4Error(f"G4_STATE_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G4Error(f"G4_STATE_LOGICAL_REF_ESCAPE:{field}")
    resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage0G4Error(f"G4_STATE_LOGICAL_REF_ESCAPE:{field}") from error
    return resolved


def load_stage0_g4_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G4FormalState:
    """Load G4 only after revalidating its G3 environment and all task commits."""

    root = Path(data_root).resolve(strict=True)
    index_path = _logical_ref_path(root, index_ref, field="index_ref")
    raw = _mapping(load_canonical_json(index_path), field="g4_index")
    expected = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g3_index_ref",
        "g3_index_sha256",
        "g3_resolution_artifact_hash",
        "config_ref",
        "config_hash",
        "task_output_refs",
        "gate_ref",
        "environment_ref",
        "environment_hash",
        "next_task_id",
        "next_input_refs",
        "artifact_hash",
    }
    if set(raw) != expected or raw.get("schema_version") != (
        "stage0-g4-formalization-index-v1"
    ):
        raise Stage0G4Error("G4_STATE_INDEX_FIELDS_OR_VERSION_INVALID")
    declared = raw.pop("artifact_hash")
    if declared != canonical_json_hash(raw):
        raise Stage0G4Error("G4_STATE_INDEX_HASH_MISMATCH")
    raw["artifact_hash"] = declared
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G4Error("G4_STATE_GENERATOR_COMMIT_MISMATCH")

    g3_state = load_stage0_g3_formal_state(
        data_root=root,
        index_ref=str(raw["g3_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    if (
        raw.get("g3_index_sha256") != g3_state.index_sha256
        or raw.get("g3_resolution_artifact_hash")
        != g3_state.resolution_artifact_hash
    ):
        raise Stage0G4Error("G4_STATE_G3_BINDING_MISMATCH")
    config_path = _logical_ref_path(root, raw["config_ref"], field="config_ref")
    config_value = _mapping(load_canonical_json(config_path), field="config")
    config = ResolvedConfigV2.from_mapping(config_value)
    if config.task_id != TASK_ID or config.config_hash != raw.get("config_hash"):
        raise Stage0G4Error("G4_STATE_CONFIG_MISMATCH")
    outputs = _mapping(raw["task_output_refs"], field="task_output_refs")
    if set(outputs) != {"resolved_config", "run_identity", "seed_plan", "provenance_record"}:
        raise Stage0G4Error("G4_STATE_OUTPUT_SET_INVALID")
    if any(not isinstance(value, str) for value in outputs.values()):
        raise Stage0G4Error("G4_STATE_OUTPUT_REF_INVALID")
    ordered_outputs = {
        key: str(outputs[key])
        for key in ("resolved_config", "run_identity", "seed_plan", "provenance_record")
    }
    request = TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=g3_state.environment,
    )
    gate = validate_formal_g4_outputs(
        request,
        root,
        ordered_outputs,
    )
    if raw.get("gate_ref") != outputs["provenance_record"]:
        raise Stage0G4Error("G4_STATE_GATE_REF_MISMATCH")

    environment_path = _logical_ref_path(
        root, raw["environment_ref"], field="environment_ref"
    )
    environment_value = _mapping(
        load_canonical_json(environment_path), field="environment"
    )
    environment = TaskRuntimeEnvironment.from_mapping(environment_value)
    if (
        environment.environment_hash != raw.get("environment_hash")
        or "stage0.G4" not in environment.passed_gate_ids
        or environment.evidence_refs.get("gate_stage0_g4") != raw.get("gate_ref")
        or raw.get("next_task_id") != "stage0.06_single_gpu_smoke"
        or set(raw.get("next_input_refs", [])) != set(outputs.values())
    ):
        raise Stage0G4Error("G4_STATE_ENVIRONMENT_OR_HANDOFF_MISMATCH")
    loaded_gate = load_committed_task_artifact(
        root, str(raw["gate_ref"]), require_formal=True
    )
    gate_hash = loaded_gate.payload.get("artifact_hash")
    if not isinstance(gate_hash, str) or _SHA256_RE.fullmatch(gate_hash) is None:
        raise Stage0G4Error("G4_STATE_GATE_ARTIFACT_HASH_INVALID")
    return Stage0G4FormalState(
        environment=environment,
        task_output_refs=ordered_outputs,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate_hash,
    )


__all__ = [
    "G4SourceBinding",
    "Stage0G4Error",
    "Stage0G4FormalState",
    "Stage0G4FormalizationResult",
    "TASK_ID",
    "build_stage0_g4_config",
    "execute_stage0_g4",
    "load_stage0_g4_formal_state",
    "run_formal_g4_task",
    "validate_formal_g4_outputs",
]
