"""Formalize Stage 1 S1.1 without replaying the Stage 0 producer chain."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any, Mapping

import yaml

from .atomic import sha256_file
from .contracts import (
    ContractFreeze,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.config_v2 import ResolvedConfigV2
from .evidence_reuse import (
    EvidenceReuseError,
    validate_evidence_reuse_attestation,
)
from .experiments import build_default_task_runtime
from .runtime import (
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunStatus,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from .stage0_g10 import (
    READINESS_SCHEMA,
    Stage0G10FormalState,
    validate_formal_g10_outputs,
)


TASK_ID = "stage1.01_entry_and_contract"
SCOPE_ID = "stage0.G0-G10"
INDEX_SCHEMA = "stage1-s1-1-formalization-index-v1"
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_STAGE0_GATE_IDS = (
    "stage0.G0-C",
    "stage0.G0-G",
    "stage0.G1",
    "stage0.G2",
    "stage0.G3",
    "stage0.G3-S1",
    "stage0.G3-S2",
    "stage0.G3-S4",
    "stage0.G3-S5",
    "stage0.G3-S6",
    "stage0.G4",
    "stage0.G5",
    "stage0.G6",
    "stage0.G7",
    "stage0.G7-LOGGING",
    "stage0.G8",
    "stage0.G8-C",
    "stage0.G8-S4",
    "stage0.G8-S5",
    "stage0.G9",
    "stage0.G10",
)


class Stage1S11Error(RuntimeError):
    """S1.1 cannot publish a complete formal entry boundary."""


@dataclass(frozen=True, slots=True)
class Stage1S11SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str


@dataclass(frozen=True, slots=True)
class Stage1S11FormalizationResult:
    index_ref: str
    config_ref: str
    environment_ref: str
    result_ref: str
    task_output_refs: Mapping[str, str]
    gate_artifact_hashes: Mapping[str, str]


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0:
        raise Stage1S11Error(f"S1_1_GIT_COMMAND_FAILED:{arguments[0]}")
    return completed.stdout.strip()


def capture_stage1_s1_1_source(repository: str | Path) -> Stage1S11SourceBinding:
    root = Path(repository).resolve(strict=True)
    commit = _git(root, "rev-parse", "HEAD")
    branch = _git(root, "branch", "--show-current")
    if _COMMIT_RE.fullmatch(commit) is None or not branch:
        raise Stage1S11Error("S1_1_GIT_IDENTITY_INVALID")
    if _git(root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S11Error("S1_1_FORMAL_REQUIRES_CLEAN_WORKTREE")
    return Stage1S11SourceBinding(root, commit, branch)


def _logical_path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S11Error(f"S1_1_LOGICAL_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S11Error(f"S1_1_LOGICAL_REF_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage1S11Error(f"S1_1_LOGICAL_REF_ESCAPE:{field}") from error
    return path


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1S11Error(f"S1_1_OBJECT_INVALID:{field}")
    return dict(value)


def _load_hashed(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(path), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage1S11Error(f"S1_1_ARTIFACT_IDENTITY_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def _write_or_verify(path: Path, value: Mapping[str, object]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage1S11Error(f"S1_1_IMMUTABLE_CONFLICT:{path.name}")
        return
    write_canonical_json(path, value)


def _yaml_mapping(path: Path, *, field: str) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _mapping(value, field=field)


def load_stage0_g10_handoff_snapshot(
    *,
    repository: str | Path,
    data_root: str | Path,
    index_ref: str,
) -> Stage0G10FormalState:
    """Validate the immutable G10 closure without replaying G0--G9.

    A downstream consumer needs G10's committed outputs, Gate, readiness bundle,
    and exact G9 lineage. Re-evaluating every historical producer against mutable
    current manifests would conflate evidence identity with present-day asset
    qualification. S1.1 validates the current assets separately in
    :func:`_handoff_assets`.
    """

    repository_root = Path(repository).resolve(strict=True)
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="g10_index_ref")
    raw = _load_hashed(
        index_path,
        schema="stage0-g10-formalization-index-v1",
        field="g10_index",
    )
    current_fields = {
        "schema_version",
        "generator_git_commit",
        "checked_at",
        "g9_index_ref",
        "g9_index_sha256",
        "g9_gate_artifact_hash",
        "g9_generator_git_commit",
        "reuse_attestation_ref",
        "reuse_attestation_sha256",
        "sync_observation_ref",
        "sync_observation_sha256",
        "config_ref",
        "config_hash",
        "input_environment_hash",
        "task_output_refs",
        "gate_ref",
        "gate_artifact_hash",
        "readiness_ref",
        "readiness_sha256",
        "readiness_artifact_hash",
        "environment_ref",
        "environment_hash",
        "next_task_id",
        "next_input_refs",
        "artifact_hash",
    }
    legacy_fields = current_fields - {
        "g9_generator_git_commit",
        "reuse_attestation_ref",
        "reuse_attestation_sha256",
    }
    actual_fields = frozenset(raw)
    legacy_index = actual_fields == frozenset(legacy_fields)
    generator = raw.get("generator_git_commit")
    if (
        actual_fields not in {frozenset(current_fields), frozenset(legacy_fields)}
        or not isinstance(generator, str)
        or _COMMIT_RE.fullmatch(generator) is None
    ):
        raise Stage1S11Error("S1_1_G10_INDEX_FIELDS_INVALID")

    g9_ref = raw.get("g9_index_ref")
    g9_path = _logical_path(root, g9_ref, field="g9_index_ref")
    g9 = _load_hashed(
        g9_path,
        schema="stage0-g9-formalization-index-v1",
        field="g9_index",
    )
    g9_generator = g9.get("generator_git_commit")
    g9_gate_hash = g9.get("gate_artifact_hash")
    if (
        not isinstance(g9_generator, str)
        or _COMMIT_RE.fullmatch(g9_generator) is None
        or not isinstance(g9_gate_hash, str)
        or re.fullmatch(r"[0-9a-f]{64}", g9_gate_hash) is None
    ):
        raise Stage1S11Error("S1_1_G9_LINEAGE_INVALID")
    declared_g9_generator = generator if legacy_index else raw.get(
        "g9_generator_git_commit"
    )
    reuse_ref = None if legacy_index else raw.get("reuse_attestation_ref")
    reuse_sha = None if legacy_index else raw.get("reuse_attestation_sha256")
    if declared_g9_generator != g9_generator:
        raise Stage1S11Error("S1_1_G9_GENERATOR_BINDING_INVALID")
    if g9_generator == generator:
        if reuse_ref is not None or reuse_sha is not None:
            raise Stage1S11Error("S1_1_G10_REUSE_ATTESTATION_UNNECESSARY")
    else:
        if not isinstance(reuse_ref, str) or not isinstance(reuse_sha, str):
            raise Stage1S11Error("S1_1_G10_REUSE_ATTESTATION_MISSING")
        reuse_path = _logical_path(root, reuse_ref, field="g10_reuse_attestation_ref")
        reuse_raw = _mapping(
            load_canonical_json(reuse_path), field="g10_reuse_attestation"
        )
        reuse_branch = reuse_raw.get("consumer_git_branch")
        if sha256_file(reuse_path) != reuse_sha or not isinstance(reuse_branch, str):
            raise Stage1S11Error("S1_1_G10_REUSE_ATTESTATION_IDENTITY_INVALID")
        try:
            validate_evidence_reuse_attestation(
                repository=repository_root,
                data_root=root,
                attestation_ref=reuse_ref,
                producer_commit=g9_generator,
                consumer_commit=generator,
                consumer_branch=reuse_branch,
                scope_id="stage0.G0-G9",
                source_evidence_ref=str(g9_ref),
                required_gate_ids=sorted(
                    gate_id
                    for gate_id in REQUIRED_STAGE0_GATE_IDS
                    if gate_id != "stage0.G10"
                ),
            )
        except EvidenceReuseError as error:
            raise Stage1S11Error(
                f"S1_1_G10_REUSE_ATTESTATION_INVALID:{error}"
            ) from error

    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, raw["config_ref"], field="g10_config_ref")
            ),
            field="g10_config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(
                    root, raw["environment_ref"], field="g10_environment_ref"
                )
            ),
            field="g10_environment",
        )
    )
    outputs_raw = _mapping(raw.get("task_output_refs"), field="g10_task_outputs")
    if set(outputs_raw) != set(config.task_definition.artifact_kinds):
        raise Stage1S11Error("S1_1_G10_TASK_OUTPUT_SET_INVALID")
    outputs = {
        kind: str(outputs_raw[kind]) for kind in config.task_definition.artifact_kinds
    }
    input_refs = dict(environment.evidence_refs)
    for key in (
        "g10_delivery_manifest",
        "g10_worklog",
        "g10_sync_report",
        "gate_stage0_g10",
        "stage0_readiness",
        "stage1_handoff",
    ):
        input_refs.pop(key, None)
    input_environment = TaskRuntimeEnvironment(
        capabilities=environment.capabilities,
        frozen_contract_stages=environment.frozen_contract_stages,
        passed_gate_ids=environment.passed_gate_ids - frozenset({"stage0.G10"}),
        estimator_decision_ref=environment.estimator_decision_ref,
        evidence_refs=input_refs,
    )
    request = TaskExecutionRequest(config, config.task_definition, input_environment)
    gate = validate_formal_g10_outputs(request, root, outputs)
    readiness_path = _logical_path(
        root, raw["readiness_ref"], field="g10_readiness_ref"
    )
    readiness = _load_hashed(
        readiness_path,
        schema=READINESS_SCHEMA,
        field="g10_readiness",
    )
    next_inputs = raw.get("next_input_refs")
    if (
        raw.get("g9_index_sha256") != sha256_file(g9_path)
        or raw.get("g9_gate_artifact_hash") != g9_gate_hash
        or raw.get("sync_observation_sha256")
        != sha256_file(
            _logical_path(
                root, raw["sync_observation_ref"], field="g10_sync_observation_ref"
            )
        )
        or config.config_hash != raw.get("config_hash")
        or input_environment.environment_hash != raw.get("input_environment_hash")
        or environment.environment_hash != raw.get("environment_hash")
        or "stage0.G10" not in environment.passed_gate_ids
        or raw.get("gate_ref") != outputs["sync_report"]
        or raw.get("gate_artifact_hash") != gate.artifact_hash
        or raw.get("readiness_sha256") != sha256_file(readiness_path)
        or raw.get("readiness_artifact_hash") != readiness["artifact_hash"]
        or raw.get("next_task_id") != TASK_ID
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(outputs.values())
    ):
        raise Stage1S11Error("S1_1_G10_HANDOFF_INVALID")
    return Stage0G10FormalState(
        environment=environment,
        task_output_refs=outputs,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate.artifact_hash,
        readiness_ref=str(raw["readiness_ref"]),
        readiness_artifact_hash=str(readiness["artifact_hash"]),
        g9_index_ref=str(g9_ref),
        generator_git_commit=generator,
        legacy_index=legacy_index,
    )


def _handoff_assets(
    data_root: Path,
    state: Stage0G10FormalState,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    handoff_ref = state.environment.evidence_refs.get("stage1_handoff")
    if handoff_ref is None:
        raise Stage1S11Error("S1_1_HANDOFF_REF_MISSING")
    handoff = _load_hashed(
        _logical_path(data_root, handoff_ref, field="stage1_handoff"),
        schema="stage0-g10-stage1-handoff-v1",
        field="stage1_handoff",
    )
    if handoff.get("status") != "READY_FOR_STAGE1_ENTRY":
        raise Stage1S11Error("S1_1_HANDOFF_NOT_READY")
    reusable = handoff.get("reusable_assets")
    if not isinstance(reusable, list):
        raise Stage1S11Error("S1_1_HANDOFF_ASSETS_INVALID")
    by_name = {
        str(item.get("logical_name")): dict(item)
        for item in reusable
        if isinstance(item, Mapping)
    }

    def load_asset(name: str) -> dict[str, Any]:
        descriptor = by_name.get(name)
        if descriptor is None:
            raise Stage1S11Error(f"S1_1_HANDOFF_ASSET_MISSING:{name}")
        manifest_path = _logical_path(
            data_root, descriptor.get("manifest_ref"), field=name
        )
        manifest = _mapping(
            load_canonical_json(manifest_path),
            field=f"manifest:{name}",
        )
        if (
            manifest.get("state") != "ready"
            or manifest.get("name") != name
            or manifest.get("revision") != descriptor.get("revision")
        ):
            raise Stage1S11Error(f"S1_1_HANDOFF_ASSET_IDENTITY_DRIFT:{name}")
        qualification = _load_hashed(
            _logical_path(
                data_root,
                f"manifests/qualifications/{name}.json",
                field=f"qualification:{name}",
            ),
            schema="stage0-asset-qualification-v1",
            field=f"qualification:{name}",
        )
        checks = qualification.get("checks")
        if (
            qualification.get("formal") is not True
            or qualification.get("asset_id") != manifest.get("asset_id")
            or qualification.get("verified_manifest_sha256") != sha256_file(manifest_path)
            or not isinstance(checks, list)
            or not checks
            or any(
                not isinstance(check, Mapping) or check.get("status") != "PASS"
                for check in checks
            )
        ):
            raise Stage1S11Error(f"S1_1_ASSET_QUALIFICATION_INVALID:{name}")
        return manifest

    return (
        handoff,
        load_asset("pythia-14m-step0"),
        load_asset("pile-selected-prefix"),
        load_asset("pythia-tokenizer"),
    )


def validate_stage0_g10_reuse(
    *,
    binding: Stage1S11SourceBinding,
    data_root: str | Path,
    state: Stage0G10FormalState,
    reuse_attestation_ref: str | None,
) -> dict[str, Any] | None:
    root = Path(data_root).resolve(strict=True)
    required = sorted(REQUIRED_STAGE0_GATE_IDS)
    if not set(required).issubset(state.environment.passed_gate_ids):
        raise Stage1S11Error("S1_1_STAGE0_GATE_SET_INCOMPLETE")
    if state.generator_git_commit == binding.git_commit:
        if reuse_attestation_ref is not None:
            raise Stage1S11Error("S1_1_REUSE_ATTESTATION_UNNECESSARY")
        return None
    if reuse_attestation_ref is None:
        raise Stage1S11Error("S1_1_REUSE_ATTESTATION_REQUIRED")
    try:
        return validate_evidence_reuse_attestation(
            repository=binding.repository,
            data_root=root,
            attestation_ref=reuse_attestation_ref,
            producer_commit=state.generator_git_commit,
            consumer_commit=binding.git_commit,
            consumer_branch=binding.git_branch,
            scope_id=SCOPE_ID,
            source_evidence_ref=state.index_ref,
            required_gate_ids=required,
        )
    except EvidenceReuseError as error:
        raise Stage1S11Error(f"S1_1_REUSE_ATTESTATION_INVALID:{error}") from error


def build_stage1_s1_1_config(
    *,
    binding: Stage1S11SourceBinding,
    data_root: str | Path,
    state: Stage0G10FormalState,
    reuse_attestation_ref: str | None,
    output_dir: str,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    _handoff, model, data, tokenizer = _handoff_assets(root, state)
    science = _yaml_mapping(
        binding.repository / "configs/run-ready/layers/formal-stage1-pythia14m.yaml",
        field="science_layer",
    )
    science["identity"].update(
        {"task": TASK_ID, "route": "stage1-s1-1-entry-contract"}
    )
    science["runtime"].update(
        {
            "output_root": output_dir,
            "cache_root": "cache/stage1/s1-1",
            "temp_root": f"tmp/stage1/s1-1/{binding.git_commit[:12]}",
        }
    )
    model_metadata = _mapping(model.get("metadata"), field="model.metadata")
    science["model"].update(
        {
            "asset_id": str(model["asset_id"]),
            "revision": str(model["revision"]),
            "tokenizer_asset_id": str(tokenizer["asset_id"]),
            "initialization_id": str(model_metadata["initialization_id"]),
            "architecture": str(model_metadata["architecture"]),
        }
    )
    science["data"].update(
        {"asset_id": str(data["asset_id"]), "revision": str(data["revision"])}
    )
    base_fixture = load_canonical_json(
        binding.repository / "configs/local-fixtures/resolved-config-v1.json"
    )
    base = ResolvedConfig.resolve(_mapping(base_fixture, field="base_fixture"), science)

    execution = _yaml_mapping(
        binding.repository / "configs/run-ready/v2/stage1-pythia14m-formal.yaml",
        field="execution_layer",
    )
    execution.setdefault("execution", {}).update(
        {"timeout_seconds": 1800, "max_attempts": 1}
    )
    execution["providers"].update(
        {
            # Formal providers resolve assets only through the verified Stage 0 G3/G10
            # logical handoff. Direct manifest/root overrides would create a second,
            # weaker resolution path and are therefore intentionally null.
            "model_manifest_ref": None,
            "model_root_ref": None,
            "data_manifest_ref": None,
            "data_root_ref": None,
            "tokenizer_manifest_ref": None,
            "tokenizer_root_ref": None,
            "local_files_only": True,
            "trust_remote_code": False,
        }
    )
    execution["orchestration"] = {
        "route_spec_ref": state.index_ref,
        "quadrature_decision_ref": reuse_attestation_ref,
        "matrix_ref": None,
        "input_result_refs": [
            state.task_output_refs[kind]
            for kind in ("delivery_manifest", "worklog", "sync_report")
        ],
    }
    execution["artifacts"]["output_dir"] = output_dir
    return ResolvedConfigV2.resolve(base, task_id=TASK_ID, overrides=execution)


def validate_formal_stage1_s1_1_outputs(
    *,
    data_root: str | Path,
    result: TaskRunResult,
    expected_git_commit: str,
    reuse_attestation_ref: str | None,
) -> dict[str, str]:
    root = Path(data_root).resolve(strict=True)
    if (
        result.status is not TaskRunStatus.PASS
        or not result.formal_eligible
        or result.task_id != TASK_ID
    ):
        raise Stage1S11Error("S1_1_TASK_RESULT_NOT_FORMAL_PASS")
    loaded = {
        kind: load_committed_task_artifact(root, reference, require_formal=True)
        for kind, reference in result.artifact_refs.items()
    }
    contract_payload = loaded["stage_contract"].payload
    core = contract_payload.get("core_evidence")
    if not isinstance(core, Mapping):
        raise Stage1S11Error("S1_1_CONTRACT_CORE_MISSING")
    freeze = ContractFreeze.from_mapping(
        _mapping(core.get("contract_freeze"), field="contract_freeze")
    )
    entry = _mapping(core.get("entry_snapshot"), field="entry_snapshot")
    repository = _mapping(entry.get("repository"), field="entry.repository")
    external = _mapping(entry.get("external_evidence"), field="entry.external")
    reuse = _mapping(external.get("g10_reuse"), field="entry.g10_reuse")
    gates_raw = core.get("formal_gate_records")
    gates = (
        [
            GateRecord.from_mapping(_mapping(item, field="formal_gate"))
            for item in gates_raw
        ]
        if isinstance(gates_raw, list)
        else []
    )
    statuses = {gate.gate_id: gate for gate in gates}
    matrix = _mapping(core.get("requirements_matrix"), field="requirements_matrix")
    summary = _mapping(matrix.get("summary"), field="requirements_matrix.summary")
    if (
        not freeze.formal_eligible
        or freeze.stage != 1
        or entry.get("formal_eligible") is not True
        or entry.get("failed_checks") != []
        or repository.get("head") != expected_git_commit
        or reuse.get("status") != "PASS"
        or reuse.get("reuse_attestation_ref") != reuse_attestation_ref
        or set(statuses) != {"stage1.G1-ENTRY", "stage1.G1-CONTRACT"}
        or any(gate.status is not GateStatus.PASS for gate in statuses.values())
        or summary.get("formal_blocked") != 0
    ):
        raise Stage1S11Error("S1_1_FORMAL_OUTPUT_IDENTITY_INVALID")
    expected_sources = {str(reuse["g10_index_ref"])}
    if reuse_attestation_ref is not None:
        expected_sources.add(reuse_attestation_ref)
    if any(
        not expected_sources.issubset(set(artifact.source_refs))
        for artifact in loaded.values()
    ):
        raise Stage1S11Error("S1_1_FORMAL_OUTPUT_SOURCE_LINEAGE_MISSING")
    return {gate_id: gate.artifact_hash for gate_id, gate in sorted(statuses.items())}


def load_stage1_s1_1_formalization(
    *,
    repository: str | Path,
    data_root: str | Path,
    index_ref: str,
) -> dict[str, Any]:
    """Reload every identity behind a published S1.1 PASS index."""

    repository_root = Path(repository).resolve(strict=True)
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    index = _load_hashed(index_path, schema=INDEX_SCHEMA, field="index")
    expected_fields = {
        "schema_version",
        "status",
        "generator_git_commit",
        "git_branch",
        "checked_at",
        "g10_generator_git_commit",
        "g10_index_ref",
        "g10_index_sha256",
        "g10_legacy_index",
        "reuse_attestation_ref",
        "reuse_attestation_sha256",
        "reuse_artifact_hash",
        "config_ref",
        "config_hash",
        "environment_ref",
        "environment_hash",
        "result_ref",
        "result_hash",
        "task_output_refs",
        "gate_artifact_hashes",
        "next_task_id",
        "artifact_hash",
    }
    commit = index.get("generator_git_commit")
    branch = index.get("git_branch")
    if (
        set(index) != expected_fields
        or index.get("status") != "PASS"
        or not isinstance(commit, str)
        or _COMMIT_RE.fullmatch(commit) is None
        or not isinstance(branch, str)
        or not branch
    ):
        raise Stage1S11Error("S1_1_INDEX_FIELDS_OR_SOURCE_INVALID")
    binding = Stage1S11SourceBinding(repository_root, commit, branch)
    state = load_stage0_g10_handoff_snapshot(
        repository=repository_root,
        data_root=root,
        index_ref=str(index["g10_index_ref"]),
    )
    raw_reuse_ref = index.get("reuse_attestation_ref")
    reuse_ref = raw_reuse_ref if isinstance(raw_reuse_ref, str) else None
    attestation = validate_stage0_g10_reuse(
        binding=binding,
        data_root=root,
        state=state,
        reuse_attestation_ref=reuse_ref,
    )
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, index["config_ref"], field="config_ref")
            ),
            field="config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, index["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    result = TaskRunResult.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, index["result_ref"], field="result_ref")
            ),
            field="result",
        )
    )
    gates = validate_formal_stage1_s1_1_outputs(
        data_root=root,
        result=result,
        expected_git_commit=commit,
        reuse_attestation_ref=reuse_ref,
    )
    expected_reuse_hash = None if attestation is None else attestation["artifact_hash"]
    expected_reuse_sha = (
        None
        if reuse_ref is None
        else sha256_file(_logical_path(root, reuse_ref, field="reuse_attestation_ref"))
    )
    if (
        state.generator_git_commit != index.get("g10_generator_git_commit")
        or state.index_sha256 != index.get("g10_index_sha256")
        or state.legacy_index is not index.get("g10_legacy_index")
        or expected_reuse_sha != index.get("reuse_attestation_sha256")
        or expected_reuse_hash != index.get("reuse_artifact_hash")
        or config.task_id != TASK_ID
        or config.config_hash != index.get("config_hash")
        or environment.to_dict() != state.environment.to_dict()
        or environment.environment_hash != index.get("environment_hash")
        or result.result_hash != index.get("result_hash")
        or dict(result.artifact_refs) != index.get("task_output_refs")
        or gates != index.get("gate_artifact_hashes")
        or index.get("next_task_id")
        != "stage1.02_architecture_and_parameter_registry"
    ):
        raise Stage1S11Error("S1_1_INDEX_HANDOFF_INVALID")
    return index


def execute_stage1_s1_1(
    *,
    repository: str | Path,
    data_root: str | Path,
    g10_index_ref: str,
    reuse_attestation_ref: str | None,
    attempt_id: str,
) -> Stage1S11FormalizationResult:
    if re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", attempt_id) is None:
        raise Stage1S11Error("S1_1_ATTEMPT_ID_INVALID")
    binding = capture_stage1_s1_1_source(repository)
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g10_handoff_snapshot(
        repository=binding.repository,
        data_root=root,
        index_ref=g10_index_ref,
    )
    attestation = validate_stage0_g10_reuse(
        binding=binding,
        data_root=root,
        state=state,
        reuse_attestation_ref=reuse_attestation_ref,
    )
    formal_dir = (
        f"evidence/stage1/s1-1-formal/{binding.git_commit}/{attempt_id}"
    )
    task_dir = f"evidence/stage1/tasks/01-{binding.git_commit[:12]}-{attempt_id}"
    config = build_stage1_s1_1_config(
        binding=binding,
        data_root=root,
        state=state,
        reuse_attestation_ref=reuse_attestation_ref,
        output_dir=task_dir,
    )
    config_ref = f"{formal_dir}/resolved-config.json"
    environment_ref = f"{formal_dir}/environment.json"
    result_ref = f"{formal_dir}/result.json"
    _write_or_verify(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    _write_or_verify(
        _logical_path(root, environment_ref, field="environment_ref"),
        state.environment.to_dict(),
    )
    runtime = build_default_task_runtime(root)
    blockers = runtime.preflight(config, environment=state.environment)
    if blockers:
        raise Stage1S11Error(
            "S1_1_PREFLIGHT_BLOCKED:"
            + ",".join(f"{item.code.value}:{item.requirement}" for item in blockers)
        )
    result = runtime.execute(config, environment=state.environment)
    _write_or_verify(_logical_path(root, result_ref, field="result_ref"), result.to_dict())
    gate_hashes = validate_formal_stage1_s1_1_outputs(
        data_root=root,
        result=result,
        expected_git_commit=binding.git_commit,
        reuse_attestation_ref=reuse_attestation_ref,
    )
    checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    index: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA,
        "status": "PASS",
        "generator_git_commit": binding.git_commit,
        "git_branch": binding.git_branch,
        "checked_at": checked_at,
        "g10_generator_git_commit": state.generator_git_commit,
        "g10_index_ref": state.index_ref,
        "g10_index_sha256": state.index_sha256,
        "g10_legacy_index": state.legacy_index,
        "reuse_attestation_ref": reuse_attestation_ref,
        "reuse_attestation_sha256": (
            None
            if reuse_attestation_ref is None
            else sha256_file(
                _logical_path(root, reuse_attestation_ref, field="reuse_attestation_ref")
            )
        ),
        "reuse_artifact_hash": None if attestation is None else attestation["artifact_hash"],
        "config_ref": config_ref,
        "config_hash": config.config_hash,
        "environment_ref": environment_ref,
        "environment_hash": state.environment.environment_hash,
        "result_ref": result_ref,
        "result_hash": result.result_hash,
        "task_output_refs": dict(result.artifact_refs),
        "gate_artifact_hashes": gate_hashes,
        "next_task_id": "stage1.02_architecture_and_parameter_registry",
    }
    index["artifact_hash"] = canonical_json_hash(index)
    index_ref = f"{formal_dir}/index.json"
    _write_or_verify(_logical_path(root, index_ref, field="index_ref"), index)
    load_stage1_s1_1_formalization(
        repository=binding.repository,
        data_root=root,
        index_ref=index_ref,
    )
    return Stage1S11FormalizationResult(
        index_ref=index_ref,
        config_ref=config_ref,
        environment_ref=environment_ref,
        result_ref=result_ref,
        task_output_refs=dict(result.artifact_refs),
        gate_artifact_hashes=gate_hashes,
    )


__all__ = [
    "INDEX_SCHEMA",
    "REQUIRED_STAGE0_GATE_IDS",
    "SCOPE_ID",
    "TASK_ID",
    "Stage1S11Error",
    "Stage1S11FormalizationResult",
    "Stage1S11SourceBinding",
    "build_stage1_s1_1_config",
    "capture_stage1_s1_1_source",
    "execute_stage1_s1_1",
    "load_stage1_s1_1_formalization",
    "load_stage0_g10_handoff_snapshot",
    "validate_formal_stage1_s1_1_outputs",
    "validate_stage0_g10_reuse",
]
