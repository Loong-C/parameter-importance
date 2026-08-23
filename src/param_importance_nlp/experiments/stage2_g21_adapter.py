"""Thin, fail-closed consumer for the formal Stage 2 G2.1 handoff.

The S2.2 runner emits a candidate (``handoff_manifest``,
``fixed_state_contract`` and ``gate_record``) but it does not sign G2.1.  This
module is the independent signer.  It consumes the three formal S2.1 commits,
the seven *real* S1.10/S1.11 TaskArtifact commits, the persisted formal
ResolvedConfig and the current handoff evidence.  A single formal
``stage2.G2.1`` GateRecord TaskArtifact is published only after all of those
objects have been reloaded successfully.

Stage1 validation is intentionally delegated to the reviewed S1.11 producer
authority in ``ops/stage1/formalize_s1_11.py``.  The adapter does not copy its
schema validator or make a PASS-looking JSON object into an authority.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import subprocess

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.g21_formal_handoff import (
    ALLOWED_DEVICES,
    AUTH_HASH,
    EXCLUDED_PCI,
    G21FormalHandoffError,
    load_g21_formal_handoff,
)
from ..contracts.jsonio import JSONValue, load_canonical_json
from ..contracts.stage2_authorization import (
    AUTHORIZATION_REF,
    load_stage2_authorization,
)
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    PublishedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from .preregistration import (
    build_stage2_hypothesis_contract,
    validate_stage2_preregistration,
)


GATE_ID = "stage2.G2.1"
TASK_ID = "stage2.02_stage1_handoff_and_fixed_state_contract"
S201_TASK_ID = "stage2.01_scope_hypotheses_and_preregistration"
S111_TASK_ID = "stage1.11_reporting_and_exit_gate"
S110_TASK_ID = "stage1.10_checkpoint_resume_and_artifacts"

S201_ARTIFACT_KINDS: tuple[str, ...] = (
    "preregistration",
    "hypothesis_contract",
    "gate_record",
)
S110_ARTIFACT_KINDS: tuple[str, ...] = (
    "training_state_manifest",
    "resume_equivalence_report",
    "gate_record",
)
S111_ARTIFACT_KINDS: tuple[str, ...] = (
    "stage_report",
    "requirements_matrix",
    "gate_summary",
    "delivery_manifest",
)
STAGE1_ARTIFACT_KINDS: tuple[str, ...] = S110_ARTIFACT_KINDS + S111_ARTIFACT_KINDS
ARTIFACT_KINDS: tuple[str, ...] = S201_ARTIFACT_KINDS
ADAPTER_SCHEMA_VERSION = "stage2-g2.1-gate-adapter-v1"

# These are the append-only r1 evidence bytes, not the old tracked BLOCKED
# report.  Keep the byte identity here because a caller-controlled handoff
# object must never be able to select a different smoke or authorization.
HANDOFF_REF = "reports/stage2/s2.2/g2.1-formal-stage1-handoff-evidence-20260823-r1.json"
HANDOFF_SHA256 = "50bb320b2e0cea74ccd8e388d7719373f40932b4d1c546ac0aef385972f4df7a"
HANDOFF_ARTIFACT_HASH = "259831e2a1b16afbbef34c9cea602e636756b0f6173d1a8f4c32ec554c653f79"
SMOKE_REF = "evidence/stage2/s202/current-gpu-smoke/ab0a530/current-20260823-02/report.json"
SMOKE_SHA256 = "3c7a6b1428bb3676c02930b58f7282e8943e31e47458be5de5690c017d1555d9"

# ``formalize_s1_11`` owns these constants and the complete producer
# validation.  They are repeated only as stable input names for the seven
# TaskArtifact commit map; no Stage1 schema is reimplemented here.
S111_R4_INDEX_REF = (
    "evidence/stage1/s1-11-formal/"
    "3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r4-20260821/index.json"
)
S110_R12_INDEX_REF = (
    "evidence/stage1/s1-10-formal/"
    "fbb09e4d338125954fc614c745cf7ab88c58d3b2/s1-10-r12-20260821/index.json"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_AGENT_FILES = (
    "Agent/git.md",
    "Agent/local_temp.md",
    "Agent/remote_access.md",
    "Agent/server.md",
    "Agent/sync.md",
    "Agent/worklogs.md",
)
_SOURCE_PATHS = (
    "src/param_importance_nlp/experiments/stage2_g21_adapter.py",
    "src/param_importance_nlp/contracts/g21_formal_handoff.py",
    "src/param_importance_nlp/contracts/stage2_authorization.py",
    "src/param_importance_nlp/contracts/config_v2.py",
    "src/param_importance_nlp/runtime/task_artifacts.py",
    "ops/stage1/formalize_s1_11.py",
    "plan/stage2/02_stage1_handoff_and_fixed_state_contract.md",
    "schemas/shared/stage2-fixed-state-contract-v1.json",
    "schemas/shared/stage2-stage1-handoff-manifest-v1.json",
    "docs/stage1-handoff.md",
)


class G21Blocked(RuntimeError):
    """A missing, stale, altered, or unsafe formal input."""


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise G21Blocked(f"{field}:SHA256_REQUIRED")
    return value


def _commit(value: object, field: str) -> str:
    if not isinstance(value, str) or _COMMIT.fullmatch(value) is None:
        raise G21Blocked(f"{field}:COMMIT_REQUIRED")
    return value


def _logical(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise G21Blocked(f"{field}:LOGICAL_PATH_REQUIRED")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise G21Blocked(f"{field}:PATH_ESCAPE")
    return path.as_posix()


def _check_chain(path: Path, root: Path, field: str) -> None:
    """Reject symlinks at every existing component of a root-relative path."""

    root = root.absolute()
    path = path.absolute()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise G21Blocked(f"{field}:PATH_ESCAPE") from error
    current = root
    if current.is_symlink():
        raise G21Blocked(f"{field}:SYMLINK_FORBIDDEN")
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise G21Blocked(f"{field}:SYMLINK_FORBIDDEN")


def _root(value: str | Path, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.absolute()
    if path.is_symlink() or not path.is_dir():
        raise G21Blocked(f"{field}:DIRECTORY_REQUIRED")
    _check_chain(path, path, field)
    return path


def _resolve(root: Path, value: object, field: str) -> Path:
    logical = _logical(value, field)
    candidate = root.joinpath(*PurePosixPath(logical).parts)
    _check_chain(candidate, root, field)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root.resolve(strict=False))
    except ValueError as error:
        raise G21Blocked(f"{field}:PATH_ESCAPE") from error
    return resolved


def _file_sha256(path: Path, field: str) -> str:
    try:
        if not path.is_file():
            raise G21Blocked(f"{field}:FILE_REQUIRED")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except G21Blocked:
        raise
    except OSError as error:
        raise G21Blocked(f"{field}:UNREADABLE") from error


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise G21Blocked("repository_git:COMMAND_FAILED") from error
    if result.returncode:
        raise G21Blocked(f"repository_git:FAILED:{args[0]}")
    return result.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise G21Blocked("repository_git:COMMAND_FAILED") from error
    if result.returncode:
        raise G21Blocked(f"repository_git:FAILED:{args[0]}")
    return bytes(result.stdout)


def _repository_identity(repository: Path) -> dict[str, JSONValue]:
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise G21Blocked("repository:WORKTREE_DIRTY")
    head = _commit(_git(repository, "rev-parse", "HEAD"), "repository.head")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    if not re.fullmatch(r"[0-9a-f]{40}", tree):
        raise G21Blocked("repository.tree:OBJECT_INVALID")
    source_hashes: dict[str, JSONValue] = {}
    for relative in _SOURCE_PATHS:
        path = _resolve(repository, relative, f"repository.source.{relative}")
        actual = path.read_bytes()
        committed = _git_bytes(repository, "show", f"{head}:{relative}")
        # Git autocrlf is a checkout detail; source provenance also records the
        # actual worktree SHA while this check prevents semantic worktree drift.
        if committed.replace(b"\r\n", b"\n") != actual.replace(b"\r\n", b"\n"):
            raise G21Blocked(f"repository.source.{relative}:WORKTREE_DRIFT")
        source_hashes[relative] = {
            "sha256": hashlib.sha256(actual).hexdigest(),
            "git_blob": _git(repository, "rev-parse", f"HEAD:{relative}"),
        }
    agent_hashes: dict[str, JSONValue] = {}
    for relative in _AGENT_FILES:
        path = _resolve(repository, relative, f"repository.agent.{relative}")
        agent_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "head": head,
        "tree": tree,
        "worktree_clean": True,
        "source_hashes": source_hashes,
        "agent_sha256": agent_hashes,
    }


def _load_task(root: Path, reference: str, *, task_id: str, kind: str) -> LoadedTaskArtifact:
    reference = _logical(reference, f"{kind}.commit_ref")
    path = _resolve(root, reference, f"{kind}.commit_ref")
    try:
        commit = load_canonical_json(path)
        if not isinstance(commit, Mapping) or not isinstance(commit.get("object_ref"), str):
            raise ValueError("object_ref missing")
        _resolve(root, commit["object_ref"], f"{kind}.object_ref")
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise G21Blocked(f"{kind}:TASK_ARTIFACT_INVALID") from error
    if loaded.identity.task_id != task_id or loaded.identity.artifact_kind != kind:
        raise G21Blocked(f"{kind}:TASK_ARTIFACT_IDENTITY_INVALID")
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise G21Blocked(f"{kind}:FORMAL_ENVELOPE_REQUIRED")
    return loaded


def _refs(value: object, kinds: tuple[str, ...], field: str) -> dict[str, str]:
    if isinstance(value, Mapping):
        if set(value) != set(kinds):
            raise G21Blocked(f"{field}:EXACT_KIND_SET_REQUIRED")
        raw = {kind: value[kind] for kind in kinds}
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != len(kinds):
            raise G21Blocked(f"{field}:EXACT_COUNT_REQUIRED")
        raw = dict(zip(kinds, value, strict=True))
    else:
        raise G21Blocked(f"{field}:MAPPING_REQUIRED")
    result = {kind: _logical(raw[kind], f"{field}.{kind}") for kind in kinds}
    if len(set(result.values())) != len(result):
        raise G21Blocked(f"{field}:DUPLICATE_REF")
    return result


def _stage1_refs(value: object) -> dict[str, str]:
    """Accept the canonical flat map and the explicit s110/s111 nested map."""

    if isinstance(value, Mapping) and set(value) == {"s110", "s111"}:
        s110 = _refs(value["s110"], S110_ARTIFACT_KINDS, "stage1_artifact_refs.s110")
        s111 = _refs(value["s111"], S111_ARTIFACT_KINDS, "stage1_artifact_refs.s111")
        return {**s110, **s111}
    return _refs(value, STAGE1_ARTIFACT_KINDS, "stage1_artifact_refs")


def _load_s201(root: Path, references: object) -> tuple[dict[str, LoadedTaskArtifact], str]:
    refs = _refs(references, S201_ARTIFACT_KINDS, "s201_artifact_refs")
    loaded = {
        kind: _load_task(root, refs[kind], task_id=S201_TASK_ID, kind=kind)
        for kind in S201_ARTIFACT_KINDS
    }
    configs = {item.identity.config_hash for item in loaded.values()}
    if len(configs) != 1:
        raise G21Blocked("s201:CONFIG_HASH_NOT_SHARED")
    source_sets = {item.source_refs for item in loaded.values()}
    if len(source_sets) != 1:
        raise G21Blocked("s201:SOURCE_REFS_NOT_SHARED")
    prereg = loaded["preregistration"].payload
    hypothesis = loaded["hypothesis_contract"].payload
    candidate = loaded["gate_record"].payload
    try:
        validate_stage2_preregistration(prereg)
    except (TypeError, ValueError, KeyError) as error:
        raise G21Blocked("s201:CONTRACT_PAYLOAD_INVALID") from error
    provenance = prereg.get("provenance")
    if not isinstance(provenance, Mapping) or not isinstance(provenance.get("upstream_binding_hash"), str):
        raise G21Blocked("s201:PREREGISTRATION_PROVENANCE_INVALID")
    try:
        expected_hypothesis = build_stage2_hypothesis_contract(
            prereg,
            upstream_binding_hash=str(provenance["upstream_binding_hash"]),
        )
    except (TypeError, ValueError, KeyError) as error:
        raise G21Blocked("s201:HYPOTHESIS_REBUILD_FAILED") from error
    if dict(hypothesis) != expected_hypothesis:
        raise G21Blocked("s201:HYPOTHESIS_CONTENT_MISMATCH")
    if prereg.get("scope") != "formal" or prereg.get("formal_eligible") is not False:
        raise G21Blocked("s201:PREREGISTRATION_SCOPE_INVALID")
    if set(candidate) != {
        "schema_version", "task_id", "gate_ids", "gate_status",
        "local_validation_status", "formal_eligible", "reason", "gate_id",
        "preregistration_hash", "hypothesis_contract_hash",
        "quality_gate_status", "sample_generation_status",
    }:
        raise G21Blocked("s201:CANDIDATE_FIELDS_INVALID")
    if (
        candidate.get("schema_version") != "stage23-task-gate-candidate-v1"
        or candidate.get("task_id") != S201_TASK_ID
        or candidate.get("gate_id") != "stage2.G2.0"
        or candidate.get("gate_ids") != ["stage1.G1-EXIT"]
        or candidate.get("gate_status") != "NOT_RUN"
        or candidate.get("local_validation_status") != "NOT_RUN"
        or candidate.get("formal_eligible") is not False
        or candidate.get("reason") != "formal_gate_requires_independent_review"
        or candidate.get("preregistration_hash") != prereg.get("preregistration_hash")
        or candidate.get("hypothesis_contract_hash") != hypothesis.get("hypothesis_contract_hash")
        or candidate.get("quality_gate_status") != "NOT_RUN"
        or candidate.get("sample_generation_status") != "FORBIDDEN_UNTIL_COMMITTED"
    ):
        raise G21Blocked("s201:CANDIDATE_SELF_SIGNED_OR_HASH_INVALID")
    return loaded, next(iter(configs))


def _expected_group_sources(
    *,
    authority_sources: Sequence[str],
    refs: Mapping[str, str],
    group_output: str,
    include_prior: Sequence[str] = (),
) -> set[str]:
    config_ref = f"{group_output}/producer-config.json"
    return {
        *authority_sources,
        *include_prior,
        config_ref,
        *refs.values(),
    }


def _load_stage1_group(
    root: Path,
    references: object,
    *,
    s110_authority: Mapping[str, object],
    s111_authority: Mapping[str, object],
) -> dict[str, LoadedTaskArtifact]:
    refs = _stage1_refs(references)
    loaded: dict[str, LoadedTaskArtifact] = {}
    for kind in STAGE1_ARTIFACT_KINDS:
        task_id = S110_TASK_ID if kind in S110_ARTIFACT_KINDS else S111_TASK_ID
        path = refs[kind]
        if PurePosixPath(path).name != f"{kind}.json" or PurePosixPath(path).parts[-2] != "commits":
            raise G21Blocked(f"stage1.{kind}:COMMIT_PATH_NOT_CANONICAL")
        loaded[kind] = _load_task(root, path, task_id=task_id, kind=kind)

    s110 = {kind: loaded[kind] for kind in S110_ARTIFACT_KINDS}
    s111 = {kind: loaded[kind] for kind in S111_ARTIFACT_KINDS}
    role_map_110 = {
        "training_state_manifest": "artifact_manifest",
        "resume_equivalence_report": "resume_report",
        "gate_record": "gate_record",
    }
    role_map_111 = {kind: kind for kind in S111_ARTIFACT_KINDS}
    for kind, role in role_map_110.items():
        expected = s110_authority.get("roles")
        if not isinstance(expected, Mapping) or dict(s110[kind].payload) != dict(expected[role]):
            raise G21Blocked(f"stage1.{kind}:AUTHORITY_PAYLOAD_MISMATCH")
    for kind, role in role_map_111.items():
        expected = s111_authority.get("roles")
        if not isinstance(expected, Mapping) or dict(s111[kind].payload) != dict(expected[role]):
            raise G21Blocked(f"stage1.{kind}:AUTHORITY_PAYLOAD_MISMATCH")

    def _group_check(
        group: Mapping[str, LoadedTaskArtifact],
        authority: Mapping[str, object],
        refs_for_group: Mapping[str, str],
        output: str,
        prior: Sequence[str] = (),
    ) -> None:
        source_sets = {frozenset(item.source_refs) for item in group.values()}
        if len(source_sets) != 1:
            raise G21Blocked(f"stage1.{output}:SOURCE_REFS_NOT_SHARED")
        authority_sources = authority.get("source_refs")
        if not isinstance(authority_sources, list) or any(not isinstance(item, str) for item in authority_sources):
            raise G21Blocked(f"stage1.{output}:AUTHORITY_SOURCE_REFS_INVALID")
        expected = _expected_group_sources(
            authority_sources=authority_sources,
            refs=refs_for_group,
            group_output=output,
            include_prior=prior,
        )
        if source_sets != {frozenset(expected)}:
            raise G21Blocked(f"stage1.{output}:SOURCE_CLOSURE_MISMATCH")
        if len({item.identity.config_hash for item in group.values()}) != 1:
            raise G21Blocked(f"stage1.{output}:CONFIG_HASH_NOT_SHARED")

    s110_output = str(PurePosixPath(refs["training_state_manifest"]).parent.parent)
    s111_output = str(PurePosixPath(refs["stage_report"]).parent.parent)
    s110_group_refs = {kind: refs[kind] for kind in S110_ARTIFACT_KINDS}
    s111_group_refs = {kind: refs[kind] for kind in S111_ARTIFACT_KINDS}
    s110_source_refs = tuple(sorted(next(iter({frozenset(item.source_refs) for item in s110.values()}))))
    _group_check(s110, s110_authority, s110_group_refs, s110_output)
    _group_check(s111, s111_authority, s111_group_refs, s111_output, prior=s110_source_refs)
    return loaded


def _load_stage1_authority(repository: Path, data_root: Path) -> tuple[dict[str, object], dict[str, object]]:
    """Call the released S1.10/S1.11 group verifier, never a copied validator."""

    try:
        formalizer = importlib.import_module("ops.stage1.formalize_s1_11")
        s111 = formalizer._emit_load_r4(  # noqa: SLF001 - released authority API
            repository=repository,
            evidence_root=data_root,
            evidence_ref=S111_R4_INDEX_REF,
            approved_data_root=data_root,
        )
        s110 = formalizer._emit_load_s110(  # noqa: SLF001 - released authority API
            repository=repository,
            evidence_root=data_root,
            approved_data_root=data_root,
        )
    except Exception as error:
        raise G21Blocked(f"stage1:CANONICAL_AUTHORITY_LOADER_FAILED:{type(error).__name__}") from error
    if not isinstance(s110, dict) or not isinstance(s111, dict):
        raise G21Blocked("stage1:CANONICAL_AUTHORITY_LOADER_NOT_OBJECT")
    return s110, s111


def _load_handoff(data_root: Path, reference: str) -> dict[str, JSONValue]:
    if reference != HANDOFF_REF:
        raise G21Blocked("handoff:NONCANONICAL_REF")
    path = _resolve(data_root, reference, "handoff.ref")
    if _file_sha256(path, "handoff") != HANDOFF_SHA256:
        raise G21Blocked("handoff:FILE_SHA256_MISMATCH")
    try:
        value = load_g21_formal_handoff(path, data_root=data_root)
    except (G21FormalHandoffError, FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise G21Blocked(f"handoff:CONTRACT_REJECTED:{type(error).__name__}") from error
    if value.get("artifact_hash") != HANDOFF_ARTIFACT_HASH:
        raise G21Blocked("handoff:ARTIFACT_HASH_MISMATCH")
    if value.get("status") != "PASS" or value.get("gate_id") != GATE_ID:
        raise G21Blocked("handoff:OLD_BLOCKED_EVIDENCE")
    auth = value.get("authorization")
    smoke = value.get("current_gpu_smoke")
    if not isinstance(auth, Mapping) or not isinstance(smoke, Mapping):
        raise G21Blocked("handoff:AUTH_OR_SMOKE_BINDING_MISSING")
    if auth.get("ref") != AUTHORIZATION_REF or auth.get("artifact_hash") != AUTH_HASH:
        raise G21Blocked("handoff:AUTHORIZATION_BINDING_INVALID")
    if smoke.get("ref") != SMOKE_REF or smoke.get("sha256") != SMOKE_SHA256:
        raise G21Blocked("handoff:SMOKE_BINDING_INVALID")
    try:
        _resolve(data_root, AUTHORIZATION_REF, "authorization.ref")
        authorization = load_stage2_authorization(data_root, AUTHORIZATION_REF)
    except Exception as error:
        raise G21Blocked(f"authorization:CONTRACT_REJECTED:{type(error).__name__}") from error
    if authorization.get("artifact_hash") != AUTH_HASH:
        raise G21Blocked("authorization:ARTIFACT_HASH_MISMATCH")
    smoke_path = _resolve(data_root, SMOKE_REF, "smoke.ref")
    if _file_sha256(smoke_path, "smoke") != SMOKE_SHA256:
        raise G21Blocked("smoke:FILE_SHA256_MISMATCH")
    devices = tuple(
        (item.get("pci_bus_id"), item.get("uuid"))
        for item in smoke.get("allowed_devices", [])
        if isinstance(item, Mapping)
    )
    if devices != ALLOWED_DEVICES or smoke.get("excluded_pci_bus_ids") != [EXCLUDED_PCI] or smoke.get("excluded_scheduled") is not False:
        raise G21Blocked("smoke:GPU_EXCLUSION_OR_UUID_DRIFT")
    return {
        "ref": reference,
        "sha256": HANDOFF_SHA256,
        "artifact_hash": HANDOFF_ARTIFACT_HASH,
        "authorization_ref": AUTHORIZATION_REF,
        "authorization_sha256": str(authorization.get("file_sha256")),
        "authorization_artifact_hash": AUTH_HASH,
        "smoke_ref": SMOKE_REF,
        "smoke_sha256": SMOKE_SHA256,
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED_DEVICES
        ],
        "excluded_pci_bus_id": EXCLUDED_PCI,
    }


def _load_config(data_root: Path, reference: str) -> tuple[str, ResolvedConfigV2]:
    logical = _logical(reference, "resolved_config_ref")
    value = load_canonical_json(_resolve(data_root, logical, "resolved_config_ref"))
    if not isinstance(value, Mapping):
        raise G21Blocked("resolved_config:OBJECT_REQUIRED")
    if value.get("schema_version") == "task-output-commit-v1":
        item = _load_task(data_root, logical, task_id=TASK_ID, kind="resolved_config")
        value = item.payload
    try:
        config = ResolvedConfigV2.from_mapping(value)
    except (TypeError, ValueError, KeyError) as error:
        raise G21Blocked("resolved_config:V2_INVALID") from error
    if config.task_id != TASK_ID or config.run_intent != "formal" or config.formal_eligible is not True:
        raise G21Blocked("resolved_config:FORMAL_SCOPE_INVALID")
    return logical, config


def _runtime_provenance(data_root: Path) -> dict[str, JSONValue]:
    """Record fixed temp/cache policy without changing process environment."""

    names = (
        "TEMP", "TMP", "TMPDIR", "PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX",
        "HF_HOME", "HF_DATASETS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
        "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE", "TRITON_CACHE_DIR",
        "TORCHINDUCTOR_CACHE_DIR",
    )
    values = {name: os.environ.get(name) for name in names}
    return {
        "environment": values,
        "data_root_tmp_policy": "adapter_writes_no_temp_files",
        "cache_policy": "adapter_does_not_read_or_write_model_or_dataset_cache",
        "data_root": str(data_root),
    }


def _gate(
    *,
    status: GateStatus,
    checked_at: str,
    measured: Mapping[str, JSONValue],
    refs: Sequence[str],
    reasons: Sequence[str] = (),
) -> GateRecord:
    return GateRecord(
        gate_id=GATE_ID,
        stage=2,
        status=status,
        checked_at=checked_at,
        measured=dict(measured),
        threshold={
            "required_status": "PASS",
            "s201_formal_artifact_count": 3,
            "stage1_formal_artifact_count": 7,
            "stage1_authority": "s1.10-r12-plus-s1.11-r4-canonical-group-loader",
            "handoff_ref": HANDOFF_REF,
            "smoke_ref": SMOKE_REF,
            "excluded_pci_bus_id": EXCLUDED_PCI,
            "allowed_devices": [
                {"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED_DEVICES
            ],
        },
        evidence_refs=tuple(refs),
        reasons=tuple(reasons),
    )


def _result(
    gate: GateRecord,
    *,
    published: PublishedTaskArtifact | None,
    source_refs: Sequence[str],
    config_ref: str | None,
    reused: bool = False,
    reason: str | None = None,
) -> dict[str, JSONValue]:
    return {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "gate_record": gate.to_dict(),
        "status": gate.status.value,
        "formal_eligible": gate.status is GateStatus.PASS and published is not None,
        "commit_ref": None if published is None else published.commit_ref,
        "envelope_artifact_hash": None if published is None else published.artifact_hash,
        "source_refs": list(source_refs),
        "resolved_config_ref": config_ref,
        "reused": reused,
        "reason": reason,
    }


def evaluate_formal_g21(
    *,
    repository_root: str | Path,
    data_root: str | Path,
    s201_artifact_refs: Mapping[str, str] | Sequence[str],
    stage1_artifact_refs: Mapping[str, str] | Sequence[str],
    resolved_config_ref: str,
    handoff_ref: str = HANDOFF_REF,
    output_dir: str | Path = "runs/stage2-g2.1",
) -> dict[str, JSONValue]:
    """Validate G2.1 inputs and publish/reuse exactly one GateRecord commit."""

    checked_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    repository: Path | None = None
    data: Path | None = None
    source_refs: list[str] = []
    config_ref: str | None = None
    try:
        repository = _root(repository_root, "repository_root")
        data = _root(data_root, "data_root")
        if repository == data:
            raise G21Blocked("dual_root:ROOTS_MUST_BE_DISTINCT")
        repo_identity = _repository_identity(repository)
        s201, s201_config_hash = _load_s201(data, s201_artifact_refs)
        config_ref, config = _load_config(data, resolved_config_ref)
        s110_authority, s111_authority = _load_stage1_authority(repository, data)
        stage1 = _load_stage1_group(
            data,
            stage1_artifact_refs,
            s110_authority=s110_authority,
            s111_authority=s111_authority,
        )
        handoff = _load_handoff(data, handoff_ref)
        s201_refs = _refs(s201_artifact_refs, S201_ARTIFACT_KINDS, "s201_artifact_refs")
        stage1_refs = _stage1_refs(stage1_artifact_refs)
        source_refs = [
            *[s201_refs[kind] for kind in S201_ARTIFACT_KINDS],
            *[stage1_refs[kind] for kind in STAGE1_ARTIFACT_KINDS],
            config_ref,
            HANDOFF_REF,
            AUTHORIZATION_REF,
            SMOKE_REF,
        ]
        if len(source_refs) != len(set(source_refs)):
            raise G21Blocked("evidence_refs:DUPLICATE")
        stage1_identity = {
            kind: {
                "commit_ref": stage1_refs[kind],
                "artifact_hash": stage1[kind].identity.artifact_hash,
                "config_hash": stage1[kind].identity.config_hash,
                "run_intent": stage1[kind].run_intent,
                "formal_eligible": stage1[kind].identity.formal_eligible,
            }
            for kind in STAGE1_ARTIFACT_KINDS
        }

        def _authority_summary(authority: Mapping[str, object], index_ref: str) -> dict[str, JSONValue]:
            index_path = authority.get("index_path")
            index_sha256 = (
                _file_sha256(index_path, f"stage1.authority.{index_ref}")
                if isinstance(index_path, Path)
                else None
            )
            artifact_hashes = authority.get("artifact_hashes")
            role_hashes = authority.get("role_file_sha256")
            producer_sources = authority.get("producer_source_sha256")
            return {
                "index_ref": index_ref,
                "index_sha256": index_sha256,
                "index_artifact_hash": (
                    authority.get("index", {}).get("artifact_hash")
                    if isinstance(authority.get("index"), Mapping)
                    else None
                ),
                "artifact_hashes": dict(artifact_hashes) if isinstance(artifact_hashes, Mapping) else {},
                "role_file_sha256": dict(role_hashes) if isinstance(role_hashes, Mapping) else {},
                "producer_source_sha256": dict(producer_sources) if isinstance(producer_sources, Mapping) else {},
                "source_refs": list(authority.get("source_refs", [])),
                "data_root_identity": authority.get("data_root_identity"),
            }

        measured: dict[str, JSONValue] = {
            "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
            "task_id": TASK_ID,
            "gate_id": GATE_ID,
            "s201": {
                "config_hash": s201_config_hash,
                "artifacts": {
                    kind: {
                        "commit_ref": s201_refs[kind],
                        "artifact_hash": s201[kind].identity.artifact_hash,
                        "source_refs": list(s201[kind].source_refs),
                    }
                    for kind in S201_ARTIFACT_KINDS
                },
            },
            "stage1": {
                "artifacts": stage1_identity,
                "s110_authority": _authority_summary(s110_authority, S110_R12_INDEX_REF),
                "s111_authority": _authority_summary(s111_authority, S111_R4_INDEX_REF),
            },
            "handoff": handoff,
            "resolved_config": {
                "ref": config_ref,
                "task_id": config.task_id,
                "config_hash": config.config_hash,
                "full_hash": config.full_hash,
                "run_intent": config.run_intent,
                "formal_eligible": config.formal_eligible,
            },
            "repository": repo_identity,
            "runtime": _runtime_provenance(data),
        }
        gate = _gate(
            status=GateStatus.PASS,
            checked_at=checked_at,
            measured=measured,
            refs=source_refs,
        )
        logical_output = _logical(
            PurePosixPath(*Path(output_dir).parts).as_posix()
            if isinstance(output_dir, Path)
            else output_dir,
            "output_dir",
        )
        _resolve(data, logical_output, "output_dir")
        store = TaskArtifactStore(data, logical_output)
        existing = store.discover_complete(
            task_id=TASK_ID,
            config_hash=config.config_hash,
            artifact_kinds=("gate_record",),
            formal_eligible=True,
        )
        if existing is not None:
            loaded = load_committed_task_artifact(data, existing["gate_record"], require_formal=True)
            try:
                previous = GateRecord.from_mapping(dict(loaded.payload))
            except (TypeError, ValueError) as error:
                raise G21Blocked("output:EXISTING_GATE_RECORD_INVALID") from error
            observed = previous.to_dict()
            expected = gate.to_dict()
            for value in (observed, expected):
                value.pop("checked_at", None)
                value.pop("artifact_hash", None)
            if observed != expected or loaded.source_refs != tuple(source_refs):
                raise G21Blocked("output:SEMANTIC_GATE_DRIFT")
            return _result(
                previous,
                published=loaded.identity and PublishedTaskArtifact(
                    task_id=loaded.identity.task_id,
                    artifact_kind=loaded.identity.artifact_kind,
                    artifact_hash=loaded.identity.artifact_hash,
                    config_hash=loaded.identity.config_hash,
                    object_ref=loaded.identity.object_ref,
                    commit_ref=loaded.identity.commit_ref,
                    formal_eligible=loaded.identity.formal_eligible,
                ),
                source_refs=source_refs,
                config_ref=config_ref,
                reused=True,
            )
        published = store.publish(
            task_id=TASK_ID,
            artifact_kind="gate_record",
            config_hash=config.config_hash,
            run_intent="formal",
            payload=gate.to_dict(),
            formal_eligible=True,
            source_refs=tuple(source_refs),
        )
        return _result(
            gate,
            published=published,
            source_refs=source_refs,
            config_ref=config_ref,
        )
    except (G21Blocked, FileNotFoundError, OSError, TypeError, ValueError, KeyError) as error:
        reason = str(error) or type(error).__name__
        blocked = _gate(
            status=GateStatus.BLOCKED,
            checked_at=checked_at,
            measured={
                "adapter_schema_version": ADAPTER_SCHEMA_VERSION,
                "task_id": TASK_ID,
                "gate_id": GATE_ID,
                "repository_root": None if repository is None else str(repository),
                "data_root": None if data is None else str(data),
            },
            refs=(),
            reasons=(reason,),
        )
        return _result(
            blocked,
            published=None,
            source_refs=source_refs,
            config_ref=config_ref,
            reason=reason,
        )


evaluate_g21 = evaluate_formal_g21


__all__ = [
    "ADAPTER_SCHEMA_VERSION",
    "ARTIFACT_KINDS",
    "G21Blocked",
    "GATE_ID",
    "HANDOFF_ARTIFACT_HASH",
    "HANDOFF_REF",
    "HANDOFF_SHA256",
    "S110_ARTIFACT_KINDS",
    "S111_ARTIFACT_KINDS",
    "S201_ARTIFACT_KINDS",
    "SMOKE_REF",
    "SMOKE_SHA256",
    "STAGE1_ARTIFACT_KINDS",
    "TASK_ID",
    "evaluate_formal_g21",
    "evaluate_g21",
]
