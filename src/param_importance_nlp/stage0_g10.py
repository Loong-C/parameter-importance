"""Formal Stage 0 S0.12 delivery, synchronization, and readiness gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
from typing import Any, Final, Mapping, Sequence

from .atomic import atomic_write_bytes, sha256_file
from .contracts import (
    GateRecord,
    GateStatus,
    ResolvedConfigV2,
    RuntimeCapabilityEvidence,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .evidence_reuse import (
    EvidenceReuseError,
    validate_evidence_reuse_attestation,
)
from .experiments import build_default_task_runtime
from .runtime import (
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g9 import Stage0G9FormalState, load_stage0_g9_formal_state
from .stage0_g10_sync import (
    AGENT_FILES,
    REMOTE_URL,
    SERVER_DATA_ROOT,
    SERVER_HOST,
    SERVER_REPOSITORY,
    SYNC_OBSERVATION_SCHEMA,
)
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID: Final = "stage0.12_delivery_and_sync"
GATE_ID: Final = "stage0.G10"
READINESS_SCHEMA: Final = "stage0-g10-readiness-v1"
DELIVERY_SCHEMA: Final = "stage0-g10-delivery-manifest-v1"
HANDOFF_SCHEMA: Final = "stage0-g10-stage1-handoff-v1"
WORKLOG_SCHEMA: Final = "stage0-g10-worklog-v1"
SYNC_REPORT_SCHEMA: Final = "stage0-g10-sync-report-v1"
_OUTPUT_KINDS = {"delivery_manifest", "worklog", "sync_report"}
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024
_REQUIRED_GATES = frozenset(
    {
        "stage0.G0-C", "stage0.G0-G", "stage0.G1", "stage0.G2",
        "stage0.G3", "stage0.G3-S1", "stage0.G3-S2", "stage0.G3-S4",
        "stage0.G3-S5", "stage0.G3-S6", "stage0.G4", "stage0.G5",
        "stage0.G6", "stage0.G7-LOGGING", "stage0.G7", "stage0.G8-C",
        "stage0.G8-S4", "stage0.G8-S5", "stage0.G8", "stage0.G9",
    }
)
_CRITICAL_SOURCE_REFS = (
    "docs/mathematics.md",
    "docs/stage0-delivery-runbook.md",
    "docs/stage0-replay-runbook.md",
    "docs/stage1-handoff.md",
    "ops/stage0/collect_g10_sync_observation.py",
    "ops/stage0/attest_g9_evidence_reuse.py",
    "ops/stage0/formalize_g10.py",
    "plan/stage0/12_delivery_and_sync.md",
    "policies/evidence-validity-and-rerun.md",
    "reports/stage0/g1-persistence-decision-20260719.json",
    "schemas/stage0-g10-delivery-manifest-v1.json",
    "schemas/stage0-g10-evidence-v1.json",
    "schemas/stage0-g10-formalization-index-v1.json",
    "schemas/stage0-g10-readiness-v1.json",
    "schemas/stage0-g10-stage1-handoff-v1.json",
    "schemas/stage0-g10-sync-observation-v1.json",
    "schemas/stage0-g10-sync-report-v1.json",
    "schemas/stage0-g10-worklog-v1.json",
    "schemas/shared/evidence-reuse-attestation-v1.json",
    "src/param_importance_nlp/evidence_reuse.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/stage0_g10.py",
    "src/param_importance_nlp/stage0_g10_sync.py",
    "tests/test_stage0_g10.py",
    "worklogs/2026-08-03-stage0-remaining-tasks.md",
)
_FORBIDDEN_TOP_LEVEL = {
    ".cache", ".venv", "checkpoints", "data", "runs", "venv", "wheelhouse"
}
_FORBIDDEN_SUFFIXES = {
    ".bin", ".ckpt", ".pt", ".pth", ".safetensors", ".whl"
}
_SECRET_PATTERNS = (
    re.compile(
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"
        r"\s+[A-Za-z0-9+/=\r\n]{64,}\s+"
        r"-----END (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----",
        re.MULTILINE,
    ),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"https?://[^\s\"']*(?:X-Amz-Signature|[?&]Signature=)[^\s\"']*", re.I),
)


class Stage0G10Error(RuntimeError):
    """The final Stage 0 delivery or consistency gate failed closed."""


@dataclass(frozen=True, slots=True)
class G10SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str
    critical_source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Stage0G10FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str
    readiness_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G10FormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    readiness_ref: str
    readiness_artifact_hash: str
    g9_index_ref: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise Stage0G10Error(f"G10_TIMESTAMP_INVALID:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0G10Error(f"G10_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None:
        raise Stage0G10Error(f"G10_TIMESTAMP_TZ_MISSING:{field}")
    return parsed.astimezone(timezone.utc)


def _beijing_time(value: str) -> str:
    return _parse_time(value, field="beijing_projection").astimezone(
        timezone(timedelta(hours=8))
    ).isoformat()


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G10Error(f"G10_OBJECT_INVALID:{field}")
    return dict(value)


def _array(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise Stage0G10Error(f"G10_ARRAY_INVALID:{field}")
    return value


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G10Error(f"G10_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G10Error(f"G10_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G10Error(f"G10_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G10Error(f"G10_IMMUTABLE_JSON_DRIFT:{path.name}")
        return
    write_canonical_json(path, dict(value))


def _write_bytes_or_verify(path: Path, value: bytes) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise Stage0G10Error(f"G10_IMMUTABLE_TEXT_DRIFT:{path.name}")
        return
    atomic_write_bytes(path, value)


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _capture_source() -> G10SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    probes = {
        "top": _git(repository, "rev-parse", "--show-toplevel"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "tracked": _git(repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS),
        "status": _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if any(item.returncode != 0 for item in probes.values()):
        raise Stage0G10Error("G10_SOURCE_GIT_PROBE_FAILED")
    commit = probes["head"].stdout.strip()
    branch = probes["branch"].stdout.strip()
    if (
        Path(probes["top"].stdout.strip()).resolve() != repository
        or _GIT_COMMIT_RE.fullmatch(commit) is None
        or not branch
        or probes["status"].stdout.strip()
    ):
        raise Stage0G10Error("G10_FORMAL_SOURCE_NOT_CLEAN_OR_IDENTIFIED")
    hashes = {reference: sha256_file(repository / reference) for reference in _CRITICAL_SOURCE_REFS}
    return G10SourceBinding(repository, commit, branch, hashes)


def _load_hashed(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(path), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage0G10Error(f"G10_HASHED_OBJECT_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def validate_sync_observation(
    root: Path,
    reference: str,
    source: G10SourceBinding,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    path = _logical_path(root, reference, field="sync_observation_ref")
    observation = _load_hashed(
        path,
        schema=SYNC_OBSERVATION_SCHEMA,
        field="sync_observation",
    )
    observed_at = _parse_time(observation.get("observed_at"), field="observed_at")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    age = (current - observed_at).total_seconds()
    local = _mapping(observation.get("local"), field="observation.local")
    github = _mapping(observation.get("github"), field="observation.github")
    server = _mapping(observation.get("server"), field="observation.server")
    agent = _mapping(observation.get("agent_sync"), field="observation.agent_sync")
    cleanup = _mapping(observation.get("bundle_cleanup"), field="observation.bundle_cleanup")
    preserved = _mapping(
        observation.get("preserved_user_content"), field="observation.preserved_user_content"
    )
    expected_agent = list(AGENT_FILES)
    local_hashes = _mapping(agent.get("local_sha256"), field="agent.local_sha256")
    expected_bundle_name = f"stage0-g10-sync-{source.git_commit[:12]}.bundle"
    if (
        age < -300
        or age > 7200
        or observation.get("expected_commit") != source.git_commit
        or observation.get("branch") != source.git_branch
        or observation.get("fast_forward_ancestry_verified") is not True
        or observation.get("force_push_used") is not False
        or not isinstance(observation.get("authorization_ref"), str)
        or not str(observation["authorization_ref"]).strip()
        or local.get("head") != source.git_commit
        or local.get("branch") != source.git_branch
        or local.get("worktree_clean") is not True
        or github.get("remote") != "origin"
        or github.get("remote_url") != REMOTE_URL
        or github.get("branch_ref") != f"refs/heads/{source.git_branch}"
        or github.get("head") != source.git_commit
        or github.get("push_verified") is not True
        or server.get("host_alias") != SERVER_HOST
        or server.get("repository") != SERVER_REPOSITORY
        or server.get("data_root") != SERVER_DATA_ROOT
        or server.get("head") != source.git_commit
        or server.get("branch") != source.git_branch
        or server.get("worktree_clean") is not True
        or server.get("fast_forward_verified") is not True
        or agent.get("file_count_each_side") != 5
        or agent.get("files") != expected_agent
        or agent.get("all_equal") is not True
        or local_hashes != agent.get("server_sha256")
        or set(local_hashes) != set(AGENT_FILES)
        or any(_SHA256_RE.fullmatch(str(value)) is None for value in local_hashes.values())
        or cleanup.get("bundle_name") != expected_bundle_name
        or not str(cleanup.get("local_path", "")).endswith(f"/.{expected_bundle_name}")
        or cleanup.get("server_path") != f"{SERVER_DATA_ROOT}/tmp/{expected_bundle_name}"
        or cleanup.get("local_absent") is not True
        or cleanup.get("server_absent") is not True
        or preserved.get("path") != "docs/mathematics.md"
        or preserved.get("tracked") is not True
        or preserved.get("sha256") != sha256_file(source.repository / "docs/mathematics.md")
    ):
        raise Stage0G10Error("G10_SYNC_OBSERVATION_NOT_CURRENT_OR_CONSISTENT")
    for field in ("previous_github_head", "previous_server_head"):
        previous = observation.get(field)
        if not isinstance(previous, str) or _GIT_COMMIT_RE.fullmatch(previous) is None:
            raise Stage0G10Error(f"G10_SYNC_PREVIOUS_HEAD_INVALID:{field}")
        ancestry = _git(source.repository, "merge-base", "--is-ancestor", previous, source.git_commit)
        if ancestry.returncode != 0:
            raise Stage0G10Error(f"G10_SYNC_FAST_FORWARD_ANCESTRY_INVALID:{field}")
    return observation


def _gate_key(gate_id: str) -> str:
    return "gate_" + re.sub(r"[^a-z0-9]+", "_", gate_id.casefold()).strip("_")


def _find_gate(value: object, gate_id: str) -> GateRecord | None:
    if isinstance(value, Mapping):
        if value.get("gate_id") == gate_id:
            try:
                return GateRecord.from_mapping(value)
            except Exception as error:
                raise Stage0G10Error(f"G10_GATE_RECORD_INVALID:{gate_id}") from error
        for nested in value.values():
            found = _find_gate(nested, gate_id)
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _find_gate(nested, gate_id)
            if found is not None:
                return found
    return None


def _validated_gate_records(
    environment: TaskRuntimeEnvironment,
    root: Path,
) -> tuple[dict[str, GateRecord], dict[str, str]]:
    missing = sorted(_REQUIRED_GATES - environment.passed_gate_ids)
    if missing:
        raise Stage0G10Error(f"G10_REQUIRED_GATES_MISSING:{missing}")
    records: dict[str, GateRecord] = {}
    refs: dict[str, str] = {}
    for gate_id in sorted(_REQUIRED_GATES):
        key = _gate_key(gate_id)
        reference = environment.evidence_refs.get(key)
        if reference is None:
            raise Stage0G10Error(f"G10_GATE_EVIDENCE_REF_MISSING:{gate_id}")
        try:
            payload = load_committed_task_artifact(root, reference, require_formal=True).payload
        except Exception as error:
            raise Stage0G10Error(f"G10_GATE_EVIDENCE_COMMIT_INVALID:{gate_id}") from error
        gate = _find_gate(payload, gate_id)
        if gate is None or gate.status is not GateStatus.PASS:
            raise Stage0G10Error(f"G10_GATE_NOT_PASS:{gate_id}")
        records[gate_id] = gate
        refs[gate_id] = reference
    return records, refs


def _category(reference: str) -> str:
    if reference in {"pyproject.toml", "requirements.lock"} or reference.startswith("environment/"):
        return "dependency_and_environment"
    prefix = reference.split("/", 1)[0]
    return {
        "configs": "configuration",
        "docs": "documentation",
        "legacy": "reviewed_legacy_archive",
        "ops": "operations",
        "plan": "plan",
        "reports": "checked_in_report",
        "schemas": "schema",
        "src": "implementation",
        "tests": "test",
        "worklogs": "worklog",
    }.get(prefix, "repository_support")


def _validate_stage_links(repository: Path) -> list[str]:
    listed = _git(repository, "ls-files", "-z")
    if listed.returncode != 0:
        raise Stage0G10Error("G10_REPOSITORY_TRACKED_FILE_LIST_FAILED")
    tracked = {item for item in listed.stdout.split("\0") if item}
    checked: list[str] = []
    for reference in (
        "docs/stage0-delivery-runbook.md",
        "docs/stage0-replay-runbook.md",
        "docs/stage1-handoff.md",
    ):
        path = repository / reference
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
            relative = target.split("#", 1)[0]
            if not relative or "://" in relative:
                continue
            if relative.startswith("/") or "\\" in relative:
                raise Stage0G10Error(f"G10_REPOSITORY_LINK_INVALID:{reference}:{target}")
            target_reference = posixpath.normpath(
                f"{PurePosixPath(reference).parent.as_posix()}/{relative}"
            )
            if target_reference not in tracked:
                raise Stage0G10Error(
                    f"G10_REPOSITORY_LINK_NOT_TRACKED_EXACT:{reference}:{target}"
                )
            resolved = (path.parent / relative).resolve()
            try:
                resolved.relative_to(repository)
            except ValueError as error:
                raise Stage0G10Error(f"G10_REPOSITORY_LINK_ESCAPE:{reference}:{target}") from error
            if not resolved.exists():
                raise Stage0G10Error(f"G10_REPOSITORY_LINK_MISSING:{reference}:{target}")
            checked.append(f"{reference}->{target}")
    return checked


def _repository_inventory(
    source: G10SourceBinding,
    observation: Mapping[str, Any],
) -> dict[str, JSONValue]:
    listed = _git(source.repository, "ls-files", "-z")
    if listed.returncode != 0:
        raise Stage0G10Error("G10_REPOSITORY_TRACKED_FILE_LIST_FAILED")
    references = sorted(item for item in listed.stdout.split("\0") if item)
    if not references or any(reference.startswith("Agent/") for reference in references):
        raise Stage0G10Error("G10_REPOSITORY_TRACKED_SET_INVALID")
    rows: list[dict[str, JSONValue]] = []
    secret_hits: list[str] = []
    forbidden: list[str] = []
    oversized: list[str] = []
    for reference in references:
        path = source.repository.joinpath(*PurePosixPath(reference).parts)
        if not path.is_file() or path.is_symlink():
            raise Stage0G10Error(f"G10_REPOSITORY_FILE_INVALID:{reference}")
        size = path.stat().st_size
        top = reference.split("/", 1)[0].casefold()
        if (
            top in _FORBIDDEN_TOP_LEVEL
            or "__pycache__" in PurePosixPath(reference).parts
            or path.suffix.casefold() in _FORBIDDEN_SUFFIXES
        ):
            forbidden.append(reference)
        if size > _MAX_TRACKED_FILE_BYTES:
            oversized.append(reference)
        if size <= _MAX_TRACKED_FILE_BYTES:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = ""
            if any(pattern.search(text) for pattern in _SECRET_PATTERNS):
                secret_hits.append(reference)
        rows.append(
            {
                "git_path": reference,
                "server_path": f"{SERVER_REPOSITORY}/{reference}",
                "size_bytes": size,
                "sha256": sha256_file(path),
                "category": _category(reference),
                "acceptance_status": "PASS",
            }
        )
    previous = str(observation["previous_github_head"])
    whitespace = _git(source.repository, "diff", "--check", f"{previous}..{source.git_commit}")
    if whitespace.returncode != 0 or whitespace.stdout.strip() or whitespace.stderr.strip():
        raise Stage0G10Error("G10_REPOSITORY_WHITESPACE_CHECK_FAILED")
    required = {
        "docs/mathematics.md", "environment/requirements.lock", "pyproject.toml",
        "docs/stage0-delivery-runbook.md", "docs/stage0-replay-runbook.md",
        "docs/stage1-handoff.md",
        "worklogs/2026-08-03-stage0-remaining-tasks.md",
    }
    missing_required = sorted(required - set(references))
    if missing_required or forbidden or oversized or secret_hits:
        raise Stage0G10Error(
            f"G10_REPOSITORY_AUDIT_FAILED:missing={missing_required}:forbidden={forbidden}:oversized={oversized}:secrets={secret_hits}"
        )
    links = _validate_stage_links(source.repository)
    category_counts: dict[str, int] = {}
    for row in rows:
        category = str(row["category"])
        category_counts[category] = category_counts.get(category, 0) + 1
    return {
        "status": "PASS",
        "git_commit": source.git_commit,
        "tracked_file_count": len(rows),
        "tracked_bytes": sum(int(row["size_bytes"]) for row in rows),
        "max_allowed_file_bytes": _MAX_TRACKED_FILE_BYTES,
        "category_counts": category_counts,
        "forbidden_paths": forbidden,
        "oversized_paths": oversized,
        "high_confidence_secret_hits": secret_hits,
        "checked_markdown_links": links,
        "whitespace_check_range": f"{previous}..{source.git_commit}",
        "agent_directory_tracked": False,
        "rows": rows,
    }


def _asset_inventory(
    environment: TaskRuntimeEnvironment,
    root: Path,
    source_commit: str,
) -> dict[str, JSONValue]:
    reference = environment.evidence_refs.get("g3_resolution")
    if reference is None:
        raise Stage0G10Error("G10_G3_RESOLUTION_REF_MISSING")
    payload = _mapping(
        load_committed_task_artifact(root, reference, require_formal=True).payload,
        field="g3_resolution",
    )
    entries = _array(payload.get("entries"), field="g3_resolution.entries")
    if len(entries) != 13:
        raise Stage0G10Error("G10_ASSET_ENTRY_COUNT_INVALID")
    rows: list[dict[str, JSONValue]] = []
    for index, raw in enumerate(entries):
        entry = _mapping(raw, field=f"g3_resolution.entries[{index}]")
        manifest_ref = str(entry.get("manifest_ref"))
        asset_root_ref = str(entry.get("asset_root_ref"))
        manifest_path = _logical_path(root, manifest_ref, field="asset.manifest_ref")
        asset_root = _logical_path(root, asset_root_ref, field="asset.asset_root_ref")
        manifest = _mapping(load_canonical_json(manifest_path), field="asset.manifest")
        files = _array(manifest.get("files"), field="asset.manifest.files")
        if (
            entry.get("status") != "PASS"
            or manifest.get("state") != "ready"
            or manifest.get("asset_id") != entry.get("asset_id")
            or canonical_json_hash(manifest) != entry.get("ready_manifest_sha256")
            or not asset_root.is_dir()
        ):
            raise Stage0G10Error(f"G10_ASSET_MANIFEST_IDENTITY_INVALID:{index}")
        declared_bytes = 0
        observed_bytes = 0
        for position, raw_file in enumerate(files):
            descriptor = _mapping(raw_file, field=f"asset.files[{position}]")
            relative = descriptor.get("path")
            if not isinstance(relative, str):
                raise Stage0G10Error("G10_ASSET_FILE_PATH_INVALID")
            file_path = asset_root.joinpath(*PurePosixPath(relative).parts).resolve()
            try:
                file_path.relative_to(asset_root.resolve())
            except ValueError as error:
                raise Stage0G10Error("G10_ASSET_FILE_ESCAPE") from error
            declared_size = descriptor.get("size_bytes")
            if (
                isinstance(declared_size, bool)
                or not isinstance(declared_size, int)
                or not file_path.is_file()
                or file_path.is_symlink()
                or file_path.stat().st_size != declared_size
                or _SHA256_RE.fullmatch(str(descriptor.get("sha256"))) is None
            ):
                raise Stage0G10Error(f"G10_ASSET_FILE_SIZE_OR_DIGEST_INVALID:{index}:{position}")
            declared_bytes += declared_size
            observed_bytes += file_path.stat().st_size
        rows.append(
            {
                "logical_name": str(entry["logical_name"]),
                "kind": str(entry["kind"]),
                "asset_id": str(entry["asset_id"]),
                "revision": str(manifest["revision"]),
                "manifest_ref": manifest_ref,
                "manifest_path": str(manifest_path),
                "ready_manifest_sha256": str(entry["ready_manifest_sha256"]),
                "asset_root_ref": asset_root_ref,
                "asset_root_path": str(asset_root),
                "file_count": len(files),
                "declared_size_bytes": declared_bytes,
                "observed_size_bytes": observed_bytes,
                "associated_git_commit": source_commit,
                "verification_basis": "G3_FULL_SHA256_AND_G10_EXISTENCE_SIZE_RECHECK",
                "acceptance_status": "PASS",
            }
        )
    return {
        "status": "PASS",
        "authoritative_runtime_root": str(root),
        "authoritative_runtime_copy_count": 1,
        "authoritative_runtime_copy_is_backup": False,
        "asset_count": len(rows),
        "declared_size_bytes": sum(int(row["declared_size_bytes"]) for row in rows),
        "observed_size_bytes": sum(int(row["observed_size_bytes"]) for row in rows),
        "rows": rows,
    }


def _evidence_inventory(environment: TaskRuntimeEnvironment, root: Path) -> dict[str, JSONValue]:
    rows: list[dict[str, JSONValue]] = []
    for key, reference in sorted(environment.evidence_refs.items()):
        path = _logical_path(root, reference, field=f"environment.evidence_refs.{key}")
        if not path.is_file() or path.is_symlink():
            raise Stage0G10Error(f"G10_EVIDENCE_FILE_INVALID:{key}")
        rows.append(
            {
                "key": key,
                "ref": reference,
                "absolute_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "acceptance_status": "PASS",
            }
        )
    return {
        "status": "PASS",
        "evidence_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "rows": rows,
    }


def _persistence_status(
    source: G10SourceBinding,
    gate_records: Mapping[str, GateRecord],
    *,
    now: datetime | None = None,
) -> dict[str, JSONValue]:
    decision_path = source.repository / "reports/stage0/g1-persistence-decision-20260719.json"
    # G1-D predates the canonical one-line publisher and is deliberately kept
    # byte-for-byte as historical evidence (see stage0_bootstrap._load_report);
    # parse it with the strict reader and bind its original SHA-256 instead of
    # rewriting history in place.
    decision = _mapping(
        loads_strict_json(decision_path.read_bytes()), field="persistence_decision"
    )
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    expires = _parse_time(decision.get("expires_at"), field="persistence.expires_at")
    g1 = gate_records["stage0.G1"]
    if (
        decision.get("gate") != "G1-D"
        or decision.get("status") != "PASS"
        or decision.get("satisfaction") != "TIME_BOUNDED_RISK_ACCEPTANCE"
        or current >= expires
        or decision.get("independent_failure_domain_copy_count") != 0
        or decision.get("authoritative_large_asset_runtime_copy") != SERVER_DATA_ROOT
        or g1.measured.get("persistence_satisfaction") != decision.get("satisfaction")
        or g1.measured.get("persistence_expires_at") != decision.get("expires_at")
    ):
        raise Stage0G10Error("G10_G1_D_PERSISTENCE_STATUS_INVALID_OR_EXPIRED")
    return {
        "status": "PASS",
        "satisfaction": str(decision["satisfaction"]),
        "scope": list(decision["scope"]),
        "excluded": list(decision["excluded"]),
        "expires_at": str(decision["expires_at"]),
        "approval_source": str(decision["approval_source"]),
        "independent_failure_domain_copy_count": 0,
        "authoritative_runtime_copy_is_backup": False,
        "stage4_requirement": str(decision["stage4_requirement"]),
        "decision_ref": decision_path.relative_to(source.repository).as_posix(),
        "decision_sha256": sha256_file(decision_path),
    }


def build_stage0_g10_config(
    *,
    binding: Stage0SourceBinding,
    state: Stage0G9FormalState,
    sync_observation_ref: str,
    reuse_attestation_ref: str | None = None,
) -> ResolvedConfigV2:
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=(
            *tuple(state.task_output_refs.values()),
            *((reuse_attestation_ref,) if reuse_attestation_ref is not None else ()),
        ),
        output_dir=f"evidence/stage0/tasks/12-{state.gate_artifact_hash}",
        base_overrides={"identity": {"route": f"stage0-g10-{state.gate_artifact_hash[:12]}"}},
        v2_overrides={
            "execution": {"timeout_seconds": 1800, "max_attempts": 1},
            "orchestration": {
                "route_spec_ref": state.index_ref,
                # This field is the v2 contract's explicit decision reference.
                # For G10 it binds the reviewed cross-commit reuse decision.
                "quadrature_decision_ref": reuse_attestation_ref,
                "matrix_ref": sync_observation_ref,
            },
            "recovery": {
                "mode": "manual_external",
                "resume_ref": None,
                "max_restarts": 0,
                "safe_boundary": "immutable_publish",
            },
        },
    )


def _observation_ref(request: TaskExecutionRequest) -> str:
    orchestration = _mapping(request.config.section("orchestration"), field="orchestration")
    reference = orchestration.get("matrix_ref")
    if not isinstance(reference, str):
        raise Stage0G10Error("G10_SYNC_OBSERVATION_REF_MISSING")
    return reference


def _validated_reuse_attestation(
    request: TaskExecutionRequest,
    root: Path,
    source: G10SourceBinding,
) -> tuple[str | None, dict[str, Any] | None]:
    orchestration = _mapping(request.config.section("orchestration"), field="orchestration")
    g9_index_ref = orchestration.get("route_spec_ref")
    reference = orchestration.get("quadrature_decision_ref")
    input_references = orchestration.get("input_result_refs")
    if not isinstance(g9_index_ref, str) or not isinstance(input_references, list):
        raise Stage0G10Error("G10_REUSE_BINDING_MISSING")
    index = _load_hashed(
        _logical_path(root, g9_index_ref, field="g9_index_ref"),
        schema="stage0-g9-formalization-index-v1",
        field="g9_index",
    )
    producer_commit = index.get("generator_git_commit")
    if not isinstance(producer_commit, str) or _GIT_COMMIT_RE.fullmatch(producer_commit) is None:
        raise Stage0G10Error("G10_REUSE_PRODUCER_COMMIT_INVALID")
    if producer_commit == source.git_commit:
        if reference is not None:
            raise Stage0G10Error("G10_REUSE_ATTESTATION_UNNECESSARY")
        return None, None
    if not isinstance(reference, str) or reference not in input_references:
        raise Stage0G10Error("G10_REUSE_ATTESTATION_REQUIRED")
    try:
        attestation = validate_evidence_reuse_attestation(
            repository=source.repository,
            data_root=root,
            attestation_ref=reference,
            producer_commit=producer_commit,
            consumer_commit=source.git_commit,
            consumer_branch=source.git_branch,
            scope_id="stage0.G0-G9",
            source_evidence_ref=g9_index_ref,
            required_gate_ids=sorted(_REQUIRED_GATES),
        )
    except EvidenceReuseError as error:
        raise Stage0G10Error(f"G10_REUSE_ATTESTATION_INVALID:{error}") from error
    return reference, attestation


def _handoff(
    request: TaskExecutionRequest,
    source: G10SourceBinding,
    assets: Mapping[str, Any],
    persistence: Mapping[str, Any],
    checked_at: str,
) -> dict[str, JSONValue]:
    fixture = _mapping(
        load_canonical_json(source.repository / "fixtures/stage0/deterministic-training-v1.json"),
        field="deterministic_fixture",
    )
    rows = _array(assets.get("rows"), field="assets.rows")
    reusable_assets = [
        {
            "logical_name": str(row["logical_name"]),
            "kind": str(row["kind"]),
            "asset_id": str(row["asset_id"]),
            "revision": str(row["revision"]),
            "manifest_ref": str(row["manifest_ref"]),
        }
        for row in rows
    ]
    return _with_hash(
        {
            "schema_version": HANDOFF_SCHEMA,
            "status": "READY_FOR_STAGE1_ENTRY",
            "generated_at": checked_at,
            "generator_git_commit": source.git_commit,
            "environment_id": request.environment.environment_hash,
            "configuration_refs": [
                "configs/run-ready/layers/formal-stage1-pythia14m.yaml",
                "configs/run-ready/v2/stage1-pythia14m-formal.yaml",
                "configs/stage0/g9-test-matrix-v1.json",
            ],
            "reusable_assets": reusable_assets,
            "deterministic_fixture": {
                "ref": "fixtures/stage0/deterministic-training-v1.json",
                "fixture_id": str(fixture["fixture_id"]),
                "artifact_hash": str(fixture["artifact_hash"]),
                "tolerances": dict(_mapping(fixture["tolerances"], field="fixture.tolerances")),
            },
            "verified_training_semantics": [
                "single_gpu_bfloat16_forward_backward_optimizer_checkpoint",
                "four_gpu_nccl_global_loss_numerator_and_effective_count_reduction",
                "gradient_accumulation_and_no_sync_last_microbatch_only",
                "canonical_jsonl_lineage_and_tensorboard_projection",
                "group_checkpoint_resume_and_retention",
                "capacity_preflight_and_project_gpu_lease",
                "fresh_process_single_and_four_gpu_offline_replay",
            ],
            "entry_requirements": {
                "required_gate": GATE_ID,
                "required_status": "PASS",
                "allowed_model_scope": "Pythia 14M fixture before scale-up",
                "importance_correctness_must_be_proven_in_stage1": True,
            },
            "non_blocking_risks": [
                "Stage 0 large assets have one authoritative runtime copy and no independent failure-domain backup.",
                f"The Stage 0 risk acceptance expires at {persistence['expires_at']} or earlier under its termination conditions.",
                "Stage 0 validates infrastructure semantics only; it does not prove parameter-importance mathematics.",
            ],
            "prohibited_inferences": [
                "Do not treat the deterministic fixture as an oracle for any importance estimator.",
                "Do not reuse Stage 0 readiness after source, environment, asset, topology, or gate evidence drift.",
            ],
        }
    )


def _handoff_markdown(value: Mapping[str, Any]) -> bytes:
    lines = [
        "# Stage 1 正式交接",
        "",
        f"- Git commit: `{value['generator_git_commit']}`",
        f"- Environment ID: `{value['environment_id']}`",
        f"- 状态: **{value['status']}**",
        "",
        "## 可复用资产",
        "",
        "| 名称 | 类型 | Asset ID | Revision |",
        "|---|---|---|---|",
    ]
    for row in value["reusable_assets"]:
        lines.append(
            f"| {row['logical_name']} | {row['kind']} | `{row['asset_id']}` | `{row['revision']}` |"
        )
    lines.extend(
        [
            "",
            "## 已验证基础设施语义",
            "",
            *[f"- `{item}`" for item in value["verified_training_semantics"]],
            "",
            "## 边界",
            "",
            "Stage 0 只证明基础设施、训练与恢复语义；Stage 1 仍须独立证明参数重要性数学实现正确。",
            "",
            "静态入口说明见仓库文件 `docs/stage1-handoff.md`。",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _worklog(
    source: G10SourceBinding,
    checked_at: str,
    gate_records: Mapping[str, GateRecord],
    observation_ref: str,
    delivery_ref: str,
    sync_ref: str,
    handoff_ref: str,
    readiness_ref: str,
    persistence: Mapping[str, Any],
) -> dict[str, JSONValue]:
    phases = (
        ("G0-G4", "基线、存储、环境、资产与 provenance 冻结"),
        ("G5", "真实单卡训练 smoke"),
        ("G6", "四卡 DDP、累积与 no_sync 语义"),
        ("G7", "日志、checkpoint、恢复与保留"),
        ("G8", "容量、运维预检与故障演练"),
        ("G9", "分层测试与独立离线重放"),
        ("G10", "交付清单、三端同步、Agent 哈希与 readiness"),
    )
    entries: list[dict[str, JSONValue]] = []
    for phase, objective in phases:
        covered = [
            gate_id
            for gate_id in sorted(gate_records)
            if gate_id.startswith(f"stage0.{phase}")
            or phase == "G0-G4" and gate_id.split(".", 1)[1].startswith(("G0", "G1", "G2", "G3", "G4"))
        ]
        entries.append(
            {
                "phase": phase,
                "objective": objective,
                "actual_changes": "按版本化计划实现并固化机器可读合同、运行证据与故障闭环。",
                "validation": {gate_id: gate_records[gate_id].status.value for gate_id in covered},
                "exit_status": "PASS",
                "evidence_refs": [
                    observation_ref if phase == "G10" else gate_records[gate_id].evidence_refs[0]
                    for gate_id in covered
                ] if covered else [observation_ref],
                "risk": (
                    f"Stage 0 可再生产物单盘风险接受有效至 {persistence['expires_at']}。"
                    if phase == "G10"
                    else "历史失败与中止保留在原始 append-only 证据中，未重写为通过。"
                ),
            }
        )
    return _with_hash(
        {
            "schema_version": WORKLOG_SCHEMA,
            "language": "zh-CN",
            "timezone": "Asia/Shanghai",
            "generated_at": _beijing_time(checked_at),
            "git_branch": source.git_branch,
            "git_commit": source.git_commit,
            "status": "PASS",
            "entries": entries,
            "final_evidence": {
                "delivery_manifest_ref": delivery_ref,
                "sync_report_ref": sync_ref,
                "stage1_handoff_ref": handoff_ref,
                "readiness_ref": readiness_ref,
            },
            "secrets_included": False,
            "temporary_download_urls_included": False,
        }
    )


def _worklog_markdown(value: Mapping[str, Any]) -> bytes:
    lines = [
        "# Stage 0 最终工作日志（机器生成投影）",
        "",
        f"- 分支：`{value['git_branch']}`",
        f"- 提交：`{value['git_commit']}`",
        f"- 状态：**{value['status']}**",
        "",
    ]
    for entry in value["entries"]:
        lines.extend(
            [
                f"## {entry['phase']} — {entry['objective']}",
                "",
                f"- 实际修改：{entry['actual_changes']}",
                f"- 退出状态：`{entry['exit_status']}`",
                f"- 风险：{entry['risk']}",
                "",
            ]
        )
    return "\n".join(lines).encode("utf-8")


def _gate_payloads(
    *,
    request: TaskExecutionRequest,
    root: Path,
    source: G10SourceBinding,
    checked_at: str,
    observation: Mapping[str, Any],
    observation_ref: str,
    gate_records: Mapping[str, GateRecord],
    gate_refs: Mapping[str, str],
    repository_inventory: Mapping[str, JSONValue],
    assets: Mapping[str, JSONValue],
    evidence_inventory: Mapping[str, JSONValue],
    persistence: Mapping[str, JSONValue],
    delivery_ref: str,
    worklog_ref: str,
    sync_ref: str,
    handoff_ref: str,
    gate_report_ref: str,
    reuse_attestation_ref: str | None,
) -> tuple[GateRecord, Stage0GateReport, tuple[str, ...]]:
    evidence_refs = tuple(
        dict.fromkeys(
            (
                observation_ref,
                delivery_ref,
                worklog_ref,
                worklog_ref.removesuffix(".json") + ".md",
                sync_ref,
                handoff_ref,
                handoff_ref.removesuffix(".json") + ".md",
                *((reuse_attestation_ref,) if reuse_attestation_ref is not None else ()),
                *gate_refs.values(),
            )
        )
    )
    checks = (
        Stage0GateCheck(
            "stage0.G10-UPSTREAM-GATES",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "Every Stage 0 G0-G9 hard gate was loaded from formal committed evidence and is PASS; cross-commit reuse, when present, is independently attested.",
            measurements={
                "gate_statuses": {key: value.status.value for key, value in gate_records.items()},
                "cross_commit_reuse": reuse_attestation_ref is not None,
            },
            evidence_refs=tuple(
                dict.fromkeys(
                    (
                        *gate_refs.values(),
                        *((reuse_attestation_ref,) if reuse_attestation_ref is not None else ()),
                    )
                )
            ),
        ),
        Stage0GateCheck(
            "stage0.G10-THREE-END-SYNC",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "Local, GitHub and server heads match; both worktrees are clean and the reviewed update is fast-forward ancestry without force push.",
            measurements={
                "commit": source.git_commit,
                "branch": source.git_branch,
                "agent_hashes": observation["agent_sync"],
                "bundle_cleanup": observation["bundle_cleanup"],
            },
            evidence_refs=(observation_ref, sync_ref),
        ),
        Stage0GateCheck(
            "stage0.G10-DELIVERY-AND-ASSETS",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "Every tracked repository file and all thirteen authoritative server assets are inventoried; forbidden large runtime artifacts and high-confidence secrets are absent from Git.",
            measurements={
                "repository_file_count": repository_inventory["tracked_file_count"],
                "asset_count": assets["asset_count"],
                "evidence_count": evidence_inventory["evidence_count"],
            },
            evidence_refs=(delivery_ref,),
        ),
        Stage0GateCheck(
            "stage0.G10-PERSISTENCE-AND-HANDOFF",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "The time-bounded G1-D risk acceptance remains valid, is not described as backup, and the Stage 1 handoff preserves the mathematical boundary.",
            measurements={"persistence": persistence, "stage1_handoff_ref": handoff_ref},
            evidence_refs=(delivery_ref, handoff_ref),
        ),
    )
    gate = GateRecord(
        gate_id=GATE_ID,
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "upstream_gate_count": len(gate_records),
            "matching_git_heads": 3,
            "agent_file_count": 5,
            "tracked_file_count": repository_inventory["tracked_file_count"],
            "asset_count": assets["asset_count"],
            "bundle_cleanup_complete": True,
            "readiness_status": "READY",
        },
        threshold={
            "all_upstream_gates": "PASS",
            "matching_git_heads_required": 3,
            "agent_hashes_equal": True,
            "forbidden_git_artifacts_max": 0,
            "bundle_residue_max": 0,
            "persistence_decision_must_be_current": True,
        },
        evidence_refs=(*evidence_refs, gate_report_ref),
    )
    report = Stage0GateReport(
        gate_id=GATE_ID,
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=request.environment.environment_hash,
        config_hashes={TASK_ID: request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                reference,
                sha256_file(_logical_path(root, reference, field="gate.input_evidence")),
                "stage0-g10-supporting-evidence",
            )
            for reference in evidence_refs
        ),
        checks=checks,
    )
    return gate, report, evidence_refs


def run_formal_g10_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G10Error("G10_FORMAL_REQUEST_INVALID")
    if not {"git", "github", "server"} <= request.environment.capabilities:
        raise Stage0G10Error("G10_REQUIRED_CAPABILITIES_MISSING")
    observation_ref = _observation_ref(request)
    observation = validate_sync_observation(root, observation_ref, source)
    reuse_attestation_ref, _reuse_attestation = _validated_reuse_attestation(
        request,
        root,
        source,
    )
    gate_records, gate_refs = _validated_gate_records(request.environment, root)
    checked_at = str(observation["observed_at"])
    repository_inventory = _repository_inventory(source, observation)
    assets = _asset_inventory(request.environment, root, source.git_commit)
    evidence_inventory = _evidence_inventory(request.environment, root)
    persistence = _persistence_status(source, gate_records)
    base_ref = f"evidence/stage0/g10-final/{source.git_commit}/{request.config.config_hash}"
    delivery_ref = f"{base_ref}/delivery-manifest.json"
    sync_ref = f"{base_ref}/sync-report.json"
    worklog_ref = f"{base_ref}/worklog.json"
    handoff_ref = f"{base_ref}/stage1-handoff.json"
    gate_report_ref = f"{base_ref}/gate-report.json"
    readiness_ref = f"evidence/stage0/readiness/{source.git_commit}/{request.config.config_hash}/READY.json"
    handoff = _handoff(request, source, assets, persistence, checked_at)
    sync_report = _with_hash(
        {
            "schema_version": SYNC_REPORT_SCHEMA,
            "status": "PASS",
            "generated_at": checked_at,
            "generator_git_commit": source.git_commit,
            "branch": source.git_branch,
            "sync_observation_ref": observation_ref,
            "sync_observation_sha256": sha256_file(
                _logical_path(root, observation_ref, field="sync_observation_ref")
            ),
            "sync_observation_artifact_hash": str(observation["artifact_hash"]),
            "local_head": source.git_commit,
            "github_head": str(_mapping(observation["github"], field="github")["head"]),
            "server_head": str(_mapping(observation["server"], field="server")["head"]),
            "local_worktree_clean": True,
            "server_worktree_clean": True,
            "fast_forward_only": True,
            "force_push_used": False,
            "agent_sync": observation["agent_sync"],
            "bundle_cleanup": observation["bundle_cleanup"],
        }
    )
    delivery = _with_hash(
        {
            "schema_version": DELIVERY_SCHEMA,
            "status": "PASS",
            "generated_at": checked_at,
            "generator_git_commit": source.git_commit,
            "git_branch": source.git_branch,
            "repository_inventory": repository_inventory,
            "server_asset_inventory": assets,
            "server_evidence_inventory": evidence_inventory,
            "gate_records": {key: value.to_dict() for key, value in gate_records.items()},
            "gate_evidence_refs": dict(gate_refs),
            "persistence_status": persistence,
            "sync_observation_ref": observation_ref,
            "stage1_handoff_ref": handoff_ref,
            "readiness_ref": readiness_ref,
            "large_runtime_artifacts_in_git": False,
            "authoritative_runtime_copy_described_as_backup": False,
        }
    )
    worklog = _worklog(
        source,
        checked_at,
        gate_records,
        observation_ref,
        delivery_ref,
        sync_ref,
        handoff_ref,
        readiness_ref,
        persistence,
    )
    _write_or_verify(_logical_path(root, delivery_ref, field="delivery_ref"), delivery)
    _write_or_verify(_logical_path(root, sync_ref, field="sync_ref"), sync_report)
    _write_or_verify(_logical_path(root, worklog_ref, field="worklog_ref"), worklog)
    _write_bytes_or_verify(
        _logical_path(root, worklog_ref.removesuffix(".json") + ".md", field="worklog_md_ref"),
        _worklog_markdown(worklog),
    )
    _write_or_verify(_logical_path(root, handoff_ref, field="handoff_ref"), handoff)
    _write_bytes_or_verify(
        _logical_path(root, handoff_ref.removesuffix(".json") + ".md", field="handoff_md_ref"),
        _handoff_markdown(handoff),
    )
    gate, gate_report, gate_support_refs = _gate_payloads(
        request=request,
        root=root,
        source=source,
        checked_at=checked_at,
        observation=observation,
        observation_ref=observation_ref,
        gate_records=gate_records,
        gate_refs=gate_refs,
        repository_inventory=repository_inventory,
        assets=assets,
        evidence_inventory=evidence_inventory,
        persistence=persistence,
        delivery_ref=delivery_ref,
        worklog_ref=worklog_ref,
        sync_ref=sync_ref,
        handoff_ref=handoff_ref,
        gate_report_ref=gate_report_ref,
        reuse_attestation_ref=reuse_attestation_ref,
    )
    _write_or_verify(
        _logical_path(root, gate_report_ref, field="gate_report_ref"),
        gate_report.to_dict(),
    )
    all_gates = {key: value.to_dict() for key, value in gate_records.items()}
    all_gates[GATE_ID] = gate.to_dict()
    readiness_evidence_refs = tuple(
        dict.fromkeys(
            (
                *gate_support_refs,
                gate_report_ref,
                *source_refs,
                *((reuse_attestation_ref,) if reuse_attestation_ref is not None else ()),
            )
        )
    )
    readiness = _with_hash(
        {
            "schema_version": READINESS_SCHEMA,
            "status": "READY",
            "stage": 0,
            "published_at": checked_at,
            "generator_git_commit": source.git_commit,
            "git_branch": source.git_branch,
            "environment_hash": request.environment.environment_hash,
            "config_hash": request.config.config_hash,
            "all_hard_gates_pass": True,
            "approved_exceptions": [],
            "gate_records": all_gates,
            "persistence_status": persistence,
            "delivery_manifest_ref": delivery_ref,
            "sync_report_ref": sync_ref,
            "worklog_ref": worklog_ref,
            "stage1_handoff_ref": handoff_ref,
            "gate_report_ref": gate_report_ref,
            "evidence": [
                {
                    "ref": reference,
                    "sha256": sha256_file(
                        _logical_path(root, reference, field="readiness.evidence")
                    ),
                }
                for reference in readiness_evidence_refs
            ],
        }
    )
    _write_or_verify(_logical_path(root, readiness_ref, field="readiness_ref"), readiness)
    canonical = _with_hash(
        {
            "schema_version": "stage0-g10-canonical-evidence-v1",
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "critical_source_hashes": dict(source.critical_source_hashes),
            "config_hash": request.config.config_hash,
            "environment_hash": request.environment.environment_hash,
            "sync_observation_ref": observation_ref,
            "sync_observation_artifact_hash": observation["artifact_hash"],
            "artifacts": {
                "delivery_manifest": {
                    "ref": delivery_ref,
                    "sha256": sha256_file(_logical_path(root, delivery_ref, field="delivery_ref")),
                    "artifact_hash": delivery["artifact_hash"],
                },
                "sync_report": {
                    "ref": sync_ref,
                    "sha256": sha256_file(_logical_path(root, sync_ref, field="sync_ref")),
                    "artifact_hash": sync_report["artifact_hash"],
                },
                "worklog": {
                    "ref": worklog_ref,
                    "sha256": sha256_file(_logical_path(root, worklog_ref, field="worklog_ref")),
                    "artifact_hash": worklog["artifact_hash"],
                },
                "stage1_handoff": {
                    "ref": handoff_ref,
                    "sha256": sha256_file(_logical_path(root, handoff_ref, field="handoff_ref")),
                    "artifact_hash": handoff["artifact_hash"],
                },
                "gate_report": {
                    "ref": gate_report_ref,
                    "sha256": sha256_file(_logical_path(root, gate_report_ref, field="gate_report_ref")),
                    "artifact_hash": gate_report.artifact_hash,
                },
                "readiness": {
                    "ref": readiness_ref,
                    "sha256": sha256_file(_logical_path(root, readiness_ref, field="readiness_ref")),
                    "artifact_hash": readiness["artifact_hash"],
                },
            },
            "upstream_gate_records": {key: value.to_dict() for key, value in gate_records.items()},
            "gate_record": gate.to_dict(),
            "gate_report": gate_report.to_dict(),
        }
    )
    publication_refs = tuple(
        dict.fromkeys((*readiness_evidence_refs, readiness_ref))
    )
    outputs: dict[str, str] = {}
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g10-evidence-v1",
            "artifact_role": kind,
            "task_id": TASK_ID,
            "config_hash": request.config.config_hash,
            "environment_hash": request.environment.environment_hash,
            "canonical_evidence_hash": canonical["artifact_hash"],
            "canonical_evidence": canonical,
            "role_summary": (
                {
                    "tracked_file_count": repository_inventory["tracked_file_count"],
                    "asset_count": assets["asset_count"],
                    "delivery_manifest_ref": delivery_ref,
                }
                if kind == "delivery_manifest"
                else {
                    "language": "zh-CN",
                    "entry_count": len(worklog["entries"]),
                    "worklog_ref": worklog_ref,
                }
                if kind == "worklog"
                else {
                    "matching_heads": 3,
                    "agent_file_count": 5,
                    "readiness_ref": readiness_ref,
                }
            ),
        }
        published = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
            source_refs=publication_refs,
        )
        outputs[kind] = published.commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=outputs,
        message="Stage 0 G10 delivery, synchronization, and readiness gate passed",
        metadata={
            "stage0_g10_specialized": True,
            "gate_id": GATE_ID,
            "readiness_ref": readiness_ref,
        },
    )


def _load_canonical_from_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G10Error("G10_OUTPUT_KINDS_INVALID")
    canonical: dict[str, Any] | None = None
    for kind in request.task.artifact_kinds:
        loaded = load_committed_task_artifact(root, outputs[kind], require_formal=True)
        payload = _mapping(loaded.payload, field=f"payload.{kind}")
        if (
            payload.get("schema_version") != "stage0-g10-evidence-v1"
            or payload.get("artifact_role") != kind
            or payload.get("task_id") != TASK_ID
            or payload.get("config_hash") != request.config.config_hash
            or payload.get("environment_hash") != request.environment.environment_hash
        ):
            raise Stage0G10Error("G10_OUTPUT_ENVELOPE_INVALID")
        current = _mapping(payload.get("canonical_evidence"), field="canonical_evidence")
        if payload.get("canonical_evidence_hash") != current.get("artifact_hash"):
            raise Stage0G10Error("G10_OUTPUT_CANONICAL_HASH_INVALID")
        if canonical is None:
            canonical = current
        elif canonical != current:
            raise Stage0G10Error("G10_OUTPUT_CANONICAL_EVIDENCE_DRIFT")
    assert canonical is not None
    declared = canonical.pop("artifact_hash", None)
    if declared != canonical_json_hash(canonical):
        raise Stage0G10Error("G10_OUTPUT_CANONICAL_ARTIFACT_HASH_INVALID")
    canonical["artifact_hash"] = declared
    return canonical


def validate_formal_g10_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    canonical = _load_canonical_from_outputs(request, root, outputs)
    if (
        canonical.get("status") != "PASS"
        or canonical.get("generator_git_commit") != source.git_commit
        or canonical.get("critical_source_hashes") != dict(source.critical_source_hashes)
        or canonical.get("config_hash") != request.config.config_hash
        or canonical.get("environment_hash") != request.environment.environment_hash
    ):
        raise Stage0G10Error("G10_OUTPUT_SOURCE_OR_IDENTITY_INVALID")
    observation_ref = canonical.get("sync_observation_ref")
    if not isinstance(observation_ref, str) or observation_ref != _observation_ref(request):
        raise Stage0G10Error("G10_OUTPUT_SYNC_OBSERVATION_REF_INVALID")
    observation = validate_sync_observation(root, observation_ref, source)
    if canonical.get("sync_observation_artifact_hash") != observation["artifact_hash"]:
        raise Stage0G10Error("G10_OUTPUT_SYNC_OBSERVATION_HASH_DRIFT")
    gate_records, gate_refs = _validated_gate_records(request.environment, root)
    if canonical.get("upstream_gate_records") != {
        key: value.to_dict() for key, value in gate_records.items()
    }:
        raise Stage0G10Error("G10_OUTPUT_UPSTREAM_GATE_DRIFT")
    artifacts = _mapping(canonical.get("artifacts"), field="canonical.artifacts")
    expected_schemas = {
        "delivery_manifest": DELIVERY_SCHEMA,
        "sync_report": SYNC_REPORT_SCHEMA,
        "worklog": WORKLOG_SCHEMA,
        "stage1_handoff": HANDOFF_SCHEMA,
        "readiness": READINESS_SCHEMA,
    }
    loaded: dict[str, dict[str, Any]] = {}
    expected_keys = set(expected_schemas) | {"gate_report"}
    if set(artifacts) != expected_keys:
        raise Stage0G10Error("G10_OUTPUT_EXTERNAL_ARTIFACT_SET_INVALID")
    for name, descriptor_raw in artifacts.items():
        descriptor = _mapping(descriptor_raw, field=f"canonical.artifacts.{name}")
        reference = descriptor.get("ref")
        path = _logical_path(root, reference, field=f"canonical.artifacts.{name}.ref")
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise Stage0G10Error(f"G10_OUTPUT_EXTERNAL_ARTIFACT_SHA_DRIFT:{name}")
        if name != "gate_report":
            value = _load_hashed(path, schema=expected_schemas[name], field=name)
            if value.get("artifact_hash") != descriptor.get("artifact_hash"):
                raise Stage0G10Error(f"G10_OUTPUT_EXTERNAL_ARTIFACT_HASH_DRIFT:{name}")
            loaded[name] = value
    delivery = loaded["delivery_manifest"]
    repository_inventory = _repository_inventory(source, observation)
    assets = _asset_inventory(request.environment, root, source.git_commit)
    evidence_inventory = _evidence_inventory(request.environment, root)
    persistence = _persistence_status(source, gate_records)
    if (
        delivery.get("status") != "PASS"
        or delivery.get("generator_git_commit") != source.git_commit
        or delivery.get("repository_inventory") != repository_inventory
        or delivery.get("server_asset_inventory") != assets
        or delivery.get("server_evidence_inventory") != evidence_inventory
        or delivery.get("gate_records") != {key: value.to_dict() for key, value in gate_records.items()}
        or delivery.get("gate_evidence_refs") != gate_refs
        or delivery.get("persistence_status") != persistence
        or delivery.get("large_runtime_artifacts_in_git") is not False
        or delivery.get("authoritative_runtime_copy_described_as_backup") is not False
    ):
        raise Stage0G10Error("G10_OUTPUT_DELIVERY_MANIFEST_DRIFT")
    sync = loaded["sync_report"]
    if (
        sync.get("status") != "PASS"
        or sync.get("generator_git_commit") != source.git_commit
        or sync.get("sync_observation_ref") != observation_ref
        or sync.get("sync_observation_artifact_hash") != observation["artifact_hash"]
        or sync.get("local_head") != source.git_commit
        or sync.get("github_head") != source.git_commit
        or sync.get("server_head") != source.git_commit
        or sync.get("fast_forward_only") is not True
        or sync.get("force_push_used") is not False
        or sync.get("agent_sync") != observation["agent_sync"]
        or sync.get("bundle_cleanup") != observation["bundle_cleanup"]
    ):
        raise Stage0G10Error("G10_OUTPUT_SYNC_REPORT_DRIFT")
    worklog = loaded["worklog"]
    handoff = loaded["stage1_handoff"]
    if (
        worklog.get("status") != "PASS"
        or worklog.get("language") != "zh-CN"
        or worklog.get("git_commit") != source.git_commit
        or worklog.get("git_branch") != source.git_branch
        or worklog.get("secrets_included") is not False
        or worklog.get("temporary_download_urls_included") is not False
        or handoff.get("status") != "READY_FOR_STAGE1_ENTRY"
        or handoff.get("generator_git_commit") != source.git_commit
        or handoff.get("environment_id") != request.environment.environment_hash
        or _mapping(handoff.get("entry_requirements"), field="handoff.entry_requirements").get(
            "importance_correctness_must_be_proven_in_stage1"
        ) is not True
    ):
        raise Stage0G10Error("G10_OUTPUT_WORKLOG_OR_HANDOFF_INVALID")
    gate = GateRecord.from_mapping(_mapping(canonical.get("gate_record"), field="gate_record"))
    gate_report = Stage0GateReport.from_mapping(
        _mapping(canonical.get("gate_report"), field="gate_report")
    )
    gate_descriptor = _mapping(artifacts["gate_report"], field="gate_report_descriptor")
    stored_gate_report = Stage0GateReport.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, gate_descriptor["ref"], field="gate_report.ref")
            ),
            field="stored_gate_report",
        )
    )
    if (
        gate.gate_id != GATE_ID
        or gate.status is not GateStatus.PASS
        or gate_report.status.value != "PASS"
        or stored_gate_report != gate_report
        or gate_report.artifact_hash != gate_descriptor.get("artifact_hash")
    ):
        raise Stage0G10Error("G10_OUTPUT_GATE_NOT_PASS_OR_DRIFTED")
    for item in gate_report.input_evidence:
        path = _logical_path(root, item.ref, field="gate_report.input_evidence")
        if not path.is_file() or sha256_file(path) != item.sha256:
            raise Stage0G10Error("G10_OUTPUT_GATE_EVIDENCE_SHA_DRIFT")
    readiness = loaded["readiness"]
    readiness_gates = _mapping(readiness.get("gate_records"), field="readiness.gate_records")
    expected_gate_ids = set(_REQUIRED_GATES) | {GATE_ID}
    if (
        readiness.get("status") != "READY"
        or readiness.get("stage") != 0
        or readiness.get("generator_git_commit") != source.git_commit
        or readiness.get("git_branch") != source.git_branch
        or readiness.get("all_hard_gates_pass") is not True
        or readiness.get("approved_exceptions") != []
        or readiness.get("persistence_status") != persistence
        or set(readiness_gates) != expected_gate_ids
        or any(
            (
                (parsed := GateRecord.from_mapping(
                    _mapping(value, field=f"readiness.gate.{key}")
                )).status is not GateStatus.PASS
                or parsed.gate_id != key
            )
            for key, value in readiness_gates.items()
        )
        or readiness_gates.get(GATE_ID) != gate.to_dict()
    ):
        raise Stage0G10Error("G10_OUTPUT_READINESS_INVALID")
    for raw in _array(readiness.get("evidence"), field="readiness.evidence"):
        descriptor = _mapping(raw, field="readiness.evidence.item")
        path = _logical_path(root, descriptor.get("ref"), field="readiness.evidence.ref")
        if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
            raise Stage0G10Error("G10_OUTPUT_READINESS_EVIDENCE_DRIFT")
    return gate


def _environment_with_github_capability(
    *,
    root: Path,
    state: Stage0G9FormalState,
    observation_ref: str,
    observation: Mapping[str, Any],
) -> TaskRuntimeEnvironment:
    formal_dir = f"evidence/stage0/g10-capability/{state.gate_artifact_hash}"
    capability = RuntimeCapabilityEvidence(
        capability="github",
        status="VERIFIED",
        checked_at=str(observation["observed_at"]),
        evidence_refs=(observation_ref,),
        metadata={
            "remote_url": REMOTE_URL,
            "branch": str(observation["branch"]),
            "head": str(observation["expected_commit"]),
            "push_verified": True,
            "force_push_used": False,
        },
    )
    published = TaskArtifactStore(root, formal_dir).publish(
        task_id=TASK_ID,
        artifact_kind="capability_github",
        config_hash=state.config.config_hash,
        run_intent="formal",
        payload=capability.to_dict(),
        formal_eligible=True,
        source_refs=(observation_ref,),
    )
    refs = dict(state.environment.evidence_refs)
    refs["capability_github"] = published.commit_ref
    return TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities | frozenset({"github"}),
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids,
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )


def execute_stage0_g10(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g9_index_ref: str,
    sync_observation_ref: str,
    reuse_attestation_ref: str | None = None,
) -> Stage0G10FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g9_formal_state(
        data_root=root,
        index_ref=g9_index_ref,
    )
    source = _capture_source()
    if (
        source.git_commit != binding.git_commit
        or source.git_branch != binding.git_branch
        or source.repository != binding.repository
    ):
        raise Stage0G10Error("G10_EXECUTION_SOURCE_BINDING_MISMATCH")
    if state.generator_git_commit != source.git_commit and reuse_attestation_ref is None:
        raise Stage0G10Error("G10_CROSS_COMMIT_REUSE_ATTESTATION_REQUIRED")
    if state.generator_git_commit == source.git_commit and reuse_attestation_ref is not None:
        raise Stage0G10Error("G10_CROSS_COMMIT_REUSE_ATTESTATION_UNNECESSARY")
    observation = validate_sync_observation(root, sync_observation_ref, source)
    input_environment = _environment_with_github_capability(
        root=root,
        state=state,
        observation_ref=sync_observation_ref,
        observation=observation,
    )
    config = build_stage0_g10_config(
        binding=binding,
        state=state,
        sync_observation_ref=sync_observation_ref,
        reuse_attestation_ref=reuse_attestation_ref,
    )
    formal_dir = f"evidence/stage0/g10-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    _write_or_verify(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=input_environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G10Error(f"G10_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}")
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, input_environment)
    gate = validate_formal_g10_outputs(request, root, outputs)
    canonical = _load_canonical_from_outputs(request, root, outputs)
    artifacts = _mapping(canonical["artifacts"], field="canonical.artifacts")
    readiness_descriptor = _mapping(artifacts["readiness"], field="readiness_descriptor")
    handoff_descriptor = _mapping(artifacts["stage1_handoff"], field="handoff_descriptor")
    refs = dict(input_environment.evidence_refs)
    refs.update(
        {
            "g10_delivery_manifest": outputs["delivery_manifest"],
            "g10_worklog": outputs["worklog"],
            "g10_sync_report": outputs["sync_report"],
            "gate_stage0_g10": outputs["sync_report"],
            "stage0_readiness": str(readiness_descriptor["ref"]),
            "stage1_handoff": str(handoff_descriptor["ref"]),
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=input_environment.capabilities,
        frozen_contract_stages=input_environment.frozen_contract_stages,
        passed_gate_ids=input_environment.passed_gate_ids | frozenset({GATE_ID}),
        estimator_decision_ref=input_environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    _write_or_verify(
        _logical_path(root, environment_ref, field="environment_ref"), environment.to_dict()
    )
    index = _with_hash(
        {
            "schema_version": "stage0-g10-formalization-index-v1",
            "generator_git_commit": source.git_commit,
            "checked_at": gate.checked_at,
            "g9_index_ref": state.index_ref,
            "g9_index_sha256": state.index_sha256,
            "g9_gate_artifact_hash": state.gate_artifact_hash,
            "g9_generator_git_commit": state.generator_git_commit,
            "reuse_attestation_ref": reuse_attestation_ref,
            "reuse_attestation_sha256": (
                None
                if reuse_attestation_ref is None
                else sha256_file(
                    _logical_path(root, reuse_attestation_ref, field="reuse_attestation_ref")
                )
            ),
            "sync_observation_ref": sync_observation_ref,
            "sync_observation_sha256": sha256_file(
                _logical_path(root, sync_observation_ref, field="sync_observation_ref")
            ),
            "config_ref": config_ref,
            "config_hash": config.config_hash,
            "input_environment_hash": input_environment.environment_hash,
            "task_output_refs": outputs,
            "gate_ref": outputs["sync_report"],
            "gate_artifact_hash": gate.artifact_hash,
            "readiness_ref": str(readiness_descriptor["ref"]),
            "readiness_sha256": str(readiness_descriptor["sha256"]),
            "readiness_artifact_hash": str(readiness_descriptor["artifact_hash"]),
            "environment_ref": environment_ref,
            "environment_hash": environment.environment_hash,
            "next_task_id": "stage1.01_entry_and_contract",
            "next_input_refs": list(outputs.values()),
        }
    )
    index_ref = f"{formal_dir}/index.json"
    _write_or_verify(_logical_path(root, index_ref, field="index_ref"), index)
    return Stage0G10FormalizationResult(
        environment,
        outputs,
        config_ref,
        environment_ref,
        index_ref,
        str(readiness_descriptor["ref"]),
    )


def load_stage0_g10_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G10FormalState:
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _load_hashed(
        index_path,
        schema="stage0-g10-formalization-index-v1",
        field="g10_index",
    )
    expected_fields = {
        "schema_version", "generator_git_commit", "checked_at", "g9_index_ref",
        "g9_index_sha256", "g9_gate_artifact_hash", "g9_generator_git_commit",
        "reuse_attestation_ref", "reuse_attestation_sha256", "sync_observation_ref",
        "sync_observation_sha256", "config_ref", "config_hash",
        "input_environment_hash", "task_output_refs", "gate_ref",
        "gate_artifact_hash", "readiness_ref", "readiness_sha256",
        "readiness_artifact_hash", "environment_ref", "environment_hash",
        "next_task_id", "next_input_refs", "artifact_hash",
    }
    if set(raw) != expected_fields or raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G10Error("G10_STATE_INDEX_FIELDS_OR_SOURCE_INVALID")
    g9 = load_stage0_g9_formal_state(
        data_root=root,
        index_ref=str(raw["g9_index_ref"]),
        expected_git_commit=str(raw["g9_generator_git_commit"]),
    )
    reuse_ref = raw.get("reuse_attestation_ref")
    reuse_sha256 = raw.get("reuse_attestation_sha256")
    if g9.generator_git_commit == expected_git_commit:
        if reuse_ref is not None or reuse_sha256 is not None:
            raise Stage0G10Error("G10_STATE_REUSE_ATTESTATION_UNNECESSARY")
    else:
        if not isinstance(reuse_ref, str) or not isinstance(reuse_sha256, str):
            raise Stage0G10Error("G10_STATE_REUSE_ATTESTATION_MISSING")
        reuse_path = _logical_path(root, reuse_ref, field="reuse_attestation_ref")
        if sha256_file(reuse_path) != reuse_sha256:
            raise Stage0G10Error("G10_STATE_REUSE_ATTESTATION_SHA_DRIFT")
        source = _capture_source()
        if source.git_commit != expected_git_commit:
            raise Stage0G10Error("G10_STATE_CONSUMER_SOURCE_MISMATCH")
        try:
            validate_evidence_reuse_attestation(
                repository=source.repository,
                data_root=root,
                attestation_ref=reuse_ref,
                producer_commit=g9.generator_git_commit,
                consumer_commit=expected_git_commit,
                consumer_branch=source.git_branch,
                scope_id="stage0.G0-G9",
                source_evidence_ref=g9.index_ref,
                required_gate_ids=sorted(_REQUIRED_GATES),
            )
        except EvidenceReuseError as error:
            raise Stage0G10Error(f"G10_STATE_REUSE_ATTESTATION_INVALID:{error}") from error
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, raw["config_ref"], field="config_ref")),
            field="config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, raw["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    outputs_raw = _mapping(raw.get("task_output_refs"), field="task_output_refs")
    outputs = {kind: str(outputs_raw[kind]) for kind in config.task_definition.artifact_kinds}
    input_refs = dict(environment.evidence_refs)
    for key in (
        "g10_delivery_manifest", "g10_worklog", "g10_sync_report",
        "gate_stage0_g10", "stage0_readiness", "stage1_handoff",
    ):
        input_refs.pop(key, None)
    input_environment = TaskRuntimeEnvironment(
        capabilities=environment.capabilities,
        frozen_contract_stages=environment.frozen_contract_stages,
        passed_gate_ids=environment.passed_gate_ids - frozenset({GATE_ID}),
        estimator_decision_ref=environment.estimator_decision_ref,
        evidence_refs=input_refs,
    )
    request = TaskExecutionRequest(config, config.task_definition, input_environment)
    gate = validate_formal_g10_outputs(request, root, outputs)
    readiness_path = _logical_path(root, raw["readiness_ref"], field="readiness_ref")
    readiness = _load_hashed(
        readiness_path,
        schema=READINESS_SCHEMA,
        field="readiness",
    )
    next_inputs = raw.get("next_input_refs")
    if (
        raw.get("g9_index_sha256") != g9.index_sha256
        or raw.get("g9_gate_artifact_hash") != g9.gate_artifact_hash
        or raw.get("g9_generator_git_commit") != g9.generator_git_commit
        or raw.get("sync_observation_sha256")
        != sha256_file(_logical_path(root, raw["sync_observation_ref"], field="sync_ref"))
        or config.config_hash != raw.get("config_hash")
        or input_environment.environment_hash != raw.get("input_environment_hash")
        or environment.environment_hash != raw.get("environment_hash")
        or GATE_ID not in environment.passed_gate_ids
        or raw.get("gate_ref") != outputs["sync_report"]
        or raw.get("gate_artifact_hash") != gate.artifact_hash
        or raw.get("readiness_sha256") != sha256_file(readiness_path)
        or raw.get("readiness_artifact_hash") != readiness["artifact_hash"]
        or raw.get("next_task_id") != "stage1.01_entry_and_contract"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(outputs.values())
    ):
        raise Stage0G10Error("G10_STATE_HANDOFF_INVALID")
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
        g9_index_ref=str(raw["g9_index_ref"]),
    )


__all__ = [
    "DELIVERY_SCHEMA",
    "GATE_ID",
    "G10SourceBinding",
    "HANDOFF_SCHEMA",
    "READINESS_SCHEMA",
    "SYNC_REPORT_SCHEMA",
    "Stage0G10Error",
    "Stage0G10FormalState",
    "Stage0G10FormalizationResult",
    "TASK_ID",
    "WORKLOG_SCHEMA",
    "build_stage0_g10_config",
    "execute_stage0_g10",
    "load_stage0_g10_formal_state",
    "run_formal_g10_task",
    "validate_formal_g10_outputs",
    "validate_sync_observation",
]
