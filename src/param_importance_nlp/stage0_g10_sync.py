"""Read-only three-end synchronization observation for Stage 0 G10.

The collector runs on the local workstation *after* the reviewed branch has
been pushed, fast-forwarded on the server, the five ignored ``Agent`` files
have been synchronized, and the deterministic transfer bundle has been
removed.  It does not perform any mutation.  The resulting canonical JSON is
copied into ``DATA_ROOT`` and consumed by the formal server-side G10 gate.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Final, Mapping, Sequence

from .atomic import atomic_write_bytes, sha256_file
from .contracts import canonical_json_hash, load_canonical_json
from .contracts.jsonio import JSONValue


SYNC_OBSERVATION_SCHEMA: Final = "stage0-g10-sync-observation-v1"
SYNC_COLLECTOR_VERSION: Final = "stage0-g10-sync-collector-v1"
SERVER_HOST: Final = "sophgo13-via-lab"
SERVER_REPOSITORY: Final = "/home/sophgo13/cjl/parameter-importance"
SERVER_DATA_ROOT: Final = "/home/sophgo13/cjl/storage/parameter-importance"
REMOTE_NAME: Final = "origin"
REMOTE_URL: Final = "https://github.com/Loong-C/parameter-importance.git"
AGENT_FILES: Final = (
    "git.md",
    "remote_access.md",
    "server.md",
    "sync.md",
    "worklogs.md",
)
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BRANCH_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")


class Stage0G10SyncError(RuntimeError):
    """The read-only synchronization observation is incomplete or inconsistent."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path,
    timeout: int = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _checked(
    arguments: Sequence[str],
    *,
    cwd: Path,
    field: str,
    timeout: int = 60,
) -> str:
    result = _run(arguments, cwd=cwd, timeout=timeout)
    if result.returncode != 0:
        raise Stage0G10SyncError(
            f"G10_SYNC_COMMAND_FAILED:{field}:{result.returncode}:{result.stderr[-2000:]}"
        )
    return result.stdout.strip()


def _git(repository: Path, *arguments: str, field: str) -> str:
    return _checked(
        (
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            *arguments,
        ),
        cwd=repository,
        field=field,
    )


def _agent_hashes(root: Path) -> dict[str, str]:
    directory = root / "Agent"
    if not directory.is_dir():
        raise Stage0G10SyncError("G10_SYNC_LOCAL_AGENT_DIRECTORY_MISSING")
    files = sorted(path.name for path in directory.iterdir() if path.is_file())
    if files != sorted(AGENT_FILES):
        raise Stage0G10SyncError(f"G10_SYNC_LOCAL_AGENT_FILE_SET_INVALID:{files}")
    return {name: sha256_file(directory / name) for name in AGENT_FILES}


def _remote_probe_script(bundle_name: str) -> str:
    # All paths and the host are fixed project contracts.  The only interpolated
    # value is derived from a validated hexadecimal commit.
    return f"""
import hashlib, json, pathlib, subprocess
repo = pathlib.Path({SERVER_REPOSITORY!r})
data_root = pathlib.Path({SERVER_DATA_ROOT!r})
names = {list(AGENT_FILES)!r}
def git(*args):
    result = subprocess.run(
        ['git', '-c', 'safe.directory={SERVER_REPOSITORY}', '-C', str(repo), *args],
        check=False, capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError('git:' + ':'.join(args) + ':' + result.stderr[-1000:])
    return result.stdout.strip()
agent_dir = repo / 'Agent'
observed_names = sorted(path.name for path in agent_dir.iterdir() if path.is_file())
hashes = {{
    name: hashlib.sha256((agent_dir / name).read_bytes()).hexdigest()
    for name in names
}}
print(json.dumps({{
    'head': git('rev-parse', 'HEAD'),
    'branch': git('branch', '--show-current'),
    'worktree_porcelain': git('status', '--porcelain=v1', '--untracked-files=all'),
    'agent_files': observed_names,
    'agent_hashes': hashes,
    'bundle_path': str(data_root / 'tmp' / {bundle_name!r}),
    'bundle_exists': (data_root / 'tmp' / {bundle_name!r}).exists(),
    'repository': str(repo),
    'data_root': str(data_root),
}}, sort_keys=True, separators=(',', ':')))
""".strip()


def _remote_snapshot(repository: Path, *, bundle_name: str) -> dict[str, Any]:
    script = _remote_probe_script(bundle_name)
    output = _checked(
        ("ssh", SERVER_HOST, "python3", "-c", shlex.quote(script)),
        cwd=repository,
        field="server_snapshot",
        timeout=120,
    )
    try:
        value = json.loads(output)
    except json.JSONDecodeError as error:
        raise Stage0G10SyncError("G10_SYNC_SERVER_SNAPSHOT_JSON_INVALID") from error
    if not isinstance(value, Mapping):
        raise Stage0G10SyncError("G10_SYNC_SERVER_SNAPSHOT_OBJECT_INVALID")
    return dict(value)


def _parse_remote_head(output: str, *, branch: str) -> str:
    rows = [line.split() for line in output.splitlines() if line.strip()]
    expected_ref = f"refs/heads/{branch}"
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != expected_ref:
        raise Stage0G10SyncError("G10_SYNC_GITHUB_BRANCH_RESULT_INVALID")
    commit = rows[0][0]
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise Stage0G10SyncError("G10_SYNC_GITHUB_HEAD_INVALID")
    return commit


def _validate_ancestor(repository: Path, ancestor: str, descendant: str, *, field: str) -> None:
    if _GIT_COMMIT_RE.fullmatch(ancestor) is None or _GIT_COMMIT_RE.fullmatch(descendant) is None:
        raise Stage0G10SyncError(f"G10_SYNC_COMMIT_INVALID:{field}")
    result = _run(
        (
            "git",
            "-c",
            f"safe.directory={repository.as_posix()}",
            "-C",
            str(repository),
            "merge-base",
            "--is-ancestor",
            ancestor,
            descendant,
        ),
        cwd=repository,
    )
    if result.returncode != 0:
        raise Stage0G10SyncError(f"G10_SYNC_NOT_FAST_FORWARD:{field}")


def collect_stage0_g10_sync_observation(
    *,
    repository: str | Path,
    branch: str,
    previous_github_head: str,
    previous_server_head: str,
    authorization_ref: str,
) -> dict[str, JSONValue]:
    """Collect and validate a post-sync observation without changing any endpoint."""

    root = Path(repository).resolve(strict=True)
    if _BRANCH_RE.fullmatch(branch) is None or ".." in branch or branch.endswith("/"):
        raise Stage0G10SyncError("G10_SYNC_BRANCH_INVALID")
    if not authorization_ref.strip() or len(authorization_ref) > 512:
        raise Stage0G10SyncError("G10_SYNC_AUTHORIZATION_REF_INVALID")
    top = Path(_git(root, "rev-parse", "--show-toplevel", field="local_top")).resolve()
    if top != root:
        raise Stage0G10SyncError("G10_SYNC_LOCAL_REPOSITORY_ROOT_INVALID")
    head = _git(root, "rev-parse", "HEAD", field="local_head")
    local_branch = _git(root, "branch", "--show-current", field="local_branch")
    local_status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        field="local_status",
    )
    remote_url = _git(root, "remote", "get-url", REMOTE_NAME, field="remote_url")
    if (
        _GIT_COMMIT_RE.fullmatch(head) is None
        or local_branch != branch
        or local_status
        or remote_url != REMOTE_URL
    ):
        raise Stage0G10SyncError("G10_SYNC_LOCAL_IDENTITY_OR_CLEANLINESS_INVALID")
    _validate_ancestor(root, previous_github_head, head, field="github")
    _validate_ancestor(root, previous_server_head, head, field="server")
    remote_output = _checked(
        ("git", "ls-remote", "--heads", REMOTE_NAME, f"refs/heads/{branch}"),
        cwd=root,
        field="github_head",
        timeout=120,
    )
    github_head = _parse_remote_head(remote_output, branch=branch)
    bundle_name = f"stage0-g10-sync-{head[:12]}.bundle"
    local_bundle_path = root / f".{bundle_name}"
    local_agent_hashes = _agent_hashes(root)
    server = _remote_snapshot(root, bundle_name=bundle_name)
    server_hashes = server.get("agent_hashes")
    if (
        github_head != head
        or server.get("head") != head
        or server.get("branch") != branch
        or server.get("worktree_porcelain") != ""
        or server.get("repository") != SERVER_REPOSITORY
        or server.get("data_root") != SERVER_DATA_ROOT
        or server.get("agent_files") != sorted(AGENT_FILES)
        or server_hashes != local_agent_hashes
        or local_bundle_path.exists()
        or server.get("bundle_exists") is not False
    ):
        raise Stage0G10SyncError("G10_SYNC_THREE_END_OR_AGENT_STATE_INVALID")
    mathematics = root / "docs" / "mathematics.md"
    if not mathematics.is_file():
        raise Stage0G10SyncError("G10_SYNC_MATHEMATICS_DOCUMENT_MISSING")
    observation: dict[str, JSONValue] = {
        "schema_version": SYNC_OBSERVATION_SCHEMA,
        "collector_version": SYNC_COLLECTOR_VERSION,
        "observed_at": _now(),
        "authorization_ref": authorization_ref.strip(),
        "branch": branch,
        "expected_commit": head,
        "previous_github_head": previous_github_head,
        "previous_server_head": previous_server_head,
        "fast_forward_ancestry_verified": True,
        "force_push_used": False,
        "local": {
            "repository": root.as_posix(),
            "head": head,
            "branch": local_branch,
            "worktree_clean": True,
        },
        "github": {
            "remote": REMOTE_NAME,
            "remote_url": remote_url,
            "branch_ref": f"refs/heads/{branch}",
            "head": github_head,
            "push_verified": True,
        },
        "server": {
            "host_alias": SERVER_HOST,
            "repository": SERVER_REPOSITORY,
            "data_root": SERVER_DATA_ROOT,
            "head": str(server["head"]),
            "branch": str(server["branch"]),
            "worktree_clean": True,
            "fast_forward_verified": True,
        },
        "agent_sync": {
            "file_count_each_side": len(AGENT_FILES),
            "files": list(AGENT_FILES),
            "local_sha256": local_agent_hashes,
            "server_sha256": dict(server_hashes),
            "all_equal": True,
        },
        "bundle_cleanup": {
            "bundle_name": bundle_name,
            "local_path": local_bundle_path.as_posix(),
            "server_path": str(server["bundle_path"]),
            "local_absent": True,
            "server_absent": True,
        },
        "preserved_user_content": {
            "path": "docs/mathematics.md",
            "tracked": True,
            "sha256": sha256_file(mathematics),
        },
    }
    observation["artifact_hash"] = canonical_json_hash(observation)
    return observation


def write_sync_observation(path: str | Path, observation: Mapping[str, JSONValue]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        current = load_canonical_json(output)
        if current != dict(observation):
            raise Stage0G10SyncError("G10_SYNC_OBSERVATION_OUTPUT_COLLISION")
        return
    encoded = json.dumps(
        dict(observation),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    atomic_write_bytes(output, encoded)


__all__ = [
    "AGENT_FILES",
    "REMOTE_URL",
    "SERVER_DATA_ROOT",
    "SERVER_HOST",
    "SERVER_REPOSITORY",
    "SYNC_COLLECTOR_VERSION",
    "SYNC_OBSERVATION_SCHEMA",
    "Stage0G10SyncError",
    "collect_stage0_g10_sync_observation",
    "write_sync_observation",
]
