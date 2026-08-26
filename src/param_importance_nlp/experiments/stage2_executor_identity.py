"""Identity of the clean source that starts a formal stage-2 launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess
from typing import Mapping

from .stage2_path_security import DataRootPathError, resolve_no_symlink_path


EXECUTOR_IDENTITY_FIELDS = frozenset(
    {"execution_commit", "launcher_source_sha256", "worktree_clean"}
)
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REAL_POPEN = subprocess.Popen


class ExecutorIdentityError(ValueError):
    """The launcher source or its repository identity is not admissible."""


def _sha256_file(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError as error:
        raise ExecutorIdentityError("LAUNCHER_SOURCE_UNREADABLE") from error


def _git(repository: Path, *args: str) -> str:
    try:
        # Keep identity's Git probe independent from the launcher's child
        # ``Popen`` seam.  Tests and failure handling may replace that seam to
        # model spawn/receipt errors; the identity probe must still be an
        # actual fixed ``git -C <resolved repo>`` command.
        process = _REAL_POPEN(
            ["git", "-C", str(repository), *args],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, _stderr = process.communicate()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExecutorIdentityError("GIT_IDENTITY_UNAVAILABLE") from error
    return stdout.strip()


def compute_executor_identity(
    repository: str | Path,
    launcher_source: str | Path,
) -> dict[str, object]:
    """Read the exact 40-hex HEAD, clean status, and launcher source bytes."""

    try:
        repo = resolve_no_symlink_path(
            repository,
            field="EXECUTOR_REPOSITORY",
            require_directory=True,
        )
        source = resolve_no_symlink_path(
            launcher_source,
            field="LAUNCHER_SOURCE",
            require_regular_file=True,
        )
    except DataRootPathError as error:
        raise ExecutorIdentityError(str(error)) from error
    try:
        source.relative_to(repo)
    except ValueError as error:
        raise ExecutorIdentityError("LAUNCHER_SOURCE_OUTSIDE_REPOSITORY") from error

    # A caller may pass a repository subdirectory, another worktree, or a
    # path whose lexical spelling differs from the actual Git checkout.  Use
    # the resolved checkout as the sole command anchor and bind it to Git's
    # own top-level report before recording any identity fields.
    try:
        top_level = resolve_no_symlink_path(
            _git(repo, "rev-parse", "--show-toplevel"),
            field="EXECUTOR_GIT_TOP_LEVEL",
            require_directory=True,
        )
    except (DataRootPathError, ExecutorIdentityError) as error:
        raise ExecutorIdentityError("EXECUTOR_GIT_TOP_LEVEL_INVALID") from error
    if top_level != repo:
        raise ExecutorIdentityError("EXECUTOR_REPOSITORY_NOT_GIT_TOP_LEVEL")

    commit = _git(repo, "rev-parse", "--verify", "HEAD")
    if _COMMIT.fullmatch(commit) is None:
        raise ExecutorIdentityError("EXECUTION_COMMIT_INVALID")
    status = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if status:
        raise ExecutorIdentityError("EXECUTOR_WORKTREE_DIRTY")
    source_sha = _sha256_file(source)
    return {
        "execution_commit": commit,
        "launcher_source_sha256": source_sha,
        "worktree_clean": True,
    }


def validate_executor_identity(value: object, *, field: str = "executor_identity") -> dict[str, object]:
    """Require the exact three-field identity with no aliases or omissions."""

    if not isinstance(value, Mapping) or set(value) != EXECUTOR_IDENTITY_FIELDS:
        raise ExecutorIdentityError(f"{field}:FIELDS_INVALID")
    commit = value.get("execution_commit")
    source_sha = value.get("launcher_source_sha256")
    if not isinstance(commit, str) or _COMMIT.fullmatch(commit) is None:
        raise ExecutorIdentityError(f"{field}:EXECUTION_COMMIT_INVALID")
    if not isinstance(source_sha, str) or _SHA256.fullmatch(source_sha) is None:
        raise ExecutorIdentityError(f"{field}:LAUNCHER_SOURCE_SHA256_INVALID")
    if value.get("worktree_clean") is not True:
        raise ExecutorIdentityError(f"{field}:WORKTREE_MUST_BE_CLEAN")
    return {
        "execution_commit": commit,
        "launcher_source_sha256": source_sha,
        "worktree_clean": True,
    }


__all__ = [
    "EXECUTOR_IDENTITY_FIELDS",
    "ExecutorIdentityError",
    "compute_executor_identity",
    "validate_executor_identity",
]
