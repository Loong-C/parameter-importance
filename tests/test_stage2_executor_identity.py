from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from param_importance_nlp.experiments.stage2_executor_identity import (
    ExecutorIdentityError,
    compute_executor_identity,
    validate_executor_identity,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.invalid")
    _git(repo, "config", "user.name", "test")
    source = repo / "ops" / "stage2" / "run_s207_formal.py"
    source.parent.mkdir(parents=True)
    source.write_text("# launcher\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo, source


def test_executor_identity_is_exact_and_rejects_dirty_or_invalid_wire(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    identity = compute_executor_identity(repo, source)
    assert set(identity) == {"execution_commit", "launcher_source_sha256", "worktree_clean"}
    assert validate_executor_identity(identity) == identity

    (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ExecutorIdentityError, match="DIRTY"):
        compute_executor_identity(repo, source)
    with pytest.raises(ExecutorIdentityError, match="FIELDS_INVALID"):
        validate_executor_identity({**identity, "unexpected": True})
    with pytest.raises(ExecutorIdentityError, match="FIELDS_INVALID"):
        validate_executor_identity({key: value for key, value in identity.items() if key != "worktree_clean"})


def test_executor_identity_rejects_repo_subdir_source_outside_and_nonregular(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    with pytest.raises(ExecutorIdentityError, match="NOT_GIT_TOP_LEVEL"):
        compute_executor_identity(repo / "ops", source)
    outside = tmp_path / "outside.py"
    outside.write_text("# outside\n", encoding="utf-8")
    with pytest.raises(ExecutorIdentityError, match="OUTSIDE_REPOSITORY"):
        compute_executor_identity(repo, outside)
    directory = repo / "ops" / "stage2" / "directory.py"
    directory.mkdir()
    with pytest.raises(ExecutorIdentityError, match="REGULAR_FILE"):
        compute_executor_identity(repo, directory)


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlink API unavailable")
def test_executor_identity_rejects_repo_and_source_symlinks(tmp_path: Path) -> None:
    repo, source = _repo(tmp_path)
    repo_link = tmp_path / "repo-link"
    source_link = tmp_path / "source-link.py"
    try:
        repo_link.symlink_to(repo, target_is_directory=True)
        source_link.symlink_to(source)
    except OSError:
        pytest.skip("symlink creation unavailable on this host")
    with pytest.raises(ExecutorIdentityError, match="SYMLINK_COMPONENT"):
        compute_executor_identity(repo_link, source)
    with pytest.raises(ExecutorIdentityError, match="SYMLINK_COMPONENT"):
        compute_executor_identity(repo, source_link)
