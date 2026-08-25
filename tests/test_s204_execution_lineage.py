"""Fail-closed S2.4 launcher execution-lineage tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import ops.stage2.run_s204_formal as launcher


def test_execution_lineage_binds_clean_detached_head(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    commit = "a" * 40
    monkeypatch.setattr(launcher, "_repository_root", lambda: tmp_path)

    def run(command, **_kwargs):
        if command[-3:] == ["rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout=f"{commit}\n", stderr="")
        if command[-4:] == ["symbolic-ref", "--quiet", "--short", "HEAD"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if command[-3:] == ["status", "--porcelain=v1", "--untracked-files=all"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    lineage = launcher._validate_execution_lineage(commit)
    assert lineage["execution_commit"] == commit
    assert lineage["role"] == "stage2.04_s204_launcher_execution"
    assert lineage["detached"] is True and lineage["worktree_clean"] is True


@pytest.mark.parametrize(
    ("commit", "branch", "status", "match"),
    [
        ("b" * 39, 1, "", "exactly 40"),
        ("b" * 40, 0, "", "REQUIRES_DETACHED_HEAD"),
        ("b" * 40, 1, " M tracked.py\n", "WORKTREE_NOT_CLEAN"),
    ],
)
def test_execution_lineage_rejects_identity_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    commit: str,
    branch: int,
    status: str,
    match: str,
) -> None:
    expected = "b" * 40
    monkeypatch.setattr(launcher, "_repository_root", lambda: tmp_path)

    def run(command, **_kwargs):
        if "rev-parse" in command:
            return SimpleNamespace(returncode=0, stdout=f"{expected}\n", stderr="")
        if "symbolic-ref" in command:
            return SimpleNamespace(returncode=branch, stdout=("main\n" if branch == 0 else ""), stderr="")
        if "status" in command:
            return SimpleNamespace(returncode=0, stdout=status, stderr="")
        raise AssertionError(command)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(ValueError, match=match):
        launcher._validate_execution_lineage(commit)
