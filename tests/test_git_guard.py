from __future__ import annotations

from pathlib import Path

from param_importance_nlp.git_guard import inspect_path


def _file(root: Path, name: str, payload: bytes = b"fixture") -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def test_guard_rejects_checkpoint_type(tmp_path: Path) -> None:
    target = _file(tmp_path, "artifact/model.safetensors")
    finding = inspect_path(target, repo_root=tmp_path)
    assert finding is not None
    assert finding.reason == "forbidden_type:.safetensors"


def test_guard_rejects_forbidden_directory(tmp_path: Path) -> None:
    target = _file(tmp_path, "datasets/readme.md")
    finding = inspect_path(target, repo_root=tmp_path)
    assert finding is not None
    assert finding.reason == "forbidden_directory"


def test_guard_rejects_large_file(tmp_path: Path) -> None:
    target = _file(tmp_path, "report.txt", b"x" * 11)
    finding = inspect_path(target, repo_root=tmp_path, max_bytes=10)
    assert finding is not None
    assert finding.reason == "exceeds_10_bytes"


def test_guard_requires_binary_review(tmp_path: Path) -> None:
    target = _file(tmp_path, "summary.npz")
    finding = inspect_path(target, repo_root=tmp_path)
    assert finding is not None
    assert finding.reason == "binary_review_required:.npz"


def test_guard_honors_exact_allowlist(tmp_path: Path) -> None:
    target = _file(tmp_path, "reports/approved.zip")
    finding = inspect_path(
        target, repo_root=tmp_path, allowlist=frozenset({"reports/approved.zip"})
    )
    assert finding is None
