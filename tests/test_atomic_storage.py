from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from param_importance_nlp.atomic import atomic_write_bytes, stable_json_hash
from param_importance_nlp.storage import (
    DATA_ROOT_ENV,
    REQUIRED_DIRECTORIES,
    StorageLayout,
    is_within,
    require_data_root,
    run_storage_canary,
)


def _layout(tmp_path: Path) -> StorageLayout:
    root = tmp_path / "storage" / "parameter-importance"
    root.mkdir(parents=True)
    for name in REQUIRED_DIRECTORIES:
        (root / name).mkdir()
    return StorageLayout.from_value(root)


def test_require_data_root_rejects_missing_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DATA_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="must be set explicitly"):
        require_data_root()


def test_require_data_root_rejects_relative_value() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        require_data_root("relative/data-root")


def test_layout_rejects_escape(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    with pytest.raises(ValueError, match="escapes DATA_ROOT"):
        layout.path("runs", "..", "..", "outside")


def test_layout_validates_all_required_directories(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    assert layout.validate(require_writable=True) == []
    layout.path("reports").rmdir()
    assert layout.validate() == ["missing_directory:reports"]


def test_storage_canary_is_atomic_and_precisely_cleaned(tmp_path: Path) -> None:
    layout = _layout(tmp_path)
    before = set(layout.path("runs").iterdir())
    report = run_storage_canary(layout, "runs")
    assert report["ok"] is True
    assert report["first_sha256"] != report["published_sha256"]
    assert set(layout.path("runs").iterdir()) == before


def test_atomic_write_replaces_content(tmp_path: Path) -> None:
    target = tmp_path / "published.json"
    atomic_write_bytes(target, b"first\n")
    atomic_write_bytes(target, b"second\n")
    assert target.read_bytes() == b"second\n"
    assert not list(tmp_path.glob(".*.tmp-*"))


def test_stable_json_hash_is_order_independent() -> None:
    left = stable_json_hash({"b": 2, "a": True})
    right = stable_json_hash({"a": True, "b": 2})
    assert left == right
    assert left == hashlib.sha256(b'{"a":true,"b":2}\n').hexdigest()


def test_is_within_respects_path_boundary(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert is_within(root / "child", root)
    assert not is_within(tmp_path / "root-sibling", root)
