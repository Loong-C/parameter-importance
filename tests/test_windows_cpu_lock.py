from __future__ import annotations

from pathlib import Path

from ops.stage0.rebuild_environment import hashed_requirements_bytes
from param_importance_nlp.environment import (
    parse_requirements_lock,
    parse_wheelhouse_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_windows_cpu_lock_has_one_hash_bound_wheel_per_pin() -> None:
    """Windows 本机锁与 wheel 清单必须完整一一对应。

    这里只验证仓库内的不可变元数据，不访问网络，也不要求 wheel 留在仓库中。
    ``hashed_requirements_bytes`` 会同时拒绝缺 wheel、版本不匹配或多候选歧义。
    """

    lock = parse_requirements_lock(
        REPOSITORY_ROOT / "environment" / "windows-cpu-requirements.lock"
    )
    manifest = parse_wheelhouse_manifest(
        REPOSITORY_ROOT / "environment" / "windows-cpu-wheel-manifest.tsv"
    )

    rendered = hashed_requirements_bytes(lock, manifest).decode("ascii")
    checked_in_hashed_lock = (
        REPOSITORY_ROOT / "environment" / "windows-cpu-requirements-hashed.lock"
    ).read_text(encoding="utf-8")
    checked_in_requirement_lines = [
        line
        for line in checked_in_hashed_lock.splitlines()
        if line and not line.startswith("#")
    ]
    requirement_lines = [line for line in rendered.splitlines() if line]

    assert len(lock.versions) == 32
    assert len(manifest.entries) == len(lock.versions)
    assert len(requirement_lines) == len(lock.versions)
    assert all(" --hash=sha256:" in line for line in requirement_lines)
    assert checked_in_requirement_lines == requirement_lines
    assert any(line.startswith("torch==2.10.0+cpu ") for line in requirement_lines)
