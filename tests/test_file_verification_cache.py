"""Fail-closed cross-process file verification certificate tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest

import param_importance_nlp.data.pythia_mmap as pythia_mmap


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _certificates(cache_root: Path) -> list[Path]:
    return sorted(cache_root.rglob("*.json"))


def test_verified_sha256_writes_certificate_and_second_call_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"verified artifact")
    cache_root = tmp_path / "formal-cache" / ".file-verification"
    expected = _sha256(source)

    assert pythia_mmap.verified_sha256(source, expected, cache_root) == expected
    certificates = _certificates(cache_root)
    assert len(certificates) == 1
    certificate = json.loads(certificates[0].read_text(encoding="utf-8"))
    assert certificate["resolved_path"] == str(source.resolve())
    assert certificate["expected_sha256"] == expected
    assert certificate["actual_sha256"] == expected
    assert certificate["artifact_hash"] == pythia_mmap.canonical_json_hash(
        {key: value for key, value in certificate.items() if key != "artifact_hash"}
    )

    def fail_if_hashed(_path: str | Path, **_kwargs: object) -> str:
        raise AssertionError("certificate hit unexpectedly hashed the file")

    monkeypatch.setattr(pythia_mmap, "sha256_file", fail_if_hashed)
    assert pythia_mmap.verified_sha256(source, expected, cache_root) == expected


def test_verified_sha256_uses_explicit_environment_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"environment opt-in")
    cache_root = tmp_path / "formal-cache" / ".file-verification"
    expected = _sha256(source)
    monkeypatch.setenv(pythia_mmap.FILE_VERIFICATION_CACHE_ENV, str(cache_root))

    assert pythia_mmap.verified_sha256(source, expected) == expected
    assert len(_certificates(cache_root)) == 1


def test_verified_sha256_rejects_unsafe_cache_roots_without_creating_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"cache root safety")
    expected = _sha256(source)
    system_root = (
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / ".file-verification"
        if os.name == "nt"
        else Path("/etc/.file-verification")
    )
    unsafe_roots = (
        "",
        " ",
        ".",
        str(Path.cwd()),
        str(Path.cwd() / ".file-verification"),
        str(Path.cwd().anchor),
        str(Path.home()),
        str(Path.home() / ".file-verification"),
        str(Path.home().parent / ".file-verification"),
        str(system_root),
        "relative-cache",
    )
    for unsafe in unsafe_roots:
        with pytest.raises(ValueError, match="file verification cache_root"):
            pythia_mmap.verified_sha256(source, expected, unsafe)

    env_root = tmp_path / "relative-cache" / ".file-verification"
    monkeypatch.setenv(pythia_mmap.FILE_VERIFICATION_CACHE_ENV, "relative-cache")
    with pytest.raises(ValueError, match="cache_root must be absolute"):
        pythia_mmap.verified_sha256(source, expected)
    assert not env_root.exists()


def test_verified_sha256_accepts_a_safe_explicit_leaf(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"safe cache root")
    cache_root = tmp_path / "external-formal-cache" / ".file-verification"

    assert pythia_mmap.verified_sha256(source, _sha256(source), cache_root) == _sha256(source)
    assert len(_certificates(cache_root)) == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX path-shape validation")
def test_cache_root_allows_deep_formal_linux_path_but_rejects_broad_home_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "home",
        classmethod(lambda _cls: Path("/home/sophgo13")),
    )
    formal = Path(
        "/home/sophgo13/cjl/storage/pile/formal-cache/.file-verification"
    )
    assert pythia_mmap._verification_cache_root(formal) == formal

    for unsafe in (
        Path("/home"),
        Path("/home/user"),
        Path("/home/user/.file-verification"),
        Path("/etc/.file-verification"),
        Path("/root"),
        Path("/root/.file-verification"),
        Path("/root/data/formal-cache/.file-verification"),
    ):
        with pytest.raises(ValueError, match="file verification cache_root"):
            pythia_mmap._verification_cache_root(unsafe)


@pytest.mark.skipif(os.name == "nt", reason="POSIX protected-root validation")
def test_cache_root_rejects_inaccessible_protected_root_before_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = 0

    def fail_if_inspected(_path: Path) -> bool:
        nonlocal touched
        touched += 1
        raise PermissionError("protected root is not searchable")

    monkeypatch.setattr(Path, "is_symlink", fail_if_inspected)
    with pytest.raises(ValueError, match="file verification cache_root"):
        pythia_mmap._verification_cache_root(Path("/root/.file-verification"))
    assert touched == 0


def test_verified_sha256_disabled_by_default_still_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"disabled")
    monkeypatch.delenv(pythia_mmap.FILE_VERIFICATION_CACHE_ENV, raising=False)
    calls = 0
    original = pythia_mmap.sha256_file

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    assert pythia_mmap.verified_sha256(source, _sha256(source)) == _sha256(source)
    assert calls == 1


def _create_symlink(link: Path, target: Path) -> None:
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError) as error:
        pytest.skip(f"symlink creation is unavailable: {error}")


def test_verified_sha256_binds_logical_symlink_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"symlink target")
    logical = tmp_path / "logical.bin"
    _create_symlink(logical, target)
    cache_root = tmp_path / "external-formal-cache" / ".file-verification"
    expected = _sha256(target)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == expected
    real_hash = pythia_mmap.sha256_file

    def fail_if_hashed(_path: str | Path, **_kwargs: object) -> str:
        raise AssertionError("unchanged symlink should hit its certificate")

    monkeypatch.setattr(pythia_mmap, "sha256_file", fail_if_hashed)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == expected

    replacement = tmp_path / "replacement-link.bin"
    _create_symlink(replacement, target)
    logical.unlink()
    os.replace(replacement, logical)
    calls = 0

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return real_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == expected
    assert calls == 1

    monkeypatch.setattr(pythia_mmap, "sha256_file", fail_if_hashed)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == expected


def test_verified_sha256_rehashes_when_symlink_target_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_target = tmp_path / "first.bin"
    second_target = tmp_path / "second.bin"
    first_target.write_bytes(b"first target")
    second_target.write_bytes(b"second target")
    logical = tmp_path / "logical.bin"
    _create_symlink(logical, first_target)
    cache_root = tmp_path / "external-formal-cache" / ".file-verification"
    expected = _sha256(first_target)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == expected

    replacement = tmp_path / "replacement-link.bin"
    _create_symlink(replacement, second_target)
    logical.unlink()
    os.replace(replacement, logical)
    calls = 0
    original_hash = pythia_mmap.sha256_file

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    assert pythia_mmap.verified_sha256(logical, expected, cache_root) == _sha256(second_target)
    assert calls == 1


@pytest.mark.parametrize("mutation", ("mtime", "size", "content", "inode"))
def test_verified_sha256_stat_or_content_change_is_a_miss(
    tmp_path: Path, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"original")
    cache_root = tmp_path / "formal-cache" / ".file-verification"
    expected = _sha256(source)
    assert pythia_mmap.verified_sha256(source, expected, cache_root) == expected

    if mutation == "mtime":
        original_stat = source.stat()
        os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns + 1_000_000))
        expected_after = expected
    elif mutation == "size":
        source.write_bytes(b"replacement with a different size")
        expected_after = _sha256(source)
    elif mutation == "content":
        source.write_bytes(b"changed")
        expected_after = expected
    else:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(b"original")
        os.replace(replacement, source)
        expected_after = expected

    calls = 0
    original_hash = pythia_mmap.sha256_file

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    actual = pythia_mmap.verified_sha256(source, expected_after, cache_root)
    assert calls >= 1
    assert actual == _sha256(source)


def test_verified_sha256_corrupt_or_tampered_certificate_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"certificate")
    cache_root = tmp_path / "cache"
    expected = _sha256(source)
    pythia_mmap.verified_sha256(source, expected, cache_root)
    certificate_path = _certificates(cache_root)[0]
    certificate_path.write_text("{\"schema_version\": \"tampered\"}\n", encoding="utf-8")

    calls = 0
    original_hash = pythia_mmap.sha256_file

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    assert pythia_mmap.verified_sha256(source, expected, cache_root) == expected
    assert calls == 1


def test_verified_sha256_expected_change_uses_a_distinct_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"expected binding")
    cache_root = tmp_path / "cache"
    expected = _sha256(source)
    pythia_mmap.verified_sha256(source, expected, cache_root)

    calls = 0
    original_hash = pythia_mmap.sha256_file

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    wrong_expected = "0" * 64
    assert pythia_mmap.verified_sha256(source, wrong_expected, cache_root) == expected
    assert calls == 1


def test_verified_sha256_mismatch_does_not_write_certificate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"mismatch")
    cache_root = tmp_path / "cache"
    wrong_expected = "0" * 64

    assert pythia_mmap.verified_sha256(source, wrong_expected, cache_root) != wrong_expected
    assert not cache_root.exists()


def test_verified_sha256_concurrent_writers_publish_valid_atomic_certificate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"concurrent")
    cache_root = tmp_path / "cache"
    expected = _sha256(source)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _index: pythia_mmap.verified_sha256(source, expected, cache_root),
                range(16),
            )
        )
    assert results == [expected] * 16
    certificates = _certificates(cache_root)
    assert len(certificates) == 1
    assert json.loads(certificates[0].read_text(encoding="utf-8"))["actual_sha256"] == expected


def test_mmap_index_and_ordered_reader_share_certificate_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_path = tmp_path / "document.idx"
    index_path.write_bytes(
        pythia_mmap.MMAP_INDEX_MAGIC
        + struct.pack("<Q", 1)
        + struct.pack("<B", 8)
        + struct.pack("<QQ", 1, 1)
        + np.asarray([1], dtype="<i4").tobytes()
        + np.asarray([0], dtype="<i8").tobytes()
        + np.asarray([1], dtype="<i8").tobytes()
    )
    shard_path = tmp_path / "document-00000-of-00000.bin"
    shard_path.write_bytes(b"\x01\x00")
    descriptor = pythia_mmap.PythiaShardDescriptor(
        0, shard_path, shard_path.stat().st_size, _sha256(shard_path)
    )
    cache_root = tmp_path / "cache"
    index_expected = _sha256(index_path)
    original_hash = pythia_mmap.sha256_file
    calls = 0

    def count_hashes(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original_hash(path, **kwargs)

    monkeypatch.setattr(pythia_mmap, "sha256_file", count_hashes)
    with pythia_mmap.MMapIndex(index_path, expected_sha256=index_expected, cache_root=cache_root):
        reader = pythia_mmap.OrderedShardReader((descriptor,), cache_root=cache_root)
        assert reader.read_exact(0, 2) == b"\x01\x00"
    assert calls == 2

    def fail_if_hashed(_path: str | Path, **_kwargs: object) -> str:
        raise AssertionError("integration certificate hit unexpectedly hashed a file")

    monkeypatch.setattr(pythia_mmap, "sha256_file", fail_if_hashed)
    with pythia_mmap.MMapIndex(index_path, expected_sha256=index_expected, cache_root=cache_root):
        reader = pythia_mmap.OrderedShardReader((descriptor,), cache_root=cache_root)
        assert reader.read_exact(0, 2) == b"\x01\x00"
