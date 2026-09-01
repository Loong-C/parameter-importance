"""Targeted tests for the non-critical Stage 3 verification shim."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import param_importance_nlp.data.pythia_mmap as pythia_mmap
from ops.stage3 import file_verification_cache as cache
from ops.stage3.install_file_verification_cache import install_startup_hook

QUALIFIED = "f32daa2a6c45c08730444df9177388daa39e3787"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _certificates(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def _subprocess(code: str, *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=cwd, env=env, text=True, capture_output=True, check=False,
    )


def test_all_g3_critical_refs_are_qualified() -> None:
    repository = Path(__file__).resolve().parents[1]
    refs = (
        "configs/stage0/g3-asset-requirements-v1.json",
        "configs/stage0/g3-asset-layout-v1.json",
        "configs/stage0/g3-download-plan-v1.json",
        "ops/stage0/attest_g3_materialization.py",
        "ops/stage0/materialize_and_publish_g3.py",
        "ops/stage0/verify_g3_assets.py",
        "schemas/stage0/asset-layout-v1.json",
        "schemas/stage0/asset-requirements-v1.json",
        "schemas/stage0/download-plan-v1.json",
        "schemas/stage0-asset-manifest-v1.json",
        "schemas/stage0-g3-acquisition-report-v1.json",
        "schemas/stage0-g3-verify-only-report-v1.json",
        "src/param_importance_nlp/asset_acquisition.py",
        "src/param_importance_nlp/asset_download_plan.py",
        "src/param_importance_nlp/asset_layout.py",
        "src/param_importance_nlp/asset_requirements.py",
        "src/param_importance_nlp/assets.py",
        "src/param_importance_nlp/atomic.py",
        "src/param_importance_nlp/contracts/__init__.py",
        "src/param_importance_nlp/contracts/jsonio.py",
        "src/param_importance_nlp/data/pythia_mmap.py",
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
        "src/param_importance_nlp/g3_asset_publication.py",
        "src/param_importance_nlp/g3_gate.py",
        "src/param_importance_nlp/g3_lifecycle_evidence.py",
        "src/param_importance_nlp/g3_semantic_evidence.py",
        "src/param_importance_nlp/glue_builder.py",
        "src/param_importance_nlp/providers/optional.py",
        "src/param_importance_nlp/runtime/task_artifacts.py",
    )
    assert subprocess.run(
        ["git", "diff", "--quiet", QUALIFIED, "--", *refs],
        cwd=repository, check=False,
    ).returncode == 0


def test_disabled_by_default_does_not_patch(tmp_path: Path) -> None:
    hook_dir = tmp_path / "hook"
    install_startup_hook(hook_dir)
    env = os.environ.copy()
    env.pop(cache.FILE_VERIFICATION_CACHE_ENV, None)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(hook_dir), str(Path(__file__).resolve().parents[1] / "src"))
    )
    result = _subprocess(
        "import param_importance_nlp.data.pythia_mmap as p; "
        "print(getattr(p, '_stage3_file_verification_cache_installed', False))",
        env=env, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_enabled_startup_hook_patches_only_pythia(tmp_path: Path) -> None:
    hook_dir = tmp_path / "hook"
    install_startup_hook(hook_dir)
    root = tmp_path / "formal-run" / ".file-verification"
    env = os.environ.copy()
    env[cache.FILE_VERIFICATION_CACHE_ENV] = str(root)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(hook_dir), str(Path(__file__).resolve().parents[1] / "src"))
    )
    result = _subprocess(
        "import param_importance_nlp.data.pythia_mmap as p; "
        "print(getattr(p, '_stage3_file_verification_cache_installed', False)); "
        "print(p.sha256_file.__module__)",
        env=env, cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["True", "param_importance_nlp.data.pythia_mmap"]
    manifest = json.loads((hook_dir / "sitecustomize.manifest.json").read_text())
    assert manifest["installer_identity"] == cache.INSTALLER_IDENTITY
    assert manifest["hook_sha256"] == _sha(hook_dir / "sitecustomize.py")


def test_cross_process_hit_does_not_call_underlying_hash(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"cross-process")
    root = tmp_path / "formal-run" / ".file-verification"
    expected = _sha(source)
    repository = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repository)
    first = _subprocess(
        f"from ops.stage3.file_verification_cache import verified_sha256; "
        f"print(verified_sha256({str(source)!r}, {expected!r}, {str(root)!r}))",
        env=env, cwd=repository,
    )
    assert first.returncode == 0, first.stderr
    second = _subprocess(
        f"import hashlib; "
        f"hashlib.sha256 = lambda *a, **k: (_ for _ in ()).throw(AssertionError('hashed')); "
        f"from ops.stage3.file_verification_cache import verified_sha256; "
        f"print(verified_sha256({str(source)!r}, {expected!r}, {str(root)!r}))",
        env=env, cwd=repository,
    )
    assert second.returncode == 0, second.stderr
    assert second.stdout.strip() == expected


def test_stat_content_and_certificate_tamper_are_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"original")
    root = tmp_path / "formal-run" / ".file-verification"
    expected = _sha(source)
    assert cache.verified_sha256(source, expected, root) == expected
    certificate = _certificates(root)[0]
    certificate.write_text("tampered", encoding="utf-8")
    calls = 0
    original = cache._stream_sha256

    def counted(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(path, **kwargs)

    monkeypatch.setattr(cache, "_stream_sha256", counted)
    assert cache.verified_sha256(source, expected, root) == expected
    assert calls == 1
    source.write_bytes(b"changed")
    assert cache.verified_sha256(source, _sha(source), root) == _sha(source)
    assert calls == 2


def test_symlink_identity_is_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"target")
    logical = tmp_path / "logical.bin"
    try:
        logical.symlink_to(target)
    except (OSError, NotImplementedError) as error:
        pytest.skip(str(error))
    root = tmp_path / "formal-run" / ".file-verification"
    expected = _sha(target)
    assert cache.verified_sha256(logical, expected, root) == expected
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"new target")
    logical.unlink()
    logical.symlink_to(replacement)
    calls = 0
    original = cache._stream_sha256

    def counted(path: str | Path, **kwargs: object) -> str:
        nonlocal calls
        calls += 1
        return original(path, **kwargs)

    monkeypatch.setattr(cache, "_stream_sha256", counted)
    assert cache.verified_sha256(logical, _sha(replacement), root) == _sha(replacement)
    assert calls == 1


def test_expected_mismatch_does_not_publish_certificate(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"mismatch")
    root = tmp_path / "formal-run" / ".file-verification"
    assert cache.verified_sha256(source, "0" * 64, root) != "0" * 64
    assert not root.exists()


def test_unsafe_roots_are_rejected_without_creation(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"root safety")
    expected = _sha(source)
    unsafe = ("", " ", ".", str(Path.cwd()), str(Path.cwd() / ".file-verification"),
              str(Path.cwd().anchor), str(Path.home()), str(Path.home() / ".file-verification"),
              "relative-cache")
    for value in unsafe:
        with pytest.raises(ValueError, match="file verification cache_root"):
            cache.verified_sha256(source, expected, value)
    env_root = tmp_path / "relative-cache" / ".file-verification"
    os.environ[cache.FILE_VERIFICATION_CACHE_ENV] = "relative-cache"
    try:
        with pytest.raises(ValueError, match="absolute"):
            cache.verified_sha256(source, expected)
    finally:
        os.environ.pop(cache.FILE_VERIFICATION_CACHE_ENV, None)
    assert not env_root.exists()


def test_concurrent_writers_are_atomic(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"concurrent")
    root = tmp_path / "formal-run" / ".file-verification"
    expected = _sha(source)
    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(
            lambda _index: cache.verified_sha256(source, expected, root), range(16)
        ))
    assert values == [expected] * 16
    assert len(_certificates(root)) == 1


def test_pythia_expected_mismatch_keeps_original_error_and_no_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "document.idx"
    source.write_bytes(b"not an index")
    root = tmp_path / "formal-run" / ".file-verification"
    monkeypatch.setenv(cache.FILE_VERIFICATION_CACHE_ENV, str(root))
    assert cache.install_file_verification_cache(pythia_mmap)
    with pytest.raises(pythia_mmap.PythiaDataError, match="SHA-256 mismatch"):
        pythia_mmap.MMapIndex(source, expected_sha256="0" * 64)
    assert not root.exists()
