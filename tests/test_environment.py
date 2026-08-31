from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from param_importance_nlp.environment import (
    DependencyFormatError,
    RuntimeVersions,
    build_environment_manifest,
    collect_runtime_versions,
    compare_freeze_to_lock,
    compare_wheelhouse_to_manifest,
    compute_environment_id,
    parse_freeze_text,
    parse_requirements_lock,
    parse_requirements_lock_text,
    parse_wheelhouse_manifest,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _runtime(**changes: str | None) -> RuntimeVersions:
    values: dict[str, str | None] = {
        "os_name": "Linux",
        "os_version": "Ubuntu 24.04",
        "kernel_release": "6.8.0",
        "architecture": "x86_64",
        "libc_name": "glibc",
        "libc_version": "2.39",
        "python_implementation": "CPython",
        "python_version": "3.12.3",
        "nvidia_driver_version": "575.57.08",
        "system_cuda_toolkit_version": "12.9",
        "torch_version": "2.12.1+cu126",
        "torch_cuda_runtime_version": "12.6",
        "cudnn_version": "9.10.2",
        "nccl_version": "2.29.3",
    }
    values.update(changes)
    return RuntimeVersions(**values)  # type: ignore[arg-type]


def _manifest_line(filename: str, payload: bytes) -> str:
    return f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{filename}\n"


def test_repository_server_lock_is_strict_and_exact() -> None:
    lock = parse_requirements_lock(REPOSITORY_ROOT / "environment" / "requirements.lock")
    assert len(lock.pins) == 89
    assert lock.versions["torch"] == "2.12.1+cu126"
    assert [source.option for source in lock.index_sources] == [
        "--extra-index-url",
        "--extra-index-url",
    ]
    assert lock.source_sha256 != lock.semantic_sha256


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("numpy>=2", "exact name==version"),
        ("numpy==2.*", "exact name==version"),
        ("numpy==2; sys_platform == 'win32'", "markers are forbidden"),
        ("-e .", "editable requirements are forbidden"),
        ("git+https://example.invalid/repo.git", "VCS requirements are forbidden"),
        ("demo @ https://example.invalid/demo.whl", "direct URL requirements"),
        ("--find-links /tmp/wheels", "unsupported pip option"),
    ],
)
def test_lock_rejects_ambiguous_or_unrecorded_sources(line: str, message: str) -> None:
    with pytest.raises(DependencyFormatError, match=message):
        parse_requirements_lock_text(line + "\n")


def test_lock_rejects_duplicate_normalized_names_and_index_credentials() -> None:
    with pytest.raises(DependencyFormatError, match="duplicate distribution"):
        parse_requirements_lock_text("typing_extensions==1\ntyping-extensions==1\n")
    with pytest.raises(DependencyFormatError, match="credentials are forbidden"):
        parse_requirements_lock_text(
            "--index-url https://user:secret@example.invalid/simple\nnumpy==2.5.1\n"
        )


@pytest.mark.parametrize(
    "line",
    [
        "-e file:///tmp/project",
        "demo @ file:///tmp/demo.whl",
        "demo @ git+https://example.invalid/demo.git",
    ],
)
def test_freeze_rejects_editable_vcs_and_direct_url(line: str) -> None:
    with pytest.raises(DependencyFormatError):
        parse_freeze_text(line + "\n")


def test_freeze_lock_comparison_reports_every_drift_class() -> None:
    lock = parse_requirements_lock_text("alpha==1\nbeta==2\ngamma==3\n")
    freeze = parse_freeze_text("alpha==1\nbeta==9\ndelta==4\n")
    comparison = compare_freeze_to_lock(lock, freeze)
    assert comparison.ok is False
    assert comparison.missing == ("gamma",)
    assert comparison.extra == ("delta",)
    assert [item.as_dict() for item in comparison.version_mismatches] == [
        {"name": "beta", "locked": "2", "installed": "9"}
    ]


def test_wheelhouse_manifest_matches_files_and_lock(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    alpha = b"alpha-wheel"
    beta = b"beta-wheel"
    (wheelhouse / "alpha-1-py3-none-any.whl").write_bytes(alpha)
    (wheelhouse / "typing_extensions-2-py3-none-any.whl").write_bytes(beta)
    manifest_path = tmp_path / "wheelhouse-sha256.tsv"
    manifest_path.write_text(
        _manifest_line("alpha-1-py3-none-any.whl", alpha)
        + _manifest_line("typing_extensions-2-py3-none-any.whl", beta),
        encoding="ascii",
    )
    manifest = parse_wheelhouse_manifest(manifest_path)
    lock = parse_requirements_lock_text("alpha==1\ntyping-extensions==2\n")
    comparison = compare_wheelhouse_to_manifest(wheelhouse, manifest, lock=lock)
    assert comparison.ok is True
    assert comparison.as_dict()["uncovered_requirements"] == []


def test_wheelhouse_comparison_reports_set_size_hash_and_coverage(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    expected_payload = b"expected"
    actual_payload = b"tampered-payload"
    (wheelhouse / "alpha-1-py3-none-any.whl").write_bytes(actual_payload)
    (wheelhouse / "extra-1-py3-none-any.whl").write_bytes(b"extra")
    manifest_path = tmp_path / "wheelhouse-sha256.tsv"
    manifest_path.write_text(
        _manifest_line("alpha-1-py3-none-any.whl", expected_payload)
        + _manifest_line("orphan-1-py3-none-any.whl", b"missing"),
        encoding="ascii",
    )
    manifest = parse_wheelhouse_manifest(manifest_path)
    lock = parse_requirements_lock_text("alpha==1\nbeta==2\n")
    comparison = compare_wheelhouse_to_manifest(wheelhouse, manifest, lock=lock)
    assert comparison.ok is False
    assert comparison.missing_files == ("orphan-1-py3-none-any.whl",)
    assert comparison.unexpected_files == ("extra-1-py3-none-any.whl",)
    assert [item.filename for item in comparison.size_mismatches] == [
        "alpha-1-py3-none-any.whl"
    ]
    assert [item.filename for item in comparison.sha256_mismatches] == [
        "alpha-1-py3-none-any.whl"
    ]
    assert comparison.uncovered_requirements == ("beta==2",)
    assert comparison.unmatched_manifest_files == ("orphan-1-py3-none-any.whl",)


@pytest.mark.parametrize(
    "filename",
    [
        "torch-2.12.1+cu126-cp312-cp312-manylinux_2_28_x86_64.whl",
        "hf_xet-1.5.1-cp37-abi3-manylinux2014_x86_64.whl",
        "nvidia_nccl_cu12-2.29.3-py3-none-manylinux_2_18_x86_64.whl",
        "colorama-0.4.6-py2.py3-none-any.whl",
    ],
)
def test_linux_cp312_wheel_tags_are_accepted(
    tmp_path: Path, filename: str
) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    payload = b"fixture"
    (wheelhouse / filename).write_bytes(payload)
    distribution, version = filename.split("-", 2)[:2]
    lock = parse_requirements_lock_text(
        f"{distribution.replace('_', '-')}=={version}\n"
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{filename}\n",
        encoding="ascii",
    )

    comparison = compare_wheelhouse_to_manifest(
        wheelhouse, parse_wheelhouse_manifest(manifest_path), lock=lock
    )

    assert comparison.ok
    assert comparison.incompatible_manifest_files == ()


def test_windows_wheel_is_rejected_for_linux_cp312_target(tmp_path: Path) -> None:
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    filename = "alpha-1-cp312-cp312-win_amd64.whl"
    payload = b"fixture"
    (wheelhouse / filename).write_bytes(payload)
    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        f"{hashlib.sha256(payload).hexdigest()}\t{len(payload)}\t{filename}\n",
        encoding="ascii",
    )

    comparison = compare_wheelhouse_to_manifest(
        wheelhouse,
        parse_wheelhouse_manifest(manifest_path),
        lock=parse_requirements_lock_text("alpha==1\n"),
    )

    assert not comparison.ok
    assert comparison.incompatible_manifest_files == (filename,)
    assert comparison.uncovered_requirements == ("alpha==1",)


def test_environment_id_is_semantic_and_order_independent() -> None:
    left_lock = parse_requirements_lock_text("alpha==1\nbeta==2\n")
    right_lock = parse_requirements_lock_text("# reordered\nbeta==2\nalpha==1\n")
    left_freeze = parse_freeze_text("beta==2\nalpha==1\n")
    right_freeze = parse_freeze_text("alpha==1\nbeta==2\n")
    inputs_left = {"base-requirements.in": "a" * 64, "requirements.in": "b" * 64}
    inputs_right = {"requirements.in": "b" * 64, "base-requirements.in": "a" * 64}
    left = compute_environment_id(
        role="server-cuda",
        dependency_inputs=inputs_left,
        lock=left_lock,
        freeze=left_freeze,
        runtime=_runtime(),
    )
    right = compute_environment_id(
        role="server-cuda",
        dependency_inputs=inputs_right,
        lock=right_lock,
        freeze=right_freeze,
        runtime=_runtime(),
    )
    assert left == right
    assert left.startswith("env-v1-") and len(left) == len("env-v1-") + 64


def test_environment_id_changes_when_a_runtime_layer_changes() -> None:
    lock = parse_requirements_lock_text("alpha==1\n")
    freeze = parse_freeze_text("alpha==1\n")
    common = {
        "role": "server-cuda",
        "dependency_inputs": {"requirements.in": "a" * 64},
        "lock": lock,
        "freeze": freeze,
    }
    baseline = compute_environment_id(runtime=_runtime(), **common)
    toolkit_changed = compute_environment_id(
        runtime=_runtime(system_cuda_toolkit_version="13.0"), **common
    )
    torch_runtime_changed = compute_environment_id(
        runtime=_runtime(torch_cuda_runtime_version="12.7"), **common
    )
    assert baseline != toolkit_changed
    assert baseline != torch_runtime_changed
    assert toolkit_changed != torch_runtime_changed


def test_manifest_keeps_path_and_gpu_health_out_of_environment_id(tmp_path: Path) -> None:
    lock = parse_requirements_lock_text("alpha==1\n")
    freeze = parse_freeze_text("alpha==1\n")
    common = {
        "role": "server-cuda",
        "dependency_inputs": {"requirements.in": "a" * 64},
        "lock": lock,
        "freeze": freeze,
        "runtime": _runtime(),
    }
    first = build_environment_manifest(
        environment_path=tmp_path / "env-a",
        gpu_health_report_ref="gpu-health-a.json",
        **common,
    )
    second = build_environment_manifest(
        environment_path=tmp_path / "env-b",
        gpu_health_report_ref="gpu-health-b.json",
        **common,
    )
    assert first["environment_id"] == second["environment_id"]
    assert first["observations"]["environment_path_sha256"] != second["observations"][
        "environment_path_sha256"
    ]


def test_runtime_versions_separate_driver_toolkit_and_torch_runtime() -> None:
    runtime = collect_runtime_versions(
        nvidia_driver_version="575.57.08",
        system_cuda_toolkit_version="12.9",
        torch_version="2.12.1+cu126",
        torch_cuda_runtime_version="12.6",
        cudnn_version="9.10.2",
        nccl_version="2.29.3",
    )
    payload = runtime.as_dict()
    assert payload["nvidia_driver_version"] == "575.57.08"
    assert payload["system_cuda_toolkit_version"] == "12.9"
    assert payload["torch_cuda_runtime_version"] == "12.6"
    assert "cuda_version" not in payload
