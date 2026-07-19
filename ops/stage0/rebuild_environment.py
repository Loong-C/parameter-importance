from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import time
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from param_importance_nlp.atomic import (  # noqa: E402
    atomic_write_bytes,
    sha256_file,
    stable_json_bytes,
    stable_json_hash,
)
from param_importance_nlp.environment import (  # noqa: E402
    build_environment_manifest,
    collect_runtime_versions,
    compare_freeze_to_lock,
    compare_wheelhouse_to_manifest,
    parse_freeze_file,
    parse_requirements_lock,
    parse_wheelhouse_manifest,
)


NETWORK_MARKERS = ("AF_INET", "AF_INET6")
EXTERNAL_NETWORK_CALLS = ("connect(", "sendto(", "sendmsg(", "sendmmsg(")
CANDIDATE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
DRIVER_VERSION = re.compile(r"NVRM version:.*?\s([0-9]+(?:\.[0-9]+)+)\s+Release Build")
NVCC_RELEASE = re.compile(r"(?:release\s+|cuda_)([0-9]+\.[0-9]+)", re.IGNORECASE)
REQUIRED_DEPENDENCY_INPUTS = (
    "environment/base-requirements.in",
    "environment/requirements.in",
    "environment/linux-only-requirements.in",
)
CORE_DISTRIBUTION_FIELDS = {
    "torch": "torch",
    "transformers": "transformers",
    "datasets": "datasets",
    "accelerate": "accelerate",
    "tensorboard": "tensorboard",
    "cudnn_distribution": "nvidia-cudnn-cu12",
    "nccl_distribution": "nvidia-nccl-cu12",
}


def stable_hash(value: Any) -> str:
    """Compatibility wrapper used by focused tests."""

    return stable_json_hash(value)


def atomic_write(path: Path, payload: bytes) -> None:
    atomic_write_bytes(path, payload)


def parse_pins(path: Path) -> dict[str, str]:
    """Compatibility wrapper around the canonical strict lock parser."""

    return parse_requirements_lock(path).versions


def parse_wheel_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """Compatibility wrapper around the canonical ASCII TSV parser."""

    manifest = parse_wheelhouse_manifest(path)
    return {
        entry.filename: {"sha256": entry.sha256, "size": entry.size}
        for entry in manifest.entries
    }


def directory_size(path: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        current = Path(directory)
        directory_names[:] = [
            name for name in directory_names if not (current / name).is_symlink()
        ]
        for name in file_names:
            metadata = (current / name).lstat()
            if stat.S_ISREG(metadata.st_mode):
                total += metadata.st_size
    return total


def resolve_approved_directory(root: Path, relative: str) -> Path:
    approved_root = root.resolve(strict=True)
    candidate = approved_root / relative
    current = approved_root
    for part in Path(relative).parts:
        if part in {"", ".", ".."}:
            raise ValueError(f"Unsafe approved-directory component: {part!r}")
        current = current / part
        if current.is_symlink():
            raise RuntimeError(f"Approved directory may not use symlinks: {current}")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(approved_root)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def ensure_approved_subdirectory(root: Path, parent: Path, name: str) -> Path:
    approved_root = root.resolve(strict=True)
    resolved_parent = parent.resolve(strict=True)
    resolved_parent.relative_to(approved_root)
    if parent.is_symlink() or not resolved_parent.is_dir():
        raise RuntimeError(f"Unsafe manifest parent: {parent}")
    if not CANDIDATE_NAME.fullmatch(name):
        raise ValueError(f"Unsafe subdirectory name: {name!r}")
    candidate = resolved_parent / name
    try:
        candidate.mkdir()
    except FileExistsError:
        pass
    if candidate.is_symlink():
        raise RuntimeError(f"Manifest subdirectory may not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    resolved.relative_to(approved_root)
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def read_current_driver_version(path: Path = Path("/proc/driver/nvidia/version")) -> str:
    match = DRIVER_VERSION.search(path.read_text(encoding="utf-8", errors="strict"))
    if match is None:
        raise RuntimeError(f"Cannot parse current NVIDIA driver version from {path}")
    return match.group(1)


def parse_nvcc_release(output: str) -> str:
    matches = NVCC_RELEASE.findall(output)
    if not matches:
        raise RuntimeError("Cannot parse the system CUDA toolkit version from nvcc")
    if len(set(matches)) != 1:
        raise RuntimeError(f"nvcc reports inconsistent CUDA releases: {matches}")
    return matches[0]


def validate_direct_requirement_inputs(
    paths: list[Path], lock_snapshot: Any
) -> dict[str, Any]:
    """Prove every declared direct server dependency is satisfied by the lock."""

    exact_constraints: dict[str, str] = {}
    declarations: dict[str, list[dict[str, Any]]] = {}
    locked_indexes = {item.url for item in lock_snapshot.index_sources}
    for path in paths:
        source_declarations: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8", errors="strict").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            index_match = re.fullmatch(
                r"(?:--index-url|--extra-index-url)(?:=|\s+)(\S+)", line
            )
            if index_match is not None:
                url = index_match.group(1)
                if url not in locked_indexes:
                    raise ValueError(
                        f"{path}:{line_number}: index is absent from the lock: {url}"
                    )
                source_declarations.append({"kind": "index", "url": url})
                continue
            if line.startswith("-") or " @ " in line or "://" in line:
                raise ValueError(
                    f"{path}:{line_number}: unsupported direct requirement syntax"
                )
            try:
                requirement = Requirement(line)
            except InvalidRequirement as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid direct requirement"
                ) from error
            if requirement.marker is not None or requirement.url is not None:
                raise ValueError(
                    f"{path}:{line_number}: markers and direct URLs are forbidden"
                )
            normalized = str(canonicalize_name(requirement.name))
            locked_version = lock_snapshot.versions.get(normalized)
            if locked_version is None:
                raise ValueError(
                    f"{path}:{line_number}: direct dependency is absent from lock: "
                    f"{normalized}"
                )
            specifiers = list(requirement.specifier)
            if len(specifiers) > 1 or (
                specifiers and specifiers[0].operator != "=="
            ):
                raise ValueError(
                    f"{path}:{line_number}: only bare or exact direct dependencies "
                    "are allowed"
                )
            exact_version = specifiers[0].version if specifiers else None
            if exact_version is not None and Version(exact_version) != Version(
                locked_version
            ):
                raise ValueError(
                    f"{path}:{line_number}: direct pin {exact_version} conflicts with "
                    f"locked {locked_version}"
                )
            previous = exact_constraints.get(normalized)
            if previous is not None and exact_version is not None and Version(
                previous
            ) != Version(exact_version):
                raise ValueError(
                    f"Conflicting direct pins for {normalized}: "
                    f"{previous} != {exact_version}"
                )
            if exact_version is not None:
                exact_constraints[normalized] = exact_version
            source_declarations.append(
                {
                    "kind": "requirement",
                    "name": normalized,
                    "extras": sorted(requirement.extras),
                    "exact_version": exact_version,
                    "locked_version": locked_version,
                }
            )
        declarations[str(path)] = source_declarations
    return {
        "status": "PASS",
        "direct_distribution_count": len(
            {
                item["name"]
                for values in declarations.values()
                for item in values
                if item["kind"] == "requirement"
            }
        ),
        "declarations": declarations,
    }


def load_lock_provenance(
    path: Path,
    repo: Path,
    *,
    lock: Path,
    lock_snapshot: Any,
    dependency_inputs: dict[str, str],
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(repo)
    payload = json.loads(resolved.read_text(encoding="utf-8", errors="strict"))
    if payload.get("schema_version") != "stage0.lock-provenance.v1":
        raise ValueError("Lock provenance has an unexpected schema")
    lock_ref = str(lock.relative_to(repo)).replace("\\", "/")
    if payload.get("lock_path") != lock_ref:
        raise ValueError("Lock provenance references a different lock path")
    if payload.get("lock_sha256") != sha256_file(lock):
        raise ValueError("Lock provenance hash does not match the selected lock")
    if payload.get("distribution_count") != len(lock_snapshot.pins):
        raise ValueError("Lock provenance distribution count is stale")
    if payload.get("dependency_inputs") != dependency_inputs:
        raise ValueError("Lock provenance is not bound to the required input hashes")
    target = payload.get("target")
    required_target = {
        "implementation": "CPython",
        "python": "3.12",
        "abi": "cp312",
        "os": "Linux",
        "architecture": "x86_64",
    }
    if not isinstance(target, dict) or any(
        target.get(key) != value for key, value in required_target.items()
    ):
        raise ValueError("Lock provenance target does not match Linux CPython 3.12")
    runtime_expectations = payload.get("runtime_expectations")
    if not isinstance(runtime_expectations, dict):
        raise ValueError("Lock provenance lacks runtime expectations")
    for field, distribution in CORE_DISTRIBUTION_FIELDS.items():
        expected = runtime_expectations.get(field)
        locked = lock_snapshot.versions.get(distribution)
        if field in {"cudnn_distribution", "nccl_distribution"}:
            if expected != locked:
                raise ValueError(
                    f"Runtime expectation {field} is not bound to the lock"
                )
        elif expected is not None and expected != locked:
            raise ValueError(
                f"Runtime expectation {field} is not bound to the lock"
            )
    for required_field in (
        "python_implementation",
        "python_series",
        "torch_cuda_runtime",
        "cudnn_runtime",
        "nccl_runtime_validation",
    ):
        if not isinstance(runtime_expectations.get(required_field), str):
            raise ValueError(
                f"Lock provenance runtime expectation is missing: {required_field}"
            )
    return {
        "path": resolved,
        "ref": str(resolved.relative_to(repo)).replace("\\", "/"),
        "sha256": sha256_file(resolved),
        "runtime_expectations": runtime_expectations,
    }


def validate_core_runtime_versions(
    core_versions: dict[str, Any], lock_snapshot: Any, expectations: dict[str, str]
) -> dict[str, Any]:
    mismatches: dict[str, dict[str, Any]] = {}
    if core_versions.get("implementation") != expectations["python_implementation"]:
        mismatches["python_implementation"] = {
            "expected": expectations["python_implementation"],
            "observed": core_versions.get("implementation"),
        }
    python_version = str(core_versions.get("python", ""))
    if not python_version.startswith(f"{expectations['python_series']}."):
        mismatches["python_series"] = {
            "expected": expectations["python_series"],
            "observed": python_version,
        }
    for field, distribution in CORE_DISTRIBUTION_FIELDS.items():
        observed = core_versions.get(field)
        expected = lock_snapshot.versions[distribution]
        if observed != expected:
            mismatches[field] = {"expected": expected, "observed": observed}
    if core_versions.get("torch_cuda_runtime") != expectations["torch_cuda_runtime"]:
        mismatches["torch_cuda_runtime"] = {
            "expected": expectations["torch_cuda_runtime"],
            "observed": core_versions.get("torch_cuda_runtime"),
        }
    observed_cudnn = normalize_cudnn_version(core_versions.get("cudnn"))
    if observed_cudnn != expectations["cudnn_runtime"]:
        mismatches["cudnn_runtime"] = {
            "expected": expectations["cudnn_runtime"],
            "observed": observed_cudnn,
        }
    if expectations["nccl_runtime_validation"] != "DEFERRED_UNTIL_G0_G":
        mismatches["nccl_runtime_validation"] = {
            "expected": "DEFERRED_UNTIL_G0_G",
            "observed": expectations["nccl_runtime_validation"],
        }
    if mismatches:
        raise RuntimeError(f"Core runtime version policy mismatch: {mismatches}")
    return {
        "status": "PASS",
        "expectations": expectations,
        "observed": {
            **core_versions,
            "cudnn_runtime": observed_cudnn,
            "nccl_runtime": None,
        },
    }


def normalize_cudnn_version(value: int | str | None) -> str | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric <= 0:
        raise ValueError(f"Invalid cuDNN runtime version: {value!r}")
    if numeric >= 10_000:
        major = numeric // 10_000
        minor = (numeric % 10_000) // 100
    else:
        major = numeric // 1_000
        minor = (numeric % 1_000) // 100
    patch = numeric % 100
    return f"{major}.{minor}.{patch}"


def load_gpu_baseline(
    path: Path,
    repo: Path,
    *,
    expected_hostname: str,
    now: datetime | None = None,
    maximum_age_seconds: int = 86_400,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(repo)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage0.gate-report.v1":
        raise ValueError("GPU baseline has an unexpected schema")
    generated_text = payload.get("generated_at")
    if not isinstance(generated_text, str):
        raise ValueError("GPU baseline lacks generated_at")
    try:
        generated_at = datetime.fromisoformat(generated_text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("GPU baseline generated_at is invalid") from error
    if generated_at.tzinfo is None:
        raise ValueError("GPU baseline generated_at lacks a timezone")
    current_time = now or datetime.now(timezone.utc)
    age_seconds = (current_time - generated_at).total_seconds()
    if age_seconds < -300 or age_seconds > maximum_age_seconds:
        raise ValueError(f"GPU baseline age is outside the allowed window: {age_seconds}")
    subgates = payload.get("subgates")
    if not isinstance(subgates, dict) or subgates.get("G0-C", {}).get("status") != "PASS":
        raise ValueError("GPU baseline does not prove G0-C PASS")
    gpu_status = subgates.get("G0-G", {}).get("status")
    if gpu_status not in {"PASS", "BLOCKED"}:
        raise ValueError(f"Unexpected G0-G status: {gpu_status!r}")
    driver = subgates.get("G0-G", {}).get("evidence", {}).get("driver_version")
    toolkit = (
        payload.get("system_snapshot", {})
        .get("runtime_versions", {})
        .get("system_cuda_toolkit")
    )
    if not isinstance(driver, str) or not driver.strip():
        raise ValueError("GPU baseline lacks a driver version")
    if not isinstance(toolkit, str) or not toolkit.strip():
        raise ValueError("GPU baseline lacks a system CUDA toolkit version")
    overall_status = payload.get("status")
    expected_overall = "PASS" if gpu_status == "PASS" else "BLOCKED"
    if overall_status != expected_overall:
        raise ValueError(
            f"GPU baseline overall status conflicts with G0-G: {overall_status!r}"
        )
    hostname = payload.get("system_snapshot", {}).get("hostname")
    if hostname != expected_hostname:
        raise ValueError(
            f"GPU baseline hostname mismatch: {hostname!r} != {expected_hostname!r}"
        )
    return {
        "path": resolved,
        "ref": str(resolved.relative_to(repo)).replace("\\", "/"),
        "sha256": sha256_file(resolved),
        "g0_g_status": gpu_status,
        "overall_status": overall_status,
        "generated_at": generated_text,
        "age_seconds": age_seconds,
        "hostname": hostname,
        "driver_version": driver.strip(),
        "system_cuda_toolkit_version": toolkit.strip(),
    }


def wheel_entries_by_distribution(wheel_snapshot: Any) -> dict[str, list[Any]]:
    entries: dict[str, list[Any]] = {}
    for entry in wheel_snapshot.entries:
        distribution, _version, _build, _tags = parse_wheel_filename(entry.filename)
        entries.setdefault(str(canonicalize_name(distribution)), []).append(entry)
    return entries


def hashed_requirements_bytes(lock_snapshot: Any, wheel_snapshot: Any) -> bytes:
    by_distribution = wheel_entries_by_distribution(wheel_snapshot)
    lines: list[str] = []
    for pin in lock_snapshot.pins:
        candidates = []
        for entry in by_distribution.get(pin.normalized_name, []):
            _name, version, _build, _tags = parse_wheel_filename(entry.filename)
            if version == Version(pin.version):
                candidates.append(entry)
        if not candidates:
            raise ValueError(
                f"No manifest wheel matches {pin.normalized_name}=={pin.version}"
            )
        hashes = " ".join(
            f"--hash=sha256:{entry.sha256}"
            for entry in sorted(candidates, key=lambda item: item.filename)
        )
        lines.append(f"{pin.normalized_name}=={pin.version} {hashes}")
    return ("\n".join(lines) + "\n").encode("ascii")


def inventory_owned_tree(root: Path) -> list[dict[str, Any]]:
    if root.is_symlink() or not root.is_dir() or os.path.ismount(root):
        raise RuntimeError(f"Unsafe owned-tree root: {root}")
    entries: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in list(directory_names):
            path = current / name
            metadata = path.lstat()
            relative = str(path.relative_to(root)).replace("\\", "/")
            if stat.S_ISLNK(metadata.st_mode):
                entries.append({"path": relative, "kind": "symlink"})
                directory_names.remove(name)
            elif stat.S_ISDIR(metadata.st_mode):
                if os.path.ismount(path):
                    raise RuntimeError(f"Owned tree contains a mount point: {path}")
                entries.append({"path": relative, "kind": "directory"})
            else:
                raise RuntimeError(f"Unsafe object in owned tree: {path}")
        for name in file_names:
            path = current / name
            metadata = path.lstat()
            relative = str(path.relative_to(root)).replace("\\", "/")
            if stat.S_ISREG(metadata.st_mode):
                entries.append(
                    {"path": relative, "kind": "file", "size": metadata.st_size}
                )
            elif stat.S_ISLNK(metadata.st_mode):
                entries.append({"path": relative, "kind": "symlink"})
            else:
                raise RuntimeError(f"Unsafe object in owned tree: {path}")
    return sorted(entries, key=lambda item: (item["path"], item["kind"]))


def remove_exact_owned_tree(
    root: Path, *, owner_marker: dict[str, Any], expected: list[dict[str, Any]]
) -> None:
    marker = root / ".stage0-owner.json"
    if json.loads(marker.read_text(encoding="utf-8")) != owner_marker:
        raise RuntimeError("Owned-tree marker mismatch")
    observed = inventory_owned_tree(root)
    if observed != expected:
        raise RuntimeError("Owned-tree contents drifted after cleanup inventory")
    for item in sorted(
        expected,
        key=lambda value: (value["path"].count("/"), value["path"]),
        reverse=True,
    ):
        path = root / item["path"]
        path.relative_to(root)
        if item["kind"] in {"file", "symlink"}:
            path.unlink()
        else:
            path.rmdir()
    root.rmdir()


def evidence_records(root: Path) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or (path.exists() and not path.is_file() and not path.is_dir()):
            raise RuntimeError(f"Unsafe evidence object: {path}")
        if path.is_file():
            records.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return records


def acquire_advisory_lock(path: Path, *, allowed_root: Path) -> int:
    import fcntl

    root = allowed_root.resolve(strict=True)
    if path.parent.is_symlink():
        raise RuntimeError(f"Advisory-lock directory may not be a symlink: {path.parent}")
    parent = path.parent.resolve(strict=True)
    parent.relative_to(root)
    target = parent / path.name
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        raise RuntimeError("O_NOFOLLOW is required for the server advisory lock")
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(
        target,
        os.O_RDWR | os.O_CREAT | no_follow | close_on_exec,
        0o644,
    )
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError("Advisory lock must be a single-link regular file")
        if metadata.st_uid != os.geteuid():
            raise RuntimeError("Advisory lock must be owned by the current user")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.ftruncate(descriptor, 0)
        payload = stable_json_bytes(
            {
                "pid": os.getpid(),
                "acquired_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        os.write(descriptor, payload)
        os.fsync(descriptor)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def release_advisory_lock(descriptor: int | None) -> None:
    if descriptor is None:
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
        if os.name != "nt":
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def read_immutable(path: Path) -> bytes:
    """Read a single-link regular evidence file without following a symlink."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if no_follow is None:
        if os.name != "nt":
            raise RuntimeError("O_NOFOLLOW is required for immutable evidence")
        if path.is_symlink():
            raise RuntimeError(f"Immutable evidence may not be a symlink: {path}")
    else:
        flags |= no_follow
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(
                f"Immutable evidence must be a single-link regular file: {path}"
            )
        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
            raise RuntimeError(f"Immutable evidence has an unexpected owner: {path}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def run_logged(
    command: list[str],
    *,
    environment: dict[str, str],
    log_path: Path,
) -> subprocess.CompletedProcess[str]:
    started = time.monotonic()
    result = subprocess.run(
        command,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.monotonic() - started
    log_path.write_text(
        f"elapsed_seconds={elapsed:.6f}\nexit_code={result.returncode}\n{result.stdout}",
        encoding="utf-8",
    )
    return result


def external_network_attempts(trace_path: Path) -> list[str]:
    if not trace_path.is_file():
        return ["missing_trace"]
    return [
        line.rstrip()
        for line in trace_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if any(call in line for call in EXTERNAL_NETWORK_CALLS)
        and any(marker in line for marker in NETWORK_MARKERS)
    ]


def controlled_environment(data_root: Path, cache: Path, temporary: Path) -> dict[str, str]:
    isolated_home = cache / "home"
    isolated_hf = cache / "huggingface"
    isolated_xdg_cache = cache / "xdg-cache"
    isolated_xdg_config = cache / "xdg-config"
    isolated_xdg_data = cache / "xdg-data"
    for path in (
        isolated_home,
        isolated_hf,
        isolated_xdg_cache,
        isolated_xdg_config,
        isolated_xdg_data,
    ):
        path.mkdir()
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": str(isolated_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PIP_NO_INDEX": "1",
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "PIP_CACHE_DIR": str(cache),
        "PIP_REQUIRE_VIRTUALENV": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": str(isolated_hf),
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "XDG_CACHE_HOME": str(isolated_xdg_cache),
        "XDG_CONFIG_HOME": str(isolated_xdg_config),
        "XDG_DATA_HOME": str(isolated_xdg_data),
        "TMPDIR": str(temporary),
        "TMP": str(temporary),
        "TEMP": str(temporary),
    }


def create_venv(
    python: Path,
    target: Path,
    environment: dict[str, str],
    log: Path,
    trace: Path,
) -> list[str]:
    if target.exists():
        raise FileExistsError(target)
    result = run_logged(
        [
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=network",
            "-o",
            str(trace),
            "--",
            str(python),
            "-I",
            "-m",
            "venv",
            str(target),
        ],
        environment=environment,
        log_path=log,
    )
    attempts = external_network_attempts(trace)
    if attempts:
        raise RuntimeError(f"Environment creation attempted network access: {attempts[:5]}")
    if result.returncode != 0:
        raise RuntimeError(f"Environment creation failed; see {log}")
    return attempts


def venv_python(path: Path) -> Path:
    return path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def audited_pip_install(
    *,
    python: Path,
    lock: Path,
    wheelhouse: Path,
    environment: dict[str, str],
    log: Path,
    trace: Path,
    expected_success: bool,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    command = [
        "strace",
        "-f",
        "-qq",
        "-e",
        "trace=network",
        "-o",
        str(trace),
        "--",
        str(python),
        "-I",
        "-m",
        "pip",
        "install",
        "--no-index",
        "--require-hashes",
        "--only-binary=:all:",
        f"--find-links={wheelhouse}",
        "--requirement",
        str(lock),
    ]
    result = run_logged(
        command,
        environment=environment,
        log_path=log,
    )
    attempts = external_network_attempts(trace)
    if attempts:
        raise RuntimeError(f"pip attempted external network access: {attempts[:5]}")
    if expected_success and result.returncode != 0:
        raise RuntimeError(f"pip install failed ({result.returncode}); see {log}")
    if not expected_success and result.returncode == 0:
        raise RuntimeError(f"Negative pip install unexpectedly succeeded; see {log}")
    return result, attempts


def command_output(
    command: list[str], environment: dict[str, str], *, cwd: Path | None = None
) -> str:
    return subprocess.check_output(
        command, env=environment, cwd=cwd, text=True
    ).strip()


def audited_command_output(
    command: list[str],
    *,
    environment: dict[str, str],
    log: Path,
    trace: Path,
) -> tuple[str, list[str]]:
    result = run_logged(
        [
            "strace",
            "-f",
            "-qq",
            "-e",
            "trace=network",
            "-o",
            str(trace),
            "--",
            *command,
        ],
        environment=environment,
        log_path=log,
    )
    attempts = external_network_attempts(trace)
    if attempts:
        raise RuntimeError(f"Command attempted external network access: {attempts[:5]}")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")
    return result.stdout.strip(), attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--existing-env", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--lock-provenance", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--wheel-manifest", type=Path, required=True)
    parser.add_argument("--gpu-health-report", type=Path, required=True)
    parser.add_argument("--missing-distribution", default="pyyaml")
    arguments = parser.parse_args()

    repo = arguments.repo.resolve(strict=True)
    if repo != REPOSITORY_ROOT.resolve(strict=True):
        raise ValueError("--repo must identify the checkout containing this script")
    data_root = arguments.data_root.resolve(strict=True)
    existing_env = arguments.existing_env.resolve(strict=True)
    lock = arguments.lock.resolve(strict=True)
    lock_provenance_path = arguments.lock_provenance.resolve(strict=True)
    wheelhouse = arguments.wheelhouse.resolve(strict=True)
    wheel_manifest = arguments.wheel_manifest.resolve(strict=True)
    if not CANDIDATE_NAME.fullmatch(arguments.candidate_name):
        raise ValueError("--candidate-name must be one safe path component")
    candidate = (data_root / "envs" / arguments.candidate_name).resolve(strict=False)
    cache = (data_root / "cache" / f"pip-{arguments.candidate_name}").resolve(strict=False)
    temporary = (data_root / "tmp" / f"environment-{arguments.candidate_name}").resolve(
        strict=False
    )
    report_root = (data_root / "reports" / "stage0" / arguments.candidate_name).resolve(
        strict=False
    )
    for path in (candidate, cache, temporary, report_root):
        path.relative_to(data_root)
    existing_env.relative_to(data_root)
    lock.relative_to(repo)
    lock_provenance_path.relative_to(repo)
    wheelhouse.relative_to(data_root)
    wheel_manifest.relative_to(data_root)
    if candidate == existing_env:
        raise ValueError("Candidate environment must differ from the existing environment")
    dependency_input_paths = [
        (repo / relative).resolve(strict=True)
        for relative in REQUIRED_DEPENDENCY_INPUTS
    ]
    for path in dependency_input_paths:
        path.relative_to(repo)
    if len(set(dependency_input_paths)) != len(dependency_input_paths):
        raise ValueError("Dependency input paths must be unique")
    started_at = datetime.now(timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    report_root.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "stage0.environment-rebuild.v1",
        "status": "RUNNING",
        "started_at": started_at,
        "repo": str(repo),
        "lock": str(lock),
        "lock_provenance": str(lock_provenance_path),
        "wheelhouse": str(wheelhouse),
        "wheel_manifest": str(wheel_manifest),
        "candidate": str(candidate),
        "existing_environment": str(existing_env),
        "network_control": {
            "preventive_namespace": "UNAVAILABLE: unprivileged uid_map denied",
            "audit": "strace -f -e trace=network on every Python/pip child",
            "installer": "pip --no-index --require-hashes --only-binary",
        },
    }
    report_path = report_root / "environment-rebuild.json"
    lock_descriptor: int | None = None
    try:
        locks_root = resolve_approved_directory(data_root, "locks")
        manifests_root = resolve_approved_directory(data_root, "manifests")
        lock_descriptor = acquire_advisory_lock(
            locks_root / "stage0-environment-rebuild.lock",
            allowed_root=data_root,
        )
        if candidate.exists() or cache.exists() or temporary.exists():
            raise FileExistsError(
                "Candidate, cache, and temporary paths must all be new"
            )
        dependency_inputs = {
            str(path.relative_to(repo)).replace("\\", "/"): sha256_file(path)
            for path in dependency_input_paths
        }
        lock_snapshot = parse_requirements_lock(lock)
        lock_provenance = load_lock_provenance(
            lock_provenance_path,
            repo,
            lock=lock,
            lock_snapshot=lock_snapshot,
            dependency_inputs=dependency_inputs,
        )
        direct_requirement_audit = validate_direct_requirement_inputs(
            dependency_input_paths, lock_snapshot
        )
        gpu_report_path = arguments.gpu_health_report
        if not gpu_report_path.is_absolute():
            gpu_report_path = repo / gpu_report_path
        current_hostname = socket.gethostname()
        gpu_baseline = load_gpu_baseline(
            gpu_report_path,
            repo,
            expected_hostname=current_hostname,
        )
        current_driver = read_current_driver_version()
        nvcc_path = Path(
            f"/usr/local/cuda-{gpu_baseline['system_cuda_toolkit_version']}/bin/nvcc"
        ).resolve(strict=True)
        cache.mkdir(parents=False)
        temporary.mkdir(parents=False)
        environment = controlled_environment(data_root, cache, temporary)
        nvcc_output, nvcc_attempts = audited_command_output(
            [str(nvcc_path), "--version"],
            environment=environment,
            log=report_root / "current-nvcc-version.log",
            trace=report_root / "current-nvcc-version-network.trace",
        )
        current_toolkit = parse_nvcc_release(nvcc_output)
        if current_driver != gpu_baseline["driver_version"]:
            raise RuntimeError(
                "Current safe driver fact does not match the GPU baseline"
            )
        if current_toolkit != gpu_baseline["system_cuda_toolkit_version"]:
            raise RuntimeError(
                "Current nvcc release does not match the GPU baseline"
            )
        current_host_facts = {
            "hostname": current_hostname,
            "nvidia_driver_version": current_driver,
            "system_cuda_toolkit_version": current_toolkit,
            "nvcc_path": str(nvcc_path),
        }
        atomic_write(
            report_root / "current-safe-host-facts.json",
            stable_json_bytes(current_host_facts),
        )
        report.update(
            {
                "git_commit": command_output(
                    ["git", "rev-parse", "HEAD"], environment, cwd=repo
                ),
                "lock_sha256": sha256_file(lock),
                "lock_provenance_ref": lock_provenance["ref"],
                "lock_provenance_sha256": lock_provenance["sha256"],
                "wheel_manifest_sha256": sha256_file(wheel_manifest),
                "dependency_inputs": dependency_inputs,
                "direct_requirement_audit": direct_requirement_audit,
                "gpu_health_report": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in gpu_baseline.items()
                },
                "current_safe_host_facts": current_host_facts,
                "controlled_environment": dict(sorted(environment.items())),
            }
        )
        report["locked_distribution_count"] = len(lock_snapshot.pins)
        report["lock_semantic_sha256"] = lock_snapshot.semantic_sha256
        report["lock_index_sources"] = [
            source.as_dict() for source in lock_snapshot.index_sources
        ]
        old_freeze_before, old_freeze_before_attempts = audited_command_output(
            [
                str(venv_python(existing_env)),
                "-I",
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
            environment=environment,
            log=report_root / "existing-freeze-before.log",
            trace=report_root / "existing-freeze-before-network.trace",
        )
        old_freeze_before_path = report_root / "existing-freeze-before.txt"
        old_freeze_before_path.write_text(
            old_freeze_before + "\n", encoding="utf-8"
        )
        old_freeze_before_hash = hashlib.sha256(
            (old_freeze_before + "\n").encode("utf-8")
        ).hexdigest()
        old_snapshot = parse_freeze_file(old_freeze_before_path)
        old_comparison = compare_freeze_to_lock(lock_snapshot, old_snapshot)
        expected_bootstrap = tuple(
            sorted({"pip"} - set(lock_snapshot.versions))
        )
        if (
            old_comparison.missing
            or old_comparison.version_mismatches
            or old_comparison.extra != expected_bootstrap
        ):
            raise RuntimeError(
                f"Existing environment differs from lock/tool layer: {old_comparison.as_dict()}"
            )
        report["existing_environment_audit"] = {
            "freeze_all_sha256": old_freeze_before_hash,
            "comparison": old_comparison.as_dict(),
            "bootstrap_tools": {
                name: old_snapshot.versions[name] for name in expected_bootstrap
            },
            "external_network_attempt_count": len(old_freeze_before_attempts),
        }
        wheel_snapshot = parse_wheelhouse_manifest(wheel_manifest)
        wheel_comparison = compare_wheelhouse_to_manifest(
            wheelhouse,
            wheel_snapshot,
            lock=lock_snapshot,
            verify_hashes=True,
        )
        wheel_records = [entry.as_dict() for entry in wheel_snapshot.entries]
        report["wheelhouse_audit"] = {
            "count": len(wheel_records),
            "total_bytes": sum(item["size"] for item in wheel_records),
            "comparison": wheel_comparison.as_dict(),
            "manifest_semantic_sha256": wheel_snapshot.semantic_sha256,
            "records_digest": stable_hash(wheel_records),
        }
        if not wheel_comparison.ok:
            raise RuntimeError(
                f"Wheelhouse verification failed: {wheel_comparison.as_dict()}"
            )
        hashed_lock_path = report_root / "requirements-with-wheel-hashes.txt"
        atomic_write(
            hashed_lock_path,
            hashed_requirements_bytes(lock_snapshot, wheel_snapshot),
        )

        candidate_create_attempts = create_venv(
            venv_python(existing_env),
            candidate,
            environment,
            report_root / "candidate-create.log",
            report_root / "candidate-create-network.trace",
        )
        install, network_attempts = audited_pip_install(
            python=venv_python(candidate),
            lock=hashed_lock_path,
            wheelhouse=wheelhouse,
            environment=environment,
            log=report_root / "candidate-install.log",
            trace=report_root / "candidate-install-network.trace",
            expected_success=True,
        )
        freeze, freeze_attempts = audited_command_output(
            [str(venv_python(candidate)), "-I", "-m", "pip", "freeze"],
            environment=environment,
            log=report_root / "candidate-freeze.log",
            trace=report_root / "candidate-freeze-network.trace",
        )
        freeze_path = report_root / "pip-freeze.txt"
        freeze_path.write_text(freeze + "\n", encoding="utf-8")
        freeze_snapshot = parse_freeze_file(freeze_path)
        freeze_comparison = compare_freeze_to_lock(lock_snapshot, freeze_snapshot)
        report["freeze_lock_comparison"] = freeze_comparison.as_dict()
        if not freeze_comparison.ok:
            raise RuntimeError(f"Freeze mismatch: {freeze_comparison.as_dict()}")
        freeze_all, freeze_all_attempts = audited_command_output(
            [
                str(venv_python(candidate)),
                "-I",
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
            environment=environment,
            log=report_root / "candidate-freeze-all.log",
            trace=report_root / "candidate-freeze-all-network.trace",
        )
        freeze_all_path = report_root / "pip-freeze-all.txt"
        freeze_all_path.write_text(freeze_all + "\n", encoding="utf-8")
        freeze_all_snapshot = parse_freeze_file(freeze_all_path)
        freeze_all_comparison = compare_freeze_to_lock(
            lock_snapshot, freeze_all_snapshot
        )
        if (
            freeze_all_comparison.missing
            or freeze_all_comparison.version_mismatches
            or freeze_all_comparison.extra != expected_bootstrap
        ):
            raise RuntimeError(
                "Candidate full tool layer differs from lock: "
                f"{freeze_all_comparison.as_dict()}"
            )
        candidate_bootstrap = {
            name: freeze_all_snapshot.versions[name] for name in expected_bootstrap
        }
        existing_bootstrap = {
            name: old_snapshot.versions[name] for name in expected_bootstrap
        }
        if candidate_bootstrap != existing_bootstrap:
            raise RuntimeError(
                "Candidate bootstrap tool versions differ from the controller environment"
            )
        pip_check, pip_check_attempts = audited_command_output(
            [str(venv_python(candidate)), "-I", "-m", "pip", "check"],
            environment=environment,
            log=report_root / "pip-check.log",
            trace=report_root / "pip-check-network.trace",
        )
        (report_root / "pip-check.txt").write_text(pip_check + "\n", encoding="utf-8")

        core_script = (
            "import importlib.metadata as m,json,platform,site,sys;"
            "import torch,transformers,datasets,accelerate,tensorboard;"
            "print(json.dumps({"
            "'python':platform.python_version(),"
            "'implementation':platform.python_implementation(),"
            "'python_executable':sys.executable,"
            "'site_packages':site.getsitepackages(),"
            "'torch':torch.__version__,"
            "'torch_cuda_runtime':torch.version.cuda,"
            "'cudnn':torch.backends.cudnn.version(),"
            "'cudnn_distribution':m.version('nvidia-cudnn-cu12'),"
            "'nccl_distribution':m.version('nvidia-nccl-cu12'),"
            "'pip':m.version('pip'),"
            "'transformers':transformers.__version__,"
            "'datasets':datasets.__version__,"
            "'accelerate':accelerate.__version__,"
            "'tensorboard':tensorboard.__version__},sort_keys=True))"
        )
        core_output, core_network_attempts = audited_command_output(
            [str(venv_python(candidate)), "-I", "-c", core_script],
            environment=environment,
            log=report_root / "core-import.log",
            trace=report_root / "core-import-network.trace",
        )
        core_versions = json.loads(core_output)
        core_runtime_validation = validate_core_runtime_versions(
            core_versions,
            lock_snapshot,
            lock_provenance["runtime_expectations"],
        )
        report["core_runtime_validation"] = core_runtime_validation

        negative_root = temporary / "negative-missing-wheel"
        negative_wheels = negative_root / "wheelhouse"
        negative_env = negative_root / "venv"
        missing_distribution = str(canonicalize_name(arguments.missing_distribution))
        if missing_distribution not in lock_snapshot.versions:
            raise ValueError(
                f"Negative-test distribution is not locked: {missing_distribution}"
            )
        omitted_entries = wheel_entries_by_distribution(wheel_snapshot).get(
            missing_distribution, []
        )
        if not omitted_entries:
            raise RuntimeError(
                f"No wheel can be omitted for {missing_distribution}"
            )
        omitted_names = {entry.filename for entry in omitted_entries}
        negative_root.mkdir()
        owner_marker = {
            "schema_version": "stage0.owned-temporary.v1",
            "candidate_name": arguments.candidate_name,
            "started_at": started_at,
        }
        atomic_write(
            negative_root / ".stage0-owner.json",
            stable_json_bytes(owner_marker),
        )
        negative_wheels.mkdir()
        for entry in wheel_snapshot.entries:
            if entry.filename not in omitted_names:
                os.link(
                    wheelhouse / entry.filename,
                    negative_wheels / entry.filename,
                )
        negative_create_attempts = create_venv(
            venv_python(existing_env),
            negative_env,
            environment,
            report_root / "negative-create.log",
            report_root / "negative-create-network.trace",
        )
        negative_install, negative_network_attempts = audited_pip_install(
            python=venv_python(negative_env),
            lock=hashed_lock_path,
            wheelhouse=negative_wheels,
            environment=environment,
            log=report_root / "negative-install.log",
            trace=report_root / "negative-install-network.trace",
            expected_success=False,
        )
        negative_output = (negative_install.stdout or "").lower().replace("_", "-")
        missing_version = lock_snapshot.versions[missing_distribution].lower()
        resolver_failure_markers = (
            "could not find a version that satisfies the requirement",
            "no matching distribution found for",
        )
        if (
            missing_distribution not in negative_output
            or missing_version not in negative_output
            or not any(marker in negative_output for marker in resolver_failure_markers)
        ):
            raise RuntimeError(
                "Negative install did not fail for the exact omitted distribution pin"
            )

        old_freeze_after, old_freeze_after_attempts = audited_command_output(
            [
                str(venv_python(existing_env)),
                "-I",
                "-m",
                "pip",
                "freeze",
                "--all",
            ],
            environment=environment,
            log=report_root / "existing-freeze-after.log",
            trace=report_root / "existing-freeze-after-network.trace",
        )
        (report_root / "existing-freeze-after.txt").write_text(
            old_freeze_after + "\n", encoding="utf-8"
        )
        old_freeze_after_hash = hashlib.sha256(
            (old_freeze_after + "\n").encode("utf-8")
        ).hexdigest()
        if old_freeze_before_hash != old_freeze_after_hash:
            raise RuntimeError("Existing environment freeze changed during candidate rebuild")

        if sha256_file(lock) != report["lock_sha256"]:
            raise RuntimeError("Lock file changed during environment rebuild")
        if sha256_file(lock_provenance_path) != report["lock_provenance_sha256"]:
            raise RuntimeError("Lock provenance changed during environment rebuild")
        if sha256_file(wheel_manifest) != report["wheel_manifest_sha256"]:
            raise RuntimeError("Wheel manifest changed during environment rebuild")
        if any(
            sha256_file(path) != dependency_inputs[str(path.relative_to(repo)).replace("\\", "/")]
            for path in dependency_input_paths
        ):
            raise RuntimeError("A dependency input changed during environment rebuild")
        post_wheel_snapshot = parse_wheelhouse_manifest(wheel_manifest)
        post_wheel_comparison = compare_wheelhouse_to_manifest(
            wheelhouse,
            post_wheel_snapshot,
            lock=lock_snapshot,
            verify_hashes=True,
        )
        if (
            not post_wheel_comparison.ok
            or post_wheel_snapshot.semantic_sha256 != wheel_snapshot.semantic_sha256
        ):
            raise RuntimeError(
                "Wheelhouse changed or failed verification after installation: "
                f"{post_wheel_comparison.as_dict()}"
            )
        report["wheelhouse_post_install_audit"] = post_wheel_comparison.as_dict()

        negative_root.relative_to(temporary)
        cleanup_entries = inventory_owned_tree(negative_root)
        cleanup_inventory = {
            "schema_version": "stage0.cleanup-inventory.v1",
            "owner": owner_marker,
            "root": str(negative_root),
            "entries": cleanup_entries,
        }
        atomic_write(
            report_root / "negative-cleanup-inventory.json",
            stable_json_bytes(cleanup_inventory),
        )
        remove_exact_owned_tree(
            negative_root,
            owner_marker=owner_marker,
            expected=cleanup_entries,
        )
        if negative_root.exists():
            raise RuntimeError("Negative-test temporary directory was not removed")
        report["negative_temporary_cleaned"] = True
        residual_temporary = [str(path) for path in temporary.iterdir()]
        if residual_temporary:
            raise RuntimeError(
                f"Unexpected files remain in isolated temporary root: {residual_temporary}"
            )
        temporary.rmdir()
        report["temporary_root_cleaned"] = True

        runtime = collect_runtime_versions(
            nvidia_driver_version=gpu_baseline["driver_version"],
            system_cuda_toolkit_version=gpu_baseline[
                "system_cuda_toolkit_version"
            ],
            torch_version=core_versions["torch"],
            torch_cuda_runtime_version=core_versions["torch_cuda_runtime"],
            cudnn_version=normalize_cudnn_version(core_versions["cudnn"]),
            nccl_version=None,
        )
        if core_versions["python"] != runtime.python_version:
            raise RuntimeError(
                "Candidate and rebuild-controller Python versions do not match"
            )
        canonical_manifest = build_environment_manifest(
            role="server-cuda",
            environment_path=candidate,
            dependency_inputs=dependency_inputs,
            lock=lock_snapshot,
            freeze=freeze_snapshot,
            runtime=runtime,
            wheelhouse_manifest=wheel_snapshot,
            gpu_health_report_ref=gpu_baseline["ref"],
        )
        environment_id = str(canonical_manifest["environment_id"])
        network_attempt_counts = {
            "current_nvcc_version": len(nvcc_attempts),
            "existing_freeze_before": len(old_freeze_before_attempts),
            "candidate_create": len(candidate_create_attempts),
            "candidate_install": len(network_attempts),
            "candidate_freeze": len(freeze_attempts),
            "candidate_freeze_all": len(freeze_all_attempts),
            "pip_check": len(pip_check_attempts),
            "core_import": len(core_network_attempts),
            "negative_create": len(negative_create_attempts),
            "negative_install": len(negative_network_attempts),
            "existing_freeze_after": len(old_freeze_after_attempts),
        }
        if any(network_attempt_counts.values()):
            raise RuntimeError(
                f"External network attempts were recorded: {network_attempt_counts}"
            )
        evidence = evidence_records(report_root)
        evidence_digest = stable_hash(evidence)
        identity_document = {
            "schema_version": "stage0.environment-identity-manifest.v1",
            "environment_id": environment_id,
            "identity": canonical_manifest["identity"],
        }
        identity_root = ensure_approved_subdirectory(
            data_root, manifests_root, "environment-identities"
        )
        identity_path = identity_root / f"{environment_id}.json"
        identity_bytes = stable_json_bytes(identity_document)
        try:
            existing_identity_bytes = read_immutable(identity_path)
        except FileNotFoundError:
            write_immutable(identity_path, identity_bytes)
        else:
            if existing_identity_bytes != identity_bytes:
                raise RuntimeError(
                    f"Existing identity manifest conflicts with {environment_id}"
                )
        build_identity = {
            "environment_id": environment_id,
            "candidate": str(candidate),
            "started_at": started_at,
            "git_commit": report["git_commit"],
            "evidence_digest": evidence_digest,
        }
        build_id = f"env-build-v1-{stable_hash(build_identity)}"
        build_document = {
            "schema_version": "stage0.environment-build-observation.v1",
            "build_id": build_id,
            "environment_id": environment_id,
            "environment_identity_manifest": str(identity_path),
            "classification": "CPU_ONLY_CANDIDATE",
            "training_eligible": False,
            "g2_status": "BLOCKED",
            "pending_requirements": [
                "administrator-approved G0-G path B four-GPU allowlist",
                "post-clearance per-card CUDA import and health revalidation",
                "actual NCCL runtime version and four-GPU communication validation",
            ],
            "observations": {
                **canonical_manifest["observations"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": report["git_commit"],
                "report_path": str(report_path),
                "gpu_health_report_sha256": gpu_baseline["sha256"],
                "gpu_gate_status": gpu_baseline["g0_g_status"],
                "core_versions": core_versions,
                "core_runtime_validation": core_runtime_validation,
                "bootstrap_tools": candidate_bootstrap,
                "existing_environment_unchanged": True,
                "existing_environment_freeze_sha256": old_freeze_after_hash,
                "hashed_requirements_sha256": sha256_file(hashed_lock_path),
                "wheelhouse_post_install_verified": True,
                "network_control": report["network_control"],
                "external_network_attempt_counts": network_attempt_counts,
                "evidence_digest": evidence_digest,
                "evidence": evidence,
            },
        }
        build_root = ensure_approved_subdirectory(
            data_root, manifests_root, "environment-builds"
        )
        build_path = build_root / f"{build_id}.json"
        write_immutable(build_path, stable_json_bytes(build_document))
        recommendation = {
            "schema_version": "stage0.environment-recommendation.v1",
            "classification": "CPU_ONLY_CANDIDATE",
            "training_eligible": False,
            "g2_status": "BLOCKED",
            "environment_id": environment_id,
            "environment_identity_manifest": str(identity_path),
            "build_id": build_id,
            "build_observation_manifest": str(build_path),
            "path": str(candidate),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": report["git_commit"],
        }
        recommendation_path = manifests_root / "environment-cpu-candidate.json"
        report.update(
            {
                "status": "PASS",
                "classification": "CPU_ONLY_CANDIDATE",
                "training_eligible": False,
                "g2_status": "BLOCKED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "environment_id": environment_id,
                "environment_identity_manifest": str(identity_path),
                "build_id": build_id,
                "build_observation_manifest": str(build_path),
                "cpu_candidate_reference": str(recommendation_path),
                "candidate_install_exit_code": install.returncode,
                "candidate_create_external_network_attempt_count": len(
                    candidate_create_attempts
                ),
                "candidate_external_network_attempt_count": len(network_attempts),
                "core_import_external_network_attempt_count": len(
                    core_network_attempts
                ),
                "negative_omitted_distribution": missing_distribution,
                "negative_omitted_version": missing_version,
                "negative_omitted_wheels": sorted(omitted_names),
                "negative_install_exit_code": negative_install.returncode,
                "negative_create_external_network_attempt_count": len(
                    negative_create_attempts
                ),
                "negative_external_network_attempt_count": len(
                    negative_network_attempts
                ),
                "freeze_sha256": sha256_file(freeze_path),
                "freeze_all_sha256": sha256_file(freeze_all_path),
                "freeze_matches_lock": True,
                "bootstrap_tools": candidate_bootstrap,
                "pip_check": pip_check,
                "core_versions": core_versions,
                "core_runtime_validation": core_runtime_validation,
                "runtime_versions": runtime.as_dict(),
                "candidate_size_bytes": directory_size(candidate),
                "isolated_cache_size_bytes": directory_size(cache),
                "existing_environment_unchanged": True,
                "existing_environment_freeze_sha256": old_freeze_after_hash,
                "external_network_attempt_counts": network_attempt_counts,
                "evidence_digest": evidence_digest,
            }
        )
        atomic_write(report_path, stable_json_bytes(report))
        atomic_write(recommendation_path, stable_json_bytes(recommendation))
        try:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        except BrokenPipeError:
            pass
        return 0
    except BaseException as error:
        report.update(
            {
                "status": "FAIL",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "error_type": type(error).__name__,
                "error": str(error),
                "candidate_retained_for_diagnosis": candidate.exists(),
                "cache_retained_for_diagnosis": cache.exists(),
                "temporary_retained_for_diagnosis": temporary.exists(),
            }
        )
        atomic_write(report_path, stable_json_bytes(report))
        try:
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        except BrokenPipeError:
            pass
        return 1
    finally:
        release_advisory_lock(lock_descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
