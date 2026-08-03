#!/usr/bin/env python3
"""Promote the rebuilt Stage 0 environment after exact-four-GPU qualification.

The offline rebuild deliberately publishes only a CPU candidate while G0-G is
blocked.  This command validates the root-owned service-finalization evidence,
the unprivileged CUDA/NCCL smoke evidence, and the current live host before it
atomically publishes the training-eligible environment reference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised by Windows collection
    fcntl = None  # type: ignore[assignment]


EXPECTED_HOST = "sophgo13"
EXPECTED_DRIVER = "575.57.08"
EXPECTED_BDFS = [
    "00000000:9C:00.0",
    "00000000:9D:00.0",
    "00000000:A0:00.0",
    "00000000:A4:00.0",
]
EXPECTED_UUIDS = [
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4",
]
EXPECTED_EXCLUDED_UUIDS = [
    "GPU-6ff7389b-eaf8-aefd-b2c6-1611be41fa5d",
    "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c",
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-d0ce0b43-7e46-6bca-b078-5aa7043928d7",
]
ADMIN_PREFIX = Path(
    "/var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize"
)
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
FULL_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class PromotionError(RuntimeError):
    """A fail-closed promotion validation error."""


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PromotionError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise PromotionError(f"JSON root must be an object: {path}")
    return value


def parse_marker(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PromotionError(f"cannot read marker {path}: {error}") from error
    for line in lines:
        if not line or "=" not in line:
            raise PromotionError(f"invalid marker line in {path}: {line!r}")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise PromotionError(f"duplicate/empty marker key in {path}: {key!r}")
        values[key] = value
    return values


def require_directory(path: Path, *, within: Path | None = None) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PromotionError(f"required directory is unavailable: {path}: {error}") from error
    if not resolved.is_dir() or path.is_symlink():
        raise PromotionError(f"required path is not a real directory: {path}")
    if within is not None:
        try:
            resolved.relative_to(within)
        except ValueError as error:
            raise PromotionError(f"directory escapes approved root: {resolved}") from error
    return resolved


def verify_sha256sums(root: Path, required: set[str]) -> dict[str, str]:
    manifest = root / "SHA256SUMS"
    records: dict[str, str] = {}
    try:
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PromotionError(f"cannot read checksum manifest {manifest}: {error}") from error
    for line in lines:
        match = SHA_LINE.fullmatch(line)
        if not match:
            raise PromotionError(f"invalid checksum line in {manifest}: {line!r}")
        expected, name = match.groups()
        candidate = Path(name)
        if candidate.is_absolute() or len(candidate.parts) != 1 or name in records:
            raise PromotionError(f"unsafe or duplicate checksum target: {name!r}")
        target = root / name
        if target.is_symlink() or not target.is_file():
            raise PromotionError(f"checksum target is not a regular file: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise PromotionError(f"checksum mismatch: {target}: {actual} != {expected}")
        records[name] = actual
    missing = required - set(records)
    if missing:
        raise PromotionError(f"checksum manifest is missing required files: {sorted(missing)}")
    return records


def normalize_uuid(value: Any) -> str:
    text = str(value or "")
    if text.startswith("GPU-"):
        text = text[4:]
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", text):
        raise PromotionError(f"invalid GPU UUID: {value!r}")
    return "GPU-" + text.lower()


def validate_admin_evidence(root: Path, boot_id: str) -> dict[str, Any]:
    required = {
        "administrator-response-final.json",
        "excluded-gpus-release.txt",
        "gpu-clients-release.json",
        "kernel-delta-release.txt",
        "module-params-release.txt",
        "nvml-release.csv",
        "pytorch-release.json",
        "row-remapper-release.txt",
    }
    sums = verify_sha256sums(root, required)
    marker = parse_marker(root / "SUCCESS")
    if marker.get("status") != "G0_G_UUID_EXCLUSION_SERVICE_FINALIZE_PASS":
        raise PromotionError("administrator finalizer did not publish PASS")
    if marker.get("boot_id") != boot_id or Path(marker.get("evidence", "")) != root:
        raise PromotionError("administrator marker does not bind the current boot/evidence")
    if marker.get("fabric_manager") not in {"ACTIVE", "NOT_APPLICABLE_MASKED"}:
        raise PromotionError("Fabric Manager outcome is not accepted")
    response = load_json(root / "administrator-response-final.json")
    if response.get("status") != "PASS" or response.get("boot_id") != boot_id:
        raise PromotionError("administrator response is not a current-boot PASS")
    if response.get("driver_version") != EXPECTED_DRIVER:
        raise PromotionError("administrator response driver mismatch")
    if response.get("gpu_contract") != "four allowed GPUs visible; four exact UUIDs excluded":
        raise PromotionError("administrator response GPU contract mismatch")
    return {
        "path": str(root),
        "success_sha256": sha256_file(root / "SUCCESS"),
        "checksums_sha256": sha256_file(root / "SHA256SUMS"),
        "response_sha256": sums["administrator-response-final.json"],
        "run_id": marker.get("run_id"),
        "fabric_manager": marker.get("fabric_manager"),
    }


def validate_smoke_evidence(root: Path, boot_id: str) -> dict[str, Any]:
    required = {
        "per-gpu-tensor.json",
        "nccl-allreduce.json",
        "nvml-before.csv",
        "nvml-after.csv",
        "row-remapper-before.txt",
        "row-remapper-after.txt",
        "compute-apps-before.csv",
        "compute-apps-after.csv",
        "kernel-delta.txt",
        "kernel-critical.txt",
    }
    sums = verify_sha256sums(root, required)
    marker = parse_marker(root / "SUCCESS")
    if marker.get("status") != "PASS" or marker.get("boot_id") != boot_id:
        raise PromotionError("CUDA/NCCL smoke is not a current-boot PASS")
    if Path(marker.get("evidence", "")) != root:
        raise PromotionError("CUDA/NCCL marker evidence path mismatch")
    if (root / "compute-apps-before.csv").stat().st_size != 0:
        raise PromotionError("GPU client existed before the smoke")
    if (root / "compute-apps-after.csv").stat().st_size != 0:
        raise PromotionError("GPU client remained after the smoke")
    if (root / "kernel-critical.txt").stat().st_size != 0:
        raise PromotionError("critical kernel event occurred during the smoke")
    per_gpu = load_json(root / "per-gpu-tensor.json")
    devices = per_gpu.get("devices")
    if per_gpu.get("status") != "PASS" or per_gpu.get("device_count") != 4:
        raise PromotionError("per-GPU tensor smoke did not pass on four devices")
    if per_gpu.get("torch") != "2.12.1+cu126" or per_gpu.get("cuda_runtime") != "12.6":
        raise PromotionError("PyTorch/CUDA runtime differs from the rebuilt environment")
    if per_gpu.get("cudnn") != 91002 or per_gpu.get("nccl") != [2, 29, 3]:
        raise PromotionError("cuDNN/NCCL runtime differs from the locked expectation")
    if not isinstance(devices, list) or [normalize_uuid(x.get("uuid")) for x in devices] != EXPECTED_UUIDS:
        raise PromotionError("per-GPU tensor UUID order mismatch")
    nccl = load_json(root / "nccl-allreduce.json")
    ranks = nccl.get("ranks")
    if (
        nccl.get("status") != "PASS"
        or nccl.get("backend") != "nccl"
        or nccl.get("world_size") != 4
        or int(nccl.get("collectives", 0)) < 3
        or nccl.get("expected_sum") != 10.0
        or not isinstance(ranks, list)
        or [normalize_uuid(x.get("uuid")) for x in ranks] != EXPECTED_UUIDS
    ):
        raise PromotionError("four-rank NCCL all-reduce evidence mismatch")
    for name in ("nvml-before.csv", "nvml-after.csv"):
        rows = list(csv.reader((root / name).read_text(encoding="utf-8").splitlines()))
        if len(rows) != 4:
            raise PromotionError(f"{name} does not contain four GPUs")
        for row, expected_bdf, expected_uuid in zip(rows, EXPECTED_BDFS, EXPECTED_UUIDS):
            fields = [item.strip() for item in row]
            if len(fields) != 8:
                raise PromotionError(f"invalid NVML row in {name}: {row!r}")
            if fields[1].upper() != expected_bdf or normalize_uuid(fields[2]) != expected_uuid:
                raise PromotionError(f"NVML identity mismatch in {name}: {row!r}")
            if fields[3] != "0" or fields[4] != "0":
                raise PromotionError(f"uncorrectable ECC is nonzero in {name}: {row!r}")
    return {
        "path": str(root),
        "success_sha256": sha256_file(root / "SUCCESS"),
        "checksums_sha256": sha256_file(root / "SHA256SUMS"),
        "per_gpu_tensor_sha256": sums["per-gpu-tensor.json"],
        "nccl_allreduce_sha256": sums["nccl-allreduce.json"],
        "run_id": marker.get("run_id"),
        "torch": per_gpu.get("torch"),
        "cuda_runtime": per_gpu.get("cuda_runtime"),
        "cudnn_runtime": "9.10.2",
        "nccl_runtime": "2.29.3",
    }


def validate_cpu_candidate(path: Path, data_root: Path) -> dict[str, Any]:
    candidate = load_json(path)
    if (
        candidate.get("schema_version") != "stage0.environment-recommendation.v1"
        or candidate.get("classification") != "CPU_ONLY_CANDIDATE"
        or candidate.get("training_eligible") is not False
        or candidate.get("g2_status") != "BLOCKED"
    ):
        raise PromotionError("source environment reference is not the blocked CPU candidate")
    environment_path = require_directory(Path(str(candidate.get("path", ""))), within=data_root)
    expected_env_root = data_root / "envs"
    try:
        environment_path.relative_to(expected_env_root)
    except ValueError as error:
        raise PromotionError("candidate environment is outside DATA_ROOT/envs") from error
    python = environment_path / "bin/python"
    if not python.exists() or not python.resolve(strict=True).is_file() or not os.access(python, os.X_OK):
        raise PromotionError("candidate Python is unavailable")
    for key in ("environment_identity_manifest", "build_observation_manifest"):
        manifest_path = Path(str(candidate.get(key, ""))).resolve(strict=True)
        try:
            manifest_path.relative_to(data_root / "manifests")
        except ValueError as error:
            raise PromotionError(f"{key} escapes DATA_ROOT/manifests") from error
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise PromotionError(f"{key} is not a regular file")
    identity_path = Path(str(candidate["environment_identity_manifest"]))
    build_path = Path(str(candidate["build_observation_manifest"]))
    identity = load_json(identity_path)
    build = load_json(build_path)
    environment_id = candidate.get("environment_id")
    build_id = candidate.get("build_id")
    if identity.get("environment_id") != environment_id:
        raise PromotionError("environment identity reference mismatch")
    if build.get("environment_id") != environment_id or build.get("build_id") != build_id:
        raise PromotionError("environment build reference mismatch")
    if build.get("classification") != "CPU_ONLY_CANDIDATE" or build.get("training_eligible") is not False:
        raise PromotionError("source build observation is not CPU-only")
    return {
        "reference": candidate,
        "reference_path": str(path),
        "reference_sha256": sha256_file(path),
        "environment_path": str(environment_path),
        "python": str(python),
        "identity_path": str(identity_path),
        "identity_sha256": sha256_file(identity_path),
        "build_path": str(build_path),
        "build_sha256": sha256_file(build_path),
    }


def run(command: list[str], *, timeout: int = 60, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        env=env,
    )
    if completed.returncode != 0:
        raise PromotionError(
            f"command failed ({completed.returncode}): {command!r}: {completed.stderr.strip()}"
        )
    return completed.stdout


def validate_live_host(python: Path, boot_id: str, data_root: Path) -> dict[str, Any]:
    if socket.gethostname().split(".", 1)[0] != EXPECTED_HOST:
        raise PromotionError(f"promotion may run only on {EXPECTED_HOST}")
    params = Path("/proc/driver/nvidia/params").read_text(encoding="utf-8")
    match = re.search(r"^ExcludedGpus:\s*(.*)$", params, flags=re.MULTILINE)
    if not match or match.group(1).split(",") != EXPECTED_EXCLUDED_UUIDS:
        raise PromotionError("live NVIDIA excluded-UUID contract mismatch")
    nvml_command = [
        "nvidia-smi",
        "--query-gpu=pci.bus_id,uuid,driver_version,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total",
        "--format=csv,noheader,nounits",
    ]

    def validate_nvml(text: str, phase: str) -> None:
        rows = list(csv.reader(text.splitlines()))
        if len(rows) != 4:
            raise PromotionError(f"live NVML {phase} enumeration is not exactly four GPUs")
        for row, expected_bdf, expected_uuid in zip(rows, EXPECTED_BDFS, EXPECTED_UUIDS):
            fields = [item.strip() for item in row]
            if (
                len(fields) != 5
                or fields[0].upper() != expected_bdf
                or normalize_uuid(fields[1]) != expected_uuid
                or fields[2] != EXPECTED_DRIVER
                or fields[3:] != ["0", "0"]
            ):
                raise PromotionError(f"live NVML {phase} health/identity mismatch: {row!r}")

    nvml_text = run(nvml_command)
    validate_nvml(nvml_text, "pre-tensor")
    clients = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader",
        ]
    )
    if clients.strip():
        raise PromotionError("live GPU compute client detected during promotion")
    service_expectations = {
        "nvidia-persistenced.service": ("active", "enabled"),
        "containerd.service": ("active", "enabled"),
        "docker.service": ("active", "enabled"),
        "docker.socket": ("active", "enabled"),
        "nvidia-fabricmanager.service": ("inactive", "masked"),
        "snap.lxd.activate.service": ("inactive", "enabled"),
        "snap.lxd.daemon.unix.socket": ("active", "enabled"),
        "snap.lxd.user-daemon.unix.socket": ("active", "enabled"),
    }
    service_rows = []
    for unit, expected in service_expectations.items():
        active = run(["systemctl", "is-active", unit]).strip() if expected[0] == "active" else subprocess.run(
            ["systemctl", "is-active", unit], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        ).stdout.strip()
        enabled_process = subprocess.run(
            ["systemctl", "is-enabled", unit], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        enabled = enabled_process.stdout.strip()
        if (active, enabled) != expected:
            raise PromotionError(f"service state mismatch: {unit}={(active, enabled)} != {expected}")
        service_rows.append({"unit": unit, "active": active, "enabled": enabled})
    python_code = r'''
import json, torch
expected = [
 "5c672d04-4f83-3cc0-80d0-0108b1b63267",
 "e78c55cd-db97-b761-f559-dc6eae3be81d",
 "9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
 "5a81500d-5e9c-b0d7-5607-fdfdaab65ff4",
]
assert torch.cuda.is_available() and torch.cuda.device_count() == 4
observed = [str(torch.cuda.get_device_properties(i).uuid).lower().removeprefix("gpu-") for i in range(4)]
assert observed == expected
values = []
for i in range(4):
    with torch.cuda.device(i):
        value = float((torch.arange(1024, dtype=torch.float32, device=i) + 1).sum().item())
        torch.cuda.synchronize(i)
        assert value == 524800.0
        values.append(value)
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "nccl": list(torch.cuda.nccl.version()), "uuids": observed, "tensor_sums": values}))
'''
    clean_env = {
        "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(data_root / "tmp"),
        "TMPDIR": str(data_root / "tmp"),
        "XDG_CACHE_HOME": str(data_root / "cache"),
        "TORCH_HOME": str(data_root / "cache/torch"),
        "CUDA_CACHE_PATH": str(data_root / "cache/torch"),
    }
    runtime = json.loads(run([str(python), "-I", "-c", python_code], timeout=120, env=clean_env))
    if (
        runtime.get("torch") != "2.12.1+cu126"
        or runtime.get("cuda") != "12.6"
        or runtime.get("cudnn") != 91002
        or runtime.get("nccl") != [2, 29, 3]
        or [normalize_uuid(value) for value in runtime.get("uuids", [])] != EXPECTED_UUIDS
    ):
        raise PromotionError("live candidate runtime validation mismatch")
    nvml_after = run(nvml_command)
    validate_nvml(nvml_after, "post-tensor")
    clients_after = run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,gpu_uuid,process_name",
            "--format=csv,noheader",
        ]
    )
    if clients_after.strip():
        raise PromotionError("live GPU compute client remained after promotion tensor checks")
    row_remapper = run(["nvidia-smi", "-q", "-d", "ROW_REMAPPER"])
    blocks = re.split(r"(?=^GPU\s+[0-9A-Fa-f:.]+\s*$)", row_remapper, flags=re.MULTILINE)
    row_states = []
    for block in blocks:
        if not re.search(r"^GPU\s+[0-9A-Fa-f:.]+\s*$", block, flags=re.MULTILINE):
            continue
        pending = re.search(r"^\s*Pending\s*:\s*(\S+)", block, flags=re.MULTILINE)
        failure = re.search(
            r"^\s*Remapping Failure Occurred\s*:\s*(\S+)", block, flags=re.MULTILINE
        )
        if not pending or pending.group(1) != "No" or not failure or failure.group(1) != "No":
            raise PromotionError("live row-remap pending/failure state is not clean")
        row_states.append({"pending": pending.group(1), "failure": failure.group(1)})
    if len(row_states) != 4:
        raise PromotionError("live row-remap report does not contain exactly four GPUs")
    kernel = run(["journalctl", "-k", "-b", "0", "--no-pager"], timeout=120)
    critical = re.findall(
        r"^.*(?:NVRM: Xid|RmInitAdapter failed|AER.*(?:Uncorrected|Fatal)|PCIe Bus Error.*severity=(?:Uncorrected|Fatal)).*$",
        kernel,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if critical:
        raise PromotionError(f"critical GPU/PCI event exists in current boot: {critical[-1]}")
    return {
        "hostname": EXPECTED_HOST,
        "boot_id": boot_id,
        "driver_version": EXPECTED_DRIVER,
        "allowed_bdfs": EXPECTED_BDFS,
        "allowed_uuids": EXPECTED_UUIDS,
        "excluded_uuids": EXPECTED_EXCLUDED_UUIDS,
        "runtime": runtime,
        "row_remapper": row_states,
        "services": service_rows,
        "current_boot_critical_event_count": 0,
    }


def acquire_lock(path: Path) -> int:
    if fcntl is None:
        raise PromotionError("environment promotion requires Linux fcntl locking")
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise PromotionError("promotion lock directory may not be a symlink")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1 or metadata.st_uid != os.getuid():
        os.close(descriptor)
        raise PromotionError("promotion lock must be a single-link, caller-owned regular file")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        os.close(descriptor)
        raise PromotionError("another environment promotion is active") from error
    return descriptor


def write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    os.chmod(path, 0o644)


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--admin-evidence", type=Path, required=True)
    parser.add_argument("--smoke-evidence", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    arguments = parser.parse_args(argv)
    if not FULL_COMMIT.fullmatch(arguments.source_commit):
        raise PromotionError("--source-commit must be a full lowercase Git commit")
    data_root = require_directory(arguments.data_root)
    manifests_root = require_directory(data_root / "manifests", within=data_root)
    operations_root = require_directory(data_root / "operations", within=data_root)
    admin_root = require_directory(arguments.admin_evidence)
    try:
        admin_root.relative_to(ADMIN_PREFIX)
    except ValueError as error:
        raise PromotionError("administrator evidence is outside the approved root") from error
    smoke_root = require_directory(arguments.smoke_evidence, within=data_root)
    try:
        smoke_root.relative_to(data_root / "operations/stage0/g0-g-uuid-exclusion")
    except ValueError as error:
        raise PromotionError("smoke evidence is outside the approved operations root") from error
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    lock_descriptor = acquire_lock(operations_root / "stage0-environment-promotion.lock")
    try:
        candidate = validate_cpu_candidate(
            manifests_root / "environment-cpu-candidate.json", data_root
        )
        admin = validate_admin_evidence(admin_root, boot_id)
        smoke = validate_smoke_evidence(smoke_root, boot_id)
        live = validate_live_host(Path(candidate["python"]), boot_id, data_root)
        qualification_identity = {
            "environment_id": candidate["reference"]["environment_id"],
            "build_id": candidate["reference"]["build_id"],
            "boot_id": boot_id,
            "allowed_uuids": EXPECTED_UUIDS,
            "excluded_uuids": EXPECTED_EXCLUDED_UUIDS,
            "admin_success_sha256": admin["success_sha256"],
            "smoke_success_sha256": smoke["success_sha256"],
            "smoke_checksums_sha256": smoke["checksums_sha256"],
            "source_commit": arguments.source_commit,
        }
        qualification_id = "gpuq-v1-" + sha256_bytes(stable_json_bytes(qualification_identity))
        generated_at = datetime.strptime(
            str(smoke["run_id"]), "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=timezone.utc).isoformat()
        qualification = {
            "schema_version": "stage0.environment-gpu-qualification.v1",
            "qualification_id": qualification_id,
            "status": "PASS",
            "gate": "G0-G",
            "environment_id": candidate["reference"]["environment_id"],
            "build_id": candidate["reference"]["build_id"],
            "generated_at": generated_at,
            "source_commit": arguments.source_commit,
            "candidate_reference": {
                "path": candidate["reference_path"],
                "sha256": candidate["reference_sha256"],
            },
            "environment_identity": {
                "path": candidate["identity_path"],
                "sha256": candidate["identity_sha256"],
            },
            "build_observation": {
                "path": candidate["build_path"],
                "sha256": candidate["build_sha256"],
            },
            "administrator_evidence": admin,
            "cuda_nccl_smoke_evidence": smoke,
            "live_validation": live,
            "scope_note": "This validates G0-G and the S0.3 CUDA/NCCL environment boundary; it does not claim the S0.7 G6 distributed-semantics gate.",
        }
        qualification_root = manifests_root / "environment-gpu-qualifications"
        qualification_path = qualification_root / f"{qualification_id}.json"
        qualification_bytes = stable_json_bytes(qualification)
        if qualification_path.exists():
            if qualification_path.read_bytes() != qualification_bytes:
                raise PromotionError("existing GPU qualification conflicts with this identity")
        else:
            write_immutable(qualification_path, qualification_bytes)
        recommendation = {
            "schema_version": "stage0.environment-recommendation.v1",
            "classification": "TRAINING_ELIGIBLE",
            "training_eligible": True,
            "g2_status": "PASS",
            "environment_id": candidate["reference"]["environment_id"],
            "environment_identity_manifest": candidate["identity_path"],
            "build_id": candidate["reference"]["build_id"],
            "build_observation_manifest": candidate["build_path"],
            "gpu_qualification_id": qualification_id,
            "gpu_qualification_manifest": str(qualification_path),
            "path": candidate["environment_path"],
            "updated_at": generated_at,
            "git_commit": arguments.source_commit,
        }
        recommendation_path = manifests_root / "environment-recommended.json"
        atomic_write(recommendation_path, stable_json_bytes(recommendation))
        result = {
            "status": "PASS",
            "g2_status": "PASS",
            "training_eligible": True,
            "qualification_id": qualification_id,
            "qualification_manifest": str(qualification_path),
            "qualification_sha256": sha256_file(qualification_path),
            "recommendation": str(recommendation_path),
            "recommendation_sha256": sha256_file(recommendation_path),
            "environment_id": candidate["reference"]["environment_id"],
            "path": candidate["environment_path"],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    finally:
        assert fcntl is not None
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PromotionError as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
