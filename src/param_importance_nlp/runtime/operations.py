"""Stage 0 project-scoped leases, launch preflight, and failure policy.

The lease deliberately does not pretend to be a cluster scheduler.  It only
serializes this project's use of an explicitly approved GPU UUID set.  Every
hardware/external-process check remains a separate fail-closed preflight item.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from ..atomic import atomic_write_json, stable_json_hash
from ..contracts.jsonio import load_canonical_json
from ..lifecycle import validate_identifier


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _lock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - exercised by the formal Linux server
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


def process_is_alive(pid: int) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass(frozen=True, slots=True)
class GpuLeaseIdentity:
    run_id: str
    lease_id: str
    gpu_uuids: tuple[str, ...]
    owner: str
    config_hash: str
    environment_hash: str

    def __post_init__(self) -> None:
        validate_identifier(self.run_id, field="run_id")
        validate_identifier(self.lease_id, field="lease_id")
        values = tuple(self.gpu_uuids)
        if (
            not values
            or len(values) != len(set(values))
            or any(not isinstance(item, str) or not item.startswith("GPU-") for item in values)
            or not self.owner
        ):
            raise ValueError("GPU_LEASE_IDENTITY_INVALID")
        for digest in (self.config_hash, self.environment_hash):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("GPU_LEASE_HASH_INVALID")
        object.__setattr__(self, "gpu_uuids", values)

    @property
    def resource_key(self) -> str:
        return stable_json_hash(sorted(self.gpu_uuids))[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "lease_id": self.lease_id,
            "gpu_uuids": list(self.gpu_uuids),
            "owner": self.owner,
            "config_hash": self.config_hash,
            "environment_hash": self.environment_hash,
            "resource_key": self.resource_key,
        }


class ProjectGpuLease:
    """Non-blocking advisory lease for one exact project GPU UUID set."""

    def __init__(self, data_root: str | Path, identity: GpuLeaseIdentity) -> None:
        root = Path(data_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("GPU_LEASE_DATA_ROOT_INVALID")
        self.root = root
        self.identity = identity
        self.lease_root = root / "operations" / "gpu-leases"
        self.current_root = self.lease_root / "current"
        self.history_root = self.lease_root / "history"
        self.lock_root = self.lease_root / "locks"
        for path in (self.current_root, self.history_root, self.lock_root):
            path.mkdir(parents=True, exist_ok=True)
        key = identity.resource_key
        self.lock_path = self.lock_root / f"{key}.lock"
        self.current_path = self.current_root / f"{key}.json"
        self._descriptor = -1
        self._acquired_at: str | None = None

    def acquire(self) -> Mapping[str, Any]:
        if self._descriptor >= 0:
            raise RuntimeError("GPU_LEASE_ALREADY_ACQUIRED")
        descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            _lock(descriptor)
        except OSError as error:
            os.close(descriptor)
            raise RuntimeError(
                f"GPU_LEASE_RESOURCE_BUSY:{self.identity.resource_key}"
            ) from error
        self._descriptor = descriptor
        try:
            if self.current_path.exists():
                current = load_canonical_json(self.current_path)
                if not isinstance(current, Mapping):
                    raise RuntimeError("GPU_LEASE_STALE_RECORD_INVALID")
                pid = current.get("pid")
                if isinstance(pid, int) and process_is_alive(pid):
                    raise RuntimeError(f"GPU_LEASE_LIVE_OWNER_PRESENT:{pid}")
                # A free OS lock plus a stale current record is still not
                # silently reaped.  An operator must preserve and adjudicate it.
                raise RuntimeError("GPU_LEASE_STALE_RECORD_REQUIRES_REVIEW")
            acquired_at = _now()
            body: dict[str, Any] = {
                "schema_version": "runtime.project-gpu-lease.v1",
                **self.identity.to_dict(),
                "pid": os.getpid(),
                "acquired_at": acquired_at,
                "heartbeat_at": acquired_at,
                "status": "HELD",
            }
            body["artifact_hash"] = stable_json_hash(body)
            atomic_write_json(self.current_path, body)
            self._acquired_at = acquired_at
            return MappingProxyType(body)
        except BaseException:
            self._release_descriptor()
            raise

    def heartbeat(self) -> Mapping[str, Any]:
        if self._descriptor < 0 or not self.current_path.exists():
            raise RuntimeError("GPU_LEASE_NOT_HELD")
        raw = load_canonical_json(self.current_path)
        if not isinstance(raw, Mapping):
            raise RuntimeError("GPU_LEASE_RECORD_INVALID")
        value = dict(raw)
        declared = value.pop("artifact_hash", None)
        if declared != stable_json_hash(value) or value.get("pid") != os.getpid():
            raise RuntimeError("GPU_LEASE_RECORD_IDENTITY_DRIFT")
        value["heartbeat_at"] = _now()
        value["artifact_hash"] = stable_json_hash(value)
        atomic_write_json(self.current_path, value)
        return MappingProxyType(value)

    def release(self, *, outcome: str) -> Path:
        if self._descriptor < 0 or self._acquired_at is None:
            raise RuntimeError("GPU_LEASE_NOT_HELD")
        if not outcome:
            raise ValueError("GPU_LEASE_RELEASE_OUTCOME_REQUIRED")
        raw = load_canonical_json(self.current_path)
        if not isinstance(raw, Mapping):
            raise RuntimeError("GPU_LEASE_RECORD_INVALID")
        current = dict(raw)
        declared = current.pop("artifact_hash", None)
        if declared != stable_json_hash(current) or current.get("pid") != os.getpid():
            raise RuntimeError("GPU_LEASE_RELEASE_IDENTITY_DRIFT")
        completed_at = _now()
        history: dict[str, Any] = {
            "schema_version": "runtime.project-gpu-lease-history.v1",
            "lease": {**current, "artifact_hash": declared},
            "released_at": completed_at,
            "outcome": outcome,
        }
        history["artifact_hash"] = stable_json_hash(history)
        acquisition_suffix = stable_json_hash(self._acquired_at)[:12]
        target = self.history_root / (
            f"{self.identity.lease_id}-{os.getpid()}-{acquisition_suffix}.json"
        )
        if target.exists():
            raise FileExistsError(f"GPU_LEASE_HISTORY_COLLISION:{target}")
        atomic_write_json(target, history)
        self.current_path.unlink()
        self._release_descriptor()
        return target

    def _release_descriptor(self) -> None:
        if self._descriptor < 0:
            return
        try:
            _unlock(self._descriptor)
        finally:
            os.close(self._descriptor)
            self._descriptor = -1

    def close(self) -> None:
        """Release only the OS descriptor; never delete an unreviewed record."""

        self._release_descriptor()

    def __enter__(self) -> "ProjectGpuLease":
        self.acquire()
        return self

    def __exit__(self, exc_type: object, _exc: object, _tb: object) -> None:
        if self._descriptor >= 0:
            self.release(outcome="FAILED" if exc_type is not None else "SUCCESS")


class LaunchClaimRegistry:
    """Immutable idempotency claim preventing duplicate SSH-triggered launches."""

    def __init__(self, data_root: str | Path) -> None:
        self.root = Path(data_root).resolve(strict=True) / "operations" / "launch-claims"
        self.root.mkdir(parents=True, exist_ok=True)

    def claim(
        self,
        *,
        launch_id: str,
        run_id: str,
        config_hash: str,
        environment_hash: str,
    ) -> Path:
        validate_identifier(launch_id, field="launch_id")
        validate_identifier(run_id, field="run_id")
        target = self.root / f"{launch_id}.json"
        body: dict[str, Any] = {
            "schema_version": "runtime.launch-claim.v1",
            "launch_id": launch_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "environment_hash": environment_hash,
            "owner_pid": os.getpid(),
            "claimed_at": _now(),
            "status": "CLAIMED",
        }
        body["artifact_hash"] = stable_json_hash(body)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("xb") as handle:
                payload = __import__("json").dumps(
                    body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise RuntimeError(f"LAUNCH_CLAIM_ALREADY_EXISTS:{launch_id}") from error
        return target


class FailureClass(StrEnum):
    OOM = "OOM"
    NCCL_TIMEOUT = "NCCL_OR_TIMEOUT"
    DEVICE_HEALTH = "ECC_OR_DEVICE_DISAPPEARED"
    DISK_FULL = "DISK_INSUFFICIENT"
    TRUTH_LOG_WRITE = "JSONL_OR_STATUS_WRITE_FAILED"
    DERIVED_TRACKING = "TENSORBOARD_DERIVATION_FAILED"
    CHECKPOINT_CORRUPT = "CHECKPOINT_CORRUPT"
    SSH_DISCONNECT = "SSH_CLIENT_DISCONNECTED"
    DOWNLOAD = "ASSET_DOWNLOAD_FAILED"
    DATA_ROOT_FAILURE = "DATA_ROOT_FILESYSTEM_FAILED"


_FAILURE_RESPONSE: Mapping[FailureClass, Mapping[str, Any]] = MappingProxyType(
    {
        FailureClass.OOM: {"stop": True, "retry": "new_config_identity", "preserve_peak": True},
        FailureClass.NCCL_TIMEOUT: {"stop": True, "retry": "fresh_process_group", "bounded_group_exit": True},
        FailureClass.DEVICE_HEALTH: {"stop": True, "retry": "administrator_requalification", "invalidate_hardware_gate": True},
        FailureClass.DISK_FULL: {"stop": True, "retry": "after_capacity_preflight", "delete_unknown_files": False},
        FailureClass.TRUTH_LOG_WRITE: {"stop": True, "retry": "after_truth_store_repair", "truth_preserved": False},
        FailureClass.DERIVED_TRACKING: {"stop": False, "retry": "rebuild_from_jsonl", "truth_preserved": True},
        FailureClass.CHECKPOINT_CORRUPT: {"stop": True, "retry": "previous_complete_checkpoint", "preserve_corrupt_evidence": True},
        FailureClass.SSH_DISCONNECT: {"stop": False, "retry": "inspect_launch_claim_and_status", "duplicate_launch": False},
        FailureClass.DOWNLOAD: {"stop": True, "retry": "existing_acquisition_supervisor", "mix_with_training": False},
        FailureClass.DATA_ROOT_FAILURE: {"stop": True, "retry": "authorized_second_fault_domain_only", "same_disk_is_backup": False},
    }
)


def failure_response(failure: FailureClass | str) -> dict[str, Any]:
    selected = FailureClass(failure)
    return {"failure_class": selected.value, **dict(_FAILURE_RESPONSE[selected])}


def classify_stale_heartbeat(*, heartbeat_stale: bool, process_alive: bool) -> str:
    if heartbeat_stale and process_alive:
        return "ACTIVE_PROCESS_HEARTBEAT_STALE_DO_NOT_REAP"
    if heartbeat_stale:
        return "STALE_NO_PROCESS_REVIEW_REQUIRED"
    return "ACTIVE_OR_RECENT"


def exercise_canary_writer(writer: Callable[[], object]) -> dict[str, Any]:
    """Run an injected canary writer and never turn a write error into success."""

    try:
        writer()
    except OSError as error:
        return {
            "status": "EXPECTED_FAILURE",
            "failure_class": FailureClass.TRUTH_LOG_WRITE.value,
            "error_type": type(error).__name__,
            "stop_required": True,
        }
    return {"status": "PASS", "failure_class": None, "stop_required": False}


def evaluate_launch_preflight(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate a complete immutable preflight snapshot using hard rules."""

    required = {
        "required_gate_ids",
        "passed_gate_ids",
        "source_clean",
        "identity_hashes_match",
        "selected_gpu_uuids",
        "gpu_health_ok",
        "external_gpu_processes",
        "active_competing_downloads",
        "data_cursor_covered",
        "data_free_bytes",
        "expected_new_bytes",
        "root_free_bytes",
        "inode_free",
        "fd_soft_limit",
        "predicted_open_fds",
        "memory_available_bytes",
        "predicted_host_peak_bytes",
        "output_collision",
        "lease_available",
        "g1d_status",
    }
    if set(snapshot) != required:
        raise ValueError(f"OPERATIONS_PREFLIGHT_FIELDS_MISMATCH:{sorted(set(snapshot)^required)}")
    blockers: list[dict[str, Any]] = []

    def block(code: str, evidence: Any) -> None:
        blockers.append({"code": code, "hard": True, "evidence": evidence})

    required_gates = set(snapshot["required_gate_ids"])
    passed_gates = set(snapshot["passed_gate_ids"])
    missing = sorted(required_gates - passed_gates)
    if missing:
        block("PREFLIGHT_GATE_MISSING", missing)
    if snapshot["source_clean"] is not True or snapshot["identity_hashes_match"] is not True:
        block("PREFLIGHT_SOURCE_OR_IDENTITY_DRIFT", {
            "source_clean": snapshot["source_clean"],
            "identity_hashes_match": snapshot["identity_hashes_match"],
        })
    uuids = snapshot["selected_gpu_uuids"]
    if (
        not isinstance(uuids, Sequence)
        or isinstance(uuids, (str, bytes))
        or len(uuids) != 4
        or len(set(uuids)) != 4
        or snapshot["gpu_health_ok"] is not True
    ):
        block("PREFLIGHT_GPU_HEALTH_OR_MAPPING", uuids)
    if snapshot["external_gpu_processes"]:
        block("PREFLIGHT_EXTERNAL_GPU_PROCESS", snapshot["external_gpu_processes"])
    if snapshot["active_competing_downloads"]:
        block("PREFLIGHT_COMPETING_DOWNLOAD", snapshot["active_competing_downloads"])
    if snapshot["data_cursor_covered"] is not True:
        block("PREFLIGHT_DATA_CURSOR_NOT_COVERED", False)
    expected = int(snapshot["expected_new_bytes"])
    required_free = expected + max((expected + 4) // 5, 100 * 1024**3)
    if int(snapshot["data_free_bytes"]) < required_free:
        block("PREFLIGHT_DATA_DISK_INSUFFICIENT", {
            "free": snapshot["data_free_bytes"], "required": required_free,
        })
    if int(snapshot["root_free_bytes"]) < 10 * 1024**3:
        block("PREFLIGHT_ROOT_DISK_INSUFFICIENT", snapshot["root_free_bytes"])
    if int(snapshot["inode_free"]) <= 0:
        block("PREFLIGHT_INODE_EXHAUSTED", snapshot["inode_free"])
    if int(snapshot["predicted_open_fds"]) > math_floor_70_percent(int(snapshot["fd_soft_limit"])):
        block("PREFLIGHT_FD_HEADROOM_INSUFFICIENT", {
            "predicted": snapshot["predicted_open_fds"], "soft": snapshot["fd_soft_limit"],
        })
    if int(snapshot["predicted_host_peak_bytes"]) > int(snapshot["memory_available_bytes"]):
        block("PREFLIGHT_HOST_MEMORY_INSUFFICIENT", {
            "predicted": snapshot["predicted_host_peak_bytes"],
            "available": snapshot["memory_available_bytes"],
        })
    if snapshot["output_collision"] is True:
        block("PREFLIGHT_OUTPUT_COLLISION", True)
    if snapshot["lease_available"] is not True:
        block("PREFLIGHT_PROJECT_LEASE_UNAVAILABLE", False)
    if snapshot["g1d_status"] not in {"AUTHORIZED_SECOND_FAULT_DOMAIN", "ACCEPTED_SINGLE_DISK_RISK"}:
        block("PREFLIGHT_G1D_STATUS_INVALID", snapshot["g1d_status"])
    body: dict[str, Any] = {
        "schema_version": "stage0.operations-preflight-report.v1",
        "status": "PASS" if not blockers else "FAIL",
        "running_state_may_publish": not blockers,
        "required_data_free_bytes": required_free,
        "snapshot": dict(snapshot),
        "blockers": blockers,
    }
    body["artifact_hash"] = stable_json_hash(body)
    return body


def math_floor_70_percent(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("OPERATIONS_LIMIT_INVALID")
    return value * 7 // 10


__all__ = [
    "FailureClass",
    "GpuLeaseIdentity",
    "LaunchClaimRegistry",
    "ProjectGpuLease",
    "classify_stale_heartbeat",
    "evaluate_launch_preflight",
    "exercise_canary_writer",
    "failure_response",
    "math_floor_70_percent",
    "process_is_alive",
]
