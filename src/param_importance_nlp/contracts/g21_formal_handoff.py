"""Strict builder/loader for the append-only Stage 2 G2.1 handoff.

This contract binds the current bounded GPU smoke to the historical Stage 0
role identities, the current Stage 1 exit identity, and the dated user
authorization.  It is deliberately independent from the old blocked JSON so
an amendment can only be published at a new path.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path
import re
from typing import Any

from .jsonio import canonical_json_hash, load_canonical_json, loads_strict_json, write_canonical_json

SCHEMA = "stage2-s2.2-g2.1-formal-handoff-v1"
AUTH_HASH = "51cb1ed87ff6ded4f4001f2a0b67dd469ebf048df2592b27707bc1f535b6db0c"
STAGE1_IDENTITY = "3f18b04df8922be9894678ae4842bd999c7e8fd5"
EXCLUDED_PCI = "0000:50:00.0"
ALLOWED_DEVICES = (
    ("0000:53:00.0", "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd"),
    ("0000:9C:00.0", "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"),
    ("0000:9D:00.0", "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"),
    ("0000:A0:00.0", "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"),
)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class G21FormalHandoffError(ValueError):
    """Raised for any malformed, drifting, or unsafe G2.1 handoff."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise G21FormalHandoffError(f"{field}_INVALID")
    return value


def _sha(value: Any, field: str) -> str:
    value = _text(value, field)
    if not _SHA.fullmatch(value):
        raise G21FormalHandoffError(f"{field}_SHA256_INVALID")
    return value


def _commit(value: Any, field: str) -> str:
    value = _text(value, field)
    if not _COMMIT.fullmatch(value):
        raise G21FormalHandoffError(f"{field}_COMMIT_INVALID")
    return value


def _object(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise G21FormalHandoffError(f"{field}_OBJECT_REQUIRED")
    return dict(value)


def _relative(root: Path, value: Any, field: str) -> tuple[str, Path]:
    ref = _text(value, field)
    path = Path(ref)
    if path.is_absolute() or "\\" in ref or ".." in path.parts:
        raise G21FormalHandoffError(f"{field}_REF_INVALID")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise G21FormalHandoffError(f"{field}_REF_ESCAPES_ROOT") from error
    return ref, resolved


def _file_sha(root: Path, ref: Any, field: str) -> str:
    _, path = _relative(root, ref, field)
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise G21FormalHandoffError(f"{field}_MISSING") from error


def build_g21_formal_handoff(fields: Mapping[str, Any]) -> dict[str, Any]:
    """Add the canonical self-hash to a validated handoff payload."""
    payload = dict(fields)
    payload.pop("artifact_hash", None)
    payload.setdefault("schema_version", SCHEMA)
    payload.setdefault("status", "PASS")
    _validate(payload, root=None)
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def write_g21_formal_handoff(path: str | Path, fields: Mapping[str, Any]) -> Path:
    """Atomically publish one new canonical G2.1 handoff path."""
    target = Path(path)
    if target.exists():
        raise G21FormalHandoffError("APPEND_ONLY_TARGET_EXISTS")
    return write_canonical_json(target, build_g21_formal_handoff(fields))


def _validate(value: Mapping[str, Any], root: Path | None) -> dict[str, Any]:
    required = {
        "schema_version", "status", "gate_id", "producer_commit", "execution_commit",
        "consumer_commit", "authorization", "current_gpu_smoke", "historical_stage0",
        "stage1_g1_exit", "checks", "artifact_hash",
    }
    expected_fields = required if "artifact_hash" in value else required - {"artifact_hash"}
    if set(value) != expected_fields:
        raise G21FormalHandoffError(
            f"FIELDS_MISMATCH:missing={sorted(expected_fields-set(value))}:extra={sorted(set(value)-expected_fields)}"
        )
    if value["schema_version"] != SCHEMA or value["status"] != "PASS" or value["gate_id"] != "stage2.G2.1":
        raise G21FormalHandoffError("IDENTITY_OR_STATUS_INVALID")
    for name in ("producer_commit", "execution_commit", "consumer_commit"):
        _commit(value[name], name)

    auth = _object(value["authorization"], "authorization")
    if set(auth) != {"ref", "artifact_hash", "user_authorization_original", "issued_at", "expires_at", "scope", "single_copy_accepted", "excluded_pci_bus_ids", "excluded_non_reproducible_human_evidence"}:
        raise G21FormalHandoffError("AUTHORIZATION_FIELDS_INVALID")
    _text(auth["ref"], "authorization.ref")
    if _sha(auth["artifact_hash"], "authorization.artifact_hash") != AUTH_HASH:
        raise G21FormalHandoffError("AUTHORIZATION_HASH_MISMATCH")
    if auth["user_authorization_original"] != "允许 Stage 2 结束前继续使用单副本存储，排除故障 GPU 0000:50:00.0，继续执行":
        raise G21FormalHandoffError("AUTHORIZATION_ORIGINAL_MISMATCH")
    if auth["single_copy_accepted"] is not True or auth["excluded_pci_bus_ids"] != [EXCLUDED_PCI]:
        raise G21FormalHandoffError("AUTHORIZATION_SCOPE_INVALID")
    if auth["scope"] != ["reproducible_stage0_artifacts", "reproducible_stage2_artifacts"]:
        raise G21FormalHandoffError("AUTHORIZATION_REPRODUCIBLE_SCOPE_INVALID")
    if auth["excluded_non_reproducible_human_evidence"] is not True:
        raise G21FormalHandoffError("AUTHORIZATION_HUMAN_EVIDENCE_NOT_EXCLUDED")

    smoke = _object(value["current_gpu_smoke"], "current_gpu_smoke")
    expected_smoke = {"ref", "sha256", "schema_version", "status", "atomic_publication", "excluded_pci_bus_ids", "excluded_scheduled", "allowed_devices"}
    if set(smoke) != expected_smoke or smoke["schema_version"] != "stage2-s202-current-gpu-smoke-v1" or smoke["status"] != "PASS":
        raise G21FormalHandoffError("CURRENT_SMOKE_INVALID")
    _sha(smoke["sha256"], "current_gpu_smoke.sha256")
    if smoke["atomic_publication"] is not True or smoke["excluded_scheduled"] is not False or smoke["excluded_pci_bus_ids"] != [EXCLUDED_PCI]:
        raise G21FormalHandoffError("CURRENT_SMOKE_EXCLUSION_INVALID")
    devices = tuple((item.get("pci_bus_id"), item.get("uuid")) for item in smoke["allowed_devices"] if isinstance(item, Mapping))
    if devices != ALLOWED_DEVICES:
        raise G21FormalHandoffError("CURRENT_SMOKE_ALLOWED_DEVICES_INVALID")
    if root is not None:
        ref, path = _relative(root, smoke["ref"], "current_gpu_smoke.ref")
        if _file_sha(root, ref, "current_gpu_smoke.ref") != smoke["sha256"]:
            raise G21FormalHandoffError("CURRENT_SMOKE_HASH_MISMATCH")
        report = loads_strict_json(path.read_bytes())
        if not isinstance(report, Mapping):
            raise G21FormalHandoffError("CURRENT_SMOKE_REPORT_ROOT_INVALID")
        if report.get("status") != "PASS" or report.get("schema_version") != "stage2-s202-current-gpu-smoke-v1":
            raise G21FormalHandoffError("CURRENT_SMOKE_REPORT_INVALID")

    stage1 = _object(value["stage1_g1_exit"], "stage1_g1_exit")
    if stage1.get("status") != "PASS" or stage1.get("identity") != STAGE1_IDENTITY:
        raise G21FormalHandoffError("STAGE1_IDENTITY_INVALID")
    _sha(stage1.get("sha256"), "stage1_g1_exit.sha256")
    _text(stage1.get("ref"), "stage1_g1_exit.ref")
    _commit(stage1.get("producer_commit"), "stage1_g1_exit.producer_commit")

    history = _object(value["historical_stage0"], "historical_stage0")
    if set(history) != {"g5", "g6", "g10"}:
        raise G21FormalHandoffError("HISTORICAL_STAGE0_ROLE_SET_INVALID")
    for role, item in history.items():
        record = _object(item, f"historical_stage0.{role}")
        if record.get("status") != "PASS":
            raise G21FormalHandoffError(f"HISTORICAL_STAGE0_{role.upper()}_NOT_PASS")
        _text(record.get("ref"), f"historical_stage0.{role}.ref")
        _sha(record.get("sha256"), f"historical_stage0.{role}.sha256")
        _commit(record.get("producer_commit"), f"historical_stage0.{role}.producer_commit")

    checks = _object(value["checks"], "checks")
    expected_checks = {"producer_identity", "execution_identity", "consumer_identity", "authorization_scope", "stage0_g5_g6_g10_identity", "stage1_identity", "gpu_exclusion", "atomic_publication", "hash_verified", "replay_verified", "loader_verified"}
    if set(checks) != expected_checks or any(item is not True for item in checks.values()):
        raise G21FormalHandoffError("G2_1_CHECKS_NOT_ALL_PASS")
    if "artifact_hash" in value:
        supplied_hash = _sha(value["artifact_hash"], "artifact_hash")
        payload = dict(value)
        payload.pop("artifact_hash")
        if canonical_json_hash(payload) != supplied_hash:
            raise G21FormalHandoffError("ARTIFACT_HASH_MISMATCH")
    return dict(value)


def load_g21_formal_handoff(path: str | Path, *, data_root: str | Path | None = None) -> dict[str, Any]:
    """Load and fail closed on a canonical G2.1 handoff."""
    target = Path(path)
    value = load_canonical_json(target)
    if not isinstance(value, Mapping):
        raise G21FormalHandoffError("ROOT_OBJECT_REQUIRED")
    return _validate(value, Path(data_root).resolve() if data_root is not None else None)


__all__ = [
    "ALLOWED_DEVICES", "AUTH_HASH", "EXCLUDED_PCI", "G21FormalHandoffError",
    "SCHEMA", "STAGE1_IDENTITY", "build_g21_formal_handoff", "load_g21_formal_handoff",
    "write_g21_formal_handoff",
]
