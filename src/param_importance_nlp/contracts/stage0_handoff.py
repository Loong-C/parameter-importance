"""Fail-closed Stage 0 handoff consumed by the Stage 2 formal adapter.

The historical G10 bundle is useful as a role/hash index, but it is not a
current readiness assertion.  This contract makes that distinction explicit:
all required Stage 0 roles are recorded, while formal execution is unlocked
only by a manifest whose current hardware and persistence checks are READY.
Temporary paths are never accepted as formal evidence authority.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
import hashlib
from pathlib import Path, PurePosixPath

from .jsonio import canonical_json_hash, load_canonical_json


STAGE0_HANDOFF_SCHEMA = "stage0-handoff-manifest-v1"
STAGE0_G10_GENERATOR_COMMIT = "a15f0e2970b7cae6951dd606ebd396a8df68255c"
STAGE0_G10_EVIDENCE_ROOT = (
    "evidence/stage0/g10-final/"
    "a15f0e2970b7cae6951dd606ebd396a8df68255c/"
    "b4974a642168994eec7d62ba38a453fa3834ee50201da55d9549b1080a5b90f0"
)
STAGE0_HANDOFF_ROLES = (
    "environment_freeze",
    "storage_cache",
    "single_gpu",
    "four_gpu_ddp_no_sync",
    "logging_run",
    "checkpoint_recovery",
    "performance",
)
_CACHE_VARIABLES = (
    "HF_HOME",
    "HF_DATASETS_CACHE",
    "TORCH_HOME",
    "XDG_CACHE_HOME",
)
_TEMP_VARIABLES = ("TEMP", "TMP", "TMPDIR", "PYTHONPYCACHEPREFIX")


class Stage0HandoffError(ValueError):
    """Raised when Stage 0 evidence is absent, stale, or structurally invalid."""


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_INVALID")
    return value


def _sha(value: object, *, field: str) -> str:
    value = _text(value, field=field)
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_HASH_INVALID")
    return value


def _commit(value: object, *, field: str) -> str:
    value = _text(value, field=field)
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_COMMIT_INVALID")
    return value


def _object(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_OBJECT_REQUIRED")
    return value


def _safe_relative(root: Path, value: object, *, field: str) -> Path:
    ref = _text(value, field=f"{field}.ref")
    if "\\" in ref:
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_REF_INVALID")
    logical = PurePosixPath(ref)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0HandoffError(f"STAGE0_HANDOFF_{field.upper()}_REF_INVALID")
    candidate = (root / Path(*logical.parts)).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage0HandoffError(
            f"STAGE0_HANDOFF_{field.upper()}_REF_ESCAPES_ROOT"
        ) from error
    return candidate


def _role_ref(value: object, *, role: str) -> str:
    ref = _text(value, field=f"role.{role}.ref")
    path = PurePosixPath(ref)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise Stage0HandoffError(f"STAGE0_HANDOFF_ROLE_{role.upper()}_REF_INVALID")
    if "\\" in ref or path.parts[:2] != ("evidence", "stage0"):
        raise Stage0HandoffError(
            f"STAGE0_HANDOFF_ROLE_{role.upper()}_CANONICAL_ROOT_REQUIRED"
        )
    if "tmp" in path.parts or "reports" in path.parts or "fixtures" in path.parts:
        raise Stage0HandoffError(
            f"STAGE0_HANDOFF_ROLE_{role.upper()}_TEMPORARY_OR_FIXTURE_FORBIDDEN"
        )
    return ref


def _artifact_hash(value: Mapping[str, object]) -> str:
    supplied = _sha(value.get("artifact_hash"), field="artifact")
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise Stage0HandoffError("STAGE0_HANDOFF_ARTIFACT_HASH_MISMATCH")
    return supplied


@dataclass(frozen=True, slots=True)
class Stage0HandoffEvidence:
    """Validated Stage 0 role/hash binding and current validity state."""

    manifest_ref: str
    manifest_sha256: str
    artifact_hash: str
    producer_commit: str
    status: str
    accepted_at: str
    roles: tuple[tuple[str, str, str, str, str, str], ...]
    hardware_validity: str
    persistence_validity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_ref": self.manifest_ref,
            "manifest_sha256": self.manifest_sha256,
            "artifact_hash": self.artifact_hash,
            "producer_commit": self.producer_commit,
            "status": self.status,
            "accepted_at": self.accepted_at,
            "roles": {
                role: {
                    "ref": ref,
                    "sha256": sha,
                    "producer_commit": commit,
                    "accepted_at": role_accepted,
                    "status": role_status,
                }
                for role, ref, sha, commit, role_accepted, role_status in self.roles
            },
            "hardware_validity": self.hardware_validity,
            "persistence_validity": self.persistence_validity,
        }


def validate_stage0_handoff(
    root: Path,
    manifest_ref: str,
    *,
    require_ready: bool = False,
    evidence_root: Path | None = None,
) -> Stage0HandoffEvidence:
    """Validate the Stage 0 manifest and optionally require formal readiness."""

    path = _safe_relative(root, manifest_ref, field="manifest")
    try:
        value = load_canonical_json(path)
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise Stage0HandoffError("STAGE0_HANDOFF_MANIFEST_MISSING_OR_INVALID") from error
    manifest = _object(value, field="manifest")
    artifact_hash = _artifact_hash(manifest)
    if manifest.get("schema_version") != STAGE0_HANDOFF_SCHEMA:
        raise Stage0HandoffError("STAGE0_HANDOFF_SCHEMA_INVALID")
    producer = _commit(manifest.get("producer_commit"), field="producer")
    _commit(manifest.get("execution_commit"), field="execution")
    _commit(manifest.get("consumer_commit"), field="consumer")
    status = _text(manifest.get("status"), field="status")
    accepted_at = _text(manifest.get("accepted_at"), field="accepted_at")
    try:
        datetime.fromisoformat(accepted_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise Stage0HandoffError("STAGE0_HANDOFF_ACCEPTED_AT_INVALID") from error

    authority = _object(manifest.get("authority"), field="authority")
    if authority.get("kind") != "canonical_historical_role_index":
        raise Stage0HandoffError("STAGE0_HANDOFF_AUTHORITY_KIND_INVALID")
    if _commit(authority.get("generator_commit"), field="authority.generator") != STAGE0_G10_GENERATOR_COMMIT:
        raise Stage0HandoffError("STAGE0_HANDOFF_HISTORICAL_GENERATOR_INVALID")
    authority_root = _text(authority.get("root"), field="authority.root")
    if not authority_root.startswith("evidence/stage0/") or "tmp" in PurePosixPath(authority_root).parts:
        raise Stage0HandoffError("STAGE0_HANDOFF_AUTHORITY_ROOT_INVALID")
    if authority.get("temporary_authority_forbidden") is not True:
        raise Stage0HandoffError("STAGE0_HANDOFF_TEMPORARY_AUTHORITY_NOT_FORBIDDEN")

    roles_value = _object(manifest.get("roles"), field="roles")
    if set(roles_value) != set(STAGE0_HANDOFF_ROLES):
        raise Stage0HandoffError("STAGE0_HANDOFF_ROLE_SET_INVALID")
    roles: list[tuple[str, str, str, str, str, str]] = []
    for role in STAGE0_HANDOFF_ROLES:
        item = _object(roles_value.get(role), field=f"role.{role}")
        ref = _role_ref(item.get("ref"), role=role)
        sha = _sha(item.get("sha256"), field=f"role.{role}.sha256")
        role_commit = _commit(item.get("producer_commit"), field=f"role.{role}.producer")
        role_accepted = _text(item.get("accepted_at"), field=f"role.{role}.accepted_at")
        role_status = _text(item.get("status"), field=f"role.{role}.status")
        if evidence_root is not None:
            source = _safe_relative(evidence_root, ref, field=f"role.{role}")
            try:
                observed = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as error:
                raise Stage0HandoffError(
                    f"STAGE0_HANDOFF_ROLE_{role.upper()}_SOURCE_MISSING"
                ) from error
            if observed != sha:
                raise Stage0HandoffError(
                    f"STAGE0_HANDOFF_ROLE_{role.upper()}_SOURCE_HASH_MISMATCH"
                )
        roles.append((role, ref, sha, role_commit, role_accepted, role_status))

    hardware = _object(manifest.get("hardware_validity"), field="hardware_validity")
    hardware_status = _text(hardware.get("status"), field="hardware_validity.status")
    if not isinstance(hardware.get("excluded_devices"), list) or not hardware.get(
        "excluded_devices"
    ):
        raise Stage0HandoffError("STAGE0_HANDOFF_HARDWARE_EXCLUSION_REQUIRED")
    storage = _object(manifest.get("storage_cache"), field="storage_cache")
    if storage.get("home_unchanged") is not True:
        raise Stage0HandoffError("STAGE0_HANDOFF_HOME_MUTATION_FORBIDDEN")
    variables = _object(storage.get("environment_variables"), field="environment_variables")
    for name in _CACHE_VARIABLES + _TEMP_VARIABLES:
        value = variables.get(name)
        if not isinstance(value, str) or "${DATA_ROOT}" not in value:
            raise Stage0HandoffError(f"STAGE0_HANDOFF_{name}_PATH_INVALID")
    if not variables["TMPDIR"].startswith("${DATA_ROOT}/tmp/stage2/"):
        raise Stage0HandoffError("STAGE0_HANDOFF_TMPDIR_PATH_INVALID")
    if not variables["PYTHONPYCACHEPREFIX"].startswith("${DATA_ROOT}/tmp/stage2/"):
        raise Stage0HandoffError("STAGE0_HANDOFF_PYCACHE_PATH_INVALID")

    persistence = _object(manifest.get("persistence"), field="persistence")
    persistence_status = _text(persistence.get("status"), field="persistence.status")
    expiry = persistence.get("risk_acceptance_expires_at")
    if expiry is not None:
        try:
            expires = datetime.fromisoformat(_text(expiry, field="risk_acceptance_expires_at").replace("Z", "+00:00"))
            now = datetime.now(expires.tzinfo)
            if expires <= now:
                raise Stage0HandoffError("STAGE0_HANDOFF_RISK_ACCEPTANCE_EXPIRED")
        except Stage0HandoffError:
            raise
        except ValueError as error:
            raise Stage0HandoffError("STAGE0_HANDOFF_RISK_ACCEPTANCE_EXPIRY_INVALID") from error

    observed_manifest_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if require_ready:
        if status != "READY":
            raise Stage0HandoffError("STAGE0_HANDOFF_STATUS_NOT_READY")
        if hardware_status != "VALID":
            raise Stage0HandoffError("STAGE0_HANDOFF_HARDWARE_NOT_CURRENT")
        if persistence_status != "VALID":
            raise Stage0HandoffError("STAGE0_HANDOFF_PERSISTENCE_NOT_VALID")
        if any(role_status != "PASS" for *_, role_status in roles):
            raise Stage0HandoffError("STAGE0_HANDOFF_ROLE_NOT_PASS")
    return Stage0HandoffEvidence(
        manifest_ref=manifest_ref,
        manifest_sha256=observed_manifest_sha,
        artifact_hash=artifact_hash,
        producer_commit=producer,
        status=status,
        accepted_at=accepted_at,
        roles=tuple(roles),
        hardware_validity=hardware_status,
        persistence_validity=persistence_status,
    )


__all__ = [
    "STAGE0_G10_EVIDENCE_ROOT",
    "STAGE0_G10_GENERATOR_COMMIT",
    "STAGE0_HANDOFF_ROLES",
    "STAGE0_HANDOFF_SCHEMA",
    "Stage0HandoffError",
    "Stage0HandoffEvidence",
    "validate_stage0_handoff",
]
