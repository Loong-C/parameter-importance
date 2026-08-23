"""Fail-closed loader for the append-only Stage 2 authorization amendment."""

from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Mapping

from .jsonio import canonical_json_hash, load_canonical_json


AUTHORIZATION_REF = "reports/stage2/s2.2/formal-authorization-amendment-20260823.json"
EXCLUDED_PCI = "0000:50:00.0"
USER_AUTHORIZATION = (
    "允许 Stage 2 结束前继续使用单副本存储，排除故障 GPU 0000:50:00.0，继续执行"
)


class Stage2AuthorizationError(ValueError):
    """Raised when the Stage 2 authorization amendment is absent or invalid."""


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_OBJECT_REQUIRED")
    return value


def load_stage2_authorization(root: Path, reference: str = AUTHORIZATION_REF) -> Mapping[str, object]:
    """Load and validate the exact, append-only user authorization artifact."""

    if not reference or reference.startswith("/") or "\\" in reference or ".." in Path(reference).parts:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_REF_INVALID")
    path = (root / Path(reference)).resolve()
    try:
        path.relative_to(root.resolve())
        value = _object(load_canonical_json(path))
    except (FileNotFoundError, OSError, TypeError, ValueError) as error:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_MISSING_OR_INVALID") from error
    if value.get("schema_version") != "stage2-formal-authorization-amendment-v1":
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_SCHEMA_INVALID")
    if value.get("status") != "ACTIVE" or value.get("single_copy_accepted") is not True:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_NOT_ACTIVE")
    if value.get("user_authorization_original") != USER_AUTHORIZATION:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_ORIGINAL_TEXT_MISMATCH")
    if value.get("timezone") != "Asia/Shanghai" or value.get("stage2_exit_condition") != "Stage 2 exit":
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_SCOPE_INVALID")
    if value.get("excluded_pci_bus_ids") != [EXCLUDED_PCI]:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_GPU_EXCLUSION_INVALID")
    scopes = value.get("scope")
    if scopes != ["reproducible_stage0_artifacts", "reproducible_stage2_artifacts"]:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_ARTIFACT_SCOPE_INVALID")
    excluded = value.get("excluded_non_reproducible_evidence")
    if not isinstance(excluded, list) or "non_reproducible_human_evidence" not in excluded:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_HUMAN_EVIDENCE_NOT_EXCLUDED")
    try:
        issued = datetime.fromisoformat(str(value["issued_at"]))
        expires = datetime.fromisoformat(str(value["expires_at"]))
    except (KeyError, TypeError, ValueError) as error:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_TIME_INVALID") from error
    if issued.tzinfo is None or expires.tzinfo is None or expires <= issued:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_TIME_WINDOW_INVALID")
    supplied = value.get("artifact_hash")
    if not isinstance(supplied, str) or len(supplied) != 64:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_ARTIFACT_HASH_INVALID")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    # The file hash is the byte-level identity; artifact_hash is the canonical payload identity.
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise Stage2AuthorizationError("STAGE2_AUTHORIZATION_ARTIFACT_HASH_MISMATCH")
    return {**value, "file_sha256": observed}


__all__ = ["AUTHORIZATION_REF", "EXCLUDED_PCI", "USER_AUTHORIZATION", "Stage2AuthorizationError", "load_stage2_authorization"]
