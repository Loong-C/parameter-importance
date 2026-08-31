from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Final, Iterator

from .atomic import atomic_write_bytes, sha256_file, stable_json_bytes, stable_json_hash


SCHEMA_VERSION: Final = "stage0-asset-manifest-v1"
VERIFICATION_SCHEMA_VERSION: Final = "stage0-asset-verification-v1"
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


class AssetManifestError(ValueError):
    """Base class for an invalid or unusable asset manifest."""


class AssetEncodingError(AssetManifestError):
    """Raised when a manifest is not strict UTF-8 JSON without a BOM."""


class AssetValidationError(AssetManifestError):
    """Raised when a decoded manifest violates the Stage 0 contract."""


class AssetVerificationError(AssetManifestError):
    """Raised when files do not match a valid manifest."""


class AssetNotReadyError(AssetManifestError):
    """Raised when a non-ready manifest is passed to the runtime resolver."""


class AssetType(StrEnum):
    MODEL = "model"
    TOKENIZER = "tokenizer"
    DATASET = "dataset"
    SOURCE = "source"


class AssetState(StrEnum):
    DOWNLOADING = "downloading"
    DOWNLOADED = "downloaded"
    VERIFIED = "verified"
    READY = "ready"
    INVALID = "invalid"


class AssetActorRole(StrEnum):
    FETCHER = "fetcher"
    VERIFIER = "verifier"
    GATE = "gate"


ASSET_TRANSITIONS: Final[dict[AssetState, frozenset[AssetState]]] = {
    AssetState.DOWNLOADING: frozenset({AssetState.DOWNLOADED}),
    AssetState.DOWNLOADED: frozenset({AssetState.VERIFIED, AssetState.INVALID}),
    AssetState.VERIFIED: frozenset({AssetState.READY, AssetState.INVALID}),
    # A later integrity audit may invalidate an admitted asset, but a new
    # candidate manifest must be published rather than overwriting history.
    AssetState.READY: frozenset({AssetState.INVALID}),
    AssetState.INVALID: frozenset(),
}
ROLE_TRANSITIONS: Final[
    dict[AssetActorRole, frozenset[tuple[AssetState, AssetState]]]
] = {
    AssetActorRole.FETCHER: frozenset(
        {(AssetState.DOWNLOADING, AssetState.DOWNLOADED)}
    ),
    AssetActorRole.VERIFIER: frozenset(
        {
            (AssetState.DOWNLOADED, AssetState.VERIFIED),
            (AssetState.DOWNLOADED, AssetState.INVALID),
            (AssetState.VERIFIED, AssetState.INVALID),
        }
    ),
    AssetActorRole.GATE: frozenset(
        {
            (AssetState.VERIFIED, AssetState.READY),
            (AssetState.READY, AssetState.INVALID),
        }
    ),
}

_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "state",
        "name",
        "source",
        "revision",
        "files",
        "state_history",
        "created_at",
        "generator_version",
        "metadata",
    }
)
_REQUIRED_TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "asset_id",
        "asset_type",
        "state",
        "name",
        "source",
        "revision",
        "files",
        "state_history",
        "created_at",
        "generator_version",
        "metadata",
    }
)
_FILE_FIELDS: Final = frozenset({"path", "size_bytes", "sha256", "role"})
_HISTORY_FIELDS: Final = frozenset(
    {
        "from",
        "to",
        "at",
        "actor",
        "actor_role",
        "evidence_ref",
        "summary",
    }
)
_TEMPORARY_PATH_TOKENS: Final = frozenset({"part", "partial", "lock", "tmp", "temp"})
_GENERIC_REVISIONS: Final = frozenset(
    {
        "unknown",
        "latest",
        "main",
        "master",
        "head",
        "default",
        "current",
        "none",
        "null",
        "unspecified",
        "na",
        "n/a",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_text(value: Any, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise AssetValidationError(f"{field} must be a non-empty string")
    if any(ord(character) < 32 for character in value):
        raise AssetValidationError(f"{field} contains a control character")
    return value


def _require_timestamp(value: Any, *, field: str) -> str:
    text = _require_text(value, field=field, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise AssetValidationError(f"{field} is not an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise AssetValidationError(f"{field} must include a timezone")
    return text


def _require_integer(value: Any, *, field: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "non-negative"
        raise AssetValidationError(f"{field} must be a {qualifier} integer")
    return value


def _validate_revision(value: Any) -> str:
    revision = _require_text(value, field="revision", maximum=512)
    if revision != revision.strip():
        raise AssetValidationError("revision may not contain surrounding whitespace")
    if revision.casefold() in _GENERIC_REVISIONS:
        raise AssetValidationError(
            f"revision must be immutable and specific, not {revision!r}"
        )
    return revision


def _validate_metadata(
    asset_type: AssetType,
    value: Any,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AssetValidationError("metadata must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AssetValidationError("metadata object keys must be strings")
    metadata = deepcopy(dict(value))
    try:
        stable_json_bytes(metadata)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("metadata is not canonical JSON data") from error

    if asset_type is AssetType.MODEL:
        _require_text(
            metadata.get("architecture"),
            field="metadata.architecture",
            maximum=256,
        )
        _require_integer(
            metadata.get("parameter_count"),
            field="metadata.parameter_count",
            minimum=1,
        )
        _require_text(metadata.get("dtype"), field="metadata.dtype", maximum=128)
        _require_text(
            metadata.get("initialization_id"),
            field="metadata.initialization_id",
            maximum=512,
        )
    elif asset_type is AssetType.TOKENIZER:
        _require_text(
            metadata.get("tokenizer_class"),
            field="metadata.tokenizer_class",
            maximum=256,
        )
        _require_integer(
            metadata.get("vocab_size"),
            field="metadata.vocab_size",
            minimum=1,
        )
        special_tokens = metadata.get("special_tokens")
        if not isinstance(special_tokens, Mapping):
            raise AssetValidationError("metadata.special_tokens must be an object")
        if any(not isinstance(key, str) for key in special_tokens):
            raise AssetValidationError(
                "metadata.special_tokens object keys must be strings"
            )
        _require_text(
            metadata.get("normalization"),
            field="metadata.normalization",
            maximum=256,
        )
    elif asset_type is AssetType.DATASET:
        splits = metadata.get("splits")
        if not isinstance(splits, Mapping) or not splits:
            raise AssetValidationError("metadata.splits must be a non-empty object")
        for split_name, split in splits.items():
            normalized_name = _require_text(
                split_name,
                field="metadata.splits key",
                maximum=128,
            )
            if not isinstance(split, Mapping):
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name} must be an object"
                )
            _require_integer(
                split.get("sample_count"),
                field=f"metadata.splits.{normalized_name}.sample_count",
                minimum=0,
            )
            fields = split.get("fields")
            if not isinstance(fields, list) or not fields:
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name}.fields must be a non-empty array"
                )
            normalized_fields = [
                _require_text(
                    field_name,
                    field=f"metadata.splits.{normalized_name}.fields",
                    maximum=128,
                )
                for field_name in fields
            ]
            if len(normalized_fields) != len(set(normalized_fields)):
                raise AssetValidationError(
                    f"metadata.splits.{normalized_name}.fields must be unique"
                )
        _require_text(
            metadata.get("preprocessing_version"),
            field="metadata.preprocessing_version",
            maximum=256,
        )
    else:
        _require_text(
            metadata.get("source_kind"),
            field="metadata.source_kind",
            maximum=128,
        )
        _require_text(
            metadata.get("license"),
            field="metadata.license",
            maximum=256,
        )
    return metadata


def _validate_evidence_ref(
    value: Any,
    *,
    field: str,
    required: bool,
) -> str | None:
    if value is None:
        if required:
            raise AssetValidationError(f"{field} requires a non-empty evidence_ref")
        return None
    text = _require_text(value, field=field, maximum=4096)
    if "?" in text:
        raise AssetValidationError(
            f"{field} must be a stable evidence reference, not a query or signed URL"
        )
    return text


def validate_asset_path(value: str) -> str:
    """Return a normalized safe manifest-relative POSIX path.

    Asset paths are deliberately platform-neutral.  Absolute paths, path
    traversal, Windows separators/drive syntax, and temporary/lock suffixes
    are rejected before any filesystem access occurs.
    """

    text = _require_text(value, field="file.path", maximum=4096)
    if "\\" in text or ":" in text:
        raise AssetValidationError(f"Unsafe asset path syntax: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or str(path) != text or not path.parts:
        raise AssetValidationError(f"Asset path must be normalized and relative: {text!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AssetValidationError(f"Asset path contains traversal: {text!r}")
    for part in path.parts:
        tokens = {token for token in re.split(r"[._-]+", part.casefold()) if token}
        if tokens & _TEMPORARY_PATH_TOKENS:
            raise AssetValidationError(
                f"Asset path references a temporary or lock object: {text!r}"
            )
    return text


@dataclass(frozen=True, slots=True)
class AssetFile:
    path: str
    size_bytes: int
    sha256: str
    role: str | None = None

    @classmethod
    def from_value(cls, value: "AssetFile | Mapping[str, Any]") -> "AssetFile":
        if isinstance(value, cls):
            candidate = value
        elif isinstance(value, Mapping):
            extra = set(value) - _FILE_FIELDS
            missing = {"path", "size_bytes", "sha256"} - set(value)
            if missing or extra:
                raise AssetValidationError(
                    f"Invalid file descriptor fields; missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            size = value["size_bytes"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise AssetValidationError("file.size_bytes must be a non-negative integer")
            digest = value["sha256"]
            if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
                raise AssetValidationError("file.sha256 must be 64 lowercase hex characters")
            role_value = value.get("role")
            role = (
                None
                if role_value is None
                else _require_text(role_value, field="file.role", maximum=128)
            )
            candidate = cls(
                path=validate_asset_path(value["path"]),
                size_bytes=size,
                sha256=digest,
                role=role,
            )
        else:
            raise AssetValidationError("Each files entry must be an object")

        # Revalidate dataclass instances so constructing AssetFile directly
        # cannot bypass the same contract.
        validate_asset_path(candidate.path)
        if (
            isinstance(candidate.size_bytes, bool)
            or not isinstance(candidate.size_bytes, int)
            or candidate.size_bytes < 0
        ):
            raise AssetValidationError("file.size_bytes must be a non-negative integer")
        if not SHA256_PATTERN.fullmatch(candidate.sha256):
            raise AssetValidationError("file.sha256 must be 64 lowercase hex characters")
        if candidate.role is not None:
            _require_text(candidate.role, field="file.role", maximum=128)
        return candidate

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }
        if self.role is not None:
            value["role"] = self.role
        return value


def _normalize_files(
    files: Iterable[AssetFile | Mapping[str, Any]],
) -> list[AssetFile]:
    normalized = [AssetFile.from_value(value) for value in files]
    if not normalized:
        raise AssetValidationError("files must contain at least one final object")
    normalized.sort(key=lambda item: item.path)
    lowered = [item.path.casefold() for item in normalized]
    if len(lowered) != len(set(lowered)):
        raise AssetValidationError("files contains duplicate or case-colliding paths")
    return normalized


def compute_asset_id(
    manifest: Mapping[str, Any] | None = None,
    *,
    asset_type: AssetType | str | None = None,
    name: str | None = None,
    source: str | None = None,
    revision: str | None = None,
    files: Iterable[AssetFile | Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> str:
    """Compute the stable logical identity of an immutable asset.

    State, timestamps, local roots, and audit actors are intentionally absent
    from the identity.  File order is normalized, so equivalent inventories
    produce the same ID on every platform.
    """

    if manifest is not None:
        if any(
            value is not None
            for value in (asset_type, name, source, revision, files, metadata)
        ):
            raise TypeError("Pass either manifest or explicit identity fields, not both")
        asset_type = manifest.get("asset_type")
        name = manifest.get("name")
        source = manifest.get("source")
        revision = manifest.get("revision")
        files = manifest.get("files")
        metadata = manifest.get("metadata")
    if files is None:
        raise AssetValidationError("files is required to compute asset_id")
    if metadata is None:
        raise AssetValidationError("metadata is required to compute asset_id")
    try:
        normalized_type = AssetType(asset_type)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown asset_type: {asset_type!r}") from error
    normalized_name = _require_text(name, field="name", maximum=256)
    normalized_source = _require_text(source, field="source", maximum=512)
    normalized_revision = _validate_revision(revision)
    if "?" in normalized_source:
        raise AssetValidationError(
            "source must be a stable identifier, not a query or signed URL"
        )
    identity = {
        "asset_type": normalized_type.value,
        "name": normalized_name,
        "source": normalized_source,
        "revision": normalized_revision,
        "files": [item.as_dict() for item in _normalize_files(files)],
        "metadata": _validate_metadata(normalized_type, metadata),
    }
    return stable_json_hash(identity)


def validate_state_transition(
    previous: AssetState | str,
    target: AssetState | str,
    *,
    actor_role: AssetActorRole | str,
) -> None:
    try:
        current_state = AssetState(previous)
        target_state = AssetState(target)
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown asset state or actor_role") from error
    if target_state not in ASSET_TRANSITIONS[current_state]:
        raise AssetValidationError(
            f"Forbidden asset state transition: {current_state.value} -> "
            f"{target_state.value}"
        )
    if (current_state, target_state) not in ROLE_TRANSITIONS[normalized_role]:
        raise AssetValidationError(
            f"actor_role={normalized_role.value} is not authorized for "
            f"{current_state.value} -> {target_state.value}"
        )


def _validate_state_history(value: Any, final_state: AssetState) -> None:
    if not isinstance(value, list) or not value:
        raise AssetValidationError("state_history must be a non-empty array")
    previous: AssetState | None = None
    for index, event in enumerate(value):
        if not isinstance(event, Mapping):
            raise AssetValidationError("state_history entries must be objects")
        missing = _HISTORY_FIELDS - set(event)
        extra = set(event) - _HISTORY_FIELDS
        if missing or extra:
            raise AssetValidationError(
                f"Invalid state_history[{index}] fields; missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        raw_from = event["from"]
        try:
            event_from = None if raw_from is None else AssetState(raw_from)
            event_to = AssetState(event["to"])
            actor_role = AssetActorRole(event["actor_role"])
        except (TypeError, ValueError) as error:
            raise AssetValidationError(
                f"state_history[{index}] contains an unknown state or actor_role"
            ) from error
        if index == 0:
            if (
                event_from is not None
                or event_to is not AssetState.DOWNLOADING
                or actor_role is not AssetActorRole.FETCHER
            ):
                raise AssetValidationError(
                    "state_history must begin with a fetcher null -> downloading event"
                )
        else:
            if event_from is not previous:
                raise AssetValidationError(
                    f"state_history[{index}] does not continue the previous state"
                )
            validate_state_transition(
                event_from,
                event_to,
                actor_role=actor_role,
            )
        _require_timestamp(event["at"], field=f"state_history[{index}].at")
        _require_text(event["actor"], field=f"state_history[{index}].actor", maximum=256)
        _validate_evidence_ref(
            event["evidence_ref"],
            field=f"state_history[{index}]",
            required=event_to
            in {AssetState.VERIFIED, AssetState.READY, AssetState.INVALID},
        )
        if not isinstance(event["summary"], str):
            raise AssetValidationError(f"state_history[{index}].summary must be a string")
        previous = event_to
    if previous is not final_state:
        raise AssetValidationError(
            "state does not match the final state_history transition"
        )


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest, Mapping):
        raise AssetValidationError("Manifest root must be an object")
    missing = _REQUIRED_TOP_LEVEL_FIELDS - set(manifest)
    extra = set(manifest) - _TOP_LEVEL_FIELDS
    if missing or extra:
        raise AssetValidationError(
            f"Invalid manifest fields; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise AssetValidationError(
            f"Unsupported schema_version: {manifest['schema_version']!r}"
        )
    try:
        state = AssetState(manifest["state"])
        asset_type = AssetType(manifest["asset_type"])
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown asset_type or state") from error
    _require_text(manifest["name"], field="name", maximum=256)
    source = _require_text(manifest["source"], field="source", maximum=512)
    if "?" in source:
        raise AssetValidationError(
            "source must be a stable identifier, not a query or signed URL"
        )
    _validate_revision(manifest["revision"])
    _require_timestamp(manifest["created_at"], field="created_at")
    _require_text(
        manifest["generator_version"], field="generator_version", maximum=256
    )
    if not isinstance(manifest["files"], list):
        raise AssetValidationError("files must be an array")
    _normalize_files(manifest["files"])
    _validate_state_history(manifest["state_history"], state)
    _validate_metadata(asset_type, manifest["metadata"])
    asset_id = manifest["asset_id"]
    if not isinstance(asset_id, str) or not SHA256_PATTERN.fullmatch(asset_id):
        raise AssetValidationError("asset_id must be 64 lowercase hex characters")
    expected = compute_asset_id(manifest)
    if asset_id != expected:
        raise AssetValidationError(
            f"asset_id mismatch: declared={asset_id}, computed={expected}"
        )


def build_manifest(
    *,
    asset_type: AssetType | str,
    name: str,
    source: str,
    revision: str,
    files: Iterable[AssetFile | Mapping[str, Any]],
    actor: str,
    actor_role: AssetActorRole | str,
    evidence_ref: str | None,
    generator_version: str,
    metadata: Mapping[str, Any],
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build the initial immutable-identity manifest in `downloading` state."""

    timestamp = created_at or _utc_now()
    normalized_files = _normalize_files(files)
    try:
        normalized_type = AssetType(asset_type)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown asset_type: {asset_type!r}") from error
    normalized_metadata = _validate_metadata(normalized_type, metadata)
    try:
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError(f"Unknown actor_role: {actor_role!r}") from error
    if normalized_role is not AssetActorRole.FETCHER:
        raise AssetValidationError(
            "Only actor_role=fetcher may create a downloading candidate"
        )
    _validate_evidence_ref(
        evidence_ref,
        field="candidate creation",
        required=False,
    )
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "asset_id": compute_asset_id(
            asset_type=asset_type,
            name=name,
            source=source,
            revision=revision,
            files=normalized_files,
            metadata=normalized_metadata,
        ),
        "asset_type": normalized_type.value,
        "state": AssetState.DOWNLOADING.value,
        "name": name,
        "source": source,
        "revision": revision,
        "files": [item.as_dict() for item in normalized_files],
        "state_history": [
            {
                "from": None,
                "to": AssetState.DOWNLOADING.value,
                "at": timestamp,
                "actor": actor,
                "actor_role": normalized_role.value,
                "evidence_ref": evidence_ref,
                "summary": "candidate created",
            }
        ],
        "created_at": timestamp,
        "generator_version": generator_version,
        "metadata": normalized_metadata,
    }
    validate_manifest(value)
    return value


def transition_manifest(
    manifest: Mapping[str, Any],
    target: AssetState | str,
    *,
    actor: str,
    actor_role: AssetActorRole | str,
    evidence_ref: str | None,
    summary: str,
    at: str | None = None,
) -> dict[str, Any]:
    """Return a new manifest value with one audited state transition."""

    validate_manifest(manifest)
    previous = AssetState(manifest["state"])
    try:
        target_state = AssetState(target)
        normalized_role = AssetActorRole(actor_role)
    except (TypeError, ValueError) as error:
        raise AssetValidationError("Unknown target state or actor_role") from error
    validate_state_transition(
        previous,
        target_state,
        actor_role=normalized_role,
    )
    _require_text(actor, field="actor", maximum=256)
    if not isinstance(summary, str):
        raise AssetValidationError("summary must be a string")
    timestamp = at or _utc_now()
    _require_timestamp(timestamp, field="at")
    _validate_evidence_ref(
        evidence_ref,
        field=f"{previous.value} -> {target_state.value}",
        required=target_state
        in {AssetState.VERIFIED, AssetState.READY, AssetState.INVALID},
    )
    updated = deepcopy(dict(manifest))
    updated["state"] = target_state.value
    updated["state_history"].append(
        {
            "from": previous.value,
            "to": target_state.value,
            "at": timestamp,
            "actor": actor,
            "actor_role": normalized_role.value,
            "evidence_ref": evidence_ref,
            "summary": summary,
        }
    )
    validate_manifest(updated)
    return updated


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AssetEncodingError(f"Duplicate JSON object key: {key!r}")
        value[key] = item
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and validate strict UTF-8 JSON, explicitly rejecting a BOM."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise AssetEncodingError(f"UTF-8 BOM is forbidden: {manifest_path}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise AssetEncodingError(f"Manifest is not UTF-8: {manifest_path}") from error

    def reject_constant(value: str) -> None:
        raise AssetEncodingError(f"Non-finite JSON number is forbidden: {value}")

    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except AssetEncodingError:
        raise
    except json.JSONDecodeError as error:
        raise AssetEncodingError(f"Invalid JSON: {manifest_path}: {error}") from error
    if not isinstance(decoded, dict):
        raise AssetValidationError("Manifest root must be an object")
    validate_manifest(decoded)
    return decoded


def _coerce_manifest(value: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return load_manifest(value)
    copied = deepcopy(dict(value))
    validate_manifest(copied)
    return copied


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _publish_new_file(target: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        # A hard-link publication is atomic and fails if the immutable target
        # already exists.  The temporary name is then removed precisely.
        os.link(temporary, target)
        _fsync_directory(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(
        getattr(path, "is_junction", lambda: False)()
    )


def _reject_symlink_chain(path: Path, *, field: str) -> None:
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise AssetValidationError(
                    f"{field} may not contain symlinks or junctions: {current}"
                )


def _approved_manifest_target(
    path: str | Path,
    manifest_root: str | Path,
) -> tuple[Path, Path]:
    supplied_root = Path(manifest_root)
    if ".." in supplied_root.parts:
        raise AssetValidationError("manifest_root may not contain '..'")
    root = Path(os.path.abspath(supplied_root))
    _reject_symlink_chain(root, field="manifest_root")
    if not root.exists() or not root.is_dir():
        raise AssetValidationError(
            f"manifest_root must be an existing directory: {supplied_root}"
        )

    supplied_target = Path(path)
    if ".." in supplied_target.parts:
        raise AssetValidationError("Manifest target may not contain '..'")
    target = (
        Path(os.path.abspath(supplied_target))
        if supplied_target.is_absolute()
        else Path(os.path.abspath(root / supplied_target))
    )
    try:
        target.relative_to(root)
    except ValueError as error:
        raise AssetValidationError(
            f"Manifest target escapes approved manifest_root: {target}"
        ) from error
    validate_asset_path(target.name)
    if not target.parent.exists() or not target.parent.is_dir():
        raise AssetValidationError(
            f"Manifest target parent must already exist: {target.parent}"
        )
    _reject_symlink_chain(target.parent, field="Manifest target parent")
    resolved_root = root.resolve(strict=True)
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(resolved_root)
    except ValueError as error:
        raise AssetValidationError(
            f"Manifest target parent escapes approved manifest_root: {target.parent}"
        ) from error
    if target.exists() or _is_link_like(target):
        if _is_link_like(target):
            raise AssetValidationError(f"Manifest target may not be a symlink: {target}")
    return target, resolved_root


@contextmanager
def _advisory_manifest_lock(target: Path) -> Iterator[None]:
    """Hold a persistent per-target advisory lock across a CAS replacement."""

    lock_path = target.parent / f".{target.name}.publish.lock"
    if _is_link_like(lock_path):
        raise AssetValidationError(f"Manifest lock may not be a symlink: {lock_path}")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, 0o644)
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise AssetValidationError(
                f"Manifest lock must be a single-link regular file: {lock_path}"
            )
        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
            raise AssetValidationError(
                f"Manifest lock has an unexpected owner: {lock_path}"
            )
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_replacement(previous: Mapping[str, Any], value: Mapping[str, Any]) -> None:
    previous_state = AssetState(previous["state"])
    if previous_state is AssetState.READY:
        raise AssetValidationError(
            "A published ready manifest is immutable at its existing path; "
            "publish the invalidation record to a new path"
        )
    if previous_state is AssetState.INVALID:
        raise AssetValidationError(
            "A published invalid manifest is terminal and cannot be replaced"
        )
    previous_history = previous["state_history"]
    next_history = value["state_history"]
    if value["asset_id"] != previous["asset_id"]:
        raise AssetValidationError("Replacement cannot change asset_id")
    if len(next_history) <= len(previous_history) or (
        next_history[: len(previous_history)] != previous_history
    ):
        raise AssetValidationError(
            "Replacement must append to the existing state_history"
        )
    immutable_previous = {
        key: item
        for key, item in previous.items()
        if key not in {"state", "state_history"}
    }
    immutable_next = {
        key: item
        for key, item in value.items()
        if key not in {"state", "state_history"}
    }
    if immutable_next != immutable_previous:
        raise AssetValidationError(
            "Replacement cannot rewrite identity or manifest metadata"
        )


def publish_manifest_atomic(
    path: str | Path,
    manifest: Mapping[str, Any],
    *,
    manifest_root: str | Path,
    allow_replace: bool = False,
    expected_previous_sha256: str | None = None,
) -> Path:
    """Validate and atomically publish canonical UTF-8 JSON.

    Historical manifests are immutable by default.  `allow_replace` exists for
    an audited state advance of the same candidate, not for changing identity,
    rewriting metadata, or flipping an invalid candidate back to success.
    """

    value = deepcopy(dict(manifest))
    validate_manifest(value)
    target, _ = _approved_manifest_target(path, manifest_root)
    payload = stable_json_bytes(value)
    if payload.startswith(b"\xef\xbb\xbf"):
        raise AssertionError("Canonical JSON encoder unexpectedly emitted a BOM")
    if expected_previous_sha256 is not None and (
        not isinstance(expected_previous_sha256, str)
        or not SHA256_PATTERN.fullmatch(expected_previous_sha256)
    ):
        raise AssetValidationError(
            "expected_previous_sha256 must be 64 lowercase hex characters"
        )
    if expected_previous_sha256 is not None and not allow_replace:
        raise AssetValidationError(
            "expected_previous_sha256 is only valid with allow_replace=True"
        )

    if allow_replace:
        optimistic_exists = target.exists()
        optimistic_digest = sha256_file(target) if optimistic_exists else None
        with _advisory_manifest_lock(target):
            if _is_link_like(target):
                raise AssetValidationError(
                    f"Manifest target may not become a symlink: {target}"
                )
            current_exists = target.exists()
            current_digest = sha256_file(target) if current_exists else None
            if (optimistic_exists, optimistic_digest) != (
                current_exists,
                current_digest,
            ):
                raise AssetValidationError(
                    "Concurrent manifest publication detected during CAS"
                )
            if not current_exists:
                if expected_previous_sha256 is not None:
                    raise AssetValidationError(
                        "Stale expected_previous_sha256: target does not exist"
                    )
                _publish_new_file(target, payload)
            else:
                if expected_previous_sha256 is None:
                    raise AssetValidationError(
                        "allow_replace requires expected_previous_sha256 for an "
                        "existing target"
                    )
                if current_digest != expected_previous_sha256:
                    raise AssetValidationError(
                        "Stale expected_previous_sha256 for manifest replacement"
                    )
                previous = load_manifest(target)
                if sha256_file(target) != current_digest:
                    raise AssetValidationError(
                        "Concurrent manifest publication detected during CAS read"
                    )
                _validate_replacement(previous, value)
                atomic_write_bytes(target, payload)
    else:
        try:
            _publish_new_file(target, payload)
        except FileExistsError as error:
            raise FileExistsError(f"Manifest already exists: {target}") from error
    return target


@dataclass(frozen=True, slots=True)
class ResolvedAssetFile:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str
    role: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    asset_id: str
    asset_type: AssetType
    name: str
    revision: str
    root: Path
    files: tuple[ResolvedAssetFile, ...]

    def path_for(self, relative_path: str) -> Path:
        normalized = validate_asset_path(relative_path)
        for item in self.files:
            if item.relative_path == normalized:
                return item.path
        raise KeyError(relative_path)


def _resolve_file(root: Path, descriptor: AssetFile) -> Path:
    relative = PurePosixPath(descriptor.path)
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise AssetVerificationError(
                f"Symlinks are forbidden in ready assets: {descriptor.path}"
            )
    if not candidate.exists():
        raise AssetVerificationError(f"Asset file is missing: {descriptor.path}")
    if not candidate.is_file():
        raise AssetVerificationError(f"Asset path is not a regular file: {descriptor.path}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise AssetVerificationError(
            f"Asset file escapes its root: {descriptor.path}"
        ) from error
    return resolved


def _verify_files(
    manifest: Mapping[str, Any],
    asset_root: str | Path,
) -> tuple[ResolvedAssetFile, ...]:
    supplied_root = Path(asset_root)
    if supplied_root.is_symlink():
        raise AssetVerificationError(f"Asset root may not be a symlink: {supplied_root}")
    try:
        root = supplied_root.resolve(strict=True)
    except FileNotFoundError as error:
        raise AssetVerificationError(f"Asset root is missing: {supplied_root}") from error
    if not root.is_dir():
        raise AssetVerificationError(f"Asset root is not a directory: {root}")
    resolved_files: list[ResolvedAssetFile] = []
    for descriptor in _normalize_files(manifest["files"]):
        path = _resolve_file(root, descriptor)
        actual_size = path.stat().st_size
        if actual_size != descriptor.size_bytes:
            raise AssetVerificationError(
                f"Size mismatch for {descriptor.path}: expected "
                f"{descriptor.size_bytes}, got {actual_size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != descriptor.sha256:
            raise AssetVerificationError(
                f"SHA-256 mismatch for {descriptor.path}: expected "
                f"{descriptor.sha256}, got {actual_hash}"
            )
        resolved_files.append(
            ResolvedAssetFile(
                relative_path=descriptor.path,
                path=path,
                size_bytes=descriptor.size_bytes,
                sha256=descriptor.sha256,
                role=descriptor.role,
            )
        )
    return tuple(resolved_files)


def verify_only(
    manifest: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
) -> dict[str, Any]:
    """Fully verify an existing candidate without writing files or state."""

    value = _coerce_manifest(manifest)
    state = AssetState(value["state"])
    if state in {AssetState.DOWNLOADING, AssetState.INVALID}:
        raise AssetVerificationError(
            f"verify-only refuses state={state.value}; acquisition must finish first"
        )
    files = _verify_files(value, asset_root)
    return {
        "schema_version": VERIFICATION_SCHEMA_VERSION,
        "asset_id": value["asset_id"],
        "state": state.value,
        "files_checked": len(files),
        "bytes_checked": sum(item.size_bytes for item in files),
        "ok": True,
    }


def resolve_ready_asset(
    manifest: Mapping[str, Any] | str | Path,
    asset_root: str | Path,
) -> ResolvedAsset:
    """Resolve a ready logical asset to safe final paths.

    READY resolution always performs full size and SHA-256 verification.
    """

    value = _coerce_manifest(manifest)
    state = AssetState(value["state"])
    if state is not AssetState.READY:
        raise AssetNotReadyError(
            f"Runtime resolution requires state=ready, got {state.value}"
        )
    supplied_root = Path(asset_root)
    files = _verify_files(value, supplied_root)
    resolved_root = supplied_root.resolve(strict=True)
    return ResolvedAsset(
        asset_id=value["asset_id"],
        asset_type=AssetType(value["asset_type"]),
        name=value["name"],
        revision=value["revision"],
        root=resolved_root,
        files=files,
    )


# Explicit aliases keep call sites readable without weakening the single
# implementation of parsing and resolution semantics.
load_asset_manifest = load_manifest
resolve_ready_manifest = resolve_ready_asset
