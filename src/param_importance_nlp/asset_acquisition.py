from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
import errno
import hashlib
import http.client
import math
import os
from pathlib import Path, PurePosixPath
import queue
import re
import stat
import threading
import time
from typing import Any, BinaryIO, Callable, Final, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_RANGE_PATTERN: Final = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
_SPEC_FIELDS: Final = frozenset(
    {
        "schema_version",
        "source_id",
        "revision",
        "expected_size",
        "expected_sha256",
    }
)
_TEMPORARY_PATH_TOKENS: Final = frozenset(
    {"part", "partial", "lock", "tmp", "temp"}
)
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


class AcquisitionStatus(StrEnum):
    DOWNLOADED = "downloaded"
    ALREADY_READY = "already_ready"
    PUBLISHED_BY_PEER = "published_by_peer"


class AcquisitionFailureCode(StrEnum):
    INVALID_RUNTIME_URL = "INVALID_RUNTIME_URL"
    LOCK_TIMEOUT = "LOCK_TIMEOUT"
    TARGET_CONFLICT = "TARGET_CONFLICT"
    LOCAL_STATE_INVALID = "LOCAL_STATE_INVALID"
    LOCAL_IO_ERROR = "LOCAL_IO_ERROR"
    HTTP_STATUS = "HTTP_STATUS"
    NETWORK_ERROR = "NETWORK_ERROR"
    OVERALL_TIMEOUT = "OVERALL_TIMEOUT"
    RANGE_NOT_HONORED = "RANGE_NOT_HONORED"
    CONTENT_RANGE_INVALID = "CONTENT_RANGE_INVALID"
    CONTENT_LENGTH_INVALID = "CONTENT_LENGTH_INVALID"
    TRANSFER_INCOMPLETE = "TRANSFER_INCOMPLETE"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    HASH_MISMATCH = "HASH_MISMATCH"


_FAILURE_MESSAGES: Final[dict[AcquisitionFailureCode, str]] = {
    AcquisitionFailureCode.INVALID_RUNTIME_URL: "runtime URL is not a valid HTTP endpoint",
    AcquisitionFailureCode.LOCK_TIMEOUT: "timed out waiting for the object acquisition lock",
    AcquisitionFailureCode.TARGET_CONFLICT: "the immutable target exists with a different identity",
    AcquisitionFailureCode.LOCAL_STATE_INVALID: "a local acquisition object has an unsafe type or link count",
    AcquisitionFailureCode.LOCAL_IO_ERROR: "a local filesystem operation failed",
    AcquisitionFailureCode.HTTP_STATUS: "the HTTP endpoint returned an unsuccessful status",
    AcquisitionFailureCode.NETWORK_ERROR: "the HTTP transfer failed before completion",
    AcquisitionFailureCode.OVERALL_TIMEOUT: "the bounded acquisition deadline expired",
    AcquisitionFailureCode.RANGE_NOT_HONORED: "the server did not honor the requested byte range",
    AcquisitionFailureCode.CONTENT_RANGE_INVALID: "the server returned an invalid Content-Range",
    AcquisitionFailureCode.CONTENT_LENGTH_INVALID: "the server returned an invalid Content-Length",
    AcquisitionFailureCode.TRANSFER_INCOMPLETE: "the response ended before its declared byte range completed",
    AcquisitionFailureCode.SIZE_MISMATCH: "the completed object size does not match the fixed specification",
    AcquisitionFailureCode.HASH_MISMATCH: "the completed object SHA-256 does not match the fixed specification",
}


def _fixed_text(value: str, *, field: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(f"{field} must be normalized text")
    if "?" in value or "://" in value:
        raise ValueError(f"{field} must be a stable identifier, not a runtime URL")
    return value


@dataclass(frozen=True, slots=True)
class AssetObjectSpec:
    """Immutable identity expected from a runtime-only HTTP endpoint."""

    source_id: str
    revision: str
    expected_size: int
    expected_sha256: str

    def __post_init__(self) -> None:
        source_id = _fixed_text(self.source_id, field="source_id")
        revision = _fixed_text(self.revision, field="revision")
        if revision.casefold() in _GENERIC_REVISIONS:
            raise ValueError("revision must be immutable and specific")
        if isinstance(self.expected_size, bool) or not isinstance(
            self.expected_size, int
        ) or self.expected_size < 0:
            raise ValueError("expected_size must be a non-negative integer")
        if not isinstance(self.expected_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be 64 lowercase hex characters")
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "revision", revision)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AssetObjectSpec":
        if not isinstance(value, Mapping):
            raise ValueError("asset object spec must be a JSON object")
        missing = _SPEC_FIELDS - set(value)
        extra = set(value) - _SPEC_FIELDS
        if missing or extra:
            raise ValueError(
                f"asset object spec fields are invalid; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if value["schema_version"] != "stage0-http-object-spec-v1":
            raise ValueError("unsupported asset object spec schema_version")
        return cls(
            source_id=value["source_id"],
            revision=value["revision"],
            expected_size=value["expected_size"],
            expected_sha256=value["expected_sha256"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-http-object-spec-v1",
            "source_id": self.source_id,
            "revision": self.revision,
            "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionPolicy:
    max_attempts: int = 4
    request_timeout_seconds: float = 30.0
    overall_timeout_seconds: float = 600.0
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    lock_timeout_seconds: float = 30.0
    lock_poll_interval_seconds: float = 0.05
    chunk_size: int = 1024 * 1024

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_attempts, bool)
            or not isinstance(self.max_attempts, int)
            or self.max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        if (
            isinstance(self.chunk_size, bool)
            or not isinstance(self.chunk_size, int)
            or self.chunk_size < 1
        ):
            raise ValueError("chunk_size must be a positive integer")
        for field_name in (
            "request_timeout_seconds",
            "overall_timeout_seconds",
            "lock_timeout_seconds",
            "lock_poll_interval_seconds",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be a positive finite number")
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError(f"{field_name} must be a positive finite number")
        for field_name in ("initial_backoff_seconds", "max_backoff_seconds"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be a non-negative finite number")
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative finite number")
        if self.max_backoff_seconds < self.initial_backoff_seconds:
            raise ValueError(
                "max_backoff_seconds may not be less than initial_backoff_seconds"
            )


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    status: AcquisitionStatus
    source_id: str
    revision: str
    size_bytes: int
    sha256: str
    attempts: int
    resumed: bool
    network_accessed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-asset-acquisition-result-v1",
            "status": self.status.value,
            "source_id": self.source_id,
            "revision": self.revision,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "attempts": self.attempts,
            "resumed": self.resumed,
            "network_accessed": self.network_accessed,
        }


@dataclass(frozen=True, slots=True)
class StreamReceptionPlan:
    """Receiver-side handshake for an externally relayed immutable object."""

    source_id: str
    revision: str
    expected_size: int
    expected_sha256: str
    offset: int
    already_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-asset-stream-reception-plan-v1",
            "source_id": self.source_id,
            "revision": self.revision,
            "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
            "offset": self.offset,
            "already_ready": self.already_ready,
        }


@dataclass(frozen=True, slots=True)
class LegacyAcquisitionState:
    """Read-only view of the former asset-adjacent relay checkpoint."""

    source_id: str
    legacy_part_exists: bool
    legacy_part_size: int | None
    legacy_lock_exists: bool
    legacy_lock_state: str
    central_part_exists: bool
    final_state: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-legacy-acquisition-state-v1",
            "source_id": self.source_id,
            "legacy_part_exists": self.legacy_part_exists,
            "legacy_part_size": self.legacy_part_size,
            "legacy_lock_exists": self.legacy_lock_exists,
            "legacy_lock_state": self.legacy_lock_state,
            "central_part_exists": self.central_part_exists,
            "final_state": self.final_state,
        }


@dataclass(frozen=True, slots=True)
class LegacyMigrationResult:
    source_id: str
    status: str
    migrated_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-legacy-acquisition-migration-v1",
            "source_id": self.source_id,
            "status": self.status,
            "migrated_size": self.migrated_size,
        }


@dataclass(frozen=True, slots=True)
class AcquisitionFailureReport:
    code: AcquisitionFailureCode
    retryable: bool
    source_id: str
    revision: str
    expected_size: int
    expected_sha256: str
    attempts: int
    observed_size: int | None = None
    http_status: int | None = None
    exhausted: bool = False

    @property
    def message(self) -> str:
        return _FAILURE_MESSAGES[self.code]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stage0-asset-acquisition-failure-v1",
            "status": "failed",
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "source_id": self.source_id,
            "revision": self.revision,
            "expected_size": self.expected_size,
            "expected_sha256": self.expected_sha256,
            "attempts": self.attempts,
            "observed_size": self.observed_size,
            "http_status": self.http_status,
            "exhausted": self.exhausted,
        }


class AssetAcquisitionError(RuntimeError):
    def __init__(self, report: AcquisitionFailureReport) -> None:
        self.report = report
        super().__init__(f"{report.code.value}: {report.message}")


class _AttemptFailure(Exception):
    def __init__(
        self,
        code: AcquisitionFailureCode,
        retryable: bool,
        observed_size: int | None = None,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.observed_size = observed_size
        self.http_status = http_status
        super().__init__(code.value)


@dataclass(frozen=True, slots=True)
class _TransferResult:
    resumed_from: int
    network_accessed: bool


def _central_state_path(
    target: str | Path,
    *,
    data_root: str | Path,
    directory: str,
    suffix: str,
) -> Path:
    root = Path(os.path.abspath(Path(data_root)))
    destination = Path(os.path.abspath(Path(target)))
    if not root.is_absolute():  # pragma: no cover - abspath is always absolute
        raise ValueError("data_root must be absolute")
    try:
        relative = destination.relative_to(root)
    except ValueError as error:
        raise ValueError("asset target must remain below DATA_ROOT") from error
    if not relative.parts or relative.parts[0] not in {"models", "datasets"}:
        raise ValueError("asset target must be below DATA_ROOT/models or datasets")
    identity = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()
    return root / directory / f"{identity}{suffix}"


def part_path_for(
    target: str | Path,
    *,
    data_root: str | Path | None = None,
) -> Path:
    destination = Path(target)
    if data_root is not None:
        return _central_state_path(
            destination,
            data_root=data_root,
            directory="tmp/g3-acquisition-parts",
            suffix=".part",
        )
    return destination.parent / f".{destination.name}.part"


def lock_path_for(
    target: str | Path,
    *,
    data_root: str | Path | None = None,
) -> Path:
    destination = Path(target)
    if data_root is not None:
        return _central_state_path(
            destination,
            data_root=data_root,
            directory="operations/g3-acquisition-locks",
            suffix=".acquire.lock",
        )
    return destination.parent / f".{destination.name}.acquire.lock"


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def resolve_approved_asset_target(
    asset_root: str | Path,
    relative_final_path: str,
) -> Path:
    """Resolve a manifest-safe final path beneath an existing approved root."""

    supplied_root = Path(asset_root)
    if ".." in supplied_root.parts:
        raise ValueError("asset_root may not contain parent traversal")
    root = Path(os.path.abspath(supplied_root))
    current = Path(root.anchor)
    for part in root.parts[1:]:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current):
                raise ValueError("asset_root may not contain symlinks or junctions")
    if not root.exists() or not root.is_dir():
        raise ValueError("asset_root must be an existing directory")

    text = _fixed_text(
        relative_final_path,
        field="relative_final_path",
        maximum=4096,
    )
    if "\\" in text or ":" in text:
        raise ValueError("relative_final_path must use relative POSIX syntax")
    logical = PurePosixPath(text)
    if logical.is_absolute() or str(logical) != text or not logical.parts:
        raise ValueError("relative_final_path must be normalized and relative")
    if any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError("relative_final_path contains traversal")
    for part in logical.parts:
        tokens = {
            token for token in re.split(r"[._-]+", part.casefold()) if token
        }
        if tokens & _TEMPORARY_PATH_TOKENS:
            raise ValueError("relative_final_path names a temporary or lock object")

    target = root.joinpath(*logical.parts)
    current = root
    for part in logical.parts[:-1]:
        current = current / part
        if current.exists() or _is_link_like(current):
            if _is_link_like(current) or not current.is_dir():
                raise ValueError("relative_final_path has an unsafe parent chain")
    return target


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_file(path: Path, *, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_regular_object(path: Path, *, require_single_link: bool) -> os.stat_result:
    if _is_link_like(path):
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    try:
        metadata = path.stat()
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    if not stat.S_ISREG(metadata.st_mode) or (
        require_single_link and metadata.st_nlink != 1
    ):
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    return metadata


def _verify_final(
    target: Path,
    spec: AssetObjectSpec,
    *,
    chunk_size: int,
) -> bool:
    if not target.exists() and not _is_link_like(target):
        return False
    metadata = _validate_regular_object(target, require_single_link=False)
    if metadata.st_size != spec.expected_size:
        raise _AttemptFailure(
            AcquisitionFailureCode.TARGET_CONFLICT,
            False,
            observed_size=metadata.st_size,
        )
    try:
        digest = _sha256_file(target, chunk_size=chunk_size)
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    if digest != spec.expected_sha256:
        raise _AttemptFailure(
            AcquisitionFailureCode.TARGET_CONFLICT,
            False,
            observed_size=metadata.st_size,
        )
    return True


def _try_lock(descriptor: int) -> bool:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK, 13, 36}:
                return False
            raise
    else:
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN}:
                return False
            raise
    return True


def _unlock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _advisory_object_lock(
    target: Path,
    *,
    lock_path: Path | None = None,
    timeout_seconds: float,
    overall_deadline: float,
    poll_interval_seconds: float,
) -> Iterator[None]:
    lock_path = lock_path or lock_path_for(target)
    if _is_link_like(lock_path):
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o644)
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
        get_effective_uid = getattr(os, "geteuid", None)
        if get_effective_uid is not None and metadata.st_uid != get_effective_uid():
            raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
        if metadata.st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        lock_deadline = min(
            overall_deadline,
            time.monotonic() + timeout_seconds,
        )
        while not _try_lock(descriptor):
            remaining = lock_deadline - time.monotonic()
            if remaining <= 0:
                raise _AttemptFailure(AcquisitionFailureCode.LOCK_TIMEOUT, True)
            time.sleep(min(poll_interval_seconds, remaining))
        locked = True
        yield
    finally:
        if locked:
            _unlock(descriptor)
        os.close(descriptor)


def _open_existing_lock(path: Path) -> int:
    if _is_link_like(path):
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    flags = os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False) from None
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    return descriptor


@contextmanager
def _existing_advisory_lock(
    path: Path,
    *,
    timeout_seconds: float,
    overall_deadline: float,
    poll_interval_seconds: float,
) -> Iterator[None]:
    descriptor = _open_existing_lock(path)
    locked = False
    try:
        lock_deadline = min(overall_deadline, time.monotonic() + timeout_seconds)
        while not _try_lock(descriptor):
            remaining = lock_deadline - time.monotonic()
            if remaining <= 0:
                raise _AttemptFailure(AcquisitionFailureCode.LOCK_TIMEOUT, True)
            time.sleep(min(poll_interval_seconds, remaining))
        locked = True
        yield
    finally:
        if locked:
            _unlock(descriptor)
        os.close(descriptor)


def _inspect_existing_lock(path: Path) -> str:
    if not path.exists() and not _is_link_like(path):
        return "absent"
    try:
        descriptor = _open_existing_lock(path)
    except _AttemptFailure:
        return "unsafe"
    locked = False
    try:
        locked = _try_lock(descriptor)
        return "available" if locked else "held"
    except OSError:
        return "unsafe"
    finally:
        if locked:
            _unlock(descriptor)
        os.close(descriptor)


def _ensure_real_directory(path: Path, *, parent: Path) -> None:
    if _is_link_like(parent) or not parent.is_dir():
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    if path.exists() or _is_link_like(path):
        if _is_link_like(path) or not path.is_dir():
            raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
        return
    try:
        path.mkdir(mode=0o750)
    except FileExistsError:
        pass
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    if _is_link_like(path) or not path.is_dir():
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)


def _prepare_state_paths(
    target: Path,
    *,
    data_root: str | Path | None,
    allow_legacy_state: bool = False,
) -> tuple[Path, Path]:
    if data_root is None:
        return part_path_for(target), lock_path_for(target)
    root = Path(os.path.abspath(Path(data_root)))
    if _is_link_like(root) or not root.is_dir():
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    temporary_root = root / "tmp"
    operations_root = root / "operations"
    for required in (temporary_root, operations_root):
        if _is_link_like(required) or not required.is_dir():
            raise _AttemptFailure(
                AcquisitionFailureCode.LOCAL_STATE_INVALID,
                False,
            )
    part_root = temporary_root / "g3-acquisition-parts"
    lock_root = operations_root / "g3-acquisition-locks"
    _ensure_real_directory(part_root, parent=temporary_root)
    _ensure_real_directory(lock_root, parent=operations_root)
    if not allow_legacy_state:
        legacy_part = part_path_for(target)
        if legacy_part.exists() or _is_link_like(legacy_part):
            raise _AttemptFailure(
                AcquisitionFailureCode.LOCAL_STATE_INVALID,
                False,
            )
        legacy_lock = lock_path_for(target)
        legacy_lock_state = _inspect_existing_lock(legacy_lock)
        if legacy_lock_state in {"held", "unsafe"}:
            raise _AttemptFailure(
                AcquisitionFailureCode.LOCK_TIMEOUT
                if legacy_lock_state == "held"
                else AcquisitionFailureCode.LOCAL_STATE_INVALID,
                legacy_lock_state == "held",
            )
    return (
        part_path_for(target, data_root=root),
        lock_path_for(target, data_root=root),
    )


@contextmanager
def _open_part(path: Path) -> Iterator[Any]:
    if _is_link_like(path):
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o644)
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
        with os.fdopen(descriptor, "r+b", buffering=0) as handle:
            descriptor = -1
            yield handle
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _truncate_part(handle: Any) -> None:
    handle.seek(0)
    handle.truncate(0)
    handle.flush()
    os.fsync(handle.fileno())


def _bounded_call(
    operation: Callable[[], Any],
    *,
    overall_deadline: float,
    observed_size: int | None,
) -> Any:
    """Run a potentially blocking stream/callback operation under one deadline."""

    remaining = overall_deadline - time.monotonic()
    if remaining <= 0:
        raise _AttemptFailure(
            AcquisitionFailureCode.OVERALL_TIMEOUT,
            True,
            observed_size=observed_size,
        )
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, operation()))
        except BaseException as error:  # propagated in the caller thread
            outcome.put((False, error))

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    try:
        succeeded, value = outcome.get(timeout=remaining)
    except queue.Empty:
        raise _AttemptFailure(
            AcquisitionFailureCode.OVERALL_TIMEOUT,
            True,
            observed_size=observed_size,
        ) from None
    if not succeeded:
        raise value
    return value


def _validate_runtime_url(runtime_url: str) -> bool:
    if not isinstance(runtime_url, str) or not runtime_url:
        return False
    try:
        parsed = urlsplit(runtime_url)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _response_layout(
    *,
    status: int,
    headers: Any,
    offset: int,
    expected_size: int,
) -> int:
    raw_length = headers.get("Content-Length")
    try:
        content_length = int(raw_length)
    except (TypeError, ValueError) as error:
        raise _AttemptFailure(
            AcquisitionFailureCode.CONTENT_LENGTH_INVALID,
            True,
            observed_size=offset,
            http_status=status,
        ) from error
    if content_length < 0:
        raise _AttemptFailure(
            AcquisitionFailureCode.CONTENT_LENGTH_INVALID,
            True,
            observed_size=offset,
            http_status=status,
        )

    if offset == 0:
        if status != 200:
            raise _AttemptFailure(
                AcquisitionFailureCode.RANGE_NOT_HONORED,
                True,
                observed_size=offset,
                http_status=status,
            )
        if content_length != expected_size:
            raise _AttemptFailure(
                AcquisitionFailureCode.CONTENT_LENGTH_INVALID,
                True,
                observed_size=offset,
                http_status=status,
            )
        return content_length

    if status != 206:
        raise _AttemptFailure(
            AcquisitionFailureCode.RANGE_NOT_HONORED,
            True,
            observed_size=offset,
            http_status=status,
        )
    raw_range = headers.get("Content-Range")
    match = (
        _CONTENT_RANGE_PATTERN.fullmatch(raw_range.strip())
        if isinstance(raw_range, str)
        else None
    )
    if match is None:
        raise _AttemptFailure(
            AcquisitionFailureCode.CONTENT_RANGE_INVALID,
            True,
            observed_size=offset,
            http_status=status,
        )
    start, end, total = (int(value) for value in match.groups())
    if (
        start != offset
        or end < start
        or total != expected_size
        or end >= total
        or content_length != end - start + 1
    ):
        raise _AttemptFailure(
            AcquisitionFailureCode.CONTENT_RANGE_INVALID,
            True,
            observed_size=offset,
            http_status=status,
        )
    return content_length


def _download_once(
    *,
    spec: AssetObjectSpec,
    runtime_url: str,
    part_path: Path,
    policy: AcquisitionPolicy,
    overall_deadline: float,
) -> _TransferResult:
    with _open_part(part_path) as part:
        part.seek(0, os.SEEK_END)
        offset = part.tell()
        if offset > spec.expected_size:
            _truncate_part(part)
            offset = 0
        elif offset == spec.expected_size:
            return _TransferResult(resumed_from=offset, network_accessed=False)

        remaining_time = overall_deadline - time.monotonic()
        if remaining_time <= 0:
            raise _AttemptFailure(
                AcquisitionFailureCode.OVERALL_TIMEOUT,
                True,
                observed_size=offset,
            )
        headers = {
            "Accept-Encoding": "identity",
            "User-Agent": "param-importance-asset-acquisition/1",
        }
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = Request(runtime_url, headers=headers, method="GET")
        timeout = min(policy.request_timeout_seconds, remaining_time)
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as error:
            status = error.code
            error.close()
            retryable = status in {408, 425, 429} or 500 <= status <= 599
            raise _AttemptFailure(
                AcquisitionFailureCode.HTTP_STATUS,
                retryable,
                observed_size=offset,
                http_status=status,
            ) from None
        except (URLError, TimeoutError, OSError):
            raise _AttemptFailure(
                AcquisitionFailureCode.NETWORK_ERROR,
                True,
                observed_size=offset,
            ) from None

        received = 0
        try:
            with response:
                status = int(getattr(response, "status", response.getcode()))
                response_length = _response_layout(
                    status=status,
                    headers=response.headers,
                    offset=offset,
                    expected_size=spec.expected_size,
                )
                part.seek(offset)
                while received < response_length:
                    if time.monotonic() >= overall_deadline:
                        raise _AttemptFailure(
                            AcquisitionFailureCode.OVERALL_TIMEOUT,
                            True,
                            observed_size=offset + received,
                            http_status=status,
                        )
                    read_size = min(policy.chunk_size, response_length - received)
                    try:
                        chunk = _bounded_call(
                            lambda: response.read(read_size),
                            overall_deadline=overall_deadline,
                            observed_size=offset + received,
                        )
                    except http.client.IncompleteRead as error:
                        if error.partial:
                            part.write(error.partial)
                            received += len(error.partial)
                        part.flush()
                        os.fsync(part.fileno())
                        raise _AttemptFailure(
                            AcquisitionFailureCode.TRANSFER_INCOMPLETE,
                            True,
                            observed_size=offset + received,
                            http_status=status,
                        ) from None
                    except _AttemptFailure:
                        part.flush()
                        os.fsync(part.fileno())
                        raise
                    except (TimeoutError, OSError):
                        part.flush()
                        os.fsync(part.fileno())
                        raise _AttemptFailure(
                            AcquisitionFailureCode.NETWORK_ERROR,
                            True,
                            observed_size=offset + received,
                            http_status=status,
                        ) from None
                    if not chunk:
                        break
                    part.write(chunk)
                    received += len(chunk)
                part.flush()
                os.fsync(part.fileno())
        except _AttemptFailure:
            raise
        except (TimeoutError, OSError):
            part.flush()
            os.fsync(part.fileno())
            raise _AttemptFailure(
                AcquisitionFailureCode.NETWORK_ERROR,
                True,
                observed_size=offset + received,
            ) from None

        final_size = offset + received
        if received != response_length:
            raise _AttemptFailure(
                AcquisitionFailureCode.TRANSFER_INCOMPLETE,
                True,
                observed_size=final_size,
                http_status=status,
            )
        if final_size != spec.expected_size:
            raise _AttemptFailure(
                AcquisitionFailureCode.SIZE_MISMATCH,
                True,
                observed_size=final_size,
                http_status=status,
            )
        return _TransferResult(resumed_from=offset, network_accessed=True)


def _publish_no_clobber(
    *,
    part_path: Path,
    target: Path,
    spec: AssetObjectSpec,
    chunk_size: int,
) -> AcquisitionStatus:
    metadata = _validate_regular_object(part_path, require_single_link=True)
    if metadata.st_size != spec.expected_size:
        raise _AttemptFailure(
            AcquisitionFailureCode.SIZE_MISMATCH,
            True,
            observed_size=metadata.st_size,
        )
    try:
        candidate_digest = _sha256_file(part_path, chunk_size=chunk_size)
    except OSError as error:
        raise _AttemptFailure(
            AcquisitionFailureCode.LOCAL_IO_ERROR,
            False,
            observed_size=metadata.st_size,
        ) from error
    if candidate_digest != spec.expected_sha256:
        with _open_part(part_path) as part:
            _truncate_part(part)
        raise _AttemptFailure(
            AcquisitionFailureCode.HASH_MISMATCH,
            True,
            observed_size=metadata.st_size,
        )
    try:
        os.link(part_path, target, follow_symlinks=False)
    except FileExistsError:
        if _verify_final(target, spec, chunk_size=chunk_size):
            try:
                part_path.unlink(missing_ok=True)
            except OSError:
                pass
            return AcquisitionStatus.PUBLISHED_BY_PEER
        raise _AttemptFailure(AcquisitionFailureCode.TARGET_CONFLICT, False)
    except OSError as error:
        raise _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False) from error

    linked_metadata: os.stat_result | None = None
    try:
        linked_metadata = target.stat()
        if (
            _is_link_like(target)
            or not stat.S_ISREG(linked_metadata.st_mode)
            or linked_metadata.st_size != spec.expected_size
            or _sha256_file(target, chunk_size=chunk_size) != spec.expected_sha256
        ):
            raise _AttemptFailure(
                AcquisitionFailureCode.HASH_MISMATCH,
                True,
                observed_size=linked_metadata.st_size,
            )
    except (OSError, _AttemptFailure) as error:
        try:
            current = target.stat()
            if (
                linked_metadata is not None
                and current.st_dev == linked_metadata.st_dev
                and current.st_ino == linked_metadata.st_ino
            ):
                target.unlink(missing_ok=True)
                _fsync_directory(target.parent)
        except OSError:
            pass
        if isinstance(error, _AttemptFailure) and part_path.exists():
            try:
                with _open_part(part_path) as part:
                    _truncate_part(part)
            except _AttemptFailure:
                pass
        if isinstance(error, _AttemptFailure):
            raise
        raise _AttemptFailure(
            AcquisitionFailureCode.LOCAL_IO_ERROR,
            False,
            observed_size=spec.expected_size,
        ) from error
    try:
        part_path.unlink(missing_ok=True)
    finally:
        _fsync_directory(target.parent)
    return AcquisitionStatus.DOWNLOADED


def _failure_report(
    *,
    spec: AssetObjectSpec,
    failure: _AttemptFailure,
    attempts: int,
    exhausted: bool,
) -> AcquisitionFailureReport:
    return AcquisitionFailureReport(
        code=failure.code,
        retryable=failure.retryable,
        source_id=spec.source_id,
        revision=spec.revision,
        expected_size=spec.expected_size,
        expected_sha256=spec.expected_sha256,
        attempts=attempts,
        observed_size=failure.observed_size,
        http_status=failure.http_status,
        exhausted=exhausted,
    )


def inspect_legacy_acquisition_state(
    spec: AssetObjectSpec,
    target: str | Path,
    *,
    data_root: str | Path,
) -> LegacyAcquisitionState:
    """Inspect, without migrating or deleting, the pre-central-state checkpoint."""

    destination = Path(target)
    legacy_part = part_path_for(destination)
    legacy_lock = lock_path_for(destination)
    central_part = part_path_for(destination, data_root=data_root)
    part_exists = legacy_part.exists() or _is_link_like(legacy_part)
    part_size: int | None = None
    if part_exists and not _is_link_like(legacy_part):
        try:
            metadata = legacy_part.stat()
            if stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                part_size = metadata.st_size
        except OSError:
            part_size = None
    if not destination.exists() and not _is_link_like(destination):
        final_state = "absent"
    else:
        try:
            final_state = (
                "ready"
                if _verify_final(destination, spec, chunk_size=1024 * 1024)
                else "absent"
            )
        except _AttemptFailure as failure:
            final_state = (
                "conflict"
                if failure.code is AcquisitionFailureCode.TARGET_CONFLICT
                else "unsafe"
            )
    return LegacyAcquisitionState(
        source_id=spec.source_id,
        legacy_part_exists=part_exists,
        legacy_part_size=part_size,
        legacy_lock_exists=legacy_lock.exists() or _is_link_like(legacy_lock),
        legacy_lock_state=_inspect_existing_lock(legacy_lock),
        central_part_exists=central_part.exists() or _is_link_like(central_part),
        final_state=final_state,
    )


def migrate_legacy_acquisition_state(
    spec: AssetObjectSpec,
    target: str | Path,
    *,
    data_root: str | Path,
    policy: AcquisitionPolicy | None = None,
) -> LegacyMigrationResult:
    """Explicitly move one quiescent legacy prefix into central DATA_ROOT state.

    The operation is intentionally never called by either acquisition path.  It
    requires the former lock file to exist and be acquirable, then holds both
    the legacy and central locks while it verifies that exactly one checkpoint
    exists.  This prevents an active old relay and a new relay from consuming
    different resume offsets concurrently.
    """

    selected_policy = policy or AcquisitionPolicy()
    destination = Path(target)
    deadline = time.monotonic() + selected_policy.overall_timeout_seconds
    legacy_part = part_path_for(destination)
    legacy_lock = lock_path_for(destination)
    try:
        central_part, central_lock = _prepare_state_paths(
            destination,
            data_root=data_root,
            allow_legacy_state=True,
        )
        if not legacy_lock.exists() or _is_link_like(legacy_lock):
            raise _AttemptFailure(AcquisitionFailureCode.LOCAL_STATE_INVALID, False)
        with _existing_advisory_lock(
            legacy_lock,
            timeout_seconds=selected_policy.lock_timeout_seconds,
            overall_deadline=deadline,
            poll_interval_seconds=selected_policy.lock_poll_interval_seconds,
        ):
            with _advisory_object_lock(
                destination,
                lock_path=central_lock,
                timeout_seconds=selected_policy.lock_timeout_seconds,
                overall_deadline=deadline,
                poll_interval_seconds=selected_policy.lock_poll_interval_seconds,
            ):
                if _verify_final(
                    destination,
                    spec,
                    chunk_size=selected_policy.chunk_size,
                ):
                    return LegacyMigrationResult(
                        source_id=spec.source_id,
                        status="final_ready",
                        migrated_size=0,
                    )
                if central_part.exists() or _is_link_like(central_part):
                    raise _AttemptFailure(
                        AcquisitionFailureCode.LOCAL_STATE_INVALID,
                        False,
                    )
                metadata = _validate_regular_object(
                    legacy_part,
                    require_single_link=True,
                )
                if metadata.st_size > spec.expected_size:
                    raise _AttemptFailure(
                        AcquisitionFailureCode.SIZE_MISMATCH,
                        False,
                        observed_size=metadata.st_size,
                    )
                if metadata.st_size == spec.expected_size:
                    try:
                        digest = _sha256_file(
                            legacy_part,
                            chunk_size=selected_policy.chunk_size,
                        )
                    except OSError as error:
                        raise _AttemptFailure(
                            AcquisitionFailureCode.LOCAL_IO_ERROR,
                            False,
                            observed_size=metadata.st_size,
                        ) from error
                    if digest != spec.expected_sha256:
                        raise _AttemptFailure(
                            AcquisitionFailureCode.HASH_MISMATCH,
                            False,
                            observed_size=metadata.st_size,
                        )
                try:
                    os.replace(legacy_part, central_part)
                    _fsync_directory(legacy_part.parent)
                    _fsync_directory(central_part.parent)
                except OSError as error:
                    raise _AttemptFailure(
                        AcquisitionFailureCode.LOCAL_IO_ERROR,
                        False,
                        observed_size=metadata.st_size,
                    ) from error
                return LegacyMigrationResult(
                    source_id=spec.source_id,
                    status="migrated",
                    migrated_size=metadata.st_size,
                )
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=0,
                exhausted=False,
            )
        ) from None


def acquire_http_asset(
    spec: AssetObjectSpec,
    runtime_url: str,
    target: str | Path,
    *,
    policy: AcquisitionPolicy | None = None,
    data_root: str | Path | None = None,
) -> AcquisitionResult:
    """Acquire one immutable object without ever persisting its runtime URL.

    The caller owns URL refresh.  This primitive bounds one supplied endpoint,
    serializes writers per target, resumes only after validating HTTP range
    semantics, verifies the complete fixed identity, and publishes with an
    atomic no-clobber hard link.
    """

    selected_policy = policy or AcquisitionPolicy()
    destination = Path(target)
    start = time.monotonic()
    overall_deadline = start + selected_policy.overall_timeout_seconds
    if not _validate_runtime_url(runtime_url):
        report = AcquisitionFailureReport(
            code=AcquisitionFailureCode.INVALID_RUNTIME_URL,
            retryable=False,
            source_id=spec.source_id,
            revision=spec.revision,
            expected_size=spec.expected_size,
            expected_sha256=spec.expected_sha256,
            attempts=0,
        )
        raise AssetAcquisitionError(report) from None

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if _verify_final(
            destination,
            spec,
            chunk_size=selected_policy.chunk_size,
        ):
            return AcquisitionResult(
                status=AcquisitionStatus.ALREADY_READY,
                source_id=spec.source_id,
                revision=spec.revision,
                size_bytes=spec.expected_size,
                sha256=spec.expected_sha256,
                attempts=0,
                resumed=False,
                network_accessed=False,
            )
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=0,
                exhausted=False,
            )
        ) from None
    except OSError:
        failure = _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False)
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=0,
                exhausted=False,
            )
        ) from None

    attempts = 0
    resumed = False
    network_accessed = False
    try:
        part_path, object_lock_path = _prepare_state_paths(
            destination,
            data_root=data_root,
        )
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=0,
                exhausted=False,
            )
        ) from None
    try:
        with _advisory_object_lock(
            destination,
            lock_path=object_lock_path,
            timeout_seconds=selected_policy.lock_timeout_seconds,
            overall_deadline=overall_deadline,
            poll_interval_seconds=selected_policy.lock_poll_interval_seconds,
        ):
            if _verify_final(
                destination,
                spec,
                chunk_size=selected_policy.chunk_size,
            ):
                return AcquisitionResult(
                    status=AcquisitionStatus.ALREADY_READY,
                    source_id=spec.source_id,
                    revision=spec.revision,
                    size_bytes=spec.expected_size,
                    sha256=spec.expected_sha256,
                    attempts=0,
                    resumed=False,
                    network_accessed=False,
                )

            last_failure: _AttemptFailure | None = None
            while attempts < selected_policy.max_attempts:
                if time.monotonic() >= overall_deadline:
                    last_failure = _AttemptFailure(
                        AcquisitionFailureCode.OVERALL_TIMEOUT,
                        True,
                        observed_size=(
                            part_path.stat().st_size if part_path.exists() else None
                        ),
                    )
                    break
                attempts += 1
                try:
                    transfer = _download_once(
                        spec=spec,
                        runtime_url=runtime_url,
                        part_path=part_path,
                        policy=selected_policy,
                        overall_deadline=overall_deadline,
                    )
                    resumed = resumed or (
                        transfer.resumed_from > 0
                        and transfer.resumed_from < spec.expected_size
                    )
                    network_accessed = network_accessed or transfer.network_accessed
                    status = _publish_no_clobber(
                        part_path=part_path,
                        target=destination,
                        spec=spec,
                        chunk_size=selected_policy.chunk_size,
                    )
                    return AcquisitionResult(
                        status=status,
                        source_id=spec.source_id,
                        revision=spec.revision,
                        size_bytes=spec.expected_size,
                        sha256=spec.expected_sha256,
                        attempts=attempts,
                        resumed=resumed,
                        network_accessed=network_accessed,
                    )
                except _AttemptFailure as failure:
                    last_failure = failure
                    if not failure.retryable or attempts >= selected_policy.max_attempts:
                        break
                    remaining = overall_deadline - time.monotonic()
                    if remaining <= 0:
                        last_failure = _AttemptFailure(
                            AcquisitionFailureCode.OVERALL_TIMEOUT,
                            True,
                            observed_size=failure.observed_size,
                            http_status=failure.http_status,
                        )
                        break
                    backoff = min(
                        selected_policy.initial_backoff_seconds
                        * (2 ** (attempts - 1)),
                        selected_policy.max_backoff_seconds,
                        remaining,
                    )
                    if backoff > 0:
                        time.sleep(backoff)

            assert last_failure is not None
            raise AssetAcquisitionError(
                _failure_report(
                    spec=spec,
                    failure=last_failure,
                    attempts=attempts,
                    exhausted=(
                        last_failure.retryable
                        and attempts >= selected_policy.max_attempts
                    ),
                )
            ) from None
    except AssetAcquisitionError:
        raise
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=attempts,
                exhausted=False,
            )
        ) from None
    except OSError:
        failure = _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False)
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=attempts,
                exhausted=False,
            )
        ) from None


def receive_streamed_asset(
    spec: AssetObjectSpec,
    target: str | Path,
    stream: BinaryIO,
    *,
    on_ready: Callable[[StreamReceptionPlan], None] | None = None,
    policy: AcquisitionPolicy | None = None,
    data_root: str | Path | None = None,
) -> AcquisitionResult:
    """Receive one immutable object from a URL-free external byte relay.

    The receiver shares the same per-target advisory lock, ``.part`` path,
    complete SHA-256 verification, and no-clobber publication primitive as the
    direct HTTP acquisition path.  ``on_ready`` runs while the object lock is
    held and communicates the exact resume offset to the relay before any
    bytes are read.  A short stream preserves the verified prefix for a later
    retry; a full object with the wrong digest is truncated so it cannot be
    mistaken for a resumable prefix.
    """

    selected_policy = policy or AcquisitionPolicy()
    destination = Path(target)
    overall_deadline = time.monotonic() + selected_policy.overall_timeout_seconds
    try:
        part_path, object_lock_path = _prepare_state_paths(
            destination,
            data_root=data_root,
        )
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=0,
                exhausted=False,
            )
        ) from None
    callback = on_ready or (lambda _plan: None)
    attempts = 0
    resumed = False
    network_accessed = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _advisory_object_lock(
            destination,
            lock_path=object_lock_path,
            timeout_seconds=selected_policy.lock_timeout_seconds,
            overall_deadline=overall_deadline,
            poll_interval_seconds=selected_policy.lock_poll_interval_seconds,
        ):
            if _verify_final(
                destination,
                spec,
                chunk_size=selected_policy.chunk_size,
            ):
                ready_plan = StreamReceptionPlan(
                    source_id=spec.source_id,
                    revision=spec.revision,
                    expected_size=spec.expected_size,
                    expected_sha256=spec.expected_sha256,
                    offset=spec.expected_size,
                    already_ready=True,
                )
                _bounded_call(
                    lambda: callback(ready_plan),
                    overall_deadline=overall_deadline,
                    observed_size=spec.expected_size,
                )
                return AcquisitionResult(
                    status=AcquisitionStatus.ALREADY_READY,
                    source_id=spec.source_id,
                    revision=spec.revision,
                    size_bytes=spec.expected_size,
                    sha256=spec.expected_sha256,
                    attempts=0,
                    resumed=False,
                    network_accessed=False,
                )

            attempts = 1
            with _open_part(part_path) as part:
                part.seek(0, os.SEEK_END)
                offset = part.tell()
                if offset > spec.expected_size:
                    _truncate_part(part)
                    offset = 0
                elif offset == spec.expected_size:
                    try:
                        digest = _sha256_file(
                            part_path,
                            chunk_size=selected_policy.chunk_size,
                        )
                    except OSError as error:
                        raise _AttemptFailure(
                            AcquisitionFailureCode.LOCAL_IO_ERROR,
                            False,
                            observed_size=offset,
                        ) from error
                    if digest != spec.expected_sha256:
                        _truncate_part(part)
                        offset = 0

                resumed = 0 < offset < spec.expected_size
                ready_plan = StreamReceptionPlan(
                    source_id=spec.source_id,
                    revision=spec.revision,
                    expected_size=spec.expected_size,
                    expected_sha256=spec.expected_sha256,
                    offset=offset,
                    already_ready=False,
                )
                _bounded_call(
                    lambda: callback(ready_plan),
                    overall_deadline=overall_deadline,
                    observed_size=offset,
                )

                received = 0
                remaining = spec.expected_size - offset
                part.seek(offset)
                while received < remaining:
                    if time.monotonic() >= overall_deadline:
                        part.flush()
                        os.fsync(part.fileno())
                        raise _AttemptFailure(
                            AcquisitionFailureCode.OVERALL_TIMEOUT,
                            True,
                            observed_size=offset + received,
                        )
                    read_size = min(selected_policy.chunk_size, remaining - received)
                    try:
                        read_deadline = min(
                            overall_deadline,
                            time.monotonic()
                            + selected_policy.request_timeout_seconds,
                        )
                        chunk = _bounded_call(
                            lambda: stream.read(read_size),
                            overall_deadline=read_deadline,
                            observed_size=offset + received,
                        )
                    except _AttemptFailure:
                        part.flush()
                        os.fsync(part.fileno())
                        raise
                    except (OSError, TimeoutError) as error:
                        part.flush()
                        os.fsync(part.fileno())
                        raise _AttemptFailure(
                            AcquisitionFailureCode.NETWORK_ERROR,
                            True,
                            observed_size=offset + received,
                        ) from error
                    if not chunk:
                        part.flush()
                        os.fsync(part.fileno())
                        raise _AttemptFailure(
                            AcquisitionFailureCode.TRANSFER_INCOMPLETE,
                            True,
                            observed_size=offset + received,
                        )
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise _AttemptFailure(
                            AcquisitionFailureCode.LOCAL_STATE_INVALID,
                            False,
                            observed_size=offset + received,
                        )
                    try:
                        written = _bounded_call(
                            lambda: part.write(chunk),
                            overall_deadline=overall_deadline,
                            observed_size=offset + received,
                        )
                    except _AttemptFailure:
                        raise
                    except OSError as error:
                        raise _AttemptFailure(
                            AcquisitionFailureCode.LOCAL_IO_ERROR,
                            False,
                            observed_size=offset + received,
                        ) from error
                    if written != len(chunk):
                        raise _AttemptFailure(
                            AcquisitionFailureCode.LOCAL_IO_ERROR,
                            False,
                            observed_size=offset + received,
                        )
                    received += written
                    network_accessed = True
                part.flush()
                os.fsync(part.fileno())
                try:
                    digest = _sha256_file(
                        part_path,
                        chunk_size=selected_policy.chunk_size,
                    )
                except OSError as error:
                    raise _AttemptFailure(
                        AcquisitionFailureCode.LOCAL_IO_ERROR,
                        False,
                        observed_size=spec.expected_size,
                    ) from error
                if digest != spec.expected_sha256:
                    _truncate_part(part)
                    raise _AttemptFailure(
                        AcquisitionFailureCode.HASH_MISMATCH,
                        True,
                        observed_size=spec.expected_size,
                    )

            status = _publish_no_clobber(
                part_path=part_path,
                target=destination,
                spec=spec,
                chunk_size=selected_policy.chunk_size,
            )
            return AcquisitionResult(
                status=status,
                source_id=spec.source_id,
                revision=spec.revision,
                size_bytes=spec.expected_size,
                sha256=spec.expected_sha256,
                attempts=attempts,
                resumed=resumed,
                network_accessed=network_accessed,
            )
    except AssetAcquisitionError:
        raise
    except _AttemptFailure as failure:
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=attempts,
                exhausted=False,
            )
        ) from None
    except OSError:
        failure = _AttemptFailure(AcquisitionFailureCode.LOCAL_IO_ERROR, False)
        raise AssetAcquisitionError(
            _failure_report(
                spec=spec,
                failure=failure,
                attempts=attempts,
                exhausted=False,
            )
        ) from None


__all__ = [
    "AcquisitionFailureCode",
    "AcquisitionFailureReport",
    "AcquisitionPolicy",
    "AcquisitionResult",
    "AcquisitionStatus",
    "AssetAcquisitionError",
    "AssetObjectSpec",
    "LegacyAcquisitionState",
    "LegacyMigrationResult",
    "StreamReceptionPlan",
    "acquire_http_asset",
    "inspect_legacy_acquisition_state",
    "lock_path_for",
    "migrate_legacy_acquisition_state",
    "part_path_for",
    "receive_streamed_asset",
    "resolve_approved_asset_target",
]
