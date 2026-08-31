"""Strict read-only access to Pythia's sharded Megatron mmap dataset.

The index contains byte offsets into one logical byte stream.  The stream may
be physically split across multiple ``document-*.bin`` files, including in
the middle of a record.  Callers must supply the ordered shard inventory
explicitly; this module never scans or globs a dataset directory.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
import os
from pathlib import Path
import re
import stat
import struct
import tempfile
from typing import Any, Sequence

import numpy as np
import torch

from ..atomic import atomic_write_bytes
from ..contracts.jsonio import canonical_json_bytes, canonical_json_hash, loads_strict_json


MMAP_INDEX_MAGIC = b"MMIDIDX\x00\x00"
MMAP_INDEX_VERSION = 1
MMAP_INDEX_HEADER_BYTES = 34
PYTHIA_TOKENS_PER_RECORD = 2049
FILE_VERIFICATION_CACHE_ENV = "PARAM_IMPORTANCE_FILE_VERIFICATION_CACHE"
FILE_VERIFICATION_SCHEMA_VERSION = "param-importance-file-verification-v2"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHARD_NAME_RE = re.compile(
    r"^document-(?P<ordinal>[0-9]{5})-of-(?P<last>[0-9]{5})\.bin$"
)
_DTYPE_BY_CODE: dict[int, np.dtype[Any]] = {
    1: np.dtype(np.uint8),
    2: np.dtype(np.int8),
    3: np.dtype(np.int16),
    4: np.dtype(np.int32),
    5: np.dtype(np.int64),
    6: np.dtype(np.float32),
    7: np.dtype(np.float64),
    8: np.dtype(np.uint16),
}
_FILE_VERIFICATION_CERT_FIELDS = frozenset(
    {
        "schema_version",
        "logical_path",
        "logical_is_symlink",
        "logical_st_dev",
        "logical_st_ino",
        "logical_st_size",
        "logical_st_mtime_ns",
        "logical_st_ctime_ns",
        "resolved_path",
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
        "expected_sha256",
        "actual_sha256",
        "verified_at",
        "artifact_hash",
    }
)
_LOGGER = logging.getLogger(__name__)


class PythiaDataError(ValueError):
    """Raised when a Pythia index, shard inventory, or record is invalid."""


def _require_nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PythiaDataError(f"{field} must be a non-negative integer")
    return value


def sha256_file(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    """Return the streaming SHA-256 of one regular file."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verification_cache_root(cache_root: str | Path | None) -> Path | None:
    """Resolve the explicitly opted-in certificate cache root."""

    from_environment = cache_root is None
    if cache_root is None:
        configured = os.environ.get(FILE_VERIFICATION_CACHE_ENV)
        if not configured:
            return None
        cache_root = configured
    try:
        raw = os.fspath(cache_root)
    except TypeError as error:
        raise ValueError("file verification cache_root must be a path") from error
    if not isinstance(raw, str) or not raw or raw.strip() != raw or not raw.strip():
        raise ValueError("file verification cache_root must be a non-blank path")
    try:
        candidate_input = Path(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("file verification cache_root must be a valid path") from error
    if not candidate_input.is_absolute():
        raise ValueError("file verification cache_root must be absolute")
    try:
        lexical = candidate_input.expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification cache_root cannot be resolved") from error
    if lexical.is_symlink():
        raise ValueError("file verification cache_root must not be a symlink")
    try:
        candidate = lexical.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification cache_root cannot be resolved") from error

    if from_environment and candidate.name != ".file-verification":
        raise ValueError(
            "environment file verification cache_root must be a dedicated "
            ".file-verification leaf"
        )
    if candidate == Path(candidate.anchor):
        raise ValueError("file verification cache_root cannot be a filesystem root")

    try:
        cwd = Path.cwd().resolve()
        home = Path.home().resolve()
    except (OSError, RuntimeError) as error:
        raise ValueError("file verification cache_root anchors cannot be resolved") from error

    def _contains(parent: Path, child: Path) -> bool:
        return child == parent or child.is_relative_to(parent)

    # A verification cache must never be placed in the current checkout (or in
    # one of its parents/children).  This also catches ``.`` and the worktree
    # root without relying on a particular repository layout.
    repository_roots = [
        parent
        for parent in (cwd, *cwd.parents)
        if (parent / ".git").exists()
    ]
    for anchor in (cwd, *repository_roots):
        if _contains(anchor, candidate) or _contains(candidate, anchor):
            raise ValueError(
                "file verification cache_root cannot overlap cwd or worktree"
            )

    # Keep broad user/system locations out of the dedicated cache namespace.
    # Temporary directories are allowed only below a per-run/user-created leaf,
    # never directly as the cache root itself.
    protected: list[Path] = []
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        protected.extend(
            Path(value).resolve()
            for value in (
                system_root,
                os.environ.get("ProgramFiles", r"C:\Program Files"),
                os.environ.get("ProgramData", r"C:\ProgramData"),
            )
        )
    else:
        protected.extend(
            Path(value)
            for value in (
                "/bin",
                "/boot",
                "/dev",
                "/etc",
                "/lib",
                "/lib64",
                "/home",
                "/media",
                "/mnt",
                "/opt",
                "/proc",
                "/root",
                "/run",
                "/sbin",
                "/srv",
                "/sys",
                "/usr",
                "/var",
            )
        )
    temporary_root = Path(tempfile.gettempdir()).resolve()
    for protected_root in protected:
        if _contains(protected_root, candidate):
            raise ValueError(
                f"file verification cache_root cannot be under system directory {protected_root}"
            )
    if candidate == home or candidate.parent == home:
        raise ValueError("file verification cache_root cannot be home or its direct child")
    if os.name == "nt" and (
        candidate == home.parent or candidate.parent == home.parent
    ):
        raise ValueError("file verification cache_root cannot be the broad Users directory")
    if candidate == temporary_root or candidate.parent == temporary_root:
        raise ValueError("file verification cache_root cannot be a broad temp directory")

    # Validate the existing/creatable parent without creating anything.  The
    # actual leaf is created only after a successful file hash.
    if candidate.exists():
        if not candidate.is_dir():
            raise ValueError("file verification cache_root must be a directory")
    else:
        existing_parent = candidate.parent
        while not existing_parent.exists() and existing_parent != existing_parent.parent:
            existing_parent = existing_parent.parent
        if not existing_parent.is_dir():
            raise ValueError("file verification cache_root parent is not a directory")
    return candidate


def _regular_file_stat(path: Path) -> dict[str, int] | None:
    """Return the stat identity used by a certificate, or ``None`` if unsafe."""

    try:
        observed = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(observed.st_mode):
        return None
    return {
        "st_dev": int(observed.st_dev),
        "st_ino": int(observed.st_ino),
        "st_size": int(observed.st_size),
        "st_mtime_ns": int(observed.st_mtime_ns),
        "st_ctime_ns": int(observed.st_ctime_ns),
    }


def _logical_file_stat(path: Path) -> dict[str, int | bool] | None:
    """Return the original logical path's lstat identity."""

    try:
        observed = path.lstat()
    except OSError:
        return None
    if not (stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode)):
        return None
    return {
        "is_symlink": stat.S_ISLNK(observed.st_mode),
        "st_dev": int(observed.st_dev),
        "st_ino": int(observed.st_ino),
        "st_size": int(observed.st_size),
        "st_mtime_ns": int(observed.st_mtime_ns),
        "st_ctime_ns": int(observed.st_ctime_ns),
    }


def _verification_snapshot(
    logical_path: Path,
) -> tuple[Path, dict[str, int | bool], dict[str, int]] | None:
    """Capture logical-link and resolved-target identity for one verification."""

    logical_stat = _logical_file_stat(logical_path)
    if logical_stat is None:
        return None
    try:
        resolved_path = logical_path.resolve()
    except (OSError, RuntimeError):
        return None
    target_stat = _regular_file_stat(resolved_path)
    if target_stat is None:
        return None
    return resolved_path, logical_stat, target_stat


def _verification_certificate_path(
    cache_root: Path,
    logical_path: Path,
    resolved_path: Path,
    expected_sha256: str,
) -> Path:
    """Return a path keyed by logical/target paths and expected digest."""

    key = hashlib.sha256(
        f"{logical_path}\x00{resolved_path}\x00{expected_sha256}".encode("utf-8")
    ).hexdigest()
    return cache_root / key[:2] / f"{key}.json"


def _valid_verified_at(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _certificate_hit(
    certificate_path: Path,
    *,
    logical_path: Path,
    logical_stat: Mapping[str, int | bool],
    resolved_path: Path,
    expected_sha256: str,
    target_stat: Mapping[str, int],
) -> bool:
    """Validate a certificate completely; malformed data is always a miss."""

    try:
        encoded = certificate_path.read_bytes()
        decoded = loads_strict_json(encoded)
        if not isinstance(decoded, Mapping) or set(decoded) != _FILE_VERIFICATION_CERT_FIELDS:
            return False
        certificate = dict(decoded)
        if encoded != canonical_json_bytes(certificate):
            return False
        if certificate.get("schema_version") != FILE_VERIFICATION_SCHEMA_VERSION:
            return False
        if certificate.get("logical_path") != str(logical_path):
            return False
        if type(certificate.get("logical_is_symlink")) is not bool:
            return False
        if certificate["logical_is_symlink"] != logical_stat["is_symlink"]:
            return False
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            logical_field = f"logical_{field}"
            if (
                type(certificate.get(logical_field)) is not int
                or certificate[logical_field] != logical_stat[field]
            ):
                return False
        if certificate.get("resolved_path") != str(resolved_path):
            return False
        if certificate.get("expected_sha256") != expected_sha256:
            return False
        for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if type(certificate.get(field)) is not int or certificate[field] != target_stat[field]:
                return False
        actual_sha256 = certificate.get("actual_sha256")
        if (
            not isinstance(actual_sha256, str)
            or _SHA256_RE.fullmatch(actual_sha256) is None
            or actual_sha256 != expected_sha256
        ):
            return False
        if not _valid_verified_at(certificate.get("verified_at")):
            return False
        artifact_hash = certificate.get("artifact_hash")
        if (
            not isinstance(artifact_hash, str)
            or _SHA256_RE.fullmatch(artifact_hash) is None
            or artifact_hash
            != canonical_json_hash(
                {
                    field: value
                    for field, value in certificate.items()
                    if field != "artifact_hash"
                }
            )
        ):
            return False
    except Exception:
        return False
    return True


def _write_verification_certificate(
    certificate_path: Path,
    *,
    logical_path: Path,
    logical_stat: Mapping[str, int | bool],
    resolved_path: Path,
    expected_sha256: str,
    actual_sha256: str,
    target_stat: Mapping[str, int],
) -> None:
    certificate: dict[str, object] = {
        "schema_version": FILE_VERIFICATION_SCHEMA_VERSION,
        "logical_path": str(logical_path),
        "logical_is_symlink": logical_stat["is_symlink"],
        "logical_st_dev": logical_stat["st_dev"],
        "logical_st_ino": logical_stat["st_ino"],
        "logical_st_size": logical_stat["st_size"],
        "logical_st_mtime_ns": logical_stat["st_mtime_ns"],
        "logical_st_ctime_ns": logical_stat["st_ctime_ns"],
        "resolved_path": str(resolved_path),
        **dict(target_stat),
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    certificate["artifact_hash"] = canonical_json_hash(certificate)
    try:
        atomic_write_bytes(certificate_path, canonical_json_bytes(certificate), mode=0o644)
    except OSError as error:
        # Verification remains authoritative if an optional cache is unwritable.
        _LOGGER.debug("file verification certificate write skipped: %s", error)


def verified_sha256(
    path: str | Path,
    expected_sha256: str,
    cache_root: str | Path | None = None,
) -> str:
    """Return a file SHA-256, using an opt-in fail-closed stat certificate cache.

    A cache hit is accepted only when the certificate is canonical, complete, bound
    to the resolved path and expected digest, and all recorded filesystem identity
    fields still match.  A miss always falls back to hashing the file.  The actual
    digest is returned even when it differs from ``expected_sha256`` so existing
    callers retain their established, diagnostic mismatch errors.
    """

    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    root = _verification_cache_root(cache_root)
    if root is None:
        _LOGGER.debug("file verification cache disabled for %s", path)
        return sha256_file(path)

    try:
        logical_path = Path(path).expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification logical path cannot be resolved") from error
    initial = _verification_snapshot(logical_path)
    if initial is None:
        raise ValueError("file verification target must be a regular file")
    resolved_path, logical_stat, target_stat = initial
    certificate_path = _verification_certificate_path(
        root,
        logical_path,
        resolved_path,
        expected_sha256,
    )
    if _certificate_hit(
        certificate_path,
        logical_path=logical_path,
        logical_stat=logical_stat,
        resolved_path=resolved_path,
        expected_sha256=expected_sha256,
        target_stat=target_stat,
    ):
        # A second stat closes the read/validation window before accepting a hit.
        if _verification_snapshot(logical_path) == initial:
            _LOGGER.debug("file verification cache hit: %s", resolved_path)
            return expected_sha256

    _LOGGER.debug("file verification cache miss: %s", resolved_path)
    actual_sha256 = sha256_file(resolved_path)
    final = _verification_snapshot(logical_path)
    if final != initial and final is not None:
        # Do not certify a file that changed while it was being hashed.  One retry
        # gives a concurrently published file a chance to settle without changing
        # the existing mismatch behavior for callers.
        actual_sha256 = sha256_file(final[0])
        final = _verification_snapshot(logical_path)
    if (
        actual_sha256 == expected_sha256
        and final is not None
        and final == initial
    ):
        _write_verification_certificate(
            certificate_path,
            logical_path=logical_path,
            logical_stat=logical_stat,
            resolved_path=final[0],
            expected_sha256=expected_sha256,
            actual_sha256=actual_sha256,
            target_stat=final[2],
        )
    return actual_sha256


@dataclass(frozen=True, slots=True)
class PythiaShardDescriptor:
    """Expected immutable identity of one physical Pythia byte-stream shard."""

    ordinal: int
    path: Path
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_nonnegative_integer(self.ordinal, field="shard ordinal")
        _require_nonnegative_integer(self.size_bytes, field="shard size_bytes")
        if self.size_bytes == 0:
            raise PythiaDataError("shard size_bytes must be positive")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            raise PythiaDataError("shard sha256 must be 64 lowercase hexadecimal characters")
        try:
            path = Path(self.path).expanduser()
        except TypeError as exc:
            raise PythiaDataError("shard path must be path-like") from exc
        if path.name.casefold().endswith(".part"):
            raise PythiaDataError(f"temporary Pile shard is forbidden: {path}")
        object.__setattr__(self, "path", path)


class OrderedShardReader:
    """Read a verified ordered shard inventory as one continuous byte stream."""

    def __init__(
        self,
        shards: Sequence[PythiaShardDescriptor],
        *,
        cache_root: str | Path | None = None,
    ) -> None:
        descriptors = tuple(shards)
        if not descriptors:
            raise PythiaDataError("at least one Pythia shard descriptor is required")
        if not all(isinstance(item, PythiaShardDescriptor) for item in descriptors):
            raise TypeError("shards must contain only PythiaShardDescriptor values")

        ordinals = tuple(item.ordinal for item in descriptors)
        if len(set(ordinals)) != len(ordinals):
            raise PythiaDataError("duplicate Pythia shard ordinal")
        if ordinals != tuple(range(len(descriptors))):
            raise PythiaDataError(
                "Pythia shard descriptors must be ordered and contiguous from ordinal zero"
            )

        logical_paths = tuple(item.path.expanduser().absolute() for item in descriptors)
        resolved_paths = tuple(path.resolve() for path in logical_paths)
        if len(set(resolved_paths)) != len(resolved_paths):
            raise PythiaDataError("duplicate Pythia shard path")

        declared_last: int | None = None
        starts: list[int] = []
        ends: list[int] = []
        cursor = 0
        for item, resolved in zip(descriptors, resolved_paths, strict=True):
            match = _SHARD_NAME_RE.fullmatch(item.path.name)
            if match is None:
                raise PythiaDataError(f"invalid Pythia shard filename: {item.path.name}")
            name_ordinal = int(match.group("ordinal"))
            name_last = int(match.group("last"))
            if name_ordinal != item.ordinal or item.ordinal > name_last:
                raise PythiaDataError(
                    f"Pythia shard filename/ordinal mismatch: {item.path.name}"
                )
            if declared_last is None:
                declared_last = name_last
            elif name_last != declared_last:
                raise PythiaDataError("Pythia shard filenames disagree on the final ordinal")
            if not resolved.is_file():
                raise FileNotFoundError(f"Pythia shard does not exist: {resolved}")
            actual_size = resolved.stat().st_size
            if actual_size != item.size_bytes:
                raise PythiaDataError(
                    f"Pythia shard size mismatch for {resolved}: "
                    f"{actual_size} != {item.size_bytes}"
                )
            actual_sha256 = verified_sha256(
                resolved,
                item.sha256,
                cache_root=cache_root,
            )
            if actual_sha256 != item.sha256:
                raise PythiaDataError(
                    f"Pythia shard SHA-256 mismatch for {resolved}: "
                    f"{actual_sha256} != {item.sha256}"
                )
            starts.append(cursor)
            cursor += item.size_bytes
            ends.append(cursor)

        self._shards = descriptors
        self._logical_paths = logical_paths
        self._paths = resolved_paths
        self._starts = tuple(starts)
        self._ends = tuple(ends)
        self._total_size = cursor

    @property
    def shards(self) -> tuple[PythiaShardDescriptor, ...]:
        return self._shards

    @property
    def logical_paths(self) -> tuple[Path, ...]:
        """Return the original absolute shard paths, without resolving links."""

        return self._logical_paths

    @property
    def resolved_paths(self) -> tuple[Path, ...]:
        """Return the resolved target paths captured during reader construction."""

        return self._paths

    @property
    def total_size(self) -> int:
        return self._total_size

    def read_exact(self, offset: int, length: int) -> bytes:
        """Read an exact global byte interval, crossing shard boundaries as needed."""

        offset = _require_nonnegative_integer(offset, field="byte offset")
        length = _require_nonnegative_integer(length, field="byte length")
        end = offset + length
        if end > self.total_size:
            raise PythiaDataError(
                f"byte interval [{offset}, {end}) exceeds shard stream size {self.total_size}"
            )
        if length == 0:
            return b""

        shard_index = bisect_right(self._ends, offset)
        chunks: list[bytes] = []
        position = offset
        remaining = length
        while remaining:
            if shard_index >= len(self._shards):
                raise PythiaDataError("Pythia shard stream ended before the requested interval")
            local_offset = position - self._starts[shard_index]
            available = self._ends[shard_index] - position
            count = min(remaining, available)
            with self._paths[shard_index].open("rb") as handle:
                handle.seek(local_offset)
                chunk = handle.read(count)
            if len(chunk) != count:
                raise PythiaDataError(
                    f"short read from Pythia shard {self._paths[shard_index]}: "
                    f"{len(chunk)} != {count}"
                )
            chunks.append(chunk)
            position += count
            remaining -= count
            shard_index += 1
        return b"".join(chunks)


class MMapIndex:
    """Strict read-only view of a Megatron ``MMIDIDX`` version-1 index."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_sha256: str | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        self.logical_path = Path(path).expanduser().absolute()
        self.path = self.logical_path.resolve()
        if self.path.name.casefold().endswith(".part"):
            raise PythiaDataError(f"temporary Pythia index is forbidden: {self.path}")
        if not self.path.is_file():
            raise FileNotFoundError(f"Pythia index does not exist: {self.path}")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(
                expected_sha256
            ):
                raise PythiaDataError(
                    "expected index sha256 must be 64 lowercase hexadecimal characters"
                )
            actual_sha256 = verified_sha256(
                self.path,
                expected_sha256,
                cache_root=cache_root,
            )
            if actual_sha256 != expected_sha256:
                raise PythiaDataError(
                    f"Pythia index SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
                )

        with self.path.open("rb") as handle:
            header = handle.read(MMAP_INDEX_HEADER_BYTES)
        if len(header) != MMAP_INDEX_HEADER_BYTES:
            raise PythiaDataError(f"truncated Pythia index header: {self.path}")
        if header[:9] != MMAP_INDEX_MAGIC:
            raise PythiaDataError(f"invalid Pythia index magic: {header[:9]!r}")
        self.version = struct.unpack_from("<Q", header, 9)[0]
        if self.version != MMAP_INDEX_VERSION:
            raise PythiaDataError(f"unsupported Pythia index version: {self.version}")
        self.dtype_code = header[17]
        try:
            self.dtype = _DTYPE_BY_CODE[self.dtype_code]
        except KeyError as exc:
            raise PythiaDataError(
                f"unsupported Pythia index dtype code: {self.dtype_code}"
            ) from exc
        self.sequence_count, self.document_count = struct.unpack_from("<QQ", header, 18)
        if self.sequence_count <= 0:
            raise PythiaDataError("Pythia index sequence_count must be positive")
        if self.document_count <= 0:
            raise PythiaDataError("Pythia index document_count must be positive")

        expected_size = (
            MMAP_INDEX_HEADER_BYTES
            + 4 * self.sequence_count
            + 8 * self.sequence_count
            + 8 * self.document_count
        )
        actual_size = self.path.stat().st_size
        if actual_size != expected_size:
            raise PythiaDataError(
                f"Pythia index size mismatch: {actual_size} != {expected_size}"
            )

        self._mmap: np.memmap[Any] | None = np.memmap(
            self.path, mode="r", dtype=np.uint8
        )
        sizes_offset = MMAP_INDEX_HEADER_BYTES
        pointers_offset = sizes_offset + 4 * self.sequence_count
        documents_offset = pointers_offset + 8 * self.sequence_count
        self.sizes = np.ndarray(
            (self.sequence_count,),
            dtype="<i4",
            buffer=self._mmap,
            offset=sizes_offset,
        )
        self.pointers = np.ndarray(
            (self.sequence_count,),
            dtype="<i8",
            buffer=self._mmap,
            offset=pointers_offset,
        )
        self.document_index = np.ndarray(
            (self.document_count,),
            dtype="<i8",
            buffer=self._mmap,
            offset=documents_offset,
        )

    def close(self) -> None:
        for name in ("sizes", "pointers", "document_index"):
            if hasattr(self, name):
                delattr(self, name)
        mapping = getattr(self, "_mmap", None)
        if mapping is not None:
            mapping._mmap.close()
            self._mmap = None

    def __enter__(self) -> "MMapIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class PythiaNextTokenBatch:
    """A 2049-token record batch mapped to 2048 explicit next-token targets."""

    tokens: torch.Tensor
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor
    source_record_start: int
    source_record_stop: int

    def __post_init__(self) -> None:
        if self.tokens.ndim != 2 or self.tokens.shape[1] < 2:
            raise PythiaDataError("Pythia batch tokens must have shape [B, L>=2]")
        expected = (self.tokens.shape[0], self.tokens.shape[1] - 1)
        for name, value in (
            ("input_ids", self.input_ids),
            ("target_ids", self.target_ids),
            ("attention_mask", self.attention_mask),
        ):
            if tuple(value.shape) != expected:
                raise PythiaDataError(f"Pythia batch {name} has the wrong shape")
        if self.source_record_stop - self.source_record_start != self.tokens.shape[0]:
            raise PythiaDataError("Pythia batch source record interval is inconsistent")

    def payload(self) -> dict[str, torch.Tensor]:
        """Return the exact payload expected by the pre-shifted training route."""

        return {
            "input_ids": self.input_ids,
            "target_ids": self.target_ids,
            "attention_mask": self.attention_mask,
        }


class PythiaIndexedDataset:
    """Read a validated record interval through global index byte pointers."""

    _VALIDATION_CHUNK_RECORDS = 1_000_000

    def __init__(
        self,
        idx_path: str | Path,
        shards: Sequence[PythiaShardDescriptor],
        *,
        record_start: int = 0,
        record_stop: int | None = None,
        tokens_per_record: int = PYTHIA_TOKENS_PER_RECORD,
        expected_idx_sha256: str | None = None,
        cache_root: str | Path | None = None,
    ) -> None:
        record_start = _require_nonnegative_integer(record_start, field="record_start")
        if record_stop is not None:
            record_stop = _require_nonnegative_integer(record_stop, field="record_stop")
        tokens_per_record = _require_nonnegative_integer(
            tokens_per_record, field="tokens_per_record"
        )
        if tokens_per_record < 2:
            raise PythiaDataError("tokens_per_record must be at least two")

        self.index = MMapIndex(
            idx_path,
            expected_sha256=expected_idx_sha256,
            cache_root=cache_root,
        )
        try:
            self.reader = OrderedShardReader(shards, cache_root=cache_root)
            if self.index.dtype_code != 8 or self.index.dtype != np.dtype(np.uint16):
                raise PythiaDataError(
                    "Pythia token records require MMIDIDX dtype code 8 (uint16)"
                )
            stop = self.index.sequence_count if record_stop is None else record_stop
            if record_start >= stop or stop > self.index.sequence_count:
                raise PythiaDataError(
                    f"invalid Pythia record interval [{record_start}, {stop}) for "
                    f"{self.index.sequence_count} indexed records"
                )
            self.record_start = record_start
            self.record_stop = stop
            self.tokens_per_record = tokens_per_record
            self._validate_interval()
        except BaseException:
            self.index.close()
            raise

    def _validate_interval(self) -> None:
        itemsize = self.index.dtype.itemsize
        previous_end: int | None = None
        if self.record_start > 0:
            previous_pointer = int(self.index.pointers[self.record_start - 1])
            previous_size = int(self.index.sizes[self.record_start - 1])
            previous_end = previous_pointer + previous_size * itemsize

        for start in range(
            self.record_start,
            self.record_stop,
            self._VALIDATION_CHUNK_RECORDS,
        ):
            stop = min(start + self._VALIDATION_CHUNK_RECORDS, self.record_stop)
            sizes = self.index.sizes[start:stop]
            pointers = self.index.pointers[start:stop]
            bad_size = np.flatnonzero(sizes != self.tokens_per_record)
            if bad_size.size:
                position = start + int(bad_size[0])
                raise PythiaDataError(
                    f"Pythia record {position} has {int(sizes[bad_size[0]])} tokens; "
                    f"expected {self.tokens_per_record}"
                )
            negative = np.flatnonzero(pointers < 0)
            if negative.size:
                raise PythiaDataError(
                    f"Pythia record {start + int(negative[0])} has a negative byte pointer"
                )
            misaligned = np.flatnonzero(pointers % itemsize)
            if misaligned.size:
                raise PythiaDataError(
                    f"Pythia record {start + int(misaligned[0])} byte pointer is unaligned"
                )
            if previous_end is not None and int(pointers[0]) != previous_end:
                raise PythiaDataError(
                    f"Pythia record {start} is not contiguous with its predecessor"
                )
            if len(pointers) > 1:
                expected_next = (
                    pointers[:-1]
                    + sizes[:-1].astype(np.int64, copy=False) * itemsize
                )
                mismatch = np.flatnonzero(pointers[1:] != expected_next)
                if mismatch.size:
                    position = start + int(mismatch[0]) + 1
                    raise PythiaDataError(
                        f"Pythia record {position} is not contiguous with its predecessor"
                    )
            previous_end = int(pointers[-1]) + int(sizes[-1]) * itemsize

        assert previous_end is not None
        if self.record_start == 0 and int(self.index.pointers[0]) != 0:
            raise PythiaDataError("the first Pythia record must start at byte zero")
        if self.record_stop < self.index.sequence_count:
            next_pointer = int(self.index.pointers[self.record_stop])
            if next_pointer != previous_end:
                raise PythiaDataError(
                    f"Pythia record {self.record_stop} is not contiguous with its predecessor"
                )
        if previous_end > self.reader.total_size:
            raise PythiaDataError(
                f"Pythia record interval ends at byte {previous_end}, beyond verified "
                f"shard coverage {self.reader.total_size}"
            )
        self.max_end_offset = previous_end

    def __len__(self) -> int:
        return self.record_stop - self.record_start

    def _global_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Pythia record index must be an integer")
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.record_start + index

    def raw_record(self, index: int) -> np.ndarray[Any, np.dtype[Any]]:
        global_index = self._global_index(index)
        pointer = int(self.index.pointers[global_index])
        size = int(self.index.sizes[global_index])
        raw = self.reader.read_exact(pointer, size * self.index.dtype.itemsize)
        return np.frombuffer(raw, dtype=self.index.dtype, count=size).copy()

    def raw_records(
        self, start: int, count: int
    ) -> np.ndarray[Any, np.dtype[Any]]:
        start = _require_nonnegative_integer(start, field="record batch start")
        count = _require_nonnegative_integer(count, field="record batch count")
        if start + count > len(self):
            raise IndexError(
                f"record range [{start}, {start + count}) exceeds dataset length {len(self)}"
            )
        if count == 0:
            return np.empty((0, self.tokens_per_record), dtype=self.index.dtype)
        global_start = self.record_start + start
        pointer = int(self.index.pointers[global_start])
        value_count = count * self.tokens_per_record
        raw = self.reader.read_exact(pointer, value_count * self.index.dtype.itemsize)
        values = np.frombuffer(raw, dtype=self.index.dtype, count=value_count).copy()
        return values.reshape(count, self.tokens_per_record)

    def batch(self, start: int, count: int) -> PythiaNextTokenBatch:
        count = _require_nonnegative_integer(count, field="Pythia batch count")
        if count == 0:
            raise PythiaDataError("Pythia batch count must be positive")
        records = self.raw_records(start, count)
        tokens = torch.from_numpy(records.astype(np.int64, copy=False))
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        source_start = self.record_start + start
        return PythiaNextTokenBatch(
            tokens=tokens,
            input_ids=input_ids,
            target_ids=target_ids,
            attention_mask=attention_mask,
            source_record_start=source_start,
            source_record_stop=source_start + count,
        )

    def close(self) -> None:
        self.index.close()

    def __enter__(self) -> "PythiaIndexedDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
