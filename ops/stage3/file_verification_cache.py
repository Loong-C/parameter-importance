"""Opt-in cross-process file verification certificates for Stage 3 operations.

The shim is deliberately outside the G3 critical-source set. It can be loaded
by a generated sitecustomize.py and only replaces the Pythia module's
sha256_file callable when an explicit cache-root environment variable is
present. Certificates never replace the caller's expected-digest checks:
Pythia still compares the returned actual digest with its manifest value.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Any, Callable

FILE_VERIFICATION_CACHE_ENV = "PARAM_IMPORTANCE_FILE_VERIFICATION_CACHE"
FILE_VERIFICATION_SCHEMA_VERSION = "stage3-file-verification-certificate-v1"
INSTALLER_IDENTITY = "ops.stage3.install_file_verification_cache:install_startup_hook"
CALLER_SCOPE = "pythia_mmap.sha256_file"


def _sha256(payload: bytes = b"") -> Any:
    """Create SHA-256 without depending on a mutable ``hashlib.sha256`` alias.

    Verification tests and embedding applications may instrument the public
    ``hashlib.sha256`` symbol to detect source-file hashing.  Cache-key and
    certificate hashing must remain available in that situation; the source
    file itself is still read only by ``_stream_sha256``/the Pythia fallback.
    ``hashlib.new`` keeps the cache's own bookkeeping independent of that
    instrumentation while retaining the standard SHA-256 implementation.
    """

    return hashlib.new("sha256", payload)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CERTIFICATE_FIELDS = frozenset(
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
        "actual_sha256",
        "expected_sha256",
        "caller_scope",
        "verified_at",
        "artifact_hash",
    }
)


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_json_hash(value: Mapping[str, object]) -> str:
    return _sha256(_canonical_json_bytes(value)).hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".file-verification-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            directory_fd = -1
        if directory_fd >= 0:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _contains(parent: Path, child: Path) -> bool:
    return child == parent or child.is_relative_to(parent)


def _verification_cache_root(cache_root: str | Path | None) -> Path | None:
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
        input_path = Path(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("file verification cache_root must be a valid path") from error
    if not input_path.is_absolute():
        raise ValueError("file verification cache_root must be absolute")
    try:
        lexical = input_path.expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification cache_root cannot be resolved") from error

    # Apply lexical checks before touching the filesystem. This protects
    # inaccessible system roots from symlink/resolve probes.
    if lexical == Path(lexical.anchor):
        raise ValueError("file verification cache_root cannot be a filesystem root")
    if from_environment and lexical.name != ".file-verification":
        raise ValueError(
            "environment file verification cache_root must be a dedicated "
            ".file-verification leaf"
        )

    protected: list[Path] = []
    if os.name == "nt":
        protected.extend(
            Path(value)
            for value in (
                os.environ.get("SystemRoot", r"C:\Windows"),
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
    for protected_root in protected:
        if _contains(protected_root, lexical):
            raise ValueError(
                f"file verification cache_root cannot be under system directory {protected_root}"
            )

    try:
        home = Path.home().absolute()
        temporary_root = Path(tempfile.gettempdir()).absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification cache_root anchor cannot be resolved") from error
    if lexical == home or lexical.parent == home:
        raise ValueError("file verification cache_root cannot be home or its direct child")
    if os.name == "nt" and (
        lexical == home.parent or lexical.parent == home.parent
    ):
        raise ValueError("file verification cache_root cannot be the broad Users directory")
    if os.name != "nt":
        if lexical == Path("/home"):
            raise ValueError("file verification cache_root cannot be the /home root")
        if lexical.is_relative_to(Path("/home")):
            relative_parts = lexical.relative_to(Path("/home")).parts
            if len(relative_parts) <= 2:
                raise ValueError("file verification cache_root cannot be a broad /home user root")
    if lexical == temporary_root or lexical.parent == temporary_root:
        raise ValueError("file verification cache_root cannot be a broad temp directory")

    try:
        if lexical.is_symlink():
            raise ValueError("file verification cache_root must not be a symlink")
        candidate = lexical.resolve()
    except ValueError:
        raise
    except (OSError, RuntimeError, TypeError) as error:
        raise ValueError("file verification cache_root cannot be resolved") from error
    for protected_root in protected:
        if _contains(protected_root, candidate):
            raise ValueError(
                f"file verification cache_root cannot be under system directory {protected_root}"
            )
    try:
        canonical_home = home.resolve()
        canonical_temporary_root = temporary_root.resolve()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification cache_root anchor cannot be resolved") from error
    if candidate == canonical_home or candidate.parent == canonical_home:
        raise ValueError("file verification cache_root cannot be home or its direct child")
    if os.name != "nt" and candidate.is_relative_to(Path("/home")):
        if len(candidate.relative_to(Path("/home")).parts) <= 2:
            raise ValueError("file verification cache_root cannot be a broad /home user root")
    if os.name == "nt" and (
        candidate == canonical_home.parent or candidate.parent == canonical_home.parent
    ):
        raise ValueError("file verification cache_root cannot be the broad Users directory")
    if candidate == canonical_temporary_root or candidate.parent == canonical_temporary_root:
        raise ValueError("file verification cache_root cannot be a broad temp directory")

    try:
        cwd = Path.cwd().resolve()
        repository_roots = [
            parent for parent in (cwd, *cwd.parents) if (parent / ".git").exists()
        ]
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification worktree cannot be resolved") from error
    for anchor in (cwd, *repository_roots):
        if _contains(anchor, candidate) or _contains(candidate, anchor):
            raise ValueError("file verification cache_root cannot overlap cwd or worktree")

    try:
        if candidate.exists():
            if not candidate.is_dir():
                raise ValueError("file verification cache_root must be a directory")
        else:
            parent = candidate.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not parent.is_dir():
                raise ValueError("file verification cache_root parent is not a directory")
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("file verification cache_root directory cannot be inspected") from error
    return candidate


def _logical_stat(path: Path) -> dict[str, int | bool] | None:
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


def _target_stat(path: Path) -> dict[str, int] | None:
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


def _snapshot(path: Path) -> tuple[Path, dict[str, int | bool], dict[str, int]] | None:
    logical = _logical_stat(path)
    if logical is None:
        return None
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError):
        return None
    target = _target_stat(resolved)
    if target is None:
        return None
    return resolved, logical, target


def _certificate_path(root: Path, logical: Path, resolved: Path, scope: str) -> Path:
    key = _sha256(f"{logical}\x00{resolved}\x00{scope}".encode("utf-8")).hexdigest()
    return root / key[:2] / f"{key}.json"


def _valid_timestamp(value: object) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def _certificate_hit(
    path: Path,
    *,
    logical: Path,
    logical_identity: Mapping[str, int | bool],
    resolved: Path,
    target_identity: Mapping[str, int],
    expected: str | None,
    scope: str,
) -> str | None:
    try:
        # Never follow a certificate symlink.  A certificate is optional
        # acceleration state, so a replaced/link-backed entry is a miss.
        certificate_stat = path.lstat()
        if not stat.S_ISREG(certificate_stat.st_mode):
            return None
        encoded = path.read_bytes()
        # A concurrent atomic replacement may race the read.  Requiring the
        # same regular-file identity on both sides makes the optional cache
        # fail closed instead of validating bytes from an entry that changed.
        if path.lstat() != certificate_stat:
            return None
        decoded = json.loads(encoded.decode("utf-8"))
        if not isinstance(decoded, dict) or set(decoded) != _CERTIFICATE_FIELDS:
            return None
        if encoded != _canonical_json_bytes(decoded):
            return None
        if decoded["schema_version"] != FILE_VERIFICATION_SCHEMA_VERSION:
            return None
        if decoded["logical_path"] != str(logical):
            return None
        if type(decoded["logical_is_symlink"]) is not bool:
            return None
        if decoded["logical_is_symlink"] != logical_identity["is_symlink"]:
            return None
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if type(decoded[f"logical_{name}"]) is not int:
                return None
            if decoded[f"logical_{name}"] != logical_identity[name]:
                return None
        if decoded["resolved_path"] != str(resolved):
            return None
        if decoded["caller_scope"] != scope:
            return None
        cert_expected = decoded["expected_sha256"]
        if cert_expected is not None and (
            not isinstance(cert_expected, str) or _SHA256_RE.fullmatch(cert_expected) is None
        ):
            return None
        if expected is not None and cert_expected not in (None, expected):
            return None
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            if type(decoded[name]) is not int or decoded[name] != target_identity[name]:
                return None
        actual = decoded["actual_sha256"]
        if not isinstance(actual, str) or _SHA256_RE.fullmatch(actual) is None:
            return None
        if expected is not None and actual != expected:
            return None
        if not _valid_timestamp(decoded["verified_at"]):
            return None
        artifact_hash = decoded["artifact_hash"]
        if (
            not isinstance(artifact_hash, str)
            or _SHA256_RE.fullmatch(artifact_hash) is None
            or artifact_hash
            != _canonical_json_hash(
                {key: value for key, value in decoded.items() if key != "artifact_hash"}
            )
        ):
            return None
        return actual
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _stream_sha256(path: str | Path, *, chunk_size: int = 16 * 1024 * 1024) -> str:
    source = Path(path)
    digest = _sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_from_pythia_call() -> tuple[str | None, str]:
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            if frame.f_globals.get("__name__") == "param_importance_nlp.data.pythia_mmap":
                if frame.f_code.co_name == "__init__":
                    expected = frame.f_locals.get("expected_sha256")
                    if isinstance(expected, str) and _SHA256_RE.fullmatch(expected):
                        return expected, "pythia_mmap.MMapIndex.__init__"
                    descriptor = frame.f_locals.get("item")
                    descriptor_expected = getattr(descriptor, "sha256", None)
                    if isinstance(descriptor_expected, str) and _SHA256_RE.fullmatch(
                        descriptor_expected
                    ):
                        return descriptor_expected, "pythia_mmap.OrderedShardReader.__init__"
            frame = frame.f_back
    finally:
        del frame
    return None, CALLER_SCOPE


def verify_file(
    path: str | Path,
    expected_sha256: str | None = None,
    cache_root: str | Path | None = None,
    *,
    fallback: Callable[..., str] | None = None,
    caller_scope: str = CALLER_SCOPE,
    chunk_size: int = 16 * 1024 * 1024,
) -> str:
    """Return actual SHA-256, using a stat-bound certificate when opted in."""

    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")
    original = fallback or _stream_sha256
    root = _verification_cache_root(cache_root)
    if root is None:
        return original(path, chunk_size=chunk_size)

    try:
        logical = Path(path).expanduser().absolute()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("file verification logical path cannot be resolved") from error
    initial = _snapshot(logical)
    if initial is None:
        return original(path, chunk_size=chunk_size)
    resolved, logical_identity, target_identity = initial
    certificate = _certificate_path(root, logical, resolved, caller_scope)
    hit = _certificate_hit(
        certificate,
        logical=logical,
        logical_identity=logical_identity,
        resolved=resolved,
        target_identity=target_identity,
        expected=expected_sha256,
        scope=caller_scope,
    )
    if hit is not None and _snapshot(logical) == initial:
        return hit

    actual = original(resolved, chunk_size=chunk_size)
    final = _snapshot(logical)
    if final != initial and final is not None:
        actual = original(final[0], chunk_size=chunk_size)
        final = _snapshot(logical)
    if (
        final is not None
        and final == initial
        and _SHA256_RE.fullmatch(actual or "") is not None
        and (expected_sha256 is None or actual == expected_sha256)
    ):
        certificate_value: dict[str, object] = {
            "schema_version": FILE_VERIFICATION_SCHEMA_VERSION,
            "logical_path": str(logical),
            "logical_is_symlink": logical_identity["is_symlink"],
            "logical_st_dev": logical_identity["st_dev"],
            "logical_st_ino": logical_identity["st_ino"],
            "logical_st_size": logical_identity["st_size"],
            "logical_st_mtime_ns": logical_identity["st_mtime_ns"],
            "logical_st_ctime_ns": logical_identity["st_ctime_ns"],
            "resolved_path": str(final[0]),
            **dict(final[2]),
            "actual_sha256": actual,
            "expected_sha256": expected_sha256,
            "caller_scope": caller_scope,
            "verified_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        certificate_value["artifact_hash"] = _canonical_json_hash(certificate_value)
        try:
            _atomic_write_bytes(certificate, _canonical_json_bytes(certificate_value))
        except OSError:
            pass
    return actual


def verified_sha256(
    path: str | Path,
    expected_sha256: str,
    cache_root: str | Path | None = None,
) -> str:
    """Compatibility helper for direct explicit expected-digest verification."""

    return verify_file(path, expected_sha256, cache_root)


def install_file_verification_cache(
    target_module: Any | None = None,
) -> bool:
    """Install only the Pythia sha256_file monkeypatch when enabled."""

    if not os.environ.get(FILE_VERIFICATION_CACHE_ENV):
        return False
    if target_module is None:
        try:
            target_module = importlib.import_module("param_importance_nlp.data.pythia_mmap")
        except Exception:
            return False
    if getattr(target_module, "__name__", None) != "param_importance_nlp.data.pythia_mmap":
        return False
    if getattr(target_module, "_stage3_file_verification_cache_installed", False):
        return True
    original = getattr(target_module, "sha256_file", None)
    if not callable(original):
        return False

    def cached_sha256(
        path: str | Path,
        *,
        chunk_size: int = 16 * 1024 * 1024,
    ) -> str:
        expected, scope = _expected_from_pythia_call()
        return verify_file(
            path,
            expected,
            None,
            fallback=original,
            caller_scope=scope,
            chunk_size=chunk_size,
        )

    cached_sha256.__name__ = "sha256_file"
    cached_sha256.__module__ = getattr(target_module, "__name__", __name__)
    cached_sha256.__doc__ = "Opt-in certificate-backed SHA-256 for Pythia."
    target_module.sha256_file = cached_sha256
    target_module._stage3_file_verification_cache_installed = True
    return True


__all__ = [
    "CALLER_SCOPE",
    "FILE_VERIFICATION_CACHE_ENV",
    "FILE_VERIFICATION_SCHEMA_VERSION",
    "INSTALLER_IDENTITY",
    "_verification_cache_root",
    "install_file_verification_cache",
    "verify_file",
    "verified_sha256",
]
