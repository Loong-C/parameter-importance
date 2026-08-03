"""Process-local outbound network guard used by the Stage 0 G9 replay.

The guard is intentionally narrow and auditable: it blocks Python socket calls
whose destination is not loopback/local, records only a hash of the attempted
destination, and leaves loopback traffic available for ``torchrun`` rendezvous.
It is not presented as a host firewall.  The G9 report records this scope
explicitly and combines it with the Hugging Face offline environment flags.
"""

from __future__ import annotations

import atexit
from datetime import datetime, timezone
import errno
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Final


OFFLINE_GUARD_SCHEMA_VERSION: Final = "stage0-python-offline-guard-audit-v1"
OFFLINE_GUARD_ENABLE_ENV: Final = "PARAM_IMPORTANCE_OFFLINE_GUARD"
OFFLINE_GUARD_AUDIT_DIR_ENV: Final = "PARAM_IMPORTANCE_NETWORK_AUDIT_DIR"
OFFLINE_GUARD_ALLOWED_HOSTS_ENV: Final = "PARAM_IMPORTANCE_OFFLINE_ALLOWED_HOSTS"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_bytes(value: object) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return (encoded + "\n").encode("utf-8")


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _configured_local_names() -> frozenset[str]:
    values = {"localhost", "localhost.localdomain", "ip6-localhost"}
    # ``getfqdn`` may itself consult name service before the guard is installed;
    # avoid creating an unaudited bootstrap lookup.
    for candidate in (
        socket.gethostname(),
        os.environ.get("HOSTNAME"),
        os.environ.get("COMPUTERNAME"),
    ):
        if candidate:
            values.add(candidate.casefold().rstrip("."))
    declared = os.environ.get(OFFLINE_GUARD_ALLOWED_HOSTS_ENV, "")
    values.update(
        item.strip().casefold().rstrip(".")
        for item in declared.split(",")
        if item.strip()
    )
    return frozenset(values)


def destination_is_local(destination: object, *, local_names: frozenset[str] | None = None) -> bool:
    """Return whether a socket destination is confined to this host.

    Unix-domain paths are local.  For IP sockets only loopback/unspecified
    addresses and explicitly named local hosts are accepted; private network
    addresses are deliberately *not* treated as offline.
    """

    if isinstance(destination, (str, bytes, os.PathLike)):
        # A bare path is a Unix-domain socket.  Host names are supplied inside
        # an address tuple by the socket APIs intercepted below.
        return True
    if not isinstance(destination, tuple) or not destination:
        return False
    host = destination[0]
    if isinstance(host, bytes):
        host = host.decode("ascii", errors="ignore")
    if not isinstance(host, str) or not host:
        return False
    normalized = host.casefold().rstrip(".")
    names = _configured_local_names() if local_names is None else local_names
    if normalized in names:
        return True
    zone_free = normalized.split("%", 1)[0]
    try:
        address = ipaddress.ip_address(zone_free)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


class _GuardState:
    def __init__(self, audit_dir: Path) -> None:
        self.audit_dir = audit_dir.resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.started_at = _now()
        self.local_names = _configured_local_names()
        self.allowed_local_calls = 0
        self.external_attempts: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._completed = False
        self.path = self.audit_dir / f"python-network-{os.getpid()}-{time.time_ns()}.json"
        self._write(status="ACTIVE")

    def _body(self, *, status: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "schema_version": OFFLINE_GUARD_SCHEMA_VERSION,
            "pid": os.getpid(),
            "started_at": self.started_at,
            "completed_at": _now() if status == "COMPLETE" else None,
            "status": status,
            "scope": "python_socket_layer_with_loopback_allowed",
            "local_names_sha256": _hash(sorted(self.local_names)),
            "allowed_local_calls": self.allowed_local_calls,
            "external_attempts": list(self.external_attempts),
        }
        body["artifact_hash"] = _hash(body)
        return body

    def _write(self, *, status: str) -> None:
        body = self._body(status=status)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        with temporary.open("xb") as handle:
            handle.write(_canonical_bytes(body))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def allow_local(self) -> None:
        with self._lock:
            self.allowed_local_calls += 1

    def block(self, operation: str, destination: object) -> None:
        projection = {
            "operation": operation,
            "destination_sha256": _hash(repr(destination)),
            "destination_type": type(destination).__name__,
        }
        with self._lock:
            self.external_attempts.append(projection)
            self._write(status="ACTIVE")

    def finalize(self) -> None:
        with self._lock:
            if self._completed:
                return
            self._completed = True
            self._write(status="COMPLETE")


_STATE: _GuardState | None = None
_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_SENDTO = socket.socket.sendto
_ORIGINAL_GETADDRINFO = socket.getaddrinfo


def _guarded_connect(sock: socket.socket, address: object) -> Any:
    assert _STATE is not None
    if destination_is_local(address, local_names=_STATE.local_names):
        _STATE.allow_local()
        return _ORIGINAL_CONNECT(sock, address)
    _STATE.block("connect", address)
    raise PermissionError(errno.EACCES, "Stage 0 offline replay blocked external connect")


def _guarded_connect_ex(sock: socket.socket, address: object) -> int:
    assert _STATE is not None
    if destination_is_local(address, local_names=_STATE.local_names):
        _STATE.allow_local()
        return int(_ORIGINAL_CONNECT_EX(sock, address))
    _STATE.block("connect_ex", address)
    return errno.EACCES


def _guarded_sendto(sock: socket.socket, data: bytes, *arguments: object) -> int:
    assert _STATE is not None
    if not arguments:
        raise TypeError("sendto requires a destination")
    address = arguments[-1]
    if destination_is_local(address, local_names=_STATE.local_names):
        _STATE.allow_local()
        return int(_ORIGINAL_SENDTO(sock, data, *arguments))
    _STATE.block("sendto", address)
    raise PermissionError(errno.EACCES, "Stage 0 offline replay blocked external sendto")


def _guarded_getaddrinfo(host: object, *arguments: object, **kwargs: object) -> Any:
    assert _STATE is not None
    if host is None:
        _STATE.allow_local()
        return _ORIGINAL_GETADDRINFO(host, *arguments, **kwargs)
    address = (host, 0)
    if destination_is_local(address, local_names=_STATE.local_names):
        _STATE.allow_local()
        return _ORIGINAL_GETADDRINFO(host, *arguments, **kwargs)
    _STATE.block("getaddrinfo", address)
    raise socket.gaierror(socket.EAI_NONAME, "Stage 0 offline replay blocked external DNS")


def install_offline_guard(*, audit_dir: str | Path | None = None) -> Path:
    """Install the process-local guard once and return its audit file path."""

    global _STATE
    if _STATE is not None:
        return _STATE.path
    selected = audit_dir or os.environ.get(OFFLINE_GUARD_AUDIT_DIR_ENV)
    if selected is None:
        raise RuntimeError("OFFLINE_GUARD_AUDIT_DIR_REQUIRED")
    state = _GuardState(Path(selected))
    _STATE = state
    socket.socket.connect = _guarded_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _guarded_connect_ex  # type: ignore[method-assign]
    socket.socket.sendto = _guarded_sendto  # type: ignore[method-assign]
    socket.getaddrinfo = _guarded_getaddrinfo
    atexit.register(state.finalize)
    return state.path


def finalize_offline_guard() -> None:
    if _STATE is not None:
        _STATE.finalize()


def install_from_environment() -> Path | None:
    if os.environ.get(OFFLINE_GUARD_ENABLE_ENV) != "1":
        return None
    return install_offline_guard()


__all__ = [
    "OFFLINE_GUARD_ALLOWED_HOSTS_ENV",
    "OFFLINE_GUARD_AUDIT_DIR_ENV",
    "OFFLINE_GUARD_ENABLE_ENV",
    "OFFLINE_GUARD_SCHEMA_VERSION",
    "destination_is_local",
    "finalize_offline_guard",
    "install_from_environment",
    "install_offline_guard",
]
