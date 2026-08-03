"""Process-wide execution guards for the formal Stage 0 G3 boundary.

This module intentionally depends only on the Python standard library.  G3
modules use it to freeze the bytes from which they were loaded and to block
child-process creation even when a caller cached a launch function before an
offline guard became active.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import hashlib
from pathlib import Path
import re
import sys
import threading
from typing import Final, NoReturn


_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_PROCESS_AUDIT_EVENTS: Final = frozenset(
    {
        "_winapi.CreateProcess",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.system",
        "subprocess.Popen",
    }
)
_PROCESS_AUDIT_PREFIXES: Final = ("os.exec", "os.spawn")
_PROCESS_GUARD_LOCK = threading.RLock()
_active_process_blocker: Callable[[str], NoReturn] | None = None


def freeze_loaded_source_sha256(module_file: str) -> str:
    """Hash ``module_file`` now so later disk changes cannot rewrite history."""

    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError("G3_LOADED_SOURCE_ORIGIN_INVALID")
    path = Path(module_file)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("G3_LOADED_SOURCE_ORIGIN_UNSAFE")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if _SHA256.fullmatch(digest) is None:  # pragma: no cover - hashlib contract
        raise RuntimeError("G3_LOADED_SOURCE_DIGEST_INVALID")
    return digest


def _is_process_creation_event(event: str) -> bool:
    return event in _PROCESS_AUDIT_EVENTS or event.startswith(
        _PROCESS_AUDIT_PREFIXES
    )


def _process_creation_audit_hook(event: str, _args: tuple[object, ...]) -> None:
    blocker = _active_process_blocker
    if blocker is not None and _is_process_creation_event(event):
        blocker(event)


sys.addaudithook(_process_creation_audit_hook)


@contextmanager
def block_process_creation(
    blocker: Callable[[str], NoReturn],
) -> Iterator[None]:
    """Activate one process-wide child-process blocker for an offline guard."""

    if not callable(blocker):
        raise TypeError("blocker must be callable")
    global _active_process_blocker
    with _PROCESS_GUARD_LOCK:
        if _active_process_blocker is not None:
            raise RuntimeError("G3_PROCESS_GUARD_ALREADY_ACTIVE")
        _active_process_blocker = blocker
        try:
            yield
        finally:
            _active_process_blocker = None


__g3_loaded_source_sha256__: Final = freeze_loaded_source_sha256(__file__)


__all__ = [
    "block_process_creation",
    "freeze_loaded_source_sha256",
]
