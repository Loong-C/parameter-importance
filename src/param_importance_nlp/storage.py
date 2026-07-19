from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Final

from .atomic import atomic_write_bytes, sha256_file


DATA_ROOT_ENV: Final = "PARAM_IMPORTANCE_DATA_ROOT"
REQUIRED_DIRECTORIES: Final[tuple[str, ...]] = (
    "datasets",
    "models",
    "cache",
    "checkpoints",
    "runs",
    "results",
    "reports",
    "manifests",
    "operations",
    "wheelhouse",
    "envs",
    "source",
    "tmp",
)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def is_within(path: str | Path, root: str | Path) -> bool:
    candidate = _resolved(Path(path))
    boundary = _resolved(Path(root))
    try:
        candidate.relative_to(boundary)
    except ValueError:
        return False
    return True


def require_data_root(value: str | Path | None = None) -> Path:
    raw = str(value) if value is not None else os.environ.get(DATA_ROOT_ENV, "")
    if not raw:
        raise RuntimeError(
            f"{DATA_ROOT_ENV} must be set explicitly; implicit home/root/tmp "
            "fallbacks are forbidden"
        )
    supplied = Path(raw).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"DATA_ROOT must be absolute: {raw!r}")
    root = _resolved(supplied)
    forbidden = {_resolved(Path.home())}
    if os.name != "nt":
        forbidden.update({_resolved(Path("/")), _resolved(Path("/tmp"))})
    if root in forbidden:
        raise ValueError(f"Unsafe DATA_ROOT boundary: {root}")
    return root


@dataclass(frozen=True, slots=True)
class StorageLayout:
    root: Path

    @classmethod
    def from_value(cls, value: str | Path | None = None) -> "StorageLayout":
        return cls(require_data_root(value))

    def path(self, kind: str, *parts: str) -> Path:
        if kind not in REQUIRED_DIRECTORIES:
            raise KeyError(f"Unknown DATA_ROOT directory kind: {kind!r}")
        candidate = _resolved(self.root.joinpath(kind, *parts))
        if not is_within(candidate, self.root):
            raise ValueError(f"Path escapes DATA_ROOT: {candidate}")
        return candidate

    def validate(self, *, require_writable: bool = False) -> list[str]:
        failures: list[str] = []
        if not self.root.is_dir():
            failures.append(f"missing_root:{self.root}")
            return failures
        for name in REQUIRED_DIRECTORIES:
            path = self.path(name)
            if not path.is_dir():
                failures.append(f"missing_directory:{name}")
            elif require_writable and not os.access(path, os.R_OK | os.W_OK):
                failures.append(f"not_read_write:{name}")
        return failures


def run_storage_canary(layout: StorageLayout, kind: str) -> dict[str, str | int | bool]:
    """Verify write/read/hash/atomic replace/precise cleanup for one directory."""

    import uuid

    directory = layout.path(kind)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    token = uuid.uuid4().hex
    target = directory / f".stage0-canary-{token}.txt"
    first = f"stage0-canary:{kind}:{token}:first\n".encode()
    second = f"stage0-canary:{kind}:{token}:published\n".encode()
    try:
        atomic_write_bytes(target, first)
        first_hash = sha256_file(target)
        atomic_write_bytes(target, second)
        second_hash = sha256_file(target)
        if target.read_bytes() != second:
            raise RuntimeError(f"Canary content mismatch: {target}")
        if first_hash == second_hash:
            raise RuntimeError(f"Canary publication did not change content: {target}")
        return {
            "kind": kind,
            "path": str(target),
            "first_sha256": first_hash,
            "published_sha256": second_hash,
            "published_size": len(second),
            "ok": True,
        }
    finally:
        target.unlink(missing_ok=True)
