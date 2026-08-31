from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Iterable, Final

from .storage import is_within


DEFAULT_MAX_BYTES: Final = 10 * 1024 * 1024
FORBIDDEN_PARTS: Final[frozenset[str]] = frozenset(
    {
        "checkpoints",
        "datasets",
        "models",
        "wheelhouse",
        "envs",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
    }
)
FORBIDDEN_SUFFIXES: Final[frozenset[str]] = frozenset(
    {
        ".bin",
        ".ckpt",
        ".onnx",
        ".part",
        ".pt",
        ".pth",
        ".safetensors",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".whl",
        ".zip",
    }
)
REVIEW_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".7z", ".bz2", ".dll", ".exe", ".gz", ".npz", ".npy", ".so", ".xz"}
)


@dataclass(frozen=True, slots=True)
class GuardFinding:
    path: str
    size: int
    reason: str


def _suffix(path: Path) -> str:
    lowered = path.name.lower()
    if lowered.endswith(".tar.gz"):
        return ".tar.gz"
    return path.suffix.lower()


def inspect_path(
    path: Path,
    *,
    repo_root: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowlist: frozenset[str] = frozenset(),
) -> GuardFinding | None:
    resolved_repo = repo_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    if not is_within(resolved, resolved_repo):
        return GuardFinding(str(path), 0, "path_outside_repository")
    relative = resolved.relative_to(resolved_repo).as_posix()
    if relative in allowlist:
        return None
    size = resolved.stat().st_size
    parts = {part.lower() for part in Path(relative).parts}
    suffix = _suffix(resolved)
    if parts & FORBIDDEN_PARTS:
        return GuardFinding(relative, size, "forbidden_directory")
    if suffix in FORBIDDEN_SUFFIXES:
        return GuardFinding(relative, size, f"forbidden_type:{suffix}")
    if suffix in REVIEW_SUFFIXES:
        return GuardFinding(relative, size, f"binary_review_required:{suffix}")
    if size > max_bytes:
        return GuardFinding(relative, size, f"exceeds_{max_bytes}_bytes")
    return None


def git_candidate_paths(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).resolve(strict=True)
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    output = subprocess.check_output(command, cwd=root)
    names = [item for item in output.decode("utf-8").split("\0") if item]
    return [root / name for name in names if (root / name).is_file()]


def scan_repository(
    repo_root: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    allowlist: Iterable[str] = (),
) -> list[GuardFinding]:
    root = Path(repo_root).resolve(strict=True)
    allowed = frozenset(Path(value).as_posix() for value in allowlist)
    findings = [
        finding
        for path in git_candidate_paths(root)
        if (finding := inspect_path(
            path, repo_root=root, max_bytes=max_bytes, allowlist=allowed
        ))
        is not None
    ]
    return sorted(findings, key=lambda item: item.path)


def format_findings(findings: Iterable[GuardFinding]) -> str:
    return "\n".join(
        f"{finding.path}\t{finding.size}\t{finding.reason}" for finding in findings
    )
