"""Lexical DATA_ROOT path identity checks shared by formal stage 2 paths.

The formal producers and consumers must inspect a path before resolving it.
Otherwise a symlink can make an out-of-root file appear to have an in-root
canonical reference.  This module intentionally keeps the policy small and
raises ``DataRootPathError`` so each caller can map it to its public blocker.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath


class DataRootPathError(ValueError):
    """A DATA_ROOT or DATA_ROOT-relative path is not an allowed identity."""


def _lexical_absolute(value: str | Path) -> Path:
    path = Path(value)
    # ``absolute`` is deliberately used before ``resolve``: it preserves
    # symlink components for the lexical audit.
    return path.absolute()


def _assert_no_symlink_components(path: Path) -> None:
    """Reject symlinks in every existing lexical component of ``path``."""

    absolute = _lexical_absolute(path)
    anchor = Path(absolute.anchor)
    current = anchor
    for component in absolute.parts[1:]:
        current = current / component
        try:
            if current.is_symlink():
                raise DataRootPathError("SYMLINK_COMPONENT")
        except OSError as error:
            raise DataRootPathError("PATH_UNREADABLE") from error


def resolve_data_root(value: str | Path) -> Path:
    """Return a canonical root after auditing its lexical components."""

    lexical = _lexical_absolute(value)
    _assert_no_symlink_components(lexical)
    try:
        canonical = lexical.resolve()
        if not canonical.is_dir():
            raise DataRootPathError("DATA_ROOT_NOT_DIRECTORY")
    except OSError as error:
        raise DataRootPathError("DATA_ROOT_UNREADABLE") from error
    return canonical


def resolve_no_symlink_path(
    value: str | Path,
    *,
    field: str,
    require_directory: bool = False,
    require_regular_file: bool = False,
) -> Path:
    """Resolve an existing path only after auditing every lexical component.

    This is intentionally independent of ``DATA_ROOT``.  Executor identity
    uses it for the repository and launcher source, where accepting a
    symlinked directory or file would make the recorded Git/source identity
    refer to a different checkout than the caller named.
    """

    lexical = _lexical_absolute(value)
    _assert_no_symlink_components(lexical)
    try:
        canonical = lexical.resolve(strict=True)
    except OSError as error:
        raise DataRootPathError(f"{field}:PATH_UNREADABLE") from error
    if require_directory and not canonical.is_dir():
        raise DataRootPathError(f"{field}:DIRECTORY_REQUIRED")
    if require_regular_file and (not canonical.is_file() or canonical.is_symlink()):
        raise DataRootPathError(f"{field}:REGULAR_FILE_REQUIRED")
    return canonical


def resolve_data_root_ref(
    root: str | Path,
    value: str | Path,
    *,
    field: str,
    allow_absolute: bool = True,
) -> tuple[str, Path]:
    """Resolve one path and return its canonical POSIX DATA_ROOT ref.

    Relative references are required to use POSIX separators and cannot
    contain ``.``, ``..`` or empty lexical components.  Absolute paths are
    accepted only when their *lexical* path is under the lexical root; the
    returned identity is always relative and POSIX.
    """

    root_lexical = _lexical_absolute(root)
    _assert_no_symlink_components(root_lexical)
    root_canonical = resolve_data_root(root_lexical)
    raw = str(value) if isinstance(value, Path) else value
    if not isinstance(raw, str) or not raw:
        raise DataRootPathError(f"{field}:REFERENCE_REQUIRED")
    supplied = Path(raw)
    if supplied.is_absolute():
        if not allow_absolute:
            raise DataRootPathError(f"{field}:ABSOLUTE_REFERENCE_FORBIDDEN")
        candidate_lexical = _lexical_absolute(supplied)
    else:
        if "\\" in raw:
            raise DataRootPathError(f"{field}:NON_CANONICAL_SEPARATOR")
        logical = PurePosixPath(raw)
        if logical.is_absolute() or not logical.parts or any(
            part in {"", ".", ".."} for part in logical.parts
        ):
            raise DataRootPathError(f"{field}:PATH_ESCAPE")
        candidate_lexical = root_lexical.joinpath(*logical.parts)
    try:
        relative = candidate_lexical.relative_to(root_lexical)
    except ValueError as error:
        raise DataRootPathError(f"{field}:OUTSIDE_DATA_ROOT") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise DataRootPathError(f"{field}:PATH_ESCAPE")
    _assert_no_symlink_components(candidate_lexical)
    try:
        canonical = candidate_lexical.resolve()
        canonical.relative_to(root_canonical)
    except (OSError, ValueError) as error:
        raise DataRootPathError(f"{field}:PATH_ESCAPE_OR_UNREADABLE") from error
    return relative.as_posix(), canonical


__all__ = [
    "DataRootPathError",
    "resolve_data_root",
    "resolve_data_root_ref",
    "resolve_no_symlink_path",
]
