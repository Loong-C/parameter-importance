"""Generate an auditable Python startup hook for the verification cache shim."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .file_verification_cache import (
    FILE_VERIFICATION_CACHE_ENV,
    INSTALLER_IDENTITY,
)


HOOK_SCHEMA_VERSION = "stage3-file-verification-sitecustomize-v1"


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".sitecustomize-",
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
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hook_source(
    *,
    source_path: Path,
    source_sha256: str,
    source_root: Path,
) -> bytes:
    # Keep the loader self-contained so PYTHONPATH only needs the hook output
    # directory. The source hash and installer identity are visible in the file.
    return (f'''"""Generated Stage 3 verification startup hook.
schema_version = {HOOK_SCHEMA_VERSION!r}
installer_identity = {INSTALLER_IDENTITY!r}
source_path = {str(source_path)!r}
source_sha256 = {source_sha256!r}
"""
from __future__ import annotations

import hashlib as _hashlib
import importlib.util as _importlib_util
import os as _os
from pathlib import Path as _Path
import sys as _sys

_CACHE_ENV = {FILE_VERIFICATION_CACHE_ENV!r}
_SOURCE_PATH = _Path({str(source_path)!r})
_SOURCE_SHA256 = {source_sha256!r}
_SOURCE_ROOT = _Path({str(source_root)!r})
_MODULE_NAME = "_stage3_file_verification_cache_loaded"


def _load_stage3_file_verification_cache() -> None:
    if not _os.environ.get(_CACHE_ENV):
        return
    if not _SOURCE_PATH.is_file():
        return
    try:
        digest = _hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()
        if digest != _SOURCE_SHA256:
            return
        source_root = str(_SOURCE_ROOT)
        if source_root not in _sys.path:
            _sys.path.insert(0, source_root)
        spec = _importlib_util.spec_from_file_location(
            _MODULE_NAME,
            _SOURCE_PATH,
        )
        if spec is None or spec.loader is None:
            return
        module = _importlib_util.module_from_spec(spec)
        _sys.modules[_MODULE_NAME] = module
        spec.loader.exec_module(module)
        module.install_file_verification_cache()
    except Exception:
        # Startup hooks must never change normal interpreter startup. An
        # unavailable or tampered optional shim leaves normal hashing intact.
        return


_load_stage3_file_verification_cache()
''').encode("utf-8")


def install_startup_hook(
    output_dir: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, object]:
    """Install sitecustomize.py and a canonical audit manifest.

    The output directory is a deployment directory, not the certificate root.
    The returned manifest records the exact shim/source and installer identity.
    """

    output = Path(output_dir).expanduser().absolute()
    if output == Path(output.anchor):
        raise ValueError("startup hook output_dir cannot be a filesystem root")
    if repository_root is None:
        repository = Path(__file__).resolve().parents[2]
    else:
        repository = Path(repository_root).expanduser().absolute()
    source_path = (repository / "ops" / "stage3" / "file_verification_cache.py").resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"verification cache source does not exist: {source_path}")
    source_sha256 = _sha256(source_path)
    hook_path = output / "sitecustomize.py"
    _atomic_write(
        hook_path,
        _hook_source(
            source_path=source_path,
            source_sha256=source_sha256,
            source_root=repository / "src",
        ),
    )
    manifest: dict[str, object] = {
        "schema_version": HOOK_SCHEMA_VERSION,
        "hook_path": str(hook_path.resolve()),
        "hook_sha256": _sha256(hook_path),
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "installer_identity": INSTALLER_IDENTITY,
        "installed_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _atomic_write(output / "sitecustomize.manifest.json", _canonical_json_bytes(manifest))
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--repository-root", type=Path, default=None)
    args = parser.parse_args()
    manifest = install_startup_hook(args.output_dir, repository_root=args.repository_root)
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
