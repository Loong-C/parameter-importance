from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import platform
import re
from typing import Any, Final
from urllib.parse import urlsplit

from .atomic import sha256_file, stable_json_hash


ENVIRONMENT_SCHEMA_VERSION: Final = "stage0-environment-v1"
ALLOWED_ENVIRONMENT_ROLES: Final[frozenset[str]] = frozenset(
    {"local-dev", "server-cuda"}
)

_DISTRIBUTION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"==(?P<version>[A-Za-z0-9][A-Za-z0-9.!+_-]*)$"
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_VCS_PREFIXES = ("git+", "hg+", "svn+", "bzr+")


class DependencyFormatError(ValueError):
    """A dependency file contains ambiguous or non-reproducible syntax."""


def normalize_distribution_name(name: str) -> str:
    if not _DISTRIBUTION_RE.fullmatch(name):
        raise DependencyFormatError(f"Invalid distribution name: {name!r}")
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True, slots=True)
class IndexSource:
    option: str
    url: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DistributionPin:
    name: str
    normalized_name: str
    version: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.normalized_name,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class DependencySnapshot:
    source: str
    source_sha256: str
    pins: tuple[DistributionPin, ...]
    index_sources: tuple[IndexSource, ...] = ()

    @property
    def versions(self) -> dict[str, str]:
        return {pin.normalized_name: pin.version for pin in self.pins}

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "index_sources": [item.as_dict() for item in self.index_sources],
            "packages": [
                pin.as_dict()
                for pin in sorted(self.pins, key=lambda item: item.normalized_name)
            ],
        }

    @property
    def semantic_sha256(self) -> str:
        return stable_json_hash(self.semantic_payload())


def _format_error(source: str, line_number: int, message: str) -> DependencyFormatError:
    return DependencyFormatError(f"{source}:{line_number}: {message}")


def _parse_index_source(
    line: str, *, source: str, line_number: int
) -> IndexSource | None:
    match = re.fullmatch(r"(--index-url|--extra-index-url)(?:=|\s+)(\S+)", line)
    if match is None:
        return None
    option, url = match.groups()
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise _format_error(source, line_number, "index URL must be absolute HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise _format_error(source, line_number, "credentials are forbidden in index URLs")
    if parsed.query or parsed.fragment:
        raise _format_error(
            source, line_number, "query/fragment data is forbidden in index URLs"
        )
    return IndexSource(option=option, url=url)


def _reject_unsafe_reference(line: str, *, source: str, line_number: int) -> None:
    lowered = line.casefold()
    if lowered == "-e" or lowered.startswith(("-e ", "--editable")):
        raise _format_error(source, line_number, "editable requirements are forbidden")
    if any(prefix in lowered for prefix in _VCS_PREFIXES):
        raise _format_error(source, line_number, "VCS requirements are forbidden")
    if "@" in line or "://" in line:
        raise _format_error(source, line_number, "direct URL requirements are forbidden")


def _parse_dependency_text(
    text: str,
    *,
    source: str,
    source_sha256: str,
    allow_index_sources: bool,
) -> DependencySnapshot:
    pins: dict[str, DistributionPin] = {}
    indexes: list[IndexSource] = []
    seen_indexes: set[tuple[str, str]] = set()
    primary_index_seen = False

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith("\\"):
            raise _format_error(source, line_number, "line continuations are forbidden")

        index = _parse_index_source(line, source=source, line_number=line_number)
        if index is not None:
            if not allow_index_sources:
                raise _format_error(source, line_number, "pip options are forbidden here")
            identity = (index.option, index.url)
            if identity in seen_indexes:
                raise _format_error(source, line_number, "duplicate index source")
            if index.option == "--index-url":
                if primary_index_seen:
                    raise _format_error(source, line_number, "multiple primary indexes")
                primary_index_seen = True
            seen_indexes.add(identity)
            indexes.append(index)
            continue

        _reject_unsafe_reference(line, source=source, line_number=line_number)
        if line.startswith("-"):
            raise _format_error(source, line_number, "unsupported pip option")
        if ";" in line:
            raise _format_error(source, line_number, "environment markers are forbidden")
        match = _PIN_RE.fullmatch(line)
        if match is None or "*" in line:
            raise _format_error(
                source, line_number, "requirement must be an exact name==version pin"
            )
        name = match.group("name")
        version = match.group("version")
        normalized = normalize_distribution_name(name)
        if normalized in pins:
            raise _format_error(
                source, line_number, f"duplicate distribution: {normalized}"
            )
        pins[normalized] = DistributionPin(name, normalized, version)

    if not pins:
        raise DependencyFormatError(f"{source}: lock/freeze contains no distributions")
    return DependencySnapshot(
        source=source,
        source_sha256=source_sha256,
        pins=tuple(sorted(pins.values(), key=lambda item: item.normalized_name)),
        index_sources=tuple(indexes),
    )


def _decode_dependency_file(path: str | Path) -> tuple[Path, bytes, str]:
    source = Path(path)
    payload = source.read_bytes()
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DependencyFormatError(f"{source}: file must be UTF-8") from error
    return source, payload, text


def parse_requirements_lock(path: str | Path) -> DependencySnapshot:
    source, payload, text = _decode_dependency_file(path)
    return _parse_dependency_text(
        text,
        source=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        allow_index_sources=True,
    )


def parse_requirements_lock_text(
    text: str, *, source: str = "<requirements.lock>"
) -> DependencySnapshot:
    payload = text.encode("utf-8")
    return _parse_dependency_text(
        text,
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        allow_index_sources=True,
    )


def parse_freeze_file(path: str | Path) -> DependencySnapshot:
    source, payload, text = _decode_dependency_file(path)
    return _parse_dependency_text(
        text,
        source=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        allow_index_sources=False,
    )


def parse_freeze_text(
    text: str, *, source: str = "<pip-freeze>"
) -> DependencySnapshot:
    payload = text.encode("utf-8")
    return _parse_dependency_text(
        text,
        source=source,
        source_sha256=hashlib.sha256(payload).hexdigest(),
        allow_index_sources=False,
    )


@dataclass(frozen=True, slots=True)
class VersionMismatch:
    name: str
    locked: str
    installed: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FreezeLockComparison:
    missing: tuple[str, ...]
    extra: tuple[str, ...]
    version_mismatches: tuple[VersionMismatch, ...]

    @property
    def ok(self) -> bool:
        return not (self.missing or self.extra or self.version_mismatches)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "missing": list(self.missing),
            "extra": list(self.extra),
            "version_mismatches": [item.as_dict() for item in self.version_mismatches],
        }


def compare_freeze_to_lock(
    lock: DependencySnapshot, freeze: DependencySnapshot
) -> FreezeLockComparison:
    locked = lock.versions
    installed = freeze.versions
    shared = sorted(locked.keys() & installed.keys())
    mismatches = tuple(
        VersionMismatch(name, locked[name], installed[name])
        for name in shared
        if locked[name] != installed[name]
    )
    return FreezeLockComparison(
        missing=tuple(sorted(locked.keys() - installed.keys())),
        extra=tuple(sorted(installed.keys() - locked.keys())),
        version_mismatches=mismatches,
    )


@dataclass(frozen=True, slots=True)
class WheelhouseEntry:
    filename: str
    size: int
    sha256: str

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WheelhouseManifest:
    source: str
    source_sha256: str
    entries: tuple[WheelhouseEntry, ...]

    @property
    def by_filename(self) -> dict[str, WheelhouseEntry]:
        return {item.filename: item for item in self.entries}

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "files": [
                item.as_dict() for item in sorted(self.entries, key=lambda item: item.filename)
            ]
        }

    @property
    def semantic_sha256(self) -> str:
        return stable_json_hash(self.semantic_payload())


def parse_wheelhouse_manifest(path: str | Path) -> WheelhouseManifest:
    source = Path(path)
    payload = source.read_bytes()
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise DependencyFormatError(f"{source}: wheelhouse manifest must be ASCII") from error

    entries: dict[str, WheelhouseEntry] = {}
    casefolded: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line or raw_line.startswith("#"):
            continue
        fields = raw_line.split("\t")
        if len(fields) != 3:
            raise _format_error(
                str(source), line_number, "expected SHA-256<TAB>size<TAB>filename"
            )
        digest, size_text, filename = fields
        if not _SHA256_RE.fullmatch(digest):
            raise _format_error(str(source), line_number, "invalid SHA-256")
        if not size_text.isdecimal() or int(size_text) <= 0:
            raise _format_error(str(source), line_number, "wheel size must be positive")
        if (
            Path(filename).name != filename
            or "/" in filename
            or "\\" in filename
            or not filename.lower().endswith(".whl")
        ):
            raise _format_error(str(source), line_number, "invalid wheel filename")
        folded = filename.casefold()
        if filename in entries or folded in casefolded:
            raise _format_error(str(source), line_number, "duplicate wheel filename")
        casefolded.add(folded)
        entries[filename] = WheelhouseEntry(filename, int(size_text), digest.lower())

    if not entries:
        raise DependencyFormatError(f"{source}: wheelhouse manifest is empty")
    return WheelhouseManifest(
        source=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        entries=tuple(sorted(entries.values(), key=lambda item: item.filename)),
    )


@dataclass(frozen=True, slots=True)
class FileSizeMismatch:
    filename: str
    expected: int
    actual: int

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FileHashMismatch:
    filename: str
    expected: str
    actual: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class WheelhouseComparison:
    missing_files: tuple[str, ...]
    unexpected_files: tuple[str, ...]
    unsafe_entries: tuple[str, ...]
    size_mismatches: tuple[FileSizeMismatch, ...]
    sha256_mismatches: tuple[FileHashMismatch, ...]
    incompatible_manifest_files: tuple[str, ...]
    uncovered_requirements: tuple[str, ...]
    unmatched_manifest_files: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not any(
            (
                self.missing_files,
                self.unexpected_files,
                self.unsafe_entries,
                self.size_mismatches,
                self.sha256_mismatches,
                self.incompatible_manifest_files,
                self.uncovered_requirements,
                self.unmatched_manifest_files,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "missing_files": list(self.missing_files),
            "unexpected_files": list(self.unexpected_files),
            "unsafe_entries": list(self.unsafe_entries),
            "size_mismatches": [item.as_dict() for item in self.size_mismatches],
            "sha256_mismatches": [item.as_dict() for item in self.sha256_mismatches],
            "incompatible_manifest_files": list(self.incompatible_manifest_files),
            "uncovered_requirements": list(self.uncovered_requirements),
            "unmatched_manifest_files": list(self.unmatched_manifest_files),
        }


def _parse_wheel_identity_and_tags(
    filename: str,
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...]] | None:
    if not filename.casefold().endswith(".whl"):
        return None
    parts = filename[:-4].split("-")
    if len(parts) not in {5, 6}:
        return None
    if len(parts) == 6 and re.fullmatch(r"[0-9][0-9A-Za-z_]*", parts[2]) is None:
        return None
    distribution, version = parts[0], parts[1]
    python_tag, abi_tag, platform_tag = parts[-3:]
    if not distribution or not version:
        return None
    tag_pattern = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")
    if any(
        tag_pattern.fullmatch(value) is None
        for value in (python_tag, abi_tag, platform_tag)
    ):
        return None
    try:
        normalized_distribution = normalize_distribution_name(distribution)
    except DependencyFormatError:
        return None
    return (
        normalized_distribution,
        version,
        tuple(python_tag.casefold().split(".")),
        tuple(abi_tag.casefold().split(".")),
        tuple(platform_tag.casefold().split(".")),
    )


def _linux_cp312_platform_compatible(platform_tag: str) -> bool:
    if platform_tag in {
        "manylinux1_x86_64",
        "manylinux2010_x86_64",
        "manylinux2014_x86_64",
    }:
        return True
    match = re.fullmatch(r"manylinux_2_([0-9]+)_x86_64", platform_tag)
    return match is not None and 5 <= int(match.group(1)) <= 28


def _linux_cp312_tag_compatible(
    python_tag: str, abi_tag: str, platform_tag: str
) -> bool:
    platform_ok = platform_tag == "any" or _linux_cp312_platform_compatible(
        platform_tag
    )
    if not platform_ok:
        return False
    if python_tag in {"py3", "py312"} and abi_tag == "none":
        return True
    if python_tag == "cp312" and abi_tag in {"cp312", "abi3", "none"}:
        return True
    match = re.fullmatch(r"cp([0-9])([0-9]+)", python_tag)
    if match is None or abi_tag != "abi3":
        return False
    version = (int(match.group(1)), int(match.group(2)))
    return (3, 2) <= version <= (3, 12)


def wheel_is_linux_cp312_x86_64_compatible(filename: str) -> bool:
    parsed = _parse_wheel_identity_and_tags(filename)
    if parsed is None:
        return False
    _distribution, _version, python_tags, abi_tags, platform_tags = parsed
    return any(
        _linux_cp312_tag_compatible(python_tag, abi_tag, platform_tag)
        for python_tag in python_tags
        for abi_tag in abi_tags
        for platform_tag in platform_tags
    )


def _wheel_matches_pin(filename: str, pin: DistributionPin) -> bool:
    parsed = _parse_wheel_identity_and_tags(filename)
    if parsed is None or not wheel_is_linux_cp312_x86_64_compatible(filename):
        return False
    distribution, version, _python_tags, _abi_tags, _platform_tags = parsed
    normalized_version = version.replace("_", "-").casefold()
    expected_version = pin.version.replace("_", "-").casefold()
    return distribution == pin.normalized_name and normalized_version == expected_version


def compare_wheelhouse_to_manifest(
    wheelhouse: str | Path,
    manifest: WheelhouseManifest,
    *,
    lock: DependencySnapshot | None = None,
    verify_hashes: bool = True,
) -> WheelhouseComparison:
    root = Path(wheelhouse)
    if not root.is_dir():
        raise FileNotFoundError(root)

    actual_files: dict[str, Path] = {}
    unsafe: list[str] = []
    for item in root.iterdir():
        if item.is_symlink() or not item.is_file():
            unsafe.append(item.name)
        else:
            actual_files[item.name] = item

    expected = manifest.by_filename
    missing = tuple(sorted(expected.keys() - actual_files.keys()))
    unexpected = tuple(sorted(actual_files.keys() - expected.keys()))
    size_mismatches: list[FileSizeMismatch] = []
    hash_mismatches: list[FileHashMismatch] = []
    for filename in sorted(expected.keys() & actual_files.keys()):
        record = expected[filename]
        path = actual_files[filename]
        size = path.stat().st_size
        if size != record.size:
            size_mismatches.append(FileSizeMismatch(filename, record.size, size))
        if verify_hashes:
            digest = sha256_file(path)
            if digest != record.sha256:
                hash_mismatches.append(FileHashMismatch(filename, record.sha256, digest))

    incompatible = tuple(
        sorted(
            entry.filename
            for entry in manifest.entries
            if not wheel_is_linux_cp312_x86_64_compatible(entry.filename)
        )
    )
    uncovered: list[str] = []
    matched_files: set[str] = set()
    if lock is not None:
        for pin in lock.pins:
            candidates = [
                entry.filename
                for entry in manifest.entries
                if _wheel_matches_pin(entry.filename, pin)
            ]
            if not candidates:
                uncovered.append(f"{pin.normalized_name}=={pin.version}")
            matched_files.update(candidates)
    unmatched = (
        tuple(sorted(expected.keys() - matched_files)) if lock is not None else ()
    )
    return WheelhouseComparison(
        missing_files=missing,
        unexpected_files=unexpected,
        unsafe_entries=tuple(sorted(unsafe)),
        size_mismatches=tuple(size_mismatches),
        sha256_mismatches=tuple(hash_mismatches),
        incompatible_manifest_files=incompatible,
        uncovered_requirements=tuple(uncovered),
        unmatched_manifest_files=unmatched,
    )


@dataclass(frozen=True, slots=True)
class RuntimeVersions:
    os_name: str
    os_version: str
    kernel_release: str
    architecture: str
    libc_name: str | None
    libc_version: str | None
    python_implementation: str
    python_version: str
    nvidia_driver_version: str | None = None
    system_cuda_toolkit_version: str | None = None
    torch_version: str | None = None
    torch_cuda_runtime_version: str | None = None
    cudnn_version: str | None = None
    nccl_version: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _optional_version(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def collect_runtime_versions(
    *,
    nvidia_driver_version: str | None = None,
    system_cuda_toolkit_version: str | None = None,
    torch_version: str | None = None,
    torch_cuda_runtime_version: str | None = None,
    cudnn_version: str | None = None,
    nccl_version: str | None = None,
) -> RuntimeVersions:
    """Collect host/Python fields without importing torch or initializing CUDA."""

    libc_name, libc_version = platform.libc_ver()
    return RuntimeVersions(
        os_name=platform.system(),
        os_version=platform.version(),
        kernel_release=platform.release(),
        architecture=platform.machine(),
        libc_name=_optional_version(libc_name),
        libc_version=_optional_version(libc_version),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        nvidia_driver_version=_optional_version(nvidia_driver_version),
        system_cuda_toolkit_version=_optional_version(system_cuda_toolkit_version),
        torch_version=_optional_version(torch_version),
        torch_cuda_runtime_version=_optional_version(torch_cuda_runtime_version),
        cudnn_version=_optional_version(cudnn_version),
        nccl_version=_optional_version(nccl_version),
    )


def _require_sha256(value: str, *, label: str) -> str:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{label} must be a 64-character SHA-256")
    return value.lower()


def _environment_identity_payload(
    *,
    role: str,
    dependency_inputs: Mapping[str, str],
    lock: DependencySnapshot,
    freeze: DependencySnapshot,
    runtime: RuntimeVersions,
    wheelhouse_manifest: WheelhouseManifest | None,
) -> dict[str, Any]:
    if role not in ALLOWED_ENVIRONMENT_ROLES:
        raise ValueError(f"Unknown environment role: {role!r}")
    if not dependency_inputs:
        raise ValueError("At least one dependency input digest is required")
    normalized_inputs = {
        str(name): _require_sha256(str(digest), label=f"dependency input {name!r}")
        for name, digest in sorted(dependency_inputs.items())
    }
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "role": role,
        "dependency_inputs": normalized_inputs,
        "lock_semantic_sha256": lock.semantic_sha256,
        "freeze_semantic_sha256": freeze.semantic_sha256,
        "wheelhouse_semantic_sha256": (
            wheelhouse_manifest.semantic_sha256
            if wheelhouse_manifest is not None
            else None
        ),
        "runtime_versions": runtime.as_dict(),
    }


def compute_environment_id(
    *,
    role: str,
    dependency_inputs: Mapping[str, str],
    lock: DependencySnapshot,
    freeze: DependencySnapshot,
    runtime: RuntimeVersions,
    wheelhouse_manifest: WheelhouseManifest | None = None,
) -> str:
    payload = _environment_identity_payload(
        role=role,
        dependency_inputs=dependency_inputs,
        lock=lock,
        freeze=freeze,
        runtime=runtime,
        wheelhouse_manifest=wheelhouse_manifest,
    )
    return f"env-v1-{stable_json_hash(payload)}"


def build_environment_manifest(
    *,
    role: str,
    environment_path: str | Path,
    dependency_inputs: Mapping[str, str],
    lock: DependencySnapshot,
    freeze: DependencySnapshot,
    runtime: RuntimeVersions,
    wheelhouse_manifest: WheelhouseManifest | None = None,
    gpu_health_report_ref: str | None = None,
) -> dict[str, Any]:
    identity = _environment_identity_payload(
        role=role,
        dependency_inputs=dependency_inputs,
        lock=lock,
        freeze=freeze,
        runtime=runtime,
        wheelhouse_manifest=wheelhouse_manifest,
    )
    resolved_path = str(Path(environment_path).expanduser().resolve(strict=False))
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "environment_id": f"env-v1-{stable_json_hash(identity)}",
        "identity": identity,
        "observations": {
            "environment_path": resolved_path,
            "environment_path_sha256": stable_json_hash(
                {"environment_path": resolved_path}
            ),
            "lock_file_sha256": lock.source_sha256,
            "freeze_file_sha256": freeze.source_sha256,
            "wheelhouse_manifest_file_sha256": (
                wheelhouse_manifest.source_sha256
                if wheelhouse_manifest is not None
                else None
            ),
            "gpu_health_report_ref": gpu_health_report_ref,
        },
    }
