"""S2.3 six-cell parameter-registry producer.

This module is deliberately a control-plane producer.  It never samples a
draw, computes a gradient delta, or writes model weights.  A cell is loaded
sequentially, through the same offline HF model adapter, fixed-state provider,
optimizer factory, and :class:`ParameterRegistry` used by formal runners.

The large model assets remain outside Git.  Only small, canonical JSON
manifests and an index are written to the caller supplied output directory.
Existing output is immutable: an identical replay is accepted and any drift
is rejected.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import subprocess
import argparse
from typing import Any

import torch

from ..assets import validate_asset_path
from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from ..core.registry import ParameterRegistry
from ..providers import (
    FixedStateGradientProvider,
    FrozenSampleResolver,
    OfflineHuggingFaceModelAdapter,
    PythiaMMapFrozenSampleResolver,
    TorchFixedStateGradientProvider,
)
from ..runtime.training_factory import build_optimizer
from ..data.pythia_mmap import PythiaIndexedDataset, PythiaShardDescriptor
from .stage2_assets import (
    AssetResolutionManifest,
    CheckpointRecord,
    DataRangeManifest,
    validate_formal_asset_identity,
)


REGISTRY_MANIFEST_SCHEMA = "stage2-parameter-registry-manifest-v1"
REGISTRY_INDEX_SCHEMA = "stage2-parameter-registry-index-v1"
S203_TASK_ID = "stage2.03_assets_checkpoints_and_sampling"
S203_ARTIFACT_KINDS = (
    "sampling_plan",
    "draw_manifest",
    "asset_resolution",
    "gate_record",
)
_SHA256 = frozenset("0123456789abcdef")


class RegistryProducerError(ValueError):
    """Fail-closed S2.3 registry production error."""


@dataclass(frozen=True, slots=True)
class RegistryCellResult:
    cell_id: str
    manifest_ref: str
    manifest_sha256: str
    manifest_size_bytes: int
    registry_hash: str


@dataclass(frozen=True, slots=True)
class RegistryProductionResult:
    index_ref: str
    index_sha256: str
    cells: tuple[RegistryCellResult, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_logical(value: object, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RegistryProducerError(f"S203_REGISTRY_PATH_INVALID:{field}")
    try:
        normalized = validate_asset_path(value)
    except (TypeError, ValueError) as error:
        raise RegistryProducerError(f"S203_REGISTRY_PATH_ESCAPE:{field}") from error
    return PurePosixPath(normalized)


def _link_like(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attrs = path.lstat().st_file_attributes
    except (FileNotFoundError, OSError, AttributeError):
        attrs = 0
    return bool(attrs & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _real_root(path: str | Path, *, field: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.exists() or not candidate.is_dir() or _link_like(candidate):
        raise RegistryProducerError(f"S203_REGISTRY_ROOT_INVALID:{field}")
    # A root with any reparse ancestor is not an immutable root either.
    current = candidate.absolute()
    for parent in (current, *current.parents):
        if _link_like(parent):
            raise RegistryProducerError(f"S203_REGISTRY_LINK_LIKE_ROOT:{field}")
        if parent.parent == parent:
            break
    return current.resolve(strict=True)


def _safe_join(root: Path, logical: str, *, field: str, directory: bool = False) -> Path:
    parsed = _safe_logical(logical, field=field)
    # Prevent a common operational mistake where --data-root already points at
    # ``.../models`` but a root_ref still starts with ``models/``.
    if parsed.parts and parsed.parts[0].casefold() == root.name.casefold():
        raise RegistryProducerError(f"S203_REGISTRY_DOUBLE_ROOT:{field}")
    current = root
    for part in parsed.parts:
        current = current / part
        if _link_like(current):
            raise RegistryProducerError(f"S203_REGISTRY_LINK_LIKE_PATH:{field}")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise RegistryProducerError(f"S203_REGISTRY_PATH_ESCAPE:{field}") from error
    if directory and not resolved.is_dir():
        raise RegistryProducerError(f"S203_REGISTRY_DIRECTORY_REQUIRED:{field}")
    if not directory and not resolved.is_file():
        raise RegistryProducerError(f"S203_REGISTRY_FILE_REQUIRED:{field}")
    return resolved


def _relative_ref(path: Path, root: Path, *, field: str) -> str:
    try:
        value = path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (ValueError, FileNotFoundError) as error:
        raise RegistryProducerError(f"S203_REGISTRY_REF_OUTSIDE_ROOT:{field}") from error
    _safe_logical(value, field=field)
    return value


def _git_identity(source_root: Path, source_file: Path) -> dict[str, JSONValue]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-c", f"safe.directory={source_root.as_posix()}", "-C", str(source_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise RegistryProducerError("S203_REGISTRY_GIT_UNAVAILABLE") from error
        return completed.stdout.strip()

    try:
        relative = source_file.resolve(strict=True).relative_to(source_root.resolve(strict=True)).as_posix()
    except (ValueError, FileNotFoundError) as error:
        raise RegistryProducerError("S203_REGISTRY_SOURCE_OUTSIDE_ROOT") from error
    commit = run("rev-parse", "HEAD")
    if len(commit) != 40 or any(ch not in _SHA256 for ch in commit):
        raise RegistryProducerError("S203_REGISTRY_SOURCE_COMMIT_INVALID")
    object_id = run("hash-object", relative)
    if len(object_id) != 40 or any(ch not in _SHA256 for ch in object_id):
        raise RegistryProducerError("S203_REGISTRY_SOURCE_OBJECT_INVALID")
    return {
        "repository_root": source_root.as_posix(),
        "source_file": relative,
        "source_commit": commit,
        "source_git_object": object_id,
        "source_sha256": _sha256_file(source_file),
    }


def _resolved_config_identity(
    resolved_config: Mapping[str, Any] | ResolvedConfigV2,
    *,
    path: Path | None,
) -> dict[str, JSONValue]:
    if isinstance(resolved_config, ResolvedConfigV2):
        payload = resolved_config.to_dict()
        config_hash = resolved_config.config_hash
        full_hash = resolved_config.full_hash
    else:
        payload = dict(resolved_config)
        config_hash = canonical_json_hash(payload)
        full_hash = config_hash
    result: dict[str, JSONValue] = {
        "config_hash": config_hash,
        "full_hash": full_hash,
        "payload_sha256": canonical_json_hash(payload),
    }
    if path is not None:
        result.update({"path": path.as_posix(), "file_sha256": _sha256_file(path), "file_size_bytes": path.stat().st_size})
    return result


def _config_sections(config: Mapping[str, Any] | ResolvedConfigV2) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if isinstance(config, ResolvedConfigV2):
        base = config.base_config.to_dict()
        providers = config.section("providers")
        runtime = config.section("runtime") if "runtime" in config.to_dict() else {}
        optimizer_runtime = config.section("optimizer_runtime")
    else:
        payload = dict(config)
        base = payload.get("base_config", payload)
        if not isinstance(base, Mapping):
            raise RegistryProducerError("S203_REGISTRY_CONFIG_BASE_INVALID")
        providers = payload.get("providers", {})
        optimizer_runtime = payload.get("optimizer_runtime", {})
    if not isinstance(base, Mapping) or not isinstance(base.get("optimizer"), Mapping):
        raise RegistryProducerError("S203_REGISTRY_CONFIG_OPTIMIZER_MISSING")
    if not isinstance(providers, Mapping) or not isinstance(optimizer_runtime, Mapping):
        raise RegistryProducerError("S203_REGISTRY_CONFIG_SECTIONS_INVALID")
    return base, providers, optimizer_runtime


def construct_registry_provider(
    model: Any,
    resolver: FrozenSampleResolver,
    *,
    optimizer: Mapping[str, Any],
    optimizer_runtime: Mapping[str, Any] | None,
    fixed_state_id: str,
    output_dtype: torch.dtype = torch.float64,
) -> tuple[TorchFixedStateGradientProvider, ParameterRegistry]:
    """Construct the exact provider/registry pair used by formal execution.

    The order is intentional and is part of the S2.3 producer contract:
    ``build_optimizer`` → ``ParameterRegistry.from_model`` →
    ``TorchFixedStateGradientProvider(..., registry=registry)``.
    """
    try:
        registry_optimizer = build_optimizer(model.module.parameters(), optimizer, optimizer_runtime or {})
        registry = ParameterRegistry.from_model(model.module, registry_optimizer)
        provider = TorchFixedStateGradientProvider(
            model,
            resolver,
            fixed_state_id=fixed_state_id,
            registry=registry,
            output_dtype=output_dtype,
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise RegistryProducerError(f"S203_REGISTRY_PROVIDER_CONSTRUCTION_FAILED:{error}") from error
    if provider.registry_hash != registry.coordinate_registry_hash:
        raise RegistryProducerError("S203_REGISTRY_PROVIDER_REGISTRY_HASH_DRIFT")
    return provider, registry


def _mapping_rows(registry: ParameterRegistry) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for record in registry:
        tags = dict(record.tags)
        rows.append({
            "canonical_name": record.canonical_name,
            "aliases": list(record.aliases),
            "order": record.order,
            "shape": list(record.shape),
            "numel": record.numel,
            "eligible": record.eligible,
            "layer": tags.get("layer"),
            "module": tags.get("module"),
            "module_type": tags.get("module_type"),
            "parameter_role": tags.get("parameter_role"),
        })
    return rows


def _write_immutable(path: Path, payload: Mapping[str, JSONValue]) -> tuple[str, int]:
    if path.exists():
        if _link_like(path):
            raise RegistryProducerError("S203_REGISTRY_OUTPUT_LINK_LIKE")
        existing = load_canonical_json(path)
        if existing != dict(payload):
            raise RegistryProducerError(f"S203_REGISTRY_OUTPUT_DRIFT:{path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(path, dict(payload))
    data = path.read_bytes()
    if load_canonical_json(path) != dict(payload):
        raise RegistryProducerError("S203_REGISTRY_POST_WRITE_DRIFT")
    return _sha256_bytes(data), len(data)


def _checkpoint_files(
    checkpoint: CheckpointRecord,
    *,
    data_root: Path,
    manifest_root: Path,
) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    root = _safe_join(data_root, checkpoint.root_ref, field="checkpoint.root_ref", directory=True)
    actual_files: list[dict[str, JSONValue]] = []
    for descriptor in checkpoint.files:
        path = _safe_join(root, descriptor.path, field=f"checkpoint.files.{descriptor.path}")
        size = path.stat().st_size
        sha = _sha256_file(path)
        if size != descriptor.size_bytes or sha != descriptor.sha256:
            raise RegistryProducerError(f"S203_REGISTRY_CHECKPOINT_FILE_MISMATCH:{descriptor.path}")
        actual_files.append({"path": descriptor.path, "size_bytes": size, "sha256": sha, "role": descriptor.role})
    if checkpoint.manifest_ref is None or checkpoint.manifest_sha256 is None:
        raise RegistryProducerError("S203_REGISTRY_CHECKPOINT_MANIFEST_MISSING")
    manifest_path = _safe_join(manifest_root, checkpoint.manifest_ref, field="checkpoint.manifest_ref")
    manifest_size = manifest_path.stat().st_size
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != checkpoint.manifest_sha256:
        raise RegistryProducerError("S203_REGISTRY_CHECKPOINT_MANIFEST_MISMATCH")
    return (
        {"path": checkpoint.manifest_ref, "size_bytes": manifest_size, "sha256": manifest_sha},
        actual_files,
    )


def _data_range_files(
    data_range: DataRangeManifest,
    *,
    data_root: Path,
    data_asset_root: str | Path,
    manifest_root: Path,
) -> tuple[Path, Path]:
    """Resolve the frozen data files under the qualified data-asset root.

    Stage 2 ``DataFile.path`` values are manifest-relative paths, just like
    ``ResolvedAsset.path_for`` in the formal G3 runtime.  They are *not*
    relative to the storage root: the real server asset lives below
    ``datasets/pile-deduped-pythia-preshuffled``.  The range manifest itself is
    still resolved against the evidence/manifest root and its hash is checked
    before any data object is opened.
    """

    range_manifest = _safe_join(
        manifest_root,
        data_range.manifest_ref,
        field="data_range.manifest_ref",
    )
    if _sha256_file(range_manifest) != data_range.manifest_sha256:
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_MISMATCH")
    asset_root = _real_root(data_asset_root, field="data_asset_root")
    try:
        asset_root.relative_to(data_root)
    except ValueError as error:
        raise RegistryProducerError("S203_REGISTRY_DATA_ASSET_ROOT_OUTSIDE_DATA_ROOT") from error
    descriptors = {item.path: item for item in data_range.files}
    try:
        index_descriptor = descriptors["document.idx"]
        shard_descriptor = descriptors["document-00000-of-00020.bin"]
    except KeyError as error:
        raise RegistryProducerError("S203_REGISTRY_DATA_ALLOWLIST_INVALID") from error

    try:
        raw_manifest = loads_strict_json(range_manifest.read_bytes(), allow_bom=True)
    except (OSError, TypeError, ValueError) as error:
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_INVALID") from error
    if not isinstance(raw_manifest, Mapping):
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_INVALID")

    def _manifest_path(field: str, expected: str) -> None:
        raw_path = raw_manifest.get(field)
        if not isinstance(raw_path, str) or not raw_path:
            raise RegistryProducerError(f"S203_REGISTRY_DATA_MANIFEST_FIELD_INVALID:{field}")
        if "\\" in raw_path or any(part in {"", ".", ".."} for part in PurePosixPath(raw_path).parts):
            raise RegistryProducerError(f"S203_REGISTRY_DATA_MANIFEST_PATH_INVALID:{field}")
        candidate = Path(raw_path)
        if candidate.is_absolute():
            try:
                resolved = candidate.resolve(strict=True)
                resolved.relative_to(asset_root)
            except (FileNotFoundError, ValueError) as error:
                raise RegistryProducerError(f"S203_REGISTRY_DATA_MANIFEST_PATH_ESCAPE:{field}") from error
            if _link_like(candidate) or resolved.relative_to(asset_root).as_posix() != expected:
                raise RegistryProducerError(f"S203_REGISTRY_DATA_MANIFEST_PATH_DRIFT:{field}")
        else:
            try:
                if _safe_logical(raw_path, field=f"data_range.manifest.{field}").as_posix() != expected:
                    raise RegistryProducerError(f"S203_REGISTRY_DATA_MANIFEST_PATH_DRIFT:{field}")
            except RegistryProducerError:
                raise
        return None

    _manifest_path("idx", index_descriptor.path)
    _manifest_path("bin", shard_descriptor.path)
    if raw_manifest.get("idx_sha256") != index_descriptor.sha256:
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_INDEX_DRIFT")
    if raw_manifest.get("bin_sha256") != shard_descriptor.sha256:
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_SHARD_DRIFT")
    if raw_manifest.get("bin_size") != shard_descriptor.size_bytes:
        raise RegistryProducerError("S203_REGISTRY_DATA_MANIFEST_SHARD_SIZE_DRIFT")

    index_path = _safe_join(
        asset_root,
        index_descriptor.path,
        field="data_range.document.idx",
    )
    shard_path = _safe_join(
        asset_root,
        shard_descriptor.path,
        field="data_range.document-00000-of-00020.bin",
    )
    if (
        index_path.stat().st_size != index_descriptor.size_bytes
        or _sha256_file(index_path) != index_descriptor.sha256
    ):
        raise RegistryProducerError("S203_REGISTRY_DATA_INDEX_MISMATCH")
    if (
        shard_path.stat().st_size != shard_descriptor.size_bytes
        or _sha256_file(shard_path) != shard_descriptor.sha256
    ):
        raise RegistryProducerError("S203_REGISTRY_DATA_SHARD_MISMATCH")
    return index_path, shard_path


def _registry_manifest(
    *,
    checkpoint: CheckpointRecord,
    registry: ParameterRegistry,
    actual_manifest: Mapping[str, JSONValue],
    actual_files: Sequence[Mapping[str, JSONValue]],
    config_identity: Mapping[str, JSONValue],
    producer_identity: Mapping[str, JSONValue],
    asset_resolution_hash: str,
) -> dict[str, JSONValue]:
    records = list(registry)
    names = [record.canonical_name for record in records]
    payload: dict[str, JSONValue] = {
        "schema_version": REGISTRY_MANIFEST_SCHEMA,
        "task_id": S203_TASK_ID,
        "scope": "formal",
        "cell": {
            "cell_id": f"{checkpoint.model_id}:{checkpoint.training_stage}",
            "model_id": checkpoint.model_id,
            "training_stage": checkpoint.training_stage,
            "training_step": checkpoint.training_step,
        },
        "model": {"model_id": checkpoint.model_id, "repository": checkpoint.repository},
        "checkpoint": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "revision": checkpoint.revision,
            "root_ref": checkpoint.root_ref,
            "training_step": checkpoint.training_step,
        },
        "asset_resolution_hash": asset_resolution_hash,
        "actual_manifest": dict(actual_manifest),
        "actual_files": [dict(item) for item in actual_files],
        "registry": registry.to_manifest(),
        "registry_hash": registry.coordinate_registry_hash,
        "parameter_count": len(records),
        "eligible_parameter_count": len(registry.eligible_names),
        "parameter_numel": sum(record.numel for record in records),
        "eligible_parameter_numel": sum(record.numel for record in records if record.eligible),
        "parameter_names": names,
        "parameter_order": names,
        "parameter_mapping": _mapping_rows(registry),
        "resolved_config": dict(config_identity),
        "producer": dict(producer_identity),
    }
    payload["manifest_hash"] = canonical_json_hash(payload)
    return payload


def produce_registry_manifests(
    asset_resolution: AssetResolutionManifest | Mapping[str, Any],
    *,
    data_root: str | Path,
    data_asset_root: str | Path | None = None,
    manifest_root: str | Path,
    output_root: str | Path,
    resolved_config: Mapping[str, Any] | ResolvedConfigV2,
    resolved_config_path: str | Path | None = None,
    source_root: str | Path | None = None,
    source_file: str | Path | None = None,
    provider_builder: Callable[[CheckpointRecord, Path], tuple[Any, FrozenSampleResolver]] | None = None,
    formal: bool = True,
) -> RegistryProductionResult:
    """Produce six immutable cell manifests and a lineage index in order.

    ``provider_builder`` is intended for the tiny real-model fixture and test
    doubles.  The default server path uses the offline HF adapter; callers
    should provide a Pythia resolver when they want to bind actual data assets.
    """
    if isinstance(asset_resolution, AssetResolutionManifest):
        assets = asset_resolution
    else:
        assets = AssetResolutionManifest.from_mapping(asset_resolution)
    if formal:
        if assets.scope != "formal" or assets.status != "READY":
            raise RegistryProducerError("S203_REGISTRY_FORMAL_ASSET_RESOLUTION_NOT_READY")
        try:
            validate_formal_asset_identity(assets)
        except ValueError as error:
            raise RegistryProducerError(f"S203_REGISTRY_FORMAL_ASSET_IDENTITY_DRIFT:{error}") from error
    data_root_path = _real_root(data_root, field="data_root")
    manifest_root_path = _real_root(manifest_root, field="manifest_root")
    output_root_path = _real_root(output_root, field="output_root") if Path(output_root).exists() else Path(output_root).absolute()
    if _link_like(output_root_path):
        raise RegistryProducerError("S203_REGISTRY_OUTPUT_LINK_LIKE")
    output_root_path.mkdir(parents=True, exist_ok=True)
    source_root_path = _real_root(source_root or Path(__file__).resolve().parents[2], field="source_root")
    source_file_path = Path(source_file or __file__).resolve()
    producer_identity = _git_identity(source_root_path, source_file_path)
    config_path = None if resolved_config_path is None else Path(resolved_config_path).resolve(strict=True)
    config_identity = _resolved_config_identity(resolved_config, path=config_path)
    base, providers, optimizer_runtime = _config_sections(resolved_config)
    optimizer = base["optimizer"]
    assert isinstance(optimizer, Mapping)
    task_type = str(providers.get("task_type", "causal_lm"))
    shared_resolver: FrozenSampleResolver | None = None
    cells: list[RegistryCellResult] = []
    if provider_builder is None:
        # Formal server path: construct one read-only frozen Pile resolver once
        # and reuse it for all six sequential model loads.  No cursor/draw is
        # touched by registry production.
        data = assets.data_range
        if data_asset_root is None:
            raise RegistryProducerError("S203_REGISTRY_DATA_ASSET_ROOT_REQUIRED")
        data_descriptors = {item.path: item for item in data.files}
        index_descriptor = data_descriptors["document.idx"]
        shard_descriptor = data_descriptors["document-00000-of-00020.bin"]
        index_path, shard_path = _data_range_files(
            data,
            data_root=data_root_path,
            data_asset_root=data_asset_root,
            manifest_root=manifest_root_path,
        )
        if task_type != "causal_lm":
            raise RegistryProducerError("S203_REGISTRY_DEFAULT_RESOLVER_ONLY_SUPPORTS_CAUSAL_LM")
        try:
            dataset = PythiaIndexedDataset(
                index_path,
                [PythiaShardDescriptor(0, shard_path, shard_descriptor.size_bytes, shard_descriptor.sha256)],
                record_start=data.sample_id_min,
                record_stop=data.sample_id_max_exclusive,
                tokens_per_record=data.source_tokens_per_record,
                expected_idx_sha256=index_descriptor.sha256,
            )
            shared_resolver = PythiaMMapFrozenSampleResolver(
                dataset,
                asset_id=data.digest,
                ready_manifest_sha256=data.manifest_sha256,
                qualification_sha256=data.manifest_sha256,
                g3_resolution_artifact_hash=assets.digest,
                g3_source_commit=assets.producer_commit,
                g3_runtime_lineage_sha256=canonical_json_hash({"asset_resolution_hash": assets.digest, "data_range_hash": data.digest}),
                split_start=data.sample_id_min,
                split_stop=data.sample_id_max_exclusive,
                sampling_design=data.sampling_design,
                weights_exogenous=True,
                common_mean_assumption=True,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise RegistryProducerError(f"S203_REGISTRY_PYTHIA_RESOLVER_CONSTRUCTION_FAILED:{error}") from error

        def _default_builder(_checkpoint: CheckpointRecord, root: Path):
            model = OfflineHuggingFaceModelAdapter.from_local_directory(
                root, task_type=task_type, num_labels=providers.get("num_labels"), torch_dtype=torch.float64
            )
            assert shared_resolver is not None
            return model, shared_resolver

        provider_builder = _default_builder
    for checkpoint in assets.checkpoints:
        if not checkpoint.ready or checkpoint.revision is None:
            raise RegistryProducerError(f"S203_REGISTRY_CHECKPOINT_NOT_READY:{checkpoint.checkpoint_id}")
        actual_manifest, actual_files = _checkpoint_files(
            checkpoint, data_root=data_root_path, manifest_root=manifest_root_path
        )
        root = _safe_join(data_root_path, checkpoint.root_ref, field="checkpoint.root_ref", directory=True)
        assert provider_builder is not None
        model, resolver = provider_builder(checkpoint, root)
        provider, registry = construct_registry_provider(
            model,
            resolver,
            optimizer=optimizer,
            optimizer_runtime=optimizer_runtime,
            fixed_state_id=f"offline-{checkpoint.model_id}-{checkpoint.training_stage}-{checkpoint.revision}",
            output_dtype=torch.float64,
        )
        if provider.parameter_names != registry.eligible_names:
            raise RegistryProducerError("S203_REGISTRY_PARAMETER_ORDER_DRIFT")
        if checkpoint.parameter_registry_hash is not None and checkpoint.parameter_registry_hash != registry.coordinate_registry_hash:
            raise RegistryProducerError(
                "S203_REGISTRY_CHECKPOINT_REGISTRY_HASH_MISMATCH:"
                f"checkpoint_id={checkpoint.checkpoint_id}:"
                f"expected={checkpoint.parameter_registry_hash}:"
                f"observed={registry.coordinate_registry_hash}"
            )
        payload = _registry_manifest(
            checkpoint=checkpoint,
            registry=registry,
            actual_manifest=actual_manifest,
            actual_files=actual_files,
            config_identity=config_identity,
            producer_identity=producer_identity,
            asset_resolution_hash=assets.digest,
        )
        cell_id = f"{checkpoint.model_id}:{checkpoint.training_stage}"
        path = output_root_path / "manifests" / f"{checkpoint.model_id}-{checkpoint.training_stage}.json"
        sha, size = _write_immutable(path, payload)
        cells.append(RegistryCellResult(cell_id, path.relative_to(output_root_path).as_posix(), sha, size, registry.coordinate_registry_hash))
    if shared_resolver is not None:
        close = getattr(shared_resolver, "close", None)
        if callable(close):
            close()
    if len(cells) != 6:
        raise RegistryProducerError("S203_REGISTRY_CELL_COUNT_INVALID")
    rows: list[dict[str, JSONValue]] = [
        {
            "cell_id": cell.cell_id,
            "manifest_ref": cell.manifest_ref,
            "manifest_sha256": cell.manifest_sha256,
            "manifest_size_bytes": cell.manifest_size_bytes,
            "registry_hash": cell.registry_hash,
        }
        for cell in cells
    ]
    index_payload: dict[str, JSONValue] = {
        "schema_version": REGISTRY_INDEX_SCHEMA,
        "task_id": S203_TASK_ID,
        "scope": "formal" if formal else "local_fixture",
        "asset_resolution_artifact_kind": "asset_resolution",
        "asset_resolution_hash": assets.digest,
        "allowed_s203_artifact_kinds": list(S203_ARTIFACT_KINDS),
        "registry_manifests_are_source_artifacts": True,
        "source_artifact_refs": [row["manifest_ref"] for row in rows],
        "cells": rows,
        "producer": producer_identity,
        "resolved_config": config_identity,
    }
    index_payload["index_hash"] = canonical_json_hash(index_payload)
    index_path = output_root_path / "registry-index.json"
    index_sha, _ = _write_immutable(index_path, index_payload)
    return RegistryProductionResult(
        index_ref=index_path.relative_to(output_root_path).as_posix(),
        index_sha256=index_sha,
        cells=tuple(cells),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Offline server entry point; all six cells are processed in order."""
    parser = argparse.ArgumentParser(description="Produce immutable S2.3 six-cell registries")
    parser.add_argument("--asset-resolution", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--data-asset-root", required=True, type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    asset = load_canonical_json(args.asset_resolution)
    config = load_canonical_json(args.resolved_config)
    if not isinstance(asset, Mapping) or not isinstance(config, Mapping):
        raise RegistryProducerError("S203_REGISTRY_CLI_INPUT_NOT_OBJECT")
    resolved: Mapping[str, Any] | ResolvedConfigV2
    if config.get("schema_version") == "resolved-config-v2":
        resolved = ResolvedConfigV2.from_mapping(config)
    else:
        resolved = config
    result = produce_registry_manifests(
        asset,
        data_root=args.data_root,
        data_asset_root=args.data_asset_root,
        manifest_root=args.manifest_root or args.data_root,
        output_root=args.output_root,
        resolved_config=resolved,
        resolved_config_path=args.resolved_config,
        source_root=args.source_root,
    )
    print(json.dumps({
        "index_ref": result.index_ref,
        "index_sha256": result.index_sha256,
        "cells": [
            {"cell_id": item.cell_id, "manifest_ref": item.manifest_ref,
             "manifest_sha256": item.manifest_sha256, "registry_hash": item.registry_hash}
            for item in result.cells
        ],
    }, sort_keys=True))
    return 0


__all__ = [
    "REGISTRY_INDEX_SCHEMA",
    "REGISTRY_MANIFEST_SCHEMA",
    "RegistryCellResult",
    "RegistryProducerError",
    "RegistryProductionResult",
    "S203_ARTIFACT_KINDS",
    "S203_TASK_ID",
    "construct_registry_provider",
    "main",
    "produce_registry_manifests",
]


if __name__ == "__main__":  # pragma: no cover - exercised by server command
    raise SystemExit(main())
