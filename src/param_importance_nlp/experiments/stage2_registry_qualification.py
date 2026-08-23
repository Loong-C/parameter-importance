"""Append-only S2.3 registry qualification and asset-resolution amendment.

The original S2.3 asset-resolution manifest names the provider-derived registry
namespace.  The formal registry producer also materializes the coordinate
registry through ``build_optimizer`` and ``ParameterRegistry.from_model``;
that identity is a distinct, stronger namespace.  This module records that
qualification without editing the original manifest or offline-load evidence.

The amendment is intentionally not an ``asset_resolution`` replacement.  It is
an immutable, lineage-bound envelope containing a materialized v1 manifest and
six qualification files.  Consumers must validate the envelope before handing
the materialized manifest to the existing formal producer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import argparse
import json
from pathlib import Path
from typing import Any

import torch

from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import JSONValue, canonical_json_hash, load_canonical_json
from ..core.errors import RegistryError
from ..core.registry import ParameterRegistry
from ..data.pythia_mmap import PythiaIndexedDataset, PythiaShardDescriptor
from ..providers import (
    FrozenSampleResolver,
    OfflineHuggingFaceModelAdapter,
    PythiaMMapFrozenSampleResolver,
)
from .stage2_assets import AssetResolutionManifest, CheckpointRecord, validate_formal_asset_identity
from .stage2_registry_producer import (
    S203_TASK_ID,
    _checkpoint_files,
    _config_sections,
    _data_range_files,
    _git_identity,
    _link_like,
    _mapping_rows,
    _real_root,
    _relative_ref,
    _resolved_config_identity,
    _safe_join,
    _sha256_file,
    _write_immutable,
    construct_registry_provider,
    RegistryProducerError,
)


REGISTRY_QUALIFICATION_SCHEMA = "stage2-registry-qualification-v1"
REGISTRY_QUALIFICATION_INDEX_SCHEMA = "stage2-registry-qualification-index-v1"
ASSET_RESOLUTION_AMENDMENT_SCHEMA = "stage2-asset-resolution-amendment-v1"
AMENDMENT_KIND = "parameter_registry_coordinate_hash_qualification"
_SHA256 = frozenset("0123456789abcdef")


class RegistryQualificationError(RegistryProducerError):
    """A qualification or amendment violates its immutable lineage contract."""


@dataclass(frozen=True, slots=True)
class RegistryQualificationCell:
    cell_id: str
    qualification_ref: str
    qualification_sha256: str
    qualification_size_bytes: int
    qualification_hash: str
    provider_derived_registry_hash: str
    registry_hash: str
    parameter_count: int
    parameter_numel: int


@dataclass(frozen=True, slots=True)
class RegistryQualificationResult:
    index_ref: str
    index_sha256: str
    amendment_ref: str
    amendment_sha256: str
    cells: tuple[RegistryQualificationCell, ...]


def _fail(code: str, detail: object | None = None) -> None:
    if detail is None:
        raise RegistryQualificationError(code)
    raise RegistryQualificationError(f"{code}:{detail}")


def _require_sha(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in _SHA256 for ch in value):
        _fail("S203_REGISTRY_QUALIFICATION_SHA_INVALID", field)
    return value


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        _fail("S203_REGISTRY_QUALIFICATION_TEXT_INVALID", field)
    return value


def _require_object(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail("S203_REGISTRY_QUALIFICATION_OBJECT_INVALID", field)
    return value


def _output_ref(path: Path, root: Path, *, field: str) -> str:
    """Return a safe relative ref for a path that may not exist yet."""

    try:
        value = path.absolute().resolve(strict=False).relative_to(root.resolve(strict=True)).as_posix()
    except (ValueError, FileNotFoundError) as error:
        raise RegistryQualificationError(f"S203_REGISTRY_QUALIFICATION_REF_OUTSIDE_ROOT:{field}") from error
    if not value or value.startswith("/") or "\\" in value or any(
        part in {"", ".", ".."} for part in value.split("/")
    ):
        _fail("S203_REGISTRY_QUALIFICATION_REF_INVALID", field)
    return value


def _reject_link_like_ancestors(path: Path, *, field: str) -> None:
    for candidate in (path, *path.parents):
        if candidate.exists() and _link_like(candidate):
            _fail("S203_REGISTRY_QUALIFICATION_LINK_LIKE_PATH", field)


def _root(path: str | Path, *, field: str) -> Path:
    try:
        return _real_root(path, field=field)
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error


def _resolve_source_root(source_root: str | Path | None) -> Path:
    """Use the producer's exact source-root contract for Git identity.

    In particular, a detached worktree is an ordinary directory and is
    accepted; a symlink/reparse root, missing path, or non-directory remains
    rejected by the shared ``_real_root`` helper.  Keeping this as a separate
    boundary prevents qualification-only path policy from changing the
    producer's source identity semantics.
    """

    candidate = source_root or Path(__file__).resolve().parents[2]
    try:
        return _real_root(candidate, field="source_root")
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error


def _read_canonical_ref(
    root: Path,
    reference: object,
    expected_sha256: object,
    *,
    field: str,
) -> tuple[Path, Mapping[str, Any], int]:
    ref = _require_text(reference, field=f"{field}.ref")
    expected = _require_sha(expected_sha256, field=f"{field}.sha256")
    try:
        path = _safe_join(root, ref, field=f"{field}.ref")
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error
    observed = _sha256_file(path)
    if observed != expected:
        _fail("S203_REGISTRY_QUALIFICATION_FILE_HASH_MISMATCH", field)
    try:
        raw = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise RegistryQualificationError(f"S203_REGISTRY_QUALIFICATION_JSON_INVALID:{field}") from error
    return path, _require_object(raw, field=field), path.stat().st_size


def _parent_binding(path: Path, manifest_root: Path, assets: AssetResolutionManifest) -> dict[str, JSONValue]:
    return {
        "asset_resolution_ref": _relative_ref(path, manifest_root, field="parent.asset_resolution_ref"),
        "asset_resolution_sha256": _sha256_file(path),
        "asset_resolution_size_bytes": path.stat().st_size,
        "asset_resolution_hash": assets.digest,
    }


def _validate_parent_file(
    path: str | Path,
    supplied: AssetResolutionManifest | Mapping[str, Any],
) -> tuple[AssetResolutionManifest, Mapping[str, Any], str, int]:
    parent_path = Path(path).resolve(strict=True)
    try:
        raw = load_canonical_json(parent_path)
    except (OSError, TypeError, ValueError) as error:
        raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_PARENT_INVALID") from error
    if not isinstance(raw, Mapping):
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_NOT_OBJECT")
    try:
        parent = AssetResolutionManifest.from_mapping(raw)
    except (TypeError, ValueError) as error:
        raise RegistryQualificationError(f"S203_REGISTRY_QUALIFICATION_PARENT_INVALID:{error}") from error
    if isinstance(supplied, AssetResolutionManifest):
        expected = supplied
    else:
        try:
            expected = AssetResolutionManifest.from_mapping(supplied)
        except (TypeError, ValueError) as error:
            raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_INPUT_INVALID") from error
    if parent.to_dict() != expected.to_dict():
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_INPUT_DRIFT")
    return parent, raw, _sha256_file(parent_path), parent_path.stat().st_size


def _validate_checkpoint_files(
    checkpoint: CheckpointRecord,
    *,
    data_root: Path,
    manifest_root: Path,
) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    try:
        actual_manifest, actual_files = _checkpoint_files(
            checkpoint, data_root=data_root, manifest_root=manifest_root
        )
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error
    by_role = {str(item["role"]): str(item["sha256"]) for item in actual_files}
    if by_role.get("config") != checkpoint.config_sha256:
        _fail("S203_REGISTRY_QUALIFICATION_CONFIG_FILE_DRIFT", checkpoint.checkpoint_id)
    if by_role.get("tokenizer") != checkpoint.tokenizer_sha256:
        _fail("S203_REGISTRY_QUALIFICATION_TOKENIZER_FILE_DRIFT", checkpoint.checkpoint_id)
    return actual_manifest, actual_files


def _offline_load_binding(
    checkpoint: CheckpointRecord,
    *,
    data_root: Path,
) -> dict[str, JSONValue]:
    if checkpoint.load_evidence_ref is None or checkpoint.load_evidence_sha256 is None:
        _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_MISSING", checkpoint.checkpoint_id)
    try:
        path = _safe_join(data_root, checkpoint.load_evidence_ref, field="checkpoint.load_evidence_ref")
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error
    observed = _sha256_file(path)
    if observed != checkpoint.load_evidence_sha256:
        _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_HASH_MISMATCH", checkpoint.checkpoint_id)
    try:
        raw = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise RegistryQualificationError(
            f"S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_INVALID:{checkpoint.checkpoint_id}"
        ) from error
    if isinstance(raw, Mapping):
        for key in ("parameter_registry_hash", "registry_hash"):
            if key in raw and raw[key] != checkpoint.parameter_registry_hash:
                _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_REGISTRY_DRIFT", checkpoint.checkpoint_id)
    return {
        "ref": checkpoint.load_evidence_ref,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _qualification_payload(
    *,
    checkpoint: CheckpointRecord,
    registry: ParameterRegistry,
    scope: str,
    parent_binding: Mapping[str, JSONValue],
    actual_manifest: Mapping[str, JSONValue],
    actual_files: Sequence[Mapping[str, JSONValue]],
    offline_load: Mapping[str, JSONValue],
    config_identity: Mapping[str, JSONValue],
    producer_identity: Mapping[str, JSONValue],
) -> dict[str, JSONValue]:
    records = list(registry)
    names = [record.canonical_name for record in records]
    payload: dict[str, JSONValue] = {
        "schema_version": REGISTRY_QUALIFICATION_SCHEMA,
        "task_id": S203_TASK_ID,
        "scope": scope,
        "cell": {
            "cell_id": f"{checkpoint.model_id}:{checkpoint.training_stage}",
            "model_id": checkpoint.model_id,
            "training_stage": checkpoint.training_stage,
            "training_step": checkpoint.training_step,
            "checkpoint_id": checkpoint.checkpoint_id,
        },
        "parent": {
            "asset_resolution": dict(parent_binding),
            "checkpoint_id": checkpoint.checkpoint_id,
            "provider_derived_registry_hash": checkpoint.parameter_registry_hash,
            "offline_load": dict(offline_load),
            "checkpoint_manifest": dict(actual_manifest),
            "checkpoint_files": [dict(item) for item in actual_files],
        },
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
    payload["qualification_hash"] = canonical_json_hash(payload)
    return payload


def _default_provider_builder(
    assets: AssetResolutionManifest,
    *,
    data_root: Path,
    data_asset_root: str | Path,
    manifest_root: Path,
    task_type: str,
    num_labels: object,
) -> tuple[Callable[[CheckpointRecord, Path], tuple[Any, FrozenSampleResolver]], FrozenSampleResolver]:
    data = assets.data_range
    descriptors = {item.path: item for item in data.files}
    try:
        index_descriptor = descriptors["document.idx"]
        shard_descriptor = descriptors["document-00000-of-00020.bin"]
        index_path, shard_path = _data_range_files(
            data,
            data_root=data_root,
            data_asset_root=data_asset_root,
            manifest_root=manifest_root,
        )
    except (KeyError, RegistryProducerError) as error:
        raise RegistryQualificationError(f"S203_REGISTRY_QUALIFICATION_DATA_INVALID:{error}") from error
    if task_type != "causal_lm":
        _fail("S203_REGISTRY_QUALIFICATION_RESOLVER_TASK_UNSUPPORTED", task_type)
    try:
        dataset = PythiaIndexedDataset(
            index_path,
            [PythiaShardDescriptor(0, shard_path, shard_descriptor.size_bytes, shard_descriptor.sha256)],
            record_start=data.sample_id_min,
            record_stop=data.sample_id_max_exclusive,
            tokens_per_record=data.source_tokens_per_record,
            expected_idx_sha256=index_descriptor.sha256,
        )
        resolver = PythiaMMapFrozenSampleResolver(
            dataset,
            asset_id=data.digest,
            ready_manifest_sha256=data.manifest_sha256,
            qualification_sha256=data.manifest_sha256,
            g3_resolution_artifact_hash=assets.digest,
            g3_source_commit=assets.producer_commit,
            g3_runtime_lineage_sha256=canonical_json_hash(
                {"asset_resolution_hash": assets.digest, "data_range_hash": data.digest}
            ),
            split_start=data.sample_id_min,
            split_stop=data.sample_id_max_exclusive,
            sampling_design=data.sampling_design,
            weights_exogenous=True,
            common_mean_assumption=True,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        raise RegistryQualificationError(
            f"S203_REGISTRY_QUALIFICATION_PYTHIA_RESOLVER_FAILED:{error}"
        ) from error

    def builder(_checkpoint: CheckpointRecord, root: Path) -> tuple[Any, FrozenSampleResolver]:
        try:
            model = OfflineHuggingFaceModelAdapter.from_local_directory(
                root,
                task_type=task_type,
                num_labels=num_labels,
                torch_dtype=torch.float64,
            )
        except (OSError, TypeError, ValueError, RuntimeError) as error:
            raise RegistryQualificationError(
                f"S203_REGISTRY_QUALIFICATION_MODEL_LOAD_FAILED:{error}"
            ) from error
        return model, resolver

    return builder, resolver


def _close_resolvers(resolvers: Sequence[object]) -> None:
    seen: set[int] = set()
    for resolver in resolvers:
        if id(resolver) in seen:
            continue
        seen.add(id(resolver))
        close = getattr(resolver, "close", None)
        if callable(close):
            close()


def qualify_registry_assets(
    asset_resolution: AssetResolutionManifest | Mapping[str, Any],
    *,
    asset_resolution_path: str | Path,
    data_root: str | Path,
    data_asset_root: str | Path | None,
    manifest_root: str | Path,
    output_root: str | Path,
    amendment_output: str | Path,
    resolved_config: Mapping[str, Any] | ResolvedConfigV2,
    resolved_config_path: str | Path | None = None,
    source_root: str | Path | None = None,
    source_file: str | Path | None = None,
    provider_builder: Callable[[CheckpointRecord, Path], tuple[Any, FrozenSampleResolver]] | None = None,
    formal: bool = True,
) -> RegistryQualificationResult:
    """Qualify all six cells and publish an immutable amendment.

    The model loop is deliberately sequential.  No resolver draw, gradient,
    delta, or model write is performed.  ``provider_builder`` exists only for
    the tiny real-model fixture; the default uses the same offline model and
    Pythia resolver construction as the formal registry producer.
    """

    manifest_root_path = _root(manifest_root, field="manifest_root")
    data_root_path = _root(data_root, field="data_root")
    parent_path = Path(asset_resolution_path).resolve(strict=True)
    try:
        parent_path.relative_to(manifest_root_path)
    except ValueError as error:
        raise RegistryQualificationError(
            "S203_REGISTRY_QUALIFICATION_PARENT_OUTSIDE_MANIFEST_ROOT"
        ) from error
    parent, parent_raw, parent_sha, parent_size = _validate_parent_file(parent_path, asset_resolution)
    if formal:
        if parent.scope != "formal" or parent.status != "READY":
            _fail("S203_REGISTRY_QUALIFICATION_FORMAL_ASSET_NOT_READY")
        try:
            validate_formal_asset_identity(parent)
        except ValueError as error:
            raise RegistryQualificationError(
                f"S203_REGISTRY_QUALIFICATION_FORMAL_ASSET_IDENTITY_DRIFT:{error}"
            ) from error
    output_root_path = (
        _root(output_root, field="output_root")
        if Path(output_root).exists()
        else Path(output_root).absolute()
    )
    if output_root_path.is_symlink():
        _fail("S203_REGISTRY_QUALIFICATION_OUTPUT_LINK_LIKE")
    _reject_link_like_ancestors(output_root_path, field="output_root")
    _output_ref(output_root_path, manifest_root_path, field="output_root")
    output_root_path.mkdir(parents=True, exist_ok=True)
    amendment_path = Path(amendment_output).absolute()
    if amendment_path.resolve(strict=False) == parent_path.resolve(strict=True):
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_OVERWRITE")
    _reject_link_like_ancestors(amendment_path, field="amendment_output")
    _output_ref(amendment_path, manifest_root_path, field="amendment_output")
    source_root_path = _resolve_source_root(source_root)
    source_file_path = Path(source_file or __file__).resolve(strict=True)
    producer_identity = _git_identity(source_root_path, source_file_path)
    constructor_file = Path(__file__).with_name("stage2_registry_producer.py").resolve(strict=True)
    producer_identity["registry_constructor"] = _git_identity(source_root_path, constructor_file)
    config_path = None if resolved_config_path is None else Path(resolved_config_path).resolve(strict=True)
    config_identity = _resolved_config_identity(resolved_config, path=config_path)
    base, providers, optimizer_runtime = _config_sections(resolved_config)
    optimizer = base["optimizer"]
    if not isinstance(optimizer, Mapping):
        _fail("S203_REGISTRY_QUALIFICATION_OPTIMIZER_INVALID")
    task_type = str(providers.get("task_type", "causal_lm"))
    cleanup: list[object] = []
    if provider_builder is None:
        if data_asset_root is None:
            _fail("S203_REGISTRY_QUALIFICATION_DATA_ASSET_ROOT_REQUIRED")
        provider_builder, shared_resolver = _default_provider_builder(
            parent,
            data_root=data_root_path,
            data_asset_root=data_asset_root,
            manifest_root=manifest_root_path,
            task_type=task_type,
            num_labels=providers.get("num_labels"),
        )
        cleanup.append(shared_resolver)

    parent_binding = _parent_binding(parent_path, manifest_root_path, parent)
    cells: list[RegistryQualificationCell] = []
    rows: list[dict[str, JSONValue]] = []
    try:
        for checkpoint in parent.checkpoints:
            if not checkpoint.ready or checkpoint.revision is None:
                _fail("S203_REGISTRY_QUALIFICATION_CHECKPOINT_NOT_READY", checkpoint.checkpoint_id)
            actual_manifest, actual_files = _validate_checkpoint_files(
                checkpoint, data_root=data_root_path, manifest_root=manifest_root_path
            )
            offline_load = _offline_load_binding(checkpoint, data_root=data_root_path)
            try:
                root = _safe_join(data_root_path, checkpoint.root_ref, field="checkpoint.root_ref", directory=True)
            except RegistryProducerError as error:
                raise RegistryQualificationError(str(error)) from error
            assert provider_builder is not None
            model, resolver = provider_builder(checkpoint, root)
            cleanup.append(resolver)
            provider, registry = construct_registry_provider(
                model,
                resolver,
                optimizer=optimizer,
                optimizer_runtime=optimizer_runtime,
                fixed_state_id=f"offline-{checkpoint.model_id}-{checkpoint.training_stage}-{checkpoint.revision}",
                output_dtype=torch.float64,
            )
            if provider.parameter_names != registry.eligible_names:
                _fail("S203_REGISTRY_QUALIFICATION_PARAMETER_ORDER_DRIFT", checkpoint.checkpoint_id)
            payload = _qualification_payload(
                checkpoint=checkpoint,
                registry=registry,
                scope="formal" if formal else "local_fixture",
                parent_binding=parent_binding,
                actual_manifest=actual_manifest,
                actual_files=actual_files,
                offline_load=offline_load,
                config_identity=config_identity,
                producer_identity=producer_identity,
            )
            cell_id = f"{checkpoint.model_id}:{checkpoint.training_stage}"
            path = output_root_path / "qualifications" / f"{checkpoint.model_id}-{checkpoint.training_stage}.json"
            qualification_ref = _output_ref(path, manifest_root_path, field="qualification_ref")
            qualification_sha, qualification_size = _write_immutable(path, payload)
            cell = RegistryQualificationCell(
                cell_id=cell_id,
                qualification_ref=qualification_ref,
                qualification_sha256=qualification_sha,
                qualification_size_bytes=qualification_size,
                qualification_hash=str(payload["qualification_hash"]),
                provider_derived_registry_hash=str(checkpoint.parameter_registry_hash),
                registry_hash=registry.coordinate_registry_hash,
                parameter_count=len(registry),
                parameter_numel=sum(record.numel for record in registry),
            )
            cells.append(cell)
            rows.append({
                "cell_id": cell.cell_id,
                "checkpoint_id": checkpoint.checkpoint_id,
                "qualification_ref": cell.qualification_ref,
                "qualification_sha256": cell.qualification_sha256,
                "qualification_size_bytes": cell.qualification_size_bytes,
                "qualification_hash": cell.qualification_hash,
                "provider_derived_registry_hash": cell.provider_derived_registry_hash,
                "registry_hash": cell.registry_hash,
                "parameter_count": cell.parameter_count,
                "parameter_numel": cell.parameter_numel,
            })
    finally:
        _close_resolvers(cleanup)
    if len(cells) != 6 or {item.cell_id for item in cells} != {
        f"{item.model_id}:{item.training_stage}" for item in parent.checkpoints
    }:
        _fail("S203_REGISTRY_QUALIFICATION_CELL_COUNT_INVALID")
    index_payload: dict[str, JSONValue] = {
        "schema_version": REGISTRY_QUALIFICATION_INDEX_SCHEMA,
        "task_id": S203_TASK_ID,
        "scope": "formal" if formal else "local_fixture",
        "parent": dict(parent_binding),
        "cells": rows,
        "producer": producer_identity,
        "resolved_config": config_identity,
    }
    index_payload["index_hash"] = canonical_json_hash(index_payload)
    index_path = output_root_path / "registry-qualification-index.json"
    index_ref = _output_ref(index_path, manifest_root_path, field="qualification_index_ref")
    index_sha, _ = _write_immutable(index_path, index_payload)

    new_checkpoints = tuple(
        replace(checkpoint, parameter_registry_hash=rows[index]["registry_hash"])
        for index, checkpoint in enumerate(parent.checkpoints)
    )
    materialized = replace(parent, checkpoints=new_checkpoints).to_dict()
    # This guard makes the amendment contract explicit: only the six registry
    # hash leaves may differ from the parent v1 manifest.
    expected_checkpoints = []
    for index, checkpoint in enumerate(parent.checkpoints):
        expected = checkpoint.to_dict()
        expected["parameter_registry_hash"] = rows[index]["registry_hash"]
        expected_checkpoints.append(expected)
    expected_parent = dict(parent_raw)
    expected_parent["checkpoints"] = expected_checkpoints
    expected_parent["asset_resolution_hash"] = canonical_json_hash({
        key: value for key, value in expected_parent.items() if key != "asset_resolution_hash"
    })
    if materialized != expected_parent:
        _fail("S203_REGISTRY_QUALIFICATION_MATERIALIZATION_DRIFT")
    amendment_payload: dict[str, JSONValue] = {
        "schema_version": ASSET_RESOLUTION_AMENDMENT_SCHEMA,
        "task_id": S203_TASK_ID,
        "scope": "formal" if formal else "local_fixture",
        "amendment_kind": AMENDMENT_KIND,
        "parent": dict(parent_binding),
        "qualification_index": {
            "ref": index_ref,
            "sha256": index_sha,
            "size_bytes": index_path.stat().st_size,
            "index_hash": index_payload["index_hash"],
        },
        "qualification_cells": rows,
        "materialized_asset_resolution": materialized,
        "producer": producer_identity,
        "resolved_config": config_identity,
    }
    amendment_payload["amendment_hash"] = canonical_json_hash(amendment_payload)
    amendment_ref = _output_ref(amendment_path, manifest_root_path, field="amendment_ref")
    amendment_sha, _ = _write_immutable(amendment_path, amendment_payload)
    return RegistryQualificationResult(
        index_ref=index_ref,
        index_sha256=index_sha,
        amendment_ref=amendment_ref,
        amendment_sha256=amendment_sha,
        cells=tuple(cells),
    )


_AMENDMENT_FIELDS = {
    "schema_version", "task_id", "scope", "amendment_kind", "parent",
    "qualification_index", "qualification_cells", "materialized_asset_resolution",
    "producer", "resolved_config", "amendment_hash",
}
_PARENT_FIELDS = {
    "asset_resolution_ref", "asset_resolution_sha256", "asset_resolution_size_bytes",
    "asset_resolution_hash",
}
_INDEX_FIELDS = {
    "schema_version", "task_id", "scope", "parent", "cells", "producer",
    "resolved_config", "index_hash",
}
_CELL_FIELDS = {
    "cell_id", "checkpoint_id", "qualification_ref", "qualification_sha256",
    "qualification_size_bytes", "qualification_hash", "provider_derived_registry_hash",
    "registry_hash", "parameter_count", "parameter_numel",
}


def _check_parent_binding(value: object, expected: Mapping[str, JSONValue]) -> None:
    parent = _require_object(value, field="parent")
    if set(parent) != _PARENT_FIELDS or dict(parent) != dict(expected):
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_LINEAGE_MISMATCH")


def _check_hash_envelope(raw: Mapping[str, Any], field: str) -> dict[str, Any]:
    body = dict(raw)
    declared = body.pop(field, None)
    _require_sha(declared, field=field)
    if canonical_json_hash(body) != declared:
        _fail("S203_REGISTRY_QUALIFICATION_CONTENT_HASH_MISMATCH", field)
    return body


def _check_qualification_evidence(
    raw: Mapping[str, Any],
    *,
    checkpoint: CheckpointRecord,
    scope: object,
    parent_binding: Mapping[str, JSONValue],
    data_root: Path,
    manifest_root: Path,
) -> tuple[str, str]:
    body = _check_hash_envelope(raw, "qualification_hash")
    required = {
        "schema_version", "task_id", "scope", "cell", "parent", "registry", "registry_hash",
        "parameter_count", "eligible_parameter_count", "parameter_numel",
        "eligible_parameter_numel", "parameter_names", "parameter_order", "parameter_mapping",
        "resolved_config", "producer",
    }
    if (
        set(body) != required
        or body["schema_version"] != REGISTRY_QUALIFICATION_SCHEMA
        or body["scope"] != scope
    ):
        _fail("S203_REGISTRY_QUALIFICATION_EVIDENCE_FIELDS_INVALID", checkpoint.checkpoint_id)
    cell = _require_object(body["cell"], field="evidence.cell")
    expected_cell = {
        "cell_id": f"{checkpoint.model_id}:{checkpoint.training_stage}",
        "model_id": checkpoint.model_id,
        "training_stage": checkpoint.training_stage,
        "training_step": checkpoint.training_step,
        "checkpoint_id": checkpoint.checkpoint_id,
    }
    if set(cell) != set(expected_cell) or dict(cell) != expected_cell:
        _fail("S203_REGISTRY_QUALIFICATION_EVIDENCE_CELL_MISMATCH", checkpoint.checkpoint_id)
    parent = _require_object(body["parent"], field="evidence.parent")
    expected_parent_keys = {
        "asset_resolution", "checkpoint_id", "provider_derived_registry_hash",
        "offline_load", "checkpoint_manifest", "checkpoint_files",
    }
    if set(parent) != expected_parent_keys or parent["checkpoint_id"] != checkpoint.checkpoint_id:
        _fail("S203_REGISTRY_QUALIFICATION_EVIDENCE_PARENT_INVALID", checkpoint.checkpoint_id)
    _check_parent_binding(parent["asset_resolution"], parent_binding)
    _require_sha(parent["provider_derived_registry_hash"], field="provider_derived_registry_hash")
    if parent["provider_derived_registry_hash"] != checkpoint.parameter_registry_hash:
        _fail("S203_REGISTRY_QUALIFICATION_OLD_HASH_MISMATCH", checkpoint.checkpoint_id)
    offline = _require_object(parent["offline_load"], field="evidence.offline_load")
    if set(offline) != {"ref", "sha256", "size_bytes"}:
        _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_FIELDS_INVALID", checkpoint.checkpoint_id)
    offline_path, _, offline_size = _read_canonical_ref(
        data_root, offline["ref"], offline["sha256"], field="evidence.offline_load"
    )
    if offline_size != offline["size_bytes"]:
        _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_SIZE_MISMATCH", checkpoint.checkpoint_id)
    if checkpoint.load_evidence_ref != offline["ref"] or checkpoint.load_evidence_sha256 != offline["sha256"]:
        _fail("S203_REGISTRY_QUALIFICATION_OFFLINE_LOAD_PARENT_MISMATCH", checkpoint.checkpoint_id)
    actual_manifest, actual_files = _validate_checkpoint_files(
        checkpoint, data_root=data_root, manifest_root=manifest_root
    )
    if parent["checkpoint_manifest"] != actual_manifest or parent["checkpoint_files"] != actual_files:
        _fail("S203_REGISTRY_QUALIFICATION_CHECKPOINT_FILE_LINEAGE_MISMATCH", checkpoint.checkpoint_id)
    registry_raw = _require_object(body["registry"], field="evidence.registry")
    try:
        registry = ParameterRegistry.from_manifest(registry_raw)
    except (TypeError, ValueError, RegistryError) as error:
        raise RegistryQualificationError(
            f"S203_REGISTRY_QUALIFICATION_REGISTRY_INVALID:{checkpoint.checkpoint_id}"
        ) from error
    registry_hash = _require_sha(body["registry_hash"], field="evidence.registry_hash")
    if registry_hash != registry.coordinate_registry_hash:
        _fail("S203_REGISTRY_QUALIFICATION_REGISTRY_HASH_MISMATCH", checkpoint.checkpoint_id)
    names = [record.canonical_name for record in registry]
    if body["parameter_names"] != names or body["parameter_order"] != names:
        _fail("S203_REGISTRY_QUALIFICATION_PARAMETER_ORDER_MISMATCH", checkpoint.checkpoint_id)
    if body["parameter_mapping"] != _mapping_rows(registry):
        _fail("S203_REGISTRY_QUALIFICATION_PARAMETER_MAPPING_MISMATCH", checkpoint.checkpoint_id)
    if body["parameter_count"] != len(registry) or body["eligible_parameter_count"] != len(registry.eligible_names):
        _fail("S203_REGISTRY_QUALIFICATION_PARAMETER_COUNT_MISMATCH", checkpoint.checkpoint_id)
    if body["parameter_numel"] != sum(record.numel for record in registry):
        _fail("S203_REGISTRY_QUALIFICATION_PARAMETER_NUMEL_MISMATCH", checkpoint.checkpoint_id)
    if body["eligible_parameter_numel"] != sum(record.numel for record in registry if record.eligible):
        _fail("S203_REGISTRY_QUALIFICATION_ELIGIBLE_NUMEL_MISMATCH", checkpoint.checkpoint_id)
    return str(body["registry_hash"]), str(raw["qualification_hash"])


def load_asset_resolution_input(
    path: str | Path,
    *,
    root: str | Path,
    data_root: str | Path | None = None,
) -> Mapping[str, JSONValue]:
    """Strictly load either a v1 asset manifest or a qualification amendment."""

    manifest_root = _root(root, field="manifest_root")
    data_root_path = _root(data_root or root, field="data_root")
    source = Path(path).resolve(strict=True)
    try:
        source.relative_to(manifest_root)
    except ValueError as error:
        raise RegistryQualificationError(
            "S203_REGISTRY_QUALIFICATION_INPUT_OUTSIDE_ROOT"
        ) from error
    try:
        raw = load_canonical_json(source)
    except (OSError, TypeError, ValueError) as error:
        raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_INPUT_INVALID") from error
    top = _require_object(raw, field="asset_resolution_input")
    if top.get("schema_version") == "stage2-asset-resolution-v1":
        try:
            AssetResolutionManifest.from_mapping(top)
        except (TypeError, ValueError) as error:
            raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_PARENT_INVALID") from error
        return top
    if set(top) != _AMENDMENT_FIELDS or top.get("schema_version") != ASSET_RESOLUTION_AMENDMENT_SCHEMA:
        _fail("S203_REGISTRY_QUALIFICATION_AMENDMENT_FIELDS_INVALID")
    body = _check_hash_envelope(top, "amendment_hash")
    if body["task_id"] != S203_TASK_ID or body["amendment_kind"] != AMENDMENT_KIND:
        _fail("S203_REGISTRY_QUALIFICATION_AMENDMENT_IDENTITY_INVALID")
    if body["scope"] not in {"formal", "local_fixture"}:
        _fail("S203_REGISTRY_QUALIFICATION_SCOPE_INVALID")
    parent_meta = _require_object(body["parent"], field="amendment.parent")
    if set(parent_meta) != _PARENT_FIELDS:
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_FIELDS_INVALID")
    parent_ref = _require_text(parent_meta["asset_resolution_ref"], field="parent.asset_resolution_ref")
    try:
        parent_path = _safe_join(manifest_root, parent_ref, field="parent.asset_resolution_ref")
    except RegistryProducerError as error:
        raise RegistryQualificationError(str(error)) from error
    parent_sha = _sha256_file(parent_path)
    if parent_sha != parent_meta["asset_resolution_sha256"] or parent_path.stat().st_size != parent_meta["asset_resolution_size_bytes"]:
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_FILE_MISMATCH")
    try:
        parent_raw = load_canonical_json(parent_path)
        parent = AssetResolutionManifest.from_mapping(parent_raw)  # type: ignore[arg-type]
    except (OSError, TypeError, ValueError) as error:
        raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_PARENT_INVALID") from error
    if parent.digest != parent_meta["asset_resolution_hash"]:
        _fail("S203_REGISTRY_QUALIFICATION_PARENT_HASH_MISMATCH")
    index_meta = _require_object(body["qualification_index"], field="qualification_index")
    if set(index_meta) != {"ref", "sha256", "size_bytes", "index_hash"}:
        _fail("S203_REGISTRY_QUALIFICATION_INDEX_REF_FIELDS_INVALID")
    index_path, index_raw, index_size = _read_canonical_ref(
        manifest_root, index_meta["ref"], index_meta["sha256"], field="qualification_index"
    )
    if index_size != index_meta["size_bytes"]:
        _fail("S203_REGISTRY_QUALIFICATION_INDEX_SIZE_MISMATCH")
    index_body = _check_hash_envelope(index_raw, "index_hash")
    if canonical_json_hash(index_body) != index_meta["index_hash"]:
        _fail("S203_REGISTRY_QUALIFICATION_INDEX_HASH_MISMATCH")
    if set(index_raw) != _INDEX_FIELDS or index_body["schema_version"] != REGISTRY_QUALIFICATION_INDEX_SCHEMA:
        _fail("S203_REGISTRY_QUALIFICATION_INDEX_FIELDS_INVALID")
    if index_body["task_id"] != S203_TASK_ID:
        _fail("S203_REGISTRY_QUALIFICATION_INDEX_IDENTITY_INVALID")
    if index_body["scope"] != body["scope"]:
        _fail("S203_REGISTRY_QUALIFICATION_SCOPE_MISMATCH")
    _check_parent_binding(index_body["parent"], parent_meta)
    index_cells = index_body["cells"]
    amendment_cells = body["qualification_cells"]
    if not isinstance(index_cells, list) or len(index_cells) != 6 or index_cells != amendment_cells:
        _fail("S203_REGISTRY_QUALIFICATION_CELL_INDEX_MISMATCH")
    checkpoints = parent.checkpoints
    expected_ids = {f"{item.model_id}:{item.training_stage}" for item in checkpoints}
    seen: set[str] = set()
    observed_hashes: dict[str, str] = {}
    for row_index, row_raw in enumerate(index_cells):
        row = _require_object(row_raw, field=f"qualification_cells[{row_index}]")
        if set(row) != _CELL_FIELDS:
            _fail("S203_REGISTRY_QUALIFICATION_CELL_FIELDS_INVALID", row_index)
        cell_id = _require_text(row["cell_id"], field="cell_id")
        if cell_id in seen or cell_id not in expected_ids:
            _fail("S203_REGISTRY_QUALIFICATION_CELL_ID_INVALID", cell_id)
        seen.add(cell_id)
        checkpoint = next(item for item in checkpoints if f"{item.model_id}:{item.training_stage}" == cell_id)
        if row["checkpoint_id"] != checkpoint.checkpoint_id:
            _fail("S203_REGISTRY_QUALIFICATION_CELL_CHECKPOINT_MISMATCH", cell_id)
        qualification_ref = _require_text(row["qualification_ref"], field="qualification_ref")
        try:
            qualification_path = _safe_join(manifest_root, qualification_ref, field="qualification_ref")
        except RegistryProducerError as error:
            raise RegistryQualificationError(str(error)) from error
        if _sha256_file(qualification_path) != row["qualification_sha256"] or qualification_path.stat().st_size != row["qualification_size_bytes"]:
            _fail("S203_REGISTRY_QUALIFICATION_EVIDENCE_FILE_MISMATCH", cell_id)
        try:
            evidence_raw = load_canonical_json(qualification_path)
        except (OSError, TypeError, ValueError) as error:
            raise RegistryQualificationError(f"S203_REGISTRY_QUALIFICATION_EVIDENCE_INVALID:{cell_id}") from error
        evidence = _require_object(evidence_raw, field=f"qualification[{cell_id}]")
        registry_hash, qualification_hash = _check_qualification_evidence(
            evidence,
            checkpoint=checkpoint,
            scope=index_body["scope"],
            parent_binding=parent_meta,
            data_root=data_root_path,
            manifest_root=manifest_root,
        )
        if row["qualification_hash"] != qualification_hash or row["registry_hash"] != registry_hash:
            _fail("S203_REGISTRY_QUALIFICATION_INDEX_EVIDENCE_HASH_MISMATCH", cell_id)
        if row["provider_derived_registry_hash"] != checkpoint.parameter_registry_hash:
            _fail("S203_REGISTRY_QUALIFICATION_INDEX_OLD_HASH_MISMATCH", cell_id)
        observed_hashes[cell_id] = registry_hash
    if seen != expected_ids:
        _fail("S203_REGISTRY_QUALIFICATION_CELL_SET_INVALID")
    try:
        materialized_raw = _require_object(body["materialized_asset_resolution"], field="materialized_asset_resolution")
        materialized = AssetResolutionManifest.from_mapping(materialized_raw)
    except (TypeError, ValueError) as error:
        raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_MATERIALIZED_INVALID") from error
    expected_checkpoints = []
    for checkpoint in checkpoints:
        key = f"{checkpoint.model_id}:{checkpoint.training_stage}"
        expected = checkpoint.to_dict()
        expected["parameter_registry_hash"] = observed_hashes[key]
        expected_checkpoints.append(expected)
    expected_materialized = dict(parent_raw)
    expected_materialized["checkpoints"] = expected_checkpoints
    expected_materialized["asset_resolution_hash"] = canonical_json_hash({
        key: value for key, value in expected_materialized.items() if key != "asset_resolution_hash"
    })
    if materialized.to_dict() != expected_materialized:
        _fail("S203_REGISTRY_QUALIFICATION_MATERIALIZED_PARENT_DRIFT")
    return materialized.to_dict()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Qualify six S2.3 coordinate registries and publish an amendment")
    parser.add_argument("--asset-resolution", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--data-asset-root", required=True, type=Path)
    parser.add_argument("--manifest-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--amendment-output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args(argv)
    asset = load_canonical_json(args.asset_resolution)
    config = load_canonical_json(args.resolved_config)
    if not isinstance(asset, Mapping) or not isinstance(config, Mapping):
        raise RegistryQualificationError("S203_REGISTRY_QUALIFICATION_CLI_INPUT_NOT_OBJECT")
    resolved: Mapping[str, Any] | ResolvedConfigV2
    if config.get("schema_version") == "resolved-config-v2":
        resolved = ResolvedConfigV2.from_mapping(config)
    else:
        resolved = config
    result = qualify_registry_assets(
        asset,
        asset_resolution_path=args.asset_resolution,
        data_root=args.data_root,
        data_asset_root=args.data_asset_root,
        manifest_root=args.manifest_root,
        output_root=args.output_root,
        amendment_output=args.amendment_output,
        resolved_config=resolved,
        resolved_config_path=args.resolved_config,
        source_root=args.source_root,
    )
    print(json.dumps({
        "index_ref": result.index_ref,
        "index_sha256": result.index_sha256,
        "amendment_ref": result.amendment_ref,
        "amendment_sha256": result.amendment_sha256,
        "cells": [
            {
                "cell_id": item.cell_id,
                "qualification_ref": item.qualification_ref,
                "qualification_sha256": item.qualification_sha256,
                "registry_hash": item.registry_hash,
                "provider_derived_registry_hash": item.provider_derived_registry_hash,
            }
            for item in result.cells
        ],
    }, sort_keys=True))
    return 0


__all__ = [
    "AMENDMENT_KIND",
    "ASSET_RESOLUTION_AMENDMENT_SCHEMA",
    "REGISTRY_QUALIFICATION_INDEX_SCHEMA",
    "REGISTRY_QUALIFICATION_SCHEMA",
    "RegistryQualificationCell",
    "RegistryQualificationError",
    "RegistryQualificationResult",
    "load_asset_resolution_input",
    "main",
    "qualify_registry_assets",
]


if __name__ == "__main__":  # pragma: no cover - exercised by server command
    raise SystemExit(main())
