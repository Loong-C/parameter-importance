"""Strict S2.8 production input bridge.

The statistical consumer is deliberately easy to exercise with small JSON
fixtures.  A formal run must not use that fixture path: it consumes the
immutable S2.4 ``reference-result-v1`` candidate and a separately published
G2.3 PASS evaluation, then opens the referenced tensor bundle read-only.  The
large arrays are exposed as ``numpy.memmap`` objects and are never decoded as
one JSON list or copied into a second resident package.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping

import numpy as np

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..runtime.task_artifacts import LoadedTaskArtifact, load_committed_task_artifact
from ..runtime.tensor_bundle import _strict_manifest, _verify_bundle_files


S208_REFERENCE_BUNDLE_SCHEMA = "stage2-s208-reference-bundle-v1"
S208_G23_SCHEMA = "stage2-g23-reference-evaluation-v1"
S208_MEMMAP_SCHEMA = "stage2-s208-memmap-reference-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,511}$")


class S208ProductionBlocked(RuntimeError):
    """Raised when formal S2.8 inputs are absent or not content-addressed."""


def _sha(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S208ProductionBlocked(f"{field}:SHA256_REQUIRED")
    return value


def _ref(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or not _SAFE_REF.fullmatch(value):
        raise S208ProductionBlocked(f"{field}:SAFE_RELATIVE_REFERENCE_REQUIRED")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S208ProductionBlocked(f"{field}:PATH_ESCAPE")
    return path.as_posix()


def _safe_path(root: Path, value: object, field: str) -> Path:
    logical = _ref(value, field)
    current = root
    for part in PurePosixPath(logical).parts:
        current = current / part
        if current.is_symlink():
            raise S208ProductionBlocked(f"{field}:SYMLINK_FORBIDDEN")
    result = current.resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S208ProductionBlocked(f"{field}:PATH_ESCAPE") from error
    return result


def _verify_object_hash(value: Mapping[str, Any], field: str, key: str = "artifact_hash") -> str:
    digest = _sha(value.get(key), f"{field}.{key}")
    body = {name: item for name, item in value.items() if name != key}
    if canonical_json_hash(body) != digest:
        raise S208ProductionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return digest


def _load_json(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise S208ProductionBlocked(f"{field}:CANONICAL_READ_FAILED") from error
    if not isinstance(value, Mapping):
        raise S208ProductionBlocked(f"{field}:OBJECT_REQUIRED")
    return dict(value)


def _load_candidate(root: Path, reference_ref: str, field: str) -> tuple[LoadedTaskArtifact | None, Mapping[str, Any], str]:
    """Load an S2.4 candidate, never a caller-authored vector payload."""

    path = _safe_path(root, reference_ref, f"{field}.ref")
    try:
        loaded = load_committed_task_artifact(root, reference_ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        # A direct reference-result file is useful for a materialized server
        # bundle, but it is still required to be content-addressed and a
        # candidate (formal_eligible=false).  It is not accepted as inline
        # Python data.
        payload = _load_json(path, field)
        if payload.get("schema_version") != "reference-result-v1":
            raise S208ProductionBlocked(f"{field}:S204_CANDIDATE_REQUIRED") from error
        if payload.get("scope") != "formal":
            raise S208ProductionBlocked(f"{field}:S204_FORMAL_CANDIDATE_REQUIRED")
        if payload.get("formal_eligible") is not False:
            raise S208ProductionBlocked(f"{field}:CANDIDATE_MUST_NOT_SELF_QUALIFY")
        digest = _verify_object_hash(payload, field)
        return None, payload, digest
    payload = dict(loaded.payload)
    if loaded.identity.task_id != "stage2.04_reference_target" or loaded.identity.artifact_kind != "reference_result":
        raise S208ProductionBlocked(f"{field}:S204_REFERENCE_ARTIFACT_IDENTITY_INVALID")
    if payload.get("schema_version") != "reference-result-v1" or payload.get("scope") != "formal":
        raise S208ProductionBlocked(f"{field}:S204_FORMAL_CANDIDATE_REQUIRED")
    # S2.4 intentionally remains a candidate.  Qualification comes only from
    # the independent G2.3 artifact below; a payload claiming formal_eligible
    # here is suspicious rather than a qualification shortcut.
    if payload.get("formal_eligible") is not False:
        raise S208ProductionBlocked(f"{field}:CANDIDATE_MUST_NOT_SELF_QUALIFY")
    return loaded, payload, loaded.identity.artifact_hash


def _load_g23(root: Path, gate_ref: str) -> tuple[Mapping[str, Any], str]:
    path = _safe_path(root, gate_ref, "g23_gate")
    payload = _load_json(path, "g23_gate")
    digest = _verify_object_hash(payload, "g23_gate")
    if payload.get("schema_version") == S208_G23_SCHEMA:
        if payload.get("gate_id") not in (None, "stage2.G2.3") or payload.get("status") != "PASS" or payload.get("formal_eligible") is not True:
            raise S208ProductionBlocked("g23_gate:INDEPENDENT_PASS_REQUIRED")
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != 6 or any(not isinstance(item, Mapping) or item.get("status") != "PASS" for item in cells):
            raise S208ProductionBlocked("g23_gate:SIX_CELL_PASS_REQUIRED")
    else:
        # Generic GateRecord is accepted only as a separately loaded path.
        if payload.get("gate_id") != "stage2.G2.3" or payload.get("status") != "PASS" or payload.get("formal_eligible") is False:
            raise S208ProductionBlocked("g23_gate:INDEPENDENT_PASS_REQUIRED")
    return payload, digest


def _bundle_cells(bundle: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    if bundle.get("schema_version") == "stage2-s27-six-cell-plan-v1":
        if bundle.get("task_id") != "stage2.07_main_sweep" or bundle.get("scope") != "formal" or bundle.get("status") != "READY" or bundle.get("formal_eligible") is not True:
            raise S208ProductionBlocked("reference_bundle:S27_FORMAL_PLAN_REQUIRED")
    elif bundle.get("schema_version") == S208_REFERENCE_BUNDLE_SCHEMA:
        if bundle.get("scope") != "formal" or bundle.get("formal_eligible") is not True:
            raise S208ProductionBlocked("reference_bundle:FORMAL_BUNDLE_REQUIRED")
    else:
        raise S208ProductionBlocked("reference_bundle:SCHEMA_REQUIRED")
    _verify_object_hash(bundle, "reference_bundle")
    cells = bundle.get("cells")
    if not isinstance(cells, list) or len(cells) != 6 or any(not isinstance(item, Mapping) for item in cells):
        raise S208ProductionBlocked("reference_bundle:SIX_CELL_ROWS_REQUIRED")
    return tuple(dict(item) for item in cells)  # type: ignore[return-value]


def _wire_to_memmap(value: Any, entries: Mapping[str, Mapping[str, Any]], root: Path, cache: Path, stem: str) -> Any:
    if isinstance(value, Mapping):
        kind = value.get("kind")
        if kind == "tensor_ref":
            identifier = value.get("id")
            entry = entries.get(str(identifier))
            if entry is None:
                raise S208ProductionBlocked(f"tensor_bundle:{stem}:UNKNOWN_TENSOR")
            path = root / str(entry["path"])
            dtype_name = str(entry["dtype"])
            try:
                dtype = np.dtype(dtype_name)
            except TypeError:
                dtype_map = {"float32": np.dtype("<f4"), "float64": np.dtype("<f8"), "float16": np.dtype("<f2"), "int64": np.dtype("<i8"), "int32": np.dtype("<i4"), "int16": np.dtype("<i2"), "int8": np.dtype("i1"), "uint8": np.dtype("u1")}
                if dtype_name not in dtype_map:
                    raise S208ProductionBlocked(f"tensor_bundle:{stem}:UNSUPPORTED_DTYPE")
                dtype = dtype_map[dtype_name]
            shape = tuple(int(item) for item in entry["shape"])
            if int(entry["size"]) != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
                raise S208ProductionBlocked(f"tensor_bundle:{stem}:SIZE_MISMATCH")
            return np.memmap(path, mode="r", dtype=dtype, shape=shape)
        if kind == "dict":
            return {str(_wire_to_memmap(k, entries, root, cache, f"{stem}.key")): _wire_to_memmap(v, entries, root, cache, f"{stem}.{k}") for k, v in value["items"]}
        if kind in {"list", "tuple"}:
            items = [_wire_to_memmap(item, entries, root, cache, f"{stem}[{i}]") for i, item in enumerate(value["items"])]
            return tuple(items) if kind == "tuple" else items
    return value


def _decode_bundle(bundle_path: Path, cache_root: Path, stem: str) -> tuple[Mapping[str, Any], str]:
    try:
        manifest = _strict_manifest(bundle_path / "manifest.json")
        _verify_bundle_files(bundle_path, manifest)
    except (OSError, TypeError, ValueError) as error:
        raise S208ProductionBlocked(f"tensor_bundle:{stem}:INVALID") from error
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    # runtime.tensor_bundle uses stable JSON bytes; canonical_json_hash is the
    # repository's public equivalent and is what S2.4 stores in the candidate.
    manifest_hash = canonical_json_hash(manifest)
    entries = {str(item["id"]): item for item in manifest["tensors"]}
    cache = cache_root / stem
    cache.mkdir(parents=True, exist_ok=True)
    state = _wire_to_memmap(manifest["state"], entries, bundle_path, cache, stem)
    if not isinstance(state, Mapping):
        raise S208ProductionBlocked(f"tensor_bundle:{stem}:STATE_OBJECT_REQUIRED")
    return dict(state), manifest_hash


def _flat_memmap(value: Any, ids: list[str], cache_root: Path, stem: str) -> np.memmap:
    supplied_ids = bool(ids)
    if isinstance(value, np.ndarray):
        array = value.reshape(-1)
        if not np.isfinite(array).all():
            raise S208ProductionBlocked(f"reference:{stem}:NONFINITE")
        if ids and len(ids) != array.size:
            raise S208ProductionBlocked(f"reference:{stem}:COORDINATE_COUNT_MISMATCH")
        if not supplied_ids:
            ids.extend(str(i) for i in range(array.size))
        return array  # type: ignore[return-value]
    if not isinstance(value, Mapping) or not value:
        raise S208ProductionBlocked(f"reference:{stem}:VECTOR_REQUIRED")
    arrays: list[tuple[str, np.ndarray]] = []
    for name in sorted(value):
        raw = np.asarray(value[name])
        if raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
            raise S208ProductionBlocked(f"reference:{stem}:{name}:NONFINITE")
        arrays.append((str(name), raw.reshape(-1)))
    total = sum(int(item.size) for _, item in arrays)
    destination = cache_root / f"{stem}.f64.dat"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size != total * 8:
        raise S208ProductionBlocked(f"reference:{stem}:MEMMAP_COLLISION")
    if not destination.exists():
        with destination.open("wb") as handle:
            handle.truncate(total * 8)
    mapped = np.memmap(destination, mode="r+", dtype=np.float64, shape=(total,))
    offset = 0
    for name, array in arrays:
        length = int(array.size)
        mapped[offset : offset + length] = array.astype(np.float64, copy=False)
        if not supplied_ids:
            if len(arrays) == 1 and length == 1:
                ids.append(name)
            else:
                ids.extend(f"{name}[{i}]" for i in range(length))
        offset += length
    if supplied_ids and len(ids) != total:
        raise S208ProductionBlocked(f"reference:{stem}:COORDINATE_COUNT_MISMATCH")
    mapped.flush()
    mapped.flags.writeable = False
    return mapped


def _candidate_bundle_path(root: Path, payload: Mapping[str, Any], row: Mapping[str, Any], reference_root: Path | None) -> Path:
    bundle_ref = payload.get("tensor_bundle_ref")
    if not isinstance(bundle_ref, str):
        raise S208ProductionBlocked("reference:tensor_bundle_ref:REQUIRED")
    candidates: list[Path] = []
    for base in (reference_root, root):
        if base is not None:
            try:
                candidates.append(_safe_path(base, bundle_ref, "reference.tensor_bundle_ref"))
            except S208ProductionBlocked:
                pass
    if isinstance(row.get("reference_output_root_ref"), str):
        try:
            candidates.append(_safe_path(root, PurePosixPath(str(row["reference_output_root_ref"])) .joinpath(bundle_ref).as_posix(), "reference.reference_output_root_ref"))
        except S208ProductionBlocked:
            pass
    for candidate in candidates:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
    raise S208ProductionBlocked("reference:tensor_bundle:NOT_FOUND")


def _materialize_cell(root: Path, row: Mapping[str, Any], gate_hash: str, cache_root: Path, reference_root: Path | None) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    cell_id = row.get("cell_id")
    if not isinstance(cell_id, str):
        cell_id = str(row.get("anchor_id", ""))
    if not cell_id:
        raise S208ProductionBlocked("reference_bundle:cell_id:REQUIRED")
    reference_ref = row.get("reference_ref", row.get("reference_artifact_ref"))
    loaded, candidate, candidate_hash = _load_candidate(root, _ref(reference_ref, f"reference.{cell_id}.ref"), f"reference.{cell_id}")
    declared = row.get("reference_hash")
    if declared is not None and _sha(declared, f"reference.{cell_id}.hash") != candidate_hash:
        raise S208ProductionBlocked(f"reference.{cell_id}:CANDIDATE_HASH_MISMATCH")
    bundle_path = _candidate_bundle_path(root, candidate, row, reference_root)
    state, bundle_hash = _decode_bundle(bundle_path, cache_root, re.sub(r"[^A-Za-z0-9_.-]", "_", cell_id))
    declared_bundle = candidate.get("tensor_bundle_manifest_hash")
    if _sha(declared_bundle, f"reference.{cell_id}.tensor_bundle_manifest_hash") != bundle_hash:
        raise S208ProductionBlocked(f"reference.{cell_id}:BUNDLE_HASH_MISMATCH")
    ids: list[str] = []
    source_vectors: dict[str, np.memmap] = {}
    for short, long_name in (("bias", "bias_reference"), ("cross", "cross_reference"), ("ranking", "ranking_reference")):
        if long_name not in state:
            raise S208ProductionBlocked(f"reference.{cell_id}:{long_name}:MISSING")
        source_ids = list(state.get("coordinate_ids", [])) if isinstance(state.get("coordinate_ids"), list) else ([] if short != "bias" else ids)
        vector = _flat_memmap(state[long_name], source_ids, cache_root, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', cell_id)}-{short}")
        if short == "bias":
            ids[:] = source_ids
        elif tuple(source_ids) != tuple(ids):
            raise S208ProductionBlocked(f"reference.{cell_id}:COORDINATE_REGISTRY_MISMATCH")
        declared_view = _sha(candidate.get(f"{short}_reference_hash"), f"reference.{cell_id}.{short}_reference_hash")
        # Bind the opened bytes to the candidate's authoritative per-view
        # digest.  The digest is computed in chunks, so it never materializes
        # the complete parameter vector in a second resident array.
        source_digest = _vector_digest_stream(state[long_name]) if isinstance(state[long_name], Mapping) else _flat_digest(vector)
        if declared_view != source_digest:
            raise S208ProductionBlocked(f"reference.{cell_id}:{short}:VIEW_HASH_MISMATCH")
        source_vectors[short] = vector
    raw_blocks = state.get("reference_blocks", state.get("block_vectors"))
    blocks: dict[str, list[np.memmap]] | None = None
    variances: dict[str, np.memmap] | None = None
    uncertainty_metadata: Mapping[str, Any] | None = None
    uncertainty_mode: str
    if isinstance(raw_blocks, Mapping):
        blocks = {}
        for short in ("bias", "cross", "ranking"):
            source = raw_blocks.get(short)
            if not isinstance(source, list) or len(source) < 3:
                raise S208ProductionBlocked(f"reference.{cell_id}:{short}:THREE_BLOCKS_REQUIRED")
            blocks[short] = []
            for index, item in enumerate(source):
                block_ids: list[str] = []
                block = _flat_memmap(item, block_ids, cache_root, f"{re.sub(r'[^A-Za-z0-9_.-]', '_', cell_id)}-{short}-block-{index}")
                if tuple(block_ids) != tuple(ids) or block.shape != source_vectors[short].shape:
                    raise S208ProductionBlocked(f"reference.{cell_id}:{short}:BLOCK_SHAPE_MISMATCH")
                blocks[short].append(block)
        uncertainty_mode = "reference_block_bootstrap"
    else:
        # bounded-online-fp64-v1 intentionally retains the exact delete-one
        # jackknife variance vectors, not the raw block vectors.  S2.8 may use
        # the preregistered equivalent independent-variance combination, but
        # only after binding every opened vector to the S2.4 candidate.
        raw_uncertainty = state.get("uncertainty")
        candidate_metadata = candidate.get("metadata")
        declared_uncertainty = candidate_metadata.get("uncertainty") if isinstance(candidate_metadata, Mapping) else None
        if not isinstance(raw_uncertainty, Mapping) or not isinstance(declared_uncertainty, Mapping):
            raise S208ProductionBlocked(f"reference.{cell_id}:REFERENCE_BLOCKS_OR_BOUNDED_VARIANCE_REQUIRED")
        if (
            declared_uncertainty.get("schema_version") != "stage2-reference-uncertainty-v1"
            or declared_uncertainty.get("estimator") != "block_u_delete_one_jackknife"
            or not isinstance(declared_uncertainty.get("block_count_a"), int)
            or not isinstance(declared_uncertainty.get("block_count_b"), int)
            or int(declared_uncertainty["block_count_a"]) < 3
            or int(declared_uncertainty["block_count_b"]) < 3
        ):
            raise S208ProductionBlocked(f"reference.{cell_id}:BOUNDED_UNCERTAINTY_METADATA_INVALID")
        declared_uncertainty_hash = _sha(
            declared_uncertainty.get("artifact_hash"),
            f"reference.{cell_id}.uncertainty.artifact_hash",
        )
        if canonical_json_hash({key: value for key, value in declared_uncertainty.items() if key != "artifact_hash"}) != declared_uncertainty_hash:
            raise S208ProductionBlocked(f"reference.{cell_id}:BOUNDED_UNCERTAINTY_ARTIFACT_HASH_MISMATCH")
        variances = {}
        for short in ("bias", "cross", "ranking"):
            long_name = f"{short}_variance"
            source = raw_uncertainty.get(long_name)
            variance_ids: list[str] = []
            variance = _flat_memmap(
                source,
                variance_ids,
                cache_root,
                f"{re.sub(r'[^A-Za-z0-9_.-]', '_', cell_id)}-{short}-variance",
            )
            if tuple(variance_ids) != tuple(ids) or variance.shape != source_vectors[short].shape:
                raise S208ProductionBlocked(f"reference.{cell_id}:{short}:VARIANCE_SHAPE_MISMATCH")
            if np.any(variance < 0):
                raise S208ProductionBlocked(f"reference.{cell_id}:{short}:VARIANCE_NEGATIVE")
            declared_variance_hash = _sha(
                declared_uncertainty.get(f"{short}_variance_hash"),
                f"reference.{cell_id}.uncertainty.{short}_variance_hash",
            )
            source_digest = _vector_digest_stream(source) if isinstance(source, Mapping) else _flat_digest(variance)
            if source_digest != declared_variance_hash:
                raise S208ProductionBlocked(f"reference.{cell_id}:{short}:VARIANCE_HASH_MISMATCH")
            variances[short] = variance
        uncertainty_metadata = dict(declared_uncertainty)
        uncertainty_mode = "independent_reference_variance_combination"

    sequence_variance: np.memmap | None = None
    raw_sequence_variance = state.get("sequence_variance")
    candidate_metadata = candidate.get("metadata")
    if raw_sequence_variance is not None:
        if not isinstance(candidate_metadata, Mapping):
            raise S208ProductionBlocked(f"reference.{cell_id}:SEQUENCE_VARIANCE_METADATA_REQUIRED")
        sequence_ids: list[str] = []
        sequence_variance = _flat_memmap(
            raw_sequence_variance,
            sequence_ids,
            cache_root,
            f"{re.sub(r'[^A-Za-z0-9_.-]', '_', cell_id)}-sequence-variance",
        )
        if tuple(sequence_ids) != tuple(ids) or sequence_variance.shape != source_vectors["bias"].shape:
            raise S208ProductionBlocked(f"reference.{cell_id}:SEQUENCE_VARIANCE_SHAPE_MISMATCH")
        if np.any(sequence_variance < 0):
            raise S208ProductionBlocked(f"reference.{cell_id}:SEQUENCE_VARIANCE_NEGATIVE")
        declared_sequence_hash = _sha(
            candidate_metadata.get("sequence_variance_hash"),
            f"reference.{cell_id}.sequence_variance_hash",
        )
        source_digest = _vector_digest_stream(raw_sequence_variance) if isinstance(raw_sequence_variance, Mapping) else _flat_digest(sequence_variance)
        if source_digest != declared_sequence_hash:
            raise S208ProductionBlocked(f"reference.{cell_id}:SEQUENCE_VARIANCE_HASH_MISMATCH")
    elif variances is not None:
        raise S208ProductionBlocked(f"reference.{cell_id}:BOUNDED_SEQUENCE_VARIANCE_REQUIRED")
    result: dict[str, Any] = {
        "schema_version": "reference-result-v1",
        "reference_hash": candidate_hash,
        "candidate_artifact_hash": candidate_hash,
        "candidate_ref": _ref(reference_ref, f"reference.{cell_id}.ref"),
        "formal_eligible": True,
        "coordinate_ids": ids,
        "vectors": source_vectors,
        "reference_uncertainty_mode": uncertainty_mode,
        "metadata": {
            "qualification_gate_hash": gate_hash,
            "candidate_artifact_hash": candidate_hash,
            "tensor_bundle_manifest_hash": bundle_hash,
            "source_schema": S208_MEMMAP_SCHEMA,
        },
        "_streaming_payload": True,
    }
    if blocks is not None:
        result["reference_blocks"] = blocks
    if variances is not None:
        result["reference_variances"] = variances
        result["reference_uncertainty"] = dict(uncertainty_metadata or {})
    if sequence_variance is not None:
        result["sequence_variance"] = sequence_variance
    # S2.8 margins/half-widths come from the frozen S2.6 matrix.  Preserve any
    # producer annotations without treating them as qualification evidence.
    for key in ("sizing_denominator", "bias_half_width_l2", "numeric_error"):
        if key in candidate:
            result[key] = candidate[key]
    return cell_id, result, {"candidate_hash": candidate_hash, "bundle_manifest_hash": bundle_hash, "candidate_ref": _ref(reference_ref, f"reference.{cell_id}.ref")}


def _flat_digest(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(array.astype(np.float64, copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _vector_digest_stream(value: Any) -> str:
    """Match ``stage2_formal._vector_digest`` without a second full copy."""

    if not isinstance(value, Mapping) or not value:
        raise S208ProductionBlocked("reference:VECTOR_MAPPING_REQUIRED")
    digest = hashlib.sha256()
    for name in sorted(value):
        array = np.asarray(value[name])
        digest.update(len(str(name).encode("utf-8")).to_bytes(8, "big"))
        digest.update(str(name).encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
        flat = array.reshape(-1)
        for start in range(0, flat.size, 1_000_000):
            chunk = np.asarray(flat[start : start + 1_000_000], dtype=np.float64)
            digest.update(np.ascontiguousarray(chunk).tobytes(order="C"))
    return digest.hexdigest()


def load_s208_reference_bundle(
    data_root: str | Path,
    reference_bundle_ref: str | Path,
    g23_gate_ref: str | Path,
    *,
    reference_root: str | Path | None = None,
    memmap_root: str | Path | None = None,
) -> dict[str, Any]:
    """Load six actual S2.4 candidates plus one independent G2.3 PASS.

    ``reference_bundle_ref`` and ``g23_gate_ref`` are paths, never mappings;
    this is the production boundary that prevents inline/fake references.
    """

    root = Path(data_root).resolve()
    if not isinstance(reference_bundle_ref, (str, Path)) or not isinstance(g23_gate_ref, (str, Path)):
        raise S208ProductionBlocked("S208_PRODUCTION_PATH_INPUTS_REQUIRED")
    bundle_path = _safe_path(root, str(reference_bundle_ref), "reference_bundle")
    bundle = _load_json(bundle_path, "reference_bundle")
    rows = _bundle_cells(bundle)
    gate_path = _ref(str(g23_gate_ref), "g23_gate")
    gate, gate_hash = _load_g23(root, gate_path)
    expected_cells = (
        "pythia-14m:initialization",
        "pythia-14m:early",
        "pythia-14m:mid_late",
        "pythia-31m-deduped:initialization",
        "pythia-31m-deduped:early",
        "pythia-31m-deduped:mid_late",
    )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        cell_id = row.get("cell_id", row.get("anchor_id"))
        if not isinstance(cell_id, str) or cell_id in by_id:
            raise S208ProductionBlocked("reference_bundle:CELL_IDENTITY_INVALID")
        by_id[cell_id] = row
    if set(by_id) != set(expected_cells):
        raise S208ProductionBlocked("reference_bundle:SIX_PRIMARY_CELL_SET_MISMATCH")
    cache = Path(memmap_root).resolve() if memmap_root is not None else bundle_path.parent / ".s208-memmap"
    cache.mkdir(parents=True, exist_ok=True)
    refs: dict[str, Mapping[str, Any]] = {}
    lineage: dict[str, Any] = {"schema_version": "stage2-s208-reference-lineage-v1", "reference_bundle_ref": str(reference_bundle_ref), "reference_bundle_hash": bundle.get("artifact_hash"), "g23_gate_ref": gate_path, "g23_gate_hash": gate_hash, "cells": {}}
    gate_cells = {str(item.get("cell_id")): item for item in gate.get("cells", []) if isinstance(item, Mapping)}
    candidate_hashes: set[str] = set()
    bundle_hashes: set[str] = set()
    for cell_id in expected_cells:
        row = by_id[cell_id]
        if gate_cells and (cell_id not in gate_cells or gate_cells[cell_id].get("status") != "PASS"):
            raise S208ProductionBlocked(f"g23_gate:{cell_id}:PASS_REQUIRED")
        loaded_id, reference, identity = _materialize_cell(root, row, gate_hash, cache, Path(reference_root).resolve() if reference_root else None)
        if loaded_id != cell_id:
            raise S208ProductionBlocked(f"reference:{cell_id}:CELL_IDENTITY_MISMATCH")
        refs[cell_id] = reference
        lineage["cells"][cell_id] = identity
        if identity["candidate_hash"] in candidate_hashes or identity["bundle_manifest_hash"] in bundle_hashes:
            raise S208ProductionBlocked("reference_bundle:DUPLICATE_CANDIDATE_OR_BUNDLE_IDENTITY")
        candidate_hashes.add(identity["candidate_hash"])
        bundle_hashes.add(identity["bundle_manifest_hash"])
    lineage["artifact_hash"] = canonical_json_hash({key: value for key, value in lineage.items() if key != "artifact_hash"})
    return {"schema_version": S208_MEMMAP_SCHEMA, "cells": refs, "lineage": lineage, "g23_gate": gate, "g23_gate_hash": gate_hash}


__all__ = ["S208ProductionBlocked", "S208_MEMMAP_SCHEMA", "S208_REFERENCE_BUNDLE_SCHEMA", "load_s208_reference_bundle"]
