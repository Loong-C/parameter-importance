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
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

import numpy as np

from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from ..runtime.task_artifacts import LoadedTaskArtifact, load_committed_task_artifact
from ..runtime.tensor_bundle import _strict_manifest, _verify_bundle_files
from .preregistration import BATCH_SIZES, PRIMARY_CELLS
from .stage2_g23_contracts import validate_weighting_contract


S208_REFERENCE_BUNDLE_SCHEMA = "stage2-s208-reference-bundle-v1"
S208_G23_SCHEMA = "stage2-g23-reference-evaluation-v1"
S208_MEMMAP_SCHEMA = "stage2-s208-memmap-reference-v1"
S208_MATRIX_MATERIALIZATION_SCHEMA = "stage2-s208-matrix-materialization-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/:-]{0,511}$")


class S208ProductionBlocked(RuntimeError):
    """Raised when formal S2.8 inputs are absent or not content-addressed."""


_S208_CELL_IDS = tuple(f"{item['model']}:{item['stage']}" for item in PRIMARY_CELLS)
_S208_ENDPOINTS = {
    "model_total": "model_total_signed_bias",
    "layer": "layer_total_l1_bias",
    "module": "module_total_l1_bias",
}
_S208_PILOT_ENDPOINTS = {
    "bias": "model_total_signed_bias",
    "nmse": "layer_total_l1_bias",
    "rank": "module_total_l1_bias",
}
_CORRECTED_DELTA_SCHEMA = "stage2-g23-corrected-delta-sci-v1"
_CORRECTED_DELTA_BATCH_SIZES = (32, 64, 128, 256)
_CORRECTED_DELTA_FIELDS = frozenset(
    {
        "schema_version",
        "source_producer_schema_version",
        "source_producer_ref",
        "source_producer_artifact_hash",
        "source_producer_table_mode",
        "source_producer_commit",
        "evaluator_commit",
        "evaluator_source_sha256",
        "formula_contract_hash",
        "formula_version",
        "formula",
        "absolute_floors",
        "reference_id",
        "sizing_result_hash",
        "sizing_plan_hash",
        "registry_hash",
        "candidate_sample_counts",
        "delta_sci_batch_sizes",
        "selected_sample_count_per_stream",
        "delta_sci_by_endpoint",
        "signal_scale_by_endpoint",
        "noise_scale_by_endpoint",
        "sizing_nodes",
        "correction_reason",
        "artifact_hash",
    }
)
_S206_CORRECTED_BINDING_FIELDS = frozenset(
    {
        "cell_id",
        "config_hash",
        "result_hash",
        "corrected_delta_sci_hash",
        "corrected_delta_sci_ref",
        "corrected_delta_sci_batch_sizes",
        "delta_sci_source",
    }
)
_S206_PILOT_BINDING_FIELDS = {
    "cell_id": "corrected_delta_sci_cell_id",
    "config_hash": "corrected_delta_sci_config_hash",
    "result_hash": "corrected_delta_sci_result_hash",
    "corrected_delta_sci_hash": "corrected_delta_sci_hash",
    "corrected_delta_sci_ref": "corrected_delta_sci_ref",
    "corrected_delta_sci_batch_sizes": "corrected_delta_sci_batch_sizes",
    "delta_sci_source": "delta_sci_source",
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _strict_cell_bindings(value: Mapping[str, Any], field: str) -> dict[str, Mapping[str, Any]]:
    """Validate the S2.6 seven-field ordered corrected-sidecar binding list."""

    raw = value.get("corrected_delta_sci_bindings")
    if not isinstance(raw, list) or len(raw) != len(_S208_CELL_IDS):
        raise S208ProductionBlocked(f"{field}:SIX_CELL_BINDINGS_REQUIRED")
    result: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping) or frozenset(item) != _S206_CORRECTED_BINDING_FIELDS:
            raise S208ProductionBlocked(f"{field}:BINDING_FIELDS_INVALID")
        raw_cell = item.get("cell_id")
        if not isinstance(raw_cell, str) or raw_cell not in _S208_CELL_IDS:
            raise S208ProductionBlocked(f"{field}:CELL_BINDING_SET_INVALID")
        cell_id = raw_cell
        if cell_id in result:
            raise S208ProductionBlocked(f"{field}:CELL_BINDING_SET_INVALID")
        _sha(item.get("config_hash"), f"{field}.{cell_id}.config_hash")
        _sha(item.get("result_hash"), f"{field}.{cell_id}.result_hash")
        _sha(item.get("corrected_delta_sci_hash"), f"{field}.{cell_id}.corrected_delta_sci_hash")
        _ref(item.get("corrected_delta_sci_ref"), f"{field}.{cell_id}.corrected_delta_sci_ref")
        if item.get("corrected_delta_sci_batch_sizes") != list(_CORRECTED_DELTA_BATCH_SIZES) or item.get("delta_sci_source") != "g23_output_derived_corrected_sidecar":
            raise S208ProductionBlocked(f"{field}.{cell_id}:BINDING_CONTRACT_MISMATCH")
        result[cell_id] = dict(item)
    if tuple(result) != _S208_CELL_IDS:
        raise S208ProductionBlocked(f"{field}:SIX_CELL_BINDING_ORDER_REQUIRED")
    return result


def _verify_cell_bindings_hash(value: Mapping[str, Any], field: str) -> str:
    """Verify the S2.6 wrapper hash over the exact ordered binding list."""

    bindings = value.get("corrected_delta_sci_bindings")
    if not isinstance(bindings, list):
        raise S208ProductionBlocked(f"{field}:SIX_CELL_BINDINGS_REQUIRED")
    declared = _sha(
        value.get("corrected_delta_sci_bindings_hash"),
        f"{field}.corrected_delta_sci_bindings_hash",
    )
    if declared != canonical_json_hash({"bindings": bindings}):
        raise S208ProductionBlocked(f"{field}:CORRECTED_DELTA_BINDINGS_HASH_MISMATCH")
    return declared


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
        "registry_hash": candidate.get("registry_hash"),
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


def _load_authoritative_object(
    root: Path,
    reference: str | Path,
    *,
    field: str,
    expected_task_id: str | None = None,
    expected_artifact_kind: str | None = None,
) -> tuple[Mapping[str, Any], str]:
    """Load a direct or committed formal source without name-based discovery."""

    path = _safe_path(root, str(reference), f"{field}.ref")
    try:
        loaded = load_committed_task_artifact(root, str(reference), require_formal=True)
    except (OSError, TypeError, ValueError):
        if expected_task_id is not None or expected_artifact_kind is not None:
            raise S208ProductionBlocked(f"{field}:TASK_ARTIFACT_REQUIRED")
        value = _load_json(path, field)
        if "artifact_hash" in value:
            digest = _verify_object_hash(value, field)
        else:
            hash_key = "manifest_hash" if "manifest_hash" in value else "index_hash" if "index_hash" in value else None
            if hash_key is None:
                raise S208ProductionBlocked(f"{field}:CONTENT_HASH_REQUIRED")
            digest = _sha(value.get(hash_key), f"{field}.{hash_key}")
            if canonical_json_hash({key: item for key, item in value.items() if key != hash_key}) != digest:
                raise S208ProductionBlocked(f"{field}:ARTIFACT_HASH_MISMATCH")
        return value, digest
    if expected_task_id is not None and loaded.identity.task_id != expected_task_id:
        raise S208ProductionBlocked(f"{field}:TASK_ARTIFACT_IDENTITY_INVALID")
    if expected_artifact_kind is not None and loaded.identity.artifact_kind != expected_artifact_kind:
        raise S208ProductionBlocked(f"{field}:TASK_ARTIFACT_IDENTITY_INVALID")
    value = dict(loaded.payload)
    declared = value.get("artifact_hash")
    if declared is not None:
        if _verify_object_hash(value, field) != loaded.identity.artifact_hash:
            raise S208ProductionBlocked(f"{field}:TASK_ARTIFACT_HASH_MISMATCH")
    return value, loaded.identity.artifact_hash


def _load_materialization_input(
    root: Path,
    value: Mapping[str, Any] | str | Path,
    *,
    field: str,
) -> tuple[Mapping[str, Any], str]:
    if isinstance(value, Mapping):
        payload = dict(value)
        return payload, _verify_object_hash(payload, field)
    return _load_authoritative_object(root, value, field=field)


def _load_preregistration_input(
    root: Path,
    value: Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any], str, str]:
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        payload, external_hash = _load_authoritative_object(
            root,
            value,
            field="preregistration",
            expected_task_id="stage2.01_scope_hypotheses_and_preregistration",
            expected_artifact_kind="preregistration",
        )
        declared = _sha(payload.get("preregistration_hash"), "preregistration.preregistration_hash")
        if canonical_json_hash({key: item for key, item in payload.items() if key != "preregistration_hash"}) != declared:
            raise S208ProductionBlocked("preregistration:ARTIFACT_HASH_MISMATCH")
        return payload, declared, external_hash
    declared = _sha(payload.get("preregistration_hash"), "preregistration.preregistration_hash")
    if canonical_json_hash({key: item for key, item in payload.items() if key != "preregistration_hash"}) != declared:
        raise S208ProductionBlocked("preregistration:ARTIFACT_HASH_MISMATCH")
    return payload, declared, declared


def _finite_positive(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0:
        raise S208ProductionBlocked(f"{field}:POSITIVE_FINITE_REQUIRED")
    return float(value)


def _registry_groups(
    registry: Mapping[str, Any],
    coordinate_ids: Sequence[str],
    *,
    cell_id: str,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    parameter_groups = registry.get("parameter_groups")
    if not isinstance(parameter_groups, Mapping) or not parameter_groups:
        raise S208ProductionBlocked(f"registry.{cell_id}:PARAMETER_GROUPS_REQUIRED")
    layer: dict[str, list[str]] = {}
    module: dict[str, list[str]] = {}
    for coordinate in coordinate_ids:
        parameter = coordinate.split("[", 1)[0]
        group = parameter_groups.get(coordinate, parameter_groups.get(parameter))
        if not isinstance(group, Mapping) or not isinstance(group.get("layer"), str) or not group.get("layer") or not isinstance(group.get("module"), str) or not group.get("module"):
            raise S208ProductionBlocked(f"registry.{cell_id}:COORDINATE_NOT_REGISTERED:{coordinate}")
        layer.setdefault(str(group["layer"]), []).append(coordinate)
        module.setdefault(str(group["module"]), []).append(coordinate)
    for scope, groups in (("layer", layer), ("module", module)):
        seen: set[str] = set()
        for name, coordinates in groups.items():
            if not name or not coordinates or seen.intersection(coordinates):
                raise S208ProductionBlocked(f"registry.{cell_id}:{scope}:OVERLAP_OR_EMPTY")
            seen.update(coordinates)
        if seen != set(coordinate_ids):
            raise S208ProductionBlocked(f"registry.{cell_id}:{scope}:NOT_EXHAUSTIVE")
    return layer, module


def _delta_v2(
    root: Path,
    value: Mapping[str, Any],
    *,
    cell_id: str,
    formula_hash: str,
    batch_size: int,
) -> tuple[dict[str, float], str | None]:
    if value.get("schema_version") != "stage2-reference-delta-sci-v2":
        raise S208ProductionBlocked(f"delta.{cell_id}:POST_SIZING_NUMERIC_REQUIRED")
    if value.get("source_kind") != "reference_sizing_bounded_online":
        raise S208ProductionBlocked(f"delta.{cell_id}:BOUNDED_SIZING_SOURCE_REQUIRED")
    if value.get("cell_id") not in (None, cell_id):
        raise S208ProductionBlocked(f"delta.{cell_id}:CELL_IDENTITY_MISMATCH")
    if value.get("formula_contract_hash") != formula_hash:
        raise S208ProductionBlocked(f"delta.{cell_id}:S21_FORMULA_BINDING_MISMATCH")
    if value.get("formula_version") not in (None, "stage2-reference-sizing-margin-v1"):
        raise S208ProductionBlocked(f"delta.{cell_id}:FORMULA_VERSION_MISMATCH")
    table = value.get("delta_sci_by_endpoint")
    if not isinstance(table, Mapping):
        raise S208ProductionBlocked(f"delta.{cell_id}:ENDPOINT_TABLE_REQUIRED")
    margins: dict[str, float] = {}
    for short, endpoint in _S208_ENDPOINTS.items():
        candidates = table.get(short)
        if not isinstance(candidates, Mapping):
            raise S208ProductionBlocked(f"delta.{cell_id}:{short}:ENDPOINT_REQUIRED")
        for candidate_batch in BATCH_SIZES:
            if str(candidate_batch) not in candidates:
                raise S208ProductionBlocked(f"delta.{cell_id}:B{candidate_batch}:CANDIDATE_REQUIRED")
            _finite_positive(candidates[str(candidate_batch)], f"delta.{cell_id}:{short}:B{candidate_batch}")
        margins[endpoint] = _finite_positive(candidates.get(str(batch_size)), f"delta.{cell_id}:{short}:B{batch_size}")
    denominator: str | None = None
    if "nmse_denominator" in value:
        _finite_positive(value["nmse_denominator"], f"delta.{cell_id}.nmse_denominator")
        denominator = "delta_ref"
    return margins, denominator


def _validate_historical_delta_source(
    root: Path,
    convergence: Mapping[str, Any],
    *,
    cell_id: str,
    formula_hash: str,
    sizing: Mapping[str, Any],
    sizing_hash: str,
    registry_hash: str,
    precision: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Validate the immutable pre-correction producer, without consuming it.

    G2.3 now publishes a corrected sidecar in the S2.6 batch-size domain.  The
    convergence report's candidate table remains useful provenance only.  It
    is checked against its content-addressed source, but never used to obtain
    the S2.8 margin.
    """

    candidate = convergence.get("candidate_delta_sci")
    if candidate is None:
        raise S208ProductionBlocked(f"convergence.{cell_id}:DELTA_SOURCE_BINDING_REQUIRED")
    if not isinstance(candidate, Mapping):
        raise S208ProductionBlocked(f"convergence.{cell_id}:DELTA_SOURCE_BINDING_REQUIRED")
    source_ref = candidate.get("source_ref")
    source_hash = candidate.get("source_hash")
    if not isinstance(source_ref, str) or not isinstance(source_hash, str):
        raise S208ProductionBlocked(f"convergence.{cell_id}:DELTA_SOURCE_BINDING_REQUIRED")
    source_ref = _ref(source_ref, f"delta_source.{cell_id}.ref")
    source_hash = _sha(source_hash, f"delta_source.{cell_id}.hash")
    if convergence.get("candidate_delta_sci_source") != source_ref or convergence.get("candidate_delta_sci_source_hash") != source_hash:
        raise S208ProductionBlocked(f"convergence.{cell_id}:DELTA_SOURCE_BINDING_REQUIRED")
    source, loaded_hash = _load_authoritative_object(root, source_ref, field=f"delta_source.{cell_id}")
    if loaded_hash != source_hash or source.get("artifact_hash") != source_hash:
        raise S208ProductionBlocked(f"delta_source.{cell_id}:HASH_MISMATCH")
    expected_source = {
        key: item
        for key, item in candidate.items()
        if key not in {"source_ref", "source_hash", "source_artifact_hash"}
    }
    if dict(source) != expected_source:
        raise S208ProductionBlocked(f"delta_source.{cell_id}:CONTENT_MISMATCH")
    if source.get("schema_version") != "stage2-reference-delta-sci-v2" or source.get("source_kind") not in {
        "reference_sizing_raw_shards",
        "reference_sizing_bounded_online",
    }:
        raise S208ProductionBlocked(f"delta_source.{cell_id}:SCHEMA_REQUIRED")
    if (
        source.get("cell_id") not in (None, cell_id)
        or source.get("formula_contract_hash") != formula_hash
        or source.get("formula_version") != "stage2-reference-sizing-margin-v1"
        or source.get("formula") != "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B"
        or source.get("reference_id") != sizing.get("reference_id")
        or source.get("sizing_plan_hash") != sizing_hash
        or source.get("registry_hash") != registry_hash
        or source.get("absolute_floors") != precision.get("absolute_floors")
    ):
        raise S208ProductionBlocked(f"delta_source.{cell_id}:IDENTITY_MISMATCH")
    counts = sizing.get("candidate_sample_counts")
    if not isinstance(counts, list) or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise S208ProductionBlocked(f"delta_source.{cell_id}:SIZING_COUNTS_REQUIRED")
    if source.get("candidate_sample_counts") != counts:
        raise S208ProductionBlocked(f"delta_source.{cell_id}:SIZING_COUNTS_MISMATCH")
    nodes = source.get("sizing_nodes")
    if not isinstance(nodes, list) or len(nodes) != len(counts):
        raise S208ProductionBlocked(f"delta_source.{cell_id}:SIZING_NODES_REQUIRED")
    node_counts = [item.get("sample_count") for item in nodes if isinstance(item, Mapping)]
    if len(node_counts) != len(nodes) or node_counts != counts:
        raise S208ProductionBlocked(f"delta_source.{cell_id}:SIZING_NODES_MISMATCH")
    table = source.get("delta_sci_by_endpoint")
    if not isinstance(table, Mapping):
        raise S208ProductionBlocked(f"delta_source.{cell_id}:ENDPOINT_TABLE_REQUIRED")
    key_sets: list[set[str]] = []
    for short in ("model_total", "layer", "module"):
        values = table.get(short)
        if not isinstance(values, Mapping):
            raise S208ProductionBlocked(f"delta_source.{cell_id}:{short}:ENDPOINT_REQUIRED")
        key_sets.append({str(key) for key in values})
        for key, value in values.items():
            _finite_positive(value, f"delta_source.{cell_id}:{short}:{key}")
    if not key_sets or any(keys != key_sets[0] for keys in key_sets[1:]) or key_sets[0] not in ({str(value) for value in counts}, set(map(str, _CORRECTED_DELTA_BATCH_SIZES))):
        raise S208ProductionBlocked(f"delta_source.{cell_id}:ENDPOINT_DOMAIN_INVALID")
    return source


def _validate_delta_plan(
    root: Path,
    delta_ref: str,
    *,
    cell_id: str,
    preregistration: Mapping[str, Any],
    sizing: Mapping[str, Any],
) -> tuple[Mapping[str, Any], str]:
    """Validate the pre-sizing S2.4 plan carried by the materialization index."""

    plan, plan_hash = _load_authoritative_object(root, delta_ref, field=f"delta_plan.{cell_id}")
    if plan.get("schema_version") != "stage2-reference-delta-sci-plan-v1" or plan.get("status") != "READY" or plan.get("scope") != "formal" or plan.get("phase") != "pre_sizing" or plan.get("cell_id") != cell_id:
        raise S208ProductionBlocked(f"delta_plan.{cell_id}:FORMAL_PRE_SIZING_REQUIRED")
    counts = sizing.get("candidate_sample_counts")
    if not isinstance(counts, list) or plan.get("candidate_sample_counts") != counts:
        raise S208ProductionBlocked(f"delta_plan.{cell_id}:CANDIDATE_COUNTS_MISMATCH")
    precision = preregistration.get("equivalence_and_precision")
    if not isinstance(precision, Mapping) or plan.get("formula_contract") != dict(precision) or plan.get("formula_contract_hash") != canonical_json_hash(dict(precision)):
        raise S208ProductionBlocked(f"delta_plan.{cell_id}:FORMULA_CONTRACT_MISMATCH")
    refs = plan.get("source_contract_refs")
    hashes = plan.get("source_contract_artifact_hashes")
    if not isinstance(refs, list) or not isinstance(hashes, list) or len(refs) != 2 or len(hashes) != 2:
        raise S208ProductionBlocked(f"delta_plan.{cell_id}:SOURCE_CONTRACT_BINDING_REQUIRED")
    for raw_ref, raw_hash, kind in zip(refs, hashes, ("preregistration", "hypothesis_contract"), strict=True):
        if not isinstance(raw_ref, str):
            raise S208ProductionBlocked(f"delta_plan.{cell_id}:SOURCE_CONTRACT_REF_INVALID")
        _, loaded_hash = _load_authoritative_object(
            root,
            raw_ref,
            field=f"delta_plan.{cell_id}.source_contract",
            expected_task_id="stage2.01_scope_hypotheses_and_preregistration",
            expected_artifact_kind=kind,
        )
        if loaded_hash != _sha(raw_hash, f"delta_plan.{cell_id}.source_contract_hash"):
            raise S208ProductionBlocked(f"delta_plan.{cell_id}:SOURCE_CONTRACT_HASH_MISMATCH")
    return plan, plan_hash


def _load_corrected_delta_sidecar(
    root: Path,
    *,
    ref: object,
    declared_hash: object,
    g23_cell: Mapping[str, Any],
    rebind_row: Mapping[str, Any],
    cell_id: str,
    formula_hash: str,
    precision: Mapping[str, Any],
    sizing: Mapping[str, Any],
    sizing_hash: str,
    registry_hash: str,
    convergence: Mapping[str, Any],
    historical_source: Mapping[str, Any] | None,
    gate_calculator: Mapping[str, Any],
    expected_output_root_ref: str | None,
) -> tuple[Mapping[str, Any], str]:
    """Load and fully bind one evaluator-owned corrected G2.3 sidecar."""

    if not isinstance(ref, str) or not isinstance(declared_hash, str):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_REF_HASH_REQUIRED")
    sidecar_ref = _ref(ref, f"g23.{cell_id}.corrected_delta_sci_ref")
    sidecar_hash = _sha(declared_hash, f"g23.{cell_id}.corrected_delta_sci_hash")
    parts = PurePosixPath(sidecar_ref).parts
    if len(parts) < 2 or parts[-2] != "g2.3-corrected-delta-sci" or parts[-1] != f"{sidecar_hash}.json":
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_REF_SUFFIX_INVALID")
    if expected_output_root_ref is None:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_OUTPUT_ROOT_BINDING_REQUIRED")
    output_parts = PurePosixPath(expected_output_root_ref).parts
    if PurePosixPath(*parts[:-2]) != PurePosixPath(*output_parts):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_OUTPUT_ROOT_MISMATCH")
    sidecar, loaded_hash = _load_authoritative_object(root, sidecar_ref, field=f"corrected_delta_sci.{cell_id}")
    if loaded_hash != sidecar_hash or sidecar.get("artifact_hash") != sidecar_hash:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_HASH_MISMATCH")
    if frozenset(sidecar) != _CORRECTED_DELTA_FIELDS:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_FIELD_SET_INVALID")
    if sidecar.get("schema_version") != _CORRECTED_DELTA_SCHEMA:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SCHEMA_REQUIRED")
    source_ref = _ref(sidecar.get("source_producer_ref"), f"corrected_delta_sci.{cell_id}.source_ref")
    source, source_hash = _load_authoritative_object(root, source_ref, field=f"corrected_delta_sci.{cell_id}.source_producer")
    if source_hash != sidecar.get("source_producer_artifact_hash") or source.get("artifact_hash") != source_hash:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_HASH_MISMATCH")
    if historical_source is not None and dict(source) != dict(historical_source):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_HISTORY_MISMATCH")
    if source.get("schema_version") != "stage2-reference-delta-sci-v2":
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_SCHEMA_REQUIRED")
    mode = sidecar.get("source_producer_table_mode")
    if mode not in {"sizing_nodes_legacy", "candidate_batch_sizes"}:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_MODE_INVALID")
    for name in ("source_producer_artifact_hash", "evaluator_source_sha256"):
        _sha(sidecar.get(name), f"g23.{cell_id}.{name}")
    for name in ("source_producer_commit", "evaluator_commit"):
        if not isinstance(sidecar.get(name), str) or _COMMIT.fullmatch(sidecar[name]) is None:
            raise S208ProductionBlocked(f"g23.{cell_id}.{name}:COMMIT_REQUIRED")
    if (
        sidecar.get("source_producer_schema_version") != source.get("schema_version")
        or sidecar.get("formula_contract_hash") != formula_hash
        or sidecar.get("formula_version") != "stage2-reference-sizing-margin-v1"
        or sidecar.get("formula") != "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B"
        or sidecar.get("absolute_floors") != precision.get("absolute_floors")
        or sidecar.get("reference_id") != sizing.get("reference_id")
        or sidecar.get("sizing_plan_hash") != sizing_hash
        or sidecar.get("registry_hash") != registry_hash
    ):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_IDENTITY_MISMATCH")
    counts = sizing.get("candidate_sample_counts")
    if not isinstance(counts, list) or sidecar.get("candidate_sample_counts") != counts:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_COUNTS_MISMATCH")
    if sidecar.get("delta_sci_batch_sizes") != list(_CORRECTED_DELTA_BATCH_SIZES):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_BATCH_DOMAIN_INVALID")
    selected = sidecar.get("selected_sample_count_per_stream")
    if selected != convergence.get("selected_sample_count_per_stream"):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SELECTED_COUNT_MISMATCH")
    sizing_result_hash = convergence.get("sizing_result_hash")
    if sizing_result_hash is None and isinstance(convergence.get("one_shot_result"), Mapping):
        sizing_result_hash = convergence["one_shot_result"].get("sizing_result_hash")
    if sidecar.get("sizing_result_hash") != sizing_result_hash:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SIZING_RESULT_MISMATCH")
    nodes = sidecar.get("sizing_nodes")
    if not isinstance(nodes, list) or len(nodes) != len(counts) or [item.get("sample_count") for item in nodes if isinstance(item, Mapping)] != counts:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_NODES_INVALID")
    source_nodes = source.get("sizing_nodes")
    if source_nodes != nodes:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_NODES_MISMATCH")
    for endpoint in ("delta_sci_by_endpoint", "signal_scale_by_endpoint", "noise_scale_by_endpoint"):
        table = sidecar.get(endpoint)
        if not isinstance(table, Mapping) or set(table) != {"model_total", "layer", "module"}:
            raise S208ProductionBlocked(f"g23.{cell_id}:{endpoint}:ENDPOINT_TABLE_INVALID")
        for short in ("model_total", "layer", "module"):
            values = table.get(short)
            if not isinstance(values, Mapping) or set(values) != {str(value) for value in _CORRECTED_DELTA_BATCH_SIZES}:
                raise S208ProductionBlocked(f"g23.{cell_id}:{endpoint}.{short}:B_DOMAIN_INVALID")
            for batch, value in values.items():
                _finite_positive(value, f"g23.{cell_id}:{endpoint}.{short}.{batch}")
    if historical_source is not None and mode == "candidate_batch_sizes" and sidecar.get("delta_sci_by_endpoint") != historical_source.get("delta_sci_by_endpoint"):
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_RECOMPUTE_MISMATCH")
    if historical_source is not None and mode == "candidate_batch_sizes":
        for field in ("signal_scale_by_endpoint", "noise_scale_by_endpoint"):
            if sidecar.get(field) != historical_source.get(field):
                raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_SOURCE_SCALE_MISMATCH")
    reason = sidecar.get("correction_reason")
    expected_reason = (
        "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes"
        if mode == "sizing_nodes_legacy"
        else "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator"
    )
    if reason != expected_reason:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_REASON_MISMATCH")
    signal_table = sidecar.get("signal_scale_by_endpoint")
    noise_table = sidecar.get("noise_scale_by_endpoint")
    delta_table = sidecar.get("delta_sci_by_endpoint")
    assert isinstance(signal_table, Mapping) and isinstance(noise_table, Mapping) and isinstance(delta_table, Mapping)
    for short in ("model_total", "layer", "module"):
        signals = signal_table[short]
        noises = noise_table[short]
        deltas = delta_table[short]
        assert isinstance(signals, Mapping) and isinstance(noises, Mapping) and isinstance(deltas, Mapping)
        signal_values = [_finite_positive(signals[str(batch)], f"g23.{cell_id}.signal.{short}.{batch}") for batch in _CORRECTED_DELTA_BATCH_SIZES]
        noise_values = [_finite_positive(noises[str(batch)], f"g23.{cell_id}.noise.{short}.{batch}") for batch in _CORRECTED_DELTA_BATCH_SIZES]
        if any(value != signal_values[0] for value in signal_values) or any(not math.isclose(noise_values[index] * _CORRECTED_DELTA_BATCH_SIZES[index], noise_values[0] * _CORRECTED_DELTA_BATCH_SIZES[0], rel_tol=1e-12, abs_tol=1e-15) for index in range(len(noise_values))):
            raise S208ProductionBlocked(f"g23.{cell_id}:{short}:CORRECTED_DELTA_SCALE_DRIFT")
        for batch, signal, noise in zip(_CORRECTED_DELTA_BATCH_SIZES, signal_values, noise_values, strict=True):
            expected = max(0.10 * noise, 0.01 * signal)
            observed = _finite_positive(deltas[str(batch)], f"g23.{cell_id}.delta.{short}.{batch}")
            if observed != expected:
                raise S208ProductionBlocked(f"g23.{cell_id}:{short}:CORRECTED_DELTA_FORMULA_MISMATCH")
    identities = g23_cell.get("identities")
    metrics = g23_cell.get("metrics")
    if not isinstance(identities, Mapping) or not isinstance(metrics, Mapping):
        raise S208ProductionBlocked(f"g23.{cell_id}:CELL_IDENTITY_FIELDS_REQUIRED")
    result_hash = _sha(identities.get("result_hash"), f"g23.{cell_id}.result_hash")
    config_hash = _sha(identities.get("config_hash"), f"g23.{cell_id}.config_hash")
    producer_commit = identities.get("producer_commit")
    if not isinstance(producer_commit, str) or _COMMIT.fullmatch(producer_commit) is None or sidecar.get("source_producer_commit") != producer_commit:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_PRODUCER_COMMIT_MISMATCH")
    evaluator_commit = gate_calculator.get("evaluator_commit")
    evaluator_source_sha256 = gate_calculator.get("source_sha256")
    if sidecar.get("evaluator_commit") != evaluator_commit or sidecar.get("evaluator_source_sha256") != evaluator_source_sha256:
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_EVALUATOR_IDENTITY_MISMATCH")
    if rebind_row.get("result_hash") != result_hash or rebind_row.get("config_hash") != config_hash:
        raise S208ProductionBlocked(f"g23.{cell_id}:REBIND_CELL_IDENTITY_MISMATCH")
    if identities.get("corrected_delta_sci_ref") != sidecar_ref or identities.get("corrected_delta_sci_hash") != sidecar_hash or metrics.get("corrected_delta_sci_ref") != sidecar_ref or metrics.get("corrected_delta_sci_hash") != sidecar_hash or metrics.get("corrected_delta_sci_batch_sizes") != list(_CORRECTED_DELTA_BATCH_SIZES) or metrics.get("delta_sci_source") != "g23_output_derived_corrected_sidecar":
        raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_CELL_BINDING_MISMATCH")
    return sidecar, sidecar_hash


def _bounded_sizing_denominator(
    root: Path,
    *,
    reference: Mapping[str, Any],
    convergence: Mapping[str, Any],
    delta: Mapping[str, Any],
    cell_id: str,
    tau_coord: float,
) -> float:
    """Rebuild D_c from the selected hash-bound bounded sizing checkpoint."""

    candidate_ref = reference.get("candidate_ref")
    plan = convergence.get("sizing_plan")
    one_shot = convergence.get("one_shot_result")
    if not isinstance(candidate_ref, str) or not isinstance(plan, Mapping) or not isinstance(one_shot, Mapping):
        raise S208ProductionBlocked(f"reference.{cell_id}:BOUNDED_SIZING_LINEAGE_REQUIRED")
    block_size = plan.get("block_size")
    selected = convergence.get("selected_sample_count_per_stream")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0 or isinstance(selected, bool) or not isinstance(selected, int) or selected <= 0:
        raise S208ProductionBlocked(f"sizing.{cell_id}:SELECTED_CHECKPOINT_IDENTITY_REQUIRED")
    registry_identity = convergence.get("registry_identity")
    registry_hash = registry_identity.get("registry_hash") if isinstance(registry_identity, Mapping) else delta.get("registry_hash")
    plan_hash = convergence.get("sizing_plan_artifact_hash", plan.get("artifact_hash"))
    identities = {
        "plan_hash": plan_hash,
        "provider_state_digest": one_shot.get("provider_state_digest"),
        "registry_hash": registry_hash,
        "weighting_assumptions": one_shot.get("weighting_assumptions"),
        "sizing_draw_hash": convergence.get("sizing_draw_hash"),
        "sizing_identity_hash": convergence.get("sizing_identity_hash"),
        "sizing_stream": True,
    }
    identity_strings = tuple(key for key in identities if key not in {"weighting_assumptions", "sizing_stream"})
    if any(not isinstance(identities[key], str) for key in identity_strings) or identities["sizing_stream"] is not True or not isinstance(identities["weighting_assumptions"], Mapping):
        raise S208ProductionBlocked(f"sizing.{cell_id}:BOUNDED_IDENTITY_FIELDS_REQUIRED")
    try:
        validate_weighting_contract(
            identities["weighting_assumptions"],
            field=f"sizing.{cell_id}.weighting_assumptions",
        )
    except (TypeError, ValueError) as error:
        raise S208ProductionBlocked(f"sizing.{cell_id}:WEIGHTING_CONTRACT_INVALID") from error
    try:
        from .stage2_g23_evaluator import _bounded_moments_strict, _load_resume_commits, _resume_roots

        sizing_root, _ = _resume_roots(root, candidate_ref)
        if not (sizing_root / "bounded-checkpoint").is_dir():
            raise S208ProductionBlocked(f"sizing.{cell_id}:BOUNDED_CHECKPOINT_REQUIRED")
        states = _load_resume_commits(root, sizing_root, "stage2-reference-progress-state-v1", identities=identities)
        state = next((item for item in states if int(item.get("processed_block_pairs", 0)) * block_size == selected), None)
        if state is None:
            raise S208ProductionBlocked(f"sizing.{cell_id}:SELECTED_CHECKPOINT_MISSING")
        moments = _bounded_moments_strict(state.get("a"), f"sizing.{cell_id}.bounded.a", require_higher=False)
        mean = moments.mean()
    except S208ProductionBlocked:
        raise
    except Exception as error:
        raise S208ProductionBlocked(f"sizing.{cell_id}:BOUNDED_CHECKPOINT_INVALID") from error
    # The frozen S2.8 contract is D_c=max(sum_k a_k^2, P*tau_coord^2),
    # with a=mu_sizing^2.  This is deliberately not computed from final
    # reference vectors or from a caller-provided scalar.
    signal = float(sum(np.sum(np.square(np.square(value))) for value in mean.values()))
    parameter_count = int(sum(value.size for value in mean.values()))
    denominator = max(signal, float(parameter_count) * float(tau_coord) ** 2)
    if not np.isfinite(denominator) or denominator <= 0:
        raise S208ProductionBlocked(f"sizing.{cell_id}:NMSE_DENOMINATOR_INVALID")
    nodes = delta.get("sizing_nodes")
    if not isinstance(nodes, list):
        raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_NODES_REQUIRED")
    counts = delta.get("candidate_sample_counts", plan.get("candidate_sample_counts"))
    if not isinstance(counts, list) or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts):
        raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_COUNTS_REQUIRED")
    if len(nodes) != len(counts) or [item.get("sample_count") for item in nodes if isinstance(item, Mapping)] != counts:
        raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_NODE_ORDER_INVALID")
    from .stage2_formal import _BoundedCheckpointStore, _bounded_moments_digest

    for node in nodes:
        if not isinstance(node, Mapping):
            raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_NODE_INVALID")
        node_count = node.get("sample_count")
        state_for_node = next((item for item in states if int(item.get("processed_block_pairs", 0)) * block_size == node_count), None)
        if state_for_node is None:
            raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_NODE_CHECKPOINT_MISSING")
        moments_for_node = _bounded_moments_strict(state_for_node.get("a"), f"sizing.{cell_id}.bounded.a.{node_count}", require_higher=False)
        expected_digest = canonical_json_hash({"checkpoint_schema": _BoundedCheckpointStore.schema_version, "plan_hash": plan_hash, "sample_count": node_count, "moments_hash": _bounded_moments_digest(moments_for_node)})
        if node.get("state_digest") != expected_digest:
            raise S208ProductionBlocked(f"sizing.{cell_id}:SIZING_NODE_HASH_MISMATCH")
    return denominator


def materialize_s208_matrix(
    data_root: str | Path,
    materialization_index: str | Path,
    *,
    matrix: Mapping[str, Any] | str | Path,
    preregistration: Mapping[str, Any] | str | Path,
    g23_gate: Mapping[str, Any] | str | Path,
    g24a_gate: Mapping[str, Any] | str | Path,
    g24b_gate: Mapping[str, Any] | str | Path | None = None,
    pilot_report: Mapping[str, Any] | str | Path | None = None,
    references: Mapping[str, Any] | None = None,
    reference_convergence_refs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Materialize S2.8 science inputs from the real upstream artifacts.

    S2.6 remains the authority for B/M/R and cost, while S2.4 sizing/delta,
    its registry, and the bound G2.4a metrics supply the fields S2.6
    intentionally does not contain.  No margin, denominator, grouping, or
    precision value is derived from a vector or guessed from a cell name.
    """

    root = Path(data_root).resolve()
    if isinstance(materialization_index, Mapping):
        index = dict(materialization_index)
    else:
        index = _load_json(_safe_path(root, str(materialization_index), "s204_materialization_index"), "s204_materialization_index")
    if index.get("index_hash") is None or canonical_json_hash({key: item for key, item in index.items() if key != "index_hash"}) != _sha(index.get("index_hash"), "s204_materialization_index.index_hash"):
        raise S208ProductionBlocked("s204_materialization_index:INDEX_HASH_MISMATCH")
    index_hash = str(index["index_hash"])
    if index.get("schema_version") != "stage2-s204-six-cell-materialization-index-v1" or index.get("scope") != "formal":
        raise S208ProductionBlocked("s204_materialization_index:FORMAL_SCHEMA_REQUIRED")
    if "index_hash" in index:
        declared = _sha(index.get("index_hash"), "s204_materialization_index.index_hash")
        if canonical_json_hash({key: item for key, item in index.items() if key != "index_hash"}) != declared:
            raise S208ProductionBlocked("s204_materialization_index:INDEX_HASH_MISMATCH")
        index_hash = declared
    rows = index.get("cells")
    if not isinstance(rows, list) or tuple(item.get("cell_id") for item in rows if isinstance(item, Mapping)) != _S208_CELL_IDS:
        raise S208ProductionBlocked("s204_materialization_index:SIX_CELL_ORDER_REQUIRED")
    if any(not isinstance(item, Mapping) for item in rows):
        raise S208ProductionBlocked("s204_materialization_index:ROW_INVALID")
    matrix_payload, matrix_hash = _load_materialization_input(root, matrix, field="matrix")
    prereg_payload, prereg_hash, prereg_artifact_hash = _load_preregistration_input(root, preregistration)
    if (
        matrix_payload.get("schema_version") != "stage2-formal-pilot-matrix-freeze-v1"
        or matrix_payload.get("scope") != "formal"
        or matrix_payload.get("status") != "FORMAL_FROZEN"
        or matrix_payload.get("formal_eligible") is not True
    ):
        raise S208ProductionBlocked("matrix:FORMAL_FROZEN_REQUIRED")
    if (
        prereg_payload.get("schema_version") != "stage2-preregistration-v1"
        or prereg_payload.get("scope") != "formal"
        or prereg_payload.get("state") != "FROZEN"
    ):
        raise S208ProductionBlocked("preregistration:FORMAL_FROZEN_REQUIRED")
    precision = prereg_payload.get("equivalence_and_precision")
    if not isinstance(precision, Mapping):
        raise S208ProductionBlocked("preregistration:EQUIVALENCE_PRECISION_REQUIRED")
    formula_hash = prereg_artifact_hash
    floors = precision.get("absolute_floors")
    tau_coord = floors.get("tau_coord") if isinstance(floors, Mapping) else None
    if isinstance(tau_coord, bool) or not isinstance(tau_coord, (int, float)) or not np.isfinite(float(tau_coord)) or float(tau_coord) <= 0:
        raise S208ProductionBlocked("preregistration:TAU_COORD_REQUIRED")
    g23_payload, g23_hash = _load_materialization_input(root, g23_gate, field="g23_gate")
    g24a_payload, g24a_hash = _load_materialization_input(root, g24a_gate, field="g24a_gate")
    g23_gate_ref: str | None = None
    g23_output_root_ref: str | None = None
    if not isinstance(g23_gate, Mapping):
        gate_ref = _ref(str(g23_gate), "g23_gate.ref")
        g23_gate_ref = gate_ref
        gate_parts = PurePosixPath(gate_ref).parts
        if "g2.3-attempts" in gate_parts:
            g23_output_root_ref = PurePosixPath(*gate_parts[: gate_parts.index("g2.3-attempts")]).as_posix()
        elif len(gate_parts) > 1:
            g23_output_root_ref = PurePosixPath(*gate_parts[:-1]).as_posix()
    g23_calculator = g23_payload.get("calculator")
    if not isinstance(g23_calculator, Mapping):
        raise S208ProductionBlocked("g23_gate:CALCULATOR_IDENTITY_REQUIRED")
    g24b_payload: Mapping[str, Any] | None = None
    g24b_hash: str | None = None
    if g24b_gate is not None:
        g24b_payload, g24b_hash = _load_materialization_input(root, g24b_gate, field="g24b_gate")
        if g24b_payload.get("status") not in {"PASS", "G2.4B_PASS_MATRIX_FROZEN"}:
            raise S208ProductionBlocked("g24b_gate:PASS_REQUIRED")
    if g23_payload.get("gate_id") not in (None, "stage2.G2.3") or g23_payload.get("status") != "PASS" or g23_payload.get("formal_eligible") is not True:
        raise S208ProductionBlocked("g23_gate:INDEPENDENT_PASS_REQUIRED")
    g23_rows = g23_payload.get("cells")
    if (
        not isinstance(g23_rows, list)
        or tuple(item.get("cell_id") for item in g23_rows if isinstance(item, Mapping)) != _S208_CELL_IDS
        or g23_payload.get("required_cell_count") != len(_S208_CELL_IDS)
        or g23_payload.get("complete_cell_count") != len(_S208_CELL_IDS)
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "PASS"
            or item.get("formal_eligible") is not True
            for item in g23_rows
        )
    ):
        raise S208ProductionBlocked("g23_gate:SIX_CELL_PASS_REQUIRED")
    g23_by_cell = {str(item["cell_id"]): item for item in g23_rows if isinstance(item, Mapping)}
    if g24a_payload.get("schema_version") != "stage2-g24a-formal-evaluation-v1" or g24a_payload.get("gate_id") != "stage2.G2.4a" or g24a_payload.get("status") != "PASS" or g24a_payload.get("formal_eligible") is not True:
        raise S208ProductionBlocked("g24a_gate:FORMAL_PASS_REQUIRED")
    if (
        g23_gate_ref is None
        or g24a_payload.get("g23_evaluation_ref") != g23_gate_ref
        or g24a_payload.get("g23_evaluation_hash") != g23_hash
    ):
        raise S208ProductionBlocked("g24a_gate:G23_BINDING_MISMATCH")
    g24a_rows = g24a_payload.get("results")
    if not isinstance(g24a_rows, list) or tuple(item.get("cell_id") for item in g24a_rows if isinstance(item, Mapping)) != _S208_CELL_IDS:
        raise S208ProductionBlocked("g24a_gate:SIX_CELL_ORDER_REQUIRED")
    g24a_by_cell = {str(item["cell_id"]): item for item in g24a_rows if isinstance(item, Mapping)}
    pilot_hash = matrix_payload.get("pilot_report_hash")
    if not isinstance(pilot_hash, str) or _SHA256.fullmatch(pilot_hash) is None:
        raise S208ProductionBlocked("matrix:PILOT_REPORT_HASH_REQUIRED")
    pilot_payload: Mapping[str, Any] | None = None
    pilot_ref: str | None = None
    if pilot_report is not None:
        pilot_payload, loaded_pilot_hash = _load_materialization_input(root, pilot_report, field="s206_pilot_report")
        if loaded_pilot_hash != pilot_hash:
            raise S208ProductionBlocked("s206_pilot_report:MATRIX_HASH_MISMATCH")
        pilot_ref = str(pilot_report) if not isinstance(pilot_report, Mapping) else None
    elif isinstance(g24b_payload, Mapping):
        measured = g24b_payload.get("measured")
        if not isinstance(measured, Mapping) or measured.get("pilot_report_hash") != pilot_hash:
            raise S208ProductionBlocked("g24b_gate:PILOT_REPORT_BINDING_MISMATCH")
        evidence_refs = g24b_payload.get("evidence_refs")
        if not isinstance(evidence_refs, list):
            raise S208ProductionBlocked("g24b_gate:PILOT_REPORT_REF_REQUIRED")
        matches: list[tuple[str, Mapping[str, Any], str]] = []
        for raw_ref in evidence_refs:
            if not isinstance(raw_ref, str):
                continue
            candidate_ref = raw_ref.split("::", 1)[0]
            try:
                candidate, candidate_hash = _load_authoritative_object(root, candidate_ref, field="s206_pilot_report")
            except S208ProductionBlocked:
                continue
            if candidate.get("schema_version") == "stage2-formal-blinded-pilot-report-v1" and candidate_hash == pilot_hash:
                matches.append((candidate_ref, candidate, candidate_hash))
        if len(matches) != 1:
            raise S208ProductionBlocked("g24b_gate:PILOT_REPORT_EVIDENCE_NOT_UNIQUE")
        pilot_ref, pilot_payload, _ = matches[0]
    else:
        raise S208ProductionBlocked("s206_pilot_report:EXPLICIT_HASH_BOUND_SOURCE_REQUIRED")
    if not isinstance(pilot_payload, Mapping) or pilot_payload.get("schema_version") != "stage2-formal-blinded-pilot-report-v1" or pilot_payload.get("formal_eligible") is not False:
        raise S208ProductionBlocked("s206_pilot_report:BLINDED_REPORT_REQUIRED")
    pilot_cell_bindings = _strict_cell_bindings(pilot_payload, "s206_pilot_report")
    matrix_cell_bindings = _strict_cell_bindings(matrix_payload, "matrix")
    if not isinstance(g24b_payload, Mapping):
        raise S208ProductionBlocked("g24b_gate:SIX_CELL_BINDINGS_REQUIRED")
    g24b_measured = g24b_payload.get("measured")
    if not isinstance(g24b_measured, Mapping):
        raise S208ProductionBlocked("g24b_gate:MEASURED_BINDINGS_REQUIRED")
    g24b_cell_bindings = _strict_cell_bindings(g24b_measured, "g24b_gate.measured")
    _verify_cell_bindings_hash(pilot_payload, "s206_pilot_report")
    _verify_cell_bindings_hash(matrix_payload, "matrix")
    _verify_cell_bindings_hash(g24b_measured, "g24b_gate.measured")
    pilot_measurements = pilot_payload.get("measurements")
    if not isinstance(pilot_measurements, list) or len(pilot_measurements) != 6 * 4 * 5:
        raise S208ProductionBlocked("s206_pilot_report:MEASUREMENT_GRID_REQUIRED")
    pilot_by_cell: dict[str, Mapping[str, Any]] = {}
    for measurement in pilot_measurements:
        if not isinstance(measurement, Mapping):
            raise S208ProductionBlocked("s206_pilot_report:MEASUREMENT_ROW_INVALID")
        if not all(source in measurement for source in _S206_PILOT_BINDING_FIELDS.values()):
            raise S208ProductionBlocked("s206_pilot_report:MEASUREMENT_BINDING_FIELDS_REQUIRED")
        anchor = measurement.get("anchor_id")
        cell = str(anchor).replace(".", ":") if isinstance(anchor, str) else ""
        if cell not in _S208_CELL_IDS:
            raise S208ProductionBlocked("s206_pilot_report:ANCHOR_SET_INVALID")
        key = f"{cell}:{measurement.get('batch_size')}:{measurement.get('microbatch_count')}"
        if key in pilot_by_cell:
            raise S208ProductionBlocked("s206_pilot_report:MEASUREMENT_DUPLICATE")
        pilot_by_cell[key] = measurement
    rebind_cells: dict[str, Mapping[str, Any]] = {}
    rebind_ref = g24a_payload.get("rebind_plan_ref")
    if not isinstance(rebind_ref, str) or not rebind_ref:
        raise S208ProductionBlocked("g24a_gate:REBIND_PLAN_REF_REQUIRED")
    rebind, rebind_hash = _load_authoritative_object(root, rebind_ref, field="s205_rebind_plan")
    if rebind.get("schema_version") != "stage2-s205-rebind-plan-v1" or rebind.get("status") != "READY" or rebind.get("formal_eligible") is not True or rebind_hash != g24a_payload.get("rebind_plan_hash"):
        raise S208ProductionBlocked("s205_rebind_plan:FORMAL_HASH_BINDING_REQUIRED")
    raw_rebind_cells = rebind.get("cells")
    if not isinstance(raw_rebind_cells, list) or tuple(item.get("cell_id") for item in raw_rebind_cells if isinstance(item, Mapping)) != _S208_CELL_IDS:
        raise S208ProductionBlocked("s205_rebind_plan:SIX_CELL_ORDER_REQUIRED")
    rebind_cells = {str(item["cell_id"]): item for item in raw_rebind_cells if isinstance(item, Mapping)}
    manifest_ref = index.get("six_cell_manifest_ref")
    if not isinstance(manifest_ref, str):
        raise S208ProductionBlocked("s204_materialization_index:SIX_CELL_MANIFEST_REQUIRED")
    manifest, manifest_hash = _load_authoritative_object(root, manifest_ref, field="s204_six_cell_manifest")
    if manifest.get("schema_version") != "stage2-s204-six-cell-manifest-v1" or manifest.get("scope") != "formal" or manifest.get("status") != "READY":
        raise S208ProductionBlocked("s204_six_cell_manifest:FORMAL_SCHEMA_REQUIRED")
    checkpoints = manifest.get("checkpoints")
    if not isinstance(checkpoints, list) or tuple(item.get("cell_id") for item in checkpoints if isinstance(item, Mapping)) != _S208_CELL_IDS:
        raise S208ProductionBlocked("s204_six_cell_manifest:SIX_CELL_ORDER_REQUIRED")
    checkpoint_by_cell = {str(item["cell_id"]): item for item in checkpoints if isinstance(item, Mapping)}
    source_refs = references.get("cells", references) if isinstance(references, Mapping) else None
    if not isinstance(source_refs, Mapping):
        raise S208ProductionBlocked("references:REQUIRED_FOR_REGISTRY_BINDING")
    cells: dict[str, Any] = {}
    for row in rows:
        assert isinstance(row, Mapping)
        cell_id = str(row["cell_id"])
        checkpoint = checkpoint_by_cell[cell_id]
        registry_ref = row.get("registry_ref", row.get("parameter_registry_ref"))
        sizing_ref = row.get("sizing_ref")
        delta_ref = row.get("delta_ref")
        if not all(isinstance(item, str) and item for item in (registry_ref, sizing_ref, delta_ref)):
            raise S208ProductionBlocked(f"materialization.{cell_id}:SIZING_REGISTRY_DELTA_REFS_REQUIRED")
        registry, registry_hash = _load_authoritative_object(root, registry_ref, field=f"registry.{cell_id}")
        if registry.get("schema_version") != "stage2-parameter-registry-artifact-v1" or registry.get("status") != "READY" or registry.get("scope", "formal") != "formal":
            raise S208ProductionBlocked(f"registry.{cell_id}:FORMAL_READY_REQUIRED")
        for key in ("cell_id", "checkpoint_id", "model_id", "training_stage"):
            expected = cell_id if key == "cell_id" else checkpoint.get(key)
            if registry.get(key) != expected:
                raise S208ProductionBlocked(f"registry.{cell_id}:{key}:IDENTITY_MISMATCH")
        if registry.get("registry_hash") != checkpoint.get("registry_hash"):
            raise S208ProductionBlocked(f"registry.{cell_id}:CHECKPOINT_HASH_MISMATCH")
        reference = source_refs.get(cell_id)
        if not isinstance(reference, Mapping):
            raise S208ProductionBlocked(f"reference.{cell_id}:INPUT_REQUIRED")
        declared_registry_hash = reference.get("registry_hash")
        if declared_registry_hash is not None and declared_registry_hash != registry.get("registry_hash"):
            raise S208ProductionBlocked(f"reference.{cell_id}:REGISTRY_HASH_MISMATCH")
        coordinate_ids = reference.get("coordinate_ids")
        if not isinstance(coordinate_ids, list) or not coordinate_ids or not all(isinstance(item, str) for item in coordinate_ids):
            raise S208ProductionBlocked(f"reference.{cell_id}:COORDINATE_REGISTRY_REQUIRED")
        layer_groups, module_groups = _registry_groups(registry, coordinate_ids, cell_id=cell_id)
        sizing, sizing_hash = _load_authoritative_object(root, sizing_ref, field=f"sizing.{cell_id}")
        if sizing.get("schema_version") != "stage2-reference-sizing-plan-v1" or sizing.get("reference_id") is None:
            raise S208ProductionBlocked(f"sizing.{cell_id}:FORMAL_PLAN_REQUIRED")
        matrix_cells = matrix_payload.get("cells", matrix_payload.get("anchors"))
        if isinstance(matrix_cells, Mapping):
            matrix_cell = matrix_cells.get(cell_id, matrix_cells.get(cell_id.replace(":", ".")))
        elif isinstance(matrix_cells, list):
            matrix_cell = next((item for item in matrix_cells if isinstance(item, Mapping) and str(item.get("cell_id", item.get("anchor_id", ""))).replace(".", ":") == cell_id), None)
        else:
            matrix_cell = None
        matrix_cell = matrix_cell if isinstance(matrix_cell, Mapping) else {}
        batch_size = matrix_cell.get("b_primary", matrix_payload.get("b_primary"))
        microbatch_count = matrix_cell.get("m_primary", matrix_payload.get("m_primary"))
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size not in BATCH_SIZES or isinstance(microbatch_count, bool) or not isinstance(microbatch_count, int):
            raise S208ProductionBlocked(f"matrix.{cell_id}:FROZEN_B_M_REQUIRED")
        rebind_row = rebind_cells[cell_id]
        artifact_refs = rebind_row.get("reference_artifact_refs")
        if not isinstance(artifact_refs, Mapping) or not isinstance(artifact_refs.get("reference_result"), str) or not isinstance(artifact_refs.get("reference_convergence_report"), str):
            raise S208ProductionBlocked(f"s205_rebind_plan:{cell_id}:REFERENCE_ARTIFACT_REFS_REQUIRED")
        if reference.get("candidate_ref") != artifact_refs.get("reference_result"):
            raise S208ProductionBlocked(f"reference.{cell_id}:REBIND_REFERENCE_MISMATCH")
        explicit_convergence = row.get("reference_convergence_ref") or row.get("convergence_ref") or artifact_refs.get("reference_convergence_report")
        if explicit_convergence is None and reference_convergence_refs is not None:
            explicit_convergence = reference_convergence_refs.get(cell_id)
        if not isinstance(explicit_convergence, str) or not explicit_convergence:
            raise S208ProductionBlocked(f"sizing.{cell_id}:BOUNDED_CONVERGENCE_REF_REQUIRED")
        convergence_ref = explicit_convergence
        convergence, _ = _load_authoritative_object(root, convergence_ref, field=f"convergence.{cell_id}")
        delta_plan, delta_plan_hash = _validate_delta_plan(
            root,
            _ref(delta_ref, f"delta_plan.{cell_id}.ref"),
            cell_id=cell_id,
            preregistration=prereg_payload,
            sizing=sizing,
        )
        lineage = convergence.get("external_lineage")
        lineage_prereg = lineage.get("preregistration") if isinstance(lineage, Mapping) else None
        if not isinstance(lineage_prereg, Mapping) or not isinstance(lineage_prereg.get("commit_ref"), str):
            raise S208ProductionBlocked(f"convergence.{cell_id}:S21_TASK_ARTIFACT_REF_REQUIRED")
        bound_prereg, bound_prereg_hash = _load_authoritative_object(root, lineage_prereg["commit_ref"], field=f"convergence.{cell_id}.preregistration")
        if bound_prereg != prereg_payload or bound_prereg_hash != lineage_prereg.get("artifact_hash") or convergence.get("formula_contract_hash") != bound_prereg_hash or convergence.get("formula_contract") != bound_prereg:
            raise S208ProductionBlocked(f"convergence.{cell_id}:S21_FORMULA_BINDING_MISMATCH")
        formula_hash = bound_prereg_hash
        g23_cell = g23_by_cell[cell_id]
        g23_identities = g23_cell.get("identities")
        g23_metrics = g23_cell.get("metrics")
        if not isinstance(g23_identities, Mapping) or not isinstance(g23_metrics, Mapping):
            raise S208ProductionBlocked(f"g23.{cell_id}:CELL_IDENTITY_FIELDS_REQUIRED")
        historical_source = _validate_historical_delta_source(
            root,
            convergence,
            cell_id=cell_id,
            formula_hash=formula_hash,
            sizing=sizing,
            sizing_hash=sizing_hash,
            registry_hash=str(registry.get("registry_hash")),
            precision=precision,
        )
        delta_payload, corrected_delta_hash = _load_corrected_delta_sidecar(
            root,
            ref=g23_identities.get("corrected_delta_sci_ref"),
            declared_hash=g23_identities.get("corrected_delta_sci_hash"),
            g23_cell=g23_cell,
            rebind_row=rebind_row,
            cell_id=cell_id,
            formula_hash=formula_hash,
            precision=precision,
            sizing=sizing,
            sizing_hash=sizing_hash,
            registry_hash=str(registry.get("registry_hash")),
            convergence=convergence,
            historical_source=historical_source,
            gate_calculator=g23_calculator,
            expected_output_root_ref=g23_output_root_ref,
        )
        expected_sidecar_binding = (str(g23_identities["corrected_delta_sci_ref"]), corrected_delta_hash)
        for binding_name, bindings in (("s206_pilot_report", pilot_cell_bindings), ("matrix", matrix_cell_bindings), ("g24b_gate", g24b_cell_bindings)):
            binding = bindings[cell_id]
            if binding["corrected_delta_sci_ref"] != expected_sidecar_binding[0] or binding["corrected_delta_sci_hash"] != corrected_delta_hash or binding["config_hash"] != g23_identities.get("config_hash") or binding["result_hash"] != g23_identities.get("result_hash") or g23_metrics.get("corrected_delta_sci_ref") != binding["corrected_delta_sci_ref"] or g23_metrics.get("corrected_delta_sci_hash") != binding["corrected_delta_sci_hash"] or g23_metrics.get("corrected_delta_sci_batch_sizes") != binding["corrected_delta_sci_batch_sizes"] or g23_metrics.get("delta_sci_source") != binding["delta_sci_source"]:
                raise S208ProductionBlocked(f"{binding_name}.{cell_id}:CORRECTED_DELTA_BINDING_MISMATCH")
        # The corrected sidecar is the only scientific margin source.  The
        # S2.4 index delta_ref is retained as the pre-sizing plan provenance;
        # its numeric table is never consumed here.
        delta: dict[str, float] = {}
        sidecar_table = delta_payload.get("delta_sci_by_endpoint")
        if not isinstance(sidecar_table, Mapping):
            raise S208ProductionBlocked(f"g23.{cell_id}:CORRECTED_DELTA_TABLE_REQUIRED")
        for short, endpoint in _S208_ENDPOINTS.items():
            values = sidecar_table.get(short)
            if not isinstance(values, Mapping):
                raise S208ProductionBlocked(f"g23.{cell_id}:{short}:CORRECTED_DELTA_ENDPOINT_REQUIRED")
            delta[endpoint] = _finite_positive(values.get(str(batch_size)), f"g23.{cell_id}:{short}:B{batch_size}")
        pilot_key = f"{cell_id}:{batch_size}:{microbatch_count}"
        pilot_measurement = pilot_by_cell.get(pilot_key)
        if pilot_measurement is None:
            raise S208ProductionBlocked(f"s206_pilot_report:{cell_id}:SELECTED_B_M_MISSING")
        binding = pilot_cell_bindings[cell_id]
        if any(pilot_measurement.get(source) != binding[target] for target, source in _S206_PILOT_BINDING_FIELDS.items()):
            raise S208ProductionBlocked(f"s206_pilot_report:{cell_id}:CORRECTED_DELTA_BINDING_MISMATCH")
        pilot_delta = pilot_measurement.get("delta_sci_by_endpoint")
        pilot_half = pilot_measurement.get("reference_half_width_by_endpoint")
        if not isinstance(pilot_delta, Mapping) or not isinstance(pilot_half, Mapping):
            raise S208ProductionBlocked(f"s206_pilot_report:{cell_id}:SIZING_FIELDS_REQUIRED")
        for pilot_short, endpoint in _S208_PILOT_ENDPOINTS.items():
            observed = _finite_positive(pilot_delta.get(pilot_short), f"s206_pilot_report.{cell_id}.{pilot_short}")
            if observed != delta[endpoint]:
                raise S208ProductionBlocked(f"s206_pilot_report:{cell_id}:{pilot_short}:DELTA_MISMATCH")
        convergence_plan = convergence.get("sizing_plan")
        if not isinstance(convergence_plan, Mapping) or convergence.get("sizing_plan_artifact_hash") not in (None, sizing_hash) or convergence_plan.get("artifact_hash") not in (None, sizing_hash):
            raise S208ProductionBlocked(f"convergence.{cell_id}:SIZING_PLAN_BINDING_MISMATCH")
        denominator = _bounded_sizing_denominator(root, reference=reference, convergence=convergence, delta=delta_payload, cell_id=cell_id, tau_coord=float(tau_coord))
        denominator_source = "selected_bounded_sizing_checkpoint"
        result = g24a_by_cell[cell_id]
        metrics = result.get("metrics") if isinstance(result, Mapping) else None
        if not isinstance(metrics, Mapping):
            raise S208ProductionBlocked(f"g24a.{cell_id}:METRICS_REQUIRED")
        raw_half_widths = {"model_total_signed_bias": metrics.get("h_ref_model_total"), "layer_total_l1_bias": metrics.get("h_ref_layer"), "module_total_l1_bias": metrics.get("h_ref_module")}
        half_widths = {key: _finite_positive(value, f"g24a.{cell_id}.{key}") if value != 0 else 0.0 for key, value in raw_half_widths.items()}
        for pilot_short, endpoint in _S208_PILOT_ENDPOINTS.items():
            observed = pilot_half.get(pilot_short)
            observed_value = _finite_positive(observed, f"s206_pilot_report.{cell_id}.{pilot_short}.half_width") if observed != 0 else 0.0
            if observed_value != half_widths[endpoint]:
                raise S208ProductionBlocked(f"s206_pilot_report:{cell_id}:{pilot_short}:HALF_WIDTH_MISMATCH")
        numeric_by_endpoint = metrics.get("epsilon_num_by_endpoint")
        if not isinstance(numeric_by_endpoint, Mapping):
            g23_cell_rows = g23_payload.get("cells")
            g23_cell = next((item for item in g23_cell_rows if isinstance(item, Mapping) and str(item.get("cell_id")).replace(".", ":") == cell_id), None) if isinstance(g23_cell_rows, list) else None
            g23_metrics = g23_cell.get("metrics") if isinstance(g23_cell, Mapping) else None
            numeric_by_endpoint = g23_metrics.get("epsilon_num_by_endpoint") if isinstance(g23_metrics, Mapping) else None
        if not isinstance(numeric_by_endpoint, Mapping):
            raise S208ProductionBlocked(f"g23_g24a.{cell_id}:NUMERIC_ERROR_BY_ENDPOINT_REQUIRED")
        numeric_error = max(_finite_positive(numeric_by_endpoint.get(short), f"g24a.{cell_id}.epsilon_num.{short}") if numeric_by_endpoint.get(short) != 0 else 0.0 for short in _S208_ENDPOINTS)
        cells[cell_id] = {
            "cell_id": cell_id,
            "batch_size": batch_size,
            "microbatch_count": microbatch_count,
            "nmse_denominator": denominator,
            "margins": delta,
            "reference_half_widths": half_widths,
            "numeric_error": numeric_error,
            "numeric_error_by_endpoint": { _S208_ENDPOINTS[short]: (float(numeric_by_endpoint[short]) if numeric_by_endpoint[short] != 0 else 0.0) for short in _S208_ENDPOINTS },
            "group_registry": layer_groups,
            "layer_group_registry": layer_groups,
            "module_group_registry": module_groups,
            "source_refs": {"sizing_ref": sizing_ref, "registry_ref": registry_ref, "delta_ref": delta_ref, "corrected_delta_sci_ref": g23_identities["corrected_delta_sci_ref"], "corrected_delta_sci_source_ref": delta_payload["source_producer_ref"], "convergence_ref": convergence_ref, "pilot_report_ref": pilot_ref, "g23_gate_ref": g23_gate_ref, "g24a_gate_ref": str(g24a_gate) if not isinstance(g24a_gate, Mapping) else None, "g24b_gate_ref": str(g24b_gate) if not isinstance(g24b_gate, Mapping) else None, "g23_gate": g23_hash, "g24a_gate": g24a_hash, "g24b_gate": g24b_hash},
            "source_hashes": {"sizing_hash": sizing_hash, "registry_hash": registry_hash, "delta_plan_hash": delta_plan_hash, "registry_artifact_hash": registry_hash, "manifest_hash": manifest_hash, "corrected_delta_sci_hash": corrected_delta_hash, "corrected_delta_sci_source_hash": delta_payload["source_producer_artifact_hash"], "g23_cell_result_hash": g23_identities["result_hash"], "g23_cell_config_hash": g23_identities["config_hash"]},
            "denominator_source": denominator_source,
        }
    body: dict[str, Any] = {"schema_version": S208_MATRIX_MATERIALIZATION_SCHEMA, "scope": "formal", "matrix_hash": matrix_hash, "materialization_index_hash": index_hash, "six_cell_manifest_hash": manifest_hash, "preregistration_hash": prereg_payload.get("preregistration_hash", prereg_hash), "formula_contract_hash": formula_hash, "pilot_report_hash": pilot_hash, "g23_gate_hash": g23_hash, "g24a_gate_hash": g24a_hash, "g24b_gate_hash": g24b_hash, "g24a_rebind_plan_hash": rebind_hash, "cells": cells}
    body["artifact_hash"] = canonical_json_hash(body)
    return body


__all__ = ["S208ProductionBlocked", "S208_MEMMAP_SCHEMA", "S208_REFERENCE_BUNDLE_SCHEMA", "S208_MATRIX_MATERIALIZATION_SCHEMA", "load_s208_reference_bundle", "materialize_s208_matrix"]
