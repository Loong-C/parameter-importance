"""Fail-closed S2.8/G2.6 statistics for the Stage 2 estimator study.

The S2.7 producer deliberately publishes only an immutable raw-results
manifest.  This module is the detached consumer for that manifest.  It never
loads a model, creates a draw, or silently drops a failed repetition.  The
statistical implementation is intentionally small and dependency-light so it
can be audited on the local CPU host before a formal run.

The public entry point is :func:`analyze_s208_g26`.  It accepts JSON mappings
or canonical JSON paths for the four upstream Gate records, the frozen matrix,
the preregistration/hypothesis contracts, the sealed raw manifest, and one
reference payload per primary cell.  A reference payload must contain the
three reference views plus either independent raw reference blocks or the
bounded producer's hash-bound jackknife variance vectors.  The latter uses
the preregistered equivalent independent-variance combination and never
reconstructs pseudo blocks.  Parameter coordinates are never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import random
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from ..contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from ..contracts.status import GateRecord, GateStatus
from .preregistration import (
    ABSOLUTE_FLOORS,
    BATCH_SIZES,
    MICROBATCH_COUNTS,
    PRIMARY_CELLS,
    TOP_FRACTIONS,
    validate_stage2_hypothesis_contract,
    validate_stage2_preregistration,
)


S28_ANALYSIS_SCHEMA = "stage2-s208-g26-analysis-v1"
S28_INPUT_AUDIT_SCHEMA = "stage2-s208-input-audit-v1"
S28_LINEAGE_SCHEMA = "stage2-s208-lineage-v1"
S28_FAMILY_SCHEMA = "stage2-s208-confirmatory-family-v1"
S28_GATE_SCHEMA = "stage2-s208-g26-gate-v1"
S28_RAW_MANIFEST_SCHEMA = "stage2-s27-sealed-raw-manifest-v1"
S28_PRIMARY_GATE_IDS = ("stage2.G2.3", "stage2.G2.4a", "stage2.G2.4b", "stage2.G2.5")
S28_CELL_IDS = tuple(f"{item['model']}:{item['stage']}" for item in PRIMARY_CELLS)
S28_METHODS = ("raw", "double")
S28_BIAS_METHODS = ("double", "u")
S28_BOOTSTRAP_UNIT = "repetition_with_reference_block_strata"
S28_CONFIDENCE_LEVELS = {"equivalence": 0.90, "upper": 0.95, "noninferiority": 0.95}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class S28G26Blocked(RuntimeError):
    """Raised whenever formal statistical validity cannot be established."""


def _finite(value: Any, *, field: str = "value") -> None:
    # Production S2.4 bundles are exposed as read-only ndarray/memmap views.
    # They are checked chunk-wise by the loader; retaining this recursive
    # guard here prevents a memmap from being mistaken for a JSON scalar while
    # still rejecting NaN/Inf before analysis.
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or not np.isfinite(value).all():
            raise S28G26Blocked(f"{field}:NONFINITE_OR_EMPTY_ARRAY")
        return
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise S28G26Blocked(f"{field}:NONFINITE")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite(item, field=f"{field}[{i}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise S28G26Blocked(f"{field}:NON_STRING_KEY")
            _finite(item, field=f"{field}.{key}")
        return
    raise S28G26Blocked(f"{field}:NOT_JSON_VALUE")


def _sha(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise S28G26Blocked(f"{field}:SHA256_REQUIRED")
    return value


def _load(value: Mapping[str, Any] | str | Path, *, field: str) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    else:
        try:
            loaded = load_canonical_json(value)
        except Exception as error:  # pragma: no cover - error text is not stable
            raise S28G26Blocked(f"{field}:CANONICAL_READ_FAILED") from error
        if not isinstance(loaded, Mapping):
            raise S28G26Blocked(f"{field}:OBJECT_REQUIRED")
        result = dict(loaded)
    _finite(result, field=field)
    return result


def _verify_hash(value: Mapping[str, Any], *, field: str, required: bool = True) -> str | None:
    if value.get("_streaming_payload") is True:
        declared = value.get("candidate_artifact_hash", value.get("reference_hash"))
        return _sha(declared, field=f"{field}.candidate_artifact_hash")
    declared = value.get("artifact_hash")
    if declared is None and not required:
        return None
    digest = _sha(declared, field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if digest != canonical_json_hash(body):
        raise S28G26Blocked(f"{field}:ARTIFACT_HASH_MISMATCH")
    return digest


def _path_under(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S28G26Blocked(f"{field}:UNSAFE_REFERENCE")
    candidate = (root / reference).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise S28G26Blocked(f"{field}:OUTSIDE_RAW_ROOT") from error
    return candidate


def _vector(value: Any, *, field: str, coordinate_ids: Sequence[str] | None = None) -> tuple[tuple[str, ...], np.ndarray]:
    """Normalize a signed vector while preserving a canonical coordinate order."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        keys = tuple(str(item) for item in (coordinate_ids or tuple(range(len(array)))))
    elif isinstance(value, Mapping):
        if not value:
            raise S28G26Blocked(f"{field}:EMPTY_VECTOR")
        keys = tuple(sorted(str(key) for key in value))
        if len(keys) != len(set(keys)):
            raise S28G26Blocked(f"{field}:DUPLICATE_COORDINATE")
        try:
            raw = [value[key] for key in keys]
            if raw and all(isinstance(item, np.ndarray) for item in raw):
                # Parameter-name -> tensor mappings are the native S2.4
                # representation.  Production loaders flatten them into a
                # memmap before this point; this fallback keeps small direct
                # bundles auditable without JSON list expansion.
                array = np.concatenate([np.asarray(item, dtype=np.float64).reshape(-1) for item in raw])
                keys = tuple(str(item) for item in (coordinate_ids or tuple(range(len(array)))))
            else:
                array = np.asarray([float(item) for item in raw], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise S28G26Blocked(f"{field}:NUMERIC_VECTOR_REQUIRED") from error
    elif isinstance(value, (list, tuple)):
        if not value:
            raise S28G26Blocked(f"{field}:EMPTY_VECTOR")
        try:
            array = np.asarray([float(item) for item in value], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise S28G26Blocked(f"{field}:NUMERIC_VECTOR_REQUIRED") from error
        keys = tuple(str(item) for item in (coordinate_ids or tuple(range(len(array)))))
    else:
        raise S28G26Blocked(f"{field}:VECTOR_REQUIRED")
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise S28G26Blocked(f"{field}:NONFINITE_OR_EMPTY_VECTOR")
    if len(keys) != array.size or len(set(keys)) != len(keys):
        raise S28G26Blocked(f"{field}:COORDINATE_ORDER_INVALID")
    return keys, array


def _same_vector(left: tuple[tuple[str, ...], np.ndarray], right: tuple[tuple[str, ...], np.ndarray], *, field: str) -> None:
    if left[0] != right[0] or left[1].shape != right[1].shape:
        raise S28G26Blocked(f"{field}:COORDINATE_REGISTRY_MISMATCH")


@dataclass(frozen=True)
class _VerifiedGate:
    gate_id: str
    artifact_hash: str
    measured: Mapping[str, Any]


def _verify_gate(value: Mapping[str, Any] | str | Path, gate_id: str) -> _VerifiedGate:
    payload = _load(value, field=gate_id)
    # G2.3 and G2.4a predate the generic GateRecord and have producer-specific
    # schemas.  Their qualification records are still content-addressed and
    # are accepted only after the complete six-cell PASS checks below.
    if payload.get("schema_version") in {"stage2-g23-reference-evaluation-v1", "stage2-g24a-formal-evaluation-v1"}:
        digest = _verify_hash(payload, field=gate_id)
        if payload.get("gate_id") not in (None, gate_id):
            raise S28G26Blocked(f"{gate_id}:ID_MISMATCH")
        if payload.get("status") != "PASS" or payload.get("formal_eligible") is not True:
            raise S28G26Blocked(f"{gate_id}:PASS_REQUIRED")
        if gate_id == "stage2.G2.3":
            cells = payload.get("cells")
            if payload.get("required_cell_count") != 6 or payload.get("complete_cell_count") != 6 or not isinstance(cells, list) or len(cells) != 6:
                raise S28G26Blocked(f"{gate_id}:SIX_CELL_COMPLETENESS_REQUIRED")
            if any(not isinstance(item, Mapping) or item.get("status") != "PASS" or item.get("formal_eligible") is False for item in cells):
                raise S28G26Blocked(f"{gate_id}:CELL_PASS_REQUIRED")
        else:
            results = payload.get("results")
            if payload.get("cell_count") != 6 or not isinstance(results, list) or len(results) != 6:
                raise S28G26Blocked(f"{gate_id}:SIX_CELL_COMPLETENESS_REQUIRED")
            if any(not isinstance(item, Mapping) or item.get("status") != "PASS" or item.get("formal_eligible") is not True for item in results):
                raise S28G26Blocked(f"{gate_id}:CELL_PASS_REQUIRED")
        return _VerifiedGate(gate_id, str(digest), dict(payload))
    try:
        gate = GateRecord.from_mapping(dict(payload))
    except Exception as error:
        raise S28G26Blocked(f"{gate_id}:INVALID_GATE_RECORD") from error
    if gate.gate_id != gate_id or gate.effective_status() is not GateStatus.PASS:
        raise S28G26Blocked(f"{gate_id}:PASS_REQUIRED")
    return _VerifiedGate(gate.gate_id, gate.artifact_hash, dict(gate.measured) if isinstance(gate.measured, Mapping) else {})


def _reference_views(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> tuple[
    dict[str, tuple[tuple[str, ...], np.ndarray]],
    dict[str, tuple[tuple[str, ...], np.ndarray]] | None,
    dict[str, tuple[tuple[str, ...], np.ndarray]] | None,
    tuple[tuple[str, ...], np.ndarray] | None,
    str,
]:
    """Read raw-block or bounded reference uncertainty without fabricating blocks."""

    declared_hash = _verify_hash(payload, field=field)
    del declared_hash
    view_sources: dict[str, Any] = {}
    vectors = payload.get("vectors")
    if isinstance(vectors, Mapping):
        view_sources.update(vectors)
    for name, aliases in {
        "bias": ("bias_reference", "bias"),
        "cross": ("cross_reference", "cross"),
        "ranking": ("ranking_reference", "ranking"),
    }.items():
        for alias in aliases:
            if alias in payload:
                view_sources.setdefault(name, payload[alias])
    missing = [name for name in ("bias", "cross", "ranking") if name not in view_sources]
    if missing:
        raise S28G26Blocked(f"{field}:REFERENCE_VIEWS_MISSING:{','.join(missing)}")
    coordinate_ids = payload.get("coordinate_ids")
    if coordinate_ids is not None and (not isinstance(coordinate_ids, list) or not all(isinstance(item, str) for item in coordinate_ids)):
        raise S28G26Blocked(f"{field}:COORDINATE_IDS_INVALID")
    views: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in ("bias", "cross", "ranking"):
        views[name] = _vector(view_sources[name], field=f"{field}.{name}", coordinate_ids=coordinate_ids)
    for name in ("cross", "ranking"):
        _same_vector(views["bias"], views[name], field=f"{field}.{name}")

    raw_blocks = payload.get("reference_blocks", payload.get("block_vectors"))
    if isinstance(raw_blocks, Mapping):
        blocks: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        for name in ("bias", "cross", "ranking"):
            source = raw_blocks.get(name)
            if not isinstance(source, list) or len(source) < 3:
                raise S28G26Blocked(f"{field}.{name}:AT_LEAST_THREE_BLOCKS_REQUIRED")
            normalized = [_vector(item, field=f"{field}.{name}[{index}]", coordinate_ids=views[name][0]) for index, item in enumerate(source)]
            for item in normalized:
                _same_vector(views[name], item, field=f"{field}.{name}.block")
            # Keep formal block matrices on disk when the source vectors are
            # memmaps.  ``np.stack`` here used to duplicate every reference
            # block in RAM (catastrophic for 14M/31M coordinates).
            arrays = [item[1] for item in normalized]
            if arrays and all(isinstance(item, np.memmap) for item in arrays):
                source = arrays[0]
                target_path = Path(source.filename).with_name(Path(source.filename).name + f".{name}.blocks.f64.dat")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                expected_size = len(arrays) * int(source.size) * 8
                if target_path.exists() and target_path.stat().st_size != expected_size:
                    raise S28G26Blocked(f"{field}.{name}:BLOCK_MEMMAP_COLLISION")
                if not target_path.exists():
                    with target_path.open("wb") as handle:
                        handle.truncate(expected_size)
                stacked = np.memmap(target_path, mode="r+", dtype=np.float64, shape=(len(arrays), source.size))
                for index, array in enumerate(arrays):
                    stacked[index] = array
                stacked.flush()
                stacked.flags.writeable = False
            else:
                stacked = np.stack(arrays, axis=0)
            blocks[name] = (views[name][0], stacked)
        return views, blocks, None, None, "reference_block_bootstrap"

    raw_variances = payload.get("reference_variances")
    metadata = payload.get("reference_uncertainty")
    if not isinstance(raw_variances, Mapping) or not isinstance(metadata, Mapping):
        raise S28G26Blocked(f"{field}:REFERENCE_BLOCKS_OR_BOUNDED_VARIANCE_REQUIRED")
    if (
        metadata.get("schema_version") != "stage2-reference-uncertainty-v1"
        or metadata.get("estimator") != "block_u_delete_one_jackknife"
        or not isinstance(metadata.get("block_count_a"), int)
        or not isinstance(metadata.get("block_count_b"), int)
        or int(metadata["block_count_a"]) < 3
        or int(metadata["block_count_b"]) < 3
    ):
        raise S28G26Blocked(f"{field}:BOUNDED_UNCERTAINTY_METADATA_INVALID")
    variances: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for name in ("bias", "cross", "ranking"):
        item = _vector(raw_variances.get(name), field=f"{field}.{name}_variance", coordinate_ids=views[name][0])
        _same_vector(views[name], item, field=f"{field}.{name}_variance")
        if np.any(item[1] < 0):
            raise S28G26Blocked(f"{field}.{name}:REFERENCE_VARIANCE_NEGATIVE")
        variances[name] = item
    raw_sequence = payload.get("sequence_variance")
    if raw_sequence is None:
        raise S28G26Blocked(f"{field}:BOUNDED_SEQUENCE_VARIANCE_REQUIRED")
    sequence = _vector(raw_sequence, field=f"{field}.sequence_variance", coordinate_ids=views["bias"][0])
    _same_vector(views["bias"], sequence, field=f"{field}.sequence_variance")
    if np.any(sequence[1] < 0):
        raise S28G26Blocked(f"{field}:SEQUENCE_VARIANCE_NEGATIVE")
    return views, None, variances, sequence, "independent_reference_variance_combination"


def _read_raw_payload(
    root: Path,
    descriptor: Mapping[str, Any],
    *,
    memmap_root: Path | None = None,
) -> Mapping[str, Any]:
    ref = descriptor.get("raw_artifact_ref")
    if not isinstance(ref, str):
        inline = descriptor.get("artifact")
        if not isinstance(inline, Mapping):
            raise S28G26Blocked("raw.unit:RAW_ARTIFACT_REFERENCE_REQUIRED")
        payload = dict(inline)
        source_path = None
    else:
        source_path = _path_under(root, ref, field="raw_artifact_ref")
        if not source_path.exists():
            raise S28G26Blocked(f"raw_artifact_ref:NOT_FOUND:{ref}")
        if source_path.is_dir():
            if memmap_root is None:
                raise S28G26Blocked("raw_artifact_bundle:EXPLICIT_MEMMAP_ROOT_REQUIRED")
            try:
                from .stage2_s208_production import _decode_bundle, _flat_memmap

                cache_root = memmap_root.resolve() / "raw"
                cache_stem = hashlib.sha256(str(ref).encode("utf-8")).hexdigest()[:24]
                state, manifest_hash = _decode_bundle(source_path, cache_root, cache_stem)
                vectors = state.get("vectors")
                if not isinstance(vectors, Mapping):
                    raise S28G26Blocked("raw_artifact_bundle:VECTORS_REQUIRED")
                coordinate_ids = state.get("coordinate_ids") if isinstance(state.get("coordinate_ids"), list) else []
                flat_vectors: dict[str, np.ndarray] = {}
                for method, vector in vectors.items():
                    ids: list[str] = list(coordinate_ids) if coordinate_ids else []
                    flat_vectors[str(method)] = _flat_memmap(vector, ids, cache_root, f"{cache_stem}-{method}")
                    if coordinate_ids and ids != coordinate_ids:
                        raise S28G26Blocked("raw_artifact_bundle:COORDINATE_REGISTRY_MISMATCH")
                    if not coordinate_ids:
                        coordinate_ids = ids
                payload = {key: value for key, value in state.items() if key != "vectors"}
                payload.update({"vectors": flat_vectors, "coordinate_ids": coordinate_ids, "_streaming_payload": True, "candidate_artifact_hash": manifest_hash})
            except S28G26Blocked:
                raise
            except (OSError, TypeError, ValueError) as error:
                raise S28G26Blocked("raw_artifact_bundle:INVALID") from error
        else:
            try:
                payload = dict(_load(source_path, field=f"raw_artifact_ref:{ref}"))
            except S28G26Blocked:
                raise
    declared = descriptor.get("raw_artifact_hash")
    _sha(declared, field="raw_artifact_hash")
    # Producer artifacts may carry their own canonical hash; otherwise the
    # descriptor binds the full canonical body.  A byte hash is accepted only
    # when it is exactly the hash of the canonical source file.
    if payload.get("_streaming_payload") is True:
        body_hash = str(payload.get("candidate_artifact_hash", ""))
    else:
        body_hash = canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"})
    object_hash = payload.get("artifact_hash")
    candidates = {body_hash, object_hash}
    if source_path is not None and source_path.is_file():
        candidates.add(hashlib.sha256(source_path.read_bytes()).hexdigest())
    if declared not in candidates:
        raise S28G26Blocked("raw_artifact_hash:CONTENT_MISMATCH")
    _finite(payload, field="raw_artifact")
    return payload


def _extract_vectors(payload: Mapping[str, Any], *, field: str) -> dict[str, tuple[tuple[str, ...], np.ndarray]]:
    source = payload.get("vectors")
    if not isinstance(source, Mapping) and isinstance(payload.get("state"), Mapping):
        source = payload["state"].get("vectors")  # type: ignore[union-attr]
    if not isinstance(source, Mapping):
        raise S28G26Blocked(f"{field}:VECTORS_REQUIRED")
    result: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    coordinate_ids = payload.get("coordinate_ids")
    for method, value in source.items():
        if not isinstance(method, str):
            raise S28G26Blocked(f"{field}:METHOD_ID_INVALID")
        result[method] = _vector(value, field=f"{field}.{method}", coordinate_ids=coordinate_ids if isinstance(coordinate_ids, list) else None)
    if not result:
        raise S28G26Blocked(f"{field}:EMPTY_METHOD_SET")
    return result


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _top_metrics(observed: np.ndarray, expected: np.ndarray) -> dict[str, float | int]:
    result: dict[str, float | int] = {}
    for fraction in TOP_FRACTIONS:
        key = f"{fraction:g}"
        k = max(1, int(math.ceil(fraction * observed.size)))
        observed_top = set(np.argsort(-np.abs(observed), kind="mergesort")[:k].tolist())
        expected_top = set(np.argsort(-np.abs(expected), kind="mergesort")[:k].tolist())
        overlap = len(observed_top.intersection(expected_top)) / k
        union = len(observed_top.union(expected_top))
        result[f"overlap_at_{key}"] = float(overlap)
        result[f"jaccard_at_{key}"] = float(len(observed_top.intersection(expected_top)) / union) if union else 0.0
        result[f"k_at_{key}"] = k
    return result


def _bootstrap(
    outer: Sequence[np.ndarray],
    reference_blocks: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Two-stage bootstrap over repetitions and reference blocks only."""

    if len(outer) < 2 or reference_blocks.ndim != 2 or reference_blocks.shape[0] < 3:
        raise S28G26Blocked("BOOTSTRAP_REQUIRES_REPETITIONS_AND_REFERENCE_BLOCKS")
    if replicates < 100:
        raise S28G26Blocked("BOOTSTRAP_REPLICATES_TOO_SMALL")
    rng = random.Random(int(seed))
    outer_values = tuple(outer)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        outer_indices = [rng.randrange(len(outer_values)) for _ in range(len(outer_values))]
        inner_indices = [rng.randrange(reference_blocks.shape[0]) for _ in range(reference_blocks.shape[0])]
        outer_mean = np.zeros_like(outer_values[0], dtype=np.float64)
        for item in outer_indices:
            outer_mean += outer_values[item]
        outer_mean /= float(len(outer_indices))
        reference_mean = np.zeros_like(reference_blocks[0], dtype=np.float64)
        for item in inner_indices:
            reference_mean += reference_blocks[item]
        reference_mean /= float(len(inner_indices))
        draws[index] = statistic(outer_mean, reference_mean)
    if not np.isfinite(draws).all():
        raise S28G26Blocked("BOOTSTRAP_NONFINITE")
    return {
        "unit": S28_BOOTSTRAP_UNIT,
        "replicates": int(replicates),
        "seed": int(seed),
        "quantile_0.025": float(np.quantile(draws, 0.025)),
        "quantile_0.05": float(np.quantile(draws, 0.05)),
        "quantile_0.95": float(np.quantile(draws, 0.95)),
        "quantile_0.975": float(np.quantile(draws, 0.975)),
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "parameter_coordinate_resampling": False,
        "reference_uncertainty_mode": "reference_block_bootstrap",
    }


def _independent_variance_bootstrap(
    outer: Sequence[np.ndarray],
    reference: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    reference_standard_error: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Combine repetition bootstrap variance with an independent reference SE.

    The bounded S2.4 producer publishes the delete-one-block jackknife
    variance, but deliberately does not retain raw block vectors.  The S2.1
    plan permits an equivalent independent variance combination.  We therefore
    bootstrap only the independent estimator repetitions and add one scalar
    reference-error draw at the endpoint level.  Coordinates are never
    resampled and no pseudo reference blocks are reconstructed.
    """

    if len(outer) < 2 or replicates < 100:
        raise S28G26Blocked("BOOTSTRAP_REQUIRES_REPETITIONS_AND_100_REPLICATES")
    if not math.isfinite(reference_standard_error) or reference_standard_error < 0:
        raise S28G26Blocked("REFERENCE_STANDARD_ERROR_INVALID")
    rng = random.Random(int(seed))
    outer_values = tuple(outer)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        outer_mean = np.zeros_like(outer_values[0], dtype=np.float64)
        for _ in range(len(outer_values)):
            outer_mean += outer_values[rng.randrange(len(outer_values))]
        outer_mean /= float(len(outer_values))
        reference_error = rng.gauss(0.0, reference_standard_error) if reference_standard_error else 0.0
        draws[index] = statistic(outer_mean, reference) + reference_error
    if not np.isfinite(draws).all():
        raise S28G26Blocked("BOOTSTRAP_NONFINITE")
    return {
        "unit": S28_BOOTSTRAP_UNIT,
        "replicates": int(replicates),
        "seed": int(seed),
        "quantile_0.025": float(np.quantile(draws, 0.025)),
        "quantile_0.05": float(np.quantile(draws, 0.05)),
        "quantile_0.95": float(np.quantile(draws, 0.95)),
        "quantile_0.975": float(np.quantile(draws, 0.975)),
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "parameter_coordinate_resampling": False,
        "reference_uncertainty_mode": "independent_reference_variance_combination",
        "reference_standard_error": float(reference_standard_error),
        "coordinate_covariance_assumption": "none_worst_case_standard_error_bound",
        "raw_reference_blocks_reconstructed": False,
    }


def two_stage_bootstrap(
    estimator_vectors: Sequence[Sequence[float]],
    reference_blocks: Sequence[Sequence[float]],
    *,
    metric: str = "signed_bias_sum",
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Public bootstrap helper used by tests and downstream audit scripts."""

    outer = [np.asarray(item, dtype=np.float64) for item in estimator_vectors]
    blocks = np.asarray(reference_blocks, dtype=np.float64)
    if not outer or blocks.ndim != 2 or any(item.shape != outer[0].shape for item in outer) or blocks.shape[1] != outer[0].size:
        raise S28G26Blocked("BOOTSTRAP_VECTOR_SHAPES_INVALID")
    if metric == "signed_bias_sum":
        fn = lambda mean, reference: float(np.sum(mean - reference))
    elif metric == "absolute_bias_sum":
        fn = lambda mean, reference: float(np.sum(np.abs(mean - reference)))
    elif metric == "nmse":
        fn = lambda mean, reference: float(np.mean(np.square(mean - reference)))
    else:
        raise S28G26Blocked(f"BOOTSTRAP_METRIC_UNSUPPORTED:{metric}")
    return _bootstrap(outer, blocks, fn, replicates=replicates, seed=seed)


@dataclass(frozen=True)
class _CellData:
    cell_id: str
    model: str
    stage: str
    batch_size: int
    microbatch_count: int
    repetitions: Mapping[str, tuple[dict[str, tuple[tuple[str, ...], np.ndarray]], Mapping[str, Any]]]
    references: Mapping[str, tuple[tuple[str, ...], np.ndarray]]
    reference_blocks: Mapping[str, np.ndarray] | None
    reference_variances: Mapping[str, np.ndarray] | None
    sequence_variance: np.ndarray | None
    reference_uncertainty_mode: str
    denominator: float
    margins: Mapping[str, float]
    reference_half_width: float
    reference_half_widths: Mapping[str, float]
    numeric_error: float
    group_registry: Mapping[str, tuple[str, ...]]


def _reference_variance_vector(cell: _CellData, view: str) -> np.ndarray:
    if cell.reference_variances is None or view not in cell.reference_variances:
        raise S28G26Blocked(f"reference.{cell.cell_id}.{view}:BOUNDED_VARIANCE_REQUIRED")
    return cell.reference_variances[view]


def _reference_scalar_standard_error(cell: _CellData, view: str) -> float:
    # The producer retains only marginal jackknife variances.  Summing marginal
    # standard errors is the Cauchy upper bound for any compatible coordinate
    # covariance matrix; unlike sqrt(sum(var)), it does not assume coordinate
    # independence.
    variance = _reference_variance_vector(cell, view)
    return float(np.sum(np.sqrt(variance)))


def _cell_bootstrap(
    cell: _CellData,
    outer: Sequence[np.ndarray],
    view: str,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    *,
    replicates: int,
    seed: int,
    reference_standard_error: float | None = None,
) -> dict[str, Any]:
    if cell.reference_blocks is not None:
        return _bootstrap(
            outer,
            cell.reference_blocks[view],
            statistic,
            replicates=replicates,
            seed=seed,
        )
    if cell.reference_variances is None:
        raise S28G26Blocked(f"reference.{cell.cell_id}:UNCERTAINTY_REQUIRED")
    return _independent_variance_bootstrap(
        outer,
        cell.references[view][1],
        statistic,
        reference_standard_error=(
            _reference_scalar_standard_error(cell, view)
            if reference_standard_error is None
            else reference_standard_error
        ),
        replicates=replicates,
        seed=seed,
    )


def _cell_parts(cell_id: str) -> tuple[str, str]:
    if cell_id not in S28_CELL_IDS or ":" not in cell_id:
        raise S28G26Blocked(f"cell:{cell_id}:PRIMARY_CELL_REQUIRED")
    return tuple(cell_id.split(":", 1))  # type: ignore[return-value]


def _resolve_matrix_cell(matrix: Mapping[str, Any], cell_id: str) -> Mapping[str, Any]:
    for key in ("cells", "anchors", "scientific_margins"):
        source = matrix.get(key)
        if isinstance(source, Mapping) and isinstance(source.get(cell_id), Mapping):
            return dict(source[cell_id])
        if isinstance(source, list):
            for item in source:
                if isinstance(item, Mapping) and item.get("cell_id") == cell_id:
                    return dict(item)
    return {}


def _extract_cell_data(
    cell_id: str,
    descriptors: Sequence[Mapping[str, Any]],
    *,
    raw_root: Path,
    reference_payload: Mapping[str, Any],
    matrix: Mapping[str, Any],
    memmap_root: Path | None = None,
) -> _CellData:
    model, stage = _cell_parts(cell_id)
    cell_matrix = _resolve_matrix_cell(matrix, cell_id)
    views, blocks, variances, sequence_variance, uncertainty_mode = _reference_views(reference_payload, field=f"reference.{cell_id}")
    reference_hash = reference_payload.get("reference_hash", reference_payload.get("artifact_hash"))
    if reference_hash is None:
        raise S28G26Blocked(f"reference.{cell_id}:REFERENCE_HASH_REQUIRED")
    _sha(reference_hash, field=f"reference.{cell_id}.reference_hash")
    rows: dict[str, tuple[dict[str, tuple[tuple[str, ...], np.ndarray]], Mapping[str, Any]]] = {}
    batch_values: set[int] = set()
    m_values: set[int] = set()
    for descriptor in descriptors:
        if descriptor.get("status") != "SUCCESS":
            raise S28G26Blocked(f"raw.{cell_id}:FAILED_UNIT_PRESENT")
        unit_id = descriptor.get("unit_id")
        repetition_id = descriptor.get("repetition_id")
        if not isinstance(unit_id, str) or not isinstance(repetition_id, str):
            raise S28G26Blocked(f"raw.{cell_id}:UNIT_ID_REQUIRED")
        if unit_id in rows or repetition_id in {item[1].get("repetition_id") for item in rows.values()}:
            raise S28G26Blocked(f"raw.{cell_id}:DUPLICATE_REPETITION")
        payload = _read_raw_payload(raw_root, descriptor, memmap_root=memmap_root)
        if payload.get("unit_id") not in (None, unit_id, repetition_id):
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:UNIT_IDENTITY_MISMATCH")
        if payload.get("cell_id") not in (None, cell_id):
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:CELL_IDENTITY_MISMATCH")
        if payload.get("reference_hash") not in (None, reference_hash):
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:REFERENCE_HASH_MISMATCH")
        if payload.get("clamp_applied") is True or payload.get("clip_mode") not in (None, "none") or payload.get("mean_gradient_consistent") is False:
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:ESTIMATOR_INTEGRITY_MARKER_FAILED")
        m2_error = payload.get(
            "m2_identity_max_abs_error",
            payload.get("m2_identity_max_abs", payload.get("m2_double_max_abs_error")),
        )
        if m2_error is not None and (isinstance(m2_error, bool) or not isinstance(m2_error, (int, float)) or not math.isfinite(float(m2_error)) or float(m2_error) > 1.0e-12):
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:M2_DOUBLE_IDENTITY_FAILED")
        vectors = _extract_vectors(payload, field=f"raw.{cell_id}.{unit_id}")
        for method in S28_METHODS:
            if method not in vectors:
                raise S28G26Blocked(f"raw.{cell_id}.{unit_id}:METHOD_MISSING:{method}")
            _same_vector(views["bias"], vectors[method], field=f"raw.{cell_id}.{unit_id}.{method}")
        m_primary = payload.get("microbatch_count", payload.get("m_primary"))
        b_primary = payload.get("batch_size", payload.get("b_primary"))
        if b_primary is None:
            b_primary = cell_matrix.get("batch_size", cell_matrix.get("b_primary", matrix.get("b_primary")))
        if m_primary is None:
            m_primary = cell_matrix.get("microbatch_count", cell_matrix.get("m_primary", matrix.get("m_primary")))
        if isinstance(b_primary, bool) or not isinstance(b_primary, int) or isinstance(m_primary, bool) or not isinstance(m_primary, int):
            raise S28G26Blocked(f"raw.{cell_id}:{unit_id}:B_M_REQUIRED")
        u_method = f"u_m{m_primary}"
        if u_method not in vectors:
            raise S28G26Blocked(f"raw.{cell_id}.{unit_id}:METHOD_MISSING:{u_method}")
        _same_vector(views["bias"], vectors[u_method], field=f"raw.{cell_id}.{unit_id}.{u_method}")
        batch_values.add(b_primary)
        m_values.add(m_primary)
        rows[unit_id] = (vectors, {**payload, "repetition_id": repetition_id, "batch_size": b_primary, "microbatch_count": m_primary})
    if not rows or len(batch_values) != 1 or len(m_values) != 1:
        raise S28G26Blocked(f"raw.{cell_id}:B_M_NOT_CONSTANT")
    b_value, m_value = next(iter(batch_values)), next(iter(m_values))
    expected_b = matrix.get("b_primary")
    expected_m = matrix.get("m_primary")
    if expected_b is not None and b_value != expected_b or expected_m is not None and m_value != expected_m:
        raise S28G26Blocked(f"raw.{cell_id}:B_M_NOT_FROZEN_PRIMARY")
    denominator = cell_matrix.get("nmse_denominator", cell_matrix.get("sizing_denominator", matrix.get("nmse_denominator")))
    if denominator is None:
        denominator = reference_payload.get("sizing_denominator")
    if isinstance(denominator, bool) or not isinstance(denominator, (int, float)) or not math.isfinite(float(denominator)) or float(denominator) <= 0:
        raise S28G26Blocked(f"reference.{cell_id}:FROZEN_NMSE_DENOMINATOR_REQUIRED")
    if any(key in cell_matrix for key in ("model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias", "model_total")):
        raw_margins = cell_matrix
    else:
        raw_margins = cell_matrix.get("margins", cell_matrix.get("scientific_margins", matrix.get("scientific_margins")))
    if not isinstance(raw_margins, Mapping):
        raw_margins = matrix.get("delta_sci")
    if not isinstance(raw_margins, Mapping):
        raise S28G26Blocked(f"matrix.{cell_id}:FROZEN_SCIENTIFIC_MARGINS_REQUIRED")
    margins: dict[str, float] = {}
    for endpoint in ("model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias"):
        raw_margin = raw_margins.get(endpoint)
        if raw_margin is None and endpoint == "model_total_signed_bias":
            raw_margin = raw_margins.get("model_total")
        if isinstance(raw_margin, Mapping):
            raw_margin = raw_margin.get("margin", raw_margin.get("delta_sci"))
        if isinstance(raw_margin, bool) or not isinstance(raw_margin, (int, float)) or not math.isfinite(float(raw_margin)) or float(raw_margin) <= 0:
            raise S28G26Blocked(f"matrix.{cell_id}:{endpoint}:FROZEN_MARGIN_REQUIRED")
        margins[endpoint] = float(raw_margin)
    raw_half_widths = cell_matrix.get("reference_half_widths", cell_matrix.get("reference_half_width_by_endpoint"))
    half_widths: dict[str, float] = {}
    if isinstance(raw_half_widths, Mapping):
        aliases = {
            "model_total_signed_bias": ("model_total_signed_bias", "model_total", "bias", "h_ref_model_total"),
            "layer_total_l1_bias": ("layer_total_l1_bias", "layer", "nmse", "h_ref_layer"),
            "module_total_l1_bias": ("module_total_l1_bias", "module", "rank", "h_ref_module"),
        }
        for endpoint, names in aliases.items():
            value = next((raw_half_widths[name] for name in names if name in raw_half_widths), None)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
                raise S28G26Blocked(f"reference.{cell_id}:{endpoint}:REFERENCE_HALF_WIDTH_REQUIRED")
            half_widths[endpoint] = float(value)
        half_width = max(half_widths.values())
    else:
        half_width = cell_matrix.get("reference_half_width", reference_payload.get("bias_half_width_l2"))
        if isinstance(half_width, bool) or not isinstance(half_width, (int, float)) or not math.isfinite(float(half_width)) or float(half_width) < 0:
            raise S28G26Blocked(f"reference.{cell_id}:REFERENCE_HALF_WIDTH_REQUIRED")
        half_widths = {endpoint: float(half_width) for endpoint in margins}
    numeric_error = cell_matrix.get("numeric_error", reference_payload.get("numeric_error", 0.0))
    if isinstance(numeric_error, bool) or not isinstance(numeric_error, (int, float)) or not math.isfinite(float(numeric_error)) or float(numeric_error) < 0:
        raise S28G26Blocked(f"reference.{cell_id}:NUMERIC_ERROR_INVALID")
    groups_source = cell_matrix.get("group_registry", matrix.get("group_registry"))
    if not isinstance(groups_source, Mapping):
        raise S28G26Blocked(f"matrix.{cell_id}:CANONICAL_GROUP_REGISTRY_REQUIRED")
    ids = views["bias"][0]
    groups: dict[str, tuple[str, ...]] = {}
    seen: set[str] = set()
    for name, coordinates in groups_source.items():
        if not isinstance(name, str) or not isinstance(coordinates, list) or not coordinates or not all(isinstance(item, str) for item in coordinates):
            raise S28G26Blocked(f"matrix.{cell_id}:GROUP_REGISTRY_INVALID")
        if set(coordinates).difference(ids) or seen.intersection(coordinates):
            raise S28G26Blocked(f"matrix.{cell_id}:GROUP_REGISTRY_OVERLAP_OR_UNKNOWN_COORDINATE")
        seen.update(coordinates)
        groups[name] = tuple(coordinates)
    if not groups or seen != set(ids):
        raise S28G26Blocked(f"matrix.{cell_id}:GROUP_REGISTRY_NOT_EXHAUSTIVE")
    return _CellData(
        cell_id,
        model,
        stage,
        b_value,
        m_value,
        rows,
        views,
        None if blocks is None else {name: value[1] for name, value in blocks.items()},
        None if variances is None else {name: value[1] for name, value in variances.items()},
        None if sequence_variance is None else sequence_variance[1],
        uncertainty_mode,
        float(denominator),
        margins,
        float(half_width),
        half_widths,
        float(numeric_error),
        groups,
    )


def _group_values(vector: np.ndarray, ids: Sequence[str], groups: Mapping[str, Sequence[str]]) -> dict[str, float]:
    positions = {item: index for index, item in enumerate(ids)}
    return {name: float(sum(vector[positions[item]] for item in coordinates)) for name, coordinates in groups.items()}


def _mean_vectors(vectors: Sequence[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise S28G26Blocked("VECTOR_SEQUENCE_EMPTY")
    result = np.zeros_like(vectors[0], dtype=np.float64)
    for vector in vectors:
        if vector.shape != result.shape:
            raise S28G26Blocked("VECTOR_SEQUENCE_SHAPE_MISMATCH")
        result += vector
    result /= float(len(vectors))
    return result


def _mean_square_error(vectors: Sequence[np.ndarray], target: np.ndarray) -> float:
    return sum(float(np.mean(np.square(vector - target))) for vector in vectors) / float(len(vectors))


def _variance_scalar(vectors: Sequence[np.ndarray], mean: np.ndarray, *, ddof: int) -> float:
    if len(vectors) <= ddof:
        return 0.0
    return sum(float(np.mean(np.square(vector - mean))) for vector in vectors) / float(len(vectors) - ddof)


def _negative_stats(vectors: Sequence[np.ndarray]) -> tuple[float, float, float]:
    count = 0
    negative_mass = 0.0
    positive_mass = 0.0
    for vector in vectors:
        count += int(np.count_nonzero(vector < 0))
        negative_mass += float(np.abs(vector[vector < 0]).sum())
        positive_mass += float(vector[vector > 0].sum())
    return count / float(len(vectors) * vectors[0].size), negative_mass, positive_mass


def _method_statistics(cell: _CellData, method: str, *, bootstrap_replicates: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = cell.references["bias"][0]
    reference = cell.references["bias"][1]
    rows = list(cell.repetitions.values())
    vectors = [item[0][method][1] for item in rows]
    mean = _mean_vectors(vectors)
    error = mean - reference
    v_r = _variance_scalar(vectors, mean, ddof=0)
    s2 = _variance_scalar(vectors, mean, ddof=1)
    observed_nmse = float(np.mean(np.square(error)) / cell.denominator)
    if cell.reference_blocks is not None:
        ref_blocks = cell.reference_blocks["bias"]
        v_ref = _variance_scalar(tuple(ref_blocks), _mean_vectors(tuple(ref_blocks)), ddof=1) / cell.denominator
    else:
        # S2.1 defines V_ref as trace(Sigma_ref) / D_c.  The bounded producer
        # publishes that diagonal exactly through its jackknife variance.
        v_ref = float(np.sum(_reference_variance_vector(cell, "bias"))) / cell.denominator
    negative_fraction, negative_mass, positive_mass = _negative_stats(vectors)
    corrected = observed_nmse - v_ref
    long_rows: list[dict[str, Any]] = []
    for view_name in ("bias", "cross", "ranking"):
        target = cell.references[view_name][1]
        view_error = mean - target
        view_seed = {"bias": 11, "cross": 23, "ranking": 37}[view_name]
        bootstrap = _cell_bootstrap(
            cell,
            vectors,
            view_name,
            lambda m, r: float(np.sum(m - r)),
            replicates=bootstrap_replicates,
            seed=seed + view_seed,
        )
        metric: dict[str, Any] = {
            "cell_id": cell.cell_id,
            "model": cell.model,
            "training_stage": cell.stage,
            "batch_size": cell.batch_size,
            "microbatch_count": cell.microbatch_count,
            "method": method,
            "scope": "parameter",
            "reference_view": view_name,
            "repetitions": len(rows),
            "coordinate_count": int(mean.shape[0]),
            "signed_bias": float(np.sum(view_error)),
            "absolute_bias": float(np.sum(np.abs(view_error))),
            "variance_s2": s2,
            "variance_v_r": v_r,
            "mse_observed": _mean_square_error(vectors, target),
            "mae": float(sum(float(np.mean(np.abs(vector - target))) for vector in vectors) / len(vectors)),
            "bootstrap": bootstrap,
            "negative_fraction": negative_fraction,
            "negative_mass": negative_mass,
            "positive_mass": positive_mass,
        }
        if view_name == "ranking":
            metric["pearson"] = _pearson(mean, target)
            metric["spearman"] = _spearman(mean, target)
            metric.update(_top_metrics(mean, target))
        long_rows.append(metric)
    rank_target = cell.references["ranking"][1]
    ranking = {"pearson": [], "spearman": [], **{f"overlap_at_{fraction:g}": [] for fraction in TOP_FRACTIONS}, **{f"jaccard_at_{fraction:g}": [] for fraction in TOP_FRACTIONS}}
    for vector in vectors:
        ranking["pearson"].append(_pearson(vector, rank_target))
        ranking["spearman"].append(_spearman(vector, rank_target))
        top = _top_metrics(vector, rank_target)
        for name in ranking:
            if name in top:
                ranking[name].append(top[name])
    long_rows.append({"cell_id": cell.cell_id, "model": cell.model, "training_stage": cell.stage, "batch_size": cell.batch_size, "microbatch_count": cell.microbatch_count, "method": method, "scope": "parameter", "reference_view": "ranking_repetition_mean", "repetitions": len(rows), "coordinate_count": int(mean.shape[0]), "pearson": _pearson(mean, rank_target), "spearman": _spearman(mean, rank_target), **_top_metrics(mean, rank_target)})
    summary = {
        "cell_id": cell.cell_id,
        "model": cell.model,
        "training_stage": cell.stage,
        "batch_size": cell.batch_size,
        "microbatch_count": cell.microbatch_count,
        "method": method,
        "repetitions": len(rows),
        "signed_bias": float(np.sum(error)),
        "absolute_bias": float(np.sum(np.abs(error))),
        "variance_s2": s2,
        "variance_v_r": v_r,
        "mse_observed": float(np.mean(np.square(error)) + v_r),
        "mse_identity_residual": float(_mean_square_error(vectors, reference) - (v_r + np.mean(np.square(error)))),
        "reference_variance_v_ref": v_ref,
        "corrected_nmse": corrected,
        "negative_fraction": negative_fraction,
        "negative_mass": negative_mass,
        "positive_mass": positive_mass,
        "ranking_repetition_mean": {name: float(np.mean(values)) for name, values in ranking.items()},
        "ranking_repeat_mean": {"pearson": _pearson(mean, rank_target), "spearman": _spearman(mean, rank_target), **_top_metrics(mean, rank_target)},
        "reference_sensitivity": {view: float(np.sum(mean - cell.references[view][1])) for view in ("bias", "cross", "ranking")},
    }
    return long_rows, summary


def _family_endpoint(cell: _CellData, method: str, endpoint: str, *, bootstrap_replicates: int, seed: int) -> dict[str, Any]:
    ids = cell.references["bias"][0]
    target = cell.references["bias"][1]
    vectors = [item[0][method][1] for item in cell.repetitions.values()]
    mean = _mean_vectors(vectors)
    bias = mean - target
    groups = _group_values(bias, ids, cell.group_registry)
    if endpoint == "model_total_signed_bias":
        observed = float(np.sum(bias))
        statistic = lambda m, r: float(np.sum(m - r))
        interval_name = "joint_90"
        lower, upper = 0.05, 0.95
    elif endpoint == "layer_total_l1_bias":
        observed = float(sum(abs(value) for value in groups.values()))
        statistic = lambda m, r: float(sum(abs(value) for value in _group_values(m - r, ids, cell.group_registry).values()))
        interval_name = "upper_95"
        lower, upper = 0.0, 0.95
    elif endpoint == "module_total_l1_bias":
        observed = float(sum(abs(value) for value in groups.values()))
        statistic = lambda m, r: float(sum(abs(value) for value in _group_values(m - r, ids, cell.group_registry).values()))
        interval_name = "upper_95"
        lower, upper = 0.0, 0.95
    else:  # pragma: no cover - endpoint list is closed above
        raise S28G26Blocked(f"endpoint:{endpoint}:UNSUPPORTED")
    bootstrap = _cell_bootstrap(
        cell,
        vectors,
        "bias",
        statistic,
        replicates=bootstrap_replicates,
        seed=seed,
        # In bounded mode the authoritative G2.4a endpoint half-width is
        # combined below as a conservative bound.  Passing zero here keeps
        # the outer repetition bootstrap distinct and auditable.
        reference_standard_error=0.0 if cell.reference_blocks is None else None,
    )
    reference_bound = cell.reference_half_widths[endpoint] if cell.reference_blocks is None else 0.0
    if reference_bound:
        bootstrap["reference_uncertainty_mode"] = "independent_reference_endpoint_bound_combination"
        bootstrap["reference_error_bound"] = reference_bound
        bootstrap["raw_reference_blocks_reconstructed"] = False
    if interval_name == "joint_90":
        lo, hi = float(bootstrap["quantile_0.05"]) - reference_bound, float(bootstrap["quantile_0.95"]) + reference_bound
        passed = lo >= -cell.margins[endpoint] and hi <= cell.margins[endpoint]
    else:
        lo, hi = 0.0, float(bootstrap["quantile_0.95"]) + reference_bound
        passed = hi < cell.margins[endpoint]
    endpoint_half_width = cell.reference_half_widths[endpoint]
    precision_ok = endpoint_half_width <= cell.margins[endpoint] / 4.0 and cell.numeric_error <= cell.margins[endpoint] / 10.0
    state = "PASS" if passed and precision_ok else "INCONCLUSIVE" if passed or not precision_ok else "FAIL"
    cross_bootstrap = _cell_bootstrap(
        cell,
        vectors,
        "cross",
        statistic,
        replicates=bootstrap_replicates,
        seed=seed + 77,
        reference_standard_error=0.0 if cell.reference_blocks is None else None,
    )
    if reference_bound:
        cross_bootstrap["reference_uncertainty_mode"] = "independent_reference_endpoint_bound_combination"
        cross_bootstrap["reference_error_bound"] = reference_bound
        cross_bootstrap["raw_reference_blocks_reconstructed"] = False
    if interval_name == "joint_90":
        cross_pass = float(cross_bootstrap["quantile_0.05"]) - reference_bound >= -cell.margins[endpoint] and float(cross_bootstrap["quantile_0.95"]) + reference_bound <= cell.margins[endpoint]
    else:
        cross_pass = float(cross_bootstrap["quantile_0.95"]) + reference_bound < cell.margins[endpoint]
    if cross_pass != passed:
        state = "INCONCLUSIVE"
    return {"cell_id": cell.cell_id, "method": method, "endpoint": endpoint, "effect": observed, "interval": {"lower": lo, "upper": hi, "confidence_level": S28_CONFIDENCE_LEVELS["equivalence"] if interval_name == "joint_90" else S28_CONFIDENCE_LEVELS["upper"]}, "margin": cell.margins[endpoint], "reference_half_width": endpoint_half_width, "numeric_error": cell.numeric_error, "reference_precision_pass": precision_ok, "cross_reference_state": "PASS" if cross_pass else "FAIL", "bootstrap": bootstrap, "cross_bootstrap": cross_bootstrap, "multiplicity": "intersection_union_across_six_primary_cells", "state": state}


def _paired_bootstrap(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    reference_blocks: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Paired two-stage bootstrap for estimator comparisons.

    The same outer repetition index is used for both estimators, preserving
    the S2.5 paired design.  Inner draws resample reference blocks only.
    """

    if len(left) != len(right) or len(left) < 2 or reference_blocks.ndim != 2 or reference_blocks.shape[0] < 3:
        raise S28G26Blocked("PAIRED_BOOTSTRAP_SHAPES_INVALID")
    rng = random.Random(int(seed))
    left_values, right_values = tuple(left), tuple(right)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        outer_indices = [rng.randrange(len(left_values)) for _ in range(len(left_values))]
        inner_indices = [rng.randrange(reference_blocks.shape[0]) for _ in range(reference_blocks.shape[0])]
        left_mean = np.zeros_like(left_values[0], dtype=np.float64)
        right_mean = np.zeros_like(right_values[0], dtype=np.float64)
        for item in outer_indices:
            left_mean += left_values[item]
            right_mean += right_values[item]
        left_mean /= float(len(outer_indices))
        right_mean /= float(len(outer_indices))
        reference_mean = np.zeros_like(reference_blocks[0], dtype=np.float64)
        for item in inner_indices:
            reference_mean += reference_blocks[item]
        reference_mean /= float(len(inner_indices))
        draws[index] = statistic(left_mean, right_mean, reference_mean)
    if not np.isfinite(draws).all():
        raise S28G26Blocked("PAIRED_BOOTSTRAP_NONFINITE")
    return {"unit": S28_BOOTSTRAP_UNIT, "replicates": int(replicates), "seed": int(seed), "quantile_0.025": float(np.quantile(draws, 0.025)), "quantile_0.05": float(np.quantile(draws, 0.05)), "quantile_0.95": float(np.quantile(draws, 0.95)), "quantile_0.975": float(np.quantile(draws, 0.975)), "mean": float(draws.mean()), "std": float(draws.std(ddof=1)), "parameter_coordinate_resampling": False, "reference_uncertainty_mode": "reference_block_bootstrap"}


def _paired_independent_variance_bootstrap(
    left: Sequence[np.ndarray],
    right: Sequence[np.ndarray],
    reference: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray, np.ndarray], float],
    *,
    reference_standard_error: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right) or len(left) < 2 or replicates < 100:
        raise S28G26Blocked("PAIRED_BOOTSTRAP_SHAPES_INVALID")
    if not math.isfinite(reference_standard_error) or reference_standard_error < 0:
        raise S28G26Blocked("REFERENCE_STANDARD_ERROR_INVALID")
    rng = random.Random(int(seed))
    left_values, right_values = tuple(left), tuple(right)
    draws = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        left_mean = np.zeros_like(left_values[0], dtype=np.float64)
        right_mean = np.zeros_like(right_values[0], dtype=np.float64)
        for _ in range(len(left_values)):
            item = rng.randrange(len(left_values))
            left_mean += left_values[item]
            right_mean += right_values[item]
        left_mean /= float(len(left_values))
        right_mean /= float(len(right_values))
        reference_error = rng.gauss(0.0, reference_standard_error) if reference_standard_error else 0.0
        draws[index] = statistic(left_mean, right_mean, reference) + reference_error
    if not np.isfinite(draws).all():
        raise S28G26Blocked("PAIRED_BOOTSTRAP_NONFINITE")
    return {
        "unit": S28_BOOTSTRAP_UNIT,
        "replicates": int(replicates),
        "seed": int(seed),
        "quantile_0.025": float(np.quantile(draws, 0.025)),
        "quantile_0.05": float(np.quantile(draws, 0.05)),
        "quantile_0.95": float(np.quantile(draws, 0.95)),
        "quantile_0.975": float(np.quantile(draws, 0.975)),
        "mean": float(draws.mean()),
        "std": float(draws.std(ddof=1)),
        "parameter_coordinate_resampling": False,
        "reference_uncertainty_mode": "independent_reference_variance_combination",
        "reference_standard_error": float(reference_standard_error),
        "coordinate_covariance_assumption": "none_worst_case_standard_error_bound",
        "raw_reference_blocks_reconstructed": False,
    }


def _noninferiority_rows(cell: _CellData, *, bootstrap_replicates: int, seed: int) -> list[dict[str, Any]]:
    """Compute preregistered U-vs-double secondary family endpoints."""

    u_method = f"u_m{cell.microbatch_count}"
    left = [item[0][u_method][1] for item in cell.repetitions.values()]
    right = [item[0]["double"][1] for item in cell.repetitions.values()]
    target = cell.references["bias"][1]
    ref_blocks = None if cell.reference_blocks is None else cell.reference_blocks["bias"]
    denominator = cell.denominator
    if ref_blocks is not None:
        ref_var = float(np.mean(np.var(ref_blocks, axis=0, ddof=1)) / denominator)
    else:
        ref_var = float(np.sum(_reference_variance_vector(cell, "bias")) / denominator)
    left_mean, right_mean = _mean_vectors(left), _mean_vectors(right)
    u_nmse = float(np.mean(np.square(left_mean - target)) / denominator - ref_var)
    d_nmse = float(np.mean(np.square(right_mean - target)) / denominator - ref_var)
    rows: list[dict[str, Any]] = []
    if d_nmse <= float(ABSOLUTE_FLOORS["tau_nmse"]):
        nmse_state = "INCONCLUSIVE"
        nmse_bootstrap: dict[str, Any] = {"reason": "DOUBLE_CORRECTED_NMSE_NOT_ABOVE_POSITIVE_FLOOR"}
    else:
        statistic = lambda u, d, r: float((np.mean(np.square(u - r)) / denominator - ref_var) / (np.mean(np.square(d - r)) / denominator - ref_var))
        if ref_blocks is not None:
            nmse_bootstrap = _paired_bootstrap(
                left,
                right,
                ref_blocks,
                statistic,
                replicates=bootstrap_replicates,
                seed=seed,
            )
        else:
            variance = _reference_variance_vector(cell, "bias")
            numerator = np.mean(np.square(left_mean - target)) / denominator - ref_var
            baseline = np.mean(np.square(right_mean - target)) / denominator - ref_var
            scale = float(target.size) * denominator
            derivative_left = 2.0 * (target - left_mean) / scale
            derivative_right = 2.0 * (target - right_mean) / scale
            derivative = (derivative_left * baseline - numerator * derivative_right) / (baseline * baseline)
            reference_se = float(np.sum(np.abs(derivative) * np.sqrt(variance)))
            nmse_bootstrap = _paired_independent_variance_bootstrap(
                left,
                right,
                target,
                statistic,
                reference_standard_error=0.0,
                replicates=bootstrap_replicates,
                seed=seed,
            )
            reference_bound = 1.96 * reference_se
            nmse_bootstrap["reference_uncertainty_mode"] = "independent_reference_delta_bound_combination"
            nmse_bootstrap["reference_standard_error_bound"] = reference_se
            nmse_bootstrap["reference_error_bound_95"] = reference_bound
            nmse_bootstrap["combined_quantile_0.95"] = float(nmse_bootstrap["quantile_0.95"]) + reference_bound
        nmse_upper = float(nmse_bootstrap.get("combined_quantile_0.95", nmse_bootstrap["quantile_0.95"]))
        nmse_state = "PASS" if nmse_upper <= 1.10 else "FAIL"
    rows.append({"cell_id": cell.cell_id, "method": u_method, "endpoint": "corrected_parameter_nmse_noninferiority", "effect": u_nmse / d_nmse if d_nmse > 0 else None, "baseline": "double", "threshold": {"upper": 1.10, "positive_floor": float(ABSOLUTE_FLOORS["tau_nmse"])}, "interval": nmse_bootstrap, "multiplicity": "intersection_union_across_six_primary_cells", "state": nmse_state})
    rank_target = cell.references["ranking"][1]
    rank_statistic = lambda u, d, r: _spearman(u, r) - _spearman(d, r)
    if cell.reference_blocks is not None:
        rank_bootstrap = _paired_bootstrap(left, right, cell.reference_blocks["ranking"], rank_statistic, replicates=bootstrap_replicates, seed=seed + 17)
        rank_state = "PASS" if float(rank_bootstrap["quantile_0.05"]) >= -0.02 else "FAIL"
    elif not np.any(_reference_variance_vector(cell, "ranking")):
        rank_bootstrap = _paired_independent_variance_bootstrap(left, right, rank_target, rank_statistic, reference_standard_error=0.0, replicates=bootstrap_replicates, seed=seed + 17)
        rank_state = "PASS" if float(rank_bootstrap["quantile_0.05"]) >= -0.02 else "FAIL"
    else:
        rank_bootstrap = {"reason": "BOUNDED_RANKING_VARIANCE_HAS_NO_RAW_BLOCK_ORDER", "reference_uncertainty_mode": cell.reference_uncertainty_mode, "raw_reference_blocks_reconstructed": False}
        rank_state = "INCONCLUSIVE"
    rows.append({"cell_id": cell.cell_id, "method": u_method, "endpoint": "parameter_spearman_noninferiority", "effect": _spearman(left_mean, rank_target) - _spearman(right_mean, rank_target), "baseline": "double", "threshold": {"lower": -0.02}, "interval": rank_bootstrap, "multiplicity": "intersection_union_across_six_primary_cells", "state": rank_state})
    overlap_key = "overlap_at_0.01"
    overlap_statistic = lambda u, d, r: float(_top_metrics(u, r)[overlap_key] - _top_metrics(d, r)[overlap_key])
    if cell.reference_blocks is not None:
        overlap_bootstrap = _paired_bootstrap(left, right, cell.reference_blocks["ranking"], overlap_statistic, replicates=bootstrap_replicates, seed=seed + 31)
        overlap_state = "PASS" if float(overlap_bootstrap["quantile_0.05"]) >= -0.03 else "FAIL"
    elif not np.any(_reference_variance_vector(cell, "ranking")):
        overlap_bootstrap = _paired_independent_variance_bootstrap(left, right, rank_target, overlap_statistic, reference_standard_error=0.0, replicates=bootstrap_replicates, seed=seed + 31)
        overlap_state = "PASS" if float(overlap_bootstrap["quantile_0.05"]) >= -0.03 else "FAIL"
    else:
        overlap_bootstrap = {"reason": "BOUNDED_RANKING_VARIANCE_HAS_NO_RAW_BLOCK_ORDER", "reference_uncertainty_mode": cell.reference_uncertainty_mode, "raw_reference_blocks_reconstructed": False}
        overlap_state = "INCONCLUSIVE"
    rows.append({"cell_id": cell.cell_id, "method": u_method, "endpoint": "parameter_overlap_at_1_percent_noninferiority", "effect": float(_top_metrics(left_mean, rank_target)[overlap_key] - _top_metrics(right_mean, rank_target)[overlap_key]), "baseline": "double", "threshold": {"lower": -0.03}, "interval": overlap_bootstrap, "multiplicity": "intersection_union_across_six_primary_cells", "state": overlap_state})
    return rows


def _raw_calibration_row(cell: _CellData) -> dict[str, Any]:
    """Return the preregistered raw-vs-sigma²/B calibration status.

    S2.7 may omit sequence-variance vectors from a lightweight raw artifact;
    in that case the row is explicitly inconclusive rather than substituting
    a variance estimated from the same repetitions.
    """

    if cell.sequence_variance is not None:
        theory = cell.sequence_variance / cell.batch_size
        sequence_source = "s204_hash_bound_sequence_variance"
    else:
        values: list[np.ndarray] = []
        for vectors, payload in cell.repetitions.values():
            source = payload.get("sequence_variance", payload.get("sigma2"))
            if source is None:
                return {"cell_id": cell.cell_id, "batch_size": cell.batch_size, "status": "INCONCLUSIVE", "reason": "SEQUENCE_VARIANCE_NOT_BOUND"}
            ids, vector = _vector(source, field=f"raw_calibration.{cell.cell_id}", coordinate_ids=cell.references["bias"][0])
            if ids != cell.references["bias"][0]:
                raise S28G26Blocked(f"raw_calibration.{cell.cell_id}:COORDINATE_REGISTRY_MISMATCH")
            values.append(vector)
        theory = _mean_vectors(values) / cell.batch_size
        sequence_source = "s27_repetition_payload"
    raw_vectors = [item[0]["raw"][1] for item in cell.repetitions.values()]
    observed = _mean_vectors(raw_vectors) - cell.references["bias"][1]
    slope = float(np.dot(theory, observed) / np.dot(theory, theory)) if float(np.dot(theory, theory)) > 0 else 0.0
    intercept = float(np.mean(observed - slope * theory))
    return {"cell_id": cell.cell_id, "batch_size": cell.batch_size, "observed_signed_bias_sum": float(np.sum(observed)), "theoretical_sigma2_over_B_sum": float(np.sum(theory)), "slope": slope, "intercept": intercept, "slope_threshold": [0.8, 1.2], "sequence_variance_source": sequence_source, "status": "PASS" if 0.8 <= slope <= 1.2 else "NOT_SUPPORTED"}


def _quality_record(name: str, status: str, *, observed: Any = None, threshold: Any = None, reasons: Sequence[str] = (), evidence: Sequence[str] = ()) -> dict[str, Any]:
    return {"gate": name, "status": status, "observed": observed, "threshold": threshold, "reasons": list(reasons), "evidence_refs": list(evidence)}


def _decisions(family: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], *, quality_pass: bool, noninferiority: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    by_hyp: dict[str, str] = {f"H{i}": "inconclusive" for i in range(1, 7)}
    if not quality_pass:
        return {"schema_version": "stage2-s208-hypothesis-decisions-v1", "decision_states": by_hyp, "quality_gate_dependency": "blocked", "multiplicity": "preregistered_family_no_posthoc_promotion", "evidence_refs": ["confirmatory_family_decisions.json", "statistics_summary.json"], "effect_rows": [], "thresholds": {}}
    family_states = {(str(item["method"]), str(item["cell_id"])): str(item["state"]) for item in family}
    family_methods = sorted({str(item["method"]) for item in family})
    for method in family_methods:
        if method == "double":
            continue
        if all(family_states.get((method, cell), "INCONCLUSIVE") == "PASS" for cell in S28_CELL_IDS):
            by_hyp["H3"] = "supported"
            break
    if by_hyp["H3"] == "inconclusive" and any(value == "FAIL" for value in family_states.values()):
        by_hyp["H3"] = "not_supported"
    if noninferiority:
        endpoints = {str(item["endpoint"]) for item in noninferiority}
        grouped = {endpoint: [str(item["state"]) for item in noninferiority if item.get("endpoint") == endpoint] for endpoint in endpoints}
        if all(states and all(state == "PASS" for state in states) for states in grouped.values()):
            by_hyp["H5"] = "supported"
            by_hyp["H6"] = "supported"
        elif any("FAIL" in states for states in grouped.values()):
            by_hyp["H5"] = "not_supported"
            by_hyp["H6"] = "not_supported"
    # H1/H2/H4/H5/H6 are deliberately conservative: their criteria require
    # secondary B/M/cost strata and are not inferred from a single confirmatory
    # B/M wave.
    if summaries:
        # H6 is a descriptive decision only when both methods have ranking data
        # and the paired non-inferiority endpoint is present in every cell.
        ranking_rows = [item for item in summaries if item.get("method") in S28_METHODS and isinstance(item.get("ranking_repeat_mean"), Mapping)]
        if len(ranking_rows) >= len(S28_CELL_IDS) * 2:
            by_hyp["H6"] = "inconclusive"
    return {"schema_version": "stage2-s208-hypothesis-decisions-v1", "decision_states": by_hyp, "quality_gate_dependency": "passed" if quality_pass else "blocked", "multiplicity": "preregistered_family_no_posthoc_promotion", "primary_family_methods": {method: all(family_states.get((method, cell), "INCONCLUSIVE") == "PASS" for cell in S28_CELL_IDS) for method in family_methods}, "evidence_refs": ["confirmatory_family_decisions.json", "statistics_summary.json"], "effect_rows": [dict(item) for item in family], "noninferiority_effect_rows": [dict(item) for item in noninferiority], "thresholds": {"u_double_nmse_ratio_upper": 1.10, "u_double_spearman_difference_lower": -0.02, "u_double_overlap_1pct_difference_lower": -0.03}}


def _write_once(root: Path, name: str, value: Mapping[str, Any]) -> str:
    path = root / name
    if path.exists():
        raise S28G26Blocked(f"OUTPUT_ALREADY_EXISTS:{name}")
    write_canonical_json(path, value)
    return name


def analyze_s208_g26(
    *,
    raw_manifest: Mapping[str, Any] | str | Path,
    raw_root: str | Path | None = None,
    references: Mapping[str, Any] | str | Path,
    matrix: Mapping[str, Any] | str | Path,
    preregistration: Mapping[str, Any] | str | Path,
    hypothesis_contract: Mapping[str, Any] | str | Path,
    upstream_gates: Mapping[str, Mapping[str, Any] | str | Path],
    output_root: str | Path | None = None,
    memmap_root: str | Path | None = None,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260825,
) -> dict[str, Any]:
    """Run the detached, formal-eligible S2.8/G2.6 analysis.

    The function raises :class:`S28G26Blocked` before publishing derived
    results when any input identity, completeness, freeze, or statistical
    invariant fails.  A successful run publishes all artifacts exactly once
    into a new analysis directory.
    """

    prereg = _load(preregistration, field="preregistration")
    hypothesis = _load(hypothesis_contract, field="hypothesis_contract")
    try:
        validate_stage2_preregistration(prereg)
        validate_stage2_hypothesis_contract(hypothesis, preregistration=prereg)
    except Exception as error:
        raise S28G26Blocked("PREREGISTRATION_OR_HYPOTHESIS_CONTRACT_INVALID") from error
    gates: dict[str, GateRecord] = {}
    for gate_id in S28_PRIMARY_GATE_IDS:
        if gate_id not in upstream_gates:
            raise S28G26Blocked(f"{gate_id}:INPUT_REQUIRED")
        gates[gate_id] = _verify_gate(upstream_gates[gate_id], gate_id)
    manifest = _load(raw_manifest, field="raw_manifest")
    manifest_hash = _verify_hash(manifest, field="raw_manifest")
    if manifest.get("schema_version") != S28_RAW_MANIFEST_SCHEMA or manifest.get("status") != "SEALED" or manifest.get("formal_eligible") is not True or manifest.get("stream") not in (None, "confirmatory"):
        raise S28G26Blocked("raw_manifest:SEALED_FORMAL_REQUIRED")
    if gates["stage2.G2.5"].measured.get("raw_manifest_hash") != manifest_hash:
        raise S28G26Blocked("stage2.G2.5:RAW_MANIFEST_HASH_MISMATCH")
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        raise S28G26Blocked("raw_manifest:UNITS_REQUIRED")
    expected_count = manifest.get("expected_unit_count")
    completed_count = manifest.get("completed_unit_count")
    failed_count = manifest.get("failed_unit_count")
    if expected_count != completed_count or failed_count != 0 or len(units) != expected_count:
        raise S28G26Blocked("raw_manifest:COMPLETENESS_DENOMINATOR_FAILED")
    grouped: dict[str, list[Mapping[str, Any]]] = {cell: [] for cell in S28_CELL_IDS}
    draw_hashes: list[str] = []
    for descriptor in units:
        if not isinstance(descriptor, Mapping) or descriptor.get("cell_id") not in grouped:
            raise S28G26Blocked("raw_manifest:PRIMARY_CELL_SET_MISMATCH")
        draw_hash = descriptor.get("draw_id_hash")
        if draw_hash is not None:
            draw_hashes.append(_sha(draw_hash, field="raw.unit.draw_id_hash"))
        grouped[str(descriptor["cell_id"])].append(dict(descriptor))
    if draw_hashes and len(draw_hashes) != len(set(draw_hashes)):
        raise S28G26Blocked("raw_manifest:DRAW_ID_COLLISION")
    if any(not grouped[cell] for cell in S28_CELL_IDS):
        raise S28G26Blocked("raw_manifest:SIX_PRIMARY_CELLS_REQUIRED")
    matrix_payload = _load(matrix, field="matrix")
    matrix_hash = _verify_hash(matrix_payload, field="matrix")
    if matrix_payload.get("preregistration_hash") not in (None, prereg.get("preregistration_hash")):
        raise S28G26Blocked("matrix:PREREGISTRATION_HASH_MISMATCH")
    if manifest.get("matrix_hash") not in (None, matrix_hash):
        raise S28G26Blocked("raw_manifest:MATRIX_HASH_MISMATCH")
    if matrix_payload.get("status") not in ("FORMAL_FROZEN", "FROZEN") or matrix_payload.get("formal_eligible") is not True:
        raise S28G26Blocked("matrix:FORMAL_FROZEN_REQUIRED")
    if matrix_payload.get("qualification_gate_hash") not in (None, gates["stage2.G2.4b"].artifact_hash):
        raise S28G26Blocked("matrix:G2.4B_BINDING_MISMATCH")
    # Current S2.6 matrices carry the G2.4a identity through execution
    # evidence; newer matrix revisions may expose it directly.  If present,
    # never allow a silent mismatch (and retain the explicit gate requirement
    # above for older producer schemas).
    direct_g24a = matrix_payload.get("g24a_gate_hash", matrix_payload.get("runner_gate_hash"))
    if direct_g24a is not None and direct_g24a != gates["stage2.G2.4a"].artifact_hash:
        raise S28G26Blocked("matrix:G2.4A_BINDING_MISMATCH")
    references_payload = _load(references, field="references")
    raw_root_path = Path(raw_root) if raw_root is not None else (Path(raw_manifest).parent if isinstance(raw_manifest, (str, Path)) else Path.cwd())
    refs_by_cell: dict[str, Mapping[str, Any]] = {}
    source_refs = references_payload.get("cells", references_payload)
    if not isinstance(source_refs, Mapping):
        raise S28G26Blocked("references:CELL_MAPPING_REQUIRED")
    for cell in S28_CELL_IDS:
        source = source_refs.get(cell)
        if not isinstance(source, Mapping):
            raise S28G26Blocked(f"reference.{cell}:INPUT_REQUIRED")
        refs_by_cell[cell] = dict(source)
        if source.get("formal_eligible") is not True:
            raise S28G26Blocked(f"reference.{cell}:FORMAL_REFERENCE_REQUIRED")
        metadata = source.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("qualification_gate_hash") != gates["stage2.G2.3"].artifact_hash:
            raise S28G26Blocked(f"reference.{cell}:G2.3_BINDING_MISMATCH")
    cell_data: list[_CellData] = []
    for cell in S28_CELL_IDS:
        cell_data.append(
            _extract_cell_data(
                cell,
                grouped[cell],
                raw_root=raw_root_path,
                reference_payload=refs_by_cell[cell],
                matrix=matrix_payload,
                memmap_root=None if memmap_root is None else Path(memmap_root).resolve(),
            )
        )
    long_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    noninferiority_rows: list[dict[str, Any]] = []
    raw_calibration_rows: list[dict[str, Any]] = []
    for index, cell in enumerate(cell_data):
        methods_for_cell = ("raw", "double", f"u_m{cell.microbatch_count}")
        for method_index, method in enumerate(methods_for_cell):
            rows, summary = _method_statistics(cell, method, bootstrap_replicates=bootstrap_replicates, seed=bootstrap_seed + index * 100 + method_index * 17)
            long_rows.extend(rows)
            summaries.append(summary)
        for method_index, method in enumerate(("double", f"u_m{cell.microbatch_count}")):
            for endpoint_index, endpoint in enumerate(("model_total_signed_bias", "layer_total_l1_bias", "module_total_l1_bias")):
                family_rows.append(_family_endpoint(cell, method, endpoint, bootstrap_replicates=bootstrap_replicates, seed=bootstrap_seed + index * 1000 + method_index * 100 + endpoint_index * 17))
        noninferiority_rows.extend(_noninferiority_rows(cell, bootstrap_replicates=bootstrap_replicates, seed=bootstrap_seed + index * 2000))
        raw_calibration_rows.append(_raw_calibration_row(cell))
    quality_metrics = {
        "six_primary_cells": len(cell_data) == 6,
        "methods": ["raw", "double", *sorted({f"u_m{cell.microbatch_count}" for cell in cell_data})],
        "bootstrap_unit": S28_BOOTSTRAP_UNIT,
        "parameter_coordinate_resampling": False,
        "raw_manifest_hash": manifest_hash,
        "matrix_hash": matrix_hash,
        "repetitions_per_cell": {cell.cell_id: len(cell.repetitions) for cell in cell_data},
        "reference_uncertainty_modes": {cell.cell_id: cell.reference_uncertainty_mode for cell in cell_data},
    }
    precision_ok = all(bool(item.get("reference_precision_pass")) for item in family_rows)
    quality_records = [
        _quality_record("fixed_state", "PASS", observed={"clamp_applied": False, "clip_mode": "none"}, threshold={"required": True}),
        _quality_record("sample_independence", "PASS", observed={"draw_id_reuse_checked": True}, threshold={"cross_stream_reuse": False}),
        _quality_record("reference_convergence", "PASS" if precision_ok else "BLOCKED", observed={"reference_half_widths": {cell.cell_id: dict(cell.reference_half_widths) for cell in cell_data}}, threshold={"precision_margin_divisors": {"reference": 4.0, "numeric": 10.0}}, reasons=() if precision_ok else ("REFERENCE_OR_NUMERIC_PRECISION_INSUFFICIENT",)),
        _quality_record("result_completeness", "PASS", observed={"expected": expected_count, "completed": completed_count, "failed": failed_count}, threshold={"failed": 0}),
        _quality_record("finite_numeric_values", "PASS", observed={"statistics_rows": len(long_rows)}, threshold={"nonfinite": 0}),
        _quality_record("fair_total_draw_budget", "PASS", observed={"batch_sizes": sorted({cell.batch_size for cell in cell_data})}, threshold={"equal_budget": True}),
        _quality_record("replayability", "PASS", observed={"raw_manifest_hash": manifest_hash, "lineage_hash_bound": True}, threshold={"immutable_inputs": True}),
    ]
    quality_pass = all(item["status"] == "PASS" for item in quality_records)
    decisions = _decisions(family_rows, summaries, quality_pass=quality_pass, noninferiority=noninferiority_rows)
    family_methods = sorted({str(item["method"]) for item in family_rows})
    family_qualified = {method: {"bias_qualified": all((item["method"] != method) or (item["state"] == "PASS") for item in family_rows)} for method in family_methods}
    noninf_endpoints = ("corrected_parameter_nmse_noninferiority", "parameter_spearman_noninferiority", "parameter_overlap_at_1_percent_noninferiority")
    noninf_global = {}
    for endpoint in noninf_endpoints:
        endpoint_rows = [item for item in noninferiority_rows if item["endpoint"] == endpoint]
        noninf_global[endpoint] = {"all_cells": bool(endpoint_rows) and all(item["state"] == "PASS" for item in endpoint_rows), "states_by_cell": {str(item["cell_id"]): str(item["state"]) for item in endpoint_rows}}
    family_payload: dict[str, Any] = {"schema_version": S28_FAMILY_SCHEMA, "primary_cells": list(S28_CELL_IDS), "b_primary": matrix_payload.get("b_primary"), "m_primary": matrix_payload.get("m_primary"), "multiplicity": "intersection_union_across_six_primary_cells", "methods": family_methods, "rows": family_rows, "noninferiority_rows": noninferiority_rows, "global": family_qualified, "noninferiority_global": noninf_global, "artifact_hash": ""}
    family_payload["artifact_hash"] = canonical_json_hash({key: value for key, value in family_payload.items() if key != "artifact_hash"})
    audit: dict[str, Any] = {"schema_version": S28_INPUT_AUDIT_SCHEMA, "raw_manifest_hash": manifest_hash, "matrix_hash": matrix_hash, "gate_hashes": {key: value.artifact_hash for key, value in gates.items()}, "primary_cells": list(S28_CELL_IDS), "unit_count": len(units), "units_per_cell": {cell.cell_id: len(cell.repetitions) for cell in cell_data}, "bootstrap_unit": S28_BOOTSTRAP_UNIT, "reference_uncertainty_modes": {cell.cell_id: cell.reference_uncertainty_mode for cell in cell_data}, "raw_reference_blocks_reconstructed": False, "parameter_coordinate_pseudoreplication": False, "status": "PASS", "artifact_hash": ""}
    audit["artifact_hash"] = canonical_json_hash({key: value for key, value in audit.items() if key != "artifact_hash"})
    lineage: dict[str, Any] = {"schema_version": S28_LINEAGE_SCHEMA, "producer_task": "stage2.08_statistics_and_robustness", "consumer_commit_required": "record_at_execution", "raw_manifest_hash": manifest_hash, "matrix_hash": matrix_hash, "gate_hashes": {key: value.artifact_hash for key, value in gates.items()}, "reference_hashes": {cell.cell_id: refs_by_cell[cell.cell_id].get("reference_hash", refs_by_cell[cell.cell_id].get("artifact_hash")) for cell in cell_data}, "derived_artifacts": ["analysis_input_audit.json", "statistics_long_table.json", "statistics_summary.json", "raw_calibration.json", "confirmatory_family_decisions.json", "quality_gates.json", "hypothesis_decisions.json", "lineage_manifest.json"], "artifact_hash": ""}
    lineage["artifact_hash"] = canonical_json_hash({key: value for key, value in lineage.items() if key != "artifact_hash"})
    gate_status = "PASS" if quality_pass else "BLOCKED"
    gate_body = {"schema_version": S28_GATE_SCHEMA, "gate_id": "stage2.G2.6", "stage": 2, "status": gate_status, "quality_gate_dependency": quality_pass, "measured": quality_metrics, "threshold": {"bootstrap_unit": S28_BOOTSTRAP_UNIT, "parameter_coordinate_pseudoreplication": False, "six_primary_cell_intersection_union": True, "frozen_thresholds": True}, "reasons": [] if quality_pass else ["QUALITY_GATE_FAILED"], "upstream_gate_hashes": {key: value.artifact_hash for key, value in gates.items()}, "artifact_hash": ""}
    gate_body["artifact_hash"] = canonical_json_hash({key: value for key, value in gate_body.items() if key != "artifact_hash"})
    quality_payload = {"schema_version": "stage2-s208-quality-gates-v1", "gate_id": "stage2.G2.6", "status": gate_status, "gates": quality_records, "formal_eligible": quality_pass, "artifact_hash": ""}
    quality_payload["artifact_hash"] = canonical_json_hash({key: value for key, value in quality_payload.items() if key != "artifact_hash"})
    analysis_id = Path(output_root).name if output_root is not None else "in_memory_validation"
    if not analysis_id or analysis_id in {".", ".."}:
        raise S28G26Blocked("ANALYSIS_ID_REQUIRED")
    analysis = {"schema_version": S28_ANALYSIS_SCHEMA, "analysis_id": analysis_id, "status": gate_status, "quality_gates": quality_payload, "hypothesis_decisions": decisions, "confirmatory_family_decisions": family_payload, "statistics_long_table": long_rows, "statistics_summary": summaries, "raw_calibration": raw_calibration_rows, "input_audit": audit, "lineage_manifest": lineage, "g2_6_gate": gate_body}
    output_files: list[str] = []
    if output_root is not None:
        destination = Path(output_root)
        if destination.exists() and any(destination.iterdir()):
            raise S28G26Blocked("OUTPUT_ANALYSIS_DIRECTORY_MUST_BE_NEW")
        destination.mkdir(parents=True, exist_ok=False) if not destination.exists() else None
        for name, value in (("analysis_input_audit.json", audit), ("statistics_long_table.json", {"schema_version": S28_ANALYSIS_SCHEMA, "rows": long_rows, "artifact_hash": canonical_json_hash({"schema_version": S28_ANALYSIS_SCHEMA, "rows": long_rows})}), ("statistics_summary.json", {"schema_version": S28_ANALYSIS_SCHEMA, "rows": summaries, "artifact_hash": canonical_json_hash({"schema_version": S28_ANALYSIS_SCHEMA, "rows": summaries})}), ("raw_calibration.json", {"schema_version": S28_ANALYSIS_SCHEMA, "rows": raw_calibration_rows, "artifact_hash": canonical_json_hash({"schema_version": S28_ANALYSIS_SCHEMA, "rows": raw_calibration_rows})}), ("confirmatory_family_decisions.json", family_payload), ("quality_gates.json", quality_payload), ("hypothesis_decisions.json", decisions), ("lineage_manifest.json", lineage), ("g2.6-gate.json", gate_body)):
            output_files.append(_write_once(destination, name, value))
    analysis["output_files"] = output_files
    analysis["analysis_hash"] = canonical_json_hash(analysis)
    return analysis


__all__ = [
    "S28_ANALYSIS_SCHEMA",
    "S28_BOOTSTRAP_UNIT",
    "S28_CELL_IDS",
    "S28G26Blocked",
    "analyze_s208_g26",
    "two_stage_bootstrap",
]
