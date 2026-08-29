"""Fail-closed storage preflight for the formal S3.07 streaming matrix.

The streaming runner keeps immutable reference/raw/observation/receipt shards,
all content-addressed aggregate snapshots, and a bounded node-gradient cache.
This module contains no runner side effects: it turns exact formal-unit and
measured-manifest metadata into a conservative byte/inode budget and checks
already measured filesystem capacity.  The command-line wrapper lives in
``ops/stage3/preflight_stage3_streaming_capacity.py``.

The formulas are the frozen S3.07 design formulas.  In particular, a path
vector is FP64 (eight bytes per element), the formal matrix is exactly 99
units and 13 candidate rules, and a resume may deduct only explicitly
hash-bound durable units.  Existing append-only aggregate snapshots are never
deducted: they remain on disk for replay and audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any
from pathlib import PurePosixPath

from ..capacity import StorageBudget
from ..contracts.jsonio import canonical_json_hash, load_canonical_json
from .stage3_protocol import DEFAULT_CANDIDATE_RULES


STAGE3_STREAMING_CAPACITY_SCHEMA = "stage3-streaming-capacity-preflight-v1"
STAGE3_STREAMING_METADATA_SCHEMA = "stage3-streaming-metadata-upper-bound-v1"
STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA = "stage3-streaming-parameter-counts-v1"
STAGE3_STREAMING_RESUME_SCHEMA = "stage3-streaming-resume-manifest-v1"
STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA = "stage3-streaming-capacity-launch-spec-v1"
FORMAL_UNIT_COUNT = 99
FORMAL_CANDIDATE_COUNT = 13
VECTOR_BYTES = 8
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[^\\?]+$")
_SNAPSHOT_KINDS = ("reference", "raw", "stream")
_MODEL_NAMES = frozenset({"14M", "31M"})


class StreamingCapacityError(ValueError):
    """An input or measurement cannot be accepted for formal preflight."""


def _fail(code: str, detail: object | None = None) -> StreamingCapacityError:
    if detail is None:
        return StreamingCapacityError(code)
    return StreamingCapacityError(f"{code}:{detail}")


def _int(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail("INTEGER_INVALID", field)
    return value


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise _fail("HASH_INVALID", field)
    return value


def _ref(value: object, *, field: str) -> str:
    # All operational references are logical, workspace-relative POSIX paths.
    # Reject path traversal and platform-specific absolute/drive forms before
    # callers turn them into filesystem paths.
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not _REF_RE.fullmatch(value)
        or "//" in value
        or value.startswith("/")
        or "\\" in value
        or ":" in value
        or "fixture" in value.casefold()
        or "synthetic" in value.casefold()
    ):
        raise _fail("REFERENCE_INVALID", field)
    logical = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in logical.parts):
        raise _fail("REFERENCE_INVALID", field)
    return value


def _checked_hash_bound(value: Mapping[str, object], *, field: str) -> str:
    supplied = _hash(value.get("artifact_hash"), field=f"{field}.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(body) != supplied:
        raise _fail("ARTIFACT_HASH_MISMATCH", field)
    return supplied


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fail("MAPPING_REQUIRED", field)
    return value


def _unit_ids(value: object, *, field: str, expected: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("UNIT_IDS_INVALID", field)
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise _fail("UNIT_ID_INVALID", field)
    if len(set(result)) != len(result):
        raise _fail("UNIT_ID_DUPLICATE", field)
    if expected is not None and result != expected:
        raise _fail("UNIT_ID_SET_MISMATCH", field)
    return result


def _metadata_counter(value: object, *, field: str) -> tuple[int, int]:
    item = _mapping(value, field=field)
    if set(item) != {"bytes", "inode_count"}:
        raise _fail("METADATA_COUNTER_FIELDS_INVALID", field)
    return (
        _int(item["bytes"], field=f"{field}.bytes"),
        _int(item["inode_count"], field=f"{field}.inode_count"),
    )


def _snapshot_rows(value: object, *, field: str) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("SNAPSHOT_ROWS_INVALID", field)
    rows: list[tuple[int, int, int]] = []
    for ordinal, raw in enumerate(value, start=1):
        item = _mapping(raw, field=f"{field}[{ordinal - 1}]")
        if set(item) != {"unit_ordinal", "bytes", "inode_count"}:
            raise _fail("SNAPSHOT_ROW_FIELDS_INVALID", field)
        if _int(item["unit_ordinal"], field=f"{field}[{ordinal - 1}].unit_ordinal", minimum=1) != ordinal:
            raise _fail("SNAPSHOT_ORDINAL_INVALID", field)
        rows.append(
            (
                ordinal,
                _int(item["bytes"], field=f"{field}[{ordinal - 1}].bytes"),
                _int(item["inode_count"], field=f"{field}[{ordinal - 1}].inode_count"),
            )
        )
    if len(rows) != FORMAL_UNIT_COUNT:
        raise _fail("SNAPSHOT_COUNT_INVALID", field)
    return tuple(rows)


def _cache_key_digests(value: object, *, field: str) -> frozenset[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _fail("CACHE_KEY_SET_INVALID", field)
    values = tuple(_hash(item, field=f"{field}[{index}]") for index, item in enumerate(value))
    if len(set(values)) != len(values):
        raise _fail("CACHE_KEY_DUPLICATE", field)
    if not values:
        raise _fail("CACHE_KEY_SET_EMPTY", field)
    return frozenset(values)


@dataclass(frozen=True, slots=True)
class ModelStreamingCapacity:
    model: str
    unit_count: int
    parameter_elements: int
    reference_completed_level_count: int
    cache_key_count: int
    unit_vector_bytes: int
    unit_metadata_bytes: int
    unit_metadata_inodes: int
    cache_peak_bytes: int
    cache_peak_inodes: int

    @property
    def unit_retained_bytes(self) -> int:
        return self.unit_vector_bytes + self.unit_metadata_bytes

    def to_dict(self) -> dict[str, int | str]:
        return {
            "model": self.model,
            "unit_count": self.unit_count,
            "parameter_elements": self.parameter_elements,
            "reference_completed_level_count": self.reference_completed_level_count,
            "cache_key_count": self.cache_key_count,
            "unit_vector_bytes": self.unit_vector_bytes,
            "unit_metadata_bytes": self.unit_metadata_bytes,
            "unit_metadata_inodes": self.unit_metadata_inodes,
            "cache_peak_bytes": self.cache_peak_bytes,
            "cache_peak_inodes": self.cache_peak_inodes,
            "unit_retained_bytes": self.unit_retained_bytes,
        }


@dataclass(frozen=True, slots=True)
class Stage3StreamingCapacityEstimate:
    unit_count: int
    candidate_count: int
    candidate_rule_names: tuple[str, ...]
    durable_unit_count: int
    pending_unit_count: int
    retained_vector_bytes: int
    retained_unit_metadata_bytes: int
    aggregate_snapshot_bytes: int
    aggregate_snapshot_inodes: int
    active_cache_peak_bytes: int
    active_cache_peak_inodes: int
    fixed_manifest_bytes: int
    fixed_manifest_inodes: int
    inflight_bytes: int
    inflight_inodes: int
    temporary_json_bytes: int
    temporary_json_inodes: int
    expected_output_bytes: int
    expected_output_inodes: int
    expected_cache_bytes: int
    expected_cache_inodes: int
    expected_new_bytes: int
    expected_new_inodes: int
    failure_residue_bytes: int
    failure_residue_inodes: int
    total_budget: StorageBudget
    output_budget: StorageBudget
    cache_budget: StorageBudget
    models: tuple[ModelStreamingCapacity, ...]

    @property
    def safety_margin_bytes(self) -> int:
        return self.total_budget.safety_margin_bytes

    @property
    def required_free_bytes(self) -> int:
        return self.total_budget.required_free_bytes

    @property
    def inode_margin(self) -> int:
        return max(1, math.ceil(self.expected_new_inodes * 0.20))

    @property
    def required_free_inodes(self) -> int:
        return self.expected_new_inodes + self.inode_margin

    @staticmethod
    def _required_inodes(expected: int) -> int:
        return expected + max(1, math.ceil(expected * 0.20))

    @property
    def required_output_free_inodes(self) -> int:
        return self._required_inodes(self.expected_output_inodes)

    @property
    def required_cache_free_inodes(self) -> int:
        return self._required_inodes(self.expected_cache_inodes)

    def to_dict(self) -> dict[str, object]:
        return {
            "unit_count": self.unit_count,
            "candidate_count": self.candidate_count,
            "candidate_rule_names": list(self.candidate_rule_names),
            "durable_unit_count": self.durable_unit_count,
            "pending_unit_count": self.pending_unit_count,
            "retained_vector_bytes": self.retained_vector_bytes,
            "retained_unit_metadata_bytes": self.retained_unit_metadata_bytes,
            "aggregate_snapshot_bytes": self.aggregate_snapshot_bytes,
            "aggregate_snapshot_inodes": self.aggregate_snapshot_inodes,
            "active_cache_peak_bytes": self.active_cache_peak_bytes,
            "active_cache_peak_inodes": self.active_cache_peak_inodes,
            "fixed_manifest_bytes": self.fixed_manifest_bytes,
            "fixed_manifest_inodes": self.fixed_manifest_inodes,
            "inflight_bytes": self.inflight_bytes,
            "inflight_inodes": self.inflight_inodes,
            "temporary_json_bytes": self.temporary_json_bytes,
            "temporary_json_inodes": self.temporary_json_inodes,
            "expected_output_bytes": self.expected_output_bytes,
            "expected_output_inodes": self.expected_output_inodes,
            "expected_cache_bytes": self.expected_cache_bytes,
            "expected_cache_inodes": self.expected_cache_inodes,
            "expected_new_bytes": self.expected_new_bytes,
            "expected_new_inodes": self.expected_new_inodes,
            "safety_margin_bytes": self.safety_margin_bytes,
            "required_free_bytes": self.required_free_bytes,
            "inode_margin": self.inode_margin,
            "required_free_inodes": self.required_free_inodes,
            "required_output_free_inodes": self.required_output_free_inodes,
            "required_cache_free_inodes": self.required_cache_free_inodes,
            "failure_residue_bytes": self.failure_residue_bytes,
            "failure_residue_inodes": self.failure_residue_inodes,
            "budgets": {
                "total": self.total_budget.as_dict(),
                "output": self.output_budget.as_dict(),
                "cache": self.cache_budget.as_dict(),
            },
            "models": [item.to_dict() for item in self.models],
        }


@dataclass(frozen=True, slots=True)
class FilesystemCapacityCheck:
    name: str
    path: str
    free_bytes: int | None
    free_inodes: int | None
    required_free_bytes: int
    required_free_inodes: int
    bytes_ok: bool
    inodes_ok: bool
    ok: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "path": self.path,
            "free_bytes": self.free_bytes,
            "free_inodes": self.free_inodes,
            "required_free_bytes": self.required_free_bytes,
            "required_free_inodes": self.required_free_inodes,
            "bytes_ok": self.bytes_ok,
            "inodes_ok": self.inodes_ok,
            "ok": self.ok,
            "reason": self.reason,
        }


def _parse_model_metadata(
    model: str,
    raw: Mapping[str, object],
    *,
    expected_level_count: int | None = None,
) -> tuple[int, int, int, int, frozenset[str]]:
    """Return level count, unit metadata bytes/inodes, cache bytes/inodes, keys."""

    expected = {"reference_completed_level_count", "single_unit", "node_cache"}
    if set(raw) != expected:
        raise _fail("MODEL_METADATA_FIELDS_INVALID", model)
    levels = _int(
        raw["reference_completed_level_count"],
        field=f"models.{model}.reference_completed_level_count",
        minimum=1,
    )
    if expected_level_count is not None and levels != expected_level_count:
        raise _fail("REFERENCE_LEVEL_COUNT_MISMATCH", model)
    single = _mapping(raw["single_unit"], field=f"models.{model}.single_unit")
    if set(single) != {"reference_bytes", "observation_bytes", "raw_bytes", "receipt_bytes", "inode_count"}:
        raise _fail("SINGLE_UNIT_METADATA_FIELDS_INVALID", model)
    component_bytes = sum(
        _int(single[name], field=f"models.{model}.single_unit.{name}")
        for name in ("reference_bytes", "observation_bytes", "raw_bytes", "receipt_bytes")
    )
    component_inodes = _int(single["inode_count"], field=f"models.{model}.single_unit.inode_count")
    if component_inodes <= 0:
        raise _fail("SINGLE_UNIT_INODE_COUNT_INVALID", model)

    cache = _mapping(raw["node_cache"], field=f"models.{model}.node_cache")
    expected_cache = {
        "reference_level_cache_key_digests",
        "candidate_cache_key_digests",
        "object_bytes",
        "inode_count",
    }
    if set(cache) != expected_cache:
        raise _fail("NODE_CACHE_METADATA_FIELDS_INVALID", model)
    reference_keys = _cache_key_digests(
        cache["reference_level_cache_key_digests"],
        field=f"models.{model}.node_cache.reference_level_cache_key_digests",
    )
    if len(reference_keys) < levels:
        raise _fail("REFERENCE_CACHE_KEY_COUNT_BELOW_LEVEL_COUNT", model)
    candidate = cache["candidate_cache_key_digests"]
    if not isinstance(candidate, Mapping) or set(candidate) != set(DEFAULT_CANDIDATE_RULES):
        raise _fail("CANDIDATE_CACHE_RULE_SET_INVALID", model)
    candidate_keys: set[str] = set()
    for rule in DEFAULT_CANDIDATE_RULES:
        candidate_keys.update(
            _cache_key_digests(
                candidate[rule],
                field=f"models.{model}.node_cache.candidate_cache_key_digests.{rule}",
            )
        )
    keys = frozenset(reference_keys | candidate_keys)
    object_bytes = _int(cache["object_bytes"], field=f"models.{model}.node_cache.object_bytes")
    object_inodes = _int(cache["inode_count"], field=f"models.{model}.node_cache.inode_count")
    if object_inodes <= 0:
        raise _fail("NODE_CACHE_INODE_COUNT_INVALID", model)
    return levels, component_bytes, component_inodes, object_bytes, keys | frozenset()


def estimate_stage3_streaming_capacity(
    *,
    unit_model_by_id: Mapping[str, str],
    parameter_elements_by_model: Mapping[str, int],
    metadata_by_model: Mapping[str, Mapping[str, object]],
    aggregate_snapshots: Mapping[str, Sequence[Mapping[str, object]]],
    fixed_manifests: Mapping[str, object],
    inflight: Mapping[str, object],
    temporary_json: Mapping[str, object],
    durable_unit_ids: Sequence[str] = (),
    candidate_rule_names: Sequence[str] = DEFAULT_CANDIDATE_RULES,
) -> Stage3StreamingCapacityEstimate:
    """Compute the exact formal streaming byte/inode upper bound.

    ``unit_model_by_id`` must come from the validated formal production index.
    ``metadata_by_model`` is measured single-unit metadata; it is not a
    synthetic per-parameter guess.  Aggregate rows are the measured/declared
    upper bound for each immutable snapshot after unit ordinal 1..99.
    """

    units = tuple(unit_model_by_id)
    if len(units) != FORMAL_UNIT_COUNT or len(set(units)) != FORMAL_UNIT_COUNT:
        raise _fail("FORMAL_UNIT_COUNT_INVALID")
    if set(parameter_elements_by_model) != _MODEL_NAMES or set(metadata_by_model) != _MODEL_NAMES:
        raise _fail("MODEL_INPUT_SET_INVALID")
    rules = tuple(candidate_rule_names)
    if len(rules) != FORMAL_CANDIDATE_COUNT or rules != DEFAULT_CANDIDATE_RULES:
        raise _fail("FORMAL_CANDIDATE_SET_INVALID")
    model_counts: dict[str, int] = {}
    for unit_id, model in unit_model_by_id.items():
        if model not in _MODEL_NAMES:
            raise _fail("UNIT_MODEL_INVALID", unit_id)
        model_counts[model] = model_counts.get(model, 0) + 1
    if model_counts != {"14M": 72, "31M": 27}:
        raise _fail("FORMAL_MODEL_UNIT_COVERAGE_INVALID", model_counts)

    durable = tuple(durable_unit_ids)
    if len(set(durable)) != len(durable) or any(item not in unit_model_by_id for item in durable):
        raise _fail("DURABLE_UNIT_SET_INVALID")
    durable_set = set(durable)
    model_rows: list[ModelStreamingCapacity] = []
    parsed: dict[str, tuple[int, int, int, int, frozenset[str]]] = {}
    for model in sorted(_MODEL_NAMES):
        elements = _int(parameter_elements_by_model[model], field=f"parameter_elements.{model}", minimum=1)
        parsed[model] = _parse_model_metadata(model, metadata_by_model[model])
        levels, unit_meta_bytes, unit_meta_inodes, cache_bytes, keys = parsed[model]
        vector_bytes = VECTOR_BYTES * elements * (FORMAL_CANDIDATE_COUNT + 2 + levels)
        cache_peak = VECTOR_BYTES * elements * len(keys) + cache_bytes
        cache_inodes = _int(
            _mapping(metadata_by_model[model]["node_cache"], field=f"models.{model}.node_cache")["inode_count"],
            field=f"models.{model}.node_cache.inode_count",
            minimum=1,
        )
        model_rows.append(
            ModelStreamingCapacity(
                model=model,
                unit_count=model_counts[model],
                parameter_elements=elements,
                reference_completed_level_count=levels,
                cache_key_count=len(keys),
                unit_vector_bytes=vector_bytes,
                unit_metadata_bytes=unit_meta_bytes,
                unit_metadata_inodes=unit_meta_inodes,
                cache_peak_bytes=cache_peak,
                cache_peak_inodes=cache_inodes,
            )
        )

    if set(aggregate_snapshots) != set(_SNAPSHOT_KINDS):
        raise _fail("AGGREGATE_SNAPSHOT_KIND_SET_INVALID")
    snapshot_rows = {kind: _snapshot_rows(aggregate_snapshots[kind], field=f"aggregate_snapshots.{kind}") for kind in _SNAPSHOT_KINDS}
    aggregate_bytes = sum(row[1] for rows in snapshot_rows.values() for row in rows)
    aggregate_inodes = sum(row[2] for rows in snapshot_rows.values() for row in rows)

    fixed_bytes, fixed_inodes = _metadata_counter(fixed_manifests, field="fixed_manifests")
    inflight_bytes, inflight_inodes = _metadata_counter(inflight, field="inflight")
    temporary_bytes, temporary_inodes = _metadata_counter(temporary_json, field="temporary_json")

    # Keep the model arithmetic explicit: a durable unit removes only that
    # unit's vectors and four measured metadata files.  Snapshot bytes are
    # intentionally untouched on resume.
    pending_counts = {model: model_counts[model] - sum(unit_model_by_id[item] == model for item in durable_set) for model in _MODEL_NAMES}
    retained_vectors = sum(row.unit_vector_bytes * pending_counts[row.model] for row in model_rows)
    retained_unit_metadata = sum(row.unit_metadata_bytes * pending_counts[row.model] for row in model_rows)
    retained_unit_inodes = sum(row.unit_metadata_inodes * pending_counts[row.model] for row in model_rows)
    cache_peak_row = max(model_rows, key=lambda row: row.cache_peak_bytes)
    active_cache_bytes = cache_peak_row.cache_peak_bytes
    active_cache_inodes = cache_peak_row.cache_peak_inodes
    expected_output_bytes = retained_vectors + retained_unit_metadata + aggregate_bytes + fixed_bytes
    expected_output_inodes = retained_unit_inodes + aggregate_inodes + fixed_inodes
    expected_cache_bytes = active_cache_bytes
    expected_cache_inodes = active_cache_inodes
    expected_new_bytes = expected_output_bytes + expected_cache_bytes
    expected_new_inodes = expected_output_inodes + expected_cache_inodes

    max_snapshot_bytes = max(
        sum(snapshot_rows[kind][ordinal - 1][1] for kind in _SNAPSHOT_KINDS)
        for ordinal in range(1, FORMAL_UNIT_COUNT + 1)
    )
    max_snapshot_inodes = max(
        sum(snapshot_rows[kind][ordinal - 1][2] for kind in _SNAPSHOT_KINDS)
        for ordinal in range(1, FORMAL_UNIT_COUNT + 1)
    )
    max_unit_row = max(model_rows, key=lambda row: row.unit_retained_bytes)
    failure_residue_bytes = active_cache_bytes + inflight_bytes + max_unit_row.unit_retained_bytes + max_snapshot_bytes + temporary_bytes
    failure_residue_inodes = active_cache_inodes + inflight_inodes + max_unit_row.unit_metadata_inodes + max_snapshot_inodes + temporary_inodes

    total_budget = StorageBudget.from_expected("stage3-streaming-total", expected_new_bytes)
    output_budget = StorageBudget.from_expected("stage3-streaming-output", expected_output_bytes)
    cache_budget = StorageBudget.from_expected("stage3-streaming-cache", expected_cache_bytes)
    return Stage3StreamingCapacityEstimate(
        unit_count=FORMAL_UNIT_COUNT,
        candidate_count=FORMAL_CANDIDATE_COUNT,
        candidate_rule_names=rules,
        durable_unit_count=len(durable_set),
        pending_unit_count=FORMAL_UNIT_COUNT - len(durable_set),
        retained_vector_bytes=retained_vectors,
        retained_unit_metadata_bytes=retained_unit_metadata,
        aggregate_snapshot_bytes=aggregate_bytes,
        aggregate_snapshot_inodes=aggregate_inodes,
        active_cache_peak_bytes=active_cache_bytes,
        active_cache_peak_inodes=active_cache_inodes,
        fixed_manifest_bytes=fixed_bytes,
        fixed_manifest_inodes=fixed_inodes,
        inflight_bytes=inflight_bytes,
        inflight_inodes=inflight_inodes,
        temporary_json_bytes=temporary_bytes,
        temporary_json_inodes=temporary_inodes,
        expected_output_bytes=expected_output_bytes,
        expected_output_inodes=expected_output_inodes,
        expected_cache_bytes=expected_cache_bytes,
        expected_cache_inodes=expected_cache_inodes,
        expected_new_bytes=expected_new_bytes,
        expected_new_inodes=expected_new_inodes,
        failure_residue_bytes=failure_residue_bytes,
        failure_residue_inodes=failure_residue_inodes,
        total_budget=total_budget,
        output_budget=output_budget,
        cache_budget=cache_budget,
        models=tuple(model_rows),
    )


def check_filesystem_capacity(
    *,
    name: str,
    path: str | Path,
    budget: StorageBudget,
    required_free_inodes: int,
    free_bytes: int | None = None,
    free_inodes: int | None = None,
) -> FilesystemCapacityCheck:
    """Measure one filesystem; missing inode information is BLOCKED."""

    target = Path(path)
    reason: str | None = None
    if free_bytes is None:
        try:
            free_bytes = int(shutil.disk_usage(target).free)
        except (OSError, ValueError) as error:
            reason = f"FILESYSTEM_BYTES_UNAVAILABLE:{type(error).__name__}"
    if free_inodes is None and reason is None:
        try:
            if not hasattr(os, "statvfs"):
                raise OSError("statvfs unavailable")
            free_inodes = int(os.statvfs(target).f_favail)
        except (OSError, ValueError, AttributeError) as error:
            reason = f"FILESYSTEM_INODES_UNAVAILABLE:{type(error).__name__}"
    bytes_ok = reason is None and free_bytes is not None and free_bytes >= budget.required_free_bytes
    inodes_ok = reason is None and free_inodes is not None and free_inodes >= required_free_inodes
    if reason is None and not bytes_ok:
        reason = "FILESYSTEM_FREE_BYTES_INSUFFICIENT"
    if reason is None and not inodes_ok:
        reason = "FILESYSTEM_FREE_INODES_INSUFFICIENT"
    return FilesystemCapacityCheck(
        name=name,
        path=str(target),
        free_bytes=free_bytes,
        free_inodes=free_inodes,
        required_free_bytes=budget.required_free_bytes,
        required_free_inodes=required_free_inodes,
        bytes_ok=bool(bytes_ok),
        inodes_ok=bool(inodes_ok),
        ok=bool(bytes_ok and inodes_ok),
        reason=reason,
    )


def build_capacity_preflight_report(
    *,
    estimate: Stage3StreamingCapacityEstimate,
    output_check: FilesystemCapacityCheck,
    cache_check: FilesystemCapacityCheck,
    formal_plan_ref: str,
    formal_plan_hash: str,
    production_index_ref: str,
    production_index_hash: str,
    parameter_counts_ref: str,
    parameter_counts_hash: str = "0" * 64,
    metadata_ref: str,
    metadata_hash: str = "0" * 64,
    resume_ref: str | None = None,
    resume_hash: str | None = None,
) -> dict[str, object]:
    """Build the canonical PASS/BLOCKED machine-readable preflight result."""

    status = "PASS" if output_check.ok and cache_check.ok else "BLOCKED"
    payload: dict[str, object] = {
        "schema_version": STAGE3_STREAMING_CAPACITY_SCHEMA,
        "status": status,
        "scope": "formal",
        "formal_eligible": status == "PASS",
        "formal_plan_ref": _ref(formal_plan_ref, field="formal_plan_ref"),
        "formal_plan_hash": _hash(formal_plan_hash, field="formal_plan_hash"),
        "production_index_ref": _ref(production_index_ref, field="production_index_ref"),
        "production_index_hash": _hash(production_index_hash, field="production_index_hash"),
        "parameter_counts_ref": _ref(parameter_counts_ref, field="parameter_counts_ref"),
        "parameter_counts_hash": _hash(parameter_counts_hash, field="parameter_counts_hash"),
        "metadata_ref": _ref(metadata_ref, field="metadata_ref"),
        "metadata_hash": _hash(metadata_hash, field="metadata_hash"),
        "resume_ref": None if resume_ref is None else _ref(resume_ref, field="resume_ref"),
        "resume_hash": None if resume_hash is None else _hash(resume_hash, field="resume_hash"),
        "estimate": estimate.to_dict(),
        "filesystems": {
            "output": output_check.to_dict(),
            "cache": cache_check.to_dict(),
        },
        "reasons": [
            item
            for item in (output_check.reason, cache_check.reason)
            if item is not None
        ],
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def load_canonical_mapping(path: str | Path) -> Mapping[str, object]:
    """Read a canonical JSON object for the read-only preflight CLI."""

    try:
        value = load_canonical_json(path)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("CANONICAL_INPUT_LOAD_FAILED", path) from error
    return _mapping(value, field=str(path))


def validate_streaming_launch_spec(value: Mapping[str, object]) -> Mapping[str, object]:
    """Validate the immutable environment-bound S3.07 capacity launch spec.

    The fan-out process must not synthesize preflight inputs from its own
    environment.  Every input path and binding is therefore frozen in this
    exact-field, canonical-hash-bound document.  ``None`` resume fields are
    deliberate: a fresh matrix has no durable-unit manifest, while a resumed
    matrix binds one explicitly.
    """

    required = {
        "schema_version",
        "formal_plan_ref",
        "formal_plan_hash",
        "production_index_ref",
        "production_index_hash",
        "parameter_counts_ref",
        "parameter_counts_hash",
        "metadata_ref",
        "metadata_hash",
        "output_filesystem_ref",
        "cache_filesystem_ref",
        "candidate_rule_names",
        "resume_manifest_ref",
        "resume_manifest_hash",
        "artifact_hash",
    }
    if set(value) != required or value.get("schema_version") != STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA:
        raise _fail("LAUNCH_SPEC_FIELDS_INVALID")
    _checked_hash_bound(value, field="launch_spec")
    for field in (
        "formal_plan_ref",
        "production_index_ref",
        "parameter_counts_ref",
        "metadata_ref",
        "output_filesystem_ref",
        "cache_filesystem_ref",
    ):
        _ref(value[field], field=f"launch_spec.{field}")
    for field in (
        "formal_plan_hash",
        "production_index_hash",
        "parameter_counts_hash",
        "metadata_hash",
    ):
        _hash(value[field], field=f"launch_spec.{field}")
    resume_ref = value["resume_manifest_ref"]
    resume_hash = value["resume_manifest_hash"]
    if resume_ref is None or resume_hash is None:
        if resume_ref is not None or resume_hash is not None:
            raise _fail("LAUNCH_SPEC_RESUME_BINDING_INVALID")
    else:
        _ref(resume_ref, field="launch_spec.resume_manifest_ref")
        _hash(resume_hash, field="launch_spec.resume_manifest_hash")
    rules = value["candidate_rule_names"]
    if not isinstance(rules, Sequence) or isinstance(rules, (str, bytes)) or tuple(rules) != DEFAULT_CANDIDATE_RULES:
        raise _fail("LAUNCH_SPEC_CANDIDATE_SET_INVALID")
    return value


def load_streaming_launch_spec(root: str | Path, reference: str) -> Mapping[str, object]:
    """Safely reload an environment-bound launch spec beneath ``root``."""

    root_path = Path(root).resolve()
    logical = _ref(reference, field="launch_spec_ref")
    target = (root_path / PurePosixPath(logical)).resolve()
    try:
        target.relative_to(root_path)
    except ValueError as error:
        raise _fail("LAUNCH_SPEC_OUTSIDE_WORKSPACE") from error
    if not target.is_file():
        raise _fail("LAUNCH_SPEC_NOT_FILE", logical)
    return validate_streaming_launch_spec(load_canonical_mapping(target))


__all__ = [
    "FORMAL_CANDIDATE_COUNT",
    "FORMAL_UNIT_COUNT",
    "FilesystemCapacityCheck",
    "ModelStreamingCapacity",
    "STAGE3_STREAMING_CAPACITY_SCHEMA",
    "STAGE3_STREAMING_METADATA_SCHEMA",
    "STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA",
    "STAGE3_STREAMING_RESUME_SCHEMA",
    "STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA",
    "Stage3StreamingCapacityEstimate",
    "StreamingCapacityError",
    "build_capacity_preflight_report",
    "check_filesystem_capacity",
    "estimate_stage3_streaming_capacity",
    "load_canonical_mapping",
    "load_streaming_launch_spec",
    "validate_streaming_launch_spec",
]
