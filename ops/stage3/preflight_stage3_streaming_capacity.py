#!/usr/bin/env python3
"""Read-only CLI for the formal S3.07 streaming storage preflight.

All source documents must be canonical, hash-bound JSON.  The command does
not create directories, launch work, or modify DATA_ROOT unless the caller
explicitly supplies ``--output``; in that case only that requested evidence
file is written.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

# Keep the repository's ``src/`` layout directly executable for the operational
# command, while remaining a no-op when the package is already installed.
if __package__ in {None, ""}:
    _source_root = Path(__file__).resolve().parents[2] / "src"
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage3_production_plan import load_production_unit_index
from param_importance_nlp.experiments.stage3_protocol import DEFAULT_CANDIDATE_RULES
from param_importance_nlp.experiments.stage3_streaming_capacity import (
    STAGE3_STREAMING_CAPACITY_SCHEMA,
    STAGE3_STREAMING_METADATA_SCHEMA,
    STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA,
    STAGE3_STREAMING_RESUME_SCHEMA,
    STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA,
    StreamingCapacityError,
    _checked_hash_bound,
    _hash,
    _HASH_RE,
    _mapping,
    _ref,
    build_capacity_preflight_report,
    check_filesystem_capacity,
    estimate_stage3_streaming_capacity,
    load_canonical_mapping,
    load_streaming_launch_spec,
    validate_streaming_launch_spec,
)


def _logical_ref(path: Path, root: Path, *, field: str) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise StreamingCapacityError(f"{field.upper()}_OUTSIDE_WORKSPACE") from error


def _same_filesystem(left: Path, right: Path) -> bool | None:
    """Return whether output/cache share a device; unknown is fail-closed."""

    if left.resolve() == right.resolve():
        return True
    try:
        return os.stat(left).st_dev == os.stat(right).st_dev
    except OSError:
        return None


def _safe_bound_document(
    root: Path,
    reference: object,
    expected_hash: object,
    *,
    field: str,
) -> Mapping[str, object]:
    """Reload one hash-bound canonical document beneath ``root``."""

    logical = _ref(reference, field=f"{field}.ref")
    expected = _hash(expected_hash, field=f"{field}.hash")
    target = (root / Path(logical)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise StreamingCapacityError(f"{field.upper()}_OUTSIDE_WORKSPACE") from error
    if not target.is_file():
        raise StreamingCapacityError(f"{field.upper()}_NOT_FILE")
    value = load_canonical_mapping(target)
    observed = _checked_hash_bound(value, field=field)
    if observed != expected:
        raise StreamingCapacityError(f"{field.upper()}_HASH_MISMATCH")
    return value


def _source_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = value.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return value


def _model_alias_matches(value: object, model: str) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("_", "-")
    token = model.casefold()
    return normalized == token or token in normalized or normalized.endswith(token)


def _recomputed_parameter_elements(value: Mapping[str, object], *, model: str, field: str) -> int:
    """Extract a real registry/checkpoint count, rejecting ambiguous data.

    Stage 2 registry commits generally carry a source manifest under ``payload``
    while endpoint commits carry a checkpoint/model count directly.  This
    deliberately accepts only semantically named count fields and requires a
    model identity somewhere in the loaded artifact; an arbitrary integer in
    a forged JSON file is not a parameter count.
    """

    count_keys = {
        "parameter_elements",
        "parameter_element_count",
        "parameter_count",
        "trainable_parameter_count",
        "eligible_numel",
        "total_parameter_elements",
        "total_numel",
    }
    identity_keys = {"model", "model_id", "model_name", "architecture", "logical_name"}
    counts: list[int] = []
    identities: list[object] = []

    def visit(node: object) -> None:
        if isinstance(node, Mapping):
            for key, item in node.items():
                if key in identity_keys:
                    identities.append(item)
                if key in count_keys and isinstance(item, int) and not isinstance(item, bool) and item > 0:
                    counts.append(item)
                visit(item)
        elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
            for item in node:
                visit(item)

    visit(value)
    if not identities or not any(_model_alias_matches(item, model) for item in identities):
        raise StreamingCapacityError(f"{field.upper()}_MODEL_IDENTITY_MISSING:{model}")
    unique = set(counts)
    if not unique:
        raise StreamingCapacityError(f"{field.upper()}_PARAMETER_COUNT_MISSING:{model}")
    if len(unique) != 1:
        raise StreamingCapacityError(f"{field.upper()}_PARAMETER_COUNT_AMBIGUOUS:{model}")
    return next(iter(unique))


def _load_source_count(
    root: Path,
    value: Mapping[str, object],
    *,
    model: str,
    field: str,
) -> int:
    """Reload a source registry/checkpoint and recompute its element count."""

    source = _source_mapping(value)
    try:
        return _recomputed_parameter_elements(source, model=model, field=field)
    except StreamingCapacityError as primary:
        # A formal parameter-registry task commit references its immutable S2
        # source manifest by raw SHA-256 rather than artifact_hash.  Reload it
        # as well, so the count cannot be reduced by editing only the wrapper.
        source_ref = source.get("source_s203_manifest_ref")
        source_sha = source.get("source_s203_manifest_sha256")
        if not isinstance(source_ref, str) or not isinstance(source_sha, str):
            raise primary
        logical = _ref(source_ref, field=f"{field}.source_s203_manifest_ref")
        target = (root / Path(logical)).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as error:
            raise StreamingCapacityError(f"{field.upper()}_SOURCE_OUTSIDE_WORKSPACE") from error
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != source_sha:
            raise StreamingCapacityError(f"{field.upper()}_SOURCE_HASH_MISMATCH") from primary
        loaded = load_canonical_mapping(target)
        return _recomputed_parameter_elements(loaded, model=model, field=field)


def _parameter_elements(
    value: Mapping[str, object],
    *,
    root: Path,
    formal_plan_hash: str,
    production_index_hash: str,
    candidate_rule_names: Sequence[str],
) -> dict[str, int]:
    if set(value) != {
        "schema_version",
        "formal_eligible",
        "formal_plan_hash",
        "production_index_hash",
        "candidate_rule_names",
        "models",
        "artifact_hash",
    } or value.get("schema_version") != STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA:
        raise StreamingCapacityError("PARAMETER_COUNTS_FIELDS_INVALID")
    if value.get("formal_eligible") is not True:
        raise StreamingCapacityError("PARAMETER_COUNTS_FORMAL_REQUIRED")
    _checked_hash_bound(value, field="parameter_counts")
    if value.get("formal_plan_hash") != formal_plan_hash or value.get("production_index_hash") != production_index_hash:
        raise StreamingCapacityError("PARAMETER_COUNTS_PLAN_INDEX_BINDING_INVALID")
    if value.get("candidate_rule_names") != list(candidate_rule_names):
        raise StreamingCapacityError("PARAMETER_COUNTS_CANDIDATE_BINDING_INVALID")
    models = value.get("models")
    if not isinstance(models, Mapping) or set(models) != {"14M", "31M"}:
        raise StreamingCapacityError("PARAMETER_COUNTS_MODEL_SET_INVALID")
    result: dict[str, int] = {}
    for model in ("14M", "31M"):
        item = models[model]
        expected_fields = {
            "parameter_elements",
            "parameter_registry_ref",
            "parameter_registry_hash",
            "endpoint_ref",
            "endpoint_hash",
        }
        if not isinstance(item, Mapping) or set(item) != expected_fields:
            raise StreamingCapacityError(f"PARAMETER_ELEMENTS_FIELDS_INVALID:{model}")
        raw = item["parameter_elements"]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise StreamingCapacityError(f"PARAMETER_ELEMENTS_INVALID:{model}")
        registry = _safe_bound_document(
            root,
            item["parameter_registry_ref"],
            item["parameter_registry_hash"],
            field=f"parameter_registry.{model}",
        )
        endpoint = _safe_bound_document(
            root,
            item["endpoint_ref"],
            item["endpoint_hash"],
            field=f"endpoint.{model}",
        )
        registry_count = _load_source_count(root, registry, model=model, field=f"parameter_registry.{model}")
        endpoint_count = _load_source_count(root, endpoint, model=model, field=f"endpoint.{model}")
        if registry_count != raw or endpoint_count != raw:
            raise StreamingCapacityError(f"PARAMETER_ELEMENTS_SOURCE_COUNT_MISMATCH:{model}")
        result[model] = raw
    return result


def _metadata_parts(
    value: Mapping[str, object],
    *,
    formal_plan_hash: str,
    production_index_hash: str,
    parameter_counts_hash: str,
    candidate_rule_names: Sequence[str],
) -> tuple[
    Mapping[str, Mapping[str, object]],
    Mapping[str, Sequence[Mapping[str, object]]],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    required = {
        "schema_version",
        "formal_eligible",
        "formal_plan_hash",
        "production_index_hash",
        "parameter_counts_hash",
        "candidate_rule_names",
        "models",
        "aggregate_snapshots",
        "fixed_manifests",
        "inflight",
        "temporary_json",
        "artifact_hash",
    }
    if set(value) != required or value.get("schema_version") != STAGE3_STREAMING_METADATA_SCHEMA:
        raise StreamingCapacityError("METADATA_FIELDS_INVALID")
    if value.get("formal_eligible") is not True:
        raise StreamingCapacityError("METADATA_FORMAL_REQUIRED")
    _checked_hash_bound(value, field="streaming_metadata")
    if (
        value.get("formal_plan_hash") != formal_plan_hash
        or value.get("production_index_hash") != production_index_hash
        or value.get("parameter_counts_hash") != parameter_counts_hash
    ):
        raise StreamingCapacityError("METADATA_PLAN_INDEX_COUNTS_BINDING_INVALID")
    if value.get("candidate_rule_names") != list(candidate_rule_names):
        raise StreamingCapacityError("METADATA_CANDIDATE_BINDING_INVALID")
    models = value.get("models")
    if not isinstance(models, Mapping) or set(models) != {"14M", "31M"}:
        raise StreamingCapacityError("METADATA_MODEL_SET_INVALID")
    snapshots = value.get("aggregate_snapshots")
    if not isinstance(snapshots, Mapping) or set(snapshots) != {"reference", "raw", "stream"}:
        raise StreamingCapacityError("METADATA_SNAPSHOT_SET_INVALID")
    if any(
        not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
        for rows in snapshots.values()
    ):
        raise StreamingCapacityError("METADATA_SNAPSHOT_ROWS_INVALID")
    return (
        models,  # type: ignore[return-value]
        snapshots,  # type: ignore[return-value]
        _mapping(value["fixed_manifests"], field="fixed_manifests"),
        _mapping(value["inflight"], field="inflight"),
        _mapping(value["temporary_json"], field="temporary_json"),
    )


def _resume_units(
    value: Mapping[str, object],
    *,
    root: Path,
    production_index_hash: str,
    expected_units: Sequence[str],
) -> tuple[str, ...]:
    required = {
        "schema_version",
        "formal_eligible",
        "production_index_hash",
        "durable_units",
        "artifact_hash",
    }
    if set(value) != required or value.get("schema_version") != STAGE3_STREAMING_RESUME_SCHEMA:
        raise StreamingCapacityError("RESUME_FIELDS_INVALID")
    if value.get("formal_eligible") is not True:
        raise StreamingCapacityError("RESUME_FORMAL_REQUIRED")
    _checked_hash_bound(value, field="resume_manifest")
    if value.get("production_index_hash") != production_index_hash:
        raise StreamingCapacityError("RESUME_PRODUCTION_INDEX_HASH_MISMATCH")
    raw_units = value.get("durable_units")
    if not isinstance(raw_units, Sequence) or isinstance(raw_units, (str, bytes)):
        raise StreamingCapacityError("RESUME_DURABLE_UNITS_INVALID")
    result: list[str] = []
    for index, raw in enumerate(raw_units):
        item = _mapping(raw, field=f"durable_units[{index}]")
        if set(item) != {"unit_id", "artifact_ref", "artifact_hash"}:
            raise StreamingCapacityError(f"RESUME_DURABLE_UNIT_FIELDS_INVALID:{index}")
        unit_id = item["unit_id"]
        if not isinstance(unit_id, str) or not unit_id:
            raise StreamingCapacityError(f"RESUME_DURABLE_UNIT_ID_INVALID:{index}")
        artifact_ref = _ref(item["artifact_ref"], field=f"durable_units[{index}].artifact_ref")
        artifact_hash = _hash(item["artifact_hash"], field=f"durable_units[{index}].artifact_hash")
        target = (root / Path(artifact_ref)).resolve()
        try:
            target.relative_to(root.resolve())
        except ValueError as error:
            raise StreamingCapacityError(f"RESUME_ARTIFACT_OUTSIDE_WORKSPACE:{index}") from error
        if not target.is_file():
            raise StreamingCapacityError(f"RESUME_ARTIFACT_NOT_FILE:{index}")
        artifact = load_canonical_mapping(target)
        schema = artifact.get("schema_version")
        if (
            not isinstance(schema, str)
            or not schema
            or "fixture" in schema.casefold()
            or "synthetic" in schema.casefold()
        ):
            raise StreamingCapacityError(f"RESUME_ARTIFACT_SCHEMA_INVALID:{index}")
        observed_hashes: list[str] = []
        for hash_field in ("artifact_hash", "result_hash"):
            supplied = artifact.get(hash_field)
            if isinstance(supplied, str) and _HASH_RE.fullmatch(supplied):
                body = {key: value for key, value in artifact.items() if key != hash_field}
                if canonical_json_hash(body) == supplied:
                    observed_hashes.append(supplied)
        if artifact_hash not in observed_hashes:
            raise StreamingCapacityError(f"RESUME_ARTIFACT_HASH_MISMATCH:{index}")
        identities: list[str] = []

        def visit_identity(node: object) -> None:
            if isinstance(node, Mapping):
                for key, child in node.items():
                    if key in {"unit_id", "path_unit_id"} and isinstance(child, str):
                        identities.append(child)
                    visit_identity(child)
            elif isinstance(node, Sequence) and not isinstance(node, (str, bytes)):
                for child in node:
                    visit_identity(child)

        visit_identity(artifact)
        if not identities or any(identity != unit_id for identity in identities):
            raise StreamingCapacityError(f"RESUME_ARTIFACT_UNIT_ID_MISMATCH:{index}")
        result.append(unit_id)
    if len(set(result)) != len(result):
        raise StreamingCapacityError("RESUME_DURABLE_UNIT_DUPLICATE")
    expected = set(expected_units)
    if any(item not in expected for item in result):
        raise StreamingCapacityError("RESUME_DURABLE_UNIT_UNKNOWN")
    return tuple(result)


def run_preflight(arguments: argparse.Namespace) -> dict[str, object]:
    """Load frozen inputs, estimate capacity, and measure both filesystems."""

    root = arguments.workspace_root.resolve()
    plan_path = arguments.formal_plan.resolve()
    index_path = arguments.production_index.resolve()
    counts_path = arguments.parameter_counts.resolve()
    metadata_path = arguments.metadata.resolve()
    # Every declared input is a logical workspace reference.  Do this before
    # opening any file so an absolute/escape path cannot become an unbound
    # source of capacity metadata.
    for path, field in (
        (plan_path, "formal_plan"),
        (index_path, "production_index"),
        (counts_path, "parameter_counts"),
        (metadata_path, "metadata"),
    ):
        _logical_ref(path, root, field=field)
    plan = load_canonical_mapping(plan_path)
    plan_hash = _checked_hash_bound(plan, field="formal_plan")
    if (
        plan.get("schema_version") != "stage3-formal-pilot-plan-v1"
        or plan.get("scope") != "formal"
        or plan.get("formal_eligible") is not True
        or plan.get("plan_kind") != "matrix"
        or plan.get("candidate_rules") != list(DEFAULT_CANDIDATE_RULES)
    ):
        raise StreamingCapacityError("FORMAL_MATRIX_PLAN_INVALID")
    candidate_rule_names = tuple(DEFAULT_CANDIDATE_RULES)
    index = load_production_unit_index(
        index_path,
        workspace_root=root,
        expected_scope="formal",
    )
    index_ref = _logical_ref(index_path, root, field="production_index")
    if (
        plan.get("production_unit_index_ref") != index_ref
        or plan.get("production_unit_index_hash") != index.artifact_hash
    ):
        raise StreamingCapacityError("FORMAL_PLAN_INDEX_BINDING_INVALID")
    units = tuple(unit.path_unit_id for unit in index.units)
    if plan.get("required_unit_ids") != list(units):
        raise StreamingCapacityError("FORMAL_PLAN_UNIT_ORDER_INVALID")
    counts = load_canonical_mapping(counts_path)
    counts_hash = _checked_hash_bound(counts, field="parameter_counts")
    parameter_elements = _parameter_elements(
        counts,
        root=root,
        formal_plan_hash=plan_hash,
        production_index_hash=index.artifact_hash,
        candidate_rule_names=candidate_rule_names,
    )
    metadata = load_canonical_mapping(metadata_path)
    metadata_hash = _checked_hash_bound(metadata, field="streaming_metadata")
    metadata_by_model, snapshots, fixed, inflight, temporary = _metadata_parts(
        metadata,
        formal_plan_hash=plan_hash,
        production_index_hash=index.artifact_hash,
        parameter_counts_hash=counts_hash,
        candidate_rule_names=candidate_rule_names,
    )
    durable: tuple[str, ...] = ()
    resume_ref: str | None = None
    resume_hash: str | None = None
    if arguments.resume_manifest is not None:
        resume_path = arguments.resume_manifest.resolve()
        resume = load_canonical_mapping(resume_path)
        durable = _resume_units(
            resume,
            root=root,
            production_index_hash=index.artifact_hash,
            expected_units=units,
        )
        resume_hash = _checked_hash_bound(resume, field="resume_manifest")
        if any(item not in units for item in durable):
            raise StreamingCapacityError("RESUME_DURABLE_UNIT_UNKNOWN")
        resume_ref = _logical_ref(resume_path, root, field="resume_manifest")
    estimate = estimate_stage3_streaming_capacity(
        unit_model_by_id={unit.path_unit_id: unit.model for unit in index.units},
        parameter_elements_by_model=parameter_elements,
        metadata_by_model=metadata_by_model,
        aggregate_snapshots=snapshots,
        fixed_manifests=fixed,
        inflight=inflight,
        temporary_json=temporary,
        durable_unit_ids=durable,
    )
    output_ref = _logical_ref(arguments.output_filesystem, root, field="output_filesystem")
    cache_ref = _logical_ref(arguments.cache_filesystem, root, field="cache_filesystem")
    shared_filesystem = _same_filesystem(
        arguments.output_filesystem,
        arguments.cache_filesystem,
    )
    if shared_filesystem is None:
        raise StreamingCapacityError("OUTPUT_CACHE_FILESYSTEM_IDENTITY_UNAVAILABLE")
    if shared_filesystem:
        # A single filesystem must have room for both logical allocations and
        # one total margin; checking each logical budget independently would
        # incorrectly allow their requirements to overlap.
        filesystem_budget = estimate.total_budget
        filesystem_inodes = estimate.required_free_inodes
    else:
        filesystem_budget = None
        filesystem_inodes = None
    output_check = check_filesystem_capacity(
        name="output",
        path=arguments.output_filesystem,
        budget=filesystem_budget or estimate.output_budget,
        required_free_inodes=filesystem_inodes or estimate.required_output_free_inodes,
    )
    cache_check = check_filesystem_capacity(
        name="cache",
        path=arguments.cache_filesystem,
        budget=filesystem_budget or estimate.cache_budget,
        required_free_inodes=filesystem_inodes or estimate.required_cache_free_inodes,
    )
    report = build_capacity_preflight_report(
        estimate=estimate,
        output_check=output_check,
        cache_check=cache_check,
        formal_plan_ref=_logical_ref(plan_path, root, field="formal_plan"),
        formal_plan_hash=plan_hash,
        production_index_ref=index_ref,
        production_index_hash=index.artifact_hash,
        parameter_counts_ref=_logical_ref(counts_path, root, field="parameter_counts"),
        parameter_counts_hash=counts_hash,
        metadata_ref=_logical_ref(metadata_path, root, field="metadata"),
        metadata_hash=metadata_hash,
        resume_ref=resume_ref,
        resume_hash=resume_hash,
    )
    # The filesystem paths are operational inputs, not scientific refs, but
    # retaining their logical forms makes the preflight independently auditable.
    report["filesystem_refs"] = {"output": output_ref, "cache": cache_ref}
    report["filesystem_identity"] = {"same_filesystem": shared_filesystem}
    report["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in report.items() if key != "artifact_hash"}
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--formal-plan", type=Path, required=True)
    parser.add_argument("--production-index", type=Path, required=True)
    parser.add_argument("--parameter-counts", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output-filesystem", type=Path, required=True)
    parser.add_argument("--cache-filesystem", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = run_preflight(arguments)
    except (OSError, TypeError, ValueError, StreamingCapacityError) as error:
        report = {
            "schema_version": STAGE3_STREAMING_CAPACITY_SCHEMA,
            "status": "BLOCKED",
            "scope": "formal",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        report["artifact_hash"] = canonical_json_hash(report)
    if arguments.output is not None:
        write_canonical_json(arguments.output, report)
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
