"""Materialize hash-bound Stage 3.05/3.06/3.07 unit configs and schedules.

This is a control-plane compiler.  It consumes a real Stage 2 authority, a
strict production unit index, a formal v1/v2 base config, and task-specific v2
overrides.  It writes no scientific result and never launches a process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from ops.stage3.run_stage3_fanout import SCHEMA_VERSION as FANOUT_SCHEMA
from ops.stage3.run_stage3_formal import (
    EXPECTED_STAGE2_RUN_ID,
    Stage3OrchestratorError,
    _canonical_hash,
    _fail,
    _load_json,
    _resolve_ref,
    _safe_rel,
    _write_atomic,
    load_unit_index,
    validate_stage2_identity,
)


SOURCE_SCHEMA = "stage3-fanout-materialization-source-v1"
MODEL_CONFIG_MAP_SCHEMA = "stage3-model-config-map-v1"
MODEL_OVERRIDES_MAP_SCHEMA = "stage3-model-overrides-map-v1"
FORBIDDEN_RE = re.compile(r"(?:^|[^a-z])(fixture|synthetic)(?:[^a-z]|$)", re.I)
SUPPORTED = {
    "stage3.05_reference_integral_and_precision": "pilot",
    "stage3.06_pilot_and_threshold_freeze": "pilot",
    "stage3.07_formal_experiment_matrix": "formal",
}


def _walk(value: object):
    yield value
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(key)
            yield from _walk(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            yield from _walk(child)


def _no_forbidden(value: object, field: str) -> None:
    for item in _walk(value):
        if isinstance(item, str) and FORBIDDEN_RE.search(item):
            raise _fail("MATERIALIZATION_FORBIDDEN_FORMAL_VALUE", f"{field}:{item}")


def _logical(value: object, field: str) -> str:
    return _safe_rel(value, field).as_posix()


def _hash(value: object, field: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise _fail("MATERIALIZATION_HASH_INVALID", field)
    return value


def _load_mapping_ref(
    value: object,
    *,
    roots: Sequence[Path],
    field: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = _resolve_ref(value, roots=roots, field=field)
    return path, _load_json(path)


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != value:
            raise _fail("MATERIALIZATION_IMMUTABLE_CONFLICT", path)
        return
    _write_atomic(path, value)


def _base_v1(value: Mapping[str, Any]) -> dict[str, Any]:
    if value.get("schema_version") == "resolved-config-v2":
        base = value.get("base_config")
        if not isinstance(base, Mapping):
            raise _fail("MATERIALIZATION_V2_BASE_MISSING")
        return deepcopy(dict(base))
    identity = value.get("identity")
    if not isinstance(identity, Mapping) or identity.get("schema_version") != "resolved-config-v1":
        raise _fail("MATERIALIZATION_BASE_CONFIG_SCHEMA_INVALID")
    return deepcopy(dict(value))


def _model_key(value: object) -> str | None:
    """Normalize a model identity to the production-index key.

    Production units deliberately use the compact ``14M``/``31M`` labels,
    while resolved configs use names such as ``pythia-14m-deduped``.  Matching
    the terminal model-size token keeps the binding strict without requiring
    the control-plane map to rewrite scientific model names.
    """

    if not isinstance(value, str):
        return None
    match = re.search(r"(?:^|[-_])((?:14|31)m)(?:$|[-_])", value, re.I)
    return None if match is None else match.group(1).upper()


def _seed(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _fail("MATERIALIZATION_SEED_INVALID", field)
    return value


def _validate_model_base(
    value: Mapping[str, Any],
    *,
    model: str,
    seed: int,
    field: str,
) -> dict[str, Any]:
    """Validate one map entry and return its v1 scientific base.

    Map entries are required to be complete, hash-checked resolved-config-v2
    objects.  The fanout compiler then derives the Stage 3 task envelope from
    the embedded v1 base exactly as it does for the legacy single-file input.
    """

    try:
        from param_importance_nlp.contracts import ResolvedConfigV2

        resolved = ResolvedConfigV2.from_mapping(value)
    except Exception as error:
        raise _fail("MATERIALIZATION_MODEL_CONFIG_INVALID", field) from error
    if resolved.run_intent != "formal" or resolved.formal_eligible is not True:
        raise _fail("MATERIALIZATION_MODEL_CONFIG_NOT_FORMAL", field)
    base = _base_v1(value)
    identity = base.get("identity")
    model_section = base.get("model")
    if not isinstance(identity, Mapping) or not isinstance(model_section, Mapping):
        raise _fail("MATERIALIZATION_MODEL_CONFIG_IDENTITY_MISSING", field)
    if identity.get("formal_eligible") is not True or identity.get("run_intent") != "formal":
        raise _fail("MATERIALIZATION_MODEL_CONFIG_NOT_FORMAL", field)
    if identity.get("master_seed") != seed:
        raise _fail("MATERIALIZATION_MODEL_SEED_MISMATCH", f"{field}:{seed}")
    # input_run_id is the frozen Stage 2 authority shared by all three
    # trajectories.  A seed-specific base must not silently point at a
    # different upstream run; prepare_base preserves this identity below.
    if identity.get("input_run_id") != EXPECTED_STAGE2_RUN_ID:
        raise _fail("MATERIALIZATION_MODEL_INPUT_RUN_MISMATCH", field)
    identities = (
        ("architecture", model_section.get("architecture")),
        ("asset_id", model_section.get("asset_id")),
        ("initialization_id", model_section.get("initialization_id")),
        ("input_checkpoint_id", identity.get("input_checkpoint_id")),
    )
    recognized = False
    for identity_field, item in identities:
        normalized = _model_key(item)
        if normalized is None:
            # input_checkpoint_id is optional and the other fields may use a
            # registry name without an explicit size token.  The v1 contract
            # handles their type/non-empty requirements; only identifiable
            # model tokens are bound here.
            continue
        recognized = True
        if normalized != model:
            raise _fail(
                "MATERIALIZATION_MODEL_IDENTITY_MISMATCH",
                f"{field}:{identity_field}:{model}",
            )
    if not recognized:
        raise _fail("MATERIALIZATION_MODEL_IDENTITY_MISMATCH", f"{field}:{model}")
    return base


def _load_model_config_map(
    value: Mapping[str, Any],
    *,
    roots: Sequence[Path],
    units: Sequence[Any],
    scope: str,
) -> dict[tuple[str, int], dict[str, Any]] | None:
    """Load a strict ``(model, seed)``→resolved-config-v2 map.

    A model-only map is not sufficient for formal work: 14M has two frozen
    trajectories.  The list form is canonical; a mapping is accepted only
    when each key is the explicit ``MODEL:SEED`` identity.
    """

    if value.get("schema_version") != MODEL_CONFIG_MAP_SCHEMA:
        return None
    expected = {"schema_version", "scope", "entries", "artifact_hash"}
    if set(value) != expected or value.get("scope") != scope:
        raise _fail("MATERIALIZATION_MODEL_CONFIG_MAP_FIELDS_INVALID")
    declared = _hash(value.get("artifact_hash"), "base_config_map.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if declared != _canonical_hash(body):
        raise _fail("MATERIALIZATION_MODEL_CONFIG_MAP_HASH_INVALID")
    pairs = {(str(unit.model), _seed(unit.seed, "unit.seed")) for unit in units}
    if scope == "formal" and pairs != {
        ("14M", 4301),
        ("14M", 4302),
        ("31M", 5301),
    }:
        raise _fail("MATERIALIZATION_FORMAL_MODEL_SEED_COVERAGE_INVALID", sorted(pairs))
    entries = value.get("entries")
    rows = _identity_entry_rows(
        entries,
        expected=pairs,
        field="base_config_map",
        list_fields={"model", "seed", "ref", "config_hash"},
        mapping_fields={"ref", "config_hash"},
        error_code="MATERIALIZATION_MODEL_CONFIG_MAP",
    )
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for (model, seed), entry in rows.items():
        ref = entry.get("ref")
        identity_ref = f"{model}:{seed}"
        config_hash = _hash(
            entry.get("config_hash"), f"base_config_map.{identity_ref}.config_hash"
        )
        path = _resolve_ref(
            ref, roots=roots, field=f"base_config_map.{identity_ref}.ref"
        )
        loaded_value = _load_json(path)
        if loaded_value.get("schema_version") != "resolved-config-v2":
            raise _fail("MATERIALIZATION_MODEL_CONFIG_SCHEMA_INVALID", identity_ref)
        if loaded_value.get("config_hash") != config_hash:
            raise _fail("MATERIALIZATION_MODEL_CONFIG_HASH_MISMATCH", identity_ref)
        loaded[(model, seed)] = _validate_model_base(
            loaded_value,
            model=model,
            seed=seed,
            field=f"base_config_map.{identity_ref}",
        )
    return loaded


def _load_model_overrides_map(
    value: Mapping[str, Any],
    *,
    units: Sequence[Any],
    scope: str,
) -> dict[tuple[str, int], dict[str, Any]] | None:
    """Load optional strict ``(model, seed)``-specific override maps."""

    if value.get("schema_version") != MODEL_OVERRIDES_MAP_SCHEMA:
        return None
    expected = {"schema_version", "scope", "entries", "artifact_hash"}
    if set(value) != expected or value.get("scope") != scope:
        raise _fail("MATERIALIZATION_MODEL_OVERRIDES_MAP_FIELDS_INVALID")
    declared = _hash(value.get("artifact_hash"), "config_overrides_map.artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if declared != _canonical_hash(body):
        raise _fail("MATERIALIZATION_MODEL_OVERRIDES_MAP_HASH_INVALID")
    pairs = {(str(unit.model), _seed(unit.seed, "unit.seed")) for unit in units}
    entries = value.get("entries")
    rows = _identity_entry_rows(
        entries,
        expected=pairs,
        field="config_overrides_map",
        list_fields={"model", "seed", "overrides"},
        mapping_fields={"overrides"},
        error_code="MATERIALIZATION_MODEL_OVERRIDES_MAP",
    )
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for identity, entry in rows.items():
        model, seed = identity
        override = entry.get("overrides")
        if not isinstance(override, Mapping):
            raise _fail(
                "MATERIALIZATION_MODEL_OVERRIDES_MAP_ENTRY_INVALID",
                f"{model}:{seed}",
            )
        loaded[identity] = deepcopy(dict(override))
    return loaded


def _identity_entry_rows(
    entries: object,
    *,
    expected: set[tuple[str, int]],
    field: str,
    list_fields: set[str],
    mapping_fields: set[str],
    error_code: str,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Normalize canonical list or explicit ``MODEL:SEED`` map entries."""

    rows: dict[tuple[str, int], Mapping[str, Any]] = {}
    if isinstance(entries, list):
        for index, raw in enumerate(entries):
            if not isinstance(raw, Mapping) or set(raw) != list_fields:
                raise _fail(f"{error_code}_ENTRY_INVALID", index)
            model = raw.get("model")
            seed = _seed(raw.get("seed"), f"{field}[{index}].seed")
            if not isinstance(model, str) or not model:
                raise _fail(f"{error_code}_ENTRY_INVALID", index)
            identity = (model, seed)
            if identity in rows:
                raise _fail(f"{error_code}_KEYS_INVALID", identity)
            rows[identity] = raw
    elif isinstance(entries, Mapping):
        for raw_key, raw in entries.items():
            if not isinstance(raw_key, str):
                raise _fail(f"{error_code}_KEYS_INVALID")
            match = re.fullmatch(r"([^:]+):(\d+)", raw_key)
            if match is None:
                raise _fail(f"{error_code}_KEYS_INVALID", raw_key)
            model, seed_text = match.groups()
            seed = _seed(int(seed_text), f"{field}.{raw_key}.seed")
            if not isinstance(raw, Mapping) or set(raw) != mapping_fields:
                raise _fail(f"{error_code}_ENTRY_INVALID", raw_key)
            identity = (model, seed)
            if identity in rows:
                raise _fail(f"{error_code}_KEYS_INVALID", raw_key)
            rows[identity] = {
                "model": model,
                "seed": seed,
                **dict(raw),
            }
    else:
        raise _fail(f"{error_code}_KEYS_INVALID")
    if set(rows) != expected:
        raise _fail(f"{error_code}_KEYS_INVALID", sorted(set(rows) ^ expected))
    return rows


def _merge_section(overrides: dict[str, Any], name: str, values: Mapping[str, Any]) -> None:
    current = overrides.get(name)
    if current is None:
        current = {}
    if not isinstance(current, Mapping):
        raise _fail("MATERIALIZATION_OVERRIDE_SECTION_INVALID", name)
    merged = dict(current)
    merged.update(values)
    overrides[name] = merged


def _selector_payload(scope: str, index_hash: str, unit_id: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "stage3-path-unit-selector-v1",
        "scope": scope,
        "unit_index_hash": index_hash,
        "active_unit_id": unit_id,
    }
    value["artifact_hash"] = _canonical_hash(value)
    return value


def _schedule(task_id: str, unit_ids: Sequence[str]) -> list[dict[str, Any]]:
    if task_id.startswith("stage3.05"):
        return [
            {
                "unit_id": unit_id,
                "completes_phases": ["reference"],
                "expected_status": "PASS" if index == len(unit_ids) - 1 else "BLOCKED",
                "expected_blocker_requirements": (
                    [] if index == len(unit_ids) - 1 else ["stage3.05_reference_coverage"]
                ),
            }
            for index, unit_id in enumerate(unit_ids)
        ]
    if task_id.startswith("stage3.06"):
        return [
            {
                "unit_id": unit_id,
                "completes_phases": ["observation"],
                "expected_status": "PASS" if index == len(unit_ids) - 1 else "BLOCKED",
                "expected_blocker_requirements": (
                    [] if index == len(unit_ids) - 1 else ["stage3.06_pilot_coverage"]
                ),
            }
            for index, unit_id in enumerate(unit_ids)
        ]
    # S3.07 streams each unit's reference and observation together.  The
    # matrix aggregate remains intentionally blocked until the final unit, so
    # every prefix has the same single, retryable coverage boundary.
    return [
        {
            "unit_id": unit_id,
            "completes_phases": ["reference", "observation"],
            "expected_status": "PASS" if index == len(unit_ids) - 1 else "BLOCKED",
            "expected_blocker_requirements": (
                [] if index == len(unit_ids) - 1 else ["stage3.07_matrix_coverage"]
            ),
        }
        for index, unit_id in enumerate(unit_ids)
    ]


def materialize(
    source: Mapping[str, Any],
    *,
    workspace_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    expected = {
        "schema_version",
        "task_id",
        "scope",
        "run_config_hash",
        "base_config_ref",
        "stage2_authority_ref",
        "unit_index_ref",
        "unit_index_hash",
        "config_overrides_ref",
        "input_result_refs_by_endpoint",
        "artifact_output_dir",
        "cache_root",
        "config_dir",
        "selector_dir",
        "result_dir",
        "state_dir",
        "status_ref",
        "manifest_ref",
    }
    if set(source) != expected or source.get("schema_version") != SOURCE_SCHEMA:
        raise _fail("MATERIALIZATION_SOURCE_FIELDS_INVALID")
    task_id = source.get("task_id")
    scope = source.get("scope")
    if task_id not in SUPPORTED or scope != SUPPORTED[task_id]:
        raise _fail("MATERIALIZATION_TASK_SCOPE_INVALID")
    run_config_hash = _hash(source.get("run_config_hash"), "run_config_hash")
    index_hash = _hash(source.get("unit_index_hash"), "unit_index_hash")
    logical_fields = (
        "base_config_ref",
        "stage2_authority_ref",
        "unit_index_ref",
        "config_overrides_ref",
        "artifact_output_dir",
        "cache_root",
        "config_dir",
        "selector_dir",
        "result_dir",
        "state_dir",
        "status_ref",
        "manifest_ref",
    )
    for field in logical_fields:
        _logical(source.get(field), field)
    refs_by_endpoint = source.get("input_result_refs_by_endpoint")
    if not isinstance(refs_by_endpoint, Mapping) or not refs_by_endpoint:
        raise _fail("MATERIALIZATION_ENDPOINT_INPUT_REFS_INVALID")
    for endpoint_digest, refs in refs_by_endpoint.items():
        _hash(endpoint_digest, "input_result_refs_by_endpoint.endpoint_digest")
        if (
            not isinstance(refs, list)
            or not refs
            or any(not isinstance(item, str) or not item for item in refs)
            or len(refs) != len(set(refs))
        ):
            raise _fail("MATERIALIZATION_ENDPOINT_INPUT_REFS_INVALID", endpoint_digest)
    _no_forbidden(source, "source")
    roots = (data_root, workspace_root)
    _, stage2 = _load_mapping_ref(
        source["stage2_authority_ref"], roots=roots, field="stage2_authority_ref"
    )
    validate_stage2_identity(stage2)
    _, base_value = _load_mapping_ref(
        source["base_config_ref"], roots=roots, field="base_config_ref"
    )
    _, override_value = _load_mapping_ref(
        source["config_overrides_ref"], roots=roots, field="config_overrides_ref"
    )
    _no_forbidden(base_value, "base_config")
    _no_forbidden(override_value, "config_overrides")
    index_path = _resolve_ref(source["unit_index_ref"], roots=roots, field="unit_index_ref")
    loaded_hash, units = load_unit_index(index_path, scope=str(scope))
    if loaded_hash != index_hash:
        raise _fail("MATERIALIZATION_UNIT_INDEX_HASH_DRIFT")
    model_bases = _load_model_config_map(
        base_value, roots=roots, units=units, scope=str(scope)
    )
    model_overrides = _load_model_overrides_map(
        override_value, units=units, scope=str(scope)
    )
    if model_bases is None:
        base = _base_v1(base_value)
        bases_by_identity: Mapping[tuple[str, int], dict[str, Any]] | None = None
    else:
        if model_overrides is None:
            raise _fail("MATERIALIZATION_MODEL_OVERRIDES_MAP_REQUIRED")
        bases_by_identity = model_bases
        base = None

    def prepare_base(raw_base: dict[str, Any]) -> dict[str, Any]:
        """Apply only task-wide Stage 3 execution identity to one model base."""

        prepared = deepcopy(raw_base)
        identity = prepared.get("identity")
        runtime = prepared.get("runtime")
        data = prepared.get("data")
        sampling = prepared.get("sampling")
        importance = prepared.get("importance")
        path_integration = prepared.get("path_integration")
        if not all(
            isinstance(item, dict)
            for item in (identity, runtime, data, sampling, importance, path_integration)
        ):
            raise _fail("MATERIALIZATION_BASE_SECTIONS_INVALID")
        identity.update(
            {
                "stage": 3,
                "task": task_id,
                "route": "path_integration",
                "run_intent": "formal",
                "formal_eligible": True,
                "input_run_id": EXPECTED_STAGE2_RUN_ID,
            }
        )
        runtime.update(
            {
                "device": "cuda",
                "allow_dirty_worktree": False,
                "offline": True,
                "cache_root": str(source["cache_root"]),
                "output_root": str(PurePosixPath(str(source["artifact_output_dir"])).parent),
                "temp_root": str(PurePosixPath(str(source["cache_root"])) / "tmp"),
            }
        )
        data.update(
            {
                "split": "probe",
                "sampler": "frozen-probe-panel",
                "sampling_design": "disjoint_frozen_probe_panel",
            }
        )
        sampling.update({"reference_batch_size": 32})
        importance.update(
            {
                "estimator_name": "u",
                "clip_mode": "none",
                "require_decision_for_formal": True,
            }
        )
        path_integration.update(
            {
                "enabled": True,
                "probe_count": 2 if scope == "pilot" else 3,
                "default_rule": "simpson",
                "fallback_rule": "gauss_legendre_8",
            }
        )
        return prepared

    if bases_by_identity is None:
        assert base is not None
        base = prepare_base(base)
    else:
        bases_by_identity = {
            identity: prepare_base(model_base)
            for identity, model_base in bases_by_identity.items()
        }
    unit_ids = tuple(item.unit_id for item in units)
    endpoint_digests = {item.endpoint_hash for item in units}
    if set(refs_by_endpoint) != endpoint_digests:
        raise _fail("MATERIALIZATION_ENDPOINT_INPUT_COVERAGE_INVALID")
    unit_by_id = {item.unit_id: item for item in units}
    schedule = _schedule(str(task_id), unit_ids)
    config_dir = _resolve_ref(source["config_dir"], roots=(data_root,), field="config_dir")
    selector_dir = _resolve_ref(source["selector_dir"], roots=(data_root,), field="selector_dir")
    result_dir = _resolve_ref(source["result_dir"], roots=(data_root,), field="result_dir")
    from param_importance_nlp.contracts import ResolvedConfigV2

    # Resolve every step, including all per-seed identities and retry configs,
    # before creating any output directory or writing any selector/config.
    # This makes a late bad override fail atomically rather than leaving an
    # apparently complete prefix of immutable artifacts.
    prepared_steps: list[dict[str, Any]] = []
    first = True
    for index, step in enumerate(schedule):
        unit_id = str(step["unit_id"])
        unit = unit_by_id[unit_id]
        model = str(unit.model)
        seed = _seed(unit.seed, f"unit.{unit_id}.seed")
        identity_key = (model, seed)
        refs = refs_by_endpoint[unit.endpoint_hash]
        selector_path = selector_dir / f"{unit_id}.json"
        selector_ref = selector_path.relative_to(data_root).as_posix()
        selector = _selector_payload(str(scope), index_hash, unit_id)
        if model_overrides is None:
            overrides = deepcopy(dict(override_value))
        else:
            if identity_key not in model_overrides:
                raise _fail(
                    "MATERIALIZATION_MODEL_OVERRIDES_MAP_KEYS_INVALID",
                    f"{model}:{seed}",
                )
            overrides = deepcopy(model_overrides[identity_key])
        orchestration = overrides.get("orchestration")
        if orchestration is None:
            orchestration = {}
        if not isinstance(orchestration, Mapping):
            raise _fail("MATERIALIZATION_ORCHESTRATION_OVERRIDE_INVALID")
        orchestration = dict(orchestration)
        orchestration.update(
            {
                "route_spec_ref": selector_ref,
                "input_result_refs": list(refs),
            }
        )
        overrides["orchestration"] = orchestration
        _merge_section(
            overrides,
            "recovery",
            {"resume_ref": None if first else str(source["cache_root"])},
        )
        _merge_section(
            overrides,
            "artifacts",
            {"output_dir": str(source["artifact_output_dir"]), "publish_partial": False},
        )
        selected_base = (
            base
            if bases_by_identity is None
            else bases_by_identity.get(identity_key)
        )
        if selected_base is None:
            raise _fail(
                "MATERIALIZATION_MODEL_CONFIG_MAP_KEYS_INVALID",
                f"{model}:{seed}",
            )
        resolved = ResolvedConfigV2.resolve(
            selected_base,
            task_id=str(task_id),
            overrides=overrides,
        )
        retry_overrides = deepcopy(overrides)
        if first:
            _merge_section(
                retry_overrides,
                "recovery",
                # The first shard's refinement commits live below the shared
                # task artifact root.  The cache root may contain reusable
                # node gradients from an earlier immutable attempt, but those
                # objects are not task-resume authority.  Bind a retry to the
                # actual partial task output so a warm cache remains a fresh
                # run while a failed first shard can resume its commits.
                {"resume_ref": str(source["artifact_output_dir"])},
            )
            retry_resolved = ResolvedConfigV2.resolve(
                selected_base,
                task_id=str(task_id),
                overrides=retry_overrides,
            )
        else:
            retry_resolved = resolved
        config_path = config_dir / f"step-{index:03d}-{unit_id}.json"
        retry_config_path = (
            config_dir / f"step-{index:03d}-{unit_id}.resume.json"
            if first
            else config_path
        )
        result_path = result_dir / f"step-{index:03d}-{unit_id}.json"
        prepared_steps.append(
            {
                "index": index,
                "step": step,
                "unit_id": unit_id,
                "selector": selector,
                "selector_path": selector_path,
                "resolved": resolved,
                "retry_resolved": retry_resolved,
                "config_path": config_path,
                "retry_config_path": retry_config_path,
                "result_path": result_path,
                "first": first,
            }
        )
        first = False

    config_dir.mkdir(parents=True, exist_ok=True)
    selector_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    manifest_steps: list[dict[str, Any]] = []
    for prepared in prepared_steps:
        index = int(prepared["index"])
        step = prepared["step"]
        unit_id = str(prepared["unit_id"])
        selector = prepared["selector"]
        selector_path = prepared["selector_path"]
        resolved = prepared["resolved"]
        retry_resolved = prepared["retry_resolved"]
        config_path = prepared["config_path"]
        retry_config_path = prepared["retry_config_path"]
        result_path = prepared["result_path"]
        _write_immutable(selector_path, selector)
        _write_immutable(config_path, resolved.to_dict())
        if bool(prepared["first"]):
            _write_immutable(retry_config_path, retry_resolved.to_dict())
        manifest_steps.append(
            {
                "step_id": f"step-{index:03d}-{unit_id}",
                "unit_id": unit_id,
                "completes_phases": list(step["completes_phases"]),
                "action": "run" if bool(prepared["first"]) else "resume",
                "config_ref": config_path.relative_to(data_root).as_posix(),
                "config_hash": resolved.config_hash,
                "retry_config_ref": retry_config_path.relative_to(data_root).as_posix(),
                "retry_config_hash": retry_resolved.config_hash,
                "result_ref": result_path.relative_to(data_root).as_posix(),
                "command": [
                    "{python}",
                    "-m",
                    "param_importance_nlp",
                    "task",
                    "{action}",
                    "--config",
                    "{config}",
                    "--environment",
                    "{environment}",
                    "--result",
                    "{result}",
                ],
                "expected_status": step["expected_status"],
                "expected_blocker_requirements": list(
                    step["expected_blocker_requirements"]
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": FANOUT_SCHEMA,
        "task_id": task_id,
        "scope": scope,
        "run_config_hash": run_config_hash,
        "unit_index_ref": str(source["unit_index_ref"]),
        "unit_index_hash": index_hash,
        "state_dir": str(source["state_dir"]),
        "status_ref": str(source["status_ref"]),
        "steps": manifest_steps,
    }
    manifest["manifest_hash"] = _canonical_hash(manifest)
    manifest_path = _resolve_ref(source["manifest_ref"], roots=(data_root,), field="manifest_ref")
    _write_immutable(manifest_path, manifest)
    final = manifest_steps[-1]
    result = {
        "schema_version": "stage3-fanout-materialization-receipt-v1",
        "task_id": task_id,
        "scope": scope,
        "unit_count": len(unit_ids),
        "step_count": len(manifest_steps),
        "manifest_ref": str(source["manifest_ref"]),
        "manifest_hash": manifest["manifest_hash"],
        "final_config_ref": final["config_ref"],
        "final_config_hash": final["config_hash"],
        "final_result_ref": final["result_ref"],
        "status_ref": str(source["status_ref"]),
        "artifact_output_dir": str(source["artifact_output_dir"]),
        "expected_output_refs": {
            kind: f"{source['artifact_output_dir']}/commits/{kind}.json"
            for kind in (
                ("path_integral_reference", "precision_budget")
                if str(task_id).startswith("stage3.05")
                else ("quadrature_pilot_report", "threshold_freeze")
                if str(task_id).startswith("stage3.06")
                else ("formal_path_results", "completeness_report")
            )
        },
    }
    result["artifact_hash"] = _canonical_hash(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        value = materialize(
            _load_json(arguments.source.resolve()),
            workspace_root=arguments.workspace_root.resolve(),
            data_root=arguments.data_root.resolve(),
        )
        if arguments.receipt is not None:
            receipt = arguments.receipt.resolve()
            try:
                receipt.relative_to(arguments.data_root.resolve())
            except ValueError as error:
                raise _fail("MATERIALIZATION_RECEIPT_OUTSIDE_DATA_ROOT") from error
            _write_immutable(receipt, value)
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_CONFIG_MAP_SCHEMA",
    "MODEL_OVERRIDES_MAP_SCHEMA",
    "SOURCE_SCHEMA",
    "materialize",
    "main",
]
