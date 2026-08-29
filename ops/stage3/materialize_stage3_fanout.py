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
    base = _base_v1(base_value)
    identity = base.get("identity")
    runtime = base.get("runtime")
    data = base.get("data")
    sampling = base.get("sampling")
    importance = base.get("importance")
    path_integration = base.get("path_integration")
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
    assert isinstance(data, dict)
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
    index_path = _resolve_ref(source["unit_index_ref"], roots=roots, field="unit_index_ref")
    loaded_hash, units = load_unit_index(index_path, scope=str(scope))
    if loaded_hash != index_hash:
        raise _fail("MATERIALIZATION_UNIT_INDEX_HASH_DRIFT")
    unit_ids = tuple(item.unit_id for item in units)
    endpoint_digests = {item.endpoint_hash for item in units}
    if set(refs_by_endpoint) != endpoint_digests:
        raise _fail("MATERIALIZATION_ENDPOINT_INPUT_COVERAGE_INVALID")
    unit_by_id = {item.unit_id: item for item in units}
    schedule = _schedule(str(task_id), unit_ids)
    config_dir = _resolve_ref(source["config_dir"], roots=(data_root,), field="config_dir")
    selector_dir = _resolve_ref(source["selector_dir"], roots=(data_root,), field="selector_dir")
    result_dir = _resolve_ref(source["result_dir"], roots=(data_root,), field="result_dir")
    config_dir.mkdir(parents=True, exist_ok=True)
    selector_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    from param_importance_nlp.contracts import ResolvedConfigV2

    manifest_steps: list[dict[str, Any]] = []
    first = True
    for index, step in enumerate(schedule):
        unit_id = str(step["unit_id"])
        refs = refs_by_endpoint[unit_by_id[unit_id].endpoint_hash]
        selector = _selector_payload(str(scope), index_hash, unit_id)
        selector_path = selector_dir / f"{unit_id}.json"
        _write_immutable(selector_path, selector)
        selector_ref = selector_path.relative_to(data_root).as_posix()
        overrides = deepcopy(dict(override_value))
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
        resolved = ResolvedConfigV2.resolve(
            base,
            task_id=str(task_id),
            overrides=overrides,
        )
        config_path = config_dir / f"step-{index:03d}-{unit_id}.json"
        _write_immutable(config_path, resolved.to_dict())
        if first:
            retry_overrides = deepcopy(overrides)
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
                base,
                task_id=str(task_id),
                overrides=retry_overrides,
            )
            retry_config_path = config_dir / f"step-{index:03d}-{unit_id}.resume.json"
            _write_immutable(retry_config_path, retry_resolved.to_dict())
        else:
            retry_resolved = resolved
            retry_config_path = config_path
        result_path = result_dir / f"step-{index:03d}-{unit_id}.json"
        manifest_steps.append(
            {
                "step_id": f"step-{index:03d}-{unit_id}",
                "unit_id": unit_id,
                "completes_phases": list(step["completes_phases"]),
                "action": "run" if first else "resume",
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
        first = False
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


__all__ = ["SOURCE_SCHEMA", "materialize", "main"]
