"""Materialize one non-fan-out Stage 3 task as a strict formal v2 config."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

from ops.stage3.materialize_stage3_fanout import (
    FORBIDDEN_RE,
    _base_v1,
    _hash,
    _logical,
    _merge_section,
    _no_forbidden,
    _write_immutable,
)
from ops.stage3.run_stage3_formal import (
    EXPECTED_OUTPUTS,
    EXPECTED_STAGE2_RUN_ID,
    Stage3OrchestratorError,
    _canonical_hash,
    _fail,
    _load_json,
    _resolve_ref,
    validate_stage2_identity,
)


SOURCE_SCHEMA = "stage3-task-materialization-source-v1"
FANOUT_TASKS = {
    "stage3.05_reference_integral_and_precision",
    "stage3.06_pilot_and_threshold_freeze",
    "stage3.07_formal_experiment_matrix",
}


def materialize(
    source: Mapping[str, Any],
    *,
    workspace_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    expected = {
        "schema_version",
        "task_id",
        "base_config_ref",
        "stage2_authority_ref",
        "config_overrides_ref",
        "input_result_refs",
        "artifact_output_dir",
        "authority_output_dir",
        "cache_root",
        "config_ref",
        "result_ref",
        "evidence_refs",
        "external_gate_ref",
        "route_spec_ref",
    }
    if set(source) != expected or source.get("schema_version") != SOURCE_SCHEMA:
        raise _fail("TASK_MATERIALIZATION_SOURCE_FIELDS_INVALID")
    task_id = source.get("task_id")
    if task_id not in EXPECTED_OUTPUTS or task_id in FANOUT_TASKS:
        raise _fail("TASK_MATERIALIZATION_TASK_INVALID", task_id)
    for field in (
        "base_config_ref",
        "stage2_authority_ref",
        "config_overrides_ref",
        "artifact_output_dir",
        "authority_output_dir",
        "cache_root",
        "config_ref",
        "result_ref",
    ):
        _logical(source.get(field), field)
    refs = source.get("input_result_refs")
    if (
        not isinstance(refs, list)
        or any(not isinstance(item, str) or not item for item in refs)
        or len(refs) != len(set(refs))
    ):
        raise _fail("TASK_MATERIALIZATION_INPUT_REFS_INVALID")
    evidence_refs = source.get("evidence_refs")
    if not isinstance(evidence_refs, Mapping) or any(
        not isinstance(key, str)
        or not key
        or not isinstance(value, str)
        or not value
        for key, value in evidence_refs.items()
    ):
        raise _fail("TASK_MATERIALIZATION_EVIDENCE_REFS_INVALID")
    external_gate_ref = source.get("external_gate_ref")
    if external_gate_ref is not None:
        _logical(external_gate_ref, "external_gate_ref")
    route_spec_ref = source.get("route_spec_ref")
    selector_tasks = {
        "stage3.03_endpoint_and_probe_pipeline",
        "stage3.04_quadrature_engine_and_unit_tests",
    }
    if task_id in selector_tasks:
        _logical(route_spec_ref, "route_spec_ref")
        selector = _load_json(
            _resolve_ref(
                route_spec_ref,
                roots=(data_root, workspace_root),
                field="route_spec_ref",
            )
        )
        if (
            selector.get("schema_version") != "stage3-probe-selector-v1"
            or selector.get("scope") != "formal"
            or selector.get("artifact_hash")
            != _canonical_hash(
                {key: item for key, item in selector.items() if key != "artifact_hash"}
            )
        ):
            raise _fail("TASK_MATERIALIZATION_ROUTE_SELECTOR_INVALID", task_id)
    elif route_spec_ref is not None:
        raise _fail("TASK_MATERIALIZATION_ROUTE_SELECTOR_UNEXPECTED", task_id)
    _no_forbidden(source, "source")
    roots = (data_root, workspace_root)
    stage2 = _load_json(
        _resolve_ref(
            source["stage2_authority_ref"], roots=roots, field="stage2_authority_ref"
        )
    )
    validate_stage2_identity(stage2)
    base_value = _load_json(
        _resolve_ref(source["base_config_ref"], roots=roots, field="base_config_ref")
    )
    overrides = deepcopy(
        dict(
            _load_json(
                _resolve_ref(
                    source["config_overrides_ref"],
                    roots=roots,
                    field="config_overrides_ref",
                )
            )
        )
    )
    _no_forbidden(base_value, "base_config")
    _no_forbidden(overrides, "config_overrides")
    base = _base_v1(base_value)
    identity = base.get("identity")
    runtime = base.get("runtime")
    sampling = base.get("sampling")
    importance = base.get("importance")
    path_integration = base.get("path_integration")
    if not all(
        isinstance(item, dict)
        for item in (identity, runtime, sampling, importance, path_integration)
    ):
        raise _fail("TASK_MATERIALIZATION_BASE_SECTIONS_INVALID")
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
            "probe_count": 3,
            "default_rule": "simpson",
            "fallback_rule": "gauss_legendre_8",
        }
    )
    orchestration = overrides.get("orchestration")
    if orchestration is None:
        orchestration = {}
    if not isinstance(orchestration, Mapping):
        raise _fail("TASK_MATERIALIZATION_ORCHESTRATION_INVALID")
    orchestration = dict(orchestration)
    orchestration["input_result_refs"] = list(refs)
    orchestration["route_spec_ref"] = route_spec_ref
    overrides["orchestration"] = orchestration
    _merge_section(overrides, "recovery", {"resume_ref": None})
    _merge_section(
        overrides,
        "artifacts",
        {"output_dir": str(source["artifact_output_dir"]), "publish_partial": False},
    )
    from param_importance_nlp.contracts import ResolvedConfigV2

    resolved = ResolvedConfigV2.resolve(base, task_id=str(task_id), overrides=overrides)
    config_path = _resolve_ref(source["config_ref"], roots=(data_root,), field="config_ref")
    _write_immutable(config_path, resolved.to_dict())
    output_refs = {
        kind: f"{source['artifact_output_dir']}/commits/{kind}.json"
        for kind in EXPECTED_OUTPUTS[str(task_id)]
    }
    receipt: dict[str, Any] = {
        "schema_version": "stage3-task-materialization-receipt-v1",
        "task_id": task_id,
        "config_ref": str(source["config_ref"]),
        "config_hash": resolved.config_hash,
        "result_ref": str(source["result_ref"]),
        "output_refs": output_refs,
        "artifact_output_dir": str(source["artifact_output_dir"]),
        "authority_output_dir": str(source["authority_output_dir"]),
        "evidence_refs": dict(evidence_refs),
        "external_gate_ref": external_gate_ref,
        "command": [
            "{python}",
            "-m",
            "param_importance_nlp",
            "task",
            "run",
            "--config",
            "{config}",
            "--environment",
            "{environment}",
            "--result",
            "{result}",
        ],
    }
    receipt["artifact_hash"] = _canonical_hash(receipt)
    return receipt


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
                raise _fail("TASK_MATERIALIZATION_RECEIPT_OUTSIDE_DATA_ROOT") from error
            _write_immutable(receipt, value)
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(dict(value), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOURCE_SCHEMA", "materialize", "main"]
