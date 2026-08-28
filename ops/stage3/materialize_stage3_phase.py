"""Assemble one immutable pilot/formal Stage 3 orchestrator manifest.

This is a control-plane compiler.  It consumes task materialization receipts;
it neither launches a worker nor invents an experiment result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
from pathlib import Path
from typing import Any

from ops.stage3.materialize_stage3_fanout import _hash, _logical, _no_forbidden, _write_immutable
from ops.stage3.run_stage3_formal import (
    FORMAL_TASK_ORDER,
    PILOT_TASK_ORDER,
    STAGE3_ORCHESTRATOR_SCHEMA,
    Stage3Orchestrator,
    Stage3OrchestratorError,
    _canonical_hash,
    _fail,
    _load_json,
    _resolve_ref,
)


SOURCE_SCHEMA = "stage3-phase-materialization-source-v1"
DIRECT_RECEIPT_SCHEMA = "stage3-task-materialization-receipt-v1"
FANOUT_RECEIPT_SCHEMA = "stage3-fanout-materialization-receipt-v1"
FANOUT_TASKS = {
    "stage3.05_reference_integral_and_precision",
    "stage3.06_pilot_and_threshold_freeze",
    "stage3.07_formal_experiment_matrix",
}


def _verified_receipt(
    ref: str,
    *,
    task_id: str,
    workspace_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    path = _resolve_ref(ref, roots=(data_root, workspace_root), field=f"{task_id}.receipt_ref")
    value = _load_json(path)
    if value.get("task_id") != task_id:
        raise _fail("PHASE_RECEIPT_TASK_MISMATCH", task_id)
    declared = value.get("artifact_hash")
    if declared != _canonical_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
        raise _fail("PHASE_RECEIPT_HASH_INVALID", task_id)
    expected_schema = FANOUT_RECEIPT_SCHEMA if task_id in FANOUT_TASKS else DIRECT_RECEIPT_SCHEMA
    if value.get("schema_version") != expected_schema:
        raise _fail("PHASE_RECEIPT_SCHEMA_INVALID", task_id)
    _no_forbidden(value, f"{task_id}.receipt")
    return value


def _fanout_spec(
    receipt: Mapping[str, Any],
    *,
    task_id: str,
    scope: str,
    run_config_hash: str,
    data_root: Path,
) -> Mapping[str, Any]:
    if receipt.get("scope") != scope:
        raise _fail("PHASE_FANOUT_SCOPE_MISMATCH", task_id)
    manifest_ref = _logical(receipt.get("manifest_ref"), f"{task_id}.manifest_ref")
    manifest = _load_json(
        _resolve_ref(manifest_ref, roots=(data_root,), field=f"{task_id}.manifest_ref")
    )
    if (
        manifest.get("manifest_hash") != receipt.get("manifest_hash")
        or manifest.get("run_config_hash") != run_config_hash
    ):
        raise _fail("PHASE_FANOUT_MANIFEST_DRIFT", task_id)
    output_refs = receipt.get("expected_output_refs")
    if not isinstance(output_refs, Mapping):
        raise _fail("PHASE_RECEIPT_OUTPUTS_INVALID", task_id)
    status_ref = receipt.get("status_ref")
    unit_status_ref = (
        str(status_ref)
        if task_id
        in {
            "stage3.06_pilot_and_threshold_freeze",
            "stage3.07_formal_experiment_matrix",
        }
        else None
    )
    return {
        "task_id": task_id,
        "config_ref": str(receipt["final_config_ref"]),
        "config_hash": str(receipt["final_config_hash"]),
        "environment_ref": None,
        "evidence_refs": {},
        "command": [
            "{python}",
            "-m",
            "ops.stage3.run_stage3_fanout",
            "--manifest",
            "{data_root}/" + str(manifest_ref),
            "--workspace-root",
            "{workspace_root}",
            "--data-root",
            "{data_root}",
            "--environment",
            "{environment}",
        ],
        "output_refs": dict(output_refs),
        "output_dir": str(receipt["artifact_output_dir"]),
        "result_ref": str(receipt["final_result_ref"]),
        "unit_status_ref": unit_status_ref,
        "external_gate_ref": None,
    }


def _direct_spec(receipt: Mapping[str, Any], *, task_id: str) -> Mapping[str, Any]:
    command = receipt.get("command")
    outputs = receipt.get("output_refs")
    evidence = receipt.get("evidence_refs")
    if (
        not isinstance(command, list)
        or not command
        or not isinstance(outputs, Mapping)
        or not isinstance(evidence, Mapping)
    ):
        raise _fail("PHASE_DIRECT_RECEIPT_INVALID", task_id)
    return {
        "task_id": task_id,
        "config_ref": str(receipt["config_ref"]),
        "config_hash": str(receipt["config_hash"]),
        "environment_ref": None,
        "evidence_refs": dict(evidence),
        "command": list(command),
        "output_refs": dict(outputs),
        "output_dir": str(receipt["artifact_output_dir"]),
        "result_ref": str(receipt["result_ref"]),
        "unit_status_ref": None,
        "external_gate_ref": receipt.get("external_gate_ref"),
    }


def materialize(
    source: Mapping[str, Any],
    *,
    workspace_root: Path,
    data_root: Path,
) -> Mapping[str, Any]:
    expected = {
        "schema_version",
        "scope",
        "config_hash",
        "state_dir",
        "unit_index_ref",
        "stage2_authority_ref",
        "initial_execution_evidence_ref",
        "initial_execution_config_hash",
        "initial_environment_ref",
        "g30_gate_ref",
        "health_snapshot_ref",
        "scope_decision_ref",
        "contract_freeze_hash",
        "asset_manifest_hashes",
        "task_receipt_refs",
    }
    if set(source) != expected or source.get("schema_version") != SOURCE_SCHEMA:
        raise _fail("PHASE_MATERIALIZATION_SOURCE_FIELDS_INVALID")
    scope = source.get("scope")
    if scope not in {"pilot", "formal"}:
        raise _fail("PHASE_MATERIALIZATION_SCOPE_INVALID")
    order = PILOT_TASK_ORDER if scope == "pilot" else FORMAL_TASK_ORDER
    run_config_hash = _hash(source.get("config_hash"), "config_hash")
    _hash(source.get("initial_execution_config_hash"), "initial_execution_config_hash")
    _hash(source.get("contract_freeze_hash"), "contract_freeze_hash")
    for field in (
        "state_dir",
        "unit_index_ref",
        "stage2_authority_ref",
        "initial_execution_evidence_ref",
        "initial_environment_ref",
        "g30_gate_ref",
        "health_snapshot_ref",
        "scope_decision_ref",
    ):
        _logical(source.get(field), field)
    assets = source.get("asset_manifest_hashes")
    if not isinstance(assets, list) or not assets or any(not isinstance(item, str) for item in assets):
        raise _fail("PHASE_ASSET_MANIFEST_HASHES_INVALID")
    for item in assets:
        _hash(item, "asset_manifest_hash")
    receipt_refs = source.get("task_receipt_refs")
    if not isinstance(receipt_refs, Mapping) or set(receipt_refs) != set(order):
        raise _fail("PHASE_RECEIPT_COVERAGE_INVALID")
    if any(not isinstance(item, str) or not item for item in receipt_refs.values()):
        raise _fail("PHASE_RECEIPT_REF_INVALID")
    _no_forbidden(source, "source")

    tasks: list[Mapping[str, Any]] = []
    for task_id in order:
        receipt = _verified_receipt(
            str(receipt_refs[task_id]),
            task_id=task_id,
            workspace_root=workspace_root,
            data_root=data_root,
        )
        if task_id in FANOUT_TASKS:
            task = _fanout_spec(
                receipt,
                task_id=task_id,
                scope=str(scope),
                run_config_hash=run_config_hash,
                data_root=data_root,
            )
        else:
            task = _direct_spec(receipt, task_id=task_id)
        tasks.append(task)

    manifest = {
        "schema_version": STAGE3_ORCHESTRATOR_SCHEMA,
        "scope": scope,
        "config_hash": run_config_hash,
        "state_dir": str(source["state_dir"]),
        "unit_index_ref": str(source["unit_index_ref"]),
        "stage2_authority_ref": str(source["stage2_authority_ref"]),
        "initial_execution_evidence_ref": str(source["initial_execution_evidence_ref"]),
        "initial_execution_config_hash": str(source["initial_execution_config_hash"]),
        "initial_environment_ref": str(source["initial_environment_ref"]),
        "g30_gate_ref": str(source["g30_gate_ref"]),
        "health_snapshot_ref": str(source["health_snapshot_ref"]),
        "scope_decision_ref": str(source["scope_decision_ref"]),
        "contract_freeze_hash": str(source["contract_freeze_hash"]),
        "asset_manifest_hashes": list(assets),
        "tasks": tasks,
    }
    # Constructor validation catches task-output and DAG shape drift before the
    # manifest is published, without touching a GPU or launching a worker.
    Stage3Orchestrator(manifest, workspace_root=workspace_root, data_root=data_root)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        data_root = arguments.data_root.resolve()
        manifest_path = arguments.manifest.resolve()
        try:
            manifest_path.relative_to(data_root)
        except ValueError as error:
            raise _fail("PHASE_MANIFEST_OUTSIDE_DATA_ROOT") from error
        value = materialize(
            _load_json(arguments.source.resolve()),
            workspace_root=arguments.workspace_root.resolve(),
            data_root=data_root,
        )
        _write_immutable(manifest_path, value)
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SOURCE_SCHEMA", "materialize", "main"]
