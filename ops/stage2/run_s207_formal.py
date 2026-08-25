"""Detached production launcher for the frozen S2.7/G2.5 sweep.

The command has four deliberately separate modes:

``--preflight`` reads the frozen matrix/mapping, six-cell materialization and
GPU inventory without constructing a provider; ``--execute`` starts at most
four UUID-bound workers in a dynamic queue; ``--worker`` runs one cell; and
``--wait``/``--status`` are read-only recovery views.  Nothing in this file
creates a confirmatory draw or emits a statistical conclusion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
    S27G25Blocked,
    S27StatusStore,
    S27DetachedStatus,
)
from param_importance_nlp.experiments.stage2_s207_runner import (
    S27DetachedLauncher,
    S27ExecutionBlocked,
    S27ProductionWorker,
    build_s27_worker_command,
    load_s27_frozen_mappings,
    load_s27_materialized_inputs,
    load_s27_plan,
    load_s27_shard_plan,
    normalized_gpu_inventory,
    nvidia_smi_inventory,
)
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.runtime.task_runtime import TaskExecutionRequest, TaskRuntimeEnvironment


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S27ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE")
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S27ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    return result


def _load_inventory(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return nvidia_smi_inventory()
    value = load_canonical_json(path.resolve())
    rows = value.get("gpus") if isinstance(value, Mapping) else value
    if not isinstance(rows, list) or not all(isinstance(item, Mapping) for item in rows):
        raise S27ExecutionBlocked("S27_GPU_INVENTORY_JSON_INVALID")
    return [dict(item) for item in rows]


def _load_execution(root: Path, reference: str) -> FormalExecutionEvidence:
    path = _logical(root, reference, field="execution_evidence_ref")
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise S27ExecutionBlocked("S27_EXECUTION_EVIDENCE_OBJECT_REQUIRED")
    try:
        evidence = FormalExecutionEvidence.from_mapping(dict(value))
        evidence.require_for_stage(2)
    except (TypeError, ValueError, RuntimeError) as error:
        raise S27ExecutionBlocked(f"S27_EXECUTION_EVIDENCE_INVALID:{error}") from error
    return evidence


def _derive_s207_config(source_config: ResolvedConfigV2) -> ResolvedConfigV2:
    """Retarget the complete S2.4 v2 wire object for the S2.7 adapter.

    Re-resolving from ``base_config`` would drop the materialized S2.4
    provider roots, orchestration lineage, execution overrides and output
    contract.  Keep those fields byte-for-byte stable and change only the
    task identity, runner kind, and required artifact kinds.
    """

    source_wire = source_config.to_dict()
    payload = {
        key: value
        for key, value in source_wire.items()
        if key not in {"config_hash", "full_hash"}
    }
    payload["task_id"] = "stage2.07_main_sweep"
    execution = payload.get("execution")
    artifacts = payload.get("artifacts")
    if not isinstance(execution, Mapping) or not isinstance(artifacts, Mapping):
        raise S27ExecutionBlocked("S27_CONFIG_SECTIONS_MISSING")
    execution_wire = dict(execution)
    execution_wire["runner_kind"] = DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep").runner_kind.value
    payload["execution"] = execution_wire
    artifacts_wire = dict(artifacts)
    artifacts_wire["required_kinds"] = list(
        DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep").artifact_kinds
    )
    payload["artifacts"] = artifacts_wire
    try:
        derived = ResolvedConfigV2(payload)
    except (TypeError, ValueError) as error:
        raise S27ExecutionBlocked("S27_CONFIG_DERIVATION_FAILED") from error

    derived_wire = derived.to_dict()
    preserved_sections = set(source_wire) - {
        "task_id", "execution", "artifacts", "config_hash", "full_hash"
    }
    for section in preserved_sections:
        if derived_wire.get(section) != source_wire.get(section):
            raise S27ExecutionBlocked(f"S27_CONFIG_BINDING_DROPPED:{section}")
    derived_execution = derived_wire.get("execution")
    source_execution = source_wire.get("execution")
    if not isinstance(derived_execution, Mapping) or not isinstance(source_execution, Mapping):
        raise S27ExecutionBlocked("S27_CONFIG_EXECUTION_INVALID")
    for field in set(source_execution) - {"runner_kind"}:
        if derived_execution.get(field) != source_execution.get(field):
            raise S27ExecutionBlocked(f"S27_CONFIG_EXECUTION_OVERRIDE_DROPPED:{field}")
    derived_artifacts = derived_wire.get("artifacts")
    source_artifacts = source_wire.get("artifacts")
    if not isinstance(derived_artifacts, Mapping) or not isinstance(source_artifacts, Mapping):
        raise S27ExecutionBlocked("S27_CONFIG_ARTIFACTS_INVALID")
    for field in set(source_artifacts) - {"required_kinds"}:
        if derived_artifacts.get(field) != source_artifacts.get(field):
            raise S27ExecutionBlocked(f"S27_CONFIG_ARTIFACT_OVERRIDE_DROPPED:{field}")
    return derived


def _build_request(
    root: Path,
    *,
    cell_id: str,
    materialization_index_ref: str,
    execution_evidence_ref: str,
) -> TaskExecutionRequest:
    """Rebind the immutable S2.4 config/environment to the S2.7 task ID."""

    materialized = load_s27_materialized_inputs(root, materialization_index_ref)[cell_id]
    config_value = load_canonical_json(_logical(root, materialized.config_ref, field=f"config.{cell_id}"))
    env_value = load_canonical_json(_logical(root, materialized.environment_ref, field=f"environment.{cell_id}"))
    if not isinstance(config_value, Mapping) or not isinstance(env_value, Mapping):
        raise S27ExecutionBlocked(f"S27_CELL_INPUT_OBJECT_REQUIRED:{cell_id}")
    try:
        source_config = ResolvedConfigV2.from_mapping(dict(config_value))
        environment = TaskRuntimeEnvironment.from_mapping(dict(env_value))
    except (TypeError, ValueError, RuntimeError) as error:
        raise S27ExecutionBlocked(f"S27_CELL_INPUT_INVALID:{cell_id}:{error}") from error
    if source_config.run_intent != "formal":
        raise S27ExecutionBlocked(f"S27_CELL_CONFIG_NOT_FORMAL:{cell_id}")
    config = source_config if source_config.task_id == "stage2.07_main_sweep" else _derive_s207_config(source_config)
    refs = dict(environment.evidence_refs)
    refs["formal_execution"] = execution_evidence_ref
    environment = TaskRuntimeEnvironment(
        capabilities=environment.capabilities | frozenset({"server", "cuda", "model_assets", "data_assets"}),
        frozen_contract_stages=environment.frozen_contract_stages | frozenset({2}),
        passed_gate_ids=environment.passed_gate_ids | frozenset({"stage2.G2.4b"}),
        estimator_decision_ref=environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    return TaskExecutionRequest(config=config, task=DEFAULT_TASK_CATALOG.get("stage2.07_main_sweep"), environment=environment)


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    plan = load_s27_plan(root, args.plan_ref)
    # Re-read all producer artifacts at launch time.  A prepared plan alone is
    # not permission to consume a replaced matrix, mapping, or G2.4b Gate.
    matrix_path = _logical(root, plan.frozen_inputs.matrix_ref, field="g24b_matrix")
    matrix = load_canonical_json(matrix_path)
    if not isinstance(matrix, Mapping) or matrix.get("artifact_hash") != plan.frozen_inputs.matrix_hash:
        raise S27ExecutionBlocked("S27_G24B_MATRIX_HASH_DRIFT")
    gate_path = _logical(root, plan.frozen_inputs.g24b_gate_ref, field="g24b_gate")
    gate = load_canonical_json(gate_path)
    if not isinstance(gate, Mapping) or gate.get("artifact_hash") != plan.frozen_inputs.g24b_gate_hash or gate.get("gate_id") != "stage2.G2.4b" or gate.get("status") != "PASS":
        raise S27ExecutionBlocked("S27_G24B_GATE_NOT_PASS_OR_DRIFT")
    mapping = load_s27_frozen_mappings(root, plan)
    materialized = load_s27_materialized_inputs(root, args.materialization_index_ref)
    if set(materialized) != set(plan.cells[i].cell_id for i in range(len(plan.cells))):
        raise S27ExecutionBlocked("S27_MATERIALIZATION_CELL_SET_MISMATCH")
    inventory = _load_inventory(args.gpu_inventory_json)
    gpu = normalized_gpu_inventory(inventory)
    execution = _load_execution(root, args.execution_evidence_ref)
    execution_g24b = [gate for gate in execution.prerequisite_gates if gate.gate_id == "stage2.G2.4b"]
    if len(execution_g24b) != 1 or execution_g24b[0].artifact_hash != plan.frozen_inputs.g24b_gate_hash:
        raise S27ExecutionBlocked("S27_EXECUTION_EVIDENCE_G24B_BINDING_INVALID")
    confirmatory_draws = sum(unit.batch_size for unit in plan.frozen_inputs.units)
    throughput = args.throughput_sequences_per_second
    wall_model = {
        "confirmatory_draw_events": confirmatory_draws,
        "approved_gpu_count": len(APPROVED_GPU_UUIDS),
        "formula": "confirmatory_draw_events/(approved_gpu_count*throughput_sequences_per_second)",
        "throughput_sequences_per_second": throughput,
        "lower_bound_seconds": (
            None
            if throughput is None or throughput <= 0
            else confirmatory_draws / (len(APPROVED_GPU_UUIDS) * throughput)
        ),
        "note": "ideal four-GPU lower bound; frozen checkpoint-wave barriers may increase wall time",
    }
    return {
        "schema_version": "stage2-s27-production-preflight-v1",
        "status": "READY",
        "formal_eligible": True,
        "plan_ref": args.plan_ref,
        "plan_hash": plan.artifact_hash,
        "matrix_ref": plan.frozen_inputs.matrix_ref,
        "matrix_hash": plan.frozen_inputs.matrix_hash,
        "mapping_ref": plan.frozen_inputs.mapping_ref,
        "mapping_hash": plan.frozen_inputs.mapping_hash,
        "g24b_gate_ref": plan.frozen_inputs.g24b_gate_ref,
        "g24b_gate_hash": plan.frozen_inputs.g24b_gate_hash,
        "cell_count": len(plan.cells),
        "wave_order": [cell.cell_id for cell in plan.cells],
        "within_wave_shard_count": len(APPROVED_GPU_UUIDS),
        "expected_unit_count": plan.frozen_inputs.completion_denominator,
        "B": plan.frozen_inputs.batch_size,
        "M": plan.frozen_inputs.microbatch_count,
        "R": plan.frozen_inputs.repetitions,
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI,
        "gpu_inventory": gpu,
        "confirmatory_draws_generated": False,
        "mapping_units_loaded": len(mapping),
        "optional_stopping": False,
        "silent_skip": False,
        "max_attempts": 1,
        "wall_time_model": wall_model,
    }


def _worker(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    plan = load_s27_plan(root, args.plan_ref)
    if args.cell_id not in {cell.cell_id for cell in plan.cells}:
        raise S27ExecutionBlocked("S27_WORKER_CELL_UNKNOWN")
    if args.gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S27ExecutionBlocked("S27_WORKER_GPU_UNAPPROVED")
    if (args.shard_plan_ref is None) != (args.shard_index is None):
        raise S27ExecutionBlocked("S27_WORKER_SHARD_PLAN_AND_INDEX_REQUIRED")
    shard_index: int | None = None
    shard_count: int | None = None
    shard_unit_ids: tuple[str, ...] | None = None
    if args.shard_plan_ref is not None and args.shard_index is not None:
        try:
            shard_index = int(args.shard_index)
        except (TypeError, ValueError) as error:
            raise S27ExecutionBlocked("S27_WORKER_SHARD_INDEX_INVALID") from error
        shards = load_s27_shard_plan(
            root,
            plan,
            _logical(root, args.run_root, field="run_root"),
            run_id=args.run_id,
            cell_id=args.cell_id,
            shard_plan_ref=args.shard_plan_ref,
        )
        if shard_index < 0 or shard_index >= len(shards):
            raise S27ExecutionBlocked("S27_WORKER_SHARD_INDEX_OUT_OF_RANGE")
        shard = shards[shard_index]
        if shard.gpu_uuid != args.gpu_uuid:
            raise S27ExecutionBlocked("S27_WORKER_SHARD_GPU_BINDING_INVALID")
        shard_count = shard.shard_count
        shard_unit_ids = shard.unit_ids
    execution = _load_execution(root, args.execution_evidence_ref)
    execution_g24b = [gate for gate in execution.prerequisite_gates if gate.gate_id == "stage2.G2.4b"]
    if len(execution_g24b) != 1 or execution_g24b[0].artifact_hash != plan.frozen_inputs.g24b_gate_hash:
        raise S27ExecutionBlocked("S27_WORKER_EXECUTION_G24B_BINDING_INVALID")
    materialized = load_s27_materialized_inputs(root, args.materialization_index_ref)[args.cell_id]
    request = _build_request(root, cell_id=args.cell_id, materialization_index_ref=args.materialization_index_ref, execution_evidence_ref=args.execution_evidence_ref)
    run_root = _logical(root, args.run_root, field="run_root")
    cell_status_root = run_root / "waves" / args.cell_id.replace(":", "__")
    status_suffix = "status.json" if shard_index is None else f"shard-{shard_index:02d}-status.json"
    status_store = S27StatusStore(cell_status_root / status_suffix, run_id=f"{args.run_id}-{args.cell_id.replace(':', '-')}{'' if shard_index is None else f'-shard-{shard_index:02d}'}", plan_hash=plan.artifact_hash)
    if (cell_status_root / status_suffix).exists():
        status_store.require_recoverable()
    status_store.publish(S27DetachedStatus(status_store.run_id, plan.artifact_hash, "PREPARED", 0, _now()))
    status_store.publish(S27DetachedStatus(status_store.run_id, plan.artifact_hash, "RUNNING", 0, _now(), os.getpid(), args.gpu_uuid, time.time()))
    try:
        result = S27ProductionWorker(
            plan=plan,
            run_id=args.run_id,
            cell_id=args.cell_id,
            gpu_uuid=args.gpu_uuid,
            data_root=root,
            run_root=run_root,
            request=request,
            materialized_input=materialized,
            unit_ids=shard_unit_ids,
            shard_index=shard_index,
            shard_count=shard_count,
        ).run()
    except Exception as error:
        status_store.publish(S27DetachedStatus(status_store.run_id, plan.artifact_hash, "FAILED", 1, _now(), terminal_reason=f"{type(error).__name__}:{error}"))
        raise
    terminal = "SEALED" if result.get("status") in {"SEALED", "SHARD_SEALED"} else "FAILED"
    status_store.publish(S27DetachedStatus(status_store.run_id, plan.artifact_hash, terminal, 1, _now(), terminal_reason=str(result.get("status"))))
    return result


def _detach(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    run_root = _logical(root, args.run_root, field="run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    pid_path = run_root / "launcher.pid.json"
    if pid_path.exists():
        value = load_canonical_json(pid_path)
        if isinstance(value, Mapping) and isinstance(value.get("pid"), int):
            try:
                os.kill(int(value["pid"]), 0)
            except OSError:
                pass
            else:
                raise S27ExecutionBlocked(f"S27_DETACHED_LAUNCH_ALREADY_RUNNING:{value['pid']}")
    child = [str(item) for item in sys.argv[1:] if item != "--detach"]
    log_path = run_root / "launcher.log"
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [str(args.python), str(Path(__file__).resolve()), *child],
            cwd=_REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "schema_version": "stage2-s27-detached-launch-v1",
        "pid": int(process.pid),
        "run_id": args.run_id,
        "plan_ref": args.plan_ref,
        "run_root": str(run_root),
        "log_ref": str(log_path),
        "status_ref": str(run_root / "launcher-status.json"),
        "recovery_command": "--wait",
        "confirmatory_draws_generated": False,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(pid_path, payload)
    return payload


def _status(args: argparse.Namespace, *, wait: bool) -> int:
    root = args.data_root.resolve()
    path = _logical(root, args.run_root, field="run_root") / "launcher-status.json"
    deadline = None if args.timeout_seconds is None else time.monotonic() + float(args.timeout_seconds)
    while True:
        if path.exists():
            value = load_canonical_json(path)
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            if not wait or (isinstance(value, Mapping) and value.get("status") in {"SEALED", "FAILED"}):
                return 0 if not isinstance(value, Mapping) or value.get("status") == "SEALED" else 3
        if not wait:
            return 4
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(max(0.1, float(args.poll_seconds)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.7/G2.5 production launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true")
    action.add_argument("--detach", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--plan-ref", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="s207-formal-g25")
    parser.add_argument("--materialization-index-ref", required=True)
    parser.add_argument("--execution-evidence-ref", required=True)
    parser.add_argument("--gpu-inventory-json", type=Path)
    parser.add_argument("--repository", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--cell-id")
    parser.add_argument("--shard-plan-ref")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--throughput-sequences-per-second", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.detach:
            print(json.dumps(_detach(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.status:
            return _status(args, wait=False)
        if args.wait:
            return _status(args, wait=True)
        if args.preflight:
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.worker:
            if not args.cell_id or not args.gpu_uuid:
                raise S27ExecutionBlocked("S27_WORKER_REQUIRES_CELL_AND_GPU")
            print(json.dumps(_worker(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        # --execute is the only foreground launcher mode.  It repeats the
        # read-only preflight immediately before creating a child process.
        _preflight(args)
        inventory = _load_inventory(args.gpu_inventory_json)
        result = S27DetachedLauncher(
            data_root=args.data_root,
            plan_ref=args.plan_ref,
            run_root=args.run_root,
            run_id=args.run_id,
            python=args.python,
            launcher_script=Path(__file__).resolve(),
            materialization_index_ref=args.materialization_index_ref,
            execution_evidence_ref=args.execution_evidence_ref,
            approved_inventory=inventory,
        ).execute()
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (S27ExecutionBlocked, S27G25Blocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.7/G2.5 blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
