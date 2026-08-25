"""Detached S2.6/G2.4b pilot launcher.

The launcher has a strict read-only preflight.  ``--execute`` calls that same
preflight immediately before starting any TaskRuntime child and then runs the
24 prepared anchor/B configs in four-GPU waves.  It never creates a
confirmatory mapping; matrix qualification and confirmatory mapping are
separate functions in ``stage2_s206_formal``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_UUIDS,
    CELL_GPU_BINDINGS,
    EXCLUDED_PCI,
    PILOT_B_VALUES,
    S206PreflightSpec,
    S206PreparationBlocked,
    GlobalPilotMappingManifest,
    BlindPilotMeasurement,
    build_g24b_gate,
    build_formal_cell_specs,
    build_global_pilot_mapping,
    qualify_formal_matrix,
    reduce_blinded_pilot,
    run_formal_pilot_cell,
    strict_preflight,
)
from param_importance_nlp.experiments.stage2_pilot import CostSemantics
from param_importance_nlp.experiments.sampling import SamplingPlan


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, value: str, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S206PreparationBlocked(f"{field}:INVALID_LOGICAL_PATH")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE")
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S206PreparationBlocked(f"{field}:PATH_ESCAPE") from error
    return result


def _load_inventory(path: Path | None) -> list[dict[str, object]]:
    if path is not None:
        value = load_canonical_json(path.resolve())
        if isinstance(value, Mapping) and isinstance(value.get("gpus"), list):
            rows = value["gpus"]
        else:
            rows = value
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise S206PreparationBlocked("GPU_INVENTORY_JSON_INVALID")
        return [dict(row) for row in rows]
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S206PreparationBlocked(f"GPU_INVENTORY_UNAVAILABLE:{type(error).__name__}") from error
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "uuid": parts[0],
                "pci_bus_id": parts[1],
                "memory_used_mib": int(parts[2]) if parts[2].isdigit() else parts[2],
                "utilization_gpu": int(parts[3]) if parts[3].isdigit() else parts[3],
            }
        )
    if not rows:
        raise S206PreparationBlocked("GPU_INVENTORY_EMPTY")
    return rows


def _write_status(path: Path, payload: Mapping[str, object]) -> None:
    value = {"schema_version": "stage2-s206-formal-detached-status-v1", "updated_at": _now(), **payload}
    write_canonical_json(path, value)


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    inventory = _load_inventory(args.gpu_inventory_json)
    result = strict_preflight(
        S206PreflightSpec(
            data_root=root,
            s204_root=args.s204_root,
            g23_ref=args.g23_evaluation,
            g24a_ref=args.g24a_evaluation,
        ),
        gpu_inventory=inventory,
    )
    if tuple(result["gpu"]["approved_gpu_uuids"]) != APPROVED_GPU_UUIDS:  # type: ignore[index]
        raise S206PreparationBlocked("APPROVED_GPU_BINDING_DRIFT")
    result["excluded_pci"] = EXCLUDED_PCI
    result["launcher"] = {
        "repository": str(args.repository.resolve()),
        "data_root": str(root),
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "cell_gpu_bindings": dict(CELL_GPU_BINDINGS),
    }
    return result


def _config_path(root: Path, config_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    base = _logical(root, config_root, field="config_root")
    candidates = (base / f"{component}.json", base / component / "resolved-config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise S206PreparationBlocked(f"CONFIG_MISSING:{component}")


def _environment_path(root: Path, environment_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    base = _logical(root, environment_root, field="environment_root")
    candidates = (base / f"{component}.json", base / component / "environment.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise S206PreparationBlocked(f"ENVIRONMENT_MISSING:{component}")


def _result_path(root: Path, result_root: str, anchor_id: str, batch_size: int) -> Path:
    component = f"{anchor_id.replace('.', '__')}__b{batch_size:03d}"
    return _logical(root, result_root, field="result_root") / f"{component}.json"


def _run_anchor(
    *,
    args: argparse.Namespace,
    anchor_id: str,
    gpu_uuid: str,
    status_root: Path,
) -> list[dict[str, object]]:
    root = args.data_root.resolve()
    python = str(args.python)
    records: list[dict[str, object]] = []
    for batch_size in PILOT_B_VALUES:
        config = _config_path(root, args.config_root, anchor_id, batch_size)
        environment = _environment_path(root, args.environment_root, anchor_id, batch_size)
        result = _result_path(root, args.result_root, anchor_id, batch_size)
        result.parent.mkdir(parents=True, exist_ok=True)
        if result.exists():
            try:
                existing = load_canonical_json(result)
            except (OSError, TypeError, ValueError) as error:
                raise S206PreparationBlocked(f"RESULT_NOT_CANONICAL:{result}") from error
            if not isinstance(existing, Mapping) or existing.get("status") not in {"PASS", "COMPLETE"}:
                raise S206PreparationBlocked(f"RESULT_EXISTS_REQUIRES_EXPLICIT_RESUME:{result}")
            records.append({"anchor_id": anchor_id, "batch_size": batch_size, "gpu_uuid": gpu_uuid, "status": "RESUMED_EXISTING", "result_ref": str(result)})
            continue
        env = os.environ.copy()
        env.update(
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": gpu_uuid,
                "NVIDIA_VISIBLE_DEVICES": gpu_uuid,
                "PYTHONPATH": str(args.repository.resolve() / "src"),
            }
        )
        command = [
            python,
            "-m",
            "param_importance_nlp",
            "task",
            "run",
            "--config",
            str(config),
            "--environment",
            str(environment),
            "--result",
            str(result),
        ]
        log = status_root / f"{anchor_id.replace('.', '__')}__b{batch_size:03d}.log"
        with log.open("a", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=args.repository.resolve(), env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        record = {
            "anchor_id": anchor_id,
            "batch_size": batch_size,
            "gpu_uuid": gpu_uuid,
            "status": "PASS" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "config_ref": str(config),
            "environment_ref": str(environment),
            "result_ref": str(result),
            "log_ref": str(log),
        }
        records.append(record)
        if completed.returncode != 0:
            raise S206PreparationBlocked(f"S206_CELL_FAILED:{anchor_id}:B{batch_size}:{completed.returncode}")
    return records


def _load_formal_execution(args: argparse.Namespace) -> FormalExecutionEvidence:
    """Load the explicit formal authorization consumed by every child cell."""

    if not args.execution_evidence_ref:
        raise S206PreparationBlocked("EXECUTE_REQUIRES_FORMAL_EXECUTION_EVIDENCE")
    path = _logical(args.data_root.resolve(), args.execution_evidence_ref, field="execution_evidence_ref")
    value = load_canonical_json(path)
    if not isinstance(value, Mapping):
        raise S206PreparationBlocked("EXECUTION_EVIDENCE_NOT_OBJECT")
    try:
        execution = FormalExecutionEvidence.from_mapping(dict(value))
        execution.require_for_stage(2)
    except (TypeError, ValueError, S206PreparationBlocked) as error:
        raise S206PreparationBlocked(f"EXECUTION_EVIDENCE_INVALID:{error}") from error
    return execution


def _execute(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
    execution = _load_formal_execution(args)
    root = args.data_root.resolve()
    operations = _logical(root, args.operations_root, field="operations_root")
    operations.mkdir(parents=True, exist_ok=True)
    output = operations / "status.json"
    if not args.sampling_plan_ref or not args.pilot_mapping_output:
        raise S206PreparationBlocked("EXECUTE_REQUIRES_SAMPLING_PLAN_AND_PILOT_MAPPING_OUTPUT")
    sampling_ref = _logical(root, args.sampling_plan_ref, field="sampling_plan_ref")
    mapping_output = _logical(root, args.pilot_mapping_output, field="pilot_mapping_output")
    sampling_value = load_canonical_json(sampling_ref)
    if not isinstance(sampling_value, Mapping):
        raise S206PreparationBlocked("SAMPLING_PLAN_NOT_OBJECT")
    try:
        sampling = SamplingPlan.from_mapping(sampling_value)
    except (TypeError, ValueError) as error:
        raise S206PreparationBlocked(f"SAMPLING_PLAN_INVALID:{error}") from error
    mapping = build_global_pilot_mapping(sampling)
    if mapping_output.exists():
        existing = load_canonical_json(mapping_output)
        if not isinstance(existing, Mapping) or existing.get("artifact_hash") != mapping.artifact_hash:
            raise S206PreparationBlocked("PILOT_MAPPING_OUTPUT_CONFLICT")
    else:
        mapping_output.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(mapping_output, mapping.to_dict())
    cell_plan_dir = operations / "cell-plans"
    cell_specs = build_formal_cell_specs(
        mapping,
        execution,
        config_root=args.config_root,
        environment_root=args.environment_root,
        result_root=args.result_root,
    )
    cell_plan_refs: list[str] = []
    for spec in cell_specs:
        plan_path = cell_plan_dir / f"{spec.cell_id}.json"
        plan_payload = spec.to_dict()
        if plan_path.exists():
            if load_canonical_json(plan_path) != plan_payload:
                raise S206PreparationBlocked(f"CELL_PLAN_CONFLICT:{spec.cell_id}")
        else:
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_json(plan_path, plan_payload)
        cell_plan_refs.append(str(plan_path))
    _write_status(output, {"run_id": args.run_id, "stage": "PREFLIGHT_PASS", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": []})
    records: list[dict[str, object]] = []
    try:
        # Wave A occupies the four approved cards.  Wave B starts only after
        # all Wave A cells have committed, preventing same-card concurrency.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                pool.submit(_run_anchor, args=args, anchor_id=anchor_id, gpu_uuid=CELL_GPU_BINDINGS[anchor_id], status_root=operations): anchor_id
                for anchor_id in ANCHOR_IDS[:4]
            }
            for future in as_completed(futures):
                records.extend(future.result())
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_RUNNING", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
        _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_COMPLETE", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_anchor, args=args, anchor_id=anchor_id, gpu_uuid=CELL_GPU_BINDINGS[anchor_id], status_root=operations): anchor_id
                for anchor_id in ANCHOR_IDS[4:]
            }
            for future in as_completed(futures):
                records.extend(future.result())
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_B_RUNNING", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records})
    except Exception as error:
        _write_status(output, {"run_id": args.run_id, "stage": "BLOCKED_EXECUTION", "formal_eligible": False, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records, "reason": f"{type(error).__name__}:{error}"})
        raise
    _write_status(output, {"run_id": args.run_id, "stage": "PILOT_COMPLETE", "formal_eligible": True, "preflight": preflight, "execution_evidence_hash": execution.artifact_hash, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "cell_plan_refs": cell_plan_refs, "completed_cells": records, "expected_cell_count": 24, "confirmatory_draws_generated": False})
    return load_canonical_json(output)  # type: ignore[return-value]


def _reduce(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    if not args.pilot_mapping_ref or not args.measurements_ref or not args.costs_ref or not args.report_output:
        raise S206PreparationBlocked("REDUCE_REQUIRES_MAPPING_MEASUREMENTS_COSTS_AND_OUTPUT")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    measurements_path = _logical(root, args.measurements_ref, field="measurements_ref")
    costs_path = _logical(root, args.costs_ref, field="costs_ref")
    report_path = _logical(root, args.report_output, field="report_output")
    raw_mapping = load_canonical_json(mapping_path)
    raw_measurements = load_canonical_json(measurements_path)
    raw_costs = load_canonical_json(costs_path)
    if not isinstance(raw_mapping, Mapping) or not isinstance(raw_costs, Mapping):
        raise S206PreparationBlocked("REDUCE_INPUT_OBJECT_REQUIRED")
    measurement_values = raw_measurements.get("measurements") if isinstance(raw_measurements, Mapping) else raw_measurements
    if not isinstance(measurement_values, list) or not all(isinstance(item, Mapping) for item in measurement_values):
        raise S206PreparationBlocked("REDUCE_MEASUREMENTS_ARRAY_REQUIRED")
    costs = CostSemantics(
        scientific_equal_sample_cost=dict(raw_costs["scientific_equal_sample_cost"]),  # type: ignore[arg-type]
        isolated_estimator_cost=dict(raw_costs["isolated_estimator_cost"]),  # type: ignore[arg-type]
        online_training_incremental_cost=dict(raw_costs["online_training_incremental_cost"]),  # type: ignore[arg-type]
        cost_io_quiescent=raw_costs["cost_io_quiescent"],  # type: ignore[arg-type]
    )
    report = reduce_blinded_pilot(
        GlobalPilotMappingManifest.from_mapping(dict(raw_mapping)),
        tuple(BlindPilotMeasurement.from_mapping(dict(item)) for item in measurement_values),
        cost_semantics=costs,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(report_path, report.to_dict())
    return report.to_dict()


def _synthetic_cell(args: argparse.Namespace) -> dict[str, object]:
    """Direct bridge entry point used for a deterministic formal-cell smoke test."""

    root = args.data_root.resolve()
    # Keep the direct bridge under the same G2.3/G2.4a and approved-GPU
    # preflight as the long-running launcher; synthetic execution is not a
    # bypass around those gates.
    _preflight(args)
    required = (
        args.pilot_mapping_ref,
        args.execution_evidence_ref,
        args.cell_artifact_root,
        args.cell_output,
        args.reference_json,
        args.cell_sizing_ref,
    )
    if not all(required) or args.cell_anchor is None or args.cell_batch_size is None:
        raise S206PreparationBlocked("SYNTHETIC_CELL_REQUIRES_MAPPING_EXECUTION_CELL_AND_SIZING_INPUTS")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    mapping_value = load_canonical_json(mapping_path)
    if not isinstance(mapping_value, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_MAPPING_NOT_OBJECT")
    mapping = GlobalPilotMappingManifest.from_mapping(dict(mapping_value))
    execution = _load_formal_execution(args)
    try:
        cell = next(
            item for item in mapping.cells
            if item.anchor_id == args.cell_anchor and item.batch_size == args.cell_batch_size
        )
    except StopIteration as error:
        raise S206PreparationBlocked("SYNTHETIC_CELL_NOT_IN_MAPPING") from error
    reference_value = load_canonical_json(_logical(root, args.reference_json, field="reference_json"))
    sizing_value = load_canonical_json(_logical(root, args.cell_sizing_ref, field="cell_sizing_ref"))
    if not isinstance(reference_value, Mapping) or not isinstance(sizing_value, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_REFERENCE_OR_SIZING_NOT_OBJECT")
    delta = sizing_value.get("delta_sci_by_endpoint")
    half_width = sizing_value.get("reference_half_width_by_endpoint")
    if not isinstance(delta, Mapping) or not isinstance(half_width, Mapping):
        raise S206PreparationBlocked("SYNTHETIC_CELL_SIZING_FIELDS_REQUIRED")
    try:
        from param_importance_nlp.experiments.stage2_formal import _vector_digest
        from param_importance_nlp.providers.synthetic import SyntheticGradientProvider

        shapes = {
            str(name): tuple(len(value) if isinstance(value, list) else 1 for _ in [0])
            for name, value in reference_value.items()
        }
        # The synthetic bridge is intentionally limited to flat vectors.  A
        # production provider must be supplied through run_formal_pilot_cell.
        if any(not isinstance(value, list) or not value for value in reference_value.values()):
            raise ValueError("synthetic reference vectors must be non-empty lists")
        provider = SyntheticGradientProvider.from_location_scale(
            parameter_shapes=shapes,
            sample_count=int(args.synthetic_sample_count),
            seed=int(args.synthetic_seed),
        )
        reference_hash = str(args.reference_hash) if args.reference_hash else _vector_digest(reference_value)
        result = run_formal_pilot_cell(
            cell,
            mapping=mapping,
            provider=provider,
            execution=execution,
            reference=reference_value,
            reference_hash=reference_hash,
            artifact_root=_logical(root, args.cell_artifact_root, field="cell_artifact_root"),
            delta_sci_by_endpoint=dict(delta),
            reference_half_width_by_endpoint=dict(half_width),
            resource_within_budget=True,
            cost_io_quiescent=True,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise S206PreparationBlocked(f"SYNTHETIC_CELL_BRIDGE_INVALID:{error}") from error
    output = _logical(root, args.cell_output, field="cell_output")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(output, result.to_dict())
    return result.to_dict()


def _qualify(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    if not args.pilot_mapping_ref or not args.measurements_ref or not args.costs_ref or not args.execution_evidence_ref or not args.matrix_output or not args.gate_output:
        raise S206PreparationBlocked("QUALIFY_REQUIRES_REDUCER_INPUTS_EXECUTION_AND_OUTPUTS")
    mapping_path = _logical(root, args.pilot_mapping_ref, field="pilot_mapping_ref")
    measurements_path = _logical(root, args.measurements_ref, field="measurements_ref")
    costs_path = _logical(root, args.costs_ref, field="costs_ref")
    execution_path = _logical(root, args.execution_evidence_ref, field="execution_evidence_ref")
    matrix_path = _logical(root, args.matrix_output, field="matrix_output")
    gate_path = _logical(root, args.gate_output, field="gate_output")
    raw_mapping = load_canonical_json(mapping_path)
    raw_measurements = load_canonical_json(measurements_path)
    raw_costs = load_canonical_json(costs_path)
    raw_execution = load_canonical_json(execution_path)
    if not all(isinstance(value, Mapping) for value in (raw_mapping, raw_costs, raw_execution)):
        raise S206PreparationBlocked("QUALIFY_INPUT_OBJECT_REQUIRED")
    measurement_values = raw_measurements.get("measurements") if isinstance(raw_measurements, Mapping) else raw_measurements
    if not isinstance(measurement_values, list) or not all(isinstance(item, Mapping) for item in measurement_values):
        raise S206PreparationBlocked("QUALIFY_MEASUREMENTS_ARRAY_REQUIRED")
    costs = CostSemantics(
        scientific_equal_sample_cost=dict(raw_costs["scientific_equal_sample_cost"]),  # type: ignore[index]
        isolated_estimator_cost=dict(raw_costs["isolated_estimator_cost"]),  # type: ignore[index]
        online_training_incremental_cost=dict(raw_costs["online_training_incremental_cost"]),  # type: ignore[index]
        cost_io_quiescent=raw_costs["cost_io_quiescent"],  # type: ignore[index]
    )
    mapping = GlobalPilotMappingManifest.from_mapping(dict(raw_mapping))
    report = reduce_blinded_pilot(
        mapping,
        tuple(BlindPilotMeasurement.from_mapping(dict(item)) for item in measurement_values),
        cost_semantics=costs,
    )
    execution = FormalExecutionEvidence.from_mapping(dict(raw_execution))
    refs = tuple(args.evidence_ref or (args.measurements_ref, args.pilot_mapping_ref, args.execution_evidence_ref))
    gate = build_g24b_gate(report, execution, evidence_refs=refs)
    matrix = qualify_formal_matrix(report, execution, gate, freeze_id=args.freeze_id)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(gate_path, gate.to_dict())
    write_canonical_json(matrix_path, matrix.to_dict())
    return {"status": "FORMAL_FROZEN", "formal_eligible": True, "gate_ref": str(gate_path), "matrix_ref": str(matrix_path), "gate": gate.to_dict(), "matrix": matrix.to_dict()}


def _wait(args: argparse.Namespace) -> int:
    status_path = _logical(args.data_root.resolve(), f"{args.operations_root}/status.json", field="status")
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    while True:
        if status_path.exists():
            value = load_canonical_json(status_path)
            if isinstance(value, Mapping):
                print(json.dumps(value, ensure_ascii=False, sort_keys=True))
                if value.get("stage") in {"PILOT_COMPLETE", "BLOCKED_EXECUTION"}:
                    return 0 if value.get("stage") == "PILOT_COMPLETE" else 3
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(args.poll_seconds)


def _detach(args: argparse.Namespace) -> int:
    """Re-exec the exact execute command in a new session and return its PID."""

    if not args.execute:
        raise S206PreparationBlocked("DETACH_REQUIRES_EXECUTE")
    operations = _logical(args.data_root.resolve(), args.operations_root, field="operations_root")
    operations.mkdir(parents=True, exist_ok=True)
    log_path = operations / "launcher.log"
    pid_path = operations / "launcher.pid.json"
    if pid_path.exists():
        try:
            existing = load_canonical_json(pid_path)
        except (OSError, TypeError, ValueError):
            existing = None
        if isinstance(existing, Mapping) and isinstance(existing.get("pid"), int):
            try:
                os.kill(int(existing["pid"]), 0)
            except OSError:
                pass
            else:
                raise S206PreparationBlocked(f"DETACHED_LAUNCH_ALREADY_RUNNING:{existing['pid']}")
    child_argv = [item for item in sys.argv[1:] if item != "--detach"]
    with log_path.open("ab") as handle:
        process = subprocess.Popen(
            [str(args.python), str(Path(__file__).resolve()), *child_argv],
            cwd=args.repository.resolve(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    payload = {
        "schema_version": "stage2-s206-detached-launch-v1",
        "pid": int(process.pid),
        "run_id": args.run_id,
        "operations_root": str(operations),
        "log_ref": str(log_path),
        "status_ref": str(operations / "status.json"),
        "recovery_command": "--wait",
        "confirmatory_draws_generated": False,
    }
    write_canonical_json(pid_path, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.6/G2.4b pilot launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--reduce", action="store_true")
    action.add_argument("--qualify", action="store_true")
    action.add_argument("--wait", action="store_true")
    action.add_argument("--synthetic-cell", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--s204-root", required=True)
    parser.add_argument("--g23-evaluation", required=True)
    parser.add_argument("--g24a-evaluation", required=True)
    parser.add_argument("--gpu-inventory-json", type=Path)
    parser.add_argument("--plan-output", type=Path)
    parser.add_argument("--run-id", default="s206-formal-g24b")
    parser.add_argument("--operations-root", required=True)
    parser.add_argument("--config-root")
    parser.add_argument("--environment-root")
    parser.add_argument("--result-root")
    parser.add_argument("--sampling-plan-ref")
    parser.add_argument("--pilot-mapping-output")
    parser.add_argument("--pilot-mapping-ref")
    parser.add_argument("--measurements-ref")
    parser.add_argument("--costs-ref")
    parser.add_argument("--report-output")
    parser.add_argument("--execution-evidence-ref")
    parser.add_argument("--matrix-output")
    parser.add_argument("--gate-output")
    parser.add_argument("--freeze-id", default="s206-formal-matrix-freeze")
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--detach", action="store_true", help="detach --execute and return a PID")
    parser.add_argument("--cell-anchor")
    parser.add_argument("--cell-batch-size", type=int)
    parser.add_argument("--cell-artifact-root")
    parser.add_argument("--cell-output")
    parser.add_argument("--reference-json")
    parser.add_argument("--reference-hash")
    parser.add_argument("--cell-sizing-ref")
    parser.add_argument("--synthetic-sample-count", type=int, default=4096)
    parser.add_argument("--synthetic-seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.detach:
            return _detach(args)
        if args.synthetic_cell:
            result = _synthetic_cell(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.wait:
            return _wait(args)
        if args.reduce:
            result = _reduce(args)
            if args.plan_output is not None:
                output = args.plan_output.resolve()
                output.parent.mkdir(parents=True, exist_ok=True)
                write_canonical_json(output, result)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.qualify:
            result = _qualify(args)
            print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.execute and not all((args.config_root, args.environment_root, args.result_root)):
            raise S206PreparationBlocked("EXECUTE_REQUIRES_CONFIG_ENVIRONMENT_RESULT_ROOTS")
        result = _execute(args) if args.execute else _preflight(args)
        if args.plan_output is not None:
            output = args.plan_output.resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_json(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, S206PreparationBlocked) as error:
        payload = {
            "schema_version": "stage2-s206-formal-preflight-v1",
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
