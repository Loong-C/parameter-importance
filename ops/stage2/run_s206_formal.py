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
    build_global_pilot_mapping,
    reduce_blinded_pilot,
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


def _execute(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
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
    _write_status(output, {"run_id": args.run_id, "stage": "PREFLIGHT_PASS", "formal_eligible": True, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": []})
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
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_RUNNING", "formal_eligible": True, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": records})
        _write_status(output, {"run_id": args.run_id, "stage": "WAVE_A_COMPLETE", "formal_eligible": True, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": records})
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(_run_anchor, args=args, anchor_id=anchor_id, gpu_uuid=CELL_GPU_BINDINGS[anchor_id], status_root=operations): anchor_id
                for anchor_id in ANCHOR_IDS[4:]
            }
            for future in as_completed(futures):
                records.extend(future.result())
                _write_status(output, {"run_id": args.run_id, "stage": "WAVE_B_RUNNING", "formal_eligible": True, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": records})
    except Exception as error:
        _write_status(output, {"run_id": args.run_id, "stage": "BLOCKED_EXECUTION", "formal_eligible": False, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": records, "reason": f"{type(error).__name__}:{error}"})
        raise
    _write_status(output, {"run_id": args.run_id, "stage": "PILOT_COMPLETE", "formal_eligible": True, "preflight": preflight, "pilot_mapping_ref": str(mapping_output), "pilot_mapping_hash": mapping.artifact_hash, "completed_cells": records, "expected_cell_count": 24, "confirmatory_draws_generated": False})
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.6/G2.4b pilot launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--reduce", action="store_true")
    action.add_argument("--wait", action="store_true")
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
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
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
