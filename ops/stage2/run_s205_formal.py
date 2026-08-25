"""Detached S2.5/G2.4a production launcher.

``--preflight`` is read-only and revalidates the fresh S2.4/G2.3 rebind,
sampling plan and preregistered pilot B/M/R plan.  ``--execute`` reruns that
preflight immediately before starting six UUID-bound workers through a
deterministic four-GPU LPT queue;
the workers call the S2.5-only strict runner and never create draws.  A final
G2.4a object is published only after all six immutable cell summaries pass.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
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

from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_s25_formal import (
    APPROVED_GPU_UUIDS,
    EXPECTED_CELL_IDS,
    S25ExecutionBlocked,
    S25FormalRunner,
    load_s25_rebind_plan,
    preflight_s25,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, value: str, *, field: str, allow_missing: bool = False) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise S25ExecutionBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE")
    result = (root / path).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise S25ExecutionBlocked(f"{field}:PATH_ESCAPE") from error
    if not allow_missing and not result.exists():
        raise S25ExecutionBlocked(f"{field}:MISSING")
    return result


def _inventory(path: Path | None) -> list[dict[str, object]]:
    if path is not None:
        value = load_canonical_json(path.resolve())
        rows = value.get("gpus") if isinstance(value, Mapping) else value
        if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
            raise S25ExecutionBlocked("S205_GPU_INVENTORY_JSON_INVALID")
        return [dict(row) for row in rows]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise S25ExecutionBlocked(f"S205_GPU_INVENTORY_UNAVAILABLE:{type(error).__name__}") from error
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) >= 2:
            rows.append({"uuid": fields[0], "pci_bus_id": fields[1]})
    if not rows:
        raise S25ExecutionBlocked("S205_GPU_INVENTORY_EMPTY")
    return rows


def _validate_inventory(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    by_uuid = {str(row.get("uuid")): row for row in rows}
    missing = [uuid for uuid in APPROVED_GPU_UUIDS if uuid not in by_uuid]
    if missing:
        raise S25ExecutionBlocked(f"S205_APPROVED_GPU_MISSING:{','.join(missing)}")
    for uuid in APPROVED_GPU_UUIDS:
        if str(by_uuid[uuid].get("pci_bus_id", "")).lower() == "0000:50:00.0":
            raise S25ExecutionBlocked("S205_APPROVED_GPU_BOUND_TO_EXCLUDED_PCI")
    return {"approved_gpu_uuids": list(APPROVED_GPU_UUIDS), "inventory": [dict(row) for row in rows]}


def _write_status(path: Path, payload: Mapping[str, object]) -> None:
    value = {"schema_version": "stage2-s205-formal-status-v1", "updated_at": _now(), **dict(payload)}
    write_canonical_json(path, value)


class _S205DynamicLPTQueue:
    """Deterministic four-GPU queue for the six frozen S2.5 cells."""

    _MODEL_RANK = {"pythia-14m": 1, "pythia-31m-deduped": 2}
    _STAGE_RANK = {"initialization": 1, "early": 2, "mid_late": 3}

    def __init__(self, cell_ids: Sequence[str], gpu_ids: Sequence[str]) -> None:
        if tuple(cell_ids) != tuple(EXPECTED_CELL_IDS):
            raise ValueError("S205_QUEUE_CELL_SET_MISMATCH")
        if tuple(gpu_ids) != tuple(APPROVED_GPU_UUIDS):
            raise ValueError("S205_QUEUE_GPU_SET_MISMATCH")
        self._pending = deque(sorted(cell_ids, key=self._lpt_key))
        self._free = deque(gpu_ids)
        self._active: dict[str, str] = {}
        self._completed: list[str] = []

    @classmethod
    def _lpt_key(cls, cell_id: str) -> tuple[int, int, int]:
        model, stage = cell_id.split(":", 1)
        return (-cls._MODEL_RANK[model], -cls._STAGE_RANK[stage], EXPECTED_CELL_IDS.index(cell_id))

    def fill(self) -> tuple[tuple[str, str], ...]:
        assignments: list[tuple[str, str]] = []
        while self._pending and self._free:
            cell_id = self._pending.popleft()
            gpu_uuid = self._free.popleft()
            self._active[cell_id] = gpu_uuid
            assignments.append((cell_id, gpu_uuid))
        return tuple(assignments)

    def complete(self, cell_id: str) -> tuple[tuple[str, str], ...]:
        if cell_id not in self._active:
            raise ValueError(f"S205_QUEUE_CELL_NOT_ACTIVE:{cell_id}")
        gpu_uuid = self._active.pop(cell_id)
        self._completed.append(cell_id)
        free = [*self._free, gpu_uuid]
        self._free = deque(sorted(free, key=APPROVED_GPU_UUIDS.index))
        return self.fill()

    @property
    def active(self) -> dict[str, str]:
        return dict(self._active)

    @property
    def pending(self) -> tuple[str, ...]:
        return tuple(self._pending)

    @property
    def completed(self) -> tuple[str, ...]:
        return tuple(self._completed)


def _preflight(args: argparse.Namespace) -> dict[str, object]:
    root = args.data_root.resolve()
    plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
    if args.artifact_root != plan.get("s205_output_root"):
        raise S25ExecutionBlocked("S205_ARTIFACT_ROOT_MUST_MATCH_REBIND")
    if args.operations_root != plan.get("operations_root"):
        raise S25ExecutionBlocked("S205_OPERATIONS_ROOT_MUST_MATCH_REBIND")
    result = preflight_s25(
        root,
        rebind_ref=args.s205_rebind_ref,
        sampling_ref=args.sampling_plan_ref,
        experiment_plan_ref=args.experiment_plan_ref,
        artifact_root=args.artifact_root,
    )
    inventory = _validate_inventory(_inventory(args.gpu_inventory_json))
    result["gpu"] = inventory
    result["launcher"] = {
        "repository": str(args.repository.resolve()),
        "data_root": str(root),
        "run_id": args.run_id,
    }
    return result


def _worker(args: argparse.Namespace) -> dict[str, object]:
    if not args.cell_id or not args.gpu_uuid or args.cell_id not in EXPECTED_CELL_IDS:
        raise S25ExecutionBlocked("S205_WORKER_CELL_AND_GPU_REQUIRED")
    if args.gpu_uuid not in APPROVED_GPU_UUIDS:
        raise S25ExecutionBlocked("S205_WORKER_GPU_UNAPPROVED")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible != args.gpu_uuid:
        raise S25ExecutionBlocked("S205_WORKER_GPU_BINDING_MISMATCH")
    root = args.data_root.resolve()
    preflight = _preflight(args)
    plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
    from param_importance_nlp.experiments.stage2_s25_formal import (
        _load_object,
        load_s25_experiment_plan,
    )
    _, sampling_value = _load_object(root, args.sampling_plan_ref, field="sampling_plan_ref")
    from param_importance_nlp.experiments.sampling import SamplingPlan

    sampling = SamplingPlan.from_mapping(sampling_value)
    experiment = load_s25_experiment_plan(root, args.experiment_plan_ref)
    runner = S25FormalRunner(
        data_root=root,
        rebind_plan=plan,
        experiment_plan=experiment,
        sampling_plan=sampling,
        artifact_root=_logical(root, args.artifact_root, field="artifact_root", allow_missing=True),
    )
    result = runner.run_cell(args.cell_id)
    return {"status": "CELL_COMPLETE", "cell_id": args.cell_id, "gpu_uuid": args.gpu_uuid, "preflight": preflight, "cell": result}


def _execute(args: argparse.Namespace) -> dict[str, object]:
    preflight = _preflight(args)
    root = args.data_root.resolve()
    operations = _logical(root, args.operations_root, field="operations_root", allow_missing=True)
    operations.mkdir(parents=True, exist_ok=True)
    status_path = operations / "status.json"
    status: dict[str, object] = {
        "run_id": args.run_id,
        "stage": "PREPARED",
        "formal_eligible": True,
        "preflight": preflight,
        "completed_cells": [],
        "confirmatory_draws_generated": False,
    }
    _write_status(status_path, status)
    base = [
        str(args.python), str(Path(__file__).resolve()), "--worker",
        "--data-root", str(root), "--s205-rebind-ref", args.s205_rebind_ref,
        "--sampling-plan-ref", args.sampling_plan_ref, "--experiment-plan-ref", args.experiment_plan_ref,
        "--artifact-root", args.artifact_root, "--operations-root", args.operations_root,
        "--run-id", args.run_id,
    ]
    if args.gpu_inventory_json is not None:
        base += ["--gpu-inventory-json", str(args.gpu_inventory_json.resolve())]
    records: list[dict[str, object]] = []
    queue = _S205DynamicLPTQueue(EXPECTED_CELL_IDS, APPROVED_GPU_UUIDS)
    try:
        with ThreadPoolExecutor(max_workers=len(APPROVED_GPU_UUIDS)) as pool:
            futures = {}

            def submit(assignments: Sequence[tuple[str, str]]) -> None:
                for cell_id, gpu_uuid in assignments:
                    command = [*base, "--cell-id", cell_id, "--gpu-uuid", gpu_uuid]
                    log = operations / "workers" / f"{cell_id.replace(':', '__')}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    handle = log.open("a", encoding="utf-8")
                    future = pool.submit(
                        subprocess.run,
                        command,
                        cwd=args.repository.resolve(),
                        env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu_uuid, "NVIDIA_VISIBLE_DEVICES": gpu_uuid, "PYTHONPATH": str(args.repository.resolve() / "src")},
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                    futures[future] = (cell_id, gpu_uuid, log, handle)

            submit(queue.fill())
            while futures:
                done, _ = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: futures[item][0]):
                    cell_id, gpu_uuid, log, handle = futures.pop(future)
                    completed = future.result()
                    handle.close()
                    record = {"cell_id": cell_id, "gpu_uuid": gpu_uuid, "returncode": completed.returncode, "log_ref": str(log.relative_to(root))}
                    records.append(record)
                    status.update({"stage": "RUNNING", "completed_cells": list(records)})
                    _write_status(status_path, status)
                    if completed.returncode != 0:
                        raise S25ExecutionBlocked(f"S205_CELL_WORKER_FAILED:{cell_id}:{completed.returncode}")
                    # Refill immediately on each individual completion; there is
                    # intentionally no second wave/barrier.
                    submit(queue.complete(cell_id))
        if queue.pending or queue.active or set(queue.completed) != set(EXPECTED_CELL_IDS):
            raise S25ExecutionBlocked("S205_QUEUE_DID_NOT_COMPLETE_ALL_CELLS")
        from param_importance_nlp.experiments.stage2_s25_formal import _load_object
        _, sampling_value = _load_object(root, args.sampling_plan_ref, field="sampling_plan_ref")
        from param_importance_nlp.experiments.sampling import SamplingPlan

        plan = load_s25_rebind_plan(root, args.s205_rebind_ref)
        runner = S25FormalRunner(
            data_root=root,
            rebind_plan=plan,
            experiment_plan=load_s25_experiment_plan(root, args.experiment_plan_ref),
            sampling_plan=SamplingPlan.from_mapping(sampling_value),
            artifact_root=_logical(root, args.artifact_root, field="artifact_root", allow_missing=True),
        )
        gate = runner.run_all()
        status.update({"stage": "G2.4A_PASS" if gate.get("status") == "PASS" else "G2.4A_BLOCKED", "formal_eligible": gate.get("formal_eligible"), "completed_cells": list(records), "gate_ref": str((_logical(root, args.artifact_root, field="artifact_root", allow_missing=True) / "g2.4a-evaluation.json").relative_to(root)), "gate_hash": gate.get("artifact_hash")})
        _write_status(status_path, status)
        return status
    except Exception as error:
        status.update({"stage": "BLOCKED", "formal_eligible": False, "completed_cells": list(records), "reason": f"{type(error).__name__}:{error}"})
        _write_status(status_path, status)
        raise


def _status(args: argparse.Namespace, *, wait: bool) -> int:
    path = _logical(args.data_root.resolve(), f"{args.operations_root}/status.json", field="status")
    deadline = None if args.timeout_seconds is None else time.monotonic() + float(args.timeout_seconds)
    while True:
        if path.exists():
            value = load_canonical_json(path)
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            if not wait or (isinstance(value, Mapping) and value.get("stage") in {"G2.4A_PASS", "G2.4A_BLOCKED", "BLOCKED"}):
                return 0 if isinstance(value, Mapping) and value.get("stage") == "G2.4A_PASS" else 3
        if not wait or (deadline is not None and time.monotonic() >= deadline):
            return 4
        time.sleep(max(0.1, float(args.poll_seconds)))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.5/G2.4a production launcher")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--worker", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s205-rebind-ref", required=True)
    parser.add_argument("--sampling-plan-ref", required=True)
    parser.add_argument("--experiment-plan-ref", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--operations-root", required=True)
    parser.add_argument("--run-id", default="s205-formal-g24a")
    parser.add_argument("--gpu-inventory-json", type=Path)
    parser.add_argument("--repository", type=Path, default=_REPOSITORY_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--cell-id")
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.status:
            return _status(args, wait=False)
        if args.wait:
            return _status(args, wait=True)
        if args.preflight:
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.worker:
            print(json.dumps(_worker(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        print(json.dumps(_execute(args), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (S25ExecutionBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.5/G2.4a blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
