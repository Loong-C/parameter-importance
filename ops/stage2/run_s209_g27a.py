"""Detached production launcher for S2.9/G2.7a profiler validation.

``--prepare`` creates only a frozen measurement plan after binding G2.4b and
the sealed S2.7 manifest.  ``--preflight`` re-reads every producer artifact,
GPU inventory, and I/O evidence.  ``--execute`` is the sole mode allowed to
start a profiler worker; the worker command must emit actual measured rows.
``--detach``/``--status``/``--wait`` are control-plane operations and never
invent a result when a profiler or prerequisite is missing.
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
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_s209_runner import (
    S29RunnerBlocked,
    S29ProfilerRunner,
    S29StatusStore,
    load_s209_preflight,
    prepare_s209_plan,
    subprocess_profiler_executor,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _logical(root: Path, reference: str, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise S29RunnerBlocked(f"{field}:INVALID_LOGICAL_REFERENCE")
    parts = reference.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE")
    path = (root / Path(*parts)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise S29RunnerBlocked(f"{field}:PATH_ESCAPE") from error
    return path


def _load_optional(root: Path, reference: str | None, *, field: str) -> Any:
    if reference is None:
        return None
    path = _logical(root, reference, field=field)
    value = load_canonical_json(path)
    return value


def _pid_alive(pid: Any) -> bool:
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _claim_detach_lease(run_root: Path, *, run_id: str) -> Path:
    """Atomically reserve a run namespace before the detached child starts."""

    lease_path = run_root / "launcher.lease.json"
    pid_path = run_root / "launcher.pid.json"
    # A detached launch is a formal attempt identity, not a disposable mutex.
    # Once either manifest exists, the run-root cannot be fresh-launched again,
    # even if its previous owner has exited.  Reusing a dead PID would erase
    # provenance and permit a second measurement attempt under the same ID.
    if pid_path.exists():
        try:
            previous = load_canonical_json(pid_path)
        except Exception as error:
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_PID_INVALID") from error
        if not isinstance(previous, Mapping) or previous.get("run_id") != run_id:
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_PID_IDENTITY_MISMATCH")
        declared = previous.get("artifact_hash")
        body = {key: value for key, value in previous.items() if key != "artifact_hash"}
        if (
            previous.get("schema_version") != "stage2-s209-g27a-detached-launch-v1"
            or isinstance(previous.get("pid"), bool)
            or not isinstance(previous.get("pid"), int)
            or previous.get("pid") <= 0
            or not isinstance(declared, str)
            or canonical_json_hash(body) != declared
        ):
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_PID_INVALID")
        raise S29RunnerBlocked(f"S29_DETACHED_LAUNCH_ALREADY_RUNNING:{previous.get('pid')}")
    try:
        descriptor = os.open(str(lease_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            lease = load_canonical_json(lease_path)
        except Exception as error:
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_LEASE_INVALID") from error
        if not isinstance(lease, Mapping):
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_LEASE_INVALID")
        declared = lease.get("artifact_hash")
        body = {key: value for key, value in lease.items() if key != "artifact_hash"}
        if (
            lease.get("schema_version") != "stage2-s209-g27a-detached-lease-v1"
            or lease.get("run_id") != run_id
            or not isinstance(lease.get("owner_pid"), int)
            or lease.get("owner_pid") <= 0
            or not isinstance(declared, str)
            or canonical_json_hash(body) != declared
        ):
            raise S29RunnerBlocked("S29_DETACHED_LAUNCH_LEASE_INVALID")
        raise S29RunnerBlocked(f"S29_DETACHED_LAUNCH_ALREADY_RUNNING:{lease['owner_pid']}")
    payload = {
        "schema_version": "stage2-s209-g27a-detached-lease-v1",
        "owner_pid": os.getpid(),
        "run_id": run_id,
        "started_at": _now(),
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
    except BaseException:
        # Keep the lease as an append-only failed launch marker.  Removing it
        # would make a retry under the same run identity indistinguishable from
        # a new formal attempt.
        raise
    return lease_path


def _write_once(path: Path, value: Mapping[str, Any], *, field: str) -> None:
    """Publish one detached manifest without overwriting prior provenance."""

    if path.exists():
        try:
            existing = load_canonical_json(path)
        except Exception as error:
            raise S29RunnerBlocked(f"{field}:INVALID_EXISTING") from error
        if isinstance(existing, Mapping) and dict(existing) == dict(value):
            return
        raise S29RunnerBlocked(f"{field}:IMMUTABLE_OUTPUT_EXISTS")
    write_canonical_json(path, value)


def _inventory_envelope(root: Path, reference: str) -> Mapping[str, Any]:
    value = _load_optional(root, reference, field="gpu_inventory")
    if not isinstance(value, Mapping):
        raise S29RunnerBlocked("GPU_INVENTORY_ENVELOPE_REQUIRED")
    return dict(value)


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    preflight = load_s209_preflight(
        data_root=args.data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    if preflight.measurement_plan.get("run_id") != args.run_id:
        raise S29RunnerBlocked("MEASUREMENT_PLAN_RUN_ID_MISMATCH")
    return {
        "schema_version": "stage2-s209-g27a-production-preflight-v1",
        "status": "READY",
        "formal_eligible": True,
        "run_id": args.run_id,
        "measurement_plan_hash": preflight.plan_hash,
        "matrix_hash": preflight.frozen.matrix_hash,
        "raw_manifest_hash": preflight.frozen.raw_manifest_hash,
        "raw_run_id": preflight.frozen.raw_run_id,
        "g25_gate_hash": preflight.frozen.g25_gate_hash,
        "approved_gpu_uuids": preflight.inventory["approved_gpu_uuids"],
        "excluded_gpu_uuid": preflight.inventory["excluded_gpu_uuid"],
        "excluded_pci": preflight.inventory["excluded_pci"],
        "inventory_hash": preflight.inventory["inventory_artifact_hash"],
        "inventory_source_sha256": preflight.inventory["inventory_source_sha256"],
        "inventory_source_ref": preflight.inventory["inventory_source_ref"],
        "io_evidence_hash": preflight.io_evidence["artifact_hash"],
        "cost_io_quiescent": preflight.io_evidence["cost_io_quiescent"],
        "capacity_ref": preflight.capacity_ref,
        "capacity_evidence_hash": preflight.capacity_inputs["capacity_evidence_hash"] if preflight.capacity_inputs else None,
        "ulimit_ref": preflight.ulimit_ref,
        "ulimit_evidence_hash": preflight.ulimit_evidence["ulimit_evidence_hash"] if preflight.ulimit_evidence else None,
        "actual_measurements_required": True,
        "four_gpu_anchor_required": True,
        "resumable_terminal_rows": True,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = args.data_root.resolve()
    inventory = _inventory_envelope(root, args.gpu_inventory_ref) if args.gpu_inventory_ref else None
    io_value = _load_optional(root, args.io_evidence_ref, field="io_evidence") if args.io_evidence_ref else None
    plan = prepare_s209_plan(
        data_root=root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        run_id=args.run_id,
        anchor_ids=tuple(args.anchor_id),
        repetitions=args.repetitions,
        randomization_seed=args.randomization_seed,
        inventory=inventory,
        io_evidence=io_value,
        output_ref=args.measurement_plan_ref,
    )
    return plan


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    preflight = load_s209_preflight(
        data_root=args.data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    root = args.data_root.resolve()
    single_anchor = _load_optional(root, args.single_gpu_anchor_ref, field="single_gpu_anchor")
    four_anchor = _load_optional(root, args.four_gpu_anchor_ref, field="four_gpu_anchor")
    accuracy = _load_optional(root, args.accuracy_ref, field="accuracy") if args.accuracy_ref else []
    if isinstance(accuracy, Mapping):
        accuracy = accuracy.get("rows", accuracy.get("accuracy_rows", []))
    if not isinstance(accuracy, list) or not all(isinstance(item, Mapping) for item in accuracy):
        raise S29RunnerBlocked("ACCURACY_ROWS_INVALID")
    profiler = subprocess_profiler_executor(args.profiler_command)
    runner = S29ProfilerRunner(
        preflight=preflight,
        run_id=args.run_id,
        run_root=_logical(root, args.run_root, field="run_root"),
        profiler=profiler,
        single_gpu_anchor=single_anchor if isinstance(single_anchor, Mapping) else None,
        four_gpu_anchor=four_anchor if isinstance(four_anchor, Mapping) else None,
        shared_attribution_cross_check=(
            _load_optional(root, args.crosscheck_ref, field="crosscheck")
            if args.crosscheck_ref else None
        ),
        accuracy_rows=[dict(item) for item in accuracy],
        capacity_inputs=dict(preflight.capacity_inputs or {}),
    )
    return runner.run()


def _detach(args: argparse.Namespace) -> dict[str, Any]:
    if not args.profiler_command:
        raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
    # No detached child may exist until all immutable input, inventory, I/O,
    # capacity, and ulimit evidence has passed the same launch-time preflight.
    _preflight(args)
    root = args.data_root.resolve()
    run_root = _logical(root, args.run_root, field="run_root")
    run_root.mkdir(parents=True, exist_ok=True)
    lease_path = _claim_detach_lease(run_root, run_id=args.run_id)
    log_path = run_root / "launcher.log"
    child = [str(item) for item in sys.argv[1:] if item != "--detach"]
    try:
        with log_path.open("ab") as handle:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *child],
                cwd=_REPOSITORY_ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
    except BaseException:
        # The lease remains as an immutable launch-attempt marker; do not make
        # a failed Popen look like an unused formal run-root.
        raise
    payload: dict[str, Any] = {
        "schema_version": "stage2-s209-g27a-detached-launch-v1",
        "pid": int(process.pid),
        "run_id": args.run_id,
        "run_root": args.run_root,
        "log_ref": args.run_root.rstrip("/") + "/launcher.log",
        "status_ref": args.run_root.rstrip("/") + "/status.json",
        "lease_ref": args.run_root.rstrip("/") + "/launcher.lease.json",
        "started_at": _now(),
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _write_once(run_root / "launcher.pid.json", payload, field="detached_launch")
    return payload


def _status(args: argparse.Namespace, *, wait: bool) -> int:
    # Status/replay consumers must reject inventory or frozen-cost identity drift
    # before trusting an existing detached status file.
    preflight = load_s209_preflight(
        data_root=args.data_root,
        matrix_ref=args.matrix_ref,
        gate_ref=args.gate_ref,
        raw_manifest_ref=args.raw_manifest_ref,
        g25_gate_ref=args.g25_gate_ref,
        measurement_plan_ref=args.measurement_plan_ref,
        gpu_inventory_ref=args.gpu_inventory_ref,
        io_evidence_ref=args.io_evidence_ref,
        capacity_ref=args.capacity_ref,
        ulimit_ref=args.ulimit_ref,
    )
    inventory_identity = preflight.inventory.get("inventory_identity")
    if not isinstance(inventory_identity, Mapping):
        raise S29RunnerBlocked("STATUS_INVENTORY_IDENTITY_MISSING")
    if (
        inventory_identity.get("artifact_hash") != preflight.inventory.get("inventory_artifact_hash")
        or inventory_identity.get("source_sha256") != preflight.inventory.get("inventory_source_sha256")
    ):
        raise S29RunnerBlocked("STATUS_INVENTORY_IDENTITY_DRIFT")
    store = S29StatusStore(
        _logical(args.data_root.resolve(), args.run_root, field="run_root") / "status.json",
        run_id=args.run_id,
        plan_hash=preflight.plan_hash,
        inventory_identity=inventory_identity,
        cost_identity={
            "matrix_hash": preflight.frozen.matrix_hash,
            "raw_manifest_hash": preflight.frozen.raw_manifest_hash,
        },
    )
    path = _logical(args.data_root.resolve(), args.run_root, field="run_root") / "status.json"
    deadline = None if args.timeout_seconds is None else time.monotonic() + args.timeout_seconds
    while True:
        if path.exists():
            value = store.load().to_dict()
            print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
            if not wait or value.get("status") in {"SEALED", "FAILED", "BLOCKED"}:
                return 0 if value.get("status") == "SEALED" else 3
        elif not wait:
            return 4
        if deadline is not None and time.monotonic() >= deadline:
            return 4
        time.sleep(max(0.1, args.poll_seconds))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Strict detached S2.9/G2.7a profiler runner")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--detach", action="store_true")
    action.add_argument("--status", action="store_true")
    action.add_argument("--wait", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", default="s209-g27a-formal")
    parser.add_argument("--matrix-ref", required=True)
    parser.add_argument("--gate-ref", required=True)
    parser.add_argument("--g25-gate-ref", required=True)
    parser.add_argument("--raw-manifest-ref", required=True)
    parser.add_argument("--measurement-plan-ref", required=True)
    parser.add_argument("--gpu-inventory-ref", required=True)
    parser.add_argument("--io-evidence-ref", required=True)
    parser.add_argument("--profiler-command", nargs="+")
    parser.add_argument("--single-gpu-anchor-ref")
    parser.add_argument("--four-gpu-anchor-ref")
    parser.add_argument("--accuracy-ref")
    parser.add_argument("--capacity-ref")
    parser.add_argument("--ulimit-ref")
    parser.add_argument("--crosscheck-ref")
    parser.add_argument("--anchor-id", action="append", default=["method-only-anchor-0", "method-only-anchor-1"])
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--randomization-seed", type=int, default=2909)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
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
        if args.prepare:
            print(json.dumps(_prepare(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.preflight:
            print(json.dumps(_preflight(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        if args.execute:
            if not args.profiler_command:
                raise S29RunnerBlocked("PROFILER_COMMAND_REQUIRED")
            print(json.dumps(_execute(args), ensure_ascii=False, sort_keys=True, indent=2))
            return 0
        raise S29RunnerBlocked("S29_ACTION_REQUIRED")
    except (S29RunnerBlocked, OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"S2.9/G2.7a blocked: {type(error).__name__}:{error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
