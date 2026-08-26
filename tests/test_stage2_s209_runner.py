from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace
import os

import pytest

import ops.stage2.run_s209_g27a as launcher

from param_importance_nlp.contracts.g21_formal_handoff import ALLOWED_DEVICES
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
)
from param_importance_nlp.experiments.stage2_s209_runner import (
    S29_COUNT_FIELDS,
    S29_INVENTORY_SCHEMA,
    S29_IO_SCHEMA,
    S29_SHARED_POOL_SCHEMA,
    S29_SHARED_RUN_SCHEMA,
    S29_TIMING_FIELDS,
    S29RunnerBlocked,
    S29ProfilerRunner,
    S29StatusStore,
    _task_list,
    _validate_shared_bundle,
    _validate_measured_row,
    _load_inventory_envelope,
    validate_s209_gpu_inventory,
    validate_s209_io_evidence,
)
from param_importance_nlp.experiments.stage2_s209_g27a import S29FrozenInputs, S29G27ABlocked, _formal_file


def _inventory() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pci, uuid in ALLOWED_DEVICES:
        rows.append(
            {
                "uuid": uuid,
                "pci_bus_id": pci,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "row_remap_status": "None",
                "gpu_recovery_action": "None",
                "xid_errors": 0,
                "gpu_class": "A100-80GB",
            }
        )
    rows.append(
        {
            "uuid": EXCLUDED_GPU_UUID,
            "pci_bus_id": EXCLUDED_PCI,
            "memory_used_mib": 0,
            "memory_total_mib": 81920,
            "utilization_gpu_percent": 0,
            "ecc_uncorrected_volatile": 113,
            "ecc_uncorrected_aggregate": 179,
            "row_remap_status": "Pending",
            "gpu_recovery_action": "None",
            "xid_errors": 0,
            "gpu_class": "A100-80GB",
        }
    )
    for index, pci in enumerate(("0000:4F:00.0", "0000:51:00.0", "0000:57:00.0")):
        rows.append(
            {
                "uuid": f"GPU-extra-{index}",
                "pci_bus_id": pci,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "row_remap_status": "None",
                "gpu_recovery_action": "None",
                "xid_errors": 0,
                "gpu_class": "A100-80GB",
            }
        )
    return rows


def _io(status: str = "QUIESCENT") -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": S29_IO_SCHEMA,
        "status": status,
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "run_id": "io-s29",
        "probe_id": "probe-0",
        "active_transfers": [],
        "failure_evidence": [],
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _write_s27_inventory(path: Path) -> dict[str, object]:
    """Build the real S2.6 13-key wire used by the formal S2.9 loader."""

    data_root = path.parent.parent
    source_path = path.parent / "gpu-inventory.capture.txt"
    source_bytes = b"test s2.6 live gpu capture\n"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(source_bytes)
    rows: list[dict[str, object]] = []
    for row in _inventory():
        rows.append(
            {
                "uuid": row["uuid"],
                "pci_bus_id": row["pci_bus_id"],
                "gpu_name": row["gpu_class"],
                "temperature_c": 40.0,
                "memory_used_mib": row["memory_used_mib"],
                "memory_total_mib": row["memory_total_mib"],
                "utilization_gpu_percent": row["utilization_gpu_percent"],
                "compute_mode": "Default",
                "ecc_uncorrected_volatile": row["ecc_uncorrected_volatile"],
                "ecc_uncorrected_aggregate": row["ecc_uncorrected_aggregate"],
                "row_remap_failure": 0,
                "row_remap_pending": 0,
                "row_remap_status": row["row_remap_status"],
                "gpu_recovery_action": row["gpu_recovery_action"],
                "health_state": "HEALTHY",
                "compute_apps": [],
            }
        )
    payload: dict[str, object] = {
        "schema_version": "stage2-s206-gpu-inventory-v1",
        "scope": "formal",
        "status": "OBSERVED",
        "checked_at": "2026-08-26T00:00:00+00:00",
        "artifact_ref": path.relative_to(data_root).as_posix(),
        "source_ref": source_path.relative_to(data_root).as_posix(),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rows": rows,
        "compute_apps": [],
        "approved_gpu_uuids": list(APPROVED_GPU_UUIDS),
        "excluded_pci": EXCLUDED_PCI,
        "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, payload)
    return payload


def _identity() -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": launcher.S29_EXECUTION_IDENTITY_SCHEMA,
        "repository_head": "a" * 40,
        "launcher_source_sha256": "b" * 64,
        "profiler_command_hash": "c" * 64,
        "repository_clean": True,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def test_s209_inventory_keeps_excluded_gpu_out_of_selection() -> None:
    summary = validate_s209_gpu_inventory(_inventory())
    assert summary["selected_gpu_uuids"] == list(APPROVED_GPU_UUIDS)
    assert summary["excluded_gpu_uuid"] == EXCLUDED_GPU_UUID
    assert summary["excluded_pci"] == EXCLUDED_PCI
    assert summary["excluded_present"] is True


def test_s209_inventory_rejects_excluded_pci_bound_to_approved_uuid() -> None:
    rows = _inventory()
    rows[0] = {**rows[0], "pci_bus_id": EXCLUDED_PCI}
    with pytest.raises(S29RunnerBlocked, match="APPROVED_GPU_BOUND_TO_EXCLUDED_PCI"):
        validate_s209_gpu_inventory(rows)


def test_s209_inventory_envelope_is_hash_bound_and_complete(tmp_path: Path) -> None:
    path = tmp_path / "evidence/gpu-inventory.json"
    payload = _write_s27_inventory(path)
    summary, identity = _load_inventory_envelope(payload, root=tmp_path, inventory_ref="evidence/gpu-inventory.json")
    assert summary["inventory_count"] == 8
    assert identity["artifact_hash"] == payload["artifact_hash"]
    assert len(identity["source_sha256"]) == 64
    tampered = dict(payload)
    tampered["rows"] = list(payload["rows"])  # type: ignore[arg-type]
    tampered["rows"][0] = {**tampered["rows"][0], "memory_used_mib": 1}  # type: ignore[index]
    with pytest.raises(S29RunnerBlocked, match="CALLER_MAPPING_DRIFT"):
        _load_inventory_envelope(tampered, root=tmp_path, inventory_ref="evidence/gpu-inventory.json")

    wrong_ref = dict(payload)
    wrong_ref["source_ref"] = "evidence/another-inventory.json"
    wrong_ref["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in wrong_ref.items() if key != "artifact_hash"}
    )
    with pytest.raises(S29RunnerBlocked, match="CALLER_MAPPING_DRIFT"):
        _load_inventory_envelope(wrong_ref, root=tmp_path, inventory_ref="evidence/gpu-inventory.json")


@pytest.mark.parametrize(
    ("field", "value", "pattern"),
    [
        ("memory_used_mib", 1, "CARD_NOT_IDLE"),
        ("ecc_uncorrected_volatile", 1, "ECC_NOT_CLEAN"),
        ("xid_errors", 1, "XID_NOT_CLEAN"),
        ("row_remap_status", "Pending", "ROW_REMAP_NOT_CLEAN"),
        ("gpu_recovery_action", "Reset", "RECOVERY_NOT_CLEAN"),
    ],
)
def test_s209_approved_health_and_remap_fields_fail_closed(field: str, value: object, pattern: str) -> None:
    rows = _inventory()
    rows[0][field] = value
    with pytest.raises(S29RunnerBlocked, match=pattern):
        validate_s209_gpu_inventory(rows)


def test_s209_approved_gpu_class_must_be_identical() -> None:
    rows = _inventory()
    rows[0]["gpu_class"] = "different-class"
    with pytest.raises(S29RunnerBlocked, match="GPU_CLASS_DRIFT"):
        validate_s209_gpu_inventory(rows)


def test_s209_compute_app_on_approved_card_is_not_idle() -> None:
    with pytest.raises(S29RunnerBlocked, match="CARD_NOT_IDLE"):
        validate_s209_gpu_inventory(_inventory(), compute_apps=[{"gpu_uuid": APPROVED_GPU_UUIDS[0], "pid": 123}])


def test_s209_io_evidence_is_explicit_and_fail_closed() -> None:
    assert validate_s209_io_evidence(_io())["cost_io_quiescent"] is True
    value = _io()
    value["active_transfers"] = ["curl-download"]
    value["artifact_hash"] = canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    with pytest.raises(S29RunnerBlocked, match="QUIESCENT_WITH_ACTIVE_TRANSFER"):
        validate_s209_io_evidence(value)


def test_s209_status_store_resumes_only_after_owner_exits(tmp_path) -> None:
    path = tmp_path / "status.json"
    plan_hash = "a" * 64
    store = S29StatusStore(path, run_id="s29-run", plan_hash=plan_hash)
    status = store.acquire(expected_tasks=2)
    assert status.status == "RUNNING"
    assert store.load().owner_pid is not None


def test_s209_status_store_rejects_frozen_identity_tamper(tmp_path: Path) -> None:
    path = tmp_path / "status.json"
    store = S29StatusStore(
        path,
        run_id="s29-run",
        plan_hash="a" * 64,
        inventory_identity={"artifact_hash": "b" * 64, "source_sha256": "c" * 64},
        cost_identity={"matrix_hash": "d" * 64, "raw_manifest_hash": "e" * 64},
    )
    store.acquire(expected_tasks=2)
    tampered = load = store.load().to_dict()
    tampered["matrix_hash"] = "f" * 64
    tampered["artifact_hash"] = canonical_json_hash({key: value for key, value in tampered.items() if key != "artifact_hash"})
    write_canonical_json(path, tampered)
    with pytest.raises(S29RunnerBlocked, match="STATUS_MATRIX_HASH_DRIFT"):
        store.load()


def test_s209_task_list_is_single_gpu_for_every_method_task() -> None:
    preflight = SimpleNamespace(
        measurement_plan={
            "run_id": "s209-run",
            "artifact_hash": "d" * 64,
            "rows": [
                {"anchor_id": "anchor-a", "repetition": 0, "method_order": ["double", "raw", "u"]},
                {"anchor_id": "anchor-a", "repetition": 1, "method_order": ["raw", "u", "double"]},
            ]
        },
        plan_hash="d" * 64,
        frozen=S29FrozenInputs(
            matrix_hash="a" * 64,
            g24b_gate_hash="b" * 64,
            raw_manifest_hash="c" * 64,
            raw_run_id="s207-run",
            plan_hash="e" * 64,
            mapping_hash="f" * 64,
            sampling_plan_hash="1" * 64,
            expected_unit_ids=("unit-0",),
            batch_size=32,
            microbatch_count=16,
            repetitions=2,
            completion_denominator=1,
        ),
    )
    tasks = _task_list(preflight, run_id="s209-run")
    assert tasks
    assert len(tasks) == 18
    assert {task["anchor_id"] for task in tasks} == {"anchor-a"}
    assert {task["repetition"] for task in tasks} == {0, 1}
    assert all(task["device_count"] == 1 for task in tasks)
    assert all(task["gpu_uuid"] in APPROVED_GPU_UUIDS for task in tasks)
    assert all(len(str(task["gpu_uuid"])) > 0 for task in tasks)


def test_s209_task_list_rejects_plan_run_id_rebinding() -> None:
    preflight = SimpleNamespace(
        measurement_plan={
            "run_id": "planned-run",
            "rows": [
                {"anchor_id": "anchor-a", "repetition": 0, "method_order": ["double", "raw", "u"]},
            ],
        },
        plan_hash="d" * 64,
        frozen=S29FrozenInputs(
            matrix_hash="a" * 64,
            g24b_gate_hash="b" * 64,
            raw_manifest_hash="c" * 64,
            raw_run_id="s207-run",
            plan_hash="e" * 64,
            mapping_hash="f" * 64,
            sampling_plan_hash="1" * 64,
            expected_unit_ids=("unit-0",),
            batch_size=32,
            microbatch_count=16,
            repetitions=2,
            completion_denominator=1,
        ),
    )
    with pytest.raises(S29RunnerBlocked, match="MEASUREMENT_PLAN_RUN_ID_MISMATCH"):
        _task_list(preflight, run_id="different-run")


def _shared_group_fixture(tmp_path: Path) -> tuple[SimpleNamespace, list[dict[str, object]], dict[str, object], dict[str, object]]:
    frozen = S29FrozenInputs(
        matrix_hash="a" * 64,
        g24b_gate_hash="b" * 64,
        raw_manifest_hash="c" * 64,
        raw_run_id="s207-run",
        plan_hash="e" * 64,
        mapping_hash="f" * 64,
        sampling_plan_hash="1" * 64,
        expected_unit_ids=("unit-0",),
        batch_size=32,
        microbatch_count=16,
        repetitions=2,
        completion_denominator=1,
    )
    plan = {
        "run_id": "s209-run",
        "artifact_hash": "d" * 64,
        "rows": [{"anchor_id": "shared-a", "repetition": 0, "method_order": ["double", "raw", "u"]}],
    }
    preflight = SimpleNamespace(
        root=tmp_path,
        measurement_plan=plan,
        plan_hash="d" * 64,
        frozen=frozen,
        inventory={"inventory_identity": {"artifact_hash": "1" * 64, "source_sha256": "2" * 64}},
        io_evidence={"artifact_hash": "3" * 64, "cost_io_quiescent": True},
    )
    tasks = [task for task in _task_list(preflight, run_id="s209-run") if task["semantic"] == "scientific_equal_sample_cost"]
    return preflight, tasks, {"artifact_hash": "3" * 64, "cost_io_quiescent": True}, {"artifact_hash": "1" * 64, "source_sha256": "2" * 64}


def _shared_bundle(tasks: list[dict[str, object]]) -> dict[str, object]:
    task = tasks[0]
    pool: dict[str, object] = {
        "schema_version": S29_SHARED_POOL_SCHEMA,
        "paired_run_id": task["paired_run_id"],
        "paired_run_identity_hash": task["paired_run_identity_hash"],
        "measurement_plan_hash": task["measurement_plan_hash"],
        "matrix_hash": "a" * 64,
        "raw_manifest_hash": "c" * 64,
        "source_raw_run_id": "s207-run",
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": 1,
        "batch_size": 32,
        "microbatch_count": 16,
        "method_order": list(task["shared_method_order"]),
        "pool_id": "4" * 64,
        "sample_mapping_hash": "5" * 64,
        "gradient_pool_hash": "6" * 64,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "cost_io_quiescent": True,
        "shared_pool_ref": task["shared_pool_ref"],
    }
    pool["artifact_hash"] = canonical_json_hash(pool)
    rows: list[dict[str, object]] = []
    for method_task in tasks:
        method = str(method_task["method"])
        row: dict[str, object] = {
            "measurement_kind": "actual",
            "measured": True,
            "semantic": "scientific_equal_sample_cost",
            "method": method,
            "anchor_id": method_task["anchor_id"],
            "repetition": method_task["repetition"],
            "run_id": method_task["run_id"],
            "source_raw_run_id": "s207-run",
            "matrix_hash": "a" * 64,
            "raw_manifest_hash": "c" * 64,
            "gpu_uuid": method_task["gpu_uuid"],
            "device_count": 1,
            "inventory_artifact_hash": "1" * 64,
            "inventory_source_sha256": "2" * 64,
            "cost_io_quiescent": True,
            "health_ok": True,
            "batch_size": 32,
            "microbatch_count": 16,
            "sequence_count": 32,
            "token_count": 1024,
            "backward_count": 1,
            "communication_bytes": 0,
            "output_bytes": 4096,
            "data_wait_seconds": 0.5,
            "forward_seconds": 2.0,
            "backward_seconds": 3.0,
            "gradient_aggregation_seconds": 0.5,
            "formula_seconds": 0.5,
            "statistics_seconds": 0.5,
            "communication_seconds": 0.0,
            "write_seconds": 1.0,
            "wall_seconds": 20.0,
            "allocated_peak_bytes": 100,
            "reserved_peak_bytes": 120,
            "device_peak_bytes": 140,
            "anchor_kind": "shared_runner",
            "paired_run_id": method_task["paired_run_id"],
            "paired_run_identity_hash": method_task["paired_run_identity_hash"],
            "measurement_plan_hash": method_task["measurement_plan_hash"],
            "shared_pool_id": pool["pool_id"],
            "shared_pool_artifact_hash": pool["artifact_hash"],
            "shared_pool_ref": pool["shared_pool_ref"],
            "shared_sample_mapping_hash": pool["sample_mapping_hash"],
            "shared_gradient_pool_hash": pool["gradient_pool_hash"],
            "shared_method_order": list(method_task["shared_method_order"]),
            "shared_method_index": method_task["shared_method_index"],
            "shared_sample_sequence_count": pool["sequence_count"],
            "shared_sample_token_count": pool["token_count"],
        }
        rows.append(row)
    return {
        "schema_version": S29_SHARED_RUN_SCHEMA,
        "semantic": "scientific_equal_sample_cost",
        "paired_run_id": task["paired_run_id"],
        "paired_run_identity_hash": task["paired_run_identity_hash"],
        "run_id": task["run_id"],
        "measurement_plan_hash": task["measurement_plan_hash"],
        "anchor_id": task["anchor_id"],
        "repetition": task["repetition"],
        "gpu_uuid": task["gpu_uuid"],
        "device_count": 1,
        "method_order": list(task["shared_method_order"]),
        "methods": ["raw", "double", "u"],
        "shared_pool": pool,
        "rows": rows,
    }


def test_s209_shared_semantic_uses_one_serial_pooled_worker(tmp_path: Path) -> None:
    preflight, tasks, io_evidence, inventory_identity = _shared_group_fixture(tmp_path)
    calls: list[dict[str, str]] = []

    def profiler(_task, *, environment):
        calls.append(dict(environment))
        return _shared_bundle(tasks)

    runner = S29ProfilerRunner(preflight=preflight, run_id="s209-run", run_root=tmp_path / "run", profiler=profiler)
    completed: dict[str, dict[str, object]] = {}
    runner._run_shared_group(tasks, completed=completed)
    assert len(calls) == 1
    assert calls[0]["S29_METHOD"] == "shared"
    assert calls[0]["CUDA_VISIBLE_DEVICES"] == APPROVED_GPU_UUIDS[0]
    assert len(completed) == 3
    assert (tmp_path / "run" / "shared-pools" / f"{tasks[0]['paired_run_id']}.json").exists()


def test_s209_shared_bundle_rejects_pool_tamper_and_method_mislabel(tmp_path: Path) -> None:
    preflight, tasks, io_evidence, inventory_identity = _shared_group_fixture(tmp_path)
    bundle = _shared_bundle(tasks)
    tampered_pool = dict(bundle["shared_pool"])
    tampered_pool["token_count"] = 2048
    with pytest.raises(S29RunnerBlocked, match="SHARED_POOL_HASH_MISMATCH"):
        _validate_shared_bundle(
            {**bundle, "shared_pool": tampered_pool},
            tasks=tasks,
            frozen=preflight.frozen,
            io_evidence=io_evidence,
            inventory_identity=inventory_identity,
        )
    tampered_rows = [dict(row) for row in bundle["rows"]]
    tampered_rows[0]["anchor_kind"] = "method_only"
    with pytest.raises(S29RunnerBlocked, match="PROFILER_IDENTITY_DRIFT:anchor_kind"):
        _validate_shared_bundle(
            {**bundle, "rows": tampered_rows},
            tasks=tasks,
            frozen=preflight.frozen,
            io_evidence=io_evidence,
            inventory_identity=inventory_identity,
        )
    drifted_tasks = [dict(task) for task in tasks]
    drifted_tasks[1]["anchor_id"] = "other-anchor"
    with pytest.raises(S29RunnerBlocked, match="PROFILER_SHARED_TASK_IDENTITY_DRIFT"):
        _validate_shared_bundle(
            bundle,
            tasks=drifted_tasks,
            frozen=preflight.frozen,
            io_evidence=io_evidence,
            inventory_identity=inventory_identity,
        )


def test_s209_four_card_anchor_requires_worker_gpu_uuid_set() -> None:
    frozen = S29FrozenInputs(
        matrix_hash="a" * 64,
        g24b_gate_hash="b" * 64,
        raw_manifest_hash="c" * 64,
        raw_run_id="s207-run",
        plan_hash="d" * 64,
        mapping_hash="e" * 64,
        sampling_plan_hash="f" * 64,
        expected_unit_ids=("unit-0",),
        batch_size=32,
        microbatch_count=16,
        repetitions=2,
        completion_denominator=1,
    )
    task = {
        "semantic": "anchor",
        "method": "anchor",
        "anchor_id": "four-gpu-anchor",
        "repetition": 0,
        "run_id": "s209-run",
        "gpu_uuid": APPROVED_GPU_UUIDS[0],
        "gpu_uuids": list(APPROVED_GPU_UUIDS),
        "device_count": 4,
    }
    row: dict[str, object] = {
        "measurement_kind": "actual",
        "measured": True,
        "gpu_uuids": list(APPROVED_GPU_UUIDS),
        "source_raw_run_id": "s207-run",
        "matrix_hash": "a" * 64,
        "raw_manifest_hash": "c" * 64,
        "batch_size": 32,
        "microbatch_count": 16,
        "inventory_artifact_hash": "1" * 64,
        "inventory_source_sha256": "2" * 64,
        "cost_io_quiescent": True,
        "health_ok": True,
    }
    row.update({name: 0.1 for name in S29_TIMING_FIELDS})
    row.update({"wall_seconds": 10.0, "allocated_peak_bytes": 100, "reserved_peak_bytes": 120, "device_peak_bytes": 140})
    row.update({name: (0 if name.endswith("bytes") else 1) for name in S29_COUNT_FIELDS})
    validated = _validate_measured_row(
        row,
        task=task,
        frozen=frozen,
        io_evidence={"artifact_hash": "3" * 64, "cost_io_quiescent": True},
        inventory_identity={"artifact_hash": "1" * 64, "source_sha256": "2" * 64},
    )
    assert validated["device_count"] == 4
    with pytest.raises(S29RunnerBlocked, match="PROFILER_GPU_UUID_SET_INVALID"):
        _validate_measured_row(
            {**row, "gpu_uuids": [APPROVED_GPU_UUIDS[0]]},
            task=task,
            frozen=frozen,
            io_evidence={"artifact_hash": "3" * 64, "cost_io_quiescent": True},
            inventory_identity={"artifact_hash": "1" * 64, "source_sha256": "2" * 64},
        )


def test_s209_detach_rejects_duplicate_launch_lease(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {"execution_identity": _identity()})
    monkeypatch.setattr(launcher, "_repository_path", lambda _path: Path(__file__).parents[1])
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=os.getpid()))
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", "--detach"])
    args = SimpleNamespace(
        profiler_command=["profiler"],
        data_root=tmp_path,
        repository=Path(__file__).parents[1],
        run_root="runs/s209",
        run_id="s209-run",
    )
    returned = launcher._detach(args)
    assert returned["log_ref"] == "runs/s209/attempts/" + returned["attempt_id"] + "/launcher.log"
    assert returned == load_canonical_json(tmp_path / "runs/s209" / "attempts" / returned["attempt_id"] / "launch-running.json")
    with pytest.raises(S29RunnerBlocked, match="ALREADY_RUNNING"):
        launcher._detach(args)


def test_s209_detached_receipts_are_full_lineage_bound(tmp_path: Path) -> None:
    identity = _identity()
    run_root = tmp_path / "runs" / "s209"
    attempt_id = "launch-" + "1" * 32
    attempt_dir = run_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    receipt = launcher._attempt_identity(
        run_id="s209-run",
        run_root_ref="runs/s209",
        identity=identity,
        attempt_id=attempt_id,
        child_argv_hash="d" * 64,
        parent_pid=999999,
    )
    write_canonical_json(attempt_dir / "launch-receipt.json", receipt)
    running = dict(receipt)
    running.update({"status": "RUNNING", "child_pid": 999999, "started_at": "2026-08-26T00:00:00+00:00"})
    running["artifact_hash"] = canonical_json_hash({key: value for key, value in running.items() if key != "artifact_hash"})
    write_canonical_json(attempt_dir / "launch-running.json", running)
    (attempt_dir / "launcher.log").write_bytes(b"worker output\n")
    assert launcher._launch_attempts(run_root, run_id="s209-run", identity=identity)

    bad_log = dict(receipt, log_ref=f"runs/other/attempts/{attempt_id}/launcher.log")
    bad_log["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in bad_log.items() if key != "artifact_hash"}
    )
    with pytest.raises(S29RunnerBlocked, match="LOG_REF_DRIFT"):
        launcher._validate_attempt_receipt(bad_log, run_id="s209-run", identity=identity)

    drifted = dict(
        running,
        run_root_ref="runs/other",
        log_ref=f"runs/other/attempts/{attempt_id}/launcher.log",
    )
    drifted["artifact_hash"] = canonical_json_hash({key: value for key, value in drifted.items() if key != "artifact_hash"})
    write_canonical_json(attempt_dir / "launch-running.json", drifted)
    with pytest.raises(S29RunnerBlocked, match="RUNNING_RECEIPT_DRIFT"):
        launcher._launch_attempts(run_root, run_id="s209-run", identity=identity)

    failure = launcher._attempt_failure(receipt, reason="SPAWN_FAILED")
    bad_failure = dict(
        failure,
        attempt=dict(
            receipt,
            run_root_ref="runs/other",
            log_ref=f"runs/other/attempts/{attempt_id}/launcher.log",
        ),
    )
    bad_failure["attempt"]["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in bad_failure["attempt"].items() if key != "artifact_hash"}
    )
    bad_failure["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in bad_failure.items() if key != "artifact_hash"}
    )
    with pytest.raises(S29RunnerBlocked, match="FAILURE_RECEIPT_DRIFT"):
        launcher._validate_attempt_failure(bad_failure, run_id="s209-run", identity=identity, receipt=receipt)


def test_s209_detached_child_waits_for_parent_running_receipt(tmp_path: Path, monkeypatch) -> None:
    identity = _identity()
    run_root = tmp_path / "runs" / "s209"
    attempt_id = "launch-" + "2" * 32
    attempt_dir = run_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    child_argv = ["--execute", "--attempt-id", attempt_id, "--detached-child-marker", attempt_id]
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", *child_argv])
    receipt = launcher._attempt_identity(
        run_id="s209-run",
        run_root_ref="runs/s209",
        identity=identity,
        attempt_id=attempt_id,
        child_argv_hash=launcher._child_argv_hash(child_argv),
        parent_pid=999999,
    )
    write_canonical_json(attempt_dir / "launch-receipt.json", receipt)
    state = {"published": False}

    def publish_running(_seconds: float) -> None:
        if state["published"]:
            return
        running = dict(receipt, status="RUNNING", child_pid=os.getpid(), started_at="2026-08-26T00:00:00+00:00")
        running["artifact_hash"] = canonical_json_hash(
            {key: value for key, value in running.items() if key != "artifact_hash"}
        )
        write_canonical_json(attempt_dir / "launch-running.json", running)
        state["published"] = True

    monkeypatch.setattr(launcher.time, "sleep", publish_running)
    args = SimpleNamespace(
        data_root=tmp_path,
        run_root="runs/s209",
        run_id="s209-run",
        detached_child_marker=attempt_id,
        attempt_id=attempt_id,
    )
    launcher._require_detached_child(args, identity=identity)
    assert state["published"] is True


def test_s209_stale_running_receipt_rejects_command_substitution(tmp_path: Path) -> None:
    identity = _identity()
    run_root = tmp_path / "runs" / "s209"
    attempt_id = "launch-" + "3" * 32
    attempt_dir = run_root / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True)
    receipt = launcher._attempt_identity(
        run_id="s209-run",
        run_root_ref="runs/s209",
        identity=identity,
        attempt_id=attempt_id,
        child_argv_hash="d" * 64,
        parent_pid=999999,
    )
    write_canonical_json(attempt_dir / "launch-receipt.json", receipt)
    running = dict(receipt, status="RUNNING", child_pid=999999, started_at="2026-08-26T00:00:00+00:00")
    running["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in running.items() if key != "artifact_hash"}
    )
    write_canonical_json(attempt_dir / "launch-running.json", running)
    with pytest.raises(S29RunnerBlocked, match="COMMAND_DRIFT"):
        launcher._claim_detached_attempt(
            run_root,
            run_id="s209-run",
            run_root_ref="runs/s209",
            identity=identity,
            child_argv_hash="e" * 64,
            attempt_id="launch-" + "4" * 32,
        )


def test_s209_logical_root_and_component_symlinks_fail_closed(tmp_path: Path) -> None:
    linked_root = tmp_path / "root-link"
    try:
        linked_root.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(S29RunnerBlocked, match="SYMLINK_COMPONENT"):
        launcher._logical(linked_root, "child.json", field="data_root")

    real_dir = tmp_path / "real"
    real_dir.mkdir()
    component_link = tmp_path / "component-link"
    try:
        component_link.symlink_to(real_dir, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this Windows host")
    with pytest.raises(S29RunnerBlocked, match="SYMLINK_COMPONENT"):
        launcher._logical(tmp_path, "component-link/child.json", field="input")


def test_s209_repository_probe_failure_and_timeout_block(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("git unavailable")),
    )
    with pytest.raises(S29RunnerBlocked, match="GIT_TOP_LEVEL_UNAVAILABLE"):
        launcher._repository_path(tmp_path)

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="git", timeout=30)

    monkeypatch.setattr(launcher.subprocess, "run", timeout)
    with pytest.raises(S29RunnerBlocked, match="GIT_TOP_LEVEL_UNAVAILABLE"):
        launcher._repository_path(tmp_path)


def test_s209_formal_anchor_reload_is_disk_and_hash_bound(tmp_path: Path) -> None:
    path = tmp_path / "run" / "anchors" / "single-gpu-anchor.json"
    payload: dict[str, object] = {"status": "PASS", "measurement_kind": "actual"}
    payload["artifact_hash"] = canonical_json_hash(payload)
    path.parent.mkdir(parents=True)
    write_canonical_json(path, payload)
    loaded, reference = _formal_file(
        tmp_path,
        "run/anchors/single-gpu-anchor.json",
        filename="single-gpu-anchor.json",
        field="single_anchor",
    )
    assert loaded == payload
    assert reference == "run/anchors/single-gpu-anchor.json"
    tampered = dict(payload, status="BLOCKED")
    write_canonical_json(path, tampered)
    with pytest.raises(S29G27ABlocked, match="ARTIFACT_HASH_MISMATCH"):
        _formal_file(
            tmp_path,
            "run/anchors/single-gpu-anchor.json",
            filename="single-gpu-anchor.json",
            field="single_anchor",
        )


def test_s209_detach_child_runs_execute_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {"execution_identity": _identity()})
    monkeypatch.setattr(launcher, "_repository_path", lambda _path: Path(__file__).parents[1])
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", "--detach", "--data-root", "fixture"])
    args = SimpleNamespace(
        profiler_command=["profiler"],
        data_root=tmp_path,
        repository=Path(__file__).parents[1],
        run_root="runs/s209",
        run_id="s209-run",
    )
    launcher._detach(args)
    command = captured["command"]
    assert isinstance(command, list)
    assert "--detach" not in command
    assert command.count("--execute") == 1


def test_s209_detach_rewrites_only_launcher_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {"execution_identity": _identity()})
    monkeypatch.setattr(launcher, "_repository_path", lambda _path: Path(__file__).parents[1])
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        launcher.sys,
        "argv",
        [
            "run_s209_g27a.py",
            "--detach",
            "--data-root",
            "fixture",
            "--profiler-command",
            "worker",
            "--detach",
        ],
    )
    args = SimpleNamespace(
        profiler_command=["worker", "--detach"],
        data_root=tmp_path,
        repository=Path(__file__).parents[1],
        run_root="runs/s209",
        run_id="s209-run",
    )
    launcher._detach(args)
    command = captured["command"]
    assert isinstance(command, list)
    assert command.count("--execute") == 1
    assert command.count("--detach") == 1


def test_s209_anchor_failure_publishes_terminal_blocked_status(tmp_path: Path) -> None:
    frozen = S29FrozenInputs(
        matrix_hash="a" * 64,
        g24b_gate_hash="b" * 64,
        raw_manifest_hash="c" * 64,
        raw_run_id="s207-run",
        plan_hash="d" * 64,
        mapping_hash="e" * 64,
        sampling_plan_hash="f" * 64,
        expected_unit_ids=("unit-0",),
        batch_size=32,
        microbatch_count=16,
        repetitions=2,
        completion_denominator=1,
    )
    plan = {
        "run_id": "s209-run",
        "artifact_hash": "d" * 64,
        "rows": [
            {"anchor_id": "anchor-a", "repetition": 0, "method_order": ["raw", "double", "u"]},
        ],
    }
    preflight = SimpleNamespace(
        root=tmp_path,
        measurement_plan=plan,
        frozen=frozen,
        inventory={"inventory_identity": {"artifact_hash": "1" * 64, "source_sha256": "2" * 64}},
        io_evidence={"artifact_hash": "3" * 64, "cost_io_quiescent": True},
        plan_hash="d" * 64,
    )

    def failing_profiler(_task, *, environment):
        raise RuntimeError("anchor unavailable")

    runner = S29ProfilerRunner(
        preflight=preflight,
        run_id="s209-run",
        run_root=tmp_path / "run",
        profiler=failing_profiler,
    )
    result = runner.run()
    assert result["status"] == "BLOCKED"
    assert runner.status_store.load().status == "BLOCKED"


def test_s209_detach_rejects_stale_lease_without_relaunch_or_unlink(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {"execution_identity": _identity()})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("stale lease must block before Popen"))
    run_root = tmp_path / "runs" / "s209"
    run_root.mkdir(parents=True)
    lease = {
        "schema_version": "stage2-s209-g27a-detached-lease-v1",
        "owner_pid": 999999,
        "run_id": "s209-run",
        "started_at": "2026-08-26T00:00:00+00:00",
    }
    lease["artifact_hash"] = canonical_json_hash(lease)
    lease_path = run_root / "launcher.lease.json"
    write_canonical_json(lease_path, lease)
    before = lease_path.read_bytes()
    args = SimpleNamespace(profiler_command=["profiler"], data_root=tmp_path, repository=Path(__file__).parents[1], run_root="runs/s209", run_id="s209-run")
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", "--detach"])
    with pytest.raises(S29RunnerBlocked, match="LEGACY_LAUNCH_MANIFEST"):
        launcher._detach(args)
    assert lease_path.read_bytes() == before


def test_s209_detach_rejects_existing_dead_pid_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {"execution_identity": _identity()})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("PID manifest must block before Popen"))
    run_root = tmp_path / "runs" / "s209"
    run_root.mkdir(parents=True)
    payload = {"schema_version": "stage2-s209-g27a-detached-launch-v1", "pid": 999999, "run_id": "s209-run"}
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(run_root / "launcher.pid.json", payload)
    args = SimpleNamespace(profiler_command=["profiler"], data_root=tmp_path, repository=Path(__file__).parents[1], run_root="runs/s209", run_id="s209-run")
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", "--detach"])
    with pytest.raises(S29RunnerBlocked, match="LEGACY_LAUNCH_MANIFEST"):
        launcher._detach(args)


def test_s209_detach_runs_preflight_before_popen(monkeypatch, tmp_path: Path) -> None:
    def reject(_args):
        raise S29RunnerBlocked("GPU_INVENTORY_SOURCE_SHA256_MISMATCH")

    monkeypatch.setattr(launcher, "_preflight", reject)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Popen must not run"))
    args = SimpleNamespace(
        profiler_command=["profiler"],
        data_root=tmp_path,
        run_root="runs/s209",
    )
    with pytest.raises(S29RunnerBlocked, match="SOURCE_SHA256_MISMATCH"):
        launcher._detach(args)
