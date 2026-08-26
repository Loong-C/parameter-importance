from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import os

import pytest

import ops.stage2.run_s209_g27a as launcher

from param_importance_nlp.contracts.g21_formal_handoff import ALLOWED_DEVICES
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
)
from param_importance_nlp.experiments.stage2_s209_runner import (
    S29_COUNT_FIELDS,
    S29_INVENTORY_SCHEMA,
    S29_IO_SCHEMA,
    S29_TIMING_FIELDS,
    S29RunnerBlocked,
    S29ProfilerRunner,
    S29StatusStore,
    _task_list,
    _validate_measured_row,
    _load_inventory_envelope,
    validate_s209_gpu_inventory,
    validate_s209_io_evidence,
)
from param_importance_nlp.experiments.stage2_s209_g27a import S29FrozenInputs


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
    payload: dict[str, object] = {
        "schema_version": S29_INVENTORY_SCHEMA,
        "source_ref": "evidence/gpu-inventory.json",
        "rows": _inventory(),
        "compute_apps": [],
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    path = tmp_path / "evidence/gpu-inventory.json"
    path.parent.mkdir(parents=True)
    write_canonical_json(path, payload)
    summary, identity = _load_inventory_envelope(payload, root=tmp_path, inventory_ref=path)
    assert summary["inventory_count"] == 8
    assert identity["artifact_hash"] == payload["artifact_hash"]
    assert len(identity["source_sha256"]) == 64
    tampered = dict(payload)
    tampered["rows"] = list(payload["rows"])  # type: ignore[arg-type]
    tampered["rows"][0] = {**tampered["rows"][0], "memory_used_mib": 1}  # type: ignore[index]
    with pytest.raises(S29RunnerBlocked, match="ARTIFACT_HASH_MISMATCH"):
        _load_inventory_envelope(tampered, root=tmp_path, inventory_ref=path)

    wrong_ref = dict(payload)
    wrong_ref["source_ref"] = "evidence/another-inventory.json"
    wrong_ref["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in wrong_ref.items() if key != "artifact_hash"}
    )
    with pytest.raises(S29RunnerBlocked, match="SOURCE_REF_PATH_MISMATCH"):
        _load_inventory_envelope(wrong_ref, root=tmp_path, inventory_ref=path)


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
            "rows": [
                {"anchor_id": "anchor-a", "repetition": 0, "method_order": ["double", "raw", "u"]},
                {"anchor_id": "anchor-a", "repetition": 1, "method_order": ["raw", "u", "double"]},
            ]
        }
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
        }
    )
    with pytest.raises(S29RunnerBlocked, match="MEASUREMENT_PLAN_RUN_ID_MISMATCH"):
        _task_list(preflight, run_id="different-run")


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
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: SimpleNamespace(pid=os.getpid()))
    args = SimpleNamespace(
        profiler_command=["profiler"],
        data_root=tmp_path,
        run_root="runs/s209",
        run_id="s209-run",
    )
    launcher._detach(args)
    with pytest.raises(S29RunnerBlocked, match="ALREADY_RUNNING"):
        launcher._detach(args)


def test_s209_detach_child_runs_execute_action(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {})
    captured: dict[str, object] = {}

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        return SimpleNamespace(pid=os.getpid())

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(launcher.sys, "argv", ["run_s209_g27a.py", "--detach", "--data-root", "fixture"])
    args = SimpleNamespace(
        profiler_command=["profiler"],
        data_root=tmp_path,
        run_root="runs/s209",
        run_id="s209-run",
    )
    launcher._detach(args)
    command = captured["command"]
    assert isinstance(command, list)
    assert "--detach" not in command
    assert command.count("--execute") == 1


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
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {})
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
    args = SimpleNamespace(profiler_command=["profiler"], data_root=tmp_path, run_root="runs/s209", run_id="s209-run")
    with pytest.raises(S29RunnerBlocked, match="ALREADY_RUNNING"):
        launcher._detach(args)
    assert lease_path.read_bytes() == before


def test_s209_detach_rejects_existing_dead_pid_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_preflight", lambda _args: {})
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("PID manifest must block before Popen"))
    run_root = tmp_path / "runs" / "s209"
    run_root.mkdir(parents=True)
    payload = {"schema_version": "stage2-s209-g27a-detached-launch-v1", "pid": 999999, "run_id": "s209-run"}
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(run_root / "launcher.pid.json", payload)
    args = SimpleNamespace(profiler_command=["profiler"], data_root=tmp_path, run_root="runs/s209", run_id="s209-run")
    with pytest.raises(S29RunnerBlocked, match="ALREADY_RUNNING"):
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
