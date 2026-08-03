"""Project-scoped GPU lease, preflight, and fault policy regressions."""

from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.runtime import (
    FailureClass,
    GpuLeaseIdentity,
    LaunchClaimRegistry,
    ProjectGpuLease,
    classify_stale_heartbeat,
    evaluate_launch_preflight,
    exercise_canary_writer,
    failure_response,
)


def _identity() -> GpuLeaseIdentity:
    return GpuLeaseIdentity(
        run_id="capacity-run",
        lease_id="capacity-lease",
        gpu_uuids=("GPU-a", "GPU-b", "GPU-c", "GPU-d"),
        owner="stage0-g8",
        config_hash="a" * 64,
        environment_hash="b" * 64,
    )


def _snapshot() -> dict[str, object]:
    return {
        "required_gate_ids": ["stage0.G7"],
        "passed_gate_ids": ["stage0.G7"],
        "source_clean": True,
        "identity_hashes_match": True,
        "selected_gpu_uuids": ["GPU-a", "GPU-b", "GPU-c", "GPU-d"],
        "gpu_health_ok": True,
        "external_gpu_processes": [],
        "active_competing_downloads": [],
        "data_cursor_covered": True,
        "data_free_bytes": 500 * 1024**3,
        "expected_new_bytes": 10 * 1024**3,
        "root_free_bytes": 20 * 1024**3,
        "inode_free": 1000,
        "fd_soft_limit": 1024,
        "predicted_open_fds": 100,
        "memory_available_bytes": 100 * 1024**3,
        "predicted_host_peak_bytes": 10 * 1024**3,
        "output_collision": False,
        "lease_available": True,
        "g1d_status": "ACCEPTED_SINGLE_DISK_RISK",
    }


def test_project_gpu_lease_is_exclusive_and_releases_with_history(tmp_path: Path) -> None:
    first = ProjectGpuLease(tmp_path, _identity())
    record = first.acquire()
    assert record["status"] == "HELD"
    with pytest.raises(RuntimeError, match="GPU_LEASE_RESOURCE_BUSY|LIVE_OWNER"):
        ProjectGpuLease(tmp_path, _identity()).acquire()
    heartbeat = first.heartbeat()
    assert heartbeat["heartbeat_at"] >= record["heartbeat_at"]
    history = first.release(outcome="SUCCESS")
    assert history.is_file()
    assert not first.current_path.exists()
    replay = ProjectGpuLease(tmp_path, _identity())
    replay.acquire()
    replay.release(outcome="SUCCESS")


def test_live_process_with_stale_record_is_never_silently_reaped(tmp_path: Path) -> None:
    lease = ProjectGpuLease(tmp_path, _identity())
    lease.acquire()
    lease.close()  # Simulate a stale heartbeat while this process is still alive.
    with pytest.raises(RuntimeError, match="LIVE_OWNER_PRESENT"):
        ProjectGpuLease(tmp_path, _identity()).acquire()
    assert lease.current_path.exists()
    lease.current_path.unlink()


def test_launch_claim_prevents_duplicate_ssh_restart(tmp_path: Path) -> None:
    registry = LaunchClaimRegistry(tmp_path)
    path = registry.claim(
        launch_id="formal-capacity",
        run_id="capacity-run",
        config_hash="a" * 64,
        environment_hash="b" * 64,
    )
    assert path.is_file()
    with pytest.raises(RuntimeError, match="LAUNCH_CLAIM_ALREADY_EXISTS"):
        registry.claim(
            launch_id="formal-capacity",
            run_id="capacity-run",
            config_hash="a" * 64,
            environment_hash="b" * 64,
        )


def test_preflight_passes_only_when_every_hard_condition_passes() -> None:
    passed = evaluate_launch_preflight(_snapshot())
    assert passed["status"] == "PASS"
    assert passed["running_state_may_publish"] is True
    for field, value, code in (
        ("external_gpu_processes", [{"pid": 9}], "PREFLIGHT_EXTERNAL_GPU_PROCESS"),
        ("root_free_bytes", 1, "PREFLIGHT_ROOT_DISK_INSUFFICIENT"),
        ("predicted_open_fds", 900, "PREFLIGHT_FD_HEADROOM_INSUFFICIENT"),
        ("output_collision", True, "PREFLIGHT_OUTPUT_COLLISION"),
    ):
        candidate = _snapshot()
        candidate[field] = value
        report = evaluate_launch_preflight(candidate)
        assert report["status"] == "FAIL"
        assert report["running_state_may_publish"] is False
        assert code in {item["code"] for item in report["blockers"]}


def test_fault_policy_and_injected_canary_failure_fail_closed() -> None:
    def fail() -> None:
        raise OSError("injected disk canary failure")

    report = exercise_canary_writer(fail)
    assert report == {
        "status": "EXPECTED_FAILURE",
        "failure_class": "JSONL_OR_STATUS_WRITE_FAILED",
        "error_type": "OSError",
        "stop_required": True,
    }
    assert failure_response(FailureClass.OOM)["retry"] == "new_config_identity"
    assert failure_response(FailureClass.DERIVED_TRACKING)["stop"] is False
    assert classify_stale_heartbeat(
        heartbeat_stale=True, process_alive=True
    ) == "ACTIVE_PROCESS_HEARTBEAT_STALE_DO_NOT_REAP"
