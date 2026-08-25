from __future__ import annotations

from datetime import datetime, timezone

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
)
from param_importance_nlp.experiments.stage2_s209_runner import (
    S29_IO_SCHEMA,
    S29RunnerBlocked,
    S29StatusStore,
    validate_s209_gpu_inventory,
    validate_s209_io_evidence,
)


def _inventory() -> list[dict[str, object]]:
    return [
        {"uuid": uuid, "pci_bus_id": f"0000:{index + 0x40:02x}:00.0", "health_ok": True}
        for index, uuid in enumerate(APPROVED_GPU_UUIDS)
    ] + [{"uuid": EXCLUDED_GPU_UUID, "pci_bus_id": EXCLUDED_PCI, "health_ok": False}]


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
