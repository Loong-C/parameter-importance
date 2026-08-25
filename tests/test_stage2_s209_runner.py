from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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
    S29_INVENTORY_SCHEMA,
    S29_IO_SCHEMA,
    S29RunnerBlocked,
    S29StatusStore,
    _load_inventory_envelope,
    validate_s209_gpu_inventory,
    validate_s209_io_evidence,
)


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
