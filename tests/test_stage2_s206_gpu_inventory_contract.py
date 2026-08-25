from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import ops.stage2.run_s206_formal as launcher
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_BINDINGS,
    EXCLUDED_PCI,
    EXCLUDED_UUID,
    GPU_INVENTORY_SCHEMA,
    S206PreparationBlocked,
    validate_gpu_inventory,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pci, uuid in APPROVED_GPU_BINDINGS:
        rows.append(
            {
                "uuid": uuid,
                "pci_bus_id": pci,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "gpu_recovery_action": "None",
            }
        )
    rows.append(
        {
            "uuid": EXCLUDED_UUID,
            "pci_bus_id": EXCLUDED_PCI,
            "memory_used_mib": 0,
            "memory_total_mib": 81920,
            "utilization_gpu_percent": 0,
            "ecc_uncorrected_volatile": 113,
            "ecc_uncorrected_aggregate": 179,
            "gpu_recovery_action": "None",
        }
    )
    for index, pci in enumerate(("0000:4F:00.0", "0000:51:00.0", "0000:57:00.0")):
        rows.append(
            {
                "uuid": f"GPU-test-extra-{index}",
                "pci_bus_id": pci,
                "memory_used_mib": 0,
                "memory_total_mib": 81920,
                "utilization_gpu_percent": 0,
                "ecc_uncorrected_volatile": 0,
                "ecc_uncorrected_aggregate": 0,
                "gpu_recovery_action": "None",
            }
        )
    return rows


def _write_inventory(path: Path, *, rows: list[dict[str, object]] | None = None, **extra: object) -> None:
    payload: dict[str, object] = {
        "schema_version": GPU_INVENTORY_SCHEMA,
        "source_ref": "evidence/gpu-inventory.json",
        "rows": rows if rows is not None else _rows(),
        "compute_apps": [],
        **extra,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(path, payload)


def test_formal_inventory_argument_is_required() -> None:
    with pytest.raises(S206PreparationBlocked, match="GPU_INVENTORY_JSON_REQUIRED"):
        launcher._load_inventory_snapshot(None)


def test_rows_wire_normalizes_aliases_and_retains_bad_card_ecc(tmp_path: Path) -> None:
    rows = _rows()
    for row in rows:
        row["utilization_percent"] = row.pop("utilization_gpu_percent")
        row["ecc_volatile_uncorrected"] = row.pop("ecc_uncorrected_volatile")
        row["ecc_aggregate_uncorrected"] = row.pop("ecc_uncorrected_aggregate")
    path = tmp_path / "gpu-inventory.json"
    _write_inventory(path, rows=rows)
    loaded, identity = launcher._load_inventory_snapshot(path)
    bad = next(row for row in loaded if row["uuid"] == EXCLUDED_UUID)
    assert bad["ecc_volatile_uncorrected"] == 113
    assert bad["ecc_aggregate_uncorrected"] == 179
    assert bad["ecc_uncorrected_volatile"] == 113
    assert bad["ecc_uncorrected_aggregate"] == 179
    assert identity["source_ref"] == "evidence/gpu-inventory.json"
    assert len(str(identity["artifact_hash"])) == 64
    assert len(str(identity["source_sha256"])) == 64


def test_inventory_hash_and_source_hash_tamper_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "gpu-inventory.json"
    _write_inventory(path)
    payload = launcher.load_canonical_json(path)
    assert isinstance(payload, dict)
    payload["rows"][0]["memory_used_mib"] = 1  # type: ignore[index]
    write_canonical_json(path, payload)
    with pytest.raises(S206PreparationBlocked, match="ARTIFACT_HASH_MISMATCH"):
        launcher._load_inventory_snapshot(path)

    _write_inventory(path, source_sha256="0" * 64)
    with pytest.raises(S206PreparationBlocked, match="SOURCE_SHA256_MISMATCH"):
        launcher._load_inventory_snapshot(path)


def test_inventory_source_ref_must_match_resolved_data_root_path(tmp_path: Path) -> None:
    path = tmp_path / "evidence/gpu-inventory.json"
    path.parent.mkdir(parents=True)
    _write_inventory(path)
    _rows, identity = launcher._load_inventory_snapshot(path, data_root=tmp_path)
    assert identity["source_ref"] == "evidence/gpu-inventory.json"

    payload = launcher.load_canonical_json(path)
    assert isinstance(payload, dict)
    payload["source_ref"] = "evidence/another-inventory.json"
    payload["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    write_canonical_json(path, payload)
    with pytest.raises(S206PreparationBlocked, match="SOURCE_REF_PATH_MISMATCH"):
        launcher._load_inventory_snapshot(path, data_root=tmp_path)


@pytest.mark.parametrize(
    "mutation,pattern",
    [
        (lambda rows: rows.pop(), "LIVE_CARD_COUNT_INVALID"),
        (lambda rows: rows.__setitem__(7, {**rows[0]}), "DUPLICATE_UUID"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "memory_used_mib": 1}), "NOT_IDLE"),
        (lambda rows: rows.__setitem__(4, {**rows[4], "pci_bus_id": "0000:52:00.0"}), "IDENTITY_DRIFT"),
    ],
)
def test_inventory_identity_health_negative_cases(mutation, pattern: str) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises(S206PreparationBlocked, match=pattern):
        validate_gpu_inventory(rows)


def test_bad_card_can_never_be_selected() -> None:
    args = SimpleNamespace(cell_anchor=ANCHOR_IDS[0], cell_batch_size=32, cell_gpu_uuid=EXCLUDED_UUID)
    with pytest.raises(S206PreparationBlocked, match="GPU_NOT_APPROVED"):
        launcher._production_cell(args)
