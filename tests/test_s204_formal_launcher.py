from __future__ import annotations

import json
from pathlib import Path

import pytest

from ops.stage2.run_s204_formal import (
    ASSET_DIGEST,
    APPROVED_GPU_BINDINGS,
    DATA_DIGEST,
    EXCLUDED_PCI,
    EXCLUDED_UUID,
    _Heartbeat,
    _g23_gate,
    _bind_gpu,
    _file_sha256,
    _validate_gpu_smoke_artifact,
    _runtime_cell_paths,
    G21_ARTIFACT,
    build_plan,
    execute_with_task_runtime,
)


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    g21 = {
        "status": "PASS",
        "artifact_hash": G21_ARTIFACT,
        "current_gpu_smoke": {
            "status": "PASS",
            "excluded_pci_bus_ids": [EXCLUDED_PCI],
            "excluded_scheduled": False,
            "excluded_device": {"uuid": EXCLUDED_UUID},
            "allowed_devices": [
                {"pci_bus_id": pci, "uuid": uuid}
                for pci, uuid in APPROVED_GPU_BINDINGS.items()
            ],
        },
    }
    checkpoints = [
        {
            "checkpoint_id": f"cell-{i}",
            "model_id": "pythia-14m" if i < 3 else "pythia-31m-deduped",
            "training_stage": ("initialization", "early", "mid_late")[i % 3],
            "asset_id": f"asset-{i}",
            "initialization_id": f"init-{i}",
            "architecture": "pythia-14m" if i < 3 else "pythia-31m-deduped",
            "revision": f"r{i}",
            "parameter_registry_hash": "a" * 64,
            "state": "ready",
        }
        for i in range(6)
    ]
    assets = {
        "asset_resolution_hash": ASSET_DIGEST,
        "status": "READY",
        "checkpoint_matrix_complete": True,
        "checkpoints": checkpoints,
    }
    data = {
        "data_range_hash": DATA_DIGEST,
        "sample_id_min": 0,
        "sample_id_max_exclusive": 524288,
        "input_sequence_length": 2048,
        "dataset_id": "dataset-formal",
        "revision": "data-r1",
    }
    return g21, assets, data


def test_plan_only_is_six_cell_and_does_not_freeze_b_ref(tmp_path: Path) -> None:
    plan = build_plan(*_inputs(), output_root=tmp_path, candidates=(512, 1024), block_size=32)
    assert plan["cell_count"] == 6
    assert plan["b_ref_status"] == "UNFROZEN_UNTIL_INDEPENDENT_SIZING_PASS"
    assert all(cell["final_sample_count_per_stream"] == "UNFROZEN" for cell in plan["cells"])
    assert all(tmp_path.as_posix() in cell["progress_path"] for cell in plan["cells"])


def test_plan_rejects_missing_failed_gpu_exclusion(tmp_path: Path) -> None:
    g21, assets, data = _inputs()
    g21["current_gpu_smoke"] = {"status": "PASS", "excluded_pci_bus_ids": [], "excluded_scheduled": True}
    with pytest.raises(ValueError, match="GPU exclusion"):
        build_plan(g21, assets, data, output_root=tmp_path)


def test_runtime_launcher_requires_smoke_approved_gpu_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="smoke-approved"):
        execute_with_task_runtime(
            {"cells": []},
            runtime_config_paths=(),
            runtime_environment_path=tmp_path / "environment.json",
            data_root=tmp_path,
            output_root=tmp_path / "out",
            cuda_visible_devices="0,1",
            cell_id=None,
            heartbeat_seconds=1.0,
        )


def test_gpu_binding_rejects_index_reordering(monkeypatch: pytest.MonkeyPatch) -> None:
    approved_pci, approved_uuid = next(iter(APPROVED_GPU_BINDINGS.items()))
    monkeypatch.setattr(
        "ops.stage2.run_s204_formal._gpu_inventory",
        lambda: [
            {"index": "0", "pci_bus_id": EXCLUDED_PCI, "uuid": EXCLUDED_UUID},
            {"index": "2", "pci_bus_id": approved_pci, "uuid": approved_uuid},
        ],
    )
    with pytest.raises(ValueError, match="excluded GPU"):
        _bind_gpu("0")


def test_execute_reloads_hash_bound_g21_smoke_report(tmp_path: Path) -> None:
    report = {
        "status": "PASS",
        "excluded_pci_bus_ids": [EXCLUDED_PCI],
        "excluded_device": {
            "pci_bus_id": EXCLUDED_PCI,
            "uuid": EXCLUDED_UUID,
            "scheduled": False,
        },
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid}
            for pci, uuid in APPROVED_GPU_BINDINGS.items()
        ],
    }
    report_path = tmp_path / "smoke.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    smoke = _validate_gpu_smoke_artifact(
        {
            "gpu_smoke_ref": "smoke.json",
            "gpu_smoke_sha256": _file_sha256(report_path),
        },
        tmp_path,
    )
    assert smoke["sha256"] == _file_sha256(report_path)


def test_runtime_cell_dispatch_and_heartbeat_are_recoverable(tmp_path: Path) -> None:
    paths = []
    cells = [{"cell_id": f"cell-{index}"} for index in range(6)]
    for index in range(6):
        path = tmp_path / f"config-{index}.json"
        path.write_text('{"task_id":"stage2.04_reference_target"}\n', encoding="utf-8")
        paths.append(path)
    selected = _runtime_cell_paths({"cells": cells}, tuple(paths), None)
    assert [item[2] for item in selected] == [f"cell-{index}" for index in range(6)]
    one = _runtime_cell_paths({"cells": cells}, (paths[3],), "cell-3")
    assert [item[2] for item in one] == ["cell-3"]

    heartbeat = tmp_path / "progress.jsonl"
    with _Heartbeat(heartbeat, "cell-3", 0.01):
        pass
    phases = [row.split('"phase": "', 1)[1].split('"', 1)[0] for row in heartbeat.read_text(encoding="utf-8").splitlines()]
    assert phases[0] == "STARTED"
    assert phases[-1] == "STOPPED"


def test_g23_qualification_is_fail_closed_without_complete_metrics(tmp_path: Path) -> None:
    plan = {"cells": [{"cell_id": f"cell-{index}"} for index in range(6)]}
    task_hashes = {f"cell-{index}": f"{index + 1:064x}" for index in range(6)}
    bundle_hashes = {f"cell-{index}": f"{index + 10:064x}" for index in range(6)}
    status, gate_hash = _g23_gate(
        plan=plan,
        metrics_path=None,
        task_result_refs=tuple(f"runs/cell-{index}/task-run-result.json" for index in range(6)),
        task_result_hashes=task_hashes,
        bundle_hashes=bundle_hashes,
        output_root=tmp_path,
    )
    assert status == "BLOCKED"
    assert gate_hash == ""
    preflights = list((tmp_path / "g2.3-preflight").glob("*/preflight.json"))
    assert len(preflights) == 1
    preflight = json.loads(preflights[0].read_text(encoding="utf-8"))
    assert preflight["status"] == "BLOCKED"
    assert preflight["formal_eligible"] is False
