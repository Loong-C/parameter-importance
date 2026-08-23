from __future__ import annotations

from pathlib import Path

import pytest

from ops.stage2.run_s204_formal import (
    ASSET_DIGEST,
    DATA_DIGEST,
    EXCLUDED_PCI,
    G21_ARTIFACT,
    build_plan,
)


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    g21 = {
        "status": "PASS",
        "artifact_hash": G21_ARTIFACT,
        "current_gpu_smoke": {
            "status": "PASS",
            "excluded_pci_bus_ids": [EXCLUDED_PCI],
            "excluded_scheduled": False,
        },
    }
    checkpoints = [
        {
            "checkpoint_id": f"cell-{i}",
            "model_id": "pythia-14m" if i < 3 else "pythia-31m-deduped",
            "training_stage": ("initialization", "early", "mid_late")[i % 3],
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
