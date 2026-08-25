"""Bounded formal-r20 queue scheduling/identity tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ops.stage2.run_s204_r20_queue import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    _child_command,
    _parse_cell_config,
    _parse_cell_environment,
    lpt_order,
)


def test_lpt_order_is_deterministic_and_longest_first() -> None:
    estimates = {
        "pythia-14m-initialization": 1.0,
        "pythia-31m-initialization": 2.0,
        "pythia-14m-early": 1.0,
        "pythia-31m-early": 2.0,
        "pythia-14m-mid_late": 1.0,
        "pythia-31m-mid_late": 2.0,
    }
    order = lpt_order(estimates)
    assert order[:3] == (
        "pythia-31m-early",
        "pythia-31m-initialization",
        "pythia-31m-mid_late",
    )
    assert set(order) == set(estimates)


def test_child_command_rejects_excluded_or_unknown_gpu() -> None:
    kwargs = {
        "python": "python",
        "launcher": Path("run_s204_formal.py"),
        "execution_commit": "a" * 40,
        "cell_id": "cell",
        "cell_config": Path("cell.json"),
        "g21_evidence": Path("g21.json"),
        "asset_resolution": Path("assets.json"),
        "data_range": Path("data.json"),
        "data_root": Path("data-root"),
        "output_root": Path("output"),
        "runtime_environment": Path("environment.json"),
        "heartbeat_seconds": 30.0,
    }
    for uuid in (EXCLUDED_GPU_UUID, "GPU-unknown"):
        with pytest.raises(ValueError, match="approved r20 UUID"):
            _child_command(gpu_uuid=uuid, **kwargs)

    command = _child_command(gpu_uuid=APPROVED_GPU_UUIDS[0], **kwargs)
    assert "--execution-commit" in command
    assert command[command.index("--cuda-visible-devices") + 1] == APPROVED_GPU_UUIDS[0]
    assert "--execute" in command and "--cell-id" in command


def test_runtime_environment_is_bound_one_to_one_to_cell() -> None:
    values = [
        "cell-a=/tmp/a.json",
        "cell-b=/tmp/b.json",
        "cell-c=/tmp/c.json",
        "cell-d=/tmp/d.json",
        "cell-e=/tmp/e.json",
        "cell-f=/tmp/f.json",
    ]
    cells = _parse_cell_config([f"{cell}=/{cell}.json" for cell in ("cell-a", "cell-b", "cell-c", "cell-d", "cell-e", "cell-f")])
    environments = _parse_cell_environment(values, cells)
    assert set(environments) == set(cells)
    with pytest.raises(ValueError, match="one runtime environment per cell"):
        _parse_cell_environment(values[:-1], cells)
