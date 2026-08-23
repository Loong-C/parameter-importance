"""Regression checks for strict fail-closed G2.3 input handling.

The complete PASS fixture lives in ``test_stage2_g23_real_external_fixture``;
this module keeps the old test path focused on structural rejection and never
publishes pseudo formal evidence.
"""

from __future__ import annotations

from pathlib import Path

from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    EXPECTED_CELL_IDS,
    evaluate_formal_g23,
)


def test_unbound_duplicate_cell_set_is_blocked(tmp_path: Path) -> None:
    cells = [CellInput(cell, f"missing/{index}.json") for index, cell in enumerate(EXPECTED_CELL_IDS)]
    cells[-1] = cells[0]
    result = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False


def test_incomplete_cell_set_is_not_run(tmp_path: Path) -> None:
    cells = [CellInput(cell, f"missing/{index}.json") for index, cell in enumerate(EXPECTED_CELL_IDS[:-1])]
    result = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert result["status"] == "NOT_RUN"
    assert result["formal_eligible"] is False
