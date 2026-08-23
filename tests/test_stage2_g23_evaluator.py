"""Fail-closed and content-addressing tests for the independent G2.3 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    _array,
    _moments_from_blocks,
    _pearson,
    _top_overlap,
    evaluate_formal_g23,
)
from param_importance_nlp.experiments.stage2_formal import ReferenceSizingPlan
from param_importance_nlp.experiments.stage23_task_runners import _reference_capacity_preflight


def test_missing_cells_are_not_a_formal_decision(tmp_path: Path) -> None:
    result = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    assert result["status"] == "NOT_RUN"
    assert result["formal_eligible"] is False
    assert result["complete_cell_count"] == 0
    assert (tmp_path / "attempts" / "g2.3-attempts" / result["artifact_hash"] / "evaluation.json").is_file()


def test_partial_cell_set_is_blocked_and_does_not_lock_next_attempt(tmp_path: Path) -> None:
    cells = [CellInput("cell-0", "runs/cell-0/task-run-result.json")]
    first = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert first["status"] == "BLOCKED"
    # A later complete set gets a different content address; no stale partial
    # attempt can be overwritten or treated as the current formal decision.
    second = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    assert first["artifact_hash"] != second["artifact_hash"]
    lines = (tmp_path / "attempts" / "g2.3-attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert first["artifact_hash"] in lines and second["artifact_hash"] in lines


def test_nonfinite_raw_array_is_rejected_closed() -> None:
    with pytest.raises(ValueError, match="NON_FINITE"):
        _array([1.0, float("nan")], "diagnostic")


def test_boundary_metrics_use_inclusive_preregistered_comparisons() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0])
    right = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert _pearson(left, right) >= 0.995
    assert _top_overlap(left, right, 0.01) >= 0.98


def test_weighted_u_hand_calculation_is_recomputed_from_raw_blocks() -> None:
    blocks = [
        {"p": np.asarray([1.0])},
        {"p": np.asarray([3.0])},
    ]
    moments = _moments_from_blocks(blocks, [1.0, 2.0], "hand")
    # n1=3, n2=5, g1=7, g2=38 => (49-38)/(9-5)=2.75.
    from param_importance_nlp.experiments.stage2_g23_evaluator import _u_from_moments

    assert np.array_equal(_u_from_moments(moments, "hand.u")["p"], np.asarray([2.75]))


def test_capacity_preflight_uses_full_14m_and_31m_counts(tmp_path: Path) -> None:
    class _Provider:
        pass

    plan = ReferenceSizingPlan(
        reference_id="capacity-test",
        candidate_sample_counts=(2, 4),
        block_size=1,
        convergence_tolerance=0.02,
        required_consecutive=1,
    )
    reports = [
        _reference_capacity_preflight(_Provider(), plan, tmp_path, model_manifest={"parameter_count": count})
        for count in (14_000_000, 31_000_000)
    ]
    assert [item["parameter_count"] for item in reports] == [14_000_000, 31_000_000]
    for item, count in zip(reports, (14_000_000, 31_000_000)):
        assert item["single_copy_shard_bytes"] == 4 * 2 * count * 8
        assert item["snapshot_moment_bytes"] == 4 * 4 * count * 8
        assert item["disk_ok"] is True and item["ram_ok"] is True


def test_attempt_json_is_hash_bound_and_tamper_detected(tmp_path: Path) -> None:
    result = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    path = tmp_path / "attempts" / "g2.3-attempts" / result["artifact_hash"] / "evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "PASS"
    path.write_text(json.dumps(payload), encoding="utf-8")
    # Re-evaluation does not trust an existing attempt and must detect the
    # content-address collision rather than silently accepting a tampered file.
    with pytest.raises(RuntimeError, match="CONTENT_ADDRESS_COLLISION"):
        evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
