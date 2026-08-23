"""Fail-closed and content-addressing tests for the independent G2.3 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    _array,
    _bootstrap_independent_bias_interval,
    _bootstrap_independent_cross_interval,
    _bootstrap_u_diagnostics,
    _moments_from_blocks,
    _pearson,
    _top_overlap,
    evaluate_formal_g23,
)
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_formal import (
    ReferenceSizingPlan,
    _ReferenceShardStore,
    _moments_from_shards,
    estimate_reference_uncertainty_shards,
)
from param_importance_nlp.experiments.stage2_g23_contracts import (
    generator_boundary,
    validate_generator_boundary,
    validate_resume_prefix,
    validate_sizing_plan_contract,
    validate_weighting_contract,
)
from param_importance_nlp.experiments.stage23_task_runners import _actual_sampling_state, _reference_capacity_preflight, _sizing_delta_sci


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
    # n1=3, n2=5, g1=7, g2=37 => (49-37)/(9-5)=3.
    from param_importance_nlp.experiments.stage2_g23_evaluator import _u_from_moments

    assert np.array_equal(_u_from_moments(moments, "hand.u")["p"], np.asarray([3.0]))
    assumptions = validate_weighting_contract({
        "statistical_unit": "sequence",
        "weight_unit": "tokens",
        "sampling_design": "with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    })
    assert assumptions["weights_exogenous"] is True
    with pytest.raises(ValueError, match="WEIGHTED_U_ASSUMPTIONS_NOT_DECLARED"):
        validate_weighting_contract({
            **assumptions,
            "common_mean_assumption": False,
        })


def test_content_addressed_shard_dedup_and_weighted_jackknife(tmp_path: Path) -> None:
    store = _ReferenceShardStore(tmp_path / "sizing")
    vector = {"p": np.asarray([1.0, 2.0])}
    first = store.publish(vector, 1.0)
    duplicate = store.publish(vector, 1.0)
    assert first == duplicate
    refs = [first, store.publish({"p": np.asarray([2.0, 4.0])}, 2.0), store.publish({"p": np.asarray([3.0, 6.0])}, 3.0)]
    assumptions = {
        "statistical_unit": "sequence",
        "weight_unit": "tokens",
        "sampling_design": "with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    moments = _moments_from_shards(store, refs, assumptions)
    assert moments.count == 3 and moments.n1 == 6.0
    uncertainty = estimate_reference_uncertainty_shards(store, refs, refs, assumptions)
    assert uncertainty.block_count_a == 3 and uncertainty.block_count_b == 3
    assert all(np.all(np.isfinite(value)) for value in uncertainty.bias_variance.values())


def test_independent_ab_bootstrap_and_endpoint_h_ref_are_block_bootstrapped() -> None:
    blocks_a = [{"p": np.asarray([value, value + 1.0])} for value in (1.0, 2.0, 3.0)]
    blocks_b = [{"p": np.asarray([value, value + 1.0])} for value in (1.5, 2.5, 3.5)]
    weights = [1.0, 2.0, 1.0]
    cross_low, cross_high = _bootstrap_independent_cross_interval(blocks_a, weights, blocks_b, weights, "hand.cross")
    bias_low, bias_high = _bootstrap_independent_bias_interval(blocks_a, weights, blocks_b, weights, "hand.bias")
    center = {"p": np.asarray([2.0, 3.0])}
    h_ref, model_half, layer_q95, module_q95 = _bootstrap_u_diagnostics(
        blocks_a,
        blocks_b,
        weights,
        weights,
        center,
        {"layer0": ["p"]},
        {"module0": ["p"]},
    )
    assert np.all(cross_low <= cross_high) and np.all(bias_low <= bias_high)
    assert np.isfinite(h_ref) and np.isfinite(model_half) and np.isfinite(layer_q95) and np.isfinite(module_q95)


def test_sampling_rng_state_is_replayable_from_frozen_manifest() -> None:
    sampling = SamplingPlan(
        universe=SamplingUniverse("hand-universe", (0, 1, 2, 3)),
        stream_seeds={"reference_sizing": 7, "reference_A": 11, "reference_B": 13, "pilot": 17, "confirmatory": 19},
    )
    state = _actual_sampling_state(sampling, "reference_A", 3)
    assert state["stream"] == "reference_A" and state["count"] == 3
    assert state["state_before_sha256"] != state["state_after_sha256"]
    assert state == generator_boundary(sampling, "reference_A", 3)
    validate_generator_boundary(state, sampling=sampling, stream="reference_A", count=3, field="hand.rng")


def test_canonical_sizing_and_resume_validators_are_strict() -> None:
    plan = {
        "schema_version": "stage2-reference-sizing-plan-v1",
        "candidate_sample_counts": [2, 4, 8],
        "block_size": 2,
        "required_consecutive": 1,
    }
    assert validate_sizing_plan_contract(plan, selected_sample_count=4) == (2, 4, 8)
    with pytest.raises(ValueError, match="ADJACENT_DOUBLING_REQUIRED"):
        validate_sizing_plan_contract({**plan, "candidate_sample_counts": [2, 6]})
    first = [{"shard_hash": "a"}]
    second = [{"shard_hash": "a"}, {"shard_hash": "b"}]
    validate_resume_prefix(first, [], second, [], field="hand.resume")
    with pytest.raises(ValueError, match="PREFIX_DRIFT_A"):
        validate_resume_prefix(first, [], [{"shard_hash": "x"}], [], field="hand.resume")


def test_sizing_delta_formula_uses_noise_or_signal_floor_at_boundary() -> None:
    assert _sizing_delta_sci(100.0, 2.0) == pytest.approx(1.0)
    assert _sizing_delta_sci(100.0, 30.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        _sizing_delta_sci(float("nan"), 1.0)


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
