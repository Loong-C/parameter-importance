"""Fail-closed and content-addressing tests for the independent G2.3 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    EXPECTED_CELL_IDS,
    G23Blocked,
    _array,
    _bootstrap_independent_bias_interval,
    _bootstrap_independent_cross_interval,
    _bootstrap_u_diagnostics,
    _bounded_moments_strict,
    _moments_from_blocks,
    _pearson,
    _top_overlap,
    _validate_six_cell_manifest,
    evaluate_formal_g23,
)
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_formal import (
    ReferenceSizingPlan,
    _BoundedMoments,
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


def _six_cell_manifest_for_registry_hashes(
    registry_hashes: tuple[str, ...],
    *,
    include_map: bool = True,
) -> dict[str, object]:
    assert len(registry_hashes) == len(EXPECTED_CELL_IDS)
    rows = [
        {
            "cell_id": cell_id,
            "model_id": cell_id.split(":", 1)[0],
            "training_stage": cell_id.split(":", 1)[1],
            "checkpoint_id": f"checkpoint-{index}",
            "checkpoint_hash": f"{index + 1:064x}",
            "checkpoint_revision": f"revision-{index}",
            "registry_hash": registry_hash,
            "config_hash": f"{index + 101:064x}",
        }
        for index, (cell_id, registry_hash) in enumerate(zip(EXPECTED_CELL_IDS, registry_hashes))
    ]
    by_cell = dict(zip(EXPECTED_CELL_IDS, registry_hashes))
    body: dict[str, object] = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY",
        "scope": "formal",
        "asset_resolution_hash": "a" * 64,
        "asset_producer_commit": "b" * 40,
        "asset_execution_commit": "c" * 40,
        "checkpoints": rows,
        "data": {"data_range_hash": "d" * 64},
        "data_range_hash": "d" * 64,
        "registry_hash": (
            next(iter(set(registry_hashes)))
            if len(set(registry_hashes)) == 1
            else canonical_json_hash(by_cell)
        ),
    }
    if include_map:
        body["registry_hashes_by_cell"] = by_cell
    body["manifest_hash"] = canonical_json_hash(body)
    return body


def test_six_cell_manifest_binds_model_specific_registry_hashes() -> None:
    model_specific = ("1" * 64,) * 3 + ("2" * 64,) * 3

    rows = _validate_six_cell_manifest(
        _six_cell_manifest_for_registry_hashes(model_specific)
    )

    assert tuple(row["cell_id"] for row in rows) == EXPECTED_CELL_IDS
    assert tuple(row["registry_hash"] for row in rows) == model_specific


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        ("map", "REGISTRY_ROW_MAP_MISMATCH"),
        ("row", "REGISTRY_ROW_MAP_MISMATCH"),
        ("top", "REGISTRY_DIGEST_MISMATCH"),
    ),
)
def test_six_cell_manifest_registry_binding_tamper_is_rejected(
    tamper: str,
    expected_error: str,
) -> None:
    model_specific = ("1" * 64,) * 3 + ("2" * 64,) * 3
    manifest = _six_cell_manifest_for_registry_hashes(model_specific)
    if tamper == "map":
        assert isinstance(manifest["registry_hashes_by_cell"], dict)
        manifest["registry_hashes_by_cell"][EXPECTED_CELL_IDS[0]] = "3" * 64  # type: ignore[index]
    elif tamper == "row":
        assert isinstance(manifest["checkpoints"], list)
        manifest["checkpoints"][0]["registry_hash"] = "3" * 64  # type: ignore[index]
    else:
        manifest["registry_hash"] = "3" * 64
    manifest["manifest_hash"] = canonical_json_hash(
        {key: item for key, item in manifest.items() if key != "manifest_hash"}
    )

    with pytest.raises(G23Blocked, match=expected_error):
        _validate_six_cell_manifest(manifest)


def test_six_cell_manifest_legacy_common_registry_form_is_explicit() -> None:
    common = ("1" * 64,) * 6
    legacy = _six_cell_manifest_for_registry_hashes(common, include_map=False)
    assert tuple(row["registry_hash"] for row in _validate_six_cell_manifest(legacy)) == common

    model_specific_without_map = _six_cell_manifest_for_registry_hashes(
        ("1" * 64,) * 3 + ("2" * 64,) * 3,
        include_map=False,
    )
    with pytest.raises(G23Blocked, match="LEGACY_COMMON_REGISTRY_REQUIRED"):
        _validate_six_cell_manifest(model_specific_without_map)


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


def test_bounded_final_moments_missing_higher_fields_are_blocked() -> None:
    moments = _BoundedMoments(include_higher=False)
    moments.update_vector({"p": np.asarray([1.0, 2.0])}, 1.0)

    with pytest.raises(G23Blocked, match="HIGHER_MOMENTS_REQUIRED"):
        _bounded_moments_strict(
            moments.to_state(), "bounded.final", require_higher=True
        )


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
