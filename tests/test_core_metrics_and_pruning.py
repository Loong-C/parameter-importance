from __future__ import annotations

import math

import pytest
import torch

from param_importance_nlp.core import (
    CoordinateSelection,
    CoreContractError,
    PruningContext,
    PruningPlan,
    TensorMap,
    bias,
    confidence_interval,
    cosine_similarity,
    damage_auc,
    empirical_fisher,
    effective_parameter_count,
    entropy,
    gini_coefficient,
    hhi,
    mae,
    mse,
    normalized_l1_error,
    normalized_l2_error,
    normalized_linf_error,
    pearson_correlation,
    select_pruned_coordinates,
    sign_agreement,
    spearman_correlation,
    top_k_jaccard,
    top_k_overlap,
    top_q_mass,
    synaptic_intelligence,
    variance,
)


def test_error_and_vector_metrics() -> None:
    estimates = [1.0, 2.0, 3.0]
    assert bias(estimates, 1.0).value == 1.0
    assert variance(estimates).value == 1.0
    assert mse(estimates, 1.0).value == 5.0 / 3.0
    assert mae(estimates, 1.0).value == 1.0
    assert math.isclose(pearson_correlation([1, 2, 3], [2, 4, 6]).value or 0, 1.0)
    assert math.isclose(spearman_correlation([1, 2, 2, 4], [10, 20, 20, 40]).value or 0, 1.0)
    assert normalized_l1_error([1, 3], [1, 1]).value == 1.0
    assert math.isclose(normalized_l2_error([1, 3], [1, 1]).value or 0, math.sqrt(2))
    assert normalized_linf_error([1, 3], [1, 1]).value == 2.0
    assert cosine_similarity([1, 0], [1, 0]).value == 1.0
    assert sign_agreement([1, -2, 0], [2, -1, 0], active_threshold=0).value == 1.0


def test_undefined_semantics_are_explicit() -> None:
    assert not pearson_correlation([1], [1]).defined
    assert pearson_correlation([1, 1], [2, 3]).reason == "constant_vector"
    assert not cosine_similarity([0, 0], [1, 2]).defined
    assert not gini_coefficient([0, 0]).defined
    assert not effective_parameter_count([1]).defined
    assert not variance([1]).defined
    assert not sign_agreement([1, 2], [0, 0]).defined

    # 未提供 canonical ID 时，top-k 边界并列没有唯一集合，必须显式 undefined。
    overlap = top_k_overlap([2, 1, 1], [2, 1, 1], 2)
    assert not overlap.defined
    assert "non_unique" in (overlap.reason or "")


def test_top_k_and_mass_metrics_with_canonical_tie_break() -> None:
    ids = ["a", "b", "c", "d"]
    left = [4, 3, 2, 1]
    right = [4, 2, 3, 1]
    assert top_k_overlap(left, right, 2, coordinate_ids=ids).value == 0.5
    assert math.isclose(top_k_jaccard(left, right, 2, coordinate_ids=ids).value or 0, 1 / 3)

    masses = [4.0, 3.0, 2.0, 1.0]
    assert 0 <= (gini_coefficient(masses).value or 0) <= 1
    assert 0 <= (entropy(masses, normalized=True).value or 0) <= 1
    assert math.isclose(hhi([1, 1, 1, 1]).value or 0, 0.25)
    assert math.isclose(effective_parameter_count([1, 1, 1, 1]).value or 0, 4.0)
    assert math.isclose(top_q_mass(masses, 0.5, coordinate_ids=ids).value or 0, 0.7)


def test_confidence_interval_and_damage_auc() -> None:
    interval = confidence_interval([1, 2, 3, 4, 5])
    assert interval.defined
    assert interval.value is not None and interval.value > 0
    assert interval.details["lower"] < interval.details["mean"] < interval.details["upper"]
    assert math.isclose(damage_auc([0, 0.5, 1], [0, 1, 2]).value or 0, 1.0)


def test_pruning_canonical_tie_break_random_determinism_and_restore() -> None:
    scores = TensorMap(
        {
            "layer_a.weight": torch.tensor([5.0, 5.0, 1.0]),
            "layer_b.weight": torch.tensor([4.0, 2.0, 0.0]),
        }
    )
    high = select_pruned_coordinates(scores, PruningPlan(0.5, "high"))
    assert high.selected_count == 3
    assert high.coordinate_ids == (
        "layer_a.weight#000000000000",
        "layer_a.weight#000000000001",
        "layer_b.weight#000000000000",
    )
    random_a = select_pruned_coordinates(scores, PruningPlan(0.5, "random", seed=42))
    random_b = select_pruned_coordinates(scores, PruningPlan(0.5, "random", seed=42))
    assert random_a.coordinate_ids == random_b.coordinate_ids

    parameters = TensorMap(
        {
            "layer_a.weight": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0])),
            "layer_b.weight": torch.nn.Parameter(torch.tensor([4.0, 5.0, 6.0])),
        }
    )
    original = parameters.clone()
    with PruningContext(parameters, high):
        assert parameters["layer_a.weight"].tolist() == [0.0, 0.0, 3.0]
        assert parameters["layer_b.weight"].tolist() == [0.0, 5.0, 6.0]
    torch.testing.assert_close(parameters["layer_a.weight"], original["layer_a.weight"])
    torch.testing.assert_close(parameters["layer_b.weight"], original["layer_b.weight"])


def test_layer_balanced_pruning_keeps_exact_global_k() -> None:
    scores = TensorMap(
        {
            "l1.weight": torch.arange(3, dtype=torch.float32),
            "l2.weight": torch.arange(7, dtype=torch.float32),
        }
    )
    selection = select_pruned_coordinates(scores, PruningPlan(0.4, "low", scope="layer_balanced"))
    assert selection.selected_count == math.floor(0.4 * 10)
    assert sum(len(indices) for indices in selection.indices.values()) == selection.selected_count


def test_pruning_context_rejects_invalid_selection_before_mutation() -> None:
    parameters = TensorMap({"weight": torch.nn.Parameter(torch.tensor([1.0, 2.0]))})
    invalid = CoordinateSelection(
        indices={"weight": (0, 2)},
        selected_count=2,
        eligible_count=2,
        coordinate_ids=(
            "weight#000000000000",
            "weight#000000000002",
        ),
    )
    before = parameters["weight"].detach().clone()

    with pytest.raises(CoreContractError, match="index"):
        with PruningContext(parameters, invalid):
            raise AssertionError("不应进入上下文")
    torch.testing.assert_close(parameters["weight"], before)


def test_empirical_fisher_and_si_baselines() -> None:
    gradients = [
        TensorMap({"p": torch.tensor([1.0, 2.0])}),
        TensorMap({"p": torch.tensor([3.0, 4.0])}),
    ]
    fisher = empirical_fisher(gradients)
    torch.testing.assert_close(fisher["p"], torch.tensor([5.0, 10.0], dtype=torch.float64))

    updates = [
        TensorMap({"p": torch.tensor([-0.5, -0.5])}),
        TensorMap({"p": torch.tensor([-0.5, 0.5])}),
    ]
    initial = TensorMap({"p": torch.tensor([1.0, 1.0])})
    final = TensorMap({"p": torch.tensor([0.0, 1.0])})
    si = synaptic_intelligence(gradients, updates, initial, final, xi=1.0)
    # numerator=[2, -1] -> clamp=[2,0]；task delta^2+xi=[2,1]。
    torch.testing.assert_close(si["p"], torch.tensor([1.0, 0.0], dtype=torch.float64))
