from __future__ import annotations

import itertools

import pytest
import torch

from param_importance_nlp.core import (
    PLUGIN_SAME_BATCH_CLIP,
    UNBIASED_FIXED_STATE,
    CoreContractError,
    EqualSufficientStatistics,
    EstimatorResult,
    ImportanceAccumulator,
    ParameterRegistry,
    TensorMap,
    WeightedSufficientStatistics,
    cross_u_importance,
    double_sample_importance,
    equal_u_importance,
    global_clip_factor,
    raw_importance,
    weighted_u_importance,
)


def _sample(*values: float) -> TensorMap:
    return TensorMap({"p": torch.tensor(values, dtype=torch.float64)})


def test_equal_u_matches_ordered_pair_oracle_and_is_permutation_invariant() -> None:
    samples = [_sample(1.0, 2.0), _sample(3.0, -2.0), _sample(-1.0, 4.0)]
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    result = equal_u_importance(statistics)["p"]
    ordered = [samples[i]["p"] * samples[j]["p"] for i, j in itertools.permutations(range(3), 2)]
    oracle = torch.stack(ordered).mean(dim=0)
    torch.testing.assert_close(result, oracle)

    reversed_result = equal_u_importance(
        EqualSufficientStatistics.from_samples(list(reversed(samples)), accumulation_dtype=torch.float64)
    )["p"]
    torch.testing.assert_close(result, reversed_result)


def test_u_boundaries_negative_value_and_double_equivalence() -> None:
    first = _sample(2.0, -3.0)
    second = _sample(-2.0, 3.0)
    statistics = EqualSufficientStatistics.from_samples([first, second], accumulation_dtype=torch.float64)
    u = equal_u_importance(statistics)
    double = double_sample_importance(first, second)
    torch.testing.assert_close(u["p"], double["p"])
    assert bool((u["p"] < 0).all())

    one = EqualSufficientStatistics.from_samples([first], accumulation_dtype=torch.float64)
    try:
        equal_u_importance(one)
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("M=1 必须失败")


def test_weighted_u_matches_explicit_weighted_cross_pairs_and_equal_limit() -> None:
    samples = [_sample(1.0, 2.0), _sample(3.0, -2.0), _sample(-1.0, 4.0)]
    weights = [1.0, 2.0, 4.0]
    statistics = WeightedSufficientStatistics.from_samples(
        samples,
        weights,
        accumulation_dtype=torch.float64,
        statistical_unit="target_token_microbatch",
        weight_unit="effective_target_tokens",
        sampling_design="iid_disjoint_microbatches",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    result = weighted_u_importance(statistics, require_unbiasedness_assumptions=True)["p"]
    numerator = torch.zeros(2, dtype=torch.float64)
    denominator = 0.0
    for i, j in itertools.permutations(range(3), 2):
        numerator += weights[i] * weights[j] * samples[i]["p"] * samples[j]["p"]
        denominator += weights[i] * weights[j]
    torch.testing.assert_close(result, numerator / denominator)

    equal_weights = WeightedSufficientStatistics.from_samples(
        samples,
        [2, 2, 2],
        accumulation_dtype=torch.float64,
        statistical_unit="microbatch",
        weight_unit="sample",
        sampling_design="iid_disjoint_microbatches",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    torch.testing.assert_close(
        weighted_u_importance(equal_weights)["p"],
        equal_u_importance(EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64))["p"],
    )


def test_cross_u_and_raw_are_distinct_targets() -> None:
    x = [_sample(1.0, 2.0), _sample(3.0, 4.0), _sample(5.0, 6.0)]
    y = [_sample(2.0, 1.0), _sample(4.0, 3.0), _sample(6.0, 5.0)]
    cross = cross_u_importance(x, y)
    explicit = torch.stack(
        [x[i]["p"] * y[j]["p"] for i, j in itertools.permutations(range(3), 2)]
    ).mean(dim=0)
    torch.testing.assert_close(cross["p"], explicit)

    mean = EqualSufficientStatistics.from_samples(x, accumulation_dtype=torch.float64).mean_gradient
    raw = raw_importance(mean)
    assert bool((raw["p"] >= 0).all())
    assert not torch.equal(raw["p"], equal_u_importance(EqualSufficientStatistics.from_samples(x, accumulation_dtype=torch.float64))["p"])


class _TwoGroupModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.Parameter(torch.tensor([2.0]))


def test_estimator_result_applies_per_group_lr_and_same_batch_clip_is_plugin() -> None:
    model = _TwoGroupModel()
    optimizer = torch.optim.SGD(
        [{"params": [model.a], "lr": 0.1}, {"params": [model.b], "lr": 0.2}]
    )
    registry = ParameterRegistry.from_model(model, optimizer)
    core = TensorMap({"a": torch.tensor([3.0]), "b": torch.tensor([5.0])}, registry=registry)
    result = EstimatorResult.from_core(
        "local_gradient_space_importance_u_clipped",
        core,
        {"group_0000": 0.1, "group_0001": 0.2},
        clip_factor=0.5,
        clip_source="same_batch_mean_gradient",
        unbiasedness_claim=UNBIASED_FIXED_STATE,
    )
    torch.testing.assert_close(result.core["a"], torch.tensor([3.0]))
    torch.testing.assert_close(result.score["a"], torch.tensor([0.15]))
    torch.testing.assert_close(result.score["b"], torch.tensor([0.5]))
    assert result.unbiasedness_claim == PLUGIN_SAME_BATCH_CLIP

    factor = global_clip_factor(TensorMap({"p": torch.tensor([3.0, 4.0])}), 2.5, epsilon=1e-20)
    assert abs(factor - 0.5) < 1e-12


def test_unbiasedness_claim_is_fail_closed_and_weighted_assumptions_are_recorded() -> None:
    sample_a = _sample(1.0, 2.0)
    sample_b = _sample(3.0, 4.0)
    equal_statistics = EqualSufficientStatistics.from_samples(
        [sample_a, sample_b], accumulation_dtype=torch.float64
    )
    equal_result = EstimatorResult.from_equal_u(equal_statistics, {"default": 0.1})
    assert equal_result.unbiasedness_claim == UNBIASED_FIXED_STATE
    clipped_equal = EstimatorResult.from_equal_u(
        equal_statistics,
        {"default": 0.1},
        clip_factor=0.5,
        clip_source="same_batch_mean_gradient",
    )
    assert clipped_equal.estimator_name == "local_gradient_space_importance_u_clipped"
    assert clipped_equal.unbiasedness_claim == PLUGIN_SAME_BATCH_CLIP

    weighted_statistics = WeightedSufficientStatistics.from_samples(
        [sample_a, sample_b],
        [2, 3],
        accumulation_dtype=torch.float64,
        statistical_unit="target_token_microbatch",
        weight_unit="effective_target_tokens",
        sampling_design="iid_disjoint_microbatches",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    weighted_result = EstimatorResult.from_weighted_u(
        weighted_statistics, {"default": 0.2}
    )
    assert weighted_result.unbiasedness_claim == UNBIASED_FIXED_STATE
    assert weighted_result.metadata["weights_exogenous"] is True
    assert weighted_result.metadata["common_mean_assumption"] is True
    clipped_weighted = EstimatorResult.from_weighted_u(
        weighted_statistics,
        {"default": 0.2},
        clip_factor=0.5,
        clip_source="same_batch_mean_gradient",
    )
    assert clipped_weighted.estimator_name == "local_gradient_space_importance_u_clipped"

    try:
        EstimatorResult.from_core(
            "local_gradient_space_importance_raw",
            raw_importance(equal_statistics.mean_gradient),
            {"default": 0.1},
            unbiasedness_claim=UNBIASED_FIXED_STATE,
        )
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("raw estimator 不能冒充 fixed-state 无偏 U")

    try:
        EstimatorResult.from_core(
            "local_gradient_space_importance_u",
            equal_u_importance(equal_statistics),
            {"default": 0.1},
            clip_factor=0.5,
            clip_source="independent",
            unbiasedness_claim=UNBIASED_FIXED_STATE,
        )
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("任何被裁剪结果都不能保留未裁剪 U 的严格声明")


def test_sufficient_statistics_reject_declared_dtype_drift_and_non_integer_count() -> None:
    """声明精度必须等于实际张量精度；bool/小数不能冒充 int64 计数。"""

    value64 = TensorMap({"p": torch.tensor([1.0], dtype=torch.float64)})
    with pytest.raises(CoreContractError, match="accumulation_dtype"):
        EqualSufficientStatistics(
            count=1,
            s1=value64,
            s2=value64,
            accumulation_dtype=torch.float32,
        )
    with pytest.raises(CoreContractError, match="count"):
        EqualSufficientStatistics(
            count=True,
            s1=value64,
            s2=value64,
            accumulation_dtype=torch.float64,
        )
    with pytest.raises(CoreContractError, match="count"):
        WeightedSufficientStatistics(
            count=1.5,  # type: ignore[arg-type]
            g1=value64,
            g2=value64,
            n1=1.0,
            n2=1.0,
            accumulation_dtype=torch.float64,
            statistical_unit="microbatch",
            weight_unit="effective_token",
            sampling_design="iid",
            weights_exogenous=True,
            common_mean_assumption=True,
        )


def test_accumulator_four_identities_movement_magnitude_and_skip() -> None:
    template = TensorMap({"p": torch.zeros(3, dtype=torch.float64)})
    accumulator = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    accumulator.add_step(
        TensorMap({"p": torch.tensor([2.0, -3.0, 0.0])}),
        raw=TensorMap({"p": torch.tensor([4.0, 9.0, 0.0])}),
        data_update=TensorMap({"p": torch.tensor([-1.0, 2.0, 0.5])}),
        total_update=TensorMap({"p": torch.tensor([-1.5, 1.5, 0.25])}),
        current_parameters=TensorMap({"p": torch.tensor([-5.0, 2.0, 1.0])}),
    )
    accumulator.add_step(
        TensorMap({"p": torch.tensor([-1.0, 1.0, -2.0])}),
        data_update=TensorMap({"p": torch.tensor([0.5, -1.0, 0.5])}),
        total_update=TensorMap({"p": torch.tensor([0.5, -0.5, -0.25])}),
    )
    accumulator.record_skip()
    accumulator.validate_invariants()
    expected = lambda values: torch.tensor(values, dtype=torch.float64)
    torch.testing.assert_close(accumulator.signed["p"], expected([1.0, -2.0, -2.0]))
    torch.testing.assert_close(accumulator.positive["p"], expected([2.0, 1.0, 0.0]))
    torch.testing.assert_close(accumulator.negative_mass["p"], expected([1.0, 3.0, 2.0]))
    torch.testing.assert_close(accumulator.absolute["p"], expected([3.0, 4.0, 2.0]))
    torch.testing.assert_close(accumulator.data_movement["p"], expected([1.5, 3.0, 1.0]))
    torch.testing.assert_close(accumulator.net_data_movement["p"], expected([0.5, 1.0, 1.0]))
    torch.testing.assert_close(accumulator.total_endpoint_movement["p"], expected([1.0, 1.0, 0.0]))
    torch.testing.assert_close(accumulator.magnitude["p"], expected([5.0, 2.0, 1.0]))
    assert accumulator.successful_steps == 2
    assert accumulator.skipped_steps == 1

    restored = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    restored.load_state_dict(accumulator.state_dict())
    torch.testing.assert_close(restored.signed["p"], accumulator.signed["p"])
    torch.testing.assert_close(restored.total_endpoint_movement["p"], accumulator.total_endpoint_movement["p"])


def test_accumulator_migrates_03x_v1_state_without_inventing_new_views() -> None:
    """旧状态可恢复，但不可从 magnitude 猜测参数符号或裁剪/weight-decay 分解。"""

    template = TensorMap({"p": torch.zeros(2, dtype=torch.float64)})
    source = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    source.add_step(
        TensorMap({"p": torch.tensor([1.0, -2.0], dtype=torch.float64)}),
        raw=TensorMap({"p": torch.tensor([3.0, 4.0], dtype=torch.float64)}),
        data_update=TensorMap({"p": torch.tensor([-0.1, 0.2], dtype=torch.float64)}),
        total_update=TensorMap({"p": torch.tensor([-0.2, 0.1], dtype=torch.float64)}),
        current_parameters=TensorMap(
            {"p": torch.tensor([-5.0, 6.0], dtype=torch.float64)}
        ),
    )
    current = source.state_dict()
    v1_keys = {
        "accumulation_dtype", "successful_steps", "skipped_steps", "positive",
        "negative_mass", "raw", "data_movement", "data_displacement",
        "total_movement", "total_displacement", "magnitude",
    }
    legacy = {"version": 1, **{key: current[key] for key in v1_keys}}

    restored = ImportanceAccumulator(template, accumulation_dtype=torch.float64)
    restored.load_state_dict(legacy)

    torch.testing.assert_close(restored.signed["p"], source.signed["p"])
    torch.testing.assert_close(restored.raw["p"], source.raw["p"])
    torch.testing.assert_close(restored.raw_clipped["p"], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(
        restored.weight_decay_movement["p"], torch.zeros(2, dtype=torch.float64)
    )
    assert restored.state_dict()["version"] == 2
    assert restored.state_dict()["has_initial_parameters"] is False
