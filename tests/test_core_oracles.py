from __future__ import annotations

import itertools
import math

import torch

from param_importance_nlp.core import (
    ConstantGradientFixture,
    EqualSufficientStatistics,
    EstimatorResult,
    ParameterRegistry,
    PathSpec,
    QuadraticLossFixture,
    TensorMap,
    WeightedSufficientStatistics,
    ZeroMeanNoiseFixture,
    central_difference_gradient,
    compare_tensor_maps_fp64,
    cross_u_importance,
    double_sample_importance,
    equal_u_importance,
    fp64_apply_group_learning_rates,
    fp64_cross_u_oracle,
    fp64_double_sample_oracle,
    fp64_equal_u_oracle,
    fp64_mean_gradient_oracle,
    fp64_raw_oracle,
    fp64_weighted_u_oracle,
    gauss_legendre_rule,
    integrate_path,
    left_rule,
    midpoint_rule,
    raw_importance,
    right_rule,
    simpson_rule,
    trapezoid_rule,
    weighted_u_importance,
)


def _sample(p: tuple[float, ...], q: float, *, dtype: torch.dtype) -> TensorMap:
    return TensorMap(
        {
            "p": torch.tensor(p, dtype=dtype),
            "q": torch.tensor([[q]], dtype=dtype),
        }
    )


def test_independent_fp64_oracles_cover_raw_double_u_cross_u_and_permutations() -> None:
    samples = [
        _sample((1.0, 2.0), -1.0, dtype=torch.float32),
        _sample((3.0, -2.0), 4.0, dtype=torch.float32),
        _sample((-1.0, 4.0), 2.0, dtype=torch.float32),
    ]
    mean_oracle = fp64_mean_gradient_oracle(samples)
    assert all(value.dtype == torch.float64 and value.device.type == "cpu" for value in mean_oracle.values())
    raw_oracle = fp64_raw_oracle(mean_oracle)
    raw_actual = raw_importance(
        EqualSufficientStatistics.from_samples(
            samples, accumulation_dtype=torch.float64
        ).mean_gradient
    )
    assert compare_tensor_maps_fp64(
        raw_actual, raw_oracle, natural_scale=16.0
    ).passed

    u_oracle = fp64_equal_u_oracle(samples)
    u_actual = equal_u_importance(
        EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float64)
    )
    comparison = compare_tensor_maps_fp64(u_actual, u_oracle, natural_scale=16.0)
    assert comparison.passed
    assert comparison.comparison_dtype == "torch.float64_cpu"

    # 显式 oracle 与流式实现都必须对 microbatch 排列严格不敏感。
    for permutation in itertools.permutations(samples):
        permuted_oracle = fp64_equal_u_oracle(permutation)
        permuted_actual = equal_u_importance(
            EqualSufficientStatistics.from_samples(
                permutation, accumulation_dtype=torch.float64
            )
        )
        assert compare_tensor_maps_fp64(
            permuted_oracle, u_oracle, natural_scale=16.0
        ).passed
        assert compare_tensor_maps_fp64(
            permuted_actual, u_oracle, natural_scale=16.0
        ).passed

    x = samples
    y = [
        _sample((2.0, 1.0), 3.0, dtype=torch.float32),
        _sample((4.0, 3.0), -1.0, dtype=torch.float32),
        _sample((6.0, 5.0), 2.0, dtype=torch.float32),
    ]
    cross_oracle = fp64_cross_u_oracle(
        x,
        y,
        x_weights=(1.0, 2.0, 4.0),
        y_weights=(3.0, 1.0, 2.0),
    )
    cross_actual = cross_u_importance(
        x,
        y,
        x_weights=(1.0, 2.0, 4.0),
        y_weights=(3.0, 1.0, 2.0),
    )
    assert compare_tensor_maps_fp64(
        cross_actual,
        cross_oracle,
        natural_scale=30.0,
        atol=1e-7,
        rtol=1e-5,
        normalized_l2_limit=1e-5,
    ).passed

    mean_x = fp64_mean_gradient_oracle(x)
    mean_y = fp64_mean_gradient_oracle(y)
    double_oracle = fp64_double_sample_oracle(mean_x, mean_y)
    double_actual = double_sample_importance(mean_x, mean_y)
    assert compare_tensor_maps_fp64(
        double_actual, double_oracle, natural_scale=20.0
    ).passed
    cartesian_cross = fp64_cross_u_oracle(x, y, exclude_matching_pairs=False)
    assert compare_tensor_maps_fp64(
        cartesian_cross, double_oracle, natural_scale=20.0
    ).passed


def test_weighted_ordered_pair_oracle_is_independent_of_fast_formula() -> None:
    samples = [
        _sample((1.0, 2.0), -1.0, dtype=torch.float64),
        _sample((3.0, -2.0), 4.0, dtype=torch.float64),
        _sample((-1.0, 4.0), 2.0, dtype=torch.float64),
    ]
    weights = (1.0, 2.0, 5.0)
    oracle = fp64_weighted_u_oracle(samples, weights)
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
    actual = weighted_u_importance(
        statistics, require_unbiasedness_assumptions=True
    )
    assert compare_tensor_maps_fp64(actual, oracle, natural_scale=20.0).passed


class _TwoGroupModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.second = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32))


def test_group_learning_rate_oracle_and_comparator_always_use_fp64() -> None:
    model = _TwoGroupModel()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.first], "lr": 0.125},
            {"params": [model.second], "lr": 0.5},
        ]
    )
    registry = ParameterRegistry.from_model(model, optimizer)
    core = TensorMap(
        {
            "first": torch.tensor([8.0], dtype=torch.float32),
            "second": torch.tensor([-3.0], dtype=torch.float32),
        },
        registry=registry,
    )
    rates = {"group_0000": 0.125, "group_0001": 0.5}
    actual = EstimatorResult.from_core(
        "local_gradient_space_importance_raw", core, rates
    ).score
    oracle = fp64_apply_group_learning_rates(core, rates)
    assert actual["first"].dtype == torch.float32
    assert oracle["first"].dtype == torch.float64
    comparison = compare_tensor_maps_fp64(actual, oracle, natural_scale=2.0)
    assert comparison.passed
    assert comparison.comparison_dtype == "torch.float64_cpu"
    torch.testing.assert_close(oracle["first"], torch.tensor([1.0], dtype=torch.float64))
    torch.testing.assert_close(oracle["second"], torch.tensor([-1.5], dtype=torch.float64))


def test_quadratic_closed_form_autograd_and_central_difference_agree() -> None:
    diagonal = TensorMap(
        {
            "weight": torch.tensor([[2.0, -1.0]], dtype=torch.float64),
            "bias": torch.tensor([3.0], dtype=torch.float64),
        }
    )
    linear = TensorMap(
        {
            "weight": torch.tensor([[0.5, 2.0]], dtype=torch.float64),
            "bias": torch.tensor([-4.0], dtype=torch.float64),
        }
    )
    fixture = QuadraticLossFixture(diagonal, linear, constant=1.25)
    leaves = {
        "weight": torch.tensor([[1.5, -2.0]], dtype=torch.float64, requires_grad=True),
        "bias": torch.tensor([0.25], dtype=torch.float64, requires_grad=True),
    }
    point = TensorMap(leaves)
    analytic = fixture.gradient_at(point)
    finite_difference = central_difference_gradient(
        fixture.loss,
        point,
        step={
            "weight": torch.full((1, 2), 1e-4, dtype=torch.float64),
            "bias": 1e-4,
        },
    )
    fixture.loss(point).backward()
    autograd = TensorMap(
        {name: tensor.grad.detach().clone() for name, tensor in leaves.items()}
    )
    finite_comparison = compare_tensor_maps_fp64(
        finite_difference,
        analytic,
        natural_scale=8.0,
        atol=1e-11,
        rtol=1e-9,
        normalized_l2_limit=1e-9,
    )
    assert finite_comparison.passed
    assert compare_tensor_maps_fp64(
        autograd, analytic, natural_scale=8.0
    ).passed
    # 有限差分只操作独立副本，原始叶子值不能被扰动残留。
    torch.testing.assert_close(
        leaves["weight"].detach(),
        torch.tensor([[1.5, -2.0]], dtype=torch.float64),
    )


def test_constant_and_quadratic_path_fixtures_have_coordinate_truth() -> None:
    pre = TensorMap({"x": torch.tensor([1.0, -2.0], dtype=torch.float64)})
    post = TensorMap({"x": torch.tensor([3.0, 1.0], dtype=torch.float64)})
    linear = ConstantGradientFixture(
        TensorMap({"x": torch.tensor([2.0, -4.0], dtype=torch.float64)}),
        constant=0.75,
    )
    linear_path = PathSpec(pre, post, loss_id="linear-fixture")
    expected_linear = linear.path_contribution(pre, post)
    rules = (
        left_rule(),
        right_rule(),
        midpoint_rule(),
        trapezoid_rule(),
        simpson_rule(),
        gauss_legendre_rule(4),
    )
    for rule in rules:
        result = integrate_path(
            linear_path,
            rule,
            lambda _alpha, state: linear.gradient_at(state),
            loss_fn=linear.loss,
        )
        assert compare_tensor_maps_fp64(
            result.signed, expected_linear, natural_scale=12.0
        ).passed
        assert result.completeness_absolute_residual <= 2e-15

    quadratic = QuadraticLossFixture(
        TensorMap({"x": torch.tensor([2.0, 3.0], dtype=torch.float64)}),
        TensorMap({"x": torch.tensor([-1.0, 0.5], dtype=torch.float64)}),
        constant=-2.0,
    )
    quadratic_path = PathSpec(pre, post, loss_id="quadratic-fixture")
    expected_quadratic = quadratic.path_contribution(pre, post)
    result = integrate_path(
        quadratic_path,
        trapezoid_rule(),
        lambda _alpha, state: quadratic.gradient_at(state),
        loss_fn=quadratic.loss,
    )
    assert compare_tensor_maps_fp64(
        result.signed, expected_quadratic, natural_scale=30.0
    ).passed
    assert math.isclose(
        result.signed.scalar_sum(dtype=torch.float64).item(),
        quadratic.loss(pre).item() - quadratic.loss(post).item(),
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert result.completeness_relative_residual is not None
    assert result.completeness_relative_residual <= 1e-12


def test_zero_mean_gaussian_noise_matches_preregistered_raw_and_u_standard_errors() -> None:
    fixture = ZeroMeanNoiseFixture(
        seed=20260722,
        sigma=2.0,
        microbatch_size=8,
        microbatch_count=4,
        repetitions=64,
        coordinate_shapes={"p": (1,)},
    )
    raw_values: list[float] = []
    u_values: list[float] = []
    for samples in fixture.repetitions_iter():
        statistics = EqualSufficientStatistics.from_samples(
            samples, accumulation_dtype=torch.float64
        )
        raw_values.append(float(raw_importance(statistics.mean_gradient)["p"].item()))
        u_values.append(float(equal_u_importance(statistics)["p"].item()))

    raw_mean = sum(raw_values) / len(raw_values)
    u_mean = sum(u_values) / len(u_values)
    assert raw_mean > 0
    assert abs(raw_mean - fixture.raw_expectation) <= 5 * fixture.raw_mean_standard_error
    assert abs(u_mean) <= 5 * fixture.u_mean_standard_error

    # 相同 seed 的生成结果必须可重放，且不能依赖全局 torch RNG 状态。
    replay = ZeroMeanNoiseFixture(
        seed=fixture.seed,
        sigma=fixture.sigma,
        microbatch_size=fixture.microbatch_size,
        microbatch_count=fixture.microbatch_count,
        repetitions=fixture.repetitions,
        coordinate_shapes=fixture.coordinate_shapes,
    )
    original_first = next(fixture.repetitions_iter())
    replay_first = next(replay.repetitions_iter())
    for original, reproduced in zip(original_first, replay_first, strict=True):
        torch.testing.assert_close(original["p"], reproduced["p"])


def test_fp64_comparator_uses_absolute_branch_for_zero_oracle() -> None:
    oracle = TensorMap({"p": torch.zeros(2, dtype=torch.float64)})
    within = TensorMap({"p": torch.tensor([5e-13, -5e-13], dtype=torch.float32)})
    outside = TensorMap({"p": torch.tensor([2e-12, 0.0], dtype=torch.float32)})
    accepted = compare_tensor_maps_fp64(within, oracle, natural_scale=1.0)
    rejected = compare_tensor_maps_fp64(outside, oracle, natural_scale=1.0)
    assert accepted.passed and accepted.branch == "near_zero_absolute"
    assert accepted.normalized_l2_error is None
    assert not rejected.passed and rejected.max_absolute_error > rejected.absolute_threshold

    nonfinite = TensorMap(
        {"p": torch.tensor([float("nan"), 0.0], dtype=torch.float32)},
        require_finite=False,
    )
    invalid = compare_tensor_maps_fp64(nonfinite, oracle, natural_scale=1.0)
    assert not invalid.passed
    assert invalid.nonfinite_count == 1
    assert invalid.worst_parameter == "p"
    assert invalid.to_dict()["comparison_dtype"] == "torch.float64_cpu"
    assert invalid.to_dict()["max_absolute_error"] is None
