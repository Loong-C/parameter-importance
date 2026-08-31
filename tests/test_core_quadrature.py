from __future__ import annotations

import math

import torch

from param_importance_nlp.core import (
    CoreContractError,
    PathSpec,
    TensorMap,
    composite_left_rule,
    composite_midpoint_rule,
    composite_simpson_rule,
    composite_trapezoid_rule,
    gauss_legendre_rule,
    integrate_path,
    integrate_scalar_function,
    left_rule,
    midpoint_rule,
    simpson_rule,
    trapezoid_rule,
)


def test_rule_nodes_weights_and_polynomial_exactness() -> None:
    rules = [left_rule(), midpoint_rule(), trapezoid_rule(), simpson_rule(), gauss_legendre_rule(4)]
    for rule in rules:
        assert bool(((rule.nodes >= 0) & (rule.nodes <= 1)).all())
        torch.testing.assert_close(rule.weights.sum(), torch.tensor(1.0, dtype=torch.float64))
        assert rule.node_count == len(rule.weights)

    # Simpson 对三次多项式精确，4 点 Gauss-Legendre 对七次多项式精确。
    assert abs(integrate_scalar_function(lambda x: x**3, simpson_rule()) - 0.25) < 1e-15
    assert abs(integrate_scalar_function(lambda x: x**7, gauss_legendre_rule(4)) - 0.125) < 1e-14


def test_composite_rules_and_simpson_odd_subinterval_guard() -> None:
    target = 1.0 / 6.0  # integral x**5 on [0,1]
    left_errors = [
        abs(integrate_scalar_function(lambda x: x**5, composite_left_rule(n)) - target)
        for n in (4, 8, 16)
    ]
    midpoint_errors = [
        abs(integrate_scalar_function(lambda x: x**5, composite_midpoint_rule(n)) - target)
        for n in (4, 8, 16)
    ]
    trapezoid_errors = [
        abs(integrate_scalar_function(lambda x: x**5, composite_trapezoid_rule(n)) - target)
        for n in (4, 8, 16)
    ]
    simpson_errors = [
        abs(integrate_scalar_function(lambda x: x**5, composite_simpson_rule(n)) - target)
        for n in (4, 8, 16)
    ]
    assert left_errors[0] / left_errors[1] > 1.7
    assert midpoint_errors[0] / midpoint_errors[1] > 3.5
    assert trapezoid_errors[0] / trapezoid_errors[1] > 3.5
    assert simpson_errors[0] / simpson_errors[1] > 14

    try:
        composite_simpson_rule(3)
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("奇数子区间 Simpson 必须失败")


def test_quadratic_path_trapezoid_is_exact_and_inputs_are_unchanged() -> None:
    pre = TensorMap({"x": torch.tensor([0.0, -1.0], dtype=torch.float64)})
    post = TensorMap({"x": torch.tensor([2.0, 3.0], dtype=torch.float64)})
    pre_copy = pre.clone()
    post_copy = post.clone()
    path = PathSpec(pre, post, probe_id="analytic", loss_id="half_squared_norm")

    def gradient(alpha: float, state: TensorMap) -> TensorMap:
        del alpha
        return TensorMap({"x": state["x"].clone()})

    def loss(state: TensorMap) -> torch.Tensor:
        return 0.5 * state["x"].square().sum()

    result = integrate_path(path, trapezoid_rule(), gradient, loss_fn=loss)
    # 平均梯度为 (pre+post)/2=[1,1]，delta=[2,4]，贡献为 [-2,-4]。
    torch.testing.assert_close(result.average_gradient["x"], torch.tensor([1.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(result.signed["x"], torch.tensor([-2.0, -4.0], dtype=torch.float64))
    assert result.loss_drop == -6.0
    assert result.completeness_absolute_residual == 0.0
    assert result.unique_gradient_evaluations == 2
    assert result.ranked_coordinates("absolute")[0] == ("x#000000000001", 4.0)
    summary = result.group_summary("absolute", tag="layer")
    assert summary["x"]["sum"] == 6.0
    assert summary["x"]["mean_per_parameter"] == 3.0
    assert summary["x"]["mass_fraction"] == 1.0
    torch.testing.assert_close(pre["x"], pre_copy["x"])
    torch.testing.assert_close(post["x"], post_copy["x"])


def test_constant_gradient_all_rules_agree_and_cancellation_is_visible() -> None:
    path = PathSpec(
        TensorMap({"p": torch.tensor([0.0, 0.0], dtype=torch.float64)}),
        TensorMap({"p": torch.tensor([2.0, -2.0], dtype=torch.float64)}),
        probe_id="linear",
        loss_id="linear",
    )

    def gradient(alpha: float, state: TensorMap) -> TensorMap:
        del alpha, state
        return TensorMap({"p": torch.tensor([3.0, 3.0], dtype=torch.float64)})

    def loss(state: TensorMap) -> torch.Tensor:
        return 3.0 * state["p"].sum()

    expected = torch.tensor([-6.0, 6.0], dtype=torch.float64)
    for rule in (left_rule(), midpoint_rule(), trapezoid_rule(), simpson_rule(), gauss_legendre_rule(3)):
        result = integrate_path(path, rule, gradient, loss_fn=loss)
        torch.testing.assert_close(result.signed["p"], expected)
        assert result.completeness_absolute_residual == 0.0
        assert math.isclose(result.absolute.scalar_sum().item(), 12.0, abs_tol=1e-12)


def test_path_rejects_extrapolation_and_nonfinite_gradient() -> None:
    path = PathSpec(TensorMap({"x": torch.tensor([0.0])}), TensorMap({"x": torch.tensor([1.0])}))
    for alpha in (-0.1, 1.1, float("nan")):
        try:
            path.interpolate(alpha)
        except CoreContractError:
            pass
        else:  # pragma: no cover
            raise AssertionError("越界或非有限 alpha 必须失败")

    def bad_gradient(alpha: float, state: TensorMap) -> TensorMap:
        del alpha, state
        return TensorMap({"x": torch.tensor([float("inf")])}, require_finite=False)

    try:
        integrate_path(path, left_rule(), bad_gradient)
    except CoreContractError:
        pass
    else:  # pragma: no cover
        raise AssertionError("非有限路径梯度必须失败")


def test_path_identity_binds_endpoint_values() -> None:
    first = PathSpec(
        TensorMap({"x": torch.tensor([0.0])}),
        TensorMap({"x": torch.tensor([1.0])}),
    )
    second = PathSpec(
        TensorMap({"x": torch.tensor([0.0])}),
        TensorMap({"x": torch.tensor([2.0])}),
    )
    assert first.identity_hash != second.identity_hash
