from __future__ import annotations

import math

import pytest
import torch

from param_importance_nlp.runtime import GradientAttempt, GradientPhase


def test_scaled_gradients_must_unscale_before_finite_check_and_clip() -> None:
    original = {
        "left": torch.tensor([6.0, 8.0], dtype=torch.float32),
        "right": torch.tensor([0.0], dtype=torch.float32),
    }
    captured = GradientAttempt.capture(original, gradient_scale=2.0, scaled=True)
    original["left"].zero_()

    with pytest.raises(ValueError, match="FINITE_CHECK_FROM_INVALID_PHASE"):
        captured.check_finite()
    unscaled = captured.unscale()
    with pytest.raises(ValueError, match="UNSCALE_FROM_INVALID_PHASE"):
        unscaled.unscale()
    finite = unscaled.check_finite()

    assert finite.phase is GradientPhase.FINITE
    assert finite.global_norm == pytest.approx(5.0)
    assert torch.equal(finite.gradients["left"], torch.tensor([3.0, 4.0]))

    clipped = finite.clip(2.5)
    assert clipped.phase is GradientPhase.CLIPPED
    assert clipped.clip_factor == pytest.approx(0.5)
    assert torch.equal(clipped.gradients["left"], torch.tensor([1.5, 2.0]))
    assert clipped.global_norm == pytest.approx(5.0)


def test_nonfinite_unscaled_gradient_enters_skip_and_cannot_be_installed() -> None:
    attempt = GradientAttempt.capture(
        {"weight": torch.tensor([math.inf]), "bias": None},
        scaled=False,
    ).check_finite()

    assert attempt.phase is GradientPhase.SKIPPED
    assert attempt.skip_reason == "NONFINITE_UNSCALED_GRADIENT"
    with pytest.raises(ValueError, match="CLIP_FROM_INVALID_PHASE"):
        attempt.clip(1.0)
    with pytest.raises(ValueError, match="INSTALL_FROM_INVALID_PHASE"):
        attempt.install(
            {
                "weight": torch.nn.Parameter(torch.zeros(1)),
                "bias": torch.nn.Parameter(torch.zeros(1)),
            }
        )


def test_install_is_validated_before_gradients_are_mutated() -> None:
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    bias = torch.nn.Parameter(torch.tensor([3.0]))
    weight.grad = torch.tensor([9.0, 9.0])
    bias.grad = torch.tensor([8.0])
    ready = GradientAttempt.capture(
        {"weight": torch.tensor([2.0, 4.0]), "bias": None},
        scaled=False,
    ).check_finite()

    with pytest.raises(ValueError, match="SHAPE_MISMATCH"):
        ready.install(
            {
                "weight": torch.nn.Parameter(torch.zeros(3)),
                "bias": bias,
            }
        )
    # 验证失败发生在任何 copy 之前。
    assert torch.equal(bias.grad, torch.tensor([8.0]))

    ready.install({"weight": weight, "bias": bias})
    assert torch.equal(weight.grad, torch.tensor([2.0, 4.0]))
    assert bias.grad is None


def test_sparse_gradient_and_all_none_fail_closed() -> None:
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]), torch.tensor([1.0]), size=(2,)
    )
    with pytest.raises(TypeError, match="SPARSE_GRADIENT_UNSUPPORTED"):
        GradientAttempt.capture({"weight": sparse})
    with pytest.raises(ValueError, match="ALL_GRADIENTS_ARE_NONE"):
        GradientAttempt.capture({"weight": None})
