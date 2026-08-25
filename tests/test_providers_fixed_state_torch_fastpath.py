"""Focused tests for bounded fixed-state gradient execution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import param_importance_nlp.providers.fixed_state_torch as fixed_state_torch
from param_importance_nlp.providers.fixed_state_torch import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
)
from param_importance_nlp.providers.training import TorchModelAdapter, TrainingMicrobatch


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        values = input_ids.to(dtype=self.weight.dtype).sum(dim=1) * self.weight
        return {"logits": torch.stack((values, -values), dim=-1)}


def _resolver() -> InMemoryFrozenSampleResolver:
    def batch(batch_id: str, value: int, label: int) -> TrainingMicrobatch:
        return TrainingMicrobatch(
            batch_id,
            {
                "input_ids": torch.tensor([[value, value + 1]], dtype=torch.int64),
                "labels": torch.tensor([label], dtype=torch.int64),
            },
            (batch_id,),
        )

    return InMemoryFrozenSampleResolver(
        {"a": batch("a", 1, 0), "b": batch("b", 2, 1), "c": batch("c", 3, 0)},
        resolver_id="fastpath-fixture-v1",
        loss_unit="sample",
        statistical_unit="sample",
        weight_unit="samples",
        sampling_design="uniform_with_replacement_over_frozen_rows",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _provider(module: torch.nn.Module, *, chunk_size: int) -> TorchFixedStateGradientProvider:
    return TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="sequence_classification"),
        _resolver(),
        fixed_state_id=f"fastpath-{chunk_size}",
        output_dtype=torch.float64,
        gradient_chunk_size=chunk_size,
    )


def test_chunked_gradient_matches_strict_per_sample_reference() -> None:
    torch.manual_seed(20260825)
    reference_module = _TinyClassifier()
    chunked_module = _TinyClassifier()
    chunked_module.load_state_dict(reference_module.state_dict())
    reference = _provider(reference_module, chunk_size=1)
    chunked = _provider(chunked_module, chunk_size=3)
    draws = [
        SimpleNamespace(sample_id="a"),
        SimpleNamespace(sample_id="b"),
        SimpleNamespace(sample_id="a"),
        SimpleNamespace(sample_id="c"),
        SimpleNamespace(sample_id="b"),
    ]

    expected = reference.gradient(draws)
    observed = chunked.gradient(draws)

    assert observed.statistical_weight == expected.statistical_weight
    assert observed.sample_ids == expected.sample_ids
    assert observed.loss == pytest.approx(expected.loss, rel=0.0, abs=1e-12)
    torch.testing.assert_close(
        observed.gradients["weight"], expected.gradients["weight"], rtol=1e-10, atol=1e-12
    )


def test_chunk_size_bounds_autograd_calls_and_assert_unchanged_uses_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider(_TinyClassifier(), chunk_size=2)
    before = provider.state_digest()
    original_grad = torch.autograd.grad
    calls: list[object] = []

    def counted_grad(*args: object, **kwargs: object):
        calls.append(args[0])
        return original_grad(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    provider.gradient(
        [
            SimpleNamespace(sample_id="a"),
            SimpleNamespace(sample_id="b"),
            SimpleNamespace(sample_id="c"),
            SimpleNamespace(sample_id="a"),
            SimpleNamespace(sample_id="b"),
        ]
    )
    assert len(calls) == 3

    def unexpected_full_digest() -> str:
        raise AssertionError("cached unchanged check unexpectedly serialized model")

    monkeypatch.setattr(provider, "state_digest", unexpected_full_digest)
    provider.assert_unchanged(before)


def test_gradient_reuses_valid_resident_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _provider(_TinyClassifier(), chunk_size=2)
    provider.state_digest()

    def unexpected_snapshot(*args: object, **kwargs: object):
        raise AssertionError("gradient unexpectedly cloned a resident baseline")

    monkeypatch.setattr(fixed_state_torch, "_take_snapshot", unexpected_snapshot)
    result = provider.gradient(
        [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]
    )
    assert torch.isfinite(result.gradients["weight"]).all()
