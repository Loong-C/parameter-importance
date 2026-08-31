"""Torch 固定状态梯度 provider 的 CPU 合同测试。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from param_importance_nlp.core.registry import ParameterRegistry
from param_importance_nlp.providers.fixed_state_torch import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
)
from param_importance_nlp.providers.protocols import FixedStateGradientProvider
from param_importance_nlp.providers.training import TorchModelAdapter, TrainingMicrobatch


class _ScaleLM(torch.nn.Module):
    def __init__(self, *, dropout: float = 0.0, mutate_buffer: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25, dtype=torch.float64))
        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer("forward_counter", torch.zeros((), dtype=torch.int64))
        self.mutate_buffer = mutate_buffer

    def forward(self, input_ids, attention_mask=None):
        if self.mutate_buffer:
            with torch.no_grad():
                self.forward_counter.add_(1)
        values = self.dropout(input_ids.to(dtype=self.weight.dtype))
        positive = self.weight * values
        return {"logits": torch.stack((positive, -positive), dim=-1)}


def _batch(batch_id: str, labels: list[int]) -> TrainingMicrobatch:
    return TrainingMicrobatch(
        batch_id,
        {
            "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            "labels": torch.tensor([labels], dtype=torch.int64),
        },
        (batch_id,),
    )


def _resolver(*, loss_unit: str = "target_token") -> InMemoryFrozenSampleResolver:
    # causal shift 后，a 只有 1 个有效 target，b 有 3 个。
    return InMemoryFrozenSampleResolver(
        {
            "a": _batch("sample-a", [0, 0, -100, -100]),
            "b": _batch("sample-b", [0, 1, 0, 1]),
        },
        resolver_id="tiny-pile-resolver-v1",
        loss_unit=loss_unit,
        statistical_unit="pretokenized_sequence_draw_group_mean",
        weight_unit="effective_target_tokens",
        sampling_design="uniform_with_replacement_over_frozen_rows",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def test_weighted_draw_mean_matches_explicit_autograd_and_preserves_state() -> None:
    module = _ScaleLM()
    adapter = TorchModelAdapter(module, task_type="causal_lm")
    resolver = _resolver()
    module.weight.grad = torch.tensor(7.0, dtype=module.weight.dtype)
    provider = TorchFixedStateGradientProvider(
        adapter,
        resolver,
        fixed_state_id="tiny-checkpoint-v1",
        output_dtype=torch.float32,
    )
    assert isinstance(provider, FixedStateGradientProvider)
    before = provider.state_digest()
    existing_grad = module.weight.grad.clone()

    # sample b 有放回碰撞两次；统计权重应为 1 + 3 + 3 = 7。
    draws = [
        SimpleNamespace(sample_id="a", draw_id="draw-a"),
        SimpleNamespace(sample_id="b", draw_id="draw-b-1"),
        SimpleNamespace(sample_id="b", draw_id="draw-b-2"),
    ]
    result = provider.gradient(draws)

    losses = [
        adapter.loss(resolver.resolve("a")),
        adapter.loss(resolver.resolve("b")),
        adapter.loss(resolver.resolve("b")),
    ]
    total_count = sum(loss.effective_count for loss in losses)
    objective = sum((loss.loss_numerator for loss in losses[1:]), losses[0].loss_numerator)
    expected = torch.autograd.grad(objective / total_count, module.weight)[0]

    assert result.statistical_weight == 7.0
    assert result.sample_ids == ("a", "b", "b")
    assert result.weight_unit == "effective_target_tokens"
    assert result.gradients["weight"].dtype == torch.float32
    torch.testing.assert_close(
        result.gradients["weight"], expected.to(dtype=torch.float32)
    )
    assert provider.state_digest() == before
    torch.testing.assert_close(module.weight.grad, existing_grad)
    assert module.training is True
    assert module.forward_counter.item() == 0


def test_parameter_registry_identity_and_resolver_defensive_copy() -> None:
    module = _ScaleLM()
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    registry = ParameterRegistry.from_model(module, optimizer)
    resolver = _resolver()
    resolver_digest = resolver.state_digest()
    leaked = resolver.resolve("a")
    leaked.payload["input_ids"].fill_(99)
    assert resolver.state_digest() == resolver_digest

    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="causal_lm"),
        resolver,
        fixed_state_id="registry-bound-state",
        registry=registry,
    )
    assert provider.registry_hash == registry.coordinate_registry_hash
    assert provider.parameter_names == registry.eligible_names


def test_rng_is_replayed_and_stochastic_forward_is_repeatable() -> None:
    torch.manual_seed(1234)
    module = _ScaleLM(dropout=0.5)
    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="causal_lm"),
        _resolver(),
        fixed_state_id="dropout-fixed-state",
    )
    before = provider.state_digest()
    first = provider.gradient([SimpleNamespace(sample_id="b")])
    second = provider.gradient([SimpleNamespace(sample_id="b")])
    torch.testing.assert_close(first.gradients["weight"], second.gradients["weight"])
    assert first.loss == second.loss
    assert provider.state_digest() == before


def test_illegal_buffer_mutation_is_restored_then_rejected() -> None:
    module = _ScaleLM(mutate_buffer=True)
    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="causal_lm"),
        _resolver(),
        fixed_state_id="mutating-model-state",
    )
    before = provider.state_digest()
    with pytest.raises(RuntimeError, match="MODEL_BUFFER_MUTATED"):
        provider.gradient([SimpleNamespace(sample_id="a")])
    assert module.forward_counter.item() == 0
    assert provider.state_digest() == before


def test_loss_unit_mismatch_and_external_state_drift_fail_closed() -> None:
    module = _ScaleLM()
    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="causal_lm"),
        _resolver(loss_unit="sample"),
        fixed_state_id="wrong-loss-unit",
    )
    before = provider.state_digest()
    with pytest.raises(ValueError, match="RESOLVER_LOSS_UNIT_MISMATCH"):
        provider.gradient([SimpleNamespace(sample_id="a")])
    assert provider.state_digest() == before

    with torch.no_grad():
        module.weight.add_(1.0)
    with pytest.raises(RuntimeError, match="STATE_CHANGED"):
        provider.assert_unchanged(before)
