"""Tests for the opt-in, strictly guarded Pythia draw-group fast path."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from param_importance_nlp.experiments.preregistration import (
    build_stage2_preregistration,
)
from param_importance_nlp.experiments import stage23_task_runners
from param_importance_nlp.providers.fixed_state_torch import (
    InMemoryFrozenSampleResolver,
    TorchFixedStateGradientProvider,
)
from param_importance_nlp.providers.training import TorchModelAdapter, TrainingMicrobatch


class _TinyPythia(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(17, 5)
        self.projection = torch.nn.Linear(5, 17, bias=False)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        del attention_mask
        return {"logits": self.projection(self.embedding(input_ids))}


class _FormalResolver(InMemoryFrozenSampleResolver):
    pass


# The fast path uses exact type identity rather than duck typing.  Giving this
# test double the production identity exercises the guard without importing or
# constructing a real mmap file in CPU unit tests.
_FormalResolver.__module__ = "param_importance_nlp.providers.pythia_mmap"
_FormalResolver.__name__ = "PythiaMMapFrozenSampleResolver"


def _resolver(
    *, count: int = 5, formal_identity: bool = True
) -> InMemoryFrozenSampleResolver:
    samples: dict[str, TrainingMicrobatch] = {}
    for index in range(count):
        sample_id = f"sample-{index}"
        target = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        mask = torch.ones((1, 4), dtype=torch.int64)
        if index == 1:
            target[0, 3] = -100
        if index == 2:
            mask[0, 2] = 0
        samples[sample_id] = TrainingMicrobatch(
            f"batch-{index}",
            {
                "input_ids": torch.tensor(
                    [[index % 16 + 1, 2, 3, 4]], dtype=torch.int64
                ),
                "target_ids": target,
                "attention_mask": mask,
            },
            (sample_id,),
            {
                "schema_version": "pythia-mmap-frozen-sample-metadata-v1",
                "global_record_index": index,
            },
        )
    klass = _FormalResolver if formal_identity else InMemoryFrozenSampleResolver
    return klass(
        samples,
        resolver_id="formal-batched-test-v1",
        loss_unit="target_token",
        statistical_unit="target_token",
        weight_unit="effective_target_tokens",
        sampling_design="with_replacement_over_frozen_records",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _provider(
    module: torch.nn.Module,
    resolver: InMemoryFrozenSampleResolver,
    *,
    batched: bool,
    chunk_size: int = 1,
    formal_chunk_size: int = 4,
) -> TorchFixedStateGradientProvider:
    provider = TorchFixedStateGradientProvider(
        TorchModelAdapter(module, task_type="causal_lm"),
        resolver,
        fixed_state_id=f"formal-{batched}-{chunk_size}",
        output_dtype=torch.float32,
        gradient_chunk_size=chunk_size,
        enable_formal_batched=batched,
        formal_batch_chunk_size=formal_chunk_size,
    )
    # Short CPU fixtures retain the production guard's logic while avoiding a
    # 32*2048-token test tensor.  The formal runtime never changes this value.
    provider._FORMAL_SEQUENCE_LENGTH = 4
    return provider


def _draws(count: int) -> list[SimpleNamespace]:
    return [SimpleNamespace(sample_id=f"sample-{index}") for index in range(count)]


def test_formal_batched_matches_sequential_with_duplicates_and_effective_counts() -> None:
    torch.manual_seed(20260825)
    sequential_module = _TinyPythia()
    batched_module = _TinyPythia()
    batched_module.load_state_dict(sequential_module.state_dict())
    draws = [
        SimpleNamespace(sample_id="sample-0"),
        SimpleNamespace(sample_id="sample-1"),
        SimpleNamespace(sample_id="sample-0"),
        SimpleNamespace(sample_id="sample-2"),
    ]
    sequential = _provider(sequential_module, _resolver(), batched=False)
    batched = _provider(batched_module, _resolver(), batched=True)

    expected = sequential.gradient(draws)
    observed = batched.gradient(draws)
    assert observed.sample_ids == expected.sample_ids
    assert observed.statistical_weight == expected.statistical_weight
    assert observed.loss == pytest.approx(expected.loss, rel=2e-6, abs=2e-7)
    for name in expected.gradients:
        torch.testing.assert_close(
            observed.gradients[name], expected.gradients[name], rtol=2e-5, atol=2e-6
        )
    assert batched.state_digest() == batched.state_digest()


def test_formal_flag_and_chunk_size_are_digest_bound() -> None:
    module = _TinyPythia()
    first = _provider(module, _resolver(), batched=False, chunk_size=1)
    second = _provider(module, _resolver(), batched=False, chunk_size=2)
    formal = _provider(module, _resolver(), batched=True, chunk_size=1)
    formal_chunked = _provider(
        module, _resolver(), batched=True, chunk_size=1, formal_chunk_size=2
    )
    assert first.state_digest() != second.state_digest()
    assert first.state_digest() != formal.state_digest()
    assert formal.state_digest() != formal_chunked.state_digest()


def test_formal_guard_falls_back_for_generic_resolver_and_training_model() -> None:
    torch.manual_seed(3)
    module = _TinyPythia()
    module.train()
    provider = _provider(module, _resolver(), batched=True)
    result = provider.gradient(_draws(3))
    assert result.statistical_weight > 0
    assert module.training


def test_formal_batch_is_capped_at_public_block_size(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _TinyPythia()
    # A generic resolver with 33 rows cannot enter the formal path.  The
    # bounded generic chunk still limits each autograd call to eight rows.
    resolver = _resolver(count=33, formal_identity=True)
    provider = _provider(module, resolver, batched=True, chunk_size=8)
    calls: list[object] = []
    original_grad = torch.autograd.grad

    def counted_grad(*args: object, **kwargs: object):
        calls.append(args[0])
        return original_grad(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(torch.autograd, "grad", counted_grad)
    provider.gradient(_draws(33))
    assert len(calls) == 5  # 33 draws exceed the 32-draw formal cap
    assert provider._FORMAL_BATCH_MAX_DRAWS == 32


def test_preregistration_and_formal_runner_bind_fp32_main_gradients() -> None:
    registration = build_stage2_preregistration(
        seed_plan_hash="0" * 64,
        producer_commit="141c9799fa1888eb7b487a3b3c76b8822ae67acf",
        scope="formal",
    )
    estimand = registration["estimand"]
    assert estimand["gradient_dtype"] == "float32"
    assert estimand["reference_accumulation_dtype"] == "float64"
    source = inspect.getsource(stage23_task_runners._formal_provider)
    assert "torch_dtype=torch.float32" in source
    assert "output_dtype=torch.float32" in source
    assert "enable_formal_batched=True" in source
