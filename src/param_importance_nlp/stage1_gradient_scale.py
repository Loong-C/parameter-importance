"""Deterministic CPU evidence for Stage 1 S1.4 loss/gradient-scale checks.

The module deliberately keeps the two sides of the most important comparison
separate.  Production routes use :func:`core.losses.causal_lm_loss`; the full
batch reference computes masked token negative-log-likelihood directly with
``log_softmax`` and ``gather``.  This prevents a shared loss-reduction helper
from making a broken adapter and its oracle agree by accident.

It is a small, fixed-state fixture only.  It neither opens model/data assets
nor claims that a local run is a formal Gate.  ``formalize_s1_4.py`` publishes
the exact same deterministic calculation only after closing the S1.3 handoff.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch

from .contracts.jsonio import JSONValue, canonical_json_hash
from .core.errors import CoreContractError
from .core.losses import causal_lm_loss, pre_shifted_causal_lm_loss
from .core.oracles import compare_tensor_maps_fp64
from .core.tensors import TensorMap


TASK_ID = "stage1.04_loss_and_gradient_scale"
GATE_ID = "G1-GRAD"
FIXTURE_ID = "stage1-s14-loss-gradient-fixture-v1"
REPORT_SCHEMA = "stage1-g1-grad-report-v1"
TABLE_SCHEMA = "stage1-g1-grad-comparison-table-v1"
GATE_SCHEMA = "stage1-g1-grad-gate-record-v1"

_PROFILES: Mapping[str, Mapping[str, float]] = {
    "T64_ORACLE": {"atol": 1e-12, "rtol": 1e-10, "normalized_l2_limit": 1e-10},
    "T32_SINGLE": {"atol": 1e-7, "rtol": 1e-5, "normalized_l2_limit": 1e-5},
}
_NATURAL_GRADIENT_SCALE = 1.0
_NATURAL_LOSS_SCALE = 1.0
_S1_3_V2_INDEX_SHA256 = "51eb16bf87d73d68f6c1da49b7635fa42bd0456e9305f7263326a794b9b2f2ab"
_FORMAL_UPSTREAM_FIELDS = {
    "s1_3_index_ref",
    "s1_3_index_sha256",
    "s1_3_gate_artifact_hash",
    "s1_3_fixture_manifest_sha256",
    "s1_3_oracle_bundle_sha256",
    "s1_3_oracle_validation_report_sha256",
    "s1_3_replay_sha256",
    "s1_3_validation_sha256",
    "s1_3_frozen_gradient_input_hash",
}
_SOURCE_FILES = (
    "src/param_importance_nlp/core/__init__.py",
    "src/param_importance_nlp/core/losses.py",
    "src/param_importance_nlp/stage1_gradient_scale.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "ops/stage1/formalize_s1_4.py",
    "schemas/stage1/g1-grad-report-v1.json",
    "schemas/stage1/g1-grad-comparison-table-v1.json",
    "schemas/stage1/g1-grad-gate-record-v1.json",
    "schemas/stage1/s1-4-formalization-index-v1.json",
    "schemas/stage1/s1-4-validation-v1.json",
    "tests/test_stage1_s14_loss_gradient_scale.py",
    "tests/test_stage1_s14_handoff_and_charts.py",
)
_PROFILE_COMPARISON_IDS = (
    "adapter_full_gradient",
    "adapter_full_loss",
    "pre_shifted_adapter_full_gradient",
    "pre_shifted_adapter_full_loss",
    "per_sample_reconstruction_gradient",
    "per_sample_reconstruction_loss",
    "per_token_reconstruction_gradient",
    "per_token_reconstruction_loss",
    "equal_microbatch_m2_gradient",
    "equal_microbatch_m2_loss",
    "equal_microbatch_reordered_gradient",
    "equal_microbatch_reordered_loss",
    "equal_microbatch_m4_gradient",
    "equal_microbatch_m4_loss",
    "token_weighted_microbatch_gradient",
    "token_weighted_microbatch_loss",
    "sum_divided_by_effective_count_gradient",
    "sum_divided_by_effective_count_loss",
    "accumulation_m2_gradient",
    "accumulation_m2_local_gradient_reconstruction",
    "accumulation_m2_local_loss_reconstruction",
    "accumulation_m4_gradient",
    "accumulation_m4_local_gradient_reconstruction",
    "accumulation_m4_local_loss_reconstruction",
    "accumulation_weighted_m2_gradient",
    "accumulation_weighted_m2_local_gradient_reconstruction",
    "accumulation_weighted_m2_local_loss_reconstruction",
    "rng_eval_repeat_gradient",
)
_PROFILE_NAMES = tuple(_PROFILES)
_SAMPLE_IDS = ("s14-sample-0", "s14-sample-1", "s14-sample-2", "s14-sample-3")


class Stage1GradientScaleError(RuntimeError):
    """The fixed S1.4 fixture or its serialized evidence is invalid."""


class _TinyCausalModel(torch.nn.Module):
    """A manually initialized causal LM without provider defaults or dropout RNG.

    ``Dropout`` exists solely for the explicitly non-gating RNG smoke.  All
    exact-equivalence routes switch the complete module to ``eval`` first.
    """

    def __init__(self, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(7, 3, dtype=dtype)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.output = torch.nn.Linear(3, 7, bias=True, dtype=dtype)
        with torch.no_grad():
            embedding = torch.arange(21, dtype=dtype).reshape(7, 3)
            projection = torch.arange(21, dtype=dtype).reshape(7, 3)
            bias = torch.arange(7, dtype=dtype)
            self.embedding.weight.copy_((embedding - 10.0) / 17.0)
            self.output.weight.copy_((projection - 9.0) / 19.0)
            self.output.bias.copy_((bias - 3.0) / 23.0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.embedding(input_ids))
        return self.output(hidden)


@dataclass(frozen=True, slots=True)
class _RouteResult:
    loss: float
    numerator: float
    effective_count: int
    gradient: TensorMap


def _fixture_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Four samples, each with exactly two valid shifted target tokens.

    Equal-count partitions can therefore use M=2 and M=4.  The distinct
    sample contents make the deliberately unweighted partition ``(1, 3)``
    versus the correct 2:6 token weighting a stable negative control.
    """

    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 4, 1, 0],
            [2, 1, 5, 3],
            [5, 0, 4, 2],
        ],
        dtype=torch.long,
    )
    labels = input_ids.clone()
    # Target position 1 of sample zero is ignored even though its attention
    # bit is one.  Position 3 is deliberately enabled for that one sample so
    # every sample remains a two-effective-token unit.  This freezes both the
    # equal-M partitions and the unequal 2:6 negative-control partition while
    # proving that ignore_index, rather than attention alone, controls the
    # numerator/count.
    labels[0, 1] = -100
    attention_mask = torch.tensor(
        [[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
        dtype=torch.long,
    )
    return input_ids, labels, attention_mask


def _rng_digest() -> str:
    return hashlib.sha256(bytes(torch.random.get_rng_state().tolist())).hexdigest()


def _tensor_map_from_parameters(model: torch.nn.Module) -> TensorMap:
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in sorted(model.named_parameters()):
        if parameter.grad is None:
            raise Stage1GradientScaleError(f"S1_4_MISSING_GRADIENT:{name}")
        tensors[name] = parameter.grad.detach().clone()
    return TensorMap(tensors)


def _autograd_gradient(loss: torch.Tensor, model: torch.nn.Module) -> TensorMap:
    parameters = tuple(parameter for _, parameter in sorted(model.named_parameters()))
    names = tuple(name for name, _ in sorted(model.named_parameters()))
    gradients = torch.autograd.grad(loss, parameters, allow_unused=False)
    return TensorMap(
        {name: gradient.detach().clone() for name, gradient in zip(names, gradients, strict=True)}
    )


def _weighted_gradient(
    gradients: Sequence[TensorMap],
    counts: Sequence[int],
) -> TensorMap:
    if not gradients or len(gradients) != len(counts):
        raise Stage1GradientScaleError("S1_4_WEIGHTED_GRADIENT_INPUT_INVALID")
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
        raise Stage1GradientScaleError("S1_4_EFFECTIVE_COUNT_INVALID")
    reference = gradients[0]
    for value in gradients[1:]:
        reference.assert_compatible(value)
    total = sum(counts)
    return TensorMap(
        {
            name: sum(
                (gradient[name].to(dtype=torch.float64) * (count / total))
                for gradient, count in zip(gradients, counts, strict=True)
            ).to(dtype=gradients[0][name].dtype)
            for name in reference
        }
    )


def _equal_gradient(gradients: Sequence[TensorMap]) -> TensorMap:
    return _weighted_gradient(gradients, [1] * len(gradients))


def _manual_token_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Independent full-batch NLL oracle; intentionally does not call LossBatch."""

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid = shifted_labels.ne(-100) & attention_mask[:, 1:].to(dtype=torch.bool)
    count = int(valid.sum().item())
    if count <= 0:
        raise Stage1GradientScaleError("S1_4_ORACLE_ZERO_EFFECTIVE_TOKEN")
    log_probabilities = torch.log_softmax(shifted_logits, dim=-1)
    # ``gather`` cannot legally consume ignore_index=-100.  Replace only
    # invalid targets before gathering and remove those positions afterwards;
    # this remains an independent implementation from the production adapter.
    safe_labels = torch.where(valid, shifted_labels, torch.zeros_like(shifted_labels))
    gathered = -log_probabilities.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return gathered.masked_select(valid).sum(), count


def _manual_per_token_routes(model: _TinyCausalModel) -> list[_RouteResult]:
    """Differentiate each valid target-token scalar NLL independently.

    The reconstructed equal-token average is intentionally not expressed via
    a batch reduction helper.  Every token has its own scalar gather and
    autograd call, making this a useful oracle for both numerator semantics
    and gradients in the presence of an in-attention ``ignore_index``.
    """

    inputs, labels, mask = _fixture_tensors()
    logits = model(inputs)
    targets = labels[:, 1:]
    valid = targets.ne(-100) & mask[:, 1:].to(dtype=torch.bool)
    coordinates = valid.nonzero(as_tuple=False).tolist()
    if not coordinates:
        raise Stage1GradientScaleError("S1_4_PER_TOKEN_ZERO_EFFECTIVE_TOKEN")
    parameters = tuple(parameter for _, parameter in sorted(model.named_parameters()))
    names = tuple(name for name, _ in sorted(model.named_parameters()))
    routes: list[_RouteResult] = []
    for offset, (sample, target_position) in enumerate(coordinates):
        token_id = int(targets[sample, target_position].item())
        scalar = -torch.log_softmax(logits[sample, target_position, :], dim=-1)[token_id]
        gradients = torch.autograd.grad(
            scalar,
            parameters,
            allow_unused=False,
            retain_graph=offset != len(coordinates) - 1,
        )
        routes.append(
            _RouteResult(
                loss=float(scalar.detach().to(dtype=torch.float64).item()),
                numerator=float(scalar.detach().to(dtype=torch.float64).item()),
                effective_count=1,
                gradient=TensorMap(
                    {name: gradient.detach().clone() for name, gradient in zip(names, gradients, strict=True)}
                ),
            )
        )
    return routes


def _adapter_route(
    model: _TinyCausalModel,
    indices: Sequence[int],
) -> _RouteResult:
    inputs, labels, mask = _fixture_tensors()
    index = torch.tensor(indices, dtype=torch.long)
    loss = causal_lm_loss(model(inputs[index]), labels[index], mask[index])
    return _RouteResult(
        loss=float(loss.mean_loss.detach().to(dtype=torch.float64).item()),
        numerator=float(loss.loss_numerator.detach().to(dtype=torch.float64).item()),
        effective_count=loss.effective_count,
        gradient=_autograd_gradient(loss.mean_loss, model),
    )


def _pre_shifted_adapter_route(
    model: _TinyCausalModel,
    indices: Sequence[int],
) -> _RouteResult:
    """Exercise the Pythia-aligned adapter without applying a second shift."""

    inputs, labels, mask = _fixture_tensors()
    index = torch.tensor(indices, dtype=torch.long)
    logits = model(inputs[index])
    loss = pre_shifted_causal_lm_loss(
        logits[:, :-1, :],
        labels[index][:, 1:],
        mask[index][:, 1:],
    )
    return _RouteResult(
        loss=float(loss.mean_loss.detach().to(dtype=torch.float64).item()),
        numerator=float(loss.loss_numerator.detach().to(dtype=torch.float64).item()),
        effective_count=loss.effective_count,
        gradient=_autograd_gradient(loss.mean_loss, model),
    )


def _manual_full_route(model: _TinyCausalModel) -> _RouteResult:
    inputs, labels, mask = _fixture_tensors()
    numerator, count = _manual_token_loss(model(inputs), labels, mask)
    mean = numerator / count
    return _RouteResult(
        loss=float(mean.detach().to(dtype=torch.float64).item()),
        numerator=float(numerator.detach().to(dtype=torch.float64).item()),
        effective_count=count,
        gradient=_autograd_gradient(mean, model),
    )


def _adapter_accumulation_route(
    model: _TinyCausalModel,
    groups: Sequence[Sequence[int]],
) -> tuple[TensorMap, list[_RouteResult], bool]:
    """Read local ``.grad`` only after clearing it, then perform true accumulation."""

    local: list[_RouteResult] = []
    for group in groups:
        model.zero_grad(set_to_none=True)
        result = _adapter_route(model, group)
        # ``_adapter_route`` uses autograd.grad, so deliberately exercise the
        # runtime .grad path independently and assert it starts clean.
        inputs, labels, mask = _fixture_tensors()
        index = torch.tensor(group, dtype=torch.long)
        loss = causal_lm_loss(model(inputs[index]), labels[index], mask[index])
        loss.mean_loss.backward()
        grad = _tensor_map_from_parameters(model)
        local.append(
            _RouteResult(result.loss, result.numerator, result.effective_count, grad)
        )

    total = sum(item.effective_count for item in local)
    if total <= 0:
        raise Stage1GradientScaleError("S1_4_ACCUMULATION_ZERO_EFFECTIVE_COUNT")
    model.zero_grad(set_to_none=True)
    for group in groups:
        inputs, labels, mask = _fixture_tensors()
        index = torch.tensor(group, dtype=torch.long)
        loss = causal_lm_loss(model(inputs[index]), labels[index], mask[index])
        # Backward numerator/total, not an unweighted local mean.
        (loss.loss_numerator / total).backward()
    accumulated = _tensor_map_from_parameters(model)

    # A prior step must not leak after the next explicit clear.
    model.zero_grad(set_to_none=True)
    first = causal_lm_loss(model(inputs[:1]), labels[:1], mask[:1])
    first.mean_loss.backward()
    model.zero_grad(set_to_none=True)
    clear_was_complete = all(parameter.grad is None for parameter in model.parameters())
    full = causal_lm_loss(model(inputs), labels, mask)
    full.mean_loss.backward()
    _tensor_map_from_parameters(model)  # finite/missing guards on the next step itself
    return accumulated, local, clear_was_complete


def _map_comparison(
    *,
    comparison_id: str,
    profile: str,
    actual: TensorMap,
    oracle: TensorMap,
    natural_scale: float = _NATURAL_GRADIENT_SCALE,
) -> dict[str, JSONValue]:
    settings = _PROFILES[profile]
    global_result = compare_tensor_maps_fp64(
        actual,
        oracle,
        natural_scale=natural_scale,
        **settings,
    ).to_dict()
    rows: list[dict[str, JSONValue]] = []
    for name in actual:
        result = compare_tensor_maps_fp64(
            TensorMap({name: actual[name]}),
            TensorMap({name: oracle[name]}),
            natural_scale=natural_scale,
            **settings,
        ).to_dict()
        rows.append({"parameter_name": name, **result})
    passed = bool(global_result["passed"]) and all(bool(row["passed"]) for row in rows)
    return {
        "comparison_id": comparison_id,
        "profile": profile,
        "object_id": "mean_gradient",
        "passed": passed,
        "global": global_result,
        "per_tensor": rows,
    }


def _scalar_comparison(
    *,
    comparison_id: str,
    profile: str,
    actual: float,
    oracle: float,
) -> dict[str, JSONValue]:
    result = _map_comparison(
        comparison_id=comparison_id,
        profile=profile,
        actual=TensorMap({"loss": torch.tensor([actual], dtype=torch.float64)}),
        oracle=TensorMap({"loss": torch.tensor([oracle], dtype=torch.float64)}),
        natural_scale=_NATURAL_LOSS_SCALE,
    )
    result["object_id"] = "mean_loss"
    return result


def _scatter_rows(
    profile: str,
    route_id: str,
    actual: TensorMap,
    oracle: TensorMap,
) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for name in actual:
        for coordinate, (candidate, reference) in enumerate(
            zip(
                actual[name].detach().to(dtype=torch.float64).reshape(-1).tolist(),
                oracle[name].detach().to(dtype=torch.float64).reshape(-1).tolist(),
                strict=True,
            )
        ):
            rows.append(
                {
                    "profile": profile,
                    "route_id": route_id,
                    "parameter_name": name,
                    "coordinate": coordinate,
                    "candidate": float(candidate),
                    "reference": float(reference),
                }
            )
    return rows


def _profile_evidence(dtype: torch.dtype, profile: str) -> dict[str, JSONValue]:
    manual_model = _TinyCausalModel(dtype=dtype)
    manual_model.eval()
    reference = _manual_full_route(manual_model)

    per_token_model = _TinyCausalModel(dtype=dtype)
    per_token_model.eval()
    per_token_routes = _manual_per_token_routes(per_token_model)
    per_token_gradient = _equal_gradient([item.gradient for item in per_token_routes])
    per_token_loss = sum(item.loss for item in per_token_routes) / len(per_token_routes)

    adapter_model = _TinyCausalModel(dtype=dtype)
    adapter_model.eval()
    adapter_full = _adapter_route(adapter_model, (0, 1, 2, 3))
    pre_shifted_model = _TinyCausalModel(dtype=dtype)
    pre_shifted_model.eval()
    pre_shifted_full = _pre_shifted_adapter_route(pre_shifted_model, (0, 1, 2, 3))

    sample_model = _TinyCausalModel(dtype=dtype)
    sample_model.eval()
    sample_routes = [_adapter_route(sample_model, (sample,)) for sample in range(4)]
    sample_reconstruction = _weighted_gradient(
        [item.gradient for item in sample_routes], [item.effective_count for item in sample_routes]
    )
    sample_loss = sum(item.numerator for item in sample_routes) / sum(
        item.effective_count for item in sample_routes
    )

    equal_groups_m2 = ((0, 1), (2, 3))
    equal_groups_m4 = ((0,), (1,), (2,), (3,))
    equal_m2_model = _TinyCausalModel(dtype=dtype)
    equal_m2_model.eval()
    equal_m2 = [_adapter_route(equal_m2_model, group) for group in equal_groups_m2]
    if len({item.effective_count for item in equal_m2}) != 1:
        raise Stage1GradientScaleError("S1_4_EQUAL_M2_COUNTS_NOT_EQUAL")
    equal_m2_gradient = _equal_gradient([item.gradient for item in equal_m2])
    equal_m2_loss = sum(item.loss for item in equal_m2) / len(equal_m2)

    equal_reverse_model = _TinyCausalModel(dtype=dtype)
    equal_reverse_model.eval()
    equal_reversed = [_adapter_route(equal_reverse_model, group) for group in reversed(equal_groups_m2)]
    equal_reversed_gradient = _equal_gradient([item.gradient for item in equal_reversed])
    equal_reversed_loss = sum(item.loss for item in equal_reversed) / len(equal_reversed)

    equal_m4_model = _TinyCausalModel(dtype=dtype)
    equal_m4_model.eval()
    equal_m4 = [_adapter_route(equal_m4_model, group) for group in equal_groups_m4]
    if len({item.effective_count for item in equal_m4}) != 1:
        raise Stage1GradientScaleError("S1_4_EQUAL_M4_COUNTS_NOT_EQUAL")
    equal_m4_gradient = _equal_gradient([item.gradient for item in equal_m4])
    equal_m4_loss = sum(item.loss for item in equal_m4) / len(equal_m4)

    weighted_groups = ((0,), (1, 2, 3))
    weighted_model = _TinyCausalModel(dtype=dtype)
    weighted_model.eval()
    weighted_routes = [_adapter_route(weighted_model, group) for group in weighted_groups]
    weighted_counts = [item.effective_count for item in weighted_routes]
    weighted_gradient = _weighted_gradient([item.gradient for item in weighted_routes], weighted_counts)
    weighted_loss = sum(item.numerator for item in weighted_routes) / sum(weighted_counts)
    incorrect_equal_gradient = _equal_gradient([item.gradient for item in weighted_routes])
    incorrect_equal_loss = sum(item.loss for item in weighted_routes) / len(weighted_routes)

    sum_model = _TinyCausalModel(dtype=dtype)
    sum_model.eval()
    inputs, labels, mask = _fixture_tensors()
    loss_batch = causal_lm_loss(sum_model(inputs), labels, mask)
    sum_gradient = _autograd_gradient(loss_batch.loss_numerator, sum_model)
    mean_model = _TinyCausalModel(dtype=dtype)
    mean_model.eval()
    mean_loss_batch = causal_lm_loss(mean_model(inputs), labels, mask)
    mean_gradient = _autograd_gradient(mean_loss_batch.mean_loss, mean_model)
    sum_divided_gradient = TensorMap(
        {name: value / loss_batch.effective_count for name, value in sum_gradient.items()}
    )

    accumulation_rows: list[dict[str, JSONValue]] = []
    comparisons: list[dict[str, JSONValue]] = []

    def add_gradient(
        comparison_id: str, actual: TensorMap, oracle: TensorMap
    ) -> dict[str, JSONValue]:
        comparison = _map_comparison(
            comparison_id=comparison_id,
            profile=profile,
            actual=actual,
            oracle=oracle,
        )
        comparisons.append(comparison)
        return comparison

    def add_loss(comparison_id: str, actual: float, oracle: float) -> dict[str, JSONValue]:
        comparison = _scalar_comparison(
            comparison_id=comparison_id,
            profile=profile,
            actual=actual,
            oracle=oracle,
        )
        comparisons.append(comparison)
        return comparison

    add_gradient("adapter_full_gradient", adapter_full.gradient, reference.gradient)
    add_loss("adapter_full_loss", adapter_full.loss, reference.loss)
    add_gradient("pre_shifted_adapter_full_gradient", pre_shifted_full.gradient, reference.gradient)
    add_loss("pre_shifted_adapter_full_loss", pre_shifted_full.loss, reference.loss)
    add_gradient("per_sample_reconstruction_gradient", sample_reconstruction, reference.gradient)
    add_loss("per_sample_reconstruction_loss", sample_loss, reference.loss)
    add_gradient("per_token_reconstruction_gradient", per_token_gradient, reference.gradient)
    add_loss("per_token_reconstruction_loss", per_token_loss, reference.loss)
    add_gradient("equal_microbatch_m2_gradient", equal_m2_gradient, reference.gradient)
    add_loss("equal_microbatch_m2_loss", equal_m2_loss, reference.loss)
    add_gradient("equal_microbatch_reordered_gradient", equal_reversed_gradient, reference.gradient)
    add_loss("equal_microbatch_reordered_loss", equal_reversed_loss, reference.loss)
    add_gradient("equal_microbatch_m4_gradient", equal_m4_gradient, reference.gradient)
    add_loss("equal_microbatch_m4_loss", equal_m4_loss, reference.loss)
    add_gradient("token_weighted_microbatch_gradient", weighted_gradient, reference.gradient)
    add_loss("token_weighted_microbatch_loss", weighted_loss, reference.loss)
    add_gradient("sum_divided_by_effective_count_gradient", sum_divided_gradient, mean_gradient)
    add_loss(
        "sum_divided_by_effective_count_loss",
        float(loss_batch.mean_loss.detach().to(dtype=torch.float64).item()),
        reference.loss,
    )

    for split_id, groups in (
        ("accumulation_m2", equal_groups_m2),
        ("accumulation_m4", equal_groups_m4),
        ("accumulation_weighted_m2", weighted_groups),
    ):
        accumulation_model = _TinyCausalModel(dtype=dtype)
        accumulation_model.eval()
        accumulated, local, clear_was_complete = _adapter_accumulation_route(accumulation_model, groups)
        comparison_id = f"{split_id}_gradient"
        accumulated_comparison = add_gradient(comparison_id, accumulated, reference.gradient)
        local_reconstruction = _weighted_gradient(
            [item.gradient for item in local], [item.effective_count for item in local]
        )
        add_gradient(f"{split_id}_local_gradient_reconstruction", local_reconstruction, reference.gradient)
        add_loss(
            f"{split_id}_local_loss_reconstruction",
            sum(item.numerator for item in local) / sum(item.effective_count for item in local),
            reference.loss,
        )
        accumulation_rows.append(
            {
                "split_id": split_id,
                "microbatch_count": len(groups),
                "effective_counts": [item.effective_count for item in local],
                "clear_was_complete": clear_was_complete,
                "comparison_id": comparison_id,
                "normalized_l2_error": accumulated_comparison["global"]["normalized_l2_error"],  # type: ignore[index]
            }
        )

    negative = _map_comparison(
        comparison_id="negative_control_unweighted_unequal_microbatch_gradient",
        profile=profile,
        actual=incorrect_equal_gradient,
        oracle=reference.gradient,
    )
    negative_loss = _scalar_comparison(
        comparison_id="negative_control_unweighted_unequal_microbatch_loss",
        profile=profile,
        actual=incorrect_equal_loss,
        oracle=reference.loss,
    )
    def threshold_exceeded(comparison: Mapping[str, JSONValue]) -> bool:
        result = comparison["global"]
        assert isinstance(result, Mapping)
        if result.get("nonfinite_count") != 0:
            return False
        maximum = result.get("max_absolute_error")
        absolute_threshold = result.get("absolute_threshold")
        scaled = result.get("scaled_max_error")
        rtol = result.get("rtol")
        normalized = result.get("normalized_l2_error")
        normalized_limit = result.get("normalized_l2_limit")
        return bool(
            (isinstance(maximum, (float, int)) and isinstance(absolute_threshold, (float, int)) and maximum > absolute_threshold)
            or (isinstance(scaled, (float, int)) and isinstance(rtol, (float, int)) and scaled > rtol)
            or (isinstance(normalized, (float, int)) and isinstance(normalized_limit, (float, int)) and normalized > normalized_limit)
        )

    if (
        bool(negative["passed"])
        or bool(negative_loss["passed"])
        or not threshold_exceeded(negative)
        or not threshold_exceeded(negative_loss)
    ):
        raise Stage1GradientScaleError("S1_4_NEGATIVE_CONTROL_NOT_DETECTABLE")

    # Exact routes run with dropout disabled.  Their forward/backward calls
    # must not advance the CPU RNG, and replaying them must be bit-identical.
    rng_model = _TinyCausalModel(dtype=dtype)
    rng_model.eval()
    rng_before = _rng_digest()
    rng_first = _adapter_route(rng_model, (0, 1, 2, 3))
    rng_middle = _rng_digest()
    rng_second = _adapter_route(rng_model, (0, 1, 2, 3))
    rng_after = _rng_digest()
    rng_repeat = _map_comparison(
        comparison_id="rng_eval_repeat_gradient",
        profile=profile,
        actual=rng_second.gradient,
        oracle=rng_first.gradient,
    )
    if not bool(rng_repeat["passed"]) or not (rng_before == rng_middle == rng_after):
        raise Stage1GradientScaleError("S1_4_RNG_EXACT_BOUNDARY_FAILED")
    comparisons.append(rng_repeat)

    dropout_model = _TinyCausalModel(dtype=dtype)
    dropout_model.train()
    torch.manual_seed(20260814)
    dropout_before = _rng_digest()
    dropout_model(_fixture_tensors()[0][:1])
    dropout_middle = _rng_digest()
    dropout_model(_fixture_tensors()[0][1:2])
    dropout_after = _rng_digest()
    if len({dropout_before, dropout_middle, dropout_after}) != 3:
        raise Stage1GradientScaleError("S1_4_DROPOUT_STREAM_NOT_ADVANCING")

    comparison_ids = tuple(str(item["comparison_id"]) for item in comparisons)
    if comparison_ids != _PROFILE_COMPARISON_IDS:
        raise Stage1GradientScaleError("S1_4_COMPARISON_MATRIX_INTERNAL_INVALID")

    return {
        "profile": profile,
        "dtype": str(dtype),
        "reference": {
            "route": "independent_full_batch_token_nll",
            "loss": reference.loss,
            "numerator": reference.numerator,
            "effective_count": reference.effective_count,
        },
        "effective_token_counts": {
            "per_sample": [item.effective_count for item in sample_routes],
            "equal_m2": [item.effective_count for item in equal_m2],
            "equal_m4": [item.effective_count for item in equal_m4],
            "weighted_m2": weighted_counts,
            "weighted_normalized_weights": [count / sum(weighted_counts) for count in weighted_counts],
            "equal_m2_weights": [1.0 / len(equal_m2)] * len(equal_m2),
            "equal_m4_weights": [1.0 / len(equal_m4)] * len(equal_m4),
        },
        "comparisons": comparisons,
        "negative_control": {
            "gradient": negative,
            "loss": negative_loss,
            "expected": "FAIL",
            "detected": True,
            "unequal_counts": weighted_counts,
        },
        "sum_mean": {
            "effective_count": loss_batch.effective_count,
            "sum_to_mean_factor": loss_batch.effective_count,
            "comparison_id": "sum_divided_by_effective_count_gradient",
        },
        "accumulation": accumulation_rows,
        "rng": {
            "exact_equivalence": {
                "model_mode": "eval",
                "dropout_disabled": True,
                "cpu_rng_before": rng_before,
                "cpu_rng_between": rng_middle,
                "cpu_rng_after": rng_after,
                "comparison_id": "rng_eval_repeat_gradient",
            },
            "dropout_smoke": {
                "model_mode": "train",
                "exact_equivalence_gate": False,
                "cpu_rng_before": dropout_before,
                "cpu_rng_after_first_microbatch": dropout_middle,
                "cpu_rng_after_second_microbatch": dropout_after,
                "independent_streams_observed": True,
            },
        },
        "scatter_rows": _scatter_rows(
            profile, "equal_microbatch_m2_gradient", equal_m2_gradient, reference.gradient
        )
        + _scatter_rows(
            profile, "token_weighted_microbatch_gradient", weighted_gradient, reference.gradient
        ),
    }


def _source_hashes(repository_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise Stage1GradientScaleError(f"S1_4_SOURCE_FILE_MISSING:{relative}")
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _comparison_rows(profiles: Iterable[Mapping[str, JSONValue]]) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for profile in profiles:
        for comparison in profile["comparisons"]:  # type: ignore[index]
            assert isinstance(comparison, Mapping)
            for tensor in comparison["per_tensor"]:  # type: ignore[index]
                assert isinstance(tensor, Mapping)
                rows.append(
                    {
                        "profile": comparison["profile"],
                        "comparison_id": comparison["comparison_id"],
                        "object_id": comparison["object_id"],
                        **dict(tensor),
                    }
                )
    return rows


def _table_projection(
    profiles: Iterable[Mapping[str, JSONValue]],
) -> tuple[list[dict[str, JSONValue]], list[dict[str, JSONValue]], list[dict[str, JSONValue]]]:
    """Return the only table projection admitted for a report.

    The table is a mechanically reproducible presentation role, not an
    independently editable assertion.  Keeping this transformation central
    lets the validator fail closed even when an attacker rehashes all roles.
    """

    profile_list = list(profiles)
    rows = _comparison_rows(profile_list)
    scatter = [dict(row) for profile in profile_list for row in profile["scatter_rows"]]  # type: ignore[index]
    accumulation = [
        {"profile": profile["profile"], **dict(row)}  # type: ignore[index]
        for profile in profile_list
        for row in profile["accumulation"]  # type: ignore[index]
    ]
    return rows, scatter, accumulation


def _projection_hash(
    rows: list[dict[str, JSONValue]],
    scatter: list[dict[str, JSONValue]],
    accumulation: list[dict[str, JSONValue]],
) -> str:
    return canonical_json_hash(
        {"rows": rows, "scatter_rows": scatter, "accumulation_rows": accumulation}
    )


def build_stage1_s14_evidence(
    repository_root: str | Path,
    *,
    producer_commit: str,
    scope: str = "local_fixture",
    upstream_evidence: Mapping[str, JSONValue] | None = None,
) -> dict[str, dict[str, JSONValue]]:
    """Build three role-separated S1.4 artifacts without publishing them."""

    if not isinstance(producer_commit, str) or re.fullmatch(r"[0-9a-f]{40}", producer_commit) is None:
        raise Stage1GradientScaleError("S1_4_PRODUCER_COMMIT_INVALID")
    if scope not in {"local_fixture", "formal"}:
        raise Stage1GradientScaleError("S1_4_SCOPE_INVALID")
    if scope == "local_fixture" and upstream_evidence not in (None, {}):
        raise Stage1GradientScaleError("S1_4_LOCAL_UPSTREAM_MUST_BE_EMPTY")
    if scope == "formal" and (
        not isinstance(upstream_evidence, Mapping)
        or set(upstream_evidence) != _FORMAL_UPSTREAM_FIELDS
        or upstream_evidence.get("s1_3_index_sha256") != _S1_3_V2_INDEX_SHA256
    ):
        raise Stage1GradientScaleError("S1_4_FORMAL_S1_3_V2_HANDOFF_REQUIRED")
    if scope == "formal":
        assert isinstance(upstream_evidence, Mapping)
        for field in _FORMAL_UPSTREAM_FIELDS:
            value = upstream_evidence[field]
            if field == "s1_3_index_ref":
                valid = isinstance(value, str) and bool(value)
            else:
                valid = isinstance(value, str) and len(value) == 64 and all(
                    character in "0123456789abcdef" for character in value
                )
            if not valid:
                raise Stage1GradientScaleError(f"S1_4_FORMAL_S1_3_FIELD_INVALID:{field}")
    repository = Path(repository_root).resolve()
    # Do not leak a fixture seed into callers.  This also makes the RNG-state
    # summaries stable across independently constructed local artifacts.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026081404)
        profiles = [
            _profile_evidence(torch.float64, "T64_ORACLE"),
            _profile_evidence(torch.float32, "T32_SINGLE"),
        ]
        inputs, labels, mask = _fixture_tensors()
        try:
            causal_lm_loss(
                _TinyCausalModel(dtype=torch.float64)(inputs[:1]),
                labels[:1],
                torch.zeros_like(mask[:1]),
            )
        except CoreContractError:
            zero_effective_token_rejected = True
        else:  # pragma: no cover - a contract regression must stop publication
            raise Stage1GradientScaleError("S1_4_ZERO_EFFECTIVE_TOKEN_ACCEPTED")
        try:
            pre_shifted_causal_lm_loss(
                _TinyCausalModel(dtype=torch.float64)(inputs[:1])[:, :-1, :],
                labels[:1, 1:],
                torch.zeros_like(mask[:1, 1:]),
            )
        except CoreContractError:
            pre_shifted_zero_effective_token_rejected = True
        else:  # pragma: no cover - a contract regression must stop publication
            raise Stage1GradientScaleError("S1_4_PRE_SHIFTED_ZERO_EFFECTIVE_TOKEN_ACCEPTED")
    comparisons, scatter_rows, accumulation_rows = _table_projection(profiles)
    positive_pass = all(
        bool(comparison["passed"])
        for profile in profiles
        for comparison in profile["comparisons"]  # type: ignore[index]
    )
    negative_detected = all(bool(profile["negative_control"]["detected"]) for profile in profiles)  # type: ignore[index]
    gate_status = "PASS" if scope == "formal" and positive_pass and negative_detected else "NOT_RUN"
    report_body: dict[str, JSONValue] = {
        "schema_version": REPORT_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "task_id": TASK_ID,
        "gate_id": GATE_ID,
        "producer_commit": producer_commit,
        "scope": scope,
        "status": "PASS" if positive_pass and negative_detected else "FAIL",
        "gate_status": gate_status,
        "loss_adapter": {
            "name": "causal_lm_loss",
            "pre_shifted_adapter": "pre_shifted_causal_lm_loss_without_second_shift",
            "provider_default_reduction": "forbidden",
            "returns": ["loss_numerator", "effective_count", "mean_loss"],
            "reduction": "sum_valid_target_token_nll / global_effective_target_token_count",
            "zero_effective_token": "reject_fail_closed",
        },
        "sample_contract": {
            "ordered_sample_ids": ["s14-sample-0", "s14-sample-1", "s14-sample-2", "s14-sample-3"],
            "equal_m2_groups": [["s14-sample-0", "s14-sample-1"], ["s14-sample-2", "s14-sample-3"]],
            "equal_m4_groups": [["s14-sample-0"], ["s14-sample-1"], ["s14-sample-2"], ["s14-sample-3"]],
            "weighted_m2_groups": [["s14-sample-0"], ["s14-sample-1", "s14-sample-2", "s14-sample-3"]],
            "ignored_target_locations": [
                {
                    "sample_id": "s14-sample-0",
                    "target_position": 1,
                    "attention_mask": 1,
                    "label": -100,
                }
            ],
            "per_sample_effective_target_token_counts": [2, 2, 2, 2],
            "zero_effective_token_rejected": zero_effective_token_rejected,
            "pre_shifted_zero_effective_token_rejected": pre_shifted_zero_effective_token_rejected,
        },
        "profiles": profiles,
        "upstream": dict(upstream_evidence or {}),
        "implementation_source_sha256": _source_hashes(repository),
    }
    report = dict(report_body)
    report["report_hash"] = canonical_json_hash(report_body)

    table_body: dict[str, JSONValue] = {
        "schema_version": TABLE_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "report_hash": report["report_hash"],
        "rows": comparisons,
        "scatter_rows": scatter_rows,
        "accumulation_rows": accumulation_rows,
        "projection_hash": _projection_hash(comparisons, scatter_rows, accumulation_rows),
    }
    table = dict(table_body)
    table["table_hash"] = canonical_json_hash(table_body)

    gate_body: dict[str, JSONValue] = {
        "schema_version": GATE_SCHEMA,
        "task_id": TASK_ID,
        "gate_id": GATE_ID,
        "scope": scope,
        "status": gate_status,
        "report_hash": report["report_hash"],
        "comparison_table_hash": table["table_hash"],
        "requirements": {
            "full_batch_and_per_sample": positive_pass,
            "per_token_oracle": positive_pass,
            "equal_microbatch_m_and_permutation": positive_pass,
            "token_weighted_microbatch": positive_pass,
            "sum_mean_and_accumulation": positive_pass,
            "negative_control_detected": negative_detected,
            "rng_boundary": positive_pass,
            "all_tensor_rows_pass": all(bool(row["passed"]) for row in comparisons),
            "table_projection_exact": True,
        },
    }
    gate = dict(gate_body)
    gate["artifact_hash"] = canonical_json_hash(gate_body)
    return {
        "gradient_scale_report": report,
        "comparison_table": table,
        "gate_record": gate,
    }


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def validate_stage1_s14_evidence(evidence: Mapping[str, Any]) -> dict[str, JSONValue]:
    """Fail closed on rehashed role, projection, or contract tampering."""

    expected_roles = {"gradient_scale_report", "comparison_table", "gate_record"}
    expected_report_keys = {
        "schema_version", "fixture_id", "task_id", "gate_id", "producer_commit", "scope", "status",
        "gate_status", "loss_adapter", "sample_contract", "profiles", "upstream",
        "implementation_source_sha256", "report_hash",
    }
    expected_table_keys = {
        "schema_version", "fixture_id", "report_hash", "rows", "scatter_rows", "accumulation_rows",
        "projection_hash", "table_hash",
    }
    expected_gate_keys = {
        "schema_version", "task_id", "gate_id", "scope", "status", "report_hash",
        "comparison_table_hash", "requirements", "artifact_hash",
    }
    if set(evidence) != expected_roles:
        raise Stage1GradientScaleError("S1_4_EVIDENCE_ROLE_SET_INVALID")
    report = evidence["gradient_scale_report"]
    table = evidence["comparison_table"]
    gate = evidence["gate_record"]
    if not all(isinstance(value, Mapping) for value in (report, table, gate)):
        raise Stage1GradientScaleError("S1_4_EVIDENCE_ROLE_NOT_OBJECT")
    if set(report) != expected_report_keys or set(table) != expected_table_keys or set(gate) != expected_gate_keys:
        raise Stage1GradientScaleError("S1_4_ROLE_KEY_SET_INVALID")
    report_body = dict(report)
    report_hash = report_body.pop("report_hash", None)
    if report_hash != canonical_json_hash(report_body):
        raise Stage1GradientScaleError("S1_4_REPORT_HASH_INVALID")
    table_body = dict(table)
    table_hash = table_body.pop("table_hash", None)
    if table_hash != canonical_json_hash(table_body) or table.get("report_hash") != report_hash:
        raise Stage1GradientScaleError("S1_4_TABLE_HASH_OR_REPORT_BINDING_INVALID")
    gate_body = dict(gate)
    gate_hash = gate_body.pop("artifact_hash", None)
    if gate_hash != canonical_json_hash(gate_body):
        raise Stage1GradientScaleError("S1_4_GATE_HASH_INVALID")
    if gate.get("report_hash") != report_hash or gate.get("comparison_table_hash") != table_hash:
        raise Stage1GradientScaleError("S1_4_GATE_ROLE_BINDING_INVALID")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or table.get("schema_version") != TABLE_SCHEMA
        or gate.get("schema_version") != GATE_SCHEMA
        or report.get("fixture_id") != FIXTURE_ID
        or report.get("task_id") != TASK_ID
        or report.get("gate_id") != GATE_ID
        or gate.get("gate_id") != GATE_ID
        or not isinstance(report.get("producer_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", report["producer_commit"]) is None
    ):
        raise Stage1GradientScaleError("S1_4_SCHEMA_IDENTITY_INVALID")
    scope = report.get("scope")
    if scope not in {"local_fixture", "formal"} or gate.get("scope") != scope:
        raise Stage1GradientScaleError("S1_4_SCOPE_BINDING_INVALID")
    if (
        (scope == "formal" and (report.get("gate_status") != "PASS" or gate.get("status") != "PASS"))
        or (scope == "local_fixture" and (report.get("gate_status") != "NOT_RUN" or gate.get("status") != "NOT_RUN"))
    ):
        raise Stage1GradientScaleError("S1_4_GATE_SCOPE_STATUS_INVALID")
    profiles = report.get("profiles")
    sample_contract = report.get("sample_contract")
    upstream = report.get("upstream")
    rows = table.get("rows")
    scatter_rows = table.get("scatter_rows")
    accumulation_rows = table.get("accumulation_rows")
    requirements = gate.get("requirements")
    if not isinstance(profiles, list) or not all(isinstance(item, Mapping) for item in profiles):
        raise Stage1GradientScaleError("S1_4_PROFILE_SHAPE_INVALID")
    if not isinstance(rows, list) or not isinstance(scatter_rows, list) or not isinstance(accumulation_rows, list) or not isinstance(requirements, Mapping):
        raise Stage1GradientScaleError("S1_4_REPORT_SHAPE_INVALID")
    if scope == "formal":
        if (
            not isinstance(upstream, Mapping)
            or set(upstream) != _FORMAL_UPSTREAM_FIELDS
            or upstream.get("s1_3_index_sha256") != _S1_3_V2_INDEX_SHA256
            or not all(
                isinstance(upstream[field], str)
                and (bool(upstream[field]) if field == "s1_3_index_ref" else _is_digest(upstream[field]))
                for field in _FORMAL_UPSTREAM_FIELDS
            )
        ):
            raise Stage1GradientScaleError("S1_4_FORMAL_UPSTREAM_BINDING_INVALID")
    elif upstream != {}:
        raise Stage1GradientScaleError("S1_4_LOCAL_UPSTREAM_MUST_BE_EMPTY")
    expected_sample_ids = list(_SAMPLE_IDS)
    expected_sample_contract = {
        "ordered_sample_ids": expected_sample_ids,
        "equal_m2_groups": [expected_sample_ids[:2], expected_sample_ids[2:]],
        "equal_m4_groups": [[sample_id] for sample_id in expected_sample_ids],
        "weighted_m2_groups": [[expected_sample_ids[0]], expected_sample_ids[1:]],
        "ignored_target_locations": [{"sample_id": "s14-sample-0", "target_position": 1, "attention_mask": 1, "label": -100}],
        "per_sample_effective_target_token_counts": [2, 2, 2, 2],
        "zero_effective_token_rejected": True,
        "pre_shifted_zero_effective_token_rejected": True,
    }
    if not isinstance(sample_contract, Mapping) or dict(sample_contract) != expected_sample_contract:
        raise Stage1GradientScaleError("S1_4_SAMPLE_OR_ZERO_TOKEN_CONTRACT_INVALID")
    source_hashes = report.get("implementation_source_sha256")
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != set(_SOURCE_FILES)
        or not all(_is_digest(value) for value in source_hashes.values())
    ):
        raise Stage1GradientScaleError("S1_4_IMPLEMENTATION_SOURCE_SET_INVALID")
    profile_names = tuple(item.get("profile") for item in profiles)
    if profile_names != _PROFILE_NAMES:
        raise Stage1GradientScaleError("S1_4_PROFILE_SET_INVALID")
    expected_loss_adapter = {
        "name": "causal_lm_loss",
        "pre_shifted_adapter": "pre_shifted_causal_lm_loss_without_second_shift",
        "provider_default_reduction": "forbidden",
        "returns": ["loss_numerator", "effective_count", "mean_loss"],
        "reduction": "sum_valid_target_token_nll / global_effective_target_token_count",
        "zero_effective_token": "reject_fail_closed",
    }
    if report.get("loss_adapter") != expected_loss_adapter:
        raise Stage1GradientScaleError("S1_4_LOSS_ADAPTER_CONTRACT_INVALID")
    expected_profile_keys = {
        "profile", "dtype", "reference", "effective_token_counts", "comparisons", "negative_control",
        "sum_mean", "accumulation", "rng", "scatter_rows",
    }
    expected_comparison_keys = {"comparison_id", "profile", "object_id", "passed", "global", "per_tensor"}
    expected_tensor_keys = {
        "parameter_name", "passed", "branch", "comparison_dtype", "natural_scale", "atol", "rtol",
        "normalized_l2_limit", "near_zero_threshold", "absolute_threshold", "max_absolute_error",
        "scaled_max_error", "normalized_l2_error", "nonfinite_count", "worst_parameter",
    }
    expected_scatter_keys = {"profile", "route_id", "parameter_name", "coordinate", "candidate", "reference"}
    expected_accumulation_keys = {
        "split_id", "microbatch_count", "effective_counts", "clear_was_complete", "comparison_id",
        "normalized_l2_error",
    }
    for profile in profiles:
        if set(profile) != expected_profile_keys:
            raise Stage1GradientScaleError("S1_4_PROFILE_KEY_SET_INVALID")
        comparisons = profile.get("comparisons")
        negative = profile.get("negative_control")
        counts = profile.get("effective_token_counts")
        if not isinstance(comparisons, list) or not isinstance(negative, Mapping) or not isinstance(counts, Mapping):
            raise Stage1GradientScaleError("S1_4_PROFILE_CONTENT_INVALID")
        if tuple(item.get("comparison_id") for item in comparisons if isinstance(item, Mapping)) != _PROFILE_COMPARISON_IDS:
            raise Stage1GradientScaleError("S1_4_COMPARISON_SET_INVALID")
        if not all(
            isinstance(item, Mapping)
            and set(item) == expected_comparison_keys
            and item.get("profile") == profile["profile"]
            and item.get("passed") is True
            and isinstance(item.get("per_tensor"), list)
            and item["per_tensor"]
            and all(isinstance(tensor, Mapping) and set(tensor) == expected_tensor_keys for tensor in item["per_tensor"])
            for item in comparisons
        ):
            raise Stage1GradientScaleError("S1_4_POSITIVE_COMPARISON_FAILED")
        if (
            set(negative) != {"gradient", "loss", "expected", "detected", "unequal_counts"}
            or negative.get("expected") != "FAIL"
            or negative.get("detected") is not True
            or negative.get("unequal_counts") != [2, 6]
            or not isinstance(negative.get("gradient"), Mapping)
            or not isinstance(negative.get("loss"), Mapping)
            or negative["gradient"].get("passed") is not False
            or negative["loss"].get("passed") is not False
        ):
            raise Stage1GradientScaleError("S1_4_NEGATIVE_CONTROL_INVALID")
        if (
            counts.get("per_sample") != [2, 2, 2, 2]
            or counts.get("equal_m2") != [4, 4]
            or counts.get("equal_m4") != [2, 2, 2, 2]
            or counts.get("weighted_m2") != [2, 6]
            or counts.get("equal_m2_weights") != [0.5, 0.5]
            or counts.get("weighted_normalized_weights") != [0.25, 0.75]
            or counts.get("equal_m4_weights") != [0.25, 0.25, 0.25, 0.25]
        ):
            raise Stage1GradientScaleError("S1_4_WEIGHT_CONTRACT_INVALID")
        if (
            not isinstance(profile.get("reference"), Mapping)
            or set(profile["reference"]) != {"route", "loss", "numerator", "effective_count"}
            or profile["reference"].get("route") != "independent_full_batch_token_nll"
            or profile["reference"].get("effective_count") != 8
            or not isinstance(profile.get("scatter_rows"), list)
            or not profile["scatter_rows"]
            or not all(
                isinstance(row, Mapping)
                and set(row) == expected_scatter_keys
                and row.get("profile") == profile["profile"]
                and row.get("route_id") in {"equal_microbatch_m2_gradient", "token_weighted_microbatch_gradient"}
                for row in profile["scatter_rows"]
            )
            or not isinstance(profile.get("accumulation"), list)
            or len(profile["accumulation"]) != 3
            or not all(
                isinstance(row, Mapping)
                and set(row) == expected_accumulation_keys
                and row.get("clear_was_complete") is True
                for row in profile["accumulation"]
            )
        ):
            raise Stage1GradientScaleError("S1_4_PROFILE_PROJECTION_SHAPE_INVALID")
    expected_rows, expected_scatter, expected_accumulation = _table_projection(profiles)
    if rows != expected_rows or scatter_rows != expected_scatter or accumulation_rows != expected_accumulation:
        raise Stage1GradientScaleError("S1_4_TABLE_PROJECTION_INVALID")
    if table.get("projection_hash") != _projection_hash(expected_rows, expected_scatter, expected_accumulation):
        raise Stage1GradientScaleError("S1_4_TABLE_PROJECTION_HASH_INVALID")
    if not rows or not all(isinstance(row, Mapping) and row.get("passed") is True for row in rows):
        raise Stage1GradientScaleError("S1_4_PER_TENSOR_GATE_FAILED")
    expected_requirements = {
        "full_batch_and_per_sample", "per_token_oracle", "equal_microbatch_m_and_permutation",
        "token_weighted_microbatch", "sum_mean_and_accumulation", "negative_control_detected",
        "rng_boundary", "all_tensor_rows_pass", "table_projection_exact",
    }
    if set(requirements) != expected_requirements or not all(value is True for value in requirements.values()):
        raise Stage1GradientScaleError("S1_4_GATE_REQUIREMENT_FAILED")
    return {
        "status": "PASS",
        "report_hash": report_hash,
        "comparison_table_hash": table_hash,
        "gate_artifact_hash": gate_hash,
        "profile_count": len(profiles),
        "tensor_row_count": len(rows),
        "replay_hash": canonical_json_hash(
            {
                "report_hash": report_hash,
                "table_hash": table_hash,
                "gate_hash": gate_hash,
                "profiles": list(profile_names),
                "tensor_row_count": len(rows),
                "projection_hash": table["projection_hash"],
            }
        ),
    }


def replay_stage1_s14_evidence(
    evidence: Mapping[str, Any], repository_root: str | Path
) -> dict[str, JSONValue]:
    """Recompute the fixture from serialized producer/scope/upstream inputs.

    This is deliberately more than hash aggregation: a jointly rehashed
    numerical mutation can survive structural checks but cannot match the
    deterministic rebuild's per-comparison payload.
    """

    structural = validate_stage1_s14_evidence(evidence)
    report = evidence["gradient_scale_report"]
    assert isinstance(report, Mapping)
    rebuilt = build_stage1_s14_evidence(
        repository_root,
        producer_commit=str(report["producer_commit"]),
        scope=str(report["scope"]),
        upstream_evidence=report["upstream"],
    )
    for role in ("gradient_scale_report", "comparison_table", "gate_record"):
        if evidence[role] != rebuilt[role]:
            raise Stage1GradientScaleError(f"S1_4_OFFLINE_REPLAY_ROLE_MISMATCH:{role}")
    serialized_profiles = report["profiles"]
    rebuilt_profiles = rebuilt["gradient_scale_report"]["profiles"]
    assert isinstance(serialized_profiles, list) and isinstance(rebuilt_profiles, list)
    comparison_hashes: dict[str, str] = {}
    for serialized_profile, rebuilt_profile in zip(serialized_profiles, rebuilt_profiles, strict=True):
        assert isinstance(serialized_profile, Mapping) and isinstance(rebuilt_profile, Mapping)
        for serialized_comparison, rebuilt_comparison in zip(
            serialized_profile["comparisons"], rebuilt_profile["comparisons"], strict=True  # type: ignore[index]
        ):
            assert isinstance(serialized_comparison, Mapping) and isinstance(rebuilt_comparison, Mapping)
            comparison_id = str(serialized_comparison["comparison_id"])
            if serialized_comparison != rebuilt_comparison:
                raise Stage1GradientScaleError(
                    f"S1_4_OFFLINE_REPLAY_COMPARISON_MISMATCH:{serialized_profile['profile']}:{comparison_id}"
                )
            comparison_hashes[f"{serialized_profile['profile']}:{comparison_id}"] = canonical_json_hash(
                dict(serialized_comparison)
            )
    replay_body: dict[str, JSONValue] = {
        "schema_version": "stage1-s1-4-offline-replay-v1",
        "status": "PASS",
        "producer_commit": report["producer_commit"],
        "scope": report["scope"],
        "upstream": dict(report["upstream"]),
        "report_hash": structural["report_hash"],
        "comparison_table_hash": structural["comparison_table_hash"],
        "gate_artifact_hash": structural["gate_artifact_hash"],
        "role_hashes": {
            "gradient_scale_report": structural["report_hash"],
            "comparison_table": structural["comparison_table_hash"],
            "gate_record": structural["gate_artifact_hash"],
        },
        "comparison_hashes": comparison_hashes,
    }
    replay = dict(replay_body)
    replay["replay_hash"] = canonical_json_hash(replay_body)
    return replay


__all__ = [
    "FIXTURE_ID",
    "GATE_ID",
    "GATE_SCHEMA",
    "REPORT_SCHEMA",
    "TABLE_SCHEMA",
    "Stage1GradientScaleError",
    "build_stage1_s14_evidence",
    "replay_stage1_s14_evidence",
    "validate_stage1_s14_evidence",
]
