"""Independent FP64 oracle primitives for the S1.5 estimator Gate.

This module deliberately does not import the production estimator kernels or
their sufficient-statistics classes.  It uses explicit Python loops over
serialized microbatch gradients so a shared algebraic helper cannot make the
producer and reference agree by accident.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from .core.tensors import TensorMap


class Stage1EstimatorOracleError(RuntimeError):
    """The immutable S1.5 fixture or its serialized oracle input is invalid."""


def _check_map(value: TensorMap, *, field: str, reference: TensorMap | None = None) -> None:
    if reference is not None:
        reference.assert_compatible(value, require_dtype_device=True)
    for name, tensor in value.items():
        if not tensor.is_floating_point():
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] is not floating point")
        if not bool(torch.isfinite(tensor).all()):
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] is non-finite")


def _check_samples(samples: Sequence[TensorMap], *, field: str) -> TensorMap:
    if not samples:
        raise Stage1EstimatorOracleError(f"{field} is empty")
    reference = samples[0]
    _check_map(reference, field=f"{field}[0]")
    for index, sample in enumerate(samples[1:], start=1):
        _check_map(sample, field=f"{field}[{index}]", reference=reference)
    return reference


def mean_gradient(samples: Sequence[TensorMap]) -> TensorMap:
    reference = _check_samples(samples, field="samples")
    totals = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    for sample in samples:
        for name, value in sample.items():
            totals[name] = totals[name] + value.to(dtype=torch.float64)
    return TensorMap({name: value / len(samples) for name, value in totals.items()}, registry=reference.registry)


def raw_core(mean: TensorMap) -> TensorMap:
    _check_map(mean, field="mean")
    return TensorMap({name: value.to(dtype=torch.float64).square() for name, value in mean.items()}, registry=mean.registry)


def double_core(left: TensorMap, right: TensorMap) -> TensorMap:
    _check_map(left, field="left")
    _check_map(right, field="right", reference=left)
    return TensorMap(
        {name: left[name].to(dtype=torch.float64) * right[name].to(dtype=torch.float64) for name in left},
        registry=left.registry,
    )


def ordered_pair_u_core(samples: Sequence[TensorMap]) -> TensorMap:
    reference = _check_samples(samples, field="samples")
    if len(samples) < 2:
        raise Stage1EstimatorOracleError("ordered pair U requires M >= 2")
    totals = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    for left_index, left in enumerate(samples):
        for right_index, right in enumerate(samples):
            if left_index == right_index:
                continue
            for name in totals:
                totals[name] = totals[name] + (
                    left[name].to(dtype=torch.float64) * right[name].to(dtype=torch.float64)
                )
    denominator = len(samples) * (len(samples) - 1)
    return TensorMap({name: value / denominator for name, value in totals.items()}, registry=reference.registry)


def unordered_pair_u_core(samples: Sequence[TensorMap]) -> TensorMap:
    reference = _check_samples(samples, field="samples")
    if len(samples) < 2:
        raise Stage1EstimatorOracleError("unordered pair U requires M >= 2")
    totals = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    for left_index in range(len(samples)):
        for right_index in range(left_index + 1, len(samples)):
            for name in totals:
                totals[name] = totals[name] + 2.0 * (
                    samples[left_index][name].to(dtype=torch.float64)
                    * samples[right_index][name].to(dtype=torch.float64)
                )
    denominator = len(samples) * (len(samples) - 1)
    return TensorMap({name: value / denominator for name, value in totals.items()}, registry=reference.registry)


def weighted_ordered_pair_u_core(
    samples: Sequence[TensorMap], weights: Sequence[float | int]
) -> TensorMap:
    reference = _check_samples(samples, field="samples")
    if len(samples) < 2 or len(samples) != len(weights):
        raise Stage1EstimatorOracleError("weighted ordered pair inputs are invalid")
    normalized = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in normalized):
        raise Stage1EstimatorOracleError("weighted ordered pair weights are invalid")
    totals = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    denominator = 0.0
    for left_index, left in enumerate(samples):
        for right_index, right in enumerate(samples):
            if left_index == right_index:
                continue
            pair_weight = normalized[left_index] * normalized[right_index]
            denominator += pair_weight
            for name in totals:
                totals[name] = totals[name] + pair_weight * (
                    left[name].to(dtype=torch.float64) * right[name].to(dtype=torch.float64)
                )
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise Stage1EstimatorOracleError("weighted ordered pair denominator is invalid")
    return TensorMap({name: value / denominator for name, value in totals.items()}, registry=reference.registry)


def sufficient_statistics(
    samples: Sequence[TensorMap], weights: Sequence[float | int]
) -> dict[str, Any]:
    """Explicitly recompute the S1/S2 and G1/G2/N1/N2 payloads in FP64."""

    reference = _check_samples(samples, field="samples")
    if len(samples) != len(weights):
        raise Stage1EstimatorOracleError("statistics weights have wrong length")
    normalized = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight <= 0.0 for weight in normalized):
        raise Stage1EstimatorOracleError("statistics weights are invalid")
    s1 = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    s2 = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    g1 = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    g2 = {name: torch.zeros_like(value, dtype=torch.float64) for name, value in reference.items()}
    for sample, weight in zip(samples, normalized, strict=True):
        for name in s1:
            gradient = sample[name].to(dtype=torch.float64)
            s1[name] = s1[name] + gradient
            s2[name] = s2[name] + gradient.square()
            g1[name] = g1[name] + weight * gradient
            g2[name] = g2[name] + weight**2 * gradient.square()
    n1 = sum(normalized)
    n2 = sum(weight**2 for weight in normalized)
    return {
        "s1": TensorMap(s1, registry=reference.registry),
        "s2": TensorMap(s2, registry=reference.registry),
        "g1": TensorMap(g1, registry=reference.registry),
        "g2": TensorMap(g2, registry=reference.registry),
        "n1": n1,
        "n2": n2,
    }


def apply_learning_rates(
    core: TensorMap,
    learning_rates: Mapping[str, float],
    *,
    coordinate_to_group: Mapping[str, str] | None = None,
    clip_factor: float = 1.0,
) -> TensorMap:
    _check_map(core, field="core")
    if coordinate_to_group is None:
        if set(learning_rates) != set(core):
            raise Stage1EstimatorOracleError("learning-rate keys must exactly match oracle coordinates")
    elif set(coordinate_to_group) != set(core) or set(learning_rates) != set(coordinate_to_group.values()):
        raise Stage1EstimatorOracleError("learning-rate group mapping must exactly match oracle coordinates")
    if not math.isfinite(clip_factor) or not 0.0 <= clip_factor <= 1.0:
        raise Stage1EstimatorOracleError("clip factor is invalid")
    scaled: dict[str, torch.Tensor] = {}
    for name, value in core.items():
        learning_rate = float(learning_rates[name if coordinate_to_group is None else coordinate_to_group[name]])
        if not math.isfinite(learning_rate) or learning_rate < 0.0:
            raise Stage1EstimatorOracleError(f"learning rate for {name!r} is invalid")
        scaled[name] = value.to(dtype=torch.float64) * learning_rate * clip_factor
    return TensorMap(scaled, registry=core.registry)


def tensor_map_to_wire(value: TensorMap) -> dict[str, dict[str, object]]:
    _check_map(value, field="wire")
    return {
        name: {
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "values": [float(item) for item in tensor.detach().to(dtype=torch.float64).reshape(-1).tolist()],
        }
        for name, tensor in value.items()
    }


def tensor_map_from_wire(
    value: object,
    *,
    field: str,
    coordinate_order: Sequence[str] | None = None,
    registry: object | None = None,
) -> TensorMap:
    if not isinstance(value, Mapping) or not value:
        raise Stage1EstimatorOracleError(f"{field} is not a non-empty tensor map")
    if coordinate_order is not None:
        if not coordinate_order or any(not isinstance(name, str) or not name for name in coordinate_order):
            raise Stage1EstimatorOracleError(f"{field} coordinate order is invalid")
        if len(set(coordinate_order)) != len(coordinate_order) or set(value) != set(coordinate_order):
            raise Stage1EstimatorOracleError(f"{field} coordinate identity is invalid")
        ordered_items = ((name, value[name]) for name in coordinate_order)
    else:
        ordered_items = value.items()
    tensors: dict[str, torch.Tensor] = {}
    for name, record in ordered_items:
        if not isinstance(name, str) or not name or not isinstance(record, Mapping):
            raise Stage1EstimatorOracleError(f"{field} has malformed coordinate")
        if set(record) != {"dtype", "shape", "values"}:
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] key set is invalid")
        dtype_name = record.get("dtype")
        shape = record.get("shape")
        entries = record.get("values")
        if dtype_name not in {"torch.float32", "torch.float64"} or not isinstance(shape, list) or not isinstance(entries, list):
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] wire type is invalid")
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in shape):
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] shape is invalid")
        expected = math.prod(shape)
        if len(entries) != expected:
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] value count is invalid")
        if any(type(item) not in {int, float} for item in entries):
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] value type is invalid")
        numeric = [float(item) for item in entries]
        if any(not math.isfinite(item) for item in numeric):
            raise Stage1EstimatorOracleError(f"{field}[{name!r}] has non-finite value")
        dtype = torch.float64 if dtype_name == "torch.float64" else torch.float32
        tensors[name] = torch.tensor(numeric, dtype=dtype).reshape(shape)
    return TensorMap(tensors, registry=registry)  # type: ignore[arg-type]


__all__ = [
    "Stage1EstimatorOracleError",
    "apply_learning_rates",
    "double_core",
    "mean_gradient",
    "ordered_pair_u_core",
    "raw_core",
    "sufficient_statistics",
    "tensor_map_from_wire",
    "tensor_map_to_wire",
    "unordered_pair_u_core",
    "weighted_ordered_pair_u_core",
]
