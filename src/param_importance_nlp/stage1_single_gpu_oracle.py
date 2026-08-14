"""Independent, array-only oracle for the S1.7 Pythia single-GPU gate.

This module deliberately does not import the production estimator, training,
provider, or registry implementations.  It consumes CPU tensors loaded from a
hash-verified safetensors file, so an error shared by the online tracker cannot
silently certify the fixed-state replay.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch


T32_ATOL = 1.0e-7
T32_RTOL = 1.0e-5
T32_NORMALIZED_L2_LIMIT = 1.0e-5


class Stage1S17OracleError(RuntimeError):
    """Raised when an S1.7 offline reconstruction is malformed."""


def _maps(samples: Sequence[Mapping[str, torch.Tensor]]) -> tuple[str, ...]:
    if len(samples) < 2:
        raise Stage1S17OracleError("S17_ORACLE_REQUIRES_AT_LEAST_TWO_MICROBATCHES")
    names = tuple(sorted(samples[0]))
    if not names or any(tuple(sorted(sample)) != names for sample in samples):
        raise Stage1S17OracleError("S17_ORACLE_PARAMETER_SET_DRIFT")
    reference_shapes = {name: tuple(samples[0][name].shape) for name in names}
    for sample in samples:
        for name in names:
            value = sample[name]
            if (
                value.dtype != torch.float32
                or value.device.type != "cpu"
                or tuple(value.shape) != reference_shapes[name]
                or not bool(torch.isfinite(value).all())
            ):
                raise Stage1S17OracleError(f"S17_ORACLE_TENSOR_INVALID:{name}")
    return names


def _compatible(reference: Sequence[Mapping[str, torch.Tensor]], candidate: Sequence[Mapping[str, torch.Tensor]]) -> tuple[str, ...]:
    names = _maps(reference)
    other = _maps(candidate)
    if names != other:
        raise Stage1S17OracleError("S17_ORACLE_DOUBLE_PARAMETER_SET_DRIFT")
    for left, right in zip(reference, candidate, strict=True):
        for name in names:
            if left[name].shape != right[name].shape:
                raise Stage1S17OracleError(f"S17_ORACLE_DOUBLE_SHAPE_DRIFT:{name}")
    return names


def reconstruct_mean(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Independent FP32 equal-weight mean reconstruction."""

    names = _maps(samples)
    count = len(samples)
    return {
        name: torch.stack([sample[name] for sample in samples], dim=0).mean(dim=0).to(torch.float32)
        for name in names
    }


def raw_double_and_u(
    samples_a: Sequence[Mapping[str, torch.Tensor]],
    samples_b: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Return raw, independent-double, explicit-U and streaming-U arrays.

    ``explicit_u`` intentionally loops over ordered pairs.  ``streaming_u``
    uses the algebraic sufficient-statistics form in FP64 before returning
    FP32; keeping both paths here catches a diagonal-term or denominator bug.
    """

    if len(samples_a) != 4 or len(samples_b) != 4:
        raise Stage1S17OracleError("S17_ORACLE_FIXED_FOUR_MICROBATCHES_REQUIRED")
    names = _compatible(samples_a, samples_b)
    mean_a = reconstruct_mean(samples_a)
    mean_b = reconstruct_mean(samples_b)
    explicit: dict[str, torch.Tensor] = {}
    streaming: dict[str, torch.Tensor] = {}
    raw: dict[str, torch.Tensor] = {}
    double: dict[str, torch.Tensor] = {}
    count = len(samples_a)
    denominator = count * (count - 1)
    if denominator <= 0:
        raise Stage1S17OracleError("S17_ORACLE_U_DENOMINATOR_INVALID")
    for name in names:
        ordered = torch.zeros_like(samples_a[0][name], dtype=torch.float64)
        s1 = torch.zeros_like(ordered)
        s2 = torch.zeros_like(ordered)
        for left_index, left in enumerate(samples_a):
            current = left[name].to(torch.float64)
            s1.add_(current)
            s2.add_(current.square())
            for right_index, right in enumerate(samples_a):
                if left_index != right_index:
                    ordered.add_(current * right[name].to(torch.float64))
        explicit[name] = (ordered / denominator).to(torch.float32)
        streaming[name] = ((s1.square() - s2) / denominator).to(torch.float32)
        raw[name] = mean_a[name].square()
        double[name] = mean_a[name] * mean_b[name]
    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "raw": raw,
        "double": double,
        "explicit_u": explicit,
        "streaming_u": streaming,
    }


def max_error(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, object]:
    """Return the frozen T32_SINGLE absolute/relative/NL2 comparison."""

    if set(left) != set(right) or not left:
        raise Stage1S17OracleError("S17_ORACLE_COMPARE_PARAMETER_SET_DRIFT")
    worst = -1.0; scaled_worst = -1.0; total_delta_sq = 0.0; total_reference_sq = 0.0; violations = 0; near_zero = 0
    worst_name = ""
    worst_index: list[int] = []
    per_tensor: dict[str, dict[str, object]] = {}
    for name in sorted(left):
        if (
            left[name].dtype != torch.float32
            or right[name].dtype != torch.float32
            or left[name].device.type != "cpu"
            or right[name].device.type != "cpu"
            or left[name].shape != right[name].shape
            or not bool(torch.isfinite(left[name]).all())
            or not bool(torch.isfinite(right[name]).all())
        ):
            raise Stage1S17OracleError(f"S17_ORACLE_COMPARE_TENSOR_INVALID:{name}")
        delta = (left[name].to(torch.float64) - right[name].to(torch.float64)).abs()
        reference = right[name].to(torch.float64).abs()
        threshold = T32_ATOL + T32_RTOL * reference
        violations += int((delta > threshold).sum().item())
        near_zero += int((reference <= T32_ATOL).sum().item())
        total_delta_sq += float(delta.square().sum().item())
        total_reference_sq += float(reference.square().sum().item())
        scaled_worst = max(scaled_worst, float((delta / threshold).max().item()))
        value, index = torch.max(delta.reshape(-1), dim=0)
        numeric = float(value.item())
        if not math.isfinite(numeric):
            raise Stage1S17OracleError("S17_ORACLE_NONFINITE_COMPARISON")
        if numeric > worst:
            worst = numeric
            worst_name = name
            worst_index = list(torch.unravel_index(index, delta.shape))
            worst_index = [int(item) for item in worst_index]
        tensor_delta_sq = float(delta.square().sum().item())
        tensor_reference_sq = float(reference.square().sum().item())
        per_tensor[name] = {
            "max_abs_error": numeric,
            "max_scaled_error": float((delta / threshold).max().item()),
            "normalized_l2_error": math.sqrt(tensor_delta_sq) / max(math.sqrt(tensor_reference_sq), T32_ATOL),
            "near_zero_coordinates": int((reference <= T32_ATOL).sum().item()),
            "violation_count": int((delta > threshold).sum().item()),
            "within_t32": bool((delta <= threshold).all().item())
            and math.sqrt(tensor_delta_sq) / max(math.sqrt(tensor_reference_sq), T32_ATOL) <= T32_NORMALIZED_L2_LIMIT,
        }
    normalized_l2 = math.sqrt(total_delta_sq) / max(math.sqrt(total_reference_sq), T32_ATOL)
    return {"max_abs_error": worst, "parameter": worst_name, "index": worst_index, "max_scaled_error": scaled_worst, "normalized_l2_error": normalized_l2, "near_zero_coordinates": near_zero, "violation_count": violations, "within_t32": violations == 0 and normalized_l2 <= T32_NORMALIZED_L2_LIMIT, "per_tensor": per_tensor}


__all__ = [
    "Stage1S17OracleError",
    "max_error",
    "raw_double_and_u",
    "reconstruct_mean",
]
