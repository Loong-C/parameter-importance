"""Import-isolated FP64 replay for the S1.8 DDP/accumulation gate.

This file intentionally imports neither the online worker nor the estimator or
training engine.  It receives CPU tensors verified against the safetensors
manifest and reconstructs ordered-pair U-statistics with explicit Python
loops.  A corrupted or jointly rehashed worker report therefore cannot turn an
incorrect no-sync accumulation into a formal PASS.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
from typing import Any

import torch


class Stage1S18OracleError(RuntimeError):
    """Raised for an incomplete or semantically inconsistent replay input."""


def _stable_l2_components(values: Sequence[torch.Tensor]) -> tuple[float, float]:
    """Return a scaled FP64 sum-of-squares, without ``x*x`` overflow.

    The comparator deliberately does not use ``sqrt(sum(square(x)))``: that
    expression turns finite 1e200 test values into ``inf`` and finite 1e-200
    values into zero.  This is the independent counterpart of the worker's
    clipping norm implementation.
    """

    scale, ssq = 0.0, 1.0
    for value in values:
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or not bool(torch.isfinite(value).all()):
            raise Stage1S18OracleError("S18_ORACLE_STABLE_NORM_TENSOR_INVALID")
        flat = value.to(torch.float64).reshape(-1)
        if flat.numel() == 0:
            continue
        block_scale = float(flat.abs().max().item())
        if block_scale == 0.0:
            continue
        block_ssq = float((flat / block_scale).square().sum().item())
        if scale == 0.0:
            scale, ssq = block_scale, block_ssq
        elif scale < block_scale:
            ssq = block_ssq + ssq * (scale / block_scale) ** 2
            scale = block_scale
        else:
            ssq += block_ssq * (block_scale / scale) ** 2
    return scale, ssq


def _stable_l2(values: Sequence[torch.Tensor]) -> float:
    scale, ssq = _stable_l2_components(values)
    return 0.0 if scale == 0.0 else scale * math.sqrt(ssq)


def _stable_ratio(numerator: Sequence[torch.Tensor], denominator: Sequence[torch.Tensor]) -> float:
    """Compute ||numerator||₂ / ||denominator||₂ from scaled components."""

    n_scale, n_ssq = _stable_l2_components(numerator)
    d_scale, d_ssq = _stable_l2_components(denominator)
    if n_scale == 0.0:
        return 0.0
    if d_scale == 0.0:
        return math.inf
    # The log form avoids a needless overflow/underflow in the intermediate
    # scale ratio.  It returns the mathematically correct finite ratio for
    # normal comparison inputs and inf only for a genuinely unrepresentable one.
    exponent = math.log(n_scale) - math.log(d_scale) + 0.5 * (math.log(n_ssq) - math.log(d_ssq))
    return math.exp(exponent) if exponent < math.log(float.fromhex("0x1.fffffffffffffp+1023")) else math.inf


def _maps(samples: Sequence[Mapping[str, torch.Tensor]]) -> tuple[str, ...]:
    if len(samples) != 8:
        raise Stage1S18OracleError("S18_ORACLE_REQUIRES_EXACTLY_EIGHT_MICROBATCHES")
    names = tuple(sorted(samples[0]))
    if not names or any(tuple(sorted(sample)) != names for sample in samples):
        raise Stage1S18OracleError("S18_ORACLE_PARAMETER_SET_DRIFT")
    shapes = {name: tuple(samples[0][name].shape) for name in names}
    for sample in samples:
        for name in names:
            value = sample[name]
            if (
                value.device.type != "cpu" or value.dtype != torch.float32
                or tuple(value.shape) != shapes[name] or not bool(torch.isfinite(value).all())
            ):
                raise Stage1S18OracleError(f"S18_ORACLE_MICRO_GRADIENT_INVALID:{name}")
    return names


def reconstruct_equal(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, dict[str, torch.Tensor]]:
    """Explicit FP64 ordered-pair reconstruction for the equal-weight case."""

    names = _maps(samples)
    output = {field: {} for field in ("s1", "s2", "mean", "raw", "u")}
    for name in names:
        reference = samples[0][name].to(torch.float64)
        s1, s2, ordered = torch.zeros_like(reference), torch.zeros_like(reference), torch.zeros_like(reference)
        for i, left in enumerate(samples):
            gradient = left[name].to(torch.float64)
            s1.add_(gradient); s2.add_(gradient.square())
            for j, right in enumerate(samples):
                if i != j:
                    ordered.add_(gradient * right[name].to(torch.float64))
        # Keep every reference tensor in FP64.  Casting here would let a
        # production FP32 rounding mistake certify itself in the comparator.
        output["s1"][name] = s1
        output["s2"][name] = s2
        output["mean"][name] = s1 / len(samples)
        output["raw"][name] = (s1 / len(samples)).square()
        output["u"][name] = ordered / (len(samples) * (len(samples) - 1))
    return output


def reconstruct_weighted(
    samples: Sequence[Mapping[str, torch.Tensor]], *, weights: Sequence[int]
) -> dict[str, dict[str, torch.Tensor]]:
    """Explicit FP64 weighted ordered-pair reconstruction; no production kernel."""

    names = _maps(samples)
    if len(weights) != len(samples) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in weights):
        raise Stage1S18OracleError("S18_ORACLE_WEIGHT_SET_INVALID")
    n1, n2 = sum(weights), sum(weight * weight for weight in weights)
    denominator = n1 * n1 - n2
    if denominator <= 0:
        raise Stage1S18OracleError("S18_ORACLE_WEIGHTED_DENOMINATOR_INVALID")
    output = {field: {} for field in ("g1", "g2", "mean", "raw", "u")}
    for name in names:
        reference = samples[0][name].to(torch.float64)
        g1, g2, ordered = torch.zeros_like(reference), torch.zeros_like(reference), torch.zeros_like(reference)
        for i, left in enumerate(samples):
            wi = weights[i]; gradient = left[name].to(torch.float64)
            g1.add_(gradient, alpha=float(wi)); g2.add_(gradient.square(), alpha=float(wi * wi))
            for j, right in enumerate(samples):
                if i != j:
                    ordered.add_(gradient * right[name].to(torch.float64), alpha=float(wi * weights[j]))
        output["g1"][name] = g1
        output["g2"][name] = g2
        output["mean"][name] = g1 / n1
        output["raw"][name] = (g1 / n1).square()
        output["u"][name] = ordered / denominator
    return output


def compare_maps(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], *, natural_scale: float,
    atol: float, rtol: float, normalized_l2_limit: float,
) -> dict[str, object]:
    """Apply the Stage 1 near-zero/T32_DISTRIBUTED comparison contract."""

    if (
        set(candidate) != set(reference) or not candidate or not math.isfinite(natural_scale)
        or natural_scale <= 0 or any(not math.isfinite(value) or value < 0 for value in (atol, rtol, normalized_l2_limit))
    ):
        raise Stage1S18OracleError("S18_ORACLE_COMPARE_CONTRACT_INVALID")
    threshold_zero, absolute_threshold = 10.0 * atol * natural_scale, atol * natural_scale
    all_delta: list[torch.Tensor] = []; all_value: list[torch.Tensor] = []; all_reference: list[torch.Tensor] = []; worst = -1.0
    worst_name = ""; worst_index: list[int] = []; violations = 0; per_tensor: dict[str, object] = {}
    near_zero_objects = 0
    for name in sorted(candidate):
        left, right = candidate[name], reference[name]
        if (
            not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.device.type != "cpu" or right.device.type != "cpu" or left.dtype != torch.float32
            or right.dtype != torch.float64 or left.shape != right.shape
            or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all())
        ):
            raise Stage1S18OracleError(f"S18_ORACLE_COMPARE_TENSOR_INVALID:{name}")
        lhs, rhs = left.to(torch.float64), right.to(torch.float64)
        delta = (lhs - rhs).abs(); reference_max = float(rhs.abs().max().item())
        near_zero = reference_max <= threshold_zero
        if near_zero:
            passed = bool((delta <= absolute_threshold).all().item())
            normalized_l2: float | None = None
            scaled_max = float(delta.max().item()) / absolute_threshold
            near_zero_objects += 1
        else:
            scale = reference_max
            lhs_scaled, rhs_scaled = lhs / scale, rhs / scale
            shape_scale = max(float(lhs_scaled.abs().max().item()), float(rhs_scaled.abs().max().item()))
            scaled_delta = (lhs_scaled - rhs_scaled).abs()
            scaled_max = float(scaled_delta.max().item())
            tensor_norm = max(_stable_l2((lhs_scaled,)), _stable_l2((rhs_scaled,)), 1.0e-300)
            normalized_l2 = _stable_l2((scaled_delta,)) / tensor_norm
            passed = bool((scaled_delta <= atol + rtol * shape_scale).all().item()) and normalized_l2 <= normalized_l2_limit
        maximum, flat_index = torch.max(delta.reshape(-1), dim=0)
        numeric = float(maximum.item())
        if numeric > worst:
            worst, worst_name = numeric, name
            worst_index = [int(item) for item in torch.unravel_index(flat_index, delta.shape)]
        all_delta.append(lhs - rhs)
        all_value.append(lhs)
        all_reference.append(rhs)
        violations += 0 if passed else 1
        per_tensor[name] = {
            "near_zero_branch": near_zero,
            "natural_scale": natural_scale,
            "near_zero_threshold": threshold_zero,
            "absolute_threshold": absolute_threshold,
            "max_abs_error": numeric,
            "max_scaled_error": scaled_max,
            "normalized_l2_error": normalized_l2,
            "candidate_dtype": str(left.dtype),
            "reference_dtype": str(right.dtype),
            "within_t32_distributed": passed,
        }
    denominator = max(_stable_l2(all_value), _stable_l2(all_reference), 1.0e-300)
    normalized_l2_all = _stable_l2(all_delta) / denominator
    return {
        "max_abs_error": worst,
        "worst_parameter": worst_name,
        "worst_index": worst_index,
        "normalized_l2_error": normalized_l2_all,
        "near_zero_object_count": near_zero_objects,
        "violation_count": violations,
        "within_t32_distributed": violations == 0 and normalized_l2_all <= normalized_l2_limit,
        "per_tensor": per_tensor,
    }


def compare_peer_maps(
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], *, natural_scale: float,
    atol: float, rtol: float, normalized_l2_limit: float,
) -> dict[str, object]:
    """Compare two persisted FP32 route outputs without mislabelling it oracle.

    Peer comparisons (notably A↔D optimizer state) are still evaluated in FP64,
    but their reference is explicitly tagged as a production route rather than
    being allowed to masquerade as an independent FP64 reconstruction.
    """

    if any(value.dtype != torch.float32 or value.device.type != "cpu" for value in reference.values()):
        raise Stage1S18OracleError("S18_ORACLE_PEER_REFERENCE_INVALID")
    result = compare_maps(
        candidate,
        {name: value.to(torch.float64) for name, value in reference.items()},
        natural_scale=natural_scale, atol=atol, rtol=rtol, normalized_l2_limit=normalized_l2_limit,
    )
    result["reference_kind"] = "production_peer_fp32_promoted_to_fp64_for_metric_only"
    return result


def collect_route_phase(
    arrays: Mapping[str, torch.Tensor], *, prefix: str
) -> dict[str, torch.Tensor]:
    """Extract a complete named parameter map from a slash-delimited phase."""

    expected_prefix = prefix.rstrip("/") + "/"
    result = {key[len(expected_prefix):]: value for key, value in arrays.items() if key.startswith(expected_prefix)}
    if not result or any("/" in name for name in result):
        raise Stage1S18OracleError(f"S18_ORACLE_PHASE_INVALID:{prefix}")
    return result


def _phase_reference(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Promote a verified production FP32 input for independent FP64 algebra."""

    if not values or any(value.dtype != torch.float32 or value.device.type != "cpu" for value in values.values()):
        raise Stage1S18OracleError("S18_ORACLE_PRODUCTION_INPUT_INVALID")
    return {name: value.to(torch.float64) for name, value in values.items()}


def _tensor_map_digest(values: Mapping[str, torch.Tensor]) -> str:
    """Worker-compatible checksum implemented locally, without worker import."""

    if not values:
        raise Stage1S18OracleError("S18_ORACLE_CHECKSUM_EMPTY")
    digest = hashlib.sha256(b"stage1-s1-8-tensor-map-v1\0")
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
            raise Stage1S18OracleError("S18_ORACLE_CHECKSUM_TENSOR_INVALID")
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(b"\0")
        digest.update(value.detach().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def _case_row(report: Mapping[str, Any], *, case: str) -> Mapping[str, Any]:
    rows = report.get("cases")
    matches = [item for item in rows if isinstance(item, Mapping) and item.get("case") == case] if isinstance(rows, list) else []
    if len(matches) != 1:
        raise Stage1S18OracleError(f"S18_ORACLE_CASE_REPORT_MISSING_OR_DUPLICATE:{case}")
    return matches[0]


def _scalar_check(expected: float, observed: object, *, precision: Mapping[str, Any]) -> dict[str, object]:
    if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isfinite(float(observed)):
        raise Stage1S18OracleError("S18_ORACLE_SCALAR_INVALID")
    error = abs(float(observed) - expected)
    limit = float(precision["atol"]) + float(precision["rtol"]) * max(1.0, abs(expected))
    return {
        "expected_fp64": expected, "observed": float(observed), "absolute_error": error,
        "within_t32_distributed": error <= limit,
    }


def _exact_scalar_check(expected: int, observed: object) -> dict[str, object]:
    """Record a discrete contract without giving it a floating tolerance."""

    passed = not isinstance(observed, bool) and isinstance(observed, int) and observed == expected
    return {"expected": expected, "observed": observed, "exact_equality": True, "within_t32_distributed": passed}


def _exact_step_check(candidate: Mapping[str, torch.Tensor], *, expected_step: int) -> dict[str, object]:
    """AdamW's scalar step is a discrete state value, not a numeric estimate."""

    if not candidate:
        raise Stage1S18OracleError("S18_ORACLE_ADAMW_STEP_STATE_EMPTY")
    per_tensor: dict[str, object] = {}
    for name, value in candidate.items():
        passed = (
            value.dtype == torch.float32 and value.device.type == "cpu" and value.numel() == 1
            and bool(torch.isfinite(value).all()) and float(value.item()) == float(expected_step)
        )
        per_tensor[name] = {"expected_step": expected_step, "observed": float(value.item()) if value.numel() == 1 else None, "exact_equality": True, "within_t32_distributed": passed}
    return {"max_abs_error": 0.0 if all(bool(item["within_t32_distributed"]) for item in per_tensor.values()) else math.inf, "worst_parameter": "", "worst_index": [], "normalized_l2_error": None, "near_zero_object_count": 0, "violation_count": sum(not bool(item["within_t32_distributed"]) for item in per_tensor.values()), "within_t32_distributed": all(bool(item["within_t32_distributed"]) for item in per_tensor.values()), "per_tensor": per_tensor}


def _split_optimizer_state(values: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    exp_avg: dict[str, torch.Tensor] = {}; exp_avg_sq: dict[str, torch.Tensor] = {}; step: dict[str, torch.Tensor] = {}
    for key, value in values.items():
        if "::" not in key:
            raise Stage1S18OracleError("S18_ORACLE_ADAMW_STATE_KEY_INVALID")
        name, field = key.rsplit("::", 1)
        if not name:
            raise Stage1S18OracleError("S18_ORACLE_ADAMW_STATE_KEY_INVALID")
        target = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step}.get(field)
        if target is None or name in target:
            raise Stage1S18OracleError("S18_ORACLE_ADAMW_STATE_FIELD_INVALID")
        target[name] = value
    if set(exp_avg) != set(exp_avg_sq) or set(exp_avg) != set(step) or not exp_avg:
        raise Stage1S18OracleError("S18_ORACLE_ADAMW_STATE_COVERAGE_INVALID")
    return exp_avg, exp_avg_sq, step


def _validate_rank_contract(*, route: str, report: Mapping[str, Any], row: Mapping[str, Any], case: str, precision: Mapping[str, Any]) -> dict[str, object]:
    """Verify real rank ownership and no-sync observability per formal case."""

    rank_records = row.get("rank_records")
    layout = report.get("route_layout")
    if not isinstance(rank_records, list) or not isinstance(layout, Mapping):
        raise Stage1S18OracleError(f"S18_ORACLE_RANK_RECORDS_MISSING:{case}:{route}")
    expected = layout.get("rank_microbatch_ids")
    if not isinstance(expected, list) or len(rank_records) != len(expected):
        raise Stage1S18OracleError(f"S18_ORACLE_RANK_RECORD_COUNT_INVALID:{case}:{route}")
    seen: list[int] = []; checksums: list[object] = []; effective_total = 0; local_losses: list[float] = []
    for rank, record in enumerate(rank_records):
        if not isinstance(record, Mapping) or record.get("rank") != rank or record.get("local_microbatch_ids") != expected[rank]:
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_PARTITION_DRIFT:{case}:{route}:{rank}")
        ids = record["local_microbatch_ids"]
        if not isinstance(ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in ids):
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_IDS_INVALID:{case}:{route}:{rank}")
        seen.extend(ids)
        checksums.append(record.get("global_statistic_checksums"))
        tokens = record.get("local_effective_tokens")
        loss = record.get("local_loss_numerator")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0 or isinstance(loss, bool) or not isinstance(loss, (int, float)) or not math.isfinite(float(loss)):
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_LOSS_OR_COUNT_INVALID:{case}:{route}:{rank}")
        gradients = record.get("local_gradient_checksums")
        if not isinstance(gradients, list) or len(gradients) != len(ids) or any(not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in gradients):
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_GRADIENT_CHECKSUM_INVALID:{case}:{route}:{rank}")
        effective_total += tokens
        local_losses.append(float(loss))
    if effective_total != row.get("global_loss_valid_token_count"):
        raise Stage1S18OracleError(f"S18_ORACLE_RANK_TOKEN_TOTAL_DRIFT:{case}:{route}")
    loss_check = _scalar_check(math.fsum(local_losses), row.get("global_loss_numerator"), precision=precision)
    if not bool(loss_check["within_t32_distributed"]):
        raise Stage1S18OracleError(f"S18_ORACLE_RANK_LOSS_TOTAL_DRIFT:{case}:{route}")
    if route != "A":
        if sorted(seen) != list(range(8)) or len(seen) != len(set(seen)):
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_UNIT_COVERAGE_INVALID:{case}:{route}")
        if not checksums or any(item != checksums[0] or item != row.get("global_statistic_checksums") for item in checksums):
            raise Stage1S18OracleError(f"S18_ORACLE_RANK_STATISTIC_CHECKSUM_DRIFT:{case}:{route}")
        if row.get("ordinary_ddp_gradient_collectives") != 0:
            raise Stage1S18OracleError(f"S18_ORACLE_NO_SYNC_COLLECTIVE_DRIFT:{case}:{route}")
    return loss_check


def _adamw_step(
    *, pre: Mapping[str, torch.Tensor], mean: Mapping[str, torch.Tensor], clip_factor: float,
    previous_exp_avg: Mapping[str, torch.Tensor], previous_exp_avg_sq: Mapping[str, torch.Tensor],
    previous_step: int, optimizer: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], int]:
    """Independent FP64 AdamW recurrence from persisted pre-step arrays."""

    if set(pre) != set(mean) or set(pre) != set(previous_exp_avg) or set(pre) != set(previous_exp_avg_sq):
        raise Stage1S18OracleError("S18_ORACLE_ADAMW_PARAMETER_SET_DRIFT")
    beta1, beta2 = (float(value) for value in optimizer["betas"])
    lr, wd, eps = (float(optimizer[key]) for key in ("learning_rate", "weight_decay", "eps"))
    step = previous_step + 1
    output: dict[str, torch.Tensor] = {}; exp_avg: dict[str, torch.Tensor] = {}; exp_avg_sq: dict[str, torch.Tensor] = {}
    for name in sorted(pre):
        gradient = mean[name] * clip_factor
        exp_avg[name] = previous_exp_avg[name] * beta1 + gradient * (1.0 - beta1)
        exp_avg_sq[name] = previous_exp_avg_sq[name] * beta2 + gradient.square() * (1.0 - beta2)
        corrected_avg = exp_avg[name] / (1.0 - beta1 ** step)
        corrected_sq = exp_avg_sq[name] / (1.0 - beta2 ** step)
        output[name] = pre[name] * (1.0 - lr * wd) - lr * corrected_avg / (corrected_sq.sqrt() + eps)
    return output, exp_avg, exp_avg_sq, step


def _add_check(
    checks: dict[str, object], rows: list[dict[str, object]], *, key: str,
    candidate: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], natural_scale: float,
    precision: Mapping[str, Any],
) -> None:
    check = compare_maps(
        candidate, reference, natural_scale=natural_scale, atol=float(precision["atol"]),
        rtol=float(precision["rtol"]), normalized_l2_limit=float(precision["normalized_l2_limit"]),
    )
    checks[key] = check
    for name, detail in dict(check["per_tensor"]).items():
        rows.append({"comparison": key, "parameter": name, **dict(detail)})


def _expected_manual_contract(*, case: str, parameter_count: int) -> dict[str, object]:
    if case == "equal":
        scalar = ["M", "loss_numerator", "loss_valid_token_count"]
        tensor = ["S1", "S2"]
    else:
        scalar = ["N1", "N2", "loss_numerator", "loss_valid_token_count"]
        tensor = ["G1", "G2"]
    return {
        "backend": "nccl", "operation": "SUM", "tensor_statistics": tensor,
        "tensor_all_reduce_count": 2 * parameter_count, "scalar_statistics": scalar,
        "scalar_all_reduce_count": len(scalar), "total_all_reduce_count": 2 * parameter_count + len(scalar),
    }


def replay(
    *, route_arrays: Mapping[str, Mapping[str, torch.Tensor]], fixture: Mapping[str, Any],
    route_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, object]:
    """FP64 array-only replay of statistics, AdamW, and two-step accumulation.

    The worker tensors remain FP32 production observations.  Every independent
    statistic, clipping value, public score, optimizer-state recurrence, and
    cumulative accumulator view below remains CPU FP64 until comparison.
    """

    routes = ("A", "B", "C", "D")
    if set(route_arrays) != set(routes) or set(route_reports) != set(routes):
        raise Stage1S18OracleError("S18_ORACLE_ROUTE_SET_INVALID")
    cases = fixture.get("cases"); precision = fixture.get("precision"); scales = fixture.get("comparison_natural_scales")
    optimizer = fixture.get("optimizer")
    if not all(isinstance(item, Mapping) for item in (cases, precision, scales, optimizer)):
        raise Stage1S18OracleError("S18_ORACLE_FIXTURE_CONTRACT_INVALID")
    required_scales = {"n_g", "s1", "s2", "g1", "g2", "mean_gradient", "core", "score", "optimizer_delta", "parameter"}
    if set(scales) != required_scales:
        raise Stage1S18OracleError("S18_ORACLE_NATURAL_SCALE_SET_INVALID")
    if any("/u_" in key for key in route_arrays["A"]):
        raise Stage1S18OracleError("S18_ORACLE_A_MUST_NOT_BE_U_REFERENCE")
    if any(key.startswith("accumulator/") for key in route_arrays["A"]):
        raise Stage1S18OracleError("S18_ORACLE_A_MUST_NOT_HAVE_ACCUMULATOR")

    checks: dict[str, object] = {}; scalar_checks: dict[str, object] = {}; rows: list[dict[str, object]] = []
    lr, wd = float(optimizer["learning_rate"]), float(optimizer["weight_decay"])
    max_norm = float(fixture["gradient_clip_max_norm"])
    previous: dict[str, dict[str, object]] = {}
    a_previous: dict[str, object] = {}
    cumulative: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for route in ("B", "C", "D"):
        pre_equal = _phase_reference(collect_route_phase(route_arrays[route], prefix="pre/equal"))
        zero = {name: torch.zeros_like(value) for name, value in pre_equal.items()}
        previous[route] = {"exp_avg": zero.copy(), "exp_avg_sq": zero.copy(), "step": 0}
        cumulative[route] = {
            "positive": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "negative_mass": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "raw": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "raw_clipped": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "data_movement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "data_displacement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "total_movement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "total_displacement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "weight_decay_movement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "weight_decay_displacement": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "actual_update_raw_importance": {name: torch.zeros_like(value) for name, value in pre_equal.items()},
            "magnitude": {name: value.abs() for name, value in pre_equal.items()},
        }

    for case in ("equal", "weighted"):
        micro = [collect_route_phase(route_arrays["B"], prefix=f"micro/{case}/{index:02d}") for index in range(8)]
        reference = reconstruct_equal(micro) if case == "equal" else reconstruct_weighted(
            micro, weights=list(cases[case]["effective_target_tokens"]),
        )
        stat_fields = ("s1", "s2") if case == "equal" else ("g1", "g2")
        mean, raw_core, u_core = reference["mean"], reference["raw"], reference["u"]
        mean_norm = _stable_l2(tuple(mean.values()))
        oracle_clip = min(1.0, max_norm / (mean_norm + 1.0e-6))
        if not 0.0 < oracle_clip <= 1.0:
            raise Stage1S18OracleError("S18_ORACLE_CLIP_FACTOR_INVALID")
        score_views = {
            "mean_gradient": mean, "raw_core": raw_core,
            "raw_core_clipped_diagnostic": {name: value * oracle_clip for name, value in raw_core.items()},
            "u_core": u_core, "u_core_clipped": {name: value * oracle_clip for name, value in u_core.items()},
            "raw_score": {name: value * lr for name, value in raw_core.items()},
            "raw_score_clipped": {name: value * lr * oracle_clip for name, value in raw_core.items()},
            "u_score": {name: value * lr for name, value in u_core.items()},
            "u_score_clipped": {name: value * lr * oracle_clip for name, value in u_core.items()},
        }
        a_score_views = {key: value for key, value in score_views.items() if not key.startswith("u_")}
        loss_rows: dict[str, Mapping[str, Any]] = {}
        for route in routes:
            row = _case_row(route_reports[route], case=case)
            loss_rows[route] = row
            scalar_checks[f"{case}:{route}:rank_local_loss_sum"] = _validate_rank_contract(route=route, report=route_reports[route], row=row, case=case, precision=precision)
            scalar_checks[f"{case}:{route}:clip_factor"] = _scalar_check(oracle_clip, row.get("clip_factor"), precision=precision)
            expected_m = 8
            weights = list(cases[case]["effective_target_tokens"])
            expected_n1, expected_n2 = sum(weights), sum(int(value) ** 2 for value in weights)
            for key, expected in (("global_microbatch_count", expected_m), ("global_n1", expected_n1), ("global_n2", expected_n2), ("global_loss_valid_token_count", expected_n1)):
                scalar_checks[f"{case}:{route}:{key}"] = _exact_scalar_check(expected, row.get(key))
            numerator, count, global_mean = row.get("global_loss_numerator"), row.get("global_loss_valid_token_count"), row.get("global_mean_loss")
            if isinstance(numerator, bool) or not isinstance(numerator, (int, float)) or not isinstance(count, int) or count <= 0:
                raise Stage1S18OracleError("S18_ORACLE_LOSS_REPORT_INVALID")
            scalar_checks[f"{case}:{route}:global_mean_loss_once"] = _scalar_check(float(numerator) / count, global_mean, precision=precision)
            if route == "A":
                expected_contract = {"backend": "nccl", "operation": "SUM", "tensor_statistics": [], "tensor_all_reduce_count": 0, "scalar_statistics": [], "scalar_all_reduce_count": 0, "total_all_reduce_count": 0}
            else:
                expected_contract = _expected_manual_contract(case=case, parameter_count=len(mean))
            if row.get("manual_statistic_collectives") != expected_contract:
                raise Stage1S18OracleError(f"S18_ORACLE_MANUAL_COLLECTIVE_CONTRACT_DRIFT:{case}:{route}")
        reference_loss = loss_rows["A"]
        for route in ("B", "C", "D"):
            observed = loss_rows[route]
            scalar_checks[f"{case}:A:{route}:global_loss_numerator_peer"] = _scalar_check(float(reference_loss["global_loss_numerator"]), observed.get("global_loss_numerator"), precision=precision)
            scalar_checks[f"{case}:A:{route}:global_mean_loss_peer"] = _scalar_check(float(reference_loss["global_mean_loss"]), observed.get("global_mean_loss"), precision=precision)
        for route in ("B", "C", "D"):
            for field in stat_fields:
                _add_check(checks, rows, key=f"{case}:{route}:stats:{field}", candidate=collect_route_phase(route_arrays[route], prefix=f"stats/{case}/{field}"), reference=reference[field], natural_scale=float(scales[field]), precision=precision)
            for field, expected in score_views.items():
                scale_key = "mean_gradient" if field == "mean_gradient" else "core" if field.startswith(("raw_core", "u_core")) else "score"
                _add_check(checks, rows, key=f"{case}:{route}:scores:{field}", candidate=collect_route_phase(route_arrays[route], prefix=f"scores/{case}/{field}"), reference=expected, natural_scale=float(scales[scale_key]), precision=precision)
        for field, expected in a_score_views.items():
            scale_key = "mean_gradient" if field == "mean_gradient" else "core" if field.startswith("raw_core") else "score"
            _add_check(checks, rows, key=f"{case}:A:a_reference:{field}", candidate=collect_route_phase(route_arrays["A"], prefix=f"a-reference/{case}/{field}"), reference=expected, natural_scale=float(scales[scale_key]), precision=precision)

        # Independently replay the optimizer, state tensors and accumulator for
        # every actual route.  A participates in update parity but never in U
        # accumulation; B/C/D continue their accumulator from equal to weighted.
        a_pre = collect_route_phase(route_arrays["A"], prefix=f"pre/{case}")
        for route in routes:
            row = _case_row(route_reports[route], case=case)
            pre_live = collect_route_phase(route_arrays[route], prefix=f"pre/{case}")
            post_live = collect_route_phase(route_arrays[route], prefix=f"post/{case}")
            if row.get("pre_parameter_checksum") != _tensor_map_digest(pre_live) or row.get("post_parameter_checksum") != _tensor_map_digest(post_live):
                raise Stage1S18OracleError(f"S18_ORACLE_PARAMETER_CHECKSUM_DRIFT:{case}:{route}")
            case_keys = {
                key for key in route_arrays[route]
                if key.startswith((f"micro/{case}/", f"a-reference/{case}/", f"stats/{case}/", f"scores/{case}/", f"accumulator/{case}/", f"pre/{case}/", f"post/{case}/", f"optimizer-state/{case}/", f"optimizer-state-pre/{case}/"))
            }
            if row.get("array_keys") != sorted(case_keys):
                raise Stage1S18OracleError(f"S18_ORACLE_ARRAY_KEY_REPORT_DRIFT:{case}:{route}")
            if route != "A":
                expected_checksums = {
                    field: _tensor_map_digest(collect_route_phase(route_arrays[route], prefix=f"stats/{case}/{field}"))
                    for field in stat_fields
                }
                if row.get("global_statistic_checksums") != expected_checksums:
                    raise Stage1S18OracleError(f"S18_ORACLE_GLOBAL_STATISTIC_CHECKSUM_DRIFT:{case}:{route}")
            if case == "weighted":
                continuity = compare_peer_maps(
                    pre_live, collect_route_phase(route_arrays[route], prefix="post/equal"),
                    natural_scale=float(scales["parameter"]), atol=float(precision["atol"]),
                    rtol=float(precision["rtol"]), normalized_l2_limit=float(precision["normalized_l2_limit"]),
                )
                checks[f"{route}:equal_to_weighted:parameter_continuity"] = continuity
                previous_avg, previous_sq, previous_step = _split_optimizer_state(collect_route_phase(route_arrays[route], prefix="optimizer-state/equal"))
                weighted_avg, weighted_sq, weighted_step = _split_optimizer_state(collect_route_phase(route_arrays[route], prefix="optimizer-state-pre/weighted"))
                state_continuity = compare_peer_maps(
                    weighted_avg, previous_avg, natural_scale=float(scales["n_g"]),
                    atol=float(precision["atol"]), rtol=float(precision["rtol"]),
                    normalized_l2_limit=float(precision["normalized_l2_limit"]),
                )
                checks[f"{route}:equal_to_weighted:optimizer_exp_avg_continuity"] = state_continuity
                checks[f"{route}:equal_to_weighted:optimizer_exp_avg_sq_continuity"] = compare_peer_maps(
                    weighted_sq, previous_sq, natural_scale=float(scales["core"]), atol=float(precision["atol"]),
                    rtol=float(precision["rtol"]), normalized_l2_limit=float(precision["normalized_l2_limit"]),
                )
                checks[f"{route}:equal_to_weighted:optimizer_step_continuity"] = _exact_step_check(weighted_step, expected_step=1)
            pre_fp64 = _phase_reference(collect_route_phase(route_arrays[route], prefix=f"pre/{case}"))
            if route != "A":
                _add_check(checks, rows, key=f"{case}:A:{route}:pre_parameters", candidate=collect_route_phase(route_arrays[route], prefix=f"pre/{case}"), reference=_phase_reference(a_pre), natural_scale=float(scales["parameter"]), precision=precision)
                state = previous[route]
                expected_post, exp_avg, exp_avg_sq, step = _adamw_step(
                    pre=pre_fp64, mean=mean, clip_factor=oracle_clip,
                    previous_exp_avg=state["exp_avg"], previous_exp_avg_sq=state["exp_avg_sq"],
                    previous_step=int(state["step"]), optimizer=optimizer,
                )
                _add_check(checks, rows, key=f"{case}:{route}:post_parameters", candidate=collect_route_phase(route_arrays[route], prefix=f"post/{case}"), reference=expected_post, natural_scale=float(scales["parameter"]), precision=precision)
                observed_avg, observed_avg_sq, observed_step = _split_optimizer_state(collect_route_phase(route_arrays[route], prefix=f"optimizer-state/{case}"))
                _add_check(checks, rows, key=f"{case}:{route}:optimizer_state:exp_avg", candidate=observed_avg, reference=exp_avg, natural_scale=float(scales["n_g"]), precision=precision)
                _add_check(checks, rows, key=f"{case}:{route}:optimizer_state:exp_avg_sq", candidate=observed_avg_sq, reference=exp_avg_sq, natural_scale=float(scales["core"]), precision=precision)
                step_check = _exact_step_check(observed_step, expected_step=step)
                checks[f"{case}:{route}:optimizer_state:step"] = step_check
                for name, detail in dict(step_check["per_tensor"]).items():
                    rows.append({"comparison": f"{case}:{route}:optimizer_state:step", "parameter": name, **dict(detail)})
                previous[route] = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step}
                total_update = {name: expected_post[name] - pre_fp64[name] for name in pre_fp64}
                decay_update = {name: -lr * wd * pre_fp64[name] for name in pre_fp64}
                data_update = {name: total_update[name] - decay_update[name] for name in pre_fp64}
                contribution = {
                    "signed": score_views["u_score_clipped"], "raw": score_views["raw_score"],
                    "raw_clipped": score_views["raw_score_clipped"], "data_update": data_update,
                    "total_update": total_update, "weight_decay_update": decay_update,
                    "actual_update_raw_importance": {name: -(data_update[name] * mean[name]) for name in mean},
                }
                for field, expected in contribution.items():
                    scale = float(scales["score"] if field in {"signed", "raw", "raw_clipped", "actual_update_raw_importance"} else scales["optimizer_delta"])
                    _add_check(checks, rows, key=f"{case}:{route}:accumulator:contribution:{field}", candidate=collect_route_phase(route_arrays[route], prefix=f"accumulator/{case}/contribution/{field}"), reference=expected, natural_scale=scale, precision=precision)
                state_c = cumulative[route]
                for name in mean:
                    signed = contribution["signed"][name]
                    state_c["positive"][name] += signed.clamp_min(0)
                    state_c["negative_mass"][name] += (-signed).clamp_min(0)
                    state_c["raw"][name] += contribution["raw"][name]
                    state_c["raw_clipped"][name] += contribution["raw_clipped"][name]
                    for destination, source in (("data_movement", "data_update"), ("data_displacement", "data_update"), ("total_movement", "total_update"), ("total_displacement", "total_update"), ("weight_decay_movement", "weight_decay_update"), ("weight_decay_displacement", "weight_decay_update"), ("actual_update_raw_importance", "actual_update_raw_importance")):
                        state_c[destination][name] += contribution[source][name].abs() if destination.endswith("movement") else contribution[source][name]
                    state_c["magnitude"][name] = expected_post[name].abs()
                cumulative_views = {
                    "signed": {name: state_c["positive"][name] - state_c["negative_mass"][name] for name in mean},
                    "positive": state_c["positive"], "negative_mass": state_c["negative_mass"],
                    "absolute": {name: state_c["positive"][name] + state_c["negative_mass"][name] for name in mean},
                    "raw": state_c["raw"], "raw_clipped": state_c["raw_clipped"],
                    "data_movement": state_c["data_movement"],
                    "net_data_movement": {name: state_c["data_displacement"][name].abs() for name in mean},
                    "total_movement": state_c["total_movement"],
                    "total_endpoint_movement": {name: state_c["total_displacement"][name].abs() for name in mean},
                    "weight_decay_movement": state_c["weight_decay_movement"],
                    "net_weight_decay_movement": {name: state_c["weight_decay_displacement"][name].abs() for name in mean},
                    "actual_update_raw_importance": state_c["actual_update_raw_importance"], "magnitude": state_c["magnitude"],
                }
                for field, expected in cumulative_views.items():
                    scale = float(scales["score"] if field in {"signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped", "actual_update_raw_importance"} else scales["optimizer_delta"] if "movement" in field else scales["parameter"])
                    _add_check(checks, rows, key=f"{case}:{route}:accumulator:cumulative:{field}", candidate=collect_route_phase(route_arrays[route], prefix=f"accumulator/{case}/cumulative/{field}"), reference=expected, natural_scale=scale, precision=precision)
                report_acc = _case_row(route_reports[route], case=case).get("accumulator")
                if not isinstance(report_acc, Mapping) or report_acc.get("successful_steps") != (1 if case == "equal" else 2) or report_acc.get("skipped_steps") != 0:
                    raise Stage1S18OracleError(f"S18_ORACLE_ACCUMULATOR_COUNT_DRIFT:{case}:{route}")
            else:
                # A's post parameter and AdamW state must obey the same
                # independent recurrence, but no U accumulator is permitted.
                zero = {name: torch.zeros_like(value) for name, value in pre_fp64.items()} if case == "equal" else a_previous["exp_avg"]
                zero_sq = {name: torch.zeros_like(value) for name, value in pre_fp64.items()} if case == "equal" else a_previous["exp_avg_sq"]
                old_step = 0 if case == "equal" else int(a_previous["step"])
                expected_post, exp_avg, exp_avg_sq, step = _adamw_step(pre=pre_fp64, mean=mean, clip_factor=oracle_clip, previous_exp_avg=zero, previous_exp_avg_sq=zero_sq, previous_step=old_step, optimizer=optimizer)
                _add_check(checks, rows, key=f"{case}:A:post_parameters", candidate=collect_route_phase(route_arrays["A"], prefix=f"post/{case}"), reference=expected_post, natural_scale=float(scales["parameter"]), precision=precision)
                observed_avg, observed_avg_sq, observed_step = _split_optimizer_state(collect_route_phase(route_arrays["A"], prefix=f"optimizer-state/{case}"))
                _add_check(checks, rows, key=f"{case}:A:optimizer_state:exp_avg", candidate=observed_avg, reference=exp_avg, natural_scale=float(scales["n_g"]), precision=precision)
                _add_check(checks, rows, key=f"{case}:A:optimizer_state:exp_avg_sq", candidate=observed_avg_sq, reference=exp_avg_sq, natural_scale=float(scales["core"]), precision=precision)
                step_check = _exact_step_check(observed_step, expected_step=step)
                checks[f"{case}:A:optimizer_state:step"] = step_check
                for name, detail in dict(step_check["per_tensor"]).items():
                    rows.append({"comparison": f"{case}:A:optimizer_state:step", "parameter": name, **dict(detail)})
                total_update = {name: expected_post[name] - pre_fp64[name] for name in pre_fp64}
                decay_update = {name: -lr * wd * pre_fp64[name] for name in pre_fp64}
                data_update = {name: total_update[name] - decay_update[name] for name in pre_fp64}
                a_update_views = {
                    "data_update": data_update, "data_movement": {name: value.abs() for name, value in data_update.items()},
                    "total_update": total_update, "weight_decay_update": decay_update,
                    "actual_update_raw_importance": {name: -(data_update[name] * mean[name]) for name in mean},
                    "magnitude": {name: value.abs() for name, value in expected_post.items()},
                }
                for field, expected in a_update_views.items():
                    scale = float(scales["score"] if field == "actual_update_raw_importance" else scales["optimizer_delta"] if field != "magnitude" else scales["parameter"])
                    _add_check(checks, rows, key=f"{case}:A:a_reference:{field}", candidate=collect_route_phase(route_arrays["A"], prefix=f"a-reference/{case}/{field}"), reference=expected, natural_scale=scale, precision=precision)
                a_previous = {"exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq, "step": step}

    passed = all(bool(value["within_t32_distributed"]) for value in checks.values()) and all(bool(value["within_t32_distributed"]) for value in scalar_checks.values())
    return {
        "schema_version": "stage1-s1-8-replay-validation-v1", "status": "PASS" if passed else "FAIL",
        "oracle_import_isolated": True, "oracle_reference_dtype": "torch.float64",
        "production_candidate_dtype": "torch.float32", "checks": checks, "scalar_checks": scalar_checks,
        "comparison_rows": rows,
    }


__all__ = [
    "Stage1S18OracleError", "collect_route_phase", "compare_maps", "compare_peer_maps", "reconstruct_equal",
    "reconstruct_weighted", "replay",
]
