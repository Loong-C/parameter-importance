"""S1.9 precision, clipping, GradScaler, and optimizer-boundary checks.

The production calculations in this module intentionally exercise the public
runtime/core APIs.  :mod:`stage1_precision_oracle` is a separate FP64 Python
loop reference.  Neither local CPU validation nor a caller-provided BF16
observation may be promoted to ``G1-NUMERIC=PASS`` without the formal GPU
publisher and its immutable S1.7 handoff check.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
import random
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import torch
import numpy as np

from .contracts.jsonio import canonical_json_hash, load_canonical_json
from .core.accumulator import ImportanceAccumulator
from .core.estimators import equal_u_importance, raw_importance
from .core.sufficient_statistics import EqualSufficientStatistics, WeightedSufficientStatistics
from .core.tensors import TensorMap
from .runtime.gradients import GradientAttempt, GradientPhase
from .runtime.checkpoint import CheckpointStore
from .runtime.optimizer import OptimizerBridge, compute_global_clip_factor
from .providers.training import InMemoryDatasetAdapter, TorchModelAdapter, TrainingMicrobatch
from .runtime.training import (
    TrainingEngine,
    TrainingRunSpec,
)
from .stage1_precision_oracle import FIXTURE_ID, TASK_ID, build_stage1_s19_oracle, load_stage1_s19_fixture


GATE_ID = "G1-NUMERIC"
REPORT_SCHEMA = "stage1-s1-9-numeric-report-v1"
TRACE_SCHEMA = "stage1-s1-9-trace-bundle-v1"
TABLE_SCHEMA = "stage1-s1-9-comparison-table-v1"
GATE_SCHEMA = "stage1-s1-9-gate-record-v1"
EXPECTED_S1_7_PRODUCER = "dcc92506947c3ea30bed75542e006a26d5a2af1b"
EXPECTED_S1_7_INDEX_SHA256 = "4ca26c82d3e6246e0b99c7fc7a35882f712fc1142fa8f3fe9f5191bce64c2a7f"
EXPECTED_S1_7_INDEX_ARTIFACT_HASH = "21b14bdec009bee827dea5d604b363c6ce46ce55c06334d0409a2dc4400292cb"
EXPECTED_S1_7_GATE_HASH = "0c8d91dc010533a5c99229fe0c8577e10278f41d0f3fd754d885749c511e7f37"
# r11's five immutable primary roles predate a uniform self-hash envelope.
# This map is deliberately exhaustive: no role may select an arbitrary hash
# field, and index/file SHA-256 binding remains a separate mandatory check.
_S1_7_ROLE_SELF_HASH_FIELDS = {
    "fixture_manifest": "fixture_hash",
    "single_gpu_report": "artifact_hash",
    "gradient_bundle": "artifact_hash",
    "comparison_table": "table_hash",
    "gate_record": "artifact_hash",
}
# S1.8 V8 has one deliberately non-uniform primary role.  Freeze both the
# digest field and status-presence rule per exact role, so fixture metadata
# cannot be treated as a PASS/``artifact_hash`` envelope by accident.
_S1_8_ROLE_HASH_AND_STATUS = {
    "fixture_manifest": ("fixture_hash", None),
    "ddp_report": ("artifact_hash", "PASS"),
    "array_bundle": ("artifact_hash", "PASS"),
    "comparison_table": ("artifact_hash", "PASS"),
    "gate_record": ("artifact_hash", "PASS"),
}


class Stage1PrecisionError(RuntimeError):
    pass


def _with_hash(value: Mapping[str, Any], *, field: str = "artifact_hash") -> dict[str, Any]:
    if field in value:
        raise Stage1PrecisionError(f"S1_9_HASH_FIELD_ALREADY_PRESENT:{field}")
    result = dict(value)
    result[field] = canonical_json_hash(result)
    return result


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wire(value: Mapping[str, torch.Tensor] | TensorMap) -> dict[str, list[float]]:
    return {name: [float(item) for item in tensor.detach().cpu().to(torch.float64).reshape(-1).tolist()] for name, tensor in value.items()}


def _wire_equal(left: object, right: object, *, atol: float = 1e-6, rtol: float = 5e-4) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_wire_equal(left[key], right[key], atol=atol, rtol=rtol) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_wire_equal(a, b, atol=atol, rtol=rtol) for a, b in zip(left, right, strict=True))
    if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= atol + rtol * max(abs(float(left)), abs(float(right)))
    return left == right


T_AMP_SCALE_PROFILE = {"name": "T_AMP_SCALE", "atol": 1e-6, "rtol": 5e-4, "normalized_l2_limit": 5e-4}
REQUIREMENT_KEYS = (
    "fp32_statistics_containers",
    "mixed_magnitude_fp64_norm_and_nonfinite_preflight",
    "same_autocast_scale_unscale_equivalence",
    "unscale_negative_first_and_second_order_detected",
    "single_scaler_transition_and_skip",
    "clip_factor_matches_analytic_oracle",
    "u_clipped_uses_one_factor",
    "per_microbatch_clip_negative_control_detected",
    "raw_unclipped_field_preserved",
    "adamw_moment_and_decay_boundary",
    "repeated_short_run_bitwise_deterministic",
    "bf16_cuda_smoke",
    "ddp_global_nonfinite_skip",
)
SCALE_STATISTIC_OBJECTS = (
    "mean_gradient", "s1", "s2", "g1", "g2", "raw_core", "u_core", "raw_score", "u_score",
    "weighted_mean_gradient", "weighted_raw_core", "weighted_u_core", "weighted_raw_score", "weighted_u_score",
)
_NEAR_ZERO_T_AMP_CONTRACTS = {
    # These are frozen comparison vectors for the README §6 absolute branch.
    # They enter the published table alongside the real autocast objects, so a
    # schema/replay change cannot leave the required zero path unexercised.
    "t_amp_exact_zero_contract": {"actual": [0.0], "reference": [0.0]},
    "t_amp_near_zero_contract": {"actual": [4.9e-6], "reference": [5.0e-6]},
}

_NORMALIZED_T_AMP_ROW_FIELDS = frozenset({
    "object", "coordinate", "profile", "oracle_hash", "n_q", "n_q_rule", "branch",
    "oracle_norm_inf", "original_unit_max_abs_error", "scaled_max_abs_error",
    "scaled_threshold", "normalized_l2_error", "normalized_l2_limit", "nonfinite_count",
    "passed",
})
_NEAR_ZERO_T_AMP_ROW_FIELDS = frozenset({
    "object", "coordinate", "profile", "oracle_hash", "n_q", "n_q_rule", "branch",
    "near_zero_threshold", "absolute_threshold", "original_unit_max_abs_error",
    "scaled_max_abs_error", "normalized_l2_error", "nonfinite_count", "passed",
})
_PUBLIC_IMPORTANCE_ACCUMULATOR_VIEWS = frozenset(
    name for name, member in vars(ImportanceAccumulator).items() if isinstance(member, property)
)
_S19_ACCUMULATOR_VIEW_KEYS = frozenset({
    "signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped",
    "data_movement", "net_data_movement", "total_movement", "total_endpoint_movement",
    "weight_decay_movement", "net_weight_decay_movement", "actual_update_raw_importance",
    "actual_update_raw_importance_available", "magnitude", "attempted_steps",
})


def _validate_t_amp_row_contract(row: Mapping[str, Any], *, field: str) -> None:
    """Enforce the two mutually-exclusive T_AMP_SCALE row shapes.

    The repository's frozen stdlib schema subset predates JSON Schema
    ``oneOf``.  S1.9 therefore owns this exact-shape check in both runtime
    replay and formal validation, rather than silently treating oneOf as a
    comment.  This also protects the flat table projection from a jointly
    rehashed branch-field crossover.
    """

    branch = row.get("branch")
    if branch == "normalized":
        expected = _NORMALIZED_T_AMP_ROW_FIELDS
        if set(row) != expected or row.get("scaled_max_abs_error") is None or row.get("normalized_l2_error") is None:
            raise Stage1PrecisionError(f"S1_9_T_AMP_NORMALIZED_ROW_CONTRACT_INVALID:{field}")
    elif branch == "near_zero_absolute":
        expected = _NEAR_ZERO_T_AMP_ROW_FIELDS
        if set(row) != expected or row.get("scaled_max_abs_error") is not None or row.get("normalized_l2_error") is not None:
            raise Stage1PrecisionError(f"S1_9_T_AMP_NEAR_ZERO_ROW_CONTRACT_INVALID:{field}")
    else:
        raise Stage1PrecisionError(f"S1_9_T_AMP_BRANCH_INVALID:{field}")


def _validate_t_amp_table_contract(table: Mapping[str, Any]) -> None:
    comparisons = table.get("comparison_objects")
    rows = table.get("rows")
    if not isinstance(comparisons, list) or not isinstance(rows, list):
        raise Stage1PrecisionError("S1_9_T_AMP_TABLE_CONTAINER_INVALID")
    expected_projection: list[dict[str, Any]] = []
    for item_index, comparison in enumerate(comparisons):
        if not isinstance(comparison, Mapping):
            raise Stage1PrecisionError("S1_9_T_AMP_COMPARISON_NOT_OBJECT")
        object_id, oracle_hash = comparison.get("object"), comparison.get("oracle_hash")
        comparison_rows = comparison.get("rows")
        if not isinstance(object_id, str) or not isinstance(oracle_hash, str) or not isinstance(comparison_rows, list):
            raise Stage1PrecisionError("S1_9_T_AMP_COMPARISON_FIELDS_INVALID")
        for row_index, row in enumerate(comparison_rows):
            if not isinstance(row, Mapping):
                raise Stage1PrecisionError("S1_9_T_AMP_ROW_NOT_OBJECT")
            _validate_t_amp_row_contract(row, field=f"comparison_objects[{item_index}].rows[{row_index}]")
            if row.get("object") != object_id or row.get("oracle_hash") != oracle_hash:
                raise Stage1PrecisionError("S1_9_T_AMP_COMPARISON_ROW_BINDING_INVALID")
            expected_projection.append({"section": "t_amp_scale", "case_id": row["object"], "field": row["coordinate"], "actual": dict(row), "reference": {"profile": "T_AMP_SCALE", "oracle_hash": oracle_hash}, "passed": row["passed"]})
    for row_index, flat_row in enumerate(rows):
        if not isinstance(flat_row, Mapping) or not isinstance(flat_row.get("actual"), Mapping):
            raise Stage1PrecisionError("S1_9_T_AMP_FLAT_ROW_INVALID")
        _validate_t_amp_row_contract(flat_row["actual"], field=f"rows[{row_index}].actual")
    if rows != expected_projection:
        raise Stage1PrecisionError("S1_9_T_AMP_FLAT_PROJECTION_BINDING_INVALID")


def _stable_l2(values: list[float]) -> float:
    maximum = max((abs(value) for value in values), default=0.0)
    if maximum == 0.0:
        return 0.0
    return maximum * math.sqrt(math.fsum((value / maximum) ** 2 for value in values))


def _t_amp_compare(*, object_id: str, actual: Mapping[str, list[float]], reference: Mapping[str, list[float]], natural_scales: Mapping[str, Mapping[str, Any]], oracle_hash: str) -> dict[str, Any]:
    """Apply the frozen Stage1 README §6 comparison rule per tensor.

    This records both branches.  In particular, near-zero tensors do not get a
    misleading relative or normalised-L2 pass, while non-near-zero tensors are
    first normalised by the independent oracle's own maximum magnitude.
    """

    if set(actual) != set(reference) or set(actual) != set(natural_scales):
        raise Stage1PrecisionError(f"S1_9_T_AMP_COORDINATE_SET_INVALID:{object_id}")
    atol, rtol, l2_limit = (float(T_AMP_SCALE_PROFILE[key]) for key in ("atol", "rtol", "normalized_l2_limit"))
    rows: list[dict[str, Any]] = []
    for coordinate in sorted(reference):
        left, right = actual[coordinate], reference[coordinate]
        if len(left) != len(right) or not left or not all(math.isfinite(float(value)) for value in (*left, *right)):
            rows.append({"object": object_id, "coordinate": coordinate, "profile": "T_AMP_SCALE", "oracle_hash": oracle_hash, "passed": False, "reason": "shape_or_nonfinite"})
            continue
        registered = natural_scales[coordinate]
        if not isinstance(registered, Mapping):
            raise Stage1PrecisionError(f"S1_9_T_AMP_NATURAL_SCALE_REGISTRATION_INVALID:{object_id}:{coordinate}")
        natural = float(registered.get("n_q", float("nan")))
        natural_rule = registered.get("rule")
        if not math.isfinite(natural) or natural <= 0.0:
            raise Stage1PrecisionError(f"S1_9_T_AMP_NATURAL_SCALE_INVALID:{object_id}:{coordinate}")
        if not isinstance(natural_rule, str) or not natural_rule:
            raise Stage1PrecisionError(f"S1_9_T_AMP_NATURAL_SCALE_RULE_INVALID:{object_id}:{coordinate}")
        diff = [float(a) - float(b) for a, b in zip(left, right, strict=True)]
        max_abs_error, reference_norm_inf = max(abs(item) for item in diff), max(abs(float(item)) for item in right)
        near_threshold, absolute_threshold = 10.0 * atol * natural, atol * natural
        near_zero = reference_norm_inf <= near_threshold
        if near_zero:
            passed = max_abs_error <= absolute_threshold
            rows.append({"object": object_id, "coordinate": coordinate, "profile": "T_AMP_SCALE", "oracle_hash": oracle_hash, "n_q": natural, "n_q_rule": natural_rule, "branch": "near_zero_absolute", "near_zero_threshold": near_threshold, "absolute_threshold": absolute_threshold, "original_unit_max_abs_error": max_abs_error, "scaled_max_abs_error": None, "normalized_l2_error": None, "nonfinite_count": 0, "passed": passed})
            continue
        normal_actual = [float(item) / reference_norm_inf for item in left]
        normal_reference = [float(item) / reference_norm_inf for item in right]
        normal_diff = [a - b for a, b in zip(normal_actual, normal_reference, strict=True)]
        scaled_max = max(abs(item) for item in normal_diff)
        scale_factor = max(max(abs(item) for item in normal_actual), max(abs(item) for item in normal_reference))
        threshold = atol + rtol * scale_factor
        l2 = _stable_l2(normal_diff) / max(_stable_l2(normal_actual), _stable_l2(normal_reference), 1e-300)
        passed = scaled_max <= threshold and l2 <= l2_limit
        rows.append({"object": object_id, "coordinate": coordinate, "profile": "T_AMP_SCALE", "oracle_hash": oracle_hash, "n_q": natural, "n_q_rule": natural_rule, "branch": "normalized", "oracle_norm_inf": reference_norm_inf, "original_unit_max_abs_error": max_abs_error, "scaled_max_abs_error": scaled_max, "scaled_threshold": threshold, "normalized_l2_error": l2, "normalized_l2_limit": l2_limit, "nonfinite_count": 0, "passed": passed})
    return {"object": object_id, "profile": "T_AMP_SCALE", "oracle_hash": oracle_hash, "rows": rows, "passed": bool(rows) and all(row["passed"] for row in rows)}


def _t_amp_natural_scales(fixture: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    gradients = fixture.get("micro_gradients")
    weights = fixture.get("statistical_weights")
    learning_rates = fixture.get("learning_rates")
    if not isinstance(gradients, list) or not isinstance(weights, list) or not isinstance(learning_rates, Mapping):
        raise Stage1PrecisionError("S1_9_T_AMP_FIXTURE_INVALID")
    n_g = max(abs(float(value)) for sample in gradients if isinstance(sample, Mapping) for values in sample.values() for value in values)
    count, weight_sum, weight_square_sum = len(gradients), sum(abs(float(value)) for value in weights), sum(float(value) ** 2 for value in weights)
    names = set(learning_rates)
    if n_g <= 0.0 or not names:
        raise Stage1PrecisionError("S1_9_T_AMP_NG_INVALID")
    def registered(values: Mapping[str, float], rule: str) -> dict[str, dict[str, Any]]:
        return {name: {"n_q": float(value), "rule": rule} for name, value in values.items()}

    common = {name: n_g for name in names}
    squared = {name: n_g * n_g for name in names}
    score = {name: float(learning_rates[name]) * n_g * n_g for name in names}
    # These rules are registered before the actual values are compared.  The
    # two extra first-order fields make the scale-negative path auditable too;
    # the README's fixed second-order rules remain exactly M*n_g² and Σb²*n_g².
    return {
        "mean_gradient": registered(common, "n_g"),
        "s1": registered({name: count * n_g for name in names}, "M*n_g"),
        "s2": registered({name: count * n_g * n_g for name in names}, "M*n_g^2"),
        "g1": registered({name: weight_sum * n_g for name in names}, "sum(abs(b))*n_g"),
        "g2": registered({name: weight_square_sum * n_g * n_g for name in names}, "sum(b^2)*n_g^2"),
        "raw_core": registered(squared, "n_g^2"),
        "u_core": registered(squared, "n_g^2"),
        "raw_score": registered(score, "eta*n_g^2"),
        "u_score": registered(score, "eta*n_g^2"),
        "weighted_mean_gradient": registered(common, "n_g"),
        "weighted_raw_core": registered(squared, "n_g^2"),
        "weighted_u_core": registered(squared, "n_g^2"),
        "weighted_raw_score": registered(score, "eta*n_g^2"),
        "weighted_u_score": registered(score, "eta*n_g^2"),
        "optimizer_delta": registered({"parameter": max(float(fixture["sgd"]["learning_rate"]) * max(abs(float(value)) for value in fixture["sgd"]["gradient"]), 1e-12)}, "independent_optimizer_delta_scale"),
    }


def _near_zero_t_amp_comparisons(*, oracle_hash: str) -> list[dict[str, Any]]:
    """Exercise the frozen exact-zero and near-zero absolute comparison path.

    These are deliberately independent of a particular tensor's accidental
    cancellation: both vectors have their own positive pre-registered natural
    scale and therefore test the mutually-exclusive schema shape used by real
    comparison rows.
    """

    scales = {"value": {"n_q": 1.0, "rule": "frozen_zero_near_zero_contract_n_q"}}
    return [
        _t_amp_compare(
            object_id=object_id,
            actual={"value": values["actual"]},
            reference={"value": values["reference"]},
            natural_scales=scales,
            oracle_hash=oracle_hash,
        )
        for object_id, values in _NEAR_ZERO_T_AMP_CONTRACTS.items()
    ]


def _map_from_fixture(fixture: Mapping[str, Any], *, device: torch.device, scale: float = 1.0) -> list[TensorMap]:
    raw = fixture.get("micro_gradients")
    if not isinstance(raw, list):
        raise Stage1PrecisionError("S1_9_FIXTURE_MICRO_GRADIENTS_INVALID")
    result: list[TensorMap] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise Stage1PrecisionError("S1_9_FIXTURE_MICRO_GRADIENT_INVALID")
        values = {str(name): torch.tensor(numbers, dtype=torch.float32, device=device) * scale for name, numbers in item.items()}
        result.append(TensorMap(values))
    return result


def _statistics(samples: list[TensorMap], *, weights: list[float] | None = None, learning_rates: Mapping[str, Any] | None = None) -> dict[str, Any]:
    statistics = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float32, statistical_unit="microbatch_mean_gradient", sampling_design="s1_9_fixed_fixture")
    mean, raw, u = statistics.mean_gradient, raw_importance(statistics.mean_gradient), equal_u_importance(statistics)
    result = {
        "mean_gradient": _wire(mean), "s1": _wire(statistics.s1), "s2": _wire(statistics.s2),
        "raw_core": _wire(raw), "u_core": _wire(u),
        "accumulation_dtypes": {"s1": str(statistics.s1["w"].dtype), "s2": str(statistics.s2["w"].dtype)},
    }
    if weights is not None:
        weighted = WeightedSufficientStatistics.from_samples(samples, weights, accumulation_dtype=torch.float32, statistical_unit="microbatch_mean_gradient", weight_unit="frozen_s1_9_weight", sampling_design="s1_9_fixed_fixture", weights_exogenous=True, common_mean_assumption=True)
        weighted_mean = weighted.g1 / weighted.n1
        weighted_raw = weighted_mean * weighted_mean
        weighted_u = (weighted.g1 * weighted.g1 - weighted.g2) / weighted.denominator
        result.update({
            "g1": _wire(weighted.g1), "g2": _wire(weighted.g2),
            "weighted_mean_gradient": _wire(weighted_mean), "weighted_raw_core": _wire(weighted_raw), "weighted_u_core": _wire(weighted_u),
            "accumulation_dtypes": {**result["accumulation_dtypes"], "g1": str(weighted.g1["w"].dtype), "g2": str(weighted.g2["w"].dtype)},
        })
    if learning_rates is not None:
        result.update({
            "raw_score": _wire(_score(dict(raw.items()), learning_rates)), "u_score": _wire(_score(dict(u.items()), learning_rates)),
            "weighted_raw_score": _wire(_score(dict(weighted_raw.items()), learning_rates)) if weights is not None else None,
            "weighted_u_score": _wire(_score(dict(weighted_u.items()), learning_rates)) if weights is not None else None,
        })
    return result


class _S19AutocastRegression(torch.nn.Module):
    """Frozen two-coordinate regression used for every scale and BF16 path."""

    def __init__(self, fixture: Mapping[str, Any], *, device: torch.device) -> None:
        super().__init__()
        contract = fixture.get("autocast_regression")
        if not isinstance(contract, Mapping):
            raise Stage1PrecisionError("S1_9_AUTOCAST_REGRESSION_CONTRACT_INVALID")
        self.linear = torch.nn.Linear(int(contract["input_dim"]), 1, bias=True, device=device, dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor(contract["initial_weight"], dtype=torch.float32, device=device).reshape(1, -1))
            self.linear.bias.fill_(float(contract["initial_bias"]))


def _actual_autocast_local_gradients(fixture: Mapping[str, Any], *, device: torch.device, loss_scale: float, autocast_enabled: bool = True) -> list[TensorMap]:
    """Collect scaled local gradients from the same real autocast F/B path.

    This is intentionally not a tensor-fixture shortcut: each local gradient
    originates from a newly initialised frozen model, an autocast forward, and
    an autograd backward.  The caller alone decides whether its copied tensor
    is manually unscaled for statistics; optimizer ``unscale_`` state is not
    touched here.
    """

    contract = fixture.get("autocast_regression")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("microbatches"), list):
        raise Stage1PrecisionError("S1_9_AUTOCAST_MICROBATCH_CONTRACT_INVALID")
    model = _S19AutocastRegression(fixture, device=device)
    result: list[TensorMap] = []
    for index, item in enumerate(contract["microbatches"]):
        if not isinstance(item, Mapping):
            raise Stage1PrecisionError(f"S1_9_AUTOCAST_MICROBATCH_INVALID:{index}")
        model.zero_grad(set_to_none=True)
        inputs = torch.tensor(item["input"], dtype=torch.float32, device=device).reshape(1, -1)
        target = torch.tensor([[float(item["target"])]], dtype=torch.float32, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
            residual = model.linear(inputs) - target
            loss = residual.square().mean() * 0.5
        (loss * loss_scale).backward()
        weight, bias = model.linear.weight.grad, model.linear.bias.grad
        if weight is None or bias is None:
            raise Stage1PrecisionError("S1_9_AUTOCAST_GRADIENT_MISSING")
        result.append(TensorMap({"w": weight.detach().reshape(-1).clone(), "b": bias.detach().reshape(-1).clone()}))
    return result


def _wire_scaled(value: Mapping[str, list[float]], scalar: float) -> dict[str, list[float]]:
    return {name: [float(item) * scalar for item in items] for name, items in value.items()}


def _scale_trace(fixture: Mapping[str, Any], oracle: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    expected = oracle["statistics"]
    if not isinstance(expected, Mapping):
        raise Stage1PrecisionError("S1_9_ORACLE_STATISTICS_INVALID")
    scales = fixture.get("loss_scales")
    weights = fixture.get("statistical_weights")
    learning_rates = fixture.get("learning_rates")
    if not isinstance(scales, list) or not isinstance(weights, list) or not isinstance(learning_rates, Mapping):
        raise Stage1PrecisionError("S1_9_FIXTURE_LOSS_SCALES_INVALID")
    numeric_weights = [float(value) for value in weights]
    baseline_scaled = _actual_autocast_local_gradients(fixture, device=device, loss_scale=1.0)
    baseline = _statistics(baseline_scaled, weights=numeric_weights, learning_rates=learning_rates)
    baseline_fields = {key: baseline[key] for key in SCALE_STATISTIC_OBJECTS}
    rows: list[dict[str, Any]] = []
    for scale_value in scales:
        scale = float(scale_value)
        scaled = _actual_autocast_local_gradients(fixture, device=device, loss_scale=scale)
        recovered: list[TensorMap] = []
        for sample in scaled:
            attempt = GradientAttempt.capture(dict(sample.items()), gradient_scale=scale, scaled=True).unscale().check_finite()
            if attempt.phase is not GradientPhase.FINITE:
                raise Stage1PrecisionError("S1_9_SCALE_UNSCALE_DID_NOT_REMAIN_FINITE")
            recovered.append(TensorMap(dict(attempt.gradients)))
        actual = _statistics(recovered, weights=numeric_weights, learning_rates=learning_rates)
        negative = _statistics(scaled, weights=numeric_weights, learning_rates=learning_rates)
        fields = {key: actual[key] for key in SCALE_STATISTIC_OBJECTS}
        # The deliberately wrong path retains scaled local gradients.  Every
        # registered statistic must participate in exactly one of the two
        # scaling laws; weighted routes are not allowed to evade this control.
        first_order_fields = ("mean_gradient", "s1", "g1", "weighted_mean_gradient")
        second_order_fields = tuple(field for field in SCALE_STATISTIC_OBJECTS if field not in first_order_fields)
        first_order = all(_wire_equal(negative[field], _wire_scaled(baseline[field], scale)) for field in first_order_fields)
        second_order = all(_wire_equal(negative[field], _wire_scaled(baseline[field], scale * scale)) for field in second_order_fields)
        rows.append({
            "loss_scale": scale, "scaled_local_gradients": [_wire(sample) for sample in scaled], "manually_unscaled_local_gradients": [_wire(sample) for sample in recovered],
            "actual": fields, "reference_same_autocast": baseline_fields,
            "negative_control_excluded_from_scores": {key: negative[key] for key in baseline_fields},
            "all_fp32": all(dtype == "torch.float32" for dtype in actual["accumulation_dtypes"].values()),
            "passed": _wire_equal(fields, baseline_fields), "negative_first_order_fields": list(first_order_fields), "negative_second_order_fields": list(second_order_fields), "negative_first_order_scale_detected": first_order,
            "negative_second_order_scale_squared_detected": second_order,
        })
    frozen_fixture_match = _wire_equal(baseline_fields, {key: expected[key] for key in SCALE_STATISTIC_OBJECTS})
    return {"device": str(device), "autocast_dtype": "torch.bfloat16", "reference_same_autocast": baseline_fields, "frozen_fixture_fp64_oracle_match": frozen_fixture_match, "rows": rows, "all_passed": all(row["passed"] for row in rows), "all_negative_first_order_detected": all(row["negative_first_order_scale_detected"] for row in rows), "all_negative_second_order_detected": all(row["negative_second_order_scale_squared_detected"] for row in rows), "all_statistics_fp32": all(row["all_fp32"] for row in rows)}


def _mixed_magnitude_trace(fixture: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    """Audit FP64 norm stability separately from FP32 second-moment safety."""

    contract = fixture.get("mixed_magnitude_gradients")
    if not isinstance(contract, Mapping) or not isinstance(contract.get("stable_norm"), list):
        raise Stage1PrecisionError("S1_9_MIXED_MAGNITUDE_FIXTURE_INVALID")
    values = torch.tensor(contract["stable_norm"], dtype=torch.float32, device=device)
    sample = TensorMap({"mixed": values})
    norm, factor = compute_global_clip_factor(dict(sample.items()), max_norm=None)
    s2_nonfinite = g2_nonfinite = False
    try:
        EqualSufficientStatistics.from_samples([sample, sample], accumulation_dtype=torch.float32)
    except Exception:
        s2_nonfinite = True
    try:
        WeightedSufficientStatistics.from_samples([sample, sample], [1.0, 2.0], accumulation_dtype=torch.float32, statistical_unit="microbatch_mean_gradient", weight_unit="frozen_s1_9_weight", sampling_design="s1_9_mixed_magnitude", weights_exogenous=True, common_mean_assumption=True)
    except Exception:
        g2_nonfinite = True
    expected_nonfinite = contract.get("expected_fp32_second_moment_nonfinite") is True
    return {"device": str(device), "fp64_global_norm": norm, "clip_factor": factor, "norm_finite": math.isfinite(norm), "small_component": float(values[-1].detach().cpu()), "s2_nonfinite_detected": s2_nonfinite, "g2_nonfinite_detected": g2_nonfinite, "expected_nonfinite": expected_nonfinite, "passed": math.isfinite(norm) and factor == 1.0 and s2_nonfinite and g2_nonfinite and expected_nonfinite}


def _score(core: Mapping[str, torch.Tensor], learning_rates: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    return {name: value * float(learning_rates[name]) for name, value in core.items()}


def _clip_trace(fixture: Mapping[str, Any], oracle: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    # Use the identical autocast-forward local-gradient route used by the
    # scale tests; clipping is not allowed to validate only a hand-built map.
    samples = _actual_autocast_local_gradients(fixture, device=device, loss_scale=1.0)
    stats = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float32)
    mean = stats.mean_gradient
    clip = fixture.get("clip")
    learning_rates = fixture.get("learning_rates")
    if not isinstance(clip, Mapping) or not isinstance(learning_rates, Mapping):
        raise Stage1PrecisionError("S1_9_CLIP_FIXTURE_INVALID")
    norm, factor = compute_global_clip_factor(dict(mean.items()), float(clip["max_norm"]), eps=float(clip["eps"]))
    clipped_mean = {name: value * factor for name, value in mean.items()}
    post_clip_norm, _ = compute_global_clip_factor(clipped_mean, max_norm=None)
    u = equal_u_importance(stats)
    score = _score(dict(u.items()), learning_rates)
    clipped = {name: value * factor for name, value in score.items()}
    squared = {name: value * factor * factor for name, value in score.items()}
    individual: list[TensorMap] = []
    for sample in samples:
        _, local_factor = compute_global_clip_factor(dict(sample.items()), float(clip["max_norm"]), eps=float(clip["eps"]))
        individual.append(TensorMap({name: value * local_factor for name, value in sample.items()}))
    per_micro_u = _score(dict(equal_u_importance(EqualSufficientStatistics.from_samples(individual, accumulation_dtype=torch.float32)).items()), learning_rates)
    oracle_clip = oracle.get("clip")
    if not isinstance(oracle_clip, Mapping):
        raise Stage1PrecisionError("S1_9_ORACLE_CLIP_INVALID")
    return {
        "global_norm": norm, "post_clip_global_norm": post_clip_norm, "clip_factor": factor, "u_score_natural_scales": _t_amp_natural_scales(fixture)["u_score"], "u_unclipped_score": _wire(score), "u_clipped_score": _wire(clipped),
        "u_squared_factor_negative_control": _wire(squared), "per_microbatch_clip_negative_control": _wire(per_micro_u),
        "raw_unclipped_score": _wire(_score(dict(raw_importance(mean).items()), learning_rates)),
        "raw_unclipped_oracle_score": oracle["statistics"]["raw_score"],
        # The independent expression is FP64 while this branch intentionally
        # reduces frozen FP32 local gradients.  This tolerance is only for the
        # analytic scalar factor; all tensor-valued score objects are checked
        # by the stricter, per-coordinate T_AMP_SCALE records.
        "factor_matches_oracle": _wire_equal(factor, oracle_clip["clip_factor"], atol=1e-7, rtol=5e-7),
        "single_factor_identity": _wire_equal(_wire(clipped), oracle_clip["u_clipped_score"]),
        "squared_factor_detected": not _wire_equal(_wire(squared), oracle_clip["u_clipped_score"]),
        "per_microbatch_clip_detected": not _wire_equal(_wire(per_micro_u), oracle_clip["u_clipped_score"]),
        "raw_unclipped_matches_clip_pre_oracle": _wire_equal(_wire(_score(dict(raw_importance(mean).items()), learning_rates)), oracle["statistics"]["raw_score"]),
        "clip_source": "same_batch_global_mean", "unbiasedness_claim": "none",
    }


class _CountingAdamW(torch.optim.AdamW):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.step_calls = 0

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        self.step_calls += 1
        return super().step(closure)


class _CountingGradScaler(torch.amp.GradScaler):
    """Real GradScaler with call counters; it is not a behavioral shim."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.unscale_calls = 0
        self.step_calls = 0
        self.update_calls = 0

    def unscale_(self, optimizer: torch.optim.Optimizer) -> Any:  # type: ignore[override]
        self.unscale_calls += 1
        return super().unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.step_calls += 1
        return super().step(optimizer, *args, **kwargs)

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:  # type: ignore[override]
        self.update_calls += 1
        return super().update(new_scale)


class _S19GradientInjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, parameter: torch.Tensor, inject_nonfinite: bool) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.inject_nonfinite = inject_nonfinite
        ctx.parameter_shape = parameter.shape
        return parameter.new_zeros(())

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[no-untyped-def]
        # Preserve the upstream loss scale: returning a literal one here would
        # make a perfectly finite synthetic gradient appear different after
        # GradScaler's formal unscale at 8× versus 16×.  The injected bad path
        # remains an actual Inf gradient on every element.
        multiplier = float("inf") if ctx.inject_nonfinite else 1.0
        return gradient.expand(ctx.parameter_shape) * multiplier, None


class _S19FiniteSkipFiniteClassifier(torch.nn.Module):
    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float32, device=device))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        injected = _S19GradientInjection.apply(self.weight, bool(torch.isinf(input_ids).any().item()))
        return torch.stack((injected.expand(input_ids.shape[0]), torch.zeros_like(injected).expand(input_ids.shape[0])), dim=1)


def _state_wire(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor": _wire({"value": value})["value"], "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, Mapping):
        return {str(key): _state_wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_state_wire(item) for item in value]
    if isinstance(value, list):
        return [_state_wire(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise Stage1PrecisionError(f"S1_9_STATE_WIRE_UNSUPPORTED:{type(value).__name__}")


def _engine_microbatch(*, sample_id: str, micro: int, value: float, device: torch.device) -> TrainingMicrobatch:
    """Build a frozen data sample whose identity is independent of attempt count.

    A retry/skip is a control-flow event, not a new data identity.  Keeping
    these IDs stable lets the finite→skip→finite execution prove that its
    second committed update consumed the same frozen sample as the direct
    finite reference, even though its successful attempt index is one larger.
    """

    return TrainingMicrobatch(
        f"s19-sample-{sample_id}-micro-{micro}",
        {"input_ids": torch.full((2, 1), value, dtype=torch.float32, device=device), "labels": torch.zeros(2, dtype=torch.long, device=device)},
        tuple(f"s19-{sample_id}-{micro}-{index}" for index in range(2)), {"fixture_id": FIXTURE_ID, "sample_id": sample_id},
    )


def _engine_skip_trace(*, device: torch.device) -> dict[str, Any]:
    """Run real TrainingEngine finite → skip → finite with a real scaler.

    The reference consumes the two finite batches only.  It therefore lets the
    trace demonstrate that a skipped attempt advances only its declared
    control state while parameters, AdamW state, scheduler, long-term
    accumulator and the next finite batch rejoin the reference path.
    """

    def state_snapshot(engine: TrainingEngine, optimizer: torch.optim.Optimizer, scheduler: Any, scaler: _CountingGradScaler) -> dict[str, Any]:
        if engine.tracker is None:
            raise Stage1PrecisionError("S1_9_SKIP_TRACKER_MISSING")
        # Complete objects, not state-size proxies: this is deliberately
        # serialised before/after the skipped attempt and used below for every
        # state family required by the finite→skip→finite contract.
        return _state_wire({
            "parameters": {name: parameter.detach().clone() for name, parameter in engine.model.module.named_parameters()},
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "long_term_accumulator": engine.tracker.accumulator.state_dict(),
            "cursor": engine.cursor.state_dict(), "rng": {
                "cpu": torch.random.get_rng_state().clone(),
                # A one-GPU worker must never inspect every visible device.
                "cuda_current": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
            },
            "attempt_skip_state": engine.state.to_dict(), "scaler": scaler.state_dict(),
        })

    def run(with_skip: bool) -> tuple[TrainingEngine, list[Any], torch.optim.Optimizer, Any, _CountingGradScaler, list[dict[str, Any]]]:
        torch.manual_seed(19091)
        module = _S19FiniteSkipFiniteClassifier(device=device)
        first_finite = (
            _engine_microbatch(sample_id="finite-first", micro=0, value=1.0, device=device),
            _engine_microbatch(sample_id="finite-first", micro=1, value=2.0, device=device),
        )
        next_finite = (
            _engine_microbatch(sample_id="finite-next", micro=0, value=3.0, device=device),
            _engine_microbatch(sample_id="finite-next", micro=1, value=4.0, device=device),
        )
        skipped = (
            _engine_microbatch(sample_id="injected-nonfinite", micro=0, value=float("inf"), device=device),
            _engine_microbatch(sample_id="injected-nonfinite", micro=1, value=float("inf"), device=device),
        )
        batches = [first_finite, skipped, next_finite] if with_skip else [first_finite, next_finite]
        adapter = TorchModelAdapter(module, task_type="sequence_classification")
        dataset = InMemoryDatasetAdapter("s19-skip-fixture", tuple(batches))
        optimizer = _CountingAdamW(module.parameters(), lr=0.05, weight_decay=0.1, foreach=False, fused=False)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        scaler = _CountingGradScaler(device=device.type, enabled=True, init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1)
        engine = TrainingEngine(spec=TrainingRunSpec(f"s19-skip-{'observed' if with_skip else 'reference'}", "local_fixture", max_steps=2, max_attempts=3 if with_skip else 2, importance_enabled=True, estimator_name="u", accumulation_dtype="float32", weights_exogenous=True, common_mean_assumption=True, autocast_dtype="bfloat16" if device.type == "cuda" else "none"), model=adapter, optimizer=optimizer, scheduler=scheduler, scaler=scaler, cursor=dataset.cursor(seed=19), capture_boundary_trace=True)
        records: list[Any] = []
        attempts: list[dict[str, Any]] = []
        # This is the exact inner path used by ``run``.  Retaining the
        # snapshots here gives S1.9 a complete pre/post state tree at the
        # skipped attempt without changing the shared runtime observability.
        while engine.state.global_step < engine.spec.max_steps and engine.state.attempt_index < engine.spec.max_attempts:
            attempt_pre = state_snapshot(engine, optimizer, scheduler, scaler)
            try:
                microbatches = engine.cursor.next_microbatches()
            except StopIteration as error:
                raise Stage1PrecisionError("S1_9_SKIP_FIXTURE_EXHAUSTED") from error
            post_fetch = state_snapshot(engine, optimizer, scheduler, scaler)
            record = engine._run_attempt(microbatches)
            engine._records.append(record)  # mirrors TrainingEngine.run bookkeeping.
            post_attempt = state_snapshot(engine, optimizer, scheduler, scaler)
            attempts.append({"record": record.to_dict(), "attempt_pre_state": attempt_pre, "post_fetch_state": post_fetch, "post_attempt_state": post_attempt})
            records.append(record)
        return engine, records, optimizer, scheduler, scaler, attempts

    observed_engine, observed_records, observed_optimizer, observed_scheduler, observed_scaler, observed_attempts = run(True)
    reference_engine, reference_records, reference_optimizer, reference_scheduler, reference_scaler, reference_attempts = run(False)
    if observed_engine.tracker is None or reference_engine.tracker is None:
        raise Stage1PrecisionError("S1_9_SKIP_TRACKER_MISSING")
    observed_lifecycle = [dict(item) for item in observed_engine.boundary_trace]
    skip_boundaries = [item for item in observed_lifecycle if item["boundary"] == "09_skip_discard_and_scaler"]
    loss_scales = [item["observation"]["loss_scale"] for item in observed_lifecycle if item["boundary"] == "03_freeze_loss_scale"]
    def normalized_success_records(records: Any) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for record in records:
            value = record.to_dict() if hasattr(record, "to_dict") else dict(record)
            value.pop("attempt_index", None); value.pop("attempt_commit_state_hash", None)
            result.append(value)
        return result

    state_equal = {
        "parameters": _wire({name: parameter.detach() for name, parameter in observed_engine.model.module.named_parameters()}) == _wire({name: parameter.detach() for name, parameter in reference_engine.model.module.named_parameters()}),
        "optimizer": _state_wire(observed_optimizer.state_dict()) == _state_wire(reference_optimizer.state_dict()),
        "scheduler": _state_wire(observed_scheduler.state_dict()) == _state_wire(reference_scheduler.state_dict()),
        "accumulator_except_skip_counter": _state_wire({key: value for key, value in observed_engine.tracker.accumulator.state_dict().items() if key != "skipped_steps"}) == _state_wire({key: value for key, value in reference_engine.tracker.accumulator.state_dict().items() if key != "skipped_steps"}),
        "successful_records": normalized_success_records([record for record in observed_records if record.status == "COMMITTED"]) == normalized_success_records(reference_records),
        "next_finite_batch": observed_records[-1].batch_ids == reference_records[-1].batch_ids,
    }
    skipped_attempt = observed_attempts[1]
    skip_pre, skip_fetched, skip_post = skipped_attempt["attempt_pre_state"], skipped_attempt["post_fetch_state"], skipped_attempt["post_attempt_state"]
    if not isinstance(skip_pre, Mapping) or not isinstance(skip_fetched, Mapping) or not isinstance(skip_post, Mapping):
        raise Stage1PrecisionError("S1_9_SKIP_SNAPSHOT_INVALID")
    def unchanged_after_fetch(key: str) -> bool:
        return skip_fetched.get(key) == skip_post.get(key)
    pre_control, post_control = skip_pre.get("attempt_skip_state"), skip_post.get("attempt_skip_state")
    if not isinstance(pre_control, Mapping) or not isinstance(post_control, Mapping):
        raise Stage1PrecisionError("S1_9_SKIP_CONTROL_SNAPSHOT_INVALID")
    scaler_pre, scaler_post = skip_fetched.get("scaler"), skip_post.get("scaler")
    if not isinstance(scaler_pre, Mapping) or not isinstance(scaler_post, Mapping):
        raise Stage1PrecisionError("S1_9_SKIP_SCALER_SNAPSHOT_INVALID")
    skip_full_state_checks = {
        "parameters_unchanged": unchanged_after_fetch("parameters"), "optimizer_unchanged": unchanged_after_fetch("optimizer"),
        "scheduler_unchanged": unchanged_after_fetch("scheduler"), "long_term_accumulator_except_skip_counter": (
            isinstance(skip_fetched.get("long_term_accumulator"), Mapping) and isinstance(skip_post.get("long_term_accumulator"), Mapping) and
            {key: value for key, value in skip_fetched["long_term_accumulator"].items() if key != "skipped_steps"} == {key: value for key, value in skip_post["long_term_accumulator"].items() if key != "skipped_steps"}
        ),
        "cursor_advanced_exactly_one_frozen_batch": (
            isinstance(skip_pre.get("cursor"), Mapping) and isinstance(skip_fetched.get("cursor"), Mapping) and
            int(skip_fetched["cursor"].get("index", -1)) == int(skip_pre["cursor"].get("index", -2)) + 1 and skip_fetched.get("cursor") == skip_post.get("cursor")
        ),
        # The fixed in-memory fixture has an explicit zero-random-consumption
        # contract: data selection is a deterministic index fetch and the
        # model contains no stochastic layer.  Equality here proves that
        # contract, while the cursor check above proves the actual progression.
        "rng_zero_consumption_contract": skip_pre.get("rng") == skip_fetched.get("rng") == skip_post.get("rng"),
        "attempt_and_skip_counters_advanced_once": (post_control.get("global_step") == pre_control.get("global_step") and post_control.get("attempt_index") == int(pre_control.get("attempt_index", -1)) + 1 and post_control.get("skipped_steps") == int(pre_control.get("skipped_steps", -1)) + 1),
        "scaler_state_changed_only_by_backoff": (set(scaler_pre) == set(scaler_post) and all(scaler_pre[key] == scaler_post[key] for key in scaler_pre if key in {"growth_factor", "backoff_factor", "growth_interval"}) and float(scaler_post.get("scale", float("nan"))) == float(scaler_pre.get("scale", float("nan"))) * 0.5),
    }
    skip_observation = skip_boundaries[0]["observation"] if len(skip_boundaries) == 1 else {}
    return {
        # The complete boundary list is consumed to derive the loss-scale and
        # skip observations above.  Publish its deterministic digest and
        # cardinality rather than an untyped heterogeneous event array: the
        # formal role remains replayable while its schema can stay fail-closed.
        "device": str(device), "records": [record.to_dict() for record in observed_records], "attempt_state_snapshots": observed_attempts, "reference_attempt_state_snapshots": reference_attempts, "lifecycle": {"entry_count": len(observed_lifecycle), "canonical_hash": canonical_json_hash(observed_lifecycle)},
        "state": observed_engine.state.to_dict(), "cursor_state": dict(observed_engine.cursor.state_dict()),
        "parameters": _wire({name: parameter.detach() for name, parameter in observed_engine.model.module.named_parameters()}),
        "optimizer_state": _state_wire(observed_optimizer.state_dict()), "scheduler_state": _state_wire(observed_scheduler.state_dict()),
        "accumulator_state": _state_wire(observed_engine.tracker.accumulator.state_dict()), "scaler_state": _state_wire(observed_scaler.state_dict()),
        "loss_scales": loss_scales, "scaler_calls": {"unscale": observed_scaler.unscale_calls, "step": observed_scaler.step_calls, "update": observed_scaler.update_calls}, "actual_optimizer_step_calls": observed_optimizer.step_calls,
        "skip_observation": skip_observation, "rng_contract": {"fixture_rng_consumption": 0, "validated": True}, "skip_pre_state": skip_pre, "skip_post_fetch_state": skip_fetched, "skip_post_state": skip_post,
        "skip_full_state_checks": skip_full_state_checks, "reference_comparison": state_equal,
        "all_full_state_checks": all(state_equal.values()) and all(skip_full_state_checks.values()), "skip_control_progression": observed_engine.state.global_step == 2 and observed_engine.state.attempt_index == 3 and observed_engine.state.skipped_steps == 1 and int(observed_engine.cursor.state_dict()["index"]) == 3,
    }


def _scaler_trace(*, device: torch.device) -> dict[str, Any]:
    # The complete state snapshots include RNG; reset the local fixture seed so
    # an independent CPU replay does not inherit caller process history.
    torch.manual_seed(29019)
    if device.type == "cuda":
        torch.cuda.manual_seed(29019)
    parameter = torch.nn.Parameter(torch.tensor([0.75, -0.25], dtype=torch.float32, device=device))
    optimizer = _CountingAdamW([parameter], lr=0.1, weight_decay=0.1, foreach=False, fused=False)
    scaler = _CountingGradScaler(device=device.type, enabled=True, init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1)
    finite_gradient = torch.tensor([0.5, -1.0], dtype=torch.float32, device=device)

    def snapshot() -> dict[str, Any]:
        return _state_wire({
            "parameter": parameter.detach().clone(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            "rng": {"cpu": torch.random.get_rng_state().clone(), "cuda_current": torch.cuda.get_rng_state(device) if device.type == "cuda" else None},
        })

    attempts: list[dict[str, Any]] = []
    scaled = unscaled = None
    for label, gradient in (("finite_first", finite_gradient), ("injected_nonfinite", torch.full_like(finite_gradient, float("nan"))), ("finite_second", finite_gradient)):
        optimizer.zero_grad(set_to_none=True)
        before = snapshot()
        scaler.scale((parameter * gradient).sum()).backward()
        if label == "finite_first":
            scaled = parameter.grad.detach().clone()
        scaler.unscale_(optimizer)
        if label == "finite_first":
            unscaled = parameter.grad.detach().clone()
        before_step = snapshot()
        scaler.step(optimizer)
        scaler.update()
        attempts.append({"label": label, "pre_state": before, "post_unscale_state": before_step, "post_state": snapshot()})
    if scaled is None or unscaled is None:
        raise Stage1PrecisionError("S1_9_SCALER_GRADIENT_CAPTURE_MISSING")
    first, skipped, second = attempts
    skip_optimizer_unchanged = skipped["post_unscale_state"]["optimizer"] == skipped["post_state"]["optimizer"]
    skip_parameter_unchanged = skipped["post_unscale_state"]["parameter"] == skipped["post_state"]["parameter"]
    scaler_before_skip, scaler_after_skip = skipped["post_unscale_state"]["scaler"], skipped["post_state"]["scaler"]
    skip_scaler_backoff_once = isinstance(scaler_before_skip, Mapping) and isinstance(scaler_after_skip, Mapping) and float(scaler_after_skip.get("scale", float("nan"))) == float(scaler_before_skip.get("scale", float("nan"))) * 0.5
    engine = _engine_skip_trace(device=device)
    return {
        "device": str(device), "attempts": attempts,
        "scaled_gradient": _wire({"parameter": scaled}), "unscaled_gradient": _wire({"parameter": unscaled}), "reference_gradient": _wire({"parameter": finite_gradient}),
        "finite_parameter_changed": first["pre_state"]["parameter"] != first["post_state"]["parameter"] and second["pre_state"]["parameter"] != second["post_state"]["parameter"], "skip_parameter_unchanged": skip_parameter_unchanged,
        "skip_optimizer_state_unchanged": skip_optimizer_unchanged, "manual_actual_optimizer_calls": optimizer.step_calls,
        "scaler_calls": {"unscale": scaler.unscale_calls, "step": scaler.step_calls, "update": scaler.update_calls}, "skip_scaler_backoff_once": skip_scaler_backoff_once,
        "scaler_transitions_once": [float(attempt["pre_state"]["scaler"]["scale"]) for attempt in attempts] == [8.0, 16.0, 8.0] and float(second["post_state"]["scaler"]["scale"]) == 16.0,
        "unscale_matches_reference": _wire_equal(_wire({"parameter": unscaled}), _wire({"parameter": finite_gradient})),
        "engine_finite_skip_finite": engine,
    }


def _optimizer_trace(fixture: Mapping[str, Any], oracle: Mapping[str, Any], *, device: torch.device) -> dict[str, Any]:
    sgd = fixture.get("sgd")
    adamw = fixture.get("adamw")
    if not isinstance(sgd, Mapping) or not isinstance(adamw, Mapping):
        raise Stage1PrecisionError("S1_9_OPTIMIZER_FIXTURE_INVALID")
    sgd_parameter = torch.nn.Parameter(torch.tensor(sgd["initial"], dtype=torch.float64, device=device))
    sgd_optimizer = torch.optim.SGD([sgd_parameter], lr=float(sgd["learning_rate"]), foreach=False)
    sgd_parameter.grad = torch.tensor(sgd["gradient"], dtype=torch.float64, device=device)
    sgd_outcome = OptimizerBridge({"parameter": sgd_parameter}, sgd_optimizer).step()
    sgd_wire = {"data_delta": _wire(sgd_outcome.data_delta), "weight_decay_delta": _wire(sgd_outcome.weight_decay_delta), "total_delta": _wire(sgd_outcome.total_delta)}
    sgd_actual_update = {"parameter": [-(float(delta) * float(gradient)) for delta, gradient in zip(sgd_wire["data_delta"]["parameter"], sgd["gradient"], strict=True)]}
    initial = torch.tensor(adamw["initial"], dtype=torch.float64, device=device)
    gradient = torch.tensor(adamw["gradient"], dtype=torch.float64, device=device)

    def run_adamw(*, prime: bool) -> dict[str, Any]:
        parameter = torch.nn.Parameter(initial.clone())
        optimizer = torch.optim.AdamW([parameter], lr=float(adamw["learning_rate"]), betas=tuple(float(item) for item in adamw["betas"]), eps=float(adamw["eps"]), weight_decay=float(adamw["weight_decay"]), foreach=False, fused=False)
        if prime:
            parameter.grad = torch.tensor(adamw["priming_gradient"], dtype=torch.float64, device=device)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            parameter.data.copy_(initial)
        parameter.grad = gradient.clone()
        outcome = OptimizerBridge({"parameter": parameter}, optimizer).step()
        state = optimizer.state[parameter]
        return {"data_delta": _wire(outcome.data_delta), "weight_decay_delta": _wire(outcome.weight_decay_delta), "total_delta": _wire(outcome.total_delta), "exp_avg": _wire({"parameter": state["exp_avg"]}), "exp_avg_sq": _wire({"parameter": state["exp_avg_sq"]}), "step": int(state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"])}

    fresh, primed = run_adamw(prime=False), run_adamw(prime=True)
    oracle_optimizer = oracle.get("optimizer")
    if not isinstance(oracle_optimizer, Mapping) or not isinstance(oracle_optimizer.get("adamw"), Mapping):
        raise Stage1PrecisionError("S1_9_ORACLE_OPTIMIZER_INVALID")
    oracle_adamw = oracle_optimizer["adamw"]
    expected_fresh, expected_primed = oracle_adamw.get("fresh"), oracle_adamw.get("primed_after_parameter_reset")
    if not isinstance(expected_fresh, Mapping) or not isinstance(expected_primed, Mapping):
        raise Stage1PrecisionError("S1_9_ORACLE_ADAMW_CASES_INVALID")
    # The isolated oracle uses flat vectors; production uses named tensors.
    def expected_wire(case: Mapping[str, Any]) -> dict[str, Any]:
        return {key: {"parameter": list(case[key])} if key != "step" else int(case[key]) for key in ("data_delta", "weight_decay_delta", "total_delta", "exp_avg", "exp_avg_sq", "step")}

    expected_sgd = {"parameter": list(oracle_optimizer["sgd_data_delta"])}
    return {
        "sgd": sgd_wire, "sgd_actual_update_raw_importance": sgd_actual_update,
        "adamw_fresh": fresh, "adamw_primed": primed,
        "oracle": {"sgd_data_delta": expected_sgd, "adamw_fresh": expected_wire(expected_fresh), "adamw_primed": expected_wire(expected_primed)},
        "sgd_data_delta_matches_oracle": _wire_equal(sgd_wire["data_delta"], expected_sgd, atol=1e-12, rtol=1e-10),
        "adamw_fresh_matches_oracle": _wire_equal(fresh, expected_wire(expected_fresh), atol=1e-12, rtol=1e-10),
        "adamw_primed_matches_oracle": _wire_equal(primed, expected_wire(expected_primed), atol=1e-12, rtol=1e-10),
        "adamw_moment_history_changes_actual_update": not _wire_equal(fresh["total_delta"], primed["total_delta"], atol=1e-12, rtol=1e-10),
        "actual_update_is_separate_from_gradient_space_u": _wire_equal(sgd_actual_update, {"parameter": list(oracle_optimizer["actual_update_uses_negative_data_delta_times_current_gradient"])}),
    }


def _quality_metric(actual: Mapping[str, list[float]], reference: Mapping[str, list[float]]) -> dict[str, Any]:
    if set(actual) != set(reference):
        raise Stage1PrecisionError("S1_9_BF16_QUALITY_COORDINATES_INVALID")
    per_tensor: dict[str, Any] = {}
    for name in sorted(actual):
        observed, expected = actual[name], reference[name]
        if len(observed) != len(expected) or not observed:
            raise Stage1PrecisionError(f"S1_9_BF16_QUALITY_SHAPE_INVALID:{name}")
        dot = math.fsum(float(left) * float(right) for left, right in zip(observed, expected, strict=True))
        observed_norm, expected_norm = _stable_l2([float(value) for value in observed]), _stable_l2([float(value) for value in expected])
        cosine = 1.0 if observed_norm == expected_norm == 0.0 else dot / max(observed_norm * expected_norm, 1e-300)
        errors = [abs(float(left) - float(right)) for left, right in zip(observed, expected, strict=True)]
        per_tensor[name] = {"cosine": cosine, "norm_ratio": observed_norm / max(expected_norm, 1e-300), "max_abs_error": max(errors), "mean_abs_error": math.fsum(errors) / len(errors)}
    return {"per_tensor": per_tensor, "mean_cosine": math.fsum(value["cosine"] for value in per_tensor.values()) / len(per_tensor), "mean_norm_ratio": math.fsum(value["norm_ratio"] for value in per_tensor.values()) / len(per_tensor), "mean_max_abs_error": math.fsum(value["max_abs_error"] for value in per_tensor.values()) / len(per_tensor)}


def _all_wire_finite(value: object) -> bool:
    if isinstance(value, Mapping):
        return all(_all_wire_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_wire_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run_stage1_s19_bf16_smoke(*, source_root: str | Path, device: str | torch.device, checkpoint_dir: str | Path) -> dict[str, Any]:
    """Exercise BF16 and FP32 on one frozen fixture, including save/restore.

    This is intentionally a quality and executable-state smoke, not a second
    formula oracle: scale correctness is tested separately using identical
    autocast forwards, while every BF16-vs-FP32 discrepancy remains visible.
    """

    selected, root, checkpoints = torch.device(device), Path(source_root), Path(checkpoint_dir)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise Stage1PrecisionError("S1_9_BF16_REQUIRES_CUDA")
    if not torch.are_deterministic_algorithms_enabled():
        raise Stage1PrecisionError("S1_9_BF16_DETERMINISM_NOT_CONFIGURED")
    fixture = load_stage1_s19_fixture(root)
    weights, learning_rates = fixture.get("statistical_weights"), fixture.get("learning_rates")
    contract = fixture.get("autocast_regression")
    if not isinstance(weights, list) or not isinstance(learning_rates, Mapping) or not isinstance(contract, Mapping):
        raise Stage1PrecisionError("S1_9_BF16_FIXTURE_INVALID")

    def seed() -> None:
        torch.manual_seed(1909)
        torch.cuda.manual_seed(1909)

    def optimizer_step(*, autocast_enabled: bool) -> dict[str, Any]:
        seed()
        model = _S19AutocastRegression(fixture, device=selected)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False)
        losses: list[float] = []
        total_loss = model.linear.weight.new_zeros(())
        for item in contract["microbatches"]:
            inputs = torch.tensor(item["input"], dtype=torch.float32, device=selected).reshape(1, -1)
            target = torch.tensor([[float(item["target"])]], dtype=torch.float32, device=selected)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                loss = (model.linear(inputs) - target).square().mean() * 0.5
            losses.append(float(loss.detach().cpu()))
            total_loss = total_loss + loss / len(contract["microbatches"])
        total_loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}
        pre = _state_wire({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        saved = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(optimizer.state_dict())}
        restored_model = _S19AutocastRegression(fixture, device=selected)
        restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False)
        restored_model.load_state_dict(saved["model"], strict=True)
        restored_optimizer.load_state_dict(saved["optimizer"])
        post = _state_wire({"model": model.state_dict(), "optimizer": optimizer.state_dict()})
        restored = _state_wire({"model": restored_model.state_dict(), "optimizer": restored_optimizer.state_dict()})
        norm, _ = compute_global_clip_factor(gradients, max_norm=None)
        return {"losses": losses, "mean_loss": float(total_loss.detach().cpu()), "gradients": _wire(gradients), "global_gradient_norm": norm, "pre_step_state": pre, "post_step_state": post, "restored_state": restored, "save_restore_exact": post == restored, "finite": all(math.isfinite(item) for item in (*losses, norm)) and _all_wire_finite(_wire(gradients))}

    def resume_smoke() -> dict[str, Any]:
        """Save and fresh-reload the real TrainingEngine checkpoint payload.

        This deliberately uses the runtime's production tracker and checkpoint
        validator rather than a hand-maintained square-gradient stand-in.  The
        persisted payload therefore binds every public importance accumulator
        field together with control, cursor, RNG, optimizer, scheduler and
        scaler state.
        """

        checkpoints.mkdir(parents=True, exist_ok=False)
        checkpoint_store = CheckpointStore(checkpoints)

        def build_engine(*, store: CheckpointStore) -> TrainingEngine:
            torch.manual_seed(1909); torch.cuda.manual_seed(1909)
            module = _S19FiniteSkipFiniteClassifier(device=selected)
            batches = tuple(
                (
                    _engine_microbatch(sample_id=f"bf16-resume-{index}", micro=0, value=float(1 + 2 * index), device=selected),
                    _engine_microbatch(sample_id=f"bf16-resume-{index}", micro=1, value=float(2 + 2 * index), device=selected),
                )
                for index in range(3)
            )
            optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            scaler = _CountingGradScaler("cuda", enabled=True, init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1)
            return TrainingEngine(
                spec=TrainingRunSpec("s19-bf16-engine-resume", "local_fixture", max_steps=3, max_attempts=3, importance_enabled=True, estimator_name="u", accumulation_dtype="float32", weights_exogenous=True, common_mean_assumption=True, autocast_dtype="bfloat16"),
                model=TorchModelAdapter(module, task_type="sequence_classification"), optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, cursor=InMemoryDatasetAdapter("s19-bf16-resume", batches).cursor(seed=1909), checkpoint_store=store,
            )

        def run_one(engine: TrainingEngine) -> dict[str, Any]:
            record = engine._run_attempt(engine.cursor.next_microbatches())
            engine._records.append(record)
            return record.to_dict()

        def field_hashes(engine: TrainingEngine) -> dict[str, str]:
            if engine.tracker is None:
                raise Stage1PrecisionError("S1_9_BF16_ENGINE_TRACKER_MISSING")
            accumulator = engine.tracker.accumulator
            importance_views = {
                    "signed": accumulator.signed,
                    "positive": accumulator.positive,
                    "negative_mass": accumulator.negative_mass,
                    "absolute": accumulator.absolute,
                    "raw": accumulator.raw,
                    "raw_clipped": accumulator.raw_clipped,
                    "data_movement": accumulator.data_movement,
                    "net_data_movement": accumulator.net_data_movement,
                    "total_movement": accumulator.total_movement,
                    "total_endpoint_movement": accumulator.total_endpoint_movement,
                    "weight_decay_movement": accumulator.weight_decay_movement,
                    "net_weight_decay_movement": accumulator.net_weight_decay_movement,
                    "actual_update_raw_importance": accumulator.actual_update_raw_importance,
                    "actual_update_raw_importance_available": accumulator.actual_update_raw_importance_available,
                    "magnitude": accumulator.magnitude,
                    "attempted_steps": accumulator.attempted_steps,
            }
            if set(importance_views) != _S19_ACCUMULATOR_VIEW_KEYS or _S19_ACCUMULATOR_VIEW_KEYS != _PUBLIC_IMPORTANCE_ACCUMULATOR_VIEWS:
                raise Stage1PrecisionError("S1_9_BF16_ACCUMULATOR_PUBLIC_VIEW_SET_DRIFT")
            # Bind the live production streams directly; ``CheckpointStore``
            # is the only persistence boundary, never a hand-serialized
            # ``_checkpoint_state`` substitute.
            rng = {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch_cpu": torch.random.get_rng_state(),
                "torch_cuda": tuple(torch.cuda.get_rng_state(index) for index in range(torch.cuda.device_count())),
            }
            serializable = {
                "model": _state_wire(engine.model.module.state_dict()),
                "optimizer": _state_wire(engine.optimizer.state_dict()),
                "scheduler": _state_wire(None if engine.scheduler is None else engine.scheduler.state_dict()),
                "scaler": _state_wire(None if engine.scaler is None else engine.scaler.state_dict()),
                "cursor": _state_wire(dict(engine.cursor.state_dict())),
                "training_state": _state_wire(engine.state.to_dict()),
                "importance": _state_wire(accumulator.state_dict()),
                "records": _state_wire([record.to_dict() for record in engine._records]),
                "importance_trajectory_points": _state_wire([point.to_dict() for point in engine._importance_points]),
                "checkpoint_ids": _state_wire(list(engine._checkpoint_ids)),
                **{f"importance_view_{key}": _state_wire(value) for key, value in importance_views.items()},
                "rng": {
                    "python": hashlib.sha256(repr(rng["python"]).encode("utf-8")).hexdigest(),
                    "numpy": hashlib.sha256(repr(rng["numpy"]).encode("utf-8")).hexdigest(),
                    "torch_cpu": _state_wire(rng["torch_cpu"]),
                    "torch_cuda": _state_wire(rng["torch_cuda"]),
                },
                "run_spec_hash": engine.spec.spec_hash,
                "registry_hash": engine.registry.coordinate_registry_hash,
                "optimizer_contract_hash": engine.registry.optimizer_contract_hash,
                "runtime_layout_hash": engine.registry.runtime_layout_hash,
            }
            return {key: canonical_json_hash(value) for key, value in serializable.items()}

        def metadata(engine: TrainingEngine) -> dict[str, Any]:
            return {
                "run_spec_hash": engine.spec.spec_hash,
                "registry_hash": engine.registry.coordinate_registry_hash,
                "optimizer_contract_hash": engine.registry.optimizer_contract_hash,
                "runtime_layout_hash": engine.registry.runtime_layout_hash,
                "world_size": 1,
            }

        def negative_payload(raw: Mapping[str, Any], *, checkpoint_id: str) -> dict[str, Any]:
            """Rebind a copied payload to a controlled negative store commit."""

            payload = copy.deepcopy(dict(raw))
            state = dict(payload["training_state"])
            state["last_checkpoint_id"] = checkpoint_id
            payload["training_state"] = state
            payload["checkpoint_ids"] = [checkpoint_id]
            # The negative stores own a one-node lineage, so no trajectory
            # point may retain the authoritative checkpoint's identity.
            payload["importance_trajectory_points"] = []
            return payload

        def negative_resume_rejected(label: str, payload: Mapping[str, Any]) -> bool:
            store = CheckpointStore(checkpoints / f"negative-{label}")
            checkpoint_id = f"s19-bf16-negative-{label}"
            store.publish(checkpoint_id, negative_payload(payload, checkpoint_id=checkpoint_id), generation=2, metadata=metadata(engine))
            probe = build_engine(store=store)
            before = field_hashes(probe)
            try:
                probe.resume_checkpoint(checkpoint_id)
            except Exception:
                return field_hashes(probe) == before
            return False

        engine = build_engine(store=checkpoint_store)
        first_step, second_step = run_one(engine), run_one(engine)
        # ``save_checkpoint`` is TrainingEngine's production checkpoint-now
        # surface: it publishes a tensor bundle and an authoritative commit.
        checkpoint_id = engine.save_checkpoint()
        checkpoint = checkpoint_store.commits / f"{checkpoint_id}.json"
        loaded, commit = checkpoint_store.load(checkpoint_id, expected_metadata=metadata(engine))
        checkpoint_role_fields = {"schema_version", "run_spec_hash", "registry_hash", "optimizer_contract_hash", "runtime_layout_hash", "training_state", "model", "optimizer", "scheduler", "scaler", "rng", "cursor", "importance", "records", "importance_trajectory_points", "checkpoint_ids"}
        if not isinstance(loaded, Mapping) or set(loaded) != checkpoint_role_fields:
            raise Stage1PrecisionError("S1_9_BF16_RESUME_ROLE_SET_INVALID")

        omitted = dict(loaded); omitted.pop("importance")
        omission_rejected = negative_resume_rejected("omission", omitted)
        reordered = dict(loaded); reordered["records"] = list(reversed(loaded["records"]))
        record_order_rejected = negative_resume_rejected("record-order", reordered)

        # Corrupt only a separately published negative artifact.  The source
        # checkpoint remains authoritative and is never modified in-place.
        corrupt_store = CheckpointStore(checkpoints / "negative-corruption")
        corrupt_id = "s19-bf16-negative-corruption"
        corrupt_store.publish(corrupt_id, negative_payload(loaded, checkpoint_id=corrupt_id), generation=2, metadata=metadata(engine))
        corrupt_manifest = corrupt_store.objects / corrupt_id / "manifest.json"
        corrupt_bytes = corrupt_manifest.read_bytes()
        corrupt_manifest.write_bytes(corrupt_bytes[:-1] + bytes([corrupt_bytes[-1] ^ 1]))
        corrupt_probe = build_engine(store=corrupt_store)
        corrupt_before = field_hashes(corrupt_probe)
        try:
            corrupt_probe.resume_checkpoint(corrupt_id)
            corruption_detected = False
        except Exception:
            corruption_detected = field_hashes(corrupt_probe) == corrupt_before

        fresh = build_engine(store=checkpoint_store)
        resumed_id = fresh.resume_checkpoint(checkpoint_id)
        distinct_reload = fresh is not engine and fresh.tracker is not engine.tracker and fresh.tracker is not None and engine.tracker is not None and fresh.tracker.accumulator is not engine.tracker.accumulator
        continuous_next = run_one(engine)
        restored_next = run_one(fresh)
        continuous_hashes, restored_hashes = field_hashes(engine), field_hashes(fresh)
        return {"checkpoint_sha256": _sha256_file(checkpoint), "checkpoint_size_bytes": checkpoint.stat().st_size, "checkpoint_id": checkpoint_id, "checkpoint_manifest_sha256": commit.manifest_sha256, "checkpoint_role_fields": sorted(loaded), "first_step": first_step, "second_step": second_step, "continuous_next_step": continuous_next, "restored_next_step": restored_next, "continuous_state_field_hashes": continuous_hashes, "restored_state_field_hashes": restored_hashes, "next_step_exact": continuous_next == restored_next and continuous_hashes == restored_hashes and continuous_next is not restored_next, "next_step_independent_engines": distinct_reload, "accumulator_public_view_set_exact": _S19_ACCUMULATOR_VIEW_KEYS == _PUBLIC_IMPORTANCE_ACCUMULATOR_VIEWS, "data_cursor_restored": resumed_id == checkpoint_id and int(loaded["cursor"]["index"]) == 2, "corruption_detected": corruption_detected, "omission_rejected": omission_rejected, "record_order_rejected": record_order_rejected}

    def run_once() -> dict[str, Any]:
        seed()
        fp32_samples = _actual_autocast_local_gradients(fixture, device=selected, loss_scale=1.0, autocast_enabled=False)
        seed()
        bf16_samples = _actual_autocast_local_gradients(fixture, device=selected, loss_scale=1.0, autocast_enabled=True)
        fp32_stats = _statistics(fp32_samples, weights=[float(value) for value in weights], learning_rates=learning_rates)
        bf16_stats = _statistics(bf16_samples, weights=[float(value) for value in weights], learning_rates=learning_rates)
        fp32_step, bf16_step = optimizer_step(autocast_enabled=False), optimizer_step(autocast_enabled=True)
        return {
            "fp32_statistics": fp32_stats, "bf16_statistics": bf16_stats,
            "mean_gradient_quality": _quality_metric(bf16_stats["mean_gradient"], fp32_stats["mean_gradient"]),
            "raw_core_quality": _quality_metric(bf16_stats["raw_core"], fp32_stats["raw_core"]),
            "u_core_quality": _quality_metric(bf16_stats["u_core"], fp32_stats["u_core"]),
            "raw_score_quality": _quality_metric(bf16_stats["raw_score"], fp32_stats["raw_score"]),
            "u_score_quality": _quality_metric(bf16_stats["u_score"], fp32_stats["u_score"]),
            "fp32_optimizer_step": fp32_step, "bf16_optimizer_step": bf16_step,
            "fp32_containers": all(dtype == "torch.float32" for dtype in fp32_stats["accumulation_dtypes"].values()),
            "bf16_containers": all(dtype == "torch.float32" for dtype in bf16_stats["accumulation_dtypes"].values()),
            "all_finite": _all_wire_finite(fp32_stats) and _all_wire_finite(bf16_stats) and fp32_step["finite"] and bf16_step["finite"],
        }

    first, second = run_once(), run_once()
    repeated_bitwise_equal = canonical_json_hash(first) == canonical_json_hash(second)
    resume = resume_smoke()
    status = "PASS" if first["all_finite"] and first["fp32_containers"] and first["bf16_containers"] and first["fp32_optimizer_step"]["save_restore_exact"] and first["bf16_optimizer_step"]["save_restore_exact"] and resume["next_step_exact"] and resume["data_cursor_restored"] and repeated_bitwise_equal else "FAIL"
    return {
        "status": status, "device": str(selected), "fixture_id": FIXTURE_ID, "model_parameter_dtype": "torch.float32",
        "autocast_dtype": "torch.bfloat16", "observation": first, "resume": resume, "repeated_bitwise_equal": repeated_bitwise_equal,
        "determinism": {"algorithms_enabled": torch.are_deterministic_algorithms_enabled(), "cudnn_deterministic": bool(torch.backends.cudnn.deterministic), "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "allowed_nondeterministic_kernel_classes": [], "kernel_policy": "empty_pre_registered_allowlist"},
    }


def _source_hashes(source_root: Path) -> dict[str, str]:
    # This is the executable dependency closure of the local numeric trace,
    # rather than a short list of files that happen to share the task prefix.
    relative = (
        "fixtures/stage1/stage1-s19-precision-fixture-v1.json",
        "src/param_importance_nlp/stage1_precision.py", "src/param_importance_nlp/stage1_precision_oracle.py",
        "src/param_importance_nlp/contracts/jsonio.py", "src/param_importance_nlp/contracts/runtime_evidence.py",
        "src/param_importance_nlp/core/accumulator.py", "src/param_importance_nlp/core/estimators.py",
        "src/param_importance_nlp/core/registry.py", "src/param_importance_nlp/core/sufficient_statistics.py", "src/param_importance_nlp/core/tensors.py",
        "src/param_importance_nlp/providers/training.py", "src/param_importance_nlp/runtime/gradients.py",
        "src/param_importance_nlp/runtime/optimizer.py", "src/param_importance_nlp/runtime/reducers.py", "src/param_importance_nlp/runtime/task_artifacts.py", "src/param_importance_nlp/runtime/training.py", "src/param_importance_nlp/runtime/operations.py",
        "ops/stage1/formalize_s1_6.py", "ops/stage1/formalize_s1_9.py", "ops/stage1/run_s1_9_single_bf16_worker.py", "ops/stage1/run_s1_9_ddp_skip_worker.py",
        "tests/test_stage1_s19_precision.py",
        "schemas/stage1/s1-9-precision-fixture-v1.json", "schemas/stage1/s1-9-numeric-report-v1.json",
        "schemas/stage1/s1-9-oracle-bundle-v1.json", "schemas/stage1/s1-9-trace-bundle-v1.json",
        "schemas/stage1/s1-9-comparison-table-v1.json", "schemas/stage1/s1-9-gate-record-v1.json",
        "schemas/stage1/s1-9-replay-validation-v1.json", "schemas/stage1/s1-9-validation-v1.json",
        "schemas/stage1/s1-9-formalization-index-v1.json", "schemas/stage1/s1-9-single-bf16-worker-v1.json", "schemas/stage1/s1-9-ddp-skip-worker-v1.json", "schemas/stage1/s1-9-bf16-checkpoint-store-reproduction-v1.json",
    )
    result: dict[str, str] = {}
    for item in relative:
        path = source_root / item
        if not path.is_file():
            raise Stage1PrecisionError(f"S1_9_SOURCE_FILE_MISSING:{item}")
        result[item] = _sha256_file(path)
    return result


def build_stage1_s19_evidence(source_root: str | Path, *, producer_commit: str, scope: str, upstream_evidence: Mapping[str, Any] | None = None, bf16_observation: Mapping[str, Any] | None = None, ddp_skip_observation: Mapping[str, Any] | None = None, device: str | torch.device = "cpu") -> dict[str, dict[str, Any]]:
    """Build local evidence; it remains NOT_RUN until a BF16 GPU observation exists."""

    root, selected = Path(source_root), torch.device(device)
    fixture = load_stage1_s19_fixture(root)
    oracle = build_stage1_s19_oracle(root)
    scale = _scale_trace(fixture, oracle, device=selected)
    mixed_magnitude = _mixed_magnitude_trace(fixture, device=selected)
    clip = _clip_trace(fixture, oracle, device=selected)
    scaler = _scaler_trace(device=selected)
    optimizer = _optimizer_trace(fixture, oracle, device=selected)
    bf16 = {"status": "NOT_RUN", "reason": "FORMAL_GPU_BF16_REQUIRED"} if bf16_observation is None else dict(bf16_observation)
    ddp_skip = {"status": "NOT_RUN", "reason": "S1_8_HANDOFF_AND_FORMAL_NCCL_REQUIRED"} if ddp_skip_observation is None else dict(ddp_skip_observation)
    bf16_passed = bf16.get("status") == "PASS"
    ddp_skip_passed = ddp_skip.get("status") == "PASS"
    repeat_scale = _scale_trace(fixture, oracle, device=selected)
    deterministic_cpu = canonical_json_hash(scale) == canonical_json_hash(repeat_scale)
    engine_skip = scaler["engine_finite_skip_finite"]
    if not isinstance(engine_skip, Mapping):
        raise Stage1PrecisionError("S1_9_ENGINE_SKIP_TRACE_INVALID")
    natural_scales = _t_amp_natural_scales(fixture)
    amp_comparisons: list[dict[str, Any]] = []
    for scale_row in scale["rows"]:
        for object_id in SCALE_STATISTIC_OBJECTS:
            amp_comparisons.append(_t_amp_compare(object_id=object_id, actual=scale_row["actual"][object_id], reference=oracle["statistics"][object_id], natural_scales=natural_scales[object_id], oracle_hash=str(oracle["oracle_hash"])))
    amp_comparisons.append(_t_amp_compare(object_id="optimizer_delta", actual=optimizer["sgd"]["data_delta"], reference=optimizer["oracle"]["sgd_data_delta"], natural_scales=natural_scales["optimizer_delta"], oracle_hash=str(oracle["oracle_hash"])))
    amp_comparisons.extend(_near_zero_t_amp_comparisons(oracle_hash=str(oracle["oracle_hash"])))
    t_amp_passed = all(comparison["passed"] for comparison in amp_comparisons)
    requirements = {
        "fp32_statistics_containers": scale["all_statistics_fp32"],
        "mixed_magnitude_fp64_norm_and_nonfinite_preflight": mixed_magnitude["passed"],
        "same_autocast_scale_unscale_equivalence": scale["all_passed"] and scale["frozen_fixture_fp64_oracle_match"] and t_amp_passed,
        "unscale_negative_first_and_second_order_detected": scale["all_negative_first_order_detected"] and scale["all_negative_second_order_detected"],
        "single_scaler_transition_and_skip": scaler["scaler_transitions_once"] and scaler["scaler_calls"] == {"unscale": 3, "step": 3, "update": 3} and scaler["manual_actual_optimizer_calls"] == 2 and scaler["skip_parameter_unchanged"] and scaler["skip_optimizer_state_unchanged"] and scaler["skip_scaler_backoff_once"] and engine_skip["scaler_calls"] == {"unscale": 3, "step": 3, "update": 3} and engine_skip["actual_optimizer_step_calls"] == 2 and engine_skip["all_full_state_checks"] and engine_skip["skip_control_progression"],
        "clip_factor_matches_analytic_oracle": clip["factor_matches_oracle"],
        "u_clipped_uses_one_factor": clip["single_factor_identity"] and clip["squared_factor_detected"],
        "per_microbatch_clip_negative_control_detected": clip["per_microbatch_clip_detected"],
        "raw_unclipped_field_preserved": clip["raw_unclipped_matches_clip_pre_oracle"],
        "adamw_moment_and_decay_boundary": optimizer["sgd_data_delta_matches_oracle"] and optimizer["adamw_fresh_matches_oracle"] and optimizer["adamw_primed_matches_oracle"] and optimizer["adamw_moment_history_changes_actual_update"] and optimizer["actual_update_is_separate_from_gradient_space_u"],
        "repeated_short_run_bitwise_deterministic": deterministic_cpu and (bf16.get("repeated_bitwise_equal") is True if bf16_passed else False),
        "bf16_cuda_smoke": bf16_passed,
        "ddp_global_nonfinite_skip": ddp_skip_passed,
    }
    if tuple(requirements) != REQUIREMENT_KEYS:
        raise Stage1PrecisionError("S1_9_REQUIREMENT_KEYSET_DRIFT")
    deferred = {"bf16_cuda_smoke", "ddp_global_nonfinite_skip", "repeated_short_run_bitwise_deterministic"}
    status = "PASS" if all(requirements.values()) else "NOT_RUN" if not bf16_passed and not ddp_skip_passed and all(value for key, value in requirements.items() if key not in deferred) else "FAIL"
    trace = _with_hash({"schema_version": TRACE_SCHEMA, "fixture_id": FIXTURE_ID, "scale_unscale": scale, "mixed_magnitude": mixed_magnitude, "clip": clip, "scaler": scaler, "optimizer": optimizer, "determinism": {"first_scale_trace_hash": canonical_json_hash(scale), "second_scale_trace_hash": canonical_json_hash(repeat_scale), "bitwise_equal": deterministic_cpu}, "bf16": bf16, "ddp_skip": ddp_skip}, field="trace_hash")
    rows: list[dict[str, Any]] = []
    for comparison in amp_comparisons:
        for comparison_row in comparison["rows"]:
            rows.append({"section": "t_amp_scale", "case_id": comparison_row["object"], "field": comparison_row["coordinate"], "actual": comparison_row, "reference": {"profile": "T_AMP_SCALE", "oracle_hash": oracle["oracle_hash"]}, "passed": comparison_row["passed"]})
    table = _with_hash({"schema_version": TABLE_SCHEMA, "fixture_id": FIXTURE_ID, "profile": "T_AMP_SCALE", "rows": rows, "comparison_objects": amp_comparisons}, field="table_hash")
    upstream = dict(upstream_evidence or {})
    report = _with_hash({"schema_version": REPORT_SCHEMA, "status": status, "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "scope": scope, "producer_commit": producer_commit, "upstream": upstream, "implementation_source_sha256": _source_hashes(root), "requirements": requirements, "bf16": bf16, "ddp_skip": ddp_skip, "oracle_hash": oracle["oracle_hash"], "trace_hash": trace["trace_hash"], "table_hash": table["table_hash"]}, field="report_hash")
    gate = _with_hash({"schema_version": GATE_SCHEMA, "status": status, "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "requirements": requirements}, field="artifact_hash")
    return {"numeric_report": report, "oracle_bundle": oracle, "trace_bundle": trace, "comparison_table": table, "gate_record": gate}


def _strict_validate_s1_9_evidence_schemas(source_root: Path, evidence: Mapping[str, Any]) -> None:
    """Run the repository's strict schema subset before trusting role hashes.

    A canonical self-hash protects bytes, not the meaning of a jointly edited
    evidence object.  Runtime replay therefore uses exactly the same local
    stdlib validator as formal publication for every persisted S1.9 role.
    """

    import importlib.util

    from .contracts.jsonio import loads_strict_json

    validator_path = source_root / "ops" / "stage1" / "formalize_s1_6.py"
    schema_root = source_root / "schemas" / "stage1"
    if not validator_path.is_file() or not schema_root.is_dir():
        raise Stage1PrecisionError("S1_9_RUNTIME_SCHEMA_RESOURCES_MISSING")
    try:
        spec = importlib.util.spec_from_file_location("_s19_runtime_schema_subset", validator_path)
        if spec is None or spec.loader is None:
            raise TypeError("validator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        registry: dict[str, Mapping[str, Any]] = {}
        for path in sorted(schema_root.glob("s1-9-*.json")):
            schema = loads_strict_json(path.read_bytes())
            if not isinstance(schema, Mapping) or not isinstance(schema.get("$id"), str):
                raise TypeError(path.name)
            registry[path.name] = schema
            registry[str(schema["$id"])] = schema
        role_schemas = {
            "numeric_report": "s1-9-numeric-report-v1.json", "oracle_bundle": "s1-9-oracle-bundle-v1.json",
            "trace_bundle": "s1-9-trace-bundle-v1.json", "comparison_table": "s1-9-comparison-table-v1.json",
            "gate_record": "s1-9-gate-record-v1.json",
        }
        if set(evidence) != set(role_schemas):
            raise TypeError("role_set")
        for role, schema_name in role_schemas.items():
            value, schema = evidence[role], registry[schema_name]
            if not isinstance(value, Mapping):
                raise TypeError(role)
            module._validate_schema(value, schema, registry, document=schema, path=f"runtime.{role}")
            if role == "comparison_table":
                _validate_t_amp_table_contract(value)
    except Stage1PrecisionError:
        raise
    except Exception as error:
        raise Stage1PrecisionError("S1_9_RUNTIME_SCHEMA_VALIDATION_FAILED") from error


def validate_stage1_s19_evidence(evidence: Mapping[str, Any], *, source_root: str | Path | None = None) -> dict[str, str]:
    expected = {"numeric_report", "oracle_bundle", "trace_bundle", "comparison_table", "gate_record"}
    if set(evidence) != expected or not all(isinstance(value, Mapping) for value in evidence.values()):
        raise Stage1PrecisionError("S1_9_EVIDENCE_ROLE_SET_INVALID")
    root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[2]
    _strict_validate_s1_9_evidence_schemas(root, evidence)
    report, oracle, trace, table, gate = (evidence["numeric_report"], evidence["oracle_bundle"], evidence["trace_bundle"], evidence["comparison_table"], evidence["gate_record"])
    assert isinstance(report, Mapping) and isinstance(oracle, Mapping) and isinstance(trace, Mapping) and isinstance(table, Mapping) and isinstance(gate, Mapping)
    roles = ((report, "report_hash", REPORT_SCHEMA), (oracle, "oracle_hash", "stage1-s1-9-oracle-bundle-v1"), (trace, "trace_hash", TRACE_SCHEMA), (table, "table_hash", TABLE_SCHEMA), (gate, "artifact_hash", GATE_SCHEMA))
    for value, hash_key, schema in roles:
        body = dict(value); supplied = body.pop(hash_key, None)
        if value.get("schema_version") != schema or not isinstance(supplied, str) or supplied != canonical_json_hash(body):
            raise Stage1PrecisionError("S1_9_ROLE_SELF_HASH_OR_SCHEMA_INVALID")
    if report.get("gate_id") != GATE_ID or report.get("task_id") != TASK_ID or report.get("fixture_id") != FIXTURE_ID or gate.get("requirements") != report.get("requirements"):
        raise Stage1PrecisionError("S1_9_REPORT_GATE_BINDING_INVALID")
    requirements = report.get("requirements")
    if (
        not isinstance(requirements, Mapping)
        or tuple(requirements) != REQUIREMENT_KEYS
        or any(type(value) is not bool for value in requirements.values())
    ):
        raise Stage1PrecisionError("S1_9_REQUIREMENT_KEYSET_INVALID")
    if (report.get("status") == "PASS" or gate.get("status") == "PASS") and not all(requirements.values()):
        raise Stage1PrecisionError("S1_9_PASS_GATE_REQUIREMENT_FALSE")
    if report.get("oracle_hash") != oracle.get("oracle_hash") or report.get("trace_hash") != trace.get("trace_hash") or report.get("table_hash") != table.get("table_hash"):
        raise Stage1PrecisionError("S1_9_ROLE_CROSS_BINDING_INVALID")
    if report.get("implementation_source_sha256") != _source_hashes(root):
        raise Stage1PrecisionError("S1_9_SOURCE_MAP_DRIFT")
    return {"report_hash": str(report["report_hash"]), "oracle_hash": str(oracle["oracle_hash"]), "trace_hash": str(trace["trace_hash"]), "table_hash": str(table["table_hash"]), "gate_artifact_hash": str(gate["artifact_hash"])}


def replay_stage1_s19_evidence(evidence: Mapping[str, Any], *, source_root: str | Path, device: str | torch.device = "cpu") -> dict[str, Any]:
    """Rebuild only the deterministic CPU roles and bind their source hashes."""

    hashes = validate_stage1_s19_evidence(evidence, source_root=source_root)
    report = evidence["numeric_report"]
    trace = evidence["trace_bundle"]
    if not isinstance(report, Mapping) or not isinstance(trace, Mapping):
        raise Stage1PrecisionError("S1_9_REPLAY_ROLE_INVALID")
    rebuilt = build_stage1_s19_evidence(source_root, producer_commit=str(report["producer_commit"]), scope=str(report["scope"]), upstream_evidence=report.get("upstream") if isinstance(report.get("upstream"), Mapping) else None, bf16_observation=trace.get("bf16") if isinstance(trace.get("bf16"), Mapping) else None, ddp_skip_observation=trace.get("ddp_skip") if isinstance(trace.get("ddp_skip"), Mapping) else None, device=device)
    # A replay never accepts a jointly re-hashed report/gate/trace.  The
    # external BF16/DDP observations are supplied only from the persisted
    # trace, then every derived role is rebuilt and must match byte-for-byte.
    for role in ("numeric_report", "oracle_bundle", "comparison_table", "gate_record"):
        if rebuilt[role] != evidence[role]:
            raise Stage1PrecisionError(f"S1_9_REPLAY_ROLE_MISMATCH:{role}")
    rebuilt_trace = dict(rebuilt["trace_bundle"]); saved_trace = dict(trace)
    if rebuilt_trace != saved_trace:
        raise Stage1PrecisionError("S1_9_REPLAY_TRACE_MISMATCH")
    return _with_hash({"schema_version": "stage1-s1-9-replay-validation-v1", "status": "PASS", "source_report_hash": hashes["report_hash"], "source_oracle_hash": hashes["oracle_hash"], "source_trace_hash": hashes["trace_hash"], "source_table_hash": hashes["table_hash"], "source_gate_artifact_hash": hashes["gate_artifact_hash"], "replayed_roles": ["oracle_bundle", "trace_bundle", "comparison_table"]}, field="replay_hash")


def _safe_reference(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage1PrecisionError(f"S1_9_S1_7_REFERENCE_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1PrecisionError(f"S1_9_S1_7_REFERENCE_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1PrecisionError(f"S1_9_S1_7_REFERENCE_ESCAPE:{field}") from error
    return candidate


def _validate_s1_7_frozen_schemas(objects: Mapping[str, Mapping[str, Any]]) -> None:
    """Validate all consumed S1.7 documents against their frozen schemas.

    The S1.7 formalizer owns these schemas, but an S1.9 consumer cannot treat
    a self-hashed JSON object as an equivalent replacement.  Load the same
    project validator and registry used by that formalizer so an unknown or
    missing nested key prevents S1.9 from obtaining a lease.
    """

    import importlib.util

    repository = Path(__file__).resolve().parents[2]
    formalizer_path = repository / "ops" / "stage1" / "formalize_s1_6.py"
    schema_dir = repository / "schemas" / "stage1"
    if not formalizer_path.is_file() or not schema_dir.is_dir():
        raise Stage1PrecisionError("S1_9_S1_7_FROZEN_SCHEMA_RESOURCES_MISSING")
    spec = importlib.util.spec_from_file_location("_s19_s17_schema_subset", formalizer_path)
    if spec is None or spec.loader is None:
        raise Stage1PrecisionError("S1_9_S1_7_FROZEN_SCHEMA_VALIDATOR_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        from .contracts.jsonio import loads_strict_json

        registry: dict[str, Mapping[str, Any]] = {}
        paths = sorted(schema_dir.glob("s1-7-*.json"))
        if len(paths) != 8:
            raise Stage1PrecisionError("S1_9_S1_7_FROZEN_SCHEMA_REGISTRY_INCOMPLETE")
        for path in paths:
            schema = loads_strict_json(path.read_bytes())
            if not isinstance(schema, Mapping) or not isinstance(schema.get("$id"), str):
                raise Stage1PrecisionError(f"S1_9_S1_7_FROZEN_SCHEMA_INVALID:{path.name}")
            registry[path.name] = schema
            registry[str(schema["$id"])] = schema
        filenames = {
            "fixture_manifest": "s1-7-fixture-manifest-v1.json",
            "single_gpu_report": "s1-7-single-gpu-report-v1.json",
            "gradient_bundle": "s1-7-gradient-bundle-v1.json",
            "comparison_table": "s1-7-comparison-table-v1.json",
            "gate_record": "s1-7-gate-record-v1.json",
            "replay": "s1-7-replay-validation-v1.json",
            "validation": "s1-7-validation-v1.json",
            "index": "s1-7-formalization-index-v1.json",
        }
        if set(objects) != set(filenames):
            raise Stage1PrecisionError("S1_9_S1_7_FROZEN_SCHEMA_ROLE_SET_INVALID")
        for role, filename in filenames.items():
            schema = registry[filename]
            module._validate_schema(objects[role], schema, registry, document=schema, path=role)
    except Stage1PrecisionError:
        raise
    except Exception as error:
        raise Stage1PrecisionError("S1_9_S1_7_FROZEN_SCHEMA_VALIDATION_FAILED") from error


def validate_s1_7_handoff(data_root: str | Path, index_ref: str) -> dict[str, Any]:
    """Strictly consume the immutable S1.7 producer without rewriting it."""

    root = Path(data_root).resolve(strict=True)
    index_path = _safe_reference(root, index_ref, field="index")
    if not index_path.is_file() or _sha256_file(index_path) != EXPECTED_S1_7_INDEX_SHA256:
        raise Stage1PrecisionError("S1_9_S1_7_INDEX_SHA256_MISMATCH")
    index = load_canonical_json(index_path)
    if not isinstance(index, Mapping):
        raise Stage1PrecisionError("S1_9_S1_7_INDEX_NOT_OBJECT")
    body = dict(index); supplied = body.pop("artifact_hash", None)
    expected_next = {"stage1.08_ddp_and_gradient_accumulation", TASK_ID}
    if supplied != canonical_json_hash(body) or index.get("schema_version") != "stage1-s1-7-formalization-index-v1" or index.get("status") != "PASS" or index.get("gate_id") != "G1-SINGLE" or index.get("task_id") != "stage1.07_single_gpu_pythia14m" or index.get("generator_git_commit") != EXPECTED_S1_7_PRODUCER or index.get("consumer_git_commit") != EXPECTED_S1_7_PRODUCER or index.get("artifact_hash") != EXPECTED_S1_7_INDEX_ARTIFACT_HASH or index.get("gate_artifact_hash") != EXPECTED_S1_7_GATE_HASH or set(index.get("next_task_ids", [])) != expected_next:
        raise Stage1PrecisionError("S1_9_S1_7_INDEX_SEMANTIC_BINDING_INVALID")
    refs, hashes = index.get("role_refs"), index.get("role_sha256")
    required = set(_S1_7_ROLE_SELF_HASH_FIELDS)
    if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or set(refs) != required or set(hashes) != required:
        raise Stage1PrecisionError("S1_9_S1_7_ROLE_SET_INVALID")
    roles: dict[str, Mapping[str, Any]] = {}
    for role in sorted(required):
        path = _safe_reference(root, refs[role], field=role)
        if not path.is_file():
            path = (index_path.parent / str(refs[role])).resolve()
        if not path.is_file() or not isinstance(hashes[role], str) or _sha256_file(path) != hashes[role]:
            raise Stage1PrecisionError(f"S1_9_S1_7_ROLE_HASH_INVALID:{role}")
        role_value = load_canonical_json(path)
        if not isinstance(role_value, Mapping):
            raise Stage1PrecisionError(f"S1_9_S1_7_ROLE_NOT_OBJECT:{role}")
        hash_field = _S1_7_ROLE_SELF_HASH_FIELDS[role]
        role_body = dict(role_value); role_hash = role_body.pop(hash_field, None)
        if not isinstance(role_hash, str) or role_hash != canonical_json_hash(role_body):
            raise Stage1PrecisionError(f"S1_9_S1_7_ROLE_SELF_HASH_INVALID:{role}")
        roles[role] = role_value
    gate = roles["gate_record"]
    gate_requirements = gate.get("requirements")
    if gate.get("status") != "PASS" or gate.get("artifact_hash") != EXPECTED_S1_7_GATE_HASH or not isinstance(gate_requirements, Mapping) or not gate_requirements or not all(value is True for value in gate_requirements.values()):
        raise Stage1PrecisionError("S1_9_S1_7_GATE_SEMANTIC_INVALID")
    auxiliaries: dict[str, Mapping[str, Any]] = {}
    for role, ref_key, sha_key, hash_field in (("replay", "replay_ref", "replay_sha256", "replay_hash"), ("validation", "validation_ref", "validation_sha256", "artifact_hash")):
        path = _safe_reference(root, index.get(ref_key), field=role)
        if not path.is_file(): path = (index_path.parent / str(index.get(ref_key))).resolve()
        if not path.is_file() or not isinstance(index.get(sha_key), str) or _sha256_file(path) != index[sha_key]:
            raise Stage1PrecisionError(f"S1_9_S1_7_AUXILIARY_HASH_INVALID:{role}")
        auxiliary = load_canonical_json(path)
        if not isinstance(auxiliary, Mapping) or auxiliary.get("status") != "PASS":
            raise Stage1PrecisionError(f"S1_9_S1_7_AUXILIARY_STATUS_INVALID:{role}")
        auxiliary_body = dict(auxiliary); auxiliary_hash = auxiliary_body.pop(hash_field, None)
        if not isinstance(auxiliary_hash, str) or auxiliary_hash != canonical_json_hash(auxiliary_body):
            raise Stage1PrecisionError(f"S1_9_S1_7_AUXILIARY_SELF_HASH_INVALID:{role}")
        auxiliaries[role] = auxiliary
    _validate_s1_7_frozen_schemas({"index": index, **roles, **auxiliaries})
    return {"s1_7_index_ref": index_ref, "s1_7_index_sha256": EXPECTED_S1_7_INDEX_SHA256, "s1_7_index_artifact_hash": EXPECTED_S1_7_INDEX_ARTIFACT_HASH, "s1_7_generator_commit": EXPECTED_S1_7_PRODUCER, "s1_7_gate_artifact_hash": EXPECTED_S1_7_GATE_HASH, "s1_7_role_sha256": dict(hashes)}


def validate_s1_8_handoff(data_root: str | Path, index_ref: str, *, expected_binding: Mapping[str, str] | None) -> dict[str, Any]:
    """Consume S1.8 only with a caller-pinned formal identity.

    S1.8 runs concurrently with this task, so its producer hashes must never
    be guessed here.  The formal caller must provide the immutable binding once
    S1.8 publishes it; an omitted, extra, malformed, or mismatched binding
    fails closed before a GPU lease can be acquired.
    """

    required_binding = {"index_sha256", "index_artifact_hash", "gate_artifact_hash", "producer_commit", "schema_version", "task_id", "gate_id"}
    if not isinstance(expected_binding, Mapping) or set(expected_binding) != required_binding or any(not isinstance(value, str) or not value for value in expected_binding.values()):
        raise Stage1PrecisionError("S1_9_S1_8_EXPECTED_BINDING_REQUIRED")
    root = Path(data_root).resolve(strict=True)
    index_path = _safe_reference(root, index_ref, field="s1_8_index")
    if not index_path.is_file() or _sha256_file(index_path) != expected_binding["index_sha256"]:
        raise Stage1PrecisionError("S1_9_S1_8_INDEX_SHA256_MISMATCH")
    index = load_canonical_json(index_path)
    if not isinstance(index, Mapping):
        raise Stage1PrecisionError("S1_9_S1_8_INDEX_NOT_OBJECT")
    body = dict(index); supplied = body.pop("artifact_hash", None)
    if supplied != canonical_json_hash(body) or index.get("status") != "PASS" or index.get("schema_version") != expected_binding["schema_version"] or index.get("task_id") != expected_binding["task_id"] or index.get("gate_id") != expected_binding["gate_id"] or index.get("generator_git_commit") != expected_binding["producer_commit"] or index.get("consumer_git_commit") != expected_binding["producer_commit"] or index.get("artifact_hash") != expected_binding["index_artifact_hash"] or index.get("gate_artifact_hash") != expected_binding["gate_artifact_hash"]:
        raise Stage1PrecisionError("S1_9_S1_8_INDEX_SEMANTIC_BINDING_INVALID")
    refs, hashes = index.get("role_refs"), index.get("role_sha256")
    required_roles = set(_S1_8_ROLE_HASH_AND_STATUS)
    if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or set(refs) != required_roles or set(hashes) != required_roles:
        raise Stage1PrecisionError("S1_9_S1_8_ROLE_SET_INVALID")
    roles: dict[str, Mapping[str, Any]] = {}
    for role in sorted(refs):
        path = _safe_reference(root, refs[role], field=f"s1_8.{role}")
        if not path.is_file(): path = (index_path.parent / str(refs[role])).resolve()
        if not path.is_file() or not isinstance(hashes[role], str) or _sha256_file(path) != hashes[role]:
            raise Stage1PrecisionError(f"S1_9_S1_8_ROLE_HASH_INVALID:{role}")
        role_value = load_canonical_json(path)
        if not isinstance(role_value, Mapping):
            raise Stage1PrecisionError(f"S1_9_S1_8_ROLE_NOT_OBJECT:{role}")
        hash_field, expected_status = _S1_8_ROLE_HASH_AND_STATUS[role]
        role_body = dict(role_value); role_hash = role_body.pop(hash_field, None)
        status_valid = "status" not in role_value if expected_status is None else role_value.get("status") == expected_status
        if not isinstance(role_hash, str) or role_hash != canonical_json_hash(role_body) or not status_valid:
            raise Stage1PrecisionError(f"S1_9_S1_8_ROLE_SEMANTIC_INVALID:{role}")
        roles[role] = role_value
    gate = roles["gate_record"]
    gate_requirements = gate.get("requirements")
    if gate.get("artifact_hash") != expected_binding["gate_artifact_hash"] or not isinstance(gate_requirements, Mapping) or not gate_requirements or not all(value is True for value in gate_requirements.values()):
        raise Stage1PrecisionError("S1_9_S1_8_GATE_SEMANTIC_INVALID")
    # S1.8 publishes its replay through the same canonical ``artifact_hash``
    # envelope as its other roles.  Do not reuse S1.7's older ``replay_hash``
    # wire here: accepting that synthetic shape would reject the real G1-DDP
    # handoff before S1.9 can acquire a lease.
    for role, ref_key, sha_key, hash_field in (("validation", "validation_ref", "validation_sha256", "artifact_hash"), ("replay", "replay_ref", "replay_sha256", "artifact_hash")):
        path = _safe_reference(root, index.get(ref_key), field=f"s1_8.{role}")
        if not path.is_file():
            path = (index_path.parent / str(index.get(ref_key))).resolve()
        if not path.is_file() or not isinstance(index.get(sha_key), str) or _sha256_file(path) != index[sha_key]:
            raise Stage1PrecisionError(f"S1_9_S1_8_AUXILIARY_HASH_INVALID:{role}")
        auxiliary = load_canonical_json(path)
        if not isinstance(auxiliary, Mapping) or auxiliary.get("status") != "PASS":
            raise Stage1PrecisionError(f"S1_9_S1_8_AUXILIARY_STATUS_INVALID:{role}")
        auxiliary_body = dict(auxiliary); auxiliary_hash = auxiliary_body.pop(hash_field, None)
        if not isinstance(auxiliary_hash, str) or auxiliary_hash != canonical_json_hash(auxiliary_body):
            raise Stage1PrecisionError(f"S1_9_S1_8_AUXILIARY_SELF_HASH_INVALID:{role}")
    return {"s1_8_index_ref": index_ref, "s1_8_index_sha256": expected_binding["index_sha256"], "s1_8_index_artifact_hash": expected_binding["index_artifact_hash"], "s1_8_generator_commit": expected_binding["producer_commit"], "s1_8_gate_artifact_hash": expected_binding["gate_artifact_hash"], "s1_8_role_sha256": dict(hashes)}


__all__ = ["EXPECTED_S1_7_GATE_HASH", "EXPECTED_S1_7_INDEX_ARTIFACT_HASH", "EXPECTED_S1_7_INDEX_SHA256", "EXPECTED_S1_7_PRODUCER", "GATE_ID", "REQUIREMENT_KEYS", "Stage1PrecisionError", "build_stage1_s19_evidence", "replay_stage1_s19_evidence", "run_stage1_s19_bf16_smoke", "validate_s1_7_handoff", "validate_s1_8_handoff", "validate_stage1_s19_evidence"]
