"""Independent FP64 oracle for the S1.6 training-step fixture.

This module intentionally uses only the Python standard library.  In
particular it does not import the production training engine, optimizer bridge,
accumulator, estimator kernels, ``torch`` or any Stage 1 production helper.
The formal replay uses this isolation to ensure that a shared implementation
mistake cannot validate its own training trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any



FIXTURE_ID = "stage1-s16-training-fixture-v1"
FIXTURE_SCHEMA = "stage1-s1-6-training-fixture-v1"
FROZEN_FIXTURE_HASH = "5bc3e3f87672a8311c98219fe666b146d3c9b9ffd13430d572b7691b679cca2e"


class Stage1TrainingOracleError(RuntimeError):
    """The frozen S1.6 fixture or its independent calculation is invalid."""


def _canonical_json_hash(value: object) -> str:
    """Minimal standalone canonical JSON identity for the frozen numeric input."""

    try:
        encoded = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise Stage1TrainingOracleError("S1_6_ORACLE_CANONICAL_JSON_INVALID") from error
    return hashlib.sha256(encoded).hexdigest()


def _require_object(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1TrainingOracleError(f"S1_6_ORACLE_OBJECT_REQUIRED:{field}")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1TrainingOracleError(f"S1_6_ORACLE_NUMBER_REQUIRED:{field}")
    result = float(value)
    if not math.isfinite(result):
        raise Stage1TrainingOracleError(f"S1_6_ORACLE_FINITE_REQUIRED:{field}")
    return result


def _vector_map(value: object, names: Sequence[str], *, field: str) -> dict[str, list[float]]:
    raw = _require_object(value, field=field)
    if set(raw) != set(names):
        raise Stage1TrainingOracleError(f"S1_6_ORACLE_COORDINATES_INVALID:{field}")
    result: dict[str, list[float]] = {}
    for name in names:
        source = raw[name]
        if not isinstance(source, list) or not source:
            raise Stage1TrainingOracleError(f"S1_6_ORACLE_VECTOR_REQUIRED:{field}.{name}")
        result[name] = [_number(item, field=f"{field}.{name}") for item in source]
    return result


def _same_shape(left: Mapping[str, list[float]], right: Mapping[str, list[float]], *, field: str) -> None:
    if set(left) != set(right) or any(len(left[name]) != len(right[name]) for name in left):
        raise Stage1TrainingOracleError(f"S1_6_ORACLE_SHAPE_MISMATCH:{field}")


def _zip(left: Mapping[str, list[float]], right: Mapping[str, list[float]], operation: str) -> dict[str, list[float]]:
    _same_shape(left, right, field=operation)
    if operation == "add":
        return {name: [a + b for a, b in zip(left[name], right[name], strict=True)] for name in left}
    if operation == "mul":
        return {name: [a * b for a, b in zip(left[name], right[name], strict=True)] for name in left}
    raise AssertionError(operation)


def _scale(value: Mapping[str, list[float]], scalar: float) -> dict[str, list[float]]:
    return {name: [item * scalar for item in items] for name, items in value.items()}


def _zeros_like(value: Mapping[str, list[float]]) -> dict[str, list[float]]:
    return {name: [0.0 for _ in items] for name, items in value.items()}


def _copy_vector_map(value: Mapping[str, list[float]]) -> dict[str, list[float]]:
    """Detach a wire snapshot from mutable oracle accumulator storage."""

    return {name: [float(item) for item in items] for name, items in value.items()}


def _mean(samples: Sequence[Mapping[str, list[float]]]) -> dict[str, list[float]]:
    if len(samples) < 2:
        raise Stage1TrainingOracleError("S1_6_ORACLE_REQUIRES_TWO_MICROBATCHES")
    result = _zeros_like(samples[0])
    for sample in samples:
        result = _zip(result, sample, "add")
    return _scale(result, 1.0 / len(samples))


def _nonnegative_part(value: Mapping[str, list[float]], *, negative: bool) -> dict[str, list[float]]:
    return {
        name: [max(-item, 0.0) if negative else max(item, 0.0) for item in items]
        for name, items in value.items()
    }


def _abs(value: Mapping[str, list[float]]) -> dict[str, list[float]]:
    return {name: [abs(item) for item in items] for name, items in value.items()}


def _add_inplace(destination: dict[str, list[float]], source: Mapping[str, list[float]]) -> None:
    _same_shape(destination, source, field="accumulation")
    for name in destination:
        destination[name] = [
            left + right for left, right in zip(destination[name], source[name], strict=True)
        ]


def load_stage1_s16_fixture(source_root: str | Path) -> dict[str, Any]:
    """Load and strictly validate the numeric fixture without production code."""

    path = Path(source_root) / "fixtures" / "stage1" / f"{FIXTURE_ID}.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1TrainingOracleError("S1_6_ORACLE_FIXTURE_UNREADABLE") from error
    value = _require_object(raw, field="fixture")
    expected = {"schema_version", "fixture_id", "coordinate_order", "parameter_to_group", "sgd", "adamw_cases", "tiny_transformer", "scaler", "skip", "clip", "fixture_hash"}
    if set(value) != expected or value.get("schema_version") != FIXTURE_SCHEMA or value.get("fixture_id") != FIXTURE_ID:
        raise Stage1TrainingOracleError("S1_6_ORACLE_FIXTURE_ID_OR_FIELDS_INVALID")
    fixture_body = dict(value)
    fixture_hash = fixture_body.pop("fixture_hash")
    if not isinstance(fixture_hash, str) or fixture_hash != FROZEN_FIXTURE_HASH or fixture_hash != _canonical_json_hash(fixture_body):
        raise Stage1TrainingOracleError("S1_6_ORACLE_FIXTURE_HASH_INVALID")
    names = value["coordinate_order"]
    if not isinstance(names, list) or names != ["left", "right"]:
        raise Stage1TrainingOracleError("S1_6_ORACLE_COORDINATE_ORDER_INVALID")
    parameter_to_group = value["parameter_to_group"]
    if parameter_to_group != {"left": "group_0000", "right": "group_0001"}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_PARAMETER_GROUP_MAP_INVALID")
    sgd = _require_object(value["sgd"], field="sgd")
    if set(sgd) != {"initial_parameters", "steps"}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_SGD_FIELDS_INVALID")
    _vector_map(sgd["initial_parameters"], names, field="sgd.initial_parameters")
    steps = sgd["steps"]
    if not isinstance(steps, list) or len(steps) != 2:
        raise Stage1TrainingOracleError("S1_6_ORACLE_SGD_STEP_COUNT_INVALID")
    for index, step in enumerate(steps):
        step_value = _require_object(step, field=f"sgd.steps[{index}]")
        if set(step_value) != {"learning_rates", "microbatch_gradients"}:
            raise Stage1TrainingOracleError("S1_6_ORACLE_SGD_STEP_FIELDS_INVALID")
        rates = _require_object(step_value["learning_rates"], field=f"sgd.steps[{index}].learning_rates")
        if set(rates) != {"group_0000", "group_0001"} or any(_number(rates[name], field=f"lr.{name}") < 0.0 for name in rates):
            raise Stage1TrainingOracleError("S1_6_ORACLE_SGD_LEARNING_RATES_INVALID")
        gradients = step_value["microbatch_gradients"]
        if not isinstance(gradients, list) or len(gradients) != 2:
            raise Stage1TrainingOracleError("S1_6_ORACLE_MICROBATCH_COUNT_INVALID")
        initial = _vector_map(sgd["initial_parameters"], names, field="sgd.initial_parameters")
        for micro_index, gradient in enumerate(gradients):
            mapped = _vector_map(gradient, names, field=f"sgd.steps[{index}].microbatch_gradients[{micro_index}]")
            _same_shape(initial, mapped, field="sgd.gradient")
    cases = value["adamw_cases"]
    if not isinstance(cases, list) or len(cases) != 2:
        raise Stage1TrainingOracleError("S1_6_ORACLE_ADAMW_CASE_COUNT_INVALID")
    required_adamw = {"case_id", "initial_parameter", "learning_rate", "weight_decay", "betas", "epsilon", "gradients"}
    seen_cases: set[str] = set()
    for case_index, raw_case in enumerate(cases):
        adamw = _require_object(raw_case, field=f"adamw_cases[{case_index}]")
        if set(adamw) != required_adamw or adamw.get("case_id") not in {"zero_weight_decay", "decoupled_weight_decay"}:
            raise Stage1TrainingOracleError("S1_6_ORACLE_ADAMW_FIELDS_INVALID")
        case_id = str(adamw["case_id"])
        seen_cases.add(case_id)
        for field in ("initial_parameter", "learning_rate", "weight_decay", "epsilon"):
            _number(adamw[field], field=f"adamw.{field}")
        betas = adamw["betas"]
        gradients = adamw["gradients"]
        if not isinstance(betas, list) or len(betas) != 2 or not all(0.0 <= _number(item, field="adamw.beta") < 1.0 for item in betas):
            raise Stage1TrainingOracleError("S1_6_ORACLE_ADAMW_BETAS_INVALID")
        if not isinstance(gradients, list) or len(gradients) != 2:
            raise Stage1TrainingOracleError("S1_6_ORACLE_ADAMW_GRADIENTS_INVALID")
        for gradient in gradients:
            _number(gradient, field="adamw.gradient")
    if seen_cases != {"zero_weight_decay", "decoupled_weight_decay"}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_ADAMW_CASE_IDS_INVALID")
    tiny = _require_object(value["tiny_transformer"], field="tiny_transformer")
    if set(tiny) != {"profile", "architecture", "initialization", "optimizer", "scheduler", "cursor_seed", "rng_seed", "microbatches"} or tiny.get("profile") != "T32_SINGLE":
        raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_FIELDS_INVALID")
    architecture = _require_object(tiny["architecture"], field="tiny.architecture")
    initialization = _require_object(tiny["initialization"], field="tiny.initialization")
    optimizer = _require_object(tiny["optimizer"], field="tiny.optimizer")
    scheduler = _require_object(tiny["scheduler"], field="tiny.scheduler")
    if architecture != {"vocab_size": 17, "hidden_size": 8, "num_heads": 2, "max_length": 6, "mlp_multiplier": 2, "dropout": 0.0} or initialization != {"rule": "arithmetic-v1", "scale": 0.0005, "offset_modulus": 23, "bias": -0.004} or optimizer != {"name": "AdamW", "learning_rate": 0.05, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "foreach": False} or scheduler != {"name": "StepLR", "step_size": 1, "gamma": 0.8} or tiny.get("cursor_seed") != 1 or tiny.get("rng_seed") != 6106:
        raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_CONTRACT_INVALID")
    micro_steps = tiny["microbatches"]
    if not isinstance(micro_steps, list) or len(micro_steps) != 2:
        raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_STEP_COUNT_INVALID")
    seen_batch_ids: set[str] = set()
    for step_index, micros in enumerate(micro_steps):
        if not isinstance(micros, list) or len(micros) != 2:
            raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_MICRO_COUNT_INVALID")
        for micro_index, raw_micro in enumerate(micros):
            micro = _require_object(raw_micro, field=f"tiny.microbatches[{step_index}][{micro_index}]")
            expected_id = f"s16-transformer-step-{step_index}-micro-{micro_index}"
            if set(micro) != {"batch_id", "input_ids", "labels", "sample_ids"} or micro.get("batch_id") != expected_id or expected_id in seen_batch_ids:
                raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_MICRO_ID_INVALID")
            seen_batch_ids.add(expected_id)
            input_ids, labels, sample_ids = micro["input_ids"], micro["labels"], micro["sample_ids"]
            if not isinstance(input_ids, list) or not isinstance(labels, list) or input_ids != labels or len(input_ids) != 12 or any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item < 17 for item in input_ids) or not isinstance(sample_ids, list) or len(sample_ids) != 2 or len(set(sample_ids)) != 2 or not all(isinstance(item, str) and item for item in sample_ids):
                raise Stage1TrainingOracleError("S1_6_ORACLE_TINY_MICRO_CONTENT_INVALID")
    scaler = _require_object(value["scaler"], field="scaler")
    if scaler != {"device": "cpu", "enabled": True, "init_scale": 8.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 1, "single_process_found_inf": "local_is_global", "tiny_scale_before": [8.0, 16.0], "tiny_scale_after": [16.0, 32.0], "skip_scale_before": [8.0, 16.0, 8.0], "skip_scale_after": [16.0, 8.0, 16.0]}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_SCALER_CONTRACT_INVALID")
    skip = _require_object(value["skip"], field="skip")
    if set(skip) != {"nonfinite_step_index", "consumes_batch", "long_term_fields_unchanged"}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_SKIP_FIELDS_INVALID")
    if skip["nonfinite_step_index"] != 1 or skip["consumes_batch"] is not True or not isinstance(skip["long_term_fields_unchanged"], list):
        raise Stage1TrainingOracleError("S1_6_ORACLE_SKIP_CONTRACT_INVALID")
    if value["clip"] != {"max_grad_norm": 0.5, "mean_gradient": {"left": [3.0], "right": [4.0]}, "global_norm": 5.0, "clip_epsilon": 1e-12, "clip_factor": 0.09999999999998}:
        raise Stage1TrainingOracleError("S1_6_ORACLE_CLIP_CONTRACT_INVALID")
    return dict(value)


def build_stage1_s16_oracle(source_root: str | Path) -> dict[str, Any]:
    """Compute SGD/AdamW and all long-horizon views with scalar FP64 loops."""

    fixture = load_stage1_s16_fixture(source_root)
    names = tuple(fixture["coordinate_order"])
    sgd = _require_object(fixture["sgd"], field="sgd")
    parameters = _vector_map(sgd["initial_parameters"], names, field="sgd.initial_parameters")
    zeros = _zeros_like(parameters)
    positive = _zeros_like(parameters)
    negative = _zeros_like(parameters)
    raw_total = _zeros_like(parameters)
    data_movement = _zeros_like(parameters)
    data_displacement = _zeros_like(parameters)
    actual_total = _zeros_like(parameters)
    total_movement = _zeros_like(parameters)
    initial_parameters = {name: list(values) for name, values in parameters.items()}

    def sgd_accumulator_views() -> dict[str, dict[str, list[float]]]:
        return {
            "signed": _zip(positive, _scale(negative, -1.0), "add"),
            "positive": _copy_vector_map(positive), "negative_mass": _copy_vector_map(negative),
            "absolute": _zip(positive, negative, "add"),
            "raw": _copy_vector_map(raw_total), "raw_clipped": _copy_vector_map(raw_total),
            "data_movement": _copy_vector_map(data_movement), "net_data_movement": _abs(data_displacement),
            "total_movement": _copy_vector_map(total_movement),
            "total_endpoint_movement": _abs(data_displacement),
            "weight_decay_movement": _zeros_like(parameters),
            "net_weight_decay_movement": _zeros_like(parameters),
            "actual_update_raw_importance": _copy_vector_map(actual_total),
            "magnitude": _abs(parameters), "initial_parameters": _copy_vector_map(initial_parameters),
            "last_parameters": _copy_vector_map(parameters),
        }

    previous_sgd_views = sgd_accumulator_views()
    trace: list[dict[str, Any]] = []
    for index, raw_step in enumerate(sgd["steps"]):
        step = _require_object(raw_step, field=f"sgd.steps[{index}]")
        samples = [
            _vector_map(item, names, field=f"sgd.steps[{index}].microbatch_gradients")
            for item in step["microbatch_gradients"]
        ]
        mean = _mean(samples)
        raw_core = _zip(mean, mean, "mul")
        u_core = _zip(samples[0], samples[1], "mul")
        rates_raw = _require_object(step["learning_rates"], field="learning_rates")
        rates = {name: _number(rates_raw[name], field=f"learning_rates.{name}") for name in rates_raw}
        parameter_to_group = fixture["parameter_to_group"]
        raw_score = {name: [value * rates[parameter_to_group[name]] for value in raw_core[name]] for name in names}
        u_score = {name: [value * rates[parameter_to_group[name]] for value in u_core[name]] for name in names}
        data_delta = {name: [-rates[parameter_to_group[name]] * value for value in mean[name]] for name in names}
        actual_update = _scale(_zip(data_delta, mean, "mul"), -1.0)
        parameters = _zip(parameters, data_delta, "add")
        _add_inplace(positive, _nonnegative_part(u_score, negative=False))
        _add_inplace(negative, _nonnegative_part(u_score, negative=True))
        _add_inplace(raw_total, raw_score)
        _add_inplace(data_movement, _abs(data_delta))
        _add_inplace(total_movement, _abs(data_delta))
        _add_inplace(data_displacement, data_delta)
        _add_inplace(actual_total, actual_update)
        accumulator_after = sgd_accumulator_views()
        accumulator_interval = {
            field: {
                name: [value - prior for value, prior in zip(accumulator_after[field][name], previous_sgd_views[field][name], strict=True)]
                for name in names
            }
            for field in accumulator_after
        }
        previous_sgd_views = accumulator_after
        trace.append({
            "step": index + 1,
            "learning_rates": rates,
            "mean_gradient": mean,
            "raw_core": raw_core,
            "u_core": u_core,
            "raw_score": raw_score,
            "u_score": u_score,
            "data_delta": data_delta,
            "weight_decay_delta": _zeros_like(data_delta),
            "total_delta": data_delta,
            "actual_update_raw_importance": actual_update,
            "parameters_post": parameters,
            "accumulator_after": accumulator_after,
            "accumulator_interval_delta": accumulator_interval,
        })
    sgd_summary = {
        **{field: value for field, value in sgd_accumulator_views().items() if field not in {"initial_parameters", "last_parameters"}},
        "parameters_final": parameters,
    }
    adamw_trace: list[dict[str, Any]] = []
    for raw_case in fixture["adamw_cases"]:
        adamw = _require_object(raw_case, field="adamw_case")
        beta1, beta2 = (_number(item, field="adamw.betas") for item in adamw["betas"])
        lr = _number(adamw["learning_rate"], field="adamw.learning_rate")
        wd = _number(adamw["weight_decay"], field="adamw.weight_decay")
        epsilon = _number(adamw["epsilon"], field="adamw.epsilon")
        parameter = _number(adamw["initial_parameter"], field="adamw.initial_parameter")
        initial_parameter = parameter
        first = second = 0.0
        positive = negative = raw_total = raw_clipped_total = 0.0
        data_movement = net_data = total_movement = net_total = 0.0
        decay_movement = net_decay = actual_total = 0.0

        def accumulator_views() -> dict[str, dict[str, list[float]]]:
            return {
                "signed": {"weight": [positive - negative]},
                "positive": {"weight": [positive]},
                "negative_mass": {"weight": [negative]},
                "absolute": {"weight": [positive + negative]},
                "raw": {"weight": [raw_total]},
                "raw_clipped": {"weight": [raw_clipped_total]},
                "data_movement": {"weight": [data_movement]},
                "net_data_movement": {"weight": [abs(net_data)]},
                "total_movement": {"weight": [total_movement]},
                "total_endpoint_movement": {"weight": [abs(parameter - initial_parameter)]},
                "weight_decay_movement": {"weight": [decay_movement]},
                "net_weight_decay_movement": {"weight": [abs(net_decay)]},
                "actual_update_raw_importance": {"weight": [actual_total]},
                "magnitude": {"weight": [abs(parameter)]},
                "initial_parameters": {"weight": [initial_parameter]},
                "last_parameters": {"weight": [parameter]},
            }

        previous_views = accumulator_views()
        for step, gradient_value in enumerate(adamw["gradients"], start=1):
            gradient = _number(gradient_value, field="adamw.gradient")
            before = parameter
            pre_first, pre_second, pre_step = first, second, step - 1
            first = beta1 * first + (1.0 - beta1) * gradient
            second = beta2 * second + (1.0 - beta2) * gradient * gradient
            data_delta = -lr * (first / (1.0 - beta1**step)) / (
                math.sqrt(second / (1.0 - beta2**step)) + epsilon
            )
            decay_delta = -lr * wd * before
            total_delta = data_delta + decay_delta
            parameter = before + total_delta
            score = lr * gradient * gradient
            positive += max(score, 0.0)
            negative += max(-score, 0.0)
            raw_total += score
            raw_clipped_total += score
            data_movement += abs(data_delta)
            net_data += data_delta
            total_movement += abs(total_delta)
            net_total += total_delta
            decay_movement += abs(decay_delta)
            net_decay += decay_delta
            actual_total += -data_delta * gradient
            views = accumulator_views()
            interval = {
                field: {"weight": [views[field]["weight"][0] - previous_views[field]["weight"][0]]}
                for field in views
            }
            previous_views = views
            adamw_trace.append({
                "case_id": adamw["case_id"], "step": float(step), "parameter_pre": before,
                "gradient": gradient, "pre_exp_avg": pre_first, "pre_exp_avg_sq": pre_second,
                "pre_optimizer_step": pre_step, "exp_avg": first, "exp_avg_sq": second,
                "optimizer_step": step, "data_delta": data_delta,
                "weight_decay_delta": decay_delta, "total_delta": total_delta,
                "parameter_post": parameter,
                "accumulator_after": views, "accumulator_interval_delta": interval,
                # The independent scalar state is itself the canonical v3
                # reload result: no production accumulator is imported here.
                "v3_roundtrip": views,
            })
    clip_factor = float(fixture["clip"]["max_grad_norm"]) / (
        float(fixture["clip"]["global_norm"]) + float(fixture["clip"]["clip_epsilon"])
    )
    clip_raw = {"left": [0.9], "right": [3.2]}
    clip_mean = fixture["clip"]["mean_gradient"]
    if not isinstance(clip_mean, Mapping):
        raise Stage1TrainingOracleError("S1_6_ORACLE_CLIP_MEAN_INVALID")
    clip_clipped = {name: [float(clip_raw[name][0]) * clip_factor] for name in ("left", "right")}
    clip_optimizer = {name: [float(clip_mean[name][0]) * clip_factor] for name in ("left", "right")}
    clip_delta = {"left": [-0.1 * clip_optimizer["left"][0]], "right": [-0.2 * clip_optimizer["right"][0]]}
    return {
        "schema_version": "stage1-s1-6-training-oracle-v1",
        "fixture_id": FIXTURE_ID,
        "independent_implementation": True,
        "sgd_trace": trace,
        "sgd_summary": sgd_summary,
        "adamw_trace": adamw_trace,
        "skip_contract": fixture["skip"],
        "clip_oracle": {
            "clip_factor": clip_factor,
            "mean_gradient": {name: [float(clip_mean[name][0])] for name in ("left", "right")},
            "raw_unclipped": clip_raw,
            "raw_clipped": clip_clipped,
            "optimizer_gradient": clip_optimizer,
            "data_delta": clip_delta,
        },
    }


__all__ = [
    "FIXTURE_ID",
    "FIXTURE_SCHEMA",
    "Stage1TrainingOracleError",
    "build_stage1_s16_oracle",
    "load_stage1_s16_fixture",
]
