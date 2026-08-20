"""Independent FP64/Python oracle for S1.9 numerical-boundary evidence.

This module deliberately has no import from the production estimator, runtime,
or optimizer modules.  It is kept to scalar/list arithmetic so a shared tensor
kernel cannot make both the implementation and its reference agree on the same
mistake.
"""

from __future__ import annotations

from hashlib import sha256
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ID = "stage1.09_precision_clipping_and_optimizer_boundaries"
FIXTURE_ID = "stage1-s19-precision-fixture-v1"
# The fixture is an immutable v1 test vector, not a user-supplied tuning file.
# Keep this independently of the in-file ``fixture_hash`` so a coordinated
# edit of a value and that self-hash cannot silently redefine the oracle.
FROZEN_FIXTURE_BODY_SHA256 = "3f0cc79bc6de8837c015a3c74f21bfa7aa2d3114dd4fb9cb036a9e1b9bd18dfd"


class Stage1PrecisionOracleError(RuntimeError):
    pass


def _canonical_hash(value: object) -> str:
    # Mirror the repository's canonical wire representation without importing
    # its production JSON helper into the isolated oracle: compact sorted
    # UTF-8 JSON plus exactly one terminal LF.
    return sha256((json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()


def _vectors(value: object, *, field: str) -> dict[str, list[float]]:
    if not isinstance(value, Mapping) or not value:
        raise Stage1PrecisionOracleError(f"S1_9_ORACLE_{field}_NOT_OBJECT")
    result: dict[str, list[float]] = {}
    for name, numbers in value.items():
        if not isinstance(name, str) or not name or not isinstance(numbers, list) or not numbers:
            raise Stage1PrecisionOracleError(f"S1_9_ORACLE_{field}_INVALID")
        converted = [float(number) for number in numbers]
        if not all(math.isfinite(number) for number in converted):
            raise Stage1PrecisionOracleError(f"S1_9_ORACLE_{field}_NONFINITE")
        result[name] = converted
    return result


def _validate_s1_9_fixture_schema(source_root: Path, value: object) -> None:
    """Use the project's strict stdlib schema subset for the frozen fixture.

    The scalar oracle deliberately remains independent of numerical runtime
    code, but a hashed object alone is not enough: fixture IDs, nested keys,
    and exact frozen cardinalities must be rejected before any arithmetic.
    """

    from param_importance_nlp.contracts.jsonio import loads_strict_json

    schema_path = source_root / "schemas" / "stage1" / "s1-9-precision-fixture-v1.json"
    validator_path = source_root / "ops" / "stage1" / "formalize_s1_6.py"
    if not schema_path.is_file() or not validator_path.is_file():
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_SCHEMA_RESOURCES_MISSING")
    try:
        schema = loads_strict_json(schema_path.read_bytes())
        if not isinstance(schema, Mapping) or not isinstance(schema.get("$id"), str):
            raise TypeError("schema")
        spec = importlib.util.spec_from_file_location("_s19_fixture_schema_subset", validator_path)
        if spec is None or spec.loader is None:
            raise TypeError("validator")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        registry = {schema_path.name: schema, str(schema["$id"]): schema}
        module._validate_schema(value, schema, registry, document=schema, path="s1_9_fixture")
    except Stage1PrecisionOracleError:
        raise
    except Exception as error:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_SCHEMA_VALIDATION_FAILED") from error


def _validate_frozen_fixture_body(value: Mapping[str, Any]) -> None:
    """Reject a coordinated fixture edit even when its self-hash is recomputed."""

    body = dict(value)
    supplied = body.pop("fixture_hash", None)
    observed_body_hash = _canonical_hash(body)
    if not isinstance(supplied, str) or supplied != observed_body_hash:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_HASH_INVALID")
    if supplied != FROZEN_FIXTURE_BODY_SHA256 or observed_body_hash != FROZEN_FIXTURE_BODY_SHA256:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_FROZEN_BODY_HASH_MISMATCH")


def load_stage1_s19_fixture(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    path = root / "fixtures" / "stage1" / "stage1-s19-precision-fixture-v1.json"
    try:
        from param_importance_nlp.contracts.jsonio import loads_strict_json

        value = loads_strict_json(path.read_bytes())
    except Exception as error:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_LOAD_FAILED") from error
    if not isinstance(value, dict):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_NOT_OBJECT")
    _validate_s1_9_fixture_schema(root, value)
    expected = {"schema_version", "fixture_id", "coordinates", "micro_gradients", "loss_scales", "learning_rates", "statistical_weights", "mixed_magnitude_gradients", "autocast_regression", "clip", "sgd", "adamw", "bf16", "fixture_hash"}
    if set(value) != expected or value.get("schema_version") != FIXTURE_ID or value.get("fixture_id") != FIXTURE_ID:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_FIXTURE_FIELDS_INVALID")
    _validate_frozen_fixture_body(value)
    _vectors(value["coordinates"], field="coordinates")
    if not isinstance(value["micro_gradients"], list) or len(value["micro_gradients"]) < 2:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_MICROBATCHES_INVALID")
    gradients = [_vectors(item, field="micro_gradient") for item in value["micro_gradients"]]
    names = set(gradients[0])
    shapes = {name: len(values) for name, values in gradients[0].items()}
    if any(set(item) != names or any(len(item[name]) != shapes[name] for name in names) for item in gradients[1:]):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_MICROBATCH_SHAPES_INVALID")
    scales = value["loss_scales"]
    if not isinstance(scales, list) or not scales or any(not math.isfinite(float(scale)) or float(scale) <= 0.0 for scale in scales):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_LOSS_SCALES_INVALID")
    weights = value["statistical_weights"]
    if not isinstance(weights, list) or len(weights) != len(gradients) or any(not math.isfinite(float(weight)) or float(weight) <= 0.0 for weight in weights):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_STATISTICAL_WEIGHTS_INVALID")
    autocast = value["autocast_regression"]
    if not isinstance(autocast, Mapping) or set(autocast) != {"input_dim", "initial_weight", "initial_bias", "microbatches", "loss", "autocast_dtype"} or autocast.get("input_dim") != 2 or autocast.get("loss") != "half_squared_error" or autocast.get("autocast_dtype") != "torch.bfloat16":
        raise Stage1PrecisionOracleError("S1_9_ORACLE_AUTOCAST_FIXTURE_INVALID")
    if not isinstance(autocast.get("microbatches"), list) or len(autocast["microbatches"]) != len(gradients):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_AUTOCAST_MICROBATCHES_INVALID")
    return value


def _mean(samples: Sequence[Mapping[str, list[float]]]) -> dict[str, list[float]]:
    return {name: [sum(sample[name][index] for sample in samples) / len(samples) for index in range(len(samples[0][name]))] for name in samples[0]}


def _s1(samples: Sequence[Mapping[str, list[float]]]) -> dict[str, list[float]]:
    return {name: [sum(sample[name][index] for sample in samples) for index in range(len(samples[0][name]))] for name in samples[0]}


def _s2(samples: Sequence[Mapping[str, list[float]]]) -> dict[str, list[float]]:
    return {name: [sum(sample[name][index] ** 2 for sample in samples) for index in range(len(samples[0][name]))] for name in samples[0]}


def _weighted_statistics(samples: Sequence[Mapping[str, list[float]]], weights: Sequence[float]) -> dict[str, Any]:
    n1, n2 = sum(weights), sum(weight * weight for weight in weights)
    g1 = {name: [sum(weights[sample_index] * sample[name][index] for sample_index, sample in enumerate(samples)) for index in range(len(samples[0][name]))] for name in samples[0]}
    g2 = {name: [sum(weights[sample_index] ** 2 * sample[name][index] ** 2 for sample_index, sample in enumerate(samples)) for index in range(len(samples[0][name]))] for name in samples[0]}
    denominator = n1 * n1 - n2
    weighted_mean = {name: [g1[name][index] / n1 for index in range(len(g1[name]))] for name in g1}
    weighted_raw = {name: [value * value for value in values] for name, values in weighted_mean.items()}
    weighted_u = {name: [(g1[name][index] ** 2 - g2[name][index]) / denominator for index in range(len(g1[name]))] for name in g1}
    return {"g1": g1, "g2": g2, "n1": n1, "n2": n2, "weighted_mean_gradient": weighted_mean, "weighted_raw_core": weighted_raw, "weighted_u_core": weighted_u}


def _u(samples: Sequence[Mapping[str, list[float]]]) -> dict[str, list[float]]:
    count = len(samples)
    return {
        name: [
            sum(samples[left][name][index] * samples[right][name][index] for left in range(count) for right in range(count) if left != right) / (count * (count - 1))
            for index in range(len(samples[0][name]))
        ]
        for name in samples[0]
    }


def _scale(values: Mapping[str, list[float]], scalar: float) -> dict[str, list[float]]:
    return {name: [item * scalar for item in items] for name, items in values.items()}


def _score(values: Mapping[str, list[float]], learning_rates: Mapping[str, object]) -> dict[str, list[float]]:
    return {name: _scale({name: items}, float(learning_rates[name]))[name] for name, items in values.items()}


def _adamw_step(
    *, initial: Sequence[float], gradient: Sequence[float], learning_rate: float,
    beta1: float, beta2: float, epsilon: float, weight_decay: float,
    previous_m: Sequence[float] | None = None, previous_v: Sequence[float] | None = None,
    previous_step: int = 0,
) -> dict[str, list[float] | int]:
    """Independent scalar AdamW math for the reset-parameter moment fixture."""

    if len(initial) != len(gradient):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_ADAMW_SHAPE_INVALID")
    m0 = [0.0] * len(initial) if previous_m is None else [float(value) for value in previous_m]
    v0 = [0.0] * len(initial) if previous_v is None else [float(value) for value in previous_v]
    if len(m0) != len(initial) or len(v0) != len(initial) or previous_step < 0:
        raise Stage1PrecisionOracleError("S1_9_ORACLE_ADAMW_STATE_INVALID")
    step = previous_step + 1
    m = [beta1 * m0[index] + (1.0 - beta1) * float(gradient[index]) for index in range(len(initial))]
    v = [beta2 * v0[index] + (1.0 - beta2) * float(gradient[index]) ** 2 for index in range(len(initial))]
    correction1, correction2 = 1.0 - beta1**step, 1.0 - beta2**step
    data = [-(learning_rate / correction1) * m[index] / (math.sqrt(v[index]) / math.sqrt(correction2) + epsilon) for index in range(len(initial))]
    decay = [-learning_rate * weight_decay * float(initial[index]) for index in range(len(initial))]
    return {"data_delta": data, "weight_decay_delta": decay, "total_delta": [data[index] + decay[index] for index in range(len(initial))], "exp_avg": m, "exp_avg_sq": v, "step": step}


def _adamw_cases(fixture: Mapping[str, Any]) -> dict[str, Any]:
    adamw = fixture["adamw"]
    if not isinstance(adamw, Mapping):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_ADAMW_INVALID")
    initial = [float(value) for value in adamw["initial"]]
    gradient = [float(value) for value in adamw["gradient"]]
    lr, epsilon, decay = float(adamw["learning_rate"]), float(adamw["eps"]), float(adamw["weight_decay"])
    beta1, beta2 = (float(value) for value in adamw["betas"])
    if not (len(initial) == len(gradient) and all(math.isfinite(value) for value in (*initial, *gradient, lr, epsilon, decay, beta1, beta2))):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_ADAMW_VALUES_INVALID")
    fresh = _adamw_step(initial=initial, gradient=gradient, learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=decay)
    priming = [float(value) for value in adamw["priming_gradient"]]
    prime = _adamw_step(initial=initial, gradient=priming, learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=decay)
    primed = _adamw_step(initial=initial, gradient=gradient, learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, weight_decay=decay, previous_m=prime["exp_avg"], previous_v=prime["exp_avg_sq"], previous_step=int(prime["step"]))
    return {"fresh": fresh, "primed_after_parameter_reset": primed, "current_gradient": gradient}


def build_stage1_s19_oracle(source_root: str | Path) -> dict[str, Any]:
    fixture = load_stage1_s19_fixture(source_root)
    samples = [_vectors(item, field="micro_gradient") for item in fixture["micro_gradients"]]
    mean, s1, s2, u = _mean(samples), _s1(samples), _s2(samples), _u(samples)
    weighted = _weighted_statistics(samples, [float(value) for value in fixture["statistical_weights"]])
    raw = {name: [value * value for value in values] for name, values in mean.items()}
    clip = fixture["clip"]
    if not isinstance(clip, Mapping) or set(clip) != {"max_norm", "eps", "near_zero_ratio"} or not all(math.isfinite(float(clip[key])) and float(clip[key]) > 0.0 for key in ("max_norm", "eps", "near_zero_ratio")):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_CLIP_INVALID")
    norm = math.sqrt(sum(value * value for values in mean.values() for value in values))
    factor = min(1.0, float(clip["max_norm"]) / (norm + float(clip["eps"])))
    learning_rates = fixture["learning_rates"]
    if not isinstance(learning_rates, Mapping) or set(learning_rates) != set(mean):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_LEARNING_RATES_INVALID")
    sgd = fixture["sgd"]
    if not isinstance(sgd, Mapping):
        raise Stage1PrecisionOracleError("S1_9_ORACLE_SGD_INVALID")
    sgd_delta = [-float(sgd["learning_rate"]) * float(value) for value in sgd["gradient"]]
    oracle = {
        "schema_version": "stage1-s1-9-oracle-bundle-v1",
        "fixture_id": FIXTURE_ID,
        "independent_implementation": {"formal_estimator_imported": False, "runtime_or_optimizer_imported": False, "arithmetic": "python_fp64_scalar_loops"},
        "statistics": {"mean_gradient": mean, "s1": s1, "s2": s2, "g1": weighted["g1"], "g2": weighted["g2"], "raw_core": raw, "u_core": u, "weighted_mean_gradient": weighted["weighted_mean_gradient"], "weighted_raw_core": weighted["weighted_raw_core"], "weighted_u_core": weighted["weighted_u_core"], "raw_score": _score(raw, learning_rates), "u_score": _score(u, learning_rates), "weighted_raw_score": _score(weighted["weighted_raw_core"], learning_rates), "weighted_u_score": _score(weighted["weighted_u_core"], learning_rates)},
        "loss_scale_contract": {"scales": [float(value) for value in fixture["loss_scales"]], "unscaled_statistics_equal": True, "negative_control_scales_quadratically": True},
        "clip": {"global_norm": norm, "clip_factor": factor, "u_clipped_score": _scale(_score(u, learning_rates), factor), "u_wrong_squared_factor": _scale(_score(u, learning_rates), factor * factor)},
        "optimizer": {"sgd_data_delta": sgd_delta, "adamw": _adamw_cases(fixture), "actual_update_uses_negative_data_delta_times_current_gradient": [-sgd_delta[index] * float(sgd["gradient"][index]) for index in range(len(sgd_delta))]},
    }
    oracle["oracle_hash"] = _canonical_hash(oracle)
    return oracle


__all__ = ["FIXTURE_ID", "FROZEN_FIXTURE_BODY_SHA256", "TASK_ID", "Stage1PrecisionOracleError", "build_stage1_s19_oracle", "load_stage1_s19_fixture"]
