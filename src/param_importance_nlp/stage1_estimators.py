"""Deterministic, replayable CPU evidence for S1.5 / G1-EST.

The production side uses the streaming estimator kernels.  The oracle side is
kept in :mod:`stage1_estimator_oracle` and performs explicit FP64 loops over
saved microbatch gradients.  This module is deliberately model-, optimizer-,
DDP-, and filesystem-independent; the formal publisher owns the immutable
filesystem transaction and the S1.4 handoff.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch

from .contracts.jsonio import JSONValue, canonical_json_hash
from .core.estimators import (
    DoubleSampleProvenance,
    EstimatorResult,
    NO_UNBIASEDNESS_CLAIM,
    PLUGIN_SAME_BATCH_CLIP,
    UNBIASED_FIXED_STATE,
    double_sample_importance,
    equal_u_importance,
    explicit_ordered_pair_u_reference,
    explicit_unordered_pair_u_reference,
    raw_importance,
    weighted_u_importance,
)
from .core.errors import CoreContractError, NumericalError, TensorMapError
from .core.oracles import compare_tensor_maps_fp64
from .core.sufficient_statistics import EqualSufficientStatistics, WeightedSufficientStatistics
from .core.tensors import TensorMap
from .core.registry import ParameterRegistry
from .stage1_estimator_oracle import (
    Stage1EstimatorOracleError,
    apply_learning_rates,
    double_core,
    mean_gradient,
    ordered_pair_u_core,
    raw_core,
    sufficient_statistics,
    tensor_map_from_wire,
    tensor_map_to_wire,
    unordered_pair_u_core,
    weighted_ordered_pair_u_core,
)


TASK_ID = "stage1.05_estimators"
GATE_ID = "G1-EST"
FIXTURE_ID = "stage1-s15-estimator-fixture-v1"
REPORT_SCHEMA = "stage1-g1-est-report-v1"
ORACLE_SCHEMA = "stage1-g1-est-oracle-report-v1"
BUNDLE_SCHEMA = "stage1-g1-est-tensor-bundle-v1"
TABLE_SCHEMA = "stage1-g1-est-comparison-table-v1"
GATE_SCHEMA = "stage1-g1-est-gate-record-v1"
INDEX_SCHEMA = "stage1-s1-5-formalization-index-v1"
VALIDATION_SCHEMA = "stage1-s1-5-validation-v1"
EXPECTED_S1_4_INDEX_SHA256 = "346c86daeb8cc61c9b4145891fc195fd3ffa29ae7a8ee28d6e670c27a0ee62c0"
EXPECTED_S1_4_GATE_ARTIFACT_HASH = "56e8e5d2128eb1f5c26d20ae7cfbcc4d01d1470cb5e55ede075f253e4f85b7d6"
EXPECTED_S1_4_PRODUCER_COMMIT = "92e3fa5ec286afa43c51be691895f9a7210199ff"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_PROFILES: Mapping[str, Mapping[str, float]] = {
    "T64_ORACLE": {"atol": 1e-12, "rtol": 1e-10, "normalized_l2_limit": 1e-10},
    "T32_SINGLE": {"atol": 1e-7, "rtol": 1e-5, "normalized_l2_limit": 1e-5},
}
_PROFILE_DTYPES = {"T64_ORACLE": torch.float64, "T32_SINGLE": torch.float32}
_CORE_SCALE = 16.0
_SCORE_SCALE = 4.0
_CLIP_FACTOR = 0.4
_LEARNING_RATES = {"group_0000": 0.25, "group_0001": 0.05}
_WEIGHTS = (1.0, 3.0, 2.0, 4.0)
_COMPARISON_IDS = (
    "raw_core",
    "raw_score",
    "raw_clipped_score",
    "double_core",
    "double_score",
    "equal_u_ordered_core",
    "equal_u_unordered_core",
    "equal_u_streaming_core",
    "equal_u_score",
    "weighted_u_core",
    "weighted_u_score",
    "weighted_equal_count_degenerate_core",
    "m2_u_equals_double_core",
    "double_exchange_core",
    "identical_gradient_u_core",
    "identical_gradient_u_score",
    "equal_u_reverse_permutation_core",
    "equal_u_rotate_permutation_core",
    "weighted_u_reverse_paired_permutation_core",
    "weighted_u_rotate_paired_permutation_core",
)
_REJECTION_IDS = (
    "m1_rejected",
    "same_batch_object_rejected",
    "shared_sampler_state_rejected",
    "nonfinite_rejected",
    "shape_mismatch_rejected",
    "registry_mismatch_rejected",
    "dtype_device_mismatch_rejected",
    "nonpositive_weight_rejected",
    "nonpositive_denominator_rejected",
    "negative_learning_rate_rejected",
    "nonfinite_learning_rate_rejected",
    "nonfinite_clip_rejected",
    "out_of_range_clip_rejected",
    "float16_accumulation_rejected",
)
_WEIGHTED_STATISTICAL_CONTRACT: Mapping[str, JSONValue] = {
    "statistical_unit": "microbatch",
    "weight_unit": "effective_target_tokens",
    "sampling_design": "with_replacement_product_sampling",
    "weights_exogenous": True,
    "common_mean_assumption": True,
}
_ESTIMATOR_PUBLIC_FIELDS: Mapping[str, Mapping[str, str]] = {
    "raw": {
        "estimator_name": "local_gradient_space_importance_raw",
        "unbiasedness_claim": NO_UNBIASEDNESS_CLAIM,
    },
    "raw_clipped": {
        "estimator_name": "local_gradient_space_importance_raw_clipped",
        "unbiasedness_claim": PLUGIN_SAME_BATCH_CLIP,
    },
    "double": {
        "estimator_name": "double_sample_gradient_importance",
        "unbiasedness_claim": UNBIASED_FIXED_STATE,
    },
    "equal_u": {
        "estimator_name": "local_gradient_space_importance_u",
        "unbiasedness_claim": UNBIASED_FIXED_STATE,
    },
    "weighted_u": {
        "estimator_name": "local_gradient_space_importance_u_weighted",
        "unbiasedness_claim": UNBIASED_FIXED_STATE,
    },
    "weighted_u_assumptions_false": {
        "estimator_name": "local_gradient_space_importance_u_weighted",
        "unbiasedness_claim": NO_UNBIASEDNESS_CLAIM,
    },
}
_SOURCE_FILES = (
    "src/param_importance_nlp/core/__init__.py",
    "src/param_importance_nlp/core/estimators.py",
    "src/param_importance_nlp/core/sufficient_statistics.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/stage1_estimator_oracle.py",
    "src/param_importance_nlp/stage1_estimators.py",
    "ops/stage1/formalize_s1_5.py",
    "schemas/stage1/g1-est-report-v1.json",
    "schemas/stage1/g1-est-oracle-report-v1.json",
    "schemas/stage1/g1-est-tensor-bundle-v1.json",
    "schemas/stage1/g1-est-comparison-table-v1.json",
    "schemas/stage1/g1-est-gate-record-v1.json",
    "schemas/stage1/s1-5-formalization-index-v1.json",
    "schemas/stage1/s1-5-validation-v1.json",
    "schemas/stage1/s1-5-fixture-manifest-v1.json",
    "fixtures/stage1/stage1-s15-estimator-fixture-v1.json",
    "tests/test_stage1_s15_estimators.py",
    "tests/test_stage1_s15_handoff_and_charts.py",
    "tests/test_stage01_task_runners.py",
)


class Stage1EstimatorError(RuntimeError):
    """The S1.5 fixture, evidence roles, or handoff contract is invalid."""


class _FixtureRegistryModel(torch.nn.Module):
    """Three fixed coordinates, with bias/weight intentionally sharing a group."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
        self.head = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
        self.weight = torch.nn.Parameter(torch.zeros(3, dtype=torch.float64))


def _fixture_registry() -> ParameterRegistry:
    model = _FixtureRegistryModel()
    optimizer = torch.optim.SGD(
        [
            {"params": [model.weight, model.bias], "lr": _LEARNING_RATES["group_0000"]},
            {"params": [model.head], "lr": _LEARNING_RATES["group_0001"]},
        ]
    )
    return ParameterRegistry.from_model(model, optimizer)


def _coordinate_contract(registry: ParameterRegistry) -> dict[str, JSONValue]:
    return {
        "identity": "stage1-s15-fixture-coordinate-contract-v1",
        "coordinate_order": list(registry.eligible_names),
        "coordinate_shapes": {name: list(registry.record(name).shape) for name in registry.eligible_names},
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "parameter_to_group": {name: registry.record(name).group_id for name in registry.eligible_names},
    }


def _estimator_input_contract(registry: ParameterRegistry) -> dict[str, JSONValue]:
    """Non-numerical call-site contract consumed by S1.6, never by a kernel."""

    return {
        "gradient_semantics": {
            "raw_double_boundary": "global_mean_gradient",
            "u_statistical_input": "local_microbatch_mean_gradient",
        },
        "gradient_scale_restored": True,
        "device": "cpu",
        "world_size": 1,
        "rank": 0,
        "reduction": "no_collective_local_mean",
        "collective": "none",
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "profile_dtypes": {"T64_ORACLE": "torch.float64", "T32_SINGLE": "torch.float32"},
    }


def _map(
    registry: ParameterRegistry,
    dtype: torch.dtype,
    weight: Sequence[float],
    bias: Sequence[float],
    head: Sequence[float],
) -> TensorMap:
    return TensorMap(
        {
            "weight": torch.tensor(weight, dtype=dtype),
            "bias": torch.tensor(bias, dtype=dtype),
            "head": torch.tensor(head, dtype=dtype),
        },
        registry=registry,
    )


def _hash_role(value: Mapping[str, Any], *, field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return canonical_json_hash(body)


def _require_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _COMMIT_RE.fullmatch(value) is None:
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_INVALID")
    return value


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_INVALID")
    return value


def _source_hashes(source_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for reference in _SOURCE_FILES:
        path = source_root / reference
        if not path.is_file():
            raise Stage1EstimatorError(f"S1_5_SOURCE_FILE_MISSING:{reference}")
        result[reference] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _json_number(value: object, *, field: str, positive: bool = False, nonnegative: bool = False) -> float:
    if type(value) not in {int, float}:
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_TYPE_INVALID")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0) or (nonnegative and result < 0.0):
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_VALUE_INVALID")
    return result


def _load_fixture_manifest(source_root: Path) -> tuple[dict[str, Any], str]:
    """Load the repository-owned immutable input contract before any estimate."""

    from .atomic import sha256_file
    from .contracts.jsonio import load_canonical_json
    path = source_root / "fixtures/stage1/stage1-s15-estimator-fixture-v1.json"
    if not path.is_file():
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_MISSING")
    loaded = load_canonical_json(path)
    if not isinstance(loaded, dict):
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_NOT_OBJECT")
    expected_keys = {"schema_version", "fixture_id", "coordinate_contract", "estimator_input_contract", "learning_rates", "clip_factor", "weights", "weighted_statistical_contract", "double_provenance", "double_sample_ids_a", "double_sample_ids_b", "microbatch_samples", "identical_gradient_samples", "permutation_orders", "double_samples_a", "double_samples_b", "negative_samples", "zero_samples", "manifest_hash"}
    if set(loaded) != expected_keys or loaded.get("schema_version") != "stage1-s1-5-fixture-manifest-v1" or loaded.get("fixture_id") != FIXTURE_ID:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_SCHEMA_INVALID")
    body = dict(loaded)
    manifest_hash = body.pop("manifest_hash", None)
    if not isinstance(manifest_hash, str) or manifest_hash != canonical_json_hash(body):
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_HASH_INVALID")
    registry = _fixture_registry()
    contract = _coordinate_contract(registry)
    if loaded.get("coordinate_contract") != contract:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_REGISTRY_INVALID")
    if loaded.get("estimator_input_contract") != _estimator_input_contract(registry):
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_INPUT_CONTRACT_INVALID")
    learning_rates = loaded.get("learning_rates")
    if not isinstance(learning_rates, Mapping) or set(learning_rates) != set(_LEARNING_RATES) or {name: _json_number(value, field=f"learning_rate_{name}", nonnegative=True) for name, value in learning_rates.items()} != _LEARNING_RATES:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_LR_INVALID")
    if _json_number(loaded.get("clip_factor"), field="clip_factor", nonnegative=True) != _CLIP_FACTOR:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_CLIP_INVALID")
    weights = loaded.get("weights")
    if not isinstance(weights, list) or tuple(_json_number(value, field="weight", positive=True) for value in weights) != _WEIGHTS:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_WEIGHTS_INVALID")
    if loaded.get("weighted_statistical_contract") != _WEIGHTED_STATISTICAL_CONTRACT:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_WEIGHTED_CONTRACT_INVALID")
    permutation_orders = loaded.get("permutation_orders")
    if permutation_orders != {"reverse": [3, 2, 1, 0], "rotate": [1, 2, 3, 0]}:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_PERMUTATION_ORDER_INVALID")
    provenance = loaded.get("double_provenance")
    if not isinstance(provenance, Mapping):
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_PROVENANCE_INVALID")
    try:
        DoubleSampleProvenance(**dict(provenance))
        sample_ids_a, sample_ids_b = loaded.get("double_sample_ids_a"), loaded.get("double_sample_ids_b")
        if not isinstance(sample_ids_a, list) or not isinstance(sample_ids_b, list) or len(sample_ids_a) != 2 or len(sample_ids_b) != 2 or any(not isinstance(value, str) or not value for value in (*sample_ids_a, *sample_ids_b)) or set(sample_ids_a) & set(sample_ids_b) != {"overlap-0"}:
            raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_SAMPLE_IDS_INVALID")
        parsed_by_field: dict[str, list[TensorMap]] = {}
        for field, expected_count in (("microbatch_samples", 4), ("identical_gradient_samples", 3), ("double_samples_a", 2), ("double_samples_b", 2), ("negative_samples", 2), ("zero_samples", 2)):
            entries = loaded.get(field)
            if not isinstance(entries, list) or len(entries) != expected_count:
                raise Stage1EstimatorError(f"S1_5_FIXTURE_MANIFEST_{field.upper()}_INVALID")
            parsed: list[TensorMap] = []
            for index, entry in enumerate(entries):
                parsed.append(tensor_map_from_wire(entry, field=f"manifest.{field}[{index}]", coordinate_order=contract["coordinate_order"], registry=registry))
            parsed_by_field[field] = parsed
        identical = parsed_by_field["identical_gradient_samples"]
        if any(not torch.equal(sample[name], identical[0][name]) or torch.count_nonzero(sample[name]).item() != sample[name].numel() for sample in identical for name in sample):
            raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_IDENTICAL_GRADIENT_INVALID")
    except (Stage1EstimatorOracleError, TypeError) as error:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_INPUT_INVALID") from error
    return loaded, sha256_file(path)


def _manifest_samples(manifest: Mapping[str, Any], field: str, *, dtype: torch.dtype, registry: ParameterRegistry) -> tuple[TensorMap, ...]:
    contract = _coordinate_contract(registry)
    entries = manifest.get(field)
    if not isinstance(entries, list):
        raise Stage1EstimatorError(f"S1_5_MANIFEST_{field.upper()}_INVALID")
    try:
        return tuple(tensor_map_from_wire(entry, field=f"manifest.{field}[{index}]", coordinate_order=contract["coordinate_order"], registry=registry).to(dtype=dtype) for index, entry in enumerate(entries))
    except Stage1EstimatorOracleError as error:
        raise Stage1EstimatorError(f"S1_5_MANIFEST_{field.upper()}_INVALID") from error


def _fixture_samples(dtype: torch.dtype, registry: ParameterRegistry) -> tuple[TensorMap, ...]:
    raw = (
        ((1.0, -2.0, 3.0), (0.5, -1.0), (0.25,)),
        ((-2.0, 4.0, 1.0), (1.5, 2.0), (-0.5,)),
        ((3.0, 0.5, -4.0), (-2.0, 0.25), (1.75,)),
        ((-1.0, -3.0, 2.0), (0.75, -1.5), (-1.25,)),
    )
    return tuple(
        _map(registry, dtype, weight, bias, head)
        for weight, bias, head in raw
    )


def _double_samples(dtype: torch.dtype, registry: ParameterRegistry) -> tuple[tuple[TensorMap, ...], tuple[TensorMap, ...]]:
    left = (
        _map(registry, dtype, (2.0, -1.0, 0.5), (1.0, -2.0), (0.5,)),
        _map(registry, dtype, (4.0, 3.0, -1.5), (-1.0, 2.0), (-0.75,)),
    )
    right = (
        _map(registry, dtype, (-3.0, 2.0, 1.5), (0.5, 1.0), (1.25,)),
        _map(registry, dtype, (1.0, 5.0, -0.5), (2.5, -1.0), (-0.25,)),
    )
    return left, right


def _negative_samples(dtype: torch.dtype, registry: ParameterRegistry) -> tuple[TensorMap, TensorMap]:
    return (
        _map(registry, dtype, (2.0, -3.0, 1.0), (1.0, -2.0), (0.5,)),
        _map(registry, dtype, (-2.0, 3.0, -1.0), (-1.0, 2.0), (-0.5,)),
    )


def _zero_samples(dtype: torch.dtype, registry: ParameterRegistry) -> tuple[TensorMap, TensorMap]:
    zero = _map(registry, dtype, (0.0, 0.0, 0.0), (0.0, 0.0), (0.0,))
    return zero, zero.clone()


def _wire_samples(samples: Sequence[TensorMap]) -> list[dict[str, dict[str, object]]]:
    return [tensor_map_to_wire(sample) for sample in samples]


def _unwire_samples(
    value: object,
    *,
    field: str,
    coordinate_contract: Mapping[str, Any],
    registry: ParameterRegistry,
) -> tuple[TensorMap, ...]:
    if not isinstance(value, list) or not value:
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_INVALID")
    try:
        return tuple(tensor_map_from_wire(item, field=f"{field}[{index}]", coordinate_order=coordinate_contract["coordinate_order"], registry=registry) for index, item in enumerate(value))
    except Stage1EstimatorOracleError as error:
        raise Stage1EstimatorError(f"S1_5_{field.upper()}_INVALID") from error


def _tensor_values(value: TensorMap) -> dict[str, list[float]]:
    return {name: [float(item) for item in tensor.detach().to(torch.float64).reshape(-1).tolist()] for name, tensor in value.items()}


def _comparison(
    *, profile: str, comparison_id: str, actual: TensorMap, oracle: TensorMap, natural_scale: float
) -> dict[str, JSONValue]:
    settings = _PROFILES[profile]
    global_result = compare_tensor_maps_fp64(
        actual, oracle, natural_scale=natural_scale, **settings
    ).to_dict()
    per_tensor: list[dict[str, JSONValue]] = []
    for name in actual:
        result = compare_tensor_maps_fp64(
            TensorMap({name: actual[name]}),
            TensorMap({name: oracle[name]}),
            natural_scale=natural_scale,
            **settings,
        ).to_dict()
        per_tensor.append({"parameter_name": name, **result})
    passed = bool(global_result["passed"]) and all(bool(row["passed"]) for row in per_tensor)
    return {
        "comparison_id": comparison_id,
        "profile": profile,
        "object_id": "estimator_core" if comparison_id.endswith("_core") else "estimator_score",
        "natural_scale": natural_scale,
        "passed": passed,
        "global": global_result,
        "per_tensor": per_tensor,
    }


def _profile_payload(profile: str, manifest: Mapping[str, Any]) -> tuple[dict[str, JSONValue], dict[str, JSONValue], dict[str, JSONValue]]:
    dtype = _PROFILE_DTYPES[profile]
    registry = _fixture_registry()
    samples = _manifest_samples(manifest, "microbatch_samples", dtype=dtype, registry=registry)
    identical_samples = _manifest_samples(manifest, "identical_gradient_samples", dtype=dtype, registry=registry)
    double_a_samples = _manifest_samples(manifest, "double_samples_a", dtype=dtype, registry=registry)
    double_b_samples = _manifest_samples(manifest, "double_samples_b", dtype=dtype, registry=registry)
    negative = _manifest_samples(manifest, "negative_samples", dtype=dtype, registry=registry)
    zeros = _manifest_samples(manifest, "zero_samples", dtype=dtype, registry=registry)
    equal = EqualSufficientStatistics.from_samples(samples, accumulation_dtype=dtype)
    weighted = WeightedSufficientStatistics.from_samples(
        samples,
        manifest["weights"],
        accumulation_dtype=dtype,
        **_WEIGHTED_STATISTICAL_CONTRACT,
    )
    equal_weights = WeightedSufficientStatistics.from_samples(
        samples,
        (1.0, 1.0, 1.0, 1.0),
        accumulation_dtype=dtype,
        statistical_unit="microbatch",
        weight_unit="equal_fixture_weight",
        sampling_design="with_replacement_product_sampling",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    mean_a = EqualSufficientStatistics.from_samples(double_a_samples, accumulation_dtype=dtype).mean_gradient
    mean_b = EqualSufficientStatistics.from_samples(double_b_samples, accumulation_dtype=dtype).mean_gradient
    provenance = DoubleSampleProvenance(**dict(manifest["double_provenance"]))
    raw = EstimatorResult.from_core(
        "local_gradient_space_importance_raw", raw_importance(equal.mean_gradient), _LEARNING_RATES
    )
    raw_clipped = EstimatorResult.from_core(
        "local_gradient_space_importance_raw_clipped",
        raw_importance(equal.mean_gradient),
        _LEARNING_RATES,
        clip_factor=_CLIP_FACTOR,
        clip_source="same_batch_mean_gradient",
    )
    double = EstimatorResult.from_core(
        "double_sample_gradient_importance",
        double_sample_importance(mean_a, mean_b, provenance=provenance),
        _LEARNING_RATES,
        unbiasedness_claim=UNBIASED_FIXED_STATE,
        metadata={"sampling_provenance": provenance.to_dict(), "sampling_design": provenance.sampling_design},
    )
    equal_u = EstimatorResult.from_equal_u(equal, _LEARNING_RATES)
    weighted_u = EstimatorResult.from_weighted_u(weighted, _LEARNING_RATES)
    weighted_without_assumptions = EstimatorResult.from_weighted_u(
        WeightedSufficientStatistics.from_samples(
            samples,
            manifest["weights"],
            accumulation_dtype=dtype,
            statistical_unit=str(_WEIGHTED_STATISTICAL_CONTRACT["statistical_unit"]),
            weight_unit=str(_WEIGHTED_STATISTICAL_CONTRACT["weight_unit"]),
            sampling_design=str(_WEIGHTED_STATISTICAL_CONTRACT["sampling_design"]),
            weights_exogenous=False,
            common_mean_assumption=False,
        ),
        _LEARNING_RATES,
    )
    estimator_public_fields = {
        key: {"estimator_name": result.estimator_name, "unbiasedness_claim": result.unbiasedness_claim}
        for key, result in {
            "raw": raw,
            "raw_clipped": raw_clipped,
            "double": double,
            "equal_u": equal_u,
            "weighted_u": weighted_u,
            "weighted_u_assumptions_false": weighted_without_assumptions,
        }.items()
    }
    if estimator_public_fields != _ESTIMATOR_PUBLIC_FIELDS:
        raise Stage1EstimatorError("S1_5_ESTIMATOR_PUBLIC_FIELD_INTERNAL_INVALID")
    ordered_production = explicit_ordered_pair_u_reference(samples)
    unordered_production = explicit_unordered_pair_u_reference(samples)
    streaming = equal_u_importance(equal)
    identical_statistics = EqualSufficientStatistics.from_samples(identical_samples, accumulation_dtype=dtype)
    identical_u = equal_u_importance(identical_statistics)
    identical_u_score = EstimatorResult.from_equal_u(identical_statistics, _LEARNING_RATES).score
    permutation_orders = manifest["permutation_orders"]
    reverse_order = tuple(int(index) for index in permutation_orders["reverse"])
    rotate_order = tuple(int(index) for index in permutation_orders["rotate"])
    reverse_samples = tuple(samples[index] for index in reverse_order)
    rotate_samples = tuple(samples[index] for index in rotate_order)
    reverse_weights = tuple(float(manifest["weights"][index]) for index in reverse_order)
    rotate_weights = tuple(float(manifest["weights"][index]) for index in rotate_order)
    reverse_equal = equal_u_importance(EqualSufficientStatistics.from_samples(reverse_samples, accumulation_dtype=dtype))
    rotate_equal = equal_u_importance(EqualSufficientStatistics.from_samples(rotate_samples, accumulation_dtype=dtype))
    reverse_weighted = weighted_u_importance(WeightedSufficientStatistics.from_samples(
        reverse_samples, reverse_weights, accumulation_dtype=dtype, **_WEIGHTED_STATISTICAL_CONTRACT,
    ))
    rotate_weighted = weighted_u_importance(WeightedSufficientStatistics.from_samples(
        rotate_samples, rotate_weights, accumulation_dtype=dtype, **_WEIGHTED_STATISTICAL_CONTRACT,
    ))
    m2_u = equal_u_importance(
        EqualSufficientStatistics.from_samples(samples[:2], accumulation_dtype=dtype)
    )
    m2_double = double_sample_importance(samples[0], samples[1])
    exchanged_double = double_sample_importance(mean_b, mean_a, provenance=DoubleSampleProvenance(
        batch_object_a=provenance.batch_object_b, batch_object_b=provenance.batch_object_a,
        sampler_state_a=provenance.sampler_state_b, sampler_state_b=provenance.sampler_state_a,
        sampling_design=provenance.sampling_design, product_sampling=True,
        sample_ids_may_overlap=provenance.sample_ids_may_overlap,
        rng_stream_a=provenance.rng_stream_b, rng_stream_b=provenance.rng_stream_a,
        rng_seed_a=provenance.rng_seed_b, rng_seed_b=provenance.rng_seed_a,
        rng_state_digest_a=provenance.rng_state_digest_b, rng_state_digest_b=provenance.rng_state_digest_a,
    ))

    oracle_stats = sufficient_statistics(samples, _WEIGHTS)
    oracle_equal = ordered_pair_u_core(samples)
    oracle_unordered = unordered_pair_u_core(samples)
    oracle_weighted = weighted_ordered_pair_u_core(samples, _WEIGHTS)
    oracle_mean = mean_gradient(samples)
    oracle_raw = raw_core(oracle_mean)
    oracle_double = double_core(mean_gradient(double_a_samples), mean_gradient(double_b_samples))
    parameter_to_group = {name: str(registry.record(name).group_id) for name in registry.eligible_names}
    oracle_raw_score = apply_learning_rates(oracle_raw, _LEARNING_RATES, coordinate_to_group=parameter_to_group)
    oracle_raw_clipped_score = apply_learning_rates(oracle_raw, _LEARNING_RATES, coordinate_to_group=parameter_to_group, clip_factor=_CLIP_FACTOR)
    oracle_double_score = apply_learning_rates(oracle_double, _LEARNING_RATES, coordinate_to_group=parameter_to_group)
    oracle_equal_score = apply_learning_rates(oracle_equal, _LEARNING_RATES, coordinate_to_group=parameter_to_group)
    oracle_weighted_score = apply_learning_rates(oracle_weighted, _LEARNING_RATES, coordinate_to_group=parameter_to_group)
    oracle_equal_weighted = weighted_ordered_pair_u_core(samples, (1.0, 1.0, 1.0, 1.0))
    oracle_m2_double = double_core(samples[0], samples[1])
    oracle_exchanged_double = double_core(mean_gradient(double_b_samples), mean_gradient(double_a_samples))
    oracle_identical = raw_core(mean_gradient(identical_samples))
    oracle_identical_score = apply_learning_rates(oracle_identical, _LEARNING_RATES, coordinate_to_group=parameter_to_group)
    comparisons = [
        _comparison(profile=profile, comparison_id="raw_core", actual=raw.core, oracle=oracle_raw, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="raw_score", actual=raw.score, oracle=oracle_raw_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="raw_clipped_score", actual=raw_clipped.score, oracle=oracle_raw_clipped_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="double_core", actual=double.core, oracle=oracle_double, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="double_score", actual=double.score, oracle=oracle_double_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_ordered_core", actual=ordered_production, oracle=oracle_equal, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_unordered_core", actual=unordered_production, oracle=oracle_unordered, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_streaming_core", actual=streaming, oracle=oracle_equal, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_score", actual=equal_u.score, oracle=oracle_equal_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="weighted_u_core", actual=weighted_u.core, oracle=oracle_weighted, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="weighted_u_score", actual=weighted_u.score, oracle=oracle_weighted_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="weighted_equal_count_degenerate_core", actual=weighted_u_importance(equal_weights), oracle=oracle_equal_weighted, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="m2_u_equals_double_core", actual=m2_u, oracle=m2_double, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="double_exchange_core", actual=exchanged_double, oracle=oracle_double, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="identical_gradient_u_core", actual=identical_u, oracle=oracle_identical, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="identical_gradient_u_score", actual=identical_u_score, oracle=oracle_identical_score, natural_scale=_SCORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_reverse_permutation_core", actual=reverse_equal, oracle=oracle_equal, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="equal_u_rotate_permutation_core", actual=rotate_equal, oracle=oracle_equal, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="weighted_u_reverse_paired_permutation_core", actual=reverse_weighted, oracle=oracle_weighted, natural_scale=_CORE_SCALE),
        _comparison(profile=profile, comparison_id="weighted_u_rotate_paired_permutation_core", actual=rotate_weighted, oracle=oracle_weighted, natural_scale=_CORE_SCALE),
    ]
    if tuple(item["comparison_id"] for item in comparisons) != _COMPARISON_IDS or not all(item["passed"] for item in comparisons):
        raise Stage1EstimatorError("S1_5_INTERNAL_COMPARISON_FAILED")
    negative_u = equal_u_importance(EqualSufficientStatistics.from_samples(negative, accumulation_dtype=dtype))
    zero_stats = EqualSufficientStatistics.from_samples(zeros, accumulation_dtype=dtype)
    zero_equal = equal_u_importance(zero_stats)
    rejection_ids = _expected_rejections(samples, dtype=dtype)
    if not all(value is True for value in rejection_ids.values()):
        raise Stage1EstimatorError("S1_5_NEGATIVE_CONTROL_FAILED")
    statistics_payload: dict[str, JSONValue] = {
        "M": equal.count,
        "S1": tensor_map_to_wire(equal.s1),
        "S2": tensor_map_to_wire(equal.s2),
        "G1": tensor_map_to_wire(weighted.g1),
        "G2": tensor_map_to_wire(weighted.g2),
        "N1": weighted.n1,
        "N2": weighted.n2,
    }
    profile_report: dict[str, JSONValue] = {
        "profile": profile,
        "dtype": str(dtype),
        "natural_scales": {"core": _CORE_SCALE, "score": _SCORE_SCALE},
        "learning_rates": dict(_LEARNING_RATES),
        "clip_factor": manifest["clip_factor"],
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "parameter_to_group": parameter_to_group,
        "weighted_statistical_contract": dict(_WEIGHTED_STATISTICAL_CONTRACT),
        "estimator_public_fields": estimator_public_fields,
        "comparisons": comparisons,
        "raw_clip_boundary": _raw_clip_checks(raw, raw_clipped),
        "double_sampling": {
            "provenance": provenance.to_dict(),
            "sample_ids_a": list(manifest["double_sample_ids_a"]),
            "sample_ids_b": list(manifest["double_sample_ids_b"]),
            "sample_id_overlap_allowed": True,
            "a_b_exchange_passed": True,
            "m2_equals_double_passed": True,
        },
        "negative_u": _tensor_values(negative_u),
        "zero_outputs": {"raw_core": _tensor_values(raw_importance(zero_stats.mean_gradient)), "u_core": _tensor_values(zero_equal)},
        "rejections": rejection_ids,
        "sufficient_statistics": statistics_payload,
        "sufficient_statistics_natural_scales": {"S1": 4.0, "S2": 16.0, "G1": 40.0, "G2": 160.0, "N1": 10.0, "N2": 30.0},
        "scatter_rows": _scatter_rows(profile, ordered_production, unordered_production, streaming, oracle_equal),
        "scaling_rows": _scaling_rows(profile, raw.core, raw.score, raw_clipped.score, equal_u.core, equal_u.score),
    }
    bundle_profile: dict[str, JSONValue] = {
        "profile": profile,
        "dtype": str(dtype),
        "learning_rates": dict(_LEARNING_RATES),
        "clip_factor": _CLIP_FACTOR,
        "coordinate_registry_hash": registry.coordinate_registry_hash,
        "parameter_to_group": parameter_to_group,
        "weighted_statistical_contract": dict(_WEIGHTED_STATISTICAL_CONTRACT),
        "estimator_public_fields": estimator_public_fields,
        "microbatch_samples": _wire_samples(samples),
        "identical_gradient_samples": _wire_samples(identical_samples),
        "permutation_orders": dict(permutation_orders),
        "weights": list(manifest["weights"]),
        "double_samples_a": _wire_samples(double_a_samples),
        "double_samples_b": _wire_samples(double_b_samples),
        "double_provenance": provenance.to_dict(),
        "double_sample_ids_a": list(manifest["double_sample_ids_a"]),
        "double_sample_ids_b": list(manifest["double_sample_ids_b"]),
        "sufficient_statistics": statistics_payload,
        "production_outputs": {
            "raw_core": tensor_map_to_wire(raw.core),
            "raw_score": tensor_map_to_wire(raw.score),
            "raw_clipped_score": tensor_map_to_wire(raw_clipped.score),
            "double_core": tensor_map_to_wire(double.core),
            "double_score": tensor_map_to_wire(double.score),
            "equal_u_ordered_core": tensor_map_to_wire(ordered_production),
            "equal_u_unordered_core": tensor_map_to_wire(unordered_production),
            "equal_u_streaming_core": tensor_map_to_wire(streaming),
            "equal_u_score": tensor_map_to_wire(equal_u.score),
            "weighted_u_core": tensor_map_to_wire(weighted_u.core),
            "weighted_u_score": tensor_map_to_wire(weighted_u.score),
            "weighted_equal_count_degenerate_core": tensor_map_to_wire(weighted_u_importance(equal_weights)),
            "m2_u_equals_double_core": tensor_map_to_wire(m2_u),
            "double_exchange_core": tensor_map_to_wire(exchanged_double),
            "identical_gradient_u_core": tensor_map_to_wire(identical_u),
            "identical_gradient_u_score": tensor_map_to_wire(identical_u_score),
            "equal_u_reverse_permutation_core": tensor_map_to_wire(reverse_equal),
            "equal_u_rotate_permutation_core": tensor_map_to_wire(rotate_equal),
            "weighted_u_reverse_paired_permutation_core": tensor_map_to_wire(reverse_weighted),
            "weighted_u_rotate_paired_permutation_core": tensor_map_to_wire(rotate_weighted),
        },
        "boundary_outputs": {
            "raw_unclipped_core": tensor_map_to_wire(raw.core),
            "raw_clipped_core": tensor_map_to_wire(raw_clipped.core),
            "negative_u": tensor_map_to_wire(negative_u),
            "zero_raw_core": tensor_map_to_wire(raw_importance(zero_stats.mean_gradient)),
            "zero_u_core": tensor_map_to_wire(zero_equal),
        },
    }
    oracle_profile: dict[str, JSONValue] = {
        "profile": profile,
        "natural_scales": {"core": _CORE_SCALE, "score": _SCORE_SCALE},
        "sufficient_statistics": {
            "S1": tensor_map_to_wire(oracle_stats["s1"]), "S2": tensor_map_to_wire(oracle_stats["s2"]),
            "G1": tensor_map_to_wire(oracle_stats["g1"]), "G2": tensor_map_to_wire(oracle_stats["g2"]),
            "N1": oracle_stats["n1"], "N2": oracle_stats["n2"],
        },
        "oracle_outputs": {
            "raw_core": tensor_map_to_wire(oracle_raw), "raw_score": tensor_map_to_wire(oracle_raw_score),
            "raw_clipped_score": tensor_map_to_wire(oracle_raw_clipped_score), "double_core": tensor_map_to_wire(oracle_double),
            "double_score": tensor_map_to_wire(oracle_double_score), "equal_u_ordered_core": tensor_map_to_wire(oracle_equal),
            "equal_u_unordered_core": tensor_map_to_wire(oracle_unordered), "equal_u_streaming_core": tensor_map_to_wire(oracle_equal),
            "equal_u_score": tensor_map_to_wire(oracle_equal_score), "weighted_u_core": tensor_map_to_wire(oracle_weighted),
            "weighted_u_score": tensor_map_to_wire(oracle_weighted_score),
            "weighted_equal_count_degenerate_core": tensor_map_to_wire(oracle_equal_weighted),
            "m2_u_equals_double_core": tensor_map_to_wire(oracle_m2_double),
            "double_exchange_core": tensor_map_to_wire(oracle_exchanged_double),
            "identical_gradient_u_core": tensor_map_to_wire(oracle_identical),
            "identical_gradient_u_score": tensor_map_to_wire(oracle_identical_score),
            "equal_u_reverse_permutation_core": tensor_map_to_wire(oracle_equal),
            "equal_u_rotate_permutation_core": tensor_map_to_wire(oracle_equal),
            "weighted_u_reverse_paired_permutation_core": tensor_map_to_wire(oracle_weighted),
            "weighted_u_rotate_paired_permutation_core": tensor_map_to_wire(oracle_weighted),
        },
        "boundary_outputs": {
            "raw_unclipped_core": tensor_map_to_wire(oracle_raw),
            "raw_clipped_core": tensor_map_to_wire(oracle_raw),
            "negative_u": tensor_map_to_wire(ordered_pair_u_core(negative)),
            "zero_raw_core": tensor_map_to_wire(raw_core(mean_gradient(zeros))),
            "zero_u_core": tensor_map_to_wire(ordered_pair_u_core(zeros)),
        },
    }
    return profile_report, bundle_profile, oracle_profile


def _expected_rejections(samples: Sequence[TensorMap], *, dtype: torch.dtype) -> dict[str, bool]:
    def rejected(callable_: object) -> bool:
        try:
            assert callable(callable_)
            callable_()
        except (CoreContractError, TensorMapError, NumericalError):
            return True
        return False
    registry = samples[0].registry
    assert registry is not None
    nonfinite = TensorMap({"weight": torch.tensor((float("inf"), 0.0, 1.0), dtype=dtype), "bias": torch.zeros(2, dtype=dtype), "head": torch.zeros(1, dtype=dtype)}, registry=registry, require_finite=False)
    wrong_shape = TensorMap({"weight": torch.ones(4, dtype=dtype), "bias": torch.ones(2, dtype=dtype), "head": torch.ones(1, dtype=dtype)}, require_finite=False)
    one = EqualSufficientStatistics.from_samples(samples[:1], accumulation_dtype=dtype)
    # An unbound map is a different registry identity, even when its names and
    # shapes happen to look identical; mixed bound/unbound arithmetic is banned.
    registry_mismatch = TensorMap({name: value.clone() for name, value in samples[1].items()})
    dtype_mismatch = samples[1].to(dtype=torch.float64 if dtype == torch.float32 else torch.float32)
    device_mismatch = TensorMap(
        {name: value.to(device="meta") for name, value in samples[1].items()},
        registry=registry,
        require_finite=False,
    )
    zero_denominator = WeightedSufficientStatistics(
        count=2,
        g1=TensorMap.zeros_like(samples[0]),
        g2=TensorMap.zeros_like(samples[0]),
        n1=1.0,
        n2=1.0,
        accumulation_dtype=dtype,
        statistical_unit="microbatch",
        weight_unit="effective_target_tokens",
        sampling_design="with_replacement_product_sampling",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    shared_sampler_provenance = {
        "batch_object_a": "a", "batch_object_b": "b",
        "sampler_state_a": "same", "sampler_state_b": "same",
        "sampling_design": "with_replacement_product_sampling", "product_sampling": True,
        "sample_ids_may_overlap": True,
        "rng_stream_a": "stream-a", "rng_stream_b": "stream-b",
        "rng_seed_a": 1, "rng_seed_b": 2,
        "rng_state_digest_a": "a" * 64, "rng_state_digest_b": "b" * 64,
    }
    result = {
        "m1_rejected": rejected(lambda: equal_u_importance(one)),
        "same_batch_object_rejected": rejected(lambda: double_sample_importance(samples[0], samples[0])),
        "shared_sampler_state_rejected": rejected(lambda: DoubleSampleProvenance(**shared_sampler_provenance)),
        "nonfinite_rejected": rejected(lambda: raw_importance(nonfinite)),
        "shape_mismatch_rejected": rejected(lambda: double_sample_importance(samples[0], wrong_shape)),
        "registry_mismatch_rejected": rejected(lambda: double_sample_importance(samples[0], registry_mismatch)),
        "dtype_device_mismatch_rejected": rejected(lambda: double_sample_importance(samples[0], dtype_mismatch)) and rejected(lambda: double_sample_importance(samples[0], device_mismatch)),
        "nonpositive_weight_rejected": rejected(lambda: WeightedSufficientStatistics.from_samples(samples, (1.0, 0.0, 1.0, 1.0), accumulation_dtype=dtype, statistical_unit="microbatch", weight_unit="effective_target_tokens", sampling_design="with_replacement_product_sampling", weights_exogenous=True, common_mean_assumption=True)),
        "nonpositive_denominator_rejected": rejected(lambda: weighted_u_importance(zero_denominator)),
        "negative_learning_rate_rejected": rejected(lambda: EstimatorResult.from_core("local_gradient_space_importance_raw", raw_importance(samples[0]), {"group_0000": -0.25, "group_0001": 0.05})),
        "nonfinite_learning_rate_rejected": rejected(lambda: EstimatorResult.from_core("local_gradient_space_importance_raw", raw_importance(samples[0]), {"group_0000": float("nan"), "group_0001": 0.05})),
        "nonfinite_clip_rejected": rejected(lambda: EstimatorResult.from_core("local_gradient_space_importance_raw_clipped", raw_importance(samples[0]), _LEARNING_RATES, clip_factor=float("nan"), clip_source="external_constant")),
        "out_of_range_clip_rejected": rejected(lambda: EstimatorResult.from_core("local_gradient_space_importance_raw_clipped", raw_importance(samples[0]), _LEARNING_RATES, clip_factor=1.1, clip_source="external_constant")),
        "float16_accumulation_rejected": rejected(lambda: EqualSufficientStatistics.from_samples(samples, accumulation_dtype=torch.float16)),
    }
    if tuple(result) != _REJECTION_IDS:
        raise Stage1EstimatorError("S1_5_REJECTION_ID_SET_INTERNAL_INVALID")
    return result


def _scatter_rows(
    profile: str,
    ordered: TensorMap,
    unordered: TensorMap,
    streaming: TensorMap,
    oracle: TensorMap,
) -> list[dict[str, JSONValue]]:
    """Exact scatter projection for three independent U production routes."""

    rows: list[dict[str, JSONValue]] = []
    for route_id, actual in (
        ("equal_u_ordered_vs_oracle", ordered),
        ("equal_u_unordered_vs_oracle", unordered),
        ("equal_u_streaming_vs_oracle", streaming),
    ):
        for name in actual:
            for coordinate, (candidate, reference) in enumerate(zip(actual[name].detach().to(torch.float64).reshape(-1).tolist(), oracle[name].detach().to(torch.float64).reshape(-1).tolist(), strict=True)):
                rows.append({"profile": profile, "route_id": route_id, "parameter_name": name, "coordinate": coordinate, "candidate": float(candidate), "reference": float(reference)})
    return rows


def _scaling_rows(profile: str, raw_core_value: TensorMap, raw_score: TensorMap, raw_clipped: TensorMap, u_core: TensorMap, u_score: TensorMap) -> list[dict[str, JSONValue]]:
    rows: list[dict[str, JSONValue]] = []
    for name in raw_core_value:
        for coordinate, values in enumerate(zip(raw_core_value[name].detach().to(torch.float64).reshape(-1).tolist(), raw_score[name].detach().to(torch.float64).reshape(-1).tolist(), raw_clipped[name].detach().to(torch.float64).reshape(-1).tolist(), u_core[name].detach().to(torch.float64).reshape(-1).tolist(), u_score[name].detach().to(torch.float64).reshape(-1).tolist(), strict=True)):
            raw_core_entry, raw_score_entry, raw_clip_entry, u_core_entry, u_score_entry = values
            assert raw_core_value.registry is not None
            group_id = raw_core_value.registry.record(name).group_id
            assert group_id is not None
            rows.append({"profile": profile, "parameter_name": name, "parameter_group": group_id, "coordinate": coordinate, "learning_rate": _LEARNING_RATES[group_id], "clip_factor": _CLIP_FACTOR, "raw_core": float(raw_core_entry), "raw_score": float(raw_score_entry), "raw_clipped_score": float(raw_clip_entry), "u_core": float(u_core_entry), "u_score": float(u_score_entry)})
    return rows


def _raw_clip_checks(raw: EstimatorResult, raw_clipped: EstimatorResult) -> dict[str, JSONValue]:
    """Separate raw core/raw score invariance from the one-factor clipped score."""

    raw.core.assert_compatible(raw_clipped.core, require_dtype_device=True)
    raw.score.assert_compatible(raw_clipped.score, require_dtype_device=True)
    return {
        "raw_core_unchanged_by_clip": {name: bool(torch.equal(raw.core[name], raw_clipped.core[name])) for name in raw.core},
        "raw_score_unchanged_by_clip": {name: bool(torch.equal(raw.score[name], EstimatorResult.from_core("local_gradient_space_importance_raw", raw.core, _LEARNING_RATES).score[name])) for name in raw.score},
        "raw_clipped_one_factor": {name: bool(torch.equal(raw_clipped.score[name], raw.score[name] * _CLIP_FACTOR)) for name in raw.score},
    }


def _derive_requirement_checks(report: Mapping[str, Any]) -> dict[str, bool]:
    profiles = report.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != len(_PROFILES):
        raise Stage1EstimatorError("S1_5_REQUIREMENT_DERIVATION_PROFILE_INVALID")
    expected_public_fields = dict(_ESTIMATOR_PUBLIC_FIELDS)
    def all_profiles(predicate: object) -> bool:
        assert callable(predicate)
        return len(profiles) == len(_PROFILES) and all(isinstance(profile, Mapping) and bool(predicate(profile)) for profile in profiles)
    def comparison_set_passes(profile: Mapping[str, Any], expected: set[str]) -> bool:
        comparisons = profile.get("comparisons")
        if not isinstance(comparisons, list):
            return False
        by_id = {
            item.get("comparison_id"): item
            for item in comparisons
            if isinstance(item, Mapping) and isinstance(item.get("comparison_id"), str)
        }
        return set(by_id) == set(_COMPARISON_IDS) and all(by_id[comparison_id].get("passed") is True for comparison_id in expected)
    return {
        "raw_core_and_scores": all_profiles(lambda p: comparison_set_passes(p, {"raw_core", "raw_score", "raw_clipped_score"})),
        "multi_group_learning_rates": all_profiles(lambda p: p.get("parameter_to_group") == {"bias": "group_0000", "head": "group_0001", "weight": "group_0000"} and p.get("learning_rates") == _LEARNING_RATES),
        "raw_clip_separation": all_profiles(lambda p: all(all(value is True for value in values.values()) for values in p.get("raw_clip_boundary", {}).values() if isinstance(values, Mapping))),
        "double_provenance": all_profiles(lambda p: isinstance(p.get("double_sampling"), Mapping) and p["double_sampling"].get("sample_id_overlap_allowed") is True and p["double_sampling"].get("a_b_exchange_passed") is True and p["double_sampling"].get("m2_equals_double_passed") is True),
        "ordered_unordered_streaming_identity": all_profiles(lambda p: comparison_set_passes(p, {"equal_u_ordered_core", "equal_u_unordered_core", "equal_u_streaming_core"})),
        "weighted_u_and_equal_degeneracy": all_profiles(lambda p: comparison_set_passes(p, {"weighted_u_core", "weighted_u_score", "weighted_equal_count_degenerate_core"})),
        "identical_gradient_degeneracy": all_profiles(lambda p: comparison_set_passes(p, {"identical_gradient_u_core", "identical_gradient_u_score"})),
        "paired_permutation_invariance": all_profiles(lambda p: comparison_set_passes(p, {"equal_u_reverse_permutation_core", "equal_u_rotate_permutation_core", "weighted_u_reverse_paired_permutation_core", "weighted_u_rotate_paired_permutation_core"})),
        "estimator_public_field_bindings": all_profiles(lambda p: p.get("estimator_public_fields") == expected_public_fields and p.get("weighted_statistical_contract") == _WEIGHTED_STATISTICAL_CONTRACT),
        "negative_and_zero_boundaries": all_profiles(lambda p: all(float(value) < 0.0 for values in p.get("negative_u", {}).values() for value in values) and all(float(value) == 0.0 for values in p.get("zero_outputs", {}).values() for coordinates in values.values() for value in coordinates)),
        "m1_and_invalid_inputs_fail_closed": all_profiles(lambda p: isinstance(p.get("rejections"), Mapping) and set(p["rejections"]) == set(_REJECTION_IDS) and len(p["rejections"]) == len(_REJECTION_IDS) and all(value is True for value in p["rejections"].values())),
        "all_tensor_rows_pass": all_profiles(lambda p: comparison_set_passes(p, set(_COMPARISON_IDS))),
    }


def _derive_gate_requirements(report: Mapping[str, Any], table: Mapping[str, Any]) -> dict[str, bool]:
    """Derive every gate bit from a saved check object, never a literal flag."""

    requirements = _derive_requirement_checks(report)
    expected_table = _build_table(report)
    requirements["table_projection_exact"] = (
        table.get("report_hash") == report.get("report_hash")
        and {key: table.get(key) for key in ("rows", "scatter_rows", "scaling_rows", "negative_u_rows", "projection_hash")}
        == {key: expected_table.get(key) for key in ("rows", "scatter_rows", "scaling_rows", "negative_u_rows", "projection_hash")}
    )
    return requirements


def _formal_upstream(upstream: Mapping[str, JSONValue] | None) -> dict[str, JSONValue]:
    if upstream is None:
        return {}
    required = {
        "s1_4_index_ref", "s1_4_index_sha256", "s1_4_index_artifact_hash", "s1_4_gate_artifact_hash",
        "s1_4_gradient_scale_report_sha256", "s1_4_comparison_table_sha256", "s1_4_gate_record_sha256",
        "s1_4_replay_sha256", "s1_4_validation_sha256",
    }
    if set(upstream) != required:
        raise Stage1EstimatorError("S1_5_FORMAL_S1_4_HANDOFF_REQUIRED")
    result = dict(upstream)
    if result["s1_4_index_sha256"] != EXPECTED_S1_4_INDEX_SHA256 or result["s1_4_gate_artifact_hash"] != EXPECTED_S1_4_GATE_ARTIFACT_HASH:
        raise Stage1EstimatorError("S1_5_FORMAL_S1_4_HANDOFF_NOT_CURRENT")
    for field in required - {"s1_4_index_ref"}:
        _require_digest(result[field], field=field)
    if not isinstance(result["s1_4_index_ref"], str) or not result["s1_4_index_ref"]:
        raise Stage1EstimatorError("S1_5_FORMAL_S1_4_INDEX_REF_INVALID")
    return result


def _build_table(report: Mapping[str, Any]) -> dict[str, JSONValue]:
    rows: list[dict[str, JSONValue]] = []
    scatter_rows: list[dict[str, JSONValue]] = []
    scaling_rows: list[dict[str, JSONValue]] = []
    negative_rows: list[dict[str, JSONValue]] = []
    for profile in report["profiles"]:
        assert isinstance(profile, Mapping)
        for comparison in profile["comparisons"]:
            assert isinstance(comparison, Mapping)
            for tensor in comparison["per_tensor"]:
                assert isinstance(tensor, Mapping)
                rows.append({"profile": profile["profile"], "comparison_id": comparison["comparison_id"], "object_id": comparison["object_id"], **dict(tensor)})
        scatter_rows.extend(profile["scatter_rows"])
        scaling_rows.extend(profile["scaling_rows"])
        for name, values in profile["negative_u"].items():
            for coordinate, value in enumerate(values):
                negative_rows.append({"profile": profile["profile"], "parameter_name": name, "coordinate": coordinate, "u_core": value})
    projection = {"rows": rows, "scatter_rows": scatter_rows, "scaling_rows": scaling_rows, "negative_u_rows": negative_rows}
    return {"schema_version": TABLE_SCHEMA, "fixture_id": FIXTURE_ID, "report_hash": report["report_hash"], **projection, "projection_hash": canonical_json_hash(projection)}


def build_stage1_s15_evidence(source_root: str | Path, *, producer_commit: str, scope: str = "local_fixture", upstream_evidence: Mapping[str, JSONValue] | None = None) -> dict[str, JSONValue]:
    """Build role-separated S1.5 evidence without writing a formal artifact."""

    root = Path(source_root).resolve()
    _require_commit(producer_commit, field="producer_commit")
    if scope not in {"local_fixture", "formal"}:
        raise Stage1EstimatorError("S1_5_SCOPE_INVALID")
    upstream = {} if scope == "local_fixture" else _formal_upstream(upstream_evidence)
    if scope == "local_fixture" and upstream_evidence:
        raise Stage1EstimatorError("S1_5_LOCAL_SCOPE_MUST_NOT_CLAIM_UPSTREAM")
    fixture_manifest, fixture_manifest_sha256 = _load_fixture_manifest(root)
    report_profiles: list[dict[str, JSONValue]] = []
    bundle_profiles: list[dict[str, JSONValue]] = []
    oracle_profiles: list[dict[str, JSONValue]] = []
    for profile in _PROFILES:
        report, bundle, oracle = _profile_payload(profile, fixture_manifest)
        report_profiles.append(report)
        bundle_profiles.append(bundle)
        oracle_profiles.append(oracle)
    bundle: dict[str, JSONValue] = {
        "schema_version": BUNDLE_SCHEMA, "fixture_id": FIXTURE_ID, "producer_commit": producer_commit,
        "scope": scope, "fixture_manifest_hash": fixture_manifest["manifest_hash"], "fixture_manifest_sha256": fixture_manifest_sha256,
        "coordinate_contract": _coordinate_contract(_fixture_registry()), "estimator_input_contract": fixture_manifest["estimator_input_contract"], "profiles": bundle_profiles,
    }
    bundle["bundle_hash"] = _hash_role(bundle, field="bundle_hash")
    oracle_report: dict[str, JSONValue] = {
        "schema_version": ORACLE_SCHEMA, "fixture_id": FIXTURE_ID, "producer_commit": producer_commit,
        "scope": scope, "oracle_algorithm": "explicit_fp64_python_pair_loops_without_production_estimators",
        "bundle_hash": bundle["bundle_hash"], "profiles": oracle_profiles,
    }
    oracle_report["oracle_hash"] = _hash_role(oracle_report, field="oracle_hash")
    report: dict[str, JSONValue] = {
        "schema_version": REPORT_SCHEMA, "fixture_id": FIXTURE_ID, "task_id": TASK_ID, "gate_id": GATE_ID,
        "producer_commit": producer_commit, "scope": scope, "status": "PASS",
        "gate_status": "PASS" if scope == "formal" else "NOT_RUN", "upstream": upstream,
        "bundle_hash": bundle["bundle_hash"], "oracle_hash": oracle_report["oracle_hash"],
        "fixture_manifest_hash": fixture_manifest["manifest_hash"], "fixture_manifest_sha256": fixture_manifest_sha256,
        "coordinate_contract_hash": canonical_json_hash(bundle["coordinate_contract"]),
        "estimator_input_contract": fixture_manifest["estimator_input_contract"],
        "profiles": report_profiles, "implementation_source_sha256": _source_hashes(root),
    }
    report["requirement_checks"] = _derive_requirement_checks(report)
    report["report_hash"] = _hash_role(report, field="report_hash")
    table = _build_table(report)
    table["table_hash"] = _hash_role(table, field="table_hash")
    requirements = _derive_gate_requirements(report, table)
    gate: dict[str, JSONValue] = {
        "schema_version": GATE_SCHEMA, "task_id": TASK_ID, "gate_id": GATE_ID, "scope": scope,
        "status": "PASS" if scope == "formal" else "NOT_RUN", "report_hash": report["report_hash"],
        "oracle_hash": oracle_report["oracle_hash"], "bundle_hash": bundle["bundle_hash"],
        "comparison_table_hash": table["table_hash"], "requirements": requirements,
    }
    gate["artifact_hash"] = _hash_role(gate, field="artifact_hash")
    evidence: dict[str, JSONValue] = {"estimator_report": report, "oracle_report": oracle_report, "tensor_bundle": bundle, "comparison_table": table, "gate_record": gate}
    validate_stage1_s15_evidence(evidence)
    return evidence


def _role_key_set(value: Mapping[str, Any], expected: set[str], *, role: str) -> None:
    if set(value) != expected:
        raise Stage1EstimatorError(f"S1_5_{role.upper()}_KEY_SET_INVALID")


def validate_stage1_s15_evidence(
    evidence: Mapping[str, Any], *, source_root: str | Path | None = None
) -> dict[str, JSONValue]:
    """Fail-closed structural validation; numeric replay is intentionally separate."""

    _role_key_set(evidence, {"estimator_report", "oracle_report", "tensor_bundle", "comparison_table", "gate_record"}, role="evidence")
    report, oracle, bundle, table, gate = (evidence[key] for key in ("estimator_report", "oracle_report", "tensor_bundle", "comparison_table", "gate_record"))
    if not all(isinstance(value, Mapping) for value in (report, oracle, bundle, table, gate)):
        raise Stage1EstimatorError("S1_5_ROLE_NOT_OBJECT")
    assert isinstance(report, Mapping) and isinstance(oracle, Mapping) and isinstance(bundle, Mapping) and isinstance(table, Mapping) and isinstance(gate, Mapping)
    _role_key_set(report, {"schema_version", "fixture_id", "task_id", "gate_id", "producer_commit", "scope", "status", "gate_status", "upstream", "bundle_hash", "oracle_hash", "fixture_manifest_hash", "fixture_manifest_sha256", "coordinate_contract_hash", "estimator_input_contract", "profiles", "implementation_source_sha256", "requirement_checks", "report_hash"}, role="report")
    _role_key_set(oracle, {"schema_version", "fixture_id", "producer_commit", "scope", "oracle_algorithm", "bundle_hash", "profiles", "oracle_hash"}, role="oracle")
    _role_key_set(bundle, {"schema_version", "fixture_id", "producer_commit", "scope", "fixture_manifest_hash", "fixture_manifest_sha256", "coordinate_contract", "estimator_input_contract", "profiles", "bundle_hash"}, role="bundle")
    _role_key_set(table, {"schema_version", "fixture_id", "report_hash", "rows", "scatter_rows", "scaling_rows", "negative_u_rows", "projection_hash", "table_hash"}, role="table")
    _role_key_set(gate, {"schema_version", "task_id", "gate_id", "scope", "status", "report_hash", "oracle_hash", "bundle_hash", "comparison_table_hash", "requirements", "artifact_hash"}, role="gate")
    if report["schema_version"] != REPORT_SCHEMA or oracle["schema_version"] != ORACLE_SCHEMA or bundle["schema_version"] != BUNDLE_SCHEMA or table["schema_version"] != TABLE_SCHEMA or gate["schema_version"] != GATE_SCHEMA:
        raise Stage1EstimatorError("S1_5_SCHEMA_VERSION_INVALID")
    if any(role.get("fixture_id") != FIXTURE_ID for role in (report, oracle, bundle, table)) or report["task_id"] != TASK_ID or gate["task_id"] != TASK_ID or report["gate_id"] != GATE_ID or gate["gate_id"] != GATE_ID:
        raise Stage1EstimatorError("S1_5_ROLE_IDENTITY_INVALID")
    if report["scope"] not in {"local_fixture", "formal"} or any(role["scope"] != report["scope"] for role in (oracle, bundle, gate)):
        raise Stage1EstimatorError("S1_5_SCOPE_BINDING_INVALID")
    _require_commit(report["producer_commit"], field="producer_commit")
    if oracle["producer_commit"] != report["producer_commit"] or bundle["producer_commit"] != report["producer_commit"]:
        raise Stage1EstimatorError("S1_5_PRODUCER_BINDING_INVALID")
    if report["bundle_hash"] != bundle["bundle_hash"] or oracle["bundle_hash"] != bundle["bundle_hash"] or report["oracle_hash"] != oracle["oracle_hash"]:
        raise Stage1EstimatorError("S1_5_ROLE_HASH_BINDING_INVALID")
    if report["report_hash"] != _hash_role(report, field="report_hash") or oracle["oracle_hash"] != _hash_role(oracle, field="oracle_hash") or bundle["bundle_hash"] != _hash_role(bundle, field="bundle_hash") or table["table_hash"] != _hash_role(table, field="table_hash") or gate["artifact_hash"] != _hash_role(gate, field="artifact_hash"):
        raise Stage1EstimatorError("S1_5_SELF_HASH_INVALID")
    if table["report_hash"] != report["report_hash"] or gate["report_hash"] != report["report_hash"] or gate["oracle_hash"] != oracle["oracle_hash"] or gate["bundle_hash"] != bundle["bundle_hash"] or gate["comparison_table_hash"] != table["table_hash"]:
        raise Stage1EstimatorError("S1_5_CROSS_ROLE_HASH_INVALID")
    expected_table = _build_table(report)
    if {key: table[key] for key in ("rows", "scatter_rows", "scaling_rows", "negative_u_rows", "projection_hash")} != {key: expected_table[key] for key in ("rows", "scatter_rows", "scaling_rows", "negative_u_rows", "projection_hash")}:
        raise Stage1EstimatorError("S1_5_TABLE_PROJECTION_INVALID")
    coordinate_contract = _coordinate_contract(_fixture_registry())
    if bundle.get("coordinate_contract") != coordinate_contract or report.get("coordinate_contract_hash") != canonical_json_hash(coordinate_contract):
        raise Stage1EstimatorError("S1_5_FIXTURE_COORDINATE_CONTRACT_INVALID")
    manifest_root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[2]
    source_manifest, source_manifest_sha256 = _load_fixture_manifest(manifest_root)
    if report.get("fixture_manifest_hash") != source_manifest["manifest_hash"] or bundle.get("fixture_manifest_hash") != source_manifest["manifest_hash"] or report.get("fixture_manifest_sha256") != source_manifest_sha256 or bundle.get("fixture_manifest_sha256") != source_manifest_sha256:
        raise Stage1EstimatorError("S1_5_FIXTURE_MANIFEST_BINDING_INVALID")
    if report.get("estimator_input_contract") != source_manifest["estimator_input_contract"] or bundle.get("estimator_input_contract") != source_manifest["estimator_input_contract"] or source_manifest["estimator_input_contract"] != _estimator_input_contract(_fixture_registry()):
        raise Stage1EstimatorError("S1_5_ESTIMATOR_INPUT_CONTRACT_BINDING_INVALID")
    if not isinstance(report["profiles"], list) or tuple(item.get("profile") for item in report["profiles"] if isinstance(item, Mapping)) != tuple(_PROFILES):
        raise Stage1EstimatorError("S1_5_REPORT_PROFILE_INVALID")
    if not isinstance(bundle["profiles"], list) or not isinstance(oracle["profiles"], list) or [item.get("profile") for item in bundle["profiles"] if isinstance(item, Mapping)] != list(_PROFILES) or [item.get("profile") for item in oracle["profiles"] if isinstance(item, Mapping)] != list(_PROFILES):
        raise Stage1EstimatorError("S1_5_BUNDLE_OR_ORACLE_PROFILE_INVALID")
    if report["scope"] == "formal":
        _formal_upstream(report["upstream"] if isinstance(report["upstream"], Mapping) else None)
        if report["gate_status"] != "PASS" or gate["status"] != "PASS":
            raise Stage1EstimatorError("S1_5_FORMAL_GATE_STATUS_INVALID")
    elif report["upstream"] != {} or report["gate_status"] != "NOT_RUN" or gate["status"] != "NOT_RUN":
        raise Stage1EstimatorError("S1_5_LOCAL_GATE_STATUS_INVALID")
    report_profiles_by_name = {item.get("profile"): item for item in report["profiles"] if isinstance(item, Mapping)}
    for bundle_profile, oracle_profile in zip(bundle["profiles"], oracle["profiles"], strict=True):
        if not isinstance(bundle_profile, Mapping) or not isinstance(oracle_profile, Mapping):
            raise Stage1EstimatorError("S1_5_BUNDLE_OR_ORACLE_PROFILE_NOT_OBJECT")
        profile_name = bundle_profile.get("profile")
        report_profile = report_profiles_by_name.get(profile_name)
        if not isinstance(report_profile, Mapping) or oracle_profile.get("profile") != profile_name:
            raise Stage1EstimatorError("S1_5_PROFILE_CROSS_ROLE_BINDING_INVALID")
        expected_bundle_keys = {
            "profile", "dtype", "learning_rates", "clip_factor", "coordinate_registry_hash", "parameter_to_group",
            "weighted_statistical_contract", "estimator_public_fields", "microbatch_samples", "identical_gradient_samples",
            "permutation_orders", "weights", "double_samples_a", "double_samples_b", "double_provenance",
            "double_sample_ids_a", "double_sample_ids_b", "sufficient_statistics", "production_outputs", "boundary_outputs",
        }
        expected_oracle_keys = {"profile", "natural_scales", "sufficient_statistics", "oracle_outputs", "boundary_outputs"}
        if set(bundle_profile) != expected_bundle_keys or set(oracle_profile) != expected_oracle_keys:
            raise Stage1EstimatorError("S1_5_PROFILE_ROLE_KEY_SET_INVALID")
        if bundle_profile.get("coordinate_registry_hash") != coordinate_contract["coordinate_registry_hash"] or bundle_profile.get("parameter_to_group") != coordinate_contract["parameter_to_group"]:
            raise Stage1EstimatorError("S1_5_PROFILE_REGISTRY_BINDING_INVALID")
        if report_profile.get("coordinate_registry_hash") != coordinate_contract["coordinate_registry_hash"] or report_profile.get("parameter_to_group") != coordinate_contract["parameter_to_group"]:
            raise Stage1EstimatorError("S1_5_REPORT_REGISTRY_BINDING_INVALID")
        expected_report_profile_keys = {
            "profile", "dtype", "natural_scales", "learning_rates", "clip_factor", "coordinate_registry_hash",
            "parameter_to_group", "weighted_statistical_contract", "estimator_public_fields", "comparisons", "raw_clip_boundary", "double_sampling", "negative_u",
            "zero_outputs", "rejections", "sufficient_statistics", "sufficient_statistics_natural_scales",
            "scatter_rows", "scaling_rows",
        }
        if set(report_profile) != expected_report_profile_keys:
            raise Stage1EstimatorError("S1_5_REPORT_PROFILE_KEY_SET_INVALID")
        if report_profile.get("sufficient_statistics") != bundle_profile.get("sufficient_statistics"):
            raise Stage1EstimatorError("S1_5_REPORT_STATISTICS_BINDING_INVALID")
        if report_profile.get("weighted_statistical_contract") != source_manifest["weighted_statistical_contract"] or bundle_profile.get("weighted_statistical_contract") != source_manifest["weighted_statistical_contract"]:
            raise Stage1EstimatorError("S1_5_WEIGHTED_STATISTICAL_CONTRACT_BINDING_INVALID")
        if report_profile.get("estimator_public_fields") != _ESTIMATOR_PUBLIC_FIELDS or bundle_profile.get("estimator_public_fields") != _ESTIMATOR_PUBLIC_FIELDS or report_profile.get("estimator_public_fields") != bundle_profile.get("estimator_public_fields"):
            raise Stage1EstimatorError("S1_5_ESTIMATOR_PUBLIC_FIELD_BINDING_INVALID")
        if report_profile.get("sufficient_statistics_natural_scales") != {"S1": 4.0, "S2": 16.0, "G1": 40.0, "G2": 160.0, "N1": 10.0, "N2": 30.0}:
            raise Stage1EstimatorError("S1_5_REPORT_STATISTICS_SCALE_INVALID")
        for field, expected_keys in (("production_outputs", set(_COMPARISON_IDS)), ("oracle_outputs", set(_COMPARISON_IDS)), ("boundary_outputs", {"raw_unclipped_core", "raw_clipped_core", "negative_u", "zero_raw_core", "zero_u_core"})):
            value = bundle_profile.get(field) if field != "oracle_outputs" else oracle_profile.get(field)
            if not isinstance(value, Mapping) or set(value) != expected_keys:
                raise Stage1EstimatorError(f"S1_5_{field.upper()}_KEY_SET_INVALID")
        if not isinstance(oracle_profile.get("sufficient_statistics"), Mapping) or set(oracle_profile["sufficient_statistics"]) != {"S1", "S2", "G1", "G2", "N1", "N2"}:
            raise Stage1EstimatorError("S1_5_ORACLE_STATISTICS_KEY_SET_INVALID")
        expected_sampling = {
            "provenance": source_manifest["double_provenance"],
            "sample_ids_a": source_manifest["double_sample_ids_a"],
            "sample_ids_b": source_manifest["double_sample_ids_b"],
            "sample_id_overlap_allowed": True,
            "a_b_exchange_passed": True,
            "m2_equals_double_passed": True,
        }
        if report_profile.get("double_sampling") != expected_sampling:
            raise Stage1EstimatorError("S1_5_REPORT_DOUBLE_PROVENANCE_BINDING_INVALID")
        if bundle_profile.get("double_provenance") != source_manifest["double_provenance"] or bundle_profile.get("double_sample_ids_a") != source_manifest["double_sample_ids_a"] or bundle_profile.get("double_sample_ids_b") != source_manifest["double_sample_ids_b"]:
            raise Stage1EstimatorError("S1_5_BUNDLE_DOUBLE_PROVENANCE_BINDING_INVALID")
        profile_dtype = _PROFILE_DTYPES.get(str(profile_name))
        if profile_dtype is None:
            raise Stage1EstimatorError("S1_5_PROFILE_DTYPE_BINDING_INVALID")
        expected_identical = _wire_samples(_manifest_samples(source_manifest, "identical_gradient_samples", dtype=profile_dtype, registry=_fixture_registry()))
        if bundle_profile.get("identical_gradient_samples") != expected_identical or bundle_profile.get("permutation_orders") != source_manifest["permutation_orders"]:
            raise Stage1EstimatorError("S1_5_BUNDLE_PERMUTATION_FIXTURE_BINDING_INVALID")
    for profile in report["profiles"]:
        if not isinstance(profile, Mapping) or tuple(item.get("comparison_id") for item in profile.get("comparisons", []) if isinstance(item, Mapping)) != _COMPARISON_IDS:
            raise Stage1EstimatorError("S1_5_COMPARISON_MATRIX_INVALID")
        if not all(item.get("passed") is True for item in profile["comparisons"] if isinstance(item, Mapping)):
            raise Stage1EstimatorError("S1_5_TENSOR_COMPARISON_FAILED")
        raw_clip = profile.get("raw_clip_boundary")
        if not isinstance(raw_clip, Mapping) or not all(isinstance(values, Mapping) and all(value is True for value in values.values()) for values in raw_clip.values()):
            raise Stage1EstimatorError("S1_5_RAW_CLIP_BOUNDARY_INVALID")
        sampling = profile.get("double_sampling")
        if not isinstance(sampling, Mapping) or sampling.get("sample_id_overlap_allowed") is not True or sampling.get("a_b_exchange_passed") is not True or sampling.get("m2_equals_double_passed") is not True:
            raise Stage1EstimatorError("S1_5_DOUBLE_SAMPLING_CONTRACT_INVALID")
        if not isinstance(profile.get("rejections"), Mapping) or set(profile["rejections"]) != set(_REJECTION_IDS) or len(profile["rejections"]) != len(_REJECTION_IDS) or not all(value is True for value in profile["rejections"].values()):
            raise Stage1EstimatorError("S1_5_REJECTION_CONTRACT_INVALID")
        if not all(float(value) < 0.0 for values in profile.get("negative_u", {}).values() for value in values):
            raise Stage1EstimatorError("S1_5_NEGATIVE_U_INVALID")
    derived = _derive_requirement_checks(report)
    gate_requirements = _derive_gate_requirements(report, table)
    if report.get("requirement_checks") != derived or gate.get("requirements") != gate_requirements or set(derived) != {"raw_core_and_scores", "multi_group_learning_rates", "raw_clip_separation", "double_provenance", "ordered_unordered_streaming_identity", "weighted_u_and_equal_degeneracy", "identical_gradient_degeneracy", "paired_permutation_invariance", "estimator_public_field_bindings", "negative_and_zero_boundaries", "m1_and_invalid_inputs_fail_closed", "all_tensor_rows_pass"} or set(gate_requirements) != set(derived) | {"table_projection_exact"} or not all(gate_requirements.values()):
        raise Stage1EstimatorError("S1_5_REQUIREMENT_DERIVATION_INVALID")
    return {"schema_version": "stage1-s1-5-evidence-validation-v1", "status": "PASS", "report_hash": report["report_hash"], "bundle_hash": bundle["bundle_hash"], "oracle_hash": oracle["oracle_hash"], "table_hash": table["table_hash"], "gate_artifact_hash": gate["artifact_hash"]}


def _stats_comparisons(profile: str, bundle_profile: Mapping[str, Any], oracle_profile: Mapping[str, Any]) -> list[dict[str, JSONValue]]:
    settings = _PROFILES[profile]
    output: list[dict[str, JSONValue]] = []
    saved = bundle_profile["sufficient_statistics"]
    expected = oracle_profile["sufficient_statistics"]
    for saved_name, expected_name, scale in (("S1", "S1", 4.0), ("S2", "S2", 16.0), ("G1", "G1", 40.0), ("G2", "G2", 160.0)):
        registry = _fixture_registry()
        contract = _coordinate_contract(registry)
        candidate = tensor_map_from_wire(saved[saved_name], field=f"saved.{saved_name}", coordinate_order=contract["coordinate_order"], registry=registry)
        oracle = tensor_map_from_wire(expected[expected_name], field=f"oracle.{expected_name}", coordinate_order=contract["coordinate_order"], registry=registry)
        comparison = _comparison(profile=profile, comparison_id=f"statistics_{saved_name}", actual=candidate, oracle=oracle, natural_scale=scale)
        output.append(comparison)
    for name in ("N1", "N2"):
        if not math.isclose(float(saved[name]), float(expected[name]), rel_tol=0.0, abs_tol=0.0):
            raise Stage1EstimatorError(f"S1_5_{name}_REPLAY_MISMATCH")
    return output


def replay_stage1_s15_evidence(
    evidence: Mapping[str, Any], *, source_root: str | Path | None = None
) -> dict[str, JSONValue]:
    """Offline replay from saved inputs, sufficient statistics, and oracle roles.

    This path imports no production estimator function.  It replays explicit
    FP64 formula loops, checks saved S1/S2/G1/G2/N1/N2, and then compares each
    saved production output coordinate-by-coordinate under its frozen profile.
    """

    manifest_root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[2]
    validated = validate_stage1_s15_evidence(evidence, source_root=manifest_root)
    fixture_manifest, _ = _load_fixture_manifest(manifest_root)
    bundle = evidence["tensor_bundle"]
    oracle_report = evidence["oracle_report"]
    report = evidence["estimator_report"]
    assert isinstance(bundle, Mapping) and isinstance(oracle_report, Mapping) and isinstance(report, Mapping)
    if report.get("estimator_input_contract") != fixture_manifest["estimator_input_contract"] or bundle.get("estimator_input_contract") != fixture_manifest["estimator_input_contract"]:
        raise Stage1EstimatorError("S1_5_REPLAY_INPUT_CONTRACT_MISMATCH")
    report_profiles = {item.get("profile"): item for item in report["profiles"] if isinstance(item, Mapping)}
    replay_profiles: list[dict[str, JSONValue]] = []
    for bundle_profile, oracle_profile in zip(bundle["profiles"], oracle_report["profiles"], strict=True):
        if not isinstance(bundle_profile, Mapping) or not isinstance(oracle_profile, Mapping):
            raise Stage1EstimatorError("S1_5_REPLAY_PROFILE_INVALID")
        profile = bundle_profile.get("profile")
        if profile not in _PROFILES or profile != oracle_profile.get("profile"):
            raise Stage1EstimatorError("S1_5_REPLAY_PROFILE_BINDING_INVALID")
        registry = _fixture_registry()
        coordinate_contract = _coordinate_contract(registry)
        if bundle.get("coordinate_contract") != coordinate_contract:
            raise Stage1EstimatorError("S1_5_REPLAY_COORDINATE_CONTRACT_INVALID")
        if bundle_profile.get("coordinate_registry_hash") != coordinate_contract["coordinate_registry_hash"] or bundle_profile.get("parameter_to_group") != coordinate_contract["parameter_to_group"]:
            raise Stage1EstimatorError("S1_5_REPLAY_PROFILE_REGISTRY_INVALID")
        report_profile = report_profiles.get(profile)
        if not isinstance(report_profile, Mapping):
            raise Stage1EstimatorError("S1_5_REPLAY_REPORT_PROFILE_INVALID")
        profile_dtype = _PROFILE_DTYPES[str(profile)]
        expected_samples = _manifest_samples(fixture_manifest, "microbatch_samples", dtype=profile_dtype, registry=registry)
        expected_identical = _manifest_samples(fixture_manifest, "identical_gradient_samples", dtype=profile_dtype, registry=registry)
        expected_left = _manifest_samples(fixture_manifest, "double_samples_a", dtype=profile_dtype, registry=registry)
        expected_right = _manifest_samples(fixture_manifest, "double_samples_b", dtype=profile_dtype, registry=registry)
        if bundle_profile.get("microbatch_samples") != _wire_samples(expected_samples) or bundle_profile.get("identical_gradient_samples") != _wire_samples(expected_identical) or bundle_profile.get("permutation_orders") != fixture_manifest["permutation_orders"] or bundle_profile.get("double_samples_a") != _wire_samples(expected_left) or bundle_profile.get("double_samples_b") != _wire_samples(expected_right) or bundle_profile.get("double_provenance") != fixture_manifest["double_provenance"] or bundle_profile.get("double_sample_ids_a") != fixture_manifest["double_sample_ids_a"] or bundle_profile.get("double_sample_ids_b") != fixture_manifest["double_sample_ids_b"]:
            raise Stage1EstimatorError("S1_5_FROZEN_FIXTURE_INPUT_MISMATCH")
        samples = _unwire_samples(bundle_profile.get("microbatch_samples"), field="replay_samples", coordinate_contract=coordinate_contract, registry=registry)
        identical = _unwire_samples(bundle_profile.get("identical_gradient_samples"), field="replay_identical_gradients", coordinate_contract=coordinate_contract, registry=registry)
        left = _unwire_samples(bundle_profile.get("double_samples_a"), field="replay_double_a", coordinate_contract=coordinate_contract, registry=registry)
        right = _unwire_samples(bundle_profile.get("double_samples_b"), field="replay_double_b", coordinate_contract=coordinate_contract, registry=registry)
        weights = bundle_profile.get("weights")
        if not isinstance(weights, list) or len(weights) != len(samples):
            raise Stage1EstimatorError("S1_5_REPLAY_WEIGHTS_INVALID")
        numeric_weights = [_json_number(value, field="replay_weight", positive=True) for value in weights]
        learning_rates = bundle_profile.get("learning_rates")
        if not isinstance(learning_rates, Mapping):
            raise Stage1EstimatorError("S1_5_REPLAY_LR_INVALID")
        if set(learning_rates) != set(_LEARNING_RATES):
            raise Stage1EstimatorError("S1_5_REPLAY_LR_KEY_INVALID")
        numeric_lrs = {str(key): _json_number(value, field=f"replay_lr_{key}", nonnegative=True) for key, value in learning_rates.items()}
        clip_factor = _json_number(bundle_profile.get("clip_factor"), field="replay_clip", nonnegative=True)
        if tuple(numeric_weights) != _WEIGHTS or numeric_lrs != _LEARNING_RATES or clip_factor != _CLIP_FACTOR:
            raise Stage1EstimatorError("S1_5_FROZEN_SCALAR_INPUT_MISMATCH")
        parameter_to_group = coordinate_contract["parameter_to_group"]
        permutation_orders = bundle_profile.get("permutation_orders")
        if not isinstance(permutation_orders, Mapping):
            raise Stage1EstimatorError("S1_5_REPLAY_PERMUTATION_ORDERS_INVALID")
        try:
            reverse_samples = tuple(samples[int(index)] for index in permutation_orders["reverse"])
            rotate_samples = tuple(samples[int(index)] for index in permutation_orders["rotate"])
            reverse_weights = tuple(numeric_weights[int(index)] for index in permutation_orders["reverse"])
            rotate_weights = tuple(numeric_weights[int(index)] for index in permutation_orders["rotate"])
        except (KeyError, TypeError, ValueError, IndexError) as error:
            raise Stage1EstimatorError("S1_5_REPLAY_PERMUTATION_ORDERS_INVALID") from error
        stats = sufficient_statistics(samples, numeric_weights)
        equal_weights = (1.0,) * len(samples)
        oracle_outputs = {
            "raw_core": raw_core(mean_gradient(samples)),
            "raw_score": apply_learning_rates(raw_core(mean_gradient(samples)), numeric_lrs, coordinate_to_group=parameter_to_group),
            "raw_clipped_score": apply_learning_rates(raw_core(mean_gradient(samples)), numeric_lrs, coordinate_to_group=parameter_to_group, clip_factor=clip_factor),
            "double_core": double_core(mean_gradient(left), mean_gradient(right)),
            "double_score": apply_learning_rates(double_core(mean_gradient(left), mean_gradient(right)), numeric_lrs, coordinate_to_group=parameter_to_group),
            "equal_u_ordered_core": ordered_pair_u_core(samples),
            "equal_u_unordered_core": unordered_pair_u_core(samples),
            "equal_u_streaming_core": ordered_pair_u_core(samples),
            "equal_u_score": apply_learning_rates(ordered_pair_u_core(samples), numeric_lrs, coordinate_to_group=parameter_to_group),
            "weighted_u_core": weighted_ordered_pair_u_core(samples, numeric_weights),
            "weighted_u_score": apply_learning_rates(weighted_ordered_pair_u_core(samples, numeric_weights), numeric_lrs, coordinate_to_group=parameter_to_group),
            "weighted_equal_count_degenerate_core": weighted_ordered_pair_u_core(samples, equal_weights),
            "m2_u_equals_double_core": double_core(samples[0], samples[1]),
            "double_exchange_core": double_core(mean_gradient(right), mean_gradient(left)),
            "identical_gradient_u_core": raw_core(mean_gradient(identical)),
            "identical_gradient_u_score": apply_learning_rates(raw_core(mean_gradient(identical)), numeric_lrs, coordinate_to_group=parameter_to_group),
            "equal_u_reverse_permutation_core": ordered_pair_u_core(reverse_samples),
            "equal_u_rotate_permutation_core": ordered_pair_u_core(rotate_samples),
            "weighted_u_reverse_paired_permutation_core": weighted_ordered_pair_u_core(reverse_samples, reverse_weights),
            "weighted_u_rotate_paired_permutation_core": weighted_ordered_pair_u_core(rotate_samples, rotate_weights),
        }
        if oracle_profile.get("oracle_outputs") != {name: tensor_map_to_wire(value) for name, value in oracle_outputs.items()}:
            raise Stage1EstimatorError("S1_5_ORACLE_ROLE_REPLAY_MISMATCH")
        stats_checks = _stats_comparisons(str(profile), bundle_profile, oracle_profile)
        statistics_wire = bundle_profile.get("sufficient_statistics")
        if report_profile.get("sufficient_statistics") != statistics_wire:
            raise Stage1EstimatorError("S1_5_REPLAY_REPORT_STATISTICS_MISMATCH")
        replay_oracle_statistics = {
            "S1": tensor_map_to_wire(stats["s1"]), "S2": tensor_map_to_wire(stats["s2"]),
            "G1": tensor_map_to_wire(stats["g1"]), "G2": tensor_map_to_wire(stats["g2"]),
            "N1": stats["n1"], "N2": stats["n2"],
        }
        if oracle_profile.get("sufficient_statistics") != replay_oracle_statistics:
            raise Stage1EstimatorError("S1_5_ORACLE_STATISTICS_REPLAY_MISMATCH")
        if not isinstance(statistics_wire, Mapping) or statistics_wire.get("M") != len(samples) or statistics_wire.get("N1") != stats["n1"] or statistics_wire.get("N2") != stats["n2"]:
            raise Stage1EstimatorError("S1_5_SAVED_STATISTICS_SCALAR_REPLAY_MISMATCH")
        saved_outputs = bundle_profile.get("production_outputs")
        if not isinstance(saved_outputs, Mapping) or set(saved_outputs) != set(oracle_outputs):
            raise Stage1EstimatorError("S1_5_PRODUCTION_OUTPUT_SET_INVALID")
        raw_saved = tensor_map_from_wire(saved_outputs["raw_core"], field="saved_output.raw_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        raw_score_saved = tensor_map_from_wire(saved_outputs["raw_score"], field="saved_output.raw_score", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        raw_clipped_saved = tensor_map_from_wire(saved_outputs["raw_clipped_score"], field="saved_output.raw_clipped_score", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        output_checks = [
            _comparison(profile=str(profile), comparison_id=name, actual=tensor_map_from_wire(saved_outputs[name], field=f"saved_output.{name}", coordinate_order=coordinate_contract["coordinate_order"], registry=registry), oracle=oracle_value, natural_scale=_CORE_SCALE if name.endswith("_core") else _SCORE_SCALE)
            for name, oracle_value in oracle_outputs.items()
        ]
        if output_checks != report_profile.get("comparisons"):
            raise Stage1EstimatorError("S1_5_REPORT_COMPARISON_REPLAY_MISMATCH")
        negative = _manifest_samples(fixture_manifest, "negative_samples", dtype=profile_dtype, registry=registry)
        zeros = _manifest_samples(fixture_manifest, "zero_samples", dtype=profile_dtype, registry=registry)
        boundary_outputs = {
            "raw_unclipped_core": raw_core(mean_gradient(samples)),
            "raw_clipped_core": raw_core(mean_gradient(samples)),
            "negative_u": ordered_pair_u_core(negative),
            "zero_raw_core": raw_core(mean_gradient(zeros)),
            "zero_u_core": ordered_pair_u_core(zeros),
        }
        boundary_wire = {name: tensor_map_to_wire(value) for name, value in boundary_outputs.items()}
        saved_boundaries = bundle_profile.get("boundary_outputs")
        if not isinstance(saved_boundaries, Mapping) or set(saved_boundaries) != set(boundary_outputs) or oracle_profile.get("boundary_outputs") != boundary_wire:
            raise Stage1EstimatorError("S1_5_BOUNDARY_REPLAY_MISMATCH")
        for name, oracle_boundary in boundary_outputs.items():
            candidate = tensor_map_from_wire(saved_boundaries[name], field=f"saved_boundary.{name}", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
            if not _comparison(profile=str(profile), comparison_id=f"boundary_{name}", actual=candidate, oracle=oracle_boundary, natural_scale=_CORE_SCALE)["passed"]:
                raise Stage1EstimatorError("S1_5_BOUNDARY_PRODUCTION_REPLAY_MISMATCH")
        if report_profile.get("negative_u") != _tensor_values(boundary_outputs["negative_u"]) or report_profile.get("zero_outputs") != {"raw_core": _tensor_values(boundary_outputs["zero_raw_core"]), "u_core": _tensor_values(boundary_outputs["zero_u_core"])}:
            raise Stage1EstimatorError("S1_5_REPORT_BOUNDARY_REPLAY_MISMATCH")
        saved_raw_unclipped = tensor_map_from_wire(saved_boundaries["raw_unclipped_core"], field="saved_boundary.raw_unclipped_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        saved_raw_clipped = tensor_map_from_wire(saved_boundaries["raw_clipped_core"], field="saved_boundary.raw_clipped_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        raw_score_from_clipped_core = TensorMap(
            {name: saved_raw_clipped[name] * numeric_lrs[str(parameter_to_group[name])] for name in saved_raw_clipped},
            registry=registry,
        )
        raw_clip_truth = {
            "raw_core_unchanged_by_clip": {
                name: bool(torch.equal(saved_raw_unclipped[name], saved_raw_clipped[name]))
                for name in saved_raw_unclipped
            },
            "raw_score_unchanged_by_clip": {
                name: bool(torch.equal(raw_score_saved[name], raw_score_from_clipped_core[name]))
                for name in raw_score_saved
            },
            "raw_clipped_one_factor": {
                name: bool(torch.equal(raw_clipped_saved[name], raw_score_saved[name] * clip_factor))
                for name in raw_score_saved
            },
        }
        if report_profile.get("raw_clip_boundary") != raw_clip_truth:
            raise Stage1EstimatorError("S1_5_RAW_CLIP_REPLAY_MISMATCH")
        ordered_production = tensor_map_from_wire(saved_outputs["equal_u_ordered_core"], field="saved_output.equal_u_ordered_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        unordered_production = tensor_map_from_wire(saved_outputs["equal_u_unordered_core"], field="saved_output.equal_u_unordered_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        streamed = tensor_map_from_wire(saved_outputs["equal_u_streaming_core"], field="saved_output.equal_u_streaming_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        ordered = tensor_map_from_wire(oracle_profile["oracle_outputs"]["equal_u_ordered_core"], field="oracle_output.equal_u_ordered_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        equal_core_saved = tensor_map_from_wire(saved_outputs["equal_u_streaming_core"], field="saved_output.equal_u_streaming_core", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        equal_score_saved = tensor_map_from_wire(saved_outputs["equal_u_score"], field="saved_output.equal_u_score", coordinate_order=coordinate_contract["coordinate_order"], registry=registry)
        if report_profile.get("scatter_rows") != _scatter_rows(str(profile), ordered_production, unordered_production, streamed, ordered) or report_profile.get("scaling_rows") != _scaling_rows(str(profile), raw_saved, raw_score_saved, raw_clipped_saved, equal_core_saved, equal_score_saved):
            raise Stage1EstimatorError("S1_5_REPORT_CHART_PROJECTION_REPLAY_MISMATCH")
        if not all(item["passed"] for item in (*stats_checks, *output_checks)):
            raise Stage1EstimatorError("S1_5_OFFLINE_REPLAY_NUMERICAL_MISMATCH")
        replay_profiles.append({"profile": profile, "statistics_comparisons": stats_checks, "output_comparisons": output_checks, "boundary_outputs": boundary_wire, "chart_rows": {"scatter_rows": report_profile["scatter_rows"], "scaling_rows": report_profile["scaling_rows"]}})
    replay: dict[str, JSONValue] = {
        "schema_version": "stage1-s1-5-offline-replay-v1", "status": "PASS", "source_report_hash": validated["report_hash"],
        "source_bundle_hash": validated["bundle_hash"], "source_oracle_hash": validated["oracle_hash"],
        "profiles": replay_profiles,
    }
    replay["replay_hash"] = _hash_role(replay, field="replay_hash")
    return replay


__all__ = [
    "BUNDLE_SCHEMA", "FIXTURE_ID", "GATE_ID", "GATE_SCHEMA", "INDEX_SCHEMA", "ORACLE_SCHEMA", "REPORT_SCHEMA",
    "Stage1EstimatorError", "TABLE_SCHEMA", "TASK_ID", "VALIDATION_SCHEMA", "build_stage1_s15_evidence",
    "replay_stage1_s15_evidence", "validate_stage1_s15_evidence",
]
