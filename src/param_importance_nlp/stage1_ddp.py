"""S1.8 distributed-gradient contracts and the real ``torchrun`` worker.

The module deliberately keeps two things separate:

* the small, pure route/aggregation functions are exercised on CPU; and
* :func:`execute_worker` is only a measurement producer.  It performs real
  NCCL/DDP ``no_sync`` backward calls, but never decides that a formal gate has
  passed.  The independent array-only replay in ``stage1_ddp_oracle`` and the
  formalizer own that decision.

This prevents a successful all-reduce or a matching loss checksum from hiding
an error in the microbatch partition, the diagonal-removal term, or the
optimizer bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import torch

from .atomic import sha256_file
from .contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from .core.accumulator import ImportanceAccumulator
from .core.tensors import TensorMap


TASK_ID = "stage1.08_ddp_and_gradient_accumulation"
WORKER_PLAN_SCHEMA = "stage1-s1-8-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage1-s1-8-worker-report-v1"
ARRAY_MANIFEST_SCHEMA = "stage1-s1-8-safetensors-manifest-v1"
FIXTURE_SCHEMA = "stage1-s1-8-fixture-manifest-v1"
PRE_ROUTE_SCALE_ORACLE_SCHEMA = "stage1-s1-8-pre-route-gradient-scale-oracle-v1"
ROUTES: tuple[str, ...] = ("A", "B", "C", "D")
CASES: tuple[str, ...] = ("equal", "weighted")
ROUTE_WORLD_SIZE = {"A": 1, "B": 1, "C": 2, "D": 4}
PERMUTATIONS: tuple[str, ...] = ("identity", "rank_swap", "local_reverse")
EXECUTION_MODES: tuple[str, ...] = ("formal", "ordinary_sync_negative", "inject_rank_failure")
MICROBATCH_COUNT = 8
T32_DISTRIBUTED_ATOL = 1.0e-7
T32_DISTRIBUTED_RTOL = 1.0e-4
T32_DISTRIBUTED_L2_LIMIT = 1.0e-4
NCCL_TRANSPORT_PROTOCOL = {
    "schema_version": "stage1-s1-8-nccl-transport-v1",
    "qualification_basis_gate_ids": ["stage0.G6", "stage0.G10"],
    "current_cuda_capability_artifact_hash": "a536e191cd59318325289d238db727f8939767e384bfccd961ae7ca1c6a11ce4",
    "nccl_p2p_disable": "1",
    "process_group_timeout_seconds": 180,
}


class Stage1S18Error(RuntimeError):
    """Raised for a malformed S1.8 route, plan, or worker observation."""


@dataclass(frozen=True, slots=True)
class RouteLayout:
    """One immutable assignment of the eight statistical units to ranks."""

    route: str
    world_size: int
    ranks: tuple[tuple[int, ...], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "route": self.route,
            "world_size": self.world_size,
            "rank_microbatch_ids": [list(ids) for ids in self.ranks],
        }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.device.type != "cpu":
        value = value.detach().cpu()
    return value.detach().contiguous().numpy().tobytes(order="C")


def tensor_map_digest(values: Mapping[str, torch.Tensor]) -> str:
    """Return an order-independent, dtype/shape-bound tensor-map digest."""

    digest = hashlib.sha256(b"stage1-s1-8-tensor-map-v1\0")
    for name in sorted(values):
        value = values[name]
        if not isinstance(name, str) or not name or not isinstance(value, torch.Tensor):
            raise Stage1S18Error("S18_TENSOR_MAP_ENTRY_INVALID")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _require_hash(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage1S18Error(f"S18_{field.upper()}_HASH_INVALID")
    return value


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise Stage1S18Error(f"S18_{field.upper()}_OBJECT_REQUIRED")
    return dict(value)


def with_artifact_hash(value: Mapping[str, object]) -> dict[str, object]:
    body = dict(value)
    if "artifact_hash" in body:
        raise Stage1S18Error("S18_ARTIFACT_HASH_MUST_BE_DERIVED")
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def self_hash_matches(value: Mapping[str, object]) -> bool:
    body = dict(value)
    declared = body.pop("artifact_hash", None)
    return isinstance(declared, str) and declared == canonical_json_hash(body)


def build_route_layout(route: str) -> RouteLayout:
    """Freeze the B/C/D units; A is only the full-batch gradient reference."""

    if route not in ROUTES:
        raise Stage1S18Error("S18_ROUTE_UNKNOWN")
    world_size = ROUTE_WORLD_SIZE[route]
    if route == "A":
        return RouteLayout(route, world_size, (tuple(range(MICROBATCH_COUNT)),))
    if MICROBATCH_COUNT % world_size:
        raise Stage1S18Error("S18_MICROBATCH_WORLD_SIZE_NOT_DIVISIBLE")
    width = MICROBATCH_COUNT // world_size
    layout = RouteLayout(
        route,
        world_size,
        tuple(tuple(rank * width + offset for offset in range(width)) for rank in range(world_size)),
    )
    validate_route_layout(layout)
    return layout


def validate_route_layout(layout: RouteLayout) -> None:
    """Reject missing, duplicated, or route-incompatible statistical units."""

    if layout.route not in ROUTES or layout.world_size != ROUTE_WORLD_SIZE.get(layout.route):
        raise Stage1S18Error("S18_ROUTE_LAYOUT_IDENTITY_INVALID")
    if len(layout.ranks) != layout.world_size or any(not ids for ids in layout.ranks):
        raise Stage1S18Error("S18_ROUTE_LAYOUT_RANKS_INVALID")
    flat = [item for rank in layout.ranks for item in rank]
    if layout.route == "A":
        if flat != list(range(MICROBATCH_COUNT)):
            raise Stage1S18Error("S18_A_FULL_BATCH_LAYOUT_INVALID")
        return
    if sorted(flat) != list(range(MICROBATCH_COUNT)) or len(flat) != len(set(flat)):
        raise Stage1S18Error("S18_MICROBATCH_PARTITION_INVALID")
    expected = MICROBATCH_COUNT // layout.world_size
    if any(len(ids) != expected for ids in layout.ranks):
        raise Stage1S18Error("S18_MICROBATCH_LOCAL_COUNT_INVALID")


def permute_route_layout(layout: RouteLayout, *, permutation: str) -> RouteLayout:
    """Return a provably equivalent rank or local-order permutation."""

    validate_route_layout(layout)
    if permutation not in PERMUTATIONS:
        raise Stage1S18Error("S18_ROUTE_PERMUTATION_UNKNOWN")
    if permutation == "identity" or layout.route == "A":
        return layout
    ranks = layout.ranks
    if permutation == "rank_swap":
        ranks = tuple(reversed(ranks))
    elif permutation == "local_reverse":
        ranks = tuple(tuple(reversed(items)) for items in ranks)
    candidate = RouteLayout(layout.route, layout.world_size, ranks)
    validate_route_layout(candidate)
    return candidate


def build_fixture(
    *,
    token_sha256: Mapping[str, str],
    upstream_fixture_hash: str,
    gradient_design_scale: float,
    optimizer_delta_design_scale: float,
    pre_route_scale_oracle_hash: str,
    pre_route_parameter_registry_hash: str,
    pre_route_case_state_checksums: Mapping[str, str],
) -> dict[str, object]:
    """Create the S1.8 fixture contract from S1.7's first eight records.

    ``weighted`` intentionally keeps the same records and microbatch boundaries
    but changes only ignored target suffixes.  It therefore proves weighted
    sufficient statistics without falsely treating a re-partitioned batch as
    the same finite-sample U-statistic.
    """

    if set(token_sha256) != {str(index) for index in range(16)}:
        raise Stage1S18Error("S18_UPSTREAM_TOKEN_HASH_SET_INVALID")
    for value in token_sha256.values():
        _require_hash(value, field="upstream_token")
    _require_hash(upstream_fixture_hash, field="upstream_fixture")
    _require_hash(pre_route_scale_oracle_hash, field="pre_route_scale_oracle")
    _require_hash(pre_route_parameter_registry_hash, field="pre_route_parameter_registry")
    if set(pre_route_case_state_checksums) != set(CASES):
        raise Stage1S18Error("S18_PRE_ROUTE_CASE_STATE_CHECKSUM_SET_INVALID")
    for checksum in pre_route_case_state_checksums.values():
        _require_hash(checksum, field="pre_route_case_state")
    if (
        isinstance(gradient_design_scale, bool) or not isinstance(gradient_design_scale, (int, float))
        or not math.isfinite(float(gradient_design_scale)) or float(gradient_design_scale) <= 0.0
    ):
        raise Stage1S18Error("S18_PRE_ROUTE_GRADIENT_DESIGN_SCALE_INVALID")
    if (
        isinstance(optimizer_delta_design_scale, bool) or not isinstance(optimizer_delta_design_scale, (int, float))
        or not math.isfinite(float(optimizer_delta_design_scale)) or float(optimizer_delta_design_scale) <= 0.0
    ):
        raise Stage1S18Error("S18_PRE_ROUTE_OPTIMIZER_DELTA_SCALE_INVALID")
    layouts = {route: build_route_layout(route).to_dict() for route in ROUTES}
    equal_weights = [2048] * MICROBATCH_COUNT
    weighted_weights = [2048, 2032, 2016, 2000, 1984, 1968, 1952, 1936]
    # ``n_g`` is a frozen *per-microbatch mean-gradient* scale.  Every
    # comparison scale is derived here, before a worker is launched, rather
    # than estimated from a potentially broken worker array.  In particular,
    # S2 and G2 cannot share one made-up ``sufficient_statistic`` scale.
    n_g = float(gradient_design_scale)
    eta = 0.0003
    scales = {
        "n_g": n_g,
        "s1": MICROBATCH_COUNT * n_g,
        "s2": MICROBATCH_COUNT * n_g * n_g,
        "g1": sum(weighted_weights) * n_g,
        "g2": sum(weight * weight for weight in weighted_weights) * n_g * n_g,
        "mean_gradient": n_g,
        "core": n_g * n_g,
        "score": eta * n_g * n_g,
        # This comes from the independent pre-route AdamW replay, rather than
        # a score-scale formula or a route under test.
        "optimizer_delta": float(optimizer_delta_design_scale),
        "parameter": 1.0,
    }
    body: dict[str, object] = {
        "schema_version": FIXTURE_SCHEMA,
        "fixture_id": "stage1-s1-8-pythia14m-ddp-v1",
        "upstream_s1_7_fixture_hash": upstream_fixture_hash,
        "pre_route_gradient_scale": {
            "schema_version": PRE_ROUTE_SCALE_ORACLE_SCHEMA,
            "artifact_hash": pre_route_scale_oracle_hash,
            "maximum_unit_gradient_abs": n_g,
            "maximum_abs_data_update": float(optimizer_delta_design_scale),
            "source": "independent_pre_route_single_gpu_autograd_oracle",
            "parameter_registry_hash": pre_route_parameter_registry_hash,
            "case_pre_parameter_checksums": dict(pre_route_case_state_checksums),
        },
        "record_ids": list(range(MICROBATCH_COUNT)),
        "token_sha256": {str(index): token_sha256[str(index)] for index in range(MICROBATCH_COUNT)},
        "routes": layouts,
        "cases": {
            "equal": {
                "label_ignore_suffixes": [0] * MICROBATCH_COUNT,
                "effective_target_tokens": equal_weights,
                "statistics": "equal_s1_s2",
            },
            "weighted": {
                "label_ignore_suffixes": [0, 16, 32, 48, 64, 80, 96, 112],
                "effective_target_tokens": weighted_weights,
                "statistics": "weighted_g1_g2_n1_n2",
            },
        },
        "precision": {
            "compute": "float32",
            "statistics": "float32",
            "replay": "float64",
            "profile": "T32_DISTRIBUTED",
            "atol": T32_DISTRIBUTED_ATOL,
            "rtol": T32_DISTRIBUTED_RTOL,
            "normalized_l2_limit": T32_DISTRIBUTED_L2_LIMIT,
        },
        "comparison_natural_scales": scales,
        "comparison_natural_scale_rules": {
        "n_g": "independent_pre_route_oracle_maximum_unit_gradient_abs",
            "s1": "M*n_g",
            "s2": "M*n_g^2",
            "g1": "sum(b_m)*n_g",
            "g2": "sum(b_m^2)*n_g^2",
            "mean_gradient": "n_g",
            "core": "n_g^2",
            "score": "eta*n_g^2",
            "optimizer_delta": "independent_pre_route_adamw_data_update_design_scale",
            "parameter": "frozen_parameter_scale",
        },
        "randomness": {"dropout": "disabled", "model_seed": 1707, "training_seed": 2707},
        "optimizer": {"type": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1e-8, "foreach": False, "fused": False},
        "gradient_clip_max_norm": 1.0,
        "ddp": {"backend": "nccl", "ordinary_gradient_collectives_during_local_backward": 0, "manual_statistic_collectives": "sum_all_reduce"},
    }
    body["fixture_hash"] = canonical_json_hash(body)
    return body


def validate_fixture(value: Mapping[str, object]) -> dict[str, Any]:
    fixture = _mapping(value, field="fixture")
    expected = {
        "schema_version", "fixture_id", "upstream_s1_7_fixture_hash", "pre_route_gradient_scale", "record_ids", "token_sha256",
        "routes", "cases", "precision", "comparison_natural_scales", "comparison_natural_scale_rules", "randomness", "optimizer", "gradient_clip_max_norm", "ddp", "fixture_hash",
    }
    if set(fixture) != expected or fixture.get("schema_version") != FIXTURE_SCHEMA:
        raise Stage1S18Error("S18_FIXTURE_SCHEMA_INVALID")
    declared = fixture.pop("fixture_hash")
    if not isinstance(declared, str) or declared != canonical_json_hash(fixture):
        raise Stage1S18Error("S18_FIXTURE_HASH_INVALID")
    fixture["fixture_hash"] = declared
    if fixture.get("record_ids") != list(range(MICROBATCH_COUNT)):
        raise Stage1S18Error("S18_FIXTURE_RECORD_IDS_INVALID")
    hashes = _mapping(fixture.get("token_sha256"), field="fixture.token_sha256")
    if set(hashes) != {str(index) for index in range(MICROBATCH_COUNT)}:
        raise Stage1S18Error("S18_FIXTURE_TOKEN_HASH_SET_INVALID")
    for digest in hashes.values():
        _require_hash(digest, field="fixture_token")
    routes = _mapping(fixture.get("routes"), field="fixture.routes")
    if set(routes) != set(ROUTES):
        raise Stage1S18Error("S18_FIXTURE_ROUTE_SET_INVALID")
    for route in ROUTES:
        route_value = _mapping(routes[route], field=f"fixture.routes.{route}")
        raw = route_value.get("rank_microbatch_ids")
        if not isinstance(raw, list) or any(not isinstance(item, list) for item in raw):
            raise Stage1S18Error("S18_FIXTURE_ROUTE_LAYOUT_INVALID")
        layout = RouteLayout(route, int(route_value.get("world_size", -1)), tuple(tuple(int(item) for item in items) for items in raw))
        validate_route_layout(layout)
    cases = _mapping(fixture.get("cases"), field="fixture.cases")
    if set(cases) != set(CASES):
        raise Stage1S18Error("S18_FIXTURE_CASE_SET_INVALID")
    for case in CASES:
        current = _mapping(cases[case], field=f"fixture.cases.{case}")
        suffixes, counts = current.get("label_ignore_suffixes"), current.get("effective_target_tokens")
        if (
            not isinstance(suffixes, list) or not isinstance(counts, list)
            or len(suffixes) != MICROBATCH_COUNT or len(counts) != MICROBATCH_COUNT
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 or item >= 2048 for item in suffixes)
            or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item > 2048 for item in counts)
            or any(2048 - int(suffix) != int(count) for suffix, count in zip(suffixes, counts, strict=True))
        ):
            raise Stage1S18Error("S18_FIXTURE_CASE_COUNTS_INVALID")
    if cases["equal"].get("effective_target_tokens") != [2048] * MICROBATCH_COUNT:
        raise Stage1S18Error("S18_EQUAL_CASE_NOT_EQUAL_WEIGHT")
    if len(set(cases["weighted"]["effective_target_tokens"])) <= 1:
        raise Stage1S18Error("S18_WEIGHTED_CASE_NOT_NONUNIFORM")
    clip = fixture.get("gradient_clip_max_norm")
    if isinstance(clip, bool) or not isinstance(clip, (int, float)) or not math.isfinite(float(clip)) or float(clip) <= 0:
        raise Stage1S18Error("S18_FIXTURE_CLIP_CONFIGURATION_INVALID")
    scales = _mapping(fixture.get("comparison_natural_scales"), field="fixture.comparison_natural_scales")
    expected_scale_keys = {"n_g", "s1", "s2", "g1", "g2", "mean_gradient", "core", "score", "optimizer_delta", "parameter"}
    if set(scales) != expected_scale_keys or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0
        for item in scales.values()
    ):
        raise Stage1S18Error("S18_FIXTURE_NATURAL_SCALES_INVALID")
    rules = _mapping(fixture.get("comparison_natural_scale_rules"), field="fixture.comparison_natural_scale_rules")
    expected_rules = {
        "n_g": "independent_pre_route_oracle_maximum_unit_gradient_abs", "s1": "M*n_g", "s2": "M*n_g^2",
        "g1": "sum(b_m)*n_g", "g2": "sum(b_m^2)*n_g^2", "mean_gradient": "n_g",
        "core": "n_g^2", "score": "eta*n_g^2", "optimizer_delta": "independent_pre_route_adamw_data_update_design_scale",
        "parameter": "frozen_parameter_scale",
    }
    if rules != expected_rules:
        raise Stage1S18Error("S18_FIXTURE_NATURAL_SCALE_RULES_INVALID")
    pre_route_scale = _mapping(fixture.get("pre_route_gradient_scale"), field="fixture.pre_route_gradient_scale")
    if (
        set(pre_route_scale) != {"schema_version", "artifact_hash", "maximum_unit_gradient_abs", "maximum_abs_data_update", "source", "parameter_registry_hash", "case_pre_parameter_checksums"}
        or pre_route_scale.get("schema_version") != PRE_ROUTE_SCALE_ORACLE_SCHEMA
        or pre_route_scale.get("source") != "independent_pre_route_single_gpu_autograd_oracle"
    ):
        raise Stage1S18Error("S18_FIXTURE_PRE_ROUTE_SCALE_SCHEMA_INVALID")
    _require_hash(pre_route_scale.get("artifact_hash"), field="pre_route_scale_oracle")
    _require_hash(pre_route_scale.get("parameter_registry_hash"), field="pre_route_parameter_registry")
    pre_route_case_checksums = _mapping(pre_route_scale.get("case_pre_parameter_checksums"), field="fixture.pre_route_scale.case_pre_parameter_checksums")
    if set(pre_route_case_checksums) != set(CASES):
        raise Stage1S18Error("S18_FIXTURE_PRE_ROUTE_CASE_STATE_CHECKSUM_SET_INVALID")
    for checksum in pre_route_case_checksums.values():
        _require_hash(checksum, field="pre_route_case_state")
    observed_scale = pre_route_scale.get("maximum_unit_gradient_abs")
    if isinstance(observed_scale, bool) or not isinstance(observed_scale, (int, float)) or not math.isfinite(float(observed_scale)) or float(observed_scale) <= 0.0:
        raise Stage1S18Error("S18_FIXTURE_PRE_ROUTE_SCALE_VALUE_INVALID")
    observed_delta_scale = pre_route_scale.get("maximum_abs_data_update")
    if isinstance(observed_delta_scale, bool) or not isinstance(observed_delta_scale, (int, float)) or not math.isfinite(float(observed_delta_scale)) or float(observed_delta_scale) <= 0.0:
        raise Stage1S18Error("S18_FIXTURE_PRE_ROUTE_OPTIMIZER_DELTA_SCALE_VALUE_INVALID")
    n_g = float(scales["n_g"])
    if n_g != float(observed_scale):
        raise Stage1S18Error("S18_FIXTURE_NG_SOURCE_BINDING_INVALID")
    learning_rate = float(_mapping(fixture.get("optimizer"), field="fixture.optimizer").get("learning_rate", math.nan))
    expected_scales = {
        "n_g": n_g,
        "s1": MICROBATCH_COUNT * n_g,
        "s2": MICROBATCH_COUNT * n_g * n_g,
        "g1": sum(int(item) for item in cases["weighted"]["effective_target_tokens"]) * n_g,
        "g2": sum(int(item) * int(item) for item in cases["weighted"]["effective_target_tokens"]) * n_g * n_g,
        "mean_gradient": n_g,
        "core": n_g * n_g,
        "score": learning_rate * n_g * n_g,
        "optimizer_delta": float(observed_delta_scale),
        "parameter": float(scales["parameter"]),
    }
    if any(not math.isclose(float(scales[key]), expected, rel_tol=0.0, abs_tol=0.0) for key, expected in expected_scales.items()):
        raise Stage1S18Error("S18_FIXTURE_NATURAL_SCALE_BINDING_INVALID")
    return fixture


def _assert_maps(samples: Sequence[Mapping[str, torch.Tensor]]) -> tuple[str, ...]:
    if not samples:
        raise Stage1S18Error("S18_STATISTICS_SAMPLES_EMPTY")
    names = tuple(sorted(samples[0]))
    if not names or any(tuple(sorted(sample)) != names for sample in samples):
        raise Stage1S18Error("S18_STATISTICS_PARAMETER_SET_DRIFT")
    shapes = {name: tuple(samples[0][name].shape) for name in names}
    for sample in samples:
        for name in names:
            value = sample[name]
            if value.dtype != torch.float32 or tuple(value.shape) != shapes[name] or not bool(torch.isfinite(value).all()):
                raise Stage1S18Error(f"S18_STATISTICS_TENSOR_INVALID:{name}")
    return names


def local_sufficient_statistics(
    samples: Sequence[Mapping[str, torch.Tensor]], *, weights: Sequence[int]
) -> dict[str, object]:
    """Compute local equal and weighted statistics without any collective.

    The public worker applies a sum all-reduce to these values exactly once per
    statistic.  Keeping this function collective-free makes accidental use of
    a DDP-reduced gradient visible in the test and formal replay contracts.
    """

    names = _assert_maps(samples)
    if len(samples) != len(weights) or not weights or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in weights):
        raise Stage1S18Error("S18_STATISTICS_WEIGHTS_INVALID")
    result: dict[str, dict[str, torch.Tensor]] = {key: {} for key in ("s1", "s2", "g1", "g2")}
    for name in names:
        reference = samples[0][name]
        s1 = torch.zeros_like(reference)
        s2 = torch.zeros_like(reference)
        g1 = torch.zeros_like(reference)
        g2 = torch.zeros_like(reference)
        for sample, weight in zip(samples, weights, strict=True):
            value = sample[name]
            s1.add_(value); s2.add_(value.square())
            g1.add_(value, alpha=float(weight)); g2.add_(value.square(), alpha=float(weight * weight))
        result["s1"][name], result["s2"][name] = s1, s2
        result["g1"][name], result["g2"][name] = g1, g2
    return {**result, "m": len(samples), "n1": int(sum(weights)), "n2": int(sum(weight * weight for weight in weights))}


def scores_from_global_statistics(statistics: Mapping[str, object], *, learning_rates: Mapping[str, float]) -> dict[str, dict[str, torch.Tensor]]:
    """Derive mean/raw/U exactly once from globally reduced sufficient stats."""

    required = {"s1", "s2", "g1", "g2", "m", "n1", "n2"}
    if set(statistics) != required:
        raise Stage1S18Error("S18_GLOBAL_STATISTIC_FIELDS_INVALID")
    maps = {name: _mapping(statistics[name], field=f"statistics.{name}") for name in ("s1", "s2", "g1", "g2")}
    names = tuple(sorted(maps["s1"]))
    if not names or any(set(values) != set(names) for values in maps.values()):
        raise Stage1S18Error("S18_GLOBAL_STATISTIC_PARAMETER_SET_INVALID")
    m, n1, n2 = statistics["m"], statistics["n1"], statistics["n2"]
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (m, n1, n2)) or int(m) < 2 or int(n1) <= 0 or int(n1) * int(n1) <= int(n2):
        raise Stage1S18Error("S18_GLOBAL_STATISTIC_DENOMINATOR_INVALID")
    output = {field: {} for field in ("equal_mean", "equal_raw", "equal_u", "weighted_mean", "weighted_raw", "weighted_u", "equal_score", "weighted_score")}
    for name in names:
        values = {field: maps[field][name] for field in maps}
        if not all(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.shape == values["s1"].shape for value in values.values()):
            raise Stage1S18Error(f"S18_GLOBAL_STATISTIC_TENSOR_INVALID:{name}")
        if name not in learning_rates or not math.isfinite(float(learning_rates[name])) or float(learning_rates[name]) < 0:
            raise Stage1S18Error(f"S18_LEARNING_RATE_INVALID:{name}")
        equal_mean = values["s1"] / int(m)
        weighted_mean = values["g1"] / int(n1)
        equal_u = (values["s1"].square() - values["s2"]) / (int(m) * (int(m) - 1))
        weighted_u = (values["g1"].square() - values["g2"]) / (int(n1) * int(n1) - int(n2))
        output["equal_mean"][name] = equal_mean
        output["equal_raw"][name] = equal_mean.square()
        output["equal_u"][name] = equal_u
        output["weighted_mean"][name] = weighted_mean
        output["weighted_raw"][name] = weighted_mean.square()
        output["weighted_u"][name] = weighted_u
        output["equal_score"][name] = equal_u * float(learning_rates[name])
        output["weighted_score"][name] = weighted_u * float(learning_rates[name])
    return output


def _scores_from_pair(
    first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor], *, denominator: int,
    mean_denominator: int, learning_rates: Mapping[str, float], prefix: str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Compute one statistic family's mean/raw/U/score without cross-family data."""

    if denominator <= 0 or mean_denominator <= 0 or set(first) != set(second) or not first:
        raise Stage1S18Error("S18_SCORE_PAIR_DENOMINATOR_OR_KEYS_INVALID")
    output = {field: {} for field in ("mean", "raw", "u", "score")}
    for name in sorted(first):
        left, right = first[name], second[name]
        if (
            not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor)
            or left.dtype != torch.float32 or right.dtype != torch.float32 or left.shape != right.shape
            or name not in learning_rates or not math.isfinite(float(learning_rates[name]))
            or float(learning_rates[name]) < 0
        ):
            raise Stage1S18Error(f"S18_{prefix.upper()}_SCORE_TENSOR_OR_LR_INVALID:{name}")
        mean = left / mean_denominator
        score = (left.square() - right) / denominator
        output["mean"][name] = mean
        output["raw"][name] = mean.square()
        output["u"][name] = score
        output["score"][name] = score * float(learning_rates[name])
    return output


def validate_worker_plan(value: Mapping[str, object], *, path: Path) -> dict[str, Any]:
    plan = _mapping(value, field="worker_plan")
    expected = {
        "schema_version", "task_id", "execution_commit", "route", "fixture", "fixture_tokens_ref",
        "fixture_tokens_sha256", "data_root", "model_root", "cache_root", "run_token", "visible_gpu_uuids", "nccl_transport_protocol",
        "output_dir", "permutation", "execution_mode", "artifact_hash",
    }
    if set(plan) != expected or plan.get("schema_version") != WORKER_PLAN_SCHEMA or plan.get("task_id") != TASK_ID or not self_hash_matches(plan):
        raise Stage1S18Error("S18_WORKER_PLAN_SCHEMA_INVALID")
    if plan.get("route") not in ROUTES or not isinstance(plan.get("execution_commit"), str) or len(str(plan["execution_commit"])) != 40:
        raise Stage1S18Error("S18_WORKER_PLAN_IDENTITY_INVALID")
    fixture = validate_fixture(_mapping(plan.get("fixture"), field="worker_plan.fixture"))
    if not isinstance(plan.get("fixture_tokens_ref"), str) or Path(str(plan["fixture_tokens_ref"])).is_absolute():
        raise Stage1S18Error("S18_WORKER_FIXTURE_REFERENCE_INVALID")
    if not isinstance(plan.get("output_dir"), str) or Path(str(plan["output_dir"])).is_absolute():
        raise Stage1S18Error("S18_WORKER_OUTPUT_REFERENCE_INVALID")
    if not isinstance(plan.get("run_token"), str) or len(str(plan["run_token"])) != 64:
        raise Stage1S18Error("S18_WORKER_RUN_TOKEN_INVALID")
    if plan.get("permutation") not in PERMUTATIONS or plan.get("execution_mode") not in EXECUTION_MODES:
        raise Stage1S18Error("S18_WORKER_EXECUTION_VARIANT_INVALID")
    if plan.get("nccl_transport_protocol") != NCCL_TRANSPORT_PROTOCOL:
        raise Stage1S18Error("S18_WORKER_NCCL_TRANSPORT_PROTOCOL_INVALID")
    uuids = plan.get("visible_gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != ROUTE_WORLD_SIZE[str(plan["route"])] or len(set(uuids)) != len(uuids) or any(not isinstance(item, str) or not item.startswith("GPU-") for item in uuids):
        raise Stage1S18Error("S18_WORKER_UUID_SET_INVALID")
    _require_hash(plan.get("fixture_tokens_sha256"), field="fixture_tokens")
    fixture_path = (path.parent / str(plan["fixture_tokens_ref"])).resolve()
    if not fixture_path.is_file() or sha256_file(fixture_path) != plan["fixture_tokens_sha256"]:
        raise Stage1S18Error("S18_WORKER_FIXTURE_FILE_HASH_INVALID")
    if fixture["routes"][str(plan["route"])] != build_route_layout(str(plan["route"])).to_dict():
        raise Stage1S18Error("S18_WORKER_ROUTE_FIXTURE_DRIFT")
    try:
        data_root = Path(str(plan.get("data_root"))).resolve(strict=True)
        model_root = Path(str(plan.get("model_root"))).resolve(strict=True)
        cache_root = Path(str(plan.get("cache_root"))).resolve(strict=True)
        model_root.relative_to(data_root); cache_root.relative_to(data_root)
    except (OSError, TypeError, ValueError) as error:
        raise Stage1S18Error("S18_WORKER_DATA_ROOT_CONFINEMENT_INVALID") from error
    if not model_root.is_dir() or cache_root != data_root / "cache":
        raise Stage1S18Error("S18_WORKER_MODEL_OR_CACHE_BINDING_INVALID")
    return plan


def seed_rank_local_generators(*, seed: int, local_rank: int) -> dict[str, int]:
    """Seed CPU plus exactly the already-selected rank-local CUDA generator.

    ``torch.manual_seed`` reaches every visible CUDA generator.  Formal
    torchrun workers intentionally see a UUID set, not a single device, so
    only the current device selected for ``local_rank`` may be touched.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or isinstance(local_rank, bool) or not isinstance(local_rank, int) or local_rank < 0:
        raise Stage1S18Error("S18_RANK_LOCAL_SEED_INVALID")
    torch.random.default_generator.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return {"cpu_default_seed": seed, "cuda_current_device_seed": seed, "local_rank": local_rank}


def validate_nccl_transport_environment() -> dict[str, object]:
    """Fail before CUDA/NCCL initialization unless the frozen transport is set."""

    if os.environ.get("NCCL_P2P_DISABLE") != "1":
        raise Stage1S18Error("S18_NCCL_P2P_ENVIRONMENT_INVALID")
    return dict(NCCL_TRANSPORT_PROTOCOL)


def _load_tokens(path: Path, fixture: Mapping[str, object]) -> dict[int, torch.Tensor]:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - server-only optional dependency
        raise Stage1S18Error("S18_SAFETENSORS_UNAVAILABLE") from error
    values = load_file(str(path), device="cpu")
    required = {f"record_{index:012d}" for index in range(16)}
    if set(values) != required:
        raise Stage1S18Error("S18_FIXTURE_TENSOR_SET_INVALID")
    hashes = _mapping(fixture["token_sha256"], field="fixture.token_sha256")
    result: dict[int, torch.Tensor] = {}
    for index in range(MICROBATCH_COUNT):
        value = values[f"record_{index:012d}"]
        if value.dtype != torch.int64 or tuple(value.shape) != (2049,) or not bool(torch.all((value >= 0) & (value < 50277))):
            raise Stage1S18Error(f"S18_FIXTURE_TOKEN_INVALID:{index}")
        if hashlib.sha256(value.contiguous().numpy().tobytes(order="C")).hexdigest() != hashes[str(index)]:
            raise Stage1S18Error(f"S18_FIXTURE_TOKEN_HASH_INVALID:{index}")
        result[index] = value.contiguous()
    return result


def _parameter_maps(module: torch.nn.Module, *, gradients: bool) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, parameter in sorted(module.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        key = name.removeprefix("module.")
        value = parameter.grad if gradients else parameter
        if value is None:
            raise Stage1S18Error(f"S18_PARAMETER_GRADIENT_MISSING:{key}")
        values[key] = value.detach().to(torch.float32).cpu().clone().contiguous()
    if not values:
        raise Stage1S18Error("S18_ELIGIBLE_PARAMETER_SET_EMPTY")
    return values


def _parameter_registry_hash(module: torch.nn.Module) -> str:
    """Bind the pre-route scale oracle to the exact route parameter topology."""

    parameters = [
        (name.removeprefix("module."), parameter)
        for name, parameter in module.named_parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise Stage1S18Error("S18_PARAMETER_REGISTRY_EMPTY")
    return canonical_json_hash({
        "names": [name for name, _ in parameters],
        "shapes": {name: list(parameter.shape) for name, parameter in parameters},
        "dtypes": {name: str(parameter.dtype) for name, parameter in parameters},
    })


def learning_rate_map(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Bind optimizer groups by ``Parameter`` identity, never tensor equality.

    List membership for tensors calls ``__eq__``.  Besides being semantically
    wrong for a parameter registry, a non-first multidimensional parameter can
    raise an ambiguous-bool error.  This identity map rejects unknown,
    duplicate, frozen, and omitted parameters before scores are calculated.
    """

    named: dict[int, str] = {}
    for raw_name, parameter in module.named_parameters():
        if parameter.requires_grad:
            name = raw_name.removeprefix("module.")
            if id(parameter) in named or name in named.values():
                raise Stage1S18Error("S18_LEARNING_RATE_MODEL_ALIAS_OR_NAME_DRIFT")
            named[id(parameter)] = name
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        value = group.get("lr")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise Stage1S18Error("S18_LEARNING_RATE_GROUP_INVALID")
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise Stage1S18Error("S18_OPTIMIZER_GROUP_PARAMETERS_INVALID")
        for parameter in parameters:
            if not isinstance(parameter, torch.nn.Parameter) or id(parameter) not in named:
                raise Stage1S18Error("S18_OPTIMIZER_PARAMETER_UNKNOWN_OR_FROZEN")
            name = named[id(parameter)]
            if name in result:
                raise Stage1S18Error("S18_OPTIMIZER_PARAMETER_DUPLICATE")
            result[name] = float(value)
    if set(result) != set(named.values()):
        raise Stage1S18Error("S18_OPTIMIZER_PARAMETER_OMISSION")
    return result


def weight_decay_map(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Bind decoupled AdamW decay to parameters by exact object identity."""

    named = {
        id(parameter): raw_name.removeprefix("module.")
        for raw_name, parameter in module.named_parameters() if parameter.requires_grad
    }
    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        decay = group.get("weight_decay")
        parameters = group.get("params")
        if (
            isinstance(decay, bool) or not isinstance(decay, (int, float))
            or not math.isfinite(float(decay)) or float(decay) < 0
            or not isinstance(parameters, list)
        ):
            raise Stage1S18Error("S18_OPTIMIZER_WEIGHT_DECAY_GROUP_INVALID")
        for parameter in parameters:
            name = named.get(id(parameter)) if isinstance(parameter, torch.nn.Parameter) else None
            if name is None:
                raise Stage1S18Error("S18_OPTIMIZER_WEIGHT_DECAY_PARAMETER_UNKNOWN")
            if name in result:
                raise Stage1S18Error("S18_OPTIMIZER_WEIGHT_DECAY_PARAMETER_DUPLICATE")
            result[name] = float(decay)
    if set(result) != set(named.values()):
        raise Stage1S18Error("S18_OPTIMIZER_WEIGHT_DECAY_PARAMETER_OMISSION")
    return result


def _optimizer_state_maps(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    """Serialize every AdamW state tensor under its canonical parameter name."""

    names = {id(parameter): raw_name.removeprefix("module.") for raw_name, parameter in module.named_parameters() if parameter.requires_grad}
    values: dict[str, torch.Tensor] = {}
    for parameter, state in optimizer.state.items():
        name = names.get(id(parameter))
        if name is None or not isinstance(state, Mapping):
            raise Stage1S18Error("S18_OPTIMIZER_STATE_PARAMETER_DRIFT")
        if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise Stage1S18Error("S18_ADAMW_STATE_FIELD_DRIFT")
        for field, value in state.items():
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise Stage1S18Error(f"S18_ADAMW_STATE_INVALID:{name}:{field}")
            values[f"{name}::{field}"] = value.detach().to(torch.float32).cpu().clone().contiguous()
    expected = {f"{name}::{field}" for name in names.values() for field in ("step", "exp_avg", "exp_avg_sq")}
    if set(values) != expected:
        raise Stage1S18Error("S18_ADAMW_STATE_COVERAGE_DRIFT")
    return values


def _stable_l2_components(values: Mapping[str, torch.Tensor]) -> tuple[float, float]:
    """Return a LAPACK-style scaled sum-of-squares representation.

    ``sqrt(sum(x*x))`` overflows for perfectly valid finite FP32 values near
    1e20 and underflows for values near 1e-30.  The worker's clipping contract
    must therefore use the same stable FP64 formulation as the replay oracle.
    The returned norm is ``scale * sqrt(ssq)`` when that real number fits.
    """

    if not values:
        raise Stage1S18Error("S18_STABLE_NORM_VALUES_EMPTY")
    scale, ssq = 0.0, 1.0
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise Stage1S18Error(f"S18_STABLE_NORM_TENSOR_INVALID:{name}")
        flat = value.detach().to(torch.float64).reshape(-1)
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


def _stable_l2(values: Mapping[str, torch.Tensor]) -> float:
    scale, ssq = _stable_l2_components(values)
    return 0.0 if scale == 0.0 else scale * math.sqrt(ssq)


def _tensor_map(values: Mapping[str, torch.Tensor]) -> TensorMap:
    """Make the accumulator's explicit CPU/FP32 coordinate map."""

    if not values:
        raise Stage1S18Error("S18_ACCUMULATOR_VALUES_EMPTY")
    result = {
        name: value.detach().to(torch.float32).cpu().clone().contiguous()
        for name, value in sorted(values.items())
    }
    if any(not bool(torch.isfinite(value).all()) for value in result.values()):
        raise Stage1S18Error("S18_ACCUMULATOR_VALUES_NONFINITE")
    return TensorMap(result)


def _plain_map(values: TensorMap) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().to(torch.float32).cpu().clone().contiguous()
        for name, value in values.items()
    }


def _accumulator_views(accumulator: ImportanceAccumulator) -> dict[str, dict[str, torch.Tensor]]:
    """Persist all public long-horizon views after one committed success."""

    accumulator.validate_invariants(atol=T32_DISTRIBUTED_ATOL, rtol=T32_DISTRIBUTED_RTOL)
    return {
        "signed": _plain_map(accumulator.signed),
        "positive": _plain_map(accumulator.positive),
        "negative_mass": _plain_map(accumulator.negative_mass),
        "absolute": _plain_map(accumulator.absolute),
        "raw": _plain_map(accumulator.raw),
        "raw_clipped": _plain_map(accumulator.raw_clipped),
        "data_movement": _plain_map(accumulator.data_movement),
        "net_data_movement": _plain_map(accumulator.net_data_movement),
        "total_movement": _plain_map(accumulator.total_movement),
        "total_endpoint_movement": _plain_map(accumulator.total_endpoint_movement),
        "weight_decay_movement": _plain_map(accumulator.weight_decay_movement),
        "net_weight_decay_movement": _plain_map(accumulator.net_weight_decay_movement),
        "actual_update_raw_importance": _plain_map(accumulator.actual_update_raw_importance),
        "magnitude": _plain_map(accumulator.magnitude),
    }


def _clip_factor(mean: Mapping[str, torch.Tensor], *, max_norm: float) -> tuple[float, float]:
    if not math.isfinite(max_norm) or max_norm <= 0 or not mean:
        raise Stage1S18Error("S18_CLIP_CONFIGURATION_INVALID")
    norm = _stable_l2(mean)
    factor = min(1.0, max_norm / (norm + 1.0e-6))
    if not math.isfinite(norm) or not math.isfinite(factor) or not 0.0 < factor <= 1.0:
        raise Stage1S18Error("S18_CLIP_FACTOR_INVALID")
    return norm, factor


def _batch(tokens: Mapping[int, torch.Tensor], ids: Sequence[int], suffixes: Sequence[int], *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    joined = torch.stack([tokens[index] for index in ids], dim=0).to(device, non_blocking=True)
    inputs, labels = joined[:, :-1], joined[:, 1:].clone()
    for row, index in enumerate(ids):
        suffix = int(suffixes[index])
        if suffix:
            labels[row, -suffix:] = -100
    count = int((labels != -100).sum().item())
    if count <= 0:
        raise Stage1S18Error("S18_BATCH_EFFECTIVE_TOKEN_COUNT_ZERO")
    return inputs, labels, count


def _load_model(plan: Mapping[str, object], device: torch.device) -> torch.nn.Module:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as error:  # pragma: no cover - server-only optional dependency
        raise Stage1S18Error("S18_TRANSFORMERS_UNAVAILABLE") from error
    model = AutoModelForCausalLM.from_pretrained(
        str(plan["model_root"]), local_files_only=True, torch_dtype=torch.float32,
    ).to(device)
    for field in ("attention_dropout", "hidden_dropout"):
        if float(getattr(model.config, field, math.nan)) != 0.0:
            raise Stage1S18Error(f"S18_RANDOM_DROPOUT_ACTIVE:{field}")
    model.train()
    return model


def _all_reduce_map(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    import torch.distributed as dist

    output: dict[str, torch.Tensor] = {}
    for name in sorted(values):
        value = values[name].detach().clone().to(torch.device("cuda", int(os.environ["LOCAL_RANK"])))
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        output[name] = value.cpu()
    return output


def _all_reduce_int(value: int, *, device: torch.device) -> int:
    import torch.distributed as dist

    item = torch.tensor([value], dtype=torch.int64, device=device)
    dist.all_reduce(item, op=dist.ReduceOp.SUM)
    return int(item.item())


def _all_reduce_float(value: float, *, device: torch.device) -> float:
    """Sum a loss numerator without rounding it through an integer surrogate."""

    import torch.distributed as dist

    item = torch.tensor([value], dtype=torch.float64, device=device)
    dist.all_reduce(item, op=dist.ReduceOp.SUM)
    return float(item.item())


def _save_route_arrays(path: Path, values: Mapping[str, torch.Tensor]) -> dict[str, object]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover - server-only optional dependency
        raise Stage1S18Error("S18_SAFETENSORS_UNAVAILABLE") from error
    if path.exists():
        raise Stage1S18Error("S18_ROUTE_ARRAY_FILE_COLLISION")
    save_file(dict(values), str(path), metadata={"schema_version": ARRAY_MANIFEST_SCHEMA})
    tensors: dict[str, object] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(values) or handle.metadata() != {"schema_version": ARRAY_MANIFEST_SCHEMA}:
            raise Stage1S18Error("S18_ROUTE_ARRAY_SAVE_DRIFT")
        for key in sorted(values):
            tensor = handle.get_tensor(key)
            tensors[key] = {"sha256": hashlib.sha256(_tensor_bytes(tensor)).hexdigest(), "dtype": str(tensor.dtype), "shape": list(tensor.shape)}
    return with_artifact_hash({"schema_version": ARRAY_MANIFEST_SCHEMA, "file": path.name, "file_sha256": sha256_file(path), "file_size_bytes": path.stat().st_size, "tensors": tensors})


def _run_case(
    *, ddp: torch.nn.Module, optimizer: torch.optim.Optimizer, tokens: Mapping[int, torch.Tensor],
    layout: RouteLayout, case: str, fixture: Mapping[str, object], device: torch.device,
    comm_state: dict[str, int], route: str, suppress_ddp_sync: bool,
    accumulator: ImportanceAccumulator | None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Run one equal or weighted case and return rank-zero serializable arrays."""

    import torch.distributed as dist

    current = _mapping(_mapping(fixture["cases"], field="fixture.cases")[case], field=f"fixture.case.{case}")
    suffixes = current["label_ignore_suffixes"]
    if not isinstance(suffixes, list):  # validated by the plan, retained for type narrowing
        raise Stage1S18Error("S18_CASE_SUFFIXES_INVALID")
    rank = int(dist.get_rank())
    local_ids = layout.ranks[rank]
    local_samples: list[dict[str, torch.Tensor]] = []
    local_weights: list[int] = []
    local_loss_numerator = 0.0
    learning_rates = learning_rate_map(ddp, optimizer)
    weight_decays = weight_decay_map(ddp, optimizer)
    optimizer.zero_grad(set_to_none=True)
    if route == "A":
        inputs, labels, count = _batch(tokens, local_ids, suffixes, device=device)
        output = ddp(input_ids=inputs, labels=labels)
        if output.loss is None:
            raise Stage1S18Error("S18_MODEL_LOSS_MISSING")
        output.loss.backward()
        local_samples.append(_parameter_maps(ddp, gradients=True))
        local_weights.append(count)
        local_loss_numerator = float(output.loss.detach().cpu()) * count
    else:
        for microbatch_id in local_ids:
            optimizer.zero_grad(set_to_none=True)
            inputs, labels, count = _batch(tokens, (microbatch_id,), suffixes, device=device)
            context = ddp.no_sync() if suppress_ddp_sync and hasattr(ddp, "no_sync") else nullcontext()
            with context:
                output = ddp(input_ids=inputs, labels=labels)
                if output.loss is None:
                    raise Stage1S18Error("S18_MODEL_LOSS_MISSING")
                output.loss.backward()
            local_samples.append(_parameter_maps(ddp, gradients=True))
            local_weights.append(count)
            local_loss_numerator += float(output.loss.detach().cpu()) * count
    if route == "A":
        # A is deliberately *not* an U-statistic route.  It has exactly one
        # full-global-batch backward and only proves mean/raw/update parity.
        installed = local_samples[0]
        total_m = MICROBATCH_COUNT
        total_n1 = local_weights[0]
        total_n2 = sum(item * item for item in _mapping(_mapping(fixture["cases"], field="fixture.cases")[case], field=f"fixture.case.{case}")["effective_target_tokens"])
        total_valid_tokens = local_weights[0]
        total_loss_numerator = local_loss_numerator
        global_stats: dict[str, object] = {}
        scores: dict[str, dict[str, torch.Tensor]] = {
            "mean": installed,
            "raw": {name: value.square() for name, value in installed.items()},
        }
        scalar_collective_count = 0
        tensor_collective_count = 0
    else:
        local_stats = local_sufficient_statistics(local_samples, weights=local_weights)
        total_loss_numerator = _all_reduce_float(local_loss_numerator, device=device)
        total_valid_tokens = _all_reduce_int(sum(local_weights), device=device)
        if case == "equal":
            total_m = _all_reduce_int(int(local_stats["m"]), device=device)
            if total_valid_tokens != total_m * 2048:
                raise Stage1S18Error("S18_EQUAL_VALID_TOKEN_COUNT_DRIFT")
            global_stats = {
                "s1": _all_reduce_map(_mapping(local_stats["s1"], field="local.s1")),
                "s2": _all_reduce_map(_mapping(local_stats["s2"], field="local.s2")),
            }
            total_n1, total_n2 = total_valid_tokens, total_m * 2048 * 2048
            scores = _scores_from_pair(
                _mapping(global_stats["s1"], field="global.s1"), _mapping(global_stats["s2"], field="global.s2"),
                denominator=total_m * (total_m - 1), mean_denominator=total_m,
                learning_rates=learning_rates, prefix="equal",
            )
            scalar_collective_count, tensor_collective_count = 3, 2 * len(scores["mean"])
        else:
            total_n1 = _all_reduce_int(int(local_stats["n1"]), device=device)
            total_n2 = _all_reduce_int(int(local_stats["n2"]), device=device)
            if total_n1 != total_valid_tokens:
                raise Stage1S18Error("S18_WEIGHTED_N1_VALID_TOKEN_COUNT_DRIFT")
            total_m = MICROBATCH_COUNT
            global_stats = {
                "g1": _all_reduce_map(_mapping(local_stats["g1"], field="local.g1")),
                "g2": _all_reduce_map(_mapping(local_stats["g2"], field="local.g2")),
            }
            scores = _scores_from_pair(
                _mapping(global_stats["g1"], field="global.g1"), _mapping(global_stats["g2"], field="global.g2"),
                denominator=total_n1 * total_n1 - total_n2, mean_denominator=total_n1,
                learning_rates=learning_rates, prefix="weighted",
            )
            scalar_collective_count, tensor_collective_count = 4, 2 * len(scores["mean"])
        installed = scores["mean"]
    pre_parameters = _parameter_maps(ddp, gradients=False)
    if case == "equal":
        # AdamW has no allocated state until its first success.  Persist the
        # explicit mathematical zero state so the replay can still bind the
        # equal pre-state and, critically, post(equal)==pre(weighted).
        optimizer_states_pre = {
            f"{name}::{field}": (
                torch.zeros_like(value) if field != "step" else torch.zeros((), dtype=torch.float32)
            )
            for name, value in pre_parameters.items() for field in ("step", "exp_avg", "exp_avg_sq")
        }
    else:
        optimizer_states_pre = _optimizer_state_maps(ddp, optimizer)
    global_gradient_norm, clip_factor = _clip_factor(
        installed, max_norm=float(fixture["gradient_clip_max_norm"]),
    )
    optimizer.zero_grad(set_to_none=True)
    named = {name.removeprefix("module."): parameter for name, parameter in ddp.named_parameters() if parameter.requires_grad}
    if set(named) != set(installed):
        raise Stage1S18Error("S18_OPTIMIZER_PARAMETER_SET_DRIFT")
    for name, parameter in named.items():
        parameter.grad = (installed[name] * clip_factor).to(device=device, dtype=parameter.dtype).clone()
    optimizer.step()
    post_parameters = _parameter_maps(ddp, gradients=False)
    optimizer_states = _optimizer_state_maps(ddp, optimizer)
    data_delta: dict[str, torch.Tensor] = {}
    weight_decay_delta: dict[str, torch.Tensor] = {}
    for name in sorted(pre_parameters):
        decay = -float(learning_rates[name]) * float(weight_decays[name]) * pre_parameters[name]
        total = post_parameters[name] - pre_parameters[name]
        weight_decay_delta[name] = decay
        data_delta[name] = total - decay
    arrays: dict[str, torch.Tensor] = {}
    if route == "B":
        for microbatch_id, sample in zip(local_ids, local_samples, strict=True):
            for name, value in sample.items():
                arrays[f"micro/{case}/{microbatch_id:02d}/{name}"] = value
    raw_core = scores["raw"]
    raw_core_clipped = {name: value * clip_factor for name, value in raw_core.items()}
    raw_score = {name: value * float(learning_rates[name]) for name, value in raw_core.items()}
    raw_score_clipped = {name: value * clip_factor for name, value in raw_score.items()}
    accumulator_summary: dict[str, object] | None = None
    if route == "A":
        # A is deliberately only a mean/raw/update reference.  It owns no U
        # core, U score, or long-horizon importance accumulator.
        a_reference = {
            "mean_gradient": scores["mean"], "raw_core": raw_core,
            # This core-only factor is an audit diagnostic, not a public
            # estimator output.  The public clipped raw value is the score
            # below and therefore includes both eta and clip.
            "raw_core_clipped_diagnostic": raw_core_clipped, "raw_score": raw_score,
            "raw_score_clipped": raw_score_clipped,
            # A has no U accumulator, but its real optimizer step still
            # supplies the applicable movement/magnitude/update diagnostics.
            "data_update": data_delta,
            "data_movement": {name: value.abs() for name, value in data_delta.items()},
            "total_update": {name: post_parameters[name] - pre_parameters[name] for name in pre_parameters},
            "weight_decay_update": weight_decay_delta,
            "actual_update_raw_importance": {
                name: -(data_delta[name] * scores["mean"][name]) for name in data_delta
            },
            "magnitude": {name: value.abs() for name, value in post_parameters.items()},
        }
        for field, values in a_reference.items():
            for name, value in values.items():
                arrays[f"a-reference/{case}/{field}/{name}"] = value
    else:
        if accumulator is None:
            raise Stage1S18Error("S18_ACCUMULATOR_REQUIRED_FOR_U_ROUTE")
        for statistic in sorted(global_stats):
            for name, value in _mapping(global_stats[statistic], field=f"global.{statistic}").items():
                arrays[f"stats/{case}/{statistic}/{name}"] = value  # type: ignore[assignment]
        u_core_clipped = {name: value * clip_factor for name, value in scores["u"].items()}
        u_score_clipped = {name: value * clip_factor for name, value in scores["score"].items()}
        score_views = {
            "mean_gradient": scores["mean"], "raw_core": raw_core,
            "raw_core_clipped_diagnostic": raw_core_clipped, "u_core": scores["u"],
            "u_core_clipped": u_core_clipped, "raw_score": raw_score,
            "raw_score_clipped": raw_score_clipped, "u_score": scores["score"],
            "u_score_clipped": u_score_clipped,
        }
        for field, values in score_views.items():
            for name, value in values.items():
                arrays[f"scores/{case}/{field}/{name}"] = value
        contribution = {
            "signed": u_score_clipped,
            "raw": raw_score,
            "raw_clipped": raw_score_clipped,
            "data_update": data_delta,
            "total_update": {name: post_parameters[name] - pre_parameters[name] for name in pre_parameters},
            "weight_decay_update": weight_decay_delta,
            "actual_update_raw_importance": {
                name: -(data_delta[name] * scores["mean"][name]) for name in data_delta
            },
        }
        accumulator.add_step(
            _tensor_map(contribution["signed"]), raw=_tensor_map(contribution["raw"]),
            raw_clipped=_tensor_map(contribution["raw_clipped"]), data_update=_tensor_map(contribution["data_update"]),
            total_update=_tensor_map(contribution["total_update"]), weight_decay_update=_tensor_map(contribution["weight_decay_update"]),
            actual_update_raw_importance=_tensor_map(contribution["actual_update_raw_importance"]),
            current_parameters=_tensor_map(post_parameters),
        )
        for field, values in contribution.items():
            for name, value in values.items():
                arrays[f"accumulator/{case}/contribution/{field}/{name}"] = value
        cumulative = _accumulator_views(accumulator)
        for field, values in cumulative.items():
            for name, value in values.items():
                arrays[f"accumulator/{case}/cumulative/{field}/{name}"] = value
        accumulator_summary = {
            "successful_steps": accumulator.successful_steps,
            "skipped_steps": accumulator.skipped_steps,
            "signed_identity": True,
            "absolute_identity": True,
            "contribution_checksums": {
                field: tensor_map_digest(values) for field, values in contribution.items()
            },
            "cumulative_checksums": {
                field: tensor_map_digest(values) for field, values in cumulative.items()
            },
        }
    for name, value in pre_parameters.items():
        arrays[f"pre/{case}/{name}"] = value
    for name, value in post_parameters.items():
        arrays[f"post/{case}/{name}"] = value
    for name, value in optimizer_states.items():
        arrays[f"optimizer-state/{case}/{name}"] = value
    for name, value in optimizer_states_pre.items():
        arrays[f"optimizer-state-pre/{case}/{name}"] = value
    rank_digest = (
        {key: tensor_map_digest(_mapping(global_stats[key], field=f"global.{key}")) for key in sorted(global_stats)}
        if route != "A" else {}
    )
    gathered: list[object] = [None] * layout.world_size
    dist.all_gather_object(gathered, {"rank": rank, "local_microbatch_ids": list(local_ids), "local_gradient_checksums": [tensor_map_digest(item) for item in local_samples], "global_statistic_checksums": rank_digest, "local_loss_numerator": local_loss_numerator, "local_effective_tokens": sum(local_weights)})
    report = {
        "case": case,
        "global_loss_numerator": total_loss_numerator,
        "global_loss_valid_token_count": total_valid_tokens,
        "global_mean_loss": total_loss_numerator / total_valid_tokens,
        "global_microbatch_count": total_m,
        "global_n1": total_n1,
        "global_n2": total_n2,
        "global_gradient_norm": global_gradient_norm,
        "clip_factor": clip_factor,
        "rank_records": gathered,
        "global_statistic_checksums": rank_digest,
        "ordinary_ddp_gradient_collectives": comm_state["calls"],
        "manual_statistic_collectives": {
            "backend": "nccl",
            "operation": "SUM",
            "tensor_statistics": (["S1", "S2"] if case == "equal" else ["G1", "G2"]) if route != "A" else [],
            "tensor_all_reduce_count": tensor_collective_count,
            "scalar_statistics": (["M", "loss_numerator", "loss_valid_token_count"] if case == "equal" else ["N1", "N2", "loss_numerator", "loss_valid_token_count"]) if route != "A" else [],
            "scalar_all_reduce_count": scalar_collective_count,
            "total_all_reduce_count": tensor_collective_count + scalar_collective_count,
        },
        "post_parameter_checksum": tensor_map_digest(_parameter_maps(ddp, gradients=False)),
        "pre_parameter_checksum": tensor_map_digest(pre_parameters),
        "accumulator": accumulator_summary,
        "array_keys": sorted(arrays),
    }
    return arrays, report


def execute_worker(plan_path: str | Path) -> dict[str, object]:
    """Run exactly one torchrun route; rank zero publishes one immutable report."""

    plan_file = Path(plan_path).resolve(strict=True)
    plan = validate_worker_plan(_mapping(load_canonical_json(plan_file), field="worker_plan"), path=plan_file)
    route = str(plan["route"])
    execution_mode = str(plan["execution_mode"])
    expected_world = ROUTE_WORLD_SIZE[route]
    try:
        import torch.distributed as dist
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
        from torch.nn.parallel import DistributedDataParallel
    except ImportError as error:  # pragma: no cover - server-only distributed optional dependency
        raise Stage1S18Error("S18_TORCH_DISTRIBUTED_UNAVAILABLE") from error
    if os.environ.get("CUDA_VISIBLE_DEVICES") != ",".join(plan["visible_gpu_uuids"]):
        raise Stage1S18Error("S18_CUDA_VISIBLE_DEVICES_DRIFT")
    validate_nccl_transport_environment()
    required_environment = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    if any(name not in os.environ for name in required_environment):
        raise Stage1S18Error("S18_TORCHRUN_ENVIRONMENT_MISSING")
    rank, local_rank, world_size = (int(os.environ[name]) for name in required_environment)
    if world_size != expected_world or not 0 <= rank < world_size or local_rank != rank:
        raise Stage1S18Error("S18_TORCHRUN_IDENTITY_INVALID")
    if not torch.cuda.is_available():
        raise Stage1S18Error("S18_NCCL_REQUIRES_CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(seconds=int(NCCL_TRANSPORT_PROTOCOL["process_group_timeout_seconds"])),
    )
    try:
        fixture = validate_fixture(_mapping(plan["fixture"], field="plan.fixture"))
        if execution_mode == "inject_rank_failure" and rank == 0:
            raise Stage1S18Error("S18_INJECTED_RANK_FAILURE:rank=0:phase=before_backward")
        tokens = _load_tokens(plan_file.parent / str(plan["fixture_tokens_ref"]), fixture)
        seed_rank_local_generators(
            seed=int(_mapping(fixture["randomness"], field="fixture.randomness")["model_seed"]),
            local_rank=local_rank,
        )
        model = _load_model(plan, device)
        ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)
        comm_state = {"calls": 0}
        def hook(state: dict[str, int], bucket: object) -> object:
            state["calls"] += 1
            return allreduce_hook(None, bucket)  # type: ignore[arg-type]
        ddp.register_comm_hook(comm_state, hook)
        layout = permute_route_layout(build_route_layout(route), permutation=str(plan["permutation"]))
        output_dir = plan_file.parent / str(plan["output_dir"])
        if rank == 0:
            output_dir.mkdir(parents=False, exist_ok=False)
        dist.barrier()
        all_arrays: dict[str, torch.Tensor] = {}
        case_reports: list[dict[str, object]] = []
        optimizer_config = _mapping(fixture["optimizer"], field="fixture.optimizer")
        # Equal then weighted are two genuinely consecutive successful
        # optimizer steps.  Do not reset either parameters or AdamW moments
        # between them: B/C/D must certify the long-horizon accumulator rather
        # than two isolated one-step contributions.
        optimizer = torch.optim.AdamW(
            ddp.parameters(), lr=float(optimizer_config["learning_rate"]),
            weight_decay=float(optimizer_config["weight_decay"]),
            betas=tuple(float(item) for item in optimizer_config["betas"]),
            eps=float(optimizer_config["eps"]), foreach=False, fused=False,
        )
        accumulator: ImportanceAccumulator | None = None
        if route != "A":
            initial_parameters = _tensor_map(_parameter_maps(ddp, gradients=False))
            accumulator = ImportanceAccumulator(initial_parameters, accumulation_dtype=torch.float32)
            accumulator.set_initial_parameters(initial_parameters)
        for case in CASES:
            comm_state["calls"] = 0
            arrays, case_report = _run_case(
                ddp=ddp, optimizer=optimizer, tokens=tokens, layout=layout, case=case,
                fixture=fixture, device=device, comm_state=comm_state, route=route,
                suppress_ddp_sync=execution_mode == "formal", accumulator=accumulator,
            )
            if route != "A" and execution_mode == "formal" and int(case_report["ordinary_ddp_gradient_collectives"]) != 0:
                raise Stage1S18Error("S18_NO_SYNC_DDP_COLLECTIVE_DETECTED")
            if route != "A" and execution_mode == "ordinary_sync_negative" and int(case_report["ordinary_ddp_gradient_collectives"]) <= 0:
                raise Stage1S18Error("S18_ORDINARY_SYNC_NEGATIVE_GUARD_NOT_OBSERVED")
            all_arrays.update(arrays); case_reports.append(case_report)
        if execution_mode == "ordinary_sync_negative":
            # The route ran far enough to prove a normal DDP gradient hook was
            # observed, then is deliberately rejected before any PASS report
            # can be published.  This is the real negative control, not a
            # synthetic boolean in the formalizer.
            raise Stage1S18Error("S18_NO_SYNC_DDP_COLLECTIVE_DETECTED")
        result: dict[str, object] = {
            "schema_version": WORKER_REPORT_SCHEMA,
            "status": "PASS",
            "task_id": TASK_ID,
            "execution_commit": plan["execution_commit"],
            "run_token": plan["run_token"],
            "route": route,
            "permutation": plan["permutation"],
            "execution_mode": execution_mode,
            "world_size": world_size,
            "backend": "nccl",
            "nccl_transport_protocol": dict(NCCL_TRANSPORT_PROTOCOL),
            "visible_gpu_uuids": plan["visible_gpu_uuids"],
            "rank_to_gpu_uuid": list(plan["visible_gpu_uuids"]),
            "parameter_registry_hash": _parameter_registry_hash(ddp),
            "fixture_hash": fixture["fixture_hash"],
            "route_layout": layout.to_dict(),
            "cases": case_reports,
        }
        if rank == 0:
            arrays_path = output_dir / f"route-{route}.safetensors"
            result["arrays"] = _save_route_arrays(arrays_path, all_arrays)
            result = with_artifact_hash(result)
            write_canonical_json(output_dir / "route-report.json", result)
        dist.barrier()
        return result if rank == 0 else {"status": "RANK_COMPLETE", "rank": rank}
    finally:
        dist.destroy_process_group()


__all__ = [
    "ARRAY_MANIFEST_SCHEMA", "CASES", "EXECUTION_MODES", "FIXTURE_SCHEMA", "MICROBATCH_COUNT", "PERMUTATIONS", "PRE_ROUTE_SCALE_ORACLE_SCHEMA", "ROUTES",
    "RouteLayout", "Stage1S18Error", "TASK_ID", "T32_DISTRIBUTED_ATOL", "T32_DISTRIBUTED_L2_LIMIT",
    "T32_DISTRIBUTED_RTOL", "NCCL_TRANSPORT_PROTOCOL", "WORKER_PLAN_SCHEMA", "WORKER_REPORT_SCHEMA", "build_fixture",
    "build_route_layout", "execute_worker", "learning_rate_map", "local_sufficient_statistics", "permute_route_layout", "scores_from_global_statistics", "validate_nccl_transport_environment", "weight_decay_map",
    "self_hash_matches", "tensor_map_digest", "validate_fixture", "validate_route_layout", "validate_worker_plan",
    "with_artifact_hash",
]
