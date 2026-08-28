"""Immutable Stage 3 pilot/formal protocol contracts.

This module is deliberately limited to declarative contracts.  It does not
load a model, inspect a dataset, run a probe, or choose a result-dependent
endpoint.  The objects below make the G3-5 boundaries explicit and provide a
stable content digest for a pre-registered protocol.

The protocol has three independent pieces:

* :class:`ExperimentMatrix` describes endpoint/probe coverage;
* :class:`DataPartitionProtocol` owns the four globally disjoint sample-ID
  domains; and
* :class:`FormalFreeze` binds the candidate rules, two reference refinement
  families, and thresholds before a formal matrix can be constructed.

All public value objects are frozen dataclasses.  Sequence and mapping inputs
are copied into tuples or read-only views at construction time, so changing a
caller-owned list or dict cannot change a matrix digest after construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from types import MappingProxyType
from typing import Any

from ..contracts.jsonio import canonical_json_hash
from ..contracts.stage23 import FormalExecutionEvidence
from .stage3_production_plan import FORMAL_MODEL_SEEDS


STAGE3_PROTOCOL_SCHEMA = "stage3-protocol-v1"
PILOT_SCOPE = "pilot"
FORMAL_SCOPE = "formal"
STAGES: tuple[str, str, str] = ("early", "middle", "late")
FORMAL_MODELS: tuple[str, str] = ("14M", "31M")

DEFAULT_CANDIDATE_RULES: tuple[str, ...] = (
    "left",
    "right",
    "midpoint",
    "trapezoid",
    "simpson",
    "composite_trapezoid_4",
    "composite_trapezoid_8",
    "composite_simpson_4",
    "composite_simpson_8",
    "composite_simpson_16",
    "gauss_legendre_2",
    "gauss_legendre_4",
    "gauss_legendre_8",
)
DEFAULT_GAUSS_LEGENDRE_LEVELS: tuple[int, ...] = (8, 16, 32, 64)
DEFAULT_COMPOSITE_SIMPSON_LEVELS: tuple[int, ...] = (16, 32, 64, 128)
DEFAULT_THRESHOLDS: dict[str, float | int] = {
    "max_normalized_l1_error": 0.01,
    "max_normalized_l2_error": 0.01,
    "max_normalized_linf_error": 0.01,
    "max_completeness_l1_scaled_residual": 0.01,
    "min_active_spearman": 0.99,
    "min_topq_overlap": 0.95,
    "min_topq_jaccard": 0.95,
    "min_sign_consistency": 0.99,
    "max_layer_quality_tv": 0.01,
    "max_module_quality_tv": 0.01,
    "max_reference_normalized_l1_error": 0.001,
    "max_unique_nodes": 16,
}
REQUIRED_FORMAL_GATE_IDS: tuple[str, ...] = tuple(
    f"stage3.G3-{index}" for index in range(6)
)


def _stable_hash_value(digest: str) -> int:
    """Return a process-independent integer suitable for ``__hash__``."""

    # Python's hash(str) is intentionally salted per process.  Deriving the
    # integer from the SHA-256 digest keeps ``hash(value)`` stable as well as
    # providing the stronger, serializable ``artifact_hash`` property.
    return int(digest[:16], 16)


def _non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} 必须是非空字符串")
    return value


def _unique_strings(
    value: Sequence[str] | set[str] | frozenset[str],
    *,
    field_name: str,
    sort_sets: bool = True,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} 必须是字符串序列，不能直接是字符串")
    try:
        raw = tuple(value)
    except TypeError as error:
        raise TypeError(f"{field_name} 必须是字符串序列") from error
    if sort_sets and isinstance(value, (set, frozenset)):
        raw = tuple(sorted(raw))
    result = tuple(_non_empty_string(item, field_name=field_name) for item in raw)
    if len(set(result)) != len(result):
        raise ValueError(f"{field_name} 不能重复")
    return result


def _positive_integer_levels(
    value: Sequence[int], *, field_name: str, minimum_count: int = 2
) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} 必须是正整数序列")
    try:
        levels = tuple(value)
    except TypeError as error:
        raise TypeError(f"{field_name} 必须是正整数序列") from error
    if len(levels) < minimum_count:
        raise ValueError(f"{field_name} 至少需要 {minimum_count} 个 refinement level")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in levels):
        raise ValueError(f"{field_name} 必须全部为正整数")
    if tuple(sorted(set(levels))) != levels:
        raise ValueError(f"{field_name} 必须严格递增且无重复")
    return levels


def _contains_forbidden_formal_label(value: object) -> bool:
    return isinstance(value, str) and any(
        marker in value.casefold() for marker in ("fixture", "synthetic")
    )


class _StableValue:
    """Mixin for deterministic serialized and Python hashes."""

    def to_dict(self) -> dict[str, Any]:  # pragma: no cover - abstract protocol
        raise NotImplementedError

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    @property
    def content_hash(self) -> str:
        return self.artifact_hash

    @property
    def digest(self) -> str:
        return self.artifact_hash

    def __hash__(self) -> int:
        return _stable_hash_value(self.artifact_hash)


@dataclass(frozen=True, slots=True)
class CandidateRules(_StableValue):
    """Ordered, unique candidate rule names.

    ``frozen`` is a semantic pre-registration flag; it is independent of the
    Python dataclass immutability supplied by ``frozen=True``.
    """

    names: tuple[str, ...] | Sequence[str]
    frozen: bool = False
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        names = _unique_strings(self.names, field_name="candidate_rules")
        if not names:
            raise ValueError("candidate_rules 不能为空")
        if type(self.frozen) is not bool:
            raise TypeError("candidate_rules.frozen 必须是 bool")
        object.__setattr__(self, "names", names)

    @property
    def rules(self) -> tuple[str, ...]:
        return self.names

    def freeze(self) -> "CandidateRules":
        return CandidateRules(self.names, frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "rules": list(self.names),
            "frozen": self.frozen,
        }


@dataclass(frozen=True, slots=True)
class ReferenceRefinementLadder(_StableValue):
    """The two required, independently refined reference families."""

    gauss_legendre_levels: tuple[int, ...] | Sequence[int] = DEFAULT_GAUSS_LEGENDRE_LEVELS
    composite_simpson_levels: tuple[int, ...] | Sequence[int] = DEFAULT_COMPOSITE_SIMPSON_LEVELS
    frozen: bool = False
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        gauss = _positive_integer_levels(
            self.gauss_legendre_levels, field_name="gauss_legendre_levels"
        )
        simpson = _positive_integer_levels(
            self.composite_simpson_levels, field_name="composite_simpson_levels"
        )
        if type(self.frozen) is not bool:
            raise TypeError("reference ladder.frozen 必须是 bool")
        object.__setattr__(self, "gauss_legendre_levels", gauss)
        object.__setattr__(self, "composite_simpson_levels", simpson)

    @property
    def family_names(self) -> tuple[str, str]:
        return ("gauss_legendre", "composite_simpson")

    @property
    def families(self) -> tuple[tuple[str, tuple[int, ...]], tuple[str, tuple[int, ...]]]:
        return (
            ("gauss_legendre", self.gauss_legendre_levels),
            ("composite_simpson", self.composite_simpson_levels),
        )

    @property
    def gauss_levels(self) -> tuple[int, ...]:
        return self.gauss_legendre_levels

    @property
    def simpson_levels(self) -> tuple[int, ...]:
        return self.composite_simpson_levels

    def freeze(self) -> "ReferenceRefinementLadder":
        return ReferenceRefinementLadder(
            self.gauss_legendre_levels,
            self.composite_simpson_levels,
            frozen=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "families": {
                "gauss_legendre": list(self.gauss_legendre_levels),
                "composite_simpson": list(self.composite_simpson_levels),
            },
            "frozen": self.frozen,
        }


@dataclass(frozen=True, slots=True)
class FrozenThresholds(_StableValue):
    """Named finite thresholds, copied and optionally marked frozen."""

    values: Mapping[str, float | int] | Sequence[tuple[str, float | int]] = field(
        default_factory=lambda: dict(DEFAULT_THRESHOLDS)
    )
    frozen: bool = False
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        if isinstance(self.values, Mapping):
            items = tuple(self.values.items())
        else:
            try:
                items = tuple(self.values)
            except TypeError as error:
                raise TypeError("thresholds 必须是 mapping 或键值序列") from error
        normalized: list[tuple[str, float | int]] = []
        for key, value in items:
            key = _non_empty_string(key, field_name="threshold name")
            if isinstance(value, bool):
                raise ValueError(f"threshold {key} 不能是 bool")
            if key == "max_unique_nodes":
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("threshold max_unique_nodes 必须是正整数")
                numeric: float | int = value
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"threshold {key} 必须是数值") from error
                if not math.isfinite(numeric):
                    raise ValueError(f"threshold {key} 必须是有限数")
            normalized.append((key, numeric))
        if not normalized:
            raise ValueError("thresholds 不能为空")
        if len({key for key, _ in normalized}) != len(normalized):
            raise ValueError("threshold 名称不能重复")
        normalized.sort(key=lambda item: item[0])
        if type(self.frozen) is not bool:
            raise TypeError("thresholds.frozen 必须是 bool")
        object.__setattr__(self, "values", tuple(normalized))

    @property
    def mapping(self) -> Mapping[str, float | int]:
        return MappingProxyType(dict(self.values))

    def __getitem__(self, key: str) -> float | int:
        return self.mapping[key]

    def freeze(self) -> "FrozenThresholds":
        return FrozenThresholds(self.mapping, frozen=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "values": {key: value for key, value in self.values},
            "frozen": self.frozen,
        }


def _validate_formal_thresholds(thresholds: FrozenThresholds) -> None:
    observed = dict(thresholds.mapping)
    if set(observed) != set(DEFAULT_THRESHOLDS):
        raise ValueError("formal thresholds 必须完整覆盖冻结指标集合")
    for key, principle in DEFAULT_THRESHOLDS.items():
        value = observed[key]
        if key.startswith("max_"):
            if float(value) < 0 or float(value) > float(principle):
                raise ValueError(f"formal threshold {key} 宽于 Stage 3 原则上限")
        elif key.startswith("min_"):
            if not 0 <= float(value) <= 1 or float(value) < float(principle):
                raise ValueError(f"formal threshold {key} 低于 Stage 3 原则下限")
    if float(observed["max_reference_normalized_l1_error"]) > (
        float(observed["max_normalized_l1_error"]) / 10.0
    ):
        raise ValueError("reference error 必须不高于候选 L1 容许误差的十分之一")


@dataclass(frozen=True, slots=True)
class FormalFreeze(_StableValue):
    """A content-bound declaration made before formal execution."""

    candidate_rules: CandidateRules | Sequence[str]
    reference_ladder: ReferenceRefinementLadder = field(
        default_factory=ReferenceRefinementLadder
    )
    thresholds: FrozenThresholds | Mapping[str, float | int] = field(
        default_factory=FrozenThresholds
    )
    freeze_id: str | None = None
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        candidate = (
            self.candidate_rules
            if isinstance(self.candidate_rules, CandidateRules)
            else CandidateRules(self.candidate_rules)
        ).freeze()
        ladder = (
            self.reference_ladder
            if isinstance(self.reference_ladder, ReferenceRefinementLadder)
            else ReferenceRefinementLadder(*self.reference_ladder)  # type: ignore[arg-type]
        ).freeze()
        threshold = (
            self.thresholds
            if isinstance(self.thresholds, FrozenThresholds)
            else FrozenThresholds(self.thresholds)
        ).freeze()
        _validate_formal_thresholds(threshold)
        freeze_id = self.freeze_id
        if freeze_id is not None:
            _non_empty_string(freeze_id, field_name="freeze_id")
        else:
            freeze_id = canonical_json_hash(
                {
                    "candidate_rules": candidate.to_dict(),
                    "reference_ladder": ladder.to_dict(),
                    "thresholds": threshold.to_dict(),
                }
            )
        object.__setattr__(self, "candidate_rules", candidate)
        object.__setattr__(self, "reference_ladder", ladder)
        object.__setattr__(self, "thresholds", threshold)
        object.__setattr__(self, "freeze_id", freeze_id)

    @property
    def candidate_rules_hash(self) -> str:
        return self.candidate_rules.artifact_hash

    @property
    def reference_ladder_hash(self) -> str:
        return self.reference_ladder.artifact_hash

    @property
    def thresholds_hash(self) -> str:
        return self.thresholds.artifact_hash

    @property
    def is_frozen(self) -> bool:
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "freeze_id": self.freeze_id,
            "candidate_rules": self.candidate_rules.to_dict(),
            "reference_ladder": self.reference_ladder.to_dict(),
            "thresholds": self.thresholds.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EndpointSpec(_StableValue):
    """One immutable training endpoint and its fixed probe IDs."""

    model_size: str
    seed: int
    stage: str
    endpoint_id: str
    probe_ids: tuple[str, ...] | Sequence[str]
    update_sample_ids: tuple[str, ...] | Sequence[str] = ()
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        model = _non_empty_string(self.model_size, field_name="model_size")
        stage = _non_empty_string(self.stage, field_name="stage")
        endpoint = _non_empty_string(self.endpoint_id, field_name="endpoint_id")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed 必须是整数")
        probes = _unique_strings(self.probe_ids, field_name="probe_ids")
        updates = _unique_strings(
            self.update_sample_ids, field_name="update_sample_ids"
        )
        if not probes:
            raise ValueError("probe_ids 不能为空")
        object.__setattr__(self, "model_size", model)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "endpoint_id", endpoint)
        object.__setattr__(self, "probe_ids", probes)
        object.__setattr__(self, "update_sample_ids", updates)

    @property
    def probes(self) -> tuple[str, ...]:
        return self.probe_ids

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{self.endpoint_id}::probe::{probe_id}" for probe_id in self.probe_ids
        )

    @property
    def endpoint_probe_unit_ids(self) -> tuple[str, ...]:
        return self.unit_ids

    def unit_id(self, probe_id: str) -> str:
        if probe_id not in self.probe_ids:
            raise KeyError(probe_id)
        return f"{self.endpoint_id}::probe::{probe_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "model_size": self.model_size,
            "seed": self.seed,
            "stage": self.stage,
            "endpoint_id": self.endpoint_id,
            "probe_ids": list(self.probe_ids),
            "update_sample_ids": list(self.update_sample_ids),
        }


# Names used in different Stage 3 documents are intentionally kept as aliases.
MatrixEndpoint = EndpointSpec
Endpoint = EndpointSpec


@dataclass(frozen=True, slots=True)
class DataPartitionProtocol(_StableValue):
    """Four globally disjoint sample-ID domains.

    Empty domains are allowed so the type can be used while a partition is
    being assembled; every non-empty domain is still checked for internal and
    cross-domain duplicates.  Builders for complete Stage 3 protocols provide
    all four domains.
    """

    update_sample_ids: tuple[str, ...] | Sequence[str] = ()
    pilot_probe_sample_ids: tuple[str, ...] | Sequence[str] = ()
    formal_probe_sample_ids: tuple[str, ...] | Sequence[str] = ()
    replay_sample_ids: tuple[str, ...] | Sequence[str] = ()
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        fields = (
            ("update_sample_ids", self.update_sample_ids),
            ("pilot_probe_sample_ids", self.pilot_probe_sample_ids),
            ("formal_probe_sample_ids", self.formal_probe_sample_ids),
            ("replay_sample_ids", self.replay_sample_ids),
        )
        normalized = {
            name: _unique_strings(value, field_name=name)
            for name, value in fields
        }
        seen: dict[str, str] = {}
        for name, ids in normalized.items():
            for sample_id in ids:
                previous = seen.get(sample_id)
                if previous is not None:
                    raise ValueError(
                        f"sample ID {sample_id!r} 同时属于 {previous} 与 {name}；"
                        "update/pilot probe/formal probe/replay 必须全局互斥"
                    )
                seen[sample_id] = name
        for name, ids in normalized.items():
            object.__setattr__(self, name, ids)

    @property
    def update_ids(self) -> tuple[str, ...]:
        return self.update_sample_ids

    @property
    def pilot_probe_ids(self) -> tuple[str, ...]:
        return self.pilot_probe_sample_ids

    @property
    def formal_probe_ids(self) -> tuple[str, ...]:
        return self.formal_probe_sample_ids

    @property
    def replay_ids(self) -> tuple[str, ...]:
        return self.replay_sample_ids

    @property
    def all_sample_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                self.update_sample_ids
                + self.pilot_probe_sample_ids
                + self.formal_probe_sample_ids
                + self.replay_sample_ids
            )
        )

    @property
    def partitions(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType(
            {
                "update": self.update_sample_ids,
                "pilot_probe": self.pilot_probe_sample_ids,
                "formal_probe": self.formal_probe_sample_ids,
                "replay": self.replay_sample_ids,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "update_sample_ids": list(self.update_sample_ids),
            "pilot_probe_sample_ids": list(self.pilot_probe_sample_ids),
            "formal_probe_sample_ids": list(self.formal_probe_sample_ids),
            "replay_sample_ids": list(self.replay_sample_ids),
        }


@dataclass(frozen=True, slots=True)
class ExperimentMatrix(_StableValue):
    """Validated immutable pilot or formal endpoint/probe matrix."""

    scope: str
    endpoints: tuple[EndpointSpec, ...] | Sequence[EndpointSpec | Mapping[str, Any]]
    candidate_rules: CandidateRules | Sequence[str] = field(
        default_factory=lambda: CandidateRules(DEFAULT_CANDIDATE_RULES)
    )
    reference_ladder: ReferenceRefinementLadder = field(
        default_factory=ReferenceRefinementLadder
    )
    thresholds: FrozenThresholds | Mapping[str, float | int] | None = None
    formal_freeze: FormalFreeze | None = None
    formal_execution_evidence: FormalExecutionEvidence | None = None
    data_partitions: DataPartitionProtocol | None = None
    matrix_id: str = STAGE3_PROTOCOL_SCHEMA
    formal_eligible: bool | None = None
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        if self.scope not in {PILOT_SCOPE, FORMAL_SCOPE}:
            raise ValueError("scope 必须是 pilot 或 formal")
        if isinstance(self.endpoints, (str, bytes)):
            raise TypeError("endpoints 必须是 EndpointSpec 序列")
        try:
            raw_endpoints = tuple(self.endpoints)
        except TypeError as error:
            raise TypeError("endpoints 必须是 EndpointSpec 序列") from error
        endpoints = tuple(_coerce_endpoint(item) for item in raw_endpoints)
        endpoints = tuple(sorted(endpoints, key=lambda item: item.endpoint_id))
        if not endpoints:
            raise ValueError("endpoints 不能为空")
        endpoint_ids = tuple(item.endpoint_id for item in endpoints)
        if len(set(endpoint_ids)) != len(endpoint_ids):
            raise ValueError("endpoint_id 必须全局唯一")
        unit_ids = tuple(unit for item in endpoints for unit in item.unit_ids)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("endpoint×probe unit IDs 必须全局唯一")

        candidate = (
            self.candidate_rules
            if isinstance(self.candidate_rules, CandidateRules)
            else CandidateRules(self.candidate_rules)
        )
        ladder = self.reference_ladder
        if not isinstance(ladder, ReferenceRefinementLadder):
            raise TypeError("reference_ladder 必须是 ReferenceRefinementLadder")
        thresholds: FrozenThresholds | None
        if self.thresholds is None:
            thresholds = None
        elif isinstance(self.thresholds, FrozenThresholds):
            thresholds = self.thresholds
        else:
            thresholds = FrozenThresholds(self.thresholds)
        if self.data_partitions is not None and not isinstance(
            self.data_partitions, DataPartitionProtocol
        ):
            raise TypeError("data_partitions 必须是 DataPartitionProtocol")
        matrix_id = _non_empty_string(self.matrix_id, field_name="matrix_id")
        formal_eligible = self.formal_eligible
        if formal_eligible is not None and type(formal_eligible) is not bool:
            raise TypeError("formal_eligible 必须是 bool 或 None")

        if self.scope == PILOT_SCOPE:
            if formal_eligible is True:
                raise ValueError("pilot matrix 不能标记 formal_eligible=true")
            if self.formal_execution_evidence is not None:
                raise ValueError("pilot matrix 不能携带 formal execution evidence")
            formal_eligible = False
            _validate_pilot_coverage(endpoints)
        else:
            if self.formal_freeze is None:
                raise ValueError(
                    "formal matrix 必须绑定 formal 前已完成的 FormalFreeze"
                )
            if not isinstance(self.formal_freeze, FormalFreeze):
                raise TypeError("formal_freeze 必须是 FormalFreeze")
            freeze = self.formal_freeze
            evidence = self.formal_execution_evidence
            if not isinstance(evidence, FormalExecutionEvidence):
                raise ValueError("formal matrix 必须绑定 FormalExecutionEvidence")
            evidence.require_for_stage(3)
            observed_gate_ids = {gate.gate_id for gate in evidence.prerequisite_gates}
            missing_gates = set(REQUIRED_FORMAL_GATE_IDS) - observed_gate_ids
            if missing_gates:
                raise ValueError(
                    f"formal matrix 缺少前置 Gate: {sorted(missing_gates)}"
                )
            # A formal matrix may not silently replace a frozen component with
            # a mutable/pilot copy.  Compare semantic values and then retain
            # the exact frozen objects from the binding.
            if candidate.names != freeze.candidate_rules.names:
                raise ValueError("formal candidate_rules 与 freeze 不一致")
            if ladder != freeze.reference_ladder:
                raise ValueError("formal reference_ladder 与 freeze 不一致")
            if thresholds is not None and thresholds.values != freeze.thresholds.values:
                raise ValueError("formal thresholds 与 freeze 不一致")
            candidate = freeze.candidate_rules
            ladder = freeze.reference_ladder
            thresholds = freeze.thresholds
            if formal_eligible is False:
                raise ValueError("formal matrix 必须标记 formal_eligible=true")
            formal_eligible = True
            _validate_formal_coverage(endpoints)
            labels = [matrix_id]
            labels.extend(
                value
                for endpoint in endpoints
                for value in (
                    endpoint.model_size,
                    endpoint.stage,
                    endpoint.endpoint_id,
                    *endpoint.probe_ids,
                )
            )
            labels.extend(candidate.names)
            if any(_contains_forbidden_formal_label(value) for value in labels):
                raise ValueError("formal matrix 禁止 fixture/synthetic 标识")

        object.__setattr__(self, "endpoints", endpoints)
        object.__setattr__(self, "candidate_rules", candidate)
        object.__setattr__(self, "reference_ladder", ladder)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "matrix_id", matrix_id)
        object.__setattr__(self, "formal_eligible", formal_eligible)

    @property
    def unit_ids(self) -> tuple[str, ...]:
        return tuple(unit for endpoint in self.endpoints for unit in endpoint.unit_ids)

    @property
    def required_unit_ids(self) -> tuple[str, ...]:
        return self.unit_ids

    @property
    def endpoint_count(self) -> int:
        return len(self.endpoints)

    @property
    def endpoint_probe_unit_count(self) -> int:
        return len(self.unit_ids)

    @property
    def coverage(self) -> dict[str, Any]:
        return matrix_coverage(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "matrix_id": self.matrix_id,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "endpoints": [item.to_dict() for item in self.endpoints],
            "candidate_rules": self.candidate_rules.to_dict(),
            "reference_ladder": self.reference_ladder.to_dict(),
            "thresholds": None if self.thresholds is None else self.thresholds.to_dict(),
            "formal_freeze": (
                None if self.formal_freeze is None else self.formal_freeze.to_dict()
            ),
            "formal_execution_evidence_hash": (
                None
                if self.formal_execution_evidence is None
                else self.formal_execution_evidence.artifact_hash
            ),
            "data_partitions": (
                None
                if self.data_partitions is None
                else self.data_partitions.to_dict()
            ),
        }


# These aliases keep the public vocabulary close to the plan while retaining
# one validator for both scopes.
PilotMatrix = ExperimentMatrix
FormalMatrix = ExperimentMatrix


def _coerce_endpoint(value: EndpointSpec | Mapping[str, Any]) -> EndpointSpec:
    if isinstance(value, EndpointSpec):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("endpoint 必须是 EndpointSpec 或 mapping")
    required = {"model_size", "seed", "stage", "endpoint_id"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"endpoint 缺少字段: {sorted(missing)}")
    probes = value.get("probe_ids", value.get("probes"))
    if probes is None:
        raise ValueError("endpoint 缺少 probe_ids")
    return EndpointSpec(
        model_size=value["model_size"],
        seed=value["seed"],
        stage=value["stage"],
        endpoint_id=value["endpoint_id"],
        probe_ids=probes,
        update_sample_ids=value.get("update_sample_ids", ()),
    )


def _validate_pilot_coverage(endpoints: Sequence[EndpointSpec]) -> None:
    if {item.model_size for item in endpoints} != {"14M"}:
        raise ValueError("pilot 只能使用 14M")
    if len({item.seed for item in endpoints}) != 1:
        raise ValueError("pilot 必须固定一个训练 seed")
    if {item.stage for item in endpoints} != set(STAGES):
        raise ValueError("pilot 必须覆盖 early/middle/late 三个阶段")
    for stage in STAGES:
        selected = [item for item in endpoints if item.stage == stage]
        if len(selected) < 2:
            raise ValueError("pilot 每个阶段至少需要两个 endpoint")
        if any(len(item.probe_ids) < 2 for item in selected):
            raise ValueError("pilot 每个 endpoint 至少需要两个 probe")
    if sum(len(item.probe_ids) for item in endpoints) < 12:
        raise ValueError("pilot endpoint×probe coverage 必须至少为 12")


def _validate_formal_coverage(endpoints: Sequence[EndpointSpec]) -> None:
    if any(
        _contains_forbidden_formal_label(value)
        for endpoint in endpoints
        for value in (endpoint.model_size, endpoint.stage, endpoint.endpoint_id, *endpoint.probe_ids)
    ):
        raise ValueError("formal matrix 禁止 fixture/synthetic 标识")
    allowed = {"14M", "31M"}
    if {item.model_size for item in endpoints} != allowed:
        raise ValueError("formal 必须同时覆盖 14M 与 31M")
    for model, expected_seeds, endpoints_per_stage in (
        ("14M", FORMAL_MODEL_SEEDS["14M"], 4),
        ("31M", FORMAL_MODEL_SEEDS["31M"], 3),
    ):
        model_endpoints = [item for item in endpoints if item.model_size == model]
        if any(len(item.probe_ids) < 3 for item in model_endpoints):
            raise ValueError("formal 每个 endpoint 至少需要三个 probe")
        for seed in {item.seed for item in model_endpoints}:
            for stage in STAGES:
                selected = [
                    item
                    for item in model_endpoints
                    if item.seed == seed and item.stage == stage
                ]
                if len(selected) != endpoints_per_stage:
                    raise ValueError(
                        f"formal {model}/seed{seed}/{stage} 必须精确包含 "
                        f"{endpoints_per_stage} 个 endpoint"
                    )
        if {item.seed for item in model_endpoints} != set(expected_seeds):
            raise ValueError(f"formal {model} seed 覆盖不符合预注册矩阵")
    # 24 endpoints from 14M plus 9 from 31M, three probes each, is the
    # machine-checkable minimum/denominator for G3-5.
    if len(endpoints) != 33 or len({unit for item in endpoints for unit in item.unit_ids}) != 99:
        raise ValueError("formal endpoint×probe coverage 必须精确为 99")


def build_pilot_matrix(
    endpoints: Sequence[EndpointSpec | Mapping[str, Any]] | None = None,
    *,
    candidate_rules: CandidateRules | Sequence[str] = DEFAULT_CANDIDATE_RULES,
    reference_ladder: ReferenceRefinementLadder | None = None,
    thresholds: FrozenThresholds | Mapping[str, float | int] | None = None,
    data_partitions: DataPartitionProtocol | None = None,
    matrix_id: str = "stage3-pilot-v1",
) -> ExperimentMatrix:
    """Build the pre-registered 14M pilot matrix (12 units by default)."""

    if endpoints is None:
        endpoints = _default_pilot_endpoints()
    return ExperimentMatrix(
        scope=PILOT_SCOPE,
        endpoints=endpoints,
        candidate_rules=candidate_rules,
        reference_ladder=reference_ladder or ReferenceRefinementLadder(),
        thresholds=thresholds,
        data_partitions=data_partitions,
        matrix_id=matrix_id,
        formal_eligible=False,
    )


def build_formal_matrix(
    formal_freeze: FormalFreeze | None = None,
    endpoints: Sequence[EndpointSpec | Mapping[str, Any]] | None = None,
    *,
    formal_execution_evidence: FormalExecutionEvidence | None = None,
    data_partitions: DataPartitionProtocol | None = None,
    matrix_id: str = "stage3-formal-v1",
) -> ExperimentMatrix:
    """Build the exact 14M/31M formal matrix, requiring a prior freeze."""

    if formal_freeze is None:
        raise ValueError("formal matrix 必须先提供 FormalFreeze；禁止 formal 后补冻结")
    if not isinstance(formal_freeze, FormalFreeze):
        raise TypeError("formal_freeze 必须是 FormalFreeze")
    if not isinstance(formal_execution_evidence, FormalExecutionEvidence):
        raise ValueError("formal matrix 必须提供 FormalExecutionEvidence")
    if endpoints is None:
        endpoints = _default_formal_endpoints()
    return ExperimentMatrix(
        scope=FORMAL_SCOPE,
        endpoints=endpoints,
        candidate_rules=formal_freeze.candidate_rules,
        reference_ladder=formal_freeze.reference_ladder,
        thresholds=formal_freeze.thresholds,
        formal_freeze=formal_freeze,
        formal_execution_evidence=formal_execution_evidence,
        data_partitions=data_partitions,
        matrix_id=matrix_id,
        formal_eligible=True,
    )


def freeze_formal_protocol(
    source: ExperimentMatrix | CandidateRules | Sequence[str] | None = None,
    reference_ladder: ReferenceRefinementLadder | None = None,
    thresholds: FrozenThresholds | Mapping[str, float | int] | None = None,
) -> FormalFreeze:
    """Freeze pilot-selected declarations for use by formal construction.

    Passing a pilot matrix as ``source`` copies its candidate rules, reference
    ladder and thresholds.  Passing explicit components is also supported for
    a pure pre-registration builder; no observed metric or run result is
    accepted by this function.
    """

    if isinstance(source, ExperimentMatrix):
        if source.scope != PILOT_SCOPE:
            raise ValueError("formal freeze 的 source 必须是 pilot matrix")
        candidate: CandidateRules | Sequence[str] = source.candidate_rules
        ladder = reference_ladder or source.reference_ladder
        selected_thresholds = thresholds
        if selected_thresholds is None:
            selected_thresholds = source.thresholds or FrozenThresholds()
    else:
        candidate = source if source is not None else DEFAULT_CANDIDATE_RULES
        ladder = reference_ladder or ReferenceRefinementLadder()
        selected_thresholds = thresholds or FrozenThresholds()
    if selected_thresholds is None:  # defensive narrowing for type checkers
        selected_thresholds = FrozenThresholds()
    return FormalFreeze(candidate, ladder, selected_thresholds)


# Friendly aliases used by callers that describe this step as a freeze gate.
freeze_formal = freeze_formal_protocol
freeze_for_formal = freeze_formal_protocol


def validate_global_data_partition(
    update_sample_ids: Sequence[str] = (),
    pilot_probe_sample_ids: Sequence[str] = (),
    formal_probe_sample_ids: Sequence[str] = (),
    replay_sample_ids: Sequence[str] = (),
) -> DataPartitionProtocol:
    """Construct and validate all four mutually exclusive sample domains."""

    return DataPartitionProtocol(
        update_sample_ids=update_sample_ids,
        pilot_probe_sample_ids=pilot_probe_sample_ids,
        formal_probe_sample_ids=formal_probe_sample_ids,
        replay_sample_ids=replay_sample_ids,
    )


validate_sample_partitions = validate_global_data_partition


def matrix_coverage(matrix: ExperimentMatrix) -> dict[str, Any]:
    """Return deterministic endpoint/probe coverage counts and strata."""

    if not isinstance(matrix, ExperimentMatrix):
        raise TypeError("matrix 必须是 ExperimentMatrix")
    by_model: dict[str, int] = {}
    by_model_seed_stage: dict[str, int] = {}
    for endpoint in matrix.endpoints:
        by_model[endpoint.model_size] = by_model.get(endpoint.model_size, 0) + 1
        key = f"{endpoint.model_size}:{endpoint.seed}:{endpoint.stage}"
        by_model_seed_stage[key] = by_model_seed_stage.get(key, 0) + 1
    probe_count = sum(len(endpoint.probe_ids) for endpoint in matrix.endpoints)
    units = matrix.unit_ids
    return {
        "scope": matrix.scope,
        "endpoint_count": len(matrix.endpoints),
        "probe_count": probe_count,
        "endpoint_probe_unit_count": len(units),
        "unit_count": len(units),
        "by_model": dict(sorted(by_model.items())),
        "by_model_seed_stage": dict(sorted(by_model_seed_stage.items())),
        "unit_ids": list(units),
    }


coverage = matrix_coverage


def matrix_hash(matrix: ExperimentMatrix) -> str:
    if not isinstance(matrix, ExperimentMatrix):
        raise TypeError("matrix 必须是 ExperimentMatrix")
    return matrix.artifact_hash


@dataclass(frozen=True, slots=True)
class Stage3Protocol(_StableValue):
    """Complete G3-5 declaration tying pilot, freeze, formal and partitions."""

    pilot_matrix: ExperimentMatrix
    formal_freeze: FormalFreeze
    formal_matrix: ExperimentMatrix
    data_partitions: DataPartitionProtocol
    __hash__ = _StableValue.__hash__

    def __post_init__(self) -> None:
        if self.pilot_matrix.scope != PILOT_SCOPE:
            raise ValueError("Stage3Protocol.pilot_matrix 必须是 pilot")
        if self.formal_matrix.scope != FORMAL_SCOPE:
            raise ValueError("Stage3Protocol.formal_matrix 必须是 formal")
        if self.formal_matrix.formal_freeze != self.formal_freeze:
            raise ValueError("formal matrix 未绑定 Stage3Protocol 的 freeze")
        if not isinstance(self.data_partitions, DataPartitionProtocol):
            raise TypeError("data_partitions 必须是 DataPartitionProtocol")

    @property
    def pilot(self) -> ExperimentMatrix:
        return self.pilot_matrix

    @property
    def formal(self) -> ExperimentMatrix:
        return self.formal_matrix

    @property
    def coverage(self) -> dict[str, Any]:
        return {
            "pilot": matrix_coverage(self.pilot_matrix),
            "formal": matrix_coverage(self.formal_matrix),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STAGE3_PROTOCOL_SCHEMA,
            "pilot_matrix": self.pilot_matrix.to_dict(),
            "formal_freeze": self.formal_freeze.to_dict(),
            "formal_matrix": self.formal_matrix.to_dict(),
            "data_partitions": self.data_partitions.to_dict(),
        }


def build_stage3_protocol(
    *,
    formal_execution_evidence: FormalExecutionEvidence | None = None,
    data_partitions: DataPartitionProtocol | None = None,
    pilot_matrix: ExperimentMatrix | None = None,
    formal_freeze: FormalFreeze | None = None,
    formal_matrix: ExperimentMatrix | None = None,
) -> Stage3Protocol:
    """Build a complete deterministic default G3-5 protocol."""

    partitions = data_partitions or _default_data_partitions()
    pilot = pilot_matrix or build_pilot_matrix(data_partitions=partitions)
    freeze = formal_freeze or freeze_formal_protocol(pilot)
    formal = formal_matrix or build_formal_matrix(
        freeze,
        formal_execution_evidence=formal_execution_evidence,
        data_partitions=partitions,
    )
    return Stage3Protocol(pilot, freeze, formal, partitions)


def _default_pilot_endpoints() -> tuple[EndpointSpec, ...]:
    result: list[EndpointSpec] = []
    for stage in STAGES:
        for index in range(1, 3):
            endpoint_id = f"pilot-14M-seed4301-{stage}-endpoint{index}"
            result.append(
                EndpointSpec(
                    model_size="14M",
                    seed=4301,
                    stage=stage,
                    endpoint_id=endpoint_id,
                    probe_ids=(f"{endpoint_id}-probe1", f"{endpoint_id}-probe2"),
                )
            )
    return tuple(result)


def _default_formal_endpoints() -> tuple[EndpointSpec, ...]:
    result: list[EndpointSpec] = []
    for model, seeds, endpoints_per_stage in (
        ("14M", tuple(sorted(FORMAL_MODEL_SEEDS["14M"])), 4),
        ("31M", tuple(sorted(FORMAL_MODEL_SEEDS["31M"])), 3),
    ):
        for seed in seeds:
            for stage in STAGES:
                for index in range(1, endpoints_per_stage + 1):
                    endpoint_id = f"formal-{model}-seed{seed}-{stage}-endpoint{index}"
                    result.append(
                        EndpointSpec(
                            model_size=model,
                            seed=seed,
                            stage=stage,
                            endpoint_id=endpoint_id,
                            probe_ids=tuple(
                                f"{endpoint_id}-probe{probe_index}"
                                for probe_index in range(1, 4)
                            ),
                        )
                    )
    return tuple(result)


def _default_data_partitions() -> DataPartitionProtocol:
    return DataPartitionProtocol(
        update_sample_ids=tuple(f"update-sample-{index:03d}" for index in range(1, 34)),
        pilot_probe_sample_ids=tuple(
            f"pilot-probe-sample-{index:03d}" for index in range(1, 13)
        ),
        formal_probe_sample_ids=tuple(
            f"formal-probe-sample-{index:03d}" for index in range(1, 100)
        ),
        replay_sample_ids=tuple(f"replay-sample-{index:03d}" for index in range(1, 34)),
    )


__all__ = [
    "CandidateRules",
    "DataPartitionProtocol",
    "DEFAULT_CANDIDATE_RULES",
    "DEFAULT_THRESHOLDS",
    "Endpoint",
    "EndpointSpec",
    "ExperimentMatrix",
    "FORMAL_MODELS",
    "FORMAL_SCOPE",
    "FormalFreeze",
    "FormalMatrix",
    "FrozenThresholds",
    "MatrixEndpoint",
    "PILOT_SCOPE",
    "PilotMatrix",
    "ReferenceRefinementLadder",
    "STAGE3_PROTOCOL_SCHEMA",
    "STAGES",
    "Stage3Protocol",
    "build_formal_matrix",
    "build_pilot_matrix",
    "build_stage3_protocol",
    "coverage",
    "freeze_for_formal",
    "freeze_formal",
    "freeze_formal_protocol",
    "matrix_coverage",
    "matrix_hash",
    "validate_global_data_partition",
    "validate_sample_partitions",
]
