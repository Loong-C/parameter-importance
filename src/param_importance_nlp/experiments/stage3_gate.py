"""Independent Stage 3 Gate evaluator.

The task runner is intentionally not an authority for the formal decision.  It
may produce a candidate table, but a formal recommendation is only eligible
after this module has consumed all real ``stage3.G3-0`` through
``stage3.G3-5`` evidence and the complete frozen metric table.  This module is
pure validation/aggregation: it never starts an experiment, contacts a
server, or writes a scientific result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from types import MappingProxyType
from typing import Any

from ..contracts.errors import FormalRunRejected
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import canonical_json_hash
from ..contracts.provenance import ProvenanceRecord, ProvenanceStatus
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.stage3_scope import validate_stage3_scope_authority
from ..contracts.status import GateRecord, GateStatus
from .stage3_protocol import (
    DEFAULT_CANDIDATE_RULES,
    DEFAULT_THRESHOLDS,
    REQUIRED_FORMAL_GATE_IDS,
)


STAGE3_GATE_EVALUATION_SCHEMA = "stage3-gate-evaluation-v1"
REQUIRED_STAGE3_GATE_IDS: tuple[str, ...] = REQUIRED_FORMAL_GATE_IDS
REQUIRED_STAGE3_STRATA: tuple[str, ...] = (
    "model",
    "stage",
    "update",
    "probe",
)
REQUIRED_STAGE3_TOP_Q: tuple[float, ...] = (0.001, 0.01, 0.05)

# S3.2 defines principle bounds only for the keys in DEFAULT_THRESHOLDS.  The
# absolute and loss-drop-relative completeness tolerances must still be frozen
# before formal execution, but their numerical scale is calibrated by S3.6;
# assigning them an invented 1% ceiling here would change the preregistration.
_CANONICAL_THRESHOLD_BOUNDS: dict[str, float | int] = dict(DEFAULT_THRESHOLDS)
_FORMAL_REQUIRED_THRESHOLD_KEYS: frozenset[str] = frozenset(
    set(DEFAULT_THRESHOLDS)
    | {
        "max_completeness_absolute_residual",
        "max_completeness_relative_residual",
        "completeness_stability_epsilon",
        "active_set_threshold",
    }
)
_LEGACY_DIAGNOSTIC_THRESHOLD_KEYS: frozenset[str] = frozenset(
    {"min_spearman", "min_cosine_similarity", "min_topk_overlap"}
)
_MAX_THRESHOLD_KEYS = {
    name
    for name in _CANONICAL_THRESHOLD_BOUNDS
    if name.startswith("max_")
}
_MIN_THRESHOLD_KEYS = {
    name
    for name in _CANONICAL_THRESHOLD_BOUNDS
    if name.startswith("min_")
}


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _forbidden_formal_label(value: object) -> bool:
    if isinstance(value, str):
        return any(marker in value.casefold() for marker in ("fixture", "synthetic"))
    if isinstance(value, Mapping):
        return any(
            _forbidden_formal_label(key) or _forbidden_formal_label(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_forbidden_formal_label(item) for item in value)
    return False


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _metric_value(value: object) -> float | None:
    """Extract a defined metric from a raw scalar or MetricResult wire value."""

    if isinstance(value, Mapping):
        if "defined" in value and value["defined"] is not True:
            return None
        if "value" not in value:
            return None
        value = value["value"]
    return _finite(value)


def _metric(row: Mapping[str, object], *names: str) -> float | None:
    for name in names:
        if name in row:
            return _metric_value(row[name])
    return None


def _nested_metric(row: Mapping[str, object], group: str, *names: str) -> float | None:
    nested = row.get(group)
    if not isinstance(nested, Mapping):
        return None
    for name in names:
        if name in nested:
            value = _metric_value(nested[name])
            if value is not None:
                return value
    return None


def _q_key(value: float) -> tuple[str, ...]:
    return (str(value), f"{value:g}", f"{value:.3f}")


def _topq_metric(
    row: Mapping[str, object], q: float, metric_name: str
) -> float | None:
    for field_name in ("top_q", "topq"):
        values = row.get(field_name)
        if not isinstance(values, Mapping):
            continue
        entry: object | None = None
        for key in _q_key(q):
            if key in values:
                entry = values[key]
                break
            try:
                if float(key) == q:
                    entry = values[key]
                    break
            except (TypeError, ValueError):
                continue
        if isinstance(entry, Mapping):
            value = entry.get(metric_name)
            parsed = _metric_value(value)
            if parsed is not None:
                return parsed
        else:
            parsed = _metric_value(entry)
            if metric_name == "overlap" and parsed is not None:
                return parsed
    for direct_name in (f"top_q_{metric_name}", f"topq_{metric_name}"):
        direct = row.get(direct_name)
        if isinstance(direct, Mapping):
            for key in _q_key(q):
                if key in direct:
                    return _metric_value(direct[key])
    return None


def _as_row(item: object) -> dict[str, object]:
    if isinstance(item, Mapping):
        return dict(thaw_json_value(dict(item)))
    # QuadratureObservation is deliberately imported lazily, keeping this
    # evaluator independent from the recommendation implementation at import
    # time and avoiding a cycle.
    fields = (
        "unit_id", "rule_name", "unique_nodes", "normalized_l1_error",
        "completeness_absolute_residual", "spearman", "topk_overlap",
        "wall_seconds", "normalized_l2_error", "normalized_linf_error",
        "completeness_relative_residual", "completeness_l1_scaled_residual",
        "active_spearman", "cosine_similarity", "sign_consistency",
        "topq_overlap", "topq_jaccard", "layer_quality_tv",
        "module_quality_tv", "reference_normalized_l1_error", "strata",
        "worst_case", "evidence_refs", "source_artifact_refs", "scope",
    )
    try:
        return {name: getattr(item, name) for name in fields if hasattr(item, name)}
    except Exception as error:  # pragma: no cover - defensive boundary
        raise TypeError("Stage 3 observation 必须是 mapping 或 QuadratureObservation") from error


def _thresholds(value: Mapping[str, object] | object) -> dict[str, object]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()  # type: ignore[union-attr]
    if not isinstance(value, Mapping):
        raise TypeError("Stage 3 thresholds 必须是 object")
    raw = dict(value)
    required = set(_FORMAL_REQUIRED_THRESHOLD_KEYS) | {
        "top_q_values", "required_strata", "require_worst_case"
    }
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError(f"STAGE3_FORMAL_THRESHOLDS_INCOMPLETE:{','.join(missing)}")
    allowed = required | set(_LEGACY_DIAGNOSTIC_THRESHOLD_KEYS)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"STAGE3_FORMAL_THRESHOLDS_UNKNOWN:{','.join(unknown)}")
    numeric_names = set(_FORMAL_REQUIRED_THRESHOLD_KEYS) | set(
        _LEGACY_DIAGNOSTIC_THRESHOLD_KEYS
    )
    for name in numeric_names:
        if name not in raw:
            continue
        parsed = _finite(raw[name])
        if parsed is None:
            raise ValueError(f"STAGE3_THRESHOLD_INVALID:{name}")
        if name.startswith("max_") and parsed < 0:
            raise ValueError(f"STAGE3_THRESHOLD_INVALID:{name}")
        if name.startswith("min_") and not -1 <= parsed <= 1:
            raise ValueError(f"STAGE3_THRESHOLD_INVALID:{name}")
        if name in _CANONICAL_THRESHOLD_BOUNDS:
            bound = float(_CANONICAL_THRESHOLD_BOUNDS[name])
            if name in _MAX_THRESHOLD_KEYS and parsed > bound:
                raise ValueError(f"STAGE3_THRESHOLD_WIDENED:{name}")
            if name in _MIN_THRESHOLD_KEYS and parsed < bound:
                raise ValueError(f"STAGE3_THRESHOLD_WIDENED:{name}")
        if name == "max_unique_nodes" and not parsed.is_integer():
            raise ValueError("STAGE3_THRESHOLD_INVALID:max_unique_nodes")
        if name == "completeness_stability_epsilon" and parsed <= 0:
            raise ValueError("STAGE3_THRESHOLD_INVALID:completeness_stability_epsilon")
        if name == "active_set_threshold" and parsed < 0:
            raise ValueError("STAGE3_THRESHOLD_INVALID:active_set_threshold")
        raw[name] = int(parsed) if name == "max_unique_nodes" else parsed
    if isinstance(raw["max_unique_nodes"], bool) or int(raw["max_unique_nodes"]) <= 0:
        raise ValueError("STAGE3_THRESHOLD_INVALID:max_unique_nodes")
    q_values = raw["top_q_values"]
    if not isinstance(q_values, (list, tuple)):
        raise ValueError("STAGE3_TOP_Q_VALUES_INVALID")
    normalized_q = tuple(float(q) for q in q_values)
    if normalized_q != REQUIRED_STAGE3_TOP_Q:
        raise ValueError("STAGE3_TOP_Q_VALUES_NOT_PREREGISTERED")
    raw["top_q_values"] = list(normalized_q)
    strata = raw["required_strata"]
    if not isinstance(strata, (list, tuple)) or tuple(str(item) for item in strata) != REQUIRED_STAGE3_STRATA:
        raise ValueError("STAGE3_STRATA_NOT_PREREGISTERED")
    raw["required_strata"] = list(REQUIRED_STAGE3_STRATA)
    if raw["require_worst_case"] is not True:
        raise ValueError("STAGE3_WORST_CASE_REQUIRED")
    return raw


def _provenance(value: ProvenanceRecord | Mapping[str, object] | None) -> ProvenanceRecord:
    if isinstance(value, ProvenanceRecord):
        record = value
    elif isinstance(value, Mapping):
        record = ProvenanceRecord.from_mapping(dict(value))
    else:
        raise FormalRunRejected("STAGE3_PROVENANCE_REQUIRED")
    if (
        record.scope != "formal"
        or not record.formal_eligible
        or not record.worktree_clean
        or record.status is not ProvenanceStatus.COMPLETED
    ):
        raise FormalRunRejected("STAGE3_PROVENANCE_NOT_FORMAL_COMPLETED_CLEAN")
    if any(_forbidden_formal_label(item) for item in record.artifact_refs):
        raise FormalRunRejected("STAGE3_PROVENANCE_FORBIDDEN_EVIDENCE_LABEL")
    return record


def _formal_plan(
    value: Mapping[str, object] | None,
    execution: FormalExecutionEvidence,
) -> tuple[
    str,
    tuple[str, ...],
    tuple[str, ...],
    dict[str, dict[str, str]],
    dict[str, object],
]:
    if not isinstance(value, Mapping):
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_REQUIRED")
    expected = {
        "schema_version",
        "plan_id",
        "scope",
        "candidate_rules",
        "required_unit_ids",
        "unit_strata",
        "thresholds",
        "execution_evidence_hash",
        "formal_eligible",
        "artifact_hash",
    }
    index_fields = {
        "plan_kind",
        "production_unit_index_scope",
        "production_unit_index_ref",
        "production_unit_index_hash",
    }
    if set(value) != expected and set(value) != expected | index_fields:
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_FIELDS_MISMATCH")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    artifact_hash = value["artifact_hash"]
    _sha256(artifact_hash, field="formal_plan.artifact_hash")
    if canonical_json_hash(body) != artifact_hash:
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_HASH_MISMATCH")
    if (
        value["schema_version"] != "stage3-formal-pilot-plan-v1"
        or value["scope"] != "formal"
        or value["formal_eligible"] is not True
        or value["execution_evidence_hash"] != execution.artifact_hash
    ):
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_NOT_QUALIFIED")
    raw_rules = value["candidate_rules"]
    raw_units = value["required_unit_ids"]
    if any(
        not isinstance(items, list)
        or not items
        or any(not isinstance(item, str) or not item for item in items)
        or len(set(items)) != len(items)
        for items in (raw_rules, raw_units)
    ):
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_COVERAGE_INVALID")
    if set(value) == expected | index_fields:
        plan_kind = value["plan_kind"]
        index_scope = value["production_unit_index_scope"]
        index_ref = value["production_unit_index_ref"]
        index_hash = value["production_unit_index_hash"]
        try:
            _sha256(index_hash, field="formal_plan.production_unit_index_hash")
        except ValueError as error:
            raise FormalRunRejected(
                "STAGE3_FROZEN_FORMAL_PLAN_INDEX_BINDING_INVALID"
            ) from error
        if (
            plan_kind not in {"pilot", "matrix"}
            or index_scope != ("pilot" if plan_kind == "pilot" else "formal")
            or not isinstance(index_ref, str)
            or not index_ref
        ):
            raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_INDEX_BINDING_INVALID")
        expected_unit_count = 12 if plan_kind == "pilot" else 99
        if len(raw_units) != expected_unit_count:
            raise FormalRunRejected(
                "STAGE3_FROZEN_FORMAL_PLAN_UNIT_COVERAGE_INVALID:"
                f"expected={expected_unit_count},actual={len(raw_units)}"
            )
        if tuple(raw_rules) != DEFAULT_CANDIDATE_RULES:
            raise FormalRunRejected(
                "STAGE3_FROZEN_FORMAL_PLAN_CANDIDATE_COVERAGE_INVALID"
            )
    elif len(raw_units) in {12, 99}:
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_INDEX_REQUIRED_FOR_PRODUCTION_UNITS")
    raw_strata = value["unit_strata"]
    if (
        not isinstance(raw_strata, Mapping)
        or set(raw_strata) != set(raw_units)
        or any(
            not isinstance(item, Mapping)
            or set(item) != set(REQUIRED_STAGE3_STRATA)
            or any(not isinstance(child, str) or not child for child in item.values())
            or item["stage"] not in {"early", "middle", "late"}
            for item in raw_strata.values()
        )
    ):
        raise FormalRunRejected("STAGE3_FROZEN_FORMAL_PLAN_UNIT_STRATA_INVALID")
    unit_strata = {
        str(unit_id): {str(key): str(child) for key, child in item.items()}
        for unit_id, item in raw_strata.items()  # type: ignore[union-attr]
    }
    thresholds = _thresholds(value["thresholds"])  # type: ignore[arg-type]
    return artifact_hash, tuple(raw_rules), tuple(raw_units), unit_strata, thresholds


def _gates(
    execution: FormalExecutionEvidence,
    supplied: Sequence[GateRecord] | None,
) -> tuple[GateRecord, ...]:
    if supplied is None:
        supplied = execution.prerequisite_gates
    if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
        raise FormalRunRejected("STAGE3_GATES_REQUIRED")
    records = tuple(supplied)
    if any(not isinstance(gate, GateRecord) for gate in records):
        raise FormalRunRejected("STAGE3_GATE_RECORD_REQUIRED")
    by_id = {gate.gate_id: gate for gate in records}
    missing = [gate_id for gate_id in REQUIRED_STAGE3_GATE_IDS if gate_id not in by_id]
    if missing:
        raise FormalRunRejected(f"STAGE3_REQUIRED_GATES_MISSING:{','.join(missing)}")
    if len(by_id) != len(records):
        raise FormalRunRejected("STAGE3_GATE_IDS_DUPLICATE")
    execution_by_id = {gate.gate_id: gate for gate in execution.prerequisite_gates}
    for gate_id in REQUIRED_STAGE3_GATE_IDS:
        gate = by_id[gate_id]
        bound = execution_by_id.get(gate_id)
        if bound is None or bound.artifact_hash != gate.artifact_hash:
            raise FormalRunRejected(f"STAGE3_GATE_EXECUTION_BINDING_MISMATCH:{gate_id}")
        if gate.stage != 3 or gate.effective_status(at=datetime.now(timezone.utc)) is not GateStatus.PASS:
            raise FormalRunRejected(f"STAGE3_GATE_NOT_REAL_PASS:{gate_id}")
        if not gate.evidence_refs or any(_forbidden_formal_label(ref) for ref in gate.evidence_refs):
            raise FormalRunRejected(f"STAGE3_GATE_EVIDENCE_INVALID:{gate_id}")
    return tuple(by_id[gate_id] for gate_id in REQUIRED_STAGE3_GATE_IDS)


def _validate_scope_authority(
    *,
    execution: FormalExecutionEvidence,
    decision: Mapping[str, object] | None,
    gate: GateRecord | Mapping[str, object] | None,
    decision_ref: str | None,
    gate_ref: str | None,
) -> tuple[str, str, str, str, GateRecord]:
    """Require the explicit, hash-bound G3-0 authority for formal evaluation.

    A normal ``stage3.G3-0`` PASS in ``execution.prerequisite_gates`` is not
    sufficient.  The evaluator must receive the reviewed user decision and a
    G3-0 GateRecord that the strict scope contract can bind to that decision.
    The execution copy of G3-0 is also required to be the exact same hashed
    GateRecord, preventing a detached authority file from silently replacing
    the Gate consumed by the run.
    """

    if not isinstance(decision, Mapping):
        raise FormalRunRejected("STAGE3_G30_SCOPE_DECISION_REQUIRED")
    if isinstance(gate, Mapping):
        try:
            gate_record = GateRecord.from_mapping(dict(gate))
        except (TypeError, ValueError) as error:
            raise FormalRunRejected("STAGE3_G30_SCOPE_GATE_INVALID") from error
    elif isinstance(gate, GateRecord):
        gate_record = gate
    else:
        raise FormalRunRejected("STAGE3_G30_SCOPE_GATE_REQUIRED")
    if (
        not isinstance(decision_ref, str)
        or not decision_ref
        or "?" in decision_ref
        or any(marker in decision_ref.casefold() for marker in ("fixture", "synthetic"))
    ):
        raise FormalRunRejected("STAGE3_G30_SCOPE_DECISION_REF_INVALID")
    if (
        not isinstance(gate_ref, str)
        or not gate_ref
        or "?" in gate_ref
        or any(marker in gate_ref.casefold() for marker in ("fixture", "synthetic"))
    ):
        raise FormalRunRejected("STAGE3_G30_SCOPE_GATE_REF_INVALID")
    class _ScopeGateView:
        """Compatibility view for the strict scope contract's wire ``scope``.

        ``GateRecord`` derives its scope from its formal type and therefore
        intentionally has no mutable ``scope`` attribute.  The scope contract
        reads that wire value explicitly; expose the fixed value on a read-only
        proxy while delegating every other field to the already hash-validated
        GateRecord.
        """

        scope = "formal"

        def __init__(self, record: GateRecord) -> None:
            self._record = record

        def __getattr__(self, name: str) -> object:
            return getattr(self._record, name)

    try:
        validate_stage3_scope_authority(
            decision,
            _ScopeGateView(gate_record),  # type: ignore[arg-type]
            decision_ref=decision_ref,
        )
    except (TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G30_SCOPE_AUTHORITY_INVALID:{error}") from error
    execution_gate = next(
        (item for item in execution.prerequisite_gates if item.gate_id == "stage3.G3-0"),
        None,
    )
    if execution_gate is None or execution_gate.artifact_hash != gate_record.artifact_hash:
        raise FormalRunRejected("STAGE3_G30_SCOPE_GATE_EXECUTION_BINDING_MISMATCH")
    artifact_hash = decision.get("artifact_hash")
    if not isinstance(artifact_hash, str) or canonical_json_hash(
        {key: item for key, item in decision.items() if key != "artifact_hash"}
    ) != artifact_hash:
        # The strict validator already checks this.  Keep a named failure here
        # so callers can distinguish a decision hash problem from a Gate drift.
        raise FormalRunRejected("STAGE3_G30_SCOPE_DECISION_HASH_MISMATCH")
    return decision_ref, artifact_hash, gate_ref, gate_record.artifact_hash, gate_record


_METRIC_FIELDS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("normalized_l1_error", ("normalized_l1_error", "normalized_l1"), "max"),
    ("normalized_l2_error", ("normalized_l2_error", "normalized_l2"), "max"),
    ("normalized_linf_error", ("normalized_linf_error", "normalized_linf"), "max"),
    ("completeness_absolute_residual", ("completeness_absolute_residual",), "max"),
    ("completeness_relative_residual", ("completeness_relative_residual",), "max"),
    ("completeness_l1_scaled_residual", ("completeness_l1_scaled_residual",), "max"),
    ("spearman", ("spearman",), "min"),
    ("active_spearman", ("active_spearman",), "min"),
    ("cosine_similarity", ("cosine_similarity", "cosine"), "min"),
    ("sign_consistency", ("sign_consistency",), "min"),
    ("topk_overlap", ("topk_overlap",), "min"),
    ("layer_quality_tv", ("layer_quality_tv", "layer_total_variation"), "max"),
    ("module_quality_tv", ("module_quality_tv", "module_total_variation"), "max"),
    ("reference_normalized_l1_error", ("reference_normalized_l1_error",), "max"),
)


@dataclass(frozen=True, slots=True)
class Stage3GateEvaluation:
    """Machine-readable result of the independent Stage 3 Gate evaluation."""

    evaluation_id: str
    status: str
    scope: str
    formal_eligible: bool
    execution_evidence_hash: str
    formal_plan_hash: str
    formal_plan_ref: str
    thresholds_hash: str
    required_gate_ids: tuple[str, ...]
    gate_hashes: tuple[str, ...]
    required_rule_names: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    required_strata: tuple[str, ...]
    required_top_q: tuple[float, ...]
    rule_evaluations: Mapping[str, object]
    provenance_hash: str | None
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    stage3_scope_decision_ref: str | None = None
    stage3_scope_decision_hash: str | None = None
    stage3_scope_gate_ref: str | None = None
    stage3_scope_gate_hash: str | None = None
    schema_version: str = STAGE3_GATE_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_GATE_EVALUATION_SCHEMA:
            raise ValueError("STAGE3_GATE_EVALUATION_SCHEMA_UNSUPPORTED")
        _sha256(self.execution_evidence_hash, field="execution_evidence_hash")
        _sha256(self.formal_plan_hash, field="formal_plan_hash")
        if not isinstance(self.formal_plan_ref, str) or not self.formal_plan_ref:
            raise ValueError("STAGE3_FORMAL_PLAN_REF_INVALID")
        _sha256(self.thresholds_hash, field="thresholds_hash")
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("STAGE3_GATE_EVALUATION_STATUS_INVALID")
        if self.scope not in {"formal", "local_fixture"}:
            raise ValueError("STAGE3_GATE_EVALUATION_SCOPE_INVALID")
        if type(self.formal_eligible) is not bool:
            raise TypeError("formal_eligible 必须是 bool")
        if self.formal_eligible != (self.status == "PASS" and self.scope == "formal"):
            raise FormalRunRejected("STAGE3_GATE_EVALUATION_ELIGIBILITY_MISMATCH")
        if tuple(self.required_gate_ids) != REQUIRED_STAGE3_GATE_IDS:
            raise ValueError("STAGE3_REQUIRED_GATE_IDS_INVALID")
        if len(self.gate_hashes) != len(self.required_gate_ids):
            raise ValueError("STAGE3_GATE_HASHES_INCOMPLETE")
        for item in self.gate_hashes:
            _sha256(item, field="gate_hashes")
        if (
            not self.required_rule_names
            or len(set(self.required_rule_names)) != len(self.required_rule_names)
            or any(not isinstance(item, str) or not item for item in self.required_rule_names)
        ):
            raise ValueError("STAGE3_REQUIRED_RULES_INVALID")
        if not self.required_unit_ids or len(set(self.required_unit_ids)) != len(self.required_unit_ids):
            raise ValueError("STAGE3_REQUIRED_UNITS_INVALID")
        if tuple(self.required_strata) != REQUIRED_STAGE3_STRATA:
            raise ValueError("STAGE3_REQUIRED_STRATA_INVALID")
        if tuple(self.required_top_q) != REQUIRED_STAGE3_TOP_Q:
            raise ValueError("STAGE3_REQUIRED_TOP_Q_INVALID")
        if self.provenance_hash is not None:
            _sha256(self.provenance_hash, field="provenance_hash")
        scope_refs = (
            self.stage3_scope_decision_ref,
            self.stage3_scope_gate_ref,
        )
        scope_hashes = (
            self.stage3_scope_decision_hash,
            self.stage3_scope_gate_hash,
        )
        for value, field in zip(
            scope_hashes,
            ("stage3_scope_decision_hash", "stage3_scope_gate_hash"),
        ):
            if value is not None:
                _sha256(value, field=field)
        for value, field in zip(
            scope_refs,
            ("stage3_scope_decision_ref", "stage3_scope_gate_ref"),
        ):
            if value is not None:
                if not isinstance(value, str) or not value or "?" in value:
                    raise ValueError(f"{field.upper()}_INVALID")
                if _forbidden_formal_label(value):
                    raise FormalRunRejected(f"{field.upper()}_FORBIDDEN_LABEL")
        if any(value is None for value in (*scope_refs, *scope_hashes)) and any(
            value is not None for value in (*scope_refs, *scope_hashes)
        ):
            raise FormalRunRejected("STAGE3_SCOPE_AUTHORITY_BINDING_INCOMPLETE")
        if not isinstance(self.evaluation_id, str) or not self.evaluation_id.strip():
            raise ValueError("STAGE3_GATE_EVALUATION_ID_INVALID")
        if any(not isinstance(item, str) or not item for item in self.required_unit_ids):
            raise ValueError("STAGE3_REQUIRED_UNITS_INVALID")
        if not isinstance(self.rule_evaluations, Mapping):
            raise ValueError("STAGE3_RULE_EVALUATIONS_INVALID")
        rule_fields = {
            "passing",
            "unit_ids",
            "missing_unit_ids",
            "missing_metrics",
            "worst_case_unit_ids",
            "worst_case_source",
            "strata_count",
            "violations",
        }
        for rule, raw in self.rule_evaluations.items():
            if not isinstance(rule, str) or not rule or not isinstance(raw, Mapping):
                raise ValueError("STAGE3_RULE_EVALUATIONS_INVALID")
            if set(raw) != rule_fields:
                raise ValueError("STAGE3_RULE_EVALUATION_FIELDS_INVALID")
            if type(raw.get("passing")) is not bool:
                raise ValueError("STAGE3_RULE_EVALUATION_PASSING_INVALID")
            if raw.get("worst_case_source") != "derived_from_complete_frozen_table":
                raise ValueError("STAGE3_RULE_EVALUATION_WORST_CASE_SOURCE_INVALID")
            for field_name in (
                "unit_ids",
                "missing_unit_ids",
                "missing_metrics",
                "worst_case_unit_ids",
                "violations",
            ):
                values = raw[field_name]
                if (
                    not isinstance(values, list)
                    or any(not isinstance(item, str) or not item for item in values)
                    or len(set(values)) != len(values)
                ):
                    raise ValueError(
                        f"STAGE3_RULE_EVALUATION_ARRAY_INVALID:{field_name}"
                    )
            if (
                isinstance(raw["strata_count"], bool)
                or not isinstance(raw["strata_count"], int)
                or raw["strata_count"] < 0
            ):
                raise ValueError("STAGE3_RULE_EVALUATION_STRATA_COUNT_INVALID")
        if any(not isinstance(item, str) or not item for item in self.source_artifact_refs):
            raise ValueError("STAGE3_SOURCE_REFS_INVALID")
        if len(set(self.source_artifact_refs)) != len(self.source_artifact_refs):
            raise ValueError("STAGE3_SOURCE_REFS_DUPLICATE")
        if _forbidden_formal_label(self.source_artifact_refs):
            raise FormalRunRejected("STAGE3_SOURCE_ARTIFACT_FORBIDDEN_LABEL")
        if any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("STAGE3_GATE_REASONS_INVALID")
        if self.status == "PASS":
            if self.provenance_hash is None or not self.source_artifact_refs:
                raise FormalRunRejected("STAGE3_PASS_REQUIRES_PROVENANCE_AND_SOURCES")
            if any(value is None for value in (*scope_refs, *scope_hashes)):
                raise FormalRunRejected("STAGE3_PASS_REQUIRES_SCOPE_AUTHORITY")
            if self.stage3_scope_decision_ref not in self.source_artifact_refs:
                raise FormalRunRejected("STAGE3_PASS_SCOPE_DECISION_REF_UNBOUND")
            if self.stage3_scope_gate_ref not in self.source_artifact_refs:
                raise FormalRunRejected("STAGE3_PASS_SCOPE_GATE_REF_UNBOUND")
            if self.reasons:
                raise FormalRunRejected("STAGE3_PASS_CANNOT_HAVE_BLOCKING_REASONS")
            if not self.rule_evaluations:
                raise FormalRunRejected("STAGE3_PASS_REQUIRES_RULE_EVALUATIONS")
            if not any(
                raw.get("passing") is True
                for raw in self.rule_evaluations.values()
                if isinstance(raw, Mapping)
            ):
                raise FormalRunRejected("STAGE3_PASS_REQUIRES_PASSING_RULE")

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "status": self.status,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "execution_evidence_hash": self.execution_evidence_hash,
            "formal_plan_hash": self.formal_plan_hash,
            "formal_plan_ref": self.formal_plan_ref,
            "thresholds_hash": self.thresholds_hash,
            "required_gate_ids": list(self.required_gate_ids),
            "gate_hashes": list(self.gate_hashes),
            "required_rule_names": list(self.required_rule_names),
            "required_unit_ids": list(self.required_unit_ids),
            "required_strata": list(self.required_strata),
            "required_top_q": list(self.required_top_q),
            "rule_evaluations": thaw_json_value(dict(self.rule_evaluations)),
            "provenance_hash": self.provenance_hash,
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
            "stage3_scope_decision_ref": self.stage3_scope_decision_ref,
            "stage3_scope_decision_hash": self.stage3_scope_decision_hash,
            "stage3_scope_gate_ref": self.stage3_scope_gate_ref,
            "stage3_scope_gate_hash": self.stage3_scope_gate_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def require_pass(self) -> "Stage3GateEvaluation":
        if not self.formal_eligible:
            raise FormalRunRejected("STAGE3_FORMAL_GATE_EVALUATION_BLOCKED")
        return self

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3GateEvaluation":
        required = set(Stage3GateEvaluation.__dataclass_fields__) - {"artifact_hash"}
        expected = required | {"artifact_hash"}
        if set(value) != expected:
            raise ValueError("STAGE3_GATE_EVALUATION_FIELDS_MISMATCH")
        supplied = value["artifact_hash"]
        _sha256(supplied, field="artifact_hash")
        payload = {key: item for key, item in value.items() if key != "artifact_hash"}
        if canonical_json_hash(payload) != supplied:
            raise ValueError("STAGE3_GATE_EVALUATION_HASH_MISMATCH")
        arrays = ("required_gate_ids", "gate_hashes", "required_rule_names", "required_unit_ids", "required_strata", "required_top_q", "source_artifact_refs", "reasons")
        if any(not isinstance(value[name], list) for name in arrays):
            raise TypeError("STAGE3_GATE_EVALUATION_ARRAYS_INVALID")
        result = cls(
            evaluation_id=value["evaluation_id"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            scope=value["scope"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            execution_evidence_hash=value["execution_evidence_hash"],  # type: ignore[arg-type]
            formal_plan_hash=value["formal_plan_hash"],  # type: ignore[arg-type]
            formal_plan_ref=value["formal_plan_ref"],  # type: ignore[arg-type]
            thresholds_hash=value["thresholds_hash"],  # type: ignore[arg-type]
            required_gate_ids=tuple(value["required_gate_ids"]),  # type: ignore[arg-type]
            gate_hashes=tuple(value["gate_hashes"]),  # type: ignore[arg-type]
            required_rule_names=tuple(value["required_rule_names"]),  # type: ignore[arg-type]
            required_unit_ids=tuple(value["required_unit_ids"]),  # type: ignore[arg-type]
            required_strata=tuple(value["required_strata"]),  # type: ignore[arg-type]
            required_top_q=tuple(float(item) for item in value["required_top_q"]),
            rule_evaluations=value["rule_evaluations"],  # type: ignore[arg-type]
            provenance_hash=value["provenance_hash"],  # type: ignore[arg-type]
            source_artifact_refs=tuple(value["source_artifact_refs"]),  # type: ignore[arg-type]
            reasons=tuple(value["reasons"]),  # type: ignore[arg-type]
            stage3_scope_decision_ref=value["stage3_scope_decision_ref"],  # type: ignore[arg-type]
            stage3_scope_decision_hash=value["stage3_scope_decision_hash"],  # type: ignore[arg-type]
            stage3_scope_gate_ref=value["stage3_scope_gate_ref"],  # type: ignore[arg-type]
            stage3_scope_gate_hash=value["stage3_scope_gate_hash"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        return result


class Stage3GateEvaluator:
    """Evaluate frozen Stage 3 evidence without performing any experiment."""

    def evaluate(
        self,
        *,
        evaluation_id: str,
        execution: FormalExecutionEvidence,
        observations: Sequence[Mapping[str, object] | object],
        formal_plan: Mapping[str, object] | None = None,
        formal_plan_ref: str = "",
        required_rule_names: Sequence[str] = (),
        required_unit_ids: Sequence[str] = (),
        thresholds: Mapping[str, object] | object | None = None,
        gates: Sequence[GateRecord] | None = None,
        provenance: ProvenanceRecord | Mapping[str, object] | None = None,
        source_artifact_refs: Sequence[str] = (),
        stage3_scope_decision: Mapping[str, object] | None = None,
        stage3_scope_gate: GateRecord | Mapping[str, object] | None = None,
        stage3_scope_decision_ref: str | None = None,
        stage3_scope_gate_ref: str | None = None,
    ) -> Stage3GateEvaluation:
        reasons: list[str] = []
        gate_records: tuple[GateRecord, ...] = ()
        provenance_record: ProvenanceRecord | None = None
        scope_decision_ref_value: str | None = None
        scope_decision_hash_value: str | None = None
        scope_gate_ref_value: str | None = None
        scope_gate_hash_value: str | None = None
        try:
            if not isinstance(execution, FormalExecutionEvidence):
                raise FormalRunRejected("STAGE3_EXECUTION_EVIDENCE_REQUIRED")
            if execution.run_intent != "formal":
                raise FormalRunRejected("STAGE3_FORMAL_SCOPE_REQUIRED")
            execution.require_for_stage(3)
            gate_records = _gates(execution, gates)
        except (FormalRunRejected, TypeError, ValueError) as error:
            reasons.append(str(error))
        if isinstance(execution, FormalExecutionEvidence) and execution.run_intent == "formal":
            try:
                (
                    scope_decision_ref_value,
                    scope_decision_hash_value,
                    scope_gate_ref_value,
                    scope_gate_hash_value,
                    _scope_gate,
                ) = _validate_scope_authority(
                    execution=execution,
                    decision=stage3_scope_decision,
                    gate=stage3_scope_gate,
                    decision_ref=stage3_scope_decision_ref,
                    gate_ref=stage3_scope_gate_ref,
                )
            except (FormalRunRejected, TypeError, ValueError) as error:
                reasons.append(str(error))
        elif any(
            value is not None
            for value in (
                stage3_scope_decision,
                stage3_scope_gate,
                stage3_scope_decision_ref,
                stage3_scope_gate_ref,
            )
        ):
            reasons.append("STAGE3_SCOPE_AUTHORITY_REQUIRES_FORMAL_EXECUTION")
        formal_plan_hash = "0" * 64
        rules: tuple[str, ...] = ("<missing-rule>",)
        units: tuple[str, ...] = ("<missing-unit>",)
        planned_unit_strata: dict[str, dict[str, str]] = {}
        try:
            (
                formal_plan_hash,
                rules,
                units,
                planned_unit_strata,
                declared_thresholds,
            ) = _formal_plan(formal_plan, execution)
            if tuple(required_rule_names) not in ((), rules):
                raise ValueError("STAGE3_REQUIRED_RULES_DISAGREE_WITH_FROZEN_PLAN")
            if tuple(required_unit_ids) not in ((), units):
                raise ValueError("STAGE3_REQUIRED_UNITS_DISAGREE_WITH_FROZEN_PLAN")
            if thresholds is not None and _thresholds(thresholds) != declared_thresholds:
                raise ValueError("STAGE3_THRESHOLDS_DISAGREE_WITH_FROZEN_PLAN")
        except (FormalRunRejected, TypeError, ValueError) as error:
            declared_thresholds = {}
            reasons.append(str(error))
        thresholds_hash = canonical_json_hash(declared_thresholds)
        try:
            provenance_record = _provenance(provenance)
            if provenance_record is not None:
                gate_refs = {ref for gate in gate_records for ref in gate.evidence_refs}
                if not gate_refs.issubset(set(provenance_record.artifact_refs)):
                    raise FormalRunRejected("STAGE3_PROVENANCE_DOES_NOT_BIND_GATE_EVIDENCE")
        except (FormalRunRejected, TypeError, ValueError) as error:
            reasons.append(str(error))

        try:
            if isinstance(source_artifact_refs, (str, bytes)) or any(
                not isinstance(item, str) for item in source_artifact_refs
            ):
                raise ValueError("STAGE3_SOURCE_ARTIFACT_REFS_REQUIRED")
            refs = tuple(source_artifact_refs)
            if not refs or len(set(refs)) != len(refs):
                raise ValueError("STAGE3_SOURCE_ARTIFACT_REFS_REQUIRED")
            if any(_forbidden_formal_label(ref) for ref in refs):
                raise ValueError("STAGE3_SOURCE_ARTIFACT_FORBIDDEN_LABEL")
        except ValueError as error:
            refs = tuple(
                item for item in source_artifact_refs if isinstance(item, str)
            ) if not isinstance(source_artifact_refs, (str, bytes)) else ()
            reasons.append(str(error))
        if provenance_record is not None and not set(refs).issubset(
            set(provenance_record.artifact_refs)
        ):
            reasons.append("STAGE3_PROVENANCE_DOES_NOT_BIND_SOURCE_ARTIFACTS")
        if (
            not isinstance(formal_plan_ref, str)
            or not formal_plan_ref
            or formal_plan_ref not in refs
        ):
            reasons.append("STAGE3_FROZEN_FORMAL_PLAN_REF_UNBOUND")
        g35 = next(
            (gate for gate in gate_records if gate.gate_id == "stage3.G3-5"),
            None,
        )
        if g35 is None or formal_plan_ref not in g35.evidence_refs:
            reasons.append("STAGE3_G3_5_DOES_NOT_BIND_FROZEN_FORMAL_PLAN")

        normalized_rows: list[dict[str, object]] = []
        try:
            for raw in observations:
                row = _as_row(raw)
                if _forbidden_formal_label(row):
                    raise ValueError("STAGE3_FORMAL_OBSERVATION_FORBIDDEN_LABEL")
                normalized_rows.append(row)
        except (TypeError, ValueError) as error:
            reasons.append(str(error))

        rule_rows: dict[str, list[dict[str, object]]] = {}
        for row in normalized_rows:
            rule = row.get("rule_name")
            unit = row.get("unit_id")
            if not isinstance(rule, str) or not rule or not isinstance(unit, str) or not unit:
                reasons.append("STAGE3_OBSERVATION_ID_INVALID")
                continue
            rule_rows.setdefault(rule, []).append(row)
        missing_rules = sorted(set(rules) - set(rule_rows))
        unexpected_rules = sorted(set(rule_rows) - set(rules))
        if missing_rules:
            reasons.append("STAGE3_REQUIRED_RULES_MISSING:" + ",".join(missing_rules))
        if unexpected_rules:
            reasons.append("STAGE3_UNEXPECTED_RULES:" + ",".join(unexpected_rules))
        rule_evaluations: dict[str, object] = {}
        for rule in sorted(set(rule_rows) & set(rules)):
            rows = rule_rows[rule]
            violations: list[str] = []
            missing_metrics: set[str] = set()
            seen_units: set[str] = set()
            strata_seen: set[tuple[object, ...]] = set()
            strata_by_unit: dict[str, dict[str, object]] = {}
            row_by_unit: dict[str, dict[str, object]] = {}
            for row in rows:
                unit = str(row["unit_id"])
                if unit in seen_units:
                    violations.append(f"duplicate_unit:{unit}")
                seen_units.add(unit)
                if unit not in units:
                    violations.append(f"unexpected_unit:{unit}")
                row_refs_raw = row.get("evidence_refs") or row.get(
                    "source_artifact_refs"
                )
                if (
                    not isinstance(row_refs_raw, Sequence)
                    or isinstance(row_refs_raw, (str, bytes))
                    or not row_refs_raw
                    or any(not isinstance(ref, str) or not ref for ref in row_refs_raw)
                ):
                    violations.append(f"evidence_missing:{unit}")
                elif not set(row_refs_raw).issubset(set(refs)):
                    violations.append(f"evidence_unbound:{unit}")
                unique_nodes = row.get("unique_nodes")
                if (
                    isinstance(unique_nodes, bool)
                    or not isinstance(unique_nodes, int)
                    or unique_nodes <= 0
                ):
                    violations.append(f"unique_nodes_invalid:{unit}")
                elif unique_nodes > int(declared_thresholds.get("max_unique_nodes", -1)):
                    violations.append(
                        f"unique_nodes:{unique_nodes}>{int(declared_thresholds.get('max_unique_nodes', -1))}"
                    )
                for metric_name, names, direction in _METRIC_FIELDS:
                    value = _metric(row, *names)
                    if value is None:
                        missing_metrics.add(metric_name)
                        continue
                    threshold_name = (
                        "min_active_spearman" if metric_name == "active_spearman" else metric_name
                    )
                    if metric_name == "normalized_l1_error":
                        threshold_name = "max_normalized_l1_error"
                    elif metric_name == "normalized_l2_error":
                        threshold_name = "max_normalized_l2_error"
                    elif metric_name == "normalized_linf_error":
                        threshold_name = "max_normalized_linf_error"
                    elif metric_name.startswith("completeness_"):
                        threshold_name = f"max_{metric_name}"
                    elif metric_name == "spearman":
                        threshold_name = "min_spearman"
                    elif metric_name == "cosine_similarity":
                        threshold_name = "min_cosine_similarity"
                    elif metric_name == "sign_consistency":
                        threshold_name = "min_sign_consistency"
                    elif metric_name == "topk_overlap":
                        threshold_name = "min_topk_overlap"
                    elif metric_name.endswith("quality_tv"):
                        threshold_name = f"max_{metric_name}"
                    elif metric_name == "reference_normalized_l1_error":
                        threshold_name = "max_reference_normalized_l1_error"
                    limit = declared_thresholds.get(threshold_name)
                    if direction != "diagnostic" and limit is not None and (
                        value > float(limit) if direction == "max" else value < float(limit)
                    ):
                        violations.append(f"{metric_name}:{value:g}{'>' if direction == 'max' else '<'}{float(limit):g}")
                for q in REQUIRED_STAGE3_TOP_Q:
                    for metric_name in ("overlap", "jaccard"):
                        value = _topq_metric(row, q, metric_name)
                        if value is None:
                            missing_metrics.add(f"topq_{metric_name}_{q:g}")
                        elif value < float(declared_thresholds.get(f"min_topq_{metric_name}", math.inf)):
                            violations.append(f"topq_{metric_name}_{q:g}:{value:g}<threshold")
                strata = row.get("strata")
                if not isinstance(strata, Mapping):
                    strata = {key: row.get(key) for key in REQUIRED_STAGE3_STRATA}
                if any(strata.get(key) is None for key in REQUIRED_STAGE3_STRATA):
                    violations.append(f"strata_missing:{unit}")
                else:
                    marker = tuple(strata[key] for key in REQUIRED_STAGE3_STRATA)
                    strata_seen.add(marker)
                    normalized_strata = {
                        key: strata[key] for key in REQUIRED_STAGE3_STRATA
                    }
                    strata_by_unit[unit] = normalized_strata
                    if normalized_strata != planned_unit_strata.get(unit):
                        violations.append(f"strata_plan_mismatch:{unit}")
                row_by_unit[unit] = row
                if row.get("scope") not in (None, "formal"):
                    violations.append(f"scope_not_formal:{unit}")
            missing_units = sorted(set(units) - seen_units)
            if missing_units:
                violations.append(f"missing_units:{','.join(missing_units)}")
            if not strata_seen:
                violations.append("strata_empty")
            # Worst cases are derived from the complete frozen table.  A
            # producer-supplied boolean is never accepted as evidence because
            # marking every row true would otherwise satisfy the old contract.
            worst_units: set[str] = set()
            for stratum_key in REQUIRED_STAGE3_STRATA:
                values = {
                    strata[stratum_key] for strata in strata_by_unit.values()
                }
                for stratum_value in values:
                    group = [
                        (unit, row_by_unit[unit])
                        for unit, strata in strata_by_unit.items()
                        if strata[stratum_key] == stratum_value
                    ]
                    if not group:
                        continue
                    node_rows = [
                        (unit, row.get("unique_nodes")) for unit, row in group
                        if isinstance(row.get("unique_nodes"), int)
                        and not isinstance(row.get("unique_nodes"), bool)
                    ]
                    if node_rows:
                        maximum = max(int(value) for _unit, value in node_rows)
                        worst_units.update(
                            unit for unit, value in node_rows if int(value) == maximum
                        )
                    for _metric_name, names, direction in _METRIC_FIELDS:
                        if direction == "diagnostic":
                            continue
                        metric_rows = [
                            (unit, value)
                            for unit, row in group
                            if (value := _metric(row, *names)) is not None
                        ]
                        if not metric_rows:
                            continue
                        boundary = (
                            max(value for _unit, value in metric_rows)
                            if direction == "max"
                            else min(value for _unit, value in metric_rows)
                        )
                        worst_units.update(
                            unit for unit, value in metric_rows if value == boundary
                        )
                    for q in REQUIRED_STAGE3_TOP_Q:
                        for metric_name in ("overlap", "jaccard"):
                            top_rows = [
                                (unit, value)
                                for unit, row in group
                                if (value := _topq_metric(row, q, metric_name)) is not None
                            ]
                            if top_rows:
                                boundary = min(value for _unit, value in top_rows)
                                worst_units.update(
                                    unit for unit, value in top_rows if value == boundary
                                )
            if not worst_units:
                violations.append("worst_case_unit_missing")
            if missing_metrics:
                violations.append("missing_metrics:" + ",".join(sorted(missing_metrics)))
            passed = not violations and not missing_metrics and set(units) == seen_units
            rule_evaluations[rule] = {
                "passing": passed,
                "unit_ids": sorted(seen_units),
                "missing_unit_ids": missing_units,
                "missing_metrics": sorted(missing_metrics),
                "worst_case_unit_ids": sorted(worst_units),
                "worst_case_source": "derived_from_complete_frozen_table",
                "strata_count": len(strata_seen),
                "violations": sorted(set(violations)),
            }
        if not rule_evaluations:
            reasons.append("STAGE3_OBSERVATIONS_REQUIRED")
        any_rule_passes = bool(rule_evaluations) and any(
            bool(value.get("passing")) for value in rule_evaluations.values() if isinstance(value, Mapping)
        )
        if rule_evaluations and not any_rule_passes:
            reasons.append("STAGE3_NO_PASSING_RULES")
        eligible = (
            not reasons
            and any_rule_passes
            and provenance_record is not None
            and len(gate_records) >= len(REQUIRED_STAGE3_GATE_IDS)
        )
        gate_hashes = tuple(gate.artifact_hash for gate in gate_records)
        # Keep the wire shape valid even for a blocked audit with malformed
        # input; a zero hash is never emitted as a fake formal gate.
        if len(gate_hashes) != len(REQUIRED_STAGE3_GATE_IDS):
            gate_hashes = tuple("0" * 64 for _ in REQUIRED_STAGE3_GATE_IDS)
        return Stage3GateEvaluation(
            evaluation_id=str(evaluation_id),
            status="PASS" if eligible else "BLOCKED",
            scope="formal" if isinstance(execution, FormalExecutionEvidence) and execution.run_intent == "formal" else "local_fixture",
            formal_eligible=eligible,
            execution_evidence_hash=(execution.artifact_hash if isinstance(execution, FormalExecutionEvidence) else "0" * 64),
            formal_plan_hash=formal_plan_hash,
            formal_plan_ref=formal_plan_ref or "<missing-plan-ref>",
            thresholds_hash=thresholds_hash,
            required_gate_ids=REQUIRED_STAGE3_GATE_IDS,
            gate_hashes=gate_hashes,
            required_rule_names=rules,
            required_unit_ids=units,
            required_strata=REQUIRED_STAGE3_STRATA,
            required_top_q=REQUIRED_STAGE3_TOP_Q,
            rule_evaluations=MappingProxyType(rule_evaluations),
            provenance_hash=None if provenance_record is None else provenance_record.artifact_hash,
            source_artifact_refs=refs,
            reasons=tuple(sorted(set(reasons))),
            stage3_scope_decision_ref=scope_decision_ref_value,
            stage3_scope_decision_hash=scope_decision_hash_value,
            stage3_scope_gate_ref=scope_gate_ref_value,
            stage3_scope_gate_hash=scope_gate_hash_value,
        )

    def require_pass(self, **kwargs: object) -> Stage3GateEvaluation:
        return self.evaluate(**kwargs).require_pass()  # type: ignore[arg-type]


def evaluate_stage3_gates(**kwargs: object) -> Stage3GateEvaluation:
    """Convenience API used by the CLI and external audit callers."""

    return Stage3GateEvaluator().evaluate(**kwargs)  # type: ignore[arg-type]


evaluate_stage3_gate = evaluate_stage3_gates


__all__ = [
    "REQUIRED_STAGE3_GATE_IDS",
    "REQUIRED_STAGE3_STRATA",
    "REQUIRED_STAGE3_TOP_Q",
    "STAGE3_GATE_EVALUATION_SCHEMA",
    "Stage3GateEvaluation",
    "Stage3GateEvaluator",
    "evaluate_stage3_gate",
    "evaluate_stage3_gates",
]
