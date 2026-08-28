"""Independent, fail-closed finalization of the formal Stage 3 result.

The task runner may publish a candidate recommendation, but it is not an
authority for G3-6 or G3-7.  This module is the narrow finalization boundary:
it reloads the complete frozen observation table, the *live* execution
evidence, G3-0..G3-5 GateRecords, the independent Gate evaluation, and the
completed provenance record.  Only after those objects agree does it construct
G3-6 and G3-7 and qualify a recommendation with
``QuadratureRecommendation.from_mapping_authorized``.

No numbers are generated here and no threshold is inferred.  A malformed,
incomplete, stale, or mismatched input produces a BLOCKED finalization; the
module never turns a partial table into a method selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math

from ..analysis.report import FrozenSourceTable
from ..contracts.errors import FormalRunRejected
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import canonical_json_hash
from ..contracts.provenance import ProvenanceRecord, ProvenanceStatus
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from .stage3_formal import QuadratureRecommendation
from .stage3_gate import (
    REQUIRED_STAGE3_GATE_IDS,
    REQUIRED_STAGE3_STRATA,
    REQUIRED_STAGE3_TOP_Q,
    Stage3GateEvaluation,
)


STAGE3_FINALIZATION_SCHEMA = "stage3-finalization-v1"
_ROW_METRICS = (
    "normalized_l1_error",
    "normalized_l2_error",
    "normalized_linf_error",
    "completeness_absolute_residual",
    "completeness_relative_residual",
    "completeness_l1_scaled_residual",
    "spearman",
    "topk_overlap",
    "active_spearman",
    "cosine_similarity",
    "sign_consistency",
    "layer_quality_tv",
    "module_quality_tv",
    "reference_normalized_l1_error",
)


def _sha256(value: object, *, field: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _ref(value: object, *, field: str, allow_none: bool = False) -> str | None:
    if allow_none and value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "?" in value or "://" in value:
        raise ValueError(f"{field} 必须是稳定 artifact ref")
    lowered = value.casefold()
    if "fixture" in lowered or "synthetic" in lowered:
        raise FormalRunRejected(f"{field} 禁止 fixture/synthetic formal ref")
    return value


def _id(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value[0].isalnum()
        or any(not (char.isalnum() or char in "._-") for char in value)
    ):
        raise ValueError(f"{field} 不是安全标识")
    return value


def _canonical_object(value: Mapping[str, object], *, field: str) -> str:
    supplied = value.get("artifact_hash")
    if not isinstance(supplied, str):
        raise ValueError(f"{field}.artifact_hash 缺失")
    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise ValueError(f"{field}.artifact_hash 与内容不一致")
    return supplied


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} 必须是 object")
    return value


def _as_table(value: FrozenSourceTable | Mapping[str, object]) -> FrozenSourceTable:
    if isinstance(value, FrozenSourceTable):
        table = value
    else:
        table = FrozenSourceTable.from_mapping(dict(_mapping(value, field="frozen_table")))
    lowered = f"{table.name} {table.schema_version}".casefold()
    if "fixture" in lowered or "synthetic" in lowered:
        raise FormalRunRejected("formal finalization 不接受 fixture/synthetic source table")
    return table


def _as_plan(value: Mapping[str, object]) -> tuple[Mapping[str, object], str, tuple[str, ...], tuple[str, ...], dict[str, dict[str, str]]]:
    plan = _mapping(value, field="formal_plan")
    plan_hash = _canonical_object(plan, field="formal_plan")
    raw_units = plan.get("required_unit_ids")
    raw_rules = plan.get("candidate_rules")
    if (
        not isinstance(raw_units, list)
        or not raw_units
        or any(not isinstance(item, str) or not item for item in raw_units)
        or len(set(raw_units)) != len(raw_units)
    ):
        raise ValueError("formal plan required_unit_ids 不完整")
    if (
        not isinstance(raw_rules, list)
        or not raw_rules
        or any(not isinstance(item, str) or not item for item in raw_rules)
        or len(set(raw_rules)) != len(raw_rules)
    ):
        raise ValueError("formal plan candidate_rules 不完整")
    raw_strata = plan.get("unit_strata", {})
    if not isinstance(raw_strata, Mapping):
        raise ValueError("formal plan unit_strata 缺失")
    unit_strata: dict[str, dict[str, str]] = {}
    for unit in raw_units:
        item = raw_strata.get(unit)
        if not isinstance(item, Mapping) or set(item) != set(REQUIRED_STAGE3_STRATA):
            raise ValueError(f"formal plan unit_strata 缺失:{unit}")
        if any(not isinstance(item[key], str) or not item[key] for key in REQUIRED_STAGE3_STRATA):
            raise ValueError(f"formal plan unit_strata 无效:{unit}")
        unit_strata[unit] = {key: item[key] for key in REQUIRED_STAGE3_STRATA}  # type: ignore[index]
    return plan, plan_hash, tuple(raw_units), tuple(raw_rules), unit_strata


def _check_table(
    table: FrozenSourceTable,
    *,
    units: tuple[str, ...],
    rules: tuple[str, ...],
    unit_strata: Mapping[str, Mapping[str, str]],
    thresholds: Mapping[str, object],
) -> tuple[int, int]:
    expected = {(unit, rule) for unit in units for rule in rules}
    observed: set[tuple[str, str]] = set()
    max_reference = thresholds.get("max_reference_normalized_l1_error")
    if not isinstance(max_reference, (int, float)) or isinstance(max_reference, bool):
        raise ValueError("formal plan max_reference_normalized_l1_error 缺失")
    for index, raw in enumerate(table.rows):
        row = _mapping(raw, field=f"frozen_table.rows[{index}]")
        unit, rule = row.get("unit_id"), row.get("rule_name")
        if not isinstance(unit, str) or not isinstance(rule, str):
            raise ValueError(f"frozen table row identity invalid:{index}")
        pair = (unit, rule)
        if pair in observed:
            raise ValueError(f"frozen table duplicate pair:{unit}:{rule}")
        observed.add(pair)
        if pair not in expected:
            raise ValueError(f"frozen table unexpected pair:{unit}:{rule}")
        if row.get("scope") != "formal":
            raise FormalRunRejected(f"frozen table row scope invalid:{unit}:{rule}")
        refs = row.get("evidence_refs")
        if not isinstance(refs, (list, tuple)) or not refs or any(not isinstance(item, str) for item in refs):
            raise FormalRunRejected(f"frozen table row evidence missing:{unit}:{rule}")
        strata = row.get("strata")
        if not isinstance(strata, Mapping) or set(strata) != set(REQUIRED_STAGE3_STRATA):
            raise ValueError(f"frozen table row strata invalid:{unit}:{rule}")
        if dict(strata) != dict(unit_strata[unit]):
            raise ValueError(f"frozen table row strata drift:{unit}:{rule}")
        for metric in _ROW_METRICS:
            value = row.get(metric)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"frozen table row metric invalid:{metric}:{unit}:{rule}")
            if float(value) < 0:
                raise ValueError(f"frozen table row metric negative:{metric}:{unit}:{rule}")
        for metric in ("active_spearman", "cosine_similarity"):
            value = float(row[metric])
            if not -1 <= value <= 1:
                raise ValueError(f"frozen table row metric domain:{metric}:{unit}:{rule}")
        for metric in ("sign_consistency",):
            value = float(row[metric])
            if not 0 <= value <= 1:
                raise ValueError(f"frozen table row metric domain:{metric}:{unit}:{rule}")
        for metric in ("topq_overlap", "topq_jaccard"):
            values = row.get(metric)
            if not isinstance(values, Mapping):
                raise ValueError(f"frozen table row {metric} missing:{unit}:{rule}")
            for q in REQUIRED_STAGE3_TOP_Q:
                key = f"{q:g}"
                raw_q = values.get(key, values.get(str(q)))
                if not isinstance(raw_q, (int, float)) or isinstance(raw_q, bool) or not math.isfinite(float(raw_q)) or not 0 <= float(raw_q) <= 1:
                    raise ValueError(f"frozen table row {metric}[{key}] invalid:{unit}:{rule}")
        for field in ("unique_nodes", "deterministic_node_cost_units"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"frozen table row cost invalid:{field}:{unit}:{rule}")
        if row.get("deterministic_node_cost_units") != row.get("unique_nodes"):
            raise ValueError(f"frozen table row node cost drift:{unit}:{rule}")
        wall = row.get("wall_seconds")
        if not isinstance(wall, (int, float)) or isinstance(wall, bool) or not math.isfinite(float(wall)) or float(wall) < 0:
            raise ValueError(f"frozen table row wall_seconds invalid:{unit}:{rule}")
        if float(row["reference_normalized_l1_error"]) > float(max_reference):
            raise FormalRunRejected(f"frozen table unresolved reference:{unit}:{rule}")
    if observed != expected:
        missing = sorted(expected - observed)
        raise ValueError("frozen table incomplete:" + ",".join(f"{u}:{r}" for u, r in missing[:8]))
    return len(units), len(rules)


def _gate_map(gates: Sequence[GateRecord], *, execution: FormalExecutionEvidence) -> dict[str, GateRecord]:
    expected = set(REQUIRED_STAGE3_GATE_IDS)
    selected = [gate for gate in gates if isinstance(gate, GateRecord) and gate.gate_id in expected]
    by_id = {gate.gate_id: gate for gate in selected}
    if set(by_id) != expected or len(selected) != len(expected):
        raise FormalRunRejected("STAGE3_FINALIZATION_REQUIRES_G3_0_THROUGH_G3_5")
    execution_by_id = {gate.gate_id: gate for gate in execution.prerequisite_gates}
    for gate_id in REQUIRED_STAGE3_GATE_IDS:
        gate = by_id[gate_id]
        if gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
            raise FormalRunRejected(f"STAGE3_FINALIZATION_GATE_NOT_LIVE_PASS:{gate_id}")
        if execution_by_id.get(gate_id) is None or execution_by_id[gate_id].artifact_hash != gate.artifact_hash:
            raise FormalRunRejected(f"STAGE3_FINALIZATION_EXECUTION_GATE_MISMATCH:{gate_id}")
    return by_id


def _candidate_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    candidate: object = value.get("recommendation", value)
    candidate = _mapping(candidate, field="recommendation")
    if candidate.get("schema_version") != "stage3-quadrature-recommendation-v1":
        raise ValueError("formal recommendation schema invalid")
    return candidate


@dataclass(frozen=True, slots=True)
class Stage3Finalization:
    """Reporting-safe immutable Stage 3 finalization payload."""

    finalization_id: str
    status: str
    scope: str
    formal_eligible: bool
    execution_evidence_hash: str | None
    frozen_table_ref: str | None
    frozen_table_hash: str | None
    formal_plan_ref: str | None
    formal_plan_hash: str | None
    evaluation_ref: str | None
    evaluation_hash: str | None
    provenance_ref: str | None
    provenance_hash: str | None
    recommendation_ref: str | None
    g3_6_ref: str | None
    g3_7_ref: str | None
    recommendation: QuadratureRecommendation | None
    candidate_recommendation: Mapping[str, object] | None
    gate_evaluation: Stage3GateEvaluation | None
    prerequisite_gates: tuple[GateRecord, ...]
    g3_6_gate: GateRecord | None
    g3_7_gate: GateRecord | None
    method_selection: Mapping[str, object] | None
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    checked_at: str | None = None
    schema_version: str = STAGE3_FINALIZATION_SCHEMA

    def __post_init__(self) -> None:
        _id(self.finalization_id, field="finalization_id")
        if self.schema_version != STAGE3_FINALIZATION_SCHEMA:
            raise ValueError("STAGE3_FINALIZATION_SCHEMA_UNSUPPORTED")
        if self.status not in {"PASS", "BLOCKED"} or self.scope != "formal":
            raise ValueError("STAGE3_FINALIZATION_STATUS_OR_SCOPE_INVALID")
        if self.formal_eligible is not (self.status == "PASS"):
            raise FormalRunRejected("STAGE3_FINALIZATION_ELIGIBILITY_MISMATCH")
        for name, value in (
            ("execution_evidence_hash", self.execution_evidence_hash),
            ("frozen_table_hash", self.frozen_table_hash),
            ("formal_plan_hash", self.formal_plan_hash),
            ("evaluation_hash", self.evaluation_hash),
            ("provenance_hash", self.provenance_hash),
        ):
            _sha256(value, field=name, allow_none=True)
        for name, value in (
            ("frozen_table_ref", self.frozen_table_ref),
            ("formal_plan_ref", self.formal_plan_ref),
            ("evaluation_ref", self.evaluation_ref),
            ("provenance_ref", self.provenance_ref),
            ("recommendation_ref", self.recommendation_ref),
            ("g3_6_ref", self.g3_6_ref),
            ("g3_7_ref", self.g3_7_ref),
        ):
            _ref(value, field=name, allow_none=True)
        refs = tuple(self.source_artifact_refs)
        if any(not isinstance(item, str) or not item for item in refs) or len(set(refs)) != len(refs):
            raise ValueError("STAGE3_FINALIZATION_SOURCE_REFS_INVALID")
        if any("fixture" in item.casefold() or "synthetic" in item.casefold() for item in refs):
            raise FormalRunRejected("STAGE3_FINALIZATION_SOURCE_REF_FORBIDDEN")
        object.__setattr__(self, "source_artifact_refs", refs)
        if any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("STAGE3_FINALIZATION_REASONS_INVALID")
        if self.status == "PASS":
            required = (
                self.execution_evidence_hash,
                self.frozen_table_ref,
                self.frozen_table_hash,
                self.formal_plan_ref,
                self.formal_plan_hash,
                self.evaluation_ref,
                self.evaluation_hash,
                self.provenance_ref,
                self.provenance_hash,
                self.recommendation_ref,
                self.g3_6_ref,
                self.g3_7_ref,
            )
            if any(item is None for item in required):
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_ALL_BINDINGS")
            if self.recommendation is None or self.recommendation.status != "QUALIFIED":
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_QUALIFIED_RECOMMENDATION")
            if self.gate_evaluation is None or not self.gate_evaluation.formal_eligible:
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_LIVE_EVALUATION")
            if len(self.prerequisite_gates) != len(REQUIRED_STAGE3_GATE_IDS):
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_G3_0_THROUGH_G3_5")
            if self.g3_6_gate is None or self.g3_7_gate is None:
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_G3_6_G3_7")
            if self.g3_6_gate.status is not GateStatus.PASS or self.g3_7_gate.status is not GateStatus.PASS:
                raise FormalRunRejected("STAGE3_FINALIZATION_PASS_REQUIRES_REAL_GATES")
            if not isinstance(self.method_selection, Mapping):
                raise FormalRunRejected("STAGE3_FINALIZATION_METHOD_SELECTION_MISSING")
        if self.checked_at is not None and not isinstance(self.checked_at, str):
            raise ValueError("STAGE3_FINALIZATION_CHECKED_AT_INVALID")

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "finalization_id": self.finalization_id,
            "status": self.status,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "execution_evidence_hash": self.execution_evidence_hash,
            "frozen_table_ref": self.frozen_table_ref,
            "frozen_table_hash": self.frozen_table_hash,
            "formal_plan_ref": self.formal_plan_ref,
            "formal_plan_hash": self.formal_plan_hash,
            "evaluation_ref": self.evaluation_ref,
            "evaluation_hash": self.evaluation_hash,
            "provenance_ref": self.provenance_ref,
            "provenance_hash": self.provenance_hash,
            "recommendation_ref": self.recommendation_ref,
            "g3_6_ref": self.g3_6_ref,
            "g3_7_ref": self.g3_7_ref,
            "recommendation": None if self.recommendation is None else self.recommendation.to_dict(),
            "candidate_recommendation": None if self.candidate_recommendation is None else thaw_json_value(dict(self.candidate_recommendation)),
            "gate_evaluation": None if self.gate_evaluation is None else self.gate_evaluation.to_dict(),
            "prerequisite_gates": [gate.to_dict() for gate in self.prerequisite_gates],
            "g3_6_gate": None if self.g3_6_gate is None else self.g3_6_gate.to_dict(),
            "g3_7_gate": None if self.g3_7_gate is None else self.g3_7_gate.to_dict(),
            "method_selection": None if self.method_selection is None else thaw_json_value(dict(self.method_selection)),
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
            "checked_at": self.checked_at,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def require_pass(self) -> "Stage3Finalization":
        if self.status != "PASS" or not self.formal_eligible:
            raise FormalRunRejected("STAGE3_FINALIZATION_BLOCKED")
        return self

    @classmethod
    def from_mapping_live(
        cls,
        value: Mapping[str, object],
        **kwargs: object,
    ) -> "Stage3Finalization":
        """Strict authority-aware loader kept on the value object as well."""

        return Stage3Finalizer().reload_live(value, **kwargs)  # type: ignore[arg-type]


class Stage3Finalizer:
    """Construct and live-reload formal G3-6/G3-7 finalization."""

    def finalize(
        self,
        *,
        finalization_id: str,
        frozen_table: FrozenSourceTable | Mapping[str, object],
        frozen_table_ref: str,
        formal_plan: Mapping[str, object],
        formal_plan_ref: str,
        execution: FormalExecutionEvidence,
        prerequisite_gates: Sequence[GateRecord] | None = None,
        gate_evaluation: Stage3GateEvaluation,
        evaluation_ref: str,
        provenance: ProvenanceRecord,
        provenance_ref: str,
        recommendation: Mapping[str, object],
        recommendation_ref: str,
        checked_at: str | None = None,
        g3_6_ref: str | None = None,
        g3_7_ref: str | None = None,
        existing_g3_6_gate: GateRecord | None = None,
        existing_g3_6_ref: str | None = None,
    ) -> Stage3Finalization:
        """Return PASS only when every live formal binding verifies.

        All validation errors are represented as a BLOCKED payload so a
        reporting consumer cannot mistake an exception or partial output for
        a successful Stage 3 handoff.
        """

        refs_for_block: list[str] = []
        for raw, name in (
            (frozen_table_ref, "frozen_table_ref"),
            (formal_plan_ref, "formal_plan_ref"),
            (evaluation_ref, "evaluation_ref"),
            (provenance_ref, "provenance_ref"),
            (recommendation_ref, "recommendation_ref"),
        ):
            if isinstance(raw, str) and raw:
                refs_for_block.append(raw)
        if existing_g3_6_ref is not None and g3_6_ref is not None and existing_g3_6_ref != g3_6_ref:
            refs_for_block.append(existing_g3_6_ref)
        g36: GateRecord | None = None
        g37: GateRecord | None = None
        g36_ref_value = existing_g3_6_ref or g3_6_ref or f"{finalization_id}.g3-6"
        g37_ref_value = g3_7_ref or f"{finalization_id}.g3-7"
        timestamp = checked_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        try:
            _id(finalization_id, field="finalization_id")
            table = _as_table(frozen_table)
            _ref(frozen_table_ref, field="frozen_table_ref")
            _ref(formal_plan_ref, field="formal_plan_ref")
            _ref(evaluation_ref, field="evaluation_ref")
            _ref(provenance_ref, field="provenance_ref")
            _ref(recommendation_ref, field="recommendation_ref")
            if existing_g3_6_ref is not None and g3_6_ref is not None and existing_g3_6_ref != g3_6_ref:
                raise FormalRunRejected("STAGE3_FINALIZATION_G3_6_REF_MISMATCH")
            _ref(g36_ref_value, field="g3_6_ref")
            _ref(g37_ref_value, field="g3_7_ref")
            execution.require_for_stage(3)
            if not isinstance(gate_evaluation, Stage3GateEvaluation):
                raise TypeError("live Stage3GateEvaluation required")
            if not isinstance(provenance, ProvenanceRecord):
                raise TypeError("live ProvenanceRecord required")
            if provenance.scope != "formal" or provenance.status is not ProvenanceStatus.COMPLETED or not provenance.worktree_clean or not provenance.formal_eligible:
                raise FormalRunRejected("STAGE3_FINALIZATION_PROVENANCE_NOT_COMPLETED_CLEAN")
            plan, plan_hash, units, rules, strata = _as_plan(formal_plan)
            plan_thresholds = plan.get("thresholds")
            if not isinstance(plan_thresholds, Mapping):
                raise ValueError("formal plan thresholds 缺失")
            if gate_evaluation.formal_eligible is not True or gate_evaluation.status != "PASS":
                raise FormalRunRejected("STAGE3_FINALIZATION_GATE_EVALUATION_NOT_PASS")
            if gate_evaluation.execution_evidence_hash != execution.artifact_hash:
                raise FormalRunRejected("STAGE3_FINALIZATION_EXECUTION_EVALUATION_MISMATCH")
            if gate_evaluation.formal_plan_hash != plan_hash or gate_evaluation.formal_plan_ref != formal_plan_ref:
                raise FormalRunRejected("STAGE3_FINALIZATION_PLAN_EVALUATION_MISMATCH")
            if gate_evaluation.required_unit_ids != units:
                raise FormalRunRejected("STAGE3_FINALIZATION_UNIT_EVALUATION_MISMATCH")
            if gate_evaluation.provenance_hash != provenance.artifact_hash:
                raise FormalRunRejected("STAGE3_FINALIZATION_PROVENANCE_EVALUATION_MISMATCH")
            if frozen_table_ref not in gate_evaluation.source_artifact_refs or formal_plan_ref not in gate_evaluation.source_artifact_refs:
                raise FormalRunRejected("STAGE3_FINALIZATION_EVALUATION_SOURCE_BINDING_MISSING")
            if not set(gate_evaluation.source_artifact_refs).issubset(set(provenance.artifact_refs)):
                raise FormalRunRejected("STAGE3_FINALIZATION_PROVENANCE_SOURCE_BINDING_MISSING")
            _check_table(
                table,
                units=units,
                rules=rules,
                unit_strata=strata,
                thresholds=plan_thresholds,
            )
            gate_input = tuple(prerequisite_gates or ())
            pre = _gate_map(gate_input, execution=execution)
            expected_hashes = tuple(pre[gate_id].artifact_hash for gate_id in REQUIRED_STAGE3_GATE_IDS)
            if gate_evaluation.gate_hashes != expected_hashes:
                raise FormalRunRejected("STAGE3_FINALIZATION_EVALUATION_GATE_HASHES_MISMATCH")
            expected_g36_evidence = {
                frozen_table_ref,
                evaluation_ref,
                provenance_ref,
                formal_plan_ref,
            }
            expected_passing_rules = [
                rule
                for rule, raw in gate_evaluation.rule_evaluations.items()
                if isinstance(raw, Mapping) and raw.get("passing") is True
            ]
            expected_g36_measured = {
                "source_table_hash": table.content_hash,
                "evaluation_hash": gate_evaluation.artifact_hash,
                "observation_count": len(table.rows),
                "expected_observation_count": len(units) * len(rules),
                "unit_count": len(units),
                "rule_count": len(rules),
                "all_rows_finite": True,
                "passing_rules": expected_passing_rules,
            }
            expected_g36_threshold = {
                "complete_unit_rule_coverage": True,
                "all_rows_finite": True,
                "independent_evaluation_status": "PASS",
            }
            if existing_g3_6_gate is not None:
                if not isinstance(existing_g3_6_gate, GateRecord):
                    raise TypeError("existing_g3_6_gate 必须是 live GateRecord")
                if existing_g3_6_gate.gate_id != "stage3.G3-6" or existing_g3_6_gate.stage != 3:
                    raise FormalRunRejected("STAGE3_FINALIZATION_EXISTING_G3_6_ID_INVALID")
                if existing_g3_6_gate.status is not GateStatus.PASS or existing_g3_6_gate.effective_status() is not GateStatus.PASS:
                    raise FormalRunRejected("STAGE3_FINALIZATION_EXISTING_G3_6_NOT_LIVE_PASS")
                if not expected_g36_evidence.issubset(set(existing_g3_6_gate.evidence_refs)):
                    raise FormalRunRejected("STAGE3_FINALIZATION_EXISTING_G3_6_EVIDENCE_UNBOUND")
                if dict(existing_g3_6_gate.measured) != expected_g36_measured:
                    raise FormalRunRejected("STAGE3_FINALIZATION_EXISTING_G3_6_MEASURED_DRIFT")
                if dict(existing_g3_6_gate.threshold) != expected_g36_threshold:
                    raise FormalRunRejected("STAGE3_FINALIZATION_EXISTING_G3_6_THRESHOLD_DRIFT")
                g36 = existing_g3_6_gate
            else:
                # Compatibility path for local callers predating the
                # Stage3.08 producer.  Formal orchestration must pass the
                # committed Stage3.08 GateRecord through existing_g3_6_gate.
                g36 = GateRecord(
                    gate_id="stage3.G3-6",
                    stage=3,
                    status=GateStatus.PASS,
                    checked_at=timestamp,
                    measured=expected_g36_measured,
                    threshold=expected_g36_threshold,
                    evidence_refs=tuple(dict.fromkeys(expected_g36_evidence)),
                )
            candidate_wire = _candidate_mapping(recommendation)
            candidate = QuadratureRecommendation.from_mapping(candidate_wire)
            candidate_thresholds = candidate.thresholds
            candidate_thresholds.require_formal_contract()
            if candidate.execution_evidence_hash != execution.artifact_hash or candidate.required_unit_ids != units or candidate_thresholds.artifact_hash != gate_evaluation.thresholds_hash:
                raise FormalRunRejected("STAGE3_FINALIZATION_RECOMMENDATION_BINDING_MISMATCH")
            passing = tuple(expected_passing_rules)
            if set(passing) != set(candidate.passing_rules) or not candidate.passing_rules:
                raise FormalRunRejected("STAGE3_FINALIZATION_PASSING_RULES_MISMATCH")
            if candidate.default_rule is None or candidate.default_rule not in candidate.passing_rules:
                raise FormalRunRejected("STAGE3_FINALIZATION_DEFAULT_RULE_INVALID")
            if candidate.fallback_rule is None or candidate.fallback_rule not in candidate.passing_rules or candidate.fallback_rule == candidate.default_rule:
                raise FormalRunRejected("STAGE3_FINALIZATION_FALLBACK_RULE_INVALID")
            if candidate.status != "FORMAL_CANDIDATE" or candidate.scope != "formal" or candidate.qualification.formal_eligible:
                raise FormalRunRejected("STAGE3_FINALIZATION_CANDIDATE_STATUS_INVALID")
            g37_evidence = tuple(dict.fromkeys((g36_ref_value, recommendation_ref, evaluation_ref, provenance_ref, frozen_table_ref, formal_plan_ref)))
            g37 = GateRecord(
                gate_id="stage3.G3-7",
                stage=3,
                status=GateStatus.PASS,
                checked_at=timestamp,
                measured={
                    "default_rule": candidate.default_rule,
                    "fallback_rule": candidate.fallback_rule,
                    "passing_rules": list(candidate.passing_rules),
                    "evaluation_hash": gate_evaluation.artifact_hash,
                    "g3_6_hash": g36.artifact_hash,
                    "frozen_table_hash": table.content_hash,
                    "provenance_hash": provenance.artifact_hash,
                },
                threshold={"unique_default": True, "explicit_fallback": True, "gate_evaluation_status": "PASS"},
                evidence_refs=g37_evidence,
            )
            qualified = QuadratureRecommendation.from_mapping_authorized(
                candidate_wire,
                execution=execution,
                gate=g37,
                artifact_ref=recommendation_ref,
                gate_evaluation=gate_evaluation,
                provenance=provenance,
            )
            source_refs = tuple(dict.fromkeys((*gate_evaluation.source_artifact_refs, frozen_table_ref, formal_plan_ref, evaluation_ref, provenance_ref, recommendation_ref, g36_ref_value, g37_ref_value)))
            method_selection = {
                "schema_version": "stage3-formal-method-selection-v1",
                "status": "QUALIFIED",
                "scope": "formal",
                "path_type": "full_update",
                "default_rule": qualified.default_rule,
                "fallback_rule": qualified.fallback_rule,
                "passing_rules": list(qualified.passing_rules),
                "required_unit_ids": list(qualified.required_unit_ids),
                "thresholds_hash": qualified.thresholds.artifact_hash,
                "execution_evidence_hash": execution.artifact_hash,
                "frozen_table_ref": frozen_table_ref,
                "frozen_table_hash": table.content_hash,
                "formal_plan_ref": formal_plan_ref,
                "formal_plan_hash": plan_hash,
                "evaluation_ref": evaluation_ref,
                "evaluation_hash": gate_evaluation.artifact_hash,
                "provenance_ref": provenance_ref,
                "provenance_hash": provenance.artifact_hash,
                "g3_6_gate_hash": g36.artifact_hash,
                "g3_7_gate_hash": g37.artifact_hash,
            }
            return Stage3Finalization(
                finalization_id=finalization_id,
                status="PASS",
                scope="formal",
                formal_eligible=True,
                execution_evidence_hash=execution.artifact_hash,
                frozen_table_ref=frozen_table_ref,
                frozen_table_hash=table.content_hash,
                formal_plan_ref=formal_plan_ref,
                formal_plan_hash=plan_hash,
                evaluation_ref=evaluation_ref,
                evaluation_hash=gate_evaluation.artifact_hash,
                provenance_ref=provenance_ref,
                provenance_hash=provenance.artifact_hash,
                recommendation_ref=recommendation_ref,
                g3_6_ref=g36_ref_value,
                g3_7_ref=g37_ref_value,
                recommendation=qualified,
                candidate_recommendation=candidate_wire,
                gate_evaluation=gate_evaluation,
                prerequisite_gates=tuple(pre[gate_id] for gate_id in REQUIRED_STAGE3_GATE_IDS),
                g3_6_gate=g36,
                g3_7_gate=g37,
                method_selection=method_selection,
                source_artifact_refs=source_refs,
                checked_at=timestamp,
            )
        except (FormalRunRejected, TypeError, ValueError, KeyError) as error:
            reason = f"{type(error).__name__}:{error}"
            if g36 is not None and g37 is None:
                try:
                    g37 = GateRecord(
                        gate_id="stage3.G3-7",
                        stage=3,
                        status=GateStatus.BLOCKED,
                        checked_at=timestamp,
                        measured={"g3_6_hash": g36.artifact_hash},
                        threshold={"required_complete_g3_6": True},
                        evidence_refs=tuple(dict.fromkeys((g36_ref_value, *refs_for_block))),
                        reasons=(reason,),
                    )
                except (TypeError, ValueError, FormalRunRejected):
                    g37 = None
            return Stage3Finalization(
                finalization_id=finalization_id,
                status="BLOCKED",
                scope="formal",
                formal_eligible=False,
                execution_evidence_hash=getattr(execution, "artifact_hash", None),
                frozen_table_ref=frozen_table_ref if isinstance(frozen_table_ref, str) and frozen_table_ref else None,
                frozen_table_hash=getattr(frozen_table, "content_hash", None),
                formal_plan_ref=formal_plan_ref if isinstance(formal_plan_ref, str) and formal_plan_ref else None,
                formal_plan_hash=getattr(formal_plan, "get", lambda _key: None)("artifact_hash") if isinstance(formal_plan, Mapping) else None,
                evaluation_ref=evaluation_ref if isinstance(evaluation_ref, str) and evaluation_ref else None,
                evaluation_hash=getattr(gate_evaluation, "artifact_hash", None),
                provenance_ref=provenance_ref if isinstance(provenance_ref, str) and provenance_ref else None,
                provenance_hash=getattr(provenance, "artifact_hash", None),
                recommendation_ref=recommendation_ref if isinstance(recommendation_ref, str) and recommendation_ref else None,
                g3_6_ref=g36_ref_value if g36 is not None else (g3_6_ref if isinstance(g3_6_ref, str) and g3_6_ref else None),
                g3_7_ref=g37_ref_value if g37 is not None else (g3_7_ref if isinstance(g3_7_ref, str) and g3_7_ref else None),
                recommendation=None,
                candidate_recommendation=None,
                gate_evaluation=gate_evaluation if isinstance(gate_evaluation, Stage3GateEvaluation) else None,
                prerequisite_gates=tuple(gate for gate in (prerequisite_gates or ()) if isinstance(gate, GateRecord)),
                g3_6_gate=g36,
                g3_7_gate=g37,
                method_selection=None,
                source_artifact_refs=tuple(dict.fromkeys(refs_for_block)),
                reasons=(reason,),
                checked_at=checked_at,
            )

    def reload_live(
        self,
        value: Mapping[str, object],
        *,
        frozen_table: FrozenSourceTable | Mapping[str, object],
        formal_plan: Mapping[str, object],
        execution: FormalExecutionEvidence,
        prerequisite_gates: Sequence[GateRecord] | None = None,
        gate_evaluation: Stage3GateEvaluation,
        provenance: ProvenanceRecord,
        existing_g3_6_gate: GateRecord | None = None,
        existing_g3_6_ref: str | None = None,
    ) -> Stage3Finalization:
        """Rebuild from live authorities and reject any persisted hash drift."""

        payload = _mapping(value, field="stage3_finalization")
        supplied = payload.get("artifact_hash")
        if not isinstance(supplied, str):
            raise FormalRunRejected("STAGE3_FINALIZATION_ARTIFACT_HASH_MISSING")
        expected_keys = set(Stage3Finalization.__dataclass_fields__) - {"artifact_hash"}
        if set(payload) != expected_keys | {"artifact_hash"}:
            raise ValueError("STAGE3_FINALIZATION_FIELDS_MISMATCH")
        recomputed = canonical_json_hash({key: item for key, item in payload.items() if key != "artifact_hash"})
        if recomputed != supplied:
            raise ValueError("STAGE3_FINALIZATION_ARTIFACT_HASH_MISMATCH")
        candidate = _mapping(payload.get("candidate_recommendation"), field="candidate_recommendation")
        result = self.finalize(
            finalization_id=str(payload["finalization_id"]),
            frozen_table=frozen_table,
            frozen_table_ref=str(payload["frozen_table_ref"]),
            formal_plan=formal_plan,
            formal_plan_ref=str(payload["formal_plan_ref"]),
            execution=execution,
            prerequisite_gates=prerequisite_gates,
            gate_evaluation=gate_evaluation,
            evaluation_ref=str(payload["evaluation_ref"]),
            provenance=provenance,
            provenance_ref=str(payload["provenance_ref"]),
            recommendation=candidate,
            recommendation_ref=str(payload["recommendation_ref"]),
            g3_6_ref=str(payload["g3_6_ref"]),
            g3_7_ref=str(payload["g3_7_ref"]),
            existing_g3_6_gate=existing_g3_6_gate,
            existing_g3_6_ref=existing_g3_6_ref,
            checked_at=str(payload["checked_at"]),
        )
        if result.status != "PASS" or result.artifact_hash != supplied:
            raise FormalRunRejected("STAGE3_FINALIZATION_LIVE_RELOAD_MISMATCH")
        return result


def finalize_stage3(**kwargs: object) -> Stage3Finalization:
    """Functional wrapper around :class:`Stage3Finalizer`."""

    return Stage3Finalizer().finalize(**kwargs)  # type: ignore[arg-type]


def reload_stage3_finalization(value: Mapping[str, object], **kwargs: object) -> Stage3Finalization:
    """Functional wrapper for strict live reload."""

    return Stage3Finalizer().reload_live(value, **kwargs)  # type: ignore[arg-type]


load_stage3_finalization = reload_stage3_finalization


__all__ = [
    "STAGE3_FINALIZATION_SCHEMA",
    "Stage3Finalization",
    "Stage3Finalizer",
    "finalize_stage3",
    "reload_stage3_finalization",
    "load_stage3_finalization",
]
