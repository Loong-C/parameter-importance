"""Independent, commit-only publisher for the formal Stage 3 G3-7 handoff.

This module is deliberately downstream of the Stage3.08 publisher.  It does
not run an experiment and it does not construct G3-6.  All inputs are loaded
again from formal task *commits* before anything is written.  The only Gate
constructed here is G3-7; it is committed before a candidate can be qualified.

The ordering is important::

    frozen source/cost/candidate + plan + execution + provenance + G3-6
      -> G3-7 Gate commit
      -> qualified recommendation commit
      -> finalization commit
      -> publication receipt commit

In particular, neither the candidate nor the G3-7 Gate contains a reference to
an artifact that has not existed at the point it is created.  Provenance is a
completed, clean input and is never amended with output references.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any

from ..analysis.report import FrozenSourceTable
from ..contracts.errors import FormalRunRejected
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import JSONValue, canonical_json_hash
from ..contracts.provenance import ProvenanceRecord, ProvenanceStatus
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.status import GateRecord, GateStatus
from ..contracts.stage3_scope import validate_stage3_scope_decision
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from .stage3_finalization import Stage3Finalization
from .stage3_formal import QuadratureRecommendation, QuadratureThresholds
from .stage3_gate import (
    REQUIRED_STAGE3_GATE_IDS,
    REQUIRED_STAGE3_STRATA,
    REQUIRED_STAGE3_TOP_Q,
    Stage3GateEvaluation,
)


STAGE3_G37_PUBLICATION_SCHEMA = "stage3-g37-publication-v1"
STAGE3_G37_TASK_ID = "stage3.09_g3_7_publisher"
STAGE3_G37_GATE_ARTIFACT_KIND = "gate_record"
STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND = "quadrature_decision"
STAGE3_G37_FINALIZATION_ARTIFACT_KIND = "finalization"
STAGE3_G37_PUBLICATION_ARTIFACT_KIND = "g37_publication"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[^?]+$")
_TABLE_KINDS = frozenset({"frozen_source_table"})
_COST_KINDS = frozenset({"cost_accuracy_table"})
_CANDIDATE_KINDS = frozenset({"quadrature_decision"})
_PLAN_KINDS = frozenset({"formal_plan", "stage3_formal_plan"})
_EXECUTION_KINDS = frozenset({"formal_execution_evidence", "execution_evidence"})
_PROVENANCE_KINDS = frozenset({"provenance", "provenance_record"})
_EVALUATION_KINDS = frozenset({"gate_evaluation"})
_GATE_KINDS = frozenset({"gate_record"})


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _ref(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or _REF_RE.fullmatch(value) is None
        or "\\" in value
        or "://" in value
        or value.startswith("/")
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError(f"{field} must be a stable commit ref")
    if any(marker in value.casefold() for marker in ("fixture", "synthetic")):
        raise FormalRunRejected(f"{field} contains a fixture/synthetic ref")
    return value


def _identifier(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", value)
    ):
        raise ValueError(f"{field} is not a safe identifier")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _load_formal(
    root: Path,
    reference: str,
    *,
    field: str,
    kinds: frozenset[str],
) -> LoadedTaskArtifact:
    """Load one and only one formal task commit, with no path fallback."""

    _ref(reference, field=field)
    try:
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G37_{field.upper()}_COMMIT_INVALID") from error
    if loaded.identity.artifact_kind not in kinds:
        raise FormalRunRejected(
            f"STAGE3_G37_{field.upper()}_ARTIFACT_KIND_INVALID:{loaded.identity.artifact_kind}"
        )
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise FormalRunRejected(f"STAGE3_G37_{field.upper()}_FORMAL_COMMIT_REQUIRED")
    return loaded


def _canonical(value: Mapping[str, object], *, field: str) -> str:
    supplied = value.get("artifact_hash")
    _hash(supplied, field=f"{field}.artifact_hash")
    observed = canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    if observed != supplied:
        raise FormalRunRejected(f"{field}.artifact_hash does not match content")
    return supplied


def _as_candidate(value: Mapping[str, object]) -> Mapping[str, object]:
    """Accept direct Stage3.09 output and the historical one-key wrapper."""

    candidate: object = value
    for key in ("quadrature_decision", "recommendation", "candidate_recommendation"):
        if isinstance(candidate, Mapping) and key in candidate:
            candidate = candidate[key]
            break
    candidate = _mapping(candidate, field="quadrature_decision")
    if candidate.get("schema_version") != "stage3-quadrature-recommendation-v1":
        raise FormalRunRejected("STAGE3_G37_CANDIDATE_SCHEMA_INVALID")
    return candidate


def _formal_plan(
    value: Mapping[str, object], execution: FormalExecutionEvidence
) -> tuple[Mapping[str, object], str, tuple[str, ...], tuple[str, ...], dict[str, dict[str, str]], QuadratureThresholds]:
    plan = _mapping(value, field="formal_plan")
    plan_hash = _canonical(plan, field="formal_plan")
    if (
        plan.get("schema_version") != "stage3-formal-pilot-plan-v1"
        or plan.get("scope") != "formal"
        or plan.get("formal_eligible") is not True
        or plan.get("execution_evidence_hash") != execution.artifact_hash
    ):
        raise FormalRunRejected("STAGE3_G37_FORMAL_PLAN_BINDING_INVALID")
    rules = plan.get("candidate_rules")
    units = plan.get("required_unit_ids")
    if (
        not isinstance(rules, list)
        or not rules
        or any(not isinstance(item, str) or not item for item in rules)
        or len(set(rules)) != len(rules)
        or not isinstance(units, list)
        or not units
        or any(not isinstance(item, str) or not item for item in units)
        or len(set(units)) != len(units)
    ):
        raise FormalRunRejected("STAGE3_G37_FORMAL_PLAN_COVERAGE_INVALID")
    raw_strata = plan.get("unit_strata")
    if not isinstance(raw_strata, Mapping) or set(raw_strata) != set(units):
        raise FormalRunRejected("STAGE3_G37_FORMAL_PLAN_UNIT_STRATA_INVALID")
    strata: dict[str, dict[str, str]] = {}
    for unit in units:
        item = raw_strata.get(unit)
        if (
            not isinstance(item, Mapping)
            or set(item) != set(REQUIRED_STAGE3_STRATA)
            or any(not isinstance(item[key], str) or not item[key] for key in REQUIRED_STAGE3_STRATA)
        ):
            raise FormalRunRejected(f"STAGE3_G37_FORMAL_PLAN_UNIT_STRATA_INVALID:{unit}")
        strata[unit] = {key: item[key] for key in REQUIRED_STAGE3_STRATA}  # type: ignore[index]
    raw_thresholds = plan.get("thresholds")
    if not isinstance(raw_thresholds, Mapping):
        raise FormalRunRejected("STAGE3_G37_FORMAL_PLAN_THRESHOLDS_MISSING")
    try:
        thresholds = QuadratureThresholds(**dict(raw_thresholds)).require_formal_contract()
    except (TypeError, ValueError, FormalRunRejected) as error:
        raise FormalRunRejected(f"STAGE3_G37_FORMAL_PLAN_THRESHOLDS_INVALID:{error}") from error
    return plan, plan_hash, tuple(units), tuple(rules), strata, thresholds


def _rows(table: FrozenSourceTable, *, field: str) -> dict[tuple[str, str], Mapping[str, object]]:
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for index, raw in enumerate(table.rows):
        row = _mapping(raw, field=f"{field}.rows[{index}]")
        # FrozenSourceTable deliberately freezes arrays as tuples internally;
        # canonical JSON and the on-disk wire representation use lists.
        thawed = thaw_json_value(dict(row))
        if not isinstance(thawed, Mapping):  # pragma: no cover - defensive
            raise TypeError(f"{field}.rows[{index}] must be an object")
        row = thawed
        unit, rule = row.get("unit_id"), row.get("rule_name")
        if not isinstance(unit, str) or not unit or not isinstance(rule, str) or not rule:
            raise FormalRunRejected(f"STAGE3_G37_{field.upper()}_ROW_ID_INVALID:{index}")
        pair = (unit, rule)
        if pair in result:
            raise FormalRunRejected(f"STAGE3_G37_DUPLICATE_ROW:{unit}:{rule}")
        if row.get("scope") != "formal":
            raise FormalRunRejected(f"STAGE3_G37_{field.upper()}_ROW_SCOPE_INVALID:{unit}:{rule}")
        # A cost table is not allowed to silently change a scientific scalar.
        # Finite checking here also catches a cost-only table that bypasses
        # the evaluator's complete observation pass.
        for name, value in row.items():
            if name in {"top_q", "topq_overlap", "topq_jaccard"} and isinstance(value, Mapping):
                values: list[object] = list(value.values())
                for child in values:
                    if isinstance(child, Mapping):
                        values.extend(child.values())
                for child in values:
                    if isinstance(child, Mapping):
                        continue
                    if isinstance(child, (int, float)) and not isinstance(child, bool) and not math.isfinite(float(child)):
                        raise FormalRunRejected(f"STAGE3_G37_NONFINITE_ROW:{unit}:{rule}:{name}")
            elif isinstance(value, float) and not math.isfinite(value):
                raise FormalRunRejected(f"STAGE3_G37_NONFINITE_ROW:{unit}:{rule}:{name}")
        result[pair] = row
    if not result:
        raise FormalRunRejected(f"STAGE3_G37_{field.upper()}_EMPTY")
    return result


def _check_row_equality(
    frozen: FrozenSourceTable,
    cost: FrozenSourceTable,
    *,
    units: tuple[str, ...],
    rules: tuple[str, ...],
) -> tuple[dict[tuple[str, str], Mapping[str, object]], dict[tuple[str, str], Mapping[str, object]]]:
    frozen_rows = _rows(frozen, field="frozen_source_table")
    cost_rows = _rows(cost, field="cost_accuracy_table")
    expected = {(unit, rule) for unit in units for rule in rules}
    if set(frozen_rows) != expected:
        raise FormalRunRejected("STAGE3_G37_FROZEN_TABLE_PLAN_ROW_SET_MISMATCH")
    if set(cost_rows) != expected:
        raise FormalRunRejected("STAGE3_G37_COST_TABLE_PLAN_ROW_SET_MISMATCH")
    for pair in sorted(expected):
        left = canonical_json_hash(dict(frozen_rows[pair]))
        right = canonical_json_hash(dict(cost_rows[pair]))
        if left != right:
            raise FormalRunRejected(
                f"STAGE3_G37_COST_FROZEN_ROW_MISMATCH:{pair[0]}:{pair[1]}"
            )
    return frozen_rows, cost_rows


def _scope_from_evaluation(
    root: Path,
    evaluation: Stage3GateEvaluation,
    execution: FormalExecutionEvidence,
    *,
    decision_ref: str | None,
    gate_ref: str | None,
) -> tuple[str, str]:
    """Live-reload the G3-0 authority referenced by the committed evaluator."""

    evaluated_decision_ref = evaluation.stage3_scope_decision_ref
    evaluated_gate_ref = evaluation.stage3_scope_gate_ref
    if decision_ref is not None and decision_ref != evaluated_decision_ref:
        raise FormalRunRejected("STAGE3_G37_SCOPE_DECISION_REF_MISMATCH")
    if gate_ref is not None and gate_ref != evaluated_gate_ref:
        raise FormalRunRejected("STAGE3_G37_SCOPE_GATE_REF_MISMATCH")
    decision_ref = decision_ref or evaluated_decision_ref
    gate_ref = gate_ref or evaluated_gate_ref
    if not isinstance(decision_ref, str) or not isinstance(gate_ref, str):
        raise FormalRunRejected("STAGE3_G37_SCOPE_AUTHORITY_REFS_REQUIRED")
    decision_artifact = _load_formal(root, decision_ref, field="scope_decision", kinds=frozenset({"scope_authority", "stage3_scope_authority"}))
    gate_artifact = _load_formal(root, gate_ref, field="scope_gate", kinds=_GATE_KINDS)
    decision = _mapping(decision_artifact.payload, field="scope_decision")
    gate = GateRecord.from_mapping(dict(gate_artifact.payload))
    validate_stage3_scope_decision(decision)
    if (
        decision.get("artifact_hash") != evaluation.stage3_scope_decision_hash
        or gate.artifact_hash != evaluation.stage3_scope_gate_hash
        or gate.gate_id != "stage3.G3-0"
        or gate.status is not GateStatus.PASS
        or gate.effective_status() is not GateStatus.PASS
    ):
        raise FormalRunRejected("STAGE3_G37_SCOPE_AUTHORITY_EVALUATION_MISMATCH")
    execution_gate = next((item for item in execution.prerequisite_gates if item.gate_id == "stage3.G3-0"), None)
    if execution_gate is None or execution_gate.artifact_hash != gate.artifact_hash:
        raise FormalRunRejected("STAGE3_G37_SCOPE_GATE_EXECUTION_MISMATCH")
    return decision_ref, gate_ref


def _check_execution(
    execution: FormalExecutionEvidence,
) -> tuple[GateRecord, ...]:
    execution.require_for_stage(3)
    by_id = {gate.gate_id: gate for gate in execution.prerequisite_gates}
    if set(by_id) != set(REQUIRED_STAGE3_GATE_IDS):
        raise FormalRunRejected("STAGE3_G37_EXECUTION_G3_0_THROUGH_G3_5_REQUIRED")
    for gate_id in REQUIRED_STAGE3_GATE_IDS:
        gate = by_id[gate_id]
        if gate.stage != 3 or gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
            raise FormalRunRejected(f"STAGE3_G37_EXECUTION_GATE_NOT_LIVE_PASS:{gate_id}")
        if not gate.evidence_refs:
            raise FormalRunRejected(f"STAGE3_G37_EXECUTION_GATE_EVIDENCE_MISSING:{gate_id}")
    return tuple(by_id[item] for item in REQUIRED_STAGE3_GATE_IDS)


def _candidate_consistency(
    candidate: QuadratureRecommendation,
    evaluation: Stage3GateEvaluation,
    *,
    execution: FormalExecutionEvidence,
    plan_hash: str,
    units: tuple[str, ...],
    rules: tuple[str, ...],
    thresholds: QuadratureThresholds,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if candidate.scope != "formal" or candidate.status != "FORMAL_CANDIDATE" or candidate.qualification.formal_eligible:
        reasons.append("CANDIDATE_MUST_BE_UNQUALIFIED_FORMAL_CANDIDATE")
    if candidate.execution_evidence_hash != execution.artifact_hash:
        reasons.append("CANDIDATE_EXECUTION_EVIDENCE_MISMATCH")
    if candidate.required_unit_ids != units:
        reasons.append("CANDIDATE_REQUIRED_UNITS_MISMATCH")
    if len(set(candidate.passing_rules)) != len(candidate.passing_rules):
        reasons.append("CANDIDATE_PASSING_RULES_DUPLICATE")
    if candidate.thresholds.artifact_hash != thresholds.artifact_hash:
        reasons.append("CANDIDATE_THRESHOLDS_MISMATCH")
    if candidate.passing_rules and set(candidate.passing_rules) - set(rules):
        reasons.append("CANDIDATE_RULES_OUTSIDE_FORMAL_PLAN")
    evaluated = {
        name for name, raw in evaluation.rule_evaluations.items()
        if isinstance(raw, Mapping) and raw.get("passing") is True
    }
    if set(candidate.passing_rules) != evaluated:
        reasons.append("CANDIDATE_EVALUATOR_PASSING_RULES_MISMATCH")
    if evaluation.status != "PASS" or evaluation.formal_eligible is not True:
        reasons.append("G3_6_EVALUATION_NOT_PASS")
    if evaluation.formal_plan_hash != plan_hash:
        reasons.append("G3_6_EVALUATION_PLAN_HASH_MISMATCH")
    if evaluation.required_unit_ids != units or set(evaluation.required_rule_names) != set(rules):
        reasons.append("G3_6_EVALUATION_PLAN_COVERAGE_MISMATCH")
    if evaluation.thresholds_hash != thresholds.artifact_hash:
        reasons.append("G3_6_EVALUATION_THRESHOLDS_MISMATCH")
    return tuple(dict.fromkeys(reasons))


def _cost_key(rows: Mapping[tuple[str, str], Mapping[str, object]], rule: str, units: tuple[str, ...]) -> tuple[float, float, float, str]:
    selected = [rows[(unit, rule)] for unit in units]
    def number(row: Mapping[str, object], *names: str, default: float = 0.0) -> float:
        for name in names:
            value = row.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                return float(value)
        return default
    # Prefer the measured callback count, then deterministic nodes, then wall
    # time.  The final rule name tie-break is deterministic and is not a
    # scientific preference.
    measured_gradients = [number(row, "gradient_evaluations", default=0.0) for row in selected]
    # Some early formal ledgers leave callback counts at zero while retaining
    # the deterministic node count.  Zero must not make every rule tie on the
    # primary cost axis.
    gradients = max(measured_gradients)
    if gradients <= 0:
        gradients = max(number(row, "deterministic_node_cost_units", "unique_nodes") for row in selected)
    wall = sum(number(row, "wall_seconds") for row in selected) / len(selected)
    memory = max(number(row, "peak_gpu_memory_bytes") for row in selected)
    return gradients, wall, memory, rule


def _g37_measured(
    candidate: QuadratureRecommendation,
    cost_rows: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    units: tuple[str, ...],
    evaluation: Stage3GateEvaluation,
    g36: GateRecord,
    frozen: FrozenSourceTable,
    cost: FrozenSourceTable,
) -> dict[str, JSONValue]:
    passing_costs = {
        rule: list(_cost_key(cost_rows, rule, units))
        for rule in candidate.passing_rules
    }
    return {
        "candidate_hash": candidate.artifact_hash,
        "frozen_source_table_hash": frozen.content_hash,
        "cost_accuracy_table_hash": cost.content_hash,
        "evaluation_hash": evaluation.artifact_hash,
        "g3_6_hash": g36.artifact_hash,
        "passing_rules": list(candidate.passing_rules),
        "default_rule": candidate.default_rule,
        "fallback_rule": candidate.fallback_rule,
        "passing_rule_cost_keys": passing_costs,
    }


def _reason_tuple(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(item) for item in values if isinstance(item, str) and item))


@dataclass(frozen=True, slots=True)
class Stage3G37Publication:
    """Immutable receipt for the G3-7 Gate and all downstream commits."""

    publication_id: str
    task_id: str
    config_hash: str
    input_config_hash: str
    input_config_hashes: Mapping[str, str]
    publication_config_hash: str
    status: str
    formal_eligible: bool
    frozen_source_table_ref: str
    frozen_source_table_hash: str
    cost_accuracy_table_ref: str
    cost_accuracy_table_hash: str
    quadrature_decision_ref: str
    quadrature_decision_hash: str
    formal_plan_ref: str
    formal_plan_hash: str
    execution_evidence_ref: str
    execution_evidence_hash: str
    provenance_ref: str
    provenance_hash: str
    evaluation_ref: str
    evaluation_hash: str
    g3_6_ref: str
    g3_6_hash: str
    g3_7_ref: str
    g3_7_hash: str
    recommendation_ref: str | None
    recommendation_hash: str | None
    finalization_ref: str | None
    finalization_hash: str | None
    candidate_recommendation: Mapping[str, object]
    recommendation: Mapping[str, object] | None
    finalization: Mapping[str, object] | None
    gate_evaluation: Stage3GateEvaluation
    g3_6_gate: GateRecord
    g3_7_gate: GateRecord
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = STAGE3_G37_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_G37_PUBLICATION_SCHEMA:
            raise ValueError("STAGE3_G37_PUBLICATION_SCHEMA_UNSUPPORTED")
        _identifier(self.publication_id, field="publication_id")
        _identifier(self.task_id, field="task_id")
        for field, value in (
            ("config_hash", self.config_hash),
            ("input_config_hash", self.input_config_hash),
            ("publication_config_hash", self.publication_config_hash),
            ("frozen_source_table_hash", self.frozen_source_table_hash),
            ("cost_accuracy_table_hash", self.cost_accuracy_table_hash),
            ("quadrature_decision_hash", self.quadrature_decision_hash),
            ("formal_plan_hash", self.formal_plan_hash),
            ("execution_evidence_hash", self.execution_evidence_hash),
            ("provenance_hash", self.provenance_hash),
            ("evaluation_hash", self.evaluation_hash),
            ("g3_6_hash", self.g3_6_hash),
            ("g3_7_hash", self.g3_7_hash),
        ):
            _hash(value, field=field)
        if not self.input_config_hashes or any(not isinstance(key, str) or not key for key in self.input_config_hashes):
            raise ValueError("STAGE3_G37_PUBLICATION_INPUT_CONFIG_HASHES_INVALID")
        for key, value in self.input_config_hashes.items():
            _hash(value, field=f"input_config_hashes.{key}")
        for field, value in (
            ("frozen_source_table_ref", self.frozen_source_table_ref),
            ("cost_accuracy_table_ref", self.cost_accuracy_table_ref),
            ("quadrature_decision_ref", self.quadrature_decision_ref),
            ("formal_plan_ref", self.formal_plan_ref),
            ("execution_evidence_ref", self.execution_evidence_ref),
            ("provenance_ref", self.provenance_ref),
            ("evaluation_ref", self.evaluation_ref),
            ("g3_6_ref", self.g3_6_ref),
            ("g3_7_ref", self.g3_7_ref),
        ):
            _ref(value, field=field)
        for field, value in (("recommendation_hash", self.recommendation_hash), ("finalization_hash", self.finalization_hash)):
            if value is not None:
                _hash(value, field=field)
        for field, value in (("recommendation_ref", self.recommendation_ref), ("finalization_ref", self.finalization_ref)):
            if value is not None:
                _ref(value, field=field)
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("STAGE3_G37_PUBLICATION_STATUS_INVALID")
        if type(self.formal_eligible) is not bool or self.formal_eligible != (self.status == "PASS"):
            raise FormalRunRejected("STAGE3_G37_PUBLICATION_ELIGIBILITY_MISMATCH")
        if not isinstance(self.gate_evaluation, Stage3GateEvaluation) or not isinstance(self.g3_6_gate, GateRecord) or not isinstance(self.g3_7_gate, GateRecord):
            raise TypeError("STAGE3_G37_PUBLICATION_NESTED_TYPES_INVALID")
        if self.gate_evaluation.artifact_hash != self.evaluation_hash or self.g3_6_gate.artifact_hash != self.g3_6_hash or self.g3_7_gate.artifact_hash != self.g3_7_hash:
            raise ValueError("STAGE3_G37_PUBLICATION_NESTED_HASH_MISMATCH")
        if self.g3_6_gate.gate_id != "stage3.G3-6" or self.g3_7_gate.gate_id != "stage3.G3-7":
            raise ValueError("STAGE3_G37_PUBLICATION_GATE_ID_INVALID")
        if self.g3_7_gate.status.value != self.status:
            raise FormalRunRejected("STAGE3_G37_PUBLICATION_GATE_STATUS_MISMATCH")
        if self.g3_7_ref in self.g3_7_gate.evidence_refs:
            raise FormalRunRejected("STAGE3_G37_GATE_SELF_BINDING")
        if self.recommendation is not None:
            if self.recommendation_hash is None or self.recommendation.get("artifact_hash") != self.recommendation_hash:
                raise ValueError("STAGE3_G37_RECOMMENDATION_HASH_MISMATCH")
            if self.recommendation_ref is None:
                raise ValueError("STAGE3_G37_RECOMMENDATION_REF_MISSING")
        elif self.recommendation_ref is not None or self.recommendation_hash is not None:
            raise ValueError("STAGE3_G37_RECOMMENDATION_BINDING_INCOMPLETE")
        if self.finalization is not None:
            if self.finalization_hash is None or self.finalization.get("artifact_hash") != self.finalization_hash:
                raise ValueError("STAGE3_G37_FINALIZATION_HASH_MISMATCH")
            if self.finalization_ref is None:
                raise ValueError("STAGE3_G37_FINALIZATION_REF_MISSING")
        elif self.finalization_ref is not None or self.finalization_hash is not None:
            raise ValueError("STAGE3_G37_FINALIZATION_BINDING_INCOMPLETE")
        refs = tuple(self.source_artifact_refs)
        if not refs or len(refs) != len(set(refs)) or any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("STAGE3_G37_PUBLICATION_SOURCE_REFS_INVALID")
        required = {
            self.frozen_source_table_ref, self.cost_accuracy_table_ref, self.quadrature_decision_ref,
            self.formal_plan_ref, self.execution_evidence_ref, self.provenance_ref,
            self.evaluation_ref, self.g3_6_ref, self.g3_7_ref,
        }
        if not required.issubset(set(refs)):
            raise FormalRunRejected("STAGE3_G37_PUBLICATION_OUTPUT_REFS_UNBOUND")
        if self.status == "PASS" and (self.recommendation is None or self.recommendation_ref is None or self.finalization is None or self.finalization_ref is None):
            raise FormalRunRejected("STAGE3_G37_PASS_OUTPUTS_MISSING")
        if self.recommendation_ref is not None and self.recommendation_ref in self.g3_7_gate.evidence_refs:
            raise FormalRunRejected("STAGE3_G37_GATE_FUTURE_RECOMMENDATION_BINDING")
        if self.finalization_ref is not None and self.finalization_ref in self.g3_7_gate.evidence_refs:
            raise FormalRunRejected("STAGE3_G37_GATE_FUTURE_FINALIZATION_BINDING")
        if self.status == "BLOCKED" and not self.reasons:
            raise FormalRunRejected("STAGE3_G37_BLOCKED_REASON_REQUIRED")

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "input_config_hash": self.input_config_hash,
            "input_config_hashes": dict(self.input_config_hashes),
            "publication_config_hash": self.publication_config_hash,
            "status": self.status,
            "scope": "formal",
            "formal_eligible": self.formal_eligible,
            "frozen_source_table_ref": self.frozen_source_table_ref,
            "frozen_source_table_hash": self.frozen_source_table_hash,
            "cost_accuracy_table_ref": self.cost_accuracy_table_ref,
            "cost_accuracy_table_hash": self.cost_accuracy_table_hash,
            "quadrature_decision_ref": self.quadrature_decision_ref,
            "quadrature_decision_hash": self.quadrature_decision_hash,
            "formal_plan_ref": self.formal_plan_ref,
            "formal_plan_hash": self.formal_plan_hash,
            "execution_evidence_ref": self.execution_evidence_ref,
            "execution_evidence_hash": self.execution_evidence_hash,
            "provenance_ref": self.provenance_ref,
            "provenance_hash": self.provenance_hash,
            "evaluation_ref": self.evaluation_ref,
            "evaluation_hash": self.evaluation_hash,
            "g3_6_ref": self.g3_6_ref,
            "g3_6_hash": self.g3_6_hash,
            "g3_7_ref": self.g3_7_ref,
            "g3_7_hash": self.g3_7_hash,
            "recommendation_ref": self.recommendation_ref,
            "recommendation_hash": self.recommendation_hash,
            "finalization_ref": self.finalization_ref,
            "finalization_hash": self.finalization_hash,
            "candidate_recommendation": thaw_json_value(dict(self.candidate_recommendation)),
            "recommendation": None if self.recommendation is None else thaw_json_value(dict(self.recommendation)),
            "finalization": None if self.finalization is None else thaw_json_value(dict(self.finalization)),
            "gate_evaluation": self.gate_evaluation.to_dict(),
            "g3_6_gate": self.g3_6_gate.to_dict(),
            "g3_7_gate": self.g3_7_gate.to_dict(),
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    @property
    def gate_ref(self) -> str:
        """Convenience alias for consumers that expose the current Gate."""

        return self.g3_7_ref

    @property
    def gate_hash(self) -> str:
        return self.g3_7_hash

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3G37Publication":
        required = set(cls.__dataclass_fields__) - {"artifact_hash"}
        # ``scope`` is a fixed wire discriminator rather than a mutable
        # dataclass field, just like GateRecord's formal scope.
        expected = required | {"scope", "artifact_hash"}
        if set(value) != expected:
            raise ValueError("STAGE3_G37_PUBLICATION_FIELDS_MISMATCH")
        if value.get("scope") != "formal":
            raise FormalRunRejected("STAGE3_G37_PUBLICATION_SCOPE_INVALID")
        supplied = value.get("artifact_hash")
        _hash(supplied, field="artifact_hash")
        if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != supplied:
            raise ValueError("STAGE3_G37_PUBLICATION_HASH_MISMATCH")
        for field in ("candidate_recommendation", "gate_evaluation", "g3_6_gate", "g3_7_gate"):
            if not isinstance(value.get(field), Mapping):
                raise TypeError(f"{field} must be an object")
        candidate = _as_candidate(dict(value["candidate_recommendation"]))
        _canonical(candidate, field="candidate_recommendation")
        for field in ("recommendation", "finalization"):
            nested = value.get(field)
            if isinstance(nested, Mapping):
                _canonical(nested, field=field)
        raw_eval = Stage3GateEvaluation.from_mapping(dict(value["gate_evaluation"]))
        g36 = GateRecord.from_mapping(dict(value["g3_6_gate"]))
        g37 = GateRecord.from_mapping(dict(value["g3_7_gate"]))
        source_refs = value["source_artifact_refs"]
        reasons = value["reasons"]
        if not isinstance(source_refs, list) or not isinstance(reasons, list) or not isinstance(value["input_config_hashes"], Mapping):
            raise TypeError("STAGE3_G37_PUBLICATION_ARRAYS_INVALID")
        return cls(
            publication_id=value["publication_id"], task_id=value["task_id"], config_hash=value["config_hash"], input_config_hash=value["input_config_hash"], input_config_hashes=dict(value["input_config_hashes"]), publication_config_hash=value["publication_config_hash"],
            status=value["status"], formal_eligible=value["formal_eligible"],
            frozen_source_table_ref=value["frozen_source_table_ref"], frozen_source_table_hash=value["frozen_source_table_hash"],
            cost_accuracy_table_ref=value["cost_accuracy_table_ref"], cost_accuracy_table_hash=value["cost_accuracy_table_hash"],
            quadrature_decision_ref=value["quadrature_decision_ref"], quadrature_decision_hash=value["quadrature_decision_hash"],
            formal_plan_ref=value["formal_plan_ref"], formal_plan_hash=value["formal_plan_hash"],
            execution_evidence_ref=value["execution_evidence_ref"], execution_evidence_hash=value["execution_evidence_hash"],
            provenance_ref=value["provenance_ref"], provenance_hash=value["provenance_hash"],
            evaluation_ref=value["evaluation_ref"], evaluation_hash=value["evaluation_hash"],
            g3_6_ref=value["g3_6_ref"], g3_6_hash=value["g3_6_hash"], g3_7_ref=value["g3_7_ref"], g3_7_hash=value["g3_7_hash"],
            recommendation_ref=value["recommendation_ref"], recommendation_hash=value["recommendation_hash"],
            finalization_ref=value["finalization_ref"], finalization_hash=value["finalization_hash"],
            candidate_recommendation=value["candidate_recommendation"], recommendation=value["recommendation"], finalization=value["finalization"],
            gate_evaluation=raw_eval, g3_6_gate=g36, g3_7_gate=g37,
            source_artifact_refs=tuple(source_refs), reasons=tuple(reasons), schema_version=value["schema_version"],
        )  # type: ignore[arg-type]


class Stage3G37Publisher:
    """Reload formal inputs, commit G3-7, then publish the formal handoff."""

    def publish(
        self,
        *,
        workspace_root: str | Path,
        output_dir: str,
        frozen_source_table_ref: str,
        cost_accuracy_table_ref: str,
        quadrature_decision_ref: str,
        formal_plan_ref: str,
        execution_evidence_ref: str,
        provenance_ref: str,
        evaluation_ref: str | None = None,
        g3_6_ref: str | None = None,
        g3_6_evaluation_ref: str | None = None,
        g3_6_gate_ref: str | None = None,
        stage3_scope_decision_ref: str | None = None,
        stage3_scope_gate_ref: str | None = None,
        config_hash: str | None = None,
        publication_id: str | None = None,
        task_id: str = STAGE3_G37_TASK_ID,
    ) -> Stage3G37Publication:
        root = Path(workspace_root).resolve()
        _identifier(task_id, field="task_id")
        # Compatibility aliases are accepted, but never used as authority.
        evaluation_ref = evaluation_ref or g3_6_evaluation_ref
        g3_6_ref = g3_6_ref or g3_6_gate_ref
        if not isinstance(evaluation_ref, str) or not isinstance(g3_6_ref, str):
            raise FormalRunRejected("STAGE3_G37_G3_6_EVALUATION_AND_GATE_REFS_REQUIRED")

        loaded = {
            "frozen_source_table": _load_formal(root, frozen_source_table_ref, field="frozen_source_table", kinds=_TABLE_KINDS),
            "cost_accuracy_table": _load_formal(root, cost_accuracy_table_ref, field="cost_accuracy_table", kinds=_COST_KINDS),
            "quadrature_decision": _load_formal(root, quadrature_decision_ref, field="quadrature_decision", kinds=_CANDIDATE_KINDS),
            "formal_plan": _load_formal(root, formal_plan_ref, field="formal_plan", kinds=_PLAN_KINDS),
            "execution": _load_formal(root, execution_evidence_ref, field="execution_evidence", kinds=_EXECUTION_KINDS),
            "provenance": _load_formal(root, provenance_ref, field="provenance", kinds=_PROVENANCE_KINDS),
            "evaluation": _load_formal(root, evaluation_ref, field="g3_6_evaluation", kinds=_EVALUATION_KINDS),
            "g3_6": _load_formal(root, g3_6_ref, field="g3_6", kinds=_GATE_KINDS),
        }
        # Scope authorities are immutable producers in their own right. Load
        # the exact refs committed by G3-6 so their real config identities are
        # included in the publication binding instead of being relabelled as
        # the S3.08/G3-6 producer config.
        scope_ref_from_eval = loaded["evaluation"].payload.get(
            "stage3_scope_decision_ref"
        )
        scope_gate_ref_from_eval = loaded["evaluation"].payload.get(
            "stage3_scope_gate_ref"
        )
        if isinstance(scope_ref_from_eval, str):
            loaded["scope_decision"] = _load_formal(
                root,
                scope_ref_from_eval,
                field="scope_decision",
                kinds=frozenset({"scope_authority", "stage3_scope_authority"}),
            )
        if isinstance(scope_gate_ref_from_eval, str):
            loaded["scope_gate"] = _load_formal(
                root,
                scope_gate_ref_from_eval,
                field="scope_gate",
                kinds=_GATE_KINDS,
            )
        input_config_hashes = {
            name: item.identity.config_hash for name, item in loaded.items()
        }
        g36_authority_names = (
            "frozen_source_table",
            "formal_plan",
            "provenance",
            "evaluation",
            "g3_6",
        )
        g36_config_values = {
            input_config_hashes[name] for name in g36_authority_names
        }
        if len(g36_config_values) != 1:
            raise FormalRunRejected("STAGE3_G37_G3_6_AUTHORITY_CONFIG_HASH_MISMATCH")
        if (
            input_config_hashes["cost_accuracy_table"]
            != input_config_hashes["quadrature_decision"]
        ):
            raise FormalRunRejected("STAGE3_G37_STAGE309_CONFIG_HASH_MISMATCH")
        # Different canonical tasks have different producer configurations.
        # Preserve every immutable producer identity, and collapse the set
        # only by hashing the complete mapping instead of pretending that
        # S3.08 and S3.09 share one task configuration.
        input_config_hash = canonical_json_hash(
            {
                "schema_version": "stage3-g37-input-config-set-v1",
                "input_config_hashes": input_config_hashes,
            }
        )
        _hash(input_config_hash, field="input_config_hash")
        # ``config_hash`` is an untrusted compatibility argument.  The output
        # binding is derived from immutable commit identities below.
        input_binding = {
            name: {
                "commit_ref": item.identity.commit_ref,
                "artifact_hash": item.identity.artifact_hash,
                "config_hash": item.identity.config_hash,
            }
            for name, item in loaded.items()
        }
        publication_config_hash = canonical_json_hash({
            "schema_version": "stage3-g37-publication-config-v1",
            "input_config_hash": input_config_hash,
            "input_config_hashes": input_config_hashes,
            "inputs": input_binding,
        })
        if config_hash is not None:
            _hash(config_hash, field="config_hash")
        normalized_output_dir = output_dir.replace("\\", "/").rstrip("/")
        future_refs = {
            f"{normalized_output_dir}/commits/{kind}.json"
            for kind in (
                STAGE3_G37_GATE_ARTIFACT_KIND,
                STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
                STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
                STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
            )
        }
        if future_refs.intersection(set(loaded["quadrature_decision"].source_refs)):
            raise FormalRunRejected("STAGE3_G37_CANDIDATE_FUTURE_OUTPUT_BINDING")

        try:
            frozen = FrozenSourceTable.from_mapping(dict(loaded["frozen_source_table"].payload))
            cost = FrozenSourceTable.from_mapping(dict(loaded["cost_accuracy_table"].payload))
            if not frozen.frozen or not cost.frozen:
                raise FormalRunRejected("STAGE3_G37_TABLES_MUST_BE_FROZEN")
            if any(
                marker in f"{table.name} {table.schema_version}".casefold()
                for table in (frozen, cost)
                for marker in ("fixture", "synthetic")
            ):
                raise FormalRunRejected("STAGE3_G37_TABLE_FORBIDDEN_LABEL")
            execution = FormalExecutionEvidence.from_mapping(dict(loaded["execution"].payload))
            prereq = _check_execution(execution)
            provenance = ProvenanceRecord.from_mapping(dict(loaded["provenance"].payload))
            if provenance.scope != "formal" or provenance.status is not ProvenanceStatus.COMPLETED or provenance.formal_eligible is not True or provenance.worktree_clean is not True:
                raise FormalRunRejected("STAGE3_G37_PROVENANCE_NOT_COMPLETED_CLEAN")
            plan, plan_hash, units, rules, strata, thresholds = _formal_plan(dict(loaded["formal_plan"].payload), execution)
            frozen_rows, cost_rows = _check_row_equality(frozen, cost, units=units, rules=rules)
            candidate_wire = _as_candidate(dict(loaded["quadrature_decision"].payload))
            _canonical(candidate_wire, field="quadrature_decision")
            candidate = QuadratureRecommendation.from_mapping(candidate_wire)
            evaluation = Stage3GateEvaluation.from_mapping(dict(loaded["evaluation"].payload))
            g36 = GateRecord.from_mapping(dict(loaded["g3_6"].payload))
            if g36.gate_id != "stage3.G3-6":
                raise FormalRunRejected("STAGE3_G37_G3_6_GATE_ID_INVALID")
            gate_block_reasons: list[str] = []
            if g36.status is not GateStatus.PASS or g36.effective_status() is not GateStatus.PASS:
                gate_block_reasons.append("G3_6_GATE_NOT_LIVE_PASS")
            # The task envelope hash is intentionally distinct from the
            # domain payload hash carried by GateRecord/Stage3GateEvaluation;
            # compare the parsed payloads above, never conflate the two.
            if evaluation.provenance_hash != provenance.artifact_hash:
                raise FormalRunRejected("STAGE3_G37_EVALUATION_PROVENANCE_MISMATCH")
            if provenance_ref in evaluation.source_artifact_refs or provenance_ref in provenance.artifact_refs:
                raise FormalRunRejected("STAGE3_G37_PROVENANCE_SELF_BINDING")
            if not set(evaluation.source_artifact_refs).issubset(set(provenance.artifact_refs)):
                raise FormalRunRejected("STAGE3_G37_PROVENANCE_SOURCE_COVERAGE_MISSING")
            for ref in (frozen_source_table_ref, formal_plan_ref, evaluation_ref, provenance_ref):
                if ref not in g36.evidence_refs:
                    raise FormalRunRejected(f"STAGE3_G37_G3_6_EVIDENCE_UNBOUND:{ref}")
            if evaluation.execution_evidence_hash != execution.artifact_hash or evaluation.formal_plan_ref != formal_plan_ref or evaluation.formal_plan_hash != plan_hash:
                raise FormalRunRejected("STAGE3_G37_EVALUATION_INPUT_BINDING_MISMATCH")
            expected_gate_hashes = tuple(
                next(gate for gate in prereq if gate.gate_id == gate_id).artifact_hash
                for gate_id in REQUIRED_STAGE3_GATE_IDS
            )
            if evaluation.gate_hashes != expected_gate_hashes:
                raise FormalRunRejected("STAGE3_G37_EVALUATION_GATE_HASHES_MISMATCH")
            if evaluation.required_rule_names != rules or evaluation.required_unit_ids != units:
                raise FormalRunRejected("STAGE3_G37_EVALUATION_PLAN_COVERAGE_MISMATCH")
            if (
                not isinstance(g36.measured, Mapping)
                or g36.measured.get("source_table_hash") != frozen.content_hash
                or g36.measured.get("evaluation_hash") != evaluation.artifact_hash
            ):
                gate_block_reasons.append("G3_6_MEASURED_BINDING_MISMATCH")
            scope_decision_ref, scope_gate_ref = _scope_from_evaluation(
                root, evaluation, execution, decision_ref=stage3_scope_decision_ref, gate_ref=stage3_scope_gate_ref,
            )
            reasons = list(gate_block_reasons)
            reasons.extend(_candidate_consistency(candidate, evaluation, execution=execution, plan_hash=plan_hash, units=units, rules=rules, thresholds=thresholds))
            passing = tuple(candidate.passing_rules)
            if not passing:
                reasons.append("NO_PASSING_RULE")
            if candidate.default_rule is None or candidate.default_rule not in passing:
                reasons.append("UNIQUE_DEFAULT_MISSING")
            if candidate.fallback_rule is None or candidate.fallback_rule not in passing or candidate.fallback_rule == candidate.default_rule:
                reasons.append("EXPLICIT_FALLBACK_MISSING")
            expected_order = tuple(sorted(passing, key=lambda rule: _cost_key(cost_rows, rule, units))) if passing else ()
            if expected_order and candidate.default_rule != expected_order[0]:
                reasons.append("DEFAULT_RULE_NOT_COST_MINIMUM")
            reasons = list(_reason_tuple(reasons))
        except (OSError, TypeError, ValueError, FormalRunRejected) as error:
            raise FormalRunRejected(f"STAGE3_G37_INPUTS_INVALID:{error}") from error

        publication_id = publication_id or f"stage3-g37-{evaluation.artifact_hash[:16]}"
        _identifier(publication_id, field="publication_id")
        checked_at = provenance.ended_at or provenance.started_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        gate_status = GateStatus.PASS if not reasons else GateStatus.BLOCKED
        unique_default = bool(
            candidate.default_rule
            and candidate.default_rule in candidate.passing_rules
            and len(set(candidate.passing_rules)) == len(candidate.passing_rules)
        )
        explicit_fallback = bool(
            candidate.fallback_rule
            and candidate.fallback_rule in candidate.passing_rules
            and candidate.fallback_rule != candidate.default_rule
        )
        g37 = GateRecord(
            gate_id="stage3.G3-7",
            stage=3,
            status=gate_status,
            checked_at=checked_at,
            measured=_g37_measured(candidate, cost_rows, units=units, evaluation=evaluation, g36=g36, frozen=frozen, cost=cost),
            threshold={
                "candidate_status": "FORMAL_CANDIDATE",
                "cost_rows_equal_frozen_rows": True,
                "g3_6_status": g36.status.value,
                "evaluation_status": evaluation.status,
                "unique_default": unique_default,
                "explicit_fallback": explicit_fallback,
            },
            # Deliberately no g37/recommendation/finalization/receipt ref: all
            # are future outputs at this point.
            evidence_refs=tuple(dict.fromkeys((
                quadrature_decision_ref, cost_accuracy_table_ref, frozen_source_table_ref,
                evaluation_ref, g3_6_ref, formal_plan_ref, execution_evidence_ref,
                provenance_ref, scope_decision_ref, scope_gate_ref,
            ))),
            reasons=tuple(reasons),
        )
        store = TaskArtifactStore(root, output_dir)
        g37_artifact = store.publish(
            task_id=task_id, artifact_kind=STAGE3_G37_GATE_ARTIFACT_KIND,
            config_hash=publication_config_hash, run_intent="formal",
            # A formal task envelope records where the artifact was produced;
            # the payload's Gate status carries scientific eligibility.
            payload=g37.to_dict(), formal_eligible=True,
            source_refs=g37.evidence_refs,
        )

        recommendation: QuadratureRecommendation | None = None
        recommendation_artifact = None
        finalization: Stage3Finalization | None = None
        finalization_artifact = None
        if gate_status is GateStatus.PASS:
            try:
                recommendation = QuadratureRecommendation.from_mapping_authorized(
                    candidate_wire, execution=execution, gate=g37,
                    # The Gate binds the already-committed Stage3.09
                    # candidate, not its own future commit (or this output).
                    artifact_ref=quadrature_decision_ref, gate_evaluation=evaluation,
                    provenance=provenance,
                )
                recommendation_artifact = store.publish(
                    task_id=task_id, artifact_kind=STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND,
                    config_hash=publication_config_hash, run_intent="formal",
                    payload=recommendation.to_dict(), formal_eligible=True,
                    source_refs=tuple(dict.fromkeys((*g37.evidence_refs, g37_artifact.commit_ref))),
                )
                selection = {
                    "schema_version": "stage3-formal-method-selection-v1",
                    "status": "QUALIFIED", "scope": "formal", "path_type": "full_update",
                    "default_rule": recommendation.default_rule, "fallback_rule": recommendation.fallback_rule,
                    "passing_rules": list(recommendation.passing_rules), "required_unit_ids": list(recommendation.required_unit_ids),
                    "thresholds_hash": recommendation.thresholds.artifact_hash,
                    "execution_evidence_hash": execution.artifact_hash,
                    "frozen_table_ref": frozen_source_table_ref, "frozen_table_hash": frozen.content_hash,
                    "formal_plan_ref": formal_plan_ref, "formal_plan_hash": plan_hash,
                    "evaluation_ref": evaluation_ref, "evaluation_hash": evaluation.artifact_hash,
                    "provenance_ref": provenance_ref, "provenance_hash": provenance.artifact_hash,
                    "g3_6_gate_hash": g36.artifact_hash, "g3_7_gate_hash": g37.artifact_hash,
                }
                finalization = Stage3Finalization(
                    finalization_id=f"stage3-final-{candidate.artifact_hash[:16]}", status="PASS", scope="formal", formal_eligible=True,
                    execution_evidence_hash=execution.artifact_hash,
                    frozen_table_ref=frozen_source_table_ref, frozen_table_hash=frozen.content_hash,
                    formal_plan_ref=formal_plan_ref, formal_plan_hash=plan_hash,
                    evaluation_ref=evaluation_ref, evaluation_hash=evaluation.artifact_hash,
                    provenance_ref=provenance_ref, provenance_hash=provenance.artifact_hash,
                    recommendation_ref=recommendation_artifact.commit_ref,
                    g3_6_ref=g3_6_ref, g3_7_ref=g37_artifact.commit_ref,
                    recommendation=recommendation, candidate_recommendation=candidate_wire,
                    gate_evaluation=evaluation, prerequisite_gates=prereq,
                    g3_6_gate=g36, g3_7_gate=g37, method_selection=selection,
                    source_artifact_refs=tuple(dict.fromkeys((*g37.evidence_refs, g37_artifact.commit_ref, recommendation_artifact.commit_ref))),
                    checked_at=checked_at,
                )
                finalization_artifact = store.publish(
                    task_id=task_id, artifact_kind=STAGE3_G37_FINALIZATION_ARTIFACT_KIND,
                    config_hash=publication_config_hash, run_intent="formal",
                    payload=finalization.to_dict(), formal_eligible=True,
                    source_refs=finalization.source_artifact_refs,
                )
            except (TypeError, ValueError, FormalRunRejected) as error:
                # G3-7 was already committed.  Never turn a failed qualification
                # into PASS and never replace the committed Gate with a second one.
                raise FormalRunRejected(f"STAGE3_G37_QUALIFICATION_FAILED:{error}") from error

        source_refs = tuple(dict.fromkeys((
            *g37.evidence_refs, g37_artifact.commit_ref,
            *((recommendation_artifact.commit_ref,) if recommendation_artifact is not None else ()),
            *((finalization_artifact.commit_ref,) if finalization_artifact is not None else ()),
        )))
        receipt = Stage3G37Publication(
            publication_id=publication_id, task_id=task_id, config_hash=publication_config_hash,
            input_config_hash=input_config_hash, status=gate_status.value,
            input_config_hashes=input_config_hashes,
            publication_config_hash=publication_config_hash,
            formal_eligible=gate_status is GateStatus.PASS,
            frozen_source_table_ref=frozen_source_table_ref, frozen_source_table_hash=frozen.content_hash,
            cost_accuracy_table_ref=cost_accuracy_table_ref, cost_accuracy_table_hash=cost.content_hash,
            quadrature_decision_ref=quadrature_decision_ref, quadrature_decision_hash=candidate.artifact_hash,
            formal_plan_ref=formal_plan_ref, formal_plan_hash=plan_hash,
            execution_evidence_ref=execution_evidence_ref, execution_evidence_hash=execution.artifact_hash,
            provenance_ref=provenance_ref, provenance_hash=provenance.artifact_hash,
            evaluation_ref=evaluation_ref, evaluation_hash=evaluation.artifact_hash,
            g3_6_ref=g3_6_ref, g3_6_hash=g36.artifact_hash,
            g3_7_ref=g37_artifact.commit_ref, g3_7_hash=g37.artifact_hash,
            recommendation_ref=None if recommendation_artifact is None else recommendation_artifact.commit_ref,
            recommendation_hash=None if recommendation is None else recommendation.artifact_hash,
            finalization_ref=None if finalization_artifact is None else finalization_artifact.commit_ref,
            finalization_hash=None if finalization is None else finalization.artifact_hash,
            candidate_recommendation=candidate_wire,
            recommendation=None if recommendation is None else recommendation.to_dict(),
            finalization=None if finalization is None else finalization.to_dict(),
            gate_evaluation=evaluation, g3_6_gate=g36, g3_7_gate=g37,
            source_artifact_refs=source_refs, reasons=tuple(reasons),
        )
        store.publish(
            task_id=task_id, artifact_kind=STAGE3_G37_PUBLICATION_ARTIFACT_KIND,
            config_hash=publication_config_hash, run_intent="formal",
            payload=receipt.to_dict(), formal_eligible=True,
            source_refs=receipt.source_artifact_refs,
        )
        return receipt


def publish_stage3_g37(**kwargs: object) -> Stage3G37Publication:
    """Functional wrapper around :class:`Stage3G37Publisher`."""

    return Stage3G37Publisher().publish(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "STAGE3_G37_FINALIZATION_ARTIFACT_KIND",
    "STAGE3_G37_GATE_ARTIFACT_KIND",
    "STAGE3_G37_PUBLICATION_ARTIFACT_KIND",
    "STAGE3_G37_PUBLICATION_SCHEMA",
    "STAGE3_G37_RECOMMENDATION_ARTIFACT_KIND",
    "STAGE3_G37_TASK_ID",
    "Stage3G37Publication",
    "Stage3G37Publisher",
    "publish_stage3_g37",
]
