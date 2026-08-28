"""Publish the independent Stage 3 G3-6 decision boundary.

Stage 3.08 produces the frozen observation table.  This module is the small
post-Stage-3.08 handoff that *reads* the already committed inputs, runs the
independent :class:`Stage3GateEvaluator`, and publishes its evaluation before
publishing G3-6.  In particular, provenance is completed before this module
runs and is never amended with the two artifacts produced here.  That keeps
the provenance/evaluation/Gate dependency acyclic.

The function deliberately accepts commit references rather than arbitrary
objects or paths.  Every input is reloaded through ``load_committed_task_artifact``
with ``require_formal=True``; a local file, an object without a commit, or a
fixture envelope cannot enter the formal decision.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import math
import re

from ..analysis.report import FrozenSourceTable
from ..contracts.errors import FormalRunRejected
from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import JSONValue, canonical_json_hash
from ..contracts.provenance import ProvenanceRecord, ProvenanceStatus
from ..contracts.stage23 import FormalExecutionEvidence
from ..contracts.stage3_scope import (
    validate_stage3_scope_authority,
    validate_stage3_scope_decision,
)
from ..contracts.status import GateRecord, GateStatus
from ..runtime.task_artifacts import (
    LoadedTaskArtifact,
    TaskArtifactStore,
    load_committed_task_artifact,
)
from .stage3_gate import Stage3GateEvaluation, Stage3GateEvaluator


STAGE3_G36_PUBLICATION_SCHEMA = "stage3-g36-publication-v1"
STAGE3_G36_TASK_ID = "stage3.08_g3_6_publisher"
STAGE3_G36_EVALUATION_ARTIFACT_KIND = "gate_evaluation"
STAGE3_G36_GATE_ARTIFACT_KIND = "gate_record"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[^\\?]+$")
_TABLE_KINDS = frozenset({"frozen_source_table"})
_PLAN_KINDS = frozenset({"formal_plan", "stage3_formal_plan"})
_PROVENANCE_KINDS = frozenset({"provenance_record", "provenance"})
_EXECUTION_KINDS = frozenset({"formal_execution_evidence", "execution_evidence"})
_SCOPE_KINDS = frozenset({"scope_authority", "stage3_scope_authority"})


def _hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _ref(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or not _REF_RE.fullmatch(value)
        or "://" in value
    ):
        raise ValueError(f"{field} 必须是稳定 commit ref")
    lowered = value.casefold()
    if "fixture" in lowered or "synthetic" in lowered:
        raise FormalRunRejected(f"{field} 禁止 fixture/synthetic ref")
    return value


def _safe_task_id(value: object) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 160
        or not value[0].isalnum()
        or any(not (char.isalnum() or char in "._-") for char in value)
    ):
        raise ValueError("task_id 不是安全标识")
    return value


def _load_formal_commit(
    root: Path,
    reference: str,
    *,
    field: str,
    kinds: frozenset[str],
) -> LoadedTaskArtifact:
    ref = _ref(reference, field=field)
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_COMMIT_INVALID") from error
    if loaded.identity.artifact_kind not in kinds:
        raise FormalRunRejected(
            f"STAGE3_G36_{field.upper()}_ARTIFACT_KIND_INVALID:{loaded.identity.artifact_kind}"
        )
    if loaded.run_intent != "formal" or loaded.identity.formal_eligible is not True:
        raise FormalRunRejected(f"STAGE3_G36_{field.upper()}_FORMAL_COMMIT_REQUIRED")
    return loaded


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} 必须是 object")
    return value


def _source_refs_from_rows(table: FrozenSourceTable) -> tuple[str, ...]:
    """Collect every row evidence ref that the evaluator can inspect.

    Both names are collected because older Stage 3 observation rows used
    ``source_artifact_refs`` while the formal table contract uses
    ``evidence_refs``.  The evaluator itself remains the authority for whether
    either representation is present and bound.
    """

    refs: list[str] = []
    for index, raw in enumerate(table.rows):
        if not isinstance(raw, Mapping):
            raise TypeError(f"frozen_source_table.rows[{index}] 必须是 object")
        for field in ("evidence_refs", "source_artifact_refs"):
            value = raw.get(field)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                for item in value:
                    if not isinstance(item, str) or not item:
                        raise ValueError(f"frozen_source_table.rows[{index}].{field} 无效")
                    _ref(item, field=f"frozen_source_table.rows[{index}].{field}")
                    refs.append(item)
    return tuple(dict.fromkeys(refs))


def _finite_observation_table(table: FrozenSourceTable) -> bool:
    """Mirror the finite scalar part of the Stage3.08 G3-6 measured fact."""

    fields = (
        "normalized_l1_error",
        "normalized_l2_error",
        "normalized_linf_error",
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
        "active_spearman",
        "cosine_similarity",
        "sign_consistency",
        "layer_quality_tv",
        "module_quality_tv",
        "reference_normalized_l1_error",
    )
    for raw in table.rows:
        if not isinstance(raw, Mapping):
            return False
        for field in fields:
            value = raw.get(field)
            if isinstance(value, Mapping):
                if value.get("defined") is not True or "value" not in value:
                    return False
                value = value["value"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if not math.isfinite(float(value)):
                return False
    return True


def _plan_shape(plan: Mapping[str, object], execution: FormalExecutionEvidence) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Require the exact evaluator plan identity before publishing anything."""

    declared = plan.get("artifact_hash")
    if not isinstance(declared, str) or canonical_json_hash(
        {key: item for key, item in plan.items() if key != "artifact_hash"}
    ) != declared:
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_HASH_INVALID")
    if (
        plan.get("schema_version") != "stage3-formal-pilot-plan-v1"
        or plan.get("scope") != "formal"
        or plan.get("formal_eligible") is not True
        or plan.get("execution_evidence_hash") != execution.artifact_hash
    ):
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_BINDING_INVALID")
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
        raise FormalRunRejected("STAGE3_G36_FORMAL_PLAN_COVERAGE_INVALID")
    return tuple(str(item) for item in units), tuple(str(item) for item in rules)


def _scope_authority(
    *,
    decision_payload: Mapping[str, object],
    gate_payload: Mapping[str, object],
    decision_ref: str,
    execution: FormalExecutionEvidence,
) -> tuple[Mapping[str, object], GateRecord]:
    validate_stage3_scope_decision(decision_payload)
    try:
        gate = GateRecord.from_mapping(dict(gate_payload))
        validate_stage3_scope_authority(decision_payload, gate, decision_ref=decision_ref)
    except (TypeError, ValueError) as error:
        raise FormalRunRejected("STAGE3_G36_SCOPE_AUTHORITY_INVALID") from error
    if gate.status is not GateStatus.PASS or gate.effective_status() is not GateStatus.PASS:
        raise FormalRunRejected("STAGE3_G36_SCOPE_GATE_NOT_LIVE_PASS")
    execution_gate = next(
        (item for item in execution.prerequisite_gates if item.gate_id == "stage3.G3-0"),
        None,
    )
    if execution_gate is None or execution_gate.artifact_hash != gate.artifact_hash:
        raise FormalRunRejected("STAGE3_G36_SCOPE_GATE_EXECUTION_BINDING_MISMATCH")
    return decision_payload, gate


def _evaluation_sources(
    *,
    frozen_ref: str,
    execution_ref: str,
    plan_ref: str,
    decision_ref: str,
    scope_gate_ref: str,
    execution: FormalExecutionEvidence,
    table: FrozenSourceTable,
) -> tuple[str, ...]:
    refs: list[str] = [
        frozen_ref,
        execution_ref,
        plan_ref,
        decision_ref,
        scope_gate_ref,
    ]
    for gate in execution.prerequisite_gates:
        refs.extend(gate.evidence_refs)
    refs.extend(_source_refs_from_rows(table))
    normalized = tuple(dict.fromkeys(refs))
    for index, ref in enumerate(normalized):
        _ref(ref, field=f"evaluator_source_artifact_refs[{index}]")
    return normalized


def _g36_measured(
    *,
    table: FrozenSourceTable,
    evaluation: Stage3GateEvaluation,
    units: tuple[str, ...],
    rules: tuple[str, ...],
    all_rows_finite: bool,
) -> dict[str, JSONValue]:
    return {
        "source_table_hash": table.content_hash,
        "evaluation_hash": evaluation.artifact_hash,
        "observation_count": len(table.rows),
        "expected_observation_count": len(units) * len(rules),
        "unit_count": len(units),
        "rule_count": len(rules),
        "all_rows_finite": all_rows_finite,
        "passing_rules": [
            name
            for name, raw in evaluation.rule_evaluations.items()
            if isinstance(raw, Mapping) and raw.get("passing") is True
        ],
    }


@dataclass(frozen=True, slots=True)
class Stage3G36Publication:
    """Immutable receipt for the two formal artifacts published by G3-6."""

    publication_id: str
    task_id: str
    config_hash: str
    status: str
    formal_eligible: bool
    frozen_source_table_ref: str
    frozen_source_table_hash: str
    formal_plan_ref: str
    formal_plan_hash: str
    execution_evidence_ref: str
    execution_evidence_hash: str
    provenance_ref: str
    provenance_hash: str
    stage3_scope_decision_ref: str
    stage3_scope_decision_hash: str
    stage3_scope_gate_ref: str
    stage3_scope_gate_hash: str
    evaluation_ref: str
    evaluation_hash: str
    g3_6_ref: str
    g3_6_hash: str
    gate_evaluation: Stage3GateEvaluation
    g3_6_gate: GateRecord
    source_artifact_refs: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    schema_version: str = STAGE3_G36_PUBLICATION_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != STAGE3_G36_PUBLICATION_SCHEMA:
            raise ValueError("STAGE3_G36_PUBLICATION_SCHEMA_UNSUPPORTED")
        _safe_task_id(self.task_id)
        if not isinstance(self.publication_id, str) or not self.publication_id:
            raise ValueError("STAGE3_G36_PUBLICATION_ID_INVALID")
        _hash(self.config_hash, field="config_hash")
        if self.status not in {"PASS", "BLOCKED"}:
            raise ValueError("STAGE3_G36_PUBLICATION_STATUS_INVALID")
        if type(self.formal_eligible) is not bool or self.formal_eligible != (self.status == "PASS"):
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_ELIGIBILITY_MISMATCH")
        for field, value in (
            ("frozen_source_table_ref", self.frozen_source_table_ref),
            ("formal_plan_ref", self.formal_plan_ref),
            ("execution_evidence_ref", self.execution_evidence_ref),
            ("provenance_ref", self.provenance_ref),
            ("stage3_scope_decision_ref", self.stage3_scope_decision_ref),
            ("stage3_scope_gate_ref", self.stage3_scope_gate_ref),
            ("evaluation_ref", self.evaluation_ref),
            ("g3_6_ref", self.g3_6_ref),
        ):
            _ref(value, field=field)
        for field, value in (
            ("frozen_source_table_hash", self.frozen_source_table_hash),
            ("formal_plan_hash", self.formal_plan_hash),
            ("execution_evidence_hash", self.execution_evidence_hash),
            ("provenance_hash", self.provenance_hash),
            ("stage3_scope_decision_hash", self.stage3_scope_decision_hash),
            ("stage3_scope_gate_hash", self.stage3_scope_gate_hash),
            ("evaluation_hash", self.evaluation_hash),
            ("g3_6_hash", self.g3_6_hash),
        ):
            _hash(value, field=field)
        if not isinstance(self.gate_evaluation, Stage3GateEvaluation) or not isinstance(self.g3_6_gate, GateRecord):
            raise TypeError("STAGE3_G36_PUBLICATION_NESTED_TYPES_INVALID")
        if self.gate_evaluation.artifact_hash != self.evaluation_hash or self.g3_6_gate.artifact_hash != self.g3_6_hash:
            raise ValueError("STAGE3_G36_PUBLICATION_NESTED_HASH_MISMATCH")
        if self.gate_evaluation.status != self.status or self.g3_6_gate.status.value != self.status:
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_STATUS_BINDING_MISMATCH")
        if self.g3_6_gate.gate_id != "stage3.G3-6":
            raise ValueError("STAGE3_G36_PUBLICATION_GATE_ID_INVALID")
        refs = tuple(self.source_artifact_refs)
        if not refs or len(set(refs)) != len(refs) or any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("STAGE3_G36_PUBLICATION_SOURCE_REFS_INVALID")
        object.__setattr__(self, "source_artifact_refs", refs)
        if any(not isinstance(item, str) or not item for item in self.reasons):
            raise ValueError("STAGE3_G36_PUBLICATION_REASONS_INVALID")
        if self.status == "PASS" and self.reasons:
            raise FormalRunRejected("STAGE3_G36_PUBLICATION_PASS_HAS_REASONS")
        required_refs = {
            self.frozen_source_table_ref,
            self.formal_plan_ref,
            self.execution_evidence_ref,
            self.provenance_ref,
            self.stage3_scope_decision_ref,
            self.stage3_scope_gate_ref,
            self.evaluation_ref,
            self.g3_6_ref,
        }
        if not required_refs.issubset(set(refs)):
            raise ValueError("STAGE3_G36_PUBLICATION_OUTPUT_REFS_UNBOUND")

    @property
    def gate_ref(self) -> str:
        """Alias used by consumers that call the G3-6 output simply ``gate``."""

        return self.g3_6_ref

    @property
    def gate_hash(self) -> str:
        return self.g3_6_hash

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "publication_id": self.publication_id,
            "task_id": self.task_id,
            "config_hash": self.config_hash,
            "status": self.status,
            "scope": "formal",
            "formal_eligible": self.formal_eligible,
            "frozen_source_table_ref": self.frozen_source_table_ref,
            "frozen_source_table_hash": self.frozen_source_table_hash,
            "formal_plan_ref": self.formal_plan_ref,
            "formal_plan_hash": self.formal_plan_hash,
            "execution_evidence_ref": self.execution_evidence_ref,
            "execution_evidence_hash": self.execution_evidence_hash,
            "provenance_ref": self.provenance_ref,
            "provenance_hash": self.provenance_hash,
            "stage3_scope_decision_ref": self.stage3_scope_decision_ref,
            "stage3_scope_decision_hash": self.stage3_scope_decision_hash,
            "stage3_scope_gate_ref": self.stage3_scope_gate_ref,
            "stage3_scope_gate_hash": self.stage3_scope_gate_hash,
            "evaluation_ref": self.evaluation_ref,
            "evaluation_hash": self.evaluation_hash,
            "g3_6_ref": self.g3_6_ref,
            "g3_6_hash": self.g3_6_hash,
            "gate_evaluation": self.gate_evaluation.to_dict(),
            "g3_6_gate": self.g3_6_gate.to_dict(),
            "source_artifact_refs": list(self.source_artifact_refs),
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "Stage3G36Publication":
        # The wire fields are intentionally enumerated so a future dataclass
        # implementation detail cannot silently become part of the contract.
        required = {
            "schema_version", "publication_id", "task_id", "config_hash", "status", "scope", "formal_eligible",
            "frozen_source_table_ref", "frozen_source_table_hash", "formal_plan_ref", "formal_plan_hash",
            "execution_evidence_ref", "execution_evidence_hash", "provenance_ref", "provenance_hash",
            "stage3_scope_decision_ref", "stage3_scope_decision_hash", "stage3_scope_gate_ref", "stage3_scope_gate_hash",
            "evaluation_ref", "evaluation_hash", "g3_6_ref", "g3_6_hash", "gate_evaluation", "g3_6_gate",
            "source_artifact_refs", "reasons", "artifact_hash",
        }
        if set(value) != required:
            raise ValueError("STAGE3_G36_PUBLICATION_FIELDS_MISMATCH")
        supplied = value["artifact_hash"]
        _hash(supplied, field="artifact_hash")
        if value["scope"] != "formal":
            raise ValueError("STAGE3_G36_PUBLICATION_SCOPE_INVALID")
        if canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}) != supplied:
            raise ValueError("STAGE3_G36_PUBLICATION_HASH_MISMATCH")
        raw_eval = value["gate_evaluation"]
        raw_gate = value["g3_6_gate"]
        if not isinstance(raw_eval, Mapping) or not isinstance(raw_gate, Mapping):
            raise TypeError("STAGE3_G36_PUBLICATION_NESTED_OBJECTS_REQUIRED")
        arrays = ("source_artifact_refs", "reasons")
        if any(not isinstance(value[name], list) for name in arrays):
            raise TypeError("STAGE3_G36_PUBLICATION_ARRAYS_INVALID")
        return cls(
            publication_id=value["publication_id"],  # type: ignore[arg-type]
            task_id=value["task_id"],  # type: ignore[arg-type]
            config_hash=value["config_hash"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
            frozen_source_table_ref=value["frozen_source_table_ref"],  # type: ignore[arg-type]
            frozen_source_table_hash=value["frozen_source_table_hash"],  # type: ignore[arg-type]
            formal_plan_ref=value["formal_plan_ref"],  # type: ignore[arg-type]
            formal_plan_hash=value["formal_plan_hash"],  # type: ignore[arg-type]
            execution_evidence_ref=value["execution_evidence_ref"],  # type: ignore[arg-type]
            execution_evidence_hash=value["execution_evidence_hash"],  # type: ignore[arg-type]
            provenance_ref=value["provenance_ref"],  # type: ignore[arg-type]
            provenance_hash=value["provenance_hash"],  # type: ignore[arg-type]
            stage3_scope_decision_ref=value["stage3_scope_decision_ref"],  # type: ignore[arg-type]
            stage3_scope_decision_hash=value["stage3_scope_decision_hash"],  # type: ignore[arg-type]
            stage3_scope_gate_ref=value["stage3_scope_gate_ref"],  # type: ignore[arg-type]
            stage3_scope_gate_hash=value["stage3_scope_gate_hash"],  # type: ignore[arg-type]
            evaluation_ref=value["evaluation_ref"],  # type: ignore[arg-type]
            evaluation_hash=value["evaluation_hash"],  # type: ignore[arg-type]
            g3_6_ref=value["g3_6_ref"],  # type: ignore[arg-type]
            g3_6_hash=value["g3_6_hash"],  # type: ignore[arg-type]
            gate_evaluation=Stage3GateEvaluation.from_mapping(dict(raw_eval)),
            g3_6_gate=GateRecord.from_mapping(dict(raw_gate)),
            source_artifact_refs=tuple(value["source_artifact_refs"]),  # type: ignore[arg-type]
            reasons=tuple(value["reasons"]),  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


class Stage3G36Publisher:
    """Load formal commits, evaluate the frozen table, then publish G3-6."""

    def publish(
        self,
        *,
        workspace_root: str | Path,
        output_dir: str,
        config_hash: str,
        frozen_source_table_ref: str,
        provenance_ref: str,
        formal_plan_ref: str,
        execution_evidence_ref: str,
        stage3_scope_decision_ref: str,
        stage3_scope_gate_ref: str,
        publication_id: str | None = None,
        task_id: str = STAGE3_G36_TASK_ID,
    ) -> Stage3G36Publication:
        root = Path(workspace_root).resolve()
        _hash(config_hash, field="config_hash")
        _safe_task_id(task_id)

        # All six inputs are complete formal commits.  This is intentionally
        # done before constructing the output store, so malformed input cannot
        # leave a misleading partial publisher directory behind.
        table_artifact = _load_formal_commit(
            root, frozen_source_table_ref, field="frozen_source_table", kinds=_TABLE_KINDS
        )
        provenance_artifact = _load_formal_commit(
            root, provenance_ref, field="provenance", kinds=_PROVENANCE_KINDS
        )
        plan_artifact = _load_formal_commit(
            root, formal_plan_ref, field="formal_plan", kinds=_PLAN_KINDS
        )
        execution_artifact = _load_formal_commit(
            root, execution_evidence_ref, field="execution_evidence", kinds=_EXECUTION_KINDS
        )
        scope_decision_artifact = _load_formal_commit(
            root, stage3_scope_decision_ref, field="scope_decision", kinds=_SCOPE_KINDS
        )
        scope_gate_artifact = _load_formal_commit(
            root, stage3_scope_gate_ref, field="scope_gate", kinds=frozenset({"gate_record"})
        )
        input_artifacts = (
            table_artifact,
            provenance_artifact,
            plan_artifact,
            execution_artifact,
            scope_decision_artifact,
            scope_gate_artifact,
        )
        drifted_configs = tuple(
            item.identity.commit_ref
            for item in input_artifacts
            if item.identity.config_hash != config_hash
        )
        if drifted_configs:
            raise FormalRunRejected(
                "STAGE3_G36_INPUT_CONFIG_HASH_MISMATCH:"
                + ",".join(drifted_configs)
            )

        try:
            table = FrozenSourceTable.from_mapping(dict(table_artifact.payload))
            execution = FormalExecutionEvidence.from_mapping(dict(execution_artifact.payload))
            execution.require_for_stage(3)
            provenance = ProvenanceRecord.from_mapping(dict(provenance_artifact.payload))
            if (
                provenance.status is not ProvenanceStatus.COMPLETED
                or provenance.scope != "formal"
                or provenance.formal_eligible is not True
                or provenance.worktree_clean is not True
            ):
                raise FormalRunRejected("STAGE3_G36_PROVENANCE_NOT_COMPLETED_CLEAN_FORMAL")
            plan = _mapping(plan_artifact.payload, field="formal_plan")
            units, rules = _plan_shape(plan, execution)
            decision = _mapping(scope_decision_artifact.payload, field="stage3_scope_decision")
            scope_gate = _mapping(scope_gate_artifact.payload, field="stage3_scope_gate")
            decision, scope_gate_record = _scope_authority(
                decision_payload=decision,
                gate_payload=scope_gate,
                decision_ref=stage3_scope_decision_ref,
                execution=execution,
            )
        except (OSError, TypeError, ValueError, FormalRunRejected) as error:
            raise FormalRunRejected(f"STAGE3_G36_INPUTS_INVALID:{error}") from error

        evaluator_sources = _evaluation_sources(
            frozen_ref=frozen_source_table_ref,
            execution_ref=execution_evidence_ref,
            plan_ref=formal_plan_ref,
            decision_ref=stage3_scope_decision_ref,
            scope_gate_ref=stage3_scope_gate_ref,
            execution=execution,
            table=table,
        )
        # This is the cycle breaker.  The provenance commit existed first and
        # must cover every evaluator source, but it must not list itself (nor
        # the future evaluation/Gate refs) as an artifact it attested.
        if provenance_ref in evaluator_sources or provenance_ref in provenance.artifact_refs:
            raise FormalRunRejected("STAGE3_G36_PROVENANCE_SELF_BINDING")
        missing_provenance = sorted(set(evaluator_sources) - set(provenance.artifact_refs))
        if missing_provenance:
            raise FormalRunRejected(
                "STAGE3_G36_PROVENANCE_SOURCE_COVERAGE_MISSING:" + ",".join(missing_provenance)
            )
        if frozen_source_table_ref not in provenance.artifact_refs:
            raise FormalRunRejected("STAGE3_G36_PROVENANCE_TABLE_REF_MISSING")

        evaluation = Stage3GateEvaluator().evaluate(
            evaluation_id=publication_id or f"stage3-g36-{table.content_hash[:16]}",
            execution=execution,
            observations=tuple(thaw_json_value(dict(row)) for row in table.rows),
            formal_plan=plan,
            formal_plan_ref=formal_plan_ref,
            required_rule_names=rules,
            required_unit_ids=units,
            thresholds=_mapping(plan["thresholds"], field="formal_plan.thresholds"),
            provenance=provenance,
            source_artifact_refs=evaluator_sources,
            stage3_scope_decision=decision,
            stage3_scope_gate=scope_gate_record,
            stage3_scope_decision_ref=stage3_scope_decision_ref,
            stage3_scope_gate_ref=stage3_scope_gate_ref,
        )
        all_rows_finite = _finite_observation_table(table)
        gate_status = GateStatus.PASS if evaluation.formal_eligible and all_rows_finite else GateStatus.BLOCKED
        reasons = tuple(evaluation.reasons)
        if not all_rows_finite:
            reasons = (*reasons, "FORMAL_OBSERVATION_TABLE_NONFINITE")
        if gate_status is GateStatus.BLOCKED and not reasons:
            reasons = ("STAGE3_G36_EVALUATION_BLOCKED",)
        checked_at = provenance.ended_at or provenance.started_at
        store = TaskArtifactStore(root, output_dir)

        # Publish evaluator output first.  Its commit ref is then immutable
        # evidence in the G3-6 Gate; provenance is not changed.
        evaluation_artifact = store.publish(
            task_id=task_id,
            artifact_kind=STAGE3_G36_EVALUATION_ARTIFACT_KIND,
            config_hash=config_hash,
            run_intent="formal",
            payload=evaluation.to_dict(),  # type: ignore[arg-type]
            formal_eligible=True,
            source_refs=tuple(dict.fromkeys((*evaluator_sources, provenance_ref))),
        )
        measured = _g36_measured(
            table=table,
            evaluation=evaluation,
            units=units,
            rules=rules,
            all_rows_finite=all_rows_finite,
        )
        gate = GateRecord(
            gate_id="stage3.G3-6",
            stage=3,
            status=gate_status,
            checked_at=checked_at,
            measured=measured,
            threshold={
                "complete_unit_rule_coverage": True,
                "all_rows_finite": True,
                "independent_evaluation_status": "PASS",
            },
            evidence_refs=tuple(
                dict.fromkeys(
                    (
                        frozen_source_table_ref,
                        evaluation_artifact.commit_ref,
                        provenance_ref,
                        formal_plan_ref,
                    )
                )
            ),
            reasons=reasons,
        )
        gate_artifact = store.publish(
            task_id=task_id,
            artifact_kind=STAGE3_G36_GATE_ARTIFACT_KIND,
            config_hash=config_hash,
            run_intent="formal",
            payload=gate.to_dict(),  # type: ignore[arg-type]
            formal_eligible=True,
            source_refs=tuple(
                dict.fromkeys((*evaluator_sources, provenance_ref, evaluation_artifact.commit_ref))
            ),
        )
        source_refs = tuple(
            dict.fromkeys(
                (
                    *evaluator_sources,
                    provenance_ref,
                    evaluation_artifact.commit_ref,
                    gate_artifact.commit_ref,
                )
            )
        )
        publication = Stage3G36Publication(
            publication_id=publication_id or f"stage3-g36-{table.content_hash[:16]}",
            task_id=task_id,
            config_hash=config_hash,
            status=gate_status.value,
            formal_eligible=gate_status is GateStatus.PASS,
            frozen_source_table_ref=frozen_source_table_ref,
            frozen_source_table_hash=table.content_hash,
            formal_plan_ref=formal_plan_ref,
            formal_plan_hash=str(plan["artifact_hash"]),
            execution_evidence_ref=execution_evidence_ref,
            execution_evidence_hash=execution.artifact_hash,
            provenance_ref=provenance_ref,
            provenance_hash=provenance.artifact_hash,
            stage3_scope_decision_ref=stage3_scope_decision_ref,
            stage3_scope_decision_hash=str(decision["artifact_hash"]),
            stage3_scope_gate_ref=stage3_scope_gate_ref,
            stage3_scope_gate_hash=scope_gate_record.artifact_hash,
            evaluation_ref=evaluation_artifact.commit_ref,
            evaluation_hash=evaluation.artifact_hash,
            g3_6_ref=gate_artifact.commit_ref,
            g3_6_hash=gate.artifact_hash,
            gate_evaluation=evaluation,
            g3_6_gate=gate,
            source_artifact_refs=source_refs,
            reasons=tuple(reasons),
        )
        # Persist the complete two-output receipt after both immutable commits
        # exist.  The receipt does not contain its own ref, so this final step
        # remains acyclic while giving downstream consumers one bundle to
        # reload and validate.
        store.publish(
            task_id=task_id,
            artifact_kind="g36_publication",
            config_hash=config_hash,
            run_intent="formal",
            payload=publication.to_dict(),
            formal_eligible=True,
            source_refs=source_refs,
        )
        return publication


def publish_stage3_g36(**kwargs: object) -> Stage3G36Publication:
    """Functional entry point for the Stage3.08→G3-6 handoff."""

    return Stage3G36Publisher().publish(**kwargs)  # type: ignore[arg-type]


publish_stage3_g3_6 = publish_stage3_g36


__all__ = [
    "STAGE3_G36_EVALUATION_ARTIFACT_KIND",
    "STAGE3_G36_GATE_ARTIFACT_KIND",
    "STAGE3_G36_PUBLICATION_SCHEMA",
    "STAGE3_G36_TASK_ID",
    "Stage3G36Publication",
    "Stage3G36Publisher",
    "publish_stage3_g36",
    "publish_stage3_g3_6",
]
