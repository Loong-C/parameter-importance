"""Stage 2/3 正式编排共享的证据与资格合同。

本模块只描述“某次编排是否有资格进入 formal 运行边界”，不执行梯度、求积或
模型状态变换。正式资格刻意分成两步：

1. :class:`FormalExecutionEvidence` 证明运行前已经绑定冻结合同、真实资产与前置
   Gate；
2. :class:`ArtifactQualification` 证明运行后产物又经过本阶段权威 Gate 验收。

本机 fixture 可以构造第一类对象的 ``local_fixture`` 形态，但它的
``formal_eligible`` 永远为 ``False``，也不能调用产物资格化入口。这样可以在 CPU
上完整测试状态机，却不会把本机通过数写成正式科学结论。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import PurePosixPath
from typing import Mapping, Sequence

from .errors import FormalRunRejected
from .immutable import freeze_json_mapping, thaw_json_value
from .jsonio import canonical_json_hash
from .status import GateRecord, GateStatus


_ACCEPTABLE_GATE_STATUSES = {
    GateStatus.PASS,
    GateStatus.CONDITIONALLY_ACCEPTED,
}


def _require_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _require_int(value: object, field_name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} 必须是不小于 {minimum} 的整数")
    return value


def _require_unique_hashes(values: Sequence[str], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(
        _require_sha256(value, field_name=f"{field_name}[{index}]")
        for index, value in enumerate(values)
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} 不能包含重复摘要")
    return normalized


def require_accepted_gate(
    gate: GateRecord,
    *,
    stage: int,
    checked_at: datetime | None = None,
) -> GateRecord:
    """要求一个仍有效、属于指定阶段的正式 Gate。

    ``LocalValidationRecord`` 不满足参数类型；FAIL/BLOCKED/STALE/NOT_RUN 以及过期
    的条件接受都被统一拒绝。调用方应把返回记录的 ``artifact_hash`` 写入下游
    artifact，而不是只保存文字状态。
    """

    if not isinstance(gate, GateRecord):
        raise FormalRunRejected("FORMAL_GATE_RECORD_REQUIRED")
    if gate.stage != stage:
        raise FormalRunRejected(
            f"FORMAL_GATE_STAGE_MISMATCH:expected=stage{stage},actual=stage{gate.stage}"
        )
    instant = checked_at or datetime.now(timezone.utc)
    status = gate.effective_status(at=instant)
    if status not in _ACCEPTABLE_GATE_STATUSES:
        raise FormalRunRejected(f"FORMAL_GATE_NOT_ACCEPTABLE:{gate.gate_id}:{status.value}")
    return gate


@dataclass(frozen=True, slots=True)
class FormalExecutionEvidence:
    """正式或本机编排的不可变前置证据。

    Parameters
    ----------
    run_intent:
        只能为 ``local_fixture`` 或 ``formal``。前者不承载任何正式资格。
    contract_freeze_hash:
        本次运行所消费合同冻结 artifact 的 SHA-256。正式运行必填。
    asset_manifest_hashes:
        模型、数据、checkpoint 等真实资产 manifest 摘要。正式运行至少一个；本机
        fixture 可以为空。
    prerequisite_gates:
        已完成前置阶段的权威 Gate。正式运行至少一个，且所有 Gate 必须仍有效、
        不得来自晚于当前阶段的 stage。
    metadata:
        仅允许严格 JSON 值。不得在这里放 token、任意对象或本机验证冒充 Gate。
    """

    run_intent: str
    contract_freeze_hash: str | None = None
    asset_manifest_hashes: tuple[str, ...] = ()
    prerequisite_gates: tuple[GateRecord, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)
    schema_version: str = "formal-execution-evidence-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "formal-execution-evidence-v1":
            raise ValueError("不支持的 FormalExecutionEvidence schema")
        if self.run_intent not in {"local_fixture", "formal"}:
            raise ValueError("run_intent 只能是 local_fixture 或 formal")
        if self.contract_freeze_hash is not None:
            _require_sha256(
                self.contract_freeze_hash, field_name="contract_freeze_hash"
            )
        hashes = _require_unique_hashes(
            self.asset_manifest_hashes, field_name="asset_manifest_hashes"
        )
        object.__setattr__(self, "asset_manifest_hashes", hashes)
        gates = tuple(self.prerequisite_gates)
        if any(not isinstance(gate, GateRecord) for gate in gates):
            raise TypeError("prerequisite_gates 只能包含 GateRecord")
        gate_ids = tuple(gate.gate_id for gate in gates)
        if len(set(gate_ids)) != len(gate_ids):
            raise ValueError("prerequisite_gates 不能重复 gate_id")
        object.__setattr__(self, "prerequisite_gates", gates)
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, field="FormalExecutionEvidence.metadata"),
        )

        if self.run_intent == "local_fixture":
            # fixture 可以绑定本机输入摘要，但不能夹带可接受的正式 Gate，从而避免
            # 同一个对象在调用栈中被误判成 formal authorization。
            if any(gate.status in _ACCEPTABLE_GATE_STATUSES for gate in gates):
                raise FormalRunRejected("LOCAL_FIXTURE_MUST_NOT_CARRY_ACCEPTED_GATE")
            return
        if self.contract_freeze_hash is None:
            raise FormalRunRejected("FORMAL_CONTRACT_FREEZE_REQUIRED")
        if not hashes:
            raise FormalRunRejected("FORMAL_ASSET_MANIFESTS_REQUIRED")
        if not gates:
            raise FormalRunRejected("FORMAL_PREREQUISITE_GATES_REQUIRED")

    @property
    def formal_eligible(self) -> bool:
        """仅表示前置执行证据是否完整，不代表任何输出已经通过本阶段 Gate。"""

        if self.run_intent != "formal":
            return False
        now = datetime.now(timezone.utc)
        return bool(
            self.contract_freeze_hash
            and self.asset_manifest_hashes
            and self.prerequisite_gates
            and all(
                gate.effective_status(at=now) in _ACCEPTABLE_GATE_STATUSES
                for gate in self.prerequisite_gates
            )
        )

    def require_for_stage(self, stage: int) -> "FormalExecutionEvidence":
        """在正式入口再次检查时效与 stage 单向依赖，失败即拒绝运行。"""

        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9:
            raise ValueError("stage 必须是 0..9 的整数")
        if self.run_intent != "formal":
            raise FormalRunRejected("FORMAL_RUN_INTENT_REQUIRED")
        if not self.formal_eligible:
            raise FormalRunRejected("FORMAL_EXECUTION_EVIDENCE_NOT_ELIGIBLE")
        for gate in self.prerequisite_gates:
            if gate.stage > stage:
                raise FormalRunRejected(
                    f"FORMAL_PREREQUISITE_GATE_FROM_FUTURE_STAGE:{gate.gate_id}"
                )
            require_accepted_gate(gate, stage=gate.stage)
        return self

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_intent": self.run_intent,
            "contract_freeze_hash": self.contract_freeze_hash,
            "asset_manifest_hashes": list(self.asset_manifest_hashes),
            "prerequisite_gates": [gate.to_dict() for gate in self.prerequisite_gates],
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FormalExecutionEvidence":
        required = {
            "schema_version",
            "run_intent",
            "contract_freeze_hash",
            "asset_manifest_hashes",
            "prerequisite_gates",
            "metadata",
            "artifact_hash",
        }
        if set(value) != required:
            raise ValueError(
                "FormalExecutionEvidence 字段集合不匹配："
                f"missing={sorted(required-set(value))}, extra={sorted(set(value)-required)}"
            )
        hashes = value["asset_manifest_hashes"]
        gates = value["prerequisite_gates"]
        metadata = value["metadata"]
        for field_name in ("schema_version", "run_intent", "artifact_hash"):
            if not isinstance(value[field_name], str):
                raise TypeError(f"{field_name} 必须是字符串")
        if value["contract_freeze_hash"] is not None and not isinstance(
            value["contract_freeze_hash"], str
        ):
            raise TypeError("contract_freeze_hash 必须是字符串或 null")
        if not isinstance(hashes, list) or not all(isinstance(item, str) for item in hashes):
            raise TypeError("asset_manifest_hashes 必须是字符串数组")
        if not isinstance(gates, list) or not all(isinstance(item, Mapping) for item in gates):
            raise TypeError("prerequisite_gates 必须是 GateRecord object 数组")
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata 必须是 object")
        evidence = cls(
            run_intent=value["run_intent"],
            contract_freeze_hash=(
                None
                if value["contract_freeze_hash"] is None
                else value["contract_freeze_hash"]
            ),
            asset_manifest_hashes=tuple(hashes),
            prerequisite_gates=tuple(GateRecord.from_mapping(dict(item)) for item in gates),
            metadata=metadata,
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != evidence.artifact_hash:
            raise ValueError("FormalExecutionEvidence artifact_hash 与内容不一致")
        return evidence


@dataclass(frozen=True, slots=True)
class ArtifactQualification:
    """输出 artifact 的 scope 与本阶段 Gate 绑定。

    未资格化产物可以是 ``pilot`` 或 ``formal`` scope 的候选证据，但
    ``formal_eligible`` 必须为 ``False``；只有 formal 产物携带本阶段可接受 Gate
    的摘要时才能为真。
    """

    scope: str
    formal_eligible: bool = False
    qualification_gate_hash: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"local_fixture", "pilot", "formal"}:
            raise ValueError("ArtifactQualification.scope 不受支持")
        if type(self.formal_eligible) is not bool:
            raise TypeError("formal_eligible 必须是显式 bool")
        if self.qualification_gate_hash is not None:
            _require_sha256(
                self.qualification_gate_hash, field_name="qualification_gate_hash"
            )
        if self.scope in {"local_fixture", "pilot"} and self.formal_eligible:
            raise FormalRunRejected("NON_FORMAL_SCOPE_CANNOT_BE_FORMAL_ELIGIBLE")
        if self.formal_eligible and self.qualification_gate_hash is None:
            raise FormalRunRejected("FORMAL_QUALIFICATION_GATE_HASH_REQUIRED")
        if not self.formal_eligible and self.qualification_gate_hash is not None:
            raise FormalRunRejected("UNQUALIFIED_ARTIFACT_CANNOT_CARRY_GATE_HASH")

    @classmethod
    def candidate(cls, scope: str) -> "ArtifactQualification":
        return cls(scope=scope, formal_eligible=False, qualification_gate_hash=None)

    @classmethod
    def from_gate(
        cls,
        *,
        scope: str,
        gate: GateRecord,
        stage: int,
    ) -> "ArtifactQualification":
        if scope != "formal":
            raise FormalRunRejected("ONLY_FORMAL_ARTIFACT_CAN_BE_QUALIFIED")
        accepted = require_accepted_gate(gate, stage=stage)
        return cls(
            scope="formal",
            formal_eligible=True,
            qualification_gate_hash=accepted.artifact_hash,
        )


__all__ = [
    "ArtifactQualification",
    "FormalExecutionEvidence",
    "require_accepted_gate",
    "validate_stage23_artifact",
]


_STAGE23_ARTIFACT_FIELDS: dict[str, set[str]] = {
    "stage2-formal-experiment-plan-v1": {
        "schema_version",
        "plan_id",
        "task_id",
        "wave_id",
        "cell_id",
        "stream",
        "batch_size",
        "microbatch_counts",
        "repetitions",
        "sampling_plan_hash",
        "execution_evidence_hash",
        "source_artifact_refs",
        "selection_basis",
        "pilot_thresholds",
        "scope",
        "state",
        "formal_eligible",
        "artifact_hash",
    },
    "stage2-reference-sizing-plan-v1": {
        "schema_version",
        "reference_id",
        "candidate_sample_counts",
        "block_size",
        "convergence_tolerance",
        "required_consecutive",
        "execution_evidence_hash",
        "draw_start_position",
        "draw_end_position_exclusive",
        "require_terminal_convergence",
        "round_manifest_ref",
        "final_stream_start_position",
        "final_stream_end_position_exclusive",
        "artifact_hash",
    },
    "stage2-reference-sizing-result-v1": {
        "schema_version",
        "plan_hash",
        "registry_hash",
        "provider_state_digest",
        "processed_sample_count_per_stream",
        "selected_sample_count_per_stream",
        "converged",
        "status",
        "points",
        "bias_reference_hash",
        "cross_reference_hash",
        "ranking_reference_hash",
        "resumed_from_block_pairs",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "weighting_assumptions",
        "artifact_hash",
    },
    "stage2-reference-uncertainty-v1": {
        "schema_version",
        "estimator",
        "confidence_level",
        "block_count_a",
        "block_count_b",
        "bias_variance_hash",
        "cross_variance_hash",
        "ranking_variance_hash",
        "trace_bias_variance",
        "bias_half_width_l2",
        "artifact_hash",
    },
    "stage2-reference-one-shot-plan-v1": {
        "schema_version",
        "reference_id",
        "sizing_result_hash",
        "sample_count_per_stream",
        "block_size",
        "sizing_stream",
        "stream_a",
        "stream_b",
        "one_shot",
        "artifact_hash",
    },
    "stage2-reference-one-shot-plan-v2": {
        "schema_version",
        "reference_id",
        "sizing_result_hash",
        "sample_count_per_stream",
        "block_size",
        "sizing_stream",
        "stream_a",
        "stream_b",
        "one_shot",
        "stream_a_draw_start_position",
        "stream_a_draw_end_position_exclusive",
        "stream_b_draw_start_position",
        "stream_b_draw_end_position_exclusive",
        "artifact_hash",
    },
    "stage2-reference-one-shot-result-v1": {
        "schema_version",
        "plan_hash",
        "sizing_result_hash",
        "provider_state_digest",
        "registry_hash",
        "processed_sample_count_per_stream",
        "bias_reference_hash",
        "cross_reference_hash",
        "ranking_reference_hash",
        "uncertainty",
        "stream_a_draw_hash",
        "stream_b_draw_hash",
        "status",
        "one_shot",
        "weighting_assumptions",
        "sequence_variance_hash",
        "artifact_hash",
    },
    "stage2-paired-wave-summary-v1": {
        "schema_version",
        "wave_id",
        "registry_hash",
        "reference_hash",
        "reference_hashes",
        "expected_unit_ids",
        "completed_unit_ids",
        "complete",
        "status",
        "method_statistics",
        "reference_statistics",
        "microbatch_diagnostics",
        "replay_evidence",
        "cost_statistics",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "resumed_unit_count",
        "weighting_assumptions",
        "artifact_hash",
    },
    "stage2-paired-wave-plan-v1": {
        "schema_version",
        "wave_id",
        "reference_hash",
        "reference_hashes",
        "registry_hash",
        "provider_state_digest",
        "execution_evidence_hash",
        "weighting_assumptions",
        "mappings",
        "artifact_hash",
    },
    "stage2-estimator-recommendation-v1": {
        "schema_version",
        "recommendation_id",
        "status",
        "selected_estimator",
        "batch_size",
        "microbatch_count",
        "repetitions",
        "required_cells",
        "qualified_estimators",
        "thresholds",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "execution_evidence_hash",
        "reasons",
        "artifact_hash",
    },
    "stage2-pilot-matrix-freeze-v1": {
        "schema_version", "freeze_id", "scope", "status", "anchor_ids",
        "candidate_evaluations", "b_primary", "m_primary", "r_primary",
        "completion_denominator", "cost_semantics", "cost_observations", "cost_io_quiescent",
        "confirmatory_draw_stream", "pilot_draw_stream", "formal_eligible",
        "qualification_gate_hash", "artifact_hash",
    },
    "stage2-confirmatory-mapping-v1": {
        "schema_version", "mapping_id", "scope", "stream", "freeze_hash",
        "sampling_plan_hash", "pilot_draw_ids", "confirmatory_draw_ids",
        "mappings", "draw_id_unique", "sample_id_collision_count", "complete",
        "formal_eligible", "qualification_gate_hash", "artifact_hash",
    },
    "stage3-endpoint-capture-v1": {
        "schema_version",
        "record",
        "execution_evidence_hash",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "artifact_hash",
    },
    "stage3-probe-panel-v1": {
        "schema_version",
        "panel_id",
        "endpoint_digest",
        "entries",
        "minimum_formal_probes",
        "execution_evidence_hash",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "artifact_hash",
    },
    "stage3-reference-refinement-v1": {
        "schema_version",
        "unit_id",
        "converged",
        "convergence_defined",
        "status",
        "primary_family",
        "selected_level",
        "selected_rule_hash",
        "conservative_error",
        "within_family_errors",
        "cross_family_error",
        "completed_levels",
        "reference_contribution_hash",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "execution_evidence_hash",
        "reasons",
        "artifact_hash",
    },
    "stage3-quadrature-recommendation-v1": {
        "schema_version",
        "recommendation_id",
        "status",
        "default_rule",
        "fallback_rule",
        "passing_rules",
        "required_unit_ids",
        "thresholds",
        "thresholds_hash",
        "execution_evidence_hash",
        "scope",
        "formal_eligible",
        "qualification_gate_hash",
        "gate_evaluation_hash",
        "provenance_hash",
        "reasons",
        "artifact_hash",
    },
    "stage3-gate-evaluation-v1": {
        "schema_version",
        "evaluation_id",
        "status",
        "scope",
        "formal_eligible",
        "execution_evidence_hash",
        "formal_plan_hash",
        "formal_plan_ref",
        "thresholds_hash",
        "required_gate_ids",
        "gate_hashes",
        "required_rule_names",
        "required_unit_ids",
        "required_strata",
        "required_top_q",
        "rule_evaluations",
        "provenance_hash",
        "source_artifact_refs",
        "reasons",
        "stage3_scope_decision_ref",
        "stage3_scope_decision_hash",
        "stage3_scope_gate_ref",
        "stage3_scope_gate_hash",
        "artifact_hash",
    },
    "stage3-finalization-v1": {
        "schema_version",
        "finalization_id",
        "status",
        "scope",
        "formal_eligible",
        "execution_evidence_hash",
        "frozen_table_ref",
        "frozen_table_hash",
        "formal_plan_ref",
        "formal_plan_hash",
        "evaluation_ref",
        "evaluation_hash",
        "provenance_ref",
        "provenance_hash",
        "recommendation_ref",
        "g3_6_ref",
        "g3_7_ref",
        "recommendation",
        "candidate_recommendation",
        "gate_evaluation",
        "prerequisite_gates",
        "g3_6_gate",
        "g3_7_gate",
        "method_selection",
        "source_artifact_refs",
        "reasons",
        "checked_at",
        "artifact_hash",
    },
    "stage3-g36-publication-v1": {
        "schema_version",
        "publication_id",
        "task_id",
        "config_hash",
        "status",
        "scope",
        "formal_eligible",
        "frozen_source_table_ref",
        "frozen_source_table_hash",
        "formal_plan_ref",
        "formal_plan_hash",
        "execution_evidence_ref",
        "execution_evidence_hash",
        "provenance_ref",
        "provenance_hash",
        "stage3_scope_decision_ref",
        "stage3_scope_decision_hash",
        "stage3_scope_gate_ref",
        "stage3_scope_gate_hash",
        "evaluation_ref",
        "evaluation_hash",
        "g3_6_ref",
        "g3_6_hash",
        "gate_evaluation",
        "g3_6_gate",
        "source_artifact_refs",
        "reasons",
        "artifact_hash",
    },
}


def _validate_stage3_finalization_wire(value: Mapping[str, object]) -> None:
    """Validate finalization's nested formal authorities without upgrading it.

    A qualified recommendation is intentionally not passed to
    ``QuadratureRecommendation.from_mapping`` here: that class requires live
    execution/evaluation/provenance authorities.  This loader only checks the
    immutable wire hashes and nested Gate/evaluation contracts; callers that
    need scientific qualification must use ``Stage3Finalizer.reload_live``.
    """

    if value.get("schema_version") != "stage3-finalization-v1":
        raise ValueError("STAGE3_FINALIZATION_SCHEMA_UNSUPPORTED")
    if value.get("scope") != "formal":
        raise ValueError("STAGE3_FINALIZATION_SCOPE_INVALID")
    status = value.get("status")
    if status not in {"PASS", "BLOCKED"}:
        raise ValueError("STAGE3_FINALIZATION_STATUS_INVALID")
    if type(value.get("formal_eligible")) is not bool:
        raise TypeError("STAGE3_FINALIZATION_FORMAL_ELIGIBLE_INVALID")
    if value["formal_eligible"] is not (status == "PASS"):
        raise FormalRunRejected("STAGE3_FINALIZATION_ELIGIBILITY_MISMATCH")

    def nullable_hash(field_name: str) -> None:
        raw = value[field_name]
        if raw is not None:
            _require_sha256(raw, field_name=field_name)

    def nullable_ref(field_name: str) -> None:
        raw = value[field_name]
        if raw is not None:
            if not isinstance(raw, str) or not raw or "?" in raw or "://" in raw:
                raise ValueError(f"{field_name} 必须是稳定 artifact ref")
            if any(marker in raw.casefold() for marker in ("fixture", "synthetic")):
                raise FormalRunRejected(f"{field_name} 禁止 fixture/synthetic formal ref")

    for field_name in (
        "execution_evidence_hash",
        "frozen_table_hash",
        "formal_plan_hash",
        "evaluation_hash",
        "provenance_hash",
    ):
        nullable_hash(field_name)
    for field_name in (
        "frozen_table_ref",
        "formal_plan_ref",
        "evaluation_ref",
        "provenance_ref",
        "recommendation_ref",
        "g3_6_ref",
        "g3_7_ref",
    ):
        nullable_ref(field_name)

    refs = value["source_artifact_refs"]
    if not isinstance(refs, list) or any(not isinstance(item, str) or not item for item in refs):
        raise TypeError("STAGE3_FINALIZATION_SOURCE_REFS_INVALID")
    if len(set(refs)) != len(refs):
        raise ValueError("STAGE3_FINALIZATION_SOURCE_REFS_DUPLICATE")
    if any(any(marker in item.casefold() for marker in ("fixture", "synthetic")) for item in refs):
        raise FormalRunRejected("STAGE3_FINALIZATION_SOURCE_REF_FORBIDDEN")
    reasons = value["reasons"]
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise TypeError("STAGE3_FINALIZATION_REASONS_INVALID")
    if status == "BLOCKED" and not reasons:
        raise FormalRunRejected("STAGE3_FINALIZATION_BLOCKED_REASON_REQUIRED")

    from ..experiments.stage3_gate import REQUIRED_STAGE3_GATE_IDS, Stage3GateEvaluation
    from .status import GateRecord, GateStatus

    raw_pre = value["prerequisite_gates"]
    if not isinstance(raw_pre, list) or any(not isinstance(item, Mapping) for item in raw_pre):
        raise TypeError("STAGE3_FINALIZATION_PREREQUISITE_GATES_INVALID")
    pre = tuple(GateRecord.from_mapping(dict(item)) for item in raw_pre)
    if status == "PASS":
        if tuple(item.gate_id for item in pre) != REQUIRED_STAGE3_GATE_IDS:
            raise FormalRunRejected("STAGE3_FINALIZATION_PREREQUISITE_GATES_INCOMPLETE")
        if any(item.status is not GateStatus.PASS or item.effective_status() is not GateStatus.PASS for item in pre):
            raise FormalRunRejected("STAGE3_FINALIZATION_PREREQUISITE_GATE_NOT_PASS")

    raw_eval = value["gate_evaluation"]
    if status == "PASS":
        if not isinstance(raw_eval, Mapping):
            raise TypeError("STAGE3_FINALIZATION_GATE_EVALUATION_REQUIRED")
        evaluation = Stage3GateEvaluation.from_mapping(dict(raw_eval))
        if evaluation.status != "PASS" or not evaluation.formal_eligible:
            raise FormalRunRejected("STAGE3_FINALIZATION_GATE_EVALUATION_NOT_PASS")
        if value["evaluation_hash"] != evaluation.artifact_hash:
            raise ValueError("STAGE3_FINALIZATION_EVALUATION_HASH_MISMATCH")
    elif raw_eval is not None:
        if not isinstance(raw_eval, Mapping):
            raise TypeError("STAGE3_FINALIZATION_GATE_EVALUATION_INVALID")
        Stage3GateEvaluation.from_mapping(dict(raw_eval))

    def verify_nested_hash(raw: object, *, field_name: str) -> Mapping[str, object]:
        if not isinstance(raw, Mapping):
            raise TypeError(f"{field_name} 必须是 object")
        supplied_hash = raw.get("artifact_hash")
        _require_sha256(supplied_hash, field_name=f"{field_name}.artifact_hash")
        observed_hash = canonical_json_hash({key: item for key, item in raw.items() if key != "artifact_hash"})
        if supplied_hash != observed_hash:
            raise ValueError(f"{field_name}.artifact_hash 与内容不一致")
        return raw

    raw_candidate = value["candidate_recommendation"]
    if status == "PASS":
        if not isinstance(raw_candidate, Mapping):
            raise TypeError("STAGE3_FINALIZATION_CANDIDATE_REQUIRED")
        candidate = verify_nested_hash(raw_candidate, field_name="candidate_recommendation")
        if candidate.get("schema_version") != "stage3-quadrature-recommendation-v1" or candidate.get("status") != "FORMAL_CANDIDATE" or candidate.get("formal_eligible") is not False:
            raise FormalRunRejected("STAGE3_FINALIZATION_CANDIDATE_STATUS_INVALID")
        # The candidate is unqualified, so the normal Stage 2/3 wire checker
        # can validate its threshold/hash contract without accepting it as a
        # formal result.
        validate_stage23_artifact(candidate)
        raw_recommendation = verify_nested_hash(value["recommendation"], field_name="recommendation")
        if raw_recommendation.get("schema_version") != "stage3-quadrature-recommendation-v1" or raw_recommendation.get("status") != "QUALIFIED" or raw_recommendation.get("formal_eligible") is not True:
            raise FormalRunRejected("STAGE3_FINALIZATION_RECOMMENDATION_NOT_QUALIFIED")
        if value["recommendation_ref"] is None:
            raise FormalRunRejected("STAGE3_FINALIZATION_RECOMMENDATION_REF_REQUIRED")
        for field_name in ("g3_6_gate", "g3_7_gate", "method_selection"):
            if value[field_name] is None:
                raise FormalRunRejected(f"STAGE3_FINALIZATION_{field_name.upper()}_REQUIRED")
        g36 = GateRecord.from_mapping(dict(value["g3_6_gate"]))
        g37 = GateRecord.from_mapping(dict(value["g3_7_gate"]))
        if g36.gate_id != "stage3.G3-6" or g37.gate_id != "stage3.G3-7" or g36.status is not GateStatus.PASS or g37.status is not GateStatus.PASS:
            raise FormalRunRejected("STAGE3_FINALIZATION_G3_6_G3_7_NOT_PASS")
        if not isinstance(value["g3_6_ref"], str) or not isinstance(value["g3_7_ref"], str):
            raise FormalRunRejected("STAGE3_FINALIZATION_GATE_REFS_REQUIRED")
        if value["g3_6_ref"] not in refs or value["g3_7_ref"] not in refs:
            raise FormalRunRejected("STAGE3_FINALIZATION_GATE_REFS_UNBOUND")
        if not {
            value["frozen_table_ref"],
            value["evaluation_ref"],
            value["provenance_ref"],
            value["formal_plan_ref"],
        }.issubset(set(g36.evidence_refs)):
            raise FormalRunRejected("STAGE3_FINALIZATION_G3_6_EVIDENCE_UNBOUND")
        selection = value["method_selection"]
        if not isinstance(selection, Mapping) or selection.get("status") != "QUALIFIED":
            raise FormalRunRejected("STAGE3_FINALIZATION_METHOD_SELECTION_INVALID")
        expected_selection = {
            "default_rule": raw_recommendation.get("default_rule"),
            "fallback_rule": raw_recommendation.get("fallback_rule"),
            "passing_rules": raw_recommendation.get("passing_rules"),
            "required_unit_ids": raw_recommendation.get("required_unit_ids"),
            "thresholds_hash": raw_recommendation.get("thresholds_hash"),
            "execution_evidence_hash": value["execution_evidence_hash"],
            "frozen_table_ref": value["frozen_table_ref"],
            "frozen_table_hash": value["frozen_table_hash"],
            "formal_plan_ref": value["formal_plan_ref"],
            "formal_plan_hash": value["formal_plan_hash"],
            "evaluation_ref": value["evaluation_ref"],
            "evaluation_hash": value["evaluation_hash"],
            "provenance_ref": value["provenance_ref"],
            "provenance_hash": value["provenance_hash"],
        }
        for field_name, expected_value in expected_selection.items():
            if selection.get(field_name) != expected_value:
                raise ValueError(f"STAGE3_FINALIZATION_METHOD_SELECTION_BINDING_MISMATCH:{field_name}")
        if selection.get("g3_6_gate_hash") != g36.artifact_hash or selection.get("g3_7_gate_hash") != g37.artifact_hash:
            raise ValueError("STAGE3_FINALIZATION_METHOD_SELECTION_GATE_HASH_MISMATCH")
    else:
        if raw_candidate is not None:
            verify_nested_hash(raw_candidate, field_name="candidate_recommendation")
        if value["recommendation"] is not None:
            verify_nested_hash(value["recommendation"], field_name="recommendation")
        for field_name in ("g3_6_gate", "g3_7_gate"):
            if value[field_name] is not None:
                GateRecord.from_mapping(dict(value[field_name]))


def _validate_stage3_g36_publication_wire(value: Mapping[str, object]) -> None:
    """Validate the acyclic Stage3.08→G3-6 publication receipt.

    This is a wire-only check.  It verifies the nested evaluator/Gate hashes
    and their declared bindings; it does not reload source commits or turn a
    receipt into live authority.
    """

    if value.get("schema_version") != "stage3-g36-publication-v1":
        raise ValueError("STAGE3_G36_PUBLICATION_SCHEMA_UNSUPPORTED")
    if value.get("scope") != "formal":
        raise ValueError("STAGE3_G36_PUBLICATION_SCOPE_INVALID")
    status = value.get("status")
    if status not in {"PASS", "BLOCKED"}:
        raise ValueError("STAGE3_G36_PUBLICATION_STATUS_INVALID")
    if value.get("formal_eligible") is not (status == "PASS"):
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_ELIGIBILITY_MISMATCH")
    for field_name in (
        "config_hash",
        "frozen_source_table_hash",
        "formal_plan_hash",
        "execution_evidence_hash",
        "provenance_hash",
        "stage3_scope_decision_hash",
        "stage3_scope_gate_hash",
        "evaluation_hash",
        "g3_6_hash",
    ):
        _require_sha256(value[field_name], field_name=field_name)
    refs = value.get("source_artifact_refs")
    if not isinstance(refs, list) or not refs or len(set(refs)) != len(refs):
        raise ValueError("STAGE3_G36_PUBLICATION_SOURCE_REFS_INVALID")
    if any(
        not isinstance(item, str)
        or not item
        or "?" in item
        or "://" in item
        or any(marker in item.casefold() for marker in ("fixture", "synthetic"))
        for item in refs
    ):
        raise ValueError("STAGE3_G36_PUBLICATION_SOURCE_REFS_INVALID")
    for field_name in (
        "frozen_source_table_ref",
        "formal_plan_ref",
        "execution_evidence_ref",
        "provenance_ref",
        "stage3_scope_decision_ref",
        "stage3_scope_gate_ref",
        "evaluation_ref",
        "g3_6_ref",
    ):
        ref = value[field_name]
        if (
            not isinstance(ref, str)
            or not ref
            or "?" in ref
            or "://" in ref
            or any(marker in ref.casefold() for marker in ("fixture", "synthetic"))
        ):
            raise ValueError(f"STAGE3_G36_PUBLICATION_REF_INVALID:{field_name}")
    required_refs = {
        value["frozen_source_table_ref"],
        value["formal_plan_ref"],
        value["execution_evidence_ref"],
        value["provenance_ref"],
        value["stage3_scope_decision_ref"],
        value["stage3_scope_gate_ref"],
        value["evaluation_ref"],
        value["g3_6_ref"],
    }
    if not required_refs.issubset(set(refs)):
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_OUTPUT_REFS_UNBOUND")
    reasons = value.get("reasons")
    if not isinstance(reasons, list) or any(not isinstance(item, str) or not item for item in reasons):
        raise ValueError("STAGE3_G36_PUBLICATION_REASONS_INVALID")
    if status == "BLOCKED" and not reasons:
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_BLOCKED_REASON_REQUIRED")
    from ..experiments.stage3_gate import Stage3GateEvaluation
    from .status import GateRecord, GateStatus

    raw_eval = value.get("gate_evaluation")
    raw_gate = value.get("g3_6_gate")
    if not isinstance(raw_eval, Mapping) or not isinstance(raw_gate, Mapping):
        raise TypeError("STAGE3_G36_PUBLICATION_NESTED_OBJECTS_REQUIRED")
    evaluation = Stage3GateEvaluation.from_mapping(dict(raw_eval))
    gate = GateRecord.from_mapping(dict(raw_gate))
    if evaluation.artifact_hash != value["evaluation_hash"]:
        raise ValueError("STAGE3_G36_PUBLICATION_EVALUATION_HASH_MISMATCH")
    if gate.artifact_hash != value["g3_6_hash"]:
        raise ValueError("STAGE3_G36_PUBLICATION_GATE_HASH_MISMATCH")
    if gate.gate_id != "stage3.G3-6" or gate.status.value != status:
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_GATE_STATUS_MISMATCH")
    if evaluation.status != status or evaluation.formal_eligible != (status == "PASS"):
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_EVALUATION_STATUS_MISMATCH")
    if (
        evaluation.execution_evidence_hash != value["execution_evidence_hash"]
        or evaluation.formal_plan_hash != value["formal_plan_hash"]
        or evaluation.formal_plan_ref != value["formal_plan_ref"]
        or evaluation.provenance_hash != value["provenance_hash"]
        or evaluation.stage3_scope_decision_ref != value["stage3_scope_decision_ref"]
        or evaluation.stage3_scope_decision_hash != value["stage3_scope_decision_hash"]
        or evaluation.stage3_scope_gate_ref != value["stage3_scope_gate_ref"]
        or evaluation.stage3_scope_gate_hash != value["stage3_scope_gate_hash"]
    ):
        raise ValueError("STAGE3_G36_PUBLICATION_EVALUATION_BINDING_MISMATCH")
    if value["provenance_ref"] in evaluation.source_artifact_refs:
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_PROVENANCE_SELF_BINDING")
    if not {
        value["frozen_source_table_ref"],
        value["evaluation_ref"],
        value["provenance_ref"],
        value["formal_plan_ref"],
    }.issubset(set(gate.evidence_refs)):
        raise FormalRunRejected("STAGE3_G36_PUBLICATION_GATE_EVIDENCE_UNBOUND")
    measured = gate.measured
    if (
        not isinstance(measured, Mapping)
        or measured.get("source_table_hash") != value["frozen_source_table_hash"]
        or measured.get("evaluation_hash") != evaluation.artifact_hash
    ):
        raise ValueError("STAGE3_G36_PUBLICATION_GATE_MEASURED_BINDING_MISMATCH")


def validate_stage23_artifact(value: Mapping[str, object]) -> object:
    """严格验证 Stage 2/3 新编排 artifact 的 wire identity 与资格边界。

    项目运行时不依赖 ``jsonschema``，因此读取外部 artifact 时仍需要这条轻量、
    fail-closed 的 Python 边界。函数不会读取 tensor bundle，也不会把候选结果
    升级为正式证据；返回值沿用 :class:`contracts.artifacts.ValidatedArtifact`。
    """

    if not isinstance(value, Mapping):
        raise TypeError("Stage2/3 artifact 必须是 object")
    schema = value.get("schema_version")
    if not isinstance(schema, str) or schema not in _STAGE23_ARTIFACT_FIELDS:
        raise ValueError(f"未知 Stage2/3 artifact schema: {schema!r}")
    required = _STAGE23_ARTIFACT_FIELDS[schema]
    allowed_field_sets = (required,)
    if schema == "stage3-quadrature-recommendation-v1":
        # Local/legacy fixture recommendations predate the independent Gate
        # evaluator.  They may omit the two optional bindings, but a
        # QUALIFIED/formal artifact must carry both and is checked below.
        allowed_field_sets = (
            required,
            required - {"gate_evaluation_hash", "provenance_hash"},
        )
    if schema == "stage2-reference-sizing-plan-v1":
        legacy = required - {
            "draw_start_position",
            "draw_end_position_exclusive",
            "require_terminal_convergence",
            "round_manifest_ref",
        }
        allowed_field_sets = (legacy, required)
    if set(value) not in allowed_field_sets:
        raise ValueError(
            f"{schema} 字段集合不匹配：observed={sorted(set(value))}"
        )
    supplied = value.get("artifact_hash")
    if not isinstance(supplied, str):
        raise TypeError("artifact_hash 必须是字符串")
    _require_sha256(supplied, field_name="artifact_hash")
    payload = {name: item for name, item in value.items() if name != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise ValueError(f"{schema} artifact_hash 与完整 wire object 不一致")
    if schema == "stage3-gate-evaluation-v1":
        from ..experiments.stage3_gate import Stage3GateEvaluation

        Stage3GateEvaluation.from_mapping(value)
    if schema == "stage3-finalization-v1":
        _validate_stage3_finalization_wire(value)
    if schema == "stage3-g36-publication-v1":
        _validate_stage3_g36_publication_wire(value)

    def string_array(field_name: str, *, allow_empty: bool = True) -> tuple[str, ...]:
        raw = value[field_name]
        if not isinstance(raw, list) or not all(
            isinstance(item, str) and item for item in raw
        ):
            raise TypeError(f"{field_name} 必须是非空字符串数组")
        if not allow_empty and not raw:
            raise ValueError(f"{field_name} 不能为空")
        if len(set(raw)) != len(raw):
            raise ValueError(f"{field_name} 不能重复")
        return tuple(raw)

    def validate_weighting(raw: object) -> None:
        fields = {
            "statistical_unit",
            "weight_unit",
            "sampling_design",
            "weights_exogenous",
            "common_mean_assumption",
        }
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError("weighting_assumptions 字段不完整")
        for field_name in ("statistical_unit", "weight_unit", "sampling_design"):
            if not isinstance(raw[field_name], str) or not raw[field_name]:
                raise TypeError(f"weighting_assumptions.{field_name} 必须是非空字符串")
        for field_name in ("weights_exogenous", "common_mean_assumption"):
            if type(raw[field_name]) is not bool:
                raise TypeError(f"weighting_assumptions.{field_name} 必须是显式 bool")

    if schema == "stage2-formal-experiment-plan-v1":
        for field_name in (
            "plan_id",
            "task_id",
            "wave_id",
            "cell_id",
            "stream",
            "selection_basis",
        ):
            if not isinstance(value[field_name], str) or not value[field_name]:
                raise TypeError(f"{field_name} 必须是非空字符串")
        batch_size = value["batch_size"]
        repetitions = value["repetitions"]
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise TypeError("batch_size 必须是正整数")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions <= 0
        ):
            raise TypeError("repetitions 必须是正整数")
        microbatch_counts = value["microbatch_counts"]
        if not isinstance(microbatch_counts, list) or not microbatch_counts or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 2
            for item in microbatch_counts
        ):
            raise TypeError("microbatch_counts 必须是非空且元素 >=2 的整数数组")
        if microbatch_counts != sorted(set(microbatch_counts)):
            raise ValueError("microbatch_counts 必须严格递增且无重复")
        if batch_size not in {32, 64, 128, 256}:
            raise ValueError("batch_size 不属于冻结候选集合")
        if any(
            item not in {2, 4, 8, 16, 32} or batch_size % item
            for item in microbatch_counts
        ):
            raise ValueError("microbatch_counts 不属于冻结候选或不能整除 B")
        largest = max(microbatch_counts)
        if any(largest % item for item in microbatch_counts):
            raise ValueError("microbatch_counts 不是嵌套划分")
        source_refs = string_array("source_artifact_refs", allow_empty=False)
        for index, reference in enumerate(source_refs):
            logical = PurePosixPath(reference)
            if (
                "\\" in reference
                or logical.is_absolute()
                or any(part in {"", ".", ".."} for part in logical.parts)
            ):
                raise ValueError(f"source_artifact_refs[{index}] 不是 POSIX 引用")
        if list(source_refs) != sorted(source_refs):
            raise ValueError("source_artifact_refs 必须按 canonical 顺序排列")
        if value["scope"] != "formal" or value["state"] != "FROZEN":
            raise FormalRunRejected("FORMAL_EXPERIMENT_PLAN_NOT_FROZEN")
        if value["formal_eligible"] is not True:
            raise FormalRunRejected("FORMAL_EXPERIMENT_PLAN_ELIGIBILITY_REQUIRED")
        thresholds = value["pilot_thresholds"]
        if thresholds is not None and not isinstance(thresholds, Mapping):
            raise TypeError("pilot_thresholds 必须是 object 或 null")
        task_contract = {
            "stage2.05_paired_estimator_runner": (
                "pilot",
                "preregistered_development",
                False,
            ),
            "stage2.06_pilot_and_matrix_freeze": (
                "pilot",
                "preregistered_pilot",
                True,
            ),
            "stage2.07_main_sweep": (
                "confirmatory",
                "pilot_frozen_primary",
                False,
            ),
        }
        task_id = value["task_id"]
        if task_id not in task_contract:
            raise ValueError("task_id 不属于 Stage 2 formal B/M/R 任务")
        expected_stream, expected_basis, requires_thresholds = task_contract[task_id]
        if value["stream"] != expected_stream or value["selection_basis"] != expected_basis:
            raise ValueError("task_id、stream 与 selection_basis 不一致")
        threshold_fields = {
            "bias_margin",
            "max_corrected_nmse_ratio",
            "min_spearman",
            "min_topk_overlap",
            "max_online_cost_ratio",
        }
        if requires_thresholds:
            if not isinstance(thresholds, Mapping) or set(thresholds) != threshold_fields:
                raise ValueError("pilot_thresholds 字段集合不完整")
            numeric = {name: float(thresholds[name]) for name in threshold_fields}
            if any(not math.isfinite(item) for item in numeric.values()):
                raise ValueError("pilot_thresholds 包含非有限数")
            if (
                numeric["bias_margin"] < 0
                or numeric["max_corrected_nmse_ratio"] <= 0
                or numeric["max_online_cost_ratio"] <= 0
                or not -1 <= numeric["min_spearman"] <= 1
                or not 0 <= numeric["min_topk_overlap"] <= 1
            ):
                raise ValueError("pilot_thresholds 超出定义域")
        elif thresholds is not None:
            raise ValueError("非 pilot freeze 计划不得携带 pilot_thresholds")
        if task_id == "stage2.07_main_sweep" and (
            len(microbatch_counts) != 1 or microbatch_counts[0] <= 2
        ):
            raise ValueError("确认性计划必须冻结唯一 M_primary>2")
        for field_name in ("sampling_plan_hash", "execution_evidence_hash"):
            if not isinstance(value[field_name], str):
                raise TypeError(f"{field_name} 必须是字符串")
            _require_sha256(value[field_name], field_name=field_name)

    if schema == "stage2-pilot-matrix-freeze-v1":
        if value["scope"] != "local_fixture" or value["formal_eligible"] is not False:
            raise FormalRunRejected("S2.6_MATRIX_FORMAL_SCOPE_BLOCKED")
        if value["qualification_gate_hash"] is not None:
            raise FormalRunRejected("S2.6_MATRIX_GATE_HASH_FORBIDDEN_BEFORE_G2.4B")
        if value["pilot_draw_stream"] != "pilot" or value["confirmatory_draw_stream"] != "confirmatory":
            raise ValueError("S2.6 matrix draw stream mismatch")
        if set(value["cost_semantics"]) != {
            "scientific_equal_sample_cost", "isolated_estimator_cost", "online_training_incremental_cost"
        }:
            raise ValueError("S2.6 matrix cost semantics incomplete")
        if not isinstance(value["anchor_ids"], list) or not value["anchor_ids"]:
            raise ValueError("S2.6 matrix anchor_ids cannot be empty")
        if not isinstance(value["candidate_evaluations"], list) or not value["candidate_evaluations"]:
            raise ValueError("S2.6 matrix candidate table cannot be empty")
        if value["status"] in {"FIXTURE_FROZEN", "FORMAL_FROZEN"} and (
            value["b_primary"] is None or value["m_primary"] is None or value["r_primary"] is None
        ):
            raise ValueError("S2.6 frozen matrix requires B/M/R")

    if schema == "stage2-confirmatory-mapping-v1":
        if value["scope"] != "local_fixture" or value["formal_eligible"] is not False:
            raise FormalRunRejected("S2.6_CONFIRMATORY_MAPPING_FORMAL_SCOPE_BLOCKED")
        if value["stream"] != "confirmatory" or value["draw_id_unique"] is not True:
            raise ValueError("S2.6 confirmatory mapping stream/uniqueness invalid")
        for field_name in ("freeze_hash", "sampling_plan_hash"):
            _require_sha256(value[field_name], field_name=field_name)
        pilot_ids = string_array("pilot_draw_ids")
        confirmatory_ids = string_array("confirmatory_draw_ids", allow_empty=False)
        if set(pilot_ids).intersection(confirmatory_ids):
            raise ValueError("pilot and confirmatory draw IDs overlap")
        mappings = value["mappings"]
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("confirmatory mappings cannot be empty")

    if schema == "stage2-reference-sizing-plan-v1":
        counts = value["candidate_sample_counts"]
        if not isinstance(counts, list) or len(counts) < 2 or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in counts
        ):
            raise TypeError("candidate_sample_counts 必须至少含两个正整数")
        if tuple(counts) != tuple(sorted(set(counts))):
            raise ValueError("candidate_sample_counts 必须严格递增且无重复")
        block_size = value["block_size"]
        if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size <= 0:
            raise TypeError("block_size 必须是正整数")
        if any(item % block_size for item in counts):
            raise ValueError("candidate sample count 必须能被 block_size 整除")

    if schema == "stage2-reference-sizing-result-v1":
        validate_weighting(value["weighting_assumptions"])
        if not isinstance(value["points"], list):
            raise TypeError("points 必须是数组")
        if type(value["converged"]) is not bool:
            raise TypeError("converged 必须是显式 bool")
        if bool(value["converged"]) != (
            value["selected_sample_count_per_stream"] is not None
        ):
            raise ValueError("converged 与 selected sample count 不一致")
        for field_name in (
            "plan_hash",
            "registry_hash",
            "provider_state_digest",
            "bias_reference_hash",
            "cross_reference_hash",
            "ranking_reference_hash",
        ):
            if not isinstance(value[field_name], str):
                raise TypeError(f"{field_name} 必须是字符串")
            _require_sha256(value[field_name], field_name=field_name)

    if schema == "stage2-reference-uncertainty-v1":
        if value["estimator"] != "block_u_delete_one_jackknife":
            raise ValueError("REFERENCE_UNCERTAINTY_ESTIMATOR_UNSUPPORTED")
        confidence = value["confidence_level"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 < float(confidence) < 1:
            raise ValueError("REFERENCE_UNCERTAINTY_CONFIDENCE_INVALID")
        for field_name in ("block_count_a", "block_count_b"):
            _require_int(value[field_name], field_name, minimum=3)
        for field_name in (
            "bias_variance_hash",
            "cross_variance_hash",
            "ranking_variance_hash",
        ):
            _require_sha256(value[field_name], field_name=field_name)
        for field_name in ("trace_bias_variance", "bias_half_width_l2"):
            item = value[field_name]
            if not isinstance(item, (int, float)) or isinstance(item, bool) or not math.isfinite(float(item)) or float(item) < 0:
                raise ValueError(f"{field_name} 必须是有限非负数")

    if schema in {"stage2-reference-one-shot-plan-v1", "stage2-reference-one-shot-plan-v2"}:
        for field_name in ("reference_id", "sizing_stream", "stream_a", "stream_b"):
            if not isinstance(value[field_name], str) or not value[field_name]:
                raise TypeError(f"{field_name} 必须是非空字符串")
        for field_name in ("sizing_result_hash",):
            _require_sha256(value[field_name], field_name=field_name)
        _require_int(value["sample_count_per_stream"], "sample_count_per_stream", minimum=1)
        _require_int(value["block_size"], "block_size", minimum=1)
        if value["sample_count_per_stream"] % value["block_size"]:
            raise ValueError("one-shot sample_count_per_stream 必须可被 block_size 整除")
        if (value["sizing_stream"], value["stream_a"], value["stream_b"]) != (
            "reference_sizing", "reference_A", "reference_B"
        ) or value["one_shot"] is not True:
            raise ValueError("one-shot reference stream contract drift")
        if schema.endswith("-v2"):
            count = value["sample_count_per_stream"]
            for prefix in ("stream_a", "stream_b"):
                start = _require_int(value[f"{prefix}_draw_start_position"], f"{prefix}_draw_start_position", minimum=0)
                end = _require_int(value[f"{prefix}_draw_end_position_exclusive"], f"{prefix}_draw_end_position_exclusive", minimum=1)
                if end != start + count:
                    raise ValueError("one-shot draw segment boundary mismatch")

    if schema == "stage2-reference-one-shot-result-v1":
        for field_name in (
            "plan_hash", "sizing_result_hash", "provider_state_digest", "registry_hash",
            "bias_reference_hash", "cross_reference_hash", "ranking_reference_hash",
            "stream_a_draw_hash", "stream_b_draw_hash",
            "sequence_variance_hash",
        ):
            _require_sha256(value[field_name], field_name=field_name)
        _require_int(value["processed_sample_count_per_stream"], "processed_sample_count_per_stream", minimum=1)
        if value["status"] not in {"IN_PROGRESS", "COMPLETE", "FAILED"}:
            raise ValueError("one-shot reference status invalid")
        if value["one_shot"] is not True:
            raise ValueError("one-shot reference must be true")
        validate_weighting(value["weighting_assumptions"])
        uncertainty = value["uncertainty"]
        if not isinstance(uncertainty, Mapping) or uncertainty.get("schema_version") != "stage2-reference-uncertainty-v1":
            raise ValueError("one-shot uncertainty contract missing")
        uncertainty_payload = {key: item for key, item in uncertainty.items() if key != "artifact_hash"}
        if uncertainty.get("artifact_hash") != canonical_json_hash(uncertainty_payload):
            raise ValueError("one-shot uncertainty artifact_hash mismatch")

    if schema == "stage2-paired-wave-summary-v1":
        validate_weighting(value["weighting_assumptions"])
        reference_hashes = value["reference_hashes"]
        if not isinstance(reference_hashes, Mapping) or set(reference_hashes) != {
            "bias", "cross", "ranking"
        }:
            raise ValueError("reference_hashes 必须完整包含 bias/cross/ranking")
        for name, digest in reference_hashes.items():
            _require_sha256(digest, field_name=f"reference_hashes.{name}")
        if reference_hashes["bias"] != value["reference_hash"]:
            raise ValueError("reference_hash 必须等于 reference_hashes.bias")
        if not isinstance(value["reference_statistics"], Mapping):
            raise ValueError("reference_statistics 必须是 object")
        diagnostics = value["microbatch_diagnostics"]
        if not isinstance(diagnostics, list):
            raise ValueError("microbatch_diagnostics 必须是 array")
        replay = value["replay_evidence"]
        if not isinstance(replay, Mapping) or replay.get("schema_version") != "stage2-paired-replay-evidence-v1":
            raise ValueError("replay_evidence contract missing")
        expected = set(string_array("expected_unit_ids", allow_empty=False))
        completed = set(string_array("completed_unit_ids"))
        if not completed.issubset(expected):
            raise ValueError("completed_unit_ids 必须属于 expected_unit_ids")
        if type(value["complete"]) is not bool or value["complete"] != (
            completed == expected
        ):
            raise ValueError("complete 与 unit 集合不一致")
        costs = value["cost_statistics"]
        expected_costs = {
            "scientific_equal_sample_cost",
            "isolated_estimator_cost",
            "online_training_incremental_cost",
        }
        if not isinstance(costs, Mapping) or set(costs) != expected_costs:
            raise ValueError("cost_statistics 未完整区分三种口径")

    if schema == "stage2-paired-wave-plan-v1":
        validate_weighting(value["weighting_assumptions"])
        reference_hashes = value["reference_hashes"]
        if not isinstance(reference_hashes, Mapping) or set(reference_hashes) != {
            "bias", "cross", "ranking"
        }:
            raise ValueError("paired wave plan reference_hashes 不完整")
        for name, digest in reference_hashes.items():
            _require_sha256(digest, field_name=f"reference_hashes.{name}")
        if reference_hashes["bias"] != value["reference_hash"]:
            raise ValueError("paired wave plan reference_hash 漂移")
        mappings = value["mappings"]
        if not isinstance(mappings, list) or not mappings:
            raise ValueError("paired wave mappings 不能为空")
        unit_ids: list[str] = []
        for mapping in mappings:
            if not isinstance(mapping, Mapping) or set(mapping) != {
                "unit_id",
                "mapping_hash",
            }:
                raise ValueError("paired wave mapping 字段不匹配")
            if not isinstance(mapping["unit_id"], str) or not mapping["unit_id"]:
                raise TypeError("mapping.unit_id 必须是非空字符串")
            if not isinstance(mapping["mapping_hash"], str):
                raise TypeError("mapping.mapping_hash 必须是字符串")
            _require_sha256(mapping["mapping_hash"], field_name="mapping.mapping_hash")
            unit_ids.append(mapping["unit_id"])
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("paired wave mapping unit_id 不能重复")

    if schema == "stage2-estimator-recommendation-v1":
        string_array("required_cells", allow_empty=False)
        string_array("qualified_estimators")
        blocked = value["status"] == "BLOCKED"
        selected = value["selected_estimator"]
        counts = (
            value["batch_size"],
            value["microbatch_count"],
            value["repetitions"],
        )
        if blocked != (selected is None):
            raise ValueError("recommendation status 与 selected_estimator 不一致")
        if selected is None and any(item is not None for item in counts):
            raise ValueError("未选择 estimator 时 B/M/R 必须为 null")
        if selected is not None and any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in counts
        ):
            raise TypeError("已选择 estimator 时 B/M/R 必须是正整数")

    if schema == "stage3-endpoint-capture-v1":
        record = value["record"]
        record_fields = {
            "path_state_id",
            "source_run_id",
            "optimizer_step",
            "parameter_registry_hash",
            "pre_state",
            "parameter_post_state",
            "attempt_commit_state",
            "attempt_commit_parent_hash",
            "probe_buffer_snapshot_hash",
            "full_update_delta_hash",
            "update_sample_ids",
            "replay_verified",
            "metadata",
            "endpoint_digest",
        }
        if not isinstance(record, Mapping) or set(record) != record_fields:
            raise ValueError("endpoint record 字段集合不匹配")
        if record["replay_verified"] is not True:
            raise FormalRunRejected("ENDPOINT_REPLAY_VERIFICATION_REQUIRED")
        for field_name in (
            "parameter_registry_hash",
            "attempt_commit_parent_hash",
            "probe_buffer_snapshot_hash",
            "full_update_delta_hash",
            "endpoint_digest",
        ):
            if not isinstance(record[field_name], str):
                raise TypeError(f"record.{field_name} 必须是字符串")
            _require_sha256(record[field_name], field_name=f"record.{field_name}")

    if schema == "stage3-probe-panel-v1":
        entries = value["entries"]
        if not isinstance(entries, list) or not entries:
            raise ValueError("probe panel entries 不能为空")
        probe_ids: list[str] = []
        seen_samples: set[object] = set()
        loss_hashes: set[str] = set()
        formal_count = 0
        for entry in entries:
            entry_fields = {
                "role",
                "probe_id",
                "sample_ids",
                "content_hash",
                "loss_contract_hash",
                "effective_weight_unit",
                "metadata",
                "probe_digest",
            }
            if not isinstance(entry, Mapping) or set(entry) != entry_fields:
                raise TypeError("probe panel entry 字段集合不匹配")
            role = entry.get("role")
            probe_id = entry.get("probe_id")
            samples = entry.get("sample_ids")
            loss_hash = entry.get("loss_contract_hash")
            if role not in {"pilot", "formal", "replay"}:
                raise ValueError("probe panel role 无效")
            if not isinstance(probe_id, str) or not probe_id:
                raise TypeError("probe_id 必须是非空字符串")
            if not isinstance(samples, list) or not samples or any(
                isinstance(item, bool) or not isinstance(item, (str, int))
                for item in samples
            ):
                raise TypeError("sample_ids 必须是非空字符串/整数数组")
            if len(samples) != len(set(samples)):
                raise ValueError("单个 probe 内 sample_ids 不能重复")
            overlap = seen_samples.intersection(samples)
            if overlap:
                raise ValueError("probe panel 统计单元发生重叠")
            seen_samples.update(samples)
            probe_ids.append(probe_id)
            for hash_field in ("content_hash", "loss_contract_hash", "probe_digest"):
                digest = entry[hash_field]
                if not isinstance(digest, str):
                    raise TypeError(f"{hash_field} 必须是字符串")
                _require_sha256(digest, field_name=hash_field)
            if (
                not isinstance(entry["effective_weight_unit"], str)
                or not entry["effective_weight_unit"]
                or not isinstance(entry["metadata"], Mapping)
            ):
                raise ValueError("probe panel weight unit/metadata 无效")
            loss_hashes.add(loss_hash)
            formal_count += int(role == "formal")
        if len(set(probe_ids)) != len(probe_ids) or len(loss_hashes) != 1:
            raise ValueError("probe IDs 必须唯一且共享唯一 loss contract")
        minimum = value["minimum_formal_probes"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
            raise ValueError("minimum_formal_probes 必须是正整数")
        if value["scope"] == "formal" and (minimum < 3 or formal_count < minimum):
            raise FormalRunRejected("FORMAL_PROBE_PANEL_REQUIRES_AT_LEAST_THREE_PROBES")

    if schema == "stage3-reference-refinement-v1":
        if type(value["converged"]) is not bool:
            raise TypeError("converged 必须是显式 bool")
        if type(value["convergence_defined"]) is not bool:
            raise TypeError("convergence_defined 必须是显式 bool")
        if value["converged"] and not value["convergence_defined"]:
            raise ValueError("undefined convergence 不能标记为 converged")
        selected = value["selected_rule_hash"]
        if value["converged"] != (selected is not None):
            raise ValueError("converged 与 selected_rule_hash 不一致")
        if selected is not None:
            if not isinstance(selected, str):
                raise TypeError("selected_rule_hash 必须是字符串或 null")
            _require_sha256(selected, field_name="selected_rule_hash")
        if not isinstance(value["completed_levels"], list):
            raise TypeError("completed_levels 必须是数组")

    if schema == "stage3-quadrature-recommendation-v1":
        passing = string_array("passing_rules")
        string_array("required_unit_ids", allow_empty=False)
        default = value["default_rule"]
        if (value["status"] == "BLOCKED") != (default is None):
            raise ValueError("quadrature status 与 default_rule 不一致")
        if default is not None and default not in passing:
            raise ValueError("default_rule 必须属于 passing_rules")
        thresholds = value["thresholds"]
        if not isinstance(thresholds, Mapping):
            raise TypeError("thresholds 必须是 object")
        if canonical_json_hash(thresholds) != value["thresholds_hash"]:
            raise ValueError("thresholds_hash 与 thresholds 不一致")
        for field_name in ("gate_evaluation_hash", "provenance_hash"):
            if field_name in value and value[field_name] is not None:
                field_value = value[field_name]
                if not isinstance(field_value, str):
                    raise TypeError(f"{field_name} 必须是字符串或缺省")
                _require_sha256(field_value, field_name=field_name)

    if "execution_evidence_hash" in value:
        evidence_hash = value["execution_evidence_hash"]
        if not isinstance(evidence_hash, str):
            raise TypeError("execution_evidence_hash 必须是字符串")
        _require_sha256(evidence_hash, field_name="execution_evidence_hash")
    # formal experiment plan 是运行前的授权输入，不是等待本阶段 Gate 资格化的科学
    # 输出，因此没有 qualification_gate_hash；其 FROZEN/formal_eligible 已在专属
    # 分支中复核。其余带 scope 的 Stage2/3 结果仍执行统一 Gate 边界。
    if "scope" in value and schema not in {
        "stage2-formal-experiment-plan-v1",
        "stage3-gate-evaluation-v1",
        "stage3-finalization-v1",
        "stage3-g36-publication-v1",
    }:
        scope = value["scope"]
        pilot_scoped_artifacts = {
            "stage3-endpoint-capture-v1",
            "stage3-probe-panel-v1",
            "stage3-probe-plan-v1",
        }
        allowed_scopes = {"local_fixture", "formal"}
        if schema in pilot_scoped_artifacts:
            allowed_scopes.add("pilot")
        if scope not in allowed_scopes:
            raise ValueError("artifact scope 不受支持")
        if type(value["formal_eligible"]) is not bool:
            raise TypeError("formal_eligible 必须是显式 bool")
        formal_eligible = value["formal_eligible"]
        gate_hash = value["qualification_gate_hash"]
        if formal_eligible:
            if scope != "formal" or not isinstance(gate_hash, str):
                raise FormalRunRejected("FORMAL_ARTIFACT_QUALIFICATION_INCOMPLETE")
            _require_sha256(gate_hash, field_name="qualification_gate_hash")
        elif gate_hash is not None:
            raise FormalRunRejected("UNQUALIFIED_ARTIFACT_CANNOT_CARRY_GATE_HASH")

        candidate_only = schema in {
            "stage2-reference-sizing-result-v1",
            "stage2-paired-wave-summary-v1",
            "stage2-estimator-recommendation-v1",
            "stage3-reference-refinement-v1",
        }
        if candidate_only and formal_eligible:
            raise FormalRunRejected(f"{schema} 只能表示待 Gate 验收的候选")
        if schema == "stage3-quadrature-recommendation-v1":
            if formal_eligible != (value["status"] == "QUALIFIED"):
                raise FormalRunRejected("QUADRATURE_QUALIFICATION_STATUS_MISMATCH")
            if formal_eligible and (
                not isinstance(value.get("gate_evaluation_hash"), str)
                or not isinstance(value.get("provenance_hash"), str)
            ):
                raise FormalRunRejected(
                    "FORMAL_RECOMMENDATION_REQUIRES_GATE_EVALUATION_AND_PROVENANCE"
                )
            if formal_eligible:
                raise FormalRunRejected(
                    "QUALIFIED_RECOMMENDATION_REQUIRES_AUTHORITY_AWARE_VALIDATION"
                )

    from .artifacts import ValidatedArtifact

    return ValidatedArtifact(schema.removesuffix("-v1"), supplied)
