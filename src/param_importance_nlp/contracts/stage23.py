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

    未资格化产物可以是 ``formal`` scope 的候选证据，但 ``formal_eligible`` 必须为
    ``False``；只有携带本阶段可接受 Gate 的摘要时才能为真。
    """

    scope: str
    formal_eligible: bool = False
    qualification_gate_hash: str | None = None

    def __post_init__(self) -> None:
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("ArtifactQualification.scope 不受支持")
        if type(self.formal_eligible) is not bool:
            raise TypeError("formal_eligible 必须是显式 bool")
        if self.qualification_gate_hash is not None:
            _require_sha256(
                self.qualification_gate_hash, field_name="qualification_gate_hash"
            )
        if self.scope == "local_fixture" and self.formal_eligible:
            raise FormalRunRejected("LOCAL_FIXTURE_CANNOT_BE_FORMAL_ELIGIBLE")
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
    "stage2-paired-wave-summary-v1": {
        "schema_version",
        "wave_id",
        "registry_hash",
        "reference_hash",
        "expected_unit_ids",
        "completed_unit_ids",
        "complete",
        "status",
        "method_statistics",
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
        "reasons",
        "artifact_hash",
    },
}


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
    if set(value) != required:
        raise ValueError(
            f"{schema} 字段集合不匹配：missing={sorted(required-set(value))}, "
            f"extra={sorted(set(value)-required)}"
        )
    supplied = value.get("artifact_hash")
    if not isinstance(supplied, str):
        raise TypeError("artifact_hash 必须是字符串")
    _require_sha256(supplied, field_name="artifact_hash")
    payload = {name: item for name, item in value.items() if name != "artifact_hash"}
    if canonical_json_hash(payload) != supplied:
        raise ValueError(f"{schema} artifact_hash 与完整 wire object 不一致")

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

    if schema == "stage2-paired-wave-summary-v1":
        validate_weighting(value["weighting_assumptions"])
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
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise TypeError("probe panel entry 必须是 object")
            probe_id = entry.get("probe_id")
            samples = entry.get("sample_ids")
            loss_hash = entry.get("loss_contract_hash")
            if not isinstance(probe_id, str) or not probe_id:
                raise TypeError("probe_id 必须是非空字符串")
            if not isinstance(samples, list) or not samples:
                raise TypeError("sample_ids 必须是非空数组")
            overlap = seen_samples.intersection(samples)
            if overlap:
                raise ValueError("probe panel 统计单元发生重叠")
            seen_samples.update(samples)
            probe_ids.append(probe_id)
            if not isinstance(loss_hash, str):
                raise TypeError("loss_contract_hash 必须是字符串")
            _require_sha256(loss_hash, field_name="loss_contract_hash")
            loss_hashes.add(loss_hash)
        if len(set(probe_ids)) != len(probe_ids) or len(loss_hashes) != 1:
            raise ValueError("probe IDs 必须唯一且共享唯一 loss contract")

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

    if "execution_evidence_hash" in value:
        evidence_hash = value["execution_evidence_hash"]
        if not isinstance(evidence_hash, str):
            raise TypeError("execution_evidence_hash 必须是字符串")
        _require_sha256(evidence_hash, field_name="execution_evidence_hash")
    # formal experiment plan 是运行前的授权输入，不是等待本阶段 Gate 资格化的科学
    # 输出，因此没有 qualification_gate_hash；其 FROZEN/formal_eligible 已在专属
    # 分支中复核。其余带 scope 的 Stage2/3 结果仍执行统一 Gate 边界。
    if "scope" in value and schema != "stage2-formal-experiment-plan-v1":
        scope = value["scope"]
        if scope not in {"local_fixture", "formal"}:
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

    from .artifacts import ValidatedArtifact

    return ValidatedArtifact(schema.removesuffix("-v1"), supplied)
