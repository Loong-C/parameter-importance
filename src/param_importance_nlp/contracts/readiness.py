"""正式入口的 fail-closed 证据资格判定。

本模块不执行训练，也不推断缺失 artifact。它只把 resolved config、阶段 freeze、
EstimatorDecision 摘要和 GateRecord 组合成确定性判断。任何必需证据缺失、scope 为
``local_fixture``、状态非冻结、哈希不匹配或 Gate 过期都会得到明确 reason code；
:func:`require_formal_readiness` 随后抛出 ``FormalRunRejected``。

EstimatorDecision 为了避免合同层反向依赖 experiments，可传 ``Mapping``，也可传
带 ``to_dict()`` 的对象。其 wire object 至少需要 ``scope=formal``、冻结/已选择状态、
``selected_estimator``、``artifact_hash``；artifact hash 对除自身以外的完整对象计算。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import re
from typing import Any, Protocol, runtime_checkable

from .artifacts import validate_estimator_decision_artifact
from .config import ResolvedConfig
from .errors import FormalRunRejected, FreezeContractError, GateContractError
from .freeze import ContractFreeze
from .jsonio import JSONValue, canonical_json_hash
from .status import GateRecord, GateStatus, SHA256_PATTERN, validate_gate_id


@runtime_checkable
class DictArtifact(Protocol):
    """跨模块 artifact 的最小结构协议。"""

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class FormalReadiness:
    """正式入口资格及全部拒绝原因；不携带本机 PASS 的隐式升级。"""

    formal_eligible: bool
    reasons: tuple[str, ...]
    verified_stages: tuple[int, ...]
    verified_gate_ids: tuple[str, ...]
    estimator_decision_hash: str | None
    schema_version: str = "formal-readiness-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "formal-readiness-v1":
            raise FormalRunRejected("FormalReadiness.schema_version 必须是 formal-readiness-v1")
        if self.formal_eligible and self.reasons:
            raise FormalRunRejected("formal_eligible=true 时 reasons 必须为空")
        if not self.formal_eligible and not self.reasons:
            raise FormalRunRejected("formal_eligible=false 时必须给出拒绝原因")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "formal_eligible": self.formal_eligible,
            "reasons": list(self.reasons),
            "verified_stages": list(self.verified_stages),
            "verified_gate_ids": list(self.verified_gate_ids),
            "estimator_decision_hash": self.estimator_decision_hash,
        }


def _mapping(value: Any, *, field: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, DictArtifact):
        mapped = value.to_dict()
        if isinstance(mapped, dict):
            return mapped
    raise FormalRunRejected(f"{field} 必须是 Mapping 或提供 to_dict()")


def _coerce_freeze(value: Any) -> ContractFreeze:
    if isinstance(value, ContractFreeze):
        return value
    return ContractFreeze.from_mapping(_mapping(value, field="freeze"))


def _coerce_gate(value: Any) -> GateRecord:
    if isinstance(value, GateRecord):
        return value
    return GateRecord.from_mapping(_mapping(value, field="gate_record"))


def _validate_decision(value: Any) -> tuple[dict[str, Any] | None, list[str], str | None]:
    if value is None:
        return None, ["MISSING_ESTIMATOR_DECISION"], None
    try:
        decision = _mapping(value, field="estimator_decision")
    except FormalRunRejected:
        return None, ["INVALID_ESTIMATOR_DECISION_SHAPE"], None
    reasons: list[str] = []
    try:
        validate_estimator_decision_artifact(decision)
    except (TypeError, ValueError):
        reasons.append("INVALID_ESTIMATOR_DECISION_ARTIFACT")
    else:
        try:
            validate_estimator_decision_artifact(decision, require_formal=True)
        except (TypeError, ValueError) as error:
            code = str(error)
            reasons.append(
                code
                if code.startswith("FORMAL_DECISION_")
                else "ESTIMATOR_DECISION_NOT_FORMAL_READY"
            )
    if decision.get("schema_version") != "estimator-decision-v1":
        reasons.append("ESTIMATOR_DECISION_SCHEMA_MISMATCH")
    if not isinstance(decision.get("decision_id"), str) or not decision["decision_id"]:
        reasons.append("ESTIMATOR_DECISION_MISSING_ID")
    if decision.get("scope") != "formal":
        reasons.append("ESTIMATOR_DECISION_NOT_FORMAL")
    state = decision.get("state", decision.get("decision_status", decision.get("status")))
    if not isinstance(state, str) or state.upper() not in {
        "FROZEN",
        "SELECTED",
        "PASS",
        "READY",
    }:
        reasons.append("ESTIMATOR_DECISION_NOT_FROZEN")
    estimator = decision.get("selected_estimator", decision.get("estimator_name"))
    if not isinstance(estimator, str) or not estimator:
        reasons.append("ESTIMATOR_DECISION_MISSING_SELECTION")
    elif estimator not in {"u", "weighted_u", "double"}:
        # Stage 2 决策失败时禁止静默回退 raw。
        reasons.append("ESTIMATOR_DECISION_FORBIDDEN_SELECTION")
    gate_status = decision.get("gate_status")
    try:
        normalized_gate_status = GateStatus(gate_status)
    except (TypeError, ValueError):
        reasons.append("ESTIMATOR_DECISION_INVALID_GATE_STATUS")
    else:
        if normalized_gate_status not in {
            GateStatus.PASS,
            GateStatus.CONDITIONALLY_ACCEPTED,
        }:
            reasons.append("ESTIMATOR_DECISION_GATE_NOT_ACCEPTABLE")
    gate_id = decision.get("gate_id")
    try:
        validated_gate_id = validate_gate_id(gate_id)
    except (GateContractError, TypeError):
        reasons.append("ESTIMATOR_DECISION_INVALID_GATE_ID")
    else:
        if not validated_gate_id.startswith("stage2.G"):
            reasons.append("ESTIMATOR_DECISION_GATE_NOT_STAGE2")
    artifact_hash = decision.get("artifact_hash")
    if not isinstance(artifact_hash, str) or SHA256_PATTERN.fullmatch(artifact_hash) is None:
        reasons.append("ESTIMATOR_DECISION_INVALID_HASH")
        artifact_hash = None
    else:
        payload = dict(decision)
        payload.pop("artifact_hash", None)
        if canonical_json_hash(payload) != artifact_hash:
            reasons.append("ESTIMATOR_DECISION_HASH_MISMATCH")
    return decision, reasons, artifact_hash


def evaluate_formal_readiness(
    config: ResolvedConfig,
    *,
    freezes: Iterable[ContractFreeze | Mapping[str, Any]],
    estimator_decision: Any,
    gate_records: Iterable[GateRecord | Mapping[str, Any]] = (),
    required_stages: Iterable[int] = (0, 1, 2, 3),
    required_gate_ids: Iterable[str] = (),
    at: datetime | None = None,
) -> FormalReadiness:
    """计算正式入口资格，不抛出“证据不足”类异常。

    无法解析的 freeze/Gate 也被转换成 reason code，因此上层可以一次展示全部缺口。
    ``required_stages`` 默认对应 Stage 4 正式训练必须消费的 Stage 0–3 合同。
    """

    if not isinstance(config, ResolvedConfig):
        raise TypeError("config 必须是 ResolvedConfig")
    instant = at or datetime.now(timezone.utc)
    if instant.tzinfo is None:
        raise ValueError("at 必须包含时区")
    reasons: list[str] = []
    identity = config.section("identity")
    importance = config.section("importance")
    if identity["run_intent"] != "formal" or not config.formal_eligible:
        reasons.append("CONFIG_NOT_FORMAL")

    normalized_stages: list[int] = []
    for stage in required_stages:
        if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9:
            raise ValueError("required_stages 只能包含 0..9 整数")
        if stage not in normalized_stages:
            normalized_stages.append(stage)

    freeze_by_stage: dict[int, ContractFreeze] = {}
    invalid_freezes = 0
    for raw_freeze in freezes:
        try:
            freeze = _coerce_freeze(raw_freeze)
        except (FormalRunRejected, FreezeContractError, TypeError, ValueError):
            invalid_freezes += 1
            continue
        if freeze.stage in freeze_by_stage:
            reasons.append(f"DUPLICATE_FREEZE_STAGE_{freeze.stage}")
        else:
            freeze_by_stage[freeze.stage] = freeze
    if invalid_freezes:
        reasons.append("INVALID_FREEZE_ARTIFACT")

    verified_stages: list[int] = []
    required_gates: set[str] = set()
    for gate_id in required_gate_ids:
        required_gates.add(validate_gate_id(gate_id))
    for stage in normalized_stages:
        freeze = freeze_by_stage.get(stage)
        if freeze is None:
            reasons.append(f"MISSING_FREEZE_STAGE_{stage}")
            continue
        if not freeze.formal_eligible:
            reasons.append(f"FREEZE_STAGE_{stage}_NOT_FORMAL_FROZEN")
            continue
        if freeze.config_hash != config.config_hash:
            reasons.append(f"FREEZE_STAGE_{stage}_CONFIG_HASH_MISMATCH")
            continue
        verified_stages.append(stage)
        required_gates.update(freeze.required_gate_ids)

    decision, decision_reasons, decision_hash = _validate_decision(estimator_decision)
    reasons.extend(decision_reasons)
    decision_gate_id: str | None = None
    if decision is not None:
        candidate_gate_id = decision.get("gate_id")
        if isinstance(candidate_gate_id, str):
            try:
                decision_gate_id = validate_gate_id(candidate_gate_id)
            except GateContractError:
                pass
            else:
                required_gates.add(decision_gate_id)

    gate_by_id: dict[str, GateRecord] = {}
    invalid_gates = 0
    for raw_gate in gate_records:
        try:
            gate = _coerce_gate(raw_gate)
        except (FormalRunRejected, GateContractError, TypeError, ValueError):
            invalid_gates += 1
            continue
        if gate.gate_id in gate_by_id:
            reasons.append(f"DUPLICATE_GATE_{gate.gate_id}")
        else:
            gate_by_id[gate.gate_id] = gate
    if invalid_gates:
        reasons.append("INVALID_GATE_ARTIFACT")

    verified_gates: list[str] = []
    for gate_id in sorted(required_gates):
        gate = gate_by_id.get(gate_id)
        if gate is None:
            reasons.append(f"MISSING_GATE_{gate_id}")
            continue
        effective = gate.effective_status(at=instant)
        if effective not in {GateStatus.PASS, GateStatus.CONDITIONALLY_ACCEPTED}:
            reasons.append(f"GATE_{gate_id}_{effective.value}")
            continue
        verified_gates.append(gate_id)

    if decision_gate_id is not None:
        decision_gate = gate_by_id.get(decision_gate_id)
        if decision_gate is not None:
            effective = decision_gate.effective_status(at=instant)
            declared = decision.get("gate_status") if decision is not None else None
            if effective.value != declared:
                reasons.append("ESTIMATOR_DECISION_GATE_STATUS_MISMATCH")
    configured_ref = importance["estimator_decision_ref"]
    if configured_ref is None:
        reasons.append("CONFIG_MISSING_ESTIMATOR_DECISION_REF")
    elif decision is not None and decision_hash is not None:
        if SHA256_PATTERN.fullmatch(configured_ref):
            if configured_ref != decision_hash:
                reasons.append("CONFIG_ESTIMATOR_DECISION_HASH_MISMATCH")
        elif decision.get("artifact_ref") != configured_ref:
            reasons.append("CONFIG_ESTIMATOR_DECISION_REF_MISMATCH")

    # 保持顺序稳定并去重，便于测试与机器 diff。
    unique_reasons = tuple(dict.fromkeys(reasons))
    return FormalReadiness(
        formal_eligible=not unique_reasons,
        reasons=unique_reasons,
        verified_stages=tuple(sorted(verified_stages)),
        verified_gate_ids=tuple(sorted(verified_gates)),
        estimator_decision_hash=decision_hash,
    )


def require_formal_readiness(
    config: ResolvedConfig,
    **kwargs: Any,
) -> FormalReadiness:
    """要求正式资格；缺任一 freeze/decision/Gate 时立即拒绝入口。"""

    result = evaluate_formal_readiness(config, **kwargs)
    if not result.formal_eligible:
        raise FormalRunRejected(
            "正式运行被拒绝：" + ", ".join(result.reasons)
        )
    return result
