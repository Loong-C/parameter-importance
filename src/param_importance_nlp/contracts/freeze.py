"""Stage 0–9 版本化合同冻结 artifact。

冻结记录只说明“哪个版本的公式、schema 和源文件已经不可变”，不等价于阶段完成。
``scope=local_fixture`` 的记录即使是 ``FROZEN`` 也永远不能为正式运行提供资格；
依赖真实 pilot 的预算、阈值或方法尚未决定时使用 ``UNFROZEN``/``BLOCKED``，并给出
原因，不能填入猜测值后伪装成冻结。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any, Final

from .errors import FreezeContractError
from .jsonio import JSONValue, canonical_json_hash
from .status import ContractState, SHA256_PATTERN, validate_gate_id


CONTRACT_ID_PATTERN: Final = re.compile(
    r"^stage(?P<stage>[0-9])\.contract\.[a-z0-9]+(?:[._-][a-z0-9]+)*$"
)


def _timestamp(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise FreezeContractError(f"{field} 必须是带时区 ISO-8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FreezeContractError(f"{field} 不是有效 ISO-8601 时间") from error
    if parsed.tzinfo is None:
        raise FreezeContractError(f"{field} 必须包含时区")


def _text(value: Any, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise FreezeContractError(f"{field} 必须是非空字符串")
    if any(ord(character) < 32 for character in value):
        raise FreezeContractError(f"{field} 不能包含控制字符")
    return value


def _hash_mapping(value: dict[str, str], *, field: str, required: bool) -> dict[str, str]:
    if not isinstance(value, dict):
        raise FreezeContractError(f"{field} 必须是 object")
    if required and not value:
        raise FreezeContractError(f"FROZEN 合同的 {field} 不能为空")
    normalized: dict[str, str] = {}
    for key, digest in value.items():
        _text(key, field=f"{field} key", maximum=512)
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise FreezeContractError(f"{field}.{key} 必须是小写 SHA-256")
        normalized[key] = digest
    return dict(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class ContractFreeze:
    """一个阶段、一个 scope 下不可变合同版本的内容寻址记录。"""

    contract_id: str
    stage: int
    scope: str
    state: ContractState
    formula_version: str
    config_hash: str
    schema_hashes: dict[str, str]
    source_hashes: dict[str, str]
    required_gate_ids: tuple[str, ...] = ()
    frozen_at: str | None = None
    reason: str | None = None
    decision_ref: str | None = None
    schema_version: str = "contract-freeze-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "contract-freeze-v1":
            raise FreezeContractError("ContractFreeze.schema_version 必须是 contract-freeze-v1")
        if isinstance(self.stage, bool) or not isinstance(self.stage, int) or not 0 <= self.stage <= 9:
            raise FreezeContractError("stage 必须是 0..9 的整数")
        if not isinstance(self.contract_id, str):
            raise FreezeContractError("contract_id 必须是字符串")
        matched = CONTRACT_ID_PATTERN.fullmatch(self.contract_id)
        if matched is None or int(matched.group("stage")) != self.stage:
            raise FreezeContractError("contract_id 必须是与 stage 一致的 stageN.contract.<version>")
        if self.scope not in {"local_fixture", "formal"}:
            raise FreezeContractError("scope 必须是 local_fixture 或 formal")
        try:
            normalized_state = ContractState(self.state)
        except (TypeError, ValueError) as error:
            raise FreezeContractError(f"未知 ContractState：{self.state!r}") from error
        object.__setattr__(self, "state", normalized_state)
        _text(self.formula_version, field="formula_version", maximum=256)
        if not isinstance(self.config_hash, str) or SHA256_PATTERN.fullmatch(self.config_hash) is None:
            raise FreezeContractError("config_hash 必须是小写 SHA-256")
        object.__setattr__(
            self,
            "schema_hashes",
            _hash_mapping(
                self.schema_hashes,
                field="schema_hashes",
                required=normalized_state is ContractState.FROZEN,
            ),
        )
        object.__setattr__(
            self,
            "source_hashes",
            _hash_mapping(
                self.source_hashes,
                field="source_hashes",
                required=normalized_state is ContractState.FROZEN,
            ),
        )
        gates = tuple(
            validate_gate_id(gate_id, stage=self.stage)
            for gate_id in self.required_gate_ids
        )
        if len(gates) != len(set(gates)):
            raise FreezeContractError("required_gate_ids 不能重复")
        object.__setattr__(self, "required_gate_ids", gates)
        if self.decision_ref is not None:
            _text(self.decision_ref, field="decision_ref")
            if "?" in self.decision_ref:
                raise FreezeContractError("decision_ref 必须是稳定引用")

        if normalized_state is ContractState.FROZEN:
            if self.frozen_at is None or self.reason is not None:
                raise FreezeContractError("FROZEN 合同必须有 frozen_at 且不能有 reason")
            _timestamp(self.frozen_at, field="frozen_at")
            if self.scope == "formal" and not gates:
                raise FreezeContractError(
                    "formal FROZEN 合同必须声明至少一个 required Gate"
                )
        else:
            if self.frozen_at is not None or self.reason is None:
                raise FreezeContractError(
                    f"{normalized_state.value} 合同必须有 reason 且 frozen_at=null"
                )
            _text(self.reason, field="reason")

    @property
    def formal_eligible(self) -> bool:
        """只有 formal scope 且已冻结的合同可进入正式证据链。"""

        return self.scope == "formal" and self.state is ContractState.FROZEN

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "contract_id": self.contract_id,
            "stage": self.stage,
            "scope": self.scope,
            "state": self.state.value,
            "formal_eligible": self.formal_eligible,
            "formula_version": self.formula_version,
            "config_hash": self.config_hash,
            "schema_hashes": dict(self.schema_hashes),
            "source_hashes": dict(self.source_hashes),
            "required_gate_ids": list(self.required_gate_ids),
            "frozen_at": self.frozen_at,
            "reason": self.reason,
            "decision_ref": self.decision_ref,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "ContractFreeze":
        required = {
            "schema_version",
            "contract_id",
            "stage",
            "scope",
            "state",
            "formal_eligible",
            "formula_version",
            "config_hash",
            "schema_hashes",
            "source_hashes",
            "required_gate_ids",
            "frozen_at",
            "reason",
            "decision_ref",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise FreezeContractError(
                f"ContractFreeze 字段错误：missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if not isinstance(value["schema_hashes"], dict) or not isinstance(value["source_hashes"], dict):
            raise FreezeContractError("schema_hashes/source_hashes 必须是 object")
        if not isinstance(value["required_gate_ids"], list):
            raise FreezeContractError("required_gate_ids 必须是数组")
        freeze = cls(
            contract_id=value["contract_id"],
            stage=value["stage"],
            scope=value["scope"],
            state=value["state"],
            formula_version=value["formula_version"],
            config_hash=value["config_hash"],
            schema_hashes=dict(value["schema_hashes"]),
            source_hashes=dict(value["source_hashes"]),
            required_gate_ids=tuple(value["required_gate_ids"]),
            frozen_at=value["frozen_at"],
            reason=value["reason"],
            decision_ref=value["decision_ref"],
            schema_version=value["schema_version"],
        )
        if value["formal_eligible"] is not freeze.formal_eligible:
            raise FreezeContractError("formal_eligible 与 scope/state 派生值不一致")
        if value["artifact_hash"] != freeze.artifact_hash:
            raise FreezeContractError("ContractFreeze.artifact_hash 与内容不一致")
        return freeze
