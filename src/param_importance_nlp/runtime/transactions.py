"""单步更新的两种后状态边界。

``parameter_post_state`` 在 optimizer 成功修改参数后立即形成；scheduler、AMP
scaler 与 RNG 随后提交，得到 ``attempt_commit_state``。两者不能用同一个
“post state”名称混写，否则崩溃恢复会把“参数已更新、运行控制状态未提交”误认
为完整 step。本模块提供纯状态机，实际字节持久化交给 ``CheckpointStore``。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class StepPhase(StrEnum):
    GRADIENT_READY = "GRADIENT_READY"
    PARAMETER_POST_STATE = "PARAMETER_POST_STATE"
    ATTEMPT_COMMIT_STATE = "ATTEMPT_COMMIT_STATE"
    SKIPPED = "SKIPPED"


def _require_sha256(value: str, *, field: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"STEP_TRANSACTION_INVALID_HASH:{field}")
    return value


@dataclass(frozen=True, slots=True)
class StepTransaction:
    """一次 optimizer 尝试的不可变状态快照。"""

    global_step: int
    attempt_index: int
    phase: StepPhase = StepPhase.GRADIENT_READY
    parameter_post_state_hash: str | None = None
    attempt_commit_state_hash: str | None = None
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if self.global_step < 0 or self.attempt_index < 0:
            raise ValueError("STEP_TRANSACTION_NEGATIVE_INDEX")

    def mark_parameter_post(self, state_hash: str) -> "StepTransaction":
        if self.phase is not StepPhase.GRADIENT_READY:
            raise ValueError(f"STEP_PARAMETER_POST_FROM_INVALID_PHASE:{self.phase}")
        return replace(
            self,
            phase=StepPhase.PARAMETER_POST_STATE,
            parameter_post_state_hash=_require_sha256(
                state_hash, field="parameter_post_state_hash"
            ),
        )

    def commit_attempt(self, state_hash: str) -> "StepTransaction":
        if self.phase is not StepPhase.PARAMETER_POST_STATE:
            raise ValueError(f"STEP_ATTEMPT_COMMIT_FROM_INVALID_PHASE:{self.phase}")
        validated = _require_sha256(state_hash, field="attempt_commit_state_hash")
        if validated == self.parameter_post_state_hash:
            raise ValueError("STEP_ATTEMPT_COMMIT_MUST_INCLUDE_CONTROL_STATE_CHANGE")
        return replace(
            self,
            phase=StepPhase.ATTEMPT_COMMIT_STATE,
            attempt_commit_state_hash=validated,
        )

    def skip(self, reason: str) -> "StepTransaction":
        if self.phase is not StepPhase.GRADIENT_READY:
            raise ValueError(f"STEP_SKIP_FROM_INVALID_PHASE:{self.phase}")
        if not reason.strip():
            raise ValueError("STEP_SKIP_REASON_REQUIRED")
        return replace(self, phase=StepPhase.SKIPPED, skip_reason=reason)


__all__ = ["StepPhase", "StepTransaction"]
