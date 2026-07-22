"""experiment、run、attempt 与 session 的分层身份合同。

``experiment_id`` 只由语义配置决定，不包含时间；``run_id`` 表示一次独立逻辑轨迹，
加入创建时间和防碰撞码；恢复保持 run 不变并增加 attempt。当前非弹性运行规定每个
attempt 恰有一个 ``s001`` session，未来若启用弹性，需要升级 schema 才能一对多。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import re
import secrets
from typing import Any, Final

from .errors import IdentityContractError
from .jsonio import JSONValue, canonical_json_hash


EXPERIMENT_ID_PATTERN: Final = re.compile(
    r"^exp-s[0-9]-[a-z0-9]+(?:-[a-z0-9]+)*-[0-9a-f]{16}$"
)
RUN_ID_PATTERN: Final = re.compile(
    r"^run-[0-9a-f]{16}-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$"
)
SESSION_ID_PATTERN: Final = re.compile(
    r"^(?P<run>run-[0-9a-f]{16}-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8})"
    r"\.a(?P<attempt>[0-9]{4,})\.s001$"
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")


def _validate_timestamp(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise IdentityContractError(f"{field} 必须是带时区 ISO-8601 字符串")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise IdentityContractError(f"{field} 不是有效 ISO-8601 时间") from error
    if parsed.tzinfo is None:
        raise IdentityContractError(f"{field} 必须包含时区")
    return parsed


def _canonical_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise IdentityContractError("时间必须包含时区")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _slug(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IdentityContractError("task 必须是非空字符串")
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    if not slug:
        raise IdentityContractError("task 无法转换为安全 experiment slug")
    return slug[:48].rstrip("-")


def _validate_optional_reference(value: str | None, *, field: str) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value or len(value) > 512:
        raise IdentityContractError(f"{field} 必须是非空稳定引用")
    if "?" in value or any(ord(character) < 32 for character in value):
        raise IdentityContractError(f"{field} 不能包含查询参数或控制字符")


def derive_experiment_id(
    *,
    stage: int,
    task: str,
    model_identity: str,
    route: str,
    master_seed: int,
    config_hash: str,
) -> str:
    """根据语义字段生成不含时间的 experiment ID。"""

    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9:
        raise IdentityContractError("stage 必须是 0..9 的整数")
    for field, value in (("model_identity", model_identity), ("route", route)):
        if not isinstance(value, str) or not value:
            raise IdentityContractError(f"{field} 必须是非空字符串")
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or not 0 <= master_seed < 2**63:
        raise IdentityContractError("master_seed 必须位于 [0, 2**63)")
    if not isinstance(config_hash, str) or SHA256_PATTERN.fullmatch(config_hash) is None:
        raise IdentityContractError("config_hash 必须是小写 SHA-256")
    payload = {
        "stage": stage,
        "task": task,
        "model_identity": model_identity,
        "route": route,
        "master_seed": master_seed,
        "config_hash": config_hash,
    }
    digest = canonical_json_hash(payload)[:16]
    return f"exp-s{stage}-{_slug(task)}-{digest}"


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """一条运行轨迹在 experiment/run/attempt/session 四层的完整身份。"""

    experiment_id: str
    run_id: str
    attempt_id: int
    session_id: str
    run_created_at: str
    attempt_started_at: str
    parent_experiment_id: str | None = None
    input_run_id: str | None = None
    input_checkpoint_id: str | None = None
    schema_version: str = "run-identity-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "run-identity-v1":
            raise IdentityContractError("RunIdentity.schema_version 必须是 run-identity-v1")
        if not isinstance(self.experiment_id, str) or EXPERIMENT_ID_PATTERN.fullmatch(
            self.experiment_id
        ) is None:
            raise IdentityContractError("experiment_id 不符合稳定格式")
        if not isinstance(self.run_id, str) or RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise IdentityContractError("run_id 不符合稳定格式")
        if isinstance(self.attempt_id, bool) or not isinstance(self.attempt_id, int) or self.attempt_id < 1:
            raise IdentityContractError("attempt_id 必须从 1 开始单调增加")
        matched = SESSION_ID_PATTERN.fullmatch(self.session_id)
        if matched is None:
            raise IdentityContractError("session_id 必须是 <run_id>.aNNNN.s001")
        if matched.group("run") != self.run_id or int(matched.group("attempt")) != self.attempt_id:
            raise IdentityContractError("session_id 与 run_id/attempt_id 不一致")
        run_created = _validate_timestamp(self.run_created_at, field="run_created_at")
        attempt_started = _validate_timestamp(self.attempt_started_at, field="attempt_started_at")
        if attempt_started < run_created:
            raise IdentityContractError("attempt_started_at 不能早于 run_created_at")
        _validate_optional_reference(self.parent_experiment_id, field="parent_experiment_id")
        _validate_optional_reference(self.input_run_id, field="input_run_id")
        _validate_optional_reference(self.input_checkpoint_id, field="input_checkpoint_id")

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        created_at: datetime,
        collision_code: str | None = None,
        parent_experiment_id: str | None = None,
        input_run_id: str | None = None,
        input_checkpoint_id: str | None = None,
    ) -> "RunIdentity":
        """创建新 run 的首个 attempt；目录原子创建由 runtime 层负责。"""

        if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
            raise IdentityContractError("experiment_id 不符合稳定格式")
        code = collision_code if collision_code is not None else secrets.token_hex(4)
        if not isinstance(code, str) or re.fullmatch(r"[0-9a-f]{8}", code) is None:
            raise IdentityContractError("collision_code 必须是 8 位小写十六进制")
        timestamp = _canonical_timestamp(created_at)
        compact = timestamp.replace("-", "").replace(":", "")
        experiment_digest = experiment_id.rsplit("-", 1)[-1]
        run_id = f"run-{experiment_digest}-{compact}-{code}"
        return cls(
            experiment_id=experiment_id,
            run_id=run_id,
            attempt_id=1,
            session_id=f"{run_id}.a0001.s001",
            run_created_at=timestamp,
            attempt_started_at=timestamp,
            parent_experiment_id=parent_experiment_id,
            input_run_id=input_run_id,
            input_checkpoint_id=input_checkpoint_id,
        )

    def next_attempt(
        self,
        *,
        started_at: datetime,
        input_checkpoint_id: str,
    ) -> "RunIdentity":
        """为同一逻辑 run 创建恢复 attempt，并绑定已验证 checkpoint。"""

        _validate_optional_reference(input_checkpoint_id, field="input_checkpoint_id")
        next_id = self.attempt_id + 1
        timestamp = _canonical_timestamp(started_at)
        return replace(
            self,
            attempt_id=next_id,
            session_id=f"{self.run_id}.a{next_id:04d}.s001",
            attempt_started_at=timestamp,
            input_checkpoint_id=input_checkpoint_id,
        )

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "session_id": self.session_id,
            "run_created_at": self.run_created_at,
            "attempt_started_at": self.attempt_started_at,
            "parent_experiment_id": self.parent_experiment_id,
            "input_run_id": self.input_run_id,
            "input_checkpoint_id": self.input_checkpoint_id,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "RunIdentity":
        required = {
            "schema_version",
            "experiment_id",
            "run_id",
            "attempt_id",
            "session_id",
            "run_created_at",
            "attempt_started_at",
            "parent_experiment_id",
            "input_run_id",
            "input_checkpoint_id",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise IdentityContractError(
                f"RunIdentity 字段错误：missing={sorted(missing)}, extra={sorted(extra)}"
            )
        identity = cls(
            experiment_id=value["experiment_id"],
            run_id=value["run_id"],
            attempt_id=value["attempt_id"],
            session_id=value["session_id"],
            run_created_at=value["run_created_at"],
            attempt_started_at=value["attempt_started_at"],
            parent_experiment_id=value["parent_experiment_id"],
            input_run_id=value["input_run_id"],
            input_checkpoint_id=value["input_checkpoint_id"],
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != identity.artifact_hash:
            raise IdentityContractError("RunIdentity.artifact_hash 与内容不一致")
        return identity
