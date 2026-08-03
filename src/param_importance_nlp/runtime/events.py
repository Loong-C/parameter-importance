"""类型化 JSONL 事件真值与敏感信息保护。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, runtime_checkable
import uuid

from ..atomic import stable_json_bytes
from ..contracts.jsonio import canonical_json_hash
from ..lifecycle import validate_identifier
from ._jsonio import load_canonical_json_bytes


class EventType(StrEnum):
    RUN_LIFECYCLE = "run_lifecycle"
    OPTIMIZER_STEP = "optimizer_step"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    SYSTEM = "system"
    WARNING = "warning"
    ERROR = "error"


_SENSITIVE_PATTERNS = {
    "private_key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "bearer": re.compile(r"Bearer\s+[A-Za-z0-9._~+/-]{12,}", re.IGNORECASE),
    "signed_url": re.compile(
        r"https?://[^\s]+[?&](?:X-Amz-Signature|sig|signature|token)=",
        re.IGNORECASE,
    ),
    "credential": re.compile(
        r"(?:password|api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*[^\s,}]+",
        re.IGNORECASE,
    ),
}

_PAYLOAD_FIELDS = {
    EventType.RUN_LIFECYCLE.value: frozenset(
        {"status", "reason", "checkpoint_ref", "recoverable"}
    ),
    EventType.OPTIMIZER_STEP.value: frozenset(
        {
            "global_step",
            "attempt_index",
            "status",
            "batch_ids",
            "mean_loss",
            "effective_count",
            "global_gradient_norm",
            "clip_factor",
            "estimator_name",
            "parameter_post_state_hash",
            "attempt_commit_state_hash",
            "skip_reason",
            "microstep_count",
            "sample_count",
            "effective_token_count",
            "learning_rates_post_step",
            "learning_rate",
        }
    ),
    EventType.VALIDATION.value: frozenset(
        {"global_step", "split", "metrics", "sample_count", "effective_token_count"}
    ),
    EventType.CHECKPOINT.value: frozenset(
        {
            "checkpoint_id",
            "global_step",
            "checkpoint_ref",
            "status",
            "manifest_sha256",
            "parent_checkpoint_id",
        }
    ),
    EventType.SYSTEM.value: frozenset(
        {
            "global_step",
            "gpu_memory_bytes",
            "gpu_utilization_percent",
            "cpu_memory_bytes",
            "disk_free_bytes",
            "throughput_units_per_second",
            "gpu_uuid",
        }
    ),
    EventType.WARNING.value: frozenset(
        {"warning_code", "message", "global_step", "affected_ranks"}
    ),
    EventType.ERROR.value: frozenset(
        {
            "exception_class",
            "last_valid_step",
            "affected_ranks",
            "recoverable",
            "message",
            "traceback_ref",
            "error_code",
        }
    ),
}
_PAYLOAD_REQUIRED = {
    EventType.RUN_LIFECYCLE.value: frozenset({"status"}),
    EventType.OPTIMIZER_STEP.value: frozenset(
        {
            "global_step",
            "microstep_count",
            "sample_count",
            "effective_token_count",
            "mean_loss",
            "global_gradient_norm",
            "learning_rates_post_step",
        }
    ),
    EventType.VALIDATION.value: frozenset({"global_step", "split", "metrics"}),
    EventType.CHECKPOINT.value: frozenset(
        {"checkpoint_id", "global_step", "checkpoint_ref", "status", "manifest_sha256"}
    ),
    EventType.SYSTEM.value: frozenset(
        {
            "gpu_memory_bytes",
            "gpu_utilization_percent",
            "cpu_memory_bytes",
            "disk_free_bytes",
            "throughput_units_per_second",
        }
    ),
    EventType.WARNING.value: frozenset({"warning_code", "message"}),
    EventType.ERROR.value: frozenset(
        {"exception_class", "last_valid_step", "affected_ranks", "recoverable"}
    ),
}


def _non_negative_int(value: object, *, field: str, positive: bool = False) -> int:
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"EVENT_PAYLOAD_INTEGER_INVALID:{field}")
    return value


def _finite_number(value: object, *, field: str, nullable: bool = False) -> float | None:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"EVENT_PAYLOAD_NUMBER_INVALID:{field}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"EVENT_PAYLOAD_NUMBER_NONFINITE:{field}")
    return converted


def _validate_typed_payload(event_type: str, payload: dict[str, Any]) -> None:
    allowed = _PAYLOAD_FIELDS[event_type]
    required = _PAYLOAD_REQUIRED[event_type]
    extra = set(payload) - allowed
    missing = required - set(payload)
    if extra or missing:
        raise ValueError(
            f"EVENT_PAYLOAD_FIELDS_INVALID:{event_type}:"
            f"missing={sorted(missing)}:extra={sorted(extra)}"
        )
    if event_type == EventType.RUN_LIFECYCLE.value:
        if not isinstance(payload["status"], str) or not payload["status"]:
            raise ValueError("EVENT_LIFECYCLE_STATUS_INVALID")
    elif event_type == EventType.OPTIMIZER_STEP.value:
        _non_negative_int(payload["global_step"], field="global_step")
        _non_negative_int(payload["microstep_count"], field="microstep_count", positive=True)
        _non_negative_int(payload["sample_count"], field="sample_count", positive=True)
        _non_negative_int(payload["effective_token_count"], field="effective_token_count")
        _finite_number(payload["mean_loss"], field="mean_loss", nullable=True)
        norm = _finite_number(
            payload["global_gradient_norm"],
            field="global_gradient_norm",
            nullable=True,
        )
        if norm is not None and norm < 0:
            raise ValueError("EVENT_GLOBAL_GRADIENT_NORM_NEGATIVE")
        learning_rates = payload["learning_rates_post_step"]
        if not isinstance(learning_rates, list) or not learning_rates:
            raise ValueError("EVENT_LEARNING_RATES_INVALID")
        if any(
            (_finite_number(item, field="learning_rates_post_step") or 0.0) < 0
            for item in learning_rates
        ):
            raise ValueError("EVENT_LEARNING_RATE_NEGATIVE")
    elif event_type == EventType.VALIDATION.value:
        _non_negative_int(payload["global_step"], field="global_step")
        if not isinstance(payload["split"], str) or not payload["split"]:
            raise ValueError("EVENT_VALIDATION_SPLIT_INVALID")
        if not isinstance(payload["metrics"], dict):
            raise ValueError("EVENT_VALIDATION_METRICS_INVALID")
    elif event_type == EventType.CHECKPOINT.value:
        _non_negative_int(payload["global_step"], field="global_step")
        for field in ("checkpoint_id", "checkpoint_ref", "status"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise ValueError(f"EVENT_CHECKPOINT_FIELD_INVALID:{field}")
        manifest = payload["manifest_sha256"]
        if not isinstance(manifest, str) or re.fullmatch(r"[0-9a-f]{64}", manifest) is None:
            raise ValueError("EVENT_CHECKPOINT_MANIFEST_HASH_INVALID")
    elif event_type == EventType.SYSTEM.value:
        for field in (
            "gpu_memory_bytes",
            "gpu_utilization_percent",
            "cpu_memory_bytes",
            "disk_free_bytes",
            "throughput_units_per_second",
        ):
            number = _finite_number(payload[field], field=field)
            if number is not None and number < 0:
                raise ValueError(f"EVENT_SYSTEM_METRIC_NEGATIVE:{field}")
    elif event_type == EventType.WARNING.value:
        for field in ("warning_code", "message"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise ValueError(f"EVENT_WARNING_FIELD_INVALID:{field}")
    elif event_type == EventType.ERROR.value:
        if not isinstance(payload["exception_class"], str) or not payload["exception_class"]:
            raise ValueError("EVENT_ERROR_EXCEPTION_CLASS_INVALID")
        _non_negative_int(payload["last_valid_step"], field="last_valid_step")
        ranks = payload["affected_ranks"]
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("EVENT_ERROR_AFFECTED_RANKS_INVALID")
        for rank in ranks:
            _non_negative_int(rank, field="affected_ranks")
        if not isinstance(payload["recoverable"], bool):
            raise ValueError("EVENT_ERROR_RECOVERABLE_INVALID")


def _scan_sensitive(value: Any, path: str = "payload") -> None:
    if isinstance(value, str):
        for label, pattern in _SENSITIVE_PATTERNS.items():
            if pattern.search(value):
                raise ValueError(f"EVENT_SENSITIVE_VALUE:{label}:{path}")
        if len(value) > 8192:
            raise ValueError(f"EVENT_STRING_TOO_LONG:{path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            _scan_sensitive(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_sensitive(item, f"{path}[{index}]")


def _validate_event(event: "EventRecord") -> None:
    """验证从不可信 JSON 重建的事件，而非只信任 dataclass 构造成功。"""

    if event.schema_version != "runtime.event.v1":
        raise ValueError("EVENT_SCHEMA_MISMATCH")
    for field in ("event_id", "experiment_id", "run_id", "attempt_id", "session_id"):
        value = getattr(event, field)
        if not isinstance(value, str):
            raise ValueError(f"EVENT_IDENTIFIER_NOT_STRING:{field}")
        validate_identifier(value, field=field)
    if isinstance(event.rank, bool) or not isinstance(event.rank, int) or event.rank < 0:
        raise ValueError("EVENT_INVALID_RANK")
    if (
        isinstance(event.sequence, bool)
        or not isinstance(event.sequence, int)
        or event.sequence < 0
    ):
        raise ValueError("EVENT_INVALID_SEQUENCE")
    if event.event_type not in {item.value for item in EventType}:
        raise ValueError(f"EVENT_UNKNOWN_TYPE:{event.event_type}")
    if not isinstance(event.payload, dict):
        raise ValueError("EVENT_PAYLOAD_NOT_OBJECT")
    try:
        canonical_json_hash(event.payload)
    except (TypeError, ValueError) as error:
        raise ValueError("EVENT_PAYLOAD_NOT_STRICT_JSON") from error
    if len(stable_json_bytes(event.payload)) > 1_048_576:
        raise ValueError("EVENT_PAYLOAD_SIZE_LIMIT_EXCEEDED")
    global_step = event.payload.get("global_step")
    if global_step is not None and (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("EVENT_GLOBAL_STEP_INVALID")
    if event.event_type == EventType.OPTIMIZER_STEP.value and global_step is None:
        raise ValueError("EVENT_OPTIMIZER_STEP_MISSING_GLOBAL_STEP")
    if event.event_type == EventType.CHECKPOINT.value:
        checkpoint_id = event.payload.get("checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("EVENT_CHECKPOINT_ID_MISSING")
    _validate_typed_payload(event.event_type, event.payload)
    try:
        parsed = datetime.fromisoformat(event.occurred_at.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("EVENT_INVALID_TIMESTAMP") from exc
    if parsed.tzinfo is None:
        raise ValueError("EVENT_TIMESTAMP_WITHOUT_TIMEZONE")
    _scan_sensitive(event.payload)


@dataclass(frozen=True, slots=True)
class EventRecord:
    """单个 session/rank 内带单调序号的机器真值事件。"""

    schema_version: str
    event_id: str
    experiment_id: str
    run_id: str
    attempt_id: str
    session_id: str
    rank: int
    event_type: str
    sequence: int
    occurred_at: str
    payload: dict[str, Any]

    @classmethod
    def create(
        cls,
        *,
        experiment_id: str,
        run_id: str,
        attempt_id: str,
        session_id: str,
        rank: int,
        event_type: EventType,
        sequence: int,
        payload: dict[str, Any],
        event_id: str | None = None,
        occurred_at: str | None = None,
    ) -> "EventRecord":
        for field, value in (
            ("experiment_id", experiment_id),
            ("run_id", run_id),
            ("attempt_id", attempt_id),
            ("session_id", session_id),
        ):
            validate_identifier(value, field=field)
        if rank < 0 or sequence < 0:
            raise ValueError("EVENT_NEGATIVE_RANK_OR_SEQUENCE")
        _scan_sensitive(payload)
        result = cls(
            schema_version="runtime.event.v1",
            event_id=event_id or uuid.uuid4().hex,
            experiment_id=experiment_id,
            run_id=run_id,
            attempt_id=attempt_id,
            session_id=session_id,
            rank=rank,
            event_type=event_type.value,
            sequence=sequence,
            occurred_at=occurred_at or datetime.now(timezone.utc).isoformat(),
            payload=payload,
        )
        _validate_event(result)
        return result


@runtime_checkable
class EventSink(Protocol):
    """运行编排只依赖的最小事件写入协议。"""

    def append(self, event: EventRecord, *, critical: bool = False) -> None:
        """原子追加一个已验证事件；关键事件要求持久化屏障。"""


class JsonlEventSink:
    """一个 session/rank 独占的 append-only JSONL 写入器。"""

    def __init__(
        self,
        path: str | Path,
        *,
        max_session_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.path = Path(path)
        if (
            isinstance(max_session_bytes, bool)
            or not isinstance(max_session_bytes, int)
            or max_session_bytes <= 0
        ):
            raise ValueError("EVENT_SESSION_SIZE_LIMIT_INVALID")
        self.max_session_bytes = max_session_bytes
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.parent / f".{self.path.name}.writer.lock"
        self._lock_descriptor = os.open(
            self._lock_path, os.O_RDWR | os.O_CREAT, 0o644
        )
        try:
            if os.fstat(self._lock_descriptor).st_size == 0:
                os.write(self._lock_descriptor, b"0")
                os.fsync(self._lock_descriptor)
            os.lseek(self._lock_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._lock_descriptor, msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows 是本项目当前本机验证平台
                import fcntl

                fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1
            raise RuntimeError(f"EVENT_WRITER_ALREADY_ACTIVE:{self.path}") from exc
        self._last_sequence = -1
        if self.path.exists():
            existing = read_event_stream(self.path)
            if existing:
                self._last_sequence = existing[-1].sequence

    def close(self) -> None:
        """释放单写者租约；锁文件本身保留，避免创建/删除竞争。"""

        if self._lock_descriptor < 0:
            return
        try:
            os.lseek(self._lock_descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._lock_descriptor, msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl

                fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1

    def __enter__(self) -> "JsonlEventSink":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - 仅作为异常路径安全网
        try:
            self.close()
        except Exception:
            pass

    def append(self, event: EventRecord, *, critical: bool = False) -> None:
        if event.sequence != self._last_sequence + 1:
            raise ValueError(
                f"EVENT_SEQUENCE_GAP:expected={self._last_sequence + 1}:actual={event.sequence}"
            )
        payload = stable_json_bytes(asdict(event))
        current_size = self.path.stat().st_size if self.path.exists() else 0
        if current_size + len(payload) > self.max_session_bytes:
            raise OSError("EVENT_SESSION_SIZE_LIMIT_EXCEEDED")
        with self.path.open("ab") as handle:
            handle.write(payload)
            handle.flush()
            if critical:
                os.fsync(handle.fileno())
        self._last_sequence = event.sequence


def read_event_stream(path: str | Path) -> list[EventRecord]:
    """读取并验证单调事件流；截断行、重复 ID 和倒退序号全部失败。"""

    payload = Path(path).read_bytes()
    if payload and not payload.endswith(b"\n"):
        raise ValueError("EVENT_STREAM_TRUNCATED_FINAL_LINE")
    result: list[EventRecord] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(payload.splitlines(), start=1):
        try:
            value = load_canonical_json_bytes(
                raw, source=f"{Path(path)}:{line_number}"
            )
            event = EventRecord(**value)
            _validate_event(event)
        except Exception as exc:
            raise ValueError(f"EVENT_STREAM_INVALID_LINE:{line_number}") from exc
        if event.event_id in seen_ids:
            raise ValueError(f"EVENT_DUPLICATE_ID:{event.event_id}")
        if event.sequence != len(result):
            raise ValueError(f"EVENT_SEQUENCE_INVALID:{event.sequence}")
        _scan_sensitive(event.payload)
        seen_ids.add(event.event_id)
        result.append(event)
    return result


def canonical_optimizer_steps(
    streams: Iterable[Iterable[EventRecord]],
) -> list[EventRecord]:
    """合并选中 lineage 的 rank-0 optimizer step，并拒绝重复或缺口。"""

    steps = sorted(
        (
            event
            for stream in streams
            for event in stream
            if event.rank == 0 and event.event_type == EventType.OPTIMIZER_STEP.value
        ),
        key=lambda item: int(item.payload["global_step"]),
    )
    observed = [int(item.payload["global_step"]) for item in steps]
    if observed and observed != list(range(observed[0], observed[-1] + 1)):
        raise ValueError(f"CANONICAL_STEP_GAP_OR_DUPLICATE:{observed}")
    return steps


__all__ = [
    "EventRecord",
    "EventSink",
    "EventType",
    "JsonlEventSink",
    "canonical_optimizer_steps",
    "read_event_stream",
]
