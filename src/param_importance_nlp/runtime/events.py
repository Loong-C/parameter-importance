"""类型化 JSONL 事件真值与敏感信息保护。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
import os
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, runtime_checkable
import uuid

from ..atomic import stable_json_bytes
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

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
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
