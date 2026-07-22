"""attempt/session 状态推进与独立心跳。

状态文档是低频、带 revision 的权威快照；心跳是高频租约证据，单独写入，避免
一次心跳被误解为业务状态转换。所有时间均为带时区 UTC ISO-8601。检测为 stale
只生成诊断结论，不自动抢占或把原进程标成失败。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..lifecycle import ProcessState, validate_identifier, validate_transition
from ._jsonio import load_canonical_json


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class ProcessSnapshot:
    run_id: str
    attempt_id: str
    session_id: str | None
    state: ProcessState
    revision: int
    updated_at: str
    reason: str


class ProcessStateStore:
    """管理一个 attempt 或 session 的 fail-closed 状态文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.heartbeat_path = self.path.with_name(
            f"{self.path.stem}.heartbeat{self.path.suffix}"
        )

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        run_id: str,
        attempt_id: str,
        session_id: str | None = None,
    ) -> "ProcessStateStore":
        for field, value in (("run_id", run_id), ("attempt_id", attempt_id)):
            validate_identifier(value, field=field)
        if session_id is not None:
            validate_identifier(session_id, field="session_id")
        store = cls(path)
        if store.path.exists():
            raise FileExistsError(f"PROCESS_STATE_ALREADY_EXISTS:{store.path}")
        timestamp = _now().isoformat()
        value: dict[str, Any] = {
            "schema_version": "runtime.process-state.v1",
            "run_id": run_id,
            "attempt_id": attempt_id,
            "session_id": session_id,
            "state": ProcessState.STARTING,
            "revision": 0,
            "updated_at": timestamp,
            "reason": "created",
        }
        atomic_write_json(store.path, value)
        return store

    def read(self) -> ProcessSnapshot:
        value = load_canonical_json(self.path)
        expected = {
            "schema_version",
            "run_id",
            "attempt_id",
            "session_id",
            "state",
            "revision",
            "updated_at",
            "reason",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("PROCESS_STATE_FIELDS_MISMATCH")
        if value["schema_version"] != "runtime.process-state.v1":
            raise ValueError("PROCESS_STATE_SCHEMA_MISMATCH")
        try:
            snapshot = ProcessSnapshot(
                run_id=value["run_id"],
                attempt_id=value["attempt_id"],
                session_id=value["session_id"],
                state=ProcessState(value["state"]),
                revision=value["revision"],
                updated_at=value["updated_at"],
                reason=value["reason"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("PROCESS_STATE_INVALID") from exc
        validate_identifier(snapshot.run_id, field="run_id")
        validate_identifier(snapshot.attempt_id, field="attempt_id")
        if snapshot.session_id is not None:
            validate_identifier(snapshot.session_id, field="session_id")
        if (
            isinstance(snapshot.revision, bool)
            or not isinstance(snapshot.revision, int)
            or snapshot.revision < 0
        ):
            raise ValueError("PROCESS_STATE_INVALID_REVISION")
        _parse_timestamp(snapshot.updated_at, field="updated_at")
        return snapshot

    def transition(
        self,
        target: ProcessState,
        *,
        reason: str,
        expected_revision: int | None = None,
    ) -> ProcessSnapshot:
        if not reason.strip():
            raise ValueError("PROCESS_STATE_REASON_REQUIRED")
        previous = self.read()
        if expected_revision is not None and previous.revision != expected_revision:
            raise ValueError(
                f"PROCESS_STATE_REVISION_CONFLICT:expected={expected_revision}:actual={previous.revision}"
            )
        validate_transition(previous.state, target)
        value = {
            "schema_version": "runtime.process-state.v1",
            "run_id": previous.run_id,
            "attempt_id": previous.attempt_id,
            "session_id": previous.session_id,
            "state": target,
            "revision": previous.revision + 1,
            "updated_at": _now().isoformat(),
            "reason": reason,
        }
        atomic_write_json(self.path, value)
        return self.read()

    def heartbeat(self, *, sequence: int) -> dict[str, Any]:
        """发布一个单调心跳；倒退或重复序号均拒绝。"""

        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
            raise ValueError("PROCESS_HEARTBEAT_INVALID_SEQUENCE")
        snapshot = self.read()
        if self.heartbeat_path.exists():
            previous = self._read_heartbeat()
            if sequence != previous["sequence"] + 1:
                raise ValueError(
                    f"PROCESS_HEARTBEAT_SEQUENCE_GAP:expected={previous['sequence'] + 1}:actual={sequence}"
                )
        elif sequence != 0:
            raise ValueError(f"PROCESS_HEARTBEAT_SEQUENCE_GAP:expected=0:actual={sequence}")
        value = {
            "schema_version": "runtime.process-heartbeat.v1",
            "run_id": snapshot.run_id,
            "attempt_id": snapshot.attempt_id,
            "session_id": snapshot.session_id,
            "state_revision": snapshot.revision,
            "sequence": sequence,
            "observed_at": _now().isoformat(),
        }
        atomic_write_json(self.heartbeat_path, value)
        return value

    def _read_heartbeat(self) -> dict[str, Any]:
        value = load_canonical_json(self.heartbeat_path)
        expected = {
            "schema_version",
            "run_id",
            "attempt_id",
            "session_id",
            "state_revision",
            "sequence",
            "observed_at",
        }
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("PROCESS_HEARTBEAT_FIELDS_MISMATCH")
        if value["schema_version"] != "runtime.process-heartbeat.v1":
            raise ValueError("PROCESS_HEARTBEAT_SCHEMA_MISMATCH")
        if any(
            isinstance(value[field], bool)
            or not isinstance(value[field], int)
            or value[field] < 0
            for field in ("state_revision", "sequence")
        ):
            raise ValueError("PROCESS_HEARTBEAT_INVALID_COUNTER")
        _parse_timestamp(value["observed_at"], field="observed_at")
        return value

    def heartbeat_is_stale(
        self, *, now: datetime | None = None, threshold: timedelta
    ) -> bool:
        if threshold <= timedelta(0):
            raise ValueError("PROCESS_STALE_THRESHOLD_NONPOSITIVE")
        heartbeat = self._read_heartbeat()
        observed = _parse_timestamp(heartbeat["observed_at"], field="observed_at")
        reference = now or _now()
        if reference.tzinfo is None:
            raise ValueError("PROCESS_STALE_REFERENCE_WITHOUT_TIMEZONE")
        return reference.astimezone(timezone.utc) - observed > threshold


def _parse_timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"PROCESS_TIMESTAMP_NOT_STRING:{field}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"PROCESS_TIMESTAMP_INVALID:{field}") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"PROCESS_TIMESTAMP_WITHOUT_TIMEZONE:{field}")
    return parsed.astimezone(timezone.utc)


__all__ = ["ProcessSnapshot", "ProcessStateStore"]
