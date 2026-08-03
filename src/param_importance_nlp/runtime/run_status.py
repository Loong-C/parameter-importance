"""Atomic run/attempt/session status and heartbeat contract.

The status document is independent of TensorBoard and console logs.  Every
transition is checked against a frozen matrix and appended to an immutable-in-
meaning history before the current snapshot is atomically replaced.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
import os
from pathlib import Path
from typing import Any, Mapping

from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from ..lifecycle import validate_identifier


class RunStatus(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    RESUMABLE = "RESUMABLE"
    SUCCESS = "SUCCESS"
    FAILED_FINAL = "FAILED_FINAL"
    ABORTED_FINAL = "ABORTED_FINAL"


class SessionStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    STALE = "STALE"


_RUN_TRANSITIONS = {
    RunStatus.CREATED: frozenset({RunStatus.RUNNING, RunStatus.ABORTED_FINAL}),
    RunStatus.RUNNING: frozenset(
        {
            RunStatus.RESUMABLE,
            RunStatus.SUCCESS,
            RunStatus.FAILED_FINAL,
            RunStatus.ABORTED_FINAL,
        }
    ),
    RunStatus.RESUMABLE: frozenset(
        {RunStatus.RUNNING, RunStatus.FAILED_FINAL, RunStatus.ABORTED_FINAL}
    ),
    RunStatus.SUCCESS: frozenset(),
    RunStatus.FAILED_FINAL: frozenset(),
    RunStatus.ABORTED_FINAL: frozenset(),
}
_SESSION_TRANSITIONS = {
    SessionStatus.STARTING: frozenset(
        {SessionStatus.RUNNING, SessionStatus.FAILED, SessionStatus.ABORTED}
    ),
    SessionStatus.RUNNING: frozenset(
        {
            SessionStatus.SUCCEEDED,
            SessionStatus.FAILED,
            SessionStatus.ABORTED,
            SessionStatus.STALE,
        }
    ),
    SessionStatus.SUCCEEDED: frozenset(),
    SessionStatus.FAILED: frozenset(),
    SessionStatus.ABORTED: frozenset(),
    SessionStatus.STALE: frozenset({SessionStatus.FAILED, SessionStatus.ABORTED}),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(value: str, *, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise ValueError(f"RUN_STATUS_TIMESTAMP_INVALID:{field}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"RUN_STATUS_TIMESTAMP_NAIVE:{field}")
    return parsed.astimezone(timezone.utc)


def _text(value: object, *, field: str, maximum: int = 2048) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"RUN_STATUS_TEXT_INVALID:{field}")
    return value


def _validate_document(value: object, *, run_id: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("RUN_STATUS_DOCUMENT_NOT_OBJECT")
    result = dict(value)
    expected = {
        "schema_version",
        "run_id",
        "run_status",
        "attempts",
        "sessions",
        "heartbeat",
        "history",
        "artifact_hash",
    }
    if set(result) != expected or result.get("schema_version") != "runtime.run-status.v1":
        raise ValueError("RUN_STATUS_DOCUMENT_FIELDS_OR_VERSION_INVALID")
    declared = result.pop("artifact_hash")
    if declared != canonical_json_hash(result):
        raise ValueError("RUN_STATUS_DOCUMENT_HASH_MISMATCH")
    result["artifact_hash"] = declared
    if result.get("run_id") != run_id:
        raise ValueError("RUN_STATUS_RUN_ID_MISMATCH")
    RunStatus(result["run_status"])
    attempts = result.get("attempts")
    if not isinstance(attempts, Mapping):
        raise ValueError("RUN_STATUS_ATTEMPTS_INVALID")
    for attempt_id, raw in attempts.items():
        validate_identifier(attempt_id, field="attempt_id")
        if not isinstance(raw, Mapping) or set(raw) != {
            "status",
            "created_at",
            "updated_at",
        }:
            raise ValueError("RUN_STATUS_ATTEMPT_RECORD_INVALID")
        SessionStatus(raw["status"])
        _timestamp(raw["created_at"], field="attempt.created_at")
        _timestamp(raw["updated_at"], field="attempt.updated_at")
    sessions = result.get("sessions")
    if not isinstance(sessions, Mapping):
        raise ValueError("RUN_STATUS_SESSIONS_INVALID")
    for session_id, raw in sessions.items():
        validate_identifier(session_id, field="session_id")
        if not isinstance(raw, Mapping) or set(raw) != {
            "attempt_id",
            "status",
            "created_at",
            "updated_at",
        }:
            raise ValueError("RUN_STATUS_SESSION_RECORD_INVALID")
        validate_identifier(raw["attempt_id"], field="attempt_id")
        if raw["attempt_id"] not in attempts:
            raise ValueError("RUN_STATUS_SESSION_ATTEMPT_UNKNOWN")
        SessionStatus(raw["status"])
        _timestamp(raw["created_at"], field="session.created_at")
        _timestamp(raw["updated_at"], field="session.updated_at")
    heartbeat = result.get("heartbeat")
    if heartbeat is not None:
        if not isinstance(heartbeat, Mapping) or set(heartbeat) != {
            "session_id",
            "last_step",
            "observed_at",
        }:
            raise ValueError("RUN_STATUS_HEARTBEAT_INVALID")
        if heartbeat["session_id"] not in sessions:
            raise ValueError("RUN_STATUS_HEARTBEAT_SESSION_UNKNOWN")
        if (
            isinstance(heartbeat["last_step"], bool)
            or not isinstance(heartbeat["last_step"], int)
            or heartbeat["last_step"] < 0
        ):
            raise ValueError("RUN_STATUS_HEARTBEAT_STEP_INVALID")
        _timestamp(heartbeat["observed_at"], field="heartbeat.observed_at")
    history = result.get("history")
    if not isinstance(history, list) or any(not isinstance(item, Mapping) for item in history):
        raise ValueError("RUN_STATUS_HISTORY_INVALID")
    return result


class RunStatusStore:
    """Single-writer atomic status store with explicit recovery preconditions."""

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        created_at: str | None = None,
    ) -> None:
        validate_identifier(run_id, field="run_id")
        self.path = Path(path)
        self.run_id = run_id
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
            else:  # pragma: no cover - the server exercises this branch
                import fcntl

                fcntl.flock(self._lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            os.close(self._lock_descriptor)
            self._lock_descriptor = -1
            raise RuntimeError(f"RUN_STATUS_WRITER_ALREADY_ACTIVE:{self.path}") from error
        try:
            if self.path.exists():
                self._read()
            else:
                timestamp = created_at or _now()
                _timestamp(timestamp, field="created_at")
                self._write(
                    {
                        "schema_version": "runtime.run-status.v1",
                        "run_id": run_id,
                        "run_status": RunStatus.CREATED.value,
                        "attempts": {},
                        "sessions": {},
                        "heartbeat": None,
                        "history": [
                            {
                                "entity": "run",
                                "entity_id": run_id,
                                "from": None,
                                "to": RunStatus.CREATED.value,
                                "at": timestamp,
                                "actor": "runtime",
                                "reason": "created",
                                "checkpoint_ref": None,
                            }
                        ],
                    }
                )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
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

    def __enter__(self) -> "RunStatusStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - interpreter safety net
        try:
            self.close()
        except Exception:
            pass

    @property
    def transition_matrix(self) -> dict[str, JSONValue]:
        return {
            "run": {
                source.value: sorted(target.value for target in targets)
                for source, targets in _RUN_TRANSITIONS.items()
            },
            "attempt": {
                source.value: sorted(target.value for target in targets)
                for source, targets in _SESSION_TRANSITIONS.items()
            },
            "session": {
                source.value: sorted(target.value for target in targets)
                for source, targets in _SESSION_TRANSITIONS.items()
            },
        }

    def _read(self) -> dict[str, Any]:
        if self._lock_descriptor < 0:
            raise RuntimeError("RUN_STATUS_STORE_CLOSED")
        return _validate_document(load_canonical_json(self.path), run_id=self.run_id)

    def _write(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if self._lock_descriptor < 0:
            raise RuntimeError("RUN_STATUS_STORE_CLOSED")
        body: dict[str, JSONValue] = dict(value)  # type: ignore[assignment]
        body["artifact_hash"] = canonical_json_hash(body)
        write_canonical_json(self.path, body)
        return _validate_document(body, run_id=self.run_id)

    def snapshot(self) -> dict[str, JSONValue]:
        return self._read()  # type: ignore[return-value]

    def transition_run(
        self,
        target: RunStatus | str,
        *,
        actor: str,
        reason: str,
        at: str | None = None,
        checkpoint_ref: str | None = None,
        checkpoint_complete: bool = False,
    ) -> dict[str, JSONValue]:
        value = self._read()
        source = RunStatus(value["run_status"])
        target_value = RunStatus(target)
        if target_value not in _RUN_TRANSITIONS[source]:
            raise ValueError(f"RUN_STATUS_TRANSITION_FORBIDDEN:{source.value}:{target_value.value}")
        if target_value is RunStatus.RESUMABLE or (
            source is RunStatus.RESUMABLE and target_value is RunStatus.RUNNING
        ):
            if not checkpoint_complete or not isinstance(checkpoint_ref, str) or not checkpoint_ref:
                raise ValueError("RUN_STATUS_RECOVERY_REQUIRES_COMPLETE_CHECKPOINT")
        actor_value = _text(actor, field="actor", maximum=256)
        reason_value = _text(reason, field="reason")
        timestamp = at or _now()
        _timestamp(timestamp, field="transition.at")
        value["run_status"] = target_value.value
        value["history"].append(
            {
                "entity": "run",
                "entity_id": self.run_id,
                "from": source.value,
                "to": target_value.value,
                "at": timestamp,
                "actor": actor_value,
                "reason": reason_value,
                "checkpoint_ref": checkpoint_ref,
            }
        )
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]

    def register_session(
        self,
        *,
        attempt_id: str,
        session_id: str,
        at: str | None = None,
    ) -> dict[str, JSONValue]:
        validate_identifier(attempt_id, field="attempt_id")
        validate_identifier(session_id, field="session_id")
        value = self._read()
        if RunStatus(value["run_status"]) is not RunStatus.RUNNING:
            raise ValueError("RUN_STATUS_SESSION_REQUIRES_RUNNING_RUN")
        attempt = value["attempts"].get(attempt_id)
        if not isinstance(attempt, Mapping) or SessionStatus(attempt["status"]) is not SessionStatus.RUNNING:
            raise ValueError("RUN_STATUS_SESSION_REQUIRES_RUNNING_ATTEMPT")
        if session_id in value["sessions"]:
            raise ValueError("RUN_STATUS_SESSION_ALREADY_EXISTS")
        timestamp = at or _now()
        _timestamp(timestamp, field="session.created_at")
        value["sessions"][session_id] = {
            "attempt_id": attempt_id,
            "status": SessionStatus.STARTING.value,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        value["history"].append(
            {
                "entity": "session",
                "entity_id": session_id,
                "from": None,
                "to": SessionStatus.STARTING.value,
                "at": timestamp,
                "actor": "runtime",
                "reason": "registered",
                "checkpoint_ref": None,
            }
        )
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]

    def register_attempt(
        self,
        *,
        attempt_id: str,
        at: str | None = None,
    ) -> dict[str, JSONValue]:
        validate_identifier(attempt_id, field="attempt_id")
        value = self._read()
        if RunStatus(value["run_status"]) is not RunStatus.RUNNING:
            raise ValueError("RUN_STATUS_ATTEMPT_REQUIRES_RUNNING_RUN")
        if attempt_id in value["attempts"]:
            raise ValueError("RUN_STATUS_ATTEMPT_ALREADY_EXISTS")
        timestamp = at or _now()
        _timestamp(timestamp, field="attempt.created_at")
        value["attempts"][attempt_id] = {
            "status": SessionStatus.STARTING.value,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        value["history"].append(
            {
                "entity": "attempt",
                "entity_id": attempt_id,
                "from": None,
                "to": SessionStatus.STARTING.value,
                "at": timestamp,
                "actor": "runtime",
                "reason": "registered",
                "checkpoint_ref": None,
            }
        )
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]

    def transition_attempt(
        self,
        attempt_id: str,
        target: SessionStatus | str,
        *,
        actor: str,
        reason: str,
        at: str | None = None,
    ) -> dict[str, JSONValue]:
        value = self._read()
        record = value["attempts"].get(attempt_id)
        if not isinstance(record, dict):
            raise ValueError("RUN_STATUS_ATTEMPT_UNKNOWN")
        source = SessionStatus(record["status"])
        target_value = SessionStatus(target)
        if target_value not in _SESSION_TRANSITIONS[source]:
            raise ValueError(
                f"RUN_STATUS_ATTEMPT_TRANSITION_FORBIDDEN:{source.value}:{target_value.value}"
            )
        if target_value is SessionStatus.STALE:
            raise ValueError("RUN_STATUS_ATTEMPT_STALE_REQUIRES_SESSION_ADJUDICATION")
        timestamp = at or _now()
        _timestamp(timestamp, field="attempt.transition.at")
        if target_value in {
            SessionStatus.SUCCEEDED,
            SessionStatus.FAILED,
            SessionStatus.ABORTED,
        }:
            active_sessions = [
                session_id
                for session_id, session in value["sessions"].items()
                if session["attempt_id"] == attempt_id
                and SessionStatus(session["status"])
                in {SessionStatus.STARTING, SessionStatus.RUNNING}
            ]
            if active_sessions:
                raise ValueError("RUN_STATUS_ATTEMPT_HAS_ACTIVE_SESSIONS")
        record["status"] = target_value.value
        record["updated_at"] = timestamp
        value["history"].append(
            {
                "entity": "attempt",
                "entity_id": attempt_id,
                "from": source.value,
                "to": target_value.value,
                "at": timestamp,
                "actor": _text(actor, field="actor", maximum=256),
                "reason": _text(reason, field="reason"),
                "checkpoint_ref": None,
            }
        )
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]

    def transition_session(
        self,
        session_id: str,
        target: SessionStatus | str,
        *,
        actor: str,
        reason: str,
        at: str | None = None,
        process_exists: bool | None = None,
        stale_after_seconds: float | None = None,
    ) -> dict[str, JSONValue]:
        value = self._read()
        record = value["sessions"].get(session_id)
        if not isinstance(record, dict):
            raise ValueError("RUN_STATUS_SESSION_UNKNOWN")
        source = SessionStatus(record["status"])
        target_value = SessionStatus(target)
        if target_value not in _SESSION_TRANSITIONS[source]:
            raise ValueError(
                f"RUN_STATUS_SESSION_TRANSITION_FORBIDDEN:{source.value}:{target_value.value}"
            )
        timestamp = at or _now()
        observed_at = _timestamp(timestamp, field="session.transition.at")
        if target_value is SessionStatus.STALE:
            heartbeat = value.get("heartbeat")
            if (
                process_exists is not False
                or not isinstance(stale_after_seconds, (int, float))
                or stale_after_seconds <= 0
                or not isinstance(heartbeat, Mapping)
                or heartbeat.get("session_id") != session_id
                or (
                    observed_at
                    - _timestamp(heartbeat["observed_at"], field="heartbeat.observed_at")
                ).total_seconds()
                <= float(stale_after_seconds)
            ):
                raise ValueError("RUN_STATUS_STALE_PRECONDITIONS_FAILED")
        record["status"] = target_value.value
        record["updated_at"] = timestamp
        value["history"].append(
            {
                "entity": "session",
                "entity_id": session_id,
                "from": source.value,
                "to": target_value.value,
                "at": timestamp,
                "actor": _text(actor, field="actor", maximum=256),
                "reason": _text(reason, field="reason"),
                "checkpoint_ref": None,
            }
        )
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]

    def heartbeat(
        self,
        session_id: str,
        *,
        last_step: int,
        observed_at: str | None = None,
    ) -> dict[str, JSONValue]:
        value = self._read()
        record = value["sessions"].get(session_id)
        if not isinstance(record, Mapping) or SessionStatus(record["status"]) is not SessionStatus.RUNNING:
            raise ValueError("RUN_STATUS_HEARTBEAT_REQUIRES_RUNNING_SESSION")
        if isinstance(last_step, bool) or not isinstance(last_step, int) or last_step < 0:
            raise ValueError("RUN_STATUS_HEARTBEAT_STEP_INVALID")
        timestamp = observed_at or _now()
        _timestamp(timestamp, field="heartbeat.observed_at")
        previous = value.get("heartbeat")
        if isinstance(previous, Mapping) and int(previous["last_step"]) > last_step:
            raise ValueError("RUN_STATUS_HEARTBEAT_STEP_REGRESSION")
        value["heartbeat"] = {
            "session_id": session_id,
            "last_step": last_step,
            "observed_at": timestamp,
        }
        return self._write({key: item for key, item in value.items() if key != "artifact_hash"})  # type: ignore[return-value]


__all__ = ["RunStatus", "RunStatusStore", "SessionStatus"]
