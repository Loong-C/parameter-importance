from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
import re
from pathlib import Path
from typing import Any, Final

from .atomic import atomic_write_json
from .storage import StorageLayout


SAFE_ID: Final = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}$")


class RunState(StrEnum):
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    RESUMABLE = "RESUMABLE"
    SUCCESS = "SUCCESS"
    FAILED_FINAL = "FAILED_FINAL"
    ABORTED_FINAL = "ABORTED_FINAL"


class ProcessState(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    STALE = "STALE"


RUN_TRANSITIONS: Final[dict[RunState, frozenset[RunState]]] = {
    RunState.CREATED: frozenset(
        {RunState.RUNNING, RunState.FAILED_FINAL, RunState.ABORTED_FINAL}
    ),
    RunState.RUNNING: frozenset(
        {
            RunState.RESUMABLE,
            RunState.SUCCESS,
            RunState.FAILED_FINAL,
            RunState.ABORTED_FINAL,
        }
    ),
    RunState.RESUMABLE: frozenset(
        {RunState.RUNNING, RunState.FAILED_FINAL, RunState.ABORTED_FINAL}
    ),
    RunState.SUCCESS: frozenset(),
    RunState.FAILED_FINAL: frozenset(),
    RunState.ABORTED_FINAL: frozenset(),
}
PROCESS_TRANSITIONS: Final[dict[ProcessState, frozenset[ProcessState]]] = {
    ProcessState.STARTING: frozenset(
        {ProcessState.RUNNING, ProcessState.FAILED, ProcessState.ABORTED}
    ),
    ProcessState.RUNNING: frozenset(
        {
            ProcessState.SUCCEEDED,
            ProcessState.FAILED,
            ProcessState.ABORTED,
            ProcessState.STALE,
        }
    ),
    ProcessState.STALE: frozenset(
        {ProcessState.RUNNING, ProcessState.FAILED, ProcessState.ABORTED}
    ),
    ProcessState.SUCCEEDED: frozenset(),
    ProcessState.FAILED: frozenset(),
    ProcessState.ABORTED: frozenset(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_identifier(value: str, *, field: str) -> str:
    if not SAFE_ID.fullmatch(value):
        raise ValueError(f"Invalid {field}: {value!r}")
    return value


def validate_transition(
    previous: RunState | ProcessState,
    target: RunState | ProcessState,
) -> None:
    table: Any = RUN_TRANSITIONS if isinstance(previous, RunState) else PROCESS_TRANSITIONS
    if target not in table[previous]:
        raise ValueError(f"Forbidden state transition: {previous} -> {target}")


@dataclass(frozen=True, slots=True)
class RunDirectory:
    run_id: str
    path: Path

    @classmethod
    def create(cls, layout: StorageLayout, run_id: str) -> "RunDirectory":
        validate_identifier(run_id, field="run_id")
        path = layout.path("runs", run_id)
        path.mkdir(parents=False, exist_ok=False)
        for name in ("attempts", "lineage", "diagnostics"):
            (path / name).mkdir()
        atomic_write_json(
            path / "run-state.json",
            {
                "schema_version": "stage0.run-state.v1",
                "run_id": run_id,
                "state": RunState.CREATED,
                "updated_at": _now(),
                "reason": "created",
            },
        )
        return cls(run_id, path)

    def state(self) -> RunState:
        import json

        value = json.loads((self.path / "run-state.json").read_text(encoding="utf-8"))
        return RunState(value["state"])

    def transition(self, target: RunState, *, reason: str) -> None:
        previous = self.state()
        validate_transition(previous, target)
        atomic_write_json(
            self.path / "run-state.json",
            {
                "schema_version": "stage0.run-state.v1",
                "run_id": self.run_id,
                "state": target,
                "previous_state": previous,
                "updated_at": _now(),
                "reason": reason,
            },
        )

    def create_attempt(self) -> "AttemptDirectory":
        root = self.path / "attempts"
        for number in range(1, 1_000_000):
            attempt_id = f"attempt-{number:06d}"
            path = root / attempt_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            (path / "sessions").mkdir()
            atomic_write_json(
                path / "attempt-state.json",
                {
                    "schema_version": "stage0.process-state.v1",
                    "run_id": self.run_id,
                    "attempt_id": attempt_id,
                    "state": ProcessState.STARTING,
                    "updated_at": _now(),
                },
            )
            return AttemptDirectory(self.run_id, attempt_id, path)
        raise RuntimeError(f"Attempt space exhausted: {self.path}")


@dataclass(frozen=True, slots=True)
class AttemptDirectory:
    run_id: str
    attempt_id: str
    path: Path

    def create_session(self, session_id: str) -> Path:
        validate_identifier(session_id, field="session_id")
        path = self.path / "sessions" / session_id
        path.mkdir(exist_ok=False)
        for name in ("events", "console", "tensorboard"):
            (path / name).mkdir()
        atomic_write_json(
            path / "session-state.json",
            {
                "schema_version": "stage0.process-state.v1",
                "run_id": self.run_id,
                "attempt_id": self.attempt_id,
                "session_id": session_id,
                "state": ProcessState.STARTING,
                "updated_at": _now(),
            },
        )
        return path
