"""attempt/session lineage 的规范化选择与持久化。

恢复、人工重试和调度器重启可能让同一个 run 出现多个 attempt。事件文件仍然
保持 append-only，但分析只能消费一个显式选中的 canonical attempt；其余分支
必须被标成 ``SUPERSEDED`` 或 ``ORPHAN``，不能靠目录时间戳猜测。此模块只记录
本机可审计的运行 lineage，不把本机选择等同于 formal Gate。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..atomic import atomic_write_json
from ..lifecycle import validate_identifier
from ._jsonio import load_canonical_json


class AttemptDisposition(StrEnum):
    CANDIDATE = "CANDIDATE"
    CANONICAL = "CANONICAL"
    SUPERSEDED = "SUPERSEDED"
    ORPHAN = "ORPHAN"


@dataclass(frozen=True, slots=True)
class AttemptLineageRecord:
    """一个 attempt 在 run lineage 中的不可歧义身份。"""

    run_id: str
    attempt_id: str
    parent_attempt_id: str | None
    disposition: AttemptDisposition
    created_at: str
    updated_at: str
    reason: str
    evidence_hash: str | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class LineageStore:
    """以单个规范 JSON 文件维护一个 run 的 attempt 分支状态。"""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        validate_identifier(run_id, field="run_id")
        self.path = Path(path)
        self.run_id = run_id
        if self.path.exists():
            self._read()
        else:
            self._write([])

    def _read(self) -> list[AttemptLineageRecord]:
        value = load_canonical_json(self.path)
        if not isinstance(value, dict) or set(value) != {
            "schema_version",
            "run_id",
            "attempts",
        }:
            raise ValueError("LINEAGE_SCHEMA_FIELDS_MISMATCH")
        if value["schema_version"] != "runtime.attempt-lineage.v1":
            raise ValueError("LINEAGE_SCHEMA_MISMATCH")
        if value["run_id"] != self.run_id or not isinstance(value["attempts"], list):
            raise ValueError("LINEAGE_RUN_OR_ATTEMPTS_MISMATCH")
        records: list[AttemptLineageRecord] = []
        for item in value["attempts"]:
            if not isinstance(item, dict):
                raise ValueError("LINEAGE_RECORD_NOT_OBJECT")
            try:
                item = dict(item)
                item["disposition"] = AttemptDisposition(item["disposition"])
                record = AttemptLineageRecord(**item)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("LINEAGE_RECORD_INVALID") from exc
            validate_identifier(record.attempt_id, field="attempt_id")
            if record.parent_attempt_id is not None:
                validate_identifier(record.parent_attempt_id, field="parent_attempt_id")
            records.append(record)
        identifiers = [item.attempt_id for item in records]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("LINEAGE_DUPLICATE_ATTEMPT")
        if sum(item.disposition is AttemptDisposition.CANONICAL for item in records) > 1:
            raise ValueError("LINEAGE_MULTIPLE_CANONICAL_ATTEMPTS")
        return records

    def _write(self, records: list[AttemptLineageRecord]) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": "runtime.attempt-lineage.v1",
                "run_id": self.run_id,
                "attempts": [asdict(item) for item in records],
            },
        )

    def records(self) -> tuple[AttemptLineageRecord, ...]:
        return tuple(self._read())

    def register(
        self,
        attempt_id: str,
        *,
        parent_attempt_id: str | None,
        reason: str,
    ) -> AttemptLineageRecord:
        validate_identifier(attempt_id, field="attempt_id")
        records = self._read()
        if any(item.attempt_id == attempt_id for item in records):
            raise ValueError(f"LINEAGE_ATTEMPT_ALREADY_REGISTERED:{attempt_id}")
        if parent_attempt_id is not None:
            validate_identifier(parent_attempt_id, field="parent_attempt_id")
            if not any(item.attempt_id == parent_attempt_id for item in records):
                raise ValueError(f"LINEAGE_PARENT_UNKNOWN:{parent_attempt_id}")
        timestamp = _now()
        record = AttemptLineageRecord(
            run_id=self.run_id,
            attempt_id=attempt_id,
            parent_attempt_id=parent_attempt_id,
            disposition=AttemptDisposition.CANDIDATE,
            created_at=timestamp,
            updated_at=timestamp,
            reason=reason,
        )
        records.append(record)
        self._write(records)
        return record

    def mark_orphan(self, attempt_id: str, *, reason: str) -> None:
        self._replace_disposition(attempt_id, AttemptDisposition.ORPHAN, reason, None)

    def select_canonical(
        self, attempt_id: str, *, reason: str, evidence_hash: str
    ) -> AttemptLineageRecord:
        """选择唯一 canonical attempt，并显式 supersede 其他候选分支。"""

        if len(evidence_hash) != 64 or any(
            char not in "0123456789abcdef" for char in evidence_hash
        ):
            raise ValueError("LINEAGE_INVALID_EVIDENCE_HASH")
        records = self._read()
        target = next((item for item in records if item.attempt_id == attempt_id), None)
        if target is None:
            raise ValueError(f"LINEAGE_ATTEMPT_UNKNOWN:{attempt_id}")
        if target.disposition is AttemptDisposition.ORPHAN:
            raise ValueError("LINEAGE_ORPHAN_CANNOT_BE_CANONICAL")
        timestamp = _now()
        updated: list[AttemptLineageRecord] = []
        selected: AttemptLineageRecord | None = None
        for item in records:
            if item.attempt_id == attempt_id:
                selected = replace(
                    item,
                    disposition=AttemptDisposition.CANONICAL,
                    updated_at=timestamp,
                    reason=reason,
                    evidence_hash=evidence_hash,
                )
                updated.append(selected)
            elif item.disposition in {
                AttemptDisposition.CANDIDATE,
                AttemptDisposition.CANONICAL,
            }:
                updated.append(
                    replace(
                        item,
                        disposition=AttemptDisposition.SUPERSEDED,
                        updated_at=timestamp,
                        reason=f"superseded_by:{attempt_id}",
                    )
                )
            else:
                updated.append(item)
        assert selected is not None
        self._write(updated)
        return selected

    def _replace_disposition(
        self,
        attempt_id: str,
        disposition: AttemptDisposition,
        reason: str,
        evidence_hash: str | None,
    ) -> None:
        records = self._read()
        updated: list[AttemptLineageRecord] = []
        found = False
        for item in records:
            if item.attempt_id != attempt_id:
                updated.append(item)
                continue
            found = True
            if item.disposition is AttemptDisposition.CANONICAL:
                raise ValueError("LINEAGE_CANONICAL_RECLASSIFICATION_FORBIDDEN")
            updated.append(
                replace(
                    item,
                    disposition=disposition,
                    updated_at=_now(),
                    reason=reason,
                    evidence_hash=evidence_hash,
                )
            )
        if not found:
            raise ValueError(f"LINEAGE_ATTEMPT_UNKNOWN:{attempt_id}")
        self._write(updated)


__all__ = ["AttemptDisposition", "AttemptLineageRecord", "LineageStore"]
