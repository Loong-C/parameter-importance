"""Hash-bound canonical event segments across resumed sessions."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from ..atomic import atomic_write_bytes, sha256_file, stable_json_bytes
from ..contracts.jsonio import JSONValue, canonical_json_hash, write_canonical_json
from .events import EventRecord, EventType, read_event_stream


def _logical_path(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise ValueError(f"EVENT_LINEAGE_REF_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise ValueError(f"EVENT_LINEAGE_REF_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"EVENT_LINEAGE_REF_ESCAPE:{field}") from error
    return path


def build_canonical_event_lineage(
    *,
    root: str | Path,
    run_id: str,
    segments: Sequence[Mapping[str, JSONValue]],
    superseded_tails: Sequence[Mapping[str, JSONValue]],
    parent_checkpoint_ref: str | None,
    output_ref: str,
    canonical_event_ref: str,
) -> dict[str, JSONValue]:
    """Select immutable sequence ranges and materialize one canonical JSONL view."""

    workspace = Path(root).resolve(strict=True)
    if not segments:
        raise ValueError("EVENT_LINEAGE_SEGMENTS_EMPTY")
    selected_events: list[EventRecord] = []
    normalized_segments: list[dict[str, JSONValue]] = []
    selected_sessions: set[str] = set()
    for index, raw in enumerate(segments):
        segment = dict(raw)
        if set(segment) != {
            "attempt_id",
            "session_id",
            "rank",
            "event_ref",
            "event_sha256",
            "sequence_start",
            "sequence_end",
            "checkpoint_ref",
        }:
            raise ValueError("EVENT_LINEAGE_SEGMENT_FIELDS_INVALID")
        path = _logical_path(workspace, segment["event_ref"], field=f"segment[{index}]")
        if sha256_file(path) != segment["event_sha256"]:
            raise ValueError("EVENT_LINEAGE_RAW_STREAM_HASH_MISMATCH")
        stream = read_event_stream(path)
        start = segment["sequence_start"]
        end = segment["sequence_end"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end >= len(stream)
        ):
            raise ValueError("EVENT_LINEAGE_SEQUENCE_RANGE_INVALID")
        chosen = stream[start : end + 1]
        if any(
            event.run_id != run_id
            or event.session_id != segment["session_id"]
            or event.attempt_id != segment["attempt_id"]
            or event.rank != segment["rank"]
            for event in chosen
        ):
            raise ValueError("EVENT_LINEAGE_EVENT_IDENTITY_MISMATCH")
        session_id = str(segment["session_id"])
        if session_id in selected_sessions:
            raise ValueError("EVENT_LINEAGE_SESSION_SELECTED_TWICE")
        selected_sessions.add(session_id)
        selected_events.extend(chosen)
        normalized_segments.append({**segment, "disposition": "CANONICAL"})

    event_ids = [event.event_id for event in selected_events]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("EVENT_LINEAGE_DUPLICATE_EVENT_ID")
    optimizer_steps = [
        int(event.payload["global_step"])
        for event in selected_events
        if event.rank == 0 and event.event_type == EventType.OPTIMIZER_STEP.value
    ]
    if optimizer_steps and optimizer_steps != list(
        range(optimizer_steps[0], optimizer_steps[-1] + 1)
    ):
        raise ValueError("EVENT_LINEAGE_OPTIMIZER_STEP_GAP_OR_DUPLICATE")
    normalized_tails: list[dict[str, JSONValue]] = []
    for raw in superseded_tails:
        tail = dict(raw)
        if set(tail) != {
            "attempt_id",
            "session_id",
            "event_ref",
            "event_sha256",
            "sequence_start",
            "sequence_end",
            "reason",
        }:
            raise ValueError("EVENT_LINEAGE_SUPERSEDED_FIELDS_INVALID")
        path = _logical_path(workspace, tail["event_ref"], field="superseded.event_ref")
        if sha256_file(path) != tail["event_sha256"]:
            raise ValueError("EVENT_LINEAGE_SUPERSEDED_HASH_MISMATCH")
        stream = read_event_stream(path)
        start = tail["sequence_start"]
        end = tail["sequence_end"]
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start < 0
            or end < start
            or end >= len(stream)
        ):
            raise ValueError("EVENT_LINEAGE_SUPERSEDED_RANGE_INVALID")
        normalized_tails.append({**tail, "disposition": "SUPERSEDED"})
    canonical_events = [
        replace(event, sequence=index) for index, event in enumerate(selected_events)
    ]
    canonical_path = _logical_path(
        workspace, canonical_event_ref, field="canonical_event_ref"
    )
    canonical_bytes = b"".join(
        stable_json_bytes(asdict(event)) for event in canonical_events
    )
    atomic_write_bytes(canonical_path, canonical_bytes)
    # Re-read the derived stream through the same untrusted-input validator.
    if read_event_stream(canonical_path) != canonical_events:
        raise ValueError("EVENT_LINEAGE_CANONICAL_REPLAY_MISMATCH")
    payload: dict[str, JSONValue] = {
        "schema_version": "runtime.canonical-event-lineage.v1",
        "run_id": run_id,
        "parent_checkpoint_ref": parent_checkpoint_ref,
        "segments": normalized_segments,
        "superseded_tails": normalized_tails,
        "canonical_event_ref": canonical_event_ref,
        "canonical_event_sha256": sha256_file(canonical_path),
        "canonical_event_count": len(selected_events),
        "optimizer_steps": optimizer_steps,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    write_canonical_json(
        _logical_path(workspace, output_ref, field="output_ref"), payload
    )
    return payload


__all__ = ["build_canonical_event_lineage"]
