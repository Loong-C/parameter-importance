#!/usr/bin/env python3
"""Produce the immutable S2.4 multi-round execution summary.

This is a post-run reader.  It never polls a queue and never starts, retries,
or changes a job.  A small source manifest names the already published
``queue-final.json`` artifacts for r21 and the r22 retry2/retry3 attempts;
this producer then follows the queue manifest and the six canonical output
cell roots, records missing final-status files explicitly, and publishes one
content-addressed summary.  The amendment binds this summary, rather than a
caller-supplied summary object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from param_importance_nlp.contracts import canonical_json_hash  # noqa: E402
from param_importance_nlp.experiments.stage2_s204_ids import (  # noqa: E402
    EXPECTED_CELL_IDS,
    cell_path_component,
)


SOURCE_SCHEMA_VERSION = "stage2-multi-round-execution-source-manifest-v1"
SUMMARY_SCHEMA_VERSION = "stage2-multi-round-execution-summary-v1"
QUEUE_SCHEMA_VERSION = "stage2-s204-r20-queue-final-v1"
QUEUE_MANIFEST_SCHEMA_VERSION = "stage2-s204-r20-queue-v1"
ALLOWED_FINAL_STATUSES = frozenset({"COMPLETE", "BLOCKED", "FAIL", "SKIPPED"})
FINAL_REFERENCE_KINDS = frozenset({"reference_result", "reference_convergence_report", "gate_record"})
APPROVED_GPU_UUIDS = frozenset(
    {
        "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
        "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
        "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
        "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    }
)
EXCLUDED_GPU_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
EXCLUDED_PCI = "0000:50:00.0"
EXPECTED_EXECUTION_COMMITS = {
    "r21": "44f934dd62d1b86fcb951230c81f3bfa647791aa",
    "r22-retry2": "7a347a1449d3e07e3de96619ba7ed005c53627db",
    "r22-retry3": "9e5c4315444530371678205d5ee5c3d549e7f084",
}
_QUEUE_SHA = hashlib.sha256


def _queue_hash(value: Mapping[str, Any]) -> str:
    """Match run_s204_r20_queue._hash (compact JSON, no trailing newline)."""

    payload = {key: item for key, item in value.items() if key != "artifact_hash"}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _runtime_status_hash(value: Mapping[str, Any]) -> str:
    """Match run_s204_formal._canonical_hash for final-status.json."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_under(root: Path, value: str | Path, field: str) -> Path:
    raw = Path(value)
    lexical = Path(os.path.abspath(os.fspath(raw if raw.is_absolute() else root / PurePosixPath(str(value)))))
    root_abs = Path(os.path.abspath(os.fspath(root)))
    try:
        lexical.relative_to(root_abs)
    except ValueError as error:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_OUTSIDE_DATA_ROOT") from error
    # Check the spelling before resolve(); checking only the resolved path would
    # allow a symlinked source to masquerade as an immutable in-root artifact.
    cursor = lexical
    while True:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_LINK_LIKE")
        if cursor == root_abs:
            break
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    resolved = lexical.resolve()
    try:
        resolved.relative_to(root_abs.resolve())
    except ValueError as error:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_OUTSIDE_DATA_ROOT") from error
    return resolved


def _logical_ref(root: Path, path: Path, field: str) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_OUTSIDE_DATA_ROOT") from error
    text = PurePosixPath(relative.as_posix()).as_posix()
    if not text or any(part in {"", ".", ".."} for part in PurePosixPath(text).parts):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_INVALID")
    return text


def _load(path: Path, field: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_UNREADABLE") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_OBJECT_REQUIRED")
    return value


def _sha_source(root: Path, path: Path, field: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_MISSING")
    return {
        "ref": _logical_ref(root, path, field),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _source_ref(value: object, field: str) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"ref", "sha256"}:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_SOURCE_INVALID")
    ref, digest = value.get("ref"), value.get("sha256")
    if not isinstance(ref, str) or not ref or "\\" in ref or PurePosixPath(ref).is_absolute() or ".." in PurePosixPath(ref).parts:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_REF_INVALID")
    if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_SHA256_INVALID")
    return ref, digest


def _verify_source(root: Path, value: object, field: str) -> tuple[Path, dict[str, str]]:
    ref, digest = _source_ref(value, field)
    path = _safe_under(root, ref, field)
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_HASH_MISMATCH")
    return path, {"ref": ref, "sha256": digest}


def _validate_blockers(value: object, field: str) -> tuple[list[Mapping[str, Any]], bool]:
    if value is None:
        return [], False
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_INVALID")
    rows = [item for item in value if isinstance(item, Mapping)]
    if any(type(item.get("retryable")) is not bool for item in rows):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_RETRYABLE_MISSING")
    return rows, any(item.get("retryable") is True for item in rows)


def _validate_uuid_set(value: object, field: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or len(value) != len(APPROVED_GPU_UUIDS)
        or len(set(value)) != len(APPROVED_GPU_UUIDS)
        or set(value) != set(APPROVED_GPU_UUIDS)
    ):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_GPU_SET_INVALID")


def _validate_queue_final(root: Path, source: object, *, round_id: str, attempt_id: str) -> tuple[Mapping[str, Any], dict[str, str], Mapping[str, Any], dict[str, str], Path]:
    queue_path, queue_source = _verify_source(root, source, f"{round_id}_{attempt_id}_queue_final")
    queue = _load(queue_path, f"{round_id}_{attempt_id}_queue_final")
    if queue.get("schema_version") != QUEUE_SCHEMA_VERSION or queue.get("artifact_hash") != _queue_hash(queue):
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_QUEUE_HASH_INVALID")
    if queue.get("status") not in {"COMPLETE", "FAILED"} or queue.get("retry_policy") != "none":
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_QUEUE_TERMINAL_INVALID")
    if queue.get("execution_commit") != EXPECTED_EXECUTION_COMMITS.get(attempt_id):
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_EXECUTION_COMMIT_INVALID")
    _validate_uuid_set(queue.get("approved_gpu_uuids"), f"{round_id}_{attempt_id}_queue")
    for field, expected in (("excluded_gpu_uuid", EXCLUDED_GPU_UUID), ("excluded_pci", EXCLUDED_PCI)):
        if field in queue and queue.get(field) != expected:
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_QUEUE_{field.upper()}_INVALID")
    manifest_path = queue_path.parent / "queue-manifest.json"
    manifest_source = _sha_source(root, manifest_path, f"{round_id}_{attempt_id}_queue_manifest")
    manifest = _load(manifest_path, f"{round_id}_{attempt_id}_queue_manifest")
    if manifest.get("schema_version") != QUEUE_MANIFEST_SCHEMA_VERSION or manifest.get("artifact_hash") != _queue_hash(manifest):
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_QUEUE_MANIFEST_INVALID")
    _validate_uuid_set(manifest.get("approved_gpu_uuids"), f"{round_id}_{attempt_id}_manifest")
    if manifest.get("excluded_gpu_uuid") != EXCLUDED_GPU_UUID or manifest.get("excluded_pci") != EXCLUDED_PCI:
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_MANIFEST_EXCLUSION_INVALID")
    if queue.get("run_id") != manifest.get("run_id") or queue.get("execution_commit") != manifest.get("execution_commit"):
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_QUEUE_LINEAGE_INVALID")
    outcomes = queue.get("outcomes")
    if not isinstance(outcomes, Mapping) or set(outcomes) != set(EXPECTED_CELL_IDS):
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_CELL_SET_INVALID")
    for cell_id in EXPECTED_CELL_IDS:
        outcome = outcomes[cell_id]
        if not isinstance(outcome, Mapping) or outcome.get("cell_id") != cell_id or outcome.get("retry") is not False:
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTCOME_INVALID:{cell_id}")
        if outcome.get("run_id") != queue.get("run_id"):
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTCOME_RUN_ID_INVALID:{cell_id}")
        if outcome.get("status") not in {"COMPLETE", "FAILED"} or not isinstance(outcome.get("returncode"), int):
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTCOME_TERMINAL_INVALID:{cell_id}")
        if outcome.get("gpu_uuid") not in APPROVED_GPU_UUIDS:
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTCOME_GPU_INVALID:{cell_id}")
        if outcome.get("execution_commit") != queue.get("execution_commit"):
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTCOME_LINEAGE_INVALID:{cell_id}")
    output_root_value = manifest.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_{attempt_id.upper()}_OUTPUT_ROOT_MISSING")
    output_root = _safe_under(root, output_root_value, f"{round_id}_{attempt_id}_output_root")
    return queue, queue_source, manifest, manifest_source, output_root


def _cell_summary(root: Path, output_root: Path, cell_id: str, *, queue: Mapping[str, Any]) -> dict[str, Any]:
    cell_root = output_root / cell_path_component(cell_id)
    paths = sorted(cell_root.rglob("final-status.json")) if cell_root.is_dir() else []
    if len(paths) > 1:
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_AMBIGUOUS:{cell_id}")
    outcome = queue["outcomes"][cell_id]
    if not paths:
        return {
            "cell_id": cell_id,
            "queue_status": outcome["status"],
            "status": "MISSING",
            "attempt_id": None,
            "status_ref": None,
            "task_result_ref": None,
            "task_result_hash": None,
            "retryable": False,
            "final_reference_created": False,
            "one_shot_ab_created": False,
        }
    final_path = paths[0].resolve()
    try:
        final_path.relative_to(cell_root.resolve())
    except ValueError as error:
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_OUTSIDE_CELL_ROOT:{cell_id}") from error
    relative_parts = final_path.relative_to(cell_root.resolve()).parts
    if len(relative_parts) != 3 or relative_parts[0] != "attempts" or relative_parts[2] != "final-status.json":
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_CANONICAL_PATH_INVALID:{cell_id}")
    final = _load(final_path, f"final_status_{cell_id}")
    if final.get("attempt_id") != relative_parts[1]:
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_ATTEMPT_PATH_MISMATCH:{cell_id}")
    status = final.get("status")
    if final.get("cell_id") != cell_id or status not in ALLOWED_FINAL_STATUSES:
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_INVALID:{cell_id}")
    if final.get("execution_commit") != queue.get("execution_commit"):
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_EXECUTION_COMMIT_INVALID:{cell_id}")
    final_gpu = final.get("gpu")
    if not isinstance(final_gpu, Mapping) or final_gpu.get("selected_uuid") != outcome.get("gpu_uuid"):
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_GPU_MISMATCH:{cell_id}")
    if final.get("artifact_hash") != _runtime_status_hash({key: item for key, item in final.items() if key != "artifact_hash"}):
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_HASH_INVALID:{cell_id}")
    if not isinstance(final.get("formal_eligible"), bool):
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_STATUS_FORMAL_FLAG_MISSING:{cell_id}")
    blockers, final_retryable = _validate_blockers(final.get("blockers"), f"final_status_{cell_id}")
    refs = final.get("artifact_refs", {})
    if not isinstance(refs, Mapping) or FINAL_REFERENCE_KINDS.intersection(str(key) for key in refs):
        raise ValueError(f"S204_MULTI_SUMMARY_FINAL_REFERENCE_PRESENT:{cell_id}")
    task_result_ref = final.get("task_result_ref")
    task_source: dict[str, str] | None = None
    task_hash: str | None = None
    if isinstance(task_result_ref, str) and task_result_ref:
        task_path = _safe_under(root, task_result_ref, f"task_result_{cell_id}")
        task = _load(task_path, f"task_result_{cell_id}")
        task_source = _sha_source(root, task_path, f"task_result_{cell_id}")
        task_hash_value = task.get("result_hash")
        if not isinstance(task_hash_value, str) or task_hash_value != final.get("task_result_hash") or task_hash_value != canonical_json_hash({key: item for key, item in task.items() if key != "result_hash"}):
            raise ValueError(f"S204_MULTI_SUMMARY_TASK_RESULT_HASH_INVALID:{cell_id}")
        if task.get("status") != status or task.get("formal_eligible") != final.get("formal_eligible"):
            raise ValueError(f"S204_MULTI_SUMMARY_TASK_RESULT_BINDING_INVALID:{cell_id}")
        task_refs = task.get("artifact_refs", {})
        if not isinstance(task_refs, Mapping) or FINAL_REFERENCE_KINDS.intersection(str(key) for key in task_refs):
            raise ValueError(f"S204_MULTI_SUMMARY_TASK_FINAL_REFERENCE_PRESENT:{cell_id}")
        _, task_retryable = _validate_blockers(task.get("blockers"), f"task_result_{cell_id}")
        final_retryable = final_retryable or task_retryable
        task_hash = task_hash_value
    elif status != "MISSING":
        raise ValueError(f"S204_MULTI_SUMMARY_TASK_RESULT_REF_MISSING:{cell_id}")
    return {
        "cell_id": cell_id,
        "queue_status": outcome["status"],
        "status": status,
        "attempt_id": final.get("attempt_id"),
        "status_ref": _sha_source(root, final_path, f"final_status_{cell_id}"),
        "task_result_ref": task_source,
        "task_result_hash": task_hash,
        "retryable": bool(final_retryable),
        "final_reference_created": False,
        "one_shot_ab_created": False,
    }


def _validate_source_manifest(value: Mapping[str, Any]) -> None:
    if value.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_SCHEMA_INVALID")
    if set(value) != {"schema_version", "study_id", "rounds", "no_pooling", "read_only", "artifact_hash"}:
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_FIELDS_INVALID")
    if value.get("study_id") != "stage2-s204" or value.get("no_pooling") is not True or value.get("read_only") is not True:
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_CONTROL_INVALID")
    rounds = value.get("rounds")
    if not isinstance(rounds, Mapping) or set(rounds) != {"r21", "r22"}:
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_ROUNDS_INVALID")
    expected_attempts = {"r21": {"r21"}, "r22": {"r22-retry2", "r22-retry3"}}
    for round_id, expected in expected_attempts.items():
        item = rounds[round_id]
        if not isinstance(item, Mapping) or set(item) != {"attempts"} or not isinstance(item["attempts"], list):
            raise ValueError(f"S204_MULTI_SUMMARY_SOURCE_{round_id.upper()}_ATTEMPTS_INVALID")
        observed: set[str] = set()
        observed_sequence: list[str] = []
        for attempt in item["attempts"]:
            if not isinstance(attempt, Mapping) or set(attempt) != {"attempt_id", "queue_final", "evaluation"}:
                raise ValueError(f"S204_MULTI_SUMMARY_SOURCE_{round_id.upper()}_ATTEMPT_INVALID")
            attempt_id = attempt.get("attempt_id")
            if not isinstance(attempt_id, str) or attempt_id in observed or attempt_id not in expected:
                raise ValueError(f"S204_MULTI_SUMMARY_SOURCE_{round_id.upper()}_ATTEMPT_ID_INVALID")
            observed.add(attempt_id)
            observed_sequence.append(attempt_id)
            _source_ref(attempt.get("queue_final"), f"{round_id}_{attempt_id}_queue_final")
            evaluation = attempt.get("evaluation")
            if round_id == "r22" and attempt_id == "r22-retry3":
                _source_ref(evaluation, f"{round_id}_{attempt_id}_evaluation")
            elif evaluation is not None:
                _source_ref(evaluation, f"{round_id}_{attempt_id}_evaluation")
        if observed != expected:
            raise ValueError(f"S204_MULTI_SUMMARY_SOURCE_{round_id.upper()}_ATTEMPT_SET_INVALID")
        expected_sequence = ["r21"] if round_id == "r21" else ["r22-retry2", "r22-retry3"]
        if observed_sequence != expected_sequence:
            raise ValueError(f"S204_MULTI_SUMMARY_SOURCE_{round_id.upper()}_ATTEMPT_ORDER_INVALID")
    artifact_hash = value.get("artifact_hash")
    if artifact_hash != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_HASH_INVALID")


def _verify_evaluation(root: Path, value: object, *, required: bool, field: str) -> dict[str, str] | None:
    if value is None:
        if required:
            raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_REQUIRED")
        return None
    path, source = _verify_source(root, value, field)
    evaluation = _load(path, field)
    if evaluation.get("schema_version") != "stage2-g23-reference-evaluation-v1":
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_SCHEMA_INVALID")
    if evaluation.get("status") != "BLOCKED" or evaluation.get("formal_eligible") is not False:
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_BLOCKED_REQUIRED")
    if evaluation.get("artifact_hash") != canonical_json_hash({key: item for key, item in evaluation.items() if key != "artifact_hash"}):
        raise ValueError(f"S204_MULTI_SUMMARY_{field.upper()}_HASH_INVALID")
    return source


def build_source_manifest(
    data_root: str | Path,
    *,
    r21_queue_final: str | Path,
    r22_retry2_queue_final: str | Path,
    r22_retry3_queue_final: str | Path,
    r22_retry3_evaluation: str | Path,
) -> dict[str, Any]:
    """Build the immutable source declaration from explicit artifact refs.

    The caller supplies only already-published queue/evaluation paths.  The
    producer reads each file, derives its content hash, and then validates the
    resulting declaration; no final summary or hand-entered digest is
    accepted.  Publishing remains a separate immutable ``--source-manifest-
    output`` operation in the CLI so the summary can bind that exact file.
    """

    root = Path(data_root).resolve()

    def source(value: str | Path, field: str) -> dict[str, str]:
        path = _safe_under(root, value, field)
        return _sha_source(root, path, field)

    body: dict[str, Any] = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "study_id": "stage2-s204",
        "no_pooling": True,
        "read_only": True,
        "rounds": {
            "r21": {
                "attempts": [
                    {
                        "attempt_id": "r21",
                        "queue_final": source(r21_queue_final, "r21_queue_final"),
                        "evaluation": None,
                    }
                ]
            },
            "r22": {
                "attempts": [
                    {
                        "attempt_id": "r22-retry2",
                        "queue_final": source(r22_retry2_queue_final, "r22_retry2_queue_final"),
                        "evaluation": None,
                    },
                    {
                        "attempt_id": "r22-retry3",
                        "queue_final": source(r22_retry3_queue_final, "r22_retry3_queue_final"),
                        "evaluation": source(r22_retry3_evaluation, "r22_retry3_evaluation"),
                    },
                ]
            },
        },
    }
    body["artifact_hash"] = canonical_json_hash(body)
    _validate_source_manifest(body)
    return body


def produce_multi_round_execution_summary(data_root: str | Path, source_manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    root = Path(data_root).resolve()
    if isinstance(source_manifest, Mapping):
        raise ValueError("S204_MULTI_SUMMARY_SOURCE_MANIFEST_REF_REQUIRED")
    else:
        source_path = _safe_under(root, source_manifest, "source_manifest")
        source = _load(source_path, "source_manifest")
        source_manifest_ref = _logical_ref(root, source_path, "source_manifest")
        source_manifest_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    _validate_source_manifest(source)
    rounds: dict[str, Any] = {}
    evaluations: dict[str, dict[str, str] | None] = {}
    for round_id in ("r21", "r22"):
        attempts: list[dict[str, Any]] = []
        for raw_attempt in source["rounds"][round_id]["attempts"]:  # type: ignore[index]
            attempt_id = str(raw_attempt["attempt_id"])
            queue, queue_source, manifest, manifest_source, output_root = _validate_queue_final(
                root, raw_attempt["queue_final"], round_id=round_id, attempt_id=attempt_id
            )
            if round_id == "r22" and attempt_id == "r22-retry2" and queue.get("status") != "FAILED":
                raise ValueError("S204_MULTI_SUMMARY_R22_RETRY2_QUEUE_FAILED_REQUIRED")
            cells = [_cell_summary(root, output_root, cell_id, queue=queue) for cell_id in EXPECTED_CELL_IDS]
            attempt_key = f"{round_id}:{attempt_id}"
            evaluation_source = _verify_evaluation(
                root,
                raw_attempt.get("evaluation"),
                required=round_id == "r22" and attempt_id == "r22-retry3",
                field=f"{round_id}_{attempt_id}_evaluation",
            )
            evaluations[attempt_key] = evaluation_source
            attempt_retryable = any(bool(cell["retryable"]) for cell in cells)
            attempts.append(
                {
                    "attempt_id": attempt_id,
                    "queue_final": queue_source,
                    "queue_manifest": manifest_source,
                    "queue_status": queue["status"],
                    "run_id": queue["run_id"],
                    "execution_commit": queue["execution_commit"],
                    "output_root": _logical_ref(root, output_root, f"{round_id}_{attempt_id}_output_root"),
                    "cells": cells,
                    "evaluation": evaluation_source,
                    "retryable": attempt_retryable,
                    "final_reference_created": False,
                    "one_shot_ab_created": False,
                }
            )
        rounds[round_id] = {"attempts": attempts}
    body: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "study_id": "stage2-s204",
        "status": "BLOCKED",
        "retryable": False,
        "no_pooling": True,
        "read_only": True,
        "prior_round_ids": ["r21", "r22"],
        "rounds": rounds,
        "evaluations": evaluations,
        "final_reference_created": False,
        "one_shot_ab_created": False,
    }
    body["source_manifest"] = {"ref": source_manifest_ref, "sha256": source_manifest_sha256}
    body["artifact_hash"] = canonical_json_hash(body)
    validate_multi_round_execution_summary(body)
    return body


def validate_multi_round_execution_summary(value: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "study_id", "status", "retryable", "no_pooling", "read_only",
        "prior_round_ids", "rounds", "evaluations", "final_reference_created", "one_shot_ab_created",
        "source_manifest", "artifact_hash",
    }
    if set(value) != required:
        raise ValueError("S204_MULTI_SUMMARY_FIELDS_INVALID")
    if value.get("schema_version") != SUMMARY_SCHEMA_VERSION or value.get("study_id") != "stage2-s204":
        raise ValueError("S204_MULTI_SUMMARY_SCHEMA_INVALID")
    if value.get("status") not in {"BLOCKED", "INCONCLUSIVE"} or value.get("retryable") is not False:
        raise ValueError("S204_MULTI_SUMMARY_STATUS_INVALID")
    if value.get("no_pooling") is not True or value.get("read_only") is not True:
        raise ValueError("S204_MULTI_SUMMARY_CONTROL_INVALID")
    if value.get("prior_round_ids") != ["r21", "r22"]:
        raise ValueError("S204_MULTI_SUMMARY_PRIORS_INVALID")
    if value.get("final_reference_created") is not False or value.get("one_shot_ab_created") is not False:
        raise ValueError("S204_MULTI_SUMMARY_FINAL_AB_INVALID")
    rounds = value.get("rounds")
    if not isinstance(rounds, Mapping) or set(rounds) != {"r21", "r22"}:
        raise ValueError("S204_MULTI_SUMMARY_ROUNDS_INVALID")
    if not isinstance(rounds["r21"], Mapping) or not isinstance(rounds["r22"], Mapping):
        raise ValueError("S204_MULTI_SUMMARY_ROUND_OBJECT_INVALID")
    for round_id, expected_ids in (("r21", {"r21"}), ("r22", {"r22-retry2", "r22-retry3"})):
        attempts = rounds[round_id].get("attempts")
        if not isinstance(attempts, list) or {item.get("attempt_id") for item in attempts if isinstance(item, Mapping)} != expected_ids:
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_ATTEMPTS_INVALID")
        expected_sequence = ("r21",) if round_id == "r21" else ("r22-retry2", "r22-retry3")
        observed_sequence = tuple(item.get("attempt_id") for item in attempts if isinstance(item, Mapping))
        if observed_sequence != expected_sequence:
            raise ValueError(f"S204_MULTI_SUMMARY_{round_id.upper()}_ATTEMPT_ORDER_INVALID")
        for attempt in attempts:
            if not isinstance(attempt, Mapping):
                raise ValueError("S204_MULTI_SUMMARY_ATTEMPT_OBJECT_INVALID")
            if type(attempt.get("retryable")) is not bool or attempt.get("final_reference_created") is not False or attempt.get("one_shot_ab_created") is not False:
                raise ValueError("S204_MULTI_SUMMARY_ATTEMPT_CONTROL_INVALID")
            if round_id == "r22" and attempt.get("attempt_id") == "r22-retry2" and attempt.get("queue_status") != "FAILED":
                raise ValueError("S204_MULTI_SUMMARY_R22_RETRY2_QUEUE_FAILED_REQUIRED")
            attempt_id = attempt.get("attempt_id")
            if attempt_id == "r22-retry3":
                evaluation = attempt.get("evaluation")
                if not isinstance(evaluation, Mapping) or set(evaluation) != {"ref", "sha256"}:
                    raise ValueError("S204_MULTI_SUMMARY_RETRY3_EVALUATION_REQUIRED")
            elif attempt.get("evaluation") is not None:
                raise ValueError("S204_MULTI_SUMMARY_EVALUATION_MUST_BE_NULL")
            cells = attempt.get("cells")
            if not isinstance(cells, list) or tuple(item.get("cell_id") for item in cells if isinstance(item, Mapping)) != EXPECTED_CELL_IDS:
                raise ValueError("S204_MULTI_SUMMARY_CELL_SET_INVALID")
            for cell in cells:
                if not isinstance(cell, Mapping) or type(cell.get("retryable")) is not bool or cell.get("final_reference_created") is not False or cell.get("one_shot_ab_created") is not False:
                    raise ValueError("S204_MULTI_SUMMARY_CELL_CONTROL_INVALID")
                if cell.get("status") not in ALLOWED_FINAL_STATUSES | {"MISSING"}:
                    raise ValueError("S204_MULTI_SUMMARY_CELL_STATUS_INVALID")
    evaluations = value.get("evaluations")
    if not isinstance(evaluations, Mapping) or set(evaluations) != {"r21:r21", "r22:r22-retry2", "r22:r22-retry3"}:
        raise ValueError("S204_MULTI_SUMMARY_EVALUATION_INVALID")
    if evaluations.get("r21:r21") is not None or evaluations.get("r22:r22-retry2") is not None or not isinstance(evaluations.get("r22:r22-retry3"), Mapping):
        raise ValueError("S204_MULTI_SUMMARY_EVALUATION_NULL_OR_RETRY3_INVALID")
    _source_ref(value["source_manifest"], "source_manifest")
    if value.get("artifact_hash") != canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"}):
        raise ValueError("S204_MULTI_SUMMARY_HASH_INVALID")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError("S204_MULTI_SUMMARY_OUTPUT_OVERWRITE_FORBIDDEN")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish immutable S2.4 r21/r22 multi-round execution summary")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, help="existing immutable source manifest")
    parser.add_argument("--source-manifest-output", type=Path, help="publish a derived source manifest here")
    parser.add_argument("--r21-queue-final", type=Path)
    parser.add_argument("--r22-retry2-queue-final", type=Path)
    parser.add_argument("--r22-retry3-queue-final", type=Path)
    parser.add_argument("--r22-retry3-evaluation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        explicit = (
            args.r21_queue_final,
            args.r22_retry2_queue_final,
            args.r22_retry3_queue_final,
            args.r22_retry3_evaluation,
        )
        if args.source_manifest is not None and any(item is not None for item in explicit):
            raise ValueError("S204_MULTI_SUMMARY_SOURCE_INPUT_MODES_EXCLUSIVE")
        if args.source_manifest is None and not all(item is not None for item in explicit):
            raise ValueError("S204_MULTI_SUMMARY_EXPLICIT_SOURCE_REFS_REQUIRED")
        if args.source_manifest is not None:
            source_path = _safe_under(args.data_root.resolve(), args.source_manifest, "source_manifest")
        else:
            if args.source_manifest_output is None:
                raise ValueError("S204_MULTI_SUMMARY_SOURCE_MANIFEST_OUTPUT_REQUIRED")
            source_value = build_source_manifest(
                args.data_root,
                r21_queue_final=args.r21_queue_final,
                r22_retry2_queue_final=args.r22_retry2_queue_final,
                r22_retry3_queue_final=args.r22_retry3_queue_final,
                r22_retry3_evaluation=args.r22_retry3_evaluation,
            )
            source_output = _safe_under(args.data_root.resolve(), args.source_manifest_output, "source_manifest_output")
            _write_immutable(source_output, source_value)
            source_path = source_output
        result = produce_multi_round_execution_summary(args.data_root, source_path)
        output = _safe_under(args.data_root.resolve(), args.output, "output")
        _write_immutable(output, result)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"S2.4 multi-round summary blocked: {error}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SOURCE_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "build_source_manifest",
    "produce_multi_round_execution_summary",
    "validate_multi_round_execution_summary",
]
