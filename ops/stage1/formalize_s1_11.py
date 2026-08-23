"""Atomically publish S1.11 only from immutable, already-measured evidence.

This command deliberately has no test, CUDA, torchrun, training, or oracle
entrypoint.  It audits frozen producer files, derives tabular/chart/report
views, replays the source closure, and atomically publishes only on success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import math
import types
from typing import Any, Mapping
from xml.etree import ElementTree

REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)
from param_importance_nlp.stage1_exit_gate import (
    GATE_ID, REQUIRED_CHARTS, REQUIREMENT_IDS, TASK_ID, Stage1ExitGateError,
)


class Stage1S111FormalError(RuntimeError):
    """A failed preflight/publish is never a G1-EXIT result."""


# This is the released authority selected by the Stage 2 handoff contract.
# Emit mode deliberately cannot accept a caller-selected index, an r3 attempt,
# or a tracked/local fixture that merely has PASS-looking JSON.
S111_R4_PRODUCER_COMMIT = "3f18b04df8922be9894678ae4842bd999c7e8fd5"
S111_R4_ATTEMPT_ID = "s1-11-r4-20260821"
S111_R4_INDEX_SHA256 = "6a9af7323df37a55bf4f74f2f505d8999dcf4c588dfd64bd6cdc6f93fbecd1c8"
S111_R4_INDEX_ARTIFACT_HASH = "361de10b9aba20f5be59cd3ed18b44b16ea98167845b4cf735b299bea0403df0"
S111_R4_VALIDATOR_SOURCE_SHA256 = "c33e3e365bf3a66c203772de999d8ace0a153bfeb092365dca14f2b32d908320"
# This is a deployment identity, not the hash of the index.  Emit callers may
# explicitly supply a different approved root for an isolated test fixture;
# the root is still required to be an exact, non-symlink identity match.
S111_APPROVED_DATA_ROOT = "/home/sophgo13/cjl/storage/parameter-importance"
S111_R4_INDEX_REF = (
    "evidence/stage1/s1-11-formal/"
    f"{S111_R4_PRODUCER_COMMIT}/{S111_R4_ATTEMPT_ID}/index.json"
)
S111_TASK_OUTPUT_DEFAULT = (
    "evidence/stage1/tasks/11-s1-11-r4-20260821"
)
S110_COMPAT_OUTPUT_DEFAULT = "evidence/stage1/tasks/10-s1-10-r12-20260821"
S111_TASK_ARTIFACT_KINDS = (
    "stage_report",
    "requirements_matrix",
    "gate_summary",
    "delivery_manifest",
)

# G1-RESUME was published before TaskArtifactStore was required for the
# historical task.  These constants identify the one accepted formal r12
# evidence bundle from which compatibility commits may be reconstructed.
S110_R12_PRODUCER_COMMIT = "fbb09e4d338125954fc614c745cf7ab88c58d3b2"
S110_R12_ATTEMPT_ID = "s1-10-r12-20260821"
S110_R12_INDEX_REF = (
    "evidence/stage1/s1-10-formal/"
    f"{S110_R12_PRODUCER_COMMIT}/{S110_R12_ATTEMPT_ID}/index.json"
)
S110_R12_INDEX_SHA256 = "8dbcc2c0584bae316a56b22fc4da258cb71ef95d1334d0c091f400ef4dba95cc"
S110_R12_INDEX_ARTIFACT_HASH = "9368f7f853445e666c1be9a197e1766a5ba695cb7354863945634d05b9729797"
S110_R12_ROLE_REFS = {
    "artifact_manifest": "artifact-manifest.json",
    "comparison_table": "comparison-table.json",
    "gate_record": "g1-resume-record.json",
    "oracle_bundle": "oracle-bundle.json",
    "resume_report": "resume-report.json",
    "trace_bundle": "trace-bundle.json",
}
S110_TASK_ARTIFACT_KINDS = (
    "training_state_manifest",
    "resume_equivalence_report",
    "gate_record",
)
S110_TASK_ROLE_MAP = {
    "training_state_manifest": "artifact_manifest",
    "resume_equivalence_report": "resume_report",
    "gate_record": "gate_record",
}
S110_R12_SOURCE_PATHS = (
    "fixtures/stage1/stage1-s110-checkpoint-fixture-v1.json",
    "ops/stage1/formalize_s1_10.py",
    "ops/stage1/run_s1_10_resume_worker.py",
    "schemas/stage1/s1-10-artifact-manifest-v1.json",
    "schemas/stage1/s1-10-checkpoint-fixture-v1.json",
    "schemas/stage1/s1-10-comparison-table-v1.json",
    "schemas/stage1/s1-10-formal-observation-v1.json",
    "schemas/stage1/s1-10-formalization-index-v1.json",
    "schemas/stage1/s1-10-formalization-index-v2.json",
    "schemas/stage1/s1-10-gate-record-v1.json",
    "schemas/stage1/s1-10-oracle-bundle-v1.json",
    "schemas/stage1/s1-10-replay-validation-v1.json",
    "schemas/stage1/s1-10-resume-report-v1.json",
    "schemas/stage1/s1-10-resume-report-v2.json",
    "schemas/stage1/s1-10-trace-bundle-v1.json",
    "schemas/stage1/s1-10-validation-v1.json",
    "schemas/stage1/s1-10-validation-v2.json",
    "src/param_importance_nlp/runtime/checkpoint.py",
    "src/param_importance_nlp/runtime/checkpoint_group.py",
    "src/param_importance_nlp/runtime/training.py",
    "src/param_importance_nlp/stage1_checkpoint_oracle.py",
    "src/param_importance_nlp/stage1_checkpoint_resume.py",
    "tests/test_stage1_s110_checkpoint_resume.py",
)

# The S1.11 index is an exact wire contract.  Keeping this map in the producer
# (rather than accepting same-cardinality caller maps) prevents role
# substitution while retaining the historical r4 filenames.
S111_R4_ROLE_REFS: dict[str, str] = {
    "formal_observation": "formal-observation.json",
    "test_summary": "test-summary.json",
    "requirements_matrix": "requirements-matrix.json",
    "top_errors": "top-errors.json",
    "gate_summary": "gate-summary.json",
    "stage_report": "stage-report.json",
    "stage_report_markdown": "stage-report.md",
    "delivery_manifest": "delivery-manifest.json",
    "replay_validation": "replay-validation.json",
    **{
        f"{chart.replace('-', '_')}_{suffix}": f"{chart}.{suffix}"
        for chart in REQUIRED_CHARTS
        for suffix in ("csv", "svg")
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _object(path: Path, *, field: str) -> dict[str, Any]:
    value = load_canonical_json(path)
    if not isinstance(value, dict):
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_OBJECT_REQUIRED")
    return dict(value)


def _mapping(value: object, *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_MAPPING_REQUIRED")
    return dict(value)


def _relative(root: Path, ref: object, *, field: str) -> Path:
    if not isinstance(ref, str) or not ref or Path(ref).is_absolute():
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_REF_INVALID")
    candidate = (root / ref).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_REF_ESCAPES_ROOT") from error
    return candidate


def _write(path: Path, value: Mapping[str, object]) -> None:
    write_canonical_json(path, dict(value))


def _load_config(path: Path, *, field: str) -> Any:
    return load_canonical_json(path)


def _validate_bound_file(root: Path, value: object, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"ref", "sha256"}:
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_BINDING_INVALID")
    path = _relative(root, value["ref"], field=field)
    if not path.is_file() or not _digest(value["sha256"]) or _sha(path) != value["sha256"]:
        raise Stage1S111FormalError(f"S1_11_{field.upper()}_HASH_MISMATCH")
    return {"ref": str(value["ref"]), "sha256": str(value["sha256"])}


def _validate_failure_history(root: Path, value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise Stage1S111FormalError("S1_11_FAILURE_HISTORY_LIST_REQUIRED")
    rows: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {"failure_ref", "failure_sha256", "resolution_ref", "resolution_sha256"}:
            raise Stage1S111FormalError("S1_11_FAILURE_HISTORY_ROW_INVALID")
        row = {key: item[key] for key in item}
        for kind in ("failure", "resolution"):
            path = _relative(root, row[f"{kind}_ref"], field=f"history_{kind}")
            if not path.is_file() or not _digest(row[f"{kind}_sha256"]) or _sha(path) != row[f"{kind}_sha256"]:
                raise Stage1S111FormalError("S1_11_FAILURE_HISTORY_HASH_MISMATCH")
        failure = _object(_relative(root, row["failure_ref"], field="history_failure"), field="history_failure")
        resolution = _object(_relative(root, row["resolution_ref"], field="history_resolution"), field="history_resolution")
        if failure.get("status") not in {"FAIL", "FAILED"} or resolution.get("status") != "PASS" or resolution.get("failure_sha256") != row["failure_sha256"]:
            raise Stage1S111FormalError("S1_11_FAILURE_HISTORY_CLOSURE_INVALID")
        rows.append({key: str(value) for key, value in row.items()})
    return rows


def _validate_test_summary(root: Path, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"ref", "sha256"}:
        raise Stage1S111FormalError("S1_11_TEST_SUMMARY_BINDING_INVALID")
    path = _relative(root, value["ref"], field="test_summary")
    if not path.is_file() or not _digest(value["sha256"]) or _sha(path) != value["sha256"]:
        raise Stage1S111FormalError("S1_11_TEST_SUMMARY_HASH_MISMATCH")
    report = _object(path, field="test_summary")
    expected = {"schema_version", "status", "groups", "artifact_hash"}
    if set(report) != expected or report.get("schema_version") != "stage1-s1-11-test-summary-v1" or report.get("status") != "PASS":
        raise Stage1S111FormalError("S1_11_TEST_SUMMARY_SCHEMA_INVALID")
    body = dict(report); declared = body.pop("artifact_hash")
    if declared != canonical_json_hash(body) or not isinstance(report.get("groups"), list) or not report["groups"]:
        raise Stage1S111FormalError("S1_11_TEST_SUMMARY_SELF_HASH_INVALID")
    names: set[str] = set()
    for group in report["groups"]:
        if not isinstance(group, Mapping) or set(group) != {"name", "collected", "passed", "failed", "errors", "skipped", "duration_seconds", "junit_ref", "junit_sha256"}:
            raise Stage1S111FormalError("S1_11_TEST_SUMMARY_GROUP_INVALID")
        if not isinstance(group["name"], str) or not group["name"] or group["name"] in names or not isinstance(group["duration_seconds"], (int, float)) or isinstance(group["duration_seconds"], bool) or not math.isfinite(float(group["duration_seconds"])) or float(group["duration_seconds"]) < 0.0 or any(not isinstance(group[key], int) or isinstance(group[key], bool) or group[key] < 0 for key in ("collected", "passed", "failed", "errors", "skipped")) or group["passed"] + group["failed"] + group["errors"] + group["skipped"] != group["collected"] or group["failed"] or group["errors"]:
            raise Stage1S111FormalError("S1_11_TEST_SUMMARY_GROUP_COUNTS_INVALID")
        junit = _relative(path.parent, group["junit_ref"], field="junit")
        if not junit.is_file() or not _digest(group["junit_sha256"]) or _sha(junit) != group["junit_sha256"]:
            raise Stage1S111FormalError("S1_11_TEST_SUMMARY_JUNIT_HASH_MISMATCH")
        names.add(group["name"])
    return {"ref": str(value["ref"]), "sha256": str(value["sha256"]), "artifact_hash": str(report["artifact_hash"])}


def _validate_sync_audit(root: Path, value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != {"ref", "sha256"}:
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_BINDING_INVALID")
    path = _relative(root, value["ref"], field="sync_audit")
    if not path.is_file() or not _digest(value["sha256"]) or _sha(path) != value["sha256"]:
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_HASH_MISMATCH")
    audit = _object(path, field="sync_audit")
    expected = {"schema_version", "status", "git_publish", "server_execution", "agent_sha256", "large_artifact_manifest", "artifact_hash"}
    if set(audit) != expected or audit.get("schema_version") != "stage1-s1-11-sync-audit-v1" or audit.get("status") != "PASS":
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_SCHEMA_INVALID")
    body = dict(audit); declared = body.pop("artifact_hash")
    expected_agents = {"remote_access.md", "server.md", "git.md", "sync.md", "worklogs.md", "local_temp.md"}
    git_publish, server_execution, manifest = audit["git_publish"], audit["server_execution"], audit["large_artifact_manifest"]
    if not isinstance(git_publish, Mapping) or set(git_publish) != {"remote_ref", "execution_commit", "remote_commit", "worktree_clean"} or not isinstance(server_execution, Mapping) or set(server_execution) != {"execution_commit", "worktree_clean", "evidence_root"} or not isinstance(manifest, Mapping) or set(manifest) != {"ref", "sha256"}:
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_SHAPE_INVALID")
    manifest_path = _relative(path.parent, manifest["ref"], field="large_artifact_manifest")
    commits = (git_publish["execution_commit"], git_publish["remote_commit"], server_execution["execution_commit"])
    if declared != canonical_json_hash(body) or not isinstance(audit["agent_sha256"], Mapping) or set(audit["agent_sha256"]) != expected_agents or any(not _digest(digest) for digest in audit["agent_sha256"].values()) or any(not isinstance(commit, str) or len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit) for commit in commits) or len(set(commits)) != 1 or git_publish["worktree_clean"] is not True or server_execution["worktree_clean"] is not True or not isinstance(git_publish["remote_ref"], str) or not git_publish["remote_ref"] or not isinstance(server_execution["evidence_root"], str) or not server_execution["evidence_root"] or not manifest_path.is_file() or not _digest(manifest["sha256"]) or _sha(manifest_path) != manifest["sha256"]:
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_CLOSURE_INVALID")
    return {"ref": str(value["ref"]), "sha256": str(value["sha256"]), "artifact_hash": str(audit["artifact_hash"]), "execution_commit": str(git_publish["execution_commit"]), "large_artifact_manifest_ref": str(manifest["ref"]), "large_artifact_manifest_sha256": str(manifest["sha256"])}


def _schema_validate(
    repository: Path,
    objects: Mapping[str, Mapping[str, object]],
    *,
    producer_commit: str | None = None,
    frozen_source_sha256: Mapping[str, object] | None = None,
    schema_prefix: str = "s1-11",
    validator_source_path: str | None = None,
) -> None:
    """Validate roles against frozen producer bytes when formal evidence is read.

    The normal execute path retains its historical checkout-local validator.  Emit
    mode passes the released producer commit and source map, causing both the
    validator module and every schema byte to be loaded from Git objects.  This
    prevents a later checkout from silently changing the meaning of an old PASS.
    """

    from param_importance_nlp.contracts.jsonio import loads_strict_json

    validator_source_path = validator_source_path or "ops/stage1/formalize_s1_6.py"
    if producer_commit is None:
        validator_path = repository / validator_source_path
        try:
            validator_source = validator_path.read_bytes()
        except OSError as error:
            raise Stage1S111FormalError("S1_11_SCHEMA_VALIDATOR_UNAVAILABLE") from error
    else:
        if not isinstance(frozen_source_sha256, Mapping) or validator_source_path not in frozen_source_sha256:
            raise Stage1S111FormalError("S1_11_FROZEN_SCHEMA_SOURCE_MAP_INCOMPLETE")
        validator_source = _git_blob_bytes(repository, producer_commit, validator_source_path)
        if hashlib.sha256(validator_source).hexdigest() != frozen_source_sha256[validator_source_path]:
            raise Stage1S111FormalError("S1_11_FROZEN_VALIDATOR_SOURCE_HASH_MISMATCH")
    validator = types.ModuleType("_s111_frozen_schema_validator")
    validator.__file__ = validator_source_path
    try:
        exec(compile(validator_source, validator_source_path, "exec"), validator.__dict__)
    except Exception as error:
        raise Stage1S111FormalError("S1_11_SCHEMA_VALIDATOR_UNAVAILABLE") from error

    names = {
        "formal_observation": f"{schema_prefix}-formal-observation-v1.json",
        "requirements_matrix": f"{schema_prefix}-requirements-matrix-v1.json",
        "gate_summary": f"{schema_prefix}-gate-summary-v1.json",
        "stage_report": f"{schema_prefix}-stage-report-v1.json",
        "delivery_manifest": f"{schema_prefix}-delivery-manifest-v1.json",
        "replay": f"{schema_prefix}-replay-validation-v1.json",
        "validation": f"{schema_prefix}-validation-v2.json",
        "index": f"{schema_prefix}-formalization-index-v2.json",
    }
    if schema_prefix == "s1-10":
        names = {
            "formal_observation": "s1-10-formal-observation-v1.json",
            "resume_report": "s1-10-resume-report-v2.json",
            "artifact_manifest": "s1-10-artifact-manifest-v1.json",
            "gate_record": "s1-10-gate-record-v1.json",
            "oracle_bundle": "s1-10-oracle-bundle-v1.json",
            "trace_bundle": "s1-10-trace-bundle-v1.json",
            "comparison_table": "s1-10-comparison-table-v1.json",
            "replay": "s1-10-replay-validation-v1.json",
            "validation": "s1-10-validation-v2.json",
            "index": "s1-10-formalization-index-v2.json",
        }
    # S1.11's frozen schemas use v1 for index/validation; S1.10 uses v2.
    if schema_prefix == "s1-11":
        names["validation"] = "s1-11-validation-v1.json"
        names["index"] = "s1-11-formalization-index-v1.json"
    if producer_commit is None:
        paths = {
            path.name: path
            for path in (repository / "schemas" / "stage1").glob(f"{schema_prefix}-*.json")
        }
        source_bytes = {name: path.read_bytes() for name, path in paths.items()}
    else:
        source_bytes = {}
        prefix = f"schemas/stage1/{schema_prefix}-"
        for source in frozen_source_sha256:
            if str(source).startswith(prefix) and str(source).endswith(".json"):
                source_bytes[Path(str(source)).name] = _git_blob_bytes(
                    repository, producer_commit, str(source)
                )
    if not set(names.values()) <= set(source_bytes):
        raise Stage1S111FormalError("S1_11_SCHEMA_REGISTRY_INCOMPLETE")
    registry: dict[str, Mapping[str, object]] = {}
    for name, raw in source_bytes.items():
        try:
            schema = loads_strict_json(raw)
        except Exception as error:
            raise Stage1S111FormalError("S1_11_SCHEMA_INVALID") from error
        if not isinstance(schema, Mapping) or not isinstance(schema.get("$id"), str):
            raise Stage1S111FormalError("S1_11_SCHEMA_INVALID")
        registry[name] = schema
        registry[str(schema["$id"])] = schema
    for role, value in objects.items():
        name = names.get(role)
        if name is None:
            raise Stage1S111FormalError("S1_11_SCHEMA_ROLE_UNKNOWN")
        try:
            validator._validate_schema(
                value, registry[name], registry,
                document=registry[name], path=role,
            )
        except Exception as error:
            raise Stage1S111FormalError(
                f"S1_11_SCHEMA_VALIDATION_FAILED:{role}"
            ) from error


_SCATTERS = {"gradient-identity", "u-identity", "ddp-identity", "clip-factor"}
_HEATMAPS = {"module-metric-heatmap"}
_ERROR_BARS = {"noise-smoke"}
_IMPLEMENTATION_PATHS = (
    "ops/stage1/formalize_s1_11.py", "src/param_importance_nlp/stage1_exit_gate.py", "tests/test_stage1_s111_exit_gate.py",
    *[f"schemas/stage1/s1-11-{name}-v1.json" for name in ("delivery-manifest", "formal-observation", "formalization-index", "gate-summary", "replay-validation", "requirements-matrix", "stage-report", "sync-audit", "test-summary", "validation")],
)


def _exact_s111_index_closure(repository: Path, publication: Path, index: Mapping[str, object]) -> None:
    """Reject substitution of any same-cardinality S1.11 source or role key."""

    charts = tuple(REQUIRED_CHARTS)
    expected_roles = {
        "formal_observation", "test_summary", "requirements_matrix", "top_errors", "gate_summary", "stage_report", "stage_report_markdown", "delivery_manifest", "replay_validation",
        *{f"{chart.replace('-', '_')}_{suffix}" for chart in charts for suffix in ("csv", "svg")},
    }
    expected_role_refs = {
        "formal_observation": "formal-observation.json", "test_summary": "test-summary.json",
        "requirements_matrix": "requirements-matrix.json", "top_errors": "top-errors.json",
        "gate_summary": "gate-summary.json", "stage_report": "stage-report.json",
        "stage_report_markdown": "stage-report.md", "delivery_manifest": "delivery-manifest.json",
        "replay_validation": "replay-validation.json",
        **{f"{chart.replace('-', '_')}_{suffix}": f"{chart}.{suffix}" for chart in charts for suffix in ("csv", "svg")},
    }
    expected_csv = {f"{chart}.csv" for chart in charts}
    expected_svg = {f"{chart}.svg" for chart in charts}
    sources = index.get("implementation_source_sha256")
    refs, hashes = index.get("role_refs"), index.get("role_sha256")
    csv_hashes, svg_hashes = index.get("chart_csv_sha256"), index.get("chart_svg_sha256")
    reproduction_refs, reproduction_hashes = index.get("reproduction_role_refs"), index.get("reproduction_role_sha256")
    if not all(isinstance(value, Mapping) for value in (sources, refs, hashes, csv_hashes, svg_hashes, reproduction_refs, reproduction_hashes)):
        raise Stage1S111FormalError("S1_11_INDEX_CLOSURE_SHAPE_INVALID")
    if set(sources) != set(_IMPLEMENTATION_PATHS) or set(refs) != expected_roles or set(hashes) != expected_roles or set(csv_hashes) != expected_csv or set(svg_hashes) != expected_svg:
        raise Stage1S111FormalError("S1_11_INDEX_CLOSURE_KEYSET_INVALID")
    if dict(refs) != expected_role_refs:
        raise Stage1S111FormalError("S1_11_INDEX_ROLE_REF_WIRE_INVALID")
    if set(reproduction_refs) != {"test_summary", "sync_audit", "large_artifact_manifest", "worklog"} or set(reproduction_hashes) != set(reproduction_refs) or reproduction_refs.get("large_artifact_manifest") != "large-artifact-manifest.json":
        raise Stage1S111FormalError("S1_11_INDEX_REPRODUCTION_WIRE_INVALID")
    for path, digest in sources.items():
        if not _digest(digest) or _sha(repository / str(path)) != digest:
            raise Stage1S111FormalError("S1_11_INDEX_SOURCE_HASH_MISMATCH")
    for role, filename in refs.items():
        if not isinstance(filename, str) or not _digest(hashes[role]) or _sha(publication / filename) != hashes[role]:
            raise Stage1S111FormalError("S1_11_INDEX_ROLE_HASH_MISMATCH")
    for filename, digest in {**csv_hashes, **svg_hashes}.items():
        if not _digest(digest) or _sha(publication / str(filename)) != digest:
            raise Stage1S111FormalError("S1_11_INDEX_CHART_HASH_MISMATCH")
    if not _digest(reproduction_hashes["large_artifact_manifest"]) or _sha(publication / "large-artifact-manifest.json") != reproduction_hashes["large_artifact_manifest"]:
        raise Stage1S111FormalError("S1_11_INDEX_REPRODUCTION_HASH_MISMATCH")


def _derived_report_context(root: Path, audits: list[Mapping[str, object]], sync: Mapping[str, object], worklog: Mapping[str, str]) -> dict[str, object]:
    """Extract report claims only from roles already admitted by dependency audit."""

    by_gate = {str(row["binding"]["gate_id"]): _mapping(row["binding"], field="audit.binding") for row in audits}

    def index(gate: str) -> tuple[dict[str, object], Path, Mapping[str, object]]:
        binding = by_gate.get(gate)
        if binding is None:
            raise Stage1S111FormalError("S1_11_REPORT_DEPENDENCY_MISSING")
        path = _relative(root, binding.get("index_ref"), field=f"report_{gate}_index")
        if not path.is_file() or _sha(path) != binding.get("index_sha256"):
            raise Stage1S111FormalError("S1_11_REPORT_INDEX_HASH_MISMATCH")
        return _object(path, field=f"report_{gate}_index"), path, binding

    def indexed_role(gate: str, role: str) -> tuple[dict[str, object], str, str]:
        raw, path, _binding = index(gate)
        refs, hashes = raw.get("role_refs"), raw.get("role_sha256")
        if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or not isinstance(refs.get(role), str) or not _digest(hashes.get(role)):
            raise Stage1S111FormalError("S1_11_REPORT_ROLE_WIRE_INVALID")
        role_path = _relative(path.parent, refs[role], field=f"report_{gate}_{role}")
        if not role_path.is_file() or _sha(role_path) != hashes[role]:
            raise Stage1S111FormalError("S1_11_REPORT_ROLE_HASH_MISMATCH")
        return _object(role_path, field=f"report_{gate}_{role}"), role_path.relative_to(root).as_posix(), str(hashes[role])

    # G1-STEP is the measured AdamW boundary evidence, not an S1.11 slogan.
    step, step_ref, step_sha = indexed_role("G1-STEP", "step_report")
    checks = step.get("requirement_checks")
    if not isinstance(checks, Mapping) or checks.get("adamw_data_decay_total_decomposition") is not True or checks.get("actual_update_diagnostic_boundary") is not True:
        raise Stage1S111FormalError("S1_11_REPORT_ADAMW_BOUNDARY_MISSING")

    # S1.1's refs are DATA_ROOT-relative; use byte hashes for this report role.
    entry, _entry_path, _entry_binding = index("G1-ENTRY")
    provenance_paths: dict[str, Path] = {}
    for name in ("config_ref", "environment_ref", "result_ref"):
        candidate = _relative(root, entry.get(name), field=f"report_s1_1_{name}")
        if not candidate.is_file():
            raise Stage1S111FormalError("S1_11_REPORT_S1_1_PROVENANCE_MISSING")
        provenance_paths[name] = candidate

    registry_index, registry_index_path, _registry_binding = index("G1-REGISTRY")
    registry_ref, registry_sha = registry_index.get("report_ref"), registry_index.get("report_sha256")
    if not isinstance(registry_ref, str) or not _digest(registry_sha):
        raise Stage1S111FormalError("S1_11_REPORT_REGISTRY_WIRE_INVALID")
    registry_path = _relative(registry_index_path.parent, registry_ref, field="report_registry")
    registry = _object(registry_path, field="report_registry")
    registry_body = registry.get("registry")
    if not registry_path.is_file() or _sha(registry_path) != registry_sha or not isinstance(registry_body, Mapping) or not _digest(registry_body.get("coordinate_registry_hash")):
        raise Stage1S111FormalError("S1_11_REPORT_REGISTRY_PROVENANCE_MISSING")

    fixture, fixture_ref, fixture_sha = indexed_role("G1-SINGLE", "fixture_manifest")
    asset_identity = fixture.get("asset_identity")
    if not isinstance(asset_identity, Mapping) or not isinstance(asset_identity.get("model"), Mapping) or not isinstance(asset_identity.get("pile"), Mapping) or not _digest(asset_identity["model"].get("asset_id")) or not _digest(asset_identity["pile"].get("asset_id")):
        raise Stage1S111FormalError("S1_11_REPORT_ASSET_PROVENANCE_MISSING")

    ddp, ddp_ref, ddp_sha = indexed_role("G1-DDP", "ddp_report")
    baseline = ddp.get("baseline_routes")
    if not isinstance(baseline, Mapping) or not isinstance(baseline.get("D"), Mapping):
        raise Stage1S111FormalError("S1_11_REPORT_GPU_REPORT_MISSING")
    uuids = baseline["D"].get("visible_gpu_uuids")
    if not isinstance(uuids, list) or len(uuids) != 4 or len(set(uuids)) != 4 or any(not isinstance(value, str) or not value.startswith("GPU-") for value in uuids):
        raise Stage1S111FormalError("S1_11_REPORT_GPU_UUIDS_INVALID")
    ddp_index, ddp_index_path, _ddp_binding = index("G1-DDP")
    reproduction_refs, reproduction_hashes = ddp_index.get("reproduction_role_refs"), ddp_index.get("reproduction_role_sha256")
    if not isinstance(reproduction_refs, Mapping) or not isinstance(reproduction_hashes, Mapping) or not isinstance(reproduction_refs.get("post_lease_gpu"), str) or not _digest(reproduction_hashes.get("post_lease_gpu")):
        raise Stage1S111FormalError("S1_11_REPORT_GPU_PREFLIGHT_WIRE_INVALID")
    preflight_path = _relative(ddp_index_path.parent, reproduction_refs["post_lease_gpu"], field="report_gpu_preflight")
    preflight = _object(preflight_path, field="report_gpu_preflight")
    if not preflight_path.is_file() or _sha(preflight_path) != reproduction_hashes["post_lease_gpu"] or preflight.get("status") != "PASS" or not isinstance(preflight.get("gpu"), Mapping) or preflight["gpu"].get("requested_uuid_order") != uuids:
        raise Stage1S111FormalError("S1_11_REPORT_GPU_HEALTH_INVALID")

    ddp_fixture, _ddp_fixture_ref, _ddp_fixture_sha = indexed_role("G1-DDP", "fixture_manifest")
    scale = ddp_fixture.get("pre_route_gradient_scale")
    if not isinstance(scale, Mapping) or not _digest(scale.get("parameter_registry_hash")):
        raise Stage1S111FormalError("S1_11_REPORT_PARAMETER_REGISTRY_MISSING")

    audit_path = _relative(root, sync["ref"], field="sync_audit")
    audit = _object(audit_path, field="sync_audit")
    agents = audit.get("agent_sha256")
    if not isinstance(agents, Mapping) or set(agents) != {"remote_access.md", "server.md", "git.md", "sync.md", "worklogs.md", "local_temp.md"}:
        raise Stage1S111FormalError("S1_11_REPORT_AGENT_SYNC_MISSING")
    manifest_path = _relative(audit_path.parent, sync["large_artifact_manifest_ref"], field="large_artifact_manifest")
    manifest = _object(manifest_path, field="large_artifact_manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Stage1S111FormalError("S1_11_REPORT_LARGE_ARTIFACTS_MISSING")
    normalized: list[dict[str, object]] = []
    for item in artifacts:
        if not isinstance(item, Mapping) or set(item) != {"ref", "sha256", "size_bytes"} or not isinstance(item["ref"], str) or not _digest(item["sha256"]) or not isinstance(item["size_bytes"], int) or isinstance(item["size_bytes"], bool) or item["size_bytes"] < 1:
            raise Stage1S111FormalError("S1_11_REPORT_LARGE_ARTIFACT_ROW_INVALID")
        artifact_path = _relative(root, item["ref"], field="large_artifact")
        if not artifact_path.is_file() or _sha(artifact_path) != item["sha256"] or artifact_path.stat().st_size != item["size_bytes"]:
            raise Stage1S111FormalError("S1_11_REPORT_LARGE_ARTIFACT_HASH_MISMATCH")
        normalized.append({"ref": item["ref"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]})
    return {"adamw_boundary": {"step_report_ref": step_ref, "step_report_sha256": step_sha, "data_decay_total_decomposition": True, "actual_update_diagnostic_boundary": True}, "gpu_health": {"ddp_report_ref": ddp_ref, "ddp_report_sha256": ddp_sha, "preflight_ref": preflight_path.relative_to(root).as_posix(), "preflight_sha256": str(reproduction_hashes["post_lease_gpu"]), "approved_gpu_uuids": uuids}, "provenance": {"config_ref": str(entry["config_ref"]), "config_sha256": _sha(provenance_paths["config_ref"]), "environment_ref": str(entry["environment_ref"]), "environment_sha256": _sha(provenance_paths["environment_ref"]), "result_ref": str(entry["result_ref"]), "result_sha256": _sha(provenance_paths["result_ref"]), "registry_report_ref": registry_path.relative_to(root).as_posix(), "registry_report_sha256": str(registry_sha), "registry_coordinate_hash": str(registry_body["coordinate_registry_hash"]), "fixture_ref": fixture_ref, "fixture_sha256": fixture_sha, "model_asset_id": str(asset_identity["model"]["asset_id"]), "pile_asset_id": str(asset_identity["pile"]["asset_id"]), "parameter_registry_hash": str(scale["parameter_registry_hash"]), "execution_commit": str(entry["generator_git_commit"])}, "large_artifacts": normalized, "worklog_ref": worklog["ref"], "worklog_sha256": worklog["sha256"], "agent_sha256": {str(key): str(value) for key, value in agents.items()}}


def _svg_elements(svg: str, class_name: str) -> list[ElementTree.Element]:
    root = ElementTree.fromstring(svg)
    return [element for element in root.iter() if element.get("class") == class_name]


def _verify_chart_geometry(
    svg: str,
    *,
    chart_id: str,
    kind: str,
    points: list[tuple[str, str, float, float | None]],
    coordinates: list[tuple[float, float]] | None,
    heat_cells: list[tuple[float, float, float, float, int]] | None,
) -> None:
    """Re-parse SVG and recompute its required chart-specific geometry."""

    try:
        root = ElementTree.fromstring(svg)
    except ElementTree.ParseError as error:
        raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID") from error
    title = next((element.text for element in root.iter() if element.tag.rsplit("}", 1)[-1] == "title"), None)
    axes = _svg_elements(svg, "axes")
    if title != chart_id or len(axes) != 1 or len(list(axes[0])) != 2:
        raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
    if kind == "scatter":
        expected = [(f"{x:.3f}", f"{y:.3f}") for x, y in coordinates or []]
        observed = [(element.get("cx"), element.get("cy")) for element in _svg_elements(svg, "point")]
        identity = _svg_elements(svg, "identity-line")
        if observed != expected or len(identity) != 1 or identity[0].attrib != {"class": "identity-line", "x1": "40", "y1": "360", "x2": "600", "y2": "40"} or _svg_elements(svg, "series"):
            raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
    elif kind == "heatmap":
        expected_cells = [(f"{x:.3f}", f"{y:.3f}", f"{width:.3f}", f"{height:.3f}", f"rgb({red},0,0)") for x, y, width, height, red in heat_cells or []]
        observed_cells = [(element.get("x"), element.get("y"), element.get("width"), element.get("height"), element.get("fill")) for element in _svg_elements(svg, "heat-cell")]
        if observed_cells != expected_cells or _svg_elements(svg, "series") or _svg_elements(svg, "point"):
            raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
    else:
        expected_series = " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates or [])
        series = _svg_elements(svg, "series")
        if len(series) != 1 or series[0].get("points") != expected_series or _svg_elements(svg, "point"):
            raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
        if kind == "error":
            dy = max(max(point[2] for point in points) - min(point[2] for point in points), 1e-12)
            expected_bars = [
                (f"{x:.3f}", f"{y - 320 * float(error or 0.0) / dy:.3f}", f"{x:.3f}", f"{y + 320 * float(error or 0.0) / dy:.3f}")
                for (x, y), (_, _, _, error) in zip(coordinates or [], points, strict=True)
            ]
            observed_bars = [(element.get("x1"), element.get("y1"), element.get("x2"), element.get("y2")) for element in _svg_elements(svg, "error-bar")]
            if observed_bars != expected_bars:
                raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
        elif _svg_elements(svg, "error-bar"):
            raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")


def _render_chart(source: Path, *, chart_id: str, x_column: str, y_column: str, output_csv: Path, output_svg: Path, value_column: str | None = None, error_column: str | None = None, source_identity_sha256: str | None = None, allow_duplicate_keys: bool = False) -> dict[str, object]:
    """Render the plan-mandated geometry from a typed, finite CSV source."""

    kind = "scatter" if chart_id in _SCATTERS else "heatmap" if chart_id in _HEATMAPS else "error" if chart_id in _ERROR_BARS else "line"
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {x_column, y_column} | ({value_column} if kind == "heatmap" and value_column else set()) | ({error_column} if kind == "error" and error_column else set())
        if not reader.fieldnames or not required <= set(reader.fieldnames) or (kind == "heatmap" and value_column is None) or (kind == "error" and error_column is None):
            raise Stage1S111FormalError("S1_11_CHART_SOURCE_COLUMNS_INVALID")
        points: list[tuple[str, str, float, float | None]] = []
        for row in reader:
            try:
                x, y = row[x_column], row[y_column]
                if kind == "heatmap":
                    value = float(row[value_column])  # type: ignore[index]
                    error = None
                else:
                    value = float(y)
                    float(x)  # scatter/line/error use a numeric x axis
                    error = float(row[error_column]) if kind == "error" else None  # type: ignore[index]
                if not math.isfinite(value) or (error is not None and (not math.isfinite(error) or error < 0.0)) or (kind != "heatmap" and not math.isfinite(float(x))):
                    raise ValueError("non-finite")
                points.append((x, y, value, error))
            except (KeyError, TypeError, ValueError) as error:
                raise Stage1S111FormalError("S1_11_CHART_SOURCE_NUMERIC_DATA_INVALID") from error
    if not points:
        raise Stage1S111FormalError("S1_11_CHART_SOURCE_EMPTY")
    keys = [(point[0], point[1]) for point in points]
    if not allow_duplicate_keys and len(set(keys)) != len(keys):
        raise Stage1S111FormalError("S1_11_CHART_SOURCE_KEY_DUPLICATE")
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        header = [x_column, y_column] + ([value_column] if kind == "heatmap" else []) + ([error_column] if kind == "error" else [])
        writer.writerow(header)
        for x, y, value, error in points:
            writer.writerow([x, y] + ([value] if kind == "heatmap" else []) + ([error] if kind == "error" else []))
    coordinates: list[tuple[float, float]] | None = None
    heat_cells: list[tuple[float, float, float, float, int]] | None = None
    if kind == "heatmap":
        x_values, y_values = sorted({point[0] for point in points}), sorted({point[1] for point in points})
        maximum = max(abs(point[2]) for point in points) or 1.0
        heat_cells = [(40 + 520 * x_values.index(x) / max(1, len(x_values)), 40 + 300 * y_values.index(y) / max(1, len(y_values)), 520 / max(1, len(x_values)), 300 / max(1, len(y_values)), int(255 * abs(value) / maximum)) for x, y, value, _ in points]
        marks = "".join(f'<rect class="heat-cell" x="{x:.3f}" y="{y:.3f}" width="{width:.3f}" height="{height:.3f}" fill="rgb({red},0,0)"/>' for x, y, width, height, red in heat_cells)
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400"><title>{chart_id}</title><g class="axes"><line x1="40" y1="360" x2="600" y2="360"/><line x1="40" y1="40" x2="40" y2="360"/></g><g class="heatmap">{marks}</g></svg>\n'
    else:
        xs, ys = [float(point[0]) for point in points], [point[2] for point in points]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        dx, dy = max(max_x - min_x, 1e-12), max(max_y - min_y, 1e-12)
        coordinates = [(40 + 560 * (x - min_x) / dx, 360 - 320 * (y - min_y) / dy) for x, y in zip(xs, ys, strict=True)]
        if kind == "scatter":
            marks = "".join(f'<circle class="point" cx="{x:.3f}" cy="{y:.3f}" r="2"/>' for x, y in coordinates)
            geometry = '<line class="identity-line" x1="40" y1="360" x2="600" y2="40"/>' + marks
        elif kind == "error":
            marks = "".join(f'<line class="error-bar" x1="{x:.3f}" y1="{y - 320 * float(error or 0.0) / dy:.3f}" x2="{x:.3f}" y2="{y + 320 * float(error or 0.0) / dy:.3f}"/>' for (x, y), (_, _, _, error) in zip(coordinates, points, strict=True))
            geometry = '<polyline class="series" fill="none" points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates) + '"/>' + marks
        else:
            geometry = '<polyline class="series" fill="none" points="' + " ".join(f"{x:.3f},{y:.3f}" for x, y in coordinates) + '"/>'
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400"><title>{chart_id}</title><g class="axes"><line x1="40" y1="360" x2="600" y2="360"/><line x1="40" y1="40" x2="40" y2="360"/></g>{geometry}</svg>\n'
    output_svg.write_text(svg, encoding="utf-8", newline="\n")
    # Read back generated CSV and SVG rather than trusting the writer.
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_header = [x_column, y_column] + ([value_column] if kind == "heatmap" else []) + ([error_column] if kind == "error" else [])
        rows = list(reader)
        if reader.fieldnames != expected_header or len(rows) != len(points):
            raise Stage1S111FormalError("S1_11_CHART_READBACK_INVALID")
    _verify_chart_geometry(output_svg.read_text(encoding="utf-8"), chart_id=chart_id, kind=kind, points=points, coordinates=coordinates, heat_cells=heat_cells)
    return {"chart_id": chart_id, "source_csv_sha256": source_identity_sha256 or _sha(source), "row_count": len(points), "csv_ref": output_csv.name, "csv_sha256": _sha(output_csv), "svg_ref": output_svg.name, "svg_sha256": _sha(output_svg)}


def _project_role_rows(source: Path, destination: Path, columns: tuple[str, ...]) -> None:
    """Project an index-bound JSON table to CSV without inventing measurements."""

    value = _object(source, field="chart_role_source")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise Stage1S111FormalError("S1_11_CHART_ROLE_ROWS_INVALID")
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        projected = 0
        for ordinal, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise Stage1S111FormalError(f"S1_11_CHART_ROLE_ROW_INVALID:{ordinal}")
            present = tuple(column in row for column in columns)
            if not any(present):
                continue
            if not all(present):
                raise Stage1S111FormalError(f"S1_11_CHART_ROLE_ROW_INVALID:{ordinal}")
            writer.writerow([row[column] for column in columns])
            projected += 1
        if projected == 0:
            raise Stage1S111FormalError("S1_11_CHART_ROLE_ROWS_INVALID")


def _build_charts(evidence_root: Path, work: Path, dependencies: list[Mapping[str, object]], specs: object) -> list[dict[str, object]]:
    if not isinstance(specs, list) or len(specs) != len(REQUIRED_CHARTS):
        raise Stage1S111FormalError("S1_11_CHART_SPEC_COUNT_INVALID")
    bindings = {str(item["gate_id"]): item for item in dependencies}
    result: list[dict[str, object]] = []
    for expected, item in zip(REQUIRED_CHARTS, specs, strict=True):
        if not isinstance(item, Mapping) or set(item) != {"chart_id", "gate_id", "source_csv_ref", "source_csv_sha256", "x_column", "y_column", "value_column", "error_column"} or item.get("chart_id") != expected:
            raise Stage1S111FormalError("S1_11_CHART_SPEC_INVALID")
        binding = bindings.get(str(item["gate_id"]))
        if binding is None:
            raise Stage1S111FormalError("S1_11_CHART_SOURCE_GATE_UNKNOWN")
        index = _object(_relative(evidence_root, binding["index_ref"], field="chart_index"), field="chart_index")
        hashes = index.get("chart_csv_sha256", index.get("csv_sha256"))
        source_ref, source_sha = item["source_csv_ref"], item["source_csv_sha256"]
        role_refs, role_hashes = index.get("role_refs"), index.get("role_sha256")
        csv_bound = isinstance(hashes, Mapping) and Path(str(source_ref)).name in hashes and hashes[Path(str(source_ref)).name] == source_sha
        matching_roles = [] if not isinstance(role_refs, Mapping) or not isinstance(role_hashes, Mapping) else [
            role for role, ref in role_refs.items() if ref == source_ref and role_hashes.get(role) == source_sha
        ]
        role_bound = len(matching_roles) == 1
        if not csv_bound and not role_bound:
            raise Stage1S111FormalError("S1_11_CHART_SOURCE_INDEX_BINDING_INVALID")
        source = _relative(_relative(evidence_root, binding["index_ref"], field="chart_index").parent, source_ref, field="chart_source")
        if not source.is_file() or _sha(source) != source_sha:
            raise Stage1S111FormalError("S1_11_CHART_SOURCE_HASH_MISMATCH")
        render_source = source
        if role_bound:
            columns = tuple(dict.fromkeys(str(value) for value in (item["x_column"], item["y_column"], item["value_column"], item["error_column"]) if value is not None))
            render_source = work / f".{expected}.source.csv"
            _project_role_rows(source, render_source, columns)
        # Repeated plotted coordinates are legitimate when an immutable source
        # contains multiple parameter identities that project to the same 2-D
        # point.  The exact index/file hash remains the anti-tamper boundary.
        result.append(_render_chart(render_source, chart_id=expected, x_column=str(item["x_column"]), y_column=str(item["y_column"]), value_column=None if item["value_column"] is None else str(item["value_column"]), error_column=None if item["error_column"] is None else str(item["error_column"]), output_csv=work / f"{expected}.csv", output_svg=work / f"{expected}.svg", source_identity_sha256=str(source_sha), allow_duplicate_keys=True))
        if role_bound:
            render_source.unlink()
    return result


def _top_errors(evidence_root: Path, dependencies: list[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for binding in dependencies:
        index_path = _relative(evidence_root, binding["index_ref"], field="comparison_index")
        index = _object(index_path, field="comparison_index")
        refs, hashes = index.get("role_refs"), index.get("role_sha256")
        if not isinstance(refs, Mapping) or not isinstance(hashes, Mapping) or "comparison_table" not in refs:
            continue
        table = _object(_relative(index_path.parent, refs["comparison_table"], field="comparison_table"), field="comparison_table")
        if _sha(_relative(index_path.parent, refs["comparison_table"], field="comparison_table")) != hashes.get("comparison_table"):
            raise Stage1S111FormalError("S1_11_COMPARISON_ROLE_HASH_MISMATCH")
        candidates = table.get("rows", [])
        if not isinstance(candidates, list):
            continue
        for row in candidates:
            if not isinstance(row, Mapping):
                continue
            actual = row.get("actual") if isinstance(row.get("actual"), Mapping) else {}
            error = next((source[key] for source in (row, actual) for key in ("original_unit_max_abs_error", "max_abs_error", "max_absolute_error", "maximum_absolute_error", "absolute_error") if isinstance(source.get(key), (int, float)) and not isinstance(source.get(key), bool)), None)
            parameter = row.get("parameter_name", row.get("parameter", row.get("tensor_name", row.get("object_id", actual.get("object")))))
            coordinate = row.get("coordinate", row.get("worst_coordinate", row.get("comparison", row.get("comparison_id", row.get("field", actual.get("coordinate"))))))
            if error is not None and isinstance(parameter, str) and isinstance(coordinate, (str, int)):
                rows.append({"gate_id": binding["gate_id"], "parameter": parameter, "coordinate": str(coordinate), "max_abs_error": float(error)})
    rows.sort(key=lambda row: (-float(row["max_abs_error"]), str(row["gate_id"]), str(row["parameter"]), str(row["coordinate"])))
    if len(rows) < 20:
        raise Stage1S111FormalError("S1_11_TOP20_COMPARISON_COORDINATES_MISSING")
    return rows[:20]


def _emit_relative(root: Path, value: object, *, field: str) -> Path:
    """Resolve a producer path without accepting absolute/parent/symlink refs."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage1S111FormalError(f"S1_11_EMIT_{field.upper()}_REF_INVALID")
    logical = Path(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S111FormalError(f"S1_11_EMIT_{field.upper()}_REF_INVALID")
    candidate = root.joinpath(*logical.parts)
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S111FormalError(
            f"S1_11_EMIT_{field.upper()}_REF_ESCAPES_ROOT"
        ) from error
    return candidate


def _reject_symlink_tree(path: Path, *, field: str) -> None:
    """Reject symlinks in an authority/output tree before any write occurs."""

    if path.is_symlink():
        raise Stage1S111FormalError(f"S1_11_EMIT_{field.upper()}_SYMLINK_FORBIDDEN")
    if not path.exists():
        return
    if path.is_file():
        return
    for child in path.iterdir():
        if child.is_symlink():
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_SYMLINK_FORBIDDEN"
            )
        if child.is_dir():
            _reject_symlink_tree(child, field=field)


def _reject_symlink_ancestors(path: Path, root: Path, *, field: str) -> None:
    """Check only path ancestors, avoiding a recursive scan of DATA_ROOT."""

    current = path
    root = root.resolve()
    while True:
        if current.is_symlink():
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_SYMLINK_FORBIDDEN"
            )
        if current.resolve() == root:
            return
        if current.parent == current:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_ROOT_INVALID"
            )
        current = current.parent


def _validate_approved_data_root(
    evidence_root: Path, approved_data_root: Path | str | None,
) -> dict[str, object]:
    """Bind evidence to an explicitly approved DATA_ROOT identity.

    ``Path.resolve`` alone is insufficient: it would silently accept a symlink
    root and make a caller-selected directory look like the deployment root.
    Keep the pre-resolve symlink checks and record the canonical identity in
    every producer config/manifest.
    """

    raw_evidence = Path(evidence_root)
    approved = Path(approved_data_root or S111_APPROVED_DATA_ROOT)
    if raw_evidence.is_symlink() or approved.is_symlink():
        raise Stage1S111FormalError("S1_11_EMIT_DATA_ROOT_SYMLINK_FORBIDDEN")
    if not raw_evidence.is_dir() or not approved.is_dir():
        raise Stage1S111FormalError("S1_11_EMIT_DATA_ROOT_INVALID")
    real_evidence = raw_evidence.resolve()
    real_approved = approved.resolve()
    if real_evidence != real_approved:
        raise Stage1S111FormalError("S1_11_EMIT_DATA_ROOT_IDENTITY_MISMATCH")
    try:
        stat_result = raw_evidence.stat()
    except OSError as error:
        raise Stage1S111FormalError("S1_11_EMIT_DATA_ROOT_STAT_FAILED") from error
    return {
        "approved_ref": str(approved),
        "realpath": real_evidence.as_posix(),
        "device": int(stat_result.st_dev),
        "inode": int(stat_result.st_ino),
        "identity_hash": canonical_json_hash({
            "realpath": real_evidence.as_posix(),
            "device": int(stat_result.st_dev),
            "inode": int(stat_result.st_ino),
        }),
    }


def _emit_json(path: Path, value: Mapping[str, object], *, field: str) -> None:
    """Write once, then make every retry compare the exact canonical object."""

    if path.is_symlink():
        raise Stage1S111FormalError(f"S1_11_EMIT_{field.upper()}_SYMLINK_FORBIDDEN")
    if path.exists():
        try:
            existing = load_canonical_json(path)
        except (OSError, TypeError, ValueError) as error:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_EXISTING_INVALID"
            ) from error
        if existing != dict(value):
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_IDENTITY_DRIFT"
            )
        return
    _write(path, value)
    try:
        if load_canonical_json(path) != dict(value):
            raise Stage1S111FormalError(
                f"S1_11_EMIT_{field.upper()}_READBACK_MISMATCH"
            )
    except (OSError, TypeError, ValueError) as error:
        if isinstance(error, Stage1S111FormalError):
            raise
        raise Stage1S111FormalError(
            f"S1_11_EMIT_{field.upper()}_READBACK_INVALID"
        ) from error


def _git_blob_bytes(repository: Path, commit: str, source: str) -> bytes:
    """Read exact producer bytes from Git, never from the current checkout."""

    completed = subprocess.run(
        ["git", "-C", str(repository), "show", f"{commit}:{source}"],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise Stage1S111FormalError(
            f"S1_11_EMIT_PRODUCER_SOURCE_MISSING:{source}"
        )
    return bytes(completed.stdout)


def _git_blob_sha256(repository: Path, commit: str, source: str) -> str:
    """Hash the producer commit's bytes, not the current consumer worktree."""

    return hashlib.sha256(_git_blob_bytes(repository, commit, source)).hexdigest()


def _emit_load_r4(
    *, repository: Path, evidence_root: Path, evidence_ref: str,
    approved_data_root: Path | str | None = None,
) -> dict[str, object]:
    """Read and fully validate the released r4 publication before publishing."""

    data_root_identity = _validate_approved_data_root(
        evidence_root, approved_data_root,
    )
    if evidence_ref.replace("\\", "/") != S111_R4_INDEX_REF:
        raise Stage1S111FormalError("S1_11_EMIT_R4_AUTHORITY_REF_REQUIRED")
    index_path = _relative(evidence_root, evidence_ref, field="emit_r4_index")
    index_dir = index_path.parent
    if not index_path.is_file():
        raise Stage1S111FormalError("S1_11_EMIT_R4_INDEX_MISSING")
    # The canonical artifact_hash is a JSON claim; the file SHA binds the
    # actual authority bytes before parsing any caller-controlled JSON.
    if _sha(index_path) != S111_R4_INDEX_SHA256:
        raise Stage1S111FormalError("S1_11_EMIT_R4_INDEX_SHA256_MISMATCH")
    _reject_symlink_tree(index_dir, field="r4_authority")
    index = _object(index_path, field="emit_r4_index")
    expected_index_fields = {
        "schema_version", "status", "gate_id", "task_id",
        "generator_git_commit", "consumer_git_commit",
        "implementation_source_sha256", "role_refs", "role_sha256",
        "chart_csv_sha256", "chart_svg_sha256", "reproduction_role_refs",
        "reproduction_role_sha256", "validation_ref", "validation_sha256",
        "replay_ref", "replay_sha256", "next_task_ids", "artifact_hash",
    }
    if set(index) != expected_index_fields:
        raise Stage1S111FormalError("S1_11_EMIT_R4_INDEX_FIELDS_INVALID")
    body = dict(index)
    declared = body.pop("artifact_hash")
    if declared != canonical_json_hash(body):
        raise Stage1S111FormalError("S1_11_EMIT_R4_INDEX_SELF_HASH_INVALID")
    if (
        index.get("schema_version") != "stage1-s1-11-formalization-index-v1"
        or index.get("status") != "PASS"
        or index.get("gate_id") != GATE_ID
        or index.get("task_id") != TASK_ID
        or index.get("generator_git_commit") != S111_R4_PRODUCER_COMMIT
        or index.get("consumer_git_commit") != S111_R4_PRODUCER_COMMIT
        or index.get("next_task_ids") != ["stage2", "stage3"]
        or index.get("artifact_hash") != S111_R4_INDEX_ARTIFACT_HASH
    ):
        raise Stage1S111FormalError("S1_11_EMIT_R4_INDEX_IDENTITY_INVALID")

    source_map = index["implementation_source_sha256"]
    if not isinstance(source_map, Mapping) or set(source_map) != set(_IMPLEMENTATION_PATHS):
        raise Stage1S111FormalError("S1_11_EMIT_PRODUCER_SOURCE_KEYSET_INVALID")
    for source, digest in source_map.items():
        if not _digest(digest) or _git_blob_sha256(
            repository, S111_R4_PRODUCER_COMMIT, str(source)
        ) != digest:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_PRODUCER_SOURCE_HASH_MISMATCH:{source}"
            )

    role_refs = index["role_refs"]
    role_hashes = index["role_sha256"]
    if role_refs != S111_R4_ROLE_REFS or not isinstance(role_hashes, Mapping):
        raise Stage1S111FormalError("S1_11_EMIT_R4_ROLE_WIRE_INVALID")
    if set(role_hashes) != set(S111_R4_ROLE_REFS):
        raise Stage1S111FormalError("S1_11_EMIT_R4_ROLE_HASH_KEYSET_INVALID")
    role_values: dict[str, dict[str, object]] = {}
    role_file_hashes: dict[str, str] = {}
    for role, filename in S111_R4_ROLE_REFS.items():
        path = index_dir / filename
        if not path.is_file() or not _digest(role_hashes[role]) or _sha(path) != role_hashes[role]:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_R4_ROLE_HASH_MISMATCH:{role}"
            )
        role_file_hashes[role] = str(role_hashes[role])
        if filename.endswith(".json"):
            role_value = _object(path, field=f"emit_r4_role_{role}")
            role_body = dict(role_value)
            role_artifact_hash = role_body.pop("artifact_hash", None)
            if role_artifact_hash != canonical_json_hash(role_body):
                raise Stage1S111FormalError(
                    f"S1_11_EMIT_R4_ROLE_SELF_HASH_INVALID:{role}"
                )
            role_values[role] = role_value

    for ref_field, expected in (("validation_ref", "validation.json"), ("replay_ref", "replay-validation.json")):
        if index.get(ref_field) != expected or not _digest(index.get(ref_field.replace("_ref", "_sha256"))):
            raise Stage1S111FormalError("S1_11_EMIT_R4_VALIDATION_WIRE_INVALID")
        path = index_dir / expected
        if not path.is_file() or _sha(path) != index[ref_field.replace("_ref", "_sha256")]:
            raise Stage1S111FormalError("S1_11_EMIT_R4_VALIDATION_HASH_MISMATCH")
    validation = _object(index_dir / "validation.json", field="emit_r4_validation")
    replay = _object(index_dir / "replay-validation.json", field="emit_r4_replay")
    for value, schema, field in (
        (validation, "stage1-s1-11-validation-v1", "validation"),
        (replay, "stage1-s1-11-replay-validation-v1", "replay"),
    ):
        value_body = dict(value)
        value_hash = value_body.pop("artifact_hash", None)
        if value.get("schema_version") != schema or value.get("status") != "PASS" or value_hash != canonical_json_hash(value_body):
            raise Stage1S111FormalError(f"S1_11_EMIT_R4_{field.upper()}_INVALID")
    success_path = index_dir / "success.json"
    if not success_path.is_file():
        raise Stage1S111FormalError("S1_11_EMIT_R4_SUCCESS_MARKER_MISSING")
    success = _object(success_path, field="emit_r4_success")
    success_body = dict(success)
    success_hash = success_body.pop("artifact_hash", None)
    if (
        success.get("schema_version") != "stage1-s1-11-attempt-success-v1"
        or success.get("status") != "PASS"
        or success.get("index_sha256") != _sha(index_path)
        or success_hash != canonical_json_hash(success_body)
    ):
        raise Stage1S111FormalError("S1_11_EMIT_R4_SUCCESS_MARKER_INVALID")

    observation = role_values["formal_observation"]
    summary = role_values["gate_summary"]
    matrix = role_values["requirements_matrix"]
    report = role_values["stage_report"]
    delivery = role_values["delivery_manifest"]
    if (
        observation.get("schema_version") != "stage1-s1-11-formal-observation-v1"
        or observation.get("status") != "PASS"
        or observation.get("task_id") != TASK_ID
        or observation.get("gate_id") != GATE_ID
        or report.get("schema_version") != "stage1-s1-11-stage-report-v1"
        or report.get("status") != "PASS"
        or report.get("task_id") != TASK_ID
        or report.get("gate_id") != GATE_ID
        or summary.get("schema_version") != "stage1-s1-11-gate-summary-v1"
        or summary.get("task_id") != TASK_ID
        or summary.get("gate_id") != GATE_ID
        or matrix.get("schema_version") != "stage1-s1-11-requirements-matrix-v1"
        or delivery.get("schema_version") != "stage1-s1-11-delivery-manifest-v1"
        or delivery.get("task_id") != TASK_ID
        or delivery.get("gate_id") != GATE_ID
        or observation.get("execution_commit") != S111_R4_PRODUCER_COMMIT
        or summary.get("status") != "PASS"
        or summary.get("exit_verdict") != "STAGE1_COMPLETE"
        or summary.get("unresolved_failure_count") != 0
        or summary.get("formal_observation") != observation
        or matrix.get("task_id") != TASK_ID
        or matrix.get("gate_id") != GATE_ID
        or report.get("summary_hash") != summary.get("artifact_hash")
        or report.get("requirements_matrix_hash") != matrix.get("artifact_hash")
        or delivery.get("summary_hash") != summary.get("artifact_hash")
        or delivery.get("requirements_matrix_hash") != matrix.get("artifact_hash")
        or delivery.get("stage_report_hash") != report.get("artifact_hash")
        or delivery.get("dependency_index_hashes") != observation.get("dependency_index_sha256")
        or delivery.get("chart_ids") != list(REQUIRED_CHARTS)
        or summary.get("charts") != observation.get("charts")
        or replay.get("summary_artifact_hash") != summary.get("artifact_hash")
        or validation.get("role_sha256") != {
            filename: role_hashes[role]
            for role, filename in S111_R4_ROLE_REFS.items()
        }
    ):
        raise Stage1S111FormalError("S1_11_EMIT_R4_MUTUAL_HASH_CLOSURE_INVALID")

    from param_importance_nlp.stage1_exit_gate import (
        _validate_verification_matrix,
        audit_exit_dependencies,
    )
    audits = summary.get("dependency_audits")
    if not isinstance(audits, list) or len(audits) != 11:
        raise Stage1S111FormalError("S1_11_EMIT_R4_DEPENDENCY_CLOSURE_INVALID")
    bindings: list[Mapping[str, object]] = []
    for item in audits:
        if not isinstance(item, Mapping) or not isinstance(item.get("binding"), Mapping):
            raise Stage1S111FormalError("S1_11_EMIT_R4_DEPENDENCY_BINDING_INVALID")
        bindings.append(dict(item["binding"]))
    try:
        rebuilt = audit_exit_dependencies(evidence_root, bindings)
    except Exception as error:
        raise Stage1S111FormalError("S1_11_EMIT_R4_DEPENDENCY_REPLAY_FAILED") from error
    if rebuilt != audits:
        raise Stage1S111FormalError("S1_11_EMIT_R4_DEPENDENCY_REPLAY_MISMATCH")
    try:
        matrix_rows = _validate_verification_matrix(
            evidence_root, matrix.get("rows") if isinstance(matrix.get("rows"), list) else []
        )
    except Exception as error:
        raise Stage1S111FormalError("S1_11_EMIT_R4_REQUIREMENTS_INVALID") from error
    by_gate = {str(item["binding"]["gate_id"]): item["binding"] for item in audits}
    for row in matrix_rows:
        binding = by_gate.get(str(row["gate_id"]))
        if binding is None or row["evidence"] != {
            "ref": binding["index_ref"], "sha256": binding["index_sha256"]
        }:
            raise Stage1S111FormalError("S1_11_EMIT_R4_REQUIREMENTS_BINDING_INVALID")

    # Re-check the formal observation's test/sync source roles and every chart
    # file.  These calls also reject an internally self-consistent local copy.
    _validate_test_summary(evidence_root, observation["test_summary"])
    _validate_sync_audit(evidence_root, observation["sync_audit"])
    charts = observation.get("charts")
    if not isinstance(charts, list) or len(charts) != len(REQUIRED_CHARTS):
        raise Stage1S111FormalError("S1_11_EMIT_R4_CHART_CLOSURE_INVALID")
    observed_chart_ids: list[str] = []
    for chart in charts:
        if not isinstance(chart, Mapping) or chart.get("chart_id") not in REQUIRED_CHARTS:
            raise Stage1S111FormalError("S1_11_EMIT_R4_CHART_WIRE_INVALID")
        chart_id = str(chart["chart_id"])
        observed_chart_ids.append(chart_id)
        for suffix in ("csv", "svg"):
            role = f"{chart_id.replace('-', '_')}_{suffix}"
            if chart.get(f"{suffix}_ref") != S111_R4_ROLE_REFS[role] or chart.get(f"{suffix}_sha256") != role_hashes[role]:
                raise Stage1S111FormalError("S1_11_EMIT_R4_CHART_HASH_INVALID")
    if tuple(observed_chart_ids) != REQUIRED_CHARTS:
        raise Stage1S111FormalError("S1_11_EMIT_R4_CHART_ORDER_INVALID")

    # Schema validation is deliberately performed after the independent
    # closure checks, so schema drift cannot be hidden by a caller payload.
    _schema_validate(
        repository,
        {
            "formal_observation": observation,
            "requirements_matrix": matrix,
            "gate_summary": summary,
            "stage_report": report,
            "delivery_manifest": delivery,
            "replay": replay,
            "validation": validation,
            "index": index,
        },
        producer_commit=S111_R4_PRODUCER_COMMIT,
        frozen_source_sha256={
            **dict(source_map),
            "ops/stage1/formalize_s1_6.py": S111_R4_VALIDATOR_SOURCE_SHA256,
        },
        validator_source_path="ops/stage1/formalize_s1_6.py",
    )
    source_refs = sorted(
        path.relative_to(evidence_root.resolve()).as_posix()
        for path in index_dir.rglob("*")
        if path.is_file()
    )
    if not source_refs or S111_R4_INDEX_REF not in source_refs:
        raise Stage1S111FormalError("S1_11_EMIT_R4_SOURCE_REF_CLOSURE_INVALID")
    artifact_hashes = {
        role: str(role_values[role]["artifact_hash"])
        for role in role_values
    }
    artifact_hashes.update({
        "index": str(index["artifact_hash"]),
        "validation": str(validation["artifact_hash"]),
        "replay_validation": str(replay["artifact_hash"]),
        "success": str(success["artifact_hash"]),
    })
    return {
        "index": index,
        "index_path": index_path,
        "index_dir": index_dir,
        "roles": role_values,
        "role_file_sha256": role_file_hashes,
        "validation": validation,
        "replay": replay,
        "source_refs": source_refs,
        "artifact_hashes": artifact_hashes,
        "producer_source_sha256": dict(source_map),
        "dependency_index_sha256": dict(observation["dependency_index_sha256"]),
        "data_root_identity": data_root_identity,
    }


def _declared_role_hash(value: Mapping[str, object], *, role: str) -> str:
    """Return and verify the historical role's own canonical hash field."""

    for field in (
        "artifact_hash", "manifest_hash", "report_hash", "oracle_hash",
        "trace_hash", "table_hash", "replay_hash",
    ):
        if field in value:
            body = dict(value)
            declared = body.pop(field)
            if not _digest(declared) or declared != canonical_json_hash(body):
                raise Stage1S111FormalError(
                    f"S1_11_EMIT_S110_ROLE_SELF_HASH_INVALID:{role}"
                )
            return str(declared)
    raise Stage1S111FormalError(
        f"S1_11_EMIT_S110_ROLE_SELF_HASH_MISSING:{role}"
    )


def _emit_load_s110(
    *, repository: Path, evidence_root: Path,
    approved_data_root: Path | str | None = None,
) -> dict[str, object]:
    """Load the frozen G1-RESUME bundle used for compatibility commits.

    The historical task predates the three-kind TaskArtifactStore contract, so
    this function does not invent new PASS data.  It verifies the exact r12
    task directory, all six role files, validation/replay/success markers, and
    every historical producer source byte before returning the three roles that
    the task catalog names.
    """

    data_root_identity = _validate_approved_data_root(
        evidence_root, approved_data_root,
    )
    index_path = _emit_relative(evidence_root, S110_R12_INDEX_REF, field="s110_r12_index")
    index_dir = index_path.parent
    if not index_path.is_file():
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_INDEX_MISSING")
    if _sha(index_path) != S110_R12_INDEX_SHA256:
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_INDEX_SHA256_MISMATCH")
    _reject_symlink_tree(index_dir, field="s110_r12_authority")
    index = _object(index_path, field="emit_s110_r12_index")
    expected_index_fields = {
        "artifact_hash", "chart_csv_sha256", "chart_svg_sha256", "checked_at",
        "consumer_git_commit", "fixture_id", "formal_four_rank_report_ref",
        "formal_four_rank_report_sha256", "formal_observation_artifact_hash",
        "formal_observation_ref", "formal_observation_sha256",
        "formal_run_token_sha256", "formal_single_report_ref",
        "formal_single_report_sha256", "gate_artifact_hash", "gate_id",
        "generator_git_commit", "git_branch", "next_task_ids", "replay_hash",
        "replay_ref", "replay_sha256", "role_refs", "role_sha256",
        "schema_version", "status", "task_id", "upstream", "validation_ref",
        "validation_sha256",
    }
    if set(index) != expected_index_fields:
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_INDEX_FIELDS_INVALID")
    body = dict(index)
    declared_index_hash = body.pop("artifact_hash")
    if (
        declared_index_hash != canonical_json_hash(body)
        or declared_index_hash != S110_R12_INDEX_ARTIFACT_HASH
        or index.get("schema_version") != "stage1-s1-10-formalization-index-v2"
        or index.get("status") != "PASS"
        or index.get("task_id") != "stage1.10_checkpoint_resume_and_artifacts"
        or index.get("gate_id") != "G1-RESUME"
        or index.get("generator_git_commit") != S110_R12_PRODUCER_COMMIT
        or index.get("consumer_git_commit") != S110_R12_PRODUCER_COMMIT
        or index.get("fixture_id") != "stage1-s110-checkpoint-fixture-v1"
        or index.get("next_task_ids") != ["stage1.11_reporting_and_exit_gate"]
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_INDEX_IDENTITY_INVALID")
    if index.get("role_refs") != S110_R12_ROLE_REFS or not isinstance(index.get("role_sha256"), Mapping):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_ROLE_WIRE_INVALID")
    role_hashes = index["role_sha256"]
    if set(role_hashes) != set(S110_R12_ROLE_REFS) or any(
        not _digest(role_hashes.get(role)) for role in S110_R12_ROLE_REFS
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_ROLE_HASH_KEYSET_INVALID")

    roles: dict[str, dict[str, object]] = {}
    role_declared_hashes: dict[str, str] = {}
    for role, filename in S110_R12_ROLE_REFS.items():
        role_path = index_dir / filename
        if not role_path.is_file() or _sha(role_path) != role_hashes[role]:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_S110_R12_ROLE_HASH_MISMATCH:{role}"
            )
        role_value = _object(role_path, field=f"emit_s110_r12_{role}")
        role_declared_hashes[role] = _declared_role_hash(role_value, role=role)
        roles[role] = role_value

    # The index's independent role and report references must agree with the
    # exact role files, not merely with a same-shaped local fixture.
    observation = _object(index_dir / "formal-observation.json", field="emit_s110_observation")
    observation_declared_hash = _declared_role_hash(
        observation, role="formal_observation"
    )
    resume_report = roles["resume_report"]
    if (
        index.get("formal_observation_ref") != "formal-observation.json"
        or index.get("formal_observation_sha256") != _sha(index_dir / "formal-observation.json")
        or index.get("formal_observation_artifact_hash") != observation_declared_hash
        or index.get("formal_single_report_ref") != "formal-single-report.json"
        or index.get("formal_four_rank_report_ref") != "formal-four-rank-report.json"
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_OBSERVATION_WIRE_INVALID")
    for ref_field, hash_field, filename in (
        ("formal_single_report_ref", "formal_single_report_sha256", "formal-single-report.json"),
        ("formal_four_rank_report_ref", "formal_four_rank_report_sha256", "formal-four-rank-report.json"),
    ):
        path = index_dir / filename
        if not path.is_file() or index.get(hash_field) != _sha(path):
            raise Stage1S111FormalError("S1_11_EMIT_S110_R12_REPORT_HASH_MISMATCH")
    if (
        observation.get("schema_version") != "stage1-s1-10-formal-observation-v1"
        or observation.get("status") != "PASS"
        or observation.get("task_id") != "stage1.10_checkpoint_resume_and_artifacts"
        or observation.get("gate_id") != "G1-RESUME"
        or observation.get("execution_commit") != S110_R12_PRODUCER_COMMIT
        or observation.get("single_process_resume") is not True
        or observation.get("four_rank_resume") is not True
        or observation.get("run_owned_resources_released") is not True
        or resume_report.get("formal_observation") != observation
        or resume_report.get("report_hash") != role_declared_hashes["resume_report"]
        or resume_report.get("artifact_manifest_hash") != role_declared_hashes["artifact_manifest"]
        or resume_report.get("oracle_hash") != role_declared_hashes["oracle_bundle"]
        or resume_report.get("trace_hash") != role_declared_hashes["trace_bundle"]
        or resume_report.get("table_hash") != role_declared_hashes["comparison_table"]
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_MUTUAL_HASH_CLOSURE_INVALID")
    if (
        index.get("gate_artifact_hash") != role_declared_hashes["gate_record"]
        or index.get("formal_run_token_sha256") != observation.get("run_token_sha256")
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_GATE_HASH_CLOSURE_INVALID")

    validation = _object(index_dir / "validation.json", field="emit_s110_validation")
    replay = _object(index_dir / "replay-validation.json", field="emit_s110_replay")
    success = _object(index_dir / "success.json", field="emit_s110_success")
    if (
        index.get("validation_ref") != "validation.json"
        or index.get("validation_sha256") != _sha(index_dir / "validation.json")
        or index.get("replay_ref") != "replay-validation.json"
        or index.get("replay_sha256") != _sha(index_dir / "replay-validation.json")
        or validation.get("schema_version") != "stage1-s1-10-validation-v2"
        or validation.get("status") != "PASS"
        or validation.get("task_id") != index.get("task_id")
        or validation.get("gate_id") != index.get("gate_id")
        or validation.get("producer_commit") != S110_R12_PRODUCER_COMMIT
        or validation.get("consumer_commit") != S110_R12_PRODUCER_COMMIT
        or validation.get("role_sha256") != dict(role_hashes)
        or validation.get("replay_sha256") != index.get("replay_sha256")
        or validation.get("replay_hash") != index.get("replay_hash")
        or not isinstance(validation.get("direct_checks"), Mapping)
        or any(value is not True for value in validation["direct_checks"].values())
        or replay.get("schema_version") != "stage1-s1-10-replay-validation-v1"
        or replay.get("status") != "PASS"
        or replay.get("replay_hash") != index.get("replay_hash")
        or replay.get("source_report_hash") != role_declared_hashes["resume_report"]
        or replay.get("source_manifest_hash") != role_declared_hashes["artifact_manifest"]
        or replay.get("source_gate_artifact_hash") != role_declared_hashes["gate_record"]
        or replay.get("source_oracle_hash") != role_declared_hashes["oracle_bundle"]
        or replay.get("source_trace_hash") != role_declared_hashes["trace_bundle"]
        or replay.get("source_table_hash") != role_declared_hashes["comparison_table"]
        or success.get("schema_version") != "stage1-s1-10-attempt-success-v1"
        or success.get("status") != "PASS"
        or success.get("index_sha256") != S110_R12_INDEX_SHA256
        or success.get("validation_sha256") != index.get("validation_sha256")
        or success.get("gate_artifact_hash") != role_declared_hashes["gate_record"]
    ):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_VALIDATION_CLOSURE_INVALID")
    _declared_role_hash(validation, role="validation")
    _declared_role_hash(replay, role="replay")
    _declared_role_hash(success, role="success")

    source_map = resume_report.get("implementation_source_sha256")
    if not isinstance(source_map, Mapping) or set(source_map) != set(S110_R12_SOURCE_PATHS):
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_SOURCE_KEYSET_INVALID")
    for source in S110_R12_SOURCE_PATHS:
        digest = source_map.get(source)
        if not _digest(digest) or _git_blob_sha256(
            repository, S110_R12_PRODUCER_COMMIT, source
        ) != digest:
            raise Stage1S111FormalError(
                f"S1_11_EMIT_S110_R12_SOURCE_HASH_MISMATCH:{source}"
            )
    if resume_report.get("producer_commit") != S110_R12_PRODUCER_COMMIT:
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_PRODUCER_MISMATCH")
    _schema_validate(
        repository,
        {
            "formal_observation": observation,
            "resume_report": resume_report,
            "artifact_manifest": roles["artifact_manifest"],
            "gate_record": roles["gate_record"],
            "oracle_bundle": roles["oracle_bundle"],
            "trace_bundle": roles["trace_bundle"],
            "comparison_table": roles["comparison_table"],
            "replay": replay,
            "validation": validation,
            "index": index,
        },
        producer_commit=S110_R12_PRODUCER_COMMIT,
        frozen_source_sha256=source_map,
        schema_prefix="s1-10",
        validator_source_path="ops/stage1/formalize_s1_10.py",
    )
    source_refs = sorted(
        path.relative_to(evidence_root.resolve()).as_posix()
        for path in index_dir.rglob("*") if path.is_file()
    )
    if S110_R12_INDEX_REF not in source_refs:
        raise Stage1S111FormalError("S1_11_EMIT_S110_R12_SOURCE_REF_CLOSURE_INVALID")
    return {
        "index": index,
        "index_path": index_path,
        "index_dir": index_dir,
        "roles": roles,
        "role_hashes": role_declared_hashes,
        "role_file_sha256": dict(role_hashes),
        "validation": validation,
        "replay": replay,
        "success": success,
        "source_refs": source_refs,
        "source_sha256": dict(source_map),
        "data_root_identity": data_root_identity,
    }


def _task_artifact_hash(
    *, task_id: str, artifact_kind: str, config_hash: str,
    run_intent: str, formal_eligible: bool, source_refs: tuple[str, ...],
    payload: Mapping[str, object],
) -> str:
    return canonical_json_hash({
        "schema_version": "task-output-artifact-v1",
        "task_id": task_id,
        "artifact_kind": artifact_kind,
        "config_hash": config_hash,
        "run_intent": run_intent,
        "formal_eligible": formal_eligible,
        "source_refs": list(source_refs),
        "payload": dict(payload),
    })


def _publish_emit_group(
    *, evidence_root: Path, output_dir: str, task_id: str,
    artifact_kinds: tuple[str, ...], config_hash: str,
    source_refs: tuple[str, ...], payloads: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Resume an individual group without deleting already published objects.

    The group manifest is written before the first commit.  A separate success
    marker is written only after every existing/missing commit has been loaded
    through the full validator.  Thus a crash between any two commits leaves a
    resumable PUBLISHING state rather than a permanent partial block.
    """

    output_path = _emit_relative(evidence_root, output_dir, field="task_output")
    _reject_symlink_ancestors(output_path, evidence_root, field="task_output_parent")
    if output_path.exists():
        _reject_symlink_tree(output_path, field="task_output")
    store = TaskArtifactStore(evidence_root, output_dir)
    _reject_symlink_tree(store.root, field="task_output")
    commit_refs = {
        kind: f"{store.output_dir}/commits/{kind}.json"
        for kind in artifact_kinds
    }
    expected_artifacts = {
        kind: _task_artifact_hash(
            task_id=task_id, artifact_kind=kind, config_hash=config_hash,
            run_intent="formal", formal_eligible=True,
            source_refs=source_refs, payload=payloads[kind],
        )
        for kind in artifact_kinds
    }
    group_body = {
        "schema_version": "stage1-task-artifact-group-manifest-v1",
        "status": "PUBLISHING",
        "task_id": task_id,
        "artifact_kinds": list(artifact_kinds),
        "config_hash": config_hash,
        "source_refs_hash": canonical_json_hash(list(source_refs)),
        "source_refs": list(source_refs),
        "expected_artifact_hashes": expected_artifacts,
        "commit_refs": commit_refs,
    }
    group_manifest = {**group_body, "artifact_hash": canonical_json_hash(group_body)}
    _emit_json(output_path / "group-manifest.json", group_manifest, field="group_manifest")
    success_path = output_path / "success.json"
    if success_path.exists():
        success = _object(success_path, field="group_success")
        success_body = dict(success)
        success_hash = success_body.pop("artifact_hash", None)
        if (
            success_hash != canonical_json_hash(success_body)
            or success.get("schema_version") != "stage1-task-artifact-group-success-v1"
            or success.get("status") != "PASS"
            or success.get("group_manifest_sha256") != _sha(output_path / "group-manifest.json")
            or success.get("commit_refs") != commit_refs
        ):
            raise Stage1S111FormalError("S1_11_EMIT_GROUP_SUCCESS_DRIFT")
    refs: dict[str, str] = {}
    for kind in artifact_kinds:
        commit_path = store.commits / f"{kind}.json"
        if commit_path.is_symlink():
            raise Stage1S111FormalError(f"S1_11_EMIT_{kind.upper()}_COMMIT_SYMLINK")
        if commit_path.exists():
            ref = commit_refs[kind]
            try:
                loaded = load_committed_task_artifact(
                    evidence_root, ref, require_formal=True,
                )
            except Exception as error:
                raise Stage1S111FormalError(
                    f"S1_11_EMIT_EXISTING_COMMIT_INVALID:{kind}"
                ) from error
            if (
                loaded.identity.task_id != task_id
                or loaded.identity.artifact_kind != kind
                or loaded.identity.config_hash != config_hash
                or loaded.identity.artifact_hash != expected_artifacts[kind]
                or loaded.run_intent != "formal"
                or loaded.identity.formal_eligible is not True
                or dict(loaded.payload) != dict(payloads[kind])
                or loaded.source_refs != source_refs
            ):
                raise Stage1S111FormalError(
                    f"S1_11_EMIT_EXISTING_COMMIT_DRIFT:{kind}"
                )
            refs[kind] = ref
            continue
        published = store.publish(
            task_id=task_id,
            artifact_kind=kind,
            config_hash=config_hash,
            run_intent="formal",
            payload=payloads[kind],
            formal_eligible=True,
            source_refs=source_refs,
        )
        refs[kind] = published.commit_ref
        # A crash after publish is safe: the next invocation sees and verifies
        # this commit, then continues with the next missing kind.
    loaded_hashes: dict[str, str] = {}
    for kind in artifact_kinds:
        loaded = load_committed_task_artifact(
            evidence_root, refs[kind], require_formal=True,
        )
        if (
            loaded.identity.artifact_hash != expected_artifacts[kind]
            or dict(loaded.payload) != dict(payloads[kind])
            or loaded.source_refs != source_refs
        ):
            raise Stage1S111FormalError(f"S1_11_EMIT_COMMIT_READBACK_INVALID:{kind}")
        loaded_hashes[kind] = loaded.identity.artifact_hash
    success_body = {
        "schema_version": "stage1-task-artifact-group-success-v1",
        "status": "PASS",
        "task_id": task_id,
        "artifact_kinds": list(artifact_kinds),
        "config_hash": config_hash,
        "group_manifest_sha256": _sha(output_path / "group-manifest.json"),
        "commit_refs": refs,
        "commit_artifact_hashes": loaded_hashes,
    }
    success = {**success_body, "artifact_hash": canonical_json_hash(success_body)}
    _emit_json(success_path, success, field="group_success")
    return {
        "output_dir": store.output_dir,
        "commit_refs": refs,
        "commit_artifact_hashes": loaded_hashes,
        "expected_artifact_hashes": expected_artifacts,
        "group_manifest_ref": f"{store.output_dir}/group-manifest.json",
        "group_manifest_sha256": _sha(output_path / "group-manifest.json"),
        "success_ref": f"{store.output_dir}/success.json",
        "success_sha256": _sha(success_path),
    }


def emit_task_artifacts(
    *, repository: Path, evidence_root: Path, output_dir: str,
    evidence_ref: str = S111_R4_INDEX_REF,
    approved_data_root: Path | str | None = None,
    s110_output_dir: str = S110_COMPAT_OUTPUT_DEFAULT,
) -> dict[str, object]:
    """Publish S1.11 and historical S1.10 formal task commits.

    The only accepted inputs are the fixed r4/r12 authorities.  All payloads
    are copied from those validated roles; no caller JSON can substitute a
    PASS-looking artifact.
    """

    repository = Path(repository).resolve()
    raw_evidence_root = Path(evidence_root)
    data_root_identity = _validate_approved_data_root(
        raw_evidence_root, approved_data_root,
    )
    evidence_root = raw_evidence_root.resolve()
    if not repository.is_dir():
        raise Stage1S111FormalError("S1_11_EMIT_ROOT_INVALID")
    authority = _emit_load_r4(
        repository=repository,
        evidence_root=evidence_root,
        evidence_ref=evidence_ref,
        approved_data_root=approved_data_root,
    )
    s110 = _emit_load_s110(
        repository=repository,
        evidence_root=evidence_root,
        approved_data_root=approved_data_root,
    )
    s111_output = _emit_relative(evidence_root, output_dir, field="task_output")
    s110_output = _emit_relative(evidence_root, s110_output_dir, field="s110_task_output")
    if s111_output == s110_output:
        raise Stage1S111FormalError("S1_11_EMIT_TASK_OUTPUTS_MUST_BE_DISTINCT")
    for path, field in ((s111_output, "task_output"), (s110_output, "s110_task_output")):
        _reject_symlink_ancestors(path, evidence_root, field=f"{field}_parent")
        if path.exists():
            _reject_symlink_tree(path, field=field)

    s110_config_ref = f"{s110_output.relative_to(evidence_root).as_posix()}/producer-config.json"
    s110_commit_refs = {
        kind: f"{s110_output.relative_to(evidence_root).as_posix()}/commits/{kind}.json"
        for kind in S110_TASK_ARTIFACT_KINDS
    }
    s110_source_refs = tuple(sorted({
        *[str(value) for value in s110["source_refs"]],
        s110_config_ref,
        *s110_commit_refs.values(),
    }))
    s110_config_body: dict[str, object] = {
        "schema_version": "stage1-s1-10-compat-producer-config-v1",
        "config_kind": "s1_10_checkpoint_resume_compatibility_producer",
        "task_id": "stage1.10_checkpoint_resume_and_artifacts",
        "gate_id": "G1-RESUME",
        "run_intent": "formal",
        "formal_eligible": True,
        "authority": {
            "index_ref": S110_R12_INDEX_REF,
            "index_sha256": S110_R12_INDEX_SHA256,
            "index_artifact_hash": S110_R12_INDEX_ARTIFACT_HASH,
            "attempt_id": S110_R12_ATTEMPT_ID,
            "producer_commit": S110_R12_PRODUCER_COMMIT,
            "execution_commit": S110_R12_PRODUCER_COMMIT,
            "consumer_commit": S110_R12_PRODUCER_COMMIT,
        },
        "data_root": data_root_identity,
        "producer_source_sha256": s110["source_sha256"],
        "producer_source_hash": canonical_json_hash(s110["source_sha256"]),
        "role_sha256": s110["role_file_sha256"],
        "role_artifact_hashes": s110["role_hashes"],
        "r12_artifact_hashes": {
            "index": S110_R12_INDEX_ARTIFACT_HASH,
            "validation": _sha(s110["index_dir"] / "validation.json"),
            "replay_validation": _sha(s110["index_dir"] / "replay-validation.json"),
            "success": _sha(s110["index_dir"] / "success.json"),
            **s110["role_hashes"],
        },
        "source_refs": list(s110_source_refs),
        "artifact_kinds": list(S110_TASK_ARTIFACT_KINDS),
        "commit_refs": s110_commit_refs,
    }
    s110_config_hash = canonical_json_hash(s110_config_body)
    s110_config = {**s110_config_body, "config_hash": s110_config_hash}
    _emit_json(s110_output / "producer-config.json", s110_config, field="s110_producer_config")
    s110_config_sha256 = _sha(s110_output / "producer-config.json")
    s110_payloads = {
        kind: dict(s110["roles"][role])
        for kind, role in S110_TASK_ROLE_MAP.items()
    }
    s110_group = _publish_emit_group(
        evidence_root=evidence_root,
        output_dir=s110_output_dir,
        task_id="stage1.10_checkpoint_resume_and_artifacts",
        artifact_kinds=S110_TASK_ARTIFACT_KINDS,
        config_hash=s110_config_hash,
        source_refs=s110_source_refs,
        payloads=s110_payloads,
    )

    s111_config_ref = f"{s111_output.relative_to(evidence_root).as_posix()}/producer-config.json"
    s111_commit_refs = {
        kind: f"{s111_output.relative_to(evidence_root).as_posix()}/commits/{kind}.json"
        for kind in S111_TASK_ARTIFACT_KINDS
    }
    s111_source_refs = tuple(sorted({
        *[str(value) for value in authority["source_refs"]],
        *s110_source_refs,
        s111_config_ref,
        *s111_commit_refs.values(),
    }))
    s111_config_body: dict[str, object] = {
        "schema_version": "stage1-s1-11-producer-config-v2",
        "config_kind": "s1_11_reporting_producer",
        "task_id": TASK_ID,
        "gate_id": GATE_ID,
        "run_intent": "formal",
        "formal_eligible": True,
        "data_root": data_root_identity,
        "authority": {
            "index_ref": evidence_ref,
            "index_sha256": S111_R4_INDEX_SHA256,
            "index_artifact_hash": S111_R4_INDEX_ARTIFACT_HASH,
            "attempt_id": S111_R4_ATTEMPT_ID,
            "producer_commit": S111_R4_PRODUCER_COMMIT,
            "execution_commit": S111_R4_PRODUCER_COMMIT,
            "consumer_commit": S111_R4_PRODUCER_COMMIT,
        },
        "producer_source_sha256": authority["producer_source_sha256"],
        "producer_source_hash": canonical_json_hash(authority["producer_source_sha256"]),
        "r4_artifact_hashes": authority["artifact_hashes"],
        "r4_role_sha256": authority["role_file_sha256"],
        "dependency_index_sha256": authority["dependency_index_sha256"],
        "s110_compatibility": {
            "task_id": "stage1.10_checkpoint_resume_and_artifacts",
            "output_dir": s110_output.relative_to(evidence_root).as_posix(),
            "config_ref": s110_config_ref,
            "config_hash": s110_config_hash,
            "config_sha256": s110_config_sha256,
            "authority_index_ref": S110_R12_INDEX_REF,
            "authority_index_sha256": S110_R12_INDEX_SHA256,
            "authority_index_artifact_hash": S110_R12_INDEX_ARTIFACT_HASH,
            "source_hash": canonical_json_hash(s110["source_sha256"]),
            "source_sha256": s110["source_sha256"],
            "commit_refs": s110_group["commit_refs"],
            "commit_artifact_hashes": s110_group["commit_artifact_hashes"],
        },
        "source_refs": list(s111_source_refs),
        "artifact_kinds": list(S111_TASK_ARTIFACT_KINDS),
        "commit_refs": s111_commit_refs,
    }
    s111_config_hash = canonical_json_hash(s111_config_body)
    s111_config = {**s111_config_body, "config_hash": s111_config_hash}
    _emit_json(s111_output / "producer-config.json", s111_config, field="producer_config")
    s111_payloads = {
        kind: dict(authority["roles"][kind])
        for kind in S111_TASK_ARTIFACT_KINDS
    }
    s111_group = _publish_emit_group(
        evidence_root=evidence_root,
        output_dir=output_dir,
        task_id=TASK_ID,
        artifact_kinds=S111_TASK_ARTIFACT_KINDS,
        config_hash=s111_config_hash,
        source_refs=s111_source_refs,
        payloads=s111_payloads,
    )
    manifest_body: dict[str, object] = {
        "schema_version": "stage1-s1-11-task-artifact-manifest-v2",
        "status": "PASS",
        "task_id": TASK_ID,
        "gate_id": GATE_ID,
        "run_intent": "formal",
        "formal_eligible": True,
        "data_root": data_root_identity,
        "commit_refs": s111_group["commit_refs"],
        "commit_artifact_hashes": s111_group["commit_artifact_hashes"],
        "config_ref": s111_config_ref,
        "config_hash": s111_config_hash,
        "config_sha256": _sha(s111_output / "producer-config.json"),
        "group_manifest_ref": s111_group["group_manifest_ref"],
        "group_manifest_sha256": s111_group["group_manifest_sha256"],
        "success_ref": s111_group["success_ref"],
        "success_sha256": s111_group["success_sha256"],
        "r4_index_ref": evidence_ref,
        "r4_index_sha256": S111_R4_INDEX_SHA256,
        "r4_index_artifact_hash": S111_R4_INDEX_ARTIFACT_HASH,
        "r4_artifact_hashes": authority["artifact_hashes"],
        "r4_role_sha256": authority["role_file_sha256"],
        "producer_commit": S111_R4_PRODUCER_COMMIT,
        "execution_commit": S111_R4_PRODUCER_COMMIT,
        "consumer_commit": S111_R4_PRODUCER_COMMIT,
        "producer_source_sha256": authority["producer_source_sha256"],
        "producer_source_hash": canonical_json_hash(authority["producer_source_sha256"]),
        "s110_compatibility": s111_config_body["s110_compatibility"],
    }
    manifest = {**manifest_body, "artifact_hash": canonical_json_hash(manifest_body)}
    _emit_json(s111_output / "manifest.json", manifest, field="manifest")
    return {
        "manifest_ref": f"{s111_output.relative_to(evidence_root).as_posix()}/manifest.json",
        "manifest_sha256": _sha(s111_output / "manifest.json"),
        "config_ref": s111_config_ref,
        "config_hash": s111_config_hash,
        "config_sha256": _sha(s111_output / "producer-config.json"),
        "commit_refs": dict(s111_group["commit_refs"]),
        "r4_index_ref": evidence_ref,
        "r4_index_sha256": S111_R4_INDEX_SHA256,
        "r4_index_artifact_hash": S111_R4_INDEX_ARTIFACT_HASH,
        "r4_artifact_hashes": authority["artifact_hashes"],
        "s110_compatibility": {
            "config_ref": s110_config_ref,
            "config_hash": s110_config_hash,
            "config_sha256": s110_config_sha256,
            "commit_refs": dict(s110_group["commit_refs"]),
            "index_ref": S110_R12_INDEX_REF,
            "index_sha256": S110_R12_INDEX_SHA256,
            "index_artifact_hash": S110_R12_INDEX_ARTIFACT_HASH,
        },
    }


def execute(*, repository: Path, evidence_root: Path, attempt_root: Path, dependencies_path: Path, matrix_path: Path, chart_specs_path: Path, test_summary_binding_path: Path, sync_audit_binding_path: Path, failure_history_path: Path, execution_commit: str, attempt_id: str, worklog_binding_path: Path | None = None) -> dict[str, str]:
    if repository.resolve() != REPOSITORY.resolve() or not evidence_root.is_dir() or not attempt_id or "/" in attempt_id or "\\" in attempt_id or len(execution_commit) != 40 or any(character not in "0123456789abcdef" for character in execution_commit):
        raise Stage1S111FormalError("S1_11_ARGUMENTS_INVALID")
    current_commit = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    if current_commit != execution_commit:
        raise Stage1S111FormalError("S1_11_EXECUTION_COMMIT_MISMATCH")
    dependencies = _load_config(dependencies_path, field="dependencies")
    matrix = _load_config(matrix_path, field="matrix")
    chart_specs = _load_config(chart_specs_path, field="charts")
    test_summary = _validate_test_summary(evidence_root, _load_config(test_summary_binding_path, field="test_binding"))
    sync_audit = _validate_sync_audit(evidence_root, _load_config(sync_audit_binding_path, field="sync_binding"))
    if sync_audit["execution_commit"] != execution_commit:
        raise Stage1S111FormalError("S1_11_SYNC_AUDIT_EXECUTION_COMMIT_MISMATCH")
    failures = _validate_failure_history(evidence_root, _load_config(failure_history_path, field="failure_history"))
    if worklog_binding_path is None:
        raise Stage1S111FormalError("S1_11_WORKLOG_BINDING_REQUIRED")
    worklog = _validate_bound_file(evidence_root, _load_config(worklog_binding_path, field="worklog_binding"), field="worklog")
    if not isinstance(dependencies, list) or not isinstance(matrix, list):
        raise Stage1S111FormalError("S1_11_INPUT_LIST_INVALID")
    target = evidence_root / "evidence" / "stage1" / "s1-11-formal" / execution_commit / attempt_id
    work = attempt_root / "stage1-s1-11" / attempt_id
    if target.exists() or work.exists():
        raise Stage1S111FormalError("S1_11_ATTEMPT_TARGET_EXISTS")
    work.mkdir(parents=True)
    try:
        charts = _build_charts(evidence_root, work, dependencies, chart_specs)
        # Build the formal observation only after strict dependency/matrix/chart
        # checks.  It is derived here, never accepted from command-line input.
        from param_importance_nlp.stage1_exit_gate import audit_exit_dependencies, _validate_verification_matrix
        audits = audit_exit_dependencies(evidence_root, dependencies)
        matrix_rows = _validate_verification_matrix(evidence_root, matrix)
        immutable_indexes = {row["binding"]["gate_id"]: row["binding"] for row in audits}
        for row in matrix_rows:
            binding = immutable_indexes.get(row["gate_id"])
            evidence = row["evidence"]
            if binding is None or evidence["ref"] != binding["index_ref"] or evidence["sha256"] != binding["index_sha256"]:
                raise Stage1S111FormalError("S1_11_VERIFICATION_MATRIX_NOT_BOUND_TO_IMMUTABLE_INDEX")
        observation = {"schema_version": "stage1-s1-11-formal-observation-v1", "status": "PASS", "task_id": TASK_ID, "gate_id": GATE_ID, "execution_commit": execution_commit, "dependency_index_sha256": {row["binding"]["gate_id"]: row["binding"]["index_sha256"] for row in audits}, "test_summary": test_summary, "sync_audit": {key: sync_audit[key] for key in ("ref", "sha256", "artifact_hash", "execution_commit")}, "failure_history": failures, "charts": charts}
        observation["artifact_hash"] = canonical_json_hash(observation); _write(work / "formal-observation.json", observation)
        test_source = _relative(evidence_root, test_summary["ref"], field="test_summary")
        shutil.copy2(test_source, work / "test-summary.json")
        if _sha(work / "test-summary.json") != test_summary["sha256"]:
            raise Stage1S111FormalError("S1_11_TEST_SUMMARY_COPY_READBACK_MISMATCH")
        sync_source = _relative(evidence_root, sync_audit["ref"], field="sync_audit")
        manifest_source = _relative(sync_source.parent, sync_audit["large_artifact_manifest_ref"], field="large_artifact_manifest")
        shutil.copy2(manifest_source, work / "large-artifact-manifest.json")
        if _sha(work / "large-artifact-manifest.json") != sync_audit["large_artifact_manifest_sha256"]:
            raise Stage1S111FormalError("S1_11_LARGE_ARTIFACT_MANIFEST_COPY_READBACK_MISMATCH")
        requirements = {"schema_version": "stage1-s1-11-requirements-matrix-v1", "task_id": TASK_ID, "gate_id": GATE_ID, "rows": matrix_rows}; requirements["artifact_hash"] = canonical_json_hash(requirements); _write(work / "requirements-matrix.json", requirements)
        top20 = _top_errors(evidence_root, dependencies)
        top20_body = {"schema_version": "stage1-s1-11-top-errors-v1", "rows": top20}; top20_body["artifact_hash"] = canonical_json_hash(top20_body); _write(work / "top-errors.json", top20_body)
        summary = {"schema_version": "stage1-s1-11-gate-summary-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "exit_verdict": "STAGE1_COMPLETE", "dependency_audits": audits, "unresolved_failure_count": 0, "charts": charts, "formal_observation": observation}; summary["artifact_hash"] = canonical_json_hash(summary); _write(work / "gate-summary.json", summary)
        context = _derived_report_context(evidence_root, audits, sync_audit, worklog)
        report = {"schema_version": "stage1-s1-11-stage-report-v1", "status": "PASS", "task_id": TASK_ID, "gate_id": GATE_ID, "summary_hash": summary["artifact_hash"], "requirements_matrix_hash": requirements["artifact_hash"], "scope_statement": "Stage 1 validates implementation correctness only; it makes no scientific conclusion.", "upstream_context": context}; report["artifact_hash"] = canonical_json_hash(report); _write(work / "stage-report.json", report)
        dependency_lines = "\n".join(f"- `{row['binding']['gate_id']}`: index SHA-256 `{row['binding']['index_sha256']}`; gate artifact `{row['binding']['gate_artifact_hash']}`" for row in audits)
        matrix_lines = "\n".join(f"- `{row['requirement_id']}` — measured: `{row['measured']}`; threshold: {row['threshold']}; evidence: `{row['evidence']['ref']}`" for row in matrix_rows)
        error_lines = "\n".join(f"| {row['gate_id']} | {row['parameter']} | {row['coordinate']} | {row['max_abs_error']:.12g} |" for row in top20)
        chart_lines = "\n".join(f"| {row['chart_id']} | {row['row_count']} | `{row['source_csv_sha256']}` | `{row['csv_sha256']}` | `{row['svg_sha256']}` |" for row in charts)
        failure_lines = "\n".join(f"- failure `{row['failure_ref']}` (SHA-256 `{row['failure_sha256']}`) → resolution `{row['resolution_ref']}` (SHA-256 `{row['resolution_sha256']}`)" for row in failures) or "- No historical failures were recorded."
        artifact_lines = "\n".join(f"- `{row['ref']}` — {row['size_bytes']} bytes; SHA-256 `{row['sha256']}`" for row in context["large_artifacts"])
        markdown = f"# Stage 1 report\n\nStatus: PASS\n\n## Scope and residual risk\n\nStage 1 validates implementation correctness only; it makes no scientific conclusion. The local gradient-space score is not presented as a scientific estimate of parameter importance. Stage 2 and Stage 3 remain out of scope until this immutable delivery is accepted.\n\n## Measured implementation and platform boundaries\n\nAdamW evidence: `{context['adamw_boundary']['step_report_ref']}` (SHA-256 `{context['adamw_boundary']['step_report_sha256']}`) confirms the data/decay/total decomposition and that actual-update importance remains a separate diagnostic from gradient-space `u`. GPU health: `{context['gpu_health']['preflight_ref']}` (SHA-256 `{context['gpu_health']['preflight_sha256']}`); approved UUIDs: `{', '.join(context['gpu_health']['approved_gpu_uuids'])}`.\n\nProvenance is derived from frozen roles: config `{context['provenance']['config_ref']}` (SHA-256 `{context['provenance']['config_sha256']}`), environment `{context['provenance']['environment_ref']}` (SHA-256 `{context['provenance']['environment_sha256']}`), result `{context['provenance']['result_ref']}` (SHA-256 `{context['provenance']['result_sha256']}`), registry `{context['provenance']['registry_report_ref']}` (SHA-256 `{context['provenance']['registry_report_sha256']}`), model asset `{context['provenance']['model_asset_id']}`, pile asset `{context['provenance']['pile_asset_id']}`, and parameter registry `{context['provenance']['parameter_registry_hash']}`.\n\n## Structured test summary\n\nFrozen test-summary artifact: `{test_summary['ref']}` (SHA-256 `{test_summary['sha256']}`).\n\n## Immutable gate closure\n\n{dependency_lines}\n\n## Final verification matrix\n\n{matrix_lines}\n\n## Worst 20 parameter-coordinate errors\n\n| Gate | Parameter | Coordinate | Maximum absolute error |\n|---|---|---:|---:|\n{error_lines}\n\n## Generated chart provenance\n\n| Chart | rows | source CSV SHA-256 | derived CSV SHA-256 | SVG SHA-256 |\n|---|---:|---|---|---|\n{chart_lines}\n\n## Failure, artifact, and synchronization closure\n\nHistorical failure records:\n{failure_lines}\n\nLarge artifacts:\n{artifact_lines}\n\nSync audit: `{sync_audit['ref']}` (SHA-256 `{sync_audit['sha256']}`), including the exact six Agent documents. Worklog: `{worklog['ref']}` (SHA-256 `{worklog['sha256']}`). Local validation validates content; Git publication records the frozen commit and clean state; server execution records the evidence-root execution and clean state. No endpoint or credential is reproduced here.\n\n## Next-stage decision\n\nAll required checks are PASS. Stage 2 and Stage 3 are unlocked only for the frozen identities recorded in this delivery manifest.\n"
        (work / "stage-report.md").write_text(markdown, encoding="utf-8", newline="\n")
        delivery = {"schema_version": "stage1-s1-11-delivery-manifest-v1", "task_id": TASK_ID, "gate_id": GATE_ID, "summary_hash": summary["artifact_hash"], "requirements_matrix_hash": requirements["artifact_hash"], "stage_report_hash": report["artifact_hash"], "dependency_index_hashes": {row["binding"]["gate_id"]: row["binding"]["index_sha256"] for row in audits}, "chart_ids": list(REQUIRED_CHARTS)}; delivery["artifact_hash"] = canonical_json_hash(delivery); _write(work / "delivery-manifest.json", delivery)
        replayed_audits = audit_exit_dependencies(evidence_root, dependencies)
        if replayed_audits != audits or any(_sha(work / str(item["csv_ref"])) != item["csv_sha256"] or _sha(work / str(item["svg_ref"])) != item["svg_sha256"] for item in charts):
            raise Stage1S111FormalError("S1_11_OFFLINE_REPLAY_MISMATCH")
        replay = {"schema_version": "stage1-s1-11-replay-validation-v1", "status": "PASS", "summary_artifact_hash": summary["artifact_hash"]}; replay["artifact_hash"] = canonical_json_hash(replay); _write(work / "replay-validation.json", replay)
        validation = {"schema_version": "stage1-s1-11-validation-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "role_sha256": {name: _sha(work / name) for name in ("formal-observation.json", "test-summary.json", "requirements-matrix.json", "top-errors.json", "gate-summary.json", "stage-report.json", "delivery-manifest.json", "replay-validation.json", "stage-report.md", *[f"{name}.{suffix}" for name in REQUIRED_CHARTS for suffix in ("csv", "svg")])}}; validation["artifact_hash"] = canonical_json_hash(validation); _write(work / "validation.json", validation)
        role_refs = {
            "formal_observation": "formal-observation.json", "test_summary": "test-summary.json",
            "requirements_matrix": "requirements-matrix.json", "top_errors": "top-errors.json",
            "gate_summary": "gate-summary.json", "stage_report": "stage-report.json",
            "stage_report_markdown": "stage-report.md", "delivery_manifest": "delivery-manifest.json",
            "replay_validation": "replay-validation.json",
            **{f"{chart_id.replace('-', '_')}_csv": f"{chart_id}.csv" for chart_id in REQUIRED_CHARTS},
            **{f"{chart_id.replace('-', '_')}_svg": f"{chart_id}.svg" for chart_id in REQUIRED_CHARTS},
        }
        role_sha = {role: _sha(work / filename) for role, filename in role_refs.items()}
        index = {"schema_version": "stage1-s1-11-formalization-index-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "generator_git_commit": execution_commit, "consumer_git_commit": execution_commit, "implementation_source_sha256": {path: _sha(repository / path) for path in _IMPLEMENTATION_PATHS}, "role_refs": role_refs, "role_sha256": role_sha, "chart_csv_sha256": {f"{chart_id}.csv": _sha(work / f"{chart_id}.csv") for chart_id in REQUIRED_CHARTS}, "chart_svg_sha256": {f"{chart_id}.svg": _sha(work / f"{chart_id}.svg") for chart_id in REQUIRED_CHARTS}, "reproduction_role_refs": {"test_summary": str(test_summary["ref"]), "sync_audit": str(sync_audit["ref"]), "large_artifact_manifest": str(sync_audit["large_artifact_manifest_ref"]), "worklog": worklog["ref"]}, "reproduction_role_sha256": {"test_summary": str(test_summary["sha256"]), "sync_audit": str(sync_audit["sha256"]), "large_artifact_manifest": str(sync_audit["large_artifact_manifest_sha256"]), "worklog": worklog["sha256"]}, "validation_ref": "validation.json", "validation_sha256": _sha(work / "validation.json"), "replay_ref": "replay-validation.json", "replay_sha256": _sha(work / "replay-validation.json"), "next_task_ids": ["stage2", "stage3"], "artifact_hash": ""}; index["artifact_hash"] = canonical_json_hash({key: value for key, value in index.items() if key != "artifact_hash"}); _exact_s111_index_closure(repository, work, index); _write(work / "index.json", index)
        _schema_validate(repository, {"formal_observation": observation, "requirements_matrix": requirements, "gate_summary": summary, "stage_report": report, "delivery_manifest": delivery, "replay": replay, "validation": validation, "index": index})
        staging = target.parent / f".{attempt_id}.publishing"
        if staging.exists():
            raise Stage1S111FormalError("S1_11_STAGING_EXISTS")
        staging.mkdir(parents=True)
        for source in work.iterdir():
            if source.is_file(): shutil.copy2(source, staging / source.name)
        # Immutable readback before success marker and atomic publication.
        for name, digest in validation["role_sha256"].items():
            if _sha(staging / name) != digest:
                raise Stage1S111FormalError("S1_11_PUBLISH_READBACK_HASH_MISMATCH")
        if any(_sha(staging / filename) != index["role_sha256"][role] for role, filename in index["role_refs"].items()) or any(_sha(staging / name) != digest for name, digest in {**index["chart_csv_sha256"], **index["chart_svg_sha256"]}.items()):
            raise Stage1S111FormalError("S1_11_INDEX_ROLE_READBACK_MISMATCH")
        reloaded = {"formal_observation": _object(staging / "formal-observation.json", field="formal_observation"), "requirements_matrix": _object(staging / "requirements-matrix.json", field="requirements_matrix"), "gate_summary": _object(staging / "gate-summary.json", field="gate_summary"), "stage_report": _object(staging / "stage-report.json", field="stage_report"), "delivery_manifest": _object(staging / "delivery-manifest.json", field="delivery_manifest"), "replay": _object(staging / "replay-validation.json", field="replay"), "validation": _object(staging / "validation.json", field="validation"), "index": _object(staging / "index.json", field="index")}
        _schema_validate(repository, reloaded)
        _exact_s111_index_closure(repository, staging, reloaded["index"])
        success = {"schema_version": "stage1-s1-11-attempt-success-v1", "status": "PASS", "index_sha256": _sha(staging / "index.json")}; success["artifact_hash"] = canonical_json_hash(success); _write(staging / "success.json", success)
        os.replace(staging, target)
        return {"index_ref": (target / "index.json").relative_to(evidence_root).as_posix(), "index_sha256": _sha(target / "index.json")}
    except BaseException as error:
        if not (work / "failure.json").exists():
            failure = {"schema_version": "stage1-s1-11-attempt-failure-v1", "status": "FAILED", "error_type": type(error).__name__, "error": str(error)}; failure["artifact_hash"] = canonical_json_hash(failure); _write(work / "failure.json", failure)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--emit-task-artifacts", action="store_true",
        help="publish the four formal S1.11 TaskArtifactStore commits from r4",
    )
    parser.add_argument(
        "--workspace-root", type=Path,
        help="DATA_ROOT/workspace containing the r4 evidence and task store",
    )
    parser.add_argument(
        "--evidence-ref", default=S111_R4_INDEX_REF,
        help="fixed r4 index ref; emit mode rejects any other value",
    )
    parser.add_argument(
        "--task-output-dir", default=S111_TASK_OUTPUT_DEFAULT,
        help="POSIX-relative TaskArtifactStore output directory",
    )
    parser.add_argument(
        "--s110-task-output-dir", default=S110_COMPAT_OUTPUT_DEFAULT,
        help="POSIX-relative compatibility TaskArtifactStore output for S1.10",
    )
    parser.add_argument(
        "--approved-data-root", type=Path, default=None,
        help="explicit approved DATA_ROOT identity (defaults to released server root)",
    )
    # Execute-mode inputs remain available, but are intentionally optional at
    # argparse time so emit mode is not forced to supply unrelated JSON files.
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--attempt-root", type=Path, default=None)
    parser.add_argument("--dependencies-json", type=Path, default=None)
    parser.add_argument("--verification-matrix-json", type=Path, default=None)
    parser.add_argument("--chart-specs-json", type=Path, default=None)
    parser.add_argument("--test-summary-binding-json", type=Path, default=None)
    parser.add_argument("--sync-audit-binding-json", type=Path, default=None)
    parser.add_argument("--failure-history-json", type=Path, default=None)
    parser.add_argument("--worklog-binding-json", type=Path, default=None)
    parser.add_argument("--execution-commit", default=None)
    parser.add_argument("--attempt-id", default=None)
    args = parser.parse_args(argv)
    if args.emit_task_artifacts:
        workspace_root = args.workspace_root or args.evidence_root
        if workspace_root is None:
            parser.error("--emit-task-artifacts requires --workspace-root (DATA_ROOT)")
        print(emit_task_artifacts(
            repository=args.repository,
            evidence_root=workspace_root,
            output_dir=args.task_output_dir,
            evidence_ref=args.evidence_ref,
            approved_data_root=args.approved_data_root,
            s110_output_dir=args.s110_task_output_dir,
        ))
        return 0
    required_execute = (
        "evidence_root", "attempt_root", "dependencies_json",
        "verification_matrix_json", "chart_specs_json",
        "test_summary_binding_json", "sync_audit_binding_json",
        "failure_history_json", "worklog_binding_json", "execution_commit",
        "attempt_id",
    )
    missing = [name for name in required_execute if getattr(args, name) is None]
    if missing:
        parser.error("execute mode missing: " + ", ".join("--" + name.replace("_", "-") for name in missing))
    print(execute(repository=args.repository, evidence_root=args.evidence_root, attempt_root=args.attempt_root, dependencies_path=args.dependencies_json, matrix_path=args.verification_matrix_json, chart_specs_path=args.chart_specs_json, test_summary_binding_path=args.test_summary_binding_json, sync_audit_binding_path=args.sync_audit_binding_json, failure_history_path=args.failure_history_json, worklog_binding_path=args.worklog_binding_json, execution_commit=args.execution_commit, attempt_id=args.attempt_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
