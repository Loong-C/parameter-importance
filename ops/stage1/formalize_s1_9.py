"""Fail-closed formal publisher for S1.9 G1-NUMERIC.

The caller must name every approved GPU UUID and pin both the capability and
S1.8 handoffs.  This command rechecks those pins, current health/occupancy,
assets and upstream evidence *before* acquiring its own ProjectGpuLease.  It
does not accept a pre-existing lease from a wrapper.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


TASK_ID = "stage1.09_precision_clipping_and_optimizer_boundaries"
GATE_ID = "G1-NUMERIC"
FIXTURE_ID = "stage1-s19-precision-fixture-v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
# NVIDIA reports canonical GPU UUIDs in lower-case, fixed-width form.  Do not
# accept a permissive prefix here: this value is both the resource-lease target
# and the CUDA_VISIBLE_DEVICES binding used by the isolated workers.
_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_ATTEMPT = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_CHECKPOINT_STORE_REPRODUCTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DETERMINISM_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
_S1_7_AUTHORIZED_SHARED_DEPENDENCIES = {"src/param_importance_nlp/runtime/optimizer.py"}
_S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS = {
    "A": {"artifact_ref": "run__route-A-identity-formal__route-output__route-A.safetensors", "manifest_ref": "run__route-A-identity-formal__route-output__route-report.json"},
    "B": {"artifact_ref": "run__route-B-identity-formal__route-output__route-B.safetensors", "manifest_ref": "run__route-B-identity-formal__route-output__route-report.json"},
    "C": {"artifact_ref": "run__route-C-identity-formal__route-output__route-C.safetensors", "manifest_ref": "run__route-C-identity-formal__route-output__route-report.json"},
    "D": {"artifact_ref": "run__route-D-identity-formal__route-output__route-D.safetensors", "manifest_ref": "run__route-D-identity-formal__route-output__route-report.json"},
    "D-rank_swap": {"artifact_ref": "run__route-D-rank_swap-formal__route-output__route-D.safetensors", "manifest_ref": "run__route-D-rank_swap-formal__route-output__route-report.json"},
    "D-local_reverse": {"artifact_ref": "run__route-D-local_reverse-formal__route-output__route-D.safetensors", "manifest_ref": "run__route-D-local_reverse-formal__route-output__route-report.json"},
}
# S1.10 was implemented after the immutable S1.8 v3 producer.  These are the
# complete, reviewed files for that one downstream stage; do not turn this
# into a directory/prefix exemption.  In particular, an extra sibling under
# ``schemas/stage1`` or ``ops/stage1`` must still stop upstream reuse.
_S1_10_FROZEN_CONSUMER_FILES = {
    "fixtures/stage1/stage1-s110-checkpoint-fixture-v1.json",
    "ops/stage1/formalize_s1_10.py",
    "ops/stage1/run_s1_10_resume_worker.py",
    "schemas/stage1/s1-10-artifact-manifest-v1.json",
    "schemas/stage1/s1-10-checkpoint-fixture-v1.json",
    "schemas/stage1/s1-10-comparison-table-v1.json",
    "schemas/stage1/s1-10-formal-observation-v1.json",
    "schemas/stage1/s1-10-formalization-index-v1.json",
    "schemas/stage1/s1-10-gate-record-v1.json",
    "schemas/stage1/s1-10-oracle-bundle-v1.json",
    "schemas/stage1/s1-10-replay-validation-v1.json",
    "schemas/stage1/s1-10-resume-report-v1.json",
    "schemas/stage1/s1-10-trace-bundle-v1.json",
    "schemas/stage1/s1-10-validation-v1.json",
    "src/param_importance_nlp/stage1_checkpoint_oracle.py",
    "src/param_importance_nlp/stage1_checkpoint_resume.py",
    "tests/test_runtime_core.py",
    "tests/test_stage1_s110_checkpoint_resume.py",
}
# They are deliberately separate from the frozen consumer-only list.  The
# source-map/exclusion and CPU replay checks below are mandatory when either
# implementation changes, so a later S1.10 runtime edit cannot ride this
# compatibility bridge without being re-examined.
_S1_10_SHARED_RUNTIME_FILES = {
    "src/param_importance_nlp/runtime/training.py",
    "src/param_importance_nlp/runtime/checkpoint_group.py",
}
# S1.9's own formalizer/worker/fixture surface is also explicit.  A producer
# handoff is never permission to accept an arbitrary file merely because its
# name begins with ``s1-9`` (or, worse, ``s1-8``).  Update this set together
# with a reviewed consumer compatibility replay and its negative controls.
_S1_9_FROZEN_CONSUMER_FILES = {
    "fixtures/stage1/stage1-s19-precision-fixture-v1.json",
    "ops/stage1/formalize_s1_9.py",
    "ops/stage1/run_s1_9_ddp_skip_worker.py",
    "ops/stage1/run_s1_9_single_bf16_worker.py",
    "schemas/stage1/s1-9-bf16-checkpoint-store-reproduction-v1.json",
    "schemas/stage1/s1-9-comparison-table-v1.json",
    "schemas/stage1/s1-9-ddp-skip-worker-v1.json",
    "schemas/stage1/s1-9-formalization-index-v1.json",
    "schemas/stage1/s1-9-formalization-index-v6.json",
    "schemas/stage1/s1-9-formalization-index-v7.json",
    "schemas/stage1/s1-9-formalization-index-v8.json",
    "schemas/stage1/s1-9-gate-record-v1.json",
    "schemas/stage1/s1-9-gpu-prelease-v3.json",
    "schemas/stage1/s1-9-gpu-quiescence-v3.json",
    "schemas/stage1/s1-9-numeric-report-v1.json",
    "schemas/stage1/s1-9-oracle-bundle-v1.json",
    "schemas/stage1/s1-9-precision-fixture-v1.json",
    "schemas/stage1/s1-9-replay-validation-v1.json",
    "schemas/stage1/s1-9-single-bf16-worker-v1.json",
    "schemas/stage1/s1-9-trace-bundle-v1.json",
    "schemas/stage1/s1-9-upstream-compatibility-v5.json",
    "schemas/stage1/s1-9-upstream-compatibility-v6.json",
    "schemas/stage1/s1-9-upstream-compatibility-v7.json",
    "schemas/stage1/s1-9-validation-v1.json",
    "src/param_importance_nlp/stage1_precision.py",
    "src/param_importance_nlp/stage1_precision_oracle.py",
    "tests/test_stage1_s19_precision.py",
}
_ENVIRONMENT_SUMMARY_KEYS = {
    "torch_version", "cuda_runtime_version", "cudnn_version", "nccl_version",
    "deterministic_algorithms", "cudnn_benchmark", "cudnn_deterministic",
    "cublas_workspace_config", "pythonhashseed", "cuda_visible_devices",
    "local_rank", "local_gpu_uuid",
}


class Stage1S19FormalError(RuntimeError):
    pass


class Stage1S19ManualInterventionRequired(Stage1S19FormalError):
    """A child identity changed; its lease must remain for human inspection."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: object) -> str:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash

    return canonical_json_hash(value)


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from param_importance_nlp.contracts.jsonio import write_canonical_json

    if path.exists():
        raise Stage1S19FormalError(f"S1_9_IMMUTABLE_OUTPUT_EXISTS:{path.name}")
    write_canonical_json(path, dict(value))


def _with_hash(value: Mapping[str, Any], *, field: str = "artifact_hash") -> dict[str, Any]:
    if field in value:
        raise Stage1S19FormalError(f"S1_9_HASH_FIELD_ALREADY_PRESENT:{field}")
    result = dict(value)
    result[field] = _canonical(result)
    return result


def _self_hash_valid(value: Mapping[str, Any], *, field: str = "artifact_hash") -> bool:
    reported = value.get(field)
    if not isinstance(reported, str) or re.fullmatch(r"[0-9a-f]{64}", reported) is None:
        return False
    body = dict(value)
    body.pop(field, None)
    return reported == _canonical(body)


def _resource_budget(work: Path) -> dict[str, int]:
    """Freeze the minimum and actual local staging budget before a GPU run."""

    usage = shutil.disk_usage(work)
    minimum_free = 2 * 1024**3
    if usage.free < minimum_free:
        raise Stage1S19FormalError("S1_9_STAGING_DISK_BUDGET_INSUFFICIENT")
    return {"minimum_free_bytes": minimum_free, "actual_free_bytes": int(usage.free), "actual_total_bytes": int(usage.total), "estimated_output_bytes": 256 * 1024**2}


def _release_lease_verified(lease: Any, *, outcome: str) -> Path:
    """Release once and prove the owned current record no longer exists."""

    try:
        history = lease.release(outcome=outcome)
    except Exception as error:
        try:
            lease.close()
        finally:
            raise Stage1S19ManualInterventionRequired("S1_9_LEASE_RELEASE_FAILED_RECORD_PRESERVED") from error
    current = getattr(lease, "current_path", None)
    if isinstance(current, Path) and current.exists():
        try:
            lease.close()
        finally:
            raise Stage1S19ManualInterventionRequired("S1_9_LEASE_RELEASE_RECORD_STILL_PRESENT")
    if not isinstance(history, Path) or not history.is_file():
        raise Stage1S19ManualInterventionRequired("S1_9_LEASE_HISTORY_MISSING_AFTER_RELEASE")
    return history


def _git(repository: Path, *args: str) -> str:
    done = subprocess.run(["git", "-C", str(repository), *args], text=True, capture_output=True, timeout=30, check=False)
    if done.returncode:
        raise Stage1S19FormalError(f"S1_9_GIT_FAILED:{args[0]}")
    return done.stdout.strip()


def _logical(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S19FormalError(f"S1_9_LOGICAL_REFERENCE_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S19FormalError(f"S1_9_LOGICAL_REFERENCE_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S19FormalError(f"S1_9_LOGICAL_REFERENCE_ESCAPE:{field}") from error
    return candidate


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1S19FormalError(f"S1_9_OBJECT_INVALID:{field}")
    return dict(value)


def _parse_json_object(raw: str, *, field: str, required: set[str]) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Stage1S19FormalError(f"S1_9_JSON_ARGUMENT_INVALID:{field}") from error
    if not isinstance(value, Mapping) or set(value) != required or any(not isinstance(item, str) or not item for item in value.values()):
        raise Stage1S19FormalError(f"S1_9_JSON_ARGUMENT_FIELDS_INVALID:{field}")
    return {str(key): str(item) for key, item in value.items()}


def _parse_uuids(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(values) != 4 or len(values) != len(set(values)) or any(_UUID.fullmatch(item) is None for item in values):
        raise Stage1S19FormalError("S1_9_APPROVED_UUIDS_INVALID")
    return values


def _validate_child_environment(environment: Mapping[str, str], *, approved: tuple[str, ...], run_token: str) -> dict[str, str]:
    """Validate the pre-CUDA policy inherited by every isolated worker.

    This code deliberately does not import torch.  It freezes process-start
    settings (notably CUBLAS and Python hash seed) in the CPU-only parent and
    returns only the allowlisted values that may be published in the attempt
    record.
    """

    expected_visible = ",".join(approved)
    if (
        len(approved) != 4
        or len(set(approved)) != 4
        or any(_UUID.fullmatch(item) is None for item in approved)
        or environment.get("CUDA_VISIBLE_DEVICES") != expected_visible
        or environment.get("S1_9_RUN_TOKEN") != run_token
        or any(environment.get(key) != value for key, value in _DETERMINISM_ENV.items())
    ):
        raise Stage1S19FormalError("S1_9_PRE_CUDA_ENVIRONMENT_POLICY_INVALID")
    return {
        "cublas_workspace_config": _DETERMINISM_ENV["CUBLAS_WORKSPACE_CONFIG"],
        "pythonhashseed": _DETERMINISM_ENV["PYTHONHASHSEED"],
        "cuda_visible_devices": expected_visible,
    }


def _worker_environment_summary_valid(value: object, *, cuda_visible_devices: str, local_rank: int, local_gpu_uuid: str) -> bool:
    if not isinstance(value, Mapping) or set(value) != _ENVIRONMENT_SUMMARY_KEYS:
        return False
    summary = dict(value)
    versions = ("torch_version", "cuda_runtime_version", "cudnn_version", "nccl_version")
    return (
        all(isinstance(summary[key], str) and bool(summary[key]) for key in versions)
        and summary["deterministic_algorithms"] is True
        and summary["cudnn_benchmark"] is False
        and summary["cudnn_deterministic"] is True
        and summary["cublas_workspace_config"] == _DETERMINISM_ENV["CUBLAS_WORKSPACE_CONFIG"]
        and summary["pythonhashseed"] == _DETERMINISM_ENV["PYTHONHASHSEED"]
        and summary["cuda_visible_devices"] == cuda_visible_devices
        and summary["local_rank"] == local_rank
        and summary["local_gpu_uuid"] == local_gpu_uuid
        and isinstance(summary["local_gpu_uuid"], str)
        and _UUID.fullmatch(summary["local_gpu_uuid"]) is not None
    )


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    done = subprocess.run(list(command), text=True, capture_output=True, env=None if env is None else dict(env), timeout=timeout, check=False)
    if done.returncode:
        raise Stage1S19FormalError(f"S1_9_COMMAND_FAILED:{Path(command[0]).name}:{done.stderr[-800:]}")
    return done


def _capability(data_root: Path, reference: str, binding: Mapping[str, str], approved: tuple[str, ...]) -> dict[str, Any]:
    from param_importance_nlp.contracts.runtime_evidence import RuntimeCapabilityEvidence
    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact

    if set(binding) != {"task_id", "artifact_kind", "artifact_hash", "config_hash"}:
        raise Stage1S19FormalError("S1_9_CAPABILITY_BINDING_FIELDS_INVALID")
    _logical(data_root, reference, field="capability_ref")
    try:
        artifact = load_committed_task_artifact(data_root, reference, require_formal=True)
    except Exception as error:
        raise Stage1S19FormalError("S1_9_CAPABILITY_COMMIT_INVALID") from error
    if (artifact.identity.task_id, artifact.identity.artifact_kind, artifact.identity.artifact_hash, artifact.identity.config_hash) != (binding["task_id"], binding["artifact_kind"], binding["artifact_hash"], binding["config_hash"]):
        raise Stage1S19FormalError("S1_9_CAPABILITY_BINDING_MISMATCH")
    try:
        evidence = RuntimeCapabilityEvidence.from_mapping(artifact.payload)
    except Exception as error:
        raise Stage1S19FormalError("S1_9_CAPABILITY_PAYLOAD_INVALID") from error
    metadata = _mapping(evidence.metadata, field="capability.metadata")
    allowed = metadata.get("allowed_gpu_uuids")
    if (
        evidence.capability != "cuda"
        or evidence.status != "VERIFIED"
        or not isinstance(allowed, list)
        or len(allowed) != 4
        or len(set(allowed)) != 4
        or any(not isinstance(item, str) or _UUID.fullmatch(item) is None for item in allowed)
        or tuple(allowed) != approved
    ):
        raise Stage1S19FormalError("S1_9_APPROVED_UUIDS_NOT_IN_CAPABILITY")
    return {"reference": reference, "binding": dict(binding), "allowed_gpu_uuids": list(allowed), "artifact_commit_ref": artifact.identity.commit_ref}


def _upstream_role(data_root: Path, index_ref: str, role: str) -> dict[str, Any]:
    """Load one already hash-pinned upstream role without trusting its path."""

    from param_importance_nlp.contracts.jsonio import load_canonical_json

    index_path = _logical(data_root, index_ref, field="upstream.index")
    index = _mapping(load_canonical_json(index_path), field="upstream.index")
    refs, hashes = _mapping(index.get("role_refs"), field="upstream.role_refs"), _mapping(index.get("role_sha256"), field="upstream.role_sha256")
    if role not in refs or role not in hashes:
        raise Stage1S19FormalError(f"S1_9_UPSTREAM_ROLE_MISSING:{role}")
    try:
        path = _logical(data_root, refs[role], field=f"upstream.{role}")
    except Stage1S19FormalError:
        path = (index_path.parent / str(refs[role])).resolve()
    if not path.is_file():
        path = (index_path.parent / str(refs[role])).resolve()
    if not path.is_file() or not isinstance(hashes[role], str) or _sha(path) != hashes[role]:
        raise Stage1S19FormalError(f"S1_9_UPSTREAM_ROLE_HASH_INVALID:{role}")
    return _mapping(load_canonical_json(path), field=f"upstream.{role}")


def _source_map(value: Mapping[str, Any], *, field: str) -> dict[str, str]:
    for key in ("implementation_source_sha256", "source_sha256", "source_hashes", "source_map"):
        raw = value.get(key)
        if isinstance(raw, Mapping) and raw and all(isinstance(name, str) and isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None for name, digest in raw.items()):
            return {str(name): str(digest) for name, digest in raw.items()}
    raise Stage1S19FormalError(f"S1_9_UPSTREAM_SOURCE_MAP_MISSING:{field}")


def _s1_8_v8_handoff_attestation(data_root: Path, index_ref: str, report: Mapping[str, Any]) -> dict[str, Any]:
    """Bind S1.8's final v8 index, report, validation, and four phases.

    This is deliberately a separate attestation rather than a path-only
    assertion.  The S1.8 index pins the report and validation by digest; the
    validation must carry the same v4 quiescence-role wire as the v8 report.
    """

    from param_importance_nlp.contracts.jsonio import load_canonical_json

    index_path = _logical(data_root, index_ref, field="s1_8.v8_index")
    index = _mapping(load_canonical_json(index_path), field="s1_8.v8_index")
    if index.get("schema_version") != "stage1-s1-8-formalization-index-v8":
        raise Stage1S19FormalError("S1_9_S1_8_V8_INDEX_REQUIRED")
    validation_ref, validation_sha = index.get("validation_ref"), index.get("validation_sha256")
    if not isinstance(validation_ref, str) or not isinstance(validation_sha, str):
        raise Stage1S19FormalError("S1_9_S1_8_V8_VALIDATION_BINDING_INVALID")
    validation_path = (index_path.parent / validation_ref).resolve()
    if not validation_path.is_file() or _sha(validation_path) != validation_sha:
        raise Stage1S19FormalError("S1_9_S1_8_V8_VALIDATION_HASH_INVALID")
    validation = _mapping(load_canonical_json(validation_path), field="s1_8.v8_validation")
    if (
        report.get("schema_version") != "stage1-s1-8-ddp-report-v8"
        or validation.get("schema_version") != "stage1-s1-8-validation-v8"
        or validation.get("gpu_quiescence") != report.get("gpu_quiescence")
    ):
        raise Stage1S19FormalError("S1_9_S1_8_V8_QUIESCENCE_WIRE_INVALID")
    reproduction_refs = _mapping(index.get("reproduction_role_refs"), field="s1_8.v8_reproduction_role_refs")
    reproduction_sha = _mapping(index.get("reproduction_role_sha256"), field="s1_8.v8_reproduction_role_sha256")
    quiescence = _mapping(report.get("gpu_quiescence"), field="s1_8.v8_gpu_quiescence")
    for phase, reproduction_role in {
        "prelease": "prelease_gpu_quiescence",
        "post_worker": "post_worker_gpu_quiescence",
        "post_release": "post_release_gpu_quiescence",
        "reacquire_preflight": "reacquire_preflight_gpu_quiescence",
    }.items():
        binding = _mapping(quiescence.get(phase), field="s1_8.v8_gpu_quiescence." + phase)
        reference, digest = binding.get("ref"), binding.get("sha256")
        if not isinstance(reference, str) or not isinstance(digest, str) or reproduction_refs.get(reproduction_role) != reference or reproduction_sha.get(reproduction_role) != digest:
            raise Stage1S19FormalError("S1_9_S1_8_V8_QUIESCENCE_WIRE_INVALID")
        path = (index_path.parent / reference).resolve()
        if not path.is_file() or _sha(path) != digest:
            raise Stage1S19FormalError("S1_9_S1_8_V8_QUIESCENCE_WIRE_INVALID")
        observation = _mapping(load_canonical_json(path), field="s1_8.v8_gpu_quiescence." + phase + ".observation")
        if observation.get("schema_version") != "stage1-s1-8-gpu-quiescence-v4" or observation.get("status") != "PASS":
            raise Stage1S19FormalError("S1_9_S1_8_V8_QUIESCENCE_WIRE_INVALID")
    replay_ref, replay_sha = index.get("replay_ref"), index.get("replay_sha256")
    role_refs, role_sha = _mapping(index.get("role_refs"), field="s1_8.v8_role_refs"), _mapping(index.get("role_sha256"), field="s1_8.v8_role_sha256")
    array_ref, array_sha = role_refs.get("array_bundle"), role_sha.get("array_bundle")
    comparison_ref, comparison_sha = role_refs.get("comparison_table"), role_sha.get("comparison_table")
    if not all(isinstance(value, str) for value in (replay_ref, replay_sha, array_ref, array_sha, comparison_ref, comparison_sha)):
        raise Stage1S19FormalError("S1_9_S1_8_V8_ROLE_BINDING_INVALID")
    replay_path, array_path, comparison_path = (index_path.parent / str(replay_ref)).resolve(), (index_path.parent / str(array_ref)).resolve(), (index_path.parent / str(comparison_ref)).resolve()
    if not replay_path.is_file() or not array_path.is_file() or not comparison_path.is_file() or _sha(replay_path) != replay_sha or _sha(array_path) != array_sha or _sha(comparison_path) != comparison_sha:
        raise Stage1S19FormalError("S1_9_S1_8_V8_ROLE_HASH_INVALID")
    replay, array_bundle, comparison = _mapping(load_canonical_json(replay_path), field="s1_8.v8_replay"), _mapping(load_canonical_json(array_path), field="s1_8.v8_array_bundle"), _mapping(load_canonical_json(comparison_path), field="s1_8.v8_comparison")
    if replay.get("schema_version") != "stage1-s1-8-replay-validation-v3" or comparison.get("schema_version") != "stage1-s1-8-comparison-table-v2":
        raise Stage1S19FormalError("S1_9_S1_8_V8_ROLE_VERSION_INVALID")
    array_routes = _mapping(array_bundle.get("route_artifacts"), field="s1_8.v8_array_bundle.routes")
    if array_bundle.get("schema_version") != "stage1-s1-8-array-bundle-v2" or set(array_routes) != set(_S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS):
        raise Stage1S19FormalError("S1_9_S1_8_V8_ARRAY_BUNDLE_WIRE_INVALID")
    for route, expected_refs in _S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS.items():
        descriptor = _mapping(array_routes[route], field="s1_8.v8_array_bundle." + route)
        if {key: descriptor.get(key) for key in expected_refs} != expected_refs:
            raise Stage1S19FormalError("S1_9_S1_8_V8_ARRAY_BUNDLE_WIRE_INVALID")
    return {
        "index_schema_version": index["schema_version"],
        "ddp_report_schema_version": report["schema_version"],
        "validation_schema_version": validation["schema_version"],
        "implementation_source_sha256": index.get("implementation_source_sha256"),
        "reproduction_role_refs": index.get("reproduction_role_refs"),
        "reproduction_role_sha256": index.get("reproduction_role_sha256"),
        "gpu_quiescence": report.get("gpu_quiescence"),
        "replay_schema_version": replay["schema_version"],
        "comparison_table_schema_version": comparison["schema_version"],
        "array_bundle_schema_version": array_bundle["schema_version"],
        "array_bundle_route_refs": _S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS,
    }


def _consumer_diff(repository: Path, producer_commit: str) -> list[str]:
    if _COMMIT.fullmatch(producer_commit) is None:
        raise Stage1S19FormalError("S1_9_UPSTREAM_PRODUCER_COMMIT_INVALID")
    changed = [line for line in _git(repository, "diff", "--name-only", f"{producer_commit}..HEAD").splitlines() if line]
    allowed_exact = {"src/param_importance_nlp/runtime/optimizer.py", *_S1_9_FROZEN_CONSUMER_FILES, *_S1_10_FROZEN_CONSUMER_FILES, *_S1_10_SHARED_RUNTIME_FILES}
    rejected = [path for path in changed if path not in allowed_exact]
    if rejected:
        raise Stage1S19FormalError("S1_9_UPSTREAM_CONSUMER_DIFF_UNAUTHORIZED:" + ",".join(rejected))
    return changed


def _git_file_sha256(repository: Path, commit: str, path: str) -> str:
    """Hash an exact historical source blob without materialising it on disk."""

    if _COMMIT.fullmatch(commit) is None or path not in _S1_7_AUTHORIZED_SHARED_DEPENDENCIES:
        raise Stage1S19FormalError("S1_9_SHARED_SOURCE_HISTORY_ARGUMENT_INVALID")
    done = subprocess.run(["git", "-C", str(repository), "show", f"{commit}:{path}"], capture_output=True, check=False, timeout=30)
    if done.returncode:
        raise Stage1S19FormalError("S1_9_SHARED_SOURCE_HISTORY_UNAVAILABLE")
    return hashlib.sha256(done.stdout).hexdigest()


def _s1_7_shared_dependency_attestation(repository: Path, report: Mapping[str, Any], *, producer_commit: str, changed_paths: Sequence[str]) -> dict[str, Any]:
    """Audit r11's no-source-map report by frozen global consumer allowlist.

    The real S1.7 r11 worker report predates per-file source maps.  We do not
    invent one.  Instead, it binds its provenance commit and the global
    producer-to-consumer diff is already constrained by ``_consumer_diff``;
    only this named, reviewed helper may be a shared runtime drift.
    """

    if report.get("source_git_commit") != producer_commit:
        raise Stage1S19FormalError("S1_9_S1_7_REPORT_PROVENANCE_COMMIT_INVALID")
    unauthorised_shared = set(changed_paths) & {
        path for path in changed_paths
        if path.startswith("src/param_importance_nlp/") and not path.startswith(("src/param_importance_nlp/stage1_precision", "src/param_importance_nlp/stage1_ddp")) and path not in _S1_7_AUTHORIZED_SHARED_DEPENDENCIES
    }
    if unauthorised_shared:
        raise Stage1S19FormalError("S1_9_S1_7_SHARED_DEPENDENCY_DRIFT_REQUIRES_RERUN:" + ",".join(sorted(unauthorised_shared)))
    dependency_hashes: dict[str, dict[str, Any]] = {}
    for path in sorted(_S1_7_AUTHORIZED_SHARED_DEPENDENCIES):
        current_path = repository / path
        if not current_path.is_file():
            raise Stage1S19FormalError("S1_9_SHARED_SOURCE_CURRENT_MISSING")
        dependency_hashes[path] = {
            "producer_sha256": _git_file_sha256(repository, producer_commit, path),
            "consumer_sha256": _sha(current_path),
            "changed": path in changed_paths,
        }
    return {
        "source_map_mode": "r11_no_per_file_source_map_global_diff_allowlist",
        "report_source_git_commit": producer_commit,
        "authorized_shared_dependencies": sorted(_S1_7_AUTHORIZED_SHARED_DEPENDENCIES),
        "authorized_shared_dependency_hashes": dependency_hashes,
    }


def _optimizer_clip_cpu_replay() -> dict[str, Any]:
    """Current-source, CPU-only replay of the one authorised shared change."""

    import torch
    from param_importance_nlp.runtime.optimizer import compute_global_clip_factor

    gradients = {
        "large": torch.tensor([3.0, -4.0], dtype=torch.float32),
        "other": torch.tensor([12.0, 5.0], dtype=torch.float32),
        "absent": None,
    }
    reversed_gradients = {name: gradients[name] for name in reversed(tuple(gradients))}
    norm, factor = compute_global_clip_factor(gradients, max_norm=1.0)
    reverse_norm, reverse_factor = compute_global_clip_factor(reversed_gradients, max_norm=1.0)
    expected = math.sqrt(25.0 + 169.0)
    empty = compute_global_clip_factor({}, max_norm=1.0)
    none_only = compute_global_clip_factor({"absent": None}, max_norm=None)
    passed = (
        math.isfinite(norm)
        and math.isclose(norm, expected, abs_tol=1e-15, rel_tol=0.0)
        and math.isclose(factor, 1.0 / (expected + 1e-12), abs_tol=1e-15, rel_tol=0.0)
        and math.isclose(reverse_norm, norm, abs_tol=1e-15, rel_tol=0.0)
        and math.isclose(reverse_factor, factor, abs_tol=1e-15, rel_tol=0.0)
        and empty == (0.0, 1.0)
        and none_only == (0.0, 1.0)
    )
    return {
        "profile": "runtime_optimizer_compute_global_clip_factor_cpu_fp64_order_independent",
        "expected_norm": expected,
        "observed_norm": norm,
        "observed_factor": factor,
        "reverse_observed_norm": reverse_norm,
        "reverse_observed_factor": reverse_factor,
        # Public attestation is canonical JSON; do not leak Python tuples from
        # the helper's compatibility contract into a publishable role.
        "empty": list(empty),
        "none_only": list(none_only),
        "passed": passed,
    }


def _s1_9_checkpoint_resume_cpu_replay(work: Path) -> dict[str, Any]:
    """Exercise S1.9's real current-source checkpoint boundary on CPU.

    S1.10 changed the transaction/lineage implementation beneath
    ``TrainingEngine``.  This compact replay is intentionally S1.9-shaped:
    it writes a real CheckpointStore payload, rejects an omission before the
    target engine mutates, and compares the next production step after a
    fresh-engine restore.  It does not claim to reproduce CUDA/BF16 arithmetic.
    """

    import copy
    import torch
    from param_importance_nlp.providers import InMemoryDatasetAdapter, TorchModelAdapter
    from param_importance_nlp.runtime.checkpoint import CheckpointStore
    from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec
    from param_importance_nlp.contracts.jsonio import canonical_json_hash
    from param_importance_nlp.stage1_precision import _S19FiniteSkipFiniteClassifier, _engine_microbatch, _state_wire

    if work.exists():
        raise Stage1S19FormalError("S1_9_RUNTIME_COMPATIBILITY_REPLAY_WORK_EXISTS")
    work.mkdir(parents=True)
    store = CheckpointStore(work / "authoritative")

    def build(store_override: CheckpointStore) -> TrainingEngine:
        module = _S19FiniteSkipFiniteClassifier(device=torch.device("cpu"))
        batches = tuple(
            (
                _engine_microbatch(sample_id=f"compat-{index}", micro=0, value=float(index + 1), device=torch.device("cpu")),
                _engine_microbatch(sample_id=f"compat-{index}", micro=1, value=float(index + 2), device=torch.device("cpu")),
            )
            for index in range(3)
        )
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False)
        return TrainingEngine(
            spec=TrainingRunSpec("s19-current-source-compat", "local_fixture", max_steps=3, max_attempts=3, importance_enabled=True, estimator_name="u", accumulation_dtype="float32", weights_exogenous=True, common_mean_assumption=True),
            model=TorchModelAdapter(module, task_type="sequence_classification"),
            optimizer=optimizer,
            cursor=InMemoryDatasetAdapter("s19-current-source-compat", batches).cursor(seed=1909),
            checkpoint_store=store_override,
        )

    def run_one(engine: TrainingEngine) -> dict[str, Any]:
        record = engine._run_attempt(engine.cursor.next_microbatches())
        engine._records.append(record)  # Production run-loop bookkeeping.
        return record.to_dict()

    def identity(engine: TrainingEngine) -> str:
        return canonical_json_hash(_state_wire({
            "state": engine.state.to_dict(),
            "model": engine.model.module.state_dict(),
            "optimizer": engine.optimizer.state_dict(),
            "cursor": dict(engine.cursor.state_dict()),
            "importance": None if engine.tracker is None else engine.tracker.accumulator.state_dict(),
            "records": [record.to_dict() for record in engine._records],
            "checkpoint_ids": list(engine._checkpoint_ids),
        }))

    source = build(store)
    run_one(source); run_one(source)
    checkpoint_id = source.save_checkpoint()
    payload, commit = store.load(checkpoint_id)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "training-checkpoint-state-v2":
        raise Stage1S19FormalError("S1_9_RUNTIME_COMPATIBILITY_REPLAY_SCHEMA_INVALID")
    metadata = {
        "run_spec_hash": source.spec.spec_hash,
        "registry_hash": source.registry.coordinate_registry_hash,
        "optimizer_contract_hash": source.registry.optimizer_contract_hash,
        "runtime_layout_hash": source.registry.runtime_layout_hash,
        "world_size": 1,
    }
    negative_id = "s19-current-source-compat-omission"
    malformed = copy.deepcopy(dict(payload))
    state = dict(malformed["training_state"])
    state["last_checkpoint_id"] = negative_id
    malformed["training_state"] = state
    malformed["checkpoint_ids"] = [negative_id]
    malformed["importance_trajectory_points"] = []
    malformed.pop("importance")
    negative_store = CheckpointStore(work / "omission")
    negative_store.publish(negative_id, malformed, generation=2, metadata=metadata)
    negative = build(negative_store)
    negative_before = identity(negative)
    try:
        negative.resume_checkpoint(negative_id)
    except Exception:
        omission_rejected_before_mutation = identity(negative) == negative_before
    else:
        omission_rejected_before_mutation = False
    restored = build(store)
    restored_id = restored.resume_checkpoint(checkpoint_id)
    source_next, restored_next = run_one(source), run_one(restored)
    passed = (
        commit.checkpoint_id == checkpoint_id
        and restored_id == checkpoint_id
        and omission_rejected_before_mutation
        and source_next == restored_next
        and identity(source) == identity(restored)
    )
    return {
        "profile": "s1_9_current_source_training_checkpoint_resume_cpu",
        "checkpoint_id": checkpoint_id,
        "checkpoint_schema_version": payload["schema_version"],
        "omission_rejected_before_mutation": omission_rejected_before_mutation,
        "fresh_engine_next_step_exact": source_next == restored_next,
        "fresh_engine_final_state_exact": identity(source) == identity(restored),
        "passed": passed,
    }


def _nonproducer_runtime_attestation(
    repository: Path,
    *,
    s1_7_producer: str,
    s1_7_report: Mapping[str, Any],
    s1_8_sources: Mapping[str, str],
    changed_paths: Sequence[str],
    replay_root: Path,
) -> dict[str, Any]:
    """Permit only reviewed S1.10 runtime drift outside both producer maps."""

    affected = sorted(set(changed_paths) & _S1_10_SHARED_RUNTIME_FILES)
    if not affected:
        return {"affected_paths": [], "s1_8_source_map_excludes_paths": [], "s1_7_oracle_training_import_isolated": True, "checkpoint_group_producer_math_exclusion": "src/param_importance_nlp/runtime/checkpoint_group.py" not in s1_8_sources, "current_source_cpu_replays": {}}
    included = sorted(set(affected) & set(s1_8_sources))
    if included:
        raise Stage1S19FormalError("S1_9_S1_8_NONPRODUCER_RUNTIME_PATH_IN_SOURCE_MAP:" + ",".join(included))
    if s1_7_report.get("status") != "PASS":
        raise Stage1S19FormalError("S1_9_S1_7_ORACLE_ROLE_NOT_PASS")
    source = _git(repository, "show", f"{s1_7_producer}:ops/stage1/formalize_s1_7.py")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise Stage1S19FormalError("S1_9_S1_7_ORACLE_RUNTIME_ISOLATION_UNPROVEN") from error
    oracle = next((node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_oracle_replay"), None)
    blocked: tuple[str, ...] | None = None
    guarded_rejects = False
    installs_guard = False
    restores_import = False
    if isinstance(oracle, ast.FunctionDef):
        for node in ast.walk(oracle):
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "blocked" for target in node.targets) and isinstance(node.value, ast.Tuple) and all(isinstance(item, ast.Constant) and isinstance(item.value, str) for item in node.value.elts):
                blocked = tuple(str(item.value) for item in node.value.elts)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute) and isinstance(node.test.func.value, ast.Name) and node.test.func.value.id == "name" and node.test.func.attr == "startswith" and len(node.test.args) == 1 and isinstance(node.test.args[0], ast.Name) and node.test.args[0].id == "blocked" and any(isinstance(child, ast.Raise) and isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name) and child.exc.func.id == "ImportError" for child in node.body):
                guarded_rejects = True
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute) and isinstance(node.targets[0].value, ast.Name) and node.targets[0].value.id == "builtins" and node.targets[0].attr == "__import__":
                if isinstance(node.value, ast.Name) and node.value.id == "guarded":
                    installs_guard = True
                if isinstance(node.value, ast.Name) and node.value.id == "original":
                    restores_import = True
    if blocked != ("param_importance_nlp.core.estimators", "param_importance_nlp.runtime.training", "param_importance_nlp.stage1_single_gpu") or not (guarded_rejects and installs_guard and restores_import):
        raise Stage1S19FormalError("S1_9_S1_7_ORACLE_RUNTIME_ISOLATION_UNPROVEN")
    replays: dict[str, Any] = {}
    if "src/param_importance_nlp/runtime/training.py" in affected:
        replays["src/param_importance_nlp/runtime/training.py"] = _s1_9_checkpoint_resume_cpu_replay(replay_root / "training-resume")
    # checkpoint_group.py is excluded from S1.8's explicit source map and
    # S1.7's independent oracle import boundary.  Its only current consumer
    # in this diff is the frozen S1.10 worker, so it cannot alter either
    # producer's formal mathematical output.
    if any(not value.get("passed") for value in replays.values()):
        raise Stage1S19FormalError("S1_9_SHARED_RUNTIME_DRIFT_REPLAY_FAILED")
    return {
        "affected_paths": affected,
        "s1_8_source_map_excludes_paths": affected,
        "s1_7_oracle_training_import_isolated": True,
        "checkpoint_group_producer_math_exclusion": "src/param_importance_nlp/runtime/checkpoint_group.py" not in s1_8_sources,
        "current_source_cpu_replays": replays,
    }


def _upstream_compatibility_attestation(repository: Path, data_root: Path, *, s1_7_ref: str, s1_7: Mapping[str, Any], s1_8_ref: str, s1_8: Mapping[str, Any], replay_root: Path | None = None) -> dict[str, Any]:
    """Prove an old formal producer remains consumable at the new consumer.

    This does not rerun S1.7 GPU work.  It rejects any producer dependency
    drift other than the explicitly reviewed, CPU-replayed clipping helper.
    """

    changed_s17 = _consumer_diff(repository, str(s1_7["s1_7_generator_commit"]))
    changed_s18 = _consumer_diff(repository, str(s1_8["s1_8_generator_commit"]))
    s17_report = _upstream_role(data_root, s1_7_ref, "single_gpu_report")
    s17_attestation = _s1_7_shared_dependency_attestation(repository, s17_report, producer_commit=str(s1_7["s1_7_generator_commit"]), changed_paths=[path for path in changed_s17 if path not in _S1_10_SHARED_RUNTIME_FILES])
    affected_s17 = sorted(set(changed_s17) & _S1_7_AUTHORIZED_SHARED_DEPENDENCIES)
    s18_report = _upstream_role(data_root, s1_8_ref, "ddp_report")
    s18_sources = _source_map(s18_report, field="s1_8.ddp_report")
    s18_v8_handoff = _s1_8_v8_handoff_attestation(data_root, s1_8_ref, s18_report)
    if replay_root is None:
        replay_root = data_root / "tmp" / "s1-9-compatibility"
    nonproducer_runtime = _nonproducer_runtime_attestation(repository, s1_7_producer=str(s1_7["s1_7_generator_commit"]), s1_7_report=s17_report, s1_8_sources=s18_sources, changed_paths=[*changed_s17, *changed_s18], replay_root=replay_root)
    affected_s18 = sorted(set(changed_s18) & set(s18_sources))
    if set(affected_s18) - _S1_7_AUTHORIZED_SHARED_DEPENDENCIES:
        raise Stage1S19FormalError("S1_9_S1_8_DEPENDENCY_DRIFT_REQUIRES_RERUN:" + ",".join(affected_s18))
    s18_authorized_drift: dict[str, Any] = {}
    for path in affected_s18:
        reported_digest = s18_sources[path]
        historical_digest = _git_file_sha256(repository, str(s1_8["s1_8_generator_commit"]), path)
        if reported_digest != historical_digest:
            raise Stage1S19FormalError("S1_9_S1_8_SOURCE_MAP_PRODUCER_HASH_MISMATCH:" + path)
        s18_authorized_drift[path] = {"reported_producer_sha256": reported_digest, "historical_producer_sha256": historical_digest, "current_consumer_sha256": _sha(repository / path), "compatibility_revalidated": True}
    replay = _optimizer_clip_cpu_replay()
    if not replay["passed"]:
        raise Stage1S19FormalError("S1_9_OPTIMIZER_CLIP_COMPATIBILITY_REPLAY_FAILED")
    result = _with_hash({"schema_version": "stage1-s1-9-upstream-compatibility-v7", "status": "PASS", "s1_7_producer": s1_7["s1_7_generator_commit"], "s1_8_producer": s1_8["s1_8_generator_commit"], "consumer_commit": _git(repository, "rev-parse", "HEAD"), "s1_7_to_consumer_changed_paths": changed_s17, "s1_8_to_consumer_changed_paths": changed_s18, "s1_7_source_attestation": s17_attestation, "s1_8_source_dependencies": s18_sources, "s1_8_v8_handoff": s18_v8_handoff, "s1_7_affected_dependencies": affected_s17, "s1_8_affected_dependencies": affected_s18, "s1_8_authorized_shared_drift": s18_authorized_drift, "authorized_shared_change": "src/param_importance_nlp/runtime/optimizer.py", "current_source_cpu_clip_replay": replay, "nonproducer_runtime_attestation": nonproducer_runtime})
    _validate_s1_9_schemas(repository, {"upstream_compatibility": result})
    return result


_GPU_QUIESCENCE_TIMEOUT_SECONDS = 180.0
_GPU_QUIESCENCE_POLL_SECONDS = 1.0
_GPU_QUIESCENCE_CONSECUTIVE_SAMPLES = 3
_GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES = 2
_GPU_QUIESCENCE_MAX_SAMPLES = 9
_GPU_OPERATIONAL_TIMEOUT_BASIS = {
    "measurement_method": "frozen_linux_cpu_only_nvidia_smi_management_queries",
    "combined_inventory_recovery_seconds": 6.166325362000862,
    "compute_apps_seconds": 6.113894288000665,
    "two_query_sample_seconds": 12.280219650001527,
    "maximum_transient_samples": 2,
    "per_sample_management_budget_seconds": 15.0,
    "maximum_sample_count": 9,
    "maximum_cadence_count": 8,
    "nine_samples_plus_eight_cadences_seconds": 143.0,
    "fixed_timeout_seconds": 180.0,
    "fixed_margin_seconds": 37.0,
    "dynamic_fitting": False,
}


def _gpu_probe_exception_reason(error: BaseException) -> str:
    if isinstance(error, Stage1S19FormalError):
        return str(error)
    if isinstance(error, subprocess.TimeoutExpired):
        return "S1_9_GPU_PROBE_EXCEPTION:TimeoutExpired"
    if isinstance(error, FileNotFoundError):
        return "S1_9_GPU_PROBE_EXCEPTION:FileNotFoundError"
    if isinstance(error, OSError):
        return "S1_9_GPU_PROBE_EXCEPTION:OSError"
    raise Stage1S19FormalError("S1_9_GPU_PROBE_EXCEPTION_CLASS_UNAUTHORIZED") from error


def _gpu_prelease_evidence(approved: tuple[str, ...], quiescence: Mapping[str, Any]) -> dict[str, Any]:
    """Bind bounded prelease health/quiescence before lease construction."""

    return _with_hash({
        "schema_version": "stage1-s1-9-gpu-prelease-v3",
        "status": quiescence.get("status"),
        "approved_gpu_uuids": list(approved),
        "quiescence": dict(quiescence),
    })


def _gpu_probe_once(approved: tuple[str, ...]) -> dict[str, Any]:
    """Read selected inventory and Recovery Action atomically in one query.

    Compute applications intentionally remain a second independent command:
    it is a different nvidia-smi query class, while the inventory/Recovery
    tuple must never be stitched from separate temporal samples.
    """

    query = "index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap,gpu_recovery_action"
    rows = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]).stdout.splitlines()
    by_uuid: dict[str, dict[str, Any]] = {}
    for line in rows:
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 9 or not fields[0].isdigit() or _UUID.fullmatch(fields[1]) is None or not fields[2] or not fields[7] or not fields[8]:
            raise Stage1S19FormalError("S1_9_GPU_PREFLIGHT_PARSE_INVALID")
        try:
            row = {"physical_index": int(fields[0]), "uuid": fields[1], "name": fields[2], "memory_total_mib": int(fields[3]), "memory_used_mib": int(fields[4]), "utilization_percent": int(fields[5]), "temperature_c": int(fields[6]), "compute_capability": fields[7], "recovery_action": fields[8]}
            float(row["compute_capability"])
        except ValueError as error:
            raise Stage1S19FormalError("S1_9_GPU_PREFLIGHT_VALUE_INVALID") from error
        if any(int(row[name]) < 0 for name in ("memory_total_mib", "memory_used_mib", "utilization_percent", "temperature_c")):
            raise Stage1S19FormalError("S1_9_GPU_PREFLIGHT_VALUE_INVALID")
        if row["uuid"] in by_uuid:
            raise Stage1S19FormalError("S1_9_GPU_PREFLIGHT_DUPLICATE_UUID")
        by_uuid[row["uuid"]] = row
    if set(approved) - set(by_uuid):
        raise Stage1S19FormalError("S1_9_APPROVED_GPU_NOT_DISCOVERED")
    processes = _run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"]).stdout.strip()
    compute_apps: list[dict[str, Any]] = []
    for line in processes.splitlines():
        if not line.strip():
            continue
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 3 or _UUID.fullmatch(fields[0]) is None or not fields[1].isdigit() or not fields[2]:
            raise Stage1S19FormalError("S1_9_GPU_COMPUTE_PROCESS_PARSE_INVALID")
        compute_apps.append({"gpu_uuid": fields[0], "pid": int(fields[1]), "process_name": fields[2]})
    return {"selected": [dict(by_uuid[item]) for item in approved], "requested_uuid_order": list(approved), "compute_apps": compute_apps}


def _gpu_violations(probe: Mapping[str, Any], *, minimum_compute_capability: float, max_temperature_c: int) -> tuple[list[str], list[str]]:
    """Return all exact-idle violations and the fail-immediately subset."""

    selected = _mapping(probe, field="gpu.probe").get("selected")
    apps = probe.get("compute_apps")
    if not isinstance(selected, list) or not isinstance(apps, list):
        raise Stage1S19FormalError("S1_9_GPU_PROBE_SHAPE_INVALID")
    selected_uuids = {row.get("uuid") for row in selected if isinstance(row, Mapping)}
    violations: list[str] = []
    hard: list[str] = []
    for row in selected:
        if not isinstance(row, Mapping) or not isinstance(row.get("uuid"), str):
            raise Stage1S19FormalError("S1_9_GPU_PROBE_SHAPE_INVALID")
        uuid = row["uuid"]
        if row.get("recovery_action") != "None":
            hard.append("S1_9_GPU_RECOVERY_ACTION_NOT_NONE:" + uuid + ":" + str(row.get("recovery_action")))
        if not isinstance(row.get("compute_capability"), str) or float(row["compute_capability"]) < minimum_compute_capability or not isinstance(row.get("temperature_c"), int) or row["temperature_c"] > max_temperature_c:
            hard.append("S1_9_GPU_PREFLIGHT_HARDWARE_HEALTH_FAILED:" + uuid)
        if row.get("memory_used_mib") != 0:
            violations.append("S1_9_GPU_SELECTED_MEMORY_NONZERO:" + uuid)
        if row.get("utilization_percent") != 0:
            violations.append("S1_9_GPU_SELECTED_UTILIZATION_NONZERO:" + uuid)
    for app in apps:
        if not isinstance(app, Mapping) or not isinstance(app.get("gpu_uuid"), str):
            raise Stage1S19FormalError("S1_9_GPU_PROBE_SHAPE_INVALID")
        if app["gpu_uuid"] in selected_uuids:
            hard.append("S1_9_GPU_COMPUTE_PROCESS_PRESENT:" + app["gpu_uuid"] + ":" + str(app.get("pid")))
    return [*hard, *violations], hard


def _gpu_quiescence(approved: tuple[str, ...], *, phase: str, minimum_compute_capability: float, max_temperature_c: int, timeout_seconds: float = _GPU_QUIESCENCE_TIMEOUT_SECONDS, poll_seconds: float = _GPU_QUIESCENCE_POLL_SECONDS, consecutive_required: int = _GPU_QUIESCENCE_CONSECUTIVE_SAMPLES, max_transient_samples: int = _GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES) -> dict[str, Any]:
    """Require fixed 60 s / 1 s / three exact selected-idle samples.

    A completed third probe at the deadline is a timeout, not a PASS.  Any
    Recovery, health, selected-PID, identity, or parser fault returns a
    canonical FAIL observation before the caller can acquire or continue a
    lease.
    """

    if phase not in {"prelease", "post_worker"} or timeout_seconds != _GPU_QUIESCENCE_TIMEOUT_SECONDS or poll_seconds != _GPU_QUIESCENCE_POLL_SECONDS or consecutive_required != _GPU_QUIESCENCE_CONSECUTIVE_SAMPLES or max_transient_samples != _GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES:
        raise Stage1S19FormalError("S1_9_GPU_QUIESCENCE_ARGUMENT_INVALID")
    started_at, start = _now(), time.monotonic()
    deadline, consecutive, transients, samples, final_gpu = start + timeout_seconds, 0, 0, [], None

    def result(status: str, reason: str | None) -> dict[str, Any]:
        return _with_hash({"schema_version": "stage1-s1-9-gpu-quiescence-v3", "status": status, "phase": phase, "approved_gpu_uuids": list(approved), "started_at": started_at, "minimum_compute_capability": minimum_compute_capability, "max_temperature_c": max_temperature_c, "timeout_seconds": timeout_seconds, "sample_interval_seconds": poll_seconds, "required_consecutive_exact_idle_samples": consecutive_required, "max_transient_samples": max_transient_samples, "transient_observation_count": transients, "operational_timeout_basis": dict(_GPU_OPERATIONAL_TIMEOUT_BASIS), "samples": samples, "final_gpu": final_gpu, "failure_reason": reason})

    sample_index = 0
    while True:
        try:
            probe = _gpu_probe_once(approved)
        except (Stage1S19FormalError, subprocess.TimeoutExpired, FileNotFoundError, OSError) as error:
            reason = _gpu_probe_exception_reason(error)
            now = time.monotonic()
            samples.append({"sample_index": sample_index, "observed_at": _now(), "monotonic_elapsed_seconds": max(0.0, now - start), "requested_uuid_order": list(approved), "probe_error": reason, "exact_selected_idle": False, "consecutive_exact_idle_samples": consecutive, "transient_observation_count": transients})
            return result("FAILED", reason)
        now = time.monotonic()
        final_gpu = probe
        violations, hard = _gpu_violations(probe, minimum_compute_capability=minimum_compute_capability, max_temperature_c=max_temperature_c)
        exact_idle = not violations
        if hard:
            consecutive = 0
        elif exact_idle:
            consecutive += 1
        else:
            consecutive = 0; transients += 1
        samples.append({"sample_index": sample_index, "observed_at": _now(), "monotonic_elapsed_seconds": max(0.0, now - start), "requested_uuid_order": list(approved), "selected": probe["selected"], "compute_apps": probe["compute_apps"], "violations": violations, "exact_selected_idle": exact_idle, "consecutive_exact_idle_samples": consecutive, "transient_observation_count": transients})
        if hard:
            return result("FAILED", hard[0])
        if transients > max_transient_samples:
            return result("FAILED", "S1_9_GPU_QUIESCENCE_TRANSIENT_LIMIT")
        # Completion is checked after each whole probe and before accepting a
        # third sample.  Equality is deliberately failed closed.
        if now >= deadline:
            return result("FAILED", "S1_9_GPU_QUIESCENCE_TIMEOUT")
        if consecutive == consecutive_required:
            return result("PASS", None)
        if sample_index + 1 >= _GPU_QUIESCENCE_MAX_SAMPLES:
            return result("FAILED", "S1_9_GPU_QUIESCENCE_SAMPLE_LIMIT")
        sample_index += 1
        time.sleep(min(poll_seconds, max(0.0, deadline - now)))


def _proc_record(pid: int, *, run_token: str) -> dict[str, Any]:
    """Return the immutable Linux identity material needed before signalling.

    PID/PGID alone is vulnerable to reuse.  The launcher, UID, executable,
    start tick, parent and the inherited run token are all bound here and
    rechecked before any process-group signal is sent.
    """

    stat = Path(f"/proc/{pid}/stat")
    status, environ, cmdline, executable = (Path(f"/proc/{pid}/{name}") for name in ("status", "environ", "cmdline", "exe"))
    if not stat.is_file() or not status.is_file() or not environ.is_file() or not cmdline.is_file():
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_UNAVAILABLE")
    fields = stat.read_text(encoding="utf-8").split()
    if len(fields) < 22:
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_PARSE_INVALID")
    uid_line = next((line for line in status.read_text(encoding="utf-8").splitlines() if line.startswith("Uid:")), "")
    uid_fields = uid_line.split()
    token_present = any(item == f"S1_9_RUN_TOKEN={run_token}".encode("utf-8") for item in environ.read_bytes().split(b"\0"))
    if len(uid_fields) < 2 or not token_present:
        raise Stage1S19ManualInterventionRequired("S1_9_CHILD_TOKEN_OR_UID_INVALID")
    try:
        executable_path = os.readlink(executable)
    except OSError as error:
        raise Stage1S19FormalError("S1_9_CHILD_EXECUTABLE_UNAVAILABLE") from error
    return {"pid": pid, "pgid": os.getpgid(pid), "start_ticks": fields[21], "parent_pid": int(fields[3]), "uid": int(uid_fields[1]), "exe": executable_path, "cmdline_sha256": hashlib.sha256(cmdline.read_bytes()).hexdigest(), "run_token": run_token}


def _child_tree(fingerprint: Mapping[str, Any], *, run_token: str) -> list[dict[str, Any]]:
    pgid = fingerprint.get("pgid")
    if not isinstance(pgid, int):
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_INVALID")
    records: list[dict[str, Any]] = []
    proc = Path("/proc")
    if not proc.is_dir():
        raise Stage1S19FormalError("S1_9_PROC_AUDIT_UNAVAILABLE")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if os.getpgid(pid) == pgid:
                records.append(_proc_record(pid, run_token=run_token))
        except ProcessLookupError:
            # A process which vanished between the directory and stat reads is
            # not a signal target; a still-live unknown process is never
            # silently ignored below.
            continue
        except Stage1S19ManualInterventionRequired:
            # Never wrap a more specific identity/token failure in a generic
            # tree error: callers use this class to retain the lease and stop
            # rather than signalling an uncertain process group.
            raise
        except (PermissionError, OSError, Stage1S19FormalError) as error:
            raise Stage1S19ManualInterventionRequired("S1_9_CHILD_TREE_MEMBER_UNVERIFIABLE") from error
    return sorted(records, key=lambda item: int(item["pid"]))


def _process_group_pids(pgid: int) -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            pid = int(entry.name)
            if os.getpgid(pid) == pgid:
                result.append(pid)
        except ProcessLookupError:
            continue
        except (PermissionError, OSError) as error:
            raise Stage1S19ManualInterventionRequired("S1_9_PROCESS_GROUP_ENUMERATION_UNVERIFIABLE") from error
    return sorted(result)


def _run_token_processes(run_token: str) -> list[int]:
    result: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if f"S1_9_RUN_TOKEN={run_token}".encode("utf-8") in (entry / "environ").read_bytes().split(b"\0"):
                result.append(int(entry.name))
        except (OSError, PermissionError):
            continue
    return sorted(result)


def _assert_no_run_token_processes(run_token: str) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if not _run_token_processes(run_token):
            return
        time.sleep(0.2)
    raise Stage1S19ManualInterventionRequired("S1_9_CHILD_RESIDUAL_RUN_TOKEN_PROCESS")


def _assert_process_group_empty(fingerprint: Mapping[str, Any], *, run_token: str) -> None:
    pgid = fingerprint.get("pgid")
    if not isinstance(pgid, int):
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_INVALID")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        members = _process_group_pids(pgid)
        if not members:
            return
        # This enumerates the full group before allowing another poll.  Any
        # member without a verifiable identity/token raises ManualIntervention
        # rather than being accidentally covered by a matching parent.
        _child_tree(fingerprint, run_token=run_token)
        time.sleep(0.2)
    raise Stage1S19ManualInterventionRequired("S1_9_CHILD_RESIDUAL_PROCESS_GROUP")


def _verified_group_members(fingerprint: Mapping[str, Any], *, run_token: str) -> list[dict[str, Any]]:
    """Read the entire PGID twice and bind every surviving member to us.

    A launcher can exit between SIGTERM and the escalation check while a
    torchrun child remains alive.  Its absence is not itself unsafe, but it
    never authorizes a broad ``killpg``: every remaining PID must still be
    observable, token-bound, owned by the original UID, and in the recorded
    PGID at the instant immediately before the signal.
    """

    pgid, uid = fingerprint.get("pgid"), fingerprint.get("uid")
    if not isinstance(pgid, int) or not isinstance(uid, int):
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_INVALID")
    raw_members = _process_group_pids(pgid)
    if not raw_members:
        return []
    records = _child_tree(fingerprint, run_token=run_token)
    if [int(record.get("pid", -1)) for record in records] != raw_members:
        raise Stage1S19ManualInterventionRequired("S1_9_CHILD_TREE_MEMBERSHIP_DRIFT")
    if any(
        record.get("run_token") != run_token
        or int(record.get("pgid", -1)) != pgid
        or int(record.get("uid", -1)) != uid
        for record in records
    ):
        raise Stage1S19ManualInterventionRequired("S1_9_CHILD_TREE_IDENTITY_DRIFT")
    return records


def _verified_kill_process_group(fingerprint: Mapping[str, Any], *, run_token: str) -> None:
    pid, pgid = fingerprint.get("pid"), fingerprint.get("pgid")
    if not isinstance(pid, int) or not isinstance(pgid, int):
        raise Stage1S19FormalError("S1_9_CHILD_FINGERPRINT_INVALID")
    tree = _verified_group_members(fingerprint, run_token=run_token)
    if not tree:
        _assert_no_run_token_processes(run_token)
        return
    current = next((record for record in tree if int(record["pid"]) == pid), None)
    if current is not None:
        identity_keys = ("pid", "pgid", "start_ticks", "parent_pid", "uid", "exe", "cmdline_sha256", "run_token")
        if any(current.get(key) != fingerprint.get(key) for key in identity_keys):
            raise Stage1S19ManualInterventionRequired("S1_9_CHILD_IDENTITY_DRIFT")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if not _run_token_processes(run_token) and not _process_group_pids(pgid):
            return
        time.sleep(0.2)
    # The launcher may have exited, so do not dereference its PID here.  The
    # only safe escalation condition is a fresh, full, token/UID-bound PGID.
    surviving = _verified_group_members(fingerprint, run_token=run_token)
    if not surviving:
        _assert_no_run_token_processes(run_token)
        return
    os.killpg(pgid, signal.SIGKILL)
    _assert_process_group_empty(fingerprint, run_token=run_token)
    _assert_no_run_token_processes(run_token)


def _run_with_heartbeats(command: Sequence[str], *, env: Mapping[str, str], timeout_seconds: int, lease: Any, stdout: Path, stderr: Path, fingerprint_path: Path) -> None:
    run_token = env.get("S1_9_RUN_TOKEN")
    if not isinstance(run_token, str) or re.fullmatch(r"[0-9a-f]{64}", run_token) is None:
        raise Stage1S19FormalError("S1_9_CHILD_RUN_TOKEN_MISSING")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        child = subprocess.Popen(list(command), stdout=out, stderr=err, text=True, env=dict(env), start_new_session=True)
        fingerprint = _proc_record(child.pid, run_token=run_token)
        initial_tree = _child_tree(fingerprint, run_token=run_token)
        deadline = time.monotonic() + timeout_seconds
        while child.poll() is None:
            if time.monotonic() >= deadline:
                _verified_kill_process_group(fingerprint, run_token=run_token)
                child.wait(timeout=20)
                _write(fingerprint_path, _with_hash({"schema_version": "stage1-s1-9-child-fingerprint-v1", "status": "TIMED_OUT_CLEANED", "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(), "fingerprint": fingerprint, "initial_process_group_tree": initial_tree, "residual_run_token_pids": []}))
                raise Stage1S19FormalError("S1_9_WORKER_TIMEOUT")
            lease.heartbeat()
            time.sleep(5.0)
        _assert_process_group_empty(fingerprint, run_token=run_token)
        _assert_no_run_token_processes(run_token)
        _write(fingerprint_path, _with_hash({"schema_version": "stage1-s1-9-child-fingerprint-v1", "status": "EXITED", "command_sha256": hashlib.sha256("\0".join(command).encode("utf-8")).hexdigest(), "fingerprint": fingerprint, "initial_process_group_tree": initial_tree, "residual_run_token_pids": []}))
        if child.returncode != 0:
            raise Stage1S19FormalError(f"S1_9_WORKER_FAILED:{child.returncode}")


def _chart_svg(title: str, values: Sequence[tuple[str, float]]) -> str:
    """Exact, dependency-free CSV projection with visible axes and marks."""

    width, height, left, bottom, top = 760, 260, 70, 215, 45
    maximum = max((abs(value) for _, value in values), default=1.0)
    maximum = max(maximum, 1e-30)
    span = max(len(values) - 1, 1)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="24" font-size="14">{title}</text>', f'<line x1="{left}" y1="{bottom}" x2="730" y2="{bottom}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>']
    points: list[str] = []
    for index, (label, value) in enumerate(values):
        x = left + (650.0 * index / span)
        y = bottom - (160.0 * abs(value) / maximum)
        points.append(f"{x:.3f},{y:.3f}")
        lines.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3" fill="#1769aa"/>')
        lines.append(f'<text x="{x:.3f}" y="238" text-anchor="middle" font-size="8">{label}</text>')
    if points:
        lines.append(f'<polyline fill="none" stroke="#1769aa" points="{" ".join(points)}"/>')
    lines.append(f'<text x="8" y="{top + 5}" font-size="9">{maximum:.6g}</text><text x="24" y="{bottom}" font-size="9">0</text></svg>')
    return "".join(lines) + "\n"


def _verify_chart(csv_file: Path, svg_file: Path, *, title: str) -> bool:
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2 or "," not in lines[0]:
        return False
    values: list[tuple[str, float]] = []
    for row in lines[1:]:
        label, raw = row.rsplit(",", 1)
        try:
            values.append((label, float(raw)))
        except ValueError:
            return False
    return svg_file.read_text(encoding="utf-8") == _chart_svg(title, values)


def _identity_svg(title: str, rows: Sequence[tuple[str, float, float, bool]]) -> str:
    """Render the U single-factor identity as a real unclipped-vs-clipped plot."""

    width, height, left, bottom, top = 760, 280, 75, 225, 45
    maximum = max((max(abs(x), abs(y)) for _, x, y, _ in rows), default=1.0)
    maximum = max(maximum, 1e-30)
    def coordinate(value: float) -> float:
        return left + 630.0 * ((value / maximum) + 1.0) / 2.0
    def vertical(value: float) -> float:
        return bottom - 160.0 * ((value / maximum) + 1.0) / 2.0
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="{left}" y="24" font-size="14">{title}</text>', f'<line x1="{left}" y1="{bottom}" x2="705" y2="{bottom}" stroke="black"/>', f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>', f'<line x1="{coordinate(-maximum):.3f}" y1="{vertical(-maximum):.3f}" x2="{coordinate(maximum):.3f}" y2="{vertical(maximum):.3f}" stroke="#555" stroke-dasharray="4 3"/>', f'<text x="350" y="270" font-size="10">unclipped U score (x)</text>', f'<text x="5" y="{top}" font-size="10">clipped U score (y)</text>']
    for label, x, y, near_zero in rows:
        color = "#999999" if near_zero else "#1769aa"
        lines.append(f'<circle cx="{coordinate(x):.3f}" cy="{vertical(y):.3f}" r="3" fill="{color}"/><text x="{coordinate(x):.3f}" y="{vertical(y) - 5:.3f}" font-size="8">{label}</text>')
    return "".join(lines) + "</svg>\n"


def _verify_identity_chart(csv_file: Path, svg_file: Path, *, title: str) -> bool:
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "coordinate,unclipped,clipped,expected,identity_residual,a_q,ratio_threshold,ratio,eligible_for_ratio":
        return False
    rows: list[tuple[str, float, float, bool]] = []
    for line in lines[1:]:
        fields = line.rsplit(",", 8)
        if len(fields) != 9:
            return False
        try:
            if float(fields[5]) <= 0.0 or float(fields[6]) <= 0.0:
                return False
            rows.append((fields[0], float(fields[1]), float(fields[2]), fields[8] != "true"))
        except ValueError:
            return False
    return svg_file.read_text(encoding="utf-8") == _identity_svg(title, rows)


def _heatmap_svg(title: str, values: Sequence[tuple[str, float, int]]) -> str:
    """A module/layer heatmap, aggregating its member tensor errors by max."""

    width, height, left, top = 760, max(160, 90 + 42 * len(values)), 210, 45
    maximum = max((abs(value) for _, value, _ in values), default=1.0)
    maximum = max(maximum, 1e-30)
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="30" y="24" font-size="14">{title}</text>', '<text x="30" y="42" font-size="10">module/layer (tensor count)</text><text x="220" y="42" font-size="10">max absolute error intensity</text>']
    for index, (label, value, tensor_count) in enumerate(values):
        intensity = int(round(255.0 * abs(value) / maximum))
        color = f"rgb({intensity},0,{255 - intensity})"
        y = top + 42 * index
        lines.append(f'<text x="30" y="{y + 20}" font-size="11">{label} ({tensor_count})</text><rect x="210" y="{y}" width="460" height="30" fill="{color}"/><text x="680" y="{y + 20}" font-size="10">{value:.6g}</text>')
    return "".join(lines) + "</svg>\n"


def _verify_heatmap_chart(csv_file: Path, svg_file: Path, *, title: str) -> bool:
    lines = csv_file.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "module_layer,max_abs_error,tensor_count":
        return False
    values: list[tuple[str, float, int]] = []
    for line in lines[1:]:
        label, raw, raw_count = line.rsplit(",", 2)
        try:
            value, tensor_count = float(raw), int(raw_count)
            if not label or not math.isfinite(value) or tensor_count < 1:
                return False
            values.append((label, value, tensor_count))
        except ValueError:
            return False
    return svg_file.read_text(encoding="utf-8") == _heatmap_svg(title, values)


def _failure_marker(work: Path, error: BaseException, *, phase: str) -> None:
    if work.exists() and not (work / "success.json").exists() and not (work / "failed.json").exists():
        _write(work / "failed.json", _with_hash({"schema_version": "stage1-s1-9-attempt-failure-v1", "status": "FAILED", "phase": phase, "error_type": type(error).__name__, "error": str(error)[:1000], "failed_at": _now(), "success_marker_present": False}))


def _safe_checkpoint_store_reproduction_id(checkpoint_id: object) -> str:
    """Accept a logical CheckpointStore id, never a path-like surrogate."""

    if (
        not isinstance(checkpoint_id, str)
        or _CHECKPOINT_STORE_REPRODUCTION_ID.fullmatch(checkpoint_id) is None
        or "/" in checkpoint_id
        or "\\" in checkpoint_id
        or ".." in checkpoint_id
    ):
        raise Stage1S19FormalError("S1_9_CHECKPOINT_STORE_REPRODUCTION_ID_INVALID")
    return checkpoint_id


def _checkpoint_store_reproduction_index(store_root: Path, checkpoint_id: str) -> dict[str, Any]:
    """Describe the exact immutable CheckpointStore authority needed to reload.

    This intentionally excludes sibling negative-control stores and the
    derived ``latest`` pointer.  A reproduction consumes the explicit commit,
    its tensor bundle, and the file hashes below through ``CheckpointStore``.
    """

    from param_importance_nlp.runtime.checkpoint import CheckpointStore

    checkpoint_id = _safe_checkpoint_store_reproduction_id(checkpoint_id)
    store = CheckpointStore(store_root)
    _, commit = store.load(checkpoint_id)
    commit_path = store.commits / f"{checkpoint_id}.json"
    object_root = store.objects / checkpoint_id
    manifest_path = object_root / "manifest.json"
    if not commit_path.is_file() or not manifest_path.is_file():
        raise Stage1S19FormalError("S1_9_CHECKPOINT_STORE_REPRODUCTION_SOURCE_MISSING")
    bundle_files = {
        path.relative_to(store_root).as_posix(): _sha(path)
        for path in sorted(object_root.rglob("*"))
        if path.is_file()
    }
    if "objects/" + checkpoint_id + "/manifest.json" not in bundle_files:
        raise Stage1S19FormalError("S1_9_CHECKPOINT_STORE_MANIFEST_NOT_LISTED")
    return _with_hash({
        "schema_version": "stage1-s1-9-bf16-checkpoint-store-reproduction-v1",
        "checkpoint_id": checkpoint_id,
        "commit_ref": f"commits/{checkpoint_id}.json",
        "commit_sha256": _sha(commit_path),
        "bundle_manifest_ref": f"objects/{checkpoint_id}/manifest.json",
        "bundle_manifest_sha256": commit.manifest_sha256,
        "bundle_files_sha256": bundle_files,
    })


def _copy_checkpoint_store_reproduction(source_root: Path, target_root: Path, checkpoint_id: str) -> dict[str, Any]:
    """Copy precisely one verified commit and bundle, then read it back."""

    from param_importance_nlp.runtime.checkpoint import CheckpointStore

    checkpoint_id = _safe_checkpoint_store_reproduction_id(checkpoint_id)
    source = CheckpointStore(source_root)
    source.load(checkpoint_id)
    target = CheckpointStore(target_root)
    source_commit = source.commits / f"{checkpoint_id}.json"
    source_object = source.objects / checkpoint_id
    target_commit = target.commits / f"{checkpoint_id}.json"
    target_object = target.objects / checkpoint_id
    if target_commit.exists() or target_object.exists():
        raise Stage1S19FormalError("S1_9_CHECKPOINT_STORE_REPRODUCTION_TARGET_EXISTS")
    shutil.copytree(source_object, target_object)
    shutil.copy2(source_commit, target_commit)
    index = _checkpoint_store_reproduction_index(target_root, checkpoint_id)
    # A CheckpointStore load is the readback: it verifies commit structure,
    # bundle manifest, every tensor byte, and the explicit target lineage.
    target.load(checkpoint_id)
    return index


def _verify_checkpoint_store_reproduction(store_root: Path, index: Mapping[str, Any]) -> bool:
    """Fail closed unless the copied authority exactly matches its manifest."""

    from param_importance_nlp.runtime.checkpoint import CheckpointStore

    try:
        body = dict(index)
        supplied = body.pop("artifact_hash")
        if supplied != _with_hash(body).get("artifact_hash"):
            return False
        required = {"schema_version", "checkpoint_id", "commit_ref", "commit_sha256", "bundle_manifest_ref", "bundle_manifest_sha256", "bundle_files_sha256", "artifact_hash"}
        if set(index) != required or index.get("schema_version") != "stage1-s1-9-bf16-checkpoint-store-reproduction-v1":
            return False
        checkpoint_id = _safe_checkpoint_store_reproduction_id(index.get("checkpoint_id"))
        store = CheckpointStore(store_root)
        _, commit = store.load(checkpoint_id)
        if index.get("commit_ref") != f"commits/{checkpoint_id}.json" or index.get("bundle_manifest_ref") != f"objects/{checkpoint_id}/manifest.json":
            return False
        if index.get("commit_sha256") != _sha(store.commits / f"{checkpoint_id}.json") or index.get("bundle_manifest_sha256") != commit.manifest_sha256:
            return False
        files = index.get("bundle_files_sha256")
        if not isinstance(files, Mapping) or not files:
            return False
        actual = {
            path.relative_to(store_root).as_posix(): _sha(path)
            for path in sorted((store.objects / checkpoint_id).rglob("*"))
            if path.is_file()
        }
        return dict(files) == actual
    except Exception:
        return False


def _exact_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    mapped = _mapping(value, field=field)
    if set(mapped) != expected:
        raise Stage1S19FormalError(f"S1_9_DEEP_SCHEMA_KEYS_INVALID:{field}")
    return mapped


def _validate_t_amp_table_exact_shapes(table: Mapping[str, Any]) -> None:
    """Own S1.9's oneOf semantics while the frozen shared subset lacks it."""

    normalized = {
        "object", "coordinate", "profile", "oracle_hash", "n_q", "n_q_rule", "branch",
        "oracle_norm_inf", "original_unit_max_abs_error", "scaled_max_abs_error",
        "scaled_threshold", "normalized_l2_error", "normalized_l2_limit", "nonfinite_count",
        "passed",
    }
    near_zero = {
        "object", "coordinate", "profile", "oracle_hash", "n_q", "n_q_rule", "branch",
        "near_zero_threshold", "absolute_threshold", "original_unit_max_abs_error",
        "scaled_max_abs_error", "normalized_l2_error", "nonfinite_count", "passed",
    }

    def validate_row(row: object, *, field: str) -> dict[str, Any]:
        mapped = _mapping(row, field=field)
        if mapped.get("branch") == "normalized":
            if set(mapped) != normalized or mapped.get("scaled_max_abs_error") is None or mapped.get("normalized_l2_error") is None:
                raise Stage1S19FormalError(f"S1_9_T_AMP_NORMALIZED_ONEOF_INVALID:{field}")
        elif mapped.get("branch") == "near_zero_absolute":
            if set(mapped) != near_zero or mapped.get("scaled_max_abs_error") is not None or mapped.get("normalized_l2_error") is not None:
                raise Stage1S19FormalError(f"S1_9_T_AMP_NEAR_ZERO_ONEOF_INVALID:{field}")
        else:
            raise Stage1S19FormalError(f"S1_9_T_AMP_BRANCH_INVALID:{field}")
        return mapped

    comparisons, flat_rows = table.get("comparison_objects"), table.get("rows")
    if not isinstance(comparisons, list) or not isinstance(flat_rows, list):
        raise Stage1S19FormalError("S1_9_T_AMP_TABLE_CONTAINER_INVALID")
    projection: list[dict[str, Any]] = []
    for item_index, item in enumerate(comparisons):
        comparison = _mapping(item, field=f"comparison_objects[{item_index}]")
        object_id, oracle_hash, comparison_rows = comparison.get("object"), comparison.get("oracle_hash"), comparison.get("rows")
        if not isinstance(object_id, str) or not isinstance(oracle_hash, str) or not isinstance(comparison_rows, list):
            raise Stage1S19FormalError("S1_9_T_AMP_COMPARISON_FIELDS_INVALID")
        for row_index, item_row in enumerate(comparison_rows):
            row = validate_row(item_row, field=f"comparison_objects[{item_index}].rows[{row_index}]")
            if row.get("object") != object_id or row.get("oracle_hash") != oracle_hash:
                raise Stage1S19FormalError("S1_9_T_AMP_COMPARISON_ROW_BINDING_INVALID")
            projection.append({"section": "t_amp_scale", "case_id": row["object"], "field": row["coordinate"], "actual": dict(row), "reference": {"profile": "T_AMP_SCALE", "oracle_hash": oracle_hash}, "passed": row["passed"]})
    for row_index, flat in enumerate(flat_rows):
        mapped = _mapping(flat, field=f"rows[{row_index}]")
        validate_row(mapped.get("actual"), field=f"rows[{row_index}].actual")
    if flat_rows != projection:
        raise Stage1S19FormalError("S1_9_T_AMP_FLAT_PROJECTION_BINDING_INVALID")


def _validate_gpu_probe(probe: object, approved: list[str], *, field: str) -> None:
    snapshot = _exact_keys(_mapping(probe, field=field), {"selected", "requested_uuid_order", "compute_apps"}, field=field)
    selected, requested, apps = snapshot["selected"], snapshot["requested_uuid_order"], snapshot["compute_apps"]
    if not isinstance(selected, list) or len(selected) != 4 or not isinstance(requested, list) or requested != approved or not isinstance(apps, list):
        raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_PROBE_BINDING_INVALID")
    observed: list[str] = []
    for index, row in enumerate(selected):
        row = _exact_keys(_mapping(row, field=f"{field}.selected[{index}]"), {"physical_index", "uuid", "name", "memory_total_mib", "memory_used_mib", "utilization_percent", "temperature_c", "compute_capability", "recovery_action"}, field=f"{field}.selected[{index}]")
        if type(row["physical_index"]) is not int or not isinstance(row["uuid"], str) or _UUID.fullmatch(row["uuid"]) is None or not isinstance(row["name"], str) or not row["name"] or any(type(row[name]) is not int or row[name] < 0 for name in ("memory_total_mib", "memory_used_mib", "utilization_percent", "temperature_c")) or not isinstance(row["compute_capability"], str) or not row["compute_capability"] or not isinstance(row["recovery_action"], str) or not row["recovery_action"]:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_SELECTED_ROW_INVALID")
        try:
            float(row["compute_capability"])
        except ValueError as error:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_SELECTED_CAPABILITY_INVALID") from error
        observed.append(row["uuid"])
    if observed != approved:
        raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_SELECTED_ORDER_INVALID")
    for index, app in enumerate(apps):
        app = _exact_keys(_mapping(app, field=f"{field}.compute_apps[{index}]"), {"gpu_uuid", "pid", "process_name"}, field=f"{field}.compute_apps[{index}]")
        if not isinstance(app["gpu_uuid"], str) or _UUID.fullmatch(app["gpu_uuid"]) is None or type(app["pid"]) is not int or app["pid"] < 1 or not isinstance(app["process_name"], str) or not app["process_name"]:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_COMPUTE_APP_INVALID")


def _validate_gpu_sample(value: object, approved: list[str], *, field: str) -> bool:
    sample = _mapping(value, field=field)
    base = {"sample_index", "observed_at", "monotonic_elapsed_seconds", "requested_uuid_order", "exact_selected_idle", "consecutive_exact_idle_samples", "transient_observation_count"}
    if "probe_error" in sample:
        sample = _exact_keys(sample, {*base, "probe_error"}, field=field)
        if not isinstance(sample["probe_error"], str) or not sample["probe_error"] or sample["exact_selected_idle"] is not False or sample["consecutive_exact_idle_samples"] != 0:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_PROBE_ERROR_SAMPLE_INVALID")
        return True
    sample = _exact_keys(sample, {*base, "selected", "compute_apps", "violations"}, field=field)
    _validate_gpu_probe({"selected": sample["selected"], "requested_uuid_order": sample["requested_uuid_order"], "compute_apps": sample["compute_apps"]}, approved, field=field)
    if not isinstance(sample["violations"], list) or any(not isinstance(item, str) or not item for item in sample["violations"]) or type(sample["exact_selected_idle"]) is not bool or type(sample["consecutive_exact_idle_samples"]) is not int or sample["consecutive_exact_idle_samples"] < 0:
        raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_SAMPLE_INVALID")
    if sample["exact_selected_idle"] != (not sample["violations"]):
        raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_SAMPLE_IDLE_INVALID")
    return False


def _deep_role_contract(role: str, value: Mapping[str, Any]) -> None:
    """Exact key/cardinality policy for roles whose data-dependent maps are
    difficult to spell compactly in JSON Schema's deliberately small runtime
    validator.  It runs in addition to (never instead of) the schema files.
    """

    if role == "numeric_report":
        _exact_keys(value, {"schema_version", "status", "gate_id", "task_id", "fixture_id", "scope", "producer_commit", "upstream", "implementation_source_sha256", "requirements", "bf16", "ddp_skip", "oracle_hash", "trace_hash", "table_hash", "report_hash"}, field="numeric_report")
        return
    if role == "trace_bundle":
        trace = _exact_keys(value, {"schema_version", "fixture_id", "scale_unscale", "mixed_magnitude", "clip", "scaler", "optimizer", "determinism", "bf16", "ddp_skip", "trace_hash"}, field="trace_bundle")
        scale = _exact_keys(trace["scale_unscale"], {"device", "autocast_dtype", "reference_same_autocast", "frozen_fixture_fp64_oracle_match", "rows", "all_passed", "all_negative_first_order_detected", "all_negative_second_order_detected", "all_statistics_fp32"}, field="trace_bundle.scale_unscale")
        rows = scale["rows"]
        if not isinstance(rows, list) or len(rows) != 3:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_CARDINALITY_INVALID:scale_rows")
        for index, row in enumerate(rows):
            _exact_keys(row, {"loss_scale", "scaled_local_gradients", "manually_unscaled_local_gradients", "actual", "reference_same_autocast", "negative_control_excluded_from_scores", "all_fp32", "passed", "negative_first_order_fields", "negative_second_order_fields", "negative_first_order_scale_detected", "negative_second_order_scale_squared_detected"}, field=f"trace_bundle.scale_unscale.rows[{index}]")
        _exact_keys(trace["clip"], {"global_norm", "post_clip_global_norm", "clip_factor", "u_score_natural_scales", "u_unclipped_score", "u_clipped_score", "u_squared_factor_negative_control", "per_microbatch_clip_negative_control", "raw_unclipped_score", "raw_unclipped_oracle_score", "factor_matches_oracle", "single_factor_identity", "squared_factor_detected", "per_microbatch_clip_detected", "raw_unclipped_matches_clip_pre_oracle", "clip_source", "unbiasedness_claim"}, field="trace_bundle.clip")
        return
    if role == "comparison_table":
        _validate_t_amp_table_exact_shapes(value)
        return
    if role == "bf16_checkpoint_store":
        checkpoint = _exact_keys(value, {"schema_version", "checkpoint_id", "commit_ref", "commit_sha256", "bundle_manifest_ref", "bundle_manifest_sha256", "bundle_files_sha256", "artifact_hash"}, field="bf16_checkpoint_store")
        checkpoint_id = _safe_checkpoint_store_reproduction_id(checkpoint["checkpoint_id"])
        if checkpoint["commit_ref"] != f"commits/{checkpoint_id}.json" or checkpoint["bundle_manifest_ref"] != f"objects/{checkpoint_id}/manifest.json":
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_CHECKPOINT_STORE_REF_INVALID")
        hashes = checkpoint["bundle_files_sha256"]
        expected_prefix = f"objects/{checkpoint_id}/"
        if not isinstance(hashes, Mapping) or not hashes or any(not isinstance(ref, str) or not ref.startswith(expected_prefix) or "\\" in ref or ".." in ref for ref in hashes):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_CHECKPOINT_STORE_FILES_INVALID")
        return
    if role == "upstream_compatibility":
        compatibility = _exact_keys(value, {"schema_version", "status", "s1_7_producer", "s1_8_producer", "consumer_commit", "s1_7_to_consumer_changed_paths", "s1_8_to_consumer_changed_paths", "s1_7_source_attestation", "s1_8_source_dependencies", "s1_8_v8_handoff", "s1_7_affected_dependencies", "s1_8_affected_dependencies", "s1_8_authorized_shared_drift", "authorized_shared_change", "current_source_cpu_clip_replay", "nonproducer_runtime_attestation", "artifact_hash"}, field="upstream_compatibility")
        runtime = _exact_keys(_mapping(compatibility["nonproducer_runtime_attestation"], field="upstream_compatibility.nonproducer_runtime_attestation"), {"affected_paths", "s1_8_source_map_excludes_paths", "s1_7_oracle_training_import_isolated", "checkpoint_group_producer_math_exclusion", "current_source_cpu_replays"}, field="upstream_compatibility.nonproducer_runtime_attestation")
        s18_v8 = _exact_keys(_mapping(compatibility["s1_8_v8_handoff"], field="upstream_compatibility.s1_8_v8_handoff"), {"index_schema_version", "ddp_report_schema_version", "validation_schema_version", "implementation_source_sha256", "reproduction_role_refs", "reproduction_role_sha256", "gpu_quiescence", "replay_schema_version", "comparison_table_schema_version", "array_bundle_schema_version", "array_bundle_route_refs"}, field="upstream_compatibility.s1_8_v8_handoff")
        if runtime["affected_paths"] != runtime["s1_8_source_map_excludes_paths"] or not isinstance(runtime["current_source_cpu_replays"], Mapping) or s18_v8["implementation_source_sha256"] != compatibility["s1_8_source_dependencies"] or s18_v8["index_schema_version"] != "stage1-s1-8-formalization-index-v8" or s18_v8["ddp_report_schema_version"] != "stage1-s1-8-ddp-report-v8" or s18_v8["validation_schema_version"] != "stage1-s1-8-validation-v8" or s18_v8["replay_schema_version"] != "stage1-s1-8-replay-validation-v3" or s18_v8["comparison_table_schema_version"] != "stage1-s1-8-comparison-table-v2" or s18_v8["array_bundle_schema_version"] != "stage1-s1-8-array-bundle-v2" or s18_v8["array_bundle_route_refs"] != _S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_UPSTREAM_COMPATIBILITY_INVALID")
        return
    if role == "gpu_quiescence":
        quiescence = _exact_keys(value, {"schema_version", "status", "phase", "approved_gpu_uuids", "started_at", "minimum_compute_capability", "max_temperature_c", "timeout_seconds", "sample_interval_seconds", "required_consecutive_exact_idle_samples", "max_transient_samples", "transient_observation_count", "operational_timeout_basis", "samples", "final_gpu", "failure_reason", "artifact_hash"}, field="gpu_quiescence")
        approved = quiescence["approved_gpu_uuids"]
        samples = quiescence["samples"]
        if quiescence["schema_version"] != "stage1-s1-9-gpu-quiescence-v3" or quiescence["status"] not in {"PASS", "FAILED"} or quiescence["phase"] not in {"prelease", "post_worker"} or not isinstance(approved, list) or len(approved) != 4 or len(set(approved)) != 4 or any(not isinstance(uuid, str) or _UUID.fullmatch(uuid) is None for uuid in approved) or not isinstance(samples, list) or not samples or len(samples) > _GPU_QUIESCENCE_MAX_SAMPLES or quiescence["timeout_seconds"] != _GPU_QUIESCENCE_TIMEOUT_SECONDS or quiescence["sample_interval_seconds"] != _GPU_QUIESCENCE_POLL_SECONDS or quiescence["required_consecutive_exact_idle_samples"] != _GPU_QUIESCENCE_CONSECUTIVE_SAMPLES or quiescence["max_transient_samples"] != _GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES or quiescence["operational_timeout_basis"] != _GPU_OPERATIONAL_TIMEOUT_BASIS or not _self_hash_valid(quiescence):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_QUIESCENCE_INVALID")
        error_sample = False
        for index, sample in enumerate(samples):
            error_sample = _validate_gpu_sample(sample, approved, field=f"gpu_quiescence.samples[{index}]") or error_sample
        if not isinstance(samples[-1], Mapping) or samples[-1].get("transient_observation_count") != quiescence["transient_observation_count"] or quiescence["transient_observation_count"] > _GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES + 1:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_TRANSIENT_COUNT_INVALID")
        final_gpu = quiescence["final_gpu"]
        if final_gpu is not None:
            _validate_gpu_probe(final_gpu, approved, field="gpu_quiescence.final_gpu")
        if quiescence["status"] == "PASS" and (quiescence["failure_reason"] is not None or len(samples) < 3 or error_sample or final_gpu is None or not isinstance(samples[-1], Mapping) or samples[-1].get("exact_selected_idle") is not True or samples[-1].get("consecutive_exact_idle_samples") != 3):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_QUIESCENCE_PASS_INVALID")
        if quiescence["status"] == "FAILED" and (not isinstance(quiescence["failure_reason"], str) or not quiescence["failure_reason"] or (error_sample and final_gpu is not None)):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_QUIESCENCE_FAIL_INVALID")
        if quiescence["status"] == "FAILED":
            reason = quiescence["failure_reason"]
            if error_sample:
                if not reason.startswith("S1_9_GPU_PROBE_EXCEPTION:"):
                    raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_PROBE_ERROR_REASON_INVALID")
            else:
                last = _mapping(samples[-1], field="gpu_quiescence.last_sample")
                selected = last.get("selected")
                apps = last.get("compute_apps")
                if not isinstance(selected, list) or not isinstance(apps, list):
                    raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_HARD_FAILURE_OBSERVATION_MISSING")
                if reason.startswith("S1_9_GPU_RECOVERY_ACTION_NOT_NONE:"):
                    proven = any(isinstance(row, Mapping) and row.get("recovery_action") != "None" for row in selected)
                elif reason.startswith("S1_9_GPU_PREFLIGHT_HARDWARE_HEALTH_FAILED:"):
                    proven = any(isinstance(row, Mapping) and (not isinstance(row.get("compute_capability"), str) or float(row["compute_capability"]) < quiescence["minimum_compute_capability"] or row.get("temperature_c") > quiescence["max_temperature_c"]) for row in selected)
                elif reason.startswith("S1_9_GPU_COMPUTE_PROCESS_PRESENT:"):
                    proven = any(isinstance(app, Mapping) and app.get("gpu_uuid") in approved for app in apps)
                elif reason == "S1_9_GPU_QUIESCENCE_TIMEOUT":
                    proven = True
                elif reason == "S1_9_GPU_QUIESCENCE_TRANSIENT_LIMIT":
                    proven = quiescence["transient_observation_count"] == _GPU_QUIESCENCE_MAX_TRANSIENT_SAMPLES + 1
                elif reason == "S1_9_GPU_QUIESCENCE_SAMPLE_LIMIT":
                    proven = len(samples) == _GPU_QUIESCENCE_MAX_SAMPLES
                else:
                    proven = False
                if not proven:
                    raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_HARD_FAILURE_EVIDENCE_INVALID")
        return
    if role == "gpu_prelease":
        prelease = _exact_keys(value, {"schema_version", "status", "approved_gpu_uuids", "quiescence", "artifact_hash"}, field="gpu_prelease")
        approved = prelease["approved_gpu_uuids"]
        if prelease["schema_version"] != "stage1-s1-9-gpu-prelease-v3" or prelease["status"] not in {"PASS", "FAILED"} or not isinstance(approved, list) or len(approved) != 4 or len(set(approved)) != 4 or any(not isinstance(uuid, str) or _UUID.fullmatch(uuid) is None for uuid in approved) or not _self_hash_valid(prelease):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_PRELEASE_INVALID")
        _deep_role_contract("gpu_quiescence", _mapping(prelease["quiescence"], field="gpu_prelease.quiescence"))
        if prelease["quiescence"].get("phase") != "prelease" or prelease["quiescence"].get("approved_gpu_uuids") != approved or prelease["status"] != prelease["quiescence"].get("status"):
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_GPU_PRELEASE_STATUS_BINDING_INVALID")
        return
    if role == "index":
        index = _exact_keys(value, {"schema_version", "status", "gate_id", "task_id", "fixture_id", "generator_git_commit", "consumer_git_commit", "git_branch", "checked_at", "s1_7_handoff", "s1_8_handoff", "role_refs", "role_sha256", "reproduction_role_refs", "reproduction_role_sha256", "gate_artifact_hash", "csv_sha256", "svg_sha256", "validation_ref", "validation_sha256", "replay_ref", "replay_sha256", "replay_hash", "next_task_ids", "artifact_hash"}, field="index")
        for map_name in ("role_refs", "role_sha256", "reproduction_role_refs", "reproduction_role_sha256", "csv_sha256", "svg_sha256"):
            if not isinstance(index[map_name], Mapping) or not index[map_name]:
                raise Stage1S19FormalError(f"S1_9_DEEP_SCHEMA_MAP_INVALID:index.{map_name}")
        if index["schema_version"] != "stage1-s1-9-formalization-index-v8" or index["next_task_ids"] != ["stage1.10_checkpoint_resume_and_artifacts"]:
            raise Stage1S19FormalError("S1_9_DEEP_SCHEMA_NEXT_TASK_INVALID")


def _validate_s1_9_schemas(repository: Path, values: Mapping[str, Mapping[str, Any]]) -> None:
    """Run the repository's strict schema subset; no loose JSON is accepted."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json

    validator_path = repository / "ops" / "stage1" / "formalize_s1_6.py"
    spec = importlib.util.spec_from_file_location("_s19_schema_subset", validator_path)
    if spec is None or spec.loader is None:
        raise Stage1S19FormalError("S1_9_SCHEMA_VALIDATOR_LOAD_FAILED")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    registry: dict[str, Mapping[str, Any]] = {}
    schema_paths = [*sorted((repository / "schemas" / "stage1").glob("s1-9-*.json")), repository / "schemas" / "stage1" / "s1-8-formalization-index-v1.json", repository / "schemas" / "stage1" / "s1-8-formalization-index-v8.json", repository / "schemas" / "stage1" / "s1-8-ddp-report-v8.json", repository / "schemas" / "stage1" / "s1-8-validation-v8.json", repository / "schemas" / "stage1" / "s1-8-gpu-quiescence-v4.json", repository / "schemas" / "stage1" / "s1-8-replay-validation-v3.json", repository / "schemas" / "stage1" / "s1-8-comparison-table-v2.json", repository / "schemas" / "stage1" / "s1-8-array-bundle-v2.json"]
    for path in schema_paths:
        loaded = loads_strict_json(path.read_bytes())
        if not isinstance(loaded, Mapping) or not isinstance(loaded.get("$id"), str):
            raise Stage1S19FormalError(f"S1_9_SCHEMA_INVALID:{path.name}")
        registry[path.name] = loaded; registry[str(loaded["$id"])] = loaded
    files = {"numeric_report": "s1-9-numeric-report-v1.json", "oracle_bundle": "s1-9-oracle-bundle-v1.json", "trace_bundle": "s1-9-trace-bundle-v1.json", "comparison_table": "s1-9-comparison-table-v1.json", "gate_record": "s1-9-gate-record-v1.json", "replay": "s1-9-replay-validation-v1.json", "validation": "s1-9-validation-v1.json", "index": "s1-9-formalization-index-v8.json", "single_worker": "s1-9-single-bf16-worker-v1.json", "ddp_worker": "s1-9-ddp-skip-worker-v1.json", "bf16_checkpoint_store": "s1-9-bf16-checkpoint-store-reproduction-v1.json", "upstream_compatibility": "s1-9-upstream-compatibility-v7.json", "gpu_quiescence": "s1-9-gpu-quiescence-v3.json", "gpu_prelease": "s1-9-gpu-prelease-v3.json"}
    for role, value in values.items():
        schema = registry.get(files.get(role, ""))
        if schema is None:
            raise Stage1S19FormalError(f"S1_9_SCHEMA_ROLE_UNKNOWN:{role}")
        try:
            module._validate_schema(value, schema, registry, document=schema, path=role)
            _deep_role_contract(role, value)
        except Exception as error:
            raise Stage1S19FormalError(f"S1_9_SCHEMA_VALIDATION_FAILED:{role}") from error


def _negative_schema_controls(repository: Path, report: Mapping[str, Any], trace: Mapping[str, Any], worker: Mapping[str, Any], comparison_table: Mapping[str, Any]) -> bool:
    cases = []
    unexpected = dict(report); unexpected["unexpected"] = True; cases.append(("numeric_report", unexpected))
    trace_with_nested_drift = dict(trace); clip = dict(_mapping(trace["clip"], field="negative.trace.clip")); clip["unexpected"] = True; trace_with_nested_drift["clip"] = clip; cases.append(("trace_bundle", trace_with_nested_drift))
    scaler = dict(_mapping(trace["scaler"], field="negative.trace.scaler"))
    engine = dict(_mapping(scaler["engine_finite_skip_finite"], field="negative.trace.engine"))
    pre = dict(_mapping(engine["skip_pre_state"], field="negative.trace.engine.pre"))
    cursor = dict(_mapping(pre["cursor"], field="negative.trace.engine.cursor")); cursor["unexpected"] = True; pre["cursor"] = cursor; engine["skip_pre_state"] = pre; scaler["engine_finite_skip_finite"] = engine
    trace_with_deep_unknown = dict(trace); trace_with_deep_unknown["scaler"] = scaler; cases.append(("trace_bundle", trace_with_deep_unknown))
    scaler = dict(_mapping(trace["scaler"], field="negative.trace.scaler_missing"))
    engine = dict(_mapping(scaler["engine_finite_skip_finite"], field="negative.trace.engine_missing"))
    snapshots = list(engine["attempt_state_snapshots"]); snapshots.pop(); engine["attempt_state_snapshots"] = snapshots; scaler["engine_finite_skip_finite"] = engine
    trace_with_cardinality_drift = dict(trace); trace_with_cardinality_drift["scaler"] = scaler; cases.append(("trace_bundle", trace_with_cardinality_drift))
    missing = dict(worker); missing.pop("run_token", None); cases.append(("ddp_worker", missing))
    # Rehashed artifacts are checked by the runtime replay; formal schema
    # validation separately proves the exclusive branch shapes themselves.
    table_cross_branch = json.loads(json.dumps(comparison_table))
    near_zero = next(item for item in table_cross_branch["comparison_objects"] if item["object"] == "t_amp_exact_zero_contract")
    near_zero["rows"][0]["oracle_norm_inf"] = 1.0
    cases.append(("comparison_table", table_cross_branch))
    table_normalized_cross = json.loads(json.dumps(comparison_table))
    normalized = next(row for item in table_normalized_cross["comparison_objects"] for row in item["rows"] if row["branch"] == "normalized")
    normalized["near_zero_threshold"] = 1e-6
    cases.append(("comparison_table", table_normalized_cross))
    for role, value in cases:
        try:
            _validate_s1_9_schemas(repository, {role: value})
        except Stage1S19FormalError:
            continue
        return False
    return True


def execute(*, repository: str | Path, data_root: str | Path, s1_7_index_ref: str, s1_8_index_ref: str, s1_8_binding: Mapping[str, str], gpu_capability_ref: str, capability_binding: Mapping[str, str], approved_gpu_uuids: tuple[str, ...], attempt_id: str, lease_owner: str, timeout_seconds: int = 1800, minimum_compute_capability: float = 8.0, max_temperature_c: int = 85) -> dict[str, str]:
    """Publish G1-NUMERIC only after independently owned lease/run/replay."""

    repository_root, root = Path(repository).resolve(strict=True), Path(data_root).resolve(strict=True)
    if str(repository_root / "src") not in sys.path:
        sys.path.insert(0, str(repository_root / "src"))
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    from param_importance_nlp.runtime.checkpoint import CheckpointStore
    from param_importance_nlp.runtime.operations import GpuLeaseIdentity, ProjectGpuLease
    from param_importance_nlp.stage1_precision import (REQUIREMENT_KEYS, T_AMP_SCALE_PROFILE, build_stage1_s19_evidence, replay_stage1_s19_evidence, validate_s1_7_handoff, validate_s1_8_handoff, validate_stage1_s19_evidence)
    from param_importance_nlp.stage1_precision_oracle import load_stage1_s19_fixture

    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None or _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S19FormalError("S1_9_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if not _ATTEMPT.fullmatch(attempt_id) or not lease_owner or len(approved_gpu_uuids) != 4 or len(set(approved_gpu_uuids)) != 4 or any(_UUID.fullmatch(item) is None for item in approved_gpu_uuids):
        raise Stage1S19FormalError("S1_9_ATTEMPT_OWNER_OR_DDP_UUIDS_INVALID")
    target = root / "evidence" / "stage1" / "s1-9-formal" / commit / attempt_id
    work = root / "tmp" / "stage1-s1-9" / commit / attempt_id
    if target.exists() or work.exists():
        raise Stage1S19FormalError("S1_9_ATTEMPT_ALREADY_EXISTS")
    work.mkdir(parents=True)
    lease = None
    released = False
    staging: Path | None = None
    phase = "preflight"
    try:
        fixture = load_stage1_s19_fixture(repository_root)
        upstream_s17 = validate_s1_7_handoff(root, s1_7_index_ref)
        upstream_s18 = validate_s1_8_handoff(root, s1_8_index_ref, expected_binding=s1_8_binding)
        upstream_compatibility = _upstream_compatibility_attestation(repository_root, root, s1_7_ref=s1_7_index_ref, s1_7=upstream_s17, s1_8_ref=s1_8_index_ref, s1_8=upstream_s18, replay_root=work / "upstream-cpu-replay")
        _write(work / "upstream-compatibility.json", upstream_compatibility)
        capability = _capability(root, gpu_capability_ref, capability_binding, approved_gpu_uuids)
        # This bounded loop is before the lease constructor.  Both PASS and
        # every failure variant are schema-checked and durably written.
        prelease_quiescence = _gpu_quiescence(approved_gpu_uuids, phase="prelease", minimum_compute_capability=minimum_compute_capability, max_temperature_c=max_temperature_c)
        _validate_s1_9_schemas(repository_root, {"gpu_quiescence": prelease_quiescence})
        prelease_gpu = _gpu_prelease_evidence(approved_gpu_uuids, prelease_quiescence)
        _validate_s1_9_schemas(repository_root, {"gpu_prelease": prelease_gpu})
        _write(work / "prelease-gpu.json", prelease_gpu)
        if prelease_gpu["status"] != "PASS":
            raise Stage1S19FormalError("S1_9_GPU_PRELEASE_FAILED:" + str(prelease_quiescence["failure_reason"]))
        resource_budget = _resource_budget(work)
        _write(work / "preflight.json", _with_hash({"schema_version": "stage1-s1-9-preflight-v1", "status": "PASS", "fixture_hash": fixture["fixture_hash"], "s1_7_handoff": upstream_s17, "s1_8_handoff": upstream_s18, "upstream_compatibility_hash": upstream_compatibility["artifact_hash"], "capability": capability, "gpu": prelease_quiescence, "resource_budget": resource_budget}))
        config_hash = _canonical({"task_id": TASK_ID, "commit": commit, "s1_7": upstream_s17, "s1_8": upstream_s18, "upstream_compatibility": upstream_compatibility["artifact_hash"], "capability": capability, "approved_gpu_uuids": list(approved_gpu_uuids), "determinism_policy": _DETERMINISM_ENV, "timeout_seconds": timeout_seconds})
        identity = GpuLeaseIdentity(run_id=f"s19-{attempt_id}", lease_id=f"s19-{attempt_id}", gpu_uuids=approved_gpu_uuids, owner=lease_owner, config_hash=config_hash, environment_hash=str(prelease_quiescence["artifact_hash"]))
        lease = ProjectGpuLease(root, identity)
        lease.acquire(); lease.heartbeat()
        run_token = hashlib.sha256(_canonical({"execution_commit": commit, "config_hash": config_hash, "approved_gpu_uuids": list(approved_gpu_uuids), "attempt_id": attempt_id}).encode("ascii")).hexdigest()
        environment = dict(os.environ)
        environment.update({"PYTHONPATH": str(repository_root / "src") + os.pathsep + environment.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": ",".join(approved_gpu_uuids), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "S1_9_RUN_TOKEN": run_token})
        determinism_environment = _validate_child_environment(environment, approved=approved_gpu_uuids, run_token=run_token)
        _write(work / "attempt-start.json", _with_hash({"schema_version": "stage1-s1-9-attempt-start-v1", "status": "STARTED", "task_id": TASK_ID, "execution_commit": commit, "plan_hash": config_hash, "run_token": run_token, "expected_world_size": 4, "approved_gpu_uuids": list(approved_gpu_uuids), "cuda_visible_devices": {"single": approved_gpu_uuids[0], "ddp": ",".join(approved_gpu_uuids)}, "determinism_environment": determinism_environment, "output_roles": {"single": "single-bf16.json", "ddp": "ddp-skip.json", "checkpoint_store": "bf16-resume-store-index.json"}, "resource_budget": resource_budget}))
        # The formalizer remains CPU-only.  Every CUDA context belongs to a
        # UUID-isolated child so post-worker occupancy can stay strict.
        phase = "single_bf16"
        single_environment = dict(environment); single_environment["CUDA_VISIBLE_DEVICES"] = approved_gpu_uuids[0]
        single_path = work / "single-bf16.json"
        _run_with_heartbeats([sys.executable, str(repository_root / "ops" / "stage1" / "run_s1_9_single_bf16_worker.py"), "--repository", str(repository_root), "--output", str(single_path), "--checkpoint-dir", str(work / "bf16-resume"), "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuid", approved_gpu_uuids[0]], env=single_environment, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "single.stdout.txt", stderr=work / "single.stderr.txt", fingerprint_path=work / "single-child-fingerprint.json")
        single = _mapping(load_canonical_json(single_path), field="single_bf16")
        _validate_s1_9_schemas(repository_root, {"single_worker": single})
        if single.get("schema_version") != "stage1-s1-9-single-bf16-worker-v1" or single.get("status") != "PASS" or single.get("execution_commit") != commit or single.get("run_token") != run_token or single.get("approved_gpu_uuid") != approved_gpu_uuids[0] or single.get("cuda_visible_devices") != approved_gpu_uuids[0] or not _worker_environment_summary_valid(single.get("environment_summary"), cuda_visible_devices=approved_gpu_uuids[0], local_rank=0, local_gpu_uuid=approved_gpu_uuids[0]) or not isinstance(single.get("observation"), Mapping) or not _self_hash_valid(single):
            raise Stage1S19FormalError("S1_9_SINGLE_BF16_WORKER_INVALID")
        bf16_run = _mapping(single["observation"], field="single.observation")
        bf16_observation = _mapping(bf16_run.get("observation"), field="bf16_run.observation")
        resume = _mapping(bf16_run.get("resume"), field="bf16.resume")
        checkpoint_id = resume.get("checkpoint_id")
        checkpoint_root = work / "bf16-resume"
        checkpoint = checkpoint_root / "commits" / f"{checkpoint_id}.json" if isinstance(checkpoint_id, str) else checkpoint_root / "commits" / "invalid.json"
        expected_checkpoint_roles = ["checkpoint_ids", "cursor", "importance", "importance_trajectory_points", "model", "optimizer", "optimizer_contract_hash", "records", "registry_hash", "rng", "run_spec_hash", "runtime_layout_hash", "scaler", "scheduler", "schema_version", "training_state"]
        expected_state_hash_fields = {
            "model", "optimizer", "scheduler", "scaler", "cursor", "training_state",
            "importance", "records", "importance_trajectory_points", "checkpoint_ids", "rng",
            "run_spec_hash", "registry_hash", "optimizer_contract_hash", "runtime_layout_hash",
            "importance_view_signed", "importance_view_positive", "importance_view_negative_mass",
            "importance_view_absolute", "importance_view_raw", "importance_view_raw_clipped",
            "importance_view_data_movement", "importance_view_net_data_movement",
            "importance_view_total_movement", "importance_view_total_endpoint_movement",
            "importance_view_weight_decay_movement", "importance_view_net_weight_decay_movement",
            "importance_view_actual_update_raw_importance",
            "importance_view_actual_update_raw_importance_available", "importance_view_magnitude",
            "importance_view_attempted_steps",
        }
        continuous_state_hashes = resume.get("continuous_state_field_hashes")
        restored_state_hashes = resume.get("restored_state_field_hashes")
        try:
            stored_payload, stored_commit = CheckpointStore(checkpoint_root).load(checkpoint_id) if isinstance(checkpoint_id, str) else (None, None)
        except Exception:
            stored_payload, stored_commit = None, None
        checkpoint_checks = {
            "file_present": checkpoint.is_file(),
            "declared_sha256_matches": checkpoint.is_file() and resume.get("checkpoint_sha256") == _sha(checkpoint),
            "declared_size_matches": checkpoint.is_file() and resume.get("checkpoint_size_bytes") == checkpoint.stat().st_size,
            "checkpoint_store_loads_committed_artifact": isinstance(stored_payload, Mapping) and stored_commit is not None and getattr(stored_commit, "checkpoint_id", None) == checkpoint_id and getattr(stored_commit, "manifest_sha256", None) == resume.get("checkpoint_manifest_sha256"),
            "checkpoint_store_payload_roles_exact": isinstance(stored_payload, Mapping) and sorted(stored_payload) == expected_checkpoint_roles,
            "role_fields_exact": resume.get("checkpoint_role_fields") == expected_checkpoint_roles,
            "state_hash_field_sets_exact": isinstance(continuous_state_hashes, Mapping) and isinstance(restored_state_hashes, Mapping) and set(continuous_state_hashes) == expected_state_hash_fields and set(restored_state_hashes) == expected_state_hash_fields,
            "state_hashes_exact_after_next_step": continuous_state_hashes == restored_state_hashes,
            "engine_resume_negative_controls": all(resume.get(key) is True for key in ("next_step_exact", "next_step_independent_engines", "accumulator_public_view_set_exact", "data_cursor_restored", "corruption_detected", "omission_rejected", "record_order_rejected")),
        }
        if not all(checkpoint_checks.values()):
            raise Stage1S19FormalError("S1_9_BF16_CHECKPOINT_BINDING_INVALID")
        lease.heartbeat()
        phase = "nccl_ddp_skip"
        ddp_path = work / "ddp-skip.json"
        _run_with_heartbeats([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4", "--rdzv-backend", "c10d", "--rdzv-id", f"s19-{run_token}", "--rdzv-endpoint", "127.0.0.1:0", str(repository_root / "ops" / "stage1" / "run_s1_9_ddp_skip_worker.py"), "--output", str(ddp_path), "--inject-rank", "0", "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuids", ",".join(approved_gpu_uuids)], env=environment, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "ddp.stdout.txt", stderr=work / "ddp.stderr.txt", fingerprint_path=work / "ddp-child-fingerprint.json")
        ddp = _mapping(load_canonical_json(ddp_path), field="ddp_skip")
        _validate_s1_9_schemas(repository_root, {"ddp_worker": ddp})
        ddp_environment = ddp.get("per_rank_environment_summary")
        ddp_environment_valid = (
            _worker_environment_summary_valid(ddp.get("environment_summary"), cuda_visible_devices=",".join(approved_gpu_uuids), local_rank=0, local_gpu_uuid=approved_gpu_uuids[0])
            and isinstance(ddp_environment, list)
            and len(ddp_environment) == 4
            and all(_worker_environment_summary_valid(item, cuda_visible_devices=",".join(approved_gpu_uuids), local_rank=rank, local_gpu_uuid=approved_gpu_uuids[rank]) for rank, item in enumerate(ddp_environment))
        )
        if ddp.get("status") != "PASS" or ddp.get("execution_commit") != commit or ddp.get("run_token") != run_token or ddp.get("world_size") != 4 or ddp.get("approved_gpu_uuids") != list(approved_gpu_uuids) or ddp.get("cuda_visible_devices") != ",".join(approved_gpu_uuids) or ddp.get("rank_to_uuid") != {str(index): uuid for index, uuid in enumerate(approved_gpu_uuids)} or not ddp_environment_valid or not _self_hash_valid(ddp) or not isinstance(ddp.get("checks"), Mapping) or not all(isinstance(value, bool) and value for value in ddp["checks"].values()):
            raise Stage1S19FormalError("S1_9_DDP_SKIP_GATE_FAILED")
        lease.heartbeat()
        phase = "numeric_evidence"
        evidence = build_stage1_s19_evidence(repository_root, producer_commit=commit, scope="formal_server_gpu_single_and_ddp_skip", upstream_evidence={"s1_7": upstream_s17, "s1_8": upstream_s18, "gpu_capability": capability}, bf16_observation=bf16_run, ddp_skip_observation=ddp, device="cpu")
        validate_stage1_s19_evidence(evidence, source_root=repository_root)
        replay = replay_stage1_s19_evidence(evidence, source_root=repository_root, device="cpu")
        _validate_s1_9_schemas(repository_root, {**evidence, "replay": replay})
        schema_negative_controls = _negative_schema_controls(repository_root, evidence["numeric_report"], evidence["trace_bundle"], ddp, evidence["comparison_table"])
        if not schema_negative_controls:
            raise Stage1S19FormalError("S1_9_SCHEMA_NEGATIVE_CONTROL_FAILED")
        if evidence["numeric_report"].get("status") != "PASS" or replay.get("status") != "PASS":
            raise Stage1S19FormalError("S1_9_NUMERIC_OR_REPLAY_NOT_PASS")
        post_worker = _gpu_quiescence(approved_gpu_uuids, phase="post_worker", minimum_compute_capability=minimum_compute_capability, max_temperature_c=max_temperature_c)
        # Failure is an evidence-bearing, fail-closed outcome too; schema
        # validate it before making the canonical record durable.
        _validate_s1_9_schemas(repository_root, {"gpu_quiescence": post_worker})
        _write(work / "post-worker-quiescence.json", post_worker)
        if post_worker.get("status") != "PASS":
            raise Stage1S19FormalError("S1_9_GPU_POST_WORKER_QUIESCENCE_FAILED:" + str(post_worker.get("failure_reason")))
        history = _release_lease_verified(lease, outcome="GPU_PHASE_SUCCESS"); released = True; lease = None
        shutil.copy2(history, work / "lease-history.json")
        role_files = {"numeric_report": "numeric-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-numeric-record.json"}
        for role, filename in role_files.items():
            _write(work / filename, evidence[role])
        _write(work / "replay-validation.json", replay)
        comparison_objects = evidence["comparison_table"].get("comparison_objects")
        if not isinstance(comparison_objects, list):
            raise Stage1S19FormalError("S1_9_COMPARISON_OBJECTS_INVALID")
        rows = [row for object_row in comparison_objects if isinstance(object_row, Mapping) for row in object_row.get("rows", []) if isinstance(row, Mapping)]
        csv_rows = [{"object": str(row["object"]), "coordinate": str(row["coordinate"]), "original_unit_max_abs_error": float(row["original_unit_max_abs_error"]), "passed": bool(row["passed"])} for row in rows]
        def write_chart(stem: str, title: str, header: str, records: Sequence[tuple[str, float]]) -> tuple[Path, Path]:
            csv_file, svg_file = work / f"{stem}.csv", work / f"{stem}.svg"
            csv_file.write_text(header + "\n" + "".join(f"{label},{value:.17g}\n" for label, value in records), encoding="utf-8", newline="\n")
            svg_file.write_text(_chart_svg(title, records), encoding="utf-8", newline="\n")
            return csv_file, svg_file

        chart_files: list[tuple[Path, Path, str]] = []
        chart_files.append((*write_chart("t-amp-scale", "S1.9 T_AMP_SCALE error", "object_coordinate,original_unit_max_abs_error", [(f'{row["object"]}/{row["coordinate"]}', row["original_unit_max_abs_error"]) for row in csv_rows]), "S1.9 T_AMP_SCALE error"))
        clip = evidence["trace_bundle"]["clip"]
        chart_files.append((*write_chart("clip-norm-factor", "Clip pre/post norm and factor", "metric,value", [("pre_clip_global_norm", float(clip["global_norm"])), ("clip_factor", float(clip["clip_factor"])), ("post_clip_global_norm", float(clip["post_clip_global_norm"]))]), "Clip pre/post norm and factor"))
        u_unclipped, u_clipped = clip["u_unclipped_score"], clip["u_clipped_score"]
        if not isinstance(u_unclipped, Mapping) or not isinstance(u_clipped, Mapping):
            raise Stage1S19FormalError("S1_9_CHART_U_IDENTITY_INVALID")
        u_natural_scales = _mapping(clip.get("u_score_natural_scales"), field="clip.u_score_natural_scales")
        flattened_u = [(name, f"{name}-{index}", float(u_unclipped[name][index]), float(u_clipped[name][index])) for name in sorted(u_unclipped) for index in range(len(u_unclipped[name]))]
        identity_rows = []
        for name, label, unclipped, clipped_value in flattened_u:
            registered = _mapping(u_natural_scales.get(name), field=f"clip.u_score_natural_scales.{name}")
            a_q = float(registered.get("n_q", float("nan")))
            threshold = 100.0 * float(T_AMP_SCALE_PROFILE["atol"]) * a_q
            if not math.isfinite(a_q) or a_q <= 0.0 or not math.isfinite(threshold):
                raise Stage1S19FormalError("S1_9_CHART_U_NATURAL_SCALE_INVALID")
            expected = unclipped * float(clip["clip_factor"])
            eligible_for_ratio = abs(unclipped) >= threshold
            ratio = clipped_value / expected if eligible_for_ratio and expected != 0.0 else float("nan")
            identity_rows.append((label, unclipped, clipped_value, expected, clipped_value - expected, a_q, threshold, ratio, eligible_for_ratio))
        identity_csv, identity_svg = work / "u-single-factor-identity.csv", work / "u-single-factor-identity.svg"
        identity_csv.write_text("coordinate,unclipped,clipped,expected,identity_residual,a_q,ratio_threshold,ratio,eligible_for_ratio\n" + "".join(f"{label},{unclipped:.17g},{clipped_value:.17g},{expected:.17g},{residual:.17g},{a_q:.17g},{threshold:.17g},{ratio:.17g},{str(eligible).lower()}\n" for label, unclipped, clipped_value, expected, residual, a_q, threshold, ratio, eligible in identity_rows), encoding="utf-8", newline="\n")
        identity_title = f"U identity: y = {float(clip['clip_factor']):.6g} x (all coordinates)"
        identity_svg.write_text(_identity_svg(identity_title, [(label, unclipped, clipped_value, not eligible) for label, unclipped, clipped_value, _, _, _, _, _, eligible in identity_rows]), encoding="utf-8", newline="\n")
        chart_files.append((identity_csv, identity_svg, identity_title))
        ratio_records = [(label, ratio) for label, _, _, _, _, _, _, ratio, eligible in identity_rows if eligible]
        if not ratio_records:
            raise Stage1S19FormalError("S1_9_CHART_U_RATIO_FILTER_EMPTY")
        chart_files.append((*write_chart("u-single-factor-ratio-diagnostic", "U clipped/(factor*unclipped) after registered near-zero filter", "coordinate,ratio", ratio_records), "U clipped/(factor*unclipped) after registered near-zero filter"))
        quality = bf16_observation["mean_gradient_quality"]["per_tensor"]
        if not isinstance(quality, Mapping):
            raise Stage1S19FormalError("S1_9_CHART_BF16_QUALITY_INVALID")
        # The evidence is per-tensor, but the required BF16 diagnostic is at
        # module/layer scope.  Keep its membership cardinality in the CSV so
        # a one-layer fixture cannot silently masquerade as a tensor heatmap.
        module_errors: dict[str, list[float]] = {}
        for name, value in sorted(quality.items()):
            tensor_name = str(name)
            module = tensor_name.rpartition(".")[0] or "<root>"
            module_errors.setdefault(module, []).append(float(_mapping(value, field="bf16.quality")["max_abs_error"]))
        heatmap_values = [(module, max(errors), len(errors)) for module, errors in sorted(module_errors.items())]
        heatmap_csv, heatmap_svg = work / "bf16-fp32-heatmap.csv", work / "bf16-fp32-heatmap.svg"
        heatmap_title = "BF16 vs FP32 module/layer heatmap (max member error)"
        heatmap_csv.write_text("module_layer,max_abs_error,tensor_count\n" + "".join(f"{name},{value:.17g},{count}\n" for name, value, count in heatmap_values), encoding="utf-8", newline="\n")
        heatmap_svg.write_text(_heatmap_svg(heatmap_title, heatmap_values), encoding="utf-8", newline="\n")
        chart_files.append((heatmap_csv, heatmap_svg, heatmap_title))
        skip_checks = evidence["trace_bundle"]["scaler"]["engine_finite_skip_finite"]["skip_full_state_checks"]
        if not isinstance(skip_checks, Mapping):
            raise Stage1S19FormalError("S1_9_CHART_SKIP_STATE_INVALID")
        chart_files.append((*write_chart("skip-zero-difference", "Skip zero-difference state checks", "check,zero_difference", [(str(name), 0.0 if bool(value) else 1.0) for name, value in sorted(skip_checks.items())]), "Skip zero-difference state checks"))
        generic_charts = [item for item in chart_files if item[0].name not in {"u-single-factor-identity.csv", "bf16-fp32-heatmap.csv"}]
        if not all(_verify_chart(csv_file, svg_file, title=title) for csv_file, svg_file, title in generic_charts) or not _verify_identity_chart(identity_csv, identity_svg, title=identity_title) or not _verify_heatmap_chart(heatmap_csv, heatmap_svg, title=heatmap_title):
            raise Stage1S19FormalError("S1_9_CHART_CSV_SVG_PROJECTION_INVALID")
        csv_hashes = {csv_file.name: _sha(csv_file) for csv_file, _, _ in chart_files}
        svg_hashes = {svg_file.name: _sha(svg_file) for _, svg_file, _ in chart_files}
        role_sha = {role: _sha(work / filename) for role, filename in role_files.items()}
        formal_checks = {"bf16_worker_self_hash": _self_hash_valid(single) and _worker_environment_summary_valid(single.get("environment_summary"), cuda_visible_devices=approved_gpu_uuids[0], local_rank=0, local_gpu_uuid=approved_gpu_uuids[0]), "bf16_checkpoint_binding_and_tamper_negative_control": all(checkpoint_checks.values()), "ddp_worker_self_hash_and_all_direct_checks": _self_hash_valid(ddp) and ddp_environment_valid and all(bool(value) for value in ddp["checks"].values()), "post_worker_gpu_preflight_clean": post_worker.get("status") == "PASS", "strict_schema_missing_and_unknown_negative_controls": schema_negative_controls}
        direct_checks = [{"check_id": key, "status": "PASS" if value else "FAIL", "detail": "machine-checked formal S1.9 requirement"} for key, value in evidence["numeric_report"]["requirements"].items()] + [{"check_id": key, "status": "PASS" if value else "FAIL", "detail": "direct formal worker, checkpoint, schema, or resource observation"} for key, value in formal_checks.items()]
        if [row["check_id"] for row in direct_checks] != [*REQUIREMENT_KEYS, *formal_checks]:
            raise Stage1S19FormalError("S1_9_DIRECT_CHECK_ORDER_OR_KEYSET_INVALID")
        if any(row["status"] != "PASS" for row in direct_checks):
            raise Stage1S19FormalError("S1_9_DIRECT_REQUIREMENT_FAILED")
        validation = _with_hash({"schema_version": "stage1-s1-9-validation-v1", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "execution_scope": "formal_server_gpu_single_and_ddp_skip", "fixture_id": FIXTURE_ID, "producer_commit": commit, "consumer_commit": commit, "upstream": {"s1_7": upstream_s17, "s1_8": upstream_s18}, "regression": {"bf16": bf16_run["determinism"], "ddp_world_size": ddp["world_size"], "kernel_allowlist": bf16_run["determinism"]["allowed_nondeterministic_kernel_classes"], "environment": {"single": single["environment_summary"], "ddp_rank0": ddp["environment_summary"], "ddp_all_ranks": ddp_environment}}, "direct_checks": direct_checks, "role_sha256": role_sha, "csv_sha256": csv_hashes, "svg_sha256": svg_hashes, "replay_sha256": _sha(work / "replay-validation.json"), "replay_hash": replay["replay_hash"]})
        _validate_s1_9_schemas(repository_root, {"validation": validation})
        _write(work / "validation.json", validation)
        phase = "publish"
        staging = target.parent / f".{attempt_id}.publishing"
        if staging.exists():
            raise Stage1S19FormalError("S1_9_PUBLISH_STAGING_COLLISION")
        staging.mkdir(parents=True)
        published = set(role_files.values()) | {"replay-validation.json", "validation.json", "attempt-start.json", "upstream-compatibility.json", "prelease-gpu.json", "preflight.json", "post-worker-quiescence.json", "lease-history.json", "single-bf16.json", "single.stdout.txt", "single.stderr.txt", "single-child-fingerprint.json", "ddp-skip.json", "ddp.stdout.txt", "ddp.stderr.txt", "ddp-child-fingerprint.json"} | set(csv_hashes) | set(svg_hashes)
        for name in sorted(published):
            shutil.copy2(work / name, staging / name)
        checkpoint_store_index = _copy_checkpoint_store_reproduction(checkpoint_root, staging / "bf16-resume-store", str(checkpoint_id))
        # The generated candidate is a formal reproduction role in its own
        # right: schema-check it before it ever reaches staging.
        _validate_s1_9_schemas(repository_root, {"bf16_checkpoint_store": checkpoint_store_index})
        _write(staging / "bf16-resume-store-index.json", checkpoint_store_index)
        reproduction = {"attempt_start": "attempt-start.json", "upstream_compatibility": "upstream-compatibility.json", "preflight": "preflight.json", "prelease_gpu": "prelease-gpu.json", "post_worker_quiescence": "post-worker-quiescence.json", "lease_history": "lease-history.json", "single_worker": "single-bf16.json", "single_stdout": "single.stdout.txt", "single_stderr": "single.stderr.txt", "single_child_fingerprint": "single-child-fingerprint.json", "bf16_resume_checkpoint_store": "bf16-resume-store-index.json", "ddp_worker": "ddp-skip.json", "ddp_stdout": "ddp.stdout.txt", "ddp_stderr": "ddp.stderr.txt", "ddp_child_fingerprint": "ddp-child-fingerprint.json", **{f"chart_csv_{index}": name for index, name in enumerate(sorted(csv_hashes))}, **{f"chart_svg_{index}": name for index, name in enumerate(sorted(svg_hashes))}}
        reproduction_sha = {role: _sha(staging / filename) for role, filename in reproduction.items()}
        index = _with_hash({"schema_version": "stage1-s1-9-formalization-index-v8", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "git_branch": _git(repository_root, "branch", "--show-current"), "checked_at": _now(), "s1_7_handoff": upstream_s17, "s1_8_handoff": upstream_s18, "role_refs": role_files, "role_sha256": role_sha, "reproduction_role_refs": reproduction, "reproduction_role_sha256": reproduction_sha, "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "csv_sha256": csv_hashes, "svg_sha256": svg_hashes, "validation_ref": "validation.json", "validation_sha256": _sha(work / "validation.json"), "replay_ref": "replay-validation.json", "replay_sha256": _sha(work / "replay-validation.json"), "replay_hash": replay["replay_hash"], "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"]})
        _validate_s1_9_schemas(repository_root, {"index": index})
        _write(staging / "index.json", index)
        _write(staging / "success.json", _with_hash({"schema_version": "stage1-s1-9-attempt-success-v1", "status": "PASS", "completed_at": _now(), "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "validation_sha256": _sha(work / "validation.json"), "failed_marker_present": False}))
        staged_index = _mapping(load_canonical_json(staging / "index.json"), field="staged.index")
        staged_validation = _mapping(load_canonical_json(staging / "validation.json"), field="staged.validation")
        _validate_s1_9_schemas(repository_root, {"index": staged_index, "validation": staged_validation})
        staged_checkpoint_store_index = _mapping(load_canonical_json(staging / "bf16-resume-store-index.json"), field="staged.bf16_checkpoint_store")
        _validate_s1_9_schemas(repository_root, {"bf16_checkpoint_store": staged_checkpoint_store_index})
        if staged_index != index or staged_validation != validation or not _verify_checkpoint_store_reproduction(staging / "bf16-resume-store", staged_checkpoint_store_index) or staged_checkpoint_store_index != checkpoint_store_index or any(_sha(staging / role_files[role]) != digest for role, digest in role_sha.items()) or any(_sha(staging / reproduction[role]) != digest for role, digest in reproduction_sha.items()):
            raise Stage1S19FormalError("S1_9_PUBLISH_READBACK_BINDING_FAILED")
        os.replace(staging, target)
        return {"index_ref": (target / "index.json").relative_to(root).as_posix(), "validation_ref": (target / "validation.json").relative_to(root).as_posix()}
    except BaseException as error:
        # A failed readback must never leave a PASS marker in an unpublished
        # staging directory for a human or a later tool to mistake as output.
        if staging is not None and (staging / "success.json").is_file():
            (staging / "success.json").unlink()
        _failure_marker(work, error, phase=phase)
        if lease is not None and not released and not isinstance(error, Stage1S19ManualInterventionRequired):
            _release_lease_verified(lease, outcome="FAILED")
        elif lease is not None and not released:
            lease.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-7-index-ref", required=True); parser.add_argument("--s1-8-index-ref", required=True); parser.add_argument("--s1-8-binding-json", required=True)
    parser.add_argument("--gpu-capability-ref", required=True); parser.add_argument("--capability-binding-json", required=True)
    parser.add_argument("--approved-gpu-uuids", required=True); parser.add_argument("--attempt-id", required=True); parser.add_argument("--lease-owner", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800); parser.add_argument("--minimum-compute-capability", type=float, default=8.0); parser.add_argument("--max-temperature-c", type=int, default=85)
    args = parser.parse_args(argv)
    s18 = _parse_json_object(args.s1_8_binding_json, field="s1_8_binding", required={"index_sha256", "index_artifact_hash", "gate_artifact_hash", "producer_commit", "schema_version", "task_id", "gate_id"})
    capability = _parse_json_object(args.capability_binding_json, field="capability_binding", required={"task_id", "artifact_kind", "artifact_hash", "config_hash"})
    print(execute(repository=args.repository, data_root=args.data_root, s1_7_index_ref=args.s1_7_index_ref, s1_8_index_ref=args.s1_8_index_ref, s1_8_binding=s18, gpu_capability_ref=args.gpu_capability_ref, capability_binding=capability, approved_gpu_uuids=_parse_uuids(args.approved_gpu_uuids), attempt_id=args.attempt_id, lease_owner=args.lease_owner, timeout_seconds=args.timeout_seconds, minimum_compute_capability=args.minimum_compute_capability, max_temperature_c=args.max_temperature_c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
