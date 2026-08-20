#!/usr/bin/env python3
"""Fail-closed formal publisher for S1.10 / G1-RESUME.

This publisher deliberately has no fallback that turns its local synthetic
fixture into formal evidence.  The caller must supply two immutable upstream
bindings (S1.8 and S1.9) and a separately published, fresh-process formal
observation proving both single-process and four-rank resume paths.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Mapping


TASK_ID = "stage1.10_checkpoint_resume_and_artifacts"
GATE_ID = "G1-RESUME"
FIXTURE_ID = "stage1-s110-checkpoint-fixture-v1"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_ATTEMPT = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class Stage1S110FormalError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(repository), *args], text=True, capture_output=True, check=False, timeout=30)
    if result.returncode:
        raise Stage1S110FormalError(f"S1_10_GIT_FAILED:{args[0]}")
    return result.stdout.strip()


def _logical(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1S110FormalError(f"S1_10_REFERENCE_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1S110FormalError(f"S1_10_REFERENCE_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1S110FormalError(f"S1_10_REFERENCE_ESCAPE:{field}") from error
    return path


def _parse_binding(raw: str, *, field: str, task_id: str, gate_id: str) -> dict[str, str]:
    required = {"index_sha256", "index_artifact_hash", "gate_artifact_hash", "producer_commit", "schema_version", "task_id", "gate_id"}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Stage1S110FormalError(f"S1_10_BINDING_JSON_INVALID:{field}") from error
    if (
        not isinstance(value, Mapping) or set(value) != required
        or any(not isinstance(item, str) or not item for item in value.values())
        or value.get("task_id") != task_id or value.get("gate_id") != gate_id
    ):
        raise Stage1S110FormalError(f"S1_10_BINDING_FIELDS_INVALID:{field}")
    return {str(key): str(item) for key, item in value.items()}


def _parse_uuids(raw: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(values) != 4 or len(values) != len(set(values)) or any(_UUID.fullmatch(item) is None for item in values):
        raise Stage1S110FormalError("S1_10_APPROVED_UUIDS_INVALID")
    return values


def _parse_capability_binding(raw: str) -> dict[str, str]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Stage1S110FormalError("S1_10_CAPABILITY_BINDING_JSON_INVALID") from error
    required = {"task_id", "artifact_kind", "artifact_hash", "config_hash"}
    if not isinstance(value, Mapping) or set(value) != required or any(not isinstance(item, str) or not item for item in value.values()):
        raise Stage1S110FormalError("S1_10_CAPABILITY_BINDING_FIELDS_INVALID")
    return {str(key): str(item) for key, item in value.items()}


def _capability(root: Path, reference: str, binding: Mapping[str, str], approved: tuple[str, ...]) -> dict[str, Any]:
    """Bind CUDA authority to a committed capability before any lease/torch child."""

    from param_importance_nlp.contracts.runtime_evidence import RuntimeCapabilityEvidence
    from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact
    required = {"task_id", "artifact_kind", "artifact_hash", "config_hash"}
    if set(binding) != required:
        raise Stage1S110FormalError("S1_10_CAPABILITY_BINDING_FIELDS_INVALID")
    _logical(root, reference, field="capability")
    try:
        artifact = load_committed_task_artifact(root, reference, require_formal=True)
        evidence = RuntimeCapabilityEvidence.from_mapping(artifact.payload)
    except Exception as error:
        raise Stage1S110FormalError("S1_10_CAPABILITY_COMMIT_INVALID") from error
    if (artifact.identity.task_id, artifact.identity.artifact_kind, artifact.identity.artifact_hash, artifact.identity.config_hash) != (binding["task_id"], binding["artifact_kind"], binding["artifact_hash"], binding["config_hash"]):
        raise Stage1S110FormalError("S1_10_CAPABILITY_BINDING_MISMATCH")
    metadata = evidence.metadata if isinstance(evidence.metadata, Mapping) else {}
    allowed = metadata.get("allowed_gpu_uuids")
    if evidence.capability != "cuda" or evidence.status != "VERIFIED" or not isinstance(allowed, list) or tuple(allowed) != approved:
        raise Stage1S110FormalError("S1_10_CAPABILITY_UUID_MISMATCH")
    return {"reference": reference, "binding": dict(binding), "artifact_commit_ref": artifact.identity.commit_ref, "allowed_gpu_uuids": list(approved)}


def _gpu_preflight(approved: tuple[str, ...]) -> dict[str, Any]:
    """CPU-only A100/occupancy preflight; every selected UUID must be idle."""

    query = "uuid,name,compute_cap,temperature.gpu,utilization.gpu,memory.used"
    completed = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False, timeout=30)
    if completed.returncode:
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_QUERY_FAILED")
    rows: dict[str, dict[str, str]] = {}
    for line in completed.stdout.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 6:
            rows[values[0]] = {"name": values[1], "compute_cap": values[2], "temperature_c": values[3], "utilization_percent": values[4], "memory_used_mib": values[5]}
    selected = [rows.get(uuid) for uuid in approved]
    if any(item is None for item in selected):
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_UUID_MISSING")
    try:
        healthy = all(float(item["compute_cap"]) >= 8.0 and int(item["temperature_c"]) <= 85 and int(item["utilization_percent"]) == 0 and int(item["memory_used_mib"]) == 0 for item in selected if item is not None)
    except ValueError as error:
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_PARSE_INVALID") from error
    if not healthy:
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_NOT_IDLE_A100")
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True, capture_output=True, check=False, timeout=30)
    if processes.returncode:
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_PROCESS_QUERY_FAILED")
    occupied = sorted({line.split(",", 1)[0].strip() for line in processes.stdout.splitlines() if line.strip()})
    if set(occupied) & set(approved):
        raise Stage1S110FormalError("S1_10_GPU_PREFLIGHT_EXTERNAL_OCCUPANCY")
    return _with_hash({"schema_version": "stage1-s1-10-gpu-preflight-v1", "status": "PASS", "approved_gpu_uuids": list(approved), "selected": selected, "compute_process_uuids": occupied})


def _run_worker(command: list[str], *, environment: Mapping[str, str], timeout_seconds: int, lease: Any, stdout: Path, stderr: Path) -> None:
    """Run one token-bound process tree with bounded heartbeat and cleanup."""

    token = environment.get("S1_10_RUN_TOKEN")
    if not isinstance(token, str) or _TOKEN.fullmatch(token) is None:
        raise Stage1S110FormalError("S1_10_WORKER_TOKEN_INVALID")
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        child = subprocess.Popen(command, stdout=out, stderr=err, text=True, env=dict(environment), start_new_session=True)
        deadline = time.monotonic() + timeout_seconds
        try:
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    if os.name != "nt":
                        os.killpg(os.getpgid(child.pid), signal.SIGTERM)
                    else:  # formal servers are Linux; do not claim safe tree cleanup on Windows.
                        child.terminate()
                    child.wait(timeout=30)
                    raise Stage1S110FormalError("S1_10_WORKER_TIMEOUT_CLEANED")
                lease.heartbeat(); time.sleep(5.0)
        finally:
            if child.poll() is None:
                if os.name != "nt": os.killpg(os.getpgid(child.pid), signal.SIGKILL)
                else: child.kill()
                child.wait(timeout=30)
        if child.returncode != 0:
            raise Stage1S110FormalError(f"S1_10_WORKER_FAILED:{child.returncode}")
    # A launcher exit alone is not a closure proof: torchrun can outlive its
    # parent.  The token is only supplied to this run's process tree, so any
    # survivor makes the formal attempt fail before lease release.
    if os.name != "nt":
        survivors: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                if f"S1_10_RUN_TOKEN={token}".encode("utf-8") in (entry / "environ").read_bytes().split(b"\0"):
                    survivors.append(int(entry.name))
            except OSError:
                continue
        if survivors:
            for pid in survivors:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    continue
            time.sleep(1.0)
            raise Stage1S110FormalError("S1_10_WORKER_RESIDUAL_RUN_TOKEN_PROCESS")


def _write(path: Path, value: Mapping[str, Any]) -> None:
    from param_importance_nlp.contracts.jsonio import write_canonical_json

    if path.exists():
        raise Stage1S110FormalError(f"S1_10_IMMUTABLE_OUTPUT_EXISTS:{path.name}")
    write_canonical_json(path, dict(value))


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    from param_importance_nlp.contracts.jsonio import canonical_json_hash

    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _formal_observation(repository: Path, root: Path, reference: str, *, expected_commit: str) -> dict[str, Any]:
    """Load the separately generated GPU observation without accepting prose."""

    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json

    path = _logical(root, reference, field="formal_observation")
    value = load_canonical_json(path)
    expected = {
        "schema_version", "status", "task_id", "gate_id", "fixture_id", "execution_commit", "run_token_sha256",
        "single_process_resume", "four_rank_resume", "run_owned_resources_released",
        "single_cases", "four_rank_cases", "single_report_ref", "single_report_sha256",
        "four_rank_report_ref", "four_rank_report_sha256", "artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise Stage1S110FormalError("S1_10_FORMAL_OBSERVATION_FIELDS_INVALID")
    _schema_validate(repository, {"formal_observation": dict(value)})
    body = dict(value)
    supplied = body.pop("artifact_hash", None)
    if (
        supplied != canonical_json_hash(body)
        or value.get("schema_version") != "stage1-s1-10-formal-observation-v1"
        or value.get("status") != "PASS"
        or value.get("task_id") != TASK_ID
        or value.get("gate_id") != GATE_ID
        or value.get("fixture_id") != FIXTURE_ID
        or value.get("execution_commit") != expected_commit
        or not isinstance(value.get("run_token_sha256"), str)
        or _SHA256.fullmatch(str(value.get("run_token_sha256"))) is None
        or any(value.get(name) is not True for name in ("single_process_resume", "four_rank_resume", "run_owned_resources_released"))
        or value.get("single_cases") != ["pre_skip", "post_skip"]
        or value.get("four_rank_cases") != ["pre_skip", "post_skip"]
    ):
        raise Stage1S110FormalError("S1_10_FORMAL_OBSERVATION_NOT_PASS")
    for prefix, expected_mode, expected_world_size in (
        ("single", "single", 1),
        ("four_rank", "four-rank", 4),
    ):
        report_ref = value.get(f"{prefix}_report_ref")
        report_sha = value.get(f"{prefix}_report_sha256")
        report_path = _logical(root, report_ref, field=f"{prefix}_report")
        if not report_path.is_file() or not isinstance(report_sha, str) or _sha(report_path) != report_sha:
            raise Stage1S110FormalError(f"S1_10_FORMAL_OBSERVATION_REPORT_HASH_INVALID:{prefix}")
        report = load_canonical_json(report_path)
        if (
            not isinstance(report, Mapping)
            or report.get("status") != "PASS"
            or report.get("task_id") != TASK_ID
            or report.get("gate_id") != GATE_ID
            or report.get("fixture_id") != FIXTURE_ID
            or report.get("execution_commit") != expected_commit
            or report.get("run_token_sha256") != value["run_token_sha256"]
            or report.get("mode") != expected_mode
            or report.get("phase") != "resume"
            or report.get("world_size") != expected_world_size
        ):
            raise Stage1S110FormalError(f"S1_10_FORMAL_OBSERVATION_REPORT_NOT_PASS:{prefix}")
    return dict(value)


def _schema_validate(repository: Path, objects: Mapping[str, Mapping[str, Any]]) -> None:
    """Use the repository's dependency-free strict Draft subset validator."""

    from param_importance_nlp.contracts.jsonio import loads_strict_json

    validator_path = repository / "ops" / "stage1" / "formalize_s1_6.py"
    spec = importlib.util.spec_from_file_location("_s110_schema_validator", validator_path)
    if spec is None or spec.loader is None:
        raise Stage1S110FormalError("S1_10_SCHEMA_VALIDATOR_UNAVAILABLE")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = {
        "checkpoint_fixture": "s1-10-checkpoint-fixture-v1.json",
        "resume_report": "s1-10-resume-report-v2.json",
        "oracle_bundle": "s1-10-oracle-bundle-v1.json",
        "trace_bundle": "s1-10-trace-bundle-v1.json",
        "comparison_table": "s1-10-comparison-table-v1.json",
        "artifact_manifest": "s1-10-artifact-manifest-v1.json",
        "gate_record": "s1-10-gate-record-v1.json",
        "replay": "s1-10-replay-validation-v1.json",
        "validation": "s1-10-validation-v2.json",
        "index": "s1-10-formalization-index-v2.json",
        "formal_observation": "s1-10-formal-observation-v1.json",
    }
    schema_paths = {
        path.name: path
        for path in sorted((repository / "schemas" / "stage1").glob("s1-10-*.json"))
    }
    if not set(names.values()).issubset(schema_paths):
        raise Stage1S110FormalError("S1_10_SCHEMA_REGISTRY_SET_INVALID")
    registry: dict[str, Mapping[str, Any]] = {}
    for path in schema_paths.values():
        value = loads_strict_json(path.read_bytes())
        if not isinstance(value, Mapping) or not isinstance(value.get("$id"), str):
            raise Stage1S110FormalError(f"S1_10_SCHEMA_INVALID:{path.name}")
        registry[path.name] = value
        registry[str(value["$id"])] = value
    for role, value in objects.items():
        filename = names.get(role)
        if filename is None or filename not in registry:
            raise Stage1S110FormalError(f"S1_10_SCHEMA_ROLE_UNKNOWN:{role}")
        try:
            module._validate_schema(value, registry[filename], registry, document=registry[filename], path=role)
        except Exception as error:
            raise Stage1S110FormalError(f"S1_10_SCHEMA_VALIDATION_FAILED:{role}") from error


def _chart_rows(table: Mapping[str, Any], trace: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = table.get("rows")
    continuous = trace.get("continuous")
    if not isinstance(rows, list) or not isinstance(continuous, Mapping) or not isinstance(continuous.get("trace"), list):
        raise Stage1S110FormalError("S1_10_CHART_INPUT_INVALID")
    errors: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise Stage1S110FormalError("S1_10_CHART_ROW_INVALID")
        errors.append({"attempt": row["attempt"], "case_id": row["case_id"], "maximum_absolute_error": row["maximum_absolute_error"], "passed": row["passed"]})
    timeline: list[dict[str, Any]] = []
    for row in continuous["trace"]:
        if not isinstance(row, Mapping):
            raise Stage1S110FormalError("S1_10_CHART_TRACE_ROW_INVALID")
        timeline.append({"attempt": row["attempt"], "cursor": row["cursor"], "growth_tracker": row["growth_tracker"], "scale": row["scale"], "skip_count": row["skip_count"], "status": row["status"], "successful_step": row["successful_step"]})
    return {"resume-errors.csv": errors, "state-timeline.csv": timeline}


def _charts(work: Path, table: Mapping[str, Any], trace: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    data = _chart_rows(table, trace)
    csv_hashes: dict[str, str] = {}
    svg_hashes: dict[str, str] = {}
    for name, rows in data.items():
        path = work / name
        fields = list(rows[0]) if rows else []
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        csv_hashes[name] = _sha(path)
        circles = "".join(f'<circle class="data-point" data-row="{index}" cx="{60 + index * 80}" cy="{160 - min(140, abs(float(row.get("maximum_absolute_error", row.get("scale", 0.0)))) * 100)}" r="3"/>' for index, row in enumerate(rows))
        svg = work / name.replace(".csv", ".svg")
        svg.write_text(f'<?xml version="1.0" encoding="UTF-8"?><svg xmlns="http://www.w3.org/2000/svg" width="640" height="200" data-source="{name}"><title>{name}</title><line class="x-axis" x1="40" y1="170" x2="620" y2="170" stroke="black"/><line class="y-axis" x1="40" y1="20" x2="40" y2="170" stroke="black"/>{circles}</svg>\n', encoding="utf-8")
        if svg.read_text(encoding="utf-8").count('class="data-point"') != len(rows):
            raise Stage1S110FormalError("S1_10_CHART_PROJECTION_INVALID")
        svg_hashes[svg.name] = _sha(svg)
    return csv_hashes, svg_hashes


def execute(*, repository: str | Path, data_root: str | Path, s1_8_index_ref: str, s1_8_binding: Mapping[str, str], s1_9_index_ref: str, s1_9_binding: Mapping[str, str], gpu_capability_ref: str, capability_binding: Mapping[str, str], approved_gpu_uuids: tuple[str, ...], lease_owner: str, attempt_id: str, timeout_seconds: int = 1800) -> dict[str, str]:
    repository_root, root = Path(repository).resolve(strict=True), Path(data_root).resolve(strict=True)
    if str(repository_root / "src") not in sys.path:
        sys.path.insert(0, str(repository_root / "src"))
    commit = _git(repository_root, "rev-parse", "HEAD")
    if _COMMIT.fullmatch(commit) is None or _git(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise Stage1S110FormalError("S1_10_FORMAL_REQUIRES_CLEAN_WORKTREE")
    if _ATTEMPT.fullmatch(attempt_id) is None:
        raise Stage1S110FormalError("S1_10_ATTEMPT_ID_INVALID")
    from param_importance_nlp.stage1_checkpoint_resume import build_stage1_s110_evidence, replay_stage1_s110_evidence, validate_parameterized_handoff, validate_stage1_s110_evidence
    from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
    from param_importance_nlp.runtime.operations import GpuLeaseIdentity, ProjectGpuLease

    s1_8 = validate_parameterized_handoff(root, s1_8_index_ref, expected_binding=s1_8_binding, expected_task_id="stage1.08_ddp_and_gradient_accumulation", expected_gate_id="G1-DDP")
    s1_9 = validate_parameterized_handoff(root, s1_9_index_ref, expected_binding=s1_9_binding, expected_task_id="stage1.09_precision_clipping_and_optimizer_boundaries", expected_gate_id="G1-NUMERIC")
    if not lease_owner or len(approved_gpu_uuids) != 4 or len(set(approved_gpu_uuids)) != 4 or any(_UUID.fullmatch(item) is None for item in approved_gpu_uuids):
        raise Stage1S110FormalError("S1_10_LEASE_OR_UUID_ARGUMENT_INVALID")
    target = root / "evidence" / "stage1" / "s1-10-formal" / commit / attempt_id
    work = root / "tmp" / "stage1-s1-10" / commit / attempt_id
    if target.exists() or work.exists():
        raise Stage1S110FormalError("S1_10_ATTEMPT_ALREADY_EXISTS")
    work.mkdir(parents=True)
    lease = None
    released = False
    try:
        capability = _capability(root, gpu_capability_ref, capability_binding, approved_gpu_uuids)
        preflight = _gpu_preflight(approved_gpu_uuids)
        plan_hash = canonical_json_hash({"task_id": TASK_ID, "execution_commit": commit, "s1_8": s1_8, "s1_9": s1_9, "capability": capability, "approved_gpu_uuids": list(approved_gpu_uuids), "timeout_seconds": timeout_seconds, "determinism": {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}})
        lease = ProjectGpuLease(root, GpuLeaseIdentity(run_id=f"s110-{attempt_id}", lease_id=f"s110-{attempt_id}", gpu_uuids=approved_gpu_uuids, owner=lease_owner, config_hash=plan_hash, environment_hash=preflight["artifact_hash"]))
        lease.acquire(); lease.heartbeat()
        run_token = hashlib.sha256(canonical_json_hash({"plan_hash": plan_hash, "attempt_id": attempt_id, "commit": commit, "approved_gpu_uuids": list(approved_gpu_uuids)}).encode("ascii")).hexdigest()
        environment = dict(os.environ)
        environment.update({"PYTHONPATH": str(repository_root / "src") + os.pathsep + environment.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": ",".join(approved_gpu_uuids), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0", "S1_10_RUN_TOKEN": run_token})
        _write(work / "preflight.json", _with_hash({"schema_version": "stage1-s1-10-preflight-v1", "status": "PASS", "execution_commit": commit, "plan_hash": plan_hash, "capability": capability, "gpu": preflight, "approved_gpu_uuids": list(approved_gpu_uuids)}))
        single_source_path, single_path = work / "formal-single-source-report.json", work / "formal-single-report.json"
        four_source_path, four_path = work / "formal-four-rank-source-report.json", work / "formal-four-rank-report.json"
        single_env = dict(environment); single_env["CUDA_VISIBLE_DEVICES"] = approved_gpu_uuids[0]
        _run_worker([sys.executable, str(repository_root / "ops" / "stage1" / "run_s1_10_resume_worker.py"), "--repository", str(repository_root), "--output", str(single_source_path), "--checkpoint-root", str(work / "single-checkpoints"), "--mode", "single", "--phase", "source", "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuids", approved_gpu_uuids[0]], environment=single_env, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "single-source.stdout.txt", stderr=work / "single-source.stderr.txt")
        _run_worker([sys.executable, str(repository_root / "ops" / "stage1" / "run_s1_10_resume_worker.py"), "--repository", str(repository_root), "--output", str(single_path), "--checkpoint-root", str(work / "single-checkpoints"), "--mode", "single", "--phase", "resume", "--source-report", str(single_source_path), "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuids", approved_gpu_uuids[0]], environment=single_env, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "single.stdout.txt", stderr=work / "single.stderr.txt")
        _run_worker([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4", "--rdzv-backend", "c10d", "--rdzv-id", f"s110-source-{run_token}", "--rdzv-endpoint", "127.0.0.1:0", str(repository_root / "ops" / "stage1" / "run_s1_10_resume_worker.py"), "--repository", str(repository_root), "--output", str(four_source_path), "--checkpoint-root", str(work / "four-rank-checkpoints"), "--mode", "four-rank", "--phase", "source", "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuids", ",".join(approved_gpu_uuids)], environment=environment, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "four-source.stdout.txt", stderr=work / "four-source.stderr.txt")
        _run_worker([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "4", "--rdzv-backend", "c10d", "--rdzv-id", f"s110-resume-{run_token}", "--rdzv-endpoint", "127.0.0.1:0", str(repository_root / "ops" / "stage1" / "run_s1_10_resume_worker.py"), "--repository", str(repository_root), "--output", str(four_path), "--checkpoint-root", str(work / "four-rank-checkpoints"), "--mode", "four-rank", "--phase", "resume", "--source-report", str(four_source_path), "--execution-commit", commit, "--run-token", run_token, "--approved-gpu-uuids", ",".join(approved_gpu_uuids)], environment=environment, timeout_seconds=timeout_seconds, lease=lease, stdout=work / "four-rank.stdout.txt", stderr=work / "four-rank.stderr.txt")
        single_source, four_source = load_canonical_json(single_source_path), load_canonical_json(four_source_path)
        single, four = load_canonical_json(single_path), load_canonical_json(four_path)
        expected_worker = {"schema_version", "status", "task_id", "gate_id", "fixture_id", "execution_commit", "run_token_sha256", "mode", "phase", "world_size", "approved_gpu_uuids", "cuda_visible_devices", "environment", "checks", "per_rank", "artifact_hash"}
        token_hash = hashlib.sha256(run_token.encode("ascii")).hexdigest()
        for report, mode, world, visible in ((single, "single", 1, approved_gpu_uuids[0]), (four, "four-rank", 4, ",".join(approved_gpu_uuids))):
            required_checks = {"production_training_engine", "source_process_exited_before_resume", "pre_skip_resume_matches_uninterrupted", "post_skip_resume_matches_uninterrupted", "three_attempts_after_each_boundary", "cursor_rng_optimizer_scheduler_scaler_importance_restored", "checkpoint_lineage_complete", "group_authoritative_load_verified"}
            if mode == "four-rank": required_checks |= {"all_rank_state_parity", "barrier_atomic_group_pre_and_post"}
            if not isinstance(report, Mapping) or set(report) != expected_worker or report.get("status") != "PASS" or report.get("task_id") != TASK_ID or report.get("gate_id") != GATE_ID or report.get("fixture_id") != FIXTURE_ID or report.get("execution_commit") != commit or report.get("run_token_sha256") != token_hash or report.get("mode") != mode or report.get("phase") != "resume" or report.get("world_size") != world or report.get("approved_gpu_uuids") != ([approved_gpu_uuids[0]] if mode == "single" else list(approved_gpu_uuids)) or report.get("cuda_visible_devices") != visible or not isinstance(report.get("checks"), Mapping) or set(report["checks"]) != required_checks or not all(report["checks"].values()) or report.get("artifact_hash") != canonical_json_hash({key: value for key, value in report.items() if key != "artifact_hash"}):
                raise Stage1S110FormalError("S1_10_WORKER_REPORT_INVALID")
        for report, mode, world, visible in ((single_source, "single", 1, approved_gpu_uuids[0]), (four_source, "four-rank", 4, ",".join(approved_gpu_uuids))):
            expected_source_checks = {"source_committed_pre_and_post", "source_post_parent_is_pre"} if mode == "single" else {"source_committed_pre_and_post"}
            if (
                not isinstance(report, Mapping)
                or set(report) != expected_worker
                or report.get("status") != "PASS"
                or report.get("task_id") != TASK_ID
                or report.get("gate_id") != GATE_ID
                or report.get("fixture_id") != FIXTURE_ID
                or report.get("execution_commit") != commit
                or report.get("run_token_sha256") != token_hash
                or report.get("mode") != mode
                or report.get("phase") != "source"
                or report.get("world_size") != world
                or report.get("approved_gpu_uuids") != ([approved_gpu_uuids[0]] if mode == "single" else list(approved_gpu_uuids))
                or report.get("cuda_visible_devices") != visible
                or not isinstance(report.get("checks"), Mapping)
                or set(report["checks"]) != expected_source_checks
                or not all(report["checks"].values())
                or not isinstance(report.get("per_rank"), list)
                or len(report["per_rank"]) != world
                or report.get("artifact_hash") != canonical_json_hash({key: value for key, value in report.items() if key != "artifact_hash"})
            ):
                raise Stage1S110FormalError("S1_10_SOURCE_WORKER_REPORT_INVALID")
            for rank, row in enumerate(report["per_rank"]):
                source = row.get("source") if isinstance(row, Mapping) else None
                if (
                    not isinstance(row, Mapping)
                    or set(row) != {"rank", "environment", "source", "checks"}
                    or row.get("rank") != rank
                    or not isinstance(source, Mapping)
                    or set(source) != {"pid", "process_identity", "pre_checkpoint_id", "post_checkpoint_id", "pre_group", "post_group"}
                    or isinstance(source.get("pid"), bool)
                    or not isinstance(source.get("pid"), int)
                    or source["pid"] <= 0
                    or not isinstance(source.get("process_identity"), str)
                    or not source["process_identity"]
                    or not isinstance(source.get("pre_checkpoint_id"), str)
                    or not isinstance(source.get("post_checkpoint_id"), str)
                    or source["pre_checkpoint_id"] == source["post_checkpoint_id"]
                    or not isinstance(row.get("checks"), Mapping)
                    or set(row["checks"]) != {"source_committed_pre_and_post", "source_post_parent_is_pre"}
                    or not all(row["checks"].values())
                ):
                    raise Stage1S110FormalError("S1_10_SOURCE_WORKER_DETAIL_INVALID")
        postflight = _gpu_preflight(approved_gpu_uuids)
        history = lease.release(outcome="GPU_PHASE_SUCCESS")
        if not isinstance(history, Path) or not history.is_file() or lease.current_path.exists():
            raise Stage1S110FormalError("S1_10_LEASE_RELEASE_CLOSURE_INVALID")
        released = True; lease = None; shutil.copy2(history, work / "lease-history.json")
        observation = {"schema_version": "stage1-s1-10-formal-observation-v1", "status": "PASS", "task_id": TASK_ID, "gate_id": GATE_ID, "fixture_id": FIXTURE_ID, "execution_commit": commit, "run_token_sha256": token_hash, "single_process_resume": True, "four_rank_resume": True, "run_owned_resources_released": True, "single_cases": ["pre_skip", "post_skip"], "four_rank_cases": ["pre_skip", "post_skip"], "single_report_ref": "formal-single-report.json", "single_report_sha256": _sha(single_path), "four_rank_report_ref": "formal-four-rank-report.json", "four_rank_report_sha256": _sha(four_path)}
        observation["artifact_hash"] = canonical_json_hash(observation); _schema_validate(repository_root, {"formal_observation": observation}); _write(work / "formal-observation.json", observation)
        formal_inputs = {
            "formal-observation.json": work / "formal-observation.json",
            "formal-single-source-report.json": single_source_path,
            "formal-single-report.json": single_path,
            "formal-four-rank-source-report.json": four_source_path,
            "formal-four-rank-report.json": four_path,
        }
        command = [sys.executable, "-m", "pytest", "-q", "--disable-warnings", "--basetemp", str(work / "pytest-tmp"), "-o", f"cache_dir={work / 'pytest-cache'}", "tests/test_stage1_s110_checkpoint_resume.py", "tests/test_runtime_training_engine.py"]
        run = subprocess.run(command, cwd=repository_root, text=True, capture_output=True, check=False, timeout=timeout_seconds)
        if run.returncode:
            raise Stage1S110FormalError("S1_10_REGRESSION_FAILED")
        regression = _with_hash({"schema_version": "stage1-s1-10-regression-v1", "returncode": 0, "stdout_sha256": hashlib.sha256(run.stdout.encode()).hexdigest(), "stderr_sha256": hashlib.sha256(run.stderr.encode()).hexdigest()})
        _write(work / "regression.json", regression)
        upstream = {"s1_8": s1_8, "s1_9": s1_9}
        evidence = build_stage1_s110_evidence(repository_root, producer_commit=commit, scope="formal_server_single_and_four_rank_resume", upstream_evidence=upstream, formal_observation=observation, scratch_root=work / "checkpoint-fixtures")
        if evidence["gate_record"].get("status") != "PASS":
            raise Stage1S110FormalError("S1_10_FORMAL_REQUIREMENTS_NOT_MET")
        validate_stage1_s110_evidence(evidence, source_root=repository_root)
        replay = replay_stage1_s110_evidence(evidence, source_root=repository_root, scratch_root=work / "replay-fixtures")
        files = {"resume_report": "resume-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "artifact_manifest": "artifact-manifest.json", "gate_record": "g1-resume-record.json"}
        for role, name in files.items():
            _write(work / name, evidence[role])
        _write(work / "replay-validation.json", replay)
        persisted = {role: load_canonical_json(work / name) for role, name in files.items()}
        if not all(isinstance(value, Mapping) for value in persisted.values()):
            raise Stage1S110FormalError("S1_10_PERSISTED_ROLE_INVALID")
        reloaded = {role: dict(value) for role, value in persisted.items() if isinstance(value, Mapping)}
        _schema_validate(repository_root, reloaded)
        csv_hashes, svg_hashes = _charts(work, reloaded["comparison_table"], reloaded["trace_bundle"])
        role_sha = {role: _sha(work / name) for role, name in files.items()}
        direct = {"all_requirements_true": True, "formal_observation_hash_valid": True, "parameterized_s1_8_handoff": True, "parameterized_s1_9_handoff": True, "replay_matches": replay == replay_stage1_s110_evidence(reloaded, source_root=repository_root, scratch_root=work / "replay-verify")}
        validation = _with_hash({"schema_version": "stage1-s1-10-validation-v2", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "execution_scope": "formal_server_single_and_four_rank_resume", "fixture_id": FIXTURE_ID, "producer_commit": commit, "consumer_commit": commit, "upstream": upstream, "direct_checks": direct, "role_sha256": role_sha, "replay_sha256": _sha(work / "replay-validation.json"), "replay_hash": replay["replay_hash"]})
        _write(work / "validation.json", validation)
        index = _with_hash({"schema_version": "stage1-s1-10-formalization-index-v2", "status": "PASS", "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "generator_git_commit": commit, "consumer_git_commit": commit, "git_branch": _git(repository_root, "branch", "--show-current"), "checked_at": _now(), "upstream": upstream, "role_refs": files, "role_sha256": role_sha, "chart_csv_sha256": csv_hashes, "chart_svg_sha256": svg_hashes, "formal_observation_ref": "formal-observation.json", "formal_observation_sha256": _sha(formal_inputs["formal-observation.json"]), "formal_observation_artifact_hash": observation["artifact_hash"], "formal_run_token_sha256": observation["run_token_sha256"], "formal_single_report_ref": "formal-single-report.json", "formal_single_report_sha256": _sha(formal_inputs["formal-single-report.json"]), "formal_four_rank_report_ref": "formal-four-rank-report.json", "formal_four_rank_report_sha256": _sha(formal_inputs["formal-four-rank-report.json"]), "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "validation_ref": "validation.json", "validation_sha256": _sha(work / "validation.json"), "replay_ref": "replay-validation.json", "replay_sha256": _sha(work / "replay-validation.json"), "replay_hash": replay["replay_hash"], "next_task_ids": ["stage1.11_reporting_and_exit_gate"]})
        _schema_validate(repository_root, {"index": index, "validation": validation, "replay": replay})
        _write(work / "index.json", index)
        staging = work.parent / f".{attempt_id}.publishing"
        if staging.exists():
            raise Stage1S110FormalError("S1_10_STAGING_EXISTS")
        staging.mkdir()
        published_names = [
            *files.values(),
            "replay-validation.json",
            "validation.json",
            "index.json",
            "regression.json",
            *sorted(csv_hashes),
            *sorted(svg_hashes),
        ]
        for name in published_names:
            source = work / name
            if not source.is_file():
                raise Stage1S110FormalError(f"S1_10_PUBLISH_SOURCE_MISSING:{name}")
            shutil.copy2(source, staging / name)
        for name, source in formal_inputs.items():
            if not source.is_file():
                raise Stage1S110FormalError(f"S1_10_PUBLISH_FORMAL_INPUT_MISSING:{name}")
            shutil.copy2(source, staging / name)
        staged_roles = {
            role: load_canonical_json(staging / name)
            for role, name in files.items()
        }
        staged_replay = load_canonical_json(staging / "replay-validation.json")
        staged_validation = load_canonical_json(staging / "validation.json")
        staged_index = load_canonical_json(staging / "index.json")
        if not all(isinstance(value, Mapping) for value in (*staged_roles.values(), staged_replay, staged_validation, staged_index)):
            raise Stage1S110FormalError("S1_10_PUBLISH_READBACK_TYPE_INVALID")
        readback = {role: dict(value) for role, value in staged_roles.items() if isinstance(value, Mapping)}
        assert isinstance(staged_replay, Mapping) and isinstance(staged_validation, Mapping) and isinstance(staged_index, Mapping)
        _schema_validate(repository_root, {**readback, "replay": dict(staged_replay), "validation": dict(staged_validation), "index": dict(staged_index)})
        if (
            readback != evidence
            or dict(staged_replay) != replay
            or dict(staged_validation) != validation
            or dict(staged_index) != index
            or {role: _sha(staging / name) for role, name in files.items()} != role_sha
            or {name: _sha(staging / name) for name in csv_hashes} != csv_hashes
            or {name: _sha(staging / name) for name in svg_hashes} != svg_hashes
            or {name: _sha(staging / name) for name in formal_inputs} != {name: _sha(source) for name, source in formal_inputs.items()}
            or _sha(staging / "replay-validation.json") != _sha(work / "replay-validation.json")
            or _sha(staging / "validation.json") != _sha(work / "validation.json")
            or _sha(staging / "index.json") != _sha(work / "index.json")
        ):
            raise Stage1S110FormalError("S1_10_PUBLISH_READBACK_BINDING_FAILED")
        success = _with_hash({"schema_version": "stage1-s1-10-attempt-success-v1", "status": "PASS", "completed_at": _now(), "gate_artifact_hash": evidence["gate_record"]["artifact_hash"], "index_sha256": _sha(staging / "index.json"), "validation_sha256": _sha(staging / "validation.json")})
        _write(staging / "success.json", success)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, target)
        return {"index_ref": (target / "index.json").relative_to(root).as_posix(), "validation_ref": (target / "validation.json").relative_to(root).as_posix()}
    except BaseException as error:
        # The work directory is an auditable failed attempt.  It is deliberately
        # not renamed into the immutable formal evidence namespace.
        if work.is_dir() and not (work / "success.json").exists() and not (work / "failure.json").exists():
            _write(work / "failure.json", _with_hash({"schema_version": "stage1-s1-10-attempt-failure-v1", "status": "FAILED", "error_type": type(error).__name__, "error": str(error)}))
        if lease is not None and not released:
            try:
                lease.release(outcome="FAILED")
            except Exception:
                # An uncertain lease record is intentionally retained for an
                # operator; formal publication remains failed either way.
                lease.close()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s1-8-index-ref", required=True)
    parser.add_argument("--s1-8-binding-json", required=True)
    parser.add_argument("--s1-9-index-ref", required=True)
    parser.add_argument("--s1-9-binding-json", required=True)
    parser.add_argument("--gpu-capability-ref", required=True)
    parser.add_argument("--capability-binding-json", required=True)
    parser.add_argument("--approved-gpu-uuids", required=True)
    parser.add_argument("--lease-owner", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args(argv)
    s1_8 = _parse_binding(args.s1_8_binding_json, field="s1_8", task_id="stage1.08_ddp_and_gradient_accumulation", gate_id="G1-DDP")
    s1_9 = _parse_binding(args.s1_9_binding_json, field="s1_9", task_id="stage1.09_precision_clipping_and_optimizer_boundaries", gate_id="G1-NUMERIC")
    capability = _parse_capability_binding(args.capability_binding_json)
    print(execute(repository=args.repository, data_root=args.data_root, s1_8_index_ref=args.s1_8_index_ref, s1_8_binding=s1_8, s1_9_index_ref=args.s1_9_index_ref, s1_9_binding=s1_9, gpu_capability_ref=args.gpu_capability_ref, capability_binding=capability, approved_gpu_uuids=_parse_uuids(args.approved_gpu_uuids), lease_owner=args.lease_owner, attempt_id=args.attempt_id, timeout_seconds=args.timeout_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
