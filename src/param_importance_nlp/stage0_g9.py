"""Formal Stage 0 S0.11 layered tests and independent replay gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import time
from typing import Any, Final, Mapping, Sequence

from .atomic import atomic_write_bytes, sha256_file
from .contracts import (
    GateRecord,
    GateStatus,
    ResolvedConfigV2,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .experiments import build_default_task_runtime
from .runtime import (
    GpuLeaseIdentity,
    ProjectGpuLease,
    TaskArtifactStore,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from .stage0_bootstrap import Stage0SourceBinding, build_stage0_formal_config
from .stage0_g7_recovery import (
    Stage0G7RecoveryFormalState,
    load_stage0_g7_recovery_formal_state,
)
from .stage0_g8 import Stage0G8FormalState, load_stage0_g8_formal_state
from .stage0_g9_replay import REPLAY_PLAN_SCHEMA, REPLAY_REPORT_SCHEMA
from .stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0GateCheck,
    Stage0GateReport,
)


TASK_ID: Final = "stage0.11_test_quality_and_replay"
GATE_ID: Final = "stage0.G9"
_OUTPUT_KINDS = {"test_report", "replay_report", "gate_summary"}
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_REQUIRED_GATES = frozenset(
    {
        "stage0.G0-C", "stage0.G0-G", "stage0.G1", "stage0.G2",
        "stage0.G3", "stage0.G3-S1", "stage0.G3-S2", "stage0.G3-S4",
        "stage0.G3-S5", "stage0.G3-S6",
        "stage0.G4", "stage0.G5", "stage0.G6", "stage0.G7-LOGGING",
        "stage0.G7", "stage0.G8-C", "stage0.G8-S4", "stage0.G8-S5", "stage0.G8",
    }
)
_CRITICAL_SOURCE_REFS = (
    "configs/stage0/g9-test-matrix-v1.json",
    "docs/stage0-replay-runbook.md",
    "fixtures/stage0/deterministic-training-v1.json",
    "ops/stage0/formalize_g9.py",
    "ops/stage0/run_g9_independent_replay.py",
    "ops/stage0/offline_guard/sitecustomize.py",
    "src/param_importance_nlp/deterministic_fixture.py",
    "src/param_importance_nlp/experiments/stage01_task_runners.py",
    "src/param_importance_nlp/offline_guard.py",
    "src/param_importance_nlp/stage0_g7_recovery.py",
    "src/param_importance_nlp/stage0_g9.py",
    "src/param_importance_nlp/stage0_g9_replay.py",
    "schemas/stage0-g9-evidence-v1.json",
    "schemas/stage0-g9-formalization-index-v1.json",
    "schemas/stage0-g9-independent-replay-plan-v1.json",
    "schemas/stage0-g9-independent-replay-report-v1.json",
    "schemas/stage0-g9-test-matrix-v1.json",
    "schemas/stage0-deterministic-training-fixture-v1.json",
)


class Stage0G9Error(RuntimeError):
    """S0.11 test or independent replay evidence failed closed."""


@dataclass(frozen=True, slots=True)
class G9SourceBinding:
    repository: Path
    git_commit: str
    git_branch: str
    critical_source_hashes: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Stage0G9FormalizationResult:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config_ref: str
    environment_ref: str
    index_ref: str


@dataclass(frozen=True, slots=True)
class Stage0G9FormalState:
    environment: TaskRuntimeEnvironment
    task_output_refs: Mapping[str, str]
    config: ResolvedConfigV2
    config_ref: str
    environment_ref: str
    index_ref: str
    index_sha256: str
    gate_artifact_hash: str
    g8_index_ref: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G9Error(f"G9_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G9Error(f"G9_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G9Error(f"G9_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G9Error(f"G9_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _with_hash(value: Mapping[str, JSONValue]) -> dict[str, JSONValue]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_or_verify(path: Path, value: Mapping[str, JSONValue]) -> None:
    if path.exists():
        if load_canonical_json(path) != dict(value):
            raise Stage0G9Error("G9_CONTROL_FILE_DRIFT")
        return
    write_canonical_json(path, dict(value))


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _capture_source() -> G9SourceBinding:
    repository = Path(__file__).resolve().parents[2]
    probes = {
        "top": _git(repository, "rev-parse", "--show-toplevel"),
        "head": _git(repository, "rev-parse", "HEAD"),
        "branch": _git(repository, "branch", "--show-current"),
        "tracked": _git(repository, "ls-files", "--error-unmatch", "--", *_CRITICAL_SOURCE_REFS),
        "status": _git(repository, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    if any(item.returncode != 0 for item in probes.values()):
        raise Stage0G9Error("G9_SOURCE_GIT_PROBE_FAILED")
    if Path(probes["top"].stdout.strip()).resolve() != repository:
        raise Stage0G9Error("G9_SOURCE_ROOT_MISMATCH")
    commit = probes["head"].stdout.strip()
    branch = probes["branch"].stdout.strip()
    if _GIT_COMMIT_RE.fullmatch(commit) is None or not branch or probes["status"].stdout.strip():
        raise Stage0G9Error("G9_FORMAL_SOURCE_NOT_CLEAN_OR_IDENTIFIED")
    hashes = {reference: sha256_file(repository / reference) for reference in _CRITICAL_SOURCE_REFS}
    return G9SourceBinding(repository, commit, branch, hashes)


def _load_hashed(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    value = _mapping(load_canonical_json(path), field=field)
    declared = value.pop("artifact_hash", None)
    if value.get("schema_version") != schema or declared != canonical_json_hash(value):
        raise Stage0G9Error(f"G9_HASHED_OBJECT_INVALID:{field}")
    value["artifact_hash"] = declared
    return value


def _selected_gpu_uuids(environment: TaskRuntimeEnvironment, root: Path) -> tuple[str, ...]:
    reference = environment.evidence_refs.get("g4_provenance")
    if reference is None:
        raise Stage0G9Error("G9_G4_PROVENANCE_MISSING")
    envelope = _mapping(
        load_committed_task_artifact(root, reference, require_formal=True).payload,
        field="g4_provenance_envelope",
    )
    provenance = _mapping(envelope.get("provenance_record"), field="g4_provenance")
    selected = provenance.get("device_mapping")
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or len(set(selected)) != 4
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G9Error("G9_G4_DEVICE_MAPPING_INVALID")
    return tuple(selected)


def _prepare_controls(
    *,
    root: Path,
    source: G9SourceBinding,
    g8: Stage0G8FormalState,
    g7: Stage0G7RecoveryFormalState,
) -> str:
    control_ref = f"evidence/stage0/g9-controls/{g8.gate_artifact_hash}/control.json"
    control_root_ref = str(PurePosixPath(control_ref).parent)
    matrix_ref = f"{control_root_ref}/test-matrix.json"
    fixture_ref = f"{control_root_ref}/deterministic-fixture.json"
    runbook_ref = f"{control_root_ref}/runbook.md"
    source_matrix = source.repository / "configs/stage0/g9-test-matrix-v1.json"
    source_fixture = source.repository / "fixtures/stage0/deterministic-training-v1.json"
    source_runbook = source.repository / "docs/stage0-replay-runbook.md"
    matrix = _mapping(load_canonical_json(source_matrix), field="source_matrix")
    fixture = _mapping(load_canonical_json(source_fixture), field="source_fixture")
    _write_or_verify(_logical_path(root, matrix_ref, field="matrix_ref"), matrix)
    _write_or_verify(_logical_path(root, fixture_ref, field="fixture_ref"), fixture)
    runbook_path = _logical_path(root, runbook_ref, field="runbook_ref")
    runbook_bytes = source_runbook.read_bytes()
    if runbook_path.exists():
        if runbook_path.read_bytes() != runbook_bytes:
            raise Stage0G9Error("G9_RUNBOOK_CONTROL_DRIFT")
    else:
        atomic_write_bytes(runbook_path, runbook_bytes)
    control = _with_hash(
        {
            "schema_version": "stage0-g9-control-manifest-v1",
            "generator_git_commit": source.git_commit,
            "g8_index_ref": g8.index_ref,
            "g8_index_sha256": g8.index_sha256,
            "g8_gate_artifact_hash": g8.gate_artifact_hash,
            "g7_index_ref": g7.index_ref,
            "g7_index_sha256": g7.index_sha256,
            "g7_gate_artifact_hash": g7.gate_artifact_hash,
            "g7_config_ref": g7.config_ref,
            "g7_config_sha256": sha256_file(_logical_path(root, g7.config_ref, field="g7_config_ref")),
            "g7_config_hash": g7.config.config_hash,
            "g7_environment_ref": g7.environment_ref,
            "g7_environment_sha256": sha256_file(
                _logical_path(root, g7.environment_ref, field="g7_environment_ref")
            ),
            "g7_environment_hash": g7.environment.environment_hash,
            "test_matrix_ref": matrix_ref,
            "test_matrix_sha256": sha256_file(_logical_path(root, matrix_ref, field="matrix_ref")),
            "test_matrix_hash": matrix["artifact_hash"],
            "fixture_ref": fixture_ref,
            "fixture_sha256": sha256_file(_logical_path(root, fixture_ref, field="fixture_ref")),
            "fixture_hash": fixture["artifact_hash"],
            "runbook_ref": runbook_ref,
            "runbook_sha256": sha256_file(runbook_path),
        }
    )
    _write_or_verify(_logical_path(root, control_ref, field="control_ref"), control)
    return control_ref


def build_stage0_g9_config(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    state: Stage0G8FormalState,
    g7_state: Stage0G7RecoveryFormalState,
    control_ref: str,
) -> ResolvedConfigV2:
    root = Path(data_root).resolve(strict=True)
    selected = _selected_gpu_uuids(state.environment, root)
    base = g7_state.config.base_config
    runtime = _mapping(base.section("runtime"), field="runtime")
    precision = _mapping(base.section("precision"), field="precision")
    batching = _mapping(base.section("batching"), field="batching")
    per_device = int(batching["per_device_batch_size"])
    accumulation = int(batching["accumulation_steps"])
    base_overrides: dict[str, object] = {
        "identity": {"route": f"stage0-g9-{state.gate_artifact_hash[:12]}"},
        "runtime": {**runtime, "device": "cuda", "allow_dirty_worktree": False},
        "model": base.section("model"),
        "data": base.section("data"),
        "loss": base.section("loss"),
        "batching": {
            **batching,
            "global_batch_size": per_device * 4 * accumulation,
            "no_sync": accumulation > 1,
        },
        "distributed": {
            "world_size": 4,
            "backend": "nccl",
            "device_ids": [0, 1, 2, 3],
            "timeout_seconds": 300,
        },
        "precision": {
            **precision,
            "compute_dtype": "bfloat16",
            "gradient_dtype": "float32",
            "statistic_dtype": "float32",
            "amp": True,
        },
        "optimizer": base.section("optimizer"),
    }
    v2_overrides: dict[str, object] = {
        "execution": {"timeout_seconds": 43200, "max_attempts": 1},
        "providers": g7_state.config.section("providers"),
        "orchestration": {"matrix_ref": control_ref},
        "launcher": {
            "kind": "torchrun",
            "backend": "nccl",
            "world_size": 4,
            "init_method": "env",
            "init_ref": None,
            "rendezvous_id": f"stage0-g9-{selected[0][-8:]}",
            "max_restarts": 0,
        },
        "recovery": {
            "mode": "restart_idempotent",
            "resume_ref": None,
            "max_restarts": 0,
            "safe_boundary": "immutable_publish",
        },
    }
    return build_stage0_formal_config(
        binding.repository,
        task_id=TASK_ID,
        input_refs=tuple(state.task_output_refs.values()),
        output_dir=f"evidence/stage0/tasks/11-{state.gate_artifact_hash}",
        base_overrides=base_overrides,
        v2_overrides=v2_overrides,
    )


def _read_control(request: TaskExecutionRequest, root: Path, source: G9SourceBinding) -> dict[str, Any]:
    orchestration = _mapping(request.config.section("orchestration"), field="orchestration")
    control_ref = orchestration.get("matrix_ref")
    if not isinstance(control_ref, str):
        raise Stage0G9Error("G9_CONTROL_REF_MISSING")
    control = _load_hashed(
        _logical_path(root, control_ref, field="control_ref"),
        schema="stage0-g9-control-manifest-v1",
        field="control",
    )
    if control.get("generator_git_commit") != source.git_commit:
        raise Stage0G9Error("G9_CONTROL_SOURCE_COMMIT_DRIFT")
    for ref_field, sha_field in (
        ("g8_index_ref", "g8_index_sha256"),
        ("g7_index_ref", "g7_index_sha256"),
        ("g7_config_ref", "g7_config_sha256"),
        ("g7_environment_ref", "g7_environment_sha256"),
        ("test_matrix_ref", "test_matrix_sha256"),
        ("fixture_ref", "fixture_sha256"),
        ("runbook_ref", "runbook_sha256"),
    ):
        path = _logical_path(root, control[ref_field], field=ref_field)
        if not path.is_file() or sha256_file(path) != control[sha_field]:
            raise Stage0G9Error(f"G9_CONTROL_FILE_SHA_DRIFT:{ref_field}")
    return control


def _compute_processes(selected: Sequence[str]) -> list[dict[str, JSONValue]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise Stage0G9Error("G9_NVIDIA_PROCESS_QUERY_FAILED")
    rows: list[dict[str, JSONValue]] = []
    for line in result.stdout.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 3 and fields[0].isdigit() and fields[1] in selected:
            rows.append({"pid": int(fields[0]), "gpu_uuid": fields[1], "used_memory_mib": int(fields[2])})
    return rows


def _wait_for_no_compute_processes(selected: Sequence[str], *, timeout_seconds: float = 30.0) -> list[dict[str, JSONValue]]:
    deadline = time.monotonic() + timeout_seconds
    rows = _compute_processes(selected)
    while rows and time.monotonic() < deadline:
        time.sleep(0.25)
        rows = _compute_processes(selected)
    return rows


def _launch_replay(
    *,
    root: Path,
    source: G9SourceBinding,
    plan_ref: str,
    transcript_ref: str,
    selected: Sequence[str],
    lease: ProjectGpuLease,
    timeout_seconds: int,
) -> dict[str, JSONValue]:
    transcript_path = _logical_path(root, transcript_ref, field="transcript_ref")
    stdout_path = transcript_path.with_suffix(".stdout.log")
    stderr_path = transcript_path.with_suffix(".stderr.log")
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(
        {
            "PARAM_IMPORTANCE_DATA_ROOT": str(root),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(source.repository / "src"),
        }
    )
    command = [
        sys.executable,
        str(source.repository / "ops/stage0/run_g9_independent_replay.py"),
        "--data-root",
        str(root),
        "--plan-ref",
        plan_ref,
    ]
    started_at = _now()
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=source.repository,
            env=environment,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=os.name != "nt",
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
        )
        next_heartbeat = time.monotonic() + 30.0
        while process.poll() is None:
            now = time.monotonic()
            if now - started > timeout_seconds:
                timed_out = True
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGTERM)
                else:  # pragma: no cover - formal execution is Linux
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name != "nt":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:  # pragma: no cover
                        process.kill()
                    process.wait()
                break
            if now >= next_heartbeat:
                lease.heartbeat()
                next_heartbeat = now + 30.0
            time.sleep(1.0)
    residual = _wait_for_no_compute_processes(selected)
    stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    transcript = _with_hash(
        {
            "schema_version": "stage0-g9-independent-launch-transcript-v1",
            "plan_ref": plan_ref,
            "started_at": started_at,
            "completed_at": _now(),
            "duration_seconds": time.monotonic() - started,
            "return_code": int(process.returncode or 0),
            "timed_out": timed_out,
            "stdout_ref": stdout_path.relative_to(root).as_posix(),
            "stdout_sha256": sha256_file(stdout_path),
            "stderr_ref": stderr_path.relative_to(root).as_posix(),
            "stderr_sha256": sha256_file(stderr_path),
            "stdout_tail": stdout[-20000:],
            "stderr_tail": stderr[-20000:],
            "residual_compute_processes": residual,
        }
    )
    _write_or_verify(transcript_path, transcript)
    return transcript


def _validate_runbook(source: G9SourceBinding) -> dict[str, JSONValue]:
    path = source.repository / "docs/stage0-replay-runbook.md"
    text = path.read_text(encoding="utf-8")
    required = (
        "PARAM_IMPORTANCE_DATA_ROOT",
        "formalize_g9.py",
        "GPU UUID",
        "禁止",
        ".part",
        "manifest",
        "checkpoint",
        "run ID",
        "环境漂移",
        "Stage 1",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise Stage0G9Error(f"G9_RUNBOOK_REQUIRED_CONTENT_MISSING:{missing}")
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    checked: list[str] = []
    for target in links:
        relative = target.split("#", 1)[0]
        if not relative:
            continue
        if "://" in relative or relative.startswith("/") or "\\" in relative:
            raise Stage0G9Error(f"G9_RUNBOOK_LINK_NOT_REPOSITORY_RELATIVE:{target}")
        resolved = (path.parent / relative).resolve()
        try:
            resolved.relative_to(source.repository)
        except ValueError as error:
            raise Stage0G9Error(f"G9_RUNBOOK_LINK_ESCAPE:{target}") from error
        if not resolved.exists():
            raise Stage0G9Error(f"G9_RUNBOOK_LINK_MISSING:{target}")
        checked.append(target)
    return {
        "status": "PASS",
        "runbook_ref": path.relative_to(source.repository).as_posix(),
        "runbook_sha256": sha256_file(path),
        "required_content_count": len(required),
        "checked_links": checked,
    }


def _upstream_gate_records(request: TaskExecutionRequest, root: Path) -> dict[str, dict[str, JSONValue]]:
    bindings = {
        "stage0.G5": "g5_training_smoke",
        "stage0.G6": "g6_distributed_validation",
        "stage0.G7": "g7_resume_equivalence",
        "stage0.G8": "g8_capacity",
    }
    result: dict[str, dict[str, JSONValue]] = {}
    for gate_id, key in bindings.items():
        reference = request.environment.evidence_refs.get(key)
        if reference is None:
            raise Stage0G9Error(f"G9_UPSTREAM_GATE_REF_MISSING:{gate_id}")
        payload = _mapping(
            load_committed_task_artifact(root, reference, require_formal=True).payload,
            field=f"upstream.{gate_id}",
        )
        if gate_id == "stage0.G8":
            records = _mapping(payload.get("gate_records"), field="g8.gate_records")
            raw = _mapping(records.get(gate_id), field="g8.gate_record")
        else:
            raw = _mapping(payload.get("gate_record"), field=f"{gate_id}.gate_record")
        gate = GateRecord.from_mapping(raw)
        if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
            raise Stage0G9Error(f"G9_UPSTREAM_GATE_NOT_PASS:{gate_id}")
        result[gate_id] = gate.to_dict()
    return result


def _validate_replay_report(
    root: Path,
    reference: str,
    *,
    plan: Mapping[str, Any],
    source: G9SourceBinding,
) -> dict[str, Any]:
    report = _load_hashed(
        _logical_path(root, reference, field="replay_report_ref"),
        schema=REPLAY_REPORT_SCHEMA,
        field="replay_report",
    )
    plan_path = _logical_path(root, report.get("plan_ref"), field="report.plan_ref")
    if (
        report.get("status") != "PASS"
        or report.get("generator_git_commit") != source.git_commit
        or report.get("replay_id") != plan["replay_id"]
        or report.get("plan_sha256") != sha256_file(plan_path)
        or report.get("started_from_empty_output_root") is not True
        or report.get("g7_config_hash") != plan["g7_config_hash"]
        or report.get("g7_environment_hash") != plan["g7_environment_hash"]
        or report.get("test_matrix_hash") != plan["test_matrix_hash"]
        or report.get("fixture_hash") != plan["fixture_hash"]
    ):
        raise Stage0G9Error("G9_REPLAY_REPORT_IDENTITY_INVALID")
    pytest_report = _mapping(report.get("pytest_report"), field="pytest_report")
    counts = _mapping(pytest_report.get("counts"), field="pytest_counts")
    if (
        pytest_report.get("status") != "PASS"
        or pytest_report.get("return_code") != 0
        or pytest_report.get("timed_out") is not False
        or counts.get("failed") != 0
        or counts.get("errors") != 0
        or counts.get("hard_skipped") != 0
        or not isinstance(counts.get("passed"), int)
        or int(counts["passed"]) <= 0
    ):
        raise Stage0G9Error("G9_PYTEST_REPORT_NOT_PASS")
    layers = report.get("layer_results")
    if (
        not isinstance(layers, list)
        or {item.get("layer") for item in layers if isinstance(item, Mapping)}
        != {"local_cpu", "server_cpu", "single_gpu", "four_gpu", "fault", "replay"}
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "PASS"
            or item.get("skipped") != 0
            for item in layers
        )
    ):
        raise Stage0G9Error("G9_REPLAY_LAYER_REPORT_INVALID")
    network = _mapping(report.get("network_audit"), field="network_audit")
    if (
        network.get("external_attempt_count") != 0
        or network.get("hf_offline_environment") is not True
        or int(network.get("process_count", 0)) < 7
    ):
        raise Stage0G9Error("G9_REPLAY_NETWORK_AUDIT_INVALID")
    lineage = report.get("run_lineage")
    if (
        not isinstance(lineage, list)
        or len(lineage) != 4
        or len({row.get("run_id") for row in lineage if isinstance(row, Mapping)}) != 4
        or any(
            not isinstance(row, Mapping)
            or not isinstance(row.get("config_hash"), str)
            or re.fullmatch(r"[0-9a-f]{64}", str(row.get("config_hash"))) is None
            or row.get("environment_hash") != plan["g7_environment_hash"]
            or not isinstance(row.get("group_checkpoint_id"), str)
            or not _logical_path(root, row.get("report_ref"), field="lineage.report_ref").is_file()
            for row in lineage
        )
    ):
        raise Stage0G9Error("G9_REPLAY_LINEAGE_INVALID")
    return report


def _gate_evidence(
    request: TaskExecutionRequest,
    root: Path,
    source: G9SourceBinding,
    control: Mapping[str, Any],
    plan_ref: str,
    transcript_ref: str,
    report_ref: str,
    lease_ref: str,
    replay: Mapping[str, Any],
    runbook: Mapping[str, JSONValue],
    upstream: Mapping[str, Mapping[str, JSONValue]],
    source_refs: Sequence[str],
) -> tuple[dict[str, JSONValue], GateRecord, Stage0GateReport, tuple[str, ...]]:
    pytest_report = _mapping(replay["pytest_report"], field="pytest_report")
    counts = _mapping(pytest_report["counts"], field="pytest_counts")
    layer_results = replay["layer_results"]
    raw_evidence_refs = tuple(source_refs) + (
        str(request.config.section("orchestration")["matrix_ref"]),
        str(control["test_matrix_ref"]),
        str(control["fixture_ref"]),
        str(control["runbook_ref"]),
        plan_ref,
        transcript_ref,
        report_ref,
        report_ref.removesuffix(".json") + ".md",
        lease_ref,
        str(pytest_report["junit_ref"]),
    ) + tuple(
        str(item["ref"])
        for item in replay["network_audit"]["audit_files"]
    ) + tuple(
        str(reference)
        for values in replay["evidence_refs"].values()
        for reference in (values if isinstance(values, list) else [values])
    )
    evidence_refs = tuple(dict.fromkeys(raw_evidence_refs))
    checked_at = str(replay["completed_at"])
    checks = (
        Stage0GateCheck(
            "stage0.G9-LAYERED-TESTS",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "The full repository suite and every hard CPU/GPU/fault/replay layer passed with zero skipped hard tests.",
            measurements={
                "pytest_counts": counts,
                "declared_assertion_sites": pytest_report["declared_assertion_sites"],
                "layers": layer_results,
            },
            evidence_refs=(str(pytest_report["junit_ref"]), report_ref),
        ),
        Stage0GateCheck(
            "stage0.G9-DETERMINISTIC-FIXTURE",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "The source-generated tiny token fixture reproduced its initialization, loss and gradient identity.",
            measurements=_mapping(replay["fixture_validation"], field="fixture_validation"),
            evidence_refs=(str(control["fixture_ref"]), report_ref),
        ),
        Stage0GateCheck(
            "stage0.G9-INDEPENDENT-OFFLINE-REPLAY",
            Stage0CheckClass.CORRECTNESS,
            Stage0CheckStatus.PASS,
            "A manifest-only fresh subprocess replayed single-GPU, four-GPU/no_sync and checkpoint recovery from an empty output root with no Python-socket external attempt.",
            measurements={
                "network_audit": replay["network_audit"],
                "recovery_pair_metrics": replay["recovery_pair_metrics"],
                "run_lineage": replay["run_lineage"],
            },
            evidence_refs=(plan_ref, transcript_ref, report_ref),
        ),
        Stage0GateCheck(
            "stage0.G9-RUNBOOK-AND-REGRESSION-TRIGGERS",
            Stage0CheckClass.SAFETY,
            Stage0CheckStatus.PASS,
            "The repository runbook, evidence paths, prohibitions and regression-trigger matrix match the executable controls.",
            measurements={"runbook": runbook, "matrix_hash": control["test_matrix_hash"]},
            evidence_refs=(str(control["runbook_ref"]), str(control["test_matrix_ref"])),
        ),
    )
    gate = GateRecord(
        gate_id=GATE_ID,
        stage=0,
        status=GateStatus.PASS,
        checked_at=checked_at,
        measured={
            "pytest_passed": counts["passed"],
            "pytest_failed": counts["failed"],
            "pytest_errors": counts["errors"],
            "pytest_skipped": counts["skipped"],
            "pytest_hard_skipped": counts["hard_skipped"],
            "pytest_platform_excluded": counts["allowed_platform_skipped"],
            "hard_layers_passed": len(layer_results),
            "external_network_attempts": replay["network_audit"]["external_attempt_count"],
            "replayed_world_sizes": [1, 4],
            "fresh_output_root": replay["started_from_empty_output_root"],
        },
        threshold={
            "failed_max": 0,
            "errors_max": 0,
            "hard_skipped_max": 0,
            "external_network_attempts_max": 0,
            "required_world_sizes": [1, 4],
            "fresh_output_root_required": True,
        },
        evidence_refs=evidence_refs,
    )
    report = Stage0GateReport(
        gate_id=GATE_ID,
        generated_at=checked_at,
        generator_git_commit=source.git_commit,
        environment_id=str(request.config.base_config.section("runtime")["environment_id"]),
        config_hashes={TASK_ID: request.config.config_hash},
        input_evidence=tuple(
            Stage0EvidenceRef(
                ref=reference,
                sha256=sha256_file(_logical_path(root, reference, field="gate.evidence_ref")),
                schema_version=(
                    REPLAY_REPORT_SCHEMA
                    if reference == report_ref
                    else REPLAY_PLAN_SCHEMA
                    if reference == plan_ref
                    else "stage0-g9-supporting-evidence"
                ),
            )
            for reference in evidence_refs
        ),
        checks=checks,
    )
    canonical = _with_hash(
        {
            "schema_version": "stage0-g9-canonical-evidence-v1",
            "status": "PASS",
            "checked_at": checked_at,
            "generator_git_commit": source.git_commit,
            "critical_source_hashes": dict(source.critical_source_hashes),
            "config_hash": request.config.config_hash,
            "environment_hash": request.environment.environment_hash,
            "control_manifest_hash": control["artifact_hash"],
            "replay_report_ref": report_ref,
            "replay_report_sha256": sha256_file(_logical_path(root, report_ref, field="report_ref")),
            "replay_report_hash": replay["artifact_hash"],
            "runbook_validation": runbook,
            "upstream_gate_records": dict(upstream),
            "gate_record": gate.to_dict(),
            "gate_report": report.to_dict(),
        }
    )
    return canonical, gate, report, evidence_refs


def run_formal_g9_task(
    request: TaskExecutionRequest,
    root: Path,
    store: TaskArtifactStore,
    *,
    source_refs: Sequence[str],
) -> TaskRunResult:
    source = _capture_source()
    if request.task.task_id != TASK_ID or request.config.run_intent != "formal":
        raise Stage0G9Error("G9_FORMAL_REQUEST_INVALID")
    missing = sorted(_REQUIRED_GATES - request.environment.passed_gate_ids)
    if missing:
        raise Stage0G9Error(f"G9_REQUIRED_GATES_MISSING:{missing}")
    control = _read_control(request, root, source)
    selected = _selected_gpu_uuids(request.environment, root)
    if _compute_processes(selected):
        raise Stage0G9Error("G9_PREFLIGHT_EXTERNAL_GPU_PROCESS")
    replay_id = f"stage0-g9-{source.git_commit[:12]}-{request.config.config_hash[:12]}"
    run_prefix = f"g9-{request.config.config_hash[:12]}"
    plan_root_ref = f"evidence/stage0/g9-plans/{request.config.config_hash}"
    suite_ref = f"evidence/stage0/g9-replays/{replay_id}"
    plan_ref = f"{plan_root_ref}/plan.json"
    transcript_ref = f"{plan_root_ref}/transcript.json"
    report_ref = f"{suite_ref}/independent-replay-report.json"
    plan = _with_hash(
        {
            "schema_version": REPLAY_PLAN_SCHEMA,
            "replay_id": replay_id,
            "run_id_prefix": run_prefix,
            "generator_git_commit": source.git_commit,
            "g7_config_ref": str(control["g7_config_ref"]),
            "g7_config_sha256": str(control["g7_config_sha256"]),
            "g7_config_hash": str(control["g7_config_hash"]),
            "g7_environment_ref": str(control["g7_environment_ref"]),
            "g7_environment_sha256": str(control["g7_environment_sha256"]),
            "g7_environment_hash": str(control["g7_environment_hash"]),
            "test_matrix_ref": str(control["test_matrix_ref"]),
            "test_matrix_sha256": str(control["test_matrix_sha256"]),
            "test_matrix_hash": str(control["test_matrix_hash"]),
            "fixture_ref": str(control["fixture_ref"]),
            "fixture_sha256": str(control["fixture_sha256"]),
            "fixture_hash": str(control["fixture_hash"]),
            "suite_root_ref": suite_ref,
            "report_ref": report_ref,
            "selected_gpu_uuids": list(selected),
            "pytest_timeout_seconds": 5400,
            "replay_timeout_seconds": 21600,
        }
    )
    _write_or_verify(_logical_path(root, plan_ref, field="plan_ref"), plan)
    lease = ProjectGpuLease(
        root,
        GpuLeaseIdentity(
            run_id=replay_id,
            lease_id=f"g9-{request.config.config_hash[:16]}",
            gpu_uuids=selected,
            owner="stage0-g9-independent-replay",
            config_hash=request.config.config_hash,
            environment_hash=request.environment.environment_hash,
        ),
    )
    lease.acquire()
    outcome = "FAILED"
    try:
        transcript = _launch_replay(
            root=root,
            source=source,
            plan_ref=plan_ref,
            transcript_ref=transcript_ref,
            selected=selected,
            lease=lease,
            timeout_seconds=int(plan["replay_timeout_seconds"]),
        )
        if (
            transcript["return_code"] != 0
            or transcript["timed_out"] is not False
            or transcript["residual_compute_processes"]
        ):
            raise Stage0G9Error(
                f"G9_INDEPENDENT_REPLAY_CHILD_FAILED:{transcript['stdout_tail']}:{transcript['stderr_tail']}"
            )
        replay = _validate_replay_report(root, report_ref, plan=plan, source=source)
        outcome = "SUCCESS"
    finally:
        lease_history_path = lease.release(outcome=outcome)
    if _compute_processes(selected):
        raise Stage0G9Error("G9_POSTFLIGHT_GPU_PROCESS_RESIDUE")
    lease_ref = lease_history_path.relative_to(root).as_posix()
    runbook = _validate_runbook(source)
    upstream = _upstream_gate_records(request, root)
    canonical, gate, _gate_report, evidence_refs = _gate_evidence(
        request,
        root,
        source,
        control,
        plan_ref,
        transcript_ref,
        report_ref,
        lease_ref,
        replay,
        runbook,
        upstream,
        source_refs,
    )
    refs: dict[str, str] = {}
    for kind in request.task.artifact_kinds:
        payload: dict[str, JSONValue] = {
            "schema_version": "stage0-g9-evidence-v1",
            "artifact_role": kind,
            "task_id": TASK_ID,
            "config_hash": request.config.config_hash,
            "environment_hash": request.environment.environment_hash,
            "canonical_evidence_hash": canonical["artifact_hash"],
            "canonical_evidence": canonical,
            "role_summary": (
                {"pytest_counts": replay["pytest_report"]["counts"], "layer_results": replay["layer_results"]}
                if kind == "test_report"
                else {"replay_id": replay["replay_id"], "run_lineage": replay["run_lineage"]}
                if kind == "replay_report"
                else {"gate_id": gate.gate_id, "gate_status": gate.status.value}
            ),
        }
        published = store.publish(
            task_id=TASK_ID,
            artifact_kind=kind,
            config_hash=request.config.config_hash,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
            source_refs=evidence_refs,
        )
        refs[kind] = published.commit_ref
    return TaskRunResult.passed(
        request,
        artifact_refs=refs,
        message="Stage 0 G9 layered tests and independent offline replay passed",
        metadata={"stage0_g9_specialized": True, "gate_id": GATE_ID, "replay_id": replay_id},
    )


def validate_formal_g9_outputs(
    request: TaskExecutionRequest,
    root: Path,
    outputs: Mapping[str, str],
) -> GateRecord:
    source = _capture_source()
    if set(outputs) != _OUTPUT_KINDS:
        raise Stage0G9Error("G9_OUTPUT_KINDS_INVALID")
    canonical: dict[str, Any] | None = None
    for kind in request.task.artifact_kinds:
        loaded = load_committed_task_artifact(root, outputs[kind], require_formal=True)
        payload = _mapping(loaded.payload, field=f"payload.{kind}")
        if (
            payload.get("schema_version") != "stage0-g9-evidence-v1"
            or payload.get("artifact_role") != kind
            or payload.get("task_id") != TASK_ID
            or payload.get("config_hash") != request.config.config_hash
            or payload.get("environment_hash") != request.environment.environment_hash
        ):
            raise Stage0G9Error("G9_OUTPUT_ENVELOPE_INVALID")
        current = _mapping(payload.get("canonical_evidence"), field="canonical_evidence")
        if payload.get("canonical_evidence_hash") != current.get("artifact_hash"):
            raise Stage0G9Error("G9_OUTPUT_CANONICAL_HASH_INVALID")
        if canonical is None:
            canonical = current
        elif canonical != current:
            raise Stage0G9Error("G9_OUTPUT_CANONICAL_EVIDENCE_DRIFT")
    assert canonical is not None
    declared = canonical.pop("artifact_hash", None)
    if declared != canonical_json_hash(canonical):
        raise Stage0G9Error("G9_OUTPUT_CANONICAL_ARTIFACT_HASH_INVALID")
    canonical["artifact_hash"] = declared
    if (
        canonical.get("status") != "PASS"
        or canonical.get("generator_git_commit") != source.git_commit
        or canonical.get("critical_source_hashes") != dict(source.critical_source_hashes)
        or canonical.get("config_hash") != request.config.config_hash
        or canonical.get("environment_hash") != request.environment.environment_hash
    ):
        raise Stage0G9Error("G9_OUTPUT_SOURCE_OR_IDENTITY_INVALID")
    if canonical.get("runbook_validation") != _validate_runbook(source):
        raise Stage0G9Error("G9_OUTPUT_RUNBOOK_VALIDATION_DRIFT")
    if canonical.get("upstream_gate_records") != _upstream_gate_records(request, root):
        raise Stage0G9Error("G9_OUTPUT_UPSTREAM_GATE_REPLAY_DRIFT")
    replay_ref = canonical.get("replay_report_ref")
    if not isinstance(replay_ref, str):
        raise Stage0G9Error("G9_OUTPUT_REPLAY_REF_INVALID")
    control = _read_control(request, root, source)
    # The authoritative plan ref is preserved by the replay report; use that
    # instead of deriving a path if repository layout changes.
    replay_raw = _load_hashed(
        _logical_path(root, replay_ref, field="replay_ref"),
        schema=REPLAY_REPORT_SCHEMA,
        field="replay",
    )
    authoritative_plan_ref = replay_raw.get("plan_ref")
    if not isinstance(authoritative_plan_ref, str):
        raise Stage0G9Error("G9_OUTPUT_REPLAY_PLAN_REF_INVALID")
    plan = _load_hashed(
        _logical_path(root, authoritative_plan_ref, field="plan_ref"),
        schema=REPLAY_PLAN_SCHEMA,
        field="plan",
    )
    replay = _validate_replay_report(root, replay_ref, plan=plan, source=source)
    if (
        canonical.get("replay_report_sha256")
        != sha256_file(_logical_path(root, replay_ref, field="replay_ref"))
        or canonical.get("replay_report_hash") != replay["artifact_hash"]
        or canonical.get("control_manifest_hash") != control["artifact_hash"]
    ):
        raise Stage0G9Error("G9_OUTPUT_REPLAY_OR_CONTROL_DRIFT")
    gate = GateRecord.from_mapping(_mapping(canonical.get("gate_record"), field="gate_record"))
    gate_report = Stage0GateReport.from_mapping(
        _mapping(canonical.get("gate_report"), field="gate_report")
    )
    if gate.gate_id != GATE_ID or gate.status is not GateStatus.PASS or gate_report.status.value != "PASS":
        raise Stage0G9Error("G9_OUTPUT_GATE_NOT_PASS")
    for item in gate_report.input_evidence:
        path = _logical_path(root, item.ref, field="gate_report.input_evidence")
        if not path.is_file() or sha256_file(path) != item.sha256:
            raise Stage0G9Error("G9_OUTPUT_GATE_EVIDENCE_SHA_DRIFT")
    return gate


def execute_stage0_g9(
    *,
    binding: Stage0SourceBinding,
    data_root: str | Path,
    g8_index_ref: str,
) -> Stage0G9FormalizationResult:
    root = Path(data_root).resolve(strict=True)
    state = load_stage0_g8_formal_state(
        data_root=root,
        index_ref=g8_index_ref,
        expected_git_commit=binding.git_commit,
    )
    g7 = load_stage0_g7_recovery_formal_state(
        data_root=root,
        index_ref=state.g7_recovery_index_ref,
        expected_git_commit=binding.git_commit,
    )
    source = _capture_source()
    if source.git_commit != binding.git_commit or source.repository != binding.repository:
        raise Stage0G9Error("G9_EXECUTION_SOURCE_BINDING_MISMATCH")
    control_ref = _prepare_controls(root=root, source=source, g8=state, g7=g7)
    config = build_stage0_g9_config(
        binding=binding,
        data_root=root,
        state=state,
        g7_state=g7,
        control_ref=control_ref,
    )
    formal_dir = f"evidence/stage0/g9-formal/{state.gate_artifact_hash}"
    config_ref = f"{formal_dir}/resolved-config.json"
    _write_or_verify(_logical_path(root, config_ref, field="config_ref"), config.to_dict())
    result = build_default_task_runtime(root).execute(config, environment=state.environment)
    if result.status.value != "PASS" or not result.formal_eligible:
        raise Stage0G9Error(f"G9_FORMAL_TASK_NOT_PASS:{result.status.value}:{result.message}")
    outputs = dict(result.artifact_refs)
    request = TaskExecutionRequest(config, config.task_definition, state.environment)
    gate = validate_formal_g9_outputs(request, root, outputs)
    refs = dict(state.environment.evidence_refs)
    refs.update(
        {
            "g9_test_report": outputs["test_report"],
            "g9_replay_report": outputs["replay_report"],
            "g9_gate_summary": outputs["gate_summary"],
            "gate_stage0_g9": outputs["gate_summary"],
        }
    )
    environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids | frozenset({GATE_ID}),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs=refs,
    )
    environment_ref = f"{formal_dir}/environment.json"
    _write_or_verify(
        _logical_path(root, environment_ref, field="environment_ref"), environment.to_dict()
    )
    index = _with_hash(
        {
            "schema_version": "stage0-g9-formalization-index-v1",
            "generator_git_commit": binding.git_commit,
            "checked_at": gate.checked_at,
            "g8_index_ref": state.index_ref,
            "g8_index_sha256": state.index_sha256,
            "g8_gate_artifact_hash": state.gate_artifact_hash,
            "control_ref": control_ref,
            "control_sha256": sha256_file(_logical_path(root, control_ref, field="control_ref")),
            "config_ref": config_ref,
            "config_hash": config.config_hash,
            "task_output_refs": outputs,
            "gate_ref": outputs["gate_summary"],
            "gate_artifact_hash": gate.artifact_hash,
            "environment_ref": environment_ref,
            "environment_hash": environment.environment_hash,
            "next_task_id": "stage0.12_delivery_and_sync",
            "next_input_refs": list(outputs.values()),
        }
    )
    index_ref = f"{formal_dir}/index.json"
    _write_or_verify(_logical_path(root, index_ref, field="index_ref"), index)
    return Stage0G9FormalizationResult(environment, outputs, config_ref, environment_ref, index_ref)


def load_stage0_g9_formal_state(
    *,
    data_root: str | Path,
    index_ref: str,
    expected_git_commit: str,
) -> Stage0G9FormalState:
    root = Path(data_root).resolve(strict=True)
    index_path = _logical_path(root, index_ref, field="index_ref")
    raw = _load_hashed(index_path, schema="stage0-g9-formalization-index-v1", field="g9_index")
    expected_fields = {
        "schema_version", "generator_git_commit", "checked_at", "g8_index_ref",
        "g8_index_sha256", "g8_gate_artifact_hash", "control_ref", "control_sha256",
        "config_ref", "config_hash", "task_output_refs", "gate_ref",
        "gate_artifact_hash", "environment_ref", "environment_hash", "next_task_id",
        "next_input_refs", "artifact_hash",
    }
    if set(raw) != expected_fields:
        raise Stage0G9Error("G9_STATE_INDEX_FIELDS_INVALID")
    if raw.get("generator_git_commit") != expected_git_commit:
        raise Stage0G9Error("G9_STATE_SOURCE_COMMIT_INVALID")
    g8 = load_stage0_g8_formal_state(
        data_root=root,
        index_ref=str(raw["g8_index_ref"]),
        expected_git_commit=expected_git_commit,
    )
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, raw["config_ref"], field="config_ref")),
            field="config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, raw["environment_ref"], field="environment_ref")
            ),
            field="environment",
        )
    )
    outputs = _mapping(raw.get("task_output_refs"), field="task_output_refs")
    ordered = {kind: str(outputs[kind]) for kind in config.task_definition.artifact_kinds}
    request = TaskExecutionRequest(config, config.task_definition, g8.environment)
    gate = validate_formal_g9_outputs(request, root, ordered)
    next_inputs = raw.get("next_input_refs")
    if (
        raw.get("g8_index_sha256") != g8.index_sha256
        or raw.get("g8_gate_artifact_hash") != g8.gate_artifact_hash
        or raw.get("control_sha256")
        != sha256_file(_logical_path(root, raw["control_ref"], field="control_ref"))
        or config.config_hash != raw.get("config_hash")
        or environment.environment_hash != raw.get("environment_hash")
        or GATE_ID not in environment.passed_gate_ids
        or raw.get("gate_ref") != ordered["gate_summary"]
        or raw.get("gate_artifact_hash") != gate.artifact_hash
        or raw.get("next_task_id") != "stage0.12_delivery_and_sync"
        or not isinstance(next_inputs, list)
        or set(next_inputs) != set(ordered.values())
    ):
        raise Stage0G9Error("G9_STATE_HANDOFF_INVALID")
    return Stage0G9FormalState(
        environment=environment,
        task_output_refs=ordered,
        config=config,
        config_ref=str(raw["config_ref"]),
        environment_ref=str(raw["environment_ref"]),
        index_ref=index_ref,
        index_sha256=sha256_file(index_path),
        gate_artifact_hash=gate.artifact_hash,
        g8_index_ref=str(raw["g8_index_ref"]),
    )


__all__ = [
    "GATE_ID",
    "G9SourceBinding",
    "Stage0G9Error",
    "Stage0G9FormalState",
    "Stage0G9FormalizationResult",
    "TASK_ID",
    "build_stage0_g9_config",
    "execute_stage0_g9",
    "load_stage0_g9_formal_state",
    "run_formal_g9_task",
    "validate_formal_g9_outputs",
]
