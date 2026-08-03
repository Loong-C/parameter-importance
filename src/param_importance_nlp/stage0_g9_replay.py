"""Manifest-only independent Stage 0 G9 replay executor.

This module is launched in a fresh Python process.  Its only mutable input is a
hash-bound replay plan under ``DATA_ROOT``.  It runs the complete repository
test suite, validates the checked-in deterministic fixture, then reuses the
formal S0.9 single-GPU/four-GPU fresh-process recovery suite inside a brand-new
output root.  GPU workers run with the Python socket offline guard enabled.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any, Final, Mapping, Sequence
import xml.etree.ElementTree as ET

from .atomic import atomic_write_bytes, sha256_file
from .contracts import (
    ResolvedConfigV2,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from .contracts.jsonio import JSONValue
from .deterministic_fixture import validate_deterministic_fixture
from .offline_guard import (
    OFFLINE_GUARD_ALLOWED_HOSTS_ENV,
    OFFLINE_GUARD_AUDIT_DIR_ENV,
    OFFLINE_GUARD_ENABLE_ENV,
    OFFLINE_GUARD_SCHEMA_VERSION,
    finalize_offline_guard,
    install_offline_guard,
)
from .runtime import TaskExecutionRequest, TaskRuntimeEnvironment
from .stage0_g7_recovery import (
    G7RecoverySourceBinding,
    _run_suite as _run_g7_recovery_suite,
)


REPLAY_PLAN_SCHEMA: Final = "stage0-g9-independent-replay-plan-v1"
REPLAY_REPORT_SCHEMA: Final = "stage0-g9-independent-replay-report-v1"
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


class Stage0G9ReplayError(RuntimeError):
    """The independent replay failed or its immutable inputs drifted."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G9ReplayError(f"G9_REPLAY_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G9ReplayError(f"G9_REPLAY_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G9ReplayError(f"G9_REPLAY_LOGICAL_PATH_ESCAPE:{field}")
    path = root.joinpath(*logical.parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise Stage0G9ReplayError(f"G9_REPLAY_LOGICAL_PATH_ESCAPE:{field}") from error
    return path


def _git(repository: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", f"safe.directory={repository.as_posix()}", "-C", str(repository), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _source_binding(expected_commit: str) -> G7RecoverySourceBinding:
    repository = Path(__file__).resolve().parents[2]
    head = _git(repository, "rev-parse", "HEAD")
    branch = _git(repository, "branch", "--show-current")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    if any(item.returncode != 0 for item in (head, branch, status)):
        raise Stage0G9ReplayError("G9_REPLAY_SOURCE_GIT_PROBE_FAILED")
    observed = head.stdout.strip()
    branch_name = branch.stdout.strip()
    if (
        observed != expected_commit
        or _GIT_COMMIT_RE.fullmatch(observed) is None
        or not branch_name
        or status.stdout.strip()
    ):
        raise Stage0G9ReplayError("G9_REPLAY_SOURCE_IDENTITY_INVALID")
    return G7RecoverySourceBinding(repository, observed, branch_name)


def _load_hash_bound(path: Path, *, schema: str, field: str) -> dict[str, Any]:
    raw = _mapping(load_canonical_json(path), field=field)
    declared = raw.pop("artifact_hash", None)
    if raw.get("schema_version") != schema or declared != canonical_json_hash(raw):
        raise Stage0G9ReplayError(f"G9_REPLAY_HASH_BOUND_INPUT_INVALID:{field}")
    raw["artifact_hash"] = declared
    return raw


def _validate_plan(root: Path, plan_ref: str) -> tuple[dict[str, Any], Path, ResolvedConfigV2, TaskRuntimeEnvironment, G7RecoverySourceBinding]:
    path = _logical_path(root, plan_ref, field="plan_ref")
    plan = _load_hash_bound(path, schema=REPLAY_PLAN_SCHEMA, field="plan")
    expected = {
        "schema_version", "replay_id", "run_id_prefix", "generator_git_commit",
        "g7_config_ref", "g7_config_sha256", "g7_config_hash",
        "g7_environment_ref", "g7_environment_sha256", "g7_environment_hash",
        "test_matrix_ref", "test_matrix_sha256", "test_matrix_hash",
        "fixture_ref", "fixture_sha256", "fixture_hash", "suite_root_ref",
        "report_ref", "selected_gpu_uuids", "pytest_timeout_seconds",
        "replay_timeout_seconds", "artifact_hash",
    }
    if set(plan) != expected:
        raise Stage0G9ReplayError("G9_REPLAY_PLAN_FIELDS_INVALID")
    if not isinstance(plan.get("replay_id"), str) or not str(plan["replay_id"]).startswith("stage0-g9-"):
        raise Stage0G9ReplayError("G9_REPLAY_ID_INVALID")
    prefix = plan.get("run_id_prefix")
    if not isinstance(prefix, str) or not prefix.startswith("g9-"):
        raise Stage0G9ReplayError("G9_REPLAY_RUN_PREFIX_INVALID")
    selected = plan.get("selected_gpu_uuids")
    if (
        not isinstance(selected, list)
        or len(selected) != 4
        or len(set(selected)) != 4
        or any(not isinstance(item, str) or not item.startswith("GPU-") for item in selected)
    ):
        raise Stage0G9ReplayError("G9_REPLAY_GPU_SET_INVALID")
    for field in ("pytest_timeout_seconds", "replay_timeout_seconds"):
        value = plan.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 60:
            raise Stage0G9ReplayError(f"G9_REPLAY_TIMEOUT_INVALID:{field}")
    controls = (
        ("g7_config_ref", "g7_config_sha256"),
        ("g7_environment_ref", "g7_environment_sha256"),
        ("test_matrix_ref", "test_matrix_sha256"),
        ("fixture_ref", "fixture_sha256"),
    )
    for ref_field, sha_field in controls:
        candidate = _logical_path(root, plan[ref_field], field=ref_field)
        if not candidate.is_file() or sha256_file(candidate) != plan[sha_field]:
            raise Stage0G9ReplayError(f"G9_REPLAY_CONTROL_SHA_MISMATCH:{ref_field}")
    config = ResolvedConfigV2.from_mapping(
        _mapping(
            load_canonical_json(_logical_path(root, plan["g7_config_ref"], field="g7_config_ref")),
            field="g7_config",
        )
    )
    environment = TaskRuntimeEnvironment.from_mapping(
        _mapping(
            load_canonical_json(
                _logical_path(root, plan["g7_environment_ref"], field="g7_environment_ref")
            ),
            field="g7_environment",
        )
    )
    distributed = _mapping(config.base_config.section("distributed"), field="distributed")
    if (
        config.task_id != "stage0.09_checkpoint_and_resume"
        or config.run_intent != "formal"
        or config.config_hash != plan["g7_config_hash"]
        or environment.environment_hash != plan["g7_environment_hash"]
        or not {"stage0.G5", "stage0.G6", "stage0.G7"} <= environment.passed_gate_ids
        or distributed.get("world_size") != 4
        or distributed.get("backend") != "nccl"
        or not {"server", "cuda", "nccl", "model_assets", "data_assets"}
        <= environment.capabilities
    ):
        raise Stage0G9ReplayError("G9_REPLAY_G7_INPUT_IDENTITY_INVALID")
    suite_path = _logical_path(root, plan["suite_root_ref"], field="suite_root_ref")
    report_path = _logical_path(root, plan["report_ref"], field="report_ref")
    try:
        report_path.relative_to(suite_path)
    except ValueError as error:
        raise Stage0G9ReplayError("G9_REPLAY_REPORT_OUTSIDE_SUITE_ROOT") from error
    matrix = _load_hash_bound(
        _logical_path(root, plan["test_matrix_ref"], field="test_matrix_ref"),
        schema="stage0-g9-test-matrix-v1",
        field="test_matrix",
    )
    fixture = validate_deterministic_fixture(
        _logical_path(root, plan["fixture_ref"], field="fixture_ref")
    )
    if matrix["artifact_hash"] != plan["test_matrix_hash"] or fixture["artifact_hash"] != plan["fixture_hash"]:
        raise Stage0G9ReplayError("G9_REPLAY_MATRIX_OR_FIXTURE_HASH_MISMATCH")
    source = _source_binding(str(plan["generator_git_commit"]))
    return plan, path, config, environment, source


def _assertion_site_count(repository: Path) -> int:
    count = 0
    for path in sorted((repository / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count += sum(isinstance(node, ast.Assert) for node in ast.walk(tree))
    return count


def _parse_junit(path: Path) -> tuple[dict[str, JSONValue], list[dict[str, JSONValue]]]:
    root = ET.parse(path).getroot()
    cases: list[dict[str, JSONValue]] = []
    for case in root.iter("testcase"):
        status = "PASS"
        if case.find("failure") is not None:
            status = "FAIL"
        elif case.find("error") is not None:
            status = "ERROR"
        elif case.find("skipped") is not None:
            status = "SKIP"
        cases.append(
            {
                "test_id": f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}",
                "status": status,
                "duration_seconds": float(case.attrib.get("time", "0") or 0.0),
            }
        )
    counts = {
        "collected": len(cases),
        "passed": sum(item["status"] == "PASS" for item in cases),
        "failed": sum(item["status"] == "FAIL" for item in cases),
        "errors": sum(item["status"] == "ERROR" for item in cases),
        "skipped": sum(item["status"] == "SKIP" for item in cases),
    }
    return counts, cases


def _run_pytest(
    repository: Path,
    root: Path,
    suite_ref: str,
    *,
    timeout_seconds: int,
    allowed_skip_ids: Sequence[str],
) -> dict[str, JSONValue]:
    output_root = _logical_path(root, f"{suite_ref}/pytest", field="pytest_root")
    output_root.mkdir(parents=True, exist_ok=False)
    junit_path = output_root / "junit.xml"
    basetemp = output_root / "tmp"
    started_at = _now()
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "PYTHONPATH": str(repository / "src"),
        }
    )
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "--junitxml",
                str(junit_path),
                "--basetemp",
                str(basetemp),
            ],
            cwd=repository,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        timed_out = False
    except subprocess.TimeoutExpired as error:
        completed = subprocess.CompletedProcess(
            error.cmd,
            124,
            stdout=error.stdout or "",
            stderr=error.stderr or "",
        )
        timed_out = True
    completed_at = _now()
    if not junit_path.is_file():
        raise Stage0G9ReplayError(
            f"G9_REPLAY_PYTEST_JUNIT_MISSING:{completed.returncode}:{str(completed.stderr)[-4000:]}"
        )
    counts, cases = _parse_junit(junit_path)
    observed_skips = sorted(
        str(item["test_id"]) for item in cases if item["status"] == "SKIP"
    )
    allowed = set(allowed_skip_ids)
    unexpected_skips = sorted(set(observed_skips) - allowed)
    unused_exclusions = sorted(allowed - set(observed_skips))
    counts["allowed_platform_skipped"] = len(observed_skips) - len(unexpected_skips)
    counts["hard_skipped"] = len(unexpected_skips)
    report: dict[str, JSONValue] = {
        "schema_version": "stage0-g9-pytest-report-v1",
        "status": (
            "PASS"
            if completed.returncode == 0
            and not timed_out
            and counts["failed"] == 0
            and counts["errors"] == 0
            and counts["hard_skipped"] == 0
            else "FAIL"
        ),
        "started_at": started_at,
        "completed_at": completed_at,
        "return_code": int(completed.returncode),
        "timed_out": timed_out,
        "counts": counts,
        "declared_assertion_sites": _assertion_site_count(repository),
        "allowed_platform_skip_ids": sorted(allowed),
        "observed_platform_skip_ids": sorted(set(observed_skips) & allowed),
        "unexpected_hard_skip_ids": unexpected_skips,
        "unused_platform_exclusions": unused_exclusions,
        "junit_ref": junit_path.relative_to(root).as_posix(),
        "junit_sha256": sha256_file(junit_path),
        "stdout_sha256": __import__("hashlib").sha256(str(completed.stdout).encode()).hexdigest(),
        "stderr_sha256": __import__("hashlib").sha256(str(completed.stderr).encode()).hexdigest(),
        "stdout_tail": str(completed.stdout)[-20000:],
        "stderr_tail": str(completed.stderr)[-20000:],
        "test_cases": cases,
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_ref = f"{suite_ref}/pytest/report.json"
    write_canonical_json(_logical_path(root, report_ref, field="pytest_report_ref"), report)
    if report["status"] != "PASS":
        raise Stage0G9ReplayError("G9_REPLAY_PYTEST_NOT_PASS")
    return report


def _validate_network_audits(audit_root: Path, root: Path) -> dict[str, JSONValue]:
    files = sorted(audit_root.glob("python-network-*.json"))
    if len(files) < 7:
        raise Stage0G9ReplayError("G9_REPLAY_NETWORK_AUDIT_PROCESS_COVERAGE_INSUFFICIENT")
    refs: list[dict[str, JSONValue]] = []
    external_attempts = 0
    local_calls = 0
    for path in files:
        value = _load_hash_bound(path, schema=OFFLINE_GUARD_SCHEMA_VERSION, field="network_audit")
        attempts = value.get("external_attempts")
        if value.get("status") != "COMPLETE" or not isinstance(attempts, list):
            raise Stage0G9ReplayError("G9_REPLAY_NETWORK_AUDIT_INCOMPLETE")
        external_attempts += len(attempts)
        local_calls += int(value["allowed_local_calls"])
        refs.append(
            {
                "ref": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "pid": int(value["pid"]),
                "allowed_local_calls": int(value["allowed_local_calls"]),
            }
        )
    if external_attempts != 0:
        raise Stage0G9ReplayError("G9_REPLAY_EXTERNAL_NETWORK_ATTEMPT")
    return {
        "scope": "python_socket_layer_with_loopback_allowed",
        "hf_offline_environment": True,
        "process_count": len(files),
        "external_attempt_count": external_attempts,
        "allowed_loopback_call_count": local_calls,
        "audit_files": refs,
    }


def _layer_results(
    matrix: Mapping[str, Any],
    pytest_report: Mapping[str, Any],
    suite: Mapping[str, Any],
    network: Mapping[str, Any],
) -> list[dict[str, JSONValue]]:
    pair = _mapping(suite["pair_metrics"], field="pair_metrics")
    fault = _mapping(suite["fault_report"], field="fault_report")
    definitions = matrix.get("layers")
    if not isinstance(definitions, list):
        raise Stage0G9ReplayError("G9_REPLAY_MATRIX_LAYERS_INVALID")
    results: list[dict[str, JSONValue]] = []
    for raw in definitions:
        layer = _mapping(raw, field="matrix.layer")
        test_ids = layer.get("test_ids")
        if not isinstance(test_ids, list) or any(not isinstance(item, str) for item in test_ids):
            raise Stage0G9ReplayError("G9_REPLAY_MATRIX_TEST_IDS_INVALID")
        name = str(layer["layer"])
        measurements: dict[str, JSONValue]
        if name == "local_cpu":
            measurements = dict(pytest_report["counts"])
            passed = pytest_report["status"] == "PASS"
        elif name == "server_cpu":
            measurements = {
                "cpu_reference_status": suite["cpu_report"]["status"],
                "state_fields_exact": suite["cpu_report"]["state_fields_exact"],
            }
            passed = suite["cpu_report"]["status"] == "PASS"
        elif name == "single_gpu":
            measurements = _mapping(pair["single"], field="pair.single")
            passed = (
                measurements.get("world_size") == 1
                and measurements.get("sample_sequence_exact") is True
                and measurements.get("learning_rate_sequence_exact") is True
                and measurements.get("shared_state_hashes_exact") is True
                and measurements.get("rank_state_hashes_exact") is True
            )
            measurements["status"] = "PASS" if passed else "FAIL"
        elif name == "four_gpu":
            measurements = _mapping(pair["ddp"], field="pair.ddp")
            passed = (
                measurements.get("world_size") == 4
                and measurements.get("sample_sequence_exact") is True
                and measurements.get("learning_rate_sequence_exact") is True
                and measurements.get("shared_state_hashes_exact") is True
                and measurements.get("rank_state_hashes_exact") is True
            )
            measurements["status"] = "PASS" if passed else "FAIL"
        elif name == "fault":
            measurements = {
                "status": fault["status"],
                "rejection_count": len(fault["rejection_reasons"]),
            }
            passed = fault["status"] == "PASS" and len(fault["rejection_reasons"]) >= 5
        elif name == "replay":
            measurements = {
                "empty_output_root": True,
                "external_network_attempt_count": network["external_attempt_count"],
                "single_and_ddp_replayed": True,
                "manifest_only_subprocess": True,
            }
            passed = network["external_attempt_count"] == 0
        else:
            raise Stage0G9ReplayError(f"G9_REPLAY_UNKNOWN_LAYER:{name}")
        results.append(
            {
                "layer": name,
                "hard": bool(layer["hard"]),
                "status": "PASS" if passed else "FAIL",
                "test_ids": test_ids,
                "passed": len(test_ids) if passed else 0,
                "failed": 0 if passed else len(test_ids),
                "skipped": (
                    int(pytest_report["counts"]["hard_skipped"])
                    if name == "local_cpu"
                    else 0
                ),
                "platform_excluded": (
                    int(pytest_report["counts"]["allowed_platform_skipped"])
                    if name == "local_cpu"
                    else 0
                ),
                "measurements": measurements,
            }
        )
    if any(item["status"] != "PASS" or item["skipped"] != 0 for item in results if item["hard"]):
        raise Stage0G9ReplayError("G9_REPLAY_HARD_LAYER_NOT_PASS")
    return results


def _markdown(report: Mapping[str, Any]) -> bytes:
    lines = [
        "# Stage 0 G9 独立重放摘要",
        "",
        f"- Replay ID: `{report['replay_id']}`",
        f"- Git commit: `{report['generator_git_commit']}`",
        f"- 状态: **{report['status']}**",
        f"- Python socket 外连尝试: `{report['network_audit']['external_attempt_count']}`",
        "",
        "| 层级 | 状态 | 通过 | 失败 | 硬跳过 | 平台排除 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for layer in report["layer_results"]:
        lines.append(
            f"| {layer['layer']} | {layer['status']} | {layer['passed']} | {layer['failed']} | {layer['skipped']} | {layer['platform_excluded']} |"
        )
    lines.extend(["", "所有硬层均为 PASS；未批准的 skip 一律失败，平台排除仅按版本化矩阵逐项记录。", ""])
    return "\n".join(lines).encode("utf-8")


def run_stage0_g9_independent_replay(*, data_root: str | Path, plan_ref: str) -> dict[str, JSONValue]:
    root = Path(data_root).resolve(strict=True)
    plan, plan_path, config, environment, source = _validate_plan(root, plan_ref)
    suite_ref = str(plan["suite_root_ref"])
    suite_root = _logical_path(root, suite_ref, field="suite_root_ref")
    if suite_root.exists():
        raise Stage0G9ReplayError("G9_REPLAY_OUTPUT_ROOT_NOT_EMPTY")
    suite_root.mkdir(parents=True, exist_ok=False)
    matrix = _load_hash_bound(
        _logical_path(root, plan["test_matrix_ref"], field="test_matrix_ref"),
        schema="stage0-g9-test-matrix-v1",
        field="test_matrix",
    )
    fixture = validate_deterministic_fixture(
        _logical_path(root, plan["fixture_ref"], field="fixture_ref")
    )
    platform_key = "windows" if sys.platform == "win32" else "linux"
    exclusions = matrix.get("platform_exclusions")
    if not isinstance(exclusions, Mapping) or not isinstance(exclusions.get(platform_key), list):
        raise Stage0G9ReplayError("G9_REPLAY_PLATFORM_EXCLUSIONS_INVALID")
    pytest_report = _run_pytest(
        source.repository,
        root,
        suite_ref,
        timeout_seconds=int(plan["pytest_timeout_seconds"]),
        allowed_skip_ids=tuple(str(item) for item in exclusions[platform_key]),
    )
    audit_root = _logical_path(root, f"{suite_ref}/network-audit", field="network_audit_root")
    audit_root.mkdir(parents=True, exist_ok=False)
    guard_path = source.repository / "ops" / "stage0" / "offline_guard"
    inherited_pythonpath = os.environ.get("PYTHONPATH")
    os.environ[OFFLINE_GUARD_ENABLE_ENV] = "1"
    os.environ[OFFLINE_GUARD_AUDIT_DIR_ENV] = str(audit_root)
    os.environ[OFFLINE_GUARD_ALLOWED_HOSTS_ENV] = "localhost,127.0.0.1,::1"
    os.environ["PYTHONPATH"] = str(guard_path) + (
        os.pathsep + inherited_pythonpath if inherited_pythonpath else ""
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    install_offline_guard(audit_dir=audit_root)
    request = TaskExecutionRequest(config, config.task_definition, environment)
    replay_suite = _run_g7_recovery_suite(
        request,
        root,
        source,
        tuple(str(item) for item in plan["selected_gpu_uuids"]),
        f"{suite_ref}/recovery-replay",
        run_id_prefix=str(plan["run_id_prefix"]),
    )
    finalize_offline_guard()
    network = _validate_network_audits(audit_root, root)
    layer_results = _layer_results(matrix, pytest_report, replay_suite, network)
    report_rows = [
        value
        for _key, value in sorted(
            replay_suite["reports"].items(),
            key=lambda item: (item[0][0], item[0][1]),
        )
    ]
    lineage = [
        {
            "run_id": row["run_id"],
            "phase": row["phase"],
            "world_size": row["world_size"],
            "config_hash": row["config_hash"],
            "environment_hash": row["environment_hash"],
            "selected_gpu_uuids": row["selected_gpu_uuids"],
            "model_manifest_id": config.base_config.section("model")["asset_id"],
            "tokenizer_manifest_id": config.base_config.section("model")["tokenizer_asset_id"],
            "data_manifest_id": config.base_config.section("data")["asset_id"],
            "group_checkpoint_id": row["group_checkpoint_id"],
            "report_ref": next(
                reference
                for reference in replay_suite["report_refs"]
                if load_canonical_json(_logical_path(root, reference, field="lineage.report"))["artifact_hash"]
                == row["artifact_hash"]
            ),
        }
        for row in report_rows
    ]
    completed_at = _now()
    report: dict[str, JSONValue] = {
        "schema_version": REPLAY_REPORT_SCHEMA,
        "status": "PASS",
        "replay_id": str(plan["replay_id"]),
        "generator_git_commit": source.git_commit,
        "plan_ref": plan_path.relative_to(root).as_posix(),
        "plan_sha256": sha256_file(plan_path),
        "started_from_empty_output_root": True,
        "completed_at": completed_at,
        "g7_config_hash": config.config_hash,
        "g7_environment_hash": environment.environment_hash,
        "test_matrix_hash": matrix["artifact_hash"],
        "fixture_hash": fixture["artifact_hash"],
        "fixture_validation": {
            "status": "PASS",
            "fixture_id": fixture["fixture_id"],
            "initial_state_sha256": fixture["model"]["initial_state_sha256"],
            "gradient_state_sha256": fixture["expected"]["gradient_state_sha256"],
            "mean_loss": fixture["expected"]["mean_loss"],
            "tolerances": fixture["tolerances"],
        },
        "pytest_report": pytest_report,
        "layer_results": layer_results,
        "network_audit": network,
        "recovery_pair_metrics": replay_suite["pair_metrics"],
        "cpu_reference": replay_suite["cpu_report"],
        "fault_summary": replay_suite["fault_report"],
        "retention_summary": replay_suite["retention_report"],
        "run_lineage": lineage,
        "evidence_refs": {
            "worker_reports": list(replay_suite["report_refs"]),
            "boundaries": list(replay_suite["boundary_refs"]),
            "transcripts": list(replay_suite["transcript_refs"]),
            "cpu_report": replay_suite["cpu_ref"],
            "fault_report": replay_suite["fault_ref"],
            "retention_report": replay_suite["retention_ref"],
        },
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_path = _logical_path(root, plan["report_ref"], field="report_ref")
    if report_path.exists():
        raise Stage0G9ReplayError("G9_REPLAY_REPORT_COLLISION")
    write_canonical_json(report_path, report)
    markdown_path = report_path.with_suffix(".md")
    atomic_write_bytes(markdown_path, _markdown(report))
    return report


__all__ = [
    "REPLAY_PLAN_SCHEMA",
    "REPLAY_REPORT_SCHEMA",
    "Stage0G9ReplayError",
    "run_stage0_g9_independent_replay",
]
