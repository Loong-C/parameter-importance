"""CPU contracts for the S1.11 immutable exit-gate consumer."""

from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.stage1_exit_gate import (
    DEPENDENCIES,
    REQUIRED_CHARTS,
    REQUIREMENT_GATE_IDS,
    REQUIREMENT_IDS,
    Stage1ExitGateError,
    build_exit_gate_evidence,
    build_exit_gate_summary,
    replay_exit_gate_summary,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _formalizer() -> object:
    path = Path("ops/stage1/formalize_s1_11.py")
    spec = importlib.util.spec_from_file_location("s111_formalizer_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _with_hash(body: dict[str, object]) -> dict[str, object]:
    return {**body, "artifact_hash": canonical_json_hash(body)}


def _publish_dependency(root: Path, ordinal: int, gate_id: str, task_id: str) -> dict[str, str]:
    """Write a role/index-shaped immutable publication, never a formal claim."""

    published = root / f"publication-{ordinal:02d}"
    published.mkdir()
    index_wires = {
        "G1-REGISTRY": ("stage1-s1-2-formalization-index-v2", {"named": "registry-report.json"}),
        "G1-ORACLE": ("stage1-s1-3-formalization-index-v2", {"named": "oracle-validation-report.json"}),
        "G1-GRAD": ("stage1-s1-4-formalization-index-v1", {"named": "g1-grad-record.json"}),
        "G1-EST": ("stage1-s1-5-formalization-index-v1", {"estimator_report": "estimator-report.json", "oracle_report": "oracle-report.json", "tensor_bundle": "tensor-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-est-record.json"}),
        "G1-STEP": ("stage1-s1-6-formalization-index-v1", {"step_report": "step-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-step-record.json"}),
        "G1-SINGLE": ("stage1-s1-7-formalization-index-v1", {"fixture_manifest": "fixture-manifest.json", "single_gpu_report": "worker-report.json", "gradient_bundle": "arrays-manifest.json", "comparison_table": "comparison-table.json", "gate_record": "g1-single-record.json"}),
        "G1-DDP": ("stage1-s1-8-formalization-index-v2", {"fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json", "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-ddp-record.json"}),
        "G1-NUMERIC": ("stage1-s1-9-formalization-index-v5", {"numeric_report": "numeric-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-numeric-record.json"}),
        "G1-RESUME": ("stage1-s1-10-formalization-index-v1", {"resume_report": "resume-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "artifact_manifest": "artifact-manifest.json", "gate_record": "g1-resume-record.json"}),
    }
    schema_version, role_files = index_wires[gate_id]
    role_name = role_files["gate_record"] if "gate_record" in role_files else role_files["named"]
    replay_name = "replay-validation.json"
    gate = _with_hash({
        "schema_version": "synthetic-gate-record-v1", "status": "PASS", "gate_id": gate_id,
        "task_id": task_id, "requirements": {"independent_measurement": True, "negative_control": True},
    })
    replay = _with_hash({"schema_version": "synthetic-replay-v1", "status": "PASS", "task_id": task_id})
    validation = _with_hash({"schema_version": "synthetic-validation-v1", "status": "PASS", "task_id": task_id})
    write_canonical_json(published / role_name, gate)
    for role, filename in role_files.items():
        if filename != role_name:
            write_canonical_json(published / filename, _with_hash({"schema_version": "synthetic-bound-role-v1", "status": "PASS", "role": role}))
    write_canonical_json(published / replay_name, replay)
    write_canonical_json(published / "validation.json", validation)
    role_sha = {role: _sha(published / filename) for role, filename in role_files.items()}
    index_body: dict[str, object] = {
        "schema_version": schema_version, "status": "PASS", "gate_id": gate_id,
        "task_id": task_id, "generator_git_commit": f"{ordinal + 1:040x}",
        "consumer_git_commit": f"{ordinal + 1:040x}",
        "role_refs": role_files, "role_sha256": role_sha, "gate_artifact_hash": gate["artifact_hash"],
        "replay_ref": replay_name, "replay_sha256": _sha(published / replay_name),
        "validation_ref": "validation.json", "validation_sha256": _sha(published / "validation.json"),
        "next_task_ids": ["synthetic-next-task"],
    }
    if gate_id == "G1-REGISTRY":
        index_body.update({"report_ref": role_name, "report_sha256": role_sha["named"]})
    elif gate_id == "G1-ORACLE":
        index_body.update({"oracle_validation_report_ref": role_name, "oracle_validation_report_sha256": role_sha["named"]})
    elif gate_id == "G1-GRAD":
        index_body.update({"gate_record_ref": role_name, "gate_record_sha256": role_sha["named"]})
    elif gate_id == "G1-NUMERIC":
        reproduction_refs = {
            "attempt_start": "attempt-start.json", "upstream_compatibility": "upstream-compatibility.json",
            "preflight": "preflight.json", "prelease_gpu": "prelease-gpu.json",
            "post_worker_quiescence": "post-worker-quiescence.json", "lease_history": "lease-history.json",
            "single_worker": "single-bf16.json", "single_stdout": "single.stdout.txt",
            "single_stderr": "single.stderr.txt", "single_child_fingerprint": "single-child-fingerprint.json",
            "bf16_resume_checkpoint_store": "bf16-resume-store-index.json", "ddp_worker": "ddp-skip.json",
            "ddp_stdout": "ddp.stdout.txt", "ddp_stderr": "ddp.stderr.txt",
            "ddp_child_fingerprint": "ddp-child-fingerprint.json",
            **{f"chart_csv_{index}": name for index, name in enumerate(("bf16-fp32-heatmap.csv", "clip-norm-factor.csv", "skip-zero-difference.csv", "t-amp-scale.csv", "u-single-factor-identity.csv", "u-single-factor-ratio-diagnostic.csv"))},
            **{f"chart_svg_{index}": name for index, name in enumerate(("bf16-fp32-heatmap.svg", "clip-norm-factor.svg", "skip-zero-difference.svg", "t-amp-scale.svg", "u-single-factor-identity.svg", "u-single-factor-ratio-diagnostic.svg"))},
        }
        for name in reproduction_refs.values():
            path = published / name
            if name == "upstream-compatibility.json":
                write_canonical_json(path, _with_hash({"schema_version": "stage1-s1-9-upstream-compatibility-v4", "status": "PASS", "s1_8_v5_handoff": {"index_schema_version": "stage1-s1-8-formalization-index-v5"}}))
            elif name == "prelease-gpu.json":
                quiescence = _with_hash({"schema_version": "stage1-s1-9-gpu-quiescence-v3", "status": "PASS", "phase": "prelease"})
                write_canonical_json(path, _with_hash({"schema_version": "stage1-s1-9-gpu-prelease-v3", "status": "PASS", "quiescence": quiescence}))
            elif name == "post-worker-quiescence.json":
                write_canonical_json(path, _with_hash({"schema_version": "stage1-s1-9-gpu-quiescence-v3", "status": "PASS", "phase": "post_worker"}))
            elif name.endswith(".json"):
                write_canonical_json(path, _with_hash({"schema_version": "synthetic-s1-9-reproduction-v1", "status": "PASS", "role": name}))
            else:
                path.write_text(name + "\n", encoding="utf-8", newline="\n")
        index_body.update({
            "reproduction_role_refs": reproduction_refs,
            "reproduction_role_sha256": {role: _sha(published / name) for role, name in reproduction_refs.items()},
            "fixture_id": "stage1-s19-precision-fixture-v1", "git_branch": "fixture",
            "checked_at": "2026-08-20T00:00:00+00:00", "s1_7_handoff": {}, "s1_8_handoff": {},
            "csv_sha256": {name: _sha(published / name) for name in ("bf16-fp32-heatmap.csv", "clip-norm-factor.csv", "skip-zero-difference.csv", "t-amp-scale.csv", "u-single-factor-identity.csv", "u-single-factor-ratio-diagnostic.csv")},
            "svg_sha256": {name: _sha(published / name) for name in ("bf16-fp32-heatmap.svg", "clip-norm-factor.svg", "skip-zero-difference.svg", "t-amp-scale.svg", "u-single-factor-identity.svg", "u-single-factor-ratio-diagnostic.svg")},
            "replay_hash": replay["artifact_hash"],
            "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"],
        })
    index = _with_hash(index_body)
    write_canonical_json(published / "index.json", index)
    return {
        "gate_id": gate_id, "task_id": task_id,
        "index_ref": (published / "index.json").relative_to(root).as_posix(),
        "index_schema_version": index["schema_version"], "index_sha256": _sha(published / "index.json"), "index_artifact_hash": index["artifact_hash"],
        "producer_commit": index["generator_git_commit"], "gate_role": {"G1-REGISTRY": "report", "G1-ORACLE": "oracle_validation_report", "G1-GRAD": "gate_record"}.get(gate_id, "gate_record"), "gate_ref": role_name, "gate_schema_version": gate["schema_version"],
        "gate_sha256": role_sha["gate_record"] if "gate_record" in role_sha else role_sha["named"], "gate_artifact_hash": gate["artifact_hash"],
    }


def _publish_shared_s11(root: Path) -> list[dict[str, str]]:
    """Use S1.1's real shared-index wire for ENTRY and CONTRACT."""

    from param_importance_nlp.contracts.status import GateRecord, GateStatus
    from param_importance_nlp.contracts.task_catalog import RecoveryMode, RunnerKind
    from param_importance_nlp.runtime import TaskRunResult, TaskRunStatus

    published = root / "evidence" / "stage1" / "s1-1" / "formal"; published.mkdir(parents=True)
    inputs = root / "evidence" / "stage1" / "s1-1" / "inputs"; inputs.mkdir(parents=True)
    config_ref = "evidence/stage1/s1-1/inputs/config.json"
    environment_ref = "evidence/stage1/s1-1/inputs/environment.json"
    result_ref = "evidence/stage1/s1-1/inputs/result.json"
    write_canonical_json(root / config_ref, {"schema_version": "s1-1-fixture-config-v1", "scope": "formal"})
    write_canonical_json(root / environment_ref, {"schema_version": "s1-1-fixture-environment-v1", "scope": "formal"})
    gate = {"schema_version": "stage1-s1-1-gate-aggregate-v1", "status": "PASS", "gates": ["G1-ENTRY", "G1-CONTRACT"]}
    write_canonical_json(published / "gate-record.json", gate)
    for name in ("contract.json", "matrix.json"):
        write_canonical_json(published / name, {"schema_version": "s1-1-fixture-role-v1", "status": "PASS"})
    formal_gates = [
        GateRecord(gate_id=gate_id, stage=1, status=GateStatus.PASS, checked_at="2026-01-01T00:00:00Z", measured={"fixture": True}, threshold={"fixture": True}, evidence_refs=("evidence/stage1/s1-1/inputs/config.json",)).to_dict()
        for gate_id in ("stage1.G1-ENTRY", "stage1.G1-CONTRACT")
    ]
    hashes = {str(item["gate_id"]): str(item["artifact_hash"]) for item in formal_gates}
    role_refs = {"stage_contract": "evidence/stage1/s1-1/formal/contract.json", "gate_record": "evidence/stage1/s1-1/formal/gate-record.json", "requirements_matrix": "evidence/stage1/s1-1/formal/matrix.json"}
    result = TaskRunResult(task_id="stage1.01_entry_and_contract", stage=1, runner_kind=RunnerKind.CONTRACT, run_intent="formal", status=TaskRunStatus.PASS, config_hash="6" * 64, formal_eligible=True, artifact_refs=role_refs, checkpoint_ref=None, blockers=(), error_code=None, message="fixture", recovery_mode=RecoveryMode.RESTART_IDEMPOTENT, metadata={"formal_gate_records": formal_gates})
    write_canonical_json(root / result_ref, result.to_dict())
    index = _with_hash({
        "schema_version": "stage1-s1-1-formalization-index-v1", "status": "PASS", "generator_git_commit": "3" * 40,
        "git_branch": "fixture", "checked_at": "2026-01-01T00:00:00Z", "g10_generator_git_commit": "4" * 40,
        "g10_index_ref": "g10.json", "g10_index_sha256": "5" * 64, "g10_legacy_index": False,
        "reuse_attestation_ref": None, "reuse_attestation_sha256": None, "reuse_artifact_hash": None,
        "config_ref": config_ref, "config_hash": result.config_hash,
        "environment_ref": environment_ref, "environment_hash": "7" * 64,
        "result_ref": result_ref, "result_hash": result.result_hash,
        "task_output_refs": role_refs,
        "gate_artifact_hashes": hashes, "next_task_id": "stage1.02_architecture_and_parameter_registry",
    })
    write_canonical_json(published / "index.json", index)
    base = {"index_ref": (published / "index.json").relative_to(root).as_posix(), "index_schema_version": index["schema_version"], "index_sha256": _sha(published / "index.json"), "index_artifact_hash": index["artifact_hash"], "producer_commit": index["generator_git_commit"], "gate_role": "gate_record", "gate_ref": role_refs["gate_record"], "gate_schema_version": gate["schema_version"], "gate_sha256": _sha(published / "gate-record.json")}
    return [{**base, "gate_id": gate_id, "task_id": task_id, "gate_artifact_hash": hashes[f"stage1.{gate_id}"]} for gate_id, task_id, _role in DEPENDENCIES[:2]]


def _inputs(root: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    dependencies = _publish_shared_s11(root) + [
        _publish_dependency(root, ordinal, gate_id, task_id)
        for ordinal, (gate_id, task_id, _expected_role) in enumerate(DEPENDENCIES[2:], start=2)
    ]
    charts: dict[str, dict[str, str]] = {}
    chart_root = root / "charts"; chart_root.mkdir()
    for chart_id in REQUIRED_CHARTS:
        csv, svg = chart_root / f"{chart_id}.csv", chart_root / f"{chart_id}.svg"
        csv.write_text("step,value\n0,0\n1,1\n", encoding="utf-8", newline="\n")
        svg.write_text("<svg xmlns=\"http://www.w3.org/2000/svg\"><polyline points=\"0,1 1,0\"/></svg>\n", encoding="utf-8", newline="\n")
        charts[chart_id] = {
            "csv_ref": csv.relative_to(root).as_posix(), "csv_sha256": _sha(csv),
            "svg_ref": svg.relative_to(root).as_posix(), "svg_sha256": _sha(svg),
        }
    return dependencies, charts


def test_s111_rehearsal_is_not_run_but_offline_replays_immutable_roles(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    summary = build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=charts)
    assert summary["status"] == "NOT_RUN"
    assert summary["exit_verdict"] == "BLOCKED_FORMAL_OBSERVATION_MISSING"
    assert len(summary["dependency_audits"]) == 11
    assert replay_exit_gate_summary(tmp_path, summary)["status"] == "PASS"


def test_s111_library_rejects_a_caller_self_declared_formal_observation(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    observation = _with_hash({
        "schema_version": "stage1-s1-11-formal-observation-v1", "status": "PASS",
        "task_id": "stage1.11_reporting_and_exit_gate", "gate_id": "G1-EXIT",
    })
    with pytest.raises(Stage1ExitGateError, match="FORMAL_OBSERVATION_CALLER_SUPPLIED"):
        build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=charts, formal_observation=observation)


def test_s111_builds_all_28_measured_rows_but_cannot_claim_formal_pass(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    rows = [
        {
            "requirement_id": requirement_id, "gate_id": item["gate_id"], "measured": "independent immutable evidence verified",
            "threshold": "all bound hashes and requirements pass", "status": "PASS",
            "evidence": {"ref": item["index_ref"], "sha256": item["index_sha256"]},
        }
        for requirement_id, expected_gate_id in zip(REQUIREMENT_IDS, REQUIREMENT_GATE_IDS, strict=True)
        for item in (next(binding for binding in dependencies if binding["gate_id"] == expected_gate_id),)
    ]
    evidence = build_exit_gate_evidence(
        tmp_path, dependencies, verification_matrix=rows, unresolved_failures=[], charts=charts,
    )
    assert evidence["requirements_matrix"]["rows"] == rows
    assert evidence["gate_summary"]["status"] == "NOT_RUN"
    assert evidence["stage_report"]["status"] == "NOT_RUN"
    assert set(evidence) == {"requirements_matrix", "gate_summary", "stage_report", "delivery_manifest"}


def test_s111_rejects_repeated_requirement_wrong_chart_wire_and_s11_result_gate_tamper(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    rows = [
        {"requirement_id": requirement_id, "gate_id": gate_id, "measured": "bound", "threshold": "bound", "status": "PASS", "evidence": {"ref": next(item for item in dependencies if item["gate_id"] == gate_id)["index_ref"], "sha256": next(item for item in dependencies if item["gate_id"] == gate_id)["index_sha256"]}}
        for requirement_id, gate_id in zip(REQUIREMENT_IDS, REQUIREMENT_GATE_IDS, strict=True)
    ]
    rows[1] = {**rows[1], "requirement_id": "S1.11-R01"}
    with pytest.raises(Stage1ExitGateError, match="VERIFICATION_MATRIX_STATUS_INVALID"):
        build_exit_gate_evidence(tmp_path, dependencies, verification_matrix=rows, unresolved_failures=[], charts=charts)
    wrong_charts = dict(charts)
    wrong_charts["u-identity"] = {**wrong_charts["u-identity"], "csv_ref": wrong_charts["gradient-identity"]["csv_ref"]}
    with pytest.raises(Stage1ExitGateError, match="CHART_REF_WIRE_INVALID"):
        build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=wrong_charts)
    index_path = tmp_path / dependencies[0]["index_ref"]
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    index = load_canonical_json(index_path); assert isinstance(index, dict)
    result_path = tmp_path / index["result_ref"]
    result = load_canonical_json(result_path); assert isinstance(result, dict)
    records = result["metadata"]["formal_gate_records"]
    records[0]["artifact_hash"] = "0" * 64
    result_body = {key: value for key, value in result.items() if key != "result_hash"}
    result["result_hash"] = canonical_json_hash(result_body)
    write_canonical_json(result_path, result)
    index_body = {key: value for key, value in index.items() if key != "artifact_hash"}
    index_body["result_hash"] = result["result_hash"]
    altered_index = _with_hash(index_body); write_canonical_json(index_path, altered_index)
    broken = list(dependencies)
    for position in (0, 1):
        broken[position] = {**broken[position], "index_sha256": _sha(index_path), "index_artifact_hash": altered_index["artifact_hash"]}
    with pytest.raises(Stage1ExitGateError, match="S1_1_FORMAL_GATE_RECORDS_INVALID"):
        build_exit_gate_summary(tmp_path, broken, unresolved_failures=[], charts=charts)


def test_s111_missing_future_s19_or_s110_index_fails_closed(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    with pytest.raises(Stage1ExitGateError, match="DEPENDENCY_COUNT_INVALID"):
        build_exit_gate_summary(tmp_path, dependencies[:-1], unresolved_failures=[], charts=charts)
    broken = list(dependencies)
    broken[9] = {**broken[9], "index_ref": "missing/s1-9-index.json"}
    with pytest.raises(Stage1ExitGateError, match="INDEX_FILE_HASH_MISMATCH"):
        build_exit_gate_summary(tmp_path, broken, unresolved_failures=[], charts=charts)


def test_s111_requires_final_s19_v5_reproduction_and_source_closure(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    binding = dependencies[9]
    index_path = tmp_path / binding["index_ref"]
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    index = load_canonical_json(index_path)
    assert isinstance(index, dict)

    def resign(body: dict[str, object]) -> dict[str, object]:
        unsigned = {key: value for key, value in body.items() if key != "artifact_hash"}
        updated = _with_hash(unsigned)
        write_canonical_json(index_path, updated)
        return {**binding, "index_sha256": _sha(index_path), "index_artifact_hash": updated["artifact_hash"]}

    legacy = dict(index)
    legacy["schema_version"] = "stage1-s1-9-formalization-index-v1"
    legacy_binding = resign(legacy)
    legacy_binding["index_schema_version"] = "stage1-s1-9-formalization-index-v1"
    with pytest.raises(Stage1ExitGateError, match="UNSUPPORTED_INDEX_WIRE_VERSION"):
        build_exit_gate_summary(tmp_path, [*dependencies[:9], legacy_binding, *dependencies[10:]], unresolved_failures=[], charts=charts)

    write_canonical_json(index_path, index)
    missing = dict(index)
    refs = dict(missing["reproduction_role_refs"])
    refs.pop("prelease_gpu")
    missing["reproduction_role_refs"] = refs
    with pytest.raises(Stage1ExitGateError, match="S1_9_V5_REPRODUCTION_REF_WIRE_INVALID"):
        build_exit_gate_summary(tmp_path, [*dependencies[:9], resign(missing), *dependencies[10:]], unresolved_failures=[], charts=charts)

    write_canonical_json(index_path, index)
    missing_schema_field = dict(index)
    missing_schema_field.pop("fixture_id")
    with pytest.raises(Stage1ExitGateError, match="S1_9_V5_INDEX_SCHEMA_CLOSURE_INVALID"):
        build_exit_gate_summary(tmp_path, [*dependencies[:9], resign(missing_schema_field), *dependencies[10:]], unresolved_failures=[], charts=charts)

    write_canonical_json(index_path, index)
    source = tmp_path / binding["index_ref"].replace("index.json", "upstream-compatibility.json")
    source.write_text('{"schema_version":"tampered"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(Stage1ExitGateError, match="S1_9_V5_REPRODUCTION_FILE_HASH_MISMATCH"):
        build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=charts)

    write_canonical_json(index_path, index)
    write_canonical_json(source, _with_hash({
        "schema_version": "stage1-s1-9-upstream-compatibility-v4", "status": "PASS",
        "s1_8_v5_handoff": {"index_schema_version": "stage1-s1-8-formalization-index-v4"},
    }))
    invalid_source = dict(index)
    invalid_source["reproduction_role_sha256"] = dict(index["reproduction_role_sha256"])
    invalid_source["reproduction_role_sha256"]["upstream_compatibility"] = _sha(source)
    invalid_source_binding = resign(invalid_source)
    with pytest.raises(Stage1ExitGateError, match="S1_9_V5_UPSTREAM_SOURCE_CLOSURE_INVALID"):
        build_exit_gate_summary(tmp_path, [*dependencies[:9], invalid_source_binding, *dependencies[10:]], unresolved_failures=[], charts=charts)


def test_s111_rejects_self_consistent_text_without_bound_role_hash(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    changed = list(dependencies)
    changed[8] = {**changed[8], "gate_sha256": "0" * 64}
    with pytest.raises(Stage1ExitGateError, match="GATE_ROLE_BINDING_MISMATCH"):
        build_exit_gate_summary(tmp_path, changed, unresolved_failures=[], charts=charts)
    with pytest.raises(Stage1ExitGateError, match="UNRESOLVED_FAILURES_PRESENT"):
        build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[{"id": "r17", "state": "open"}], charts=charts)
    incomplete_charts = dict(charts); incomplete_charts.pop("noise-smoke")
    with pytest.raises(Stage1ExitGateError, match="REQUIRED_CHART_SET_INVALID"):
        build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=incomplete_charts)


def test_s111_replay_detects_chart_and_summary_drift(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    summary = build_exit_gate_summary(tmp_path, dependencies, unresolved_failures=[], charts=charts)
    chart = tmp_path / charts["u-identity"]["csv_ref"]
    chart.write_text("x,y\n1,1\n", encoding="utf-8", newline="\n")
    with pytest.raises(Stage1ExitGateError, match="CHART_FILE_HASH_MISMATCH"):
        replay_exit_gate_summary(tmp_path, summary)


def test_s111_formalizer_preflight_never_publishes_from_missing_upstream(tmp_path: Path) -> None:
    formalizer = _formalizer()
    configs = tmp_path / "configs"; configs.mkdir()
    for name, value in (("dependencies.json", []), ("matrix.json", []), ("charts.json", []), ("failures.json", [])):
        write_canonical_json(configs / name, value)
    write_canonical_json(configs / "test-binding.json", {"ref": "missing-test-summary.json", "sha256": "0" * 64})
    write_canonical_json(configs / "sync-binding.json", {"ref": "missing-sync-audit.json", "sha256": "0" * 64})
    attempt_root = tmp_path / "attempts"
    with pytest.raises(formalizer.Stage1S111FormalError, match="TEST_SUMMARY_HASH_MISMATCH"):
        formalizer.execute(
            repository=Path.cwd(), evidence_root=tmp_path, attempt_root=attempt_root,
            dependencies_path=configs / "dependencies.json", matrix_path=configs / "matrix.json",
            chart_specs_path=configs / "charts.json", test_summary_binding_path=configs / "test-binding.json",
            sync_audit_binding_path=configs / "sync-binding.json", failure_history_path=configs / "failures.json",
            execution_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            attempt_id="preflight-missing-upstream",
        )
    assert not attempt_root.exists()
    assert not (tmp_path / "stage1" / "s1-11-formal").exists()


def test_s111_formalizer_strict_schema_rejects_nested_missing_role_fields() -> None:
    formalizer = _formalizer()
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:gate_summary"):
        formalizer._schema_validate(
            Path.cwd(), {"gate_summary": {"schema_version": "stage1-s1-11-gate-summary-v1"}},
        )
    malformed_matrix = _with_hash({
        "schema_version": "stage1-s1-11-requirements-matrix-v1", "task_id": "stage1.11_reporting_and_exit_gate",
        "gate_id": "G1-EXIT", "rows": [{"requirement_id": "S1.11-R01", "gate_id": "G1-DDP", "measured": 0.0, "threshold": "0", "status": "PASS", "evidence": {"ref": "index.json"}}] * 28,
    })
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:requirements_matrix"):
        formalizer._schema_validate(Path.cwd(), {"requirements_matrix": malformed_matrix})


def test_s111_chart_renderer_requires_real_numeric_csv_and_reads_back(tmp_path: Path) -> None:
    formalizer = _formalizer()
    source = tmp_path / "source.csv"
    source.write_text("step,error\n1,0.25\n2,0.125\n", encoding="utf-8", newline="\n")
    record = formalizer._render_chart(
        source, chart_id="gradient-identity", x_column="step", y_column="error",
        output_csv=tmp_path / "derived.csv", output_svg=tmp_path / "derived.svg",
    )
    assert record["row_count"] == 2
    svg = (tmp_path / "derived.svg").read_text(encoding="utf-8")
    assert 'class="identity-line"' in svg
    assert svg.count('class="point"') == 2
    source.write_text("step,error\n", encoding="utf-8", newline="\n")
    with pytest.raises(formalizer.Stage1S111FormalError, match="CHART_SOURCE_EMPTY"):
        formalizer._render_chart(
            source, chart_id="gradient-identity", x_column="step", y_column="error",
            output_csv=tmp_path / "again.csv", output_svg=tmp_path / "again.svg",
        )


def test_s111_chart_geometry_is_typed_and_rejects_wrong_columns_nan_and_duplicate_keys(tmp_path: Path) -> None:
    formalizer = _formalizer()
    cases = (
        ("u-identity", "x,y\n0,0\n1,1\n", "x", "y", None, None, 'class="identity-line"', 'class="point"'),
        ("module-metric-heatmap", "layer,metric,value\nembed,cosine,0.5\nblock,cosine,1.0\n", "layer", "metric", "value", None, 'class="heatmap"', 'class="heat-cell"'),
        ("noise-smoke", "step,se,error\n0,0.2,0.1\n1,0.3,0.2\n", "step", "se", None, "error", 'class="series"', 'class="error-bar"'),
        ("weighted-reduction", "step,value\n0,0.2\n1,0.1\n", "step", "value", None, None, 'class="series"', "polyline"),
    )
    for ordinal, (chart_id, contents, x_col, y_col, value_col, error_col, mark_a, mark_b) in enumerate(cases):
        source = tmp_path / f"typed-{ordinal}.csv"
        source.write_text(contents, encoding="utf-8", newline="\n")
        formalizer._render_chart(source, chart_id=chart_id, x_column=x_col, y_column=y_col, value_column=value_col, error_column=error_col, output_csv=tmp_path / f"typed-{ordinal}.derived.csv", output_svg=tmp_path / f"typed-{ordinal}.svg")
        svg = (tmp_path / f"typed-{ordinal}.svg").read_text(encoding="utf-8")
        assert mark_a in svg and mark_b in svg
    malformed = tmp_path / "malformed.csv"
    malformed.write_text("layer,metric,value\nembed,cosine,nan\n", encoding="utf-8", newline="\n")
    with pytest.raises(formalizer.Stage1S111FormalError, match="CHART_SOURCE_NUMERIC_DATA_INVALID"):
        formalizer._render_chart(malformed, chart_id="module-metric-heatmap", x_column="layer", y_column="metric", value_column="value", output_csv=tmp_path / "bad.csv", output_svg=tmp_path / "bad.svg")
    malformed.write_text("step,value\n0,0.1\n0,0.1\n", encoding="utf-8", newline="\n")
    with pytest.raises(formalizer.Stage1S111FormalError, match="CHART_SOURCE_KEY_DUPLICATE"):
        formalizer._render_chart(malformed, chart_id="weighted-reduction", x_column="step", y_column="value", output_csv=tmp_path / "duplicate.csv", output_svg=tmp_path / "duplicate.svg")
    heat = tmp_path / "heat.csv"
    heat.write_text("layer,metric,value\nembed,cosine,0.5\n", encoding="utf-8", newline="\n")
    formalizer._render_chart(heat, chart_id="module-metric-heatmap", x_column="layer", y_column="metric", value_column="value", output_csv=tmp_path / "heat.derived.csv", output_svg=tmp_path / "heat.svg")
    tampered_svg = (tmp_path / "heat.svg").read_text(encoding="utf-8").replace('class="heat-cell"', 'class="series"')
    with pytest.raises(formalizer.Stage1S111FormalError, match="CHART_READBACK_INVALID"):
        formalizer._verify_chart_geometry(tampered_svg, chart_id="module-metric-heatmap", kind="heatmap", points=[("embed", "cosine", 0.5, None)], coordinates=None, heat_cells=[(40.0, 40.0, 520.0, 300.0, 255)])


def test_s111_standard_s17_s18_s19_s110_wires_reject_role_field_drift(tmp_path: Path) -> None:
    dependencies, charts = _inputs(tmp_path)
    # The positive fixture set includes the current S1.7/S1.8-v2/S1.9/S1.10
    # producer role maps.  Re-signing a changed index must still fail because
    # a consumer cannot substitute another producer role name.
    for index_position in (7, 8, 9, 10):
        broken = list(dependencies)
        binding = broken[index_position]
        index_path = tmp_path / binding["index_ref"]
        from param_importance_nlp.contracts.jsonio import load_canonical_json
        index = load_canonical_json(index_path)
        assert isinstance(index, dict)
        body = {key: value for key, value in index.items() if key != "artifact_hash"}
        refs = dict(body["role_refs"])
        first_role = next(iter(refs))
        refs[first_role] = "field-drift.json"
        body["role_refs"] = refs
        altered = _with_hash(body)
        write_canonical_json(index_path, altered)
        broken[index_position] = {**binding, "index_sha256": _sha(index_path), "index_artifact_hash": altered["artifact_hash"]}
        with pytest.raises(Stage1ExitGateError, match="INDEX_ROLE_REF_WIRE_INVALID"):
            build_exit_gate_summary(tmp_path, broken, unresolved_failures=[], charts=charts)
        # Restore before perturbing the next independent producer index.
        write_canonical_json(index_path, index)


def test_s111_sync_audit_requires_the_current_six_agent_documents(tmp_path: Path) -> None:
    formalizer = _formalizer()
    audit = _with_hash({
        "schema_version": "stage1-s1-11-sync-audit-v1", "status": "PASS",
        "git_publish": {}, "server_execution": {},
        "agent_sha256": {name: "a" * 64 for name in ("remote_access.md", "server.md", "git.md", "sync.md", "worklogs.md")},
        "large_artifact_manifest": {},
    })
    path = tmp_path / "sync-audit.json"; write_canonical_json(path, audit)
    with pytest.raises(formalizer.Stage1S111FormalError, match="SYNC_AUDIT_(SHAPE|CLOSURE)_INVALID"):
        formalizer._validate_sync_audit(tmp_path, {"ref": path.name, "sha256": _sha(path)})


def test_s111_test_summary_rejects_nested_type_and_key_cardinality_drift(tmp_path: Path) -> None:
    formalizer = _formalizer()
    junit = tmp_path / "cpu.xml"; junit.write_text("<testsuite/>", encoding="utf-8", newline="\n")
    summary = _with_hash({
        "schema_version": "stage1-s1-11-test-summary-v1", "status": "PASS",
        "groups": [{"name": "cpu", "collected": 1, "passed": 1, "failed": 0, "errors": 0, "skipped": 0, "duration_seconds": "wrong-type", "junit_ref": junit.name, "junit_sha256": _sha(junit)}],
    })
    path = tmp_path / "test-summary.json"; write_canonical_json(path, summary)
    with pytest.raises(formalizer.Stage1S111FormalError, match="TEST_SUMMARY_GROUP_COUNTS_INVALID"):
        formalizer._validate_test_summary(tmp_path, {"ref": path.name, "sha256": _sha(path)})
    summary["groups"][0]["unexpected"] = "extra"
    body = {key: value for key, value in summary.items() if key != "artifact_hash"}
    write_canonical_json(path, _with_hash(body))
    with pytest.raises(formalizer.Stage1S111FormalError, match="TEST_SUMMARY_GROUP_INVALID"):
        formalizer._validate_test_summary(tmp_path, {"ref": path.name, "sha256": _sha(path)})


def test_s111_historical_failure_requires_a_bound_pass_resolution(tmp_path: Path) -> None:
    formalizer = _formalizer()
    failure = _with_hash({"schema_version": "synthetic-failure-v1", "status": "FAILED"})
    failure_path = tmp_path / "failure.json"; write_canonical_json(failure_path, failure)
    resolution = _with_hash({"schema_version": "synthetic-resolution-v1", "status": "PASS", "failure_sha256": _sha(failure_path)})
    resolution_path = tmp_path / "resolution.json"; write_canonical_json(resolution_path, resolution)
    rows = formalizer._validate_failure_history(tmp_path, [{
        "failure_ref": failure_path.name, "failure_sha256": _sha(failure_path),
        "resolution_ref": resolution_path.name, "resolution_sha256": _sha(resolution_path),
    }])
    assert len(rows) == 1
    bad = dict(rows[0]); bad["resolution_sha256"] = "0" * 64
    with pytest.raises(formalizer.Stage1S111FormalError, match="FAILURE_HISTORY_HASH_MISMATCH"):
        formalizer._validate_failure_history(tmp_path, [bad])


def test_s111_strict_nested_wire_schemas_reject_extra_missing_type_and_cardinality() -> None:
    formalizer = _formalizer(); digest, commit = "a" * 64, "b" * 40
    gates = [gate_id for gate_id, _task_id, _role in DEPENDENCIES]
    dependency_hashes = {gate: digest for gate in gates}
    chart = {"chart_id": "gradient-identity", "source_csv_sha256": digest, "row_count": 1, "csv_ref": "a.csv", "csv_sha256": digest, "svg_ref": "a.svg", "svg_sha256": digest}
    observation = _with_hash({
        "schema_version": "stage1-s1-11-formal-observation-v1", "status": "PASS", "task_id": "stage1.11_reporting_and_exit_gate", "gate_id": "G1-EXIT", "execution_commit": commit,
        "dependency_index_sha256": dependency_hashes,
        "test_summary": {"ref": "test.json", "sha256": digest, "artifact_hash": digest},
        "sync_audit": {"ref": "sync.json", "sha256": digest, "artifact_hash": digest, "execution_commit": commit},
        "failure_history": [], "charts": [chart] * 9,
    })
    for mutation in (
        {**observation, "dependency_index_sha256": {**dependency_hashes, "G1-EXTRA": digest}},
        {**observation, "test_summary": {"ref": "test.json", "sha256": digest}},
        {**observation, "charts": [chart] * 8},
    ):
        with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:formal_observation"):
            formalizer._schema_validate(Path.cwd(), {"formal_observation": mutation})
    index = _with_hash({
        "schema_version": "stage1-s1-11-formalization-index-v1", "status": "PASS", "gate_id": "G1-EXIT", "task_id": "stage1.11_reporting_and_exit_gate", "generator_git_commit": commit, "consumer_git_commit": commit,
        "role_refs": {"gate_summary": "gate-summary.json", "requirements_matrix": "requirements-matrix.json", "stage_report": "stage-report.json", "delivery_manifest": "delivery-manifest.json"},
        "role_sha256": {key: digest for key in ("gate_summary", "requirements_matrix", "stage_report", "delivery_manifest")}, "validation_ref": "validation.json", "validation_sha256": digest, "replay_ref": "replay-validation.json", "replay_sha256": digest, "next_task_ids": ["stage2", "stage3"],
    })
    bad_index = {**index, "role_refs": {**index["role_refs"], "extra": "unexpected.json"}}
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:index"):
        formalizer._schema_validate(Path.cwd(), {"index": bad_index})


def test_s111_delivery_sync_and_validation_nested_maps_are_exact() -> None:
    formalizer = _formalizer(); digest, commit = "c" * 64, "d" * 40
    gates = [gate_id for gate_id, _task_id, _role in DEPENDENCIES]
    delivery = _with_hash({
        "schema_version": "stage1-s1-11-delivery-manifest-v1", "task_id": "stage1.11_reporting_and_exit_gate", "gate_id": "G1-EXIT",
        "summary_hash": digest, "requirements_matrix_hash": digest, "stage_report_hash": digest,
        "dependency_index_hashes": {gate: digest for gate in gates}, "chart_ids": list(REQUIRED_CHARTS),
    })
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:delivery_manifest"):
        formalizer._schema_validate(Path.cwd(), {"delivery_manifest": {**delivery, "dependency_index_hashes": {**delivery["dependency_index_hashes"], "G1-EXTRA": digest}}})
    sync = _with_hash({
        "schema_version": "stage1-s1-11-sync-audit-v1", "status": "PASS",
        "git_publish": {"remote_ref": "origin/x", "execution_commit": commit, "remote_commit": commit, "worktree_clean": True},
        "server_execution": {"execution_commit": commit, "worktree_clean": True, "evidence_root": "evidence"},
        "agent_sha256": {name: digest for name in ("remote_access.md", "server.md", "git.md", "sync.md", "worklogs.md", "local_temp.md")},
        "large_artifact_manifest": {"ref": "manifest.json", "sha256": digest},
    })
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED"):
        formalizer._schema_validate(Path.cwd(), {"formal_observation": {"schema_version": "stage1-s1-11-formal-observation-v1"}})
    schema = __import__("json").loads((Path("schemas/stage1/s1-11-validation-v1.json")).read_text(encoding="utf-8"))
    role_hashes = {name: digest for name in schema["$defs"]["role_hashes"]["required"]}
    validation = _with_hash({"schema_version": "stage1-s1-11-validation-v1", "status": "PASS", "gate_id": "G1-EXIT", "task_id": "stage1.11_reporting_and_exit_gate", "role_sha256": role_hashes})
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:validation"):
        formalizer._schema_validate(Path.cwd(), {"validation": {**validation, "role_sha256": {key: value for key, value in role_hashes.items() if key != "noise-smoke.svg"}}})


def test_s111_index_closure_rejects_unknown_missing_and_same_count_substitutions(tmp_path: Path) -> None:
    """The formalizer validates the exact frozen S1.11 closure before publish."""
    formalizer = _formalizer()
    role_refs = {
        "formal_observation": "formal-observation.json", "test_summary": "test-summary.json",
        "requirements_matrix": "requirements-matrix.json", "top_errors": "top-errors.json",
        "gate_summary": "gate-summary.json", "stage_report": "stage-report.json",
        "stage_report_markdown": "stage-report.md", "delivery_manifest": "delivery-manifest.json",
        "replay_validation": "replay-validation.json",
        **{f"{chart_id.replace('-', '_')}_{suffix}": f"{chart_id}.{suffix}" for chart_id in REQUIRED_CHARTS for suffix in ("csv", "svg")},
    }
    for filename in role_refs.values():
        (tmp_path / filename).write_text(filename, encoding="utf-8", newline="\n")
    index = {
        "implementation_source_sha256": {path: _sha(Path.cwd() / path) for path in formalizer._IMPLEMENTATION_PATHS},
        "role_refs": role_refs,
        "role_sha256": {role: _sha(tmp_path / filename) for role, filename in role_refs.items()},
        "chart_csv_sha256": {f"{chart_id}.csv": _sha(tmp_path / f"{chart_id}.csv") for chart_id in REQUIRED_CHARTS},
        "chart_svg_sha256": {f"{chart_id}.svg": _sha(tmp_path / f"{chart_id}.svg") for chart_id in REQUIRED_CHARTS},
    }
    formalizer._exact_s111_index_closure(Path.cwd(), tmp_path, index)
    for field, wrong_key in (("implementation_source_sha256", "unknown/source.py"), ("role_refs", "unknown_role"), ("chart_csv_sha256", "unknown.csv"), ("chart_svg_sha256", "unknown.svg")):
        altered = {key: (dict(value) if isinstance(value, dict) else value) for key, value in index.items()}
        values = altered[field]
        assert isinstance(values, dict)
        values.pop(next(iter(values)))
        values[wrong_key] = "a" * 64 if "sha256" in field else "unknown.json"
        with pytest.raises(formalizer.Stage1S111FormalError, match="INDEX_CLOSURE_KEYSET_INVALID"):
            formalizer._exact_s111_index_closure(Path.cwd(), tmp_path, altered)
    missing = {key: (dict(value) if isinstance(value, dict) else value) for key, value in index.items()}
    assert isinstance(missing["role_sha256"], dict)
    missing["role_sha256"].pop("gate_summary")
    with pytest.raises(formalizer.Stage1S111FormalError, match="INDEX_CLOSURE_KEYSET_INVALID"):
        formalizer._exact_s111_index_closure(Path.cwd(), tmp_path, missing)
    swapped_ref = {key: (dict(value) if isinstance(value, dict) else value) for key, value in index.items()}
    assert isinstance(swapped_ref["role_refs"], dict)
    swapped_ref["role_refs"]["gate_summary"] = "replay-validation.json"
    with pytest.raises(formalizer.Stage1S111FormalError, match="INDEX_ROLE_REF_WIRE_INVALID"):
        formalizer._exact_s111_index_closure(Path.cwd(), tmp_path, swapped_ref)


def test_s111_report_context_is_derived_and_missing_upstream_is_fail_closed(tmp_path: Path) -> None:
    formalizer = _formalizer()
    with pytest.raises(formalizer.Stage1S111FormalError, match="REPORT_DEPENDENCY_MISSING"):
        formalizer._derived_report_context(tmp_path, [], {"ref": "sync.json", "large_artifact_manifest_ref": "large.json"}, {"ref": "worklog.md", "sha256": "a" * 64})
    malformed = _with_hash({
        "schema_version": "stage1-s1-11-stage-report-v1", "status": "PASS", "task_id": "stage1.11_reporting_and_exit_gate", "gate_id": "G1-EXIT",
        "summary_hash": "a" * 64, "requirements_matrix_hash": "a" * 64, "scope_statement": "bounded", "upstream_context": {},
    })
    with pytest.raises(formalizer.Stage1S111FormalError, match="SCHEMA_VALIDATION_FAILED:stage_report"):
        formalizer._schema_validate(Path.cwd(), {"stage_report": malformed})
