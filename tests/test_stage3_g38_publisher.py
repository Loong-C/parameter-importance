from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.analysis import (
    AnalysisReportBuilder,
    ChartArtifact,
    ChartSpec,
    FrozenSourceTable,
    MetricResult,
)
from param_importance_nlp.experiments.stage3_g37_publisher import Stage3G37Publisher
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_GATE_IDS,
    Stage3G38DeliveryManifest,
    Stage3G38Publisher,
    publish_stage3_delivery_manifest,
    validate_stage3_git_sync_evidence,
    validate_stage3_replay_reports,
)
from param_importance_nlp.experiments.stage3_raw_storage import (
    persist_raw_unit_shard,
    publish_raw_aggregate,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore, load_committed_task_artifact
import torch


CONFIG = "a" * 64
STAGE310_CONFIG = "b" * 64


def _g37_test_module():
    path = Path(__file__).with_name("test_stage3_g37_publisher.py")
    spec = importlib.util.spec_from_file_location("stage3_g37_test_helpers", path)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError("cannot load Stage3 G3-7 test helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _put_file(root: Path, name: str, data: bytes) -> dict[str, object]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def _manifest(root: Path) -> dict[str, object]:
    n = 0

    def f(ext: str, data: bytes | None = None) -> dict[str, object]:
        nonlocal n
        n += 1
        return _put_file(
            root,
            f"delivery/file-{n}{ext}",
            f"real-{n}".encode() if data is None else data,
        )

    def replay(layer: str) -> dict[str, object]:
        evidence = f(f"-{layer}.log")
        body: dict[str, object] = {
            "schema_version": "stage3-g38-replay-report-v1",
            "replay_id": f"stage3-replay-{layer}",
            "layer": layer,
            "scope": "formal",
            "status": "PASS",
            "formal_eligible": True,
            "implementation_commit": "1" * 40,
            "environment_hash": "2" * 64,
            "command": ["python", "-m", "pytest", "-q", "tests/test_stage3_gate.py"],
            "returncode": 0,
            "started_at": "2026-08-28T00:00:00Z",
            "completed_at": "2026-08-28T00:01:00Z",
            "cache_mode": {
                "local_cpu": "not_applicable",
                "server_locked": "locked_environment",
                "frozen_endpoint_uncached": "uncached",
            }[layer],
            "test_summary": {
                "collected": 1,
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
            },
            "input_refs": {"authority": "evidence/stage3/replay-source.json"},
            "input_hashes": {"authority": "3" * 64},
            "evidence_files": [evidence],
        }
        payload = body | {"artifact_hash": canonical_json_hash(body)}
        return f(f"-{layer}.json", canonical_json_bytes(payload))

    def git_evidence(role: str) -> dict[str, object]:
        log = f(f"-{role}.log")
        body: dict[str, object] = {
            "schema_version": "stage3-g38-git-sync-evidence-v1",
            "evidence_id": f"stage3-git-{role}",
            "role": role,
            "scope": "formal",
            "status": "PASS",
            "formal_eligible": True,
            "checked_at": "2026-08-28T00:02:00Z",
            "branch": "codex/stage3-delivery",
            "local_commit": "4" * 40,
            "remote_commit": "4" * 40,
            "server_commit": "4" * 40,
            "remote_name": "origin",
            "local_delivery_worktree_clean": True,
            "server_worktree_clean": True,
            "agent_document_hashes": {
                name: "5" * 64
                for name in (
                    "Agent/git.md",
                    "Agent/local.md",
                    "Agent/remote_access.md",
                    "Agent/server.md",
                    "Agent/sync.md",
                    "Agent/worklogs.md",
                )
            },
            "command": ["git", "status", "--porcelain=v1"],
            "returncode": 0,
            "stdout_log": log,
        }
        payload = body | {"artifact_hash": canonical_json_hash(body)}
        return f(f"-{role}.json", canonical_json_bytes(payload))

    value: dict[str, object] = {
        "schema_version": "stage3-g38-delivery-manifest-v1",
        "manifest_id": "stage3-g38-delivery",
        "scope": "formal",
        "status": "PASS",
        "formal_eligible": True,
        "source_tables": {"csv": [f(".csv")], "json": [f(".json")]},
        "analysis_scripts": [f(".py")],
        "figures": [{"id": "overview", "png": f(".png"), "svg": f(".svg")}],
        "chinese_report": {"tex": f(".tex"), "pdf": f(".pdf")},
        "beamer": {"tex": f("-slides.tex"), "pdf": f("-slides.pdf"), "notes": [f("-notes.md")], "backups": [f("-backup-1.tex"), f("-backup-2.tex"), f("-backup-3.tex")]},
        "replay_reports": {
            layer: replay(layer)
            for layer in ("local_cpu", "server_locked", "frozen_endpoint_uncached")
        },
        "server_large_artifact_manifest": f("-large.json"),
        "git_sync": {
            role: git_evidence(role)
            for role in ("branch", "commit", "push", "remote", "server_clean_head", "sync")
        },
        "worklog": f(".md"),
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _inputs(root: Path) -> dict[str, object]:
    helper = _g37_test_module()
    g37_inputs = helper._pass_inputs(root)
    g37 = Stage3G37Publisher().publish(
        workspace_root=root,
        output_dir="inputs/g37",
        **g37_inputs,
    )
    assert g37.status == "PASS"
    assert g37.recommendation_ref is not None
    assert g37.finalization_ref is not None
    g37_publication_ref = "inputs/g37/commits/g37_publication.json"

    base_execution = FormalExecutionEvidence.from_mapping(
        load_committed_task_artifact(
            root,
            g37.execution_evidence_ref,
            require_formal=True,
        ).payload
    )
    gates: dict[str, str] = {
        "stage3.G3-0": g37_inputs["stage3_scope_gate_ref"],
        "stage3.G3-6": g37.g3_6_ref,
        "stage3.G3-7": g37.g3_7_ref,
    }
    for gate in base_execution.prerequisite_gates:
        if gate.gate_id == "stage3.G3-0":
            continue
        published = TaskArtifactStore(root, f"inputs/{gate.gate_id}").publish(
            task_id=f"{gate.gate_id}.authority",
            artifact_kind="gate_record",
            config_hash=CONFIG,
            run_intent="formal",
            formal_eligible=True,
            payload=gate.to_dict(),
            source_refs=gate.evidence_refs,
        )
        gates[gate.gate_id] = published.commit_ref
    current_execution = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=base_execution.contract_freeze_hash,
        asset_manifest_hashes=base_execution.asset_manifest_hashes,
        prerequisite_gates=(
            *base_execution.prerequisite_gates,
            g37.g3_6_gate,
            g37.g3_7_gate,
        ),
        metadata=base_execution.metadata,
    )
    execution_commit = TaskArtifactStore(root, "inputs/execution-g38").publish(
        task_id="stage3.formal_execution_authority",
        artifact_kind="formal_execution_evidence",
        config_hash=STAGE310_CONFIG,
        run_intent="formal",
        formal_eligible=True,
        payload=current_execution.to_dict(),
        source_refs=tuple(gates.values()),
    )

    raw_ledger = root / "inputs" / "raw"
    persist_raw_unit_shard(
        root=root,
        ledger_root=raw_ledger,
        unit_id="unit-1",
        required_unit_ids=("unit-1",),
        execution_evidence_hash=current_execution.artifact_hash,
        reference_binding_hash="c" * 64,
        path_identity_hash="d" * 64,
        reference_artifact_hash="e" * 64,
        reference_signed={"weight": torch.tensor([1.0, 2.0], dtype=torch.float64)},
        candidate_states={
            "trapezoid": {
                "signed": {"weight": torch.tensor([1.1, 1.9], dtype=torch.float64)},
                "positive": {"weight": torch.tensor([1.1, 1.9], dtype=torch.float64)},
                "negative_mass": {"weight": torch.zeros(2, dtype=torch.float64)},
                "absolute": {"weight": torch.tensor([1.1, 1.9], dtype=torch.float64)},
            }
        },
        rule_summaries={"trapezoid": {"rule": {"nodes": [0.0, 0.5, 1.0]}, "node_alphas": [0.0, 0.5, 1.0], "node_losses": [3.0, 2.0, 1.0], "path_identity_hash": "d" * 64, "evaluation_cost": {"gradient_evaluations": 3}}},
    )
    raw_aggregate, raw_aggregate_ref = publish_raw_aggregate(
        root=root,
        ledger_root=raw_ledger,
        required_unit_ids=("unit-1",),
        execution_evidence_hash=current_execution.artifact_hash,
        reference_binding_hash="c" * 64,
        candidate_rule_names=("trapezoid",),
    )

    source = FrozenSourceTable.from_rows(
        name="stage3-formal-cost",
        schema_version="stage3-formal-cost-v1",
        rows=({"unique_nodes": 1.0, "normalized_l1_error": 0.1}, {"unique_nodes": 2.0, "normalized_l1_error": 0.05}),
    )
    report_builder = AnalysisReportBuilder(report_id="stage3-formal-report")
    report_builder.add_source(source)
    report_builder.add_metric(
        "mean_error",
        MetricResult(True, 0.075),
        source=source,
        derivation_id="test.mean-error.v1",
        input_columns=("normalized_l1_error",),
    )
    report_payload = report_builder.build(
        metadata={"scope": "formal", "formal_eligible": True, "stage3_07_raw_aggregate_ref": raw_aggregate_ref, "stage3_07_raw_aggregate_hash": raw_aggregate["artifact_hash"]}
    ).to_dict()
    spec = ChartSpec.from_table(
        source,
        chart_id="overview",
        chart_type="line",
        x_column="unique_nodes",
        y_columns=("normalized_l1_error",),
        sort_columns=("unique_nodes",),
    )
    png_bytes, svg_bytes = b"real-png", b"real-svg"
    png_record = _put_file(root, "delivery/chart.png", png_bytes)
    svg_record = _put_file(root, "delivery/chart.svg", svg_bytes)
    chart_payload = {
        "schema_version": "stage3-task-chart-artifacts-v1",
        "scope": "formal",
        "formal_eligible": True,
        "source_table_hash": source.content_hash,
        "artifacts": [
            ChartArtifact.from_rendered_bytes(
                spec,
                png_bytes,
                renderer_id="fixture-renderer:v1",
                output_format="png",
                render_options={},
            ).to_dict(),
            ChartArtifact.from_rendered_bytes(
                spec,
                svg_bytes,
                renderer_id="fixture-renderer:v1",
                output_format="svg",
                render_options={},
            ).to_dict(),
        ],
        "rendered_figures": [{
            "id": "overview",
            "source_table": source.name,
            "source_hash": source.content_hash,
            "png": png_record,
            "svg": svg_record,
        }],
        "manual_numeric_edits_allowed": False,
    }
    handoff_payload = {
        "schema_version": "stage3-task-handoff-manifest-v1",
        "scope": "formal",
        "formal_eligible": True,
        "stage3_finalization": g37.finalization,
        "stage3_g37_publication_ref": g37_publication_ref,
        "stage3_g37_publication_hash": g37.artifact_hash,
        "stage3_g37_gate_ref": g37.g3_7_ref,
        "stage3_g37_gate_hash": g37.g3_7_hash,
        "stage3_g37_recommendation_ref": g37.recommendation_ref,
        "stage3_g37_recommendation_hash": g37.recommendation_hash,
        "stage3_g37_finalization_ref": g37.finalization_ref,
        "stage3_g37_finalization_hash": g37.finalization_hash,
        "execution_evidence_ref": execution_commit.commit_ref,
        "execution_evidence_hash": current_execution.artifact_hash,
        "source_table_hash": source.content_hash,
        "default_rule": "midpoint",
        "fallback_rule": "trapezoid",
        "passing_rules": ["midpoint", "trapezoid"],
        "cost_semantics": "measured_callback_cost_and_unique_nodes_v1",
        "report_hash": report_payload["report_hash"],
        "stage3_07_raw_aggregate_ref": raw_aggregate_ref,
        "stage3_07_raw_aggregate_hash": raw_aggregate["artifact_hash"],
        "formal_stage_complete": False,
        "completion_boundary": "PENDING_G3_8_DELIVERY_ACCEPTANCE",
    }
    gate_summary_payload = {
        "schema_version": "stage3-task-gate-summary-v1",
        "scope": "formal",
        "stage3.G3-6": "PASS",
        "stage3.G3-7": "PASS",
        "stage3.G3-8": "NOT_RUN",
        "stage3.G3-7_ref": g37.g3_7_ref,
        "stage3.G3-7_hash": g37.g3_7_hash,
        "formal_exit_gate": "NOT_RUN",
        "local_validation_status": "PASS",
    }
    stage310: dict[str, str] = {}
    for kind, payload in (
        ("analysis_report", report_payload),
        ("chart_artifacts", chart_payload),
        ("handoff_manifest", handoff_payload),
        ("gate_summary", gate_summary_payload),
    ):
        published = TaskArtifactStore(root, "inputs/stage310").publish(
            task_id="stage3.10_reports_visualizations_and_handoff",
            artifact_kind=kind,
            config_hash=STAGE310_CONFIG,
            run_intent="formal",
            formal_eligible=True,
            payload=payload,
            source_refs=(execution_commit.commit_ref, g37_publication_ref),
        )
        stage310[kind] = published.commit_ref

    manifest = _manifest(root)
    manifest_commit = publish_stage3_delivery_manifest(
        workspace_root=root,
        output_dir="inputs/manifest",
        config_hash=STAGE310_CONFIG,
        manifest=manifest,
        stage3_10_refs=stage310,
        source_refs=tuple(stage310.values()),
    )
    return {
        "gate_refs": gates,
        "stage3_10_refs": stage310,
        "execution_evidence_ref": execution_commit.commit_ref,
        "g3_7_publication_ref": g37_publication_ref,
        "recommendation_ref": g37.recommendation_ref,
        "finalization_ref": g37.finalization_ref,
        "delivery_manifest_ref": manifest_commit.commit_ref,
    }


def test_g38_requires_all_formal_inputs_and_hashes_files(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    result = Stage3G38Publisher().publish(
        workspace_root=tmp_path, output_dir="outputs/g38", config_hash=CONFIG,
        checked_at="2026-08-28T01:00:00Z", **inputs,
    )
    assert result.status == "PASS"
    assert result.formal_eligible is True
    second = Stage3G38Publisher().publish(
        workspace_root=tmp_path, output_dir="outputs/g38-second", config_hash=CONFIG,
        checked_at="2026-08-28T01:00:00Z", **inputs,
    )
    assert result.publication_config_hash == second.publication_config_hash
    assert result.config_hash == second.config_hash
    assert result.config_hash != CONFIG
    receipt = load_committed_task_artifact(tmp_path, "outputs/g38/commits/g38_publication.json", require_formal=True)
    assert receipt.payload == result.to_dict()
    assert Stage3G38DeliveryManifest.from_mapping(dict(load_committed_task_artifact(tmp_path, inputs["delivery_manifest_ref"], require_formal=True).payload)).artifact_hash


def test_g38_rejects_replay_with_skip_even_when_file_hash_matches(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = manifest["replay_reports"]["server_locked"]  # type: ignore[index]
    path = tmp_path / str(record["path"])
    report = dict(load_canonical_json(path))
    report["test_summary"] = {
        "collected": 1,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 1,
    }
    report["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in report.items() if key != "artifact_hash"}
    )
    data = canonical_json_bytes(report)
    path.write_bytes(data)
    record["size"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "artifact_hash"}
    )
    parsed = Stage3G38DeliveryManifest.from_mapping(manifest)

    with pytest.raises(FormalRunRejected, match="REPLAY_TEST_SUMMARY_NOT_PASS:server_locked"):
        validate_stage3_replay_reports(tmp_path, parsed)


def test_g38_rejects_git_head_mismatch_even_when_file_hash_matches(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    record = manifest["git_sync"]["remote"]  # type: ignore[index]
    path = tmp_path / str(record["path"])
    evidence = dict(load_canonical_json(path))
    evidence["remote_commit"] = "6" * 40
    evidence["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in evidence.items() if key != "artifact_hash"}
    )
    data = canonical_json_bytes(evidence)
    path.write_bytes(data)
    record["size"] = len(data)
    record["sha256"] = hashlib.sha256(data).hexdigest()
    manifest["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in manifest.items() if key != "artifact_hash"}
    )
    parsed = Stage3G38DeliveryManifest.from_mapping(manifest)

    with pytest.raises(FormalRunRejected, match="GIT_HEAD_MISMATCH:remote"):
        validate_stage3_git_sync_evidence(tmp_path, parsed)


def test_g38_rejects_empty_or_spec_only_chart_bundle(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    report = load_committed_task_artifact(
        tmp_path,
        inputs["stage3_10_refs"]["analysis_report"],  # type: ignore[index]
        require_formal=True,
    ).payload
    source_hash = report["source_artifacts"][0]["content_hash"]  # type: ignore[index]
    bad = TaskArtifactStore(tmp_path, "inputs/stage310-bad").publish(
        task_id="stage3.10_reports_visualizations_and_handoff",
        artifact_kind="chart_artifacts",
        config_hash=STAGE310_CONFIG,
        run_intent="formal",
        formal_eligible=True,
        payload={
            "schema_version": "stage3-task-chart-artifacts-v1",
            "scope": "formal",
            "formal_eligible": True,
            "source_table_hash": source_hash,
            "artifacts": [],
            "rendered_figures": [],
            "manual_numeric_edits_allowed": False,
        },
        source_refs=(inputs["execution_evidence_ref"], inputs["g3_7_publication_ref"]),  # type: ignore[arg-type]
    )
    refs = dict(inputs["stage3_10_refs"])  # type: ignore[arg-type]
    refs["chart_artifacts"] = bad.commit_ref
    with pytest.raises(FormalRunRejected, match="CHARTS_REQUIRE_PNG_AND_SVG"):
        Stage3G38Publisher().publish(
            workspace_root=tmp_path,
            output_dir="outputs/empty-charts",
            config_hash=CONFIG,
            checked_at="2026-08-28T01:00:00Z",
            **(inputs | {"stage3_10_refs": refs}),
        )


def test_g38_rejects_missing_gate_and_tampered_file(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    missing = dict(inputs["gate_refs"])
    missing.pop("stage3.G3-7")
    with pytest.raises(FormalRunRejected, match="G3_0_THROUGH_G3_7"):
        Stage3G38Publisher().publish(workspace_root=tmp_path, output_dir="outputs/missing", config_hash=CONFIG, **(inputs | {"gate_refs": missing}))

    source = tmp_path / "delivery/file-1.csv"
    source.write_bytes(b"tampered")
    with pytest.raises(FormalRunRejected, match="SHA256_MISMATCH|SIZE_MISMATCH"):
        Stage3G38Publisher().publish(workspace_root=tmp_path, output_dir="outputs/tampered", config_hash=CONFIG, **inputs)


def test_g38_rejects_execution_gate_drift_and_stage310_kind_swap(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    execution_artifact = load_committed_task_artifact(
        tmp_path,
        inputs["execution_evidence_ref"],
        require_formal=True,
    )
    execution = FormalExecutionEvidence.from_mapping(execution_artifact.payload)
    missing_g37 = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash=execution.contract_freeze_hash,
        asset_manifest_hashes=execution.asset_manifest_hashes,
        prerequisite_gates=tuple(
            gate for gate in execution.prerequisite_gates
            if gate.gate_id != "stage3.G3-7"
        ),
        metadata=execution.metadata,
    )
    bad_execution = TaskArtifactStore(tmp_path, "inputs/execution-drift").publish(
        task_id="stage3.formal_execution_authority",
        artifact_kind="formal_execution_evidence",
        config_hash=STAGE310_CONFIG,
        run_intent="formal",
        formal_eligible=True,
        payload=missing_g37.to_dict(),
        source_refs=tuple(inputs["gate_refs"].values()),
    )
    with pytest.raises(FormalRunRejected, match="EXACT_G3_0_THROUGH_G3_7"):
        Stage3G38Publisher().publish(
            workspace_root=tmp_path,
            output_dir="outputs/execution-drift",
            **(inputs | {"execution_evidence_ref": bad_execution.commit_ref}),
        )

    swapped = dict(inputs["stage3_10_refs"])
    swapped["analysis_report"], swapped["chart_artifacts"] = (
        swapped["chart_artifacts"],
        swapped["analysis_report"],
    )
    with pytest.raises(FormalRunRejected, match="ARTIFACT_KIND_INVALID"):
        Stage3G38Publisher().publish(
            workspace_root=tmp_path,
            output_dir="outputs/kind-swap",
            **(inputs | {"stage3_10_refs": swapped}),
        )
