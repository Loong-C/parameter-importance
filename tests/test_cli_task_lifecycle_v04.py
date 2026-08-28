from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import param_importance_nlp.cli as cli_module
from param_importance_nlp.assets import AssetActorRole, AssetFile, build_manifest
from param_importance_nlp.cli import main
from param_importance_nlp.contracts import (
    ArtifactApproval,
    ArtifactReview,
    canonical_json_hash,
    GateRecord,
    GateStatus,
    LocalValidationRecord,
    LocalValidationStatus,
    load_canonical_json,
    load_resolved_config_compatible,
    write_canonical_json,
)
from param_importance_nlp.runtime import (
    TaskExecutionRequest,
    TaskFinalizationRecord,
    TaskReplayRecord,
    TaskRunResult,
    TaskRunStatus,
    TaskRuntime,
    TaskRuntimeEnvironment,
    TaskStatusSnapshot,
    load_task_run_result,
)
from param_importance_nlp.storage import DATA_ROOT_ENV


ROOT = Path(__file__).resolve().parents[1]
V1_FIXTURE = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _v2_config(tmp_path: Path) -> Path:
    legacy = load_canonical_json(V1_FIXTURE)
    assert isinstance(legacy, dict)
    config = load_resolved_config_compatible(
        copy.deepcopy(legacy),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"artifacts": {"output_dir": "artifacts/cli-test"}},
    )
    target = tmp_path / "config-v2.json"
    write_canonical_json(target, config.to_dict())
    return target


def _formal_v2_config(tmp_path: Path) -> Path:
    legacy = load_canonical_json(V1_FIXTURE)
    assert isinstance(legacy, dict)
    legacy = copy.deepcopy(legacy)
    legacy["identity"].update(  # type: ignore[union-attr]
        {
            "run_intent": "formal",
            "formal_eligible": True,
            "route": "pretrain",
        }
    )
    legacy["runtime"]["allow_dirty_worktree"] = False  # type: ignore[index]
    legacy["importance"]["estimator_decision_ref"] = (  # type: ignore[index]
        "decisions/stage2.json"
    )
    config = load_resolved_config_compatible(
        legacy,
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"artifacts": {"output_dir": "artifacts/cli-formal-test"}},
    )
    target = tmp_path / "formal-config-v2.json"
    write_canonical_json(target, config.to_dict())
    return target


def test_endpoint_and_probe_plan_builders_compute_hashes_without_temporary_code(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint_path = tmp_path / "endpoint-plan.json"
    assert main(
        [
            "artifact",
            "endpoint-plan-build",
            "--plan-id",
            "local-endpoint-plan",
            "--step",
            "1",
            "--include-checkpoint-steps",
            "--scope",
            "local_fixture",
            "--probe-plan-ref",
            "plans/probe-plan.json",
            "--output",
            str(endpoint_path),
        ]
    ) == 0
    endpoint = json.loads(capsys.readouterr().out)
    assert endpoint["artifact_hash"] == load_canonical_json(endpoint_path)["artifact_hash"]
    assert main(["artifact-validate", str(endpoint_path)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "training_endpoint_capture_plan"

    probe_spec = tmp_path / "probe-spec.json"
    write_canonical_json(
        probe_spec,
        {
            "panel_id": "local-probe-panel",
            "endpoint_digest": hashlib.sha256(b"endpoint").hexdigest(),
            "entries": [
                {
                    "role": "formal",
                    "probe_id": "probe-0",
                    "sample_ids": ["sample-0"],
                    "content_hash": hashlib.sha256(b"sample-0").hexdigest(),
                    "loss_contract_hash": hashlib.sha256(b"loss").hexdigest(),
                    "effective_weight_unit": "sample",
                    "metadata": {},
                }
            ],
            "minimum_formal_probes": 1,
        },
    )
    probe_path = tmp_path / "probe-plan.json"
    assert main(
        [
            "artifact",
            "probe-plan-build",
            "--spec",
            str(probe_spec),
            "--scope",
            "local_fixture",
            "--output",
            str(probe_path),
        ]
    ) == 0
    probe = json.loads(capsys.readouterr().out)
    assert probe["formal_eligible"] is False
    assert main(["artifact-validate", str(probe_path)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "stage3_probe_plan"


def test_stage3_probe_selector_reads_nested_endpoint_digest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoint_digest = hashlib.sha256(b"formal-endpoint").hexdigest()
    endpoint = {
        "schema_version": "stage3-endpoint-capture-v1",
        "record": {"endpoint_digest": endpoint_digest},
        "scope": "formal",
        "formal_eligible": True,
    }
    endpoint["artifact_hash"] = canonical_json_hash(endpoint)
    endpoint_path = tmp_path / "endpoint.json"
    write_canonical_json(endpoint_path, endpoint)
    plan = {
        "schema_version": "stage3-probe-plan-v1",
        "panel_id": "formal-panel",
        "endpoint_digest": endpoint_digest,
        "entries": [
            {
                "role": "formal",
                "probe_id": "formal-probe-0",
                "sample_ids": ["sample-0"],
                "content_hash": hashlib.sha256(b"sample-0").hexdigest(),
                "loss_contract_hash": hashlib.sha256(b"loss").hexdigest(),
                "effective_weight_unit": "effective_token",
                "metadata": {},
            }
        ],
        "minimum_formal_probes": 1,
        "execution_evidence_hash": "a" * 64,
        "scope": "formal",
        "formal_eligible": True,
    }
    plan["artifact_hash"] = canonical_json_hash(plan)
    plan_path = tmp_path / "probe-plan.json"
    write_canonical_json(plan_path, plan)
    output = tmp_path / "selector.json"
    assert main(
        [
            "artifact",
            "stage3-probe-selector-build",
            "--endpoint-commit",
            str(endpoint_path),
            "--probe-plan",
            str(plan_path),
            "--active-probe-id",
            "formal-probe-0",
            "--output",
            str(output),
        ]
    ) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["endpoint_digest"] == endpoint_digest
    assert load_canonical_json(output)["artifact_hash"] == emitted["artifact_hash"]


class _PassingContractRunner:
    def __init__(self) -> None:
        from param_importance_nlp.contracts import RunnerKind

        self.runner_kind = RunnerKind.CONTRACT

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        refs = {
            kind: f"artifacts/{request.task.task_id}/{kind}.json"
            for kind in request.task.artifact_kinds
        }
        return TaskRunResult.passed(request, artifact_refs=refs)


def test_environment_and_lifecycle_contracts_roundtrip_and_reject_hash_drift() -> None:
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"git", "server"}),
        frozen_contract_stages=frozenset({0, 1}),
        passed_gate_ids=frozenset({"stage0.G1"}),
        estimator_decision_ref="decisions/estimator.json",
        evidence_refs={"server": "evidence/server.json"},
    )
    assert (
        TaskRuntimeEnvironment.from_mapping(environment.to_dict()).environment_hash
        == environment.environment_hash
    )
    tampered = environment.to_dict()
    tampered["capabilities"] = ["git"]
    with pytest.raises(Exception, match="environment_hash"):
        TaskRuntimeEnvironment.from_mapping(tampered)


def test_nested_task_cli_run_status_replay_and_finalize(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _v2_config(tmp_path)
    runtime = TaskRuntime()
    runtime.register(_PassingContractRunner())
    monkeypatch.setattr(
        cli_module,
        "_build_default_task_runtime",
        lambda workspace_root: runtime,
    )

    assert main(["task", "catalog", "--task-id", "stage0.05_config_run_identity_and_seeds"]) == 0
    catalog_item = json.loads(capsys.readouterr().out)
    assert catalog_item["task_id"] == "stage0.05_config_run_identity_and_seeds"

    assert main(["task", "preflight", "--config", str(config_path)]) == 0
    preflight = json.loads(capsys.readouterr().out)
    assert preflight["ready"] is True
    assert preflight["formal_eligible_if_completed"] is False

    first_result = tmp_path / "first-result.json"
    assert (
        main(
            [
                "task",
                "run",
                "--config",
                str(config_path),
                "--result",
                str(first_result),
            ]
        )
        == 0
    )
    run_value = json.loads(capsys.readouterr().out)
    assert run_value["status"] == "PASS"
    assert first_result.read_bytes().endswith(b"\n")

    status_path = tmp_path / "status.json"
    assert (
        main(
            [
                "task",
                "status",
                "--result",
                str(first_result),
                "--result-ref",
                "results/first.json",
                "--config",
                str(config_path),
                "--output",
                str(status_path),
            ]
        )
        == 0
    )
    status_value = json.loads(capsys.readouterr().out)
    snapshot = TaskStatusSnapshot.from_mapping(status_value)
    assert snapshot.status is TaskRunStatus.PASS

    replay_result = tmp_path / "replay-result.json"
    replay_record = tmp_path / "replay-record.json"
    assert (
        main(
            [
                "task",
                "replay",
                "--config",
                str(config_path),
                "--source-result",
                str(first_result),
                "--source-ref",
                "results/first.json",
                "--result",
                str(replay_result),
                "--result-ref",
                "results/replay.json",
                "--output",
                str(replay_record),
            ]
        )
        == 0
    )
    replay_value = json.loads(capsys.readouterr().out)
    assert TaskReplayRecord.from_mapping(replay_value).equivalent is True

    finalization_path = tmp_path / "finalization.json"
    assert (
        main(
            [
                "task",
                "finalize",
                "--result",
                str(first_result),
                "--result-ref",
                "results/first.json",
                "--output",
                str(finalization_path),
            ]
        )
        == 0
    )
    finalization = TaskFinalizationRecord.from_mapping(
        json.loads(capsys.readouterr().out)
    )
    assert finalization.scope == "local_fixture"
    assert finalization.formal_eligible is False
    assert load_task_run_result(first_result).result_hash == run_value["result_hash"]
    for path, expected_kind in (
        (first_result, "task_run_result"),
        (status_path, "task_status_snapshot"),
        (replay_record, "task_replay_record"),
        (finalization_path, "task_finalization_record"),
    ):
        assert main(["artifact-validate", str(path)]) == 0
        assert json.loads(capsys.readouterr().out)["kind"] == expected_kind


def test_task_run_and_resume_have_distinct_explicit_recovery_contracts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run 禁止隐式恢复，resume 也不能在未绑定权威引用时猜测。"""

    runtime = TaskRuntime()
    runtime.register(_PassingContractRunner())
    monkeypatch.setattr(
        cli_module,
        "_build_default_task_runtime",
        lambda workspace_root: runtime,
    )

    fresh_path = _v2_config(tmp_path)
    assert main(["task", "resume", "--config", str(fresh_path)]) == 2
    assert "TASK_RESUME_REQUIRES_RECOVERY_RESUME_REF" in capsys.readouterr().err

    legacy = load_canonical_json(V1_FIXTURE)
    assert isinstance(legacy, dict)
    resume_config = load_resolved_config_compatible(
        copy.deepcopy(legacy),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={
            "recovery": {"resume_ref": "artifacts/checkpoints"},
            "artifacts": {"output_dir": "artifacts/cli-resume-test"},
        },
    )
    resume_path = tmp_path / "resume-config-v2.json"
    write_canonical_json(resume_path, resume_config.to_dict())
    assert main(["task", "run", "--config", str(resume_path)]) == 2
    assert "TASK_RUN_FORBIDS_RECOVERY_RESUME_REF" in capsys.readouterr().err

    # resume 命令已通过动作合同，后续 runner 可按自己的安全边界消费该引用。
    assert main(["task", "resume", "--config", str(resume_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PASS"


def test_missing_runner_uses_structured_blocked_and_exit_code_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "_build_default_task_runtime",
        lambda workspace_root: TaskRuntime(workspace_root=workspace_root),
    )
    config = _v2_config(tmp_path)
    result = tmp_path / "blocked.json"
    assert main(["task", "run", "--config", str(config), "--result", str(result)]) == 3
    value = json.loads(capsys.readouterr().out)
    assert value["status"] == "BLOCKED"
    assert value["blockers"][0]["code"] == "runner_unavailable"

    finalization = tmp_path / "must-not-exist.json"
    assert (
        main(
            [
                "task",
                "finalize",
                "--result",
                str(result),
                "--output",
                str(finalization),
            ]
        )
        == 3
    )
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "BLOCKED"
    assert not finalization.exists()


def test_task_cli_workspace_is_bound_to_run_intent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local_config = _v2_config(tmp_path)
    formal_config = _formal_v2_config(tmp_path)
    local_workspace = tmp_path / "local-workspace"
    formal_data_root = tmp_path / "formal-data-root"
    local_workspace.mkdir()
    formal_data_root.mkdir()
    captured_roots: list[Path] = []

    def build_runtime(workspace_root: str | Path) -> TaskRuntime:
        root = Path(workspace_root).resolve()
        captured_roots.append(root)
        runtime = TaskRuntime(workspace_root=root)
        runtime.register(_PassingContractRunner())
        return runtime

    monkeypatch.setattr(cli_module, "_build_default_task_runtime", build_runtime)
    monkeypatch.chdir(local_workspace)

    # A local fixture is always scoped to cwd, even when a DATA_ROOT is present.
    monkeypatch.setenv(DATA_ROOT_ENV, str(formal_data_root))
    assert main(["task", "preflight", "--config", str(local_config)]) == 0
    capsys.readouterr()
    assert captured_roots == [local_workspace.resolve()]

    # Formal execution cannot infer cwd as its evidence workspace.
    monkeypatch.delenv(DATA_ROOT_ENV)
    assert main(["task", "preflight", "--config", str(formal_config)]) == 2
    assert DATA_ROOT_ENV in capsys.readouterr().err
    assert captured_roots == [local_workspace.resolve()]

    monkeypatch.setenv(DATA_ROOT_ENV, str(formal_data_root))
    assert main(["task", "preflight", "--config", str(formal_config)]) == 3
    formal_preflight = json.loads(capsys.readouterr().out)
    assert formal_preflight["ready"] is False
    assert captured_roots[-1] == formal_data_root.resolve()


def test_artifact_review_approval_never_promotes_fixture(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = LocalValidationRecord(
        validation_id="local.review.fixture",
        status=LocalValidationStatus.PASS,
        checked_at="2026-07-22T00:00:00Z",
        checks={"fixture": True},
        evidence_refs=("evidence/local.json",),
    )
    artifact = tmp_path / "local-validation.json"
    write_canonical_json(artifact, local.to_dict())
    review_path = tmp_path / "review.json"
    assert (
        main(
            [
                "artifact",
                "review",
                "--artifact",
                str(artifact),
                "--artifact-ref",
                "reports/local-validation.json",
                "--scope",
                "local_fixture",
                "--reviewer",
                "reviewer.1",
                "--decision",
                "APPROVE",
                "--output",
                str(review_path),
            ]
        )
        == 0
    )
    review = ArtifactReview.from_mapping(json.loads(capsys.readouterr().out))
    assert review.artifact_formal_eligible is False

    local_approval = tmp_path / "local-approval.json"
    assert (
        main(
            [
                "artifact",
                "approve",
                "--review",
                str(review_path),
                "--review-ref",
                "reviews/local.json",
                "--scope",
                "local_validation",
                "--approver",
                "approver.1",
                "--output",
                str(local_approval),
            ]
        )
        == 0
    )
    approval = ArtifactApproval.from_mapping(json.loads(capsys.readouterr().out))
    assert approval.approval_scope.value == "local_validation"
    for path, expected_kind in (
        (review_path, "artifact_review"),
        (local_approval, "artifact_approval"),
    ):
        assert main(["artifact-validate", str(path)]) == 0
        assert json.loads(capsys.readouterr().out)["kind"] == expected_kind

    forbidden = tmp_path / "formal-approval.json"
    assert (
        main(
            [
                "artifact",
                "approve",
                "--review",
                str(review_path),
                "--review-ref",
                "reviews/local.json",
                "--scope",
                "formal",
                "--approver",
                "approver.1",
                "--output",
                str(forbidden),
            ]
        )
        == 2
    )
    assert "formal approval" in capsys.readouterr().err
    assert not forbidden.exists()

    formal_gate = GateRecord(
        gate_id="stage0.G98",
        stage=0,
        status=GateStatus.PASS,
        checked_at="2026-07-22T00:00:00Z",
        evidence_refs=("evidence/formal-gate.json",),
    )
    gate_path = tmp_path / "formal-gate.json"
    write_canonical_json(gate_path, formal_gate.to_dict())
    formal_review_path = tmp_path / "formal-review.json"
    assert (
        main(
            [
                "artifact", "review",
                "--artifact", str(gate_path),
                "--artifact-ref", "gates/stage0.G98.json",
                "--scope", "formal",
                "--reviewer", "reviewer.2",
                "--decision", "APPROVE",
                "--output", str(formal_review_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["artifact_formal_eligible"] is True
    formal_approval_path = tmp_path / "formal-approved.json"
    assert (
        main(
            [
                "artifact", "approve",
                "--review", str(formal_review_path),
                "--review-ref", "reviews/formal-gate.json",
                "--scope", "formal",
                "--approver", "approver.2",
                "--output", str(formal_approval_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["approval_scope"] == "formal"


def test_local_asset_acquire_verify_and_nested_gate_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    payload = b"local-only\n"
    (source / "source.bin").write_bytes(payload)
    manifest = build_manifest(
        asset_type="source",
        name="local-source-fixture",
        source="local:fixture",
        revision="fixture-v1",
        files=[
            AssetFile(
                "source.bin",
                len(payload),
                hashlib.sha256(payload).hexdigest(),
                "source",
            )
        ],
        actor="fixture-fetcher",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref="evidence/acquire-start.json",
        generator_version="tests/1",
        metadata={"source_kind": "local-file", "license": "test-only"},
        created_at="2026-07-22T00:00:00Z",
    )
    manifest_path = tmp_path / "candidate.json"
    write_canonical_json(manifest_path, manifest)
    downloaded_path = tmp_path / "downloaded.json"
    assert (
        main(
            [
                "asset",
                "acquire",
                "--manifest",
                str(manifest_path),
                "--source-root",
                str(source),
                "--actor",
                "fixture-fetcher",
                "--evidence-ref",
                "evidence/acquire-complete.json",
                "--at",
                "2026-07-22T00:00:01Z",
                "--output",
                str(downloaded_path),
            ]
        )
        == 0
    )
    acquired = json.loads(capsys.readouterr().out)
    assert acquired["network_accessed"] is False
    assert load_canonical_json(downloaded_path)["state"] == "downloaded"  # type: ignore[index]

    verification_path = tmp_path / "verification.json"
    assert (
        main(
            [
                "asset",
                "verify",
                "--manifest",
                str(downloaded_path),
                "--asset-root",
                str(source),
                "--output",
                str(verification_path),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["ok"] is True

    gate_path = tmp_path / "gate.json"
    assert (
        main(
            [
                "gate",
                "build",
                "--gate-id",
                "stage0.G99",
                "--stage",
                "0",
                "--status",
                "BLOCKED",
                "--checked-at",
                "2026-07-22T00:00:00Z",
                "--reason",
                "server_unreachable",
                "--output",
                str(gate_path),
            ]
        )
        == 0
    )
    gate = GateRecord.from_mapping(json.loads(capsys.readouterr().out))
    assert gate.status is GateStatus.BLOCKED
    assert main(["gate", "summary", str(gate_path)]) == 0
    assert json.loads(capsys.readouterr().out)["formal"]["status_counts"] == {
        "BLOCKED": 1
    }
