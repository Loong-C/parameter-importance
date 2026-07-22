from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.cli import main
from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    RuntimeCapabilityEvidence,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.experiments import build_default_task_runtime
from param_importance_nlp.experiments.stage01_task_runners import (
    build_stage01_runner_overrides,
)
from param_importance_nlp.experiments.task_runners import TrainingTaskRunner
from param_importance_nlp.runtime import (
    BlockerCode,
    TaskArtifactStore,
    TaskBlockedError,
    TaskExecutionRequest,
    TaskRuntime,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _base_config(*, task_id: str, formal: bool) -> ResolvedConfig:
    task = DEFAULT_TASK_CATALOG.get(task_id)
    value = deepcopy(
        load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json")
    )
    value["identity"].update(  # type: ignore[union-attr]
        {
            "stage": task.stage,
            "task": task_id,
            "run_intent": "formal" if formal else "local_fixture",
            "formal_eligible": formal,
            "route": "pretrain",
        }
    )
    value["runtime"]["allow_dirty_worktree"] = not formal  # type: ignore[index]
    value["loss"].update(  # type: ignore[union-attr]
        {"task_type": "sequence_classification", "weighting": "sample"}
    )
    value["data"].update(  # type: ignore[union-attr]
        {"statistical_unit": "sample", "weight_unit": "sample"}
    )
    value["model"]["architecture"] = "tiny-sequence-classifier"  # type: ignore[index]
    return ResolvedConfig.from_mapping(value)


def _publish(
    root: Path,
    output: str,
    *,
    task_id: str,
    kind: str,
    payload: dict[str, object],
    formal: bool = True,
) -> str:
    return TaskArtifactStore(root, output).publish(
        task_id=task_id,
        artifact_kind=kind,
        config_hash="a" * 64,
        run_intent="formal" if formal else "local_fixture",
        payload=payload,
        formal_eligible=formal,
    ).commit_ref


def _formal_environment(
    root: Path,
    task_id: str,
) -> tuple[TaskRuntimeEnvironment, tuple[str, ...]]:
    task = DEFAULT_TASK_CATALOG.get(task_id)
    evidence_refs: dict[str, str] = {}
    for stage in task.formal_eligibility.required_contract_stages:
        freeze = ContractFreeze(
            contract_id=f"stage{stage}.contract.formal-test",
            stage=stage,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version="formal-test-v1",
            config_hash="1" * 64,
            schema_hashes={"schema": "2" * 64},
            source_hashes={"source": "3" * 64},
            required_gate_ids=(f"stage{stage}.G1",),
            frozen_at="2026-07-22T00:00:00+00:00",
        )
        evidence_refs[f"contract_stage_{stage}"] = _publish(
            root,
            f"formal-evidence/{task_id}/contract-{stage}",
            task_id="stage0.01_baseline_and_safety",
            kind="contract_freeze",
            payload=freeze.to_dict(),
        )
    for index, gate_id in enumerate(task.formal_eligibility.required_gate_ids):
        gate = GateRecord(
            gate_id=gate_id,
            stage=int(gate_id[5]),
            status=GateStatus.PASS,
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=(f"formal-probes/gate-{index}.json",),
        )
        key = "gate_" + "".join(
            character if character.isalnum() else "_"
            for character in gate_id.casefold()
        ).strip("_")
        evidence_refs[key] = _publish(
            root,
            f"formal-evidence/{task_id}/{key}",
            task_id="stage0.01_baseline_and_safety",
            kind="gate_record",
            payload=gate.to_dict(),
        )
    for capability in task.formal_eligibility.required_capabilities:
        item = RuntimeCapabilityEvidence(
            capability=capability,
            status="VERIFIED",
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=(f"formal-probes/{capability}.json",),
            metadata={"probe": "controlled-environment-test"},
        )
        evidence_refs[f"capability_{capability}"] = _publish(
            root,
            f"formal-evidence/{task_id}/capability-{capability}",
            task_id="stage0.01_baseline_and_safety",
            kind="runtime_capability_evidence",
            payload=item.to_dict(),
        )

    input_refs: list[str] = []
    for contract in task.input_artifacts:
        if not contract.required:
            continue
        producer = contract.producer_task_ids[0]
        for kind in contract.artifact_kinds:
            input_refs.append(
                _publish(
                    root,
                    f"formal-evidence/{task_id}/input-{kind}",
                    task_id=producer,
                    kind=kind,
                    payload={
                        "schema_version": "formal-upstream-test-v1",
                        "verified": True,
                    },
                )
            )
    return (
        TaskRuntimeEnvironment(
            capabilities=frozenset(task.formal_eligibility.required_capabilities),
            frozen_contract_stages=frozenset(
                task.formal_eligibility.required_contract_stages
            ),
            passed_gate_ids=frozenset(task.formal_eligibility.required_gate_ids),
            evidence_refs=evidence_refs,
        ),
        tuple(input_refs),
    )


def _formal_config(root: Path, task_id: str, input_refs: tuple[str, ...]) -> ResolvedConfigV2:
    return ResolvedConfigV2.resolve(
        _base_config(task_id=task_id, formal=True),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {"input_result_refs": list(input_refs)},
            "artifacts": {"output_dir": f"runs/formal/{task_id}"},
        },
    )


def test_claim_sets_without_committed_evidence_never_unlock_formal(tmp_path: Path) -> None:
    task_id = "stage0.01_baseline_and_safety"
    task = DEFAULT_TASK_CATALOG.get(task_id)
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(task.formal_eligibility.required_capabilities),
        frozen_contract_stages=frozenset(task.formal_eligibility.required_contract_stages),
    )
    runtime = TaskRuntime(workspace_root=tmp_path)
    runtime.register(build_stage01_runner_overrides(tmp_path)[task.runner_kind])

    result = runtime.execute(_formal_config(tmp_path, task_id, ()), environment=environment)

    assert result.status.value == "BLOCKED"
    assert {item.code for item in result.blockers} >= {
        BlockerCode.CONTRACT_UNFROZEN,
        BlockerCode.SERVER_UNREACHABLE,
        BlockerCode.CAPABILITY_UNAVAILABLE,
    }


@pytest.mark.parametrize(
    "task_id",
    (
        "stage0.01_baseline_and_safety",
        "stage0.02_storage_and_layout",
        "stage0.03_runtime_and_dependencies",
        "stage0.04_assets_and_manifests",
    ),
)
def test_external_stage0_formal_tasks_project_committed_evidence_not_local_fixture(
    tmp_path: Path,
    task_id: str,
) -> None:
    environment, input_refs = _formal_environment(tmp_path, task_id)
    config = _formal_config(tmp_path, task_id, input_refs)
    runtime = TaskRuntime(workspace_root=tmp_path)
    task = config.task_definition
    runtime.register(build_stage01_runner_overrides(tmp_path)[task.runner_kind])

    result = runtime.execute(config, environment=environment)

    assert result.status.value == "PASS", result.to_dict()
    loaded = load_committed_task_artifact(
        tmp_path,
        next(iter(result.artifact_refs.values())),
        require_formal=True,
    )
    core = loaded.payload["core_evidence"]
    assert core["evidence_type"] == "formal_committed_evidence"
    assert core["execution_mode"] == "formal_evidence_projection"
    assert core["local_fixture_executed"] is False
    assert loaded.payload["local_validation_status"] == "NOT_RUN"
    assert loaded.payload["gate_status"] == "NOT_RUN"


@pytest.mark.parametrize("failure_mode", ("direct", "local_envelope", "tampered"))
def test_bad_capability_evidence_is_a_structured_blocker(
    tmp_path: Path,
    failure_mode: str,
) -> None:
    task_id = "stage0.01_baseline_and_safety"
    environment, _ = _formal_environment(tmp_path, task_id)
    refs = dict(environment.evidence_refs)
    server_ref = refs["capability_server"]
    if failure_mode == "direct":
        capability = RuntimeCapabilityEvidence(
            capability="server",
            status="VERIFIED",
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=("formal-probes/server.json",),
        )
        direct = tmp_path / "formal-evidence" / "direct-server.json"
        write_canonical_json(direct, capability.to_dict())
        refs["capability_server"] = direct.relative_to(tmp_path).as_posix()
    elif failure_mode == "local_envelope":
        capability = RuntimeCapabilityEvidence(
            capability="server",
            status="VERIFIED",
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=("formal-probes/server.json",),
        )
        refs["capability_server"] = _publish(
            tmp_path,
            "formal-evidence/local-server-envelope",
            task_id="stage0.01_baseline_and_safety",
            kind="runtime_capability_evidence",
            payload=capability.to_dict(),
            formal=False,
        )
    else:
        commit = load_canonical_json(tmp_path / server_ref)
        object_path = tmp_path / commit["object_ref"]
        envelope = load_canonical_json(object_path)
        envelope["payload"]["status"] = "STALE"
        write_canonical_json(object_path, envelope)

    broken = TaskRuntimeEnvironment(
        capabilities=environment.capabilities,
        frozen_contract_stages=environment.frozen_contract_stages,
        passed_gate_ids=environment.passed_gate_ids,
        evidence_refs=refs,
    )
    runtime = TaskRuntime(workspace_root=tmp_path)
    task = DEFAULT_TASK_CATALOG.get(task_id)
    runtime.register(build_stage01_runner_overrides(tmp_path)[task.runner_kind])

    result = runtime.execute(_formal_config(tmp_path, task_id, ()), environment=broken)

    assert result.status.value == "BLOCKED"
    server = [item for item in result.blockers if item.requirement == "server"]
    assert len(server) == 1
    assert server[0].code is BlockerCode.SERVER_UNREACHABLE
    assert server[0].evidence_refs == (refs["capability_server"],)


def test_environment_build_cli_covers_empty_and_fully_indexed_examples(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_path = tmp_path / "empty-environment.json"
    assert main(["task", "environment-build", "--output", str(empty_path)]) == 0
    empty = TaskRuntimeEnvironment.from_mapping(load_canonical_json(empty_path))
    assert empty == TaskRuntimeEnvironment()
    capsys.readouterr()

    full_path = tmp_path / "full-environment.json"
    arguments = [
        "task",
        "environment-build",
        "--capability",
        "server",
        "--contract-stage",
        "0",
        "--passed-gate",
        "stage0.G1",
        "--evidence",
        "capability_server=evidence/capability-server/commits/runtime.json",
        "--evidence",
        "contract_stage_0=evidence/contract/commits/freeze.json",
        "--evidence",
        "gate_stage0_g1=evidence/gates/commits/g1.json",
        "--decision-ref",
        "evidence/decision/commits/estimator.json",
        "--output",
        str(full_path),
    ]
    assert main(arguments) == 0
    full = TaskRuntimeEnvironment.from_mapping(load_canonical_json(full_path))
    assert full.capabilities == frozenset({"server"})
    assert full.frozen_contract_stages == frozenset({0})
    assert full.passed_gate_ids == frozenset({"stage0.G1"})
    assert len(full.environment_hash) == 64


def test_runtime_capability_artifact_validate_and_default_workspace_binding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = RuntimeCapabilityEvidence(
        capability="server",
        status="BLOCKED",
        checked_at="2026-07-22T00:00:00+00:00",
        evidence_refs=("formal-probes/server.json",),
    )
    path = tmp_path / "capability.json"
    write_canonical_json(path, evidence.to_dict())
    assert main(["artifact-validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "runtime_capability_evidence"

    monkeypatch.chdir(tmp_path)
    assert build_default_task_runtime().workspace_root == tmp_path.resolve()


def test_offline_training_missing_manifest_is_structured_asset_blocker(
    tmp_path: Path,
) -> None:
    task_id = "stage0.06_single_gpu_smoke"
    base = _base_config(task_id=task_id, formal=True).to_dict()
    base["runtime"]["device"] = "cuda"
    config = ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(base),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 1, "max_epochs": None},
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": "missing/model.json",
                "model_root_ref": "missing/model",
                "data_manifest_ref": "missing/data.json",
                "data_root_ref": "missing/data",
                "tokenizer_manifest_ref": "missing/tokenizer.json",
                "tokenizer_root_ref": "missing/tokenizer",
                "task_name": "sst2",
                "num_labels": 2,
            },
            "evaluation": {
                "enabled": True,
                "split": "validation",
                "every_steps": 1,
                "batch_size": 1,
                "max_batches": 1,
                "metrics": ["loss", "accuracy"],
            },
            "artifacts": {"output_dir": "runs/missing-offline-assets"},
        },
    )
    request = TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=TaskRuntimeEnvironment(),
    )

    with pytest.raises(TaskBlockedError) as captured:
        TrainingTaskRunner(tmp_path).run(request)

    assert captured.value.blockers[0].code is BlockerCode.ASSET_UNAVAILABLE
    assert captured.value.blockers[0].requirement == "offline_training_assets"
