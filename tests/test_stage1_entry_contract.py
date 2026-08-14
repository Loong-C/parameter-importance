from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    load_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from param_importance_nlp.experiments.stage01_task_runners import (
    _STAGE1_ENTRY_CACHE_VARS,
    _stage1_git_snapshot,
    build_stage01_runner_overrides,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRuntime,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(
    *, formal: bool, output: str, input_refs: tuple[str, ...] = ()
) -> ResolvedConfigV2:
    task_id = "stage1.01_entry_and_contract"
    value = deepcopy(load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json"))
    value["identity"].update(
        {
            "stage": 1,
            "task": task_id,
            "run_intent": "formal" if formal else "local_fixture",
            "formal_eligible": formal,
            "route": "pretrain",
        }
    )
    value["runtime"]["allow_dirty_worktree"] = not formal
    value["loss"].update(
        {"task_type": "sequence_classification", "weighting": "sample"}
    )
    value["data"].update(
        {"statistical_unit": "sample", "weight_unit": "sample"}
    )
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(value),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 3},
            "orchestration": {"input_result_refs": list(input_refs)},
            "artifacts": {"output_dir": output},
        },
    )


def _upstream_environment(root: Path) -> tuple[TaskRuntimeEnvironment, tuple[str, ...]]:
    freeze = ContractFreeze(
        contract_id="stage0.contract.formal-test",
        stage=0,
        scope="formal",
        state=ContractState.FROZEN,
        formula_version="stage0-formal-test-v1",
        config_hash="a" * 64,
        schema_hashes={"schema": "b" * 64},
        source_hashes={"source": "c" * 64},
        required_gate_ids=("stage0.G10",),
        frozen_at="2026-08-14T00:00:00Z",
    )
    gate = GateRecord(
        gate_id="stage0.G10",
        stage=0,
        status=GateStatus.PASS,
        checked_at="2026-08-14T00:00:00Z",
        evidence_refs=("stage0/g10/sync.json",),
    )
    store = TaskArtifactStore(root, "formal-upstream")
    contract_ref = store.publish(
        task_id="stage0.05_config_run_identity_and_seeds",
        artifact_kind="resolved_config",
        config_hash="a" * 64,
        run_intent="formal",
        payload=freeze.to_dict(),
        formal_eligible=True,
    ).commit_ref
    input_refs = tuple(
        store.publish(
            task_id="stage0.12_delivery_and_sync",
            artifact_kind=kind,
            config_hash="a" * 64,
            run_intent="formal",
            payload={"schema_version": "stage0-g10-upstream-test-v1", "kind": kind},
            formal_eligible=True,
        ).commit_ref
        for kind in ("delivery_manifest", "worklog", "sync_report")
    )
    gate_ref = TaskArtifactStore(root, "formal-upstream-gate").publish(
        task_id="stage0.12_delivery_and_sync",
        artifact_kind="gate_record",
        config_hash="a" * 64,
        run_intent="formal",
        payload=gate.to_dict(),
        formal_eligible=True,
    ).commit_ref
    return (
        TaskRuntimeEnvironment(
            frozen_contract_stages=frozenset({0}),
            passed_gate_ids=frozenset({"stage0.G10"}),
            evidence_refs={
                "contract_stage_0": contract_ref,
                "gate_stage0_g10": gate_ref,
            },
        ),
        input_refs,
    )


def test_local_entry_contract_records_are_never_formal_eligible(tmp_path: Path) -> None:
    runtime = TaskRuntime(workspace_root=tmp_path)
    runtime.register(
        build_stage01_runner_overrides(tmp_path)[RunnerKind.CONTRACT]
    )

    result = runtime.execute(_config(formal=False, output="runs/local-entry"))

    assert result.status.value == "PASS"
    artifact = load_committed_task_artifact(
        tmp_path, result.artifact_refs["stage_contract"], require_formal=False
    )
    core = artifact.payload["core_evidence"]
    assert core["entry_snapshot"]["scope"] == "local_fixture"
    assert core["entry_snapshot"]["formal_eligible"] is False
    assert core["formal_gate_records"] == []
    assert ContractFreeze.from_mapping(core["contract_freeze"]).formal_eligible is False
    assert core["requirements_matrix"]["schema_version"] == "stage1-requirements-matrix-v1"


def test_formal_entry_contract_publishes_freeze_and_two_gate_records(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "param_importance_nlp.experiments.stage01_task_runners._stage1_git_snapshot",
        lambda _root: {
            "available": True,
            "branch": "feat/stage1-cpu-evidence",
            "head": "a" * 40,
            "worktree_clean": True,
            "remote_names": ["origin"],
        },
    )
    monkeypatch.setenv("PARAM_IMPORTANCE_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("PARAM_IMPORTANCE_TMP_ROOT", str(tmp_path / "tmp"))
    for variable in _STAGE1_ENTRY_CACHE_VARS:
        monkeypatch.setenv(variable, str(tmp_path / "cache" / variable.lower()))

    environment, input_refs = _upstream_environment(tmp_path)
    runtime = TaskRuntime(workspace_root=tmp_path)
    runtime.register(
        build_stage01_runner_overrides(tmp_path)[RunnerKind.CONTRACT]
    )

    result = runtime.execute(
        _config(formal=True, output="runs/formal-entry", input_refs=input_refs),
        environment=environment,
    )

    assert result.status.value == "PASS", [
        (item.code.value, item.requirement, item.message) for item in result.blockers
    ]
    stage_contract = load_committed_task_artifact(
        tmp_path, result.artifact_refs["stage_contract"], require_formal=True
    )
    gate_artifact = load_committed_task_artifact(
        tmp_path, result.artifact_refs["gate_record"], require_formal=True
    )
    core = stage_contract.payload["core_evidence"]
    freeze = ContractFreeze.from_mapping(core["contract_freeze"])
    assert freeze.formal_eligible is True
    assert core["entry_snapshot"]["formal_eligible"] is True
    assert [item["gate_id"] for item in core["formal_gate_records"]] == [
        "stage1.G1-ENTRY",
        "stage1.G1-CONTRACT",
    ]
    assert all(item["status"] == "PASS" for item in core["formal_gate_records"])

    next_environment = TaskRuntimeEnvironment(
        frozen_contract_stages=frozenset({0, 1}),
        passed_gate_ids=frozenset(
            {"stage0.G10", "stage1.G1-ENTRY", "stage1.G1-CONTRACT"}
        ),
        evidence_refs={
            "contract_stage_0": environment.evidence_refs["contract_stage_0"],
            "contract_stage_1": stage_contract.identity.commit_ref,
            "gate_stage0_g10": environment.evidence_refs["gate_stage0_g10"],
            "gate_stage1_g1_entry": gate_artifact.identity.commit_ref,
            "gate_stage1_g1_contract": gate_artifact.identity.commit_ref,
        },
    )
    assert runtime._verified_contract_ref(next_environment, 1)[0] is True
    assert runtime._verified_gate_ref(next_environment, "stage1.G1-ENTRY")[0] is True
    assert runtime._verified_gate_ref(next_environment, "stage1.G1-CONTRACT")[0] is True
