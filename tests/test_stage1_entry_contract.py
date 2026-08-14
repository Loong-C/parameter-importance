from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.cli import _validate_project_json_schema
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from param_importance_nlp.experiments.stage01_task_runners import (
    _STAGE1_ENTRY_CACHE_VARS,
    _STAGE1_ENTRY_REQUIRED_STAGE0_GATES,
    _stage1_git_snapshot,
    build_stage01_runner_overrides,
)
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRuntime,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from param_importance_nlp.runtime.optimizer import OptimizerBridge
from param_importance_nlp.stage1_contract import (
    STAGE1_REQUIREMENT_IDS,
    STAGE1_TRACEABILITY_REGISTRY,
    Stage1ContractError,
    build_stage1_math_contract,
    validate_stage1_math_contract,
    validate_stage1_requirements_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(
    *,
    formal: bool,
    output: str,
    input_refs: tuple[str, ...] = (),
    g10_index_ref: str | None = None,
    reuse_attestation_ref: str | None = None,
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
        {"task_type": "causal_lm", "reduction": "mean", "weighting": "effective_token"}
    )
    value["data"].update(
        {
            "asset_id": "2" * 64,
            "revision": "fixture-data-revision",
            "sequence_length": 2048,
            "statistical_unit": "sequence",
            "weight_unit": "effective_token",
        }
    )
    value["model"].update(
        {
            "asset_id": "1" * 64,
            "revision": "fixture-model-revision",
            "tokenizer_asset_id": "3" * 64,
            "initialization_id": "4" * 64,
            "architecture": "pythia-14m",
        }
    )
    provider_refs = {
        "model_manifest_ref": None if formal else "manifests/model.json",
        "model_root_ref": None if formal else "models/fixture",
        "data_manifest_ref": None if formal else "manifests/data.json",
        "data_root_ref": None if formal else "datasets/fixture",
        "tokenizer_manifest_ref": None if formal else "manifests/tokenizer.json",
        "tokenizer_root_ref": None if formal else "tokenizers/fixture",
    }
    return ResolvedConfigV2.resolve(
        ResolvedConfig.from_mapping(value),
        task_id=task_id,
        overrides={
            "providers": {
                "kind": "offline_hf",
                "task_type": "causal_lm",
                "task_name": "pile",
                "num_labels": None,
                **provider_refs,
                "local_files_only": True,
                "trust_remote_code": False,
            },
            "orchestration": {
                "route_spec_ref": g10_index_ref,
                "quadrature_decision_ref": reuse_attestation_ref,
                "input_result_refs": list(input_refs),
            },
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
            passed_gate_ids=frozenset(_STAGE1_ENTRY_REQUIRED_STAGE0_GATES),
            evidence_refs={
                "contract_stage_0": contract_ref,
                "gate_stage0_g10": gate_ref,
            },
        ),
        input_refs,
    )


def _g10_index(root: Path, *, commit: str = "a" * 40) -> str:
    reference = "evidence/stage0/g10-formal/fixture/index.json"
    value: dict[str, object] = {
        "schema_version": "stage0-g10-formalization-index-v1",
        "generator_git_commit": commit,
        "next_task_id": "stage1.01_entry_and_contract",
    }
    value["artifact_hash"] = canonical_json_hash(value)
    write_canonical_json(root / reference, value)
    return reference


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
    validate_stage1_math_contract(core["contract"])
    validate_stage1_requirements_matrix(core["requirements_matrix"])
    assert core["requirements_matrix"]["schema_version"] == "stage1-requirements-matrix-v2"


def test_local_entry_contract_wire_projection_is_identical_across_clean_roots(
    tmp_path: Path,
) -> None:
    def execute(root: Path) -> tuple[dict[str, str], dict[str, object]]:
        runtime = TaskRuntime(workspace_root=root)
        runtime.register(build_stage01_runner_overrides(root)[RunnerKind.CONTRACT])
        result = runtime.execute(_config(formal=False, output="runs/local-entry"))
        assert result.status.value == "PASS"
        hashes = {
            role: load_committed_task_artifact(
                root,
                reference,
                require_formal=False,
            ).identity.artifact_hash
            for role, reference in result.artifact_refs.items()
        }
        stage_contract = load_committed_task_artifact(
            root,
            result.artifact_refs["stage_contract"],
            require_formal=False,
        )
        return hashes, dict(stage_contract.payload["core_evidence"])

    first_hashes, first_core = execute(tmp_path / "first-clean-root")
    second_hashes, second_core = execute(tmp_path / "second-clean-root")

    assert set(first_hashes) == {"stage_contract", "requirements_matrix", "gate_record"}
    assert first_hashes == second_hashes
    assert first_core == second_core
    entry = first_core["entry_snapshot"]
    assert entry["scope"] == "local_fixture"
    assert entry["formal_eligible"] is False
    assert entry["checked_at"] == "1970-01-01T00:00:00Z"
    assert entry["runtime"]["checked_at_policy"] == (
        "deterministic_fixture_epoch_not_wall_clock_evidence"
    )
    path_audit = entry["write_path_audit"]
    assert path_audit["path_projection"] == "logical_path_roles"
    assert path_audit["repository_root"] == "source_repository"
    assert path_audit["output_root"] == "task_artifact_store"
    assert path_audit["approved_roots"] == ["source_repository"]
    assert str(tmp_path.resolve().as_posix()) not in json.dumps(first_core, sort_keys=True)
    assert ContractFreeze.from_mapping(first_core["contract_freeze"]).formal_eligible is False


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
    g10_index_ref = _g10_index(tmp_path)
    runtime = TaskRuntime(workspace_root=tmp_path)
    runtime.register(
        build_stage01_runner_overrides(tmp_path)[RunnerKind.CONTRACT]
    )

    result = runtime.execute(
        _config(
            formal=True,
            output="runs/formal-entry",
            input_refs=input_refs,
            g10_index_ref=g10_index_ref,
        ),
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
    assert freeze.formula_version == "stage1-entry-contract-v3"
    assert core["entry_snapshot"]["formal_eligible"] is True
    assert core["entry_snapshot"]["checked_at"] != "1970-01-01T00:00:00Z"
    assert "checked_at_policy" not in core["entry_snapshot"]["runtime"]
    assert "path_projection" not in core["entry_snapshot"]["write_path_audit"]
    assert core["entry_snapshot"]["write_path_audit"]["data_root"] == (
        tmp_path.resolve().as_posix()
    )
    validate_stage1_math_contract(core["contract"])
    validate_stage1_requirements_matrix(core["requirements_matrix"])
    assert core["contract"]["estimators"]["weighted_u"]["field_name"] == (
        "local_gradient_space_importance_u_weighted"
    )
    assert core["contract"]["estimators"]["actual_update_raw"]["formula"] == (
        "-data_delta_k,t * mean_gradient_k,t"
    )
    data_delta, mean_gradient = -0.25, 2.0
    assert -(data_delta * mean_gradient) == pytest.approx(0.5)
    assert core["contract"]["accumulators"]["negative_mass"]["formula"] == (
        "sum_t max(-score_k,t, 0)"
    )
    assert {
        item: core["contract"]["accumulators"][item]["formula"]
        for item in (
            "movement_data",
            "net_movement_data",
            "movement_total",
            "net_movement_total",
            "movement_weight_decay",
            "net_movement_weight_decay",
        )
    } == {
        "movement_data": "sum_t abs(data_delta_k,t)",
        "net_movement_data": "abs(sum_t data_delta_k,t)",
        "movement_total": "sum_t abs(total_delta_k,t)",
        "net_movement_total": "abs(Theta_T,k - Theta_0,k)",
        "movement_weight_decay": "sum_t abs(weight_decay_delta_k,t)",
        "net_movement_weight_decay": "abs(sum_t weight_decay_delta_k,t)",
    }
    expected_field_requirements = {
        f"contract.field.{definition['field_name']}": definition["field_name"]
        for section in ("estimators", "accumulators")
        for definition in core["contract"][section].values()
    }
    assert {row["requirement_id"] for row in core["requirements_matrix"]["requirements"]} == set(
        STAGE1_REQUIREMENT_IDS
    )
    assert {
        row["requirement_id"]: row["math_field"]
        for row in core["requirements_matrix"]["requirements"]
        if row["math_field"] is not None
    } == expected_field_requirements
    traceability_rows = {
        row["requirement_id"]: row for row in core["requirements_matrix"]["requirements"]
    }
    for requirement_id, expected in STAGE1_TRACEABILITY_REGISTRY.items():
        assert {
            field: traceability_rows[requirement_id][field]
            for field in expected
        } == expected
    assert traceability_rows[
        "contract.field.local_gradient_space_importance_u_weighted"
    ]["implementation_modules"] == [
        "src/param_importance_nlp/core/estimators.py",
        "src/param_importance_nlp/core/sufficient_statistics.py",
    ]
    assert traceability_rows[
        "contract.field.local_gradient_space_importance_u_weighted"
    ]["downstream_gate_ids"] == ["stage1.G1-EST"]
    assert traceability_rows[
        "contract.field.actual_update_raw_importance"
    ]["implementation_modules"] == [
        "src/param_importance_nlp/runtime/optimizer.py",
        "src/param_importance_nlp/runtime/training.py",
        "src/param_importance_nlp/core/accumulator.py",
    ]
    assert traceability_rows[
        "contract.field.actual_update_raw_importance"
    ]["artifact_roles"] == [
        "optimizer_movement_decomposition_trace",
        "actual_update_oracle_report",
        "g1_step_report",
    ]
    assert traceability_rows[
        "contract.field.parameter_movement_weight_decay"
    ]["downstream_gate_ids"] == ["stage1.G1-STEP"]
    assert traceability_rows["contract.loss_reduction"]["implementation_modules"] == [
        "src/param_importance_nlp/core/losses.py",
        "src/param_importance_nlp/runtime/gradients.py",
        "src/param_importance_nlp/runtime/training.py",
    ]
    assert traceability_rows["contract.loss_reduction"]["downstream_gate_ids"] == [
        "stage1.G1-GRAD"
    ]
    assert traceability_rows["contract.lifecycle"]["downstream_gate_ids"] == [
        "stage1.G1-GRAD",
        "stage1.G1-STEP",
    ]
    numerical_profiles = [
        {"device": "cpu", "dtype": "float64", "tolerance_profile": "T64_ORACLE"},
        {"device": "cuda:single_gpu", "dtype": "float32", "tolerance_profile": "T32_SINGLE"},
    ]
    assert traceability_rows[
        "contract.field.local_gradient_space_importance_u_weighted"
    ]["device_dtype_profiles"] == numerical_profiles
    assert traceability_rows[
        "contract.field.parameter_movement_weight_decay"
    ]["device_dtype_profiles"] == numerical_profiles
    missing_traceability = deepcopy(core["requirements_matrix"])
    missing_traceability["requirements"][0].pop("independent_oracles")
    with pytest.raises(Stage1ContractError, match="fields mismatch"):
        validate_stage1_requirements_matrix(missing_traceability)
    field_mapping_drift = deepcopy(core["requirements_matrix"])
    next(
        row for row in field_mapping_drift["requirements"] if row["math_field"] is not None
    )["math_field"] = "incorrect_field"
    with pytest.raises(Stage1ContractError, match="math_field traceability drift"):
        validate_stage1_requirements_matrix(field_mapping_drift)
    module_mapping_drift = deepcopy(core["requirements_matrix"])
    next(
        row
        for row in module_mapping_drift["requirements"]
        if row["requirement_id"] == "contract.field.local_gradient_space_importance_u_weighted"
    )["implementation_modules"] = [
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
        "src/param_importance_nlp/stage1_s1_1.py",
    ]
    with pytest.raises(Stage1ContractError, match="implementation_modules traceability drift"):
        validate_stage1_requirements_matrix(module_mapping_drift)
    gate_mapping_drift = deepcopy(core["requirements_matrix"])
    next(
        row
        for row in gate_mapping_drift["requirements"]
        if row["requirement_id"] == "contract.field.parameter_movement_weight_decay"
    )["minimal_repro_bundle"]["expected_gate_ids"] = ["stage1.G1-CONTRACT"]
    with pytest.raises(Stage1ContractError, match="minimal_repro_bundle traceability drift"):
        validate_stage1_requirements_matrix(gate_mapping_drift)
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


def test_formal_entry_rejects_cross_commit_g10_without_reuse_attestation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "param_importance_nlp.experiments.stage01_task_runners._stage1_git_snapshot",
        lambda _root: {
            "available": True,
            "branch": "feat/stage1-cpu-evidence",
            "head": "b" * 40,
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
    runtime.register(build_stage01_runner_overrides(tmp_path)[RunnerKind.CONTRACT])

    result = runtime.execute(
        _config(
            formal=True,
            output="runs/cross-commit-blocked",
            input_refs=input_refs,
            g10_index_ref=_g10_index(tmp_path, commit="a" * 40),
        ),
        environment=environment,
    )

    assert result.status.value == "BLOCKED"
    assert {item.requirement for item in result.blockers} >= {
        "stage0_g10_reuse_valid"
    }


def test_s1_1_formalization_schema_is_a_valid_project_schema() -> None:
    for relative in (
        "schemas/stage1/s1-1-formalization-index-v1.json",
        "schemas/stage1/math-contract-v1.json",
        "schemas/stage1/requirements-matrix-v2.json",
    ):
        path = ROOT / relative
        _validate_project_json_schema(json.loads(path.read_text(encoding="utf-8")))


def test_s1_1_contract_validator_rejects_formula_or_traceability_omission() -> None:
    contract = build_stage1_math_contract()
    contract["estimators"]["weighted_u"]["formula"] = "eta * weighted_mean_gradient**2"
    with pytest.raises(Stage1ContractError, match="formula, unit or lifecycle payload drift"):
        validate_stage1_math_contract(contract)


def test_actual_update_raw_contract_uses_signed_optimizer_data_delta() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    optimizer = torch.optim.SGD([parameter], lr=0.25, foreach=False)
    parameter.grad = torch.tensor([2.0], dtype=torch.float64)

    outcome = OptimizerBridge({"weight": parameter}, optimizer).step()
    data_delta = outcome.data_delta["weight"]
    mean_gradient = torch.tensor([2.0], dtype=torch.float64)

    torch.testing.assert_close(data_delta, torch.tensor([-0.5], dtype=torch.float64))
    torch.testing.assert_close(-data_delta * mean_gradient, torch.tensor([1.0], dtype=torch.float64))
    assert build_stage1_math_contract()["estimators"]["actual_update_raw"]["formula"] == (
        "-data_delta_k,t * mean_gradient_k,t"
    )
