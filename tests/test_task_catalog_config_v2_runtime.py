from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts import (
    ConfigContractError,
    ContractFreeze,
    ContractState,
    GateRecord,
    GateStatus,
    load_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import (
    CONFIG_V2_SCHEMA_VERSION,
    ResolvedConfigV2,
    load_resolved_config_compatible,
)
from param_importance_nlp.contracts.task_catalog import (
    DEFAULT_TASK_CATALOG,
    RecoveryMode,
    RunnerKind,
    TaskCatalog,
    TaskCatalogError,
)
from param_importance_nlp.runtime.task_runtime import (
    BlockerCode,
    TaskExecutionRequest,
    TaskRunResult,
    TaskRunStatus,
    TaskRuntime,
    TaskRuntimeEnvironment,
)
from param_importance_nlp.runtime import TaskArtifactStore


ROOT = Path(__file__).resolve().parents[1]
V1_FIXTURE = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _legacy_mapping() -> dict[str, object]:
    value = load_canonical_json(V1_FIXTURE)
    assert isinstance(value, dict)
    return copy.deepcopy(value)


def _formal_mapping() -> dict[str, object]:
    value = _legacy_mapping()
    value["identity"]["run_intent"] = "formal"  # type: ignore[index]
    value["identity"]["formal_eligible"] = True  # type: ignore[index]
    value["identity"]["route"] = "pretrain"  # type: ignore[index]
    value["runtime"]["allow_dirty_worktree"] = False  # type: ignore[index]
    value["importance"]["estimator_decision_ref"] = "decisions/stage2.json"  # type: ignore[index]
    return value


def _for_stage(stage: int) -> dict[str, object]:
    value = _legacy_mapping()
    value["identity"]["stage"] = stage  # type: ignore[index]
    return value


def _offline_provider(root_prefix: str = "assets/local") -> dict[str, object]:
    return {
        "kind": "offline_hf",
        "model_manifest_ref": "manifests/model.json",
        "model_root_ref": f"{root_prefix}/model",
        "data_manifest_ref": "manifests/data.json",
        "data_root_ref": f"{root_prefix}/data",
        "tokenizer_manifest_ref": "manifests/tokenizer.json",
        "tokenizer_root_ref": f"{root_prefix}/tokenizer",
    }


def test_catalog_covers_every_numbered_stage0_to_stage3_plan() -> None:
    expected_plan_refs: set[str] = set()
    for stage in range(4):
        stage_dir = ROOT / "plan" / f"stage{stage}"
        for path in stage_dir.glob("[0-9][0-9]_*.md"):
            expected_plan_refs.add(path.relative_to(ROOT).as_posix())

    actual_plan_refs = {
        task.plan_ref for task in DEFAULT_TASK_CATALOG.tasks if task.stage <= 3
    }
    assert len(expected_plan_refs) == 44
    assert actual_plan_refs == expected_plan_refs


def test_stage4_to_stage9_catalog_refs_point_to_detailed_task_anchors() -> None:
    tasks = [task for task in DEFAULT_TASK_CATALOG.tasks if task.stage >= 4]
    assert len(tasks) == 38
    for task in tasks:
        document_ref, anchor = task.plan_ref.split("#", maxsplit=1)
        assert document_ref == f"plan/stage{task.stage}/README.md"
        document = (ROOT / document_ref).read_text(encoding="utf-8")
        assert f'<a id="{anchor}"></a>' in document, task.task_id
    assert len(DEFAULT_TASK_CATALOG.tasks) == 82

    assert set(DEFAULT_TASK_CATALOG.task_ids) >= {
        "stage4.minimal_complete_loop",
        "stage5.formal_pretraining",
        "stage6.training_route_comparison",
        "stage7.functional_pruning_validation",
        "stage8.ablation_and_robustness",
        "stage9.analysis_visualization_reporting",
    }

    expected_leaf_ids = {
        "stage4.route", "stage4.pretrain", "stage4.direct_supervised",
        "stage4.finetune", "stage4.importance_trajectory",
        "stage4.pruning_validation", "stage4.report",
        "stage5.pretrain", "stage5.importance_trajectory",
        "stage5.checkpoint_analysis", "stage5.report",
        "stage6.route_matrix", "stage6.evaluate", "stage6.compare",
        "stage6.importance_reuse", "stage6.report",
        "stage7.matrix", "stage7.evaluate", "stage7.reduce", "stage7.report",
        "stage8.freeze", "stage8.execute", "stage8.reduce",
        "stage8.recommend", "stage8.report",
        "stage9.ingest", "stage9.statistics", "stage9.tables", "stage9.charts",
        "stage9.report", "stage9.bundle", "stage9.replay",
    }
    assert expected_leaf_ids <= set(DEFAULT_TASK_CATALOG.task_ids)


def test_every_task_freezes_runner_config_artifacts_recovery_and_formal_policy() -> None:
    for task in DEFAULT_TASK_CATALOG.tasks:
        assert isinstance(task.runner_kind, RunnerKind)
        assert task.config_paths
        assert task.artifact_kinds
        assert isinstance(task.recovery_mode, RecoveryMode)
        assert task.safe_boundary.value
        assert task.formal_eligibility.required_contract_stages == tuple(
            range(task.stage + 1)
        )
        assert task.to_dict()["schema_version"] == "task-definition-v2"


def test_task_catalog_roundtrip_is_hash_bound_and_unknown_fields_fail() -> None:
    wire = DEFAULT_TASK_CATALOG.to_dict()
    loaded = TaskCatalog.from_mapping(wire)
    assert loaded.catalog_hash == DEFAULT_TASK_CATALOG.catalog_hash
    assert loaded.task_ids == DEFAULT_TASK_CATALOG.task_ids

    tampered = copy.deepcopy(wire)
    tampered["tasks"][0]["title"] = "tampered"  # type: ignore[index]
    with pytest.raises(TaskCatalogError, match="catalog_hash"):
        TaskCatalog.from_mapping(tampered)

    unknown = copy.deepcopy(wire)
    unknown["unexpected"] = True
    with pytest.raises(TaskCatalogError, match="字段集合"):
        TaskCatalog.from_mapping(unknown)


def test_legacy_v1_requires_explicit_canonical_task_and_loads_without_mutation() -> None:
    legacy = _legacy_mapping()
    with pytest.raises(ConfigContractError, match="显式提供 task_id"):
        load_resolved_config_compatible(legacy)

    resolved = load_resolved_config_compatible(
        legacy,
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    assert resolved.task_id == "stage0.05_config_run_identity_and_seeds"
    assert resolved.run_intent == "local_fixture"
    assert resolved.section("execution")["runner_kind"] == "contract"  # type: ignore[index]
    assert resolved.base_config.to_dict() == legacy
    assert resolved.to_dict()["schema_version"] == CONFIG_V2_SCHEMA_VERSION

    roundtrip = ResolvedConfigV2.from_mapping(resolved.to_dict())
    assert roundtrip.config_hash == resolved.config_hash
    assert roundtrip.full_hash == resolved.full_hash


def test_v2_training_length_is_explicit_and_cross_fields_fail_closed() -> None:
    legacy = _legacy_mapping()
    with pytest.raises(ConfigContractError, match="training.max_steps"):
        load_resolved_config_compatible(
            legacy,
            task_id="stage0.06_single_gpu_smoke",
        )

    resolved = load_resolved_config_compatible(
        legacy,
        task_id="stage0.06_single_gpu_smoke",
        overrides={
            "training": {"max_steps": 3, "validation_every_steps": 1},
            "scheduler": {"kind": "linear", "total_steps": 3},
        },
    )
    assert resolved.section("training")["max_steps"] == 3  # type: ignore[index]

    with pytest.raises(ConfigContractError, match="total_steps"):
        load_resolved_config_compatible(
            legacy,
            task_id="stage0.06_single_gpu_smoke",
            overrides={
                "training": {"max_steps": 3},
                "scheduler": {"kind": "linear", "total_steps": 2},
            },
        )
    with pytest.raises(ConfigContractError, match="未知字段"):
        load_resolved_config_compatible(
            legacy,
            task_id="stage0.05_config_run_identity_and_seeds",
            overrides={"execution": {"magic": True}},
        )


def test_v2_semantic_hash_excludes_machine_output_and_retry_policy() -> None:
    first = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    second = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={
            "execution": {"timeout_seconds": 1200, "max_attempts": 3},
            "recovery": {"max_restarts": 2},
            "artifacts": {"output_dir": "another/output"},
        },
    )
    assert first.config_hash == second.config_hash
    assert first.full_hash != second.full_hash

    semantic_change = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"execution": {"dry_run": True}},
    )
    assert semantic_change.config_hash != first.config_hash


def test_v2_exposes_strict_offline_execution_sections() -> None:
    config = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    assert config.section("providers") == {
        "kind": "tiny",
        "model_manifest_ref": None,
        "model_root_ref": None,
        "data_manifest_ref": None,
        "data_root_ref": None,
        "tokenizer_manifest_ref": None,
        "tokenizer_root_ref": None,
        "task_type": "synthetic",
        "task_name": "fixture",
        "num_labels": None,
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert config.section("launcher")["kind"] == "local"  # type: ignore[index]
    assert config.section("precision_runtime")["autocast_enabled"] is False  # type: ignore[index]
    assert config.section("checkpoint_schedule")["segments"] == [  # type: ignore[index]
        {"start_step": 0, "end_step": None, "every_steps": 10}
    ]
    assert config.section("orchestration")["paired_design"]["enabled"] is False  # type: ignore[index]


def test_provider_task_name_freezes_pile_and_glue_semantics() -> None:
    base = _for_stage(1)
    base["identity"]["task"] = "stage1.06_training_integration_and_accumulators"  # type: ignore[index]
    base["loss"]["task_type"] = "sequence_classification"  # type: ignore[index]
    base["loss"]["weighting"] = "sample"  # type: ignore[index]
    base["data"]["statistical_unit"] = "sample"  # type: ignore[index]
    base["data"]["weight_unit"] = "sample"  # type: ignore[index]
    base["model"]["architecture"] = "tiny-sequence-classifier"  # type: ignore[index]
    valid = load_resolved_config_compatible(
        base,
        task_id="stage1.06_training_integration_and_accumulators",
        overrides={
            "training": {"max_steps": 1},
            "providers": {
                "task_type": "sequence_classification",
                "task_name": "mnli",
                "num_labels": 3,
            },
        },
    )
    assert valid.section("providers")["task_name"] == "mnli"  # type: ignore[index]

    with pytest.raises(ConfigContractError, match="num_labels=3"):
        load_resolved_config_compatible(
            base,
            task_id="stage1.06_training_integration_and_accumulators",
            overrides={
                "training": {"max_steps": 1},
                "providers": {
                    "task_type": "sequence_classification",
                    "task_name": "mnli",
                    "num_labels": 2,
                },
            },
        )

    with pytest.raises(ConfigContractError, match="causal_lm"):
        load_resolved_config_compatible(
            base,
            task_id="stage1.06_training_integration_and_accumulators",
            overrides={
                "training": {"max_steps": 1},
                "providers": {
                    "task_type": "sequence_classification",
                    "task_name": "pile",
                    "num_labels": 2,
                },
            },
        )


def test_formal_model_execution_requires_explicit_local_hf_manifests_and_roots() -> None:
    formal = _formal_mapping()
    formal["runtime"]["device"] = "cuda"  # type: ignore[index]
    with pytest.raises(ConfigContractError, match="offline_hf"):
        load_resolved_config_compatible(
            formal,
            task_id="stage0.06_single_gpu_smoke",
            overrides={"training": {"max_steps": 1}},
        )

    config = load_resolved_config_compatible(
        formal,
        task_id="stage0.06_single_gpu_smoke",
        overrides={
            "training": {"max_steps": 1},
            "providers": _offline_provider(),
        },
    )
    assert config.section("providers")["kind"] == "offline_hf"  # type: ignore[index]
    assert config.section("providers")["local_files_only"] is True  # type: ignore[index]


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"evaluation": {"metrics": ["loss"]}}, "evaluation.enabled=false"),
        ({"profiling": {"measure_steps": 3}}, "profiling.enabled=false"),
        (
            {"precision_runtime": {"autocast_enabled": True, "autocast_dtype": "bfloat16"}},
            "autocast_enabled",
        ),
        (
            {
                "checkpoint_schedule": {
                    "segments": [
                        {"start_step": 0, "end_step": 5, "every_steps": 1},
                        {"start_step": 6, "end_step": None, "every_steps": 2},
                    ]
                }
            },
            "必须连续",
        ),
        ({"launcher": {"world_size": 2}}, "launcher backend/world_size"),
        ({"optimizer_runtime": {"maximize": True}}, "不支持"),
    ],
)
def test_new_runtime_sections_fail_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ConfigContractError, match=message):
        load_resolved_config_compatible(
            _legacy_mapping(),
            task_id="stage0.05_config_run_identity_and_seeds",
            overrides=overrides,
        )


def test_training_profiling_window_must_fit_explicit_step_budget() -> None:
    task_id = "stage0.06_single_gpu_smoke"
    with pytest.raises(ConfigContractError, match="不能超过 training.max_steps"):
        load_resolved_config_compatible(
            _for_stage(0),
            task_id=task_id,
            overrides={
                "training": {"max_steps": 3},
                "profiling": {
                    "enabled": True,
                    "warmup_steps": 1,
                    "measure_steps": 2,
                    "repetitions": 2,
                    "capture_memory": False,
                    "capture_throughput": True,
                    "capture_communication": False,
                    "synchronize_device": False,
                },
            },
        )


def test_optimizer_runtime_adamw_and_torchrun_are_derived_from_v1_contract() -> None:
    adamw = _legacy_mapping()
    adamw["optimizer"]["type"] = "adamw"  # type: ignore[index]
    optimizer_config = load_resolved_config_compatible(
        adamw,
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    assert optimizer_config.section("optimizer_runtime")["betas"] == [0.9, 0.999]  # type: ignore[index]
    assert optimizer_config.section("optimizer_runtime")["eps"] == 1e-8  # type: ignore[index]

    distributed = _formal_mapping()
    distributed["distributed"]["world_size"] = 2  # type: ignore[index]
    distributed["distributed"]["backend"] = "gloo"  # type: ignore[index]
    distributed["distributed"]["device_ids"] = [0, 1]  # type: ignore[index]
    distributed["batching"]["global_batch_size"] = 8  # type: ignore[index]
    launcher_config = load_resolved_config_compatible(
        distributed,
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    assert launcher_config.section("launcher") == {
        "kind": "torchrun",
        "backend": "gloo",
        "world_size": 2,
        "init_method": "env",
        "init_ref": None,
        "rendezvous_id": "stage0.05_config_run_identity_and_seeds",
        "max_restarts": 0,
    }


def test_route_matrix_and_paired_design_references_are_task_specific() -> None:
    stage4 = _for_stage(4)
    with pytest.raises(ConfigContractError, match="route_spec_ref"):
        load_resolved_config_compatible(
            stage4,
            task_id="stage4.pretrain",
            overrides={"training": {"max_steps": 2}},
        )
    route = load_resolved_config_compatible(
        stage4,
        task_id="stage4.pretrain",
        overrides={
            "training": {"max_steps": 2},
            "orchestration": {"route_spec_ref": "routes/stage4.json"},
        },
    )
    assert route.section("orchestration")["route_spec_ref"] == "routes/stage4.json"  # type: ignore[index]

    stage2 = _for_stage(2)
    with pytest.raises(ConfigContractError, match="paired design"):
        load_resolved_config_compatible(
            stage2,
            task_id="stage2.05_paired_estimator_runner",
        )
    paired = load_resolved_config_compatible(
        stage2,
        task_id="stage2.05_paired_estimator_runner",
        overrides={
            "orchestration": {
                "paired_design": {
                    "enabled": True,
                    "design": "shared_draws",
                    "mapping_ref": "sampling/paired-mapping.json",
                    "budget_unit": "gradient_evaluations",
                }
            }
        },
    )
    assert paired.section("orchestration")["paired_design"]["design"] == "shared_draws"  # type: ignore[index]


def test_formal_stage4_training_requires_route_quadrature_assets_evaluation_and_nccl() -> None:
    formal = _formal_mapping()
    formal["identity"]["stage"] = 4  # type: ignore[index]
    formal["runtime"]["device"] = "cuda"  # type: ignore[index]
    formal["distributed"].update(  # type: ignore[union-attr]
        {"world_size": 2, "backend": "nccl", "device_ids": [0, 1]}
    )
    formal["batching"]["global_batch_size"] = 8  # type: ignore[index]
    common_overrides = {
        "training": {"max_steps": 2},
        "providers": _offline_provider(),
        "orchestration": {
            "route_spec_ref": "routes/stage4.json",
            "quadrature_decision_ref": "decisions/quadrature.json",
        },
    }
    with pytest.raises(ConfigContractError, match="必须启用 evaluation"):
        load_resolved_config_compatible(
            formal,
            task_id="stage4.pretrain",
            overrides=common_overrides,
        )

    valid_overrides = copy.deepcopy(common_overrides)
    valid_overrides["evaluation"] = {
        "enabled": True,
        "split": "validation",
        "batch_size": 2,
        "metrics": ["loss"],
    }
    config = load_resolved_config_compatible(
        formal,
        task_id="stage4.pretrain",
        overrides=valid_overrides,
    )
    assert config.section("launcher")["kind"] == "torchrun"  # type: ignore[index]
    assert config.section("launcher")["backend"] == "nccl"  # type: ignore[index]
    assert config.section("evaluation")["enabled"] is True  # type: ignore[index]


def test_provider_root_placement_is_nonsemantic_but_manifests_are_semantic() -> None:
    first = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"providers": _offline_provider("assets/a")},
    )
    second = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"providers": _offline_provider("assets/b")},
    )
    assert first.config_hash == second.config_hash
    assert first.full_hash != second.full_hash

    changed_manifest = _offline_provider("assets/b")
    changed_manifest["model_manifest_ref"] = "manifests/model-v2.json"
    third = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"providers": changed_manifest},
    )
    assert third.config_hash != first.config_hash


class _SuccessfulContractRunner:
    runner_kind = RunnerKind.CONTRACT

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        refs = {
            kind: f"artifacts/{request.task.task_id}/{kind}.json"
            for kind in request.task.artifact_kinds
        }
        return TaskRunResult.passed(request, artifact_refs=refs)


class _IncompleteContractRunner:
    runner_kind = RunnerKind.CONTRACT

    def run(self, request: TaskExecutionRequest) -> TaskRunResult:
        first = request.task.artifact_kinds[0]
        return TaskRunResult.passed(
            request,
            artifact_refs={first: f"artifacts/{first}.json"},
        )


def test_missing_runner_is_structured_blocked_and_result_roundtrips() -> None:
    config = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    result = TaskRuntime().execute(config)

    assert result.status is TaskRunStatus.BLOCKED
    assert result.formal_eligible is False
    assert [blocker.code for blocker in result.blockers] == [
        BlockerCode.RUNNER_UNAVAILABLE
    ]
    assert TaskRunResult.from_mapping(result.to_dict()).result_hash == result.result_hash

    tampered = result.to_dict()
    tampered["message"] = "tampered"
    with pytest.raises(Exception, match="result_hash"):
        TaskRunResult.from_mapping(tampered)


def test_formal_preflight_reports_all_missing_evidence_without_calling_runner() -> None:
    config = load_resolved_config_compatible(
        _formal_mapping(),
        task_id="stage0.01_baseline_and_safety",
    )
    result = TaskRuntime().execute(config)
    assert result.status is TaskRunStatus.BLOCKED
    codes = {blocker.code for blocker in result.blockers}
    assert BlockerCode.RUNNER_UNAVAILABLE in codes
    assert BlockerCode.CONTRACT_UNFROZEN in codes
    assert BlockerCode.SERVER_UNREACHABLE in codes
    assert BlockerCode.CAPABILITY_UNAVAILABLE in codes  # git
    assert result.formal_eligible is False


def test_registered_runner_can_pass_local_but_incomplete_artifacts_fail() -> None:
    config = load_resolved_config_compatible(
        _legacy_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
    )
    runtime = TaskRuntime()
    runtime.register(_SuccessfulContractRunner())
    result = runtime.execute(config)
    assert result.status is TaskRunStatus.PASS
    assert result.formal_eligible is False
    assert tuple(result.artifact_refs) == config.task_definition.artifact_kinds

    broken_runtime = TaskRuntime()
    broken_runtime.register(_IncompleteContractRunner())
    broken = broken_runtime.execute(config)
    assert broken.status is TaskRunStatus.FAIL
    assert broken.error_code == "runner_contract_violation"


def test_formal_pass_requires_hash_bound_catalog_evidence(tmp_path: Path) -> None:
    freeze = ContractFreeze(
        contract_id="stage0.contract.runtime-test",
        stage=0,
        scope="formal",
        state=ContractState.FROZEN,
        formula_version="runtime-test-v1",
        config_hash="1" * 64,
        schema_hashes={"schema": "2" * 64},
        source_hashes={"source": "3" * 64},
        required_gate_ids=("stage0.G0-C",),
        frozen_at="2026-07-22T00:00:00+00:00",
    )

    def publish(
        output: str,
        *,
        task_id: str,
        kind: str,
        payload: dict[str, object],
    ) -> str:
        return TaskArtifactStore(tmp_path, output).publish(
            task_id=task_id,
            artifact_kind=kind,
            config_hash="4" * 64,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
        ).commit_ref

    freeze_ref = publish(
        "evidence/contract",
        task_id="stage0.01_baseline_and_safety",
        kind="contract_freeze",
        payload=freeze.to_dict(),
    )
    gate_refs: dict[str, str] = {}
    for index, gate_id in enumerate(("stage0.G0-C", "stage0.G1")):
        gate = GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at="2026-07-22T00:00:00+00:00",
            evidence_refs=(f"evidence/gate-source-{index}.json",),
        )
        key = "gate_" + gate_id.casefold().replace(".", "_").replace("-", "_")
        gate_refs[key] = publish(
            f"evidence/{key}",
            task_id="stage0.01_baseline_and_safety",
            kind="gate_record",
            payload=gate.to_dict(),
        )

    upstream_refs = tuple(
        publish(
            f"evidence/upstream/{kind}",
            task_id="stage0.04_assets_and_manifests",
            kind=kind,
            payload={"schema_version": "formal-upstream-test-v1", "verified": True},
        )
        for kind in ("asset_manifest", "asset_audit", "asset_resolution")
    )
    config = load_resolved_config_compatible(
        _formal_mapping(),
        task_id="stage0.05_config_run_identity_and_seeds",
        overrides={"orchestration": {"input_result_refs": list(upstream_refs)}},
    )
    task = config.task_definition
    environment = TaskRuntimeEnvironment(
        frozen_contract_stages=frozenset(task.formal_eligibility.required_contract_stages),
        passed_gate_ids=frozenset(task.formal_eligibility.required_gate_ids),
        capabilities=frozenset(task.formal_eligibility.required_capabilities),
        evidence_refs={"contract_stage_0": freeze_ref, **gate_refs},
    )
    runtime = TaskRuntime(workspace_root=tmp_path)
    runtime.register(_SuccessfulContractRunner())
    result = runtime.execute(config, environment=environment)
    assert result.status is TaskRunStatus.PASS
    assert result.formal_eligible is True


def test_new_v2_schemas_are_strict_json_objects() -> None:
    schema_names = {
        "task-definition-v2.json",
        "task-catalog-v2.json",
        "resolved-config-v2.json",
        "task-run-result-v2.json",
    }
    for name in schema_names:
        value = json.loads((ROOT / "schemas" / "shared" / name).read_text(encoding="utf-8"))
        assert value["$schema"].endswith("2020-12/schema")
        assert value["type"] == "object"
        assert value["additionalProperties"] is False

    resolved_schema = json.loads(
        (ROOT / "schemas" / "shared" / "resolved-config-v2.json").read_text(
            encoding="utf-8"
        )
    )
    assert {
        "providers", "evaluation", "profiling", "checkpoint_schedule",
        "precision_runtime", "optimizer_runtime", "launcher", "orchestration",
    } <= set(resolved_schema["required"])
