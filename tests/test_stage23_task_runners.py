from __future__ import annotations

import copy
from pathlib import Path
import hashlib

import torch

import pytest

from param_importance_nlp.contracts import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import load_resolved_config_compatible
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from param_importance_nlp.core.registry import ParameterRegistry
from param_importance_nlp.experiments import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    DataFile,
    DataRangeManifest,
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_DATASET_ID,
    FORMAL_DATASET_REVISION,
    FORMAL_DATA_FILES,
    FORMAL_DATA_MANIFEST_SHA256,
    FORMAL_TOTAL_TRAINING_STEPS,
)
from param_importance_nlp.experiments.stage23_task_runners import (
    _formal_stage1_report_artifact_hash,
    _formal_stage2_asset_manifest,
    _load_formal_parameter_registry,
    _predecessor_context,
    _reference_tokenizer_identity,
    _run_stage2_contract,
    _run_formal_stage2_assets_and_sampling,
    _project_sampling_plan_to_provider,
    _sampling_plan_provider_compatible,
    _stage1_formal_bridge_identity,
    _REQUIRED_PREDECESSORS,
    build_stage23_runner_overrides,
    register_stage23_runners,
)
from param_importance_nlp.experiments import stage23_task_runners
from param_importance_nlp.experiments.stage2_formal import ReferenceSizingPlan
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse, STREAM_NAMES
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.task_runtime import (
    BlockerCode,
    TaskRunStatus,
    TaskBlockedError,
    TaskExecutionRequest,
    TaskRuntime,
    TaskRuntimeEnvironment,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _base_for(task_id: str) -> dict[str, object]:
    value = load_canonical_json(BASE_CONFIG)
    assert isinstance(value, dict)
    result = copy.deepcopy(value)
    result["identity"]["stage"] = DEFAULT_TASK_CATALOG.get(task_id).stage  # type: ignore[index]
    return result


def _config(
    task_id: str,
    output_dir: str,
    input_result_refs: tuple[str, ...] = (),
    *,
    resume_ref: str | None = None,
):
    orchestration: dict[str, object] = {
        "input_result_refs": list(input_result_refs),
    }
    overrides: dict[str, object] = {
        "artifacts": {"output_dir": output_dir},
        "orchestration": orchestration,
        "recovery": {"resume_ref": resume_ref},
    }
    if task_id in {
        "stage2.05_paired_estimator_runner",
        "stage2.07_main_sweep",
    }:
        orchestration["paired_design"] = {
            "enabled": True,
            "design": "shared_draws",
            "mapping_ref": "fixture/paired-mapping.json",
            "budget_unit": "gradient_evaluations",
        }
    return load_resolved_config_compatible(
        _base_for(task_id),
        task_id=task_id,
        overrides=overrides,
    )


def _runtime(root: Path) -> TaskRuntime:
    runtime = TaskRuntime()
    assert register_stage23_runners(runtime, root) is runtime
    return runtime


def _payload(root: Path, commit_ref: str) -> dict[str, object]:
    commit = load_canonical_json(root / Path(commit_ref))
    assert isinstance(commit, dict)
    body = load_canonical_json(root / Path(str(commit["object_ref"])))
    assert isinstance(body, dict)
    payload = body["payload"]
    assert isinstance(payload, dict)
    return payload


def test_formal_reference_tokenizer_manifest_missing_fails_closed() -> None:
    with pytest.raises(TaskBlockedError, match="hash-bound tokenizer manifest") as error:
        _reference_tokenizer_identity(
            None,
            formal=True,
            expected_asset_id="pythia-tokenizer",
            expected_checkpoint_id="checkpoint-14m-initialization",
        )
    assert error.value.blockers[0].requirement == "stage2_tokenizer_manifest"


def test_formal_reference_tokenizer_manifest_wrong_checkpoint_fails_closed() -> None:
    with pytest.raises(TaskBlockedError, match="checkpoint_id does not match"):
        _reference_tokenizer_identity(
            {
                "schema_version": "tokenizer-manifest-v1",
                "asset_id": "b" * 64,
                "revision": "tokenizer-revision",
                "checkpoint_id": "wrong-checkpoint",
            },
            formal=True,
            expected_asset_id="pythia-tokenizer",
            expected_checkpoint_id="checkpoint-14m-initialization",
        )


def test_formal_reference_tokenizer_manifest_valid_identity_is_hash_bound() -> None:
    identity = _reference_tokenizer_identity(
        {
            "schema_version": "tokenizer-manifest-v1",
            "asset_id": "a" * 64,
            "revision": "tokenizer-revision",
            "checkpoint_id": "checkpoint-14m-initialization",
        },
        formal=True,
        expected_asset_id="pythia-tokenizer",
        expected_checkpoint_id="checkpoint-14m-initialization",
    )
    assert identity["asset_id"] == "a" * 64
    assert identity["checkpoint_id"] == "checkpoint-14m-initialization"
    assert identity["identity_hash"] == canonical_json_hash(
        {key: value for key, value in identity.items() if key != "identity_hash"}
    )


def test_sampling_provider_compatibility_ignores_lineage_metadata_only() -> None:
    seeds = {name: index + 101 for index, name in enumerate(STREAM_NAMES)}
    upstream = SamplingPlan(
        SamplingUniverse(
            "s23-assets", tuple(range(8)), metadata={"asset_resolution_hash": "a"}
        ),
        seeds,
    )
    provider = SamplingPlan(
        SamplingUniverse(
            "offline-provider", tuple(range(8)), metadata={"provider_state_digest": "b"}
        ),
        seeds,
    )
    assert _sampling_plan_provider_compatible(upstream, provider)
    changed = dict(seeds)
    changed["pilot"] += 10
    assert not _sampling_plan_provider_compatible(
        upstream,
        SamplingPlan(SamplingUniverse("offline-provider", tuple(range(8))), changed),
    )


def test_sampling_provider_projection_rebinds_ids_but_preserves_s23_seeds() -> None:
    seeds = {name: index + 101 for index, name in enumerate(STREAM_NAMES)}
    upstream = SamplingPlan(
        SamplingUniverse("s23-assets", tuple(range(16)), metadata={"asset_resolution_hash": "a"}),
        seeds,
    )
    provider = SamplingPlan(
        SamplingUniverse(
            "offline-provider",
            tuple(f"pile:asset:record:{index:012d}" for index in range(8)),
            metadata={"provider_state_digest": "b"},
        ),
        seeds,
    )
    projected = _project_sampling_plan_to_provider(upstream, provider)
    assert projected.stream_seeds == upstream.stream_seeds
    assert projected.universe.sample_ids == provider.universe.sample_ids
    assert projected.universe.metadata["upstream_sampling_plan_hash"] == upstream.digest
    assert projected.draws("pilot", 8) == projected.draws("pilot", 8)


def test_formal_parameter_registry_reloads_authoritative_coordinate_manifest_and_tamper_blocks(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    registry = ParameterRegistry.from_model(model, optimizer)
    source_ref = "source/s203-registry.json"
    write_canonical_json(tmp_path / source_ref, {"registry": registry.to_manifest()})
    source_sha256 = hashlib.sha256((tmp_path / source_ref).read_bytes()).hexdigest()
    store = TaskArtifactStore(tmp_path, "runs/registry-binding")
    published = store.publish(
        task_id="stage2.04_reference_target",
        artifact_kind="parameter_registry",
        config_hash="a" * 64,
        run_intent="formal",
        formal_eligible=True,
        payload={
            "schema_version": "stage2-parameter-registry-artifact-v1",
            "registry_hash": registry.coordinate_registry_hash,
            "source_s203_manifest_ref": source_ref,
            "source_s203_manifest_sha256": source_sha256,
        },
    )
    loaded = _load_formal_parameter_registry(tmp_path, published.commit_ref, model)
    assert loaded.coordinate_registry_hash == registry.coordinate_registry_hash
    write_canonical_json(tmp_path / source_ref, {"registry": registry.to_manifest(), "tampered": True})
    with pytest.raises(ValueError, match="SOURCE_HASH_INVALID"):
        _load_formal_parameter_registry(tmp_path, published.commit_ref, model)


def _formal_stage2_01_request(
    input_result_refs: tuple[str, ...],
) -> TaskExecutionRequest:
    task_id = "stage2.01_scope_hypotheses_and_preregistration"
    value = _base_for(task_id)
    identity = value["identity"]
    assert isinstance(identity, dict)
    identity.update(
        {
            "run_intent": "formal",
            "formal_eligible": True,
            "route": "pretrain",
        }
    )
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    runtime.update({"allow_dirty_worktree": False, "device": "cuda"})
    config = load_resolved_config_compatible(
        value,
        task_id=task_id,
        overrides={
            "artifacts": {"output_dir": "runs/formal-s21"},
            "orchestration": {"input_result_refs": list(input_result_refs)},
        },
    )
    return TaskExecutionRequest(
        config=config,
        task=DEFAULT_TASK_CATALOG.get(task_id),
        environment=TaskRuntimeEnvironment(),
    )


def _publish_formal_stage1_predecessor(
    root: Path,
) -> tuple[TaskArtifactStore, tuple[str, ...], str]:
    store = TaskArtifactStore(root, "runs/stage1-11")
    output_root = "runs/stage1-11"
    artifact_kinds = DEFAULT_TASK_CATALOG.get(
        "stage1.11_reporting_and_exit_gate"
    ).artifact_kinds
    commit_refs = {
        kind: f"{output_root}/commits/{kind}.json" for kind in artifact_kinds
    }
    config_body: dict[str, object] = {
        "schema_version": "stage1-s1-11-producer-config-v2",
        "config_kind": "s1_11_reporting_producer",
        "task_id": "stage1.11_reporting_and_exit_gate",
        "gate_id": "G1-EXIT",
        "run_intent": "formal",
        "formal_eligible": True,
        "artifact_kinds": list(artifact_kinds),
        "commit_refs": commit_refs,
    }
    config_hash = canonical_json_hash(config_body)
    config_ref = f"{output_root}/producer-config.json"
    source_refs = (config_ref, *commit_refs.values(), f"{output_root}/manifest.json")
    write_canonical_json(
        root / Path(config_ref), {**config_body, "config_hash": config_hash}
    )
    refs: list[str] = []
    envelope_hashes: dict[str, str] = {}
    report_hash = ""
    for kind in artifact_kinds:
        payload: dict[str, object] = {"schema_version": f"stage1-{kind}-v1"}
        if kind == "stage_report":
            # Deliberately differ from the TaskArtifact envelope hash.  The
            # runner must bind the latter, not this payload compatibility field.
            payload["artifact_hash"] = "p" * 64
        published = store.publish(
            task_id="stage1.11_reporting_and_exit_gate",
            artifact_kind=kind,
            config_hash=config_hash,
            run_intent="formal",
            formal_eligible=True,
            source_refs=source_refs,
            payload=payload,
        )
        refs.append(published.commit_ref)
        envelope_hashes[kind] = published.artifact_hash
        if kind == "stage_report":
            report_hash = published.artifact_hash
    manifest_body: dict[str, object] = {
        "schema_version": "stage1-s1-11-task-artifact-manifest-v2",
        "status": "PASS",
        "task_id": "stage1.11_reporting_and_exit_gate",
        "gate_id": "G1-EXIT",
        "run_intent": "formal",
        "formal_eligible": True,
        "commit_refs": commit_refs,
        "commit_artifact_hashes": envelope_hashes,
        "config_ref": config_ref,
        "config_hash": config_hash,
    }
    write_canonical_json(
        root / Path(f"{output_root}/manifest.json"),
        {**manifest_body, "artifact_hash": canonical_json_hash(manifest_body)},
    )
    assert report_hash
    return store, tuple(refs), report_hash


STAGE23_TASK_IDS = tuple(
    task.task_id
    for task in DEFAULT_TASK_CATALOG.tasks
    if task.task_id.startswith(("stage2.", "stage3."))
)


def _formal_asset_manifest() -> AssetResolutionManifest:
    records = []
    for index, ((model, stage), (step, revision)) in enumerate(FORMAL_CHECKPOINT_SELECTION.items()):
        digest = f"{index + 1:064x}"
        records.append(
            CheckpointRecord(
                model_id=model,
                training_stage=stage,
                checkpoint_id=f"formal-{model}-{stage}",
                training_step=step,
                total_training_steps=FORMAL_TOTAL_TRAINING_STEPS,
                target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
                repository=f"EleutherAI/{model}",
                revision=revision,
                root_ref=f"models/{model}/{stage}",
                state="ready",
                files=(
                    CheckpointFile("model.safetensors", 1, digest, "weights"),
                    CheckpointFile("config.json", 1, digest, "config"),
                    CheckpointFile("tokenizer.json", 1, digest, "tokenizer"),
                ),
                manifest_ref=f"manifests/{model}-{stage}.json",
                manifest_sha256=digest,
                parameter_registry_hash=digest,
                config_sha256=digest,
                tokenizer_sha256=digest,
                load_status="passed",
                load_evidence_ref=f"evidence/{model}-{stage}.json",
                load_evidence_sha256=digest,
            )
        )
    return AssetResolutionManifest(
        scope="formal",
        checkpoints=tuple(records),
        data_range=DataRangeManifest(
            dataset_id=FORMAL_DATASET_ID,
            revision=FORMAL_DATASET_REVISION,
            manifest_ref="manifests/stage2/pile-prefix.json",
            manifest_sha256=FORMAL_DATA_MANIFEST_SHA256,
            files=tuple(
                DataFile(path, size, sha, "token_shard" if path.endswith(".bin") else "index")
                for path, (size, sha) in FORMAL_DATA_FILES.items()
            ),
        ),
        producer_commit="1" * 40,
        execution_commit="2" * 40,
    )


def _run_chain(
    root: Path,
    *,
    stop_after: str | None = None,
) -> tuple[TaskRuntime, dict[str, object], dict[str, object]]:
    runtime = _runtime(root)
    refs_by_task: dict[str, tuple[str, ...]] = {}
    configs: dict[str, object] = {}
    results: dict[str, object] = {}
    for task_id in STAGE23_TASK_IDS:
        # The fixture chain intentionally exercises S2.1 in isolation.  Its
        # formal path consumes the four Stage1.11 commits; the fixture path is
        # the explicit exception implemented by the runner preflight.
        predecessors = _REQUIRED_PREDECESSORS[task_id]
        if task_id == "stage2.01_scope_hypotheses_and_preregistration":
            predecessors = ()
        config = _config(
            task_id,
            f"runs/{task_id.replace('.', '-')}",
            tuple(
                ref
                for predecessor in predecessors
                for ref in refs_by_task[predecessor]
            ),
        )
        result = runtime.execute(config)
        assert result.status is TaskRunStatus.PASS, (task_id, result.to_dict())
        assert result.metadata["execution_contract"] == "stage23-specialized-v1"
        configs[task_id] = config
        results[task_id] = result
        refs_by_task[task_id] = tuple(result.artifact_refs.values())
        if task_id == stop_after:
            break
    return runtime, configs, results


@pytest.fixture(scope="module")
def completed_chain(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("stage23-complete-chain")
    runtime, configs, results = _run_chain(root)
    return root, runtime, configs, results


def test_stage2_reference_runner_executes_and_task_resume_is_hash_identical(
    completed_chain,
) -> None:
    root, runtime, configs, results = completed_chain
    config = configs["stage2.04_reference_target"]
    first = results["stage2.04_reference_target"]
    second = runtime.execute(config)

    assert first.status is TaskRunStatus.PASS
    assert first.formal_eligible is False
    assert tuple(first.artifact_refs) == DEFAULT_TASK_CATALOG.get(config.task_id).artifact_kinds
    assert second.result_hash == first.result_hash
    reference = _payload(root, first.artifact_refs["reference_result"])
    convergence = _payload(
        root, first.artifact_refs["reference_convergence_report"]
    )
    gate = _payload(root, first.artifact_refs["gate_record"])
    assert reference["schema_version"] == "reference-result-v1"
    assert reference["scope"] == "local_fixture"
    assert reference["formal_eligible"] is False
    assert convergence["converged"] is True
    assert convergence["recovery_semantics"] == "authoritative_block_pair_commits"
    assert gate["gate_status"] == "NOT_RUN"
    assert gate["local_validation_status"] == "PASS"


def test_stage2_reference_bounded_task_runtime_completes_with_reloadable_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_plan = stage23_task_runners._stage2_reference_plan

    def bounded_fixture_plan(request, root, context, authoritative=None):
        plan, refs = original_plan(request, root, context, authoritative)
        if request.config.run_intent != "local_fixture":
            return plan, refs
        return (
            ReferenceSizingPlan(
                reference_id=plan.reference_id,
                candidate_sample_counts=plan.candidate_sample_counts,
                block_size=plan.block_size,
                convergence_tolerance=plan.convergence_tolerance,
                required_consecutive=plan.required_consecutive,
                execution=plan.execution,
                require_terminal_convergence=True,
            ),
            refs,
        )

    monkeypatch.setattr(
        stage23_task_runners, "_stage2_reference_plan", bounded_fixture_plan
    )
    _runtime_value, _configs, results = _run_chain(
        tmp_path, stop_after="stage2.04_reference_target"
    )
    result = results["stage2.04_reference_target"]
    assert result.status is TaskRunStatus.PASS
    convergence = _payload(
        tmp_path, result.artifact_refs["reference_convergence_report"]
    )
    resume = tmp_path / "runs" / "stage2-04_reference_target" / "resume"
    assert (resume / "reference-sizing" / "bounded-checkpoint").is_dir()
    assert (resume / "reference-final" / "bounded-checkpoint").is_dir()
    assert not tuple(resume.rglob("commits/*.json"))
    assert (
        convergence["numerical_diagnostics"]["schema_version"]
        == "stage2-reference-numerical-diagnostics-v2"
    )
    assert (
        convergence["resume_replay"]["schema_version"]
        == "stage2-reference-resume-replay-v2"
    )
    assert (
        convergence["numerical_diagnostics"]["storage_mode"]
        == "bounded-online-fp64-v1"
    )


def test_stage2_partial_shard_requires_explicit_resume_ref(tmp_path: Path) -> None:
    """普通 run 不得静默消费 reference block-pair 的权威 commit。"""

    runtime, configs, results = _run_chain(
        tmp_path,
        stop_after="stage2.04_reference_target",
    )
    task_id = "stage2.04_reference_target"
    config = configs[task_id]
    first = results[task_id]
    artifacts = config.section("artifacts")
    assert isinstance(artifacts, dict)
    output_dir = str(artifacts["output_dir"])
    task_commit_root = tmp_path / output_dir / "commits"

    # 模拟核心 block-pair 已提交，但 task artifact 尚未形成完整权威集合时退出。
    for path in task_commit_root.glob("*.json"):
        path.unlink()
    assert tuple((tmp_path / output_dir / "resume").rglob("commits/*.json"))

    implicit = runtime.execute(config)
    assert implicit.status is TaskRunStatus.FAIL
    assert "STAGE23_RESUME_REF_REQUIRED" in implicit.message

    orchestration = config.section("orchestration")
    assert isinstance(orchestration, dict)
    resumed_config = _config(
        task_id,
        output_dir,
        tuple(str(item) for item in orchestration["input_result_refs"]),
        resume_ref=f"{output_dir}/resume",
    )
    assert resumed_config.config_hash == config.config_hash
    assert resumed_config.full_hash != config.full_hash
    resumed = runtime.execute(resumed_config)
    assert resumed.status is TaskRunStatus.PASS
    assert resumed.to_dict() == first.to_dict()


def test_stage2_restart_idempotent_task_rebuilds_partial_publish_without_resume(
    tmp_path: Path,
) -> None:
    """restart_idempotent 与 resume_shards 的入口语义必须保持区分。"""

    runtime, configs, results = _run_chain(
        tmp_path,
        stop_after="stage2.01_scope_hypotheses_and_preregistration",
    )
    task_id = "stage2.01_scope_hypotheses_and_preregistration"
    config = configs[task_id]
    first = results[task_id]
    artifacts = config.section("artifacts")
    assert isinstance(artifacts, dict)
    commit_paths = sorted(
        (tmp_path / str(artifacts["output_dir"]) / "commits").glob("*.json")
    )
    assert len(commit_paths) > 1
    for path in commit_paths[1:]:
        path.unlink()

    rebuilt = runtime.execute(config)
    assert rebuilt.status is TaskRunStatus.PASS
    assert rebuilt.to_dict() == first.to_dict()


@pytest.mark.parametrize(
    ("task_id", "primary_kind"),
    [
        ("stage2.05_paired_estimator_runner", "paired_runner_report"),
        ("stage2.07_main_sweep", "confirmatory_results"),
    ],
)
def test_stage2_estimator_experiment_runners_execute_real_paired_wave(
    completed_chain,
    task_id: str,
    primary_kind: str,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    result = results[task_id]

    assert result.status is TaskRunStatus.PASS
    report = _payload(root, result.artifact_refs[primary_kind])
    assert report["complete"] is True
    assert report["status"] == "FIXTURE_COMPLETE"
    assert set(report["method_statistics"]) >= {"raw", "double", "u_m2", "u_m4"}
    assert set(report["cost_statistics"]) == {
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    }
    assert report["cost_statistics"]["scientific_equal_sample_cost"][  # type: ignore[index]
        "wall_seconds"
    ] is None


def test_stage2_pilot_runner_computes_observations_and_fixture_recommendation(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    result = results["stage2.06_pilot_and_matrix_freeze"]

    assert result.status is TaskRunStatus.PASS
    report = _payload(root, result.artifact_refs["pilot_report"])
    matrix = _payload(root, result.artifact_refs["frozen_experiment_matrix"])
    recommendation = report["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["status"] == "FIXTURE_RECOMMENDATION"
    assert recommendation["formal_eligible"] is False
    assert recommendation["selected_estimator"] in {"u", "double"}
    assert len(report["observations"]) == 2  # type: ignore[arg-type]
    assert matrix["scope"] == "local_fixture"
    assert matrix["formal_freeze_status"] == "UNFROZEN"


def test_stage3_endpoint_runner_publishes_distinct_post_and_commit_boundaries(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    result = results["stage3.03_endpoint_and_probe_pipeline"]

    assert result.status is TaskRunStatus.PASS
    path_spec = _payload(root, result.artifact_refs["path_spec"])
    probe = _payload(root, result.artifact_refs["probe_manifest"])
    restoration = _payload(
        root, result.artifact_refs["state_restoration_report"]
    )
    endpoint = restoration["endpoint"]
    assert isinstance(endpoint, dict)
    record = endpoint["record"]
    assert isinstance(record, dict)
    assert path_spec["schema_version"] == "path-spec-v1"
    assert probe["schema_version"] == "stage3-probe-panel-v1"
    assert restoration["replay_verified"] is True
    assert record["parameter_post_state"]["artifact_id"] != record[  # type: ignore[index]
        "attempt_commit_state"
    ]["artifact_id"]


def test_stage3_reference_runner_executes_two_family_refinement_and_resumes(
    completed_chain,
) -> None:
    root, runtime, configs, results = completed_chain
    config = configs["stage3.05_reference_integral_and_precision"]
    first = results["stage3.05_reference_integral_and_precision"]
    second = runtime.execute(config)

    assert first.status is TaskRunStatus.PASS
    assert second.result_hash == first.result_hash
    reference = _payload(root, first.artifact_refs["path_integral_reference"])
    precision = _payload(root, first.artifact_refs["precision_budget"])
    refinement = reference["refinement"]
    cache = reference["node_gradient_cache"]
    assert isinstance(refinement, dict)
    assert isinstance(cache, dict)
    assert refinement["status"] == "FIXTURE_CONVERGED"
    assert refinement["converged"] is True
    assert {row["family"] for row in refinement["completed_levels"]} == {  # type: ignore[index]
        "gauss_legendre",
        "composite_simpson",
    }
    assert precision["two_independent_rule_families"] is True
    assert precision["formal_threshold_status"] == "UNFROZEN"
    assert cache["cross_rule_reused_key_count"] > 0
    commit_evidence = cache["commit_evidence"]
    assert isinstance(commit_evidence, dict)
    assert commit_evidence["all_requested_keys_committed"] is True
    assert commit_evidence["authoritative_commits"]
    assert commit_evidence["reconciliation"]["orphan_objects"] == []  # type: ignore[index]


def test_stage3_pilot_runner_builds_quadrature_recommendation_without_gate_pass(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    result = results["stage3.06_pilot_and_threshold_freeze"]

    assert result.status is TaskRunStatus.PASS
    pilot = _payload(root, result.artifact_refs["quadrature_pilot_report"])
    freeze = _payload(root, result.artifact_refs["threshold_freeze"])
    gate = _payload(root, result.artifact_refs["gate_record"])
    recommendation = pilot["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["status"] == "FIXTURE_RECOMMENDATION"
    assert recommendation["default_rule"] == "midpoint"
    assert recommendation["formal_eligible"] is False
    assert pilot["cost_semantics"] == "deterministic_unique_node_units_not_wall_clock"
    cache = pilot["node_gradient_cache"]
    assert isinstance(cache, dict)
    assert cache["cross_rule_reused_key_count"] == 3
    assert cache["commit_evidence"]["all_requested_keys_committed"] is True  # type: ignore[index]
    assert freeze["formal_freeze_status"] == "UNFROZEN"
    assert gate["gate_status"] == "NOT_RUN"


def test_stage3_matrix_runner_consumes_local_recommendation_and_integrates(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    result = results["stage3.07_formal_experiment_matrix"]

    assert result.status is TaskRunStatus.PASS
    paths = _payload(root, result.artifact_refs["formal_path_results"])
    completeness = _payload(root, result.artifact_refs["completeness_report"])
    assert paths["scope"] == "local_fixture"
    assert paths["formal_eligible"] is False
    assert paths["selected_rule"] == "midpoint"
    cache = paths["node_gradient_cache"]
    assert isinstance(cache, dict)
    assert cache["commit_evidence"]["all_requested_keys_committed"] is True  # type: ignore[index]
    assert completeness["node_gradient_cache_evidence_hash"] == cache[
        "evidence_hash"
    ]
    assert completeness["defined"] is True
    assert completeness["absolute_residual"] == pytest.approx(0.0, abs=1e-12)


def test_all_stage23_task_ids_are_specialized_and_hash_bound_to_full_predecessor(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    assert tuple(results) == STAGE23_TASK_IDS
    for task_id in STAGE23_TASK_IDS:
        result = results[task_id]
        assert tuple(result.artifact_refs) == DEFAULT_TASK_CATALOG.get(
            task_id
        ).artifact_kinds
        assert result.metadata == {"execution_contract": "stage23-specialized-v1"}
        predecessors = _REQUIRED_PREDECESSORS[task_id]
        if task_id == "stage2.01_scope_hypotheses_and_preregistration":
            predecessors = ()
        expected_sources = [
            ref
            for predecessor in predecessors
            for ref in results[predecessor].artifact_refs.values()
        ]
        for commit_ref in result.artifact_refs.values():
            commit = load_canonical_json(root / Path(commit_ref))
            assert isinstance(commit, dict)
            body = load_canonical_json(root / Path(str(commit["object_ref"])))
            assert isinstance(body, dict)
            assert body["source_refs"] == expected_sources


def test_stage2_direct_predecessors_match_plan_dag() -> None:
    assert _REQUIRED_PREDECESSORS["stage2.01_scope_hypotheses_and_preregistration"] == (
        "stage1.11_reporting_and_exit_gate",
    )
    assert _REQUIRED_PREDECESSORS["stage2.02_stage1_handoff_and_fixed_state_contract"] == (
        "stage2.01_scope_hypotheses_and_preregistration",
    )
    assert _REQUIRED_PREDECESSORS["stage2.03_assets_checkpoints_and_sampling"] == (
        "stage2.01_scope_hypotheses_and_preregistration",
    )
    assert set(_REQUIRED_PREDECESSORS["stage2.04_reference_target"]) == {
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    }
    assert set(_REQUIRED_PREDECESSORS["stage2.05_paired_estimator_runner"]) == {
        "stage2.02_stage1_handoff_and_fixed_state_contract",
        "stage2.03_assets_checkpoints_and_sampling",
    }
    assert _REQUIRED_PREDECESSORS["stage2.06_pilot_and_matrix_freeze"] == (
        "stage2.04_reference_target",
        "stage2.05_paired_estimator_runner",
    )
    assert _REQUIRED_PREDECESSORS["stage2.08_statistics_and_robustness"] == (
        "stage2.07_main_sweep",
    )
    assert _REQUIRED_PREDECESSORS["stage2.09_cost_and_system_validation"] == (
        "stage2.07_main_sweep",
    )
    assert _REQUIRED_PREDECESSORS["stage2.10_visualization_reporting_and_decision"] == (
        "stage2.08_statistics_and_robustness",
        "stage2.09_cost_and_system_validation",
    )


def test_stage2_and_stage3_derived_outputs_use_frozen_sources_without_gate_pass(
    completed_chain,
) -> None:
    root, _runtime_value, _configs, results = completed_chain
    stage2_statistics = results["stage2.08_statistics_and_robustness"]
    stage2_capacity = results["stage2.09_cost_and_system_validation"]
    stage2_reporting = results["stage2.10_visualization_reporting_and_decision"]
    stage2_delivery = results["stage2.11_delivery_and_exit_gate"]
    frozen2 = _payload(root, stage2_statistics.artifact_refs["frozen_source_table"])
    cost2 = _payload(root, stage2_capacity.artifact_refs["cost_table"])
    decision2 = _payload(root, stage2_reporting.artifact_refs["estimator_decision"])
    report2 = _payload(root, stage2_reporting.artifact_refs["analysis_report"])
    gates2 = _payload(root, stage2_delivery.artifact_refs["gate_summary"])
    sync2 = _payload(root, stage2_delivery.artifact_refs["sync_report"])
    assert frozen2["frozen"] is True
    assert report2["source_artifacts"][0]["content_hash"] == cost2["content_hash"]  # type: ignore[index]
    assert decision2["scope"] == "local_fixture"
    assert decision2["metadata"]["formal_eligible"] is False  # type: ignore[index]
    assert decision2["gate_status"] == "NOT_RUN"
    assert gates2["formal_exit_gate"] == "NOT_RUN"
    assert sync2["server"] == "BLOCKED"

    validation3 = results["stage3.04_quadrature_engine_and_unit_tests"]
    statistics3 = results["stage3.08_error_analysis_and_stability"]
    analysis3 = results["stage3.09_cost_and_method_selection"]
    reporting3 = results["stage3.10_reports_visualizations_and_handoff"]
    analytic = _payload(
        root, validation3.artifact_refs["analytic_validation_report"]
    )
    frozen3 = _payload(root, statistics3.artifact_refs["frozen_source_table"])
    decision3 = _payload(root, analysis3.artifact_refs["quadrature_decision"])
    report3 = _payload(root, reporting3.artifact_refs["analysis_report"])
    gates3 = _payload(root, reporting3.artifact_refs["gate_summary"])
    assert analytic["passed"] is True
    assert frozen3["frozen"] is True
    assert decision3["schema_version"] == "quadrature-decision-v1"
    assert decision3["formal_eligible"] is False
    assert report3["metadata"]["formal_eligible"] is False  # type: ignore[index]
    assert gates3["stage3.G3-7"] == "NOT_RUN"


def test_missing_complete_predecessor_is_structurally_blocked(tmp_path: Path) -> None:
    task_id = "stage2.02_stage1_handoff_and_fixed_state_contract"
    result = _runtime(tmp_path).execute(
        _config(task_id, "runs/missing-predecessor")
    )
    assert result.status is TaskRunStatus.BLOCKED
    assert [item.code for item in result.blockers] == [BlockerCode.ASSET_UNAVAILABLE]
    assert result.blockers[0].requirement == (
        "complete_predecessor:stage2.01_scope_hypotheses_and_preregistration"
    )


def test_override_factory_preserves_non_stage23_fallback_dispatch(tmp_path: Path) -> None:
    sentinel = object()

    class Fallback:
        runner_kind = RunnerKind.CONTRACT

        def run(self, request):
            assert request.task.task_id == "stage4.route"
            return sentinel

    runners = build_stage23_runner_overrides(tmp_path, fallbacks=(Fallback(),))
    specialized_kinds = {
        task.runner_kind
        for task in DEFAULT_TASK_CATALOG.tasks
        if task.task_id in STAGE23_TASK_IDS
    }
    assert specialized_kinds.issubset(set(runners))
    contract = runners[RunnerKind.CONTRACT]

    class Request:
        task = DEFAULT_TASK_CATALOG.get("stage4.route")

    assert contract.run(Request()) is sentinel


@pytest.mark.parametrize(
    "task_id",
    [
        "stage2.04_reference_target",
        "stage3.06_pilot_and_threshold_freeze",
    ],
)
def test_independent_fixture_workspaces_publish_identical_artifact_hashes(
    tmp_path: Path,
    task_id: str,
) -> None:
    observed: list[tuple[str, ...]] = []
    for name in ("first", "second"):
        workspace = tmp_path / name
        workspace.mkdir()
        _runtime_value, configs, results = _run_chain(workspace, stop_after=task_id)
        config = configs[task_id]
        result = results[task_id]
        store = TaskArtifactStore(
            workspace, f"runs/{task_id.replace('.', '-')}"
        )
        observed.append(
            tuple(
                store.load_commit(result.artifact_refs[kind]).artifact_hash
                for kind in DEFAULT_TASK_CATALOG.get(task_id).artifact_kinds
            )
        )
    assert observed[0] == observed[1]


def test_formal_runner_missing_execution_evidence_is_blocked_before_synthetic_fallback(
    tmp_path: Path,
) -> None:
    task_id = "stage2.04_reference_target"
    value = _base_for(task_id)
    identity = value["identity"]
    assert isinstance(identity, dict)
    identity.update(
        {
            "run_intent": "formal",
            "formal_eligible": True,
            "route": "pretrain",
        }
    )
    runtime_section = value["runtime"]
    assert isinstance(runtime_section, dict)
    runtime_section.update({"device": "cuda", "allow_dirty_worktree": False})
    provider = {
        "kind": "offline_hf",
        "model_manifest_ref": None,
        "model_root_ref": None,
        "data_manifest_ref": None,
        "data_root_ref": None,
        "tokenizer_manifest_ref": None,
        "tokenizer_root_ref": None,
    }
    config = load_resolved_config_compatible(
        value,
        task_id=task_id,
        overrides={
            "providers": provider,
            "artifacts": {"output_dir": "runs/formal-stage2-reference"},
        },
    )
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"server", "cuda", "model_assets", "data_assets"}),
        frozen_contract_stages=frozenset({0, 1, 2}),
        passed_gate_ids=frozenset({"stage2.G2.2"}),
    )

    result = _runtime(tmp_path).execute(config, environment=environment)

    assert result.status is TaskRunStatus.BLOCKED
    assert result.formal_eligible is False
    # 字符串集合不能再绕过统一 preflight。这里故意没有提供任何权威 commit，
    # 因而会在进入 Stage 2 runner 前同时报告合同、Gate、capability 和上游输入缺失。
    codes = {item.code for item in result.blockers}
    assert BlockerCode.CONTRACT_UNFROZEN in codes
    assert BlockerCode.GATE_NOT_READY in codes
    assert BlockerCode.ASSET_UNAVAILABLE in codes
    assert BlockerCode.DEVICE_UNAVAILABLE in codes
    store = TaskArtifactStore(tmp_path, "runs/formal-stage2-reference")
    assert store.discover_complete(
        task_id=task_id,
        config_hash=config.config_hash,
        artifact_kinds=DEFAULT_TASK_CATALOG.get(task_id).artifact_kinds,
        formal_eligible=True,
    ) is None


def test_formal_stage1_predecessor_binds_verified_envelope_hash(
    tmp_path: Path,
) -> None:
    store, refs, envelope_hash = _publish_formal_stage1_predecessor(tmp_path)
    request = _formal_stage2_01_request(refs)

    inputs = _predecessor_context(request, tmp_path, store)
    handoff = _stage1_formal_bridge_identity(tmp_path, inputs)
    assert handoff is not None
    manifest = load_canonical_json(tmp_path / "runs/stage1-11/manifest.json")
    assert handoff == manifest

    outputs, _ = _run_stage2_contract(request, tmp_path, store)
    provenance = outputs["preregistration"]["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["stage1_handoff"] == manifest
    assert provenance["stage1_report_hash"] == envelope_hash

    assert _formal_stage1_report_artifact_hash(inputs) == envelope_hash
    assert _formal_stage1_report_artifact_hash(inputs) != "p" * 64


def test_formal_stage1_predecessor_bad_manifest_fails_closed(tmp_path: Path) -> None:
    store, refs, _envelope_hash = _publish_formal_stage1_predecessor(tmp_path)
    manifest_path = tmp_path / "runs/stage1-11/manifest.json"
    manifest = load_canonical_json(manifest_path)
    assert isinstance(manifest, dict)
    hashes = manifest["commit_artifact_hashes"]
    assert isinstance(hashes, dict)
    hashes["stage_report"] = "d" * 64
    body = {key: item for key, item in manifest.items() if key != "artifact_hash"}
    manifest["artifact_hash"] = canonical_json_hash(body)
    write_canonical_json(manifest_path, manifest)

    request = _formal_stage2_01_request(refs)
    inputs = _predecessor_context(request, tmp_path, store)
    with pytest.raises(ValueError, match="MANIFEST_ENVELOPE_HASHES_INVALID"):
        _stage1_formal_bridge_identity(tmp_path, inputs)


def test_formal_stage1_predecessor_wrong_hash_fails_closed(tmp_path: Path) -> None:
    store, refs, _envelope_hash = _publish_formal_stage1_predecessor(tmp_path)
    report_commit = tmp_path / Path(refs[0])
    commit = load_canonical_json(report_commit)
    assert isinstance(commit, dict)
    commit["artifact_hash"] = "d" * 64
    write_canonical_json(report_commit, commit)

    request = _formal_stage2_01_request(refs)
    with pytest.raises(TaskBlockedError, match="前序引用不可验证"):
        _predecessor_context(request, tmp_path, store)


def _formal_assets_request(tmp_path: Path, *, evidence_ref: str | None) -> TaskExecutionRequest:
    task_id = "stage2.03_assets_checkpoints_and_sampling"
    value = _base_for(task_id)
    identity = value["identity"]
    assert isinstance(identity, dict)
    identity.update({"run_intent": "formal", "formal_eligible": True, "route": "pretrain"})
    runtime = value["runtime"]
    assert isinstance(runtime, dict)
    runtime.update({"allow_dirty_worktree": False, "device": "cuda"})
    evidence_refs = {} if evidence_ref is None else {"stage2_asset_resolution": evidence_ref}
    config = load_resolved_config_compatible(value, task_id=task_id)
    return TaskExecutionRequest(
        config=config,
        task=DEFAULT_TASK_CATALOG.get(task_id),
        environment=TaskRuntimeEnvironment(evidence_refs=evidence_refs),
    )


def test_formal_stage2_asset_runner_consumes_real_matrix_without_fixture_fallback(
    tmp_path: Path,
) -> None:
    reference = "evidence/stage2-asset-resolution.json"
    path = tmp_path / reference
    path.parent.mkdir(parents=True)
    write_canonical_json(path, _formal_asset_manifest().to_dict())
    request = _formal_assets_request(tmp_path, evidence_ref=reference)
    manifest, loaded_ref = _formal_stage2_asset_manifest(request, tmp_path)
    assert loaded_ref == reference
    assert manifest.scope == "formal"
    assert all(record.revision != "0" for record in manifest.checkpoints)

    class _Preregistration:
        binding_hash = "a" * 64
        references: tuple[str, ...] = ()

        @staticmethod
        def payload(_kind: str) -> dict[str, str]:
            return {"schema_version": "stage2-preregistration-v1", "scope": "formal"}

    outputs, _ = _run_formal_stage2_assets_and_sampling(
        request, tmp_path, _Preregistration()
    )
    resolution = outputs["asset_resolution"]
    assert resolution["stage2_asset_manifest"]["scope"] == "formal"  # type: ignore[index]
    assert resolution["provider"]["provider_kind"] == "offline_hf_stage2_asset_manifest"  # type: ignore[index]
    assert "fixed_state_contract_hash" not in resolution


def test_formal_stage2_asset_runner_fails_closed_without_environment_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(TaskBlockedError, match="environment-bound asset resolution"):
        _formal_stage2_asset_manifest(_formal_assets_request(tmp_path, evidence_ref=None), tmp_path)
