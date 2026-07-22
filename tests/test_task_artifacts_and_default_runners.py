"""统一 task artifact 与默认 runner 工厂的端到端本机测试。"""

from __future__ import annotations

from copy import deepcopy

import pytest

from param_importance_nlp.contracts import (
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import RunnerKind
from param_importance_nlp.experiments import build_default_task_runtime
from param_importance_nlp.runtime import (
    CheckpointStore,
    TaskArtifactStore,
    load_committed_task_artifact,
    load_tensor_bundle,
)


def _base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = deepcopy(
        load_canonical_json("configs/local-fixtures/resolved-config-v1.json")
    )
    value["identity"]["stage"] = stage
    value["identity"]["task"] = task_id
    value["loss"]["task_type"] = "sequence_classification"
    value["loss"]["weighting"] = "sample"
    value["data"]["statistical_unit"] = "sample"
    value["data"]["weight_unit"] = "sample"
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def test_default_runtime_registers_every_runner_kind(tmp_path) -> None:
    runtime = build_default_task_runtime(tmp_path)
    assert set(runtime.registered_kinds) == set(RunnerKind)


def test_training_task_is_idempotent_and_publishes_safe_state(tmp_path) -> None:
    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 2},
            "providers": {"num_labels": 3},
            "artifacts": {"output_dir": "runs/training-smoke"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )
    runtime = build_default_task_runtime(tmp_path)

    first = runtime.execute(config)
    second = runtime.execute(config)

    assert first.status.value == "PASS"
    assert first.to_dict() == second.to_dict()
    assert tuple(first.artifact_refs) == config.task_definition.artifact_kinds
    store = TaskArtifactStore(tmp_path, "runs/training-smoke")
    for reference in first.artifact_refs.values():
        published = store.load_commit(reference)
        assert published.config_hash == config.config_hash
        assert published.formal_eligible is False
    bundle_state, identity = load_tensor_bundle(
        tmp_path / "runs/training-smoke/tensor-bundles/importance-final"
    )
    assert bundle_state["metadata"]["scope"] == "local_fixture"
    assert bundle_state["accumulator"]["successful_steps"] == 2
    assert identity.tensor_count > 0


def test_two_clean_roots_publish_identical_training_semantic_hashes(tmp_path) -> None:
    """墙钟时间、event UUID 与调用前全局 RNG 不得污染功能产物身份。"""

    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 1},
            "providers": {"num_labels": 3},
            "artifacts": {"output_dir": "runs/training-determinism"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )

    results = []
    payloads = []
    for root in (tmp_path / "first", tmp_path / "second"):
        result = build_default_task_runtime(root).execute(config)
        results.append(result)
        store = TaskArtifactStore(root, "runs/training-determinism")
        published = store.load_commit(
            result.artifact_refs["training_smoke_result"]
        )
        payloads.append(load_canonical_json(root / published.object_ref)["payload"])

    assert results[0].artifact_refs == results[1].artifact_refs
    assert payloads[0] == payloads[1]
    checkpoint = payloads[0]["checkpoint_commits"][0]
    assert set(checkpoint) == {
        "checkpoint_id",
        "commit_ref",
        "commit_identity_sha256",
        "bundle_manifest_sha256",
    }
    assert payloads[0]["event_stream_semantic_sha256"]
    assert set(payloads[0]["event_streams"][0]) == {
        "ref",
        "semantic_sha256",
    }


def test_training_task_consumes_endpoint_plan_and_publishes_three_boundaries(
    tmp_path,
) -> None:
    plan_body = {
        "schema_version": "training-endpoint-capture-plan-v1",
        "plan_id": "tiny-training-endpoints",
        "selected_steps": [1],
        "include_checkpoint_steps": True,
        "scope": "local_fixture",
        "formal_eligible": False,
        "qualification_evidence_hash": None,
        "probe_plan_ref": None,
    }
    plan_body["artifact_hash"] = canonical_json_hash(plan_body)
    write_canonical_json(tmp_path / "plans/endpoints.json", plan_body)
    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 2},
            "providers": {"num_labels": 3},
            "orchestration": {
                "input_result_refs": ["plans/endpoints.json"]
            },
            "artifacts": {"output_dir": "runs/training-endpoints"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )

    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS"
    store = TaskArtifactStore(tmp_path, "runs/training-endpoints")
    published = store.load_commit(next(iter(result.artifact_refs.values())))
    payload = load_canonical_json(tmp_path / published.object_ref)["payload"]
    assert [
        item["endpoint_id"] for item in payload["endpoint_bundles"]
    ] == [
        "stage0-06_single_gpu_smoke-rank-0000-step-00000001",
        "stage0-06_single_gpu_smoke-rank-0000-step-00000002",
    ]
    for bundle in payload["endpoint_bundles"]:
        commit = load_canonical_json(tmp_path / bundle["commit_ref"])
        endpoint = load_canonical_json(tmp_path / commit["object_ref"])
        assert set(endpoint["state_bundles"]) == {
            "pre",
            "parameter_post",
            "attempt_commit",
        }
        assert endpoint["record"]["replay_verified"] is True


def test_catalog_runner_executes_contract_task_without_temporary_script(tmp_path) -> None:
    task_id = "stage1.01_entry_and_contract"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=1, task_id=task_id),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 3},
            "artifacts": {"output_dir": "runs/stage1-contract"},
        },
    )

    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS"
    assert tuple(result.artifact_refs) == config.task_definition.artifact_kinds
    artifact = TaskArtifactStore(tmp_path, "runs/stage1-contract").load_commit(
        result.artifact_refs["stage_contract"]
    )
    value = load_canonical_json(tmp_path / artifact.object_ref)
    payload = value["payload"]
    assert payload["schema_version"] == "stage01-task-evidence-v1"
    assert payload["core_evidence"]["evidence_type"] == "stage1_math_contract"
    assert payload["core_evidence"]["contract_hashes"]
    assert payload["gate_status"] == "NOT_RUN"


def test_training_task_commits_replayable_evaluation_boundaries(tmp_path) -> None:
    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 2, "validation_every_steps": 1},
            "providers": {"num_labels": 3},
            "evaluation": {
                "enabled": True,
                "split": "validation",
                "every_steps": 1,
                "batch_size": 2,
                "max_batches": 1,
                "metrics": ["accuracy"],
            },
            "artifacts": {"output_dir": "runs/training-with-evaluation"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )

    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS"
    for step in (1, 2):
        store = TaskArtifactStore(
            tmp_path,
            f"runs/training-with-evaluation/evaluation/step-{step:08d}",
        )
        published = store.load_commit(
            f"runs/training-with-evaluation/evaluation/step-{step:08d}/commits/metrics.json"
        )
        value = load_canonical_json(tmp_path / published.object_ref)
        assert value["payload"]["global_step"] == step
        assert 0.0 <= value["payload"]["metrics"]["accuracy"] <= 1.0

    task_store = TaskArtifactStore(tmp_path, "runs/training-with-evaluation")
    task_artifact = task_store.load_commit(next(iter(result.artifact_refs.values())))
    payload = load_canonical_json(tmp_path / task_artifact.object_ref)["payload"]
    assert [item["global_step"] for item in payload["evaluation_records"]] == [1, 2]


def test_training_task_profiles_real_step_windows_with_two_phase_commits(
    tmp_path,
) -> None:
    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 2},
            "providers": {"num_labels": 3},
            "profiling": {
                "enabled": True,
                "warmup_steps": 0,
                "measure_steps": 1,
                "repetitions": 2,
                "capture_memory": False,
                "capture_throughput": True,
                "capture_communication": True,
                "synchronize_device": False,
            },
            "artifacts": {"output_dir": "runs/training-with-profiling"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )

    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS"
    task_store = TaskArtifactStore(tmp_path, "runs/training-with-profiling")
    task_artifact = task_store.load_commit(next(iter(result.artifact_refs.values())))
    payload = load_canonical_json(tmp_path / task_artifact.object_ref)["payload"]
    profiles = payload["resource_profiles"]
    assert [item["repetition"] for item in profiles] == [0, 1]
    assert [(item["start_step"], item["end_step"]) for item in profiles] == [
        (0, 1),
        (1, 2),
    ]
    assert all(item["profile"]["completed_steps"] == 1 for item in profiles)
    assert all(item["profile"]["effective_units"] > 0 for item in profiles)
    assert all(item["communication"]["defined"] is False for item in profiles)
    for repetition in range(2):
        commit = load_canonical_json(
            tmp_path
            / "runs/training-with-profiling/resource-profiles"
            / "rank-0000/commits"
            / f"window-{repetition:04d}.json"
        )
        assert commit["schema_version"] == "training-resource-window-commit-v1"
        assert load_canonical_json(tmp_path / commit["object_ref"])["artifact_hash"] == (
            commit["artifact_hash"]
        )


def test_training_task_applies_prefetch_workers_and_checkpoints_pending_batches(
    tmp_path,
) -> None:
    task_id = "stage0.06_single_gpu_smoke"
    config = ResolvedConfigV2.resolve(
        _base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "training": {"max_steps": 2},
            "providers": {"num_labels": 3},
            "data_loader": {
                "num_workers": 2,
                "prefetch_factor": 2,
                "persistent_workers": True,
                "drop_last": False,
                "cursor_policy": "attempt_commit_state",
            },
            "artifacts": {"output_dir": "runs/training-with-prefetch"},
            "checkpoint_schedule": {
                "segments": [
                    {"start_step": 0, "end_step": None, "every_steps": 1}
                ]
            },
        },
    )

    result = build_default_task_runtime(tmp_path).execute(config)

    assert result.status.value == "PASS"
    task_store = TaskArtifactStore(tmp_path, "runs/training-with-prefetch")
    artifact = task_store.load_commit(next(iter(result.artifact_refs.values())))
    payload = load_canonical_json(tmp_path / artifact.object_ref)["payload"]
    checkpoint_id = payload["training_result"]["state"]["last_checkpoint_id"]
    state, _commit = CheckpointStore(
        tmp_path / "runs/training-with-prefetch/checkpoints/rank-0000"
    ).load(checkpoint_id)
    assert state["cursor"]["schema_version"] == "prefetch-batch-cursor-state-v1"
    assert state["cursor"]["num_workers"] == 2


def test_task_artifact_publish_requires_versioned_canonical_payload(tmp_path) -> None:
    store = TaskArtifactStore(tmp_path, "runs/payload-contract")
    common = {
        "task_id": "stage0.01_baseline_and_safety",
        "artifact_kind": "contract_freeze",
        "config_hash": "a" * 64,
    }

    for run_intent, formal_eligible in (
        ("local_fixture", False),
        ("formal", True),
    ):
        for payload in (
            {},
            {"schema_version": ""},
            {"schema_version": "human version 1"},
            {"schema_version": 1},
        ):
            with pytest.raises(
                (TypeError, ValueError),
                match="TASK_ARTIFACT_PAYLOAD_SCHEMA_VERSION_INVALID",
            ):
                store.publish(
                    payload=payload,  # type: ignore[arg-type]
                    run_intent=run_intent,
                    formal_eligible=formal_eligible,
                    **common,
                )

    with pytest.raises(ValueError, match="TASK_ARTIFACT_PAYLOAD_NOT_CANONICAL_JSON"):
        store.publish(
            payload={"schema_version": "test-payload-v1", "bad": (1, 2)},  # type: ignore[dict-item]
            run_intent="local_fixture",
            formal_eligible=False,
            **common,
        )
    assert not tuple(store.commits.iterdir())


def test_task_artifact_loaders_reject_hash_valid_unversioned_payload(tmp_path) -> None:
    store = TaskArtifactStore(tmp_path, "runs/payload-load-contract")
    published = store.publish(
        task_id="stage0.01_baseline_and_safety",
        artifact_kind="contract_freeze",
        config_hash="b" * 64,
        run_intent="local_fixture",
        payload={"schema_version": "test-payload-v1", "verified": True},
        formal_eligible=False,
    )
    envelope = load_canonical_json(tmp_path / published.object_ref)
    assert isinstance(envelope, dict)
    payload = envelope["payload"]
    assert isinstance(payload, dict)
    payload.pop("schema_version")
    body = {key: value for key, value in envelope.items() if key != "artifact_hash"}
    replacement_hash = canonical_json_hash(body)
    envelope["artifact_hash"] = replacement_hash
    replacement = (
        tmp_path
        / "runs/payload-load-contract/objects/contract_freeze"
        / f"{replacement_hash}.json"
    )
    write_canonical_json(replacement, envelope)

    commit_path = tmp_path / published.commit_ref
    commit = load_canonical_json(commit_path)
    assert isinstance(commit, dict)
    commit["artifact_hash"] = replacement_hash
    commit["object_ref"] = replacement.relative_to(tmp_path).as_posix()
    write_canonical_json(commit_path, commit)

    with pytest.raises(ValueError, match="TASK_ARTIFACT_PAYLOAD_SCHEMA_VERSION_INVALID"):
        store.load_commit(published.commit_ref)
    with pytest.raises(ValueError, match="TASK_ARTIFACT_PAYLOAD_SCHEMA_VERSION_INVALID"):
        load_committed_task_artifact(tmp_path, published.commit_ref)
