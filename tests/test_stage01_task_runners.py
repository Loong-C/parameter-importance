"""Stage 0/1 非训练专用 runner 的任务分派、恢复与重放测试。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts import ResolvedConfig, load_canonical_json
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, RunnerKind
from param_importance_nlp.experiments.stage01_task_runners import (
    STAGE01_HANDLED_TASK_IDS,
    Stage01CompositeTaskRunner,
    build_stage01_runner_overrides,
)
from param_importance_nlp.runtime import TaskArtifactStore, TaskRuntime


ROOT = Path(__file__).resolve().parents[1]


def _base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = deepcopy(
        load_canonical_json(ROOT / "configs/local-fixtures/resolved-config-v1.json")
    )
    value["identity"]["stage"] = stage
    value["identity"]["task"] = task_id
    value["loss"]["task_type"] = "sequence_classification"
    value["loss"]["weighting"] = "sample"
    value["data"]["statistical_unit"] = "sample"
    value["data"]["weight_unit"] = "sample"
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def _formal_base_config(*, stage: int, task_id: str) -> ResolvedConfig:
    value = _base_config(stage=stage, task_id=task_id).to_dict()
    value["identity"]["run_intent"] = "formal"
    value["identity"]["formal_eligible"] = True
    value["runtime"]["allow_dirty_worktree"] = False
    return ResolvedConfig.from_mapping(value)


def _config(tmp_path: Path, task_id: str, output: str) -> ResolvedConfigV2:
    task = DEFAULT_TASK_CATALOG.get(task_id)
    return ResolvedConfigV2.resolve(
        _base_config(stage=task.stage, task_id=task_id),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 3},
            "artifacts": {"output_dir": output},
        },
    )


def _load_payload(tmp_path: Path, output: str, ref: str) -> dict[str, object]:
    published = TaskArtifactStore(tmp_path, output).load_commit(ref)
    artifact = load_canonical_json(tmp_path / published.object_ref)
    assert isinstance(artifact, dict)
    payload = artifact["payload"]
    assert isinstance(payload, dict)
    return payload


def test_dispatch_covers_every_stage0_1_non_training_task() -> None:
    expected = {
        task.task_id
        for task in DEFAULT_TASK_CATALOG.tasks
        if task.stage in {0, 1}
        and task.runner_kind not in {RunnerKind.TRAINING, RunnerKind.DISTRIBUTED_TRAINING}
    }
    assert STAGE01_HANDLED_TASK_IDS == expected

    overrides = build_stage01_runner_overrides(ROOT)
    dispatched = {
        task_id
        for runner in overrides.values()
        for task_id in runner.handled_task_ids
    }
    assert dispatched == expected
    assert all(type(runner).__name__ == "Stage01CompositeTaskRunner" for runner in overrides.values())


@pytest.mark.parametrize("task_id", sorted(STAGE01_HANDLED_TASK_IDS))
def test_every_specialized_task_has_an_executable_local_fixture_path(
    tmp_path: Path, task_id: str
) -> None:
    task = DEFAULT_TASK_CATALOG.get(task_id)
    runner = build_stage01_runner_overrides(tmp_path)[task.runner_kind]
    runtime = TaskRuntime()
    runtime.register(runner)

    result = runtime.execute(_config(tmp_path, task_id, f"runs/all/{task_id}"))

    assert result.status.value == "PASS", result.to_dict()
    assert result.metadata["stage01_specialized"] is True
    assert tuple(result.artifact_refs) == task.artifact_kinds


def test_contract_runner_publishes_role_specific_evidence_and_never_passes_gate(
    tmp_path: Path,
) -> None:
    task_id = "stage0.05_config_run_identity_and_seeds"
    output = "runs/stage01-contract"
    runtime = TaskRuntime()
    runtime.register(Stage01CompositeTaskRunner(RunnerKind.CONTRACT, tmp_path))
    result = runtime.execute(_config(tmp_path, task_id, output))

    assert result.status.value == "PASS"
    payloads = {
        kind: _load_payload(tmp_path, output, ref)
        for kind, ref in result.artifact_refs.items()
    }
    assert {payload["schema_version"] for payload in payloads.values()} == {
        "stage01-task-evidence-v1"
    }
    assert {payload["gate_status"] for payload in payloads.values()} == {"NOT_RUN"}
    assert payloads["resolved_config"]["role_evidence"] != payloads["seed_plan"]["role_evidence"]
    core = payloads["run_identity"]["core_evidence"]
    assert core["evidence_type"] == "config_identity_seed"
    assert core["resume_attempt"]["attempt_id"] == 2


def test_checkpoint_runner_executes_two_phase_resume_and_restores_from_task_commit(
    tmp_path: Path,
) -> None:
    task_id = "stage0.09_checkpoint_and_resume"
    output = "runs/stage01-checkpoint"
    runtime = TaskRuntime()
    runtime.register(Stage01CompositeTaskRunner(RunnerKind.CHECKPOINT, tmp_path))
    config = _config(tmp_path, task_id, output)

    first = runtime.execute(config)
    second = runtime.execute(config)

    assert first.status.value == "PASS"
    assert second.status.value == "PASS"
    assert second.metadata["restored"] is True
    payload = _load_payload(tmp_path, output, first.artifact_refs["resume_equivalence_report"])
    core = payload["core_evidence"]
    assert core["resume_equivalent"] is True
    assert core["two_phase_commit"] is True
    assert core["objects_deleted"] == 0
    assert core["active_after_retention"] == ["state-0002"]


def test_estimator_runner_uses_production_kernels_and_independent_fp64_oracles(
    tmp_path: Path,
) -> None:
    task_id = "stage1.05_estimators"
    output = "runs/stage01-estimators"
    runtime = TaskRuntime()
    runtime.register(Stage01CompositeTaskRunner(RunnerKind.ESTIMATOR, tmp_path))
    result = runtime.execute(_config(tmp_path, task_id, output))

    payload = _load_payload(tmp_path, output, result.artifact_refs["estimator_validation_report"])
    core = payload["core_evidence"]
    assert core["evidence_type"] == "importance_estimators"
    assert all(item["passed"] for item in core["comparisons"].values())
    assert core["same_batch_clipped_u_claim"] == "plugin_same_batch_clip_no_strict_unbiasedness"


def test_replay_core_hash_is_identical_in_two_empty_output_directories(tmp_path: Path) -> None:
    task_id = "stage0.11_test_quality_and_replay"
    hashes: list[str] = []
    for suffix in ("a", "b"):
        output = f"runs/stage01-replay-{suffix}"
        runtime = TaskRuntime()
        runtime.register(Stage01CompositeTaskRunner(RunnerKind.TEST_MATRIX, tmp_path))
        result = runtime.execute(_config(tmp_path, task_id, output))
        payload = _load_payload(tmp_path, output, result.artifact_refs["replay_report"])
        assert payload["core_evidence"]["hashes_equal"] is True
        hashes.append(payload["core_evidence_hash"])
    assert hashes[0] == hashes[1]


def test_stage01_evidence_schema_is_strict() -> None:
    schema = json.loads(
        (ROOT / "schemas/shared/stage01-task-evidence-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["properties"]["gate_status"] == {"const": "NOT_RUN"}


def test_formal_external_conditions_are_structured_blockers_not_local_pass(
    tmp_path: Path,
) -> None:
    task_id = "stage0.01_baseline_and_safety"
    config = ResolvedConfigV2.resolve(
        _formal_base_config(stage=0, task_id=task_id),
        task_id=task_id,
        overrides={
            "providers": {"num_labels": 3},
            "artifacts": {"output_dir": "runs/formal-stage0-audit"},
        },
    )
    runtime = TaskRuntime()
    runtime.register(Stage01CompositeTaskRunner(RunnerKind.AUDIT, tmp_path))

    result = runtime.execute(config)

    assert result.status.value == "BLOCKED"
    assert result.formal_eligible is False
    assert "server_unreachable" in {item.code.value for item in result.blockers}
    assert not result.artifact_refs
