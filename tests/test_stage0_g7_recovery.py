"""Stage 0 S0.9 group-commit, recovery, and schema regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from param_importance_nlp.atomic import atomic_write_bytes, sha256_file
from param_importance_nlp.cli import _validate_project_json_schema
from param_importance_nlp.providers import DeterministicBatchCursor, TorchModelAdapter, TrainingMicrobatch
from param_importance_nlp.runtime import (
    CheckpointGroupStore,
    CheckpointRetentionPolicy,
    CheckpointStore,
    JsonlEventSink,
    TrainingEngine,
    TrainingRunSpec,
    checkpoint_state_sha256,
)
from param_importance_nlp.stage0_g7_recovery import _cpu_fp32_suite
from param_importance_nlp.stage0_g7_recovery_worker import (
    _IMPORTANCE_ENABLED,
    Stage0G7RecoveryWorkerError,
    _validate_nccl_transport_environment,
)


ROOT = Path(__file__).resolve().parents[1]


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


def _steps() -> tuple[tuple[TrainingMicrobatch, ...], ...]:
    return tuple(
        (
            TrainingMicrobatch(
                f"batch-{index}",
                {
                    "features": torch.tensor([[1.0 + index, -0.5]], dtype=torch.float32),
                    "labels": torch.tensor([index % 2], dtype=torch.int64),
                },
                (f"sample-{index}",),
            ),
        )
        for index in range(2)
    )


def _metadata() -> dict[str, object]:
    return {
        "config_hash": "a" * 64,
        "environment_hash": "b" * 64,
        "model_manifest_id": "model-fixture",
        "data_manifest_id": "data-fixture",
        "sampler_seed": 17,
        "epoch": 0,
        "committed_global_batch": 1,
        "next_global_batch": 1,
        "prefetch_policy": "direct_committed_cursor",
        "snapshot_type": "optimizer_step_checkpoint",
        "state_extension_schema": "online-importance-state-v1",
        "save_wall_seconds": 0.01,
        "checkpoint_bytes": 1,
        "peak_memory_bytes": 0,
    }


def _rank_checkpoint(tmp_path: Path) -> tuple[CheckpointStore, Path, str]:
    torch.manual_seed(11)
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    store = CheckpointStore(tmp_path / "rank-store")
    event_path = tmp_path / "events" / "rank-0000-session.jsonl"
    with JsonlEventSink(event_path) as sink:
        engine = TrainingEngine(
            spec=TrainingRunSpec(
                "group-fixture-rank-0000",
                "local_fixture",
                max_steps=2,
                max_attempts=2,
                importance_enabled=False,
            ),
            model=TorchModelAdapter(model, task_type="sequence_classification"),
            optimizer=optimizer,
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=1),
            cursor=DeterministicBatchCursor(_steps()),
            checkpoint_store=store,
            event_sink=sink,
            experiment_id="group-fixture",
            attempt_id="attempt-0",
            session_id="session-0",
        )
        result = engine.run(until_step=1)
    assert result.status == "PAUSED"
    checkpoint_id = result.state.last_checkpoint_id
    assert checkpoint_id is not None
    return store, event_path, checkpoint_id


def test_checkpoint_group_commit_is_the_only_distributed_authority(tmp_path: Path) -> None:
    store, event_path, checkpoint_id = _rank_checkpoint(tmp_path)
    state, _ = store.load(checkpoint_id)
    sequence = int(state["training_state"]["event_sequence"]) - 1
    group = CheckpointGroupStore(tmp_path, "group")
    commit = group.publish(
        "group-step-1",
        generation=1,
        run_id="group-fixture",
        world_size=1,
        rank_checkpoints=(
            {
                "rank": 0,
                "checkpoint_store_ref": "rank-store",
                "checkpoint_id": checkpoint_id,
                "event_pointer": {
                    "event_ref": event_path.relative_to(tmp_path).as_posix(),
                    "event_sha256": sha256_file(event_path),
                    "checkpoint_event_sequence": sequence,
                },
            },
        ),
        metadata=_metadata(),
    )

    loaded, replayed = group.load(
        commit.checkpoint_id,
        expected_run_id="group-fixture",
        expected_world_size=1,
        expected_config_hash="a" * 64,
        expected_data_manifest_id="data-fixture",
    )
    assert len(loaded) == 1
    assert replayed.commit_sha256 == commit.commit_sha256
    assert checkpoint_state_sha256(loaded[0]) == commit.rank_checkpoints[0]["full_state_sha256"]
    assert group.reconcile()["valid"] == ["group-step-1"]
    commit_value = json.loads(
        (group.commits / "group-step-1.json").read_text(encoding="utf-8")
    )
    lineage_value = json.loads(
        (group.root / "lineage.json").read_text(encoding="utf-8")
    )
    assert commit_value["schema_version"] == "runtime.checkpoint-group-commit.v1"
    assert lineage_value["schema_version"] == "runtime.checkpoint-group-lineage.v1"

    with pytest.raises(ValueError, match="WORLD_SIZE_INCOMPATIBLE"):
        group.load("group-step-1", expected_world_size=4)
    with pytest.raises(ValueError, match="CONFIG_INCOMPATIBLE"):
        group.load("group-step-1", expected_config_hash="c" * 64)
    with pytest.raises(ValueError, match="RANK_COUNT_INVALID"):
        CheckpointGroupStore(tmp_path, "missing-rank").publish(
            "bad",
            generation=1,
            run_id="group-fixture",
            world_size=2,
            rank_checkpoints=(),
            metadata=_metadata(),
        )


def test_group_reconcile_repairs_non_authoritative_views(tmp_path: Path) -> None:
    store, event_path, checkpoint_id = _rank_checkpoint(tmp_path)
    state, _ = store.load(checkpoint_id)
    group = CheckpointGroupStore(tmp_path, "group")
    group.publish(
        "group-step-1",
        generation=1,
        run_id="group-fixture",
        world_size=1,
        rank_checkpoints=(
            {
                "rank": 0,
                "checkpoint_store_ref": "rank-store",
                "checkpoint_id": checkpoint_id,
                "event_pointer": {
                    "event_ref": event_path.relative_to(tmp_path).as_posix(),
                    "event_sha256": sha256_file(event_path),
                    "checkpoint_event_sequence": int(state["training_state"]["event_sequence"]) - 1,
                },
            },
        ),
        metadata=_metadata(),
    )
    atomic_write_bytes(group.root / "latest.json", b"not-json\n")
    (group.root / "checkpoint-events.jsonl").unlink()
    (group.root / "lineage.json").unlink()

    result = group.reconcile()

    assert result["latest_checkpoint_id"] == "group-step-1"
    assert result["derived_diagnostics"]
    assert (group.root / "latest.json").is_file()
    assert (group.root / "checkpoint-events.jsonl").is_file()
    assert (group.root / "lineage.json").is_file()


def test_cpu_fp32_direct_prefetch_and_fresh_resume_are_exact(tmp_path: Path) -> None:
    reference, report = _cpu_fp32_suite(tmp_path, "cpu-reference")

    assert reference == "cpu-reference/report.json"
    assert report["status"] == "PASS"
    assert all(report["state_fields_exact"].values())
    assert report["direct_num_workers"] == 0
    assert report["formal_num_workers"] == 2


def test_g7_recovery_worker_disables_importance_for_single_microbatch() -> None:
    assert _IMPORTANCE_ENABLED is False


def test_g7_recovery_worker_freezes_nccl_p2p_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    with pytest.raises(
        Stage0G7RecoveryWorkerError,
        match="G7_RECOVERY_WORKER_NCCL_P2P_ENVIRONMENT_INVALID",
    ):
        _validate_nccl_transport_environment()
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    _validate_nccl_transport_environment()


def test_recovery_schemas_are_valid_project_schema_documents() -> None:
    names = (
        "runtime-checkpoint-group-commit-v1.json",
        "runtime-checkpoint-group-lineage-v1.json",
        "runtime-checkpoint-purge-intent-v1.json",
        "runtime-checkpoint-purge-record-v1.json",
        "stage0-g7-recovery-worker-plan-v1.json",
        "stage0-g7-recovery-worker-report-v1.json",
        "stage0-g7-recovery-evidence-v1.json",
        "stage0-g7-recovery-formalization-index-v1.json",
    )
    for name in names:
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        _validate_project_json_schema(schema)


def test_purge_records_validate_against_published_schemas(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "purge-schema")
    store.publish("old", {"x": torch.tensor([0])}, generation=0, metadata={})
    store.publish(
        "current",
        {"x": torch.tensor([1])},
        generation=1,
        metadata={},
        parent_checkpoint_id="old",
    )
    selection = store.select_retention(
        CheckpointRetentionPolicy(keep_latest=1)
    )
    store.apply_retention(selection, reason="schema fixture")
    purge = store.purge_tombstoned_object(
        "old",
        reason="schema fixture exact-ID purge",
        protected_checkpoint_ids=selection.keep_checkpoint_ids,
    )
    for schema_name, record_path in (
        ("runtime-checkpoint-purge-intent-v1.json", purge.purge_intent_path),
        ("runtime-checkpoint-purge-record-v1.json", purge.purge_record_path),
    ):
        schema = json.loads(
            (ROOT / "schemas" / schema_name).read_text(encoding="utf-8")
        )
        value = json.loads((store.root / record_path).read_text(encoding="utf-8"))
        _validate_project_json_schema(schema)
        assert value["schema_version"] == (
            "runtime.checkpoint-purge-intent.v1"
            if schema_name == "runtime-checkpoint-purge-intent-v1.json"
            else "runtime.checkpoint-purge-record.v1"
        )
