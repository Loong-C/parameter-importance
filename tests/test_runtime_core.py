from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from param_importance_nlp.atomic import stable_json_bytes
from param_importance_nlp.lifecycle import ProcessState
from param_importance_nlp.runtime import (
    AttemptDisposition,
    CheckpointRetentionPolicy,
    CheckpointStore,
    EventRecord,
    EventType,
    JsonlEventSink,
    LocalReducer,
    LineageStore,
    OptimizerBridge,
    ProcessStateStore,
    StepPhase,
    StepTransaction,
    canonical_optimizer_steps,
    compute_global_clip_factor,
    load_tensor_bundle,
    publish_tensor_bundle,
    read_event_stream,
)


def test_tensor_bundle_round_trip_is_dtype_and_structure_exact(tmp_path: Path) -> None:
    state = {
        "step": 7,
        "tuple": ("x", None, True),
        "torch": torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16),
        "empty": torch.empty((0, 3), dtype=torch.float32),
        "numpy": np.array([1, 2], dtype=">i4"),
    }
    identity = publish_tensor_bundle(tmp_path / "object", state)
    restored, observed = load_tensor_bundle(tmp_path / "object")

    assert observed.manifest_sha256 == identity.manifest_sha256
    assert observed.tensor_count == 3
    assert restored["step"] == 7
    assert restored["tuple"] == ("x", None, True)
    assert restored["torch"].dtype == torch.bfloat16
    assert torch.equal(restored["torch"], state["torch"])
    assert tuple(restored["empty"].shape) == (0, 3)
    assert restored["numpy"].dtype.str == ">i4"
    assert np.array_equal(restored["numpy"], state["numpy"])


def test_tensor_bundle_is_immutable_and_rejects_unknown_state(tmp_path: Path) -> None:
    target = tmp_path / "object"
    publish_tensor_bundle(target, {"x": torch.tensor([1])})
    with pytest.raises(FileExistsError, match="BUNDLE_ALREADY_EXISTS"):
        publish_tensor_bundle(target, {"x": torch.tensor([2])})
    with pytest.raises(TypeError, match="STATE_UNSUPPORTED_TYPE"):
        publish_tensor_bundle(tmp_path / "bad", {"x": object()})


def test_tensor_bundle_rejects_corruption_and_noncanonical_manifest(
    tmp_path: Path,
) -> None:
    target = tmp_path / "object"
    publish_tensor_bundle(target, {"x": torch.tensor([1.0])})
    tensor_path = next((target / "tensors").iterdir())
    tensor_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="TENSOR_(SIZE|HASH)_MISMATCH"):
        load_tensor_bundle(target)

    other = tmp_path / "other"
    publish_tensor_bundle(other, {"x": torch.tensor([1.0])})
    manifest = other / "manifest.json"
    manifest.write_bytes(b"\xef\xbb\xbf" + manifest.read_bytes())
    with pytest.raises(ValueError, match="CANONICAL_JSON_BOM_FORBIDDEN"):
        load_tensor_bundle(other)


def test_tensor_bundle_rejects_untyped_state_and_ambiguous_tensor_references(
    tmp_path: Path,
) -> None:
    untyped = tmp_path / "untyped"
    publish_tensor_bundle(untyped, {"x": torch.tensor([1.0])})
    manifest_path = untyped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["state"] = {"x": 1}
    manifest_path.write_bytes(stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="STATE_UNKNOWN_KIND"):
        load_tensor_bundle(untyped)

    duplicated = tmp_path / "duplicated"
    publish_tensor_bundle(duplicated, {"x": torch.tensor([1.0])})
    manifest_path = duplicated / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tensor_ref = manifest["state"]["items"][0][1]
    manifest["state"] = {
        "kind": "list",
        "items": [tensor_ref, tensor_ref],
    }
    manifest_path.write_bytes(stable_json_bytes(manifest))
    with pytest.raises(ValueError, match="TENSOR_REFERENCE_COUNT_MISMATCH"):
        load_tensor_bundle(duplicated)


def test_checkpoint_requires_commit_and_reconciles_orphans(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    orphan = publish_tensor_bundle(store.objects / "orphan", {"step": 0})
    assert orphan.path.is_dir()

    first = store.publish(
        "step-1",
        {"parameter": torch.tensor([1.0]), "step": 1},
        generation=1,
        metadata={"coordinate_registry_hash": "a" * 64},
    )
    second = store.publish(
        "step-2",
        {"parameter": torch.tensor([2.0]), "step": 2},
        generation=2,
        metadata={"coordinate_registry_hash": "a" * 64},
        parent_checkpoint_id="step-1",
    )
    restored, commit = store.load(
        "step-2", expected_metadata={"coordinate_registry_hash": "a" * 64}
    )
    assert torch.equal(restored["parameter"], torch.tensor([2.0]))
    assert commit == second
    assert first.parent_checkpoint_id is None

    report = store.reconcile()
    assert report["valid"] == ["step-1", "step-2"]
    assert report["orphan_objects"] == ["orphan"]
    latest = json.loads((store.root / "latest.json").read_text(encoding="utf-8"))
    assert latest["checkpoint_id"] == "step-2"


def test_checkpoint_publish_fully_validates_parent_before_creating_child(
    tmp_path: Path,
) -> None:
    """父 commit 文件存在不等于父节点可恢复，坏父节点不能产生权威后代。"""

    generation_store = CheckpointStore(tmp_path / "generation")
    generation_store.publish(
        "parent",
        {"x": torch.tensor([1.0])},
        generation=3,
        metadata={},
    )
    for child_id, generation in (("equal", 3), ("older", 2)):
        with pytest.raises(
            ValueError,
            match="CHECKPOINT_GENERATION_NOT_STRICTLY_INCREASING",
        ):
            generation_store.publish(
                child_id,
                {"x": torch.tensor([2.0])},
                generation=generation,
                metadata={},
                parent_checkpoint_id="parent",
            )
        assert not (generation_store.objects / child_id).exists()
        assert not (generation_store.commits / f"{child_id}.json").exists()

    corrupt_store = CheckpointStore(tmp_path / "corrupt-parent")
    corrupt_store.publish(
        "parent",
        {"x": torch.tensor([1.0])},
        generation=0,
        metadata={},
    )
    tensor_path = next((corrupt_store.objects / "parent" / "tensors").iterdir())
    tensor_path.write_bytes(b"corrupt-parent")
    with pytest.raises(ValueError, match="TENSOR_(SIZE|HASH)_MISMATCH"):
        corrupt_store.publish(
            "child",
            {"x": torch.tensor([2.0])},
            generation=1,
            metadata={},
            parent_checkpoint_id="parent",
        )
    assert not (corrupt_store.objects / "child").exists()
    assert not (corrupt_store.commits / "child.json").exists()

    tombstone_store = CheckpointStore(tmp_path / "tombstone-parent")
    tombstone_store.publish(
        "parent",
        {"x": torch.tensor([1.0])},
        generation=0,
        metadata={},
    )
    tombstone_store.write_tombstone("parent", reason="retained child must fork first")
    with pytest.raises(ValueError, match="CHECKPOINT_TOMBSTONED:parent"):
        tombstone_store.publish(
            "child",
            {"x": torch.tensor([2.0])},
            generation=1,
            metadata={},
            parent_checkpoint_id="parent",
        )
    assert not (tombstone_store.objects / "child").exists()


def test_checkpoint_reconcile_rejects_broken_ancestry_and_repairs_latest(
    tmp_path: Path,
) -> None:
    """子对象自身完好时，缺失父节点仍会让它退出恢复集合。"""

    store = CheckpointStore(tmp_path / "missing-parent")
    store.publish("root", {"x": torch.tensor([0.0])}, generation=0, metadata={})
    store.publish(
        "child",
        {"x": torch.tensor([1.0])},
        generation=1,
        metadata={},
        parent_checkpoint_id="root",
    )
    child_path = store.commits / "child.json"
    child_value = json.loads(child_path.read_text(encoding="utf-8"))
    child_value["parent_checkpoint_id"] = "missing"
    child_path.write_bytes(stable_json_bytes(child_value))

    with pytest.raises(ValueError, match="CHECKPOINT_LINEAGE_PARENT_MISSING"):
        store.load("child")
    with pytest.raises(ValueError, match="CHECKPOINT_LINEAGE_PARENT_MISSING"):
        store.discover()

    report = store.reconcile()
    assert report["valid"] == ["root"]
    assert [item["checkpoint_id"] for item in report["invalid"]] == ["child"]
    assert "CHECKPOINT_LINEAGE_PARENT_MISSING" in report["invalid"][0]["reason"]
    latest_state, latest_commit = store.load_latest()
    assert latest_commit.checkpoint_id == "root"
    assert torch.equal(latest_state["x"], torch.tensor([0.0]))


def test_checkpoint_lineage_rejects_cycles_reversed_generation_and_bad_root(
    tmp_path: Path,
) -> None:
    """环、generation 倒序与祖先 bundle 损坏均不能留下坏的 latest。"""

    cycle_store = CheckpointStore(tmp_path / "cycle")
    cycle_store.publish("root", {"x": torch.tensor([0.0])}, generation=0, metadata={})
    cycle_store.publish(
        "child",
        {"x": torch.tensor([1.0])},
        generation=1,
        metadata={},
        parent_checkpoint_id="root",
    )
    root_path = cycle_store.commits / "root.json"
    root_value = json.loads(root_path.read_text(encoding="utf-8"))
    # root(2) -> child(1) 在第一条边上满足 parent < child；继续回到 root 时
    # 必须由显式 visited 集合识别闭环。反向从 child 出发则立即发现 generation 倒序。
    root_value["generation"] = 2
    root_value["parent_checkpoint_id"] = "child"
    root_path.write_bytes(stable_json_bytes(root_value))
    with pytest.raises(ValueError, match="CHECKPOINT_LINEAGE_CYCLE"):
        cycle_store.load("root")
    with pytest.raises(
        ValueError,
        match="CHECKPOINT_LINEAGE_GENERATION_NOT_INCREASING",
    ):
        cycle_store.load("child")
    cycle_report = cycle_store.reconcile()
    assert cycle_report["valid"] == []
    assert {item["checkpoint_id"] for item in cycle_report["invalid"]} == {
        "root",
        "child",
    }
    assert not (cycle_store.root / "latest.json").exists()

    corrupt_store = CheckpointStore(tmp_path / "bad-root")
    corrupt_store.publish("root", {"x": torch.tensor([0.0])}, generation=0, metadata={})
    corrupt_store.publish(
        "middle",
        {"x": torch.tensor([1.0])},
        generation=2,
        metadata={},
        parent_checkpoint_id="root",
    )
    corrupt_store.publish(
        "leaf",
        {"x": torch.tensor([2.0])},
        generation=3,
        metadata={},
        parent_checkpoint_id="middle",
    )
    tensor_path = next((corrupt_store.objects / "root" / "tensors").iterdir())
    tensor_path.write_bytes(b"bad-root")
    report = corrupt_store.reconcile()
    assert report["valid"] == []
    assert {item["checkpoint_id"] for item in report["invalid"]} == {
        "root",
        "middle",
        "leaf",
    }
    assert not (corrupt_store.root / "latest.json").exists()


def test_checkpoint_latest_remains_highest_recoverable_generation(
    tmp_path: Path,
) -> None:
    """后发布的独立低 generation 根节点不能覆盖更高的恢复点。"""

    store = CheckpointStore(tmp_path / "latest-order")
    store.publish("high", {"x": torch.tensor([10])}, generation=10, metadata={})
    store.publish("low", {"x": torch.tensor([0])}, generation=0, metadata={})
    state, commit = store.load_latest()
    assert commit.checkpoint_id == "high"
    assert torch.equal(state["x"], torch.tensor([10]))


def test_checkpoint_rejects_path_escape_corruption_and_tombstone(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "checkpoints")
    store.publish("safe", {"x": torch.tensor([1])}, generation=0, metadata={})
    commit_path = store.commits / "safe.json"
    value = json.loads(commit_path.read_text(encoding="utf-8"))
    value["object_relative_path"] = "../outside"
    commit_path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="CHECKPOINT_OBJECT_PATH_MISMATCH"):
        store.load("safe")

    healthy = CheckpointStore(tmp_path / "healthy")
    healthy.publish("safe", {"x": torch.tensor([1])}, generation=0, metadata={})
    healthy.write_tombstone("safe", reason="retention policy")
    with pytest.raises(ValueError, match="CHECKPOINT_TOMBSTONED"):
        healthy.load("safe")
    assert healthy.discover() == ()


def test_checkpoint_retention_is_deterministic_idempotent_and_tombstone_only(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "retention")
    parent_id = None
    for generation in range(5):
        checkpoint_id = f"step-{generation}"
        store.publish(
            checkpoint_id,
            {"parameter": torch.tensor([float(generation)])},
            generation=generation,
            metadata={"generation": generation},
            parent_checkpoint_id=parent_id,
        )
        parent_id = checkpoint_id

    selection = store.select_retention(
        CheckpointRetentionPolicy(
            keep_latest=2,
            best_checkpoint_ids=("step-1",),
            milestone_checkpoint_ids=("step-2",),
        )
    )
    assert selection.keep_checkpoint_ids == (
        "step-1",
        "step-2",
        "step-3",
        "step-4",
    )
    assert selection.tombstone_checkpoint_ids == ("step-0",)
    assert selection.keep_reasons["step-1"] == ("best",)
    assert selection.keep_reasons["step-3"] == ("latest",)
    assert selection.to_dict()["selection_hash"] == selection.selection_hash

    object_names_before = sorted(path.name for path in store.objects.iterdir())
    commit_names_before = sorted(path.name for path in store.commits.iterdir())
    applied = store.apply_retention(selection, reason="local retention fixture")
    assert applied.newly_tombstoned == ("step-0",)
    assert applied.already_tombstoned == ()
    assert applied.objects_deleted == 0
    assert sorted(path.name for path in store.objects.iterdir()) == object_names_before
    assert sorted(path.name for path in store.commits.iterdir()) == commit_names_before
    assert [item.checkpoint_id for item in store.discover()] == [
        "step-1",
        "step-2",
        "step-3",
        "step-4",
    ]
    with pytest.raises(ValueError, match="CHECKPOINT_TOMBSTONED"):
        store.load("step-0")

    replayed = store.apply_retention(selection, reason="local retention fixture")
    assert replayed.newly_tombstoned == ()
    assert replayed.already_tombstoned == ("step-0",)
    assert replayed.objects_deleted == 0


def test_checkpoint_retention_rejects_stale_selection_and_can_keep_lineage(
    tmp_path: Path,
) -> None:
    store = CheckpointStore(tmp_path / "stale-retention")
    store.publish("step-0", {"x": torch.tensor([0])}, generation=0, metadata={})
    store.publish(
        "step-1",
        {"x": torch.tensor([1])},
        generation=1,
        metadata={},
        parent_checkpoint_id="step-0",
    )
    lineage_selection = store.select_retention(
        CheckpointRetentionPolicy(keep_latest=1, keep_lineage_ancestors=True)
    )
    assert lineage_selection.keep_checkpoint_ids == ("step-0", "step-1")
    assert lineage_selection.tombstone_checkpoint_ids == ()
    assert lineage_selection.keep_reasons["step-0"] == ("lineage_ancestor",)

    stale_selection = store.select_retention(CheckpointRetentionPolicy(keep_latest=1))
    store.publish(
        "step-2",
        {"x": torch.tensor([2])},
        generation=2,
        metadata={},
        parent_checkpoint_id="step-1",
    )
    with pytest.raises(ValueError, match="CHECKPOINT_RETENTION_ACTIVE_SET_DRIFT"):
        store.apply_retention(stale_selection)
    with pytest.raises(
        ValueError,
        match="CHECKPOINT_RETENTION_REFERENCED_ID_NOT_ACTIVE",
    ):
        store.select_retention(
            CheckpointRetentionPolicy(
                keep_latest=1,
                protected_checkpoint_ids=("missing",),
            )
        )


def test_cpu_training_checkpoint_resume_matches_uninterrupted(tmp_path: Path) -> None:
    """固定 CPU 输入上，checkpoint 边界不能改变 Momentum SGD 的最终状态。"""

    inputs = (
        (torch.tensor([1.0, -0.5]), torch.tensor([0.25, -0.75])),
        (torch.tensor([-0.25, 2.0]), torch.tensor([0.5, 1.0])),
        (torch.tensor([1.5, 0.75]), torch.tensor([-0.5, 0.25])),
        (torch.tensor([0.4, -1.25]), torch.tensor([0.1, -0.2])),
        (torch.tensor([-0.8, 0.3]), torch.tensor([0.7, -0.4])),
        (torch.tensor([1.2, 0.2]), torch.tensor([-0.3, 0.6])),
    )

    def make_training_state() -> tuple[torch.nn.Parameter, torch.optim.Optimizer]:
        parameter = torch.nn.Parameter(torch.tensor([0.75, -0.25]))
        optimizer = torch.optim.SGD([parameter], lr=0.05, momentum=0.9)
        return parameter, optimizer

    def train_range(
        parameter: torch.nn.Parameter,
        optimizer: torch.optim.Optimizer,
        start: int,
        stop: int,
    ) -> None:
        for index in range(start, stop):
            features, target = inputs[index]
            optimizer.zero_grad(set_to_none=True)
            prediction = parameter * features
            loss = torch.sum((prediction - target) ** 2)
            loss.backward()
            optimizer.step()

    uninterrupted_parameter, uninterrupted_optimizer = make_training_state()
    train_range(uninterrupted_parameter, uninterrupted_optimizer, 0, len(inputs))

    split_parameter, split_optimizer = make_training_state()
    train_range(split_parameter, split_optimizer, 0, 3)
    store = CheckpointStore(tmp_path / "training-resume")
    store.publish(
        "step-3",
        {
            "parameter": split_parameter.detach().clone(),
            "optimizer": split_optimizer.state_dict(),
            "next_step": 3,
        },
        generation=3,
        metadata={"device": "cpu", "dtype": "float32"},
    )

    restored, _ = store.load(
        "step-3",
        expected_metadata={"device": "cpu", "dtype": "float32"},
    )
    resumed_parameter, resumed_optimizer = make_training_state()
    with torch.no_grad():
        resumed_parameter.copy_(restored["parameter"])
    resumed_optimizer.load_state_dict(restored["optimizer"])
    train_range(resumed_parameter, resumed_optimizer, restored["next_step"], len(inputs))

    assert torch.equal(resumed_parameter.detach(), uninterrupted_parameter.detach())
    uninterrupted_buffer = uninterrupted_optimizer.state[uninterrupted_parameter][
        "momentum_buffer"
    ]
    resumed_buffer = resumed_optimizer.state[resumed_parameter]["momentum_buffer"]
    assert torch.equal(resumed_buffer, uninterrupted_buffer)


def _event(sequence: int, *, step: int | None = None) -> EventRecord:
    return EventRecord.create(
        experiment_id="experiment-1",
        run_id="run-1",
        attempt_id="attempt-1",
        session_id="session-1",
        rank=0,
        event_type=EventType.OPTIMIZER_STEP,
        sequence=sequence,
        payload={"global_step": sequence if step is None else step},
        event_id=f"event-{sequence}",
        occurred_at="2026-07-22T00:00:00+00:00",
    )


def test_event_stream_is_typed_monotonic_and_single_writer(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with JsonlEventSink(path) as sink:
        sink.append(_event(0), critical=True)
        with pytest.raises(RuntimeError, match="EVENT_WRITER_ALREADY_ACTIVE"):
            JsonlEventSink(path)
        with pytest.raises(ValueError, match="EVENT_SEQUENCE_GAP"):
            sink.append(_event(2))
    with JsonlEventSink(path) as resumed:
        resumed.append(_event(1))

    observed = read_event_stream(path)
    assert [item.sequence for item in observed] == [0, 1]
    assert [item.payload["global_step"] for item in canonical_optimizer_steps([observed])] == [0, 1]


def test_event_stream_rejects_sensitive_payload_and_truncation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="EVENT_SENSITIVE_VALUE"):
        EventRecord.create(
            experiment_id="e",
            run_id="r",
            attempt_id="a",
            session_id="s",
            rank=0,
            event_type=EventType.SYSTEM,
            sequence=0,
            payload={"api_key": "api_key=super-secret-value"},
        )
    path = tmp_path / "truncated.jsonl"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="EVENT_STREAM_TRUNCATED_FINAL_LINE"):
        read_event_stream(path)


def test_local_reducer_and_global_clip_are_explicit() -> None:
    reducer = LocalReducer()
    tensor = torch.tensor([3.0, 4.0])
    assert reducer.sum_tensors({"x": tensor})["x"] is tensor
    assert reducer.sum_int(3) == 3
    norm, factor = compute_global_clip_factor({"x": tensor}, 2.5)
    assert norm == pytest.approx(5.0)
    assert factor == pytest.approx(0.5)
    with pytest.raises(ValueError, match="NONFINITE_GRADIENT"):
        compute_global_clip_factor({"x": torch.tensor([float("nan")])}, 1.0)
    with pytest.raises(ValueError, match="MAX_NORM"):
        compute_global_clip_factor({"x": tensor}, float("inf"))
    with pytest.raises(ValueError, match="CLIP_EPS"):
        compute_global_clip_factor({"x": tensor}, 1.0, eps=0.0)


def test_optimizer_bridge_sgd_momentum_and_adamw_decomposition() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    sgd = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    bridge = OptimizerBridge({"p": parameter}, sgd)
    parameter.grad = torch.tensor([0.5])
    outcome = bridge.step()
    assert torch.allclose(outcome.total_delta["p"], torch.tensor([-0.05]))
    assert torch.equal(outcome.data_delta["p"], outcome.total_delta["p"])
    assert torch.equal(outcome.weight_decay_delta["p"], torch.zeros(1))

    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    adamw = torch.optim.AdamW([parameter], lr=0.1, weight_decay=0.2)
    bridge = OptimizerBridge({"p": parameter}, adamw)
    parameter.grad = torch.tensor([0.5])
    outcome = bridge.step()
    assert torch.allclose(outcome.weight_decay_delta["p"], torch.tensor([-0.04]))
    assert torch.allclose(
        outcome.total_delta["p"],
        outcome.data_delta["p"] + outcome.weight_decay_delta["p"],
    )


def test_optimizer_bridge_rejects_unsupported_policy_before_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    coupled = torch.optim.SGD([parameter], lr=0.1, weight_decay=0.2)
    with pytest.raises(ValueError, match="SGD_COUPLED_WEIGHT_DECAY"):
        OptimizerBridge({"p": parameter}, coupled)
    assert torch.equal(parameter.detach(), torch.tensor([2.0]))
    unsupported = torch.optim.Adam([parameter], lr=0.1)
    with pytest.raises(TypeError, match="OPTIMIZER_UNSUPPORTED"):
        OptimizerBridge({"p": parameter}, unsupported)
    assert torch.equal(parameter.detach(), torch.tensor([2.0]))


def test_optimizer_bridge_rejects_nonfinite_gradient_before_any_mutation() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
    bridge = OptimizerBridge({"p": parameter}, optimizer)
    parameter.grad = torch.tensor([float("nan")])
    before = parameter.detach().clone()
    with pytest.raises(ValueError, match="NONFINITE_GRADIENT_STEP_SKIPPED"):
        bridge.step()
    assert torch.equal(parameter.detach(), before)
    assert optimizer.state == {}


def test_adamw_bridge_does_not_invent_decay_for_grad_none_or_frozen_parameter() -> None:
    active = torch.nn.Parameter(torch.tensor([2.0]))
    no_grad = torch.nn.Parameter(torch.tensor([3.0]))
    frozen = torch.nn.Parameter(torch.tensor([4.0]), requires_grad=False)
    optimizer = torch.optim.AdamW([active, no_grad], lr=0.1, weight_decay=0.2)
    bridge = OptimizerBridge(
        {"active": active, "no_grad": no_grad, "frozen": frozen}, optimizer
    )
    active.grad = torch.tensor([0.5])
    no_grad.grad = None
    outcome = bridge.step()
    assert torch.equal(outcome.total_delta["no_grad"], torch.zeros(1))
    assert torch.equal(outcome.weight_decay_delta["no_grad"], torch.zeros(1))
    assert "frozen" not in outcome.total_delta


def test_lineage_requires_one_explicit_canonical_attempt(tmp_path: Path) -> None:
    store = LineageStore(tmp_path / "lineage.json", run_id="run-1")
    store.register("attempt-1", parent_attempt_id=None, reason="initial")
    store.register("attempt-2", parent_attempt_id="attempt-1", reason="resume")
    selected = store.select_canonical(
        "attempt-2", reason="completed evidence", evidence_hash="a" * 64
    )
    assert selected.disposition is AttemptDisposition.CANONICAL
    records = {item.attempt_id: item for item in store.records()}
    assert records["attempt-1"].disposition is AttemptDisposition.SUPERSEDED
    assert records["attempt-2"].disposition is AttemptDisposition.CANONICAL

    orphan_store = LineageStore(tmp_path / "orphan.json", run_id="run-2")
    orphan_store.register("attempt-1", parent_attempt_id=None, reason="discovered")
    orphan_store.mark_orphan("attempt-1", reason="missing authoritative commit")
    with pytest.raises(ValueError, match="ORPHAN_CANNOT_BE_CANONICAL"):
        orphan_store.select_canonical(
            "attempt-1", reason="invalid", evidence_hash="b" * 64
        )


def test_step_transaction_separates_parameter_post_and_attempt_commit() -> None:
    transaction = StepTransaction(global_step=3, attempt_index=0)
    parameter_post = transaction.mark_parameter_post("a" * 64)
    assert parameter_post.phase is StepPhase.PARAMETER_POST_STATE
    assert parameter_post.attempt_commit_state_hash is None
    committed = parameter_post.commit_attempt("b" * 64)
    assert committed.phase is StepPhase.ATTEMPT_COMMIT_STATE
    assert committed.parameter_post_state_hash == "a" * 64
    assert committed.attempt_commit_state_hash == "b" * 64
    with pytest.raises(ValueError, match="INVALID_PHASE"):
        transaction.commit_attempt("b" * 64)
    with pytest.raises(ValueError, match="CONTROL_STATE_CHANGE"):
        parameter_post.commit_attempt("a" * 64)

    skipped = StepTransaction(global_step=3, attempt_index=1).skip("nonfinite gradient")
    assert skipped.phase is StepPhase.SKIPPED
    assert skipped.parameter_post_state_hash is None


def test_process_state_revision_and_heartbeat_are_independent(tmp_path: Path) -> None:
    store = ProcessStateStore.create(
        tmp_path / "attempt-state.json",
        run_id="run-1",
        attempt_id="attempt-1",
    )
    assert store.read().state.value == "STARTING"
    running = store.transition(
        # 复用 Stage 0 的冻结枚举，保持旧入口兼容。
        ProcessState.RUNNING,
        reason="worker ready",
        expected_revision=0,
    )
    assert running.revision == 1
    heartbeat = store.heartbeat(sequence=0)
    assert heartbeat["state_revision"] == 1
    assert store.read().revision == 1
    with pytest.raises(ValueError, match="HEARTBEAT_SEQUENCE_GAP"):
        store.heartbeat(sequence=2)

    observed = datetime.fromisoformat(heartbeat["observed_at"])
    assert not store.heartbeat_is_stale(
        now=observed + timedelta(seconds=1), threshold=timedelta(seconds=2)
    )
    assert store.heartbeat_is_stale(
        now=observed.astimezone(timezone.utc) + timedelta(seconds=3),
        threshold=timedelta(seconds=2),
    )
