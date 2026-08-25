from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import os

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s207_formal import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
    S27CellPlan,
    S27DetachedStatus,
    S27FrozenInputs,
    S27G25Blocked,
    S27MappingUnit,
    S27Plan,
    S27PreparationBlocked,
    S27RawUnit,
    S27StatusStore,
    StrictG25Reducer,
    prepare_s27_plan,
    validate_gpu_inventory,
)
from param_importance_nlp.experiments.stage2_s207_runner import partition_s27_units


def _gate_payload() -> dict[str, object]:
    gate = GateRecord(
        gate_id="stage2.G2.4b",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={"selected_batch_size": 32},
        threshold={"required_status": "READY_FOR_QUALIFICATION"},
        evidence_refs=("s206/report.json",),
    )
    return gate.to_dict()


def _matrix_payload(gate_hash: str) -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": "stage2-formal-pilot-matrix-freeze-v1",
        "freeze_id": "s206-formal-matrix-freeze",
        "scope": "formal",
        "status": "FORMAL_FROZEN",
        "anchor_ids": list(EXPECTED_CELL_IDS),
        "candidate_evaluations": [],
        "b_primary": 32,
        "m_primary": 16,
        "r_primary": 200,
        "completion_denominator": 1200,
        "cost_semantics": ["scientific_equal_sample_cost", "isolated_estimator_cost", "online_training_incremental_cost"],
        "cost_observations": {},
        "cost_io_quiescent": True,
        "pilot_draw_stream": "pilot",
        "confirmatory_draw_stream": "confirmatory",
        "pilot_report_hash": "1" * 64,
        "pilot_mapping_hash": "2" * 64,
        "sampling_plan_hash": "3" * 64,
        "execution_evidence_hash": "4" * 64,
        "formal_eligible": True,
        "qualification_gate_hash": gate_hash,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _mapping_payload(matrix: dict[str, object], gate_hash: str) -> dict[str, object]:
    # The preparation test only needs to prove that S2.7 refuses a mapping
    # whose formal freeze identity is wrong; it must not create confirmatory
    # draws as part of the test.
    body: dict[str, object] = {
        "schema_version": "stage2-formal-confirmatory-mapping-v1",
        "mapping_id": "s206-formal-confirmatory-mapping",
        "scope": "formal",
        "stream": "confirmatory",
        "freeze_hash": str(matrix["artifact_hash"]),
        "sampling_plan_hash": str(matrix["sampling_plan_hash"]),
        "pilot_draw_ids": [],
        "pilot_draw_id_count": 0,
        "pilot_draw_id_hash": canonical_json_hash([]),
        "confirmatory_draw_id_count": 0,
        "confirmatory_draw_id_hash": canonical_json_hash([]),
        "cells": [],
        "draw_id_unique": True,
        "sample_id_collision_count": 0,
        "complete": True,
        "qualification_gate_hash": gate_hash,
        "formal_eligible": True,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _checkpoint_rows() -> dict[str, dict[str, object]]:
    return {
        cell: {
            "checkpoint_ref": f"s23/checkpoints/{index}.json",
            "checkpoint_hash": f"{index + 10:064x}",
            "checkpoint_id": f"checkpoint-{index}",
        }
        for index, cell in enumerate(EXPECTED_CELL_IDS)
    }


def _reference_rows() -> dict[str, dict[str, object]]:
    return {
        cell: {
            "reference_ref": f"s204/reference/{index}.json",
            "reference_hash": f"{index + 30:064x}",
            "gate_ref": f"s204/gates/g23-{index}.json",
            "gate_hash": f"{index + 50:064x}",
            "task_id": "stage2.04_reference_target",
            "gate_id": "stage2.G2.3",
            "gate_status": "PASS",
            "scope": "formal",
            "formal_eligible": True,
            "independent": True,
        }
        for index, cell in enumerate(EXPECTED_CELL_IDS)
    }


def _plan_for_reducer() -> S27Plan:
    units: list[S27MappingUnit] = []
    for cell_index, cell in enumerate(EXPECTED_CELL_IDS):
        for repetition in range(200):
            unit_id = f"{cell}::rep-{repetition:04d}"
            draws = tuple(f"draw-{cell_index:02d}-{repetition:04d}-{index:02d}" for index in range(32))
            samples = tuple(f"sample-{(repetition * 32 + index) % 251}" for index in range(32))
            units.append(S27MappingUnit(unit_id, cell, f"rep-{repetition:04d}", 32, 16, draws, samples, "a" * 64))
    frozen = S27FrozenInputs(
        matrix_ref="s206/matrix.json",
        matrix_hash="b" * 64,
        mapping_ref="s206/mapping.json",
        mapping_hash="c" * 64,
        g24b_gate_ref="s206/g24b.json",
        g24b_gate_hash="d" * 64,
        sampling_plan_hash="e" * 64,
        batch_size=32,
        microbatch_count=16,
        repetitions=200,
        completion_denominator=1200,
        max_failure_fraction=0.0,
        cost_required=True,
        units=tuple(units),
    )
    cells = tuple(
        S27CellPlan(
            cell_id=cell,
            model_id=cell.split(":", 1)[0],
            training_stage=cell.split(":", 1)[1],
            checkpoint_ref=f"s23/checkpoints/{index}.json",
            checkpoint_hash=f"{index + 10:064x}",
            checkpoint_id=f"checkpoint-{index}",
            reference_ref=f"s204/reference/{index}.json",
            reference_hash=f"{index + 30:064x}",
            reference_gate_ref=f"s204/gates/g23-{index}.json",
            reference_gate_hash=f"{index + 50:064x}",
            expected_unit_ids=tuple(unit.unit_id for unit in units if unit.cell_id == cell),
            assigned_gpu_uuid=APPROVED_GPU_UUIDS[index % len(APPROVED_GPU_UUIDS)],
        )
        for index, cell in enumerate(EXPECTED_CELL_IDS)
    )
    return S27Plan("s207-test-plan", "s207/plan.json", frozen, cells, ("s206/matrix.json", "s206/mapping.json", "s204/g23.json"))


def _success(record: S27MappingUnit, plan: S27Plan) -> S27RawUnit:
    cell = next(item for item in plan.cells if item.cell_id == record.cell_id)
    return S27RawUnit(
        unit_id=record.unit_id,
        cell_id=record.cell_id,
        repetition_id=record.repetition_id,
        status="SUCCESS",
        attempt_id="attempt-0001",
        matrix_hash=plan.frozen_inputs.matrix_hash,
        mapping_hash=record.mapping_hash,
        sampling_plan_hash=plan.frozen_inputs.sampling_plan_hash,
        checkpoint_hash=cell.checkpoint_hash,
        reference_hash=cell.reference_hash,
        batch_size=record.batch_size,
        microbatch_count=record.microbatch_count,
        draw_ids=record.draw_ids,
        sample_ids=record.sample_ids,
        raw_artifact_ref=f"s207/raw/{record.unit_id}.json",
        raw_artifact_hash="f" * 64,
        metrics={"finite": True, "estimate": 0.0},
        methods=("raw", "double", "u_m2", "u_m16"),
        m2_identity_max_abs=0.0,
        mean_gradient_consistent=True,
        clamp_applied=False,
        clip_mode="none",
        cost={"valid": True, "wall_seconds": 1.0},
    )


def test_s27_preparation_requires_g24b_and_does_not_promote_fixture() -> None:
    gate = _gate_payload()
    matrix = _matrix_payload(str(gate["artifact_hash"]))
    mapping = _mapping_payload(matrix, str(gate["artifact_hash"]))
    with pytest.raises(S27PreparationBlocked):
        prepare_s27_plan(
            plan_id="s207-blocked",
            plan_ref="s207/blocked.json",
            matrix_ref="s206/matrix.json",
            matrix={**matrix, "status": "FIXTURE_FROZEN", "artifact_hash": canonical_json_hash({**matrix, "status": "FIXTURE_FROZEN"})},
            mapping_ref="s206/mapping.json",
            mapping=mapping,
            g24b_gate_ref="s206/g24b.json",
            g24b_gate=gate,
            checkpoints=_checkpoint_rows(),
            references=_reference_rows(),
            source_artifact_refs=("s206/matrix.json",),
            max_failure_fraction=0.0,
        )


def test_s27_strict_reducer_seals_only_complete_immutable_denominator(tmp_path: Path) -> None:
    plan = _plan_for_reducer()
    reducer = StrictG25Reducer(plan, run_id="s207-test-run")
    records = plan.frozen_inputs.units
    for record in records[:-1]:
        assert reducer.add(_success(record, plan)) is True
    with pytest.raises(S27G25Blocked, match="EXPECTED_UNITS_MISSING"):
        reducer.seal(tmp_path / "sealed")
    assert reducer.add(_success(records[-1], plan)) is True
    sealed = reducer.seal(tmp_path / "sealed", checked_at="2026-08-25T00:00:00+00:00")
    assert sealed["manifest"]["expected_unit_count"] == 1200
    assert sealed["manifest"]["failed_unit_count"] == 0
    gate = load_canonical_json(tmp_path / "sealed" / "g2.5-gate.json")
    assert GateRecord.from_mapping(gate).gate_id == "stage2.G2.5"
    with pytest.raises(S27G25Blocked):
        reducer.add(_success(records[-1], plan))


def test_s27_four_gpu_shards_cover_each_wave_without_overlap() -> None:
    plan = _plan_for_reducer()
    cell_id = EXPECTED_CELL_IDS[0]
    shards = partition_s27_units(plan, cell_id)
    expected = [unit.unit_id for unit in plan.frozen_inputs.units if unit.cell_id == cell_id]
    assert tuple(shard.gpu_uuid for shard in shards) == APPROVED_GPU_UUIDS
    assert tuple(unit_id for shard in shards for unit_id in shard.unit_ids) == tuple(expected)
    assert len({unit_id for shard in shards for unit_id in shard.unit_ids}) == len(expected)
    assert all(shard.shard_count == 4 for shard in shards)


def test_s27_sharded_merge_has_byte_stable_manifest_vs_unsharded_fixture() -> None:
    plan = _plan_for_reducer()
    cell_id = EXPECTED_CELL_IDS[0]
    records = [_success(unit, plan) for unit in plan.frozen_inputs.units]
    shards = partition_s27_units(plan, cell_id)
    by_id = {record.unit_id: record for record in records}
    unsharded = StrictG25Reducer(plan, run_id="s207-byte-stable")
    for record in records:
        unsharded.add(record)
    sharded = StrictG25Reducer(plan, run_id="s207-byte-stable")
    for shard in shards:
        for unit_id in shard.unit_ids:
            sharded.add(by_id[unit_id])
    for record in records:
        if record.cell_id != cell_id:
            sharded.add(record)
    assert unsharded._manifest() == sharded._manifest()
    assert canonical_json_hash(unsharded._manifest()) == canonical_json_hash(sharded._manifest())


def test_s27_status_recovery_and_gpu_allowlist_are_fail_closed(tmp_path: Path) -> None:
    inventory = [
        {"uuid": APPROVED_GPU_UUIDS[0], "pci_bus_id": "0000:53:00.0"},
        {"uuid": APPROVED_GPU_UUIDS[1], "pci_bus_id": "0000:9c:00.0"},
        {"uuid": APPROVED_GPU_UUIDS[2], "pci_bus_id": "0000:9d:00.0"},
        {"uuid": APPROVED_GPU_UUIDS[3], "pci_bus_id": "0000:a0:00.0"},
        {"uuid": EXCLUDED_GPU_UUID, "pci_bus_id": EXCLUDED_PCI},
    ]
    assert validate_gpu_inventory(inventory)["excluded_pci"] == EXCLUDED_PCI
    store = S27StatusStore(tmp_path / "status.json", run_id="s207-status", plan_hash="1" * 64)
    store.publish(S27DetachedStatus("s207-status", "1" * 64, "PREPARED", 0, "2026-08-25T00:00:00+00:00"))
    store.publish(S27DetachedStatus("s207-status", "1" * 64, "RUNNING", 0, "2026-08-25T00:00:01+00:00", os.getpid(), APPROVED_GPU_UUIDS[0], 1.0))
    assert store.require_recoverable().status == "RUNNING"
    tampered = dict(load_canonical_json(tmp_path / "status.json"))
    tampered["wave_index"] = 9
    write_canonical_json(tmp_path / "status.json", tampered)
    with pytest.raises(S27G25Blocked, match="STATUS_IDENTITY_OR_HASH_DRIFT"):
        store.load()
