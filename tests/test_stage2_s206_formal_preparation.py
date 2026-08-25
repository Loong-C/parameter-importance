from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    APPROVED_GPU_UUIDS,
    CELL_GPU_BINDINGS,
    EXCLUDED_PCI,
    BlindPilotMeasurement,
    S206PreflightSpec,
    S206PreparationBlocked,
    build_formal_confirmatory_mapping,
    build_formal_cell_specs,
    build_g24b_gate,
    build_global_pilot_mapping,
    qualify_formal_matrix,
    reduce_blinded_pilot,
    run_formal_pilot_cell,
    strict_preflight,
)
from param_importance_nlp.experiments.stage2_formal import _vector_digest
from param_importance_nlp.providers.synthetic import SyntheticGradientProvider
from param_importance_nlp.experiments.stage2_pilot import CostSemantics


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("s206-formal-test", tuple(range(1024))),
        {"reference_sizing": 1, "reference_A": 2, "reference_B": 3, "pilot": 4, "confirmatory": 5},
    )


def _execution() -> FormalExecutionEvidence:
    gates = tuple(
        GateRecord(
            gate_id=gate_id,
            stage=2,
            status=GateStatus.PASS,
            checked_at="2026-08-25T00:00:00+00:00",
            measured={"status": "PASS"},
            threshold={"status": "PASS"},
            evidence_refs=(f"evidence/{gate_id}.json",),
        )
        for gate_id in ("stage2.G2.2", "stage2.G2.3", "stage2.G2.4a")
    )
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=gates,
        metadata={"test": True},
    )


def _costs() -> CostSemantics:
    return CostSemantics(
        scientific_equal_sample_cost={"defined": True, "seconds": 1.0},
        isolated_estimator_cost={"defined": True, "seconds": 1.0},
        online_training_incremental_cost={"defined": True, "seconds": 1.0},
        cost_io_quiescent=True,
    )


def _measurements() -> list[BlindPilotMeasurement]:
    rows: list[BlindPilotMeasurement] = []
    for anchor_id in ANCHOR_IDS:
        for batch_size in (32, 64, 128, 256):
            for microbatch_count in (2, 4, 8, 16, 32):
                rows.append(
                    BlindPilotMeasurement(
                        anchor_id=anchor_id,
                        batch_size=batch_size,
                        microbatch_count=microbatch_count,
                        repetitions=50,
                        anchors_runnable=True,
                        finite=True,
                        state_unchanged=True,
                        m2_equivalent=True,
                        mean_gradient_consistent=True,
                        aggregation_overhead_ratio=0.1,
                        variance_by_endpoint={"bias": 0.0, "nmse": 0.0, "rank": 0.0},
                        delta_sci_by_endpoint={"bias": 0.1, "nmse": 0.1, "rank": 0.1},
                        reference_half_width_by_endpoint={"bias": 0.01, "nmse": 0.01, "rank": 0.01},
                        storage_bytes=1024,
                        gpu_hours=0.01,
                        resource_within_budget=True,
                        cost_io_quiescent=True,
                    )
                )
    return rows


def test_global_mapping_is_24_cells_and_disjoint() -> None:
    mapping = build_global_pilot_mapping(_sampling())
    assert len(mapping.cells) == 24
    assert mapping.total_draw_count == 144_000
    assert mapping.cells[0].stream_start == 0
    assert mapping.cells[-1].stream_end == 144_000
    assert len(mapping.pilot_draw_ids) == len(set(mapping.pilot_draw_ids))
    assert mapping.to_dict()["confirmatory_draws_generated"] is False


def test_formal_cell_bridge_executes_synthetic_slice_and_masks_science(tmp_path: Path) -> None:
    mapping = build_global_pilot_mapping(_sampling())
    execution = _execution()
    specs = build_formal_cell_specs(mapping, execution)
    assert len(specs) == 24
    assert specs[0].gpu_uuid == APPROVED_GPU_UUIDS[0]
    assert specs[-1].gpu_uuid == APPROVED_GPU_UUIDS[1]

    provider = SyntheticGradientProvider.from_location_scale(
        parameter_shapes={"p": (1,)},
        sample_count=1024,
        seed=17,
    )
    cell = mapping.cells[0]
    reference = provider.gradient(cell.mappings[0].draws).gradients
    run = run_formal_pilot_cell(
        cell,
        mapping=mapping,
        provider=provider,
        execution=execution,
        reference=reference,
        reference_hash=_vector_digest(reference),
        artifact_root=tmp_path / "cell-artifacts",
        delta_sci_by_endpoint={"bias": 0.1, "nmse": 0.1, "rank": 0.1},
        reference_half_width_by_endpoint={"bias": 0.01, "nmse": 0.01, "rank": 0.01},
        resource_within_budget=True,
        cost_io_quiescent=True,
    )
    assert len(run.measurements) == 5
    assert all(item.operational_ready for item in run.measurements)
    assert len({tuple(sorted(item.variance_by_endpoint.items())) for item in run.measurements}) == 5
    assert run.costs["scientific_equal_sample_cost"]["gradient_evaluations"] == 1600
    assert run.to_dict()["scientific_values_masked"] is True


def test_blinded_reducer_selects_operational_pair_only() -> None:
    mapping = build_global_pilot_mapping(_sampling())
    report = reduce_blinded_pilot(mapping, _measurements(), cost_semantics=_costs())
    assert report.status == "READY_FOR_QUALIFICATION"
    assert report.selected is not None
    assert (report.selected.batch_size, report.selected.microbatch_count, report.selected.r_required) == (32, 32, 200)
    wire = report.to_dict()
    assert wire["scientific_values_masked"] is True
    assert "bias_interval_low" not in str(wire)
    assert len(report.anchor_rows) == 120
    assert type(report.measurements[0]).from_mapping(report.measurements[0].to_dict()) == report.measurements[0]


def test_g24b_qualification_and_confirmatory_mapping_are_ordered() -> None:
    sampling = _sampling()
    mapping = build_global_pilot_mapping(sampling)
    report = reduce_blinded_pilot(mapping, _measurements(), cost_semantics=_costs())
    execution = _execution()
    gate = build_g24b_gate(report, execution, evidence_refs=("pilot/report.json", "pilot/mapping.json"))
    assert gate.status is GateStatus.PASS
    matrix = qualify_formal_matrix(report, execution, gate)
    assert matrix.formal_eligible is True
    assert matrix.qualification_gate_hash == gate.artifact_hash
    confirmatory = build_formal_confirmatory_mapping(
        matrix,
        sampling,
        pilot_draw_ids=mapping.pilot_draw_ids,
        gate=gate,
    )
    assert confirmatory.formal_eligible is True
    assert len(confirmatory.cells) == 6
    assert confirmatory.to_dict()["confirmatory_draw_id_count"] == 6 * 200 * 32
    assert not set(mapping.pilot_draw_ids).intersection(
        draw_id
        for cell in confirmatory.cells
        for draw_id in cell.draw_ids
    )


def test_confirmatory_mapping_rejects_non_g24b_gate() -> None:
    sampling = _sampling()
    mapping = build_global_pilot_mapping(sampling)
    report = reduce_blinded_pilot(mapping, _measurements(), cost_semantics=_costs())
    execution = _execution()
    gate = build_g24b_gate(report, execution, evidence_refs=("pilot/report.json",))
    matrix = qualify_formal_matrix(report, execution, gate)
    blocked = GateRecord(
        gate_id="stage2.G2.4b",
        stage=2,
        status=GateStatus.BLOCKED,
        checked_at="2026-08-25T00:00:00+00:00",
        measured={},
        threshold={},
        evidence_refs=("pilot/report.json",),
        reasons=("test",),
    )
    with pytest.raises(Exception):
        build_formal_confirmatory_mapping(matrix, sampling, pilot_draw_ids=mapping.pilot_draw_ids, gate=blocked)


def _write_upstream_fixture(root: Path) -> tuple[str, str, str]:
    s204 = root / "evidence" / "s204"
    for anchor_id in ANCHOR_IDS:
        component = anchor_id.replace(".", "__")
        write_canonical_json(
            s204 / component / "final-status.json",
            {
                "status": "COMPLETE",
                "formal_eligible": True,
                "artifact_refs": {
                    "reference_result": f"results/{component}/reference.json",
                    "reference_convergence_report": f"results/{component}/convergence.json",
                    "gate_record": f"results/{component}/gate.json",
                },
            },
        )
    g23 = {
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "status": "PASS",
        "formal_eligible": True,
        "required_cell_count": 6,
        "complete_cell_count": 6,
        "cells": [{"cell_id": anchor_id, "status": "PASS", "formal_eligible": True} for anchor_id in ANCHOR_IDS],
    }
    g23["artifact_hash"] = canonical_json_hash(g23)
    g23_ref = "evidence/g23.json"
    write_canonical_json(root / g23_ref, g23)
    g24a = {
        "schema_version": "stage2-g24a-formal-evaluation-v1",
        "gate_id": "stage2.G2.4a",
        "status": "PASS",
        "formal_eligible": True,
        "cell_count": 6,
        "results": [{"cell_id": anchor_id, "status": "PASS", "formal_eligible": True} for anchor_id in ANCHOR_IDS],
    }
    g24a["artifact_hash"] = canonical_json_hash(g24a)
    g24a_ref = "evidence/g24a.json"
    write_canonical_json(root / g24a_ref, g24a)
    return "evidence/s204", g23_ref, g24a_ref


def test_strict_preflight_requires_all_upstream_cells_and_approved_gpus(tmp_path: Path) -> None:
    s204_root, g23_ref, g24a_ref = _write_upstream_fixture(tmp_path)
    inventory = [{"uuid": uuid, "pci_bus_id": f"0000:{index + 53:02X}:00.0"} for index, uuid in enumerate(APPROVED_GPU_UUIDS)]
    result = strict_preflight(
        S206PreflightSpec(tmp_path, s204_root, g23_ref, g24a_ref),
        gpu_inventory=inventory,
    )
    assert result["status"] == "READY"
    assert result["confirmatory_draws_generated"] is False
    bad = list(inventory)
    bad[0] = {"uuid": bad[0]["uuid"], "pci_bus_id": EXCLUDED_PCI}
    with pytest.raises(S206PreparationBlocked, match="EXCLUDED"):
        strict_preflight(S206PreflightSpec(tmp_path, s204_root, g23_ref, g24a_ref), gpu_inventory=bad)
