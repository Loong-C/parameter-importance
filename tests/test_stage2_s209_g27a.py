from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s207_formal import APPROVED_GPU_UUIDS
from param_importance_nlp.experiments.stage2_s209_g27a import (
    S29_COST_SEMANTICS,
    S29_SHARED_POOL_SCHEMA,
    S29_SHARED_RUN_SCHEMA,
    S29G27ABlocked,
    StrictS29Reducer,
    bind_s209_inputs,
    run_s209_g27a,
    shared_paired_run_identity,
)

INVENTORY_ARTIFACT_HASH = "1" * 64
INVENTORY_SOURCE_SHA256 = "2" * 64
MEASUREMENT_PLAN_HASH = "d" * 64


def _gate() -> dict[str, object]:
    return GateRecord(
        gate_id="stage2.G2.4b",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={"selected": True},
        threshold={},
        evidence_refs=("s206/matrix.json",),
    ).to_dict()


def _inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    gate = _gate()
    matrix: dict[str, object] = {
        "schema_version": "stage2-formal-pilot-matrix-freeze-v1",
        "freeze_id": "s206-formal-matrix",
        "scope": "formal",
        "status": "FORMAL_FROZEN",
        "anchor_ids": list(EXPECTED_CELL_IDS),
        "candidate_evaluations": [{"batch_size": 32, "microbatch_count": 16, "selected": True}],
        "b_primary": 32,
        "m_primary": 16,
        "r_primary": 1,
        "completion_denominator": 6,
        "cost_semantics": list(S29_COST_SEMANTICS),
        "cost_observations": {name: {"defined": True} for name in S29_COST_SEMANTICS},
        "cost_io_quiescent": True,
        "confirmatory_draw_stream": "confirmatory",
        "pilot_draw_stream": "pilot",
        "formal_eligible": True,
        "qualification_gate_hash": gate["artifact_hash"],
    }
    matrix["artifact_hash"] = canonical_json_hash(matrix)
    units = [
        {
            "unit_id": f"{cell}::rep-0",
            "cell_id": cell,
            "repetition_id": "rep-0",
            "status": "SUCCESS",
            "attempt_id": "attempt-1",
            "draw_id_hash": "1" * 64,
            "raw_artifact_ref": f"raw/{index}.json",
            "raw_artifact_hash": f"{index + 2:064x}",
            "unit_artifact_hash": f"{index + 10:064x}",
        }
        for index, cell in enumerate(EXPECTED_CELL_IDS)
    ]
    manifest: dict[str, object] = {
        "schema_version": "stage2-s27-sealed-raw-manifest-v1",
        "run_id": "s207-formal-run",
        "task_id": "stage2.07_main_sweep",
        "scope": "formal",
        "stream": "confirmatory",
        "status": "SEALED",
        "formal_eligible": True,
        "plan_ref": "s207/plan.json",
        "plan_hash": "a" * 64,
        "matrix_ref": "s206/matrix.json",
        "matrix_hash": matrix["artifact_hash"],
        "mapping_ref": "s206/mapping.json",
        "mapping_hash": "b" * 64,
        "sampling_plan_hash": "c" * 64,
        "expected_unit_count": len(units),
        "completed_unit_count": len(units),
        "failed_unit_count": 0,
        "failure_fraction": 0.0,
        "max_failure_fraction": 0.0,
        "cell_order": list(EXPECTED_CELL_IDS),
        "units": units,
    }
    manifest["artifact_hash"] = canonical_json_hash(manifest)
    return matrix, gate, manifest


def _row(method: str, *, semantic: str, anchor: str, repetition: int = 0, wall: float = 10.0, gpu: str = APPROVED_GPU_UUIDS[0], io: bool = True, run_id: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "method": method,
        "semantic": semantic,
        "run_id": run_id or f"s209-{semantic[:4]}",
        "source_raw_run_id": "s207-formal-run",
        "anchor_id": anchor,
        "repetition": repetition,
        "gpu_uuid": gpu,
        "inventory_artifact_hash": INVENTORY_ARTIFACT_HASH,
        "inventory_source_sha256": INVENTORY_SOURCE_SHA256,
        "device_count": 1,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "communication_bytes": 0,
        "output_bytes": 4096,
        "data_wait_seconds": 0.5,
        "forward_seconds": 2.0,
        "backward_seconds": 3.0,
        "gradient_aggregation_seconds": 0.5,
        "formula_seconds": 0.5 if method != "raw" else 0.1,
        "statistics_seconds": 0.5,
        "communication_seconds": 0.0,
        "write_seconds": 1.0,
        "wall_seconds": wall,
        "allocated_peak_bytes": 100,
        "reserved_peak_bytes": 120,
        "device_peak_bytes": 140,
        "cost_io_quiescent": io,
        "health_ok": True,
        "batch_size": 32,
        "microbatch_count": 16,
    }
    return value


def _shared_row(method: str, *, anchor: str = "shared", repetition: int = 0, wall: float = 10.0, gpu: str = APPROVED_GPU_UUIDS[0], io: bool = True, run_id: str = "s209-g27a") -> dict[str, object]:
    value = _row(method, semantic=S29_COST_SEMANTICS[0], anchor=anchor, repetition=repetition, wall=wall, gpu=gpu, io=io, run_id=run_id)
    matrix, _gate, manifest = _inputs()
    order = ["double", "raw", "u"]
    identity = shared_paired_run_identity(
        run_id=str(value["run_id"]),
        measurement_plan_hash=MEASUREMENT_PLAN_HASH,
        matrix_hash=str(matrix["artifact_hash"]),
        raw_manifest_hash=str(manifest["artifact_hash"]),
        source_raw_run_id="s207-formal-run",
        anchor_id=anchor,
        repetition=repetition,
        gpu_uuid=gpu,
        device_count=1,
        method_order=order,
    )
    value.update(
        {
            "paired_run_id": identity["paired_run_id"],
            "paired_run_identity_hash": identity["paired_run_identity_hash"],
            "measurement_plan_hash": MEASUREMENT_PLAN_HASH,
            "shared_pool_id": "3" * 64,
            "shared_pool_artifact_hash": "4" * 64,
            "shared_pool_ref": f"shared-pools/{identity['paired_run_id']}.json",
            "shared_sample_mapping_hash": "5" * 64,
            "shared_gradient_pool_hash": "6" * 64,
            "shared_method_order": order,
            "shared_method_index": order.index(method),
            "shared_sample_sequence_count": 32,
            "shared_sample_token_count": 1024,
        }
    )
    return value


def _costs(*, io: bool = True, run_id: str = "s209-g27a") -> dict[str, object]:
    scientific = [_shared_row(method, anchor="shared", wall=10.0, io=io, run_id=run_id) for method in ("raw", "double", "u")]
    shared_manifest = [
        {
            field: (list(scientific[0][field]) if field == "shared_method_order" else scientific[0][field])
            for field in (
                "paired_run_id", "paired_run_identity_hash", "measurement_plan_hash", "anchor_id", "repetition", "gpu_uuid", "device_count", "shared_pool_id", "shared_pool_artifact_hash", "shared_pool_ref", "shared_sample_mapping_hash", "shared_gradient_pool_hash", "shared_method_order", "shared_sample_sequence_count", "shared_sample_token_count"
            )
        }
    ]
    isolated = [_row(method, semantic=S29_COST_SEMANTICS[1], anchor="isolated", wall=11.0, io=io, run_id=run_id) for method in ("raw", "double", "u")]
    online: list[dict[str, object]] = []
    for repetition, anchor in enumerate(("online-a", "online-b")):
        # The order is intentionally not method-name order; metadata carries
        # the frozen randomization proof rather than relying on wall time.
        for method, wall in (("double", 11.0), ("raw", 10.0), ("u", 12.0)):
            online.append(_row(method, semantic=S29_COST_SEMANTICS[2], anchor=anchor, repetition=repetition, wall=wall, io=io, run_id=run_id))
    return {
        S29_COST_SEMANTICS[0]: {"defined": True, "shared_paired_runner_schema": S29_SHARED_RUN_SCHEMA, "shared_pool_schema": S29_SHARED_POOL_SCHEMA, "shared_paired_runs": shared_manifest, "observations": scientific},
        S29_COST_SEMANTICS[1]: {"defined": True, "observations": isolated},
        S29_COST_SEMANTICS[2]: {"defined": True, "randomized_method_order": True, "randomization_seed": 29, "decision_ratio_threshold": 1.25, "observations": online},
    }


def _health() -> dict[str, object]:
    return {
        "healthy": True,
        "idle": True,
        "same_gpu_class": True,
        "gpu_class": "A100-80GB",
        "gpu_uuids": list(APPROVED_GPU_UUIDS),
        "ecc_errors": 0,
        "xid_errors": 0,
        "inventory_artifact_hash": INVENTORY_ARTIFACT_HASH,
        "inventory_source_sha256": INVENTORY_SOURCE_SHA256,
        "cost_io_quiescent": True,
    }


def _anchor(device_count: int, uuids: list[str]) -> dict[str, object]:
    return {
        "status": "PASS",
        "matrix_hash": _inputs()[0]["artifact_hash"],
        "source_raw_run_id": "s207-formal-run",
        "device_count": device_count,
        "gpu_uuids": uuids,
        "cost_io_quiescent": True,
        "health_ok": True,
        "numeric_consistency": True,
        "batch_size": 32,
        "microbatch_count": 16,
        "sequence_count": 32,
        "token_count": 1024,
        "backward_count": 1,
        "communication_bytes": 0,
        "output_bytes": 4096,
    }


def _accuracy() -> list[dict[str, object]]:
    return [
        {"row_id": f"{cell}:{method}", "cell_id": cell, "method": method, "corrected_nmse": 1.0, "mse": 1.0, "spearman": 0.9, "overlap_1pct": 0.8, "token_count": 1024}
        for cell in EXPECTED_CELL_IDS
        for method in ("raw", "double", "u")
    ]


def test_s209_binds_frozen_matrix_and_sealed_raw_manifest() -> None:
    matrix, gate, manifest = _inputs()
    frozen = bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=manifest)
    assert frozen.matrix_hash == matrix["artifact_hash"]
    assert frozen.raw_manifest_hash == manifest["artifact_hash"]
    assert frozen.raw_run_id == "s207-formal-run"
    with pytest.raises(S29G27ABlocked, match="S27_MATRIX_HASH_MISMATCH"):
        tampered = {**manifest, "matrix_hash": "d" * 64}
        tampered["artifact_hash"] = canonical_json_hash({key: value for key, value in tampered.items() if key != "artifact_hash"})
        bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=tampered)


def test_s209_binds_and_verifies_g25_gate_identity() -> None:
    matrix, gate, manifest = _inputs()
    g25 = GateRecord(
        gate_id="stage2.G2.5",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured={"raw_manifest_hash": manifest["artifact_hash"]},
        threshold={},
        evidence_refs=("s207/g2.5-gate.json",),
    ).to_dict()
    frozen = bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=manifest, g25_gate=g25, require_g25=True)
    assert frozen.g25_gate_hash == g25["artifact_hash"]
    tampered = dict(g25)
    tampered["measured"] = {"raw_manifest_hash": "f" * 64}
    tampered["artifact_hash"] = GateRecord(
        gate_id="stage2.G2.5",
        stage=2,
        status="PASS",
        checked_at="2026-08-25T00:00:00+00:00",
        measured=tampered["measured"],
        threshold={},
        evidence_refs=("s207/g2.5-gate.json",),
    ).artifact_hash
    with pytest.raises(S29G27ABlocked, match="G25_RAW_MANIFEST_HASH_MISMATCH"):
        bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=manifest, g25_gate=tampered, require_g25=True)
    with pytest.raises(S29G27ABlocked, match="G25_GATE_REQUIRED"):
        bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=manifest, require_g25=True)


def test_s209_reducer_is_strict_and_idempotent() -> None:
    matrix, gate, manifest = _inputs()
    frozen = bind_s209_inputs(matrix=matrix, g24b_gate=gate, raw_manifest=manifest)
    reducer = StrictS29Reducer(frozen, run_id="s209-test")
    row = _shared_row("raw", anchor="shared", run_id="s209-test")
    assert reducer.add(row, semantic=S29_COST_SEMANTICS[0]) is True
    assert reducer.add(row, semantic=S29_COST_SEMANTICS[0]) is False
    with pytest.raises(S29G27ABlocked, match="SEMANTIC_MIXED"):
        reducer.add({**row, "semantic": S29_COST_SEMANTICS[1]}, semantic=S29_COST_SEMANTICS[0])
    with pytest.raises(S29G27ABlocked, match="SHARED_ANCHOR_REQUIRED"):
        reducer.add({**row, "anchor_kind": "method_only"}, semantic=S29_COST_SEMANTICS[0])


def test_s209_missing_four_card_or_nonquiet_cost_never_passes(tmp_path: Path) -> None:
    matrix, gate, manifest = _inputs()
    common = {
        "matrix": matrix,
        "g24b_gate": gate,
        "raw_manifest": manifest,
        "health_snapshot": _health(),
        "single_gpu_anchor": _anchor(1, [APPROVED_GPU_UUIDS[0]]),
        "four_gpu_anchor": None,
        "shared_attribution_cross_check": {method: {"shared_wall_seconds": 10.0, "isolated_wall_seconds": 11.0, "relative_difference": 1 / 11} for method in ("raw", "double", "u")},
        "accuracy_rows": _accuracy(),
        "output_root": tmp_path / "blocked",
    }
    result = run_s209_g27a(**common, cost_observations=_costs())
    assert result["status"] in {"BLOCKED", "PROVISIONAL"}
    assert result["gate"]["status"] != "PASS"
    quiet_result = run_s209_g27a(**{**common, "four_gpu_anchor": _anchor(4, list(APPROVED_GPU_UUIDS)), "output_root": tmp_path / "nonquiet"}, cost_observations=_costs(io=False), cost_io_quiescent=False)
    assert quiet_result["status"] in {"BLOCKED", "PROVISIONAL"}
    assert quiet_result["gate"]["status"] != "PASS"


def test_s209_complete_fixture_emits_pass_pareto_capacity(tmp_path: Path) -> None:
    matrix, gate, manifest = _inputs()
    result = run_s209_g27a(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=manifest,
        cost_observations=_costs(),
        health_snapshot=_health(),
        single_gpu_anchor=_anchor(1, [APPROVED_GPU_UUIDS[0]]),
        four_gpu_anchor=_anchor(4, list(APPROVED_GPU_UUIDS)),
        shared_attribution_cross_check={method: {"shared_wall_seconds": 10.0, "isolated_wall_seconds": 11.0, "relative_difference": 1 / 11} for method in ("raw", "double", "u")},
        accuracy_rows=_accuracy(),
        capacity_inputs={
            "stage4_steps": 10,
            "stage5_steps": 20,
            "disk_free_bytes": 10**12,
            "inode_free": 10**6,
            "capacity_evidence_hash": "3" * 64,
            "ulimit_evidence_hash": "4" * 64,
            "ulimit_nofile_soft": 1024,
        },
        output_root=tmp_path / "pass",
        checked_at="2026-08-25T00:00:00+00:00",
    )
    assert result["status"] == "PASS"
    assert result["gate"]["status"] == "PASS"
    assert result["online_training_incremental_cost"]["ratios"]["source"] == "online_training_incremental_cost"
    assert result["pareto"]["status"] == "PASS"
    assert result["capacity"]["forecasts"]["u"]["projected_total_a100_hours"] > 0
    assert (tmp_path / "pass" / "g2.7a-gate.json").exists()


def test_s209_g27a_shared_pool_identity_drift_blocks_gate(tmp_path: Path) -> None:
    matrix, gate, manifest = _inputs()
    costs = _costs()
    scientific = costs[S29_COST_SEMANTICS[0]]["observations"]
    scientific[1]["shared_pool_artifact_hash"] = "7" * 64
    result = run_s209_g27a(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=manifest,
        cost_observations=costs,
        health_snapshot=_health(),
        single_gpu_anchor=_anchor(1, [APPROVED_GPU_UUIDS[0]]),
        four_gpu_anchor=_anchor(4, list(APPROVED_GPU_UUIDS)),
        shared_attribution_cross_check={method: {"shared_wall_seconds": 10.0, "isolated_wall_seconds": 11.0, "relative_difference": 1 / 11} for method in ("raw", "double", "u")},
        accuracy_rows=_accuracy(),
        capacity_inputs={
            "stage4_steps": 10,
            "stage5_steps": 20,
            "disk_free_bytes": 10**12,
            "inode_free": 10**6,
            "capacity_evidence_hash": "3" * 64,
            "ulimit_evidence_hash": "4" * 64,
            "ulimit_nofile_soft": 1024,
        },
        output_root=tmp_path / "tampered",
    )
    assert result["status"] == "BLOCKED"
    assert any(reason.startswith("SHARED_PAIRED_IDENTITY_DRIFT") for reason in result["reasons"])


def test_s209_g27a_shared_manifest_duplicate_blocks_gate(tmp_path: Path) -> None:
    matrix, gate, manifest = _inputs()
    costs = _costs()
    shared = costs[S29_COST_SEMANTICS[0]]
    shared["shared_paired_runs"] = list(shared["shared_paired_runs"]) + [dict(shared["shared_paired_runs"][0])]
    result = run_s209_g27a(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=manifest,
        cost_observations=costs,
        health_snapshot=_health(),
        single_gpu_anchor=_anchor(1, [APPROVED_GPU_UUIDS[0]]),
        four_gpu_anchor=_anchor(4, list(APPROVED_GPU_UUIDS)),
        shared_attribution_cross_check={method: {"shared_wall_seconds": 10.0, "isolated_wall_seconds": 11.0, "relative_difference": 1 / 11} for method in ("raw", "double", "u")},
        accuracy_rows=_accuracy(),
        capacity_inputs={
            "stage4_steps": 10,
            "stage5_steps": 20,
            "disk_free_bytes": 10**12,
            "inode_free": 10**6,
            "capacity_evidence_hash": "3" * 64,
            "ulimit_evidence_hash": "4" * 64,
            "ulimit_nofile_soft": 1024,
        },
        output_root=tmp_path / "duplicate",
    )
    assert result["status"] == "BLOCKED"
    assert "SHARED_PAIRED_RUN_MANIFEST_DUPLICATE" in result["reasons"]


def test_s209_missing_capacity_or_ulimit_evidence_cannot_pass(tmp_path: Path) -> None:
    matrix, gate, manifest = _inputs()
    with pytest.raises(S29G27ABlocked, match="COST_SEMANTICS"):
        # A malformed/empty cost payload remains rejected before capacity
        # reduction; this guards the fail-closed entry boundary.
        run_s209_g27a(
            matrix=matrix,
            g24b_gate=gate,
            raw_manifest=manifest,
            cost_observations={},
            health_snapshot=_health(),
            single_gpu_anchor=_anchor(1, [APPROVED_GPU_UUIDS[0]]),
            four_gpu_anchor=_anchor(4, list(APPROVED_GPU_UUIDS)),
            shared_attribution_cross_check={},
        )
    result = run_s209_g27a(
        matrix=matrix,
        g24b_gate=gate,
        raw_manifest=manifest,
        cost_observations=_costs(),
        health_snapshot=_health(),
        single_gpu_anchor=_anchor(1, [APPROVED_GPU_UUIDS[0]]),
        four_gpu_anchor=_anchor(4, list(APPROVED_GPU_UUIDS)),
        shared_attribution_cross_check={method: {"shared_wall_seconds": 10.0, "isolated_wall_seconds": 11.0, "relative_difference": 1 / 11} for method in ("raw", "double", "u")},
        accuracy_rows=_accuracy(),
        capacity_inputs=None,
        output_root=tmp_path / "missing-capacity",
    )
    assert result["status"] == "BLOCKED"
    assert "CAPACITY_INPUTS_REQUIRED" in result["reasons"]
