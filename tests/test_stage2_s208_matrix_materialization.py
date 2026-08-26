from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.preregistration import (
    BATCH_SIZES,
    PRIMARY_CELLS,
    build_stage2_preregistration,
)
from param_importance_nlp.experiments.stage2_formal import (
    _BoundedCheckpointStore,
    _BoundedMoments,
    _bounded_moments_digest,
)
from param_importance_nlp.experiments.stage2_g23_contracts import WEIGHTING_CONTRACT_FIELDS
from param_importance_nlp.experiments.stage2_pilot import CostSemantics
from param_importance_nlp.experiments.stage2_s206_formal import (
    BlindPilotMeasurement,
    BlindedPilotReport,
)
from param_importance_nlp.experiments.stage2_s208_production import (
    S208ProductionBlocked,
    materialize_s208_matrix,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle


CELLS = tuple(f"{item['model']}:{item['stage']}" for item in PRIMARY_CELLS)
ANCHORS = tuple(item.replace(":", ".") for item in CELLS)
HASHES = tuple(f"{chr(ord('a') + index)}" * 64 for index in range(16))
WEIGHTING = {
    WEIGHTING_CONTRACT_FIELDS[0]: "equal_block_weight",
    WEIGHTING_CONTRACT_FIELDS[1]: "one_gradient_per_sample",
    WEIGHTING_CONTRACT_FIELDS[2]: "frozen_sampling_stream",
    WEIGHTING_CONTRACT_FIELDS[3]: True,
    WEIGHTING_CONTRACT_FIELDS[4]: True,
}


def _hashed(body: dict[str, object], *, key: str = "artifact_hash") -> dict[str, object]:
    body[key] = canonical_json_hash({name: value for name, value in body.items() if name != key})
    return body


def _write(root: Path, relative: str, body: dict[str, object], *, key: str = "artifact_hash") -> tuple[str, str]:
    payload = _hashed(body, key=key)
    write_canonical_json(root / relative, payload)
    return relative, str(payload[key])


def _moments(count: int) -> _BoundedMoments:
    result = _BoundedMoments()
    for value in range(1, count + 1):
        result.update_vector({"p": np.asarray([float(value)], dtype=np.float64)}, 1.0)
    return result


def _pilot_report() -> dict[str, object]:
    measurements = []
    for anchor in ANCHORS:
        for batch_size in BATCH_SIZES:
            for microbatch_count in (2, 4, 8, 16, 32):
                measurement = BlindPilotMeasurement(
                    anchor_id=anchor,
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
                measurements.append(measurement)
    report = BlindedPilotReport(
        report_id="s206-formal-blinded-pilot",
        mapping_hash="1" * 64,
        sampling_plan_hash="2" * 64,
        measurements=tuple(measurements),
        anchor_rows=(),
        candidate_evaluations=(),
        cost_semantics=CostSemantics(
            scientific_equal_sample_cost={"defined": True},
            isolated_estimator_cost={"defined": True},
            online_training_incremental_cost={"defined": True},
            cost_io_quiescent=True,
        ),
        status="READY_FOR_QUALIFICATION",
    )
    return report.to_dict()


@pytest.fixture(scope="module")
def formal_inputs_template(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    root = tmp_path_factory.mktemp("formal-inputs-template")
    prereg = build_stage2_preregistration(
        seed_plan_hash="3" * 64,
        producer_commit="4" * 40,
        mathematics_hash="5" * 64,
        stage1_report_hash="6" * 64,
        upstream_binding_hash="7" * 64,
        scope="formal",
    )
    prereg_store = TaskArtifactStore(root, "artifacts/s201")
    prereg_commit = prereg_store.publish(
        task_id="stage2.01_scope_hypotheses_and_preregistration",
        artifact_kind="preregistration",
        config_hash="8" * 64,
        run_intent="formal",
        payload=prereg,
        formal_eligible=True,
        source_refs=("refs/s201/source.json",),
    )
    prereg_ref = prereg_commit.commit_ref
    prereg_artifact_hash = prereg_commit.artifact_hash
    hypothesis_commit = prereg_store.publish(
        task_id="stage2.01_scope_hypotheses_and_preregistration",
        artifact_kind="hypothesis_contract",
        config_hash="8" * 64,
        run_intent="formal",
        payload=prereg,
        formal_eligible=True,
        source_refs=("refs/s201/source.json",),
    )
    hypothesis_ref = hypothesis_commit.commit_ref
    hypothesis_artifact_hash = hypothesis_commit.artifact_hash

    pilot = _pilot_report()

    references: dict[str, dict[str, object]] = {}
    index_rows = []
    manifest_rows = []
    rebind_rows = []
    g23_rows = []
    g24a_rows = []
    for index, cell_id in enumerate(CELLS):
        cell_root = f"refs/cell-{index}"
        checkpoint_id = f"checkpoint-{index}"
        model_id, training_stage = cell_id.split(":", 1)
        registry_hash = f"{chr(ord('b') + index)}" * 64
        registry_ref, _ = _write(root, f"{cell_root}/registry.json", {
            "schema_version": "stage2-parameter-registry-artifact-v1",
            "status": "READY",
            "scope": "formal",
            "cell_id": cell_id,
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "training_stage": training_stage,
            "registry_hash": registry_hash,
            "coordinate_ids": ["p"],
            "parameter_groups": {"p": {"layer": "layer-0", "module": "module-0"}},
        })
        plan_ref, plan_hash = _write(root, f"{cell_root}/sizing-plan.json", {
            "schema_version": "stage2-reference-sizing-plan-v1",
            "reference_id": f"reference-{index}",
            "candidate_sample_counts": [2, 4],
            "block_size": 1,
            "convergence_tolerance": 0.1,
            "required_consecutive": 1,
            "execution_evidence_hash": "c" * 64,
            "draw_start_position": 0,
            "draw_end_position_exclusive": 4,
            "require_terminal_convergence": True,
            "round_manifest_ref": f"{cell_root}/round-manifest.json",
            "final_stream_start_position": 0,
            "final_stream_end_position_exclusive": 4,
        })
        candidate_ref = f"{cell_root}/commits/reference_result.json"
        _write(root, candidate_ref, {
            "schema_version": "reference-result-v1",
            "reference_id": f"reference-{index}",
        })
        provider_state_digest = f"{chr(ord('d') + index)}" * 64
        sizing_draw_hash = f"{chr(ord('j') + index)}" * 64
        sizing_identity_hash = f"{chr(ord('p') + index)}" * 64
        candidate_states: dict[str, object] = {}
        selected_moments = None
        node_digests: dict[int, str] = {}
        for count in (2, 4):
            moment = _moments(count)
            candidate_states[str(count)] = {"a": moment.to_state()}
            node_digests[count] = canonical_json_hash({
                "checkpoint_schema": _BoundedCheckpointStore.schema_version,
                "plan_hash": plan_hash,
                "sample_count": count,
                "moments_hash": _bounded_moments_digest(moment),
            })
            if count == 4:
                selected_moments = moment
        assert selected_moments is not None
        checkpoint_payload = {
            "checkpoint_schema": _BoundedCheckpointStore.schema_version,
            "plan_hash": plan_hash,
            "provider_state_digest": provider_state_digest,
            "registry_hash": registry_hash,
            "weighting_assumptions": WEIGHTING,
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_identity_hash": sizing_identity_hash,
            "sizing_stream": True,
            "block_size": 1,
            "candidate_states": candidate_states,
        }
        publish_tensor_bundle(root / cell_root / "resume" / "reference-sizing" / "bounded-checkpoint", checkpoint_payload)
        selected_digest = canonical_json_hash({
            "checkpoint_schema": _BoundedCheckpointStore.schema_version,
            "plan_hash": plan_hash,
            "sample_count": 4,
            "moments_hash": _bounded_moments_digest(selected_moments),
        })
        convergence_ref = f"{cell_root}/convergence.json"
        convergence = {
            "schema_version": "stage2-reference-convergence-report-v1",
            "scope": "formal",
            "formal_eligible": True,
            "formula_contract": prereg,
            "formula_contract_hash": prereg_artifact_hash,
            "external_lineage": {"preregistration": {"commit_ref": prereg_ref, "artifact_hash": prereg_artifact_hash}},
            "sizing_plan": dict(json.loads((root / plan_ref).read_text(encoding="utf-8"))),
            "sizing_plan_artifact_hash": plan_hash,
            "selected_sample_count_per_stream": 4,
            "registry_identity": {"registry_hash": registry_hash},
            "one_shot_result": {"provider_state_digest": provider_state_digest, "weighting_assumptions": WEIGHTING},
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_identity_hash": sizing_identity_hash,
            "sizing_result_hash": "e" * 64,
        }
        _write(root, convergence_ref, convergence)
        delta_body = {
            "schema_version": "stage2-reference-delta-sci-v2",
            "source_kind": "reference_sizing_bounded_online",
            "cell_id": cell_id,
            "reference_id": f"reference-{index}",
            "sizing_plan_hash": plan_hash,
            "registry_hash": registry_hash,
            "formula_contract_hash": prereg_artifact_hash,
            "formula_version": "stage2-reference-sizing-margin-v1",
            "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
            "absolute_floors": dict(prereg["equivalence_and_precision"]["absolute_floors"]),
            "sizing_result_hash": "e" * 64,
            "candidate_sample_counts": [2, 4],
            "delta_sci_by_endpoint": {
                endpoint: {str(batch_size): 0.1 for batch_size in BATCH_SIZES}
                for endpoint in ("model_total", "layer", "module")
            },
            "signal_scale_by_endpoint": {
                endpoint: {str(batch_size): 10.0 for batch_size in BATCH_SIZES}
                for endpoint in ("model_total", "layer", "module")
            },
            "noise_scale_by_endpoint": {
                endpoint: {str(batch_size): 32.0 / batch_size for batch_size in BATCH_SIZES}
                for endpoint in ("model_total", "layer", "module")
            },
            "sizing_nodes": [{"sample_count": count, "state_digest": node_digests[count]} for count in (2, 4)],
        }
        delta_source_ref, delta_hash = _write(root, f"{cell_root}/delta-source.json", delta_body)
        convergence["candidate_delta_sci"] = dict(delta_body, artifact_hash=delta_hash, source_ref=delta_source_ref, source_hash=delta_hash, source_artifact_hash=delta_hash)
        convergence["candidate_delta_sci_source"] = delta_source_ref
        convergence["candidate_delta_sci_source_hash"] = delta_hash
        _write(root, convergence_ref, convergence)
        delta_plan_ref, _ = _write(root, f"{cell_root}/delta-plan.json", {
            "schema_version": "stage2-reference-delta-sci-plan-v1",
            "status": "READY",
            "scope": "formal",
            "phase": "pre_sizing",
            "cell_id": cell_id,
            "candidate_sample_counts": [2, 4],
            "formula_contract": dict(prereg["equivalence_and_precision"]),
            "formula_contract_hash": canonical_json_hash(dict(prereg["equivalence_and_precision"])),
            "source_contract_refs": [prereg_ref, hypothesis_ref],
            "source_contract_artifact_hashes": [prereg_artifact_hash, hypothesis_artifact_hash],
            "numeric_delta_source": "stage2-reference-sizing-raw-shards-v2-after-sizing-commit",
        })
        sidecar_body = {
            "schema_version": "stage2-g23-corrected-delta-sci-v1",
            "source_producer_schema_version": "stage2-reference-delta-sci-v2",
            "source_producer_ref": delta_source_ref,
            "source_producer_artifact_hash": delta_hash,
            "source_producer_table_mode": "candidate_batch_sizes",
            "source_producer_commit": "f" * 40,
            "evaluator_commit": "1" * 40,
            "evaluator_source_sha256": "2" * 64,
            "formula_contract_hash": prereg_artifact_hash,
            "formula_version": "stage2-reference-sizing-margin-v1",
            "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
            "absolute_floors": dict(prereg["equivalence_and_precision"]["absolute_floors"]),
            "reference_id": f"reference-{index}",
            "sizing_result_hash": "e" * 64,
            "sizing_plan_hash": plan_hash,
            "registry_hash": registry_hash,
            "candidate_sample_counts": [2, 4],
            "delta_sci_batch_sizes": list(BATCH_SIZES),
            "selected_sample_count_per_stream": 4,
            "delta_sci_by_endpoint": {endpoint: {str(batch_size): 0.1 for batch_size in BATCH_SIZES} for endpoint in ("model_total", "layer", "module")},
            "signal_scale_by_endpoint": {endpoint: {str(batch_size): 10.0 for batch_size in BATCH_SIZES} for endpoint in ("model_total", "layer", "module")},
            "noise_scale_by_endpoint": {endpoint: {str(batch_size): 32.0 / batch_size for batch_size in BATCH_SIZES} for endpoint in ("model_total", "layer", "module")},
            "sizing_nodes": [{"sample_count": count, "state_digest": node_digests[count]} for count in (2, 4)],
            "correction_reason": "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator",
        }
        sidecar_payload = _hashed(sidecar_body)
        sidecar_hash = str(sidecar_payload["artifact_hash"])
        sidecar_ref = f"refs/g23/g2.3-corrected-delta-sci/{sidecar_hash}.json"
        write_canonical_json(root / sidecar_ref, sidecar_payload)
        result_hash = ("a" + str(index)) * 32
        config_hash = ("b" + str(index)) * 32
        manifest_rows.append({
            "cell_id": cell_id,
            "checkpoint_id": checkpoint_id,
            "model_id": model_id,
            "training_stage": training_stage,
            "registry_hash": registry_hash,
        })
        references[cell_id] = {
            "candidate_ref": candidate_ref,
            "coordinate_ids": ["p"],
            "registry_hash": registry_hash,
        }
        index_rows.append({"cell_id": cell_id, "registry_ref": registry_ref, "sizing_ref": plan_ref, "delta_ref": delta_plan_ref})
        rebind_rows.append({
            "cell_id": cell_id,
            "result_hash": result_hash,
            "config_hash": config_hash,
            "reference_artifact_refs": {
                "reference_result": candidate_ref,
                "reference_convergence_report": convergence_ref,
                "gate_record": f"{cell_root}/gate.json",
            },
        })
        g23_rows.append({"cell_id": cell_id, "status": "PASS", "formal_eligible": True, "identities": {"result_hash": result_hash, "config_hash": config_hash, "producer_commit": "f" * 40, "corrected_delta_sci_ref": sidecar_ref, "corrected_delta_sci_hash": sidecar_hash}, "metrics": {"corrected_delta_sci_ref": sidecar_ref, "corrected_delta_sci_hash": sidecar_hash, "corrected_delta_sci_batch_sizes": list(BATCH_SIZES), "delta_sci_source": "g23_output_derived_corrected_sidecar", "signal_scale_by_endpoint": {endpoint: {str(batch_size): 10.0 for batch_size in BATCH_SIZES} for endpoint in ("model_total", "layer", "module")}, "noise_scale_by_endpoint": {endpoint: {str(batch_size): 32.0 / batch_size for batch_size in BATCH_SIZES} for endpoint in ("model_total", "layer", "module")}, "epsilon_num_by_endpoint": {"model_total": 0.02, "layer": 0.02, "module": 0.02}}})
        g24a_rows.append({
            "cell_id": cell_id,
            "status": "PASS",
            "formal_eligible": True,
            "metrics": {
                "h_ref_model_total": 0.01,
                "h_ref_layer": 0.01,
                "h_ref_module": 0.01,
                "epsilon_num_by_endpoint": {"model_total": 0.02, "layer": 0.02, "module": 0.02},
            },
        })
    binding_rows = [
        {
            "cell_id": cell_id,
            "config_hash": g23_rows[index]["identities"]["config_hash"],
            "result_hash": g23_rows[index]["identities"]["result_hash"],
            "corrected_delta_sci_hash": g23_rows[index]["identities"]["corrected_delta_sci_hash"],
            "corrected_delta_sci_ref": g23_rows[index]["identities"]["corrected_delta_sci_ref"],
            "corrected_delta_sci_batch_sizes": list(BATCH_SIZES),
            "delta_sci_source": "g23_output_derived_corrected_sidecar",
        }
        for index, cell_id in enumerate(CELLS)
    ]
    for measurement in pilot["measurements"]:
        assert isinstance(measurement, dict)
        cell_id = str(measurement["anchor_id"]).replace(".", ":")
        binding = next(item for item in binding_rows if item["cell_id"] == cell_id)
        measurement.update({
            "corrected_delta_sci_cell_id": binding["cell_id"],
            "corrected_delta_sci_config_hash": binding["config_hash"],
            "corrected_delta_sci_result_hash": binding["result_hash"],
            "corrected_delta_sci_hash": binding["corrected_delta_sci_hash"],
            "corrected_delta_sci_ref": binding["corrected_delta_sci_ref"],
            "corrected_delta_sci_batch_sizes": binding["corrected_delta_sci_batch_sizes"],
            "delta_sci_source": binding["delta_sci_source"],
        })
    pilot["corrected_delta_sci_bindings"] = binding_rows
    pilot_ref, pilot_hash = _write(root, "refs/s206/blinded-pilot-report.json", pilot)
    matrix = _hashed({
        "schema_version": "stage2-formal-pilot-matrix-freeze-v1",
        "freeze_id": "s206-formal-matrix-freeze",
        "pilot_report_hash": pilot_hash,
        "corrected_delta_sci_bindings": binding_rows,
        "pilot_mapping_hash": "1" * 64,
        "sampling_plan_hash": "2" * 64,
        "anchor_ids": list(ANCHORS),
        "candidate_evaluations": [],
        "b_primary": 32,
        "m_primary": 4,
        "r_primary": 200,
        "completion_denominator": 1200,
        "cost_semantics": pilot["cost_semantics"],
        "qualification_gate_hash": "9" * 64,
        "execution_evidence_hash": "a" * 64,
        "status": "FORMAL_FROZEN",
        "scope": "formal",
        "formal_eligible": True,
    })
    matrix_ref, matrix_hash = _write(root, "refs/s206/formal-matrix.json", matrix)
    manifest_ref, manifest_hash = _write(root, "refs/s204/six-cell-manifest.json", {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "scope": "formal",
        "status": "READY",
        "checkpoints": manifest_rows,
    }, key="manifest_hash")
    rebind_ref, rebind_hash = _write(root, "refs/s205/rebind-plan.json", {
        "schema_version": "stage2-s205-rebind-plan-v1",
        "status": "READY",
        "formal_eligible": True,
        "cells": rebind_rows,
    })
    g23_ref, g23_hash = _write(root, "refs/g23/g2.3-attempts/evaluation.json", {
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "gate_id": "stage2.G2.3",
        "status": "PASS",
        "formal_eligible": True,
        "required_cell_count": 6,
        "complete_cell_count": 6,
        "calculator": {"producer_commit": "f" * 40, "evaluator_commit": "1" * 40, "source_sha256": "2" * 64, "source_schema": "stage2-g23-reference-evaluation-v1"},
        "cells": g23_rows,
    })
    g24a_ref, g24a_hash = _write(root, "refs/g24a/evaluation.json", {
        "schema_version": "stage2-g24a-formal-evaluation-v1",
        "gate_id": "stage2.G2.4a",
        "status": "PASS",
        "formal_eligible": True,
        "cell_count": 6,
        "g23_evaluation_hash": g23_hash,
        "rebind_plan_ref": rebind_ref,
        "rebind_plan_hash": rebind_hash,
        "results": g24a_rows,
    })
    from param_importance_nlp.contracts.status import GateRecord, GateStatus

    g24b = GateRecord(
        gate_id="stage2.G2.4b",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-08-26T00:00:00Z",
        measured={"pilot_report_hash": pilot_hash, "corrected_delta_sci_bindings": binding_rows, "corrected_delta_sci_bindings_hash": canonical_json_hash({"bindings": binding_rows})},
        threshold={},
        evidence_refs=(pilot_ref,),
    ).to_dict()
    g24b_ref, g24b_hash = _write(root, "refs/g24b/gate.json", g24b)
    index_ref, index_hash = _write(root, "refs/s204/materialization-index.json", {
        "schema_version": "stage2-s204-six-cell-materialization-index-v1",
        "scope": "formal",
        "six_cell_manifest_ref": manifest_ref,
        "cells": index_rows,
    }, key="index_hash")
    return {
        "root": root,
        "matrix": matrix_ref,
        "matrix_hash": matrix_hash,
        "preregistration": prereg_ref,
        "g23": g23_ref,
        "g24a": g24a_ref,
        "g24b": g24b_ref,
        "pilot": pilot_ref,
        "index": index_ref,
        "references": {"cells": references},
        "hashes": {"g23": g23_hash, "g24a": g24a_hash, "g24b": g24b_hash, "index": index_hash, "manifest": manifest_hash},
    }


@pytest.fixture
def formal_inputs(formal_inputs_template: dict[str, object], tmp_path: Path) -> dict[str, object]:
    template_root = formal_inputs_template["root"]
    assert isinstance(template_root, Path)
    shutil.copytree(template_root, tmp_path, dirs_exist_ok=True)
    inputs = dict(formal_inputs_template)
    inputs["root"] = tmp_path
    return inputs


def _materialize(inputs: dict[str, object]) -> dict[str, object]:
    root = inputs["root"]
    assert isinstance(root, Path)
    return materialize_s208_matrix(
        root,
        inputs["index"],
        matrix=inputs["matrix"],
        preregistration=inputs["preregistration"],
        g23_gate=inputs["g23"],
        g24a_gate=inputs["g24a"],
        g24b_gate=inputs["g24b"],
        references=inputs["references"],
    )


def test_real_s206_payload_materializes_all_cells_from_bound_sources(formal_inputs: dict[str, object]) -> None:
    output = _materialize(formal_inputs)
    assert output["matrix_hash"] == formal_inputs["matrix_hash"]
    assert output["formula_contract_hash"] != output["preregistration_hash"]
    assert tuple(output["cells"]) == CELLS
    assert all(item["nmse_denominator"] == pytest.approx(39.0625) for item in output["cells"].values())
    assert all(item["layer_group_registry"] != item["module_group_registry"] for item in output["cells"].values())
    assert all(item["source_refs"]["corrected_delta_sci_ref"].endswith("g2.3-corrected-delta-sci/" + item["source_hashes"]["corrected_delta_sci_hash"] + ".json") for item in output["cells"].values())
    assert output["artifact_hash"] == canonical_json_hash({key: value for key, value in output.items() if key != "artifact_hash"})


def test_task_artifact_formula_tamper_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    prereg_commit = root / "artifacts/s201/commits/preregistration.json"
    payload = __import__("json").loads(prereg_commit.read_text(encoding="utf-8"))
    payload["artifact_hash"] = "f" * 64
    write_canonical_json(prereg_commit, payload)
    with pytest.raises(S208ProductionBlocked, match="preregistration"):
        _materialize(formal_inputs)


def test_missing_bounded_checkpoint_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    checkpoint = root / "refs/cell-0/resume/reference-sizing/bounded-checkpoint"
    import shutil

    shutil.rmtree(checkpoint)
    with pytest.raises(S208ProductionBlocked, match="BOUNDED_CHECKPOINT"):
        _materialize(formal_inputs)


def test_pilot_delta_mismatch_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    pilot_path = root / "refs/s206/blinded-pilot-report.json"
    payload = __import__("json").loads(pilot_path.read_text(encoding="utf-8"))
    payload["measurements"][0]["delta_sci_by_endpoint"]["bias"] = 0.2
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(pilot_path, payload)
    with pytest.raises(S208ProductionBlocked, match="PILOT_REPORT|MATRIX_HASH"):
        _materialize(formal_inputs)


def test_corrected_delta_sidecar_tamper_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    gate = json.loads((root / str(formal_inputs["g23"])).read_text(encoding="utf-8"))
    sidecar_ref = gate["cells"][0]["identities"]["corrected_delta_sci_ref"]
    sidecar_path = root / sidecar_ref
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    payload["delta_sci_by_endpoint"]["model_total"]["32"] = 0.2
    write_canonical_json(sidecar_path, payload)
    with pytest.raises(S208ProductionBlocked, match="CORRECTED_DELTA|corrected_delta"):
        _materialize(formal_inputs)


def test_corrected_delta_cell_binding_mismatch_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    gate_path = root / str(formal_inputs["g23"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["cells"][0]["identities"]["corrected_delta_sci_hash"] = "f" * 64
    gate["artifact_hash"] = canonical_json_hash({key: value for key, value in gate.items() if key != "artifact_hash"})
    write_canonical_json(gate_path, gate)
    with pytest.raises(S208ProductionBlocked, match="CORRECTED_DELTA|g23_gate|G23"):
        _materialize(formal_inputs)


def test_g24b_corrected_delta_binding_swap_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    gate_path = root / str(formal_inputs["g24b"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    measured = gate["measured"]
    measured["corrected_delta_sci_bindings"][0], measured["corrected_delta_sci_bindings"][1] = (
        measured["corrected_delta_sci_bindings"][1],
        measured["corrected_delta_sci_bindings"][0],
    )
    measured["corrected_delta_sci_bindings_hash"] = canonical_json_hash({"bindings": measured["corrected_delta_sci_bindings"]})
    gate["artifact_hash"] = canonical_json_hash({key: value for key, value in gate.items() if key != "artifact_hash"})
    write_canonical_json(gate_path, gate)
    with pytest.raises(S208ProductionBlocked, match="SIX_CELL_BINDING_ORDER_REQUIRED|BINDING"):
        _materialize(formal_inputs)


def test_g24b_corrected_delta_binding_missing_field_is_blocked(formal_inputs: dict[str, object]) -> None:
    root = formal_inputs["root"]
    assert isinstance(root, Path)
    gate_path = root / str(formal_inputs["g24b"])
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    del gate["measured"]["corrected_delta_sci_bindings"][0]["result_hash"]
    gate["measured"]["corrected_delta_sci_bindings_hash"] = canonical_json_hash({"bindings": gate["measured"]["corrected_delta_sci_bindings"]})
    gate["artifact_hash"] = canonical_json_hash({key: value for key, value in gate.items() if key != "artifact_hash"})
    write_canonical_json(gate_path, gate)
    with pytest.raises(S208ProductionBlocked, match="BINDING_FIELDS_INVALID|BINDING"):
        _materialize(formal_inputs)
