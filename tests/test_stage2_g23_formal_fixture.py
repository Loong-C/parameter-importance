"""A small, exact six-cell formal fixture for the independent G2.3 consumer."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_formal import _ReferenceSnapshotStore, _vector_digest
from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    EXPECTED_CELL_IDS,
    evaluate_formal_g23,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode("utf-8")).hexdigest()


def _hashed(value: dict[str, object]) -> dict[str, object]:
    return dict(value, artifact_hash=canonical_json_hash(value))


def _identity(value: dict[str, object]) -> dict[str, object]:
    return dict(value, identity_hash=canonical_json_hash(value))


def _moments(blocks: list[dict[str, np.ndarray]], weights: list[float]) -> dict[str, object]:
    names = tuple(sorted(blocks[0])) if blocks else ()
    return {
        "count": len(blocks),
        "n1": float(sum(weights)),
        "n2": float(sum(value * value for value in weights)),
        "g1": {name: sum((weight * block[name] for weight, block in zip(weights, blocks)), np.zeros_like(blocks[0][name])) for name in names},
        "g2": {name: sum((weight * weight * np.square(block[name]) for weight, block in zip(weights, blocks)), np.zeros_like(blocks[0][name])) for name in names},
    }


def _publish_progress(root: Path, output_dir: str, *, plan_hash: str, final_plan_hash: str, sizing_result_hash: str, provider_hash: str, registry_hash: str, sizing_draw_hash: str, sizing_identity_hash: str, stream_a_hash: str, stream_b_hash: str, sizing_result_identity_hash: str) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    assumptions = {
        "statistical_unit": "block",
        "weight_unit": "sequence",
        "sampling_design": "uniform_with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    block_a = {"p0": np.asarray([10.0, 10.0]), "p1": np.asarray([20.0, 20.0])}
    block_b = {"p0": np.asarray([10.0, 10.0]), "p1": np.asarray([20.0, 20.0])}
    blocks_a = [dict(block_a) for _ in range(4)]
    blocks_b = [dict(block_b) for _ in range(4)]
    weights = [1.0] * 4
    empty = {"count": 0, "n1": 0.0, "n2": 0.0, "g1": {}, "g2": {}}

    def publish_state(resume: Path, sequence: int, state: dict[str, object]) -> dict[str, object]:
        digest = _ReferenceSnapshotStore._state_digest(state)
        bundle = publish_tensor_bundle(resume / "objects" / digest, state)
        commit = {
            "schema_version": "stage2-reference-progress-commit-v1",
            "sequence": sequence,
            "state_digest": digest,
            "object_ref": f"objects/{digest}",
            "object_manifest_hash": bundle.manifest_sha256,
        }
        commit["artifact_hash"] = canonical_json_hash(commit)
        write_canonical_json(resume / "commits" / f"{sequence:08d}.json", commit)
        return commit

    sizing = root / output_dir / "resume" / "reference-sizing"
    final = root / output_dir / "resume" / "reference-final"
    sizing.mkdir(parents=True)
    final.mkdir(parents=True)
    sizing_latest: dict[str, object] = {}
    for sequence in (1, 2, 3, 4):
        state = {
            "schema_version": "stage2-reference-progress-state-v1",
            "plan_hash": plan_hash,
            "provider_state_digest": provider_hash,
            "registry_hash": registry_hash,
            "weighting_assumptions": assumptions,
            "processed_block_pairs": sequence,
            "convergence_streak": 0,
            "selected_sample_count_per_stream": None,
            "points": [],
            "last_bias": {},
            "a": _moments(blocks_a[:sequence], weights[:sequence]),
            "b": empty,
            "blocks_a": blocks_a[:sequence],
            "blocks_b": [{name: np.zeros_like(value) for name, value in block_a.items()} for _ in range(sequence)],
            "block_weights_a": weights[:sequence],
            "block_weights_b": [0.0] * sequence,
            "sizing_stream": True,
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_identity_hash": sizing_identity_hash,
            "rng_state_digest": sizing_draw_hash,
        }
        sizing_latest = publish_state(sizing, sequence, state)

    final_state = {
        "schema_version": "stage2-reference-one-shot-progress-v1",
        "plan_hash": final_plan_hash,
        "sizing_result_hash": sizing_result_hash,
        "provider_state_digest": provider_hash,
        "registry_hash": registry_hash,
        "weighting_assumptions": assumptions,
        "stream_a_draw_hash": stream_a_hash,
        "stream_b_draw_hash": stream_b_hash,
        "processed_block_pairs": 4,
        "a": _moments(blocks_a, weights),
        "b": _moments(blocks_b, weights),
        "blocks_a": blocks_a,
        "blocks_b": blocks_b,
        "block_weights_a": weights,
        "block_weights_b": weights,
        "final_length_required": True,
        "sizing_result_identity_hash": sizing_result_identity_hash,
        "rng_state_digest": canonical_json_hash({"stream_a_draw_hash": stream_a_hash, "stream_b_draw_hash": stream_b_hash, "sizing_result_hash": sizing_result_hash}),
    }
    final_commit: dict[str, object] = {}
    for sequence in (1, 2, 3, 4):
        prefix = dict(final_state)
        prefix["processed_block_pairs"] = sequence
        prefix["a"] = _moments(blocks_a[:sequence], weights[:sequence])
        prefix["b"] = _moments(blocks_b[:sequence], weights[:sequence])
        prefix["blocks_a"] = blocks_a[:sequence]
        prefix["blocks_b"] = blocks_b[:sequence]
        prefix["block_weights_a"] = weights[:sequence]
        prefix["block_weights_b"] = weights[:sequence]
        final_commit = publish_state(final, sequence, prefix)
    return sizing_latest, final_commit, {"blocks_a": blocks_a, "blocks_b": blocks_b}


def _build_formal_fixture(root: Path) -> tuple[list[CellInput], list[Path]]:
    registry_hash = _sha("registry")
    provider_hash = _sha("provider")
    producer_commit = _commit("producer")
    asset_producer = _commit("asset-producer")
    asset_execution = _commit("asset-execution")
    data_range_hash = _sha("data-range")
    config_hashes = {cell: _sha("config-" + cell) for cell in EXPECTED_CELL_IDS}
    checkpoint_hashes = {cell: _sha("checkpoint-" + cell) for cell in EXPECTED_CELL_IDS}
    manifest = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY",
        "scope": "formal",
        "asset_resolution_hash": _sha("asset-resolution"),
        "asset_producer_commit": asset_producer,
        "asset_execution_commit": asset_execution,
        "checkpoints": [
            {
                "cell_id": cell,
                "model_id": cell.split(":", 1)[0],
                "training_stage": cell.split(":", 1)[1],
                "checkpoint_id": "checkpoint-" + cell,
                "checkpoint_hash": checkpoint_hashes[cell],
                "checkpoint_revision": "revision-" + cell,
                "registry_hash": registry_hash,
                "config_hash": config_hashes[cell],
            }
            for cell in EXPECTED_CELL_IDS
        ],
        "data": {"data_range_hash": data_range_hash, "dataset_id": "formal-fixture"},
        "data_range_hash": data_range_hash,
        "registry_hash": registry_hash,
    }
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    delta_body = {"schema_version": "stage2-reference-delta-sci-v1", "delta_sci_by_B": {"2": 1_000_000.0, "4": 1_000_000.0}}
    delta_ref = "evidence/delta-sci.json"
    write_canonical_json(root / delta_ref, delta_body)
    delta_hash = canonical_json_hash(delta_body)
    cells: list[CellInput] = []
    output_dirs: list[Path] = []
    for index, cell in enumerate(EXPECTED_CELL_IDS):
        output_dir = f"runs/g23-cell-{index}"
        output_path = root / output_dir
        output_dirs.append(output_path)
        plan_body = {
            "schema_version": "stage2-reference-sizing-plan-v1",
            "reference_id": "reference-" + str(index),
            "candidate_sample_counts": [2, 4],
            "block_size": 1,
            "convergence_tolerance": 0.02,
            "required_consecutive": 1,
            "execution_evidence_hash": _sha("execution"),
        }
        plan = _hashed(plan_body)
        plan_hash = str(plan["artifact_hash"])
        sizing_result_hash = _sha("sizing-result-" + cell)
        sizing_draw_hash = _sha("sizing-draw-" + cell)
        stream_a_hash = _sha("stream-a-" + cell)
        stream_b_hash = _sha("stream-b-" + cell)
        sizing_identity_hash = canonical_json_hash({"plan_hash": plan_hash, "provider_state_digest": provider_hash, "registry_hash": registry_hash, "sizing_draw_hash": sizing_draw_hash, "sizing_stream": "reference_sizing"})
        sizing_result_identity_hash = canonical_json_hash({"sizing_result_hash": sizing_result_hash, "provider_state_digest": provider_hash, "registry_hash": registry_hash, "stream_a_draw_hash": stream_a_hash, "stream_b_draw_hash": stream_b_hash})
        one_shot_plan = _hashed({"schema_version": "stage2-reference-one-shot-plan-v1", "reference_id": "reference-" + str(index), "sizing_result_hash": sizing_result_hash, "sample_count_per_stream": 4, "block_size": 1, "sizing_stream": "reference_sizing", "stream_a": "reference_A", "stream_b": "reference_B", "one_shot": True})
        sizing_latest, final_commit, raw = _publish_progress(root, output_dir, plan_hash=plan_hash, final_plan_hash=str(one_shot_plan["artifact_hash"]), sizing_result_hash=sizing_result_hash, provider_hash=provider_hash, registry_hash=registry_hash, sizing_draw_hash=sizing_draw_hash, sizing_identity_hash=sizing_identity_hash, stream_a_hash=stream_a_hash, stream_b_hash=stream_b_hash, sizing_result_identity_hash=sizing_result_identity_hash)
        blocks_a, blocks_b = raw["blocks_a"], raw["blocks_b"]
        bias = {"p0": np.asarray([100.0, 100.0]), "p1": np.asarray([400.0, 400.0])}
        zeros = {name: np.zeros_like(value) for name, value in bias.items()}
        raw_block_digest = canonical_json_hash([_vector_digest(item) for item in blocks_a + blocks_b])
        numerical_meta = _hashed({"schema_version": "stage2-reference-numerical-diagnostics-v1", "recompute_method": "longdouble_pairwise_u_from_committed_blocks", "raw_block_digest": raw_block_digest, "raw_block_count_a": 4, "raw_block_count_b": 4, "high_precision_hash": _vector_digest(bias), "accumulated_hash": _vector_digest(bias), "max_abs_error": 0.0, "resume_latest_commit_ref": "commits/00000004.json", "resume_latest_commit_hash": final_commit["artifact_hash"], "resume_latest_manifest_hash": final_commit["object_manifest_hash"]})
        bundle_state = {"bias_reference": bias, "cross_reference": bias, "ranking_reference": bias, "uncertainty": {"bias_variance": zeros, "cross_variance": zeros, "ranking_variance": zeros}, "sequence_variance": zeros, "numerical_diagnostics": {"schema_version": "stage2-reference-numerical-diagnostics-v1", "raw_block_digest": raw_block_digest, "high_precision": bias, "accumulated": bias, "high_precision_hash": _vector_digest(bias), "accumulated_hash": _vector_digest(bias)}, "cell_marker": np.asarray([float(index)])}
        bundle = publish_tensor_bundle(output_path / "tensor-bundles" / "reference-final", bundle_state)
        reference_body = {"schema_version": "reference-result-v1", "reference_id": "reference-" + str(index), "bias_reference_hash": _vector_digest(bias), "cross_reference_hash": _vector_digest(bias), "ranking_reference_hash": _vector_digest(bias), "sample_count_a": 4, "sample_count_b": 4, "block_size": 1, "registry_hash": registry_hash, "scope": "formal", "formal_eligible": False, "metadata": {"sizing_result_hash": sizing_result_hash, "sequence_variance_hash": _vector_digest(zeros)}, "tensor_bundle_ref": f"{output_dir}/tensor-bundles/reference-final", "tensor_bundle_manifest_hash": bundle.manifest_sha256}
        reference = _hashed(reference_body)
        uncertainty = _hashed({"schema_version": "stage2-reference-uncertainty-v1", "estimator": "block_u_delete_one_jackknife", "confidence_level": 0.95, "block_count_a": 4, "block_count_b": 4, "bias_variance_hash": _vector_digest(zeros), "cross_variance_hash": _vector_digest(zeros), "ranking_variance_hash": _vector_digest(zeros), "trace_bias_variance": 0.0, "bias_half_width_l2": 0.0})
        one_shot = _hashed({"schema_version": "stage2-reference-one-shot-result-v1", "plan_hash": one_shot_plan["artifact_hash"], "sizing_result_hash": sizing_result_hash, "provider_state_digest": provider_hash, "registry_hash": registry_hash, "processed_sample_count_per_stream": 4, "bias_reference_hash": _vector_digest(bias), "cross_reference_hash": _vector_digest(bias), "ranking_reference_hash": _vector_digest(bias), "uncertainty": uncertainty, "stream_a_draw_hash": stream_a_hash, "stream_b_draw_hash": stream_b_hash, "status": "COMPLETE", "one_shot": True, "weighting_assumptions": {"statistical_unit": "block", "weight_unit": "sequence", "sampling_design": "uniform_with_replacement", "weights_exogenous": True, "common_mean_assumption": True}, "sequence_variance_hash": _vector_digest(zeros)})
        row = next(item for item in manifest["checkpoints"] if item["cell_id"] == cell)
        config_identity = _identity({"config_hash": config_hashes[cell], "full_hash": _sha("full-" + cell), "task_id": "stage2.04_reference_target", "checkpoint_config_hash": row["config_hash"]})
        model_identity = _identity({"asset_id": cell.split(":", 1)[0] + "-step0", "revision": "revision-" + cell})
        data_identity = _identity({"asset_id": "formal-fixture", "revision": "data-revision", "data_range_hash": data_range_hash})
        checkpoint_identity = _identity({"checkpoint_id": row["checkpoint_id"], "checkpoint_revision": row["checkpoint_revision"], "checkpoint_asset_id": model_identity["asset_id"], "model_id": row["model_id"], "training_stage": row["training_stage"], "checkpoint_hash": row["checkpoint_hash"], "config_hash": row["config_hash"], "registry_hash": registry_hash, "cell_id": cell})
        registry_artifact = _hashed({"schema_version": "stage2-parameter-registry-artifact-v1", "status": "READY", "registry_hash": registry_hash, "parameter_groups": {"p0": {"layer": "layer0", "module": "module0"}, "p1": {"layer": "layer1", "module": "module1"}}, "source": "exact-fixture"})
        registry_identity = _identity({"registry_hash": registry_hash, "parameter_registry_artifact_hash": registry_artifact["artifact_hash"]})
        replay = {"schema_version": "stage2-reference-resume-replay-v1", "artifact_ref": "commits/00000004.json", "artifact_hash": final_commit["artifact_hash"], "state_digest": final_commit["state_digest"], "object_manifest_hash": final_commit["object_manifest_hash"], "source_one_shot_result_hash": one_shot["artifact_hash"], "replayed_one_shot_result_hash": one_shot["artifact_hash"]}
        replay["replay_hash"] = canonical_json_hash(replay)
        rng_after = canonical_json_hash({"sizing_draw_hash": sizing_draw_hash, "stream_a_draw_hash": stream_a_hash, "stream_b_draw_hash": stream_b_hash})
        convergence_body: dict[str, object] = {"schema_version": "stage2-reference-convergence-report-v1", "plan": plan, "sizing_plan": plan, "sizing_plan_artifact_hash": plan_hash, "sizing_draw_hash": sizing_draw_hash, "sizing_identity_hash": sizing_identity_hash, "status": "COMPLETE", "converged": True, "selected_sample_count_per_stream": 4, "processed_sample_count_per_stream": 4, "points": [], "sizing_result_hash": sizing_result_hash, "one_shot_plan": one_shot_plan, "one_shot_result": one_shot, "sizing_stream": "reference_sizing", "final_streams": ["reference_A", "reference_B"], "final_sample_count_per_stream": 4, "reference_uncertainty": uncertainty, "provider": {}, "sampling_plan_hash": _sha("sampling"), "recovery_semantics": "authoritative_block_pair_commits", "reference_protocol": "authoritative_sizing_and_one_shot_block_pair_commits", "diagnostics_schema_version": "stage2-reference-producer-diagnostics-v1", "stage2_reference_producer_commit": producer_commit, "formal_scope": "formal", "cell_id": cell, "six_cell_manifest": manifest, "six_cell_manifest_hash": manifest["manifest_hash"], "candidate_delta_sci": dict(delta_body, source_ref=delta_ref, source_hash=delta_hash), "candidate_delta_sci_source": delta_ref, "candidate_delta_sci_source_hash": delta_hash, "config_identity": config_identity, "model_identity": model_identity, "data_identity": data_identity, "checkpoint_identity": checkpoint_identity, "registry_identity": registry_identity, "parameter_registry_artifact": registry_artifact, "numerical_diagnostics": numerical_meta, "state_invariance": {"model_state_before_hash": _sha("model"), "model_state_after_hash": _sha("model"), "rng_state_before_hash": _sha("rng-before"), "rng_state_after_hash": rng_after}, "resume_replay": replay, "sizing_result_identity_hash": sizing_result_identity_hash, "numerical_floor": 1.0e-12, "formal_eligible": False}
        convergence_body["reference_producer_diagnostics_hash"] = canonical_json_hash(convergence_body)
        convergence = dict(convergence_body, artifact_hash=canonical_json_hash(convergence_body))
        gate = {"schema_version": "stage23-task-gate-candidate-v1", "task_id": "stage2.04_reference_target", "gate_ids": ["stage2.G2.3"], "gate_status": "NOT_RUN", "local_validation_status": "NOT_RUN", "formal_eligible": False, "reason": "formal_gate_requires_independent_review"}
        store = TaskArtifactStore(root, output_dir)
        refs = {kind: store.publish(task_id="stage2.04_reference_target", artifact_kind=kind, config_hash=config_hashes[cell], run_intent="formal", payload=payload, formal_eligible=True).commit_ref for kind, payload in (("reference_result", reference), ("reference_convergence_report", convergence), ("gate_record", gate))}
        result_body = {"schema_version": "task-run-result-v2", "task_id": "stage2.04_reference_target", "stage": 2, "runner_kind": "reference", "run_intent": "formal", "status": "PASS", "config_hash": config_hashes[cell], "formal_eligible": True, "artifact_refs": refs, "checkpoint_ref": None, "blockers": [], "error_code": None, "message": "task completed", "recovery_mode": "resume_checkpoint", "metadata": {}}
        result_body["result_hash"] = canonical_json_hash({key: value for key, value in result_body.items() if key != "result_hash"})
        result_ref = output_path / "task-run-result.json"
        write_canonical_json(result_ref, result_body)
        cells.append(CellInput(cell, result_ref.relative_to(root).as_posix()))
    return cells, output_dirs


def test_exact_six_formal_fixture_passes_and_duplicate_is_blocked(tmp_path: Path) -> None:
    cells, _ = _build_formal_fixture(tmp_path)
    result = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert result["status"] == "PASS"
    assert result["formal_eligible"] is True
    duplicate = list(cells)
    duplicate[-1] = duplicate[0]
    blocked = evaluate_formal_g23(tmp_path, duplicate, output_root=tmp_path / "attempts")
    assert blocked["status"] == "BLOCKED"
    assert blocked["formal_eligible"] is False


def test_formal_fixture_tampered_resume_bundle_is_blocked(tmp_path: Path) -> None:
    cells, output_dirs = _build_formal_fixture(tmp_path)
    result = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert result["status"] == "PASS"
    tensor_file = next((output_dirs[0] / "tensor-bundles" / "reference-final" / "tensors").glob("*.bin"))
    tensor_file.write_bytes(tensor_file.read_bytes() + b"tamper")
    blocked = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert blocked["status"] == "BLOCKED"
    assert blocked["formal_eligible"] is False
