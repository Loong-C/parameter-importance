"""Fail-closed checks for the append-only S2.4 r23 recovery amendment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from stage2.prepare_s204_r22_amendment import (  # noqa: E402
    AMENDED_CANDIDATE_SAMPLE_COUNTS,
    CONSERVATIVE_TERMINAL_PROJECTED_VALUE,
    PROJECTED_REQUIRED_SAMPLE_COUNT,
    PROJECTION_MODEL,
    PROJECTION_SEMANTICS,
    _canonical_hash,
    _canonical_wire_hash,
    prepare_r22_amendment,
    validate_r22_amendment,
    verify_amendment_sources,
)
from stage2.produce_s204_multi_round_execution_summary import (  # noqa: E402
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    EXCLUDED_PCI,
    EXPECTED_EXECUTION_COMMITS,
    _queue_hash,
    _runtime_status_hash,
    build_source_manifest,
    main as multi_round_main,
    produce_multi_round_execution_summary,
    validate_multi_round_execution_summary,
)
from param_importance_nlp.contracts import canonical_json_hash  # noqa: E402
from param_importance_nlp.contracts import FormalExecutionEvidence, GateRecord, GateStatus  # noqa: E402
from param_importance_nlp.experiments.stage2_s204_ids import (  # noqa: E402
    EXPECTED_CELL_IDS,
    cell_path_component,
)
from stage2.materialize_s204 import publish_per_cell_sizing_plans  # noqa: E402


TEST_GPU_UUID = sorted(APPROVED_GPU_UUIDS)[0]


def _write_json(path: Path, value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_queue(path: Path, value: dict[str, object]) -> str:
    value = {**value, "artifact_hash": _queue_hash(value)}
    return _write_json(path, value)


def _task_result(*, retryable: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "task-run-result-v2",
        "task_id": "stage2.04_reference_target",
        "stage": 2,
        "runner_kind": "stage2_reference_target",
        "run_intent": "formal",
        "status": "BLOCKED",
        "config_hash": "c" * 64,
        "formal_eligible": False,
        "artifact_refs": {},
        "checkpoint_ref": None,
        "blockers": [{"code": "capacity_unavailable" if retryable else "contract_unfrozen", "retryable": retryable}],
        "error_code": None,
        "message": "fixture blocked",
        "recovery_mode": "fresh",
        "metadata": {},
    }
    payload["result_hash"] = canonical_json_hash(payload)
    return payload


def _final_status(cell_id: str, *, attempt_id: str, retryable: bool, execution_commit: str) -> dict[str, object]:
    task = _task_result(retryable=retryable)
    payload: dict[str, object] = {
        "schema_version": "stage2-s204-cell-final-status-v3",
        "execution_commit": execution_commit,
        "execution_lineage": {"execution_commit": execution_commit},
        "cell_id": cell_id,
        "attempt_id": attempt_id,
        "run_kind": "fresh",
        "config_path": "configs/generated/stage2/s204/r23/config.json",
        "config_hash": "c" * 64,
        "config_full_hash": "d" * 64,
        "checkpoint_revision": "fixture-checkpoint-revision",
        "input_checkpoint_id": "fixture-input-checkpoint",
        "parameter_registry_hash": "e" * 64,
        "status": "BLOCKED",
        "gpu": {"selected_uuid": TEST_GPU_UUID},
        "formal_provider": "TaskRuntime.stage2.04_reference_target",
        "formal_eligible": False,
        "g2_3_gate": "NOT_RUN",
        "task_result_hash": task["result_hash"],
        "task_result_ref": "",
        "bundle_manifest_sha256": None,
        "artifact_refs": {},
        "blockers": [{"code": "capacity_unavailable" if retryable else "contract_unfrozen", "retryable": retryable}],
        "recovery": {"resume_ref": None},
        "events_ref": "events.jsonl",
    }
    return {"final": payload, "task": task}


def _make_attempt(root: Path, *, queue_ref: str, attempt_id: str, execution_commit: str, missing: set[str], retryable: bool) -> None:
    output_root = root / "evidence" / "runs" / attempt_id
    outcomes: dict[str, object] = {}
    for cell_id in EXPECTED_CELL_IDS:
        outcomes[cell_id] = {
            "schema_version": "stage2-s204-r20-cell-status-v1", "run_id": attempt_id, "cell_id": cell_id,
            "gpu_uuid": TEST_GPU_UUID, "pid": 1, "returncode": -15 if cell_id in missing else 3,
            "status": "FAILED" if cell_id in missing else "COMPLETE", "elapsed_seconds": 1.0,
            "execution_commit": execution_commit, "retry": False,
        }
    queue = {
        "schema_version": "stage2-s204-r20-queue-final-v1", "run_name": "formal-r20-g3-v5", "run_id": attempt_id,
        "status": "FAILED", "execution_commit": execution_commit, "approved_gpu_uuids": sorted(APPROVED_GPU_UUIDS),
        "cell_order_lpt": list(EXPECTED_CELL_IDS), "outcomes": outcomes, "retry_policy": "none",
    }
    queue_path = root / queue_ref
    _write_queue(queue_path, queue)
    manifest = {
        "schema_version": "stage2-s204-r20-queue-v1", "run_name": "formal-r20-g3-v5", "run_id": attempt_id,
        "execution_commit": execution_commit, "approved_gpu_uuids": sorted(APPROVED_GPU_UUIDS), "excluded_gpu_uuid": EXCLUDED_GPU_UUID,
        "excluded_pci": EXCLUDED_PCI, "cell_order_lpt": list(EXPECTED_CELL_IDS),
        "cell_estimates_seconds": {cell: 1.0 for cell in EXPECTED_CELL_IDS}, "cell_configs": {}, "python": "python",
        "retry_policy": "none", "output_root": output_root.as_posix(),
    }
    _write_json(queue_path.parent / "queue-manifest.json", {**manifest, "artifact_hash": _queue_hash(manifest)})
    for cell_id in EXPECTED_CELL_IDS:
        if cell_id in missing:
            continue
        refs = _final_status(cell_id, attempt_id=attempt_id, retryable=retryable, execution_commit=execution_commit)
        cell_dir = output_root / cell_path_component(cell_id) / "attempts" / attempt_id
        task_path = root / "evidence" / "runs" / "task-results" / f"{attempt_id}-{cell_path_component(cell_id)}.json"
        _write_json(task_path, refs["task"])
        final = dict(refs["final"])
        final["task_result_ref"] = task_path.as_posix()
        final["artifact_hash"] = _runtime_status_hash(final)
        _write_json(cell_dir / "final-status.json", final)


def _draft(tmp_path: Path) -> dict[str, object]:
    parent = {
        "schema_version": "stage2-reference-sizing-round-v1", "round_id": "r22",
        "parent_preregistration_hash": "ecd7cdcd29bcd917bbef01337a606c20ebbafeb89218f6d405b7ffc9c54ba5f3",
        "final_reference_plan_schema": "schemas/shared/stage2-reference-one-shot-plan-v2.json",
        "sizing": {"candidate_sample_counts": [32768, 65536], "segment_end_position_exclusive": 81920, "parent_sampling_plan_hash": "b" * 64},
    }
    parent["artifact_hash"] = _canonical_hash(parent)
    parent_ref = "evidence/stage2/s204/r22-round-v3-canonical.json"
    _write_json(tmp_path / parent_ref, parent)

    final_ref = "evidence/stage2/s204/formal-r22/early-final-status.json"
    final_payload = {
        "schema_version": "stage2-s204-cell-final-status-v3", "status": "BLOCKED", "cell_id": "pythia-31m-deduped:early",
        "attempt_id": "fresh-ed1e85b9e82278bdf8c3aae1996f147c4c5a4bcf813a16bae3fa38a7f5553979", "formal_eligible": False,
        "execution_commit": "9e5c4315444530371678205d5ee5c3d549e7f084", "execution_lineage": {"execution_commit": "9e5c4315444530371678205d5ee5c3d549e7f084"},
        "config_path": "configs/generated/stage2/s204/prepared-r22-g3-v5-segment-v3/fresh/pythia-31m-deduped__early/config.json", "config_hash": "c" * 64, "config_full_hash": "d" * 64,
        "checkpoint_revision": "fixture-checkpoint-revision", "input_checkpoint_id": "fixture-input-checkpoint", "parameter_registry_hash": "e" * 64,
        "blockers": [{"code": "contract_unfrozen", "retryable": False}], "artifact_refs": {},
    }
    final_payload["artifact_hash"] = _canonical_hash(final_payload)
    final_hash = _write_json(tmp_path / final_ref, final_payload)
    checkpoint_ref = "configs/generated/stage2/s204/r22/early/resume/reference-sizing/bounded-checkpoint/manifest.json"
    checkpoint_hash = _write_json(tmp_path / checkpoint_ref, {
        "schema_version": "runtime.tensor-bundle.v1", "codec": "raw-tensor-bundle.v1",
        "state": {"checkpoint_schema": "stage2-reference-bounded-checkpoint-v1", "schema_version": "stage2-reference-progress-state-v1", "block_size": 32,
                   "processed_block_pairs": 2048, "convergence_streak": 0, "selected_sample_count_per_stream": None, "sizing_stream": True,
                   "points": [
                       {"sample_count_per_stream": 32768, "block_count_total": 2048, "bias_reference_hash": "1" * 64, "cross_reference_hash": "2" * 64, "ranking_reference_hash": "3" * 64, "comparison_defined": False, "comparison_reason": "no_previous_reference", "normalized_l1_from_previous": None, "convergence_streak": 0},
                       {"sample_count_per_stream": 65536, "block_count_total": 4096, "bias_reference_hash": "4" * 64, "cross_reference_hash": "5" * 64, "ranking_reference_hash": "6" * 64, "comparison_defined": True, "comparison_reason": None, "normalized_l1_from_previous": 0.028604503208326363, "convergence_streak": 0},
                   ]}, "tensors": [],
    })
    _make_attempt(tmp_path, queue_ref="operations/r21/queue-final.json", attempt_id="r21", execution_commit=EXPECTED_EXECUTION_COMMITS["r21"], missing={EXPECTED_CELL_IDS[4], EXPECTED_CELL_IDS[5]}, retryable=False)
    _make_attempt(tmp_path, queue_ref="operations/r22-retry2/queue-final.json", attempt_id="r22-retry2", execution_commit=EXPECTED_EXECUTION_COMMITS["r22-retry2"], missing=set(), retryable=True)
    _make_attempt(tmp_path, queue_ref="operations/r22-retry3/queue-final.json", attempt_id="r22-retry3", execution_commit=EXPECTED_EXECUTION_COMMITS["r22-retry3"], missing=set(), retryable=False)
    evaluation = {"schema_version": "stage2-g23-reference-evaluation-v1", "gate_id": "stage2.G2.3", "status": "BLOCKED", "formal_eligible": False,
                  "required_cell_count": 6, "complete_cell_count": 0, "expected_cell_ids": list(EXPECTED_CELL_IDS), "cells": [], "calculator": {}, "thresholds": {}, "reasons": ["fixture"]}
    evaluation["artifact_hash"] = canonical_json_hash(evaluation)
    eval_ref = "evidence/g23/r22-retry3/evaluation.json"
    _write_json(tmp_path / eval_ref, evaluation)
    source_manifest = build_source_manifest(
        tmp_path,
        r21_queue_final="operations/r21/queue-final.json",
        r22_retry2_queue_final="operations/r22-retry2/queue-final.json",
        r22_retry3_queue_final="operations/r22-retry3/queue-final.json",
        r22_retry3_evaluation=eval_ref,
    )
    source_ref = "evidence/stage2/s204/multi-round-source.json"
    _write_json(tmp_path / source_ref, source_manifest)
    summary = produce_multi_round_execution_summary(tmp_path, tmp_path / source_ref)
    summary_ref = "evidence/stage2/s204/multi-round-execution-summary.json"
    summary_hash = _write_json(tmp_path / summary_ref, summary)
    return {
        "schema_version": "stage2-reference-sizing-amendment-v1", "round_id": "r23", "prior_round_id": "r22", "prior_round_status": "BLOCKED", "amendment_id": "r23-amend-r1", "amendment_version": 1, "parent_round_id": "r22", "parent_round_ref": parent_ref, "parent_round_artifact_hash": parent["artifact_hash"], "parent_preregistration_hash": parent["parent_preregistration_hash"], "parent_execution_status": "BLOCKED", "parent_blocker_code": "contract_unfrozen",
        "sizing": {"stream": "reference_sizing", "reference_study_id": "stage2-s204-r23-independent-reference-study", "sizing_run_id": "r23-g3-v5-independent-segment-amend-r1", "fresh_attempt_id": "fresh-r23-amend-r1", "resume_ref": None, "seed_namespaces": {"reference_sizing": "reference_sizing:r23-independent-segment-amend-r1", "reference_A": "reference_A:r23-independent-segment-amend-r1", "reference_B": "reference_B:r23-independent-segment-amend-r1"}, "producer_commit": "f" * 40, "seed_namespace": "reference_sizing:frozen-parent-plan", "seed_namespace_mode": "same_frozen_seed_disjoint_segment", "parent_sampling_plan_hash": "b" * 64, "candidate_sample_counts": list(AMENDED_CANDIDATE_SAMPLE_COUNTS), "block_size": 32, "normalized_l1_threshold": 0.02, "required_consecutive": 1, "complete_all_candidates": True, "optional_stopping": False, "reuse_prior_sizing_prefix": False, "segment_start_position": 81920, "segment_end_position_exclusive": 344064, "prior_consumed_end_position": 81920, "final_stream_segments": {"reference_A": {"start_position": 81920, "end_position_exclusive": 344064}, "reference_B": {"start_position": 81920, "end_position_exclusive": 344064}},},
        "machine_diagnostics": {"basis_type": "bounded_checkpoint_terminal_failure", "cell_id": "pythia-31m-deduped:early", "attempt_id": final_payload["attempt_id"], "retryable": False, "formal_eligible": False, "execution_commit": final_payload["execution_commit"], "config_path": final_payload["config_path"], "config_hash": final_payload["config_hash"], "config_full_hash": final_payload["config_full_hash"], "checkpoint_revision": final_payload["checkpoint_revision"], "input_checkpoint_id": final_payload["input_checkpoint_id"], "parameter_registry_hash": final_payload["parameter_registry_hash"], "plan_identity": {"reference_id": "fixture-plan", "artifact_hash": "ddfa1983fb2915ea9965f4229b7416331b725d9c0e7801fab9a505375766dac4"}, "registry_identity": {"registry_hash": "9724c41d77c7f6d5eae972cbbbd2f6affb6ec5163b585c5b939b82856bf95afc", "parameter_registry_artifact_hash": "9724c41d77c7f6d5eae972cbbbd2f6affb6ec5163b585c5b939b82856bf95afc"}, "draw_identity": {"sampling_plan_hash": "5e4d863d17584c0b7ffa6e09813f87c1a7c19f1e98a6c2e97577532bd3e2b8fa", "seed_namespace": "reference_sizing:frozen-parent-plan", "segment_start_position": 81920, "segment_end_position_exclusive": 344064}, "sizing_points": [{"sample_count_per_stream": 32768, "count": 1024, "n1": 67108864.0, "n2": 4398046511104.0, "comparison_defined": False, "comparison_reason": "no_previous_reference", "normalized_l1_from_previous": None, "convergence_streak": 0, "selected_sample_count_per_stream": None}, {"sample_count_per_stream": 65536, "count": 2048, "n1": 134217728.0, "n2": 8796093022208.0, "comparison_defined": True, "comparison_reason": None, "normalized_l1_from_previous": 0.028604503208326363, "convergence_streak": 0, "selected_sample_count_per_stream": None}], "source_refs": {"final_status": {"ref": final_ref, "sha256": final_hash}, "bounded_checkpoint_manifest": {"ref": checkpoint_ref, "sha256": checkpoint_hash}}, "terminal_sample_count_per_stream": 65536, "terminal_comparison_defined": True, "terminal_normalized_l1_from_previous": 0.028604503208326363, "terminal_convergence_streak": 0, "terminal_selected_sample_count_per_stream": None, "threshold": 0.02, "projection_model": PROJECTION_MODEL, "projection_semantics": PROJECTION_SEMANTICS, "projected_required_sample_count": PROJECTED_REQUIRED_SAMPLE_COUNT, "conservative_terminal_sample_count": 262144, "projected_terminal_normalized_l1": CONSERVATIVE_TERMINAL_PROJECTED_VALUE},
        "multi_round_control": {"schema_version": "stage2-multi-round-control-v1", "method": "single_confirmatory_look_control_v1", "horizon_round_ids": ["r22", "r23"], "allowed_recovery_round_ids": ["r23"], "prior_round_ids": ["r21", "r22"], "prior_rounds_not_pooled": True, "prior_final_ab_created": {"r21": False, "r22": False}, "execution_summary_ref": {"ref": summary_ref, "sha256": summary_hash}, "final_ab_policy": "single_research_one_shot_only", "max_final_ab_rounds": 1, "r23_failure_policy": "INCONCLUSIVE_NO_FURTHER_RETRY", "anytime_valid": False, "multiplicity_policy": "no_cross_round_pooling;confirmatory_claims_withheld_until_single_final_A_B", "r23_is_only_recovery": True, "r23_is_only_confirmatory_look": True, "confirmatory_look_policy": "r23_only_single_final_A_B_look;failure_is_final_INCONCLUSIVE"},
        "run_identity": "r23-g3-v5-independent-segment-amend-r1-20260825T000000Z", "output_namespace": "evidence/stage2/s204/formal-r23-g3-v5-independent-segment-amend-r1", "new_draws_before_freeze": False, "final_reference_created": False, "final_reference_plan_schema": "schemas/shared/stage2-reference-one-shot-plan-v2.json", "continuation_control": "precommitted_new_round_disjoint_segment_no_pooling_with_prior_r22",
        "amendment": {"append_only": True, "version": 1, "created_before_new_sizing_draws": True, "candidate_expansion": "append_next_two_doubling_nodes_after_terminal", "unchanged_scientific_contract": ["threshold=0.02", "block_size=32", "required_consecutive=1", "margin_schema_unchanged", "evaluator_schema_unchanged", "final_A_B_schema_unchanged"], "reason": "r22_terminal_65536_gap_above_threshold", "changed_fields": ["round_id", "sizing.candidate_sample_counts", "sizing.segment_start_position", "sizing.segment_end_position_exclusive", "multi_round_control"], "non_posthoc_basis": "bounded_checkpoint_machine_diagnostic_before_r23_draws", "affected_gates": ["G2.3", "G2.4"], "review": {"status": "ACCEPTED", "reviewer_role": "root", "reviewed_before_new_draws": True, "reviewed_at": "2026-08-25T08:00:00Z"}},
    }


def _ready_draft(tmp_path: Path) -> dict[str, object]:
    value = _draft(tmp_path)
    diagnostics = value["machine_diagnostics"]
    assert isinstance(diagnostics, dict)
    diagnostics["draw_identity"] = {
        "parent_sampling_plan_hash": "b" * 64,
        "sizing_draw_hash": "5e4d863d17584c0b7ffa6e09813f87c1a7c19f1e98a6c2e97577532bd3e2b8fa",
        "sizing_identity_hash": "3375dda050e86ea4ce0abe3c562b207eeae62d748167ebc1e0589ad413c73dc6",
        "seed_namespace": "reference_sizing:frozen-parent-plan",
        "segment_start_position": 16384,
        "segment_end_position_exclusive": 81920,
    }
    point_hashes = [("1" * 64, "2" * 64, "3" * 64), ("4" * 64, "5" * 64, "6" * 64)]
    diagnostics["sizing_points"] = [
        {"sample_count_per_stream": 32768, "block_count_total": 2048, "bias_reference_hash": point_hashes[0][0], "cross_reference_hash": point_hashes[0][1], "ranking_reference_hash": point_hashes[0][2], "comparison_defined": False, "comparison_reason": "no_previous_reference", "normalized_l1_from_previous": None, "convergence_streak": 0},
        {"sample_count_per_stream": 65536, "block_count_total": 4096, "bias_reference_hash": point_hashes[1][0], "cross_reference_hash": point_hashes[1][1], "ranking_reference_hash": point_hashes[1][2], "comparison_defined": True, "comparison_reason": None, "normalized_l1_from_previous": 0.028604503208326363, "convergence_streak": 0},
    ]
    checkpoint_ref = diagnostics["source_refs"]["bounded_checkpoint_manifest"]["ref"]  # type: ignore[index]
    checkpoint_path = tmp_path / checkpoint_ref
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    state = checkpoint["state"]
    state.update(
        {
            "plan_hash": diagnostics["plan_identity"]["artifact_hash"],
            "registry_hash": diagnostics["registry_identity"]["registry_hash"],
            "sizing_stream": True,
            "sizing_draw_hash": diagnostics["draw_identity"]["sizing_draw_hash"],
            "sizing_identity_hash": diagnostics["draw_identity"]["sizing_identity_hash"],
        }
    )
    rng_state = {
        "algorithm_version": "fixture-mt19937-v1",
        "stream": "reference_sizing",
        "count": 81920,
        "state_before": ["fixture-before"],
        "state_after": ["fixture-after"],
    }
    rng_state["state_before_sha256"] = _canonical_wire_hash(
        {"algorithm_version": rng_state["algorithm_version"], "state": rng_state["state_before"]}
    )
    rng_state["state_after_sha256"] = _canonical_wire_hash(
        {"algorithm_version": rng_state["algorithm_version"], "state": rng_state["state_after"]}
    )
    state["rng_state"] = rng_state
    state["rng_state_digest"] = _canonical_wire_hash(rng_state)
    checkpoint["state"]["points"] = diagnostics["sizing_points"]
    checkpoint_hash = _write_json(checkpoint_path, checkpoint)
    diagnostics["source_refs"]["bounded_checkpoint_manifest"]["sha256"] = checkpoint_hash  # type: ignore[index]
    return value


def test_amendment_freezes_r23_next_two_nodes_and_producer_summary(tmp_path: Path) -> None:
    result = prepare_r22_amendment(_ready_draft(tmp_path))
    assert result["status"] == "FROZEN_BEFORE_NEW_SIZING_DRAWS"
    assert result["execution_contract"]["must_complete_candidate_nodes"] == [131072, 262144]  # type: ignore[index]
    assert result["multi_round_control"]["r23_is_only_confirmatory_look"] is True  # type: ignore[index]
    assert result["artifact_hash"] == _canonical_hash({k: v for k, v in result.items() if k != "artifact_hash"})
    verify_amendment_sources(result, tmp_path)


@pytest.mark.parametrize(("path", "bad"), [
    (("sizing", "candidate_sample_counts"), [131072, 524288]),
    (("sizing", "normalized_l1_threshold"), 0.021),
    (("sizing", "segment_start_position"), 16384),
    (("machine_diagnostics", "terminal_normalized_l1_from_previous"), 0.02),
    (("amendment", "append_only"), False),
])
def test_amendment_rejects_posthoc_or_contract_drift(tmp_path: Path, path: tuple[str, ...], bad: object) -> None:
    value = _ready_draft(tmp_path)
    target: object = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = bad  # type: ignore[index]
    with pytest.raises(ValueError):
        validate_r22_amendment(value)


def test_amendment_is_bound_to_parent_preregistration_and_summary(tmp_path: Path) -> None:
    value = _ready_draft(tmp_path)
    value["parent_preregistration_hash"] = "c" * 64
    with pytest.raises(ValueError):
        verify_amendment_sources(value, tmp_path)


def test_wire_absolute_config_path_normalises_to_logical_ref(tmp_path: Path) -> None:
    value = _ready_draft(tmp_path)
    diagnostics = value["machine_diagnostics"]
    assert isinstance(diagnostics, dict)
    final_ref = diagnostics["source_refs"]["final_status"]["ref"]
    final_path = tmp_path / final_ref
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["config_path"] = (tmp_path / diagnostics["config_path"]).as_posix()
    final["artifact_hash"] = _canonical_hash(final)
    final_hash = _write_json(final_path, final)
    diagnostics["source_refs"]["final_status"]["sha256"] = final_hash
    verify_amendment_sources(value, tmp_path)


def test_sizing_round_namespace_keeps_r22_identity_and_separates_r23(tmp_path: Path) -> None:
    gate = GateRecord(
        gate_id="stage2.G2.2",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-08-25T08:00:00+00:00",
        evidence_refs=("evidence/g2-2.json",),
    )
    evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=(gate,),
    )
    r22 = publish_per_cell_sizing_plans(
        tmp_path,
        formal_execution=evidence,
        require_terminal_convergence=True,
        round_manifest_ref="evidence/r22-round.json",
        output_dir="evidence/r22-sizing",
    )
    r23 = publish_per_cell_sizing_plans(
        tmp_path,
        formal_execution=evidence,
        round_namespace="r23",
        require_terminal_convergence=True,
        round_manifest_ref="evidence/r23-round.json",
        output_dir="evidence/r23-sizing",
    )
    assert set(r22) == set(r23) == set(EXPECTED_CELL_IDS)
    assert set(r22.values()).isdisjoint(r23.values())
    for cell_id in EXPECTED_CELL_IDS:
        r22_plan = json.loads((tmp_path / r22[cell_id]).read_text(encoding="utf-8"))
        r23_plan = json.loads((tmp_path / r23[cell_id]).read_text(encoding="utf-8"))
        component = cell_path_component(cell_id)
        assert r22_plan["reference_id"] == f"stage2-s204-{component}-sizing"
        assert r23_plan["reference_id"] == f"stage2-s204-r23-{component}-sizing"
        assert r22_plan["artifact_hash"] != r23_plan["artifact_hash"]


def test_summary_cli_derives_and_publishes_source_manifest(tmp_path: Path) -> None:
    _draft(tmp_path)
    source_output = tmp_path / "evidence/stage2/s204/cli-source.json"
    summary_output = tmp_path / "evidence/stage2/s204/cli-summary.json"
    code = multi_round_main(
        [
            "--data-root", str(tmp_path),
            "--r21-queue-final", "operations/r21/queue-final.json",
            "--r22-retry2-queue-final", "operations/r22-retry2/queue-final.json",
            "--r22-retry3-queue-final", "operations/r22-retry3/queue-final.json",
            "--r22-retry3-evaluation", "evidence/g23/r22-retry3/evaluation.json",
            "--source-manifest-output", str(source_output),
            "--output", str(summary_output),
        ]
    )
    assert code == 0
    source = json.loads(source_output.read_text(encoding="utf-8"))
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert summary["source_manifest"]["ref"] == "evidence/stage2/s204/cli-source.json"
    assert source["artifact_hash"] == canonical_json_hash({key: item for key, item in source.items() if key != "artifact_hash"})


def test_amendment_schema_matches_python_wire_contract(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    result = prepare_r22_amendment(_ready_draft(tmp_path))
    schema = json.loads(
        (ROOT / "schemas/shared/stage2-reference-sizing-amendment-v1.json").read_text(encoding="utf-8")
    )
    diagnostics_schema = schema["properties"]["machine_diagnostics"]
    draw_schema = diagnostics_schema["properties"]["draw_identity"]
    point_schema = diagnostics_schema["properties"]["sizing_points"]["items"]
    assert set(draw_schema["required"]) == set(result["machine_diagnostics"]["draw_identity"])
    assert set(point_schema["required"]) == set(result["machine_diagnostics"]["sizing_points"][0])
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(result))
    assert errors == []


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("plan_hash", "f" * 64),
        ("registry_hash", "f" * 64),
        ("sizing_draw_hash", "f" * 64),
        ("sizing_identity_hash", "f" * 64),
        ("sizing_stream", False),
    ],
)
def test_checkpoint_identity_must_match_real_diagnostic_wire(
    tmp_path: Path, field: str, bad: object
) -> None:
    value = _ready_draft(tmp_path)
    diagnostics = value["machine_diagnostics"]
    assert isinstance(diagnostics, dict)
    checkpoint_ref = diagnostics["source_refs"]["bounded_checkpoint_manifest"]["ref"]
    checkpoint_path = tmp_path / checkpoint_ref
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["state"][field] = bad
    checkpoint_hash = _write_json(checkpoint_path, checkpoint)
    diagnostics["source_refs"]["bounded_checkpoint_manifest"]["sha256"] = checkpoint_hash
    with pytest.raises(ValueError, match="CHECKPOINT_(IDENTITY|RNG_BOUNDARY)"):
        verify_amendment_sources(value, tmp_path)


def test_checkpoint_rng_boundary_is_bound_to_absolute_parent_segment(tmp_path: Path) -> None:
    value = _ready_draft(tmp_path)
    diagnostics = value["machine_diagnostics"]
    assert isinstance(diagnostics, dict)
    checkpoint_ref = diagnostics["source_refs"]["bounded_checkpoint_manifest"]["ref"]
    checkpoint_path = tmp_path / checkpoint_ref
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["state"]["rng_state"]["count"] = 81952
    checkpoint_hash = _write_json(checkpoint_path, checkpoint)
    diagnostics["source_refs"]["bounded_checkpoint_manifest"]["sha256"] = checkpoint_hash
    with pytest.raises(ValueError, match="RNG_BOUNDARY"):
        verify_amendment_sources(value, tmp_path)


def test_summary_source_manifest_is_required_in_schema_and_python(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    _draft(tmp_path)
    summary_path = tmp_path / "evidence/stage2/s204/multi-round-execution-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schemas/shared/stage2-multi-round-execution-summary-v1.json").read_text(encoding="utf-8")
    )
    assert list(jsonschema.Draft202012Validator(schema).iter_errors(summary)) == []
    missing = copy.deepcopy(summary)
    del missing["source_manifest"]
    missing["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in missing.items() if key != "artifact_hash"}
    )
    assert any(error.validator == "required" for error in jsonschema.Draft202012Validator(schema).iter_errors(missing))
    with pytest.raises(ValueError, match="FIELDS_INVALID"):
        validate_multi_round_execution_summary(missing)


def test_summary_rejects_r22_retry2_non_failed_queue(tmp_path: Path) -> None:
    _draft(tmp_path)
    source_path = tmp_path / "evidence/stage2/s204/multi-round-source.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    queue_path = tmp_path / "operations/r22-retry2/queue-final.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    queue["status"] = "COMPLETE"
    queue_hash = _write_queue(queue_path, queue)
    source["rounds"]["r22"]["attempts"][0]["queue_final"]["sha256"] = queue_hash
    source["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in source.items() if key != "artifact_hash"}
    )
    _write_json(source_path, source)
    with pytest.raises(ValueError, match="RETRY2_QUEUE_FAILED_REQUIRED"):
        produce_multi_round_execution_summary(tmp_path, source_path)


def test_summary_rejects_final_status_gpu_or_attempt_path_mismatch(tmp_path: Path) -> None:
    _draft(tmp_path)
    source_path = tmp_path / "evidence/stage2/s204/multi-round-source.json"
    queue = json.loads((tmp_path / "operations/r22-retry3/queue-final.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "operations/r22-retry3/queue-manifest.json").read_text(encoding="utf-8"))
    cell_id = EXPECTED_CELL_IDS[0]
    final_path = (
        Path(manifest["output_root"])
        / cell_path_component(cell_id)
        / "attempts"
        / "r22-retry3"
        / "final-status.json"
    )
    final = json.loads(final_path.read_text(encoding="utf-8"))
    final["gpu"]["selected_uuid"] = sorted(APPROVED_GPU_UUIDS)[1]
    final["artifact_hash"] = _runtime_status_hash(final)
    _write_json(final_path, final)
    with pytest.raises(ValueError, match="FINAL_STATUS_GPU_MISMATCH"):
        produce_multi_round_execution_summary(tmp_path, source_path)
