"""Fail-closed tests for S2.5 input extraction and exhaustive development sweep."""

from copy import deepcopy
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.experiments.preregistration import build_stage2_preregistration
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_s25_formal import _experiment_plan_entries, _make_mappings
from param_importance_nlp.experiments.stage2_s25_inputs import (
    S205InputBlocked,
    build_s205_formal_inputs,
    validate_s205_development_sweep,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from ops.stage2.materialize_s205_formal_inputs import main as materialize_main


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("s205-formal-input-test", tuple(range(23))),
        {"pilot": 11, "confirmatory": 12, "reference_sizing": 13, "reference_A": 14, "reference_B": 15},
    )


def _formal_fixture(root: Path) -> tuple[str, str, str, SamplingPlan]:
    sampling = _sampling()
    preregistration = build_stage2_preregistration(
        seed_plan_hash="a" * 64,
        producer_commit="b" * 40,
        scope="formal",
    )
    store = TaskArtifactStore(root, "evidence/task-inputs")
    prereg_ref = store.publish(
        task_id="stage2.01_scope_hypotheses_and_preregistration",
        artifact_kind="preregistration",
        config_hash="c" * 64,
        run_intent="formal",
        payload=preregistration,
        formal_eligible=True,
    ).commit_ref
    sampling_ref = store.publish(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        artifact_kind="sampling_plan",
        config_hash="d" * 64,
        run_intent="formal",
        payload=sampling.to_dict(),
        formal_eligible=True,
    ).commit_ref
    execution = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="e" * 64,
        asset_manifest_hashes=("f" * 64,),
        prerequisite_gates=(GateRecord(
            gate_id="stage1.G1-EXIT",
            stage=1,
            status=GateStatus.PASS,
            checked_at="2026-08-25T00:00:00+00:00",
            evidence_refs=("evidence/stage1/exit.json",),
        ),),
    )
    execution_ref = "evidence/formal-execution.json"
    write_canonical_json(root / execution_ref, execution.to_dict())
    return prereg_ref, sampling_ref, execution_ref, sampling


def test_materializer_extracts_sampling_payload_and_freezes_complete_grid(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    direct_sampling, sweep = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    assert direct_sampling == sampling.to_dict()
    assert direct_sampling["schema_version"] == "sampling-plan-v1"
    assert sweep["candidate_batch_sizes"] == [32, 64, 128, 256]
    assert sweep["candidate_microbatch_counts"] == [2, 4, 8, 16, 32]
    assert sweep["pilot_draw_count"] == 480
    assert sweep["primary_parameters_selected"] is False
    assert sweep["confirmatory_draws_generated"] is False
    assert sweep["reference_draws_generated"] is False
    assert [entry[0] for entry in _experiment_plan_entries(sweep)] == [0, 32, 96, 224]


def test_sweep_mappings_use_nonoverlapping_pilot_positions(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    _direct, sweep = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    positions: list[int] = []
    for start, plan in _experiment_plan_entries(sweep):
        mappings = _make_mappings(sampling, plan, start_position=start)
        positions.extend(draw.position for mapping in mappings for draw in mapping.draws)
    assert positions == list(range(480))
    assert len(positions) == len(set(positions))


@pytest.mark.parametrize("mutation", ["missing_batch", "overlap", "select_primary", "sampling_hash"])
def test_sweep_tamper_fails_closed(tmp_path: Path, mutation: str) -> None:
    prereg_ref, sampling_ref, execution_ref, sampling = _formal_fixture(tmp_path)
    _direct, original = build_s205_formal_inputs(
        tmp_path,
        preregistration_ref=prereg_ref,
        sampling_plan_ref=sampling_ref,
        formal_execution_ref=execution_ref,
    )
    value = deepcopy(original)
    if mutation == "missing_batch":
        value["entries"].pop()
    elif mutation == "overlap":
        value["entries"][1]["start_position"] = 0
    elif mutation == "select_primary":
        value["primary_parameters_selected"] = True
    else:
        value["sampling_plan_hash"] = "0" * 64
    value["artifact_hash"] = canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    with pytest.raises(S205InputBlocked):
        validate_s205_development_sweep(value, sampling=sampling)


def test_sampling_task_artifact_identity_is_not_accepted_as_preregistration(tmp_path: Path) -> None:
    _prereg_ref, sampling_ref, execution_ref, _sampling_plan = _formal_fixture(tmp_path)
    with pytest.raises(S205InputBlocked, match="TASK_ARTIFACT_IDENTITY_INVALID"):
        build_s205_formal_inputs(
            tmp_path,
            preregistration_ref=sampling_ref,
            sampling_plan_ref=sampling_ref,
            formal_execution_ref=execution_ref,
        )


def test_materializer_is_append_only_and_idempotent(tmp_path: Path) -> None:
    prereg_ref, sampling_ref, execution_ref, _sampling_plan = _formal_fixture(tmp_path)
    argv = [
        "--data-root", str(tmp_path),
        "--preregistration-ref", prereg_ref,
        "--sampling-plan-task-ref", sampling_ref,
        "--formal-execution-ref", execution_ref,
        "--output-root", "evidence/stage2/s205/inputs/run-unique",
    ]
    assert materialize_main(argv) == 0
    assert materialize_main(argv) == 0
    index = tmp_path / "evidence/stage2/s205/inputs/run-unique/index.json"
    tampered = {"schema_version": "tampered"}
    write_canonical_json(index, tampered)
    assert materialize_main(argv) == 3
