"""v0.4 声明式 builder CLI：不手算 hash，也不依赖临时 Python 脚本。"""

from __future__ import annotations

from pathlib import Path

from param_importance_nlp.cli import build_parser, main
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments import (
    AblationMatrix,
    AblationMatrixDeclaration,
    FormalExperimentPlan,
    SamplingPlan,
    SamplingUniverse,
    TrainingRouteSpec,
)


def _fixture_decision() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "cli-builder-fixture",
        "selected_estimator": "u",
        "scope": "local_fixture",
        "status": "FIXTURE_ONLY",
        "state": "UNFROZEN",
        "batch_size": 32,
        "microbatch_count": 2,
        "repetitions": 2,
        "gate_id": "stage2.G2.7b",
        "gate_status": "NOT_RUN",
        "artifact_ref": None,
        "metadata": {"formal_eligible": False},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def _phase(
    phase_id: str,
    phase_type: str,
    output_checkpoint_id: str,
    *,
    parent_phase_id: str | None = None,
    input_checkpoint_id: str | None = None,
    task_id: str | None = None,
) -> dict[str, object]:
    return {
        "phase_id": phase_id,
        "phase_type": phase_type,
        "base_initialization_id": "tiny-init-v1",
        "model_asset_id": "tiny-model-v1",
        "dataset_asset_id": f"tiny-{phase_id}-data-v1",
        "output_checkpoint_id": output_checkpoint_id,
        "checkpoint_frequency_steps": 1,
        "parent_phase_id": parent_phase_id,
        "input_checkpoint_id": input_checkpoint_id,
        "task_id": task_id,
        "importance_enabled": True,
        "metadata": {},
    }


def test_route_build_compiles_hash_and_requires_decision(tmp_path: Path) -> None:
    source = {
        "schema_version": "training-route-source-v1",
        "route_id": "cli-route",
        "run_intent": "local_fixture",
        "phases": [
            _phase("pretrain", "pretrain", "pretrain-checkpoint"),
            _phase(
                "finetune",
                "finetune",
                "finetune-checkpoint",
                parent_phase_id="pretrain",
                input_checkpoint_id="pretrain-checkpoint",
                task_id="fixture-task",
            ),
        ],
        "metadata": {"source": "test"},
    }
    source_path = tmp_path / "route-source.json"
    decision_path = tmp_path / "decision.json"
    output_path = tmp_path / "route.json"
    write_canonical_json(source_path, source)
    write_canonical_json(decision_path, _fixture_decision())

    assert main([
        "artifact", "route-build", "--spec", str(source_path),
        "--decision", str(decision_path), "--output", str(output_path),
    ]) == 0
    route = TrainingRouteSpec.from_mapping(load_canonical_json(output_path))
    assert route.route_id == "cli-route"
    assert route.lineage_hash == load_canonical_json(output_path)["lineage_hash"]

    missing_output = tmp_path / "missing-decision.json"
    assert main([
        "artifact", "route-build", "--spec", str(source_path),
        "--output", str(missing_output),
    ]) == 2
    assert not missing_output.exists()


def test_ablation_builder_publishes_declaration_and_compiled_matrix(
    tmp_path: Path,
) -> None:
    source = {
        "schema_version": "ablation-matrix-source-v1",
        "matrix_id": "cli-ablation",
        "base_config": {"training": {"batch_size": 32}, "importance": {"estimator": "u"}},
        "factors": [
            {
                "name": "batch_size",
                "config_path": ["training", "batch_size"],
                "baseline_value": 32,
                "alternatives": [16, 64],
            },
            {
                "name": "estimator",
                "config_path": ["importance", "estimator"],
                "baseline_value": "u",
                "alternatives": ["double"],
            },
        ],
        "base_seed": 1337,
        "seed_namespace": "stage8-cli-test",
        "scope": "local_fixture",
    }
    source_path = tmp_path / "ablation-source.json"
    declaration_path = tmp_path / "declaration.json"
    matrix_path = tmp_path / "matrix.json"
    write_canonical_json(source_path, source)

    assert main([
        "artifact", "ablation-matrix-build", "--spec", str(source_path),
        "--output", str(declaration_path), "--compiled-output", str(matrix_path),
    ]) == 0
    declaration = AblationMatrixDeclaration.from_mapping(
        load_canonical_json(declaration_path)
    )
    matrix = AblationMatrix.from_mapping(load_canonical_json(matrix_path))
    assert declaration.compile().to_dict() == matrix.to_dict()
    assert len(matrix.cells) == 4
    assert main(["artifact-validate", str(declaration_path)]) == 0


def test_run_ready_auxiliary_commands_are_registered() -> None:
    parser = build_parser(prog="param-importance")
    fixture = parser.parse_args(
        ["task", "fixture-all", "--workspace-root", "runs/full-fixture"]
    )
    tensorboard = parser.parse_args(
        [
            "task", "tensorboard-rebuild", "--event", "events.jsonl",
            "--output-dir", "runs/tensorboard",
        ]
    )
    assert fixture.handler.__name__ == "_task_fixture_all"
    assert tensorboard.handler.__name__ == "_task_tensorboard_rebuild"


def test_quadrature_pilot_plan_builder_binds_formal_evidence(tmp_path: Path) -> None:
    gate = GateRecord(
        "stage0.G10",
        0,
        GateStatus.PASS,
        "2026-07-22T00:00:00Z",
        evidence_refs=("commits/stage0-g10.json",),
    )
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="1" * 64,
        asset_manifest_hashes=("2" * 64,),
        prerequisite_gates=(gate,),
    )
    evidence_path = tmp_path / "formal-evidence.json"
    output_path = tmp_path / "pilot-plan.json"
    write_canonical_json(evidence_path, evidence.to_dict())

    source = (
        Path(__file__).resolve().parents[1]
        / "configs/run-ready/artifacts/stage3-formal-pilot-plan-source.example.yaml"
    )
    assert main(
        [
            "artifact",
            "quadrature-pilot-plan-build",
            "--spec",
            str(source),
            "--formal-execution-evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    plan = load_canonical_json(output_path)
    assert plan["schema_version"] == "stage3-formal-pilot-plan-v1"
    assert plan["execution_evidence_hash"] == evidence.artifact_hash
    assert plan["formal_eligible"] is True
    assert main(["artifact-validate", str(output_path)]) == 0


def test_stage2_experiment_plan_builder_binds_sampling_evidence_and_lineage(
    tmp_path: Path,
) -> None:
    gate = GateRecord(
        "stage1.G1-EXIT",
        1,
        GateStatus.PASS,
        "2026-07-22T00:00:00Z",
        evidence_refs=("commits/stage1-exit.json",),
    )
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="1" * 64,
        asset_manifest_hashes=("2" * 64,),
        prerequisite_gates=(gate,),
    )
    sampling = SamplingPlan(
        SamplingUniverse("stage2-builder-universe", tuple(range(64))),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )
    evidence_path = tmp_path / "formal-evidence.json"
    sampling_path = tmp_path / "sampling.json"
    output_path = tmp_path / "stage2-plan.json"
    write_canonical_json(evidence_path, evidence.to_dict())
    write_canonical_json(sampling_path, sampling.to_dict())
    source = (
        Path(__file__).resolve().parents[1]
        / "configs/run-ready/artifacts/stage2-formal-experiment-plan-source.example.yaml"
    )
    assert main(
        [
            "artifact",
            "stage2-experiment-plan-build",
            "--spec",
            str(source),
            "--sampling-plan",
            str(sampling_path),
            "--formal-execution-evidence",
            str(evidence_path),
            "--source-ref",
            "runs/stage2-05/commits/z.json",
            "--source-ref",
            "runs/stage2-05/commits/a.json",
            "--output",
            str(output_path),
        ]
    ) == 0
    plan = FormalExperimentPlan.from_mapping(load_canonical_json(output_path))
    assert plan.sampling_plan_hash == sampling.digest
    assert plan.execution_evidence_hash == evidence.artifact_hash
    assert plan.source_artifact_refs == (
        "runs/stage2-05/commits/a.json",
        "runs/stage2-05/commits/z.json",
    )
    assert plan.batch_size == 32
    assert plan.microbatch_counts == (4, 8, 16, 32)
    assert plan.repetitions == 2
    assert main(["artifact-validate", str(output_path)]) == 0
