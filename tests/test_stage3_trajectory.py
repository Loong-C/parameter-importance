"""Focused contracts for real Stage 3 endpoint trajectory production."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from param_importance_nlp.experiments import (
    STAGE3_ENDPOINT_TASK_ID,
    Stage3TrajectoryReceipt,
    TrainingEndpointObserver,
)
from param_importance_nlp.providers import build_tiny_training_fixture
from param_importance_nlp.runtime import TrainingEngine, TrainingRunSpec


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_real_observer_persists_plan_metadata_and_true_step_diagnostics(tmp_path: Path) -> None:
    fixture = build_tiny_training_fixture(
        task_type="sequence_classification", seed=391, steps=1
    )
    optimizer = torch.optim.SGD(fixture.model.module.parameters(), lr=0.05)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "stage3-trajectory-diagnostics",
            "local_fixture",
            max_steps=1,
            max_attempts=1,
            weights_exogenous=True,
            common_mean_assumption=True,
        ),
        model=fixture.model,
        optimizer=optimizer,
        cursor=fixture.dataset.cursor(seed=391),
    )
    observer = TrainingEndpointObserver(
        source_run_id="stage3-trajectory-diagnostics",
        parameter_registry_hash=engine.registry.coordinate_registry_hash,
        selected_steps={1},
        output_root=tmp_path / "endpoints",
        workspace_root=tmp_path,
        scope="pilot",
        endpoint_metadata={1: {"model": "14M", "seed": 0, "stage": "early"}},
    )
    observer.bind_engine(engine)
    engine.register_observer(observer)
    assert engine.run().status == "COMPLETE"

    object_path = tmp_path / "endpoints" / "objects" / (
        "stage3-trajectory-diagnostics-step-00000001.json"
    )
    metadata = json.loads(object_path.read_text(encoding="utf-8"))["record"]["metadata"]
    assert metadata["model"] == "14M"
    assert metadata["seed"] == 0
    assert metadata["stage"] == "early"
    diagnostics = metadata["step_diagnostics"]
    assert diagnostics["total_update_delta_hash"] == metadata["full_update_delta_hash"]
    assert diagnostics["raw_gradient_norm"] >= diagnostics["applied_optimizer_gradient_norm"]
    assert diagnostics["learning_rate_identity"]


def test_real_observer_rejects_missing_plan_metadata(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ENDPOINT_METADATA_PLAN_COVERAGE_DRIFT"):
        TrainingEndpointObserver(
            source_run_id="missing-plan-metadata",
            parameter_registry_hash=_hash("registry"),
            selected_steps={1, 2},
            output_root=tmp_path / "endpoints",
            endpoint_metadata={1: {"model": "14M", "seed": 0, "stage": "early"}},
            scope="formal",
            formal_eligible=True,
            qualification_evidence_hash=_hash("qualification"),
        )


def test_trajectory_receipt_is_hash_bound_and_never_a_stage3_task_artifact() -> None:
    receipt = Stage3TrajectoryReceipt(
        receipt_id="stage3-trajectory-receipt",
        task_id=STAGE3_ENDPOINT_TASK_ID,
        config_hash=_hash("config"),
        purpose_scope="pilot",
        formal_eligible=False,
        capture_plan_ref="inputs/capture-plan.json",
        capture_plan_hash=_hash("plan"),
        training_run_id="stage3-real-run",
        selected_steps=(1,),
        endpoint_commit_refs=("runs/endpoints/commits/step-1.json",),
        endpoint_digests=(_hash("endpoint"),),
        replay_verified_steps=(1,),
        estimator_authority_ref="inputs/estimator-decision.json",
        g30_scope_decision_ref="inputs/g30-decision.json",
        g30_gate_hash=_hash("g30"),
    )
    restored = Stage3TrajectoryReceipt.from_mapping(receipt.to_dict())
    assert restored.artifact_hash == receipt.artifact_hash
    assert restored.formal_eligible is False
    assert restored.purpose_scope == "pilot"
