from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.status import GateStatus
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.sampling import RepetitionMapping, SamplingPlan, SamplingUniverse, STREAM_NAMES
from param_importance_nlp.experiments.stage2_g22_adapter import (
    ARTIFACT_KINDS,
    G22Blocked,
    _gate,
    _validate_task_inputs,
    evaluate_formal_g22,
)
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def _formal_fixture(root: Path) -> dict[str, str]:
    store = TaskArtifactStore(root, "task_outputs/s203")
    plan = SamplingPlan(
        universe=SamplingUniverse(
            universe_id="formal-fixture-universe",
            sample_ids=tuple(range(524288)),
            metadata={"fixture": True},
        ),
        stream_seeds={name: 101 + index for index, name in enumerate(STREAM_NAMES)},
    )
    streams = {name: plan.draw_manifest(name, 4).to_manifest() for name in STREAM_NAMES}
    rows = [draw.to_manifest() for name in STREAM_NAMES for draw in plan.draws(name, 4)]
    mapping = RepetitionMapping.create(
        repetition_id="stage2-formal-sampling-nested-fixture",
        draws=plan.draws("pilot", 8),
        m_values=(2, 4, 8),
    )
    payloads = {
        "sampling_plan": plan.to_dict(),
        "draw_manifest": {
            "schema_version": "stage2-task-draw-manifest-v1",
            "sampling_plan_hash": plan.digest,
            "draws": rows,
            "draw_count_by_stream": {name: 4 for name in STREAM_NAMES},
            "stream_manifests": streams,
            "draw_id_unique": True,
            "sample_id_collisions_allowed": True,
            "replay_hash": canonical_json_hash(rows),
            "nested_mapping": mapping.to_dict(),
            "nested_mapping_hash": mapping.digest,
        },
        "asset_resolution": {
            "schema_version": "stage2-task-asset-resolution-v1",
            "provider": {"provider_kind": "fixture"},
            "stage2_asset_manifest": {"schema_version": "stage2-asset-resolution-v1"},
            "preregistration_contract_hash": "a" * 64,
            "upstream_binding_hash": "b" * 64,
            "formal_eligible": False,
        },
        "gate_record": {
            "schema_version": "stage23-task-gate-candidate-v1",
            "task_id": "stage2.03_assets_checkpoints_and_sampling",
            "gate_ids": ["stage2.G2.1"],
            "gate_status": "NOT_RUN",
            "local_validation_status": "NOT_RUN",
            "formal_eligible": False,
            "reason": "formal_gate_requires_independent_review",
        },
    }
    refs: dict[str, str] = {}
    for kind in ARTIFACT_KINDS:
        refs[kind] = store.publish(
            task_id="stage2.03_assets_checkpoints_and_sampling",
            artifact_kind=kind,
            config_hash="c" * 64,
            run_intent="formal",
            formal_eligible=True,
            payload=payloads[kind],
            source_refs=("commits/stage2.01.json",),
        ).commit_ref
    return refs


def test_formal_store_fixture_is_consumed_but_missing_authority_blocks(tmp_path: Path) -> None:
    refs = _formal_fixture(tmp_path)
    result = evaluate_formal_g22(
        repository_root=Path(__file__).parents[1],
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
    )
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert result["commit_ref"] is None


def test_candidate_or_tampered_formal_input_cannot_be_promoted(tmp_path: Path) -> None:
    refs = _formal_fixture(tmp_path)
    with pytest.raises(G22Blocked):
        _validate_task_inputs(tmp_path, {**refs, "gate_record": "task_outputs/s203/commits/missing.json"})
    blocked = _gate(
        status=GateStatus.BLOCKED,
        checked_at="2026-01-01T00:00:00Z",
        measured={"fixture": True},
        refs=(),
        reasons=("formal evidence absent",),
    )
    assert blocked.status is GateStatus.BLOCKED
