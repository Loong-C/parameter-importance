from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    FormalExecutionEvidence,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.experiments.stage3_scope_authority import (
    publish_stage3_scope_authority,
)
from param_importance_nlp.experiments.stage23_task_runners import (
    _predecessor_context,
    _stage3_scope_authority,
)
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.runtime.task_artifacts import (
    TaskArtifactStore,
    load_committed_task_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "reports" / "stage3" / "g3-0-user-scope-decision-20260828.json"
GATE = ROOT / "reports" / "stage3" / "g3-0-user-scope-gate-20260828.json"


def test_g30_user_scope_decision_is_explicit_hash_bound_and_non_relabeling() -> None:
    value = load_canonical_json(DECISION)
    assert isinstance(value, dict)
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    assert canonical_json_hash(body) == value["artifact_hash"]
    assert value["authority"] == "explicit_user_direction"
    assert value["status"] == "PASS"
    assert value["downstream_effect"]["stage3_real_experiments_authorized"] is True
    assert value["accepted_stage_inputs"]["stage2"] == {
        "batch_size": 32,
        "default_estimator": "U-32",
        "run_id": "pythia-grid-20260826T145530Z",
        "sensitivity_control": "Raw",
        "source_branch": "exp/stage2-direct-all-20260826",
        "source_commit": "000ce1e79af791ce1eae2e2b62da221a10dd3c9a",
    }
    assert any("does not relabel" in claim for claim in value["non_claims"])


def test_g30_user_scope_gate_is_a_real_pass_bound_to_decision() -> None:
    value = load_canonical_json(GATE)
    assert isinstance(value, dict)
    gate = GateRecord.from_mapping(value)
    assert gate.gate_id == "stage3.G3-0"
    assert gate.status is GateStatus.PASS
    assert gate.evidence_refs == (
        "reports/stage3/g3-0-user-scope-decision-20260828.json",
    )
    assert gate.measured["stage2_estimator"] == "U-32"
    assert gate.measured["stage2_batch_size"] == 32


def test_g30_scope_authority_is_consumable_by_formal_stage3_evidence(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "reports" / "stage3"
    decision_path = source_dir / DECISION.name
    gate_path = source_dir / GATE.name
    write_canonical_json(decision_path, load_canonical_json(DECISION))
    write_canonical_json(gate_path, load_canonical_json(GATE))
    publication = publish_stage3_scope_authority(
        workspace_root=tmp_path,
        output_dir="evidence/stage3/g30",
        decision_path=decision_path,
        gate_path=gate_path,
        config_hash="a" * 64,
    )
    decision_ref = str(publication["decision_commit_ref"])
    gate_ref = str(publication["gate_commit_ref"])
    gate = GateRecord.from_mapping(
        dict(
            load_committed_task_artifact(
                tmp_path, gate_ref, require_formal=True
            ).payload
        )
    )
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="1" * 64,
        asset_manifest_hashes=("2" * 64,),
        prerequisite_gates=(gate,),
        metadata={"stage3_scope_authority_refs": [gate_ref]},
    )
    request = SimpleNamespace(
        environment=SimpleNamespace(
            evidence_refs={
                "stage3_scope_decision": decision_ref,
                "stage3_g30_gate": gate_ref,
            }
        )
    )
    decision, observed_gate, observed_decision_ref, observed_gate_ref = (
        _stage3_scope_authority(request, tmp_path, evidence=evidence)
    )
    assert decision["artifact_hash"] == load_canonical_json(DECISION)["artifact_hash"]
    assert observed_gate.artifact_hash == gate.artifact_hash
    assert (observed_decision_ref, observed_gate_ref) == (decision_ref, gate_ref)


def test_formal_stage3_entry_does_not_restore_the_retired_stage2_predecessor(
    tmp_path: Path,
) -> None:
    request = SimpleNamespace(
        config=SimpleNamespace(
            run_intent="formal",
            section=lambda name: {"input_result_refs": []}
            if name == "orchestration"
            else None,
        ),
        task=DEFAULT_TASK_CATALOG.get("stage3.01_prerequisites_and_scope"),
    )

    context = _predecessor_context(
        request,
        tmp_path,
        TaskArtifactStore(tmp_path, "artifacts/stage3-entry"),
    )

    assert context.predecessor_task_ids == ()
    assert context.artifacts == ()
