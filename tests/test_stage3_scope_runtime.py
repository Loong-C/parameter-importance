from __future__ import annotations

import copy
from pathlib import Path

from param_importance_nlp.contracts import (
    GateRecord,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import load_resolved_config_compatible
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.experiments.stage3_scope_authority import (
    publish_stage3_scope_authority,
)
from param_importance_nlp.runtime import TaskArtifactStore
from param_importance_nlp.runtime.task_runtime import (
    BlockerCode,
    TaskRuntime,
    TaskRuntimeEnvironment,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
DECISION = ROOT / "reports" / "stage3" / "g3-0-user-scope-decision-20260828.json"
GATE = ROOT / "reports" / "stage3" / "g3-0-user-scope-gate-20260828.json"


def _formal_stage3_config() -> object:
    value = load_canonical_json(BASE_CONFIG)
    assert isinstance(value, dict)
    mapping = copy.deepcopy(value)
    mapping["identity"].update(  # type: ignore[union-attr]
        {
            "stage": 3,
            "run_intent": "formal",
            "formal_eligible": True,
            "route": "pretrain",
        }
    )
    mapping["runtime"].update(  # type: ignore[union-attr]
        {"allow_dirty_worktree": False, "device": "cuda"}
    )
    mapping["importance"]["estimator_decision_ref"] = "decisions/stage2.json"  # type: ignore[index]
    return load_resolved_config_compatible(
        mapping,
        task_id="stage3.01_prerequisites_and_scope",
        overrides={"orchestration": {"input_result_refs": []}},
    )


def _published_authority(tmp_path: Path) -> tuple[str, str]:
    decision_path = tmp_path / "reports/stage3/decision.json"
    gate_path = tmp_path / "reports/stage3/gate.json"
    write_canonical_json(decision_path, load_canonical_json(DECISION))
    source_gate = load_canonical_json(GATE)
    assert isinstance(source_gate, dict)
    source_gate["evidence_refs"] = ["reports/stage3/decision.json"]
    body = {key: item for key, item in source_gate.items() if key != "artifact_hash"}
    source_gate["artifact_hash"] = canonical_json_hash(body)
    write_canonical_json(gate_path, source_gate)
    publication = publish_stage3_scope_authority(
        workspace_root=tmp_path,
        output_dir="evidence/stage3/g30",
        decision_path=decision_path,
        gate_path=gate_path,
        config_hash="a" * 64,
    )
    return (
        str(publication["decision_commit_ref"]),
        str(publication["gate_commit_ref"]),
    )


def _environment(decision_ref: str, gate_ref: str) -> TaskRuntimeEnvironment:
    return TaskRuntimeEnvironment(
        passed_gate_ids=frozenset({"stage3.G3-0"}),
        evidence_refs={
            "stage3_scope_decision": decision_ref,
            "stage3_g30_gate": gate_ref,
            "gate_stage3_g3_0": gate_ref,
        },
    )


def test_stage3_preflight_accepts_only_hash_bound_user_scope_authority(
    tmp_path: Path,
) -> None:
    decision_ref, gate_ref = _published_authority(tmp_path)
    blockers = TaskRuntime(workspace_root=tmp_path).preflight(
        _formal_stage3_config(),
        environment=_environment(decision_ref, gate_ref),
    )
    assert BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE not in {
        item.code for item in blockers
    }
    assert DEFAULT_TASK_CATALOG.get(
        "stage3.01_prerequisites_and_scope"
    ).predecessor_task_ids == ()


def test_stage3_scope_authority_rejects_semantic_drift_inside_valid_envelopes(
    tmp_path: Path,
) -> None:
    decision = load_canonical_json(DECISION)
    assert isinstance(decision, dict)
    decision["accepted_stage_inputs"]["stage2"]["batch_size"] = 64  # type: ignore[index]
    decision["artifact_hash"] = canonical_json_hash(
        {key: item for key, item in decision.items() if key != "artifact_hash"}
    )
    store = TaskArtifactStore(tmp_path, "evidence/stage3/drift")
    published_decision = store.publish(
        task_id="stage3.01_prerequisites_and_scope",
        artifact_kind="scope_authority",
        config_hash="b" * 64,
        run_intent="formal",
        payload=decision,
        formal_eligible=True,
    )
    source_gate = load_canonical_json(GATE)
    assert isinstance(source_gate, dict)
    parsed = GateRecord.from_mapping(source_gate)
    rebound = GateRecord(
        gate_id=parsed.gate_id,
        stage=parsed.stage,
        status=parsed.status,
        checked_at=parsed.checked_at,
        measured=parsed.measured,
        threshold=parsed.threshold,
        evidence_refs=(published_decision.commit_ref,),
        reasons=parsed.reasons,
        conditions=parsed.conditions,
        expires_at=parsed.expires_at,
    )
    published_gate = store.publish(
        task_id="stage3.01_prerequisites_and_scope",
        artifact_kind="gate_record",
        config_hash="b" * 64,
        run_intent="formal",
        payload=rebound.to_dict(),
        formal_eligible=True,
    )
    blockers = TaskRuntime(workspace_root=tmp_path).preflight(
        _formal_stage3_config(),
        environment=_environment(
            published_decision.commit_ref,
            published_gate.commit_ref,
        ),
    )
    assert BlockerCode.ESTIMATOR_DECISION_UNAVAILABLE in {
        item.code for item in blockers
    }
