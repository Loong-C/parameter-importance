from __future__ import annotations

from pathlib import Path

from ops.stage2.materialize_s204 import _load_formal_contract_freeze
from ops.stage2.produce_s202_contract_freeze import (
    REQUIRED_GATE_IDS,
    produce_s202_contract_freeze,
)
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.runtime import load_committed_task_artifact


def _gate(gate_id: str, stage: int, ref: str) -> GateRecord:
    return GateRecord(
        gate_id=gate_id,
        stage=stage,
        status=GateStatus.PASS,
        checked_at="2026-08-23T00:00:00+00:00",
        evidence_refs=(ref,),
    )


def test_s202_producer_publishes_unique_freeze_and_append_only_evidence(tmp_path: Path) -> None:
    root = tmp_path / "data-root"
    root.mkdir()
    source_dir = tmp_path / "sources"
    source_dir.mkdir()
    sources: dict[str, Path] = {
        "evidence/stage2/s202-formal-auth/contract-freeze-s202.json": source_dir / "contract.json",
        "evidence/stage2/s202-formal-auth/source-snapshots/plan.md": source_dir / "plan.md",
    }
    write_canonical_json(
        sources[next(iter(sources))],
        {
            "schema_version": "stage2-s202-formal-contract-freeze-v1",
            "stage": 2,
            "scope": "formal",
            "authorization_ref": "evidence/auth.json",
        },
    )
    sources["evidence/stage2/s202-formal-auth/source-snapshots/plan.md"].write_text(
        "Stage 2 formal plan\n", encoding="utf-8"
    )
    config = source_dir / "base.json"
    write_canonical_json(
        config,
        {"identity": {"stage": 2, "run_intent": "formal", "formal_eligible": True}},
    )
    schema = source_dir / "schema.json"
    write_canonical_json(schema, {"$schema": "https://json-schema.org/draft/2020-12/schema"})
    formal_ref = "evidence/stage2/s202-formal-auth/formal-execution-s202.json"
    parent = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="e" * 64,
        asset_manifest_hashes=("a" * 64,),
        prerequisite_gates=(
            _gate("stage0.G10", 0, "evidence/g10.json"),
            _gate("stage1.G1-EXIT", 1, "evidence/g1.json"),
        ),
        metadata={"parent": "s202"},
    )
    write_canonical_json(root / formal_ref, parent.to_dict())

    result = produce_s202_contract_freeze(
        data_root=root,
        source_files=sources,
        schema_files={"schemas/shared/contract-freeze-v1.json": schema},
        base_config=config,
        formal_execution_ref=formal_ref,
        frozen_at="2026-08-23T17:00:00+00:00",
    )
    loaded = load_committed_task_artifact(root, result.contract_commit_ref, require_formal=True)
    assert loaded.payload["schema_version"] == "contract-freeze-v1"
    loaded_ref, loaded_freeze = _load_formal_contract_freeze(
        root, result.contract_commit_ref, stage=2
    )
    assert loaded_ref == result.contract_commit_ref
    assert loaded_freeze == result.freeze
    assert result.freeze.formal_eligible
    assert "stage2.G2.0" in REQUIRED_GATE_IDS
    amended = FormalExecutionEvidence.from_mapping(
        load_canonical_json(root / result.amended_evidence_ref)
    )
    assert amended.contract_freeze_hash == result.freeze.artifact_hash
    assert amended.metadata["contract_freeze_commit_ref"] == result.contract_commit_ref
    assert load_canonical_json(root / formal_ref) == parent.to_dict()
