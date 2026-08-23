"""S2.2 immutable Stage 1 handoff and fixed-state contract tests."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    STAGE1_G1_EXIT_PRODUCER_COMMIT,
    Stage1HandoffError,
    canonical_json_hash,
    validate_stage1_exit_evidence,
    write_canonical_json,
)
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG
from param_importance_nlp.experiments import stage23_task_runners as stage23


ROOT = Path(__file__).resolve().parents[1]


def _with_hash(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _stage1_fixture(root: Path, *, producer: str = STAGE1_G1_EXIT_PRODUCER_COMMIT) -> str:
    evidence = root / "evidence" / "stage1-s1-11-formal"
    evidence.mkdir(parents=True)
    common = {
        "schema_version": "stage1-s1-11-formalization-index-v1",
        "status": "PASS",
        "task_id": "stage1.11_reporting_and_exit_gate",
        "gate_id": "G1-EXIT",
    }
    roles: dict[str, dict[str, object]] = {
        "formal_observation": _with_hash({
            **common,
            "schema_version": "stage1-s1-11-formal-observation-v1",
            "execution_commit": producer,
        }),
        "stage_report": _with_hash({
            **common,
            "schema_version": "stage1-s1-11-stage-report-v1",
        }),
        "delivery_manifest": _with_hash({
            **{key: value for key, value in common.items() if key != "status"},
            "schema_version": "stage1-s1-11-delivery-manifest-v1",
        }),
        "gate_summary": _with_hash({
            **common,
            "schema_version": "stage1-s1-11-gate-summary-v1",
            "unresolved_failure_count": 0,
        }),
        "requirements_matrix": _with_hash({
            **{key: value for key, value in common.items() if key != "status"},
            "schema_version": "stage1-s1-11-requirements-matrix-v1",
            "rows": [
                {"requirement_id": f"S1.11-R{number:02d}", "status": "PASS"}
                for number in range(1, 29)
            ],
        }),
        "validation": _with_hash({
            **common,
            "schema_version": "stage1-s1-11-validation-v1",
        }),
        "replay_validation": _with_hash({
            "schema_version": "stage1-s1-11-replay-validation-v1",
            "status": "PASS",
        }),
    }
    role_refs: dict[str, str] = {}
    role_hashes: dict[str, str] = {}
    for role, value in roles.items():
        ref = f"{role.replace('_', '-')}.json"
        path = evidence / ref
        write_canonical_json(path, value)
        role_refs[role] = ref
        role_hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    index = _with_hash({
        **common,
        "generator_git_commit": producer,
        "consumer_git_commit": producer,
        "next_task_ids": ["stage2", "stage3"],
        "role_refs": role_refs,
        "role_sha256": role_hashes,
    })
    index_path = evidence / "index.json"
    write_canonical_json(index_path, index)
    return index_path.relative_to(root).as_posix()


def test_formal_stage1_exit_closure_binds_exact_producer_and_all_roles(tmp_path: Path) -> None:
    ref = _stage1_fixture(tmp_path)
    evidence = validate_stage1_exit_evidence(tmp_path, ref)

    assert evidence.producer_commit == STAGE1_G1_EXIT_PRODUCER_COMMIT
    assert evidence.execution_commit == STAGE1_G1_EXIT_PRODUCER_COMMIT
    assert evidence.index_ref == ref
    assert set(dict(evidence.role_sha256)) == {
        "formal_observation", "stage_report", "delivery_manifest", "gate_summary",
        "requirements_matrix", "validation", "replay_validation",
    }


@pytest.mark.parametrize("ref", [
    "reports/stage1/cpu-evidence/stage_report.json",
    "fixtures/stage1/stage1-s111-exit.json",
])
def test_tracked_stage1_fixture_is_never_formal_evidence(tmp_path: Path, ref: str) -> None:
    with pytest.raises(Stage1HandoffError, match="TRACKED_FIXTURE"):
        validate_stage1_exit_evidence(tmp_path, ref)


def test_stage1_exit_rejects_role_hash_drift(tmp_path: Path) -> None:
    ref = _stage1_fixture(tmp_path)
    role = tmp_path / "evidence" / "stage1-s1-11-formal" / "formal-observation.json"
    write_canonical_json(role, {"schema_version": "tampered", "status": "PASS"})
    with pytest.raises(Stage1HandoffError, match="ROLE_FORMAL_OBSERVATION_FILE_HASH"):
        validate_stage1_exit_evidence(tmp_path, ref)


def test_stage1_exit_rejects_nonreleased_producer(tmp_path: Path) -> None:
    ref = _stage1_fixture(tmp_path, producer="0" * 40)
    with pytest.raises(Stage1HandoffError, match="GENERATOR_COMMIT"):
        validate_stage1_exit_evidence(tmp_path, ref)


def test_formal_execution_evidence_consumes_same_stage1_index_ref(tmp_path: Path) -> None:
    stage1_ref = _stage1_fixture(tmp_path)
    freeze = {"schema_version": "contract-freeze-test-v1", "contract": "s2.2"}
    freeze_ref = "contract-freeze.json"
    write_canonical_json(tmp_path / freeze_ref, freeze)
    execution = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=canonical_json_hash(freeze),
        asset_manifest_hashes=("a" * 64,),
        prerequisite_gates=(
            GateRecord(
                "stage2.G2.0", 2, GateStatus.PASS,
                "2026-08-23T00:00:00+00:00", evidence_refs=("g2-0.json",)
            ),
            GateRecord(
                "stage1.G1-EXIT", 1, GateStatus.PASS,
                "2026-08-23T00:00:00+00:00", evidence_refs=(stage1_ref,)
            ),
        ),
    )
    execution_ref = "formal-execution.json"
    write_canonical_json(tmp_path / execution_ref, execution.to_dict())
    task = DEFAULT_TASK_CATALOG.get(
        "stage2.02_stage1_handoff_and_fixed_state_contract"
    )
    request = SimpleNamespace(
        task=task,
        environment=SimpleNamespace(
            evidence_refs={
                "formal_execution": execution_ref,
                "contract_freeze": freeze_ref,
                "stage1_g1_exit": stage1_ref,
            },
            passed_gate_ids=("stage2.G2.0", "stage1.G1-EXIT"),
        ),
    )
    loaded, loaded_ref = stage23._formal_execution_evidence(request, tmp_path)
    assert loaded.artifact_hash == execution.artifact_hash
    assert loaded_ref == execution_ref
