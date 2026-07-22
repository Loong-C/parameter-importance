from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts import (
    ContractFreeze,
    ContractState,
    FormalRunRejected,
    FreezeContractError,
    GateContractError,
    GateRecord,
    GateStatus,
    LocalValidationRecord,
    LocalValidationStatus,
    ResolvedConfig,
    canonical_json_hash,
    evaluate_formal_readiness,
    load_canonical_json,
    require_formal_readiness,
    validate_gate_id,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
CHECKED_AT = "2026-07-22T08:00:00Z"


def _local_config() -> ResolvedConfig:
    value = load_canonical_json(CONFIG_PATH)
    assert isinstance(value, dict)
    return ResolvedConfig.from_mapping(value)


def _decision_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "stage2-primary-v1",
        "scope": "formal",
        "status": "PASS",
        "state": "FROZEN",
        "selected_estimator": "u",
        "batch_size": 64,
        "microbatch_count": 16,
        "repetitions": 8,
        "gate_id": "stage2.G2.7b",
        "gate_status": "PASS",
        "artifact_ref": "decisions/stage2.json",
        "metadata": {},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def _formal_config() -> ResolvedConfig:
    value = load_canonical_json(CONFIG_PATH)
    assert isinstance(value, dict)
    value = deepcopy(value)
    value["identity"]["run_intent"] = "formal"
    value["identity"]["formal_eligible"] = True
    value["identity"]["route"] = "pretrain"
    value["runtime"]["allow_dirty_worktree"] = False
    value["importance"]["estimator_decision_ref"] = "decisions/stage2.json"
    return ResolvedConfig.from_mapping(value)


def _gate(stage: int) -> GateRecord:
    return GateRecord(
        gate_id=f"stage{stage}.G{stage}",
        stage=stage,
        status=GateStatus.PASS,
        checked_at=CHECKED_AT,
        measured={"ok": True},
        threshold={"required": True},
        evidence_refs=(f"evidence/stage{stage}.json",),
    )


def _decision_gate() -> GateRecord:
    return GateRecord(
        gate_id="stage2.G2.7b",
        stage=2,
        status=GateStatus.PASS,
        checked_at=CHECKED_AT,
        evidence_refs=("decisions/stage2.json",),
    )


def _freezes(config: ResolvedConfig) -> list[ContractFreeze]:
    return [
        ContractFreeze(
            contract_id=f"stage{stage}.contract.v1",
            stage=stage,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version=f"stage{stage}-formula-v1",
            config_hash=config.config_hash,
            schema_hashes={f"schemas/stage{stage}/contract-v1.json": DIGEST_A},
            source_hashes={f"plan/stage{stage}": DIGEST_B},
            required_gate_ids=(f"stage{stage}.G{stage}",),
            frozen_at=CHECKED_AT,
        )
        for stage in range(4)
    ]


def test_status_values_are_separate_and_complete() -> None:
    assert {status.value for status in GateStatus} == {
        "PASS",
        "CONDITIONALLY_ACCEPTED",
        "FAIL",
        "BLOCKED",
        "STALE",
        "NOT_RUN",
    }
    assert {status.value for status in LocalValidationStatus} == {
        "PASS",
        "FAIL",
        "SKIPPED",
        "NOT_RUN",
    }
    assert {state.value for state in ContractState} == {
        "FROZEN",
        "UNFROZEN",
        "BLOCKED",
        "STALE",
    }


@pytest.mark.parametrize(
    "gate_id",
    ["stage0.G1", "stage1.G1-CONTRACT", "stage2.G2.7b", "stage3.G3-0"],
)
def test_fully_qualified_gate_ids_are_accepted(gate_id: str) -> None:
    assert validate_gate_id(gate_id) == gate_id


@pytest.mark.parametrize("gate_id", ["G1", "stage10.G1", "stage1.contract", "stage1.g1"])
def test_unqualified_or_malformed_gate_ids_are_rejected(gate_id: str) -> None:
    with pytest.raises(GateContractError, match="全限定格式"):
        validate_gate_id(gate_id)


def test_gate_record_enforces_evidence_reason_and_conditional_expiry() -> None:
    with pytest.raises(GateContractError, match="至少引用"):
        GateRecord(
            gate_id="stage1.G1",
            stage=1,
            status=GateStatus.PASS,
            checked_at=CHECKED_AT,
        )

    with pytest.raises(GateContractError, match="reason"):
        GateRecord(
            gate_id="stage1.G1",
            stage=1,
            status=GateStatus.BLOCKED,
            checked_at=CHECKED_AT,
        )

    conditional = GateRecord(
        gate_id="stage0.G8",
        stage=0,
        status=GateStatus.CONDITIONALLY_ACCEPTED,
        checked_at=CHECKED_AT,
        evidence_refs=("evidence/capacity.json",),
        conditions=("仅限已批准容量范围",),
        expires_at="2026-08-01T00:00:00Z",
    )
    assert conditional.status is GateStatus.CONDITIONALLY_ACCEPTED


def test_gate_record_round_trip_hash_and_expiry() -> None:
    gate = GateRecord(
        gate_id="stage0.G1",
        stage=0,
        status=GateStatus.PASS,
        checked_at=CHECKED_AT,
        evidence_refs=("evidence/g1.json",),
        expires_at="2026-07-23T08:00:00Z",
    )
    assert GateRecord.from_mapping(gate.to_dict()) == gate
    assert gate.effective_status(
        at=datetime(2026, 7, 23, 8, 0, tzinfo=timezone.utc)
    ) is GateStatus.STALE

    tampered = gate.to_dict()
    tampered["measured"] = {"changed": True}
    with pytest.raises(GateContractError, match="artifact_hash"):
        GateRecord.from_mapping(tampered)


def test_local_validation_can_pass_but_is_never_formal() -> None:
    local = LocalValidationRecord(
        validation_id="local.contracts.cpu",
        status=LocalValidationStatus.PASS,
        checked_at=CHECKED_AT,
        checks={"json": True, "config": True},
        evidence_refs=("evidence/local-tests.json",),
    )
    payload = local.to_dict()
    assert payload["scope"] == "local_fixture"
    assert payload["formal_eligible"] is False
    assert LocalValidationRecord.from_mapping(payload) == local


def test_contract_freeze_distinguishes_local_and_formal_scope() -> None:
    local = ContractFreeze(
        contract_id="stage1.contract.v1",
        stage=1,
        scope="local_fixture",
        state=ContractState.FROZEN,
        formula_version="formula-v1",
        config_hash=DIGEST_A,
        schema_hashes={"schema": DIGEST_A},
        source_hashes={"source": DIGEST_B},
        frozen_at=CHECKED_AT,
    )
    assert local.formal_eligible is False
    assert ContractFreeze.from_mapping(local.to_dict()) == local

    blocked = ContractFreeze(
        contract_id="stage2.contract.v1",
        stage=2,
        scope="formal",
        state=ContractState.BLOCKED,
        formula_version="formula-v1",
        config_hash=DIGEST_A,
        schema_hashes={},
        source_hashes={},
        reason="server_unreachable",
    )
    assert blocked.formal_eligible is False


def test_contract_freeze_rejects_stage_mismatch_and_false_frozen_state() -> None:
    with pytest.raises(FreezeContractError, match="与 stage 一致"):
        ContractFreeze(
            contract_id="stage2.contract.v1",
            stage=1,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version="v1",
            config_hash=DIGEST_A,
            schema_hashes={"schema": DIGEST_A},
            source_hashes={"source": DIGEST_B},
            frozen_at=CHECKED_AT,
        )

    with pytest.raises(FreezeContractError, match="必须有 frozen_at"):
        ContractFreeze(
            contract_id="stage1.contract.v1",
            stage=1,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version="v1",
            config_hash=DIGEST_A,
            schema_hashes={"schema": DIGEST_A},
            source_hashes={"source": DIGEST_B},
        )

    with pytest.raises(FreezeContractError, match="required Gate"):
        ContractFreeze(
            contract_id="stage1.contract.v1",
            stage=1,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version="v1",
            config_hash=DIGEST_A,
            schema_hashes={"schema": DIGEST_A},
            source_hashes={"source": DIGEST_B},
            frozen_at=CHECKED_AT,
        )
    with pytest.raises(GateContractError, match="不一致"):
        ContractFreeze(
            contract_id="stage1.contract.v1",
            stage=1,
            scope="formal",
            state=ContractState.FROZEN,
            formula_version="v1",
            config_hash=DIGEST_A,
            schema_hashes={"schema": DIGEST_A},
            source_hashes={"source": DIGEST_B},
            required_gate_ids=("stage2.G1",),
            frozen_at=CHECKED_AT,
        )


def test_local_fixture_is_rejected_even_with_local_frozen_contracts() -> None:
    config = _local_config()
    local_freeze = ContractFreeze(
        contract_id="stage0.contract.local-v1",
        stage=0,
        scope="local_fixture",
        state=ContractState.FROZEN,
        formula_version="v1",
        config_hash=config.config_hash,
        schema_hashes={"schema": DIGEST_A},
        source_hashes={"source": DIGEST_B},
        frozen_at=CHECKED_AT,
    )
    readiness = evaluate_formal_readiness(
        config,
        freezes=[local_freeze],
        estimator_decision=_decision_payload(),
        required_stages=(0,),
        at=datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
    )
    assert readiness.formal_eligible is False
    assert "CONFIG_NOT_FORMAL" in readiness.reasons
    assert "FREEZE_STAGE_0_NOT_FORMAL_FROZEN" in readiness.reasons


def test_formal_readiness_rejects_missing_freeze_or_decision() -> None:
    config = _formal_config()
    missing_freeze = evaluate_formal_readiness(
        config,
        freezes=[],
        estimator_decision=_decision_payload(),
        required_stages=(0,),
    )
    assert "MISSING_FREEZE_STAGE_0" in missing_freeze.reasons

    missing_decision = evaluate_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=None,
        gate_records=[_gate(stage) for stage in range(4)],
    )
    assert "MISSING_ESTIMATOR_DECISION" in missing_decision.reasons
    with pytest.raises(FormalRunRejected, match="正式运行被拒绝"):
        require_formal_readiness(
            config,
            freezes=_freezes(config),
            estimator_decision=None,
            gate_records=[_gate(stage) for stage in range(4)],
        )


def test_formal_readiness_requires_all_freeze_gates_and_decision_hash() -> None:
    config = _formal_config()
    readiness = evaluate_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=_decision_payload(),
        gate_records=[_gate(stage) for stage in range(3)],
        at=datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
    )
    assert readiness.formal_eligible is False
    assert "MISSING_GATE_stage3.G3" in readiness.reasons

    decision = _decision_payload()
    decision["selected_estimator"] = "double"
    invalid_hash = evaluate_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=decision,
        gate_records=[*[_gate(stage) for stage in range(4)], _decision_gate()],
    )
    assert "ESTIMATOR_DECISION_HASH_MISMATCH" in invalid_hash.reasons


def test_complete_formal_evidence_chain_is_accepted() -> None:
    config = _formal_config()
    readiness = require_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=_decision_payload(),
        gate_records=[*[_gate(stage) for stage in range(4)], _decision_gate()],
        at=datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
    )
    assert readiness.formal_eligible is True
    assert readiness.reasons == ()
    assert readiness.verified_stages == (0, 1, 2, 3)
    assert readiness.verified_gate_ids == (
        "stage0.G0",
        "stage1.G1",
        "stage2.G2",
        "stage2.G2.7b",
        "stage3.G3",
    )


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda value: value.update({"status": "FAIL"}), "FORMAL_DECISION_STATUS_NOT_ACCEPTABLE"),
        (lambda value: value.update({"batch_size": None}), "FORMAL_DECISION_B_M_R_REQUIRED"),
        (lambda value: value.update({"microbatch_count": 10}), "FORMAL_DECISION_M_MUST_DIVIDE_B"),
        (lambda value: value.update({"artifact_ref": None}), "FORMAL_DECISION_ARTIFACT_REF_REQUIRED"),
        (lambda value: value.update({"extra": "forbidden"}), "INVALID_ESTIMATOR_DECISION_ARTIFACT"),
    ],
)
def test_formal_readiness_rejects_incomplete_or_failed_decision_wire(
    mutation, reason: str
) -> None:
    config = _formal_config()
    decision = _decision_payload()
    decision.pop("artifact_hash")
    mutation(decision)
    decision["artifact_hash"] = canonical_json_hash(decision)
    readiness = evaluate_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=decision,
        gate_records=[*[_gate(stage) for stage in range(4)], _decision_gate()],
        at=datetime(2026, 7, 22, 9, 0, tzinfo=timezone.utc),
    )
    assert readiness.formal_eligible is False
    assert reason in readiness.reasons


def test_formal_evidence_cannot_pass_with_zero_real_gates() -> None:
    config = _formal_config()
    decision = _decision_payload()
    readiness = evaluate_formal_readiness(
        config,
        freezes=_freezes(config),
        estimator_decision=decision,
        gate_records=[],
    )
    assert readiness.formal_eligible is False
    assert readiness.verified_gate_ids == ()
    assert "MISSING_GATE_stage2.G2.7b" in readiness.reasons


def test_shared_and_stage_contract_schema_documents_are_present_and_strict() -> None:
    shared = ROOT / "schemas" / "shared"
    expected_shared = {
        "canonical-artifact-v1.json",
        "resolved-config-v1.json",
        "gate-record-v1.json",
        "local-validation-record-v1.json",
        "run-identity-v1.json",
        "seed-plan-v1.json",
        "provenance-record-v1.json",
        "contract-freeze-v1.json",
        "formal-readiness-v1.json",
    }
    assert expected_shared <= {path.name for path in shared.glob("*.json")}
    for path in shared.glob("*.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["$id"].startswith("https://parameter-importance.invalid/")

    for stage in range(10):
        path = ROOT / "schemas" / f"stage{stage}" / "contract-v1.json"
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["allOf"][0]["$ref"] == "../shared/contract-freeze-v1.json"
        assert schema["allOf"][1]["properties"]["stage"]["const"] == stage


def test_local_freeze_fixture_is_canonical_hash_bound_and_never_formal() -> None:
    path = (
        ROOT
        / "configs"
        / "local-fixtures"
        / "local-contract-freezes-v1.json"
    )
    envelope = load_canonical_json(path)
    assert isinstance(envelope, dict)
    digest = envelope.pop("artifact_hash")
    assert digest == canonical_json_hash(envelope)
    assert envelope["formal_eligible"] is False
    assert len(envelope["freezes"]) == 10
    freezes = [ContractFreeze.from_mapping(value) for value in envelope["freezes"]]
    assert [freeze.stage for freeze in freezes] == list(range(10))
    assert all(not freeze.formal_eligible for freeze in freezes)
    assert {freeze.config_hash for freeze in freezes} == {_local_config().config_hash}
