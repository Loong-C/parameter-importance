from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from param_importance_nlp.contracts import (
    IdentityContractError,
    ProvenanceContractError,
    ProvenanceRecord,
    ProvenanceStatus,
    RunIdentity,
    SeedContractError,
    SeedPlan,
    derive_experiment_id,
    derive_seed,
)


DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
COMMIT = "c" * 40
START = datetime(2026, 7, 22, 8, 0, tzinfo=timezone.utc)


def _identity() -> RunIdentity:
    experiment_id = derive_experiment_id(
        stage=2,
        task="estimator validation",
        model_identity="synthetic:model:v1",
        route="local-fixture",
        master_seed=1337,
        config_hash=DIGEST_A,
    )
    return RunIdentity.create(
        experiment_id=experiment_id,
        created_at=START,
        collision_code="0123abcd",
    )


def _provenance(*, clean: bool = True, scope: str = "local_fixture") -> ProvenanceRecord:
    dirty = {} if clean else {
        "dirty_base_commit": COMMIT,
        "dirty_patch_ref": "evidence/local.patch",
        "dirty_patch_hash": DIGEST_B,
    }
    return ProvenanceRecord(
        identity=_identity(),
        config_hash=DIGEST_A,
        resolved_config_ref="configs/resolved.json",
        seed_plan_hash=DIGEST_B,
        git_commit=COMMIT,
        git_branch="feat/contracts",
        worktree_clean=clean,
        scope=scope,
        environment_id="windows-cpu-v1",
        hardware_snapshot_ref="evidence/hardware.json",
        device_mapping=("cpu",),
        model_manifest_id="synthetic:model:v1",
        data_manifest_id="synthetic:data:v1",
        started_at="2026-07-22T08:00:00Z",
        ended_at="2026-07-22T08:01:00Z",
        status=ProvenanceStatus.COMPLETED,
        artifact_refs=("evidence/tests.json",),
        **dirty,
    )


def test_local_provenance_never_claims_formal_eligibility() -> None:
    assert _provenance(clean=True).formal_eligible is False
    assert _provenance(clean=True, scope="formal").formal_eligible is True


def test_experiment_id_is_stable_and_semantic() -> None:
    arguments = {
        "stage": 2,
        "task": "Estimator Validation",
        "model_identity": "synthetic:model:v1",
        "route": "local-fixture",
        "master_seed": 1337,
        "config_hash": DIGEST_A,
    }
    first = derive_experiment_id(**arguments)
    second = derive_experiment_id(**dict(reversed(list(arguments.items()))))
    changed = derive_experiment_id(**{**arguments, "master_seed": 1338})

    assert first == second
    assert first.startswith("exp-s2-estimator-validation-")
    assert changed != first


def test_run_identity_preserves_run_and_advances_attempt_on_resume() -> None:
    first = _identity()
    second = first.next_attempt(
        started_at=START + timedelta(minutes=5),
        input_checkpoint_id="checkpoint:step100:sha256",
    )

    assert first.run_id == second.run_id
    assert second.attempt_id == 2
    assert second.session_id.endswith(".a0002.s001")
    assert second.input_checkpoint_id == "checkpoint:step100:sha256"
    assert RunIdentity.from_mapping(second.to_dict()) == second


def test_run_identity_rejects_session_mismatch_and_time_reversal() -> None:
    identity = _identity()
    with pytest.raises(IdentityContractError, match="session_id"):
        RunIdentity(
            **{
                **identity.payload_dict(),
                "session_id": f"{identity.run_id}.a0002.s001",
            }
        )

    with pytest.raises(IdentityContractError, match="不能早于"):
        identity.next_attempt(
            started_at=START - timedelta(seconds=1),
            input_checkpoint_id="checkpoint:valid",
        )


def test_run_identity_hash_tamper_is_detected() -> None:
    payload = _identity().to_dict()
    payload["attempt_id"] = 2
    payload["session_id"] = payload["session_id"].replace("a0001", "a0002")
    with pytest.raises(IdentityContractError, match="artifact_hash"):
        RunIdentity.from_mapping(payload)


def test_seed_plan_is_deterministic_and_domains_are_independent() -> None:
    first = SeedPlan.from_master_seed(1337, world_size=4)
    second = SeedPlan.from_master_seed(1337, world_size=4)

    assert first == second
    assert len(set(first.domains.values())) == len(first.domains)
    assert first.seed_for("model_init") != first.seed_for("sampler")
    assert first.seed_for("reference_A") != first.seed_for("reference_B")
    assert len(set(first.rank_training.values())) == 4
    assert SeedPlan.from_mapping(first.to_dict()) == first


def test_changing_one_seed_namespace_does_not_change_another() -> None:
    root = 1337
    model_seed = derive_seed(root, "model_init")
    sampler_epoch_1 = derive_seed(root, "sampler", 1)
    sampler_epoch_2 = derive_seed(root, "sampler", 2)

    assert model_seed == derive_seed(root, "model_init")
    assert sampler_epoch_1 != sampler_epoch_2
    assert model_seed != sampler_epoch_1


def test_seed_plan_rejects_unknown_domain_and_hash_tamper() -> None:
    plan = SeedPlan.from_master_seed(7)
    with pytest.raises(SeedContractError, match="未知"):
        plan.seed_for("not_registered")

    payload = plan.to_dict()
    payload["domains"]["sampler"] += 1
    with pytest.raises(SeedContractError):
        SeedPlan.from_mapping(payload)


def test_provenance_clean_and_dirty_smoke_have_distinct_formal_eligibility() -> None:
    clean = _provenance(clean=True, scope="formal")
    dirty = _provenance(clean=False, scope="formal")

    assert clean.formal_eligible is True
    assert dirty.formal_eligible is False
    assert ProvenanceRecord.from_mapping(clean.to_dict()) == clean
    assert ProvenanceRecord.from_mapping(dirty.to_dict()) == dirty


def test_dirty_provenance_requires_complete_patch_evidence() -> None:
    valid = _provenance(clean=False)
    values = valid.payload_dict()
    values.pop("formal_eligible")
    values["dirty_patch_hash"] = None
    values["identity"] = valid.identity
    values["device_mapping"] = tuple(values["device_mapping"])
    values["artifact_refs"] = tuple(values["artifact_refs"])

    with pytest.raises(ProvenanceContractError, match="必须同时记录"):
        ProvenanceRecord(**values)


def test_formal_provenance_rejects_urls_and_started_terminal_mismatch() -> None:
    valid = _provenance()
    values = valid.payload_dict()
    values.pop("formal_eligible")
    values["identity"] = valid.identity
    values["device_mapping"] = tuple(values["device_mapping"])
    values["artifact_refs"] = tuple(values["artifact_refs"])
    values["resolved_config_ref"] = "https://example.invalid/signed?token=secret"
    with pytest.raises(ProvenanceContractError, match="不能是 URL"):
        ProvenanceRecord(**values)

    values["resolved_config_ref"] = "configs/resolved.json"
    values["status"] = ProvenanceStatus.STARTED
    with pytest.raises(ProvenanceContractError, match="不能有 ended_at"):
        ProvenanceRecord(**values)
