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
    STAGE1_G1_EXIT_CANONICAL_ROOT,
    Stage1HandoffError,
    STAGE0_HANDOFF_ROLES,
    STAGE0_G10_GENERATOR_COMMIT,
    Stage0HandoffError,
    validate_stage0_handoff,
    canonical_json_hash,
    load_canonical_json,
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
    evidence = root / STAGE1_G1_EXIT_CANONICAL_ROOT / producer / "s1-11-r4-20260821"
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


def _stage0_fixture(
    workspace: Path,
    *,
    data_root: Path | None = None,
    status: str = "READY",
) -> str:
    """Create a small role/hash fixture with separate workspace/data roots."""

    authority_root = (
        "evidence/stage0/g10-final/"
        f"{STAGE0_G10_GENERATOR_COMMIT}/"
        "b4974a642168994eec7d62ba38a453fa3834ee50201da55d9549b1080a5b90f0"
    )
    source_root = (data_root or workspace) / "evidence" / "stage0" / "fixture"
    source_root.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict[str, object]] = {}
    for role in STAGE0_HANDOFF_ROLES:
        source = source_root / f"{role}.json"
        write_canonical_json(source, {"role": role, "status": "PASS"})
        roles[role] = {
            "ref": source.relative_to(data_root or workspace).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "producer_commit": "bff18458c02bfde8ee3610cede0addef3ad93782",
            "accepted_at": "2026-08-23T00:00:00+00:00",
            "status": "PASS",
        }
    manifest = _with_hash({
        "schema_version": "stage0-handoff-manifest-v1",
        "status": status,
        "producer_commit": "bff18458c02bfde8ee3610cede0addef3ad93782",
        "execution_commit": "bff18458c02bfde8ee3610cede0addef3ad93782",
        "consumer_commit": "af12dd21b2b6ce97f3e321a94010c3cfb6346a03",
        "accepted_at": "2026-08-23T00:00:00+00:00",
        "authority": {
            "kind": "canonical_historical_role_index",
            "root": authority_root,
            "generator_commit": STAGE0_G10_GENERATOR_COMMIT,
            "temporary_authority_forbidden": True,
            "forbidden_roots": ["tmp/stage0", "reports/stage0", "fixtures/stage0"],
        },
        "roles": roles,
        "storage_cache": {
            "data_root": str((data_root or workspace).resolve()),
            "cache_root": "${DATA_ROOT}/cache",
            "tmp_root": "${DATA_ROOT}/tmp/stage2/<run-id>",
            "home_unchanged": True,
            "environment_variables": {
                "HF_HOME": "${DATA_ROOT}/cache/huggingface",
                "HF_DATASETS_CACHE": "${DATA_ROOT}/cache/huggingface/datasets",
                "TORCH_HOME": "${DATA_ROOT}/cache/torch",
                "XDG_CACHE_HOME": "${DATA_ROOT}/cache/xdg",
                "TEMP": "${DATA_ROOT}/tmp/stage2/<run-id>",
                "TMP": "${DATA_ROOT}/tmp/stage2/<run-id>",
                "TMPDIR": "${DATA_ROOT}/tmp/stage2/<run-id>",
                "PYTHONPYCACHEPREFIX": "${DATA_ROOT}/tmp/stage2/<run-id>/pycache",
            },
        },
        "execution_semantics": {"four_gpu_ddp": "no_sync_last_microbatch_only"},
        "performance_fields": {
            "required": ["wall_clock_seconds", "peak_memory_bytes", "samples", "tokens", "failure_status"],
            "source_role": "performance",
            "historical_schema_status": "VERIFIED",
        },
        "persistence": {"status": "VALID", "reauthorization": "NOT_REQUIRED"},
        "hardware_validity": {
            "status": "VALID",
            "observed_at": "2026-08-23",
            "excluded_devices": [{"index": 1, "pci": "0000:50:00.0"}],
            "healthy_four_card_smoke": "PASS",
        },
        "blockers": [],
    })
    ref = "reports/stage2/s2.2/stage0-fixture.json"
    path = workspace / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(path, manifest)
    return ref


def test_stage0_handoff_validates_roles_and_separate_data_root(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    data_root = tmp_path / "data-root"
    workspace.mkdir()
    data_root.mkdir()
    ref = _stage0_fixture(workspace, data_root=data_root)
    evidence = validate_stage0_handoff(
        workspace, ref, require_ready=True, evidence_root=data_root
    )
    assert evidence.status == "READY"
    assert evidence.hardware_validity == "VALID"
    assert len(evidence.roles) == len(STAGE0_HANDOFF_ROLES)


def test_stage0_handoff_rejects_wrong_data_root(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    data_root = tmp_path / "data-root"
    workspace.mkdir()
    data_root.mkdir()
    ref = _stage0_fixture(workspace, data_root=data_root)
    with pytest.raises(Stage0HandoffError, match="SOURCE_MISSING"):
        validate_stage0_handoff(workspace, ref, evidence_root=tmp_path / "wrong")


def test_stage0_handoff_rejects_tmp_role_authority(tmp_path: Path) -> None:
    ref = _stage0_fixture(tmp_path)
    path = tmp_path / ref
    value = load_canonical_json(path)
    assert isinstance(value, dict)
    roles = value["roles"]
    assert isinstance(roles, dict)
    roles["single_gpu"]["ref"] = "tmp/stage0/single_gpu.json"  # type: ignore[index]
    value = _with_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    write_canonical_json(path, value)
    with pytest.raises(Stage0HandoffError, match="CANONICAL_ROOT_REQUIRED"):
        validate_stage0_handoff(tmp_path, ref)


def test_current_stage0_manifest_is_blocked_by_expiry_and_hardware() -> None:
    ref = "reports/stage2/s2.2/stage0-handoff-manifest.json"
    with pytest.raises(Stage0HandoffError, match="RISK_ACCEPTANCE_EXPIRED"):
        validate_stage0_handoff(ROOT, ref, require_ready=True)


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


def test_tmp_stage1_attempt_is_not_formal_authority(tmp_path: Path) -> None:
    with pytest.raises(Stage1HandoffError, match="TEMP_REF_FORBIDDEN"):
        validate_stage1_exit_evidence(
            tmp_path,
            "tmp/stage1-s1-11/s1-11-r4-20260821/index.json",
        )


def test_noncanonical_stage1_root_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(Stage1HandoffError, match="CANONICAL_ROOT_REQUIRED"):
        validate_stage1_exit_evidence(
            tmp_path,
            "evidence/stage1-s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r4-20260821/index.json",
        )


def test_stage1_exit_rejects_role_hash_drift(tmp_path: Path) -> None:
    ref = _stage1_fixture(tmp_path)
    role = (
        tmp_path
        / STAGE1_G1_EXIT_CANONICAL_ROOT
        / STAGE1_G1_EXIT_PRODUCER_COMMIT
        / "s1-11-r4-20260821"
        / "formal-observation.json"
    )
    write_canonical_json(role, {"schema_version": "tampered", "status": "PASS"})
    with pytest.raises(Stage1HandoffError, match="ROLE_FORMAL_OBSERVATION_FILE_HASH"):
        validate_stage1_exit_evidence(tmp_path, ref)


def test_stage1_exit_rejects_nonreleased_producer(tmp_path: Path) -> None:
    ref = _stage1_fixture(tmp_path, producer="0" * 40)
    with pytest.raises(Stage1HandoffError, match="CANONICAL_PRODUCER_PATH"):
        validate_stage1_exit_evidence(tmp_path, ref)


def test_formal_execution_evidence_consumes_same_stage1_index_ref(tmp_path: Path) -> None:
    stage1_ref = _stage1_fixture(tmp_path)
    stage0_ref = _stage0_fixture(tmp_path)
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
                "stage0_handoff": stage0_ref,
            },
            passed_gate_ids=("stage2.G2.0", "stage1.G1-EXIT"),
        ),
    )
    loaded, loaded_ref = stage23._formal_execution_evidence(request, tmp_path)
    assert loaded.artifact_hash == execution.artifact_hash
    assert loaded_ref == execution_ref
