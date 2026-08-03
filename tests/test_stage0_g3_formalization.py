"""G3 materialization must be replayed before it can unlock formal S0.4."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements
from param_importance_nlp.atomic import sha256_file
from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.experiments import stage01_task_runners as stage01
from param_importance_nlp.g3_gate import GATE_IDS, g3_resolution_artifact_hash
from param_importance_nlp.runtime import (
    TaskExecutionRequest,
    TaskRunResult,
    TaskRuntimeEnvironment,
    load_committed_task_artifact,
)
from param_importance_nlp.experiments.task_runners import (
    DistributedTrainingTaskRunner,
    TrainingTaskRunner,
)
from param_importance_nlp.stage0_bootstrap import (
    Stage0RuntimeSnapshot,
    Stage0SourceBinding,
    bootstrap_formal_stage0,
)
from param_importance_nlp import stage0_g3_formalization as formalization
from param_importance_nlp import stage0_g4
from param_importance_nlp import stage0_g5
from param_importance_nlp import stage0_g6
from param_importance_nlp.stage0_g3_formalization import (
    Stage0G3FormalizationError,
    formalize_stage0_g3,
    load_and_replay_g3_materialization,
)
from param_importance_nlp.storage import DATA_ROOT_ENV


ROOT = Path(__file__).resolve().parents[1]
HEAD = "a" * 40
CHECKED_AT = "2026-08-03T16:00:00Z"
ACQUISITION_SHA = hashlib.sha256(b"acquisition").hexdigest()
VERIFICATION_SHA = hashlib.sha256(b"verification").hexdigest()
ACQUISITION_REF = f"manifests/evidence/g3/acquisition/{ACQUISITION_SHA}.json"
VERIFICATION_REF = f"manifests/evidence/g3/verification/{VERIFICATION_SHA}.json"
GPU_UUIDS = (
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _binding() -> Stage0SourceBinding:
    return Stage0SourceBinding(ROOT, HEAD, "feat/stage0-completion", True)


def _snapshot(tmp_path: Path) -> Stage0RuntimeSnapshot:
    return Stage0RuntimeSnapshot(
        checked_at=CHECKED_AT,
        hostname="sophgo13",
        boot_id="04326255-b422-4f1e-8dc6-9c3cc8f0a5b9",
        kernel="6.8.0-136-generic",
        data_root=tmp_path.as_posix(),
        python_prefix=(
            "/home/sophgo13/cjl/storage/parameter-importance/envs/"
            "parameter-importance-stage0-1bd963c65f75"
        ),
        python_version="3.12.3",
        torch_version="2.12.1+cu126",
        torch_cuda_runtime="12.6",
        cuda_device_count=4,
        allowed_gpu_uuids=GPU_UUIDS,
        git_verified=True,
        server_verified=True,
        wheelhouse_verified=True,
        cuda_verified=True,
        nccl_verified=True,
    )


def _resolution() -> dict[str, object]:
    requirements = load_stage0_asset_requirements(
        ROOT / "configs/stage0/g3-asset-requirements-v1.json"
    )
    layout = load_stage0_asset_layout(
        ROOT / "configs/stage0/g3-asset-layout-v1.json",
        requirements=requirements,
    )
    entries: list[dict[str, object]] = []
    for index, item in enumerate(layout["entries"]):
        candidate_id = _digest(f"candidate:{index}")
        kind = str(item["kind"])
        entries.append(
            {
                "logical_name": item["logical_name"],
                "kind": kind,
                "requirement_name": item["requirement_name"],
                "gate_ids": list(item["gate_ids"]),
                "manifest_ref": item["manifest_ref"],
                "asset_root_ref": item["asset_root_ref"],
                "qualification_ref": item["qualification_ref"],
                "status": "PASS",
                "checks": {"qualified_resolution": True},
                "reasons": [],
                "asset_id": _digest(f"asset:{index}"),
                "candidate_id": candidate_id,
                "candidate_ref": (
                    f"manifests/candidates/g3/{item['logical_name']}/{candidate_id}.json"
                ),
                "candidate_sha256": _digest(f"candidate-file:{index}"),
                "ready_manifest_sha256": _digest(f"ready:{index}"),
                "qualification_artifact_hash": _digest(f"qualification:{index}"),
                "acquisition_ref": ACQUISITION_REF,
                "acquisition_sha256": ACQUISITION_SHA,
                "verification_ref": VERIFICATION_REF,
                "verification_sha256": VERIFICATION_SHA,
                "semantic_evidence_ref": f"manifests/evidence/g3/semantic/{index}.json",
                "semantic_evidence_sha256": _digest(f"semantic-file:{index}"),
                "semantic_evidence_artifact_hash": _digest(f"semantic:{index}"),
                "files_checked": 1,
                "bytes_checked": index + 1,
                "expected_file_policy": (
                    "qualification_bound_derived_inventory"
                    if kind == "glue_derived"
                    else "requirements_exact"
                ),
            }
        )
    gates = [
        GateRecord(
            gate_id=gate_id,
            stage=0,
            status=GateStatus.PASS,
            checked_at=CHECKED_AT,
            measured={"status": "PASS"},
            threshold={"required_status": "PASS"},
            evidence_refs=(layout["requirements_ref"],),
        ).to_dict()
        for gate_id in GATE_IDS
    ]
    value: dict[str, object] = {
        "schema_version": "stage0-g3-resolution-audit-v1",
        "scope": "formal",
        "status": "PASS",
        "checked_at": CHECKED_AT,
        "requirements_ref": layout["requirements_ref"],
        "requirements_artifact_hash": requirements["artifact_hash"],
        "layout_artifact_hash": layout["artifact_hash"],
        "entries": entries,
        "gates": gates,
    }
    value["artifact_hash"] = g3_resolution_artifact_hash(value)
    return value


def _with_hash(value: dict[str, object]) -> dict[str, object]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(result)
    return result


def _write_materialization(tmp_path: Path) -> tuple[str, dict[str, object]]:
    resolution = _resolution()
    resolution_hash = str(resolution["artifact_hash"])
    directory = Path("reports/stage0/g3") / resolution_hash
    resolution_ref = (directory / "asset-resolution.json").as_posix()
    audit_ref = (directory / "asset-audit.json").as_posix()
    index_ref = (directory / "asset-index.json").as_posix()
    write_canonical_json(tmp_path / resolution_ref, resolution)
    resolution_sha = sha256_file(tmp_path / resolution_ref)

    source_binding = {
        "head_commit": HEAD,
        "requirements_ref": "configs/stage0/g3-asset-requirements-v1.json",
        "requirements_file_sha256": sha256_file(
            ROOT / "configs/stage0/g3-asset-requirements-v1.json"
        ),
        "layout_ref": "configs/stage0/g3-asset-layout-v1.json",
        "layout_file_sha256": sha256_file(
            ROOT / "configs/stage0/g3-asset-layout-v1.json"
        ),
        "download_plan_ref": "configs/stage0/g3-download-plan-v1.json",
        "download_plan_file_sha256": sha256_file(
            ROOT / "configs/stage0/g3-download-plan-v1.json"
        ),
    }
    entries = resolution["entries"]
    assert isinstance(entries, list)
    audit = _with_hash(
        {
            "schema_version": "stage0-g3-materialization-audit-v3",
            "status": "PASS",
            "checked_at": CHECKED_AT,
            "generator_git_commit": HEAD,
            "source_binding": source_binding,
            "requirements_artifact_hash": resolution["requirements_artifact_hash"],
            "layout_artifact_hash": resolution["layout_artifact_hash"],
            "acquisition_ref": ACQUISITION_REF,
            "acquisition_sha256": ACQUISITION_SHA,
            "verification_ref": VERIFICATION_REF,
            "verification_sha256": VERIFICATION_SHA,
            "publication_count": 13,
            "publications": [
                {
                    "logical_name": item["logical_name"],
                    "kind": item["kind"],
                    "asset_id": item["asset_id"],
                    "candidate_id": item["candidate_id"],
                    "state": "ready",
                    "manifest_ref": item["manifest_ref"],
                    "candidate_ref": item["candidate_ref"],
                    "qualification_ref": item["qualification_ref"],
                    "verification_ref": item["verification_ref"],
                    "semantic_evidence_ref": item["semantic_evidence_ref"],
                }
                for item in entries
            ],
            "gate_ids": list(GATE_IDS),
            "resolution_ref": resolution_ref,
            "resolution_sha256": resolution_sha,
            "resolution_artifact_hash": resolution_hash,
        }
    )
    write_canonical_json(tmp_path / audit_ref, audit)
    audit_sha = sha256_file(tmp_path / audit_ref)
    index = _with_hash(
        {
            "schema_version": "stage0-g3-materialization-index-v2",
            "status": "PASS",
            "checked_at": CHECKED_AT,
            "generator_git_commit": HEAD,
            "source_git_commit": HEAD,
            "requirements_ref": "configs/stage0/g3-asset-requirements-v1.json",
            "requirements_artifact_hash": resolution["requirements_artifact_hash"],
            "layout_ref": "configs/stage0/g3-asset-layout-v1.json",
            "layout_artifact_hash": resolution["layout_artifact_hash"],
            "download_plan_ref": "configs/stage0/g3-download-plan-v1.json",
            "acquisition_ref": ACQUISITION_REF,
            "acquisition_sha256": ACQUISITION_SHA,
            "verification_ref": VERIFICATION_REF,
            "verification_sha256": VERIFICATION_SHA,
            "entry_count": 13,
            "entries": [
                {
                    "logical_name": item["logical_name"],
                    "kind": item["kind"],
                    "requirement_name": item["requirement_name"],
                    "asset_id": item["asset_id"],
                    "candidate_id": item["candidate_id"],
                    "manifest_ref": item["manifest_ref"],
                    "ready_manifest_sha256": item["ready_manifest_sha256"],
                    "acquisition_ref": item["acquisition_ref"],
                    "acquisition_sha256": item["acquisition_sha256"],
                    "verification_ref": item["verification_ref"],
                    "verification_sha256": item["verification_sha256"],
                }
                for item in entries
            ],
            "audit_ref": audit_ref,
            "audit_sha256": audit_sha,
            "resolution_ref": resolution_ref,
            "resolution_sha256": resolution_sha,
            "resolution_artifact_hash": resolution_hash,
        }
    )
    write_canonical_json(tmp_path / index_ref, index)
    return index_ref, resolution


def test_materialization_bundle_is_hash_bound_and_replayed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_ref, resolution = _write_materialization(tmp_path)
    calls: list[str] = []

    def replay(*args: object, checked_at: str, **kwargs: object) -> dict[str, object]:
        calls.append(checked_at)
        return resolution

    monkeypatch.setattr(formalization, "evaluate_stage0_g3", replay)
    result = load_and_replay_g3_materialization(
        binding=_binding(),
        data_root=tmp_path,
        materialization_index_ref=index_ref,
    )
    assert result.resolution_artifact_hash == resolution["artifact_hash"]
    assert calls == [CHECKED_AT]


def test_materialization_tamper_fails_before_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index_ref, _ = _write_materialization(tmp_path)
    index = load_canonical_json(tmp_path / index_ref)
    audit_path = tmp_path / str(index["audit_ref"])
    audit = load_canonical_json(audit_path)
    audit["status"] = "BLOCKED"
    write_canonical_json(audit_path, audit)
    monkeypatch.setattr(
        formalization,
        "evaluate_stage0_g3",
        lambda *args, **kwargs: pytest.fail("tampered bundle reached asset replay"),
    )
    with pytest.raises(Stage0G3FormalizationError, match="AUDIT_FILE_HASH_MISMATCH"):
        load_and_replay_g3_materialization(
            binding=_binding(),
            data_root=tmp_path,
            materialization_index_ref=index_ref,
        )


def _formalize_g3_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> formalization.Stage0G3FormalizationResult:
    bootstrap = bootstrap_formal_stage0(
        binding=_binding(), data_root=tmp_path, snapshot=_snapshot(tmp_path)
    )
    index_ref, resolution = _write_materialization(tmp_path)
    monkeypatch.setenv(DATA_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(formalization, "evaluate_stage0_g3", lambda *a, **k: resolution)
    source_binding = stage01._G3SourceBinding(
        source_root=ROOT,
        head_commit=HEAD,
        requirements_path=ROOT / "configs/stage0/g3-asset-requirements-v1.json",
        requirements_file_sha256=sha256_file(
            ROOT / "configs/stage0/g3-asset-requirements-v1.json"
        ),
        layout_path=ROOT / "configs/stage0/g3-asset-layout-v1.json",
        layout_file_sha256=sha256_file(
            ROOT / "configs/stage0/g3-asset-layout-v1.json"
        ),
    )
    monkeypatch.setattr(
        stage01, "_formal_g3_roots", lambda root: (source_binding, Path(root).resolve())
    )
    monkeypatch.setattr(
        stage01,
        "_evaluate_current_formal_g3",
        lambda binding, root: (resolution, HEAD),
    )
    monkeypatch.setattr(
        stage01, "_assert_g3_producer_commit_compatible", lambda *args: None
    )
    return formalize_stage0_g3(
        binding=_binding(),
        data_root=tmp_path,
        bootstrap_index_ref=bootstrap.index_ref,
        materialization_index_ref=index_ref,
    )


def test_formalization_publishes_s04_capabilities_and_gates_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _formalize_g3_fixture(tmp_path, monkeypatch)
    state = formalization.load_stage0_g3_formal_state(
        data_root=tmp_path,
        index_ref=first.index_ref,
        expected_git_commit=HEAD,
    )
    second = formalize_stage0_g3(
        binding=_binding(),
        data_root=tmp_path,
        bootstrap_index_ref=str(
            load_canonical_json(tmp_path / first.index_ref)["bootstrap_index_ref"]
        ),
        materialization_index_ref=str(
            load_canonical_json(tmp_path / first.index_ref)["materialization_index_ref"]
        ),
    )
    assert first.index_ref == second.index_ref
    assert first.environment.environment_hash == second.environment.environment_hash
    assert {"model_assets", "data_assets", "tokenizer_assets"}.issubset(
        first.environment.capabilities
    )
    assert set(GATE_IDS) | {"stage0.G3"} <= first.environment.passed_gate_ids
    assert tuple(first.task_output_refs) == (
        "asset_manifest",
        "asset_audit",
        "asset_resolution",
    )
    for gate_id, reference in first.gate_refs.items():
        loaded = load_committed_task_artifact(tmp_path, reference, require_formal=True)
        gate = GateRecord.from_mapping(dict(loaded.payload))
        assert gate.gate_id == gate_id
        assert gate.status is GateStatus.PASS
    index = load_canonical_json(tmp_path / first.index_ref)
    assert index["next_task_id"] == "stage0.05_config_run_identity_and_seeds"
    assert index["next_input_refs"] == list(first.task_output_refs.values())
    assert state.task_output_refs == first.task_output_refs


def test_g4_formal_runner_publishes_real_contracts_and_pass_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    g3 = _formalize_g3_fixture(tmp_path, monkeypatch)
    (tmp_path / "tmp").mkdir()
    source = stage0_g4.G4SourceBinding(ROOT, HEAD, "feat/stage0-completion")
    monkeypatch.setattr(stage0_g4, "_capture_source", lambda: source)

    first = stage0_g4.execute_stage0_g4(
        binding=_binding(),
        data_root=tmp_path,
        g3_index_ref=g3.index_ref,
    )
    second = stage0_g4.execute_stage0_g4(
        binding=_binding(),
        data_root=tmp_path,
        g3_index_ref=g3.index_ref,
    )
    assert first.index_ref == second.index_ref
    assert first.environment.environment_hash == second.environment.environment_hash
    assert "stage0.G4" in first.environment.passed_gate_ids
    assert first.environment.evidence_refs["gate_stage0_g4"] == (
        first.task_output_refs["provenance_record"]
    )
    assert tuple(first.task_output_refs) == (
        "resolved_config",
        "run_identity",
        "seed_plan",
        "provenance_record",
    )
    config = load_committed_task_artifact(
        tmp_path, first.task_output_refs["resolved_config"], require_formal=True
    )
    run = load_committed_task_artifact(
        tmp_path, first.task_output_refs["run_identity"], require_formal=True
    )
    seed = load_committed_task_artifact(
        tmp_path, first.task_output_refs["seed_plan"], require_formal=True
    )
    provenance = load_committed_task_artifact(
        tmp_path, first.task_output_refs["provenance_record"], require_formal=True
    )
    assert config.payload["schema_version"] == "resolved-config-v2"
    assert run.payload["schema_version"] == "run-identity-v1"
    assert len(seed.payload["rank_training"]) == 4
    assert provenance.payload["gate_record"]["status"] == "PASS"
    assert provenance.payload["provenance_record"]["formal_eligible"] is True
    assert all(provenance.payload["validation_report"]["checks"].values())
    state = stage0_g4.load_stage0_g4_formal_state(
        data_root=tmp_path,
        index_ref=first.index_ref,
        expected_git_commit=HEAD,
    )
    assert state.environment.environment_hash == first.environment.environment_hash
    assert state.task_output_refs == first.task_output_refs
    assert state.environment_ref == load_canonical_json(tmp_path / first.index_ref)[
        "environment_ref"
    ]
    g5_config = stage0_g5.build_stage0_g5_config(
        binding=_binding(),
        data_root=tmp_path,
        state=state,
    )
    assert g5_config.task_id == "stage0.06_single_gpu_smoke"
    assert g5_config.run_intent == "formal"
    assert g5_config.section("providers")["kind"] == "offline_hf"
    assert g5_config.base_config.section("distributed")["device_ids"] == [0]
    assert tuple(g5_config.section("orchestration")["input_result_refs"]) == tuple(
        first.task_output_refs.values()
    )
    dispatched: list[tuple[str, ...]] = []

    def fake_g5(request, root, store, *, source_refs):
        del root, store
        dispatched.append(tuple(source_refs))
        return TaskRunResult.passed(
            request,
            artifact_refs={
                kind: f"fake/{kind}.json" for kind in request.task.artifact_kinds
            },
        )

    monkeypatch.setattr(stage0_g5, "run_formal_g5_task", fake_g5)
    request = TaskExecutionRequest(
        config=g5_config,
        task=g5_config.task_definition,
        environment=state.environment,
    )
    direct = TrainingTaskRunner(tmp_path).run(request)
    assert direct.status.value == "PASS"
    assert dispatched == [tuple(first.task_output_refs.values())]

    g5_environment = TaskRuntimeEnvironment(
        capabilities=state.environment.capabilities,
        frozen_contract_stages=state.environment.frozen_contract_stages,
        passed_gate_ids=state.environment.passed_gate_ids | frozenset({"stage0.G5"}),
        estimator_decision_ref=state.environment.estimator_decision_ref,
        evidence_refs={
            **state.environment.evidence_refs,
            "gate_stage0_g5": first.task_output_refs["provenance_record"],
        },
    )
    g5_state = stage0_g5.Stage0G5FormalState(
        environment=g5_environment,
        task_output_refs=first.task_output_refs,
        config=g5_config,
        config_ref="fixture/g5-config.json",
        environment_ref="fixture/g5-environment.json",
        index_ref="fixture/g5-index.json",
        index_sha256="b" * 64,
        gate_artifact_hash="c" * 64,
        g4_index_ref=first.index_ref,
    )
    g6_config = stage0_g6.build_stage0_g6_config(
        binding=_binding(),
        data_root=tmp_path,
        state=g5_state,
    )
    assert g6_config.task_id == "stage0.07_ddp_and_gradient_semantics"
    assert g6_config.section("launcher")["world_size"] == 4
    assert g6_config.section("launcher")["backend"] == "nccl"
    assert g6_config.base_config.section("batching") == {
        "global_batch_size": 8,
        "per_device_batch_size": 1,
        "microbatch_size": 1,
        "accumulation_steps": 2,
        "no_sync": True,
    }
    g6_dispatched: list[tuple[str, ...]] = []

    def fake_g6(request, root, store, *, source_refs):
        del root, store
        g6_dispatched.append(tuple(source_refs))
        return TaskRunResult.passed(
            request,
            artifact_refs={
                kind: f"fake/{kind}.json" for kind in request.task.artifact_kinds
            },
        )

    monkeypatch.setattr(stage0_g6, "run_formal_g6_task", fake_g6)
    g6_request = TaskExecutionRequest(
        config=g6_config,
        task=g6_config.task_definition,
        environment=g5_environment,
    )
    distributed = DistributedTrainingTaskRunner(tmp_path).run(g6_request)
    assert distributed.status.value == "PASS"
    assert g6_dispatched == [tuple(first.task_output_refs.values())]
