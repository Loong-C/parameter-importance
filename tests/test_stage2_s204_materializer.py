from __future__ import annotations

from pathlib import Path
import hashlib

import pytest

from ops.stage2.materialize_s204 import (
    S204MaterializationError,
    TASK_INPUTS,
    _PAYLOAD_SCHEMAS,
    generate_six_cell_configs,
    build_formal_runtime_environment,
    bootstrap_formal_task_inputs,
    materialize_formal_task_inputs,
    publish_reference_sizing_plan,
    write_six_cell_configs,
    _load_gpu_health_identity,
)
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.g21_formal_handoff import (
    ALLOWED_DEVICES,
    AUTH_HASH,
    EXCLUDED_PCI,
    build_g21_formal_handoff,
)
from param_importance_nlp.experiments import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    DataFile,
    DataRangeManifest,
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_DATASET_ID,
    FORMAL_DATASET_REVISION,
    FORMAL_DATA_FILES,
    FORMAL_DATA_MANIFEST_SHA256,
    FORMAL_TOTAL_TRAINING_STEPS,
)
from param_importance_nlp.runtime import TaskArtifactStore, load_committed_task_artifact


def _publish(
    root: Path,
    task_id: str,
    kind: str,
    *,
    formal: bool = True,
    payload: dict[str, object] | None = None,
) -> str:
    value = payload or {"schema_version": _PAYLOAD_SCHEMAS[kind]}
    return TaskArtifactStore(root, f"inputs/{task_id.replace('.', '-')}").publish(
        task_id=task_id,
        artifact_kind=kind,
        config_hash="a" * 64,
        run_intent="formal" if formal else "local_fixture",
        payload=value,
        formal_eligible=formal,
    ).commit_ref


def _source_set(root: Path) -> dict[str, dict[str, str]]:
    return {
        task_id: {
            kind: _publish(root, task_id, kind)
            for kind in kinds
        }
        for task_id, kinds in TASK_INPUTS.items()
    }


def test_missing_or_fixture_predecessor_fails_closed(tmp_path: Path) -> None:
    sources = _source_set(tmp_path)
    sources["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"] = TaskArtifactStore(
        tmp_path, "fixture-asset"
    ).publish(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        artifact_kind="asset_resolution",
        config_hash="a" * 64,
        run_intent="local_fixture",
        payload={"schema_version": _PAYLOAD_SCHEMAS["asset_resolution"]},
        formal_eligible=False,
    ).commit_ref
    with pytest.raises(S204MaterializationError, match="S204_FORMAL_COMMIT_REQUIRED"):
        materialize_formal_task_inputs(tmp_path, sources)

    del sources["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"]
    with pytest.raises(S204MaterializationError, match="S204_TASK_INPUT_SET_INVALID"):
        materialize_formal_task_inputs(tmp_path, sources)


def test_raw_bootstrap_requires_explicit_payload_and_config_hash(tmp_path: Path) -> None:
    with pytest.raises(S204MaterializationError, match="S204_RAW_TASK_SET_INVALID"):
        bootstrap_formal_task_inputs(tmp_path, {})
    raw = {
        task_id: {
            kind: {
                "ref": f"raw/{task_id}/{kind}.json",
                "config_hash": "a" * 64,
                "source_refs": (() if task_id.startswith("stage2.01") else ("raw/upstream.json",)),
            }
            for kind in kinds
        }
        for task_id, kinds in TASK_INPUTS.items()
    }
    with pytest.raises(S204MaterializationError, match="S204_SOURCE_UNREADABLE"):
        bootstrap_formal_task_inputs(tmp_path, raw)


def test_raw_bootstrap_is_candidate_only_and_cannot_unlock_formal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "ops.stage2.materialize_s204._validate_raw_payload",
        lambda _task, _kind, _payload: None,
    )
    raw: dict[str, dict[str, dict[str, object]]] = {}
    for task_id, kinds in TASK_INPUTS.items():
        raw[task_id] = {}
        for kind in kinds:
            ref = f"raw/{task_id}/{kind}.json"
            write_canonical_json(tmp_path / ref, {"schema_version": _PAYLOAD_SCHEMAS[kind]})
            raw[task_id][kind] = {
                "ref": ref,
                "config_hash": "a" * 64,
                "source_refs": ("raw/upstream.json",),
            }
    refs = bootstrap_formal_task_inputs(tmp_path, raw)
    for task_id, kinds in TASK_INPUTS.items():
        for kind in kinds:
            loaded = load_committed_task_artifact(tmp_path, refs[task_id][kind])
            assert loaded.run_intent == "local_fixture"
            assert loaded.identity.formal_eligible is False
            with pytest.raises(ValueError):
                load_committed_task_artifact(tmp_path, refs[task_id][kind], require_formal=True)


def _g21_handoff_with_raw_report(root: Path) -> str:
    report_ref = "evidence/g21/current-gpu-smoke/report.json"
    report = {
        "schema_version": "stage2-s202-current-gpu-smoke-v1",
        "status": "PASS",
        "excluded_pci_bus_ids": [EXCLUDED_PCI],
        "excluded_device": {
            "index": 1,
            "pci_bus_id": EXCLUDED_PCI,
            "uuid": "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c",
            "scheduled": False,
        },
        "allowed_devices": [
            {"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED_DEVICES
        ],
    }
    report_path = root / report_ref
    write_canonical_json(report_path, report)
    report_sha = hashlib.sha256(report_path.read_bytes()).hexdigest()
    commit = "a" * 40
    handoff = build_g21_formal_handoff(
        {
            "current_gpu_smoke": {
                "ref": report_ref,
                "sha256": report_sha,
                "schema_version": "stage2-s202-current-gpu-smoke-v1",
                "status": "PASS",
                "atomic_publication": True,
                "excluded_pci_bus_ids": [EXCLUDED_PCI],
                "excluded_scheduled": False,
                "allowed_devices": [
                    {"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED_DEVICES
                ],
            },
            "schema_version": "stage2-s2.2-g2.1-formal-handoff-v1",
            "status": "PASS",
            "gate_id": "stage2.G2.1",
            "producer_commit": commit,
            "execution_commit": commit,
            "consumer_commit": commit,
            "authorization": {
                "ref": "evidence/auth.json",
                "artifact_hash": AUTH_HASH,
                "user_authorization_original": "允许 Stage 2 结束前继续使用单副本存储，排除故障 GPU 0000:50:00.0，继续执行",
                "issued_at": "2026-08-23T16:30:00+08:00",
                "expires_at": "Stage 2 exit",
                "scope": ["reproducible_stage0_artifacts", "reproducible_stage2_artifacts"],
                "single_copy_accepted": True,
                "excluded_pci_bus_ids": [EXCLUDED_PCI],
                "excluded_non_reproducible_human_evidence": True,
            },
            "historical_stage0": {
                role: {"ref": f"evidence/stage0/{role}.json", "sha256": "c" * 64, "producer_commit": commit, "status": "PASS"}
                for role in ("g5", "g6", "g10")
            },
            "stage1_g1_exit": {"ref": "evidence/stage1/index.json", "sha256": "d" * 64, "producer_commit": commit, "identity": "3f18b04df8922be9894678ae4842bd999c7e8fd5", "status": "PASS"},
            "checks": {name: True for name in ("producer_identity", "execution_identity", "consumer_identity", "authorization_scope", "stage0_g5_g6_g10_identity", "stage1_identity", "gpu_exclusion", "atomic_publication", "hash_verified", "replay_verified", "loader_verified")},
        }
    )
    handoff_ref = "evidence/g21/handoff.json"
    write_canonical_json(root / handoff_ref, handoff)
    return handoff_ref


def test_gpu_identity_reloads_raw_smoke_and_rejects_mapping_drift(tmp_path: Path) -> None:
    handoff_ref = _g21_handoff_with_raw_report(tmp_path)
    _, allowed = _load_gpu_health_identity(tmp_path, handoff_ref)
    assert allowed == ALLOWED_DEVICES

    report_path = tmp_path / "evidence/g21/current-gpu-smoke/report.json"
    report = load_canonical_json(report_path)
    assert isinstance(report, dict)
    report["excluded_device"]["uuid"] = "GPU-wrong"  # type: ignore[index]
    write_canonical_json(report_path, report)
    with pytest.raises(S204MaterializationError, match="S204_GPU_HEALTH_EVIDENCE_INVALID"):
        _load_gpu_health_identity(tmp_path, handoff_ref)


def test_wrong_task_identity_is_not_promoted(tmp_path: Path) -> None:
    sources = _source_set(tmp_path)
    sources["stage2.02_stage1_handoff_and_fixed_state_contract"]["handoff_manifest"] = _publish(
        tmp_path,
        "stage2.01_scope_hypotheses_and_preregistration",
        "preregistration",
    )
    with pytest.raises(S204MaterializationError, match="S204_TASK_ARTIFACT_IDENTITY_MISMATCH"):
        materialize_formal_task_inputs(tmp_path, sources)


def test_materialization_is_content_addressed_and_rereadable(tmp_path: Path) -> None:
    sources = _source_set(tmp_path)
    materialized = materialize_formal_task_inputs(tmp_path, sources)
    assert set(materialized) == set(TASK_INPUTS)
    for task_id, kinds in TASK_INPUTS.items():
        for kind in kinds:
            loaded = load_committed_task_artifact(
                tmp_path, materialized[task_id][kind], require_formal=True
            )
            assert loaded.identity.task_id == task_id
            assert loaded.identity.artifact_kind == kind
            assert loaded.payload["schema_version"] == _PAYLOAD_SCHEMAS[kind]


def _formal_asset_manifest() -> AssetResolutionManifest:
    records = []
    for index, ((model, stage), (step, revision)) in enumerate(FORMAL_CHECKPOINT_SELECTION.items()):
        digest = f"{index + 1:064x}"
        records.append(
            CheckpointRecord(
                model_id=model,
                training_stage=stage,
                checkpoint_id=f"formal-{model}-{stage}",
                training_step=step,
                total_training_steps=FORMAL_TOTAL_TRAINING_STEPS,
                target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
                repository=f"EleutherAI/{model}",
                revision=revision,
                root_ref=f"models/{model}-step{step}",
                state="ready",
                files=(
                    CheckpointFile("model.safetensors", 1, digest, "weights"),
                    CheckpointFile("config.json", 1, digest, "config"),
                    CheckpointFile("tokenizer.json", 1, digest, "tokenizer"),
                ),
                manifest_ref=f"manifests/{model}-{stage}.json",
                manifest_sha256=digest,
                parameter_registry_hash=digest,
                config_sha256=digest,
                tokenizer_sha256=digest,
                load_status="passed",
                load_evidence_ref=f"evidence/{model}-{stage}.json",
                load_evidence_sha256=digest,
            )
        )
    return AssetResolutionManifest(
        scope="formal",
        checkpoints=tuple(records),
        data_range=DataRangeManifest(
            dataset_id=FORMAL_DATASET_ID,
            revision=FORMAL_DATASET_REVISION,
            manifest_ref="manifests/stage2/pile-prefix.json",
            manifest_sha256=FORMAL_DATA_MANIFEST_SHA256,
            files=tuple(
                DataFile(path, size, sha, "token_shard" if path.endswith(".bin") else "index")
                for path, (size, sha) in FORMAL_DATA_FILES.items()
            ),
        ),
        producer_commit="1" * 40,
        execution_commit="2" * 40,
    )


def _formal_asset_payload(manifest: AssetResolutionManifest) -> dict[str, object]:
    return {
        "schema_version": "stage2-task-asset-resolution-v1",
        "provider": {"provider_kind": "offline_hf_stage2_asset_manifest"},
        "stage2_asset_manifest": manifest.to_dict(),
        "preregistration_contract_hash": "a" * 64,
        "upstream_binding_hash": "b" * 64,
        "formal_eligible": False,
    }


def test_six_cells_have_unique_identity_and_fresh_resume_separation(tmp_path: Path) -> None:
    sources = _source_set(tmp_path)
    manifest = _formal_asset_manifest()
    # Use a distinct source store so materialization can publish the same
    # artifact kind into its own immutable namespace.
    asset_ref = TaskArtifactStore(tmp_path, "raw-formal-assets").publish(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        artifact_kind="asset_resolution",
        config_hash="a" * 64,
        run_intent="formal",
        payload=_formal_asset_payload(manifest),
        formal_eligible=True,
    ).commit_ref
    sources["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"] = asset_ref
    source_refs = materialize_formal_task_inputs(tmp_path, sources)
    asset_ref = source_refs["stage2.03_assets_checkpoints_and_sampling"]["asset_resolution"]
    from param_importance_nlp.cli import _load_mapping

    base = _load_mapping(Path(__file__).parents[1] / "configs/run-ready/layers/formal-stage2-estimator.yaml")
    base_ref = "base/formal-stage2-estimator.json"
    base_path = tmp_path / base_ref
    base_path.parent.mkdir(parents=True)
    # The existing CLI loader accepts canonical JSON as the same resolved layer.
    write_canonical_json(base_path, base)
    configs = generate_six_cell_configs(
        tmp_path,
        asset_manifest_ref=asset_ref,
        predecessor_refs=source_refs,
        base_config_ref=base_ref,
    )
    assert len(configs) == 6
    assert len({config.config_hash for config in configs.values()}) == 6
    assert len({config.base_config.section("model")["revision"] for config in configs.values()}) == 6  # type: ignore[index]
    refs = write_six_cell_configs(configs, tmp_path)
    assert len(refs) == 6
    assert all("/fresh/" in f"/{ref}" for ref in refs.values())
    with pytest.raises(S204MaterializationError, match="S204_RESUME_REF_REQUIRED"):
        generate_six_cell_configs(
            tmp_path,
            asset_manifest_ref=asset_ref,
            predecessor_refs=source_refs,
            base_config_ref=base_ref,
            mode="resume",
        )


def test_sizing_plan_binds_formal_execution_evidence(tmp_path: Path) -> None:
    gate = GateRecord(
        gate_id="stage2.G2.1",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-08-23T00:00:00+00:00",
        evidence_refs=("evidence/g2-1.json",),
    )
    evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=(gate,),
    )
    plan, ref = publish_reference_sizing_plan(tmp_path, formal_execution=evidence)
    loaded = load_canonical_json(tmp_path / ref)
    assert loaded["schema_version"] == "stage2-reference-sizing-plan-v1"  # type: ignore[index]
    assert loaded["execution_evidence_hash"] == evidence.artifact_hash  # type: ignore[index]
    assert plan.execution.artifact_hash == evidence.artifact_hash


def test_environment_requires_current_gpu_health_evidence(tmp_path: Path) -> None:
    gate = GateRecord(
        gate_id="stage2.G2.1",
        stage=2,
        status=GateStatus.PASS,
        checked_at="2026-08-23T00:00:00+00:00",
        evidence_refs=("evidence/g2-1.json",),
    )
    evidence = FormalExecutionEvidence(
        run_intent="formal",
        contract_freeze_hash="a" * 64,
        asset_manifest_hashes=("b" * 64,),
        prerequisite_gates=(gate,),
    )
    evidence_ref = "evidence/formal-execution.json"
    path = tmp_path / evidence_ref
    path.parent.mkdir(parents=True)
    write_canonical_json(path, evidence.to_dict())
    with pytest.raises(S204MaterializationError, match="S204_GPU_HEALTH_EVIDENCE_REQUIRED"):
        build_formal_runtime_environment(
            tmp_path,
            formal_execution_ref=evidence_ref,
            stage0_handoff_ref="missing/stage0.json",
            stage1_g1_exit_ref="missing/stage1.json",
            contract_freeze_ref="missing/freeze.json",
            g3_resolution_ref="missing/g3.json",
            stage2_asset_resolution_ref="missing/assets.json",
            gate_refs={"stage2.G2.2": "missing/g2-2.json", "stage1.G1-EXIT": "missing/g1.json"},
            capability_refs={
                "server": "missing/server.json",
                "cuda": "missing/cuda.json",
                "model_assets": "missing/model.json",
                "data_assets": "missing/data.json",
            },
        )
