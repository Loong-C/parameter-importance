from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from param_importance_nlp.experiments import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    DataFile,
    DataRangeManifest,
    DrawStreamManifest,
    ManifestRepair,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    build_data_range_from_prefix,
    FORMAL_CHECKPOINT_SELECTION,
    FORMAL_DATASET_ID,
    FORMAL_DATASET_REVISION,
    FORMAL_DATA_FILES,
    FORMAL_DATA_MANIFEST_SHA256,
    FORMAL_TOTAL_TRAINING_STEPS,
    validate_formal_asset_identity,
)
from param_importance_nlp.cli import _validate_known_artifact


def _plan() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("stage2-fixture-universe", tuple(range(8))),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )


def _checkpoint(model: str, stage: str) -> CheckpointRecord:
    step = {"initialization": 0, "early": 1, "mid_late": 50}[stage]
    fraction = {"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage]
    return CheckpointRecord(
        model_id=model,
        training_stage=stage,
        checkpoint_id=f"fixture-{model}-{stage}",
        training_step=step,
        total_training_steps=100,
        target_fraction=fraction,
        repository=f"fixture/{model}",
        revision="0" * 40,
        root_ref=f"fixture/models/{model}/{stage}",
        state="ready",
        files=(
            CheckpointFile("model.safetensors", 1, "1" * 64, "weights"),
            CheckpointFile("config.json", 1, "2" * 64, "config"),
            CheckpointFile("tokenizer.json", 1, "3" * 64, "tokenizer"),
        ),
        manifest_ref=f"fixture/manifests/{model}-{stage}.json",
        manifest_sha256="2" * 64,
        parameter_registry_hash="3" * 64,
        config_sha256="4" * 64,
        tokenizer_sha256="5" * 64,
        load_status="passed",
        load_evidence_ref=f"fixture/evidence/{model}-{stage}.json",
        load_evidence_sha256="6" * 64,
    )


def _data_range() -> DataRangeManifest:
    return build_data_range_from_prefix(
        dataset_id="pile-selected-prefix",
        revision="a" * 40,
        manifest_ref="manifests/data/pile.json",
        manifest_sha256="b" * 64,
        shard_sha256="c" * 64,
        shard_size_bytes=30_000_000_000,
        index_sha256="d" * 64,
        index_size_bytes=1_757_184_042,
    )


def _formal_manifest() -> AssetResolutionManifest:
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
                root_ref=f"models/{model}/{stage}",
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
    data = DataRangeManifest(
        dataset_id=FORMAL_DATASET_ID,
        revision=FORMAL_DATASET_REVISION,
        manifest_ref="manifests/stage2/pile-prefix.json",
        manifest_sha256=FORMAL_DATA_MANIFEST_SHA256,
        files=tuple(
            DataFile(path, size, sha, "token_shard" if path.endswith(".bin") else "index")
            for path, (size, sha) in FORMAL_DATA_FILES.items()
        ),
    )
    return AssetResolutionManifest(
        scope="formal",
        checkpoints=tuple(records),
        data_range=data,
        producer_commit="1" * 40,
        execution_commit="2" * 40,
    )


def test_each_stream_manifest_retains_generator_state_and_collisions() -> None:
    plan = _plan()
    manifest = plan.draw_manifest("pilot", 32)
    replay = DrawStreamManifest.from_manifest(manifest.to_manifest())
    assert replay.replay_hash == manifest.replay_hash
    assert manifest.stream_state.start_position == 0
    assert manifest.stream_state.end_position == 32
    assert manifest.sample_collision_count >= 0
    assert plan.draw_manifest("pilot", 32).replay_hash == manifest.replay_hash
    assert plan.draw_manifest("confirmatory", 32).replay_hash != manifest.replay_hash


def test_repetition_mapping_roundtrip_binds_double_halves() -> None:
    mapping = RepetitionMapping.create(
        repetition_id="rep-1",
        draws=_plan().draws("confirmatory", 32),
        m_values=(2, 4, 8, 16, 32),
    )
    loaded = RepetitionMapping.from_manifest(mapping.to_dict())
    assert loaded.digest == mapping.digest
    assert loaded.double_halves[0][-1].position + 1 == loaded.double_halves[1][0].position


def test_data_range_accepts_only_stage2_two_file_allowlist() -> None:
    value = _data_range().to_dict()
    assert DataRangeManifest.from_mapping(value).sample_count == 524_288
    invalid = copy.deepcopy(value)
    invalid["files"][0]["path"] = "document-00005-of-00020.bin.part"  # type: ignore[index]
    with pytest.raises(ValueError, match="allowlist"):
        DataRangeManifest.from_mapping(invalid)


def test_checkpoint_record_requires_offline_load_before_ready() -> None:
    with pytest.raises(ValueError, match="passed offline load"):
        CheckpointRecord(
            model_id="pythia-14m",
            training_stage="early",
            checkpoint_id="bad",
            training_step=1,
            total_training_steps=100,
            target_fraction=0.01,
            repository="EleutherAI/pythia-14m",
            revision="1" * 40,
            root_ref="models/bad",
            state="ready",
            files=(CheckpointFile("model.safetensors", 1, "1" * 64, "weights"),),
            load_status="not_run",
        )


def test_asset_resolution_matrix_reports_missing_stage_as_blocked() -> None:
    records = [_checkpoint(model, stage) for model in ("pythia-14m", "pythia-31m-deduped") for stage in ("initialization", "early", "mid_late")]
    blocked = CheckpointRecord(
        model_id="pythia-31m-deduped",
        training_stage="mid_late",
        checkpoint_id="missing-mid",
        training_step=50,
        total_training_steps=100,
        target_fraction=0.5,
        repository="EleutherAI/pythia-31m-deduped",
        revision=None,
        root_ref="models/missing-mid",
        state="blocked",
        missing_reason="fixed revision unavailable; no completed download",
    )
    records[-1] = blocked
    manifest = AssetResolutionManifest(
        scope="formal",
        checkpoints=tuple(records),
        data_range=_data_range(),
        producer_commit="1" * 40,
        execution_commit="2" * 40,
        manifest_repairs=(
            ManifestRepair("manifests/pythia-31m-deduped-step0.json", "3" * 64, "4" * 64, 10, 7),
        ),
    )
    assert manifest.status == "BLOCKED"
    restored = AssetResolutionManifest.from_mapping(manifest.to_dict())
    assert restored.digest == manifest.digest


def test_formal_asset_identity_rejects_frozen_revision_drift() -> None:
    manifest = _formal_manifest()
    validate_formal_asset_identity(manifest)
    drifted = replace(
        manifest,
        checkpoints=tuple(
            replace(record, revision="f" * 40) if index == 1 else record
            for index, record in enumerate(manifest.checkpoints)
        ),
    )
    with pytest.raises(ValueError, match="selection/revision drift"):
        validate_formal_asset_identity(drifted)


def test_public_schemas_are_strict_json_objects() -> None:
    root = Path("schemas/shared")
    for name in (
        "stage2-checkpoint-manifest-v1.json",
        "stage2-data-range-manifest-v1.json",
        "stage2-draw-stream-manifest-v1.json",
        "stage2-repetition-mapping-v1.json",
        "stage2-asset-resolution-v1.json",
    ):
        value = json.loads((root / name).read_text(encoding="utf-8"))
        assert value["type"] == "object"
        assert value["additionalProperties"] is False


def test_cli_known_artifact_dispatches_new_replay_manifests() -> None:
    plan = _plan()
    stream = plan.draw_manifest("pilot", 4).to_manifest()
    assert _validate_known_artifact(stream)[0] == "stage2_draw_stream_manifest"
    mapping = RepetitionMapping.create(
        repetition_id="cli-map",
        draws=plan.draws("confirmatory", 8),
        m_values=(2, 4, 8),
    ).to_dict()
    assert _validate_known_artifact(mapping)[0] == "stage2_repetition_mapping"
