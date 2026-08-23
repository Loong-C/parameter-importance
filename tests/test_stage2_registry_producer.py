from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import pytest

from param_importance_nlp.experiments.stage2_assets import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    build_data_range_from_prefix,
)
from param_importance_nlp.experiments.stage2_registry_producer import (
    RegistryProducerError,
    _data_range_files,
    _safe_join,
    construct_registry_provider,
    produce_registry_manifests,
)
from param_importance_nlp.providers import InMemoryFrozenSampleResolver, build_tiny_training_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_shape_data_manifest_resolves_qualified_root_and_closes_traversal() -> None:
    tmp_path = Path(__file__).resolve().parent / ".tmp-s203-data-path"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    try:
        data_root = tmp_path / "storage"
        asset_root = data_root / "datasets" / "pile-deduped-pythia-preshuffled"
        asset_root.mkdir(parents=True)
        index = asset_root / "document.idx"
        shard = asset_root / "document-00000-of-00020.bin"
        index.write_bytes(b"index-fixture")
        shard.write_bytes(b"shard-fixture")
        range_manifest = tmp_path / "manifests" / "prefix_coverage.json"
        range_manifest.parent.mkdir(parents=True)
        range_manifest.write_text(
            json.dumps(
                {
                    "idx": index.resolve().as_posix(),
                    "bin": shard.resolve().as_posix(),
                    "idx_sha256": _sha(index),
                    "bin_sha256": _sha(shard),
                    "bin_size": shard.stat().st_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        data_range = build_data_range_from_prefix(
            dataset_id="fixture-data",
            revision="b" * 40,
            manifest_ref="manifests/prefix_coverage.json",
            manifest_sha256=_sha(range_manifest),
            shard_sha256=_sha(shard),
            shard_size_bytes=shard.stat().st_size,
            index_sha256=_sha(index),
            index_size_bytes=index.stat().st_size,
        )

        resolved_index, resolved_shard = _data_range_files(
            data_range,
            data_root=data_root,
            data_asset_root=asset_root,
            manifest_root=tmp_path,
        )
        assert resolved_index == index.resolve()
        assert resolved_shard == shard.resolve()

        with pytest.raises(RegistryProducerError, match="PATH_ESCAPE"):
            _safe_join(asset_root, "../document.idx", field="fixture.traversal")

        range_manifest.write_text("{\"idx\": \"tampered\"}\n", encoding="utf-8")
        with pytest.raises(RegistryProducerError, match="DATA_MANIFEST_MISMATCH"):
            _data_range_files(
                data_range,
                data_root=data_root,
                data_asset_root=asset_root,
                manifest_root=tmp_path,
            )
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_tiny_real_model_registry_is_idempotent_and_tamper_closed() -> None:
    # Keep the fixture on the repository volume: this environment's global
    # pytest temp root is ACL-restricted.  The target is exact and disposable.
    tmp_path = Path(__file__).resolve().parent / ".tmp-s203-registry"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    fixture = build_tiny_training_fixture(
        task_type="causal_lm", seed=17, steps=1, microbatches_per_step=1, microbatch_size=1
    )
    batch = fixture.dataset.steps[0][0]
    resolver = InMemoryFrozenSampleResolver(
        {batch.sample_ids[0]: batch},
        resolver_id="tiny-registry-fixture-v1",
        loss_unit="target_token",
        statistical_unit="tiny_sequence",
        weight_unit="effective_target_tokens",
        sampling_design="uniform_with_replacement",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    optimizer_options = {
        "type": "sgd", "learning_rate": 0.1, "momentum": 0.0,
        "weight_decay": 0.0, "fused": False, "foreach": False,
    }
    _, expected_registry = construct_registry_provider(
        fixture.model, resolver, optimizer=optimizer_options,
        optimizer_runtime={}, fixed_state_id="tiny-registry-preview",
    )

    data_root = tmp_path / "assets"
    manifest_root = data_root
    for model in ("pythia-14m", "pythia-31m-deduped"):
        for stage in ("initialization", "early", "mid_late"):
            root = data_root / "models" / model / stage
            root.mkdir(parents=True)
            (root / "model.safetensors").write_bytes(b"tiny-model")
            (root / "config.json").write_text('{"model_type":"tiny"}\n', encoding="utf-8")
            (root / "tokenizer.json").write_bytes(b"tiny-tokenizer")
            manifest = manifest_root / "manifests" / f"{model}-{stage}.json"
            manifest.parent.mkdir(parents=True, exist_ok=True)
            manifest.write_text('{"files":[]}', encoding="utf-8")

    def files(model: str, stage: str) -> tuple[CheckpointFile, ...]:
        root = data_root / "models" / model / stage
        return tuple(
            CheckpointFile(name, (root / name).stat().st_size, _sha(root / name), role)
            for name, role in (
                ("model.safetensors", "weights"),
                ("config.json", "config"),
                ("tokenizer.json", "tokenizer"),
            )
        )

    checkpoints = []
    for model in ("pythia-14m", "pythia-31m-deduped"):
        for stage, step in (("initialization", 0), ("early", 1000), ("mid_late", 71000)):
            manifest = manifest_root / "manifests" / f"{model}-{stage}.json"
            checkpoints.append(
                CheckpointRecord(
                    model_id=model, training_stage=stage, checkpoint_id=f"{model}-{stage}",
                    training_step=step, total_training_steps=143000,
                    target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
                    repository=f"fixture/{model}", revision="a" * 40,
                    root_ref=f"models/{model}/{stage}", state="ready", files=files(model, stage),
                    manifest_ref=f"manifests/{model}-{stage}.json", manifest_sha256=_sha(manifest),
                    parameter_registry_hash=expected_registry.coordinate_registry_hash,
                    config_sha256=_sha(data_root / "models" / model / stage / "config.json"),
                    tokenizer_sha256=_sha(data_root / "models" / model / stage / "tokenizer.json"),
                    load_status="passed", load_evidence_ref=f"evidence/{model}-{stage}.json",
                    load_evidence_sha256="b" * 64,
                )
            )
    assets = AssetResolutionManifest(
        scope="local_fixture", checkpoints=tuple(checkpoints),
        data_range=build_data_range_from_prefix(
            dataset_id="tiny", revision="b" * 40, manifest_ref="manifests/data.json",
            manifest_sha256="c" * 64, shard_sha256="d" * 64, shard_size_bytes=1,
            index_sha256="e" * 64, index_size_bytes=1,
        ), producer_commit="f" * 40, execution_commit="f" * 40,
    )
    config = {"optimizer": optimizer_options, "providers": {"task_type": "causal_lm"}}

    def builder(_checkpoint: CheckpointRecord, _root: Path):
        local = build_tiny_training_fixture(
            task_type="causal_lm", seed=17, steps=1, microbatches_per_step=1, microbatch_size=1
        )
        local_batch = local.dataset.steps[0][0]
        local_resolver = InMemoryFrozenSampleResolver(
            {local_batch.sample_ids[0]: local_batch}, resolver_id="tiny-registry-fixture-v1",
            loss_unit="target_token", statistical_unit="tiny_sequence",
            weight_unit="effective_target_tokens", sampling_design="uniform_with_replacement",
            weights_exogenous=True, common_mean_assumption=True,
        )
        return local.model, local_resolver

    first = produce_registry_manifests(
        assets, data_root=data_root, manifest_root=manifest_root,
        output_root=tmp_path / "output", resolved_config=config,
        provider_builder=builder, formal=False,
    )
    second = produce_registry_manifests(
        assets, data_root=data_root, manifest_root=manifest_root,
        output_root=tmp_path / "output", resolved_config=config,
        provider_builder=builder, formal=False,
    )
    assert first == second
    index = json.loads((tmp_path / "output" / "registry-index.json").read_text(encoding="utf-8"))
    assert len(index["cells"]) == 6
    assert index["asset_resolution_artifact_kind"] == "asset_resolution"
    assert "registry_manifest" not in index["allowed_s203_artifact_kinds"]

    tamper = data_root / "models" / "pythia-14m" / "early" / "config.json"
    tamper.write_text('{"model_type":"tampered"}\n', encoding="utf-8")
    with pytest.raises(RegistryProducerError, match="CHECKPOINT_FILE_MISMATCH"):
        produce_registry_manifests(
            assets, data_root=data_root, manifest_root=manifest_root,
            output_root=tmp_path / "output-2", resolved_config=config,
            provider_builder=builder, formal=False,
        )
    shutil.rmtree(tmp_path, ignore_errors=True)
