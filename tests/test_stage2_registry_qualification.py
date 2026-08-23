from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage2_assets import (
    AssetResolutionManifest,
    CheckpointFile,
    CheckpointRecord,
    build_data_range_from_prefix,
)
from param_importance_nlp.experiments.stage2_registry_producer import construct_registry_provider
from param_importance_nlp.experiments.stage2_registry_qualification import (
    RegistryQualificationError,
    _resolve_source_root,
    load_asset_resolution_input,
    qualify_registry_assets,
)
from param_importance_nlp.providers import InMemoryFrozenSampleResolver, build_tiny_training_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_clean_detached_source_root_uses_producer_path_contract() -> None:
    tmp_path = Path(__file__).resolve().parent / ".tmp-s203-source-root"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    try:
        repo = tmp_path / "parameter-importance"
        repo.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
        source_file = repo / "src" / "producer.py"
        source_file.parent.mkdir()
        source_file.write_text("# detached fixture\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "src/producer.py"], check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                "-C", str(repo), "commit", "--quiet", "-m", "fixture",
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "checkout", "--quiet", "--detach", "HEAD"], check=True)

        assert _resolve_source_root(repo) == repo.resolve()
        with pytest.raises(RegistryQualificationError, match="ROOT_INVALID|LINK_LIKE_ROOT"):
            _resolve_source_root(tmp_path / "missing-repository")
        linked = tmp_path / "repository-link"
        try:
            linked.symlink_to(repo, target_is_directory=True)
        except OSError:
            linked = None
        if linked is not None:
            with pytest.raises(RegistryQualificationError, match="ROOT_INVALID|LINK_LIKE_ROOT"):
                _resolve_source_root(linked)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_append_only_registry_qualification_materializes_six_cells_and_rejects_hash_tamper() -> None:
    tmp_path = Path(__file__).resolve().parent / ".tmp-s203-registry-qualification"
    shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True)
    try:
        data_root = tmp_path / "data"
        manifest_root = tmp_path
        (data_root / "models" / "shared").mkdir(parents=True)
        model_root = data_root / "models" / "shared"
        (model_root / "model.safetensors").write_bytes(b"tiny-model")
        (model_root / "config.json").write_bytes(b'{"model_type":"tiny"}\n')
        (model_root / "tokenizer.json").write_bytes(b"tiny-tokenizer")
        checkpoint_manifest = manifest_root / "manifests" / "checkpoint.json"
        checkpoint_manifest.parent.mkdir(parents=True)
        write_canonical_json(checkpoint_manifest, {"files": []})
        (data_root / "evidence").mkdir(parents=True)

        fixture = build_tiny_training_fixture(
            task_type="causal_lm", seed=17, steps=1, microbatches_per_step=1, microbatch_size=1
        )
        batch = fixture.dataset.steps[0][0]
        resolver = InMemoryFrozenSampleResolver(
            {batch.sample_ids[0]: batch},
            resolver_id="tiny-registry-qualification-v1",
            loss_unit="target_token",
            statistical_unit="tiny_sequence",
            weight_unit="effective_target_tokens",
            sampling_design="uniform_with_replacement",
            weights_exogenous=True,
            common_mean_assumption=True,
        )
        optimizer = {
            "type": "sgd", "learning_rate": 0.1, "momentum": 0.0,
            "weight_decay": 0.0, "fused": False, "foreach": False,
        }
        _, observed_registry = construct_registry_provider(
            fixture.model,
            resolver,
            optimizer=optimizer,
            optimizer_runtime={},
            fixed_state_id="qualification-preview",
        )
        old_hash = "a" * 64
        new_hash = observed_registry.coordinate_registry_hash
        assert old_hash != new_hash

        model_files = tuple(
            CheckpointFile(
                name,
                (model_root / name).stat().st_size,
                _sha(model_root / name),
                role,
            )
            for name, role in (
                ("model.safetensors", "weights"),
                ("config.json", "config"),
                ("tokenizer.json", "tokenizer"),
            )
        )
        checkpoints: list[CheckpointRecord] = []
        stages = (("initialization", 0), ("early", 1000), ("mid_late", 71000))
        for model_id in ("pythia-14m", "pythia-31m-deduped"):
            for stage, step in stages:
                cell_id = f"{model_id}-{stage}"
                offline = data_root / "evidence" / f"{cell_id}.json"
                write_canonical_json(offline, {"parameter_registry_hash": old_hash})
                checkpoints.append(
                    CheckpointRecord(
                        model_id=model_id,
                        training_stage=stage,
                        checkpoint_id=cell_id,
                        training_step=step,
                        total_training_steps=143000,
                        target_fraction={"initialization": 0.0, "early": 0.01, "mid_late": 0.5}[stage],
                        repository=f"fixture/{model_id}",
                        revision="b" * 40,
                        root_ref="models/shared",
                        state="ready",
                        files=model_files,
                        manifest_ref="manifests/checkpoint.json",
                        manifest_sha256=_sha(checkpoint_manifest),
                        parameter_registry_hash=old_hash,
                        config_sha256=_sha(model_root / "config.json"),
                        tokenizer_sha256=_sha(model_root / "tokenizer.json"),
                        load_status="passed",
                        load_evidence_ref=f"evidence/{cell_id}.json",
                        load_evidence_sha256=_sha(offline),
                    )
                )
        assets = AssetResolutionManifest(
            scope="local_fixture",
            checkpoints=tuple(checkpoints),
            data_range=build_data_range_from_prefix(
                dataset_id="tiny",
                revision="c" * 40,
                manifest_ref="manifests/data-range.json",
                manifest_sha256="d" * 64,
                shard_sha256="e" * 64,
                shard_size_bytes=1,
                index_sha256="f" * 64,
                index_size_bytes=1,
            ),
            producer_commit="1" * 40,
            execution_commit="2" * 40,
        )
        parent_path = manifest_root / "manifests" / "asset-resolution.json"
        write_canonical_json(parent_path, assets.to_dict())
        parent_before = parent_path.read_bytes()
        config = {"optimizer": optimizer, "providers": {"task_type": "causal_lm"}}
        config_path = tmp_path / "resolved-config.json"
        write_canonical_json(config_path, config)

        def builder(_checkpoint: CheckpointRecord, _root: Path):
            return fixture.model, resolver

        result = qualify_registry_assets(
            assets,
            asset_resolution_path=parent_path,
            data_root=data_root,
            data_asset_root=None,
            manifest_root=manifest_root,
            output_root=tmp_path / "evidence" / "registry-qualification-r1",
            amendment_output=manifest_root / "manifests" / "asset-resolution-amendment.json",
            resolved_config=config,
            resolved_config_path=config_path,
            source_root=Path(__file__).resolve().parents[1],
            provider_builder=builder,
            formal=False,
        )
        assert len(result.cells) == 6
        assert {cell.provider_derived_registry_hash for cell in result.cells} == {old_hash}
        assert {cell.registry_hash for cell in result.cells} == {new_hash}
        assert result.index_ref.startswith("evidence/registry-qualification-r1/")
        amendment_path = manifest_root / result.amendment_ref
        amendment = load_canonical_json(amendment_path)
        assert isinstance(amendment, dict)
        assert amendment["parent"]["asset_resolution_ref"] == "manifests/asset-resolution.json"
        assert amendment["parent"]["asset_resolution_sha256"] == _sha(parent_path)
        assert len(amendment["qualification_cells"]) == 6
        materialized = amendment["materialized_asset_resolution"]
        assert isinstance(materialized, dict)
        assert all(
            item["parameter_registry_hash"] == new_hash
            for item in materialized["checkpoints"]
        )
        assert parent_path.read_bytes() == parent_before

        loaded = load_asset_resolution_input(
            amendment_path,
            root=manifest_root,
            data_root=data_root,
        )
        assert loaded == materialized
        assert parent_path.read_bytes() == parent_before

        # Replaying against the same output is immutable/idempotent.
        assert qualify_registry_assets(
            assets,
            asset_resolution_path=parent_path,
            data_root=data_root,
            data_asset_root=None,
            manifest_root=manifest_root,
            output_root=tmp_path / "evidence" / "registry-qualification-r1",
            amendment_output=amendment_path,
            resolved_config=config,
            resolved_config_path=config_path,
            source_root=Path(__file__).resolve().parents[1],
            provider_builder=builder,
            formal=False,
        ) == result

        evidence_path = manifest_root / result.cells[0].qualification_ref
        evidence_before = evidence_path.read_bytes()
        evidence = load_canonical_json(evidence_path)
        assert isinstance(evidence, dict)
        evidence["registry_hash"] = "0" * 64
        body = dict(evidence)
        body.pop("qualification_hash")
        evidence["qualification_hash"] = canonical_json_hash(body)
        write_canonical_json(evidence_path, evidence)
        try:
            with pytest.raises(RegistryQualificationError, match="EVIDENCE_FILE_MISMATCH"):
                load_asset_resolution_input(amendment_path, root=manifest_root, data_root=data_root)
        finally:
            evidence_path.write_bytes(evidence_before)
        assert parent_path.read_bytes() == parent_before
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
