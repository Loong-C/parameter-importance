from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage23_task_runners import (
    _load_formal_selected_checkpoint,
)
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def _publish_selected_cell(
    root: Path,
    *,
    stage: str,
    task_id: str,
    config_hash: str,
    source_config_hash: str | None = None,
) -> tuple[SimpleNamespace, str]:
    source_config_hash = source_config_hash or config_hash
    producer_task_id = "stage2.04_reference_target"
    model_id = "pythia-14m"
    revision = f"{stage}-revision-20260825"
    checkpoint_id = f"{model_id}-{stage}-{revision}"
    asset_id = f"{model_id}-step{'1000' if stage == 'early' else '71000'}-{revision}"
    model_ref = f"models/{asset_id}"
    manifest_ref = f"{model_ref}/model-manifest.json"
    files: dict[str, bytes] = {
        "config.json": b'{"model_type":"gpt_neox"}',
        "generation_config.json": b"{}",
        "model.safetensors": stage.encode("ascii"),
        "special_tokens_map.json": b"{}",
        "tokenizer_config.json": b"{}",
        "tokenizer.json": b"{}",
    }
    model_root = root / model_ref
    model_root.mkdir(parents=True)
    manifest_files = []
    for name, data in files.items():
        (model_root / name).write_bytes(data)
        manifest_files.append(
            {
                "name": name,
                "official_lfs_sha256": hashlib.sha256(data).hexdigest()
                if name == "model.safetensors"
                else None,
                "sha256": hashlib.sha256(data).hexdigest(),
                "size_bytes": len(data),
            }
        )
    manifest = {
        "schema": "parameter-importance-model-manifest-v1",
        "requested_revision": "step1000" if stage == "early" else "step71000",
        "repo": "EleutherAI/pythia-14m",
        "revision": revision,
        "files": manifest_files,
        "downloaded_at": "2026-08-25T00:00:00Z",
        "transport_endpoint": "https://huggingface.co",
    }
    write_canonical_json(model_root / "model-manifest.json", manifest)
    manifest_sha256 = hashlib.sha256(
        (model_root / "model-manifest.json").read_bytes()
    ).hexdigest()
    (model_root / "SHA256SUMS").write_text(
        "".join(
            f"{item['sha256']}  {item['name']}\n" for item in manifest_files
        )
        + f"{manifest_sha256}  model-manifest.json\n",
        encoding="ascii",
    )

    registry_source_ref = f"registry/{stage}.json"
    write_canonical_json(root / registry_source_ref, {"registry": {"records": []}})
    resolved_source_ref = f"resolved/{stage}.json"
    write_canonical_json(root / resolved_source_ref, {"source": stage})
    six_source_ref = f"six-cell/{stage}.json"
    write_canonical_json(root / six_source_ref, {"source": stage})

    checkpoint_payload = {
        "schema_version": "checkpoint-manifest-v1",
        "checkpoint_id": checkpoint_id,
        "model_id": model_id,
        "revision": revision,
        "checkpoint_manifest": manifest,
        "source_manifest_ref": manifest_ref,
        "source_manifest_sha256": manifest_sha256,
    }
    selected_row = {
        "cell_id": f"{model_id}:{stage}",
        "model_id": model_id,
        "training_stage": stage,
        "checkpoint_id": checkpoint_id,
        "checkpoint_hash": canonical_json_hash(manifest_sha256),
        "checkpoint_revision": revision,
        "registry_hash": "a" * 64,
        "config_hash": source_config_hash,
        "checkpoint_root_ref": model_ref,
        "checkpoint_manifest_ref": manifest_ref,
    }
    rows = []
    for cell_id in EXPECTED_CELL_IDS:
        if cell_id == selected_row["cell_id"]:
            rows.append(selected_row)
            continue
        other_model, other_stage = cell_id.split(":", 1)
        other_revision = f"{other_stage}-revision-20260825"
        other_checkpoint_id = f"{other_model}-{other_stage}-{other_revision}"
        other_manifest_sha256 = "b" * 64
        rows.append(
            {
                "cell_id": cell_id,
                "model_id": other_model,
                "training_stage": other_stage,
                "checkpoint_id": other_checkpoint_id,
                "checkpoint_hash": canonical_json_hash(other_manifest_sha256),
                "checkpoint_revision": other_revision,
                "registry_hash": "b" * 64,
                "config_hash": source_config_hash,
                "checkpoint_root_ref": f"models/{other_model}-{other_stage}-root",
                "checkpoint_manifest_ref": f"models/{other_model}-{other_stage}-root/model-manifest.json",
            }
        )
    six_payload = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY",
        "scope": "formal",
        "checkpoints": rows,
    }
    model_config = {
        "architecture": model_id,
        "asset_id": asset_id,
        "revision": revision,
        "initialization_id": checkpoint_id,
    }
    identity = {"input_checkpoint_id": checkpoint_id}
    resolved_payload = {
        "schema_version": "resolved-config-v2",
        "base_config": {"model": model_config, "identity": identity},
        "task_id": producer_task_id,
        "config_hash": source_config_hash,
    }
    registry_payload = {
        "schema_version": "stage2-parameter-registry-artifact-v1",
        "status": "READY",
        "scope": "formal",
        "checkpoint_id": checkpoint_id,
        "model_id": model_id,
        "training_stage": stage,
        "registry_hash": "a" * 64,
        "config_hash": source_config_hash,
        "source_s203_manifest_ref": registry_source_ref,
        "source_s203_manifest_sha256": hashlib.sha256(
            (root / registry_source_ref).read_bytes()
        ).hexdigest(),
    }
    refs = {}
    for kind, payload, source_ref in (
        ("checkpoint_manifest", checkpoint_payload, manifest_ref),
        ("six_cell_manifest", six_payload, six_source_ref),
        ("resolved_config", resolved_payload, resolved_source_ref),
        ("parameter_registry", registry_payload, registry_source_ref),
    ):
        store = TaskArtifactStore(root, f"artifacts/{stage}/{kind}")
        refs[kind] = store.publish(
            task_id=producer_task_id,
            artifact_kind=kind,
            config_hash=source_config_hash,
            run_intent="formal",
            formal_eligible=True,
            source_refs=(source_ref,),
            payload=payload,
        ).commit_ref
    request = SimpleNamespace(
        task=SimpleNamespace(task_id=task_id),
        config=SimpleNamespace(
            config_hash=config_hash,
            base_config=SimpleNamespace(
                section=lambda name: (
                    model_config if name == "model" else identity
                )
            ),
        ),
        environment=SimpleNamespace(
            evidence_refs={
                "stage2_checkpoint_manifest": refs["checkpoint_manifest"],
                "stage2_s23_six_cell_manifest": refs["six_cell_manifest"],
                "stage2_resolved_config": refs["resolved_config"],
                "stage2_parameter_registry": refs["parameter_registry"],
            }
        ),
    )
    return request, model_ref


def test_formal_stage2_early_and_midlate_bind_distinct_checkpoint_roots(
    tmp_path: Path,
) -> None:
    early, early_ref = _publish_selected_cell(
        tmp_path, stage="early", task_id="stage2.04_reference_target", config_hash="1" * 64
    )
    late, late_ref = _publish_selected_cell(
        tmp_path,
        stage="mid_late",
        task_id="stage2.06_pilot_and_matrix_freeze",
        config_hash="2" * 64,
        source_config_hash="4" * 64,
    )
    early_binding = _load_formal_selected_checkpoint(early, tmp_path)
    late_binding = _load_formal_selected_checkpoint(late, tmp_path)
    assert early_binding is not None and late_binding is not None
    assert early_binding.root_ref == early_ref
    assert late_binding.root_ref == late_ref
    assert early_binding.root != late_binding.root
    assert early_binding.manifest_sha256 != late_binding.manifest_sha256


def test_formal_selected_checkpoint_root_tampering_fails_closed(tmp_path: Path) -> None:
    request, _root_ref = _publish_selected_cell(
        tmp_path, stage="early", task_id="stage2.05_paired_estimator_runner", config_hash="3" * 64
    )
    model_root = tmp_path / "models"
    config_path = next(model_root.rglob("config.json"))
    config_path.write_bytes(b'{"tampered":true}')
    with pytest.raises(ValueError, match="FORMAL_CHECKPOINT_FILE_BYTES_INVALID"):
        _load_formal_selected_checkpoint(request, tmp_path)
