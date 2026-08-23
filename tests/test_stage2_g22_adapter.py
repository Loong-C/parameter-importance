from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.status import GateStatus
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash
from param_importance_nlp.experiments.sampling import RepetitionMapping, SamplingPlan, SamplingUniverse, STREAM_NAMES, _sha256_json
from param_importance_nlp.experiments.stage2_g22_adapter import (
    ARTIFACT_KINDS,
    FORMAL_ADAPTER_OUTPUT_DIR,
    G22Blocked,
    _canonical_registry_manifest_ref,
    _gate,
    _cross_bind_offline_registry_hash,
    PRODUCER_COMMIT,
    _producer_identity,
    _validate_checkpoint_manifest_files,
    _validate_real_assets,
    _validate_sampling_replay,
    _validate_task_inputs,
    evaluate_formal_g22,
)
from param_importance_nlp.experiments import stage2_g22_adapter as adapter
from param_importance_nlp.experiments.stage2_assets import CheckpointFile
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.task_artifacts import load_committed_task_artifact


def _formal_fixture(root: Path) -> dict[str, str]:
    store = TaskArtifactStore(root, "task_outputs/s203")
    plan = SamplingPlan(
        universe=SamplingUniverse(
            universe_id="formal-fixture-universe",
            sample_ids=tuple(range(524288)),
            metadata={"fixture": True},
        ),
        stream_seeds={name: 101 + index for index, name in enumerate(STREAM_NAMES)},
    )
    streams = {name: plan.draw_manifest(name, 4).to_manifest() for name in STREAM_NAMES}
    rows = [draw.to_manifest() for name in STREAM_NAMES for draw in plan.draws(name, 4)]
    mapping = RepetitionMapping.create(
        repetition_id="stage2-formal-sampling-nested-fixture",
        draws=plan.draws("pilot", 8),
        m_values=(2, 4, 8),
    )
    payloads = {
        "sampling_plan": plan.to_dict(),
        "draw_manifest": {
            "schema_version": "stage2-task-draw-manifest-v1",
            "sampling_plan_hash": plan.digest,
            "draws": rows,
            "draw_count_by_stream": {name: 4 for name in STREAM_NAMES},
            "stream_manifests": streams,
            "draw_id_unique": True,
            "sample_id_collisions_allowed": True,
            "replay_hash": canonical_json_hash(rows),
            "nested_mapping": mapping.to_dict(),
            "nested_mapping_hash": mapping.digest,
        },
        "asset_resolution": {
            "schema_version": "stage2-task-asset-resolution-v1",
            "provider": {"provider_kind": "fixture"},
            "stage2_asset_manifest": {"schema_version": "stage2-asset-resolution-v1"},
            "preregistration_contract_hash": "a" * 64,
            "upstream_binding_hash": "b" * 64,
            "formal_eligible": False,
        },
        "gate_record": {
            "schema_version": "stage23-task-gate-candidate-v1",
            "task_id": "stage2.03_assets_checkpoints_and_sampling",
            "gate_ids": ["stage2.G2.1"],
            "gate_status": "NOT_RUN",
            "local_validation_status": "NOT_RUN",
            "formal_eligible": False,
            "reason": "formal_gate_requires_independent_review",
        },
    }
    refs: dict[str, str] = {}
    for kind in ARTIFACT_KINDS:
        refs[kind] = store.publish(
            task_id="stage2.03_assets_checkpoints_and_sampling",
            artifact_kind=kind,
            config_hash="c" * 64,
            run_intent="formal",
            formal_eligible=True,
            payload=payloads[kind],
            source_refs=("commits/stage2.01.json",),
        ).commit_ref
    return refs


def test_formal_store_fixture_is_consumed_but_missing_authority_blocks(tmp_path: Path) -> None:
    refs = _formal_fixture(tmp_path)
    result = evaluate_formal_g22(
        repository_root=Path(__file__).parents[1],
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
    )
    assert result["status"] == "BLOCKED"
    assert result["formal_eligible"] is False
    assert result["commit_ref"] is None


def test_candidate_or_tampered_formal_input_cannot_be_promoted(tmp_path: Path) -> None:
    refs = _formal_fixture(tmp_path)
    with pytest.raises(G22Blocked):
        _validate_task_inputs(tmp_path, {**refs, "gate_record": "task_outputs/s203/commits/missing.json"})
    blocked = _gate(
        status=GateStatus.BLOCKED,
        checked_at="2026-01-01T00:00:00Z",
        measured={"fixture": True},
        refs=(),
        reasons=("formal evidence absent",),
    )
    assert blocked.status is GateStatus.BLOCKED


def test_sampling_replay_rejects_self_consistent_but_wrong_draw(tmp_path: Path) -> None:
    del tmp_path
    plan = SamplingPlan(
        universe=SamplingUniverse(
            universe_id="replay-test-universe",
            sample_ids=tuple(range(32)),
            metadata={"test": True},
        ),
        stream_seeds={name: 201 + index for index, name in enumerate(STREAM_NAMES)},
    )
    payloads = {
        "sampling_plan": plan.to_dict(),
        "draw_manifest": {
            "stream_manifests": {
                name: plan.draw_manifest(name, 4).to_manifest() for name in STREAM_NAMES
            }
        },
    }
    replay = _validate_sampling_replay(payloads)
    assert replay["sampling_plan_hash"] == payloads["sampling_plan"]["plan_hash"]
    assert {item["stream"] for item in replay["streams"]} == set(STREAM_NAMES)

    tampered = copy.deepcopy(payloads)
    streams = tampered["draw_manifest"]["stream_manifests"]
    streams["pilot"]["draws"][0]["sample_id"] = 523_999
    streams["pilot"]["replay_hash"] = _sha256_json(
        {key: value for key, value in streams["pilot"].items() if key != "replay_hash"}
    )
    with pytest.raises(G22Blocked, match="G22_SAMPLING_REPLAY_MISMATCH:pilot"):
        _validate_sampling_replay(tampered)


def test_offline_registry_binding_keeps_provider_and_materialized_hashes_distinct() -> None:
    bindings = {
        "checkpoint": {
            "provider_derived_registry_hash": "a" * 64,
            "registry_hash": "b" * 64,
        }
    }
    assert _cross_bind_offline_registry_hash(
        "checkpoint", "b" * 64, "a" * 64, bindings
    ) == ("a" * 64, "b" * 64)
    with pytest.raises(G22Blocked, match="G22_OFFLINE_PROVIDER_REGISTRY_HASH_CROSS_BIND_INVALID"):
        _cross_bind_offline_registry_hash("checkpoint", "b" * 64, "b" * 64, bindings)
    with pytest.raises(G22Blocked, match="G22_OFFLINE_REGISTRY_MATERIALIZED_HASH_INVALID"):
        _cross_bind_offline_registry_hash("checkpoint", "a" * 64, "a" * 64, bindings)


def test_model_manifest_allows_canonical_null_lfs_for_small_files() -> None:
    expected = (
        CheckpointFile("config.json", 1, "a" * 64, "config"),
        CheckpointFile("tokenizer.json", 2, "b" * 64, "tokenizer"),
    )
    files = [
        {
            "name": item.path,
            "official_lfs_sha256": None,
            "sha256": item.sha256,
            "size_bytes": item.size_bytes,
        }
        for item in expected
    ]
    assert _validate_checkpoint_manifest_files(files, expected) == [
        "config.json",
        "tokenizer.json",
    ]


def test_model_manifest_rejects_null_lfs_for_weight_file() -> None:
    expected = (CheckpointFile("model.safetensors", 3, "c" * 64, "weights"),)
    files = [{
        "name": "model.safetensors",
        "official_lfs_sha256": None,
        "sha256": "c" * 64,
        "size_bytes": 3,
    }]
    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_FILE_MISMATCH"):
        _validate_checkpoint_manifest_files(files, expected)


def test_model_manifest_rejects_nonhex_lfs_declaration() -> None:
    expected = (CheckpointFile("model.safetensors", 3, "c" * 64, "weights"),)
    files = [{
        "name": "model.safetensors",
        "official_lfs_sha256": "not-a-sha256",
        "sha256": "c" * 64,
        "size_bytes": 3,
    }]
    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_FILE_MISMATCH"):
        _validate_checkpoint_manifest_files(files, expected)


def _real_asset_validation_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, object], SimpleNamespace]:
    """Build one real model directory to exercise the post-manifest checks."""
    model_root = tmp_path / "models" / "checkpoint"
    model_root.mkdir(parents=True)
    file_specs = (
        ("config.json", b"{}", "config", None),
        ("model.safetensors", b"weights", "weights", "f" * 64),
        ("tokenizer.json", b"tokenizer", "tokenizer", None),
    )
    checkpoint_files = tuple(
        CheckpointFile(
            name,
            len(content),
            hashlib.sha256(content).hexdigest(),
            role,
        )
        for name, content, role, _official_lfs in file_specs
    )
    manifest_files = []
    for (name, content, _role, official_lfs), checkpoint_file in zip(
        file_specs, checkpoint_files
    ):
        (model_root / name).write_bytes(content)
        manifest_files.append(
            {
                "name": name,
                "official_lfs_sha256": official_lfs,
                "sha256": checkpoint_file.sha256,
                "size_bytes": checkpoint_file.size_bytes,
            }
        )

    manifest = {
        "schema": "parameter-importance-model-manifest-v1",
        "requested_revision": "step0",
        "repo": "fixture/model",
        "revision": "fixture-revision",
        "files": manifest_files,
        "repair_scope": "test",
    }
    manifest_path = model_root / "model-manifest.json"
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    (model_root / "SHA256SUMS").write_text(
        "".join(
            f"{checkpoint_file.sha256}  {checkpoint_file.path}\n"
            for checkpoint_file in checkpoint_files
        )
        + f"{manifest_sha256}  model-manifest.json\n",
        encoding="ascii",
    )

    prefix_path = tmp_path / "data-prefix.json"
    prefix_bytes = b"data-prefix"
    prefix_path.write_bytes(prefix_bytes)
    prefix_sha256 = hashlib.sha256(prefix_bytes).hexdigest()
    data_range = SimpleNamespace(
        digest=adapter.DATA_DIGEST,
        files=(),
        manifest_ref="data-prefix.json",
        manifest_sha256=prefix_sha256,
    )
    checkpoint = SimpleNamespace(
        files=checkpoint_files,
        manifest_ref="models/checkpoint/model-manifest.json",
        manifest_sha256=manifest_sha256,
        repository="fixture/model",
        revision="fixture-revision",
        root_ref="models/checkpoint",
        state="ready",
        training_stage="initialization",
    )

    class _FakeManifest:
        def __init__(self, digest: str) -> None:
            self.checkpoints = (checkpoint,)
            self.data_range = data_range
            self.digest = digest
            self.producer_commit = adapter.PRODUCER_COMMIT
            self.execution_commit = adapter.PRODUCER_COMMIT
            self.consumer_commit = None

        def to_dict(self) -> dict[str, object]:
            return {"asset_resolution_hash": self.digest}

    asset = _FakeManifest(adapter.ASSET_DIGEST)
    materialized = _FakeManifest(adapter.AMENDMENT_ASSET_DIGEST)
    materialized_payload = materialized.to_dict()

    authority = {
        "active_partial_objects_untouched": True,
        "asset_resolution_hash": adapter.ASSET_DIGEST,
        "asset_resolution_ref": adapter.ASSET_REF,
        "checkpoint_count": 6,
        "combination_smoke": {
            "all_registry_hashes_present": True,
            "cell_count": 6,
            "registry_hash_count": 2,
            "status": "PASS",
        },
        "consumer_commit": None,
        "cuda_device_excluded": True,
        "data_range_hash": adapter.DATA_DIGEST,
        "data_range_ref": adapter.DATA_REF,
        "execution_commit": adapter.PRODUCER_COMMIT,
        "failed_attempts": [
            {"ref": ref, "sha256": sha} for ref, sha in adapter.FAILED_ATTEMPTS
        ],
        "gate_id": adapter.GATE_ID,
        "offline_load_count": 6,
        "producer_commit": adapter.PRODUCER_COMMIT,
        "schema_version": "stage2-g2.2-asset-gate-evidence-v1",
        "status": "PASS",
    }
    selection = {
        "schema_version": "stage2-checkpoint-selection-v1",
        "total_training_steps": adapter.FORMAL_TOTAL_TRAINING_STEPS,
        "selection_rule": "nearest checkpoint to target fraction; ties choose earlier step",
        "rows": [
            {
                "model_id": model,
                "training_stage": stage,
                "training_step": step,
                "revision": revision,
            }
            for (model, stage), (step, revision) in adapter.FORMAL_CHECKPOINT_SELECTION.items()
        ],
    }

    def fake_load_hashed(_root: Path, ref: str, _expected: str) -> dict[str, object]:
        if ref == adapter.AUTHORITY_EVIDENCE_REF:
            return authority
        if ref == adapter.SELECTION_REF:
            return selection
        return {}

    monkeypatch.setattr(adapter, "_load_hashed", fake_load_hashed)
    monkeypatch.setattr(
        adapter.AssetResolutionManifest,
        "from_mapping",
        staticmethod(lambda _value: asset),
    )
    monkeypatch.setattr(
        adapter.DataRangeManifest,
        "from_mapping",
        staticmethod(lambda _value: data_range),
    )
    monkeypatch.setattr(adapter, "validate_formal_asset_identity", lambda _asset: None)
    monkeypatch.setattr(
        adapter,
        "_validate_amendment",
        lambda _root, _asset: (
            materialized,
            {},
            {"qualification_index": {"ref": "qualification-index.json"}, "qualification_refs": []},
        ),
    )
    monkeypatch.setattr(adapter, "_validate_formal_registry_index", lambda *_args: [])
    monkeypatch.setattr(adapter, "_validate_offline", lambda *_args: [])
    return tmp_path, {"stage2_asset_manifest": materialized_payload}, checkpoint


def test_real_asset_directory_accepts_small_null_lfs_and_weight_lfs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)

    result = _validate_real_assets(root, asset_payload=asset_payload)

    assert result["model_manifests"] == [
        {
            "ref": "models/checkpoint/model-manifest.json",
            "sha256": result["model_manifests"][0]["sha256"],
            "size_bytes": result["model_manifests"][0]["size_bytes"],
        }
    ]


def test_real_asset_directory_rejects_checkpoint_bytes_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    (root / "models" / "checkpoint" / "model.safetensors").write_bytes(b"tampered")

    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_FILE_BYTES_MISMATCH"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_real_asset_directory_rejects_checkpoint_file_set_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    (root / "models" / "checkpoint" / "unexpected.bin").write_bytes(b"extra")

    with pytest.raises(G22Blocked, match="G22_MODEL_DIRECTORY_FILE_SET_MISMATCH"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_real_asset_sidecars_reject_sha256sums_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    sums_path = root / "models" / "checkpoint" / "SHA256SUMS"
    lines = sums_path.read_text(encoding="ascii").splitlines()
    lines[0] = ("0" * 64) + lines[0][64:]
    sums_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_SHA256SUMS_MISMATCH"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_real_asset_sidecars_reject_sha256sums_mode_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    sums_path = root / "models" / "checkpoint" / "SHA256SUMS"
    lines = sums_path.read_text(encoding="ascii").splitlines()
    digest, _separator, path = lines[0].partition("  ")
    lines[0] = f"{digest} *{path}"
    sums_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_SHA256SUMS_FORMAT_INVALID"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_real_asset_sidecars_reject_manifest_hash_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, _checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    manifest_path = root / "models" / "checkpoint" / "model-manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_MANIFEST_SHA_MISMATCH"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_real_asset_sidecars_reject_manifest_path_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, asset_payload, checkpoint = _real_asset_validation_fixture(tmp_path, monkeypatch)
    manifest_path = root / "models" / "checkpoint" / "model-manifest.json"
    drifted_path = root / "models" / "checkpoint" / "model-manifest-alt.json"
    drifted_path.write_bytes(manifest_path.read_bytes())
    checkpoint.manifest_ref = "models/checkpoint/model-manifest-alt.json"

    with pytest.raises(G22Blocked, match="G22_CHECKPOINT_MANIFEST_PATH_INVALID"):
        _validate_real_assets(root, asset_payload=asset_payload)


def test_current_task_producer_uses_clean_head_not_parent_authority_commit() -> None:
    repository = Path(__file__).parents[1]
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head != PRODUCER_COMMIT
    assert _producer_identity(repository, head)["commit"] == head


def _stub_formal_adapter(monkeypatch: pytest.MonkeyPatch, s203_output_dir: str) -> dict[str, str]:
    config_hash = "d" * 64
    source_refs = ("commits/stage2.01.json",)

    def section(name: str) -> object:
        if name == "artifacts":
            return {"output_dir": s203_output_dir}
        if name == "orchestration":
            return {"input_result_refs": list(source_refs)}
        raise KeyError(name)

    config = SimpleNamespace(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        run_intent="formal",
        formal_eligible=True,
        config_hash=config_hash,
        full_hash="e" * 64,
        section=section,
    )
    loaded = {
        kind: SimpleNamespace(
            payload={},
            identity=SimpleNamespace(
                commit_ref=f"{s203_output_dir}/commits/{kind}.json",
                artifact_hash=("f" * 63) + str(index),
            ),
        )
        for index, kind in enumerate(ARTIFACT_KINDS)
    }
    monkeypatch.setattr(adapter, "_validate_task_inputs", lambda _root, _refs: (loaded, config_hash))
    monkeypatch.setattr(adapter, "_config", lambda _root, _ref, _hash: config)
    monkeypatch.setattr(adapter, "_verify_s203_lineage", lambda *_args: ("a" * 64, "b" * 64))
    monkeypatch.setattr(adapter, "validate_formal_s203_payloads", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(adapter, "_validate_sampling_replay", lambda _payloads: {"streams": []})
    monkeypatch.setattr(
        adapter,
        "_git_identity",
        lambda _root: {"head": "1" * 40, "tree": "2" * 40, "sources": {}},
    )
    monkeypatch.setattr(
        adapter,
        "_producer_identity",
        lambda _root, commit: {"commit": commit, "mode": "same_commit", "source_git_blobs": {}},
    )
    monkeypatch.setattr(
        adapter,
        "_validate_real_assets",
        lambda _root, **_kwargs: {
            "amendment": {
                "qualification_index": {"ref": "qualification-index.json"},
                "qualification_refs": [],
            },
            "registry_manifests": [],
            "offline_loads": [],
        },
    )
    return {kind: f"{s203_output_dir}/commits/{kind}.json" for kind in ARTIFACT_KINDS}


def test_formal_gate_uses_independent_output_and_reloads_with_source_refs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s203_output = "task_outputs/s203"
    refs = _stub_formal_adapter(monkeypatch, s203_output)
    TaskArtifactStore(tmp_path, s203_output).publish(
        task_id="stage2.03_assets_checkpoints_and_sampling",
        artifact_kind="gate_record",
        config_hash="d" * 64,
        run_intent="formal",
        formal_eligible=True,
        payload={"schema_version": "candidate-v1", "candidate": True},
        source_refs=("commits/stage2.01.json",),
    )

    first = evaluate_formal_g22(
        repository_root=tmp_path.parent / "repository",
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
    )
    assert first["status"] == "PASS"
    assert first["reused"] is False
    assert str(first["commit_ref"]).startswith(f"{FORMAL_ADAPTER_OUTPUT_DIR}/")
    assert not str(first["commit_ref"]).startswith(f"{s203_output}/")
    loaded = load_committed_task_artifact(tmp_path, str(first["commit_ref"]), require_formal=True)
    assert loaded.source_refs == ("commits/stage2.01.json",)

    second = evaluate_formal_g22(
        repository_root=tmp_path.parent / "repository",
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
    )
    assert second["status"] == "PASS"
    assert second["reused"] is True
    assert second["commit_ref"] == first["commit_ref"]


def test_formal_gate_rejects_s203_output_equal_to_adapter_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    refs = _stub_formal_adapter(monkeypatch, FORMAL_ADAPTER_OUTPUT_DIR)
    result = evaluate_formal_g22(
        repository_root=tmp_path.parent / "repository",
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
    )
    assert result["status"] == "BLOCKED"
    assert "G22_ADAPTER_OUTPUT_DIR_COLLIDES_WITH_S203" in str(result["reason"])


def test_formal_gate_rejects_s203_output_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s203_output = "task_outputs/s203"
    refs = _stub_formal_adapter(monkeypatch, s203_output)
    result = evaluate_formal_g22(
        repository_root=tmp_path.parent / "repository",
        data_root=tmp_path,
        resolved_config_ref="inputs/resolved-config.json",
        s203_artifact_refs=refs,
        output_dir=s203_output,
    )
    assert result["status"] == "BLOCKED"
    assert "G22_ADAPTER_OUTPUT_DIR_OVERRIDE_REJECTED" in str(result["reason"])


def test_registry_index_manifest_refs_join_from_index_parent() -> None:
    assert _canonical_registry_manifest_ref("manifests/pythia-14m-early.json") == (
        "evidence/stage2/s203/formal-registry-r6/manifests/pythia-14m-early.json"
    )


@pytest.mark.parametrize("declared_ref", [
    "/absolute/manifest.json",
    "../escape.json",
    "manifests/../../escape.json",
    "C:/absolute/manifest.json",
    "manifests\\windows.json",
])
def test_registry_index_manifest_refs_reject_escape(declared_ref: str) -> None:
    with pytest.raises(G22Blocked, match="G22_FORMAL_REGISTRY_MANIFEST_REF"):
        _canonical_registry_manifest_ref(declared_ref)
