from __future__ import annotations

import copy
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from param_importance_nlp.contracts.status import GateStatus
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.sampling import RepetitionMapping, SamplingPlan, SamplingUniverse, STREAM_NAMES, _sha256_json
from param_importance_nlp.experiments.stage2_g22_adapter import (
    ARTIFACT_KINDS,
    FORMAL_ADAPTER_OUTPUT_DIR,
    G22Blocked,
    _gate,
    _cross_bind_offline_registry_hash,
    PRODUCER_COMMIT,
    _producer_identity,
    _validate_sampling_replay,
    _validate_task_inputs,
    evaluate_formal_g22,
)
from param_importance_nlp.experiments import stage2_g22_adapter as adapter
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
