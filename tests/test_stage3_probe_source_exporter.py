from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ops.stage3.export_stage3_probe_source as exporter
from ops.stage3.export_stage3_probe_source import (
    FORMAL_ENDPOINTS,
    FORMAL_PROBES,
    PILOT_ENDPOINTS,
    PILOT_PROBES,
    REPLAY_RECORDS,
    ProbeSourceExportError,
    SOURCE_SCHEMA,
    _allocate,
    _loss_contract,
    _load_receipt_and_endpoints,
    _rank_candidates,
    _select_replay_indices,
    _validate_execution_evidence,
    _validate_partition,
    _validate_source,
)
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    canonical_json_hash,
    write_canonical_json,
)
from param_importance_nlp.experiments.stage3_trajectory import Stage3TrajectoryReceipt
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore


def _h(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _source(scope: str = "pilot") -> dict[str, object]:
    body: dict[str, object] = {
        "schema_version": SOURCE_SCHEMA,
        "scope": scope,
        "g3_resolution_ref": "evidence/g3-resolution.json",
        "trajectory_receipt_ref": "receipts/trajectory.json",
        "allocation_seed": 20260829,
        "partition": {
            "probe": {"start": 0, "stop": 65536},
            "pilot": {"start": 0, "stop": 8192},
            "formal": {"start": 8192, "stop": 57344},
            "replay": {"start": 57344, "stop": 65536},
        },
        "output_dir": "outputs/stage3/probes",
    }
    return body | {"artifact_hash": _h(__import__("json").dumps(body, sort_keys=True, separators=(",", ":")) + "\n")}


def _formal_receipt_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, SimpleNamespace], tuple[Path, ...]]:
    """Build three independent formal receipts over a 24+9 endpoint matrix."""

    gates = (
        GateRecord(
            "stage3.G3-0", 3, GateStatus.PASS, "2026-08-29T00:00:00Z",
            evidence_refs=("evidence/g30.json",),
        ),
        GateRecord(
            "stage3.G3-1", 3, GateStatus.PASS, "2026-08-29T00:00:00Z",
            evidence_refs=("evidence/g31.json",),
        ),
        GateRecord(
            "stage3.G3-5", 3, GateStatus.PASS, "2026-08-29T00:00:00Z",
            evidence_refs=("evidence/g35.json",),
        ),
    )
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_h("contract"),
        asset_manifest_hashes=(_h("asset"),),
        prerequisite_gates=gates,
    )
    evidence_path = tmp_path / "evidence" / "formal.json"
    write_canonical_json(evidence_path, evidence.to_dict())

    identities: dict[str, SimpleNamespace] = {}
    receipt_paths: list[Path] = []
    groups = (("14M", 4301, 12), ("14M", 4302, 12), ("31M", 5301, 9))
    for group_index, (model, seed, endpoint_count) in enumerate(groups):
        endpoint_refs: list[str] = []
        endpoint_digests: list[str] = []
        selected_steps: list[int] = []
        for endpoint_index in range(endpoint_count):
            stage = ("early", "middle", "late")[endpoint_index // (endpoint_count // 3)]
            ref = f"endpoints/{group_index}-{endpoint_index}.json"
            digest = _h(f"endpoint:{ref}")
            endpoint_path = tmp_path / ref
            write_canonical_json(endpoint_path, {"schema_version": "endpoint-placeholder-v1"})
            identities[ref] = SimpleNamespace(
                model=model,
                seed=seed,
                stage=stage,
                endpoint_id=f"endpoint-{group_index}-{endpoint_index}",
                endpoint_digest=digest,
                ref=ref,
                update_sample_ids=[f"update-{group_index}-{endpoint_index}"],
            )
            endpoint_refs.append(ref)
            endpoint_digests.append(digest)
            selected_steps.append(endpoint_index + 1)
        receipt = Stage3TrajectoryReceipt(
            receipt_id=f"formal-receipt-{group_index}",
            task_id="stage3.03_endpoint_and_probe_pipeline",
            config_hash=_h(f"config:{group_index}"),
            purpose_scope="formal",
            formal_eligible=True,
            capture_plan_ref=f"plans/capture-{group_index}.json",
            capture_plan_hash=_h(f"capture:{group_index}"),
            training_run_id=f"run-{group_index}",
            selected_steps=tuple(selected_steps),
            endpoint_commit_refs=tuple(endpoint_refs),
            endpoint_digests=tuple(endpoint_digests),
            replay_verified_steps=tuple(selected_steps),
            estimator_authority_ref="authority/estimator.json",
            formal_execution_ref="evidence/formal.json",
            g30_scope_decision_ref="evidence/g30-decision.json",
            g30_gate_hash=gates[0].artifact_hash,
            g31_gate_hash=gates[1].artifact_hash,
        )
        receipt_path = tmp_path / "receipts" / f"formal-{group_index}.json"
        write_canonical_json(receipt_path, receipt.to_dict())
        receipt_paths.append(receipt_path)

    def fake_load_endpoint(source: object, *, scope: str, workspace_root: Path) -> SimpleNamespace:
        del scope, workspace_root
        return identities[source.ref]  # type: ignore[attr-defined]

    monkeypatch.setattr(exporter, "_load_endpoint", fake_load_endpoint)
    body: dict[str, object] = {
        "schema_version": SOURCE_SCHEMA,
        "scope": "formal",
        "g3_resolution_ref": "evidence/g3-resolution.json",
        "trajectory_receipt_refs": [
            path.relative_to(tmp_path).as_posix() for path in receipt_paths
        ],
        "allocation_seed": 20260829,
        "partition": {
            "probe": {"start": 0, "stop": 65536},
            "pilot": {"start": 0, "stop": 8192},
            "formal": {"start": 8192, "stop": 57344},
            "replay": {"start": 57344, "stop": 65536},
        },
        "output_dir": "outputs/stage3/probes",
    }
    source = body | {"artifact_hash": canonical_json_hash(body)}
    return source, identities, tuple(receipt_paths)


def test_source_requires_hash_bound_explicit_partition_and_seed() -> None:
    value = _validate_source(_source())
    assert value["allocation_seed"] == 20260829
    with pytest.raises(ProbeSourceExportError, match="SOURCE_FIELDS_MISMATCH"):
        _validate_source({key: item for key, item in _source().items() if key != "artifact_hash"})


def test_formal_receipt_group_reloads_three_real_trajectories_and_33_endpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, receipt_paths = _formal_receipt_fixture(tmp_path, monkeypatch)
    _validate_source(source)
    loaded, endpoints, endpoint_steps = _load_receipt_and_endpoints(
        source_value=source, workspace=tmp_path, data=tmp_path
    )

    assert isinstance(loaded, tuple)
    assert len(loaded) == 3
    assert len(endpoints) == FORMAL_ENDPOINTS == 33
    assert len(endpoint_steps) == FORMAL_ENDPOINTS
    assert [len(receipt.endpoint_commit_refs) for receipt in loaded] == [12, 12, 9]
    assert [receipt_paths[index].exists() for index in range(3)] == [True, True, True]


def test_formal_receipt_group_rejects_missing_receipt_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, receipt_paths = _formal_receipt_fixture(tmp_path, monkeypatch)
    missing_body = dict(source)
    missing_body["trajectory_receipt_refs"] = [
        receipt_paths[0].relative_to(tmp_path).as_posix(),
        "receipts/does-not-exist.json",
        receipt_paths[2].relative_to(tmp_path).as_posix(),
    ]
    missing_body["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in missing_body.items() if key != "artifact_hash"}
    )
    _validate_source(missing_body)
    with pytest.raises(ProbeSourceExportError, match="REFERENCE_NOT_FOUND"):
        _load_receipt_and_endpoints(
            source_value=missing_body, workspace=tmp_path, data=tmp_path
        )


def test_formal_receipt_group_rejects_duplicate_receipt_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, receipt_paths = _formal_receipt_fixture(tmp_path, monkeypatch)
    duplicate_path = tmp_path / "receipts" / "formal-0-copy.json"
    write_canonical_json(duplicate_path, json.loads(receipt_paths[0].read_text(encoding="utf-8")))
    duplicate_body = dict(source)
    duplicate_body["trajectory_receipt_refs"] = [
        receipt_paths[0].relative_to(tmp_path).as_posix(),
        duplicate_path.relative_to(tmp_path).as_posix(),
        receipt_paths[2].relative_to(tmp_path).as_posix(),
    ]
    duplicate_body["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in duplicate_body.items() if key != "artifact_hash"}
    )
    _validate_source(duplicate_body)
    with pytest.raises(ProbeSourceExportError, match="TRAJECTORY_RECEIPT_DUPLICATE"):
        _load_receipt_and_endpoints(
            source_value=duplicate_body, workspace=tmp_path, data=tmp_path
        )


def test_formal_receipt_group_rejects_mixed_execution_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, receipt_paths = _formal_receipt_fixture(tmp_path, monkeypatch)
    evidence = json.loads((tmp_path / "evidence" / "formal.json").read_text(encoding="utf-8"))
    write_canonical_json(tmp_path / "evidence" / "other.json", evidence)
    mixed_receipt = json.loads(receipt_paths[1].read_text(encoding="utf-8"))
    mixed_receipt["formal_execution_ref"] = "evidence/other.json"
    mixed_receipt["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in mixed_receipt.items() if key != "artifact_hash"}
    )
    mixed_path = tmp_path / "receipts" / "formal-mixed-execution.json"
    write_canonical_json(mixed_path, mixed_receipt)
    mixed_body = dict(source)
    mixed_body["trajectory_receipt_refs"] = [
        receipt_paths[0].relative_to(tmp_path).as_posix(),
        mixed_path.relative_to(tmp_path).as_posix(),
        receipt_paths[2].relative_to(tmp_path).as_posix(),
    ]
    mixed_body["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in mixed_body.items() if key != "artifact_hash"}
    )
    with pytest.raises(ProbeSourceExportError, match="TRAJECTORY_EXECUTION_MIXED"):
        _load_receipt_and_endpoints(
            source_value=mixed_body, workspace=tmp_path, data=tmp_path
        )


def test_formal_receipt_group_rejects_execution_before_g35(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _, receipt_paths = _formal_receipt_fixture(tmp_path, monkeypatch)
    original = FormalExecutionEvidence.from_mapping(
        json.loads((tmp_path / "evidence" / "formal.json").read_text(encoding="utf-8"))
    )
    before_g35 = FormalExecutionEvidence(
        run_intent=original.run_intent,
        contract_freeze_hash=original.contract_freeze_hash,
        asset_manifest_hashes=original.asset_manifest_hashes,
        prerequisite_gates=tuple(
            gate
            for gate in original.prerequisite_gates
            if gate.gate_id != "stage3.G3-5"
        ),
        metadata=original.metadata,
    )
    write_canonical_json(tmp_path / "evidence" / "before-g35.json", before_g35.to_dict())
    refs: list[str] = []
    for index, path in enumerate(receipt_paths):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        receipt["formal_execution_ref"] = "evidence/before-g35.json"
        receipt["artifact_hash"] = canonical_json_hash(
            {key: value for key, value in receipt.items() if key != "artifact_hash"}
        )
        target = tmp_path / "receipts" / f"before-g35-{index}.json"
        write_canonical_json(target, receipt)
        refs.append(target.relative_to(tmp_path).as_posix())
    before_source = dict(source)
    before_source["trajectory_receipt_refs"] = refs
    before_source["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in before_source.items() if key != "artifact_hash"}
    )
    with pytest.raises(
        ProbeSourceExportError, match="FORMAL_TRAJECTORY_GATE_COVERAGE_MISMATCH"
    ):
        _load_receipt_and_endpoints(
            source_value=before_source, workspace=tmp_path, data=tmp_path
        )


def test_formal_receipt_group_rejects_mixed_model_seed_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, identities, _ = _formal_receipt_fixture(tmp_path, monkeypatch)
    first_ref = next(iter(identities))
    identities[first_ref].model = "31M"
    with pytest.raises(
        ProbeSourceExportError, match="TRAJECTORY_RECEIPT_MODEL_SEED_MIXED"
    ):
        _load_receipt_and_endpoints(
            source_value=source, workspace=tmp_path, data=tmp_path
        )


def test_formal_receipt_group_rejects_duplicate_endpoint_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, identities, _ = _formal_receipt_fixture(tmp_path, monkeypatch)
    refs = list(identities)
    identities[refs[1]].endpoint_id = identities[refs[0]].endpoint_id
    with pytest.raises(ProbeSourceExportError, match="TRAJECTORY_ENDPOINT_ID_DUPLICATE"):
        _load_receipt_and_endpoints(
            source_value=source, workspace=tmp_path, data=tmp_path
        )


def test_execution_evidence_accepts_the_real_task_commit_envelope(tmp_path: Path) -> None:
    gates = (
        GateRecord(
            "stage3.G3-0", 3, GateStatus.PASS, "2026-08-29T00:00:00Z",
            evidence_refs=("evidence/g30.json",),
        ),
        GateRecord(
            "stage3.G3-1", 3, GateStatus.PASS, "2026-08-29T00:00:00Z",
            evidence_refs=("evidence/g31.json",),
        ),
    )
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_h("contract"),
        asset_manifest_hashes=(_h("asset"),),
        prerequisite_gates=gates,
    )
    published = TaskArtifactStore(tmp_path, "evidence/formal-execution").publish(
        task_id="stage3.01_prerequisites_and_scope",
        artifact_kind="formal_execution_evidence",
        config_hash=_h("config"),
        run_intent="formal",
        payload=evidence.to_dict(),
        formal_eligible=True,
    )
    receipt = SimpleNamespace(
        formal_execution_ref=published.commit_ref,
        g30_gate_hash=gates[0].artifact_hash,
        g31_gate_hash=gates[1].artifact_hash,
        formal_eligible=False,
    )
    loaded = _validate_execution_evidence(
        receipt, workspace=tmp_path, data=tmp_path
    )
    assert loaded.artifact_hash == evidence.artifact_hash


def test_hash_ranking_is_reproducible_and_without_replacement() -> None:
    first = _rank_candidates(start=10, stop=100, seed=7, scope="pilot", excluded={11, 12}, required=32)
    second = _rank_candidates(start=10, stop=100, seed=7, scope="pilot", excluded={11, 12}, required=32)
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert 11 not in first and 12 not in first


def test_loss_contract_normalizes_the_qualified_pile_storage_mapping() -> None:
    mapping = {
        "attention_mask_policy": "all_one_for_fixed_full_record",
        "effective_target_tokens": 2048,
        "input_sequence_length": 2048,
        "input_slice": [0, 2048],
        "label_sequence_length": 2048,
        "label_slice": [1, 2049],
        "labels_alignment": "pre_shifted_next_token",
        "loss_adapter_id": "pre-shifted-next-token-cross-entropy-v1",
        "source_tokens_per_record": 2049,
    }
    context = SimpleNamespace(
        asset=SimpleNamespace(
            manifest={"metadata": {"storage": {"causal_lm_mapping": mapping}}}
        )
    )
    contract = _loss_contract(context=context)
    assert contract["asset_contract"] == mapping
    assert contract["target_sequence_length"] == 2048
    assert contract["target_slice"] == [1, 2049]
    assert contract["target_tokens_per_record"] == 2048


def test_replay_reserve_exports_only_fixed_hash_selected_subset() -> None:
    asset_id = _h("asset")
    endpoints = (
        SimpleNamespace(
            endpoint_id="endpoint-early",
            update_sample_ids=[f"pile:{asset_id}:record:{57344:012d}"],
        ),
    )
    selected = _select_replay_indices(
        endpoints,
        asset_id=asset_id,
        interval=(57344, 65536),
        seed=7,
    )
    assert len(selected) == REPLAY_RECORDS
    assert len(set(selected)) == REPLAY_RECORDS
    assert 57344 not in selected
    assert all(57344 <= index < 65536 for index in selected)


def test_partition_rejects_overlap_and_allows_reserved_scope_capacity() -> None:
    dataset = SimpleNamespace(record_start=0, record_stop=65536)
    intervals = _validate_partition(_source(), dataset=dataset, scope="pilot")
    assert intervals["pilot"] == (0, 8192)
    bad = _source()
    partition = dict(bad["partition"])
    partition["formal"] = {"start": 300, "stop": 3468}
    bad["partition"] = partition
    body = {key: item for key, item in bad.items() if key != "artifact_hash"}
    bad["artifact_hash"] = _h(__import__("json").dumps(body, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises(ProbeSourceExportError, match="PARTITION_OVERLAP"):
        _validate_partition(bad, dataset=dataset, scope="pilot")


def test_allocate_binds_all_real_endpoint_probe_slots_without_replacement() -> None:
    endpoints = tuple(
        SimpleNamespace(
            model="14M",
            seed=0,
            stage=stage,
            endpoint_id=f"endpoint-{stage}-{ordinal}",
            endpoint_digest=_h(f"{stage}-{ordinal}"),
            ref=f"endpoints/{stage}-{ordinal}.json",
            update_sample_ids=[f"pile:{_h('asset')}:record:{index:012d}"],
        )
        for stage_number, stage in enumerate(("early", "middle", "late"))
        for ordinal, index in enumerate((9000 + stage_number * 2, 9001 + stage_number * 2), start=1)
    )
    allocations, assigned = _allocate(
        endpoints=endpoints,
        endpoint_steps={endpoint.ref: ordinal for ordinal, endpoint in enumerate(endpoints, start=1)},
        asset_id=_h("asset"),
        interval=(0, 384),
        scope="pilot",
        seed=7,
    )
    assert len(allocations) == PILOT_ENDPOINTS
    assert sum(len(item["probes"]) for item in allocations) == PILOT_ENDPOINTS * PILOT_PROBES
    assert len(assigned) == PILOT_ENDPOINTS * PILOT_PROBES
    assert len({index for values in assigned.values() for index in values}) == PILOT_ENDPOINTS * PILOT_PROBES * 32
