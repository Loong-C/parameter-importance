from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    _rank_candidates,
    _select_replay_indices,
    _validate_execution_evidence,
    _validate_partition,
    _validate_source,
)
from param_importance_nlp.contracts import FormalExecutionEvidence, GateRecord, GateStatus
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


def test_source_requires_hash_bound_explicit_partition_and_seed() -> None:
    value = _validate_source(_source())
    assert value["allocation_seed"] == 20260829
    with pytest.raises(ProbeSourceExportError, match="SOURCE_FIELDS_MISMATCH"):
        _validate_source({key: item for key, item in _source().items() if key != "artifact_hash"})


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
