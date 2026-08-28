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
    _rank_candidates,
    _select_replay_indices,
    _validate_partition,
    _validate_source,
)


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


def test_hash_ranking_is_reproducible_and_without_replacement() -> None:
    first = _rank_candidates(start=10, stop=100, seed=7, scope="pilot", excluded={11, 12}, required=32)
    second = _rank_candidates(start=10, stop=100, seed=7, scope="pilot", excluded={11, 12}, required=32)
    assert first == second
    assert len(first) == len(set(first)) == 32
    assert 11 not in first and 12 not in first


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
