"""Provider-level contracts for raw Pythia mmap training batches."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from param_importance_nlp.data.pythia_mmap import (
    MMAP_INDEX_MAGIC,
    PYTHIA_TOKENS_PER_RECORD,
    PythiaIndexedDataset,
    PythiaShardDescriptor,
)
from param_importance_nlp.providers import (
    BatchCursor,
    DatasetAdapter,
    PythiaMMapDatasetAdapter,
    PythiaMMapFrozenSampleResolver,
    PythiaMMapProviderError,
    PythiaSamplingDesign,
)
from param_importance_nlp.experiments.sampling import (
    STREAM_NAMES,
    SamplingPlan,
    SamplingUniverse,
)


ASSET_ID = hashlib.sha256(b"qualified-pile-asset").hexdigest()
READY_MANIFEST_SHA256 = hashlib.sha256(b"ready-manifest").hexdigest()
QUALIFICATION_SHA256 = hashlib.sha256(b"qualification").hexdigest()
G3_RESOLUTION_SHA256 = hashlib.sha256(b"g3-resolution").hexdigest()
G3_SOURCE_COMMIT = "a" * 40
G3_RUNTIME_LINEAGE_SHA256 = hashlib.sha256(b"g3-runtime-lineage").hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset(
    root: Path,
    *,
    record_count: int = 16,
    tokens_per_record: int = PYTHIA_TOKENS_PER_RECORD,
) -> tuple[np.ndarray, PythiaIndexedDataset]:
    records = (
        np.arange(record_count * tokens_per_record, dtype=np.uint64)
        .reshape(record_count, tokens_per_record)
        .astype(np.uint16)
    )
    idx_path = root / "document.idx"
    sizes = np.full(record_count, tokens_per_record, dtype="<i4")
    pointers = np.arange(record_count, dtype="<i8") * tokens_per_record * 2
    index = (
        MMAP_INDEX_MAGIC
        + struct.pack("<Q", 1)
        + struct.pack("<B", 8)
        + struct.pack("<QQ", record_count, 1)
        + sizes.tobytes()
        + pointers.tobytes()
        + np.asarray([record_count], dtype="<i8").tobytes()
    )
    idx_path.write_bytes(index)
    shard_path = root / "document-00000-of-00000.bin"
    shard_path.write_bytes(records.astype("<u2", copy=False).tobytes())
    shard = PythiaShardDescriptor(
        ordinal=0,
        path=shard_path,
        size_bytes=shard_path.stat().st_size,
        sha256=_sha256(shard_path),
    )
    return records, PythiaIndexedDataset(
        idx_path,
        (shard,),
        tokens_per_record=tokens_per_record,
        expected_idx_sha256=_sha256(idx_path),
    )


def _adapter(
    dataset: PythiaIndexedDataset,
    *,
    split_start: int = 2,
    split_stop: int = 14,
    microbatch_size: int = 1,
    microbatches_per_step: int = 1,
    design: PythiaSamplingDesign = PythiaSamplingDesign.SEQUENTIAL,
    runtime_lineage_sha256: str = G3_RUNTIME_LINEAGE_SHA256,
) -> PythiaMMapDatasetAdapter:
    return PythiaMMapDatasetAdapter(
        dataset,
        asset_id=ASSET_ID,
        ready_manifest_sha256=READY_MANIFEST_SHA256,
        qualification_sha256=QUALIFICATION_SHA256,
        g3_resolution_artifact_hash=G3_RESOLUTION_SHA256,
        g3_source_commit=G3_SOURCE_COMMIT,
        g3_runtime_lineage_sha256=runtime_lineage_sha256,
        split_start=split_start,
        split_stop=split_stop,
        microbatch_size=microbatch_size,
        microbatches_per_step=microbatches_per_step,
        sampling_design=design,
    )


def _frozen_resolver(dataset: PythiaIndexedDataset) -> PythiaMMapFrozenSampleResolver:
    return PythiaMMapFrozenSampleResolver(
        dataset,
        asset_id=ASSET_ID,
        ready_manifest_sha256=READY_MANIFEST_SHA256,
        qualification_sha256=QUALIFICATION_SHA256,
        g3_resolution_artifact_hash=G3_RESOLUTION_SHA256,
        g3_source_commit=G3_SOURCE_COMMIT,
        g3_runtime_lineage_sha256=G3_RUNTIME_LINEAGE_SHA256,
        split_start=dataset.record_start,
        split_stop=dataset.record_stop,
        sampling_design="with_replacement_versioned_universe",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _indices(microbatches: tuple[object, ...]) -> list[int]:
    result: list[int] = []
    for batch in microbatches:
        sample_ids = getattr(batch, "sample_ids")
        result.extend(int(sample_id.rsplit(":", 1)[1]) for sample_id in sample_ids)
    return result


def _consume(cursor: BatchCursor) -> list[int]:
    result: list[int] = []
    while True:
        try:
            result.extend(_indices(cursor.next_microbatches()))
        except StopIteration:
            return result


def test_sequential_adapter_emits_global_pre_shifted_microbatches(tmp_path: Path) -> None:
    records, dataset = _dataset(tmp_path)
    adapter = _adapter(
        dataset,
        split_start=2,
        split_stop=10,
        microbatch_size=2,
        microbatches_per_step=2,
    )
    assert isinstance(adapter, DatasetAdapter)
    cursor = adapter.cursor(seed=17)
    assert isinstance(cursor, BatchCursor)

    first = cursor.next_microbatches()
    assert _indices(first) == [2, 3, 4, 5]
    torch.testing.assert_close(
        first[0].payload["input_ids"],
        torch.from_numpy(records[2:4, :-1].astype(np.int64)),
    )
    torch.testing.assert_close(
        first[0].payload["target_ids"],
        torch.from_numpy(records[2:4, 1:].astype(np.int64)),
    )
    torch.testing.assert_close(
        first[0].payload["attention_mask"],
        torch.ones((2, 2048), dtype=torch.int64),
    )
    assert set(first[0].payload) == {
        "input_ids",
        "target_ids",
        "attention_mask",
    }
    assert tuple(first[0].metadata["global_record_indices"]) == (2, 3)
    assert first[0].metadata["asset_id"] == ASSET_ID
    assert first[0].metadata["ready_manifest_sha256"] == READY_MANIFEST_SHA256
    assert first[0].metadata["qualification_sha256"] == QUALIFICATION_SHA256

    assert _indices(cursor.next_microbatches()) == [6, 7, 8, 9]
    with pytest.raises(StopIteration, match="PYTHIA_MMAP_CURSOR_EXHAUSTED"):
        cursor.next_microbatches()

    adapter.close()
    adapter.close()
    assert adapter.closed is True
    with pytest.raises(RuntimeError, match="PYTHIA_MMAP_ADAPTER_CLOSED"):
        cursor.state_dict()


def test_frozen_resolver_reads_the_exact_stage3_probe_id_once(
    tmp_path: Path,
) -> None:
    records, dataset = _dataset(tmp_path)
    resolver = _frozen_resolver(dataset)
    selected = resolver.sample_ids[7]

    before = resolver.state_digest()
    batch = resolver.resolve(selected)
    after = resolver.state_digest()

    assert before == after
    assert batch.sample_ids == (selected,)
    assert batch.metadata["global_record_index"] == 7
    torch.testing.assert_close(
        batch.payload["input_ids"],
        torch.from_numpy(records[7:8, :-1].astype(np.int64)),
    )
    torch.testing.assert_close(
        batch.payload["target_ids"],
        torch.from_numpy(records[7:8, 1:].astype(np.int64)),
    )
    assert int(batch.payload["input_ids"][0, -1]) == int(records[7, -2])
    assert int(batch.payload["target_ids"][0, -1]) == int(records[7, -1])
    with pytest.raises(KeyError, match="FROZEN_SAMPLE_ID_UNKNOWN"):
        resolver.resolve(f"pile:{ASSET_ID}:record:{16:012d}")
    resolver.close()


def test_stage2_with_replacement_draws_resolve_duplicate_ids_without_redraw(
    tmp_path: Path,
) -> None:
    records, dataset = _dataset(tmp_path)
    resolver = _frozen_resolver(dataset)
    universe = SamplingUniverse(
        universe_id="stage2-pythia-mmap-test",
        sample_ids=resolver.sample_ids,
    )
    plan = SamplingPlan(
        universe,
        {name: index + 100 for index, name in enumerate(STREAM_NAMES)},
    )

    draws = plan.draws("pilot", 64)
    selected_ids = tuple(draw.sample_id for draw in draws)
    assert len(set(selected_ids)) < len(selected_ids)
    batches = tuple(resolver.resolve(sample_id) for sample_id in selected_ids)

    assert tuple(batch.sample_ids[0] for batch in batches) == selected_ids
    duplicate_id = next(
        sample_id for sample_id in selected_ids if selected_ids.count(sample_id) > 1
    )
    duplicate_batches = [
        batch for batch in batches if batch.sample_ids == (duplicate_id,)
    ]
    assert len(duplicate_batches) >= 2
    for batch in duplicate_batches[1:]:
        torch.testing.assert_close(
            batch.payload["target_ids"], duplicate_batches[0].payload["target_ids"]
        )
    index = int(str(duplicate_id).rsplit(":", 1)[1])
    torch.testing.assert_close(
        duplicate_batches[0].payload["target_ids"],
        torch.from_numpy(records[index : index + 1, 1:].astype(np.int64)),
    )
    resolver.close()


def test_without_replacement_shuffle_is_deterministic_and_ddp_disjoint(
    tmp_path: Path,
) -> None:
    _records, dataset = _dataset(tmp_path)
    adapter = _adapter(
        dataset,
        design=PythiaSamplingDesign.WITHOUT_REPLACEMENT,
    )

    rank_zero = _consume(adapter.cursor(seed=1234, rank=0, world_size=2))
    rank_one = _consume(adapter.cursor(seed=1234, rank=1, world_size=2))
    repeated = _consume(adapter.cursor(seed=1234, rank=0, world_size=2))
    global_order = _consume(adapter.cursor(seed=1234))

    assert rank_zero == repeated
    assert set(rank_zero).isdisjoint(rank_one)
    assert set(rank_zero) | set(rank_one) == set(range(2, 14))
    assert len(rank_zero) + len(rank_one) == 12
    assert global_order != list(range(2, 14))
    assert sorted(global_order) == list(range(2, 14))
    adapter.close()


def test_with_replacement_is_unbounded_by_one_pass_and_can_repeat(
    tmp_path: Path,
) -> None:
    _records, dataset = _dataset(tmp_path)
    adapter = _adapter(
        dataset,
        split_start=5,
        split_stop=6,
        microbatch_size=2,
        design=PythiaSamplingDesign.WITH_REPLACEMENT,
    )
    cursor = adapter.cursor(seed=9)

    observed = [
        index
        for _ in range(4)
        for index in _indices(cursor.next_microbatches())
    ]
    assert observed == [5] * 8
    state = cursor.state_dict()
    assert state["position"] == 8
    assert state["position"] > adapter.split_stop - adapter.split_start
    assert state["order_sha256"] is None
    assert isinstance(state["rng_identity_sha256"], str)
    adapter.close()


@pytest.mark.parametrize("design", tuple(PythiaSamplingDesign))
def test_cursor_resume_replays_exact_next_step(
    tmp_path: Path,
    design: PythiaSamplingDesign,
) -> None:
    _records, dataset = _dataset(tmp_path)
    adapter = _adapter(
        dataset,
        microbatch_size=1,
        microbatches_per_step=2,
        design=design,
    )
    source = adapter.cursor(seed=2026)
    source.next_microbatches()
    state = json.loads(json.dumps(source.state_dict()))
    expected = source.next_microbatches()

    resumed = adapter.cursor(seed=2026)
    resumed.load_state_dict(state)
    actual = resumed.next_microbatches()

    assert [batch.sample_ids for batch in actual] == [
        batch.sample_ids for batch in expected
    ]
    for actual_batch, expected_batch in zip(actual, expected, strict=True):
        assert set(actual_batch.payload) == set(expected_batch.payload)
        for field in actual_batch.payload:
            torch.testing.assert_close(
                actual_batch.payload[field], expected_batch.payload[field]
            )
    adapter.close()


def test_cursor_resume_rejects_different_full_g3_runtime_lineage(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "lineage-a"
    second_root = tmp_path / "lineage-b"
    first_root.mkdir()
    second_root.mkdir()
    _records, first_dataset = _dataset(first_root)
    _records, second_dataset = _dataset(second_root)
    first = _adapter(first_dataset)
    second = _adapter(
        second_dataset,
        runtime_lineage_sha256=hashlib.sha256(b"different-lineage").hexdigest(),
    )
    state = first.cursor(seed=2026).state_dict()

    with pytest.raises(
        PythiaMMapProviderError,
        match="PYTHIA_MMAP_CURSOR_STATE_IDENTITY_MISMATCH:"
        "g3_runtime_lineage_sha256",
    ):
        second.cursor(seed=2026).load_state_dict(state)
    first.close()
    second.close()


@pytest.mark.parametrize("operation", ("next", "state", "load"))
def test_training_cursor_rejects_idx_or_shard_stat_drift(
    tmp_path: Path,
    operation: str,
) -> None:
    _records, dataset = _dataset(tmp_path)
    adapter = _adapter(dataset)
    cursor = adapter.cursor(seed=17)
    state = dict(cursor.state_dict())
    shard_path = tmp_path / "document-00000-of-00000.bin"
    payload = bytearray(shard_path.read_bytes())
    payload[0] ^= 1
    before = shard_path.stat()
    shard_path.write_bytes(payload)
    after = shard_path.stat()
    if (
        after.st_size == before.st_size
        and after.st_mtime_ns == before.st_mtime_ns
        and after.st_ctime_ns == before.st_ctime_ns
        and after.st_dev == before.st_dev
        and after.st_ino == before.st_ino
    ):
        os.utime(
            shard_path,
            ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
        )
        after = shard_path.stat()
    assert (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
        after.st_dev,
        after.st_ino,
    ) != (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
        before.st_dev,
        before.st_ino,
    )

    with pytest.raises(RuntimeError, match="PYTHIA_MMAP_ADAPTER_FILE_STAT_CHANGED"):
        if operation == "next":
            cursor.next_microbatches()
        elif operation == "state":
            cursor.state_dict()
        else:
            cursor.load_state_dict(state)
    adapter.close()


def test_cursor_state_rejects_every_bound_identity_drift(tmp_path: Path) -> None:
    _records, dataset = _dataset(tmp_path)
    adapter = _adapter(
        dataset,
        design=PythiaSamplingDesign.WITH_REPLACEMENT,
    )
    cursor = adapter.cursor(seed=77)
    cursor.next_microbatches()
    state = dict(cursor.state_dict())
    restored = adapter.cursor(seed=77)

    mutations: dict[str, object] = {
        "schema_version": "pythia-mmap-cursor-state-v1",
        "asset_id": "0" * 64,
        "ready_manifest_sha256": "1" * 64,
        "qualification_sha256": "2" * 64,
        "g3_resolution_artifact_hash": "6" * 64,
        "g3_source_commit": "b" * 40,
        "g3_runtime_lineage_sha256": "7" * 64,
        "adapter_identity_sha256": "3" * 64,
        "split": [3, 14],
        "sampling_design": PythiaSamplingDesign.SEQUENTIAL.value,
        "sampler_algorithm": "different-algorithm-v1",
        "seed": 78,
        "rank": False,
        "world_size": 2,
        "microbatch_size": 2,
        "microbatches_per_step": 2,
        "rng_identity_sha256": "4" * 64,
        "order_sha256": "5" * 64,
    }
    for field, replacement in mutations.items():
        drifted = dict(state)
        drifted[field] = replacement
        with pytest.raises(
            PythiaMMapProviderError,
            match=f"PYTHIA_MMAP_CURSOR_STATE_IDENTITY_MISMATCH:{field}",
        ):
            restored.load_state_dict(drifted)

    misaligned = dict(state)
    misaligned["position"] = int(state["position"]) + 1
    with pytest.raises(
        PythiaMMapProviderError,
        match="PYTHIA_MMAP_CURSOR_STATE_POSITION_INVALID",
    ):
        restored.load_state_dict(misaligned)
    adapter.close()


def test_adapter_and_cursor_reject_invalid_boundaries(tmp_path: Path) -> None:
    _records, dataset = _dataset(tmp_path)
    with pytest.raises(PythiaMMapProviderError, match="non-empty absolute interval"):
        _adapter(dataset, split_start=4, split_stop=4)
    with pytest.raises(PythiaMMapProviderError, match="non-empty absolute interval"):
        _adapter(dataset, split_start=0, split_stop=17)
    with pytest.raises(PythiaMMapProviderError, match="asset_id"):
        PythiaMMapDatasetAdapter(
            dataset,
            asset_id="not-a-digest",
            ready_manifest_sha256=READY_MANIFEST_SHA256,
            qualification_sha256=QUALIFICATION_SHA256,
            g3_resolution_artifact_hash=G3_RESOLUTION_SHA256,
            g3_source_commit=G3_SOURCE_COMMIT,
            g3_runtime_lineage_sha256=G3_RUNTIME_LINEAGE_SHA256,
            split_start=0,
            split_stop=1,
            microbatch_size=1,
            microbatches_per_step=1,
            sampling_design=PythiaSamplingDesign.SEQUENTIAL,
        )

    too_large = _adapter(
        dataset,
        split_start=2,
        split_stop=4,
        microbatch_size=3,
        design=PythiaSamplingDesign.SEQUENTIAL,
    )
    with pytest.raises(PythiaMMapProviderError, match="too small"):
        too_large.cursor(seed=0)
    with pytest.raises(PythiaMMapProviderError, match="non-negative integer"):
        too_large.cursor(seed=True)
    with pytest.raises(PythiaMMapProviderError, match="smaller than world_size"):
        too_large.cursor(seed=0, rank=1, world_size=1)
    with pytest.raises(PythiaMMapProviderError, match="world_size must be positive"):
        too_large.cursor(seed=0, world_size=0)
    too_large.close()


def test_adapter_rejects_non_pythia_record_width(tmp_path: Path) -> None:
    _records, dataset = _dataset(tmp_path, tokens_per_record=5)
    with pytest.raises(PythiaMMapProviderError, match="exactly 2049"):
        _adapter(dataset)
    dataset.close()
