"""Strict Pythia mmap reader and pre-shifted target routing tests."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import struct

import numpy as np
import pytest
import torch

from param_importance_nlp.data.pythia_mmap import (
    MMAP_INDEX_MAGIC,
    MMapIndex,
    OrderedShardReader,
    PythiaDataError,
    PythiaIndexedDataset,
    PythiaShardDescriptor,
)
from param_importance_nlp.providers.training import (
    TorchModelAdapter,
    TrainingMicrobatch,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_index(
    path: Path,
    sizes: list[int],
    pointers: list[int],
    *,
    dtype_code: int = 8,
    version: int = 1,
    document_index: list[int] | None = None,
) -> Path:
    documents = [len(sizes)] if document_index is None else document_index
    header = (
        MMAP_INDEX_MAGIC
        + struct.pack("<Q", version)
        + struct.pack("<B", dtype_code)
        + struct.pack("<QQ", len(sizes), len(documents))
    )
    payload = (
        header
        + np.asarray(sizes, dtype="<i4").tobytes()
        + np.asarray(pointers, dtype="<i8").tobytes()
        + np.asarray(documents, dtype="<i8").tobytes()
    )
    path.write_bytes(payload)
    return path


def _write_shards(
    root: Path, logical_bytes: bytes, shard_sizes: list[int]
) -> tuple[PythiaShardDescriptor, ...]:
    assert sum(shard_sizes) == len(logical_bytes)
    descriptors: list[PythiaShardDescriptor] = []
    cursor = 0
    final_ordinal = len(shard_sizes) - 1
    for ordinal, size in enumerate(shard_sizes):
        path = root / f"document-{ordinal:05d}-of-{final_ordinal:05d}.bin"
        path.write_bytes(logical_bytes[cursor : cursor + size])
        cursor += size
        descriptors.append(
            PythiaShardDescriptor(
                ordinal=ordinal,
                path=path,
                size_bytes=size,
                sha256=_sha256(path),
            )
        )
    return tuple(descriptors)


def _write_record_fixture(
    root: Path,
    records: np.ndarray,
    shard_sizes: list[int],
    *,
    sizes: list[int] | None = None,
    pointers: list[int] | None = None,
) -> tuple[Path, tuple[PythiaShardDescriptor, ...]]:
    token_counts = [records.shape[1]] * records.shape[0] if sizes is None else sizes
    byte_pointers = (
        [position * records.shape[1] * 2 for position in range(records.shape[0])]
        if pointers is None
        else pointers
    )
    idx_path = _write_index(root / "document.idx", token_counts, byte_pointers)
    shards = _write_shards(root, records.astype("<u2").tobytes(), shard_sizes)
    return idx_path, shards


def test_mmap_index_parses_strict_layout(tmp_path: Path) -> None:
    idx_path = _write_index(
        tmp_path / "document.idx",
        [5, 5, 5],
        [0, 10, 20],
        document_index=[0, 3],
    )
    with MMapIndex(idx_path, expected_sha256=_sha256(idx_path)) as index:
        assert index.version == 1
        assert index.dtype_code == 8
        assert index.dtype == np.dtype(np.uint16)
        assert index.sequence_count == 3
        assert index.document_count == 2
        np.testing.assert_array_equal(index.sizes, [5, 5, 5])
        np.testing.assert_array_equal(index.pointers, [0, 10, 20])
        np.testing.assert_array_equal(index.document_index, [0, 3])
    with pytest.raises(PythiaDataError, match="index SHA-256 mismatch"):
        MMapIndex(idx_path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("truncated", "truncated Pythia index header"),
        ("magic", "invalid Pythia index magic"),
        ("version", "unsupported Pythia index version"),
        ("dtype", "unsupported Pythia index dtype code"),
        ("trailing", "Pythia index size mismatch"),
    ],
)
def test_mmap_index_rejects_corrupt_layout(
    tmp_path: Path, mutation: str, message: str
) -> None:
    idx_path = _write_index(tmp_path / "document.idx", [5], [0])
    content = bytearray(idx_path.read_bytes())
    if mutation == "truncated":
        content = content[:20]
    elif mutation == "magic":
        content[0] ^= 0xFF
    elif mutation == "version":
        struct.pack_into("<Q", content, 9, 2)
    elif mutation == "dtype":
        content[17] = 255
    elif mutation == "trailing":
        content.extend(b"x")
    idx_path.write_bytes(content)
    with pytest.raises(PythiaDataError, match=message):
        MMapIndex(idx_path)


def test_ordered_shard_reader_reads_one_continuous_stream(tmp_path: Path) -> None:
    logical = bytes(range(37))
    shards = _write_shards(tmp_path, logical, [5, 7, 25])
    reader = OrderedShardReader(shards)

    assert reader.total_size == len(logical)
    assert reader.read_exact(1, 3) == logical[1:4]
    assert reader.read_exact(5, 7) == logical[5:12]
    assert reader.read_exact(3, 16) == logical[3:19]
    assert reader.read_exact(len(logical), 0) == b""
    with pytest.raises(PythiaDataError, match="exceeds shard stream size"):
        reader.read_exact(36, 2)


def test_ordered_shard_reader_rejects_untrusted_inventory(tmp_path: Path) -> None:
    shards = _write_shards(tmp_path, b"abcdefghijkl", [4, 4, 4])

    with pytest.raises(PythiaDataError, match="ordered and contiguous"):
        OrderedShardReader(tuple(reversed(shards)))
    with pytest.raises(PythiaDataError, match="ordered and contiguous"):
        OrderedShardReader((shards[0], shards[2]))
    with pytest.raises(PythiaDataError, match="duplicate Pythia shard ordinal"):
        OrderedShardReader((shards[0], replace(shards[1], ordinal=0)))
    with pytest.raises(PythiaDataError, match="duplicate Pythia shard path"):
        OrderedShardReader((shards[0], replace(shards[0], ordinal=1)))
    with pytest.raises(PythiaDataError, match="size mismatch"):
        OrderedShardReader((replace(shards[0], size_bytes=3),))
    with pytest.raises(PythiaDataError, match="SHA-256 mismatch"):
        OrderedShardReader((replace(shards[0], sha256="0" * 64),))

    missing = tmp_path / "missing" / "document-00000-of-00000.bin"
    missing_descriptor = PythiaShardDescriptor(0, missing, 1, "0" * 64)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        OrderedShardReader((missing_descriptor,))
    with pytest.raises(PythiaDataError, match="temporary Pile shard is forbidden"):
        PythiaShardDescriptor(
            0,
            tmp_path / "document-00000-of-00000.bin.part",
            1,
            "0" * 64,
        )


def test_indexed_dataset_reads_records_split_across_shards(tmp_path: Path) -> None:
    records = np.arange(6 * 2049, dtype=np.uint16).reshape(6, 2049)
    record_bytes = 2049 * 2
    idx_path, shards = _write_record_fixture(
        tmp_path,
        records,
        [record_bytes, record_bytes + 11, records.nbytes - 2 * record_bytes - 11],
    )

    with PythiaIndexedDataset(idx_path, shards) as dataset:
        # Record zero ends exactly at shard zero; record two is split by shard one.
        np.testing.assert_array_equal(dataset.raw_record(0), records[0])
        np.testing.assert_array_equal(dataset.raw_record(2), records[2])
        np.testing.assert_array_equal(dataset.raw_records(1, 3), records[1:4])

        batch = dataset.batch(1, 3)
        torch.testing.assert_close(batch.tokens, torch.from_numpy(records[1:4]).long())
        torch.testing.assert_close(batch.input_ids, batch.tokens[:, :-1])
        torch.testing.assert_close(batch.target_ids, batch.tokens[:, 1:])
        torch.testing.assert_close(
            batch.attention_mask, torch.ones((3, 2048), dtype=torch.int64)
        )
        assert batch.input_ids.shape == (3, 2048)
        assert batch.target_ids.shape == (3, 2048)
        assert batch.source_record_start == 1
        assert batch.source_record_stop == 4
        assert set(batch.payload()) == {"input_ids", "target_ids", "attention_mask"}


def test_indexed_dataset_applies_explicit_record_interval(tmp_path: Path) -> None:
    records = np.arange(30, dtype=np.uint16).reshape(6, 5)
    idx_path, shards = _write_record_fixture(tmp_path, records, [17, 43])
    with PythiaIndexedDataset(
        idx_path,
        shards,
        record_start=1,
        record_stop=5,
        tokens_per_record=5,
    ) as dataset:
        assert len(dataset) == 4
        np.testing.assert_array_equal(dataset.raw_record(0), records[1])
        np.testing.assert_array_equal(dataset.raw_record(-1), records[4])
        assert dataset.batch(1, 2).source_record_start == 2
        with pytest.raises(IndexError):
            dataset.raw_record(4)


@pytest.mark.parametrize(
    ("sizes", "pointers", "message"),
    [
        ([5, 4, 5], [0, 10, 18], "has 4 tokens"),
        ([5, 5, 5], [0, 12, 22], "not contiguous with its predecessor"),
        ([5, 5, 5], [0, 11, 21], "byte pointer is unaligned"),
    ],
)
def test_indexed_dataset_rejects_invalid_record_contract(
    tmp_path: Path, sizes: list[int], pointers: list[int], message: str
) -> None:
    records = np.arange(15, dtype=np.uint16).reshape(3, 5)
    idx_path, shards = _write_record_fixture(
        tmp_path,
        records,
        [30],
        sizes=sizes,
        pointers=pointers,
    )
    with pytest.raises(PythiaDataError, match=message):
        PythiaIndexedDataset(idx_path, shards, tokens_per_record=5)


def test_indexed_dataset_rejects_incomplete_shard_coverage(tmp_path: Path) -> None:
    records = np.arange(15, dtype=np.uint16).reshape(3, 5)
    idx_path, shards = _write_record_fixture(tmp_path, records, [10, 20])
    with pytest.raises(PythiaDataError, match="beyond verified shard coverage"):
        PythiaIndexedDataset(idx_path, shards[:1], tokens_per_record=5)


class _RecordingLM(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.forward_calls = 0

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        self.forward_calls += 1
        assert attention_mask is not None
        batch_size, length = input_ids.shape
        return {"logits": self.bias.view(1, 1, -1).expand(batch_size, length, -1)}


def _training_batch(payload: dict[str, torch.Tensor]) -> TrainingMicrobatch:
    batch_size = next(iter(payload.values())).shape[0]
    return TrainingMicrobatch(
        "pythia-test-batch",
        payload,
        tuple(f"record-{index}" for index in range(batch_size)),
    )


def test_torch_model_adapter_routes_pre_shifted_targets_without_second_shift() -> None:
    module = _RecordingLM()
    adapter = TorchModelAdapter(module, task_type="causal_lm")
    input_ids = torch.arange(2 * 2048).reshape(2, 2048) % 32
    payload = {
        "input_ids": input_ids,
        "target_ids": (input_ids + 1) % 32,
        "attention_mask": torch.ones((2, 2048), dtype=torch.int64),
    }
    loss = adapter.loss(_training_batch(payload))

    assert loss.effective_count == 4096
    assert loss.statistical_unit == "target_token"
    assert module.forward_calls == 1
    torch.testing.assert_close(
        loss.loss_numerator,
        torch.tensor(4096.0 * np.log(32), dtype=loss.loss_numerator.dtype),
    )


def test_torch_model_adapter_keeps_legacy_labels_internal_shift() -> None:
    module = _RecordingLM()
    adapter = TorchModelAdapter(module, task_type="causal_lm")
    payload = {
        "input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "labels": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        "attention_mask": torch.ones((2, 4), dtype=torch.int64),
    }
    loss = adapter.loss(_training_batch(payload))
    assert loss.effective_count == 6
    assert module.forward_calls == 1


def test_torch_model_adapter_rejects_ambiguous_target_fields_before_forward() -> None:
    module = _RecordingLM()
    adapter = TorchModelAdapter(module, task_type="causal_lm")
    batch = _training_batch(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            "labels": torch.tensor([[1, 2, 3]]),
            "target_ids": torch.tensor([[2, 3, 4]]),
        }
    )
    with pytest.raises(
        ValueError, match="TRAINING_BATCH_LABELS_AND_TARGET_IDS_MUTUALLY_EXCLUSIVE"
    ):
        adapter.loss(batch)
    assert module.forward_calls == 0


def test_target_ids_are_restricted_to_causal_lm() -> None:
    module = _RecordingLM()
    adapter = TorchModelAdapter(module, task_type="sequence_classification")
    batch = _training_batch(
        {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            "target_ids": torch.tensor([[2, 3, 4]]),
        }
    )
    with pytest.raises(ValueError, match="TARGET_IDS_REQUIRE_CAUSAL_LM_TASK"):
        adapter.loss(batch)
    assert module.forward_calls == 0
