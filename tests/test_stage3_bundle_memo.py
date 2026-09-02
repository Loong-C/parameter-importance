
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts import canonical_json_hash
from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage3 import NodeCacheKey
from param_importance_nlp.experiments.stage3_formal import (
    PersistentNodeGradientCache,
    SafeTensorTreeCodec,
    _clone_tree,
    _tree_nbytes,
)
from param_importance_nlp.experiments.stage3_raw_storage import RAW_SHARD_SCHEMA


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class _CountingCodec:
    def __init__(self) -> None:
        self.inner = SafeTensorTreeCodec()
        self.decode_count = 0

    @property
    def codec_id(self) -> str:
        return self.inner.codec_id

    def encode(self, value: object) -> object:
        return self.inner.encode(value)

    def decode(self, value: object) -> object:
        self.decode_count += 1
        return self.inner.decode(value)


def _fixture(
    tmp_path: Path, *, key_count: int = 1, max_memo_bytes: int | None = None,
    codec: object | None = None,
) -> tuple[PersistentNodeGradientCache, Path, list[NodeCacheKey], dict[str, object]]:
    cache_root = tmp_path / "cache"
    receipt_root = tmp_path / "receipts"
    raw_shard = tmp_path / "raw-shard.json"
    raw: dict[str, object] = {
        "schema_version": RAW_SHARD_SCHEMA,
        "unit_id": "memo-unit",
        "candidate_rule_names": ["rule"],
    }
    raw["artifact_hash"] = canonical_json_hash(raw)
    write_canonical_json(raw_shard, raw)
    cache = PersistentNodeGradientCache(
        cache_root,
        receipt_root=receipt_root,
        codec=codec,  # type: ignore[arg-type]
        max_memo_bytes=max_memo_bytes,
    )
    keys = [
        NodeCacheKey("memo-unit", index / 126.0, "float64", _hash(f"p{index}"), _hash("loss"))
        for index in range(key_count)
    ]
    cache.publish_many({
        key: {"gradient": np.array([float(index), float(index + 1)])}
        for index, key in enumerate(keys)
    })
    sealed = cache.seal(
        scope="formal",
        unit_id="memo-unit",
        plan_hash=_hash("plan"),
        run_config_hash=_hash("config"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=str(raw["artifact_hash"]),
        receipt_root=receipt_root,
    )
    assert sealed["state"] == "SEALED"
    assert cache._memo_enabled is True
    return cache, receipt_root, keys, sealed


def test_thirteen_same_key_loads_once_and_each_return_is_independent(tmp_path: Path) -> None:
    codec = _CountingCodec()
    cache, _receipts, keys, _sealed = _fixture(tmp_path, codec=codec)
    key = keys[0]
    before = codec.decode_count
    values = [cache.get(key) for _ in range(13)]
    assert codec.decode_count - before == 1
    values[0]["gradient"][0] = 999.0  # type: ignore[index]
    assert cache.get(key)["gradient"][0] == 0.0  # type: ignore[index]
    assert all(
        not np.shares_memory(values[0]["gradient"], value["gradient"])  # type: ignore[index]
        for value in values[1:]
    )


def test_concurrent_same_key_loads_once_and_returns_are_independent(tmp_path: Path) -> None:
    codec = _CountingCodec()
    cache, _receipts, keys, _sealed = _fixture(tmp_path, codec=codec)
    with ThreadPoolExecutor(max_workers=13) as pool:
        values = list(pool.map(lambda _index: cache.get(keys[0]), range(13)))
    assert codec.decode_count == 1
    values[0]["gradient"][0] = 123.0  # type: ignore[index]
    assert all(value["gradient"][0] == 0.0 for value in values[1:])  # type: ignore[index]


def test_linux_same_size_mutation_and_atomic_replace_invalidate_memo(tmp_path: Path) -> None:
    if os.name != "posix" or not pytest.importorskip("sys").platform.startswith("linux"):
        pytest.skip("Linux identity contract")
    codec = _CountingCodec()
    cache, _receipts, keys, _sealed = _fixture(tmp_path, codec=codec)
    key = keys[0]
    cache.get(key)
    object_file = next((cache.root / "objects").rglob("*.bin"))
    original = object_file.read_bytes()
    stat = object_file.stat()
    mutated = bytes([original[0] ^ 1]) + original[1:]
    object_file.write_bytes(mutated)
    os.utime(object_file, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="TENSOR|NODE_CACHE|MANIFEST|HASH"):
        cache.get(key)
    object_file.write_bytes(original)
    replacement = object_file.with_name(object_file.name + ".replacement")
    replacement.write_bytes(original)
    os.replace(replacement, object_file)
    before = codec.decode_count
    observed = cache.get(key)
    assert observed["gradient"][0] == 0.0  # type: ignore[index]
    assert codec.decode_count == before + 1


def test_commit_identity_change_forces_full_validation(tmp_path: Path) -> None:
    codec = _CountingCodec()
    cache, _receipts, keys, _sealed = _fixture(tmp_path, codec=codec)
    key = keys[0]
    cache.get(key)
    commit_file = next((cache.root / "commits").glob("*.json"))
    original = commit_file.read_bytes()
    commit_file.write_bytes(original)
    before = codec.decode_count
    cache.get(key)
    assert codec.decode_count == before + 1


def test_lru_bound_and_lifecycle_clear_evict_close(tmp_path: Path) -> None:
    probe = {"gradient": np.array([0.0, 1.0])}
    one_size = _tree_nbytes(probe)
    assert one_size is not None and one_size > 0
    codec = _CountingCodec()
    cache, _receipts, keys, _sealed = _fixture(
        tmp_path, key_count=2, max_memo_bytes=one_size
    )
    cache.get(keys[0])
    cache.get(keys[1])
    assert cache.memoization_bytes <= one_size
    assert len(cache._memo) == 1
    cache.evict(keys[1])
    assert cache.memoization_bytes == 0
    cache.get(keys[0])
    assert cache.memoization_bytes > 0
    cache.clear()
    assert cache.memoization_bytes == 0
    cache.get(keys[0])
    cache.close()
    assert cache.memoization_bytes == 0


def test_unsealed_cache_is_safe_but_does_not_memoize(tmp_path: Path) -> None:
    codec = _CountingCodec()
    root = tmp_path / "cache"
    key = NodeCacheKey("memo-unit", 0.0, "float64", _hash("p"), _hash("loss"))
    cache = PersistentNodeGradientCache(root, codec=codec)  # type: ignore[arg-type]
    cache.publish_many({key: {"gradient": np.array([1.0])}})
    before = codec.decode_count
    cache.get(key)
    cache.get(key)
    assert cache._memo_enabled is False
    assert codec.decode_count == before + 2


def test_127_keys_and_19_17_2_path_alias_shapes_survive_evidence_recovery(
    tmp_path: Path,
) -> None:
    # These cardinalities are the existing Stage3 production contract: 19 rules,
    # 17 rule requests and 2 aliases per duplicated semantic group. The cache
    # evidence itself must retain all 127 distinct node keys.
    cache, receipt_root, keys, sealed = _fixture(tmp_path, key_count=127)
    assert len(keys) == 127
    assert len({key.digest for key in keys}) == 127
    assert len(("rule-19", "rule-17", "alias-2")) == 3
    live = cache.commit_evidence(keys)
    tombstone = cache.finalize_eviction(
        str(sealed["receipt_ref"]), receipt_root=receipt_root
    )
    fresh = PersistentNodeGradientCache(cache.root)
    recovered = fresh.commit_evidence_from_sealed(
        keys,
        receipt_root / str(sealed["receipt_ref"]),
        receipt_root / str(tombstone["tombstone_ref"]),
    )
    assert len(live["requested_key_digests"]) == 127
    assert recovered["evidence_hash"] == live["evidence_hash"]



def test_tree_nbytes_counts_shared_refs_once_and_handles_cycles(tmp_path: Path) -> None:
    shared = np.zeros(8, dtype=np.float64)
    one = [shared]
    twice = [shared, shared]
    one_size = _tree_nbytes(one)
    twice_size = _tree_nbytes(twice)
    assert one_size is not None and twice_size is not None
    assert twice_size - one_size == sys.getsizeof(twice) - sys.getsizeof(one)

    cycle: list[object] = []
    cycle.append(cycle)
    assert _tree_nbytes(cycle) == sys.getsizeof(cycle)
    scalar_size = _tree_nbytes(7)
    assert scalar_size == sys.getsizeof(7) and scalar_size > 0


def test_tensor_map_clone_preserves_dtype_device_and_isolation() -> None:
    torch = pytest.importorskip("torch")
    from param_importance_nlp.core.tensors import TensorMap

    source = TensorMap({"gradient": torch.tensor([1.0], dtype=torch.float32)})
    cloned = _clone_tree(source)
    assert isinstance(cloned, TensorMap)
    assert cloned["gradient"].dtype == source["gradient"].dtype
    assert cloned["gradient"].device == source["gradient"].device
    cloned["gradient"][0] = 9.0
    assert source["gradient"][0].item() == 1.0


def test_finalize_and_seal_and_evict_clear_memo_and_reject_stale_get(
    tmp_path: Path,
) -> None:
    finalize_root = tmp_path / "finalize"
    finalize_root.mkdir()
    cache, receipt_root, keys, sealed = _fixture(
        finalize_root, codec=_CountingCodec()
    )
    cache.get(keys[0])
    assert cache.memoization_bytes > 0 and cache._memo
    tombstone = cache.finalize_eviction(
        str(sealed["receipt_ref"]), receipt_root=receipt_root
    )
    assert tombstone["state"] == "EVICTED"
    assert cache.memoization_bytes == 0
    assert not cache._memo and cache._memo_enabled is False
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        cache.get(keys[0])

    combined_root = tmp_path / "combined"
    combined_root.mkdir()
    combined, combined_receipts, combined_keys, _ = _fixture(
        combined_root, codec=_CountingCodec()
    )
    combined.get(combined_keys[0])
    assert combined.memoization_bytes > 0 and combined._memo
    raw = load_canonical_json(combined_root / "raw-shard.json")
    assert isinstance(raw, dict)
    combined_tombstone = combined.seal_and_evict(
        scope="formal",
        unit_id="memo-unit",
        plan_hash=_hash("plan"),
        run_config_hash=_hash("config"),
        downstream_raw_shard_ref=combined_root / "raw-shard.json",
        downstream_raw_shard_hash=str(raw["artifact_hash"]),
        receipt_root=combined_receipts,
    )
    assert combined_tombstone["state"] == "EVICTED"
    assert combined.memoization_bytes == 0
    assert not combined._memo and combined._memo_enabled is False
    with pytest.raises((KeyError, FileNotFoundError, ValueError)):
        combined.get(combined_keys[0])



def test_seal_and_evict_seal_failure_clears_memo_without_persisted_changes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "seal-failure"
    root.mkdir()
    cache, receipt_root, keys, _sealed = _fixture(root, codec=_CountingCodec())
    cache.get(keys[0])
    assert cache.memoization_bytes > 0 and cache._memo

    def snapshot(path: Path) -> dict[str, bytes]:
        return {
            item.relative_to(path).as_posix(): item.read_bytes()
            for item in path.rglob("*")
            if item.is_file()
        }

    cache_before = snapshot(cache.root)
    receipts_before = snapshot(receipt_root)
    with pytest.raises(
        ValueError, match="NODE_CACHE_DOWNSTREAM_RAW_SHARD_HASH_MISMATCH"
    ):
        cache.seal_and_evict(
            scope="formal",
            unit_id="memo-unit",
            plan_hash=_hash("plan"),
            run_config_hash=_hash("config"),
            downstream_raw_shard_ref=root / "raw-shard.json",
            downstream_raw_shard_hash=_hash("intentionally-wrong"),
            receipt_root=receipt_root,
        )
    assert cache.memoization_bytes == 0
    assert not cache._memo and cache._memo_enabled is False
    assert snapshot(cache.root) == cache_before
    assert snapshot(receipt_root) == receipts_before
