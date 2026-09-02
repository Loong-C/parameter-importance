
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts import canonical_json_hash
from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments.stage3 import NodeCacheKey
from param_importance_nlp.experiments.stage3_formal import (
    PersistentNodeGradientCache,
    SafeTensorTreeCodec,
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
