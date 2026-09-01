"""Deterministic training cursors for verified raw Pythia mmap records.

The adapter deliberately starts from an already constructed
``PythiaIndexedDataset``.  It never discovers files, resolves manifests, or
weakens the mmap reader's verification boundary.  Callers must provide the
qualified asset identities and one explicit global record interval
``[split_start, split_stop)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import hashlib
import random
import re
from pathlib import Path
import stat
import struct
import threading
from typing import Final, Hashable

import numpy as np
import torch

from ..contracts.jsonio import canonical_json_hash
from ..data.pythia_mmap import PYTHIA_TOKENS_PER_RECORD, PythiaIndexedDataset
from .training import TrainingMicrobatch


_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_MAX_CURSOR_POSITION: Final = (1 << 63) - 1
_CURSOR_STATE_VERSION: Final = "pythia-mmap-cursor-state-v2"
_ADAPTER_IDENTITY_VERSION: Final = "pythia-mmap-adapter-identity-v2"
_SAMPLER_IDENTITY_VERSION: Final = "pythia-mmap-sampler-identity-v1"
_MICROBATCH_METADATA_VERSION: Final = "pythia-mmap-microbatch-metadata-v1"
_FROZEN_RESOLVER_IDENTITY_VERSION: Final = "pythia-mmap-frozen-resolver-v1"
_FROZEN_RESOLVER_METADATA_VERSION: Final = (
    "pythia-mmap-frozen-sample-metadata-v1"
)


def _file_identity(
    logical_path: Path,
    resolved_path: Path,
    *,
    error_code: str,
) -> tuple[object, ...]:
    """Capture immutable logical-link and resolved-target filesystem identity."""

    try:
        logical = logical_path.lstat()
        current_resolved_path = logical_path.resolve()
        target = resolved_path.stat()
    except OSError as error:
        raise RuntimeError(error_code) from error
    if current_resolved_path != resolved_path:
        raise RuntimeError(error_code)
    logical_is_symlink = stat.S_ISLNK(logical.st_mode)
    if not (logical_is_symlink or stat.S_ISREG(logical.st_mode)):
        raise RuntimeError(error_code)
    if not stat.S_ISREG(target.st_mode):
        raise RuntimeError(error_code)
    return (
        str(logical_path),
        logical_is_symlink,
        int(logical.st_dev),
        int(logical.st_ino),
        int(logical.st_size),
        int(logical.st_mtime_ns),
        int(logical.st_ctime_ns),
        str(resolved_path),
        int(target.st_dev),
        int(target.st_ino),
        int(target.st_size),
        int(target.st_mtime_ns),
        int(target.st_ctime_ns),
    )


def _dataset_file_paths(
    dataset: PythiaIndexedDataset,
) -> tuple[tuple[Path, Path], ...]:
    """Derive logical/resolved identities without requiring cache-only fields.

    The qualified mmap reader intentionally remains unchanged.  Its public
    state exposes the resolved index path and shard descriptors; provider
    cursor identity can reconstruct the descriptor logical paths locally while
    treating the index path as already resolved.  This keeps post-G3 cursor
    drift checks compatible with the f32 qualified reader and with the later
    Stage 3 provider fixes.
    """

    shard_paths = tuple(
        (
            descriptor.path.expanduser().absolute(),
            descriptor.path.expanduser().resolve(),
        )
        for descriptor in dataset.reader.shards
    )
    return ((dataset.index.path, dataset.index.path), *shard_paths)


class PythiaMMapProviderError(ValueError):
    """Raised when an mmap provider identity or cursor boundary is invalid."""


class PythiaSamplingDesign(str, Enum):
    """Sampling semantics supported by :class:`PythiaMMapDatasetAdapter`."""

    WITHOUT_REPLACEMENT = "without_replacement"
    WITH_REPLACEMENT = "with_replacement"
    SEQUENTIAL = "sequential"


_SAMPLER_ALGORITHMS: Final[dict[PythiaSamplingDesign, str]] = {
    PythiaSamplingDesign.WITHOUT_REPLACEMENT: (
        "python-random-shuffle-global-then-rank-stride-v1"
    ),
    PythiaSamplingDesign.WITH_REPLACEMENT: "sha256-counter-rejection-v1",
    PythiaSamplingDesign.SEQUENTIAL: "global-order-rank-stride-v1",
}


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PythiaMMapProviderError(
            f"{field} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_git_commit(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _GIT_COMMIT_RE.fullmatch(value) is None:
        raise PythiaMMapProviderError(
            f"{field} must be 40 lowercase hexadecimal characters"
        )
    return value


def _require_nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PythiaMMapProviderError(f"{field} must be a non-negative integer")
    return value


def _require_positive_integer(value: object, *, field: str) -> int:
    normalized = _require_nonnegative_integer(value, field=field)
    if normalized == 0:
        raise PythiaMMapProviderError(f"{field} must be positive")
    return normalized


def _normalize_sampling_design(value: PythiaSamplingDesign | str) -> PythiaSamplingDesign:
    try:
        return PythiaSamplingDesign(value)
    except (TypeError, ValueError) as error:
        raise PythiaMMapProviderError(
            "sampling_design must be without_replacement, with_replacement, "
            "or sequential"
        ) from error


def _order_sha256(indices: tuple[int, ...]) -> str:
    """Hash an integer order without first materializing a large JSON value."""

    digest = hashlib.sha256(b"pythia-mmap-global-record-order-v1\x00")
    digest.update(struct.pack(">Q", len(indices)))
    for index in indices:
        digest.update(struct.pack(">Q", index))
    return digest.hexdigest()


class PythiaMMapDatasetAdapter:
    """Own a verified mmap dataset and expose deterministic training cursors.

    ``split_start`` and ``split_stop`` are absolute record indices in the
    underlying Pythia index.  The adapter takes ownership of ``dataset``;
    callers must close the adapter (or use it as a context manager) to close
    the index mmap deterministically.
    """

    def __init__(
        self,
        dataset: PythiaIndexedDataset,
        *,
        asset_id: str,
        ready_manifest_sha256: str,
        qualification_sha256: str,
        g3_resolution_artifact_hash: str,
        g3_source_commit: str,
        g3_runtime_lineage_sha256: str,
        split_start: int,
        split_stop: int,
        microbatch_size: int,
        microbatches_per_step: int,
        sampling_design: PythiaSamplingDesign | str,
    ) -> None:
        if not isinstance(dataset, PythiaIndexedDataset):
            raise TypeError("dataset must be a PythiaIndexedDataset")
        normalized_asset_id = _require_digest(asset_id, field="asset_id")
        normalized_manifest = _require_digest(
            ready_manifest_sha256, field="ready_manifest_sha256"
        )
        normalized_qualification = _require_digest(
            qualification_sha256, field="qualification_sha256"
        )
        normalized_resolution = _require_digest(
            g3_resolution_artifact_hash,
            field="g3_resolution_artifact_hash",
        )
        normalized_source_commit = _require_git_commit(
            g3_source_commit, field="g3_source_commit"
        )
        normalized_runtime_lineage = _require_digest(
            g3_runtime_lineage_sha256,
            field="g3_runtime_lineage_sha256",
        )
        start = _require_nonnegative_integer(split_start, field="split_start")
        stop = _require_nonnegative_integer(split_stop, field="split_stop")
        if not dataset.record_start <= start < stop <= dataset.record_stop:
            raise PythiaMMapProviderError(
                "split must be a non-empty absolute interval contained in the "
                "PythiaIndexedDataset record interval"
            )
        if dataset.tokens_per_record != PYTHIA_TOKENS_PER_RECORD:
            raise PythiaMMapProviderError(
                f"Pythia mmap provider requires exactly {PYTHIA_TOKENS_PER_RECORD} "
                "tokens per record"
            )
        normalized_microbatch_size = _require_positive_integer(
            microbatch_size, field="microbatch_size"
        )
        normalized_micros = _require_positive_integer(
            microbatches_per_step, field="microbatches_per_step"
        )
        group_size = normalized_microbatch_size * normalized_micros
        if group_size > _MAX_CURSOR_POSITION:
            raise PythiaMMapProviderError("microbatch group size is too large")

        self._dataset = dataset
        self._asset_id = normalized_asset_id
        self._ready_manifest_sha256 = normalized_manifest
        self._qualification_sha256 = normalized_qualification
        self._g3_resolution_artifact_hash = normalized_resolution
        self._g3_source_commit = normalized_source_commit
        self._g3_runtime_lineage_sha256 = normalized_runtime_lineage
        self._split_start = start
        self._split_stop = stop
        self._microbatch_size = normalized_microbatch_size
        self._microbatches_per_step = normalized_micros
        self._sampling_design = _normalize_sampling_design(sampling_design)
        self._initial_file_stats = self._file_stats()
        self._closed = False
        self._lock = threading.RLock()
        self._identity_sha256 = canonical_json_hash(
            {
                "schema_version": _ADAPTER_IDENTITY_VERSION,
                "asset_id": self.asset_id,
                "ready_manifest_sha256": self.ready_manifest_sha256,
                "qualification_sha256": self.qualification_sha256,
                "g3_resolution_artifact_hash": self.g3_resolution_artifact_hash,
                "g3_source_commit": self.g3_source_commit,
                "g3_runtime_lineage_sha256": self.g3_runtime_lineage_sha256,
                "dataset_record_interval": [
                    self._dataset.record_start,
                    self._dataset.record_stop,
                ],
                "split": [self.split_start, self.split_stop],
                "tokens_per_record": self._dataset.tokens_per_record,
                "microbatch_size": self.microbatch_size,
                "microbatches_per_step": self.microbatches_per_step,
                "sampling_design": self.sampling_design,
            }
        )

    @property
    def dataset_id(self) -> str:
        return self._asset_id

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def ready_manifest_sha256(self) -> str:
        return self._ready_manifest_sha256

    @property
    def qualification_sha256(self) -> str:
        return self._qualification_sha256

    @property
    def g3_resolution_artifact_hash(self) -> str:
        return self._g3_resolution_artifact_hash

    @property
    def g3_source_commit(self) -> str:
        return self._g3_source_commit

    @property
    def g3_runtime_lineage_sha256(self) -> str:
        return self._g3_runtime_lineage_sha256

    @property
    def split_start(self) -> int:
        return self._split_start

    @property
    def split_stop(self) -> int:
        return self._split_stop

    @property
    def microbatch_size(self) -> int:
        return self._microbatch_size

    @property
    def microbatches_per_step(self) -> int:
        return self._microbatches_per_step

    @property
    def sampling_design(self) -> str:
        return self._sampling_design.value

    @property
    def identity_sha256(self) -> str:
        return self._identity_sha256

    def state_digest(self) -> str:
        """Return the immutable provider identity used by runtime provenance."""

        with self._lock:
            self._require_open()
            self._assert_files_unchanged()
            return self.identity_sha256

    @property
    def closed(self) -> bool:
        return self._closed

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PYTHIA_MMAP_ADAPTER_CLOSED")

    def _file_stats(self) -> tuple[tuple[object, ...], ...]:
        paths = _dataset_file_paths(self._dataset)
        return tuple(
            _file_identity(
                logical_path,
                resolved_path,
                error_code="PYTHIA_MMAP_ADAPTER_FILE_STAT_CHANGED",
            )
            for logical_path, resolved_path in paths
        )

    def _assert_files_unchanged(self) -> None:
        if self._file_stats() != self._initial_file_stats:
            raise RuntimeError("PYTHIA_MMAP_ADAPTER_FILE_STAT_CHANGED")

    def _sample_id(self, global_index: int) -> str:
        return f"pile:{self.asset_id}:record:{global_index:012d}"

    def _microbatch(
        self,
        global_indices: tuple[int, ...],
        *,
        batch_id: str,
        seed: int,
        rank: int,
        world_size: int,
        step: int,
        micro_index: int,
        rng_identity_sha256: str,
    ) -> TrainingMicrobatch:
        self._require_open()
        self._assert_files_unchanged()
        if len(global_indices) != self.microbatch_size:
            raise PythiaMMapProviderError("microbatch index count drifted")
        if any(
            index < self.split_start or index >= self.split_stop
            for index in global_indices
        ):
            raise PythiaMMapProviderError("microbatch index escaped the frozen split")
        records = np.stack(
            [
                self._dataset.raw_record(index - self._dataset.record_start)
                for index in global_indices
            ]
        )
        self._assert_files_unchanged()
        if records.shape != (self.microbatch_size, PYTHIA_TOKENS_PER_RECORD):
            raise PythiaMMapProviderError("Pythia record shape drifted after resolution")
        tokens = torch.from_numpy(records.astype(np.int64, copy=False))
        input_ids = tokens[:, :-1]
        target_ids = tokens[:, 1:]
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        return TrainingMicrobatch(
            batch_id,
            {
                "input_ids": input_ids,
                "target_ids": target_ids,
                "attention_mask": attention_mask,
            },
            tuple(self._sample_id(index) for index in global_indices),
            {
                "schema_version": _MICROBATCH_METADATA_VERSION,
                "asset_id": self.asset_id,
                "ready_manifest_sha256": self.ready_manifest_sha256,
                "qualification_sha256": self.qualification_sha256,
                "g3_resolution_artifact_hash": self.g3_resolution_artifact_hash,
                "g3_source_commit": self.g3_source_commit,
                "g3_runtime_lineage_sha256": self.g3_runtime_lineage_sha256,
                "split": [self.split_start, self.split_stop],
                "sampling_design": self.sampling_design,
                "seed": seed,
                "rank": rank,
                "world_size": world_size,
                "step": step,
                "micro_index": micro_index,
                "rng_identity_sha256": rng_identity_sha256,
                "global_record_indices": list(global_indices),
            },
        )

    def cursor(
        self, *, seed: int, rank: int = 0, world_size: int = 1
    ) -> "PythiaMMapBatchCursor":
        with self._lock:
            self._require_open()
            self._assert_files_unchanged()
            return PythiaMMapBatchCursor(
                self,
                seed=seed,
                rank=rank,
                world_size=world_size,
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._dataset.close()
            self._closed = True

    def __enter__(self) -> "PythiaMMapDatasetAdapter":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PythiaMMapFrozenSampleResolver:
    """Resolve exact frozen Pythia record IDs without advancing a cursor.

    Stage 2's :class:`SamplingPlan` and Stage 3's frozen probe artifact own
    sample selection.  This resolver therefore performs no shuffling, random
    draw, or cursor mutation: one canonical sample ID always maps to the same
    single-record, pre-shifted next-token microbatch.
    """

    def __init__(
        self,
        dataset: PythiaIndexedDataset,
        *,
        asset_id: str,
        ready_manifest_sha256: str,
        qualification_sha256: str,
        g3_resolution_artifact_hash: str,
        g3_source_commit: str,
        g3_runtime_lineage_sha256: str,
        split_start: int,
        split_stop: int,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> None:
        if not isinstance(dataset, PythiaIndexedDataset):
            raise TypeError("dataset must be a PythiaIndexedDataset")
        normalized_asset_id = _require_digest(asset_id, field="asset_id")
        normalized_manifest = _require_digest(
            ready_manifest_sha256, field="ready_manifest_sha256"
        )
        normalized_qualification = _require_digest(
            qualification_sha256, field="qualification_sha256"
        )
        normalized_resolution = _require_digest(
            g3_resolution_artifact_hash,
            field="g3_resolution_artifact_hash",
        )
        normalized_source_commit = _require_git_commit(
            g3_source_commit, field="g3_source_commit"
        )
        normalized_runtime_lineage = _require_digest(
            g3_runtime_lineage_sha256,
            field="g3_runtime_lineage_sha256",
        )
        start = _require_nonnegative_integer(split_start, field="split_start")
        stop = _require_nonnegative_integer(split_stop, field="split_stop")
        if (dataset.record_start, dataset.record_stop) != (start, stop):
            raise PythiaMMapProviderError(
                "frozen resolver dataset interval must exactly equal its split"
            )
        if dataset.tokens_per_record != PYTHIA_TOKENS_PER_RECORD:
            raise PythiaMMapProviderError(
                f"Pythia mmap resolver requires exactly {PYTHIA_TOKENS_PER_RECORD} "
                "tokens per record"
            )
        if not isinstance(sampling_design, str) or not sampling_design.strip():
            raise PythiaMMapProviderError("sampling_design must be non-empty")
        if type(weights_exogenous) is not bool or type(
            common_mean_assumption
        ) is not bool:
            raise TypeError("Pythia mmap weighting assumptions must be booleans")

        self._dataset = dataset
        self._asset_id = normalized_asset_id
        self._ready_manifest_sha256 = normalized_manifest
        self._qualification_sha256 = normalized_qualification
        self._g3_resolution_artifact_hash = normalized_resolution
        self._g3_source_commit = normalized_source_commit
        self._g3_runtime_lineage_sha256 = normalized_runtime_lineage
        self._split_start = start
        self._split_stop = stop
        self._sampling_design = sampling_design
        self._weights_exogenous = weights_exogenous
        self._common_mean_assumption = common_mean_assumption
        self._sample_prefix = f"pile:{self.asset_id}:record:"
        self._sample_ids = tuple(
            self._sample_id(index) for index in range(self.split_start, self.split_stop)
        )
        self._initial_file_stats = self._file_stats()
        self._closed = False
        self._lock = threading.RLock()
        self._identity_payload = {
            "schema_version": _FROZEN_RESOLVER_IDENTITY_VERSION,
            "asset_id": self.asset_id,
            "ready_manifest_sha256": self.ready_manifest_sha256,
            "qualification_sha256": self.qualification_sha256,
            "g3_resolution_artifact_hash": self.g3_resolution_artifact_hash,
            "g3_source_commit": self.g3_source_commit,
            "g3_runtime_lineage_sha256": self.g3_runtime_lineage_sha256,
            "split": [self.split_start, self.split_stop],
            "tokens_per_record": self._dataset.tokens_per_record,
            "sampling_design": self.sampling_design,
            "weights_exogenous": self.weights_exogenous,
            "common_mean_assumption": self.common_mean_assumption,
        }
        self._resolver_id = (
            "pythia-mmap-frozen-"
            + canonical_json_hash(self._identity_payload)[:32]
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PYTHIA_MMAP_FROZEN_RESOLVER_CLOSED")

    def _sample_id(self, global_index: int) -> str:
        return f"{self._sample_prefix}{global_index:012d}"

    def _file_stats(self) -> tuple[tuple[object, ...], ...]:
        paths = _dataset_file_paths(self._dataset)
        return tuple(
            _file_identity(
                logical_path,
                resolved_path,
                error_code="PYTHIA_MMAP_FROZEN_RESOLVER_FILE_STAT_CHANGED",
            )
            for logical_path, resolved_path in paths
        )

    @property
    def resolver_id(self) -> str:
        return self._resolver_id

    @property
    def sample_ids(self) -> tuple[Hashable, ...]:
        return self._sample_ids

    @property
    def asset_id(self) -> str:
        return self._asset_id

    @property
    def ready_manifest_sha256(self) -> str:
        return self._ready_manifest_sha256

    @property
    def qualification_sha256(self) -> str:
        return self._qualification_sha256

    @property
    def g3_resolution_artifact_hash(self) -> str:
        return self._g3_resolution_artifact_hash

    @property
    def g3_source_commit(self) -> str:
        return self._g3_source_commit

    @property
    def g3_runtime_lineage_sha256(self) -> str:
        return self._g3_runtime_lineage_sha256

    @property
    def split_start(self) -> int:
        return self._split_start

    @property
    def split_stop(self) -> int:
        return self._split_stop

    @property
    def loss_unit(self) -> str:
        return "target_token"

    @property
    def statistical_unit(self) -> str:
        return "pretokenized_pile_sequence_draw_group_mean"

    @property
    def weight_unit(self) -> str:
        return "effective_target_tokens"

    @property
    def sampling_design(self) -> str:
        return self._sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._common_mean_assumption

    def resolve(self, sample_id: Hashable) -> TrainingMicrobatch:
        with self._lock:
            self._require_open()
            if not isinstance(sample_id, str) or not sample_id.startswith(
                self._sample_prefix
            ):
                raise KeyError(f"FROZEN_SAMPLE_ID_UNKNOWN:{sample_id!r}")
            suffix = sample_id[len(self._sample_prefix) :]
            if len(suffix) != 12 or not suffix.isascii() or not suffix.isdigit():
                raise KeyError(f"FROZEN_SAMPLE_ID_UNKNOWN:{sample_id!r}")
            global_index = int(suffix)
            if (
                self._sample_id(global_index) != sample_id
                or global_index < self.split_start
                or global_index >= self.split_stop
            ):
                raise KeyError(f"FROZEN_SAMPLE_ID_UNKNOWN:{sample_id!r}")
            record = self._dataset.raw_record(
                global_index - self._dataset.record_start
            )
            if record.shape != (PYTHIA_TOKENS_PER_RECORD,):
                raise PythiaMMapProviderError(
                    "Pythia record shape drifted after frozen resolution"
                )
            tokens = torch.from_numpy(record.astype(np.int64, copy=False)).unsqueeze(0)
            input_ids = tokens[:, :-1]
            target_ids = tokens[:, 1:]
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            return TrainingMicrobatch(
                f"pythia-mmap-frozen:{global_index:012d}",
                {
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                    "attention_mask": attention_mask,
                },
                (sample_id,),
                {
                    "schema_version": _FROZEN_RESOLVER_METADATA_VERSION,
                    "asset_id": self.asset_id,
                    "ready_manifest_sha256": self.ready_manifest_sha256,
                    "qualification_sha256": self.qualification_sha256,
                    "split": [self.split_start, self.split_stop],
                    "sampling_design": self.sampling_design,
                    "global_record_index": global_index,
                },
            )

    def state_digest(self) -> str:
        with self._lock:
            self._require_open()
            return canonical_json_hash(
                {
                    **self._identity_payload,
                    "resolver_id": self.resolver_id,
                }
            )

    def assert_unchanged(self, expected_digest: str) -> None:
        _require_digest(expected_digest, field="expected_digest")
        if self._file_stats() != self._initial_file_stats:
            raise RuntimeError("PYTHIA_MMAP_FROZEN_RESOLVER_FILE_STAT_CHANGED")
        if self.state_digest() != expected_digest:
            raise RuntimeError("PYTHIA_MMAP_FROZEN_RESOLVER_STATE_CHANGED")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._dataset.close()
            self._closed = True

    def __enter__(self) -> "PythiaMMapFrozenSampleResolver":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PythiaMMapBatchCursor:
    """Strict, resumable cursor over one frozen Pythia record interval."""

    _STATE_FIELDS: Final = frozenset(
        {
            "schema_version",
            "asset_id",
            "ready_manifest_sha256",
            "qualification_sha256",
            "g3_resolution_artifact_hash",
            "g3_source_commit",
            "g3_runtime_lineage_sha256",
            "adapter_identity_sha256",
            "split",
            "sampling_design",
            "sampler_algorithm",
            "seed",
            "rank",
            "world_size",
            "microbatch_size",
            "microbatches_per_step",
            "rng_identity_sha256",
            "order_sha256",
            "position",
            "step",
        }
    )

    def __init__(
        self,
        adapter: PythiaMMapDatasetAdapter,
        *,
        seed: int,
        rank: int,
        world_size: int,
    ) -> None:
        if not isinstance(adapter, PythiaMMapDatasetAdapter):
            raise TypeError("adapter must be a PythiaMMapDatasetAdapter")
        normalized_seed = _require_nonnegative_integer(seed, field="seed")
        normalized_rank = _require_nonnegative_integer(rank, field="rank")
        normalized_world_size = _require_positive_integer(
            world_size, field="world_size"
        )
        if normalized_rank >= normalized_world_size:
            raise PythiaMMapProviderError("rank must be smaller than world_size")
        if normalized_seed > _MAX_CURSOR_POSITION:
            raise PythiaMMapProviderError("seed exceeds the supported 63-bit range")

        self._adapter = adapter
        self._seed = normalized_seed
        self._rank = normalized_rank
        self._world_size = normalized_world_size
        self._group_size = adapter.microbatch_size * adapter.microbatches_per_step
        self._design = PythiaSamplingDesign(adapter.sampling_design)
        self._sampler_algorithm = _SAMPLER_ALGORITHMS[self._design]
        self._lock = threading.RLock()
        self._closed = False
        self._position = 0
        self._step = 0

        sampler_identity = {
            "schema_version": _SAMPLER_IDENTITY_VERSION,
            "adapter_identity_sha256": adapter.identity_sha256,
            "sampling_design": self._design.value,
            "sampler_algorithm": self._sampler_algorithm,
            "seed": self._seed,
        }
        self._rng_identity_sha256 = canonical_json_hash(sampler_identity)

        self._order: tuple[int, ...] | None
        self._order_sha256: str | None
        self._usable_count: int | None
        if self._design is PythiaSamplingDesign.WITH_REPLACEMENT:
            self._order = None
            self._order_sha256 = None
            self._usable_count = None
        else:
            order = list(range(adapter.split_start, adapter.split_stop))
            if self._design is PythiaSamplingDesign.WITHOUT_REPLACEMENT:
                shuffle_seed = int(self._rng_identity_sha256, 16)
                random.Random(shuffle_seed).shuffle(order)
            local_order = tuple(order[self._rank :: self._world_size])
            usable = len(local_order) - len(local_order) % self._group_size
            if usable == 0:
                raise PythiaMMapProviderError(
                    "rank-local split is too small for one complete optimizer step"
                )
            self._order = local_order
            self._order_sha256 = _order_sha256(local_order)
            self._usable_count = usable

    @property
    def rng_identity_sha256(self) -> str:
        return self._rng_identity_sha256

    @property
    def position(self) -> int:
        return self._position

    @property
    def step(self) -> int:
        return self._step

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("PYTHIA_MMAP_CURSOR_CLOSED")
        self._adapter._require_open()

    def _replacement_index(self, draw_position: int) -> int:
        """Return one unbiased deterministic draw from the frozen split."""

        width = self._adapter.split_stop - self._adapter.split_start
        population = 1 << 256
        acceptance_limit = population - population % width
        key = bytes.fromhex(self._rng_identity_sha256)
        attempt = 0
        while True:
            digest = hashlib.sha256(
                b"pythia-mmap-with-replacement-draw-v1\x00"
                + key
                + self._rank.to_bytes(8, "big", signed=False)
                + self._world_size.to_bytes(8, "big", signed=False)
                + draw_position.to_bytes(8, "big", signed=False)
                + attempt.to_bytes(8, "big", signed=False)
            ).digest()
            candidate = int.from_bytes(digest, "big", signed=False)
            if candidate < acceptance_limit:
                return self._adapter.split_start + candidate % width
            attempt += 1
            if attempt > _MAX_CURSOR_POSITION:  # pragma: no cover - cryptographic bound
                raise RuntimeError("PYTHIA_MMAP_REJECTION_SAMPLER_EXHAUSTED")

    def _next_indices(self) -> tuple[int, ...]:
        end = self._position + self._group_size
        if end > _MAX_CURSOR_POSITION:
            raise PythiaMMapProviderError("cursor position exceeds the 63-bit boundary")
        if self._design is PythiaSamplingDesign.WITH_REPLACEMENT:
            return tuple(
                self._replacement_index(position)
                for position in range(self._position, end)
            )
        assert self._order is not None and self._usable_count is not None
        if end > self._usable_count:
            raise StopIteration("PYTHIA_MMAP_CURSOR_EXHAUSTED")
        return self._order[self._position : end]

    def next_microbatches(self) -> tuple[TrainingMicrobatch, ...]:
        with self._lock, self._adapter._lock:
            self._require_open()
            self._adapter._assert_files_unchanged()
            indices = self._next_indices()
            microbatches: list[TrainingMicrobatch] = []
            for micro_index in range(self._adapter.microbatches_per_step):
                start = micro_index * self._adapter.microbatch_size
                stop = start + self._adapter.microbatch_size
                micro_indices = indices[start:stop]
                microbatches.append(
                    self._adapter._microbatch(
                        micro_indices,
                        batch_id=(
                            f"{self._adapter.asset_id[:16]}:{self._design.value}:"
                            f"seed-{self._seed}:rank-{self._rank}:"
                            f"step-{self._step:012d}:micro-{micro_index:04d}"
                        ),
                        seed=self._seed,
                        rank=self._rank,
                        world_size=self._world_size,
                        step=self._step,
                        micro_index=micro_index,
                        rng_identity_sha256=self._rng_identity_sha256,
                    )
                )
            self._position += self._group_size
            self._step += 1
            self._adapter._assert_files_unchanged()
            return tuple(microbatches)

    def _static_state(self) -> dict[str, object]:
        return {
            "schema_version": _CURSOR_STATE_VERSION,
            "asset_id": self._adapter.asset_id,
            "ready_manifest_sha256": self._adapter.ready_manifest_sha256,
            "qualification_sha256": self._adapter.qualification_sha256,
            "g3_resolution_artifact_hash": (
                self._adapter.g3_resolution_artifact_hash
            ),
            "g3_source_commit": self._adapter.g3_source_commit,
            "g3_runtime_lineage_sha256": (
                self._adapter.g3_runtime_lineage_sha256
            ),
            "adapter_identity_sha256": self._adapter.identity_sha256,
            "split": [self._adapter.split_start, self._adapter.split_stop],
            "sampling_design": self._design.value,
            "sampler_algorithm": self._sampler_algorithm,
            "seed": self._seed,
            "rank": self._rank,
            "world_size": self._world_size,
            "microbatch_size": self._adapter.microbatch_size,
            "microbatches_per_step": self._adapter.microbatches_per_step,
            "rng_identity_sha256": self._rng_identity_sha256,
            "order_sha256": self._order_sha256,
        }

    def state_dict(self) -> Mapping[str, object]:
        with self._lock:
            self._require_open()
            self._adapter._assert_files_unchanged()
            return self._static_state() | {
                "position": self._position,
                "step": self._step,
            }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        with self._lock:
            self._require_open()
            self._adapter._assert_files_unchanged()
            if not isinstance(state, Mapping) or set(state) != self._STATE_FIELDS:
                raise PythiaMMapProviderError(
                    "PYTHIA_MMAP_CURSOR_STATE_FIELDS_MISMATCH"
                )
            for field, expected in self._static_state().items():
                observed = state[field]
                if type(observed) is not type(expected) or observed != expected:
                    raise PythiaMMapProviderError(
                        f"PYTHIA_MMAP_CURSOR_STATE_IDENTITY_MISMATCH:{field}"
                    )
            position = state["position"]
            step = state["step"]
            if (
                isinstance(position, bool)
                or not isinstance(position, int)
                or isinstance(step, bool)
                or not isinstance(step, int)
                or position < 0
                or step < 0
                or position > _MAX_CURSOR_POSITION
                or position != step * self._group_size
            ):
                raise PythiaMMapProviderError(
                    "PYTHIA_MMAP_CURSOR_STATE_POSITION_INVALID"
                )
            if self._usable_count is not None and position > self._usable_count:
                raise PythiaMMapProviderError(
                    "PYTHIA_MMAP_CURSOR_STATE_POSITION_OUT_OF_RANGE"
                )
            self._position = position
            self._step = step

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "PythiaMMapBatchCursor":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "PythiaMMapBatchCursor",
    "PythiaMMapDatasetAdapter",
    "PythiaMMapFrozenSampleResolver",
    "PythiaMMapProviderError",
    "PythiaSamplingDesign",
]
