"""Compact reference snapshot/reconstruction contract tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import load_canonical_json
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_formal import (
    ReferenceSizingPlan,
    StreamingReferenceSizer,
    _ReferenceSnapshotStore,
)
from param_importance_nlp.providers import SyntheticGradientProvider
from param_importance_nlp.runtime.tensor_bundle import load_tensor_bundle


def _provider() -> SyntheticGradientProvider:
    return SyntheticGradientProvider(
        {
            index: {"p": np.array([float(index + 1), -float(index + 2)])}
            for index in range(8)
        },
        statistical_unit="compact-smoke",
        weight_unit="draws",
        sampling_design="fixed_disjoint",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _draws(stream: str, count: int):
    return SamplingPlan(
        SamplingUniverse("compact", tuple(range(8))),
        {
            "reference_sizing": 11,
            "reference_A": 17,
            "reference_B": 23,
            "pilot": 29,
            "confirmatory": 31,
        },
    ).draws(stream, count)


def _plan() -> ReferenceSizingPlan:
    return ReferenceSizingPlan("compact-reference", (1, 2, 3), 1, 1e6, 1)


def test_snapshot_object_stores_hashes_and_shard_refs_only(tmp_path: Path) -> None:
    root = tmp_path / "compact"
    result = StreamingReferenceSizer(_provider()).run(
        _plan(),
        draws_a=_draws("reference_A", 3),
        draws_b=_draws("reference_B", 3),
        artifact_root=root,
    )
    commit = load_canonical_json(root / "commits" / "00000003.json")
    state, _bundle = load_tensor_bundle(root / str(commit["object_ref"]))
    assert state["snapshot_encoding"] == _ReferenceSnapshotStore._COMPACT_ENCODING
    assert "g1" not in state["a"] and "g2" not in state["a"]
    assert "g1_hash" in state["a"] and "g2_hash" in state["a"]
    restored = _ReferenceSnapshotStore.materialize(state, root)
    assert _ReferenceSnapshotStore._state_digest(restored) == commit["state_digest"]
    assert result.processed_sample_count_per_stream == 3


def test_compact_resume_reconstructs_exact_reference(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    interrupted = StreamingReferenceSizer(_provider()).run(
        _plan(),
        draws_a=_draws("reference_A", 3),
        draws_b=_draws("reference_B", 3),
        artifact_root=root,
        max_new_block_pairs=2,
    )
    resumed = StreamingReferenceSizer(_provider()).run(
        _plan(),
        draws_a=_draws("reference_A", 3),
        draws_b=_draws("reference_B", 3),
        artifact_root=root,
    )
    uninterrupted = StreamingReferenceSizer(_provider()).run(
        _plan(),
        draws_a=_draws("reference_A", 3),
        draws_b=_draws("reference_B", 3),
        artifact_root=tmp_path / "uninterrupted",
    )
    assert interrupted.resumed_from_block_pairs == 0
    assert resumed.resumed_from_block_pairs == 2
    for name in resumed.bias_reference:
        np.testing.assert_array_equal(
            resumed.bias_reference[name], uninterrupted.bias_reference[name]
        )


def test_compact_snapshot_tampering_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "tamper"
    StreamingReferenceSizer(_provider()).run(
        _plan(),
        draws_a=_draws("reference_A", 3),
        draws_b=_draws("reference_B", 3),
        artifact_root=root,
    )
    commit = load_canonical_json(root / "commits" / "00000003.json")
    object_path = root / str(commit["object_ref"])
    state, _bundle = load_tensor_bundle(object_path)
    state["a"]["g1_hash"] = "0" * 64
    with pytest.raises((ValueError, FileNotFoundError)):
        _ReferenceSnapshotStore.materialize(state, root)
