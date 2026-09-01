"""Stage 2/3 formal 编排层的本机 fixture 与 fail-closed 测试。

这些测试只使用 synthetic provider、解析路径和 BLOCKED Gate；它们不得生成任何
formal PASS。测试目的在于证明恢复/身份/数学语义，而不是冒充服务器验收。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    canonical_json_hash,
    validate_stage23_artifact,
)
from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.contracts.stage23 import FormalExecutionEvidence
from param_importance_nlp.core.quadrature import (
    composite_simpson_rule,
    gauss_legendre_rule,
)
from param_importance_nlp.core.tensors import TensorMap
from param_importance_nlp.experiments.sampling import (
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
)
from param_importance_nlp.experiments.stage2_formal import (
    FormalExperimentPlan,
    PilotCellObservation,
    PilotThresholds,
    RecoverablePairedWaveRunner,
    ReferenceSizingPlan,
    Stage2RecommendationEngine,
    StreamingReferenceSizer,
)
from param_importance_nlp.experiments.stage3 import (
    EndpointRecord,
    EndpointState,
    NodeCacheKey,
    ProbeSpec,
)
from param_importance_nlp.experiments.stage3_raw_storage import RAW_SHARD_SCHEMA
from param_importance_nlp.experiments.stage3_formal import (
    EndpointCaptureCoordinator,
    EndpointCaptureRequest,
    PersistentNodeGradientCache,
    _write_immutable_canonical_json,
    ProbePanel,
    ProbePanelEntry,
    QuadratureObservation,
    QuadratureRecommendationEngine,
    QuadratureThresholds,
    ReferenceRefinementRunner,
    ReferenceRuleLevel,
)
from param_importance_nlp.providers import SyntheticGradientProvider


def _hash(character: str) -> str:
    return character * 64


def _blocked_gate(stage: int) -> GateRecord:
    return GateRecord(
        gate_id=f"stage{stage}.G{stage}-FORMAL",
        stage=stage,
        status=GateStatus.BLOCKED,
        checked_at="2026-07-22T00:00:00+00:00",
        reasons=("server_unreachable",),
    )


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("stage23-fixture", tuple(range(8))),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )


def _provider() -> SyntheticGradientProvider:
    return SyntheticGradientProvider(
        {
            sample_id: {
                "layer.weight": np.array(
                    [1.0 + 0.1 * sample_id, (-1.0) ** sample_id], dtype=np.float64
                ),
                "layer.bias": np.array([0.25 * sample_id - 0.5], dtype=np.float64),
            }
            for sample_id in range(8)
        },
        fixed_state_id="stage23-synthetic-state",
        statistical_unit="synthetic_draw_group_mean",
        weight_unit="synthetic_draw_count",
        sampling_design="uniform_with_replacement_disjoint_draw_groups",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def test_formal_experiment_plan_freezes_stage2_bmr_and_roundtrips() -> None:
    plan = FormalExperimentPlan(
        plan_id="stage2-pilot-plan",
        task_id="stage2.06_pilot_and_matrix_freeze",
        wave_id="stage2-pilot-wave",
        cell_id="fixed-state-cell",
        stream="pilot",
        batch_size=32,
        microbatch_counts=(4, 8, 16, 32),
        repetitions=7,
        sampling_plan_hash=_sampling().digest,
        execution_evidence_hash=_hash("e"),
        source_artifact_refs=("commits/z.json", "commits/a.json"),
        selection_basis="preregistered_pilot",
        pilot_thresholds={
            "bias_margin": 0.05,
            "max_corrected_nmse_ratio": 1.0,
            "min_spearman": 0.9,
            "min_topk_overlap": 0.8,
            "max_online_cost_ratio": 1.5,
        },
    )
    assert plan.source_artifact_refs == ("commits/a.json", "commits/z.json")
    restored = FormalExperimentPlan.from_mapping(plan.to_dict())
    assert restored.to_dict() == plan.to_dict()
    assert validate_stage23_artifact(plan.to_dict()).artifact_hash == plan.artifact_hash

    tampered = plan.to_dict()
    tampered["repetitions"] = 8
    with pytest.raises(ValueError, match="artifact_hash"):
        FormalExperimentPlan.from_mapping(tampered)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"stream": "confirmatory"}, "STREAM_TASK_MISMATCH"),
        ({"batch_size": 16}, "BATCH_NOT_PREREGISTERED"),
        ({"microbatch_counts": (4, 4)}, "NOT_SORTED_UNIQUE"),
        ({"pilot_thresholds": None}, "PILOT_THRESHOLDS_REQUIRED"),
    ],
)
def test_formal_experiment_plan_rejects_unfrozen_dimensions(
    changes: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "plan_id": "stage2-pilot-plan",
        "task_id": "stage2.06_pilot_and_matrix_freeze",
        "wave_id": "stage2-pilot-wave",
        "cell_id": "fixed-state-cell",
        "stream": "pilot",
        "batch_size": 32,
        "microbatch_counts": (4, 8),
        "repetitions": 4,
        "sampling_plan_hash": _sampling().digest,
        "execution_evidence_hash": _hash("e"),
        "source_artifact_refs": ("commits/source.json",),
        "selection_basis": "preregistered_pilot",
        "pilot_thresholds": {
            "bias_margin": 0.05,
            "max_corrected_nmse_ratio": 1.0,
            "min_spearman": 0.9,
            "min_topk_overlap": 0.8,
            "max_online_cost_ratio": 1.5,
        },
    }
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=message):
        FormalExperimentPlan(**values)  # type: ignore[arg-type]


def test_formal_confirmatory_plan_requires_one_primary_u_group() -> None:
    common = {
        "plan_id": "stage2-confirmatory-plan",
        "task_id": "stage2.07_main_sweep",
        "wave_id": "stage2-confirmatory-wave",
        "cell_id": "fixed-state-cell",
        "stream": "confirmatory",
        "batch_size": 32,
        "repetitions": 8,
        "sampling_plan_hash": _sampling().digest,
        "execution_evidence_hash": _hash("e"),
        "source_artifact_refs": ("commits/source.json",),
        "selection_basis": "pilot_frozen_primary",
    }
    with pytest.raises(ValueError, match="ONE_PRIMARY_M_GT_2"):
        FormalExperimentPlan(microbatch_counts=(2,), **common)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ONE_PRIMARY_M_GT_2"):
        FormalExperimentPlan(microbatch_counts=(4, 8), **common)  # type: ignore[arg-type]


def test_formal_execution_evidence_rejects_blocked_gate_and_fixture_cannot_promote() -> None:
    fixture = FormalExecutionEvidence("local_fixture")
    assert fixture.formal_eligible is False
    assert FormalExecutionEvidence.from_mapping(fixture.to_dict()).artifact_hash == (
        fixture.artifact_hash
    )
    with pytest.raises(FormalRunRejected, match="FORMAL_RUN_INTENT_REQUIRED"):
        fixture.require_for_stage(2)

    blocked = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_hash("a"),
        asset_manifest_hashes=(_hash("b"),),
        prerequisite_gates=(_blocked_gate(1),),
    )
    assert blocked.formal_eligible is False
    with pytest.raises(FormalRunRejected, match="NOT_ELIGIBLE"):
        blocked.require_for_stage(2)


def test_streaming_reference_sizing_resumes_from_authoritative_block_commit(
    tmp_path: Path,
) -> None:
    sampling = _sampling()
    plan = ReferenceSizingPlan(
        reference_id="reference-sizing-fixture",
        candidate_sample_counts=(4, 8, 12),
        block_size=2,
        convergence_tolerance=1e6,
        required_consecutive=1,
    )
    draws_a = sampling.draws("reference_A", 12)
    draws_b = sampling.draws("reference_B", 12)
    interrupted_root = tmp_path / "reference-interrupted"

    interrupted = StreamingReferenceSizer(_provider()).run(
        plan,
        draws_a=draws_a,
        draws_b=draws_b,
        artifact_root=interrupted_root,
        max_new_block_pairs=2,
    )
    assert interrupted.converged is False
    assert interrupted.processed_sample_count_per_stream == 4
    assert interrupted.to_dict()["formal_eligible"] is False

    resumed = StreamingReferenceSizer(_provider()).run(
        plan,
        draws_a=draws_a,
        draws_b=draws_b,
        artifact_root=interrupted_root,
    )
    uninterrupted = StreamingReferenceSizer(_provider()).run(
        plan,
        draws_a=draws_a,
        draws_b=draws_b,
        artifact_root=tmp_path / "reference-uninterrupted",
    )
    assert resumed.converged and uninterrupted.converged
    assert resumed.selected_sample_count_per_stream == 8
    # Convergence freezes the first eligible B, but sizing must still publish
    # every pre-registered candidate prefix for the S2.4 delta_sci contract.
    assert resumed.processed_sample_count_per_stream == 12
    assert {
        int(path.stem)
        for path in interrupted_root.joinpath("commits").glob("*.json")
    } >= {2, 4, 6}
    assert resumed.resumed_from_block_pairs == 2
    assert validate_stage23_artifact(resumed.to_dict()).artifact_hash == (
        resumed.artifact_hash
    )
    for name in resumed.bias_reference:
        np.testing.assert_array_equal(
            resumed.bias_reference[name], uninterrupted.bias_reference[name]
        )
        np.testing.assert_array_equal(
            resumed.cross_reference[name], uninterrupted.cross_reference[name]
        )

    # 未绑定正式 Gate 的共享 reference manifest 仍明确不可 formal。
    wire = resumed.to_reference_artifact(
        reference_id="fixture-reference",
        block_size=2,
        tensor_bundle_ref="objects/reference-fixture",
        tensor_bundle_manifest_hash=_hash("c"),
    )
    assert wire["scope"] == "local_fixture"
    assert wire["formal_eligible"] is False


def test_reference_zero_norm_comparison_is_undefined_without_epsilon(
    tmp_path: Path,
) -> None:
    values = (1.0, 1.0, 1.0, 1.0, -6.0, -6.0, -6.0, 3.0)
    provider = SyntheticGradientProvider(
        {index: {"p": np.array([value])} for index, value in enumerate(values)},
        statistical_unit="synthetic_draw",
        weight_unit="draw_count",
        sampling_design="fixed_disjoint_fixture",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    result = StreamingReferenceSizer(provider).run(
        ReferenceSizingPlan("zero-reference", (2, 4), 1, 1e-6, 1),
        draws_a=(0, 1, 4, 5),
        draws_b=(2, 3, 6, 7),
        artifact_root=tmp_path / "zero-reference",
    )
    assert result.converged is False
    assert result.points[-1].comparison_defined is False
    assert result.points[-1].comparison_reason == "zero_reference_l1_norm"
    assert result.points[-1].normalized_l1_from_previous is None
    validate_stage23_artifact(result.to_dict())


def test_paired_wave_recovery_does_not_repeat_committed_repetition(
    tmp_path: Path,
) -> None:
    sampling = _sampling()
    sizing = StreamingReferenceSizer(_provider()).run(
        ReferenceSizingPlan(
            "wave-reference",
            (4, 8, 12),
            2,
            1e6,
            1,
        ),
        draws_a=sampling.draws("reference_A", 12),
        draws_b=sampling.draws("reference_B", 12),
        artifact_root=tmp_path / "wave-reference",
    )
    mappings = tuple(
        RepetitionMapping.create(
            repetition_id=f"wave-rep-{index}",
            draws=sampling.draws("confirmatory", 8, start=index * 8),
            m_values=(2, 4, 8),
        )
        for index in range(2)
    )
    root = tmp_path / "wave"
    first = RecoverablePairedWaveRunner(_provider()).run(
        wave_id="paired-wave-fixture",
        mappings=mappings,
        reference=sizing.bias_reference,
        reference_hash=str(sizing.to_dict()["bias_reference_hash"]),
        artifact_root=root,
        max_new_units=1,
    )
    assert first.complete is False
    assert len(first.completed_unit_ids) == 1

    resumed = RecoverablePairedWaveRunner(_provider()).run(
        wave_id="paired-wave-fixture",
        mappings=mappings,
        reference=sizing.bias_reference,
        reference_hash=str(sizing.to_dict()["bias_reference_hash"]),
        artifact_root=root,
    )
    assert resumed.complete is True
    assert resumed.resumed_unit_count == 1
    assert set(resumed.method_statistics) == {"raw", "double", "u_m2", "u_m4", "u_m8"}
    assert set(resumed.cost_statistics) == {
        "scientific_equal_sample_cost",
        "isolated_estimator_cost",
        "online_training_incremental_cost",
    }
    assert resumed.cost_statistics["isolated_estimator_cost"]["defined"] is False
    assert resumed.to_dict()["formal_eligible"] is False
    assert validate_stage23_artifact(resumed.to_dict()).artifact_hash == (
        resumed.artifact_hash
    )
    validate_stage23_artifact(load_canonical_json(root / "wave-plan.json"))


def test_stage2_recommendation_is_fixture_only_without_formal_gate() -> None:
    thresholds = PilotThresholds(0.2, 1.1, 0.8, 0.7, 1.25)
    observations = [
        PilotCellObservation(
            cell,
            estimator,
            32,
            4,
            8,
            -0.1,
            0.1,
            1.0,
            0.9,
            0.85,
            1.1,
        )
        for estimator in ("u", "double")
        for cell in ("model14-early", "model31-late")
    ]
    recommendation = Stage2RecommendationEngine().recommend(
        recommendation_id="fixture-estimator-recommendation",
        observations=observations,
        required_cells=("model14-early", "model31-late"),
        thresholds=thresholds,
    )
    assert recommendation.status == "FIXTURE_RECOMMENDATION"
    assert recommendation.selected_estimator == "u"
    assert recommendation.to_dict()["formal_eligible"] is False
    assert validate_stage23_artifact(recommendation.to_dict()).artifact_hash == (
        recommendation.artifact_hash
    )
    with pytest.raises(FormalRunRejected, match="FORMAL_RUN_INTENT_REQUIRED"):
        recommendation.qualify(
            execution=FormalExecutionEvidence("local_fixture"),
            gate=_blocked_gate(2),
            artifact_ref="decisions/stage2.json",
        )


def _state(
    artifact_id: str,
    artifact: str,
    parameter: str,
    *,
    optimizer: str,
    rng: str,
    cursor: str,
) -> EndpointState:
    return EndpointState(
        artifact_id=artifact_id,
        artifact_hash=_hash(artifact),
        parameter_hash=_hash(parameter),
        buffer_hash=_hash("d"),
        optimizer_hash=_hash(optimizer),
        scheduler_hash=_hash("4"),
        scaler_hash=_hash("5"),
        rng_hash=_hash(rng),
        data_cursor_hash=_hash(cursor),
        model_mode_hash=_hash("a"),
    )


class _EndpointAdapter:
    def __init__(self, *, replay: bool = True) -> None:
        self.calls: list[str] = []
        self.replay = replay
        self.restored = False
        self.pre = _state("pre", "b", "c", optimizer="0", rng="6", cursor="8")
        self.post = _state(
            "parameter-post", "e", "f", optimizer="3", rng="6", cursor="8"
        )
        self.commit = _state(
            "attempt-commit", "1", "f", optimizer="3", rng="7", cursor="9"
        )

    def capture_pre_state(self) -> EndpointState:
        self.calls.append("capture_pre")
        return self.pre

    def apply_optimizer_update(self) -> None:
        self.calls.append("optimizer_update")

    def capture_parameter_post_state(self) -> EndpointState:
        self.calls.append("capture_post")
        return self.post

    def advance_attempt_commit(self) -> None:
        self.calls.append("attempt_commit")

    def capture_attempt_commit_state(self) -> EndpointState:
        self.calls.append("capture_commit")
        return self.commit

    def full_update_delta_hash(self) -> str:
        return _hash("2")

    def probe_buffer_snapshot_hash(self) -> str:
        return _hash("d")

    def verify_replay(self, _record: EndpointRecord) -> bool:
        self.calls.append("verify_replay")
        return self.replay

    def restore_pre_state(self) -> None:
        self.calls.append("restore_pre")
        self.restored = True


def _capture_request() -> EndpointCaptureRequest:
    return EndpointCaptureRequest(
        path_state_id="path-state-fixture",
        source_run_id="source-run-fixture",
        optimizer_step=7,
        parameter_registry_hash=_hash("a"),
        update_sample_ids=(10, 11),
    )


def test_endpoint_capture_orders_states_and_rolls_back_failed_replay() -> None:
    adapter = _EndpointAdapter()
    captured = EndpointCaptureCoordinator().capture(_capture_request(), adapter)
    assert adapter.calls == [
        "capture_pre",
        "optimizer_update",
        "capture_post",
        "attempt_commit",
        "capture_commit",
        "verify_replay",
    ]
    assert captured.record.parameter_post_state.artifact_id == "parameter-post"
    assert captured.record.attempt_commit_state.artifact_id == "attempt-commit"
    assert captured.record.replay_verified
    assert captured.to_dict()["formal_eligible"] is False
    assert validate_stage23_artifact(captured.to_dict()).artifact_hash == (
        captured.artifact_hash
    )

    failing = _EndpointAdapter(replay=False)
    with pytest.raises(RuntimeError, match="REPLAY_VERIFICATION_FAILED"):
        EndpointCaptureCoordinator().capture(_capture_request(), failing)
    assert failing.restored
    assert failing.calls[-1] == "restore_pre"


def test_probe_panel_rejects_update_and_cross_probe_overlap() -> None:
    endpoint = EndpointCaptureCoordinator().capture(
        _capture_request(), _EndpointAdapter()
    ).record
    loss_hash = _hash("9")
    panel = ProbePanel.build(
        panel_id="fixture-probe-panel",
        endpoint=endpoint,
        entries=(
            ProbePanelEntry(
                "formal", ProbeSpec("probe-a", (20, 21), _hash("3"), loss_hash)
            ),
            ProbePanelEntry(
                "replay", ProbeSpec("probe-b", (30, 31), _hash("4"), loss_hash)
            ),
        ),
    )
    assert panel.to_dict()["formal_eligible"] is False
    assert validate_stage23_artifact(panel.to_dict()).artifact_hash == panel.artifact_hash

    with pytest.raises(ValueError, match="重叠"):
        ProbePanel.build(
            panel_id="bad-update-panel",
            endpoint=endpoint,
            entries=(
                ProbePanelEntry(
                    "formal", ProbeSpec("bad-update", (11, 40), _hash("5"), loss_hash)
                ),
            ),
        )
    with pytest.raises(ValueError, match="不同 probe"):
        ProbePanel.build(
            panel_id="bad-cross-panel",
            endpoint=endpoint,
            entries=(
                ProbePanelEntry(
                    "formal", ProbeSpec("bad-a", (40, 41), _hash("6"), loss_hash)
                ),
                ProbePanelEntry(
                    "replay", ProbeSpec("bad-b", (41, 42), _hash("7"), loss_hash)
                ),
            ),
        )


def test_persistent_node_cache_survives_process_restart_and_is_immutable(
    tmp_path: Path,
) -> None:
    key = NodeCacheKey("path-unit-fixture", 0.5, "float64", _hash("a"), _hash("b"))
    root = tmp_path / "node-cache"
    first = PersistentNodeGradientCache(root)
    assert first.publish_many({key: {"x": np.array([1.0, 2.0])}}) == 1
    assert first.publish_many({key: {"x": np.array([1.0, 2.0])}}) == 0

    resumed = PersistentNodeGradientCache(root)
    observed = resumed.get(key)
    assert isinstance(observed, dict)
    observed["x"][0] = 99.0
    np.testing.assert_array_equal(resumed.get(key)["x"], np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="IMMUTABLE_KEY_CONFLICT"):
        resumed.publish_many({key: {"x": np.array([9.0, 2.0])}})

    tensor_map_key = NodeCacheKey(
        "path-unit-fixture", 0.75, "float64", _hash("a"), _hash("b")
    )
    assert resumed.publish_many(
        {
            tensor_map_key: TensorMap(
                {"x": torch.tensor([3.0], dtype=torch.float64)}
            )
        }
    ) == 1
    restored_map = PersistentNodeGradientCache(root).get(tensor_map_key)
    assert isinstance(restored_map, TensorMap)
    restored_map["x"].fill_(17.0)
    assert PersistentNodeGradientCache(root).get(tensor_map_key)["x"].item() == 3.0
    assert resumed.reconcile()["orphan_objects"] == []


def _retention_fixture(tmp_path: Path) -> tuple[PersistentNodeGradientCache, Path, Path, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    cache_root = tmp_path / "retention-cache"
    receipt_root = tmp_path / "retention-receipts"
    raw_shard = tmp_path / "downstream.raw"
    raw_body: dict[str, object] = {
        "schema_version": RAW_SHARD_SCHEMA,
        "unit_id": "retention-unit",
        "candidate_rule_names": ["fixture-rule"],
    }
    raw_body["artifact_hash"] = canonical_json_hash(raw_body)
    write_canonical_json(raw_shard, raw_body)
    raw_hash = str(raw_body["artifact_hash"])
    cache = PersistentNodeGradientCache(cache_root)
    key = NodeCacheKey("retention-unit", 0.5, "float64", _hash("a"), _hash("b"))
    assert cache.publish_many({key: {"gradient": np.array([1.0, 2.0])}}) == 1
    return cache, receipt_root, raw_shard, str(raw_body["artifact_hash"])


def test_persistent_node_cache_live_evidence_roundtrips_after_seal_and_evict(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    all_keys = cache.keys()
    assert all_keys
    live = cache.commit_evidence(all_keys)
    sealed = cache.seal(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    evicted = cache.finalize_eviction(
        str(sealed["receipt_ref"]), receipt_root=receipt_root
    )
    fresh = PersistentNodeGradientCache(cache.root)
    recovered = fresh.commit_evidence_from_sealed(
        all_keys,
        receipt_root / str(sealed["receipt_ref"]),
        receipt_root / str(evicted["tombstone_ref"]),
    )
    assert recovered == live
    assert recovered["authoritative_commits"] == live["authoritative_commits"]
    assert recovered["reconciliation"] == live["reconciliation"]
    assert recovered["evidence_hash"] == live["evidence_hash"]


def test_persistent_node_cache_mapping_evicted_rejects_hash_and_binding_drift(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    all_keys = cache.keys()
    sealed = cache.seal(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    evicted = cache.finalize_eviction(
        str(sealed["receipt_ref"]), receipt_root=receipt_root
    )
    sealed_value = load_canonical_json(receipt_root / str(sealed["receipt_ref"]))
    evicted_value = load_canonical_json(receipt_root / str(evicted["tombstone_ref"]))
    assert isinstance(sealed_value, dict)
    assert isinstance(evicted_value, dict)

    hash_drift = dict(evicted_value)
    hash_drift["sealed_receipt_hash"] = _hash("f")
    hash_drift["tombstone_hash"] = canonical_json_hash(
        {key: value for key, value in hash_drift.items() if key != "tombstone_hash"}
    )
    with pytest.raises(ValueError, match="NODE_CACHE_EVICTED_SEAL_HASH_MISMATCH"):
        PersistentNodeGradientCache(cache.root).commit_evidence_from_sealed(
            all_keys, sealed_value, hash_drift
        )

    binding_drift = dict(evicted_value)
    binding_drift["cache_root_ref"] = (tmp_path / "different-cache").as_posix()
    binding_drift["tombstone_hash"] = canonical_json_hash(
        {
            key: value
            for key, value in binding_drift.items()
            if key != "tombstone_hash"
        }
    )
    with pytest.raises(ValueError, match="NODE_CACHE_EVICTED_BINDING_MISMATCH"):
        PersistentNodeGradientCache(cache.root).commit_evidence_from_sealed(
            all_keys, sealed_value, binding_drift
        )


def test_persistent_node_cache_receipt_writer_is_idempotent_and_conflict_safe(
    tmp_path: Path,
) -> None:
    receipt = tmp_path / "receipt.json"
    payload = {"schema_version": "stage3-receipt-test-v1", "value": 1}
    _write_immutable_canonical_json(receipt, payload)
    first_bytes = receipt.read_bytes()
    _write_immutable_canonical_json(receipt, payload)
    assert receipt.read_bytes() == first_bytes
    with pytest.raises(ValueError, match="NODE_CACHE_RECEIPT_IMMUTABLE_CONFLICT"):
        _write_immutable_canonical_json(
            receipt, {"schema_version": "stage3-receipt-test-v1", "value": 2}
        )


def test_persistent_node_cache_seal_and_evict_is_receipted_and_reinstantiable(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    tombstone = cache.seal_and_evict(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    assert tombstone["state"] == "EVICTED"
    assert list((tmp_path / "retention-cache").rglob("*.json")) == []
    assert PersistentNodeGradientCache(cache.root).keys() == ()
    verified = PersistentNodeGradientCache.verify_receipt(
        receipt_root, receipt_root / str(tombstone["tombstone_ref"])
    )
    assert verified["state"] == "EVICTED"
    assert verified["key_digests"]
    assert verified["commit_artifact_hashes"]
    assert verified["object_manifest_hashes"]
    assert verified["downstream_raw_shard_hash"] == raw_hash
    assert verified["downstream_raw_shard_file_sha256"] == hashlib.sha256(
        raw_shard.read_bytes()
    ).hexdigest()

    tombstone_path = receipt_root / str(tombstone["tombstone_ref"])
    forged = dict(load_canonical_json(tombstone_path))
    forged["key_digests"] = []
    forged["tombstone_hash"] = canonical_json_hash(
        {key: value for key, value in forged.items() if key != "tombstone_hash"}
    )
    write_canonical_json(tombstone_path, forged)
    with pytest.raises(ValueError, match="NODE_CACHE_EVICTED_BINDING_MISMATCH"):
        PersistentNodeGradientCache.verify_receipt(tombstone_path)


def test_persistent_node_cache_seal_only_can_finalize_after_restart(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    sealed = cache.seal(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    resumed = PersistentNodeGradientCache(cache.root, receipt_root=receipt_root)
    tombstone = resumed.finalize_eviction(str(sealed["receipt_ref"]))
    assert tombstone["state"] == "EVICTED"
    assert PersistentNodeGradientCache.verify_receipt(
        receipt_root / str(tombstone["tombstone_ref"])
    )["state"] == "EVICTED"


def test_persistent_node_cache_retention_never_deletes_writer_lock_or_partial_drift(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    sealed = cache.seal(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    lock_path = cache.root / "writer.lock"
    lock_path.write_bytes(b"active writer")
    with pytest.raises(ValueError, match="WRITER_LOCK"):
        cache.finalize_eviction(str(sealed["receipt_ref"]), receipt_root=receipt_root)
    assert lock_path.exists()


def test_persistent_node_cache_retention_finalizes_partial_delete(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    sealed = cache.seal(
        scope="formal",
        unit_id="retention-unit",
        plan_hash=_hash("c"),
        run_config_hash=_hash("d"),
        downstream_raw_shard_ref=raw_shard,
        downstream_raw_shard_hash=raw_hash,
        receipt_root=receipt_root,
    )
    one_object = next((cache.root / "objects").rglob("*.bin"))
    one_object.unlink()
    tombstone = cache.finalize_eviction(
        str(sealed["receipt_ref"]), receipt_root=receipt_root
    )
    assert tombstone["state"] == "EVICTED"


def test_persistent_node_cache_retention_rejects_hash_path_and_unknown_drift(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    object_file = next((cache.root / "objects").rglob("*.bin"))
    object_file.write_bytes(object_file.read_bytes() + b"drift")
    with pytest.raises(ValueError, match="OBJECT.*HASH|HASH.*MISMATCH|SIZE_MISMATCH"):
        cache.seal(
            scope="formal",
            unit_id="retention-unit",
            plan_hash=_hash("c"),
            run_config_hash=_hash("d"),
            downstream_raw_shard_ref=raw_shard,
            downstream_raw_shard_hash=raw_hash,
            receipt_root=receipt_root,
        )

    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path / "unknown")
    (cache.root / "objects" / "unknown.bin").write_bytes(b"do not delete")
    with pytest.raises(ValueError, match="UNKNOWN_OBJECT"):
        cache.seal(
            scope="formal",
            unit_id="retention-unit",
            plan_hash=_hash("c"),
            run_config_hash=_hash("d"),
            downstream_raw_shard_ref=raw_shard,
            downstream_raw_shard_hash=raw_hash,
            receipt_root=receipt_root,
        )


def test_persistent_node_cache_retention_rejects_wrong_downstream_or_receipt_root(
    tmp_path: Path,
) -> None:
    cache, receipt_root, raw_shard, raw_hash = _retention_fixture(tmp_path)
    with pytest.raises(ValueError, match="DOWNSTREAM_RAW_SHARD_HASH_MISMATCH"):
        cache.seal(
            scope="formal",
            unit_id="retention-unit",
            plan_hash=_hash("c"),
            run_config_hash=_hash("d"),
            downstream_raw_shard_ref=raw_shard,
            downstream_raw_shard_hash=_hash("e"),
            receipt_root=receipt_root,
        )
    with pytest.raises(ValueError, match="OUTSIDE_CACHE_ROOT"):
        cache.seal(
            scope="formal",
            unit_id="retention-unit",
            plan_hash=_hash("c"),
            run_config_hash=_hash("d"),
            downstream_raw_shard_ref=raw_shard,
            downstream_raw_shard_hash=raw_hash,
            receipt_root=cache.root / "receipts-inside-cache",
        )


def test_reference_refinement_recovers_levels_and_cross_confirms_families(
    tmp_path: Path,
) -> None:
    levels = (
        ReferenceRuleLevel("gauss_legendre", 0, gauss_legendre_rule(2)),
        ReferenceRuleLevel("gauss_legendre", 1, gauss_legendre_rule(3)),
        ReferenceRuleLevel("gauss_legendre", 2, gauss_legendre_rule(4)),
        ReferenceRuleLevel("composite_simpson", 0, composite_simpson_rule(2)),
        ReferenceRuleLevel("composite_simpson", 1, composite_simpson_rule(4)),
        ReferenceRuleLevel("composite_simpson", 2, composite_simpson_rule(6)),
    )
    calls: list[str] = []

    def evaluate(rule: object) -> dict[str, np.ndarray]:
        calls.append(str(getattr(rule, "name")))
        return {"x": np.array([1.0, -2.0], dtype=np.float64)}

    root = tmp_path / "reference-refinement"
    interrupted = ReferenceRefinementRunner().run(
        unit_id="path-unit-fixture",
        levels=levels,
        evaluator=evaluate,
        artifact_root=root,
        tolerance=1e-12,
        required_consecutive=1,
        max_new_evaluations=2,
    )
    assert interrupted.converged is False
    assert len(calls) == 2

    resumed = ReferenceRefinementRunner().run(
        unit_id="path-unit-fixture",
        levels=levels,
        evaluator=evaluate,
        artifact_root=root,
        tolerance=1e-12,
        required_consecutive=1,
    )
    assert resumed.converged
    assert resumed.conservative_error == pytest.approx(0.0)
    assert resumed.status == "FIXTURE_CONVERGED"
    # 第一轮两次求值已从权威 commit 恢复，第二轮即满足停止规则。
    assert len(calls) == 4
    assert resumed.to_dict()["formal_eligible"] is False
    assert validate_stage23_artifact(resumed.to_dict()).artifact_hash == (
        resumed.artifact_hash
    )


def test_reference_refinement_zero_norm_is_unresolved_not_infinite(
    tmp_path: Path,
) -> None:
    levels = (
        ReferenceRuleLevel("gauss_legendre", 0, gauss_legendre_rule(2)),
        ReferenceRuleLevel("gauss_legendre", 1, gauss_legendre_rule(3)),
        ReferenceRuleLevel("composite_simpson", 0, composite_simpson_rule(2)),
        ReferenceRuleLevel("composite_simpson", 1, composite_simpson_rule(4)),
    )
    first_level_hashes = {
        levels[0].rule.artifact_hash,
        levels[2].rule.artifact_hash,
    }

    def evaluate(rule: object) -> dict[str, np.ndarray]:
        value = 1.0 if getattr(rule, "artifact_hash") in first_level_hashes else 0.0
        return {"x": np.array([value], dtype=np.float64)}

    result = ReferenceRefinementRunner().run(
        unit_id="zero-path-unit",
        levels=levels,
        evaluator=evaluate,
        artifact_root=tmp_path / "zero-refinement",
        tolerance=1e-6,
        required_consecutive=1,
    )
    assert result.converged is False
    assert result.convergence_defined is False
    assert result.status == "REFERENCE_UNRESOLVED"
    assert result.conservative_error is None
    assert any("zero_reference_l1_norm" in reason for reason in result.reasons)
    validate_stage23_artifact(result.to_dict())


def test_quadrature_recommendation_uses_full_unit_intersection_and_stays_fixture() -> None:
    observations = [
        QuadratureObservation(unit, rule, nodes, 0.01, 0.001, 0.99, 0.95, seconds)
        for unit in ("unit-a", "unit-b")
        for rule, nodes, seconds in (
            ("trapezoid", 2, 0.1),
            ("simpson", 3, 0.08),
        )
    ]
    recommendation = QuadratureRecommendationEngine().recommend(
        recommendation_id="fixture-quadrature-recommendation",
        observations=observations,
        required_unit_ids=("unit-a", "unit-b"),
        thresholds=QuadratureThresholds(0.05, 0.01, 0.9, 0.9, 8),
    )
    assert recommendation.default_rule == "trapezoid"
    assert recommendation.fallback_rule == "simpson"
    assert recommendation.status == "FIXTURE_RECOMMENDATION"
    assert recommendation.to_dict()["formal_eligible"] is False
    validated = validate_stage23_artifact(recommendation.to_dict())
    assert validated.artifact_hash == recommendation.artifact_hash
    tampered = recommendation.to_dict()
    tampered["fallback_rule"] = None
    with pytest.raises(ValueError, match="artifact_hash"):
        validate_stage23_artifact(tampered)
    with pytest.raises(FormalRunRejected, match="FORMAL_RUN_INTENT_REQUIRED"):
        recommendation.qualify(
            execution=FormalExecutionEvidence("local_fixture"),
            gate=_blocked_gate(3),
        )


def test_stage23_artifact_validator_rejects_fixture_formal_promotion() -> None:
    captured = EndpointCaptureCoordinator().capture(
        _capture_request(), _EndpointAdapter()
    )
    forged = captured.to_dict()
    forged["formal_eligible"] = True
    forged["qualification_gate_hash"] = _hash("f")
    payload = {name: value for name, value in forged.items() if name != "artifact_hash"}
    forged["artifact_hash"] = canonical_json_hash(payload)
    with pytest.raises(FormalRunRejected, match="QUALIFICATION_INCOMPLETE"):
        validate_stage23_artifact(forged)
