from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    CoreEstimatorKernel,
    DeterministicShardReducer,
    FormalDecisionBlocked,
    PairedEstimatorRunner,
    PilotObservation,
    ReferenceRunner,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    ShardArtifactStore,
    Stage2FixtureStudy,
    SufficientStatisticShard,
    build_fixture_estimator_decision,
    select_primary_pair,
)
from param_importance_nlp.providers import GradientBatch, SyntheticGradientProvider


def _plan(sample_ids: tuple[int, ...] = tuple(range(8))) -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("tiny-fixture", sample_ids),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )


def _provider() -> SyntheticGradientProvider:
    table = {
        sample_id: {
            "layer.weight": np.array([sample_id - 2.0, 0.5 * sample_id]),
            "layer.bias": np.array([(-1.0) ** sample_id]),
        }
        for sample_id in range(8)
    }
    return SyntheticGradientProvider(
        table,
        fixed_state_id="fixture-checkpoint",
        statistical_unit="synthetic_draw_group_mean",
        weight_unit="synthetic_draw_count",
        sampling_design="uniform_with_replacement_disjoint_draw_groups",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def test_gradient_batch_and_provider_expose_complete_weighting_contract() -> None:
    """统计假设必须随数值批次传输，runner 不得自行填写默认值。"""

    provider = _provider()
    batch = provider.gradient((0, 1))
    assert batch.weighting_assumptions == {
        "statistical_unit": provider.statistical_unit,
        "weight_unit": provider.weight_unit,
        "sampling_design": provider.sampling_design,
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    with pytest.raises(TypeError, match="weights_exogenous"):
        GradientBatch(
            gradients={"p": np.array([1.0])},
            statistical_weight=1.0,
            statistical_unit="draw_group_mean",
            weight_unit="draw_count",
            sampling_design="iid_with_replacement",
            weights_exogenous=1,  # type: ignore[arg-type]
            common_mean_assumption=True,
            sample_ids=(0,),
        )


def test_weighted_u_uses_provider_assumptions_and_rejects_false_claims() -> None:
    """实际非等权时，任一无偏性前提为假都必须 fail-closed。"""

    table = {
        0: {"p": np.array([1.0])},
        1: {"p": np.array([3.0])},
    }
    provider = SyntheticGradientProvider(
        table,
        statistical_weights={0: 1.0, 1: 2.0},
        statistical_unit="synthetic_draw_group_mean",
        weight_unit="importance_weight",
        sampling_design="iid_with_replacement_disjoint_draw_groups",
        weights_exogenous=False,
        common_mean_assumption=True,
    )
    batches = [provider.gradient((sample_id,)) for sample_id in (0, 1)]
    kernel = CoreEstimatorKernel(accumulation_dtype="float64")
    gradients = [kernel.tensor_map(batch) for batch in batches]
    weights = [batch.statistical_weight for batch in batches]
    with pytest.raises(ValueError, match="WEIGHTED_U_UNBIASEDNESS_ASSUMPTIONS"):
        kernel.u(gradients, weights, **batches[0].weighting_assumptions)

    qualified = SyntheticGradientProvider(
        table,
        statistical_weights={0: 1.0, 1: 2.0},
        statistical_unit="synthetic_draw_group_mean",
        weight_unit="importance_weight",
        sampling_design="iid_with_replacement_disjoint_draw_groups",
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    qualified_batches = [qualified.gradient((sample_id,)) for sample_id in (0, 1)]
    qualified_gradients = [kernel.tensor_map(batch) for batch in qualified_batches]
    result = kernel.u(
        qualified_gradients,
        [batch.statistical_weight for batch in qualified_batches],
        **qualified_batches[0].weighting_assumptions,
    )
    # 两个统计单元时，加权去对角 U 与 cross/double 恒等；权重不会改变乘积。
    assert float(result["p"].item()) == pytest.approx(3.0)


def test_synthetic_state_digest_binds_statistical_weights_and_assumptions() -> None:
    table = {0: {"p": np.array([1.0])}, 1: {"p": np.array([2.0])}}
    common = {
        "statistical_unit": "synthetic_draw_group_mean",
        "weight_unit": "draw_weight",
        "sampling_design": "iid_with_replacement_disjoint_draw_groups",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    equal = SyntheticGradientProvider(table, statistical_weights={0: 1.0, 1: 1.0}, **common)
    weighted = SyntheticGradientProvider(
        table, statistical_weights={0: 1.0, 1: 2.0}, **common
    )
    endogenous = SyntheticGradientProvider(
        table,
        statistical_weights={0: 1.0, 1: 1.0},
        **(common | {"weights_exogenous": False}),
    )
    assert len({equal.state_digest(), weighted.state_digest(), endogenous.state_digest()}) == 3


def test_five_stream_draws_replay_and_keep_sample_collisions() -> None:
    plan = _plan((7,))
    first = plan.draws("pilot", 32)
    replay = plan.draws("pilot", 32)
    other = plan.draws("confirmatory", 32)

    assert first == replay
    assert len({draw.draw_id for draw in first}) == 32
    assert {draw.sample_id for draw in first} == {7}
    assert {draw.draw_id for draw in first}.isdisjoint(draw.draw_id for draw in other)

    mapping = RepetitionMapping.create(
        repetition_id="rep-collision",
        draws=first,
        m_values=CANDIDATE_MICROBATCH_COUNTS,
    )
    assert mapping.sample_collision_count == 31
    assert tuple(draw for group in mapping.groups(4) for draw in group) == first
    assert len(mapping.double_halves[0]) == len(mapping.double_halves[1]) == 16


def test_sampling_plan_rejects_reused_seed_namespace() -> None:
    universe = SamplingUniverse("u", (0, 1))
    with pytest.raises(ValueError, match="互不相同"):
        SamplingPlan(universe, {name: 1 for name in (
            "reference_sizing", "reference_A", "reference_B", "pilot", "confirmatory"
        )})


def test_blinded_primary_pair_selection_uses_frozen_order() -> None:
    observations = []
    for batch_size in CANDIDATE_BATCH_SIZES:
        for m in (32, 16, 8, 4):
            observations.append(
                PilotObservation(
                    batch_size=batch_size,
                    microbatch_count=m,
                    anchors_runnable=True,
                    finite=True,
                    aggregation_overhead_ratio=0.30 if m == 32 else 0.10,
                    r_required=10 if batch_size == 32 else 20,
                    resource_within_budget=True,
                )
            )
    decision = select_primary_pair(observations, r_max=100)
    assert decision.status == "FIXTURE_SELECTED"
    assert (decision.batch_size, decision.microbatch_count) == (32, 16)
    assert decision.formal_eligible is False
    with pytest.raises(FormalDecisionBlocked):
        select_primary_pair(observations, r_max=100, scope="formal")


def test_paired_runner_reuses_one_base_gradient_pool_and_m2_equals_double() -> None:
    plan = _plan()
    mapping = RepetitionMapping.create(
        repetition_id="rep-000",
        draws=plan.draws("pilot", 32),
        m_values=CANDIDATE_MICROBATCH_COUNTS,
    )
    runner = PairedEstimatorRunner(_provider(), m2_tolerance=1e-12)
    result = runner.run(mapping)

    assert result.gradient_evaluations == 32
    assert result.m2_double_max_abs_error <= 1e-12
    assert tuple(result.u_by_m) == CANDIDATE_MICROBATCH_COUNTS
    assert result.formal_eligible is False
    assert result.weighting_assumptions["statistical_unit"] == (
        "synthetic_draw_group_mean"
    )
    assert result.weighting_assumptions["weights_exogenous"] is True
    assert len(result.digest) == 64


def test_reference_runner_is_one_shot_fixture_and_uses_three_named_views() -> None:
    plan = _plan()
    result = ReferenceRunner(_provider()).run(
        reference_id="reference-fixture",
        draws_a=plan.draws("reference_A", 16),
        draws_b=plan.draws("reference_B", 16),
        block_size=4,
    )
    assert result.sample_count_a == result.sample_count_b == 16
    assert result.metadata["one_shot"] is True
    assert result.metadata["early_stopping"] is False
    assumptions = result.metadata["weighting_assumptions"]
    assert assumptions["weight_unit"] == "synthetic_draw_count"
    assert assumptions["common_mean_assumption"] is True
    assert len(result.digest) == 64


def test_deterministic_reducer_is_order_invariant_and_rejects_new_attempt() -> None:
    plan = _plan()
    runner = PairedEstimatorRunner(_provider(), m2_tolerance=1e-12)
    results = [
        runner.run(
            RepetitionMapping.create(
                repetition_id=f"rep-{index}",
                draws=plan.draws("pilot", 32, start=index * 32),
                m_values=CANDIDATE_MICROBATCH_COUNTS,
            )
        )
        for index in range(2)
    ]
    shards = [
        SufficientStatisticShard.from_result(result, attempt_id="attempt-1")
        for result in results
    ]
    reducer_a = DeterministicShardReducer()
    reducer_b = DeterministicShardReducer()
    for shard in shards:
        assert reducer_a.add(shard)
    for shard in reversed(shards):
        assert reducer_b.add(shard)
    assert reducer_a.reduce().digest == reducer_b.reduce().digest
    assert reducer_a.add(shards[0]) is False

    conflicting = SufficientStatisticShard(
        unit_id=shards[0].unit_id,
        attempt_id="attempt-2",
        input_hash=shards[0].input_hash,
        vectors=shards[0].vectors,
    )
    with pytest.raises(ValueError, match="拒绝重复计数"):
        reducer_a.add(conflicting)


def test_persistent_reducer_resumes_repetitions_and_matches_uninterrupted(
    tmp_path: Path,
) -> None:
    """中断边界只落在权威 shard commit 之间，恢复后不得重复计数。"""

    plan = _plan()
    runner = PairedEstimatorRunner(_provider(), m2_tolerance=1e-12)
    shards = [
        SufficientStatisticShard.from_result(
            runner.run(
                RepetitionMapping.create(
                    repetition_id=f"resume-rep-{index}",
                    draws=plan.draws("confirmatory", 32, start=index * 32),
                    m_values=CANDIDATE_MICROBATCH_COUNTS,
                )
            ),
            attempt_id="attempt-resume",
        )
        for index in range(2)
    ]
    uninterrupted = DeterministicShardReducer()
    for shard in shards:
        uninterrupted.add(shard)
    expected = uninterrupted.reduce()

    artifact_root = tmp_path / "repetition-shards"
    first_process = DeterministicShardReducer(artifact_root)
    try:
        assert first_process.add(shards[0]) is True
        with pytest.raises(RuntimeError, match="SHARD_WRITER_ALREADY_ACTIVE"):
            DeterministicShardReducer(artifact_root)
    finally:
        first_process.close()

    with DeterministicShardReducer.resume(artifact_root) as resumed:
        assert resumed.persisted is True
        # 重放已经权威提交的 repetition 是幂等 no-op。
        assert resumed.add(shards[0]) is False
        assert resumed.add(shards[1]) is True
        observed = resumed.reduce()
        report = resumed.reconcile_artifacts()

    assert observed.digest == expected.digest
    assert observed.unit_ids == expected.unit_ids
    assert report["valid_unit_ids"] == sorted(shard.unit_id for shard in shards)
    assert report["invalid_commits"] == []
    assert report["orphan_objects"] == []


def test_persistent_reducer_rejects_corruption_and_releases_failed_resume_lock(
    tmp_path: Path,
) -> None:
    plan = _plan()
    result = PairedEstimatorRunner(_provider(), m2_tolerance=1e-12).run(
        RepetitionMapping.create(
            repetition_id="corrupt-rep",
            draws=plan.draws("pilot", 32),
            m_values=CANDIDATE_MICROBATCH_COUNTS,
        )
    )
    shard = SufficientStatisticShard.from_result(
        result,
        attempt_id="attempt-corrupt",
    )
    artifact_root = tmp_path / "corrupt-shards"
    with DeterministicShardReducer(artifact_root) as reducer:
        reducer.add(shard)

    tensor_path = next((artifact_root / "objects").glob("*/tensors/*.bin"))
    tensor_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="TENSOR_(SIZE|HASH)_MISMATCH"):
        DeterministicShardReducer.resume(artifact_root)

    # 构造器恢复失败也必须释放 OS advisory lock，诊断/修复进程仍能取得写者权。
    with ShardArtifactStore(artifact_root):
        pass


def test_fixture_estimator_decision_cannot_unlock_formal_training() -> None:
    observations = [
        PilotObservation(32, m, True, True, 0.1, 8, True)
        for m in (32, 16, 8, 4)
    ]
    pair = select_primary_pair(observations, r_max=16)
    decision = build_fixture_estimator_decision(pair, selected_estimator="u")
    assert decision.status == "FIXTURE_ONLY"
    assert decision.formal_eligible is False
    wire = decision.to_dict()
    artifact_hash = wire.pop("artifact_hash")
    assert canonical_json_hash(wire) == artifact_hash
    with pytest.raises(FormalDecisionBlocked):
        decision.require_formal()


def test_synthetic_stage2_state_machine_never_promotes_fixture_to_formal() -> None:
    plan = _plan()
    provider = _provider()
    reference = ReferenceRunner(provider).run(
        reference_id="state-machine-reference",
        draws_a=plan.draws("reference_A", 16),
        draws_b=plan.draws("reference_B", 16),
        block_size=4,
    )
    observations = [
        PilotObservation(32, m, True, True, 0.1, 8, True)
        for m in (32, 16, 8, 4)
    ]
    pair = select_primary_pair(observations, r_max=16)
    result = PairedEstimatorRunner(provider).run(
        RepetitionMapping.create(
            repetition_id="state-machine-repetition",
            draws=plan.draws("confirmatory", 32),
            m_values=CANDIDATE_MICROBATCH_COUNTS,
        )
    )
    study = Stage2FixtureStudy("fixture-study")
    study.register_reference(reference)
    study.freeze_matrix(pair)
    study.select_estimator()
    study.complete((result,))
    assert study.state == "FIXTURE_COMPLETE"
    assert study.decision is not None and study.decision.formal_eligible is False
    assert len(study.digest) == 64
