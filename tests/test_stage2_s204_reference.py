from __future__ import annotations

from pathlib import Path

import numpy as np

from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.contracts.stage23 import validate_stage23_artifact
from param_importance_nlp.experiments.stage2_formal import (
    OneShotReferencePlan,
    OneShotReferenceRunner,
    ReferenceSizingPlan,
    StreamingReferenceSizer,
)
from param_importance_nlp.providers import SyntheticGradientProvider


def _provider() -> SyntheticGradientProvider:
    return SyntheticGradientProvider(
        {index: {"p": np.array([float(index), (-1.0) ** index])} for index in range(64)},
        statistical_unit="draw",
        weight_unit="draw_count",
        sampling_design="iid_with_replacement",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("s204-fixture", tuple(range(64))),
        {
            "reference_sizing": 101,
            "reference_A": 102,
            "reference_B": 103,
            "pilot": 104,
            "confirmatory": 105,
        },
    )


def test_sizing_consumes_independent_stream_then_final_is_one_shot(tmp_path: Path) -> None:
    sampling = _sampling()
    sizing_draws = sampling.draws("reference_sizing", 8)
    sizing = StreamingReferenceSizer(_provider()).run(
        ReferenceSizingPlan("s204-sizing", (4, 8), 2, 1e6, 1),
        draws_a=(),
        draws_b=(),
        draws_sizing=sizing_draws,
        artifact_root=tmp_path / "sizing",
    )
    assert sizing.converged
    assert sizing.selected_sample_count_per_stream == 8

    plan = OneShotReferencePlan("s204-final", sizing.artifact_hash, 8, 2)
    final = OneShotReferenceRunner(_provider()).run(
        plan,
        draws_a=sampling.draws("reference_A", 8),
        draws_b=sampling.draws("reference_B", 8),
        sizing_draws=sizing_draws,
        artifact_root=tmp_path / "final",
    )
    assert final.status == "COMPLETE"
    assert final.one_shot
    assert final.processed_sample_count_per_stream == 8
    assert final.uncertainty.estimator == "block_u_delete_one_jackknife"
    assert final.uncertainty.block_count_a == 4
    assert final.uncertainty.block_count_b == 4
    validate_stage23_artifact(final.to_dict())
    validate_stage23_artifact(final.uncertainty.to_dict())


def test_one_shot_resume_reuses_same_draw_identity(tmp_path: Path) -> None:
    sampling = _sampling()
    sizing = StreamingReferenceSizer(_provider()).run(
        ReferenceSizingPlan("s204-sizing", (4, 8), 2, 1e6, 1),
        draws_a=(),
        draws_b=(),
        draws_sizing=sampling.draws("reference_sizing", 8),
        artifact_root=tmp_path / "sizing",
    )
    plan = OneShotReferencePlan("s204-final", sizing.artifact_hash, 8, 2)
    runner = OneShotReferenceRunner(_provider())
    first = runner.run(
        plan,
        draws_a=sampling.draws("reference_A", 8),
        draws_b=sampling.draws("reference_B", 8),
        artifact_root=tmp_path / "final",
        max_new_block_pairs=2,
    )
    assert first.status == "IN_PROGRESS"
    resumed = runner.run(
        plan,
        draws_a=sampling.draws("reference_A", 8),
        draws_b=sampling.draws("reference_B", 8),
        artifact_root=tmp_path / "final",
    )
    assert resumed.status == "COMPLETE"
    assert resumed.to_dict()["one_shot"] is True
