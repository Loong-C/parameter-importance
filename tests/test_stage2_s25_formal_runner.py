"""Focused S2.5 paired-runner contract tests (CPU synthetic provider only)."""

from pathlib import Path

import numpy as np

from param_importance_nlp.contracts import validate_stage23_artifact
from param_importance_nlp.experiments.sampling import (
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
)
from param_importance_nlp.experiments.stage2_formal import (
    RecoverablePairedWaveRunner,
    _vector_digest,
)
from param_importance_nlp.providers import SyntheticGradientProvider


def _provider() -> SyntheticGradientProvider:
    return SyntheticGradientProvider(
        {
            index: {"p": np.array([(-1.0) ** index * (index + 1), -0.25 * index])}
            for index in range(8)
        },
        fixed_state_id="s25-synthetic-state",
        statistical_unit="synthetic_draw_group_mean",
        weight_unit="synthetic_draw_count",
        sampling_design="uniform_with_replacement_disjoint_draw_groups",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _mapping() -> RepetitionMapping:
    sampling = SamplingPlan(
        SamplingUniverse("s25-universe", tuple(range(8))),
        {"pilot": 101, "confirmatory": 102, "reference_sizing": 103, "reference_A": 104, "reference_B": 105},
    )
    return RepetitionMapping.create(
        repetition_id="s25-rep-0001",
        draws=sampling.draws("pilot", 8),
        m_values=(2, 4),
    )


def test_s25_runner_emits_three_reference_views_and_replay_evidence(tmp_path: Path) -> None:
    references = {
        "bias": {"p": np.array([0.5, -0.5])},
        "cross": {"p": np.array([0.25, -0.25])},
        "ranking": {"p": np.array([-0.75, 0.25])},
    }
    summary = RecoverablePairedWaveRunner(_provider()).run(
        wave_id="s25-focused-wave",
        mappings=(_mapping(),),
        reference=references["bias"],
        reference_hash=_vector_digest(references["bias"]),
        references=references,
        artifact_root=tmp_path / "wave",
    )
    assert summary.complete
    assert set(summary.reference_hashes) == {"bias", "cross", "ranking"}
    assert set(summary.reference_statistics) == {"bias", "cross", "ranking"}
    assert summary.reference_statistics["ranking"]["raw"]["top_k"] >= 1
    assert summary.microbatch_diagnostics[0]["token_count"] > 0
    assert "gradient_norm" in summary.microbatch_diagnostics[0]
    assert summary.replay_evidence["attempt_bound"] is True
    assert summary.replay_evidence["idempotent_reducer"] is True
    validate_stage23_artifact(summary.to_dict())

    resumed = RecoverablePairedWaveRunner(_provider()).run(
        wave_id="s25-focused-wave",
        mappings=(_mapping(),),
        reference=references["bias"],
        reference_hash=_vector_digest(references["bias"]),
        references=references,
        artifact_root=tmp_path / "wave",
    )
    assert resumed.resumed_unit_count == 1
    assert resumed.method_statistics == summary.method_statistics
    assert resumed.reference_statistics == summary.reference_statistics
    assert resumed.replay_evidence == summary.replay_evidence
