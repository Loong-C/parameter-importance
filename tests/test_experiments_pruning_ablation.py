from __future__ import annotations

import pytest

from param_importance_nlp.experiments import (
    AblationFactor,
    AblationMatrix,
    ImportanceSourceSpec,
    PruningStudySpec,
)


def _hash(character: str) -> str:
    return character * 64


def test_pruning_study_compiles_high_low_random_global_and_balanced_units() -> None:
    sources = (
        ImportanceSourceSpec("magnitude", "magnitude-1", _hash("a"), _hash("c")),
        ImportanceSourceSpec("u", "u-1", _hash("b"), _hash("c")),
    )
    study = PruningStudySpec(
        "pruning-fixture",
        sources,
        (0.1, 0.5),
        random_mask_seeds=(3, 7),
    )
    runs = study.compile()
    # 2 sources × 2 directions × 2 scopes × 2 ratios +
    # 2 scopes × 2 ratios × 2 random seeds。
    assert len(runs) == 24
    assert len({run.run_id for run in runs}) == len(runs)
    assert all(run.tie_breaker == "canonical_coordinate_id" for run in runs)
    assert any(run.direction == "random" and run.mask_seed == 7 for run in runs)


def test_pruning_study_rejects_missing_or_registry_mismatched_sources() -> None:
    with pytest.raises(ValueError, match="不可用"):
        PruningStudySpec(
            "bad",
            (ImportanceSourceSpec("u", "u", _hash("a"), _hash("b"), available=False),),
            (0.1,),
        )
    with pytest.raises(ValueError, match="registry"):
        PruningStudySpec(
            "bad-registry",
            (
                ImportanceSourceSpec("u", "u", _hash("a"), _hash("b")),
                ImportanceSourceSpec("raw", "raw", _hash("c"), _hash("d")),
            ),
            (0.1,),
        )


def test_ablation_matrix_changes_exactly_one_leaf_and_is_deterministic() -> None:
    base = {
        "importance": {
            "estimator": "u",
            "keep_signed": True,
            "nested": {"value": 1},
        },
        "batching": {"microbatches": 16},
    }
    factors = (
        AblationFactor("estimator", ("importance", "estimator"), "u", ("raw", "double")),
        AblationFactor("microbatches", ("batching", "microbatches"), 16, (8, 32)),
    )
    first = AblationMatrix.compile(
        matrix_id="matrix-1",
        base_config=base,
        factors=factors,
        base_seed=123,
    )
    second = AblationMatrix.compile(
        matrix_id="matrix-1",
        base_config=base,
        factors=factors,
        base_seed=123,
    )
    assert len(first.cells) == 5
    assert first.digest == second.digest
    assert len({cell.seed for cell in first.cells}) == len(first.cells)
    assert all(
        cell.parent_cell_id == first.baseline_cell_id
        for cell in first.cells
        if cell.cell_id != first.baseline_cell_id
    )
    with pytest.raises(TypeError):
        first.cells[0].config["importance"]["nested"]["value"] = 999  # type: ignore[index]


def test_ablation_factor_baseline_must_match_base_config() -> None:
    with pytest.raises(ValueError, match="不一致"):
        AblationMatrix.compile(
            matrix_id="bad",
            base_config={"x": {"y": 1}},
            factors=(AblationFactor("y", ("x", "y"), 2, (3,)),),
            base_seed=0,
        )
