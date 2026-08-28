from __future__ import annotations

import math

import numpy as np
import pytest

from param_importance_nlp.experiments.stage3_metrics import (
    DEFAULT_TOP_Q,
    UnitMetricObservation,
    active_set_spearman,
    aggregate_by_group,
    aggregate_group_metrics,
    aggregate_layer_module_metrics,
    cosine,
    normalized_l1,
    normalized_l2,
    normalized_linf,
    quality_total_variation,
    sign_consistency,
    summarize_unit_observations,
    top_q_metrics,
)


def test_normalized_vector_metrics_have_no_epsilon_denominator() -> None:
    candidate = [2.0, 1.0, -1.0]
    reference = [1.0, 1.0, 1.0]
    assert normalized_l1(candidate, reference).value == pytest.approx(1.0)
    assert normalized_l2(candidate, reference).value == pytest.approx(math.sqrt(5.0) / math.sqrt(3.0))
    assert normalized_linf(candidate, reference).value == pytest.approx(2.0)
    assert cosine([1.0, 0.0], [1.0, 0.0]).value == pytest.approx(1.0)
    assert not cosine([0.0, 0.0], [1.0, 2.0]).defined

    zero_reference = normalized_l1([1.0, 0.0], [0.0, 0.0])
    assert not zero_reference.defined
    assert zero_reference.reason == "zero_reference_l1_norm"


def test_active_set_spearman_and_sign_consistency_use_reference_active_set() -> None:
    candidate = [3.0, -2.0, 0.0, 100.0]
    reference = [1.0, -1.0, 0.0, 1.0e-8]
    spearman = active_set_spearman(candidate, reference, active_threshold=1.0e-6)
    assert spearman.defined
    assert spearman.value == pytest.approx(1.0)
    assert spearman.details["active_count"] == 2
    assert sign_consistency(candidate, reference, active_threshold=1.0e-6).value == 1.0

    undefined = active_set_spearman([1.0], [1.0], active_threshold=0.0)
    assert not undefined.defined
    assert undefined.reason == "fewer_than_two_active_coordinates"


def test_top_q_metrics_use_ceiling_and_canonical_tie_break() -> None:
    candidate = [10.0, 8.0, 7.0, 1.0]
    reference = [10.0, 7.0, 8.0, 1.0]
    rows = top_q_metrics(
        candidate,
        reference,
        q_values=DEFAULT_TOP_Q,
        coordinate_ids=["a", "b", "c", "d"],
    )
    assert tuple(rows) == DEFAULT_TOP_Q
    assert rows[0.001]["k"] == 1
    assert rows[0.001]["overlap"].value == 1.0
    assert rows[0.01]["jaccard"].value == 1.0

    tied = top_q_metrics([2.0, 1.0, 1.0], [2.0, 1.0, 1.0], q_values=[0.5])
    assert not tied[0.5]["overlap"].defined
    assert "non_unique" in (tied[0.5]["overlap"].reason or "")


def test_group_aggregation_reports_totals_means_fractions_and_tv() -> None:
    candidate = [2.0, -1.0, 3.0, 0.0]
    reference = [1.0, -1.0, 1.0, 1.0]
    labels = ["layer-a", "layer-a", "layer-b", "layer-b"]
    one = aggregate_by_group(candidate, labels, quality_view="absolute")
    assert one["layer-a"]["count"] == 2
    assert one["layer-a"]["total"] == pytest.approx(1.0)
    assert one["layer-a"]["mean"] == pytest.approx(0.5)
    assert one["layer-a"]["quality_total"] == pytest.approx(3.0)
    assert one["layer-a"]["quality_fraction"] == pytest.approx(3.0 / 6.0)

    result = aggregate_group_metrics(candidate, reference, labels)
    assert result["groups"]["layer-b"]["reference_total"] == pytest.approx(2.0)
    tv = result["quality_total_variation"]
    assert tv.defined
    assert 0.0 <= (tv.value or 0.0) <= 1.0
    assert quality_total_variation(candidate, reference, labels).value == tv.value

    zero = quality_total_variation([0.0, 0.0], [0.0, 0.0], ["a", "b"])
    assert not zero.defined
    assert zero.reason == "zero_total_quality"


def test_layer_and_module_aggregation_keeps_views_separate() -> None:
    result = aggregate_layer_module_metrics(
        [2.0, -1.0, 3.0, 0.0],
        [1.0, -1.0, 1.0, 1.0],
        layer_groups=["l1", "l1", "l2", "l2"],
        module_groups=["m1", "m2", "m1", "m2"],
        quality_view="positive",
    )
    assert set(result) == {"layer", "module"}
    assert result["layer"]["groups"]["l1"]["candidate_quality_total"] == pytest.approx(2.0)
    assert result["module"]["groups"]["m2"]["candidate_quality_total"] == pytest.approx(0.0)


def test_group_index_mapping_must_cover_each_coordinate_once() -> None:
    result = aggregate_by_group(
        np.asarray([1.0, 2.0, 3.0]),
        {"first": [0, 1], "second": [2]},
    )
    assert result["first"]["total"] == pytest.approx(3.0)
    with pytest.raises(ValueError, match="覆盖全部坐标"):
        aggregate_by_group([1.0, 2.0], {"only": [0]})
    with pytest.raises(ValueError, match="只能属于一个"):
        aggregate_by_group([1.0, 2.0], {"a": [0], "b": [0, 1]})


def test_unit_summary_preserves_raw_units_and_uses_linear_p95() -> None:
    observations = [
        {"unit_id": "u3", "model": "14M", "phase": "early", "value": 3.0},
        {"unit_id": "u1", "model": "14M", "phase": "early", "value": 1.0},
        {"unit_id": "u2", "model": "14M", "phase": "early", "value": 2.0},
        {"unit_id": "u4", "model": "31M", "phase": "late", "value": 8.0},
    ]
    summary = summarize_unit_observations(observations, strata_keys=["model", "phase"])
    assert summary["unit_count"] == 4
    assert summary["strata_keys"] == ["model", "phase"]
    early = summary["groups"][0]
    assert early["strata"] == {"model": "14M", "phase": "early"}
    assert early["unit_ids"] == ["u1", "u2", "u3"]
    assert early["median"] == pytest.approx(2.0)
    assert early["p95"] == pytest.approx(2.9)
    assert early["worst"] == pytest.approx(3.0)
    assert summary["raw_observations"][0]["unit_id"] == "u1"

    dataclass_summary = summarize_unit_observations(
        [
            UnitMetricObservation("u1", 1.0, {"model": "14M"}),
            UnitMetricObservation("u2", 2.0, {"model": "14M"}),
        ]
    )
    assert dataclass_summary["groups"][0]["p95"] == pytest.approx(1.95)


def test_unit_summary_rejects_pseudoreplication_and_nonfinite_values() -> None:
    duplicate = [
        {"unit_id": "u1", "phase": "early", "value": 1.0},
        {"unit_id": "u1", "phase": "early", "value": 2.0},
    ]
    with pytest.raises(ValueError, match="重复观测"):
        summarize_unit_observations(duplicate)
    with pytest.raises(ValueError, match="NaN/Inf"):
        summarize_unit_observations([{"unit_id": "u1", "value": float("nan")}])
