from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.analysis import (
    AnalysisReportBuilder,
    ChartArtifact,
    ChartSpec,
    FrozenSourceTable,
    MetricResult,
    damage_auc,
    effective_parameter_count,
    entropy,
    error_summary,
    gini,
    hhi,
    mean_confidence_interval,
    pearson,
    spearman,
    top_k_jaccard,
    top_k_overlap,
    top_q_mass,
    render_matplotlib_chart,
)


def test_bias_variance_mse_mae_follow_repetition_coordinate_definition() -> None:
    estimates = np.array([[1.0, 2.0], [3.0, 4.0]])
    reference = np.array([2.0, 2.0])
    summary = error_summary(estimates, reference)
    assert summary["bias"].value == pytest.approx(0.5)
    assert summary["variance"].value == pytest.approx(1.0)
    assert summary["mean_squared_bias"].value == pytest.approx(0.5)
    assert summary["mse"].value == pytest.approx(1.5)
    assert summary["mae"].value == pytest.approx(1.0)


def test_correlations_use_average_ranks_and_report_degenerate_vectors() -> None:
    assert pearson([1, 2, 3], [2, 4, 6]).value == pytest.approx(1.0)
    assert spearman([1, 1, 3, 4], [2, 2, 8, 9]).value == pytest.approx(1.0)
    constant = pearson([1, 1], [2, 3])
    assert constant.defined is False
    assert constant.reason == "CONSTANT_VECTOR"
    assert spearman([1], [1]).defined is False


def test_top_k_ties_are_undefined_without_canonical_ids_but_deterministic_with_them() -> None:
    left = [3.0, 2.0, 2.0, 0.0]
    right = [3.0, 2.0, 2.0, 0.0]
    undefined = top_k_overlap(left, right, 2)
    assert undefined.defined is False
    assert undefined.reason == "NON_UNIQUE_TOP_K_BOUNDARY"

    identifiers = ("a", "b", "c", "d")
    assert top_k_overlap(left, right, 2, canonical_ids=identifiers).value == 1.0
    assert top_k_jaccard(left, right, 2, canonical_ids=identifiers).value == 1.0


def test_concentration_metrics_refuse_zero_or_negative_mass() -> None:
    for metric in (gini, entropy, hhi, effective_parameter_count):
        result = metric([0.0, 0.0])
        assert result.defined is False
        assert result.reason == "ZERO_TOTAL_MASS"
    assert top_q_mass([-1.0, 2.0], 0.5).reason == "NEGATIVE_MASS"

    uniform = np.ones(4)
    assert gini(uniform).value == pytest.approx(0.0)
    assert hhi(uniform).value == pytest.approx(0.25)
    assert effective_parameter_count(uniform).value == pytest.approx(4.0)
    assert top_q_mass(uniform, 0.25).value == pytest.approx(0.25)

    # 非均匀质量 [3, 1] 归一化为 [3/4, 1/4]，HHI=10/16，故 N_eff=8/5。
    # 该值与 exp(entropy) 不同，专门防止两种“有效数量”定义再次混用。
    concentrated = [3.0, 1.0]
    assert hhi(concentrated).value == pytest.approx(0.625)
    assert effective_parameter_count(concentrated).value == pytest.approx(1.6)


def test_confidence_interval_and_damage_auc_validate_independent_axis() -> None:
    mean, lower, upper = mean_confidence_interval([1.0, 2.0, 3.0])
    assert mean.value == pytest.approx(2.0)
    # n=3, df=2: t_0.975=4.3026527，标准误为 1/sqrt(3)。
    assert lower.value == pytest.approx(-0.4841377117)
    assert upper.value == pytest.approx(4.4841377117)
    assert mean.metadata["method"] == "student_t"
    assert mean.metadata["degrees_of_freedom"] == 2

    pair_mean, pair_lower, pair_upper = mean_confidence_interval([1.0, 3.0])
    # n=2, df=1: 标准误为 1，t_0.975=12.7062047；正态近似无法通过此回归。
    assert pair_mean.value == pytest.approx(2.0)
    assert pair_lower.value == pytest.approx(-10.7062047364)
    assert pair_upper.value == pytest.approx(14.7062047364)
    assert pair_mean.metadata["degrees_of_freedom"] == 1

    assert damage_auc([0.0, 0.5, 1.0], [0.0, 1.0, 2.0]).value == pytest.approx(1.0)
    assert damage_auc([0.0, 0.0], [0.0, 1.0]).defined is False


def test_analysis_report_is_deterministic_and_hash_bound_to_frozen_sources() -> None:
    source = FrozenSourceTable.from_rows(
        name="estimator_summary",
        schema_version="summary-v1",
        rows=(
            {"method": "raw", "mse": 2.0, "metadata": {"source": "fixture"}},
            {"method": "u", "mse": 1.0},
        ),
    )

    def build():
        builder = AnalysisReportBuilder(report_id="report-fixture")
        builder.add_source(source)
        builder.add_metric(
            "pearson",
            MetricResult(True, 0.9),
            source=source,
            derivation_id="fixture.pearson.v1",
            input_columns=("mse",),
        )
        builder.add_metric(
            "constant",
            MetricResult(False, None, "CONSTANT_VECTOR"),
            source=source,
            derivation_id="fixture.constant_check.v1",
            input_columns=("mse",),
        )
        return builder.build(metadata={"scope": "local_fixture"})

    first = build()
    second = build()
    assert first.report_hash == second.report_hash
    assert first.to_dict() == second.to_dict()
    assert "CONSTANT_VECTOR" in first.render_markdown()
    binding = first.metrics["pearson"].metadata["source_binding"]
    assert binding["source_hash"] == source.content_hash
    assert binding["derivation_id"] == "fixture.pearson.v1"
    with pytest.raises(TypeError):
        source.rows[0]["metadata"]["source"] = "edited"  # type: ignore[index]

    with pytest.raises(ValueError, match="content_hash"):
        FrozenSourceTable(
            source.name,
            source.schema_version,
            source.rows,
            "0" * 64,
            True,
        )


def test_report_builder_rejects_unbound_or_forged_metric_sources() -> None:
    source = FrozenSourceTable.from_rows(
        name="source",
        schema_version="source-v1",
        rows=({"value": 1.0}, {"value": 2.0}),
    )
    builder = AnalysisReportBuilder(report_id="bound-report")
    builder.add_source(source)
    with pytest.raises(TypeError):
        builder.add_metric("manual", MetricResult(True, 1.0))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="不完整的列"):
        builder.add_metric(
            "missing-column",
            MetricResult(True, 1.0),
            source=source,
            derivation_id="fixture.identity.v1",
            input_columns=("not_present",),
        )
    forged = FrozenSourceTable.from_rows(
        name="source",
        schema_version="source-v1",
        rows=({"value": 99.0},),
    )
    with pytest.raises(ValueError, match="hash 与登记值冲突"):
        builder.add_metric(
            "forged",
            MetricResult(True, 99.0),
            source=forged,
            derivation_id="fixture.identity.v1",
            input_columns=("value",),
        )


def test_report_from_mapping_requires_strict_source_binding() -> None:
    source = FrozenSourceTable.from_rows(
        name="source",
        schema_version="source-v1",
        rows=({"value": 1.0},),
    )
    builder = AnalysisReportBuilder(report_id="strict-report")
    builder.add_source(source)
    builder.add_metric(
        "value",
        MetricResult(True, 1.0),
        source=source,
        derivation_id="fixture.identity.v1",
        input_columns=("value",),
    )
    wire = builder.build().to_dict()
    assert type(builder.build()).from_mapping(wire).report_hash == wire["report_hash"]

    missing = copy.deepcopy(wire)
    del missing["metrics"]["value"]["metadata"]["source_binding"]
    with pytest.raises(ValueError, match="source_binding"):
        type(builder.build()).from_mapping(missing)
    extra = copy.deepcopy(wire)
    extra["metrics"]["value"]["metadata"]["source_binding"]["unknown"] = True
    with pytest.raises(ValueError, match="FIELDS_MISMATCH"):
        type(builder.build()).from_mapping(extra)


def _chart_source() -> FrozenSourceTable:
    return FrozenSourceTable.from_rows(
        name="chart-source",
        schema_version="chart-source-v1",
        rows=(
            {"coordinate": "c", "score": 3.0, "split": "eval", "rank": 3},
            {"coordinate": "a", "score": 1.0, "split": "eval", "rank": 1},
            {"coordinate": "b", "score": 2.0, "split": "train", "rank": 2},
        ),
    )


def test_chart_spec_is_canonical_rebuildable_and_source_bound() -> None:
    source = _chart_source()
    spec = ChartSpec.from_table(
        source,
        chart_id="importance-by-coordinate",
        chart_type="line",
        x_column="coordinate",
        y_columns=("score",),
        filters=({"column": "split", "operator": "eq", "value": "eval"},),
        sort_columns=("rank",),
    )
    rows = spec.materialize(source)
    assert [row["coordinate"] for row in rows] == ["a", "c"]
    assert ChartSpec.from_mapping(spec.to_dict()) == spec
    assert spec.to_dict()["source_hash"] == source.content_hash
    assert spec.to_dict()["chart_type"] == "line"
    assert spec.to_dict()["sort_columns"] == ["rank"]
    assert spec.to_dict()["filters"][0]["column"] == "split"

    reversed_spec = ChartSpec.from_table(
        source,
        chart_id="importance-by-coordinate",
        chart_type="line",
        x_column="coordinate",
        y_columns=("score",),
        filters=({"column": "split", "operator": "eq", "value": "eval"},),
        sort_columns=("rank",),
        sort_descending=True,
    )
    assert reversed_spec.spec_hash != spec.spec_hash
    assert [row["coordinate"] for row in reversed_spec.materialize(source)] == [
        "c",
        "a",
    ]
    changed_source = FrozenSourceTable.from_rows(
        name=source.name,
        schema_version=source.schema_version,
        rows=({"coordinate": "a", "score": 99.0, "split": "eval", "rank": 1},),
    )
    with pytest.raises(ValueError, match="CHART_SOURCE_IDENTITY_MISMATCH"):
        spec.materialize(changed_source)
    with pytest.raises(TypeError, match="FROZEN_SOURCE_TABLE"):
        ChartSpec.from_table(  # type: ignore[arg-type]
            {"rows": []},
            chart_id="invalid",
            chart_type="line",
            x_column="x",
            y_columns=("y",),
        )


def test_chart_artifact_hash_excludes_paths_and_strictly_round_trips() -> None:
    source = _chart_source()
    spec = ChartSpec.from_table(
        source,
        chart_id="scores",
        chart_type="bar",
        x_column="coordinate",
        y_columns=("score",),
        sort_columns=("rank",),
    )
    artifact = ChartArtifact.from_spec(spec)
    assert ChartArtifact.from_mapping(artifact.to_dict()) == artifact
    assert artifact.content_sha256 is None
    assert "path" not in str(artifact.to_dict()).casefold()

    with pytest.raises(ValueError, match="ABSOLUTE_PATH"):
        ChartArtifact.from_rendered_bytes(
            spec,
            b"png",
            renderer_id="fixture-renderer:v1",
            output_format="png",
            render_options={"font_path": "C:\\fonts\\fixture.ttf"},
        )
    invalid = artifact.to_dict()
    invalid["unknown"] = True
    with pytest.raises(ValueError, match="FIELDS_MISMATCH"):
        ChartArtifact.from_mapping(invalid)


def test_optional_matplotlib_renderer_is_path_independent(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    source = _chart_source()
    spec = ChartSpec.from_table(
        source,
        chart_id="scores",
        chart_type="bar",
        x_column="coordinate",
        y_columns=("score",),
        sort_columns=("rank",),
    )
    first_path = tmp_path / "first" / "chart.png"
    second_path = tmp_path / "unrelated-absolute-location" / "chart.png"
    first = render_matplotlib_chart(spec, source, first_path)
    second = render_matplotlib_chart(spec, source, second_path)
    assert first.artifact_hash == second.artifact_hash
    assert first.content_sha256 == second.content_sha256
    assert first_path.read_bytes() == second_path.read_bytes()
    assert str(tmp_path.resolve()) not in str(first.to_dict())
