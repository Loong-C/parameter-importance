"""Stage 9 跨阶段 ETL、统计、图表和 AnalysisBundle 回归测试。"""

from __future__ import annotations

import hashlib

import pytest

from param_importance_nlp.analysis.pipeline import (
    AnalysisBundle,
    AnalysisBundleBuilder,
    BoundSourceTable,
    CrossStageSourceBuilder,
    HeatmapArtifact,
    HeatmapSpec,
    StageArtifactRows,
    TableArtifact,
    TableSpec,
    grouped_statistics,
    paired_statistics_with_holm,
    render_heatmap,
    render_table,
)


_SOURCE_HASH = "1" * 64


def _stage_rows(rows: list[dict[str, object]]) -> StageArtifactRows:
    return StageArtifactRows.create(
        artifact_id="fixture-evaluation-rows",
        source_schema_version="fixture-result-v1",
        source_artifact_hash=_SOURCE_HASH,
        stage=8,
        adapter_id="tests.fixture_adapter.v1",
        rows=rows,
        scope="local_fixture",
        formal_eligible=False,
        metadata={"statistical_unit": "seed"},
    )


def _paired_source():
    rows: list[dict[str, object]] = []
    for seed, baseline in enumerate((10.0, 11.0, 12.0, 13.0, 14.0)):
        rows.extend(
            (
                {
                    "task": "fixture-task",
                    "seed": seed,
                    "condition": "baseline",
                    "score": baseline,
                },
                {
                    "task": "fixture-task",
                    "seed": seed,
                    "condition": "candidate-a",
                    "score": baseline + float(seed + 1),
                },
                {
                    "task": "fixture-task",
                    "seed": seed,
                    "condition": "candidate-b",
                    "score": baseline,
                },
            )
        )
    builder = CrossStageSourceBuilder(scope="local_fixture")
    builder.add(_stage_rows(rows))
    return builder.build(name="paired-input", schema_version="analysis-input-v1")


def test_cross_stage_etl_grouped_statistics_and_holm_are_hash_bound() -> None:
    source = _paired_source()

    rows_wire = _stage_rows([{"value": 1.0}]).to_dict()
    assert StageArtifactRows.from_mapping(rows_wire).to_dict() == rows_wire
    assert BoundSourceTable.from_mapping(source.to_dict()).to_dict() == source.to_dict()
    assert source.role == "cross_stage_source"
    assert source.formal_eligible is False
    assert len(source.table.rows) == 15
    assert {row["source_stage"] for row in source.table.rows} == {8}
    assert {row["source_artifact_hash"] for row in source.table.rows} == {
        _SOURCE_HASH
    }

    grouped = grouped_statistics(
        source,
        group_columns=("task", "condition"),
        value_column="score",
    )
    assert grouped.parent_artifact_hashes == (source.artifact_hash,)
    assert len(grouped.table.rows) == 3
    by_condition = {row["condition"]: row for row in grouped.table.rows}
    assert by_condition["baseline"]["n"] == 5
    assert by_condition["baseline"]["mean"] == pytest.approx(12.0)
    assert by_condition["baseline"]["ci_defined"] is True

    paired = paired_statistics_with_holm(
        source,
        group_columns=("task",),
        pair_id_column="seed",
        condition_column="condition",
        value_column="score",
        baseline_condition="baseline",
    )
    assert paired.parent_artifact_hashes == (source.artifact_hash,)
    comparisons = {
        row["candidate_condition"]: row for row in paired.table.rows
    }
    assert comparisons["candidate-a"]["mean_difference"] == pytest.approx(3.0)
    assert comparisons["candidate-a"]["p_value_holm"] >= comparisons["candidate-a"][
        "p_value"
    ]
    assert comparisons["candidate-a"]["reject_h0_holm"] is True
    assert comparisons["candidate-b"]["p_value"] == pytest.approx(1.0)
    assert comparisons["candidate-b"]["p_value_holm"] == pytest.approx(1.0)
    assert comparisons["candidate-b"]["reject_h0_holm"] is False


def test_paired_statistics_rejects_missing_or_duplicate_pairs() -> None:
    missing = [
        row
        for row in (
            {"task": "t", "seed": 0, "condition": "baseline", "score": 1.0},
            {"task": "t", "seed": 1, "condition": "baseline", "score": 2.0},
            {"task": "t", "seed": 0, "condition": "candidate", "score": 1.5},
        )
    ]
    builder = CrossStageSourceBuilder(scope="local_fixture")
    builder.add(_stage_rows(missing))
    source = builder.build(name="missing-pair", schema_version="analysis-input-v1")
    with pytest.raises(ValueError, match="pair"):
        paired_statistics_with_holm(
            source,
            group_columns=("task",),
            pair_id_column="seed",
            condition_column="condition",
            value_column="score",
            baseline_condition="baseline",
        )


def test_rendered_tables_heatmap_and_bundle_are_deterministic(tmp_path) -> None:
    source = _paired_source()
    paired = paired_statistics_with_holm(
        source,
        group_columns=("task",),
        pair_id_column="seed",
        condition_column="condition",
        value_column="score",
        baseline_condition="baseline",
    )
    table_spec = TableSpec(
        table_artifact_hash=paired.artifact_hash,
        columns=(
            "task",
            "candidate_condition",
            "n_pairs",
            "mean_difference",
            "p_value_holm",
        ),
        caption="Paired fixture comparison",
    )
    rendered = render_table(paired, table_spec)
    assert TableSpec.from_mapping(table_spec.to_dict()) == table_spec
    assert TableArtifact.from_mapping(rendered.to_dict()).to_dict() == rendered.to_dict()
    assert rendered.contents["csv"].startswith("task,candidate_condition")
    assert "| candidate_condition |" in rendered.contents["markdown"]
    assert "\\begin{tabular}" in rendered.contents["latex"]
    output_paths = rendered.publish(tmp_path / "tables", stem="paired")
    assert all(path.is_file() for path in output_paths)

    heatmap_rows = _stage_rows(
        [
            {"checkpoint": 0, "layer": "a", "value": 1.0},
            {"checkpoint": 1, "layer": "a", "value": 2.0},
            {"checkpoint": 0, "layer": "b", "value": 3.0},
            {"checkpoint": 1, "layer": "b", "value": 4.0},
        ]
    )
    heatmap_builder = CrossStageSourceBuilder(scope="local_fixture")
    heatmap_builder.add(heatmap_rows)
    heatmap_source = heatmap_builder.build(
        name="heatmap-input", schema_version="analysis-input-v1"
    )
    heatmap_spec = HeatmapSpec(
        table_artifact_hash=heatmap_source.artifact_hash,
        source_content_hash=heatmap_source.table.content_hash,
        heatmap_id="fixture-trajectory",
        x_column="checkpoint",
        y_column="layer",
        value_column="value",
    )
    first_path = tmp_path / "heatmap-1.png"
    second_path = tmp_path / "heatmap-2.png"
    first = render_heatmap(heatmap_source, heatmap_spec, first_path)
    second = render_heatmap(heatmap_source, heatmap_spec, second_path)
    assert HeatmapSpec.from_mapping(heatmap_spec.to_dict()) == heatmap_spec
    assert HeatmapArtifact.from_mapping(first.to_dict()).to_dict() == first.to_dict()
    assert first.content_sha256 == second.content_sha256
    assert first.artifact_hash == second.artifact_hash
    assert first_path.read_bytes() == second_path.read_bytes()
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == first.content_sha256

    builder = AnalysisBundleBuilder(
        bundle_id="fixture-analysis", scope="local_fixture"
    )
    for table in (source, paired, heatmap_source):
        builder.add_table(table)
    builder.add_rendered_table(rendered)
    builder.add_heatmap(first)
    bundle = builder.build()
    bundle_again = builder.build()
    assert bundle.artifact_hash == bundle_again.artifact_hash
    assert AnalysisBundle.from_mapping(bundle.to_dict()).to_dict() == bundle.to_dict()
    published = bundle.publish(tmp_path / "bundle")
    assert published.is_file()
    assert (tmp_path / "bundle" / "rendered-tables" / "table-000.csv").is_file()


def test_formal_builders_are_fail_closed() -> None:
    with pytest.raises(ValueError, match="formal_authorization_hash"):
        CrossStageSourceBuilder(scope="formal")
    with pytest.raises(ValueError, match="formal_authorization_hash"):
        AnalysisBundleBuilder(bundle_id="formal", scope="formal")

    formal_builder = AnalysisBundleBuilder(
        bundle_id="formal",
        scope="formal",
        formal_authorization_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="scope"):
        formal_builder.add_table(_paired_source())
