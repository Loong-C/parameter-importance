from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.analysis import AnalysisReportBuilder, ChartSpec, MetricResult
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage3_reporting import (
    build_raw_reporting_tables,
    write_reporting_bundle,
)
from param_importance_nlp.experiments.stage3_raw_storage import (
    derive_candidate_views,
    load_raw_aggregate,
    persist_raw_unit_shard,
    publish_raw_aggregate,
)
from param_importance_nlp.experiments import stage3_raw_storage
import torch


def _raw_path_results() -> dict[str, object]:
    rule = {"nodes": [0.0, 0.5, 1.0]}
    return {
        "schema_version": "stage3-task-path-results-v1",
        "scope": "formal",
        "independent_reference": {
            "signed": {"weight": [1.0, 2.0]},
        },
        "candidate_results": {
            "trapezoid": {
                "rule": rule,
                "signed": {"weight": [1.1, 1.9]},
                "positive": {"weight": [1.1, 1.9]},
                "negative_mass": {"weight": [0.0, 0.0]},
                "absolute": {"weight": [1.1, 1.9]},
                "node_losses": [3.0, 2.0, 1.0],
            },
        },
    }


def test_formal_raw_tables_require_vectors_and_path_losses() -> None:
    vector, curves = build_raw_reporting_tables(_raw_path_results())
    assert vector.content_hash
    assert curves.content_hash
    assert len(vector.rows) == 2
    assert len(curves.rows) == 3
    with pytest.raises(ValueError, match="PATH_CURVE_MISSING"):
        bad = _raw_path_results()
        bad["candidate_results"] = {"trapezoid": {"rule": {"nodes": [0.0]}, "signed": {"weight": [1.0, 2.0]}}}
        build_raw_reporting_tables(bad)


def test_reporting_bundle_writes_hash_bound_json_csv_png_and_svg(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    raw = _raw_path_results()
    vector, curves = build_raw_reporting_tables(raw)
    report_builder = AnalysisReportBuilder(report_id="stage3-reporting-test")
    report_builder.add_source(vector)
    report_builder.add_source(curves)
    report_builder.add_metric(
        "vector_mean",
        MetricResult(True, 1.5),
        source=vector,
        derivation_id="test.vector-mean.v1",
        input_columns=("signed",),
    )
    report = report_builder.build(metadata={"scope": "formal", "formal_eligible": True})
    specs = (
        ChartSpec.from_table(
            vector,
            chart_id="vector",
            chart_type="scatter",
            x_column="coordinate_index",
            y_columns=("signed",),
            sort_columns=("unit_id", "rule_name", "coordinate_id"),
        ),
        ChartSpec.from_table(
            curves,
            chart_id="path-loss",
            chart_type="line",
            x_column="alpha",
            y_columns=("loss",),
            sort_columns=("unit_id", "rule_name", "alpha"),
        ),
    )
    bundle = write_reporting_bundle(
        workspace_root=tmp_path,
        output_root=tmp_path / "outputs" / "s310",
        tables=(vector, curves),
        raw_formal_results=raw,
        report=report,
        chart_specs=specs,
    )
    assert len(bundle["tables"]) == 2
    assert Path(tmp_path / bundle["analysis_report"]["path"]).is_file()  # type: ignore[index]
    assert len(bundle["figures"]) == 2
    for figure in bundle["figures"]:  # type: ignore[union-attr]
        assert Path(tmp_path / figure["png"]["path"]).is_file()  # type: ignore[index]
        assert Path(tmp_path / figure["svg"]["path"]).is_file()  # type: ignore[index]


def test_formal_raw_aggregate_keeps_all_units_and_rejects_bundle_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    required = ("unit-1", "unit-2")
    execution_hash = "1" * 64
    binding_hash = "2" * 64
    ledger = tmp_path / "outputs" / "raw"
    for index, unit_id in enumerate(required, start=1):
        persist_raw_unit_shard(
            root=tmp_path,
            ledger_root=ledger,
            unit_id=unit_id,
            required_unit_ids=required,
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
            path_identity_hash=f"{index + 2}" * 64,
            reference_artifact_hash=f"{index + 3}" * 64,
            reference_signed={"weight": torch.tensor([1.0, 2.0], dtype=torch.float64)},
            candidate_states={
                "trapezoid": {
                    "signed": {"weight": torch.tensor([-1.1 - index / 10, 1.9], dtype=torch.float64)},
                    "positive": {"weight": torch.tensor([0.0, 1.9], dtype=torch.float64)},
                    "negative_mass": {"weight": torch.tensor([1.1 + index / 10, 0.0], dtype=torch.float64)},
                    "absolute": {"weight": torch.tensor([1.1 + index / 10, 1.9], dtype=torch.float64)},
                }
            },
            rule_summaries={
                "trapezoid": {
                    "rule": {"nodes": [0.0, 0.5, 1.0]},
                    "node_alphas": [0.0, 0.5, 1.0],
                    "node_losses": [3.0, 2.0, 1.0],
                    "path_identity_hash": f"{index + 2}" * 64,
                    "evaluation_cost": {"gradient_evaluations": 3},
                }
            },
        )
        partial, _partial_ref = publish_raw_aggregate(
            root=tmp_path,
            ledger_root=ledger,
            required_unit_ids=required,
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
            candidate_rule_names=("trapezoid",),
        )
        if index == 1:
            assert partial["complete_unit_ids"] == ["unit-1"]
            assert partial["missing_unit_ids"] == ["unit-2"]
    aggregate, aggregate_ref = publish_raw_aggregate(
        root=tmp_path,
        ledger_root=ledger,
        required_unit_ids=required,
        execution_evidence_hash=execution_hash,
        reference_binding_hash=binding_hash,
        candidate_rule_names=("trapezoid",),
    )
    raw = {
        "schema_version": "stage3-task-path-results-v1",
        "scope": "formal",
        "raw_aggregate_ref": aggregate_ref,
        "raw_aggregate_hash": aggregate["artifact_hash"],
    }
    def fail_full_loader(**_kwargs: object) -> object:
        raise AssertionError("reporting must use bounded raw aggregate iteration")

    monkeypatch.setattr(stage3_raw_storage, "load_raw_aggregate", fail_full_loader)
    vectors, curves = build_raw_reporting_tables(raw, workspace_root=tmp_path)
    assert len(vectors.rows) == 2
    assert {row["unit_id"] for row in vectors.rows} == set(required)
    assert {row["coordinate_count"] for row in vectors.rows} == {2}
    loaded_aggregate, loaded_units = load_raw_aggregate(
        root=tmp_path,
        aggregate_ref=aggregate_ref,
        aggregate_hash=aggregate["artifact_hash"],
    )
    assert loaded_aggregate["vector_derivation_schema"] == "stage3-formal-vector-view-derivation-v1"
    state = loaded_units["unit-1"][1]
    assert set(state["candidates"]["trapezoid"]) == {"signed"}  # type: ignore[index]
    views = derive_candidate_views(state["candidates"]["trapezoid"]["signed"])  # type: ignore[index]
    assert views["positive"]["weight"].tolist() == [0.0, 1.9]
    assert views["negative_mass"]["weight"].tolist() == pytest.approx([1.2, 0.0])
    with pytest.raises(ValueError, match="DERIVED_VIEW_MISMATCH"):
        persist_raw_unit_shard(
            root=tmp_path,
            ledger_root=tmp_path / "outputs" / "raw-bad",
            unit_id="unit-1",
            required_unit_ids=("unit-1",),
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
            path_identity_hash="9" * 64,
            reference_artifact_hash="a" * 64,
            reference_signed={"weight": torch.tensor([1.0], dtype=torch.float64)},
            candidate_states={"trapezoid": {
                "signed": {"weight": torch.tensor([-1.0], dtype=torch.float64)},
                "positive": {"weight": torch.tensor([1.0], dtype=torch.float64)},
            }},
            rule_summaries={"trapezoid": {"rule": {"nodes": [0.0]}, "node_alphas": [0.0], "node_losses": [1.0], "path_identity_hash": "9" * 64}},
        )
    assert len(curves.rows) == 6
    bundle_ref = aggregate["unit_shards"]["unit-2"]["bundle_ref"]  # type: ignore[index]
    tensor_path = tmp_path / Path(str(bundle_ref)) / "tensors" / "tensor-00000000.bin"
    tensor_path.write_bytes(tensor_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="TENSOR_(?:SIZE|HASH)_MISMATCH"):
        build_raw_reporting_tables(raw, workspace_root=tmp_path)


def test_raw_aggregate_metadata_and_iterator_are_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metadata validation must not call the legacy all-units loader."""

    from param_importance_nlp.experiments.stage3_raw_storage import (
        RAW_AGGREGATE_SCHEMA,
        VECTOR_DERIVATION_HASH,
        VECTOR_DERIVATION_SCHEMA,
        iter_raw_aggregate_units,
        load_raw_aggregate_metadata,
    )

    units = ["unit-a", "unit-b", "unit-c"]
    execution_hash = "1" * 64
    binding_hash = "2" * 64
    entries = {
        unit_id: {
            "shard_ref": f"raw/{unit_id}.json",
            "shard_hash": f"{index + 1:064x}",
            "bundle_ref": f"raw/{unit_id}.bundle",
            "bundle_manifest_hash": f"{index + 101:064x}",
            "path_identity_hash": f"{index + 201:064x}",
            "reference_artifact_hash": f"{index + 301:064x}",
            "reference_identity_hash": f"{index + 401:064x}",
        }
        for index, unit_id in enumerate(units)
    }
    aggregate: dict[str, object] = {
        "schema_version": RAW_AGGREGATE_SCHEMA,
        "execution_evidence_hash": execution_hash,
        "reference_binding_hash": binding_hash,
        "required_unit_ids": units,
        "candidate_rule_names": ["trapezoid"],
        "vector_derivation_schema": VECTOR_DERIVATION_SCHEMA,
        "vector_derivation_contract_hash": VECTOR_DERIVATION_HASH,
        "complete_unit_ids": units,
        "missing_unit_ids": [],
        "unit_shards": entries,
    }
    aggregate["artifact_hash"] = canonical_json_hash(aggregate)
    aggregate_path = tmp_path / "raw" / "aggregate.json"
    write_canonical_json(aggregate_path, aggregate)

    def fail_full_loader(**_kwargs: object) -> object:
        raise AssertionError("bounded API must not call load_raw_aggregate")

    monkeypatch.setattr(stage3_raw_storage, "load_raw_aggregate", fail_full_loader)
    metadata = load_raw_aggregate_metadata(
        root=tmp_path,
        aggregate_ref="raw/aggregate.json",
        aggregate_hash=aggregate["artifact_hash"],
    )
    assert metadata["required_unit_ids"] == units

    duplicate = dict(aggregate)
    duplicate["required_unit_ids"] = ["unit-a", "unit-a", "unit-b"]
    duplicate["complete_unit_ids"] = ["unit-a", "unit-a", "unit-b"]
    duplicate["unit_shards"] = {
        "unit-a": entries["unit-a"],
        "unit-b": entries["unit-b"],
    }
    duplicate["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in duplicate.items() if key != "artifact_hash"}
    )
    duplicate_path = tmp_path / "raw" / "aggregate-duplicate.json"
    write_canonical_json(duplicate_path, duplicate)
    with pytest.raises(ValueError, match="AGGREGATE_UNIT_SET_INVALID"):
        load_raw_aggregate_metadata(
            root=tmp_path,
            aggregate_ref="raw/aggregate-duplicate.json",
            aggregate_hash=duplicate["artifact_hash"],
        )

    calls: list[str] = []

    def fake_load_shard(*, root: Path, shard_ref: object, expected: object):
        assert root == tmp_path
        assert isinstance(expected, dict)
        unit_id = str(expected["unit_id"])
        calls.append(unit_id)
        entry = entries[unit_id]
        shard = {
            "artifact_hash": entry["shard_hash"],
            "bundle_ref": entry["bundle_ref"],
            "bundle_manifest_hash": entry["bundle_manifest_hash"],
            "path_identity_hash": entry["path_identity_hash"],
            "reference_artifact_hash": entry["reference_artifact_hash"],
            "reference_identity_hash": entry["reference_identity_hash"],
        }
        return shard, {"unit_id": unit_id}, object()

    monkeypatch.setattr(stage3_raw_storage, "_load_shard", fake_load_shard)
    yielded = [
        unit_id
        for unit_id, _shard, _state, _bundle in iter_raw_aggregate_units(
            root=tmp_path,
            aggregate_ref="raw/aggregate.json",
            aggregate_hash=aggregate["artifact_hash"],
        )
    ]
    assert yielded == units
    assert calls == units
