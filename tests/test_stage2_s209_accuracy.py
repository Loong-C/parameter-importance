from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_s204_ids import EXPECTED_CELL_IDS
from param_importance_nlp.experiments.stage2_s209_accuracy import (
    S209AccuracyBlocked,
    S209_ACCURACY_SCHEMA,
    produce_s209_accuracy_sidecar,
)


def _with_hash(value: dict[str, object]) -> dict[str, object]:
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _bundle(root: Path) -> Path:
    g26 = root / "g26"
    g26.mkdir(parents=True)
    summary_rows: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    for cell in EXPECTED_CELL_IDS:
        for method in ("raw", "double", "u_m16"):
            summary_rows.append(
                {
                    "cell_id": cell,
                    "method": method,
                    "microbatch_count": 16,
                    "mse_observed": 2.0,
                    "corrected_nmse": 1.0,
                    "ranking_repeat_mean": {
                        "spearman": 0.9,
                        "overlap_at_0.01": 0.8,
                    },
                }
            )
            long_rows.extend(
                [
                    {
                        "cell_id": cell,
                        "method": method,
                        "reference_view": "bias",
                        "mse_observed": 2.0,
                    },
                    {
                        "cell_id": cell,
                        "method": method,
                        "reference_view": "ranking_repetition_mean",
                        "spearman": 0.9,
                        "overlap_at_0.01": 0.8,
                    },
                ]
            )
    names: dict[str, dict[str, object]] = {
        "g2.6-gate.json": {
            "schema_version": "stage2-s208-g26-gate-v1",
            "gate_id": "stage2.G2.6",
            "stage": 2,
            "status": "PASS",
            "quality_gate_dependency": True,
            "measured": {"matrix_hash": "a" * 64, "raw_manifest_hash": "b" * 64},
        },
        "quality_gates.json": {
            "schema_version": "stage2-s208-quality-gates-v1",
            "gate_id": "stage2.G2.6",
            "status": "PASS",
            "formal_eligible": True,
            "gates": [{"name": "all", "status": "PASS"}],
        },
        "hypothesis_decisions.json": {
            "schema_version": "stage2-s208-hypothesis-decisions-v1",
            "quality_gate_dependency": "passed",
        },
        "statistics_long_table.json": {
            "schema_version": "stage2-s208-g26-analysis-v1",
            "rows": long_rows,
        },
        "statistics_summary.json": {
            "schema_version": "stage2-s208-g26-analysis-v1",
            "rows": summary_rows,
        },
        "confirmatory_family_decisions.json": {
            "schema_version": "stage2-s208-confirmatory-family-v1",
            "primary_cells": list(EXPECTED_CELL_IDS),
            "rows": [],
        },
        "lineage_manifest.json": {
            "schema_version": "stage2-s208-lineage-v1",
            "producer_task": "stage2.08_statistics_and_robustness",
            "raw_manifest_hash": "b" * 64,
            "matrix_hash": "a" * 64,
            "derived_artifacts": [
                "analysis_input_audit.json",
                "statistics_long_table.json",
                "statistics_summary.json",
                "raw_calibration.json",
                "confirmatory_family_decisions.json",
                "quality_gates.json",
                "hypothesis_decisions.json",
                "lineage_manifest.json",
            ],
        },
    }
    for name, payload in names.items():
        write_canonical_json(g26 / name, _with_hash(payload))
    return g26


def test_producer_maps_hash_bound_g26_grid(tmp_path: Path) -> None:
    g26 = _bundle(tmp_path)
    sidecar = produce_s209_accuracy_sidecar(
        data_root=tmp_path,
        g26_root_ref="g26",
        output_ref="s209/accuracy.json",
        run_id="s209-run",
    )
    assert sidecar["schema_version"] == S209_ACCURACY_SCHEMA
    assert len(sidecar["rows"]) == 18
    assert {row["method"] for row in sidecar["rows"]} == {"raw", "double", "u"}
    assert (tmp_path / "s209/accuracy.json").exists()


def test_producer_rejects_tampered_g26_hash(tmp_path: Path) -> None:
    g26 = _bundle(tmp_path)
    path = g26 / "statistics_summary.json"
    payload = path.read_text(encoding="utf-8")
    path.write_text(payload.replace('"corrected_nmse":1.0', '"corrected_nmse":2.0', 1), encoding="utf-8")
    with pytest.raises(S209AccuracyBlocked, match="CANONICAL_READ_FAILED|ARTIFACT_HASH_MISMATCH"):
        produce_s209_accuracy_sidecar(
            data_root=tmp_path,
            g26_root_ref="g26",
            output_ref="s209/accuracy.json",
            run_id="s209-run",
        )


def test_producer_rejects_unknown_summary_method(tmp_path: Path) -> None:
    g26 = _bundle(tmp_path)
    path = g26 / "statistics_summary.json"
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["rows"][0]["method"] = "u_m32"
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(path, payload)
    with pytest.raises(S209AccuracyBlocked, match="G26_ACCURACY_METHOD_SET_INCOMPLETE|G26_ACCURACY_METHOD_UNKNOWN"):
        produce_s209_accuracy_sidecar(
            data_root=tmp_path,
            g26_root_ref="g26",
            output_ref="s209/accuracy.json",
            run_id="s209-run",
        )
