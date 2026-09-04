"""Regression tests for the formal S3.07 -> S3.08 authority boundary."""

from __future__ import annotations

import pytest

from ops.stage3.verify_stage3_s307_handoff import (
    Stage3HandoffError,
    _authoritative_s308_refs,
    _historical_s307_catalog,
)


def test_historical_candidate_gate_is_quarantined_from_s308() -> None:
    authoritative, quarantined = _authoritative_s308_refs(
        {
            "artifact_refs": {
                "formal_path_results": "results/s307/commits/formal_path_results.json",
                "completeness_report": "results/s307/commits/completeness_report.json",
                "gate_record": "results/s307/commits/gate_record.json",
            }
        }
    )
    assert authoritative == {
        "formal_path_results": "results/s307/commits/formal_path_results.json",
        "completeness_report": "results/s307/commits/completeness_report.json",
    }
    assert quarantined == {"gate_record": "results/s307/commits/gate_record.json"}


def test_historical_catalog_is_scoped_to_s307_output_contract() -> None:
    catalog = _historical_s307_catalog()
    assert catalog.get("stage3.07_formal_experiment_matrix").artifact_kinds == (
        "formal_path_results",
        "completeness_report",
        "gate_record",
    )
    assert catalog.get("stage3.08_error_analysis_and_stability").artifact_kinds == (
        "path_error_table",
        "stability_report",
        "frozen_source_table",
    )


def test_unknown_extra_output_fails_closed() -> None:
    with pytest.raises(Stage3HandoffError, match="S307_HANDOFF_OUTPUT_UNEXPECTED"):
        _authoritative_s308_refs(
            {
                "artifact_refs": {
                    "formal_path_results": "formal.json",
                    "completeness_report": "complete.json",
                    "unregistered_claim": "claim.json",
                }
            }
        )


@pytest.mark.parametrize("missing", ["formal_path_results", "completeness_report"])
def test_missing_scientific_output_fails_closed(missing: str) -> None:
    refs = {
        "formal_path_results": "formal.json",
        "completeness_report": "complete.json",
    }
    del refs[missing]
    with pytest.raises(Stage3HandoffError, match="S307_HANDOFF_OUTPUT_MISSING"):
        _authoritative_s308_refs({"artifact_refs": refs})
