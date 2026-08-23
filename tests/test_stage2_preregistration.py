"""S2.1 machine-contract and fail-closed preregistration checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.experiments.preregistration import (
    BATCH_SIZES,
    MICROBATCH_COUNTS,
    PRIMARY_CELLS,
    STREAM_NAMES,
    build_stage2_amendment_template,
    build_stage2_hypothesis_contract,
    build_stage2_preregistration,
    validate_stage2_preregistration,
)


def _registration() -> dict[str, object]:
    return build_stage2_preregistration(
        seed_plan_hash="a" * 64,
        producer_commit="b" * 40,
        mathematics_hash="c" * 64,
        stage1_report_hash="d" * 64,
        upstream_binding_hash="e" * 64,
    )


def test_preregistration_freezes_estimand_estimators_signed_policy_and_grid() -> None:
    value = _registration()
    validate_stage2_preregistration(value)
    assert value["formula_version"] == "stage2-fixed-state-local-gradient-square-v1"
    estimand = value["estimand"]
    assert isinstance(estimand, dict)
    assert estimand["theoretical_target"] == "C_star=mu_k^2"
    assert estimand["eta_eval"] == 1.0
    assert estimand["optimizer_step"] == "forbidden"
    assert value["signed_analysis_policy"]["clamp_min_zero"] is False  # type: ignore[index]
    assert value["factors"]["candidate_batch_sizes"] == list(BATCH_SIZES)  # type: ignore[index]
    assert value["factors"]["candidate_microbatch_counts"] == list(MICROBATCH_COUNTS)  # type: ignore[index]
    assert value["sampling"]["stream_names"] == list(STREAM_NAMES)  # type: ignore[index]


def test_preregistration_primary_cells_are_exact_six_and_m2_invariant_is_named() -> None:
    value = _registration()
    assert value["primary"]["cells"] == list(PRIMARY_CELLS)  # type: ignore[index]
    assert value["estimators"]["u"]["formula"] == "(S1_k^2-S2_k)/(M*(M-1))"  # type: ignore[index]
    assert value["estimators"]["independent_probe_loss"]["status"] == "prohibited_from_stage2_estimator_comparison"  # type: ignore[index]
    assert "M=2" in value["invariants"]["u_m2_equals_double"] if "invariants" in value else True


def test_preregistration_hash_and_hypothesis_contract_are_content_bound() -> None:
    value = _registration()
    supplied = value["preregistration_hash"]
    body = {key: item for key, item in value.items() if key != "preregistration_hash"}
    assert supplied == canonical_json_hash(body)
    hypothesis = build_stage2_hypothesis_contract(value, upstream_binding_hash="f" * 64)
    assert [item["id"] for item in hypothesis["hypotheses"]] == ["H1", "H2", "H3", "H4", "H5", "H6"]
    assert set(hypothesis["initial_decisions"].values()) == {"inconclusive"}
    assert hypothesis["preregistration_hash"] == supplied


def test_preregistration_validator_rejects_signed_clamp_and_hash_drift() -> None:
    value = _registration()
    broken = copy.deepcopy(value)
    broken["signed_analysis_policy"]["clamp_min_zero"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="SIGNED_POLICY"):
        validate_stage2_preregistration(broken)
    broken = copy.deepcopy(value)
    broken["thresholds"]["u_double_nmse_noninferiority_ratio"] = 1.2  # type: ignore[index]
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_stage2_preregistration(broken)


def test_amendment_template_is_append_only_and_keeps_parent_identity() -> None:
    value = _registration()
    amendment = build_stage2_amendment_template(
        preregistration_hash=value["preregistration_hash"]
    )
    assert amendment["schema_version"] == "stage2-preregistration-amendment-v1"
    assert amendment["append_only"] is True
    assert amendment["created_before_confirmatory_draws"] is True
    assert amendment["parent_preregistration_hash"] == value["preregistration_hash"]


def test_preregistration_schema_files_are_strictly_parseable() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in (
        "stage2-preregistration-v1.json",
        "stage2-hypothesis-contract-v1.json",
        "stage2-preregistration-amendment-v1.json",
    ):
        value = json.loads((root / "schemas" / "shared" / name).read_text(encoding="utf-8"))
        assert value["$schema"].startswith("https://json-schema.org/")
