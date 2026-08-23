from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_pilot import (
    ANCHOR_IDS,
    AnchorPilotResult,
    CostSemantics,
    build_confirmatory_mapping,
    freeze_fixture_matrix,
    freeze_formal_matrix,
    required_repetitions,
    run_artificial_distribution_calibration,
)


def _sampling() -> SamplingPlan:
    return SamplingPlan(
        SamplingUniverse("s206", tuple(range(64))),
        {"reference_sizing": 1, "reference_A": 2, "reference_B": 3, "pilot": 4, "confirmatory": 5},
    )


def _rows(b: int = 32, m: int = 16, r: int = 200) -> list[AnchorPilotResult]:
    return [
        AnchorPilotResult(a, b, m, 50, True, True, 0.1, r, 1024, 0.01, True, True)
        for a in ANCHOR_IDS
    ]


def test_s206_schemas_are_strict_json() -> None:
    root = Path(__file__).parents[1] / "schemas" / "shared"
    for name in ("stage2-pilot-matrix-freeze-v1.json", "stage2-confirmatory-mapping-v1.json"):
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        assert payload["additionalProperties"] is False
        assert payload["properties"]["schema_version"]["const"] in {"stage2-pilot-matrix-freeze-v1", "stage2-confirmatory-mapping-v1"}


def test_artificial_calibration_preserves_m2_and_weighted_contract() -> None:
    rng = np.random.default_rng(206)
    report = run_artificial_distribution_calibration(rng.normal(size=(64, 3)), batch_size=32)
    assert report.m2_max_abs_error < 1e-12
    assert report.weighted_u_max_abs_error < 1e-12
    assert report.scope == "local_fixture"


def test_fixture_scan_freezes_blind_pair_and_mapping() -> None:
    matrix = freeze_fixture_matrix(_rows())
    assert matrix.status == "FIXTURE_FROZEN"
    assert (matrix.b_primary, matrix.m_primary, matrix.r_primary) == (32, 16, 200)
    assert matrix.formal_eligible is False
    mapping = build_confirmatory_mapping(matrix, _sampling(), pilot_draw_ids=("pilot:000",))
    wire = mapping.to_dict()
    assert wire["stream"] == "confirmatory"
    assert wire["draw_id_unique"] is True
    assert wire["formal_eligible"] is False
    assert len(wire["mappings"]) == 200


def test_formal_freeze_and_formal_mapping_are_fail_closed() -> None:
    with pytest.raises(FormalRunRejected):
        freeze_formal_matrix(_rows(), execution=object())
    matrix = freeze_fixture_matrix(_rows())
    assert matrix.scope == "local_fixture"


def test_repetition_sizing_is_bounded_and_uses_reference_margin() -> None:
    assert 200 <= required_repetitions(estimator_variance=1.0, delta_sci=0.1, reference_half_width=0.01) <= 1000
    assert required_repetitions(estimator_variance=1.0, delta_sci=0.1, reference_half_width=0.02) > required_repetitions(estimator_variance=1.0, delta_sci=0.1, reference_half_width=0.0)
    with pytest.raises(ValueError):
        required_repetitions(estimator_variance=1.0, delta_sci=0.1, reference_half_width=0.03)
