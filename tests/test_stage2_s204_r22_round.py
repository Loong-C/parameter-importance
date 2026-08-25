"""Fail-closed design checks for the independent S2.4 r22 round."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from stage2.prepare_s204_r22_round import prepare_r22_round  # noqa: E402


def _round() -> dict[str, object]:
    return {
        "schema_version": "stage2-reference-sizing-round-v1",
        "round_id": "r22",
        "prior_round_id": "r21",
        "prior_round_status": "INCONCLUSIVE",
        "parent_preregistration_hash": "a" * 64,
        "prior_round_ref": "evidence/stage2/s204/formal-r20-g3-v5/r21/preparation.json",
        "sizing": {
            "stream": "reference_sizing",
            "seed_namespace": "stage2-s204-r22-reference-sizing-v1",
            "candidate_sample_counts": [32768, 65536],
            "block_size": 32,
            "normalized_l1_threshold": 0.02,
            "required_consecutive": 1,
            "complete_all_candidates": True,
            "optional_stopping": False,
            "reuse_prior_sizing_prefix": False,
        },
        "new_draws_before_freeze": False,
        "final_reference_created": False,
        "final_reference_plan_schema": "schemas/shared/stage2-reference-one-shot-plan-v1.json",
        "output_namespace": "evidence/stage2/s204/formal-r22-g3-v5",
        "amendment": {
            "append_only": True,
            "created_before_new_sizing_draws": True,
            "unchanged_scientific_contract": [
                "threshold=0.02", "block_size=32", "required_consecutive=1",
                "margin_schema_unchanged", "evaluator_schema_unchanged", "final_A_B_schema_unchanged",
            ],
        },
    }


def test_r22_freezes_two_nodes_without_reusing_r21() -> None:
    result = prepare_r22_round(_round())
    assert result["status"] == "FROZEN_BEFORE_NEW_SIZING_DRAWS"
    assert result["execution_contract"]["must_complete_candidate_nodes"] == [32768, 65536]  # type: ignore[index]
    assert len(result["artifact_hash"]) == 64  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "bad"),
    [
        (("sizing", "reuse_prior_sizing_prefix"), True),
        (("sizing", "optional_stopping"), True),
        (("sizing", "candidate_sample_counts"), [32768, 65536, 131072]),
        (("sizing", "seed_namespace"), "stage2-s204-r21-prefix"),
        (("final_reference_created",), True),
    ],
)
def test_r22_rejects_posthoc_or_partial_variants(path: tuple[str, ...], bad: object) -> None:
    value = copy.deepcopy(_round())
    target: object = value
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = bad  # type: ignore[index]
    with pytest.raises(ValueError):
        prepare_r22_round(value)


def test_schema_is_strict_and_round_trip_parseable() -> None:
    schema = json.loads((ROOT / "schemas/shared/stage2-reference-sizing-round-v1.json").read_text(encoding="utf-8"))
    assert schema["additionalProperties"] is False
    assert schema["properties"]["sizing"]["properties"]["candidate_sample_counts"]["const"] == [32768, 65536]
