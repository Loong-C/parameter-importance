"""Fail-closed design checks for the append-only S2.4 r22 round."""

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
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse  # noqa: E402
from param_importance_nlp.experiments.stage2_formal import OneShotReferencePlan, ReferenceSizingPlan  # noqa: E402
from param_importance_nlp.experiments.stage2_g23_contracts import generator_boundary  # noqa: E402
from param_importance_nlp.contracts.stage23 import validate_stage23_artifact  # noqa: E402


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
            "seed_namespace": "reference_sizing:frozen-parent-plan",
            "seed_namespace_mode": "same_frozen_seed_disjoint_segment",
            "parent_sampling_plan_hash": "b" * 64,
            "candidate_sample_counts": [32768, 65536],
            "block_size": 32,
            "normalized_l1_threshold": 0.02,
            "required_consecutive": 1,
            "complete_all_candidates": True,
            "optional_stopping": False,
            "reuse_prior_sizing_prefix": False,
            "segment_start_position": 16384,
            "segment_end_position_exclusive": 81920,
            "prior_consumed_end_position": 16384,
            "final_stream_segments": {
                "reference_A": {"start_position": 16384, "end_position_exclusive": 81920},
                "reference_B": {"start_position": 16384, "end_position_exclusive": 81920},
            },
        },
        "new_draws_before_freeze": False,
        "final_reference_created": False,
        "final_reference_plan_schema": "schemas/shared/stage2-reference-one-shot-plan-v2.json",
        "continuation_control": "precommitted_disjoint_segment_no_pooling_with_r21",
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
        (("sizing", "seed_namespace_mode"), "new_seed_namespace"),
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


def test_r22_segment_draw_positions_are_replayable_and_disjoint() -> None:
    sampling = SamplingPlan(
        universe=SamplingUniverse("r22-fixture-universe", tuple(range(32))),
        stream_seeds={"reference_sizing": 7, "reference_A": 11, "reference_B": 13, "pilot": 17, "confirmatory": 19},
    )
    old = sampling.draws("reference_sizing", 16384)
    segment = sampling.draws("reference_sizing", 65536, start=16384)
    replay = sampling.draws("reference_sizing", 65536, start=16384)
    assert old[-1].position == 16383
    assert segment[0].position == 16384 and segment[-1].position == 81919
    assert {item.draw_id for item in old}.isdisjoint(item.draw_id for item in segment)
    assert segment == replay


def test_r22_reference_plan_serializes_segment_and_terminal_gate() -> None:
    plan = ReferenceSizingPlan(
        reference_id="stage2-s204-r22-cell-sizing",
        candidate_sample_counts=(32768, 65536),
        block_size=32,
        convergence_tolerance=0.02,
        required_consecutive=1,
        draw_start_position=16384,
        draw_end_position_exclusive=81920,
        require_terminal_convergence=True,
        round_manifest_ref="evidence/stage2/s204/r22-round.json",
    )
    payload = plan.to_dict()
    validate_stage23_artifact(payload)
    assert payload["draw_start_position"] == 16384
    assert payload["draw_end_position_exclusive"] == 81920
    assert payload["require_terminal_convergence"] is True


def test_r22_final_ab_segments_are_disjoint_replayable_and_hash_bound() -> None:
    sampling = SamplingPlan(
        universe=SamplingUniverse("r22-ab-universe", tuple(range(32))),
        stream_seeds={"reference_sizing": 7, "reference_A": 11, "reference_B": 13, "pilot": 17, "confirmatory": 19},
    )
    old_a = sampling.draws("reference_A", 16384)
    old_b = sampling.draws("reference_B", 16384)
    new_a = sampling.draws("reference_A", 32768, start=16384)
    new_b = sampling.draws("reference_B", 32768, start=16384)
    assert {d.draw_id for d in old_a}.isdisjoint(d.draw_id for d in new_a)
    assert {d.draw_id for d in old_b}.isdisjoint(d.draw_id for d in new_b)
    assert new_a == sampling.draws("reference_A", 32768, start=16384)
    assert new_b == sampling.draws("reference_B", 32768, start=16384)
    boundary = generator_boundary(sampling, "reference_A", 32768, start=16384)
    assert boundary["start_position"] == 16384
    assert boundary["end_position_exclusive"] == 49152
    plan = OneShotReferencePlan(
        reference_id="stage2-s204-r22-final",
        sizing_result_hash="a" * 64,
        sample_count_per_stream=32768,
        block_size=32,
        schema_version="stage2-reference-one-shot-plan-v2",
        stream_a_draw_start_position=16384,
        stream_b_draw_start_position=16384,
    )
    payload = plan.to_dict()
    validate_stage23_artifact(payload)
    assert payload["artifact_hash"] == plan.artifact_hash
