#!/usr/bin/env python3
"""Prepare the append-only S2.4 r22 sizing-round contract.

This is deliberately a control-plane-only command.  It does not read a draw,
extend r21, create final A/B manifests, or run a provider.  r22 is an
append-only sequential continuation over a pre-registered, unconsumed segment
of the already frozen reference_sizing stream; it is never pooled with r21.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping


SCHEMA_VERSION = "stage2-reference-sizing-round-v1"
ROUND_ID = "r22"
PRIOR_ROUND_ID = "r21"
STREAM = "reference_sizing"
CANDIDATE_SAMPLE_COUNTS = (32768, 65536)
SEGMENT_START_POSITION = 16384
SEGMENT_END_POSITION_EXCLUSIVE = SEGMENT_START_POSITION + CANDIDATE_SAMPLE_COUNTS[-1]
BLOCK_SIZE = 32
NORMALIZED_L1_THRESHOLD = 0.02
REQUIRED_CONSECUTIVE = 1
FINAL_REFERENCE_PLAN_SCHEMA = "schemas/shared/stage2-reference-one-shot-plan-v1.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def validate_r22_round(value: Mapping[str, Any]) -> None:
    """Fail closed on post-hoc extension, prefix reuse, or partial execution."""

    if value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("S204_R22_SCHEMA_UNSUPPORTED")
    if value.get("round_id") != ROUND_ID:
        raise ValueError("S204_R22_ROUND_ID_MISMATCH")
    if value.get("prior_round_id") != PRIOR_ROUND_ID:
        raise ValueError("S204_R22_PRIOR_ROUND_MISMATCH")
    if value.get("prior_round_status") != "INCONCLUSIVE":
        raise ValueError("S204_R22_PRIOR_ROUND_MUST_REMAIN_INCONCLUSIVE")
    parent = value.get("parent_preregistration_hash")
    if not isinstance(parent, str) or _SHA256.fullmatch(parent) is None:
        raise ValueError("S204_R22_PARENT_PREREG_HASH_INVALID")
    prior_ref = value.get("prior_round_ref")
    if not isinstance(prior_ref, str) or not prior_ref or prior_ref == str(value.get("output_namespace")):
        raise ValueError("S204_R22_PRIOR_ROUND_REF_INVALID")
    sizing = value.get("sizing")
    if not isinstance(sizing, Mapping):
        raise ValueError("S204_R22_SIZING_REQUIRED")
    if sizing.get("stream") != STREAM:
        raise ValueError("S204_R22_STREAM_MISMATCH")
    namespace = sizing.get("seed_namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("S204_R22_SEED_NAMESPACE_REQUIRED")
    if sizing.get("seed_namespace_mode") != "same_frozen_seed_disjoint_segment":
        raise ValueError("S204_R22_SEED_NAMESPACE_MODE_INVALID")
    parent_sampling_plan_hash = sizing.get("parent_sampling_plan_hash")
    if not isinstance(parent_sampling_plan_hash, str) or _SHA256.fullmatch(parent_sampling_plan_hash) is None:
        raise ValueError("S204_R22_PARENT_SAMPLING_PLAN_HASH_INVALID")
    if tuple(sizing.get("candidate_sample_counts", ())) != CANDIDATE_SAMPLE_COUNTS:
        raise ValueError("S204_R22_CANDIDATE_NODES_MISMATCH")
    if sizing.get("block_size") != BLOCK_SIZE:
        raise ValueError("S204_R22_BLOCK_SIZE_MISMATCH")
    if sizing.get("normalized_l1_threshold") != NORMALIZED_L1_THRESHOLD:
        raise ValueError("S204_R22_THRESHOLD_MISMATCH")
    if sizing.get("required_consecutive") != REQUIRED_CONSECUTIVE:
        raise ValueError("S204_R22_CONSECUTIVE_MISMATCH")
    if sizing.get("complete_all_candidates") is not True:
        raise ValueError("S204_R22_ALL_NODES_REQUIRED")
    if sizing.get("optional_stopping") is not False:
        raise ValueError("S204_R22_OPTIONAL_STOPPING_FORBIDDEN")
    if sizing.get("reuse_prior_sizing_prefix") is not False:
        raise ValueError("S204_R22_PREFIX_REUSE_FORBIDDEN")
    if sizing.get("segment_start_position") != SEGMENT_START_POSITION:
        raise ValueError("S204_R22_SEGMENT_START_INVALID")
    if sizing.get("segment_end_position_exclusive") != SEGMENT_END_POSITION_EXCLUSIVE:
        raise ValueError("S204_R22_SEGMENT_END_INVALID")
    if sizing.get("prior_consumed_end_position") != SEGMENT_START_POSITION:
        raise ValueError("S204_R22_PRIOR_SEGMENT_BOUNDARY_INVALID")
    if value.get("new_draws_before_freeze") is not False:
        raise ValueError("S204_R22_FREEZE_REQUIRED_BEFORE_DRAWS")
    if value.get("final_reference_created") is not False:
        raise ValueError("S204_R22_FINAL_REFERENCE_NOT_YET_ALLOWED")
    if value.get("final_reference_plan_schema") != FINAL_REFERENCE_PLAN_SCHEMA:
        raise ValueError("S204_R22_FINAL_SCHEMA_MISMATCH")
    output_namespace = value.get("output_namespace")
    if not isinstance(output_namespace, str) or "r22" not in output_namespace or "r21" in output_namespace:
        raise ValueError("S204_R22_OUTPUT_NAMESPACE_INVALID")
    amendment = value.get("amendment")
    if not isinstance(amendment, Mapping):
        raise ValueError("S204_R22_AMENDMENT_REQUIRED")
    if amendment.get("append_only") is not True or amendment.get("created_before_new_sizing_draws") is not True:
        raise ValueError("S204_R22_AMENDMENT_ORDER_INVALID")
    if amendment.get("unchanged_scientific_contract") != [
        "threshold=0.02",
        "block_size=32",
        "required_consecutive=1",
        "margin_schema_unchanged",
        "evaluator_schema_unchanged",
        "final_A_B_schema_unchanged",
    ]:
        raise ValueError("S204_R22_SCIENTIFIC_CONTRACT_DRIFT")


def prepare_r22_round(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content-addressed frozen plan; no draw is generated."""

    validate_r22_round(value)
    body = dict(value)
    body.pop("artifact_hash", None)
    body["status"] = "FROZEN_BEFORE_NEW_SIZING_DRAWS"
    body["formal_reference_status"] = "NOT_CREATED_UNTIL_SIZING_GATE"
    body["execution_contract"] = {
        "must_complete_candidate_nodes": list(CANDIDATE_SAMPLE_COUNTS),
        "node_order": list(CANDIDATE_SAMPLE_COUNTS),
        "resume_requires_same_round_hash": True,
        "resume_requires_same_seed_namespace": True,
        "resume_requires_same_candidate_nodes": True,
        "resume_requires_same_segment_start": True,
        "resume_requires_same_segment_end": True,
        "draw_positions": {
            "start": SEGMENT_START_POSITION,
            "end_exclusive": SEGMENT_END_POSITION_EXCLUSIVE,
            "prior_r21_prefix_is_read_only": True,
        },
        "r21_output_is_read_only": True,
    }
    body["artifact_hash"] = _canonical_hash(body)
    return body


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare S2.4 r22 round contract")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    value = json.loads(args.input.read_text(encoding="utf-8"))
    result = prepare_r22_round(value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["prepare_r22_round", "validate_r22_round"]
