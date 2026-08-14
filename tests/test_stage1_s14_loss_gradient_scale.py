from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
import param_importance_nlp.stage1_gradient_scale as gradient_scale
from param_importance_nlp.stage1_gradient_scale import (
    Stage1GradientScaleError,
    build_stage1_s14_evidence,
    replay_stage1_s14_evidence,
    validate_stage1_s14_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _rebind(evidence: dict[str, object]) -> None:
    report = evidence["gradient_scale_report"]
    assert isinstance(report, dict)
    report_body = dict(report)
    report_body.pop("report_hash")
    report["report_hash"] = canonical_json_hash(report_body)
    table = evidence["comparison_table"]
    assert isinstance(table, dict)
    table["report_hash"] = report["report_hash"]
    table_body = dict(table)
    table_body.pop("table_hash")
    table["table_hash"] = canonical_json_hash(table_body)
    gate = evidence["gate_record"]
    assert isinstance(gate, dict)
    gate["report_hash"] = report["report_hash"]
    gate["comparison_table_hash"] = table["table_hash"]
    gate_body = dict(gate)
    gate_body.pop("artifact_hash")
    gate["artifact_hash"] = canonical_json_hash(gate_body)


def test_s14_executes_both_profiles_and_full_required_route_matrix() -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="a" * 40)
    replay = validate_stage1_s14_evidence(evidence)
    report = evidence["gradient_scale_report"]
    assert report["status"] == "PASS"
    assert report["gate_status"] == "NOT_RUN"
    assert {profile["profile"] for profile in report["profiles"]} == {
        "T64_ORACLE",
        "T32_SINGLE",
    }
    required = (
        "adapter_full_gradient",
        "adapter_full_loss",
        "pre_shifted_adapter_full_gradient",
        "pre_shifted_adapter_full_loss",
        "per_sample_reconstruction_gradient",
        "per_sample_reconstruction_loss",
        "per_token_reconstruction_gradient",
        "per_token_reconstruction_loss",
        "equal_microbatch_m2_gradient",
        "equal_microbatch_m2_loss",
        "equal_microbatch_reordered_gradient",
        "equal_microbatch_reordered_loss",
        "equal_microbatch_m4_gradient",
        "equal_microbatch_m4_loss",
        "token_weighted_microbatch_gradient",
        "token_weighted_microbatch_loss",
        "sum_divided_by_effective_count_gradient",
        "sum_divided_by_effective_count_loss",
        "accumulation_m2_gradient",
        "accumulation_m2_local_gradient_reconstruction",
        "accumulation_m2_local_loss_reconstruction",
        "accumulation_m4_gradient",
        "accumulation_m4_local_gradient_reconstruction",
        "accumulation_m4_local_loss_reconstruction",
        "accumulation_weighted_m2_gradient",
        "accumulation_weighted_m2_local_gradient_reconstruction",
        "accumulation_weighted_m2_local_loss_reconstruction",
        "rng_eval_repeat_gradient",
    )
    for profile in report["profiles"]:
        assert tuple(item["comparison_id"] for item in profile["comparisons"]) == required
        assert all(item["passed"] for item in profile["comparisons"])
        assert profile["effective_token_counts"]["equal_m2_weights"] == [0.5, 0.5]
        assert profile["effective_token_counts"]["weighted_normalized_weights"] == [0.25, 0.75]
        assert profile["effective_token_counts"]["per_sample"] == [2, 2, 2, 2]
        assert profile["negative_control"]["gradient"]["passed"] is False
        assert profile["negative_control"]["loss"]["passed"] is False
        assert all(item["clear_was_complete"] is True for item in profile["accumulation"])
    assert report["sample_contract"]["zero_effective_token_rejected"] is True
    assert report["sample_contract"]["pre_shifted_zero_effective_token_rejected"] is True
    assert report["sample_contract"]["ignored_target_locations"] == [
        {"sample_id": "s14-sample-0", "target_position": 1, "attention_mask": 1, "label": -100}
    ]
    assert replay["tensor_row_count"] == len(evidence["comparison_table"]["rows"])


def test_s14_is_deterministic_and_preserves_caller_rng() -> None:
    import torch

    torch.manual_seed(73)
    before = torch.random.get_rng_state().clone()
    first = build_stage1_s14_evidence(ROOT, producer_commit="b" * 40)
    middle = torch.random.get_rng_state().clone()
    second = build_stage1_s14_evidence(ROOT, producer_commit="b" * 40)
    after = torch.random.get_rng_state().clone()
    assert torch.equal(before, middle)
    assert torch.equal(middle, after)
    assert first["gradient_scale_report"]["report_hash"] == second["gradient_scale_report"]["report_hash"]
    for profile in first["gradient_scale_report"]["profiles"]:
        exact = profile["rng"]["exact_equivalence"]
        assert exact["cpu_rng_before"] == exact["cpu_rng_between"] == exact["cpu_rng_after"]
        smoke = profile["rng"]["dropout_smoke"]
        assert len(
            {
                smoke["cpu_rng_before"],
                smoke["cpu_rng_after_first_microbatch"],
                smoke["cpu_rng_after_second_microbatch"],
            }
        ) == 3


def test_s14_manual_oracle_safely_excludes_ignore_index_with_attention_enabled() -> None:
    import torch

    inputs, labels, mask = gradient_scale._fixture_tensors()
    assert labels[0, 1].item() == -100
    assert mask[0, 1].item() == 1
    numerator, count = gradient_scale._manual_token_loss(
        torch.zeros((4, 4, 7), dtype=torch.float64), labels, mask
    )
    assert count == 8
    assert float(numerator) > 0.0


def test_s14_rejects_rehashed_negative_control_or_sample_contract_tampering() -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="c" * 40)
    tampered = deepcopy(evidence)
    tampered["gradient_scale_report"]["profiles"][0]["negative_control"]["detected"] = False
    _rebind(tampered)
    with pytest.raises(Stage1GradientScaleError, match="NEGATIVE_CONTROL_INVALID"):
        validate_stage1_s14_evidence(tampered)


def test_s14_rejects_rehashed_table_projection_and_jointly_rehashed_numerics() -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="c" * 40)
    tampered = deepcopy(evidence)
    tampered["comparison_table"]["rows"][0]["max_absolute_error"] = 0.25
    _rebind(tampered)
    with pytest.raises(Stage1GradientScaleError, match="TABLE_PROJECTION_INVALID"):
        validate_stage1_s14_evidence(tampered)

    # This changes only a report-global numeric field, then rehashes every
    # role.  It intentionally survives structural/projected-table validation
    # and proves that replay recomputes the actual fixture rather than merely
    # aggregating hashes.
    tampered = deepcopy(evidence)
    tampered["gradient_scale_report"]["profiles"][0]["comparisons"][0]["global"][
        "max_absolute_error"
    ] = 0.25
    _rebind(tampered)
    assert validate_stage1_s14_evidence(tampered)["status"] == "PASS"
    with pytest.raises(Stage1GradientScaleError, match="OFFLINE_REPLAY_ROLE_MISMATCH"):
        replay_stage1_s14_evidence(tampered, ROOT)


def test_s14_replay_is_real_and_report_keys_are_fail_closed() -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="1" * 40)
    replay = replay_stage1_s14_evidence(evidence, ROOT)
    assert replay["status"] == "PASS"
    assert len(replay["comparison_hashes"]) == 56
    tampered = deepcopy(evidence)
    tampered["gradient_scale_report"]["unexpected"] = True
    _rebind(tampered)
    with pytest.raises(Stage1GradientScaleError, match="ROLE_KEY_SET_INVALID"):
        validate_stage1_s14_evidence(tampered)
    tampered = deepcopy(evidence)
    del tampered["gradient_scale_report"]["producer_commit"]
    _rebind(tampered)
    with pytest.raises(Stage1GradientScaleError, match="ROLE_KEY_SET_INVALID"):
        validate_stage1_s14_evidence(tampered)
    with pytest.raises(Stage1GradientScaleError, match="PRODUCER_COMMIT_INVALID"):
        build_stage1_s14_evidence(ROOT, producer_commit="A" * 40)
    tampered = deepcopy(evidence)
    tampered["gradient_scale_report"]["sample_contract"]["ordered_sample_ids"].reverse()
    _rebind(tampered)
    with pytest.raises(Stage1GradientScaleError, match="SAMPLE_OR_ZERO_TOKEN_CONTRACT_INVALID"):
        validate_stage1_s14_evidence(tampered)


def test_s14_cannot_declare_a_formal_gate_without_exact_s13_v2_handoff() -> None:
    with pytest.raises(Stage1GradientScaleError, match="FORMAL_S1_3_V2_HANDOFF_REQUIRED"):
        build_stage1_s14_evidence(ROOT, producer_commit="f" * 40, scope="formal")

    upstream = {
        "s1_3_index_ref": "evidence/stage1/s1-3-formal/current/index.json",
        "s1_3_index_sha256": "51eb16bf87d73d68f6c1da49b7635fa42bd0456e9305f7263326a794b9b2f2ab",
        "s1_3_gate_artifact_hash": "1" * 64,
        "s1_3_fixture_manifest_sha256": "2" * 64,
        "s1_3_oracle_bundle_sha256": "3" * 64,
        "s1_3_oracle_validation_report_sha256": "4" * 64,
        "s1_3_replay_sha256": "5" * 64,
        "s1_3_validation_sha256": "6" * 64,
        "s1_3_frozen_gradient_input_hash": "7" * 64,
    }
    formal = build_stage1_s14_evidence(
        ROOT, producer_commit="f" * 40, scope="formal", upstream_evidence=upstream
    )
    assert formal["gate_record"]["status"] == "PASS"
    assert validate_stage1_s14_evidence(formal)["status"] == "PASS"
    with pytest.raises(Stage1GradientScaleError, match="FORMAL_S1_3_V2_HANDOFF_REQUIRED"):
        build_stage1_s14_evidence(
            ROOT,
            producer_commit="f" * 40,
            scope="formal",
            upstream_evidence={**upstream, "unexpected": "x"},
        )


def test_s14_role_artifacts_roundtrip_and_schemas_are_strict(tmp_path: Path) -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="d" * 40)
    reloaded: dict[str, object] = {}
    for role, payload in evidence.items():
        path = tmp_path / f"{role}.json"
        write_canonical_json(path, payload)
        reloaded[role] = load_canonical_json(path)
    assert validate_stage1_s14_evidence(reloaded)["status"] == "PASS"
    for name, version in {
        "g1-grad-report-v1.json": "stage1-g1-grad-report-v1",
        "g1-grad-comparison-table-v1.json": "stage1-g1-grad-comparison-table-v1",
        "g1-grad-gate-record-v1.json": "stage1-g1-grad-gate-record-v1",
        "s1-4-formalization-index-v1.json": "stage1-s1-4-formalization-index-v1",
        "s1-4-validation-v1.json": "stage1-s1-4-validation-v1",
    }.items():
        schema = json.loads((ROOT / "schemas/stage1" / name).read_text(encoding="utf-8"))
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["properties"]["schema_version"]["const"] == version
    report_schema = json.loads((ROOT / "schemas/stage1/g1-grad-report-v1.json").read_text(encoding="utf-8"))
    assert report_schema["$defs"]["formalUpstream"]["additionalProperties"] is False
    assert report_schema["$defs"]["sourceHashes"]["additionalProperties"] is False
    table_schema = json.loads((ROOT / "schemas/stage1/g1-grad-comparison-table-v1.json").read_text(encoding="utf-8"))
    assert table_schema["$defs"]["scatterRow"]["additionalProperties"] is False
    assert table_schema["$defs"]["accumulationRow"]["additionalProperties"] is False
    validation_schema = json.loads((ROOT / "schemas/stage1/s1-4-validation-v1.json").read_text(encoding="utf-8"))
    assert validation_schema["$defs"]["regression"]["additionalProperties"] is False
    assert validation_schema["$defs"]["roleMap"]["additionalProperties"] is False
