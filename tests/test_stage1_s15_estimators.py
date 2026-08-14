from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from param_importance_nlp.core.estimators import DoubleSampleProvenance, EstimatorResult, double_sample_importance, raw_importance
from param_importance_nlp.core.tensors import TensorMap
from param_importance_nlp.stage1_estimator_oracle import Stage1EstimatorOracleError, tensor_map_from_wire
from param_importance_nlp.stage1_estimators import (
    Stage1EstimatorError,
    _build_table,
    _derive_gate_requirements,
    _fixture_registry,
    _hash_role,
    build_stage1_s15_evidence,
    replay_stage1_s15_evidence,
    validate_stage1_s15_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _rebalance_hashes(evidence: dict[str, object]) -> None:
    """Rehash a coordinated in-memory tamper to prove replay has an external root."""

    bundle = evidence["tensor_bundle"]
    oracle = evidence["oracle_report"]
    report = evidence["estimator_report"]
    gate = evidence["gate_record"]
    assert isinstance(bundle, dict) and isinstance(oracle, dict) and isinstance(report, dict) and isinstance(gate, dict)
    bundle["bundle_hash"] = _hash_role(bundle, field="bundle_hash")
    oracle["bundle_hash"] = bundle["bundle_hash"]
    oracle["oracle_hash"] = _hash_role(oracle, field="oracle_hash")
    report["bundle_hash"] = bundle["bundle_hash"]
    report["oracle_hash"] = oracle["oracle_hash"]
    report["report_hash"] = _hash_role(report, field="report_hash")
    table = _build_table(report)
    table["table_hash"] = _hash_role(table, field="table_hash")
    evidence["comparison_table"] = table
    gate["report_hash"] = report["report_hash"]
    gate["oracle_hash"] = oracle["oracle_hash"]
    gate["bundle_hash"] = bundle["bundle_hash"]
    gate["comparison_table_hash"] = table["table_hash"]
    gate["requirements"] = _derive_gate_requirements(report, table)
    gate["artifact_hash"] = _hash_role(gate, field="artifact_hash")


def test_s15_builds_and_replays_the_frozen_fixture() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="a" * 40)
    assert evidence["gate_record"]["status"] == "NOT_RUN"
    replay = replay_stage1_s15_evidence(evidence, source_root=ROOT)
    assert replay["status"] == "PASS"
    for profile in replay["profiles"]:
        assert len(profile["statistics_comparisons"]) == 4
        assert len(profile["output_comparisons"]) == 20


def test_s15_registry_group_mapping_and_raw_clip_gates_are_per_tensor() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="b" * 40)
    report = evidence["estimator_report"]
    assert report["coordinate_contract_hash"]
    for profile in report["profiles"]:
        assert profile["parameter_to_group"] == {"bias": "group_0000", "head": "group_0001", "weight": "group_0000"}
        assert profile["learning_rates"] == {"group_0000": 0.25, "group_0001": 0.05}
        assert all(all(values.values()) for values in profile["raw_clip_boundary"].values())
        assert profile["double_sampling"]["sample_ids_a"][0] == profile["double_sampling"]["sample_ids_b"][0] == "overlap-0"
        provenance = profile["double_sampling"]["provenance"]
        assert provenance["rng_stream_a"] != provenance["rng_stream_b"]
        assert provenance["rng_state_digest_a"] != provenance["rng_state_digest_b"]


def test_s15_full_estimator_matrix_claims_permutations_and_rejections_are_frozen() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="7" * 40)
    expected_comparisons = {
        "raw_core", "raw_score", "raw_clipped_score", "double_core", "double_score",
        "equal_u_ordered_core", "equal_u_unordered_core", "equal_u_streaming_core", "equal_u_score",
        "weighted_u_core", "weighted_u_score", "weighted_equal_count_degenerate_core",
        "m2_u_equals_double_core", "double_exchange_core", "identical_gradient_u_core",
        "identical_gradient_u_score", "equal_u_reverse_permutation_core",
        "equal_u_rotate_permutation_core", "weighted_u_reverse_paired_permutation_core",
        "weighted_u_rotate_paired_permutation_core",
    }
    expected_rejections = {
        "m1_rejected", "same_batch_object_rejected", "shared_sampler_state_rejected",
        "nonfinite_rejected", "shape_mismatch_rejected", "registry_mismatch_rejected",
        "dtype_device_mismatch_rejected", "nonpositive_weight_rejected",
        "nonpositive_denominator_rejected", "negative_learning_rate_rejected",
        "nonfinite_learning_rate_rejected", "nonfinite_clip_rejected",
        "out_of_range_clip_rejected", "float16_accumulation_rejected",
    }
    for profile in evidence["estimator_report"]["profiles"]:
        assert {row["comparison_id"] for row in profile["comparisons"]} == expected_comparisons
        assert all(row["passed"] is True for row in profile["comparisons"])
        assert set(profile["rejections"]) == expected_rejections
        assert all(profile["rejections"].values())
        fields = profile["estimator_public_fields"]
        assert fields["weighted_u"]["unbiasedness_claim"] == "unbiased_fixed_state_under_declared_sampling_assumptions"
        assert fields["weighted_u_assumptions_false"]["unbiasedness_claim"] == "no_unbiasedness_claim"
    requirements = evidence["gate_record"]["requirements"]
    assert requirements["identical_gradient_degeneracy"] is True
    assert requirements["paired_permutation_invariance"] is True
    assert requirements["estimator_public_field_bindings"] is True


def test_s15_same_optimizer_group_multiple_parameters_and_missing_extra_groups_fail_closed() -> None:
    registry = _fixture_registry()
    assert registry.record("bias").group_id == registry.record("weight").group_id == "group_0000"
    core = TensorMap({name: torch.ones(record.shape, dtype=torch.float64) for name, record in ((name, registry.record(name)) for name in registry.eligible_names)}, registry=registry)
    result = EstimatorResult.from_core("local_gradient_space_importance_raw", core, {"group_0000": 0.25, "group_0001": 0.05})
    assert torch.equal(result.score["bias"], torch.full_like(result.score["bias"], 0.25))
    assert torch.equal(result.score["weight"], torch.full_like(result.score["weight"], 0.25))
    assert torch.equal(result.score["head"], torch.full_like(result.score["head"], 0.05))
    with pytest.raises(Exception):
        EstimatorResult.from_core("local_gradient_space_importance_raw", core, {"group_0000": 0.25})
    with pytest.raises(Exception):
        EstimatorResult.from_core("local_gradient_space_importance_raw", core, {"group_0000": 0.25, "group_0001": 0.05, "group_extra": 0.1})


def test_s15_strict_wire_parser_rejects_bool_string_and_empty_coordinate_name() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="c" * 40)
    wire = evidence["tensor_bundle"]["profiles"][0]["microbatch_samples"][0]
    for replacement in (True, "1.0"):
        invalid = copy.deepcopy(wire)
        invalid["bias"]["values"][0] = replacement
        with pytest.raises(Stage1EstimatorOracleError):
            tensor_map_from_wire(invalid, field="test", coordinate_order=("bias", "head", "weight"), registry=_fixture_registry())
    invalid_name = copy.deepcopy(wire)
    invalid_name[""] = invalid_name.pop("bias")
    with pytest.raises(Stage1EstimatorOracleError):
        tensor_map_from_wire(invalid_name, field="test", coordinate_order=("bias", "head", "weight"), registry=_fixture_registry())


def test_s15_replay_rejects_joint_rehash_of_bundle_input_because_manifest_is_external() -> None:
    evidence = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="d" * 40))
    evidence["tensor_bundle"]["profiles"][0]["microbatch_samples"][0]["bias"]["values"][0] = 999.0
    _rebalance_hashes(evidence)
    assert validate_stage1_s15_evidence(evidence, source_root=ROOT)["status"] == "PASS"
    with pytest.raises(Stage1EstimatorError, match="FROZEN_FIXTURE_INPUT_MISMATCH"):
        replay_stage1_s15_evidence(evidence, source_root=ROOT)


def test_s15_double_provenance_allows_overlapping_ids_but_rejects_shared_runtime_source() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="e" * 40)
    provenance = evidence["estimator_report"]["profiles"][0]["double_sampling"]["provenance"]
    valid = DoubleSampleProvenance(**provenance)
    assert valid.sample_ids_may_overlap is True
    invalid = dict(provenance)
    invalid["sampler_state_b"] = invalid["sampler_state_a"]
    with pytest.raises(Exception):
        DoubleSampleProvenance(**invalid)
    invalid_design = dict(provenance)
    invalid_design["sampling_design"] = "without_replacement"
    with pytest.raises(Exception):
        DoubleSampleProvenance(**invalid_design)
    invalid_digest = dict(provenance)
    invalid_digest["rng_state_digest_b"] = "not-a-sha256"
    with pytest.raises(Exception):
        DoubleSampleProvenance(**invalid_digest)
    samples = evidence["tensor_bundle"]["profiles"][0]["microbatch_samples"]
    gradient = tensor_map_from_wire(samples[0], field="sample", coordinate_order=("bias", "head", "weight"), registry=_fixture_registry())
    with pytest.raises(Exception):
        double_sample_importance(gradient, gradient, provenance=valid)


def test_s15_sample_id_and_rng_stream_drift_are_bound_to_the_frozen_manifest() -> None:
    evidence = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="f" * 40))
    report_profile = evidence["estimator_report"]["profiles"][0]
    bundle_profile = evidence["tensor_bundle"]["profiles"][0]
    report_profile["double_sampling"]["provenance"]["rng_stream_b"] = "tampered-independent-stream"
    bundle_profile["double_provenance"]["rng_stream_b"] = "tampered-independent-stream"
    _rebalance_hashes(evidence)
    with pytest.raises(Stage1EstimatorError, match="DOUBLE_PROVENANCE_BINDING_INVALID"):
        validate_stage1_s15_evidence(evidence, source_root=ROOT)

    sample_id_tamper = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="f" * 40))
    report_sampling = sample_id_tamper["estimator_report"]["profiles"][0]["double_sampling"]
    bundle_sampling = sample_id_tamper["tensor_bundle"]["profiles"][0]
    report_sampling["sample_ids_b"][1] = "tampered-id"
    bundle_sampling["double_sample_ids_b"][1] = "tampered-id"
    _rebalance_hashes(sample_id_tamper)
    with pytest.raises(Stage1EstimatorError, match="DOUBLE_PROVENANCE_BINDING_INVALID"):
        validate_stage1_s15_evidence(sample_id_tamper, source_root=ROOT)


def test_s15_input_contract_and_exact_rejection_matrix_are_fail_closed() -> None:
    evidence = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="1" * 40))
    evidence["estimator_report"]["estimator_input_contract"]["gradient_scale_restored"] = False
    evidence["tensor_bundle"]["estimator_input_contract"]["gradient_scale_restored"] = False
    _rebalance_hashes(evidence)
    with pytest.raises(Stage1EstimatorError, match="ESTIMATOR_INPUT_CONTRACT_BINDING_INVALID"):
        validate_stage1_s15_evidence(evidence, source_root=ROOT)

    metadata_mismatch = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="4" * 40))
    metadata_mismatch["estimator_report"]["estimator_input_contract"]["rank"] = 1
    metadata_mismatch["tensor_bundle"]["estimator_input_contract"]["rank"] = 1
    _rebalance_hashes(metadata_mismatch)
    with pytest.raises(Stage1EstimatorError, match="ESTIMATOR_INPUT_CONTRACT_BINDING_INVALID"):
        validate_stage1_s15_evidence(metadata_mismatch, source_root=ROOT)

    missing_rejection = copy.deepcopy(build_stage1_s15_evidence(ROOT, producer_commit="2" * 40))
    missing_rejection["estimator_report"]["profiles"][0]["rejections"].pop("m1_rejected")
    _rebalance_hashes(missing_rejection)
    with pytest.raises(Stage1EstimatorError, match="REJECTION_CONTRACT_INVALID"):
        validate_stage1_s15_evidence(missing_rejection, source_root=ROOT)


def test_s15_raw_and_double_reject_fp16_and_the_frozen_device_mismatch_control() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="3" * 40)
    profile = evidence["tensor_bundle"]["profiles"][0]
    first = tensor_map_from_wire(profile["microbatch_samples"][0], field="first", coordinate_order=("bias", "head", "weight"), registry=_fixture_registry())
    second = tensor_map_from_wire(profile["microbatch_samples"][1], field="second", coordinate_order=("bias", "head", "weight"), registry=_fixture_registry())
    fp16 = first.to(dtype=torch.float16)
    with pytest.raises(Exception):
        raw_importance(fp16)
    with pytest.raises(Exception):
        double_sample_importance(fp16, second)
    assert profile["profile"] == "T64_ORACLE"
    assert evidence["estimator_report"]["profiles"][0]["rejections"]["dtype_device_mismatch_rejected"] is True
