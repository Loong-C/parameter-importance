from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib

import pytest

from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
)

from param_importance_nlp.experiments.stage3_protocol import (
    CandidateRules,
    DataPartitionProtocol,
    DEFAULT_THRESHOLDS,
    EndpointSpec,
    FormalFreeze,
    FrozenThresholds,
    ReferenceRefinementLadder,
    build_formal_matrix,
    build_pilot_matrix,
    build_stage3_protocol,
    freeze_formal_protocol,
    matrix_coverage,
    validate_global_data_partition,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _formal_evidence(*, gate_count: int = 6) -> FormalExecutionEvidence:
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_digest("contract-freeze"),
        asset_manifest_hashes=(_digest("model-assets"), _digest("data-assets")),
        prerequisite_gates=tuple(
            GateRecord(
                gate_id=f"stage3.G3-{index}",
                stage=3,
                status=GateStatus.PASS,
                checked_at="2026-08-28T00:00:00+00:00",
                evidence_refs=(f"evidence/g3-{index}.json",),
            )
            for index in range(gate_count)
        ),
    )


def test_default_pilot_is_immutable_and_has_at_least_twelve_units() -> None:
    matrix = build_pilot_matrix()
    assert matrix.scope == "pilot"
    assert matrix.endpoint_count == 6
    assert matrix.endpoint_probe_unit_count == 12
    assert matrix.coverage["by_model"] == {"14M": 6}
    assert matrix.formal_eligible is False
    assert hash(matrix) == hash(build_pilot_matrix())
    assert matrix.artifact_hash == build_pilot_matrix().artifact_hash
    with pytest.raises(FrozenInstanceError):
        matrix.matrix_id = "changed"  # type: ignore[misc]


def test_formal_matrix_requires_an_explicit_pre_formal_freeze() -> None:
    with pytest.raises(ValueError, match="先提供 FormalFreeze"):
        build_formal_matrix()

    pilot = build_pilot_matrix()
    freeze = freeze_formal_protocol(pilot)
    with pytest.raises(ValueError, match="FormalExecutionEvidence"):
        build_formal_matrix(freeze)
    formal = build_formal_matrix(
        freeze, formal_execution_evidence=_formal_evidence()
    )
    assert formal.endpoint_count == 33
    assert formal.endpoint_probe_unit_count == 99
    assert formal.coverage["by_model"] == {"14M": 24, "31M": 9}
    assert len({endpoint.seed for endpoint in formal.endpoints if endpoint.model_size == "14M"}) == 2
    assert len({endpoint.seed for endpoint in formal.endpoints if endpoint.model_size == "31M"}) == 1
    assert formal.formal_eligible is True
    assert formal.formal_freeze == freeze
    assert all("fixture" not in unit.casefold() for unit in formal.unit_ids)
    assert all("synthetic" not in unit.casefold() for unit in formal.unit_ids)


def test_formal_freeze_binds_rules_reference_families_and_thresholds() -> None:
    rules = CandidateRules(("left", "trapezoid"))
    ladder = ReferenceRefinementLadder((2, 4, 8), (2, 4, 8))
    thresholds = FrozenThresholds(dict(DEFAULT_THRESHOLDS))
    freeze = FormalFreeze(rules, ladder, thresholds)
    assert freeze.candidate_rules.frozen is True
    assert freeze.reference_ladder.frozen is True
    assert freeze.thresholds.frozen is True
    assert freeze.freeze_id == FormalFreeze(rules, ladder, thresholds).freeze_id
    assert freeze.candidate_rules_hash == freeze.candidate_rules.artifact_hash
    assert freeze.reference_ladder.family_names == (
        "gauss_legendre",
        "composite_simpson",
    )

    pilot = build_pilot_matrix(
        candidate_rules=rules,
        reference_ladder=ladder,
        thresholds=thresholds,
    )
    assert freeze_formal_protocol(pilot).artifact_hash == freeze.artifact_hash


def test_formal_rejects_fixture_or_synthetic_labels_and_wrong_coverage() -> None:
    freeze = freeze_formal_protocol()
    evidence = _formal_evidence()
    endpoints = list(
        build_formal_matrix(freeze, formal_execution_evidence=evidence).endpoints
    )
    endpoints[0] = EndpointSpec(
        "14M", 0, "early", "fixture-endpoint", endpoints[0].probe_ids
    )
    with pytest.raises(ValueError, match="fixture/synthetic"):
        build_formal_matrix(
            freeze, endpoints=endpoints, formal_execution_evidence=evidence
        )

    too_few_probes = list(
        build_formal_matrix(freeze, formal_execution_evidence=evidence).endpoints
    )
    too_few_probes[0] = EndpointSpec(
        "14M", 0, "early", too_few_probes[0].endpoint_id, ("p1", "p2")
    )
    with pytest.raises(ValueError, match="至少需要三个 probe|精确为 99"):
        build_formal_matrix(
            freeze,
            endpoints=too_few_probes,
            formal_execution_evidence=evidence,
        )

    with pytest.raises(ValueError, match="缺少前置 Gate"):
        build_formal_matrix(
            freeze, formal_execution_evidence=_formal_evidence(gate_count=5)
        )


def test_data_partition_protocol_is_globally_disjoint_and_hashable() -> None:
    partition = validate_global_data_partition(
        update_sample_ids=["u2", "u1"],
        pilot_probe_sample_ids=["p1"],
        formal_probe_sample_ids=["f1"],
        replay_sample_ids=["r1"],
    )
    assert partition.update_sample_ids == ("u2", "u1")
    assert partition.all_sample_ids == ("f1", "p1", "r1", "u1", "u2")
    assert hash(partition) == hash(
        DataPartitionProtocol(("u2", "u1"), ("p1",), ("f1",), ("r1",))
    )
    with pytest.raises(ValueError, match="全局互斥"):
        DataPartitionProtocol(("same",), ("same",), (), ())
    with pytest.raises(ValueError, match="不能重复"):
        DataPartitionProtocol(("u", "u"), (), (), ())


def test_coverage_and_unit_ids_are_unique_and_order_canonical() -> None:
    freeze = freeze_formal_protocol()
    evidence = _formal_evidence()
    formal = build_formal_matrix(freeze, formal_execution_evidence=evidence)
    reversed_endpoints = tuple(reversed(formal.endpoints))
    reordered = build_formal_matrix(
        freeze,
        endpoints=reversed_endpoints,
        formal_execution_evidence=evidence,
    )
    assert formal.unit_ids == reordered.unit_ids
    assert formal.artifact_hash == reordered.artifact_hash
    report = matrix_coverage(formal)
    assert report["endpoint_probe_unit_count"] == 99
    assert len(report["unit_ids"]) == len(set(report["unit_ids"])) == 99


def test_complete_protocol_keeps_partitions_separate_from_matrix_hashes() -> None:
    protocol = build_stage3_protocol(formal_execution_evidence=_formal_evidence())
    assert protocol.coverage["pilot"]["unit_count"] == 12
    assert protocol.coverage["formal"]["unit_count"] == 99
    assert protocol.data_partitions.update_sample_ids
    assert set(protocol.data_partitions.all_sample_ids).isdisjoint(
        set(protocol.pilot_matrix.unit_ids)
    )
    assert protocol.artifact_hash == build_stage3_protocol(
        formal_execution_evidence=_formal_evidence()
    ).artifact_hash


def test_reference_ladder_and_threshold_inputs_fail_closed() -> None:
    with pytest.raises(ValueError, match="严格递增"):
        ReferenceRefinementLadder((4, 4), (2, 4))
    with pytest.raises(ValueError, match="至少需要"):
        ReferenceRefinementLadder((4,), (2, 4))
    with pytest.raises(ValueError, match="有限数"):
        FrozenThresholds({"bad": float("nan")})
    too_wide = dict(DEFAULT_THRESHOLDS)
    too_wide["max_normalized_l1_error"] = 0.02
    with pytest.raises(ValueError, match="原则上限"):
        FormalFreeze(CandidateRules(("left",)), thresholds=too_wide)
    wrong_node_type = dict(DEFAULT_THRESHOLDS)
    wrong_node_type["max_unique_nodes"] = 16.0
    with pytest.raises(ValueError, match="正整数"):
        FormalFreeze(CandidateRules(("left",)), thresholds=wrong_node_type)
