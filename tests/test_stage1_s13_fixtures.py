from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.stage1_fixtures import (
    Stage1FixtureError,
    _bundle_gradient_input_contract,
    build_stage1_s13_evidence,
    validate_stage1_s13_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def _rebind_bundle_and_report(evidence: dict[str, object]) -> None:
    bundle = evidence["oracle_bundle"]
    assert isinstance(bundle, dict)
    bundle_body = dict(bundle)
    bundle_body.pop("bundle_hash")
    bundle["bundle_hash"] = canonical_json_hash(bundle_body)
    report = evidence["oracle_validation_report"]
    assert isinstance(report, dict)
    report["oracle_bundle_hash"] = bundle["bundle_hash"]
    report_body = dict(report)
    report_body.pop("report_hash")
    report["report_hash"] = canonical_json_hash(report_body)


def test_s13_manifest_is_canonical_and_hash_bound() -> None:
    legacy = load_canonical_json(ROOT / "fixtures/stage1/stage1-s13-v1.json")
    assert isinstance(legacy, dict)
    assert legacy["fixture_set_id"] == "stage1-s13-deterministic-v1"
    path = ROOT / "fixtures/stage1/stage1-s13-v2.json"
    value = load_canonical_json(path)
    assert isinstance(value, dict)
    body = dict(value)
    declared = body.pop("manifest_hash")
    assert declared == canonical_json_hash(body)
    assert value["fixture_set_id"] == "stage1-s13-deterministic-v2"
    assert value["schema_version"] == "stage1-fixture-manifest-v2"
    assert value["pythia_14m"]["consumed_by_gate"] is False
    double = value["gradient_matrix"]["independent_double_sample"]
    assert double["samples_a"] != double["samples_b"]
    assert double["source_a"] != double["source_b"]
    assert double["source_a"]["stream_id"] != double["source_b"]["stream_id"]
    assert double["natural_gradient_scale"] > 0.0


def test_s13_builds_three_bound_roles_and_replays_offline(tmp_path: Path) -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="a" * 40)
    report = evidence["oracle_validation_report"]
    assert report["status"] == "PASS"
    assert report["check_count"] == report["passed_check_count"]
    assert report["pythia_14m"]["status"] == "DEFERRED_TO_S1.7"
    assert set(evidence) == {
        "fixture_manifest",
        "oracle_bundle",
        "oracle_validation_report",
    }
    assert evidence["fixture_manifest"]["manifest_hash"] == evidence["oracle_bundle"]["fixture_manifest_hash"]
    assert evidence["fixture_manifest"]["fixture_set_id"] == "stage1-s13-deterministic-v2"
    assert evidence["oracle_bundle"]["schema_version"] == "stage1-oracle-bundle-v2"
    assert evidence["oracle_bundle"]["bundle_hash"] == report["oracle_bundle_hash"]
    matrix = evidence["oracle_bundle"]["gradient_matrix"]
    assert "double_sample_oracle" in matrix["independent_double_sample"]
    assert {"raw_oracle", "equal_u_oracle"}.issubset(matrix["cases"]["equal-m3"])
    assert "weighted_u_oracle" in matrix["cases"]["varied-m4"]
    assert matrix["independent_double_sample"]["source_a"] != matrix["independent_double_sample"]["source_b"]

    replay = validate_stage1_s13_evidence(evidence)
    assert replay["report_status"] == "PASS"
    assert replay["replay"]["source_bundle_hash"] == evidence["oracle_bundle"]["bundle_hash"]
    assert replay["frozen_gradient_input_hash"] == evidence["oracle_bundle"]["frozen_gradient_input_hash"]
    assert replay["replay"]["recomputed_objects"] == [
        "raw_oracle",
        "double_sample_oracle",
        "equal_u_oracle",
        "weighted_u_oracle",
    ]
    assert replay["replay"]["gradient_matrix"]["independent_double_sample"]["comparisons"]["double_sample_oracle"]["passed"] is True

    serialized_evidence = {}
    for role, payload in evidence.items():
        path = tmp_path / f"{role}.json"
        write_canonical_json(path, payload)
        serialized_evidence[role] = load_canonical_json(path)
    serialized_replay = validate_stage1_s13_evidence(serialized_evidence)
    assert serialized_replay["replay"]["recomputed_objects"] == replay["replay"]["recomputed_objects"]
    assert serialized_replay["frozen_gradient_input_hash"] == evidence["oracle_bundle"]["frozen_gradient_input_hash"]


def test_s13_rejects_tampered_serialized_oracle() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="b" * 40)
    tampered = deepcopy(evidence)
    samples = tampered["oracle_bundle"]["gradient_matrix"]["cases"]["equal-m3"]["samples"]
    samples[0]["attention.bias"]["values"][0] += 1.0
    with pytest.raises(Stage1FixtureError, match="gradient inputs"):
        validate_stage1_s13_evidence(tampered)


def test_s13_replay_rejects_each_rehashed_estimator_oracle_tamper() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="c" * 40)
    oracle_paths = (
        ("cases", "equal-m3", "raw_oracle"),
        ("cases", "equal-m3", "equal_u_oracle"),
        ("cases", "varied-m4", "weighted_u_oracle"),
        ("independent_double_sample", "double_sample_oracle"),
    )
    for path in oracle_paths:
        tampered = deepcopy(evidence)
        target = tampered["oracle_bundle"]["gradient_matrix"]
        for item in path:
            target = target[item]
        target["attention.bias"]["values"][0] += 1.0

        _rebind_bundle_and_report(tampered)

        with pytest.raises(Stage1FixtureError, match="serialized replay"):
            validate_stage1_s13_evidence(tampered)


def test_s13_rejects_rehashed_joint_gradient_input_and_oracle_tamper() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="d" * 40)
    tampered = deepcopy(evidence)
    case = tampered["oracle_bundle"]["gradient_matrix"]["cases"]["equal-m3"]
    case["samples"][0]["attention.bias"]["values"][0] += 7.0
    case["raw_oracle"]["attention.bias"]["values"][0] += 7.0
    rehashed_inputs = canonical_json_hash(_bundle_gradient_input_contract(tampered["oracle_bundle"]))
    tampered["oracle_bundle"]["frozen_gradient_input_hash"] = rehashed_inputs
    tampered["oracle_validation_report"]["frozen_gradient_input_hash"] = rehashed_inputs
    _rebind_bundle_and_report(tampered)

    with pytest.raises(Stage1FixtureError, match="manifest frozen gradient inputs"):
        validate_stage1_s13_evidence(tampered)


def test_s13_rejects_rehashed_double_stream_identity_and_seed_drift() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="d" * 40)
    tampered = deepcopy(evidence)
    source_b = tampered["oracle_bundle"]["gradient_matrix"]["independent_double_sample"]["source_b"]
    source_b["stream_id"] = "stage1-s13-double-sample-b-tampered"
    source_b["seed"] += 1
    rehashed_inputs = canonical_json_hash(_bundle_gradient_input_contract(tampered["oracle_bundle"]))
    tampered["oracle_bundle"]["frozen_gradient_input_hash"] = rehashed_inputs
    tampered["oracle_validation_report"]["frozen_gradient_input_hash"] = rehashed_inputs
    _rebind_bundle_and_report(tampered)

    with pytest.raises(Stage1FixtureError, match="manifest frozen gradient inputs"):
        validate_stage1_s13_evidence(tampered)


def test_s13_rejects_rehashed_oracle_dtype_drift() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="e" * 40)
    tampered = deepcopy(evidence)
    tampered["oracle_bundle"]["gradient_matrix"]["cases"]["equal-m3"]["raw_oracle"]["attention.bias"]["dtype"] = "torch.float32"
    _rebind_bundle_and_report(tampered)

    with pytest.raises(Stage1FixtureError, match="dtype"):
        validate_stage1_s13_evidence(tampered)


def test_s13_oracle_module_does_not_import_production_estimators() -> None:
    source = (ROOT / "src/param_importance_nlp/stage1_fixtures.py").read_text(encoding="utf-8")
    assert "core.estimators" not in source


def test_s13_schemas_are_strict_at_the_role_boundary() -> None:
    expected = {
        "fixture-manifest-v1.json": "stage1-fixture-manifest-v1",
        "oracle-bundle-v1.json": "stage1-oracle-bundle-v1",
        "oracle-validation-report-v1.json": "stage1-oracle-validation-report-v1",
    }
    for filename, schema_version in expected.items():
        value = json.loads((ROOT / "schemas/stage1" / filename).read_text(encoding="utf-8"))
        assert value["type"] == "object"
        assert value["additionalProperties"] is False
        assert value["properties"]["schema_version"]["const"] == schema_version

    fixture = json.loads((ROOT / "schemas/stage1/fixture-manifest-v1.json").read_text(encoding="utf-8"))
    assert fixture["properties"]["fixture_set_id"]["const"] == "stage1-s13-deterministic-v1"
    assert fixture["properties"]["gradient_matrix"] == {"$ref": "#/$defs/object_section"}

    fixture = json.loads((ROOT / "schemas/stage1/fixture-manifest-v2.json").read_text(encoding="utf-8"))
    fixture_matrix = fixture["$defs"]["gradient_matrix"]
    assert "independent_double_sample" in fixture_matrix["required"]
    assert fixture_matrix["additionalProperties"] is False
    oracle = json.loads((ROOT / "schemas/stage1/oracle-bundle-v2.json").read_text(encoding="utf-8"))
    oracle_matrix = oracle["$defs"]["gradient_matrix"]
    assert "independent_double_sample" in oracle_matrix["required"]
    assert oracle_matrix["additionalProperties"] is False
    assert oracle["$defs"]["offline_recompute"]["properties"]["objects"]["const"] == [
        "raw_oracle",
        "double_sample_oracle",
        "equal_u_oracle",
        "weighted_u_oracle",
    ]
    assert oracle["$defs"]["tensor_wire"]["additionalProperties"] is False
