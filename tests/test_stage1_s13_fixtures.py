from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.stage1_fixtures import (
    Stage1FixtureError,
    build_stage1_s13_evidence,
    validate_stage1_s13_evidence,
)


ROOT = Path(__file__).resolve().parents[1]


def test_s13_manifest_is_canonical_and_hash_bound() -> None:
    path = ROOT / "fixtures/stage1/stage1-s13-v1.json"
    value = load_canonical_json(path)
    assert isinstance(value, dict)
    body = dict(value)
    declared = body.pop("manifest_hash")
    assert declared == canonical_json_hash(body)
    assert value["fixture_set_id"] == "stage1-s13-deterministic-v1"
    assert value["pythia_14m"]["consumed_by_gate"] is False


def test_s13_builds_three_bound_roles_and_replays_offline() -> None:
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
    assert evidence["oracle_bundle"]["bundle_hash"] == report["oracle_bundle_hash"]

    replay = validate_stage1_s13_evidence(evidence)
    assert replay["report_status"] == "PASS"
    assert replay["replay"]["source_bundle_hash"] == evidence["oracle_bundle"]["bundle_hash"]


def test_s13_rejects_tampered_serialized_oracle() -> None:
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="b" * 40)
    tampered = deepcopy(evidence)
    samples = tampered["oracle_bundle"]["gradient_matrix"]["cases"]["equal-m3"]["samples"]
    samples[0]["attention.bias"]["values"][0] += 1.0
    with pytest.raises(Stage1FixtureError, match="bundle hash"):
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
