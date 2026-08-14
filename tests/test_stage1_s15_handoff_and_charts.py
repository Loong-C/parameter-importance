from __future__ import annotations

import csv
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import pytest

from param_importance_nlp.stage1_estimators import _build_table, _derive_gate_requirements, _hash_role, build_stage1_s15_evidence
from param_importance_nlp.cli import _validate_project_json_schema
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.stage1_gradient_scale import build_stage1_s14_evidence


ROOT = Path(__file__).resolve().parents[1]
_FORMALIZER_PATH = ROOT / "ops/stage1/formalize_s1_5.py"
_SPEC = importlib.util.spec_from_file_location("stage1_s15_formalizer_test", _FORMALIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FORMALIZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORMALIZER)


def _formal_upstream() -> dict[str, str]:
    return {
        "s1_4_index_ref": "evidence/stage1/s1-4-formal/92e3fa5ec286afa43c51be691895f9a7210199ff/formal-20260814-s14-v1-r1/index.json",
        "s1_4_index_sha256": _FORMALIZER.EXPECTED_S1_4_INDEX_SHA256,
        "s1_4_index_artifact_hash": "1" * 64,
        "s1_4_gate_artifact_hash": _FORMALIZER.EXPECTED_S1_4_GATE_ARTIFACT_HASH,
        "s1_4_gradient_scale_report_sha256": "2" * 64,
        "s1_4_comparison_table_sha256": "3" * 64,
        "s1_4_gate_record_sha256": "4" * 64,
        "s1_4_replay_sha256": "5" * 64,
        "s1_4_validation_sha256": "6" * 64,
    }


def _rehash_report_projection_and_gate(evidence: dict[str, Any]) -> None:
    report = evidence["estimator_report"]
    gate = evidence["gate_record"]
    assert isinstance(report, dict) and isinstance(gate, dict)
    report["report_hash"] = _hash_role(report, field="report_hash")
    rebuilt = _build_table(report)
    rebuilt["table_hash"] = _hash_role(rebuilt, field="table_hash")
    evidence["comparison_table"] = rebuilt
    gate["report_hash"] = report["report_hash"]
    gate["comparison_table_hash"] = rebuilt["table_hash"]
    gate["requirements"] = _derive_gate_requirements(report, rebuilt)
    gate["artifact_hash"] = _hash_role(gate, field="artifact_hash")


def test_s15_local_fixture_does_not_claim_the_formal_gate() -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="b" * 40)
    assert evidence["estimator_report"]["gate_status"] == "NOT_RUN"


def test_s15_charts_are_exact_table_projections(tmp_path: Path) -> None:
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="c" * 40)
    csv_sha, svg_sha = _FORMALIZER.write_chart_data(tmp_path, evidence)
    assert set(csv_sha) == {"tensor-errors.csv", "u-identity-scatter.csv", "scaling.csv", "negative-u.csv"}
    assert set(svg_sha) == {"u-identity-scatter.svg", "tensor-error-heatmap.svg", "scaling.svg", "negative-u.svg"}
    table = evidence["comparison_table"]
    with (tmp_path / "scaling.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(table["scaling_rows"])
    assert {row["parameter_group"] for row in rows} == {"group_0000", "group_0001"}
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    scatter = ElementTree.parse(tmp_path / "u-identity-scatter.svg").getroot()
    assert scatter.attrib["data-source"] == "u-identity-scatter.csv"
    assert scatter.findall(".//svg:line[@class='identity']", namespace)
    assert len(scatter.findall(".//svg:circle[@class='scatter-point']", namespace)) == len(table["scatter_rows"])
    assert {row["route_id"] for row in table["scatter_rows"]} == {
        "equal_u_ordered_vs_oracle",
        "equal_u_unordered_vs_oracle",
        "equal_u_streaming_vs_oracle",
    }
    assert {
        node.attrib["data-route"]
        for node in scatter.findall(".//svg:circle[@class='scatter-point']", namespace)
    } == {row["route_id"] for row in table["scatter_rows"]}
    heatmap = ElementTree.parse(tmp_path / "tensor-error-heatmap.svg").getroot()
    assert heatmap.attrib["data-source"] == "tensor-errors.csv"
    assert len(heatmap.findall(".//svg:rect[@class='heatmap-cell']", namespace)) == len(table["rows"])
    negative = ElementTree.parse(tmp_path / "negative-u.svg").getroot()
    assert len(negative.findall(".//svg:text[@class='negative-u-row']", namespace)) == len(table["negative_u_rows"])
    scaling_svg = ElementTree.parse(tmp_path / "scaling.svg").getroot()
    for class_name in ("raw-core-bar", "raw-score-bar", "raw-clipped-bar", "u-core-bar", "u-score-bar"):
        assert len(scaling_svg.findall(f".//svg:rect[@class='{class_name}']", namespace)) == len(table["scaling_rows"])


def test_s15_formalizer_direct_checks_reread_sources_and_real_replay() -> None:
    upstream = _formal_upstream()
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="d" * 40, scope="formal", upstream_evidence=upstream)
    checks, replay = _FORMALIZER._direct_checks(evidence, upstream, repository_root=ROOT)
    assert all(check["status"] == "PASS" for check in checks)
    assert {check["check_id"] for check in checks} >= {"implementation_source_binding", "offline_replay", "s1_4_handoff_closed"}
    assert replay["status"] == "PASS"


def _rehash_s14_roles(evidence: dict[str, dict[str, Any]]) -> None:
    report = evidence["gradient_scale_report"]
    table = evidence["comparison_table"]
    gate = evidence["gate_record"]
    report_body = dict(report)
    report_body.pop("report_hash", None)
    report["report_hash"] = canonical_json_hash(report_body)
    table["report_hash"] = report["report_hash"]
    table_body = dict(table)
    table_body.pop("table_hash", None)
    table["table_hash"] = canonical_json_hash(table_body)
    gate["report_hash"] = report["report_hash"]
    gate["comparison_table_hash"] = table["table_hash"]
    gate_body = dict(gate)
    gate_body.pop("artifact_hash", None)
    gate["artifact_hash"] = canonical_json_hash(gate_body)


def test_s15_s14_consumer_rebuild_allows_only_downstream_shared_source_drift() -> None:
    current = build_stage1_s14_evidence(ROOT, producer_commit=_FORMALIZER.EXPECTED_S1_4_PRODUCER_COMMIT)
    historical = copy.deepcopy(current)
    historical["gradient_scale_report"]["implementation_source_sha256"]["src/param_importance_nlp/core/__init__.py"] = "0" * 64
    historical["gradient_scale_report"]["implementation_source_sha256"]["src/param_importance_nlp/experiments/stage01_task_runners.py"] = "1" * 64
    _rehash_s14_roles(historical)
    drift = _FORMALIZER._assert_s1_4_consumer_compatibility(
        historical["gradient_scale_report"],
        historical["comparison_table"],
        historical["gate_record"],
        repository_root=ROOT,
    )
    assert drift == {
        "src/param_importance_nlp/core/__init__.py",
        "src/param_importance_nlp/experiments/stage01_task_runners.py",
    }

    unauthorized = copy.deepcopy(current)
    unauthorized["gradient_scale_report"]["implementation_source_sha256"]["src/param_importance_nlp/core/losses.py"] = "2" * 64
    _rehash_s14_roles(unauthorized)
    with pytest.raises(_FORMALIZER.Stage1S15FormalError, match="UNAUTHORIZED_SOURCE_DRIFT"):
        _FORMALIZER._assert_s1_4_consumer_compatibility(
            unauthorized["gradient_scale_report"],
            unauthorized["comparison_table"],
            unauthorized["gate_record"],
            repository_root=ROOT,
        )


def test_s15_formalizer_rejects_rehashed_source_digest_drift() -> None:
    upstream = _formal_upstream()
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="e" * 40, scope="formal", upstream_evidence=upstream)
    report = evidence["estimator_report"]
    report["implementation_source_sha256"]["src/param_importance_nlp/experiments/stage01_task_runners.py"] = "0" * 64
    _rehash_report_projection_and_gate(evidence)
    with pytest.raises(_FORMALIZER.Stage1S15FormalError, match="implementation_source_binding"):
        _FORMALIZER._direct_checks(evidence, upstream, repository_root=ROOT)


@pytest.mark.parametrize("mutation", [
    lambda hashes: hashes.__setitem__("unknown-source.py", "0" * 64),
    lambda hashes: hashes.pop("src/param_importance_nlp/stage1_estimators.py"),
])
def test_s15_formalizer_rejects_unknown_or_missing_rehashed_source_map(mutation: Any) -> None:
    upstream = _formal_upstream()
    evidence = build_stage1_s15_evidence(ROOT, producer_commit="9" * 40, scope="formal", upstream_evidence=upstream)
    mutation(evidence["estimator_report"]["implementation_source_sha256"])
    _rehash_report_projection_and_gate(evidence)
    with pytest.raises(_FORMALIZER.Stage1S15FormalError, match="implementation_source_binding"):
        _FORMALIZER._direct_checks(evidence, upstream, repository_root=ROOT)


def test_s15_regression_includes_its_chart_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_FORMALIZER.subprocess, "run", fake_run)
    _FORMALIZER._run_regression(ROOT, tmp_path, 1)
    assert "tests/test_stage1_s15_estimators.py" in captured["command"]
    assert "tests/test_stage1_s15_handoff_and_charts.py" in captured["command"]
    assert "tests/test_stage01_task_runners.py" in captured["command"]


def test_s15_all_eight_schemas_are_project_valid_and_source_hashes_are_closed() -> None:
    names = (
        "g1-est-report-v1.json", "g1-est-oracle-report-v1.json", "g1-est-tensor-bundle-v1.json",
        "g1-est-comparison-table-v1.json", "g1-est-gate-record-v1.json",
        "s1-5-formalization-index-v1.json", "s1-5-validation-v1.json", "s1-5-fixture-manifest-v1.json",
    )
    for name in names:
        schema = json.loads((ROOT / "schemas/stage1" / name).read_text(encoding="utf-8"))
        _validate_project_json_schema(schema)
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["$id"].endswith(name)
        assert schema["title"].strip()
    report_schema = json.loads((ROOT / "schemas/stage1/g1-est-report-v1.json").read_text(encoding="utf-8"))
    source_hashes = report_schema["$defs"]["sourceHashes"]
    assert source_hashes["additionalProperties"] is False
    assert set(source_hashes["required"]) == set(source_hashes["properties"])
    assert "src/param_importance_nlp/experiments/stage01_task_runners.py" in source_hashes["required"]
    assert "tests/test_stage01_task_runners.py" in source_hashes["required"]
    rejections = report_schema["$defs"]["rejections"]
    assert rejections["additionalProperties"] is False
    assert len(rejections["required"]) == 14
    assert set(rejections["required"]) == set(rejections["properties"])
    assert len(report_schema["$defs"]["comparison"]["properties"]["comparison_id"]["enum"]) == 20
    assert report_schema["$defs"]["profile"]["properties"]["comparisons"]["minItems"] == 20
    assert report_schema["$defs"]["profile"]["properties"]["comparisons"]["maxItems"] == 20
    assert set(report_schema["$defs"]["requirementChecks"]["required"]) >= {
        "identical_gradient_degeneracy",
        "paired_permutation_invariance",
        "estimator_public_field_bindings",
    }
    table_schema = json.loads((ROOT / "schemas/stage1/g1-est-comparison-table-v1.json").read_text(encoding="utf-8"))
    assert set(table_schema["$defs"]["scatterRow"]["properties"]["route_id"]["enum"]) == {
        "equal_u_ordered_vs_oracle",
        "equal_u_unordered_vs_oracle",
        "equal_u_streaming_vs_oracle",
    }
    fixture_schema = json.loads((ROOT / "schemas/stage1/s1-5-fixture-manifest-v1.json").read_text(encoding="utf-8"))
    assert fixture_schema["$defs"]["wireMap"]["properties"]["bias"]["allOf"][1]["properties"]["shape"]["const"] == [2]
    assert fixture_schema["$defs"]["wireMap"]["properties"]["head"]["allOf"][1]["properties"]["shape"]["const"] == [1]
    assert fixture_schema["$defs"]["wireMap"]["properties"]["weight"]["allOf"][1]["properties"]["shape"]["const"] == [3]
