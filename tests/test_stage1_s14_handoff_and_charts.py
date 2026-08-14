from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from xml.etree import ElementTree

import pytest

from param_importance_nlp.atomic import sha256_file
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.stage1_fixtures import build_stage1_s13_evidence, validate_stage1_s13_evidence
from param_importance_nlp.stage1_gradient_scale import build_stage1_s14_evidence


ROOT = Path(__file__).resolve().parents[1]
_FORMALIZER_PATH = ROOT / "ops/stage1/formalize_s1_4.py"
_SPEC = importlib.util.spec_from_file_location("stage1_s14_formalizer_test", _FORMALIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FORMALIZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORMALIZER)


def _with_artifact_hash(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["artifact_hash"] = canonical_json_hash(value)
    return result


def _write_s13_v2_handoff(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    attempt = data_root / "evidence/stage1/s1-3-formal" / ("7" * 40) / "v2"
    attempt.mkdir(parents=True)
    evidence = build_stage1_s13_evidence(ROOT, producer_commit="79f88d47c534840df950efac42481a317d66faba")
    role_paths: dict[str, Path] = {}
    for role, payload in evidence.items():
        path = attempt / {
            "fixture_manifest": "fixture-manifest.json",
            "oracle_bundle": "oracle-bundle.json",
            "oracle_validation_report": "oracle-validation-report.json",
        }[role]
        write_canonical_json(path, payload)
        role_paths[role] = path
    replay = validate_stage1_s13_evidence(evidence)
    replay_path = attempt / "oracle-replay-validation.json"
    write_canonical_json(replay_path, replay)
    validation = _with_artifact_hash({"status": "PASS", "fixture": "s13"})
    validation_path = attempt / "validation.json"
    write_canonical_json(validation_path, validation)
    index = _with_artifact_hash(
        {
            "schema_version": "stage1-s1-3-formalization-index-v2",
            "status": "PASS",
            "gate_id": "G1-ORACLE",
            "task_id": "stage1.03_fixtures_and_oracles",
            "generator_git_commit": "79f88d47c534840df950efac42481a317d66faba",
            "s1_2_index_sha256": _FORMALIZER.EXPECTED_S1_2_INDEX_SHA256,
            "next_task_id": "stage1.04_loss_and_gradient_scale",
            "fixture_manifest_ref": role_paths["fixture_manifest"].name,
            "fixture_manifest_sha256": sha256_file(role_paths["fixture_manifest"]),
            "oracle_bundle_ref": role_paths["oracle_bundle"].name,
            "oracle_bundle_sha256": sha256_file(role_paths["oracle_bundle"]),
            "oracle_validation_report_ref": role_paths["oracle_validation_report"].name,
            "oracle_validation_report_sha256": sha256_file(role_paths["oracle_validation_report"]),
            "oracle_bundle_hash": evidence["oracle_bundle"]["bundle_hash"],
            "frozen_gradient_input_hash": evidence["oracle_bundle"]["frozen_gradient_input_hash"],
            "replay_ref": replay_path.name,
            "replay_sha256": sha256_file(replay_path),
            "replay_hash": replay["replay_hash"],
            "validation_ref": validation_path.name,
            "validation_sha256": sha256_file(validation_path),
        }
    )
    index_path = attempt / "index.json"
    write_canonical_json(index_path, index)
    return data_root, index_path


def test_s14_requires_hash_closed_current_s13_v2_handoff(tmp_path: Path) -> None:
    data_root, index_path = _write_s13_v2_handoff(tmp_path)
    expected_sha = sha256_file(index_path)
    handoff = _FORMALIZER._load_s1_3_handoff(
        data_root,
        index_path.relative_to(data_root).as_posix(),
        expected_index_sha256=expected_sha,
    )
    assert handoff["s1_3_frozen_gradient_input_hash"]
    index_path.write_bytes(index_path.read_bytes().replace(b'"status":"PASS"', b'"status":"FAIL"', 1))
    with pytest.raises(_FORMALIZER.Stage1S14FormalError, match="INDEX_NOT_CURRENT_V2"):
        _FORMALIZER._load_s1_3_handoff(
            data_root,
            index_path.relative_to(data_root).as_posix(),
            expected_index_sha256=expected_sha,
        )


def test_s14_writes_all_four_csv_and_real_svg_chart_sources(tmp_path: Path) -> None:
    evidence = build_stage1_s14_evidence(ROOT, producer_commit="e" * 40)
    csv_sha, svg_sha = _FORMALIZER._write_chart_data(tmp_path, evidence)
    assert set(csv_sha) == {
        "tensor-errors.csv",
        "gradient-scatter.csv",
        "tensor-error-heatmap.csv",
        "accumulation-errors.csv",
    }
    assert set(svg_sha) == {
        "gradient-scatter.svg",
        "route-errors.svg",
        "tensor-error-heatmap.svg",
        "accumulation-errors.svg",
    }
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    scatter = ElementTree.parse(tmp_path / "gradient-scatter.svg").getroot()
    assert scatter.attrib["data-source"] == "gradient-scatter.csv"
    assert scatter.find(".//svg:line[@id='y-equals-x']", namespace) is not None
    assert scatter.findall(".//svg:line[@class='axis axis-x']", namespace)
    assert scatter.findall(".//svg:circle[@class='scatter-point']", namespace)
    route = ElementTree.parse(tmp_path / "route-errors.svg").getroot()
    assert route.attrib["data-source"] == "tensor-errors.csv"
    assert {mark.attrib["data-route"] for mark in route.findall(".//svg:circle[@class='route-mark']", namespace)} == {
        "equal_microbatch_m2_gradient",
        "token_weighted_microbatch_gradient",
        "negative_control_unweighted_unequal_microbatch_gradient",
    }
    heatmap = ElementTree.parse(tmp_path / "tensor-error-heatmap.svg").getroot()
    assert heatmap.attrib["data-source"] == "tensor-error-heatmap.csv"
    assert heatmap.findall(".//svg:rect[@class='heatmap-cell']", namespace)
    assert heatmap.find(".//svg:rect[@id='heatmap-color-scale']", namespace) is not None
    accumulation = ElementTree.parse(tmp_path / "accumulation-errors.svg").getroot()
    assert accumulation.attrib["data-source"] == "accumulation-errors.csv"
    assert accumulation.findall(".//svg:polyline[@class='accumulation-curve']", namespace)
    assert accumulation.findall(".//svg:circle[@class='accumulation-point']", namespace)


def test_s14_publish_returns_live_index_path_and_regression_includes_chart_tests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / "attempt.publishing"
    staging.mkdir()
    artifact_hash = "a" * 64
    write_canonical_json(staging / "index.json", {"artifact_hash": artifact_hash})
    published = _FORMALIZER._publish_staging_directory(
        staging, tmp_path / "attempt", expected_artifact_hash=artifact_hash
    )
    assert published == tmp_path / "attempt" / "index.json"
    assert published.is_file()
    assert not staging.exists()

    captured: dict[str, Any] = {}

    def fake_run(command: list[str], **_: Any) -> SimpleNamespace:
        captured["command"] = command
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(_FORMALIZER.subprocess, "run", fake_run)
    _FORMALIZER._run_regression(ROOT, tmp_path, 1)
    assert "tests/test_stage1_s14_handoff_and_charts.py" in captured["command"]


def test_s14_formalizer_roundtrip_runs_real_offline_replay() -> None:
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
    evidence = build_stage1_s14_evidence(
        ROOT, producer_commit="f" * 40, scope="formal", upstream_evidence=upstream
    )
    checks, replay = _FORMALIZER._direct_checks(ROOT, evidence, upstream)
    assert all(check["status"] == "PASS" for check in checks)
    assert replay["status"] == "PASS"
    assert len(replay["comparison_hashes"]) == 56
