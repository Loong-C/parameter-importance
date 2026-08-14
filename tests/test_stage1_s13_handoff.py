from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from param_importance_nlp.atomic import sha256_file
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json


ROOT = Path(__file__).resolve().parents[1]
_FORMALIZER_PATH = ROOT / "ops/stage1/formalize_s1_3.py"
_SPEC = importlib.util.spec_from_file_location("stage1_s13_formalizer_handoff_test", _FORMALIZER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FORMALIZER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FORMALIZER)


def _with_artifact_hash(value: dict[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["artifact_hash"] = canonical_json_hash(value)
    return body


def _write_handoff(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repository = tmp_path / "repository"
    data_root = tmp_path / "data"
    schema_ref = "schemas/stage1/s1-2-config-field-behavior-coverage-v2.json"
    manifest_ref = "configs/stage1/s1-2-config-field-behavior-coverage-v2.json"
    schema_path = repository / schema_ref
    schema_path.parent.mkdir(parents=True)
    write_canonical_json(schema_path, {"schema": "coverage-v2"})
    schema_sha256 = sha256_file(schema_path)
    shared_v1_path = repository / "schemas/shared/resolved-config-v1.json"
    shared_v2_path = repository / "schemas/shared/resolved-config-v2.json"
    shared_v1_path.parent.mkdir(parents=True)
    write_canonical_json(shared_v1_path, {"schema": "resolved-config-v1"})
    write_canonical_json(shared_v2_path, {"schema": "resolved-config-v2"})
    shared_v1_hash = sha256_file(shared_v1_path)
    shared_v2_hash = sha256_file(shared_v2_path)
    contracts = {
        "resolved-config-v1": {
            "schema_version": "resolved-config-v1",
            "schema_hash": "1" * 64,
            "shared_schema_hashes": {"resolved-config-v1": shared_v1_hash},
        },
        "resolved-config-v2": {
            "schema_version": "resolved-config-v2",
            "schema_hash": "2" * 64,
            "shared_schema_hashes": {"resolved-config-v1": shared_v1_hash, "resolved-config-v2": shared_v2_hash},
        },
    }
    manifest = _with_artifact_hash(
        {
            "schema_version": "stage1-config-field-behavior-coverage-v2",
            "schema_sha256": schema_sha256,
            "config_contracts": contracts,
            "coverage_groups": [
                {
                    "config_family": "resolved-config-v1",
                    "field_paths": ["identity.schema_version"],
                    "behavior_id": "frozen_rejection_guard",
                    "test_id": "tests.test_stage1_s12_config_coverage::test_v1_public_fields_have_executable_alternate_or_guard_behavior",
                }
            ],
        }
    )
    manifest_path = repository / manifest_ref
    manifest_path.parent.mkdir(parents=True)
    write_canonical_json(manifest_path, manifest)

    attempt = data_root / "evidence" / "stage1" / "s1-2-formal" / ("a" * 40) / "v2-test"
    attempt.mkdir(parents=True)
    coverage = {
        "schema_version": "stage1-config-field-behavior-coverage-v2",
        "artifact_hash": manifest["artifact_hash"],
        "config_contracts": contracts,
        "covered_field_counts": {"resolved-config-v1": 1, "resolved-config-v2": 1},
        "behavior_counts": {"frozen_rejection_guard": 1},
        "full_identity_only_v1_paths": [],
        "scope": "compiled_component_behavior_or_frozen_guard",
        "manifest_ref": manifest_ref,
        "manifest_sha256": sha256_file(manifest_path),
        "schema_ref": schema_ref,
        "schema_sha256": schema_sha256,
    }
    report = {
        "schema_version": "g1-registry-report-v2",
        "gate_id": "G1-REGISTRY",
        "status": "PASS",
        "execution_scope": "formal_server_cpu",
        "registry": {"coordinate_registry_hash": "3" * 64, "optimizer_contract_hash": "4" * 64, "runtime_layout_hash": "5" * 64, "record_count": 1, "eligible_numel": 1},
        "checks": [],
        "producer_commit": "a" * 40,
        "consumer_commit": "a" * 40,
        "buffer_policy": "excluded_from_parameter_registry-v1",
        "config_field_behavior_coverage": coverage,
    }
    report_path = attempt / "g1-registry-report.json"
    write_canonical_json(report_path, report)
    index = _with_artifact_hash(
        {
            "schema_version": "stage1-s1-2-formalization-index-v2",
            "status": "PASS",
            "gate_id": "G1-REGISTRY",
            "task_id": "stage1.02_architecture_and_parameter_registry",
            "generator_git_commit": "a" * 40,
            "git_branch": "main",
            "checked_at": "2026-08-14T00:00:00Z",
            "s1_1_index_ref": "evidence/stage1/s1-1-formal/index.json",
            "s1_1_index_sha256": "6" * 64,
            "s1_1_gate_artifact_hashes": {"stage1.G1-ENTRY": "7" * 64, "stage1.G1-CONTRACT": "8" * 64},
            "validation_ref": "validation.json",
            "validation_sha256": "9" * 64,
            "report_ref": "g1-registry-report.json",
            "report_sha256": sha256_file(report_path),
            "config_field_behavior_coverage_manifest_ref": manifest_ref,
            "config_field_behavior_coverage_manifest_sha256": sha256_file(manifest_path),
            "config_field_behavior_coverage_artifact_hash": manifest["artifact_hash"],
            "config_field_behavior_coverage_schema_ref": schema_ref,
            "config_field_behavior_coverage_schema_sha256": schema_sha256,
            "config_contract_hashes": {family: contract["schema_hash"] for family, contract in contracts.items()},
            "config_shared_schema_hashes": {family: contract["shared_schema_hashes"] for family, contract in contracts.items()},
            "probe_summary": {},
            "next_task_id": "stage1.03_fixtures_and_oracles",
        }
    )
    index_path = attempt / "index.json"
    write_canonical_json(index_path, index)
    return repository, data_root, index_path, report_path


def _rewrite_index(index_path: Path, mutate: callable) -> None:
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    index = load_canonical_json(index_path)
    assert isinstance(index, dict)
    index.pop("artifact_hash")
    mutate(index)
    write_canonical_json(index_path, _with_artifact_hash(index))


def test_s13_accepts_hash_closed_s12_v2_handoff(tmp_path: Path) -> None:
    repository, data_root, index_path, _ = _write_handoff(tmp_path)
    handoff = _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())
    assert handoff["schema_version"] == "stage1-s1-2-formalization-index-v2"
    assert handoff["_upstream_evidence"]["s1_2_report_sha256"] == handoff["report_sha256"]


def test_s13_rejects_s12_v1_handoff_even_when_rehashed(tmp_path: Path) -> None:
    repository, data_root, index_path, _ = _write_handoff(tmp_path)
    _rewrite_index(index_path, lambda index: index.__setitem__("schema_version", "stage1-s1-2-formalization-index-v1"))
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="HANDOFF_NOT_READY"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())


def test_s13_rejects_s12_non_v2_report_even_when_rehashed(tmp_path: Path) -> None:
    repository, data_root, index_path, report_path = _write_handoff(tmp_path)
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    report = load_canonical_json(report_path)
    assert isinstance(report, dict)
    report["schema_version"] = "g1-registry-report-v1"
    write_canonical_json(report_path, report)
    _rewrite_index(index_path, lambda index: index.__setitem__("report_sha256", sha256_file(report_path)))
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="REPORT_NOT_PASS"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())


def test_s13_rejects_s12_buffer_policy_and_contract_drift(tmp_path: Path) -> None:
    repository, data_root, index_path, report_path = _write_handoff(tmp_path)
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    report = load_canonical_json(report_path)
    assert isinstance(report, dict)
    report["buffer_policy"] = "included"
    write_canonical_json(report_path, report)
    _rewrite_index(index_path, lambda index: index.__setitem__("report_sha256", sha256_file(report_path)))
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="BUFFER_POLICY_INVALID"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())

    repository, data_root, index_path, report_path = _write_handoff(tmp_path / "contract")
    report = load_canonical_json(report_path)
    assert isinstance(report, dict)
    report["config_field_behavior_coverage"]["config_contracts"]["resolved-config-v2"]["schema_hash"] = "f" * 64
    write_canonical_json(report_path, report)
    _rewrite_index(index_path, lambda index: index.__setitem__("report_sha256", sha256_file(report_path)))
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="CONFIG_CONTRACT_MISMATCH:resolved-config-v2"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())


def test_s13_rejects_s12_shared_schema_hash_drift(tmp_path: Path) -> None:
    repository, data_root, index_path, report_path = _write_handoff(tmp_path)
    from param_importance_nlp.contracts.jsonio import load_canonical_json

    report = load_canonical_json(report_path)
    assert isinstance(report, dict)
    report["config_field_behavior_coverage"]["config_contracts"]["resolved-config-v2"]["shared_schema_hashes"]["resolved-config-v2"] = "f" * 64
    write_canonical_json(report_path, report)
    _rewrite_index(index_path, lambda index: index.__setitem__("report_sha256", sha256_file(report_path)))
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="SHARED_SCHEMA_MISMATCH:resolved-config-v2:resolved-config-v2"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())


def test_s13_rejects_s12_shared_schema_file_tamper(tmp_path: Path) -> None:
    repository, data_root, index_path, _ = _write_handoff(tmp_path)
    shared_schema_path = repository / "schemas/shared/resolved-config-v1.json"
    write_canonical_json(shared_schema_path, {"schema": "tampered"})
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="SHARED_SCHEMA_FILE_HASH_MISMATCH:resolved-config-v1"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())


def test_s13_rejects_s12_coverage_file_hash_drift(tmp_path: Path) -> None:
    repository, data_root, index_path, _ = _write_handoff(tmp_path)
    manifest_path = repository / "configs/stage1/s1-2-config-field-behavior-coverage-v2.json"
    write_canonical_json(manifest_path, {"replacement": "must fail closed"})
    with pytest.raises(_FORMALIZER.Stage1S13FormalError, match="COVERAGE_MANIFEST_SHA256_MISMATCH"):
        _FORMALIZER._load_s1_2_handoff(repository, data_root, index_path.relative_to(data_root).as_posix())
