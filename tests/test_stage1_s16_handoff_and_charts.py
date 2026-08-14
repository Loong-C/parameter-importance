from __future__ import annotations

import csv
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash
from param_importance_nlp.stage1_training_integration import build_stage1_s16_evidence


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _formalizer():
    path = _root() / "ops/stage1/formalize_s1_6.py"
    spec = importlib.util.spec_from_file_location("s16_formalizer_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s16_chart_csv_and_svg_are_exact_saved_table_projections(tmp_path: Path) -> None:
    evidence = build_stage1_s16_evidence(
        _root(), producer_commit="d21b084faa7b9c13cdd22aa09253ea0cca75c3de"
    )
    formalizer = _formalizer()
    csv_hashes, svg_hashes = formalizer._charts(tmp_path, evidence)
    projections = formalizer._chart_projections(evidence)
    for name, projected in projections.items():
        with (tmp_path / name).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [
            {field: "" if row.get(field) is None else str(row.get(field)) for field in formalizer._CSV_FIELDS}
            for row in projected
        ]
        svg = (tmp_path / name.replace(".csv", ".svg")).read_text(encoding="utf-8")
        assert f'data-source="{name}"' in svg
        mark = "stack-bar" if name == "movement-stack.csv" else "identity-point" if name == "accumulator-identities.csv" else "mass-point" if name == "score-masses.csv" else "trace-mark"
        assert svg.count(f'class="{mark}"') == len(projected)
        assert all(f'data-row="{index}"' in svg for index in range(len(projected)))
        assert name in csv_hashes and name.replace(".csv", ".svg") in svg_hashes
    assert {row["field"] for row in projections["accumulator-identities.csv"]} == {"signed_identity_residual", "absolute_identity_residual"}
    assert {row["field"] for row in projections["movement-stack.csv"]} == {"data_movement", "weight_decay_movement", "total_movement"}
    assert {row["field"] for row in projections["trajectory-parity.csv"]} == {"max_parameter_error"}
    assert {row["field"] for row in projections["score-masses.csv"]} == {"raw", "signed", "positive", "negative_mass"}
    assert formalizer._verify_charts(tmp_path, evidence, csv_hashes, svg_hashes)
    (tmp_path / "score-masses.csv").write_text("tampered\n", encoding="utf-8")
    assert not formalizer._verify_charts(tmp_path, evidence, csv_hashes, svg_hashes)


def test_s16_formalizer_is_fail_closed_and_uses_data_root_pytest_temp() -> None:
    source = (_root() / "ops/stage1/formalize_s1_6.py").read_text(encoding="utf-8")
    assert "S1_6_S1_5_INDEX_NOT_CURRENT" in source
    assert "S1_6_S1_5_SOURCE_MAP_DRIFT" in source
    assert 'str(work / "pytest-tmp")' in source
    assert "S1_6_ATTEMPT_ALREADY_EXISTS" in source
    formalizer = _formalizer()
    assert formalizer.EXPECTED_S1_5_PRODUCER == "36a792b6a89045ae49c32225038a6c10d5082d2c"
    assert formalizer.EXPECTED_INDEX_SHA256 == "890970c5886821377ca0409ca910184da99063432d3ef35fd3e3713a5e5514c5"
    assert formalizer.EXPECTED_GATE_HASH == "4bee73ef5053d78f77d688f94fd1737cc744cbcf2bb4774f8916fc70e466887a"


def test_s16_load_s15_executes_canonical_role_aux_and_rebuild_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise `_load_s15`, including its local canonical-json imports."""

    formalizer = _formalizer()

    def write(name: str, value: dict[str, object]) -> str:
        path = tmp_path / name
        path.write_bytes(canonical_json_bytes(value))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    from param_importance_nlp.stage1_estimators import build_stage1_s15_evidence

    role_names = ("estimator_report", "oracle_report", "tensor_bundle", "comparison_table", "gate_record")
    # Do not stub the historical validator or current rebuild: the normal
    # path below executes both S1.5 implementations against frozen roles.
    roles = build_stage1_s15_evidence(
        _root(), producer_commit=formalizer.EXPECTED_S1_5_PRODUCER
    )
    gate_hash = roles["gate_record"]["artifact_hash"]
    refs = {name: f"{name}.json" for name in role_names}
    role_hashes = {name: write(refs[name], roles[name]) for name in role_names}
    replay = {"schema_version": "s1-5-replay", "status": "PASS"}; replay["replay_hash"] = canonical_json_hash(replay)
    replay_sha = write("replay.json", replay)
    validation = {"schema_version": "s1-5-validation", "status": "PASS", "role_sha256": role_hashes, "replay_sha256": replay_sha, "replay_hash": replay["replay_hash"]}; validation["artifact_hash"] = canonical_json_hash(validation)
    validation_sha = write("validation.json", validation)
    index = {"schema_version": "stage1-s1-5-formalization-index-v1", "status": "PASS", "gate_id": "G1-EST", "task_id": "stage1.05_estimators", "generator_git_commit": formalizer.EXPECTED_S1_5_PRODUCER, "consumer_git_commit": formalizer.EXPECTED_S1_5_PRODUCER, "gate_artifact_hash": gate_hash, "next_task_id": "stage1.06_training_integration_and_accumulators", "role_refs": refs, "role_sha256": role_hashes, "replay_ref": "replay.json", "replay_sha256": replay_sha, "replay_hash": replay["replay_hash"], "validation_ref": "validation.json", "validation_sha256": validation_sha}
    index["artifact_hash"] = canonical_json_hash(index)
    index_sha = write("index.json", index)

    monkeypatch.setattr(formalizer, "EXPECTED_INDEX_SHA256", index_sha)
    monkeypatch.setattr(formalizer, "EXPECTED_INDEX_ARTIFACT_HASH", index["artifact_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_REPLAY_HASH", replay["replay_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_VALIDATION_ARTIFACT_HASH", validation["artifact_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_REPORT_HASH", roles["estimator_report"]["report_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_ORACLE_HASH", roles["oracle_report"]["oracle_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_BUNDLE_HASH", roles["tensor_bundle"]["bundle_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_S1_5_TABLE_HASH", roles["comparison_table"]["table_hash"])
    monkeypatch.setattr(formalizer, "EXPECTED_GATE_HASH", gate_hash)
    monkeypatch.setattr(formalizer, "_audit_s15_producer_diff", lambda repository: ("fixtures/stage1/stage1-s16-training-fixture-v1.json",))
    loaded = formalizer._load_s15(tmp_path, "index.json", _root())
    assert loaded["s1_5_index_sha256"] == index_sha
    assert loaded["s1_5_numeric_replay_role_count"] == 5

    monkeypatch.setattr(formalizer, "EXPECTED_INDEX_SHA256", "0" * 64)
    with pytest.raises(formalizer.Stage1S16FormalError, match="INDEX_NOT_CURRENT"):
        formalizer._load_s15(tmp_path, "index.json", _root())
    monkeypatch.setattr(formalizer, "EXPECTED_INDEX_SHA256", index_sha)
    monkeypatch.setattr(formalizer, "EXPECTED_GATE_HASH", "0" * 64)
    with pytest.raises(formalizer.Stage1S16FormalError, match="HANDOFF_NOT_READY"):
        formalizer._load_s15(tmp_path, "index.json", _root())
    monkeypatch.setattr(formalizer, "EXPECTED_GATE_HASH", gate_hash)
    (tmp_path / refs["oracle_report"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(formalizer.Stage1S16FormalError, match="ROLE_INVALID"):
        formalizer._load_s15(tmp_path, "index.json", _root())
    write(refs["oracle_report"], roles["oracle_report"])
    import param_importance_nlp.stage1_estimators as s15
    drifted = deepcopy(roles); drifted["estimator_report"]["implementation_source_sha256"] = {"src/frozen.py": "c" * 64}
    monkeypatch.setattr(s15, "build_stage1_s15_evidence", lambda *args, **kwargs: drifted)
    with pytest.raises(formalizer.Stage1S16FormalError, match="SOURCE_MAP_DRIFT"):
        formalizer._load_s15(tmp_path, "index.json", _root())


def test_s16_s15_producer_diff_rejects_near_prefix_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()

    monkeypatch.setattr(
        formalizer,
        "_git",
        lambda repository, *args: "src/param_importance_nlp/runtime/training.py.extra\n",
    )
    with pytest.raises(formalizer.Stage1S16FormalError, match="PRODUCER_DIFF_UNAUTHORIZED"):
        formalizer._audit_s15_producer_diff(_root())

    monkeypatch.setattr(
        formalizer,
        "_git",
        lambda repository, *args: "schemas/stage1/s1-6-extra-v1.json\n",
    )
    assert formalizer._audit_s15_producer_diff(_root()) == ("schemas/stage1/s1-6-extra-v1.json",)


def test_s16_schema_registry_strictly_rejects_duplicate_keys(tmp_path: Path) -> None:
    formalizer = _formalizer()
    target = tmp_path / "schemas" / "stage1"; target.mkdir(parents=True)
    originals = sorted((_root() / "schemas" / "stage1").glob("s1-6-*.json"))
    assert len(originals) == 8
    for source in originals:
        (target / source.name).write_bytes(source.read_bytes())
    registry = formalizer._schema_registry(tmp_path)
    assert len({key for key in registry if key.startswith("https://")}) == 8
    duplicate = target / "s1-6-validation-v1.json"
    duplicate.write_text('{"$id":"one","$id":"two"}\n', encoding="utf-8")
    with pytest.raises(formalizer.Stage1S16FormalError, match="SCHEMA_PARSE_INVALID"):
        formalizer._schema_registry(tmp_path)


def test_s16_role_schemas_use_stdlib_registry_and_reject_shape_drift() -> None:
    root = _root(); formalizer = _formalizer(); registry = formalizer._schema_registry(root)
    evidence = build_stage1_s16_evidence(root, producer_commit="d21b084faa7b9c13cdd22aa09253ea0cca75c3de")
    roles = {"step_report": "s1-6-step-report-v1.json", "oracle_bundle": "s1-6-oracle-bundle-v1.json", "trace_bundle": "s1-6-trace-bundle-v1.json", "comparison_table": "s1-6-comparison-table-v1.json", "gate_record": "s1-6-gate-record-v1.json"}
    for role, filename in roles.items():
        schema = registry[filename]
        formalizer._validate_schema(evidence[role], schema, registry, document=schema, path=role)
    malformed = dict(evidence["comparison_table"]); malformed["extra"] = True
    schema = registry["s1-6-comparison-table-v1.json"]
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(malformed, schema, registry, document=schema)
    wrong_count = dict(evidence["comparison_table"]); wrong_count["rows"] = wrong_count["rows"][:-1]
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(wrong_count, schema, registry, document=schema)
    nested_extra = dict(evidence["comparison_table"]); nested_extra["rows"] = [dict(row) for row in nested_extra["rows"]]; nested_extra["rows"][0]["extra"] = "nope"
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(nested_extra, schema, registry, document=schema)
    trace_schema = registry["s1-6-trace-bundle-v1.json"]
    trace_extra = deepcopy(evidence["trace_bundle"])
    trace_extra["tiny_transformer_trace"]["observed_events"][0]["extra"] = "nope"
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(trace_extra, trace_schema, registry, document=trace_schema)
    trace_missing = deepcopy(evidence["trace_bundle"])
    trace_missing["production_adamw_trace"][0].pop("pre_exp_avg")
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(trace_missing, trace_schema, registry, document=trace_schema)
    trace_wrong_type = deepcopy(evidence["trace_bundle"])
    trace_wrong_type["production_sgd_trace"][0]["mean_gradient"]["left"][0] = "not-a-number"
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(trace_wrong_type, trace_schema, registry, document=trace_schema)
    trace_wrong_count = deepcopy(evidence["trace_bundle"])
    trace_wrong_count["nonfinite_skip_trace"]["lifecycle"] = trace_wrong_count["nonfinite_skip_trace"]["lifecycle"][:-1]
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(trace_wrong_count, trace_schema, registry, document=trace_schema)
    fixture = json.loads((root / "fixtures/stage1/stage1-s16-training-fixture-v1.json").read_text(encoding="utf-8"))
    fixture_schema = registry["s1-6-training-fixture-v1.json"]
    formalizer._validate_schema(fixture, fixture_schema, registry, document=fixture_schema, path="fixture")
    fixture_missing = dict(fixture); fixture_missing.pop("tiny_transformer")
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(fixture_missing, fixture_schema, registry, document=fixture_schema)
    fixture_nested_extra = deepcopy(fixture); fixture_nested_extra["tiny_transformer"]["optimizer"]["extra"] = 1
    with pytest.raises(formalizer.Stage1S16FormalError): formalizer._validate_schema(fixture_nested_extra, fixture_schema, registry, document=fixture_schema)

    formal = build_stage1_s16_evidence(
        root, producer_commit="d21b084faa7b9c13cdd22aa09253ea0cca75c3de", scope="formal",
        upstream_evidence={
            "s1_5_index_ref": "evidence/stage1/s1-5-formal/36a792b6a89045ae49c32225038a6c10d5082d2c/formal-20260815-s15-schema-r2/index.json",
            "s1_5_index_sha256": "890970c5886821377ca0409ca910184da99063432d3ef35fd3e3713a5e5514c5",
            "s1_5_gate_artifact_hash": "4bee73ef5053d78f77d688f94fd1737cc744cbcf2bb4774f8916fc70e466887a",
            "s1_5_replay_sha256": "c" * 64,
        },
    )
    report_schema = registry["s1-6-step-report-v1.json"]
    formalizer._validate_schema(formal["step_report"], report_schema, registry, document=report_schema)


def test_s16_schema_objects_are_closed_or_typed_dynamic_maps() -> None:
    root = _root()
    allowed_dynamic = {
        "s1-6-trace-bundle-v1.json": {
            "$/\u0024defs/tensorMap", "$/\u0024defs/lrMap", "$/\u0024defs/stateMap", "$/\u0024defs/stateValue/anyOf/5",
        },
        "s1-6-step-report-v1.json": {"$/\u0024defs/sourceMap"},
    }
    allowed_contains = {
        "s1-6-training-fixture-v1.json": {
            "$/properties/adamw_cases/allOf/0/contains", "$/properties/adamw_cases/allOf/1/contains",
        },
    }
    for path in (root / "schemas/stage1").glob("s1-6-*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))

        def walk(value: object, location: str = "$") -> None:
            if isinstance(value, dict):
                if value.get("type") == "object":
                    if location in allowed_contains.get(path.name, set()):
                        # The fixture uses narrow ``contains`` predicates only
                        # to require both frozen AdamW case identifiers.
                        return
                    assert "additionalProperties" in value, f"bare object: {path.name}:{location}"
                    dynamic = value.get("additionalProperties")
                    if dynamic is not False:
                        assert location in allowed_dynamic.get(path.name, set()), f"unapproved dynamic map: {path.name}:{location}"
                        assert isinstance(dynamic, dict), f"untyped dynamic map: {path.name}:{location}"
                        assert "propertyNames" in value and "minProperties" in value and "maxProperties" in value, f"unbounded map: {path.name}:{location}"
                for key, child in value.items():
                    walk(child, f"{location}/{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{location}/{index}")

        walk(document)
