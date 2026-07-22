from __future__ import annotations

import json
from pathlib import Path

from param_importance_nlp.analysis import FrozenSourceTable
from param_importance_nlp.cli import main
from param_importance_nlp.contracts import (
    GateRecord,
    GateStatus,
    LocalValidationRecord,
    LocalValidationStatus,
    write_canonical_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPOSITORY / "configs" / "local-fixtures" / "resolved-config-v1.json"


def test_unified_cli_resolves_and_diffs_config(
    tmp_path: Path, capsys: object
) -> None:
    output = tmp_path / "resolved.json"
    assert main(["config-resolve", str(FIXTURE_CONFIG), "--output", str(output)]) == 0
    summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert summary["formal_eligible"] is False
    assert len(summary["config_hash"]) == 64
    assert output.is_file()

    changed = json.loads(output.read_text(encoding="utf-8"))
    changed["logging"]["log_every_steps"] = 2
    changed_path = tmp_path / "changed.json"
    write_canonical_json(changed_path, changed)
    assert main(["config-diff", str(output), str(changed_path)]) == 0
    difference = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert difference["difference_count"] == 1
    assert difference["differences"][0]["path"] == "logging.log_every_steps"


def test_unified_cli_rejects_duplicate_yaml(tmp_path: Path, capsys: object) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("identity:\n  stage: 0\n  stage: 1\n", encoding="utf-8")
    assert main(["config-resolve", str(duplicate)]) == 2
    assert "CONFIG_YAML_DUPLICATE_KEY:stage" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_artifact_validation_and_separate_gate_summary(
    tmp_path: Path, capsys: object
) -> None:
    gate = GateRecord(
        gate_id="stage0.G1",
        stage=0,
        status=GateStatus.BLOCKED,
        checked_at="2026-07-22T00:00:00Z",
        reasons=("server_unreachable",),
    )
    local = LocalValidationRecord(
        validation_id="local.cli.cpu",
        status=LocalValidationStatus.PASS,
        checked_at="2026-07-22T00:00:00Z",
        checks={"contract": True},
        evidence_refs=("reports/local.json",),
    )
    gate_path = tmp_path / "gate.json"
    local_path = tmp_path / "local.json"
    write_canonical_json(gate_path, gate.to_dict())
    write_canonical_json(local_path, local.to_dict())

    assert main(["artifact-validate", str(gate_path)]) == 0
    validation = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert validation["kind"] == "gate_record"
    assert validation["valid"] is True

    assert main(["gate-summary", str(gate_path), str(local_path)]) == 0
    summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert summary["formal"]["status_counts"] == {"BLOCKED": 1}
    assert summary["formal"]["all_pass"] is False
    assert summary["local_validation"]["status_counts"] == {"PASS": 1}
    assert summary["local_validation"]["formal_eligible"] is False


def test_contract_validate_replays_every_repository_schema(
    capsys: object,
) -> None:
    """schema 源文件允许审阅友好的缩进，但仍逐一通过严格文档边界。"""

    schemas = sorted((REPOSITORY / "schemas").rglob("*.json"))
    assert len(schemas) >= 30
    for schema in schemas:
        assert main(["contract-validate", str(schema)]) == 0, schema
        result = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
        assert result["kind"] == "json_schema"
        assert result["valid"] is True
        assert len(result["artifact_hash"]) == 64


def test_schema_source_loader_rejects_unsafe_json_and_fake_schema(
    tmp_path: Path,
    capsys: object,
) -> None:
    valid = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://parameter-importance.invalid/schemas/fixture-v1.json",
        "title": "Fixture schema",
        "type": "object",
        "properties": {"value": {"type": "number"}},
    }

    # 非 canonical 空白是 schema 源文档的正常表示。
    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(valid, indent=2) + "\n", encoding="utf-8")
    assert main(["artifact-validate", str(pretty)]) == 0
    assert json.loads(capsys.readouterr().out)["kind"] == "json_schema"  # type: ignore[attr-defined]

    unsafe_payloads = {
        "bom.json": b"\xef\xbb\xbf" + pretty.read_bytes(),
        "duplicate.json": (
            b'{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            b'"$id":"https://parameter-importance.invalid/schemas/x.json",'
            b'"title":"x","title":"y","type":"object"}'
        ),
        "nan.json": (
            b'{"$schema":"https://json-schema.org/draft/2020-12/schema",'
            b'"$id":"https://parameter-importance.invalid/schemas/x.json",'
            b'"title":"x","type":"object","default":NaN}'
        ),
    }
    for name, payload in unsafe_payloads.items():
        target = tmp_path / name
        target.write_bytes(payload)
        assert main(["contract-validate", str(target)]) == 2
        assert "ERROR:" in capsys.readouterr().err  # type: ignore[attr-defined]

    fake = tmp_path / "fake.json"
    fake.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
            },
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["artifact-validate", str(fake)]) == 2
    assert "JSON_SCHEMA_PROJECT_ID_INVALID" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_pretty_non_schema_artifact_remains_forbidden(
    tmp_path: Path,
    capsys: object,
) -> None:
    gate = GateRecord(
        gate_id="stage0.G2",
        stage=0,
        status=GateStatus.NOT_RUN,
        checked_at="2026-07-22T00:00:00Z",
        reasons=("fixture_not_run",),
    )
    pretty = tmp_path / "pretty-gate.json"
    pretty.write_text(
        json.dumps(gate.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    assert main(["artifact-validate", str(pretty)]) == 2
    assert "canonical JSON" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_formal_readiness_cli_refuses_missing_freeze_decision_and_gate(
    tmp_path: Path, capsys: object
) -> None:
    config = json.loads(FIXTURE_CONFIG.read_text(encoding="utf-8"))
    config["identity"]["run_intent"] = "formal"
    config["identity"]["formal_eligible"] = True
    config["runtime"]["allow_dirty_worktree"] = False
    formal_config = tmp_path / "formal-config.json"
    write_canonical_json(formal_config, config)

    assert main(["formal-readiness", "--config", str(formal_config)]) == 3
    readiness = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert readiness["formal_eligible"] is False
    assert "MISSING_FREEZE_STAGE_0" in readiness["reasons"]
    assert "MISSING_ESTIMATOR_DECISION" in readiness["reasons"]
    assert "CONFIG_MISSING_ESTIMATOR_DECISION_REF" in readiness["reasons"]

    assert (
        main(
            [
                "formal-readiness",
                "--config",
                str(formal_config),
                "--required-stage",
                "3",
            ]
        )
        == 3
    )
    narrowed = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert "MISSING_FREEZE_STAGE_0" in narrowed["reasons"]
    assert "MISSING_FREEZE_STAGE_1" in narrowed["reasons"]
    assert "MISSING_FREEZE_STAGE_2" in narrowed["reasons"]


def test_report_build_is_hash_bound_and_deterministic(
    tmp_path: Path, capsys: object
) -> None:
    source = tmp_path / "source.json"
    rows = [
        {"coordinate": "a", "value": 1.0, "reference": 1.5},
        {"coordinate": "b", "value": 2.0, "reference": 2.5},
        {"coordinate": "c", "value": 3.0, "reference": 3.5},
    ]
    table = FrozenSourceTable.from_rows(
        name="fixture-importance", schema_version="fixture-source-v1", rows=rows
    )
    write_canonical_json(
        source,
        {
            "name": table.name,
            "schema_version": table.schema_version,
            "rows": rows,
            "content_hash": table.content_hash,
            "frozen": True,
        },
    )
    first_json = tmp_path / "first" / "report.json"
    first_md = tmp_path / "first" / "report.md"
    arguments = [
        "report-build",
        "--source",
        str(source),
        "--report-id",
        "fixture-report",
        "--reference-field",
        "reference",
        "--top-q",
        "0.5",
        "--output-json",
        str(first_json),
        "--output-markdown",
        str(first_md),
    ]
    assert main(arguments) == 0
    first_summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]

    second_json = tmp_path / "second" / "report.json"
    second_md = tmp_path / "second" / "report.md"
    second_arguments = arguments[:-4] + [
        "--output-json",
        str(second_json),
        "--output-markdown",
        str(second_md),
    ]
    assert main(second_arguments) == 0
    second_summary = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert first_summary["report_hash"] == second_summary["report_hash"]
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first_md.read_bytes() == second_md.read_bytes()


def test_report_build_refuses_to_self_freeze_unbound_rows(
    tmp_path: Path, capsys: object
) -> None:
    source = tmp_path / "unbound.json"
    write_canonical_json(
        source,
        {"name": "manual", "schema_version": "v1", "rows": [{"value": 1.0}]},
    )
    assert (
        main(
            [
                "report-build",
                "--source",
                str(source),
                "--report-id",
                "must-fail",
                "--output-json",
                str(tmp_path / "report.json"),
                "--output-markdown",
                str(tmp_path / "report.md"),
            ]
        )
        == 2
    )
    assert "REPORT_SOURCE_FIELDS_MISMATCH" in capsys.readouterr().err  # type: ignore[attr-defined]
