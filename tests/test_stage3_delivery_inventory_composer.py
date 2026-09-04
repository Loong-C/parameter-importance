from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from ops.stage3.compose_stage3_delivery_inventory import (
    DOCUMENT_SCHEMA,
    STAGE310_TASK_ID,
    compose_stage3_delivery_inventory,
)
from ops.stage3.materialize_stage3_delivery_manifest import (
    materialize_stage3_delivery_manifest,
)
from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.experiments.stage3_g38_publisher import (
    REQUIRED_STAGE3_G38_GIT_ROLES,
    REQUIRED_STAGE3_G38_REPLAY_LAYERS,
    REQUIRED_STAGE3_G38_STAGE310_KINDS,
    Stage3G38DeliveryManifest,
)
from param_importance_nlp.runtime import TaskLifecycleError


def _file(root: Path, ref: str) -> dict[str, object]:
    path = root / ref
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = f"formal:{ref}\n".encode()
    path.write_bytes(payload)
    return {
        "path": ref,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _case(root: Path) -> dict[str, object]:
    config_hash = "a" * 64
    refs = {
        role: f"results/stage3/s310/{role}.json"
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }
    identities = {
        role: {
            "task_id": STAGE310_TASK_ID,
            "artifact_kind": role,
            "config_hash": config_hash,
            "artifact_hash": hashlib.sha256(f"artifact:{role}".encode()).hexdigest(),
            "payload_hash": hashlib.sha256(f"payload:{role}".encode()).hexdigest(),
        }
        for role in REQUIRED_STAGE3_G38_STAGE310_KINDS
    }

    tables = [
        {
            "name": f"table-{index}",
            "csv": _file(root, f"results/stage3/reporting/tables/table-{index}.csv"),
            "json": _file(root, f"results/stage3/reporting/tables/table-{index}.json"),
        }
        for index in range(1, 3)
    ]
    raw = _file(root, "results/stage3/reporting/tables/path-results.raw.json")
    shard = _file(root, "results/stage3/reporting/tables/raw-shards.json")
    figures = [
        {
            "id": f"formal-chart-{index}",
            "png": _file(root, f"results/stage3/reporting/figures/formal-chart-{index}.png"),
            "svg": _file(root, f"results/stage3/reporting/figures/formal-chart-{index}.svg"),
        }
        for index in range(1, 3)
    ]
    reporting = {
        "tables": tables,
        "raw_formal_path_results": {"file": raw},
        "raw_shard_manifest": shard,
    }
    payloads = {
        "analysis_report": {"schema_version": "stage3-task-analysis-report-v1"},
        "chart_artifacts": {"rendered_figures": figures},
        "handoff_manifest": {"reporting_files": reporting},
        "gate_summary": {"stage3.G3-8": "NOT_RUN"},
    }

    report = {
        "tex": _file(root, "evidence/stage3/documents/stage3-report-zh.tex"),
        "pdf": _file(root, "evidence/stage3/documents/stage3-report-zh.pdf"),
    }
    beamer = {
        "tex": _file(root, "evidence/stage3/documents/stage3-beamer.tex"),
        "pdf": _file(root, "evidence/stage3/documents/stage3-beamer.pdf"),
        "notes": [_file(root, "evidence/stage3/documents/stage3-speaker-notes.md")],
        "backups": [
            _file(root, f"evidence/stage3/documents/stage3-backup-{index:02d}.tex")
            for index in range(1, 5)
        ],
    }
    document_body = {
        "schema_version": DOCUMENT_SCHEMA,
        "status": "PASS",
        "scope": "formal",
        "formal_eligible": True,
        "producer_commit": "b" * 40,
        "renderer": {"name": "reportlab", "version": "4.0", "invariant_pdf": True},
        "font": {"name": "formal-cjk", "sha256": "c" * 64},
        "inputs": identities,
        "source_refs": refs,
        "figure_inputs": figures,
        "files": {"chinese_report": report, "beamer": beamer},
        "completion_boundary": "PENDING_G3_8_DELIVERY_ACCEPTANCE",
    }
    document = document_body | {"artifact_hash": canonical_json_hash(document_body)}
    document_ref = "evidence/stage3/documents/stage3-formal-document-manifest.json"
    write_canonical_json(root / document_ref, document)

    analysis_scripts = ["ops/stage3/build_formal_analysis.py"]
    for ref in analysis_scripts:
        _file(root, ref)
    replay = {
        role: f"evidence/stage3/replay/{role}.json"
        for role in REQUIRED_STAGE3_G38_REPLAY_LAYERS
    }
    for ref in replay.values():
        _file(root, ref)
    git_sync = {
        role: f"evidence/stage3/git/{role}.json"
        for role in REQUIRED_STAGE3_G38_GIT_ROLES
    }
    for ref in git_sync.values():
        _file(root, ref)
    large = "evidence/stage3/server-large-artifacts.json"
    worklog = "Agent/worklogs/stage3-formal.md"
    _file(root, large)
    _file(root, worklog)
    return {
        "refs": refs,
        "identities": identities,
        "payloads": payloads,
        "document": document,
        "document_ref": document_ref,
        "analysis_scripts": analysis_scripts,
        "replay": replay,
        "git_sync": git_sync,
        "large": large,
        "worklog": worklog,
    }


def _compose(root: Path, case: dict[str, object], **changes: object) -> object:
    arguments = {
        "workspace_root": root,
        "manifest_id": "stage3-formal-delivery-r1",
        "output": "evidence/stage3/delivery/inventory.json",
        "stage3_10_payloads": case["payloads"],
        "stage3_10_refs": case["refs"],
        "stage3_10_identities": case["identities"],
        "document_manifest": case["document"],
        "document_manifest_ref": case["document_ref"],
        "analysis_scripts": case["analysis_scripts"],
        "replay_reports": case["replay"],
        "server_large_artifact_manifest": case["large"],
        "git_sync": case["git_sync"],
        "worklog": case["worklog"],
    }
    arguments.update(changes)
    return compose_stage3_delivery_inventory(**arguments)  # type: ignore[arg-type]


def test_composer_binds_s310_documents_and_materializes_complete_inventory(tmp_path: Path) -> None:
    case = _case(tmp_path)
    inventory = _compose(tmp_path, case)
    assert load_canonical_json(tmp_path / "evidence/stage3/delivery/inventory.json") == inventory

    manifest = materialize_stage3_delivery_manifest(
        workspace_root=tmp_path,
        inventory=inventory,  # type: ignore[arg-type]
        output="evidence/stage3/delivery/manifest.json",
    )
    parsed = Stage3G38DeliveryManifest.from_mapping(
        load_canonical_json(tmp_path / "evidence/stage3/delivery/manifest.json")
    )
    assert parsed == manifest
    assert len(parsed.csv_tables) == 2
    assert len(parsed.json_tables) == 5
    assert len(parsed.figures) == 2
    assert len(parsed.beamer_backups) == 4
    assert len(parsed.file_records()) == 32


def test_composer_rejects_document_hash_identity_and_figure_drift(tmp_path: Path) -> None:
    case = _case(tmp_path)
    bad_hash = deepcopy(case["document"])
    bad_hash["status"] = "BLOCKED"
    with pytest.raises(ValueError, match="formal pre-G3-8 PASS"):
        _compose(tmp_path, case, document_manifest=bad_hash)

    bad_identity = deepcopy(case["document"])
    bad_identity["inputs"]["analysis_report"]["artifact_hash"] = "d" * 64
    bad_identity["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in bad_identity.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="input identity drift"):
        _compose(tmp_path, case, document_manifest=bad_identity)

    bad_figure = deepcopy(case["document"])
    bad_figure["figure_inputs"][0]["png"]["sha256"] = "e" * 64
    bad_figure["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in bad_figure.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="figure record drift"):
        _compose(tmp_path, case, document_manifest=bad_figure)


def test_composer_rejects_missing_roles_and_changed_immutable_retry(tmp_path: Path) -> None:
    case = _case(tmp_path)
    replay = dict(case["replay"])
    replay.pop("local_cpu")
    with pytest.raises(ValueError, match="exactly three required layers"):
        _compose(tmp_path, case, replay_reports=replay)

    _compose(tmp_path, case)
    second_script = "ops/stage3/build_formal_appendix.py"
    _file(tmp_path, second_script)
    with pytest.raises(TaskLifecycleError, match="内容不同|different"):
        _compose(
            tmp_path,
            case,
            analysis_scripts=[*case["analysis_scripts"], second_script],
        )
