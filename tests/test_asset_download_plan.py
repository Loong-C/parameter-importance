from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import param_importance_nlp.asset_download_plan as download_plan_module
from param_importance_nlp.asset_acquisition import (
    AcquisitionResult,
    AcquisitionStatus,
)
from param_importance_nlp.asset_download_plan import (
    AssetDownloadPlanError,
    download_plan_artifact_hash,
    execute_g3_download_plan,
    load_g3_download_plan,
    validate_g3_download_plan,
)
from param_importance_nlp.asset_layout import load_stage0_asset_layout
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "configs/stage0/g3-asset-requirements-v1.json"
LAYOUT_PATH = ROOT / "configs/stage0/g3-asset-layout-v1.json"
PLAN_PATH = ROOT / "configs/stage0/g3-download-plan-v1.json"


def _documents() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    layout = load_stage0_asset_layout(LAYOUT_PATH, requirements=requirements)
    plan = load_g3_download_plan(
        PLAN_PATH,
        requirements=requirements,
        layout=layout,
    )
    return requirements, layout, plan


def _rehash(value: dict[str, object]) -> None:
    value["artifact_hash"] = download_plan_artifact_hash(value)


def test_committed_download_plan_is_url_free_exact_and_hash_bound() -> None:
    requirements, layout, plan = _documents()
    assert plan["requirements_sha256"] == requirements["artifact_hash"]
    assert plan["layout_sha256"] == layout["artifact_hash"]
    assert len(plan["entries"]) == 13
    payload = PLAN_PATH.read_text(encoding="utf-8").casefold()
    assert "://" not in payload
    assert "access_token" not in payload
    assert "signature=" not in payload

    entries = plan["entries"]
    for entry in entries:
        spec_path = ROOT / entry["spec_ref"]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        assert entry["object_id"] == spec["source_id"]


def test_download_plan_schema_is_a_strict_project_schema() -> None:
    schema = json.loads(
        (ROOT / "schemas/stage0/download-plan-v1.json").read_text(encoding="utf-8")
    )
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == (
        "stage0-g3-download-plan-v1"
    )
    assert schema["properties"]["entries"]["minItems"] == 13
    assert schema["$defs"]["entry"]["additionalProperties"] is False


def test_download_plan_rejects_target_escape_unknown_root_and_hash_drift() -> None:
    _, layout, plan = _documents()

    escaped = deepcopy(plan)
    escaped["entries"][0]["final_path"] = "../config.json"
    _rehash(escaped)
    with pytest.raises(ValueError):
        validate_g3_download_plan(escaped, layout=layout)

    unknown = deepcopy(plan)
    unknown["entries"][0]["asset_root_ref"] = "models/not-in-layout"
    _rehash(unknown)
    with pytest.raises(AssetDownloadPlanError, match="absent from layout"):
        validate_g3_download_plan(unknown, layout=layout)

    drifted = deepcopy(plan)
    drifted["entries"][0]["object_id"] += "-drift"
    with pytest.raises(AssetDownloadPlanError, match="artifact_hash"):
        validate_g3_download_plan(drifted)


def test_download_executor_derives_runtime_urls_only_in_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, plan = _documents()
    data_root = tmp_path / "data-root"
    (data_root / "models").mkdir(parents=True)
    (data_root / "datasets").mkdir()
    (data_root / "operations").mkdir()
    (data_root / "tmp").mkdir()
    observed_urls: list[str] = []

    def fake_acquire(  # type: ignore[no-untyped-def]
        spec,
        runtime_url,
        target,
        *,
        policy,
        data_root: Path,
    ):
        del policy
        observed_urls.append(runtime_url)
        assert runtime_url.startswith("https://huggingface.co/")
        assert "?" not in runtime_url
        assert str(target).startswith(str(data_root))
        return AcquisitionResult(
            status=AcquisitionStatus.DOWNLOADED,
            source_id=spec.source_id,
            revision=spec.revision,
            size_bytes=spec.expected_size,
            sha256=spec.expected_sha256,
            attempts=1,
            resumed=False,
            network_accessed=True,
        )

    monkeypatch.setattr(download_plan_module, "acquire_http_asset", fake_acquire)
    report_path = data_root / "operations" / "g3-download-report.json"
    report = execute_g3_download_plan(
        plan=plan,
        source_root=ROOT,
        data_root=data_root,
        report_path=report_path,
        started_at="2026-08-03T08:30:00Z",
    )
    assert len(observed_urls) == 13
    assert len(set(observed_urls)) == 13
    assert observed_urls[0].startswith(
        "https://huggingface.co/EleutherAI/pythia-410m-deduped/"
    )
    assert observed_urls[5].startswith(
        "https://huggingface.co/datasets/nyu-mll/glue/"
    )
    assert report["status"] == "PASS"
    assert report["artifact_hash"] == download_plan_module.canonical_json_hash(
        {key: value for key, value in report.items() if key != "artifact_hash"}
    )
    report_text = report_path.read_text(encoding="utf-8")
    assert "https://" not in report_text
    assert all(url not in report_text for url in observed_urls)


def test_download_executor_rejects_report_outside_data_root(tmp_path: Path) -> None:
    _, _, plan = _documents()
    data_root = tmp_path / "data-root"
    for name in ("models", "datasets", "operations"):
        (data_root / name).mkdir(parents=True, exist_ok=True)
    with pytest.raises(AssetDownloadPlanError, match="DATA_ROOT/operations"):
        execute_g3_download_plan(
            plan=plan,
            source_root=ROOT,
            data_root=data_root,
            report_path=tmp_path / "outside.json",
            started_at="2026-08-03T08:30:00Z",
        )
