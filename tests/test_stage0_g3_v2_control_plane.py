from __future__ import annotations

import json
from pathlib import Path

from param_importance_nlp.asset_download_plan import (
    download_plan_artifact_hash,
    load_g3_download_plan,
)
from param_importance_nlp.asset_layout import (
    layout_artifact_hash,
    load_stage0_asset_layout,
)
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements
from param_importance_nlp.contracts import canonical_json_bytes


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "configs/stage0/g3-asset-requirements-v1.json"
V1_LAYOUT_PATH = ROOT / "configs/stage0/g3-asset-layout-v1.json"
V2_LAYOUT_PATH = ROOT / "configs/stage0/g3-asset-layout-v2.json"
V1_PLAN_PATH = ROOT / "configs/stage0/g3-download-plan-v1.json"
V2_PLAN_PATH = ROOT / "configs/stage0/g3-download-plan-v2.json"


def _raw(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _identities(layout: dict[str, object]) -> list[tuple[str, str, str]]:
    return [
        (entry["kind"], entry["requirement_name"], entry["logical_name"])
        for entry in layout["entries"]  # type: ignore[union-attr]
    ]


def test_v2_layout_preserves_v1_order_and_rebinds_all_publication_refs() -> None:
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    v1 = load_stage0_asset_layout(V1_LAYOUT_PATH, requirements=requirements)
    v2 = load_stage0_asset_layout(V2_LAYOUT_PATH, requirements=requirements)

    assert v1["artifact_hash"] == (
        "8b46642655157aca4eb6fb6f169db23f3e96cd381f33ada7d3ee6e243c8f2d1e"
    )
    assert v1["schema_version"] == v2["schema_version"] == (
        "stage0-g3-asset-layout-v1"
    )
    assert _identities(v1) == _identities(v2)
    assert v2["requirements_ref"] == v1["requirements_ref"]
    assert v2["requirements_sha256"] == v1["requirements_sha256"]

    for old, new in zip(v1["entries"], v2["entries"], strict=True):  # type: ignore[union-attr]
        assert new["manifest_ref"].startswith("manifests/g3-v2/")
        assert new["qualification_ref"].startswith("manifests/g3-v2/")
        assert new["manifest_ref"] != old["manifest_ref"]
        assert new["qualification_ref"] != old["qualification_ref"]


def test_v2_changes_only_the_three_derived_asset_roots() -> None:
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    v1 = load_stage0_asset_layout(V1_LAYOUT_PATH, requirements=requirements)
    v2 = load_stage0_asset_layout(V2_LAYOUT_PATH, requirements=requirements)

    for old, new in zip(v1["entries"], v2["entries"], strict=True):  # type: ignore[union-attr]
        if old["kind"] == "glue_derived":
            assert new["asset_root_ref"] == old["asset_root_ref"] + "-v2"
        else:
            assert new["asset_root_ref"] == old["asset_root_ref"]

    assert [
        entry["requirement_name"]
        for entry in v2["entries"]  # type: ignore[union-attr]
        if entry["kind"] == "glue_derived"
    ] == ["sst2", "mnli", "rte"]


def test_v2_plan_reuses_all_download_objects_and_closes_over_v2_layout() -> None:
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    v1_layout = load_stage0_asset_layout(
        V1_LAYOUT_PATH, requirements=requirements
    )
    v2_layout = load_stage0_asset_layout(
        V2_LAYOUT_PATH, requirements=requirements
    )
    v1_plan = load_g3_download_plan(
        V1_PLAN_PATH, requirements=requirements, layout=v1_layout
    )
    v2_plan = load_g3_download_plan(
        V2_PLAN_PATH, requirements=requirements, layout=v2_layout
    )

    assert v1_plan["artifact_hash"] == (
        "a49fd2cfd27902a25992e41122245789823b9b90753c738ee96d34c8c3be04ed"
    )
    assert v2_plan["schema_version"] == "stage0-g3-download-plan-v1"
    assert v2_plan["layout_ref"] == "configs/stage0/g3-asset-layout-v2.json"
    assert v2_plan["layout_sha256"] == v2_layout["artifact_hash"]
    assert v2_plan["entries"] == v1_plan["entries"]
    layout_roots = {entry["asset_root_ref"] for entry in v2_layout["entries"]}
    assert all(entry["asset_root_ref"] in layout_roots for entry in v2_plan["entries"])


def test_v2_control_plane_files_are_canonical_and_hash_bound() -> None:
    for path, artifact_hash in (
        (V2_LAYOUT_PATH, layout_artifact_hash),
        (V2_PLAN_PATH, download_plan_artifact_hash),
    ):
        value = _raw(path)
        assert path.read_bytes() == canonical_json_bytes(value)
        assert value["artifact_hash"] == artifact_hash(value)
