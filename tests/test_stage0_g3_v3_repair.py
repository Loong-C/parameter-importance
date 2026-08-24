from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from param_importance_nlp.asset_download_plan import (
    download_plan_artifact_hash,
    load_g3_download_plan,
)
from param_importance_nlp.asset_layout import (
    layout_artifact_hash,
    load_stage0_asset_layout,
)
from param_importance_nlp.asset_requirements import (
    load_stage0_asset_requirements,
    requirements_artifact_hash,
)
from param_importance_nlp.contracts import canonical_json_bytes
from param_importance_nlp.g3_gate import (
    G3GateAggregationError,
    load_legacy_model_manifest_diagnostic,
)


ROOT = Path(__file__).resolve().parents[1]
V1_REQUIREMENTS = ROOT / "configs/stage0/g3-asset-requirements-v1.json"
V2_REQUIREMENTS = ROOT / "configs/stage0/g3-asset-requirements-v2.json"
V3_REQUIREMENTS = ROOT / "configs/stage0/g3-asset-requirements-v3.json"
V3_LAYOUT = ROOT / "configs/stage0/g3-asset-layout-v3.json"
V3_PLAN = ROOT / "configs/stage0/g3-download-plan-v3.json"
V4_LAYOUT = ROOT / "configs/stage0/g3-asset-layout-v4.json"
V4_PLAN = ROOT / "configs/stage0/g3-download-plan-v4.json"


def _repair_fixture(tmp_path: Path) -> tuple[dict[str, object], Path, dict[str, object]]:
    requirements = json.loads(V1_REQUIREMENTS.read_text(encoding="utf-8"))
    model = next(
        item
        for item in requirements["models"]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    diagnostic = model["legacy_manifest_diagnostic"]
    canonical_value = {
        "schema": "parameter-importance-model-manifest-v1",
        "asset": "pythia-31m-deduped-step0",
    }
    canonical_raw = canonical_json_bytes(canonical_value)
    assert canonical_raw
    original_raw = b"\xef\xbb\xbf" + canonical_json_bytes(
        {"schema_version": "parameter-importance-model-manifest-v1", "legacy": True}
    )
    diagnostic.update(
        {
            "repair_evidence_ref": "manifests/stage2/s23-manifest-repair-20260823-01.json",
            "repair_evidence_sha256": "0" * 64,
            "canonical_sha256": hashlib.sha256(canonical_raw).hexdigest(),
            "canonical_size_bytes": len(canonical_raw),
        }
    )
    root = tmp_path / "data"
    for ref in diagnostic["refs"]:
        target = root.joinpath(*ref.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(canonical_raw)
    evidence = {
        "schema_version": "stage2-manifest-repair-evidence-v1",
        "repair_id": "s23-manifest-repair-20260823-01",
        "reason": "remove_utf8_bom_without_changing_json_value",
        "targets": [
            {
                "target": str(root.joinpath(*ref.split("/"))),
                "original_sha256": diagnostic["sha256"],
                "original_size_bytes": diagnostic["size_bytes"],
                "original_json_encoding": "utf-8-bom",
                "canonical_sha256": diagnostic["canonical_sha256"],
                "canonical_size_bytes": diagnostic["canonical_size_bytes"],
                "canonical_json_encoding": "utf-8",
                "replaced_atomically": True,
            }
            for ref in diagnostic["refs"]
        ],
        "weights_touched": False,
        "pile_objects_touched": False,
        "active_part_or_lock_touched": False,
    }
    evidence_raw = canonical_json_bytes(evidence)
    diagnostic["repair_evidence_sha256"] = hashlib.sha256(evidence_raw).hexdigest()
    evidence_path = root.joinpath(*diagnostic["repair_evidence_ref"].split("/"))
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_bytes(evidence_raw)
    return model, root, evidence


def test_v3_control_plane_is_append_only_and_bound() -> None:
    requirements = load_stage0_asset_requirements(V2_REQUIREMENTS)
    layout = load_stage0_asset_layout(V3_LAYOUT, requirements=requirements)
    plan = load_g3_download_plan(V3_PLAN, requirements=requirements, layout=layout)
    assert requirements["schema_version"] == "stage0-asset-requirements-v1"
    assert layout["schema_version"] == "stage0-g3-asset-layout-v1"
    assert plan["schema_version"] == "stage0-g3-download-plan-v1"
    assert layout["requirements_ref"].endswith("g3-asset-requirements-v2.json")
    assert plan["layout_ref"].endswith("g3-asset-layout-v3.json")
    assert all(
        entry["manifest_ref"].startswith("manifests/g3-v3/")
        and entry["qualification_ref"].startswith("manifests/g3-v3/")
        for entry in layout["entries"]
    )
    assert {
        entry["asset_root_ref"]
        for entry in layout["entries"]
        if entry["kind"] == "glue_derived"
    } == {
        "datasets/glue-sst2-pretokenized-v3",
        "datasets/glue-mnli-pretokenized-v3",
        "datasets/glue-rte-pretokenized-v3",
    }
    assert requirements["artifact_hash"] == requirements_artifact_hash(requirements)
    assert layout["artifact_hash"] == layout_artifact_hash(layout)
    assert plan["artifact_hash"] == download_plan_artifact_hash(plan)
    repaired_model = next(
        item
        for item in requirements["models"]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    assert repaired_model["legacy_manifest_diagnostic"]["canonical_size_bytes"] == 1247
    assert repaired_model["legacy_manifest_diagnostic"]["canonical_sha256"] == (
        "a9431853c8bba9249fc897661651da0108382154de41abe9843c52c698b860ea"
    )


def test_v4_control_plane_is_append_only_and_reuses_verified_derived_roots() -> None:
    previous = load_stage0_asset_requirements(V2_REQUIREMENTS)
    requirements = load_stage0_asset_requirements(V3_REQUIREMENTS)
    assert requirements == previous
    layout = load_stage0_asset_layout(V4_LAYOUT, requirements=requirements)
    plan = load_g3_download_plan(V4_PLAN, requirements=requirements, layout=layout)
    assert layout["requirements_ref"].endswith("g3-asset-requirements-v3.json")
    assert plan["layout_ref"].endswith("g3-asset-layout-v4.json")
    assert all(
        entry["manifest_ref"].startswith("manifests/g3-v4/")
        and entry["qualification_ref"].startswith("manifests/g3-v4/")
        for entry in layout["entries"]
    )
    assert {
        entry["asset_root_ref"]
        for entry in layout["entries"]
        if entry["kind"] == "glue_derived"
    } == {
        "datasets/glue-sst2-pretokenized-v3",
        "datasets/glue-mnli-pretokenized-v3",
        "datasets/glue-rte-pretokenized-v3",
    }
    assert requirements["artifact_hash"] == requirements_artifact_hash(requirements)
    assert layout["artifact_hash"] == layout_artifact_hash(layout)
    assert plan["artifact_hash"] == download_plan_artifact_hash(plan)


def test_repaired_legacy_manifest_requires_evidence_and_accepts_canonical_pair(
    tmp_path: Path,
) -> None:
    model, root, _ = _repair_fixture(tmp_path)
    diagnostic = load_legacy_model_manifest_diagnostic(root, model)
    assert diagnostic["canonical_size_bytes"] > 0
    evidence_path = root / "manifests/stage2/s23-manifest-repair-20260823-01.json"
    evidence_path.unlink()
    with pytest.raises(
        G3GateAggregationError,
        match="REPAIR_EVIDENCE",
    ):
        load_legacy_model_manifest_diagnostic(root, model)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("payload", "CANONICAL_HASH_MISMATCH"),
        ("evidence_hash", "REPAIR_EVIDENCE_HASH_MISMATCH"),
        ("targets", "REPAIR_TARGETS_MISMATCH"),
        ("weights", "SAFETY_FLAGS_INVALID"),
        ("atomic", "SAFETY_FLAGS_INVALID"),
    ],
)
def test_repaired_legacy_manifest_rejects_evidence_or_payload_drift(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    model, root, evidence = _repair_fixture(tmp_path)
    diagnostic = model["legacy_manifest_diagnostic"]
    if mutation == "payload":
        target = root.joinpath(*diagnostic["refs"][0].split("/"))
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "evidence_hash":
        diagnostic["repair_evidence_sha256"] = "0" * 64
    elif mutation == "targets":
        evidence["targets"][0]["target"] = Path(
            evidence["targets"][0]["target"]
        ).name
    elif mutation == "weights":
        evidence["weights_touched"] = True
    else:
        evidence["targets"][0]["replaced_atomically"] = False
    evidence_path = root.joinpath(*diagnostic["repair_evidence_ref"].split("/"))
    evidence_path.write_bytes(canonical_json_bytes(evidence))
    if mutation not in {"payload", "evidence_hash"}:
        diagnostic["repair_evidence_sha256"] = hashlib.sha256(
            evidence_path.read_bytes()
        ).hexdigest()
    with pytest.raises(G3GateAggregationError, match=message):
        load_legacy_model_manifest_diagnostic(root, model)


def test_old_bom_legacy_diagnostic_remains_compatible(tmp_path: Path) -> None:
    requirements = load_stage0_asset_requirements(V1_REQUIREMENTS)
    model = next(
        item
        for item in requirements["models"]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    old = canonical_json_bytes({"legacy": True})
    raw = b"\xef\xbb\xbf" + old
    model["legacy_manifest_diagnostic"]["size_bytes"] = len(raw)
    model["legacy_manifest_diagnostic"]["sha256"] = hashlib.sha256(raw).hexdigest()
    root = tmp_path / "data"
    for ref in model["legacy_manifest_diagnostic"]["refs"]:
        path = root.joinpath(*ref.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    assert load_legacy_model_manifest_diagnostic(root, model)["sha256"] == hashlib.sha256(raw).hexdigest()
