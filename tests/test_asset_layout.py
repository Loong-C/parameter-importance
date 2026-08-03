from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from param_importance_nlp.asset_layout import (
    AssetLayoutError,
    layout_artifact_hash,
    load_stage0_asset_layout,
    validate_stage0_asset_layout,
)
from param_importance_nlp.asset_requirements import load_stage0_asset_requirements


ROOT = Path(__file__).resolve().parents[1]
LAYOUT_PATH = ROOT / "configs/stage0/g3-asset-layout-v1.json"
REQUIREMENTS_PATH = ROOT / "configs/stage0/g3-asset-requirements-v1.json"


def _values() -> tuple[dict[str, object], dict[str, object]]:
    layout = json.loads(LAYOUT_PATH.read_text(encoding="utf-8"))
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    return layout, requirements


def _rehash(value: dict[str, object]) -> None:
    value["artifact_hash"] = layout_artifact_hash(value)


def test_committed_layout_exactly_implements_the_frozen_gate_matrix() -> None:
    requirements = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    layout = load_stage0_asset_layout(LAYOUT_PATH, requirements=requirements)
    assert layout["artifact_hash"] == layout_artifact_hash(layout)
    assert len(layout["entries"]) == 13
    assert all(
        not entry["asset_root_ref"].startswith("assets/")
        for entry in layout["entries"]
    )
    assert {
        entry["asset_root_ref"].split("/", 1)[0]
        for entry in layout["entries"]
    } == {"models", "datasets"}


def test_layout_rejects_gate_omission_and_requirements_drift() -> None:
    layout, requirements = _values()
    layout["entries"][5]["gate_ids"].remove("stage0.G3-S6")  # type: ignore[index]
    _rehash(layout)
    with pytest.raises(AssetLayoutError, match="gate matrix"):
        validate_stage0_asset_layout(layout, requirements=requirements)

    layout, requirements = _values()
    requirements["artifact_hash"] = "0" * 64
    with pytest.raises(AssetLayoutError, match="does not bind"):
        validate_stage0_asset_layout(layout, requirements=requirements)


def test_layout_rejects_duplicate_publication_refs_and_path_escape() -> None:
    layout, requirements = _values()
    layout["entries"][1]["manifest_ref"] = layout["entries"][0]["manifest_ref"]  # type: ignore[index]
    _rehash(layout)
    with pytest.raises(AssetLayoutError, match="refs must be unique"):
        validate_stage0_asset_layout(layout, requirements=requirements)

    layout, requirements = _values()
    layout["entries"][0]["asset_root_ref"] = "../models/escape"  # type: ignore[index]
    _rehash(layout)
    with pytest.raises(ValueError, match="contains traversal"):
        validate_stage0_asset_layout(layout, requirements=requirements)


def test_layout_loader_rejects_bom(tmp_path: Path) -> None:
    target = tmp_path / "layout.json"
    target.write_bytes(b"\xef\xbb\xbf" + LAYOUT_PATH.read_bytes())
    with pytest.raises(ValueError, match="BOM"):
        load_stage0_asset_layout(target)


def test_formal_provider_configs_do_not_embed_physical_asset_paths() -> None:
    layout, requirements = _values()
    validate_stage0_asset_layout(layout, requirements=requirements)
    paths = sorted((ROOT / "configs/run-ready/v2").glob("*-formal.yaml"))
    provider_configs = 0
    for path in paths:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        providers = value.get("providers")
        if not isinstance(providers, dict) or providers.get("kind") != "offline_hf":
            continue
        provider_configs += 1
        for prefix in ("model", "data", "tokenizer"):
            assert providers[f"{prefix}_manifest_ref"] is None, path.name
            assert providers[f"{prefix}_root_ref"] is None, path.name
    assert provider_configs >= 11
