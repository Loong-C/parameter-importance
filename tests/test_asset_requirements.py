from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from param_importance_nlp.asset_requirements import (
    AssetRequirementsError,
    load_stage0_asset_requirements,
    requirements_artifact_hash,
    validate_stage0_asset_requirements,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = ROOT / "configs" / "stage0" / "g3-asset-requirements-v1.json"


def _requirements() -> dict[str, object]:
    return json.loads(REQUIREMENTS_PATH.read_text(encoding="utf-8"))


def _rehash(value: dict[str, object]) -> dict[str, object]:
    value["artifact_hash"] = requirements_artifact_hash(value)
    return value


def test_committed_g3_asset_requirements_are_self_consistent() -> None:
    value = load_stage0_asset_requirements(REQUIREMENTS_PATH)
    assert value["artifact_hash"] == requirements_artifact_hash(value)
    assert value["stage6_route"]["architecture"] == (  # type: ignore[index]
        "pythia-410m-sequence-classification"
    )
    assert value["pile"]["required_cursor_stop"] == 1_048_576  # type: ignore[index]
    assert value["pile"]["required_target_tokens"] == 2_147_483_648  # type: ignore[index]
    assert value["gate_matrix"]["stage0.G3-S6"]["glue_tasks"] == [  # type: ignore[index]
        "sst2",
        "mnli",
        "rte",
    ]


def test_schema_document_captures_pre_shift_and_410m_route() -> None:
    schema = json.loads(
        (ROOT / "schemas" / "stage0" / "asset-requirements-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == (
        "stage0-asset-requirements-v1"
    )
    causal = schema["$defs"]["pile"]["properties"]["causal_lm_contract"]
    assert causal["properties"]["labels_alignment"]["const"] == (
        "pre_shifted_next_token"
    )
    assert causal["properties"]["input_sequence_length"]["const"] == 2048
    assert causal["properties"]["target_sequence_length"]["const"] == 2048
    route = schema["properties"]["stage6_route"]["properties"]
    assert route["architecture"]["const"] == (
        "pythia-410m-sequence-classification"
    )


def test_pile_budget_rejects_one_missing_record_or_target() -> None:
    missing_record = deepcopy(_requirements())
    missing_record["pile"]["required_cursor_stop"] -= 1  # type: ignore[index,operator]
    _rehash(missing_record)
    with pytest.raises(AssetRequirementsError, match="raw-token budget"):
        validate_stage0_asset_requirements(missing_record)

    missing_target = deepcopy(_requirements())
    missing_target["pile"]["workloads"][-2]["required_target_tokens"] -= 1  # type: ignore[index,operator]
    _rehash(missing_target)
    with pytest.raises(AssetRequirementsError, match="token arithmetic"):
        validate_stage0_asset_requirements(missing_target)


def test_pile_rejects_byte_gap_and_overlapping_main_interval() -> None:
    byte_gap = deepcopy(_requirements())
    byte_gap["pile"]["selected_shards"][0]["byte_start"] = 1  # type: ignore[index]
    _rehash(byte_gap)
    with pytest.raises(AssetRequirementsError, match="byte contiguous"):
        validate_stage0_asset_requirements(byte_gap)

    overlap = deepcopy(_requirements())
    overlap["pile"]["cursor_intervals"][2]["start"] -= 1  # type: ignore[index,operator]
    _rehash(overlap)
    with pytest.raises(AssetRequirementsError, match="one partition"):
        validate_stage0_asset_requirements(overlap)


def test_models_and_stage6_route_are_fail_closed() -> None:
    wrong_parameter_total = deepcopy(_requirements())
    wrong_parameter_total["models"][0]["dtype_counts"]["F16"] -= 1  # type: ignore[index,operator]
    _rehash(wrong_parameter_total)
    with pytest.raises(AssetRequirementsError, match="dtype counts"):
        validate_stage0_asset_requirements(wrong_parameter_total)

    wrong_route = deepcopy(_requirements())
    wrong_route["stage6_route"]["architecture"] = (  # type: ignore[index]
        "pythia-160m-sequence-classification"
    )
    _rehash(wrong_route)
    with pytest.raises(AssetRequirementsError, match="Stage 6"):
        validate_stage0_asset_requirements(wrong_route)

    missing_sst2 = deepcopy(_requirements())
    missing_sst2["gate_matrix"]["stage0.G3-S4"]["glue_tasks"] = []  # type: ignore[index]
    _rehash(missing_sst2)
    with pytest.raises(AssetRequirementsError, match="gate matrix"):
        validate_stage0_asset_requirements(missing_sst2)


def test_glue_rejects_test_as_a_derived_labeled_split() -> None:
    value = deepcopy(_requirements())
    value["glue"][0]["preprocessing"]["derived_splits"].append("test")  # type: ignore[index]
    _rehash(value)
    with pytest.raises(AssetRequirementsError, match="unlabeled test"):
        validate_stage0_asset_requirements(value)


def test_loader_rejects_bom_and_duplicate_keys(tmp_path: Path) -> None:
    bom = tmp_path / "bom.json"
    bom.write_bytes(b"\xef\xbb\xbf" + REQUIREMENTS_PATH.read_bytes())
    with pytest.raises(ValueError, match="BOM"):
        load_stage0_asset_requirements(bom)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":"x","schema_version":"y"}', encoding="utf-8")
    with pytest.raises(ValueError, match="重复键"):
        load_stage0_asset_requirements(duplicate)
