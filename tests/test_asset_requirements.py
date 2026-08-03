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
    assert value["pile"]["reference_batch_size"] == 1024  # type: ignore[index]
    assert value["pile"]["reference_reader"] == {  # type: ignore[index]
        "repository": "EleutherAI/pythia",
        "revision": "a19eecb807ec2c79a39ebf18108816e6ffffc1d5",
    }
    assert value["pile"]["reference_reader_oracle"] == {  # type: ignore[index]
        "artifact_ref": "manifests/batch-viewer-comparison.json",
        "artifact_sha256": (
            "94d25dabed134c41a703a792b5bb613c46938d3246b2d2ff8e3904eb9d4da85c"
        ),
        "official_source_ref": (
            "source/pythia-a19eecb807ec2c79a39ebf18108816e6ffffc1d5"
        ),
    }
    model_31m = next(
        item
        for item in value["models"]  # type: ignore[union-attr]
        if item["name"] == "pythia-31m-deduped-step0"
    )
    assert model_31m["legacy_manifest_diagnostic"]["condition"] == (
        "utf8_bom_strict_json_rejected"
    )
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
    pile = schema["$defs"]["pile"]["properties"]
    assert pile["reference_batch_size"]["const"] == 1024
    assert pile["reference_reader"]["properties"]["revision"]["const"] == (
        "a19eecb807ec2c79a39ebf18108816e6ffffc1d5"
    )
    assert pile["reference_reader_oracle"]["properties"]["artifact_ref"][
        "const"
    ] == "manifests/batch-viewer-comparison.json"
    legacy = schema["$defs"]["model"]["properties"][
        "legacy_manifest_diagnostic"
    ]
    assert legacy["properties"]["condition"]["const"] == (
        "utf8_bom_strict_json_rejected"
    )
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


@pytest.mark.parametrize(
    "mutation",
    ("unexpected_field", "missing_stage5_validation", "invalid_stage_type"),
)
def test_pile_workload_matrix_and_fields_are_fail_closed(
    mutation: str,
) -> None:
    value = deepcopy(_requirements())
    workloads = value["pile"]["workloads"]  # type: ignore[index]
    if mutation == "unexpected_field":
        workloads[0]["unexpected"] = True
    elif mutation == "missing_stage5_validation":
        workloads[:] = [
            item
            for item in workloads
            if not (item["stage"] == 5 and item["split"] == "validation")
        ]
    else:
        workloads[0]["stage"] = "oops"
    _rehash(value)

    with pytest.raises(
        AssetRequirementsError,
        match="fields|stage/split matrix|stage must be an integer",
    ):
        validate_stage0_asset_requirements(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("batch_size", "batch size"),
        ("reader_repository", "reference reader"),
        ("reader_revision", "reference reader"),
    ],
)
def test_pile_reference_batch_contract_is_frozen(
    mutation: str, message: str
) -> None:
    value = deepcopy(_requirements())
    if mutation == "batch_size":
        value["pile"]["reference_batch_size"] = 1  # type: ignore[index]
    elif mutation == "reader_repository":
        value["pile"]["reference_reader"]["repository"] = "fork/pythia"  # type: ignore[index]
    else:
        value["pile"]["reference_reader"]["revision"] = "0" * 40  # type: ignore[index]
    _rehash(value)
    with pytest.raises(AssetRequirementsError, match=message):
        validate_stage0_asset_requirements(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("oracle_ref", "oracle ref"),
        ("oracle_source", "source ref"),
        ("legacy_on_other_model", "only valid for the 31M"),
        ("legacy_null_on_other_model", "only valid for the 31M"),
        ("legacy_replacement_alias", "replacement manifest"),
    ],
)
def test_source_provenance_requirements_are_fail_closed(
    mutation: str,
    message: str,
) -> None:
    value = deepcopy(_requirements())
    models = value["models"]  # type: ignore[index]
    model_31m = next(
        item for item in models if item["name"] == "pythia-31m-deduped-step0"
    )
    if mutation == "oracle_ref":
        value["pile"]["reference_reader_oracle"]["artifact_ref"] = (  # type: ignore[index]
            "manifests/other.json"
        )
    elif mutation == "oracle_source":
        value["pile"]["reference_reader_oracle"]["official_source_ref"] = (  # type: ignore[index]
            "source/wrong"
        )
    elif mutation == "legacy_on_other_model":
        models[0]["legacy_manifest_diagnostic"] = deepcopy(
            model_31m["legacy_manifest_diagnostic"]
        )
    elif mutation == "legacy_null_on_other_model":
        models[0]["legacy_manifest_diagnostic"] = None
    else:
        model_31m["legacy_manifest_diagnostic"]["replacement_manifest_ref"] = (
            model_31m["legacy_manifest_diagnostic"]["refs"][0]
        )
    _rehash(value)
    with pytest.raises(AssetRequirementsError, match=message):
        validate_stage0_asset_requirements(value)


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
