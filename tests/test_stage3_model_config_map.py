from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from ops.stage3 import materialize_stage3_fanout as materializer
from ops.stage3 import run_stage3_formal as formal
from param_importance_nlp.contracts import ResolvedConfigV2, load_canonical_json


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _write(path: Path, value: object) -> None:
    formal._write_atomic(path, value)  # type: ignore[arg-type]


def _sanitize(value: object) -> object:
    if isinstance(value, str):
        return (
            value.replace("local-fixture", "stage2-real")
            .replace("synthetic", "stage2-real")
            .replace("fixture", "stage2-real")
        )
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    return value


def _v2_config(path: Path, model: str) -> ResolvedConfigV2:
    base = _sanitize(copy.deepcopy(load_canonical_json(BASE)))
    assert isinstance(base, dict)
    base["identity"].update(
        {
            "stage": 3,
            "formal_eligible": True,
            "run_intent": "formal",
            "input_checkpoint_id": f"pythia-{model.lower()}-step0",
        }
    )
    base["runtime"].update({"allow_dirty_worktree": False, "device": "cuda"})
    base["loss"]["task_type"] = "causal_lm"
    base["path_integration"].update(
        {
            "enabled": True,
            "probe_count": 3,
            "node_budget": 16,
            "default_rule": "simpson",
            "fallback_rule": "gauss_legendre_8",
            "thresholds_ref": "plans/stage3-thresholds.json",
        }
    )
    base["model"].update(
        {
            "architecture": f"pythia-{model.lower()}",
            "asset_id": f"pythia-{model.lower()}-step0",
            "initialization_id": f"pythia-{model.lower()}-step0",
            "revision": f"{model.lower()}-revision",
            "tokenizer_asset_id": "pythia-tokenizer",
        }
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id="stage3.07_formal_experiment_matrix",
        overrides={
            "providers": {
                "kind": "offline_hf",
                "model_manifest_ref": None,
                "model_root_ref": None,
                "data_manifest_ref": None,
                "data_root_ref": None,
                "tokenizer_manifest_ref": None,
                "tokenizer_root_ref": None,
            }
        },
    )


def _map(tmp_path: Path, models: tuple[str, ...] = ("14M", "31M")) -> dict[str, object]:
    entries: dict[str, object] = {}
    for model in models:
        config = _v2_config(tmp_path / f"base-{model}.json", model)
        _write(tmp_path / f"base-{model}.json", config.to_dict())
        entries[model] = {
            "ref": f"base-{model}.json",
            "config_hash": config.config_hash,
        }
    result: dict[str, object] = {
        "schema_version": materializer.MODEL_CONFIG_MAP_SCHEMA,
        "scope": "formal",
        "entries": entries,
    }
    result["artifact_hash"] = formal._canonical_hash(result)
    return result


def _overrides_map(models: tuple[str, ...] = ("14M", "31M")) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": materializer.MODEL_OVERRIDES_MAP_SCHEMA,
        "scope": "formal",
        "entries": {
            model: {"execution": {"timeout_seconds": 900 + index}}
            for index, model in enumerate(models, start=1)
        },
    }
    result["artifact_hash"] = formal._canonical_hash(result)
    return result


def _units(count_14m: int = 72, count_31m: int = 27) -> tuple[object, ...]:
    units: list[object] = []
    for index in range(count_14m + count_31m):
        model = "14M" if index < count_14m else "31M"
        units.append(
            formal.UnitRecord(
                f"unit-{index:03d}",
                model,
                4301 if model == "14M" else 5301,
                "early",
                f"{index + 1:064x}",
                f"probe-{index:03d}",
            )
        )
    return tuple(units)


def test_mixed_formal_99_map_loads_exact_model_coverage(tmp_path: Path) -> None:
    result = materializer._load_model_config_map(
        _map(tmp_path), roots=(tmp_path,), units=_units(), scope="formal"
    )
    assert result is not None
    assert set(result) == {"14M", "31M"}
    assert result["14M"]["model"]["architecture"] == "pythia-14m"
    assert result["31M"]["model"]["architecture"] == "pythia-31m"


def test_legacy_single_config_remains_compatible(tmp_path: Path) -> None:
    value = _sanitize(copy.deepcopy(load_canonical_json(BASE)))
    assert materializer._load_model_config_map(
        value, roots=(tmp_path,), units=_units(1, 0), scope="pilot"
    ) is None


def test_model_config_map_rejects_missing_key(tmp_path: Path) -> None:
    value = _map(tmp_path)
    del value["entries"]["31M"]  # type: ignore[index]
    value["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_CONFIG_MAP_KEYS_INVALID"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


def test_model_config_map_rejects_hash_drift(tmp_path: Path) -> None:
    value = _map(tmp_path)
    value["artifact_hash"] = "0" * 64
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_CONFIG_MAP_HASH_INVALID"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


def test_model_config_map_rejects_wrong_model_identity(tmp_path: Path) -> None:
    value = _map(tmp_path)
    value["entries"]["31M"] = value["entries"]["14M"]  # type: ignore[index]
    value["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_IDENTITY_MISMATCH"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


def test_model_overrides_map_rejects_model_coverage_mismatch(tmp_path: Path) -> None:
    value = _overrides_map(("14M",))
    with pytest.raises(
        formal.Stage3OrchestratorError,
        match="MODEL_OVERRIDES_MAP_KEYS_INVALID",
    ):
        materializer._load_model_overrides_map(
            value, units=_units(), scope="formal"
        )


def test_materializer_selects_base_and_overrides_by_unit_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = _units()
    base_map = _map(tmp_path)
    _write(tmp_path / "base-map.json", base_map)
    overrides: dict[str, object] = {
        "schema_version": materializer.MODEL_OVERRIDES_MAP_SCHEMA,
        "scope": "formal",
        "entries": {
            "14M": {
                "execution": {"timeout_seconds": 901},
                "providers": {
                    "kind": "offline_hf",
                    "model_manifest_ref": None,
                    "model_root_ref": None,
                    "data_manifest_ref": None,
                    "data_root_ref": None,
                    "tokenizer_manifest_ref": None,
                    "tokenizer_root_ref": None,
                },
            },
            "31M": {
                "execution": {"timeout_seconds": 902},
                "providers": {
                    "kind": "offline_hf",
                    "model_manifest_ref": None,
                    "model_root_ref": None,
                    "data_manifest_ref": None,
                    "data_root_ref": None,
                    "tokenizer_manifest_ref": None,
                    "tokenizer_root_ref": None,
                },
            },
        },
    }
    overrides["artifact_hash"] = formal._canonical_hash(overrides)
    _write(tmp_path / "overrides-map.json", overrides)
    _write(
        tmp_path / "stage2.json",
        {
            "run_id": formal.EXPECTED_STAGE2_RUN_ID,
            "default_estimator": "U-32",
            "batch_size": 32,
            "data_variant": "Raw",
        },
    )
    monkeypatch.setattr(
        materializer,
        "load_unit_index",
        lambda *_args, **_kwargs: ("d" * 64, units),
    )
    source = {
        "schema_version": materializer.SOURCE_SCHEMA,
        "task_id": "stage3.07_formal_experiment_matrix",
        "scope": "formal",
        "run_config_hash": "c" * 64,
        "base_config_ref": "base-map.json",
        "stage2_authority_ref": "stage2.json",
        "unit_index_ref": "unit-index.json",
        "unit_index_hash": "d" * 64,
        "config_overrides_ref": "overrides-map.json",
        "input_result_refs_by_endpoint": {
            unit.endpoint_hash: [f"inputs/{unit.endpoint_hash}.json"] for unit in units
        },
        "artifact_output_dir": "runs/stage3/s307",
        "cache_root": "runs/stage3/cache",
        "config_dir": "runs/stage3/configs/s307",
        "selector_dir": "runs/stage3/selectors/formal",
        "result_dir": "runs/stage3/results/s307",
        "state_dir": "runs/stage3/state/s307",
        "status_ref": "runs/stage3/status/s307.json",
        "manifest_ref": "runs/stage3/manifests/s307.json",
    }
    _write(tmp_path / "unit-index.json", {})
    receipt = materializer.materialize(
        source, workspace_root=tmp_path, data_root=tmp_path
    )
    manifest = formal._load_json(tmp_path / receipt["manifest_ref"])
    configs = [formal._load_json(tmp_path / step["config_ref"]) for step in manifest["steps"]]
    assert [item["base_config"]["model"]["architecture"] for item in configs[:72]] == [
        "pythia-14m"
    ] * 72
    assert [item["base_config"]["model"]["architecture"] for item in configs[72:]] == [
        "pythia-31m"
    ] * 27
    assert configs[0]["execution"]["timeout_seconds"] == 901
    assert configs[72]["execution"]["timeout_seconds"] == 902
    assert manifest["run_config_hash"] == source["run_config_hash"]
    assert manifest["unit_index_ref"] == source["unit_index_ref"]
    assert manifest["unit_index_hash"] == source["unit_index_hash"]
    assert source["base_config_ref"] == "base-map.json"
    assert source["config_overrides_ref"] == "overrides-map.json"
    for step, payload in zip(manifest["steps"], configs, strict=True):
        resolved = ResolvedConfigV2.from_mapping(payload)
        assert step["config_hash"] == resolved.config_hash
        assert payload["full_hash"] == resolved.full_hash
