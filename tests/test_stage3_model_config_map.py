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


def _v2_config(path: Path, model: str, seed: int) -> ResolvedConfigV2:
    base = _sanitize(copy.deepcopy(load_canonical_json(BASE)))
    assert isinstance(base, dict)
    base["identity"].update(
        {
            "stage": 3,
            "formal_eligible": True,
            "run_intent": "formal",
            "master_seed": seed,
            "input_run_id": formal.EXPECTED_STAGE2_RUN_ID,
            "input_checkpoint_id": f"pythia-{model.lower()}-seed{seed}-step0",
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
            "asset_id": f"pythia-{model.lower()}-seed{seed}-step0",
            "initialization_id": f"pythia-{model.lower()}-seed{seed}-step0",
            "revision": f"{model.lower()}-revision",
            "tokenizer_asset_id": "pythia-tokenizer",
        }
    )
    return _resolve_formal_v2(base)


def _resolve_formal_v2(base: dict[str, object]) -> ResolvedConfigV2:
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


def _map(
    tmp_path: Path,
    identities: tuple[tuple[str, int], ...] = (
        ("14M", 4301),
        ("14M", 4302),
        ("31M", 5301),
    ),
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    for model, seed in identities:
        ref = f"base-{model}-{seed}.json"
        config = _v2_config(tmp_path / ref, model, seed)
        _write(tmp_path / ref, config.to_dict())
        entries.append(
            {
                "model": model,
                "seed": seed,
                "ref": ref,
                "config_hash": config.config_hash,
            }
        )
    result: dict[str, object] = {
        "schema_version": materializer.MODEL_CONFIG_MAP_SCHEMA,
        "scope": "formal",
        "entries": entries,
    }
    result["artifact_hash"] = formal._canonical_hash(result)
    return result


def _overrides_map(
    identities: tuple[tuple[str, int], ...] = (
        ("14M", 4301),
        ("14M", 4302),
        ("31M", 5301),
    ),
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": materializer.MODEL_OVERRIDES_MAP_SCHEMA,
        "scope": "formal",
        "entries": [
            {
                "model": model,
                "seed": seed,
                "overrides": {"execution": {"timeout_seconds": 900 + index}},
            }
            for index, (model, seed) in enumerate(identities, start=1)
        ],
    }
    result["artifact_hash"] = formal._canonical_hash(result)
    return result


def _units() -> tuple[object, ...]:
    units: list[object] = []
    index = 0
    for model, seed, count in (
        ("14M", 4301, 36),
        ("14M", 4302, 36),
        ("31M", 5301, 27),
    ):
        for _ in range(count):
            units.append(
                formal.UnitRecord(
                    f"unit-{index:03d}",
                    model,
                    seed,
                    "early",
                    f"{index + 1:064x}",
                    f"probe-{index:03d}",
                )
            )
            index += 1
    return tuple(units)


def _rewrite_base(
    tmp_path: Path,
    value: dict[str, object],
    *,
    model: str,
    seed: int,
    mutate: object,
) -> None:
    entries = value["entries"]
    assert isinstance(entries, list)
    entry = next(
        item
        for item in entries
        if isinstance(item, dict)
        and item["model"] == model
        and item["seed"] == seed
    )
    path = tmp_path / str(entry["ref"])
    original = ResolvedConfigV2.from_mapping(formal._load_json(path))
    base = original.base_config.to_dict()
    assert isinstance(mutate, dict)
    for section, changes in mutate.items():
        assert isinstance(changes, dict)
        target = base[section]
        assert isinstance(target, dict)
        target.update(changes)
    rewritten = _resolve_formal_v2(base)
    _write(path, rewritten.to_dict())
    entry["config_hash"] = rewritten.config_hash
    value["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )


def _source(tmp_path: Path, units: tuple[object, ...]) -> dict[str, object]:
    _write(
        tmp_path / "stage2.json",
        {
            "run_id": formal.EXPECTED_STAGE2_RUN_ID,
            "default_estimator": "U-32",
            "batch_size": 32,
            "data_variant": "Raw",
        },
    )
    _write(tmp_path / "unit-index.json", {})
    return {
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


def test_mixed_formal_99_map_loads_exact_model_coverage(tmp_path: Path) -> None:
    result = materializer._load_model_config_map(
        _map(tmp_path), roots=(tmp_path,), units=_units(), scope="formal"
    )
    assert result is not None
    assert set(result) == {("14M", 4301), ("14M", 4302), ("31M", 5301)}
    assert result[("14M", 4301)]["identity"]["master_seed"] == 4301
    assert result[("14M", 4302)]["identity"]["master_seed"] == 4302
    assert result[("31M", 5301)]["model"]["architecture"] == "pythia-31m"
    assert result[("14M", 4301)] != result[("14M", 4302)]


def test_legacy_single_config_remains_compatible(tmp_path: Path) -> None:
    value = _sanitize(copy.deepcopy(load_canonical_json(BASE)))
    assert materializer._load_model_config_map(
        value,
        roots=(tmp_path,),
        units=(formal.UnitRecord("pilot", "14M", 4301, "early", "a" * 64, "probe"),),
        scope="pilot",
    ) is None


def test_model_config_map_rejects_missing_key(tmp_path: Path) -> None:
    value = _map(tmp_path)
    value["entries"] = value["entries"][:-1]  # type: ignore[index]
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
    entries = value["entries"]
    assert isinstance(entries, list)
    _rewrite_base(
        tmp_path,
        value,
        model="31M",
        seed=5301,
        mutate={"model": {"architecture": "pythia-14m"}},
    )
    value["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_IDENTITY_MISMATCH"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


@pytest.mark.parametrize(
    ("model", "seed", "mutate"),
    (
        (
            "14M",
            4301,
            {
                "model": {"asset_id": "pythia-31m-seed4301-step0"},
                "identity": {"input_checkpoint_id": "pythia-31m-seed4301-step0"},
            },
        ),
        (
            "31M",
            5301,
            {
                "model": {
                    "architecture": "pythia-14m",
                    "asset_id": "pythia-14m-seed5301-step0",
                    "initialization_id": "pythia-14m-seed5301-step0",
                },
                "identity": {"input_checkpoint_id": "pythia-14m-seed5301-step0"},
            },
        ),
    ),
)
def test_model_config_map_rejects_mixed_identity_fields(
    tmp_path: Path,
    model: str,
    seed: int,
    mutate: dict[str, object],
) -> None:
    value = _map(tmp_path)
    _rewrite_base(tmp_path, value, model=model, seed=seed, mutate=mutate)
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_IDENTITY_MISMATCH"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


def test_model_config_map_rejects_wrong_seed_binding(tmp_path: Path) -> None:
    value = _map(tmp_path)
    _rewrite_base(
        tmp_path,
        value,
        model="14M",
        seed=4301,
        mutate={"identity": {"master_seed": 4302}},
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_SEED_MISMATCH"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


@pytest.mark.parametrize(
    "entry_mutation",
    (
        lambda entries: entries.pop(1),
        lambda entries: entries.__setitem__(1, dict(entries[1], seed=4301)),
    ),
)
def test_model_config_map_rejects_missing_or_collapsed_seed(
    tmp_path: Path, entry_mutation: object
) -> None:
    value = _map(tmp_path)
    entries = value["entries"]
    assert isinstance(entries, list)
    assert callable(entry_mutation)
    entry_mutation(entries)
    value["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in value.items() if key != "artifact_hash"}
    )
    with pytest.raises(formal.Stage3OrchestratorError, match="MODEL_CONFIG_MAP_KEYS_INVALID"):
        materializer._load_model_config_map(
            value, roots=(tmp_path,), units=_units(), scope="formal"
        )


def test_model_overrides_map_rejects_model_coverage_mismatch(tmp_path: Path) -> None:
    value = _overrides_map((("14M", 4301),))
    with pytest.raises(
        formal.Stage3OrchestratorError,
        match="MODEL_OVERRIDES_MAP_KEYS_INVALID",
    ):
        materializer._load_model_overrides_map(
            value, units=_units(), scope="formal"
        )


def test_materializer_prevalidates_all_steps_before_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = _units()
    _write(tmp_path / "base-map.json", _map(tmp_path))
    overrides = _overrides_map()
    entries = overrides["entries"]
    assert isinstance(entries, list)
    provider = {
        "kind": "offline_hf",
        "model_manifest_ref": None,
        "model_root_ref": None,
        "data_manifest_ref": None,
        "data_root_ref": None,
        "tokenizer_manifest_ref": None,
        "tokenizer_root_ref": None,
    }
    for entry in entries:
        override = entry["overrides"]
        assert isinstance(override, dict)
        override["providers"] = copy.deepcopy(provider)
    final = entries[-1]["overrides"]
    assert isinstance(final, dict)
    final["unknown_override_field"] = True
    overrides["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in overrides.items() if key != "artifact_hash"}
    )
    _write(tmp_path / "overrides-map.json", overrides)
    source = _source(tmp_path, units)
    monkeypatch.setattr(
        materializer,
        "load_unit_index",
        lambda *_args, **_kwargs: ("d" * 64, units),
    )
    with pytest.raises(Exception, match="unknown_override_field"):
        materializer.materialize(source, workspace_root=tmp_path, data_root=tmp_path)
    for ref in (
        source["artifact_output_dir"],
        source["cache_root"],
        source["config_dir"],
        source["selector_dir"],
        source["result_dir"],
        source["state_dir"],
        source["status_ref"],
        source["manifest_ref"],
    ):
        assert not (tmp_path / str(ref)).exists(), ref


def test_materializer_selects_base_and_overrides_by_unit_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = _units()
    base_map = _map(tmp_path)
    _write(tmp_path / "base-map.json", base_map)
    provider_override = {
        "providers": {
            "kind": "offline_hf",
            "model_manifest_ref": None,
            "model_root_ref": None,
            "data_manifest_ref": None,
            "data_root_ref": None,
            "tokenizer_manifest_ref": None,
            "tokenizer_root_ref": None,
        }
    }
    overrides: dict[str, object] = {
        "schema_version": materializer.MODEL_OVERRIDES_MAP_SCHEMA,
        "scope": "formal",
        "entries": [
            {
                "model": model,
                "seed": seed,
                "overrides": {
                    **copy.deepcopy(provider_override),
                    "execution": {"timeout_seconds": timeout},
                },
            }
            for (model, seed), timeout in (
                (("14M", 4301), 901),
                (("14M", 4302), 902),
                (("31M", 5301), 903),
            )
        ],
    }
    overrides["artifact_hash"] = formal._canonical_hash(
        {key: item for key, item in overrides.items() if key != "artifact_hash"}
    )
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
    assert configs[36]["execution"]["timeout_seconds"] == 902
    assert configs[72]["execution"]["timeout_seconds"] == 903
    assert manifest["run_config_hash"] == source["run_config_hash"]
    assert manifest["unit_index_ref"] == source["unit_index_ref"]
    assert manifest["unit_index_hash"] == source["unit_index_hash"]
    assert source["base_config_ref"] == "base-map.json"
    assert source["config_overrides_ref"] == "overrides-map.json"
    for step, payload in zip(manifest["steps"], configs, strict=True):
        resolved = ResolvedConfigV2.from_mapping(payload)
        assert step["config_hash"] == resolved.config_hash
        assert payload["full_hash"] == resolved.full_hash
