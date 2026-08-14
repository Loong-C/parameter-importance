from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import torch

import param_importance_nlp.assets as assets
from param_importance_nlp.assets import AssetActorRole, AssetFile, AssetState
from param_importance_nlp.core import (
    BUFFER_POLICY,
    CoreContractError,
    ImportanceState,
    ParameterRegistry,
    RegistryError,
)
from param_importance_nlp.stage1_manifest import (
    Stage1ManifestEncodingError,
    Stage1ManifestValidationError,
    load_stage1_asset_manifest,
    parse_stage1_manifest_bytes,
)


class _RegistrationOrderModel(torch.nn.Module):
    def __init__(self, *, reverse: bool = False, width: int = 2) -> None:
        super().__init__()
        names = ("z", "a") if reverse else ("a", "z")
        for name in names:
            setattr(self, name, torch.nn.Linear(width, width, bias=False))


def _registry(*, reverse: bool = False, lr: float = 0.1) -> tuple[_RegistrationOrderModel, torch.optim.Optimizer, ParameterRegistry]:
    model = _RegistrationOrderModel(reverse=reverse)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    return model, optimizer, ParameterRegistry.from_model(model, optimizer)


def test_registry_order_is_stable_under_parameter_registration_order_changes() -> None:
    _, _, forward = _registry(reverse=False)
    _, _, reverse = _registry(reverse=True)

    assert forward.eligible_names == ("a.weight", "z.weight")
    assert reverse.eligible_names == forward.eligible_names
    assert forward.coordinate_registry_hash == reverse.coordinate_registry_hash
    assert forward.optimizer_contract_hash == reverse.optimizer_contract_hash
    assert [record.order for record in reverse] == [0, 1]
    assert sum(record.numel for record in reverse.eligible_records) == 8


def test_registry_records_aliases_labels_and_expected_gradient_absence() -> None:
    model = torch.nn.Module()
    model.attention = torch.nn.Linear(2, 2, bias=False)
    model.zz = torch.nn.Module()
    model.zz.weight = model.attention.weight
    optimizer = torch.optim.SGD([model.attention.weight], lr=0.1, weight_decay=0.01)
    registry = ParameterRegistry.from_model(model, optimizer)

    record = registry.record("zz.weight")
    assert record.canonical_name == "attention.weight"
    assert record.aliases == ("zz.weight",)
    assert record.tags["module_type"] == "attention"
    assert record.tags["parameter_role"] == "weight"
    assert record.learning_rate == 0.1
    assert record.weight_decay == 0.01

    model.attention.weight.grad = None
    with pytest.raises(RegistryError, match="异常缺失梯度"):
        registry.validate_model_gradients()
    model.attention.weight.grad = torch.ones_like(model.attention.weight)
    result = registry.validate_model_gradients()
    assert result["present"] == ("attention.weight",)


def test_registry_explicitly_excludes_buffers_from_coordinate_contract() -> None:
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("running_mean", torch.zeros(2))
    optimizer = torch.optim.SGD([model.weight], lr=0.1)
    registry = ParameterRegistry.from_model(model, optimizer)

    assert BUFFER_POLICY == "excluded_from_parameter_registry-v1"
    assert registry.eligible_names == ("weight",)
    assert all(
        record["canonical_name"] != "running_mean"
        for record in registry.to_manifest()["records"]
    )
    with pytest.raises(RegistryError, match="未知参数名称"):
        registry.canonical_name("running_mean")


def test_registry_save_reload_and_runtime_contract_reject_shape_or_lr_drift(tmp_path: Path) -> None:
    model, optimizer, registry = _registry()
    path = tmp_path / "parameter-registry.json"
    registry.save(path)
    restored = ParameterRegistry.load(path)

    assert restored.to_manifest() == registry.to_manifest()
    restored.validate_against_model(model, optimizer)

    changed_lr = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.0)
    with pytest.raises(RegistryError, match="学习率"):
        restored.validate_against_model(model, changed_lr)

    changed_model = _RegistrationOrderModel(width=3)
    changed_optimizer = torch.optim.SGD(changed_model.parameters(), lr=0.1, momentum=0.0)
    with pytest.raises(RegistryError, match="coordinate_registry_hash"):
        restored.validate_against_model(changed_model, changed_optimizer)

    tampered = deepcopy(registry.to_manifest())
    tampered["records"][0]["shape"] = [99, 99]
    with pytest.raises(RegistryError, match="numel"):
        ParameterRegistry.from_manifest(tampered)


def test_importance_state_enforces_slot_schema_and_bundle_roundtrip(tmp_path: Path) -> None:
    model, _, registry = _registry()
    state = ImportanceState(registry, include_actual_update=True)
    values = {
        name: torch.full(registry.record(name).shape, 1.5, dtype=torch.float32)
        for name in registry.eligible_names
    }
    state.set_slot("s1", values)
    with pytest.raises(CoreContractError, match="完整 optimizer step"):
        state.commit_long_term({"signed": values}, step_completed=False)
    state.commit_long_term(
        {
            "signed": values,
            "positive": values,
            "negative_mass": values,
            "absolute": values,
            "raw": values,
            "data_movement": values,
            "net_data_movement": values,
            "magnitude": values,
            "actual_update_raw_importance": values,
        },
        step_completed=True,
    )
    assert state.slot("s1")["a.weight"].mean().item() == pytest.approx(1.5)
    state.reset_temporary()
    assert state.slot("s1")["a.weight"].sum().item() == 0
    schema = state.schema_manifest()
    assert schema["registry_hash"] == registry.coordinate_registry_hash
    assert schema["byte_order"] in {"little", "big"}

    bundle = state.save_bundle(tmp_path / "importance-state.bundle")
    restored, restored_identity = ImportanceState.load_bundle(bundle.path, registry)
    assert restored_identity.manifest_sha256 == bundle.manifest_sha256
    assert restored.slot_names == state.slot_names
    restored_slots = restored.state_dict()["slots"]
    state_slots = state.state_dict()["slots"]
    for slot_name in state.slot_names:
        for name in registry.eligible_names:
            torch.testing.assert_close(restored_slots[slot_name][name], state_slots[slot_name][name])

    bad = state.state_dict()
    bad["registry_hash"] = "0" * 64
    with pytest.raises(CoreContractError, match="registry_hash"):
        restored.load_state_dict(bad)

    wrong_shape = dict(values)
    wrong_shape["a.weight"] = torch.zeros(1, dtype=torch.float32)
    with pytest.raises(RegistryError, match="shape"):
        state.set_slot("s1", wrong_shape)


def _asset_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "asset-root"
    root.mkdir()
    payload = b"stage1 manifest fixture\n"
    (root / "weights.bin").write_bytes(payload)
    manifest = assets.build_manifest(
        asset_type="model",
        name="stage1-fixture",
        source="fixture:stage1",
        revision="revision-1",
        files=[AssetFile("weights.bin", len(payload), hashlib.sha256(payload).hexdigest(), "weights")],
        actor="test-fetcher",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref="evidence/fetch.json",
        generator_version="tests/stage1",
        metadata={
            "architecture": "FixtureLM",
            "parameter_count": 1,
            "dtype": "float32",
            "initialization_id": "seed:1",
        },
        created_at="2026-08-14T00:00:00Z",
    )
    manifest = assets.transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor="test-fetcher",
        actor_role=AssetActorRole.FETCHER,
        evidence_ref="evidence/download.json",
        summary="fixture downloaded",
        at="2026-08-14T00:00:01Z",
    )
    return root, manifest


def test_stage1_manifest_reader_separates_parse_bom_revision_files_and_hash_checks(tmp_path: Path) -> None:
    root, manifest = _asset_fixture(tmp_path)
    payload = json.dumps(manifest, ensure_ascii=False).encode("utf-8")
    assert parse_stage1_manifest_bytes(
        payload,
        expected_revision="revision-1",
        expected_files=["weights.bin"],
        asset_root=root,
    )["asset_id"] == manifest["asset_id"]
    assert parse_stage1_manifest_bytes(b"\xef\xbb\xbf" + payload)["revision"] == "revision-1"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(payload)
    assert load_stage1_asset_manifest(manifest_path)["asset_id"] == manifest["asset_id"]

    with pytest.raises(Stage1ManifestValidationError, match="revision"):
        parse_stage1_manifest_bytes(payload, expected_revision="revision-2")
    with pytest.raises(Stage1ManifestValidationError, match="文件集合"):
        parse_stage1_manifest_bytes(payload, expected_files=["weights.bin", "config.json"])
    (root / "weights.bin").write_bytes(b"tampered\n")
    with pytest.raises(Stage1ManifestValidationError, match="size/hash"):
        parse_stage1_manifest_bytes(payload, asset_root=root)

    with pytest.raises(Stage1ManifestEncodingError):
        parse_stage1_manifest_bytes(b"\xff\xfe{\"x\": 1}")

    missing = deepcopy(manifest)
    del missing["metadata"]
    with pytest.raises(Stage1ManifestValidationError, match="Stage 0 asset manifest"):
        parse_stage1_manifest_bytes(json.dumps(missing).encode("utf-8"))

    with pytest.raises(Stage1ManifestEncodingError, match="UTF-8 JSON"):
        parse_stage1_manifest_bytes(b'{"duplicate":1,"duplicate":2}')
    with pytest.raises(Stage1ManifestEncodingError, match="UTF-8 JSON"):
        parse_stage1_manifest_bytes(b'{"value":NaN}')
