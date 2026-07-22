from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from param_importance_nlp.contracts import (
    CONFIG_SECTIONS,
    ConfigContractError,
    ResolvedConfig,
    diff_configs,
    load_canonical_json,
    strict_merge,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def _fixture_mapping() -> dict[str, object]:
    value = load_canonical_json(FIXTURE_PATH)
    assert isinstance(value, dict)
    return deepcopy(value)


def _formal_mapping(*, decision_ref: str = "decisions/stage2.json") -> dict[str, object]:
    value = _fixture_mapping()
    value["identity"]["run_intent"] = "formal"
    value["identity"]["formal_eligible"] = True
    value["identity"]["route"] = "pretrain"
    value["runtime"]["allow_dirty_worktree"] = False
    value["importance"]["estimator_decision_ref"] = decision_ref
    return value


def test_local_fixture_is_complete_resolved_and_never_formal() -> None:
    config = ResolvedConfig.from_mapping(_fixture_mapping())

    assert tuple(config.to_dict()) == CONFIG_SECTIONS
    assert config.section("identity")["run_intent"] == "local_fixture"
    assert config.section("identity")["formal_eligible"] is False
    assert config.formal_eligible is False
    assert config.section("precision")["reference_dtype"] == "float64"
    assert config.section("sampling")["microbatch_preference"] == [32, 16, 8, 4]
    assert len(config.config_hash) == len(config.full_hash) == 64


def test_resolved_config_returns_defensive_copies() -> None:
    config = ResolvedConfig.from_mapping(_fixture_mapping())
    exported = config.to_dict()
    exported["identity"]["task"] = "mutated"
    assert config.section("identity")["task"] == "contracts-local-fixture"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"unknown": {}}),
        lambda value: value["optimizer"].update({"betas_secret": [0.9, 0.99]}),
        lambda value: value["identity"].update({"stage": True}),
        lambda value: value["precision"].update({"reference_dtype": "float32"}),
    ],
)
def test_unknown_fields_and_wrong_types_fail_closed(mutation) -> None:
    value = _fixture_mapping()
    mutation(value)
    with pytest.raises(ConfigContractError):
        ResolvedConfig.from_mapping(value)


def test_missing_required_field_is_not_replaced_by_a_fixture_default() -> None:
    value = _fixture_mapping()
    del value["model"]["asset_id"]
    with pytest.raises(ConfigContractError, match="model.asset_id"):
        ResolvedConfig.from_mapping(value)


def test_formal_eligible_is_derived_and_cannot_be_forged() -> None:
    value = _fixture_mapping()
    value["identity"]["formal_eligible"] = True
    with pytest.raises(ConfigContractError, match="不能由调用者伪造"):
        ResolvedConfig.from_mapping(value)

    formal = ResolvedConfig.from_mapping(_formal_mapping())
    assert formal.formal_eligible is True


@pytest.mark.parametrize(
    ("section", "field", "new_value", "message"),
    [
        ("batching", "global_batch_size", 5, "global_batch_size"),
        ("batching", "microbatch_size", 3, "microbatch_size"),
        ("batching", "no_sync", True, "no_sync"),
        ("runtime", "output_root", "../escape", "路径逃逸"),
        ("runtime", "cache_root", "C:\\cache", "POSIX"),
        ("checkpoint", "two_phase_commit", False, "two_phase_commit"),
        ("optimizer", "fused", True, "fused/foreach"),
    ],
)
def test_cross_field_and_safety_rules_fail_before_runtime(
    section: str,
    field: str,
    new_value: object,
    message: str,
) -> None:
    value = _fixture_mapping()
    value[section][field] = new_value
    with pytest.raises(ConfigContractError, match=message):
        ResolvedConfig.from_mapping(value)


def test_distributed_device_ids_reject_bool_as_integer() -> None:
    value = _fixture_mapping()
    value["distributed"]["device_ids"] = [True]
    with pytest.raises(ConfigContractError, match="device_ids"):
        ResolvedConfig.from_mapping(value)


def test_stage2_candidate_grid_and_selection_order_are_frozen() -> None:
    value = _fixture_mapping()
    value["sampling"]["candidate_batch_sizes"] = [16, 32]
    with pytest.raises(ConfigContractError, match="candidate_batch_sizes"):
        ResolvedConfig.from_mapping(value)


def test_parameter_groups_and_accumulator_views_are_strict_nested_contracts() -> None:
    value = _fixture_mapping()
    value["optimizer"]["parameter_groups"] = [
        {
            "group_id": "encoder",
            "parameter_names": ["encoder.weight"],
            "learning_rate": 0.1,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "unknown": True,
        }
    ]
    with pytest.raises(ConfigContractError, match="extra"):
        ResolvedConfig.from_mapping(value)

    value = _fixture_mapping()
    value["importance"]["accumulate_views"].append("invented_view")
    with pytest.raises(ConfigContractError, match="未知视图"):
        ResolvedConfig.from_mapping(value)

    value = _fixture_mapping()
    value["sampling"]["microbatch_preference"] = [16, 32, 8, 4]
    with pytest.raises(ConfigContractError, match="microbatch_preference"):
        ResolvedConfig.from_mapping(value)


def test_strict_merge_only_overrides_declared_fields() -> None:
    base = {"identity": {"stage": 1, "task": "base"}}
    merged = strict_merge(
        base,
        {"identity": {"task": "override"}, "logging": {"log_every_steps": 5}},
    )
    assert merged["identity"] == {"stage": 1, "task": "override"}
    assert merged["logging"] == {"log_every_steps": 5}

    with pytest.raises(ConfigContractError, match="未知字段"):
        strict_merge(base, {"identity": {"typo": 1}})


def test_semantic_hash_is_order_independent() -> None:
    value = _fixture_mapping()
    reordered = {
        section: dict(reversed(list(fields.items())))
        for section, fields in reversed(list(value.items()))
    }
    assert ResolvedConfig.from_mapping(value).config_hash == ResolvedConfig.from_mapping(
        reordered
    ).config_hash


def test_machine_roots_do_not_change_semantic_hash_but_appear_in_diff() -> None:
    left = ResolvedConfig.from_mapping(_fixture_mapping())
    value = _fixture_mapping()
    value["runtime"]["output_root"] = "another/local/output"
    right = ResolvedConfig.from_mapping(value)

    assert left.config_hash == right.config_hash
    assert left.full_hash != right.full_hash
    differences = diff_configs(left, right)
    assert [difference.path for difference in differences] == ["runtime.output_root"]
    assert differences[0].semantic is False


def test_semantic_change_updates_hash_and_diff() -> None:
    left = ResolvedConfig.from_mapping(_fixture_mapping())
    value = _fixture_mapping()
    value["optimizer"]["learning_rate"] = 0.2
    right = ResolvedConfig.from_mapping(value)

    assert left.config_hash != right.config_hash
    difference = diff_configs(left, right)
    assert len(difference) == 1
    assert difference[0].path == "optimizer.learning_rate"
    assert difference[0].semantic is True
