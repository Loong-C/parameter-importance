"""Executable S1.2 public configuration coverage, including ResolvedConfigV2."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from param_importance_nlp.contracts import (
    CONFIG_SECTIONS,
    ConfigContractError,
    ResolvedConfig,
    ResolvedConfigV2,
    config_schema_contract_hash,
    config_schema_contract_payload,
    config_v2_schema_contract_hash,
    config_v2_schema_contract_payload,
    public_config_field_paths,
    public_config_v2_field_paths,
)
from param_importance_nlp.stage1_config_coverage import (
    ACCEPTED_ALTERNATE_BEHAVIOR,
    COMBINATION_GUARD_BEHAVIOR,
    FROZEN_REJECTION_BEHAVIOR,
    FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR,
    FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
    FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
    Stage1ConfigCoverageError,
    coverage_summary,
    load_config_field_behavior_coverage,
    validate_config_field_behavior_coverage,
)
from param_importance_nlp.stage1_config_behavior import compile_config_behavior


ROOT = Path(__file__).resolve().parents[1]
CONFIG_FIXTURE = ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"
COVERAGE_MANIFEST = (
    ROOT / "configs" / "stage1" / "s1-2-config-field-behavior-coverage-v2.json"
)
COVERAGE_SCHEMA = (
    ROOT / "schemas" / "stage1" / "s1-2-config-field-behavior-coverage-v2.json"
)
V1_SHARED_SCHEMA = ROOT / "schemas" / "shared" / "resolved-config-v1.json"
V2_SHARED_SCHEMA = ROOT / "schemas" / "shared" / "resolved-config-v2.json"
TASK_ID = "stage0.05_config_run_identity_and_seeds"


def _schema_sha256() -> str:
    return hashlib.sha256(COVERAGE_SCHEMA.read_bytes()).hexdigest()


def _fixture_v1() -> dict[str, Any]:
    return json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))


def _coverage() -> dict[str, Any]:
    return load_config_field_behavior_coverage(
        COVERAGE_MANIFEST,
        expected_schema_sha256=_schema_sha256(),
    )


def _local_schema_ref(root: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Resolve one local ``$defs`` reference for schema parity inspection."""

    reference = node.get("$ref")
    if reference is None:
        return node
    assert isinstance(reference, str)
    assert reference.startswith("#/$defs/"), reference
    target: Any = root
    for part in reference.removeprefix("#/").split("/"):
        target = target[part]
    assert isinstance(target, dict)
    return target


def _v2_schema_public_leaf_paths() -> set[str]:
    """Expand checked-in v2 public leaves, excluding only derived hash fields.

    ``base_config`` deliberately uses the external v1 reference.  Its exact
    field enumeration is checked independently below and composed here, so a
    newly added v1 or v2 public leaf cannot silently escape the coverage
    manifest merely because the runtime registry is hand-written.
    """

    root = json.loads(V2_SHARED_SCHEMA.read_text(encoding="utf-8"))
    leaves: set[str] = set()

    def walk(path: str, raw_node: dict[str, Any]) -> None:
        node = _local_schema_ref(root, raw_node)
        if "oneOf" in node:
            branches = [
                branch
                for branch in node["oneOf"]
                if not (isinstance(branch, dict) and branch.get("type") == "null")
            ]
            assert branches, path
            for branch in branches:
                assert isinstance(branch, dict)
                walk(path, branch)
            return
        if node.get("type") == "array":
            items = node["items"]
            assert isinstance(items, dict)
            item = _local_schema_ref(root, items)
            if "properties" in item:
                walk(f"{path}[]", item)
            else:
                leaves.add(f"{path}[]")
            return
        properties = node.get("properties")
        if isinstance(properties, dict):
            for field, child in properties.items():
                assert isinstance(field, str) and isinstance(child, dict)
                walk(f"{path}.{field}" if path else field, child)
            return
        leaves.add(path)

    for field, node in root["properties"].items():
        if field in {"config_hash", "full_hash"}:
            continue
        if field == "base_config":
            leaves.update(f"base_config.{path}" for path in public_config_field_paths())
        else:
            assert isinstance(node, dict)
            walk(field, node)
    return leaves


def _v1_schema_public_leaf_paths() -> set[str]:
    """Expand v1's public section fields and compare them to FieldRule paths."""

    root = json.loads(V1_SHARED_SCHEMA.read_text(encoding="utf-8"))
    assert set(root["properties"]) == set(CONFIG_SECTIONS)
    leaves: set[str] = set()
    for section, ref_node in root["properties"].items():
        assert isinstance(ref_node, dict)
        definition = _local_schema_ref(root, ref_node)
        properties = definition.get("properties")
        assert isinstance(properties, dict), section
        leaves.update(f"{section}.{field}" for field in properties)
    return leaves


def _resolve_schema_reference(root: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Resolve the local schema references used by the exact hash-map contract."""

    reference = node.get("$ref")
    assert isinstance(reference, str) and reference.startswith("#/")
    target: Any = root
    for part in reference.removeprefix("#/").split("/"):
        target = target[part]
    assert isinstance(target, dict)
    return target


def _exact_digest_map_keys(root: dict[str, Any], node: dict[str, Any]) -> set[str]:
    """Read an ``additionalProperties: false`` digest-map contract from schema."""

    definition = _resolve_schema_reference(root, node)
    required = definition.get("required")
    properties = definition.get("properties")
    assert definition.get("type") == "object"
    assert definition.get("additionalProperties") is False
    assert isinstance(required, list) and isinstance(properties, dict)
    expected = set(required)
    assert set(properties) == expected
    for key in expected:
        assert properties[key] == {"$ref": "#/$defs/digest"}
    return expected


def _validate_exact_digest_map(value: Any, expected_keys: set[str]) -> None:
    """Fail-closed structural validator used because jsonschema is not locked."""

    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("shared_schema_hashes keys do not match the schema contract")
    for key in expected_keys:
        digest = value[key]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"shared_schema_hashes.{key} is not a SHA-256")


def _config_contract_shared_hash_key_sets(
    schema_path: Path,
    *,
    coverage_container: tuple[str, ...],
) -> tuple[set[str], set[str]]:
    """Return exact v1/v2 shared-map keys declared in coverage/report schemas."""

    root = json.loads(schema_path.read_text(encoding="utf-8"))
    container: dict[str, Any] = root
    for property_name in coverage_container:
        properties = container.get("properties")
        assert isinstance(properties, dict)
        child = properties[property_name]
        assert isinstance(child, dict)
        container = child
    properties = container.get("properties")
    assert isinstance(properties, dict)
    v1_contract = _resolve_schema_reference(root, properties["resolved-config-v1"])
    v2_contract = _resolve_schema_reference(root, properties["resolved-config-v2"])
    return (
        _exact_digest_map_keys(root, v1_contract["properties"]["shared_schema_hashes"]),
        _exact_digest_map_keys(root, v2_contract["properties"]["shared_schema_hashes"]),
    )


def _index_shared_hash_key_sets(schema_path: Path) -> tuple[set[str], set[str]]:
    root = json.loads(schema_path.read_text(encoding="utf-8"))
    properties = root["properties"]["config_shared_schema_hashes"]["properties"]
    assert isinstance(properties, dict)
    return (
        _exact_digest_map_keys(root, properties["resolved-config-v1"]),
        _exact_digest_map_keys(root, properties["resolved-config-v2"]),
    )


def _report_schema_example() -> dict[str, Any]:
    summary = coverage_summary(_coverage())
    return {
        "schema_version": "g1-registry-report-v2",
        "gate_id": "G1-REGISTRY",
        "status": "PASS",
        "execution_scope": "formal_server_cpu",
        "registry": {
            "coordinate_registry_hash": "0" * 64,
            "optimizer_contract_hash": "1" * 64,
            "runtime_layout_hash": "2" * 64,
            "record_count": 1,
            "eligible_numel": 1,
        },
        "checks": [],
        "producer_commit": "a" * 40,
        "consumer_commit": "b" * 40,
        "buffer_policy": "excluded_from_parameter_registry-v1",
        "config_field_behavior_coverage": {
            **summary,
            "manifest_ref": "configs/stage1/s1-2-config-field-behavior-coverage-v2.json",
            "manifest_sha256": "3" * 64,
            "schema_ref": "schemas/stage1/s1-2-config-field-behavior-coverage-v2.json",
            "schema_sha256": _schema_sha256(),
        },
    }


def _index_schema_example() -> dict[str, Any]:
    contracts = _coverage()["config_contracts"]
    return {
        "schema_version": "stage1-s1-2-formalization-index-v2",
        "status": "PASS",
        "gate_id": "G1-REGISTRY",
        "task_id": "stage1.02_architecture_and_parameter_registry",
        "generator_git_commit": "a" * 40,
        "git_branch": "local-test",
        "checked_at": "2026-08-14T00:00:00Z",
        "s1_1_index_ref": "evidence/stage1/s1-1/index.json",
        "s1_1_index_sha256": "b" * 64,
        "s1_1_gate_artifact_hashes": {
            "stage1.G1-ENTRY": "c" * 64,
            "stage1.G1-CONTRACT": "d" * 64,
        },
        "validation_ref": "validation.json",
        "validation_sha256": "e" * 64,
        "report_ref": "g1-registry-report.json",
        "report_sha256": "f" * 64,
        "config_field_behavior_coverage_manifest_ref": "configs/stage1/s1-2-config-field-behavior-coverage-v2.json",
        "config_field_behavior_coverage_manifest_sha256": "0" * 64,
        "config_field_behavior_coverage_artifact_hash": _coverage()["artifact_hash"],
        "config_field_behavior_coverage_schema_ref": "schemas/stage1/s1-2-config-field-behavior-coverage-v2.json",
        "config_field_behavior_coverage_schema_sha256": _schema_sha256(),
        "config_contract_hashes": {
            family: contract["schema_hash"] for family, contract in contracts.items()
        },
        "config_shared_schema_hashes": {
            family: contract["shared_schema_hashes"]
            for family, contract in contracts.items()
        },
        "probe_summary": {},
        "next_task_id": "stage1.03_fixtures_and_oracles",
        "artifact_hash": "1" * 64,
    }


def _behaviors(coverage: dict[str, Any], family: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in coverage["coverage_groups"]:
        if group["config_family"] != family:
            continue
        for path in group["field_paths"]:
            result[path] = group["behavior_id"]
    return result


def _set_v1(value: dict[str, Any], path: str, replacement: Any) -> None:
    section, field = path.split(".", 1)
    value[section][field] = replacement


def _v1_direct_replacement(path: str, source: Any) -> Any:
    """Return a legal one-field alternate for every v1 direct-mutation leaf."""

    explicit: dict[str, Any] = {
        "identity.schema_version": "resolved-config-v1-alt",
        "identity.stage": 1,
        "identity.task": "contracts-local-fixture-alt",
        "identity.route": "local-fixture-alt",
        "identity.master_seed": 1338,
        "identity.parent_experiment_id": "parent-alt",
        "identity.input_run_id": "run-alt",
        "identity.input_checkpoint_id": "checkpoint-alt",
        "runtime.environment_id": "windows-cpu-local-v1-alt",
        "runtime.dependency_profile": "cpu-core-alt",
        "runtime.allow_dirty_worktree": False,
        "model.asset_id": "synthetic:model:v1-alt",
        "model.revision": "fixture-v1-alt",
        "model.tokenizer_asset_id": "synthetic:tokenizer:v1-alt",
        "model.initialization_id": "synthetic-init-v1-alt",
        "model.architecture": "linear-fixture-alt",
        "data.asset_id": "synthetic:data:v1-alt",
        "data.revision": "fixture-v1-alt",
        "data.split": "fixture-alt",
        "data.sequence_length": 5,
        "data.sampler": "with_replacement-alt",
        "data.statistical_unit": "sequence-alt",
        "data.weight_unit": "effective_token-alt",
        "data.sampling_design": "iid_with_replacement-alt",
        "loss.task_type": "causal_lm",
        "loss.reduction": "sum",
        "loss.ignore_index": -99,
        "loss.weighting": "sample",
        "distributed.backend": "gloo",
        "distributed.device_ids": [0],
        "distributed.timeout_seconds": 61,
        "precision.compute_dtype": "bfloat16",
        "precision.path_accumulation_dtype": "float32",
        "precision.amp": True,
        "optimizer.type": "momentum",
        "optimizer.learning_rate": 0.2,
        "optimizer.weight_decay": 0.1,
        "optimizer.momentum": 0.5,
        "optimizer.parameter_groups": [
            {
                "group_id": "encoder",
                "parameter_names": ["encoder.weight"],
                "learning_rate": 0.1,
                "weight_decay": 0.0,
                "momentum": 0.0,
            }
        ],
        "logging.log_every_steps": 2,
        "logging.tensorboard": True,
        "checkpoint.schema_version": "checkpoint-bundle-v1-alt",
        "checkpoint.save_every_steps": 11,
        "checkpoint.max_to_keep": 3,
        "importance.estimator_decision_ref": "decisions/estimator.json",
        "importance.estimator_name": "raw",
        "importance.clip_mode": "global_plugin",
        "importance.accumulate_views": [
            "signed", "positive", "negative_mass", "absolute", "magnitude"
        ],
        "importance.require_decision_for_formal": False,
        "sampling.universe_version": "synthetic-universe-v1-alt",
        "sampling.repetition_count": 2,
        "sampling.reference_batch_size": 32,
        "path_integration.path_name": "full-update-linear-alt",
        "path_integration.default_rule": "simpson",
        "path_integration.fallback_rule": "trapezoid",
        "path_integration.probe_count": 2,
        "path_integration.node_budget": 4,
        "path_integration.thresholds_ref": "thresholds/v1.json",
        "pruning.strategy": "highest",
        "pruning.scope": "layer_balanced",
        "pruning.ratios": [0.5],
        "pruning.random_repetitions": 1,
        "analysis.schema_version": "analysis-v1-alt",
        "analysis.top_fractions": [0.01, 0.1],
        "analysis.confidence_level": 0.9,
        "analysis.bootstrap_repetitions": 2,
        "analysis.source_table_hash": "0" * 64,
    }
    if path in explicit:
        return explicit[path]
    if path in {"runtime.output_root", "runtime.temp_root", "runtime.cache_root"}:
        return f"artifacts/local-fixture/{path.rsplit('.', 1)[1]}-alt"
    if isinstance(source, bool):
        return not source
    raise AssertionError(f"no v1 direct alternate declared for {path}")


def _resolve_v1_direct(path: str) -> tuple[ResolvedConfig, ResolvedConfig]:
    raw = _fixture_v1()
    baseline = ResolvedConfig.from_mapping(raw)
    mutated = deepcopy(raw)
    _set_v1(mutated, path, _v1_direct_replacement(path, mutated[path.split(".", 1)[0]][path.split(".", 1)[1]]))
    return baseline, ResolvedConfig.from_mapping(mutated)


def _v1_pair(path: str) -> tuple[ResolvedConfig, ResolvedConfig]:
    raw = _fixture_v1()
    candidate = deepcopy(raw)
    if path == "identity.run_intent":
        candidate["identity"].update(run_intent="formal", formal_eligible=True)
        candidate["runtime"]["allow_dirty_worktree"] = False
    elif path == "runtime.offline":
        candidate["identity"].update(run_intent="formal", formal_eligible=True)
        candidate["runtime"].update(allow_dirty_worktree=False, offline=False)
    elif path == "runtime.device":
        candidate["identity"].update(run_intent="formal", formal_eligible=True)
        candidate["runtime"].update(allow_dirty_worktree=False, device="cuda")
    elif path == "batching.per_device_batch_size":
        candidate["batching"].update(per_device_batch_size=2, global_batch_size=2)
    elif path == "batching.global_batch_size":
        candidate["batching"].update(global_batch_size=8, accumulation_steps=2)
    elif path == "batching.microbatch_size":
        candidate["batching"]["microbatch_size"] = 2
    elif path == "batching.accumulation_steps":
        candidate["batching"].update(accumulation_steps=2, global_batch_size=8)
    elif path == "batching.no_sync":
        candidate["batching"].update(no_sync=True, accumulation_steps=2, global_batch_size=8)
    elif path == "distributed.world_size":
        candidate["identity"].update(run_intent="formal", formal_eligible=True)
        candidate["runtime"]["allow_dirty_worktree"] = False
        candidate["distributed"].update(world_size=2, backend="gloo", device_ids=[0, 1])
        candidate["batching"]["global_batch_size"] = 8
    elif path == "path_integration.enabled":
        candidate["path_integration"].update(
            enabled=True,
            default_rule="simpson",
            probe_count=1,
            node_budget=1,
            thresholds_ref="thresholds/v1.json",
        )
    elif path == "pruning.enabled":
        candidate["pruning"].update(enabled=True, strategy="highest", ratios=[0.5])
    else:
        raise AssertionError(f"no v1 paired alternate declared for {path}")
    return ResolvedConfig.from_mapping(raw), ResolvedConfig.from_mapping(candidate)


def _assert_v1_frozen(path: str) -> None:
    raw = _fixture_v1()
    replacements = {
        "identity.formal_eligible": True,
        "precision.gradient_dtype": "float16",
        "precision.statistic_dtype": "float16",
        "precision.reference_dtype": "float32",
        "precision.quadrature_weight_dtype": "float32",
        "optimizer.fused": True,
        "optimizer.foreach": True,
        "logging.event_format": "jsonl-v2",
        "checkpoint.two_phase_commit": False,
        "sampling.candidate_batch_sizes": [16, 32],
        "sampling.candidate_microbatch_counts": [1, 2],
        "sampling.microbatch_preference": [16, 8],
    }
    _set_v1(raw, path, replacements[path])
    with pytest.raises(ConfigContractError):
        ResolvedConfig.from_mapping(raw)


def _assert_identity_effect(
    before: ResolvedConfig | ResolvedConfigV2,
    after: ResolvedConfig | ResolvedConfigV2,
    behavior: str,
    *,
    path: str,
) -> None:
    assert before.full_hash != after.full_hash, path
    if behavior in {
        FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR,
        FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
        FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
    }:
        assert before.config_hash == after.config_hash, path
    else:
        assert before.config_hash != after.config_hash, path


def _v2_base() -> ResolvedConfig:
    return ResolvedConfig.from_mapping(_fixture_v1())


def _v2_resolve(
    *,
    base: ResolvedConfig | None = None,
    task_id: str = TASK_ID,
    overrides: dict[str, Any] | None = None,
) -> ResolvedConfigV2:
    return ResolvedConfigV2.resolve(
        _v2_base() if base is None else base,
        task_id=task_id,
        overrides=overrides,
    )


def _v2_direct_replacement(path: str) -> tuple[str, str, Any]:
    replacements: dict[str, tuple[str, str, Any]] = {
        "execution.timeout_seconds": ("execution", "timeout_seconds", 601),
        "execution.max_attempts": ("execution", "max_attempts", 2),
        "execution.dry_run": ("execution", "dry_run", True),
        "execution.fail_on_blocked": ("execution", "fail_on_blocked", True),
        "training.max_steps": ("training", "max_steps", 4),
        "training.max_epochs": ("training", "max_epochs", 2),
        "training.validation_every_steps": ("training", "validation_every_steps", 1),
        "training.gradient_clip_max_norm": ("training", "gradient_clip_max_norm", 1.0),
        "training.deterministic_algorithms": ("training", "deterministic_algorithms", False),
        "data_loader.drop_last": ("data_loader", "drop_last", True),
        "data_loader.cursor_policy": ("data_loader", "cursor_policy", "draw_manifest"),
        "checkpoint_schedule.segments[].end_step": (
            "checkpoint_schedule", "segments", [{"start_step": 0, "end_step": 5, "every_steps": 10}]
        ),
        "checkpoint_schedule.segments[].every_steps": (
            "checkpoint_schedule", "segments", [{"start_step": 0, "end_step": None, "every_steps": 11}]
        ),
        "checkpoint_schedule.save_on_phase_end": ("checkpoint_schedule", "save_on_phase_end", False),
        "checkpoint_schedule.save_optimizer": ("checkpoint_schedule", "save_optimizer", False),
        "checkpoint_schedule.save_rng": ("checkpoint_schedule", "save_rng", False),
        "checkpoint_schedule.save_data_state": ("checkpoint_schedule", "save_data_state", False),
        "precision_runtime.initial_scale": ("precision_runtime", "initial_scale", 32768.0),
        "precision_runtime.growth_factor": ("precision_runtime", "growth_factor", 3.0),
        "precision_runtime.backoff_factor": ("precision_runtime", "backoff_factor", 0.25),
        "precision_runtime.growth_interval": ("precision_runtime", "growth_interval", 1000),
        "precision_runtime.global_found_inf_reduce": ("precision_runtime", "global_found_inf_reduce", False),
        "optimizer_runtime.dampening": ("optimizer_runtime", "dampening", 0.1),
        "orchestration.route_spec_ref": ("orchestration", "route_spec_ref", "routes/v1.json"),
        "orchestration.quadrature_decision_ref": ("orchestration", "quadrature_decision_ref", "decisions/q.json"),
        "orchestration.matrix_ref": ("orchestration", "matrix_ref", "matrix/v1.json"),
        "orchestration.paired_design.budget_unit": ("orchestration", "paired_design", {"enabled": False, "design": "none", "mapping_ref": None, "budget_unit": "tokens"}),
        "orchestration.input_result_refs[]": ("orchestration", "input_result_refs", ["results/input.json"]),
        "artifacts.output_dir": ("artifacts", "output_dir", "runs/stage0-alt"),
        "artifacts.publish_partial": ("artifacts", "publish_partial", True),
    }
    return replacements[path]


def _resolve_v2_direct(path: str) -> tuple[ResolvedConfigV2, ResolvedConfigV2]:
    if path.startswith("base_config."):
        v1_path = path.removeprefix("base_config.")
        before, altered = _resolve_v1_direct(v1_path)
        return _v2_resolve(base=before), _v2_resolve(base=altered)
    section, field, value = _v2_direct_replacement(path)
    return _v2_resolve(), _v2_resolve(overrides={section: {field: value}})


def _formal_cuda_float16_base() -> ResolvedConfig:
    raw = _fixture_v1()
    raw["identity"].update(run_intent="formal", formal_eligible=True)
    raw["runtime"].update(allow_dirty_worktree=False, device="cuda")
    raw["precision"].update(amp=True, compute_dtype="float16")
    return ResolvedConfig.from_mapping(raw)


def _formal_gloo_base() -> ResolvedConfig:
    raw = _fixture_v1()
    raw["identity"].update(run_intent="formal", formal_eligible=True)
    raw["runtime"]["allow_dirty_worktree"] = False
    raw["distributed"].update(world_size=2, backend="gloo", device_ids=[0, 1])
    raw["batching"]["global_batch_size"] = 8
    return ResolvedConfig.from_mapping(raw)


def _offline_provider_overrides(*, root_suffix: str = "base") -> dict[str, Any]:
    return {
        "providers": {
            "kind": "offline_hf",
            "model_manifest_ref": "manifests/model.json",
            "model_root_ref": f"roots/model-{root_suffix}",
            "data_manifest_ref": "manifests/data.json",
            "data_root_ref": f"roots/data-{root_suffix}",
            "tokenizer_manifest_ref": "manifests/tokenizer.json",
            "tokenizer_root_ref": f"roots/tokenizer-{root_suffix}",
        }
    }


def _v2_pair(path: str) -> tuple[ResolvedConfigV2, ResolvedConfigV2]:
    if path.startswith("base_config."):
        v1_path = path.removeprefix("base_config.")
        if v1_path == "identity.stage":
            raw = _fixture_v1()
            raw["identity"]["stage"] = 1
            altered = ResolvedConfig.from_mapping(raw)
            return _v2_resolve(), _v2_resolve(base=altered, task_id="stage1.02_architecture_and_parameter_registry")
        before, altered = _v1_pair(v1_path)
        return _v2_resolve(base=before), _v2_resolve(base=altered)
    if path == "task_id":
        return _v2_resolve(), _v2_resolve(task_id="stage0.01_baseline_and_safety")
    if path.startswith("scheduler."):
        field = path.split(".", 1)[1]
        value = {"kind": "constant", "warmup_steps": 1, "total_steps": 4}[field]
        overrides = {"training": {"max_steps": 4}, "scheduler": {"kind": "constant", "warmup_steps": 0, "total_steps": 4}}
        if field == "warmup_steps":
            overrides["scheduler"]["warmup_steps"] = value
        return _v2_resolve(), _v2_resolve(overrides=overrides)
    if path.startswith("data_loader."):
        field = path.split(".", 1)[1]
        overrides = {"data_loader": {"num_workers": 1, "prefetch_factor": 2, "persistent_workers": False}}
        if field == "persistent_workers":
            overrides["data_loader"]["persistent_workers"] = True
        return _v2_resolve(), _v2_resolve(overrides=overrides)
    if path.startswith("providers."):
        if path in {"providers.kind", "providers.model_manifest_ref", "providers.data_manifest_ref", "providers.tokenizer_manifest_ref"}:
            return _v2_resolve(), _v2_resolve(overrides=_offline_provider_overrides())
        if path in {"providers.model_root_ref", "providers.data_root_ref", "providers.tokenizer_root_ref"}:
            before = _v2_resolve(overrides=_offline_provider_overrides(root_suffix="base"))
            suffix = {"providers.model_root_ref": "alt", "providers.data_root_ref": "base", "providers.tokenizer_root_ref": "base"}
            overrides = _offline_provider_overrides(root_suffix="base")
            field = path.split(".", 1)[1]
            overrides["providers"][field] = f"roots/{field.removesuffix('_ref')}-alt"
            return before, _v2_resolve(overrides=overrides)
        if path in {"providers.task_type", "providers.task_name"}:
            raw = _fixture_v1()
            raw["loss"]["task_type"] = "causal_lm"
            base = ResolvedConfig.from_mapping(raw)
            return _v2_resolve(), _v2_resolve(base=base, overrides={"providers": {"task_type": "causal_lm", "task_name": "pile"}})
        if path == "providers.num_labels":
            raw = _fixture_v1()
            raw["loss"]["task_type"] = "sequence_classification"
            base = ResolvedConfig.from_mapping(raw)
            return _v2_resolve(), _v2_resolve(base=base, overrides={"providers": {"task_type": "sequence_classification", "task_name": "sst2", "num_labels": 2}})
    if path.startswith("evaluation."):
        overrides = {"evaluation": {"enabled": True, "split": "fixture", "every_steps": 1, "batch_size": 1, "max_batches": 1, "metrics": ["loss"], "save_predictions": False}}
        if path == "evaluation.save_predictions":
            overrides["evaluation"]["save_predictions"] = True
        return _v2_resolve(), _v2_resolve(overrides=overrides)
    if path.startswith("profiling."):
        overrides = {"profiling": {"enabled": True, "warmup_steps": 1, "measure_steps": 1, "repetitions": 2, "capture_memory": True, "capture_throughput": False, "capture_communication": False, "synchronize_device": True}}
        if path == "profiling.capture_throughput":
            overrides["profiling"]["capture_throughput"] = True
        if path == "profiling.capture_communication":
            overrides["profiling"]["capture_communication"] = True
        return _v2_resolve(), _v2_resolve(overrides=overrides)
    if path.startswith("precision_runtime."):
        if path == "precision_runtime.grad_scaler_enabled":
            base = _formal_cuda_float16_base()
            before = _v2_resolve(base=base, overrides={"precision_runtime": {"autocast_enabled": True, "autocast_dtype": "float16", "grad_scaler_enabled": False}})
            after = _v2_resolve(base=base, overrides={"precision_runtime": {"autocast_enabled": True, "autocast_dtype": "float16", "grad_scaler_enabled": True}})
            return before, after
        raw = _fixture_v1()
        raw["precision"].update(amp=True, compute_dtype="bfloat16")
        base = ResolvedConfig.from_mapping(raw)
        before = _v2_resolve(base=base, overrides={"precision_runtime": {"autocast_enabled": True, "autocast_dtype": "bfloat16", "grad_scaler_enabled": False}})
        return _v2_resolve(), before
    if path.startswith("optimizer_runtime."):
        if path in {"optimizer_runtime.betas[]", "optimizer_runtime.eps"}:
            raw = _fixture_v1()
            raw["optimizer"]["type"] = "adamw"
            base = ResolvedConfig.from_mapping(raw)
            overrides = {"optimizer_runtime": {"betas": [0.8, 0.99], "eps": 1e-7}}
            return _v2_resolve(), _v2_resolve(base=base, overrides=overrides)
        raw = _fixture_v1()
        raw["optimizer"].update(type="momentum", momentum=0.5)
        base = ResolvedConfig.from_mapping(raw)
        return _v2_resolve(), _v2_resolve(base=base, overrides={"optimizer_runtime": {"nesterov": True}})
    if path.startswith("launcher."):
        base = _formal_gloo_base()
        defaults = {"kind": "torchrun", "backend": "gloo", "world_size": 2, "init_method": "file", "init_ref": "launch/base", "rendezvous_id": "rendezvous-base", "max_restarts": 0}
        before = _v2_resolve(base=base, overrides={"launcher": defaults})
        if path in {"launcher.kind", "launcher.backend", "launcher.world_size"}:
            return _v2_resolve(), before
        if path == "launcher.init_ref":
            defaults["init_ref"] = "launch/alt"
        elif path == "launcher.rendezvous_id":
            defaults["rendezvous_id"] = "rendezvous-alt"
        elif path == "launcher.max_restarts":
            defaults["max_restarts"] = 1
        elif path == "launcher.init_method":
            defaults.update(init_method="env", init_ref=None)
        return before, _v2_resolve(base=base, overrides={"launcher": defaults})
    if path.startswith("orchestration.paired_design."):
        defaults = {"enabled": True, "design": "shared_draws", "mapping_ref": "pairs/v1.json", "budget_unit": "samples"}
        return _v2_resolve(), _v2_resolve(overrides={"orchestration": {"paired_design": defaults}})
    if path in {"recovery.resume_ref", "recovery.max_restarts"}:
        task = "stage0.09_checkpoint_and_resume"
        before = _v2_resolve(task_id=task)
        if path == "recovery.resume_ref":
            after = _v2_resolve(task_id=task, overrides={"recovery": {"resume_ref": "resume/one"}})
        else:
            after = _v2_resolve(task_id=task, overrides={"recovery": {"max_restarts": 1}})
        return before, after
    raise AssertionError(f"no v2 paired alternate declared for {path}")


def _v2_payload_without_hashes() -> dict[str, Any]:
    payload = _v2_resolve().to_dict()
    payload.pop("config_hash")
    payload.pop("full_hash")
    return payload


def _assert_v2_frozen(path: str) -> None:
    payload = _v2_payload_without_hashes()
    replacements: dict[str, Any] = {
        "schema_version": "resolved-config-v2-alt",
        "execution.runner_kind": "training",
        "providers.local_files_only": False,
        "providers.trust_remote_code": True,
        "checkpoint_schedule.segments[].start_step": 1,
        "optimizer_runtime.amsgrad": True,
        "optimizer_runtime.maximize": True,
        "optimizer_runtime.capturable": True,
        "optimizer_runtime.differentiable": True,
        "recovery.mode": "manual",
        "recovery.safe_boundary": "step",
        "artifacts.required_kinds[]": ["different"],
    }
    if path.startswith("base_config."):
        raw = _fixture_v1()
        v1_path = path.removeprefix("base_config.")
        replacements_v1 = {
            "identity.formal_eligible": True,
            "precision.gradient_dtype": "float16",
            "precision.statistic_dtype": "float16",
            "precision.reference_dtype": "float32",
            "precision.quadrature_weight_dtype": "float32",
            "optimizer.fused": True,
            "optimizer.foreach": True,
            "logging.event_format": "jsonl-v2",
            "checkpoint.two_phase_commit": False,
            "sampling.candidate_batch_sizes": [16, 32],
            "sampling.candidate_microbatch_counts": [1, 2],
            "sampling.microbatch_preference": [16, 8],
        }
        _set_v1(raw, v1_path, replacements_v1[v1_path])
        with pytest.raises(ConfigContractError):
            ResolvedConfig.from_mapping(raw)
        return
    if path == "checkpoint_schedule.segments[].start_step":
        payload["checkpoint_schedule"]["segments"][0]["start_step"] = replacements[path]
    elif path == "artifacts.required_kinds[]":
        payload["artifacts"]["required_kinds"] = replacements[path]
    elif path == "schema_version":
        payload["schema_version"] = replacements[path]
    else:
        section, field = path.split(".", 1)
        payload[section][field] = replacements[path]
    with pytest.raises(ConfigContractError):
        ResolvedConfigV2(payload)


def _assert_v1_boolean_guard(path: str) -> None:
    raw = _fixture_v1()
    source = raw[path.split(".", 1)[0]][path.split(".", 1)[1]]
    if not isinstance(source, bool):
        return
    _set_v1(raw, path, not source)
    with pytest.raises(ConfigContractError):
        ResolvedConfig.from_mapping(raw)


def _assert_v2_boolean_guard(path: str) -> None:
    if path.startswith("base_config."):
        return _assert_v1_boolean_guard(path.removeprefix("base_config."))
    if "." not in path:
        return
    payload = _v2_payload_without_hashes()
    if "[]" in path:
        return
    section, field = path.rsplit(".", 1)
    container: Any = payload
    for part in section.split("."):
        container = container[part]
    source = container[field]
    if not isinstance(source, bool):
        return
    container[field] = not source
    with pytest.raises(ConfigContractError):
        ResolvedConfigV2(payload)


def _lookup_behavior(value: dict[str, Any], dotted_path: str) -> Any:
    current: Any = value
    for part in dotted_path.split("."):
        current = current[part]
    return current


_V1_SWITCH_BEHAVIOR_KEYS = {
    "identity.run_intent": "v1.run_mode",
    "runtime.offline": "v1.asset_access",
    "runtime.device": "v1.device_plan",
    "runtime.allow_dirty_worktree": "v1.worktree_policy",
    "data.weights_exogenous": "v1.statistical_assumptions",
    "data.common_mean_assumption": "v1.statistical_assumptions",
    "loss.task_type": "v1.loss_plan",
    "loss.reduction": "v1.loss_plan",
    "loss.weighting": "v1.loss_plan",
    "batching.no_sync": "v1.batch_sync",
    "distributed.backend": "v1.device_plan",
    "precision.compute_dtype": "v1.precision_plan",
    "precision.path_accumulation_dtype": "v1.precision_plan",
    "precision.amp": "v1.precision_plan",
    "optimizer.type": "v1.optimizer_plan",
    "logging.tensorboard": "v1.telemetry_plan",
    "importance.estimator_name": "v1.importance_plan",
    "importance.clip_mode": "v1.importance_plan",
    "importance.require_decision_for_formal": "v1.formal_decision_policy",
    "path_integration.enabled": "v1.path_plan",
    "pruning.enabled": "v1.pruning_plan",
    "pruning.strategy": "v1.pruning_plan",
    "pruning.scope": "v1.pruning_plan",
}
_V2_SWITCH_BEHAVIOR_KEYS = {
    "execution.dry_run": "execution_action",
    "execution.fail_on_blocked": "blocked_input_action",
    "training.deterministic_algorithms": "training_determinism",
    "scheduler.kind": "scheduler_action",
    "data_loader.persistent_workers": "dataloader_action.worker_lifecycle",
    "data_loader.drop_last": "dataloader_action.last_batch",
    "data_loader.cursor_policy": "dataloader_action.cursor_commit",
    "providers.kind": "provider_action",
    "providers.task_type": "provider_action",
    "providers.task_name": "provider_action",
    "evaluation.enabled": "evaluation_action",
    "evaluation.save_predictions": "evaluation_action",
    "profiling.enabled": "profiling_action.enabled",
    "profiling.capture_memory": "profiling_action.captures",
    "profiling.capture_throughput": "profiling_action.captures",
    "profiling.capture_communication": "profiling_action.captures",
    "profiling.synchronize_device": "profiling_action.synchronize",
    "checkpoint_schedule.save_on_phase_end": "checkpoint_action.phase_end",
    "checkpoint_schedule.save_optimizer": "checkpoint_action.optimizer",
    "checkpoint_schedule.save_rng": "checkpoint_action.rng",
    "checkpoint_schedule.save_data_state": "checkpoint_action.data_state",
    "precision_runtime.autocast_enabled": "precision_action",
    "precision_runtime.autocast_dtype": "precision_action",
    "precision_runtime.grad_scaler_enabled": "precision_action",
    "precision_runtime.global_found_inf_reduce": "precision_action",
    "optimizer_runtime.nesterov": "optimizer_action",
    "launcher.kind": "launcher_action",
    "launcher.backend": "launcher_action",
    "launcher.init_method": "launcher_action",
    "orchestration.paired_design.enabled": "paired_action",
    "orchestration.paired_design.design": "paired_action",
    "orchestration.paired_design.budget_unit": "paired_action",
    "artifacts.publish_partial": "publication_action",
}


def _v1_component_pair(path: str) -> tuple[ResolvedConfig, ResolvedConfig]:
    if path == "precision.compute_dtype":
        before_raw = _fixture_v1()
        before_raw["precision"].update(amp=True, compute_dtype="float32")
        after_raw = deepcopy(before_raw)
        after_raw["precision"]["compute_dtype"] = "bfloat16"
        return ResolvedConfig.from_mapping(before_raw), ResolvedConfig.from_mapping(after_raw)
    if path in {
        "identity.run_intent", "runtime.offline", "runtime.device", "batching.no_sync",
        "path_integration.enabled", "pruning.enabled",
    }:
        return _v1_pair(path)
    if path in {"pruning.strategy", "pruning.scope"}:
        before_raw = _fixture_v1()
        before_raw["pruning"].update(enabled=True, strategy="highest", scope="global", ratios=[0.5])
        after_raw = deepcopy(before_raw)
        after_raw["pruning"][path.rsplit(".", 1)[1]] = (
            "lowest" if path == "pruning.strategy" else "layer_balanced"
        )
        return ResolvedConfig.from_mapping(before_raw), ResolvedConfig.from_mapping(after_raw)
    return _resolve_v1_direct(path)


def _v2_component_pair(path: str) -> tuple[ResolvedConfigV2, ResolvedConfigV2]:
    if path.startswith("base_config."):
        v1_path = path.removeprefix("base_config.")
        before, after = _v1_component_pair(v1_path)
        return _v2_resolve(base=before), _v2_resolve(base=after)
    if path in {
        "scheduler.kind", "data_loader.persistent_workers", "providers.kind",
        "providers.task_type", "providers.task_name", "evaluation.enabled",
        "evaluation.save_predictions", "profiling.enabled", "profiling.capture_memory",
        "profiling.capture_throughput", "profiling.capture_communication",
        "profiling.synchronize_device", "precision_runtime.autocast_enabled",
        "precision_runtime.autocast_dtype", "precision_runtime.grad_scaler_enabled",
        "optimizer_runtime.nesterov", "launcher.kind", "launcher.backend",
        "launcher.init_method", "orchestration.paired_design.enabled",
        "orchestration.paired_design.design",
    }:
        return _v2_pair(path)
    return _resolve_v2_direct(path)


def test_config_field_coverage_schema_declares_v1_v2_hash_bound_contract() -> None:
    schema = json.loads(COVERAGE_SCHEMA.read_text(encoding="utf-8"))
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "schema_version", "schema_sha256", "config_contracts", "coverage_groups", "artifact_hash"
    }
    assert schema["properties"]["schema_version"]["const"].endswith("-v2")


def test_shared_schema_hash_maps_are_exact_in_all_s12_json_schemas() -> None:
    """Locked-test structural checks enforce the three schema map contracts."""

    expected_v1 = {"resolved-config-v1"}
    expected_v2 = {"resolved-config-v1", "resolved-config-v2"}
    coverage_key_sets = _config_contract_shared_hash_key_sets(
        COVERAGE_SCHEMA,
        coverage_container=("config_contracts",),
    )
    assert coverage_key_sets == (expected_v1, expected_v2)
    coverage = _coverage()
    _validate_exact_digest_map(
        coverage["config_contracts"]["resolved-config-v1"]["shared_schema_hashes"],
        coverage_key_sets[0],
    )
    _validate_exact_digest_map(
        coverage["config_contracts"]["resolved-config-v2"]["shared_schema_hashes"],
        coverage_key_sets[1],
    )
    coverage_unknown = deepcopy(coverage)
    coverage_unknown["config_contracts"]["resolved-config-v1"][
        "shared_schema_hashes"
    ]["unexpected-schema"] = "0" * 64
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            coverage_unknown["config_contracts"]["resolved-config-v1"][
                "shared_schema_hashes"
            ],
            coverage_key_sets[0],
        )
    coverage_missing = deepcopy(coverage)
    coverage_missing["config_contracts"]["resolved-config-v2"][
        "shared_schema_hashes"
    ].pop("resolved-config-v2")
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            coverage_missing["config_contracts"]["resolved-config-v2"][
                "shared_schema_hashes"
            ],
            coverage_key_sets[1],
        )

    report_key_sets = _config_contract_shared_hash_key_sets(
        ROOT / "schemas" / "stage1" / "g1-registry-report-v2.json",
        coverage_container=("config_field_behavior_coverage", "config_contracts"),
    )
    assert report_key_sets == (expected_v1, expected_v2)
    report = _report_schema_example()
    report_contracts = report["config_field_behavior_coverage"]["config_contracts"]
    _validate_exact_digest_map(
        report_contracts["resolved-config-v1"]["shared_schema_hashes"],
        report_key_sets[0],
    )
    _validate_exact_digest_map(
        report_contracts["resolved-config-v2"]["shared_schema_hashes"],
        report_key_sets[1],
    )
    report_unknown = deepcopy(report)
    report_unknown["config_field_behavior_coverage"]["config_contracts"][
        "resolved-config-v1"
    ]["shared_schema_hashes"]["unexpected-schema"] = "0" * 64
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            report_unknown["config_field_behavior_coverage"]["config_contracts"][
                "resolved-config-v1"
            ]["shared_schema_hashes"],
            report_key_sets[0],
        )
    report_missing = deepcopy(report)
    report_missing["config_field_behavior_coverage"]["config_contracts"][
        "resolved-config-v2"
    ]["shared_schema_hashes"].pop("resolved-config-v2")
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            report_missing["config_field_behavior_coverage"]["config_contracts"][
                "resolved-config-v2"
            ]["shared_schema_hashes"],
            report_key_sets[1],
        )

    index_key_sets = _index_shared_hash_key_sets(
        ROOT / "schemas" / "stage1" / "s1-2-formalization-index-v2.json"
    )
    assert index_key_sets == (expected_v1, expected_v2)
    index = _index_schema_example()
    _validate_exact_digest_map(
        index["config_shared_schema_hashes"]["resolved-config-v1"],
        index_key_sets[0],
    )
    _validate_exact_digest_map(
        index["config_shared_schema_hashes"]["resolved-config-v2"],
        index_key_sets[1],
    )
    index_unknown = deepcopy(index)
    index_unknown["config_shared_schema_hashes"]["resolved-config-v1"][
        "unexpected-schema"
    ] = "0" * 64
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            index_unknown["config_shared_schema_hashes"]["resolved-config-v1"],
            index_key_sets[0],
        )
    index_missing = deepcopy(index)
    index_missing["config_shared_schema_hashes"]["resolved-config-v2"].pop(
        "resolved-config-v2"
    )
    with pytest.raises(ValueError, match="keys"):
        _validate_exact_digest_map(
            index_missing["config_shared_schema_hashes"]["resolved-config-v2"],
            index_key_sets[1],
        )


def test_public_field_registries_exactly_match_shared_schema_leaves() -> None:
    """Schema additions/removals must force an explicit registry/coverage update."""

    assert _v1_schema_public_leaf_paths() == set(public_config_field_paths())
    assert _v2_schema_public_leaf_paths() == set(public_config_v2_field_paths())


def test_shared_schema_content_drift_changes_contracts_and_fails_coverage_closed(
    tmp_path: Path,
) -> None:
    """Bind raw checked-in schema content, not merely its field names, to S1.2."""

    drifted_v1 = tmp_path / "resolved-config-v1-drift.json"
    drifted_v2 = tmp_path / "resolved-config-v2-drift.json"
    # Whitespace is sufficient here: the evidence contract freezes the exact
    # checked-in JSON schema bytes, so any content drift requires re-formalizing.
    drifted_v1.write_bytes(V1_SHARED_SCHEMA.read_bytes() + b"\n ")
    drifted_v2.write_bytes(V2_SHARED_SCHEMA.read_bytes() + b"\n ")

    baseline_v1 = config_schema_contract_hash()
    baseline_v2 = config_v2_schema_contract_hash()
    assert config_schema_contract_hash(shared_schema_path=drifted_v1) != baseline_v1
    assert (
        config_v2_schema_contract_hash(v1_shared_schema_path=drifted_v1)
        != baseline_v2
    )
    assert (
        config_v2_schema_contract_hash(v2_shared_schema_path=drifted_v2)
        != baseline_v2
    )
    assert config_schema_contract_payload()["shared_schema"]["sha256"] == hashlib.sha256(
        V1_SHARED_SCHEMA.read_bytes()
    ).hexdigest()
    assert config_v2_schema_contract_payload()["shared_schemas"][
        "resolved-config-v2"
    ]["sha256"] == hashlib.sha256(V2_SHARED_SCHEMA.read_bytes()).hexdigest()

    coverage = _coverage()
    drifted_manifest = deepcopy(coverage)
    drifted_manifest["config_contracts"]["resolved-config-v2"][
        "shared_schema_hashes"
    ]["resolved-config-v2"] = "0" * 64
    from param_importance_nlp.contracts import canonical_json_hash

    body = dict(drifted_manifest)
    body.pop("artifact_hash")
    drifted_manifest["artifact_hash"] = canonical_json_hash(body)
    with pytest.raises(Stage1ConfigCoverageError, match="shared_schema_hashes mismatch"):
        validate_config_field_behavior_coverage(
            drifted_manifest,
            expected_schema_sha256=_schema_sha256(),
        )


def test_v1_public_fields_have_executable_alternate_or_guard_behavior() -> None:
    behaviors = _behaviors(_coverage(), "resolved-config-v1")
    assert set(behaviors) == set(public_config_field_paths())
    for path, behavior in behaviors.items():
        if behavior == FROZEN_REJECTION_BEHAVIOR:
            _assert_v1_frozen(path)
            continue
        if behavior == COMBINATION_GUARD_BEHAVIOR:
            _assert_v1_boolean_guard(path)
            before, after = _v1_pair(path)
        else:
            before, after = _resolve_v1_direct(path)
        _assert_identity_effect(before, after, behavior, path=path)


def test_v2_public_fields_have_executable_alternate_or_guard_behavior() -> None:
    behaviors = _behaviors(_coverage(), "resolved-config-v2")
    assert set(behaviors) == set(public_config_v2_field_paths())
    for path, behavior in behaviors.items():
        if behavior == FROZEN_REJECTION_BEHAVIOR:
            _assert_v2_frozen(path)
            continue
        if behavior in {
            COMBINATION_GUARD_BEHAVIOR,
            FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
        }:
            _assert_v2_boolean_guard(path)
            before, after = _v2_pair(path)
        else:
            before, after = _resolve_v2_direct(path)
        _assert_identity_effect(before, after, behavior, path=path)


def test_every_retained_bool_or_enum_has_component_decision_or_frozen_guard() -> None:
    """No retained switch may pass only because its identity changed."""

    coverage = _coverage()
    v1_behaviors = _behaviors(coverage, "resolved-config-v1")
    v2_behaviors = _behaviors(coverage, "resolved-config-v2")
    for path, behavior_key in _V1_SWITCH_BEHAVIOR_KEYS.items():
        assert v1_behaviors[path] in {ACCEPTED_ALTERNATE_BEHAVIOR, COMBINATION_GUARD_BEHAVIOR}
        before, after = _v1_component_pair(path)
        assert _lookup_behavior(compile_config_behavior(before), behavior_key) != _lookup_behavior(
            compile_config_behavior(after), behavior_key
        ), path
    for path in set(v1_behaviors) - set(_V1_SWITCH_BEHAVIOR_KEYS):
        if path in {
            "identity.formal_eligible", "precision.gradient_dtype", "precision.statistic_dtype",
            "precision.reference_dtype", "precision.quadrature_weight_dtype", "optimizer.fused",
            "optimizer.foreach", "logging.event_format", "checkpoint.two_phase_commit",
        }:
            assert v1_behaviors[path] == FROZEN_REJECTION_BEHAVIOR
    for path, behavior_key in _V2_SWITCH_BEHAVIOR_KEYS.items():
        assert v2_behaviors[path] in {ACCEPTED_ALTERNATE_BEHAVIOR, COMBINATION_GUARD_BEHAVIOR, FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR}
        before, after = _v2_component_pair(path)
        assert _lookup_behavior(compile_config_behavior(before), behavior_key) != _lookup_behavior(
            compile_config_behavior(after), behavior_key
        ), path
    for v1_path, behavior_key in _V1_SWITCH_BEHAVIOR_KEYS.items():
        path = f"base_config.{v1_path}"
        assert v2_behaviors[path] in {ACCEPTED_ALTERNATE_BEHAVIOR, COMBINATION_GUARD_BEHAVIOR}
        before, after = _v2_component_pair(path)
        assert _lookup_behavior(compile_config_behavior(before), behavior_key) != _lookup_behavior(
            compile_config_behavior(after), behavior_key
        ), path
    for path in {
        "schema_version", "execution.runner_kind", "providers.local_files_only",
        "providers.trust_remote_code", "optimizer_runtime.amsgrad",
        "optimizer_runtime.maximize", "optimizer_runtime.capturable",
        "optimizer_runtime.differentiable", "recovery.mode", "recovery.safe_boundary",
    }:
        assert v2_behaviors[path] == FROZEN_REJECTION_BEHAVIOR


def test_config_field_coverage_manifest_fails_closed_on_schema_or_field_drift() -> None:
    coverage = _coverage()
    missing = deepcopy(coverage)
    missing["coverage_groups"][0]["field_paths"].pop()
    body = dict(missing)
    body.pop("artifact_hash")
    from param_importance_nlp.contracts import canonical_json_hash

    missing["artifact_hash"] = canonical_json_hash(body)
    with pytest.raises(Stage1ConfigCoverageError, match="must exactly match"):
        validate_config_field_behavior_coverage(
            missing,
            expected_schema_sha256=_schema_sha256(),
        )

    wrong_schema = deepcopy(coverage)
    wrong_schema["schema_sha256"] = "0" * 64
    body = dict(wrong_schema)
    body.pop("artifact_hash")
    wrong_schema["artifact_hash"] = canonical_json_hash(body)
    with pytest.raises(Stage1ConfigCoverageError, match="schema_sha256"):
        validate_config_field_behavior_coverage(
            wrong_schema,
            expected_schema_sha256=_schema_sha256(),
        )


def test_config_field_coverage_summary_binds_both_public_contracts() -> None:
    summary = coverage_summary(_coverage())
    assert summary["covered_field_counts"] == {
        "resolved-config-v1": len(public_config_field_paths()),
        "resolved-config-v2": len(public_config_v2_field_paths()),
    }
    assert summary["config_contracts"]["resolved-config-v1"]["schema_hash"] == config_schema_contract_hash()
    assert summary["config_contracts"]["resolved-config-v2"]["schema_hash"] == config_v2_schema_contract_hash()
