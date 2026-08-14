"""Executable S1.2 coverage contract for public resolved-config v1 and v2.

Every retained setting is either compiled by the S1.2 configuration-behavior
component into an execution decision, or is fail-closed by a named frozen or
cross-field guard.  A run/config hash alone is never accepted as evidence that
a retained bool or enum switch has behavior.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .contracts import (
    CONFIG_SCHEMA_VERSION,
    CONFIG_V2_SCHEMA_VERSION,
    canonical_json_hash,
    config_schema_contract_payload,
    config_schema_contract_hash,
    config_v2_schema_contract_payload,
    config_v2_schema_contract_hash,
    load_canonical_json,
    public_config_field_paths,
    public_config_v2_field_paths,
)


CONFIG_FIELD_BEHAVIOR_COVERAGE_SCHEMA_VERSION = (
    "stage1-config-field-behavior-coverage-v2"
)
V1_CONFIG_FIELD_BEHAVIOR_TEST_ID = (
    "tests.test_stage1_s12_config_coverage::"
    "test_v1_public_fields_have_executable_alternate_or_guard_behavior"
)
V2_CONFIG_FIELD_BEHAVIOR_TEST_ID = (
    "tests.test_stage1_s12_config_coverage::"
    "test_v2_public_fields_have_executable_alternate_or_guard_behavior"
)
ACCEPTED_ALTERNATE_BEHAVIOR = "accepted_component_behavior_and_semantic_identity"
FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR = "accepted_alternate_full_identity_only"
FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR = (
    "accepted_component_behavior_and_full_identity_only"
)
COMBINATION_GUARD_BEHAVIOR = "paired_component_behavior_and_semantic_identity"
FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR = (
    "paired_component_behavior_and_full_identity_only"
)
FROZEN_REJECTION_BEHAVIOR = "frozen_rejection_guard"
_SUPPORTED_BEHAVIORS = frozenset(
    {
        ACCEPTED_ALTERNATE_BEHAVIOR,
        FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR,
        FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
        COMBINATION_GUARD_BEHAVIOR,
        FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
        FROZEN_REJECTION_BEHAVIOR,
    }
)
_NON_SEMANTIC_V1_PATHS = frozenset(
    {
        "runtime.output_root",
        "runtime.temp_root",
        "runtime.cache_root",
    }
)
_V1_RETAINED_SWITCH_PATHS = frozenset(
    {
        "identity.run_intent", "identity.formal_eligible", "runtime.offline",
        "runtime.device", "runtime.allow_dirty_worktree", "data.weights_exogenous",
        "data.common_mean_assumption", "loss.task_type", "loss.reduction",
        "loss.weighting", "batching.no_sync", "distributed.backend",
        "precision.compute_dtype", "precision.gradient_dtype",
        "precision.statistic_dtype", "precision.reference_dtype",
        "precision.quadrature_weight_dtype", "precision.path_accumulation_dtype",
        "precision.amp", "optimizer.type",
        "optimizer.fused", "optimizer.foreach", "logging.event_format",
        "logging.tensorboard", "checkpoint.two_phase_commit",
        "importance.estimator_name", "importance.clip_mode",
        "importance.require_decision_for_formal", "path_integration.enabled",
        "pruning.enabled", "pruning.strategy", "pruning.scope",
    }
)
_V2_RETAINED_SWITCH_PATHS = frozenset(
    {
        *(f"base_config.{path}" for path in _V1_RETAINED_SWITCH_PATHS),
        "execution.runner_kind", "execution.dry_run", "execution.fail_on_blocked",
        "training.deterministic_algorithms", "scheduler.kind",
        "data_loader.persistent_workers", "data_loader.drop_last",
        "data_loader.cursor_policy", "providers.kind", "providers.task_type",
        "providers.task_name", "providers.local_files_only",
        "providers.trust_remote_code", "evaluation.enabled",
        "evaluation.save_predictions", "profiling.enabled",
        "profiling.capture_memory", "profiling.capture_throughput",
        "profiling.capture_communication", "profiling.synchronize_device",
        "checkpoint_schedule.save_on_phase_end",
        "checkpoint_schedule.save_optimizer", "checkpoint_schedule.save_rng",
        "checkpoint_schedule.save_data_state", "precision_runtime.autocast_enabled",
        "precision_runtime.autocast_dtype", "precision_runtime.grad_scaler_enabled",
        "precision_runtime.global_found_inf_reduce", "optimizer_runtime.amsgrad",
        "optimizer_runtime.nesterov", "optimizer_runtime.maximize",
        "optimizer_runtime.capturable", "optimizer_runtime.differentiable",
        "launcher.kind", "launcher.backend", "launcher.init_method",
        "orchestration.paired_design.enabled", "orchestration.paired_design.design",
        "orchestration.paired_design.budget_unit", "recovery.mode",
        "recovery.safe_boundary", "artifacts.publish_partial",
    }
)
_DIGEST_LENGTH = 64


class Stage1ConfigCoverageError(ValueError):
    """The S1.2 configuration coverage manifest is unsafe to consume."""


def _require_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _DIGEST_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Stage1ConfigCoverageError(f"{field} must be a lowercase SHA-256")
    return value


def _expected_contracts() -> dict[str, dict[str, Any]]:
    v1_payload = config_schema_contract_payload()
    v2_payload = config_v2_schema_contract_payload()
    return {
        "resolved-config-v1": {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "schema_hash": config_schema_contract_hash(),
            "shared_schema_hashes": {
                "resolved-config-v1": v1_payload["shared_schema"]["sha256"],
            },
            "field_paths": public_config_field_paths(),
            "test_id": V1_CONFIG_FIELD_BEHAVIOR_TEST_ID,
        },
        "resolved-config-v2": {
            "schema_version": CONFIG_V2_SCHEMA_VERSION,
            "schema_hash": config_v2_schema_contract_hash(),
            "shared_schema_hashes": {
                family: identity["sha256"]
                for family, identity in v2_payload["shared_schemas"].items()
            },
            "field_paths": public_config_v2_field_paths(),
            "test_id": V2_CONFIG_FIELD_BEHAVIOR_TEST_ID,
        },
    }


def validate_config_field_behavior_coverage(
    value: Mapping[str, Any],
    *,
    expected_schema_sha256: str,
) -> dict[str, Any]:
    """Strictly bind coverage records to the current public v1+v2 contracts."""

    required = {
        "schema_version",
        "schema_sha256",
        "config_contracts",
        "coverage_groups",
        "artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise Stage1ConfigCoverageError("coverage manifest field set is invalid")
    if value["schema_version"] != CONFIG_FIELD_BEHAVIOR_COVERAGE_SCHEMA_VERSION:
        raise Stage1ConfigCoverageError("coverage manifest schema_version is unsupported")
    observed_schema_sha = _require_digest(value["schema_sha256"], field="schema_sha256")
    if observed_schema_sha != _require_digest(
        expected_schema_sha256, field="expected_schema_sha256"
    ):
        raise Stage1ConfigCoverageError("coverage manifest schema_sha256 mismatch")

    expected_contracts = _expected_contracts()
    contracts = value["config_contracts"]
    if not isinstance(contracts, Mapping) or set(contracts) != set(expected_contracts):
        raise Stage1ConfigCoverageError("config_contracts must bind exactly v1 and v2")
    for family, expected in expected_contracts.items():
        contract = contracts[family]
        if not isinstance(contract, Mapping) or set(contract) != {
            "schema_version", "schema_hash", "shared_schema_hashes",
        }:
            raise Stage1ConfigCoverageError(f"config_contracts.{family} field set is invalid")
        if contract["schema_version"] != expected["schema_version"]:
            raise Stage1ConfigCoverageError(
                f"config_contracts.{family}.schema_version mismatch"
            )
        if contract["schema_hash"] != expected["schema_hash"]:
            raise Stage1ConfigCoverageError(
                f"config_contracts.{family}.schema_hash mismatch"
            )
        if contract["shared_schema_hashes"] != expected["shared_schema_hashes"]:
            raise Stage1ConfigCoverageError(
                f"config_contracts.{family}.shared_schema_hashes mismatch"
            )

    groups = value["coverage_groups"]
    if not isinstance(groups, list) or not groups:
        raise Stage1ConfigCoverageError("coverage_groups must be a non-empty array")
    covered: dict[str, list[str]] = {family: [] for family in expected_contracts}
    group_fields = {"config_family", "field_paths", "behavior_id", "test_id"}
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or set(group) != group_fields:
            raise Stage1ConfigCoverageError(f"coverage_groups[{index}] field set is invalid")
        family = group["config_family"]
        if family not in expected_contracts:
            raise Stage1ConfigCoverageError(f"coverage_groups[{index}].config_family is unsupported")
        paths = group["field_paths"]
        if (
            not isinstance(paths, list)
            or not paths
            or any(not isinstance(path, str) or not path for path in paths)
        ):
            raise Stage1ConfigCoverageError(
                f"coverage_groups[{index}].field_paths must be a non-empty string array"
            )
        behavior = group["behavior_id"]
        if behavior not in _SUPPORTED_BEHAVIORS:
            raise Stage1ConfigCoverageError(
                f"coverage_groups[{index}].behavior_id is unsupported"
            )
        if group["test_id"] != expected_contracts[family]["test_id"]:
            raise Stage1ConfigCoverageError(
                f"coverage_groups[{index}].test_id is not the frozen executable test"
            )
        for path in paths:
            nonsemantic = (
                family == "resolved-config-v1" and path in _NON_SEMANTIC_V1_PATHS
            ) or (
                family == "resolved-config-v2"
                and path.startswith("base_config.")
                and path.removeprefix("base_config.") in _NON_SEMANTIC_V1_PATHS
            ) or (
                family == "resolved-config-v2"
                and path
                in {
                    "execution.timeout_seconds",
                    "execution.max_attempts",
                    "execution.fail_on_blocked",
                    "providers.model_root_ref",
                    "providers.data_root_ref",
                    "providers.tokenizer_root_ref",
                    "launcher.init_ref",
                    "launcher.rendezvous_id",
                    "launcher.max_restarts",
                    "recovery.resume_ref",
                    "recovery.max_restarts",
                    "artifacts.output_dir",
                }
            )
            if nonsemantic:
                if behavior not in {
                    FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR,
                    FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
                    FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
                }:
                    raise Stage1ConfigCoverageError(
                        f"{path} must use the full-identity-only alternate behavior"
                    )
            elif behavior in {
                FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR,
                FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
                FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR,
            }:
                raise Stage1ConfigCoverageError(
                    f"{path} is not a full-identity-only public field"
                )
            covered[family].append(path)

    for family, expected in expected_contracts.items():
        paths = covered[family]
        if len(paths) != len(set(paths)):
            raise Stage1ConfigCoverageError(
                f"coverage_groups.field_paths for {family} must not repeat"
            )
        expected_paths = set(expected["field_paths"])
        if set(paths) != expected_paths:
            raise Stage1ConfigCoverageError(
                f"coverage_groups.field_paths for {family} must exactly match the public contract; "
                f"missing={sorted(expected_paths - set(paths))}, "
                f"extra={sorted(set(paths) - expected_paths)}"
            )
        switches = (
            _V1_RETAINED_SWITCH_PATHS
            if family == "resolved-config-v1"
            else _V2_RETAINED_SWITCH_PATHS
        )
        behavior_by_path = {
            path: group["behavior_id"]
            for group in groups
            if group["config_family"] == family
            for path in group["field_paths"]
        }
        for path in switches:
            behavior = behavior_by_path.get(path)
            if behavior not in {
                ACCEPTED_ALTERNATE_BEHAVIOR,
                COMBINATION_GUARD_BEHAVIOR,
                FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR,
                FROZEN_REJECTION_BEHAVIOR,
            }:
                raise Stage1ConfigCoverageError(
                    f"retained switch {path} cannot be identity-only"
                )

    body = dict(value)
    artifact_hash = _require_digest(body.pop("artifact_hash"), field="artifact_hash")
    if artifact_hash != canonical_json_hash(body):
        raise Stage1ConfigCoverageError("coverage manifest artifact_hash mismatch")
    return dict(value)


def load_config_field_behavior_coverage(
    path: str | Path,
    *,
    expected_schema_sha256: str,
) -> dict[str, Any]:
    """Load the canonical, hash-bound public configuration coverage manifest."""

    loaded = load_canonical_json(path)
    if not isinstance(loaded, dict):
        raise Stage1ConfigCoverageError("coverage manifest top level must be an object")
    return validate_config_field_behavior_coverage(
        loaded,
        expected_schema_sha256=expected_schema_sha256,
    )


def coverage_summary(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact, formal-evidence-safe summary of v1+v2 coverage."""

    groups = value["coverage_groups"]
    assert isinstance(groups, list)
    counts: dict[str, int] = Counter()
    behavior_counts: dict[str, int] = Counter()
    for group in groups:
        assert isinstance(group, Mapping)
        family = str(group["config_family"])
        paths = group["field_paths"]
        assert isinstance(paths, list)
        counts[family] += len(paths)
        behavior_counts[str(group["behavior_id"])] += len(paths)
    contracts = value["config_contracts"]
    assert isinstance(contracts, Mapping)
    return {
        "schema_version": CONFIG_FIELD_BEHAVIOR_COVERAGE_SCHEMA_VERSION,
        "artifact_hash": value["artifact_hash"],
        "config_contracts": dict(contracts),
        "covered_field_counts": dict(sorted(counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "full_identity_only_v1_paths": sorted(_NON_SEMANTIC_V1_PATHS),
        "scope": "compiled_component_behavior_or_frozen_guard",
    }


__all__ = [
    "ACCEPTED_ALTERNATE_BEHAVIOR",
    "COMBINATION_GUARD_BEHAVIOR",
    "CONFIG_FIELD_BEHAVIOR_COVERAGE_SCHEMA_VERSION",
    "FULL_IDENTITY_ONLY_ALTERNATE_BEHAVIOR",
    "FULL_IDENTITY_ONLY_COMPONENT_BEHAVIOR",
    "FULL_IDENTITY_ONLY_COMBINATION_GUARD_BEHAVIOR",
    "FROZEN_REJECTION_BEHAVIOR",
    "Stage1ConfigCoverageError",
    "V1_CONFIG_FIELD_BEHAVIOR_TEST_ID",
    "V2_CONFIG_FIELD_BEHAVIOR_TEST_ID",
    "coverage_summary",
    "load_config_field_behavior_coverage",
    "validate_config_field_behavior_coverage",
]
