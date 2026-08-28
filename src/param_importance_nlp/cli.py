from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import yaml

from .capacity import StorageBudget, check_launch_storage
from .git_guard import format_findings, scan_repository
from .storage import REQUIRED_DIRECTORIES, StorageLayout, run_storage_canary


def _emit(value: object) -> None:
    """以项目 canonical JSON 格式输出机器结果。"""

    from .contracts import canonical_json_bytes

    print(canonical_json_bytes(value).decode("utf-8").rstrip("\n"))


class _UniqueKeyLoader(yaml.SafeLoader):
    """PyYAML 默认覆盖重复键；配置合同要求重复键立即失败。"""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("CONFIG_YAML_KEY_NOT_STRING")
        if key in result:
            raise ValueError(f"CONFIG_YAML_DUPLICATE_KEY:{key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def _load_mapping(path: str | Path) -> dict[str, Any]:
    """读取严格 JSON 或 YAML 配置层，返回普通 string-keyed mapping。"""

    from .contracts import ensure_json_object, load_canonical_json

    target = Path(path)
    if target.suffix.casefold() == ".json":
        return dict(ensure_json_object(load_canonical_json(target), field=str(target)))
    raw = target.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError(f"CONFIG_YAML_BOM_FORBIDDEN:{target}")
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"CONFIG_YAML_INVALID_UTF8:{target}") from exc
    value = yaml.load(text, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"CONFIG_ROOT_NOT_OBJECT:{target}")
    # canonical encoder 同时拒绝 YAML 的 NaN/Infinity、日期对象及未知 scalar。
    from .contracts import canonical_json_bytes

    canonical_json_bytes(value)
    return value


def _load_strict_json_document(path: str | Path) -> dict[str, Any]:
    """读取允许格式化空白、但仍满足安全边界的 JSON 文档。

    JSON Schema 是供人审阅的仓库源文档，通常使用缩进与换行，因此不能调用
    :func:`load_canonical_json` 要求唯一字节表示。这里仍然复用严格解析器，拒绝
    UTF-8 BOM、重复键、``NaN``/``Infinity`` 和非 UTF-8 输入。该入口只用于识别
    schema 源文档；普通内部 artifact 仍由 ``_load_mapping`` 执行 canonical 校验。
    """

    from .contracts import ensure_json_object, loads_strict_json

    target = Path(path)
    return dict(
        ensure_json_object(
            loads_strict_json(target.read_bytes()),
            field=str(target),
        )
    )


_PROJECT_SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
_PROJECT_SCHEMA_ID_PREFIX = "https://parameter-importance.invalid/schemas/"
_SUPPORTED_SCHEMA_ROOT_KEYS = frozenset(
    {
        "$schema",
        "$id",
        "$defs",
        "$ref",
        "$comment",
        "title",
        "description",
        "type",
        "additionalProperties",
        "unevaluatedProperties",
        "required",
        "properties",
        "patternProperties",
        "allOf",
        "anyOf",
        "oneOf",
        "not",
        "if",
        "then",
        "else",
    }
)


def _validate_project_json_schema(value: Mapping[str, Any]) -> None:
    """校验仓库 JSON Schema 文档的身份元数据与顶层结构。

    本项目故意不引入 ``jsonschema`` 运行时依赖；本函数也不冒充完整的 Draft
    2020-12 实例校验器。它冻结的是 schema *源文档* 的最小可信外壳：方言、项目
    ``$id``、标题、根对象结构以及顶层关键词必须明确有效。这样不能仅靠伪造
    ``{"$schema": ..., "type": "object"}`` 绕过 artifact dispatcher。
    """

    unknown = set(value).difference(_SUPPORTED_SCHEMA_ROOT_KEYS)
    if unknown:
        raise ValueError(f"JSON_SCHEMA_ROOT_FIELDS_UNKNOWN:{sorted(unknown)}")
    if value.get("$schema") != _PROJECT_SCHEMA_URI:
        raise ValueError("JSON_SCHEMA_DIALECT_UNSUPPORTED")

    schema_id = value.get("$id")
    if not isinstance(schema_id, str) or not schema_id.startswith(
        _PROJECT_SCHEMA_ID_PREFIX
    ):
        raise ValueError("JSON_SCHEMA_PROJECT_ID_INVALID")
    relative_id = schema_id[len(_PROJECT_SCHEMA_ID_PREFIX) :]
    if (
        not relative_id
        or not relative_id.endswith(".json")
        or "\\" in relative_id
        or "?" in relative_id
        or "#" in relative_id
        or any(part in {"", ".", ".."} for part in relative_id.split("/"))
    ):
        raise ValueError("JSON_SCHEMA_PROJECT_ID_INVALID")

    title = value.get("title")
    if not isinstance(title, str) or not title.strip():
        raise ValueError("JSON_SCHEMA_TITLE_INVALID")

    root_type = value.get("type")
    compositions = ("allOf", "anyOf", "oneOf")
    has_composition = any(key in value for key in compositions)
    if root_type is not None and root_type != "object":
        raise ValueError("JSON_SCHEMA_ROOT_TYPE_MUST_BE_OBJECT")
    if root_type is None and not has_composition and "$ref" not in value:
        raise ValueError("JSON_SCHEMA_ROOT_STRUCTURE_MISSING")

    for field in ("properties", "patternProperties", "$defs"):
        member = value.get(field)
        if member is not None:
            if not isinstance(member, Mapping) or not all(
                isinstance(key, str)
                and isinstance(item, (Mapping, bool))
                for key, item in member.items()
            ):
                raise ValueError(f"JSON_SCHEMA_{field.upper()}_INVALID")

    required = value.get("required")
    if required is not None:
        if (
            not isinstance(required, list)
            or not all(isinstance(item, str) and item for item in required)
            or len(set(required)) != len(required)
        ):
            raise ValueError("JSON_SCHEMA_REQUIRED_INVALID")

    for field in compositions:
        member = value.get(field)
        if member is not None and (
            not isinstance(member, list)
            or not member
            or not all(isinstance(item, Mapping) for item in member)
        ):
            raise ValueError(f"JSON_SCHEMA_{field.upper()}_INVALID")

    additional = value.get("additionalProperties")
    if additional is not None and not isinstance(additional, (bool, Mapping)):
        raise ValueError("JSON_SCHEMA_ADDITIONAL_PROPERTIES_INVALID")


def _git_guard(arguments: argparse.Namespace) -> int:
    findings = scan_repository(
        arguments.repo,
        max_bytes=arguments.max_bytes,
        allowlist=arguments.allow,
    )
    if findings:
        print(format_findings(findings))
        return 1
    print("git-guard: PASS")
    return 0


def _storage_check(arguments: argparse.Namespace) -> int:
    layout = StorageLayout.from_value(arguments.data_root)
    failures = layout.validate(require_writable=arguments.require_writable)
    report: dict[str, object] = {
        "schema_version": "stage0.storage-check.v1",
        "data_root": str(layout.root),
        "directories": list(REQUIRED_DIRECTORIES),
        "failures": failures,
        "canaries": [],
    }
    if not failures and arguments.canary:
        report["canaries"] = [
            run_storage_canary(layout, name) for name in REQUIRED_DIRECTORIES
        ]
    report["ok"] = not failures and all(
        item["ok"] for item in report["canaries"]  # type: ignore[index,union-attr]
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _storage_budget_check(arguments: argparse.Namespace) -> int:
    budget = StorageBudget.from_expected(arguments.name, arguments.expected_new_bytes)
    report = {
        "schema_version": "stage0.storage-launch-check.v1",
        "budget": budget.as_dict(),
        "measurement": check_launch_storage(
            data_root=arguments.data_root,
            root_filesystem=arguments.root_filesystem,
            budget=budget,
            root_minimum_free_bytes=arguments.root_minimum_free_bytes,
        ),
    }
    report["ok"] = report["measurement"]["ok"]  # type: ignore[index]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _config_resolve(arguments: argparse.Namespace) -> int:
    from .contracts import ResolvedConfig, write_canonical_json

    layers = [_load_mapping(path) for path in arguments.inputs]
    resolved = ResolvedConfig.resolve(*layers)
    if arguments.output is not None:
        write_canonical_json(arguments.output, resolved.to_dict())
    _emit(
        {
            "schema_version": "cli.config-resolve.v1",
            "config_hash": resolved.config_hash,
            "full_hash": resolved.full_hash,
            "formal_eligible": resolved.formal_eligible,
            "output": None if arguments.output is None else str(arguments.output),
            "resolved": resolved.to_dict() if arguments.output is None else None,
        }
    )
    return 0


def _config_resolve_v2(arguments: argparse.Namespace) -> int:
    """把 v1 科学配置层与 v2 执行 overrides 解析成正式 task 配置。"""

    from .contracts import ResolvedConfig, ResolvedConfigV2, write_canonical_json

    layers = [_load_mapping(path) for path in arguments.inputs]
    base = ResolvedConfig.resolve(*layers)
    overrides = None if arguments.overrides is None else _load_mapping(arguments.overrides)
    resolved = ResolvedConfigV2.resolve(
        base,
        task_id=arguments.task_id,
        overrides=overrides,
    )
    if arguments.output is not None:
        write_canonical_json(arguments.output, resolved.to_dict())
    _emit(
        {
            "schema_version": "cli.config-resolve-v2.v1",
            "task_id": resolved.task_id,
            "config_hash": resolved.config_hash,
            "full_hash": resolved.full_hash,
            "formal_eligible": resolved.formal_eligible,
            "output": None if arguments.output is None else str(arguments.output),
            "resolved": resolved.to_dict() if arguments.output is None else None,
        }
    )
    return 0


def _resolved_from_path(path: Path) -> Any:
    from .contracts import ResolvedConfig

    return ResolvedConfig.from_mapping(_load_mapping(path))


def _config_diff(arguments: argparse.Namespace) -> int:
    from .contracts import diff_configs

    left = _resolved_from_path(arguments.left)
    right = _resolved_from_path(arguments.right)
    differences = [item.to_dict() for item in diff_configs(left, right)]
    _emit(
        {
            "schema_version": "cli.config-diff.v1",
            "left_config_hash": left.config_hash,
            "right_config_hash": right.config_hash,
            "difference_count": len(differences),
            "differences": differences,
        }
    )
    return 0


def _validate_known_artifact(value: dict[str, Any]) -> tuple[str, str | None]:
    """按 schema 分派严格 Python 合同验证，返回种类与内容 hash。"""

    from .contracts import (
        CONFIG_SECTIONS,
        ContractFreeze,
        GateRecord,
        LocalValidationRecord,
        ProvenanceRecord,
        ResolvedConfig,
        RunIdentity,
        SeedPlan,
        canonical_json_hash,
    )

    schema = value.get("schema_version")
    if schema == "training-endpoint-capture-plan-v1":
        expected = {
            "schema_version", "plan_id", "selected_steps",
            "include_checkpoint_steps", "scope", "formal_eligible",
            "qualification_evidence_hash", "probe_plan_ref", "artifact_hash",
        }
        if frozenset(value) not in {
            frozenset(expected),
            frozenset(expected | {"endpoint_metadata"}),
        }:
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_FIELDS_MISMATCH")
        artifact_digest = canonical_json_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        )
        if value["artifact_hash"] != artifact_digest:
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_HASH_MISMATCH")
        scope = value["scope"]
        if scope not in {"local_fixture", "pilot", "formal"} or value["formal_eligible"] != (
            scope == "formal"
        ):
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_SCOPE_MISMATCH")
        evidence_hash = value["qualification_evidence_hash"]
        if (scope == "formal") != (
            isinstance(evidence_hash, str) and len(evidence_hash) == 64
        ):
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_EVIDENCE_MISMATCH")
        steps = value["selected_steps"]
        if not isinstance(steps, list) or len(steps) != len(set(steps)) or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in steps
        ):
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_STEPS_INVALID")
        if not steps and value["include_checkpoint_steps"] is not True:
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_EMPTY")
        metadata = value.get("endpoint_metadata")
        if scope in {"pilot", "formal"}:
            if not isinstance(metadata, Mapping) or not set(steps).issubset(
                {int(item) for item in metadata if isinstance(item, str) and item.isdecimal()}
            ):
                raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_METADATA_MISSING")
        if metadata is not None and (
            not isinstance(metadata, Mapping)
            or any(
                not isinstance(step, str)
                or not step.isdecimal()
                or int(step) <= 0
                or not isinstance(item, Mapping)
                or set(item) != {"model", "seed", "stage"}
                or item["model"] not in {"14M", "31M"}
                or isinstance(item["seed"], bool)
                or not isinstance(item["seed"], int)
                or item["seed"] < 0
                or item["stage"] not in {"early", "middle", "late"}
                for step, item in metadata.items()
            )
        ):
            raise ValueError("TRAINING_ENDPOINT_CAPTURE_PLAN_METADATA_INVALID")
        return "training_endpoint_capture_plan", artifact_digest
    if schema == "stage3-probe-plan-v1":
        expected = {
            "schema_version", "panel_id", "endpoint_digest", "entries",
            "minimum_formal_probes", "execution_evidence_hash", "scope",
            "formal_eligible", "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("STAGE3_PROBE_PLAN_FIELDS_MISMATCH")
        artifact_digest = canonical_json_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        )
        if value["artifact_hash"] != artifact_digest:
            raise ValueError("STAGE3_PROBE_PLAN_HASH_MISMATCH")
        scope = value["scope"]
        if scope not in {"local_fixture", "pilot", "formal"} or value["formal_eligible"] != (
            scope == "formal"
        ):
            raise ValueError("STAGE3_PROBE_PLAN_SCOPE_MISMATCH")
        if not isinstance(value["entries"], list) or not value["entries"]:
            raise ValueError("STAGE3_PROBE_PLAN_ENTRIES_EMPTY")
        minimum = value["minimum_formal_probes"]
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= 0:
            raise ValueError("STAGE3_PROBE_PLAN_MINIMUM_INVALID")
        probe_ids: set[str] = set()
        sample_ids: set[str | int] = set()
        loss_contract_hashes: set[str] = set()
        formal_count = 0
        entry_fields = {
            "role",
            "probe_id",
            "sample_ids",
            "content_hash",
            "loss_contract_hash",
            "effective_weight_unit",
            "metadata",
        }
        for index, raw_entry in enumerate(value["entries"]):
            if not isinstance(raw_entry, Mapping) or set(raw_entry) != entry_fields:
                raise ValueError(f"STAGE3_PROBE_PLAN_ENTRY_FIELDS_MISMATCH:{index}")
            role = raw_entry["role"]
            probe_id = raw_entry["probe_id"]
            samples = raw_entry["sample_ids"]
            if role not in {"pilot", "formal", "replay"}:
                raise ValueError(f"STAGE3_PROBE_PLAN_ENTRY_ROLE_INVALID:{index}")
            if not isinstance(probe_id, str) or not probe_id or probe_id in probe_ids:
                raise ValueError(f"STAGE3_PROBE_PLAN_PROBE_ID_INVALID:{index}")
            if not isinstance(samples, list) or not samples or any(
                isinstance(item, bool) or not isinstance(item, (str, int))
                for item in samples
            ):
                raise ValueError(f"STAGE3_PROBE_PLAN_SAMPLE_IDS_INVALID:{index}")
            if len(samples) != len(set(samples)):
                raise ValueError(f"STAGE3_PROBE_PLAN_SAMPLE_IDS_INVALID:{index}")
            overlap = sample_ids.intersection(samples)
            if overlap:
                raise ValueError(f"STAGE3_PROBE_PLAN_SAMPLE_OVERLAP:{index}")
            for hash_field in ("content_hash", "loss_contract_hash"):
                field_digest = raw_entry[hash_field]
                if (
                    not isinstance(field_digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", field_digest) is None
                ):
                    raise ValueError(
                        f"STAGE3_PROBE_PLAN_ENTRY_HASH_INVALID:{index}:{hash_field}"
                    )
            if (
                not isinstance(raw_entry["effective_weight_unit"], str)
                or not raw_entry["effective_weight_unit"]
                or not isinstance(raw_entry["metadata"], Mapping)
            ):
                raise ValueError(f"STAGE3_PROBE_PLAN_ENTRY_METADATA_INVALID:{index}")
            probe_ids.add(probe_id)
            sample_ids.update(samples)
            loss_contract_hashes.add(raw_entry["loss_contract_hash"])
            formal_count += int(role == "formal")
        if len(loss_contract_hashes) != 1:
            raise ValueError("STAGE3_PROBE_PLAN_LOSS_CONTRACT_DRIFT")
        if scope == "pilot" and (
            len(value["entries"]) != 2
            or any(item["role"] != "pilot" for item in value["entries"])
        ):
            raise ValueError("STAGE3_PROBE_PLAN_PILOT_PROBE_COUNT_INVALID")
        if scope == "formal" and (minimum < 3 or formal_count < minimum):
            raise ValueError("STAGE3_PROBE_PLAN_FORMAL_PROBE_COUNT_LT_THREE")
        return "stage3_probe_plan", artifact_digest
    if schema == "stage3-formal-pilot-plan-v1":
        from .experiments import QuadratureThresholds

        expected = {
            "schema_version", "plan_id", "scope", "candidate_rules",
            "required_unit_ids", "unit_strata", "thresholds", "execution_evidence_hash",
            "formal_eligible", "artifact_hash",
        }
        index_fields = {
            "production_unit_index_ref", "production_unit_index_hash"
        }
        plan_fields = {"plan_kind", "production_unit_index_scope"}
        if (
            set(value) != expected
            and set(value) != expected | index_fields | plan_fields
        ):
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_FIELDS_MISMATCH")
        digest = canonical_json_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        )
        if value["artifact_hash"] != digest:
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_HASH_MISMATCH")
        candidates = value["candidate_rules"]
        units = value["required_unit_ids"]
        if not all(
            isinstance(item, list)
            and item
            and all(isinstance(child, str) and child for child in item)
            and len(item) == len(set(item))
            for item in (candidates, units)
        ):
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_ARRAYS_INVALID")
        unit_strata = value["unit_strata"]
        if (
            not isinstance(unit_strata, Mapping)
            or set(unit_strata) != set(units)
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"model", "stage", "update", "probe"}
                or any(not isinstance(child, str) or not child for child in item.values())
                for item in unit_strata.values()
            )
        ):
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_UNIT_STRATA_INVALID")
        thresholds = value["thresholds"]
        if not isinstance(thresholds, Mapping):
            raise TypeError("STAGE3_FORMAL_PILOT_PLAN_THRESHOLDS_INVALID")
        QuadratureThresholds(  # type: ignore[arg-type]
            **dict(thresholds)
        ).require_formal_contract()
        if value["scope"] != "formal" or value["formal_eligible"] is not True:
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SCOPE_INVALID")
        if set(value) == expected | index_fields | plan_fields:
            index_ref = value["production_unit_index_ref"]
            index_hash = value["production_unit_index_hash"]
            plan_kind = value["plan_kind"]
            index_scope = value["production_unit_index_scope"]
            if (
                not isinstance(index_ref, str)
                or not index_ref
                or not isinstance(index_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", index_hash) is None
                or plan_kind not in {"pilot", "matrix"}
                or index_scope != ("pilot" if plan_kind == "pilot" else "formal")
            ):
                raise ValueError("STAGE3_FORMAL_PILOT_PLAN_INDEX_BINDING_INVALID")
            expected_unit_count = 12 if plan_kind == "pilot" else 99
            if len(units) != expected_unit_count:
                raise ValueError(
                    "STAGE3_FORMAL_PILOT_PLAN_UNIT_COVERAGE_INVALID:"
                    f"expected={expected_unit_count},actual={len(units)}"
                )
        elif isinstance(units, list) and len(units) in {12, 99}:
            raise ValueError(
                "STAGE3_FORMAL_PILOT_PLAN_INDEX_REQUIRED_FOR_PRODUCTION_UNITS"
            )
        evidence_hash = value["execution_evidence_hash"]
        if not isinstance(evidence_hash, str) or len(evidence_hash) != 64:
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_EVIDENCE_HASH_INVALID")
        return "stage3_formal_pilot_plan", digest
    if schema == "endpoint-commit-v1":
        expected = {
            "schema_version", "endpoint_id", "optimizer_step", "endpoint_digest",
            "object_ref", "object_sha256", "scope", "formal_eligible",
            "qualification_evidence_hash", "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("ENDPOINT_COMMIT_FIELDS_MISMATCH")
        digest = canonical_json_hash(
            {key: item for key, item in value.items() if key != "artifact_hash"}
        )
        if value["artifact_hash"] != digest:
            raise ValueError("ENDPOINT_COMMIT_HASH_MISMATCH")
        return "training_endpoint_commit", digest
    if schema == "resolved-config-v2":
        from .contracts import ResolvedConfigV2

        config_v2 = ResolvedConfigV2.from_mapping(value)
        return "resolved_config_v2", config_v2.config_hash
    if set(CONFIG_SECTIONS).issubset(value):
        config = ResolvedConfig.from_mapping(value)
        return "resolved_config", config.config_hash
    if schema == "task-catalog-v2":
        from .contracts import TaskCatalog

        catalog = TaskCatalog.from_mapping(value)
        return "task_catalog", catalog.catalog_hash
    if schema == "task-definition-v2":
        from .contracts import TaskDefinition

        definition = TaskDefinition.from_mapping(value)
        return "task_definition", canonical_json_hash(definition.to_dict())
    if schema == "task-run-result-v2":
        from .runtime import TaskRunResult

        result = TaskRunResult.from_mapping(value)
        return "task_run_result", result.result_hash
    if schema == "task-run-spec-v1":
        from .runtime import TaskRunSpec

        spec = TaskRunSpec.from_mapping(value)
        return "task_run_spec", str(spec.to_dict()["spec_hash"])
    if schema == "task-runtime-environment-v1":
        from .runtime import TaskRuntimeEnvironment

        environment = TaskRuntimeEnvironment.from_mapping(value)
        return "task_runtime_environment", environment.environment_hash
    if schema == "runtime-capability-evidence-v1":
        from .contracts import RuntimeCapabilityEvidence

        evidence = RuntimeCapabilityEvidence.from_mapping(value)
        return "runtime_capability_evidence", evidence.artifact_hash
    if schema == "task-status-snapshot-v1":
        from .runtime import TaskStatusSnapshot

        snapshot = TaskStatusSnapshot.from_mapping(value)
        return "task_status_snapshot", snapshot.snapshot_hash
    if schema == "task-replay-record-v1":
        from .runtime import TaskReplayRecord

        replay = TaskReplayRecord.from_mapping(value)
        return "task_replay_record", replay.replay_hash
    if schema == "task-finalization-record-v1":
        from .runtime import TaskFinalizationRecord

        finalization = TaskFinalizationRecord.from_mapping(value)
        return "task_finalization_record", finalization.finalization_hash
    if schema == "artifact-review-v1":
        from .contracts import ArtifactReview

        review = ArtifactReview.from_mapping(value)
        return "artifact_review", review.review_hash
    if schema == "artifact-approval-v1":
        from .contracts import ArtifactApproval

        approval = ArtifactApproval.from_mapping(value)
        return "artifact_approval", approval.approval_hash
    if schema == "task-output-artifact-v1":
        expected = {
            "schema_version", "task_id", "artifact_kind", "config_hash", "run_intent",
            "formal_eligible", "source_refs", "payload", "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("TASK_OUTPUT_ARTIFACT_FIELDS_INVALID")
        payload = {key: item for key, item in value.items() if key != "artifact_hash"}
        digest = canonical_json_hash(payload)
        if value["artifact_hash"] != digest:
            raise ValueError("TASK_OUTPUT_ARTIFACT_HASH_MISMATCH")
        if value["formal_eligible"] != (value["run_intent"] == "formal"):
            raise ValueError("TASK_OUTPUT_ARTIFACT_ELIGIBILITY_MISMATCH")
        return "task_output_artifact", digest
    if schema == "task-output-commit-v1":
        expected = {
            "schema_version", "task_id", "artifact_kind", "config_hash", "artifact_hash",
            "object_ref", "formal_eligible",
        }
        if set(value) != expected:
            raise ValueError("TASK_OUTPUT_COMMIT_FIELDS_INVALID")
        for field in ("config_hash", "artifact_hash"):
            item = value[field]
            if not isinstance(item, str) or len(item) != 64:
                raise ValueError(f"TASK_OUTPUT_COMMIT_HASH_INVALID:{field}")
        return "task_output_commit", str(value["artifact_hash"])
    if schema == "importance-trajectory-v1":
        from .runtime import ImportanceTrajectory

        trajectory = ImportanceTrajectory.from_mapping(value)
        return "importance_trajectory", str(trajectory.to_dict()["artifact_hash"])
    if schema == "estimator-study-result-v1":
        from .experiments import EstimatorStudyResult

        result = EstimatorStudyResult.from_mapping(value)
        return "estimator_study_result", str(result.to_dict()["artifact_hash"])
    if schema == "path-study-result-v1":
        from .experiments import PathStudyResult

        result = PathStudyResult.from_mapping(value)
        return "path_study_result", str(result.to_dict()["artifact_hash"])
    if schema == "derived-table-index-v1":
        from .analysis import DerivedTableIndex

        index = DerivedTableIndex.from_mapping(value)
        return "derived_table_index", str(index.to_dict()["artifact_hash"])
    if schema == "stage2-draw-stream-manifest-v1":
        from .experiments import DrawStreamManifest

        manifest = DrawStreamManifest.from_manifest(value)
        return "stage2_draw_stream_manifest", manifest.replay_hash
    if schema == "stage2-repetition-mapping-v1":
        from .experiments import RepetitionMapping

        mapping = RepetitionMapping.from_manifest(value)
        return "stage2_repetition_mapping", mapping.digest
    if schema == "stage2-checkpoint-manifest-v1":
        from .experiments import CheckpointRecord

        record = CheckpointRecord.from_mapping(value)
        return "stage2_checkpoint_manifest", canonical_json_hash(record.to_dict())
    if schema == "stage2-data-range-manifest-v1":
        from .experiments import DataRangeManifest

        manifest = DataRangeManifest.from_mapping(value)
        return "stage2_data_range_manifest", manifest.digest
    if schema == "stage2-asset-resolution-v1":
        from .experiments import AssetResolutionManifest

        manifest = AssetResolutionManifest.from_mapping(value)
        return "stage2_asset_resolution", manifest.digest
    if schema == "stage3-gate-evaluation-v1":
        from .experiments.stage3_gate import Stage3GateEvaluation

        evaluation = Stage3GateEvaluation.from_mapping(value)
        return "stage3_gate_evaluation", evaluation.artifact_hash
    if isinstance(schema, str) and (
        schema.startswith("stage2-") or schema.startswith("stage3-")
    ):
        from .contracts import validate_stage23_artifact

        artifact = validate_stage23_artifact(value)
        return artifact.kind, artifact.artifact_hash
    loaders: dict[str, tuple[str, Any]] = {
        "gate-record-v1": ("gate_record", GateRecord.from_mapping),
        "local-validation-record-v1": (
            "local_validation_record",
            LocalValidationRecord.from_mapping,
        ),
        "contract-freeze-v1": ("contract_freeze", ContractFreeze.from_mapping),
        "run-identity-v1": ("run_identity", RunIdentity.from_mapping),
        "seed-plan-v1": ("seed_plan", SeedPlan.from_mapping),
        "provenance-record-v1": ("provenance", ProvenanceRecord.from_mapping),
    }
    if schema in loaders:
        kind, loader = loaders[schema]
        artifact = loader(value)
        return kind, artifact.artifact_hash
    if schema == "estimator-decision-v1":
        from .experiments import EstimatorDecision

        decision = EstimatorDecision.from_mapping(value)
        return "estimator_decision", decision.artifact_hash
    if schema == "sampling-plan-v1":
        from .experiments import SamplingPlan

        plan = SamplingPlan.from_mapping(value)
        return "sampling_plan", plan.digest
    if schema == "reference-result-v1":
        from .contracts import validate_reference_result_artifact

        artifact = validate_reference_result_artifact(value)
        return artifact.kind, artifact.artifact_hash
    if schema == "quadrature-decision-v1":
        from .experiments import QuadratureDecision

        decision = QuadratureDecision.from_mapping(value)
        return "quadrature_decision", decision.artifact_hash
    if schema == "quadrature-rule-v1":
        from .core import QuadratureRule

        rule = QuadratureRule.from_mapping(value)
        return "quadrature_rule", rule.artifact_hash
    if schema == "path-spec-v1":
        from .contracts import validate_path_spec_artifact

        artifact = validate_path_spec_artifact(value)
        return artifact.kind, artifact.artifact_hash
    if schema == "path-integral-result-v1":
        from .contracts import validate_path_integral_result_artifact

        artifact = validate_path_integral_result_artifact(value)
        return artifact.kind, artifact.artifact_hash
    if schema == "training-route-v1":
        from .experiments import TrainingRouteSpec

        route = TrainingRouteSpec.from_mapping(value)
        return "training_route", route.lineage_hash
    if schema == "pruning-plan-v1":
        from .core import PruningPlan

        plan = PruningPlan.from_mapping(value)
        return "pruning_plan", plan.digest
    if schema == "ablation-matrix-v1":
        from .experiments import AblationMatrix

        matrix = AblationMatrix.from_mapping(value)
        return "ablation_matrix", matrix.digest
    if schema == "ablation-matrix-declaration-v1":
        from .experiments import AblationMatrixDeclaration

        declaration = AblationMatrixDeclaration.from_mapping(value)
        return "ablation_matrix_declaration", declaration.artifact_hash
    if schema == "analysis-report-v1":
        from .analysis import AnalysisReport

        report = AnalysisReport.from_mapping(value)
        return "analysis_report", report.report_hash
    if schema == "analysis-chart-spec-v1":
        from .analysis import ChartSpec

        spec = ChartSpec.from_mapping(value)
        return "analysis_chart_spec", spec.spec_hash
    if schema == "analysis-chart-artifact-v1":
        from .analysis import ChartArtifact

        chart = ChartArtifact.from_mapping(value)
        return "analysis_chart_artifact", chart.artifact_hash
    if schema == "local-contract-freezes-v1":
        expected = {"schema_version", "scope", "formal_eligible", "freezes", "artifact_hash"}
        if set(value) != expected or value["scope"] != "local_fixture":
            raise ValueError("LOCAL_FREEZE_SET_FIELDS_OR_SCOPE_INVALID")
        if value["formal_eligible"] is not False or not isinstance(value["freezes"], list):
            raise ValueError("LOCAL_FREEZE_SET_ELIGIBILITY_INVALID")
        freezes = [ContractFreeze.from_mapping(item) for item in value["freezes"]]
        if any(item.formal_eligible for item in freezes):
            raise ValueError("LOCAL_FREEZE_SET_CONTAINS_FORMAL_FREEZE")
        payload = {key: item for key, item in value.items() if key != "artifact_hash"}
        digest = canonical_json_hash(payload)
        if value["artifact_hash"] != digest:
            raise ValueError("LOCAL_FREEZE_SET_HASH_MISMATCH")
        return "local_contract_freeze_set", digest
    if "$schema" in value:
        _validate_project_json_schema(value)
        return "json_schema", canonical_json_hash(value)
    raise ValueError(f"ARTIFACT_SCHEMA_UNSUPPORTED:{schema!r}")


def _artifact_validate(arguments: argparse.Namespace) -> int:
    target = Path(arguments.path)
    if target.is_dir():
        from .runtime import load_tensor_bundle

        _, identity = load_tensor_bundle(target)
        kind = "tensor_bundle"
        digest = identity.manifest_sha256
    else:
        # schema 是带缩进的仓库源文档；先用严格 JSON 文档 loader 识别它。若没有
        # ``$schema``，必须回到 canonical artifact loader，不能借此放宽内部边界。
        if target.suffix.casefold() == ".json":
            document = _load_strict_json_document(target)
            value = document if "$schema" in document else _load_mapping(target)
        else:
            value = _load_mapping(target)
        kind, digest = _validate_known_artifact(value)
    _emit(
        {
            "schema_version": "cli.artifact-validation.v1",
            "path": str(target),
            "kind": kind,
            "artifact_hash": digest,
            "valid": True,
        }
    )
    return 0


def _contract_validate(arguments: argparse.Namespace) -> int:
    # 当前合同与 artifact 都由同一严格 schema dispatcher 校验；独立命令保留
    # 语义清晰的用户入口，并允许未来加入 schema migration 检查。
    return _artifact_validate(arguments)


def _gate_summary(arguments: argparse.Namespace) -> int:
    from .contracts import GateRecord, LocalValidationRecord

    formal: list[Any] = []
    local: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for path in arguments.paths:
        value = _load_mapping(path)
        schema = value.get("schema_version")
        if schema == "gate-record-v1":
            record = GateRecord.from_mapping(value)
            key = ("formal", record.gate_id)
            formal.append(record)
        elif schema == "local-validation-record-v1":
            record = LocalValidationRecord.from_mapping(value)
            key = ("local_fixture", record.validation_id)
            local.append(record)
        else:
            raise ValueError(f"GATE_SUMMARY_UNSUPPORTED_ARTIFACT:{path}")
        if key in seen:
            raise ValueError(f"GATE_SUMMARY_DUPLICATE_ID:{key[1]}")
        seen.add(key)
    formal_counts: dict[str, int] = {}
    for record in formal:
        status = record.effective_status().value
        formal_counts[status] = formal_counts.get(status, 0) + 1
    local_counts: dict[str, int] = {}
    for record in local:
        status = record.status.value
        local_counts[status] = local_counts.get(status, 0) + 1
    _emit(
        {
            "schema_version": "cli.gate-summary.v1",
            "formal": {
                "count": len(formal),
                "status_counts": dict(sorted(formal_counts.items())),
                "all_pass": bool(formal)
                and all(item.effective_status().value == "PASS" for item in formal),
            },
            "local_validation": {
                "count": len(local),
                "status_counts": dict(sorted(local_counts.items())),
                "formal_eligible": False,
            },
        }
    )
    return 0


def _formal_readiness(arguments: argparse.Namespace) -> int:
    """验证正式入口所需的 freeze、decision 与 Gate 证据链。

    该命令只做资格判定，不启动训练。缺任何文件时不推断、不降级为 fixture，
    而是输出全部稳定 reason code 并返回退出码 3。这样调度脚本可以把“合同无效”
    （退出码 2）与“证据尚未具备”（退出码 3）区分开。
    """

    from .contracts import evaluate_formal_readiness

    config = _resolved_from_path(arguments.config)
    freezes = [_load_mapping(path) for path in arguments.freeze]
    gates = [_load_mapping(path) for path in arguments.gate]
    decision = None if arguments.decision is None else _load_mapping(arguments.decision)
    # Stage 4+ formal 入口固定依赖 0--3；命令行只能追加更晚阶段，不能通过
    # ``--required-stage 3`` 把默认前置链缩窄为单一阶段。
    required_stages = tuple(
        dict.fromkeys((0, 1, 2, 3, *(arguments.required_stage or ())))
    )
    readiness = evaluate_formal_readiness(
        config,
        freezes=freezes,
        estimator_decision=decision,
        gate_records=gates,
        required_stages=required_stages,
        required_gate_ids=tuple(arguments.required_gate),
    )
    _emit(readiness.to_dict())
    return 0 if readiness.formal_eligible else 3


def _report_build(arguments: argparse.Namespace) -> int:
    from .analysis import (
        AnalysisReportBuilder,
        FrozenSourceTable,
        effective_parameter_count,
        entropy,
        error_summary,
        gini,
        hhi,
        pearson,
        spearman,
        top_q_mass,
    )
    from .contracts import write_canonical_json

    value = _load_mapping(arguments.source)
    required_source_fields = {
        "name",
        "schema_version",
        "rows",
        "content_hash",
        "frozen",
    }
    if set(value) != required_source_fields:
        raise ValueError(
            "REPORT_SOURCE_FIELDS_MISMATCH:"
            f"missing={sorted(required_source_fields-set(value))}:"
            f"extra={sorted(set(value)-required_source_fields)}"
        )
    rows = value.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("REPORT_SOURCE_ROWS_INVALID")
    if value["frozen"] is not True or not isinstance(value["content_hash"], str):
        raise ValueError("REPORT_SOURCE_MUST_BE_HASH_BOUND_AND_FROZEN")
    # 消费入口只能验证上游冻结源，不能替任意 rows 现场“自封”为冻结结果。
    table = FrozenSourceTable.from_mapping(value)
    try:
        observed = [float(row[arguments.value_field]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"REPORT_VALUE_FIELD_INVALID:{arguments.value_field}") from exc
    builder = AnalysisReportBuilder(report_id=arguments.report_id)
    builder.add_source(table)
    for name, metric in {
        "gini": gini(observed),
        "entropy": entropy(observed),
        "hhi": hhi(observed),
        "effective_parameter_count": effective_parameter_count(observed),
        "top_q_mass": top_q_mass(observed, arguments.top_q),
    }.items():
        builder.add_metric(
            name,
            metric,
            source=table,
            derivation_id=f"stage9.{name}.v1",
            input_columns=(arguments.value_field,),
        )
    if arguments.reference_field is not None:
        try:
            reference = [float(row[arguments.reference_field]) for row in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"REPORT_REFERENCE_FIELD_INVALID:{arguments.reference_field}"
            ) from exc
        for name, metric in error_summary(observed, reference).items():
            builder.add_metric(
                name,
                metric,
                source=table,
                derivation_id=f"stage9.{name}.v1",
                input_columns=(arguments.value_field, arguments.reference_field),
            )
        builder.add_metric(
            "pearson",
            pearson(observed, reference),
            source=table,
            derivation_id="stage9.pearson.v1",
            input_columns=(arguments.value_field, arguments.reference_field),
        )
        builder.add_metric(
            "spearman",
            spearman(observed, reference),
            source=table,
            derivation_id="stage9.spearman.v1",
            input_columns=(arguments.value_field, arguments.reference_field),
        )
    report = builder.build(
        metadata={
            "value_field": arguments.value_field,
            "reference_field": arguments.reference_field,
            "top_q": arguments.top_q,
        }
    )
    write_canonical_json(arguments.output_json, report.to_dict())
    arguments.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_markdown.write_text(report.render_markdown(), encoding="utf-8")
    _emit(
        {
            "schema_version": "cli.report-build.v1",
            "report_hash": report.report_hash,
            "source_hash": table.content_hash,
            "output_json": str(arguments.output_json),
            "output_markdown": str(arguments.output_markdown),
        }
    )
    return 0


def _local_fixture(arguments: argparse.Namespace) -> int:
    from .local_fixture import run_local_fixture

    report = run_local_fixture(
        config_path=arguments.config,
        output_dir=arguments.output_dir,
    )
    _emit(report)
    return 0


# ---------------------------------------------------------------------------
# v0.4 统一 task / asset / artifact review / gate 命令
# ---------------------------------------------------------------------------


def _load_v2_config(path: str | Path) -> Any:
    """严格读取 ResolvedConfigV2；task 正式入口不接受 v1 的隐式补全。"""

    from .contracts import ResolvedConfigV2

    return ResolvedConfigV2.from_mapping(_load_mapping(path))


def _load_task_environment(path: Path | None) -> Any:
    from .runtime import TaskRuntimeEnvironment

    if path is None:
        return TaskRuntimeEnvironment()
    return TaskRuntimeEnvironment.from_mapping(_load_mapping(path))


def _task_workspace_root(config: Any) -> Path:
    """Select the task workspace without inferring a formal DATA_ROOT."""

    run_intent = getattr(config, "run_intent", None)
    if run_intent == "local_fixture":
        return Path.cwd().resolve()
    if run_intent != "formal":
        raise ValueError("TASK_RUN_INTENT_INVALID")

    from .storage import require_data_root

    root = require_data_root()
    if not root.is_dir():
        raise RuntimeError("TASK_FORMAL_DATA_ROOT_MISSING_OR_NOT_DIRECTORY")
    return root


def _build_default_task_runtime(
    workspace_root: str | Path | None = None,
) -> Any:
    """延迟接入具体 runner 工厂，核心 CLI 本身不导入训练实现。

    ``experiments.task_runners`` 在功能波次尚未落地时可能不存在。只对“模块本身
    不存在”或“尚未导出工厂”提供空 runtime 回退；模块内部真实 ImportError、语法
    错误或工厂异常不能被吞掉，否则会把实现缺陷伪装成外部条件 BLOCKED。
    """

    from .runtime import TaskRuntime

    root = (
        Path.cwd().resolve()
        if workspace_root is None
        else Path(workspace_root).resolve()
    )

    module_name = "param_importance_nlp.experiments.task_runners"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        return TaskRuntime(workspace_root=root)
    factory = getattr(module, "build_default_task_runtime", None)
    if factory is None:
        return TaskRuntime(workspace_root=root)
    if not callable(factory):
        raise TypeError("build_default_task_runtime 必须可调用")
    runtime = factory(root)
    if not isinstance(runtime, TaskRuntime):
        raise TypeError("build_default_task_runtime 必须返回 TaskRuntime")
    return runtime


def _task_exit_code(status: object) -> int:
    from .runtime import TaskRunStatus

    if status is TaskRunStatus.PASS:
        return 0
    if status is TaskRunStatus.BLOCKED:
        return 3
    if status is TaskRunStatus.FAIL:
        return 1
    return 0  # SKIPPED 是成功处理的终态，但不具有 formal 资格。


def _logical_cli_ref(explicit: str | None, path: Path, *, namespace: str) -> str:
    """把机器文件定位转换成不会泄露绝对路径的逻辑引用。"""

    from .runtime import logical_reference

    value = explicit if explicit is not None else f"{namespace}/{path.name}"
    return logical_reference(value, field_name=f"{namespace}_ref")


def _publish_if_requested(path: Path | None, value: Mapping[str, object]) -> None:
    if path is None:
        return
    from .runtime import publish_canonical_immutable

    publish_canonical_immutable(path, value)


def _task_catalog(arguments: argparse.Namespace) -> int:
    from .contracts import DEFAULT_TASK_CATALOG

    if arguments.task_id is None:
        value = DEFAULT_TASK_CATALOG.to_dict()
    else:
        value = DEFAULT_TASK_CATALOG.get(arguments.task_id).to_dict()
    _publish_if_requested(arguments.output, value)
    _emit(value)
    return 0


def _task_environment_build(arguments: argparse.Namespace) -> int:
    """由重复 CLI 字段构造并发布 hash-bound ``TaskRuntimeEnvironment``。"""

    from .runtime import TaskRuntimeEnvironment, publish_canonical_immutable

    capabilities = tuple(arguments.capability)
    stages = tuple(arguments.contract_stage)
    gates = tuple(arguments.passed_gate)
    for field_name, values in (
        ("capability", capabilities),
        ("contract_stage", stages),
        ("passed_gate", gates),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"TASK_ENVIRONMENT_DUPLICATE:{field_name}")
    evidence: dict[str, str] = {}
    for index, item in enumerate(arguments.evidence):
        if not isinstance(item, str) or "=" not in item:
            raise ValueError(f"TASK_ENVIRONMENT_EVIDENCE_INVALID:{index}")
        key, reference = item.split("=", 1)
        if not key or not reference or key in evidence:
            raise ValueError(f"TASK_ENVIRONMENT_EVIDENCE_INVALID:{index}")
        evidence[key] = reference
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset(capabilities),
        frozen_contract_stages=frozenset(stages),
        passed_gate_ids=frozenset(gates),
        estimator_decision_ref=arguments.decision_ref,
        evidence_refs=evidence,
    )
    publish_canonical_immutable(arguments.output, environment.to_dict())
    _emit(environment.to_dict())
    return 0


def _task_preflight(arguments: argparse.Namespace) -> int:
    config = _load_v2_config(arguments.config)
    environment = _load_task_environment(arguments.environment)
    runtime = _build_default_task_runtime(_task_workspace_root(config))
    blockers = runtime.preflight(config, environment=environment)
    execution = config.section("execution")
    report = {
        "schema_version": "task-preflight-v1",
        "task_id": config.task_id,
        "config_hash": config.config_hash,
        "catalog_hash": runtime.catalog.catalog_hash,
        "environment_hash": environment.environment_hash,
        "ready": not blockers,
        "formal_eligible_if_completed": bool(
            not blockers
            and config.run_intent == "formal"
            and not execution["dry_run"]
        ),
        "registered_runner_kinds": [item.value for item in runtime.registered_kinds],
        "blockers": [item.to_dict() for item in blockers],
    }
    _publish_if_requested(arguments.output, report)
    _emit(report)
    return 0 if not blockers else 3


def _execute_task(arguments: argparse.Namespace) -> int:
    config = _load_v2_config(arguments.config)
    recovery = config.section("recovery")
    if not isinstance(recovery, Mapping):  # 已由 v2 校验，保留运行时防御。
        raise TypeError("TASK_RECOVERY_SECTION_NOT_OBJECT")
    resume_ref = recovery["resume_ref"]
    action = str(arguments.task_command)
    if action == "run" and resume_ref is not None:
        raise ValueError("TASK_RUN_FORBIDS_RECOVERY_RESUME_REF")
    if action == "resume" and resume_ref is None:
        raise ValueError("TASK_RESUME_REQUIRES_RECOVERY_RESUME_REF")
    environment = _load_task_environment(arguments.environment)
    runtime = _build_default_task_runtime(_task_workspace_root(config))
    result = runtime.execute(config, environment=environment)
    _publish_if_requested(arguments.result, result.to_dict())
    _emit(result.to_dict())
    return _task_exit_code(result.status)


def _task_stage3_trajectory(arguments: argparse.Namespace) -> int:
    """Run the real training endpoint producer without publishing Stage3.03."""

    from .experiments.stage3_trajectory import Stage3TrajectoryProducer
    from .runtime import TaskExecutionRequest

    config = _load_v2_config(arguments.config)
    if config.task_id != "stage3.03_endpoint_and_probe_pipeline":
        raise ValueError("STAGE3_TRAJECTORY_REQUIRES_STAGE3_03_CONFIG")
    environment = _load_task_environment(arguments.environment)
    workspace_root = _task_workspace_root(config)
    runtime = _build_default_task_runtime(workspace_root)
    blockers = runtime.preflight(config, environment=environment)
    if blockers:
        report = {
            "schema_version": "stage3-trajectory-preflight-v1",
            "task_id": config.task_id,
            "config_hash": config.config_hash,
            "ready": False,
            "blockers": [item.to_dict() for item in blockers],
        }
        _emit(report)
        return 3
    request = TaskExecutionRequest(
        config=config,
        task=runtime.catalog.get(config.task_id),
        environment=environment,
    )
    receipt = Stage3TrajectoryProducer(workspace_root).run(
        request,
        receipt_path=arguments.receipt,
    )
    _emit(receipt.to_dict())
    return 0


def _task_status(arguments: argparse.Namespace) -> int:
    from .runtime import TaskStatusSnapshot, load_task_run_result

    result = load_task_run_result(arguments.result)
    if arguments.config is not None:
        config = _load_v2_config(arguments.config)
        if config.task_id != result.task_id or config.config_hash != result.config_hash:
            raise ValueError("TASK_STATUS_CONFIG_IDENTITY_MISMATCH")
    snapshot = TaskStatusSnapshot.from_result(
        result,
        result_ref=_logical_cli_ref(
            arguments.result_ref,
            arguments.result,
            namespace="results",
        ),
    )
    _publish_if_requested(arguments.output, snapshot.to_dict())
    _emit(snapshot.to_dict())
    return _task_exit_code(result.status)


def _task_replay(arguments: argparse.Namespace) -> int:
    from .runtime import (
        TaskReplayRecord,
        TaskRunStatus,
        load_task_run_result,
        publish_canonical_immutable,
    )

    source = load_task_run_result(arguments.source_result)
    config = _load_v2_config(arguments.config)
    if source.task_id != config.task_id or source.config_hash != config.config_hash:
        raise ValueError("TASK_REPLAY_CONFIG_IDENTITY_MISMATCH")
    environment = _load_task_environment(arguments.environment)
    replay = _build_default_task_runtime(_task_workspace_root(config)).execute(
        config,
        environment=environment,
    )
    publish_canonical_immutable(arguments.result, replay.to_dict())
    record = TaskReplayRecord.compare(
        source,
        replay,
        source_result_ref=_logical_cli_ref(
            arguments.source_ref,
            arguments.source_result,
            namespace="results",
        ),
        replay_result_ref=_logical_cli_ref(
            arguments.result_ref,
            arguments.result,
            namespace="results",
        ),
    )
    _publish_if_requested(arguments.output, record.to_dict())
    _emit(record.to_dict())
    if replay.status is TaskRunStatus.BLOCKED:
        return 3
    if replay.status is TaskRunStatus.FAIL or not record.equivalent:
        return 1
    return 0


def _task_finalize(arguments: argparse.Namespace) -> int:
    from .runtime import (
        TaskFinalizationRecord,
        TaskRunStatus,
        TaskStatusSnapshot,
        load_task_run_result,
        publish_canonical_immutable,
    )

    result = load_task_run_result(arguments.result)
    result_ref = _logical_cli_ref(
        arguments.result_ref,
        arguments.result,
        namespace="results",
    )
    if result.status is not TaskRunStatus.PASS:
        snapshot = TaskStatusSnapshot.from_result(result, result_ref=result_ref)
        _emit(snapshot.to_dict())
        return _task_exit_code(result.status)
    finalization = TaskFinalizationRecord.from_result(result, result_ref=result_ref)
    publish_canonical_immutable(arguments.output, finalization.to_dict())
    _emit(finalization.to_dict())
    return 0


def _asset_acquire(arguments: argparse.Namespace) -> int:
    """只登记并验证调用方明确提供的本地文件，不执行下载或远程访问。"""

    from .assets import (
        AssetActorRole,
        AssetState,
        load_manifest,
        transition_manifest,
        verify_only,
    )
    from .contracts import canonical_json_hash
    from .runtime import publish_canonical_immutable

    source_root = arguments.source_root
    if not source_root.exists() or not source_root.is_dir():
        raise FileNotFoundError(f"LOCAL_ASSET_SOURCE_ROOT_MISSING:{source_root}")
    manifest = load_manifest(arguments.manifest)
    if manifest["state"] != AssetState.DOWNLOADING.value:
        raise ValueError("ASSET_ACQUIRE_REQUIRES_DOWNLOADING_MANIFEST")
    downloaded = transition_manifest(
        manifest,
        AssetState.DOWNLOADED,
        actor=arguments.actor,
        actor_role=AssetActorRole.FETCHER,
        evidence_ref=arguments.evidence_ref,
        summary=arguments.summary,
        at=arguments.at,
    )
    verification = verify_only(downloaded, source_root)
    publish_canonical_immutable(arguments.output, downloaded)
    _emit(
        {
            "schema_version": "cli.asset-acquire.v1",
            "acquisition_mode": "local_existing_files",
            "network_accessed": False,
            "asset_id": downloaded["asset_id"],
            "manifest_hash": canonical_json_hash(downloaded),
            "output": str(arguments.output),
            "verification": verification,
        }
    )
    return 0


def _asset_verify(arguments: argparse.Namespace) -> int:
    from .assets import load_manifest, verify_only

    report = verify_only(load_manifest(arguments.manifest), arguments.asset_root)
    _publish_if_requested(arguments.output, report)
    _emit(report)
    return 0


_ASSET_FETCH_FAILURE_MESSAGES: Mapping[str, str] = {
    "RUNTIME_URL_ARGV_FORBIDDEN": (
        "runtime URL values are forbidden in argv; use a named environment "
        "variable or stdin"
    ),
    "OUTPUT_EXISTS": "the requested output path already exists",
    "OUTPUT_CONFLICT": "the canonical output could not be published without clobbering",
    "SPEC_INVALID": "the fixed asset object specification is invalid",
    "TARGET_PATH_INVALID": "the approved asset root or relative final path is invalid",
    "POLICY_INVALID": "the acquisition policy is outside the supported bounds",
    "RUNTIME_URL_SOURCE_INVALID": "the selected runtime URL source is invalid or unavailable",
    "INTERNAL_ERROR": "the asset acquisition command failed safely",
}
_ASSET_FETCH_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_ASSET_FETCH_MAX_URL_CHARS = 8192
_ASSET_FETCH_FORBIDDEN_URL_OPTIONS = frozenset(
    {
        "--url",
        "--runtime-url",
        "--endpoint-url",
        "--download-url",
    }
)


def _asset_fetch_cli_failure(code: str) -> dict[str, object]:
    return {
        "schema_version": "stage0-asset-fetch-http-failure-v1",
        "status": "failed",
        "code": code,
        "message": _ASSET_FETCH_FAILURE_MESSAGES[code],
    }


def _asset_fetch_output_is_occupied(path: Path) -> bool:
    return (
        path.exists()
        or path.is_symlink()
        or bool(getattr(path, "is_junction", lambda: False)())
    )


def _asset_fetch_finish(
    arguments: argparse.Namespace,
    value: Mapping[str, object],
    *,
    exit_code: int,
) -> int:
    if arguments.output is not None:
        from .runtime import publish_canonical_immutable

        try:
            publish_canonical_immutable(arguments.output, value)
        except (ValueError, TypeError, KeyError, OSError, RuntimeError):
            _emit(_asset_fetch_cli_failure("OUTPUT_CONFLICT"))
            return 2
    _emit(value)
    return exit_code


def _asset_fetch_policy(arguments: argparse.Namespace) -> object:
    from .asset_acquisition import AcquisitionPolicy

    values = {
        "max_attempts": arguments.max_attempts,
        "request_timeout_seconds": arguments.request_timeout_seconds,
        "overall_timeout_seconds": arguments.overall_timeout_seconds,
        "initial_backoff_seconds": arguments.initial_backoff_seconds,
        "max_backoff_seconds": arguments.max_backoff_seconds,
        "lock_timeout_seconds": arguments.lock_timeout_seconds,
        "lock_poll_interval_seconds": arguments.lock_poll_interval_seconds,
        "chunk_size": arguments.chunk_size,
    }
    if not 1 <= values["max_attempts"] <= 16:
        raise ValueError("max_attempts is outside CLI bounds")
    if not 0 < values["request_timeout_seconds"] <= 300:
        raise ValueError("request_timeout_seconds is outside CLI bounds")
    if not 0 < values["overall_timeout_seconds"] <= 3600:
        raise ValueError("overall_timeout_seconds is outside CLI bounds")
    if not 0 <= values["initial_backoff_seconds"] <= 60:
        raise ValueError("initial_backoff_seconds is outside CLI bounds")
    if not 0 <= values["max_backoff_seconds"] <= 60:
        raise ValueError("max_backoff_seconds is outside CLI bounds")
    if not 0 < values["lock_timeout_seconds"] <= 300:
        raise ValueError("lock_timeout_seconds is outside CLI bounds")
    if not 0 < values["lock_poll_interval_seconds"] <= 5:
        raise ValueError("lock_poll_interval_seconds is outside CLI bounds")
    if not 1 <= values["chunk_size"] <= 16 * 1024 * 1024:
        raise ValueError("chunk_size is outside CLI bounds")
    return AcquisitionPolicy(**values)


def _asset_fetch_runtime_url(
    arguments: argparse.Namespace,
) -> tuple[str | None, str | None]:
    if arguments.url_env is not None:
        if not _ASSET_FETCH_ENV_NAME_PATTERN.fullmatch(arguments.url_env):
            return None, "RUNTIME_URL_SOURCE_INVALID"
        value = os.environ.get(arguments.url_env)
        if value is None:
            return None, "RUNTIME_URL_SOURCE_INVALID"
    else:
        try:
            value = sys.stdin.read(_ASSET_FETCH_MAX_URL_CHARS + 3)
        except (OSError, UnicodeError):
            return None, "RUNTIME_URL_SOURCE_INVALID"
        if value.endswith("\r\n"):
            value = value[:-2]
        elif value.endswith("\n"):
            value = value[:-1]

    if (
        not value
        or len(value) > _ASSET_FETCH_MAX_URL_CHARS
        or value != value.strip()
        or any(character in value for character in ("\x00", "\r", "\n"))
    ):
        return None, "RUNTIME_URL_SOURCE_INVALID"
    return value, None


def _asset_fetch_http(arguments: argparse.Namespace) -> int:
    from .asset_acquisition import (
        AssetAcquisitionError,
        AssetObjectSpec,
        acquire_http_asset,
        resolve_approved_asset_target,
    )

    try:
        if arguments.output is not None and _asset_fetch_output_is_occupied(
            arguments.output
        ):
            _emit(_asset_fetch_cli_failure("OUTPUT_EXISTS"))
            return 2
    except OSError:
        _emit(_asset_fetch_cli_failure("OUTPUT_CONFLICT"))
        return 2

    if arguments.spec.suffix.casefold() != ".json":
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("SPEC_INVALID"),
            exit_code=2,
        )
    try:
        spec = AssetObjectSpec.from_mapping(_load_mapping(arguments.spec))
    except (ValueError, TypeError, KeyError, OSError):
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("SPEC_INVALID"),
            exit_code=2,
        )

    try:
        target = resolve_approved_asset_target(
            arguments.asset_root,
            arguments.final_path,
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target = resolve_approved_asset_target(
            arguments.asset_root,
            arguments.final_path,
        )
    except (ValueError, TypeError, OSError):
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("TARGET_PATH_INVALID"),
            exit_code=2,
        )

    try:
        policy = _asset_fetch_policy(arguments)
    except (ValueError, TypeError, OverflowError):
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("POLICY_INVALID"),
            exit_code=2,
        )

    runtime_url, input_error = _asset_fetch_runtime_url(arguments)
    if input_error is not None or runtime_url is None:
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("RUNTIME_URL_SOURCE_INVALID"),
            exit_code=2,
        )

    try:
        result = acquire_http_asset(
            spec,
            runtime_url,
            target,
            policy=policy,
        )
    except AssetAcquisitionError as error:
        return _asset_fetch_finish(
            arguments,
            error.report.to_dict(),
            exit_code=3,
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError):
        return _asset_fetch_finish(
            arguments,
            _asset_fetch_cli_failure("INTERNAL_ERROR"),
            exit_code=3,
        )
    return _asset_fetch_finish(arguments, result.to_dict(), exit_code=0)


def _asset_fetch_argv_contains_runtime_url(argv: list[str]) -> bool:
    if len(argv) < 2 or argv[0:2] != ["asset", "fetch-http"]:
        return False
    for argument in argv[2:]:
        option = argument.split("=", 1)[0].casefold()
        if option in _ASSET_FETCH_FORBIDDEN_URL_OPTIONS:
            return True
        lowered = argument.casefold()
        if "://" in argument or "?" in argument or lowered.startswith(
            ("http:", "https:")
        ):
            return True
    return False


def _artifact_review_identity(
    path: Path,
    *,
    declared_scope: str,
) -> tuple[str, str, bool]:
    """验证 artifact 并从其自身字段推导 formal 资格，禁止 CLI 开关伪造。"""

    if path.is_dir():
        from .runtime import load_tensor_bundle

        _, identity = load_tensor_bundle(path)
        return "tensor_bundle", identity.manifest_sha256, False

    value = _load_mapping(path)
    kind, digest = _validate_known_artifact(value)
    if digest is None:
        raise ValueError("ARTIFACT_REVIEW_REQUIRES_CONTENT_HASH")
    embedded_scope = value.get("scope", value.get("run_intent"))
    if embedded_scope in {"local_fixture", "formal"} and embedded_scope != declared_scope:
        raise ValueError("ARTIFACT_REVIEW_SCOPE_MISMATCH")

    formal_eligible = value.get("formal_eligible") is True
    if value.get("schema_version") == "gate-record-v1":
        from .contracts import GateRecord, GateStatus

        formal_eligible = (
            GateRecord.from_mapping(value).effective_status() is GateStatus.PASS
        )
    return kind, digest, bool(declared_scope == "formal" and formal_eligible)


def _artifact_review(arguments: argparse.Namespace) -> int:
    from .contracts import ArtifactReview, ReviewDecision
    from .runtime import publish_canonical_immutable

    kind, digest, formal_eligible = _artifact_review_identity(
        arguments.artifact,
        declared_scope=arguments.scope,
    )
    metadata = {} if arguments.metadata is None else _load_mapping(arguments.metadata)
    review = ArtifactReview(
        artifact_kind=kind,
        artifact_ref=arguments.artifact_ref,
        artifact_hash=digest,
        artifact_scope=arguments.scope,
        artifact_formal_eligible=formal_eligible,
        reviewer=arguments.reviewer,
        decision=ReviewDecision(arguments.decision),
        findings=tuple(arguments.finding),
        metadata=metadata,
    )
    publish_canonical_immutable(arguments.output, review.to_dict())
    _emit(review.to_dict())
    return 0


def _artifact_approve(arguments: argparse.Namespace) -> int:
    from .contracts import (
        ApprovalScope,
        ArtifactApproval,
        ArtifactReview,
        ensure_json_object,
        load_canonical_json,
    )
    from .runtime import publish_canonical_immutable

    review = ArtifactReview.from_mapping(
        ensure_json_object(
            load_canonical_json(arguments.review),
            field="artifact review",
        )
    )
    approval = ArtifactApproval.from_review(
        review,
        review_ref=arguments.review_ref,
        approval_scope=ApprovalScope(arguments.scope),
        approver=arguments.approver,
    )
    publish_canonical_immutable(arguments.output, approval.to_dict())
    _emit(approval.to_dict())
    return 0


def _logical_artifact_ref(value: str | None, *, field: str) -> str | None:
    """CLI 构建器使用的 POSIX workspace 相对引用边界。"""

    if value is None:
        return None
    from pathlib import PurePosixPath

    if not value or "\\" in value:
        raise ValueError(f"{field} 必须是 POSIX workspace 相对路径")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} 发生路径逃逸")
    return path.as_posix()


def _formal_evidence_for_builder(
    path: Path | None,
    *,
    scope: str,
    stage: int = 3,
):
    from .contracts import FormalExecutionEvidence, ensure_json_object, load_canonical_json

    if scope == "local_fixture":
        if path is not None:
            raise ValueError("local_fixture 构建器不能携带 formal execution evidence")
        return FormalExecutionEvidence("local_fixture")
    if path is None:
        raise ValueError("formal 构建器必须传 --formal-execution-evidence")
    evidence = FormalExecutionEvidence.from_mapping(
        ensure_json_object(load_canonical_json(path), field="formal execution evidence")
    )
    evidence.require_for_stage(stage)
    return evidence


def _artifact_stage2_experiment_plan_build(arguments: argparse.Namespace) -> int:
    """把声明式 Stage 2 B/M/R 源编译为 evidence/sampling/lineage-bound 计划。"""

    from .contracts import ensure_json_object, load_canonical_json
    from .experiments import FormalExperimentPlan, SamplingPlan
    from .runtime import publish_canonical_immutable

    evidence = _formal_evidence_for_builder(
        arguments.formal_execution_evidence,
        scope="formal",
        stage=2,
    )
    spec = _load_mapping(arguments.spec)
    expected = {
        "schema_version",
        "plan_id",
        "task_id",
        "wave_id",
        "cell_id",
        "stream",
        "batch_size",
        "microbatch_counts",
        "repetitions",
        "selection_basis",
        "pilot_thresholds",
    }
    if set(spec) != expected or spec.get("schema_version") != (
        "stage2-formal-experiment-plan-source-v1"
    ):
        raise ValueError(
            "STAGE2_FORMAL_EXPERIMENT_PLAN_SOURCE_FIELDS_MISMATCH:"
            f"missing={sorted(expected-set(spec))}:extra={sorted(set(spec)-expected)}"
        )
    sampling_document = ensure_json_object(
        load_canonical_json(arguments.sampling_plan),
        field="stage2 sampling plan",
    )
    sampling_payload = _unique_schema_mapping(
        sampling_document,
        schema_version="sampling-plan-v1",
    )
    sampling = SamplingPlan.from_mapping(sampling_payload)
    normalized_refs: list[str] = []
    for reference in arguments.source_ref:
        normalized = _logical_artifact_ref(reference, field="source_ref")
        assert normalized is not None  # argparse 的 append 元素始终是字符串
        normalized_refs.append(normalized)
    if len(normalized_refs) != len(set(normalized_refs)):
        raise ValueError("--source-ref 不能重复")
    source_refs = tuple(sorted(normalized_refs))
    plan = FormalExperimentPlan(
        plan_id=spec["plan_id"],  # type: ignore[arg-type]
        task_id=spec["task_id"],  # type: ignore[arg-type]
        wave_id=spec["wave_id"],  # type: ignore[arg-type]
        cell_id=spec["cell_id"],  # type: ignore[arg-type]
        stream=spec["stream"],  # type: ignore[arg-type]
        batch_size=spec["batch_size"],  # type: ignore[arg-type]
        microbatch_counts=tuple(spec["microbatch_counts"]),  # type: ignore[arg-type]
        repetitions=spec["repetitions"],  # type: ignore[arg-type]
        sampling_plan_hash=sampling.digest,
        execution_evidence_hash=evidence.artifact_hash,
        source_artifact_refs=source_refs,
        selection_basis=spec["selection_basis"],  # type: ignore[arg-type]
        pilot_thresholds=spec["pilot_thresholds"],  # type: ignore[arg-type]
    )
    payload = plan.to_dict()
    _validate_known_artifact(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_endpoint_plan_build(arguments: argparse.Namespace) -> int:
    """从 CLI 参数发布 endpoint 选择计划，并自动计算 canonical hash。"""

    from .contracts import canonical_json_hash
    from .runtime import publish_canonical_immutable

    evidence = _formal_evidence_for_builder(
        arguments.formal_execution_evidence, scope=arguments.scope
    )
    steps = sorted(set(arguments.step))
    if len(steps) != len(arguments.step):
        raise ValueError("--step 不能重复")
    if any(step <= 0 for step in steps):
        raise ValueError("--step 必须为正整数")
    if not steps and not arguments.include_checkpoint_steps:
        raise ValueError("至少传一个 --step 或 --include-checkpoint-steps")
    endpoint_metadata: Mapping[str, object] | None = None
    if arguments.endpoint_metadata is not None:
        endpoint_metadata = _load_mapping(arguments.endpoint_metadata)
        for raw_step, raw_metadata in endpoint_metadata.items():
            if (
                not isinstance(raw_step, str)
                or not raw_step.isdecimal()
                or int(raw_step) <= 0
                or not isinstance(raw_metadata, Mapping)
                or set(raw_metadata) != {"model", "seed", "stage"}
                or raw_metadata["model"] not in {"14M", "31M"}
                or isinstance(raw_metadata["seed"], bool)
                or not isinstance(raw_metadata["seed"], int)
                or raw_metadata["seed"] < 0
                or raw_metadata["stage"] not in {"early", "middle", "late"}
            ):
                raise ValueError("ENDPOINT_METADATA_INVALID")
        if not set(steps).issubset({int(item) for item in endpoint_metadata}):
            raise ValueError("ENDPOINT_METADATA_SELECTED_STEP_COVERAGE_MISSING")
    elif arguments.scope in {"pilot", "formal"}:
        raise ValueError("pilot/formal endpoint plan 必须传 --endpoint-metadata")
    payload: dict[str, Any] = {
        "schema_version": "training-endpoint-capture-plan-v1",
        "plan_id": arguments.plan_id,
        "selected_steps": steps,
        "include_checkpoint_steps": bool(arguments.include_checkpoint_steps),
        "scope": arguments.scope,
        "formal_eligible": arguments.scope == "formal",
        "qualification_evidence_hash": (
            evidence.artifact_hash if arguments.scope == "formal" else None
        ),
        "probe_plan_ref": _logical_artifact_ref(
            arguments.probe_plan_ref, field="probe_plan_ref"
        ),
    }
    if endpoint_metadata is not None:
        payload["endpoint_metadata"] = {
            step: dict(endpoint_metadata[step])
            for step in sorted(endpoint_metadata, key=int)
        }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _validate_known_artifact(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_probe_plan_build(arguments: argparse.Namespace) -> int:
    """把不含手算 hash 的 probe YAML/JSON spec 编译为正式输入 artifact。"""

    from .contracts import canonical_json_hash
    from .runtime import publish_canonical_immutable

    evidence = _formal_evidence_for_builder(
        arguments.formal_execution_evidence, scope=arguments.scope
    )
    spec = _load_mapping(arguments.spec)
    expected = {"panel_id", "endpoint_digest", "entries", "minimum_formal_probes"}
    if set(spec) != expected:
        raise ValueError(
            "PROBE_PLAN_SPEC_FIELDS_MISMATCH:"
            f"missing={sorted(expected-set(spec))}:extra={sorted(set(spec)-expected)}"
        )
    payload: dict[str, Any] = {
        "schema_version": "stage3-probe-plan-v1",
        **spec,
        "execution_evidence_hash": evidence.artifact_hash,
        "scope": arguments.scope,
        "formal_eligible": arguments.scope == "formal",
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    _validate_known_artifact(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_quadrature_pilot_plan_build(arguments: argparse.Namespace) -> int:
    """把人工预注册源编译为绑定 FormalExecutionEvidence 的 pilot plan。"""

    from .contracts import canonical_json_hash
    from .experiments import QuadratureThresholds
    from .runtime import publish_canonical_immutable

    evidence = _formal_evidence_for_builder(
        arguments.formal_execution_evidence,
        scope="formal",
    )
    spec = _load_mapping(arguments.spec)
    expected = {
        "plan_id", "candidate_rules", "required_unit_ids", "unit_strata", "thresholds"
    }
    index_fields = {
        "production_unit_index_ref", "production_unit_index_hash"
    }
    plan_fields = {"plan_kind", "production_unit_index_scope"}
    if set(spec) != expected and set(spec) != expected | index_fields | plan_fields:
        raise ValueError(
            "STAGE3_FORMAL_PILOT_PLAN_SOURCE_FIELDS_MISMATCH:"
            f"missing={sorted(expected-set(spec))}:extra={sorted(set(spec)-expected)}"
        )
    candidate_rules = spec["candidate_rules"]
    units = spec["required_unit_ids"]
    if not all(
        isinstance(item, list)
        and item
        and all(isinstance(child, str) and child for child in item)
        and len(item) == len(set(item))
        for item in (candidate_rules, units)
    ):
        raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_ARRAYS_INVALID")
    unit_strata = spec["unit_strata"]
    if (
        not isinstance(unit_strata, Mapping)
        or set(unit_strata) != set(units)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"model", "stage", "update", "probe"}
            or any(not isinstance(child, str) or not child for child in item.values())
            or item["stage"] not in {"early", "middle", "late"}
            for item in unit_strata.values()
        )
    ):
        raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_UNIT_STRATA_INVALID")
    production_index = None
    production_index_ref = spec.get("production_unit_index_ref")
    production_index_hash = spec.get("production_unit_index_hash")
    if (production_index_ref is None) != (production_index_hash is None):
        raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_INDEX_BINDING_INVALID")
    if production_index_ref is not None:
        from .experiments.stage3_production_plan import load_production_unit_index

        if (
            not isinstance(production_index_ref, str)
            or not production_index_ref
            or not isinstance(production_index_hash, str)
        ):
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_INDEX_BINDING_INVALID")
        plan_kind = spec.get("plan_kind")
        index_scope = spec.get("production_unit_index_scope")
        if plan_kind not in {"pilot", "matrix"}:
            raise ValueError("STAGE3_FORMAL_PLAN_KIND_INVALID")
        expected_scope = "pilot" if plan_kind == "pilot" else "formal"
        if index_scope != expected_scope:
            raise ValueError("STAGE3_FORMAL_PLAN_INDEX_SCOPE_INVALID")
        raw_index_path = Path(production_index_ref)
        index_path_candidates = [
            raw_index_path,
            Path(arguments.spec).resolve().parent / raw_index_path,
            Path.cwd() / raw_index_path,
        ]
        index_path = next(
            (item for item in index_path_candidates if item.is_file()), None
        )
        if index_path is None:
            raise ValueError(
                "STAGE3_FORMAL_PILOT_PLAN_SOURCE_INDEX_NOT_FOUND:"
                f"{production_index_ref}"
            )
        production_index = load_production_unit_index(
            index_path, expected_scope=expected_scope
        )
        if production_index_hash != production_index.artifact_hash:
            raise ValueError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_INDEX_HASH_MISMATCH")
        indexed_units = [unit.path_unit_id for unit in production_index.units]
        indexed_strata = production_index.unit_strata()
        if list(units) != indexed_units or dict(unit_strata) != indexed_strata:
            raise ValueError(
                "STAGE3_FORMAL_PILOT_PLAN_SOURCE_INDEX_DERIVATION_MISMATCH"
            )
    elif len(units) in {12, 99} or any(key in spec for key in plan_fields):
        raise ValueError(
            "STAGE3_FORMAL_PLAN_SOURCE_INDEX_REQUIRED_FOR_PRODUCTION_PLAN"
        )
    thresholds = spec["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise TypeError("STAGE3_FORMAL_PILOT_PLAN_SOURCE_THRESHOLDS_INVALID")
    normalized_thresholds = QuadratureThresholds(
        **dict(thresholds)  # type: ignore[arg-type]
    ).require_formal_contract().to_dict()
    payload: dict[str, Any] = {
        "schema_version": "stage3-formal-pilot-plan-v1",
        "plan_id": spec["plan_id"],
        "scope": "formal",
        "candidate_rules": list(candidate_rules),
        "required_unit_ids": list(units),
        "unit_strata": {
            unit_id: dict(unit_strata[unit_id]) for unit_id in units
        },
        "thresholds": normalized_thresholds,
        "execution_evidence_hash": evidence.artifact_hash,
        "formal_eligible": True,
    }
    if production_index is not None:
        payload["plan_kind"] = str(spec["plan_kind"])
        payload["production_unit_index_scope"] = str(
            spec["production_unit_index_scope"]
        )
        payload["production_unit_index_ref"] = str(production_index_ref)
        payload["production_unit_index_hash"] = str(production_index_hash)
    payload["artifact_hash"] = canonical_json_hash(payload)
    _validate_known_artifact(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_stage3_production_index_build(arguments: argparse.Namespace) -> int:
    """Build the exact 12/99 endpoint×probe production index from real files."""

    from .experiments.stage3_production_plan import (
        build_production_unit_index,
        write_production_unit_index,
    )

    index = build_production_unit_index(
        tuple(arguments.endpoint_commit),
        tuple(arguments.probe_plan),
        scope=arguments.scope,
        workspace_root=arguments.workspace_root,
        index_id=arguments.index_id,
    )
    write_production_unit_index(arguments.output, index)
    _emit(index.to_dict())
    return 0


def _artifact_stage3_path_unit_selector_build(arguments: argparse.Namespace) -> int:
    """Publish one canonical config-route selector bound to a production index."""

    from .contracts import canonical_json_hash
    from .experiments.stage3_production_plan import load_production_unit_index
    from .runtime import publish_canonical_immutable

    index = load_production_unit_index(
        arguments.unit_index, expected_scope=arguments.scope
    )
    index.unit(arguments.active_unit_id)
    payload: dict[str, Any] = {
        "schema_version": "stage3-path-unit-selector-v1",
        "scope": arguments.scope,
        "unit_index_hash": index.artifact_hash,
        "active_unit_id": arguments.active_unit_id,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_stage3_probe_selector_build(arguments: argparse.Namespace) -> int:
    """Publish a canonical formal probe selector for pre-index S3.03/S3.04."""

    from .contracts import canonical_json_hash
    from .runtime import publish_canonical_immutable

    endpoint = _load_mapping(arguments.endpoint_commit)
    plan = _load_mapping(arguments.probe_plan)
    record = endpoint.get("record")
    endpoint_digest = record.get("endpoint_digest") if isinstance(record, Mapping) else None
    if (
        endpoint.get("schema_version") != "stage3-endpoint-capture-v1"
        or endpoint.get("scope") != "formal"
        or endpoint.get("formal_eligible") is not True
        or endpoint.get("artifact_hash")
        != canonical_json_hash(
            {key: value for key, value in endpoint.items() if key != "artifact_hash"}
        )
    ):
        raise ValueError("STAGE3_PROBE_SELECTOR_ENDPOINT_INVALID")
    if (
        plan.get("schema_version") != "stage3-probe-plan-v1"
        or plan.get("scope") != "formal"
        or plan.get("formal_eligible") is not True
        or plan.get("endpoint_digest") != endpoint_digest
        or plan.get("artifact_hash")
        != canonical_json_hash(
            {key: value for key, value in plan.items() if key != "artifact_hash"}
        )
    ):
        raise ValueError("STAGE3_PROBE_SELECTOR_PLAN_INVALID")
    entries = plan.get("entries")
    matches = (
        [
            item
            for item in entries
            if isinstance(item, Mapping)
            and item.get("role") == "formal"
            and item.get("probe_id") == arguments.active_probe_id
        ]
        if isinstance(entries, list)
        else []
    )
    if len(matches) != 1:
        raise ValueError("STAGE3_PROBE_SELECTOR_PROBE_NOT_UNIQUE")
    payload: dict[str, Any] = {
        "schema_version": "stage3-probe-selector-v1",
        "scope": "formal",
        "endpoint_digest": endpoint_digest,
        "probe_plan_hash": plan["artifact_hash"],
        "active_probe_id": arguments.active_probe_id,
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _unique_schema_mapping(
    value: Mapping[str, Any],
    *,
    schema_version: str,
) -> dict[str, Any]:
    """从直接 artifact 或 task envelope 中提取唯一业务对象。

    builder 接受直接的 canonical 业务 artifact，也接受任务两阶段发布后的 envelope。
    递归查找时按 canonical hash 去重；找到零个或多个不同对象都会 fail-closed，避免
    靠“取第一个”把错误 decision/Gate 静默接入训练路线。
    """

    from .contracts import canonical_json_hash

    matches: dict[str, dict[str, Any]] = {}

    def visit(candidate: object) -> None:
        if isinstance(candidate, Mapping):
            normalized = dict(candidate)
            if normalized.get("schema_version") == schema_version:
                matches[canonical_json_hash(normalized)] = normalized
            for child in normalized.values():
                visit(child)
        elif isinstance(candidate, list):
            for child in candidate:
                visit(child)

    visit(value)
    if len(matches) != 1:
        raise ValueError(
            f"ARTIFACT_SCHEMA_CARDINALITY:{schema_version}:{len(matches)}"
        )
    return next(iter(matches.values()))


def _artifact_route_build(arguments: argparse.Namespace) -> int:
    """把无 hash 的版本化路线声明编译为 ``TrainingRouteSpec``。"""

    from .experiments import TrainingPhaseSpec, TrainingRouteSpec
    from .runtime import publish_canonical_immutable

    spec = _load_mapping(arguments.spec)
    expected = {"schema_version", "route_id", "run_intent", "phases", "metadata"}
    if set(spec) != expected or spec.get("schema_version") != "training-route-source-v1":
        raise ValueError(
            "TRAINING_ROUTE_SOURCE_FIELDS_OR_VERSION_INVALID:"
            f"missing={sorted(expected-set(spec))}:extra={sorted(set(spec)-expected)}"
        )
    phases_raw = spec["phases"]
    if not isinstance(phases_raw, list) or not phases_raw or not all(
        isinstance(item, Mapping) for item in phases_raw
    ):
        raise TypeError("TRAINING_ROUTE_SOURCE_PHASES_NOT_OBJECT_ARRAY")
    metadata = spec["metadata"]
    if not isinstance(metadata, Mapping):
        raise TypeError("TRAINING_ROUTE_SOURCE_METADATA_NOT_OBJECT")
    if not isinstance(spec["route_id"], str) or not spec["route_id"]:
        raise TypeError("TRAINING_ROUTE_SOURCE_ID_NOT_STRING")
    if spec["run_intent"] not in {"local_fixture", "formal"}:
        raise ValueError("TRAINING_ROUTE_SOURCE_INTENT_INVALID")
    phases = tuple(TrainingPhaseSpec.from_mapping(item) for item in phases_raw)
    importance_enabled = any(phase.importance_enabled for phase in phases)
    if importance_enabled != (arguments.decision is not None):
        raise ValueError(
            "TRAINING_ROUTE_DECISION_PRESENCE_MISMATCH:"
            "启用 importance 时必须且只能提供 --decision"
        )
    decision: Mapping[str, Any] | None = None
    if arguments.decision is not None:
        decision = _unique_schema_mapping(
            _load_mapping(arguments.decision),
            schema_version="estimator-decision-v1",
        )
    gate: Mapping[str, Any] | None = None
    if arguments.gate is not None:
        gate = _unique_schema_mapping(
            _load_mapping(arguments.gate),
            schema_version="gate-record-v1",
        )
    route = TrainingRouteSpec(
        route_id=spec["route_id"],
        phases=phases,
        run_intent=spec["run_intent"],
        estimator_decision=decision,
        estimator_gate=gate,
        metadata=metadata,
    )
    payload = route.to_dict()
    _validate_known_artifact(payload)
    publish_canonical_immutable(arguments.output, payload)
    _emit(payload)
    return 0


def _artifact_ablation_matrix_build(arguments: argparse.Namespace) -> int:
    """编译版本化消融声明，并可同时发布确定性的完整 cell 矩阵。"""

    from .experiments import AblationFactor, AblationMatrixDeclaration
    from .runtime import publish_canonical_immutable

    spec = _load_mapping(arguments.spec)
    expected = {
        "schema_version",
        "matrix_id",
        "base_config",
        "factors",
        "base_seed",
        "seed_namespace",
        "scope",
    }
    if set(spec) != expected or spec.get("schema_version") != "ablation-matrix-source-v1":
        raise ValueError(
            "ABLATION_MATRIX_SOURCE_FIELDS_OR_VERSION_INVALID:"
            f"missing={sorted(expected-set(spec))}:extra={sorted(set(spec)-expected)}"
        )
    base_config = spec["base_config"]
    factors_raw = spec["factors"]
    if not isinstance(base_config, Mapping):
        raise TypeError("ABLATION_MATRIX_SOURCE_BASE_CONFIG_NOT_OBJECT")
    if not isinstance(factors_raw, list) or not factors_raw:
        raise TypeError("ABLATION_MATRIX_SOURCE_FACTORS_NOT_ARRAY")
    for field in ("matrix_id", "seed_namespace", "scope"):
        if not isinstance(spec[field], str) or not spec[field]:
            raise TypeError(f"ABLATION_MATRIX_SOURCE_STRING_INVALID:{field}")
    if isinstance(spec["base_seed"], bool) or not isinstance(spec["base_seed"], int):
        raise TypeError("ABLATION_MATRIX_SOURCE_BASE_SEED_NOT_INTEGER")
    factors = []
    fields = {"name", "config_path", "baseline_value", "alternatives"}
    for index, raw in enumerate(factors_raw):
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise ValueError(f"ABLATION_MATRIX_SOURCE_FACTOR_FIELDS_INVALID:{index}")
        if not isinstance(raw["name"], str) or not raw["name"]:
            raise TypeError(f"ABLATION_MATRIX_SOURCE_FACTOR_NAME_INVALID:{index}")
        path = raw["config_path"]
        alternatives = raw["alternatives"]
        if not isinstance(path, list) or not all(
            isinstance(item, str) and item for item in path
        ):
            raise TypeError(f"ABLATION_MATRIX_SOURCE_FACTOR_PATH_INVALID:{index}")
        if not isinstance(alternatives, list):
            raise TypeError(f"ABLATION_MATRIX_SOURCE_ALTERNATIVES_INVALID:{index}")
        factors.append(
            AblationFactor(
                name=raw["name"],
                config_path=tuple(path),
                baseline_value=raw["baseline_value"],
                alternatives=tuple(alternatives),
            )
        )
    declaration = AblationMatrixDeclaration.create(
        matrix_id=spec["matrix_id"],
        base_config=base_config,
        factors=tuple(factors),
        base_seed=spec["base_seed"],
        seed_namespace=spec["seed_namespace"],
        scope=spec["scope"],
    )
    declaration_payload = declaration.to_dict()
    _validate_known_artifact(declaration_payload)
    publish_canonical_immutable(arguments.output, declaration_payload)
    compiled_payload: Mapping[str, object] | None = None
    if arguments.compiled_output is not None:
        compiled_payload = declaration.compile().to_dict()
        _validate_known_artifact(dict(compiled_payload))
        publish_canonical_immutable(arguments.compiled_output, compiled_payload)
    _emit(
        {
            "schema_version": "cli.ablation-matrix-build.v1",
            "declaration": declaration_payload,
            "compiled_matrix": compiled_payload,
        }
    )
    return 0


def _task_fixture_all(arguments: argparse.Namespace) -> int:
    """执行始终不具 formal 资格的 Stage 0--9 缩小本机流水线。"""

    from .experiments import run_full_fixture_pipeline

    result = run_full_fixture_pipeline(
        workspace_root=arguments.workspace_root,
        base_config_path=arguments.base_config,
    )
    _emit(result.to_dict())
    return 0


def _task_tensorboard_rebuild(arguments: argparse.Namespace) -> int:
    """从 JSONL 机器真值重建可删除的 TensorBoard 派生视图。"""

    from .runtime import rebuild_tensorboard_from_jsonl

    count = rebuild_tensorboard_from_jsonl(arguments.event, arguments.output_dir)
    _emit(
        {
            "schema_version": "tensorboard-rebuild-result-v1",
            "event_paths": [str(path) for path in arguments.event],
            "output_dir": str(arguments.output_dir),
            "scalar_count": count,
            "source_of_truth": "typed_jsonl",
        }
    )
    return 0


def _gate_build(arguments: argparse.Namespace) -> int:
    from .contracts import GateRecord, GateStatus, load_canonical_json
    from .runtime import publish_canonical_immutable

    measured = (
        None
        if arguments.measured_json is None
        else load_canonical_json(arguments.measured_json)
    )
    threshold = (
        None
        if arguments.threshold_json is None
        else load_canonical_json(arguments.threshold_json)
    )
    record = GateRecord(
        gate_id=arguments.gate_id,
        stage=arguments.stage,
        status=GateStatus(arguments.status),
        checked_at=arguments.checked_at,
        measured=measured,
        threshold=threshold,
        evidence_refs=tuple(arguments.evidence_ref),
        reasons=tuple(arguments.reason),
        conditions=tuple(arguments.condition),
        expires_at=arguments.expires_at,
    )
    publish_canonical_immutable(arguments.output, record.to_dict())
    _emit(record.to_dict())
    return 0


def _artifact_stage3_scope_authority_publish(arguments: argparse.Namespace) -> int:
    """Publish the explicit G3-0 user authority as formal task commits."""

    from .experiments.stage3_scope_authority import (
        publish_stage3_scope_authority,
    )

    result = publish_stage3_scope_authority(
        workspace_root=arguments.workspace_root,
        output_dir=arguments.output_dir,
        decision_path=arguments.decision,
        gate_path=arguments.gate,
        config_hash=arguments.config_hash,
    )
    _emit(result)
    return 0


def _artifact_stage3_g36_publish(arguments: argparse.Namespace) -> int:
    """Reload committed Stage3.08 authorities and publish independent G3-6."""

    from .experiments.stage3_g36_publisher import Stage3G36Publisher
    from .runtime import publish_canonical_immutable

    publication = Stage3G36Publisher().publish(
        workspace_root=arguments.workspace_root,
        output_dir=arguments.output_dir,
        config_hash=arguments.config_hash,
        frozen_source_table_ref=arguments.frozen_source_table_ref,
        provenance_ref=arguments.provenance_ref,
        formal_plan_ref=arguments.formal_plan_ref,
        execution_evidence_ref=arguments.execution_evidence_ref,
        stage3_scope_decision_ref=arguments.stage3_scope_decision_ref,
        stage3_scope_gate_ref=arguments.stage3_scope_gate_ref,
        publication_id=arguments.publication_id,
    )
    if arguments.receipt is not None:
        publish_canonical_immutable(arguments.receipt, publication.to_dict())
    _emit(publication.to_dict())
    return 0 if publication.formal_eligible else 3


def _artifact_stage3_g37_publish(arguments: argparse.Namespace) -> int:
    """Reload committed Stage3.09 inputs and publish independent G3-7."""

    from .experiments.stage3_g37_publisher import Stage3G37Publisher
    from .runtime import publish_canonical_immutable

    publication = Stage3G37Publisher().publish(
        workspace_root=arguments.workspace_root,
        output_dir=arguments.output_dir,
        frozen_source_table_ref=arguments.frozen_source_table_ref,
        cost_accuracy_table_ref=arguments.cost_accuracy_table_ref,
        quadrature_decision_ref=arguments.quadrature_decision_ref,
        formal_plan_ref=arguments.formal_plan_ref,
        execution_evidence_ref=arguments.execution_evidence_ref,
        provenance_ref=arguments.provenance_ref,
        evaluation_ref=arguments.evaluation_ref,
        g3_6_ref=arguments.g3_6_ref,
        stage3_scope_decision_ref=arguments.stage3_scope_decision_ref,
        stage3_scope_gate_ref=arguments.stage3_scope_gate_ref,
        config_hash=arguments.config_hash,
        publication_id=arguments.publication_id,
    )
    if arguments.receipt is not None:
        publish_canonical_immutable(arguments.receipt, publication.to_dict())
    _emit(publication.to_dict())
    return 0 if publication.formal_eligible else 3


def _artifact_stage3_g38_publish(arguments: argparse.Namespace) -> int:
    """Independently verify the complete delivery and publish G3-8."""

    from .experiments.stage3_g38_publisher import Stage3G38Publisher
    from .runtime import publish_canonical_immutable

    gate_refs: dict[str, str] = {}
    for raw in arguments.gate_ref:
        gate_id, separator, reference = raw.partition("=")
        if not separator or not gate_id or not reference or gate_id in gate_refs:
            raise ValueError("STAGE3_G38_GATE_REF_MUST_BE_UNIQUE_GATE_ID_EQUALS_REF")
        gate_refs[gate_id] = reference
    stage3_10_refs = {
        "analysis_report": arguments.analysis_report_ref,
        "chart_artifacts": arguments.chart_artifacts_ref,
        "handoff_manifest": arguments.handoff_manifest_ref,
        "gate_summary": arguments.gate_summary_ref,
    }
    publication = Stage3G38Publisher().publish(
        workspace_root=arguments.workspace_root,
        output_dir=arguments.output_dir,
        config_hash=arguments.config_hash,
        gate_refs=gate_refs,
        stage3_10_refs=stage3_10_refs,
        execution_evidence_ref=arguments.execution_evidence_ref,
        g3_7_publication_ref=arguments.g3_7_publication_ref,
        recommendation_ref=arguments.recommendation_ref,
        finalization_ref=arguments.finalization_ref,
        delivery_manifest_ref=arguments.delivery_manifest_ref,
        publication_id=arguments.publication_id,
        publication_config_hash=arguments.publication_config_hash,
        checked_at=arguments.checked_at,
    )
    if arguments.receipt is not None:
        publish_canonical_immutable(arguments.receipt, publication.to_dict())
    _emit(publication.to_dict())
    return 0


def _artifact_stage3_control_publish(arguments: argparse.Namespace) -> int:
    """Wrap a validated Stage 3 control-plane document in a formal commit."""

    from .contracts import FormalExecutionEvidence
    from .contracts.provenance import ProvenanceRecord
    from .runtime import TaskArtifactStore

    root = Path(arguments.workspace_root).resolve()
    source = Path(arguments.source)
    source_path = source.resolve() if source.is_absolute() else (root / source).resolve()
    try:
        source_ref = source_path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("STAGE3_CONTROL_SOURCE_OUTSIDE_WORKSPACE") from error
    payload = _load_mapping(source_path)
    declarations = {
        "formal-plan": (
            "stage3.formal_plan_authority",
            "stage3_formal_plan",
            "stage3-formal-pilot-plan-v1",
        ),
        "execution-evidence": (
            "stage3.formal_execution_authority",
            "formal_execution_evidence",
            "formal-execution-evidence-v1",
        ),
        "provenance": (
            "stage3.formal_provenance_authority",
            "provenance_record",
            "provenance-record-v1",
        ),
    }
    task_id, artifact_kind, schema_version = declarations[arguments.control_kind]
    if payload.get("schema_version") != schema_version:
        raise ValueError("STAGE3_CONTROL_SCHEMA_MISMATCH")
    if arguments.control_kind == "formal-plan":
        _validate_known_artifact(payload)
        if (
            payload.get("scope") != "formal"
            or payload.get("formal_eligible") is not True
            or payload.get("plan_kind") not in {"pilot", "matrix"}
        ):
            raise ValueError("STAGE3_FORMAL_PLAN_NOT_QUALIFIED")
    elif arguments.control_kind == "execution-evidence":
        FormalExecutionEvidence.from_mapping(payload).require_for_stage(3)
    else:
        provenance = ProvenanceRecord.from_mapping(dict(payload))
        if (
            provenance.scope != "formal"
            or not provenance.formal_eligible
            or provenance.status.value != "COMPLETED"
            or not provenance.worktree_clean
        ):
            raise ValueError("STAGE3_PROVENANCE_NOT_COMPLETED_CLEAN_FORMAL")
    refs = tuple(dict.fromkeys((source_ref, *arguments.source_ref)))
    published = TaskArtifactStore(root, arguments.output_dir).publish(
        task_id=task_id,
        artifact_kind=artifact_kind,
        config_hash=arguments.config_hash,
        run_intent="formal",
        payload=payload,
        formal_eligible=True,
        source_refs=refs,
    )
    result = {
        "schema_version": "stage3-control-publication-v1",
        "control_kind": arguments.control_kind,
        "task_id": task_id,
        "artifact_kind": artifact_kind,
        "config_hash": arguments.config_hash,
        "source_ref": source_ref,
        "commit_ref": published.commit_ref,
        "object_ref": published.object_ref,
        "artifact_hash": published.artifact_hash,
    }
    _emit(result)
    return 0


def _stage3_gate_evaluate(arguments: argparse.Namespace) -> int:
    """Evaluate the independent Stage 3 G3-0..G3-5 evidence chain.

    This command only reads canonical artifacts and writes the audit artifact;
    it never starts a formal runner.  Exit code 3 denotes a valid but blocked
    evidence audit, matching the existing formal-readiness convention.
    """

    from .contracts import FormalExecutionEvidence, GateRecord, write_canonical_json
    from .experiments.stage3_gate import Stage3GateEvaluator

    execution = FormalExecutionEvidence.from_mapping(
        _load_mapping(arguments.execution_evidence)
    )
    gates = tuple(
        GateRecord.from_mapping(_load_mapping(path)) for path in arguments.gate
    )
    scope_decision = _load_mapping(arguments.stage3_scope_decision)
    scope_gate = GateRecord.from_mapping(
        _load_mapping(arguments.stage3_scope_gate)
    )
    observations_value = _load_mapping(arguments.observations)
    observations = observations_value.get("observations", observations_value)
    if not isinstance(observations, list):
        raise TypeError("STAGE3_GATE_OBSERVATIONS_MUST_BE_ARRAY")
    thresholds = _load_mapping(arguments.thresholds)
    provenance = (
        None
        if arguments.provenance is None
        else _load_mapping(arguments.provenance)
    )
    source_refs = tuple(
        dict.fromkeys(
            (
                *arguments.source_artifact_ref,
                arguments.stage3_scope_decision_ref,
                arguments.stage3_scope_gate_ref,
            )
        )
    )
    result = Stage3GateEvaluator().evaluate(
        evaluation_id=arguments.evaluation_id,
        execution=execution,
        observations=observations,
        formal_plan=_load_mapping(arguments.formal_plan),
        formal_plan_ref=arguments.formal_plan_ref,
        thresholds=thresholds,
        gates=gates or None,
        provenance=provenance,
        source_artifact_refs=source_refs,
        stage3_scope_decision=scope_decision,
        stage3_scope_gate=scope_gate,
        stage3_scope_decision_ref=arguments.stage3_scope_decision_ref,
        stage3_scope_gate_ref=arguments.stage3_scope_gate_ref,
    )
    write_canonical_json(arguments.output, result.to_dict())
    _emit(result.to_dict())
    return 0 if result.formal_eligible else 3


def build_parser(*, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog)
    subparsers = parser.add_subparsers(dest="command", required=True)

    guard = subparsers.add_parser("git-guard")
    guard.add_argument("--repo", type=Path, default=Path.cwd())
    guard.add_argument("--max-bytes", type=int, default=10 * 1024 * 1024)
    guard.add_argument("--allow", action="append", default=[])
    guard.set_defaults(handler=_git_guard)

    storage = subparsers.add_parser("storage-check")
    storage.add_argument("--data-root", type=Path)
    storage.add_argument("--require-writable", action="store_true")
    storage.add_argument("--canary", action="store_true")
    storage.set_defaults(handler=_storage_check)

    budget = subparsers.add_parser("storage-budget-check")
    budget.add_argument("--data-root", type=Path, required=True)
    budget.add_argument("--root-filesystem", type=Path, required=True)
    budget.add_argument("--name", required=True)
    budget.add_argument("--expected-new-bytes", type=int, required=True)
    budget.add_argument(
        "--root-minimum-free-bytes", type=int, default=10 * 1024 * 1024 * 1024
    )
    budget.set_defaults(handler=_storage_budget_check)

    resolve = subparsers.add_parser("config-resolve")
    resolve.add_argument("inputs", type=Path, nargs="+")
    resolve.add_argument("--output", type=Path)
    resolve.set_defaults(handler=_config_resolve)

    resolve_v2 = subparsers.add_parser(
        "config-resolve-v2",
        help="从 v1 科学配置层和严格执行 overrides 构造 ResolvedConfigV2",
    )
    resolve_v2.add_argument("inputs", type=Path, nargs="+")
    resolve_v2.add_argument("--task-id", required=True)
    resolve_v2.add_argument("--overrides", type=Path)
    resolve_v2.add_argument("--output", type=Path)
    resolve_v2.set_defaults(handler=_config_resolve_v2)

    config_diff = subparsers.add_parser("config-diff")
    config_diff.add_argument("left", type=Path)
    config_diff.add_argument("right", type=Path)
    config_diff.set_defaults(handler=_config_diff)

    contract = subparsers.add_parser("contract-validate")
    contract.add_argument("path", type=Path)
    contract.set_defaults(handler=_contract_validate)

    artifact = subparsers.add_parser("artifact-validate")
    artifact.add_argument("path", type=Path)
    artifact.set_defaults(handler=_artifact_validate)

    gates = subparsers.add_parser("gate-summary")
    gates.add_argument("paths", type=Path, nargs="+")
    gates.set_defaults(handler=_gate_summary)

    readiness = subparsers.add_parser(
        "formal-readiness",
        help="fail-closed 检查 formal freeze/decision/Gate 证据链，不启动训练",
    )
    readiness.add_argument("--config", type=Path, required=True)
    readiness.add_argument("--freeze", type=Path, action="append", default=[])
    readiness.add_argument("--decision", type=Path)
    readiness.add_argument("--gate", type=Path, action="append", default=[])
    readiness.add_argument("--required-stage", type=int, action="append")
    readiness.add_argument("--required-gate", action="append", default=[])
    readiness.set_defaults(handler=_formal_readiness)

    fixture = subparsers.add_parser("local-fixture")
    fixture.add_argument(
        "--config",
        type=Path,
        default=Path("configs/local-fixtures/resolved-config-v1.json"),
    )
    fixture.add_argument("--output-dir", type=Path, required=True)
    fixture.set_defaults(handler=_local_fixture)

    report = subparsers.add_parser("report-build")
    report.add_argument("--source", type=Path, required=True)
    report.add_argument("--report-id", required=True)
    report.add_argument("--value-field", default="value")
    report.add_argument("--reference-field")
    report.add_argument("--top-q", type=float, default=0.1)
    report.add_argument("--output-json", type=Path, required=True)
    report.add_argument("--output-markdown", type=Path, required=True)
    report.set_defaults(handler=_report_build)

    # ``task`` 是 v0.4 起的唯一 Stage 0--9 编排入口。子命令都只读取 v2；v1
    # fixture 仍由上面的兼容命令负责，防止正式入口静默猜测执行字段。
    task = subparsers.add_parser("task", help="统一任务目录与生命周期入口")
    task_commands = task.add_subparsers(dest="task_command", required=True)

    task_catalog = task_commands.add_parser("catalog", help="查看 hash-bound 任务目录")
    task_catalog.add_argument("--task-id")
    task_catalog.add_argument("--output", type=Path)
    task_catalog.set_defaults(handler=_task_catalog)

    task_environment = task_commands.add_parser(
        "environment-build",
        help="构造 hash-bound runtime environment；evidence 必须引用 workspace 内 commit",
    )
    task_environment.add_argument("--capability", action="append", default=[])
    task_environment.add_argument(
        "--contract-stage", type=int, action="append", default=[]
    )
    task_environment.add_argument("--passed-gate", action="append", default=[])
    task_environment.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="KEY=COMMIT_REF",
    )
    task_environment.add_argument("--decision-ref")
    task_environment.add_argument("--output", type=Path, required=True)
    task_environment.set_defaults(handler=_task_environment_build)

    task_preflight = task_commands.add_parser(
        "preflight", help="只读检查 runner、Gate、合同、能力和资产前置条件"
    )
    task_preflight.add_argument("--config", type=Path, required=True)
    task_preflight.add_argument("--environment", type=Path)
    task_preflight.add_argument("--output", type=Path)
    task_preflight.set_defaults(handler=_task_preflight)

    def add_task_execute_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--environment", type=Path)
        command.add_argument(
            "--result",
            type=Path,
            help="可选不可变 TaskRunResult 输出；省略时仅向 stdout 输出",
        )

    task_run = task_commands.add_parser("run", help="执行新任务")
    add_task_execute_arguments(task_run)
    task_run.set_defaults(handler=_execute_task)

    task_stage3_trajectory = task_commands.add_parser(
        "stage3-trajectory",
        help="运行真实 Stage3.03 training endpoint 轨迹并仅发布 completion receipt",
    )
    task_stage3_trajectory.add_argument("--config", type=Path, required=True)
    task_stage3_trajectory.add_argument("--environment", type=Path, required=True)
    task_stage3_trajectory.add_argument(
        "--receipt",
        type=Path,
        required=True,
        help="DATA_ROOT 内不可变 receipt 路径",
    )
    task_stage3_trajectory.set_defaults(handler=_task_stage3_trajectory)

    task_resume = task_commands.add_parser(
        "resume", help="按 v2 recovery.resume_ref 从权威边界恢复"
    )
    add_task_execute_arguments(task_resume)
    task_resume.set_defaults(handler=_execute_task)

    task_status = task_commands.add_parser("status", help="严格读取已发布任务结果")
    task_status.add_argument("--result", type=Path, required=True)
    task_status.add_argument("--result-ref")
    task_status.add_argument("--config", type=Path)
    task_status.add_argument("--output", type=Path)
    task_status.set_defaults(handler=_task_status)

    task_replay = task_commands.add_parser(
        "replay", help="重新执行并比较完整 result hash"
    )
    task_replay.add_argument("--config", type=Path, required=True)
    task_replay.add_argument("--environment", type=Path)
    task_replay.add_argument("--source-result", type=Path, required=True)
    task_replay.add_argument("--source-ref")
    task_replay.add_argument("--result", type=Path, required=True)
    task_replay.add_argument("--result-ref")
    task_replay.add_argument("--output", type=Path)
    task_replay.set_defaults(handler=_task_replay)

    task_finalize = task_commands.add_parser(
        "finalize", help="为 PASS 结果发布不可变最终确认"
    )
    task_finalize.add_argument("--result", type=Path, required=True)
    task_finalize.add_argument("--result-ref")
    task_finalize.add_argument("--output", type=Path, required=True)
    task_finalize.set_defaults(handler=_task_finalize)

    task_fixture_all = task_commands.add_parser(
        "fixture-all",
        help="在指定本机工作根执行 Stage 0--9 缩小确定性流水线",
    )
    task_fixture_all.add_argument("--workspace-root", type=Path, required=True)
    task_fixture_all.add_argument(
        "--base-config",
        type=Path,
        default=Path("configs/local-fixtures/resolved-config-v1.json"),
    )
    task_fixture_all.set_defaults(handler=_task_fixture_all)

    task_tensorboard = task_commands.add_parser(
        "tensorboard-rebuild",
        help="从一个或多个 typed JSONL 事件流重建 TensorBoard 派生视图",
    )
    task_tensorboard.add_argument(
        "--event", type=Path, action="append", required=True
    )
    task_tensorboard.add_argument("--output-dir", type=Path, required=True)
    task_tensorboard.set_defaults(handler=_task_tensorboard_rebuild)

    asset_group = subparsers.add_parser(
        "asset", help="资产 HTTP 获取、本地登记与完整性验证"
    )
    asset_commands = asset_group.add_subparsers(dest="asset_command", required=True)
    asset_fetch_http = asset_commands.add_parser(
        "fetch-http",
        help="从环境变量或 stdin 提供的运行时 URL 获取固定身份资产",
    )
    asset_fetch_http.add_argument("--spec", type=Path, required=True)
    asset_fetch_http.add_argument("--asset-root", type=Path, required=True)
    asset_fetch_http.add_argument("--final-path", required=True)
    asset_fetch_url_source = asset_fetch_http.add_mutually_exclusive_group(
        required=True
    )
    asset_fetch_url_source.add_argument(
        "--url-env",
        help="只传环境变量名；运行时 URL 值不得进入 argv",
    )
    asset_fetch_url_source.add_argument(
        "--url-stdin",
        action="store_true",
        help="从 stdin 读取一行运行时 URL",
    )
    asset_fetch_http.add_argument("--output", type=Path)
    asset_fetch_http.add_argument("--max-attempts", type=int, default=4)
    asset_fetch_http.add_argument(
        "--request-timeout-seconds", type=float, default=30.0
    )
    asset_fetch_http.add_argument(
        "--overall-timeout-seconds", type=float, default=600.0
    )
    asset_fetch_http.add_argument(
        "--initial-backoff-seconds", type=float, default=0.5
    )
    asset_fetch_http.add_argument(
        "--max-backoff-seconds", type=float, default=8.0
    )
    asset_fetch_http.add_argument(
        "--lock-timeout-seconds", type=float, default=30.0
    )
    asset_fetch_http.add_argument(
        "--lock-poll-interval-seconds", type=float, default=0.05
    )
    asset_fetch_http.add_argument("--chunk-size", type=int, default=1024 * 1024)
    asset_fetch_http.set_defaults(handler=_asset_fetch_http)

    asset_acquire = asset_commands.add_parser(
        "acquire", help="把已有本地文件登记为 downloaded candidate"
    )
    asset_acquire.add_argument("--manifest", type=Path, required=True)
    asset_acquire.add_argument("--source-root", type=Path, required=True)
    asset_acquire.add_argument("--actor", required=True)
    asset_acquire.add_argument("--evidence-ref", required=True)
    asset_acquire.add_argument("--summary", default="local existing files acquired")
    asset_acquire.add_argument("--at")
    asset_acquire.add_argument("--output", type=Path, required=True)
    asset_acquire.set_defaults(handler=_asset_acquire)

    asset_verify = asset_commands.add_parser(
        "verify", help="只读复核本地资产的 size 和 SHA-256"
    )
    asset_verify.add_argument("--manifest", type=Path, required=True)
    asset_verify.add_argument("--asset-root", type=Path, required=True)
    asset_verify.add_argument("--output", type=Path)
    asset_verify.set_defaults(handler=_asset_verify)

    artifact_group = subparsers.add_parser(
        "artifact", help="不可变 artifact 审核与批准"
    )
    artifact_commands = artifact_group.add_subparsers(
        dest="artifact_command", required=True
    )
    artifact_review = artifact_commands.add_parser("review")
    artifact_review.add_argument("--artifact", type=Path, required=True)
    artifact_review.add_argument("--artifact-ref", required=True)
    artifact_review.add_argument(
        "--scope", choices=("local_fixture", "formal"), required=True
    )
    artifact_review.add_argument("--reviewer", required=True)
    artifact_review.add_argument(
        "--decision", choices=("APPROVE", "REJECT", "NEEDS_CHANGES"), required=True
    )
    artifact_review.add_argument("--finding", action="append", default=[])
    artifact_review.add_argument("--metadata", type=Path)
    artifact_review.add_argument("--output", type=Path, required=True)
    artifact_review.set_defaults(handler=_artifact_review)

    artifact_approve = artifact_commands.add_parser("approve")
    artifact_approve.add_argument("--review", type=Path, required=True)
    artifact_approve.add_argument("--review-ref", required=True)
    artifact_approve.add_argument(
        "--scope", choices=("local_validation", "formal"), required=True
    )
    artifact_approve.add_argument("--approver", required=True)
    artifact_approve.add_argument("--output", type=Path, required=True)
    artifact_approve.set_defaults(handler=_artifact_approve)

    stage2_experiment_plan = artifact_commands.add_parser(
        "stage2-experiment-plan-build",
        help="把 Stage 2 formal B/M/R 源编译为 evidence/sampling/lineage-bound 计划",
    )
    stage2_experiment_plan.add_argument("--spec", type=Path, required=True)
    stage2_experiment_plan.add_argument(
        "--sampling-plan",
        type=Path,
        required=True,
        help="sampling-plan-v1 或包含它的 canonical task artifact",
    )
    stage2_experiment_plan.add_argument(
        "--formal-execution-evidence",
        type=Path,
        required=True,
    )
    stage2_experiment_plan.add_argument(
        "--source-ref",
        action="append",
        required=True,
        help="重复传入当前任务全部直接前驱 task commit 引用",
    )
    stage2_experiment_plan.add_argument("--output", type=Path, required=True)
    stage2_experiment_plan.set_defaults(
        handler=_artifact_stage2_experiment_plan_build
    )

    endpoint_plan = artifact_commands.add_parser(
        "endpoint-plan-build",
        help="从声明式参数构建训练 endpoint 选择计划并自动计算 hash",
    )
    endpoint_plan.add_argument("--plan-id", required=True)
    endpoint_plan.add_argument("--step", type=int, action="append", default=[])
    endpoint_plan.add_argument("--include-checkpoint-steps", action="store_true")
    endpoint_plan.add_argument(
        "--scope", choices=("local_fixture", "pilot", "formal"), required=True
    )
    endpoint_plan.add_argument("--formal-execution-evidence", type=Path)
    endpoint_plan.add_argument("--probe-plan-ref")
    endpoint_plan.add_argument(
        "--endpoint-metadata",
        type=Path,
        help="pilot/formal 必填：step 到 model/seed/stage 的冻结 JSON/YAML mapping",
    )
    endpoint_plan.add_argument("--output", type=Path, required=True)
    endpoint_plan.set_defaults(handler=_artifact_endpoint_plan_build)

    probe_plan = artifact_commands.add_parser(
        "probe-plan-build",
        help="把不含顶层 hash 的 probe YAML/JSON spec 编译为冻结输入",
    )
    probe_plan.add_argument("--spec", type=Path, required=True)
    probe_plan.add_argument(
        "--scope", choices=("local_fixture", "pilot", "formal"), required=True
    )
    probe_plan.add_argument("--formal-execution-evidence", type=Path)
    probe_plan.add_argument("--output", type=Path, required=True)
    probe_plan.set_defaults(handler=_artifact_probe_plan_build)

    quadrature_pilot_plan = artifact_commands.add_parser(
        "quadrature-pilot-plan-build",
        help="把 Stage 3 formal pilot 源编译为 evidence-bound 冻结计划",
    )
    quadrature_pilot_plan.add_argument("--spec", type=Path, required=True)
    quadrature_pilot_plan.add_argument(
        "--formal-execution-evidence", type=Path, required=True
    )
    quadrature_pilot_plan.add_argument("--output", type=Path, required=True)
    quadrature_pilot_plan.set_defaults(
        handler=_artifact_quadrature_pilot_plan_build
    )

    stage3_production_index = artifact_commands.add_parser(
        "stage3-production-index-build",
        help="从真实 endpoint commits/probe plans 构建严格的 12/99 unit index",
    )
    stage3_production_index.add_argument(
        "--endpoint-commit", type=Path, action="append", required=True
    )
    stage3_production_index.add_argument(
        "--probe-plan", type=Path, action="append", required=True
    )
    stage3_production_index.add_argument(
        "--scope", choices=("pilot", "formal"), required=True
    )
    stage3_production_index.add_argument(
        "--workspace-root", type=Path, required=True
    )
    stage3_production_index.add_argument("--index-id")
    stage3_production_index.add_argument("--output", type=Path, required=True)
    stage3_production_index.set_defaults(
        handler=_artifact_stage3_production_index_build
    )

    stage3_path_unit_selector = artifact_commands.add_parser(
        "stage3-path-unit-selector-build",
        help="构建绑定 production index 的单一 Stage 3 path-unit selector",
    )
    stage3_path_unit_selector.add_argument(
        "--unit-index", type=Path, required=True
    )
    stage3_path_unit_selector.add_argument(
        "--scope", choices=("pilot", "formal"), required=True
    )
    stage3_path_unit_selector.add_argument("--active-unit-id", required=True)
    stage3_path_unit_selector.add_argument("--output", type=Path, required=True)
    stage3_path_unit_selector.set_defaults(
        handler=_artifact_stage3_path_unit_selector_build
    )

    stage3_probe_selector = artifact_commands.add_parser(
        "stage3-probe-selector-build",
        help="为 S3.03/S3.04 构建绑定 formal endpoint/probe plan 的 selector",
    )
    stage3_probe_selector.add_argument(
        "--endpoint-commit", type=Path, required=True
    )
    stage3_probe_selector.add_argument("--probe-plan", type=Path, required=True)
    stage3_probe_selector.add_argument("--active-probe-id", required=True)
    stage3_probe_selector.add_argument("--output", type=Path, required=True)
    stage3_probe_selector.set_defaults(
        handler=_artifact_stage3_probe_selector_build
    )

    stage3_scope_authority = artifact_commands.add_parser(
        "stage3-scope-authority-publish",
        help="把已审核的用户 G3-0 范围决定发布为 formal task commits",
    )
    stage3_scope_authority.add_argument(
        "--workspace-root", type=Path, required=True
    )
    stage3_scope_authority.add_argument("--output-dir", required=True)
    stage3_scope_authority.add_argument("--decision", type=Path, required=True)
    stage3_scope_authority.add_argument("--gate", type=Path, required=True)
    stage3_scope_authority.add_argument("--config-hash", required=True)
    stage3_scope_authority.set_defaults(
        handler=_artifact_stage3_scope_authority_publish
    )

    stage3_control_publish = artifact_commands.add_parser(
        "stage3-control-publish",
        help="把已严格验证的 Stage 3 plan/execution/provenance 包装为 formal commit",
    )
    stage3_control_publish.add_argument(
        "--control-kind",
        choices=("formal-plan", "execution-evidence", "provenance"),
        required=True,
    )
    stage3_control_publish.add_argument(
        "--workspace-root", type=Path, required=True
    )
    stage3_control_publish.add_argument("--output-dir", required=True)
    stage3_control_publish.add_argument("--source", type=Path, required=True)
    stage3_control_publish.add_argument("--source-ref", action="append", default=[])
    stage3_control_publish.add_argument("--config-hash", required=True)
    stage3_control_publish.set_defaults(handler=_artifact_stage3_control_publish)

    stage3_g36_publish = artifact_commands.add_parser(
        "stage3-g36-publish",
        help="从已提交 Stage3.08 表与 formal authority 独立发布 G3-6",
    )
    stage3_g36_publish.add_argument("--workspace-root", type=Path, required=True)
    stage3_g36_publish.add_argument("--output-dir", required=True)
    stage3_g36_publish.add_argument("--config-hash", required=True)
    stage3_g36_publish.add_argument("--publication-id")
    stage3_g36_publish.add_argument("--frozen-source-table-ref", required=True)
    stage3_g36_publish.add_argument("--provenance-ref", required=True)
    stage3_g36_publish.add_argument("--formal-plan-ref", required=True)
    stage3_g36_publish.add_argument("--execution-evidence-ref", required=True)
    stage3_g36_publish.add_argument("--stage3-scope-decision-ref", required=True)
    stage3_g36_publish.add_argument("--stage3-scope-gate-ref", required=True)
    stage3_g36_publish.add_argument("--receipt", type=Path)
    stage3_g36_publish.set_defaults(handler=_artifact_stage3_g36_publish)

    stage3_g37_publish = artifact_commands.add_parser(
        "stage3-g37-publish",
        help="从已提交 Stage3.09 成本表与候选、G3-6 权威独立发布 G3-7",
    )
    stage3_g37_publish.add_argument("--workspace-root", type=Path, required=True)
    stage3_g37_publish.add_argument("--output-dir", required=True)
    stage3_g37_publish.add_argument("--config-hash")
    stage3_g37_publish.add_argument("--publication-id")
    stage3_g37_publish.add_argument("--frozen-source-table-ref", required=True)
    stage3_g37_publish.add_argument("--cost-accuracy-table-ref", required=True)
    stage3_g37_publish.add_argument("--quadrature-decision-ref", required=True)
    stage3_g37_publish.add_argument("--formal-plan-ref", required=True)
    stage3_g37_publish.add_argument("--execution-evidence-ref", required=True)
    stage3_g37_publish.add_argument("--provenance-ref", required=True)
    stage3_g37_publish.add_argument("--evaluation-ref", required=True)
    stage3_g37_publish.add_argument("--g3-6-ref", required=True)
    stage3_g37_publish.add_argument("--stage3-scope-decision-ref")
    stage3_g37_publish.add_argument("--stage3-scope-gate-ref")
    stage3_g37_publish.add_argument("--receipt", type=Path)
    stage3_g37_publish.set_defaults(handler=_artifact_stage3_g37_publish)

    stage3_g38_publish = artifact_commands.add_parser(
        "stage3-g38-publish",
        help="独立验收 Stage3.10 交付文件与 G0–G7 权威并发布 G3-8",
    )
    stage3_g38_publish.add_argument("--workspace-root", type=Path, required=True)
    stage3_g38_publish.add_argument("--output-dir", required=True)
    stage3_g38_publish.add_argument("--config-hash")
    stage3_g38_publish.add_argument("--publication-id")
    stage3_g38_publish.add_argument("--publication-config-hash")
    stage3_g38_publish.add_argument("--checked-at")
    stage3_g38_publish.add_argument(
        "--gate-ref",
        action="append",
        required=True,
        help="重复八次：stage3.G3-N=commit/ref.json",
    )
    stage3_g38_publish.add_argument("--analysis-report-ref", required=True)
    stage3_g38_publish.add_argument("--chart-artifacts-ref", required=True)
    stage3_g38_publish.add_argument("--handoff-manifest-ref", required=True)
    stage3_g38_publish.add_argument("--gate-summary-ref", required=True)
    stage3_g38_publish.add_argument("--execution-evidence-ref", required=True)
    stage3_g38_publish.add_argument("--g3-7-publication-ref", required=True)
    stage3_g38_publish.add_argument("--recommendation-ref", required=True)
    stage3_g38_publish.add_argument("--finalization-ref", required=True)
    stage3_g38_publish.add_argument("--delivery-manifest-ref", required=True)
    stage3_g38_publish.add_argument("--receipt", type=Path)
    stage3_g38_publish.set_defaults(handler=_artifact_stage3_g38_publish)

    route_build = artifact_commands.add_parser(
        "route-build",
        help="把 training-route-source-v1 声明编译为 hash-bound TrainingRouteSpec",
    )
    route_build.add_argument("--spec", type=Path, required=True)
    route_build.add_argument("--decision", type=Path)
    route_build.add_argument("--gate", type=Path)
    route_build.add_argument("--output", type=Path, required=True)
    route_build.set_defaults(handler=_artifact_route_build)

    ablation_build = artifact_commands.add_parser(
        "ablation-matrix-build",
        help="把 ablation-matrix-source-v1 声明编译为声明 artifact 与可选完整矩阵",
    )
    ablation_build.add_argument("--spec", type=Path, required=True)
    ablation_build.add_argument("--output", type=Path, required=True)
    ablation_build.add_argument("--compiled-output", type=Path)
    ablation_build.set_defaults(handler=_artifact_ablation_matrix_build)

    gate_group = subparsers.add_parser("gate", help="正式 Gate 构建与汇总")
    gate_commands = gate_group.add_subparsers(dest="gate_command", required=True)
    gate_build = gate_commands.add_parser("build")
    gate_build.add_argument("--gate-id", required=True)
    gate_build.add_argument("--stage", type=int, required=True)
    gate_build.add_argument(
        "--status",
        choices=(
            "PASS",
            "CONDITIONALLY_ACCEPTED",
            "FAIL",
            "BLOCKED",
            "STALE",
            "NOT_RUN",
        ),
        required=True,
    )
    gate_build.add_argument("--checked-at", required=True)
    gate_build.add_argument("--measured-json", type=Path)
    gate_build.add_argument("--threshold-json", type=Path)
    gate_build.add_argument("--evidence-ref", action="append", default=[])
    gate_build.add_argument("--reason", action="append", default=[])
    gate_build.add_argument("--condition", action="append", default=[])
    gate_build.add_argument("--expires-at")
    gate_build.add_argument("--output", type=Path, required=True)
    gate_build.set_defaults(handler=_gate_build)

    gate_nested_summary = gate_commands.add_parser("summary")
    gate_nested_summary.add_argument("paths", type=Path, nargs="+")
    gate_nested_summary.set_defaults(handler=_gate_summary)

    gate_stage3 = gate_commands.add_parser(
        "stage3-evaluate", help="严格评估 Stage 3 G3-0..G3-5 与全量指标"
    )
    gate_stage3.add_argument("--evaluation-id", required=True)
    gate_stage3.add_argument("--execution-evidence", type=Path, required=True)
    gate_stage3.add_argument("--gate", type=Path, action="append", default=[])
    gate_stage3.add_argument("--observations", type=Path, required=True)
    gate_stage3.add_argument("--thresholds", type=Path, required=True)
    gate_stage3.add_argument("--formal-plan", type=Path, required=True)
    gate_stage3.add_argument("--formal-plan-ref", required=True)
    gate_stage3.add_argument("--provenance", type=Path)
    gate_stage3.add_argument("--source-artifact-ref", action="append", default=[])
    gate_stage3.add_argument(
        "--stage3-scope-decision",
        "--scope-decision",
        dest="stage3_scope_decision",
        type=Path,
        required=True,
        help="hash-bound explicit user G3-0 scope decision",
    )
    gate_stage3.add_argument(
        "--stage3-scope-gate",
        "--scope-gate",
        dest="stage3_scope_gate",
        type=Path,
        required=True,
        help="hash-bound explicit user G3-0 GateRecord",
    )
    gate_stage3.add_argument(
        "--stage3-scope-decision-ref",
        "--scope-decision-ref",
        dest="stage3_scope_decision_ref",
        required=True,
        help="stable source reference for the explicit G3-0 decision",
    )
    gate_stage3.add_argument(
        "--stage3-scope-gate-ref",
        "--scope-gate-ref",
        dest="stage3_scope_gate_ref",
        required=True,
        help="stable source reference for the explicit G3-0 GateRecord",
    )
    gate_stage3.add_argument("--output", type=Path, required=True)
    gate_stage3.set_defaults(handler=_stage3_gate_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    selected_argv = list(sys.argv[1:] if argv is None else argv)
    if _asset_fetch_argv_contains_runtime_url(selected_argv):
        _emit(_asset_fetch_cli_failure("RUNTIME_URL_ARGV_FORBIDDEN"))
        return 2
    arguments = build_parser(prog=Path(sys.argv[0]).stem).parse_args(selected_argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
