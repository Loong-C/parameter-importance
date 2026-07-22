from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    if set(CONFIG_SECTIONS).issubset(value):
        config = ResolvedConfig.from_mapping(value)
        return "resolved_config", config.config_hash
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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser(prog=Path(sys.argv[0]).stem).parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (ValueError, TypeError, KeyError, FileNotFoundError) as exc:
        print(f"ERROR:{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
