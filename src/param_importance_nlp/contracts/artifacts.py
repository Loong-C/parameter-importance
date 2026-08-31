"""无法直接内嵌张量的公共 artifact manifest 验证器。

Stage 2 reference 与 Stage 3 路径结果的数值主体存放在安全 tensor bundle 中；
JSON 只保存合同身份、bundle manifest SHA-256 与可审计标量。这里严格验证 JSON
边界，但不会读取 bundle，也不会把结构正确的本机文件升级为正式 Gate 证据。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from typing import Mapping

from .jsonio import canonical_json_hash
from .status import GateStatus, validate_gate_id


@dataclass(frozen=True, slots=True)
class ValidatedArtifact:
    """严格加载后返回的最小身份；不承载张量或运行时对象。"""

    kind: str
    artifact_hash: str


def _require_exact(value: Mapping[str, object], required: set[str], kind: str) -> None:
    if set(value) != required:
        raise ValueError(
            f"{kind} 字段集合不匹配：missing={sorted(required-set(value))}, "
            f"extra={sorted(set(value)-required)}"
        )


def _require_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{field} 必须是小写 SHA-256")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} 必须是非空字符串")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{field} 必须是 >= {minimum} 的整数且不能是 bool")
    return value


def _require_nullable_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} 必须是有限数或 null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} 必须是有限数")
    return number


def _require_bundle_ref(value: object, field: str) -> str:
    """只接受仓库无关的相对 POSIX 引用，拒绝盘符、反斜线与路径逃逸。"""

    text = _require_text(value, field)
    if "\\" in text:
        raise ValueError(f"{field} 必须使用 POSIX 分隔符")
    path = PurePosixPath(text)
    if text in {".", ".."} or path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{field} 必须是无路径逃逸的相对 artifact 引用")
    if ":" in path.parts[0]:
        raise ValueError(f"{field} 不得包含 Windows 盘符")
    return text


def _verify_artifact_hash(value: Mapping[str, object], *, kind: str) -> str:
    supplied = _require_sha256(value.get("artifact_hash"), f"{kind}.artifact_hash")
    payload = {name: item for name, item in value.items() if name != "artifact_hash"}
    observed = canonical_json_hash(payload)
    if supplied != observed:
        raise ValueError(f"{kind} artifact_hash 与完整 wire object 不一致")
    return supplied


def validate_estimator_decision_artifact(
    value: Mapping[str, object],
    *,
    require_formal: bool = False,
) -> ValidatedArtifact:
    """严格验证 Stage 2 estimator decision 的完整 wire object。

    一般加载允许保存 ``local_fixture``、失败或尚未冻结的决策，以便审计失败
    事实；``require_formal=True`` 则额外要求已经选定合法估计器、B/M/R 完整、
    B 可被 M 整除、Stage 2 Gate 可接受且存在稳定 artifact 引用。该函数位于
    contracts 层，formal readiness 与 experiments loader 可共享同一边界，而
    不会让 contracts 反向依赖实验实现。
    """

    required = {
        "schema_version",
        "decision_id",
        "selected_estimator",
        "scope",
        "status",
        "state",
        "batch_size",
        "microbatch_count",
        "repetitions",
        "gate_id",
        "gate_status",
        "artifact_ref",
        "metadata",
        "artifact_hash",
    }
    _require_exact(value, required, "EstimatorDecision")
    if value["schema_version"] != "estimator-decision-v1":
        raise ValueError("不支持的 EstimatorDecision schema")
    _require_text(value["decision_id"], "decision_id")
    if value["selected_estimator"] not in {None, "u", "weighted_u", "double"}:
        raise ValueError("selected_estimator 不在冻结候选集中")
    if value["scope"] not in {"local_fixture", "formal"}:
        raise ValueError("EstimatorDecision scope 只能是 local_fixture 或 formal")
    _require_text(value["status"], "status")
    if value["state"] not in {"UNFROZEN", "FROZEN", "SELECTED", "PASS", "READY"}:
        raise ValueError("EstimatorDecision state 不受支持")

    parsed_counts: dict[str, int | None] = {}
    for field, minimum in (
        ("batch_size", 1),
        ("microbatch_count", 2),
        ("repetitions", 1),
    ):
        item = value[field]
        parsed_counts[field] = (
            None if item is None else _require_int(item, field, minimum=minimum)
        )
    gate_id = validate_gate_id(value["gate_id"], stage=2)
    gate_status = value["gate_status"]
    if gate_status is not None:
        if not isinstance(gate_status, str):
            raise TypeError("gate_status 必须是字符串或 null")
        GateStatus(gate_status)
    artifact_ref = value["artifact_ref"]
    if artifact_ref is not None:
        _require_bundle_ref(artifact_ref, "artifact_ref")
    if not isinstance(value["metadata"], Mapping):
        raise TypeError("EstimatorDecision metadata 必须是 object")
    digest = _verify_artifact_hash(value, kind="EstimatorDecision")

    if require_formal:
        if value["scope"] != "formal":
            raise ValueError("FORMAL_DECISION_SCOPE_REQUIRED")
        if value["state"] not in {"FROZEN", "SELECTED", "PASS", "READY"}:
            raise ValueError("FORMAL_DECISION_STATE_NOT_FROZEN")
        if value["status"] not in {"PASS", "SELECTED", "QUALIFIED", "READY"}:
            raise ValueError("FORMAL_DECISION_STATUS_NOT_ACCEPTABLE")
        if value["selected_estimator"] not in {"u", "weighted_u", "double"}:
            raise ValueError("FORMAL_DECISION_SELECTION_REQUIRED")
        if any(item is None for item in parsed_counts.values()):
            raise ValueError("FORMAL_DECISION_B_M_R_REQUIRED")
        batch_size = parsed_counts["batch_size"]
        microbatch_count = parsed_counts["microbatch_count"]
        assert batch_size is not None and microbatch_count is not None
        if batch_size % microbatch_count != 0:
            raise ValueError("FORMAL_DECISION_M_MUST_DIVIDE_B")
        if gate_status not in {
            GateStatus.PASS.value,
            GateStatus.CONDITIONALLY_ACCEPTED.value,
        }:
            raise ValueError("FORMAL_DECISION_GATE_NOT_ACCEPTABLE")
        if not gate_id.startswith("stage2.G"):
            raise ValueError("FORMAL_DECISION_GATE_NOT_STAGE2")
        if artifact_ref is None:
            raise ValueError("FORMAL_DECISION_ARTIFACT_REF_REQUIRED")
    return ValidatedArtifact("estimator_decision", digest)


def validate_reference_result_artifact(
    value: Mapping[str, object],
) -> ValidatedArtifact:
    """验证 ``ReferenceResult`` 的 JSON manifest；数值视图由 bundle hash 绑定。"""

    required = {
        "schema_version",
        "reference_id",
        "bias_reference_hash",
        "cross_reference_hash",
        "ranking_reference_hash",
        "sample_count_a",
        "sample_count_b",
        "block_size",
        "registry_hash",
        "scope",
        "formal_eligible",
        "metadata",
        "tensor_bundle_ref",
        "tensor_bundle_manifest_hash",
        "artifact_hash",
    }
    _require_exact(value, required, "ReferenceResult")
    if value["schema_version"] != "reference-result-v1":
        raise ValueError("不支持的 ReferenceResult schema")
    _require_text(value["reference_id"], "reference_id")
    for field in (
        "bias_reference_hash",
        "cross_reference_hash",
        "ranking_reference_hash",
        "registry_hash",
        "tensor_bundle_manifest_hash",
    ):
        _require_sha256(value[field], field)
    for field in ("sample_count_a", "sample_count_b", "block_size"):
        _require_int(value[field], field, minimum=1)
    if value["scope"] not in {"local_fixture", "formal"}:
        raise ValueError("ReferenceResult scope 只能是 local_fixture 或 formal")
    if not isinstance(value["formal_eligible"], bool):
        raise TypeError("ReferenceResult formal_eligible 必须是布尔值")
    if value["scope"] == "local_fixture" and value["formal_eligible"]:
        raise ValueError("local_fixture ReferenceResult 不能 formal_eligible=true")
    if not isinstance(value["metadata"], Mapping):
        raise TypeError("ReferenceResult metadata 必须是 object")
    _require_bundle_ref(value["tensor_bundle_ref"], "tensor_bundle_ref")
    digest = _verify_artifact_hash(value, kind="ReferenceResult")
    return ValidatedArtifact("reference_result_manifest", digest)


def validate_path_spec_artifact(value: Mapping[str, object]) -> ValidatedArtifact:
    """验证路径端点 manifest；实际端点张量仍由两个独立 bundle 承载。"""

    required = {
        "schema_version",
        "path_id",
        "probe_id",
        "loss_id",
        "accumulation_dtype",
        "registry_hash",
        "parameter_names",
        "parameter_pre_bundle_ref",
        "parameter_pre_bundle_manifest_hash",
        "parameter_post_bundle_ref",
        "parameter_post_bundle_manifest_hash",
        "path_identity_hash",
        "artifact_hash",
    }
    _require_exact(value, required, "PathSpec")
    if value["schema_version"] != "path-spec-v1":
        raise ValueError("不支持的 PathSpec schema")
    for field in ("path_id", "probe_id", "loss_id"):
        _require_text(value[field], field)
    if value["accumulation_dtype"] not in {"float32", "float64"}:
        raise ValueError("PathSpec accumulation_dtype 只能是 float32 或 float64")
    for field in (
        "registry_hash",
        "parameter_pre_bundle_manifest_hash",
        "parameter_post_bundle_manifest_hash",
        "path_identity_hash",
    ):
        _require_sha256(value[field], field)
    names = value["parameter_names"]
    if not isinstance(names, list) or not names or not all(
        isinstance(item, str) and item for item in names
    ):
        raise TypeError("PathSpec parameter_names 必须是非空字符串数组")
    if len(names) != len(set(names)):
        raise ValueError("PathSpec parameter_names 不得重复")
    _require_bundle_ref(value["parameter_pre_bundle_ref"], "parameter_pre_bundle_ref")
    _require_bundle_ref(value["parameter_post_bundle_ref"], "parameter_post_bundle_ref")
    digest = _verify_artifact_hash(value, kind="PathSpec")
    return ValidatedArtifact("path_spec_manifest", digest)


def validate_path_integral_result_artifact(
    value: Mapping[str, object],
) -> ValidatedArtifact:
    """验证路径积分标量与贡献 bundle manifest，并核对 FP64 求积权重。"""

    required = {
        "schema_version",
        "path_identity_hash",
        "rule",
        "contribution_bundle_ref",
        "contribution_bundle_manifest_hash",
        "views",
        "endpoint_loss_pre",
        "endpoint_loss_post",
        "loss_drop",
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
        "node_losses",
        "unique_gradient_evaluations",
        "artifact_hash",
    }
    _require_exact(value, required, "PathIntegralResult")
    if value["schema_version"] != "path-integral-result-v1":
        raise ValueError("不支持的 PathIntegralResult schema")
    _require_sha256(value["path_identity_hash"], "path_identity_hash")
    _require_sha256(
        value["contribution_bundle_manifest_hash"],
        "contribution_bundle_manifest_hash",
    )
    _require_bundle_ref(value["contribution_bundle_ref"], "contribution_bundle_ref")
    if value["views"] != ["signed", "positive", "negative_mass", "absolute"]:
        raise ValueError("PathIntegralResult views 必须使用冻结顺序")
    rule = value["rule"]
    rule_fields = {
        "name",
        "version",
        "kind",
        "nodes",
        "weights",
        "subintervals",
        "exact_polynomial_degree",
        "theoretical_order",
    }
    if not isinstance(rule, Mapping):
        raise TypeError("PathIntegralResult rule 必须是 object")
    _require_exact(rule, rule_fields, "QuadratureRule")
    for field in ("name", "version", "kind"):
        _require_text(rule[field], f"rule.{field}")
    nodes, weights = rule["nodes"], rule["weights"]
    if (
        not isinstance(nodes, list)
        or not isinstance(weights, list)
        or not nodes
        or len(nodes) != len(weights)
    ):
        raise ValueError("rule.nodes/weights 必须是等长非空数组")
    parsed_nodes = [_require_nullable_number(item, "rule.nodes[]") for item in nodes]
    parsed_weights = [_require_nullable_number(item, "rule.weights[]") for item in weights]
    if any(item is None or not 0.0 <= item <= 1.0 for item in parsed_nodes):
        raise ValueError("rule.nodes 必须位于 [0,1]")
    if any(item is None or item < 0.0 for item in parsed_weights):
        raise ValueError("rule.weights 必须非负")
    if not math.isclose(
        sum(item for item in parsed_weights if item is not None),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("rule.weights 之和必须为 1")
    if rule["subintervals"] is not None:
        _require_int(rule["subintervals"], "rule.subintervals", minimum=1)
    _require_int(rule["exact_polynomial_degree"], "rule.exact_polynomial_degree")
    _require_int(rule["theoretical_order"], "rule.theoretical_order", minimum=1)
    for field in (
        "endpoint_loss_pre",
        "endpoint_loss_post",
        "loss_drop",
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
    ):
        _require_nullable_number(value[field], field)
    for field in (
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
    ):
        residual = _require_nullable_number(value[field], field)
        if residual is not None and residual < 0:
            raise ValueError(f"{field} 不能为负")
    node_losses = value["node_losses"]
    if not isinstance(node_losses, list) or len(node_losses) != len(nodes):
        raise ValueError("node_losses 必须与 rule.nodes 等长")
    for item in node_losses:
        _require_nullable_number(item, "node_losses[]")
    evaluations = _require_int(
        value["unique_gradient_evaluations"],
        "unique_gradient_evaluations",
        minimum=1,
    )
    if evaluations > len(set(float(item) for item in parsed_nodes if item is not None)):
        raise ValueError("unique_gradient_evaluations 超过唯一求积节点数")
    digest = _verify_artifact_hash(value, kind="PathIntegralResult")
    return ValidatedArtifact("path_integral_result_manifest", digest)
