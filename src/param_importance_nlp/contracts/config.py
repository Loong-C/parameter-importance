"""Stage 0–9 共用的严格 resolved config 合同。

配置被固定为十五个互不重叠的分区。每个分区只接受本模块声明的字段；覆盖层不能
借由新增键把临时参数偷偷送进运行时。``ResolvedConfig`` 会展开所有默认值、执行
类型与跨字段验证，并给出两个摘要：

``full_hash``
    包含完整 resolved config，适合证明一次启动实际读取了什么；
``config_hash``
    排除 ``runtime`` 中三个易变的逻辑落盘根目录，作为语义实验身份。模型/数据
    逻辑 ID、不可变 revision、损失、batch、估计器与分析设置仍全部进入摘要。

路径字段只保存规范化的 POSIX 风格逻辑路径，不能是绝对路径或包含 ``..``。实际
映射到本机/服务器授权根目录由 runtime resolver 负责，这样配置本身不会把某台机器
的用户目录写入实验身份。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Final

from .errors import ConfigContractError
from .jsonio import JSONValue, canonical_json_bytes, canonical_json_hash


_MISSING = object()


@dataclass(frozen=True, slots=True)
class FieldRule:
    """单个配置字段的 wire 类型、默认值和枚举约束。"""

    types: tuple[type, ...]
    default: Any = _MISSING
    choices: frozenset[Any] | None = None
    non_empty: bool = False


def _rule(
    *types: type,
    default: Any = _MISSING,
    choices: Sequence[Any] | None = None,
    non_empty: bool = False,
) -> FieldRule:
    return FieldRule(
        types=tuple(types),
        default=default,
        choices=None if choices is None else frozenset(choices),
        non_empty=non_empty,
    )


CONFIG_SCHEMA_VERSION: Final = "resolved-config-v1"
CONFIG_SECTIONS: Final = (
    "identity",
    "runtime",
    "model",
    "data",
    "loss",
    "batching",
    "distributed",
    "precision",
    "optimizer",
    "logging",
    "checkpoint",
    "importance",
    "sampling",
    "path_integration",
    "pruning",
    "analysis",
)


_FIELD_RULES: Final[dict[str, dict[str, FieldRule]]] = {
    "identity": {
        "schema_version": _rule(str, default=CONFIG_SCHEMA_VERSION),
        "stage": _rule(int),
        "task": _rule(str, non_empty=True),
        "route": _rule(str, non_empty=True),
        "master_seed": _rule(int),
        "run_intent": _rule(str, choices=("local_fixture", "formal")),
        # 这是派生字段；调用者若显式提供，必须与 run_intent 推导结果一致。
        "formal_eligible": _rule(bool, default=False),
        "parent_experiment_id": _rule(str, type(None), default=None),
        "input_run_id": _rule(str, type(None), default=None),
        "input_checkpoint_id": _rule(str, type(None), default=None),
    },
    "runtime": {
        "environment_id": _rule(str, non_empty=True),
        "offline": _rule(bool, default=True),
        "output_root": _rule(str, non_empty=True),
        "temp_root": _rule(str, non_empty=True),
        "cache_root": _rule(str, non_empty=True),
        "device": _rule(str, default="cpu", choices=("cpu", "cuda")),
        "dependency_profile": _rule(str, default="cpu-core", non_empty=True),
        "allow_dirty_worktree": _rule(bool, default=False),
    },
    "model": {
        "asset_id": _rule(str, non_empty=True),
        "revision": _rule(str, non_empty=True),
        "tokenizer_asset_id": _rule(str, type(None), default=None),
        "initialization_id": _rule(str, non_empty=True),
        "architecture": _rule(str, non_empty=True),
    },
    "data": {
        "asset_id": _rule(str, non_empty=True),
        "revision": _rule(str, non_empty=True),
        "split": _rule(str, non_empty=True),
        "sequence_length": _rule(int),
        "sampler": _rule(str, default="with_replacement", non_empty=True),
        "statistical_unit": _rule(str, non_empty=True),
        "weight_unit": _rule(str, non_empty=True),
        "sampling_design": _rule(str, non_empty=True),
        "weights_exogenous": _rule(bool),
        "common_mean_assumption": _rule(bool),
    },
    "loss": {
        "task_type": _rule(
            str,
            choices=("causal_lm", "sequence_classification", "synthetic"),
        ),
        "reduction": _rule(str, default="mean", choices=("mean", "sum")),
        "ignore_index": _rule(int, default=-100),
        "weighting": _rule(
            str,
            default="sample",
            choices=("sample", "effective_token", "explicit"),
        ),
    },
    "batching": {
        "per_device_batch_size": _rule(int),
        "global_batch_size": _rule(int),
        "microbatch_size": _rule(int),
        "accumulation_steps": _rule(int, default=1),
        "no_sync": _rule(bool, default=False),
    },
    "distributed": {
        "world_size": _rule(int, default=1),
        "backend": _rule(str, default="local", choices=("local", "gloo", "nccl")),
        "device_ids": _rule(list, default=[]),
        "timeout_seconds": _rule(int, default=60),
    },
    "precision": {
        "compute_dtype": _rule(
            str, default="float32", choices=("float32", "bfloat16", "float16")
        ),
        "gradient_dtype": _rule(str, default="float32", choices=("float32",)),
        "statistic_dtype": _rule(str, default="float32", choices=("float32",)),
        "reference_dtype": _rule(str, default="float64", choices=("float64",)),
        "quadrature_weight_dtype": _rule(
            str, default="float64", choices=("float64",)
        ),
        "path_accumulation_dtype": _rule(
            str, default="float64", choices=("float32", "float64")
        ),
        "amp": _rule(bool, default=False),
    },
    "optimizer": {
        "type": _rule(str, default="sgd", choices=("sgd", "momentum", "adamw")),
        "learning_rate": _rule(int, float),
        "weight_decay": _rule(int, float, default=0.0),
        "momentum": _rule(int, float, default=0.0),
        "fused": _rule(bool, default=False),
        "foreach": _rule(bool, default=False),
        "parameter_groups": _rule(list, default=[]),
    },
    "logging": {
        "event_format": _rule(str, default="jsonl-v1", choices=("jsonl-v1",)),
        "log_every_steps": _rule(int, default=1),
        "tensorboard": _rule(bool, default=False),
    },
    "checkpoint": {
        "schema_version": _rule(str, default="checkpoint-bundle-v1"),
        "save_every_steps": _rule(int, default=100),
        "max_to_keep": _rule(int, default=2),
        "two_phase_commit": _rule(bool, default=True),
    },
    "importance": {
        "estimator_decision_ref": _rule(str, type(None), default=None),
        "estimator_name": _rule(
            str,
            type(None),
            default=None,
            choices=(None, "raw", "double", "u", "weighted_u"),
        ),
        "clip_mode": _rule(str, default="none", choices=("none", "global_plugin")),
        "accumulate_views": _rule(
            list,
            default=[
                "signed",
                "positive",
                "negative_mass",
                "absolute",
                "raw",
                "data_movement",
            ],
        ),
        "require_decision_for_formal": _rule(bool, default=True),
    },
    "sampling": {
        "universe_version": _rule(str, non_empty=True),
        "candidate_batch_sizes": _rule(list, default=[32, 64, 128, 256]),
        "candidate_microbatch_counts": _rule(list, default=[2, 4, 8, 16, 32]),
        "microbatch_preference": _rule(list, default=[32, 16, 8, 4]),
        "repetition_count": _rule(int, type(None), default=None),
        "reference_batch_size": _rule(int, type(None), default=None),
    },
    "path_integration": {
        "enabled": _rule(bool, default=False),
        "path_name": _rule(str, default="full_update_linear", non_empty=True),
        "default_rule": _rule(str, type(None), default=None),
        "fallback_rule": _rule(str, type(None), default=None),
        "probe_count": _rule(int, type(None), default=None),
        "node_budget": _rule(int, type(None), default=None),
        "thresholds_ref": _rule(str, type(None), default=None),
    },
    "pruning": {
        "enabled": _rule(bool, default=False),
        "strategy": _rule(
            str,
            default="none",
            choices=("none", "highest", "lowest", "random"),
        ),
        "scope": _rule(str, default="global", choices=("global", "layer_balanced")),
        "ratios": _rule(list, default=[]),
        "random_repetitions": _rule(int, default=0),
    },
    "analysis": {
        "schema_version": _rule(str, default="analysis-v1"),
        "top_fractions": _rule(list, default=[0.0001, 0.001, 0.01, 0.05]),
        "confidence_level": _rule(int, float, default=0.95),
        "bootstrap_repetitions": _rule(int, default=0),
        "source_table_hash": _rule(str, type(None), default=None),
    },
}

_NON_SEMANTIC_PATHS: Final = frozenset(
    {
        "runtime.output_root",
        "runtime.temp_root",
        "runtime.cache_root",
    }
)
_TEMPORARY_IDENTIFIER_TOKENS: Final = (".part", ".partial", ".lock", "/tmp/", "/temp/")


def _matches_type(value: Any, allowed: tuple[type, ...]) -> bool:
    """执行严格 wire 类型判断；特别避免 bool 被当作 int。"""

    if isinstance(value, bool):
        return bool in allowed
    if isinstance(value, int):
        return int in allowed
    if isinstance(value, float):
        return float in allowed
    return any(isinstance(value, expected) for expected in allowed)


def _validate_field(section: str, field: str, value: Any) -> JSONValue:
    rule = _FIELD_RULES[section][field]
    path = f"{section}.{field}"
    if not _matches_type(value, rule.types):
        names = "/".join(expected.__name__ for expected in rule.types)
        raise ConfigContractError(f"{path} 类型错误，期望 {names}")
    if rule.non_empty and isinstance(value, str) and not value:
        raise ConfigContractError(f"{path} 不能为空")
    if rule.choices is not None and value not in rule.choices:
        raise ConfigContractError(
            f"{path}={value!r} 不在允许集合 {sorted(rule.choices, key=str)!r}"
        )
    try:
        canonical_json_bytes(value)
    except ValueError as error:
        raise ConfigContractError(f"{path} 不是严格 JSON 值") from error
    return deepcopy(value)


def _validate_layer(value: Mapping[str, Any], *, layer_name: str) -> dict[str, dict[str, JSONValue]]:
    if not isinstance(value, Mapping):
        raise ConfigContractError(f"{layer_name} 顶层必须是 object")
    extra_sections = set(value) - set(CONFIG_SECTIONS)
    if extra_sections:
        raise ConfigContractError(
            f"{layer_name} 包含未知配置分区：{sorted(extra_sections)}"
        )
    normalized: dict[str, dict[str, JSONValue]] = {}
    for section, raw_section in value.items():
        if not isinstance(raw_section, Mapping):
            raise ConfigContractError(f"{layer_name}.{section} 必须是 object")
        extra_fields = set(raw_section) - set(_FIELD_RULES[section])
        if extra_fields:
            raise ConfigContractError(
                f"{layer_name}.{section} 包含未知字段：{sorted(extra_fields)}"
            )
        normalized[section] = {
            field: _validate_field(section, field, item)
            for field, item in raw_section.items()
        }
    return normalized


def _defaults() -> dict[str, dict[str, JSONValue]]:
    result: dict[str, dict[str, JSONValue]] = {section: {} for section in CONFIG_SECTIONS}
    for section, fields in _FIELD_RULES.items():
        for name, rule in fields.items():
            if rule.default is not _MISSING:
                result[section][name] = deepcopy(rule.default)
    return result


def strict_merge(
    base: Mapping[str, Any],
    *overrides: Mapping[str, Any],
) -> dict[str, dict[str, JSONValue]]:
    """按顺序合并配置层，只允许覆盖已声明的 section/field。

    该函数不补默认值也不执行跨字段检查，便于先构造 base → stage → run 三层；
    最终结果必须交给 :class:`ResolvedConfig`。分区内只做一层字段覆盖，嵌套 JSON
    对象被视为一个不可拆分字段，防止不受 schema 管理的深层键悄然混入。
    """

    normalized = _validate_layer(base, layer_name="base")
    result = deepcopy(normalized)
    for index, override in enumerate(overrides):
        layer = _validate_layer(override, layer_name=f"override[{index}]")
        for section, fields in layer.items():
            result.setdefault(section, {}).update(deepcopy(fields))
    return result


def _validate_positive_integer(value: Any, *, path: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "非负" if allow_zero else "正"
        raise ConfigContractError(f"{path} 必须是{qualifier}整数")
    return value


def _validate_nonnegative_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ConfigContractError(f"{path} 必须是非负有限数")
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise ConfigContractError(f"{path} 必须是非负有限数")
    return float(value)


def _validate_logical_path(value: str, *, path: str) -> str:
    if "\\" in value or ":" in value:
        raise ConfigContractError(f"{path} 必须使用 POSIX 风格逻辑路径")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or str(candidate) != value or not candidate.parts:
        raise ConfigContractError(f"{path} 必须是规范化相对逻辑路径")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ConfigContractError(f"{path} 不能包含路径逃逸")
    return value


def _validate_stable_identifier(value: str | None, *, path: str) -> None:
    if value is None:
        return
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ConfigContractError(f"{path} 必须是无控制字符的稳定标识")
    lowered = value.casefold()
    if "?" in value or any(token in lowered for token in _TEMPORARY_IDENTIFIER_TOKENS):
        raise ConfigContractError(f"{path} 不能引用查询 URL 或临时/锁对象")


def _validate_string_list(values: list[Any], *, path: str, non_empty: bool = False) -> None:
    if non_empty and not values:
        raise ConfigContractError(f"{path} 不能为空数组")
    if not all(isinstance(value, str) and value for value in values):
        raise ConfigContractError(f"{path} 必须是非空字符串数组")
    if len(values) != len(set(values)):
        raise ConfigContractError(f"{path} 不能包含重复项")


def _validate_positive_list(values: list[Any], *, path: str, fractions: bool = False) -> None:
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        or (fractions and value > 1)
        for value in values
    ):
        range_text = "(0,1]" if fractions else "正数"
        raise ConfigContractError(f"{path} 的每一项必须是 {range_text}")
    if len(values) != len(set(values)):
        raise ConfigContractError(f"{path} 不能包含重复项")


def _validate_parameter_groups(values: list[Any]) -> None:
    """冻结 optimizer 参数组映射与静态超参数的严格 v1 wire 结构。"""

    fields = {
        "group_id",
        "parameter_names",
        "learning_rate",
        "weight_decay",
        "momentum",
    }
    group_ids: set[str] = set()
    assigned_parameters: set[str] = set()
    for index, raw_group in enumerate(values):
        path = f"optimizer.parameter_groups[{index}]"
        if not isinstance(raw_group, Mapping):
            raise ConfigContractError(f"{path} 必须是 object")
        if set(raw_group) != fields:
            raise ConfigContractError(
                f"{path} 字段集合不匹配："
                f"missing={sorted(fields-set(raw_group))}, "
                f"extra={sorted(set(raw_group)-fields)}"
            )
        group_id = raw_group["group_id"]
        if not isinstance(group_id, str) or not group_id:
            raise ConfigContractError(f"{path}.group_id 必须是非空字符串")
        _validate_stable_identifier(group_id, path=f"{path}.group_id")
        if group_id in group_ids:
            raise ConfigContractError("optimizer.parameter_groups 的 group_id 不能重复")
        group_ids.add(group_id)

        names = raw_group["parameter_names"]
        if not isinstance(names, list):
            raise ConfigContractError(f"{path}.parameter_names 必须是数组")
        _validate_string_list(names, path=f"{path}.parameter_names", non_empty=True)
        overlap = assigned_parameters.intersection(names)
        if overlap:
            raise ConfigContractError(
                "optimizer.parameter_groups 不允许参数跨组重复："
                f"{sorted(overlap)}"
            )
        assigned_parameters.update(names)

        _validate_nonnegative_number(
            raw_group["learning_rate"], path=f"{path}.learning_rate"
        )
        _validate_nonnegative_number(
            raw_group["weight_decay"], path=f"{path}.weight_decay"
        )
        momentum = _validate_nonnegative_number(
            raw_group["momentum"], path=f"{path}.momentum"
        )
        if momentum >= 1:
            raise ConfigContractError(f"{path}.momentum 必须小于 1")


def _validate_cross_fields(config: dict[str, dict[str, JSONValue]]) -> None:
    identity = config["identity"]
    runtime = config["runtime"]
    batching = config["batching"]
    distributed = config["distributed"]

    stage = identity["stage"]
    if isinstance(stage, bool) or not isinstance(stage, int) or not 0 <= stage <= 9:
        raise ConfigContractError("identity.stage 必须是 0..9 的整数")
    master_seed = identity["master_seed"]
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or not 0 <= master_seed < 2**63:
        raise ConfigContractError("identity.master_seed 必须位于 [0, 2**63)")

    expected_eligible = identity["run_intent"] == "formal"
    if identity["formal_eligible"] is not expected_eligible:
        raise ConfigContractError(
            "identity.formal_eligible 是派生字段，必须等于 (run_intent == 'formal')"
        )
    if identity["run_intent"] == "local_fixture":
        if runtime["offline"] is not True:
            raise ConfigContractError("local_fixture 必须 runtime.offline=true")
        if runtime["device"] != "cpu" or distributed["world_size"] != 1:
            raise ConfigContractError("local_fixture 只允许 CPU、world_size=1")
    elif runtime["allow_dirty_worktree"]:
        raise ConfigContractError("formal 配置禁止 allow_dirty_worktree=true")

    for field in ("output_root", "temp_root", "cache_root"):
        _validate_logical_path(runtime[field], path=f"runtime.{field}")

    for section, fields in {
        "model": ("asset_id", "revision", "tokenizer_asset_id", "initialization_id"),
        "data": ("asset_id", "revision"),
        "importance": ("estimator_decision_ref",),
        "path_integration": ("thresholds_ref",),
        "analysis": ("source_table_hash",),
    }.items():
        for field in fields:
            _validate_stable_identifier(config[section][field], path=f"{section}.{field}")

    per_device = _validate_positive_integer(
        batching["per_device_batch_size"], path="batching.per_device_batch_size"
    )
    global_batch = _validate_positive_integer(
        batching["global_batch_size"], path="batching.global_batch_size"
    )
    microbatch = _validate_positive_integer(
        batching["microbatch_size"], path="batching.microbatch_size"
    )
    accumulation = _validate_positive_integer(
        batching["accumulation_steps"], path="batching.accumulation_steps"
    )
    world_size = _validate_positive_integer(
        distributed["world_size"], path="distributed.world_size"
    )
    if global_batch != per_device * world_size * accumulation:
        raise ConfigContractError(
            "global_batch_size 必须等于 per_device_batch_size * world_size * accumulation_steps"
        )
    if per_device % microbatch != 0:
        raise ConfigContractError("microbatch_size 必须整除 per_device_batch_size")
    if batching["no_sync"] and accumulation <= 1:
        raise ConfigContractError("no_sync=true 只允许 accumulation_steps > 1")
    if distributed["backend"] == "local" and world_size != 1:
        raise ConfigContractError("distributed.backend=local 只允许 world_size=1")
    if distributed["backend"] == "nccl" and runtime["device"] != "cuda":
        raise ConfigContractError("NCCL 只允许 runtime.device=cuda")
    device_ids = distributed["device_ids"]
    if not isinstance(device_ids, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in device_ids
    ):
        raise ConfigContractError("distributed.device_ids 必须是整数数组")
    if len(device_ids) not in ({0, 1} if world_size == 1 else {world_size}):
        raise ConfigContractError("多进程配置必须显式给出与 world_size 等长的 device_ids")
    if len(device_ids) != len(set(device_ids)):
        raise ConfigContractError("distributed.device_ids 不能重复")
    _validate_positive_integer(distributed["timeout_seconds"], path="distributed.timeout_seconds")

    _validate_positive_integer(config["data"]["sequence_length"], path="data.sequence_length")
    _validate_nonnegative_number(config["optimizer"]["learning_rate"], path="optimizer.learning_rate")
    _validate_nonnegative_number(config["optimizer"]["weight_decay"], path="optimizer.weight_decay")
    momentum = _validate_nonnegative_number(config["optimizer"]["momentum"], path="optimizer.momentum")
    if momentum >= 1:
        raise ConfigContractError("optimizer.momentum 必须小于 1")
    if config["optimizer"]["fused"] or config["optimizer"]["foreach"]:
        raise ConfigContractError("共享 correctness 合同不接受 fused/foreach optimizer")
    if not isinstance(config["optimizer"]["parameter_groups"], list):  # pragma: no cover
        raise ConfigContractError("optimizer.parameter_groups 必须是数组")
    _validate_parameter_groups(config["optimizer"]["parameter_groups"])

    _validate_positive_integer(config["logging"]["log_every_steps"], path="logging.log_every_steps")
    _validate_positive_integer(
        config["checkpoint"]["save_every_steps"], path="checkpoint.save_every_steps"
    )
    _validate_positive_integer(
        config["checkpoint"]["max_to_keep"], path="checkpoint.max_to_keep"
    )
    if config["checkpoint"]["two_phase_commit"] is not True:
        raise ConfigContractError("checkpoint.two_phase_commit 必须为 true")

    _validate_string_list(
        config["importance"]["accumulate_views"],
        path="importance.accumulate_views",
        non_empty=True,
    )
    required_views = {"signed", "positive", "negative_mass", "absolute"}
    allowed_views = required_views | {
        "raw",
        "data_movement",
        "net_data_movement",
        "total_endpoint_movement",
        "magnitude",
    }
    configured_views = set(config["importance"]["accumulate_views"])
    if not required_views <= configured_views:
        raise ConfigContractError(
            "importance.accumulate_views 必须包含 signed/positive/negative_mass/absolute"
        )
    unknown_views = configured_views - allowed_views
    if unknown_views:
        raise ConfigContractError(
            f"importance.accumulate_views 包含未知视图：{sorted(unknown_views)}"
        )
    if identity["run_intent"] == "formal" and config["importance"]["require_decision_for_formal"] is not True:
        raise ConfigContractError("formal 配置禁止关闭 estimator decision 检查")

    expected_batches = [32, 64, 128, 256]
    expected_counts = [2, 4, 8, 16, 32]
    expected_preference = [32, 16, 8, 4]
    if config["sampling"]["candidate_batch_sizes"] != expected_batches:
        raise ConfigContractError(
            "sampling.candidate_batch_sizes 在 v1 合同中冻结为 [32,64,128,256]"
        )
    if config["sampling"]["candidate_microbatch_counts"] != expected_counts:
        raise ConfigContractError(
            "sampling.candidate_microbatch_counts 在 v1 合同中冻结为 [2,4,8,16,32]"
        )
    if config["sampling"]["microbatch_preference"] != expected_preference:
        raise ConfigContractError(
            "sampling.microbatch_preference 在 v1 合同中冻结为 [32,16,8,4]"
        )
    for field in ("repetition_count", "reference_batch_size"):
        value = config["sampling"][field]
        if value is not None:
            _validate_positive_integer(value, path=f"sampling.{field}")

    path_config = config["path_integration"]
    if path_config["enabled"]:
        for field in ("default_rule", "probe_count", "node_budget", "thresholds_ref"):
            if path_config[field] is None:
                raise ConfigContractError(
                    f"path_integration.enabled=true 时 {field} 不能为 null"
                )
        _validate_positive_integer(path_config["probe_count"], path="path_integration.probe_count")
        _validate_positive_integer(path_config["node_budget"], path="path_integration.node_budget")

    pruning = config["pruning"]
    _validate_positive_integer(
        pruning["random_repetitions"],
        path="pruning.random_repetitions",
        allow_zero=True,
    )
    _validate_positive_list(pruning["ratios"], path="pruning.ratios", fractions=True)
    if pruning["enabled"] and (pruning["strategy"] == "none" or not pruning["ratios"]):
        raise ConfigContractError("启用 pruning 时必须指定非 none strategy 和 ratios")

    analysis = config["analysis"]
    _validate_positive_list(
        analysis["top_fractions"], path="analysis.top_fractions", fractions=True
    )
    confidence = analysis["confidence_level"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ConfigContractError("analysis.confidence_level 必须位于 (0,1)")
    _validate_positive_integer(
        analysis["bootstrap_repetitions"],
        path="analysis.bootstrap_repetitions",
        allow_zero=True,
    )
    if analysis["source_table_hash"] is not None:
        digest = analysis["source_table_hash"]
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ConfigContractError("analysis.source_table_hash 必须是小写 SHA-256")


@dataclass(frozen=True, slots=True)
class ConfigDifference:
    """两个 resolved config 在一个叶子字段上的差异。"""

    path: str
    left: JSONValue
    right: JSONValue
    semantic: bool

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "path": self.path,
            "left": self.left,
            "right": self.right,
            "semantic": self.semantic,
        }


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """已展开默认值并通过全部合同检查的不可变配置视图。"""

    _value: dict[str, dict[str, JSONValue]]

    def __post_init__(self) -> None:
        # 禁止直接构造绕开 ``resolve``；重新验证后保存独立副本。
        normalized = _validate_layer(self._value, layer_name="resolved")
        missing_sections = set(CONFIG_SECTIONS) - set(normalized)
        if missing_sections:
            raise ConfigContractError(
                f"resolved config 缺少分区：{sorted(missing_sections)}"
            )
        for section, rules in _FIELD_RULES.items():
            missing_fields = set(rules) - set(normalized[section])
            if missing_fields:
                raise ConfigContractError(
                    f"resolved config 的 {section} 缺少字段：{sorted(missing_fields)}"
                )
        _validate_cross_fields(normalized)
        object.__setattr__(self, "_value", deepcopy(normalized))

    @classmethod
    def resolve(cls, *layers: Mapping[str, Any]) -> "ResolvedConfig":
        """按顺序应用配置层、展开默认值并生成 resolved config。

        至少需要一个输入层。``formal_eligible`` 由 ``run_intent`` 计算；调用者可以
        省略它，但不能提供相反值。最终正式资格还必须经过 ``readiness`` 模块检查
        freeze、decision 和 Gate artifact。
        """

        if not layers:
            raise ConfigContractError("ResolvedConfig.resolve 至少需要一个配置层")
        merged = _defaults()
        explicitly_declared_eligibility: bool | None = None
        for index, layer in enumerate(layers):
            normalized = _validate_layer(layer, layer_name=f"layer[{index}]")
            if "identity" in normalized and "formal_eligible" in normalized["identity"]:
                explicitly_declared_eligibility = normalized["identity"]["formal_eligible"]
            for section, fields in normalized.items():
                merged[section].update(deepcopy(fields))

        missing: list[str] = []
        for section, rules in _FIELD_RULES.items():
            for field, rule in rules.items():
                if rule.default is _MISSING and field not in merged[section]:
                    missing.append(f"{section}.{field}")
        if missing:
            raise ConfigContractError(f"配置缺少必需字段：{sorted(missing)}")
        expected = merged["identity"].get("run_intent") == "formal"
        if explicitly_declared_eligibility is not None and explicitly_declared_eligibility is not expected:
            raise ConfigContractError(
                "identity.formal_eligible 不能由调用者伪造；它必须与 run_intent 一致"
            )
        merged["identity"]["formal_eligible"] = expected
        return cls(merged)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ResolvedConfig":
        return cls.resolve(value)

    def to_dict(self) -> dict[str, dict[str, JSONValue]]:
        """返回深拷贝，避免调用者修改内部 resolved 状态。"""

        return deepcopy(self._value)

    def section(self, name: str) -> dict[str, JSONValue]:
        """读取一个分区的独立副本。"""

        if name not in CONFIG_SECTIONS:
            raise KeyError(name)
        return deepcopy(self._value[name])

    @property
    def full_hash(self) -> str:
        return canonical_json_hash(self._value)

    @property
    def config_hash(self) -> str:
        semantic = self.to_dict()
        for path in _NON_SEMANTIC_PATHS:
            section, field = path.split(".", 1)
            semantic[section].pop(field, None)
        return canonical_json_hash(semantic)

    @property
    def formal_eligible(self) -> bool:
        """仅返回配置意图资格；外部证据资格由 readiness 计算。"""

        return bool(self._value["identity"]["formal_eligible"])

    def diff(self, other: "ResolvedConfig") -> tuple[ConfigDifference, ...]:
        """按稳定字段路径返回差异，并标记是否影响语义摘要。"""

        if not isinstance(other, ResolvedConfig):
            raise TypeError("other 必须是 ResolvedConfig")
        differences: list[ConfigDifference] = []
        for section in CONFIG_SECTIONS:
            for field in sorted(_FIELD_RULES[section]):
                left = self._value[section][field]
                right = other._value[section][field]
                if left != right:
                    path = f"{section}.{field}"
                    differences.append(
                        ConfigDifference(
                            path=path,
                            left=deepcopy(left),
                            right=deepcopy(right),
                            semantic=path not in _NON_SEMANTIC_PATHS,
                        )
                    )
        return tuple(differences)


def diff_configs(
    left: ResolvedConfig,
    right: ResolvedConfig,
) -> tuple[ConfigDifference, ...]:
    """函数式配置差异入口，便于 CLI 与报告模块调用。"""

    return left.diff(right)
