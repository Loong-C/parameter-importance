"""任务可执行性所需的独立 ``ResolvedConfig v2``。

v1 配置冻结了 Stage 0--9 共用的科学语义，但没有训练时长、scheduler、DataLoader
游标边界、任务恢复策略或成功产物集合。直接向 v1 增键会破坏已有 hash 和本机
fixture，因此 v2 采用一个独立、严格的执行信封：内部完整嵌入并验证 v1
``base_config``，外层增加 task、execution、training、scheduler、data loader、离线
provider、评价/性能测量、分段 checkpoint、AMP/scaler、optimizer 运行参数、launcher、
跨任务 orchestration、recovery 与 artifacts 合同。

``load_resolved_config_compatible`` 是唯一兼容入口：v2 wire object 会重新计算双摘要；
legacy v1 必须能够唯一确定 canonical task ID，或由调用者显式给出 task ID。兼容加载
不会猜测训练步数、恢复点或 formal 证据，也不会修改 legacy v1 的语义摘要。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import PurePosixPath
from typing import Any, Final

from .config import CONFIG_SCHEMA_VERSION, ResolvedConfig
from .errors import ConfigContractError
from .jsonio import JSONValue, canonical_json_bytes, canonical_json_hash
from .task_catalog import (
    DEFAULT_TASK_CATALOG,
    RunnerKind,
    TaskCatalog,
    TaskCatalogError,
    TaskDefinition,
)


CONFIG_V2_SCHEMA_VERSION: Final = "resolved-config-v2"

_TOP_LEVEL_FIELDS: Final = {
    "schema_version",
    "task_id",
    "base_config",
    "execution",
    "training",
    "scheduler",
    "data_loader",
    "providers",
    "evaluation",
    "profiling",
    "checkpoint_schedule",
    "precision_runtime",
    "optimizer_runtime",
    "launcher",
    "orchestration",
    "recovery",
    "artifacts",
}
_WIRE_FIELDS: Final = _TOP_LEVEL_FIELDS | {"config_hash", "full_hash"}
_TRAINING_RUNNERS: Final = {
    RunnerKind.TRAINING,
    RunnerKind.DISTRIBUTED_TRAINING,
    RunnerKind.ROUTE_TRAINING,
}
_MODEL_EXECUTION_RUNNERS: Final = {
    RunnerKind.TRAINING,
    RunnerKind.DISTRIBUTED_TRAINING,
    RunnerKind.ROUTE_TRAINING,
    RunnerKind.REGISTRY,
    RunnerKind.ORACLE,
    RunnerKind.VALIDATION,
    RunnerKind.REFERENCE,
    RunnerKind.ESTIMATOR_EXPERIMENT,
    RunnerKind.PILOT,
    RunnerKind.PATH_INTEGRATION,
    RunnerKind.CAPACITY,
    RunnerKind.PRUNING,
    RunnerKind.ABLATION,
}
_ROUTE_SPEC_TASKS: Final = {
    "stage4.minimal_complete_loop",
    "stage4.pretrain",
    "stage4.direct_supervised",
    "stage4.finetune",
    "stage5.formal_pretraining",
    "stage5.pretrain",
    "stage6.training_route_comparison",
}
_MATRIX_INPUT_TASKS: Final = {
    "stage6.evaluate", "stage6.compare", "stage6.importance_reuse", "stage6.report",
    "stage7.evaluate", "stage7.reduce", "stage7.report",
    "stage8.execute", "stage8.reduce", "stage8.recommend", "stage8.report",
}
_PAIRED_DESIGN_TASKS: Final = {
    "stage2.05_paired_estimator_runner",
    "stage2.07_main_sweep",
    "stage6.evaluate",
    "stage6.compare",
    "stage8.execute",
}
_EVALUATION_TASKS: Final = {
    "stage4.pruning_validation",
    "stage6.evaluate",
    "stage7.evaluate",
}
_PROFILING_TASKS: Final = {
    "stage0.10_capacity_and_operations",
    "stage2.09_cost_and_system_validation",
}

_EXECUTION_FIELDS: Final = {
    "runner_kind",
    "timeout_seconds",
    "max_attempts",
    "dry_run",
    "fail_on_blocked",
}
_TRAINING_FIELDS: Final = {
    "max_steps",
    "max_epochs",
    "validation_every_steps",
    "gradient_clip_max_norm",
    "deterministic_algorithms",
}
_SCHEDULER_FIELDS: Final = {"kind", "warmup_steps", "total_steps"}
_DATA_LOADER_FIELDS: Final = {
    "num_workers",
    "prefetch_factor",
    "persistent_workers",
    "drop_last",
    "cursor_policy",
}
_PROVIDER_FIELDS: Final = {
    "kind",
    "model_manifest_ref",
    "model_root_ref",
    "data_manifest_ref",
    "data_root_ref",
    "tokenizer_manifest_ref",
    "tokenizer_root_ref",
    "task_type",
    "task_name",
    "num_labels",
    "local_files_only",
    "trust_remote_code",
}
_EVALUATION_FIELDS: Final = {
    "enabled",
    "split",
    "every_steps",
    "batch_size",
    "max_batches",
    "metrics",
    "save_predictions",
}
_PROFILING_FIELDS: Final = {
    "enabled",
    "warmup_steps",
    "measure_steps",
    "repetitions",
    "capture_memory",
    "capture_throughput",
    "capture_communication",
    "synchronize_device",
}
_CHECKPOINT_SCHEDULE_FIELDS: Final = {
    "segments",
    "save_on_phase_end",
    "save_optimizer",
    "save_rng",
    "save_data_state",
}
_CHECKPOINT_SEGMENT_FIELDS: Final = {"start_step", "end_step", "every_steps"}
_PRECISION_RUNTIME_FIELDS: Final = {
    "autocast_enabled",
    "autocast_dtype",
    "grad_scaler_enabled",
    "initial_scale",
    "growth_factor",
    "backoff_factor",
    "growth_interval",
    "global_found_inf_reduce",
}
_OPTIMIZER_RUNTIME_FIELDS: Final = {
    "betas",
    "eps",
    "amsgrad",
    "dampening",
    "nesterov",
    "maximize",
    "capturable",
    "differentiable",
}
_LAUNCHER_FIELDS: Final = {
    "kind",
    "backend",
    "world_size",
    "init_method",
    "init_ref",
    "rendezvous_id",
    "max_restarts",
}
_ORCHESTRATION_FIELDS: Final = {
    "route_spec_ref",
    "quadrature_decision_ref",
    "matrix_ref",
    "paired_design",
    "input_result_refs",
}
_PAIRED_DESIGN_FIELDS: Final = {"enabled", "design", "mapping_ref", "budget_unit"}
_RECOVERY_FIELDS: Final = {
    "mode",
    "resume_ref",
    "max_restarts",
    "safe_boundary",
}
_ARTIFACT_FIELDS: Final = {
    "output_dir",
    "required_kinds",
    "publish_partial",
}

_OVERRIDABLE_SECTIONS: Final = {
    "execution": _EXECUTION_FIELDS,
    "training": _TRAINING_FIELDS,
    "scheduler": _SCHEDULER_FIELDS,
    "data_loader": _DATA_LOADER_FIELDS,
    "providers": _PROVIDER_FIELDS,
    "evaluation": _EVALUATION_FIELDS,
    "profiling": _PROFILING_FIELDS,
    "checkpoint_schedule": _CHECKPOINT_SCHEDULE_FIELDS,
    "precision_runtime": _PRECISION_RUNTIME_FIELDS,
    "optimizer_runtime": _OPTIMIZER_RUNTIME_FIELDS,
    "launcher": _LAUNCHER_FIELDS,
    "orchestration": _ORCHESTRATION_FIELDS,
    "recovery": _RECOVERY_FIELDS,
    "artifacts": _ARTIFACT_FIELDS,
}


def _expect_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigContractError(f"{field} 必须是 object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigContractError(f"{field} 的键必须是字符串")
    return value


def _expect_fields(value: Mapping[str, Any], expected: set[str] | frozenset[str], *, field: str) -> None:
    actual = set(value)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        raise ConfigContractError(f"{field} 字段集合无效：missing={missing}, extra={extra}")


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigContractError(f"{field} 必须是 bool")
    return value


def _strict_int(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    optional: bool = False,
) -> int | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "或 null" if optional else ""
        raise ConfigContractError(f"{field} 必须是 >= {minimum} 的整数{qualifier}")
    return value


def _positive_number(value: object, *, field: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigContractError(f"{field} 必须是正有限数" + ("或 null" if optional else ""))
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise ConfigContractError(f"{field} 必须是正有限数")
    return normalized


def _finite_number(
    value: object,
    *,
    field: str,
    minimum: float | None = None,
    maximum: float | None = None,
    minimum_exclusive: bool = False,
    maximum_exclusive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigContractError(f"{field} 必须是有限数")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ConfigContractError(f"{field} 必须是有限数")
    if minimum is not None:
        invalid = normalized <= minimum if minimum_exclusive else normalized < minimum
        if invalid:
            operator = ">" if minimum_exclusive else ">="
            raise ConfigContractError(f"{field} 必须 {operator} {minimum}")
    if maximum is not None:
        invalid = normalized >= maximum if maximum_exclusive else normalized > maximum
        if invalid:
            operator = "<" if maximum_exclusive else "<="
            raise ConfigContractError(f"{field} 必须 {operator} {maximum}")
    return normalized


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigContractError(f"{field} 必须是非空字符串")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _non_empty_string(value, field=field)


def _logical_path(value: object, *, field: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    text = _non_empty_string(value, field=field)
    if "\\" in text:
        raise ConfigContractError(f"{field} 必须使用 POSIX 风格逻辑路径")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or str(path) != text:
        raise ConfigContractError(f"{field} 发生路径逃逸或不是规范相对路径")
    return str(path)


def _normalize_string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ConfigContractError(f"{field} 必须是字符串数组")
    normalized = [_non_empty_string(item, field=f"{field}[{index}]") for index, item in enumerate(value)]
    if len(set(normalized)) != len(normalized):
        raise ConfigContractError(f"{field} 不能包含重复值")
    return normalized


def _lookup_path(value: Mapping[str, Any], dotted_path: str) -> object:
    current: object = value
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ConfigContractError(f"任务要求的配置路径不存在：{dotted_path}")
        current = current[component]
    return current


def _defaults(
    task: TaskDefinition,
    base_config: ResolvedConfig,
) -> dict[str, dict[str, JSONValue]]:
    """生成不依赖机器或实验结果的 v2 默认值。

    训练长度刻意保持 ``null``；训练 runner 必须由版本化配置显式指定，而不能由兼容
    loader 猜一个“能跑起来”的步数。
    """

    precision = base_config.section("precision")
    optimizer = base_config.section("optimizer")
    distributed = base_config.section("distributed")
    loss = base_config.section("loss")
    checkpoint = base_config.section("checkpoint")
    amp_enabled = bool(precision["amp"])
    compute_dtype = str(precision["compute_dtype"])
    optimizer_type = str(optimizer["type"])
    world_size = int(distributed["world_size"])
    backend = str(distributed["backend"])
    launcher_kind = "local" if world_size == 1 and backend == "local" else "torchrun"

    return {
        "execution": {
            "runner_kind": task.runner_kind.value,
            "timeout_seconds": 600,
            "max_attempts": 1,
            "dry_run": False,
            "fail_on_blocked": False,
        },
        "training": {
            "max_steps": None,
            "max_epochs": None,
            "validation_every_steps": None,
            "gradient_clip_max_norm": None,
            "deterministic_algorithms": True,
        },
        "scheduler": {
            "kind": "none",
            "warmup_steps": 0,
            "total_steps": None,
        },
        "data_loader": {
            "num_workers": 0,
            "prefetch_factor": None,
            "persistent_workers": False,
            "drop_last": False,
            "cursor_policy": "attempt_commit_state",
        },
        "providers": {
            "kind": "tiny",
            "model_manifest_ref": None,
            "model_root_ref": None,
            "data_manifest_ref": None,
            "data_root_ref": None,
            "tokenizer_manifest_ref": None,
            "tokenizer_root_ref": None,
            "task_type": loss["task_type"],
            "task_name": "fixture",
            "num_labels": None,
            "local_files_only": True,
            "trust_remote_code": False,
        },
        "evaluation": {
            "enabled": False,
            "split": None,
            "every_steps": None,
            "batch_size": None,
            "max_batches": None,
            "metrics": [],
            "save_predictions": False,
        },
        "profiling": {
            "enabled": False,
            "warmup_steps": 0,
            "measure_steps": None,
            "repetitions": 1,
            "capture_memory": False,
            "capture_throughput": False,
            "capture_communication": False,
            "synchronize_device": False,
        },
        "checkpoint_schedule": {
            "segments": [
                {
                    "start_step": 0,
                    "end_step": None,
                    "every_steps": checkpoint["save_every_steps"],
                }
            ],
            "save_on_phase_end": True,
            "save_optimizer": True,
            "save_rng": True,
            "save_data_state": True,
        },
        "precision_runtime": {
            "autocast_enabled": amp_enabled,
            "autocast_dtype": compute_dtype if amp_enabled else "float32",
            "grad_scaler_enabled": amp_enabled and compute_dtype == "float16",
            "initial_scale": 65536.0,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
            "global_found_inf_reduce": True,
        },
        "optimizer_runtime": {
            "betas": [0.9, 0.999] if optimizer_type == "adamw" else None,
            "eps": 1e-8 if optimizer_type == "adamw" else None,
            "amsgrad": False,
            "dampening": 0.0,
            "nesterov": False,
            "maximize": False,
            "capturable": False,
            "differentiable": False,
        },
        "launcher": {
            "kind": launcher_kind,
            "backend": backend,
            "world_size": world_size,
            "init_method": "local" if launcher_kind == "local" else "env",
            "init_ref": None,
            "rendezvous_id": None if launcher_kind == "local" else task.task_id,
            "max_restarts": 0,
        },
        "orchestration": {
            "route_spec_ref": None,
            "quadrature_decision_ref": None,
            "matrix_ref": None,
            "paired_design": {
                "enabled": False,
                "design": "none",
                "mapping_ref": None,
                "budget_unit": "samples",
            },
            "input_result_refs": [],
        },
        "recovery": {
            "mode": task.recovery_mode.value,
            "resume_ref": None,
            "max_restarts": 0,
            "safe_boundary": task.safe_boundary.value,
        },
        "artifacts": {
            "output_dir": f"runs/{task.task_id}",
            "required_kinds": list(task.artifact_kinds),
            "publish_partial": False,
        },
    }


def _apply_overrides(
    base: dict[str, dict[str, JSONValue]],
    overrides: Mapping[str, Any] | None,
) -> dict[str, dict[str, JSONValue]]:
    result = deepcopy(base)
    if overrides is None:
        return result
    unknown_sections = set(overrides) - set(_OVERRIDABLE_SECTIONS)
    if unknown_sections:
        raise ConfigContractError(f"v2 overrides 包含未知或不可覆盖分区：{sorted(unknown_sections)}")
    for section, raw_fields in overrides.items():
        fields = _expect_mapping(raw_fields, field=f"overrides.{section}")
        extra = set(fields) - set(_OVERRIDABLE_SECTIONS[section])
        if extra:
            raise ConfigContractError(f"overrides.{section} 包含未知字段：{sorted(extra)}")
        for field, item in fields.items():
            # canonical 编码在此拒绝 tuple、非有限数和任意对象，避免后续 hash 才失败。
            try:
                canonical_json_bytes(item)
            except ValueError as error:
                raise ConfigContractError(f"overrides.{section}.{field} 不是严格 JSON 值") from error
            result[section][field] = deepcopy(item)
    return result


def _normalize_payload(
    payload: Mapping[str, Any],
    *,
    catalog: TaskCatalog,
) -> tuple[dict[str, JSONValue], ResolvedConfig, TaskDefinition]:
    _expect_fields(payload, _TOP_LEVEL_FIELDS, field="ResolvedConfigV2")
    if payload["schema_version"] != CONFIG_V2_SCHEMA_VERSION:
        raise ConfigContractError("ResolvedConfigV2 schema_version 不受支持")
    task_id = _non_empty_string(payload["task_id"], field="task_id")
    try:
        task = catalog.get(task_id)
    except TaskCatalogError as error:
        raise ConfigContractError(str(error)) from error

    base_mapping = _expect_mapping(payload["base_config"], field="base_config")
    base_config = ResolvedConfig.from_mapping(base_mapping)
    base_identity = base_config.section("identity")
    if base_identity["stage"] != task.stage:
        raise ConfigContractError(
            f"base_config.identity.stage={base_identity['stage']} 与 {task_id} 不一致"
        )

    execution = _expect_mapping(payload["execution"], field="execution")
    _expect_fields(execution, _EXECUTION_FIELDS, field="execution")
    runner_kind = _non_empty_string(execution["runner_kind"], field="execution.runner_kind")
    if runner_kind != task.runner_kind.value:
        raise ConfigContractError(
            f"execution.runner_kind={runner_kind!r} 与任务目录 {task.runner_kind.value!r} 不一致"
        )
    timeout_seconds = _strict_int(execution["timeout_seconds"], field="execution.timeout_seconds", minimum=1)
    max_attempts = _strict_int(execution["max_attempts"], field="execution.max_attempts", minimum=1)
    dry_run = _strict_bool(execution["dry_run"], field="execution.dry_run")
    fail_on_blocked = _strict_bool(execution["fail_on_blocked"], field="execution.fail_on_blocked")

    training = _expect_mapping(payload["training"], field="training")
    _expect_fields(training, _TRAINING_FIELDS, field="training")
    max_steps = _strict_int(training["max_steps"], field="training.max_steps", minimum=1, optional=True)
    max_epochs = _strict_int(training["max_epochs"], field="training.max_epochs", minimum=1, optional=True)
    validation_every = _strict_int(
        training["validation_every_steps"],
        field="training.validation_every_steps",
        minimum=1,
        optional=True,
    )
    clip_norm = _positive_number(
        training["gradient_clip_max_norm"],
        field="training.gradient_clip_max_norm",
        optional=True,
    )
    deterministic = _strict_bool(
        training["deterministic_algorithms"],
        field="training.deterministic_algorithms",
    )
    if task.runner_kind in _TRAINING_RUNNERS and max_steps is None and max_epochs is None:
        raise ConfigContractError(
            f"{task.task_id} 是训练任务，必须显式设置 training.max_steps 或 max_epochs"
        )
    if validation_every is not None and max_steps is not None and validation_every > max_steps:
        raise ConfigContractError("validation_every_steps 不能大于 max_steps")

    scheduler = _expect_mapping(payload["scheduler"], field="scheduler")
    _expect_fields(scheduler, _SCHEDULER_FIELDS, field="scheduler")
    scheduler_kind = _non_empty_string(scheduler["kind"], field="scheduler.kind")
    if scheduler_kind not in {"none", "constant", "linear", "cosine"}:
        raise ConfigContractError("scheduler.kind 不受支持")
    warmup_steps = _strict_int(scheduler["warmup_steps"], field="scheduler.warmup_steps", minimum=0)
    total_steps = _strict_int(scheduler["total_steps"], field="scheduler.total_steps", minimum=1, optional=True)
    if scheduler_kind == "none" and (warmup_steps != 0 or total_steps is not None):
        raise ConfigContractError("scheduler.kind=none 时 warmup_steps 必须为 0 且 total_steps 为 null")
    if scheduler_kind != "none":
        if total_steps is None:
            raise ConfigContractError("启用 scheduler 时必须设置 scheduler.total_steps")
        if warmup_steps >= total_steps:
            raise ConfigContractError("scheduler.warmup_steps 必须小于 total_steps")
        if max_steps is not None and total_steps != max_steps:
            raise ConfigContractError("scheduler.total_steps 必须与 training.max_steps 一致")

    data_loader = _expect_mapping(payload["data_loader"], field="data_loader")
    _expect_fields(data_loader, _DATA_LOADER_FIELDS, field="data_loader")
    num_workers = _strict_int(data_loader["num_workers"], field="data_loader.num_workers", minimum=0)
    prefetch_factor = _strict_int(
        data_loader["prefetch_factor"],
        field="data_loader.prefetch_factor",
        minimum=1,
        optional=True,
    )
    persistent_workers = _strict_bool(
        data_loader["persistent_workers"], field="data_loader.persistent_workers"
    )
    drop_last = _strict_bool(data_loader["drop_last"], field="data_loader.drop_last")
    cursor_policy = _non_empty_string(data_loader["cursor_policy"], field="data_loader.cursor_policy")
    if cursor_policy not in {"attempt_commit_state", "draw_manifest", "checkpoint_commit"}:
        raise ConfigContractError("data_loader.cursor_policy 不受支持")
    if num_workers == 0 and (prefetch_factor is not None or persistent_workers):
        raise ConfigContractError("num_workers=0 时不得启用 prefetch_factor/persistent_workers")
    if num_workers > 0 and prefetch_factor is None:
        raise ConfigContractError("num_workers>0 时必须显式设置 prefetch_factor")

    providers = _expect_mapping(payload["providers"], field="providers")
    _expect_fields(providers, _PROVIDER_FIELDS, field="providers")
    provider_kind = _non_empty_string(providers["kind"], field="providers.kind")
    if provider_kind not in {"tiny", "offline_hf"}:
        raise ConfigContractError("providers.kind 只能是 tiny 或 offline_hf")
    provider_refs = {
        name: _logical_path(providers[name], field=f"providers.{name}", optional=True)
        for name in (
            "model_manifest_ref", "model_root_ref", "data_manifest_ref", "data_root_ref",
            "tokenizer_manifest_ref", "tokenizer_root_ref",
        )
    }
    provider_task_type = _non_empty_string(providers["task_type"], field="providers.task_type")
    if provider_task_type not in {"causal_lm", "sequence_classification", "synthetic"}:
        raise ConfigContractError("providers.task_type 不受支持")
    if provider_task_type != base_config.section("loss")["task_type"]:
        raise ConfigContractError("providers.task_type 必须与 base_config.loss.task_type 一致")
    provider_task_name = _non_empty_string(
        providers["task_name"], field="providers.task_name"
    ).casefold().replace("-", "")
    if provider_task_name not in {"fixture", "pile", "sst2", "mnli", "rte"}:
        raise ConfigContractError("providers.task_name 只能是 fixture/pile/sst2/mnli/rte")
    if provider_task_name == "pile" and provider_task_type != "causal_lm":
        raise ConfigContractError("Pile provider 必须使用 causal_lm task_type")
    if provider_task_name in {"sst2", "mnli", "rte"} and provider_task_type != "sequence_classification":
        raise ConfigContractError("GLUE provider 必须使用 sequence_classification task_type")
    num_labels = _strict_int(providers["num_labels"], field="providers.num_labels", minimum=2, optional=True)
    if provider_task_type == "sequence_classification" and num_labels is None:
        raise ConfigContractError("sequence_classification provider 必须设置 num_labels>=2")
    if provider_task_type != "sequence_classification" and num_labels is not None:
        raise ConfigContractError("非分类 provider 的 num_labels 必须为 null")
    expected_labels = {"sst2": 2, "mnli": 3, "rte": 2}.get(provider_task_name)
    if expected_labels is not None and num_labels != expected_labels:
        raise ConfigContractError(
            f"providers.task_name={provider_task_name} 要求 num_labels={expected_labels}"
        )
    local_files_only = _strict_bool(providers["local_files_only"], field="providers.local_files_only")
    trust_remote_code = _strict_bool(providers["trust_remote_code"], field="providers.trust_remote_code")
    if not local_files_only or trust_remote_code:
        raise ConfigContractError("provider 必须 local_files_only=true 且 trust_remote_code=false")
    if provider_kind == "tiny" and any(reference is not None for reference in provider_refs.values()):
        raise ConfigContractError("tiny provider 不得伪装绑定外部 manifest/root")
    if provider_kind == "offline_hf" and any(reference is None for reference in provider_refs.values()):
        raise ConfigContractError("offline_hf provider 必须显式绑定模型、数据和 tokenizer 的 manifest/root")
    formal_model_execution = (
        base_identity["run_intent"] == "formal" and task.runner_kind in _MODEL_EXECUTION_RUNNERS
    )
    if formal_model_execution and provider_kind != "offline_hf":
        raise ConfigContractError("formal 模型执行任务禁止 tiny provider，必须使用本地 offline_hf 资产")

    evaluation = _expect_mapping(payload["evaluation"], field="evaluation")
    _expect_fields(evaluation, _EVALUATION_FIELDS, field="evaluation")
    evaluation_enabled = _strict_bool(evaluation["enabled"], field="evaluation.enabled")
    evaluation_split = _optional_string(evaluation["split"], field="evaluation.split")
    evaluation_every = _strict_int(
        evaluation["every_steps"], field="evaluation.every_steps", minimum=1, optional=True
    )
    evaluation_batch_size = _strict_int(
        evaluation["batch_size"], field="evaluation.batch_size", minimum=1, optional=True
    )
    evaluation_max_batches = _strict_int(
        evaluation["max_batches"], field="evaluation.max_batches", minimum=1, optional=True
    )
    evaluation_metrics = _normalize_string_list(evaluation["metrics"], field="evaluation.metrics")
    supported_metrics = {
        "loss", "perplexity", "accuracy", "f1", "matthews_correlation",
        "mnli_matched_accuracy", "mnli_mismatched_accuracy", "damage_auc",
    }
    unknown_metrics = set(evaluation_metrics) - supported_metrics
    if unknown_metrics:
        raise ConfigContractError(f"evaluation.metrics 包含未知指标：{sorted(unknown_metrics)}")
    save_predictions = _strict_bool(
        evaluation["save_predictions"], field="evaluation.save_predictions"
    )
    if evaluation_enabled:
        if evaluation_split is None or evaluation_batch_size is None or not evaluation_metrics:
            raise ConfigContractError("启用 evaluation 时必须设置 split、batch_size 和 metrics")
        if provider_task_type == "causal_lm" and not set(evaluation_metrics) <= {"loss", "perplexity"}:
            raise ConfigContractError("causal_lm evaluation 只能使用 loss/perplexity")
    elif any(
        value is not None
        for value in (evaluation_split, evaluation_every, evaluation_batch_size, evaluation_max_batches)
    ) or evaluation_metrics or save_predictions:
        raise ConfigContractError("evaluation.enabled=false 时其他 evaluation 字段必须为空")
    formal_requires_evaluation = (
        base_identity["run_intent"] == "formal"
        and (
            task.task_id in _EVALUATION_TASKS
            or (task.stage >= 4 and task.runner_kind in _TRAINING_RUNNERS)
        )
    )
    if formal_requires_evaluation and not evaluation_enabled:
        raise ConfigContractError(f"formal {task.task_id} 必须启用 evaluation")

    profiling = _expect_mapping(payload["profiling"], field="profiling")
    _expect_fields(profiling, _PROFILING_FIELDS, field="profiling")
    profiling_enabled = _strict_bool(profiling["enabled"], field="profiling.enabled")
    profiling_warmup = _strict_int(profiling["warmup_steps"], field="profiling.warmup_steps", minimum=0)
    profiling_measure = _strict_int(
        profiling["measure_steps"], field="profiling.measure_steps", minimum=1, optional=True
    )
    profiling_repetitions = _strict_int(
        profiling["repetitions"], field="profiling.repetitions", minimum=1
    )
    capture_memory = _strict_bool(profiling["capture_memory"], field="profiling.capture_memory")
    capture_throughput = _strict_bool(
        profiling["capture_throughput"], field="profiling.capture_throughput"
    )
    capture_communication = _strict_bool(
        profiling["capture_communication"], field="profiling.capture_communication"
    )
    synchronize_device = _strict_bool(
        profiling["synchronize_device"], field="profiling.synchronize_device"
    )
    if profiling_enabled:
        if profiling_measure is None or not (capture_memory or capture_throughput or capture_communication):
            raise ConfigContractError("启用 profiling 时必须设置 measure_steps 和至少一个测量项")
        if task.runner_kind in _TRAINING_RUNNERS:
            if max_steps is None:
                raise ConfigContractError(
                    "训练 profiling 需要显式 training.max_steps，不能只按 epoch 推断测量窗口"
                )
            required_profile_steps = (
                profiling_warmup + profiling_measure * profiling_repetitions
            )
            if required_profile_steps > max_steps:
                raise ConfigContractError(
                    "profiling warmup + measure_steps * repetitions 不能超过 training.max_steps"
                )
    elif profiling_measure is not None or any(
        (capture_memory, capture_throughput, capture_communication, synchronize_device)
    ):
        raise ConfigContractError("profiling.enabled=false 时不得启用测量字段")
    if base_identity["run_intent"] == "formal" and task.task_id in _PROFILING_TASKS and not profiling_enabled:
        raise ConfigContractError(f"formal {task.task_id} 必须启用 profiling")

    checkpoint_schedule = _expect_mapping(
        payload["checkpoint_schedule"], field="checkpoint_schedule"
    )
    _expect_fields(
        checkpoint_schedule,
        _CHECKPOINT_SCHEDULE_FIELDS,
        field="checkpoint_schedule",
    )
    raw_segments = checkpoint_schedule["segments"]
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ConfigContractError("checkpoint_schedule.segments 必须是非空数组")
    segments: list[JSONValue] = []
    previous_end: int | None = None
    for index, raw_segment in enumerate(raw_segments):
        segment = _expect_mapping(raw_segment, field=f"checkpoint_schedule.segments[{index}]")
        _expect_fields(segment, _CHECKPOINT_SEGMENT_FIELDS, field=f"checkpoint_schedule.segments[{index}]")
        start_step = _strict_int(
            segment["start_step"], field=f"checkpoint_schedule.segments[{index}].start_step", minimum=0
        )
        end_step = _strict_int(
            segment["end_step"], field=f"checkpoint_schedule.segments[{index}].end_step", minimum=1, optional=True
        )
        every_steps = _strict_int(
            segment["every_steps"], field=f"checkpoint_schedule.segments[{index}].every_steps", minimum=1
        )
        if index == 0 and start_step != 0:
            raise ConfigContractError("checkpoint_schedule 第一段必须从 step 0 开始")
        if previous_end is not None and start_step != previous_end:
            raise ConfigContractError("checkpoint_schedule 分段必须连续且不重叠")
        if previous_end is None and index > 0:
            raise ConfigContractError("只有最后一个 checkpoint segment 的 end_step 可以为 null")
        if end_step is not None and end_step <= start_step:
            raise ConfigContractError("checkpoint segment.end_step 必须大于 start_step")
        if end_step is None and index != len(raw_segments) - 1:
            raise ConfigContractError("只有最后一个 checkpoint segment 可开放结束")
        segments.append({"start_step": start_step, "end_step": end_step, "every_steps": every_steps})
        previous_end = end_step
    if max_steps is not None and previous_end is not None and previous_end < max_steps:
        raise ConfigContractError("checkpoint_schedule 未覆盖全部 training.max_steps")
    save_on_phase_end = _strict_bool(
        checkpoint_schedule["save_on_phase_end"], field="checkpoint_schedule.save_on_phase_end"
    )
    save_optimizer = _strict_bool(
        checkpoint_schedule["save_optimizer"], field="checkpoint_schedule.save_optimizer"
    )
    save_rng = _strict_bool(checkpoint_schedule["save_rng"], field="checkpoint_schedule.save_rng")
    save_data_state = _strict_bool(
        checkpoint_schedule["save_data_state"], field="checkpoint_schedule.save_data_state"
    )
    if task.runner_kind in _TRAINING_RUNNERS and not (
        save_on_phase_end and save_optimizer and save_rng and save_data_state
    ):
        raise ConfigContractError("训练任务必须保存 phase end、optimizer、RNG 和 data state")

    precision_runtime = _expect_mapping(
        payload["precision_runtime"], field="precision_runtime"
    )
    _expect_fields(precision_runtime, _PRECISION_RUNTIME_FIELDS, field="precision_runtime")
    autocast_enabled = _strict_bool(
        precision_runtime["autocast_enabled"], field="precision_runtime.autocast_enabled"
    )
    autocast_dtype = _non_empty_string(
        precision_runtime["autocast_dtype"], field="precision_runtime.autocast_dtype"
    )
    if autocast_dtype not in {"float32", "bfloat16", "float16"}:
        raise ConfigContractError("precision_runtime.autocast_dtype 不受支持")
    scaler_enabled = _strict_bool(
        precision_runtime["grad_scaler_enabled"], field="precision_runtime.grad_scaler_enabled"
    )
    initial_scale = _positive_number(
        precision_runtime["initial_scale"], field="precision_runtime.initial_scale"
    )
    growth_factor = _finite_number(
        precision_runtime["growth_factor"], field="precision_runtime.growth_factor", minimum=1.0, minimum_exclusive=True
    )
    backoff_factor = _finite_number(
        precision_runtime["backoff_factor"], field="precision_runtime.backoff_factor",
        minimum=0.0, maximum=1.0, minimum_exclusive=True, maximum_exclusive=True,
    )
    growth_interval = _strict_int(
        precision_runtime["growth_interval"], field="precision_runtime.growth_interval", minimum=1
    )
    global_found_inf_reduce = _strict_bool(
        precision_runtime["global_found_inf_reduce"],
        field="precision_runtime.global_found_inf_reduce",
    )
    base_precision = base_config.section("precision")
    base_amp = bool(base_precision["amp"])
    base_compute_dtype = str(base_precision["compute_dtype"])
    if autocast_enabled != base_amp:
        raise ConfigContractError("precision_runtime.autocast_enabled 必须与 base_config.precision.amp 一致")
    if autocast_enabled and autocast_dtype != base_compute_dtype:
        raise ConfigContractError("autocast dtype 必须与 base_config compute_dtype 一致")
    if not autocast_enabled and (autocast_dtype != "float32" or scaler_enabled):
        raise ConfigContractError("关闭 autocast 时 dtype 必须为 float32 且 scaler 关闭")
    if scaler_enabled and (
        autocast_dtype != "float16" or base_config.section("runtime")["device"] != "cuda"
    ):
        raise ConfigContractError("GradScaler 只允许 CUDA float16 路径")
    if base_identity["run_intent"] == "formal" and base_amp and not global_found_inf_reduce:
        raise ConfigContractError("formal AMP 必须启用 global_found_inf_reduce")

    optimizer_runtime = _expect_mapping(
        payload["optimizer_runtime"], field="optimizer_runtime"
    )
    _expect_fields(optimizer_runtime, _OPTIMIZER_RUNTIME_FIELDS, field="optimizer_runtime")
    raw_betas = optimizer_runtime["betas"]
    betas: list[float] | None
    if raw_betas is None:
        betas = None
    else:
        if not isinstance(raw_betas, list) or len(raw_betas) != 2:
            raise ConfigContractError("optimizer_runtime.betas 必须是长度为 2 的数组或 null")
        betas = [
            _finite_number(beta, field=f"optimizer_runtime.betas[{index}]", minimum=0.0, maximum=1.0, maximum_exclusive=True)
            for index, beta in enumerate(raw_betas)
        ]
    eps = _positive_number(optimizer_runtime["eps"], field="optimizer_runtime.eps", optional=True)
    amsgrad = _strict_bool(optimizer_runtime["amsgrad"], field="optimizer_runtime.amsgrad")
    dampening = _finite_number(
        optimizer_runtime["dampening"], field="optimizer_runtime.dampening", minimum=0.0
    )
    nesterov = _strict_bool(optimizer_runtime["nesterov"], field="optimizer_runtime.nesterov")
    maximize = _strict_bool(optimizer_runtime["maximize"], field="optimizer_runtime.maximize")
    capturable = _strict_bool(optimizer_runtime["capturable"], field="optimizer_runtime.capturable")
    differentiable = _strict_bool(
        optimizer_runtime["differentiable"], field="optimizer_runtime.differentiable"
    )
    base_optimizer = base_config.section("optimizer")
    optimizer_type = str(base_optimizer["type"])
    if optimizer_type == "adamw":
        if betas is None or eps is None or dampening != 0.0 or nesterov:
            raise ConfigContractError("AdamW 必须设置 betas/eps，且不得设置 dampening/nesterov")
    elif betas is not None or eps is not None or amsgrad:
        raise ConfigContractError("SGD/Momentum 的 betas、eps 必须为 null 且 amsgrad=false")
    if nesterov and (float(base_optimizer["momentum"]) <= 0 or dampening != 0.0):
        raise ConfigContractError("nesterov 要求 momentum>0 且 dampening=0")
    if amsgrad or maximize or capturable or differentiable:
        raise ConfigContractError("当前 correctness bridge 不支持 amsgrad/maximize/capturable/differentiable")

    launcher = _expect_mapping(payload["launcher"], field="launcher")
    _expect_fields(launcher, _LAUNCHER_FIELDS, field="launcher")
    launcher_kind = _non_empty_string(launcher["kind"], field="launcher.kind")
    launcher_backend = _non_empty_string(launcher["backend"], field="launcher.backend")
    launcher_world_size = _strict_int(launcher["world_size"], field="launcher.world_size", minimum=1)
    init_method = _non_empty_string(launcher["init_method"], field="launcher.init_method")
    init_ref = _logical_path(launcher["init_ref"], field="launcher.init_ref", optional=True)
    rendezvous_id = _optional_string(launcher["rendezvous_id"], field="launcher.rendezvous_id")
    launcher_max_restarts = _strict_int(
        launcher["max_restarts"], field="launcher.max_restarts", minimum=0
    )
    base_distributed = base_config.section("distributed")
    if launcher_backend != base_distributed["backend"] or launcher_world_size != base_distributed["world_size"]:
        raise ConfigContractError("launcher backend/world_size 必须与 base_config.distributed 一致")
    if launcher_kind == "local":
        if (
            launcher_backend != "local" or launcher_world_size != 1 or init_method != "local"
            or init_ref is not None or rendezvous_id is not None or launcher_max_restarts != 0
        ):
            raise ConfigContractError("local launcher 必须是 local backend/world_size=1/local init")
    elif launcher_kind == "torchrun":
        if launcher_backend not in {"gloo", "nccl"} or init_method not in {"env", "file"}:
            raise ConfigContractError("torchrun 必须使用 gloo/nccl 与 env/file init")
        if rendezvous_id is None:
            raise ConfigContractError("torchrun 必须设置 rendezvous_id")
        if init_method == "file" and init_ref is None:
            raise ConfigContractError("file init 必须设置 launcher.init_ref")
        if init_method == "env" and init_ref is not None:
            raise ConfigContractError("env init 不得设置 launcher.init_ref")
    else:
        raise ConfigContractError("launcher.kind 只能是 local 或 torchrun")
    if base_identity["run_intent"] == "formal":
        formal_capabilities = set(task.formal_eligibility.required_capabilities)
        runtime_device = base_config.section("runtime")["device"]
        if "cuda" in formal_capabilities and runtime_device != "cuda":
            raise ConfigContractError("formal 任务声明需要 CUDA，但 base_config.runtime.device 不是 cuda")
        if "nccl" in formal_capabilities and (
            launcher_kind != "torchrun" or launcher_backend != "nccl" or launcher_world_size < 2
        ):
            raise ConfigContractError("formal NCCL 任务必须使用 world_size>=2 的 torchrun/nccl launcher")

    orchestration = _expect_mapping(payload["orchestration"], field="orchestration")
    _expect_fields(orchestration, _ORCHESTRATION_FIELDS, field="orchestration")
    route_spec_ref = _logical_path(
        orchestration["route_spec_ref"], field="orchestration.route_spec_ref", optional=True
    )
    quadrature_decision_ref = _logical_path(
        orchestration["quadrature_decision_ref"],
        field="orchestration.quadrature_decision_ref",
        optional=True,
    )
    matrix_ref = _logical_path(
        orchestration["matrix_ref"], field="orchestration.matrix_ref", optional=True
    )
    paired_design = _expect_mapping(
        orchestration["paired_design"], field="orchestration.paired_design"
    )
    _expect_fields(paired_design, _PAIRED_DESIGN_FIELDS, field="orchestration.paired_design")
    paired_enabled = _strict_bool(
        paired_design["enabled"], field="orchestration.paired_design.enabled"
    )
    paired_kind = _non_empty_string(
        paired_design["design"], field="orchestration.paired_design.design"
    )
    if paired_kind not in {"none", "shared_draws", "independent_halves", "matched_seeds"}:
        raise ConfigContractError("orchestration.paired_design.design 不受支持")
    paired_mapping_ref = _logical_path(
        paired_design["mapping_ref"],
        field="orchestration.paired_design.mapping_ref",
        optional=True,
    )
    budget_unit = _non_empty_string(
        paired_design["budget_unit"], field="orchestration.paired_design.budget_unit"
    )
    if budget_unit not in {"samples", "tokens", "gradient_evaluations", "wall_clock"}:
        raise ConfigContractError("orchestration.paired_design.budget_unit 不受支持")
    if paired_enabled and (paired_kind == "none" or paired_mapping_ref is None):
        raise ConfigContractError("启用 paired design 时必须冻结 design 与 mapping_ref")
    if not paired_enabled and (paired_kind != "none" or paired_mapping_ref is not None):
        raise ConfigContractError("关闭 paired design 时 design 必须为 none 且 mapping_ref 为 null")
    input_result_refs = [
        _logical_path(item, field=f"orchestration.input_result_refs[{index}]")
        for index, item in enumerate(
            orchestration["input_result_refs"]
            if isinstance(orchestration["input_result_refs"], list)
            else []
        )
    ]
    if not isinstance(orchestration["input_result_refs"], list):
        raise ConfigContractError("orchestration.input_result_refs 必须是数组")
    if len(set(input_result_refs)) != len(input_result_refs):
        raise ConfigContractError("orchestration.input_result_refs 不能重复")
    if task.task_id in _ROUTE_SPEC_TASKS and route_spec_ref is None:
        raise ConfigContractError(f"{task.task_id} 必须设置 orchestration.route_spec_ref")
    if task.task_id in _MATRIX_INPUT_TASKS and matrix_ref is None:
        raise ConfigContractError(f"{task.task_id} 必须设置 orchestration.matrix_ref")
    if task.task_id in _PAIRED_DESIGN_TASKS and not paired_enabled:
        raise ConfigContractError(f"{task.task_id} 必须启用冻结的 paired design")
    if (
        base_identity["run_intent"] == "formal"
        and task.stage >= 4
        and task.runner_kind in _TRAINING_RUNNERS
        and quadrature_decision_ref is None
    ):
        raise ConfigContractError("Stage 4+ formal 训练必须绑定 quadrature_decision_ref")

    recovery = _expect_mapping(payload["recovery"], field="recovery")
    _expect_fields(recovery, _RECOVERY_FIELDS, field="recovery")
    recovery_mode = _non_empty_string(recovery["mode"], field="recovery.mode")
    safe_boundary = _non_empty_string(recovery["safe_boundary"], field="recovery.safe_boundary")
    if recovery_mode != task.recovery_mode.value or safe_boundary != task.safe_boundary.value:
        raise ConfigContractError("recovery mode/safe_boundary 必须与任务目录冻结值一致")
    resume_ref = _logical_path(recovery["resume_ref"], field="recovery.resume_ref", optional=True)
    max_restarts = _strict_int(recovery["max_restarts"], field="recovery.max_restarts", minimum=0)
    if resume_ref is not None and task.safe_boundary.value == "none":
        raise ConfigContractError("safe_boundary=none 的任务不能指定 resume_ref")

    artifacts = _expect_mapping(payload["artifacts"], field="artifacts")
    _expect_fields(artifacts, _ARTIFACT_FIELDS, field="artifacts")
    output_dir = _logical_path(artifacts["output_dir"], field="artifacts.output_dir")
    required_kinds = _normalize_string_list(artifacts["required_kinds"], field="artifacts.required_kinds")
    if tuple(required_kinds) != task.artifact_kinds:
        raise ConfigContractError("artifacts.required_kinds 必须与任务目录完全一致")
    publish_partial = _strict_bool(artifacts["publish_partial"], field="artifacts.publish_partial")
    if publish_partial and base_identity["run_intent"] == "formal":
        raise ConfigContractError("formal 配置不得把 partial artifact 作为正式发布物")

    normalized: dict[str, JSONValue] = {
        "schema_version": CONFIG_V2_SCHEMA_VERSION,
        "task_id": task.task_id,
        "base_config": base_config.to_dict(),
        "execution": {
            "runner_kind": runner_kind,
            "timeout_seconds": timeout_seconds,
            "max_attempts": max_attempts,
            "dry_run": dry_run,
            "fail_on_blocked": fail_on_blocked,
        },
        "training": {
            "max_steps": max_steps,
            "max_epochs": max_epochs,
            "validation_every_steps": validation_every,
            "gradient_clip_max_norm": clip_norm,
            "deterministic_algorithms": deterministic,
        },
        "scheduler": {
            "kind": scheduler_kind,
            "warmup_steps": warmup_steps,
            "total_steps": total_steps,
        },
        "data_loader": {
            "num_workers": num_workers,
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "drop_last": drop_last,
            "cursor_policy": cursor_policy,
        },
        "providers": {
            "kind": provider_kind,
            **provider_refs,
            "task_type": provider_task_type,
            "task_name": provider_task_name,
            "num_labels": num_labels,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        },
        "evaluation": {
            "enabled": evaluation_enabled,
            "split": evaluation_split,
            "every_steps": evaluation_every,
            "batch_size": evaluation_batch_size,
            "max_batches": evaluation_max_batches,
            "metrics": evaluation_metrics,
            "save_predictions": save_predictions,
        },
        "profiling": {
            "enabled": profiling_enabled,
            "warmup_steps": profiling_warmup,
            "measure_steps": profiling_measure,
            "repetitions": profiling_repetitions,
            "capture_memory": capture_memory,
            "capture_throughput": capture_throughput,
            "capture_communication": capture_communication,
            "synchronize_device": synchronize_device,
        },
        "checkpoint_schedule": {
            "segments": segments,
            "save_on_phase_end": save_on_phase_end,
            "save_optimizer": save_optimizer,
            "save_rng": save_rng,
            "save_data_state": save_data_state,
        },
        "precision_runtime": {
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": autocast_dtype,
            "grad_scaler_enabled": scaler_enabled,
            "initial_scale": initial_scale,
            "growth_factor": growth_factor,
            "backoff_factor": backoff_factor,
            "growth_interval": growth_interval,
            "global_found_inf_reduce": global_found_inf_reduce,
        },
        "optimizer_runtime": {
            "betas": betas,
            "eps": eps,
            "amsgrad": amsgrad,
            "dampening": dampening,
            "nesterov": nesterov,
            "maximize": maximize,
            "capturable": capturable,
            "differentiable": differentiable,
        },
        "launcher": {
            "kind": launcher_kind,
            "backend": launcher_backend,
            "world_size": launcher_world_size,
            "init_method": init_method,
            "init_ref": init_ref,
            "rendezvous_id": rendezvous_id,
            "max_restarts": launcher_max_restarts,
        },
        "orchestration": {
            "route_spec_ref": route_spec_ref,
            "quadrature_decision_ref": quadrature_decision_ref,
            "matrix_ref": matrix_ref,
            "paired_design": {
                "enabled": paired_enabled,
                "design": paired_kind,
                "mapping_ref": paired_mapping_ref,
                "budget_unit": budget_unit,
            },
            "input_result_refs": input_result_refs,
        },
        "recovery": {
            "mode": recovery_mode,
            "resume_ref": resume_ref,
            "max_restarts": max_restarts,
            "safe_boundary": safe_boundary,
        },
        "artifacts": {
            "output_dir": output_dir,
            "required_kinds": required_kinds,
            "publish_partial": publish_partial,
        },
    }
    for path in task.config_paths:
        _lookup_path(normalized, path)
    return normalized, base_config, task


@dataclass(frozen=True, slots=True)
class ResolvedConfigV2:
    """通过 v1 科学合同和 v2 执行合同的不可变配置。"""

    _payload: dict[str, JSONValue]
    _catalog: TaskCatalog = DEFAULT_TASK_CATALOG

    def __post_init__(self) -> None:
        normalized, _, _ = _normalize_payload(self._payload, catalog=self._catalog)
        object.__setattr__(self, "_payload", deepcopy(normalized))

    @classmethod
    def resolve(
        cls,
        base_config: ResolvedConfig | Mapping[str, Any],
        *,
        task_id: str,
        overrides: Mapping[str, Any] | None = None,
        catalog: TaskCatalog = DEFAULT_TASK_CATALOG,
    ) -> "ResolvedConfigV2":
        """从已解析 v1 配置生成 v2；不会为训练任务猜测运行长度。"""

        base = (
            base_config
            if isinstance(base_config, ResolvedConfig)
            else ResolvedConfig.from_mapping(base_config)
        )
        try:
            task = catalog.get(task_id)
        except TaskCatalogError as error:
            raise ConfigContractError(str(error)) from error
        sections = _apply_overrides(_defaults(task, base), overrides)
        payload: dict[str, JSONValue] = {
            "schema_version": CONFIG_V2_SCHEMA_VERSION,
            "task_id": task_id,
            "base_config": base.to_dict(),
            **sections,
        }
        return cls(payload, catalog)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        catalog: TaskCatalog = DEFAULT_TASK_CATALOG,
    ) -> "ResolvedConfigV2":
        """严格读取 v2 wire object，并验证其中两个摘要。"""

        mapping = _expect_mapping(value, field="ResolvedConfigV2")
        _expect_fields(mapping, _WIRE_FIELDS, field="ResolvedConfigV2")
        payload = {key: deepcopy(item) for key, item in mapping.items() if key not in {"config_hash", "full_hash"}}
        config = cls(payload, catalog)
        if mapping["config_hash"] != config.config_hash:
            raise ConfigContractError("ResolvedConfigV2 config_hash 与内容不一致")
        if mapping["full_hash"] != config.full_hash:
            raise ConfigContractError("ResolvedConfigV2 full_hash 与内容不一致")
        return config

    @property
    def task_id(self) -> str:
        return self._payload["task_id"]  # type: ignore[return-value]

    @property
    def task_definition(self) -> TaskDefinition:
        return self._catalog.get(self.task_id)

    @property
    def base_config(self) -> ResolvedConfig:
        value = self._payload["base_config"]
        assert isinstance(value, dict)
        return ResolvedConfig.from_mapping(value)

    @property
    def run_intent(self) -> str:
        return str(self.base_config.section("identity")["run_intent"])

    @property
    def formal_eligible(self) -> bool:
        """仅表示静态配置意图和任务类型允许 formal；运行证据仍由 task runtime 检查。"""

        return self.base_config.formal_eligible and self.task_definition.formal_eligibility.supported

    def section(self, name: str) -> JSONValue:
        if name not in _TOP_LEVEL_FIELDS - {"schema_version", "task_id"}:
            raise KeyError(name)
        return deepcopy(self._payload[name])

    @property
    def full_hash(self) -> str:
        """绑定本次启动读取的全部 v2 字段，包括机器落盘与恢复引用。"""

        return canonical_json_hash(self._payload)

    @property
    def config_hash(self) -> str:
        """语义摘要；排除超时、重试次数、机器输出目录和具体恢复对象。"""

        base = self.base_config
        execution = self.section("execution")
        training = self.section("training")
        scheduler = self.section("scheduler")
        data_loader = self.section("data_loader")
        providers = self.section("providers")
        evaluation = self.section("evaluation")
        profiling = self.section("profiling")
        checkpoint_schedule = self.section("checkpoint_schedule")
        precision_runtime = self.section("precision_runtime")
        optimizer_runtime = self.section("optimizer_runtime")
        launcher = self.section("launcher")
        orchestration = self.section("orchestration")
        recovery = self.section("recovery")
        artifacts = self.section("artifacts")
        assert isinstance(execution, dict)
        assert isinstance(providers, dict)
        assert isinstance(launcher, dict)
        assert isinstance(recovery, dict)
        assert isinstance(artifacts, dict)
        semantic: dict[str, JSONValue] = {
            "schema_version": CONFIG_V2_SCHEMA_VERSION,
            "task_id": self.task_id,
            "base_config_hash": base.config_hash,
            "execution": {
                "runner_kind": execution["runner_kind"],
                "dry_run": execution["dry_run"],
            },
            "training": training,
            "scheduler": scheduler,
            "data_loader": data_loader,
            "providers": {
                key: value
                for key, value in providers.items()
                if key not in {"model_root_ref", "data_root_ref", "tokenizer_root_ref"}
            },
            "evaluation": evaluation,
            "profiling": profiling,
            "checkpoint_schedule": checkpoint_schedule,
            "precision_runtime": precision_runtime,
            "optimizer_runtime": optimizer_runtime,
            "launcher": {
                "kind": launcher["kind"],
                "backend": launcher["backend"],
                "world_size": launcher["world_size"],
                "init_method": launcher["init_method"],
            },
            "orchestration": orchestration,
            "recovery": {
                "mode": recovery["mode"],
                "safe_boundary": recovery["safe_boundary"],
            },
            "artifacts": {
                "required_kinds": artifacts["required_kinds"],
                "publish_partial": artifacts["publish_partial"],
            },
        }
        return canonical_json_hash(semantic)

    def to_dict(self) -> dict[str, JSONValue]:
        result = deepcopy(self._payload)
        result["config_hash"] = self.config_hash
        result["full_hash"] = self.full_hash
        return result


def load_resolved_config_compatible(
    value: ResolvedConfigV2 | ResolvedConfig | Mapping[str, Any],
    *,
    task_id: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    catalog: TaskCatalog = DEFAULT_TASK_CATALOG,
) -> ResolvedConfigV2:
    """读取 v2 或把 legacy v1 包装为 v2。

    legacy ``identity.task`` 只有在它本身已经是 catalog 中的 canonical ID 时才可省略
    ``task_id``。旧的自由文本任务名没有可靠映射，必须由调用者显式选择，避免把一个
    Stage 的配置送给另一个 runner。
    """

    if isinstance(value, ResolvedConfigV2):
        if task_id is not None and value.task_id != task_id:
            raise ConfigContractError("显式 task_id 与 ResolvedConfigV2 不一致")
        if overrides is not None:
            raise ConfigContractError("已解析 v2 不能再次应用 legacy overrides")
        return value
    if isinstance(value, ResolvedConfig):
        base = value
    else:
        mapping = _expect_mapping(value, field="config")
        if mapping.get("schema_version") == CONFIG_V2_SCHEMA_VERSION:
            if overrides is not None:
                raise ConfigContractError("v2 wire object 不接受 legacy overrides")
            loaded = ResolvedConfigV2.from_mapping(mapping, catalog=catalog)
            if task_id is not None and loaded.task_id != task_id:
                raise ConfigContractError("显式 task_id 与 v2 wire object 不一致")
            return loaded
        # v1 的 schema_version 位于 identity 分区；严格 parser 会拒绝其他 legacy shape。
        base = ResolvedConfig.from_mapping(mapping)

    legacy_identity = base.section("identity")
    legacy_schema = legacy_identity["schema_version"]
    if legacy_schema != CONFIG_SCHEMA_VERSION:
        raise ConfigContractError(f"不支持的 legacy schema：{legacy_schema!r}")
    selected_task_id = task_id
    if selected_task_id is None:
        candidate = legacy_identity["task"]
        if isinstance(candidate, str) and candidate in catalog.task_ids:
            selected_task_id = candidate
        else:
            raise ConfigContractError(
                "legacy identity.task 不是 canonical task ID；兼容加载必须显式提供 task_id"
            )
    return ResolvedConfigV2.resolve(
        base,
        task_id=selected_task_id,
        overrides=overrides,
        catalog=catalog,
    )


__all__ = [
    "CONFIG_V2_SCHEMA_VERSION",
    "ResolvedConfigV2",
    "load_resolved_config_compatible",
]
