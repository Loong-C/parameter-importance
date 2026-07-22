"""Stage 0--9 缩小本机流水线的确定性验收入口。

本模块不是又一套实验实现。它只负责把已经登记到 :class:`TaskRuntime` 的正式
runner 按依赖顺序组合起来，用 tiny Torch 模型与 synthetic provider 验证以下事实：

* 训练、在线重要性、固定状态估计器与路径积分都真正执行；
* Stage 4--6 路线读取 Stage 2/3 的 hash-bound fixture 决策；
* Stage 7/8 分别执行剪枝与消融核心，Stage 9 消费它们的冻结行产物；
* 两个全新工作根不会因为绝对路径、墙钟时间或临时目录而得到不同科学哈希。

该入口始终是 ``local_fixture``，不会生成或提升任何 formal Gate。输出根应由调用者
提供；其中所有 task 使用相同的 POSIX 逻辑相对路径，因此跨工作根比较时路径不会进入
科学摘要。完整训练与 task artifact 仍由各自的两阶段 commit 存储负责。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping, Sequence

from ..contracts import ResolvedConfig, load_canonical_json
from ..contracts.config_v2 import ResolvedConfigV2
from ..contracts.jsonio import JSONValue, canonical_json_hash, write_canonical_json
from ..contracts.task_catalog import DEFAULT_TASK_CATALOG
from ..runtime.task_artifacts import TaskArtifactStore
from ..runtime.task_runtime import TaskRunResult, TaskRunStatus
from ..local_fixture import run_local_fixture
from .routes import TrainingPhaseSpec, TrainingRouteSpec
from .stage2 import EstimatorDecision
from .task_runners import build_default_task_runtime


_SCHEMA_VERSION = "full-local-fixture-pipeline-v1"
@dataclass(frozen=True, slots=True)
class FullFixturePipelineResult:
    """不含物理路径和计时字段的跨阶段确定性摘要。

    ``task_artifact_hashes`` 的外层键为 task ID，内层键为 catalog artifact kind。
    registry/seed 可能来自多个独立 fixture 子系统，因此同时发布去重后的成员集合与
    集合摘要；两次运行必须逐成员相等，而不只是集合摘要碰巧相等。
    """

    config_hashes: Mapping[str, str]
    task_artifact_hashes: Mapping[str, Mapping[str, str]]
    coordinate_registry_hashes: tuple[str, ...]
    seed_plan_hashes: tuple[str, ...]
    source_table_hash: str
    table_hash: str
    chart_hash: str
    report_hash: str
    replay_hash: str

    def _payload(self) -> dict[str, JSONValue]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "scope": "local_fixture",
            "formal_eligible": False,
            "covered_stages": list(range(10)),
            "config_hashes": dict(sorted(self.config_hashes.items())),
            "task_artifact_hashes": {
                task_id: dict(sorted(hashes.items()))
                for task_id, hashes in sorted(self.task_artifact_hashes.items())
            },
            "coordinate_registry_hashes": list(self.coordinate_registry_hashes),
            "coordinate_registry_set_hash": canonical_json_hash(
                {"hashes": list(self.coordinate_registry_hashes)}
            ),
            "seed_plan_hashes": list(self.seed_plan_hashes),
            "seed_plan_set_hash": canonical_json_hash(
                {"hashes": list(self.seed_plan_hashes)}
            ),
            "source_table_hash": self.source_table_hash,
            "table_hash": self.table_hash,
            "chart_hash": self.chart_hash,
            "report_hash": self.report_hash,
            "replay_hash": self.replay_hash,
        }

    @property
    def result_hash(self) -> str:
        """整个流水线的 canonical 科学摘要哈希。"""

        return canonical_json_hash(self._payload())

    def to_dict(self) -> dict[str, JSONValue]:
        """返回可直接写入 canonical JSON 的结果。"""

        return self._payload() | {"result_hash": self.result_hash}


def _base_v1(config_path: Path, task_id: str, *, classification: bool) -> ResolvedConfig:
    value = deepcopy(load_canonical_json(config_path))
    if not isinstance(value, dict):
        raise ValueError("FULL_FIXTURE_BASE_CONFIG_NOT_OBJECT")
    task = DEFAULT_TASK_CATALOG.get(task_id)
    identity = value["identity"]
    if not isinstance(identity, dict):
        raise ValueError("FULL_FIXTURE_BASE_IDENTITY_NOT_OBJECT")
    identity["stage"] = task.stage
    identity["task"] = task_id
    if classification:
        value["loss"].update(  # type: ignore[union-attr]
            {"task_type": "sequence_classification", "weighting": "sample"}
        )
        value["data"].update(  # type: ignore[union-attr]
            {"statistical_unit": "sample", "weight_unit": "sample"}
        )
        value["model"]["architecture"] = "tiny-sequence-classifier"  # type: ignore[index]
        value["batching"].update(  # type: ignore[union-attr]
            {
                "global_batch_size": 2,
                "per_device_batch_size": 2,
                "microbatch_size": 1,
                "accumulation_steps": 1,
            }
        )
    return ResolvedConfig.from_mapping(value)


def _config(
    config_path: Path,
    task_id: str,
    *,
    output_dir: str,
    input_refs: Sequence[str] = (),
    route_ref: str | None = None,
    quadrature_ref: str | None = None,
    matrix_ref: str | None = None,
    classification: bool = False,
    training: bool = False,
) -> ResolvedConfigV2:
    orchestration: dict[str, JSONValue] = {"input_result_refs": list(input_refs)}
    if route_ref is not None:
        orchestration["route_spec_ref"] = route_ref
    if quadrature_ref is not None:
        orchestration["quadrature_decision_ref"] = quadrature_ref
    if matrix_ref is not None:
        orchestration["matrix_ref"] = matrix_ref
    if task_id in {"stage2.05_paired_estimator_runner", "stage2.07_main_sweep"}:
        orchestration["paired_design"] = {
            "enabled": True,
            "design": "shared_draws",
            "mapping_ref": "fixture/paired-mapping.json",
            "budget_unit": "gradient_evaluations",
        }
    elif task_id in {"stage6.evaluate", "stage6.compare"}:
        if matrix_ref is None:
            raise ValueError(f"FULL_FIXTURE_STAGE6_MATRIX_REF_REQUIRED:{task_id}")
        orchestration["paired_design"] = {
            "enabled": True,
            "design": "matched_seeds",
            "mapping_ref": matrix_ref,
            "budget_unit": "samples",
        }
    overrides: dict[str, JSONValue] = {
        "providers": {"num_labels": 3 if classification else None},
        "orchestration": orchestration,
        "artifacts": {"output_dir": output_dir},
    }
    if training:
        overrides.update(
            {
                "training": {
                    "max_steps": 1,
                    "validation_every_steps": 1,
                },
                "evaluation": {
                    "enabled": True,
                    "split": "fixture",
                    "every_steps": 1,
                    "batch_size": 2,
                    "max_batches": 1,
                    "metrics": ["loss", "accuracy"],
                },
                "checkpoint_schedule": {
                    "segments": [
                        {"start_step": 0, "end_step": None, "every_steps": 1}
                    ]
                },
            }
        )
    return ResolvedConfigV2.resolve(
        _base_v1(config_path, task_id, classification=classification),
        task_id=task_id,
        overrides=overrides,
    )


def _load_published(
    root: Path,
    config: ResolvedConfigV2,
    result: TaskRunResult,
) -> tuple[dict[str, str], dict[str, Mapping[str, object]]]:
    if result.status is not TaskRunStatus.PASS:
        raise RuntimeError(
            f"FULL_FIXTURE_TASK_NOT_PASS:{result.task_id}:{result.status.value}:"
            f"{result.error_code}:{result.message}"
        )
    artifacts = config.section("artifacts")
    assert isinstance(artifacts, dict)
    store = TaskArtifactStore(root, str(artifacts["output_dir"]))
    hashes: dict[str, str] = {}
    payloads: dict[str, Mapping[str, object]] = {}
    for kind, reference in result.artifact_refs.items():
        published = store.load_commit(reference)
        value = load_canonical_json(root / published.object_ref)
        if not isinstance(value, dict) or not isinstance(value.get("payload"), dict):
            raise ValueError(f"FULL_FIXTURE_TASK_PAYLOAD_INVALID:{result.task_id}:{kind}")
        hashes[kind] = published.artifact_hash
        payloads[kind] = value["payload"]
    return hashes, payloads


def _walk_mappings(value: object):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _find_schema(values: Sequence[object], schema_version: str) -> Mapping[str, object]:
    matches: dict[str, Mapping[str, object]] = {}
    for value in values:
        for mapping in _walk_mappings(value):
            if mapping.get("schema_version") == schema_version:
                digest = canonical_json_hash(dict(mapping))
                matches[digest] = mapping
    if len(matches) != 1:
        raise ValueError(
            f"FULL_FIXTURE_SCHEMA_CARDINALITY:{schema_version}:{len(matches)}"
        )
    return next(iter(matches.values()))


def _collect_declared_hashes(values: Sequence[object], field_names: set[str]) -> tuple[str, ...]:
    found: set[str] = set()
    for value in values:
        for mapping in _walk_mappings(value):
            for name in field_names:
                item = mapping.get(name)
                if isinstance(item, str) and len(item) == 64:
                    found.add(item)
    return tuple(sorted(found))


def _route(decision: EstimatorDecision) -> TrainingRouteSpec:
    phases = (
        TrainingPhaseSpec(
            "pretrain",
            "pretrain",
            "tiny-init-v1",
            "tiny-model-v1",
            "tiny-pretrain-data-v1",
            "logical-pretrain-checkpoint",
            1,
            metadata={"runtime": {"data_seed_offset": 11}},
        ),
        TrainingPhaseSpec(
            "direct",
            "direct_supervised",
            "tiny-init-v1",
            "tiny-model-v1",
            "tiny-supervised-data-v1",
            "logical-direct-checkpoint",
            1,
            task_id="tiny-classification",
            metadata={"runtime": {"data_seed_offset": 23}},
        ),
        TrainingPhaseSpec(
            "finetune",
            "finetune",
            "tiny-init-v1",
            "tiny-model-v1",
            "tiny-supervised-data-v1",
            "logical-finetune-checkpoint",
            1,
            parent_phase_id="pretrain",
            input_checkpoint_id="logical-pretrain-checkpoint",
            task_id="tiny-classification",
            metadata={"runtime": {"data_seed_offset": 23}},
        ),
    )
    return TrainingRouteSpec(
        "full-local-fixture-stage456-route",
        phases,
        "local_fixture",
        decision,
    )


def run_full_fixture_pipeline(
    *,
    workspace_root: str | Path,
    base_config_path: str | Path,
) -> FullFixturePipelineResult:
    """在一个全新工作根中执行 Stage 0--9 缩小流水线。

    参数均为本机路径，但路径不会进入返回结果或 ``result_hash``。调用者可用两个空目录
    各执行一次，并直接比较两个 :meth:`FullFixturePipelineResult.to_dict`。
    """

    root = Path(workspace_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(base_config_path).resolve()
    runtime = build_default_task_runtime(root)
    configs: dict[str, ResolvedConfigV2] = {}
    artifact_hashes: dict[str, dict[str, str]] = {}
    payloads_by_task: dict[str, dict[str, Mapping[str, object]]] = {}
    all_payloads: list[Mapping[str, object]] = []
    extra_config_hashes: dict[str, str] = {}

    def execute(config: ResolvedConfigV2) -> TaskRunResult:
        result = runtime.execute(config)
        hashes, payloads = _load_published(root, config, result)
        configs[config.task_id] = config
        artifact_hashes[config.task_id] = hashes
        payloads_by_task[config.task_id] = payloads
        all_payloads.extend(payloads.values())
        return result

    # Stage 0 的 smoke 走真实 TrainingEngine，已经同时覆盖 Stage 1 在线重要性接线；
    # Stage 1 再发布独立合同 task，避免为覆盖标签重复训练同一个 tiny 模型。
    execute(
        _config(
            config_path,
            "stage0.06_single_gpu_smoke",
            output_dir="runs/stage0-06-single-gpu-smoke",
            classification=True,
            training=True,
        )
    )
    execute(
        _config(
            config_path,
            "stage1.01_entry_and_contract",
            output_dir="runs/stage1-01-entry-and-contract",
            classification=True,
        )
    )

    # Stage 2/3 使用仓库既有的一体化 fixture 核；它内部仍真实运行 reference、paired
    # estimator、解析路径积分与 Stage 9 数学摘要，只是不为 21 个目录 task 重复发布包装
    # artifact。通过相同的相对 runtime root，两个物理工作根得到完全一致的 config hash。
    stage23_config_value = deepcopy(load_canonical_json(config_path))
    if not isinstance(stage23_config_value, dict):
        raise ValueError("FULL_FIXTURE_STAGE23_CONFIG_NOT_OBJECT")
    stage23_runtime = stage23_config_value.get("runtime")
    if not isinstance(stage23_runtime, dict):
        raise ValueError("FULL_FIXTURE_STAGE23_RUNTIME_NOT_OBJECT")
    stage23_runtime.update(
        {
            "output_root": "fixture-output",
            "temp_root": "fixture-output/tmp",
            "cache_root": "fixture-output/cache",
        }
    )
    stage23_config = ResolvedConfig.from_mapping(stage23_config_value)
    stage23_config_ref = root / "inputs/stage23-resolved-config-v1.json"
    write_canonical_json(stage23_config_ref, stage23_config.to_dict())
    previous_cwd = Path.cwd()
    try:
        os.chdir(root)
        stage23_summary = run_local_fixture(
            config_path=stage23_config_ref,
            output_dir="fixture-output/stage23",
        )
    finally:
        os.chdir(previous_cwd)
    stage23_value = load_canonical_json(root / "fixture-output/stage23/local-fixture-result.json")
    if not isinstance(stage23_value, dict):
        raise ValueError("FULL_FIXTURE_STAGE23_RESULT_NOT_OBJECT")
    all_payloads.append(stage23_value)
    extra_config_hashes["stage2_3.fixture_core"] = stage23_config.config_hash
    artifact_hashes["stage2_3.fixture_core"] = {
        "local_fixture_result": str(stage23_summary["artifact_hash"])
    }

    stage2_value = stage23_value.get("stage2")
    stage3_value = stage23_value.get("stage3")
    if not isinstance(stage2_value, dict) or not isinstance(stage3_value, dict):
        raise ValueError("FULL_FIXTURE_STAGE23_SECTIONS_INVALID")
    estimator_mapping = stage2_value.get("estimator_decision")
    quadrature_mapping = stage3_value.get("quadrature_decision")
    if not isinstance(estimator_mapping, dict) or not isinstance(quadrature_mapping, dict):
        raise ValueError("FULL_FIXTURE_STAGE23_DECISIONS_MISSING")
    decision = EstimatorDecision.from_mapping(estimator_mapping)
    quadrature_mapping = {
        "schema_version": "quadrature-decision-v1",
        **quadrature_mapping,
    }
    fixture_route = _route(decision)
    route_ref = "inputs/full-fixture-route.json"
    write_canonical_json(root / route_ref, fixture_route.to_dict())
    phase_by_type = {phase.phase_type: phase for phase in fixture_route.phases}
    direct_route = TrainingRouteSpec(
        "full-local-fixture-stage6-direct",
        (phase_by_type["direct_supervised"],),
        "local_fixture",
        decision,
    )
    finetune_route = TrainingRouteSpec(
        "full-local-fixture-stage6-pretrain-finetune",
        (phase_by_type["pretrain"], phase_by_type["finetune"]),
        "local_fixture",
        decision,
    )
    direct_route_ref = "inputs/full-fixture-stage6-direct-route.json"
    finetune_route_ref = "inputs/full-fixture-stage6-finetune-route.json"
    write_canonical_json(root / direct_route_ref, direct_route.to_dict())
    write_canonical_json(root / finetune_route_ref, finetune_route.to_dict())
    quadrature_ref = "inputs/full-fixture-quadrature-decision.json"
    quadrature_wrapper: dict[str, JSONValue] = {
        "schema_version": "task-output-artifact-v1",
        "task_id": "stage3.09_cost_and_method_selection",
        "artifact_kind": "quadrature_decision",
        "config_hash": stage23_config.config_hash,
        "run_intent": "local_fixture",
        "formal_eligible": False,
        "source_refs": [],
        "payload": {"quadrature_decision": quadrature_mapping},
    }
    quadrature_wrapper["artifact_hash"] = canonical_json_hash(quadrature_wrapper)
    write_canonical_json(root / quadrature_ref, quadrature_wrapper)

    # Stage 4 真正执行三 phase route；Stage 5/6 读取该 route artifact 生成缩小派生
    # task。这样保留跨阶段 lineage，又不会为确定性验收重复训练相同三条 tiny phase。
    stage4 = execute(
        _config(
            config_path,
            "stage4.minimal_complete_loop",
            output_dir="runs/stage4-minimal-complete-loop",
            route_ref=route_ref,
            quadrature_ref=quadrature_ref,
            classification=True,
            training=True,
        )
    )
    route_result_ref = stage4.artifact_refs["training_route"]
    execute(
        _config(
            config_path,
            "stage5.checkpoint_analysis",
            output_dir="runs/stage5-checkpoint-analysis",
            input_refs=(route_result_ref,),
            classification=True,
        )
    )
    # Stage 6 不再从组合路线摘要旁路进入 importance_reuse。这里额外执行两条可比
    # 的真实路线，并严格串联 matrix -> evaluate -> compare -> reuse；中间任务只作为
    # 本机验收依赖，不扩大 FullFixturePipelineResult 原有的公开任务集合。
    direct_config = _config(
        config_path,
        "stage4.direct_supervised",
        output_dir="runs/stage6-direct-route",
        route_ref=direct_route_ref,
        quadrature_ref=quadrature_ref,
        classification=True,
        training=True,
    )
    direct_result = runtime.execute(direct_config)
    _load_published(root, direct_config, direct_result)
    finetune_config = _config(
        config_path,
        "stage4.finetune",
        output_dir="runs/stage6-finetune-route",
        route_ref=finetune_route_ref,
        quadrature_ref=quadrature_ref,
        classification=True,
        training=True,
    )
    finetune_result = runtime.execute(finetune_config)
    _load_published(root, finetune_config, finetune_result)

    matrix_config = _config(
        config_path,
        "stage6.route_matrix",
        output_dir="runs/stage6-route-matrix",
        input_refs=(
            *direct_result.artifact_refs.values(),
            *finetune_result.artifact_refs.values(),
        ),
        classification=True,
    )
    matrix_result = runtime.execute(matrix_config)
    _load_published(root, matrix_config, matrix_result)
    matrix_ref = matrix_result.artifact_refs["route_matrix"]

    evaluate_config = _config(
        config_path,
        "stage6.evaluate",
        output_dir="runs/stage6-evaluate",
        input_refs=tuple(matrix_result.artifact_refs.values()),
        matrix_ref=matrix_ref,
        classification=True,
    )
    evaluate_result = runtime.execute(evaluate_config)
    _load_published(root, evaluate_config, evaluate_result)
    compare_config = _config(
        config_path,
        "stage6.compare",
        output_dir="runs/stage6-compare",
        input_refs=tuple(evaluate_result.artifact_refs.values()),
        matrix_ref=matrix_ref,
        classification=True,
    )
    compare_result = runtime.execute(compare_config)
    _load_published(root, compare_config, compare_result)
    execute(
        _config(
            config_path,
            "stage6.importance_reuse",
            output_dir="runs/stage6-importance-reuse",
            input_refs=tuple(compare_result.artifact_refs.values()),
            matrix_ref=matrix_ref,
            classification=True,
        )
    )

    stage7 = execute(
        _config(
            config_path,
            "stage7.functional_pruning_validation",
            output_dir="runs/stage7-functional-pruning-validation",
            classification=True,
        )
    )
    stage8 = execute(
        _config(
            config_path,
            "stage8.ablation_and_robustness",
            output_dir="runs/stage8-ablation-and-robustness",
            classification=True,
        )
    )

    # Stage 9 ETL、表、图、报告与 replay 都明确消费同一组 Stage 7/8 权威 commit；
    # 每个 task 独立重建同一 frozen source，用哈希相等验证没有手工拼表或临时旁路。
    ingest_inputs = (
        stage7.artifact_refs["pruning_results"],
        stage8.artifact_refs["ablation_results"],
    )
    ingest = execute(
        _config(
            config_path,
            "stage9.ingest",
            output_dir="runs/stage9-ingest",
            input_refs=ingest_inputs,
            classification=True,
        )
    )
    for task_id in (
        "stage9.tables",
        "stage9.charts",
        "stage9.report",
        "stage9.analysis_visualization_reporting",
        "stage9.replay",
    ):
        execute(
            _config(
                config_path,
                task_id,
                output_dir=f"runs/{task_id.replace('.', '-')}",
                input_refs=ingest_inputs,
                classification=True,
            )
        )

    # 从 schema 对象读取科学内容哈希；不使用对象路径、mtime 或执行耗时。
    source = _find_schema(
        [payloads_by_task["stage9.ingest"]["frozen_source_table"]],
        "bound-frozen-source-table-v1",
    )
    table = _find_schema(
        [payloads_by_task["stage9.tables"]["table_artifacts"]],
        "analysis-table-artifact-v1",
    )
    chart = _find_schema(
        [payloads_by_task["stage9.charts"]["chart_artifacts"]],
        "analysis-chart-artifact-v1",
    )
    report = _find_schema(
        [payloads_by_task["stage9.report"]["analysis_report"]],
        "analysis-report-v1",
    )
    replay = _find_schema(
        [payloads_by_task["stage9.replay"]["hash_comparison"]],
        "stage9-hash-comparison-v1",
    )

    registry_hashes = _collect_declared_hashes(
        all_payloads,
        {
            "coordinate_registry_hash",
            "parameter_registry_hash",
            "registry_hash",
        },
    )
    seed_hashes = _collect_declared_hashes(
        all_payloads,
        {"seed_plan_hash", "seed_hash"},
    )
    if not registry_hashes or not seed_hashes:
        raise ValueError("FULL_FIXTURE_REGISTRY_OR_SEED_HASH_MISSING")

    pipeline = FullFixturePipelineResult(
        config_hashes={
            **{task_id: config.config_hash for task_id, config in configs.items()},
            **extra_config_hashes,
        },
        task_artifact_hashes=artifact_hashes,
        coordinate_registry_hashes=registry_hashes,
        seed_plan_hashes=seed_hashes,
        source_table_hash=str(source["artifact_hash"]),
        table_hash=str(table["artifact_hash"]),
        chart_hash=str(chart["artifact_hash"]),
        report_hash=str(report["report_hash"]),
        replay_hash=canonical_json_hash(dict(replay)),
    )
    write_canonical_json(root / "full-fixture-result.json", pipeline.to_dict())
    return pipeline


__all__ = ["FullFixturePipelineResult", "run_full_fixture_pipeline"]
