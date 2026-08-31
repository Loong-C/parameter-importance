"""Stage 0--9 本机确定性 fixture 的薄编排入口。

本模块只把仓库中已经冻结的合同和纯 CPU 核心串成一条很小、但会真实执行
数学计算的流水线。它的目的不是产生论文结论，而是回答三个工程问题：配置、
seed 与参数坐标能否稳定绑定；Stage 2 的 reference/paired estimator 能否从同一
draw manifest 重放；Stage 3 的路径积分与 Stage 9 的报告能否仅由 hash 绑定输入
重建。

安全边界是刻意写死的：输入必须是 ``run_intent=local_fixture``，输出始终带
``scope=local_fixture`` 与 ``formal_eligible=false``；Stage 2/3 decision 保持
``UNFROZEN``/fixture-only，服务器证据统一记录为 ``server_unreachable``。模块
没有任何把 fixture 升级为 formal 的函数，也不会读取网络、服务器路径、模型或
数据资产。当前时间、绝对输出路径和性能计时都不进入内容哈希。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

from param_importance_nlp.analysis.metrics import (
    MetricResult,
    error_summary,
    gini,
    pearson,
    top_q_mass,
)
from param_importance_nlp.analysis.report import AnalysisReport, AnalysisReportBuilder, FrozenSourceTable
from param_importance_nlp.atomic import atomic_write_bytes, sha256_file
from param_importance_nlp.contracts.config import ResolvedConfig
from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    ensure_json_object,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.seed import SeedPlan
from param_importance_nlp.core.quadrature import (
    PathSpec,
    gauss_legendre_rule,
    integrate_path,
    midpoint_rule,
)
from param_importance_nlp.core.registry import ParameterRegistry
from param_importance_nlp.core.tensors import TensorMap
from param_importance_nlp.experiments.sampling import (
    STREAM_NAMES,
    PilotObservation,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    select_primary_pair,
)
from param_importance_nlp.experiments.stage2 import (
    PairedEstimatorResult,
    PairedEstimatorRunner,
    ReferenceRunner,
    Stage2FixtureStudy,
)
from param_importance_nlp.experiments.stage3 import build_fixture_quadrature_decision
from param_importance_nlp.providers.synthetic import SyntheticGradientProvider


_RESULT_SCHEMA = "local-fixture-result-v1"
_PIPELINE_VERSION = "stage0-9-cpu-math-v1"
_FORMAL_DIRECTORY_PATTERN = re.compile(r"^formal(?:$|[-_.])", re.IGNORECASE)


class LocalFixtureContractError(ValueError):
    """fixture 输入或输出越过本机/正式边界时抛出的机器可识别错误。"""


class _FixtureLinearModel(torch.nn.Module):
    """只用于登记稳定参数坐标的两输出线性模型。

    模型不执行训练，也不读取任何外部权重。参数值被显式覆盖，避免默认初始化
    消耗全局 Torch RNG；参数名称和 shape 则供 :class:`ParameterRegistry` 真实
    审计 optimizer membership、group ID、dtype 和运行布局。
    """

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=True)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.tensor(
                    [[0.50, -0.25, 0.75], [-0.40, 0.20, 0.60]],
                    dtype=torch.float32,
                )
            )
            self.linear.bias.copy_(torch.tensor([0.10, -0.20], dtype=torch.float32))


def _require_local_contract(config: ResolvedConfig) -> None:
    """拒绝 formal、联网、非 CPU 或分布式配置。

    这些约束不能由调用方通过命令行开关放宽。正式运行必须走会校验服务器 Gate
    和冻结 decision 的独立入口；否则一次本机单测成功可能被误写成正式证据。
    """

    identity = config.section("identity")
    runtime = config.section("runtime")
    distributed = config.section("distributed")
    if identity["run_intent"] != "local_fixture" or bool(identity["formal_eligible"]):
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_FORMAL_FORBIDDEN: 仅接受 run_intent=local_fixture 且 formal_eligible=false"
        )
    if runtime["offline"] is not True or runtime["device"] != "cpu":
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_CPU_OFFLINE_REQUIRED: fixture 必须 offline=true、device=cpu"
        )
    if distributed["backend"] != "local" or distributed["world_size"] != 1:
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_LOCAL_REDUCER_REQUIRED: fixture 只接受 backend=local、world_size=1"
        )


def _resolve_output_dir(config: ResolvedConfig, output_dir: Path) -> Path:
    """解析并授权 fixture 输出目录。

    ``runtime.output_root`` 是 resolved config 中唯一有权授予写入范围的逻辑根。
    相对路径统一以调用进程当前工作目录为基准；这与 CLI 在仓库根运行时的路径
    解释一致，也允许测试通过切换 cwd 使用隔离临时目录。授权同时比较词法路径
    与解析符号链接后的真实路径：

    * 词法目标必须是词法根本身或后代，不能靠 ``..``/绝对路径越界；
    * 根自身若已经是 symlink/junction，拒绝把配置中的逻辑根静默改向别处；
    * 根内任一 symlink 若把目标解析到根外，真实路径的 ``relative_to`` 会失败；
    * 任一路径分量以 ``formal`` 或 ``formal-``/``formal_``/``formal.`` 开头时拒绝。

    返回值是已经解析的绝对路径，调用方后续只使用该值创建目录和文件。
    """

    runtime_root_value = config.section("runtime")["output_root"]
    if not isinstance(runtime_root_value, str) or not runtime_root_value:
        raise LocalFixtureContractError("LOCAL_FIXTURE_OUTPUT_ROOT_INVALID")
    cwd = Path.cwd().resolve()
    logical_root = Path(runtime_root_value)
    lexical_root = (
        logical_root if logical_root.is_absolute() else cwd / logical_root
    ).absolute()
    resolved_root = lexical_root.resolve(strict=False)
    # ``Path.absolute`` 保留 symlink 分量；若存在的根链被重定向，resolve 后会不同。
    if lexical_root != resolved_root:
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_RUNTIME_ROOT_SYMLINK_FORBIDDEN: runtime.output_root 不能经 symlink 重定向"
        )

    lexical_target = (output_dir if output_dir.is_absolute() else cwd / output_dir).absolute()
    try:
        lexical_target.relative_to(lexical_root)
    except ValueError as error:
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_OUTPUT_OUTSIDE_ROOT: output_dir 必须位于 runtime.output_root 内"
        ) from error
    resolved_target = lexical_target.resolve(strict=False)
    try:
        relative_target = resolved_target.relative_to(resolved_root)
    except ValueError as error:
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_SYMLINK_ESCAPE: output_dir 经 symlink 解析后越过 runtime.output_root"
        ) from error

    if any(_FORMAL_DIRECTORY_PATTERN.match(part) for part in relative_target.parts):
        raise LocalFixtureContractError(
            "LOCAL_FIXTURE_FORMAL_OUTPUT_FORBIDDEN: fixture 不能写入 formal 结果目录"
        )
    return resolved_target


def _build_registry(config: ResolvedConfig) -> tuple[_FixtureLinearModel, ParameterRegistry]:
    """建立真实的 CPU ParameterRegistry，并冻结三个互相独立的身份。"""

    optimizer_config = config.section("optimizer")
    if optimizer_config["type"] != "sgd":
        raise LocalFixtureContractError("本机固定 fixture 只支持配置中的 SGD optimizer")
    if bool(optimizer_config["foreach"]) or bool(optimizer_config["fused"]):
        raise LocalFixtureContractError("本机固定 fixture 拒绝 foreach/fused optimizer")
    model = _FixtureLinearModel()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        momentum=float(optimizer_config["momentum"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        foreach=False,
    )
    return model, ParameterRegistry.from_model(model, optimizer)


def _tensor_vector(value: Mapping[str, object]) -> tuple[list[str], np.ndarray]:
    """按 canonical name/扁平索引把张量映射转为 FP64 比较向量。

    comparator 一律先转 CPU FP64。返回的 coordinate ID 只含名字和十二位索引，
    不含对象地址、设备指针或 mapping 插入顺序。
    """

    coordinate_ids: list[str] = []
    parts: list[np.ndarray] = []
    for name in sorted(value):
        item: object = value[name]
        if hasattr(item, "detach"):
            item = item.detach()  # type: ignore[union-attr]
        if hasattr(item, "cpu"):
            item = item.cpu()  # type: ignore[union-attr]
        if hasattr(item, "numpy"):
            item = item.numpy()  # type: ignore[union-attr]
        array = np.asarray(item, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(array)):
            raise LocalFixtureContractError(f"参数 {name!r} 的 fixture 结果包含 NaN/Inf")
        coordinate_ids.extend(f"{name}#{index:012d}" for index in range(array.size))
        parts.append(array)
    if not parts:
        raise LocalFixtureContractError("fixture 张量映射不能为空")
    return coordinate_ids, np.concatenate(parts)


def _vector_payload(value: Mapping[str, object]) -> dict[str, list[float]]:
    """把小型 fixture 张量转换为 canonical JSON 可接受的有限浮点列表。"""

    result: dict[str, list[float]] = {}
    for name in sorted(value):
        _ids, vector = _tensor_vector({name: value[name]})
        result[name] = [float(item) for item in vector]
    return result


def _run_stage2(
    *,
    config: ResolvedConfig,
    seed_plan: SeedPlan,
    registry: ParameterRegistry,
) -> tuple[dict[str, object], object, tuple[PairedEstimatorResult, ...]]:
    """执行固定 draw 的 reference、pilot selector 与两次 paired repetition。"""

    shapes = {record.canonical_name: record.shape for record in registry.eligible_records}
    provider = SyntheticGradientProvider.from_location_scale(
        parameter_shapes=shapes,
        sample_count=64,
        mean=0.25,
        noise_scale=0.50,
        seed=seed_plan.seed_for("importance_sampling"),
        fixed_state_id=f"fixture-state-{config.config_hash[:16]}",
    )
    universe = SamplingUniverse(
        universe_id=str(config.section("sampling")["universe_version"]),
        sample_ids=provider.sample_ids,
        metadata={"scope": "local_fixture", "formal_eligible": False},
    )
    sampling_plan = SamplingPlan(
        universe=universe,
        stream_seeds={name: seed_plan.seed_for(name) for name in STREAM_NAMES},
    )

    # A/B 各 32 个 draw、每 8 个 draw 一个 block；样本 ID 碰撞保留，但两个 stream
    # 的 draw ID 永不复用。ReferenceRunner 会再次审计 provider 状态未发生漂移。
    reference = ReferenceRunner(provider).run(
        reference_id="local-fixture-reference-v1",
        draws_a=sampling_plan.draws("reference_A", 32),
        draws_b=sampling_plan.draws("reference_B", 32),
        block_size=8,
    )

    # 只提供预注册网格中的 B=32/M=32 可运行观测。选择器仍会遍历完整冻结顺序，
    # 并把缺失候选记录为原因；这个 fixture 选择不能冻结正式 B/M/R。
    primary_pair = select_primary_pair(
        [
            PilotObservation(
                batch_size=32,
                microbatch_count=32,
                anchors_runnable=True,
                finite=True,
                aggregation_overhead_ratio=0.01,
                r_required=2,
                resource_within_budget=True,
            )
        ],
        r_max=2,
    )
    if primary_pair.batch_size is None or primary_pair.microbatch_count is None:
        raise LocalFixtureContractError("冻结的 synthetic pilot 未能选出 fixture B/M")

    study = Stage2FixtureStudy("local-fixture-stage2-v1")
    study.register_reference(reference)
    study.freeze_matrix(primary_pair)
    study.select_estimator(selected_estimator="u", repetitions=2)
    assert study.decision is not None

    paired_runner = PairedEstimatorRunner(provider)
    paired_results: list[PairedEstimatorResult] = []
    for repetition in range(2):
        start = repetition * primary_pair.batch_size
        mapping = RepetitionMapping.create(
            repetition_id=f"fixture-repetition-{repetition:02d}",
            draws=sampling_plan.draws(
                "confirmatory",
                primary_pair.batch_size,
                start=start,
            ),
            m_values=(2, 4, 8, 16, 32),
        )
        paired_results.append(paired_runner.run(mapping))
    study.complete(paired_results)

    stage2_payload: dict[str, object] = {
        "scope": "local_fixture",
        "formal_eligible": False,
        "fixture_status": study.state,
        "formal_state": "UNFROZEN",
        "formal_gate_id": "stage2.G2.7b",
        "formal_gate_status": "BLOCKED",
        "formal_block_reason": "server_unreachable",
        "sampling_plan_hash": sampling_plan.digest,
        "sampling_universe_hash": universe.digest,
        "provider_registry_hash": provider.registry_hash,
        "provider_state_hash": provider.state_digest(),
        "reference_hash": reference.digest,
        "reference_scope": reference.scope,
        "estimator_decision": study.decision.to_dict(),
        "result_hashes": [result.digest for result in paired_results],
        "study_hash": study.digest,
        "selected_fixture_batch_size": primary_pair.batch_size,
        "selected_fixture_microbatch_count": primary_pair.microbatch_count,
        "formal_B_M_R": {
            "B_primary": None,
            "M_primary": None,
            "R": None,
            "state": "UNFROZEN",
        },
    }
    return stage2_payload, reference, tuple(paired_results)


def _run_stage3(registry: ParameterRegistry) -> tuple[dict[str, object], object]:
    """对二次 probe loss 执行真实 FP64 路径积分与跨规则比较。"""

    pre_values: dict[str, torch.Tensor] = {}
    post_values: dict[str, torch.Tensor] = {}
    offset = 1
    for record in registry.eligible_records:
        values = torch.arange(offset, offset + record.numel, dtype=torch.float64).reshape(
            record.shape
        ) / 10.0
        pre_values[record.canonical_name] = values
        post_values[record.canonical_name] = values * 0.75
        offset += record.numel
    pre_state = TensorMap(pre_values, registry=registry, clone=True)
    post_state = TensorMap(post_values, registry=registry, clone=True)
    path = PathSpec(
        pre_state,
        post_state,
        path_id="local-fixture-full-update-linear-v1",
        probe_id="local-fixture-quadratic-probe-v1",
        loss_id="half-squared-l2-v1",
        accumulation_dtype=torch.float64,
    )

    def loss_callback(state: TensorMap) -> torch.Tensor:
        # L(theta)=1/2 sum(theta_k^2)，因此解析梯度就是 theta；线性路径上的梯度
        # 关于 alpha 为一次多项式，midpoint 与二点 Gauss-Legendre 都应精确。
        return sum(
            (0.5 * torch.square(tensor).sum() for tensor in state.values()),
            torch.zeros((), dtype=torch.float64),
        )

    def gradient_callback(_alpha: float, state: TensorMap) -> TensorMap:
        return state.clone()

    midpoint_result = integrate_path(
        path,
        midpoint_rule(),
        gradient_callback,
        loss_fn=loss_callback,
    )
    gauss_result = integrate_path(
        path,
        gauss_legendre_rule(2),
        gradient_callback,
        loss_fn=loss_callback,
    )
    _ids_midpoint, midpoint_vector = _tensor_vector(midpoint_result.signed)
    _ids_gauss, gauss_vector = _tensor_vector(gauss_result.signed)
    cross_rule_max_abs_error = float(np.max(np.abs(midpoint_vector - gauss_vector)))
    if cross_rule_max_abs_error > 1e-12:
        raise LocalFixtureContractError("解析路径 fixture 的跨规则结果不一致")

    decision = build_fixture_quadrature_decision(
        passing_rules_by_cost=("midpoint", "gauss_legendre_2"),
        fallback_rule="gauss_legendre_2",
    )
    result_payload = {
        "path_identity_hash": path.identity_hash,
        "selected_rule": midpoint_result.rule.name,
        "reference_rule": gauss_result.rule.name,
        "signed": _vector_payload(midpoint_result.signed),
        "absolute": _vector_payload(midpoint_result.absolute),
        "endpoint_loss_pre": midpoint_result.endpoint_loss_pre,
        "endpoint_loss_post": midpoint_result.endpoint_loss_post,
        "loss_drop": midpoint_result.loss_drop,
        "completeness_absolute_residual": midpoint_result.completeness_absolute_residual,
        "completeness_relative_residual": midpoint_result.completeness_relative_residual,
        "completeness_l1_scaled_residual": midpoint_result.completeness_l1_scaled_residual,
        "cross_rule_max_abs_error": cross_rule_max_abs_error,
    }
    stage3_payload: dict[str, object] = {
        "scope": "local_fixture",
        "formal_eligible": False,
        "fixture_status": decision.status,
        "formal_state": "UNFROZEN",
        "formal_gate_status": "BLOCKED",
        "formal_block_reason": "server_unreachable",
        "quadrature_decision": {
            "decision_id": decision.decision_id,
            "status": decision.status,
            "default_rule": decision.default_rule,
            "fallback_rule": decision.fallback_rule,
            "scope": decision.scope,
            "formal_eligible": decision.formal_eligible,
            "artifact_hash": decision.artifact_hash,
            "unresolved_fields": list(decision.unresolved_fields),
        },
        "formal_quadrature": {
            "default_rule": None,
            "fallback_rule": None,
            "probe_count": None,
            "node_budget": None,
            "thresholds": None,
            "state": "UNFROZEN",
        },
        "result": result_payload,
        "result_hash": canonical_json_hash(result_payload),
    }
    return stage3_payload, midpoint_result


def _build_analysis_report(
    *,
    config: ResolvedConfig,
    seed_plan: SeedPlan,
    registry: ParameterRegistry,
    reference: object,
    paired_results: Sequence[PairedEstimatorResult],
    path_result: object,
) -> tuple[AnalysisReport, FrozenSourceTable]:
    """从 Stage 2/3 数值构建不可手改的源表与结构化报告。"""

    reference_map = getattr(reference, "bias_reference")
    coordinate_ids, reference_vector = _tensor_vector(reference_map)
    m_value = 32
    estimate_vectors = np.stack(
        [_tensor_vector(result.u_by_m[m_value])[1] for result in paired_results]
    )
    raw_vectors = np.stack([_tensor_vector(result.raw)[1] for result in paired_results])
    path_ids, path_vector = _tensor_vector(getattr(path_result, "signed"))
    if path_ids != coordinate_ids:
        raise LocalFixtureContractError("Stage 2 与 Stage 3 的 canonical 坐标不一致")
    mean_estimate = np.mean(estimate_vectors, axis=0)
    completeness = getattr(path_result, "completeness_absolute_residual")
    if completeness is None or not math.isfinite(float(completeness)):
        raise LocalFixtureContractError("Stage 3 fixture 缺少有限完备性残差")
    # 保留 repetition 轴而不是只发布均值，否则 Bias/Variance/MSE 等报告数字无法
    # 仅凭冻结源表重建。reference/path 标量按坐标重复是显式的关系型展开。
    rows = [
        {
            "repetition_id": paired_results[repetition_index].unit_id,
            "coordinate_id": coordinate_id,
            "u_estimate": float(estimate_vectors[repetition_index, coordinate_index]),
            "raw_estimate": float(raw_vectors[repetition_index, coordinate_index]),
            "reference_u": float(reference_vector[coordinate_index]),
            "path_signed": float(path_vector[coordinate_index]),
            "path_completeness_absolute_residual": float(completeness),
        }
        for repetition_index in range(len(paired_results))
        for coordinate_index, coordinate_id in enumerate(coordinate_ids)
    ]
    source_table = FrozenSourceTable.from_rows(
        name="local_fixture_coordinate_results",
        schema_version="local-fixture-coordinate-table-v2",
        rows=rows,
    )

    builder = AnalysisReportBuilder(report_id=f"local-fixture-{config.config_hash[:16]}")
    builder.add_source(source_table)
    for name, result in error_summary(estimate_vectors, reference_vector).items():
        columns = (
            ("repetition_id", "coordinate_id", "u_estimate")
            if name == "variance"
            else (
                "repetition_id",
                "coordinate_id",
                "u_estimate",
                "reference_u",
            )
        )
        builder.add_metric(
            f"stage2_{name}",
            result,
            source=source_table,
            derivation_id=f"stage9.error_summary.{name}.v1",
            input_columns=columns,
        )
    grouped_columns = (
        "repetition_id",
        "coordinate_id",
        "u_estimate",
    )
    builder.add_metric(
        "stage2_pearson_u_reference",
        pearson(mean_estimate, reference_vector),
        source=source_table,
        derivation_id="stage9.grouped_mean_pearson.v1",
        input_columns=(*grouped_columns, "reference_u"),
    )
    builder.add_metric(
        "stage2_gini_absolute_u",
        gini(np.abs(mean_estimate)),
        source=source_table,
        derivation_id="stage9.grouped_mean_absolute_gini.v1",
        input_columns=grouped_columns,
    )
    builder.add_metric(
        "stage2_top_half_mass",
        top_q_mass(np.abs(mean_estimate), 0.5),
        source=source_table,
        derivation_id="stage9.grouped_mean_absolute_top_q_mass.v1",
        input_columns=grouped_columns,
    )
    builder.add_metric(
        "stage3_completeness_absolute_residual",
        MetricResult(True, float(completeness)),
        source=source_table,
        derivation_id="stage3.path_integral.completeness_absolute_residual.v1",
        input_columns=("path_completeness_absolute_residual",),
    )
    report = builder.build(
        metadata={
            "scope": "local_fixture",
            "formal_eligible": False,
            "pipeline_version": _PIPELINE_VERSION,
            "config_hash": config.config_hash,
            "seed_plan_hash": seed_plan.artifact_hash,
            "coordinate_registry_hash": registry.coordinate_registry_hash,
        }
    )
    return report, source_table


def _render_markdown(artifact: Mapping[str, object], report: AnalysisReport) -> str:
    """从结构化 artifact 生成不含时间与路径的确定性中文摘要。"""

    config = artifact["config"]
    registry = artifact["registry"]
    stage2 = artifact["stage2"]
    stage3 = artifact["stage3"]
    assert isinstance(config, Mapping)
    assert isinstance(registry, Mapping)
    assert isinstance(stage2, Mapping)
    assert isinstance(stage3, Mapping)
    lines = [
        "# Stage 0--9 本机确定性 fixture",
        "",
        "> 此产物仅验证本机合同与 CPU 数学流水线，不能作为 formal Gate 或训练结论。",
        "",
        "- scope：`local_fixture`",
        "- formal eligible：`false`",
        f"- config hash：`{config['config_hash']}`",
        f"- seed plan hash：`{artifact['seed_plan_hash']}`",
        f"- coordinate registry hash：`{registry['coordinate_registry_hash']}`",
        f"- pipeline result hash：`{artifact['pipeline_result_hash']}`",
        f"- artifact hash：`{artifact['artifact_hash']}`",
        f"- analysis report hash：`{report.report_hash}`",
        "",
        "## 决策边界",
        "",
        f"- Stage 2 fixture：`{stage2['fixture_status']}`；formal：`{stage2['formal_state']}`",
        f"- Stage 3 fixture：`{stage3['fixture_status']}`；formal：`{stage3['formal_state']}`",
        "- 正式 B/M/R、求积默认规则、probe 数、节点预算和阈值仍为 `UNFROZEN`。",
        "",
        "## 正式阻塞",
        "",
        "- `server_unreachable`：服务器 HEAD、Agent 文档哈希及服务器 Gate 均未读取。",
        "- 本机 local validation 的 `PASS` 不会把任何 formal Gate 改写为 `PASS`。",
        "",
        report.render_markdown(),
    ]
    return "\n".join(lines).rstrip() + "\n"


def run_local_fixture(
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """运行确定性 CPU fixture 并发布 canonical JSON 与 Markdown。

    Parameters
    ----------
    config_path:
        ``resolved-config-v1`` canonical JSON。文件必须严格 UTF-8、无 BOM、无
        重复键且逐字节 canonical；配置必须声明本机 fixture、CPU 与 offline。
    output_dir:
        输出目录。它必须是配置 ``runtime.output_root`` 本身或其后代；符号链接
        越界以及 ``formal-v1`` 等正式结果命名都会被拒绝。固定发布
        ``local-fixture-result.json``、``analysis-report.json`` 与
        ``local-fixture-report.md``，写入采用同文件系统原子替换。

    Returns
    -------
    dict[str, object]
        给 CLI 使用的运行摘要。摘要包含动态绝对路径，但这些路径不进入任何内容
        哈希；``artifact_hash``、``result_hash`` 和 ``report_hash`` 可在两次独立
        输出目录之间直接比较。

    Raises
    ------
    LocalFixtureContractError
        输入试图声明 formal、启用网络/GPU/分布式，写入 formal 目录，或数学
        fixture 违反已冻结不变量时抛出。
    """

    raw_config = ensure_json_object(load_canonical_json(config_path), field="resolved config")
    config = ResolvedConfig.from_mapping(raw_config)
    _require_local_contract(config)
    target_dir = _resolve_output_dir(config, Path(output_dir))
    seed_plan = SeedPlan.from_master_seed(
        int(config.section("identity")["master_seed"]),
        world_size=int(config.section("distributed")["world_size"]),
    )
    _model, registry = _build_registry(config)

    stage2, reference, paired_results = _run_stage2(
        config=config,
        seed_plan=seed_plan,
        registry=registry,
    )
    stage3, path_result = _run_stage3(registry)
    report, source_table = _build_analysis_report(
        config=config,
        seed_plan=seed_plan,
        registry=registry,
        reference=reference,
        paired_results=paired_results,
        path_result=path_result,
    )

    pipeline_result = {
        "stage2_reference_hash": stage2["reference_hash"],
        "stage2_result_hashes": stage2["result_hashes"],
        "stage2_decision_hash": stage2["estimator_decision"]["artifact_hash"],  # type: ignore[index]
        "stage3_result_hash": stage3["result_hash"],
        "stage3_decision_hash": stage3["quadrature_decision"]["artifact_hash"],  # type: ignore[index]
        "source_table_hash": source_table.content_hash,
        "report_hash": report.report_hash,
    }
    pipeline_result_hash = canonical_json_hash(pipeline_result)
    artifact_without_hash: dict[str, Any] = {
        "schema_version": _RESULT_SCHEMA,
        "pipeline_version": _PIPELINE_VERSION,
        "scope": "local_fixture",
        "run_intent": "local_fixture",
        "formal_eligible": False,
        "config": {
            "schema_version": config.section("identity")["schema_version"],
            "config_hash": config.config_hash,
            "full_hash": config.full_hash,
        },
        "seed_plan_hash": seed_plan.artifact_hash,
        "seed_algorithm": seed_plan.algorithm,
        "registry": {
            "coordinate_registry_hash": registry.coordinate_registry_hash,
            "optimizer_contract_hash": registry.optimizer_contract_hash,
            "runtime_layout_hash": registry.runtime_layout_hash,
            "eligible_parameter_names": list(registry.eligible_names),
        },
        "stage1": {
            "scope": "local_fixture",
            "formal_eligible": False,
            "math_profile": "unclipped_fp64_fixed_state_u",
            "unbiasedness_claim": "unbiased_fixed_state_under_declared_sampling_assumptions",
            "same_batch_clipped_score_present": False,
        },
        "stage2": stage2,
        "stage3": stage3,
        "stages4_to_9": {
            "scope": "local_fixture",
            "formal_eligible": False,
            "local_validation_status": "NOT_RUN",
            "stage9_report_fixture_status": "PASS",
            "formal_status": "BLOCKED",
            "formal_block_reason": "server_unreachable",
            "no_training_conclusions": True,
        },
        "analysis": {
            "source_table_hash": source_table.content_hash,
            "report_hash": report.report_hash,
            "hand_edited_numbers_allowed": False,
        },
        "pipeline_result_hash": pipeline_result_hash,
        "local_validation": {
            "local_validation_status": "PASS",
            "scope": "local_fixture",
            "checks": [
                "strict_config_and_seed",
                "parameter_registry",
                "stage2_reference_and_paired_estimators",
                "stage3_path_integral",
                "stage9_hash_bound_report",
            ],
        },
        "formal_readiness": {
            "gate_status": "BLOCKED",
            "reason": "server_unreachable",
            "gate_records": [
                {"gate_id": "stage0.G1", "gate_status": "BLOCKED"},
                {"gate_id": "stage1.G1-CONTRACT", "gate_status": "BLOCKED"},
                {"gate_id": "stage2.G2.7b", "gate_status": "BLOCKED"},
            ],
            "blocked_evidence": [
                "server_head",
                "server_agent_document_hashes",
                "server_formal_gates",
                "real_model_and_data_assets",
                "gpu_or_distributed_formal_run",
            ],
        },
    }
    artifact_hash = canonical_json_hash(artifact_without_hash)
    artifact = artifact_without_hash | {"artifact_hash": artifact_hash}

    target_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = target_dir / "local-fixture-result.json"
    report_path = target_dir / "analysis-report.json"
    markdown_path = target_dir / "local-fixture-report.md"
    write_canonical_json(artifact_path, artifact)
    write_canonical_json(report_path, report.to_dict())
    atomic_write_bytes(markdown_path, _render_markdown(artifact, report).encode("utf-8"))

    return {
        "scope": "local_fixture",
        "formal_eligible": False,
        "config_hash": config.config_hash,
        "seed_plan_hash": seed_plan.artifact_hash,
        "registry_hash": registry.coordinate_registry_hash,
        "result_hash": pipeline_result_hash,
        "artifact_hash": artifact_hash,
        "report_hash": report.report_hash,
        "artifact_file_hash": sha256_file(artifact_path),
        "report_file_hash": sha256_file(report_path),
        "markdown_file_hash": sha256_file(markdown_path),
        "artifact_path": str(artifact_path.resolve()),
        "report_path": str(report_path.resolve()),
        "markdown_path": str(markdown_path.resolve()),
    }


__all__ = ["LocalFixtureContractError", "run_local_fixture"]
