# Stage 8：消融与稳健性

Stage 8 用冻结的单因素矩阵验证重要性方法对 batch/microbatch、积分节点、正负视图、归一化、checkpoint 频率、模型规模、seed 和数据规模的敏感性。

## 声明式矩阵与恢复

先撰写不含顶层 hash 的 `ablation-matrix-source-v1`，其中只允许
`matrix_id/base_config/factors/base_seed/seed_namespace/scope`。运行：

```powershell
param-importance artifact ablation-matrix-build `
  --spec <ablation-source.yaml> `
  --output <declaration.json> `
  --compiled-output <matrix.json>
```

builder 自动生成 baseline 与每个单因素 cell、独立 seed 和 config hash；
`stage8.freeze` 仍需重新验证声明和单因素约束。execute 只能消费冻结 matrix，按同一
output dir 下的 cell commit 幂等恢复；新 v2 override 必须显式填写
`recovery.resume_ref` 后使用 `task resume`。reduce/recommend/report 只从完整 canonical
cell 源表重建，不能用 resume 跳过缺失 cell。当前 execute runner 按 output dir 发现
commit，`resume_ref` 尚不能精确选择单个 cell；此外 builder 的 `scope=formal` 只声明
用途，不读取 formal execution evidence，也不替代 preflight 或独立 Gate。

<a id="stage8-freeze"></a>
## `stage8.freeze`

编译 `ablation_matrix`，验证每个子 cell 相对父 cell 只改变一个注册因子，发布 `single_factor_validation` 和 `matrix_freeze`。冻结后执行器不得增删 cell 或修改因子值。

<a id="stage8-execute"></a>
## `stage8.execute`

严格消费冻结矩阵，逐 cell 执行并发布 `ablation_cell_results`、`cell_lineage_manifest`。支持父结果复用、逐 cell commit、恢复和受控重试。

<a id="stage8-ablation-and-robustness"></a>
## `stage8.ablation_and_robustness`

组合矩阵、执行结果、稳健性分析和报告，输出 `ablation_matrix`、`ablation_results`、`robustness_report`、`stage_report`。本机只运行缩小矩阵。

<a id="stage8-reduce"></a>
## `stage8.reduce`

从完整 cell 源表生成 `ablation_summary_table`、`robustness_table`、`quality_gates`。因子、父子 lineage、seed 或预算不一致时拒绝归约。

<a id="stage8-recommend"></a>
## `stage8.recommend`

基于冻结统计产物生成 `configuration_recommendation`、`applicability_report`、`limitation_table`。推荐必须区分证据覆盖范围和未运行组合，不能外推到缺失模型/任务。

<a id="stage8-report"></a>
## `stage8.report`

从 canonical source 重建 `stage_report`、`chart_artifacts`、`gate_summary`；人工仅撰写科学解释，不手工修改结果数字。
