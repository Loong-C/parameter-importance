# Stage 9：分析、图表、报告与交付

Stage 9 把训练、重要性、路径、剪枝和消融产物转换为统一冻结源表，再生成统计、论文表图、报告和复现 bundle。派生层不读取临时日志、notebook 状态或手工数字。

<a id="stage9-ingest"></a>
## `stage9.ingest`

校验跨阶段 artifact schema、hash、formal eligibility 和 lineage，输出 `frozen_source_table`、`source_lineage_manifest`、`ingest_report`。同名字段单位或坐标合同不一致时拒绝摄取。

<a id="stage9-statistics"></a>
## `stage9.statistics`

从冻结源表计算 Bias、Variance、MSE、MAE、Pearson/Spearman、top-k、Gini、entropy、HHI、有效参数数、top-q mass、置信区间和损伤 AUC，输出 `statistics_table`、`confidence_intervals`、`undefined_metric_report`。

零总质量、零范数、常量向量、`P=1` 或统计量不唯一时必须返回 `defined=false + reason`，不得用 epsilon 伪造数值。

<a id="stage9-tables"></a>
## `stage9.tables`

由 table spec 和 source hash 确定性生成 Markdown/CSV/LaTeX `table_artifacts`。列定义、顺序、格式和缺失值规则属于 `table_specs`。

<a id="stage9-charts"></a>
## `stage9.charts`

由 chart spec 重建 heatmap、误差条、facet、轨迹和剪枝曲线，输出 `chart_specs`、`chart_artifacts`。渲染版本和源表 hash 必须进入产物。

<a id="stage9-report"></a>
## `stage9.report`

从冻结表图重建 `analysis_report`、`claim_evidence_index`。每个数值声明都必须可追溯到源表、统计定义和输入 artifact。

<a id="stage9-analysis-visualization-reporting"></a>
## `stage9.analysis_visualization_reporting`

执行 Stage 9 缩合流水线，输出 `frozen_source_table`、`analysis_report`、`chart_artifacts`、`reproduction_manifest`。它用于完整链路验收，不绕过各细分 task 的审核边界。

<a id="stage9-bundle"></a>
## `stage9.bundle`

生成 `reproduction_manifest`、`delivery_manifest`、`artifact_inventory`。bundle 只登记可提交的小型产物和外部大资产的受控路径/hash，不把模型、数据或 checkpoint 复制进 Git。

<a id="stage9-replay"></a>
## `stage9.replay`

从空输出目录按冻结配置重放，输出 `replay_report`、`hash_comparison`、`gate_summary`。配置、registry、seed、artifact、表、图和报告 hash 的差异必须逐项报告。

## 退出条件

所有正式数字只能从带哈希源表重建；人工可以解释结果，但不能编辑表图数字。服务器或正式源表缺失时 Stage 9 formal 保持 `BLOCKED/NOT_RUN`。

## 与 `task fixture-all` 的边界

`task fixture-all --workspace-root <empty-dir>` 会执行包含 Stage 9 ingest/table/chart/report/
replay 的代表性缩小链，并写出 `full-fixture-result.json`；两个全新工作根可比较其文件
hash 与内部 `result_hash`。它不是 Stage 9 formal replay，也不穷举 catalog 的每个 task，
不覆盖 offline HF、DDP 2/4、profiling 或正式冻结源表。其
`scope=local_fixture`、`formal_eligible=false` 永远不能作为本 Stage 的退出证据。
