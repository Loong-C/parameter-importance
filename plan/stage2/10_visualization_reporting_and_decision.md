# S2.10 可视化、报告与主估计器决策

## 1. 子任务目的

把 reference、偏差、方差、MSE、排序和成本组织成可审阅的证据链，并依据预注册决策树回答“U-statistic 是否应作为后续主估计器”。图表必须绑定机器可读源表和 run IDs，不能只提供无法重建的图片。

## 2. 前置条件

- G2.6 统计有效性和 G2.7a 工程成本 Gate 通过；若仅有用户批准的单卡 provisional 证据，本任务只能生成 provisional 报告，不能通过 G2.7b；
- `quality_gates.json`、`hypothesis_decisions.json`、统计长表和 Pareto 表已封存；
- 所有图表样式、主/补充图分类和比较范围已在预注册中列出。

## 3. 实施步骤

### 3.1 建立图表源数据层

1. 从封存统计长表生成每张图的专用 tidy source table。
2. 每行保留 model、checkpoint、B、M、method、scope、repetition/aggregate、效应、区间和 run ID。
3. 为每张图记录筛选条件、聚合函数、排序规则和脚本版本。
4. 对源表做行数、唯一键、缺失值和有限性检查。
5. 计算源表 SHA-256，并在图的 sidecar metadata 中引用。
6. 主图不直接读取服务器原始张量，避免图表脚本隐式重算统计。

### 3.2 绘制参考质量图

1. 绘制 reference normalized L1 difference 随 \(B_{\mathrm{ref}}\) 的收敛曲线。
2. 绘制 reference Pearson、Spearman 和 top-k overlap 随 \(B_{\mathrm{ref}}\) 的曲线。
3. 绘制无偏 `bias_ref`、`cross_ref`、低方差 `rank_ref` 和 A/B 诊断的差异摘要，并标出有限均值平方的预期正偏。
4. 在图中标出 G2.3 阈值、sizing stream 冻结的 \(B_{\mathrm{ref}}\) 和最终 A/B 全长节点；不得把 A/B 上“刚好通过”的前缀标为主 reference。
5. 按模型和训练阶段分面，不把不同 checkpoint 的收敛混成一条线。

### 3.3 绘制 raw 过估计校准图

1. 绘制观测 raw bias 对 \(\widehat\sigma^2/B\) 的散点/hexbin，并加入 `y=x`。
2. 显示校准斜率、截距和 repetition/checkpoint 级区间。
3. 绘制 signed bias 对 `1/B` 的曲线，叠加理论趋势。
4. 按 SNR、layer/module 和训练阶段分面。
5. 对 signed 轴使用线性或对称尺度，不用普通对数轴隐藏负值。
6. 对高密度参数图使用抽样或 hexbin，但源表保留总体聚合，不能只展示挑选坐标。

### 3.4 绘制 Bias、Variance、MSE 主比较

1. 绘制三种 estimator 的 signed Bias 区间图。
2. 绘制 absolute Bias、Variance、MSE 和 MAE 对 B 的对数横轴曲线。
3. 在同图或配套图中显示 U/double 的配对比值与非劣边界。
4. 固定 B 绘制 U 的 Bias/Variance/MSE 对 M 曲线。
5. 显示 M=2 与 double 的一致位置。
6. 按 model × checkpoint 分面并保持同类图轴范围可比。
7. 单独显示 repetition 均值和单次 estimator 指标，防止混淆系统偏差与单次可用性。

### 3.5 绘制排序与稳定性图

1. 绘制 Pearson、Spearman 随 B/M 的曲线和区间。
2. 绘制 Overlap@K/Jaccard 随 top 比例和资源预算的曲线。
3. 绘制 U/double 配对排序差，并标出非劣边界。
4. 绘制 parameter、layer、module 的相邻 M 稳定性热图。
5. 将旧 signed 稳定性失败阈值仅作为历史参考标记，不与新 Gate 混用。
6. 对并列 top-k 边界在图注中说明 tie policy。

### 3.6 绘制 signed U 负值诊断

1. 绘制负坐标比例对 B/M 的曲线。
2. 绘制负质量和净质量/绝对质量比。
3. 绘制负值比例对参考 SNR 或贡献强度的分箱图。
4. 叠加高精度诊断中数值误差地板。
5. 明确图注：有限样本 U 为负不是自动失败；任何 positive-only 图仅放探索性附录并标为有偏派生量。

### 3.7 绘制工程成本与 Pareto 图

1. 绘制 wall time–MSE Pareto 图。
2. 绘制峰值显存–MSE 和 token cost–ranking quality 图。
3. 分离共享科学 runner、方法独立 anchor、在线训练增量、2B double 和 reference 摊销成本。
4. 用不同形状区分实测与 Stage 4/5 外推。
5. 标记被支配配置和推荐 B/M 区域。
6. 只使用通过 I/O/GPU 可比性检查的成本 wave。

### 3.8 生成主表与补充表

1. 主表汇总每个 model/checkpoint 的 raw bias 校准、U/double 等价和方法决策。
2. 主效率表汇总等预算 Bias、Variance、MSE、Spearman、Overlap、时间和显存。
3. Gate 表列出每项阈值、效应、区间、状态和证据路径。
4. 补充表列出所有 B/M、SNR 分箱、layer/module、失败单元和 reference 敏感性。
5. 所有表同时提供 CSV/Parquet 源数据和 Markdown 可读版本。

### 3.9 撰写阶段报告

1. 先说明固定状态 \(\mu^2\) 目标和本阶段不包含的路径/AdamW 边界。
2. 说明当前硬件、资产、checkpoint、采样框和 reference 证据。
3. 按 H1–H6 逐项报告效应、区间和支持状态。
4. 报告 raw 过估计的方向、尺度、batch 规律和不可分辨条件。
5. 报告 U/double 的偏差等价、MSE、排序和负值解释。
6. 报告成本、四卡语义和 Stage 4/5 容量外推。
7. 单独列出旧证据与新结果的差异，禁止把历史 positive-only 结论并入无偏主结果。
8. 列出局限性：有限 reference、结论条件于冻结经验分布 \(\mathcal F\)、全 Pile 外推边界、checkpoint 范围、模型/数据版本差异和硬件状态。
9. 最后引用机器决策 JSON，不手工改写 Gate 状态。

### 3.10 应用主估计器决策树

1. 先检查所有质量 Gate；任一关键 Gate 失败则决策为 `blocked`。
2. 从 `confirmatory_family_decisions` 读取唯一 `B_primary`、`M_primary` 和六个主 cells；不得在本任务重新挑选 B/M/scope/top-q。
3. 候选只有在全部主 cells 的 bias endpoints 通过 intersection-union 后才具有 `bias_qualified`。
4. 若两者均不 qualified，决策为 `return_to_stage1_or_reference`，禁止选择 raw。
5. 若只有 double qualified 且满足绝对在线资源预算，选择 `double_primary`；否则阻断。
6. 若只有 U qualified，先复核 M=2/double 的理论不变量。若 double 因功效不足而非实现错误未通过，可标记 `u_provisional_revalidate_double`；不变量或 reference 异常未关闭前不得进入后续阶段。
7. 若两者均 qualified，检查 U 相对 double 在全部主 cells 的校正 NMSE、Spearman、Overlap@`1%` 非劣，以及方法独立 `online_training_incremental_cost` 的 `1.25` 边界。
8. 上述项目全部非劣时唯一选择 `u_primary_double_calibration`，不再要求额外显著优效；任一统计非劣失败则选择满足绝对资源预算的 double。
9. U 仅成本超界时选择 `double_primary`；如确有工程价值，另建独立 `u_reconfigure_and_revalidate` 预注册轮次，保留原确认结果，并预先控制跨已尝试 B/M 的多重性/序贯选择；新结果不能静默替换本轮。
10. 记录推荐 B/M、double 校准覆盖率、适用条件和进入后续阶段的必补项。

### 3.11 执行图表与报告 QA

1. 从封存输入向一个全新的 `$DATA_ROOT/results/stage2/derived/<rebuild-analysis-id>` 一键重建全部主表和主图；不得清空或复用既有 analysis 目录。
2. 比较重建文件哈希或数值 sidecar 与已发布版本。
3. 检查每张图的单位、轴尺度、误差条、样本数和图例。
4. 检查颜色和线型在不同图中对 estimator 含义一致。
5. 检查无截断坐标、隐藏失败条件或跨模型不公平轴范围。
6. 检查 PNG 可读性和 SVG 文本/元素完整性。
7. 由独立审阅者从随机选取的三张图追溯到源表、统计表和 raw wave。
8. 大型 Parquet、逐参数图源和完整 figure data 留在服务器；进入 Git 的摘要/图片逐文件通过体积和秘密检查，并以 manifest 指向大型源数据。

## 4. 产出

- Reference convergence、raw bias 校准、Bias/Variance/MSE、M 稳定性、排序、负值和 Pareto 主图；
- 每张图的 CSV/Parquet source table 和 sidecar metadata；
- 主表、完整补充表和 Gate 表；
- Stage 2 中文阶段报告；
- `estimator_decision.json` 与可读决策说明；
- 图表/报告重建与 QA 报告。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.7b 方法决策 Gate**：

- 所有主图和主表有唯一源表、脚本版本和 run IDs；
- 图表没有丢弃负值、失败单元或不利条件；
- H1–H6、质量 Gate 和方法 Gate 状态与机器 JSON 完全一致；
- 至少一种无偏候选通过偏差等价性；
- U/double 的校正 NMSE、排序和成本决策使用冻结阈值；
- 六个主 cells 的 intersection-union、唯一主 B/M 和 tie-breaker 可由机器决策表重建；
- 主估计器、B/M、校准策略和适用边界明确；
- 报告明确区分局部 \(\mu^2\)、路径贡献和 actual AdamW 更新；
- 全部主图/表可从封存统计结果重建并通过 QA。

## 6. 后续依赖

- S2.11 根据 `estimator_decision.json` 和未决项决定是否进入 Stage 3/4。
- 后续阶段不得只引用报告文字，必须引用决策文件和其输入 manifests。
