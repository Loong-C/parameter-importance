# S1.11 报告、可视化与阶段退出 Gate

## 1. 目的

把分散的单元测试、解析 oracle、单卡、四卡、数值边界和恢复结果组织成一条可审阅证据链。最终报告必须同时满足人可读、机器可判定和失败可复现三个要求。

## 2. 前置 Gate

- S1.1–S1.10 的全部必需 Gate 已产生机器可读结果。
- 任一失败均已有最小复现证据包和明确状态。
- 报告生成只读取已冻结结果，不重新计算或覆盖原始真值。

## 3. 最终验证矩阵

最终矩阵至少包含以下检查，且每项绑定 measured、threshold、结果路径和 requirement ID：

1. 参数 registry 构建/保存/加载完全一致。
2. 显式 ordered pair、unordered pair 和流式 U 代数一致。
3. 所有 microbatch 梯度相同时 U 等于 `eta*g^2`。
4. `M=1` 与仅一个正权重单元明确失败。
5. `M=2` 的 U 与同一两半样本 double 逐坐标一致。
6. microbatch 排列不改变充分统计量和 U。
7. 构造的正负梯度 case 保留负 U。
8. 零均值纯噪声 smoke 满足预注册标准误 Gate。
9. 逐样本、microbatch 和完整 batch mean gradient 一致。
10. 有效 token 加权 mean gradient 与逐 token mean oracle 一致。
11. weighted microbatch U 与显式 weighted cross-microbatch ordered-pair oracle 一致；不与 token-level U 混用。
12. raw core/score 与双采样逐元素手算一致。
13. signed/positive/negative mass/absolute 恒等式逐步成立。
14. 参数幅值、data movement、data net movement 和 total net movement 与各自独立重算一致。
15. scale/unscale 后 gradient、充分统计量、estimator core 和公开 score 一致。
16. non-finite step 在所有 rank 一致 skip，step-local 暂存结果未提交。
17. clip factor 在全局聚合后计算，且 `U_clipped ≈ s_t * U_unclipped`。
18. AdamW data/weight-decay/total 位移分解一致。
19. 统计开启与关闭的训练轨迹及 GradScaler 状态一致。
20. Pythia-14M 固定状态计算前后参数、buffer、optimizer/scheduler 和 RNG 不变。
21. Pythia-14M 单卡真实链路通过。
22. 路由 A↔D 的 mean gradient/raw/update 逐张量一致。
23. 保持同一组 `M` 个统计单元的路由 B↔C↔D 在充分统计量/U/累计量上逐张量一致。
24. all-reduce 后所有 rank 的 loss numerator/count、元数据与 checksum 一致。
25. 连续运行与 fresh-process 恢复一致，并覆盖在一次受控 skip 前后保存/恢复的场景。
26. 常梯度和二次损失路径逐坐标真值与完备性 smoke 通过。
27. 任一错误资产、配置、registry 或 checkpoint fail-closed。
28. 任一失败 case 可从保存证据离线复算。

## 4. 执行步骤

### 4.1 汇总自动测试

1. 在本机与服务器 CPU 分别运行同一组纯公式、schema 和解析模型测试。
2. 运行 Tiny Transformer、单卡 CUDA、四卡 NCCL/DDP 和恢复测试组。
3. 记录每组收集数、通过数、失败数、错误数和跳过数。
4. 正式服务器 Gate 不允许未解释 skip。
5. 记录总运行时间与最慢测试。
6. 保存结构化测试结果和人可读摘要。
7. 核对测试绑定的 Git commit 与工作树状态。
8. 核对 fixture、配置和环境 hash。

### 4.2 生成误差表

1. 为每个比较计算最大绝对误差。
2. 计算尺度化最大误差。
3. 计算 normalized L2 误差。
4. 统计 NaN/Inf 数量。
5. 列出最差 20 个参数张量。
6. 对每个最差张量记录最差坐标。
7. 对接近零参考使用预注册绝对误差分支，并记录 `n_q`、`tau_0q`、`delta_q`；相对与 normalized-L2 判定标为不适用。
8. 同时提供全局摘要和逐张量明细。
9. 不用 Pearson/Spearman 替代数值相等 Gate。
10. 对 mean gradient、`S2/G2`、raw core、U core、最终 score 和 optimizer delta 分别应用预注册尺度；不能只比较乘学习率后的字段。

### 4.3 生成核验可视化

1. 绘制 full-batch 与 microbatch gradient 的 identity-line 散点图。
2. 绘制显式 U 与流式 U 的 identity-line 散点图。
3. 绘制正确加权、等权负对照与完整 batch 的误差图。
4. 分别绘制 A↔D 的 gradient/raw identity 图与 B↔D 的 U identity 图，并在标题中标明路由和统计单元边界。
5. 绘制 `layer/module × metric` 误差热力图。
6. 绘制 clip 前 norm、clip factor 与 clip 后 norm 对照图。
7. 绘制连续运行与恢复运行逐步误差曲线。
8. 绘制累计恒等式残差时间序列。
9. 绘制零均值纯噪声 raw/U 均值与标准误图。
10. 为每张图保存 CSV 源数据和图形配置摘要。
11. 对 clip 主 Gate 绘制 `U_clipped` 对 `s_t*U_unclipped` 的 identity 图；近零过滤后的比值图只作诊断。

### 4.4 形成阶段报告

1. 写明 Stage 1 只验证代码正确性，不作科学结论。
2. 写明主指标与 AdamW 实际更新的解释边界。
3. 写明当前服务器 GPU 健康和实际使用集合。
4. 按 provenance allowlist 写明模型、数据、环境、commit、配置和 registry 身份，不复制凭据或传输端点。
5. 按 Gate 顺序给出目标、方法、阈值、测量值和结果。
6. 列出全部失败、修复和重跑历史，不覆盖旧失败记录。
7. 列出大型产物路径、大小和 SHA-256。
8. 列出遗留风险与不在本阶段解决的问题。
9. 明确 Stage 2 和 Stage 3 是否被解锁。
10. 生成 Markdown 报告与机器可读 summary。

### 4.5 完成质量与同步审计

1. 检查所有 Markdown 链接和文件存在性。
2. 检查 JSON/CSV schema 与行数。
3. 检查图表可由源数据重新生成。
4. 检查 Git 差异中的秘密、大型文件和临时路径。
5. 运行 whitespace 与格式检查。
6. 更新中文工作日志，记录用户原有数学文档修正如何处理，并核对前序里程碑的 worklog/提交/推送/服务器快进记录。
7. 形成准确的最终收尾提交；它不能替代 S1.1–S1.10 的里程碑提交。
8. 推送 GitHub。
9. 用已验证 bundle 将服务器仓库快进到同一提交。
10. 同步并核对五份 `Agent/*.md`。
11. 精确清理本次临时 bundle。
12. 记录本机、GitHub、服务器 HEAD 与工作树状态。

## 5. 最终产出

### 5.1 Git 跟踪产物

- 参数重要性实现、配置与全部测试；
- Stage 1 数学契约、参数 registry schema 和结果 schema；
- 小型 FP64 oracle 与 fixture；
- Gate summary JSON/CSV；
- 单卡/四卡一致性摘要；
- checkpoint 恢复摘要；
- Markdown 阶段报告；
- PNG/SVG 图表与 CSV 源数据；
- 中文实施工作日志。

### 5.2 服务器专属产物

- Pythia-14M 指定 step 的完整逐 microbatch gradient；
- 参数级 raw/U/累计数组；
- 单卡和四卡完整调试目录；
- 连续/恢复 checkpoint 与逐步对照；
- 失败复现 bundle；
- 包含绝对路径、大小、SHA-256、commit、配置和验收状态的 manifest。

## 6. 最终报告的必需可视化

至少交付以下核验图；若某图无数据，必须把对应 Gate 标为未完成而不是省略：

- full batch vs microbatch gradient identity 图；
- pairwise vs streaming U identity 图；
- 等权/加权 reduction 误差对比图；
- 路由 A↔D 的 mean gradient/raw 与路由 B↔D 的 U 逐张量误差热力图；
- clip factor 单次应用验证图；
- continuous vs resume 逐步误差图；
- raw/U 纯噪声 smoke 图；
- 累计恒等式残差图。

## 7. `G1-EXIT` 通过标准

只有同时满足以下条件，Stage 1 才可标记为完成：

- `G1-ENTRY` 至 `G1-RESUME` 全部为 `pass`，没有 `blocked`、`failed` 或未解释 `skip`。
- 未解决失败项为零；所有历史失败证据、修复提交和重跑结果仍保留并可追溯。
- 最终验证矩阵的所有必需项都有 measured、threshold、pass/fail 和证据路径。
- 所有逐参数张量比较通过对应容差，没有 NaN/Inf。
- Pythia-14M 单卡与真实四卡 NCCL/DDP 均完成。
- signed U 的负值被保留，negative mass 为非负，absolute 已保存。
- raw 未裁剪、可选 raw-clipped、data movement 和 total movement 字段无语义混淆。
- 统计开启不改变训练轨迹，连续/恢复轨迹一致。
- 所有小型结果可从 Git 复现，所有大型结果可由 manifest 定位和验真。
- 本机、GitHub 与服务器处于同一提交；服务器工作树状态已说明。
- 本机与服务器五份 `Agent/*.md` 文件集合和 SHA-256 一致。
- 当前工作日志已记录实现、测试、失败、修复、产物和下一阶段入口。

任一条件不满足时，Stage 1 状态只能是“进行中”“失败待修复”或“阻塞”，不得用总体趋势、相关性或部分 GPU 结果替代 Gate。

## 8. 后续阶段交接

### 8.1 解锁 Stage 2

交接包必须冻结：

- 固定 checkpoint 和 registry hash；
- raw/double/U estimator 版本；
- loss reduction 与有效 token 权重；
- batch、microbatch 和独立采样 seed 空间；
- 参考 batch 数据范围；
- 运行时间、显存和结果 schema。

Stage 2 只研究 bias、variance、MSE、排序和成本；若需要修改 Stage 1 实现，先回退并重跑受影响 Gate。

### 8.2 解锁 Stage 3

交接包必须包含：

- 常梯度和二次损失路径 smoke；
- 固定 probe 损失与随机状态机制；
- 参数端点和 optimizer state 的无扰动保存/恢复；
- 逐坐标贡献与完备性残差接口。

Stage 3 再比较左/右端点、梯形、Simpson、Gauss-Legendre 与节点数；Stage 1 不预判哪种方法足够准确。
