# S2.11 交付、工作日志与进入后续阶段

## 1. 子任务目的

完成 Stage 2 的产物封存、测试重放、工作日志、Git/多端同步和下游交接，使任何新会话都能判断使用了什么代码、环境、模型、数据、样本、reference、阈值和结果。该任务不要求把服务器大型张量复制到本机或 GitHub。

## 2. 前置条件

- G2.0、G2.1、G2.2、G2.3、G2.4a、G2.4b、G2.5、G2.6、G2.7a 和 G2.7b 均有机器状态；
- 报告、图表、决策文件和 raw/derived manifests 已完成；
- 当前共享工作树的所有用户/并发修改已识别；
- 大型结果位于 `$DATA_ROOT` 规定目录，小型 Git 产物不含秘密或大型张量。

## 3. 实施步骤

### 3.1 整理服务器产物层级

1. 将原始 repetition/wave 结果保留在 `$DATA_ROOT/results/stage2/raw/<run-id>`，以 sealed manifest/status marker 形成逻辑版本；不递归 `chmod`。
2. 将 reference 大型数组放在 `$DATA_ROOT/results/stage2/reference/<reference-id>`。
3. 将派生结果放在 `$DATA_ROOT/results/stage2/derived/<analysis-id>`，每次重建/探索使用新 ID，不清空旧目录。
4. 将运行日志和中间可恢复状态保留在 `$DATA_ROOT/runs/stage2/<run-id>`。
5. 将服务器生成的完整报告副本保留在 `$DATA_ROOT/reports/stage2/<report-id>`。
6. 将环境、模型、数据、采样、raw-results、derived-results 和 decision manifests 集中索引。
7. 为每个服务器专属产物记录绝对路径、生成时间、大小、SHA-256、schema、Git commit 和验收状态。
8. 不把数据、模型、逐 repetition 全参数数组或缓存移入 Git 仓库。

### 3.2 整理 Git 交付物

1. 保存所有现行源码、测试、配置 schema 和正式 Stage 2 configs。
2. 保存预注册、amendment、冻结矩阵和小型 manifests。
3. 保存统计长表的可审阅摘要、图表源数据、主/补充表和主图。
4. 保存阶段报告、Gate/假设/决策 JSON 和重建说明。
5. 保存不含秘密的环境摘要和服务器大型产物索引。
6. 更新项目 README 或阶段索引，使 Stage 2 入口可发现。
7. 检查每个文件体积和内容，避免提交模型、原始大张量、日志秘密或临时下载 URL。

### 3.3 执行独立重放

1. 在新的 `$DATA_ROOT/results/stage2/derived/<replay-id>` 加载冻结 config 和 manifests；不得清空、覆盖或复用已有输出目录。
2. 重放一个人工分布用例。
3. 重放一个 14M reference 小前缀。
4. 重放一个 14M B/M repetition，并核对 estimator/指标哈希。
5. 重放一个 31M confirmatory repetition。
6. 从封存统计表重建全部主表和主图。
7. 验证重放不需要网络，也不读取 shard 5 `.part` 或旧历史对象。
8. 生成 replay report，列出命令入口可适度概括，但保留影响复现的配置、seed 和路径。

### 3.4 执行最终测试与审计

1. 运行本机纯公式/schema 测试。
2. 运行服务器 CPU 测试。
3. 运行服务器单 GPU 固定状态和 estimator 集成测试。
4. 在健康四卡可用时运行 DDP/no_sync 和成本 anchor 测试。
5. 运行配置、manifest、结果 schema、链接和报告重建检查。
6. 运行 diff/check，确认没有编码、尾随空白、坏链接和超大 Git 文件。
7. 生成测试报告，记录用例数、失败、跳过、环境、提交和证据路径。

### 3.5 更新中文工作日志

1. 在目标/口径冻结时记录预注册阶段。
2. 在环境、资产、reference、pilot、14M、31M、分析和决策边界分别追加日志。
3. 对失败、中止、重试和 amendment 保留原事实与根因，不覆盖旧记录。
4. 记录 Git commit、配置、seed、模型/data revision、reference、结果目录和 GPU UUID。
5. 记录服务器大型产物路径、大小、哈希和验收状态。
6. 记录用户/并发已有修改如何审查和一并处理；若尚在进行或不可验证，不强行纳入提交。
7. 不记录密码、令牌、Cookie、签名 URL 或 SSH 秘密。

### 3.6 审查完整工作树

1. 查看全部已跟踪和未跟踪变化，不只看 Stage 2 目录。
2. 区分已完成、可验证的用户/并发成果与仍在进行的文件。
3. 对所有拟提交文件运行内容、测试、秘密和体积审查。
4. 若并发任务仍在写同一工作树，等待其稳定或与用户协调，不覆盖、不隐藏、不还原。
5. 只有完成且可验证的阶段形成提交；无法审查的并发内容保留并明确报告，不能擅自删除。

### 3.7 提交、推送与服务器同步

1. 在提交前更新对应工作日志和最终 manifest。
2. 暂存所有已审查、适合 Git 且属于当前稳定阶段的变化。
3. 检查 staged diff、空白、秘密、大文件和生成物边界。
4. 形成准确说明 Stage 2 结果的提交。
5. 获取远端状态，确认无非快进或意外分叉。
6. 推送目标分支；较大变更按 Git 规范创建 PR 并完成审阅。
7. 创建只包含目标提交的精确 bundle，上传为服务器 `$DATA_ROOT/tmp/repo-sync-<short-commit>.bundle`，记录本机临时路径、服务器路径和 SHA-256。
8. 在服务器目标仓库只执行 fetch 与 `--ff-only` 快进并复核 HEAD/工作树；不得 reset、强推或覆盖并发修改。
9. 单独同步 `Agent/` 规定的五文件，并逐文件核对 SHA-256。
10. 记录本机、GitHub、服务器 branch/HEAD 和 Agent hashes。

### 3.8 清理精确临时项

1. 先列出本阶段创建的精确 staging、bundle 和临时文件路径。
2. 确认最终产物已发布、哈希通过并可重放。
3. 只删除已确认的本机 bundle 精确文件和服务器 `$DATA_ROOT/tmp/repo-sync-<short-commit>.bundle` 等精确临时路径，不使用宽泛通配符或递归清理。
4. 不触碰 Pile `.part`、锁、既有下载任务、其他阶段输出或用户文件。
5. 在工作日志记录删除对象及是否可恢复。

### 3.9 形成下游交接包

1. 保存推荐 estimator、B/M、精度、成本和校准覆盖率。
2. 保存适用条件：独立统计单元、loss reduction、加权、AMP/DDP、负值解释和固定状态边界。
3. 保存禁止事项：positive clamp、raw 替代无偏方法、probe 命名混用和随机 batch 裁剪因子。
4. 保存 Stage 3 所需的 estimator 参考和 Stage 4 在线实现建议。
5. 保存所有未决硬件、资产、统计和容量风险。
6. 由新会话仅依靠仓库和大盘 manifests 完成一次交接核验。

## 4. 阶段产出总表

### 4.1 代码与测试

- 从 Stage 1 继承并在 G2.1 复验的固定状态 gradient provider；
- 从 Stage 1 继承并复验的 raw、double、等权/加权 U；
- Stage 2 新增的 sampling/reference 编排、paired runner、streaming reducer、profiler 和 report 适配层；
- 纯公式、CPU、单 GPU、四 GPU、恢复和重放测试。

### 4.2 配置与 provenance

- 预注册及 amendments；
- model/checkpoint/data/sample/reference manifests；
- 正式矩阵、seed 树、环境、参数注册表和 run lineage。

### 4.3 结果与报告

- raw/derived manifests 和小型汇总；
- Bias、Variance、MSE、排序、负值和成本表；
- 全部主图、补充图和图表源数据；
- 质量 Gate、假设结论、estimator decision 和阶段报告。

### 4.4 运维证据

- 工作日志；
- GPU/存储/I/O 状态；
- 大型产物路径、大小、哈希和 schema；
- 三端 Git 与 Agent hash 同步记录；
- 精确临时项清理记录。

## 5. 核验标准：Stage 2 最终退出 Gate

只有以下条件全部满足，才能通过 **G2.8 交付同步 Gate** 并将 Stage 2 标记为完成：

1. G2.0、G2.1、G2.2、G2.3、G2.4a、G2.4b、G2.5、G2.6、G2.7a 和 G2.7b 已全部通过，且本节列出的 G2.8 交付证据完整；不得用 G2.8 自身作为其前置条件。
2. raw 过估计被支持、被不支持或因高 SNR/参考精度不足而判为证据不足；三种状态都必须有预注册证据，不能只写预期结论。
3. U 与双采样的 Bias、Variance、MSE、MAE、排序、top-k 和成本全部完成公平比较。
4. 至少一种无偏候选在全部六个主 cells 通过偏差等价性；两者都失败或有主 cell 证据不足时，Stage 2 只能标为阻断/证据不足。
5. `estimator_decision.json` 给出唯一、已通过 G2.7b 的主分支；`revalidate` 分支表示计划未退出，不能同时标记 Stage 2 complete。
6. 结果覆盖两个模型和每个模型至少三个训练阶段。
7. 主图/表可从封存结果重建，独立重放通过。
8. 本机、GitHub 和服务器处于同一提交；`Agent/` 五文件两端哈希一致。
9. 大型产物只在大盘，Git 中有完整可追溯索引。
10. 健康四卡的数值与在线成本 anchor 已完成；只有单卡 provisional 证据时最终状态必须是 blocked/provisional。

## 6. 对后续阶段的 Gate

### 6.1 进入 Stage 3

Stage 3 可在以下条件下进入：

- reference 与固定状态 estimator 证据有效；
- 至少 double 或 U 通过偏差等价性；
- 主估计器决策与适用边界已冻结；
- Stage 3 明确研究路径积分数值误差，不把本阶段局部无偏性自动外推到随机路径。

### 6.2 进入 Stage 4

除上述条件外，还要求：

- 在线训练语义下的 U 或 double 实现通过单卡/四卡、AMP、裁剪和恢复 Gate；
- 推荐 B/M 的时间、显存和 checkpoint 容量满足 160M 闭环预算；
- 若 U 为主方法，双采样校准频率已预注册；若 double 为主方法，额外样本/梯度预算已纳入训练计划；
- 当前 GPU 健康和四卡候选重新验收通过。

## 7. 阻断与恢复规则

- 两种无偏方法均失败：返回 Stage 1 或 reference/independence 诊断，不回退 raw。
- 只有成本失败：本轮按决策树选择已合格的 double 或保持 blocked；新 B/M 只能进入独立预注册轮次，并预先处理跨配置选择，不能覆盖本轮结果。
- 只有 31M 资产/网络阻塞：保留 14M 结果为阶段中间成果，不能宣称跨规模完成。
- 四卡仍不可用：可以保留单卡统计成果，但不得进入依赖四卡的正式训练。
- 同步或并发工作树未稳定：保留文件和日志，暂停提交/推送，不用破坏性 Git 操作制造干净状态。
