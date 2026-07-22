# 第 2 阶段：过估计现象与无偏估计器验证计划

## 1. 文档定位

本目录把 [`general_plan.md`](../general_plan.md) 中的第 2 阶段展开为可执行、可核验、可追溯的实施计划。数学口径以 [`docs/mathematics.md`](../../docs/mathematics.md) 为准，运维和同步边界以 `Agent/` 下五份现行文档为准。

这里描述的是未来要完成的工作，不代表 Stage 0、Stage 1 或本阶段实验已经通过。所有“当前状态”都是 2026-07-18 的只读快照；执行任何子任务时必须重新采集证据，不能把本计划中的描述直接当作验收结果。

用户示例中的 `plan/stage1/` 与本次目标阶段编号不一致，因此本计划按实际阶段建立在 `plan/stage2/` 下。

## 2. 阶段目标与研究边界

本阶段只研究固定参数状态下的局部梯度空间目标：

\[
C_k^\star=\eta_{\mathrm{eval}}\mu_k^2,
\qquad
\mu_k=\mathbb E_{z\sim\mathcal F}[g_k(\Theta;z)].
\]

其中 \(\mathcal F\) 是由已验收 packed-sequence 索引构成、在 G2.2 冻结的经验分布。Reference 只是对 \(C^\star\) 的有限样本估计，不能与理论目标共用名称。偏差/无偏性主判据使用独立 reference block 的去对角 U 参考；合并均值平方只作为低方差排序参考。

其中模型 checkpoint \(\Theta\) 在一次实验中完全固定，不执行 optimizer step，也不改变 optimizer、scheduler、模型/全局 RNG 或数据游标状态。独立 sampling generator 按冻结 manifest 推进，但必须保存起止状态并可重放。主实验令 \(\eta_{\mathrm{eval}}=1\)，直接验证 \(\mu_k^2\)；如需按某个名义学习率展示，只能在主结果之后做统一线性缩放，不能给不同估计器使用不同尺度。

本阶段要回答四个问题：

1. 原始同批次估计 \(\bar g^2\) 是否呈现理论预期的 \(\sigma^2/B\) 正偏，以及该偏差是否随 batch size 按 \(1/B\) 衰减。
2. 梯度因子双采样与 microbatch-level U-statistic 的重复均值是否在参考不确定性内等价于 \(\mu^2\)。
3. 在相同总样本预算下，两种无偏方法的方差、MSE、排序质量与 top-k 恢复能力如何随 batch size、microbatch 数和训练阶段变化。
4. 综合统计效率、运行时间、峰值显存和实现复杂度，哪一种估计器应进入后续正式实验。

以下内容不属于本阶段：

- AdamW 实际更新或完整参数路径贡献的无偏性；
- 多节点路径积分和数值求积误差；
- 训练过程中的累计 positive、negative 或 absolute importance；
- 剪枝功能验证；
- 160M/410M 正式训练结论。

上述内容分别留给 Stage 3 及后续阶段。特别地，单次 U-statistic 允许为负；在偏差、方差、MSE 或排序分析前执行 `clamp_min(0)` 会重新引入正偏，本阶段禁止这样做。

## 3. 已核实的当前基线

### 3.1 本机与仓库

- 本机是 Windows 控制与开发端，没有 NVIDIA GPU；当前 Python 3.12.4 只安装了 CPU 版 PyTorch 2.10.0，没有 Transformers、Datasets 或 Accelerate。本机只承担文档、轻量单元测试、配置检查、小型表格/图表复核和 Git/SSH 调度。
- 当前 `main` 的已跟踪内容以文档、环境锁定文件、工作日志和旧实验轻量归档为主；没有现行 `src/`、`tests/`、`configs/`、项目构建声明或 Stage 2 运行入口。
- 工作树存在用户或其他并发任务正在编写的数学文档和 Stage 0/1/3 计划。它们必须保留；本计划不覆盖、不暂存隐藏，也不把“计划文件已出现”视为对应阶段已经通过。
- 旧 Stage A/B 的实现只可从历史 Git 对象中只读参考，当前主线不能直接运行；这些未引用对象还可能被 Git 回收，不能成为构建依赖。旧归档明确排除了源码、checkpoint 和约 6.5 GiB 逐单元数据，因此不能从归档续跑。

### 3.2 服务器与资产

- sophgo13 的项目环境位于 `$DATA_ROOT/envs/parameter-importance`，已有 PyTorch 2.12.1+cu126、Transformers 4.57.6、Datasets 4.8.5、Accelerate 1.14.0 和分析/绘图库；服务器环境才是 GPU 计算与正式结果的权威环境。
- PCI/驱动目录可见 8 张 A100-SXM4-80GB，但 NVML 与 PyTorch 当前只枚举 7 张；缺失的是 PCI `0000:4f:00.0` / device minor 0。由于编号前移，当前 `cuda:0` 实际对应 PCI `0000:50:00.0`，该卡累计 24 次 DRAM 不可纠正 ECC，且 row remap 为 pending。即使当前没有计算进程，也不能使用旧数字 index 或把该设备/旧四卡结果视为健康证据。
- 项目大盘有足够的当前余量，但正式运行前仍须重新检查挂载、空间、inode 和权限。根盘不得承载模型、缓存、环境或实验结果。
- Pythia 14M step0 与 Pythia 31M deduped step0 已存在；31M 的两份 JSON manifest 当前带 UTF-8 BOM，标准 JSON 解析失败，修复并重新验收前不得通过资产 Gate。
- Stage 2 要比较训练阶段，但当前只确认了两个模型的 step0。早期和中后期 checkpoint 仍须按固定 revision 单独准备与验收。
- Pile shard 0–4 是最终文件，但本阶段的默认 allowlist 只使用 `prefix_coverage.json` 已验收的 shard 0 前缀：`document-00000-of-00020.bin`、`document.idx` 和 sample ID `[0, 524288)`。shard 5 仍由既有受控链路写入 `.part`；不得递归枚举、读取、移动、改名或另起竞争下载。shard 1–4 若要进入后续扩展，必须另做逐文件 manifest 验收和 amendment。
- 本机与服务器的 `Agent/worklogs.md` 哈希当前不一致；多端同步债务未解决前不能通过进入 Gate。

### 3.3 旧证据的使用边界

旧归档可以用于提出风险和回归用例，不能用于宣称本阶段完成。旧估计器实验仅覆盖 14M、短程本地 checkpoint、`microbatch_size=1` 和 `M={4,8,16,32}`，因此 batch size 与 M 混杂；参考样本量固定为 4096，未建立参考收敛 Gate；也没有 31M、独立样本预算轴或完整端到端成本证据。

旧结果曾显示 raw 聚合偏差明显大于 U-statistic，但 signed U 的跨 M 稳定性 Gate 失败；后来使用 `mean(clamp_min(U,0))` 才通过稳定性，而该非线性量不再无偏。后续对称诊断支持 signed U 与 signed double 的功能排序接近，但范围是 160M/SST-2，与本阶段的固定状态局部目标不同。因此新实验必须保存 signed 结果、负值比例和不截断的重复统计，并预注册稳定性判据。

## 4. 预注册的主设计

- **模型**：Pythia 14M 和 Pythia 31M deduped。由于两者当前数据版本身份不同，跨模型只用于复现估计器现象，不能把效应差异单独归因为参数规模。
- **训练阶段**：每个模型至少选择初始化、早期和中后期三个固定 checkpoint。精确 revision 按训练进度分位规则选择，在查看估计器结果前冻结。
- **数据单元与 estimand**：\(\mathcal F\) 是固定长度、窗口不重叠 packed sequence 的冻结经验分布。`reference_sizing`、最终 `reference_A/B`、pilot 和 confirmatory 使用彼此独立的 RNG stream 从同一 \(\mathcal F\) 有放回抽样；draw ID/stream 不复用，sample ID 可以按理论概率偶然碰撞，必须记录且不得事后去重。先用 sizing stream 冻结固定 \(B_{\mathrm{ref}}\)，再用全新 A/B streams 一次性生成主 reference，禁止在最终 reference 上可选停时或“重抽到通过”；one-shot 失败即本轮阻断。结论首先条件于该经验分布，不自动外推到完整 Pile。有效目标 token 数用于损失加权和成本报告，不把同一序列内 token 伪装成独立重复。
- **固定状态**：`eval` 模式，禁用会改变状态的层和优化器操作；主偏差实验使用 FP32 梯度，参考累加使用更高精度；梯度裁剪关闭，即 \(s=1\)。
- **公平预算**：一次重复使用总预算 \(B\)。raw 使用全部 \(B\) 个单元；U-statistic 把相同 \(B\) 个单元分为 \(M\ge2\) 个 microbatch；双采样把同一总池按预先固定的规则分成两个独立半区，各用 \(B/2\)。
- **嵌套 M**：先产生最大预注册 `M_max` 个基础 microbatch 梯度，再以固定嵌套分组合成较小 M。这样不同 M、raw 和双采样可共享同一批样本与梯度计算，形成配对比较，不把 M 与 batch size 混淆。
- **参考目标**：偏差主参考由独立 reference blocks 的去对角 U 统计量给出；A/B 交叉乘积作独立敏感性检查；合并 reference 均值平方只用于排序和收敛。三者都报告不确定性，不能用有限参考误差放宽等价边界。
- **实验分层**：先做人工分布与 14M step0 校准，再做 14M 多阶段主实验，最后用 31M 预注册条件确认。只有前一层质量 Gate 通过才扩大。
- **统计单位**：置信区间以独立重复、checkpoint 或模型为单位；参数坐标只用于分布和分层汇总，不能把数百万相关坐标当作数百万独立样本。
- **产物目录**：大型产物限定在 `$DATA_ROOT/results/stage2/{reference,raw,derived}`、`$DATA_ROOT/runs/stage2`、`$DATA_ROOT/reports/stage2`、`$DATA_ROOT/manifests/stage2`、`$DATA_ROOT/operations/stage2` 和 `$DATA_ROOT/tmp/stage2` 的唯一 run-ID 子目录。唯一例外是按 `Agent/sync.md` 生成的精确 Git 同步临时文件 `$DATA_ROOT/tmp/repo-sync-<short-commit>.bundle`，验证后只删除该精确路径。不得清空既有目录、递归改权或使用系统 `/tmp`。
- **缓存与临时目录**：`HF_HOME`、`HF_DATASETS_CACHE`、`TORCH_HOME`、`XDG_CACHE_HOME` 必须映射到 `$DATA_ROOT/cache` 下的项目目录，`TMPDIR` 映射到 `$DATA_ROOT/tmp/stage2/<run-id>`；不得修改服务器 `HOME`，也不得回落到根盘或系统 `/tmp`。
- **四卡口径**：正常路径要求 Stage 0 的健康四卡和 DDP Gate 在进入 Stage 2 前通过。若用户明确批准单卡偏离，只能生成标为 provisional 的科学中间结果；补齐四卡语义/成本前不能宣称 Stage 2 完成，也不能进入依赖多卡的正式训练。GPU 修复只由管理员或已授权流程执行，本阶段禁止 reset、驱动变更、功率/时钟设置、持久化模式变更或 `sudo`。

## 5. 子任务与执行顺序

1. [S2.1 范围、假设、判据与预注册](01_scope_hypotheses_and_preregistration.md)
2. [S2.2 Stage 0/1 交接与固定状态代码契约](02_stage1_handoff_and_fixed_state_contract.md)
3. [S2.3 模型、checkpoint、数据与采样框](03_assets_checkpoints_and_sampling.md)
4. [S2.4 大样本参考目标与参考不确定性](04_reference_target.md)
5. [S2.5 配对估计器运行器与公平预算](05_paired_estimator_runner.md)
6. [S2.6 试运行、精度预算与正式矩阵冻结](06_pilot_and_matrix_freeze.md)
7. [S2.7 14M 主实验与 31M 确认实验](07_main_sweep.md)
8. [S2.8 统计分析、稳健性与结论规则](08_statistics_and_robustness.md)
9. [S2.9 运行时间、显存与工程成本](09_cost_and_system_validation.md)
10. [S2.10 可视化、报告与主估计器决策](10_visualization_reporting_and_decision.md)
11. [S2.11 交付、工作日志与进入后续阶段](11_delivery_and_exit_gate.md)

依赖关系为：

```text
S2.1 -> S2.2
S2.1 -> S2.3
S2.2 + S2.3 -> S2.4
S2.2 + S2.3 -> S2.5
S2.4 + S2.5 -> S2.6 -> S2.7
S2.7 -> S2.8
S2.7 -> S2.9
S2.8 + S2.9 -> S2.10 -> S2.11
```

S2.3 的额外 checkpoint 准备可与 S2.2 的交接验收并行。人工分布和 14M step0 开发 smoke 可在 `G2.2-dev` 后开展，但完整矩阵冻结必须同时通过 G2.2、G2.3 和 G2.4a。S2.9 的 profiler 可先在 pilot 中校准；正式成本表必须绑定 S2.7 的相同配置，并补充方法独立 anchor。

## 6. Gate 总览

| Gate | 负责文件 | 核心证据 | 解锁对象 |
|---|---|---|---|
| **G2.0 进入与预注册** | S2.1 | Stage 0/1 交接前提、estimand、确认性判据族、阈值和哈希冻结 | S2.2、S2.3 |
| **G2.1 Stage 1 交接** | S2.2 | `G1-EXIT` exact commit/API、公式、registry、固定状态和分层测试全部复验 | 正式 reference/runner |
| **G2.2-dev 开发资产** | S2.3 | 14M step0 与小型独立 draw stream 可重放 | 仅人工/14M step0 smoke |
| **G2.2 资产与抽样** | S2.3 | 两模型三阶段 checkpoint、经验分布、独立 RNG streams 和 manifests 冻结 | 正式 reference、matrix freeze |
| **G2.3 参考** | S2.4 | 独立 sizing 冻结所有候选 B 的 margin 与固定样本量；one-shot 无偏 bias reference、排序 reference、sequence 方差和最严格候选精度 Gate 通过 | 正式精度设计 |
| **G2.4a 运行器** | S2.5 | 公平预算、配对不变量、不可变分片/reducer、恢复和状态不变 | pilot |
| **G2.4b 试运行与矩阵** | S2.6 | 多模型/阶段功效 anchors、R、B、M、成本口径和确认矩阵冻结 | S2.7 |
| **G2.5 数据完整性** | S2.7 | 所有预注册单元完成或显式失败，raw 结果封存 | S2.8、S2.9 |
| **G2.6 统计有效性** | S2.8 | 两阶段不确定性、主判据归并、多重性和稳健性完整 | 方法决策 |
| **G2.7a 工程成本** | S2.9 | 共享科学、方法独立、在线训练三种成本口径可比 | 方法决策 |
| **G2.7b 方法决策** | S2.10 | 唯一候选分支、图表/表格可重建 | 交付 |
| **G2.8 交付同步** | S2.11 | 重放、日志、提交、多端与大型产物索引闭合 | Stage 3/4 对应 Gate |

## 7. 科学结论与执行 Gate 的区别

实验可以正确完成但不支持原假设。例如某个晚期 checkpoint 的梯度信噪比很高，使 raw 偏差小到参考误差内不可分辨，这不等于执行失败。报告必须把每个预注册假设标为“支持”“不支持”或“证据不足”，并保留完整结果。

真正阻断后续阶段的是证据链无效，例如参考未收敛、固定状态被改变、样本预算不公平、U 被截断、两种无偏方法都未通过等价性、或产物无法追溯。不得为了获得预期结论而在看到正式结果后修改 batch、M、重复次数、checkpoint 或阈值。

## 8. 阶段退出后的决策分支

- **U 作为主估计器**：U 在六个主 cells 全部通过偏差等价性，且在相同 cells 的校正 MSE、排序和在线成本均不劣于双采样；双采样保留为小比例校准基线。
- **双采样作为主估计器**：双采样在六个主 cells 全部通过，但 U 在偏差、数值稳定性、校正 MSE、排序或在线成本上失败；后续阶段使用双采样，并记录额外数据/计算预算。
- **U 经调整后再验证**：U 无偏但方差或成本不合格时，本轮保持 blocked；未来可建立独立预注册轮次研究新 B/M，但必须保留原结果，并预先控制跨已尝试配置的多重性/序贯选择。补充结果不能覆盖本轮或自动把本轮改判为通过。
- **阻断**：两种无偏方法或参考目标均不可靠。此时返回 Stage 1/本阶段诊断，不进入 Stage 3–4，也不把 raw 当作默认替代。
