# 第 1 阶段：参数重要性计算的代码正确性验证计划

## 1. 文档定位

本目录把 [`general_plan.md`](../general_plan.md) 中的第 1 阶段展开为可执行、可核验、可追溯的实施计划。这里描述的是未来要完成的工作与 Gate，不代表代码、测试或实验已经完成。

计划依据如下：

- `Agent/` 下五份现行运维文档；
- [`general_plan.md`](../general_plan.md) 的阶段目标与推进关系；
- [`docs/mathematics.md`](../../docs/mathematics.md) 的统一数学定义、工程边界和代码不变量；
- 2026-07-18 对本机、GitHub、SSH 链路和 sophgo13 服务器进行的只读核验；
- `worklogs/` 与 `legacy/2026-07-stage-ab/` 中保留的环境证据、历史实现风险和旧实验结论。

所有“当前状态”都是 2026-07-18 的快照。正式执行任何子任务前必须重新采集状态；本计划中的快照不能直接作为届时的验收证据。

## 2. 阶段目标

第 1 阶段只回答一个问题：**实现是否忠实计算了数学规格中定义的量**。

本阶段主指标是固定参数状态下，经 Microbatch-level U-statistic 去偏的局部梯度空间贡献：

\[
\widehat C_{k,t}^{U}
=
\eta_t
\frac{S_{1,k,t}^2-S_{2,k,t}}{M(M-1)}.
\]

若启用全局梯度裁剪，在线累计量只额外乘一次全局裁剪因子 \(s_t\)。它估计的是 \(\eta_t s_t\mu_{k,t}^2\)，不是 AdamW 完整实际位移的严格无偏路径积分贡献。

本阶段必须证明：

- 标量损失、有效 token 权重和梯度 reduction 的语义一致；
- raw、双采样、等权 U-statistic 和加权 U-statistic 与独立手算一致；
- 逐样本、逐 microbatch、完整 batch、梯度累积和 DDP 的尺度关系正确；
- loss scaling、FP32 累加和全局梯度裁剪的顺序正确；
- signed、positive、negative mass、absolute、raw、参数移动量和参数幅值的累计语义正确；
- 统计逻辑不会改变优化器本应执行的训练轨迹；
- checkpoint 恢复后参数坐标、数据位置、随机状态和重要性累计连续；
- 每个结论都可由保存的配置、样本 ID、参数 manifest 和调试数组离线重算。

## 3. 本阶段不回答的问题

- 不在真实训练分布上证明 raw 的偏差大小或 U-statistic 的统计效率；这些属于 Stage 2。
- 不比较多节点路径积分方法的数值精度；这些属于 Stage 3。本阶段只保留常梯度与二次损失的解析 smoke test，用于验证符号、端点和状态恢复。
- 不扩大到 160M/410M 正式训练，不形成模型机理、训练范式或泛化结论。
- 不用剪枝结果证明功能重要性；这些属于 Stage 7。
- 不把单步 U-statistic 的负值当作错误，也不在核心估计器内提前截断。
- 不把 `double_sample_gradient_importance` 与独立 probe loss 对齐混为同一估计器。

## 4. 已核实的当前基线

### 4.1 本机与仓库

- 本机是 Windows 开发与调度端，Python 3.12.4、CPU 版 PyTorch 2.10.0、NumPy、SciPy 和 pytest 可用；Transformers、Datasets 与 Accelerate 当前未安装，CUDA 不可用。
- 本机适合运行纯张量代数、解析模型和轻量 CPU 测试；Pythia、CUDA、NCCL 与正式验收必须在服务器锁定环境中运行。
- 当前 `main` 为 `34966d0819a5229569169bfe436afe9058c0ed24`。仓库没有现行 `src/`、`tests/`、配置、打包或运行入口，Stage 1 需要从数学规格建立新的最小实现骨架。
- `docs/mathematics.md` 有用户未提交修正：修复公式中的异常控制字符并补文件末尾换行。该修改必须保留、审查并与本阶段文档一起追踪。
- `Agent/` 两端文件集合一致，但 `Agent/worklogs.md` 的 SHA-256 不一致；修复并复核前不能通过 `G1-ENTRY`。

### 4.2 服务器环境与存储

- 服务器仓库 `main` 与本机当前 HEAD 一致，服务器工作树干净。
- 现有权威环境位于 `$DATA_ROOT/envs/parameter-importance`：Python 3.12.3、PyTorch 2.12.1+cu126、Transformers 4.57.6、Datasets 4.8.5、Accelerate 1.14.0；实时 `pip check` 通过，A100 支持 BF16。
- 项目大盘约 3.5 TiB，总可用约 2.9 TiB，权限和 inode 现状满足小规模 Stage 1 调试产物。
- 当前硬件不能按“8 张健康 A100”规划：系统节点仍能看到 8 个设备痕迹，NVML/PyTorch 只枚举 7 张，其中 1 张实际 CUDA 分配失败；当前只确认 6 张可分配。历史四卡 NCCL 报告已经过时，任何 DDP 任务都必须先重新逐卡和多卡预检。
- Stage 1 只需要从健康候选中取得 1 张和 4 张卡；不得在代码或计划中硬编码物理卡号，也不得把 6 张可分配解释为所有卡健康。

### 4.3 模型与数据

- Pythia-14M step0 已按固定 revision `56079904bb80b7f36d3b794089f146e7a4d6efae` 重新核对 manifest，适合作为本阶段真实模型 fixture。
- Pythia-31M-deduped step0 的权重、tokenizer、离线前向与文件哈希通过，但其 JSON manifest 带 UTF-8 BOM；Stage 1 的 manifest 读取器必须显式测试 BOM、无 BOM、缺字段和哈希错误。
- Pile shard 0–4 已达到预期文件大小；其中 shard 0 的固定前缀已通过哈希与 reader 对齐，shard 1–4 仍须在正式使用前通过 full-manifest 哈希 Gate。shard 5 正在受控下载，仍有活动 `.part` 和锁；Stage 1 不得读取、移动、改名或清理活动文件。
- 已验证的 Pile 前缀覆盖前 524,288 个 2049-token 样本，并已与官方 batch viewer 在 step 0、1、511 对齐。Stage 1 的固定调试样本必须落在该已验证范围内，不需要等待全量 Pile。
- SST-2 与 WikiText-103 已验收，但本阶段主正确性测试优先使用解析 fixture、固定 token fixture 和 Pythia-14M；避免把任务数据差异混入公式验证。

### 4.4 旧归档的使用边界

- `legacy/2026-07-stage-ab/` 保存轻量报告和结果，不含旧源码、测试、权重或可恢复 checkpoint，不能作为继续训练的实现基础。
- 旧提交对象仍可用于只读审计，但旧分支与远端引用已删除；新代码必须以当前数学规格为准，不把旧提交作为运行依赖。
- 历史实现暴露的回归风险必须转成 Stage 1 测试：负贡献曾保存为负值而不是非负 negative mass、缺少 absolute 累计、只支持等大小 microbatch、没有真实 NCCL/DDP `no_sync` 验证、raw 名称曾混入 clip factor、movement 曾包含 weight decay，以及部分配置字段声明后未真正驱动代码。
- 旧的 156 项单元测试通过只证明旧提交当时的行为，不证明新数学规格或重建后的实现正确。

## 5. 固定方法命名与解释边界

| 字段 | Stage 1 固定含义 | 是否主指标 |
|---|---|---:|
| `local_gradient_space_importance_raw` | 未裁剪的同批次 \(\eta_t\bar g_t^{\odot2}\) | 基线 |
| `local_gradient_space_importance_raw_clipped` | 可选的 \(s_t\eta_t\bar g_t^{\odot2}\)，必须与 raw 分字段 | 诊断 |
| `double_sample_gradient_importance` | 两个独立 batch 提供两个梯度因子的逐元素乘积 | 对照 |
| `local_gradient_space_importance_u` | 未裁剪的等权或加权 microbatch U-statistic | 核心估计量 |
| `local_gradient_space_importance_u_clipped` | 只乘一次全局 clip factor 的在线累计量 | 正式在线主指标 |
| `importance_signed` | 单步主指标直接跨步求和 | 必须保存 |
| `importance_positive` | 单步主指标正部的累计 | 派生量 |
| `importance_negative_mass` | 单步主指标负部绝对值的累计，始终非负 | 派生量 |
| `importance_absolute` | 单步主指标绝对值的累计 | 派生量 |
| `parameter_movement_data` | 不含 decoupled weight decay 的累计绝对数据位移 | 基线 |
| `parameter_net_movement_data` | 每步带符号 data update 累加后取绝对值，即 \(\lvert\sum_t\Delta\theta_t^{data}\rvert\) | 基线 |
| `parameter_net_movement_total` | 参数端点差 \(\lvert\Theta_T-\Theta_0\rvert\)，包含所有实际更新来源 | 诊断 |
| `parameter_magnitude` | 当前或最终参数绝对值 | 基线 |
| `actual_update_raw_importance` | 数据驱动实际位移与同批梯度的局部乘积 | 诊断，不宣称无偏 |

任何实现若更改上述定义，必须更改字段名、schema 版本和数学契约，并重新运行本阶段全部 Gate。

## 6. 数值比较规范

所有容差在运行前固定。比较同时报告原单位最大绝对误差、尺度化最大误差、归一化 L2 误差、非有限值数量和最差参数张量；不得只比较全局总和或相关系数。所有误差指标先把待测值与独立 oracle 转成 FP64，再按“比较对象”分别尺度化，不能用一个乘过很小学习率的最终分数掩盖 core 公式错误。

对每个比较对象 \(q\)，令独立参考为 \(y_q\)，并在 fixture manifest 中预注册一个严格为正的原单位自然尺度 \(n_q\)。它只能由解析 fixture、独立 oracle 或输入设计得到，不能从待测实现输出反推。对所用 profile 的 `atol` 记为 \(a\)，定义近零阈值与绝对阈值：

\[
\tau_{0,q}=10a n_q,\qquad \delta_q=a n_q.
\]

比较使用互斥的两条分支：

1. 若 \(\lVert y_q\rVert_\infty>\tau_{0,q}\)，令 \(a_q=\lVert y_q\rVert_\infty\)，并定义 \(\widetilde x_q=x_q/a_q\)、\(\widetilde y_q=y_q/a_q\)。
2. 若 \(\lVert y_q\rVert_\infty\le\tau_{0,q}\)，停用相对误差与 normalized-L2 Gate，改为要求 \(\lVert x_q-y_q\rVert_\infty\le\delta_q\)，并把该分支、\(n_q\)、\(\tau_{0,q}\) 和 \(\delta_q\) 写入结果。

对非近零分支，设 \(S_q=\max(\lVert\widetilde x_q\rVert_\infty,\lVert\widetilde y_q\rVert_\infty)\)，基础通过条件为：

\[
\lVert\widetilde x_q-\widetilde y_q\rVert_\infty
\le \mathrm{atol}+\mathrm{rtol}\,S_q.
\]

归一化 L2 误差固定为

\[
E_{2,q}=
\frac{\lVert\widetilde x_q-\widetilde y_q\rVert_2}
{\max(\lVert\widetilde x_q\rVert_2,
       \lVert\widetilde y_q\rVert_2,
       10^{-300})}.
\]

其中 `1e-300` 只在非近零分支的 FP64 指标计算中防止零分母，L2 使用避免平方下溢的稳定范数实现。非近零 oracle 对全零实现会得到接近 1 的 normalized-L2，不能因学习率小而通过；近零与精确零对象则由预注册原单位阈值判定。

每个 requirement 至少注册并分别 Gate 下列对象，不能只检查最后一个字段：

| 对象 | 独立参考 | 自然尺度 \(n_q\) 的冻结规则 | 目的 |
|---|---|---|---|
| `mean_gradient` | 完整 batch 或逐 token mean oracle | 最大单元梯度设计尺度 \(n_g\) | 检查 loss/reduction |
| `S2` | 显式平方和 | \(M n_g^2\) | 检查等权二阶充分统计量 |
| `G2` | 显式加权平方和 | \((\sum_m b_m^2)n_g^2\) | 检查加权二阶充分统计量 |
| `raw_core` | \(\bar g^{\odot2}\) | \(n_g^2\) | 排除学习率缩放掩盖错误 |
| `u_core` | 不乘 \(\eta\) 的显式 cross-microbatch pair 均值 | \(n_g^2\) | 检查去对角公式 |
| `raw_score` / `u_score` | 对应 core 乘实际参数组学习率 | 预注册学习率设计尺度乘 \(n_g^2\) | 检查公开字段映射 |
| `optimizer_delta` | 独立 SGD/AdamW step oracle | 独立解析更新的设计尺度 | 检查 optimizer bridge |

其中 \(n_g\) 来自解析/离线 oracle 中各统计单元梯度的最大设计尺度；全零 fixture 也必须在 manifest 中提供正的解析输入尺度，不能令 \(n_q=0\)。

| 配置 | 用途 | `atol` | `rtol` | 归一化 L2 上限 |
|---|---|---:|---:|---:|
| `T64_ORACLE` | FP64 解析公式、独立手算和纯张量核 | `1e-12` | `1e-10` | `1e-10` |
| `T32_SINGLE` | 单进程/单卡 FP32 | `1e-7` | `1e-5` | `1e-5` |
| `T32_DISTRIBUTED` | 相同 global batch 的 FP32 多卡归约 | `1e-7` | `1e-4` | `1e-4` |
| `T_AMP_SCALE` | 相同 autocast 前向下 scale/unscale 等价 | `1e-6` | `5e-4` | `5e-4` |

表中 `atol` 在非近零分支作用于尺度化张量，在近零分支用于形成 `delta_q = atol * n_q`；原单位误差始终完整报告。各对象的 `n_q`、分支、oracle hash 与 profile 在运行前写入契约。

补充规则：

- 样本 ID、参数名称/形状/顺序、统计单元数、有效 token 数、step、数据游标和 schema 版本必须完全一致，不使用浮点容差。
- 参考量接近零时执行上述绝对误差分支，不同时使用相对或 normalized-L2 Gate。
- BF16 与纯 FP32 的差异只作质量报告，不作为估计器公式相等的 Gate；代码尺度 Gate 使用“同一 autocast 前向，scale 开关前后等价”。
- 任一参数张量不通过，则对应 Gate 失败；全局聚合误差较小不能掩盖局部失败。
- 发现阈值不合理时，必须先记录原因并升级契约版本；不得看过结果后只为当前 case 放宽阈值。

## 7. 子任务与执行顺序

1. [S1.1 入口基线与数学契约冻结](01_entry_and_contract.md)
2. [S1.2 代码架构与参数坐标注册表](02_architecture_and_parameter_registry.md)
3. [S1.3 确定性 fixture 与独立 oracle](03_fixtures_and_oracles.md)
4. [S1.4 损失 reduction 与梯度尺度](04_loss_and_gradient_scale.md)
5. [S1.5 raw、双采样与 U-statistic 内核](05_estimators.md)
6. [S1.6 训练步集成、累计量与比较基线](06_training_integration_and_accumulators.md)
7. [S1.7 Pythia-14M 单卡真实链路验证](07_single_gpu_pythia14m.md)
8. [S1.8 DDP、梯度累积与 `no_sync` 一致性](08_ddp_and_gradient_accumulation.md)
9. [S1.9 精度、loss scaling、裁剪与优化器边界](09_precision_clipping_and_optimizer_boundaries.md)
10. [S1.10 checkpoint、断点恢复与证据包](10_checkpoint_resume_and_artifacts.md)
11. [S1.11 报告、可视化与阶段退出 Gate](11_reporting_and_exit_gate.md)

依赖关系：

```text
S1.1 -> S1.2 -> S1.3 -> S1.4 -> S1.5 -> S1.6 -> S1.7
                                                    |      |
                                                    v      v
                                                  S1.8   S1.9
                                                    \      /
                                                     v    v
                                                     S1.10
                                                       |
                                                       v
                                                     S1.11
```

S1.8 与 S1.9 可在 S1.7 通过后并行。服务器 GPU 健康问题不阻止 S1.1–S1.6 的 CPU/单卡准备，但 `G1-DDP` 未通过时不得宣称 Stage 1 完成。

Stage 1 属于跨模块、跨机器的长任务。S1.1 应在入口基线提交后创建专用开发分支；在契约、骨架、CPU 内核与测试、单卡、DDP/数值、恢复、最终报告等可独立复核边界分别更新中文 worklog、提交并推送，并用已验证 bundle 快进服务器。S1.11 只做最终收尾同步，不替代前序里程碑同步。

## 8. Gate 总览

- **G1-ENTRY**：Stage 0 核心设施、当前资产、工作树、Agent 文档同步和健康 GPU 候选通过重新核验。
- **G1-CONTRACT**：方法名称、目标量、损失 reduction、有效统计单元、参数范围、step 顺序、dtype 和容差冻结。
- **G1-REGISTRY**：参数坐标注册表在构建、保存、加载和不同运行入口之间完全一致。
- **G1-ORACLE**：解析梯度、有限差分抽查、逐样本梯度和解析路径 smoke test 通过。
- **G1-GRAD**：完整 batch、等权 microbatch、有效 token 加权 microbatch 的 loss 与平均梯度一致。
- **G1-EST**：raw、双采样、显式成对 U、流式 U 和加权 U 与独立 FP64 手算一致，负值与非法边界正确处理。
- **G1-STEP**：累计恒等式、raw、data movement、magnitude、clip 和 AdamW 边界正确，统计开启不扰动训练。
- **G1-SINGLE**：Pythia-14M 固定样本的单卡 FP32 调试运行和完整梯度证据通过。
- **G1-DDP**：路由 A 与 D 的全局 mean gradient/raw/update 一致；保持同一组 \(M\ge2\) 个 microbatch 的路由 B、C、D 在充分统计量、U 与累计量上逐参数张量一致。
- **G1-NUMERIC**：scale/unscale、non-finite skip、FP32 累加与全局裁剪顺序通过；BF16 smoke 无非有限值。
- **G1-RESUME**：连续运行与断点恢复的数据、参数、优化器和全部累计量连续。
- **G1-EXIT**：所有 Gate 机器摘要为通过，未解决失败项为零，既有失败复现包与修复/重跑历史已归档，报告、图表源数据、manifest 和多端同步完整。

## 9. 预期仓库与服务器产物

建议的职责边界如下；实际路径在 S1.1 契约中冻结：

```text
Git 仓库
  src/parameter_importance/       估计器、梯度收集、累计与状态模块
  tests/stage1/                   解析、CPU、单卡、DDP 与恢复测试
  configs/stage1/                 固定 fixture 和验证矩阵
  reports/stage1/                 小型 JSON/CSV、Markdown 报告和图表
  plan/stage1/                    本计划
  worklogs/                       实施日志

$DATA_ROOT
  runs/stage1/<run_id>/           控制台、逐步调试与临时状态
  results/stage1/<run_id>/        参数级数组和完整梯度
  reports/stage1/<run_id>/        服务器侧完整报告
  checkpoints/stage1/<run_id>/<checkpoint_id>/
                                  已完成事务发布的正式 checkpoint
  tmp/stage1/<run_id>/<attempt>/<object_id>/
                                  临时发布、torchrun rendezvous 与损坏 fixture
  operations/                     本项目按 GPU UUID 建立的运行租约
  manifests/stage1-<run_id>.json  大型产物身份、大小和哈希
```

小型 oracle、Gate JSON/CSV 和最终图表可进入 Git；Pythia 完整逐 microbatch 梯度、参数级累计数组和大型调试 bundle 只保存在 `$DATA_ROOT`，Git 中记录绝对路径、大小、SHA-256、配置摘要和验收状态。

## 10. 后续阶段的前置 Gate

- **进入 Stage 2**：`G1-EXIT` 必须通过；固定 checkpoint、局部 raw/double/U 实现版本、参数 registry、损失契约、采样 seed 空间和参考数据范围必须冻结。Stage 2 只能改变抽样规模和统计实验配置，不能静默修改估计器语义。
- **进入 Stage 3**：除 `G1-EXIT` 外，常梯度、二次损失、固定 probe 状态和训练状态恢复 smoke test 必须通过。Stage 3 再比较节点数与求积方法，不沿用 Stage 1 的 smoke test 得出精度结论。
- **进入 Stage 4 及以后**：若代码、依赖、损失 reduction、DDP 归约、AMP、clip、optimizer bridge 或 checkpoint schema 有任何实质变更，必须重跑受影响的 Stage 1 Gate；不能仅凭旧报告继续放大模型。

## 11. 当前执行阻塞

计划编写本身不受阻，但正式 Stage 1 执行前至少需要关闭以下问题：

- 同步本机和服务器的 `Agent/worklogs.md` 并核对五份文件 SHA-256；
- 重新完成健康 GPU 选择与四卡 NCCL smoke，不使用故障设备；
- 把当前数学文档修正和 Stage 1 计划形成可追溯提交并完成三端同步；
- 建立现行代码、测试、配置和打包骨架；
- 为 14M fixture 固定模型 manifest、Pile 前缀样本 ID 和调试产物预算；
- 明确 Stage 0 哪些 Gate 已有新证据、哪些仍需补验，不能用 2026-07-13 的旧 `READY` 代替当前硬件 gate。
