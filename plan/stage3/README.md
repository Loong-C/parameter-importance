# 第 3 阶段：路径积分数值近似验证计划

## 1. 文档定位

本目录把 [`general_plan.md`](../general_plan.md) 中的第 3 阶段展开为可执行、可核验、可追溯的子任务。这里描述的是实施计划，不代表第 3 阶段已经开始或完成。

计划依据包括：

- `Agent/` 下五份现行运维文档；
- [`docs/mathematics.md`](../../docs/mathematics.md) 中参数路径积分、数值求积、独立 probe、AdamW 边界和数学不变量；
- 2026-07-18 对本机仓库、服务器硬件、环境、存储和资产所做的只读核验；
- `worklogs/` 中的环境准备、旧实验归档与资产记录；
- `legacy/2026-07-stage-ab/` 中仅供风险识别的历史数值积分证据。

所有“当前状态”都是 2026-07-18 的快照。正式执行任一子任务前必须重新采集状态，不能把本计划中的快照当作届时的通过证据。

## 2. 阶段目标

第 3 阶段要回答一个窄而明确的问题：

> 对固定更新端点、固定线性参数路径和固定独立 probe 损失，使用多少个梯度求值节点、采用哪一种求积规则，才能以可接受的成本可靠近似逐参数路径积分贡献？

阶段完成时必须做到：

- 从同一个实际 optimizer step 捕获更新前后参数端点；
- 把“实际更新端点之间的线性路径”固定为本阶段的主路径约定；
- 在路径全部节点使用同一个确定性 probe 标量损失；
- 分开度量求积截断误差、probe 抽样波动和浮点/实现误差；
- 同时验证总和完备性、逐参数贡献、排序、top-q 集合、层级和模块级分配；
- 建立可信的高精度参考，而不是把某个固定节点数的方法直接称作真值；
- 在精度和成本之间选出一个唯一的默认方案，并给出明确的回退方案；
- 形成 Stage 4 可以直接引用的配置、适用边界、成本预算和 gate 证据。

本阶段不是重新验证 Stage 2 的 raw、双采样或 U-statistic 偏差，也不把局部梯度空间 U-statistic 误称为 AdamW 完整路径积分。

## 3. 已核实的当前基线

### 3.1 本机与仓库

- 本机当前位于 `main`，本机、GitHub 和服务器基线提交均为 `34966d0819a5229569169bfe436afe9058c0ed24`。
- 本机没有 NVIDIA GPU；当前 CPU 环境可承担解析测试、配置与报告审查，但不能承担 Pythia、CUDA 或性能结论的正式验证。
- 本机当前有用户修改的 `docs/mathematics.md`，以及并行起草、尚未纳入 Git 的 Stage 0–2 计划。全部内容必须保留并在后续 Git 工作中一并审查，不能还原、隐藏或绕过。
- `docs/mathematics.md` 的当前修改修复了同批梯度平方期望公式中的错误字符；第 3 阶段计划以修复后的工作树内容为数学依据。
- 当前 `main` 没有现行 `src/`、`tests/`、实验配置或运行入口；旧实现只存在于历史归档和 Git 历史中。
- Stage 0–2 的详细计划已在当前工作树中并行起草，但这些文档都明确描述未来工作，不是 gate 通过证据。
- Stage 1 和 Stage 2 尚无现行代码、测试、正式结果或通过报告。
- 本机与服务器的 `Agent/worklogs.md` 哈希当前不一致，其余四份 `Agent` 文档一致；最终同步 gate 尚未满足。

结论：当前可以撰写和审查 Stage 3 计划，但不能开始真实模型的正式 Stage 3 实验。

### 3.2 服务器资源

- 服务器现有项目环境为 Python 3.12.3、PyTorch 2.12.1+cu126，所需的 NumPy、SciPy、Pandas、Matplotlib、Seaborn 和 TensorBoard 已在锁定环境中。
- 项目大盘约有 2.9 TiB 可用，足以保存 Stage 3 的端点、节点缓存和原始结果；系统根盘仅约 15 GiB 可用，任何缓存、临时文件和实验产物都不得落到根盘。
- 服务器底层仍能识别 8 张 A100 的设备痕迹，但 NVML/PyTorch 当前只枚举 7 张；当前可见 GPU 0 有 24 次不可纠正 ECC 记录。
- Stage 3 以单张健康 A100 为主，不需要四卡吞吐，但正式运行仍必须等待硬件 gate，且不得使用有 ECC 异常的设备或在配置中硬编码物理 GPU 编号。
- 当前没有参数重要性训练任务；Pile 的既有受控下载仍在写 shard 5 的 `.part`。Stage 3 只依赖已经验收的 shard 0，不得读取或干预活动 `.part`、锁和下载任务。
- 正式运行时间和吞吐测量必须在下载与其他重 I/O 任务不会污染 wall-clock 的安静窗口进行。

### 3.3 可用资产

- Pythia 14M step0、Pythia 31M deduped step0、Pythia 160M deduped step0/step512 已位于服务器 `models/`。
- Pythia 14M 是主验证模型；Pythia 31M 用作跨规模确认。Stage 3 不要求 160M 承担高密度参考积分。
- 已验收的 Pile shard 0 与 `document.idx` 足以支持短程 14M/31M 轨迹和独立 probe 取样。
- 31M 顶层 manifest 在 Stage 0 草案核验中被记录为不可解析；修复并重新验收之前不能用于正式 Stage 3。
- 所有模型、数据和 probe 都必须通过逻辑资产 ID 与 manifest 解析，不能把“目录存在”当作 ready。

### 3.4 旧归档的使用边界

旧 Stage A/B 归档记录了一个有价值的风险信号：

- 旧 14M 实验中，左端点方法的层级 Spearman 最低为 0.56667，未通过旧门槛；
- 独立的两节点梯形 v2 中，参数和层级最低 Spearman 分别为 0.98322 和 0.96667，旧门槛通过。

这些结果只能作为风险先验和回归对照，不能作为新 Stage 3 的通过证据，原因包括：

- 旧现行源码、测试和 checkpoint 已从 `main` 清理；
- 旧结果采用 4 个 probe 样本，统计覆盖有限；
- 旧 Gauss-Legendre 16 节点结果直接充当零误差参考，没有独立的参考加密收敛 gate；
- 旧 metrics 未形成当前数学规格要求的固定 probe 端点损失完备性证据链；
- 当前硬件枚举和健康状态已漂移。

新阶段必须从现行代码、现行端点和现行 gate 重新执行，旧阈值也不得在看到新正式结果后被用作放宽理由。

## 4. 主数学对象与命名边界

固定第 \(t\) 个 optimizer step 的更新前后参数：

\[
\Theta_t,\qquad
\Theta_{t+1},\qquad
\Delta\Theta_t=\Theta_{t+1}-\Theta_t.
\]

主路径固定为端点线段：

\[
\Gamma_t(\alpha)=\Theta_t+\alpha\Delta\Theta_t,
\qquad \alpha\in[0,1].
\]

对固定独立 probe 集 \(\mathcal P\) 的确定性标量损失 \(\mathcal L_{\mathcal P}\)，逐参数贡献定义为：

\[
C_{k,t}^{\mathrm{path},\mathcal P}
=
-\Delta\theta_{k,t}
\int_0^1
\frac{\partial \mathcal L_{\mathcal P}(\Gamma_t(\alpha))}
{\partial\theta_k}
\,d\alpha.
\]

对应完备性为：

\[
\sum_k C_{k,t}^{\mathrm{path},\mathcal P}
=
\mathcal L_{\mathcal P}(\Theta_t)
-
\mathcal L_{\mathcal P}(\Theta_{t+1}).
\]

统一命名如下：

- `actual_path_left`：实际端点线性路径的一节点左端点求积；
- `actual_path_right`、`actual_path_midpoint`、`actual_path_trapezoid`、`actual_path_simpson`：同一目标的不同低成本求积；
- `actual_path_composite_*` 与 `actual_path_gauss_legendre`：同一目标的多节点求积；
- `local_gradient_space_importance_u`：Stage 2 的局部梯度空间目标，只作为桥接诊断，不参与完整路径完备性 gate；
- `full_update_path`：使用实际更新前后总参数端点，包含 AdamW 动量、预条件、裁剪和 weight decay 对最终位移的影响；
- `data_update_path`：只有在能够精确重建并验证 \(\Delta\Theta_t^{\mathrm{data}}\) 时才启用的反事实数据位移路径，不得称作实际更新后状态。

实际端点位移已经包含优化器产生的尺度，路径贡献不能再额外乘学习率或梯度裁剪因子。

## 5. 子任务与执行顺序

1. [S3.1 前置 gate、范围与现状冻结](01_prerequisites_and_scope.md)
2. [S3.2 数学对象、误差预算与指标合同](02_math_and_metric_contract.md)
3. [S3.3 更新端点、固定 probe 与无副作用状态管线](03_endpoint_and_probe_pipeline.md)
4. [S3.4 求积引擎与解析不变量测试](04_quadrature_engine_and_unit_tests.md)
5. [S3.5 高精度参考、收敛与精度底噪](05_reference_integral_and_precision.md)
6. [S3.6 小规模 pilot 与正式阈值冻结](06_pilot_and_threshold_freeze.md)
7. [S3.7 14M/31M 正式实验矩阵](07_formal_experiment_matrix.md)
8. [S3.8 误差分解、排序与层级稳定性分析](08_error_analysis_and_stability.md)
9. [S3.9 成本评估、方法选择与回退规则](09_cost_and_method_selection.md)
10. [S3.10 报告、可视化、重放与 Stage 4 交接](10_reports_visualizations_and_handoff.md)

依赖关系如下：

```text
Stage 0 + Stage 1 + Stage 2 通过
                │
                v
              S3.1
                │
                v
              S3.2
             /    \
            v      v
          S3.3    S3.4
             \    /
              v  v
              S3.5
                │
                v
              S3.6
                │
                v
              S3.7
                │
                v
              S3.8
                │
                v
              S3.9
                │
                v
             S3.10
                │
                v
          Stage 4 是否放行
```

S3.3 和 S3.4 可以在数学合同冻结后并行实现；真实模型正式实验必须等 S3.5 和 S3.6 全部通过。

## 6. Gate 总览

| Gate | 通过条件摘要 | 未通过时允许做什么 |
|---|---|---|
| G3-0 开工 gate | Stage 0 的 G0–G10、Stage 1 的 G1-EXIT、Stage 2 的 G2.7/G2.8 均通过；资产、健康单卡、存储、同步和安静运行窗口通过 | 只能继续计划、CPU 解析测试和不依赖真实资产的代码审查 |
| G3-1 数学合同 gate | 路径、端点、损失、参数范围、误差分解、指标和阈值冻结 | 不得生成可解释为正式结果的数据 |
| G3-2 状态与 probe gate | 端点可由裁剪后梯度与 update transition 唯一复原；pre/post 及全部节点共享同一 buffer 快照；probe 独立、确定；节点求值不改变训练状态 | 不得运行多节点真实模型比较 |
| G3-3 求积实现 gate | 节点权重、解析积分、理论阶、异常输入和恢复测试通过 | 不得建立高精度参考 |
| G3-4 参考积分 gate | 两类高阶方法与连续加密相互一致；参考误差进入预算 | 不得用该状态评价候选方法 |
| G3-5 pilot 冻结 gate | 正式状态选择、probe 数、节点上限、阈值和预算在看正式结果前冻结 | 只能修复实现或重新做独立 pilot |
| G3-6 正式结果完整性 gate | 预注册 14M/31M 单元全部完成、可恢复、无选择性缺失 | 不得进行方法放行判断 |
| G3-7 方法选择 gate | 精度、排序、层级、完备性和成本全部满足；唯一默认与回退方案明确 | Stage 4 阻塞，继续多节点或根因诊断 |
| G3-8 交付 gate | 原始结果、摘要、图表、报告、重放、工作日志和多端同步全部通过 | 不得宣称 Stage 3 完成 |

## 7. 方法选择总规则

正式选择只在预注册候选中进行：

1. 先淘汰任何参考未收敛、状态恢复失败或完备性不可信的方法/状态。
2. 再淘汰任何参数向量误差、排序、top-q、层级分布或最坏状态超出门槛的方法。
3. 在全部精度 gate 通过的方法中，选择实际梯度求值次数最少且 wall-clock/显存没有异常回退者。
4. 若一节点左端点通过全部 gate，则可作为默认；不得因成本低而豁免排序或层级 gate。
5. 若左端点失败、梯形通过，则选择梯形；若梯形失败、Simpson 通过，则选择 Simpson。
6. 若固定低节点方法均失败，则选择通过的最小复合或 Gauss-Legendre 方案，并给出基于诊断量的自适应回退规则。
7. 若最高预算参考自身仍不收敛，或没有任何候选通过，则 Stage 3 失败并阻断 Stage 4，不得在结果出来后下调阈值。

## 8. 阶段最终产物

- 冻结的数学与实验合同；
- 更新端点和独立 probe 的 schema、捕获与恢复机制；
- 通用求积引擎及解析测试；
- 参考积分收敛报告和 FP32/FP64 精度底噪报告；
- pilot 决策记录及正式预注册配置；
- 14M 主实验与 31M 确认实验的原始结果和 manifest；
- 完备性、向量误差、排序、top-q、层/模块稳定性和误差分解数据；
- 节点数、wall-clock、峰值显存和存储成本报告；
- 误差—成本收敛图、完备性图、排序稳定性图、路径曲线和层级对照图；
- 推荐的默认积分方案、回退方案、适用范围和失败条件；
- Stage 4 可机器校验的 gate 摘要与重放说明。

## 9. 不在第 3 阶段完成的内容

- 不重新证明 raw、双采样或 U-statistic 的统计无偏性；
- 不把同训练 batch 的 probe 结果当作 population 无偏证据；
- 不把独立 probe 的抽样波动伪装成求积截断误差；
- 不在 160M/410M 上执行高密度参考积分；
- 不决定最终剪枝有效性或训练范式差异；
- 不把 weight decay 路径静默解释为纯数据损失贡献；
- 不修改服务器驱动、重置 GPU、干预 Pile 下载或把大型产物复制到本机/GitHub。
