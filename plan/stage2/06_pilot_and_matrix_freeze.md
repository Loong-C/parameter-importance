# S2.6 试运行、精度预算与正式矩阵冻结

## 1. 子任务目的

在不接触 confirmatory draws 的前提下，用人工数据和独立 `pilot` stream 按预注册算法选择可运行的 B/M pair，确定重复次数、存储策略和调度规模，并审计已经 one-shot 冻结的 reference 预算。Pilot 只允许提供可运行性、方差与资源输入，不能用于选择“结果更好”的 checkpoint、方法或科学阈值。

## 2. 前置条件

- S2.1 预注册已提交；
- 人工和 14M step0 开发 smoke 至少要求 G2.1、G2.2-dev 与 G2.4a；
- 正式矩阵冻结要求完整 G2.2、每个主 cell 的 G2.3 和 G2.4a 全部通过；
- 正式 pilot 前，S2.4 已为六个主 cells 建立并通过 one-shot reference；
- S2.5 runner 通过配对、状态和恢复测试；
- `pilot` 与 `confirmatory` 使用独立 seed namespace/draw IDs；sample-ID 偶然碰撞按有放回理论保留并审计。

## 3. 实施步骤

### 3.1 运行人工分布校准

1. 构造均值为零、方差已知的高斯或固定离散梯度样本。
2. 覆盖 M 为 2、4、8、16、32。
3. 验证 raw 平均偏差等于理论方差项。
4. 验证 U 和双采样重复均值落入零附近的预设 Monte Carlo 区间。
5. 验证 M=2 时 U 与双采样逐元素相等。
6. 验证 M>2 时经验方差不高于等预算双采样，并与理论趋势一致。
7. 构造不等权样本，单独验证加权 U。

### 3.2 运行最小真实模型开发 pilot

1. 使用 pilot 样本和最小 B 运行少量 repetition。
2. 逐步启用 M=2、4、8、16、32。
3. 检查每个基础 microbatch 的 loss、gradient norm 和有效 token。
4. 检查状态不变、M=2 等价、负 U 传播和流式统计。
5. 从中断点恢复一个 repetition，验证结果哈希一致。
6. 测量 warm-up 后的稳定 wall time、峰值显存和结果写入量。
7. 本小节只能通过开发 smoke，不能据此确定覆盖两个模型和三训练阶段的正式 R。

### 3.3 校准 batch size 网格

1. 初始候选使用 B 为 32、64、128、256 个 sequence。
2. 对每个 B 验证能被计划保留的 M 整除。
3. 若某 B/M 因显存不可运行，先减小每次 forward 的基础 microbatch 实现批量，而不改变统计 microbatch 定义。
4. 若仍不可运行，按预注册规则把该候选标为不合格；不能修改规则或增加新 B，所有候选均失败时本轮阻断。
5. 正式网格至少保留三个相隔倍数的 B，并覆盖总跨度不小于 4 倍。
6. 31M 至少保留一个小 B、一个大 B，以及 M=2 和一个 `M>2` 的主条件；资源允许时保持与 14M 完全相同矩阵。

### 3.4 校准 M 网格与嵌套分组

1. 以 `M_max=32` 为首选基础分组数。
2. 验证 `{2,4,8,16,32}` 均可由同一 32 个基础单元嵌套合并。
3. 检查每个 M 的 mean gradient 完全一致。
4. 检查较小统计 microbatch 与实际 forward microbatch 的区分在日志和配置中清楚。
5. 若 M=32 造成过高 kernel/调度开销，可把 32 降为探索性，正式矩阵至少保留 2、4、8、16。
6. 任何删除必须基于预先登记的时间/显存上限，而不是 U 的结果好坏。

### 3.5 估计正式重复次数

1. 对每个模型的初始化、早期和中后期 checkpoint 至少运行一个轻量 pilot anchor；不能只用 14M step0 的方差代表全部条件。
2. 每个 anchor 先运行 50 个 pilot repetitions，以 repetition 为单位估计 bias、校正 NMSE 比和排序差的方差；B/M 选择表屏蔽方法均值、bias 方向、方法优劣和显著性，只暴露可运行性、方差、R 需求和资源字段。
3. 对偏差等价检验按 sizing stream 已为每个候选 B 独立冻结的 `delta_sci(B)` 计算达到目标区间半宽所需的 R，并把 reference block 不确定性纳入功效计算。
4. 对 U/double 配对校正 NMSE 比、Spearman 和 Overlap@`1%` 非劣检验分别计算所需 R。
5. 取六个 model×stage anchors 和所有主判据所需 R 的最坏值，向上取整到便于分片恢复的批次大小；全部 31M confirmatory draws 与 pilot 独立，并使用不小于该最坏值的 R。
6. 正式 R 下限设为 200，上限初设为 1000。
7. 若某候选的估计 R 超过上限，该候选按 S2.1 扫描规则判不合格并检查下一个更大 B；所有候选均不合格时本轮阻断。不得延长 one-shot reference、看到方法结果后缩小主 scope，或截断 R 后宣称证据充分。
8. 在正式运行开始后不做可选停止；只能按预注册失败/资源规则暂停。

### 3.6 校准存储与流式统计

1. 估算每个 model/checkpoint/B/M/method 的参数级累计数组大小。
2. 估算 repetition 标量表、日志、失败记录和 reference 数组大小。
3. 固定完整参数数组的 anchor 条件清单，并强制纳入六个确认性主 cells 的 `B_primary/M_primary`、raw、double 和 U。
4. 固定其他条件只保存流式参数摘要、layer/module 表和诊断坐标。
5. 预留 staging、重试和报告派生数据空间，不能只按最终文件估算。
6. 将总预算与大盘可用空间比较，并设置停止阈值。
7. 明确原始全参数梯度和每 repetition 全估计向量默认不持久化；同时逐项证明三类 reference 的主排序/top-k、跨 M 配对指标和 Gate 所需 sufficient stats 已在释放前保存。

### 3.7 校准运行调度

1. 选择一张通过健康 Gate 的 A100 作为主统计运行设备，不硬编码物理编号。
2. 固定同一成本比较内的 GPU 型号与并发条件；时钟、功耗、温度只观察和记录，不设置 power limit、clock、persistence mode，也不 reset/改驱动/使用 `sudo`。
3. 对 14M 和 31M 分别测量单 repetition 时间。
4. 估算 reference、pilot、14M 正式和 31M 确认的 A100-hours。
5. 固定并发 worker 数，避免同卡并发改变显存与时间。
6. 为每个 wave 记录 `cost_io_quiescent`。Pile 下载等 I/O 活跃时科学 smoke 可继续，但成本证据无效，必须另选窗口复测；不得停止或修改既有受控任务。
7. 设定 GPU ECC/Xid、磁盘余量、NaN/Inf 和重复失败率的 fail-closed 条件。

### 3.8 冻结确认性矩阵

1. 写出完整 model × checkpoint × B × M × repetition 单元列表。
2. 为 raw/double 只创建每个 model/checkpoint/B/repetition 一次的共享单元，避免在 M 维度重复计数。
3. 标记 14M 校准、14M confirmatory、31M confirmatory 和探索性单元。
4. 冻结每个单元的样本映射、seed、reference、设备策略和输出路径。
5. 冻结 R、最大失败重试次数和完成分母。
6. 冻结分析版本、图表清单和 Gate 版本。
7. 生成 matrix-freeze record，逐项说明 S2.1 候选被固定规则保留/删除的原因；它不能修改科学阈值、扫描顺序或新增候选。
8. 在读取任何 confirmatory 梯度前提交 amendment 和矩阵 manifest。
9. 严格执行 S2.1 的扫描规则：先为每个 B 按 `[32,16,8,4]` 指定第一个满足整除、完整 sequence、数值稳定和 25% 聚合开销上限的 `M_candidate(B)`；再按 B 升序选择第一个“六 anchors 可运行、最坏 R 不超上限、总资源不超预算”的 pair。保存所有候选过滤表，不查看方法效应方向或优劣。
10. 唯一冻结共同的 `B_primary`、`M_primary>2`、六个主 cells、主 top-q/metrics 和交集归并规则；其他 B/M 标为 secondary。
11. 冻结三种成本语义：`scientific_equal_sample_cost`、`isolated_estimator_cost`、`online_training_incremental_cost`；方法决策只使用第三种，不能用共享 runner 成本替代。
12. 在 pair、R 和完成分母全部冻结后生成正式 confirmatory repetition mappings；逐项验证 draw ID 唯一、sample ID 碰撞率、B/M 嵌套、double 两半、seed/manifest 重建和输出路径，再允许读取第一个确认性梯度。

## 4. 产出

- 人工分布校准报告；
- 14M step0 pilot 结果与恢复报告；
- B/M 可运行性与性能表；
- one-shot reference 预算审计、重复次数、A100-hours 和存储预算；
- 完整参数 anchor 条件清单；
- 冻结的 confirmatory matrix manifest；
- matrix-freeze record 和哈希；
- GPU/存储/失败停止规则；
- `B_primary`、`M_primary`、六个主 cells 和机器可读确认性判据族表；
- 全候选确定性扫描表，包含固定优先级、盲化字段、过滤原因和唯一选择；
- 正式 confirmatory repetition mapping manifest 与逐项重建审计；
- 三种成本语义与 `cost_io_quiescent` 字段。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.4b 试运行与矩阵 Gate**：

- 人工分布的 raw/U/double 均符合理论方向，M=2 精确等价；
- 各模型/训练阶段 pilot anchors 无状态漂移、NaN/Inf、draw stream 冲突、选择性重抽或 silent skip；sample-ID 偶然碰撞符合有放回规则；
- 至少三个 B 和四个 M 可正式运行，或 amendment 给出不依赖结果的缩减理由；
- R 由精度公式确定且在可接受上限内；
- 完整运行所需 GPU 时间、存储和失败余量小于冻结预算；
- confirmatory matrix、样本、阈值和分析版本已在正式数据前提交；
- R 取跨模型/阶段最坏主判据需求，全部 31M confirmatory draws 独立且唯一主 B/M 已冻结；
- B/M 完全由预注册扫描优先级和可运行性/方差/资源字段决定，未使用方法均值、方向或优劣；
- pilot/confirmatory seed namespace 与 draw IDs 独立，sample-ID 碰撞已审计而非强制为零；
- 正式 repetition mappings 已在任何确认性梯度前生成，draw ID、B/M、double 两半和重建审计全部通过；
- 成本 wave 的 `cost_io_quiescent` 与三种成本语义已冻结。

若 Gate 未通过，只能调整实现、资源或预注册并重做 pilot；不得先运行一部分 confirmatory 单元“看看结果”。

## 6. 后续依赖

- S2.7 必须逐行消费冻结矩阵，不允许临时增加/删除主单元。
- S2.8 使用冻结的 R 和完成分母判断缺失数据。
- S2.9 使用 pilot 的 profiler 口径，但成本结论以正式运行复测为准。
