# S2.9 运行时间、显存与工程成本

## 1. 子任务目的

比较 raw、双采样和 U 的完整端到端成本，而不是只比较公式中几个张量运算。成本证据要覆盖数据读取、forward/backward、梯度聚合、公式、统计、通信和结果写入，并在可比硬件与 I/O 条件下重复测量。

## 2. 前置条件

- Stage 0 profiler 字段和 GPU 监控接口通过；
- G2.4b 已冻结三种成本口径和主成本条件；
- S2.7 已产生有效成本记录，或已标记需要复测的 wave；
- GPU 枚举/ECC/row remap Gate 当前通过；
- 既有 Pile 下载任务不被停止或修改，成本 wave 选择无明显 I/O 竞争的窗口。

## 3. 实施步骤

### 3.1 固定成本测量边界

1. 定义端到端起点为样本映射已解析、正式数据读取开始。
2. 定义终点为 estimator、重复指标和原子结果写入完成。
3. 将模型首次加载和 reference 构建作为单独一次性成本，不混入每 repetition 主时间。
4. 单独记录数据等待、forward、backward、梯度读取/聚合、公式、统计和写盘时间。
5. 定义峰值显存同时报告框架 allocated 和 reserved；外部设备总占用作为辅助。
6. 报告处理 sequence、有效 token、backward 次数和跨卡通信字节。
7. 分别命名 `scientific_equal_sample_cost`（共享配对 runner）、`isolated_estimator_cost`（每种方法单独 fixed-state 运行）和 `online_training_incremental_cost`（保持 optimizer global batch 语义的在线增量）；禁止跨口径比较一个共同的 `1.25` 比值。

### 3.2 建立稳定测量环境

1. 按 GPU UUID/PCI 记录设备身份和健康状态。
2. 确认同卡没有其他计算进程。
3. 确认项目盘没有明显影响本 wave 的重 I/O；无法隔离时标记并另约复测。
4. 记录温度、功耗、时钟和驱动版本，识别降频或硬件错误；只观察，不设置 power limit、clock、persistence mode，不 reset、改驱动或使用 `sudo`。
5. 对每个配置先做不计入结果的 warm-up。
6. 随机化或交替方法测量顺序，避免缓存/温度总偏向一种方法。
7. 记录 `cost_io_quiescent` 及其证据；活动 curl/下载不被停止，I/O 非静默时科学结果可保留但成本记录无效。

### 3.3 复测等总样本预算成本

1. 选择预注册的 14M/31M、一个小 B 和一个大 B anchor。
2. raw、U、double 从同一基础梯度池计算，记录共享 gradient cost。
3. 分别记录三种公式和统计的增量成本。
4. 对每个 M 记录 U 的额外聚合和公式成本。
5. 重复多个独立测量轮次，报告中位数、分位数和区间。
6. 计算 time/estimate、time/token、peak memory 和结果字节。

### 3.4 测量部署语义下的成本

1. 为 raw、U、double 建立方法独立 anchor；每种方法单独启动、单独维护状态、随机化测量顺序，得到可归因的端到端时间与峰值显存。
2. 将方法独立 anchor 与共享梯度池的增量归因结果交叉验证；差异超阈值时以方法独立实测为准并解释缓存/共享状态。
3. 对 raw 记录普通训练 batch 可直接得到的增量成本。
4. 对 U 记录保留独立 microbatch 梯度、S1/S2、reducer 状态和延迟 DDP 同步增加的成本。
5. 对 double 分开记录科学等总预算 B、每因子 B（总 2B）和保持 optimizer global batch 不变时新增独立评价 batch 的在线成本；不得把 optimizer 实际只用 B/2 的设计称为等训练语义。
6. 将 reference 构建和离线校准成本按计划校准频率摊销，不能从主表中消失。
7. 估算 Stage 4/5 规模的每步增量显存和 A100-hours，但明确标为外推而非实测。

### 3.5 测量单卡与四卡系统一致性

1. 主统计实验以单张健康 A100 建立低通信基线。
2. 在健康四卡候选组上运行一个冻结 anchor 条件。
3. 验证四卡 estimator 数值仍通过 Stage 1 容差。
4. 记录 all-reduce S1/S2、统计单元计数和 barrier 时间。
5. 报告四卡吞吐、缩放效率和通信开销。
6. 若四卡健康不可用，不伪造成本；单卡统计只标记 provisional，G2.7a 和 Stage 2 最终退出均不能 complete，进入正式多卡训练前必须补齐该项。

### 3.6 建立精度—成本 Pareto 表

1. 将每个 B/M/method 的 Bias、Variance、描述性 MSE、主判据校正 NMSE、Spearman、Overlap 与成本合并。
2. 计算等 MSE 所需时间、等排序质量所需时间和等 token 预算的 MSE。
3. 标记被其他配置在统计与成本上同时支配的点。
4. 检查 U/double 的校正 NMSE、排序、时间和显存非劣边界。
5. 将一次性 reference 成本和持续在线成本分别展示。
6. 为后续阶段给出推荐 B/M、校准频率和预计资源区间。

### 3.7 验证容量与失败余量

1. 根据正式实测估算整个 Stage 2 的实际 A100-hours 与写入量。
2. 比较 pilot 预算和实际偏差，解释超支或节省。
3. 估算 Stage 4 在线累计需要的参数数组和 checkpoint 增量。
4. 设置后续运行的显存、磁盘、inode 和单步时间警戒值。
5. 记录 `ulimit -n=1024` 下数据加载和输出句柄峰值；若接近上限，优先减少并发/句柄生命周期，不擅自修改系统限制。

## 4. 产出

- 端到端 runtime 明细与重复测量表；
- allocated/reserved/device 峰值显存表；
- sequence/token、backward、通信和输出字节成本表；
- 共享科学预算、方法独立 fixed-state、总 2B 和在线训练增量成本口径；
- 随机顺序 method-only anchors、共享归因交叉验证和 `cost_io_quiescent` 记录；
- 单卡/四卡数值与缩放报告；
- 精度—时间、精度—显存 Pareto 表；
- Stage 4/5 资源外推和容量警戒值。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.7a 工程成本 Gate**：

- 主成本记录来自健康、空闲、`cost_io_quiescent=true` 的同类设备；
- 每个方法都有端到端、梯度、公式和写盘时间，不用 formula-only 时间代替总成本；
- 样本/token/backward/通信量完整，等预算与 2B 口径分离；
- 峰值显存口径一致且有多轮测量；
- Pareto 表与统计结果使用相同 run IDs；
- 每个方法有随机顺序 method-only anchor，峰值显存不从联合 runner 直接归因；
- 方法决策使用 `online_training_incremental_cost` 的冻结 `1.25` 边界，共享成本只作辅助；
- 四卡结果已补齐；未补齐时本 Gate 为 `blocked/provisional`，不能写成通过后再补；
- 后续在线累计的容量外推有实测 anchor 和不确定区间。

## 6. 后续依赖

- S2.10 将本任务的 Pareto 和非劣结果用于主估计器决策。
- S2.11 把四卡未完成、成本超界或容量风险写入后续阶段 Gate。
