# S2.5 配对估计器运行器与公平预算

## 1. 子任务目的

建立一次采样、一次梯度收集即可配对计算 raw、双采样和多个 M 的 U-statistic 的正式运行器。它必须把“总样本预算”“梯度计算成本”和“公式计算成本”分开记录，并支持失败恢复、流式统计和固定状态校验。

## 2. 前置条件

- G2.1 数学/固定状态接口通过；
- G2.2-dev repetition draw-to-sample-to-microbatch mapping schema 已冻结；
- reference schema、小型 fixture 与参数注册表可只读加载；正式 pilot/matrix freeze 前仍须通过 G2.2 和 G2.3；
- S2.1 已冻结公式、指标和 signed 输出规则。

## 3. 实施步骤

### 3.1 建立 repetition 生命周期

1. 由 model/checkpoint/B/repetition ID 生成唯一且稳定的运行单元 ID。
2. 从 manifest 读取 draw ID、sample ID、顺序、`M_max` 基础分组和双采样两半，不在 worker 内重新抽样或去重。
3. 加载固定 checkpoint 和 reference 句柄。
4. 保存运行前参数、buffer、模型/全局 RNG、sampling generator 起始状态、参数注册表和样本映射摘要。
5. 把 staging 固定到 `$DATA_ROOT/tmp/stage2/<run-id>/<unit-id>`，锁/租约/进度固定到 `$DATA_ROOT/operations/stage2/<run-id>`，日志固定到 `$DATA_ROOT/runs/stage2/<run-id>`；不得使用仓库、根盘或系统 `/tmp`。
6. 已有完整结果只做哈希验证；已有失败结果保留并使用新 attempt ID 显式重试。

### 3.2 产生基础 microbatch 梯度

1. 检查 `B % M_max == 0`。
2. 按 manifest 读取一个基础 microbatch 的完整 sequence。
3. 计算统一 global-token/sequence loss reduction 下的 mean gradient。
4. 在任何 DDP all-reduce 前读取真实尺度 FP32 gradient。
5. 记录该基础 microbatch 的有效 sequence、有效 token、loss 和 gradient norm。
6. 将梯度加入全局 S1、raw 统计、双采样半区统计和各嵌套 M 的组内累加器。
7. 完成一个基础 microbatch 后释放临时梯度，不把完整梯度写盘。
8. 重复到 `M_max` 个基础单元全部完成。

### 3.3 从同一梯度池计算 raw

1. 用全部 B 个样本对应的总 S1 得到平均梯度。
2. 逐坐标计算 `raw = mean_gradient^2`。
3. 保留 signed 数组（raw 理论上非负），不做额外归一化。
4. 记录公式时间，与前向/反向梯度时间分开。
5. 在估计向量仍在内存时，分别对 `bias_ref`、`cross_ref` 和 `rank_ref` 计算预注册 repetition 级误差/排序/top-k 摘要。

### 3.4 计算嵌套 M 的 U-statistic

1. 对 `M_max` 使用基础 microbatch mean gradients 计算 S1 和 S2。
2. 按固定嵌套分组把相邻基础单元的加权总和合并为较粗 microbatch。
3. 对每个 M 单独计算正确尺度的 microbatch mean 和 S2。
4. 使用 \((S1^2-S2)/(M(M-1))\) 得到 signed U。
5. 对不等权条件使用冻结的加权去对角公式；不得混用普通分母。
6. 记录每个 M 的负坐标比例、负质量、数值消减诊断和公式时间。
7. 在向量仍在内存时，对每个 reference 版本计算每个 M 的排序、top-k 和误差摘要，并计算预注册相邻/嵌套 M 配对指标。

### 3.5 计算等总预算双采样

1. 用 manifest 中预先固定的两个不相交 draw ID 半区构造 A 与 B'。
2. 分别计算两半的 mean gradient。
3. 逐坐标计算 `double = mean_A * mean_B`。
4. 验证 A/B' 各含 B/2 个独立 draw，并记录 sample ID 偶然重复数和各自有效 token 数。
5. 验证 `M=2` U 与 double 的逐坐标最大绝对/相对差落入数值容差。
6. 保存 signed double，不进行正值截断或最负/近零重排序。
7. 在向量仍在内存时，对每个 reference 版本计算 repetition 级排序、top-k 和误差摘要。

### 3.6 分离辅助成本口径

1. 主比较只使用总预算 B 的 raw/U/double。
2. 可选辅助实验允许双采样每个因子各用 B、总预算 2B。
3. 辅助结果必须使用不同方法名和结果列，不能进入主等预算 Gate。
4. 等 wall-clock 重采样只在主矩阵完成后执行，并报告实际处理样本/token 数。
5. 公式计算成本、共享梯度成本和独立额外梯度成本分别计时。

### 3.7 流式累计跨 repetition 统计

1. 每个 repetition 发布一个不可变 sufficient-stat shard，包含 method/M 的 count、sum、sum-of-squares、相对各 reference 的 error sums、正负质量和 scope 聚合。
2. 每个 shard 绑定 unit ID、attempt ID、输入哈希和唯一键；重试不得覆盖旧 shard。
3. 由单写入 reducer 按 canonical unit ID 顺序、幂等合并 shards，拒绝重复 attempt，避免并发原地更新同一个 streaming M2。
4. Reducer 同时输出逐参数 count/mean/M2、squared/absolute error、正/零/负计数及 tensor/layer/module 汇总。
5. 保存每个 repetition 对三类 reference 的标量排序、top-k、误差、跨 M 配对指标、成本和样本映射哈希。
6. 六个确认性主 cells 的 `B_primary/M_primary` 必须全部属于 anchor，并保存完整逐参数 count/mean/M2/error-sum 汇总数组；其他条件保存足以重建全部预注册声明的流式聚合和诊断坐标。
7. 不长期写入单个 repetition 的完整参数梯度或估计向量，除非属于预注册数值诊断单元；任何无法由 sufficient stats 重建的主指标必须在向量释放前落盘为标量。

### 3.8 检查数值与状态完整性

1. 每个 estimator 输出后执行 NaN/Inf 检查。
2. 检查 raw、U、double 的参数数量、顺序和 dtype 与注册表一致。
3. 检查 U 未发生意外 clamp；负值为零且理论上不应为零的条件触发警报。
4. 检查 M 改变时 mean gradient 保持相同。
5. 检查 repetition 完成后参数、buffer 和模型/全局 RNG 摘要与运行前一致；sampling generator/worker RNG 则核对冻结消费量、结束状态和可重放性。
6. 检查 reference 文件和采样 manifest 的哈希没有变化。

### 3.9 实现失败恢复和原子发布

1. 基础 microbatch 或 repetition 失败时写入独立 `FAILURE` 记录和最后有效进度。
2. 重试使用相同样本映射、checkpoint 和 seed，不生成新样本替代失败样本。
3. 从最近完整基础分组或 repetition 恢复，恢复后重算该原子单元。
4. 完成后先校验行数、数组 shape、有限性和哈希，再从同一文件系统 staging 原子发布到 `$DATA_ROOT/results/stage2/raw/<run-id>`。
5. 原始失败记录与最终成功记录并列保留，不覆盖历史。

## 4. 产出

- 配对 estimator runner；
- 嵌套 M 梯度聚合器；
- raw、double、等权/加权 U 的 signed 输出；
- repetition 级指标与成本表；
- parameter/tensor/layer/module 流式统计；
- `M=2` 等价和状态不变报告；
- 可恢复进度、失败记录和原子发布 manifest；
- 不可变 sufficient-stat shards 与确定性单写入 reducer；
- 主等预算与辅助 2B/等 wall-clock 的独立 schema。

## 5. 核验标准与 Gate

满足以下条件才通过 **G2.4a 运行器 Gate**：

- raw、多个 M 的 U 和 double 确认使用同一总样本池 B；
- double 两半的 draw IDs 互斥且各为 B/2，抽样规则保持独立有放回语义；
- `M=2` U 与 double 的逐坐标差小于 Stage 1 数值容差；
- 所有 M 的完整 batch mean gradient 一致；
- signed U/double 负值未被截断；
- 梯度时间、公式时间、样本/token 和峰值显存字段齐全；
- 状态摘要运行前后一致；
- 失败重试能在相同样本上重放，原子发布不会产生半成品成功目录；
- 流式统计与保留小规模完整数组的离线重算一致。
- 并发/重试不会重复计数；不同 worker 顺序归并产生相同 reducer 输出哈希。
- 三类 reference、全部主 top-k/排序和跨 M 配对摘要在向量释放前已完整保存。

## 6. 后续依赖

- S2.6 用本运行器做 pilot，不另写临时 estimator 实现。
- S2.7 只接受通过本 Gate 的 runner commit。
- S2.8 只读取原子发布的 schema，不从日志文本猜测缺失字段。
