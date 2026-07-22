# S1.4 损失 reduction 与梯度尺度

## 1. 目的

证明逐样本、逐 microbatch、完整 batch、梯度累积和后续 DDP 使用的是同一个标量损失及同一平均梯度。此处只验证 loss 和 gradient 的尺度，不计算重要性结论。

Stage 1 的因果语言模型主契约为：所有非忽略目标 token 的损失总和除以全局有效目标 token 总数。

## 2. 前置 Gate

- `G1-CONTRACT` 已固定主 loss reduction。
- `G1-REGISTRY` 与 `G1-ORACLE` 已通过。
- fixture 记录了样本 ID、目标 mask、microbatch 边界和有效 token 数。

## 3. 执行步骤

### 3.1 建立 loss adapter

1. 从模型输出取得逐目标 token 的未约简损失。
2. 应用固定的目标 mask。
3. 计算本 microbatch 的 loss numerator。
4. 计算本 microbatch 的有效目标 token 数。
5. 由 numerator 和 count 计算 local mean loss。
6. 同时返回 numerator、count 和 mean，避免从 mean 反推计数。
7. 对零有效 token 输入执行契约规定的拒绝或跳过逻辑。
8. 在日志中保存 reduction 名称和有效计数。
9. 禁止 provider 的隐式默认 reduction 绕过 adapter。

### 3.2 验证逐样本与完整 batch

1. 在解析模型上计算每个样本的标量损失与梯度。
2. 按有效统计单元权重重构完整 batch loss。
3. 按相同权重重构完整 batch gradient。
4. 用一次完整 batch 前向/反向得到参考 loss 和 gradient。
5. 比较逐样本重构与完整 batch 结果。
6. 对每个参数张量分别记录误差。
7. 对接近零坐标使用绝对误差规则。

### 3.3 验证等权 microbatch

1. 选择每个 microbatch 有效 token 数相同的 fixture。
2. 分别计算每个 local mean gradient。
3. 计算 microbatch gradient 的算术平均。
4. 与一次完整 batch mean gradient 比较。
5. 改变 microbatch 顺序后重复。
6. 改变 `M` 但保持 global batch 不变后重复。
7. 记录 loss、gradient、样本 ID 和 `M`。

### 3.4 验证有效 token 加权 microbatch

1. 选择至少两个有效 token 数不同的 microbatch。
2. 分别计算 local token-mean gradient。
3. 按有效 token 数计算加权 mean gradient。
4. 与完整 batch 的 global token-mean gradient 比较。
5. 用逐 token oracle 再独立重构一次。
6. 计算故意等权聚合的负对照。
7. 验证负对照与参考产生可检测差异。
8. 验证权重归一化之和为 1。
9. 验证仅一个正权重统计单元不能进入加权 U 路径。

### 3.5 验证 sum/mean 与梯度累积

1. 用同一输入分别构造显式 sum loss 和 mean loss。
2. 验证两者梯度只相差已知计数因子。
3. 将一个 global batch 划分成多个 local backward。
4. 每次 backward 前清理临时梯度容器。
5. 读取每次 local mean gradient，而不是累积后的 `.grad`。
6. 独立聚合这些 local gradients。
7. 与一次完整 batch backward 比较。
8. 重复不同 accumulation step 数。
9. 检查上一 optimizer step 的梯度不会污染下一步。

### 3.6 验证随机状态边界

1. 在等价性测试中关闭 dropout 或固定同一随机 mask。
2. 记录模型 train/eval 状态。
3. 记录 CPU 与设备 RNG 状态摘要。
4. 验证重复运行得到同一 loss 和 gradient。
5. 另设 dropout 开启的 smoke，仅验证不同 microbatch 随机流独立，不用于精确等价 Gate。

## 4. 产出

- loss adapter 与 reduction schema。
- 逐样本、等权 microbatch、加权 microbatch 和完整 batch 的 loss/gradient 表。
- 不同 `M` 与 accumulation 切分的误差矩阵。
- 故意等权的负对照结果。
- 每个参数张量的最大绝对误差与归一化 L2 误差。
- `G1-GRAD` 机器可读报告。

## 5. 可视化

- full-batch gradient 与 microbatch 重构 gradient 的逐坐标散点图，带 `y=x`。
- 等权、正确加权和故意等权三条路径的误差对比图。
- `parameter tensor × route` 的最大尺度化误差热力图。
- 不同 accumulation 切分的 normalized L2 误差曲线。

每张图必须同时提供底层 CSV；图只用于定位，Gate 以逐张量数值表为准。

## 6. 核验标准

- FP64 解析 fixture 的逐样本重构、microbatch 重构和完整 batch 通过 `T64_ORACLE`。
- 单进程 FP32 的对应比较通过 `T32_SINGLE`。
- 等权 fixture 的权重必须精确为 `1/M`。
- 加权 fixture 的权重、有效 token 数和分母必须完全一致。
- 正确加权路径通过 Gate；故意等权负对照至少在一个预先指定坐标或张量上超出 `T32_SINGLE`，证明测试具有检错能力。
- sum/mean 梯度比与有效计数因子通过对应容差。
- 所有梯度有限；零有效 token 不产生 NaN、Inf 或静默零梯度。
- 每个 global batch 的样本集合与顺序可重构，不允许重复或遗漏而未记录。

## 7. Gate 与后续依赖

- `G1-GRAD` 通过后，S1.5 才能使用这些梯度验证估计器。
- 任何 loss adapter、padding、mask 或有效 token 规则变化都必须重跑本 Gate。
- S1.8 的 DDP 等价性以本子任务的单进程完整 batch 结果为唯一参考。
