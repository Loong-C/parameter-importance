# S1.3 确定性 fixture 与独立 oracle

## 1. 目的

建立不依赖正式训练循环的真值来源。测试不能只用“同一段实现的两个封装相互比较”，而要同时包含解析梯度、有限差分、显式成对手算和保存后离线重算，才能发现共同错误。

## 2. 前置 Gate

- `G1-CONTRACT` 已冻结损失、符号和容差。
- `G1-REGISTRY` 已冻结参数坐标和状态 shape。

## 3. Fixture 层级

### 3.1 纯梯度矩阵 fixture

不依赖 autograd，直接给出小型 FP64 梯度矩阵，至少覆盖：

- 全零梯度；
- 所有 microbatch 梯度相同；
- 正负交替并使 U-statistic 为负；
- 不同数量的 microbatch，包括边界 `M=1` 与 `M=2`；
- 大小差异显著但仍有限的坐标；
- microbatch 排列的多种顺序；
- 等权和不等有效 token 权重；
- 一个参数张量与多个不同 shape 参数张量。

### 3.2 解析模型 fixture

使用可手写梯度与端点损失差的线性或二次标量模型：

- 参数量足够小，可逐坐标列出解析结果；
- 使用 FP64 CPU 作为最高精度 oracle；
- 支持逐样本、逐 microbatch 与完整 batch 三种损失；
- 支持至少一次已知 SGD 更新；
- 支持有/无 weight decay 的位移分解测试；
- 不含 dropout、BatchNorm 或跨样本状态。

### 3.3 Tiny Transformer fixture

建立远小于 Pythia-14M 的确定性 Transformer：

- 固定架构、初始化 seed、token 序列和 attention mask；
- 关闭随机层或固定一致随机状态；
- 同时提供等长序列和不同有效 token 数的 padding fixture；
- 保存完整逐样本与逐 microbatch 梯度；
- 保留一个参数子集用于有限差分抽查；
- 参数规模允许在本机 CPU 完成完整 oracle 测试。

### 3.4 Pythia-14M 固定 fixture

- 固定 step0 revision 和 tokenizer revision。
- 固定 Pile 已验证前缀中的样本 ID、token 数组哈希和顺序。
- 固定模型模式、随机状态、dtype 和设备。
- 固定一个仅用于完整梯度 dump 的 debug step。
- 不把 14M 结果当作解析真值；它用于验证真实模型接口和坐标规模。

## 4. 执行步骤

### 4.1 生成并冻结输入

1. 为每类 fixture 分配不可变 ID。
2. 保存生成规则和 seed。
3. 保存原始样本 ID、token 数组、标签或目标 mask。
4. 保存每个 microbatch 的边界与有效统计单元数。
5. 保存参数初值和 registry hash。
6. 计算每个输入文件的 SHA-256。
7. 生成 fixture manifest。
8. 验证同一 manifest 可在本机与服务器读取。

### 4.2 计算解析与 autograd 梯度

1. 为解析模型逐样本写出闭式梯度 oracle。
2. 独立计算完整 batch 的闭式平均梯度。
3. 用 autograd 计算相同逐样本梯度。
4. 用 autograd 计算相同 microbatch mean gradient。
5. 用一次完整 batch backward 计算 mean gradient。
6. 按 registry 顺序保存所有结果。
7. 在 FP64 下逐坐标比较解析与 autograd。
8. 在 FP32 下重复同一比较并使用 `T32_SINGLE`。

### 4.3 有限差分抽查

1. 从每种 module type 选择至少一个标量坐标。
2. 包含 bias、weight、embedding 或等价不同参数角色。
3. 为每个坐标记录扰动尺度和选择依据。
4. 用中心差分计算固定标量损失导数。
5. 比较有限差分、解析梯度和 autograd 梯度。
6. 对接近零导数单独报告绝对误差。
7. 不用有限差分替代全部参数的 autograd/解析对照。

### 4.4 生成离线估计器 oracle

1. 从保存的逐样本梯度直接计算 full-batch mean。
2. 从保存的 microbatch 梯度直接计算 raw。
3. 用两个独立样本集合直接计算双采样乘积。
4. 用显式 ordered pair 双循环计算逐样本 U-statistic。
5. 用显式 microbatch ordered pair 双循环计算等权 U-statistic。
6. 用显式 `m != n` 的 weighted cross-microbatch ordered pairs 计算加权 U-statistic。
7. 明确逐 token oracle 只验证 token-mean loss/gradient；它不作为 weighted microbatch U 的有限样本真值。
8. 若未来实现 token-level U，使用独立字段名和 requirement ID，不与 weighted microbatch U 混用。
9. 将离线计算放在与正式 estimator kernel 不同的模块或脚本中。
10. 保存 FP64 oracle 数组和小型 CSV 摘要。

### 4.5 解析路径 smoke test

1. 用常梯度损失构造固定起点和终点。
2. 比较左端点、右端点、中点、梯形、Simpson 和 Gauss-Legendre 的结果。
3. 用二次损失构造线性参数路径。
4. 为二次 fixture 推导每个坐标的闭式路径贡献。
5. 逐坐标比较梯形法贡献与闭式真值，不能只比较贡献总和。
6. 验证全部坐标贡献总和等于解析端点损失下降。
7. 固定全部节点使用同一个损失和随机状态。
8. 验证 smoke test 结束后模型参数与 optimizer state 恢复。
9. 仅把参数/层排序作为诊断；真实模型的排序稳定性留到 Stage 3。
10. 不在本阶段改变节点数并评价真实模型误差。

### 4.6 纯噪声小型性质测试

1. 把基础随机变量定义为相互独立的单 token/单样本高斯梯度 `h_{m,i} ~ N(0, sigma^2)`，从而同时冻结均值、方差与第四矩；分布身份写入 fixture manifest。
2. 固定每个 microbatch 的样本数 `b`、microbatch 数 `M` 和总样本数 `B=M*b`。
3. 先生成 `h_{m,i}`，再显式取均值得到 microbatch gradient `g_m`，不直接混用两层方差符号。
4. 使用预注册重复数，至少覆盖多个 `B` 和 `M`。
5. 计算 raw 与 U-statistic 的重复均值。
6. 利用高斯第四矩，从 `sigma^2`、`b`、`M` 和重复数推导 raw 理论均值 `eta*sigma^2/B`、U 理论均值 0 与两个估计量的解析标准误。
7. 标准误只来自生成分布的解析参数，不从待测实现输出反推。
8. 验证 U 均值落在预注册标准误带内。
9. 验证 raw 均值为正且与理论值落在预注册标准误带内。
10. 将该结果仅标为代码性质 smoke，不作 Stage 2 的真实梯度统计结论。

## 5. 产出

- 纯梯度矩阵 fixture 与预期结果。
- FP64 解析模型和闭式梯度表。
- Tiny Transformer 固定输入、参数和完整梯度 dump。
- Pythia-14M 固定样本清单与 token 哈希。
- 有限差分抽查报告。
- 独立 FP64 raw/double/U oracle 数组。
- 常梯度与二次损失路径 smoke 报告。
- 零均值纯噪声性质测试 JSON/CSV。
- 包含全部文件 SHA-256 的 fixture manifest。

## 6. 可视化

- autograd 与 FP64 解析梯度的 identity-line 散点图及最差坐标表。
- 有限差分扰动尺度与误差曲线，用于发现步长过大或消减误差。
- 二次 fixture 的逐坐标闭式贡献与实现贡献 identity 图，以及完备性残差表。
- 纯噪声 raw/U 重复均值、理论均值与预注册标准误带图。

## 7. 核验标准

- FP64 解析梯度与 autograd 通过 `T64_ORACLE`。
- FP32 autograd 与 FP64 下采样参考通过 `T32_SINGLE`。
- 每个有限差分抽查坐标有明确扰动尺度，误差在预注册阈值内。
- 完整 batch 梯度能由逐样本梯度独立重构。
- 显式 ordered-pair oracle 不调用正式流式 U 实现。
- 常梯度路径的所有求积方法通过 `T64_ORACLE` 一致。
- 二次损失的每个坐标贡献通过 `T64_ORACLE`，且梯形法相对完备性残差不超过 `1e-10`。
- 纯噪声 U 的重复均值绝对值不超过 5 个预注册解析标准误；raw 与理论正偏差异不超过 5 个预注册解析标准误，不得以待测输出的经验方差替代。
- fixture 重新生成时，输入和参数哈希必须一致；不一致时生成新 fixture 版本。

## 8. Gate 与后续依赖

- `G1-ORACLE` 通过后，S1.4 与 S1.5 才能把这些数组作为验收真值。
- 任何 oracle 若与正式实现共享核心公式代码，则对应 Gate 无效，必须建立真正独立路径。
- 解析路径 smoke 只解锁 Stage 3 的进一步计划，不代表多节点数值近似已验证。
