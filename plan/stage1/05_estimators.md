# S1.5 raw、双采样与 U-statistic 内核

## 1. 目的

把估计器实现为与模型、optimizer、DDP 和文件系统无关的纯张量内核，并用独立 FP64 oracle 验证公式。内核只负责数学计算和输入校验，不负责采样、截断、累计或科学解释。

## 2. 前置 Gate

- `G1-ORACLE` 已提供独立显式手算数组。
- `G1-GRAD` 已确认输入是正确的 local mean gradient。
- `G1-CONTRACT` 已冻结 raw、double、U 与 clip-adjusted 字段名称。

## 3. 执行步骤

### 3.1 实现 raw 内核

1. 接收 global mean gradient 和每参数实际学习率。
2. 逐元素计算不乘学习率的 `raw_core = mean_grad^2`。
3. 将 `raw_core` 逐参数张量乘实际学习率。
4. 输出到 `local_gradient_space_importance_raw`。
5. 将 clip-adjusted raw 作为独立可选函数或字段。
6. 禁止默认把 clip factor 混入 raw。
7. 验证多参数组学习率逐张量映射。
8. 验证 `raw_core` 与公开 score 分别通过独立 Gate。
9. 验证结果 dtype、shape 和 registry 一致。

### 3.2 实现双采样内核

1. 接收两个独立 batch 的 mean gradient。
2. 校验两组 registry、shape、dtype 与学习率映射一致。
3. 逐元素计算 `eta * mean_grad_A * mean_grad_B`。
4. 不在内核中把 A/B 解释为 probe loss。
5. 由采样层记录两个独立有放回 RNG stream 或等价 product-sampling provenance 和样本 ID。
6. Stage 1 只硬检验两个乘法因子没有误用同一个 batch 对象，且结果与手算一致。
7. 不把样本 ID 偶然重合判为不独立，也不把无放回不重叠切分误称为独立抽样。
8. 验证交换 A/B 后局部平方特例结果一致。
9. 对共享 sampler state 或重复复用同一 batch 对象的错误配置 fail-fast；严格采样设计的统计验收留给 Stage 2。

### 3.3 实现显式成对参考版

1. 接收完整 microbatch gradient 列表。
2. 显式枚举所有 `m != n` ordered pairs。
3. 逐元素累加 pair product。
4. 除以 `M(M-1)`。
5. 单独实现 unordered pair 乘 2 的参考路径。
6. 比较 ordered 与 unordered 两种结果。
7. 将该实现限制为小 fixture 测试，不进入正式训练热路径。

### 3.4 实现等权流式 U-statistic

1. 为每个参数张量建立 FP32 `S1`。
2. 为每个参数张量建立 FP32 `S2`。
3. 每个 local mean gradient 只累计一次到 `S1`。
4. 每个 local mean gradient 的逐元素平方只累计一次到 `S2`。
5. 精确累计统计单元数量 `M`。
6. 在 `M < 2` 时明确失败。
7. 计算 `(S1^2 - S2) / [M(M-1)]`。
8. 先输出未乘学习率的 `u_core` 供独立 Gate。
9. 逐张量乘以实际学习率得到公开 U 字段。
10. 保留负值，不执行 clamp。
11. 计算完成后输出充分统计量摘要，便于离线复算。

### 3.5 实现有效 token 加权 U-statistic

0. 冻结并记录 `statistical_unit`、`weight_unit`、`sampling_design`、
   `weights_exogenous` 与 `common_mean_assumption`；任一统计假设未声明时不得附加
   无偏性声明。
1. 对每个 microbatch 读取正的有效 token 数 `b_m`。
2. 累计 `G1 = sum(b_m * g_m)`。
3. 累计 `G2 = sum(b_m^2 * g_m^2)`。
4. 累计 `N1 = sum(b_m)`。
5. 累计 `N2 = sum(b_m^2)`。
6. 验证 `N1^2 - N2 > 0`。
7. 计算 `(G1^2 - G2) / (N1^2 - N2)`。
8. 先输出未乘学习率的 weighted `u_core` 供独立 Gate。
9. 逐张量乘以实际学习率得到公开 U 字段。
10. 在所有 `b_m` 相同时与等权公式比较。
11. 记录每个充分统计量，避免只保存最终分数。
12. 权重若由同批 loss、gradient 或事后筛选产生，只能输出 plug-in/描述性结果。

### 3.6 建立公共输入防线

1. 校验所有输入张量与 registry 一一对应。
2. 校验 shape 完全一致，禁止隐式广播。
3. 校验输入已恢复真实梯度尺度。
4. 校验累计 dtype 至少为 FP32。
5. 校验梯度、学习率、权重和 clip factor 有限。
6. 校验学习率非负且来自明确参数组。
7. 校验权重为正且统计单元数满足公式。
8. 校验同一调用中设备与分布式元数据一致。
9. 让错误指出具体参数张量和条件。

## 4. 测试矩阵

1. 显式 ordered pair、unordered pair 和流式公式三者比较。
2. 所有 microbatch 梯度相同，验证 U 等于 `eta * g^2`。
3. `g1=a, g2=-a`，验证单步 U 为负。
4. 全零梯度，验证所有估计器为零。
5. `M=1`，验证明确拒绝且不产生 NaN。
6. `M=2`，验证最小合法边界。
7. 将这两个 microbatch 作为双采样两半，验证 U 与 double 逐坐标一致。
8. 多次排列 microbatch，验证结果不变。
9. 等权与加权公式在等计数时退化一致。
10. 不等 token 计数时与独立显式 weighted cross-microbatch ordered-pair oracle 比较。
11. 多参数组学习率 fixture 逐张量比较。
12. 对非有限输入、shape mismatch 和 registry mismatch 做失败测试。
13. 对 raw 未裁剪与 raw clipped 字段做防混淆测试。
14. 对双采样 A/B 独立元数据做正例与负例测试。

## 5. 产出

- raw、可选 raw-clipped、double、显式 U、等权流式 U 和加权流式 U 内核。
- estimator 输入/输出 schema。
- FP64 逐坐标误差表。
- 等权/加权退化一致性报告。
- 负值保留、排列不变和非法边界测试报告。
- `G1-EST` 机器可读报告。

## 6. 可视化

- 显式 ordered-pair、unordered-pair 与流式 `u_core` 的 identity-line 散点图。
- 等计数退化与不等计数 weighted cross-microbatch oracle 的逐张量误差热力图。
- raw core/U core 与乘学习率后公开 score 的缩放关系图。
- 构造负 U case 的逐坐标值表，确认负值未被 clamp。

## 7. 核验标准

- 纯 FP64 fixture 上，显式 ordered pair、unordered pair、流式等权和独立 oracle 全部通过 `T64_ORACLE`。
- FP32 fixture 上通过 `T32_SINGLE`。
- 等计数时，加权公式与等权公式通过对应 dtype 容差。
- 负值 case 的 U 必须严格小于零，核心 API 返回值不得被截断。
- `M=1` 或加权分母非正必须明确失败，失败输出不得伪装成零分数。
- `M=2` 时，U 与同一两半样本构成的 double 通过对应 dtype 容差。
- raw 字段在 clip factor 改变时保持不变；raw-clipped 字段按一个 clip factor 变化。
- 双采样 fixture 的两个因子未复用同一 batch 对象，采样 provenance 满足已冻结的 product-sampling 设计，内核结果与手算逐元素一致；样本 ID 是否偶然重合不单独决定通过与否。
- `raw_core`、`u_core` 与乘学习率后的公开字段分别通过对应容差，不能只 Gate 最终小尺度 score。
- 任一参数张量失败都使 `G1-EST` 失败。

## 8. Gate 与后续依赖

- `G1-EST` 通过后，S1.6 才能把估计器接入训练 step。
- 对估计器核心公式、输入 dtype、加权规则或字段名的任何修改都要求重跑本 Gate 和全部下游 Gate。
- Stage 2 的 bias/variance 实验只能调用本 Gate 冻结的 estimator 版本。
