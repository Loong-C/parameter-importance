# S1.9 精度、loss scaling、裁剪与优化器边界

## 1. 目的

隔离混合精度、loss scaling、non-finite skip、全局梯度裁剪和 AdamW 状态对数值语义的影响。代码正确性的主参考仍是 FP32；BF16 只证明真实执行链可运行且没有明显尺度损坏。

## 2. 前置 Gate

- `G1-SINGLE` 已通过。
- 分布式路径测试时还要求 `G1-DDP` 的基础归约已通过。
- 正确性运行的确定性设置在任何 CUDA 初始化前建立。

## 3. 执行步骤

### 3.1 固定确定性与 kernel 路径

1. 在创建 CUDA context 前设置并记录确定性策略。
2. 固定 attention/SDPA 为已知可复现的 correctness 路径。
3. 禁用会改变等价性测试随机状态的自动优化。
4. 按 allowlist 记录 PyTorch、CUDA、cuDNN、NCCL 和确定性相关环境变量摘要；不 dump 全环境或任何凭据/传输端点。
5. 用同一配置重复一次短运行，验证可重复性。
6. 将更快但不确定的 kernel 留到后续性能实验，不用于 Stage 1 Gate。

### 3.2 验证 FP32 充分统计量

1. 检查 local gradient 在统计前转换为 FP32。
2. 检查 `S1/S2` 或加权充分统计量使用 FP32 累加。
3. 检查长期累计数组使用契约规定 dtype。
4. 用大/小混合梯度 fixture 检查溢出与下溢风险。
5. 记录所有非有限值检查点。
6. 验证模型原始参数 dtype 不被统计容器静默改变。

### 3.3 验证 scale/unscale

1. 固定相同的 autocast 前向与输入。
2. 计算未缩放参考 gradient。
3. 施加已知 loss scale。
4. 读取每个 scaled local gradient 后，把统计副本按本 step 固定 scale 做张量级真实尺度恢复，不改变 optimizer 的 `unscale_` 状态机。
5. 比较 unscale 后 gradient 与参考。
6. 比较 `S1/S2` 与参考。
7. 比较 raw 和 U 与参考。
8. 改变多个 loss scale 后重复。
9. 保留一个故意未 unscale 的负对照，验证测试能发现尺度平方错误。
10. 确认负对照不进入正式结果。
11. 验证一阶充分统计量按 scale 恢复、二阶充分统计量按 `scale^2` 恢复。
12. 聚合后只对交给 optimizer 的 scaled global mean gradient 执行一次正式 step 级 `unscale_`，不在 microbatch 循环中重复调用。
13. 验证全局 `found_inf` 决策、`scaler.step` 与 `scaler.update` 在每个 training attempt 各发生恰好一次；optimizer 实际更新次数由 execute/skip 决策决定。

### 3.4 验证 non-finite 与跨 rank skip

1. 注入可控的 Inf 或 NaN gradient。
2. 在每 rank 检测本地 non-finite 状态。
3. 聚合为全局 skip 决策。
4. 把全局 `found_inf`/skip 决策同步到每 rank 的正式 scaler inf-check 状态。
5. 验证所有 rank 对同一步作相同决策。
6. 在 skip 路径每 rank 调用一次 `scaler.step`，并验证 optimizer 实际更新次数为零。
7. 调用一次 `scaler.update`，逐 rank 比较 scale 与 growth tracker。
8. 验证参数、optimizer state、scheduler、successful step、重要性和 movement 不前进。
9. 验证数据游标、RNG、attempt、skip count 与 scaler 按契约推进。
10. 验证 skip 次数和原因只记录一次全局事件。
11. 验证下一正常 step 的 rank 状态仍一致。
12. 验证 step-local 分数暂存区被丢弃，任何长期数组都没有先提交后回滚的半更新。

### 3.5 验证全局梯度裁剪

1. 构造 global mean norm 低于阈值的 fixture。
2. 验证 clip factor 精确为 1。
3. 构造 global mean norm 高于阈值的 fixture。
4. 独立计算解析 clip factor。
5. 在所有 microbatch 聚合后计算实现 clip factor。
6. 比较解析值与实现值。
7. 验证 optimizer 使用的 gradient norm 不超过阈值及容差。
8. 逐张量直接验证 `U_clipped ≈ s_t * U_unclipped`，不是乘 `s_t^2`。
9. 验证 raw 未裁剪字段保持不变。
10. 构造逐 microbatch 先裁剪的负对照，验证它与正确结果产生差异。
11. 比值仅作诊断图：只纳入 `|U_unclipped| >= 100 * atol * a_q` 的坐标；该阈值和 `a_q` 在运行前写入契约。
12. 验证同批随机 `clip_factor` 输出字段固定为
    `local_gradient_space_importance_u_clipped`，并带
    `unbiasedness_claim=none`、`clip_source=same_batch_global_mean`；不得继承未裁剪
    `local_gradient_space_importance_u` 的无偏性声明。

### 3.6 验证 AdamW 解释边界

1. 在固定参数状态与固定 gradient 下改变 AdamW moment 历史。
2. 验证 local U 只由当前 gradient、学习率和 clip factor决定。
3. 验证实际参数更新随 moment 历史变化。
4. 验证两者写入不同字段并具有不同说明。
5. 启用非零 weight decay，验证 U 不混入 decoupled decay。
6. 验证 data movement、weight-decay movement 和 total movement 分别记录。
7. 验证报告不把 actual-update diagnostic 标记为 U 的真值。

### 3.7 BF16 执行链 smoke

1. 用与 FP32 相同的固定样本运行 BF16 autocast。
2. 验证前向、反向、充分统计量、分数和 optimizer step 无 NaN/Inf。
3. 验证统计容器仍为 FP32。
4. 报告 BF16 与 FP32 mean gradient 的 cosine、norm ratio 和逐张量误差。
5. 报告 BF16 raw/U 与 FP32 的差异，但不据此判定公式相等。
6. 验证 BF16 运行可保存和恢复完整状态。

## 4. 产出

- 确定性与 kernel 路径环境摘要。
- FP32 累加 dtype 审计报告。
- scale/unscale 多倍率对照表与负对照。
- non-finite 全局 skip 报告。
- clip factor、clip 前后 norm 和单因子验证表。
- AdamW moment/weight-decay 边界报告。
- BF16 与 FP32 数值质量报告。
- `G1-NUMERIC` 机器可读报告。

## 5. 可视化

- loss scale 倍率与 unscale 后误差曲线。
- clip 前 norm、clip factor 和 clip 后 norm 对照图。
- `U_clipped` 与 `s_t * U_unclipped` 的 identity 图；另附过滤近零坐标后的比值诊断图。
- BF16 与 FP32 mean gradient 的 layer/module 误差热力图。
- skip-step 前后参数与累计状态的零差异表。

## 6. 核验标准

- 相同 autocast 前向下，scale/unscale 前后 gradient、充分统计量、raw 和 U 通过 `T_AMP_SCALE`。
- 故意未 unscale 负对照必须超出 `T_AMP_SCALE`。
- 同一 attempt 内 scale 固定，正式 unscale、`scaler.step`、`scaler.update` 各执行一次；finite 时 optimizer 更新一次，skip 时更新零次，统计开关不改变 scale/growth tracker 轨迹。
- 所有充分统计量为 FP32，所有正式结果无 NaN/Inf。
- non-finite attempt 的所有 rank 一致 skip；参数和长期累计状态不前进，而数据/RNG/attempt/skip count/scale/growth tracker 按冻结契约推进。
- clip factor 与解析值通过 `T64_ORACLE` 或 `T32_SINGLE`。
- `U_clipped` 与 `s_t * U_unclipped` 逐张量通过对应容差；比值图不作为主要 Gate。
- optimizer 接收的 global gradient norm 满足阈值与 FP32 容差。
- 锁定 correctness kernel、相同硬件和相同输入的重复短运行必须逐步 bitwise 一致；只有运行前具名登记的非确定性算子才允许使用对应容差，不能事后笼统降级。
- BF16 smoke 的硬 Gate 是执行完整、无非有限值、FP32 统计容器和状态可恢复；与 FP32 的数值差异完整报告但不用于放宽 FP32 Gate。

## 7. Gate 与后续依赖

- `G1-NUMERIC` 通过后，S1.10 才能对含 precision/skip 状态的 checkpoint 做恢复验收。
- 混合精度、GradScaler、clip 或 AdamW bridge 变化时必须重跑本 Gate、`G1-STEP` 和受影响的 DDP Gate。
- 后续性能优化不能替换 correctness kernel 路径，除非新路径重新通过本阶段 Gate。
