# S1.6 训练步集成、累计量与比较基线

## 1. 目的

把已验证的纯估计器接入一个 optimizer step，同时证明累计量、参数移动量和实际更新诊断语义正确，并证明“打开统计”不会改变原本的 loss、gradient、optimizer state 或参数轨迹。

## 2. 前置 Gate

- `G1-GRAD` 和 `G1-EST` 已通过。
- 参数 registry、状态 schema 和 step 生命周期已冻结。
- 本子任务先在解析模型和 Tiny Transformer 的单进程 FP32 路径完成。

## 3. 固定 step 顺序

实现必须按下列逻辑边界组织；每个边界可单独观测和测试：

1. 记录 step 前参数 `Theta_t`、optimizer step、学习率和参数组元数据。
2. 清理上一步临时 gradient 与充分统计量。
3. 在本 step 内冻结唯一 loss scale；对每个 microbatch 执行 scaled backward。
4. 从每个 scaled local gradient 生成手工除以 scale 的统计副本，在未同步状态累计等权或加权充分统计量；不改变 optimizer 的 `unscale_` 状态机。
5. 同时聚合 scaled global mean gradient，并形成 global mean loss。
6. 把 scaled global mean gradient 写入 optimizer `.grad`。
7. 对 optimizer 只调用一次正式 step-level `unscale_`。
8. 检查 unscale 后 `.grad`、真实尺度充分统计量和每 rank `found_inf`，再聚合成唯一全局 execute/skip 决策，并把全局决策同步到每 rank 的 scaler inf-check 状态。
9. 若 skip，则丢弃本步临时统计与梯度；AMP 路径每 rank 仍调用一次正式 `scaler.step`（optimizer 实际更新零次）和一次 `scaler.update`，FP32 路径执行等价显式 skip；两者都推进 attempt/data/RNG/skip count，但不推进 successful optimizer step、scheduler、贡献累计或 movement。
10. 若 execute，则从 unscale 后 global mean gradient 计算未裁剪 raw、U 和全局 clip factor，并把只乘一次 clip factor 的主分数写入 step-local 暂存区。
11. 原位裁剪 optimizer 的 unscaled global mean gradient。
12. execute 路径调用一次 `scaler.step` 或 FP32 等价 optimizer step，确认 optimizer 实际更新一次，再调用一次 scaler update。
13. 事务性提交暂存的 signed、positive、negative mass、absolute 和 raw 累计。
14. 记录 step 后参数 `Theta_{t+1}`。
15. 分解数据驱动位移与 decoupled weight-decay 位移。
16. 事务性更新 data movement、带符号 data update、两类 net movement 和 magnitude。
17. execute 时更新 scheduler 与 successful optimizer step；两条路径都只推进一次 attempt，且各 rank 的 scale 与 growth tracker 一致。
18. 发布完整 step 日志和可保存状态。

## 4. 执行步骤

### 4.1 实现累计量

1. 直接累加单步主分数到 `importance_signed`。
2. 累加单步正部到 `importance_positive`。
3. 累加单步负部绝对值到 `importance_negative_mass`。
4. 累加单步绝对值到 `importance_absolute`。
5. 保持后三者为非负数组。
6. 每步核对 `signed = positive - negative_mass`。
7. 每步核对 `absolute = positive + negative_mass`。
8. 保存 checkpoint 区间增量，避免只能读取全程累计。
9. 禁止在形成 signed 之前执行 clamp。

### 4.2 实现 raw 与基线

1. 累计未裁剪 raw 到明确字段。
2. 若保存 clip-adjusted raw，使用独立字段。
3. 记录 step 前后参数的总位移。
4. 从 optimizer 规则分离数据驱动位移。
5. 从总位移中分离 decoupled weight-decay 位移。
6. 只将数据驱动位移绝对值累计到主 movement。
7. 另存 total movement 与 weight-decay movement 诊断。
8. 累计每一步带符号的 data update，并由其总和绝对值计算 data net movement。
9. 将 `|Theta_current-Theta_initial|` 另存为包含全部更新来源的 total net movement 诊断。
10. 从当前参数直接计算 magnitude。
11. 验证所有基线与 registry shape 一致。

### 4.3 先验证简单 SGD

1. 使用无动量、无 weight decay 的 SGD 解析 fixture。
2. 计算预期参数更新。
3. 计算预期 raw、U 和累计量。
4. 执行一个 step 并逐坐标比较。
5. 执行多个 step 并离线重放每一步。
6. 比较在线累计与离线累计。
7. 验证学习率 schedule 使用的是每步实际学习率。

### 4.4 验证 AdamW 边界

1. 先用 `weight_decay=0` 验证 AdamW 数据更新。
2. 保存更新前的一阶矩、二阶矩和 step 计数。
3. 独立重算 AdamW 数据驱动更新。
4. 与实际参数变化比较。
5. 再启用非零 decoupled weight decay。
6. 独立重算 weight-decay 位移。
7. 验证 `total = data + weight_decay`。
8. 验证主 movement 只使用 data 部分。
9. 验证 U 仍命名为 gradient-space importance，不随 optimizer moment 被重解释。
10. 将 `actual_update_raw_importance` 只作为诊断字段。

### 4.5 验证统计无扰动

1. 从相同初始状态建立“统计关闭”参考运行。
2. 建立“统计开启”待测运行。
3. 固定完全相同的样本、microbatch 切分和 RNG。
4. 逐步比较 mean loss。
5. 逐步比较交给 optimizer 的 gradient。
6. 逐步比较 optimizer state。
7. 逐步比较更新后参数。
8. 比较 scheduler 与逻辑 step 计数。
9. 在确定性 CPU/FP64 fixture 上优先要求 bitwise 相同。
10. 在 FP32 GPU 路径使用预注册容差，不以最终 loss 接近代替逐步比较。

### 4.6 验证失败与 skip

1. 注入一个 non-finite gradient step。
2. 验证该 step 不更新参数。
3. 验证 optimizer/scheduler 逻辑按契约处理。
4. 验证贡献累计和 movement 不前进。
5. 验证 skip 原因与次数进入结构化日志。
6. 验证下一正常 step 可以安全继续。
7. 验证单步候选分数先进入暂存区，只有确认 optimizer step 实际执行后才提交长期数组。
8. 验证动态 scale 在全部 microbatch 期间保持不变，step 结束时只更新一次。
9. 验证每个 rank 的 local `found_inf` 聚合成唯一全局 skip 决策。
10. 验证统计关闭/开启不会改变 GradScaler 的 step 前后状态。
11. 验证 clip、分数暂存和 optimizer step 均发生在正式 step-level unscale 与全局 finite 决策之后。
12. 逐步比较 finite/skip 两条路径的 scale、growth tracker、attempt、successful step、skip count、数据游标和 RNG 状态。
13. 验证 skip 消费当前 batch 后的下一批样本与参考一致，同时所有长期贡献与 movement 保持不变。

## 5. 产出

- 单进程训练 step 集成层。
- signed/positive/negative mass/absolute 累计模块。
- raw、data movement、net movement、magnitude 与 actual-update 诊断模块。
- SGD 解析重放报告。
- AdamW 数据位移/weight-decay 位移分解报告。
- 统计开启/关闭逐步对照表。
- skip-step 状态机测试报告。
- `G1-STEP` 机器可读报告。

## 6. 可视化

- 每步 `signed - (positive - negative_mass)` 与 `absolute - (positive + negative_mass)` 残差曲线。
- data、weight-decay 与 total movement 的逐步堆叠图。
- 统计开启/关闭的参数最大误差时间序列。
- raw、signed U、positive 和 negative mass 的调试级时间序列；仅用于发现实现问题，不解释模型机理。

## 7. 核验标准

- 累计恒等式在每一步、每个参数张量和保存/加载后通过对应 dtype 容差。
- `positive`、`negative_mass`、`absolute` 和 movement 全部非负。
- SGD 解析更新、在线分数和离线重放通过 `T64_ORACLE` 或 `T32_SINGLE`。
- AdamW 的 `total = data + weight_decay` 逐坐标通过 `T32_SINGLE`。
- 主 movement 与独立重算的数据驱动位移一致，不包含 weight decay。
- 非零 weight decay 下，`parameter_net_movement_data` 与累计带符号 data update 一致，`parameter_net_movement_total` 与参数端点差一致，二者不得互相代算。
- raw 未裁剪字段不受 clip factor 或 optimizer moment 影响。
- 统计开启与关闭的训练轨迹在预注册标准内一致。
- skip step 不产生参数、optimizer 逻辑、贡献或 movement 的半更新状态。
- skip step 的暂存贡献被丢弃；GradScaler 只按全局 skip 决策更新一次，后续正常 step 与参考轨迹一致。
- skip 后 `attempt_step`、数据游标、RNG、skip count 和 scaler 按契约推进，`successful_optimizer_step`、scheduler、贡献和 movement 不推进。

## 8. Gate 与后续依赖

- `G1-STEP` 通过后，S1.7 才能在 Pythia-14M 上运行真实链路。
- 任何 optimizer bridge、累计顺序或 movement 定义变化都必须重跑本 Gate。
- Stage 7 的剪枝只能使用本 Gate 明确区分的 signed、positive、negative mass、absolute 和 data movement 字段。
