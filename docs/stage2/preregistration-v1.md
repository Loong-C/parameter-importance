# Stage 2.1 预注册（G2.0）

本文件与 `src/param_importance_nlp/experiments/preregistration.py` 的
`stage2-preregistration-v1` 合同同源。运行 `stage2.01_scope_hypotheses_and_preregistration`
会产生带 `preregistration_hash` 的 canonical JSON；该 JSON 才是下游任务消费的
机器权威副本。本文件是面向审阅者的解释副本，不得在确认性 draw 生成后改写。

## 研究对象与排除项

在固定 checkpoint `Theta` 下，唯一理论目标是
`C*_k = eta_eval * mu_k^2`，`mu_k = E_F[g_k(Theta; z)]`，并冻结
`eta_eval=1`。checkpoint、模型、optimizer、scheduler、权重、数据游标和 RNG
状态在一个 reference/repetition 内不可变；不执行 optimizer step、scheduler、梯度
裁剪或 weight decay。完整参数路径积分、AdamW 实际更新贡献、独立 probe 损失对齐、
累计正/负/绝对重要性、剪枝和 160M/410M 正式训练均不属于本阶段。

所有 Bias、Variance、MSE、MAE 和排序使用 signed 原始估计。禁止在统计前
`clamp_min(0)`、取绝对值或先分离正负；负的 U 值必须保留并报告。

## 三种估计器

| ID | 名称 | 公式/抽样 | 结论边界 |
|---|---|---|---|
| `raw` | `local_gradient_space_importance_raw` | 同一总 batch 的 `(mean(g))^2` | 同源噪声会产生 `sigma^2/B` 偏差 |
| `double` | `double_sample_gradient_importance` | 两个独立半 batch 的均值乘积 | 固定状态目标的无偏估计 |
| `u` | `microbatch_u_statistic` | `(S1^2-S2)/(M*(M-1))`，删除同 microbatch 对角项 | 固定状态目标的无偏估计，允许 signed/负值 |

`M=2` 的 U 必须逐坐标等于同一两半样本映射的 double；不等有效 token 数时
只能使用正确的 exogenous-weight 加权式，主实验仅使用等 sequence 权重。
`independent_probe_loss_drop` 是不同 estimand，禁止作为 double/U 的别名或进入
本阶段方法比较。

## 因素、抽样与统计单位

- 模型：`pythia-14m`、`pythia-31m-deduped`；阶段：`initialization`、`early`、`mid_late`。
- 候选 `B`：`{32,64,128,256}` 个 sequence；候选 `M`：`{2,4,8,16,32}`，要求 `M|B`。
- 主 cells 是 2×3 的六个 model×stage 组合。人工分布和 14M step0 只校准；独立
  `confirmatory` stream 的 14M 三阶段及全部 31M cells 才是确认性单元。
- `reference_sizing`、`reference_A`、`reference_B`、`pilot`、`confirmatory` 五条
  RNG stream 独立；draw ID 不复用，sample ID 按有放回理论允许碰撞且不得去重。
  sizing 先冻结参考规模；最终 A/B one-shot，失败即阻断；确认性 mapping 在读取
  梯度前生成并封存。
- 独立性单位是 repetition/checkpoint/model，不把 parameter 坐标当独立重复。
  Top 比例固定为 `0.01%/0.1%/1%/5%`，`K=max(1,ceil(qP))`，并记录实际 K。
  Spearman 并列值使用平均秩。

指标按 parameter、tensor、layer、module 汇总；先按 canonical parameter ID 在每个
repetition 内聚合，再相对同序 reference 计算偏差。主 scope 量是总量，每参数平均
仅作 secondary。`absolute Bias=|E_r[x]-C*|` 与 `MAE=E_r|x-C*|` 不得互换。

## 等价边界、精度与选择算法

对每个 cell/endpoint，仅用独立 sizing stream 冻结
`delta_sci=max(0.10*Delta,0.01*S)`。`S` 使用 sizing `a=mu^2`，`Delta` 使用
`d=sigma^2/B`；model、layer、module 均使用 canonical non-overlapping registry。
绝对 floor (`tau_model/tau_layer/tau_module/tau_coord/tau_nmse`) 在真实 draw 前由
Stage 1 artificial fixture 固定，不能由 pilot 调整。reference 半宽和数值误差不
进入科学 margin；必须分别满足 `h_ref <= min_B(delta_sci)/4` 与
`epsilon_num <= min_B(delta_sci)/10`，否则 `inconclusive/blocked`。

model-total signed bias 用合并 estimator/reference 的 90% 区间完整落在 margin 内；
layer/module L1 bias 用 one-sided 95% 上界小于 margin。raw 校准斜率区间固定在
`[0.8,1.2]`，截距须在独立精度预算内。

先按 `[32,16,8,4]` 为每个 B 选择第一个满足整除、完整 sequence、六 anchors 无 OOM/
非有限且聚合开销不超过共享梯度时间 25% 的 `M_candidate`；再按 B 升序选第一个
满足六 anchors、最坏 `R_required <= 64`、资源上限的 pair。过滤只能看运行性、有限
性、方差所需 R 和资源，禁止看 bias 方向、方法均值、NMSE、排序或显著性。无候选
则本轮 blocked，须另起 amendment/预注册轮次。

主方法必须在六个 cells 的 intersection-union bias Gate 全部通过。U/double 的
校正 NMSE 非劣界为 `1.10`，Spearman 配对差下界 `-0.02`，Overlap@1% 下界 `-0.03`，
online-training-incremental wall time 与显存比上限 `1.25`。U 与 double 都合格且
U 在 NMSE、排序、在线成本均非劣才唯一选择 U；否则选择通过 bias 且满足绝对资源
预算的 double；两者均失败时阻断，不回退 raw。

## H1–H6 与质量状态

H1 为 raw 正偏且尺度符合 `sigma²/B`；H2 为随 `1/B` 衰减；H3 为 U/double 无偏
等价；H4 为固定 B 时 U 均值不随 M 系统漂移；H5 为相同 draw 预算下 M>2 的 U
方差/MSE 不劣于 double 且 M=2 相等；H6 为重复均值的 U/double 排序更接近 reference。
每个假设只能标记 `supported`、`not_supported` 或 `inconclusive`。

质量 Gate（fixed state、sample independence、reference convergence、completeness、
finite numeric、fair budget、replayability）失败时不得解释 H1–H6；科学假设不支持
仍须保留完整结果。后续 amendment 只能 append-only，记录 parent hash、变更字段、
非事后依据、受影响 Gate 和新 hash，不能覆盖原注册。
