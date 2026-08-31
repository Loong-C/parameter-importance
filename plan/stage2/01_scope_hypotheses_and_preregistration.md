# S2.1 范围、假设、判据与预注册

## 1. 子任务目的

在任何模型梯度计算开始前，冻结本阶段估计对象、比较方法、实验因素、统计指标和结论判据。这样可以防止看到结果后再改变 checkpoint、batch size、microbatch 数、重复次数或 Gate，从而保证“过估计”和“无偏”结论可复核。

## 2. 前置输入

- `plan/general_plan.md` 的 Stage 2 目标；
- `docs/mathematics.md` 第 7–10、14、18–21 节；
- Stage 0 的现状快照和 Gate 定义；
- 旧归档中 estimator bias、signed/positive 稳定性和 U/double 诊断结果，仅作为风险清单，不作为确认性数据。

## 3. 实施步骤

### 3.1 冻结估计对象

1. 将主目标写为固定 checkpoint 下的 \(\mu_k^2\)，并记录主实验统一使用 `eta_eval=1`。
2. 明确 checkpoint 在一个 reference 或 repetition 单元内不可变化。
3. 明确主实验不执行 optimizer、scheduler、梯度裁剪或权重衰减。
4. 明确完整路径积分、actual-update contribution 和 probe loss alignment 不进入本阶段结果表。
5. 明确所有 Bias、Variance、MSE 和排序分析使用 signed 原始估计，不做截断、绝对值或正负分离变换。

### 3.2 冻结三种估计器

1. 单独登记 raw：同一总 batch 的平均梯度平方。
2. 单独登记双采样：两个独立半 batch 的平均梯度乘积。
3. 单独登记 microbatch U-statistic：删除同 microbatch 对角项的交叉乘积平均。
4. 将 `M=2` 的 U 与同一两半样本双采样逐坐标相等列为必须通过的不变量。
5. 将不等有效 token 数的加权 U 列为实现能力；主实验只有在所有 sequence 权重相等时才可使用等权公式。
6. 将独立 probe loss estimator 设为禁止混用的名称和输出 schema，防止历史命名复用造成误解。

### 3.3 预注册科学假设

1. **H1 — raw 正偏**：固定 checkpoint 和 batch size 时，raw 的平均偏差为正，且与 \(\widehat\sigma_k^2/B\) 的方向和尺度一致。
2. **H2 — batch 规律**：raw 偏差随 \(1/B\) 衰减；对高 SNR checkpoint，效应可以小到不可分辨，但不能反向解释为理论失效。
3. **H3 — 无偏性**：双采样和 U 的平均偏差落在预注册等价区间内，并且没有 raw 的稳定正偏或 \(1/B\) 斜率。
4. **H4 — U 的 M 不变性**：固定总预算 B 时，U 的均值不随 M 系统漂移；M 主要影响方差和工程成本。
5. **H5 — 等预算效率**：在独立同分布、有放回、等权（或使用正确加权式）、同一真实目标和相同总 draw 预算下，`M>2` 时 U 的经验方差和 MSE 不高于平分预算双采样；`M=2` 时两者相等。有限 reference 噪声、非独立抽样或错误权重不得被当作反例，必须先单独诊断。
6. **H6 — 排序恢复**：重复平均后的 U/双采样排序更接近参考；单次估计排序可能因无偏估计器方差而不在所有小 B 条件下优于 raw。

### 3.4 冻结实验因素与确认性单元

1. 模型固定为 Pythia 14M 和 Pythia 31M deduped。
2. 每个模型选择初始化、早期和中后期至少三个 checkpoint。
3. batch size 候选网格在 pilot 前固定为 `{32,64,128,256}` 个 sequence；S2.6 只能按预注册可运行性/功效规则删除候选并选出一个 `B_primary`，不能新增数值或按方法优劣选择。
4. M 候选主网格登记为 `{2, 4, 8, 16, 32}`，每个条件必须满足 `B % M == 0` 且每个 microbatch 至少含一个完整统计单元。
5. 人工分布和 `pilot` stream 上的 14M step0 只用于校准；冻结后从独立 confirmatory stream 运行的 14M 初始化/早期/中后期及全部 31M 主 cells 才是确认性单元。
6. 每个确认性单元记录独立的 model/checkpoint/B/M/repetition ID，禁止把缺失单元静默从分母中删除。

### 3.5 冻结主要统计量

1. 逐参数计算重复均值、signed Bias、absolute Bias、Variance、MSE 和 MAE。
2. 逐 repetition 计算 Pearson、Spearman、Overlap@K 和 Jaccard；Spearman 并列值使用平均秩。
3. 同时计算“单次估计对参考”和“所有重复均值对参考”的排序指标。
4. 预先固定 top 比例为参数总数的 `0.01%`、`0.1%`、`1%` 和 `5%`；统一使用 `K=max(1, ceil(qP))` 并记录实际 K。
5. 对 parameter、tensor、layer、module 四个 scope 分别汇总；层总量和层平均每参数量同时保留。
6. 冻结聚合顺序：先在每个 repetition 内按 canonical parameter ID 聚合为 scope 总量，再相对同顺序聚合的 reference 计算 Bias/MSE；`absolute Bias=|E_r[\widehat C]-C^\star|`，MAE 为 `E_r|\widehat C-C^\star|`，二者不得互换。scope 排名主量使用总量，每参数平均量仅作规模敏感性分析。
7. 对接近零参考贡献的坐标按参考信号强度或梯度 SNR 分箱；相对偏差不得作为近零箱的唯一指标。
8. 独立性单位固定为 repetition/checkpoint/model；参数坐标间相关性通过分层汇总和重复级 bootstrap 处理。

### 3.6 冻结科学等价边界与精度 Gate

1. 把理论目标命名为 \(C^\star=\mu^2\)，把有限样本 reference 另命名为 \(C^{\mathrm{bias\text{-}ref}}\)、\(C^{\mathrm{cross\text{-}ref}}\) 和 \(\widetilde C^{\mathrm{rank\text{-}ref}}\)；任何报告不得写成“相对有限均值平方严格无偏”。
2. 对每个 model×stage cell、每个候选 B 和每个主 endpoint，只用独立 `reference_sizing`（必要的绝对 floor 只来自 Stage 1/人工 fixture）冻结科学等价边界：

   \[
   \delta_{\mathrm{sci},c,e}(B)
   =\max\left(0.10\,\Delta_{c,e}(B),\;0.01\,S_{c,e}\right).
   \]

   对 sizing stream 计算逐坐标 \(a_k=(\widehat\mu_k^{\mathrm{sizing}})^2\) 与 \(d_k(B)=\widehat\sigma_k^2/B\)。model-total endpoint 使用

   \[
   S_{\mathrm{model}}=\max\left(\left|\sum_k a_k\right|,\tau_{\mathrm{model}}\right),\quad
   \Delta_{\mathrm{model}}(B)=\left|\sum_k d_k(B)\right|;
   \]

   layer/module L1 endpoint 分别使用

   \[
   S_q=\max\left(\sum_{g\in q}\left|\sum_{k\in g}a_k\right|,\tau_q\right),\quad
   \Delta_q(B)=\sum_{g\in q}\left|\sum_{k\in g}d_k(B)\right|,
   \quad q\in\{\mathrm{layer,module}\}.
   \]

   layer/module 分组必须是 canonical、互不重复覆盖的 registry 分区；\(\tau_{\mathrm{model}}\)、\(\tau_{\mathrm{layer}}\)、\(\tau_{\mathrm{module}}\) 是在任何真实 sizing/pilot draw 前由 Stage 1 人工 fixture 以原生单位固定的绝对 floor。最终 one-shot A/B 只能进入检验量和 reference 不确定性，不能重新定义 \(S\)、\(\Delta\) 或 margin。
3. Reference 半宽和数值误差不进入 \(\delta_{\mathrm{sci}}\)。G2.3 对每个 cell/endpoint 要求 `h_ref <= min_B(delta_sci(B))/4` 且 `epsilon_num <= min_B(delta_sci(B))/10`，其中最小值遍历全部候选 B；因此无需先知道 `B_primary`。任一失败时本轮为 `inconclusive/blocked`，不得扩大等价区间。
4. 对 model-total signed bias，双采样或 U 只有在同时合并 estimator repetition 与 reference block 不确定性的 90% 等价区间完整落入 `[-delta_sci(c,e,B_primary),+delta_sci(c,e,B_primary)]` 时才通过；对 layer/module L1 bias，使用预注册 one-sided 95% 上界小于同 endpoint 的 `delta_sci(c,e,B_primary)`。“差异不显著”不能替代等价检验。
5. raw 理论校准的主判据为层级校准斜率区间位于 `[0.8,1.2]`、截距在独立精度预算内，且 bias 对 `1/B` 的方向正确。cell 无功效时标为证据不足，不能只挑选通过的层。

### 3.7 冻结确认性判据族与跨条件归并

1. 在运行任何真实 pilot 前冻结确定性选择算法。对每个候选 B，先按固定优先序 `[32,16,8,4]` 扫描并指定第一个满足 `M|B`、每个统计 microbatch 至少一个完整 sequence、六个 anchors 无 OOM/非有限值且 estimator 聚合开销不超过共享梯度时间 25% 的 `M_candidate(B)`；不得按 U 的 Bias/NMSE/排序结果选择 M。
2. 再按 B 从小到大扫描，选择第一个同时满足以下条件的 `(B,M_candidate(B))`：六个 anchors 均可运行；所有主 endpoint 以方法间最坏方差计算的 `R_required <= R_max`；总 A100-hours、显存和存储不超过预注册上限。方差估计可来自独立 pilot，但不得使用 bias 方向、方法均值、NMSE/排序优劣或显著性。若无候选满足，本轮阻断；未来只能新建独立预注册轮次，不能在本轮改规则后继续。
3. 上述 M 优先序和 B 升序本身就是 tie-breaker；所有过滤字段、阈值和每个候选的 pass/fail 必须落入机器表。其他 B/M 只检验 H2/H4/H5 和稳健性。
4. 主偏差 cells 固定为 `2 models × 3 training stages × B_primary`。主 endpoint 为 model-total signed bias、layer-total L1 bias 和 module-total L1 bias；parameter-level 分布为诊断，不作为独立重复。
5. 对 U 与 double 使用 intersection-union 规则：候选必须在全部六个主 cells 的预注册主 endpoints 通过等价性，才能获得全局 `bias_qualified`。任一 cell 失败即不得作为无条件主方法；精度不足标为 `inconclusive`，可形成适用范围结论但不能择优删除该 cell。因为声明要求所有主 cells 同时通过，等价性主 Gate 不以事后多重比较校正替代该交集规则。
6. 排序主判据固定为 parameter Spearman 与 Overlap@`1%`；MSE 主判据使用本节下一条的唯一归一化定义。`0.01%/0.1%/5%`、tensor/layer/module 排序和其他 B/M 均为预注册 secondary，不得用于替换失败的主 endpoint。
7. 对 cell c，令 sizing 向量为 \(a_c\)、参数数为 \(P_c\)，并在真实数据前固定逐坐标 floor \(\tau_{\mathrm{coord}}\)。冻结分母

   \[
   D_c=\max\left(\sum_{k=1}^{P_c}a_{c,k}^2,\;P_c\tau_{\mathrm{coord}}^2\right).
   \]

   对 estimator 向量 \(x_{r}\) 和最终无偏 reference \(b\)，定义

   \[
   \mathrm{NMSE}_{\mathrm{obs}}=\frac{1}{R D_c}\sum_{r=1}^{R}\sum_k(x_{r,k}-b_k)^2,
   \qquad
   V_{\mathrm{ref}}=\frac{\mathrm{tr}(\widehat\Sigma_b)}{D_c}
   =\frac{\sum_k\widehat{\mathrm{Var}}(b_k)}{D_c},
   \]

   主量为 \(\widehat{\mathrm{NMSE}}_\star=\mathrm{NMSE}_{\mathrm{obs}}-V_{\mathrm{ref}}\)。\(\widehat\Sigma_b\) 由 one-shot reference blocks 的预注册 U-statistic jackknife/influence estimator 给出，bootstrap 只负责联合区间；不得 clamp 负的有限样本结果。正 floor \(\tau_{\mathrm{NMSE}}>0\) 与 \(\tau_{\mathrm{coord}}\) 同样在任何真实 draw 前用人工 fixture 固定；U/double 比仅在 double 的校正 NMSE 及其区间均高于 \(\tau_{\mathrm{NMSE}}\) 时计算，否则该 cell 为 `inconclusive`。
8. U 相对 double 的校正 NMSE 非劣边界为 `1.10`；Spearman 配对差边界为 `-0.02`，Overlap@`1%` 边界为 `-0.03`。六个主 cells 均须通过相应 95% 区间，逐层/逐 tensor 探索性检验使用统一 FDR 或 simultaneous interval。
9. H1/H2 的全局状态采用层级规则：总体斜率通过且没有有充分功效的 cell 出现反向证据时为 `supported`；仅有低功效 cell 为 `inconclusive`；任一有充分功效 cell 明确反向时为 `not_supported`。各 cell 状态仍全部公开。
10. 主成本判据使用 S2.9 的 `online_training_incremental_cost` 方法独立 anchor，U/double 的 wall time 与峰值显存比上限均为 `1.25`；共享配对 runner 的公式增量成本只作解释，不能替代部署成本。
11. 若 U 和 double 均 `bias_qualified` 且 U 的主 NMSE、排序和在线成本全部非劣，则唯一选择 U；不要求额外“显著优效”。否则选择通过 bias Gate 且满足下游绝对资源预算的 double。两者均失败则阻断，禁止回退 raw。
12. 将上述字段写入机器可读“确认性判据族表”，每行包含 endpoint、primary/secondary、独立单位、聚合顺序、cell 集合、区间、阈值、多重性、失败传播和 tie-breaker。
13. `delta_sci`、absolute floors、非劣边界、B/M 扫描顺序和资源上限在真实 pilot 前固定，pilot 只执行这些过滤规则，不能修改规则本身。不可运行性导致无候选时，本轮为 blocked；任何新设计必须保留本轮并作为独立预注册轮次。

### 3.8 区分质量 Gate 与假设结论

1. 建立 `quality_gates`：状态固定、样本独立、参考收敛、结果完整、数值有限、预算公平和可重放。
2. 建立 `hypothesis_decisions`：H1–H6 分别标为支持、不支持或证据不足。
3. 规定质量 Gate 失败时不得解释科学结果。
4. 规定科学假设不支持时仍保留完整结果，并按预注册分支选择双采样、调整 U 或阻断后续阶段。

### 3.9 生成并冻结预注册文件

1. 生成机器可读预注册文件，包含公式版本、因素、候选矩阵、seed 规则、指标、阈值和决策树。
2. 生成面向审阅者的 Markdown 版本，逐项解释每个字段的含义。
3. 计算规范化配置摘要和文件 SHA-256。
4. 将预注册绑定到 Git commit、数学文档哈希和 Stage 1 测试报告哈希。
5. 在任何确认性样本 ID 生成前写入工作日志并提交；后续只能追加 amendment，不能覆盖原文件。

## 4. 产出

- Stage 2 预注册 Markdown；
- 机器可读的 preregistration JSON/YAML；
- H1–H6 与指标、图表、Gate 的映射清单；
- 确认性判据族表、科学等价边界、独立精度 Gate、非劣边界和唯一决策树；
- 预注册哈希与提交信息；
- amendment 模板。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.0 进入与预注册 Gate**：

- 估计对象唯一且不包含路径积分/AdamW 实际更新；
- raw、双采样、U 和 probe loss 名称互不混用；
- B、M、统计单元、学习率和 loss reduction 均有唯一解释；
- 所有确认性因素、主要指标、top-k 比例和阈值已冻结；
- `B_primary`、`M_primary` 的选择规则、六个主 cells、判据族归并和失败传播已冻结；
- 科学 margin 不依赖 reference/数值误差，二者作为独立精度 Gate；
- 质量失败与假设不支持有不同状态；
- 预注册在确认性样本生成前形成提交并可验证哈希。

若该 Gate 未通过，只能继续文档审查，不得启动正式 reference 或 repetition 计算。

## 6. 后续依赖

- S2.2、S2.3 和 S2.6 都必须引用本任务冻结的术语、因素与阈值。
- 若 S2.6 需要修改矩阵，必须先追加 amendment，再生成确认性样本 manifest。
- S2.8 和 S2.10 不得新增未预注册的主指标；探索性分析必须单独标注。
