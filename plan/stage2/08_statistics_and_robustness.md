# S2.8 统计分析、稳健性与结论规则

## 1. 子任务目的

在不重新计算模型梯度的情况下，对封存结果完成 Bias、Variance、MSE、排序、batch/M 规律和跨模型/阶段稳健性分析。分析必须同时考虑 reference 不确定性、repetition 不确定性和近零信号，不能用参数坐标数量制造伪显著性。

## 2. 前置条件

- G2.3 reference 和 G2.5 正式数据完整性 Gate 通过；
- raw-results manifest 已封存；
- 分析代码、指标定义、top-k 比例、等价/非劣阈值与 bootstrap 单位已经预注册；
- 确认性分析写入新的 `$DATA_ROOT/results/stage2/derived/<analysis-id>`，探索性分析使用另一个 analysis ID；不得清空或覆盖已有派生目录。

## 3. 实施步骤

### 3.1 验证分析输入

1. 校验 raw result、reference、参数注册表和 sample manifest 哈希。
2. 校验预期/实际行数、repetition 数、方法和 M 覆盖。
3. 校验重复表中的排序、成本和失败字段无缺失。
4. 校验 streaming mean/M2/MSE 与预注册 anchor 完整数值重算一致。
5. 校验 raw/double 没有因 M 维度被重复加权。
6. 生成 analysis input audit，失败时停止而不是自动丢行。

### 3.2 计算参考敏感性

1. 以无偏 `bias_ref` 计算主 Bias/MSE；以独立 `cross_ref` 做无偏敏感性分析，以低方差 `rank_ref` 计算主排序/top-k；A/B 平方只作诊断。
2. 比较 reference 版本下的方法方向和 Gate 敏感性，但禁止用有限正偏的 `rank_ref` 替换 bias 主判据。
3. 使用预注册两阶段 bootstrap（outer estimator repetitions，inner reference blocks）或等价的独立方差合并，在等价区间中同时计量两类不确定性。
4. 若结论对 reference 版本敏感，标为“参考不足”，不得宣称估计器有偏/无偏。
5. 对 near-zero 坐标使用绝对误差和 SNR 分箱，不以相对偏差驱动 Gate。

### 3.3 检验 raw 过估计

1. 对每个 model/checkpoint/B 计算 raw 的逐坐标 signed bias。
2. 从经 block-size 恢复且由单 sequence 诊断验证的 reference 方差得到理论预测 \(\widehat\sigma_k^2/B\)。
3. 在 parameter 诊断子集、tensor、layer 和 module scope 绘制观测 bias 对理论 bias。
4. 以 repetition/checkpoint 为独立单位拟合校准斜率和截距区间。
5. 检验 bias 对 `1/B` 的趋势，并报告翻倍 B 后的经验比例。
6. 按梯度 SNR 分箱，检查高方差/低均值参数是否被 raw 系统排高。
7. 对每个模型和训练阶段分别给出 H1/H2 状态，不把所有 checkpoint 混成一个均值。
8. 若辅助分析相对有限均值平方 `rank_ref` 展示 raw bias，理论线必须改为 `sigma2_hat × (1/B - 1/B_ref)`；该图不得替代相对 \(C^\star\) 的主结论。

### 3.4 检验双采样与 U 的偏差等价性

1. 对 double 和每个 M 的 U 计算相对无偏 `bias_ref` 的 signed bias 和 90% 等价区间。
2. 使用 S2.1 独立冻结的 `delta_sci(c,e,B_primary)` 执行等价判定；reference 半宽和数值误差只决定是否可判，不进入 margin。
3. 同时报告 absolute bias 相对 raw 的比例，避免“落入区间但没有实际改善”的误读。
4. 检验 bias 是否仍有稳定正方向或 `1/B` 斜率。
5. 检验 U 的平均值是否随 M 系统变化。
6. 对 parameter、layer、module 和 SNR 分箱分别报告，不用小 scope 的通过掩盖大 scope 的失败。
7. 对 `cross_ref` 重做敏感性判定；若两种无偏 reference 导致主状态不同，则标为 `inconclusive`。
8. 按确认性判据族表对六个主 cells 应用 intersection-union；不得从多个 B/M/scope 中选择任一通过值作为全局通过。

### 3.5 分解方差与 MSE

1. 对每个 estimator 计算 repetition variance、bias square，以及 S2.1 唯一定义的 parameter-vector `NMSE_observed`；分母 \(D_c\) 从 sizing stream 冻结，分析阶段不得改为按最终 reference 或方法结果归一化。
2. 同时报告无偏样本方差 `s2` 和分母为 R 的经验中心矩 `v_R`；对固定 reference、逐坐标和未归一化总量检查精确恒等式 `MSE_observed = v_R + Bias_observed^2`，或等价地 `MSE_observed = ((R-1)/R) × s2 + Bias_observed^2`。
3. 用 one-shot block-U jackknife/influence variance 的逐坐标对角和计算 `V_ref=trace(Sigma_ref)/D_c`，主目标为 `NMSE_star_hat=NMSE_observed-V_ref`；协方差不进入欧氏平方范数的 trace，但保留 scope 聚合所需协方差。不得 clamp 负结果。
4. 用两阶段 bootstrap 给出联合区间；若 double 的校正 NMSE 或其区间没有高于预注册正 floor，则 U/double 比标为 `inconclusive`。
5. 计算 U/double、U/raw 和 double/raw 的配对 variance/校正 NMSE 比。
6. 按 B 检查一阶 `O(1/B)` 和弱信号下二阶 `O(1/B^2)` 趋势。
7. 按 M 检查 U 方差的理论方向；M=2 与 double 方差应相等。H5 只在 iid 有放回、等权/正确加权、同总预算和同一 \(C^\star\) 条件下解释。
8. 用两阶段 repetition/reference-block bootstrap 计算比值区间，不对参数坐标独立 bootstrap。
9. 高斯理论线仅作为标注“附加假设”的辅助层，主结论以经验统计为准。

### 3.6 分析排序与 top-k 恢复

1. 对每个 repetition 计算 estimator 与 reference 的 Pearson/Spearman。
2. 计算 repetition 均值向量与 reference 的 Pearson/Spearman。
3. 对预注册 top 比例计算 Overlap@K 和 Jaccard。
4. 计算 U/double 的配对排序差和非劣区间。
5. 分别报告 parameter、tensor、layer 和 module 排名。
6. 对 tied scores 使用平均秩和固定 tie-breaking；记录 top-k 边界并列数量。
7. 比较 signed 与仅用于诊断的 positive/clamped 派生量时，必须放入探索性附录，不能进入无偏主结论。

### 3.7 分析 U 的负值与数值消减

1. 计算每个 repetition 的负坐标比例、负质量和最小分位数。
2. 按 B、M、checkpoint 和 SNR 分箱展示负值。
3. 检查负值是否集中在 near-zero reference 坐标。
4. 用高精度诊断坐标区分统计负波动和数值消减。
5. 报告 signed mass、absolute mass 和净质量比，不把负值自动标成错误。
6. 检查结果/分析代码中不存在隐式 clamp 或 abs 后再排序。

### 3.8 评估跨阶段和模型稳健性

1. 对初始化、早期、中后期分别形成完整方法比较。
2. 检查 H1–H6 是否在 14M 和 31M 方向一致。
3. 计算 model/checkpoint 间效应的异质性，而不是只报告 pooled 平均。
4. 因 14M 与 31M 的 deduped 身份不同，跨模型只判断估计器规律能否重复，不对效应量差异作纯规模因果解释。
5. 对高 SNR 条件中 raw 偏差不可分辨的情况，结合 reference half-width 给出“证据不足”解释。
6. 若 31M 与 14M 方向相反，先检查采样、reference、checkpoint、数据版本和数值诊断，再决定是否形成适用边界结论。
7. 所有 31M confirmatory draw IDs 必须与 pilot stream 独立，且在矩阵冻结前不读取任何确认性梯度；确认性结果不用于回改 B/M/R。

### 3.9 控制多重比较与探索性分析

1. 主要结论只使用 `B_primary`、`M_primary`、六个 model×stage cells 和确认性判据族表中的 primary endpoints。
2. 候选 bias 资格要求六个主 cells 全部通过（intersection-union）；校正 NMSE、Spearman 和 Overlap@`1%` 非劣也逐 cell 通过。任一 cell 的 fail/inconclusive 按预注册规则传播，不能用 pooled 平均掩盖。
3. H1/H2 使用预注册层级斜率与“无 powered cell 反向”规则；H4/H5、其他 B/M/top-q/scope 是 secondary 稳健性证据，不负责挽救主 Gate。
4. 探索性逐层/逐 tensor 检验使用统一 FDR 或 simultaneous interval，并单独列出。
5. 对任何事后新分箱、新阈值或新 top-k 标记 `exploratory`。
6. 保留原主结果，即使探索性分析给出更好解释，也不能追溯改写 Gate。

### 3.10 生成机器可读结论

1. 为每个质量 Gate 生成 pass/fail/blocked 和证据路径。
2. 为 H1–H6 生成 supported/not_supported/inconclusive。
3. 为每个 estimator 生成 `bias_equivalence`、`corrected_NMSE_non_inferiority`、`ranking_non_inferiority` 状态。
4. 保存效应值、区间、阈值和分母，不只保存布尔结果。
5. 生成从原始 wave 到统计表的 lineage manifest。
6. 输出确认性判据族逐 cell 表，包含 primary/secondary、聚合顺序、effect、联合区间、margin、multiplicity、状态与失败传播。

## 4. 产出

- 完整长表：model × checkpoint × B × M × method × scope × metric；
- Bias、Variance、MSE、MAE 和分解表；
- raw 理论校准与 `1/B` 回归表；
- 偏差等价、MSE/排序非劣检验表；
- Pearson、Spearman、Overlap@K、Jaccard 表；
- U 负值/SNR/数值诊断表；
- reference 敏感性和跨模型/阶段异质性报告；
- `quality_gates.json`、`hypothesis_decisions.json` 和 lineage manifest。
- `confirmatory_family_decisions` 逐 cell 长表和全局归并 JSON。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.6 统计有效性 Gate**：

- 所有主统计均可由封存输入重建，anchor 重算一致；
- reference 与 repetition 不确定性被分开计量；
- 偏差等价区间同时合并 estimator/reference 不确定性，且科学 margin 未随误差放宽；
- 置信区间以 repetition/checkpoint/model 为独立单位，没有参数伪重复；
- 等价检验和非劣检验使用冻结阈值，不以“不显著”替代；
- raw 的理论校准、U/double 偏差、方差/MSE、排序和负值均有完整结果；
- 14M/31M 与不同训练阶段没有被不加说明地 pooled；
- 探索性分析与确认性分析目录和标签分离；
- Gate 与 H1–H6 的机器记录含效应、区间、阈值和证据路径。
- 六个主 cells、唯一主 B/M 和 primary endpoints 的交集归并可由机器表重算，无择优通过空间。

该 Gate 只证明分析可信，不要求 H1–H6 全部得到支持。

## 6. 后续依赖

- S2.10 只能使用本任务封存的统计表和结论 JSON。
- S2.11 根据质量 Gate 和方法决策 Gate 判断能否进入后续阶段。
