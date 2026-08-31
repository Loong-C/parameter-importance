# S2.4 大样本参考目标与参考不确定性

## 1. 子任务目的

为每个 model × checkpoint 建立可用于无偏性检验的无偏有限样本 reference、可用于排序的低方差 reference，以及独立的 sequence 方差预测量。Reference 仍不是理论真值，因此必须给出收敛、不确定性和数值误差证据，且不能用参考误差放宽科学等价边界。

## 2. 前置条件

- G2.1 固定状态梯度接口通过；
- G2.2 checkpoint、`reference_sizing` manifest、最终 A/B 的保留 seed namespaces 与生成 schema 通过；
- 参数注册表、loss reduction、`eta_eval=1` 和 FP32/高精度策略已经冻结；
- 服务器当前运行前健康 GPU、存储和空闲状态通过只读预检。

## 3. 实施步骤

### 3.1 建立 reference 运行身份

1. 为每个 model × checkpoint 创建唯一 reference study ID 和 sizing run ID；最终 A/B run IDs 只在 \(B_{\mathrm{ref}}\) 冻结后生成。
2. 先绑定 checkpoint revision、参数注册表哈希、数据 manifest、sizing draw manifest、代码提交和环境摘要；A/B run 创建时再绑定其固定全长 draw manifests。
3. 把大型输出固定在 `$DATA_ROOT/results/stage2/reference/<reference-id>`，block 状态和日志固定在 `$DATA_ROOT/runs/stage2/<reference-id>`；staging 使用同一文件系统的 `$DATA_ROOT/tmp/stage2/<reference-id>`，通过校验后原子发布。
4. 对每个 draw 验证 sample ID 位于 G2.2 allowlist，且 draw stream/顺序哈希一致。
5. 保存运行前模型参数、buffer 和模型/全局 RNG 摘要；另存 sampling generator 起始状态。
6. 禁止 reference 入口创建 optimizer step 或训练 checkpoint。
7. 把 `reference_sizing`、最终 `reference_A` 与最终 `reference_B` 设为三个独立 RNG streams；draw IDs 不共享，sample ID 可按有放回抽样偶然碰撞并记录。
8. 在创建多个 FP64 参数向量前估算 gradient sums、block summaries、三类 reference、staging 和重试余量；同时检查项目盘空间/inode 与主机内存。容量不足时先调整 block 发布/汇总策略，不能写到根盘或留下半成品。

### 3.2 流式累计参考梯度

1. 按 G2.2 已冻结的随机 draw 顺序读取 reference sequence，不在运行时重新洗牌或使用自然连续索引。
2. 以能稳定运行的 microbatch size 计算 FP32 mean gradient。
3. 根据每个 microbatch 的有效统计权重恢复为总和贡献。
4. 以 FP64 或已验证的高精度累加器累计逐参数梯度总和。
5. 将等大小的 \(b_{\mathrm{ref}}\) 个 sequence 组成一个独立 reference block，累计 block mean gradient 及其逐参数二阶矩。
6. 按 tensor、layer 和 module 维护独立聚合，便于在不加载完整参数数组时检查进度。
7. 每完成一个预注册 reference block，原子保存累计状态、样本计数和最后 sample ID。
8. 不长期保存逐 sequence 的完整参数梯度；只保存累计量、诊断子集和必要 block summaries。
9. 若用 block mean 方差估计单 sequence 方差，明确计算 `sigma2_hat = b_ref × sample_variance(block_means)`；不等 block 权重时使用冻结的加权/cluster 方差式，不能漏掉 block-size 因子。
10. 对少量预注册诊断坐标直接保存单 sequence 梯度，交叉验证 block 缩放后的 \(\widehat\sigma^2\)。

### 3.3 用独立 sizing stream 冻结固定 reference 样本量

1. 只在 `reference_sizing` stream 上建立从 512 开始逐次翻倍的候选阶梯。
2. 在 512、1024、2048、4096 等累计节点生成 sizing 快照，记录收敛、区间、有效 sequence/token、wall time、显存和文件大小。
3. 根据 sizing 结果、全部候选 B 对应的 `delta_sci(B)` 精度要求和预注册容量上限选择固定、为偶数的总样本量 \(B_{\mathrm{ref}}\)，并在生成最终 A/B 的任何 draw ID 前提交 amendment/manifest。
4. Sizing stream 的统计量只用于冻结 \(S\)、\(\Delta(B)\)、margin、样本量和资源预算，不进入最终 Bias、NMSE、排序或 Gate 的观测 reference 数组。
5. 冻结 \(B_{\mathrm{ref}}\) 后，分别从全新的 `reference_A` 与 `reference_B` streams 一次性各生成 \(B_{\mathrm{ref}}/2\) 个 draws；完整消费，不执行 early stopping 或“达到阈值即停止”。
6. 最终 A/B 固定长度运行仍生成嵌套前缀收敛图，但这些前缀只做质量诊断，不决定本次 reference 的样本量。
7. 最终 A/B 是本轮唯一 one-shot 主 reference。若未通过 G2.3，本轮 Stage 2 立即标为 `inconclusive/blocked`；不得延长、重抽或用“首个通过”的新版本替换。未来若重启研究，必须建立独立预注册轮次，保留本轮失败，并预先采用能合并多轮的 multiplicity/sequential 控制或 anytime-valid 设计，不能把新轮次冒充本轮修复。
8. 对 31M 可使用较小 sizing 起点做资源 smoke，但最终固定长度必须达到与 14M 相同的精度 Gate。

### 3.4 计算无偏 bias reference 与低方差 ranking reference

1. 把理论目标固定记为 \(C^\star=\mu^2\)，不把任何有限 reference 文件命名为真值。
2. 对 A/B 的全部独立等权 block mean gradients 计算去对角 U 统计量 \(C^{\mathrm{bias\text{-}ref}}\)，作为 Bias/MSE 等价检验的主 reference。
3. 分别计算 \(\mu_A\)、\(\mu_B\) 和交叉参考 \(C^{\mathrm{cross\text{-}ref}}=\mu_A\odot\mu_B\)，作为第二个无偏敏感性 reference。
4. 合并 A/B 计算 \(\mu_{AB}\) 和 \(\widetilde C^{\mathrm{rank\text{-}ref}}=\mu_{AB}^2\)，只用于排序、top-k 和 reference 收敛主图；明确其有限样本期望含正偏 \(\sigma^2/B_{\mathrm{ref}}\)。
5. 另存 \(C_A=\mu_A^2\) 与 \(C_B=\mu_B^2\) 作为 A/B stream 诊断，不进入无偏性主 Gate。
6. 对三类 reference 保存 signed 数组、scope 聚合、schema 和不确定性标签，禁止在分析阶段静默互换。

### 3.5 量化参考不确定性

1. 将 reference sequence 按预注册 block 大小分块，计算 block mean gradients。
2. 以独立 draw block 为重采样单位估计 \(C^{\mathrm{bias\text{-}ref}}\)、\(C^{\mathrm{cross\text{-}ref}}\) 和 \(\widetilde C^{\mathrm{rank\text{-}ref}}\) 的区间及逐 endpoint reference variance，避免把参数坐标当独立重复。
3. 计算 A/B 梯度差、贡献差、层/module 总量差和 top-k 差异。
4. 用经 block-size 恢复并由单 sequence 诊断验证的 \(\widehat\sigma_k^2\)，为 raw 理论偏差 \(\widehat\sigma_k^2/B\) 提供独立预测量。
5. 对近零参考坐标报告绝对半宽和 SNR，不使用不稳定的逐坐标相对半宽做唯一判断。
6. 把 reference sampling error、数值累加误差和 checkpoint/load 重放误差分开列出。
7. 用 block-U 的预注册 jackknife/influence estimator 估计逐坐标 reference variance 和 NMSE 所需 trace；两阶段 bootstrap（outer estimator repetitions，inner reference blocks）负责后续联合区间。
8. 对 model-total 定义 `h_ref` 为总量 bootstrap 区间半宽；对 layer/module L1 定义为 bootstrap 分布中相对中心 reference 的聚合 L1 误差 95% 上分位。数值误差按相同聚合顺序形成 `epsilon_num`。

### 3.6 执行参考收敛检查

1. 比较相邻翻倍节点的参考贡献 normalized L1 difference。
2. 比较相邻节点的 Pearson；Spearman 主 Gate 只应用于按预注册规则定义的 signal-eligible 坐标以及 layer/module 聚合，完整 parameter Spearman 保留为描述性指标。
3. Signal-eligible 规则固定为参考贡献下界高于 `max(5 × reference half-width, 10 × numerical floor)`；规则本身在查看 estimator 结果前冻结。
4. 比较 top `0.1%`、`1%`、`5%` 的 Overlap 与 Jaccard；`0.01%` 作为更敏感的诊断项。
5. 比较 layer/module 总量和每参数平均量。
6. 检查 A/B 两个独立 reference streams 是否满足相同标准。
7. 检查 `C_bias_ref` 与 `C_cross` 的差异是否由 reference 区间覆盖；`C_rank_ref` 的正向差异须与估计的有限平方偏差方向一致。
8. 对每个主 cell/endpoint 检查 `h_ref <= min_B(delta_sci(B))/4`；最小值遍历 S2.1 的全部候选 B，因此 G2.3 不依赖尚未选择的 `B_primary`。若不满足，本轮标为证据不足/阻断，不能改变 margin、延长 A/B 或重抽到通过。

### 3.7 建立数值误差地板

1. 从高、中、低 SNR 参数和不同 layer/module 预注册一组诊断坐标。
2. 对该坐标集合保留基础 microbatch 梯度，并用显式 pairwise/高精度计算复核累计值。
3. 比较 FP32、FP64 累加与不同 block 切分的结果。
4. 将观测数值误差上界登记为 \(\varepsilon_{\mathrm{num}}\)。
5. 若数值误差接近预计 bias，不进入正式 estimator 比较，先调整累加精度或缩小诊断范围。
6. 对每个主 endpoint 要求 `epsilon_num <= min_B(delta_sci(B))/10`；失败时停止并改进数值实现，不能把数值误差并入等价 margin。

### 3.8 完成状态不变和重放检查

1. Reference 完成后重新生成模型参数、buffer 和模型/全局 RNG 摘要，并保存 sampling generator 结束状态。
2. 模型状态与模型/全局 RNG 必须与运行前相同；sampling generator 应与冻结 draw 消耗量一致并可重放，而不是要求起止相等。
3. 从中间 block checkpoint 重放最后两个 block，检查最终累计数组和哈希一致。
4. 用相同 manifest 完整重放一个小 reference 前缀，验证顺序和结果确定性。

## 4. 产出

服务器大型产物（全部位于 `$DATA_ROOT/results/stage2/reference/<reference-id>`，运行状态位于 `$DATA_ROOT/runs/stage2/<reference-id>`）：

- 每个 model × checkpoint 的 FP64/验证精度参考梯度；
- 无偏 bias reference、交叉敏感性 reference、低方差 ranking reference 和 A/B 诊断；
- 逐参数方差、block summaries 和高精度诊断子集；
- 可恢复累计状态和 reference 运行日志。

Git 中的小型产物：

- reference manifest、schema、路径、大小和哈希摘要；
- sizing stream 报告、冻结 \(B_{\mathrm{ref}}\) 的 amendment，以及最终 A/B 与 sizing 独立性证明；
- 相邻样本量收敛指标表；
- A/B 一致性表；
- reference convergence 图；
- 每个 checkpoint 的 reference Gate 结果 JSON/Markdown。

## 5. 核验标准与 Gate

每个 model × checkpoint 必须同时满足以下条件才通过 **G2.3 参考 Gate**：

- 相邻两个最终翻倍节点的 normalized L1 difference 不超过 2%；
- 全参数 Pearson、signal-eligible 参数 Spearman 以及 layer/module Spearman 均不低于 0.995；完整 parameter Spearman 只作描述，不单独阻断 near-zero 坐标占主导的参考；
- top `0.1%`、`1%`、`5%` overlap 均不低于 0.98；
- layer/module 总量与每参数平均量相邻变化均不超过 1%；
- reference_A 与 reference_B 满足同级排序标准，且差异由 block 区间覆盖；
- 每个主 endpoint 的 reference 半宽不超过全部候选 B 中最小 `delta_sci(B)/4`；
- 无偏 block-U reference 与交叉 reference 的差异由区间覆盖；ranking reference 的有限正偏已显式标注；
- 最终 \(B_{\mathrm{ref}}\) 在 A/B draw 生成前由独立 sizing stream 冻结，A/B 全长运行无可选停时；
- 本轮只存在一个 one-shot 最终 A/B reference；失败版本没有被新抽样替换或选择性接受；
- 单 sequence 方差与 block 方差缩放在诊断坐标上一致，raw 预测未漏掉 block-size 因子；
- 数值误差不超过全部候选 B 中最小 `delta_sci(B)/10`；
- 运行前后模型状态一致，恢复重放哈希一致。

如果只有少量 near-zero 坐标无法满足相对条件，可保留为近零分箱并使用绝对误差标准；不得删除这些坐标或用相对误差异常扩大来宣称整体失败/成功。

## 6. 后续依赖

- S2.5 读取 `bias_ref`、`cross_ref`、`rank_ref` 与方差预测量，但不能修改它们。
- S2.6 用 reference 半宽确定 repetition 精度预算。
- S2.8 必须以 `bias_ref` 做偏差主判据，并用 `cross_ref` 做敏感性分析；`rank_ref` 只负责排序。
- 任一 checkpoint 的 reference Gate 失败时，该 checkpoint 不得进入 S2.7。
