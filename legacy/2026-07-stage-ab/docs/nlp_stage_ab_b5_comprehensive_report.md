# NLP Stage A、Stage B 与 Stage B.5 参数重要性实验统一总结报告

## 0. 文档定位

本文把截至 2026-07-17 已完成的 NLP 参数重要性实验整理为一份统一、可独立阅读的中文报告，覆盖：

- 理论目标、raw、double sampling 与 Microbatch-level U-statistic 的关系；
- signed、positive-only、near-zero 与 most-negative 等记号；
- Stage A 的工程正确性、估计器偏差、数值积分与稳定性实验；
- Stage B 的 Pythia-style 160M 预训练、SST-2 两路线、剪枝和功能复用干预；
- 原 Stage 15.8 Gate 5 的正式失败；
- Stage B.5 对 signed U 与 signed double 的对称诊断；
- 当前能够支持、只能暂定和不能支持的科学结论；
- Stage C 正式 410M 实验启动前必须完成的方法修订和门禁。

本文不覆盖或改写任何旧产物。特别地，原 Gate 5 的正式状态仍为失败；Stage B.5 是独立的事后诊断，只更新科学解释，不追溯修改预注册门禁。

## 1. 现有材料及其关系

仓库与服务器已经存在以下正式材料：

1. `docs/parameter_importance_u_statistic_report.md`
   - 理论推导文档；
   - 解释同源噪声偏差、double sampling 的无偏性，以及 Microbatch-level U-statistic 的无偏性和方差；
   - 不包含 Stage A/B 的实际运行结果。
2. `plan/NLP参数重要性完整实验计划书.md`
   - 完整研究计划、阶段划分、指标和停止门槛；
   - Stage C、D、E、F 尚未完成。
3. `docs/stage_ab_runbook.md`
   - Stage A/B 正式执行、失败保留、产物验证和 B.5 诊断入口。
4. `worklogs/2026-07-16-nlp-stage-ab-final-summary.md`
   - Stage A/B 最小闭环正式总结；
   - 记录原 Gate 5 失败；
   - 形成时间早于 Stage B.5。
5. `docs/stage_b_u_double_near_zero_diagnostic_summary.md`
   - Stage B.5 的完整设计、结果、哈希和更新解释；
   - 只聚焦 U/double 与低端语义问题。
6. 服务器生成报告：
   - `/home/sophgo13/cjl/storage/parameter-importance/reports/stage-ab-minimum-loop/report.md`；
   - 同目录包含 30 个表文件和 48 个图文件；
   - 报告登记 77 个测量来源，`missing=[]`。

此前缺少的是一份把理论、全部实验、相互容易混淆的数字和 B.5 后的更新结论放在同一口径下的报告，本文用于填补这一缺口。

## 2. 实验真正研究的量

### 2.1 局部梯度空间目标

设第 $t$ 个训练状态下，参数 $k$ 对总体数据分布的平均梯度为

\[
\mu_{k,t}=\mathbb E_z\left[\frac{\partial L(z;\Theta_t)}{\partial\theta_k}\right].
\]

正式训练采用的主目标是

\[
C_{k,t}^{\mathrm{grad}}=\eta_t\mu_{k,t}^2.
\]

若该步发生全局梯度裁剪，裁剪系数为 $s_t\in(0,1]$，则正式记录的裁剪调整目标为

\[
C_{k,t}^{\mathrm{grad,clip}}=\eta_t s_t\mu_{k,t}^2.
\]

它来自一阶近似：若梯度下降更新约为 $-\eta_t\mu_{k,t}$，则该参数对数据损失下降的局部贡献约为 $\eta_t\mu_{k,t}^2$。

正式结论只能称为：

> 经 Microbatch-level U-statistic 去偏的局部梯度空间损失贡献。

不能称为 AdamW 完整参数更新路径积分的严格无偏估计。AdamW 的动量、自适应缩放和解耦权重衰减不属于上述无偏性证明；实验另外保存 actual-update raw 诊断，并从真实位移中剔除 decoupled weight decay。

### 2.2 记号速查

| 记号 | 含义 |
|---|---|
| $k$ | 参数坐标 |
| $t$ | 训练步骤或训练状态 |
| $m$ | 一个 microbatch |
| $M$ | 一个估计中使用的等大小 microbatch 数量 |
| $r$ | 固定状态实验中的独立重复编号 |
| $g_{m,k,t}$ | 第 $m$ 个 microbatch 对参数 $k$ 的平均梯度 |
| $\eta_t$ | 当前学习率 |
| $s_t$ | 全局梯度裁剪系数 |
| $\mu_{k,t}$ | 总体平均梯度 |

## 3. 三种估计器

### 3.1 Raw：同批均值梯度平方

\[
\widehat C_{k,t}^{\mathrm{raw}}
=
\eta_t\left(\frac1M\sum_{m=1}^{M}g_{m,k,t}\right)^2.
\]

在 microbatch 独立同分布的局部设定下，

\[
\mathbb E[\widehat C^{\mathrm{raw}}]
=
\eta\left(\mu^2+\frac{\operatorname{Var}(g)}M\right).
\]

因此 raw 把有限样本梯度噪声的方差也计入重要性，存在系统性正偏差。

### 3.2 Double sampling：独立两组的乘积

使用独立的两组样本计算平均梯度 $\bar g_A,\bar g_B$：

\[
\widehat C^{\mathrm{double}}=\eta\bar g_A\bar g_B.
\]

因为两组独立，

\[
\mathbb E[\bar g_A\bar g_B]=\mu^2.
\]

它与目标一致且无偏，但需要独立梯度组。在 Stage A 的等总样本预算比较中，$M$ 个 microbatch 被分成前后两半构造 double；在正式训练中，double 仍意味着额外保存或计算独立估计路径，因此不适合作为每一步的低成本默认方法。

### 3.3 Microbatch-level U-statistic：去掉自乘项

\[
\widehat C^U
=
\eta
\frac{
\left(\sum_m g_m\right)^2-\sum_m g_m^2
}{M(M-1)}
=
\eta\frac1{M(M-1)}\sum_{i\ne j}g_i g_j.
\]

它去掉同一个 microbatch 与自身相乘的对角项，只保留不同 microbatch 的交叉项，因此

\[
\mathbb E[\widehat C^U]=\eta\mu^2.
\]

U-statistic 的核心价值是：复用训练中已经计算的 microbatch gradient，在不额外抽取完整独立 probe batch 的情况下构造与 double 相同目标的无偏估计。

## 4. Signed 与 positive-only 不是同一个估计量

### 4.1 Signed 累计

\[
\Omega_k^{\mathrm{signed}}
=
\sum_{t=0}^{T-1}\widehat C_{k,t}^{U,\mathrm{clip}}.
\]

即使真实目标非负，有限样本无偏估计也可能暂时为负。signed 保留这些负值，使正负采样误差可以在重复或训练轨迹上抵消。

### 4.2 Positive-only 累计

\[
\Omega_k^+
=
\sum_{t=0}^{T-1}\max(0,\widehat C_{k,t}^{U,\mathrm{clip}}).
\]

Stage A 固定状态实验中的

\[
\operatorname{mean}_r\bigl(\max(U_r,0)\bigr)
\]

是同一变换在重复采样维度上的版本。它不是绝对值，而是 ReLU 或单边截断：负值变为 0，正值不变。

一般有

\[
\mathbb E[\max(U,0)]
\ge
\max(\mathbb E[U],0),
\]

且通常严格大于。因此 positive-only 是稳定、非负、便于操作的分数，但不再是原 (\eta\mu^2) 的无偏估计量。

例如同一参数两步估计为 (+10,-9)：

\[
\Omega^{\mathrm{signed}}=1,
\qquad
\Omega^+=10.
\]

该参数最终 signed 分数仍为正，但 positive-only 已被放大十倍。Stage B.5 的六个正式来源中，positive-only 总质量是 signed U 净质量的 $1.470$–$1.606$ 倍，说明该变换会系统性保留瞬时正贡献并删除本应发生的抵消。

### 4.3 Signed 低端的三个语义

signed 分数必须区分：

- high-positive：最大正分数；
- near-zero：绝对值最接近零；
- most-negative：最负分数。

most-negative 不能直接称为“最不重要”。它可能是较大的反向估计、噪声或抵消结构，应与 near-zero 分开报告。

## 5. 指标怎样解释

### 5.1 参考重要性向量

Stage A 估计器偏差实验在固定 checkpoint 上，用 4096 个参考样本计算高预算平均梯度：

\[
\mu_k^{\mathrm{ref}}
=
\frac1{4096}\sum_{i=1}^{4096}g_k(z_i),
\qquad
C_k^{\mathrm{ref}}
=
\eta(\mu_k^{\mathrm{ref}})^2.
\]

所有参数组成参考重要性向量

\[
\mathbf C^{\mathrm{ref}}
=(C_1^{\mathrm{ref}},\ldots,C_P^{\mathrm{ref}}).
\]

它是高预算近似，不是绝对真值；也不存在一组特殊的“参考参数”。U 或 double 的 Spearman 是其估计向量与该参考向量在参数维度上的排序相关。

### 5.2 Spearman 不是估计器的固定属性

Spearman 必须同时写明：

- 比较哪两个向量；
- 参数、层还是模块粒度；
- 单 checkpoint、相邻 $M$、相邻 checkpoint 还是最终累计；
- signed 还是 positive-only。

因此 (0.5667)、(0.5833) 与 (0.969) 可以同时成立，因为它们比较的根本不是同一对向量。

### 5.3 剪枝 effect 与 gap

对 accuracy，effect 定义为 baseline accuracy 减剪枝后 accuracy；对 NLL，effect 定义为剪枝后 NLL 减 baseline NLL。二者越大都表示模型损害越严重。

\[
\operatorname{gap}(r)
=
\operatorname{effect}_{\mathrm{high}}(r)
-
\operatorname{effect}_{\mathrm{low}}(r).
\]

gap 为正表示剪高分参数比剪低分参数更伤模型。

### 5.4 Global 与 layer-balanced

- global：在全部 eligible 参数中统一排序；
- layer-balanced：每层剪相同比例，再在层内排序。

layer-balanced 用于排除层大小、参数数量和梯度尺度主导全局排序的混杂，不能省略。

## 6. Stage A：工程与方法验证

### 6.1 实验规模

- 模型：Pythia-style 14M；
- 轨迹：32 step；
- estimator checkpoint：10 个，step `0/1/2/4/8/12/16/20/26/30`；
- microbatch 数：$M\in\{4,8,16,32\}$；
- 每个 `checkpoint × M`：200 次重复；
- reference：4096 个样本；
- 共 40 个 estimator 原子单元。

### 6.2 工程正确性

Stage A/B 代码链验证了：

- 单 GPU 与多 GPU 对同一 global batch 的梯度等价；
- $M=1$ 和不等大小 microbatch fail closed；
- 中断恢复后模型、优化器、scheduler、RNG、data cursor 与累计 importance 一致；
- gradient clipping factor 正确进入主分数；
- decoupled weight decay 不进入数据损失贡献；
- causal shift、mask、SST-2 标签 token loss 和剪枝 mask 正确；
- random pruning 固定 seed 可复现；
- checkpoint、manifest、文件大小和 SHA-256 均参与正式门禁。

Stage A/B 最小闭环最终服务器报告绑定提交 `7e3a3d8163e0d00553e45daa1d312494d3c4ee71`：151 tests、0 failures、0 errors、0 skips；显式分布式与恢复集成测试为 2 passed。

### 6.3 固定状态估计器偏差结果

40 个 `checkpoint × M` 条件中，每个估计器先对 200 次重复取平均，再与参考向量比较。正式聚合结果为：

| 指标 | 结果 | 解释 |
|---|---:|---|
| layer U/raw absolute-bias ratio | 0.02037 | U 的层级绝对偏差远小于 raw |
| module U/raw absolute-bias ratio | 0.02000 | U 的模块级绝对偏差远小于 raw |
| U 对参考参数 Spearman 均值 | 0.49731 | 单固定状态的细粒度排序仍有明显噪声 |
| double 对参考参数 Spearman 均值 | 0.47996 | double 同样不是精确排序真值 |
| U/double repetition-MSE ratio | 0.91496 | U 在该预算下不劣于 double |
| U/double formula-time ratio | 0.765 | 只表示估计公式计时，不等于完整训练墙钟成本 |
| maximum dominance fraction | 0.01258 | 没有少量参数支配全部估计 |

这组结果支持：raw 的同源噪声偏差可观测；U 对 raw 的去偏有效；在相同固定状态预算下 U 不弱于 double。但参数 Spearman 约 0.5 也说明，不能把单 checkpoint、有限 $M$ 的参数精排视为已解决问题。

### 6.4 Signed 的相邻 $M$ 稳定性

scope-aware v2 比较相邻 $M$ 下已经跨 200 次重复平均的 signed 向量：

| 粒度 | 最低相邻 $M$ Spearman | 门槛 | 状态 |
|---|---:|---:|---|
| 参数 | 0.39572 | 描述值 | 不稳定 |
| 层 signed-sum | 0.58333 | 0.90 | 失败 |
| 层 signed-mean | 0.80 | 诊断值 | 仍不足 |
| 模块 signed-sum | 0.93333 | 0.90 | 通过 |

失败集中在 step26 的 $16\rightarrow32$（0.66667）和 step30 的 $8\rightarrow16$（0.58333）。全部层级 top-5% overlap 都为 1.0；由于只有 9 个层组，这实际上相当于 top-1 一致。

因此 signed U 的遗留问题不是“完全随机”，而是中间层级排序对 $M$ 和抵消较敏感；最顶层功能组与模块级排序更稳定。原 formal stability gate 仍为失败。

### 6.5 Positive-only v3 的稳定性

positive-only v3 没有对已经平均的 signed 向量事后截断，而是正式重算每个 repetition 的

\[
U_r^+=\max(U_r,0),
\qquad
\bar U^+=\operatorname{mean}_r(U_r^+).
\]

其 $M\ge8$ 最低稳定性为：

| 粒度 | 最低 Spearman | 状态 |
|---|---:|---|
| 参数 | 0.9299466 | 通过 |
| 层 | 0.9166667 | 通过 |
| 模块 | 0.95 | 通过 |

它证明 positive-only 是稳定的操作性分数，但不证明 signed 目标已经稳定，也不保留 signed U 的无偏性。

### 6.6 数值积分

数值积分 v1 在 20 个独立路径状态上比较 left、trapezoid、Simpson 与 16-node Gauss-Legendre 参考：

- left 参数 nonzero Spearman 最低 0.71049，超过 0.70；
- left 参数 top-5% overlap 最低 0.86401，超过 0.60；
- left 层 Spearman 最低 0.56667，低于 0.90，正式失败；
- trapezoid、Simpson、Gauss 的层级最低值均为 1.0。

随后独立运行低频两节点梯形 v2，并使用新的 update/probe 数据区间：

| 指标 | 最低值 | 门槛 | 状态 |
|---|---:|---:|---|
| 参数 nonzero Spearman | 0.98322 | 0.70 | 通过 |
| 层 Spearman | 0.96667 | 0.90 | 通过 |
| 参数 top-5% overlap | 0.97389 | 0.60 | 通过 |

因此 0.56667 只说明左端点积分不可靠，不是 signed U 的排序结果。正式 Stage B 使用通过门禁的 trapezoid 路径。

## 7. Stage B：Pythia-160M 与 SST-2 最小闭环

### 7.1 正式预训练

- 模型：Pythia-style 160M；
- seed：1234；
- 数据：固定 Pythia deduplicated Pile 前缀；
- token：约 1.074B；
- steps：512；
- checkpoint：`0/1/2/4/8/16/32/64/128/256/512`；
- 正式实验 ID：`stage-b-160m-step512-seed1234-deterministic-trapezoid-v3`。

正式 v3 从 step0 开始，不从旧 v1/v2 诊断运行续接。最终 step512 checkpoint 的 manifest、`_SUCCESS`、文件大小与全部哈希通过。

固定 WikiText 验证结果：

| 指标 | 结果 |
|---|---:|
| step0 perplexity | 63585.31 |
| step512 perplexity | 423.65 |
| 相对公开 Pythia step512 的 PPL ratio | 0.6396877 |
| 预注册最大 ratio | 1.15 |

训练 loss/validation 总体下降，无 NaN/Inf 和不可恢复 spike；strict algorithms、`warn_only=false`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、eager/math SDPA 和禁用非确定性后端均有运行证据。该结果证明最小预训练是健康、可复现的，不等价于完整规模 Pythia 训练已经完成。

### 7.2 SST-2 离线基线

固定 GLUE/SST-2 validation 共 872 个样本，标签计数 `{0: 428, 1: 444}`：

| 基线 | Accuracy | NLL | ECE |
|---|---:|---:|---:|
| 多数类 | 0.5091743 | 0.6929788 | 0.0 |
| uniform random 期望 | 0.5 | 0.6931472 | 0.0 |
| 公开 Pythia-160M step0，未微调 | 0.4816514 | 0.7181957 | 0.1111430 |
| 公开 Pythia-160M step512，未微调 | 0.4908257 | 1.0809826 | 0.3642443 |

公开 checkpoint 的零样本 prompt 结果不是监督训练性能，只用于证明离线资产、prompt、verbalizer 与公开来源身份可正确评价。

### 7.3 Controlled 两路线

两条路线保持相同架构、tokenizer、step0、下游数据顺序、batch、最大步数和评价频率：

- direct：从共同 step0 随机初始化直接训练 SST-2；
- pretrained：从正式 160M step512 开始微调 SST-2，微调 importance 与预训练 importance 分开累计。

三 seed controlled 最佳验证 accuracy：

| 路线 | 三 seed 均值 | 最低 seed | 最高 seed |
|---|---:|---:|---:|
| direct | 0.8100153 | 0.7970183 | 0.8176606 |
| pretrained | 0.8176606 | 0.8165138 | 0.8188073 |

pretrained 均值比 direct 高约 0.00765，即 0.765 个百分点。两条路线均显著高于多数类基线，满足有效学习门槛；但收益幅度有限。

### 7.4 匹配性能位置

报告按预注册规则寻找两路线 accuracy 差不超过 0.01 的可比较位置：

| Seed | Direct | Pretrained | 说明 |
|---:|---|---|---|
| 1234 | step1300，acc 0.82225，NLL 0.45321 | step1700，acc 0.82225，NLL 0.43578 | accuracy 相同，pretrained NLL 较低 |
| 1337 | step1578，acc 0.83142，NLL 0.41034 | step1400，acc 0.83486，NLL 0.43512 | pretrained 略早且 acc 略高，direct NLL 较低 |
| 2027 | step2900，acc 0.82110，NLL 0.53180 | step2500，acc 0.82110，NLL 0.51329 | accuracy 相同，pretrained 较早且 NLL 较低 |

它支持“预训练后路线有小幅或优化路径上的优势”，但不支持“预训练在所有指标、所有 seed 都大幅胜出”。

### 7.5 Best-performance LR 搜索

best-performance 与 controlled 机制实验使用不同实验 ID，按最佳 validation NLL 独立选择，不参与正式 pruning 来源：

| 路线 | 选择 LR | Best step | Accuracy | NLL |
|---|---:|---:|---:|---:|
| direct | $3\times10^{-4}$ | 800 | 0.8061927 | 0.4173166 |
| pretrained | $1\times10^{-4}$ | 1000 | 0.8188073 | 0.4116327 |

该搜索只覆盖 seed1234 和有限 LR 网格，不能替代三 seed controlled 结论。

### 7.6 如何解释有限的预训练收益

当前结果合理但范围有限：

- Stage B 是 512 step、约 1.074B token 的最小闭环，不是完整 Pythia 训练；
- SST-2 是相对简单的二分类任务，direct 已达到约 81%，剩余提升空间较小；
- 预训练与直接监督路线的最优学习率不同；
- 只有一个模型规模、一个下游任务和三个 controlled seed；
- 当前只应说“有一致但有限的收益”，不能推断预训练普遍无效或普遍大幅有效。

## 8. Stage B 剪枝验证

### 8.1 设计与来源门禁

正式剪枝覆盖：

- direct/pretrained × seed `1234/1337/2027`，共六个来源；
- 每个来源精确 594 条记录；
- 比例 `0/0.1%/0.5%/1%/2%/5%/10%/20%/30%`；
- global 与 layer-balanced；
- U-stat、raw、double、magnitude、movement 与 random 等方法；
- accuracy、NLL 与 ECE。

六项都绑定对应 controlled 路线的最终 step3156 模型和 importance。best-validation 只作评价参考，`used_for_pruning=false`；best-performance 和诊断产物没有混入正式来源。六项均通过 `_SUCCESS`、manifest、结果行数、rank、curve、provenance 和哈希验证。

eligible 布局在六个来源中一致：49 个 tensor、123,568,128 个标量参数。

### 8.2 功能区分度

高 positive U 参数剪枝比低 positive U 参数剪枝造成更大损害，覆盖的 64 个非零比例 accuracy/NLL curve group 全部方向为正。

absolute gap AUC：

| 方法 | AUC |
|---|---:|
| double | 0.0759237 |
| U-statistic | 0.0725815 |
| raw | 0.0716858 |
| movement | 0.0558577 |
| magnitude | 0.0287843 |

U-statistic 略高于 raw，明显高于 movement 和 magnitude；double 数值最高。该结果支持 U 分数具有功能意义，但 U 相对 raw 的优势很小，不能夸大为压倒性改进。

### 8.3 原 Gate 5

原 Gate 5 要求 U 与 double 的排序相关和剪枝效果相近或更好。功能相近判据为

\[
\operatorname{gap}_U\ge0,
\qquad
\operatorname{gap}_U\ge0.8\operatorname{gap}_{double}.
\]

原实现实际比较：

- U 侧：positive-only high 对 positive-only low；
- double 侧：signed high-positive 对 signed most-negative。

正式结果为 68/72，通过 68 项、失败 4 项，因此 Gate 5 正式失败。四项均为 layer-balanced NLL：

| 路线 | 比例 | U gap | Double gap | 要求的 U gap |
|---|---:|---:|---:|---:|
| direct | 30% | 0.1580694 | 0.2163268 | 0.1730614 |
| pretrained | 10% | 0.1651991 | 0.2065118 | 0.1652094 |
| pretrained | 20% | 0.1276361 | 0.2145851 | 0.1716681 |
| pretrained | 30% | 0.0770344 | 0.2213179 | 0.1770543 |

pretrained 10% 仅以约 $1.0\times10^{-5}$ 的差距失败，20%/30% 则是实质差距。阈值没有事后修改。

## 9. 功能复用干预

### 9.1 设计

三个 seed 各精确 60 条，共 180 条正式记录。干预分别评价：

1. 按预训练 high mask 剪未微调模型；
2. 按预训练 high mask 剪已微调模型；
3. 按微调 high mask 剪已微调模型；
4. 剪预训练与微调 top-k 交集；
5. 剪只属于预训练、不属于微调的参数；
6. 覆盖六个比例与 global/layer-balanced。

三份结果均核验共同的 v3 预训练 step512、对应 pretrained controlled step3156、importance 身份、模型哈希、因子网格、provenance 与 `_SUCCESS`。`coverage_gate_passed=true`，required/valid seeds 均为 `1234/1337/2027`，普通 pruning 没有冒充复用干预。

### 9.2 能够得出的结论

统计 overlap、Jaccard、enrichment、全参数/层/模块 Spearman 和功能干预均已产出。功能效应会随 mask 来源、比例和 allocation 改变。

计划没有为功能复用干预预注册统一 effect-size 通过阈值，因此当前只能描述：

> 预训练与微调的重要性之间存在可测量的统计重合和功能效应，但现有单任务最小闭环不足以宣称稳定、普遍的功能复用规律。

不得根据事后看到的效应大小新增阈值并把它写成正式通过。

## 10. Stage 15.8 原正式门禁

| # | 门禁 | 原正式状态 |
|---:|---|---|
| 1 | 预训练质量门槛 | 通过 |
| 2 | SST-2 两路线高于多数类基线 | 通过 |
| 3 | 高 U-stat positive 剪枝损害大于低重要性剪枝 | 通过 |
| 4 | U-stat 区分度不低于 raw | 通过 |
| 5 | U-stat 与 double sampling 排序和剪枝效果相近或更好 | **失败** |
| 6 | signed/positive-only 至少一套产生稳定功能排序 | 通过 |
| 7 | 中断恢复和多 GPU 可复现 | 通过 |

按原计划“若不通过不得扩大到 410M”，不能仅凭 Stage A/B 原报告直接启动正式 Stage C。必须先形成独立的方法修订和新的前置门禁。

## 11. Stage B.5：U/double 对称低端诊断

### 11.1 为什么需要 B.5

原 Gate 5 同时混入两个不对称因素：

1. positive-only U 与 signed double 不是同一个估计对象；
2. U low 与 double most-negative 不是同一种低端语义。

B.5 的目的不是重新训练或改写 Gate 5，而是区分失败到底来自：

- double sampling 本身不可靠；
- most-negative 与 near-zero 的语义不对称；
- positive-only 改变了 U 的估计对象；
- 或 signed U 本身确实不能近似 signed double。

### 11.2 设计

B.5 复用 direct/pretrained × 三 seed 的六个最终 checkpoint、importance 和原 pruning 结果，只新增：

\[
\text{double-near-zero}
=
\text{按 }|\text{累计 signed double score}|\text{ 从小到大剪枝}.
\]

覆盖六个来源、两个 allocation 和九个比例，共精确 108 条新评估。每次从同一最终模型恢复，在固定 872 个 SST-2 validation 样本上评价。

### 11.3 功能结果

| 比较 | 通过 | 失败 | 解释 |
|---|---:|---:|---|
| 原 Gate 5：positive U low 对 signed double most-negative | 68/72 | 4 | 原正式失败复现 |
| positive U low 对 signed double near-zero | 69/72 | 3 | 低端语义只解释 1 项 |
| signed U near-zero 对 signed double near-zero | 72/72 | 0 | 对称 signed 比较全部满足旧功能判据 |

只修正 double 低端后，pretrained 10% 被修复，direct 30% 和 pretrained 20%/30% 仍失败；恢复 signed U 后全部通过。

四种分数定义自己的 high-low/near-zero gap 在 64 个非零比例主指标组中均为正：

| 分数与低端定义 | 正 gap | 平均 gap |
|---|---:|---:|
| positive U / low | 64/64 | 0.2436897 |
| signed U / near-zero | 64/64 | 0.2498107 |
| signed double / most-negative | 64/64 | 0.2492449 |
| signed double / near-zero | 64/64 | 0.2492822 |

double-near-zero 自身并没有失去功能区分度，因此现有结果不支持“double sampling 基准本身很差”。

### 11.4 排序结果

| 分数对 signed double | 参数 Spearman | Layer Spearman | Module Spearman |
|---|---:|---:|---:|
| positive U | 0.913501–0.935414 | 0.846154–1.0 | 0.9–1.0 |
| signed U | 0.962596–0.969190 | 1.0 | 1.0 |

最终累计 signed U 与 signed double 在当前六个来源上高度一致，且明显比 positive-only 对 double 更接近。

### 11.5 符号与质量

- signed U 最终为负的标量比例：0.9016%–1.0109%；
- signed double 最终为负的标量比例：1.2191%–1.9017%；
- signed U 净质量/绝对质量：0.998596–0.999275；
- signed double 净质量/绝对质量：0.996679–0.998398；
- positive-only 总质量/signed U 净质量：1.470–1.606。

最终负参数虽然只有约 1%，positive-only 仍大幅改变总质量，因为截断发生在每一步或每次重复之前，而不是只在最终累计后处理负参数。

### 11.6 B.5 更新后的解释

B.5 支持：

> 在当前 Pythia-160M、SST-2、direct/pretrained、三 seed 范围内，最终累计 signed U 在排序与对称剪枝功能上都高度接近 signed double。

B.5 不支持：

- 追溯把原 Gate 5 改成通过；
- 宣称 signed U 的单 checkpoint、低 $M$ 稳定性问题已经消失；
- 宣称所有模型规模和任务都可删除 double；
- 把 positive-only 继续称为同一个无偏估计量。

## 12. 所有易混淆数字的归位

| 数值 | 比较对象 | 正确含义 |
|---:|---|---|
| 0.56667 | left endpoint 对 Gauss 参考的层级排序 | 左端点积分失败，与 signed U 无关 |
| 0.98322/0.96667 | trapezoid 对 Gauss 的参数/层级排序 | 梯形积分通过 |
| 0.49731 | U 对 4096-sample 参考向量的参数 Spearman 均值 | 固定状态参数精排噪声较大 |
| 0.47996 | double 对同一参考向量的参数 Spearman 均值 | double 也不是参数级真值，U 不弱于它 |
| 0.39572 | signed U 相邻 $M$ 参数 Spearman 最低值 | 参数级局部稳定性不足 |
| 0.58333 | signed U 相邻 $M$ 层 signed-sum Spearman 最低值 | signed U 的正式局部稳定性失败 |
| 0.93333 | signed U 相邻 $M$ 模块 Spearman 最低值 | 模块级通过 |
| 0.92995/0.91667/0.95 | positive-only 参数/层/模块稳定性最低值 | positive-only 稳定，但已改变目标 |
| 0.91350–0.93541 | 最终 positive U 对 double 参数排序 | 相关较高，但对象不同 |
| 0.96260–0.96919 | 最终 signed U 对 signed double 参数排序 | 当前最强的替代支持证据 |
| 68/72 | 原非对称 Gate 5 功能比较 | 正式失败保留 |
| 72/72 | B.5 signed/near-zero 对称功能比较 | 事后诊断全部满足，不追溯放行 |

## 13. 当前科学结论分级

### 13.1 证据较强

1. raw 同批均值梯度平方存在可观测的同源噪声正偏差。
2. Microbatch-level U-statistic 在局部梯度空间目标下理论无偏，固定状态实验中对 raw 的偏差显著更小。
3. 在相同 Stage A 预算下，U 的平均参数排序和 MSE 不弱于 double。
4. trapezoid 数值积分在 Stage A 门槛上通过，left endpoint 不可作为正式方法。
5. U 分数具有功能意义：剪高分参数比剪低分或 near-zero 参数更伤模型。
6. 最终累计 signed U 在当前六个 SST-2 来源上与 signed double 高度一致。
7. positive-only 是不同的有偏操作性分数，不能用于证明 signed U 与 signed double 等价。
8. Pythia-160M 最小预训练健康；direct 与 pretrained SST-2 都有效学习任务。

### 13.2 有限范围内成立

1. 预训练在当前最小闭环中带来约 0.765 个百分点的 controlled accuracy 均值提升，但不是巨大优势。
2. 预训练后路线在部分匹配性能位置更早或 NLL 更低，但并非每个 seed、每个指标都占优。
3. U-statistic 的功能 AUC 略高于 raw，但优势很小；double 在该汇总上仍最高。
4. 最终轨迹累计可能显著平滑 signed U 的局部噪声，B.5 支持这一现象，但尚未跨模型和任务复现。
5. 预训练与微调重要性存在统计重合和功能效应，但复用强度只能描述，尚无预注册通过阈值。

### 13.3 尚未解决或不能声称

1. signed U 在单 checkpoint、有限 $M$ 下的参数和层级稳定性尚未通过。
2. 不能无条件删除所有 double sampling；目前只能考虑把它降为稀疏校准基线。
3. 不能把结论推广到 Pythia-410M、MNLI、RTE 或其他模型家族。
4. 不能把局部 gradient-space contribution 解释为完整 AdamW 路径贡献。
5. 不能把 most-negative 参数直接称为最不重要参数。
6. 不能把 B.5 当作原 Gate 5 的追溯通过。
7. 不能宣称预训练参数已经被下游任务稳定、普遍地功能复用。

## 14. 为什么仍应以 signed U 作为候选科学主估计量

signed U 的问题是方差和局部排序稳定性；positive-only 的问题是改变了估计对象。二者不是同一种缺陷。

可写为：

\[
\widehat C^U=C+\varepsilon,
\qquad
\mathbb E[\varepsilon]=0.
\]

signed U 仍测量原始目标，只是有限样本噪声较大。positive-only 则测量

\[
\max(C+\varepsilon,0),
\]

其期望通常不等于 $C$。科学上不能仅为获得稳定排名就把目标换成有偏量；应优先通过增加 $M$、独立分组、窗口累计和不确定性分析降低方差。

完整轨迹累计为

\[
\Omega_k^{\mathrm{signed}}
=
\sum_t C_{k,t}+\sum_t\varepsilon_{k,t}.
\]

不同步骤的噪声若不完全同向，会发生部分抵消。B.5 的最终累计 Spearman 和 72/72 对称剪枝结果是这种平滑在当前实验中有效的经验依据。

因此最准确的表述是：

> signed U 是与研究问题一致的候选科学主估计量，但正式进入 Stage C 仍以新的稳定性 pilot 通过为前提；“主估计量”不等于“已无条件验收”。

## 15. Stage C 前必须修改和预注册的事项

### 15.1 方法口径

1. signed U 作为候选科学主估计量；
2. positive-only 只作为明确标注的有偏 operational ablation；
3. negative 单独保存，不与 near-zero 合并；
4. 最终排序优先使用预注册的累计轨迹或固定窗口累计，而不是任意单 checkpoint；
5. 继续把 actual-update、raw、movement 和 magnitude 作为不同含义的基线。

### 15.2 稳定性 pilot

正式 410M 长训练前，至少在早期、中期和后期代表性状态上：

1. 使用相互独立的 microbatch 分组重复计算 signed U；
2. 覆盖 Stage C 实际候选 $M$，计划书建议 128 或 256；
3. 同时报告单 checkpoint、相邻 $M$、固定累计窗口的参数/层/模块 Spearman；
4. 报告 top-1%、5%、10% overlap 和 signed 正负质量；
5. 保留至少 layer/module 0.90 的既有稳定性门槛，任何新增参数级或累计门槛必须在看到 Stage C 正式结果前固定；
6. 若 fixed-state pilot 可复用既有 checkpoint，则无需重新完成整段训练；410M 架构差异仍应通过短 smoke 状态确认。

### 15.3 Double calibration

double 不必每一步全量运行，但必须预先固定稀疏覆盖，例如：

- 代表性 checkpoint；
- 代表性训练阶段；
- 固定比例的 step 或固定窗口；
- LM pruning 的 step512、step2000、step5000 等计划节点。

每个校准点对称比较 signed U 与 signed double 的 high-positive、near-zero 和 negative，并报告参数/层/模块 Spearman、top-k overlap 与功能 gap。覆盖率不得根据结果临时增减。

### 15.4 Go/No-Go

- 若 signed U 稳定性与稀疏 double 校准通过：signed U 作为 Stage C 主方法，double 降为审计基线；
- 若只在层/模块通过：参数级强结论降级，只报告通过粒度；
- 若增加 $M$ 或窗口累计后仍失败：不得宣称 U 替代 double，应保留 double 为共同主方法或暂停正式扩展；
- positive-only 通过不能替代 signed 稳定性门禁。

## 16. 关键产物、哈希与复现状态

### 16.1 Stage A/B 总报告

数据根目录：`/home/sophgo13/cjl/storage/parameter-importance`

- 报告目录：`reports/stage-ab-minimum-loop`；
- `summary.json` SHA-256：`53d4aa80688245ab0975949fee197a96923716a973c5cf72f8f4d7f0f5547f70`；
- `report.md` SHA-256：`1504865798a34816908626c6d9e7ed6adc2d2aa53e35eb3805ed68a49ea5cadd`；
- `tables/stage_15_8_gates.csv` SHA-256：`935994d600bdde2925983b0b5345a049a4932c5a3aa66bfe2e057cf67135fcb4`；
- 正式 pruning：`runs/stage-b-sst2-{direct,pretrained}-pruning-seed{1234,1337,2027}`；
- 正式 reuse：`runs/stage-b-sst2-reuse-intervention-seed{1234,1337,2027}`。

### 16.2 Stage B.5

- 结果目录：`results/stage-b-u-double-near-zero-diagnostic-v1`；
- 精确机器行数：results 108、gaps 1296、aggregate 432、pairwise 324、score-signs 18；
- provenance ID：`fb3e818056fa36e82ec37ddba306e9d924d2676d054e28e37fe444bbe3b35bfa`；
- `_SUCCESS` SHA-256：`cdc3eff0cdeaee155bdb18f249a6b817ebb583d6dfdde536dd0efe299e036ee6`；
- 六份原 pruning 再次逐一验证，每份精确 594 行且哈希不变；
- 无 NaN/Inf，无正式 `FAILURE.json`。

B.5 实验代码提交为 `78329802193d89e121a2e275d8d3ce312bbe1255`。服务器全套回归绑定该提交：156 tests、0 failures、0 errors、0 skips；显式分布式与 checkpoint 恢复测试 2 passed；reproducibility 两项均为 true。

### 16.3 运行与下载状态

B.5 完成时 supervisor、worker、训练租约和 GPU 进程均已退出。旧 `CjlNlpTrainingChain-stage-b-downstream` 不存在。Pile 全量下载断点保留，`CjlPileFull` 与 `CjlPileFullSupervisor` 已恢复启用；公共 DNS/镜像稳定性属于基础设施风险，不改变上述科学结论。

## 17. 最终结论

截至 Stage B.5，最合理的总体判断不是“U-stat 失败”，也不是“U-stat 已经完全替代 double”，而是：

1. 原始统计动机成立：raw 有同源噪声偏差，Microbatch-level U-statistic 能在不增加独立完整采样路径的情况下估计同一个局部无偏目标。
2. fixed-state 参数精排与 signed 相邻 $M$ 层级稳定性仍是明确风险，不能被掩盖。
3. positive-only 的确更稳定、也有功能预测能力，但它通过截断改变了目标，不能用于回答“无偏 U 是否替代无偏 double”。
4. 恢复成 signed U 对 signed double、high-positive 对 near-zero 的对称比较后，当前六个 SST-2 来源得到参数 Spearman 0.9626–0.9692、层/模块 1.0 和功能判据 72/72，重新强力支持原始替代设想。
5. 这份支持只覆盖 Pythia-160M、SST-2 和三 seed；原 Gate 5 正式失败仍保留，Stage C 不能按旧方案无条件启动。
6. 下一步应是预注册 signed 稳定性 pilot 与稀疏 double 校准。通过后，才能采用“signed U 主测量、double 少量审计”的方案兑现计算节省；若仍失败，则必须降低结论粒度或保留 double。

一句话概括：

> signed U 仍是正确且有希望的科学主线；positive-only 是有用但不同的操作性分数；Stage C 的真正前置问题是降低并验证 signed U 的方差，而不是继续用截断回避它。
