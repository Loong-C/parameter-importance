# NLP 模型训练过程中参数重要性实验计划书

## 1. 实验名称

**基于 Microbatch-level U-statistic 的语言模型训练损失贡献参数重要性分析：自监督预训练与直接监督训练的比较**

## 2. 实验目的

本实验研究语言模型的参数重要性如何在训练过程中形成，并比较两种训练范式：

1. **直接监督训练**：模型从随机初始化开始，直接使用带人工标签的下游任务数据训练。
2. **自监督预训练后监督微调**：模型先在大规模无人工标签文本上进行因果语言建模预训练，再在相同下游任务上监督微调。

实验不把参数重要性定义为最终参数绝对值、单次梯度大小或某个输入样本的事后归因，而是把它定义为参数在训练过程中对数据损失下降的累计贡献。

实验需要建立以下证据链：

- 验证同一 minibatch 同时决定更新方向并计算贡献时是否产生同源噪声偏差；
- 验证 Microbatch-level U-statistic 是否能够减小该偏差，并在正式训练中替代双采样作为主方法；
- 观察自监督预训练过程中参数重要性的形成和演化；
- 比较直接监督训练与预训练后微调形成的重要性分布；
- 通过高重要性、低重要性和随机剪枝验证重要性的功能意义；
- 将所提出指标与权重幅值、参数移动量、原始同批梯度平方等基线比较；
- 分析预训练的重要参数是否在下游微调中被复用。

本实验只研究**损失贡献**，不研究路径平滑贡献、EMA 路径方向贡献或二者组合。

## 3. 方法边界与解释规则

### 3.1 U-statistic 的适用边界

Microbatch-level U-statistic 对以下局部目标具有清晰的无偏性解释：

$$
C_{k,t}^{\mathrm{grad}}=\eta_t\mu_{k,t}^2,
\qquad
\mu_{k,t}=\mathbb E_z\left[\frac{\partial L(z;\Theta_t)}{\partial\theta_k}\right],
$$

其中 $\eta_t$ 是当前学习率，$\mu_{k,t}$ 是给定当前参数状态 $\Theta_t$ 时的总体数据梯度。

该目标是一次参数更新路径积分的一阶、局部、梯度空间近似。正式报告中必须称为：

> 经 Microbatch-level U-statistic 去偏的局部梯度空间损失贡献。

不能称为：

> AdamW 完整参数更新路径积分的严格无偏估计。

### 3.2 AdamW 的处理

模型训练使用 AdamW，以对齐 Pythia 的训练方式并保证语言模型训练质量。同时保存两类指标：

- **主指标：U-statistic gradient-space importance**。用于主要排序、分布分析和剪枝验证。
- **辅助指标：actual-update raw importance**。使用 AdamW 真实数据更新量与当前批次梯度计算，更接近实际参数位移，但不宣称无偏。

权重衰减是正则化更新，不属于数据损失贡献。计算 actual-update importance 时必须从总参数位移中剔除 decoupled weight decay 项。

### 3.3 训练范式比较的边界

直接监督训练与“预训练后微调”在数据规模、训练时长和目标函数上天然不同。这些差异属于现实训练范式的一部分，因此实验可以比较两条完整路线，但结论应表述为：

> 直接监督训练与自监督预训练后监督微调形成了不同的参数重要性结构。

不能过度表述为：

> 在完全控制所有其他变量后，仅“是否监督”这一因素造成了差异。

为提高可比性，必须做到：

- 模型结构、tokenizer、LM head 和参数命名完全一致；
- 两条路线从同一份随机初始化权重出发；
- 下游阶段使用相同数据顺序、batch、最大训练步数和评价频率；
- 同时报告固定训练步数、最佳验证性能和可实现时的匹配性能位置；
- 只有当两个模型均达到任务有效学习标准时，才对分布差异作强解释。

### 3.4 不预设结果方向

预期直接监督训练在小数据任务上可能形成更集中的重要性分布，预训练可能形成更广泛、更稳定、更可迁移的结构，但这只是待检验假设。若结果方向相反，必须如实报告，不能通过更换归一化方法或筛选 checkpoint 追求预期结论。

## 4. 研究问题与假设

### RQ1：估计器是否有效

同一个 minibatch 同时决定更新方向并计算梯度贡献时，原始估计是否存在正偏差？Microbatch-level U-statistic 是否比原始同批估计更接近大样本参考值，并且方差不高于双采样？

判断依据：偏差、均方误差、采样方差、Spearman 相关、top-k 重合、剪枝曲线区分能力。

### RQ2：预训练中的重要性如何演化

观察参数重要性是否由初始随机状态逐渐分化、层间贡献是否迁移、attention 与 MLP 比例是否变化、排序是否逐渐稳定。

### RQ3：训练范式是否形成不同分布

比较两条路线的 Gini、有效参数数量、top-k 累计贡献、层级覆盖、attention/MLP 比例、随机种子稳定性和剪枝敏感性。

### RQ4：预训练参数是否被微调复用

使用 top-k overlap、Jaccard、随机基线以上 enrichment、Spearman、分层相关和干预剪枝判断。

### RQ5：任务数据规模是否影响分化

分别研究 SST-2、MNLI 和 RTE，不在单个任务报告中混合。检验小数据直接监督训练是否更易出现高集中度、过拟合和随机种子不稳定。

## 5. 参数重要性定义

### 5.1 理论路径积分

设第 $t$ 步更新前后的参数为 $\Theta_t$ 和 $\Theta_{t+1}$，参数位移为：

$$
\Delta\Theta_t=\Theta_{t+1}-\Theta_t.
$$

参数 $\theta_k$ 的理想损失贡献为：

$$
c_{k,t}^{\mathrm{path}}
=-\Delta\theta_{k,t}\int_0^1 g_k(\Theta_t+\alpha\Delta\Theta_t)\,d\alpha,
$$

其中：

$$
g_k(\Theta)=\frac{\partial L(\Theta)}{\partial\theta_k}.
$$

### 5.2 正式训练采用的局部目标

把总 batch 分为 $M$ 个等大小 microbatch。第 $m$ 个 microbatch 对参数 $k$ 的平均梯度为 $g_{m,k,t}$。总体梯度均值为：

$$
\mu_{k,t}=\mathbb E[g_{m,k,t}].
$$

梯度空间的一步理想损失贡献定义为：

$$
C_{k,t}^{\mathrm{grad}}=\eta_t\mu_{k,t}^2.
$$

若发生全局梯度裁剪，裁剪系数为 $s_t\in(0,1]$，则：

$$
C_{k,t}^{\mathrm{grad,clip}}=\eta_t s_t\mu_{k,t}^2.
$$

主结果使用裁剪调整版本，同时保存未调整版本。

### 5.3 Microbatch-level U-statistic

原始同批估计为：

$$
\widehat C_{k,t}^{\mathrm{raw}}
=\eta_t\left(\frac1M\sum_{m=1}^M g_{m,k,t}\right)^2.
$$

U-statistic 为：

$$
\widehat C_{k,t}^{U}
=\eta_t\frac{\left(\sum_{m=1}^M g_{m,k,t}\right)^2-\sum_{m=1}^M g_{m,k,t}^2}{M(M-1)}.
$$

考虑梯度裁剪后：

$$
\widehat C_{k,t}^{U,\mathrm{clip}}=s_t\widehat C_{k,t}^{U}.
$$

### 5.4 Signed、positive-only 与 negative

$$
\Omega_k^{\mathrm{signed}}=\sum_{t=0}^{T-1}\widehat C_{k,t}^{U,\mathrm{clip}},
$$

$$
\Omega_k^+=\sum_{t=0}^{T-1}\max(0,\widehat C_{k,t}^{U,\mathrm{clip}}),
$$

$$
\Omega_k^-=\sum_{t=0}^{T-1}\min(0,\widehat C_{k,t}^{U,\mathrm{clip}}).
$$

Signed importance 的剪枝排序分为：最大正贡献、最接近零、最负贡献。不能把“最负”直接当成“最不重要”。Positive-only 使用最大值和最小值排序。

## 6. 数值积分校验

正式大模型训练采用局部一阶 U-statistic，但必须通过小模型实验说明一阶近似与多节点积分的关系。

### 6.1 模型与状态

- 模型：Pythia-style 14M 或 31M；
- 从训练中选择至少 20 个参数状态，覆盖早期、中期和后期。

### 6.2 校验流程

1. 使用独立 update batch 产生一次参数更新并固定路径：
   $$\Theta(\alpha)=\Theta_t+\alpha\Delta\Theta_t.$$
2. 使用与 update batch 独立的 probe batch 计算路径梯度。
3. 计算左端点、梯形、Simpson 和 16 节点 Gauss-Legendre（或足够密集均匀节点）参考积分。
4. 比较参数级相对误差、层级相对误差、Spearman 和 top-k overlap。
5. 若参数级噪声大但层级和 top-k 排序稳定，正式实验仍可使用一阶近似；若排序相关过低，正式结论降为模块级，或增加低频多节点积分。

建议工程门槛：层级 Spearman 不低于 0.90，参数级不低于 0.70，top-5% overlap 不低于 0.60。该门槛是执行标准，不是普适定理。

## 7. 估计器偏差验证

### 7.1 目的

在固定参数状态下直接比较 raw、double-sampling 和 Microbatch U-statistic。

### 7.2 参考目标

- 模型：14M 或 31M；
- checkpoint：至少 10 个；
- 固定参考集：例如 4096 条等长序列；
- 分批计算参考总体梯度 $\mu_k^{\mathrm{ref}}$；
- 参考目标：
  $$C_k^{\mathrm{ref}}=\eta(\mu_k^{\mathrm{ref}})^2.$$

### 7.3 重复采样

每个 checkpoint 至少重复 200 次，保持三种估计器总样本预算一致，并测试 $M=4,8,16,32$。

记录 Bias、Relative Bias、MSE、方差、Spearman、top-k overlap、计算时间和显存。

### 7.4 进入正式实验的门槛

- U-statistic 的层级和模块级偏差明显小于 raw；
- 与参考排序的相关不低于 double sampling；
- MSE 不高于 double sampling，或在相近 MSE 下成本更低；
- $M\ge8$ 时结果趋于稳定；
- 无 NaN、Inf 或少量参数支配全部估计的数值问题。

## 8. 软件环境与项目结构

### 8.1 推荐软件栈

- Linux；
- Python；
- PyTorch；
- Hugging Face Transformers；
- Hugging Face Datasets；
- NumPy、Pandas、SciPy；
- Matplotlib；
- safetensors；
- TensorBoard 或 Weights & Biases；
- Git；
- CUDA/NCCL。

正式执行前保存：

```bash
python --version
nvidia-smi
pip freeze > environment/requirements-lock.txt
git rev-parse HEAD > environment/git_commit.txt
```

训练日志必须记录 CUDA、cuDNN、NCCL、GPU 型号和代码 commit。

### 8.2 推荐目录

```text
parameter-importance-nlp/
├── configs/
│   ├── model/
│   ├── pretrain/
│   └── tasks/
├── data/
│   ├── pythia_deduped/
│   ├── glue/
│   └── eval/
├── src/
│   ├── models/
│   ├── data/
│   ├── importance/
│   │   ├── u_statistic.py
│   │   ├── raw_estimators.py
│   │   ├── actual_update.py
│   │   └── aggregation.py
│   ├── train/
│   │   ├── pretrain.py
│   │   └── supervised.py
│   ├── eval/
│   │   ├── language_model.py
│   │   ├── classification.py
│   │   ├── pruning.py
│   │   └── overlap.py
│   └── visualization/
├── Agents/
├── worklogs/
├── checkpoints/
├── importance/
├── results/
├── figures/
└── reports/
```

## 9. 模型配置

### 9.1 最小闭环：Pythia-style 160M

- 层数：12；
- hidden size：768；
- attention heads：12；
- head dimension：64；
- context length：2048；
- rotary position embedding，rotary percentage 0.25；
- GPT-J parallel residual；
- embedding 与 LM head 不共享权重；
- dropout：0；
- causal LM head；
- AdamW；
- peak learning rate：$6\times10^{-4}$；
- betas：$(0.9,0.95)$；
- epsilon：$10^{-8}$；
- weight decay：0.1；
- gradient clipping：1.0；
- cosine decay；
- warmup：总步数 1%。

### 9.2 正式模型：Pythia-style 410M

- 层数：24；
- hidden size：1024；
- attention heads：16；
- head dimension：64；
- context length：2048；
- rotary position embedding，rotary percentage 0.25；
- GPT-J parallel residual；
- embedding 与 LM head 不共享权重；
- dropout：0；
- causal LM head；
- AdamW；
- peak learning rate：$3\times10^{-4}$；
- betas：$(0.9,0.95)$；
- epsilon：$10^{-8}$；
- weight decay：0.1；
- gradient clipping：1.0；
- cosine decay；
- warmup：总步数 1%。

### 9.3 数值精度

推荐 BF16 forward/backward，所有重要性累加器使用 FP32。选择 BF16 是为了避免动态 loss scaling 对 microbatch gradient 的干扰。若必须使用 FP16，必须在读取梯度用于 U-statistic 前完成正确 unscale，并写单元测试确认尺度。

### 9.4 参数范围

训练时为所有可训练参数计算重要性。第一组主剪枝纳入 attention projection、MLP 和 LM head 权重；默认排除 LayerNorm、bias 和 token embedding。排除项仍需单独报告重要性。后续再做全参数剪枝稳健性分析。

## 10. 数据与 Pythia 对齐

### 10.1 预训练数据

使用 EleutherAI 发布的 **Pythia deduplicated Pile 预分词、预打乱数据流**，并与 `pythia-160m-deduped`、`pythia-410m-deduped` 对齐。

使用前必须记录：

- 数据仓库和版本；
- shard 清单与校验和；
- tokenizer 文件哈希；
- 数据起始位置和结束位置；
- data cursor；
- 是否发生样本跳过或重复。

### 10.2 存储与合规

至少预留约 600 GB 可用磁盘空间，并确认学校或实验室允许使用该数据。若无法合法或稳定获得 Pythia 数据，则改用开放许可语料，并取消“精确 Pythia 数据对齐”的表述。

### 10.3 Token 预算

最小闭环训练 160M 到 step512：

$$
512\times2{,}097{,}152=1{,}073{,}741{,}824
$$

约 1.074B tokens，可与公开 `pythia-160m-deduped step512` 对照。

正式训练 410M 到 step5000：

$$
5000\times2{,}097{,}152=10{,}485{,}760{,}000
$$

约 10.486B tokens，可与公开 `pythia-410m-deduped step5000` 对照。

### 10.4 Global batch

目标 global batch 为 2,097,152 tokens。context length 为 2048，因此每 step 为 1024 条定长序列。

4 张 GPU 推荐从以下配置试跑：

```text
sequences_per_gpu_per_microbatch = 8
gradient_accumulation_steps = 32
global sequences = 4 × 8 × 32 = 1024
M = 4 × 32 = 128
```

若 batch 8 无法容纳：

```text
sequences_per_gpu_per_microbatch = 4
gradient_accumulation_steps = 64
M = 256
```

每个统计 microbatch 必须等大小，启用 `drop_last=True`。

### 10.5 验证数据

使用与训练 token 不重合的固定验证集，例如独立 Pile validation 或 Wikitext-103 test。自己训练的模型与公开 Pythia 匹配 checkpoint 必须使用相同 tokenizer、context、stride、batch 和 loss 计算脚本。

## 11. Microbatch 梯度采集实现

### 11.1 核心流程

1. 每张 GPU 独立处理本地 microbatch；
2. 每个 microbatch 前清空临时梯度；
3. backward 得到该 microbatch 的 local mean gradient；
4. 本地流式累加：
   $$S_{1,k}^{local}\mathrel{+}=g_{m,k},$$
   $$S_{2,k}^{local}\mathrel{+}=g_{m,k}^2;$$
5. 一个 optimizer step 的 microbatch 全部结束后，对 $S_1$ 和 $S_2$ 做全局 all-reduce sum；
6. 由全局 $S_1/M$ 设置 optimizer 的平均梯度；
7. 计算 U-statistic；
8. 计算并应用全局梯度裁剪；
9. 更新累计 importance；
10. 执行 AdamW step；
11. 记录 actual-update 诊断。

不能在每个 microbatch 上直接执行普通 DDP all-reduce，否则无法保留独立 local gradient，并会增加通信成本。可使用 `no_sync` 或自定义梯度聚合。

### 11.2 尺度单元测试

必须确认：

- microbatch loss 是 mean 还是 sum；
- $g_m$ 是 microbatch 均值梯度；
- all-reduce 是 sum 还是 mean；
- optimizer 最终梯度等于所有样本平均梯度；
- U-statistic 使用相同尺度；
- padding token 不计入预训练 loss；
- 下游任务只对 label token 计算 loss。

用小型两层网络手算结果，FP32 误差应低于 $10^{-6}$ 或预先设定的 BF16 容差。

### 11.3 伪代码

```python
S1 = zeros_like_parameters(fp32=True)
S2 = zeros_like_parameters(fp32=True)
M_local = 0

for microbatch in local_microbatches:
    zero_temporary_grads()
    loss = model_loss(microbatch)
    loss.backward()
    g = read_parameter_grads_as_fp32()
    S1 += g
    S2 += g.square()
    M_local += 1

global_S1 = all_reduce_sum(S1)
global_S2 = all_reduce_sum(S2)
M = all_reduce_sum(M_local)

mean_grad = global_S1 / M
set_optimizer_grads(mean_grad)
clip_factor = compute_and_apply_global_grad_clip()

step_score = learning_rate * clip_factor * (
    global_S1.square() - global_S2
) / (M * (M - 1))

omega_signed += step_score
omega_positive += clamp_min(step_score, 0)
omega_negative += clamp_max(step_score, 0)

record_parameter_state_before_step()
optimizer.step()
record_actual_update_diagnostic()
```

### 11.4 内存与保存

不保存全部 microbatch gradients。每个参数长期保留 $S_1$、$S_2$、$\Omega^{signed}$、$\Omega^+$ 和可选 $\Omega^-$。累加器使用 FP32。关键 checkpoint 才保存参数级数组，其他时间只保存层级、模块级和 top-k 摘要。写盘前转移到 CPU，优先使用 safetensors 分片。

## 12. 对照评分方法

剪枝验证至少比较：

1. U-statistic positive-only；
2. U-statistic signed；
3. raw same-batch cumulative score；
4. actual-update raw importance；
5. 最终权重幅值 $|\theta_{k,T}|$；
6. 累计参数移动量 $\sum_t|\Delta\theta_{k,t}|$；
7. 验证集 Fisher/梯度平方 $\mathbb E[g_k^2]$；
8. 随机评分。

资源不足时最低保留 U-stat positive、U-stat signed、raw、weight magnitude、movement 和 random。

## 13. 归一化与统计口径

### 13.1 Positive-only

$$
p_k=\frac{\Omega_k^+}{\sum_j\Omega_j^++\epsilon}.
$$

计算 Gini、Shannon entropy、HHI、有效参数数量：

$$
HHI=\sum_kp_k^2,
\qquad
N_{\mathrm{eff}}=\frac1{\sum_kp_k^2},
$$

以及 top-1%、5%、10%、20% 累计贡献。

### 13.2 Signed

Signed 不能直接用于 Gini。分别报告正贡献总质量、负贡献总质量、负贡献参数比例、$|\Omega^{signed}|$ 集中度、正贡献部分集中度，以及每层 mean、median 和分位数。

### 13.3 层与模块

每层和每模块同时报告总和、每参数均值、占全模型比例、参数数量、中位数、分位数、Gini 和有效参数比例。不能只比较总和，因为模块参数数量不同。

## 14. 阶段 A：代码与方法单元测试

### 14.1 目的

在正式训练前确认模型、梯度聚合、U-statistic、checkpoint 恢复和剪枝均正确。

### 14.2 模型

使用 Pythia-style 14M 或 31M。

### 14.3 必做测试

1. 单 GPU 与 4 GPU 对同一 global batch 的最终梯度一致；
2. 不同 microbatch 切分在大样本下得到近似相同 U-statistic；
3. $M=1$ 时明确报错；
4. microbatch 不等大时拒绝运行，除非实现并验证加权公式；
5. 中断恢复后的累计 importance 与不中断训练一致；
6. 剪枝 mask 确实生效；
7. random pruning 在固定 seed 下可复现；
8. weight decay 不进入数据损失贡献；
9. gradient clipping factor 正确记录；
10. data loss 与 optimizer regularization 分开记录；
11. tokenizer shift、causal mask、padding mask 正确；
12. 下游 label loss 只作用于最终标签 token。

### 14.4 产出

```text
reports/unit_test_report.md
results/estimator_bias/
results/numerical_integration/
```

全部核心测试通过后才能开始最小闭环。

## 15. 阶段 B：最小闭环实验

### 15.1 目的

使用 Pythia-style 160M 和 SST-2 跑通预训练、直接监督、微调、重要性计算、剪枝和可视化完整流程，并判断 U-statistic 是否可替代双采样成为正式主估计器。

### 15.2 160M 自监督预训练

- 数据：Pythia deduplicated Pile 数据流前 512 step；
- token 数：约 1.074B；
- checkpoint：0、1、2、4、8、16、32、64、128、256、512；
- 每个 checkpoint 保存 model、optimizer、scheduler、RNG、data cursor、signed、positive、raw、actual-update、layer/module aggregates 和配置。

### 15.3 预训练评价

在固定外部验证集计算 loss 和 perplexity，并与公开 `pythia-160m-deduped step512` 使用同一脚本比较。

训练质量门槛：

- 无 NaN、Inf 或不可恢复 loss spike；
- validation loss 总体下降；
- 自己模型与公开匹配 checkpoint 的 perplexity ratio 建议不超过 1.15；
- tokenizer、数据 shift、global batch、学习率和初始化已核验；
- 生成文本只做健康检查，不作为主要指标。

### 15.4 SST-2 直接监督路线

1. 加载与预训练完全相同的 `step0` 初始化；
2. 使用统一 prompt/verbalizer；
3. 从随机初始化直接训练 SST-2；
4. 每步计算 U-statistic；
5. 保存 signed、positive、raw、actual-update 和基线评分；
6. 保存固定步数和最佳 validation checkpoint；
7. 完成剪枝。

### 15.5 SST-2 预训练后微调路线

1. 加载 160M `step512`；
2. 使用与直接监督相同的数据顺序、prompt、batch、最大步数和评价频率；
3. 微调 importance 独立累计，不覆盖预训练 importance；
4. 保存固定步数和最佳 validation checkpoint；
5. 完成剪枝。

### 15.6 Estimator 对照

最小闭环必须比较 raw、U-statistic 和 double sampling。双采样可在 14M/31M 固定状态实验、160M 最后 50～100 个预训练 step 和 SST-2 完整训练上执行，不要求覆盖全部 1.074B tokens。

### 15.7 最小闭环剪枝

分别测试：

- U-stat positive high/low；
- U-stat signed high-positive/near-zero/negative；
- raw high/low；
- magnitude high/low；
- movement high/low；
- random。

剪枝比例：0.1%、0.5%、1%、2%、5%、10%、20%。每个随机比例至少 20 个 mask。

### 15.8 通过标准

- 预训练质量门槛通过；
- SST-2 两条路线均高于多数类基线；
- 高 U-stat positive 参数剪枝损害大于低重要性剪枝；
- U-stat 区分度不低于 raw；
- U-stat 与 double sampling 排序相关和剪枝效果相近或更好；
- signed 与 positive-only 至少一套产生稳定功能排序；
- 中断恢复和多 GPU 结果可复现。

若不通过，不得扩大到 410M。

## 16. 阶段 C：正式 410M 自监督预训练

### 16.1 目的

只研究 410M 模型在 10.486B token 因果语言建模过程中的重要性形成，不混入下游任务结论。

### 16.2 配置

- 模型：Pythia-style 410M；
- 数据：Pythia deduplicated Pile；
- tokenizer：Pythia/GPT-NeoX tokenizer；
- context：2048；
- global batch：2,097,152 tokens；
- steps：5000；
- precision：BF16；
- $M$：推荐 128 或 256；
- 主 seed：1234。

checkpoint：

```text
0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
1000, 2000, 3000, 4000, 5000
```

### 16.3 公开 Pythia 对照

加载 `EleutherAI/pythia-410m-deduped` 的 `step5000`，在相同外部验证集比较：

- validation loss；
- perplexity；
- 可选 LM Evaluation Harness 小任务；
- 参数 norm 和 activation norm，作为训练健康检查。

不要求逐参数一致，但训练曲线和匹配 token 数性能应合理。

### 16.4 训练日志

每 10 step 记录：training loss、validation loss（按既定间隔）、learning rate、global gradient norm、clip factor、update norm、raw score sum、U-stat score sum、signed positive/negative mass、importance norm、tokens、throughput 和显存。

每个 checkpoint 保存参数级 signed、positive、raw、movement、layer/module aggregate、rank summary 和 top-k 参数 ID。

### 16.5 预训练分析

必须生成：

1. loss/perplexity 曲线；
2. layer × checkpoint heatmap；
3. module × checkpoint heatmap；
4. attention/MLP ratio；
5. top-k cumulative curve；
6. Gini、entropy、HHI、$N_{eff}$ 随训练变化；
7. 相邻 checkpoint Spearman；
8. top-1%、5%、10% 集合稳定性；
9. signed 正负质量变化；
10. U-stat 与 raw、movement、magnitude 的相关。

### 16.6 语言模型剪枝

在 step512、step2000、step5000 进行语言模型剪枝，评价外部 validation loss、perplexity 和可选公开任务。检验早期重要性是否已有功能意义，以及后期排序是否更稳定。

## 17. 下游任务通用设置

### 17.1 架构统一

所有下游任务与预训练共用同一个 decoder-only backbone、tokenizer 和 LM head，不增加 classification head。监督与自监督的差别只来自数据和训练目标。

### 17.2 Prompt/verbalizer

输出使用单个候选标签 token。执行前运行 tokenizer 检查脚本，确认每个候选标签均为单 token、预测位置正确、标签 token 不与特殊 token 冲突。

推荐使用 A/B/C，并在 prompt 中定义选项。若 A/B/C 在目标 tokenizer 下不是单 token，必须更换；不得直接使用长度不同的多 token 标签而不做长度校正。

### 17.3 Loss 与评价

- 监督 loss 只在最终 label token 位置计算；
- 评价时只比较候选标签 token 的 logits；
- 不通过自由生成决定分类；
- 报告 accuracy、negative log-likelihood 和校准指标；
- prompt 前缀不进入监督 loss。

### 17.4 两类下游运行

每个任务执行：

1. **受控机制运行**：两条路线使用相同数据顺序、batch、最大步数和评价点，importance 结论主要来自该运行。
2. **最佳性能运行**：两条路线分别做小规模学习率搜索，以 validation 选择最佳模型，用于说明训练结果足够好。

受控运行和最佳性能运行必须使用不同实验编号，不能混用 checkpoint。

### 17.5 比较位置

至少比较：训练进度 25%、50%、100%、各自最佳 validation checkpoint，以及两个模型均能达到时的 matched-performance checkpoint。

### 17.6 有效学习门槛

若直接监督模型没有稳定超过多数类或随机基线：

- 可以报告“直接监督不足以学会任务”；
- 可以把重要性作为失败机制描述；
- 不能与成功的预训练模型作强功能结构等价比较。

## 18. 阶段 D：SST-2 实验

### 18.1 目的

研究简单情感分类任务中，直接监督是否形成任务特化集中结构，预训练后微调是否复用预训练参数，以及两条路线的剪枝敏感性是否不同。

### 18.2 数据

使用 Hugging Face GLUE `sst2`：train 用于训练，validation 用于选择模型和报告。test 不参与开发。

### 18.3 Prompt

```text
Review:
{sentence}

Choose the sentiment:
A = negative
B = positive
Answer:
```

候选 token 为 A、B，必须通过 tokenizer 检查。

### 18.4 路线 S-SUP

1. 加载 410M `step0`；
2. 固定数据顺序；
3. 从随机初始化直接训练；
4. 在线累计 importance；
5. 保存 25%、50%、100% 和 best；
6. 评价 accuracy、NLL、ECE 和 generalization gap；
7. 完成全局与层平衡剪枝。

### 18.5 路线 S-PT

1. 加载 410M `step5000`；
2. 使用与 S-SUP 相同数据顺序和 prompt；
3. 微调并独立累计 importance；
4. 保存相同位置 checkpoint；
5. 评价相同指标；
6. 完成剪枝。

### 18.6 分析指标

- 性能：train/validation accuracy、NLL、ECE、generalization gap；
- 分布：Gini、$N_{eff}$、top-k、layer share、attention/MLP、seed stability；
- 复用：top-k overlap、Jaccard、enrichment、Spearman、layer correlation；
- 剪枝：high/low/random curve、AUC 和 gap。

### 18.7 产出

```text
reports/sst2_report.md
results/sst2/metrics.csv
results/sst2/importance/
figures/sst2/
```

## 19. 阶段 E：MNLI 实验

### 19.1 目的

研究大规模自然语言推理监督数据能否让随机初始化模型形成较广泛的重要性结构，预训练优势是否仍存在，以及 matched/mismatched 域上的结构和剪枝差异。

### 19.2 数据

使用 GLUE `mnli`：train、validation_matched、validation_mismatched。两套 validation 均必须报告。

### 19.3 Prompt

```text
Premise:
{premise}

Hypothesis:
{hypothesis}

Choose the relation:
A = entailment
B = neutral
C = contradiction
Answer:
```

候选 token A、B、C 必须为单 token。

### 19.4 路线 M-SUP

1. 加载 410M `step0`；
2. 固定数据顺序；
3. 直接监督训练；
4. 在线累计 importance；
5. 保存固定进度和 best；
6. 在 matched/mismatched 上评价；
7. 完成剪枝。

### 19.5 路线 M-PT

1. 加载 410M `step5000`；
2. 使用与 M-SUP 相同数据顺序；
3. 微调并累计 importance；
4. 保存相同位置；
5. 在 matched/mismatched 上评价；
6. 完成剪枝。

### 19.6 额外指标

- matched–mismatched accuracy gap；
- matched–mismatched pruning degradation gap；
- 三类别 confusion matrix；
- attention 与 MLP 的层级结构；
- 可选按类别验证集 Fisher，用于理解类别依赖，但不混入主训练 importance。

### 19.7 产出

```text
reports/mnli_report.md
results/mnli/metrics.csv
results/mnli/importance/
figures/mnli/
```

## 20. 阶段 F：RTE 实验

### 20.1 目的

研究小数据推理任务中直接监督是否严重过拟合、预训练是否成为任务成功的重要条件，以及 RTE 是否更强地复用预训练参数。

### 20.2 数据

使用 GLUE `rte` 的 train 和 validation。由于数据量小，必须多随机种子，并避免根据 validation 反复手工调整结论。

### 20.3 Prompt

```text
Premise:
{sentence1}

Hypothesis:
{sentence2}

Choose the relation:
A = entailment
B = not entailment
Answer:
```

候选 token A、B 必须为单 token。

### 20.4 路线 R-SUP

1. 加载 410M `step0`；
2. 使用预注册数据顺序和步数；
3. 直接训练；
4. 累计 importance；
5. 保存固定进度和 best；
6. 报告训练/验证差距；
7. 只有超过有效学习门槛时才作强剪枝解释。

### 20.5 路线 R-PT

1. 加载 410M `step5000`；
2. 使用与 R-SUP 相同数据顺序；
3. 微调并累计 importance；
4. 保存相同位置；
5. 评价并剪枝。

### 20.6 随机种子

RTE 至少 5 个下游随机种子。每个 seed 固定数据顺序、训练设置、评价和 random pruning masks。报告均值、标准差和 95% bootstrap CI。

### 20.7 产出

```text
reports/rte_report.md
results/rte/metrics.csv
results/rte/importance/
figures/rte/
```

## 21. 剪枝验证方案

### 21.1 粒度

第一组使用非结构化标量权重剪枝：将选定权重置零并保持为零，不做恢复训练。恢复训练会混入模型重新适应能力，不能用于验证当前功能依赖。

### 21.2 两种控制

**全局剪枝**：在所有 eligible parameters 中全局排序，测试指标能否识别全模型关键参数。

**层平衡剪枝**：每层剪相同比例，再在层内排序，排除层大小、梯度尺度和参数数量支配全局排序的混杂。

两种结果必须同时报告。

### 21.3 比例

```text
0%, 0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%, 30%
```

若 0.1% 已崩溃，增加更小比例；若 30% 仍无变化，可扩展到 40%、50%。

### 21.4 随机基线

每比例至少 20 个随机 mask，RTE 建议 50 个。随机 mask 必须使用相同 eligible set、相同参数数量和相同全局/层平衡方式。

### 21.5 评价

语言模型：loss increase、perplexity ratio。

下游任务：accuracy drop、NLL increase、ECE change。

可定义：

$$
Gap(r)=Acc_{low}(r)-Acc_{high}(r).
$$

同时计算 pruning curve AUC，并在相同 seed 间配对比较。

### 21.6 功能有效性判定

重要性至少应满足：

1. 高重要性剪枝比随机损害大；
2. 低重要性剪枝比高重要性损害小；
3. 全局和层平衡方向一致；
4. 至少在两个评价场景成立；
5. 不完全由 LM head 或最后一层支配；
6. 优于至少两个简单基线。

## 22. 预训练—微调复用分析

### 22.1 分数

预训练 importance 与微调 importance 分开保存，不直接相加。比较前使用 rank 或非负归一化。

### 22.2 Top-k overlap

设预训练 top-$k\%$ 集合为 $A_k$，微调集合为 $B_k$：

$$
Overlap(k)=\frac{|A_k\cap B_k|}{|A_k|}.
$$

随机独立集合期望 overlap 为 $k\%$。定义：

$$
Enrichment(k)=\frac{Overlap(k)}{k/100},
$$

$$
J(k)=\frac{|A_k\cap B_k|}{|A_k\cup B_k|}.
$$

使用 $k=0.1\%,0.5\%,1\%,5\%,10\%,20\%$。

### 22.3 排名相关

计算全参数、每层、每模块和 top-tail Spearman；positive-only 与 signed 分开计算。

### 22.4 功能复用干预

1. 按预训练 importance 剪高重要性参数；
2. 在未微调和已微调模型上分别评价；
3. 按微调 importance 剪枝；
4. 剪两者 top-k 交集；
5. 剪只属于预训练 top-k、不属于微调 top-k 的参数。

该干预用于区分统计重合和真实功能复用。

## 23. 随机种子与统计计划

### 23.1 最小闭环

- 固定状态 estimator：每 checkpoint 至少 200 次采样；
- 160M 下游：至少 3 seeds；
- random pruning：每比例至少 20 masks。

### 23.2 正式预训练

- 410M 完整 5000-step：至少 1 个主 seed；
- 额外 seed 可运行到 step512 或 step1000 验证早期趋势；
- 多 seed 的主要稳定性证据由 160M 补充。

单个 410M seed 的细微差异不得表述为普适规律。

### 23.3 正式下游

- SST-2：至少 3 seeds；
- MNLI：至少 3 seeds；
- RTE：至少 5 seeds。

### 23.4 报告

报告 mean、standard deviation、95% bootstrap CI、paired effect size 和每个 seed 散点。seed 很少时不依赖单一 p-value。

## 24. 可视化规范

### 24.1 训练健康图

每个运行：training loss、validation loss、learning rate、gradient norm、clip factor、update norm、throughput 和 GPU memory。

### 24.2 重要性演化热力图

横轴 checkpoint，纵轴层，颜色为归一化 importance share。signed 与 positive-only 分开。

### 24.3 模块热力图

每层拆分 q/k/v/o、attention total、MLP input/up、MLP output/down、LayerNorm、embedding、LM head。同时画总和和每参数均值。

### 24.4 分布图

log histogram、violin、CCDF、Lorenz curve、top-k cumulative curve。

### 24.5 任务图

SST-2、MNLI、RTE 各自独立画图。每个任务至少包括两条路线的 Gini、$N_{eff}$、layer share、attention/MLP、pruning curve 和 overlap curve。

### 24.6 剪枝图

每任务独立画 U-stat positive high/low、U-stat signed high-positive/near-zero、raw、magnitude、movement 和 random mean ± CI。所有曲线的 eligible set 和坐标必须一致。

## 25. 训练结果是否足够好

### 25.1 预训练

同时满足：

1. loss 曲线正常；
2. external validation perplexity 持续改善；
3. 与公开 Pythia 匹配 token checkpoint 差距合理；
4. 下游微调相对随机初始化有合理收益；
5. 无长期梯度爆炸、频繁极端裁剪或 loss spike。

### 25.2 下游

每个任务报告多数类、随机、公开相近规模模型或公开 Pythia checkpoint 的同 prompt 结果、direct-supervised 和 pretrained-finetuned。只有模型真正学会任务后，才把 importance 解释为成功模型的功能结构。

### 25.3 失败运行

失败运行不得静默删除。记录 seed、loss、梯度、失败原因、是否重跑和修改内容。修改超出预注册范围时使用新实验编号，不能覆盖原结果。

## 26. 逐步执行顺序

### 步骤 1：建立仓库并冻结环境

创建 Git 仓库、目录、依赖锁定、统一配置和日志。

产出：

```text
environment/requirements-lock.txt
environment/git_commit.txt
```

### 步骤 2：下载 tokenizer、配置和数据

下载 Pythia tokenizer、160M/410M 配置、deduplicated Pile 预打乱数据、GLUE 三任务；校验文件；验证前若干 batch 与官方 batch viewer 一致。

产出：

```text
reports/data_manifest.md
data/checksums.txt
```

### 步骤 3：搭建模型

实现或加载 GPTNeoXForCausalLM，对齐层数、维度、RoPE、残差和不共享权重；验证参数量与命名；保存 step0；确保监督和预训练共用同一模型类。

产出：

```text
reports/model_manifest.md
checkpoints/step0/
```

### 步骤 4：实现 U-statistic

实现 per-microbatch gradient、$S_1/S_2$、跨 GPU 聚合、signed/positive/negative、恢复、raw、actual-update 和单元测试。

### 步骤 5：固定状态偏差验证

运行 raw、double-sampling 和 U-statistic。未通过门槛不得继续。

### 步骤 6：数值积分校验

比较一阶、梯形、Simpson 和参考积分，形成独立报告。

### 步骤 7：160M 最小闭环预训练

训练到 step512，评价公开匹配 checkpoint，保存 importance。

### 步骤 8：160M SST-2 两条路线

运行直接监督、预训练微调、estimator 对照和剪枝。若不能区分高低重要性，不得扩大模型。

### 步骤 9：410M 正式预训练

训练到 step5000，保存指定 checkpoint 和统计。

### 步骤 10：410M 预训练分析与 LM 剪枝

确认重要性在语言模型本身上具有功能意义。

### 步骤 11：SST-2 正式实验

独立完成 SST-2 全流程和报告。

### 步骤 12：MNLI 正式实验

独立完成 MNLI 全流程和报告。

### 步骤 13：RTE 正式实验

独立完成 RTE 全流程和报告。

### 步骤 14：跨任务总结

只有三个任务报告均完成后，才比较任务数据规模、集中度、复用、剪枝敏感性和 seed 稳定性。

### 步骤 15：归档

保存代码 commit、配置、数据 manifest、checkpoint、importance、原始 metrics、作图脚本和报告。

## 27. 最终产出清单

### 27.1 方法验证

- estimator bias/MSE 表；
- U-statistic 与 double sampling 方差比较；
- 不同 $M$ 稳定性图；
- 数值积分误差表；
- 时间与显存表。

### 27.2 最小闭环

- 160M step512；
- 公开 Pythia 对照；
- SST-2 两条路线；
- 完整剪枝图；
- U-statistic 是否替代双采样的结论。

### 27.3 正式预训练

- 410M step5000；
- 关键 checkpoint；
- 参数级 signed/positive；
- 层级和模块演化图；
- LM 剪枝；
- 公开 Pythia step5000 对照。

### 27.4 下游任务

SST-2、MNLI、RTE 各自提供 direct-supervised、pretrained-finetuned、性能、分布、复用、剪枝和独立报告。

### 27.5 总结必须回答

1. raw estimator 是否有可观测过估计？
2. U-statistic 是否优于或不弱于双采样？
3. signed 与 positive-only 哪个更能预测剪枝损伤？
4. 预训练 importance 如何形成？
5. 两种训练范式的集中度是否不同？
6. 预训练重要参数是否被微调复用？
7. 三个任务差异是否与数据规模和复杂度一致？
8. U-statistic 是否优于幅值、移动量和 raw 基线？
9. 结论在哪些规模、任务和 seed 上成立？
10. 哪些结论只适用于局部梯度空间贡献，不能推广到完整 AdamW 路径积分？

## 28. 结论判定规则

### 28.1 方法有效

支持方法有效需满足：U-statistic 对参考偏差小；高重要性剪枝显著损害模型；低重要性较小；优于随机和至少两个简单基线；层平衡剪枝仍成立；多 seed 方向一致。

### 28.2 训练范式差异

当两个模型性能有效且比较位置合理，若集中度、有效参数数、层覆盖、剪枝和复用在多 seed 中稳定不同，可认为两种训练范式形成不同的重要性结构。差异方向由数据决定。

### 28.3 不能作强结论

以下情况只描述现象：直接监督未学会任务；只有一个 seed；没有层平衡剪枝；不优于简单基线；模型结构或 prompt 不一致；U-stat 未通过偏差验证；评价泄漏；只比较未归一化原始数值。

## 29. 风险与应对

| 风险 | 表现 | 应对 |
|---|---|---|
| U-statistic 与 AdamW 不完全一致 | importance 与真实更新差异大 | 主指标称 gradient-space；保存 actual-update；固定状态验证 |
| 410M 成本过高 | throughput 过低 | 先完成 160M；减少非关键参数级快照，不删除关键 token 对齐点 |
| Pile 不可用或不合规 | 无法下载/审批未通过 | 使用开放许可语料，取消精确对齐表述 |
| 直接监督 RTE 失败 | 随机水平、过拟合 | 作为失败机制报告，不作成功模型等价比较 |
| verbalizer 错误 | 标签多 token | 使用验证脚本和 A/B/C 单 token |
| importance 被最后层支配 | 全局只剪 LM head | 层平衡剪枝，单独分析/排除 LM head |
| signed 正负抵消 | 净值近零但正贡献大 | 同时保存 positive、negative、signed |
| 文件过大 | 写盘阻塞 | 关键 checkpoint 保存参数级，其余聚合；CPU 异步写入 |
| 随机剪枝波动 | 曲线不稳定 | 多 mask 和 CI |
| 恢复改变结果 | RNG/data cursor 丢失 | 保存 RNG 和 cursor，做中断一致性测试 |

## 30. 参考依据

实施时核对：EleutherAI Pythia 官方论文与仓库、Pythia-160M/410M 模型卡、Pythia deduplicated Pile 数据、GLUE SST-2/MNLI/RTE、参数积分梯度原论文、Microbatch-level U-statistic 推导文档，以及导师关于训练过程可解释性与剪枝验证的会议要求。

## 31. 最终执行原则

> 先验证估计器，再验证数值积分近似；先完成 160M 最小闭环，再训练 410M；先证明重要性能够预测功能损伤，再解释监督与预训练的结构差异；每个下游任务单独完成，最后才进行跨任务总结。

任何阶段未通过停止条件，都必须先修复该阶段，不得依靠扩大模型或增加任务掩盖方法问题。
