# 参数重要性估计核心代码分析报告

> 本报告面向需要阅读 `param-importance-nlp` 参数重要性估计实现的读者，选取最小核心路径（4 个文件），
> 对关键代码逐段给出数学推导、实现意图、契约边界与易错点注释。

## 元信息

| 项目 | 值 |
|---|---|
| 日期 | 2026-08-05 |
| 仓库 | `parameter-importance`（`D:/Personal/Code/parameter-importance`） |
| 分支 / 提交 | `feat/stage0-completion` / `a82caca29c6f09e9a04152e6a9da5b5a0e376b56`（报告生成时） |
| 分析范围 | 4 个核心文件（见下表），不包含 Stage 2/3 实验流水线、剪枝与基线 |
| 配套数学规格 | [`docs/mathematics.md`](mathematics.md) |

## 0. 摘要与阅读导航

本系统把“参数重要性”定义在**局部梯度空间**：每个训练步，用同一全局 batch 的
microbatch 梯度估计 \(\eta_{g(k),t}\mu_{k,t}^2\)（固定状态下的局部梯度平方目标），
再跨步累计成 signed / positive / negative-mass / absolute 四个视图，供后续剪枝、
轨迹分析与实验决策消费。

核心实现只依赖 4 个文件，对应 4 层职责：

| 文件 | 职责 | 关键符号 |
|---|---|---|
| `src/param_importance_nlp/core/sufficient_statistics.py` | 流式、可合并的梯度充分统计量 | S1/S2、G1/G2/N1/N2 |
| `src/param_importance_nlp/core/estimators.py` | 纯张量估计核与公开 score 包装 | \(\widehat C^U\)、\(\widehat C^{\mathrm{raw}}\) |
| `src/param_importance_nlp/core/accumulator.py` | 长期累计状态与不变量 | signed / positive / negative / absolute |
| `src/param_importance_nlp/runtime/training.py` 的 `OnlineImportanceTracker` | 把以上三者接入训练步与 checkpoint | micro_samples、clip_factor |

推荐阅读顺序：`sufficient_statistics.py` → `estimators.py` → `accumulator.py` →
`training.py` 中的 `OnlineImportanceTracker`。前三者构成“估计数学”，第四个说明
“如何在真实训练中产生、提交和恢复”。

单步数据流如下：

```mermaid
flowchart LR
    A[同一步的 M 个 microbatch 平均梯度] --> B[OnlineImportanceTracker.stage_distributed]
    B --> C[WeightedSufficientStatistics: G1 G2 N1 N2]
    C --> D[estimators.py 核: weighted_u / raw / double]
    D --> E[EstimatorResult: core + score]
    E --> F[ImportanceAccumulator.add_step]
    F --> G[四视图 + movement 状态]
    G --> H[checkpoint / importance snapshot / trajectory]
```

---

## 1. 数学地基：`core/sufficient_statistics.py`

### 1.1 设计动机

估计器需要的不是每个 microbatch 的梯度本身，而是它的**可合并汇总量**：

\[
S_1=\sum_{m=1}^{M}g_m,\qquad
S_2=\sum_{m=1}^{M}g_m^{\odot2}
\]

或加权版本：

\[
G_1=\sum_{m=1}^{M}w_mg_m,\qquad
G_2=\sum_{m=1}^{M}w_m^2g_m^{\odot2},\qquad
N_1=\sum_{m=1}^{M}w_m,\qquad
N_2=\sum_{m=1}^{M}w_m^2.
\]

这种设计的三个直接收益：

1. **流式**：不需要保存 M 份梯度，每个参数张量只维护 S1/S2 两个同形状张量；
2. **可合并**：不同 shard（例如不同 GPU、不同恢复批次）的统计量只需加法归并，
   不需要重放原始梯度；
3. **跨 GPU 高效**：DDP 场景下只需 all-reduce 上述几个张量与标量，而不是
   all-gather 每个参数梯度（见第 4 章）。

### 1.2 等权统计量 `EqualSufficientStatistics`

构造入口 `from_samples`（`sufficient_statistics.py:87`）用流式等价公式累计：

```python
first = samples[0].to(dtype=accumulation_dtype)
s1_values = {name: torch.zeros_like(value) for name, value in first.items()}
s2_values = {name: torch.zeros_like(value) for name, value in first.items()}
for sample in samples:
    converted = sample.to(dtype=accumulation_dtype)
    for name, gradient in converted.items():
        s1_values[name] = s1_values[name] + gradient
        s2_values[name] = s2_values[name] + gradient.square()
```

注释要点：

- **显式转换 dtype**：所有梯度在进入统计量时统一转换到 `accumulation_dtype`
  （FP32 或 FP64），避免 estimator 内部发生不可追溯的隐式精度变化。
- **输入顺序不影响数学结果**：S1/S2 是交换求和，因此 `merge` 可以安全地把两个
  shard 的统计量直接相加（`sufficient_statistics.py:122`）。
- **`merge` 的 fail-closed 条件**（`sufficient_statistics.py:125-135`）：dtype、
  `statistical_unit`、`sampling_design` 不一致时拒绝合并。这是为了防止把“不同
  统计语义”的 shard 静默拼在一起——例如一个 shard 按 microbatch 平均梯度、
  另一个按逐样本梯度，相加后没有任何意义。
- **`mean_gradient`**（`sufficient_statistics.py:119`）：\(S_1/M\)，这是后续 raw
  估计和优化器共用的平均梯度。

### 1.3 加权统计量 `WeightedSufficientStatistics`

语言模型场景中，每个 microbatch 的有效目标 token 数可能不同，此时不能简单等权。
加权 U-statistic 需要四元组 G1/G2/N1/N2（`sufficient_statistics.py:143`）。

`from_samples` 的累计逻辑（`sufficient_statistics.py:191`）：

```python
for sample, weight in zip(samples, numeric_weights, strict=True):
    converted = sample.to(dtype=accumulation_dtype)
    for name, gradient in converted.items():
        g1_values[name] = g1_values[name] + weight * gradient
        g2_values[name] = g2_values[name] + weight**2 * gradient.square()
```

注意 G2 累计的是 `weight**2 * gradient.square()`，不是 `(weight * gradient).square()`——
两者数学等价，但这里显式写出权重平方，与数学规格的
\(\sum_m w_m^2g_m^2\) 保持逐字对应，降低实现歧义。

两个关键属性：

```python
@property
def denominator(self) -> float:
    return self.n1**2 - self.n2

@property
def mean_gradient(self) -> TensorMap:
    return self.g1 / self.n1
```

- **分母** \(N_1^2-N_2=\sum_{m\ne n}w_mw_n\)，当 \(M=1\) 或权重全部集中在一个
  microbatch 时趋于零，估计核会拒绝计算（见 2.2 节）。
- **均值** \(G_1/N_1\) 是加权平均梯度，与优化器使用的全局平均梯度语义一致。

### 1.4 假设字段与“声明即合同”

`WeightedSufficientStatistics` 强制携带四个结构化字段：

| 字段 | 含义 |
|---|---|
| `weights_exogenous` | 权重相对被估计梯度是否为外生设计量（例如由 labels/mask 预先决定） |
| `common_mean_assumption` | 参与去对角配对的统计单元是否具有同一目标均值 |
| `statistical_unit` | 统计单元（本项目固定为 `microbatch_mean_gradient`） |
| `weight_unit` | 权重单位（`effective_loss_units`，即有效 token 数） |

设计上，**数值代码不会自动推断这些假设**。它们必须作为 artifact 的结构化字段
保存，由上层（`EstimatorResult.from_weighted_u`）决定能否生成“严格无偏”声明。
这是本项目最重要的契约之一：无偏性不是默认值，而是显式声明出来的。

---

## 2. 估计核：`core/estimators.py`

### 2.1 设计原则：core 与 score 分离

文件头注释写得很清楚：**所有 kernel 只返回“未乘学习率、未裁剪”的 `core`**。
学习率和裁剪属于训练 step 的尺度合同，统一由 `EstimatorResult.from_core` 应用。

为什么这样分层：

- kernel 是纯数学，可以在固定状态 fixture 中独立验证（FP64 oracle 对照）；
- 学习率是参数组级动态量，属于运行时事实，不应混进统计核；
- 裁剪因子来自同批随机梯度时，它本身是随机量，乘进 score 后会改变无偏性
  声明（见 2.3 节），必须在边界处显式记录 `clip_source`。

### 2.2 四个 kernel 逐个注释

#### raw：同批平均梯度平方（对照）

```python
def raw_importance(mean_gradient: TensorMap) -> TensorMap:
    return _square(mean_gradient)
```

公式：\(\widehat C^{\mathrm{raw}}=\eta_t\bar g_t^{\odot2}\)。

分析：

- 实现最简单，不需要额外前向/反向；
- 但同一个 batch 同时用于更新方向与贡献评价，估计量保留协方差偏差：
  \(\mathbb E[\bar g^2]=\mu^2+\sigma^2/M\)，即**正偏差**来自随机梯度方差；
- 因此它被定位为“关键对照”，不是 U-statistic 的别名。

#### double-sample：两个独立样本的梯度乘积

```python
def double_sample_importance(mean_gradient_a, mean_gradient_b):
    result = mean_gradient_a * mean_gradient_b
    ...
```

公式：\(\widehat\Omega_D=\eta_t\bar X_A\bar Y_{B'}\)。

分析：

- 两个独立 batch 的均值乘积在各自无偏条件下给出
  \(\mathbb E[\widehat\Omega_D]=\eta_t\mu_X\mu_Y\)，边界直观；
- 代价是固定总样本预算时把样本拆成两半，方差增大
  （数学规格 9.2 节给出了显式方差公式）；
- 在线实现中把 microbatch 按 `(rank + local_index) % 2` 稳定地切成两半
  （见 4.3 节），保证所有 rank 得到完全相同的 score。

#### equal-U：等权去对角 U-statistic

```python
def equal_u_importance(statistics: EqualSufficientStatistics) -> TensorMap:
    if statistics.count < 2:
        raise CoreContractError("等权 U-statistic 至少需要两个统计单元")
    denominator = statistics.count * (statistics.count - 1)
    result = (statistics.s1 * statistics.s1 - statistics.s2) / denominator
    ...
```

公式：

\[
\widehat C^U_{k,t}
=
\eta_{g(k),t}
\frac{S_{1,k,t}^2-S_{2,k,t}}{M(M-1)}
=
\eta_{g(k),t}
\frac{\sum_{m\ne n}g_{m,k,t}\,g_{n,k,t}}{M(M-1)}.
\]

分析：

- 代数恒等式 \((S_1)^2-S_2=\sum_{m\ne n}g_mg_n\) 删除了“自己乘自己”的对角项，
  从而消除 raw 的正偏差；
- `M=1` 时无法删除对角项，必须抛异常；
- **单步输出可能为负**：即使目标 \(\eta\mu^2\ge0\)，无偏估计器的有限样本波动
  也允许出现负值。kernel 内绝不 `clamp_min(0)`——一旦截断就重新引入正偏差。

#### weighted-U：不等权去对角（正式主公式）

```python
def weighted_u_importance(
    statistics: WeightedSufficientStatistics,
    *,
    require_unbiasedness_assumptions: bool = False,
) -> TensorMap:
    if statistics.count < 2:
        raise CoreContractError(...)
    denominator = statistics.denominator
    if not math.isfinite(denominator) or denominator <= 0:
        raise CoreContractError("加权 U-statistic 分母 N1**2-N2 必须为正")
    if require_unbiasedness_assumptions and not (
        statistics.weights_exogenous and statistics.common_mean_assumption
    ):
        raise CoreContractError("未满足加权 U 的外生权重与共同均值声明")
    result = (statistics.g1 * statistics.g1 - statistics.g2) / denominator
    ...
```

公式：

\[
\widehat{\mu^2}_{\,U,w}
=
\frac{
\left(\sum_m w_mg_m\right)^2
-
\sum_m w_m^2g_m^2
}{
1-\sum_m w_m^2
}.
\]

（代码用 \(G_1^2-G_2\) 与 \(N_1^2-N_2\) 表示，等权时退化为普通 U。）

分析：

- `require_unbiasedness_assumptions=True` 时，如果权重非外生或共同均值假设未
  声明，直接失败——这是“声明防线”在数值层的体现；
- 关闭该开关仍可计算描述性 plug-in 数值，但上层不得自动生成无偏声明；
- 分母必须为正：等权时对应 `M >= 2`，加权时对应至少两个非零权重统计单元。

#### cross-U：一般交叉 U（X≠Y 场景）

```python
def cross_u_importance(x_samples, y_samples, *, x_weights=None, y_weights=None,
                       exclude_matching_pairs=True):
    ...
    if not exclude_matching_pairs:
        return (sx / sum(wx)) * (sy / sum(wy))
    ...
    result = (sx * sy - diagonal) / denominator
```

公式（删除同索引对角项时）：

\[
\frac{
(\sum_m w^X_m X_m)(\sum_m w^Y_m Y_m)
-
\sum_m w^X_m w^Y_m X_mY_m
}{
(\sum_m w^X_m)(\sum_m w^Y_m)-\sum_m w^X_mw^Y_m
}.
\]

分析：

- 该核估计 \(\mathbb E[X]\mathbb E[Y]\)，**允许同一观测内部的 X/Y 相关**，因此
  能覆盖“起点梯度与路径梯度来自同一统计单元”的一般路径场景；
- 若两组样本来自彼此独立的 sampling stream，设
  `exclude_matching_pairs=False`，退化为两个加权均值的乘积（即 double-sample）；
- 当前在线训练只使用前四个核；cross-U 主要服务于固定状态的对照验证。

### 2.3 `EstimatorResult` 与无偏性声明

`EstimatorResult`（`estimators.py:193`）冻结 estimator core 与训练尺度后的公开
score。字段语义：

| 字段 | 语义 |
|---|---|
| `core` | 未乘学习率、未裁剪的估计核输出 |
| `score` | 按 `eta[group] * clip_factor` 只缩放一次后的公开分数 |
| `learning_rates` | 参数组 → 实际学习率映射 |
| `clip_factor` / `clip_source` | 裁剪因子及其来源 |
| `unbiasedness_claim` | `UNBIASED_FIXED_STATE` / `PLUGIN_SAME_BATCH_CLIP` / `NO_UNBIASEDNESS_CLAIM` |

`from_core`（`estimators.py:256`）的关键逻辑：

```python
if clip_source == "same_batch_mean_gradient":
    unbiasedness_claim = PLUGIN_SAME_BATCH_CLIP
...
for name, value in core.items():
    lr = _learning_rate_for_name(core, name, normalized_lr)
    scaled[name] = value * lr * float(clip_factor)
```

注释要点：

- 学习率按参数组（`registry.record(name).group_id`）逐坐标映射，**禁止用一个
  全局标量覆盖多参数组学习率**；
- 同批随机 clip 因子被强制改写为 `PLUGIN_SAME_BATCH_CLIP` 声明：因为因子来自
  同一批随机梯度，乘积是 plug-in 在线分数，不能再声称严格无偏；
- `__post_init__` 中，任何“被裁剪却仍声明严格无偏”的组合都会直接失败。

`from_equal_u` / `from_weighted_u`（`estimators.py:298/336`）自动绑定声明边界：

```python
claim = (
    UNBIASED_FIXED_STATE
    if assumptions_hold and clip_source == "none" and float(clip_factor) == 1.0
    else NO_UNBIASEDNESS_CLAIM
)
public_name = (
    "local_gradient_space_importance_u_weighted"
    if clip_source == "none" and float(clip_factor) == 1.0
    else "local_gradient_space_importance_u_clipped"
)
```

也就是说：**同一个数学核，剪不裁剪会得到不同的公开名称和不同的无偏性声明**。
未裁剪版本继承 U 核的固定状态无偏性；裁剪版本改名 `*_clipped` 并降级为
plug-in 在线分数。这保证 artifact 命名与统计性质一一对应。

### 2.4 `global_clip_factor`

```python
squared_norm = sum(
    float(value.detach().to(torch.float64).square().sum().item())
    for value in mean_gradient.values()
)
norm = math.sqrt(squared_norm)
return min(1.0, max_norm / (norm + epsilon))
```

注释要点：

- 计算在 **FP64** 下完成，避免 FP32 累加平方和时精度损失；
- 返回 `min(1, G_max/(||ḡ||+ε))`，只计算因子、不修改输入张量；
- `epsilon` 防止零梯度时除零，同时使因子恒有界于 [0,1]。

### 2.5 估计器一览表

| 公开名称 | 公式（未乘学习率） | 目标 | 严格无偏条件 |
|---|---|---|---|
| `local_gradient_space_importance_u` | \((S_1^2-S_2)/(M(M-1))\) | \(\eta\mu^2\) | 固定状态、未裁剪、i.i.d. 统计单元 |
| `local_gradient_space_importance_u_weighted` | \((G_1^2-G_2)/(N_1^2-N_2)\) | \(\eta\mu^2\) | 上述条件 + 权重外生 + 共同均值 |
| `double_sample_gradient_importance` | \(\bar g_A\bar g_{B'}\) | \(\eta\mu_X\mu_Y\) | 两个独立 batch |
| `raw_same_batch_gradient_importance` | \(\bar g^2\) | — | 无（保留正偏差作为对照） |
| `*_clipped` 系列 | 同核心 × 同批 clip 因子 | 在线 plug-in 分数 | 无，`clip_source=same_batch_mean_gradient` |

---

## 3. 长期累计：`core/accumulator.py`

### 3.1 内部状态布局

`ImportanceAccumulator`（`accumulator.py:30`）持有多组与参数同形状的 TensorMap：

| 分组 | 状态 | 含义 |
|---|---|---|
| 重要性四视图 | `_positive` / `_negative` | 正部与负部质量累计 |
| 派生态 | `_raw` / `_raw_clipped` | raw 对照与同批 clip 的 plug-in raw |
| 数据更新 | `_data_movement` / `_data_displacement` | 数据驱动更新的绝对路径与有符号位移 |
| 完整更新 | `_total_movement` / `_total_displacement` | 含 weight decay 的实际更新 |
| 权重衰减 | `_weight_decay_movement` / `_weight_decay_displacement` | 仅 weight decay 分量 |
| 端点 | `_magnitude` / `_initial_parameters` / `_last_parameters` | 幅值基线与首尾参数 |

公开视图由基础量派生，而不是独立浮点累计：

```python
@property
def signed(self) -> TensorMap:
    return self._positive - self._negative

@property
def absolute(self) -> TensorMap:
    return self._positive + self._negative
```

这样保证四个视图永远满足代数恒等式：

\[
\Omega^{\mathrm{signed}}=\Omega^+-\Omega^-,\qquad
\Omega^{\mathrm{abs}}=\Omega^++\Omega^-.
\]

### 3.2 `add_step`：原子式提交

`add_step`（`accumulator.py:132`）先校验全部候选输入，再原地提交；任一输入非有限
或坐标不一致时不会产生部分累计。核心累计逻辑：

```python
for name, value in converted.items():
    self._positive[name].add_(value.clamp_min(0))
    self._negative[name].add_((-value).clamp_min(0))
if converted_raw is not None:
    for name, value in converted_raw.items():
        self._raw[name].add_(value)
...
if converted_data is not None:
    for name, value in converted_data.items():
        self._data_movement[name].add_(value.abs())
        self._data_displacement[name].add_(value)
```

注释要点：

- **正负分解发生在累计层，而不是估计层**：单步 score 可能为负，但累计器只保存
  非负的正部/负部质量，保证 `signed = positive - negative_mass` 逐坐标成立；
- **raw 系列强制非负**（`accumulator.py:189-199`）：raw 是平方量，出现负值说明
  上游 bug，直接抛 `CoreContractError`；
- **movement 与 displacement 的区别**：movement 累计每一步的 `|δ|`（路径长度），
  displacement 累计有符号 `δ`（净位移），`net_*_movement` 是净位移的绝对值。
  这一区分让“来回震荡”与“单向移动”可被分开观察；
- `current_parameters` 只刷新幅值与最后端点，不参与正负质量累计。

### 3.3 `validate_invariants`

`validate_invariants`（`accumulator.py:252`）逐坐标检查：

1. positive / negative mass 非负；
2. `signed == positive - negative_mass`；
3. `absolute == positive + negative_mass`；
4. 首尾端点差 == 累计总位移（仅在有初始参数时）。

端点一致性的容差处理值得注意（`accumulator.py:268-274`）：首尾参数直接相减与
逐 step delta 累加的浮点运算顺序不同，FP32 下不能要求逐位相等，因此按
`eps * max(1, steps) * 4` 自适应放宽。**容差只用于一致性报错，不会生成或修饰
任何公开统计量**——这是“校验不影响数值”的干净边界。

### 3.4 区间增量与断点恢复

- `delta_since(previous)`（`accumulator.py:284`）：返回两个 commit 边界之间的
  signed / absolute / raw / movement 增量，是 Stage 4/5 阶段轨迹分析的原料；
- `state_dict()`（`accumulator.py:304`）：只输出 primitive 与 TensorMap，
  不使用 pickle 对象图，保证 checkpoint 可跨环境恢复；
- `load_state_dict()`（`accumulator.py:328`）：严格恢复，并对 0.3.x 的 v1 状态
  做**无损可知迁移**——v1 缺失的 clipped-raw、weight-decay 分解与参数端点
  无法反推，迁移时显式置零并保持 `has_initial_parameters=false`，不伪造历史。

### 3.5 数学累计公式表

| 视图 | 定义 | 用途 |
|---|---|---|
| signed | \(\sum_t \widehat C_{k,t}\) | 净损失作用，允许正负抵消 |
| positive | \(\sum_t [\widehat C_{k,t}]_+\) | 非负分布与剪枝排序 |
| negative_mass | \(\sum_t [-\widehat C_{k,t}]_+\) | 负贡献质量 |
| absolute | \(\sum_t |\widehat C_{k,t}|\) | 总活动强度 |
| data / total / weight-decay movement | \(\sum_t|\delta_t|\) | 路径长度 |
| net movement | \(|\sum_t\delta_t|\) | 净位移 |

---

## 4. 在线集成：`runtime/training.py` 中的 `OnlineImportanceTracker`

### 4.1 在训练引擎中的位置

`TrainingEngine._run_attempt`（`training.py:1436`）的单步时序：

```text
model.train() → optimizer.zero_grad()
  → _collect_micro_gradients(microbatches)        # 每个 microbatch 本地平均梯度
  → _global_mean_gradient(...)                    # 得到 micro_samples 与全局均值
  → GradientAttempt.capture().check_finite()      # 非有限则 skip
  → attempt.clip(max_grad_norm)                   # 得到 clip_factor
  → tracker.stage_distributed(micro_samples, weights, lrs, clip_factor)
  → bridge.step()                                 # 真正执行 AdamW 更新
  → tracker.commit(main, raw_unclipped, raw_clipped, outcome)
  → scheduler.step() / checkpoint 发布
```

关键位置：

- `training.py:1478`：skip 时 `tracker.record_skip()`，不伪造零贡献；
- `training.py:1524`：`stage_distributed` 在 optimizer step **之前**调用，只产生
  score、不修改累计器；
- `training.py:1545`：`commit` 在 bridge step 之后调用，把 score 与真实更新
  delta 一起原子提交。

### 4.2 构造

```python
template = TensorMap(
    {name: registry.parameter(name).detach() for name in registry.eligible_names},
    registry=registry,
)
self.accumulator = ImportanceAccumulator(
    template, accumulation_dtype=_dtype_from_name(spec.accumulation_dtype)
)
self.accumulator.set_initial_parameters(template)
```

注释要点：

- 统计范围由 `registry.eligible_names` 决定，避免把冻结/不可训练参数混入；
- 训练起点即被记为初始参数端点，供第 3 章的首尾位移校验使用。

### 4.3 `stage_distributed`：全局归约 + 估计

这是在线估计的心脏（`training.py:695`）。分四步分析。

**第一步：本地加权统计量**

```python
converted = [sample.to(dtype=dtype) for sample in micro_gradients]
g1 = TensorMap.zeros_like(converted[0], dtype=dtype)
g2 = TensorMap.zeros_like(converted[0], dtype=dtype)
for sample, weight in zip(converted, weights, strict=True):
    g1 = g1 + sample * float(weight)
    g2 = g2 + sample.map(torch.square) * float(weight) ** 2
```

对应数学规格的 \(G_1=\sum w_mg_m\)、\(G_2=\sum w_m^2g_m^2\)。

**第二步：只归约统计量，不归约梯度**

```python
reduced = reducer.sum_tensors({
    **{f"g1:{name}": value for name, value in g1.items()},
    **{f"g2:{name}": value for name, value in g2.items()},
    **scalars,   # __n1__ = sum(weights), __n2__ = sum(weights**2)
})
```

这是 DDP 语义的关键：U 只需要全局 G1/G2/N1/N2/count，**无需 all-gather 每个
参数梯度**。所有 rank 因此得到完全相同的 estimator score，同时通信量从参数
数量级降到统计量数量级。

**第三步：构造统计量与 raw 对照**

```python
global_statistics = WeightedSufficientStatistics(
    count=reducer.sum_int(len(converted)),
    g1=..., g2=..., n1=..., n2=...,
    accumulation_dtype=dtype,
    statistical_unit="microbatch_mean_gradient",
    weight_unit="effective_loss_units",
    sampling_design="ordered_disjoint_microbatches",
    weights_exogenous=self.spec.weights_exogenous,
    common_mean_assumption=self.spec.common_mean_assumption,
)
```

这里把**声明字段一次性写入统计量对象**，之后 `EstimatorResult.from_weighted_u`
会自动决定无偏性声明。

```python
clip_source = "none" if clip_factor == 1.0 else "same_batch_mean_gradient"
raw_unclipped = EstimatorResult.from_core("raw_same_batch_gradient_importance", ...)
raw_clipped = EstimatorResult.from_core("raw_same_batch_gradient_importance_clipped", ...)
```

**第四步：按配置选择主估计器**

- `estimator_name == "raw"`：直接返回 raw 系列；
- `estimator_name == "u"`（默认主指标）：`EstimatorResult.from_weighted_u(...)`，
  得到 `local_gradient_space_importance_u_weighted` 或 `*_clipped`；
- 其余走 double-sample：把本地 microbatch 按 `(rank + local_micro_index) % 2`
  稳定分成两半，分别 all-reduce 加权梯度与权重，再计算两个全局均值之积。

double-sample 的稳定映射值得注意（`training.py:777-815`）：映射基于
`rank + index` 而非纯本地 index，保证在固定 rank 布局下每个 microbatch 归属
确定，且所有 rank 计算出的两半全局均值一致。

### 4.4 `commit`：把分数与真实更新绑定

```python
def commit(self, main, raw_unclipped, raw_clipped, outcome):
    data = TensorMap(outcome.data_delta, registry=self.registry)
    total = TensorMap(outcome.total_delta, registry=self.registry)
    ...
    self.accumulator.add_step(
        main.score,
        raw=raw_unclipped.score,
        raw_clipped=raw_clipped.score,
        data_update=data,
        total_update=total,
        weight_decay_update=...,
        current_parameters=parameters,
    )
```

注释要点：

- `main.score` 是**已经乘过学习率和 clip 因子**的公开分数，正负分解发生在
  `add_step` 内；
- 同时记录数据驱动更新（不含 weight decay）与完整更新，使“权重衰减贡献”可
  从 total 中分离出来单独分析；
- skip 的 attempt 只走 `record_skip`，绝不传入零贡献伪装成功步，保证
  `successful_steps` 语义干净。

### 4.5 checkpoint 与 trajectory

- `_checkpoint_state()`（`training.py:1250`）把 `tracker.accumulator.state_dict()`
  写入 checkpoint，与模型/optimizer/scheduler/RNG/数据游标同构存储；
- `resume` 时 `accumulator.load_state_dict(importance)`（`training.py:1419`）
  严格恢复；
- 每次保存 checkpoint 时生成 `ImportanceTrajectoryPoint(global_step, checkpoint_id,
  snapshot)`（`training.py:1278-1283`），构成可审计的重要性轨迹；
- 若 checkpoint 发布失败，内存中的 trajectory point 会被回滚
  （`training.py:1290-1293`），不产生孤儿记录。

---

## 5. 契约边界与易错点汇总

### 5.1 无偏性声明矩阵

| 估计量 | 未裁剪 + 假设成立 | 同批 clip | 无声明 |
|---|---|---|---|
| `local_gradient_space_importance_u` | `UNBIASED_FIXED_STATE` | — | — |
| `local_gradient_space_importance_u_weighted` | `UNBIASED_FIXED_STATE`（还需外生权重 + 共同均值） | — | — |
| `double_sample_gradient_importance` | `UNBIASED_FIXED_STATE` | `PLUGIN_SAME_BATCH_CLIP` | — |
| `raw_same_batch_gradient_importance` | — | — | `NO_UNBIASEDNESS_CLAIM` |
| `*_clipped` 任何系列 | — | `PLUGIN_SAME_BATCH_CLIP` | — |

### 5.2 代码级不变量

1. `signed == positive - negative_mass`、`absolute == positive + negative_mass`
   逐坐标成立；
2. raw 系列永远非负；
3. 估计核输出非有限 → 直接抛异常，绝不静默；
4. dtype 声明与张量实际 dtype 不一致 → fail-closed；
5. `M=1` 拒绝计算 U；clip 因子 ≠ 1 时必须提供真实 `clip_source`；
6. 单步 U 输出为负是正常波动，kernel 内不 clamp；
7. 累计器状态迁移 v1→v2 对无法反推的字段显式置零，不伪造历史。

### 5.3 与数学规格的章节对照

| 本报告章节 | 数学规格章节 |
|---|---|
| 1 充分统计量 | §9.5–9.7、§10（多 GPU） |
| 2 估计核 | §9.1–9.9、§7（局部梯度空间） |
| 3 累计器 | §12（训练过程中的累计重要性） |
| 4 在线集成 | §18.1（算法 A：正式训练中的局部 U-statistic）、§19（不变量） |
| 5 易错边界 | §21（实现时最容易混淆的边界） |

---

## 6. 延伸阅读（不展开）

- `core/baselines.py`：Synaptic Intelligence、经验 Fisher 等对照基线；
- `core/metrics.py` / `core/pruning.py`：重要性分数的下游评价与剪枝消费；
- `experiments/stage2*.py`：固定状态下的估计器配对比较与决策；
- `core/quadrature.py` 与 `experiments/stage3*.py`：更新路径积分（另一种估计路线）。

---

## 附录 A：符号表

| 符号 | 含义 |
|---|---|
| \(M\) | 一个 optimizer step 内的全局 microbatch 数 |
| \(g_m\) | 第 m 个 microbatch 的未同步、未裁剪平均梯度 |
| \(w_m\) | 第 m 个 microbatch 的有效统计单元权重（有效 token 数） |
| \(S_1, S_2\) | 等权累计 \(\sum g_m\)、\(\sum g_m^2\) |
| \(G_1, G_2, N_1, N_2\) | 加权累计 \(\sum wg\)、\(\sum w^2g^2\)、\(\sum w\)、\(\sum w^2\) |
| \(\eta_{g(k),t}\) | 参数 k 所属 optimizer group 在第 t 步的学习率 |
| \(s_t\) | 同批全局裁剪因子 \(\min(1, G_{\max}/(\|\bar g\|+\varepsilon))\) |
| \(\widehat C_{k,t}\) | 第 t 步、参数 k 的单步重要性贡献 |
| \(\Omega_k^{\mathrm{signed/+/−/abs}}\) | 跨步累计的四个视图 |
