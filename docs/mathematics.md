# 参数重要性研究的数学规格说明

> 用途：本文件作为参数重要性实验的数学定义、代码实现和单元测试依据。所有主公式都给出明确的目标量、估计对象、成立条件与工程边界。实现代码时，应以本文件中的统一记号和符号约定为准，避免混用不同资料中的符号。

## 1. 研究范围与核心结论

本研究同时涉及两个层次的数学对象。

第一个层次是参数空间中的完整路径积分。它回答：一次实际参数更新从起点移动到终点时，每个参数坐标对某个固定损失函数的变化贡献了多少。该定义具有严格的路径积分分解和完备性。

第二个层次是正式大模型训练中采用的局部梯度空间贡献。它回答：在当前参数状态固定时，总体数据梯度的第一个参数分量平方有多大，以及如何用 minibatch 或 microbatch 梯度无偏地估计它。Microbatch-level U-statistic 的严格无偏性主要对应这一层次。

正式实验中的主指标定义为：

> 经 Microbatch-level U-statistic 去偏的局部梯度空间损失贡献。

它不能直接表述为：

> AdamW 完整参数更新路径积分的严格无偏估计。

完整路径积分、多节点数值积分和独立 probe batch 主要用于小模型数学校验；局部 U-statistic 主要用于大模型在线统计。路径平滑、EMA 路径方向贡献及其与损失贡献的组合目前属于可选扩展，不纳入第一版主指标。

---

## 2. 全局符号约定

### 2.1 参数、训练步和坐标

令 `P` 表示模型中参与研究的可学习标量参数总数。模型全部参数按固定顺序展开为向量

\[
\Theta=(\theta_1,\theta_2,\ldots,\theta_P)\in\mathbb R^P.
\]

其中，\(\theta_k\) 表示第 \(k\) 个标量参数，参数坐标索引满足 \(k\in\{1,2,\ldots,P\}\)。代码中不必真的把所有张量拼成一个长向量，但每个参数张量必须有稳定的名称、形状和扁平化顺序。

令 `T` 表示总优化步数。训练步索引记为

\[
t\in\{0,1,\ldots,T-1\}.
\]

令 \(\Theta_t\) 表示第 \(t\) 次参数更新之前的参数状态，令 \(\Theta_{t+1}\) 表示这次更新之后的参数状态。定义完整参数更新向量

\[
\Delta\Theta_t=\Theta_{t+1}-\Theta_t.
\]

令 \(\Delta\theta_{k,t}\) 表示 \(\Delta\Theta_t\) 的第 \(k\) 个分量，即

\[
\Delta\theta_{k,t}=\theta_{k,t+1}-\theta_{k,t}.
\]

本文件统一采用“更新后减更新前”的定义。因此，普通梯度下降中的 \(\Delta\Theta_t\) 通常与梯度方向相反。

### 2.2 数据、单样本损失和总体损失

令 \(z\) 表示一个独立的数据统计单元。它可以是一条分类样本、一条固定长度语言模型序列，或者一个已经明确定义损失权重的训练单元。令

\[
z\sim\mathcal D
\]

表示数据单元从总体数据分布 \(\mathcal D\) 中抽取。

令

\[
\ell(\Theta;z)\in\mathbb R
\]

表示参数为 \(\Theta\) 时，单个数据单元 \(z\) 的标量损失。

定义总体或 population 损失

\[
\mathcal L(\Theta)
=
\mathbb E_{z\sim\mathcal D}[\ell(\Theta;z)].
\]

如果研究对象是一个固定有限数据集 \(D=\{z_1,\ldots,z_{N_D}\}\)，其中 \(N_D\) 表示数据单元数量，则对应的全数据损失为

\[
\mathcal L_D(\Theta)
=
\frac{1}{N_D}\sum_{i=1}^{N_D}\ell(\Theta;z_i).
\]

后续所有“完备性”公式都要求使用同一个固定标量函数，例如始终使用 \(\mathcal L\)、始终使用 \(\mathcal L_D\)，或始终使用同一个固定 probe batch 的损失。若每个训练步使用不同 minibatch 损失，则跨步求和一般不再严格望远镜相消。

### 2.3 Batch 和 microbatch

令 \(\mathcal B\) 表示一个包含 `B` 个数据单元的 minibatch：

\[
\mathcal B=\{z_1,z_2,\ldots,z_B\}.
\]

定义该 minibatch 的平均损失

\[
\mathcal L_{\mathcal B}(\Theta)
=
\frac{1}{B}\sum_{i=1}^{B}\ell(\Theta;z_i).
\]

将该 minibatch 划分为 `M` 个互不重叠的 microbatch。令 \(I_m\) 表示第 \(m\) 个 microbatch 的样本索引集合，其中

\[
m\in\{1,2,\ldots,M\}.
\]

在等大小情形下，每个 microbatch 包含 `b` 个数据单元，并满足

\[
B=Mb.
\]

若不同 microbatch 的有效数据单元数不同，则令 \(b_m=|I_m|\) 表示第 \(m\) 个 microbatch 的有效单元数，并在后文使用加权公式。

### 2.4 梯度、向量运算和范数

定义单数据单元梯度向量

\[
g(\Theta;z)
=
\nabla_\Theta\ell(\Theta;z)
\in\mathbb R^P.
\]

其第 \(k\) 个分量写为

\[
g_k(\Theta;z)
=
\frac{\partial\ell(\Theta;z)}{\partial\theta_k}.
\]

定义总体梯度向量

\[
g^*(\Theta)
=
\nabla_\Theta\mathcal L(\Theta)
=
\mathbb E_{z\sim\mathcal D}[g(\Theta;z)].
\]

其第 \(k\) 个分量写为 \(g_k^*(\Theta)\)。

定义 minibatch 平均梯度

\[
g_{\mathcal B}(\Theta)
=
\frac{1}{B}\sum_{i=1}^{B}g(\Theta;z_i).
\]

定义第 \(m\) 个等大小 microbatch 的平均梯度

\[
g_m(\Theta)
=
\frac{1}{b}\sum_{i\in I_m}g(\Theta;z_i).
\]

符号 \(\odot\) 表示逐元素乘法。对于两个同形状向量 \(u,v\in\mathbb R^P\)，

\[
(u\odot v)_k=u_kv_k.
\]

符号 \(u^{\odot 2}=u\odot u\) 表示逐元素平方。符号 \(\langle u,v\rangle=u^\top v\) 表示欧氏内积，符号 \(\|u\|_2\) 表示欧氏范数。

---

## 3. 任务损失的明确约定

参数重要性完全由所选择的标量损失决定，因此训练、统计和 probe 评价必须使用明确且一致的损失定义。

### 3.1 因果语言模型损失

令一条语言模型序列写为

\[
z=(w_1,w_2,\ldots,w_L),
\]

其中 \(L\) 表示序列长度，\(w_j\) 表示第 \(j\) 个 token。令 \(m_j\in\{0,1\}\) 表示第 \(j\) 个目标 token 是否计入损失。定义一条序列的平均因果语言模型损失

\[
\ell_{\mathrm{LM}}(\Theta;z)
=
-
\frac{
\sum_{j=2}^{L}m_j
\log p_\Theta(w_j\mid w_1,\ldots,w_{j-1})
}{
\sum_{j=2}^{L}m_j
}.
\]

代码必须固定“按序列平均”还是“按全 batch 有效 token 平均”。若 microbatch 有效 token 数不同，而训练目标是全局 token 平均，则 U-statistic 的统计权重也必须按照有效 token 数处理，不能把不同 token 数的 microbatch 无条件等权。

### 3.2 单标签 token 分类损失

令 \(\mathcal C\) 表示候选类别集合，令 \(s_c(\Theta;z)\) 表示模型在指定标签位置对类别 \(c\in\mathcal C\) 对应单 token 的 logit。若真实类别为 \(y\in\mathcal C\)，定义分类损失

\[
\ell_{\mathrm{cls}}(\Theta;z)
=
-
\log
\frac{\exp(s_y(\Theta;z))}
{\sum_{c\in\mathcal C}\exp(s_c(\Theta;z))}.
\]

监督训练时，只对标签位置计算该损失；prompt 前缀不计入监督损失。所有候选标签必须在 tokenizer 下验证为单 token，或者采用严格的多 token 长度校正方案。

---

## 4. 输入空间积分梯度的基础定义

本节用于说明标准 Integrated Gradients 与本研究的参数空间路径积分之间的数学联系。

令

\[
F:\mathbb R^d\rightarrow\mathbb R
\]

表示一个输入为 \(d\) 维向量、输出为标量的可微函数。令 \(x\in\mathbb R^d\) 表示待解释输入，令 \(x'\in\mathbb R^d\) 表示基准输入。

定义从 \(x'\) 到 \(x\) 的线性路径

\[
\gamma(\alpha)
=
 x'+\alpha(x-x'),
\qquad \alpha\in[0,1].
\]

第 \(i\) 个输入特征的积分梯度定义为

\[
\operatorname{IG}_i(x;x')
=
(x_i-x'_i)
\int_0^1
\frac{\partial F(\gamma(\alpha))}{\partial x_i}
\,d\alpha.
\]

若 \(F\) 沿路径可微，则积分梯度满足完备性：

\[
\sum_{i=1}^{d}\operatorname{IG}_i(x;x')
=
F(x)-F(x').
\]

标准积分梯度是在输入空间中分解模型输出变化；本研究将同一条链式法则和线积分思想移到参数空间，用于分解损失函数沿优化路径的变化。项目毕业论文的第二、三章正是从这一同构关系出发建立参数路径积分框架。

---

## 5. 参数空间中的完整路径积分贡献

### 5.1 一般连续路径

固定训练步 \(t\)。令

\[
\Gamma_t:[0,1]\rightarrow\mathbb R^P
\]

表示连接参数起点 \(\Theta_t\) 和参数终点 \(\Theta_{t+1}\) 的一条连续可微路径，并满足

\[
\Gamma_t(0)=\Theta_t,
\qquad
\Gamma_t(1)=\Theta_{t+1}.
\]

令 \(\Gamma'_{t,k}(\alpha)\) 表示路径在第 \(k\) 个参数坐标上的导数。由链式法则，固定损失函数 \(\mathcal L\) 的单步变化为

\[
\mathcal L(\Theta_{t+1})-
\mathcal L(\Theta_t)
=
\int_0^1
\nabla_\Theta\mathcal L(\Gamma_t(\alpha))^\top
\Gamma'_t(\alpha)
\,d\alpha.
\]

将内积按参数坐标展开，得到

\[
\mathcal L(\Theta_{t+1})-
\mathcal L(\Theta_t)
=
\sum_{k=1}^{P}
\int_0^1
\frac{\partial\mathcal L(\Gamma_t(\alpha))}{\partial\theta_k}
\Gamma'_{t,k}(\alpha)
\,d\alpha.
\]

为了让“正值”表示对损失下降的正贡献，定义第 \(t\) 步第 \(k\) 个参数的完整路径损失下降贡献

\[
C_{k,t}^{\mathrm{path}}
=
-
\int_0^1
\frac{\partial\mathcal L(\Gamma_t(\alpha))}{\partial\theta_k}
\Gamma'_{t,k}(\alpha)
\,d\alpha.
\]

因此，单步完备性为

\[
\sum_{k=1}^{P}C_{k,t}^{\mathrm{path}}
=
\mathcal L(\Theta_t)-
\mathcal L(\Theta_{t+1}).
\]

若右侧为正，说明该步降低了损失；若右侧为负，说明该步在所评价的固定损失上增加了损失。

### 5.2 端点之间的线性路径

神经网络优化器只给出离散端点，因此本研究默认使用两端点之间的线段：

\[
\Gamma_t(\alpha)
=
\Theta_t+\alpha\Delta\Theta_t,
\qquad \alpha\in[0,1].
\]

此时

\[
\Gamma'_t(\alpha)=\Delta\Theta_t,
\]

所以第 \(k\) 个参数的路径贡献化为

\[
C_{k,t}^{\mathrm{path}}
=
-
\Delta\theta_{k,t}
\int_0^1
g_k^*(\Theta_t+\alpha\Delta\Theta_t)
\,d\alpha.
\]

为了简化记号，定义第 \(k\) 个参数在该路径上的平均总体梯度

\[
\bar g_{k,t}^{\mathrm{path}}
=
\int_0^1
 g_k^*(\Theta_t+\alpha\Delta\Theta_t)
\,d\alpha.
\]

于是

\[
C_{k,t}^{\mathrm{path}}
=
-
\Delta\theta_{k,t}
\bar g_{k,t}^{\mathrm{path}}.
\]

该乘法是逐坐标乘法，不包含不同参数坐标之间的交叉乘积。

### 5.3 整个训练过程的累计贡献

定义第 \(k\) 个参数在前 \(T\) 个更新步中的累计完整路径贡献

\[
\Omega_k^{\mathrm{path}}
=
\sum_{t=0}^{T-1}C_{k,t}^{\mathrm{path}}.
\]

若每一步始终评价同一个确定性损失函数 \(\mathcal L\)，并且积分计算精确，则

\[
\sum_{k=1}^{P}\Omega_k^{\mathrm{path}}
=
\mathcal L(\Theta_0)-
\mathcal L(\Theta_T).
\]

若每个训练步使用不同 minibatch 损失，则

\[
\sum_t
\left[
\mathcal L_{\mathcal B_t}(\Theta_t)-
\mathcal L_{\mathcal B_t}(\Theta_{t+1})
\right]
\]

一般不等于某个固定数据集在训练首尾的损失差。因此，跨步完备性检查必须使用固定 population、固定全数据集或固定 probe set。

### 5.4 正负号的解释

在本文件的统一约定下：

- \(C_{k,t}^{\mathrm{path}}>0\)：第 \(k\) 个参数的该步位移沿路径促进了损失下降；
- \(C_{k,t}^{\mathrm{path}}<0\)：该参数坐标的位移沿路径增加了所评价的损失，或者与总体梯度方向不一致；
- \(C_{k,t}^{\mathrm{path}}\approx0\)：可能表示该参数几乎没有移动、路径平均梯度接近零，或者正负作用发生抵消。

负贡献不能直接解释为“不重要”。它表示有方向的损失作用，需要与正贡献质量、绝对贡献和剪枝结果联合分析。

---

## 6. 路径积分的数值计算

### 6.1 通用求积形式

固定训练步 \(t\) 和参数坐标 \(k\)。定义标量路径梯度函数

\[
h_{k,t}(\alpha)
=
 g_k^*(\Theta_t+\alpha\Delta\Theta_t),
\qquad \alpha\in[0,1].
\]

则完整路径贡献为

\[
C_{k,t}^{\mathrm{path}}
=
-
\Delta\theta_{k,t}
\int_0^1h_{k,t}(\alpha)\,d\alpha.
\]

令 `Q` 表示任意数值求积方法使用的节点数量。令 \(\alpha_q\in[0,1]\) 表示第 \(q\) 个求积节点，令 \(a_q\) 表示对应权重，并满足通常的归一化条件

\[
\sum_{q=1}^{Q}a_q=1.
\]

定义数值平均梯度

\[
\widehat{\bar g}_{k,t}^{(Q)}
=
\sum_{q=1}^{Q}a_q
h_{k,t}(\alpha_q).
\]

相应的数值路径贡献为

\[
\widehat C_{k,t}^{(Q)}
=
-
\Delta\theta_{k,t}
\widehat{\bar g}_{k,t}^{(Q)}.
\]

代码实现时，对每个路径节点执行一次前向和反向传播，得到所有参数的梯度张量，然后按权重逐元素累加。不要为每个标量参数单独反向传播。

### 6.2 左端点、右端点和中点方法

左端点方法只使用路径起点，定义

\[
\widehat{\bar g}_{k,t}^{\mathrm{left}}
=
 h_{k,t}(0).
\]

因此

\[
\widehat C_{k,t}^{\mathrm{left}}
=
-
\Delta\theta_{k,t}
 g_k^*(\Theta_t).
\]

右端点方法只使用路径终点，定义

\[
\widehat{\bar g}_{k,t}^{\mathrm{right}}
=
 h_{k,t}(1).
\]

中点方法使用路径中点，定义

\[
\widehat{\bar g}_{k,t}^{\mathrm{mid}}
=
 h_{k,t}\left(\frac12\right).
\]

左端点方法计算成本最低。在 SGD 的局部近似中，它直接导出后文的梯度平方目标。

### 6.3 单区间梯形法

梯形法使用起点和终点两个梯度：

\[
\widehat{\bar g}_{k,t}^{\mathrm{trap}}
=
\frac12
\left[
 h_{k,t}(0)+h_{k,t}(1)
\right].
\]

对应贡献为

\[
\widehat C_{k,t}^{\mathrm{trap}}
=
-
\frac{\Delta\theta_{k,t}}{2}
\left[
 g_k^*(\Theta_t)+g_k^*(\Theta_{t+1})
\right].
\]

### 6.4 单区间 Simpson 法

Simpson 法使用起点、中点和终点三个梯度：

\[
\widehat{\bar g}_{k,t}^{\mathrm{Simp}}
=
\frac16
\left[
 h_{k,t}(0)
+4h_{k,t}\left(\frac12\right)
+h_{k,t}(1)
\right].
\]

对应贡献为

\[
\widehat C_{k,t}^{\mathrm{Simp}}
=
-
\frac{\Delta\theta_{k,t}}{6}
\left[
 g_k^*(\Theta_t)
+4g_k^*\left(\Theta_t+\frac12\Delta\Theta_t\right)
+g_k^*(\Theta_{t+1})
\right].
\]

项目毕业论文第三章比较了左端点、梯形和 Simpson 方法。本文档采用标准数值分析中的 Simpson 权重与误差阶，作为代码实现的统一依据。

### 6.5 复合等距方法

令 `n` 表示将区间 \([0,1]\) 等分的子区间数量。定义等距节点

\[
\alpha_j=\frac{j}{n},
\qquad j=0,1,\ldots,n.
\]

复合左端点方法为

\[
\widehat{\bar g}_{k,t}^{\mathrm{comp-left}}
=
\frac1n
\sum_{j=0}^{n-1}
 h_{k,t}\left(\frac{j}{n}\right).
\]

复合中点方法为

\[
\widehat{\bar g}_{k,t}^{\mathrm{comp-mid}}
=
\frac1n
\sum_{j=0}^{n-1}
 h_{k,t}\left(\frac{j+1/2}{n}\right).
\]

复合梯形法为

\[
\widehat{\bar g}_{k,t}^{\mathrm{comp-trap}}
=
\frac1n
\left[
\frac12h_{k,t}(0)
+
\sum_{j=1}^{n-1}h_{k,t}\left(\frac jn\right)
+
\frac12h_{k,t}(1)
\right].
\]

复合 Simpson 法要求 \(n\) 为正偶数，其公式为

\[
\widehat{\bar g}_{k,t}^{\mathrm{comp-Simp}}
=
\frac{1}{3n}
\left[
 h_{k,t}(0)+h_{k,t}(1)
+4\sum_{\substack{j=1\\j\ \mathrm{odd}}}^{n-1}
 h_{k,t}\left(\frac jn\right)
+2\sum_{\substack{j=2\\j\ \mathrm{even}}}^{n-2}
 h_{k,t}\left(\frac jn\right)
\right].
\]

### 6.6 Gauss-Legendre 求积

令 \(x_q\) 和 \(w_q\) 分别表示 `Q` 点 Gauss-Legendre 求积在区间 \([-1,1]\) 上的标准节点和标准权重。将其映射到 \([0,1]\)：

\[
\alpha_q=\frac{x_q+1}{2},
\qquad
 a_q=\frac{w_q}{2}.
\]

于是

\[
\int_0^1h_{k,t}(\alpha)\,d\alpha
\approx
\sum_{q=1}^{Q}a_qh_{k,t}(\alpha_q).
\]

`Q` 点 Gauss-Legendre 对次数不超过 \(2Q-1\) 的多项式积分精确。代码中可以直接使用 `numpy.polynomial.legendre.leggauss(Q)` 生成节点和权重，不建议手工硬编码高阶权重。

### 6.7 数值误差阶

假设 \(h_{k,t}\) 在 \([0,1]\) 上具有相应阶数的连续导数。对 `n` 个等距子区间，有以下典型误差上界：

\[
\left|E_{\mathrm{left}}\right|
\le
\frac{1}{2n}
\sup_{\alpha\in[0,1]}
\left|h'_{k,t}(\alpha)\right|,
\]

\[
\left|E_{\mathrm{mid}}\right|
\le
\frac{1}{24n^2}
\sup_{\alpha\in[0,1]}
\left|h''_{k,t}(\alpha)\right|,
\]

\[
\left|E_{\mathrm{trap}}\right|
\le
\frac{1}{12n^2}
\sup_{\alpha\in[0,1]}
\left|h''_{k,t}(\alpha)\right|,
\]

\[
\left|E_{\mathrm{Simp}}\right|
\le
\frac{1}{180n^4}
\sup_{\alpha\in[0,1]}
\left|h^{(4)}_{k,t}(\alpha)\right|.
\]

第 \(k\) 个参数贡献的绝对数值积分误差还要乘以 \(|\Delta\theta_{k,t}|\)：

\[
\left|
\widehat C_{k,t}^{(Q)}-C_{k,t}^{\mathrm{path}}
\right|
=
|\Delta\theta_{k,t}|
\left|
\widehat{\bar g}_{k,t}^{(Q)}-
\bar g_{k,t}^{\mathrm{path}}
\right|.
\]

### 6.8 完备性残差

令

\[
\Delta\mathcal L_t
=
\mathcal L(\Theta_t)-\mathcal L(\Theta_{t+1})
\]

表示固定评价损失在第 \(t\) 步的真实下降量。定义求积完备性绝对残差

\[
R_t^{\mathrm{abs}}
=
\left|
\sum_{k=1}^{P}
\widehat C_{k,t}^{(Q)}
-
\Delta\mathcal L_t
\right|.
\]

定义带稳定常数 \(\varepsilon_{\mathrm{num}}>0\) 的相对残差

\[
R_t^{\mathrm{rel}}
=
\frac{R_t^{\mathrm{abs}}}
{|\Delta\mathcal L_t|+\varepsilon_{\mathrm{num}}}.
\]

数值积分验证不能只检查总和残差，还应比较参数排序、层级排序和 top-k 集合。总和准确不代表每个坐标的分配准确。

---

## 7. 局部一阶梯度空间贡献

### 7.1 总体梯度记号

固定训练步 \(t\) 和参数坐标 \(k\)。定义当前参数状态下的总体梯度分量

\[
\mu_{k,t}
=
 g_k^*(\Theta_t)
=
\mathbb E_{z\sim\mathcal D}
\left[
\frac{\partial\ell(\Theta_t;z)}{\partial\theta_k}
\right].
\]

令 \(g(k)\) 表示坐标 \(k\) 所属的静态 optimizer 参数组，
\(\eta_{g(k),t}>0\) 表示第 \(t\) 步该参数组的动态学习率。参数组映射属于
``optimizer_contract_hash``；随 step 变化的学习率值属于运行时事件，不进入
``coordinate_registry_hash``。

### 7.2 理想 population SGD 更新

若使用总体梯度执行无动量、无权重衰减的 SGD，则理想参数更新为

\[
\Delta\theta_{k,t}^{\mathrm{SGD}}
=
-\eta_{g(k),t}\mu_{k,t}.
\]

在左端点局部近似下，第 \(k\) 个参数的损失下降贡献为

\[
C_{k,t}^{\mathrm{grad}}
=
-
\Delta\theta_{k,t}^{\mathrm{SGD}}
\mu_{k,t}
=
\eta_{g(k),t}\mu_{k,t}^2.
\]

这就是正式 Microbatch-level U-statistic 主要估计的目标量。

### 7.3 与完整路径积分的关系

定义总体损失在 \(\Theta_t\) 处的 Hessian 矩阵

\[
H_t
=
\nabla_\Theta^2\mathcal L(\Theta_t).
\]

由多元 Taylor 定理，存在 \(c\in(0,1)\)，使得

\[
\mathcal L(\Theta_t+\Delta\Theta_t)
-
\mathcal L(\Theta_t)
=
 g^*(\Theta_t)^\top\Delta\Theta_t
+
\frac12
\Delta\Theta_t^\top
H(\Theta_t+c\Delta\Theta_t)
\Delta\Theta_t.
\]

第一项就是左端点路径积分近似，第二项是由曲率造成的截断误差。因此，局部梯度空间贡献在步长较小、路径梯度变化较弱时更接近完整路径贡献。

### 7.4 全局梯度裁剪

令 \(G_{\max}>0\) 表示全局梯度范数阈值。定义未裁剪的总体或大样本平均梯度向量为 \(\mu_t\)。定义裁剪因子

\[
s_t
=
\min\left(
1,
\frac{G_{\max}}
{\|\mu_t\|_2+\varepsilon_{\mathrm{clip}}}
\right),
\]

其中 \(\varepsilon_{\mathrm{clip}}>0\) 用于避免除零。若裁剪后的 SGD 更新为

\[
\Delta\theta_{k,t}=-\eta_{g(k),t}s_t\mu_{k,t},
\]

则局部目标变为

\[
C_{k,t}^{\mathrm{grad,clip}}
=
\eta_{g(k),t}s_t\mu_{k,t}^2.
\]

裁剪因子只乘一次，因为一个梯度因子来自更新方向，另一个梯度因子来自损失的局部导数。正式实现应在所有 microbatch 梯度聚合之后计算全局裁剪因子；不得先对每个 microbatch 独立裁剪再代入 U-statistic。

上式中的 \(s_t\) 若由固定总体梯度给出，可视为目标的一部分；真实在线代码使用的
却是同一随机 batch 平均梯度计算出的 \(\widehat s_t\)。它与 U 核心共享随机性，
所以 \(\widehat s_t\widehat C^U\) 只能称为 plug-in 在线分数。除非另有独立裁剪
因子或专门证明，不得把未裁剪 U 的严格无偏性传递给该乘积。

---

## 8. 随机梯度噪声与同批次过估计

### 8.1 单点局部情形

固定参数状态 \(\Theta_t\) 和参数坐标 \(k\)。令单样本梯度随机变量为

\[
G_i
=
\frac{\partial\ell(\Theta_t;z_i)}{\partial\theta_k}.
\]

定义其均值和方差

\[
\mu
=
\mathbb E[G_i],
\qquad
\sigma^2
=
\operatorname{Var}(G_i).
\]

令 \(\bar G\) 表示大小为 \(B\) 的 minibatch 平均梯度：

\[
\bar G
=
\frac1B\sum_{i=1}^{B}G_i.
\]

原始同批次局部估计器为

\[
\widehat C_{\mathrm{raw}}
=
\eta_t\bar G^2.
\]

由于

\[
\mathbb E[\bar G^2]
=
\mu^2+\frac{\sigma^2}{B},
\]

所以

\[
\mathbb E[\widehat C_{\mathrm{raw}}]
=
\eta_t\mu^2
+
\eta_t\frac{\sigma^2}{B}.
\]

其偏差为

\[
\operatorname{Bias}(\widehat C_{\mathrm{raw}})
=
\eta_t\frac{\sigma^2}{B}>0.
\]

因此，即使 \(\mu=0\)，只要单样本梯度有方差，原始梯度平方仍会得到正的平均分数。这会把“样本间意见不一致”混入“稳定总体贡献”。

### 8.2 一般多节点路径情形

固定一条与后续估计样本独立的参数路径。对单个数据单元 \(z_i\)，定义更新起点处的梯度分量

\[
X_i
=
\frac{\partial\ell(\Theta_t;z_i)}{\partial\theta_k}.
\]

令 \(Q\) 表示路径求积节点数量，令 \(\alpha_q\) 和 \(a_q\) 分别表示第 \(q\) 个节点及权重。定义同一数据单元在路径上的加权梯度

\[
Y_i
=
\sum_{q=1}^{Q}
 a_q
\frac{\partial
\ell(\Theta_t+\alpha_q\Delta\Theta_t;z_i)
}{\partial\theta_k}.
\]

定义两个总体均值

\[
\mu_X=\mathbb E[X_i],
\qquad
\mu_Y=\mathbb E[Y_i].
\]

定义单样本方差和协方差

\[
\sigma_X^2=\operatorname{Var}(X_i),
\qquad
\sigma_Y^2=\operatorname{Var}(Y_i),
\qquad
c=\operatorname{Cov}(X_i,Y_i).
\]

对于理想总体 SGD 方向，吸收统一负号后定义目标量

\[
\Omega_k
=
\eta_t\mu_X\mu_Y.
\]

用同一个大小为 \(B\) 的 batch 同时估计 \(X\) 和 \(Y\)。定义

\[
\bar X
=
\frac1B\sum_{i=1}^{B}X_i,
\qquad
\bar Y
=
\frac1B\sum_{i=1}^{B}Y_i.
\]

原始估计器为

\[
\widehat\Omega_{\mathrm{raw}}
=
\eta_t\bar X\bar Y.
\]

其期望为

\[
\mathbb E[\widehat\Omega_{\mathrm{raw}}]
=
\Omega_k
+
\eta_t\frac{c}{B}.
\]

因此，一般偏差由更新点梯度与路径平均梯度之间的协方差决定，而不只是单点方差。只有在路径很短或路径梯度噪声近似不变时，\(c\) 才近似退化为梯度方差。

### 8.3 连续路径噪声协方差

令 \(\eta_k(\Theta)\) 表示参数位置 \(\Theta\) 处的 minibatch 梯度噪声，并满足

\[
\mathbb E[\eta_k(\Theta)]=0.
\]

在连续路径上，同源噪声造成的偏差近似为

\[
\eta_t
\int_0^1
\operatorname{Cov}
\left(
\eta_k(\Theta_t),
\eta_k(\Theta_t+\alpha\Delta\Theta_t)
\right)
\,d\alpha.
\]

当该积分为正时表现为过估计；当路径上出现负协方差时也可能产生低估。因此，最一般的术语应是“同源噪声协方差偏差”，而不是在所有场景下都断言正偏。

---

## 9. 不同估计方法

## 9.1 原始同批次估计

一般路径目标的原始估计器为

\[
\widehat\Omega_{\mathrm{raw}}
=
\eta_t\bar X\bar Y.
\]

局部梯度平方特例为

\[
\widehat C_{\mathrm{raw}}
=
\eta_t\bar G^2.
\]

其优点是实现最简单、无需额外前向或反向传播。缺点是同一个 batch 同时参与更新方向与贡献评价，会保留协方差偏差。

## 9.2 梯度因子双采样估计

令 \(A\) 和 \(B'\) 表示两个相互独立的 batch。为避免与 batch 大小 `B` 混淆，这里把第二个 batch 记为 \(B'\)。令 \(B_A\) 和 \(B_{B'}\) 分别表示两批数据的样本数量。

定义第一批数据上的起点平均梯度

\[
\bar X_A
=
\frac{1}{B_A}
\sum_{i\in A}X_i.
\]

定义第二批独立数据上的路径平均梯度

\[
\bar Y_{B'}
=
\frac{1}{B_{B'}}
\sum_{j\in B'}Y_j.
\]

双采样估计器定义为

\[
\widehat\Omega_D
=
\eta_t\bar X_A\bar Y_{B'}.
\]

在两批数据独立且各自均值无偏的条件下，

\[
\mathbb E[\widehat\Omega_D]
=
\eta_t\mu_X\mu_Y
=
\Omega_k.
\]

其方差为

\[
\operatorname{Var}(\widehat\Omega_D)
=
\eta_t^2
\left[
\frac{\mu_Y^2\sigma_X^2}{B_A}
+
\frac{\mu_X^2\sigma_Y^2}{B_{B'}}
+
\frac{\sigma_X^2\sigma_Y^2}{B_AB_{B'}}
\right].
\]

若固定总样本预算为 \(B\)，并平均拆成两半，即 \(B_A=B_{B'}=B/2\)，则

\[
\operatorname{Var}(\widehat\Omega_D)
=
\eta_t^2
\left[
\frac{2\mu_Y^2\sigma_X^2}{B}
+
\frac{2\mu_X^2\sigma_Y^2}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
\right].
\]

双采样无偏性直观、边界清楚，但在固定总样本预算下会因样本拆分而增加方差。

## 9.3 独立 probe 损失下降估计

本方法与上一节的“两个梯度因子双采样”不同，不能混为同一个估计器。

先由训练 batch 产生实际参数端点 \(\Theta_t\) 和 \(\Theta_{t+1}\)。再独立采样一个包含 `K` 个数据单元的 probe batch

\[
\mathcal P_K=\{z_1^{(p)},\ldots,z_K^{(p)}\}.
\]

定义 probe batch 上的损失下降估计

\[
\widehat{\Delta\mathcal L}_{K,t}
=
\mathcal L_{\mathcal P_K}(\Theta_t)
-
\mathcal L_{\mathcal P_K}(\Theta_{t+1}).
\]

在参数端点固定、probe batch 与训练更新独立的条件下，

\[
\mathbb E
\left[
\widehat{\Delta\mathcal L}_{K,t}
\mid
\Theta_t,\Theta_{t+1}
\right]
=
\mathcal L(\Theta_t)-
\mathcal L(\Theta_{t+1}).
\]

因此，它是“总损失下降”的条件无偏估计。

若已有一组原始坐标贡献 \(r_{k,t}\)，可定义损失对齐分配

\[
\widetilde C_{k,t}
=
\widehat{\Delta\mathcal L}_{K,t}
\frac{r_{k,t}}
{\sum_{j=1}^{P}r_{j,t}}.
\]

此时

\[
\sum_{k=1}^{P}\widetilde C_{k,t}
=
\widehat{\Delta\mathcal L}_{K,t}.
\]

但是，这只保证坐标贡献之和与 probe 损失下降对齐，不保证每个 \(\widetilde C_{k,t}\) 都是其真实坐标贡献的无偏估计。若分母接近零、正负抵消严重或原始相对权重错误，该缩放会非常不稳定。项目毕业论文第四章采用了类似的 probe 损失对齐思想；实现时应把它与梯度因子双采样分开命名和报告。

## 9.4 逐样本 U-statistic

令

\[
Z_i=(X_i,Y_i)
\]

表示第 \(i\) 个样本对应的一对统计量。定义二阶对称核函数

\[
h(Z_i,Z_j)
=
\frac12
\left(
X_iY_j+X_jY_i
\right).
\]

当 \(i\neq j\) 时，不同样本独立，因此

\[
\mathbb E[X_iY_j]
=
\mu_X\mu_Y.
\]

对应的逐样本 U-statistic 为

\[
\widehat\Omega_U
=
\eta_t
\frac{1}{B(B-1)}
\sum_{i\neq j}X_iY_j.
\]

定义三个流式累加量

\[
S_X=\sum_{i=1}^{B}X_i,
\qquad
S_Y=\sum_{i=1}^{B}Y_i,
\qquad
S_{XY}=\sum_{i=1}^{B}X_iY_i.
\]

则可以等价写成

\[
\widehat\Omega_U
=
\eta_t
\frac{S_XS_Y-S_{XY}}
{B(B-1)}.
\]

对于所有模型参数同时计算时，乘法应解释为逐元素乘法：

\[
\widehat\Omega_U
=
\eta_t
\frac{S_X\odot S_Y-S_{XY}}
{B(B-1)}.
\]

逐样本版本理论最直接，但逐样本梯度的显存和计算成本通常很高。

## 9.5 Microbatch-level U-statistic：一般路径版本

将一个总 batch 划分为 \(M\) 个等大小、相互独立的 microbatch，每个包含 \(b\) 个数据单元。定义第 \(m\) 个 microbatch 的起点平均梯度分量

\[
X_m^{(b)}
=
\frac1b
\sum_{i\in I_m}X_i.
\]

定义同一个 microbatch 的路径平均梯度分量

\[
Y_m^{(b)}
=
\frac1b
\sum_{i\in I_m}Y_i.
\]

Microbatch-level U-statistic 定义为

\[
\widehat\Omega_{\mathrm{MB-U}}
=
\eta_t
\frac{1}{M(M-1)}
\sum_{m\neq n}
X_m^{(b)}Y_n^{(b)}.
\]

定义流式累加量

\[
S_X=\sum_{m=1}^{M}X_m^{(b)},
\qquad
S_Y=\sum_{m=1}^{M}Y_m^{(b)},
\qquad
S_{XY}=\sum_{m=1}^{M}
X_m^{(b)}Y_m^{(b)}.
\]

则

\[
\widehat\Omega_{\mathrm{MB-U}}
=
\eta_t
\frac{S_XS_Y-S_{XY}}
{M(M-1)}.
\]

对所有参数张量同时计算时，使用逐元素乘法

\[
\widehat\Omega_{\mathrm{MB-U}}
=
\eta_t
\frac{S_X\odot S_Y-S_{XY}}
{M(M-1)}.
\]

只要不同 microbatch 是独立统计单元，并且路径在构造这些统计量时可视为固定，便有

\[
\mathbb E[
\widehat\Omega_{\mathrm{MB-U}}
]
=
\eta_t\mu_X\mu_Y.
\]

## 9.6 正式主公式：局部梯度平方 U-statistic

在局部一阶情形中，起点梯度和评价梯度相同。令 \(g_{m,k,t}\) 表示第 \(t\) 步、第 \(m\) 个 microbatch 对第 \(k\) 个参数的平均梯度。

定义逐参数的一阶累加量

\[
S_{1,k,t}
=
\sum_{m=1}^{M}g_{m,k,t}.
\]

定义逐参数的平方累加量

\[
S_{2,k,t}
=
\sum_{m=1}^{M}g_{m,k,t}^2.
\]

局部 Microbatch-level U-statistic 为

\[
\widehat C_{k,t}^{U}
=
\eta_{g(k),t}
\frac{
S_{1,k,t}^2-S_{2,k,t}
}{M(M-1)}.
\]

向量化实现按 registry 中的参数组映射，对每个坐标应用对应的
\(\eta_{g(k),t}\)；不得用一个全局标量覆盖多参数组学习率。

考虑全局梯度裁剪后，定义

\[
\widehat C_{k,t}^{U,\mathrm{clip}}
=
 \widehat s_t\widehat C_{k,t}^{U}.
\]

``local_gradient_space_importance_u`` 是固定状态下唯一继承上述 U 核心无偏性声明
的字段。``local_gradient_space_importance_u_clipped`` 是同批随机裁剪因子产生的
plug-in 在线分数，必须记录 ``clip_source=same_batch_global_mean`` 和
``unbiasedness_claim=none``；它不能再被描述为严格无偏主估计量。

需要特别注意：\(\widehat C_{k,t}^{U}\) 在有限样本下可能为负，即使目标
\(\eta_{g(k),t}\mu_{k,t}^2\) 非负。这不是代码错误，而是无偏估计器的随机波动。
若在单步层面立即执行 `clamp_min(0)`，就会重新引入正偏差。

## 9.7 不等大小 microbatch 的加权 U-statistic

若第 \(m\) 个 microbatch 的有效统计单元数为 \(b_m\)，定义总有效单元数

\[
B_{\mathrm{eff}}
=
\sum_{m=1}^{M}b_m.
\]

定义权重

\[
w_m
=
\frac{b_m}{B_{\mathrm{eff}}},
\qquad
\sum_{m=1}^{M}w_m=1.
\]

令 \(X_m\) 和 \(Y_m\) 分别表示第 \(m\) 个 microbatch 内按其有效统计单元平均得到的两个梯度量。定义加权均值

\[
\bar X_w=\sum_{m=1}^{M}w_mX_m,
\qquad
\bar Y_w=\sum_{m=1}^{M}w_mY_m.
\]

加权去对角估计器为

\[
\widehat{\mu_X\mu_Y}_{\,U,w}
=
\frac{
\bar X_w\bar Y_w
-
\sum_{m=1}^{M}w_m^2X_mY_m
}{
1-
\sum_{m=1}^{M}w_m^2
}.
\]

当 \(w_m=1/M\) 时，该公式退化为普通 Microbatch-level U-statistic。

局部平方特例为

\[
\widehat{\mu^2}_{\,U,w}
=
\frac{
\left(\sum_mw_mg_m\right)^2
-
\sum_mw_m^2g_m^2
}{
1-
\sum_mw_m^2
}.
\]

语言模型若采用全局有效 token 平均，应令 \(b_m\) 等于该 microbatch 的有效目标 token 数；若每条固定长度序列始终含有相同数量的有效目标 token，则可以使用等权公式。

加权 U 的无偏性还要求权重对被估计梯度外生（或至少满足足以推出同一目标均值
的条件），并要求参与去对角配对的统计单元具有同一目标均值。每个 artifact 必须
显式记录 ``statistical_unit``、``weight_unit``、``sampling_design``、
``weights_exogenous`` 和 ``common_mean_assumption``。有效 token 数若由 labels/mask
预先确定通常可作为外生设计量；若权重由同批 loss、gradient 或事后筛选决定，则
该结果只能标为 plug-in/描述性统计，不能自动声称无偏。

## 9.8 U-statistic 的方差

在等大小 microbatch、独立同分布和固定路径条件下，令总样本数为 \(B=Mb\)。沿用前文定义的 \(\mu_X,\mu_Y,\sigma_X^2,\sigma_Y^2,c\)。定义

\[
A
=
\mu_Y^2\sigma_X^2
+
\mu_X^2\sigma_Y^2
+
2\mu_X\mu_Yc.
\]

Microbatch-level U-statistic 的方差为

\[
\operatorname{Var}
\left(
\widehat\Omega_{\mathrm{MB-U}}
\right)
=
\eta_t^2
\left[
\frac{A}{B}
+
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}
{B^2(M-1)}
\right].
\]

对于局部平方特例，令 \(X=Y=G\)、\(\mu_X=\mu_Y=\mu\)、\(\sigma_X^2=\sigma_Y^2=c=\sigma^2\)，则

\[
\operatorname{Var}
\left(
\widehat C_U
\right)
=
\eta_t^2
\left[
\frac{4\mu^2\sigma^2}{B}
+
\frac{2M\sigma^4}{B^2(M-1)}
\right].
\]

在高斯近似且协方差已知的理想模型下，相应 Cramer-Rao 下界的一阶主项为

\[
\operatorname{CRLB}
=
\eta_t^2\frac{A}{B}.
\]

因此，Microbatch-level U-statistic 只比该下界多一个 \(O(B^{-2})\) 的有限样本项。该结论依赖高斯近似和已知协方差，只能作为效率分析，不是所有神经网络梯度分布下的无条件定理。

## 9.9 U-statistic 成立所需条件

实现和论文中必须明确以下条件：

1. 不同统计单元之间需要独立或足够接近独立。数据重复、共享随机增强、跨样本损失和跨 microbatch 状态会破坏这一条件。
2. `M` 必须至少为 2。`M=1` 时无法删除对角项。
3. 每个 \(g_m\) 必须是未经过 DDP 同步的 local mean gradient；若每次 backward 已经普通 all-reduce，则独立统计单元已经丢失。
4. 混合精度训练中必须先恢复真实梯度尺度，再累计 \(S_1\) 和 \(S_2\)。使用 GradScaler 时应在统计前 unscale。
5. BatchNorm、跨样本对比学习、in-batch negatives 等机制会让一个 microbatch 的损失依赖其他样本，必须重新定义独立统计单元或单独证明。
6. 若完整路径本身由同一组 microbatch 生成，则路径节点依赖这些 microbatch。此时一般路径版本中的 \(Y_n\) 可能通过路径间接依赖 \(X_m\)，不能未经证明便宣称严格无偏。
7. 正式局部公式是在固定 \(\Theta_t\) 下估计 \(\eta_t\mu_{k,t}^2\)，因而不受“随机路径节点依赖”这一问题的直接影响。
8. Dropout 等模型随机性可以并入随机变量 \(z\)，但不同 microbatch 的随机状态应独立。进行完整路径完备性校验时，应固定随机 mask 或关闭随机层，否则每个节点对应的不是同一个确定性损失函数。

---

## 10. 多 GPU 和梯度累积中的数学对应

令 `R` 表示 GPU 数量，令 `A` 表示每张 GPU 在一个 optimizer step 内执行的 local microbatch backward 次数。若每一次 local backward 都作为一个独立统计单元，则全局统计单元数为

\[
M=RA.
\]

令 \(g_{r,a,t}\) 表示第 \(t\) 步、第 \(r\) 张 GPU、第 \(a\) 次 local backward 得到的未同步平均梯度，其中

\[
r\in\{1,\ldots,R\},
\qquad
 a\in\{1,\ldots,A\}.
\]

每张 GPU 本地累计

\[
S_{1,t}^{(r)}
=
\sum_{a=1}^{A}g_{r,a,t},
\]

\[
S_{2,t}^{(r)}
=
\sum_{a=1}^{A}g_{r,a,t}^{\odot2}.
\]

全局 all-reduce sum 后得到

\[
S_{1,t}
=
\sum_{r=1}^{R}S_{1,t}^{(r)},
\qquad
S_{2,t}
=
\sum_{r=1}^{R}S_{2,t}^{(r)}.
\]

优化器使用的全局平均梯度应为

\[
\bar g_t
=
\frac{S_{1,t}}{M}
\]

或在不等有效权重情况下使用加权平均。

正式 U-statistic 为

\[
\widehat C_t^U
=
\eta_t
\frac{S_{1,t}^{\odot2}-S_{2,t}}
{M(M-1)}.
\]

工程上应使用 `no_sync` 或自定义梯度聚合，避免每个 local backward 立即执行普通 DDP all-reduce。

---

## 11. SGD、Momentum 与 AdamW 的关系

### 11.1 一般实际更新的局部损失贡献

对任意优化器，先定义不包含解耦权重衰减的数据驱动更新向量

\[
\Delta\Theta_t^{\mathrm{data}}.
\]

给定总体梯度 \(g^*(\Theta_t)\)，实际数据更新的一阶局部损失下降贡献定义为

\[
C_{k,t}^{\mathrm{actual-local}}
=
-
\Delta\theta_{k,t}^{\mathrm{data}}
 g_k^*(\Theta_t).
\]

如果用当前训练 batch 梯度代替总体梯度，得到 actual-update raw 诊断量

\[
\widehat C_{k,t}^{\mathrm{actual-raw}}
=
-
\Delta\theta_{k,t}^{\mathrm{data}}
 g_{\mathcal B_t,k}(\Theta_t).
\]

该量更接近实际优化器位移，但因更新量和评价梯度同源，不宣称无偏。

若使用与更新 batch 独立的 probe batch \(\mathcal P\)，可定义

\[
\widehat C_{k,t}^{\mathrm{actual-probe}}
=
-
\Delta\theta_{k,t}^{\mathrm{data}}
 g_{\mathcal P,k}(\Theta_t).
\]

条件于固定的实际更新向量，该量对一阶总体损失作用无偏。

### 11.2 Momentum SGD

令 \(\beta\in[0,1)\) 表示动量系数。令 \(v_t\) 表示第 \(t\) 步的动量状态。一个常见定义为

\[
v_t
=
\beta v_{t-1}+g_t,
\]

\[
\Delta\Theta_t^{\mathrm{data}}
=
-\eta_tv_t.
\]

此时实际局部贡献为

\[
C_{k,t}^{\mathrm{actual-local}}
=
\eta_tv_{k,t}\mu_{k,t},
\]

它不再等于 \(\eta_t\mu_{k,t}^2\)。因此，局部 U-statistic 是梯度空间分数，而不是 Momentum 实际更新贡献的严格无偏估计。

### 11.3 AdamW

令 \(\beta_1\in[0,1)\) 表示一阶矩衰减系数，令 \(\beta_2\in[0,1)\) 表示二阶矩衰减系数。令 \(m_t\) 和 \(v_t\) 分别表示 AdamW 的一阶矩与二阶矩状态。令 \(g_t\) 表示实际送入 AdamW 的全局平均梯度，通常已完成全局梯度裁剪。

定义矩状态更新

\[
m_t
=
\beta_1m_{t-1}
+(1-\beta_1)g_t,
\]

\[
v_t
=
\beta_2v_{t-1}
+(1-\beta_2)g_t^{\odot2}.
\]

若训练步从 \(t=0\) 开始计数，定义偏差修正

\[
\widehat m_t
=
\frac{m_t}{1-\beta_1^{t+1}},
\qquad
\widehat v_t
=
\frac{v_t}{1-\beta_2^{t+1}}.
\]

令 \(\varepsilon_{\mathrm{adam}}>0\) 表示 Adam 数值稳定常数。定义数据驱动更新

\[
\Delta\Theta_t^{\mathrm{data}}
=
-
\eta_t
\frac{\widehat m_t}
{\sqrt{\widehat v_t}+\varepsilon_{\mathrm{adam}}},
\]

其中除法、平方根均为逐元素运算。

令 \(\lambda_t\ge0\) 表示 decoupled weight decay 系数。定义权重衰减更新

\[
\Delta\Theta_t^{\mathrm{wd}}
=
-
\eta_t\lambda_t\Theta_t.
\]

总更新近似写为

\[
\Delta\Theta_t
=
\Delta\Theta_t^{\mathrm{data}}
+
\Delta\Theta_t^{\mathrm{wd}}.
\]

本研究的“数据损失贡献”主分析应排除 \(\Delta\Theta_t^{\mathrm{wd}}\)，因为 decoupled weight decay 是正则化位移，不是当前数据梯度直接产生的位移。若需要研究“整个优化器总位移对数据损失的作用”，必须另设指标并明确包含权重衰减。

### 11.4 AdamW 下主指标的解释边界

正式 U-statistic 仍计算

\[
\eta_ts_t\mu_{k,t}^2
\]

的无偏估计，它度量当前梯度空间中的稳定学习信号。AdamW 实际位移还受动量、二阶矩预条件和历史状态影响。因此应同时保存：

- 主指标：U-statistic gradient-space importance；
- 辅助指标：actual-update raw 或 independent-probe importance；
- 基线：参数移动量、最终参数幅值和原始同批梯度平方。

---

## 12. 训练过程中的累计重要性

令 \(\widehat C_{k,t}\) 表示所选择估计器在第 \(t\) 步给出的第 \(k\) 个参数贡献。

### 12.1 Signed 累计

定义 signed 累计贡献

\[
\Omega_k^{\mathrm{signed}}
=
\sum_{t=0}^{T-1}\widehat C_{k,t}.
\]

该量保留正负抵消，适合研究参数的净损失作用。

### 12.2 正贡献累计

定义正部函数

\[
[x]_+=\max(x,0).
\]

定义正贡献累计

\[
\Omega_k^+
=
\sum_{t=0}^{T-1}
[\widehat C_{k,t}]_+.
\]

该量便于构造非负分布和剪枝排序，但由于经过非线性截断，不再保持单步 U-statistic 的无偏性。

### 12.3 负贡献质量

定义负贡献的非负质量

\[
\Omega_k^-
=
\sum_{t=0}^{T-1}
[-\widehat C_{k,t}]_+.
\]

于是

\[
\Omega_k^{\mathrm{signed}}
=
\Omega_k^+-\Omega_k^-.
\]

### 12.4 绝对活动量

定义绝对贡献累计

\[
\Omega_k^{\mathrm{abs}}
=
\sum_{t=0}^{T-1}
|\widehat C_{k,t}|.
\]

它反映参数在损失作用上的总活动强度，不区分促进或阻碍方向。

### 12.5 分阶段增量

令 \(t_a<t_b\) 表示两个 checkpoint 对应的训练步。定义该阶段的重要性增量

\[
\Delta\Omega_k[t_a,t_b]
=
\sum_{t=t_a}^{t_b-1}\widehat C_{k,t}.
\]

保存阶段增量可以分析重要性在训练早期、中期和后期的形成，而不必只观察最终累计值。

---

## 13. 归一化、层级聚合与分布统计

### 13.1 非负参数重要性分布

令 \(a_k\ge0\) 表示用于分布分析的非负参数分数，例如 \(\Omega_k^+\) 或 \(\Omega_k^{\mathrm{abs}}\)。定义稳定常数 \(\varepsilon_{\mathrm{norm}}>0\)。定义归一化质量

\[
p_k
=
\frac{a_k}
{\sum_{j=1}^{P}a_j+\varepsilon_{\mathrm{norm}}}.
\]

当总质量明显非零时，近似有

\[
\sum_{k=1}^{P}p_k=1.
\]

signed 分数不应直接按概率分布处理。若需要归一化 signed 分数，定义

\[
q_k
=
\frac{\Omega_k^{\mathrm{signed}}}
{\sum_{j=1}^{P}|\Omega_j^{\mathrm{signed}}|+\varepsilon_{\mathrm{norm}}}.
\]

### 13.2 层和模块聚合

令 \(\mathcal K_\ell\) 表示第 \(\ell\) 个层或模块中包含的参数坐标集合。令

\[
P_\ell=|\mathcal K_\ell|
\]

表示该层参数数量。

定义层总重要性

\[
A_\ell
=
\sum_{k\in\mathcal K_\ell}a_k.
\]

定义层平均每参数重要性

\[
\bar A_\ell
=
\frac{A_\ell}{P_\ell}.
\]

定义层占全模型的重要性质量比例

\[
\pi_\ell
=
\frac{A_\ell}
{\sum_jA_j+\varepsilon_{\mathrm{norm}}}.
\]

比较不同参数量的层时，必须同时报告 \(A_\ell\)、\(\bar A_\ell\) 和 \(\pi_\ell\)，否则大层可能仅因参数数量多而获得更高总质量。

### 13.3 均值、方差与离散系数

定义非负分数的参数总体均值

\[
\bar a
=
\frac1P\sum_{k=1}^{P}a_k.
\]

定义参数总体方差

\[
V_a
=
\frac1P
\sum_{k=1}^{P}(a_k-\bar a)^2.
\]

定义标准差

\[
S_a=\sqrt{V_a}.
\]

定义离散系数

\[
\operatorname{CV}(a)
=
\frac{S_a}{|\bar a|+\varepsilon_{\mathrm{norm}}}.
\]

CV 对接近零的均值非常敏感，不适合直接用于 signed 分数。

### 13.4 Gini 系数

将非负分数按升序排列为

\[
a_{(1)}\le a_{(2)}\le\cdots\le a_{(P)}.
\]

定义 Gini 系数

\[
G
=
\frac{2\sum_{i=1}^{P}i\,a_{(i)}}
{P\sum_{i=1}^{P}a_{(i)}+\varepsilon_{\mathrm{norm}}}
-
\frac{P+1}{P}.
\]

\(G\) 越接近 0，分布越均匀；越接近 1，质量越集中于少数参数。

### 13.5 Shannon 熵

使用归一化质量 \(p_k\)，定义 Shannon 熵

\[
H
=
-
\sum_{k=1}^{P}
 p_k\log(p_k+\varepsilon_{\mathrm{norm}}).
\]

定义归一化熵

\[
H_{\mathrm{norm}}
=
\frac{H}{\log P}.
\]

\(H_{\mathrm{norm}}\) 越接近 1，分布越均匀。

### 13.6 HHI 与有效参数数量

定义 Herfindahl-Hirschman Index

\[
\operatorname{HHI}
=
\sum_{k=1}^{P}p_k^2.
\]

定义有效参数数量

\[
N_{\mathrm{eff}}
=
\frac{1}
{\operatorname{HHI}+\varepsilon_{\mathrm{norm}}}.
\]

若所有参数质量完全均匀，则 \(N_{\mathrm{eff}}\approx P\)；若质量集中于少数参数，则 \(N_{\mathrm{eff}}\) 显著减小。

### 13.7 Top-q 累计质量

将非负分数按降序排列为

\[
a_{[1]}\ge a_{[2]}\ge\cdots\ge a_{[P]}.
\]

令 \(q\in(0,1]\) 表示参数比例，定义

\[
K_q=\lceil qP\rceil.
\]

定义 top-q 累计质量

\[
M_{\mathrm{top}}(q)
=
\frac{
\sum_{i=1}^{K_q}a_{[i]}
}{
\sum_{i=1}^{P}a_{[i]}+\varepsilon_{\mathrm{norm}}
}.
\]

若很小的 \(q\) 已包含很大的质量，说明重要性高度集中。

---

## 14. 估计器验证所需统计量

### 14.1 大样本参考目标

固定参数状态 \(\Theta_t\)。令 \(\mathcal B_{\mathrm{ref}}\) 表示足够大的独立参考 batch，其大小为 \(B_{\mathrm{ref}}\)。定义参考平均梯度

\[
\mu_{k,t}^{\mathrm{ref}}
=
\frac{1}{B_{\mathrm{ref}}}
\sum_{z_i\in\mathcal B_{\mathrm{ref}}}
\frac{\partial\ell(\Theta_t;z_i)}{
\partial\theta_k
}.
\]

定义局部参考贡献

\[
C_{k,t}^{\mathrm{ref}}
=
\eta_t
\left(
\mu_{k,t}^{\mathrm{ref}}
\right)^2.
\]

该量只是有限大样本近似真值；应通过增大 \(B_{\mathrm{ref}}\) 检查其稳定性。

### 14.2 重复抽样统计

令 `R` 表示同一固定参数状态下独立重复估计的次数。令

\[
\widehat C_{k}^{(r)}
\]

表示第 \(r\) 次重复得到的第 \(k\) 个参数估计值，其中 \(r\in\{1,\ldots,R\}\)。

定义重复均值

\[
\overline C_k
=
\frac1R
\sum_{r=1}^{R}
\widehat C_k^{(r)}.
\]

定义经验偏差

\[
\widehat{\operatorname{Bias}}_k
=
\overline C_k-C_k^{\mathrm{ref}}.
\]

定义带稳定常数 \(\varepsilon_{\mathrm{stat}}>0\) 的相对偏差

\[
\widehat{\operatorname{RelBias}}_k
=
\frac{
\overline C_k-C_k^{\mathrm{ref}}
}{
|C_k^{\mathrm{ref}}|+\varepsilon_{\mathrm{stat}}
}.
\]

定义样本方差

\[
\widehat{\operatorname{Var}}_k
=
\frac1{R-1}
\sum_{r=1}^{R}
\left(
\widehat C_k^{(r)}-\overline C_k
\right)^2.
\]

定义均方误差

\[
\widehat{\operatorname{MSE}}_k
=
\frac1R
\sum_{r=1}^{R}
\left(
\widehat C_k^{(r)}-C_k^{\mathrm{ref}}
\right)^2.
\]

定义平均绝对误差

\[
\widehat{\operatorname{MAE}}_k
=
\frac1R
\sum_{r=1}^{R}
\left|
\widehat C_k^{(r)}-C_k^{\mathrm{ref}}
\right|.
\]

理论上有

\[
\operatorname{MSE}
=
\operatorname{Var}
+
\operatorname{Bias}^2.
\]

因此，无偏不是唯一目标；还必须同时比较方差和 MSE。

### 14.3 Pearson 相关

给定两个参数分数向量 \(a=(a_1,\ldots,a_P)\) 和 \(b=(b_1,\ldots,b_P)\)。定义它们的均值

\[
\bar a=\frac1P\sum_{k=1}^{P}a_k,
\qquad
\bar b=\frac1P\sum_{k=1}^{P}b_k.
\]

Pearson 相关系数为

\[
r_{\mathrm{Pearson}}
=
\frac{
\sum_{k=1}^{P}(a_k-\bar a)(b_k-\bar b)
}{
\sqrt{\sum_{k=1}^{P}(a_k-\bar a)^2}
\sqrt{\sum_{k=1}^{P}(b_k-\bar b)^2}
}.
\]

### 14.4 Spearman 排名相关

令 \(R_k(a)\) 表示 \(a_k\) 在向量 \(a\) 中的秩，令 \(R_k(b)\) 表示 \(b_k\) 在向量 \(b\) 中的秩。Spearman 相关定义为秩向量之间的 Pearson 相关：

\[
r_{\mathrm{Spearman}}
=
\operatorname{Corr}
\left(
R(a),R(b)
\right).
\]

有并列值时应使用平均秩，不要直接使用任意稳定排序产生的整数秩。

### 14.5 Top-k overlap 与 Jaccard

令 \(\mathcal T_K(a)\) 表示向量 \(a\) 中分数最大的 `K` 个参数坐标集合。定义 overlap ratio

\[
\operatorname{Overlap@K}(a,b)
=
\frac{
|\mathcal T_K(a)\cap\mathcal T_K(b)|
}{K}.
\]

定义 Jaccard 指数

\[
J_K(a,b)
=
\frac{
|\mathcal T_K(a)\cap\mathcal T_K(b)|
}{
|\mathcal T_K(a)\cup\mathcal T_K(b)|
}.
\]

### 14.6 多随机种子置信区间

令 \(x_1,\ldots,x_R\) 表示 `R` 个随机种子得到的某个标量指标。定义均值 \(\bar x\) 和样本标准差 \(s_x\)。在近似正态且样本独立时，双侧 95% t 区间为

\[
\bar x
\pm
 t_{R-1,0.975}
\frac{s_x}{\sqrt R},
\]

其中 \(t_{R-1,0.975}\) 表示自由度为 \(R-1\) 的 t 分布 97.5% 分位数。随机种子较少时应同时展示所有种子散点，不应只报告区间或 p 值。

---

## 15. 剪枝功能验证的数学定义

令 \(a_k\) 表示用于剪枝排序的参数分数。令 \(\rho\in[0,1]\) 表示剪枝比例。定义要剪除的参数数量

\[
K_\rho=\lfloor\rho P_{\mathrm{eligible}}\rfloor,
\]

其中 \(P_{\mathrm{eligible}}\) 表示纳入本次剪枝的参数数量。

令 \(m_k\in\{0,1\}\) 表示参数保留 mask。剪枝后的参数定义为

\[
\theta_k^{\mathrm{pruned}}
=
 m_k\theta_k.
\]

高重要性剪枝选择分数最大的 \(K_\rho\) 个参数令 \(m_k=0\)。低重要性剪枝选择分数最小的 \(K_\rho\) 个参数令 \(m_k=0\)。随机剪枝在 eligible set 中均匀随机选择相同数量参数。

若评价指标 \(M\) 越大越好，例如 accuracy，定义性能损伤

\[
D_M(\rho)
=
M(0)-M(\rho).
\]

若评价指标 \(L\) 越小越好，例如 loss 或 perplexity，定义性能损伤

\[
D_L(\rho)
=
L(\rho)-L(0).
\]

理想的重要性排序应满足：在相同剪枝比例下，高重要性剪枝损伤最大，低重要性剪枝损伤最小，随机剪枝位于二者之间；并且这一关系应优于参数幅值、移动量和 raw 梯度平方等简单基线。

若剪枝比例网格为

\[
0=\rho_0<\rho_1<\cdots<\rho_J,
\]

可用梯形法计算损伤曲线面积

\[
\operatorname{AUC}_D
=
\sum_{j=0}^{J-1}
\frac{D(\rho_j)+D(\rho_{j+1})}{2}
(\rho_{j+1}-\rho_j).
\]

随机剪枝必须对多个随机 mask 重复，并保存每个 mask 的独立结果。

---

## 16. 必须实现的比较基线

### 16.1 最终参数幅值

定义参数幅值分数

\[
A_k^{\mathrm{mag}}
=
|\theta_{k,T}|.
\]

### 16.2 参数移动量

定义累计绝对移动量

\[
A_k^{\mathrm{move}}
=
\sum_{t=0}^{T-1}
|\Delta\theta_{k,t}^{\mathrm{data}}|.
\]

也可额外保存首尾净位移

\[
A_k^{\mathrm{netmove}}
=
|\theta_{k,T}-\theta_{k,0}|.
\]

### 16.3 原始同批梯度平方

定义 raw 累计分数

\[
A_k^{\mathrm{raw}}
=
\sum_{t=0}^{T-1}
\eta_t
\bar g_{k,t}^2.
\]

该基线包含同批梯度方差偏差，是验证 U-statistic 是否有增益的关键对照。

### 16.4 经验 Fisher 对角线

给定评价数据集 \(\mathcal E=\{z_1,\ldots,z_{N_E}\}\)，定义经验 Fisher 对角近似

\[
A_k^{\mathrm{Fisher}}
=
\frac1{N_E}
\sum_{i=1}^{N_E}
\left[
\frac{\partial\ell(\Theta_T;z_i)}
{\partial\theta_k}
\right]^2.
\]

该量是训练完成后的静态敏感性基线，不包含整个训练轨迹信息。

### 16.5 Synaptic Intelligence 基线

定义 SI 的未归一化路径贡献

\[
\omega_k^{\mathrm{SI}}
=
-
\sum_{t=0}^{T-1}
 g_{k,t}\Delta\theta_{k,t}.
\]

令

\[
\Delta_k^{\mathrm{task}}
=
\theta_{k,T}-\theta_{k,0}
\]

表示整个任务期间的净参数变化。令 \(\xi>0\) 表示稳定常数。定义 SI 重要性

\[
\Omega_k^{\mathrm{SI}}
=
\frac{
[\omega_k^{\mathrm{SI}}]_+
}{
(\Delta_k^{\mathrm{task}})^2+\xi
}.
\]

SI 的原始目的主要是连续学习正则化；本研究的主指标更关注训练损失贡献和不同训练范式的参数结构。二者可以作为相关但不等价的路径方法进行比较。

---

## 17. 可选扩展：EMA 路径方向贡献

本节记录项目毕业论文第四章提出的路径方向思想，但不建议纳入第一版主实现。

令

\[
d_t=\Delta\Theta_t^{\mathrm{data}}
\]

表示第 \(t\) 步的数据驱动更新方向。令 \(\rho\in[0,1)\) 表示 EMA 平滑系数。定义 EMA 状态

\[
e_t
=
\rho e_{t-1}
+(1-\rho)d_t,
\qquad e_{-1}=0.
\]

若需要修正初始零值偏差，定义

\[
\widehat e_t
=
\frac{e_t}{1-\rho^{t+1}}.
\]

项目毕业论文使用的投影型路径贡献近似为

\[
p_t^{\mathrm{proj}}
=
\frac{
\langle e_{t-1},d_t\rangle
}{
\|e_{t-1}\|_2
}.
\]

该量具有“参数位移”的量纲，不是无量纲量。为了提高跨模型和跨学习率的可比性，更稳妥的方向一致性指标是余弦相似度

\[
p_t^{\mathrm{cos}}
=
\frac{
\langle \widehat e_{t-1},d_t\rangle
}{
\|\widehat e_{t-1}\|_2
\|d_t\|_2
+
\varepsilon_{\mathrm{path}}
},
\]

其中 \(\varepsilon_{\mathrm{path}}>0\) 为稳定常数，且 \(p_t^{\mathrm{cos}}\in[-1,1]\)。

毕业论文还提出了将路径贡献和参数损失贡献通过指数形式组合的方案。该方案存在量纲、底数正值、跨步尺度以及超参数解释问题。当前实验计划只研究损失贡献，因此建议第一版代码仅保存 \(d_t\)、\(e_t\)、\(p_t^{\mathrm{cos}}\) 作为独立诊断量，不把它们直接乘入或指数化到参数重要性中。任何组合公式都应在单独消融实验中验证。

另一个需要注意的事实是：若 \(p_t\) 是单个全局标量，则它会对同一步所有参数施加相同时间权重，只改变跨步加权，不改变同一步内部的参数排序。

---

## 18. 三个代码级核心算法

## 18.1 算法 A：正式训练中的局部 U-statistic

输入包括：当前模型参数 \(\Theta_t\)、坐标到参数组的静态映射与各组动态学习率
\(\eta_{g,t}\)、全局梯度裁剪阈值 \(G_{\max}\)、全局 microbatch 数量 \(M\)，
以及每个 microbatch 的未同步平均梯度。

数学流程如下：

1. 对每个 microbatch 计算未同步、已恢复真实尺度的 FP32 平均梯度 \(g_m\)。
2. 流式累计

   \[
   S_{1,t}\leftarrow S_{1,t}+g_m,
   \]

   \[
   S_{2,t}\leftarrow S_{2,t}+g_m^{\odot2}.
   \]

3. 跨 GPU 对 \(S_{1,t}\)、\(S_{2,t}\) 和统计单元数量做 all-reduce sum。
4. 设置优化器平均梯度

   \[
   \bar g_t=\frac{S_{1,t}}{M}.
   \]

5. 根据 \(\bar g_t\) 计算全局裁剪因子 \(s_t\)。
6. 分别计算未裁剪 U 分数与同批 clip plug-in 分数

   \[
   \widehat C_t^{U,\mathrm{clip}}
   =
   \eta_{g(k),t}\widehat s_t
   \frac{S_{1,t}^{\odot2}-S_{2,t}}
   {M(M-1)}.
   \]

7. 累计 signed、positive、negative mass 和 absolute 四类统计，并保存未裁剪与
   plug-in 两个命名空间，禁止覆盖。
8. 将裁剪后的平均梯度交给优化器执行 AdamW 更新。
9. 记录不含权重衰减的数据更新量，并计算 actual-update 诊断量。

最简伪代码：

```python
S1 = zeros_like_parameters(dtype=float32)
S2 = zeros_like_parameters(dtype=float32)
M_local = 0

for microbatch in local_microbatches:
    zero_temporary_grads()
    loss = mean_loss(microbatch)
    backward_without_ddp_sync(loss)
    g = read_unscaled_local_mean_grads_as_fp32()
    S1 += g
    S2 += g.square()
    M_local += 1

S1 = all_reduce_sum(S1)
S2 = all_reduce_sum(S2)
M = all_reduce_sum(M_local)

assert M >= 2
mean_grad = S1 / M
set_optimizer_grads(mean_grad)
clip_factor = compute_global_clip_factor_and_clip(mean_grad)

step_u = learning_rate_by_parameter_group * (
    S1.square() - S2
) / (M * (M - 1))
step_u_clipped_plugin = clip_factor * step_u

omega_signed += step_u
omega_positive += step_u.clamp_min(0)
omega_negative_mass += (-step_u).clamp_min(0)
omega_absolute += step_u.abs()

optimizer_step()
```

## 18.2 算法 B：固定实际路径的多节点 probe 积分

输入包括：更新前参数 \(\Theta_t\)、更新后参数 \(\Theta_{t+1}\)、独立固定 probe batch \(\mathcal P\)、节点 \(\alpha_q\) 和权重 \(a_q\)。

流程如下：

1. 计算更新向量

   \[
   \Delta\Theta_t=\Theta_{t+1}-\Theta_t.
   \]

2. 初始化路径平均梯度张量为零。
3. 对每个求积节点构造

   \[
   \Theta_{t,q}
   =
   \Theta_t+\alpha_q\Delta\Theta_t.
   \]

4. 在同一个 probe batch 和固定模型随机状态下计算

   \[
   g_{\mathcal P}(\Theta_{t,q}).
   \]

5. 累计

   \[
   \widehat{\bar g}_t
   =
   \sum_q a_qg_{\mathcal P}(\Theta_{t,q}).
   \]

6. 得到逐参数路径贡献

   \[
   \widehat C_t^{\mathrm{path,probe}}
   =
   -
   \Delta\Theta_t\odot
   \widehat{\bar g}_t.
   \]

7. 计算完备性残差

   \[
   \left|
   \sum_k\widehat C_{k,t}^{\mathrm{path,probe}}
   -
   \left[
   \mathcal L_{\mathcal P}(\Theta_t)-
   \mathcal L_{\mathcal P}(\Theta_{t+1})
   \right]
   \right|.
   \]

8. 恢复原模型参数和 optimizer state。路径校验过程不得改变训练状态。

## 18.3 算法 C：固定状态估计器偏差验证

输入包括：固定 checkpoint \(\Theta_t\)、学习率标量 \(\eta_t\)、大参考 batch 和重复次数 `R`。

流程如下：

1. 使用大参考 batch 计算 \(C^{\mathrm{ref}}\)。
2. 重复 `R` 次独立抽样。
3. 每次在完全相同的固定参数状态下计算 raw、双采样和 U-statistic。
4. 不执行 optimizer step，不让模型参数或 optimizer state 改变。
5. 统计 Bias、Variance、MSE、MAE、Spearman 和 top-k overlap。
6. 改变 batch size 与 microbatch 数量，重复整个过程。

这一实验用于隔离估计器统计性质，不能与模型训练动态混在同一个实验中判断。

---

## 19. 必须通过的数学与代码不变量

### 19.1 U-statistic 代数恒等式

对于任意 microbatch 梯度 \(g_1,\ldots,g_M\)，逐元素恒有

\[
\left(\sum_{m=1}^{M}g_m\right)^{\odot2}
-
\sum_{m=1}^{M}g_m^{\odot2}
=
\sum_{m\neq n}g_m\odot g_n.
\]

也可以写成

\[
\left(\sum_mg_m\right)^{\odot2}
-
\sum_mg_m^{\odot2}
=
2\sum_{m<n}g_m\odot g_n.
\]

代码单元测试应直接验证两种实现一致。

### 19.2 所有 microbatch 梯度相同时

若对某个参数坐标有

\[
g_{1,k}=g_{2,k}=\cdots=g_{M,k}=g_k,
\]

则

\[
\frac{
\left(\sum_mg_{m,k}\right)^2
-
\sum_mg_{m,k}^2
}{M(M-1)}
=
 g_k^2.
\]

### 19.3 零均值纯噪声测试

若 \(\mu_k=0\)，则理论上

\[
\mathbb E[\widehat C_{k}^{U}]=0,
\]

而 raw 估计满足

\[
\mathbb E[\widehat C_{k}^{\mathrm{raw}}]
=
\eta\frac{\sigma_k^2}{B}>0.
\]

可用人工高斯梯度模拟验证。

### 19.4 Microbatch 排列不变性

任意交换 microbatch 顺序，\(S_1\)、\(S_2\) 和 U-statistic 均不得改变。

### 19.5 平均梯度一致性

在等大小 microbatch 下，必须满足

\[
\frac1M\sum_{m=1}^{M}g_m
=
\frac1B\sum_{i=1}^{B}g(\Theta;z_i).
\]

单 GPU、四 GPU、不同梯度累积方式在相同 global batch 和相同随机状态下应给出相同平均梯度，误差只允许来自预先设定的浮点容差。

### 19.6 常梯度路径测试

若人工构造的损失在路径上具有常梯度，则左端点、中点、梯形、Simpson 和 Gauss-Legendre 应给出相同积分结果。

### 19.7 二次损失路径测试

对于可解析的二次损失，可计算真实端点损失差。梯形法应对沿线性路径得到的线性梯度积分精确，代码完备性残差应接近机器精度。

### 19.8 断点恢复一致性

保存并恢复模型、optimizer、scheduler、随机数状态、数据游标和累计重要性后，后续每一步的 \(S_1\)、\(S_2\)、U-statistic 和参数更新应与不中断运行一致。

---

## 20. 推荐保存的数学状态

正式训练中，每个参数张量至少长期保存以下同形状数组：

- \(\Omega^{\mathrm{signed}}\)；
- \(\Omega^+\)；
- \(\Omega^-\)；
- \(\Omega^{\mathrm{abs}}\)；
- raw 累计分数；
- 参数累计移动量；
- 必要时的 actual-update 累计诊断。

每个 optimizer step 只需要临时保存 \(S_1\) 和 \(S_2\)，计算完单步 U-statistic 后即可释放。关键 checkpoint 应保存参数级数组；普通日志步只保存层级和模块级聚合。

每一步还应保存用于解释数学尺度的标量：学习率 \(\eta_t\)、全局 microbatch 数 \(M\)、有效样本或 token 数、梯度范数、裁剪因子 \(s_t\)、参数更新范数和权重衰减系数。

---

## 21. 实现时最容易混淆的边界

1. 标准 Integrated Gradients 解释输入特征；本研究的主对象是参数空间中的损失贡献，应称为参数路径积分贡献或局部梯度空间贡献。
2. 路径积分的精确完备性只对同一个固定标量损失成立。
3. Minibatch 梯度本身在 i.i.d. 条件下可以是无偏的；偏差来自两个相关随机梯度因子的乘积，而不是“随机梯度均值一定有偏”。
4. 梯度因子双采样对 \(\eta\mu_X\mu_Y\) 无偏；独立 probe 损失方法对总损失下降条件无偏。二者目标不同。
5. Probe 损失对齐后的逐参数分配不自动获得逐参数无偏性。
6. Microbatch U-statistic 的一般路径无偏性要求路径固定或与评价统计单元独立；正式主公式只声称对固定状态下的局部梯度平方目标无偏。
7. AdamW 的真实更新含动量、预条件和权重衰减。局部 U-statistic 是 gradient-space importance，不是完整 AdamW 路径积分。
8. 全局梯度裁剪因子在局部贡献公式中只乘一次；若因子来自同一随机 batch，所得
   clipped 字段是 plug-in 在线分数，不继承未裁剪 U 的严格无偏性。
9. 单步 U-statistic 可能为负；先截断再累计会引入偏差。
10. signed、positive、negative mass 和 absolute 四类累计量回答不同问题，且固定
    满足 ``signed = positive - negative_mass``、
    ``absolute = positive + negative_mass``，不能只保存其中一种。
11. 归一化只能解决尺度可比性，不能修复错误的估计器或错误的参数排序。
12. 层总重要性和层平均每参数重要性必须同时报告。
13. 数值积分总和误差很小，不代表参数级归因一定准确；必须同时比较排名和 top-k 集合。
14. 对比学习、BatchNorm、in-batch negatives 和共享随机增强会破坏简单独立假设，必须单独处理。
15. 任何超出本文件主定义的路径平滑组合，都应作为独立消融，而不能静默替换主指标。

---

## 22. 推荐的最终方法命名

为避免论文和代码中的概念混淆，建议使用以下命名：

- `path_integrated_loss_contribution`：固定损失、固定端点、沿参数线段进行多节点数值积分得到的完整路径贡献；
- `local_gradient_space_importance_raw`：同批次平均梯度平方累计；
- `local_gradient_space_importance_u`：Microbatch-level U-statistic 去偏的局部梯度空间贡献；
- `double_sample_gradient_importance`：两个独立 batch 分别提供更新梯度因子和评价梯度因子；
- `independent_probe_loss_drop`：独立 probe batch 计算的更新前后总损失下降；
- `probe_aligned_coordinate_score`：用 probe 总损失下降缩放原始坐标权重后的分配；
- `actual_update_raw_importance`：AdamW 数据驱动实际位移与同批梯度的局部乘积；
- `actual_update_probe_importance`：AdamW 数据驱动实际位移与独立 probe 梯度的局部乘积；
- `ema_path_alignment`：单独保存的 EMA 路径方向一致性诊断。

---

## 23. 项目资料依据与修订说明

本数学规格主要参考以下项目资料：

1. 《基于可学习参数重要性估计的神经网络可解释性研究》，重点参考第二章积分梯度基础、第三章参数空间路径积分与数值积分、第四章随机梯度过估计和路径贡献。
2. `parameter_importance_u_statistic_report.md`，重点参考协方差偏差、双采样、逐样本 U-statistic、Microbatch-level U-statistic、方差与 Cramer-Rao 效率分析。
3. `NLP参数重要性完整实验计划书.md`，重点参考正式实验的局部目标、AdamW 边界、梯度裁剪、DDP microbatch 聚合以及 signed/positive/negative 累计。
4. 导师会议纪要，重点参考“先验证重要性，再用剪枝检验功能意义，最后解释训练范式差异”的研究闭环。

为了便于直接编写代码，本文件对资料中的符号进行了统一，并作出以下数学区分：

- 将“正值表示损失下降”作为唯一符号约定；
- 将输入空间 IG 与参数空间路径积分明确分开；
- 将同源噪声偏差写成一般协方差，而非只写成方差；
- 将梯度因子双采样与独立 probe 损失对齐明确分开；
- 使用标准 Newton-Cotes 和 Simpson 数值积分权重及误差阶；
- 将 U-statistic 的严格结论限制在固定状态或固定路径目标上；
- 将 AdamW 实际更新贡献与梯度空间主指标明确分开；
- 将 EMA 路径贡献保留为可选诊断，不纳入第一版主指标。

---

## 24. 一页式主公式汇总

完整参数路径贡献：

\[
C_{k,t}^{\mathrm{path}}
=
-
\Delta\theta_{k,t}
\int_0^1
 g_k^*(\Theta_t+\alpha\Delta\Theta_t)
\,d\alpha.
\]

数值求积：

\[
\widehat C_{k,t}^{(Q)}
=
-
\Delta\theta_{k,t}
\sum_{q=1}^{Q}a_q
 g_k(\Theta_t+\alpha_q\Delta\Theta_t).
\]

局部梯度空间目标：

\[
C_{k,t}^{\mathrm{grad}}
=
\eta_t\mu_{k,t}^2.
\]

原始同批估计：

\[
\widehat C_{k,t}^{\mathrm{raw}}
=
\eta_t\bar g_{k,t}^2.
\]

其局部正偏差：

\[
\operatorname{Bias}
\left(
\widehat C_{k,t}^{\mathrm{raw}}
\right)
=
\eta_t\frac{\sigma_{k,t}^2}{B}.
\]

梯度因子双采样：

\[
\widehat C_{k,t}^{D}
=
\eta_t
\bar g_{A,k,t}
\bar g_{B',k,t}.
\]

Microbatch-level U-statistic 主公式：

\[
\widehat C_{k,t}^{U}
=
\eta_t
\frac{
\left(\sum_{m=1}^{M}g_{m,k,t}\right)^2
-
\sum_{m=1}^{M}g_{m,k,t}^2
}{M(M-1)}.
\]

考虑全局梯度裁剪：

\[
\widehat C_{k,t}^{U,\mathrm{clip}}
=
 s_t\widehat C_{k,t}^{U}.
\]

累计 signed 重要性：

\[
\Omega_k^{\mathrm{signed}}
=
\sum_{t=0}^{T-1}
\widehat C_{k,t}^{U,\mathrm{clip}}.
\]

累计正贡献：

\[
\Omega_k^+
=
\sum_{t=0}^{T-1}
\max
\left(
0,
\widehat C_{k,t}^{U,\mathrm{clip}}
\right).
\]

累计负贡献质量：

\[
\Omega_k^-
=
\sum_{t=0}^{T-1}
\max
\left(
0,
-
\widehat C_{k,t}^{U,\mathrm{clip}}
\right).
\]

非负归一化质量：

\[
p_k
=
\frac{\Omega_k^+}
{\sum_j\Omega_j^++\varepsilon}.
\]

有效参数数量：

\[
N_{\mathrm{eff}}
=
\frac{1}
{\sum_kp_k^2+\varepsilon}.
\]
