# 参数积分梯度重要性的噪声偏差、双采样估计与 Microbatch-level U-statistic 估计

## 摘要

本文整理一个关于参数积分梯度重要性估计的完整推导。我们关心的问题是：当参数重要性由沿参数更新路径的梯度积分定义，而实际梯度只能由 minibatch 估计时，随机梯度噪声会怎样影响重要性估计？

最初的直觉是，若同一个 minibatch 同时用于决定参数更新方向和计算参数重要性，那么更新方向中的噪声会和重要性计算中的噪声相乘，从而产生一个正的均值偏差。若只在更新点附近做局部近似，这个偏差看起来像是一个方差项；但在真正的多节点路径积分中，梯度噪声会随路径位置变化，此时偏差的正确形式不是单纯的方差，而是“更新点噪声”和“积分节点噪声”之间的协方差。

在此基础上，我们讨论两类无偏估计方法。第一类是双采样法：用一个 minibatch 决定更新方向，用另一个独立 minibatch 计算路径积分梯度。它能够消除同源噪声造成的均值偏差，但由于把样本预算拆成两份，其方差通常较大。第二类是单采样的 U-statistic 方法：在同一个 batch 或同一组 microbatch 内，只使用不同样本单元之间的交叉乘积，排除同一个样本单元的自乘项，从而在不额外采样的情况下构造无偏估计。由于逐样本梯度在工程中较难获得，本文重点推导 microbatch-level U-statistic，即把每个 microbatch 或每张卡上的 local gradient 当作一个统计单元来构造估计器。

最后，本文在高斯近似和协方差已知的正则模型下推导 Cramer-Rao 方差下界，说明 microbatch-level U-statistic 的方差达到该下界的一阶主项，只比下界多一个二阶项；而双采样法虽然也无偏，但在相同总样本预算下通常离下界更远。

## 1. 基本记号与目标量

固定一个参数坐标 $k$。为避免记号过重，下面所有梯度都默认指第 $k$ 个参数分量。

令当前更新点为

$$
u = \theta_t.$$

沿着一次参数更新的路径，积分节点写成

$$v_\alpha = \theta_t + \alpha \Delta\theta, \qquad \alpha\in[0,1].$$

如果使用数值积分，例如梯形法、辛普森法或高斯积分法，则连续积分可以近似为若干个节点的加权和：

$$\int_0^1 h(v_\alpha)d\alpha \approx \sum_{q=1}^{Q} a_q h(v_q), \qquad v_q=\theta_t+\alpha_q\Delta\theta.$$

这里 $a_q$ 是积分权重。通常有 $\sum_q a_q=1$，但后面的推导不依赖它的具体形式。

对单个样本 $z_i$，定义它在更新点处的梯度为

$$X_i = \nabla_{\theta_k}L(z_i,\nu).$$

再定义它在路径积分上的加权梯度为

$$Y_i = \sum_{q=1}^{Q}a_q\nabla_{\theta_k}L(z_i,v_q).$$

于是，全数据或总体意义下的两个均值为

$$\mu_X=\mathbb E[X_i]=g_k^*(\nu),$$

$$\mu_Y=\mathbb E[Y_i]=\sum_{q=1}^{Q}a_qg_k^*(v_q).$$

在本文中，我们把一步的理想参数积分梯度重要性写成

$$\Omega_k = \gamma\mu_X\mu_Y.$$

这里 $\gamma$ 是学习率。为了不让符号干扰主要问题，本文吸收了 SGD 更新方向中的负号。若使用标准写法 $\Delta\theta=-\gamma g$，只需在重要性定义中统一处理符号，不影响无偏性与方差推导。

这个目标量可以理解为：用全数据梯度决定更新方向 $\gamma\mu_X$，再乘以全数据路径积分梯度 $\mu_Y$。我们希望用随机 minibatch 或 microbatch 梯度来估计它。

设单样本层面的方差和协方差为

$$\sigma_X^2=\operatorname{Var}(X_i),$$

$$\sigma_Y^2=\operatorname{Var}(Y_i),$$

$$c=\operatorname{Cov}(X_i,Y_i).$$

这里 $c$ 已经包含路径上多个积分节点的协方差结构。因为

$$Y_i=\sum_{q=1}^{Q}a_q\nabla_{\theta_k}L(z_i,v_q),$$

所以

$$c
=\operatorname{Cov}\left(\nabla_{\theta_k}L(z_i,\nu),\sum_{q=1}^{Q}a_q\nabla_{\theta_k}L(z_i,v_q)\right)$$

$$=\sum_{q=1}^{Q}a_q\operatorname{Cov}\left(\nabla_{\theta_k}L(z_i,\nu),\nabla_{\theta_k}L(z_i,v_q)\right).$$

同理，路径积分梯度 $Y_i$ 的方差为

$$\sigma_Y^2
=\operatorname{Var}\left(\sum_{q=1}^{Q}a_q\nabla_{\theta_k}L(z_i,v_q)\right)$$

$$=\sum_{q=1}^{Q}\sum_{r=1}^{Q}a_qa_r\operatorname{Cov}\left(\nabla_{\theta_k}L(z_i,v_q),\nabla_{\theta_k}L(z_i,v_r)\right).$$

因此，后文虽然使用简洁的 $X_i,Y_i$ 记号，但它并没有忽略路径积分节点之间的噪声相关性。

## 2. 噪声随路径变化时的过估计推导

先回到路径积分形式。对任意参数位置 $\theta$，minibatch 梯度可以写成

$$g_k(\theta)=g_k^*(\theta)+\eta_k(\theta),$$

其中 $g_k^*(\theta)$ 是全数据梯度，$\eta_k(\theta)$ 是由 minibatch 造成的随机梯度噪声。我们假设 minibatch 梯度在每个位置无偏，即

$$\mathbb E[\eta_k(\theta)]=0.$$

关键点在于：$\eta_k(\theta)$ 不是一个固定随机变量，而是随参数位置变化的随机过程。更新点 $\nu$ 处的噪声是 $\eta_k(\nu)$，路径积分节点 $v_q$ 处的噪声是 $\eta_k(v_q)$。这两个随机变量通常相关，但不一定相等。

理想重要性写成连续形式为

$$\omega_k^t=\Delta\theta_k\int_0^1 g_k^*(v_\alpha)d\alpha.$$

实际观测时，路径上的梯度由 minibatch 估计，所以

$$\hat\omega_k^t=\Delta\theta_k\int_0^1 g_k(v_\alpha)d\alpha.$$

代入 $g_k(v_\alpha)=g_k^*(v_\alpha)+\eta_k(v_\alpha)$，得到

$$\hat\omega_k^t
=\Delta\theta_k\int_0^1\left(g_k^*(v_\alpha)+\eta_k(v_\alpha)\right)d\alpha.$$

拆开积分：

$$\hat\omega_k^t
=\Delta\theta_k\int_0^1g_k^*(v_\alpha)d\alpha
+\Delta\theta_k\int_0^1\eta_k(v_\alpha)d\alpha.$$

第一项就是理想重要性 $\omega_k^t$，所以误差为

$$\hat\omega_k^t-\omega_k^t
=\Delta\theta_k\int_0^1\eta_k(v_\alpha)d\alpha.$$

到这里还没有出现过估计。过估计来自 $\Delta\theta_k$ 本身也是由 noisy gradient 决定的。按照 SGD 的局部近似，吸收符号后写成

$$\Delta\theta_k\approx \gamma g_k(\nu)=\gamma\left(g_k^*(\nu)+\eta_k(\nu)\right).$$

代入误差项：

$$\hat\omega_k^t-\omega_k^t
\approx
\gamma\left(g_k^*(\nu)+\eta_k(\nu)\right)
\int_0^1\eta_k(v_\alpha)d\alpha.$$

现在对误差取期望：

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\mathbb E\left[
\left(g_k^*(\nu)+\eta_k(\nu)\right)
\int_0^1\eta_k(v_\alpha)d\alpha
\right].$$

在可以交换期望和积分的条件下，得到

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\int_0^1
\mathbb E\left[
\left(g_k^*(\nu)+\eta_k(\nu)\right)
\eta_k(v_\alpha)
\right]d\alpha.$$

展开括号：

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\int_0^1
\left(
\mathbb E[g_k^*(\nu)\eta_k(v_\alpha)]
+
\mathbb E[\eta_k(\nu)\eta_k(v_\alpha)]
\right)d\alpha.$$

由于 $g_k^*(\nu)$ 是给定参数位置下的全数据梯度，不是 minibatch 随机噪声，因此

$$\mathbb E[g_k^*(\nu)\eta_k(v_\alpha)]
=g_k^*(\nu)\mathbb E[\eta_k(v_\alpha)]=0.$$

于是只剩下第二项：

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\int_0^1
\mathbb E[\eta_k(\nu)\eta_k(v_\alpha)]d\alpha.$$

又因为两处噪声均值为零，

$$\mathbb E[\eta_k(\nu)\eta_k(v_\alpha)]
=
\operatorname{Cov}(\eta_k(\nu),\eta_k(v_\alpha)).$$

因此，在噪声随路径变化时，均值过估计的连续形式为

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\int_0^1
\operatorname{Cov}(\eta_k(\nu),\eta_k(v_\alpha))d\alpha.$$

若使用离散数值积分，误差项为

$$\hat\omega_k^t-\omega_k^t
\approx
\gamma\left(g_k^*(\nu)+\eta_k(\nu)\right)
\sum_{q=1}^{Q}a_q\eta_k(v_q).$$

同样取期望，得到

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\sum_{q=1}^{Q}a_q
\operatorname{Cov}(\eta_k(\nu),\eta_k(v_q)).$$

这就是“方差形式”需要推广为“协方差形式”的原因。若所有积分节点都近似等于更新点，即 $v_q\approx\nu$，或者路径上的噪声几乎不变，即 $\eta_k(v_q)\approx\eta_k(\nu)$，那么

$$\operatorname{Cov}(\eta_k(\nu),\eta_k(v_q))
\approx
\operatorname{Var}(\eta_k(\nu)).$$

这时上式退化为

$$\mathbb E[\hat\omega_k^t-\omega_k^t]
\approx
\gamma\operatorname{Var}(\eta_k(\nu)),$$

也就是局部近似下的“减方差”形式。但在真正的多节点路径积分中，更一般、更严格的对象是协方差。

## 3. 双采样法：无偏性与方差

双采样法的思想很直接：既然偏差来自“同一个 minibatch 的更新点噪声”和“同一个 minibatch 的积分节点噪声”相乘，那么就用两个独立 minibatch 把这两个噪声来源拆开。

令 batch $A$ 用于决定更新方向，batch $B$ 用于计算路径积分梯度。定义

$$\bar X_A=\frac{1}{B_A}\sum_{i\in A}X_i,$$

$$\bar Y_B=\frac{1}{B_B}\sum_{j\in B}Y_j.$$

双采样估计器为

$$\hat\Omega_D=\gamma\bar X_A\bar Y_B.$$

因为 batch $A$ 和 batch $B$ 独立，$\bar X_A$ 与 $\bar Y_B$ 独立。同时

$$\mathbb E[\bar X_A]=\mu_X,\qquad \mathbb E[\bar Y_B]=\mu_Y.$$

所以

$$\mathbb E[\hat\Omega_D]
=\gamma\mathbb E[\bar X_A\bar Y_B]
=\gamma\mathbb E[\bar X_A]\mathbb E[\bar Y_B]
=\gamma\mu_X\mu_Y
=\Omega_k.$$

因此，双采样法是无偏的。它的无偏性不依赖 $X_i$ 与 $Y_i$ 在同一个样本内是否相关，因为它根本不让同一个样本或同一个 batch 同时提供这两个因子。

接下来计算方差。由于 $\bar X_A$ 与 $\bar Y_B$ 独立，

$$\operatorname{Var}(\hat\Omega_D)
=\gamma^2\operatorname{Var}(\bar X_A\bar Y_B).$$

由

$$\operatorname{Var}(\bar X_A\bar Y_B)
=\mathbb E[\bar X_A^2]\mathbb E[\bar Y_B^2]-\mu_X^2\mu_Y^2,$$

而

$$\mathbb E[\bar X_A^2]=\mu_X^2+\frac{\sigma_X^2}{B_A},$$

$$\mathbb E[\bar Y_B^2]=\mu_Y^2+\frac{\sigma_Y^2}{B_B},$$

所以

$$\operatorname{Var}(\hat\Omega_D)
=
\gamma^2\left[
\frac{\mu_Y^2\sigma_X^2}{B_A}
+
\frac{\mu_X^2\sigma_Y^2}{B_B}
+
\frac{\sigma_X^2\sigma_Y^2}{B_AB_B}
\right].$$

若为了公平比较，总样本预算为 $B$，双采样法把它拆成两半，即

$$B_A=B_B=\frac{B}{2},$$

则

$$\operatorname{Var}(\hat\Omega_D)
=
\gamma^2\left[
\frac{2\mu_Y^2\sigma_X^2}{B}
+
\frac{2\mu_X^2\sigma_Y^2}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
\right].$$

这个结果说明，双采样虽然消除了同源噪声偏差，但代价是每一侧只有一半样本，所以方差的一阶项会增大。

## 4. 标准 U-statistic 方法是什么

U-statistic 是统计学中一种构造无偏估计器的标准方法。设 $Z_1,\ldots,Z_n$ 是独立同分布样本，若我们要估计的目标可以写成

$$\theta=\mathbb E[h(Z_1,\ldots,Z_r)],$$

其中 $h$ 是一个对 $r$ 个独立样本作用的核函数，那么对应的 $r$ 阶 U-statistic 定义为

$$U_n=\binom{n}{r}^{-1}\sum_{1\le i_1<\cdots<i_r\le n}h(Z_{i_1},\ldots,Z_{i_r}).$$

它的核心思想是：把所有互不相同的样本组合都用上，然后取平均。由于每一组互异样本都服从与 $(Z_1,\ldots,Z_r)$ 相同的联合分布，所以

$$\mathbb E[U_n]=\theta.$$

在我们的问题中，每个统计单元不是只有一个数，而是一对随机变量

$$Z_i=(X_i,Y_i).$$

目标是

$$\mu_X\mu_Y=\mathbb E[X_1]\mathbb E[Y_2],$$

其中 $X_1$ 和 $Y_2$ 来自两个独立样本。于是可以定义二阶对称核函数

$$h(Z_i,Z_j)=\frac{1}{2}(X_iY_j+X_jY_i).$$

当 $i\ne j$ 时，$X_i$ 与 $Y_j$ 独立，因此

$$\mathbb E[X_iY_j]=\mu_X\mu_Y.$$

所以

$$\mathbb E[h(Z_i,Z_j)]=\mu_X\mu_Y.$$

对应的二阶 U-statistic 为

$$U_B=\binom{B}{2}^{-1}\sum_{i<j}\frac{1}{2}(X_iY_j+X_jY_i).$$

它也可以写成更直观的有序对形式：

$$U_B=\frac{1}{B(B-1)}\sum_{i\ne j}X_iY_j.$$

这个形式非常重要。它说明 U-statistic 方法的本质是：只保留 $i\ne j$ 的交叉项，去掉 $i=j$ 的自乘项。自乘项 $X_iY_i$ 会包含同一样本在更新点和路径积分上的协方差，正是偏差来源；而交叉项 $X_iY_j$ 中两个因子来自不同样本，因而无偏。

因此，若能够获得逐样本梯度，则可以构造

$$\hat\Omega_U=\gamma\frac{1}{B(B-1)}\sum_{i\ne j}X_iY_j.$$

它就是标准二阶 U-statistic 在参数积分梯度重要性估计中的应用。

## 5. Microbatch-level U-statistic：工程可行版本

逐样本梯度在神经网络中通常难以获得，因为它需要保存或计算每个样本对每个参数的梯度。更实际的做法是把一个总 batch 拆成 $M$ 个 microbatch，每个 microbatch 大小为

$$b=\frac{B}{M}.$$

第 $m$ 个 microbatch 的更新点梯度均值定义为

$$X_m^{(b)}=\frac{1}{b}\sum_{i\in I_m}X_i.$$

第 $m$ 个 microbatch 的路径积分梯度均值定义为

$$Y_m^{(b)}=\frac{1}{b}\sum_{i\in I_m}Y_i.$$

在工程实现中，$X_m^{(b)}$ 可以理解为第 $m$ 个 microbatch 在更新点 $\nu$ 上反向传播得到的 local gradient；$Y_m^{(b)}$ 可以理解为同一个 microbatch 在路径节点 $v_q$ 上分别反向传播、再按积分权重加权后的 local gradient。若使用多卡数据并行，每张卡同步前的 local gradient 可以被视为一个 microbatch 梯度；若使用 gradient accumulation，每次 accumulation 的梯度也可以被视为一个 microbatch 梯度。

由于 microbatch 是 $b$ 个独立样本的平均，

$$\mathbb E[X_m^{(b)}]=\mu_X,\qquad \mathbb E[Y_m^{(b)}]=\mu_Y.$$

它们的方差和协方差为

$$\operatorname{Var}(X_m^{(b)})=\frac{\sigma_X^2}{b},$$

$$\operatorname{Var}(Y_m^{(b)})=\frac{\sigma_Y^2}{b},$$

$$\operatorname{Cov}(X_m^{(b)},Y_m^{(b)})=\frac{c}{b}.$$

如果直接用所有 microbatch 的平均值相乘，令

$$\bar X=\frac{1}{M}\sum_{m=1}^{M}X_m^{(b)},\qquad
\bar Y=\frac{1}{M}\sum_{m=1}^{M}Y_m^{(b)},$$

则原始单采样估计为

$$\hat\Omega_{raw}=\gamma\bar X\bar Y.$$

它的期望为

$$\mathbb E[\hat\Omega_{raw}]
=\gamma\left(\mu_X\mu_Y+\operatorname{Cov}(\bar X,\bar Y)\right).$$

而

$$\operatorname{Cov}(\bar X,\bar Y)
=\frac{1}{M^2}\sum_{m=1}^{M}\sum_{n=1}^{M}
\operatorname{Cov}(X_m^{(b)},Y_n^{(b)}).$$

当 $m\ne n$ 时，不同 microbatch 独立，协方差为零；当 $m=n$ 时，协方差为 $c/b$。所以

$$\operatorname{Cov}(\bar X,\bar Y)
=\frac{1}{M^2}\cdot M\cdot\frac{c}{b}
=\frac{c}{Mb}
=\frac{c}{B}.$$

因此

$$\mathbb E[\hat\Omega_{raw}]
=\Omega_k+\gamma\frac{c}{B}.$$

这说明原始单采样乘积仍然有协方差偏差。为了去掉这个偏差，我们在 microbatch 层面仿照 U-statistic，只使用不同 microbatch 之间的交叉项：

$$\hat\Omega_{MB-U}
=
\gamma\frac{1}{M(M-1)}\sum_{m\ne n}X_m^{(b)}Y_n^{(b)}.$$

这就是 microbatch-level U-statistic。

## 6. Microbatch-level U-statistic 的无偏性

对 $\hat\Omega_{MB-U}$ 取期望：

$$\mathbb E[\hat\Omega_{MB-U}]
=
\gamma\frac{1}{M(M-1)}\sum_{m\ne n}
\mathbb E[X_m^{(b)}Y_n^{(b)}].$$

当 $m\ne n$ 时，两个 microbatch 独立，所以

$$\mathbb E[X_m^{(b)}Y_n^{(b)}]
=
\mathbb E[X_m^{(b)}]\mathbb E[Y_n^{(b)}]
=
\mu_X\mu_Y.$$

一共有 $M(M-1)$ 个有序交叉项，因此

$$\mathbb E[\hat\Omega_{MB-U}]
=
\gamma\frac{1}{M(M-1)}\cdot M(M-1)\mu_X\mu_Y
=\gamma\mu_X\mu_Y
=\Omega_k.$$

所以，microbatch-level U-statistic 是无偏的。

从另一个角度看，它等价于把原始单采样乘积中的“对角线项”减掉。因为

$$\left(\sum_{m=1}^{M}X_m^{(b)}\right)
\left(\sum_{n=1}^{M}Y_n^{(b)}\right)
=
\sum_{m=1}^{M}\sum_{n=1}^{M}X_m^{(b)}Y_n^{(b)},$$

其中包含 $m=n$ 的自乘项。去掉这些自乘项，得到

$$\sum_{m\ne n}X_m^{(b)}Y_n^{(b)}
=
\left(\sum_{m=1}^{M}X_m^{(b)}\right)
\left(\sum_{n=1}^{M}Y_n^{(b)}\right)
-
\sum_{m=1}^{M}X_m^{(b)}Y_m^{(b)}.$$

因此

$$\hat\Omega_{MB-U}
=
\gamma
\frac{
\left(\sum_m X_m^{(b)}\right)
\left(\sum_m Y_m^{(b)}\right)
-
\sum_m X_m^{(b)}Y_m^{(b)}
}{M(M-1)}.$$

对所有参数同时计算时，只需把乘法理解成逐参数乘法 $\odot$：

$$\hat\Omega_{MB-U}
=
\gamma
\frac{S_X\odot S_Y-S_{XY}}{M(M-1)},$$

其中

$$S_X=\sum_m X_m^{(b)},\qquad
S_Y=\sum_m Y_m^{(b)},\qquad
S_{XY}=\sum_m X_m^{(b)}\odot Y_m^{(b)}.$$

这个形式说明它非常适合工程实现：不一定要保存所有 microbatch 的完整梯度，只要在流式计算中累积 $S_X,S_Y,S_{XY}$ 即可。

## 7. Microbatch-level U-statistic 的方差

下面计算 $\hat\Omega_{MB-U}$ 的方差。为了使用标准 U-statistic 方差公式，定义 microbatch 统计单元

$$Z_m=(X_m^{(b)},Y_m^{(b)}).$$

对应的对称核函数为

$$h(Z_m,Z_n)=\frac{1}{2}\left(X_m^{(b)}Y_n^{(b)}+X_n^{(b)}Y_m^{(b)}\right).$$

于是

$$\frac{1}{\binom{M}{2}}\sum_{m<n}h(Z_m,Z_n)
=
\frac{1}{M(M-1)}\sum_{m\ne n}X_m^{(b)}Y_n^{(b)}.$$

也就是说，$\hat\Omega_{MB-U}/\gamma$ 是一个二阶 U-statistic。

先计算 U-statistic 方差公式中需要的第一部分。给定 $Z_1=(X_1^{(b)},Y_1^{(b)})$，有

$$\mathbb E[h(Z_1,Z_2)\mid Z_1]
=
\frac{1}{2}\mathbb E\left[X_1^{(b)}Y_2^{(b)}+X_2^{(b)}Y_1^{(b)}\mid Z_1\right].$$

因为 $Z_2$ 与 $Z_1$ 独立，

$$\mathbb E[Y_2^{(b)}]=\mu_Y,
\qquad
\mathbb E[X_2^{(b)}]=\mu_X.$$

因此

$$\mathbb E[h(Z_1,Z_2)\mid Z_1]
=
\frac{1}{2}\left(\mu_YX_1^{(b)}+\mu_XY_1^{(b)}\right).$$

它的方差为

$$\operatorname{Var}\left(\mathbb E[h(Z_1,Z_2)\mid Z_1]\right)
=
\frac{1}{4}\operatorname{Var}\left(\mu_YX_1^{(b)}+\mu_XY_1^{(b)}\right).$$

利用 microbatch 层面的方差和协方差，得到

$$\operatorname{Var}\left(\mathbb E[h(Z_1,Z_2)\mid Z_1]\right)
=
\frac{1}{4}\left(
\mu_Y^2\frac{\sigma_X^2}{b}
+
\mu_X^2\frac{\sigma_Y^2}{b}
+2\mu_X\mu_Y\frac{c}{b}
\right).$$

记

$$A=\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+2\mu_X\mu_Yc.$$

则上式为

$$\operatorname{Var}\left(\mathbb E[h(Z_1,Z_2)\mid Z_1]\right)=\frac{A}{4b}.$$

接着计算核函数本身的方差。由

$$h(Z_1,Z_2)=\frac{1}{2}\left(X_1^{(b)}Y_2^{(b)}+X_2^{(b)}Y_1^{(b)}\right),$$

可得

$$\operatorname{Var}(h)
=
\frac{1}{4}\operatorname{Var}\left(X_1^{(b)}Y_2^{(b)}+X_2^{(b)}Y_1^{(b)}\right).$$

展开后，

$$\operatorname{Var}(h)
=
\frac{1}{4}\left[
\operatorname{Var}(X_1^{(b)}Y_2^{(b)})
+
\operatorname{Var}(X_2^{(b)}Y_1^{(b)})
+2\operatorname{Cov}(X_1^{(b)}Y_2^{(b)},X_2^{(b)}Y_1^{(b)})
\right].$$

两个方差项对称。由于 $X_1^{(b)}$ 与 $Y_2^{(b)}$ 来自不同 microbatch，彼此独立，因此

$$\operatorname{Var}(X_1^{(b)}Y_2^{(b)})
=
\frac{\sigma_X^2\sigma_Y^2}{b^2}
+
\frac{\mu_Y^2\sigma_X^2}{b}
+
\frac{\mu_X^2\sigma_Y^2}{b}.$$

协方差项为

$$\operatorname{Cov}(X_1^{(b)}Y_2^{(b)},X_2^{(b)}Y_1^{(b)})
=
2\mu_X\mu_Y\frac{c}{b}+\frac{c^2}{b^2}.$$

代回得到

$$\operatorname{Var}(h)
=
\frac{1}{2}\left(
\frac{A}{b}+
\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2}
\right).$$

二阶 U-statistic 的方差分解为

$$\operatorname{Var}(U_M)
=
\frac{2}{M(M-1)}\operatorname{Var}(h)
+
\frac{4(M-2)}{M(M-1)}
\operatorname{Var}\left(\mathbb E[h(Z_1,Z_2)\mid Z_1]\right).$$

把上面两个结果代入，得到

$$\operatorname{Var}(\hat\Omega_{MB-U}/\gamma)
=
\frac{2}{M(M-1)}\cdot
\frac{1}{2}\left(
\frac{A}{b}+\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2}
\right)
+
\frac{4(M-2)}{M(M-1)}\cdot\frac{A}{4b}.$$

整理：

$$\operatorname{Var}(\hat\Omega_{MB-U}/\gamma)
=
\frac{1}{M(M-1)}\left(
\frac{A}{b}+\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2}
\right)
+
\frac{M-2}{M(M-1)}\frac{A}{b}.$$

合并 $A$ 项：

$$\operatorname{Var}(\hat\Omega_{MB-U}/\gamma)
=
\frac{M-1}{M(M-1)}\frac{A}{b}
+
\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2M(M-1)}.$$

即

$$\operatorname{Var}(\hat\Omega_{MB-U}/\gamma)
=
\frac{A}{Mb}
+
\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2M(M-1)}.$$

由于 $B=Mb$，乘回 $\gamma^2$ 后得到

$$\operatorname{Var}(\hat\Omega_{MB-U})
=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+
2\mu_X\mu_Yc
}{B}
+
\frac{\sigma_X^2\sigma_Y^2+c^2}{b^2M(M-1)}
\right].$$

也可以写成

$$\operatorname{Var}(\hat\Omega_{MB-U})
=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+
2\mu_X\mu_Yc
}{B}
+
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}
\right].$$

这个式子很有解释性。第一项是 $O(1/B)$ 的主导项；第二项是 $O(1/B^2)$ 的二阶项。使用 microbatch 而非逐样本时，主导项不变，但二阶项中出现了 $M/(M-1)$ 这样的 microbatch 数量修正。这说明 microbatch-level U-statistic 在统计上仍然有效，只是因为统计单元数量从 $B$ 个样本减少到 $M$ 个 microbatch，有限样本二阶项会略大。

## 8. 与双采样法的方差比较

在相同总样本预算 $B$ 下，双采样法使用两个独立 batch，各自大小为 $B/2$，其方差为

$$\operatorname{Var}(\hat\Omega_D)
=
\gamma^2\left[
\frac{2\mu_Y^2\sigma_X^2}{B}
+
\frac{2\mu_X^2\sigma_Y^2}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
\right].$$

Microbatch-level U-statistic 的方差为

$$\operatorname{Var}(\hat\Omega_{MB-U})
=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+
2\mu_X\mu_Yc
}{B}
+
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}
\right].$$

二者相减：

$$\operatorname{Var}(\hat\Omega_D)-\operatorname{Var}(\hat\Omega_{MB-U})$$

$$=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2-2\mu_X\mu_Yc
}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
-
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}
\right].$$

第一项可以写成

$$\frac{\operatorname{Var}(\mu_YX_i-\mu_XY_i)}{B},$$

所以一定非负。第二项也通常非负。由 Cauchy-Schwarz 不等式，

$$c^2\le \sigma_X^2\sigma_Y^2.$$

因此

$$\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{M-1}
\le
\frac{2M\sigma_X^2\sigma_Y^2}{M-1}.$$

于是第二项至少不小于

$$\frac{\sigma_X^2\sigma_Y^2}{B^2}\left(4-\frac{2M}{M-1}\right)
=
\frac{\sigma_X^2\sigma_Y^2}{B^2}\frac{2M-4}{M-1}.$$

当 $M\ge 2$ 时，这一项非负；当 $M>2$ 时通常严格为正。因此，在相同总样本预算下，microbatch-level U-statistic 的方差通常不大于双采样法。

直观原因是：双采样为了制造独立性，把样本预算拆成两半；而 U-statistic 方法没有浪费这部分信息，它在同一个 batch 内通过去掉对角项、保留交叉项来制造独立结构。因此它既保留无偏性，又更充分地利用样本。

## 9. Cramer-Rao 方差下界与效率比较

现在推导一个理想化的方差下界，用来判断这些估计器离理论最优有多远。

假设 microbatch 统计单元近似服从二维高斯分布：

$$Z_m=\begin{pmatrix}X_m^{(b)}\\Y_m^{(b)}\end{pmatrix}
\sim
N\left(
\begin{pmatrix}\mu_X\\\mu_Y\end{pmatrix},
\frac{1}{b}
\begin{pmatrix}
\sigma_X^2 & c\\
c & \sigma_Y^2
\end{pmatrix}
\right),$$

且 $m=1,\ldots,M$ 之间独立。这里暂时把协方差矩阵视为已知，只把均值

$$\mu=(\mu_X,\mu_Y)^\top$$

看作待估参数。目标函数是

$$\Omega_k=\gamma\mu_X\mu_Y.$$

对多元正态分布，如果协方差矩阵已知，均值参数的 Fisher 信息矩阵为

$$I(\mu)=M\Sigma_b^{-1},$$

其中

$$\Sigma_b=\frac{1}{b}
\begin{pmatrix}
\sigma_X^2 & c\\
c & \sigma_Y^2
\end{pmatrix}.$$

因此

$$I(\mu)^{-1}=\frac{1}{M}\Sigma_b
=
\frac{1}{Mb}
\begin{pmatrix}
\sigma_X^2 & c\\
c & \sigma_Y^2
\end{pmatrix}
=
\frac{1}{B}
\begin{pmatrix}
\sigma_X^2 & c\\
c & \sigma_Y^2
\end{pmatrix}.$$

由于

$$\Omega_k=\gamma\mu_X\mu_Y,$$

其梯度为

$$\nabla_\mu\Omega_k
=
\gamma
\begin{pmatrix}
\mu_Y\\
\mu_X
\end{pmatrix}.$$

Cramer-Rao 下界给出，对于任意正则无偏估计器 $\hat\Omega_k$，都有

$$\operatorname{Var}(\hat\Omega_k)
\ge
\nabla_\mu\Omega_k^\top I(\mu)^{-1}\nabla_\mu\Omega_k.$$

代入上面的表达式：

$$\operatorname{Var}(\hat\Omega_k)
\ge
\gamma^2
\begin{pmatrix}
\mu_Y & \mu_X
\end{pmatrix}
\frac{1}{B}
\begin{pmatrix}
\sigma_X^2 & c\\
c & \sigma_Y^2
\end{pmatrix}
\begin{pmatrix}
\mu_Y\\
\mu_X
\end{pmatrix}.$$

展开矩阵乘法，得到

$$\operatorname{Var}(\hat\Omega_k)
\ge
\gamma^2\frac{
\mu_Y^2\sigma_X^2+
2\mu_X\mu_Yc+
\mu_X^2\sigma_Y^2
}{B}.$$

记

$$A=\mu_Y^2\sigma_X^2+\mu_X^2\sigma_Y^2+2\mu_X\mu_Yc,$$

则 Cramer-Rao 下界为

$$\operatorname{CRLB}=\gamma^2\frac{A}{B}.$$

而 microbatch-level U-statistic 的方差为

$$\operatorname{Var}(\hat\Omega_{MB-U})
=
\gamma^2\left[
\frac{A}{B}
+
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}
\right].$$

因此

$$\operatorname{Var}(\hat\Omega_{MB-U})-\operatorname{CRLB}
=
\gamma^2\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}.$$

这说明 microbatch-level U-statistic 正好达到 Cramer-Rao 下界的一阶主项，只比下界多一个 $O(1/B^2)$ 的有限样本二阶项。随着总样本数 $B$ 增大，这个差距相对于主项会快速变小。因此，在这个理想高斯模型下，microbatch-level U-statistic 可以看作渐近有效的无偏估计器。

双采样法在相同总样本预算 $B$ 下的方差为

$$\operatorname{Var}(\hat\Omega_D)
=
\gamma^2\left[
\frac{2\mu_Y^2\sigma_X^2}{B}
+
\frac{2\mu_X^2\sigma_Y^2}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
\right].$$

它与 CRLB 的差为

$$\operatorname{Var}(\hat\Omega_D)-\operatorname{CRLB}
=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2-2\mu_X\mu_Yc
}{B}
+
\frac{4\sigma_X^2\sigma_Y^2}{B^2}
\right].$$

其中第一项等于

$$\gamma^2\frac{\operatorname{Var}(\mu_YX_i-\mu_XY_i)}{B},$$

是 $O(1/B)$ 的差距。因此，双采样法虽然无偏，但在相同总样本预算下，它距离 full paired-data 设计下的 Cramer-Rao 下界通常有一阶差距。

这里需要强调一个细节：如果把“双采样”本身看作一种固定实验设计，即只观测 $B/2$ 个 $X$ 和 $B/2$ 个独立的 $Y$，那么它也可以接近自己这个实验设计下的方差下界。但这个实验设计本身比“同一批 microbatch 同时观测 $X$ 和 $Y$，再用 U-statistic 去掉偏差”的设计信息更少。因此，从相同总样本预算和可观测 microbatch local gradient 的角度看，U-statistic 方法更接近理论最优。

## 10. 结论与实践含义

在噪声随路径变化的设定下，参数积分梯度重要性的过估计不应简单理解为更新点处的噪声方差，而应理解为更新点噪声与路径积分节点噪声之间的协方差。连续形式为

$$\gamma\int_0^1\operatorname{Cov}(\eta_k(\nu),\eta_k(v_\alpha))d\alpha,$$

离散数值积分形式为

$$\gamma\sum_q a_q\operatorname{Cov}(\eta_k(\nu),\eta_k(v_q)).$$

双采样法通过使用两个独立 minibatch，令更新方向噪声和路径积分噪声相互独立，因此可以消除这个协方差偏差。它的优点是简单、直观、无偏性容易保证；缺点是在相同总样本预算下方差较大。

U-statistic 方法的核心是去掉同一统计单元内部的自乘项，只保留不同统计单元之间的交叉项。逐样本版本需要逐样本梯度，工程代价较高；microbatch-level 版本则把每个 microbatch 或每张卡的 local gradient 当作一个统计单元，更容易实现。它的估计器为

$$\hat\Omega_{MB-U}
=
\gamma\frac{1}{M(M-1)}\sum_{m\ne n}X_m^{(b)}Y_n^{(b)},$$

或等价地写成

$$\hat\Omega_{MB-U}
=
\gamma\frac{S_X\odot S_Y-S_{XY}}{M(M-1)}.$$

它在理论上无偏，方差为

$$\operatorname{Var}(\hat\Omega_{MB-U})
=
\gamma^2\left[
\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+
2\mu_X\mu_Yc
}{B}
+
\frac{M(\sigma_X^2\sigma_Y^2+c^2)}{B^2(M-1)}
\right].$$

在高斯近似和协方差已知的 Cramer-Rao 框架下，方差下界为

$$\operatorname{CRLB}
=
\gamma^2\frac{
\mu_Y^2\sigma_X^2+
\mu_X^2\sigma_Y^2+
2\mu_X\mu_Yc
}{B}.$$

因此，microbatch-level U-statistic 达到了下界的一阶主项，只多出 $O(1/B^2)$ 的有限样本二阶项；而双采样法在相同总样本预算下通常与该下界存在 $O(1/B)$ 的差距。

对参数重要性的实际应用而言，尤其是剪枝、稀疏化、结构选择等需要较准确排序的任务，无偏和低方差非常重要。若逐样本梯度不可行，而训练过程本身已经使用多卡或 microbatch accumulation，那么截取每个 microbatch 的 local gradient 来构造 microbatch-level U-statistic，是一个理论上合理且工程上可实现的估计方案。

