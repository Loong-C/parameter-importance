# S2.2 Stage 0/1 交接与固定状态代码契约

## 1. 子任务目的

把 Stage 0 的基础设施和 Stage 1 的公式正确性转化为 Stage 2 可以信任的冻结交接契约。当前 `main` 尚无可执行实现，因此以下条目描述对未来 `G1-EXIT` 产物的验收与复用，不在 Stage 2 重新建立 estimator/provider/registry 内核。若验收发现语义或实现缺陷，必须返回 Stage 1 修复并重跑受影响 Gate。

## 2. 前置条件

- Stage 0 所有进入 Stage 1/2 的核心 Gate 有机器可读报告；
- GPU 枚举、ECC 和候选四卡健康问题已由管理员或已授权受控流程解决；本阶段不执行 reset、驱动/持久化模式/功率/时钟变更或 `sudo`；
- 本机、GitHub、服务器 Git 起点一致，`Agent/` 五文件两端哈希一致；
- Stage 1 已建立现行项目骨架、依赖入口、测试入口和配置 schema；
- 用户及并发任务的现有修改已审查并保留。

## 3. 实施步骤

### 3.1 验收 Stage 0 交接包

1. 读取最新环境、GPU、存储、模型和数据 manifest。
2. 核对服务器 venv 的现场 freeze 与环境锁文件。
3. 核对单卡、四卡 DDP、梯度累积和 `no_sync` 报告。
4. 核对日志、run ID、checkpoint、恢复和原子发布能力。
5. 核对 Stage 2 需要的 wall-clock、峰值显存、样本/token 计数和失败状态字段可被记录。
6. 将每项 Stage 0 证据的路径、哈希、提交和验收时间写入 Stage 2 handoff manifest。
7. 任一证据过期或与当前硬件不一致时重新执行对应 smoke test，不复用旧 `READY`。
8. 核对 `HF_HOME`、`HF_DATASETS_CACHE`、`TORCH_HOME`、`XDG_CACHE_HOME` 都位于 `$DATA_ROOT/cache`，`TMPDIR` 位于 `$DATA_ROOT/tmp/stage2/<run-id>`；不得修改 `HOME` 或使用根盘缓存。

### 3.2 验收 Stage 1 数学不变量

1. 验证 `S1^2-S2` 与显式跨 microbatch 对求和逐坐标一致。
2. 验证所有 microbatch 梯度相同时 U 精确退化为梯度平方。
3. 验证人工零均值噪声下 raw 正偏而 U 重复均值为零。
4. 验证 microbatch 顺序置换不改变 S1、S2 和 U。
5. 验证等权 microbatch 平均梯度与完整 batch 梯度一致。
6. 验证不等有效 token 数时的加权平均和加权去对角公式。
7. 验证 `M=2` 的 U 与同一两半样本双采样逐坐标一致。
8. 验证单卡与四卡在相同 global batch、顺序和精度下落入预设浮点容差。
9. 验证 AMP unscale 后统计与 FP32 对照一致；Stage 2 主实验仍固定使用 FP32。
10. 验证 U 的负输出被完整保留，任何公共 API 都不会默认 clamp。

### 3.3 验收并冻结 Stage 1 参数注册表

1. 读取 `G1-REGISTRY` 交付的 canonical parameter registry 和 schema 版本。
2. 复核所有 `requires_grad` 参数的名称、shape、dtype、numel、layer 和 module 分类。
3. 复核 tied/shared storage 只对应一个 canonical ID，不发生重复统计。
4. 复核扁平化顺序和反向映射在 checkpoint、reference、估计器和图表入口一致。
5. 复核“预期无梯度”和“异常无梯度”的区分，异常情况 fail closed。
6. 为每个模型和 checkpoint 记录 registry hash；结构变化时返回 Stage 1，不在 Stage 2 修改映射。

### 3.4 验收并冻结 Stage 1 fixed-state provider

1. 绑定 `G1-EXIT` 的 exact commit、provider API、loss reduction、dtype 和参数 registry 版本。
2. 复核模型/checkpoint 加载与状态快照不产生可见状态变化。
3. 复核 packed sequence、label、有效目标 token 和统一 scalar loss reduction 路径。
4. 复核一次 microbatch 返回真实尺度 FP32 mean gradient，并在每个单元前清理临时梯度。
5. 复核 optimizer/scheduler step、权重更新、running-stat 更新和隐式数据游标推进均被禁止。
6. 分别记录模型参数、buffer、optimizer/scheduler（若对象存在）和模型/全局 RNG 的运行前后摘要。
7. 单独记录 sampling generator 与 data-loader worker RNG 的起止状态和派生 seed；它们按 manifest 推进是预期行为，不能要求与运行前相等。
8. 任一内核摘要或接口与 Stage 1 交接不一致时停止 Stage 2，建立回退项并重跑受影响的 `G1-*`，不能静默修补。

### 3.5 只新增 Stage 2 适配与编排组件

1. **数据与采样组件**：只负责从冻结采样 manifest 解析 sample ID，不负责随机临时选样。
2. **梯度适配层**：只调用冻结 provider 产生 microbatch mean gradients 和数值元数据，不复制 provider 实现。
3. **估计器适配层**：只调用冻结 raw/double/U API，不在 Stage 2 重写公式。
4. **reference 编排层**：流式累计参考均值、sequence 方差和参考诊断，不改变 estimator 内核。
5. **重复运行组件**：处理 repetition 状态、断点、原子发布和失败记录。
6. **统计组件**：使用流式均值/M2/MSE 累加和重复级指标，不修改原始估计值。
7. **profiling 组件**：记录完整梯度时间与公式时间，避免把两者混为一个成本。
8. **报告组件**：只读取冻结结果 schema 生成表格、图和 Gate，不重新计算模型梯度。

### 3.6 审查旧实现而不继承旧结论

1. 从固定旧提交只读提取 estimator、provider、测试和结果 schema 的设计清单。
2. 标记可复用概念、已知 bug、正值截断、M/B 混杂和 reference 设计缺口。
3. 旧对象可能被 Git 垃圾回收，只能作为一次性审阅材料，不能成为构建、测试或运行依赖。
4. 仅在现行主线通过 Stage 1 的正式实现和测试后复用已经冻结的公共 API。
5. 禁止整分支恢复、无审查 cherry-pick 或把旧测试数量视为新实现证据。
6. 将新旧公式对照和不兼容字段写入迁移说明。

### 3.7 建立本地与服务器测试层次

1. 本机运行不依赖 CUDA 的纯公式、schema、seed 和小张量测试。
2. 服务器 CPU 运行相同测试，验证平台无关性。
3. 服务器单 GPU 运行真实 14M 小样本固定状态测试。
4. 服务器健康四卡运行最小 DDP/no_sync 对照。
5. 为每层测试生成机器报告，包含用例数、失败、跳过、环境和提交。
6. 禁止用“GPU 不可用所以跳过”报告通过 Stage 2 进入 Gate。

## 4. 产出

- Stage 0/1 handoff manifest；
- Stage 1 exact commit/API/registry/provider/estimator 的验收与冻结 manifest；
- Stage 2 sampling、reference、runner、streaming reducer、profiler 和 report 适配层契约；
- 本机、服务器 CPU、单 GPU、四 GPU 分层测试报告；
- 旧实现迁移/拒绝复用说明。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.1 Stage 1 交接与固定状态 Gate**：

- Stage 0 handoff 没有过期或缺失证据；
- Stage 1 所列代数、尺度、加权、`M=2` 和分布式不变量全部通过；
- 14M 真实模型固定状态运行前后参数、buffer 和模型/全局 RNG 摘要一致；sampling/worker RNG 起止状态可重放；
- 参数注册表能检测 tied parameter 和异常 `grad=None`；
- signed U 负值可穿过计算、存储和报告全链路；
- 所有正式 GPU 测试实际执行，没有关键 skip；
- 测试报告绑定当前提交、环境和资产哈希。

若固定状态、loss reduction 或梯度尺度任一项失败，必须回到 Stage 1 修复；不得通过增加重复次数掩盖实现错误。

Stage 2 只允许改变抽样规模、reference/pilot/confirmatory 编排和统计/报告配置。formula、provider、registry、loss、DDP/AMP/clip 或 checkpoint schema 的实质变更必须形成 Stage 1 amendment，并重新通过受影响 Gate 后再更新本交接 manifest。

## 6. 后续依赖

- S2.4 使用本任务的 fixed-state provider 和参数注册表。
- S2.5 只能调用纯估计器接口，不能绕过状态检查直接操作模型。
- S2.7 的每个正式结果必须引用本任务的测试报告哈希。
