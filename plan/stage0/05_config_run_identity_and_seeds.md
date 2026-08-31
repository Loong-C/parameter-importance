# S0.5 配置、实验编号、seed 与 provenance

## 目的

建立所有训练、评估、分析和 smoke test 共用的配置与身份系统，使任何运行都能被唯一识别、可靠比较和完整追溯，并防止参数缺省、seed 混用和重复运行覆盖结果。

## 当前依据

- 当前 `main` 尚无现行配置系统或训练入口。
- 后续实验需要系统改变模型规模、batch、microbatch、积分节点、checkpoint 频率、训练路线和多个 seed。
- Stage 1 需要精确比较单卡/多卡和不同梯度构造；隐式默认值会直接污染结论。
- 旧归档中有 resolved config 和 provenance 形式的历史证据，可审计其字段，但不能直接把旧代码视为当前实现。

## 前置条件

- G0-C 已记录仓库基线、用户修改和两端运维文档状态。
- G1 已固定运行目录及大型资产位置。
- 模型和数据资产可以通过逻辑 ID 引用，配置不直接依赖临时绝对路径。

## 实施步骤

### 1. 定义配置域

1. 为模型身份、初始化和精度建立独立配置段。
2. 为数据资产、split、序列长度和 sampler 建立独立配置段。
3. 为 optimizer、scheduler、训练步数和 batch 语义建立独立配置段。
4. 为分布式 world size、设备列表、backend 和超时建立独立配置段。
5. 为梯度累积、`no_sync` 和 loss reduction 建立显式字段。
6. 为日志、TensorBoard、checkpoint 和保留策略建立独立配置段。
7. 为环境、输出根目录和离线模式建立运行时配置段。
8. 为后续参数重要性、剪枝和分析预留命名空间，但不在 Stage 0 实现算法。

### 2. 实现加载与合并规则

1. 定义基础模板、阶段模板和单次运行覆盖的优先级。
2. 限制覆盖只能修改 schema 中已声明的字段。
3. 对未知字段、重复字段和错误类型立即失败。
4. 对命令行或外部覆盖保留原始值和解析后值。
5. 禁止同一语义在多个字段中重复表达。
6. 在启动前生成完整 resolved config，不在运行中悄悄改变配置。

### 3. 实现跨字段验证

1. 验证 global batch、world size、per-device batch 和 accumulation steps 的数学关系。
2. 验证 `no_sync` 只在 accumulation steps 大于 1 时启用。
3. 验证所选精度受目标 GPU 支持。
4. 验证模型、tokenizer 和数据资产的兼容关系。
5. 验证训练游标不超过数据 manifest 的覆盖范围。
6. 验证 checkpoint 保存频率与最大保留数能满足容量预算。
7. 验证正式运行不能指向 `.part`、临时目录或未知资产 ID。
8. 验证四卡任务明确提供四个可见设备，而不是依赖全机枚举顺序。

### 4. 建立配置规范化与摘要

1. 对字段顺序、数字、布尔值、空值和路径表示做规范化。
2. 将逻辑资产 ID 与稳定 revision 纳入摘要，不把易变缓存绝对路径纳入。
3. 将所有有效默认值展开进 resolved config。
4. 对规范化结果计算稳定摘要。
5. 验证同一语义配置无论输入字段顺序如何都得到同一摘要。
6. 验证任一影响实验语义的字段变化都会改变摘要。

### 5. 定义 experiment、run、attempt 和 session 身份

1. `experiment_id` 表示语义实验，由阶段、任务、模型规模、路线、主 seed 和完整配置摘要确定，不包含时间。
2. `run_id` 表示一条逻辑训练轨迹，由 `experiment_id`、创建时间和防碰撞短码组成；同一语义配置的独立重复使用不同 run ID。
3. 新 run 只能原子创建一次；目录已存在时创建失败。
4. `attempt_id` 表示一次操作者启动或重试，在同一 run 内单调增加。
5. `session_id` 表示一次连续进程组/日志写入期；当前不使用弹性自动重启，因此每个 attempt 恰有一个 session，未来启用弹性时才允许一对多。
6. 从 checkpoint 恢复保留原 run ID，新建 attempt/session，并从已验证 checkpoint 继续。
7. 从初始状态重新做独立重复不属于恢复，必须创建新 run ID。
8. 超参数扫描用父 experiment ID 关联每个子 experiment/run。
9. 派生分析和剪枝运行记录输入 run/checkpoint ID。

### 6. 定义 seed 体系

1. 设置一个主 seed 作为配置入口。
2. 稳定派生模型初始化 seed，所有 DDP rank 使用同一初始化身份。
3. 稳定派生 sampler seed，并记录 epoch/step 对 sampler 的推进规则。
4. 稳定派生每 rank 随机流，用于训练随机算子；数值等价测试关闭 dropout 等额外随机性。
5. 为数据增强、参数重要性采样、剪枝 mask 和统计 bootstrap 分配独立 seed 域。
6. 记录 Python、NumPy、PyTorch CPU 和每张 CUDA 设备的 seed 状态。
7. checkpoint 保存并恢复所有相关 RNG 状态。
8. 禁止用同一个未区分用途的随机流同时驱动训练数据和剪枝 mask。

### 7. 定义 provenance

1. 记录 experiment ID、run ID、attempt ID、session ID、父实验和输入 run/checkpoint。
2. 记录 Git 提交、分支和工作树状态。
3. 正式实验默认拒绝脏工作树；开发 smoke test 若允许，必须保存完整可应用补丁、补丁 SHA-256、基线提交和显式脏状态，不能只保存 diff 摘要。
4. 记录 resolved config 及其摘要。
5. 记录 environment ID、硬件快照引用和所选 GPU 映射。
6. 记录模型、tokenizer、数据 manifest ID 和哈希。
7. 记录所有 seed 域及每 rank 派生结果。
8. 记录启动时间、结束时间、状态和恢复来源。
9. provenance 使用 schema 验证并在运行开始时先写入最小版本，结束时原子补全。

### 8. 建立配置差异工具

1. 对两个 resolved config 输出语义差异。
2. 区分会影响数值结果的字段和纯展示字段。
3. 支持检查扫描是否只改变预期的单一因素。
4. 支持检查直接监督输入的 `base_initialization_id` 与预训练轨迹起点一致，并验证微调输入 checkpoint 确实属于该预训练轨迹。
5. 将差异结果作为 Stage 6–8 实验审查证据。

## 产出

- 配置 schema、模板、加载器、合并器和跨字段验证器；
- resolved config 规范和稳定摘要算法；
- experiment/run/attempt/session、父子实验和恢复关系规范；
- 多域 seed 规则和 RNG 状态接口；
- provenance schema、写入器和差异工具；
- 配置与 seed 单元测试。

## 核验标准

- 未知字段、错误类型、非法 batch 关系、未知资产和输出目录碰撞都能在启动前失败。
- 相同语义配置得到相同摘要；任何影响实验结果的字段变化都改变摘要。
- 同一 run ID 不能被第二次创建并覆盖；恢复只增加 attempt/session，独立重复必须创建新 run。
- 同一主 seed 两次运行得到相同模型初始化哈希和调试数据顺序。
- 改变数据 seed 不改变模型初始化；改变 mask seed 不改变训练数据顺序。
- DDP 各 rank 的模型初始化一致，sampler 分片可重建，每 rank seed 均有记录。
- 正式运行的 provenance 能追溯到 Git、配置、环境、设备、模型、数据、seed 和输入 checkpoint；允许的脏 smoke 能从基线提交和完整补丁重建代码。
- 配置差异工具能证明单因素实验只改变预期字段。

## Gate 与后续依赖

- 本子任务通过形成 **G4 配置与 provenance gate**。
- S0.6–S0.11 的所有 smoke test 和报告必须使用这套配置与 run ID，不能另写临时参数通道。
- Stage 6 的统一初始化比较、Stage 8 的消融和 Stage 9 的追溯直接依赖 G4。
