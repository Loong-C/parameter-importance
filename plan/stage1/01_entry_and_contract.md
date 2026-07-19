# S1.1 入口基线与数学契约冻结

## 1. 目的

在写参数重要性代码之前，先把当前运行条件、目标量、符号、损失语义和验收矩阵冻结，防止实现过程中出现“字段名相同但数学对象已变化”或“环境已漂移却沿用旧 Gate”的情况。

本子任务完成后，应能从一份机器可读契约回答：每个输出字段计算什么、使用哪一步的参数/梯度/学习率、哪些参数参与统计、在哪个环境上验证，以及失败时阻止哪个后续任务。

## 2. 前置条件

- 已阅读 `Agent/` 五份文档、`plan/general_plan.md` 和 `docs/mathematics.md`。
- 3.1 的入口快照采集保持只读，不修改正在下载的 Pile `.part`、锁和任务；后续获准的写入只落到 Git 工作区或授权的 `$DATA_ROOT` 项目路径。
- 当前数学文档的用户修正保留在工作树中，不回滚、不隐藏。

## 3. 执行步骤

### 3.1 重新采集入口快照

1. 记录本机分支、HEAD、工作树全部差异和远端分支位置。
2. 记录服务器分支、HEAD 和工作树状态。
3. 核对本机、GitHub 与服务器是否指向同一提交。
4. 在入口基线形成可审计提交后建立专用 Stage 1 分支，并记录分支基点。
5. 为本阶段建立中文 worklog，并冻结契约、骨架、CPU、单卡、DDP/数值、恢复和收尾等提交边界。
6. 分别列出本机与服务器 `Agent/` 文件集合。
7. 逐文件比较五份 `Agent/*.md` 的大小和 SHA-256。
8. 记录服务器挂载、剩余空间、inode、`DATA_ROOT` 权限和规定目录集合。
9. 验证 HF、Datasets、Torch、XDG 与 Matplotlib 缓存解析到 `$DATA_ROOT/cache` 下的项目目录。
10. 对适用路径同时冻结 CUDA driver、Triton 与 TorchInductor 编译缓存；若 correctness 配置禁用对应编译功能，则记录禁用证据。
11. 用入口写路径审计验证不会回退 `~/.cache`、`~/.nv` 或其他用户目录，未知写入目标 fail-closed。
12. 验证运行临时文件、torchrun rendezvous 与日志目录解析到 `$DATA_ROOT/tmp/stage1/<run_id>/`，不使用系统 `/tmp`。
13. 记录活动下载、训练或 GPU 进程，标记不得竞争或触碰的资源。
14. 对每张候选 GPU 进行最小 CUDA 上下文和分配检查。
15. 从健康候选中形成单卡候选集合和四卡候选集合，但不把物理编号写入代码。
16. 取得资源所有者或管理员对本次四卡 smoke 的确认，并在建进程组前再次检查外部占用；发现未知占用只退出，不终止对方进程。
17. 在 `$DATA_ROOT/operations` 下为本次 preflight 的 GPU UUID 建立带 heartbeat 的短期项目租约，并记录 UID、PID/PGID、启动时间、父子树/launcher 和 run token 指纹。
18. 对四卡候选重新执行最小 NCCL collective smoke；旧报告不能替代本次结果。
19. smoke 完成后立即释放本次租约；项目租约只防本项目并发，不代表集群所有权，S1.7/S1.8 启动前必须重新检查并重新取得。

### 3.2 复核环境与资产身份

1. 记录权威服务器 venv 的 Python、PyTorch、CUDA runtime、cuDNN、NCCL 和核心 Python 包版本。
2. 运行依赖完整性检查并保存结果摘要。
3. 核对 `environment/requirements.lock` 与现场环境的差异。
4. 核对 Pythia-14M step0 的固定 revision、文件集合、大小与 SHA-256。
5. 完全离线加载 14M 模型和 tokenizer，并验证一次有限值前向。
6. 在 import/provider 初始化前启用 HF、Transformers 与 Datasets offline 模式，并对模型/tokenizer 使用本地文件限定。
7. 启用进程级 socket/HTTP guard，验证 provider 初始化与前向不发起网络连接。
8. 核对 Pile 前缀 manifest 的 revision、覆盖范围和官方 reader 对比状态，并把 reader oracle 绑定到 Pythia source revision `a19eecb807ec2c79a39ebf18108816e6ffffc1d5` 及其归档哈希。
9. 选择仅落在已验证前缀中的 Stage 1 样本 ID 范围。
10. 验证选定范围不引用任何 `.part`、活动锁或未验收 shard。
11. 用无 BOM 和带 BOM 的 JSON manifest 各做一次读取检查。
12. 对缺字段、错误 revision 和错误哈希的 manifest fixture 验证 fail-closed 行为。

### 3.3 冻结数学对象和字段语义

1. 固定“正值表示对损失下降的正贡献”的唯一符号约定。
2. 固定 Stage 1 主目标为局部梯度空间贡献，不称为 AdamW 完整路径积分。
3. 固定 raw 为未裁剪 `eta * mean_grad^2`。
4. 若需要 clip-adjusted raw，使用独立字段，不覆盖 raw。
5. 固定双采样为两个独立数据流提供两个梯度因子。
6. 固定等权 U-statistic 和有效 token 加权 U-statistic 的公式。
7. 固定 clip factor 在全局平均梯度形成后计算，并只乘一次。
8. 固定 `signed`、`positive`、`negative_mass` 和 `absolute` 的定义与恒等关系。
9. 固定参数移动量主基线只累计数据驱动位移。
10. 把 total movement 与 weight-decay movement 标为诊断量。
11. 固定参数幅值和首尾净位移的采样时点。
12. 固定多学习率参数组的处理：逐参数组使用实际 step 学习率；无法可靠映射时直接拒绝运行。

### 3.4 冻结损失与统计单元

1. 固定 Stage 1 因果语言模型主损失为全局有效目标 token 平均。
2. 记录 loss numerator、有效目标 token 数和最终 mean loss。
3. 定义一个统计单元为一次独立 local microbatch backward。
4. 规定每个 microbatch 梯度是该 microbatch 有效 token mean gradient。
5. 规定全局平均梯度按有效目标 token 数加权。
6. 规定零有效 token 的 microbatch 不能静默进入统计。
7. 固定 dropout 和随机层在等价性 Gate 中关闭或使用可证明一致的随机状态。
8. 排除 BatchNorm、in-batch negatives、跨样本损失和共享随机增强 fixture。
9. 固定每个 optimizer step 内统计单元数量和样本 ID 的记录方式。

### 3.5 冻结 step 生命周期

1. 定义 `Theta_t` 为 optimizer step 前参数。
2. 定义本步使用的学习率为 optimizer 实际将消耗的学习率。
3. 明确何时读取未同步、已 unscale 的 local mean gradient。
4. 明确何时形成全局充分统计量和全局 mean gradient。
5. 明确何时计算 clip factor 与单步分数。
6. 明确何时累计 signed/positive/negative/absolute。
7. 明确何时调用 optimizer step。
8. 明确何时从前后参数和 optimizer 状态分解数据位移与 weight-decay 位移。
9. 规定 non-finite 或 GradScaler skip 的 attempt 不推进贡献、参数移动量和 `successful_optimizer_step`。
10. 区分 `attempt_step` 与 `successful_optimizer_step`：skip 仍消费当前 batch/RNG，推进数据游标、attempt、skip count 与 scaler，但不推进 successful step、scheduler、贡献或 movement。
11. 把 execute 或 skip 完成后的状态都定义为完整 training-attempt 事务边界；只在该边界发布 checkpoint。
12. 冻结有限/skip 两条 GradScaler 状态机，保证正式 inf-check、`scaler.step` 和 `scaler.update` 的次数可观测且所有 rank 一致。

### 3.6 冻结验收矩阵

1. 为每个数学字段分配唯一 requirement ID。
2. 为每个 requirement 指定实现模块。
3. 为每个 requirement 指定至少一个独立 oracle。
4. 为每个 requirement 指定必须运行的设备和 dtype。
5. 为每个比较指定 `README.md` 中的容差配置。
6. 为每个 Gate 指定机器可读 measured、threshold、pass/fail 字段。
7. 为每个失败指定最小复现包内容。
8. 冻结 Stage 1 不运行的统计与科学实验，防止范围漂移。
9. 为环境与 provenance 采集定义字段 allowlist，只保留版本、公开 revision、hash、路径类别和运行身份。
10. 明确排除 token、Cookie、认证头、签名 URL、SSH 配置和资产 manifest 中的传输端点。

## 4. 产出

- Stage 1 数学契约文件，包含 schema 版本、字段名、公式、单位和解释边界。
- 环境与资产入口快照 JSON。
- 参数组、学习率和 weight-decay 处理决策记录。
- loss reduction 与有效统计单元规范。
- requirement-to-test-to-artifact 追溯矩阵。
- 专用分支、里程碑提交与三端同步节奏记录。
- 缓存/tmp 解析和 GPU UUID 租约快照。
- 单卡和四卡候选的当次健康检查摘要。
- `G1-ENTRY` 与 `G1-CONTRACT` 机器可读 Gate 记录。

## 5. 可视化与呈现

- 本机/GitHub/服务器的 HEAD、工作树与 Agent 文件 hash 对照表。
- GPU UUID × CUDA 分配/NVML/NCCL/租约状态矩阵；故障设备只标记，不纳入候选。
- requirement → test → artifact → downstream Gate 的追溯矩阵或依赖图。
- 缓存、临时目录、checkpoint 与大型结果的路径归属表。

这些图表只呈现入口资格与追溯关系，不用于替代机器 Gate。

## 6. 核验标准

- 开发准备阶段允许保留已记录的用户修改；任何计入正式 Gate 的服务器运行，其执行代码、配置和数学契约必须已提交并完成本机、GitHub、服务器同一 HEAD 同步，dirty executable diff 直接阻塞。
- `Agent/` 文件集合必须恰为规定的五份，且逐文件 SHA-256 一致。
- 四卡候选中的每张卡必须通过单卡 CUDA 检查，组合必须通过本次 NCCL smoke。
- 缓存、临时目录、torchrun rendezvous/log 和 GPU 租约必须全部落在预注册的 `$DATA_ROOT` 项目路径，且租约对应本次 UUID 与 run-owned PID/PGID。
- 适用的 CUDA/Triton/TorchInductor 编译缓存不得写入用户目录；禁用编译路径时必须有写路径审计证据。
- preflight/smoke 租约带 heartbeat 与完整进程指纹，并在任务结束立即释放；它不能替代资源所有者确认。
- 14M 模型和固定调试数据必须具有 revision、大小、哈希与离线加载证据。
- offline 模式、本地文件限定和进程级网络 guard 共同证明 provider 没有外部访问。
- 每个公开输出字段必须在契约中有唯一公式；不得存在含混的 `raw`、`negative` 或 `movement`。
- 每个配置字段必须映射到一个可观察行为和测试；不接受“声明但未驱动代码”的开关。
- 容差、seed、dtype、world size 和样本范围必须在看到正式结果前冻结。
- 环境与 provenance 输出只能包含 allowlist 字段，不得泄露凭据或传输端点。
- 任一入口事实未知时，Gate 状态只能是 `blocked` 或 `failed`，不能写成 `pass`。

## 7. Gate 与后续依赖

- `G1-ENTRY` 通过后，才允许服务器上的正式实现与 GPU 验证进入验收计数。
- `G1-CONTRACT` 通过后，S1.2–S1.6 才能冻结 API 与输出 schema。
- GPU 健康问题不阻止本机编写纯 CPU 模块，但 S1.8 和 `G1-EXIT` 始终被阻塞。
- 若后续修改数学契约、loss reduction、学习率语义或参数范围，必须回到本子任务升级 schema 并重跑全部受影响 Gate。
