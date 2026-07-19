# 第 0 阶段：环境配置与实验基础设施建设计划

## 1. 文档定位

本目录把 [`general_plan.md`](../general_plan.md) 中的第 0 阶段展开为可执行、可核验、可追溯的子任务。这里描述的是实施计划，不代表对应环境、代码或资产已经全部完成。

计划依据包括：

- `Agent/` 下五份现行运维文档；
- `plan/general_plan.md` 的阶段目标和后续阶段依赖；
- 2026-07-18 对本机、GitHub、lab-pc 链路和 sophgo13 服务器进行的只读核验；
- `worklogs/` 中已有的环境准备、资产清理和同步记录。

所有“当前状态”都是 2026-07-18 的快照。正式执行任一子任务时必须重新采集状态，不能把本计划中的快照直接当作验收证据。

## 2. 阶段目标

第 0 阶段要建立一套能够支撑后续 Stage 1–9 的实验底座，使任何正式实验都满足以下条件：

- 能在明确的本机与服务器职责边界内运行；
- 能从固定的软件、模型和数据身份重建；
- 能在单卡和四卡上正确执行；
- 能解释 DDP、梯度累积和 `no_sync` 的数值语义；
- 能生成不覆盖、可关联，并且不会因部分写入或已支持的进程/SSH 中断而丢失的日志、checkpoint 和结果；介质级故障按 G1-D 的获授权备份或显式风险接受处理；
- 能从中断点恢复，并证明恢复前后的训练状态连续；
- 能把每个结果追溯到代码、配置、seed、环境、模型、数据和输入 checkpoint；
- 能通过统一 gate 阻止不满足条件的实验进入下一阶段。

第 0 阶段只证明“实验基础设施可信”，不证明参数重要性算法正确。参数重要性公式、估计器和路径积分实现的正确性仍属于 Stage 1–3。

## 3. 已核实的当前基线

### 3.1 本机定位

- 系统为 Windows 10 64 位，CPU 为 4 核 8 线程的 Intel i7-1165G7，内存约 23.7 GiB。
- 本机没有 NVIDIA GPU；当前只有集成显卡，因此不得把本机作为 CUDA、NCCL 或性能结论的验证端。
- 本机已有 Python 3.12.4、Git 和 OpenSSH；没有 Conda、Mamba 或 uv。
- 本机 Python 当前安装 CPU 版 PyTorch 2.10.0，但没有 Transformers、Datasets 或 Accelerate。
- 本机应承担代码编辑、轻量单元测试、配置检查、文档、Git、SSH 调度和小型结果审阅；大型模型、数据和 GPU 运行不在本机保存或执行。

### 3.2 仓库与同步基线

- 本机、GitHub `main` 和服务器 `main` 当前均指向提交 `34966d0819a5229569169bfe436afe9058c0ed24`。
- 服务器工作树干净；本机已有一处用户修改：`docs/mathematics.md` 修复了一个公式字符并增加文件末尾换行。后续工作必须保留、审查并按 `Agent/git.md` 处理该修改。
- `Agent/` 两端均有规定的五个文件，其中四个文件哈希一致；`Agent/worklogs.md` 内容不一致。该同步债务必须在环境实施前解决并记录，未解决时不能通过基线 gate。
- 当前 `main` 只保留文档、依赖声明、工作日志和旧实验轻量归档；现行训练源代码、测试、运行入口和配置骨架尚未建立。

### 3.3 服务器硬件与存储基线

- 系统为 Ubuntu 24.04.3 LTS，128 个逻辑 CPU，内存约 1 TiB。
- PCI/驱动目录可见 8 张 NVIDIA A100-SXM4-80GB，但 NVML 与 PyTorch 当前只枚举 7 张，驱动为 575.57.08。缺失设备必须由服务器管理员诊断；Stage 0 不得自行重载驱动、重置设备或绕过该异常。
- 当前被枚举的 7 张 GPU 没有计算进程，但可见 GPU 0 累计了 24 次易失性和聚合不可纠正 ECC 错误。因此“空闲”不等于“健康”，旧的 8 卡验收与当前任何四卡候选都必须作废后重验。
- 当前拓扑快照中 GPU 3–6 位于同一 NUMA 节点并避免跨 CPU socket；这只能作为故障排除后的候选组线索，不能在 GPU 枚举/ECC gate 通过前用于正式实验。
- 当前 NVLink 状态未显示活动链路，拓扑主要表现为 PCIe 路径。四卡通信性能必须重新建立实测基线，不能沿用对 NVLink/NVSwitch 的假设。
- 项目大盘为可读写 ext4，约 3.5 TiB，总可用空间约 2.9 TiB，inode 使用率约 1%。
- 系统根盘约 98 GiB，当前只余约 15 GiB；系统级根盘不得承载项目环境、缓存、编译目录或运行产物。
- 当前进程文件描述符软上限为 1024，需在数据加载并发测试中验证是否足够；不得未经必要性验证修改系统级限制。
- `DATA_ROOT` 下规定的 `datasets`、`models`、`cache`、`checkpoints`、`runs`、`results`、`reports`、`manifests`、`operations`、`wheelhouse`、`envs`、`source` 和 `tmp` 均已存在。

### 3.4 服务器软件栈基线

- 现有项目虚拟环境位于 `$DATA_ROOT/envs/parameter-importance`，Python 为 3.12.3。
- 核心版本为 PyTorch 2.12.1+cu126、CUDA runtime 12.6、cuDNN 9.10.2、NCCL 2.29.3、Transformers 4.57.6、Datasets 4.8.5、Accelerate 1.14.0、TensorBoard 2.21.0。
- 当前 `pip check` 通过，现场 `pip freeze` 与已有 `manifests/pip-freeze.txt` 哈希一致。
- 系统侧存在 CUDA 12.9 工具链和 `nvcc`，但 `nvcc` 不在默认 PATH；PyTorch wheel 使用的是 CUDA 12.6 runtime。计划必须分别记录驱动能力、系统 toolkit 和 PyTorch runtime，不能把三者混成一个“CUDA 版本”。只有明确需要编译自定义 CUDA 扩展时，才启用独立编译兼容性 gate，且不得使用 `sudo` 修改系统环境。
- DeepSpeed 和 Weights & Biases 当前未安装。基于模型规模、离线要求和参数重要性对梯度语义的敏感性，Stage 0 默认采用原生 PyTorch DDP/`torchrun`，保留 Accelerate，暂不引入 DeepSpeed；追踪采用结构化本地日志加 TensorBoard，暂不依赖 W&B。

### 3.5 模型与数据基线

- 已存在模型目录的资产包括 Pythia 14M step0、31M deduped step0、160M deduped step0 和 160M deduped step512；其中当前 31M 顶层 manifest 无法通过 JSON 解析，修复并重验前不能把该资产标记为 gate 通过。
- Pythia 410M 的统一初始化资产当前缺失。
- 已存在 SST-2、WikiText-103 raw 和 Pythia 预打乱 Pile 数据；MNLI 和 RTE 当前缺失。
- Pile 的 shard 0–4 已有完整文件，shard 5 仍以 `.part` 形式由既有受控下载链路写入。不得读取、移动、改名或另起下载任务竞争该文件。
- 2026-07-13 的 `READY` 只证明当时定义的最小前缀、环境和离线 smoke test 通过。此后内核版本、GPU 枚举与健康状态已经漂移，故旧 `READY` 不能作为当前环境 gate；它也不证明后续新增 shard、410M、MNLI 或 RTE 已就绪。

### 3.6 当前 gate 判定

- **G0-C 当前不通过**：`Agent/worklogs.md` 两端哈希不一致；访问、路径和三端 Git 基线本身已完成只读确认。
- **G0-G 当前不通过**：GPU 8/7 枚举不一致且有不可纠正 ECC。
- **G0 总 gate 当前不通过**：必须同时修复并复验 G0-C 与 G0-G。
- **G1 尚待本轮复验**：G1-B 的大盘空间与权限快照良好，但仍需完成写入 canary、根盘保护和生命周期规则；G1-D 的第二故障域备份或用户风险接受也尚未作出本轮决定。
- **G2 为可复用候选、尚未通过**：现有 venv/锁/freeze 一致，但尚未完成本轮非破坏性离线重建和硬件恢复后的 GPU 重验。
- **G3 部分通过、整体不通过**：已有若干资产，但 31M manifest、410M、MNLI、RTE 和预训练数据预算仍未闭环。
- **G4–G9 尚未建立现行实现与报告**：当前 `main` 没有对应代码和测试。
- **G10 当前不通过**：必须等全部前置 gate、工作日志、三端 Git 和五份 `Agent` 哈希统一后才能验收。

## 4. 技术路线决策

- **本机/服务器分工**：本机是控制与开发端；服务器是 Linux/CUDA 权威运行端，`DATA_ROOT` 保存大型资产的权威运行副本；只有 G1-D 明确授权时才建立独立备份副本。
- **环境策略**：先复核并复用现有服务器环境；只有复核失败或依赖方案改变时才在新路径重建，避免直接破坏已验收环境。
- **分布式策略**：以原生 PyTorch DDP 为数值语义基准，使用显式四卡列表，不在代码中硬编码物理 GPU 编号。
- **精度策略**：功能 smoke test 覆盖 BF16；数值等价和梯度语义 gate 使用小模型、固定数据和 FP32，以降低混合精度和归约顺序造成的噪声。
- **追踪策略**：JSONL/JSON 作为机器可读真值，TensorBoard 作为可视化派生层，控制台日志作为诊断层。
- **配置策略**：所有入口共享同一配置 schema；每次运行必须保存解析后的不可变配置和配置摘要哈希。
- **资产策略**：先冻结每个阶段的最大数据游标和模型清单，再判断现有资产是否覆盖；不以“目录存在”或旧 `READY` 代替逐资产 gate。
- **恢复策略**：checkpoint 使用完整状态、临时写入和原子发布；只有完整 checkpoint 可被恢复入口发现。

### 4.1 统一数值与性能测量协议

凡是使用“下降、稳定、开销、p95、变异系数、scaling、显存峰值或性能回归”作 gate 的测试，都必须采用同一份版本化测量协议，不能临场挑选窗口：

1. 每份报告固定并记录 fixture ID、resolved config 摘要、Git 提交/脏补丁摘要、environment ID、资产 ID、设备 UUID/拓扑、测量协议版本和重复编号。
2. 每次重复开始和结束都记录 CPU、GPU、内存、磁盘、网络、目标 GPU 外部进程和项目活动下载快照。既有 Pile 下载或其他负载只要竞争本次测量涉及的资源，该重复即无效并延后；不能把“没有其他 GPU 进程”当作无竞争的充分条件。
3. 训练吞吐与日志开销测试每次先执行 10 个 optimizer warmup step，再测量 30 个 optimizer step；以三个全新进程的独立重复为一组。每个重复报告中位数、p95，组报告使用三个重复中位数的中位数，并报告其变异系数。
4. NCCL 每个消息规模先执行 20 次 warmup，再测量 50 次 collective；进程组重建三次。延迟报告中位数、p95 和重复间变异系数，吞吐使用相同样本计算。
5. 显存测试在 5 个 warmup step 后重置峰值计数，再测量 20 个 step；做三个全新进程重复。容量 gate 使用三次中观测到的最高峰值，而不是较乐观的中位数。
6. loss 下降使用固定的极小过拟合 fixture，连续运行 50 个 optimizer step 并重复三次；每次最后 10 步 loss 中位数至少比最初 10 步低 5%，且全程 loss、梯度和更新均为有限值。
7. 日志开销采用同一配置的成对 A/B 测试：最小真值日志与正式追踪交替执行，各做三次，按两组吞吐中位数计算相对开销。
8. correctness、无重复/遗漏、哈希、schema、数值容差和安全边界属于硬标准，不允许豁免。仅性能/容量阈值可在优化无效后由用户和相应资源管理员书面批准例外；例外必须记录原因、替代保护、适用配置和失效日期，单项状态记为 `APPROVED_EXCEPTION` 而不是 `PASS`。gate 汇总器只能把预先声明为可例外的性能/容量项计作“带例外满足”，并必须在总结果中显著保留该状态。
9. 代码、配置、依赖锁、驱动、内核、GPU 白名单/拓扑或资产身份变化时，受影响的测量组自动失效并重跑。

## 5. 子任务与执行顺序

1. [S0.1 基线、安全边界与多端状态](01_baseline_and_safety.md)
2. [S0.2 存储布局与目录生命周期](02_storage_and_layout.md)
3. [S0.3 运行时、依赖锁定与环境重建](03_runtime_and_dependencies.md)
4. [S0.4 模型、数据与 manifest](04_assets_and_manifests.md)
5. [S0.5 配置、实验编号、seed 与 provenance](05_config_run_identity_and_seeds.md)
6. [S0.6 单卡训练 smoke test](06_single_gpu_smoke.md)
7. [S0.7 四卡 DDP、梯度累积与 `no_sync`](07_ddp_and_gradient_semantics.md)
8. [S0.8 日志、追踪与运行状态](08_logging_and_tracking.md)
9. [S0.9 checkpoint 与断点续训](09_checkpoint_and_resume.md)
10. [S0.10 容量评估、运行保护与故障处理](10_capacity_and_operations.md)
11. [S0.11 自动测试、端到端重放与交接文档](11_test_quality_and_replay.md)
12. [S0.12 交付、工作日志与多端同步](12_delivery_and_sync.md)

依赖关系按“接口可先实现、集成验收需等前置 gate”的原则执行：

| 子任务 | 可以开始的条件 | 完成验收的额外条件 |
|---|---|---|
| S0.2 | G0-C | G1-B 存储机制与 G1-D 持久性决策都完成验收后才形成 G1 |
| S0.3 | G0-C、G1 | G2 最终验收还需 G0-G 后的逐白名单 GPU 核心导入/分配重验 |
| S0.4 | G0-C、G1 | 各阶段资产分别通过 G3 子 gate |
| S0.5 | G0-C、G1；S0.4 的逻辑资产 ID/schema 已冻结 | provenance 集成还需稳定的环境与资产摘要 |
| S0.6 | G0-G、G2、G3-S1、G4 | 日志需接入 S0.8 事件接口；最小 checkpoint 合同由 S0.6 自身定义，S0.9 再扩展 |
| S0.7 | G0-G、G5 | 无 |
| S0.8 | G1、G4 后可先实现接口 | 单卡、四卡、失败与恢复日志需 G5、G6、S0.9 |
| S0.9 | G1、G4、G5、S0.8 的事件 schema/状态接口 | 四卡恢复部分还需 G6；不要求 S0.8 已完成全部集成验收 |
| S0.10 | G0-G、G5、G6、G7 | 160M 实测还需 G3-S4；410M 实测还需 G3-S5 |
| S0.11 | G0–G8 | 无 |
| S0.12 | G0–G9 | 无 |

S0.4 的资产下载可在 S0.3–S0.9 的代码建设期间并行，但任何未通过资产 gate 的文件都不能进入 smoke test 或正式实验。

### 5.1 `general_plan.md` 要求追踪表

| 第 0 阶段要求 | 主要子任务 | 必需产出 | 验收 gate |
|---|---|---|---|
| 环境说明、精确锁定与重建 | S0.2、S0.3 | 路径规范、锁文件、wheelhouse 清单、离线重建报告、environment ID | G1、G2 |
| 模型/数据获取、固定 revision 与校验 | S0.4 | 资产获取入口、manifest、离线加载及拒绝测试 | G3 各子 gate |
| 配置管理与实验编号 | S0.5 | schema、resolved config、experiment/run/attempt/session 身份与 provenance | G4 |
| seed 管理 | S0.5、S0.11 | seed 派生合同、跨 rank/worker 规则、确定性测试 | G4、G9 |
| 日志、TensorBoard 与运行状态 | S0.8 | JSONL 真值、派生视图、状态机、心跳、开销报告 | G7、G8-C |
| 单卡训练闭环 | S0.6 | FP32/BF16 smoke、数据边界、最小 checkpoint 报告 | G5 |
| 四卡 DDP、累积与 `no_sync` | S0.7 | NCCL、分片、全局分母、梯度等价和失败退出报告 | G6 |
| checkpoint 与断点续训 | S0.9 | 完整状态合同、原子保存、单/四卡精确恢复和 canonical lineage | G7 |
| 显存/内存/磁盘/吞吐评估 | S0.10 | 分模型预算、实测基线、运行保护和故障 runbook | G8-C、G8-S4、G8-S5 |
| 自动测试、运行说明与独立重放 | S0.11 | 分层测试、统一报告、运行说明、独立离线重放 | G9 |
| 日志、提交与多端交付 | S0.12 | 中文工作日志、交付清单、三端 Git 与 `Agent/` 哈希证据 | G10 |

## 6. Gate 总览

Gate 状态统一为 `PASS`、`CONDITIONALLY_ACCEPTED`、`FAIL`、`BLOCKED`、`STALE`。硬正确性/安全项只能以 `PASS` 满足；预先声明可例外的性能/容量单项在批准后记为 `APPROVED_EXCEPTION`，使所属 gate 汇总为 `CONDITIONALLY_ACCEPTED` 而非 `PASS`。只要存在 `FAIL/BLOCKED/STALE` 就不得推进；存在条件接受时，阶段可在批准范围内推进，但最终 readiness 必须命名并标记为 `READY_WITH_APPROVED_EXCEPTIONS`，列出范围和失效日期，不能发布普通 `READY`。

- **G0-C 基线与安全子 gate**：远程身份、允许写入路径、仓库起点和用户改动均已确认；`Agent/` 同步债务已解决且两端哈希重新对齐。
- **G0-G GPU 健康子 gate**：GPU 的物理/驱动、NVML 与 PyTorch 枚举一致，ECC/设备健康由管理员处理并完成本轮重验。
- **G0 基线总 gate**：只有 G0-C 与 G0-G 同时通过才成立；G0-G 阻塞时只允许继续明确标注的非 GPU 子任务。
- **G1 存储 gate**：由 **G1-B 存储机制子 gate** 与 **G1-D 持久性决策子 gate** 共同组成。G1-B 要求挂载、空间、inode、权限、目录用途、保留策略和 Git 大文件守卫通过；G1-D 要求不可再生产物已有获授权的异盘备份/恢复方案，或用户对单盘数据丢失风险作出有范围和期限的明确接受。两者都满足后 G1 才成立，下载与训练才可开始。
- **G2 环境 gate**：锁文件、wheelhouse、venv 和现场 freeze 一致，从空候选环境进行的受控离线重建通过；G0-G 解除后，还需在每张白名单 GPU 上完成核心 CUDA 导入、设备分配与最小张量检查，才能最终形成 G2。
- **G3 资产 gate**：每个模型/数据集具有固定 revision、大小、哈希、split/游标覆盖和离线加载证据；`.part`/活动锁永不作为输入。
- **G4 配置与 provenance gate**：配置验证、规范化、run ID、seed 派生和运行清单稳定且无覆盖。
- **G5 单卡 gate**：单卡前向、反向、优化器、BF16、日志和最小 checkpoint 全部通过。
- **G6 分布式语义 gate**：四卡通信、数据划分、DDP 等价、梯度累积和 `no_sync` 等价通过，异常退出无残留。
- **G7 可恢复性 gate**：日志连续、checkpoint 完整、恢复轨迹与不中断轨迹在预定容差内一致。
- **G8 容量与运维 gate**：显存、内存、磁盘、吞吐、保存频率和失败预算有证据，四卡候选在运行前可安全取得。G8 由基础设施与 synthetic shape fixture 的 **G8-C**、依赖 G3-S4 的 160M 实测 **G8-S4**、依赖 G3-S5 的 410M 实测 **G8-S5** 共同组成；三者都为 `PASS` 时 G8 为 `PASS`，仅含获批性能/容量例外且无失败时为 `CONDITIONALLY_ACCEPTED`。
- **G9 重放 gate**：新会话仅依靠仓库文档和 manifest 能离线重放单卡、四卡和恢复 smoke test。
- **G10 最终同步 gate**：工作日志完整，本机/GitHub/服务器同提交，`Agent/` 五文件两端哈希一致；大型资产的权威运行副本只放 `DATA_ROOT`，G1-D 明确授权的备份副本不受此限制。

只有 G0–G10 的硬项全部 `PASS`，且每个 gate 为 `PASS` 或按上述规则 `CONDITIONALLY_ACCEPTED`，才能宣称 Stage 0 完成并进入 Stage 1；本计划不设置跳过大型资产、硬件或正确性 gate 的“核心设施”例外。

## 7. 后续阶段的资产与基础设施前置条件

除下面列出的 Stage 0 专项条件外，每个阶段还必须先通过 `general_plan.md` 规定的全部前序阶段 gate：

- **进入 Stage 1**：G0–G10 按统一状态规则满足；仍有效的 G3-S1、Pythia 14M、固定调试数据和单卡/四卡梯度证据可用；若有条件接受，交接必须携带完整例外清单。
- **进入 Stage 2**：Stage 1 先通过；Pythia 31M 和重复采样数据清单通过仍有效的 G3-S2，运行系统能记录峰值显存和 wall-clock。
- **进入 Stage 3**：Stage 2 先通过；checkpoint 能唯一关联更新前后状态、优化器状态和数据位置。
- **进入 Stage 4**：Stage 3 先通过；仍有效的 G3-S4、Pythia 160M 统一初始化、预训练数据游标覆盖、SST-2、在线统计和恢复 gate、G8-S4 以及仍有效的 G1-D 通过。
- **进入 Stage 5**：Stage 4 完整闭环先通过；仍有效的 G3-S5、Pythia 410M 统一初始化、正式数据预算、G8-S5、高频 checkpoint 容量 gate 以及仍有效的 G1-D 通过。
- **进入 Stage 6**：Stage 5 先通过；仍有效的 G3-S6 覆盖 SST-2、MNLI、RTE 和统一预处理。直接监督路线的 `input_checkpoint_id` 指向共同 base initialization 的实际 checkpoint/权重对象，并另记相同的 `base_initialization_id`；微调路线的 `input_checkpoint_id` 指向由该 `base_initialization_id` 训练得到的预训练 checkpoint，两个不同类型的 ID 不直接互相赋值。
- **进入 Stage 7**：Stage 6 先通过；配置系统支持独立训练 seed、采样 seed 和 mask seed，并能生成父子实验关系且不覆盖结果。
- **进入 Stage 8**：Stage 7 先通过；全部消融变体能继承并校验对应父实验身份。
- **进入 Stage 9**：Stage 8 先通过；所有原始与派生结果均可通过 run ID、配置摘要和 manifest 反向追溯。

## 8. 不在第 0 阶段完成的内容

- 参数重要性公式、raw/双采样/U-statistic 的正式实现与正确性证明；
- 路径积分近似方法的比较；
- 160M 或 410M 的正式训练结论；
- 正式剪枝、消融和论文统计分析；
- 系统级驱动升级、系统 CUDA 安装、DNS/SSH 拓扑修改或任何需要 `sudo` 的改造。
