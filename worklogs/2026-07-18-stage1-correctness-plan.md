# 2026-07-18 Stage 1 参数重要性代码正确性验证计划

- 任务范围：通读 `Agent/`、`plan/general_plan.md` 和相关数学规格，核对本机/服务器/旧归档现状，把 Stage 1 展开为可执行、可核验、可追溯的计划；不实现代码、不运行正式实验。
- 当前状态：计划定稿、三路交叉审查和结构核验完成；因工作树存在其他并行计划修改，本任务未擅自统一提交或同步。
- 工作分支：`main`

## 2026-07-18 21:23 CST — 完成现状核对与计划初稿

### 目标与范围

- 为每个 Stage 1 子任务分别说明目的、原子化步骤、产物、可视化、核验标准和前后置 Gate。
- 以 `docs/mathematics.md` 为公式、符号、损失 reduction、U-statistic、DDP、clip、AdamW 和累计量语义的权威依据。
- 根据当前本机 CPU-only 环境、服务器锁定 venv、GPU 故障、已验收 14M/Pile 前缀和当前空代码基线制定实施顺序。
- 不把旧归档测试和实验结果当作新实现的通过证据。
- 不修改服务器文件、GPU 状态、活动 Pile 下载、旧归档或用户现有数学文档内容。

### 实际修改

- 新建 `plan/stage1/README.md`，记录目标、范围、当前基线、方法命名、容差、依赖图、Gate 和下游交接。
- 在 `plan/stage1/` 新建 11 个子任务文件，分别覆盖入口契约、代码架构、fixture/oracle、梯度尺度、估计器、训练累计、14M 单卡、DDP、数值边界、恢复与最终报告。
- 在 `plan/general_plan.md` 的 Stage 1 节增加详细计划入口。
- 保留用户对 `docs/mathematics.md` 的未提交公式字符修正与末尾换行，不覆盖、不还原。
- 检测到工作树中还有其他并行任务形成的 Stage 0、Stage 2、Stage 3 计划与 Stage 0 日志；本任务未修改这些文件，也未抢先提交或同步整个工作树。

### 只读现状核对

- 本机：Python 3.12.4、PyTorch 2.10.0 CPU-only；无 Transformers/Datasets/Accelerate 和 CUDA。
- 服务器：`main`/`34966d0819a5229569169bfe436afe9058c0ed24`，工作树干净；权威 venv 为 Python 3.12.3、PyTorch 2.12.1+cu126、Transformers 4.57.6、Datasets 4.8.5，实时 `pip check` 通过。
- GPU：NVML/PyTorch 只枚举 7 张 A100，其中 1 张实际 CUDA 分配失败；另 1 张设备痕迹不被 NVML 枚举。当前只确认 6 张可分配，旧四卡 NCCL 报告不能作为现状 Gate。
- 资产：Pythia-14M step0 和 24 项核心资产实时哈希通过；Pile 已验证前缀覆盖 524,288 个样本，shard 5 仍为受控活动 `.part`；31M 权重与离线前向通过但 JSON manifest 带 UTF-8 BOM。
- 同步：`Agent/` 两端文件集合一致，前四份 SHA-256 一致，`worklogs.md` 不一致，因此当前不能宣称 Agent 文档同步完成。
- 代码：现行 `main` 没有 `src/`、`tests/`、配置或打包骨架；旧归档明确排除旧源码、测试、checkpoint 和大张量。

### 历史风险转化

- 新计划明确区分 raw 未裁剪与可选 raw-clipped。
- negative contribution 固定为非负 `negative_mass`，并要求同时保存 signed、positive 和 absolute。
- 参数移动量主基线只使用 data-driven update，不混入 decoupled weight decay。
- 新实现必须支持有效 token 加权 U-statistic。
- 真实四卡 Gate 使用 NCCL/DDP/`no_sync`，不以旧 CPU Gloo 多进程测试替代。
- 所有公开配置字段必须有行为测试，避免旧实现“配置声明但不生效”。

### 验证

| 项目 | 方法 | 当前结果 | 证据路径 |
|---|---|---|---|
| Agent 文档阅读 | UTF-8 全文读取五份文件 | 完成 | `Agent/` |
| 总计划阅读 | 全文读取 | 完成 | `plan/general_plan.md` |
| 数学口径 | 阅读相关定义、估计器、DDP、累计、算法和不变量章节 | 完成 | `docs/mathematics.md` |
| 本机现状 | Python 与模块只读探测 | 完成 | 本日志与 `plan/stage1/README.md` |
| 服务器现状 | 既有 SSH 链路只读检查 | 完成；发现 GPU 与 Agent 同步债务 | 本日志与 `plan/stage1/README.md` |
| 旧归档审计 | 归档报告与悬空旧提交只读审计 | 完成 | `legacy/2026-07-stage-ab/` |
| Markdown 链接 | 解析 `plan/stage1/*.md` 本地链接并检查目标 | 全部存在 | 本日志 |
| 空白检查 | `git diff --check` | 通过，仅有现有 LF/CRLF 提示 | 本日志 |

### 产物与证据

| 路径 | 类型 | 状态 |
|---|---|---|
| `plan/stage1/README.md` | 阶段总览 | 初稿完成 |
| `plan/stage1/01_entry_and_contract.md` 至 `11_reporting_and_exit_gate.md` | 11 个子任务计划 | 初稿完成 |
| `plan/general_plan.md` | 总计划入口 | 已增加 Stage 1 链接 |
| `worklogs/2026-07-18-stage1-correctness-plan.md` | 工作日志 | 进行中 |

### 问题、原因与风险

- GPU 现状已漂移，正式 Stage 1 DDP 执行前必须重新逐卡预检和四卡 NCCL smoke。
- `Agent/worklogs.md` 两端不同步，`G1-ENTRY` 当前不能通过。
- 当前没有现行实现；Stage 1 是按新规格重建，而不是恢复旧分支。
- 工作树存在其他并行计划草稿，现阶段不安全提交并推送整个工作树；必须等待并行修改稳定后统一审查。

### 当时下一步

- 完成数学、服务器事实和历史风险三路交叉审查。
- 修正审查发现并重复链接、格式与差异检查。
- 在不覆盖并行工作成果的前提下决定统一提交和多端同步时点。

## 2026-07-18 21:58 CST — 完成交叉审查与计划定稿

### 交叉审查

- 数学审查重点核对 DDP 对照、加权 U oracle、近零容差、纯噪声解析量、clip 与 GradScaler 生命周期。
- 本机/旧实现审查重点核对 raw/negative/absolute/movement 历史回归、配置闭环、skip 事务与 checkpoint 恢复。
- 服务器审查重点核对现有资产事实、缓存/tmp/checkpoint 路径、GPU 租约、进程清理、离线运行和多端同步规则。
- 三路审查在最终差量复核中均报告“无仍未闭合的实质问题”；审查过程未让子任务编辑文件。

### 审查后修正

- 把 full-batch 路由 A 限定为 loss/mean-gradient/raw/update 参考；U 仅在保持相同 `M>=2` microbatch 边界的 B/C/D 间比较。
- 将 weighted microbatch U 的真值改为显式 cross-microbatch ordered-pair oracle；逐 token oracle 只验证 token-mean loss/gradient。
- 为 mean gradient、二阶充分统计量、raw/U core、最终 score 与 optimizer delta 分别建立非近零/近零两分支 Gate，避免小学习率让全零实现通过。
- 把纯噪声 fixture 冻结为 iid 高斯基础变量，并用高斯第四矩推导解析标准误。
- 把 AMP 生命周期冻结为每 attempt 一次正式 unscale、scaler step 和 scaler update；finite/skip 决策先全局一致，再事务性提交长期累计。
- 区分 `attempt_step` 与 `successful_optimizer_step`：skip 消费数据/RNG并更新 scaler，但不推进 scheduler、贡献或 movement。
- 明确 data-only net movement 来自带符号 data update 累计；参数端点差另存 total net movement。
- 补齐四卡 checkpoint 的 shared/private 状态所有权、跨 rank checksum、有限超时 pre-commit、唯一原子 rename commit point 和 commit 后广播失败语义。
- 增加单卡/四卡跨 skip 的 fresh-process 恢复、每类 RNG/Generator 直接校验和全部启用长期字段的恢复 Gate。
- 增加 `$DATA_ROOT` 缓存/临时/checkpoint 路径、CUDA/Triton/TorchInductor 缓存审计、进程级 offline guard、短期 heartbeat GPU 租约与进程指纹清理规则。
- 为全部 11 个子任务补齐目的、前置条件、原子执行步骤、产出、可视化/呈现、核验标准和后续 Gate。

### 最终核验

| 项目 | 结果 |
|---|---|
| Stage 1 文件数 | 12 个 Markdown（总览 + 11 个子任务） |
| 结构覆盖 | 11/11 子任务均含目的、前置、步骤、产出、可视化、核验和 Gate |
| 本地 Markdown 链接 | 全部目标存在 |
| 未完成占位标记 | 0 |
| 尾随空白/冲突标记 | 0 |
| tracked diff whitespace 检查 | 通过；仅有现有 LF→CRLF 提示 |
| 最终审查 | 数学、工程、服务器三路均无未闭合实质问题 |

### 提交与同步状态

- 本任务没有 stage、commit、push 或写服务器。
- 原因是同一工作树中还有并行生成的 Stage 0、Stage 2、Stage 3 计划与日志；直接提交整个工作树会混入其他任务的尚未统一审查成果。
- 后续应先统一审查并行文件归属，再按 Agent/git.md 形成准确提交；正式 Stage 1 执行仍受 `Agent/worklogs.md` 哈希不一致和当前四卡 NCCL 未重验阻塞。
