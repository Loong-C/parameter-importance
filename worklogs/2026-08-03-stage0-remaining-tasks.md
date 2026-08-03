# 2026-08-03 Stage 0 剩余任务实施

- 任务范围：按 `plan/stage0/` 的依赖与 Gate 要求，从 S0.4 继续完成至 S0.12；本轮只建设和验收实验基础设施，不把 Stage 0 的 synthetic/训练 smoke 解释为参数重要性算法或科学结论。
- 当前状态：进行中
- 工作分支：`feat/stage0-completion`

## 2026-08-03 14:10 CST — 恢复 S0.4–S0.12 执行

### 目标与范围

- 本阶段要完成什么：重新核对现有代码、测试、服务器资产与 Gate 证据，从 G3 开始逐项形成 G3–G10 的当前有效证据；每个可独立复核的 Gate 完成后测试、记录、提交、推送并同步。
- 不在本阶段处理什么：不修改 SSH/反向隧道拓扑，不绕过资产 manifest 或硬件 Gate，不使用 `.part`/活动锁作为输入，不执行系统级驱动/CUDA 改造，不把本机 CPU/synthetic 结果冒充服务器 CUDA/NCCL 或正式资产证据。
- 用户决定：用户在当前任务中要求继续并完成 Stage 0 剩余任务，因此 2026-08-03 早先记录的“S0.4–S0.12 暂停”从本条开始解除。

### 实际状态

- 从干净提交 `11a3f9072b01ab122d3bd3dc947f637dcfe8d755` 创建专用分支 `feat/stage0-completion`。
- 本机与 GitHub 原功能分支起点一致；`origin/main` 仍为历史基线，不作为本轮工作分支。
- 最新封存证据保持 G0、G1、G2 为 `PASS`；S0.4–S0.12 尚无 formal Gate 通过声明。
- `lab-pc` 与 `sophgo13-via-lab` 当前均在 SSH banner 交换阶段超时。按 `Agent/remote_access.md` 未修改连接参数或启动替代拓扑。
- 本机存在预装 ToDesk 客户端，但其窗口无法由受控 Windows 应用接口可靠捕获；未尝试自动化认证、未读取连接凭据、未通过该客户端执行远端操作。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 本机 Git 基线 | `git status -sb`、`git rev-parse HEAD`、`git ls-remote` | 工作树干净；本机/GitHub功能分支为 `11a3f907` | 本日志；当前终端记录 |
| 服务器 SSH 复核 | 两个规定别名、`BatchMode=yes`、有限连接超时 | `BLOCKED`：SSH banner 交换超时 | 本日志；当前终端记录 |
| 现有 formal 证据盘点 | `reports/stage0/`、工作日志、任务目录 | 只有 G0–G2 当前报告；G3–G10 未形成 | `reports/stage0/` |
| 本机完整测试基线 | `.venv`、`PYTHONPATH=src`、`pytest -q` | 运行中，结果待追加 | 本日志后续条目 |

### 问题、原因与风险

- S0.4 的真实模型/数据盘点以及 S0.6–S0.11 的服务器/GPU Gate 依赖 SSH 链路恢复；在链路恢复前只能完成本机代码、合同和负向测试，不能宣称相应 Gate 通过。
- G1-D 的单盘风险接受只覆盖 Stage 0 可再生 smoke，有效至 2026-08-18 23:59 CST 或 Stage 4 开始前（先发生者）；本轮不得扩展其范围。
- 当前仓库存在完整的本机 run-ready 实现，但本机验证报告明确无 formal 资格；后续必须逐 Gate 生成真实服务器证据。

### 下一步

- 完成 S0.4–S0.12 的代码/证据缺口审计。
- SSH 链路恢复后先只读复核推荐环境、GPU qualification、服务器仓库、活动下载和资产目录，再执行 S0.4。
- 在服务器不可达期间，补齐不依赖服务器的确定性、拒绝路径和报告生成缺口。

## 2026-08-03 14:42 CST — SSH 恢复、服务器只读复采与未跟踪文件封存

### 目标与范围

- 响应用户“SSH 应可连接但较慢”的信息，仅延长两条既定别名的连接等待时间；不修改 SSH 配置、跳板拓扑或反向隧道。
- 在执行 S0.4 前复采服务器仓库、存储、GPU、资产与活动下载状态，并先保存服务器仓库内已知未跟踪恢复文件。

### 实际状态

- `lab-pc` 与 `sophgo13-via-lab` 均连接成功，典型握手约 20 秒。
- 服务器仓库原状态为 `feat/stage0-infrastructure@5cc53930`，存在 6 个未跟踪的 GPU 恢复脚本/授权报告；其中 3 个与本机相同，3 个为本机已归档版本的较早版本。
- 按服务器写回流程，把 6 个文件与 `worklogs/2026-07-19-stage0-execution.md` 的封存记录提交为 `fc5a4fc8fe8630c1be2655a622b7e940ca8a44b0`（`chore: preserve server GPU recovery files`）。服务器仓库使用 repo-local Git identity；未修改全局 Git 配置。
- 生成并验证增量 bundle `server-preserve-fc5a4fc-incremental.bundle`，前置提交为 `5cc53930a3f745fbd3e9ea4e171bd0773172984a`，大小 22,911 字节；已回传本机并验证完整。
- DATA_ROOT 为 `/home/sophgo13/cjl/storage/parameter-importance`，底层 `/dev/nvme0n1` 为 ext4/rw；可用空间约 2.8 TiB、inode 使用约 1%，目录权限为 `sophgo13:sophgo13 0750`。
- 四张白名单 A100-SXM4-80GB 均可见，检查时显存使用为 0 MiB、无 compute process、ECC volatile/aggregate 为 0。
- 当前模型目录只有 Pythia 14M step0、31M-deduped step0、160M-deduped step0/step512；Pythia 410M 缺失。数据目录只有 SST-2、Pile 与 WikiText；MNLI、RTE 缺失。
- Pile 活动对象为 `document-00009-of-00020.bin.part`，大小 22,882,025,472 字节，最后写入时间为 2026-07-19 13:32:38 UTC。精确文件持有者检查为空，且没有 curl/wget/aria2/python 下载进程；因此判定下载当前停滞，不删除锁、不恢复下载，也不把 `.part` 作为资产输入。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| SSH 复核 | 两个既定 alias，`ConnectTimeout=60` | `PASS`；两端均成功登录 | 本日志；当前终端记录 |
| 服务器未跟踪文件语法/内容检查 | `bash -n`、JSON parse、本机/服务器逐文件比较 | `PASS`；确认 6 个目标及版本差异 | `artifacts/server-untracked-audit-20260803/`（本机忽略目录） |
| 服务器保护提交 | Git staged diff、commit、bundle verify | `PASS`；提交 `fc5a4fc`，增量 bundle 验证通过 | DATA_ROOT `tmp/`；本机忽略目录 |
| Stage 0 定向回归 | 11 个资产/合同/runtime/GPU promotion 测试文件，workspace `--basetemp` | `142 passed, 7 skipped`；52.66 秒 | 本日志；当前终端记录 |

### 问题、原因与风险

- 首次完整 bundle 为 15,550,450 字节，经慢速中继传输长期停滞；未使用其不完整本机副本。随后改为以已存在共同前置提交为 prerequisite 的 22,911 字节增量 bundle，既保留提交对象又避免无意义重复传输。
- 默认 pytest 临时目录因 Windows 权限导致 setup error；改用工作区内 `--basetemp` 后定向测试通过。7 个 skip 分别来自 Windows 目录 symlink 权限和该 Windows Torch wheel 不支持所需 Gloo device，不能作为服务器 CUDA/NCCL Gate 证据。
- S0.4 正式 G3 仍为 `NOT_RUN`：缺少 410M、MNLI、RTE，31M 旧 manifest 需要 BOM 修复，Pile 当前格式/覆盖合同尚未接入 formal provider。

### 下一步

- 将服务器保护提交归并到 `feat/stage0-completion`，冲突时保留已审计的本机新版本内容，同时保留服务器提交历史和日志追加。
- 冻结 S0.4 资产矩阵、序列/目标 token 语义与最大游标预算；先补 manifest、语义审计和 Pile/GLUE 构建合同，再取得/发布正式资产。
