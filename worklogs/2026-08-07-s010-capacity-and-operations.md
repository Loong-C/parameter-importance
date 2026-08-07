# 2026-08-07 S0.10 容量评估、运行保护与故障处理

- 任务范围：核验 S0.9 完成状态；按 `plan/stage0/10_capacity_and_operations.md` 继续完成 S0.10，包含正式 G8 链（G8-C/G8-S4/G8-S5/G8）执行、证据核验与多端同步。
- 当前状态：进行中（S0.9 已完成；S0.10 正式链准备启动）
- 工作分支：`feat/stage1-cpu-evidence`

## 2026-08-07 14:30 CST — S0.9 完成确认与 S0.10 启动准备

### 目标与范围

- 完成：按 `Agent/` 运维文档只读核验 S0.9 的正式证据；确认 G8 前置条件（G0-G、G3-S4、G3-S5、G5、G6、G7）；准备 S0.10 正式链并受控暂停 Pile 下载。
- 不在本阶段处理：S0.11 及之后任务；管理员级 GPU 动作。

### S0.9 完成核验

| 项目 | 命令/来源 | 结果 | 证据路径 |
|---|---|---|---|
| 服务器 Git | SSH `git branch/rev-parse/status` | `feat/stage1-cpu-evidence` @ `c3a7d74`，工作树干净 | 本日志 |
| G7-RECOVERY 索引 | `cat evidence/stage0/g7-recovery-formal/4ecf7008…/index.json` | status=PASS，`next_task_id=stage0.10_capacity_and_operations`，generator commit `c73d8a4` | `$DATA_ROOT/evidence/stage0/g7-recovery-formal/4ecf7008…/index.json` |
| G7 环境 gate | `cat …/environment.json` | passed_gate_ids 含 G0-G、G3-S1/S2/S4/S5/S6、G4、G5、G6、G7、G7-LOGGING；environment_hash=`f50e5df7…` | 同上 `environment.json` |
| 正式链日志 | `tail g7-recovery-chain-c73d8a4.log` | `G7R_REF=…`、`CHAIN_STATUS=PASS` | `$DATA_ROOT/tmp/g7-recovery-chain-c73d8a4.log` |
| 服务器锚点 | `git branch --list formal/run-c73d8a4` | 存在，HEAD=`c73d8a4e143a43b8ef9bff67ac7d42b04b539134` | 本日志 |
| GPU 现状 | `nvidia-smi` | 四卡 53/9c/9d/a0 全部 0 MiB、ECC 0/0、无 compute apps | 本日志 |

### 实际修改

- 代码、配置、文档：
  - 新增本工作日志。
  - 纳入用户原有未提交文件 `docs/importance-estimation-core-annotated-report-readable.md` 与 `.pdf`（随本阶段提交，不覆盖、不丢弃）。
- 服务器或外部状态：
  - Pile 下载监督器 `CjlPileFullSupervisor`/`CjlPileFull` 当前在 lab-pc 运行（shard 12 `.part` 约 2.7 GiB/30 GiB）。
  - 按 S0.10 preflight 硬要求（`PREFLIGHT_COMPETING_DOWNLOAD`）和此前 S0.9 的同款授权模式，S0.10 正式链前受控暂停下载：先停止 lab-pc 两个计划任务，再核对服务器 `NO_CURL`；G8 完成后恢复。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 监督器状态 | `schtasks /query /tn CjlPileFullSupervisor,CjlPileFull /v /fo list` | 两者均 Running（上次结果 267009） | 本日志 |
| 下载暂停 | `schtasks /end /tn …` 后服务器 `ps -ef` 核对 | 无 `server_xet_download.sh`/curl 进程；`.part` 保留续传 | 本日志 |

### 判定

- S0.9 完成状态成立：G7 完整可恢复性 gate PASS，且 G8 所需前置 gate 全部在 G7R 环境记录中为 PASS。
- S0.10 正式链具备启动条件：GPU 空闲健康、服务器锚点 `formal/run-c73d8a4` 就位、G7R 索引引用可用。

### 问题、原因与风险

- lab-pc 初次不可达是本地 `ssh` 沙箱 shim 对带 `-o` 参数命令的误拦截；改用批准前缀 `ssh lab-pc` 后恢复，非隧道故障。
- 风险：下载暂停期间不产生竞争；`.part` 与锁文件保留，恢复后由监督器续传。
- 遗留：S0.10 完成后恢复 `CjlPileFullSupervisor` 并核对 shard 12 续传；随后 S0.11。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence`（本日志提交后更新）。
- 服务器：正式链在 `formal/run-c73d8a4`（c73d8a4）执行；工作分支后续同步至收尾提交。
- `Agent/*.md`：不受影响。

### 下一步

- 生成 `$DATA_ROOT/tmp/g8-capacity-chain-c73d8a4.sh` 并在服务器锚点启动，监控至 `CHAIN_STATUS=PASS`。
- 核验 G8 索引（next_task_id=`stage0.11_test_quality_and_replay`）与 G8-C/S4/S5 记录。
- 收尾：恢复 Pile 下载、更新本日志、三端同步与 `Agent/*.md` 哈希核对。

## 2026-08-08 00:30 CST — S0.10 完成（G8 完整 gate PASS）

### 目标与范围

- 完成：在服务器锚点 `formal/run-c73d8a4`（commit `c73d8a4`）执行 G8 正式链并核验 G8-C/G8-S4/G8-S5/G8；恢复 Pile 下载；完成日志/同步/清理。
- 不在本阶段处理：S0.11 及之后任务。

### 实际修改与执行

- 代码、配置、文档：无功能代码改动；追加本日志。
- 服务器或外部状态：
  - `$DATA_ROOT/tmp/g8-capacity-chain-c73d8a4.sh` 正式链后台执行：`formalize_g8.py` 于 2026-08-07T15:01:58Z 启动，2026-08-07T16:14:05Z 完成。
  - Pile 下载监督器已恢复：`schtasks /run /tn CjlPileFullSupervisor`，服务器端重新出现 `server_xet_download.sh`（先校验既有 shard，随后续传 shard 12 `.part`）。

### 实验与验证

| 项目 | 命令/来源 | 结果 | 证据路径 |
|---|---|---|---|
| G8 正式链 | `g8-capacity-chain-c73d8a4.log` | `CHAIN_STATUS=PASS`；formalize_g8 输出 status=PASS | `$DATA_ROOT/tmp/g8-capacity-chain-c73d8a4.log` |
| G8 索引 | `cat evidence/stage0/g8-formal/1d76ea6e…/index.json` | status=PASS，`next_task_id=stage0.11_test_quality_and_replay`，generator commit c73d8a4，checked_at=2026-08-07T16:14:05Z | `$DATA_ROOT/evidence/stage0/g8-formal/1d76ea6e…/index.json` |
| G8 gate 记录 | 容量证据对象 `gate_records` | stage0.G8 / G8-C / G8-S4 / G8-S5 全部 PASS；G8-C 六项检查 PASS | `$DATA_ROOT/evidence/stage0/tasks/10-1d76ea6e…/objects/capacity_envelope/…json` |
| G8 环境 | `…/environment.json` | passed_gate_ids 新增 G8、G8-C、G8-S4、G8-S5；environment_hash=`7b4fbaa9…` | `$DATA_ROOT/evidence/stage0/g8-formal/1d76ea6e…/environment.json` |
| 测量摘要 | `…/reports/measurement-summary.json` | status=PASS；峰值显存 32.93%（限 80%）；open fds 8.59%（限 70%）；rank 不平衡 0.021%；日志开销中位数全部 ≤6.74%；loader 饱和点 4 worker；410M checkpoint 暂停 58.6s；36 份 worker report | `$DATA_ROOT/evidence/stage0/g8-suite/8773c880…/reports/measurement-summary.json` |
| 后置检查 | `…/reports/postflight.json` | status=PASS；四卡显存归零、无残留进程、ECC 0/0、topology 未漂移 | 同上 `reports/postflight.json` |
| worker 报告 | `reports/` 目录 | 40 个文件（preflight + 36 worker + summary + fault + postflight） | `$DATA_ROOT/evidence/stage0/g8-suite/8773c880…/reports/` |

### 判定

- **S0.10 完成**：G8-C（基础设施与 synthetic shape 容量）、G8-S4（160M 实测）、G8-S5（410M 实测）与 G8 总 gate 全部正式 PASS；下一任务为 S0.11。
- Stage 0 当前全部硬 gate（G0-C、G0-G、G1、G2、G3 系列、G4、G5、G6、G7、G7-LOGGING、G8-C、G8-S4、G8-S5、G8）通过。

### 问题、原因与风险

- 执行期间 VPS 跳板 SSH 多次超时/重置，G8 链在服务器端 nohup 不受影响；恢复后一次性核验完成。
- 下载恢复先校验既有 shard（00000–00011 与 00012 `.part`），随后续传；`.part` 与锁文件保留。
- 风险：410M checkpoint 实测约 51.9 GiB/边界，暂停 58.6s 已进入容量预算；后续 S0.11 重放需沿用同一容量包络。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence`（本日志提交后更新）。
- 服务器：正式证据锚点 `formal/run-c73d8a4` 保留；工作分支同步至收尾提交。
- `Agent/*.md`：不受影响。

### 下一步

- 三端同步与清理：删除本次 bundle 与临时启动脚本；核对 `Agent/*.md` 哈希。
- 继续 S0.11：`formalize_g9.py --g8-index-ref evidence/stage0/g8-formal/1d76ea6e…/index.json`。
