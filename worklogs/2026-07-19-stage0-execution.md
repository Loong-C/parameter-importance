# 2026-07-19 第 0 阶段基础设施实施

- 任务范围：完整执行 `plan/stage0/` 定义的 S0.1–S0.12；本轮属于基础设施建设和 gate 核验，不进行参数重要性正式实现，也不形成 Stage 1–9 的实验结论。
- 当前状态：进行中
- 工作分支：`feat/stage0-infrastructure`

## 2026-07-19 10:00 CST — 冻结范围并启动基线复采

### 目标与范围

- 本阶段要完成什么：重新采集本机、GitHub、lab-pc 与 sophgo13 的当前状态，审查用户已有修改，解决 `Agent/` 同步债务，并把 GPU 枚举/ECC 异常形成明确的 G0-G 判定。
- 不在本阶段处理什么：不重载驱动、不重置 GPU、不清除 ECC、不修改系统 CUDA/DNS/SSH/隧道配置，不使用 `sudo`，不触碰活动 Pile `.part`/锁，不启动训练。
- 职责边界：本机负责代码、轻量测试、Git 与调度；lab-pc 只承担既有 SSH 链路；sophgo13 只在仓库和 `DATA_ROOT` 内承载受控环境、资产与 GPU 核验；GitHub 是 Git 远端，不保存大型资产。
- 管理员事项：GPU 物理/驱动层、NVML、PyTorch 枚举不一致，以及不可纠正 ECC 的处置结论；Agent 只负责只读采集和复验。

### 实际修改

- 已完整读取 `Agent/` 五份规范、`plan/general_plan.md`、`plan/stage0/README.md` 和 S0.1–S0.12 全部实施文档。
- 从基线提交 `34966d0819a5229569169bfe436afe9058c0ed24` 创建阶段分支 `feat/stage0-infrastructure`，保留工作树中全部用户已有改动。
- 本轮开始时未改动服务器、环境、资产或 GPU。
- 用户原有修改一并保留：`docs/mathematics.md` 公式字符修复；`plan/general_plan.md` 计划入口；未跟踪的 Stage 0–3 计划和对应 2026-07-18 工作日志。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 本机 Git 起点 | `status -sb`、`rev-parse HEAD`、`fetch origin --prune` | 本机与 `origin/main` 均为 `34966d0`；工作树含上述用户改动 | 本日志；本轮终端记录 |
| 用户修改检查 | `git diff --check`、`git diff --stat`、完整差异 | 已审查；未发现空白错误，未丢弃或隐藏 | `docs/mathematics.md`、`plan/`、`worklogs/` |
| 分支隔离 | `git switch -c feat/stage0-infrastructure` | 成功；全部修改随工作树保留 | 本地 Git |

### 问题、原因与风险

- 当前计划快照显示 G0-G 曾因 8/7 GPU 枚举和不可纠正 ECC 失败；必须以本轮服务器复采结果为准，旧 `READY` 不作通过证据。
- `Agent/worklogs.md` 先前两端不一致；需完成集合、大小、语义差异与 SHA-256 审查后才能选择权威版本并同步。
- G1-D 尚无已授权的第二故障域；若仍无备份目标，后续必须由用户明确接受有范围、有效期和终止条件的单盘风险，Agent 不会自行推定接受。

### Git 与多端同步

- 本机分支/HEAD：`feat/stage0-infrastructure` / `34966d0819a5229569169bfe436afe9058c0ed24`（脏工作树，含已记录用户改动）。
- GitHub 分支/HEAD：`main` / `34966d0819a5229569169bfe436afe9058c0ed24`。
- 服务器分支/HEAD：等待本轮只读复采。
- `Agent/*.md` 哈希核对：等待本轮复采与语义比较。
- 临时 bundle：尚未创建。

### 下一步

- 完成远程身份、三端仓库、`Agent/` 集合/哈希、服务器系统/存储/GPU 健康复采。
- 根据 G0-C 与 G0-G 结果，只推进依赖允许的非 GPU 子任务；硬件未通过前不运行 CUDA/NCCL/DDP。

## 2026-07-19 10:12 CST — 完成 G0 基线复采并封存硬件阻塞

### 实际修改

- 审查两端唯一差异：服务器 `Agent/worklogs.md` 缺少本机版本中“使用中文”的明确要求；该要求与同文件后文一致，因此选择本机版本为权威，只精确同步该文件。
- 同步后重新列出两端文件集合并核对五份 SHA-256，全部一致。
- 新增 `reports/stage0/g0-baseline-20260719.json` 和 Markdown 摘要；旧 `READY` 保留为历史证据，未覆盖或复用。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| SSH 公钥链路 | 两个别名均使用 `BatchMode=yes` | 两端退出码 0；身份和初始目录符合规范 | `reports/stage0/g0-baseline-20260719.json` |
| 三端仓库基线 | 本机/GitHub/服务器 HEAD 与服务器工作树 | 三端基线均为 `34966d0`；服务器干净 | 同上 |
| Agent 集合与哈希 | 两端精确五文件、SHA-256 | 5/5 同大小同哈希 | 同上 |
| 系统与存储 | OS、内核、glibc、CPU、内存、挂载、空间、inode、limits | 采集完成；根盘余 15 GiB，大盘余约 2.9 TiB | 同上 |
| GPU 多层枚举 | PCI、驱动目录、NVML、PyTorch | 8/8/7/7，不一致 | 同上 |
| GPU 健康 | ECC、row remap、筛选后的 Xid | `50:00.0` 有 24 次不可纠正 ECC、row-remap pending、Xid 95 | 同上 |

### Gate 判定

- G0-C：`PASS`。
- G0-G：`BLOCKED`，责任方为服务器管理员；需诊断缺失的 `4f:00.0`，处置 `50:00.0` 的 ECC/Xid/row-remap，并提供 8/8/8 修复结论或明确降配白名单。
- G0 总 gate：`BLOCKED`。按计划只继续 S0.2–S0.5 中不占用 GPU 的内容。

### Git 与多端同步

- 本机：`feat/stage0-infrastructure` / 基线 `34966d0`，工作树修改均已记录。
- GitHub：`main` / `34966d0`。
- 服务器：`main` / `34966d0`，工作树干净。
- `Agent/*.md`：五文件最终 SHA-256 已写入 G0 JSON，全部一致。
- 未创建 bundle；Git 跟踪内容尚未进行阶段提交与推送。

## 2026-07-19 10:20 CST — 完成 S0.2 本地存储机制候选

### 目标与范围

- 本阶段完成路径边界、目录生命周期、缓存定向、容量预算、Git 守卫和 canary 实现。
- 本阶段不触碰活动 Pile shard 7，不创建大型资产，不清理现有目录；服务器 canary 在代码形成提交并同步后运行。

### 实际修改

- 建立现行 Python package/`pyproject.toml` 和共享原子写入、SHA-256、规范 JSON 底层。
- `StorageLayout` 强制显式 `PARAM_IMPORTANCE_DATA_ROOT`，拒绝 home/root/system tmp 回退和路径逃逸。
- 建立 13 个目录的读写/原子发布 canary；每次只创建、替换并精确删除唯一小文件。
- 建立 run/attempt/session 不覆盖目录和受测试的状态转换骨架。
- 显式生成 Hugging Face、Datasets、Torch、XDG 和临时缓存变量，全部限定在 `DATA_ROOT`。
- 建立容量预算公式、Stage 0/160M/410M 初始分析预算、根盘 10 GiB 保护和 `E + max(0.2E, 100 GiB)` 大盘启动余量。
- 建立机器可读生命周期/保留策略和 Git 禁止目录、禁止类型、未知二进制审查、10 MiB 阈值守卫。
- 更新 `.gitignore`，防止环境、缓存、checkpoint、wheel、模型权重和运行产物进入 Git。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| CPU 单元测试 | `pytest -q` | 19 passed，退出码 0 | `tests/` |
| Git 守卫 | `PYTHONPATH=src python -m param_importance_nlp git-guard --repo .` | `PASS`，退出码 0 | `src/param_importance_nlp/git_guard.py` |
| JSON 语法 | 两份新增 policy/report 使用标准解析器 | 通过，退出码 0 | `policies/`、`reports/stage0/` |
| 空白检查 | `git diff --check` | 通过；仅 Windows CRLF 提示 | 本轮终端记录 |

### Gate 判定与风险

- G1-B：本地机制和拒绝测试已通过；仍需在服务器同一提交上完成 13 目录 canary、根盘/大盘启动检查后最终判定。
- G1-D：`PENDING_USER_DECISION`。当前无已授权第二故障域；建议仅对 Stage 0 可再生 smoke 产物接受至 2026-08-18 或 Stage 4 开始前（先发生者）的单盘风险，正式 Stage 4/5 产物不在建议范围。
- G1 总 gate：尚未通过，因此不开始新资产下载或环境重建。
