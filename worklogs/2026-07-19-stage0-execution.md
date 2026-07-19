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

## 2026-07-19 10:35 CST — G1-B 服务器核验通过

### 实际修改与同步

- 提交 `50282e3d879f09a7ba9fc1dd2541bbe24bda00b2` 已推送到 GitHub `feat/stage0-infrastructure`。
- 已验证 bundle 后，在服务器从 `main@34966d0` 创建同名分支；服务器 HEAD 与本机/GitHub 一致且工作树干净。
- 服务器只运行 CPU 测试、Git 守卫、存储/原子 canary 和空间/inode 只读检查；未导入 CUDA、未查询 GPU、未启动训练。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 服务器 CPU 测试 | 现有锁定 venv，三个 Stage 0 测试文件 | 21 passed，退出码 0 | 本轮终端记录 |
| 服务器 Git 守卫 | 同一提交、完整工作树 | `PASS`，退出码 0 | 本轮终端记录 |
| 13 目录 canary | `storage-check --require-writable --canary` | 13/13 通过，0 失败，0 残留 | `reports/stage0/g1-storage-canary-20260719.json`；服务器同名报告 |
| canary 哈希 | 本机下载后与服务器比对 | SHA-256 `76fa7a93...35757fac` 一致 | `reports/stage0/g1-storage-mechanism-20260719.json` |
| Stage 0 启动空间 | 预计新增 620,000,000,000 B | 需 744,000,000,000 B；大盘可用 3,079,944,097,792 B | 同上 |
| 根盘/inode 保护 | 根盘 10 GiB 门槛；大盘 inode | 根盘可用 15,176,171,520 B；inode 可用 233,426,897 | 同上 |

### Gate 判定

- G1-B：`PASS`。
- G1-D：仍为 `PENDING_USER_DECISION`。
- G1 总 gate：`BLOCKED`；在用户作出明确持久性决定前，不启动 S0.3 环境重建或 S0.4 新资产获取。

## 2026-07-19 10:42 CST — 修复服务器暴露的磁盘测试隐式前提

### 失败与根因

- 在服务器提交 `bb8c899` 上运行完整 CPU 测试得到 1 failed / 21 passed。
- 失败用例 `test_storage_budget_cli_reports_success` 使用 pytest 临时目录作为 data/root 文件系统，却断言最小 100 GiB 安全余量必然满足。
- 本机 D 盘偶然有足够空间，服务器 pytest 临时目录位于余量约 15 GiB 的根盘；实现按设计 fail-closed 返回 1，因此是测试环境依赖错误，不是容量保护实现错误。

### 修复与安全性

- 单元测试改为注入确定性的磁盘/inode 快照，只验证 CLI 组装和退出语义。
- 真实服务器空间仍由独立集成命令读取实际大盘和根盘，不在单元测试中伪造通过。
- 失败记录保留；修复后将重新运行本机/服务器全套 CPU 测试和真实启动预算检查。

### 修复后复验

- 修复提交 `02f4d4376ecdf624552c70a8f066e8d1957cfa06` 已推送并通过增量 bundle 快进到服务器。
- 本机与服务器完整 CPU 测试均为 22 passed、退出码 0。
- 服务器真实 `storage-budget-check` 返回 `ok=true`：大盘可用 3,079,252,004,864 B，要求 744,000,000,000 B；根盘可用 15,175,081,984 B，高于 10 GiB；inode 可用 233,426,895。
- G1-B 保持 `PASS`；G1-D 仍等待用户明确决定。

## 2026-07-19 10:50 CST — G1-D 获得有期限风险接受

### 用户决定

- 用户明确接受仅 Stage 0 可再生 smoke 产物的单盘丢失风险，有效至 2026-08-18 23:59 CST 或 Stage 4 开始前（先发生者）。
- 用户明确排除 Stage 4/5 正式产物；该决定不能自动扩展到正式 checkpoint、正式原始结果、人工判定或论文唯一证据。
- 决定全文、范围、期限和提前失效条件已写入 `docs/stage0/persistence-decision.md` 与机器可读报告。

### Gate 判定

- G1-B：`PASS`。
- G1-D：`PASS — TIME_BOUNDED_RISK_ACCEPTANCE`。
- G1：`PASS`。S0.3 环境重建和 S0.4 受控资产工作可以按前置条件开始。
- Stage 4 开始前必须重新审核并取得覆盖正式产物的备份/风险决定；当前批准届时自动失效。

## 2026-07-19 11:35 CST — GPU 范围收缩为管理员隔离的健康四卡

### 用户决定与准确解释

- 用户确认项目只需要四张 GPU，并选择 G0-G 验收路径 B（稳定隔离/降配）。
- 项目侧接受把异常设备排除在候选四卡之外，但不把故障记为“已忽略”或“已修复”。
- 当前至少有两个不同异常对象：`0000:4f:00.0` 在 PCI/驱动层存在但 NVIDIA RM
  初始化失败；`0000:50:00.0` 有不可纠正 ECC、pending row remap 和 Xid 95。
- 两设备在管理员逐卡清除前均为默认排除项；不能仅用变化后的数字 GPU index 隐藏
  8/7 枚举差异。

### Gate 与责任边界

- 项目所有者状态：`APPROVED`；服务器管理员状态：`PENDING`。
- 管理员仍须给出并强制实施精确四卡 PCI/UUID 白名单、两张异常设备的隔离机制与
  处置结论、审批期限和失效条件，并确认 NVML/PyTorch 对白名单映射一致。
- 管理员清场前不再调用 GPU 管理/计算 runtime，不执行 reset、驱动重载、节点重启、
  ECC 清除、PCI unbind/rebind 或 `sudo` 操作。
- G0-G 与 G0 继续为 `BLOCKED`；S0.2–S0.5 的纯 CPU 工作可继续，S0.6 GPU、S0.7、
  GPU 容量实测和正式训练仍禁止。

### 环境重建收口

- S0.3 已实现严格 lock/freeze/wheelhouse 身份模块和非破坏性离线重建入口。
- 服务器无特权 network namespace 因 `uid_map: Operation not permitted` 不可用；入口
  明确使用环境白名单、pip hash-checking/`--no-index` 和所有 Python/pip 子进程的
  `strace` 连接审计，不声称内核级断网。
- 在 G0-G 阻塞期间，成功的 CPU 重建只能发布
  `CPU_ONLY_CANDIDATE`（`training_eligible=false`、`g2_status=BLOCKED`），不能更新
  训练推荐环境或冒充 G2 通过。

## 2026-07-19 16:25 CST — S0.3/S0.4 本地实现收口

### 环境与 lock 加固

- S0.3 重建入口固定读取三份服务器依赖输入，并以机器可读 lock provenance 绑定
  输入 SHA-256、lock SHA-256、目标平台和核心 runtime 策略；调用者不能替换必需
  输入，直接依赖缺失或精确 pin 冲突均在创建候选 venv 前失败。
- wheel 覆盖从名称/版本前缀检查升级为 CPython 3.12、Linux x86-64、manylinux
  glibc 2.28 兼容标签检查；Windows/macOS、错误架构、错误 ABI 和过新的 manylinux
  wheel 不再计入覆盖。
- 核心导入现在逐项核对 Python series、Torch/Transformers/Datasets/Accelerate/
  TensorBoard、Torch CUDA runtime、cuDNN runtime 与 cuDNN/NCCL 分发包版本。
  实际 NCCL runtime 仍明确等待 G0-G 后的四卡复验。
- immutable evidence 和 advisory lock 均拒绝符号链接、非普通文件和多硬链接，避免
  锁文件截断/污染其他对象。

### 资产合同基础

- S0.4 新增严格资产状态机、角色/证据授权、类型化元数据和固定 revision 校验。
- READY 解析始终执行完整 SHA-256；READY/INVALID 历史不能在原路径覆盖。
- manifest 发布要求批准根目录，使用 per-object lock 与 digest CAS，拒绝越界、
  symlink/junction、并发旧写和无证据状态推进。
- 这些是 S0.4 的合同与测试基础，不代表服务器实际资产已经全部盘点或 G3 已通过。

### 本地验证

- `pytest -q`：121 passed，1 skipped；唯一 skip 是当前 Windows 无创建目录符号链接
  权限，对应测试必须在 Linux 服务器补跑。
- `python -m compileall -q src ops tests`：通过。
- 8 份 JSON 解析、Draft 2020-12 schema 自检和 `git diff --check`：通过（仅既有
  Windows checkout 的 LF/CRLF 提示）。
- 服务器离线重建尚未启动，现有 venv、候选目录和训练任务均未改动；S0.3/G2 仍未
  因本地测试而标记为通过。

## 2026-07-19 16:30 CST — Linux 测试与重建路径预检

- 本机、GitHub、服务器均同步到 `8295f0800e5a45d179d0a760f1a39eabd266d3d7`；
  两端精确 Git bundle 临时文件在三端提交核对后删除。
- 服务器现有环境在隔离的大盘 `TMPDIR` 下执行全量纯 CPU/Linux 测试：122 passed，
  退出码 0；Windows 跳过的目录 symlink 场景在 Linux 实际通过。
- 测试专属临时根先核对绝对路径和所有者，再精确递归删除；未接触活动 Pile shard。
- 重建预检发现 `DATA_ROOT/locks` 不存在，且它不属于 S0.2 批准的 13 个顶层目录。
  因此没有临时创建新根目录；重建锁修正为既有 `operations/` 下的固定文件后再执行。
