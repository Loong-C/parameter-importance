# 2026-07-18 回到 main 并重整 Agent 文档

- 任务范围：将本机和 sophgo13 仓库安全切回 `main`；把 `Agent/` 运维说明拆分为五个职责单一的文件；建立新的 Git、同步和日志规则。
- 当前状态：完成
- 工作分支：`main`

## 2026-07-18 10:16 CST — 状态核对与文档重整

### 目标与范围

- 保留已有实验分支和提交，只切换工作分支，不删除实验数据和历史。
- 根据现有 SSH 配置、lab-pc 隧道脚本和服务器真实目录整理说明。
- 不修改 SSH 别名、隧道脚本、端口、密钥、服务器数据集或实验结果。

### 实际修改

- 本机和服务器从 `feat/nlp-pythia-stage-ab` 切换到 `main`。
- 将旧的 `Agent/remote_access.md` 和综合同步说明重整为 `remote_access.md`、`server.md`、`git.md`、`sync.md`、`worklogs.md`。
- 更新 `worklogs/README.md`，移除对旧综合说明的引用。
- 用户原有修改：切换前两端工作树均干净，本阶段没有需要合并的未提交用户修改。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 本机分支核对 | `git status --short --branch`、`git rev-parse HEAD` | 已切换 `main`；切换前工作树干净 | 本日志 |
| lab-pc 连接和隧道入口核对 | `ssh lab-pc`，只读查看桌面 `start.cmd` | lab-pc 可达；确认脚本自动重试且同时维持两条反向转发 | `Agent/remote_access.md` |
| 服务器连接和分支核对 | `ssh sophgo13-via-lab` | 服务器可达并已切换 `main` | 本日志 |
| 数据目录核对 | 只读列出 `$DATA_ROOT/datasets` | 确认 Pile、GLUE SST-2、WikiText-103 目录 | `Agent/server.md` |

### 产物与证据

| 路径 | 类型 | 大小 | SHA-256/revision | 验收状态 |
|---|---|---:|---|---|
| `Agent/git.md` | 本机/服务器运维文档 | 3294 bytes | `183f4ba702d22a3a97a459d4873aed62377d51b666d2515990a2f408ecd856ca` | 两端一致 |
| `Agent/remote_access.md` | 本机/服务器运维文档 | 2888 bytes | `795c677e717827492a30342e5b91a4b5959f0df22c72354f14a506ecb023f7a1` | 两端一致 |
| `Agent/server.md` | 本机/服务器运维文档 | 3099 bytes | `9f2d4370ac64990cd29d33ef13de5c20cca65efb4655e928118ae4f3ca012c68` | 两端一致 |
| `Agent/sync.md` | 本机/服务器运维文档 | 4559 bytes | `1bf84f8379018b918eb1680c49dacb7d6d75d764c306782f5170212ceb190015` | 两端一致 |
| `Agent/worklogs.md` | 本机/服务器运维文档 | 3223 bytes | `ca6e9c9f663d55db7dc6a48dd3b61c78903c347a2451eb72ca4dfc68eaa5b43b` | 两端一致 |
| `worklogs/2026-07-18-reset-main-and-reorganize-agent-docs.md` | 工作日志 | 小型文本 | 随 Git 提交 | 进行中 |

### 问题、原因与风险

- 本机 Git 在沙箱账户下触发仓库所有者保护；所有本机 Git 命令使用单次 `safe.directory` 参数，没有修改用户全局配置。
- `Agent/` 被 Git 忽略，五个文件必须按 `Agent/sync.md` 通过 SCP 单独同步并逐文件核对哈希。
- 旧实验分支保留，未合并到 `main`；本任务只改变当前工作分支。

### Git 与多端同步

- 本机分支/HEAD：`main` / `4c91ff6356724d014da9667d2e93d6aac2f23fb8`（提交前）。
- GitHub `main`：`4c91ff6356724d014da9667d2e93d6aac2f23fb8`（提交前）。
- 服务器分支/HEAD：`main` / `4c91ff6356724d014da9667d2e93d6aac2f23fb8`（提交前）。
- `Agent/*.md` 哈希核对：五个文件的文件集合、大小和 SHA-256 已确认一致。
- 工作树状态和临时 bundle 清理：待完成。

### 下一步

- 检查文档链接、敏感信息和 Git 差异。
- 同步五个 `Agent/` 文件并核对 SHA-256。
- 阶段性提交、推送 GitHub，并用 Git bundle 将服务器快进到同一提交。

## 2026-07-18 10:18 CST — Agent 文档同步验收

### 实际完成

- 通过既有 `sophgo13-via-lab` 链路把五个文件逐个复制到服务器仓库的 `Agent/`。
- 两端 `Agent/` 均只包含规定的五个 Markdown 文件。
- 本机和服务器逐文件 SHA-256 与上表完全一致。
- 敏感模式扫描未发现私钥正文、密码/令牌赋值或带 query 的 URL；`git diff --check` 通过。

### 当前状态与下一步

- 文档重整和非 Git 同步阶段已完成。
- 下一步提交并推送 Git 跟踪的旧规范删除、日志入口更新和本日志，再以 Git bundle 快进服务器。

## 2026-07-18 10:22 CST — Git 三端同步完成

### 实际完成

- 阶段提交 `4d05ed3f9899109f488fce8ef7142d5439977dc2` 已推送到 GitHub `main`。
- 已创建并验证只包含 `4c91ff6..4d05ed3` 的增量 Git bundle；服务器从干净的 `main` 以 `git merge --ff-only FETCH_HEAD` 快进到同一提交。
- 本机与服务器上的本次临时 bundle 均已按精确路径删除。
- 本机、GitHub 和服务器的阶段提交一致；五份 `Agent/*.md` 的文件集合与 SHA-256 仍一致。

### 最终状态

- 本机分支/阶段 HEAD：`main` / `4d05ed3f9899109f488fce8ef7142d5439977dc2`。
- GitHub `main`：`4d05ed3f9899109f488fce8ef7142d5439977dc2`。
- 服务器分支/阶段 HEAD：`main` / `4d05ed3f9899109f488fce8ef7142d5439977dc2`。
- 遗留问题：无。旧实验分支和服务器大盘中的实验资产均保留，后续实验可从干净 `main` 开始。
- 本条收尾日志将通过紧随其后的文档提交和相同 bundle 流程同步，不改变上述运维文档内容或实验状态。

## 2026-07-18 11:40 CST — 清理切分支后的忽略缓存

### 核对与清理

- 本机与服务器的 `git ls-files -- src ops tests .pytest_cache` 均无输出，确认四个目录中没有受 Git 跟踪文件。
- 本机与服务器的 `src/`、`ops/`、`tests/` 只含 `__pycache__/*.pyc`；没有发现其他文件。`.pytest_cache/` 为 pytest 运行缓存。
- 已从本机和服务器仓库精确删除 `src/`、`ops/`、`tests/`、`.pytest_cache/`，删除后逐项确认路径不存在。
- 未修改 `Agent/`、Git 跟踪的代码与文档、旧实验分支或 `/home/sophgo13/cjl/storage/parameter-importance` 下的数据和实验资产。

### 结果

- 本机 `main` 工作树在追加本日志前保持干净，仓库根目录不再存在上述四个残留目录。
- 服务器 `main` 同样不再存在上述四个残留目录。
- 被删除内容均为可由 Python/pytest 重新生成的忽略缓存；实验源代码仍完整保存在原实验分支的 Git 历史中。
