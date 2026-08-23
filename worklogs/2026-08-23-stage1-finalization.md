# Stage 1 收尾与 Stage 2 交接

## 2026-08-23 10:10 CST — 正式完成核验与仓库整合

### 目标与范围
- 本阶段包含：复核 Stage 1 正式服务器证据、三端发布状态和 Agent 文档哈希；整合并清理 Stage 1 临时开发分支；冻结 Stage 2 的进入边界。
- 本阶段不包含：重跑已通过的 GPU Gate、修改估计器语义、处理演示文稿中的用户未提交修改。

### 实际修改
- 将 `feat/stage1-s111-formal-final` 的最终 S1.10/S1.11 实现合并进 `feat/stage1-cpu-evidence`。
- 以仅保留提交可达性的合并记录纳入 S1.8 producer `6d0dcb7cadaa1539024f2b8dbd1e0d340ff50eef` 与 S1.9 producer `e30538e27d5f90cc384978187a047db7677b2312`；合并后的 Stage 1 代码树与 S1.11 正式 producer `3f18b04df8922be9894678ae4842bd999c7e8fd5` 完全相同。
- 删除 8 个已被整合或取代的本地/远端 Stage 1 开发分支及其干净临时 worktree；正式 producer 提交仍由整合分支可达。
- 用户已有修改处理：`presentation/parameter_importance_workplan.{tex,pdf}` 和 `presentation/parameter_importance_workplan_notes.pdf` 保持未暂存、未提交。

### 验证
| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| S1.11 正式发布 | 只读检查最终 `index.json`、`validation.json`、`replay-validation.json`、`stage-report.json` | `G1-EXIT=PASS`，发布验证和严格重放均为 `PASS` | `$DATA_ROOT/evidence/stage1/s1-11-formal/3f18b04df8922be9894678ae4842bd999c7e8fd5/s1-11-r4-20260821/` |
| S1.11 针对性测试 | 复核不可变 `test-summary.json` | 23 collected，23 passed，0 failed/error/skipped | 同上 `test-summary.json` |
| Agent 规则同步 | 本机 `Get-FileHash` 与服务器 `sha256sum Agent/*.md` | 六份文件逐项 SHA-256 一致 | S1.11 `stage-report.json` 的 `upstream_context.agent_sha256` |
| 代码影响判定 | `git diff --quiet 3f18b04..67ef399 -- ops schemas src tests configs` | 无差异；仅演示文稿和提交可达性变化 | 本机 Git 历史 |
| Git 发布 | `git push origin feat/stage1-cpu-evidence`、`git ls-remote` | 远端已发布 `67ef39988059aeb53d42d7d9d91d33418a1d1223` | `origin/feat/stage1-cpu-evidence` |

### 证据身份与有效性
- producer commit / execution commit：S1.11 `3f18b04df8922be9894678ae4842bd999c7e8fd5`；S1.8 `6d0dcb7cadaa1539024f2b8dbd1e0d340ff50eef`；S1.9 `e30538e27d5f90cc384978187a047db7677b2312`；S1.10 `fbb09e4d338125954fc614c745cf7ab88c58d3b2`。
- consumer commit：仓库整合后为 `67ef39988059aeb53d42d7d9d91d33418a1d1223`。
- S1.11 index artifact hash：`361de10b9aba20f5be59cd3ed18b44b16ea98167845b4cf735b299bea0403df0`；stage report hash：`0a2cd2228d6a9ff379a398bd7276a0840a2b16e46194540187e0c111cb6e8fac`。
- 仍有效证据及理由：S1.1--S1.11 全部正式 PASS 证据继续有效；producer 到 consumer 的正式计算代码、配置和 schema 无内容差异，仅增加演示文稿和合并可达性。
- 失效证据：无。
- 需要的最小重验：无；Stage 2 只需消费已发布的 G1-EXIT 交接并验证自己的固定状态合同。

### 失败与恢复
- 本次没有实验失败或需要恢复的 attempt。
- 一次远端只读命令因 PowerShell/Bash 变量转义失败；随即改用无变量的精确绝对路径命令并成功，未重复执行昂贵任务，也未修改服务器状态。

### Git 与同步
- 本机/GitHub：`feat/stage1-cpu-evidence` 已整合并发布；本日志提交后远端提交将前移，属于不影响 Stage 1 生产语义的 consumer 文档提交。
- 服务器：仍干净检出 S1.11 producer `3f18b04df8922be9894678ae4842bd999c7e8fd5`，符合复核旧证据的职责；进入 Stage 2 正式执行前再对齐新的 Stage 2 execution commit。
- Agent 文档哈希：六份文件本机与服务器一致。

### 下一步
- 从整合分支建立 Stage 2 工作分支；按 `plan/stage2/` 的依赖图把每个子任务交给独立 `gpt-5.6-luna/xhigh` 子代理实现，主代理只做审查、合并与验收。
