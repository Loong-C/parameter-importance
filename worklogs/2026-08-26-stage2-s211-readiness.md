# Stage 2 S2.11/G2.8 readiness 与闭环控制面记录（2026-08-26）

## 范围与权限边界

- 工作树：`D:/Personal/Code/parameter-importance/.agent-temp/worktrees/s211-readiness`；分支：`agent/s211-readiness`。
- 基线：`feat/stage2-estimator-study` 的 `24b3e448773b1ffc19ac0be2278651e2aa8fe5b`；已复用其祖先中的 `6882ebc` S2.11/G2.8 delivery 实现，没有重写已完成工作。
- 本记录只覆盖 S2.11 readiness、G2.8 fail-closed 验收和 Stage 2 交付控制面；未启动 S2.10/G2.7b 或任何 formal/replay 计算。
- 已完整读取 `Agent/git.md`、`Agent/local.md`、`Agent/remote_access.md`、`Agent/server.md`、`Agent/sync.md`、`Agent/worklogs.md`。所有本机 pytest 临时目录均显式定向到 `.agent-temp/pytest/...`，没有使用默认系统临时目录。
- 根工作树中的 presentation 三个 dirty 文件及既有 S2.08 dirty 文件未触碰、未暂存、未提交。

## Readiness 控制面修改

- S2.10/G2.7b 入口现在实际调用 G2.7b 校验器，兼容当前 generic `GateRecord` 与 producer-specific `stage2-s210-g27b-gate-v1`，并验证 gate ID、Stage、PASS、formal scope/eligibility、measured decision hash 与 canonical artifact hash。
- 对 `task-output-artifact-v1` wrapper 增加严格字段、S2.10 task identity、formal run intent、config/source refs、wrapper hash 和嵌套 gate 校验；decision hash 绑定会沿 wrapper payload 解包到 producer gate。
- 所有文件引用（包括相对引用和 output root）先按 DATA_ROOT 解析再做 containment/symlink 检查；拒绝 cwd 依赖和路径逃逸。
- Stage 2 lineage 强制 schema、完整任务集合、formal eligibility、artifact refs/hash，并拒绝重复 task ID；31M replay 必须明确绑定 `source_artifact_hash` 到 formal_31m boundary。
- delivery role 强制 producer-specific role schema；未知 boundary/role、缺少 consumer commit、非法 estimator decision 都保持 BLOCKED。缺失 role 的 BLOCKED fallback 也使用正确 role schema。

## 验收

| 范围 | 命令结果 |
| --- | --- |
| S2.11 + S2.10 定向回归 | `tests/test_stage2_s211_delivery.py tests/test_stage2_s210_g27b.py`：19 passed（142.43s） |
| 邻接与 schema 回归 | `tests/test_stage2_s209_g27a.py tests/test_stage2_s210_g27b.py tests/test_artifact_schemas_and_loaders.py`：18 passed（15.77s） |

新增回归覆盖 producer-specific gate + task-output wrapper 正向链路、DATA_ROOT 相对路径、31M source hash 缺失、duplicate lineage、delivery role schema mismatch，以及 BLOCKED role 输出 schema。

## 证据身份与当前阻塞

- S2.10 代码实际发布的 `g2.7b-gate.json` 是 generic `gate-record-v1`；控制面同时保留 producer-specific 兼容分支，wrapper 按仓库既有 `task-output-artifact-v1` 合同严格验收。
- 对 `sophgo13-via-lab` 的只读探针观察到 server repo detached HEAD `44f934dd62d1b86fcb951230c81f3bfa647791aa`，当前活动的是 S2.04 进程；未发现真实 S2.10/G2.7b 或 S2.11/G2.8 formal artifact。
- 因真实 S2.10/G2.7b producer evidence 尚未生成，G2.8 仍被真实上游锁定；本任务不以 fixture 或缺失输入推断 PASS，也没有重启、重复或清理服务器作业。

## 后续闭环

待上游正式 producer 交付真实 gate/decision、完整 Stage 2 lineage、formal boundaries、独立 31M replay audit、全部 delivery role artifacts 及同步 commit 后，再运行本控制面生成 append-only G2.8 bundle 并决定 Stage 2 closure；在此之前保持 BLOCKED。

## 2026-08-26 18:00 CST — 独立 S2.11 formal producer/CLI 审计

### 目标与范围
- 在独立分支审计并补齐 S2.11/G2.8 formal delivery consumer 与 one-shot CLI；未访问服务器、未访问 S2.4、未启动 S2.10/G2.7b formal 或 recorded replay。
- 现有 `presentation/` 和 S2.10 两个 dirty 文件属于并发工作，未触碰、未暂存、未提交。

### 实际修改
- 引入前置 G2.0–G2.7b 闭集、G2.3/G2.4a/G2.6 producer-specific schema 与上游 hash 交叉绑定。
- 接受并严格验收真实 `stage2-s210-estimator-decision-v1`，同时保留通用 estimator decision 合同。
- 保留六份 Agent 文档哈希、server/local clean sync、large artifact index、worklog/history role 和 append-only output inventory 校验；CLI 仅物化 instructions，绝不执行 replay command。
- 显式拒绝 local fixture/synthetic 标记进入 formal PASS。

### 验证
| 项目 | 命令/配置 | 结果 | 证据路径 |
| --- | --- | --- | --- |
| S2.11 + S2.10 定向回归 | `tests/test_stage2_s211_delivery.py tests/test_stage2_s210_g27b.py` | 32 passed，退出码 0 | `.agent-temp/pytest/s211-formal-closure-20260826-final/` |
| S2.9 邻接与 schema | `tests/test_stage2_s209_g27a.py tests/test_stage2_s210_g27b.py tests/test_artifact_schemas_and_loaders.py` | 43 passed（含 S2.11），退出码 0 | `.agent-temp/pytest/s211-formal-closure-20260826-adj/` |
| 语法/空白 | `py_compile`、`git diff --check` | 通过 | `.agent-temp/compile/s211-formal-closure-20260826-v2/` |

### 证据身份与有效性
- producer commit：待真实 S2.10 producer 运行时传入并严格要求 40 位 commit；consumer commit：本独立提交形成后记录。
- 当前仅验证控制面与 fixture-shaped negative/positive contracts；没有把 fixture 结果、dirty 主线或缺失 server evidence 宣称为 G2.8 PASS。
- 真实执行前置：data root 下的全部 gate/boundary/role/replay artifacts、实际 Agent hashes、server/local sync report 和 clean execution commit 必须逐项通过；否则 bundle 保持 BLOCKED。

### 失败与恢复
- 本次无昂贵运行、无服务器 attempt、无 replay attempt；没有新增外部状态或临时 bundle。

### Git 与同步
- 本地分支：`fix/s211-formal-closure`；当前提交前基线为 detached `9e5c4315444530371678205d5ee5c3d549e7f084` 的实际工作树，S2.10 两个 dirty 文件保持原状。
- 本次不执行服务器同步或 Agent 文档复制；后续真实交付须按 `Agent/sync.md` 逐文件核对六份文档哈希。

### 下一步
- 待真实上游 producer 与同步证据到位后，使用 `ops/stage2/run_s211_delivery.py` 的全参数命令，在新的 DATA_ROOT output root 运行一次并按 gate status 接受或保留 BLOCKED。
