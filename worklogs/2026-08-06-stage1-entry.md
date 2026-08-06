# 2026-08-06 Stage 1 入口基线与 CPU 可完成部分收口

- 任务范围：完成 Stage 1 中不依赖 GPU/服务器即可完成的部分——S1.1 入口快照与
  数学契约证据、S1.2–S1.6（及 1.09/1.10/1.11）的本地任务执行证据、全量本机测试
  基线，并修复与 e6d6863（Stage 0 关闭重要性）相关的过期测试。
- 不在此范围：G1-ENTRY 的 GPU 健康复验、S1.7 正式单卡、S1.8 DDP、G1-NUMERIC
  GPU 侧、G1-RESUME 服务器侧与 G1-EXIT——均被 S0.9/G0-G 的 GPU 问题阻塞。
- 当前分支：`feat/stage0-completion` @ `975f83c`（本机与服务器一致）。

## 入口快照（S1.1 只读部分）

| 项目 | 本机 | 服务器 sophgo13 |
|---|---|---|
| 分支 / HEAD | `feat/stage0-completion` / `975f83c` | `formal/run-975f83c` / `975f83c` |
| 工作树 | 干净（本报告随提交入库前仅新增/修改下述文件） | 干净 |
| `Agent/*.md` | 5 份 | 5 份，SHA-256 与本机 5/5 一致 |
| DATA_ROOT | — | 3.5T，可用 2.6T（23%），inode 1%，750 sophgo13 |
| 缓存/tmp 解析 | 本机不承载大型缓存 | 运行时由环境注入到 `$DATA_ROOT/cache`，非交互 shell 无默认回退 |

`Agent/*.md` 哈希（两端一致）：git.md `183f4ba7…`、remote_access.md
`795c677e…`、server.md `9f2d4370…`、sync.md `1bf84f83…`、worklogs.md
`4a61b34b…`。

GPU 阻塞事实（Stage 1 CPU 不依赖，但 G1-ENTRY 正式通过依赖其解除）：
服务器重启后当前 boot 为 `7a54a465…`，允许名单内 `0000:a4:00` 在本 boot 出现
Xid 63 row-remap（pending），管理员 finalizer fail-closed；S0.9 正式链停在
bootstrap `STAGE0_BOOTSTRAP_G0_G_CURRENT_RUNTIME_MISMATCH`。

## 实际修改

- `configs/stage0/g9-test-matrix-v1.json`：末尾 CRLF 改为 canonical JSON（LF）。
- `tests/test_task_artifacts_and_default_runners.py`：训练任务重要性 bundle
  断言从 stage0.06 改为 stage1.07（Stage 0 已关闭重要性）。
- `tests/test_full_fixture_pipeline.py`：stage0.06 smoke 的
  `importance_snapshot` 断言改为 `None`，并修正注释。
- `src/param_importance_nlp/experiments/full_fixture_pipeline.py`：同步修正
  stage0.06 smoke 覆盖说明（在线重要性由 Stage 4 minimal_complete_loop 覆盖）。
- `tests/test_stage79_run_ready_completion.py`：Stage 7 剪枝输入源从
  stage0.06 改为 stage1.07（其输出含 checkpoint commit + importance bundle）。
- 新增 `ops/stage1/generate_cpu_evidence.py`（可复现的 Stage 1 CPU 证据生成器）。
- 新增 `reports/stage1/`：gap-analysis、test-baseline、cpu-evidence-20260806、
  stage1-cpu-status。
- 新增本工作日志。

## 实验与验证

| 项目 | 命令 | 结果 | 证据路径 |
|---|---|---|---|
| 全量本机测试 | `python -m pytest -q` | 967 passed，10 skipped，0 failed（跳过均为 Windows 环境限制：symlink 权限、POSIX 权限语义、Gloo 无设备） | `reports/stage1/stage1-cpu-test-baseline-20260806.json` |
| Stage 1 CPU 证据 | `python ops/stage1/generate_cpu_evidence.py --date 20260806` | 10/10 任务 PASS（local_fixture；`gate_status=NOT_RUN`） | `reports/stage1/cpu-evidence-20260806/` |
| 差距分析 | 子代理只读分析 | 见报告 | `reports/stage1/gap-analysis-20260806.md` |

## Gate 状态（CPU 可完成部分）

- G1-CONTRACT：本地契约证据已生成（math doc + plan hash、冻结公式），正式判定
  `NOT_RUN`；S1.2–S1.6 的 API 冻结仍以 G1-CONTRACT 正式通过为前提。
- G1-REGISTRY / G1-ORACLE / G1-GRAD / G1-EST / G1-STEP：本地任务 runner 全部
  PASS（`local_validation_status=PASS`），对应单元测试全绿；正式 gate 记录
  `NOT_RUN`，需服务器侧按正式流程发布。
- G1-ENTRY / G1-SINGLE / G1-DDP / G1-NUMERIC / G1-RESUME / G1-EXIT：被
  S0.9/G0-G GPU 阻塞，`BLOCKED`。

## 下一步

- 管理员处理 `0000:a4:00`（reset + 干净 boot）→ 复验 G0-G → S0.9 链通过后，
  消费 G10 handoff 补全 S1.7 配置占位符，再执行服务器侧 G1-SINGLE/DDP/NUMERIC/
  RESUME/EXIT。
- 本机证据已就绪，可先做 S2.1/S3.1 预注册文档与 Stage 2/3 代码级准备。
