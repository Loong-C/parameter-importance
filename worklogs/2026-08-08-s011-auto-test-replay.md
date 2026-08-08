# 2026-08-08 S0.11 自动测试与独立重放（G9）修复

- 任务范围：确认 S0.11（`stage0.11_test_quality_and_replay`）完成情况，修复 G9 正式链 preflight 阻塞，继续完成 S0.11 正式链与独立重放。包含：定位 G9 阻塞根因、修复 G8 gate record 发布、本地测试、提交推送、服务器同步与重跑。
- 当前状态：进行中
- 工作分支：`feat/stage1-cpu-evidence`

## 2026-08-08 10:40 CST — 确认 S0.11 状态与根因定位

### 目标与范围

- 本阶段要完成什么：按 `Agent/` 文档确认 S0.11/G9 是否完成；若未完成则定位阻塞根因。
- 不在本阶段处理什么：不修改既有正式证据；不重跑任何 GPU 测量；不创建新分支。

### 实际修改

- 代码、配置、文档：无修改，仅只读核验。
- 服务器或外部状态：在服务器 `$DATA_ROOT/tmp/g9_diag.py` 创建只读诊断脚本（后续清理）。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| S0.10/G8 完成确认 | 服务器 worklog + `evidence/stage0/g8-formal/1d76ea6e…/index.json` | status=PASS，next_task_id=stage0.11_test_quality_and_replay | `$DATA_ROOT/evidence/stage0/g8-formal/1d76ea6e…/index.json` |
| G9 正式链状态 | `$DATA_ROOT/evidence/stage0/g9-formal/b8024556…/` | 无 index.json，仅 resolved-config.json，失败残留 | 同上 |
| 根因诊断 | `g9_diag.py`（只读 preflight） | `BLOCKERS=1`；`gate_not_ready\|stage0.G8-C\|前置 Gate 缺少匹配且状态为 PASS 的 GateRecord：stage0.G8-C` | `$DATA_ROOT/tmp/g9_diag.py` |
| G8 环境 gate refs | `environment.json` 中 `evidence_refs` | `gate_stage0_g8/g8_c/g8_s4/g8_s5` 四键均指向 `capacity_envelope`（schema `stage0-g8-capacity-evidence-v1`，非 `gate-record-v1`） | `$DATA_ROOT/evidence/stage0/g8-formal/1d76ea6e…/environment.json` |

### 判定

- S0.11 未完成。G9 正式链被 preflight 阻塞。
- 根因：`task_runtime._verified_gate_ref` 要求 `evidence_refs[gate_<key>]` 指向 schema `gate-record-v1` 的 GateRecord；G8 正式链此前把四个 gate ref 全部指向 `capacity_envelope`，其 schema 是 `stage0-g8-capacity-evidence-v1`，因此验证失败。

### 问题原因与风险

- 风险：修复需要重跑 G8 正式链（发布独立 gate records），将重新执行完整 GPU 容量 suite（约 1 小时、42 次 worker 启动），需按 S0.10 先例暂停 Pile 下载。

### Git与多端同步

- 本阶段无提交。

### 下一步

- 审查工作树未提交修复（`stage0_g8.py` 发布 per-gate GateRecord + 测试改造），本地测试通过后提交推送，再同步服务器并重跑 G8 正式链。

## 2026-08-08 11:20 CST — 审查并本地验证修复

### 目标与范围

- 本阶段要完成什么：审查工作树中的未提交修复，确认其完整解决 gate refs 根因；本地运行相关测试。
- 不在本阶段处理什么：不提交（下一阶段）；不触碰服务器。

### 实际修改

- 代码、配置、文档：仅审查，无新修改。工作树已有修改（用户原有）：
  - `src/param_importance_nlp/stage0_g8.py`：新增 `_gate_key()`；`run_formal_g8_task` 为每个 GATE_IDS 发布独立 `gate-record-v1` artifact（kind `gate_g8_c`/`gate_g8_s4`/`gate_g8_s5`/`gate_g8`）并在 metadata 返回 `gate_refs`；`execute_stage0_g8` 校验 `G8_GATE_REFS_INVALID` 并把 `evidence_refs` 的四个 gate 键指向新发布的 GateRecord commit。
  - 测试文件（g8/g9/g10/g7_recovery）：把 `jsonschema.Draft202012Validator` 校验替换为项目自有 `_validate_project_json_schema`（`cli.py`），并补充 schema_version 断言；`test_stage0_g8.py` 新增两个契约测试。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| G8/G9/G10/G7 本地测试 | `python -m pytest tests/test_stage0_g8.py tests/test_stage0_g9.py tests/test_stage0_g10.py tests/test_stage0_g7_recovery.py -q --basetemp …/.tmp-pytest -p no:cacheprovider` | 31 passed（68.3s） | 本机 pytest 输出 |
| 链路审查 | `task_runtime._verified_gate_ref` / `_extract_schema_payload` / `load_committed_task_artifact` / `GateRecord.from_mapping` | `to_dict()` 顶层含 `schema_version=gate-record-v1` 与 `artifact_hash`，`from_mapping` 严格校验通过；publish 的 formal envelope 满足 `require_formal=True` | `src/param_importance_nlp/runtime/task_runtime.py`、`contracts/status.py`、`runtime/task_artifacts.py` |
| 临时目录清理 | 删除 `.tmp-probe-b4e662…/`、`.tmp-pytest/` | 已清理 | — |

### 判定

- 修复完整正确：`_gate_key` 与 runtime `_gate_evidence_key` 归一化一致；GateRecord payload 满足 preflight 全部校验；测试 31 项全通过。

### 问题原因与风险

- 风险：G8 正式链重跑需 GPU 约 1 小时；服务器正式链必须在新 commit 锚点运行（commit 绑定）。

### Git与多端同步

- 本阶段未提交。

### 下一步

- 提交并推送修复；bundle 同步到服务器；服务器创建新 formal 锚点并重跑 G8 正式链；随后跑 G9 正式链。
