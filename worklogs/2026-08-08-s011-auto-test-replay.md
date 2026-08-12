# 2026-08-08 S0.11 自动测试与独立重放（G9）修复

- 任务范围：确认 S0.11（`stage0.11_test_quality_and_replay`）完成情况，修复 G9 正式链 preflight 阻塞，继续完成 S0.11 正式链与独立重放。包含：定位 G9 阻塞根因、修复 G8 gate record 发布、本地测试、提交推送、服务器同步与重跑。
- 当前状态：**已完成（CHAIN_STATUS=PASS）**
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

## 2026-08-09 01:36 CST — 提交修复并全链重跑（01a14df 锚点）

### 目标与范围

- 本阶段要完成什么：本地提交 per-gate GateRecord 修复并推送；服务器建新 formal 锚点 branch 后全链重跑 bootstrap→G9，使 G9 正式链 PASS。
- 不在本阶段处理什么：不修改既有证据；不重跑 G8 以外其他 GPU 测量（G6/G7 由全链顺带重跑）。

### 实际修改

- 代码、配置、文档：
  - 修复 commit `01a14df63c45e3356b0ebc4ca5f38ce48515766c`（含 1bb8b00 全部修复 + **补回 `configs/stage0/g9-test-matrix-v1.json` 丢失的单个尾随 LF**，2637 字节，canonical 通过，pop artifact_hash 后 hash 匹配 `9195ab77…`）
  - 三段链脚本 `.tmp-chain-s1/s2/s3-01a14df.sh`（s1: bootstrap→attest→verify→materialize→G3→G4→G5→G6→G7→G7R；s2: formalize_g8 含 launch-claims 备份+清空；s3: formalize_g9 含离线 HF env vars）
  - refs 经 `tmp/chain-01a14df-refs.env` 传递
- 服务器或外部状态：
  - 服务器 branch `formal/run-01a14df` @ 01a14df（worktree 干净）
  - backup-g3 清理（01:36:46Z）：备份+移走 `datasets/glue-{sst2,mnli,rte}-pretokenized` + `manifests/{model,data,tokenizer,qualifications}` → `tmp/g3-pre-ed865b0-backup-20260809T013646Z/`（3.7G）——**解决 GLUE_SIDECAR_IDENTITY_MISMATCH 根因**（旧 GLUE derived sidecar raw_asset_id 与新 raw 资产不匹配）

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 段1全链 | 01:37:04→02:31:56Z（~55min），10 步全 PASS | bootstrap→attest（GLUE 重建 27min，无 Mismatch）→verify→materialize→G3(02:09:56)→G4(02:11:24)→G5(02:11:26)→G6(02:29:22, 训练 18min)→G7(02:31:28)→G7R(02:31:56) | `tmp/full-chain-01a14df-s1.log` |
| 段2 G8 正式链 | 07:00:10→~08:25Z（~1h25m），4 gates 全 PASS | gates `stage0.G8-C/S4/S5/G8`；environment_hash=`aa1350cf…`；launch-claims 备份 `tmp/launch-claims-pre-g8-01a14df-backup-20260809T070010Z` | `tmp/full-chain-01a14df-s2.log` |
| 段3 G9 正式链 | 08:14:17→~08:30Z（~16min），PASS | **CHAIN_STATUS=PASS**；index `evidence/stage0/g9-formal/314c40fd…/index.json`：`next_task_id=stage0.12_delivery_and_sync`、`generator_git_commit=01a14df63…`、`g8_index_ref=evidence/stage0/g8-formal/ec75f73c…/index.json`、checked_at 08:27:52Z | `tmp/full-chain-01a14df-s3.log` |

### 01a14df 链最终 refs（全部 PASS）

```
BOOT_REF=evidence/stage0/bootstrap/01a14df63c45e3356b0ebc4ca5f38ce48515766c/index.json
ACQ_REF=manifests/evidence/g3/acquisition/571526ce6cb3d8a86693bda01eef9bbc8fe883c497e19b1632a938f6955d0545.json
VER_REF=manifests/evidence/g3/verification/571526ce6cb3d8a86693bda01eef9bbc8fe883c497e19b1632a938f6955d0545.json
MAT_REF=reports/stage0/g3/11ef76024c2f0c085c02a227eaf46659c560fcf6a13ec0e3de8718d58b26077f/asset-index.json
G3_REF=evidence/stage0/g3-formal/11ef76024c2f0c085c02a227eaf46659c560fcf6a13ec0e3de8718d58b26077f/index.json
G4_REF=evidence/stage0/g4-formal/11ef76024c2f0c085c02a227eaf46659c560fcf6a13ec0e3de8718d58b26077f/index.json
G5_REF=evidence/stage0/g5-formal/9f0ae3111106f5f2ca2b098b94e8e210c6e2d07eb7d78e40b63b3d5a1a91f387/index.json
G6_REF=evidence/stage0/g6-formal/b802a4b3e7367b4f1a9f70b34a5213949f75e25884a5bffdd493847031a1ada0/index.json
G7_REF=evidence/stage0/g7-formal/61ab626611e4979440885c2a8e55b62a4fd93bef21e06f7745ac9ec45d795eae/index.json
G7R_REF=evidence/stage0/g7-recovery-formal/756c6c4f417d51138f4122621b49f2857014886abceea822b9c2d4f1ec000b61/index.json
G8_REF=evidence/stage0/g8-formal/ec75f73c838d566b91e111ec17d41dd506ecc207f9245f12225b563041a31f73/index.json
G9_REF=evidence/stage0/g9-formal/314c40fd22aead6104579a54e46b3c3e24166046f15d5cf45242dda1f4c78962/index.json
```

### 判定

- S0.11 G9 正式链 **CHAIN_STATUS=PASS**，`next_task_id=stage0.12_delivery_and_sync`。阻塞彻底解决。

### 问题原因与风险

- 1bb8b00 链失败根因：fix commit 编辑 matrix 文件时丢失尾随 LF（blob 2636 字节），canonical 校验失败。
- GLUE_SIDECAR_IDENTITY_MISMATCH 根因：旧 GLUE pretokenized 数据集 sidecar 的 raw_asset_id 与新 raw 资产不匹配（metadata 含 code_git_commit）；重跑前备份+删除 derived 数据集即可由 attest 重建。
- 风险：全链重跑使证据全部基于 01a14df 新锚点；旧 1bb8b00 证据目录保留勿删（新链全新哈希目录）。

### Git与多端同步

- 本地 `feat/stage1-cpu-evidence` @ 01a14df；服务器 `formal/run-01a14df` @ 01a14df；已 push GitHub。
- 服务器 worktree 干净，evidence 全在 `$DATA_ROOT`。

### 下一步

- S0.12 delivery & sync：将 G9 index/refs 与 worklog 同步，执行正式交付流程（stage0.12_delivery_and_sync）。

## 2026-08-09 S0.12 交付、工作日志与多端同步（启动与范围冻结）

### 目标与范围

- 本阶段要完成什么：按 `docs/stage0-delivery-runbook.md` 的 9 步不可颠倒顺序完成 S0.12 正式交付：本地提交最终 worklog 与测试 → 推送 GitHub → bundle 快进同步服务器 → 同步 Agent/ 五文件并核对 SHA-256 → 精确删除 bundle → 在最终提交上重跑 G0–G9 全链 → 采集只读三端观察 → 运行 G10 formalizer 发布 READY → 冻结。
- 不在本阶段处理什么：不修改既有证据目录；不重建 G1-D 以外的故障域副本（沿用 2026-07-19 风险接受）；不改动 SSH 拓扑与 Agent/ 之外的文件。
- 当前状态：进行中（本日志为 S0.12 第一条记录，目标/范围冻结）。

### 前置确认（本次会话）

- 本地分支 `feat/stage1-cpu-evidence` @ `01a14df63c45e3356b0ebc4ca5f38ce48515766c`；GitHub `origin/feat/stage1-cpu-evidence` 同提交；服务器 `formal/run-01a14df` @ 同提交、worktree 干净。
- 服务器 G9 正式 index：`next_task_id=stage0.12_delivery_and_sync`、`generator_git_commit=01a14df…`、checked_at `2026-08-09T08:27:52Z`、CHAIN_STATUS=PASS —— S0.11 完成确认。
- 工作树仅含上次会话遗留的 `worklogs/2026-08-08-s011-auto-test-replay.md` 修改（01:36 段，62 行）：记录 G9 全链重跑成功，属用户/前次会话成果，按 `Agent/git.md` 必须随本次提交一起审查、记录、提交、推送，不得丢弃。
- 本地 G10 测试：`python -m pytest tests/test_stage0_g10.py -q --basetemp .tmp-pytest-g10 -p no:cacheprovider` → **9 passed in 41.33s**。

### 执行策略（针对 runbook 第 6 步的说明）

- G10 的 `load_stage0_g9_formal_state` 强制要求 G9 index 的 `generator_git_commit == binding.git_commit`；G9 又强制要求 G8 同提交，逐级向上。当前 G0–G9 证据全部锚定 `01a14df`。
- 本次提交将产生新 commit，三端 HEAD 变化后旧 G9 index 不再匹配，因此 **runbook 第 6 步要求在最终同一提交上重跑 G0–G9 全链**；重跑后全部 gate index 锚定新 commit，G10 才能通过。
- 观察采集器要求：本地工作树干净、三端同 HEAD/同分支、旧 HEAD 是新 HEAD 祖先、无强推、远端 URL 固定、Agent/ 五文件哈希两端一致、bundle 残留不存在、`docs/mathematics.md` 保留。这些条件在重跑前完成 Git/Agent/bundle 三步后自然满足。

### 下一步

- 提交本次 worklog 修改（含 S0.11 01:36 段与本 S0.12 启动段）；用户授权后推送 GitHub；随后按 runbook 依次执行。

## 2026-08-12 19:09 CST — S0.12 G6 失败归档与通信层恢复诊断

### 本轮绑定与已完成项

- 当前本地与服务器仓库均为分支 `feat/stage1-cpu-evidence`、提交 `34f18dbea9f783a3671921a36b8e03e535044e10`，服务器 worktree 干净；本轮未修改 Stage 0 G6 源码。
- 本地回归：`python -m pytest tests/test_stage0_g10.py -q --basetemp .tmp-pytest-g10-current -p no:cacheprovider` → **9 passed in 5.53s**（exit 0）；测试生成的临时目录已清理。
- 服务器上复用的最终提交绑定证据：G3 `evidence/stage0/g3-formal/69c57b1760d8ebf4e54f5f73f2ef5d99da66a86865e2377ccb3ba8d05feb7b6f/index.json`、G4 同哈希目录、G5 `evidence/stage0/g5-formal/06587b5917e901e586d1ade1003a14fc06a3256e57c24f5e4d562157b25b16d4/index.json` 均为 PASS；这些是本轮正式链在 G6 失败前可复用的中间结果，不是最终 G9/G10 证据。

### G6 formal 失败与证据归档

- 最新 G6 formal 的 `collective-00` transcript 位于 `$DATA_ROOT/tmp/g6-failed-34f18db-20260812T0853Z/transcripts/collective-00.json`，`return_code=1`、`timed_out=false`、`duration_seconds=369.712967`；失败点为 `stage0_g6_worker.py:_gather_objects` 的 NCCL `ALLGATHER`，rank 1 watchdog 报告 `Timeout(ms)=300000`，随后进程组退出。没有将该失败目录删除，已从正式 suite 路径移入 `$DATA_ROOT/tmp/` 以避免与重试 transcript 冲突。
- 失败发生在 NCCL 通信等待，不是 G10 读取器修复引起的代码回归：`git diff 9beefb4..34f18db -- src/param_importance_nlp/stage0_g6.py src/param_importance_nlp/stage0_g6_worker.py ops/stage0/formalize_g6.py` 无输出。
- 本轮发现的孤儿诊断进程 PID `815603` 已确认属于本轮 G6/NCCL 诊断（stdout/stderr 指向 `$DATA_ROOT/tmp/smoke-full-out.txt`，带本轮分布式环境变量），按精确 PID 发送 `TERM`，命令 exit 0；随后 `nvidia-smi` 四卡均为 `0 MiB / 0%`。

### 通信层最小复现与判定

- 使用一次性明确脚本复现同一四卡进程组，服务器输出 `$DATA_ROOT/tmp/g6-nccl-diag-34f18db.out`：初始化、tensor `all_reduce`、tensor `all_gather`、object `all_gather`、`barrier` 均输出 `*_OK`，整体 `EXIT=0`；一次性 `.py/.sh` 脚本已删除，输出保留作诊断证据。
- 结论：当前 G6 formal 失败更像一次瞬时或残留通信状态，已有足够证据进行一次干净 G6 重试；在重试成功前不能宣称 G6 PASS，也不能跳过 G6 直接复用历史 G7/G8/G9。

### S0.12 当前判定

- **未完成，状态为 `IN_PROGRESS/BLOCKED_ON_G6_RETRY`。** S0.11 的 G9 `CHAIN_STATUS=PASS` 已确认；S0.12 尚无最终提交上的 G9、三端只读观察、G10 formal 产物或 `READY`/`READY_WITH_APPROVED_EXCEPTIONS`。
- 本日志追加后将形成新的最终提交，因此 `34f18db` 上的旧 G3–G5 不能作为交付闭环依据；后续必须按 runbook 重新完成最终提交同步，并在该最终提交上重跑 G0–G9。

### 下一步

- 审查并提交本日志；非强制推送到 GitHub，生成 bundle 快进同步服务器并再次核对 Agent/ 五文件 SHA-256。
- 清理本次临时 bundle 后，以最终提交重跑 G0–G9；若 G6 重试通过，再采集三端只读观察、运行 G10 formalizer，并只发布新的 READY 产物。

## 2026-08-12 19:34 CST — S0.12 最终提交重跑：G3 attest 的派生资产漂移

### 运行与失败

- S0.12 最终提交已同步为 `0c9cac39a7545d43a5cb4b75cd2a2055c484a17b`；服务器分支 `feat/stage1-cpu-evidence`、worktree 干净，四卡预检为 `0 MiB / 0%`，无项目残留进程。
- 使用最终提交专用脚本 `$DATA_ROOT/tmp/stage0-chain-s1-0c9cac3.sh` 运行 bootstrap→attest→verify→materialize→G3→G4→G5→G6→G7→G7R；bootstrap PASS，随后 attest 在 `2026-08-12T11:30:29Z` 失败，命令 exit 1：`GLUE_SIDECAR_IDENTITY_MISMATCH:raw_asset_id`。
- 失败日志已复制并保留在 `$DATA_ROOT/tmp/stage0-chain-s1-0c9cac3-attest-failure.log`，服务器 formal 证据未被覆盖；失败发生在 G3 派生缓存校验阶段，尚未进入新的 G3/G4/G5/G6。

### 根因确认与恢复

- 只读盘点确认原始目录 `datasets/glue-{sst2,mnli,rte}` 均存在；冲突对象是既有派生目录 `datasets/glue-{sst2,mnli,rte}-pretokenized`，规模分别约 534 MiB、3.2 GiB、22 MiB；无 `attest_g3`、formal gate 或 torchrun 残留进程。
- 按既有 G3 恢复规则，将三份派生目录整体、可恢复地移入 `$DATA_ROOT/tmp/glue-derived-0c9cac3-backup-20260812T113348Z/`；原始资产未删除或改名，备份内容完整保留。随后可由 attest 在新路径重建派生数据并重新生成绑定证据。
- 一次性归档脚本 `$DATA_ROOT/tmp/stage0-archive-glue-derived-0c9cac3.sh` 已完成任务；本地临时脚本与失败输出已清理，服务器脚本将在重跑后精确删除。

### 判定与下一步

- 本次失败是旧派生 sidecar 与当前 raw asset identity 不一致导致的资产缓存漂移，不是 `0c9cac3` 代码失败；但它使当前 G0–G9 重跑尚未形成任何新的 G3–G9 PASS。
- S0.12 继续保持 **`IN_PROGRESS/BLOCKED_ON_G3_ASSET_REBUILD`**；派生资产重建后必须从 attest 起按最终提交完整重跑，不能把旧 G3–G9 证据拼入新链。
