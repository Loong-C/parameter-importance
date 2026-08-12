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

## 2026-08-12 20:14 CST — S0.12 G3 materialize 旧发布物冲突归档

### 运行与失败

- `32cdad42c821be9f74df175d4701ca0e4489f86f` 上的正式链已完成 bootstrap PASS、attest PASS（13 assets，acquisition `manifests/evidence/g3/acquisition/0bdb873d0fb61933efc0e4c29557dddab24b3e087932b1142c6e06a8b733ce0d.json`）、verify PASS（verification 同一 artifact identity，`verification_sha256=b70d2deb96c464319d7ba12ac3e9fa533f9cbb67a71a3d070faa7fe5ecc3a478`）。
- materialize 在 `2026-08-12T12:09:17Z` 安全失败，exit 1：`G3AssetPublicationError: existing READY does not descend from the supplied VERIFIED input`。完整输出已保存为 `$DATA_ROOT/tmp/stage0-chain-s1-32cdad4-materialize-failure.log`；没有覆盖旧 READY 或旧 evidence。

### 根因与精确恢复

- 发布器代码明确要求既有 canonical READY 必须由本次 VERIFIED 输入派生；服务器已有 13 个 layout 指定的 READY manifest 与 13 个 qualification 属于旧 acquisition/verification 链，不能与本次新派生链混用。
- 按 layout 精确归档 26 个发布文件（13 manifest + 13 qualification）到 `$DATA_ROOT/tmp/g3-publications-32cdad4-backup-20260812T121415Z/`，脚本输出 `ARCHIVED_COUNT=26` 且对全部备份文件记录 SHA-256；没有移动 raw/derived 数据、旧 acquisition/verification、旧 G3/G4/G5/G6/G7/G8/G9 evidence。
- 本轮旧发布物归档是可恢复操作；materialize 后续将从本次 VERIFIED 输入新建 canonical READY/qualification，再继续正式 G3。

### 判定与下一步

- 当前仍无 `32cdad4` 上新的 G3–G9 PASS；S0.12 状态为 **`IN_PROGRESS/BLOCKED_ON_G3_PUBLICATION_REBUILD`**，不能使用旧 G3–G9 拼接完成。
- 服务器失败日志与 26 文件备份已留存；本地一次性链脚本、归档脚本和输出将在下一次提交前清理。必须将本次记录提交后再次同步三端，并从最终新 commit 完整重跑 bootstrap→G9。

## 2026-08-12 20:25 CST — S0.12 G3 attest 二次派生半成品清理

### 运行与失败

- `d2fc6d7c0323f5ad00f4987891184c5b5644fe00` 上重新启动 bootstrap→G9；bootstrap 在 `2026-08-12T12:22:09Z` PASS，attest 于 `2026-08-12T12:22:14Z` 再次 exit 1，仍为 `GLUE_SIDECAR_IDENTITY_MISMATCH:raw_asset_id`。
- 失败输出保留在 `$DATA_ROOT/tmp/stage0-chain-s1-d2fc6d7-attest-failure.log`；当时无 attest/materialize/formal/torchrun 残留进程。

### 根因确认与恢复

- 读取三份 `stage0-glue-derived-build.json` 确认：`glue-{sst2,mnli,rte}-pretokenized` 目录中的 sidecar 仍绑定上一轮 `generator_git_commit=32cdad42…` 与旧 `raw_asset_id`；它们是上一轮 materialize 失败后留下的半成品，不是当前 `d2fc6d7` 输入。
- 将这三份派生目录完整移入 `$DATA_ROOT/tmp/glue-derived-d2fc6d7-backup-20260812T122519Z/`，输出 `ARCHIVED_COUNT=3`，目录规模约 534 MiB、3.2 GiB、22 MiB；原始 `datasets/glue-{sst2,mnli,rte}` 未移动。

### 判定与下一步

- S0.12 仍为 **`IN_PROGRESS/BLOCKED_ON_G3_ASSET_REBUILD`**；当前没有新的 G3–G9 PASS。下一轮最终提交需从 attest 重新开始，重建派生数据、重新 materialize 13 个 READY 后再继续正式 gate 链。
- 本地临时链脚本、归档脚本和输出在提交前清理；服务器失败日志、两轮派生目录备份和 26 个旧发布物备份均保留在 DATA_ROOT/tmp 供审计与恢复。

## 2026-08-12 21:52 CST — S0.12 新最终提交 G3–G5 通过与 G6 transcript 冲突归档

### 本轮提交与 G3–G5 结果

- 本轮最终提交 `19c02a01b2c0e65f7cb7a41236196428479b2a26` 已推送 GitHub，并通过 bundle 快进同步到服务器；本地、GitHub、服务器均位于 `feat/stage1-cpu-evidence`。
- 在该提交上，bootstrap、G3 attest、G3 verify、G3 materialize、formal G3、formal G4、formal G5 均通过。关键 refs：
  - G3 acquisition：`manifests/evidence/g3/acquisition/d216848623dc07c167e4937186257ec52b3f6a034577de0787c9ec7227ca9a62.json`
  - G3 verification：`manifests/evidence/g3/verification/d216848623dc07c167e4937186257ec52b3f6a034577de0787c9ec7227ca9a62.json`，`verification_sha256=c0be24d7e1dda0ec981d88daf350a495ee55249b777c2a4419f90f34984f74be`
  - G3 materialize：`reports/stage0/g3/2b7c041cb06b86334cdd6ee84c34cfdb3289373075032a4a58fc714a0ea956ef/asset-index.json`
  - G3 formal：`evidence/stage0/g3-formal/2b7c041cb06b86334cdd6ee84c34cfdb3289373075032a4a58fc714a0ea956ef/index.json`
  - G4 formal：`evidence/stage0/g4-formal/2b7c041cb06b86334cdd6ee84c34cfdb3289373075032a4a58fc714a0ea956ef/index.json`
  - G5 formal：`evidence/stage0/g5-formal/6a9c1054baa9733414e959111c5531dde83c5c298b4f618363e7dd5086d5b334/index.json`
- G5 在 `2026-08-12T13:25:59.715082Z` 完成 PASS；其 `generator_git_commit=19c02a0...`，`g4_index_ref` 指向本轮 G4，未生成 G6/G7/G8/G9 结果。

### G6 失败、重试保护与证据归档

- 首次 G6 formal 在 `2026-08-12T13:26:07Z` 启动，`collective-00` 于 `2026-08-12T13:32:29.133877Z` 结束，`return_code=1`、`duration_seconds=381.885166`、`timed_out=false`。失败根因是四卡 NCCL `ALLREDUCE`：`SeqNum=3`、`Numel=65536`、`Timeout(ms)=300000`；rank 1/0/3 watchdog 均记录超时，随后 torchrun 退出。四卡计算进程均已回收，未产生 G6 formal index。
- 首次失败留下的 suite 目录为：`evidence/stage0/g6-suite/7ceb409b5ed533f576d57379ba3ed6033afb56357a38618315079088e7e94dfa/7a0c00a8b516e00dc52aeec8feaa3b4cba61a271cff93886ac5282a86a766586/`。其中保留 `environment.json`、`resolved-config.json`、`plans/collective-00.json`、`transcripts/collective-00.json`；transcript SHA256 为 `2bb6a964266377e9d63c160af5c3b21d3aa48ece0cc6ba28fc456a4a7c459b0f`。
- 随后重试同一 config/environment 时，正式入口检测到既有 `transcripts/collective-00.json`，按设计返回 `G6_LAUNCH_TRANSCRIPT_COLLISION`；这是失败证据保护，不是新的通信结论。确认无 `formalize_g6`、`stage0_g6`、`torchrun` 或 worker 残留，四卡均为 `0 MiB / 0%`。
- 为避免覆盖失败证据，已将上述精确 suite 目录整体移入：`$DATA_ROOT/tmp/g6-failed-19c02a0-20260812T134955Z/`，并保留四个文件的 SHA256；归档前检查无计算进程。

### 当前判定与下一步

- S0.12 仍未完成，当前为 **`IN_PROGRESS/BLOCKED_ON_G6_RETRY`**：本轮 G3–G5 PASS 已确认，但没有本轮 G6/G7/G7R/G8/G9、三端只读同步观察、G10 formal 或 `READY`/`READY_WITH_APPROVED_EXCEPTIONS`。
- 由于本节工作记录会形成新 Git 提交，不能继续复用 `19c02a0` 的中间证据作为交付闭环。下一步先完成本记录提交、GitHub push、server bundle 快进同步及 Agent 五文件 hash 核对；再归档本轮派生/发布物并从新提交重新执行 bootstrap→G9，保留失败证据，最后才进行 G10。

## 2026-08-12 22:08 CST — S0.12 新链环境入口失败记录

### 运行与失败

- `96f3216969c1215337704aa7357aa40d6643f1a0` 已完成 GitHub、服务器 bundle、Agent 五文件同步；服务器分支为 `feat/stage1-cpu-evidence`，工作树干净。
- 为该提交准备的 G0–G5 链在 bootstrap 入口立即失败：临时脚本错误调用不存在的 `python` 命令，服务器仅提供 `/usr/bin/python3` 及 Stage0 虚拟环境 `$DATA_ROOT/envs/parameter-importance-stage0-1bd963c65f75/bin/python`；退出码 `127`，失败时间 `2026-08-12T14:06:52Z`。
- 进一步核对确认临时脚本还使用了不存在的简化入口 `bootstrap_stage0.py`、`attest_g3_assets.py`、`materialize_g3_assets.py`；正式链应使用仓库中已验证的 `bootstrap_formal_stage0.py`、`attest_g3_materialization.py`、`materialize_and_publish_g3.py`，并以虚拟环境解释器执行。该错误未启动 GPU、未生成本轮 G0–G5 evidence，也没有留下计算进程。

### 恢复与判定

- 失败没有修改正式资产；其退出输出已保留在 `$DATA_ROOT/tmp/full-chain-96f3216-s1.log`，错误脚本会在修正后清理。
- S0.12 仍为 **`IN_PROGRESS/BLOCKED_ON_FINAL_CHAIN_RETRY`**。下一步将本节记录提交并同步三端，以新提交使用正式入口从 bootstrap 重新执行；此前归档的 `$DATA_ROOT/tmp/g3-rebuild-96f3216-20260812T140307Z/` 继续保留并作为可恢复审计副本。

## 2026-08-12 23:20 CST — S0.12 f51f317 正式链 G3–G5 通过与 G6 NCCL 失败归档

### 正式链与 G3–G5

- `f51f3174ef4c9b2dbf83bebb4254dba60907f24c` 已在 GitHub、服务器和 Agent 五文件上同步；正式入口使用 Stage0 虚拟环境 `$DATA_ROOT/envs/parameter-importance-stage0-1bd963c65f75/bin/python`。
- bootstrap PASS：`evidence/stage0/bootstrap/f51f3174ef4c9b2dbf83bebb4254dba60907f24c/index.json`，`environment_hash=68e947b6ba7ed135b9abf981bf0efc47427aa3d36cb9d7729dc6d89ad0e89bb7`。
- G3 attest PASS（13 assets）：`manifests/evidence/g3/acquisition/a040ed22c66fb27b0817cde2250a8069130b492ef799d9dff10664be92bcdbd7.json`，`acquisition_sha256=a040ed22c66fb27b0817cde2250a8069130b492ef799d9dff10664be92bcdbd7`。
- G3 verify PASS（13 assets）：`manifests/evidence/g3/verification/a040ed22c66fb27b0817cde2250a8069130b492ef799d9dff10664be92bcdbd7.json`，`verification_sha256=cb0ad5d599486728b2c6cb076f31d89693d84ea003b0d1cda3dec1f85fde487d`。
- materialize PASS（13 assets）：`reports/stage0/g3/91c8e478d88197c8f4007980926f7380339ff3d1670ddf50e1e16cad9d0f7231/asset-index.json`；formal G3 与 G4 均在同一 `91c8e478…` artifact 目录 PASS。
- formal G5 PASS：`evidence/stage0/g5-formal/d01618a1ed1d9dabb8bc3aac8581182bcc09c354f9ba0a547b3536f81040c05b/index.json`，14 个 worker report 齐全，GPU 进程结束后已释放。

### G5 并发保护与 G6 失败

- G3→G4 阶段因原始 SSH 会话断开，正式 s1 任务仍在服务器运行；误启动的后台 s2 在 G5 入口因检测到既有 GPU worker，按设计返回 `G5_SELECTED_GPU_NOT_EXCLUSIVE_BEFORE_SUITE`。该并发尝试未进入 G5 suite，失败日志保留在 `$DATA_ROOT/tmp/full-chain-f51f317-s2.log`；正式 s1 未被中断并最终产生上述 G5 PASS。
- 使用正式 G5 ref 单独启动 G6→G7→G7R 后，G6 `collective-00` 在 `2026-08-12T15:11:33.249466Z` 启动、`2026-08-12T15:17:42.787173Z` 结束，`duration_seconds=369.537707`、`return_code=1`、`timed_out=false`。四卡 NCCL 在 rank 1 的 `all_gather_object` 处失败：`SeqNum=2`、`NumelIn=394`、`NumelOut=1576`、`Timeout(ms)=300000`；四卡均随后退出，未生成 G6 formal index。
- 失败 transcript 原路径 suite 为 `evidence/stage0/g6-suite/1482d427164518946cecb10296b52c04417e8ef2780453feac94391136e7e4b3/14352f68a03e8060a63472a1f74aeeadad3c9b30b63118fff4b5ff0a211cf3de/`，transcript artifact hash 为 `2d47add9f4bef068795efe9665f123f3df51e25badeba93fbd0c33afd1369e74`，文件 SHA256 为 `0f9641f5c80701b00ce21d18ad684aa65afb2fef8045c4ba39ea8823078ab977`。
- 确认无 G6 formalizer、torchrun 或 worker 残留，四卡为空闲后，将精确 suite 整体移入 `$DATA_ROOT/tmp/g6-failed-f51f317-20260812T151903Z/`；四个 suite 文件 SHA256 已随归档输出保留。

### 当前判定与下一步

- S0.12 仍未完成，状态为 **`IN_PROGRESS/BLOCKED_ON_G6_RETRY`**：f51f317 上 G0–G5 PASS 已确认，但没有新的 G6/G7/G7R/G8/G9、三端同步观察、G10 formal 或 READY 产物。
- 本节记录会形成新提交，因此下一步先提交并同步三端；新提交必须再次从 bootstrap→G5 产生绑定证据，然后在确认四卡独占后重试 G6。当前 G6 失败证据和 3.7 GiB G3 重建备份均保留在 DATA_ROOT/tmp，不覆盖旧证据。

## 2026-08-13 00:03 CST — S0.12 9eb7256 G3 派生成功但 verify actor 约束失败

### 本轮启动与保护

- 本轮代码绑定提交为 `9eb7256000d91d9587494b21c2ccc26b71914507`；启动前本地、GitHub、服务器均在 `feat/stage1-cpu-evidence`，服务器 worktree clean，四张卡均为 `0% / 0 MiB`。
- 为避免覆盖旧发布物，已先将当前 9eb7256 之前的 3 个派生目录与 26 个 canonical 发布/qualification 文件整体归档到 `$DATA_ROOT/tmp/g3-rebuild-9eb7256-20260812T152500Z/`；归档计数 `29`，约 `3.7 GiB` 数据与 `152 KiB` manifest，原始数据未移动。
- 正式链以服务器后台 PID `862892` 启动，使用冻结解释器 `$DATA_ROOT/envs/parameter-importance-stage0-1bd963c65f75/bin/python`；未启动第二份链。bootstrap PASS：`evidence/stage0/bootstrap/9eb7256000d91d9587494b21c2ccc26b71914507/index.json`，`environment_hash=baf276a6ce0e62adc92605493841951c9ec7fe2c407424a47fb530e099cd8d8b`。

### G3 attest 与失败证据

- G3 attest 成功重建并发布 `13` 个候选；执行时间约 `2026-08-12T15:32:48Z` 至 `16:03:00Z`，acquisition ref 为 `manifests/evidence/g3/acquisition/92dac060cf648083bd8dfd0e8d2d04e6f635db884ce18c9bc8617f73485ffbf0.json`，`acquisition_sha256=92dac060cf648083bd8dfd0e8d2d04e6f635db884ce18c9bc8617f73485ffbf0`。
- 三个派生目录均已完整生成：`datasets/glue-sst2-pretokenized`（约 `534 MiB`）、`datasets/glue-mnli-pretokenized`（约 `3.2 GiB`）、`datasets/glue-rte-pretokenized`（约 `22 MiB`）。
- verify 入口随后退出，exit `1`，根因是本轮脚本错误复用了同一个 `actor_instance_id`：`G3LifecycleEvidenceError: verifier actor_instance_id must differ from the fetcher`。因此没有生成本轮 verification ref，也没有进入 materialize/G3/G4/G5；该失败不是数据内容校验失败。
- 失败完整输出保留在 `$DATA_ROOT/tmp/full-chain-9eb7256-s1.log`；退出后确认无 attest、verify、materialize、formalizer、torchrun 或 worker 残留，GPU 全部空闲。正式资产未被宣称为本轮 VERIFIED/READY。

### 当前判定与恢复动作

- S0.12 仍未完成，状态保持 **`IN_PROGRESS/BLOCKED_ON_G3_VERIFY_ACTOR_RETRY`**。本轮只证明 9eb7256 的 G3 派生构建成功，不能作为 G3–G9 交付闭环。
- 下一步先归档本轮新生成的 3 个派生目录与 acquisition 失败链的相关临时产物，保留 sha256 与原始数据不变；随后提交并同步本节 worklog 到 GitHub/服务器，使用独立的 fetcher/verifier actor 重跑 bootstrap→G5，再在四卡独占核验后推进 G6。

### 归档结果

- 本轮 3 个派生目录已整体移入 `$DATA_ROOT/tmp/g3-rebuild-9eb7256-verify-actor-failure-20260812T160500Z/`，`ARCHIVED_COUNT=3`，约 `3.7 GiB`；归档内 `sha256sums.txt` 已生成，原始 `datasets/glue-{sst2,mnli,rte}` 未移动。
- 归档清单文件 SHA256：`9b4d728dc543eee9cf7aa336f6ba4c4e80a5f94627376b30bb9692ae19df9046`。失败 acquisition 仍保留在 DATA_ROOT 的 immutable manifests 目录，未删除或覆盖。

## 2026-08-13 01:19 CST — S0.12 64172a7 G0–G5 通过但 G6 NCCL 重复超时

### 本轮 G0–G5

- `64172a79daeb91cc91a4ddc9f8c6bce6a82f3571` 已在本地、GitHub、服务器同步，服务器 worktree clean；正式链使用冻结 Stage0 venv，且 G3 fetcher/verifier/gate 使用三个不同 UUID。
- bootstrap PASS：`evidence/stage0/bootstrap/64172a79daeb91cc91a4ddc9f8c6bce6a82f3571/index.json`，`environment_hash=f4f11350592d175fa398cbcee34f2aec41ad97a7108c243cc1dd84c71b49d5ec`。
- G3 attest PASS：13 assets，acquisition `manifests/evidence/g3/acquisition/f76f8f2692a24b64de742dfafcabf0960c425beaac88db52244694d1fd589a76.json`，`acquisition_sha256=f76f8f2692a24b64de742dfafcabf0960c425beaac88db52244694d1fd589a76`。
- G3 verify PASS：13 assets，verification `manifests/evidence/g3/verification/f76f8f2692a24b64de742dfafcabf0960c425beaac88db52244694d1fd589a76.json`，`verification_sha256=bd912861a9cbae2b49d595528ac30d14b6c5d26848111e51a42ac677162f26d4`；materialize PASS，resolution `650d98014f57e8790602174c1149302888b430560e1267d11c2e06791b161ac0`。
- formal G3 PASS：`evidence/stage0/g3-formal/650d98014f57e8790602174c1149302888b430560e1267d11c2e06791b161ac0/index.json`；formal G4 PASS：同 resolution 下 `evidence/stage0/g4-formal/650d98014f57e8790602174c1149302888b430560e1267d11c2e06791b161ac0/index.json`。
- formal G5 PASS：`evidence/stage0/g5-formal/156fe86328ab3e3581ab37e7285e10d274e404580667bcbf6dcc62c77382c2a0/index.json`，14 个 worker report 完整；父链退出后四卡均为 `0% / 0 MiB`。

### G6 失败证据

- 单独绑定上述 G5 ref 启动 G6，启动前无 compute app，随后四卡各启动一个 worker（PID `877590`–`877593`）。
- `collective-00` transcript：`$DATA_ROOT/evidence/stage0/g6-suite/04e2c72da172251dc3036cf7e1c7b35a5c76853dbe6e7da2fca1e267704f7ef4/1a73966ee114a34f1e3b1dcd0b654aed42a6e111d2731eb30ce57edb8f20faa2/transcripts/collective-00.json`；artifact hash `bd5fdcab706e5adea7cb4cb1d7e3350ec383145ddc35592865d3cd04a92b36f0`。
- 运行时间 `2026-08-12T17:05:45.740570Z`–`17:11:55.484915Z`，`duration_seconds=369.744345`，`return_code=1`，`timed_out=false`，无残留 compute PID。rank 1 在 `all_gather_object` 的 NCCL `ALLGATHER` 再次超时：`SeqNum=2`、`NumelIn=394`、`NumelOut=1576`、`Timeout(ms)=300000`；rank 0/2/3 随后收到 watchdog dump 并被 SIGTERM，G6 未生成 formal index。
- 本轮完整 G6 日志：`$DATA_ROOT/tmp/full-chain-64172a7-g6.log`。退出后确认无 formalize_g6、torchrun 或 worker 残留，四卡恢复 `0% / 0 MiB`。这与 f51f317 轮次相同的 NCCL object-gather timeout，说明当前阻塞是可重复的 G6 通信层问题，不是 G3 actor 约束或 G0–G5 链失败。

### 归档与当前判定

- 失败 suite 已整体移入 `$DATA_ROOT/tmp/g6-failed-64172a7-20260812T171903Z/`，4 files，sha256 清单 `13019e440fdd84fea0dae1219c294909eed7cc8a26867b4fb50aea4d2f333572`。
- 本轮 3 个派生目录与 26 个 canonical 发布/qualification 文件已整体移入 `$DATA_ROOT/tmp/g3-rebuild-64172a7-g6-failed-20260812T171903Z/`，约 `3.7 GiB`，sha256 清单 `c9a6f508673ea58a2eaf60433a6fc2b7c6dcfa6272234c911cd1ed2f517a69ec`；原始数据未移动。
- S0.12 仍未完成，状态为 **`IN_PROGRESS/BLOCKED_ON_G6_NCCL_RETRY`**。本轮只能确认 G0–G5 PASS，不能推进 G7/G7R/G8/G9 或 G10/READY。下一步需先把本记录同步三端；随后应针对固定的 `all_gather_object` 小对象启动/通信路径做最小、可审计的修正或诊断，再以新 commit 重跑完整 G0–G5 后重试 G6。

## 2026-08-13 01:52 CST — S0.12 G6 NCCL 组合拓扑诊断与最小修复决策

### 诊断边界

- 诊断基线为提交 `f6f87ba198cb38fc5c0ebaeb93e23151d95b29d8`，未修改正式 G6 代码或正式 evidence 目录；服务器使用冻结解释器 `$DATA_ROOT/envs/parameter-importance-stage0-1bd963c65f75/bin/python`，PyTorch `2.12.1+cu126`、CUDA runtime `12.6`、NCCL `2.29.3`。
- GPU0 在整个诊断期间被外部 `/home/sophgo13/lyx` 下的 MAE 训练占用：PID `880257` 及其 worker，`nvidia-smi` 约 `8056 MiB`；该进程不属于本项目，未被终止。正式四卡 G6 未在该竞争状态下启动。
- 其余 GPU1–3 空闲；系统日志未发现新的 NVIDIA Xid/ECC 错误。拓扑为 PCIe：GPU1↔GPU2 为 `PIX`，GPU1/GPU2/GPU3 的其他连接为 `PXB`，NVLink links inactive。

### 最小复现结果

- 三卡 GPU1–3 重复 3 轮 object-gather 诊断：`$DATA_ROOT/tmp/g6-nccl-diag-f6f87ba-gpu1-3.out`，SHA-256 `20aed680b39db36ebd9a449884fd4b88caa943f89aa7e7b5a06d0629802e0d44`。每轮均未完成；NCCL `all_gather_object` 底层 `ALLGATHER` 在不同轮次分别于第 3 次、第 1 次及初始化后的首个 object gather 超时，失败位置与正式 G6 同类且不依赖四卡 rank 映射。
- 同一 GPU1–3 组合的拆分诊断 `$DATA_ROOT/tmp/g6-collective-split-f6f87ba.out`，SHA-256 `fe0ed7586cc08330d25cb510f583356c85e77091f705a054c9b4540c11d121c6`：默认 NCCL 纯 tensor `all_reduce` 在第二个 collective 超时；Gloo object gather 连续 20 次、barrier 和退出全部通过（`rc=0`）。
- 空闲卡逐对 NCCL 诊断 `$DATA_ROOT/tmp/g6-nccl-pairwise-f6f87ba.out`，SHA-256 `c7185305b9c7b6cafc1d5794662b0769725c7789822171394f62273b421819d9`：GPU1–2、GPU1–3、GPU2–3 三对均完成 20 次 tensor `all_reduce`、20 次 tensor `all_gather`、barrier，三组均 `rc=0`。
- 三卡 transport 矩阵 `$DATA_ROOT/tmp/g6-nccl-transport-matrix-f6f87ba.out`，SHA-256 `7ed0424eb083c550ddebd6f62e1e44106529b71ebfe892a4a839fc6bb15f04de`：默认环境 `rc=124`；仅 `NCCL_P2P_DISABLE=1` `rc=0`；仅 `NCCL_SHM_DISABLE=1` `rc=124`；两者同时设置 `rc=0`。每个变体执行 5 次 tensor `all_reduce`、5 次 tensor `all_gather` 和 barrier。

### 判定与最小修复

- 逐对通过而三卡组合失败，且 `NCCL_P2P_DISABLE=1` 单独即可恢复，说明当前阻塞是本机多卡 PCIe/NCCL P2P transport 选择问题，不是 G6 数据、对象内容、rank 顺序或 G3–G5 语义链问题。该结论仍不替代正式四卡 gate。
- 采用最小可审计修复：在 G6 formal launcher 显式注入 `NCCL_P2P_DISABLE=1`，并将 `nccl_p2p_disable=1` 写入 worker protocol/schema、由 worker 强制校验并在 collective report 中回显；不改变消息规模、20/50/median-of-3 测量协议、semantic/recovery/failure-rank 测试。
- 本节不宣称 G6 PASS。修复提交后必须从该新提交重新执行 bootstrap→G5，再在四卡独占且无外部竞争进程时重试 G6→G9；旧 `64172a7` 的 G0–G5 证据不能跨提交复用。

### 当前判定

- S0.12 仍未完成，状态保持 **`IN_PROGRESS/BLOCKED_ON_G6_NCCL_RETRY`**。本轮完成了通信层定位和修复决策，尚无新提交上的 G0–G9、G10 formal 或 `READY` 产物。

### 修复实现与本机回归

- 已在工作树实施上述最小修复：G6 formal launcher 固定注入 `NCCL_P2P_DISABLE=1`；worker plan v2、worker report、worker 预检和 collective replay 均要求并回显 `nccl_p2p_disable=1`；Stage 0 G6 计划文档同步记录该冻结 transport 条件。消息规模、20/50/median-of-3、semantic、recovery 和 controlled-failure 合同均未改变。
- 本机回归：`python -m pytest tests/test_stage0_g6.py -q --basetemp .tmp-pytest-g6-fix -p no:cacheprovider` → **4 passed**；`python -m pytest tests/test_stage0_g10.py -q --basetemp .tmp-pytest-g10-fix -p no:cacheprovider` → **9 passed**；两个 G6 JSON schema 均可由 `python -m json.tool` 解析，相关源文件 `compileall` 通过。
- 该修复将形成新的 generator commit；此前 `64172a7` 的 G0–G5 evidence 不能跨修复提交复用。修复提交同步后，必须从 bootstrap 重新生成 G0–G5，再在四卡独占、无外部 GPU 竞争的窗口重试 G6→G9。

## 2026-08-13 03:24 CST — 344a326 G6/G7 通过、G7R DDP NCCL 失败

### 本轮 G0–G7

- 本轮 generator commit 为 `344a326cd568223e4501692880a74dd5662f3bfb`，服务器分支 `feat/stage1-cpu-evidence`，正式链使用冻结 Stage0 venv；启动前四卡均为 `0 MiB / 0%`，服务器仓库 clean。
- 正式链 `$DATA_ROOT/tmp/full-chain-344a326-s1.log` 输出 `CHAIN_STATUS=G0_G5_PASS`；G3/G4/G5 refs 分别为：
  - `evidence/stage0/g3-formal/c2177055fa60ab547669b2493f61f7a79aec5c60e4368ff4071c80b77181b5c1/index.json`
  - `evidence/stage0/g4-formal/c2177055fa60ab547669b2493f61f7a79aec5c60e4368ff4071c80b77181b5c1/index.json`
  - `evidence/stage0/g5-formal/5c9ba81d383477fee719ef91b44baee602463cc5e2a071e60d731d1f0b15bf44/index.json`
- G6 formal PASS：`evidence/stage0/g6-formal/b03a3eadf67ac2a047b66628c84bf672d14768d8f919846e274e3cf16225db47/index.json`；G7 formal PASS：`evidence/stage0/g7-formal/145ec391af6ffc26a413f03564c115bba3627d6b9d8db9cbbfb520a3acd4b1eb/index.json`。G6 的 collective、semantic、recovery、failure-rank 全部完成，日志没有 NCCL timeout。

### G7R 失败证据与归档

- G7R formal 使用唯一 suite 根 `evidence/stage0/g7-recovery-suite/a330f2501c6fc4c847737cebf6ba5f4c83548764be4751b13aad49e22866be74/`，在四卡 DDP baseline 的第一条控制面 NCCL `ALLGATHER` 失败：`SeqNum=1`、`NumelIn=1`、`NumelOut=4`、`Timeout(ms)=300000`，时间约为 `2026-08-12T19:23:38Z`。单卡 baseline/resume 已完成，但 G7R 未生成 formal index，不能将 G7R 判为 PASS。
- 完整失败链日志保留于 `$DATA_ROOT/tmp/full-chain-344a326-s2.log`，SHA-256 为 `2263b5edf19b132684823d2344a7a169717cb65f9868b4dc0efdcbe346473406`，大小 `22809` bytes；失败 suite 已整体移动至 `$DATA_ROOT/tmp/g7-recovery-failed-344a326-20260812T192354Z/suite/`，687 个文件聚合 SHA-256 为 `7b6eb843ce57f6dda140730bc1c07816bab9134eb72153ba1dde87a0936afb9f`。
- 根因：G6 formal launcher 显式设置了 `NCCL_P2P_DISABLE=1`，但 G7R `_launch_worker` 只复制当前 shell 环境，未将该 transport 条件写入 G7R launcher/worker 合同；G7R worker 仍以默认 NCCL P2P 路径初始化 DDP。该结论由 G6 已通过、G7R 首条四卡 DDP `ALLGATHER` 在相同机器栈 timeout、以及退出后无残留 GPU 进程共同支持。

### G7R 修复与本机回归

- 已实施最小修复：G7R worker plan/report schema 与生成器新增并固定 `nccl_p2p_disable=1`；recovery formal launcher 对所有 fresh process 注入 `NCCL_P2P_DISABLE=1`；worker 在 rank 初始化前拒绝环境变量漂移并在 report 回显该值；S0.9 计划同步记录该 transport 合同。
- 本机回归：`python -m pytest tests/test_stage0_g7_recovery.py tests/test_stage0_g9.py tests/test_stage0_g10.py -q --basetemp .tmp-pytest-g7r-fix -p no:cacheprovider` → **26 passed in 35.55s**；`compileall`、G7R 两个 JSON schema 解析和 `git diff --check` 均通过。

### 当前判定与下一步

- S0.12 仍未完成，当前状态为 **`IN_PROGRESS/BLOCKED_ON_G7R_NCCL_RETRY`**。本轮只能确认新提交 `344a326` 的 G0–G7 PASS；没有本提交的 G7R/G8/G9、三端同步观察、G10 formal 或 `READY` 产物。
- G7R 失败 suite 不覆盖、不复用；修复形成新的 generator commit 后必须从 bootstrap 重新生成 G0–G5，再在四卡独占且实际继承 `NCCL_P2P_DISABLE=1` 的窗口完整重跑 G6→G9。

## 2026-08-13 03:50 CST — 37f7934 G3 sidecar identity 失败与可恢复归档

### 本轮启动与失败边界

- G7R transport 修复提交为 `37f793401648979da687fdaf51b64b6c55103a08`，本地、GitHub、服务器分支均已同步，服务器 worktree clean；服务器四卡在链启动前均为 `0 MiB / 0%`，无外部计算进程占用。
- 正式链 `$DATA_ROOT/tmp/full-chain-37f7934-s1.log` 已从 bootstrap 开始执行。bootstrap PASS：`evidence/stage0/bootstrap/37f793401648979da687fdaf51b64b6c55103a08/index.json`，`environment_hash=96c67ce87856d44b9e2d4134e4d2e854f6f023008108c3845e5467b58ecb88ca`。
- G3 attest 在 `2026-08-12T19:43:13Z` 进入派生 GLUE 校验时失败，异常为 `GLUE_SIDECAR_IDENTITY_MISMATCH:raw_asset_id`；未生成本轮 G3 verification、materialize、formal G3/G4/G5 或任何 G6/G7/G7R 产物。失败日志 SHA-256 为 `ec5485c8bab25eae1cf132fb8d895eb08605ced55ff94b7b4dc7fe761e62a6af`，大小 `2611` bytes。

### 根因确认与归档

- `datasets/glue-{sst2,mnli,rte}-pretokenized/stage0-glue-derived-build.json` 均绑定上一轮 `generator_git_commit=344a326...` 及上一轮 raw asset identity；当前 lifecycle 从 raw publication 重新计算的 identity 与旧 sidecar 不一致。builder 的只读 sidecar 校验拒绝复用该旧派生资产，符合不可变派生资产合同；没有手工修改 sidecar，也没有移动 raw `datasets/glue-{sst2,mnli,rte}`。
- 在确认无 attest、formalizer、torchrun、worker 残留后，将三个明确的旧派生目录及本轮唯一失败 staging 目录整体、可逆地移动至 `$DATA_ROOT/tmp/g3-rebuild-37f7934-sidecar-failure-20260812T195028Z/`。归档共 `34` 个文件、`3963538359` bytes；迁移前清单 SHA-256 为 `82d2648954719f57a5f57dc3bac40e87c6749b9c251f6eadbc780b91dce7e886`，迁移后清单 SHA-256 为 `28e1998300b0f9f10f2a2f1198f61b710c8ccbd9bac32f337fe6372230915cd8`。原始 raw 数据保持原路径不变。

### 当前判定与下一步

- S0.12 仍未完成，当前状态为 **`IN_PROGRESS/BLOCKED_ON_G3_SIDECAR_REBUILD`**。`37f7934` 只证明 bootstrap PASS 和本机 G7R 回归，不能形成新的 G0–G5 或 G6–G9 交付闭环。
- 本段 Worklog 提交后 generator commit 会再次变化，因此不能复用 `37f7934` 的 bootstrap；下一步先完成 GitHub/bundle/服务器同步与 Agent/ 五文件 hash 核对，再以新 commit 从 bootstrap 重新跑 G0→G5，确认派生资产重建和 verification/materialize/formal G3/G4/G5 完成后，才进入四卡独占窗口重跑 G6→G9。

## 2026-08-13 04:32 CST — e378299 G3 重建通过但 materialize 发布血缘冲突

### 本轮 G0–G3 边界

- 本轮 generator commit 为 `e378299417aa8bbc02c427a17f8cfb7fcd72fb3f`；本地、GitHub、服务器分支均已同步，服务器 worktree clean，启动前四卡空闲。正式链 `$DATA_ROOT/tmp/full-chain-e378299-s1.log` 从 bootstrap 开始执行，未启动第二份链。
- bootstrap PASS：`evidence/stage0/bootstrap/e378299417aa8bbc02c427a17f8cfb7fcd72fb3f/index.json`，`environment_hash=e6e46a73a4c41464dc66b59335173d7d468eeeb7d5b6d433c4fcef9187ace86b`。
- G3 attest 成功重建并发布 `13` 个候选；acquisition ref 为 `manifests/evidence/g3/acquisition/35353284e0b4eec46d02a26ccb4a27022fc237b22a5ef82fb1c9ea3fa6eb8cc3.json`，`acquisition_sha256=35353284e0b4eec46d02a26ccb4a27022fc237b22a5ef82fb1c9ea3fa6eb8cc3`。三个 GLUE 派生目录均已按当前 raw identity 与当前提交重建并发布。
- G3 verify 成功验证 `13` 个候选；verification ref 为 `manifests/evidence/g3/verification/35353284e0b4eec46d02a26ccb4a27022fc237b22a5ef82fb1c9ea3fa6eb8cc3.json`，`verification_sha256=ab774ef2a1d1c648e5a4286067e84c89c48953ac0aac2dd1702c269932c17d13`。
- materialize 随后失败，异常为 `G3AssetPublicationError: existing READY does not descend from the supplied VERIFIED input`。因此本轮没有生成新的 materialize resolution，也没有进入 formal G3/G4/G5 或 G6–G9；不能将本轮 G3–G9 宣称为通过。

### 冲突发布物的可逆归档

- 只读核对确认当前 13 个旧 manifest 与 13 个旧 qualification 均绑定旧提交 `344a326cd568223e4501692880a74dd5662f3bfb` 及旧 acquisition/verification 血缘，不是本轮 `35353284…` VERIFIED 输入。未手工编辑 manifest、未移动 raw 数据、未覆盖 immutable evidence。
- 在确认无 attest、verify、materialize、formalizer、torchrun 或 worker 残留后，将这 26 个精确 canonical 发布/资格文件整体移动到 `$DATA_ROOT/tmp/g3-publications-e378299-materialize-failure-20260812T202823Z/`。归档元数据记录 `archived_file_count=26`；迁移前后清单 SHA-256 均为 `4e6d59b13de78aa68b910abbf897182caa46f4dc1e3e5aac25554c637cf1b923`；本轮完整链日志 SHA-256 为 `4376cc9fb4474f849103605453ea915f93c4c1ac577428c8dc14f6b1ac804911`，大小 `223620` bytes。
- 归档内 26/26 文件以 `sha256sum -c` 逐项通过；相应 canonical refs 已全部确认缺失，下一轮 materialize 可在不覆盖旧 READY 的前提下重新发布。归档保留于 `$DATA_ROOT/tmp`，未删除。

### 当前判定与下一步

- S0.12 仍未完成，当前状态为 **`IN_PROGRESS/BLOCKED_ON_G3_PUBLICATION_REBUILD`**。本轮只确认新提交的 bootstrap、G3 attest、G3 verify 通过；G3 materialize 失败，G4–G10 与新 `READY` 均不存在。
- 本节记录会形成新的 generator commit，因此不能复用 `e378299` 的 bootstrap 或 G3 refs。下一步先将本节提交并完成 GitHub、Git bundle、服务器三端快进同步及 Agent/ 五文件 hash 核对；然后以新提交从 bootstrap 重新跑 G0→G5。若新提交导致派生 sidecar identity 需要更新，按同一只读校验与可逆归档流程处理后再重试 materialize；只有新的 G3/G4/G5 完整通过，才在四卡独占窗口重跑 G6→G9。

### 记录提交与三端同步

- 本节追加记录形成提交 `f1262c422671e914c4dea900888da2de8f6392ca`（`docs(stage0): record g3 materialize lineage conflict`），本地、GitHub `origin/feat/stage1-cpu-evidence` 与服务器 `/home/sophgo13/cjl/parameter-importance` 均已核对为该 HEAD；服务器分支为 `feat/stage1-cpu-evidence` 且 worktree clean。
- 本次 Git bundle 使用服务器临时路径 `$DATA_ROOT/tmp/repo-sync-f1262c4.bundle`，本机/服务器 SHA-256 均为 `d76ae095a479bc3fd1207383ae8d4b3511f9fac4bd780b7b20d86554f509e726`，大小 `18062520` bytes；服务器以 `git merge --ff-only` 快进成功。三端确认后，本机与服务器 bundle 均已精确删除。
- 本机与服务器 `Agent/` 五文件 SHA-256 完全一致：`git.md=183f4ba702d22a3a97a459d4873aed62377d51b666d2515990a2f408ecd856ca`；`remote_access.md=795c677e717827492a30342e5b91a4b5959f0df22c72354f14a506ecb023f7a1`；`server.md=9f2d4370ac64990cd29d33ef13de5c20cca65efb4655e928118ae4f3ca012c68`；`sync.md=1bf84f8379018b918eb1680c49dacb7d6d75d764c306782f5170212ceb190015`；`worklogs.md=4a61b34b02a7070b5d3321b349d13016548615a0f3ce069901d18049072a10da`。
