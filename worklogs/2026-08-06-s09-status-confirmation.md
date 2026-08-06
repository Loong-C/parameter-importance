# 2026-08-06 S0.9 完成状态核验

- 任务范围：按 `Agent/` 运维文档核验 S0.9 完成状态与下一步；只读审计，不执行管理员级 GPU 动作、不启动正式链、不恢复 Pile 下载。
- 当前状态：S0.9 未完成（BLOCKED，阻塞于 G0-G 当前 boot 健康复验与外部 GPU 占用）
- 工作分支：`feat/stage1-cpu-evidence`（本日志提交前为 `185a657`）

## 2026-08-06 23:00 CST — S0.9 完成状态核验

### 目标与范围

- 本阶段要完成：核对 S0.9 代码/正式证据/三端 Git/Agent 文档哈希，确认完成度、阻塞点与下一步；记录本核验日志并随阶段提交推送。
- 不在本阶段处理：GPU reset/驱动/重启等管理员动作；`$DATA_ROOT/tmp/g7-recovery-chain-975f83c.sh` 正式链执行；Pile 下载恢复。

### 实际修改

- 代码、配置、文档：新增本工作日志；无功能代码改动。
- 服务器或外部状态：仅只读核验，未改动任何文件。
- 用户原有修改：无；本机工作树干净。
- 更正说明：`worklogs/2026-08-06-stage1-entry.md` 中的“当前分支”为该文书写入时快照（当时在 `feat/stage0-completion`）；该文件随提交 `185a657` 实际位于 `feat/stage1-cpu-evidence`。按规范不改写历史，本日志以最新三端状态为准。

### 实验与验证

| 项目 | 命令/来源 | 结果 | 证据路径 |
|---|---|---|---|
| 本机 Git | `git status -sb`、`git rev-parse HEAD` | `feat/stage1-cpu-evidence` @ `185a657`，工作树干净 | 本日志 |
| GitHub 远端 | `git ls-remote origin` | `feat/stage1-cpu-evidence`=`185a657`；`feat/stage0-completion`=`975f83c`；`main`=`34966d08` | 本日志 |
| 服务器 Git | SSH `git status/branch/rev-parse` | `formal/run-975f83c` @ `975f83c`，工作树干净；`feat/stage0-completion` @ `975f83c` | 本日志 |
| Agent 文档哈希 | 本机 `Get-FileHash SHA256` + 服务器 `sha256sum` | 5/5 文件一致（`183f4ba7`/`795c677e`/`9f2d4370`/`1bf84f83`/`4a61b34b`） | 本日志 |
| S0.9 正式链（975f83c） | `$DATA_ROOT/tmp/g7-recovery-chain-975f83c.log` | 失败：bootstrap `STAGE0_BOOTSTRAP_G0_G_CURRENT_RUNTIME_MISMATCH`（2026-08-06T04:50:05Z） | `$DATA_ROOT/tmp/g7-recovery-chain-975f83c.log` |
| G0-G 历史证据 | `reports/stage0/g0-g-gpu-final-20260804.json` | PASS，boot_id `1dc04123-…`，允许 4 卡 UUID 与当前 `nvidia-smi -L` 完全一致 | `reports/stage0/g0-g-gpu-final-20260804.json` |
| 当前 boot | 服务器 `/proc/sys/kernel/random/boot_id`、`uname -r` | boot_id `7a54a465-…`，kernel `6.8.0-136-generic` | 本日志 |
| 管理员 finalizer | `/var/lib/parameter-importance/…/service-finalize/finalize-20260806T*/FAILURE` | 5 次尝试全部 FAILED（exit 1/20/20/20/22）；最新 `075220Z` exit 22：`Xid (PCI:0000:a4:00): 63`，Row Remapper 新行待激活 | 各 FAILURE 与 `admin-finalize.log` |
| GPU 占用 | `nvidia-smi --query-compute-apps` | 4/4 卡各约 47.9 GiB 被 `ray::WorkerDict` 外部训练占用；无 stage0 进程 | 本日志 |
| 失败残留 | `evidence/stage0/` 与 `tmp/` | `g7-recovery-formal/b2bfb5ab…`、`g7-recovery-suite/1c784b97…` 为 9da2af6 失败残留；`tmp/repo-sync-975f83c.bundle`、`g3-pre-975f83c-backup-…` 待确认后清理 | `$DATA_ROOT/evidence/stage0/`、`$DATA_ROOT/tmp/` |

### 判定

- S0.9：未完成。代码修复已就位（`975f83c`，S0.9 恢复控制面改用 gloo），但正式链未产生任何 PASS 证据；当前阻塞在 G0-G 复验：本 boot（`7a54a465`）内 `0000:a4:00` 出现 Xid 63 row-remap pending，管理员 finalizer fail-closed，且 4 张允许 GPU 正被外部 Ray 训练占用。
- Stage 1 CPU 侧：`185a657` 已完成（967 passed/10 skipped、10/10 任务 PASS、`gate_status=NOT_RUN`），正式 GPU gates 仍依赖 S0.9/G0-G 解除阻塞。

### 问题、原因与风险

- 问题：G0-G 证据 boot（`1dc04123`）与当前 boot（`7a54a465`）不一致；当前 boot 日志含 `0000:a4:00` Xid 63（新行 `0x0000000022b09478` 标记 remapping，需 reset GPU 激活）。
- 已确认根因：重启后该卡 row remap 未激活/未清除，管理员 finalizer 按合同 fail-closed，未产出新 boot 的 G0-G PASS。
- 未解决问题和风险：a4:00 处理前不能复验 G0-G；外部 Ray 训练占满 4 卡期间即使 G0-G 复验也不能跑正式链；GPU 3 corrected volatile ECC 观察项仍存在（uncorrected=0）。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence` @ `185a657`（本日志提交后更新为新 HEAD）。
- 服务器：`formal/run-975f83c` @ `975f83c`（S0.9 锚点分支，保留供链重跑）；`feat/stage0-completion` @ `975f83c`。
- `Agent/*.md` 哈希：本机与服务器 5/5 一致（本日志同步后复核）。
- 工作树：本机与服务器均干净。

### 下一步

1. 管理员处理 `0000:a4:00`：reset GPU 激活 row remap 或干净重启，确保当前 boot kernel 日志无 Xid/GPU/PCI critical 事件且 uptime ≥ 900 s；停掉外部 Ray 训练，使 4 张允许 GPU 无计算客户端。
2. 管理员以 root 运行 `$DATA_ROOT/tmp/stage0-gpu-service-finalize-677c7bfc6beaedca.sh` 直至 `SUCCESS`（产出新 boot_id 的 G0-G 证据），随后 bootstrap 复验 G0-G。
3. 在服务器锚点 `formal/run-975f83c` 执行 `$DATA_ROOT/tmp/g7-recovery-chain-975f83c.sh` 完整链（bootstrap→G3→G4→G5→G6→G7-LOGGING→G7-RECOVERY），确认 `G7R_REF` 与 `full_g7_status=PASS`。
4. 通过后更新工作日志、三端同步并清理 `tmp/repo-sync-975f83c.bundle` 与 `g3-pre-975f83c-backup-…`；恢复 Pile 下载（`CjlPileFullSupervisor`）并核对 shard 10 续传。
5. 随后消费 G10 handoff，补齐 S1.7 `FILL_*`/`estimator_decision_ref` 配置，执行服务器侧 G1-SINGLE/DDP/NUMERIC/RESUME/EXIT（本机 CPU 证据已就绪）。
