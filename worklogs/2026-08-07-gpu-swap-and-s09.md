# 2026-08-07 GPU 换卡与 S0.9 继续执行

- 任务范围：按项目所有者指示将允许四卡中的 `0000:a4:00.0`（GPU-5a81500d）替换为健康备卡 `0000:53:00.0`（GPU-180ff767），重新完成 G0-G 复验，并继续执行 S0.9 正式链直到完成。
- 当前状态：进行中（G0-G 已复验 PASS；S0.9 链即将在新锚点分支重跑）
- 工作分支：`feat/stage1-cpu-evidence`（换卡提交将推送到此分支，链锚点用新提交）

## 2026-08-07 08:30 CST — 换卡、G0-G 复验与代码/证据更新

### 目标与范围

- 完成：诊断 a4:00 无法清空的 aggregate ECC；执行换卡（a4→53）；更新 modprobe 排除配置并重启；重跑管理员 finalizer 与 CUDA/NCCL smoke；更新仓库脚本/报告/测试；提交推送。
- 不在本阶段处理：S0.9 正式链本体（下一步执行）；Stage 1 正式 GPU gates。

### 实际修改

- 代码、配置、文档：
  - `ops/stage0/admin_restore_gpu_services_after_exclusion.sh`、`verify_gpu_uuid_exclusion_post_reboot.sh`、`admin_apply_gpu_path_b.sh`、`admin_install_gpu_uuid_exclusion_and_reboot.sh`：允许/排除 UUID 与 BDF 清单改为 53/9c/9d/a0 允许、4f/50/57/a4 排除；`SELECTED_DEVICE_NODES`、resume 检查、响应 JSON 同步更新；内核清单加入 `6.8.0-137-generic`。
  - `ops/stage0/promote_environment_after_gpu_gate.py`、`run_cuda_nccl_smoke.py`：`EXPECTED_BDFS/UUIDS/EXCLUDED_UUIDS` 更新。
  - `src/param_importance_nlp/stage0_bootstrap.py`：`_SOURCE_REPORT_REFS` 指向新 G0-G 报告。
  - 新增 `reports/stage0/g0-g-gpu-final-20260807.json` 与 `reports/stage0/g0-g-swap-a4-to-53-20260807.json`。
  - `tests/test_stage0_bootstrap.py`、`tests/test_stage0_g3_formalization.py`：UUID 顺序、boot_id、kernel 同步。
  - 新增本工作日志。
- 服务器或外部状态：
  - `/etc/modprobe.d/parameter-importance-stage0-gpu-exclusion.conf` 更新为新排除 CSV（root:root 644），`update-initramfs -u` 完成（6.8.0-137）。
  - 服务器重启至新 boot `69a1d6bb-…`（kernel 6.8.0-137），新四卡生效：53/9c/9d/a0 可见，a4 消失，全部 ECC 0/0，无计算进程。
  - 管理员 finalizer `finalize-20260807T001522Z.UmXAlXcj` 通过；CUDA/NCCL smoke `cuda-nccl-smoke-20260807T001804Z` 通过。
  - 更新后的 finalizer 已同步到 `$DATA_ROOT/tmp/stage0-gpu-service-finalize-677c7bfc6beaedca.sh`（LF、700、语法 OK）。
- 用户原有修改：无；本机工作树仅含上述改动。

### 实验与验证

| 项目 | 命令/来源 | 结果 | 证据路径 |
|---|---|---|---|
| aggregate ECC 诊断 | `nvidia-smi -q -i 3 -d ECC`、`-p 1`、`-r` | aggregate=2 无法清零（-p not supported；reboot 与 GPU reset 均保留计数） | 本日志 |
| 新四卡枚举 | `nvidia-smi -L` | 53/9c/9d/a0，四卡 ECC 全部 0/0 | 本日志 |
| 管理员 finalizer | `sudo bash tmp/stage0-gpu-service-finalize-677c7bfc6beaedca.sh` | `G0_G_UUID_EXCLUSION_SERVICE_FINALIZE_PASS`，boot `69a1d6bb` | `/var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize/finalize-20260807T001522Z.UmXAlXcj` |
| CUDA/NCCL smoke | `run_cuda_nccl_smoke.py --data-root $DATA_ROOT` | PASS，torch 2.12.1+cu126 / NCCL 2.29.3，四卡 all-reduce 正确 | `$DATA_ROOT/operations/stage0/g0-g-uuid-exclusion/cuda-nccl-smoke-20260807T001804Z` |
| 单元测试 | `pytest tests/test_stage0_bootstrap.py tests/test_stage0_g3_formalization.py` | 8 passed | 本日志 |

### 判定

- G0-G：新 boot `69a1d6bb` 上以新四卡（53/9c/9d/a0）复验 PASS；新报告 `g0-g-gpu-final-20260807.json` 生成，bootstrap 引用已切换。
- 换卡决策：a4 的 aggregate uncorrectable ECC=2 为寿命计数，无法清零且 G0-G 硬 gate 要求为 0；按所有者指示改用健康备卡 53。

### 问题、原因与风险

- 根因：a4 曾发生 Xid 63（row remap），留下 2 次 aggregate uncorrectable；NVML aggregate 按设计只增不清，`nvidia-smi -p` 在该设备不支持。
- 风险：53 此前为“healthy spare”，本 boot ECC 0/0、row-remap Pending=No；仍按 G0-G 观察窗口约束执行。
- 遗留：`$DATA_ROOT/tmp/repo-sync-975f83c.bundle`、`g3-pre-975f83c-backup-…` 待 S0.9 链完成后清理。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence`（提交换卡改动后更新）。
- 服务器：将用 bundle 快进同一提交并创建新锚点 `formal/run-<newshort>`。
- `Agent/*.md` 哈希：不受影响。

### 下一步

- 提交推送换卡改动；bundle 同步服务器；创建新锚点分支；生成新链脚本；备份当前 G3 控制面/GLUE derived 资产；在新锚点执行 bootstrap→G3→G4→G5→G6→G7-LOGGING→G7-RECOVERY 完整链并核验 G7。

## 2026-08-07 13:30 CST — S0.9 完成（G7 完整 gate PASS）

### 目标与范围

- 完成：在新四卡（53/9c/9d/a0）上跑通 S0.9 完整正式链，核验 G7 可恢复性 gate，并完成日志/同步/清理/下载恢复。

### 实际修改与执行

- 修复 S0.9 正式恢复配置与计划不一致的问题：`_config_overrides` 从 BF16+AMP+num_workers=2/prefetch 改为确定性 FP32+num_workers=0（`stage0_g7_recovery.py`），并在 worker 顶部设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、关闭 TF32/benchmark、启用 cudnn deterministic。
- 依据计划“若只能数值一致必须记录具体非确定性来源”，在 `_compare_trajectory_pair` 中实现数值容差等价：模型张量按 `atol=1e-6/rtol=1e-5` 比较，步骤记录对 loss/grad_norm/clip 做容差比较，忽略非确定性的参数哈希；结果写入 pair metrics 与 gate measured。同步放行 `stage0_g9_replay.py` 并更新测试。
- 修复 `_load_rank_checkpoint_state` 引用不存在辅助函数的问题。
- 正式链在锚点 `formal/run-c73d8a4`（commit `c73d8a4e143a43b8ef9bff67ac7d42b04b539134`）完整执行：bootstrap→G3→G4→G5→G6→G7-LOGGING→G7-RECOVERY，全部 PASS。

### 实验与验证

| 项目 | 结果 | 证据路径 |
|---|---|---|
| 完整正式链 | `CHAIN_STATUS=PASS` | `$DATA_ROOT/tmp/g7-recovery-chain-c73d8a4.log` |
| G7-RECOVERY 索引 | status=PASS，`next_task_id=stage0.10_capacity_and_operations`，generator commit c73d8a4 | `evidence/stage0/g7-recovery-formal/4ecf7008…/index.json` |
| G7 gate 记录 | PASS；measured：single/four_gpu_exact=true、shared_state_hashes_exact=false、state_numeric_tolerance_met=true、formal_num_workers=0 | `evidence/stage0/tasks/09-4ecf7008…/commits/resume_equivalence_report.json` |
| 数值容差 | single/ddp：max_abs_diff=7.45e-09、mismatched_elements=0；max_rel_diff 0.0165/0.00959；非确定性来源已记录 | 同上 gate_report.checks[1].measurements |
| 环境 | passed_gate_ids 含 stage0.G7 与 stage0.G7-LOGGING | `evidence/stage0/g7-recovery-formal/4ecf7008…/environment.json` |

### 判定

- **S0.9 完成**：G7 完整可恢复性 gate = PASS；新四卡（53/9c/9d/a0）证据链 bootstrap→G7-RECOVERY 全部正式 PASS。
- Stage 0 当前全部硬 gate（G0-C、G0-G、G1、G2、G3 系列、G4、G5、G6、G7、G7-LOGGING）通过；下一任务为 S0.10。

### 问题、原因与风险

- GPU FP32 训练在此 A100/驱动/torch 组合上存在 ~1e-9 绝对量级的低阶内核非确定性，字节哈希不等；已按计划改为容差等价并记录来源，CPU FP32 参考仍为字节精确。
- 换卡后的 `0000:53:00.0` 全程无 ECC/row-remap 异常；服务器重启后环境稳定。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence`（本轮提交 `c73d8a4` 及收尾日志提交后更新）。
- 服务器：正式证据锚点 `formal/run-c73d8a4` 保留；主分支同步至收尾提交。
- `Agent/*.md`：不受影响。

### 收尾动作

- 清理 `tmp/repo-sync-975f83c.bundle` 与 `tmp/g3-pre-975f83c-backup-20260806T044840Z`（按前序日志约定）。
- 恢复 Pile 下载：启动 lab-pc `CjlPileFullSupervisor`，核对 shard 10 `.part` 续传。
- 下一步：S0.10 容量/运维；随后 G10 handoff → S1.7 配置补齐 → Stage 1 服务器侧正式 GPU gates。
