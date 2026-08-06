# Stage 1 CPU 状态报告（2026-08-06）

## 结论

Stage 1 中不依赖 GPU/服务器的部分已在本机收口：全量测试 967 passed / 10 skipped
（跳过均为 Windows 环境限制）；stage1.01–1.07、1.09–1.11 的本地任务 runner 全部
PASS 并产出证据；与 e6d6863（Stage 0 关闭重要性跟踪）相关的 4 处过期测试/配置已
修复。正式 gate（G1-ENTRY、G1-SINGLE/DDP/NUMERIC/RESUME/EXIT）仍被 S0.9/G0-G
的 GPU 问题阻塞，本地证据一律 `gate_status=NOT_RUN`，不冒充正式通过。

## 测试基线

- 收集 977，通过 967，跳过 10，失败 0。
- 跳过原因：Windows symlink 权限（4 项）、POSIX 权限语义（1 项）、
  Gloo 无设备（3 项）。
- 详情：`reports/stage1/stage1-cpu-test-baseline-20260806.json`。

## 修复清单

| 文件 | 问题 | 修复 |
|---|---|---|
| `configs/stage0/g9-test-matrix-v1.json` | 末尾 CRLF，非 canonical JSON | 用 canonical 发布器重写 |
| `tests/test_task_artifacts_and_default_runners.py` | 用 stage0.06 断言 importance bundle | 改用 stage1.07 |
| `tests/test_full_fixture_pipeline.py` | 断言 stage0.06 的 importance_snapshot 非空 | 改为 `None` 并更新注释 |
| `src/.../experiments/full_fixture_pipeline.py` | 过时注释 | 更新说明（Stage 4 覆盖在线重要性） |
| `tests/test_stage79_run_ready_completion.py` | 用 stage0.06 作为 Stage 7 剪枝输入 | 改用 stage1.07 |

## Stage 1 任务执行证据（CPU）

`reports/stage1/cpu-evidence-20260806/`（生成器 `ops/stage1/generate_cpu_evidence.py`）：

| 任务 | 状态 | 证据 |
|---|---|---|
| stage1.01_entry_and_contract | PASS（local_fixture） | stage_contract / requirements_matrix / gate_record |
| stage1.02_architecture_and_parameter_registry | PASS | parameter_registry / registry_validation_report / gate_record |
| stage1.03_fixtures_and_oracles | PASS | fixture_manifest / oracle_bundle / oracle_validation_report |
| stage1.04_loss_and_gradient_scale | PASS | gradient_scale_report / comparison_table / gate_record |
| stage1.05_estimators | PASS | estimator_validation_report / estimator_tensor_bundle / gate_record |
| stage1.06_training_integration_and_accumulators | PASS | step_validation_report / importance_trajectory / gate_record |
| stage1.07_single_gpu_pythia14m | PASS（仅执行链，非 G1-SINGLE） | single_gpu_report / importance_trajectory / checkpoint_commit |
| stage1.09_precision_clipping_and_optimizer_boundaries | PASS | numeric_boundary_report / skip_lifecycle_report / gate_record |
| stage1.10_checkpoint_resume_and_artifacts | PASS | training_state_manifest / resume_equivalence_report / gate_record |
| stage1.11_reporting_and_exit_gate | PASS | stage_report / requirements_matrix / gate_summary / delivery_manifest |

所有任务 `gate_status=NOT_RUN`，`scope=local_fixture`，`formal_eligible=false`。

## Gate 状态

| Gate | 本地 CPU | 正式 |
|---|---|---|
| G1-CONTRACT | 证据已生成（math doc/plan hash、冻结公式） | NOT_RUN（需入口正式通过） |
| G1-REGISTRY / ORACLE / GRAD / EST / STEP | runner PASS + 单元测试全绿 | NOT_RUN |
| G1-ENTRY | 部分（Agent 哈希 5/5、DATA_ROOT 事实） | BLOCKED（GPU 健康未复验） |
| G1-SINGLE / DDP / NUMERIC / RESUME / EXIT | 执行链本地可跑（1.07/1.09/1.10） | BLOCKED（S0.9/G0-G） |

## 依赖关系提醒

- S1.7 正式配置仍含 `FILL_*` 占位符与 `estimator_decision_ref: null`，需在
  G10 handoff 消费后补全（见 `configs/run-ready/layers/formal-stage1-pythia14m.yaml`）。
- Stage 2/3 的预注册文档（S2.1/S3.1）与代码级准备不依赖本报告之外的条件，可并行。
