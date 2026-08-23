# Stage 2 S2.6 试运行与矩阵冻结实现记录（2026-08-23）

## 范围与权限边界

- 工作树：`.agent-temp/worktrees/stage2-s206`；分支：`feat/stage2-s206`；基线：`eba3d0eda4f572bd491d9ba210ef64c771f99803`。
- 本记录只覆盖 S2.6/G2.4b 的合同、fixture runner 接线和开发验收；未修改预注册、依赖版本、主工作树 presentation 文件或其他问题。
- 当前 G2.3 仍为 `PENDING_EXTERNAL_AUTHORIZATION`，Stage0 单副本授权已于 2026-08-18 过期。因此没有生成正式 matrix、没有读取正式 confirmatory 梯度、没有声称 G2.4b PASS。

## Producer 交付

- 新增 `stage2-pilot-matrix-freeze-v1` 与 `stage2-confirmatory-mapping-v1` 严格 schema。
- 新增 dependency-light `stage2_pilot.py`：人工分布 raw/U/double 与不等权 U 校准、六个 model×stage anchor 合同、B/M 盲化扫描、R sizing（含 reference 半宽预算）、三种成本语义和 `cost_io_quiescent`、fixture matrix freeze、confirmatory repetition mapping、hash/唯一性/stream/样本碰撞审计。
- formal constructor 统一 fail-closed：没有当前 `FormalExecutionEvidence`/外部授权时拒绝 formal matrix/mapping；fixture 产物固定 `formal_eligible=false`、`qualification_gate_hash=null`。
- `_run_stage2_pilot` 生成 deterministic local artificial calibration 与 BLOCKED fixture matrix contract，并明确 `G2.3_PENDING_EXTERNAL_AUTHORIZATION`；不生成 confirmatory mapping。
- `validate_stage23_artifact`、Stage 2 public exports 与最小 S2.6 tests 已接线。mapping schema 明确允许并要求双半比较的 `M=2`。

## Local fixture / producer 证据

- fixture 合同使用六 anchor 的合成结果测试可冻结 `(B=32,M=16,R=200)`，并生成 200 个 confirmatory-stream repetition mappings；这只证明本地合同/重建逻辑，不是正式资格证据。
- runner 使用固定 seed `206` 的人工样本；当前真实模型 pilot、GPU/存储成本和成本静默窗口均未运行，成本字段保持未定义且 `cost_io_quiescent=false`。
- 正式执行的可恢复入口仍由既有 S2.4/S2.5 draw/repetition contracts 提供；本任务没有擅自长跑或下载模型。

## 验收与失败恢复

- 逐层校验：`tests/test_stage2_s206_pilot_contracts.py` 5 passed；独立 JSON Schema 复核通过（mapping 首条 `m_values=[2,16]`）。
- 最终组合（显式工作树 `basetemp/cache`）：
  `python -m pytest --basetemp .agent-temp/pytest/s206-final-02/basetemp -o cache_dir=.agent-temp/pytest/s206-final-02/cache tests/test_stage2_s206_pilot_contracts.py tests/test_experiments_sampling_stage2.py tests/test_stage23_formal_orchestration.py tests/test_stage23_task_runners.py -q -rA`
  结果：`58 passed in 46.50s`。
- 首次组合尝试的首个 traceback 是 pytest 默认目录 `C:\Users\HP\AppData\Local\Temp\pytest-of-HP` 的 WinError 5 PermissionError，导致 27 个 setup errors/31 passed；保留该故障证据后切换到工作树内显式 basetemp/cache，最终全通过。不是代码失败，未重复原命令。
- 修复记录：补齐 mapping schema 的 M=2 enum；修正 R sizing 使 `reference_half_width` 直接消耗 precision margin，并加入对应边界断言。

## Pending formal blocker

只有获得当前外部授权并完成 G2.3/G2.4a 所需真实六-anchor producer 证据后，才能由后续流程生成正式 matrix、冻结正式 confirmatory mapping 并重新判定 G2.4b；本记录不把 local fixture 通过升级为 formal PASS。
