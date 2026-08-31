# 2026-07-22 Stage 0–9 Run-Ready 功能补全

- 任务范围：补齐 `param-importance-nlp` 0.4.0 的机器可读任务目录、训练与在线重要性、Stage 2–9 执行器、统一任务生命周期、严格离线 provider、恢复/重放、报告流水线和运行文档。
- 当前状态：主要运行核心与本机验收已完成；等待本机提交与 GitHub 推送。
- 工作分支：`feat/local-contracts-core-stage0-9`

## 2026-07-22 16:52 CST — 功能与文档收口

### 目标与范围

- 本阶段要完成：使每个计划内 Stage/task 都有正式 CLI、严格配置、runner、恢复边界、预期产物与报告路径；后续只需填写版本化配置/manifest、执行、审核和重跑。
- 不在本阶段处理：连接服务器、运行 SSH、执行管理员脚本、下载模型/数据、安装 CUDA/NCCL 或可选 Hugging Face 依赖、生成正式科学结论。

### 实际修改

- 合同与入口：新增 `StageTaskCatalog`、`ResolvedConfig v2`、task preflight/run/resume/status/replay/finalize、`task fixture-all`、资产登记/验证、artifact 审核/批准、endpoint/probe/route/ablation builder 和 Gate 构建/汇总入口。
- 训练：新增真实 `TrainingEngine`、本机/DDP executor、模型/数据/evaluator 协议、tiny fixture、严格离线 provider、在线 raw/double/U 重要性、完整 checkpoint、可恢复有序 prefetch、显式 profiling window 和重要性轨迹。
- Stage 2–3：新增 formal/fixture 共用执行核心、可恢复 reference/paired wave、端点捕获、probe panel、节点缓存、reference refinement 与 recommendation。
- Stage 4–6：新增可恢复 route runner、共享初始化/lineage 约束、路线矩阵、训练轨迹、配对分析 runner，以及按 phase 两阶段提交并汇总的 `resource_profiles`。
- Stage 7–9：新增剪枝 study、真实训练驱动的消融 cell、矩阵执行、权威 shard/cell commit 恢复校验、跨阶段 ETL、统计、表图、报告、bundle 和 replay runner。
- 文档：重写 `Readme.md`，明确训练与在线重要性代码位置、状态边界、空目录运行、代表性 `fixture-all`、声明式 builder、prefetch/profiling 与显式 resume；为 Stage 4–9 增加逐 task 稳定锚点、恢复边界和当前实现限制。
- 用户原有和并行工作树改动：保留并纳入统一审查，未执行 reset、checkout 覆盖或删除。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 完整 pytest 集合 | 四个互斥文件分区，均使用 `.venv`、`PYTHONPATH=src` 与独立 `MPLCONFIGDIR` | `530 passed, 9 skipped, 0 failed`；共收集 539 项 | `reports/local-validation-stage0-9-v04.json` |
| compileall | `.venv\\Scripts\\python.exe -m compileall -q src tests ops` | `PASS`，退出码 0 | 本工作日志；机器报告同上 |
| schema 示例重放 | `tests/test_cli_v03.py::test_contract_validate_replays_every_repository_schema` 及 run-ready 配置/schema 测试 | `PASS` | pytest 分区结果 |
| tiny Stage 0–9 双次流水线 | 两个全新输出根加同根恢复重放 | `1 passed`；结果逐字段、逐字节及科学 hash 一致 | `tests/test_full_fixture_pipeline.py` |
| DDP Gloo world size 1/2/4 | `tests/test_runtime_distributed.py` | `6 skipped`：当前 Windows Torch wheel 声明 Gloo，但无受支持 device；未伪造 PASS | pytest 分区 B |
| `git diff --check` | Git 工作树检查 | `PASS`，退出码 0 | 本工作日志 |
| 秘密/大文件 guard | `param-importance git-guard --repo .` 加常见令牌/私钥模式扫描 | `PASS`，无命中 | 本工作日志 |
| formal fail-closed | 缺服务器、GPU、真实资产或 Gate 的 formal preflight | `PASS`：返回结构化 `BLOCKED`，不进入 runner/synthetic fallback | formal preflight 测试 |

完整测试分区明细：合同/核心/runtime `271 passed, 3 skipped`；Stage 0–6、训练与 Stage 2/3 `124 passed, 6 skipped`；Stage 7–9/分析 `112 passed`；CLI 与完整流水线 `23 passed`。集成过程中测试曾发现 Stage 6 fixture 旁路、Stage 7 训练源变量覆盖、Stage 8 legacy manifest 分流和两处陈旧断言；均已按冻结合同修复并对失败节点重跑通过，未从最终数字中隐藏未修复失败。

实验元数据：

- Git 提交：核心收口提交 `e09a3d289ba81d86e422e0b5b0af8cfcfb14d8c4`（`feat: make Stage 0-9 pipeline run-ready`）；本日志由后续证据提交绑定。
- 随机种子：以各 resolved v2 config 和 `SeedPlan` artifact 为准。
- 配置文件：`configs/local-fixtures/`、`configs/run-ready/`，最终清单待回填。
- 数据集及 revision：本机未使用真实数据；tiny/synthetic fixture 无 formal 资格。
- 模型及 revision：本机未使用真实模型资产；tiny fixture 无 formal 资格。
- checkpoint：本机回归产物由 Git 忽略；正式 checkpoint 未运行。
- 设备/关键环境：Windows 本机 CPU；Python 3.12.4、Torch 2.10.0+cpu、NumPy 2.5.1、SciPy 1.18.0、Pandas 3.0.3、Matplotlib 3.11.0、Seaborn 0.13.2、PyYAML 6.0.3、Packaging 26.2、pytest 9.1.1。
- 结果目录：本机 fixture 目录由 Git 忽略；正式结果目录未创建。

### 产物与证据

| 路径 | 类型 | 大小 | SHA-256/revision | 验收状态 |
|---|---|---:|---|---|
| `Readme.md` | 项目说明与运行入口 | 随提交绑定 | Git blob/提交 SHA | `PASS` |
| `plan/run_ready_stage4_9.md` | Stage 4–9 索引/运行手册 | 随提交绑定 | Git blob/提交 SHA | `PASS` |
| `plan/stage4/README.md` … `plan/stage9/README.md` | 逐 task 计划与稳定锚点 | 随提交绑定 | Git blob/提交 SHA | `PASS` |
| `schemas/` | v2 配置与运行产物 schema | 随提交绑定 | Git blob/提交 SHA | `PASS`：schema 重放通过 |
| `src/param_importance_nlp/` | Run-Ready 功能代码 | 随提交绑定 | Git blob/提交 SHA | `PASS`：本机适用测试通过 |
| `reports/local-validation-stage0-9-v04.json` | canonical 本机验证摘要 | 小型文本产物 | Git blob/提交 SHA | `PASS` |

### 问题、原因与风险

- 服务器不可连接；因此服务器 HEAD、服务器 `Agent/*.md` SHA-256、CUDA/NCCL、真实资产和 formal Gate 必须记录为 `BLOCKED: server_unreachable`，不得宣称多端同步或正式 Stage 完成。
- 正式 B/M/R、真实 reference、quadrature 默认规则、probe 数、节点预算和科学阈值必须由真实 pilot/审核决定，当前保持 `UNFROZEN/BLOCKED`。
- 本机 fixture 只证明执行链和数学合同可回归，不能支持 Pythia/Pile/GLUE 或剪枝/消融的科学结论。
- `task fixture-all` 当前运行每个 Stage 的代表性缩小路径，不穷举 catalog 每个 task，也不覆盖 DDP 2/4、offline HF、prefetch/profiling 或 formal 环境；不得把“covered_stages=0..9”解释为所有任务均已逐项验收。
- profiling 资源窗口 producer 已同时接入通用 `TrainingTaskRunner` 和 Stage 4–6 route runner；route 按 phase 发布不可变窗口 object/commit，并在 route 产物中汇总 `resource_profiles`。本机产物仍不能作为正式性能结论，formal 资格继续依赖真实环境、审核与 Gate。
- Stage 7/8 shard runner 已使用 `resume_ref` 校验部分恢复边界：引用必须位于当前 `artifacts.output_dir`、真实存在并覆盖全部已发现的权威 cell commit；无引用的普通 run 不会静默续跑。该语义用于授权同一冻结矩阵/output root 的幂等恢复，不把一个引用解释为可任意忽略其他已提交 cell 的子集选择器。
- `artifact ablation-matrix-build` 的 formal scope 只声明矩阵用途；它不读取 formal execution evidence，也不构造 Gate。Stage 8 formal 资格仍必须由 task preflight、输入证据和独立 Gate 决定。
- `feature-code-complete/run-ready` 允许真实首跑后的普通缺陷修复；若发现某项计划功能或 runner 根本未实现，则本次验收失败。

### Git 与多端同步

- 本机分支/HEAD：`feat/local-contracts-core-stage0-9`；核心提交 `e09a3d289ba81d86e422e0b5b0af8cfcfb14d8c4`。
- GitHub 分支/HEAD：待最终提交和推送后回填。
- 服务器分支/HEAD：`BLOCKED: server_unreachable`。
- `Agent/*.md` 哈希核对：本机文件未修改；服务器核对 `BLOCKED: server_unreachable`。
- 工作树状态和临时产物：代码、测试、schema、配置、README 和验证报告已稳定；运行目录继续由 Git 忽略。

### 下一步

- 创建验证日志提交并推送当前跨阶段分支。
- GitHub 推送成功后记录远端 HEAD；服务器同步继续如实保持 `BLOCKED: server_unreachable`。
- 首次真实服务器运行只允许填写版本化配置/manifest、执行、审核和常规缺陷修复；若仍需新增计划内实验逻辑，则不得宣称 `feature-code-complete`。
