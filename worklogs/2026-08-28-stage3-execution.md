# 2026-08-28 Stage 3 路径求积执行日志

## 2026-08-28 11:15–11:40 CST — 接管、基线与 G3-0 审计

- 完整读取 `Agent/git.md`、`local.md`、`remote_access.md`、`server.md`、`sync.md`、`worklogs.md`，随后读取 Stage 3 总计划、十个子计划、数学规格、CLI、schema、runner 和相关测试。
- 保留根工作树中用户已有的三个 presentation dirty 文件；没有修改或暂存它们。
- 从已交付 Stage 2 提交 `000ce1e79af791ce1eae2e2b62da221a10dd3c9a` 创建独立分支 `exp/stage3-path-formal-20260828` 和独立工作树 `.agent-temp/worktrees/stage3-path-formal-20260828`。
- 服务器主 repo 在只读核验时为 clean detached HEAD `44f934dd62d1b86fcb951230c81f3bfa647791aa`。没有进行服务器写入、同步或实验启动。
- 服务器目标 Stage 2 run 为 `pythia-grid-20260826T145530Z`，真实 Pythia 数据位于 `results/stage2/direct-unvalidated/...`。交付状态为 `COMPLETE`，但 S2.7/S2.8/S2.10/S2.11 均明确 `formal_eligible=false`。
- 广域只读检索未发现该 run 的正式 G2.7/G2.8 artifact；发现的同名 Gate 文件均位于 pytest 临时 fixture。
- 31M `mid_late` 原始 attempt 进程无状态退出，continuation 从 repetition 46 开始，导致 B=64 缺少前 18 repetitions。
- Stage 0 G10 历史报告为 PASS，但其中 G1 persistence risk acceptance 已于 `2026-08-18T23:59:00+08:00` 过期。
- PID 724839 是独立的 `s205-formal-r8-g3-v5-wrapper.py`，与目标 direct run 的 PID 集合不同；未干预。
- 结论：G3-0 为 `BLOCKED`。不能把 direct-unvalidated 数据或 pytest fixture 改标成正式证据，不能跨 Gate 启动 Stage 3 formal。
- 机器可读审计：`reports/stage3/g3-0-prerequisite-audit.json`，artifact hash `267bb9f98f7357b44fb238f3ff23dd177e3b6b14a255ab6492e48c16f9e6d4cb`。

## 2026-08-28 11:20–11:42 CST — 不越 Gate 的 G3-1/G3-2 合同修复

- 修复干净 Python 进程中 `core.estimators -> contracts -> stage0_handoff -> runtime.training -> core.estimators` 的循环导入：将 Stage 0 handoff 的 runtime artifact loader 改为调用点惰性导入。
- 修正 Stage 3 contract 的路径贡献符号，冻结为 `-delta_theta * integral gradient`；没有重复乘学习率、裁剪或 optimizer 因子。
- 补齐 Stage 3 metric contract：normalized L1/L2/L∞、cosine、active Spearman、sign consistency、top-q 0.1%/1%/5% overlap/Jaccard、layer/module TV、真实梯度求值成本、model/stage/update/probe strata。
- 新增纯数值 Stage 3 指标模块，零分母、空 active set、零质量等情况显式 undefined，不加 epsilon，不改变科学阈值。
- 正式 probe plan/panel 现强制至少三个互不重叠的 formal probes；local fixture 仍可使用较小 panel，但不能声明正式资格。
- 相关测试先后得到 `69 passed`、`12 passed`，合并 Stage 3 套件得到 `87 passed in 72.44s`；`git diff --check` 通过。

## 当前完成边界

- 已完成：G3-0 真实审计与 fail-closed 结论；G3-1 数学/指标合同修复；正式 probe 下限与本地回归；不越 Gate 的分析指标核心。
- 未启动：任何 Stage 3 formal 服务器实验、pilot/formal 数据、G3-5 至 G3-8、报告和 Beamer。原因是 G3-0 未通过，不是代码 readiness 或 GPU 排队。
- 最小外部解阻条件：重新验证或续期 Stage 0 时限风险接受；在既有冻结预注册下执行 Stage 2 正式 replay/validation，补全覆盖并发布 `formal_eligible=true` 的 G2.7/G2.8/S2.11。该动作与当前“不得重跑/正式 replay Stage 2”的限制冲突，必须先获得明确授权变更。

## 2026-08-28 11:42–11:55 CST — G3-5 协议与独立复核

- 新增不可变、可复算哈希的 Stage 3 protocol：pilot 固定为 14M 单 seed、三阶段、每阶段至少两个 endpoint、每 endpoint 两个 probe；formal 固定为 14M 的 24 个 endpoint 与 31M 的 9 个 endpoint，每 endpoint 三个 probe，共 99 个 endpoint×probe 单元。
- 冻结候选覆盖 left/right/midpoint/trapezoid/Simpson、复合 trapezoid/Simpson 和低阶 Gauss-Legendre；参考 ladder 使用 Gauss-Legendre 8/16/32/64 与复合 Simpson 16/32/64/128 两个 family。
- 正式 threshold contract 不允许宽于：normalized L1/L2/L∞ 1%、active Spearman 0.99、top-q overlap/Jaccard 0.95、符号一致率 0.99、层/模块质量 TV 1%；reference L1 上限 0.1%，且必须不高于候选 L1 容许误差的十分之一。`max_unique_nodes` 保持严格整数。
- Formal matrix 不能自签资格：必须绑定仍有效的 `FormalExecutionEvidence`，并显式包含 G3-0 至 G3-5 全部 PASS Gate；缺任一 Gate 均拒绝构造。
- 独立 diff 复核发现并推动修复：probe 嵌套字段漏验、伪 SHA、loss contract 漂移、协议阈值过宽、formal matrix 自签，以及新指标尚未接生产分析。修复后 Stage 3 statistics 已调用新指标核心。
- 最终相关回归：`96 passed in 73.39s`；probe 专项 `4 passed`；protocol 专项 `8 passed`；G3-0 artifact 专项 `1 passed`；JSON Schema draft 2020-12 自检与 `git diff --check` 通过。

## 2026-08-28 11:59–12:01 CST — 部分交付与同步边界

- 原子功能提交：`d5d4080972e879fdd2b10269d78dd754a1df629f`（`feat(stage3): harden formal protocol and audit prerequisites`）。
- 分支 `exp/stage3-path-formal-20260828` 已推送到 `origin`，本地 HEAD、remote-tracking ref 与 `ls-remote` 均为同一提交。
- 提交和推送后 Stage 3 工作树 clean；服务器主 repo 仍为 clean detached `44f934dd62d1b86fcb951230c81f3bfa647791aa`，既有 Stage 2 direct-all 工作树仍为 clean `7cd29ed6a2acb958095f96ad1514d8609b74c535`。
- 根工作树的三个用户 presentation dirty 文件保持原状且未进入 Stage 3 提交。
- 未创建服务器 Stage 3 worktree、未同步代码到服务器、未启动 GPU 作业；这是 G3-0 fail-closed 的预期结果。

## 2026-08-28 12:01–12:46 CST — runner、独立 Gate 与恢复边界加固

- 正式多 probe panel 不再隐式取第一条：多 probe 必须用 `active_probe_id` 或 `active_unit_id` 唯一绑定；单 probe 仍保持无歧义兼容。
- 正式 reference 不再沿用本地低节点 fixture ladder：要求显式、冻结、可复算的 Gauss–Legendre 与 composite Simpson 双家族逐级加密配置。
- 路径 runner 分开记录理论唯一节点数与真实 callback 成本，新增 gradient/loss、forward/backward、wall-clock、peak GPU memory、cache hit/miss 诊断；fixture wall-clock 固定为零以保持跨工作树 artifact hash 可复现。
- 新增按 `execution_evidence_hash` 隔离的 per-unit observation ledger；恢复时复算 ledger hash、核对 unit、候选集合与 execution evidence，禁止旧 run 的同名 unit 混入跨 unit 推荐。
- 新增独立 Stage 3 Gate evaluator、CLI 与 JSON Schema。Gate 强制绑定 G3-0..G3-5、完成且 clean 的 formal provenance、冻结阈值、完整候选集合、完整 unit 集合和源 artifact。
- 代码审查修正了两个科学判据错误：失败候选只被淘汰，不再要求所有候选同时通过；完备性绝对/相对残差由 pilot 冻结，不再擅自套用 1% 阈值。S3.2 明定的 L1/L2/L∞、active Spearman、top-q、sign、层/模块 TV、reference error 与节点上限仍禁止放宽。
- Gate 现在实际执行 `max_unique_nodes`，递归拒绝 fixture/synthetic 标签，并要求 observation evidence 是已声明 source refs 的子集，source refs 又必须由 provenance 绑定。正式 recommendation 的资格化还需 G3-7、Gate evaluation、provenance、execution、threshold、unit 与 passing-rule 一致。
- 定向回归：Gate `8 passed`；runner/endpoint `37 passed`；formal orchestration `18 passed`；最终 Gate+runner+orchestration 合并回归 `63 passed`。扩大后的 Stage 3/CLI 套件为 `98 passed, 1 failed`；唯一失败是既有 `test_run_ready_source_examples_compile_with_their_public_schemas` 对 Stage 2 已新增三个 source example 的静态集合仍写死旧四项，与本批 Stage 3 代码无关，未做无关一致性修复。
- JSON Schema draft 2020-12 自检通过；`git diff --check` 通过（仅 Windows LF/CRLF 提示）。
- 独立 G3-0 复核再次确认原阻塞未变化，并发现服务器 GPU/ECC、formal Stage 2 chain 和 Agent 文档名的当前漂移。刷新证据写入 `reports/stage3/g3-0-refresh-20260828.json`，artifact hash `cec23e3f6241dbf0141ebb0838c863bc5b90d4337a07930a1f1a8434d302f490`。
- Beamer、PDF、分析报告和可视化技能已按要求读取；由于没有合法 formal 数据，未创建会被误读为真实结论的报告、图或 slides，也未写 PDF artifact marker。

## 12:46 CST 当前完成边界

- 已完成的是不越 G3-0 的生产 runner/Gate 基础设施、严格回归与刷新审计；这些结果不能替代真实 pilot/formal 数据。
- 仍未启动任何 Stage 3 GPU 作业。PID 724839 保持不触碰；没有服务器写入或同步。
- G3-0 仍需两项明确外部决定：续期 Stage 0 persistence-risk acceptance（或建立第二故障域并完成恢复演练），以及授权 append-only 的 Stage 2 formal validation/replay 来补齐覆盖并发布正式 G2.7/G2.8/S2.11。没有这两项决定时，G3-5 及其后全部正式数据、报告和 Beamer 仍被硬阻断。

## 2026-08-28 15:11 CST — 用户明确变更 Stage 3 前置范围

- 用户明确指示：不再处理此前 Stage 的补做；Stage 0 和 Stage 1 视为全部通过且仍可用；直接采用已完成 Stage 2 的估计器结论进入 Stage 3。
- 本次 Stage 3 输入固定为：默认估计器 `U-32`、`B=32`，`Raw` 仅保留为敏感性对照；Stage 2 来源仍是 `pythia-grid-20260826T145530Z`、分支 `exp/stage2-direct-all-20260826`、提交 `000ce1e79af791ce1eae2e2b62da221a10dd3c9a`。
- 该指令记录为本次 Stage 3 的显式 G3-0 前置范围决定，允许启动 Stage 3 真实实验；它不把任何 `direct-unvalidated` 产物改标为 formal，也不放宽 G3-1 及之后任何科学阈值、tolerance、margin 或数学判据。
- 机器可读证据：`reports/stage3/g3-0-user-scope-decision-20260828.json`；正式 GateRecord：`reports/stage3/g3-0-user-scope-gate-20260828.json`。此前两份 `BLOCKED` 审计原样保留，作为范围变更前的历史证据。

## 2026-08-28 15:11–16:10 CST — 正式观测、矩阵和 Gate 生产闭环加固

- `FormalExecutionEvidence` 与真实 fixed-state provider 已能显式消费用户 G3-0 范围决定；Stage 3 不再要求重验 Stage 0/1 handoff，但 Stage 2 runner 的旧严格边界保持不变。
- 正式求积计划新增逐 `unit_id` 冻结的 `model/stage/update/probe` 分层映射；CLI、JSON Schema、runner 与独立 Gate 都复算并核对这份映射，观测行不能事后改写阶段或 probe 标签。
- 正式参考不再使用硬编码 `1e-12`/单次相邻一致；其归一化 L1 上限来自冻结计划的 `max_reference_normalized_l1_error`，且要求两个家族连续两轮同时通过，避免单个相邻级别偶然一致。
- 正式 observation wire 已补齐 normalized L1/L2/L∞、三种完备性残差、active Spearman、cosine、sign consistency、top-q overlap/Jaccard、layer/module quality TV、reference uncertainty、真实 callback 成本、分层与源证据引用。层/模块标签从上游冻结 S2.3 registry 重载，不重新猜测。
- 完备性相对残差使用冻结稳定常数重算；L1-scaled 残差使用独立 reference contribution 的 L1 质量，不再使用候选自身质量作为分母。
- 独立 Gate 不再信任 producer 提供的 `worst_case` 布尔值；最坏单元按每个冻结分层和所有 Gate 指标从完整表派生，并写出 `worst_case_source=derived_from_complete_frozen_table`。
- 正式矩阵 runner 在 G3-7 前固定 `selected_rule=null`，执行并保存全部冻结候选；G3-6 只在全部预注册单元完成后冻结观测表，G3-9 分析只发布 `PENDING_G3_7` 的未资格化 recommendation 和独立 Gate evaluation。
- 新增 authority-aware recommendation loader：只有同时提供当前 execution、G3-7 PASS、独立 Gate evaluation 与完成/clean provenance 才能重载并资格化正式 recommendation；仅有哈希的 payload 继续 fail-closed。
- 定向回归先后为 `49 passed`、`48 passed`、G3-0/Gate `13 passed`；扩展 Stage 3/CLI 套件为 `88 passed, 1 failed`，唯一失败为测试中漏导入新 authority loader 类，修正后专项通过。run-ready source schema 单独验证为 PASS；既有 source example 静态集合失败仍是 Stage 2 三个历史新增文件导致，未做无关修复。

## 2026-08-29 — 真实执行控制面闭环（服务器启动前）

- 新增严格的单任务物化器、S3.05/S3.06/S3.07 endpoint×probe fan-out 物化器与 runner、pilot/formal phase manifest 组装器；这些组件只编译 hash-bound 配置、selector、状态和命令，不生成科学结果。
- pilot DAG 明确为 S3.01–S3.06，formal DAG 明确为 S3.07–S3.09；S3.01 自身真实执行，不再把 G3-0 scope authority 误当成 S3.01 输出。S3.05 reference coverage 不提前完成 unit ledger；pilot 在 S3.06、formal 在 S3.07 才可提交完整 observation coverage。
- pilot 固定 12 个路径单元；formal 固定 99 个路径单元。S3.07 调度为 99 次 reference 覆盖加 98 次剩余 observation 覆盖，共 197 步；每一步使用不可变 `stage3-path-unit-selector-v1`，正式 S3.03/S3.04 使用独立 `stage3-probe-selector-v1`。
- 修复每任务配置哈希、初始 execution 配置哈希、同根引用解析、Windows live-PID lock 与 G3-5 threshold 重构校验；不放宽任何数值阈值。
- 当前短批回归：orchestrator/fan-out/phase `18 passed`；endpoint/lineage/finalization/scope/trajectory `25 passed`；G3-0/G36/G37/G38/Gate `30 passed`；production plan `5 passed`；Stage 3 task runner 子集 `7 passed`；合计本轮 85 项通过。21 个 `stage3-*.json` 使用 Draft 2020-12 元 schema 校验通过，Python compileall 与 `git diff --check` 通过。
- 当前完成边界仍是服务器启动前：本轮尚未同步服务器、未创建 Stage 3 服务器 worktree、未启动 GPU 作业，也没有任何 pilot/formal 数值结果。下一边界是完成代码审计、提交并推送 clean baseline 后，再按服务器事实物化并 detached 启动真实 pilot。
- 启动前独立审计发现并修复四个真实阻断：fan-out 现在按 endpoint digest 绑定唯一 endpoint/probe refs；S3.03/S3.04 物化必须绑定 canonical formal probe selector；首 shard 失败后使用独立 resume config 恢复；formal phase 现覆盖 S3.07–S3.10，而 G3-8 仍在 S3.10 输出完成后独立发布，消除循环依赖。
- CLI/producer 审计同时修复：pilot endpoint/probe 计划可被严格 dispatcher 接受；endpoint metadata 不再被丢弃；probe selector 从 `record.endpoint_digest` 读取真实 digest；生产索引可消费带 `qualification_gate_hash` 的 probe panel；selector schema `$id` 恢复为仓库统一 `.invalid` authority。
- 审计修复后回归：critical endpoint/fanout/index `26 passed`；orchestrator/fanout/phase `19 passed`；CLI selector/builders `2 passed`；task DAG `1 passed`；21 个 Stage 3 schema 的 Draft 2020-12 与仓库 `$id` 前缀检查均通过。
- 独立报告链审计确认 S3.10 当前仍不足以完成中文报告、PNG/SVG、Beamer 和 G3-8 delivery authority。该缺口不影响 pilot 启动，但属于 Stage 3 最终闭环的硬未完成项，必须在 pilot 长跑期间补齐并在 G3-8 前做端到端验收。

## 2026-09-04 — S3.07 正式长跑与交付桥接加固（进行中）

- 当前正式控制面为 `s307-r4`，冻结范围仍是 99 个 endpoint×probe 单元，没有删减、跳过或重启已完成单元。截至本条记录，状态已持久化到 `next_step=29`，共有 29 份单位结果、34 次含历史恢复的 attempt；`step-029` 子进程仍持续消耗 CPU，唯一实验 GPU 进程位于 GPU0。受保护 PID `724839` 保持存活且未被干预。
- S3.07 的工作区、结果、缓存和恢复状态继续分别固定在 DATA_ROOT 下的命名路径。针对 sealed node bundle、unit memo、average-rank 与 top-q 热路径的修复均以提交和回归测试落地；现有 fan-out 父进程与当前子进程保持原地运行，没有因下游代码准备而重启。
- 独立 delivery bridge `codex/stage3-g36-production-bridge-20260904` 已补齐 S3.08 timed boundary、S3.07→S3.08 handoff audit、G3-6 provenance/publisher、S3.09 base execution、G3-7 finalization、S3.10 四类正式提交、分析表/PNG+SVG/中文报告/Beamer PDF 物化、large-artifact closure、Git sync evidence、source snapshot、G3-8 publisher 与 Stage4 handoff audit。上述能力只准备控制面；尚未把缺失的下游正式结果伪造为 PASS。
- G3-8 replay 报告现在在 producer 和 consumer 两侧都重新打开每个 `input_ref`，检查 workspace/symlink 边界，并把声明的 SHA-256 与真实文件字节绑定。专项 replay、G3-8 publisher 和生产入口回归为 `12 passed`；提交 `230458975b91c9d81168a1391da96b48585b07b7`。
- 一次文件传输误指向正在运行的 S3.07 工作区；核验确认范围仅为两个 tracked G3-8 文件和两个新文件。两个 tracked 文件已从该工作区自身 HEAD 精确恢复，两个新文件移动到 `$DATA_ROOT/tmp/stage3/s307-accidental-edit-recovery-20260904-r1/quarantine`，恢复后工作区 `git status` 为空；运行中的 fan-out/子进程路径未修改。正式补丁随后只应用到 delivery bridge。
- S3.08 dry validation 在 delivery bridge 上保持 fail-closed：handoff 状态为 `IN_PROGRESS`，`completed_unit_count=29`、`required_unit_count=99`，audit hash 为 `e82efeef137ec2225f41e4a04d8a569ae562f7775ca0b4a6866627a26bbb24b9`。S3.08 result、timed state 和 timed receipt 均不存在，未提前启动 S3.08。

## 2026-09-04 当前完成边界

- S3.07 仍在执行第 30 个正式单元（零基 `step-029`）；因此 S3.08、G3-6、S3.09、G3-7、S3.10、G3-8 和 Stage4 handoff 都尚未发布正式 PASS。
- 后续顺序保持为：S3.07 全部 99 单元与聚合提交完成 → S3.08 timed execution/receipt → provenance 与 G3-6 → 在 base execution 上运行 S3.09 → G3-7 → 向 execution chain 依次追加 G3-6、G3-7 → S3.10 → 三层 replay、最终表图文档、large/source/Git manifests → G3-8 → Stage4 audit。不得越过任何前置 authority，也不得用 dry-run、fixture 或未来占位文件替代正式结果。
