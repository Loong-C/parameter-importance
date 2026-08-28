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
