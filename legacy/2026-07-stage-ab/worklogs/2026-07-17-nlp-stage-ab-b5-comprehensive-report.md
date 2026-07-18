# 2026-07-17 NLP Stage A/B/B.5 统一总结报告

## 本次目标与范围

- 核对仓库中是否已经存在一份能够独立解释 Stage A、Stage B、Stage B.5 全部实验与结论的正式文档。
- 保留既有理论报告、Stage A/B 正式总结和 B.5 诊断，不覆盖、不重命名、不追溯修改原 Gate 5。
- 新增一份中文 Markdown 统一报告，重点解决以下长期混淆：
  - “参考参数”与参考重要性向量的区别；
  - `0.56667`、`0.49731/0.47996`、`0.58333`、`0.962596–0.969190` 分别回答什么问题；
  - signed 与 positive-only 是否仍是同一估计量；
  - 原 Gate 5 失败与 B.5 对称诊断为什么能同时成立；
  - signed U 为什么只能作为有前置稳定性门禁的候选科学主估计量；
  - Stage C 正式实验启动前的 Go/No-Go 条件。

本次只修改文档和工作日志，不修改代码、配置、正式实验产物、服务器任务或下载状态。

## 开始前状态

- 本地仓库：`D:\Personal\Code\parameter-importance`。
- 分支：`feat/nlp-pythia-stage-ab`。
- 开始 HEAD：`e27813513a90c5a73ac1544a2be3948be032b5b7`。
- `origin/feat/nlp-pythia-stage-ab`：`e27813513a90c5a73ac1544a2be3948be032b5b7`。
- 开始时 tracked worktree 为空。
- `git fetch origin` 首次在受限文件系统中因不能写 `.git/FETCH_HEAD` 失败；按授权使用提升权限重新执行后成功。本地与 origin 未分叉。
- 尝试读取服务器 HEAD 时，`sophgo13-via-lab` 主机名临时无法解析；未把连接失败解释为服务器状态异常，待提交同步阶段重试。
- 用户此前要求保留的 `Agents/工作日志与多端同步规范.md` 原有删除没有被暂存、恢复或覆盖。

## 现有材料核对

确认现有材料分为三类，尚无一份 B.5 后的统一总报告：

1. `docs/parameter_importance_u_statistic_report.md`：理论推导，不含实际 Stage A/B 结果；
2. `worklogs/2026-07-16-nlp-stage-ab-final-summary.md`：Stage A/B 正式总结，形成于 B.5 之前；
3. `docs/stage_b_u_double_near_zero_diagnostic_summary.md`：B.5 专项诊断，只聚焦 U/double 语义与剪枝；
4. `plan/NLP参数重要性完整实验计划书.md` 与 `docs/stage_ab_runbook.md`：计划和执行规范；
5. 服务器 `reports/stage-ab-minimum-loop/report.md`：正式机器汇总，包含图表索引，但没有纳入 B.5 更新解释。

因此新增统一报告是必要的，不以任一旧文档冒充已经存在的整合结论。

## 实际修改

新增：

- `docs/nlp_stage_ab_b5_comprehensive_report.md`
  - 17 个主章节；
  - 统一定义局部 gradient-space 目标、raw、double、U、signed 与 positive-only；
  - 解释 4096-sample 参考重要性向量和 40 个 `checkpoint × M` 条件中 `0.49731/0.47996` 的真实含义；
  - 完整记录 Stage A estimator、相邻 M 稳定性、positive-only v3 和 numerical integration v1/v2；
  - 完整记录 Stage B 预训练、离线 baseline、controlled、matched-performance、best-performance、六项 pruning 和三项 reuse intervention；
  - 保留原 Stage 15.8 七项门禁及 Gate 5 正式失败；
  - 纳入 B.5 的 108 行设计、68/72、69/72、72/72、排序范围、符号质量和更新解释；
  - 对结论分为“证据较强”“有限范围”“尚未解决”；
  - 给出 Stage C 方法口径、稳定性 pilot、稀疏 double calibration 和 Go/No-Go；
  - 记录关键结果目录、提交、测试、provenance 和 `_SUCCESS` 哈希。
- `worklogs/2026-07-17-nlp-stage-ab-b5-comprehensive-report.md`
  - 本日志。

没有修改以下旧报告和正式结果：

- `docs/parameter_importance_u_statistic_report.md`；
- `worklogs/2026-07-16-nlp-stage-ab-final-summary.md`；
- `docs/stage_b_u_double_near_zero_diagnostic_summary.md`；
- `/home/sophgo13/cjl/storage/parameter-importance/reports/stage-ab-minimum-loop`；
- `/home/sophgo13/cjl/storage/parameter-importance/results/stage-b-u-double-near-zero-diagnostic-v1`。

## 内容依据与关键复核

- Stage A estimator 设计：10 checkpoints、`M={4,8,16,32}`、每单元 200 repetitions、4096 reference、共 40 单元。
- estimator 聚合：layer/module U/raw bias ratio `0.02037/0.02000`；U/double 参数 Spearman 均值 `0.49731/0.47996`；MSE ratio `0.91496`；formula-time ratio `0.765`。
- signed 相邻 M：参数最低 `0.39572`、层 signed-sum `0.58333`、模块 `0.93333`；positive-only 参数/层/模块最低 `0.9299466/0.9166667/0.95`。
- numerical：left layer `0.56667` 正式失败；trapezoid 参数/层/top-5% 最低 `0.98322/0.96667/0.97389` 通过。
- controlled accuracy：direct `0.8100153`、pretrained `0.8176606`；多数类 `0.5091743`。
- pruning AUC：double `0.0759237`、U `0.0725815`、raw `0.0716858`、movement `0.0558577`、magnitude `0.0287843`。
- 原 Gate 5：`68/72`，四项 layer-balanced NLL 失败，正式状态不变。
- B.5：positive U 对 double near-zero `69/72`；signed U 对 signed double near-zero `72/72`；signed U 对 double 参数 Spearman `0.962596–0.969190`，layer/module 均为 `1.0`。
- B.5 正式结果：108 行，provenance `fb3e818056fa36e82ec37ddba306e9d924d2676d054e28e37fe444bbe3b35bfa`，`_SUCCESS` SHA-256 `cdc3eff0cdeaee155bdb18f249a6b817ebb583d6dfdde536dd0efe299e036ee6`。

## 检查与结果

已执行：

```powershell
git -c safe.directory=D:/Personal/Code/parameter-importance diff --check
git -c safe.directory=D:/Personal/Code/parameter-importance diff --stat
rg -n "^## |0\.49731|0\.47996|0\.58333|0\.962596|68/72|72/72|Gate 5|Stage C" docs/nlp_stage_ab_b5_comprehensive_report.md
```

结果：

- `git diff --check` 通过；
- 新报告为中文 Markdown，共 786 行；
- 23 组块级公式起止定界符数量一致；
- 关键数字、原 Gate 5 状态、B.5 更新解释和 Stage C 条件均可在文档中定位；
- 显式暂存后 `git diff --cached --check` 通过；暂存区只有本次两个新增文件，共 897 行，无其他路径；
- 本次无代码变化，因此不触发新的服务器全套代码回归；正式实验回归证据仍绑定原代码提交，不把 Markdown 检查冒充代码测试。

## 问题、恢复与限制

1. 首次写入报告时，部分 inline LaTeX 的反斜杠被工具字符串转义吞掉；已用补丁改为 `$...$` 定界，并重新检查关键行。块级公式 23 组保持成对。
2. 当前报告是仓库正式 Markdown，不是 DOCX；因此 DOCX 渲染和页面 PNG 检查不适用。没有生成额外 Word/PDF 副本。
3. 服务器 SSH 在开始阶段出现跳板主机名临时解析失败；本地报告编写不依赖服务器写操作。提交同步阶段必须重试并核验服务器工作树，不得在无法连接时宣称三端同步。

## 遗留风险与后续动作

- 科学风险仍为 signed U 的单 checkpoint/相邻 M 稳定性；统一报告没有把 B.5 写成该风险已消失。
- 原 Gate 5 正式失败仍保留；Stage C 仍需独立预注册稳定性 pilot 与 sparse double calibration。
- Pile 下载与公共 DNS/镜像稳定性属于既有基础设施风险，本次未修改。
- 下一步：完成文档差异复核，显式暂存本次两个文件，提交、推送，并通过安全快进把服务器同步到同一提交；随后记录最终三端状态。

## Git 与多端同步

- 首次文档提交：`7769b812001240c721338d146baae35351b142a9`，提交说明 `docs: add unified NLP Stage A B B.5 report`。
- 推送前重新执行 `git fetch origin`，确认 `origin/feat/nlp-pythia-stage-ab` 的 `e27813513a90c5a73ac1544a2be3948be032b5b7` 是本地提交祖先，随后正常快进推送，未强推。
- 创建增量 bundle `.sync-7769b81.bundle`，要求基底 `e27813513a90c5a73ac1544a2be3948be032b5b7`，`git bundle verify` 通过，大小 19,156 bytes。
- 服务器同步前 HEAD 为 `e27813513a90c5a73ac1544a2be3948be032b5b7`，tracked worktree 为空；通过 bundle `git fetch` 后 `git merge --ff-only 7769b812...` 成功，服务器到达 `7769b812001240c721338d146baae35351b142a9`，tracked worktree 仍为空。
- 远端与本地精确临时 bundle 均已删除并验证不存在。
- 本日志补记形成收口提交后，将再次按相同方式推送并快进服务器；最终 HEAD 以本文件所在提交为准，并在最终回复中给出本地、origin、服务器三个完整提交号。
