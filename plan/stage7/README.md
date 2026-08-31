# Stage 7：功能性剪枝验证

Stage 7 用真实模型任务损伤验证重要性排序。剪枝采用 alias-aware 非破坏性上下文；退出 cell 后恢复原参数，禁止把 mask 写回权威 checkpoint。

<a id="stage7-matrix"></a>
## `stage7.matrix`

根据 importance source、方向（高/低/随机）、范围（全局/层平衡）、比例和 seed 编译 canonical 矩阵，输出 `pruning_matrix`、`pruning_plans`、`mask_manifest`。并列按 canonical coordinate ID 决胜。

<a id="stage7-evaluate"></a>
## `stage7.evaluate`

逐 cell 加载 hash-bound checkpoint/importance，应用临时 mask，运行任务 evaluator 并恢复参数。输出 `pruning_evaluation_results`、`damage_curves`；按 cell shard commit 恢复。

<a id="stage7-functional-pruning-validation"></a>
## `stage7.functional_pruning_validation`

组合 plan、实际评价、damage AUC 和阶段摘要，输出 `pruning_plan`、`pruning_results`、`damage_auc_table`、`stage_report`。local fixture 只验证执行链，不能生成正式剪枝结论。

<a id="stage7-reduce"></a>
## `stage7.reduce`

从完整 cell 源表计算 `pruning_summary_table`、`damage_auc_table`、`confidence_intervals`。随机基线和各重要性来源必须使用冻结 seed/预算，缺 cell 不得静默忽略。

<a id="stage7-report"></a>
## `stage7.report`

重建 `stage_report`、`chart_artifacts`、`gate_summary`，包括高/低/随机、多比例和层平衡对比。图表和结论都绑定冻结源表 hash。

## 支持的重要性来源

magnitude、movement、raw、U/double、empirical Fisher 和 SI 都通过 artifact producer 接口接入；某个来源能否使用由输入 artifact 决定，而不是由报告代码假设。

## 显式恢复

Stage 7 的 matrix/evaluate 按 `shard_commit` 恢复。再次执行时必须使用相同的冻结矩阵、
checkpoint/importance hash 和 `artifacts.output_dir`，在 v2 recovery 中填入权威 shard
引用后调用 `task resume`。runner 只复用通过内容 hash 与 cell 身份校验的 commit；单独
存在的 object、临时 mask 或日志行都不算完成。当前实现按相同 output dir 自动发现
cell commit，`resume_ref` 还不能用来精确挑选单个 cell；它只明确区分 resume 与新 run。
reduce/report 属于
`canonical_source` 重建，不应从训练 checkpoint 恢复。
