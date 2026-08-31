# Stage 6：监督与预训练微调路线比较

Stage 6 比较 SST-2、MNLI、RTE 等任务上的 direct supervised 与 pretrain→finetune 路线。任务名、label 数和 evaluator 由配置/manifest 决定，runner 不绑定特定服务器路径。

## 路线声明、预取与恢复

每条待比较路线先分别用 `artifact route-build` 编译。`stage6.route_matrix` 从
`orchestration.input_result_refs` 读取至少两条 hash-bound `TrainingRouteSpec`，重新验证
共享 base initialization，再冻结配对矩阵；builder 不替代这个跨路线验证 task。

训练路线可启用有序线程 prefetch，pending batch 与 cursor 一同保存。中断后必须在新
override 中显式设置位于自身 `route-execution/<lineage-hash>/` 内的
`recovery.resume_ref` 并调用 `task resume`，不能从另一条路线或另一 seed 的目录恢复。
当前 route 输出不包含资源 profiling window；路线公平性中的性能预算需由独立测量产物
证明。

<a id="stage6-route-matrix"></a>
## `stage6.route_matrix`

编译并验证路线矩阵，输出 `training_routes`、`route_matrix`、`matrix_validation_report`。每个比较对必须共享 base initialization，只有预训练路线的对应 finetune phase 可以消费其 pretrain lineage。

<a id="stage6-training-route-comparison"></a>
## `stage6.training_route_comparison`

执行多路线 phase DAG，输出 `training_routes`、`route_comparison_table`、`importance_reuse_report`、`stage_report`。训练中断从最近权威 `attempt_commit_state` 恢复。

<a id="stage6-evaluate"></a>
## `stage6.evaluate`

在固定任务 split、metric 和 seed 上评价各路线，输出 `evaluation_results`、`paired_route_metrics`。评价按 shard commit 恢复，并保留样本/预测 lineage；formal 必须使用真实 evaluator 和资产。

<a id="stage6-compare"></a>
## `stage6.compare`

从配对评价源表计算 `route_comparison_table`、`confidence_intervals`、`quality_gates`。配对单位、缺失 cell、多重比较和未定义统计量必须显式记录。

<a id="stage6-importance-reuse"></a>
## `stage6.importance_reuse`

分析预训练重要性在下游任务中的复用，输出 `importance_reuse_table`、`topk_overlap_table`、`layer_module_difference`。坐标集合必须由同一 registry/alias 合同对齐。

<a id="stage6-report"></a>
## `stage6.report`

从冻结比较与复用源表重建 `stage_report`、`chart_artifacts`、`gate_summary`，不从日志文本或手工 CSV 获取数字。

## 路线公平性

比较预算、初始化、seed 域、评价 split 和任务指标必须在 matrix artifact 中冻结。配置差异超过声明的单一路线因素时，比较任务应 fail-closed。
