# Stage 4：160M 最小完整闭环

Stage 4 验证同一 base initialization 下的 pretrain、direct supervised、finetune、在线重要性轨迹和剪枝评价能组成可恢复闭环。模型规模和任务来自资产 ID/config，不在 runner 中硬编码。

## 共同前置条件

- 已验证的模型、tokenizer、数据 manifest 与离线 root；
- 合格的 Stage 2 `EstimatorDecision`；启用路径审计时还需 Stage 3 recommendation；
- route spec 中的 base initialization、phase 输入和 checkpoint lineage 全部 hash-bound；
- 本机 fixture 可用 tiny provider 验证执行链，但 `formal_eligible=false`。

## 声明式入口与运行边界

- 路线 source 使用 `training-route-source-v1`，经
  `artifact route-build --spec ... --decision ... --gate ... --output ...` 编译；调用者
  不手算 `lineage_hash`。只要任一 phase 启用 importance，就必须绑定 Stage 2 decision；
  formal 还必须绑定匹配的独立 PASS Gate。
- 训练 endpoint 选择使用 `artifact endpoint-plan-build`；endpoint commit 产生 digest
  后，再用 `artifact probe-plan-build` 冻结 probe。两个 formal builder 都必须绑定同一
  `FormalExecutionEvidence`，local fixture 则不得携带它。
- `data_loader.num_workers>0` 会启用有序线程预取；pending microbatch 与 cursor 一同
  checkpoint。`num_workers=0` 时必须令 `prefetch_factor=null`、
  `persistent_workers=false`。
- route 训练从同一 `artifacts.output_dir/route-execution/<lineage-hash>/` 下的权威
  phase/checkpoint commit 恢复；新 override 必须显式填 `recovery.resume_ref`，然后调用
  `task resume`。`task run` 不得带恢复引用。
- 当前 route catalog 不产出 `resource_profiles`；profiling 性能证据应由独立
  TrainingTaskRunner/capacity task 生成，不能根据 route 配置字段推断已经测量。

<a id="stage4-route"></a>
## `stage4.route`

冻结并验证 training route、phase DAG、共享初始化、decision 引用和 finetune lineage。输出 `training_route`、`route_validation_report`；按 `immutable_publish` 幂等重启。

<a id="stage4-pretrain"></a>
## `stage4.pretrain`

执行 pretrain phase，产生 `checkpoint_lineage`、`training_metrics`、`importance_trajectory`。从 optimizer-step 的 `attempt_commit_state` 恢复；checkpoint 包含训练状态、数据游标、RNG 和重要性累计器。

<a id="stage4-direct-supervised"></a>
## `stage4.direct_supervised`

从与预训练路线相同的 base initialization 启动直接监督训练，禁止读取预训练后权重。输出与 pretrain 相同的三类 lineage/metrics/trajectory 产物，恢复边界为 `attempt_commit_state`。

<a id="stage4-finetune"></a>
## `stage4.finetune`

只接受对应 pretrain lineage 的 checkpoint 作为输入，执行监督微调并发布新的子 lineage。跨路线 checkpoint 或不匹配的 base hash 必须拒绝。

<a id="stage4-importance-trajectory"></a>
## `stage4.importance_trajectory`

从冻结训练输出重建 `importance_trajectory_table` 与 `layer_module_summary`。这是 `canonical_source` 派生任务；恢复方式是重新构建，不修改原始训练数值。

<a id="stage4-pruning-validation"></a>
## `stage4.pruning_validation`

加载 hash-bound checkpoint 和 importance，执行非破坏性剪枝评价。输出 `pruning_plan`、`pruning_results`、`damage_auc_table`；每个 mask/cell 使用不可变 shard commit 恢复。

<a id="stage4-minimal-complete-loop"></a>
## `stage4.minimal_complete_loop`

执行 Stage 4 的缩合闭环：路线训练、重要性轨迹、剪枝结果和阶段报告。输出 `training_route`、`importance_trajectory`、`pruning_results`、`stage_report`。formal 条件缺失时整体 `BLOCKED`，不得以 tiny 结果替代。

<a id="stage4-report"></a>
## `stage4.report`

只从冻结路线、轨迹和剪枝源表重建 `stage_report`、`chart_artifacts`、`gate_summary`。报告任务不自行修改 Gate，也不手工填入数字。

## 退出条件

每个 task 的预期产物、恢复边界和 hash 可由 catalog/status 重放；formal Gate 是否通过由独立证据与审核决定。
