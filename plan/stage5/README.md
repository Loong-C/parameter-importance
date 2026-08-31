# Stage 5：410M 预训练轨迹

Stage 5 把 Stage 4 已验证的训练事务扩展到更大模型和较长预训练，重点是 checkpoint 级重要性轨迹、集中度和 top-k 稳定性。规模、步数、保存频率和数据 revision 均来自配置。

## 声明与执行纪律

Stage 5 route 仍由 `artifact route-build` 从 `training-route-source-v1` 编译，并绑定
Stage 2 decision/PASS Gate；不能复制 Stage 4 输出后手工改模型 ID。后台 prefetch 的
source cursor、pending microbatch 和配置身份进入 checkpoint。恢复时在新 v2 override
显式填同一 route execution 根内的 `recovery.resume_ref`，再调用 `task resume`；存在旧
checkpoint 却调用无引用的 `task run` 必须拒绝。

Stage 5 的 route runner 当前不发布 `training-resource-window-v1`。吞吐、内存或通信
Gate 必须引用单独 profiling/capacity task 的权威产物，不能从训练时长日志或 route
配置推测。服务器、410M 资产与 CUDA 不可用时 formal 保持结构化 `BLOCKED`。

<a id="stage5-pretrain"></a>
## `stage5.pretrain`

执行配置驱动的预训练，输出 `checkpoint_lineage`、`training_metrics`、`importance_trajectory`。按权威训练 checkpoint 恢复，禁止从不完整目录猜测状态。

<a id="stage5-formal-pretraining"></a>
## `stage5.formal_pretraining`

组合正式 route、checkpoint lineage、在线轨迹和阶段报告，输出 `training_route`、`checkpoint_lineage`、`importance_trajectory`、`stage_report`。必须消费合格 Stage 2 decision，并满足真实模型、数据、GPU 和环境前置条件。

<a id="stage5-importance-trajectory"></a>
## `stage5.importance_trajectory`

从冻结 checkpoint/importance snapshot 重建 `importance_trajectory_table`、`concentration_table`、`topk_stability_table`。常量、零质量或未定义统计量保留 `defined=false + reason`。

<a id="stage5-checkpoint-analysis"></a>
## `stage5.checkpoint_analysis`

输出 `checkpoint_analysis_table`、`layer_module_summary`、`heatmap_sources`。只消费 hash-bound 训练源，不修改 checkpoint 或 importance bundle。

<a id="stage5-report"></a>
## `stage5.report`

从 canonical source 重建 `stage_report`、`chart_artifacts`、`gate_summary`。图表必须绑定源表 hash，报告不得把本机 fixture 解释为 410M 正式结论。

## 恢复与阻塞

训练 task 从 `attempt_commit_state` 恢复；分析/报告 task 以冻结源表重新构建。服务器、模型/数据、CUDA 或科学预算缺失时保持 `BLOCKED/UNFROZEN`。
