# Stage 4–9 Run-Ready 任务索引与运行手册

本索引补充 `general_plan.md` 的研究路线，详细任务合同以各 Stage README 和机器可读 `StageTaskCatalog` 为准：

- [Stage 4：160M 最小闭环](stage4/README.md)
- [Stage 5：410M 预训练轨迹](stage5/README.md)
- [Stage 6：监督与微调路线比较](stage6/README.md)
- [Stage 7：功能性剪枝验证](stage7/README.md)
- [Stage 8：消融与稳健性](stage8/README.md)
- [Stage 9：分析、图表、报告与交付](stage9/README.md)

## 任务引用规则

每个 task 均有稳定锚点。`plan_ref` 的规则是：

```text
plan/stage<stage>/README.md#<task_id 中的点和下划线替换为连字符>
```

例如：

```text
stage4.direct_supervised
-> plan/stage4/README.md#stage4-direct-supervised

stage9.analysis_visualization_reporting
-> plan/stage9/README.md#stage9-analysis-visualization-reporting
```

## 通用执行顺序

1. 用 `task catalog --task-id <id>` 读取 hash-bound task 定义。
2. 用 builder 把 route、endpoint/probe 或 ablation source 编译成 hash-bound artifact；
   不手算 hash，也不覆盖已有目标。
3. 从版本化 v1 科学合同和 v2 override 生成完整 `ResolvedConfig v2`。
4. 用 `asset verify` 校验所有 manifest/root 引用，不从运行入口下载资产。
5. 用 `task preflight` 检查 runner、能力、Gate、decision 和输入 artifact。
6. `ready=true` 后执行 `task run`；中断后在新 override 中显式设置
   `recovery.resume_ref`，重新 resolve 后使用 `task resume`。
7. 用 `task status` 读取结果；需要确定性复核时使用 `task replay`。
8. 审核 pilot/decision/recommendation，随后另行构建 Gate。
9. 只有可最终确认的任务结果才执行 `task finalize`。

任务成功表示 runner 按合同完成，不自动表示 formal Gate `PASS`。本机 fixture 的输出始终没有 formal 资格。

## 配置与产物纪律

- 允许变更：版本化 YAML/JSON 配置、资产 manifest、审核和 Gate 决策。
- 不允许：在运行期间临时增加 Python/shell/PowerShell 实验逻辑、手工拼表、手工填结果数字。
- 所有正式输入必须是 canonical JSON、安全 TensorBundle 或不可变 task commit，并绑定 SHA-256。
- 训练路线必须满足共享 base initialization 和 pretrain/finetune lineage；Stage 4–6 在线重要性必须消费通过前置 Gate 的 Stage 2 decision。
- Stage 9 只能读取冻结源表，派生表、图和报告必须可由 source hash 重建。

## 声明式构建入口

builder 的输入是应纳入版本控制的无顶层 hash source/spec，输出是不可变 canonical
artifact。以下路径只是命令形状，调用者需要按对应合同创建 source 文件：

```powershell
# Stage 4--6：phase DAG、共享初始化和 checkpoint lineage
param-importance artifact route-build `
  --spec <training-route-source-v1.yaml> `
  --decision <stage2-estimator-decision-or-envelope.json> `
  --gate <stage2-pass-gate-or-envelope.json> `
  --output manifests/routes/<route-id>.json

# Stage 8：声明与完整单因素矩阵可一次生成
param-importance artifact ablation-matrix-build `
  --spec <ablation-matrix-source-v1.yaml> `
  --output manifests/ablation/<matrix-id>-declaration.json `
  --compiled-output manifests/ablation/<matrix-id>.json
```

Stage 4+ 启用路径审计前，还需要 Stage 3 训练 endpoint 与固定 probe：

```powershell
param-importance artifact endpoint-plan-build `
  --plan-id <plan-id> --step <optimizer-step> --include-checkpoint-steps `
  --scope formal --formal-execution-evidence <formal-execution-evidence.json> `
  --output manifests/stage3/endpoint-plan.json

# 训练捕获 endpoint commit 并产生 endpoint_digest 后，再编译 probe plan。
param-importance artifact probe-plan-build `
  --spec <probe-plan-without-top-level-hash.yaml> `
  --scope formal --formal-execution-evidence <formal-execution-evidence.json> `
  --output manifests/stage3/probe-plan.json
```

local fixture 使用 `--scope local_fixture` 且必须省略 formal evidence。formal builder
不会把本机 plan 提升为正式输入；它会复核 execution evidence 与前置 Gate。

## 从空输出目录执行

完整命令示例见仓库根 [Readme](../Readme.md)。对任意 Stage 4–9 task，命令形状保持一致：

```powershell
param-importance task preflight --config <resolved-v2.json> --environment <environment.json>
param-importance task run       --config <resolved-v2.json> --environment <environment.json> --result <result.json>
param-importance task resume    --config <resume-v2.json>   --environment <environment.json> --result <resumed.json>
param-importance task status    --result <result.json>      --config <resolved-v2.json>
param-importance task replay    --config <resolved-v2.json> --source-result <result.json> --result <replay-result.json>
param-importance task finalize  --result <result.json>      --output <finalization.json>
```

缺少服务器、CUDA/NCCL、模型/数据资产或 formal decision 时，正式任务应返回结构化 `BLOCKED`。这类 preflight 结果是正确的 fail-closed 行为。

`task run` 与 `task resume` 不是别名：前者要求 `recovery.resume_ref=null`，后者要求
非空。训练任务可绑定 checkpoint 根、`latest.json` 或确切 commit；route task 的引用
必须位于自身 `route-execution/<lineage-hash>/`。按 shard 恢复的 Stage 7/8 task 只复用
相同 output dir 中通过 hash 校验的 cell commit；当前 shard runner 不使用
`resume_ref` 精确选择某个 cell，该字段只使 CLI 进入显式 resume 语义。派生任务按 catalog 的
`canonical_source` 重新构建。不得把临时目录、未提交 object 或另一次 run 的 shard
填入 `resume_ref`。

## Data loader、profiling 与实现边界

- `num_workers=0` 时 `prefetch_factor` 必须为 `null` 且
  `persistent_workers=false`；`num_workers>0` 时必须显式给正的 prefetch factor。
- 当前 prefetch 是有序线程 cursor。pending microbatch 与 source cursor 一同进入安全
  checkpoint；Stage 0/1 通用训练和 Stage 4–6 route runner 都消费这一状态。
- profiling 要求显式 step 预算、完整 warmup/measure/repetition 窗口和至少一个测量项；
  中断恢复必须回到窗口起点，不能拼接两段墙钟时间。
- 当前 `training-resource-window-v1` producer 位于通用 `TrainingTaskRunner`。Stage 4–6
  route catalog 尚未声明 `resource_profiles` 输出，所以 route 中出现 profiling 配置不
  等价于已有正式性能证据。性能 Gate 必须引用独立 profiling/capacity task 产物。
- backend 无精确通信计数时结果保持 `defined=false + reason`；本机 Python memory 仅是
  `tracemalloc` 范围，不能冒充 CUDA allocator 峰值。

## 代表性 Stage 0–9 fixture

从两个全新工作根执行：

```powershell
param-importance task fixture-all --workspace-root artifacts/fixture-all/run-a
param-importance task fixture-all --workspace-root artifacts/fixture-all/run-b
```

比较两个 `full-fixture-result.json` 的文件 hash 和内部 `result_hash`。该命令执行每个
Stage 的代表性缩小路径，而不是 catalog 每个 task 的穷举矩阵；它也不覆盖 DDP 2/4、
offline HF、profiling 或 formal 环境。结果始终是 `scope=local_fixture`、
`formal_eligible=false`，不能改变任何 Gate。
