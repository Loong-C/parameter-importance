# S1.10 checkpoint、断点恢复与证据包

## 1. 目的

证明 Stage 1 的训练与统计状态能够在完整 step 边界安全保存，并在新进程中恢复后继续产生与不中断运行一致的数据、梯度、重要性、optimizer 更新和累计结果。同时定义失败时足以离线复算的证据包。

## 2. 前置 Gate

- `G1-STEP`、`G1-SINGLE` 与 `G1-NUMERIC` 已通过。
- 四卡恢复验收还要求 `G1-DDP` 已通过。
- registry、配置和 checkpoint schema 版本已冻结。

## 3. 必须保存的状态

### 3.1 训练状态

- 模型参数与 buffer；
- optimizer state、参数组、`attempt_step` 与 `successful_optimizer_step`；
- scheduler state 与下一步实际学习率；
- GradScaler 或等价 loss-scaling 的 scale、growth tracker 与 inf-check 状态；
- Python `random`、NumPy、PyTorch CPU、每 rank CUDA RNG，以及代码显式创建的 `Generator` 状态；
- 数据 planner、sampler generator、DataLoader worker seed、epoch/step 和精确样本游标；
- 当前 world size、rank 映射和 global batch 规划摘要。

### 3.2 参数重要性状态

- registry manifest 与 hash；
- `importance_signed`；
- `importance_positive`；
- `importance_negative_mass`；
- `importance_absolute`；
- raw 累计，以及启用时独立保存的 raw-clipped 累计；
- data movement、带符号累计 data update、data net movement、total/weight-decay movement；
- 启用时的 `actual_update_raw_importance` 等长期诊断累计；
- 初始参数或计算 total net movement 所需的稳定参考；
- attempt 数、successful optimizer step 数、attempted/committed 统计单元数和 skip 次数；
- 上一个完整 training attempt 的 execute/skip 状态与标量元数据。

所有在解析后配置中启用的长期公开字段都必须进入 checkpoint schema、hash 和 continuous/resume 对照；不得用“必要诊断”作为遗漏字段的兜底。

### 3.3 Provenance

- Git commit、分支和工作树状态摘要；
- 解析后配置与摘要 hash；
- 环境与依赖身份；
- 模型、tokenizer、数据 revision 与文件 hash；
- fixture ID、样本 ID 和 token hash；
- schema/version 与生成时间；
- 父运行、恢复来源 checkpoint 和新运行 ID。

Provenance 只按预注册 allowlist 保存；不得 dump 全环境，不复制 token、Cookie、认证头、签名 URL、SSH 配置或资产 manifest 中的 `transport_endpoint`。

## 4. 执行步骤

### 4.1 实现安全发布

1. 只允许在完整 training-attempt 边界触发正式保存；该边界可以是一次成功 optimizer step，也可以是已完整提交数据/RNG/scaler/skip 状态的 skip。
2. 把正式目标冻结为 `$DATA_ROOT/checkpoints/stage1/<run_id>/<checkpoint_id>/`。
3. 先把各状态写入同一文件系统的 `$DATA_ROOT/tmp/stage1/<run_id>/<attempt>/<object_id>/`，禁止回退系统 `/tmp`。
4. 在完整 attempt 边界让所有 rank 进入带有限超时的保存 barrier。
5. 发布前收集每 rank 的模型、buffer、optimizer、scheduler、scaler、registry 与全部长期累计摘要，并验证共享状态 checksum 一致。
6. Stage 1 要求各 rank 的 scale/growth tracker 一致；不一致直接失败，不退化成含混的 per-rank scaler schema。
7. 由 rank 0 写模型、buffer、optimizer、scheduler、scaler、registry 与共享重要性状态。
8. 各 rank 只写其私有 RNG/Generator、数据分片与 worker seed 状态，并向 rank 0 汇报文件身份。
9. 为每个文件记录大小和 SHA-256。
10. 由 rank 0 验证所有预期 rank 分片齐全，并写入文件清单与 schema/version。
11. 所有 rank 在有限超时内返回 pre-commit ack；任一缺失或失败都不得进入 commit。
12. rank 0 在临时目录内写好完成标记并完成结构与哈希自检。
13. rank 0 通过同盘原子目录 rename 发布含完成标记的不可变 checkpoint 对象；
    rename 只表示对象字节已发布，不是可恢复性的 commit point。
14. rank 0 重新打开并校验已发布对象后，原子发布独立的权威
    `checkpoint_commit`，绑定对象 manifest hash、父 lineage、generation 与
    canonical event 指针；该小型 commit 才是唯一恢复发现依据。
15. commit 后把结果广播给所有 rank；若广播失败，已提交 checkpoint 保持有效且不可回滚，同时本次运行记录协调失败供人工处理。
16. 恢复入口只发现 commit 指向且完整标记、manifest、文件 hash 均通过的对象；
    `latest` 与目录时间戳都是可重建派生视图，不具权威性。
17. 不覆盖已有 checkpoint。
18. commit 前失败的临时目录或已发布孤儿对象保留为诊断，并由 reconciliation
    标成 orphan；二者都不伪装成可恢复状态。
19. barrier、汇报、ack 和广播均有预注册有限超时，rank 死亡不能让其余进程无限等待。
20. 清理时只处理本次 `<run_id>/<attempt>/<object_id>`，不使用宽泛递归目标。

### 4.2 实现恢复前校验

1. 读取 checkpoint schema/version。
2. 核对 registry hash。
3. 核对参数名称、shape、numel 和参数组。
4. 核对模型与 tokenizer revision。
5. 核对 loss reduction 与 estimator 契约版本。
6. 核对配置中不允许变化的字段。
7. 核对全部文件大小和 SHA-256。
8. 核对 world-size 兼容策略。
9. 任一硬条件不一致时 fail-closed。
10. 错误信息指出具体不一致项。
11. correctness 恢复 Gate 固定使用单进程/无预取数据加载；若启用 worker prefetch，必须额外保存并恢复“已预取但未消费”的边界后才可计入 Gate。

### 4.3 单卡连续/恢复对照

1. 固定总 attempt 数 `N`、中断点 `K`，并在轨迹中注入一次受控 non-finite skip。
2. 分别把 `K` 放在 skip 前的成功边界和 skip 完成后的边界，形成两组恢复 case。
3. 运行一条不中断参考轨迹。
4. 从同一初始状态运行到 `K` 并保存。
5. 在新进程中加载该 checkpoint。
6. 继续执行剩余 attempts。
7. 比较恢复后的下一批样本 ID。
8. 直接比较 Python/NumPy/Torch/CUDA/显式 Generator/sampler 的状态摘要，并在不扰动正式状态的副本上比较下一段随机序列。
9. 比较每一步 attempt、successful step、skip count、scale、growth tracker、学习率、loss 和 clip factor。
10. 比较每一步 `S1/S2` 或加权充分统计量。
11. 比较每一步 raw、raw-clipped、U 和四类累计量中的全部启用字段。
12. 比较每一步 optimizer state 与参数。
13. 比较 data/total/weight-decay movement、两类 net movement、magnitude 与全部启用诊断累计。
14. 至少比较恢复后连续 3 个完整 attempts，且覆盖 skip 后的第一个正常 step。

### 4.4 四卡连续/恢复对照

1. 固定四卡健康 UUID 集合与 rank 映射。
2. 在完整 attempt barrier 后收集每 rank 的 Python/NumPy/Torch/CUDA RNG、显式 Generator 和数据状态。
3. 比较各 rank 共享模型/optimizer/scaler/累计状态 checksum，再由 rank 0 按唯一 commit point 发布。
4. 在新四进程组中加载 checkpoint。
5. 验证每 rank 恢复到预期样本分片。
6. 验证全局 sample multiset 与不中断运行一致。
7. 验证 all-reduce 后充分统计量一致。
8. 逐步比较全部训练与重要性状态。
9. 验证所有 rank 的 attempt、successful step 与 skip count 一致。
10. 验证恢复运行正常退出且 run-owned PID/PGID/租约均已释放。
11. 在四卡轨迹中也跨越一次受控 skip，逐 rank 直接比较 RNG/Generator 摘要、下一段随机序列、scale 与 growth tracker。

### 4.5 破损与不兼容测试

1. 只在 `$DATA_ROOT/tmp/stage1/<run_id>/corrupt-fixtures/` 下复制并构造缺少完成标记的 checkpoint，不修改正式 checkpoint。
2. 构造缺少一个状态文件的 checkpoint。
3. 构造一个哈希不匹配文件。
4. 构造 registry hash 不匹配状态。
5. 构造 schema/version 不兼容状态。
6. 构造 loss reduction 或 estimator 契约不一致状态。
7. 验证恢复入口逐一拒绝。
8. 验证拒绝不会修改当前模型、正式 checkpoint 或运行目录。
9. 测试结束后只按 fixture manifest 精确清理本次损坏副本。

### 4.6 定义失败复现证据包

任一 Gate 失败时，保存以下最小集合：

1. 解析后配置与配置 hash。
2. 环境、commit、资产与 registry manifest。
3. 失败样本 ID、token hash、rank/microbatch 映射和有效 token 数。
4. 失败 step 的 local gradients 或可重构的小型梯度子集。
5. `S1/S2` 或 `G1/G2/N1/N2`。
6. 学习率、gradient norm、clip factor 和 optimizer step 元数据。
7. raw、U、累计量和参考 oracle。
8. measured、threshold、误差位置和 traceback。
9. 大型数组的绝对服务器路径、大小和 SHA-256。
10. 明确的失败状态文件，不生成成功标记。

## 5. 产出

- 版本化 checkpoint schema。
- checkpoint 原子发布和完整性校验模块。
- 分布式 pre-commit/commit/广播协议与超时测试报告。
- 单卡连续/恢复逐步对照表。
- 四卡 fresh-process 连续/恢复逐步对照表。
- 破损/不兼容 checkpoint 拒绝测试报告。
- 标准失败复现 bundle schema。
- 大型调试产物 manifest。
- `G1-RESUME` 机器可读报告。

## 6. 可视化

- 连续运行与恢复运行的逐步最大参数误差曲线。
- `S1/S2/U/accumulator` 的逐步 normalized L2 误差图。
- 数据样本游标与 optimizer step 的对齐时间线。
- attempt/successful step/skip、scale 与 growth tracker 的对齐时间线。
- checkpoint 文件大小、保存时长和读取时长表。
- 破损 fixture 的拒绝原因矩阵。

## 7. 核验标准

- 样本 ID、token hash、registry、schema、attempt/successful step、统计单元计数、skip count 和数据游标完全一致。
- 同一路由、同一硬件、锁定 correctness kernel 和确定性模式下，单卡恢复后的样本序列、参数、optimizer/scaler state 和累计数组必须逐步 bitwise 一致；只有运行前具名登记的非确定性算子才允许改用对应容差。
- 四卡连续/恢复逐张量通过 `T32_DISTRIBUTED`。
- 恢复后至少 3 步的 `S1/S2`、raw、U、clip、更新和累计状态全部通过。
- 所有破损或不兼容 checkpoint 都被拒绝，且拒绝前不修改活动状态。
- 只有 manifest、哈希、自检和完成标记全部通过的 checkpoint 可被发现。
- 四卡保存只有在有限超时 barrier、共享状态 checksum、四份私有 rank 状态和全部 pre-commit ack 通过后，才由 rank 0 执行唯一原子 commit；commit 后广播失败不得删除或回滚有效 checkpoint。
- 每类 RNG/Generator 在每 rank 的状态摘要与下一段随机序列均一致，受控 skip 前后恢复轨迹无跳样或重样。
- 解析后配置中启用的全部长期公开字段都已保存并参加逐步 continuous/resume 对照。
- 任一失败可由证据包在不重新采样的条件下离线复算。

## 8. Gate 与后续依赖

- `G1-RESUME` 通过后，S1.11 才能生成最终阶段报告。
- checkpoint schema、registry、数据 planner、精度或累计状态变化时必须重跑本 Gate。
- Stage 3 依赖本 Gate 保证 probe 路径验证后能恢复训练状态；Stage 4 及以后依赖本 Gate 进行长训练恢复。
