# S0.9 checkpoint 与断点续训

## 目的

建立完整、原子、可验证的训练状态保存与恢复机制，使 SSH 断开、进程失败或计划暂停后能够继续同一条训练轨迹，并为 Stage 3 的更新前后状态和 Stage 4–5 的长期训练提供可靠基础。

## 当前依据

- `checkpoints` 当前基本为空，可以从一开始采用统一格式。
- 旧归档中存在 checkpoint/恢复的历史经验，但当前 `main` 没有现行实现。
- 后续阶段需要关联参数更新前后状态、optimizer、scheduler、RNG、数据位置和在线重要性累计量。
- 四卡、梯度累积和 `no_sync` 使“何时可以安全保存”成为显式语义问题。

## 前置条件

- G4 配置/provenance 和 S0.8 的事件 schema/状态接口已可用；S0.8 不需要先完成恢复场景的最终验收。
- G5 单卡 gate 通过；四卡恢复验证还依赖 G6。
- G1 的 checkpoint 路径、容量和原子写入语义已验证。

## 实施步骤

### 1. 定义 checkpoint 状态合同

1. 保存模型参数和必要 buffer。
2. 保存 optimizer 完整状态。
3. 保存 scheduler 状态和当前学习率位置。
4. 保存混合精度 scaler 状态；使用纯 BF16 无 scaler 时显式记录为空。
5. 保存 global step、microstep、epoch、token/sample 计数。
6. 保存 Python、NumPy、PyTorch CPU 和各 rank CUDA RNG 状态。
7. 保存 sampler、shuffle、最后已提交 batch 和恢复后下一个全局 batch 的逻辑游标。
8. 保存或可确定性重建 worker RNG，并记录预取数据“不算已提交、恢复时丢弃”的策略。
9. 保存 resolved config 摘要、environment ID、模型/数据 manifest ID。
10. 保存日志 session、checkpoint 对应的 canonical event 序号和恢复所需状态。
11. 为后续在线重要性累计器提供版本化扩展状态接口。

### 2. 固定安全保存边界

1. 默认只在完整 optimizer step 结束后保存。
2. 在保存前确认没有未归约梯度和未完成的 accumulation window。
3. 若未来需要在 microstep 中间保存，必须额外保存梯度、microstep 和同步状态，并单独通过 gate。
4. Stage 3 的更新前/后快照使用明确的 snapshot 类型，不与普通恢复 checkpoint 混淆。
5. 每个 checkpoint 记录其对应的最后已完成 step 和恢复后的下一个 step。
6. 保存边界以“训练已消费并提交的 batch”为准；DataLoader 已预取但尚未提交的数据不推进 checkpoint 游标。

### 3. 定义多卡保存职责

1. 在当前原生、未分片 DDP 方案中，rank 0 负责共享模型、optimizer、scheduler 和主 manifest 的发布。
2. 每个 rank 提供自己的 RNG、sampler 和必要的局部状态。
3. 发布前比较各 rank 模型/optimizer 关键状态摘要，证明共享状态一致。
4. rank 0 在所有 rank 状态收齐后才发布完整标记。
5. 任一 rank 状态缺失时 checkpoint 保持不完整，不进入可恢复索引。
6. 保存期间设置有限超时，失败时让所有 rank 一致退出或继续使用上一个完整 checkpoint。
7. 一旦引入 ZeRO、FSDP 或分片 optimizer，本合同自动失效，必须重新定义 per-rank shard 保存和恢复 gate。

### 4. 实现原子发布

1. 为每次保存分配唯一 transaction ID 和单调 generation，并创建对象专属临时目录。
2. 各组件写入后分别记录大小和 SHA-256。
3. 生成 checkpoint manifest，写入最后 canonical event 指针、父 checkpoint 和 generation，并通过 schema。
4. 重新打开关键文件做最小加载检查。
5. 先原子发布不可变 checkpoint 对象；此时对象存在但尚不可被恢复入口选择。
6. 重新验证已发布对象后，再原子发布小型 `checkpoint_commit` 记录，将 checkpoint ID、manifest 摘要、canonical event 指针、父 lineage 和 generation 绑定；该记录是“可发现 checkpoint”的唯一权威来源。
7. `latest`、checkpoint typed event、canonical lineage 索引和 TensorBoard 标记都从已提交记录派生；它们不先于 commit 发布，也不作为 checkpoint 完整性的权威来源。
8. 启动和恢复前执行 reconciliation：验证每条 commit 指向完整对象，并从 commit 记录重建/修复缺失或陈旧的 `latest`、checkpoint event 与 lineage 派生视图。
9. 只有对象没有 commit 的保存视为未提交孤儿并从恢复入口忽略；只有派生 event/index 没有有效 commit 的记录视为无效引用并报告。
10. 中断留下的临时目录或孤儿对象只由精确清单和保留策略处理，不在 reconciliation 中静默删除。

### 5. 实现严格加载

1. 根据 checkpoint ID 查找，不接受模糊“最近修改时间”选择。
2. 只从权威 `checkpoint_commit` 集合查找，再验证对象完整标记、manifest、文件集合、大小和哈希。
3. 验证 checkpoint schema 版本和迁移规则。
4. 验证模型架构、配置摘要、optimizer 类型和数据身份兼容。
5. 验证 world size 变化是否被支持；Stage 0 默认要求恢复 world size 相同。
6. 在任何状态应用到活动模型前完成全部验证。
7. 只加载项目自己生成且通过 manifest/哈希的受信任 checkpoint；模型权重优先使用安全张量格式，optimizer 等必须使用对象序列化的状态只在该信任边界内加载。

### 6. 定义可精确恢复的数据流水线

1. 用 `(data_manifest_id, sampler_seed, epoch, committed_global_batch)` 唯一确定恢复后的下一个 batch。
2. sampler 和 shuffle 必须能从逻辑游标重建，不能依赖只存在于迭代器内部的位置。
3. 数据变换/worker 随机性优先按样本 ID、epoch 和 seed 无状态派生；不能无状态化的 worker 状态必须显式保存。
4. persistent worker 和预取缓冲在恢复时重建，checkpoint 后预取但未提交的数据全部丢弃。
5. 先用 `num_workers=0` 建立参考，再用正式 `num_workers>0`/prefetch 配置证明样本序列和最终状态等价。
6. 若生产数据管线无法满足上述条件，只能把 `num_workers=0` 作为可恢复正式配置；不能在未验证时宣称多 worker 精确恢复。

### 7. 验证单卡轨迹恢复

1. 使用确定性 FP32 极小配置运行固定总步数作为不中断基线。
2. 从同一初始化运行前半段并保存 checkpoint。
3. 在新进程中加载 checkpoint 完成后半段。
4. 比较样本 ID、loss、学习率、梯度摘要和最终参数。
5. 比较 global step、canonical 事件序号和 checkpoint 链。
6. 故意在 checkpoint K 后继续写若干 step 再崩溃，从 K 恢复并验证旧 session 的 K 后事件保留为 raw/orphaned，新 lineage 成为 canonical。
7. 重复一次从较早 checkpoint 恢复，确认不会污染原运行目录。

### 8. 验证四卡轨迹恢复

1. 使用通过 G6 的四卡确定性配置建立不中断基线。
2. 在安全 optimizer 边界保存四卡 checkpoint。
3. 结束全部进程，确认无残留后重新启动四卡运行。
4. 恢复各 rank RNG、sampler 和数据分片。
5. 完成剩余步骤并与不中断基线比较。
6. 验证恢复前后每步全局样本并集连续且不重复。
7. 若 resolved formal config 使用 `num_workers>0`/prefetch，则用该正式配置验证 checkpoint 游标不会被预取位置提前推进；若选择受支持的 `num_workers=0` 回退，则锁定该配置并把多 worker 精确恢复明确标为未支持，不能把未执行路径写成通过。

### 9. 验证损坏与中断写入

1. 分别在组件写入、对象发布、commit 发布、`latest` 更新、checkpoint event 和 lineage 派生视图更新边界注入中断。
2. 重新启动 reconciliation，验证对象已发布但未 commit 时不会被恢复入口选择。
3. 验证 commit 已发布但派生视图未完成时，能够从 commit 重建 `latest`、checkpoint event 和 lineage，且不会生成重复 event ID。
4. 验证任何 event、`latest` 或 lineage 都不会有效指向未提交/不完整 checkpoint；发现伪造引用时明确失败并保留诊断。
5. 验证 `latest` 在新 commit 未完成时仍指向上一个已提交 checkpoint。
6. 截断一个 checkpoint 文件，验证哈希检查拒绝加载。
7. 删除一个 rank 状态，验证完整性检查拒绝加载。
8. 修改配置或数据 manifest，验证不兼容恢复被拒绝。
9. 记录失败原因和可安全回退的 checkpoint ID。

### 10. 定义保留与索引

1. 保留最新完整 checkpoint、最佳验证 checkpoint 和阶段性里程碑 checkpoint。
2. 早期密集/后期稀疏保存由配置表达。
3. 删除前更新索引并确认目标不被活动 run 或派生实验引用。
4. 只按 checkpoint ID 精确删除，不按时间通配。
5. 对删除的 checkpoint 保留小型 tombstone，记录身份、原因和释放空间。

## 产出

- checkpoint schema、状态注册接口和版本规则；
- 单卡/四卡保存与严格加载实现；
- 原子发布、完整性 manifest 和 checkpoint 索引；
- checkpoint object/commit/event/lineage 的跨对象提交顺序与 reconciliation 规则；
- committed cursor、worker RNG/prefetch 和 canonical 日志 lineage 规范；
- 单卡与四卡不中断/恢复等价报告；
- 损坏、缺 rank、不兼容和中断写入拒绝测试；
- checkpoint 保留与清理规范。

## 核验标准

- 每个可恢复 checkpoint 都包含合同规定的状态、完整标记、文件大小和 SHA-256。
- 恢复入口永不选择临时、截断、缺 rank、哈希错误或 schema 不兼容的 checkpoint。
- 单卡和四卡恢复后的样本 ID、global step、学习率和 canonical 日志序列连续，无重复或跳过；raw session 日志允许保留带 session 身份的重放 step。
- `num_workers=0` 参考路径必须满足 committed cursor 和样本恰好一次语义；resolved formal config 若使用 `num_workers>0`/prefetch，则该路径也必须满足同一标准，否则正式配置必须锁定为 `num_workers=0` 并显式声明多 worker 精确恢复未获支持。
- 确定性 FP32 的不中断与恢复最终状态满足 `atol=1e-6, rtol=1e-5`；模型初始化和数据序列摘要完全相同。
- 在支持确定性算法的极小 fixture 上，优先要求关键状态字节或张量哈希完全一致；若只能数值一致，必须记录具体非确定性来源。
- 中断写入后，上一个完整 checkpoint 和索引保持可用。
- 每个可恢复 checkpoint 都有唯一有效 commit；commit 后的派生视图缺失可确定性重建，任何派生引用都不会指向未提交或不完整对象。
- 恢复运行不覆盖旧 session、旧 checkpoint 或旧 provenance。
- checkpoint 保存/加载时间、峰值内存和占用空间已进入容量报告。

## Gate 与后续依赖

- 本子任务与 S0.8 共同通过形成 **G7 可恢复性 gate**。
- Stage 3 只有在更新前/后 snapshot 与普通 checkpoint 语义清晰后才能开始。
- Stage 4–5 的长程训练只有在四卡恢复等价和容量 gate 同时通过后才能开始。
