# S3.3 更新端点、固定 probe 与无副作用状态管线

## 目的

建立一条能从真实 optimizer step 生成可重放端点、为端点绑定独立固定 probe、并在积分分析后完整恢复状态的管线。这个子任务确保后续比较的差异只来自求积规则，而不是端点漂移、batch 变化、随机层变化或分析过程污染训练状态。

## 当前依据

- 当前服务器 `runs/`、`checkpoints/` 和 `results/` 已清场，没有可直接使用的现行 Stage 3 端点。
- 旧归档的 checkpoint 和旧源码已被有意排除，不能用归档 CSV 反推出当前端点。
- Stage 0 总览要求进入 Stage 3 前，checkpoint 能唯一关联更新前后状态、optimizer 状态和数据位置。
- 数学规格要求全部节点使用同一个固定损失，并在分析后恢复原模型与 optimizer state。

## 前置条件

- G3-0 与 G3-1 通过。
- Stage 1 已提供稳定参数注册表和损失实现。
- Stage 0 的 checkpoint、原子写入、运行状态和恢复机制通过。
- 14M 资产与选定 Pile 范围 ready；31M 只在其资产 gate 通过后启用。

## 实施步骤

### 1. 设计端点记录 schema

1. 为每个端点对分配唯一 `path_state_id`。
2. 记录来源 run ID、模型资产 ID、训练 seed 和 optimizer step。
3. 记录更新前 checkpoint 身份。
4. 分别记录 optimizer 更新后 `parameter_post_state` 与 scheduler/scaler/RNG
   完整提交后的 `attempt_commit_state` 身份；不得共用含混的“更新后”字段。
5. 记录参数注册表摘要和参数总数。
6. 记录 optimizer、scheduler、GradScaler、RNG 和数据游标身份。
7. 记录训练 batch 的不可逆样本/序列标识摘要。
8. 记录学习率、参数组、梯度裁剪阈值和实际裁剪因子。
9. 记录更新前梯度范数、总更新范数和各参数组更新范数。
10. 记录裁剪后梯度分片的 artifact ID、参数映射、摘要和完整性状态。
11. 记录 weight decay 配置，但不把配置值当作已验证的位移分解。
12. 记录训练阶段标签和预注册选择层。
13. 为 schema 设置版本号和必填字段校验。

### 2. 捕获真实更新前状态

1. 在 optimizer step 前确认梯度已经完成 Stage 1 规定的 unscale、聚合和裁剪流程。
2. 保存参与研究的可学习参数。
3. 保存每个参与参数的裁剪后梯度张量，并显式记录 `.grad is None` 的参数。
4. 校验梯度名称、形状、dtype、参数组归属和逐分片哈希。
5. 将“恢复裁剪后梯度并只重放更新 transition”冻结为规范重放方案；规范重放不重新执行 forward/backward。
6. 保存模型 buffer，并把它登记为该端点对唯一的 `probe_buffer_snapshot` 候选。
7. 保存 optimizer state。
8. 保存 scheduler 和 GradScaler state。
9. 保存 Python、NumPy、CPU Torch 和 CUDA RNG state。
10. 保存模型 train/eval 模式和每个随机层状态。
11. 保存当前数据游标、sampler 状态和 batch 身份。
12. 对保存内容生成清单和哈希。
13. 在参数、梯度和所有必填状态落盘并校验前，不发布 pre-state 完成标记。

### 3. 分别捕获参数更新后状态与 attempt 提交状态

1. 只执行一次目标 optimizer step。
2. optimizer 返回后立即捕获 `parameter_post_state`：保存与 pre-state 相同范围的
   参数、buffer 和 optimizer state；此时 scheduler/scaler/RNG 尚未声明提交完成。
3. 按训练合同推进 scheduler、GradScaler、RNG 和数据游标，随后捕获
   `attempt_commit_state`，并把它与前述 `parameter_post_state` hash 绑定。
4. 恢复入口只有在 `attempt_commit_state` 权威 commit 存在时才把该 step 视为完整；
   仅有 `parameter_post_state` 时必须进入 reconciliation，不得自动重复 optimizer step。
5. 保存两个状态各自的 schema、hash 和事件序号。
6. 计算逐参数总位移 \(\Delta\Theta_t\)。
7. 验证 pre-state 加总位移能逐张量重构 `parameter_post_state`。
8. 逐项比较 pre/post buffer，并要求影响 probe 前向的 buffer 完全相同。
9. 将通过比较的共同 buffer 摘要冻结为唯一 `probe_buffer_snapshot`。
10. buffer 不一致时拒绝参数路径主合同；只有另行预注册扩展状态路径后才能作为不同研究对象处理。
11. 检查 frozen 参数是否保持不变。
12. 检查任何意外新增、丢失或改名的参数。
13. 对 `parameter_post_state`、`attempt_commit_state` 和位移清单分别生成哈希。
14. 只在 pre/parameter-post/attempt-commit/位移/共同 buffer 全部一致时，通过
    “不可变对象 + 独立权威 commit”两阶段协议发布端点对。

### 4. 处理 full-update 与 data-only 路径

1. 将实际 pre/post 差定义为 `full_update_delta`。
2. 以 full-update 线性路径作为所有正式端点的必需主路径。
3. 若 optimizer instrumentation 能给出数据驱动位移，单独保存 `data_update_delta`。
4. 若能精确重建解耦 weight decay 位移，单独保存 `weight_decay_delta`。
5. 验证位移分解与实际总位移在预定浮点容差内一致。
6. 为 data-only 构造反事实终点并单独计算该终点损失。
7. 不用真实 post-state 损失检查 data-only 路径完备性。
8. 分解验证失败时禁用 data-only，不影响 full-update 主路径继续。
9. 不通过配置中的 weight decay 系数事后猜测 fused optimizer 的精确逐参数位移。

### 5. 建立固定 probe panel

1. 从已验收数据范围中预先分配主 probe 区间。
2. 分配至少两个额外独立 probe panel 用于抽样稳健性。
3. 为 pilot 和 formal 使用不同的 probe 区间。
4. 为重放审计保留一个未参与方法选择的 probe 区间。
5. 记录每个 panel 的样本/序列数量。
6. 记录 token IDs、labels、attention mask 和有效 token 计数摘要。
7. 证明每个 panel 与对应 update batch 不重叠。
8. 证明不同 panel 之间不重叠。
9. 证明所有记录都落在 ready shard 中。
10. 为 panel 内容、顺序、预处理和 tokenizer 身份生成 manifest 与哈希。
11. 禁止运行时重新随机抽取或自动补样本。

### 6. 固定 probe 损失求值

1. 从 manifest 加载固定 probe 内容。
2. 将模型切换到数学合同规定的确定性模式。
3. 加载端点对冻结的唯一 `probe_buffer_snapshot`。
4. 在 pre、post 和所有插值节点保持该 buffer 完全不变。
5. 关闭 autocast 和 GradScaler 对分析梯度的影响。
6. 以 FP32 作为真实模型主求值精度。
7. 对选定审计状态支持 FP64 模型副本求值。
8. 若 probe 必须分 microbatch，记录每个 microbatch 的有效 token 数。
9. 按全 probe 有效 token 权重合并损失和梯度。
10. 验证不同等价 microbatch 切分得到相同总损失和平均梯度。
11. 重复求值同一端点，验证 loss 和梯度在容差内一致。
12. 检查所有 loss 与梯度有限。

### 7. 实现只读路径分析上下文

1. 优先使用独立分析模型副本或无状态参数调用。
2. 不在仍将继续训练的模型实例上反复覆盖参数。
3. 加载端点前记录分析模型的参数、buffer、模式和 RNG 摘要。
4. 每个节点求值前清空临时梯度。
5. 节点求值后释放不再需要的计算图和梯度。
6. 禁止 optimizer、scheduler 或数据 sampler 在分析中 step。
7. 分析完成后恢复模型、buffer、模式和 RNG。
8. 比较分析前后全部状态摘要。
9. 状态不一致时把整个 `path_state_id × probe_id` 单元标记为失败。
10. 失败单元不得发布成功标记或进入聚合。

### 8. 验证端点重放

1. 从 pre-state 启动一个全新进程或全新模型实例。
2. 恢复 optimizer、scheduler、GradScaler、RNG 和数据游标。
3. 恢复已保存的裁剪后梯度及 `.grad is None` 映射。
4. 校验恢复梯度的名称、形状、dtype、参数组和摘要。
5. 只重放记录的 optimizer transition，以及 Stage 1 明确定义为该 transition 一部分的 GradScaler/scheduler 动作。
6. 明确禁止在规范重放中重新执行 forward/backward；update batch 仅通过保存的身份摘要核验来源。
7. 比较重放 post-state 与原捕获 post-state。
8. 比较学习率、裁剪因子、梯度范数和更新范数。
9. 比较 optimizer、scheduler、GradScaler、RNG 和数据游标的 post-state。
10. 记录逐张量最大差异和是否通过。
11. 只有重放通过的端点才进入正式积分。

### 9. 定义原子单元与恢复

1. 将一个 `path_state_id × path_type × probe_id` 定义为最小可恢复单元。
2. 单元启动时保存 resolved config 和输入哈希。
3. 节点结果写入单元专属临时目录。
4. 所有节点、聚合和状态恢复检查完成后再原子发布。
5. 中断时保留最后完整节点与失败原因。
6. 恢复时重新校验端点、probe 和配置摘要。
7. 摘要不一致时拒绝续跑并创建新实验 ID。
8. 已完成单元只读，不允许静默覆盖。

## 产出

- 端点对与位移 schema；
- 含裁剪后梯度分片的 pre/post 状态捕获与原子发布机制；
- 唯一 `probe_buffer_snapshot`、pre/post buffer 等同性报告与失败协议；
- full-update 主路径及可选 data-only 位移记录；
- 固定独立 probe panel 与 manifest；
- 固定损失和有效 token 加权实现；
- 分析状态无副作用测试；
- 单步端点重放报告；
- 原子单元、失败记录和恢复协议。

## 核验标准

- 每个端点对能唯一追溯到 run、提交、模型、数据、step、seed 和 optimizer state。
- pre-state 加位移能重构 post-state，参数集合和顺序不漂移。
- 裁剪后梯度可按参数注册表完整恢复，规范重放无需重新执行 forward/backward。
- pre/post buffer 相同，两个端点和所有路径节点使用同一份 `probe_buffer_snapshot`。
- update batch 与所有主 probe 在统计单元层面无重叠。
- 同一 probe 在所有节点使用完全相同的输入、mask、损失归约和随机状态。
- 同一端点重复 probe 求值得到相同 loss 和梯度。
- 等价 probe microbatch 切分得到一致的全局 token 加权结果。
- 分析前后参数、buffer、optimizer、scheduler、RNG、模式和数据游标保持不变。
- 全新进程通过恢复裁剪后梯度并只重放 update transition，能恢复同一 post-state。
- 任何失败或中断单元不会被聚合器视为成功。

## Gate 与后续依赖

- 本子任务通过形成 **G3-2 状态与 probe gate**。
- G3-2 与 G3-3 都通过后，才能建立真实模型参考积分。
- data-only 分解未通过只阻断 data-only 次要分析；full-update 主路径仍可在其他 gate 通过后继续。
- 端点重放失败时，所有基于该端点的数值结果作废并回到 Stage 0/1 的状态恢复问题处理。

## 失败与恢复

- 若保存体积过大，按端点分片并保留完整 manifest；不得只保存少量参数来代替全参数完备性。
- 若分析模型 OOM，减少同时驻留的端点或节点，不减少参数范围。
- 若 probe 太大，分 microbatch 并按有效 token 加权，不在不同节点使用不同子集。
- 若随机层无法固定，先关闭或改造确定性求值；不能用跨节点随机均值掩盖目标函数变化。
