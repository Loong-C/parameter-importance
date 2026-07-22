# S1.7 Pythia-14M 单卡真实链路验证

## 1. 目的

在真实 Transformers/Pythia 模型、真实 tokenizer 和固定 Pile token fixture 上验证 Stage 1 模块的接口、参数 registry、显存生命周期和产物 schema。该子任务仍只验证代码正确性，不解释参数分布或训练现象。

## 2. 前置 Gate

- `G1-ENTRY`、`G1-REGISTRY`、`G1-ORACLE`、`G1-GRAD`、`G1-EST` 和 `G1-STEP` 已通过。
- Pythia-14M step0 的 revision、文件哈希和离线加载已在本次运行前重新验收。
- 至少一张候选 A100 通过按 UUID 的最小 CUDA 分配检查。
- 调试样本全部位于已验证 Pile 前缀范围内。
- Stage 1 缓存与临时目录已解析到 `$DATA_ROOT/cache` 和 `$DATA_ROOT/tmp/stage1/<run_id>/`；启动前重新检查外部占用并为选定 GPU UUID 取得带 heartbeat 的本次运行租约。项目租约不代表集群所有权。
- 资源所有者/管理员已确认本次用卡；若发现未知外部占用，只退出并报告，不终止对方进程。
- 执行代码、配置和数学契约已经提交并完成三端同一 HEAD 同步；dirty executable diff 不计入正式 Gate。

## 3. 执行步骤

### 3.1 冻结真实 fixture

1. 记录 14M 模型绝对路径、revision 和 manifest hash。
2. 记录 tokenizer 文件集合与 hash。
3. 固定 Pile 样本 ID 和顺序。
4. 保存输入 token 数组 hash。
5. 固定 global batch、microbatch 切分和有效 token 数。
6. 固定模型 train/eval 状态和随机层策略。
7. 固定 FP32 correctness 配置。
8. 固定 optimizer、参数组、学习率和 clip 配置。
9. 生成带 schema/version 和内容 hash 的 fixture manifest，并通过身份校验逻辑冻结；不依赖修改文件权限来表达“只读”。

### 3.2 验证 provider 与参数 registry

1. 在 import/provider 初始化前启用 HF/Transformers/Datasets offline 模式与本地文件限定，并加载 config、tokenizer 和模型。
2. 验证模型结构、参数量和 vocab size 与 manifest 一致。
3. 构建参数 registry。
4. 核对 registry `numel` 与 eligible 参数实际数量。
5. 检查共享权重和别名处理。
6. 保存 layer/module 标签汇总。
7. 保存并重载 registry，比较 hash。
8. 验证模型 reload 后 registry 不漂移。
9. 启用进程级 socket/HTTP guard，验证 provider 初始化、reload 和前向均未发起网络连接。

### 3.3 固定状态梯度验证

1. 在 step0 参数状态冻结模型。
2. 摘要保存模型参数、buffer、optimizer/scheduler 和 CPU/GPU RNG 状态。
3. 用一次完整 batch backward 计算 FP32 参考 gradient。
4. 用固定 microbatch 切分逐次计算 local mean gradient。
5. 保存每个 microbatch 的完整参数梯度，仅覆盖一个指定 debug step。
6. 从保存数组离线重构 mean gradient。
7. 比较完整 batch、在线重构和离线重构。
8. 计算 raw、double、显式 U 和流式 U。
9. 用独立离线 oracle 复算 estimator。
10. 记录逐张量误差和最差坐标。
11. 再次摘要模型参数、buffer、optimizer/scheduler 和 RNG 状态。
12. 验证固定状态计算前后全部摘要未改变。

### 3.4 小步训练验证

1. 从同一 step0 状态启动统计关闭参考运行。
2. 启动统计开启待测运行。
3. 使用完全相同的少量固定 global batches。
4. 逐步保存 loss、mean gradient norm、clip factor 和更新 norm。
5. 逐步保存 raw、U 和四类累计量摘要。
6. 保存 data movement、weight-decay movement 和 magnitude 摘要。
7. 比较两条训练轨迹。
8. 在最终 step 保存参数级累计数组。
9. 验证所有长期数组可按 registry 离线读取。
10. 删除或复用每步临时 `S1/S2`，避免无界显存增长。

### 3.5 资源与失败行为

1. 记录本次 Stage 1 进程的峰值 GPU 显存。
2. 记录本次 run 目录写入量和进程级主机内存，不把后台下载流量归因于本运行。
3. 记录本次进程的逐 microbatch backward 与统计聚合耗时。
4. 记录完整 gradient dump 的大小和生成时长。
5. 验证异常退出产生明确失败标记，不产生成功标记。
6. 验证失败不会覆盖已有通过运行。
7. 清理前同时核对运行 manifest 中的 UID、PID/PGID、启动时间、父子树/launcher 和 run token；全部吻合时才处理并释放租约。
8. 任一进程指纹不匹配时只报告，不发送信号。
9. 禁止用 broad `pkill`、按进程名清理或触碰其他用户的 GPU context。

## 4. 产出

- Pythia-14M Stage 1 fixture manifest。
- 参数 registry 和 layer/module 汇总。
- 指定 debug step 的完整逐 microbatch 梯度。
- full-batch、在线重构与离线重构的误差表。
- raw/double/显式 U/流式 U 的逐参数结果与摘要。
- 少量 step 的统计开启/关闭对照运行。
- 峰值显存、内存、wall-clock 和产物大小报告。
- offline/socket guard 与运行租约生命周期报告。
- `G1-SINGLE` 机器可读报告。

## 5. 可视化

- full-batch 与 microbatch 重构 gradient 的散点图。
- 显式 U 与流式 U 的散点图。
- layer/module × metric 的尺度化误差热力图。
- 统计开启/关闭逐步参数误差曲线。
- 临时与长期状态的显存占用时间线。

## 6. 核验标准

- 所有资产与 fixture 身份字段精确匹配；Stage 1 进程/provider 发起的网络访问次数为零，不以主机上其他已记录任务的网络活动判定本 Gate。
- registry hash 在 reload 前后完全一致。
- full-batch、在线重构和离线重构的 FP32 gradient 通过 `T32_SINGLE`。
- estimator 与独立离线 oracle 通过 `T32_SINGLE`。
- 固定状态计算前后的模型参数、buffer、optimizer/scheduler 和 RNG 摘要完全一致。
- 所有 loss、gradient、充分统计量、分数和参数更新均有限。
- 统计开启/关闭的 loss、optimizer state 和参数轨迹通过 `T32_SINGLE`；确定性路径若可 bitwise 相同则额外记录。
- 临时状态不随 step 数线性累积；指定 gradient dump 之外不长期保存每步完整 gradient。
- 运行目录、manifest、失败/成功标记和大型数组 hash 完整。

## 7. Gate 与后续依赖

- `G1-SINGLE` 通过后，S1.8 与 S1.9 才能进入真实 GPU 验收。
- 若 14M provider、Transformers 版本、模型 revision、eligible 参数或 loss adapter 变化，必须重跑本 Gate。
- Stage 2 可复用该固定 checkpoint/fixture，但不得把本子任务的少量 step 当作偏差或稳定性结论。
