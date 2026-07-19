# S0.7 四卡 DDP、梯度累积与 `no_sync`

## 目的

证明四卡运行不仅“能够启动”，而且设备映射、NCCL 集合通信、数据分片、loss reduction、梯度平均、梯度累积和 `no_sync` 的数值语义符合 Stage 1 参数重要性计算的要求。

## 当前依据

- 旧环境曾有四卡 NCCL BF16 all-reduce 通过记录，但当前 GPU 枚举、内核和 ECC 状态已变化，旧记录失效。
- 当前拓扑快照中 GPU 3–6 位于同一 NUMA 节点，但只能在硬件问题解除后重新评估。
- A100 之间当前未显示活动 NVLink，不能假设四卡通信性能等同于 NVSwitch 节点。
- 后续 Stage 1 明确需要单卡/多卡一致性和 microbatch 梯度尺度验证。

## 前置条件

- G0-G 与 G5 单卡 gate 通过。
- PCI/驱动、NVML、PyTorch 枚举一致，候选四卡健康且空闲。
- 四卡列表由运行配置显式提供，代码不硬编码物理编号。
- 候选四卡的拓扑和 NUMA 关系已记录。
- `README.md` 4.1 节的统一测量协议版本已冻结并写入 provenance。

## 实施步骤

### 1. 建立四卡启动前检查

1. 验证配置中恰好有四个不同设备。
2. 将物理 index、PCI bus、GPU UUID 与进程内 local rank 映射写入 provenance。
3. 确认四卡无外部计算进程、无活动 MIG、温度和 ECC 状态符合 gate。
4. 优先选择同一 NUMA/PCIe 组；若只能跨 NUMA，必须显式记录并重新建立性能基线。
5. 检查通信端口、进程超时和文件描述符预算。
6. 任何检查失败时在创建进程组前退出。

### 2. 验证进程组与设备映射

1. 启动四个 rank，每个 rank 绑定唯一逻辑设备。
2. 每个 rank 记录 world size、global rank、local rank 和设备映射。
3. 验证所有 rank 使用同一 resolved config、environment ID 和模型初始化摘要。
4. 设置有限的初始化和 collective 超时，禁止无限等待。
5. 正常结束时显式销毁进程组。

### 3. 验证 NCCL collective

1. 对标量执行 all-reduce，并核对预期和。
2. 对多个张量规模执行 all-reduce，覆盖小消息和训练相关大消息。
3. 验证 broadcast、all-gather 和 barrier 等训练所需操作。
4. 每个消息规模按统一协议执行 20 次 warmup 和 50 次测量，记录中位延迟、p95、吞吐和波动。
5. 重建进程组三次并完成三个独立重复，确认没有 hang、超时或 communicator 残留。
6. 将三个健康重复的聚合结果设为当前拓扑性能基线；后续较短的 smoke 低于基线 70% 只产生告警并触发完整统一协议复验，只有完整复验仍低于 70% 才阻断并调查。

### 4. 验证数据分片

1. 为全局 batch 中每个样本分配稳定 ID。
2. 记录每个 rank 实际读取的样本 ID 和顺序。
3. 核对四个 rank 的并集等于预期全局 batch。
4. 核对同一步内没有非预期重复或遗漏。
5. 验证 epoch/step 推进和 sampler seed 在重跑时可复现。
6. 对不能整除 world size 的配置在启动前拒绝，除非显式采用并验证 padding/drop 规则。

### 5. 验证单卡与四卡等价

1. 使用相同模型初始状态、固定 FP32 数据和关闭 dropout 的配置。
2. 构造相同有效 global batch 的单卡基准 step。
3. 构造四卡 DDP step，每卡处理对应分片。
4. 为一个 optimizer window 分别累计 loss numerator 和有效样本/token denominator，并用单进程完整 batch 结果作为 oracle。
5. 对默认 DDP“跨 rank 平均梯度”语义，令 world size 为 `W`、全局有效计数为 `T`、rank/microbatch 的 loss sum 为 `S[r,m]`；本地 backward 标量按 `W × S[r,m] / T` 缩放，全局展示 loss 单独按 `sum(S) / T` 计算。
6. 构造各 rank 有效 token 数不同、包含 `ignore_index` 的 fixture，验证上述缩放而不是只测等分 batch。
7. 比较全局 loss、每个参数梯度摘要和 optimizer step 后参数。
8. 对零梯度、共享参数和 tied weights 单独核对。
9. 保存差异最大的张量名、绝对误差和相对误差。

### 6. 验证梯度累积

1. 建立一次性处理完整有效 batch 的参考 step。
2. 将同一 batch 按确定顺序拆成多个 microbatch，并包含 microbatch 大小/有效 token 数不等的 fixture。
3. 在每个 microbatch 后累积梯度，只在边界执行 optimizer step。
4. 使用整个 optimizer window 的全局 numerator/denominator 缩放每个 microbatch，不按局部 mean 再做等权平均。
5. 比较累积后的梯度、参数更新和全局 loss 与参考 step。
6. 分别验证单卡累积和四卡累积。
7. 对不足一个 accumulation window 的尾 batch，使用实际 microbatch 和有效计数重新计算分母与缩放；若实现选择丢弃尾窗，则在配置解析时明确拒绝/丢弃并验证样本计数。

### 7. 验证 `no_sync`

1. 在四卡累积的非末尾 microbatch 用 `no_sync` 同时包住 forward 和 backward，不能只包 backward。
2. 在最后一个 microbatch 恢复同步并完成该次 forward/backward。
3. 通过通信 hook 或等价计数证明中间 microbatch 没有发生梯度 bucket all-reduce；参数/buffer broadcast、barrier 等控制通信单独分类，不计作梯度同步。
4. 只有最后一次同步 backward 完成后，才允许 gradient clipping、optimizer step 和 scheduler step。
5. 比较启用和不启用 `no_sync` 的最终梯度与参数更新。
6. 记录通信次数、step 时间和显存差异。
7. 验证异常中止不会让下一次运行继承未完成的同步状态。

### 8. 验证 BF16 功能路径

1. 用 Pythia 14M 或 160M 的小 batch 执行四卡 BF16 前向、反向和更新。
2. 检查每个 rank 的 loss、梯度和更新均有限。
3. 检查各 rank step 数和样本数一致。
4. 记录峰值显存和吞吐。
5. BF16 差异只用于合理性诊断，不替代 FP32 语义 gate。

### 9. 验证失败与退出

1. 在一个 rank 主动注入受控异常。
2. 验证其他 rank 在超时内退出，而不是永久等待 collective。
3. 验证主进程返回非零状态并写入失败原因。
4. 验证所有子进程、GPU 上下文和项目锁被释放。
5. 验证下一次四卡 smoke test 可正常启动。

## 产出

- 四卡 DDP/NCCL smoke test；
- GPU 映射、拓扑和通信性能报告；
- 数据分片证据；
- 单卡/四卡 FP32 等价报告；
- 梯度累积和 `no_sync` 数值/通信报告；
- BF16 四卡功能报告；
- 受控失败与无残留进程报告。

## 核验标准

- 三次连续四卡 collective 测试均无 hang、NCCL 错误、ECC 增长或残留进程。
- 标量 collective 结果与预期完全一致；三个重复的性能样本均为正值，每个消息规模的 p95 不超过同一重复中位数的 2 倍，三个重复中位数的变异系数不超过 20%，否则必须诊断并重验。
- rank 与物理设备一一对应，四个 rank 的配置、模型初始化和环境身份一致。
- 每步样本 ID 并集精确等于全局 batch，重复/遗漏数为 0。
- 不等 rank token 数、不等 microbatch 和不完整尾窗 fixture 的全局 loss 与梯度均匹配单进程 numerator/denominator oracle。
- 单卡与四卡 FP32 的 loss、梯度和更新满足 `atol=1e-6, rtol=1e-5`，梯度整体相对 L2 误差不超过 `1e-5`。
- 完整 batch、普通累积和 `no_sync` 累积满足同一数值容差。
- 通信计数证明只有 accumulation window 的最后一个 forward/backward 执行梯度 bucket all-reduce，且 clipping/optimizer/scheduler 均发生在同步完成后。
- BF16 四卡路径所有数值有限，无 OOM，四个 rank step 和样本计数一致。
- 单 rank 故障后，所有 rank 在 60 秒内退出并释放资源；后续 smoke test 可重新启动。

## Gate 与后续依赖

- 本子任务通过形成 **G6 分布式语义 gate**。
- G6 是 Stage 1 单卡/多卡一致性和 microbatch 梯度验证的硬前置。
- 任一 GPU 枚举、ECC、数据分片、loss 分母或梯度尺度问题都必须阻断，不得以“训练 loss 看起来正常”放行。
