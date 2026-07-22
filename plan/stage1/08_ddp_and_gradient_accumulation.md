# S1.8 DDP、梯度累积与 `no_sync` 一致性

## 1. 目的

证明在同一 global batch、同一参数状态和同一随机状态下，单卡 full-batch 与真实四卡得到相同的全局 mean loss/gradient/raw/update；并证明在保持同一组 \(M\ge2\) 个 microbatch 统计单元时，单卡累积、不同 rank 分配和真实四卡 NCCL/DDP 得到相同的充分统计量、U 与累计状态。

历史 CPU Gloo 多进程或旧 NCCL smoke 不能替代本子任务；必须使用当前健康 A100 和真实 DDP/`no_sync` 路径。

## 2. 前置 Gate

- `G1-SINGLE` 已通过。
- 四张候选 GPU 分别通过本次 CUDA 上下文、最小分配和健康检查。
- 四卡组合通过本次 NCCL collective smoke。
- 没有使用当前已确认故障或不可枚举的 GPU。
- 后台 Pile 下载状态已记录；运行规模不会显著争抢同一大盘吞吐。
- 缓存、torchrun rendezvous/log 和临时目录已解析到 Stage 1 的 `$DATA_ROOT` 项目路径；启动前重新检查外部占用，并为四个 GPU UUID 取得带 heartbeat 的本次运行租约。
- 资源所有者/管理员已确认本次四卡使用；项目租约只防本项目并发，不代表集群所有权。发现未知外部占用只退出并报告。

## 3. 固定对照矩阵

至少执行以下路由，全部使用相同 global samples：

| 路由 | world size | 每 rank local backward 数 | 用途 |
|---|---:|---:|---|
| A | 1 | 1 个完整 global batch | loss、mean gradient 与更新参考；不计算 U |
| B | 1 | `M` 个 microbatch | 单卡累积参考 |
| C | 2 | `M/2` 个 microbatch | 中间归约诊断，条件允许时执行 |
| D | 4 | `M/4` 个 microbatch | 正式多卡 Gate |

`M` 必须能被正式路由合理切分。B、C、D 必须保留完全相同的 microbatch 边界、每个单元样本和全局 `M`，只改变单元到 rank 的分配；改变 microbatch 边界的配置只能作为另一组实验，不能做有限样本 U 相等 Gate。若测试不等有效 token 数，使用加权充分统计量而不是人为补成等权。

## 4. 执行步骤

### 4.1 冻结全局样本规划

1. 固定 global batch 的样本 ID 和顺序。
2. 固定每个样本的 token hash。
3. 为每条路由生成 rank/microbatch 映射。
4. 验证各 rank 合集恰好重构 global batch。
5. 验证没有未记录的重复或遗漏。
6. 保存每个 microbatch 的有效 token 数。
7. 固定每条路由的 RNG 派生规则。
8. 在精确等价 Gate 中关闭随机层或固定一致随机状态。

### 4.2 验证未同步 local gradient

1. 用 DDP 包装真实模型。
2. 在每次 local backward 时使用 `no_sync` 或等价的明确未同步路径，并确保上下文同时包住该 microbatch 的 forward 与 backward。
3. 在任何普通 DDP all-reduce 前读取 local mean gradient。
4. 每次读取后清理临时 `.grad`，避免下一 microbatch 累积污染。
5. 为每个 local backward 记录 rank、序号、样本 ID 和 gradient checksum。
6. 增加同步状态断言或可观察标志。
7. 构造一个故意普通同步的负对照，验证 guard 能发现独立统计单元已丢失。
8. 不允许负对照进入正式结果目录。
9. 记录实际 backend=`nccl`、DDP 包装类型、world size 和 rank 到 GPU UUID 映射。
10. 记录 local backward 期间 DDP gradient collective 计数为零，并核对手工充分统计量 collective 的类型与次数。

### 4.3 跨 rank 聚合等权充分统计量

1. 每 rank 本地累计 `S1_local`。
2. 每 rank 本地累计 `S2_local`。
3. 每 rank 精确累计 `M_local`。
4. 每 rank 累计 loss numerator 与有效 token count。
5. 对 `S1/S2/M`、loss numerator 和有效 token count 执行 sum all-reduce。
6. 验证所有 rank 获得相同的 global 充分统计量与 loss 计数。
7. 只计算一次 `global_mean_loss = global_loss_numerator/global_valid_token_count`。
8. 计算 `mean_grad = S1/M`。
9. 把相同 mean gradient 设置给各 rank optimizer。
10. 计算 raw 和 U。
11. 记录 global checksum 与逐张量误差。

### 4.4 跨 rank 聚合加权充分统计量

1. 每 rank 累计 `G1_local = sum(b_m*g_m)`。
2. 每 rank 累计 `G2_local = sum(b_m^2*g_m^2)`。
3. 每 rank 累计 `N1_local` 与 `N2_local`。
4. 每 rank 累计 loss numerator 与有效 token count。
5. 对四个梯度充分统计量、loss numerator 和有效 token count 执行 sum all-reduce。
6. 验证所有 rank 元数据一致。
7. 只计算一次 `global_mean_loss = global_loss_numerator/global_valid_token_count`，禁止平均 rank-local mean loss。
8. 计算加权 global mean gradient。
9. 计算加权 U-statistic。
10. mean gradient 与单卡逐 token mean oracle 比较；weighted U 与显式 weighted cross-microbatch ordered-pair oracle 比较。

### 4.5 比较每条路由

1. 在 A、B、C、D 间比较 mean loss。
2. 在 A、B、C、D 间比较 global mean gradient。
3. 在 A、B、C、D 间比较 raw 与 clip factor。
4. 在 A、B、C、D 间比较 optimizer step 后参数和 optimizer state。
5. 仅在保持同一组 `M` 个统计单元的 B、C、D 间比较每个参数张量的 `S1`。
6. 仅在 B、C、D 间比较每个参数张量的 `S2`。
7. 仅在 B、C、D 间比较未裁剪 U 与裁剪后 U。
8. 仅在 B、C、D 间比较 signed/positive/negative mass/absolute。
9. 在全部适用路由间比较 data movement 与 magnitude。
10. 记录每项的最差参数张量和最差坐标。

### 4.6 验证排列与 rank 不变性

1. 保持 global batch 不变，交换 rank 分片。
2. 保持 rank 分片不变，交换 local microbatch 顺序。
3. 重复等权路径。
4. 重复加权路径。
5. 验证最终充分统计量和分数不变。
6. 不要求会影响浮点归约顺序的结果 bitwise 相同，使用 `T32_DISTRIBUTED`。

### 4.7 失败清理

1. 注入一个 rank 的受控失败。
2. 验证所有 rank 在有限时间内退出。
3. 验证结果目录没有成功标记。
4. 验证失败信息包含 rank 与阶段。
5. 清理前同时核对 manifest 中的 UID、PID/PGID、启动时间、父子树/launcher 和 run token；全部吻合时才处理并释放租约。
6. 任一进程指纹不匹配时只报告，不发送信号。
7. 禁止 broad `pkill`、按进程名清理或触碰其他用户进程和 GPU context。
8. 验证下一次独立运行可以重新取得同一候选卡。

## 5. 产出

- 本次 GPU UUID 健康与 NCCL 资格清单。
- global sample 到 rank/microbatch 的映射表。
- 1/2/4 卡路由配置与解析后配置。
- 每条路由的 global loss numerator/count，以及适用路由的 `S1/S2` 或 `G1/G2/N1/N2` 摘要。
- backend、DDP 包装、rank→GPU UUID 和 collective 计数证据。
- 资源确认、外部占用复查和带进程指纹的租约生命周期记录。
- 逐参数张量的 loss、gradient、raw、U、clip、更新和累计对照表。
- rank 一致性 checksum 表。
- 普通同步负对照与受控 rank failure 报告。
- `G1-DDP` 机器可读报告。

## 6. 可视化

- 路由 A 与 D 的 mean gradient/raw 散点图，以及路由 B 与 D 的 U 散点图；不得把 A 标成 U 参考。
- `parameter tensor × route` 的最大尺度化误差热力图。
- 不同 world size/accumulation 切分的 normalized L2 误差曲线。
- 各 rank local gradient norm 与有效 token 数诊断图。
- 更新后参数的最差张量误差条形图。

## 7. 核验标准

- 全部路由的 global sample ID、多重集、token hash 和有效 token 总数完全一致。
- DDP 统计读取点位于普通 gradient all-reduce 之前；故意同步负对照必须被 guard 拒绝。
- all-reduce 后所有 rank 的 loss numerator、有效计数、统计单元数和 registry hash 完全一致；global mean loss 由全局 numerator/count 只除一次。
- A↔D 的每个参数张量仅在 mean gradient、raw、clip、更新后参数和适用基线上通过 `T32_DISTRIBUTED`。
- B↔C↔D 保持同一组 `M` 个统计单元，并在 `S1/S2` 或 `G1/G2/N1/N2`、mean gradient、U、clip 和累计状态上通过 `T32_DISTRIBUTED`。
- 正式证据确认 backend 为 NCCL、模型由 DDP 包装、world size 为 4、rank→GPU UUID 完整，local backward 期间没有普通 DDP gradient collective。
- 不能只凭 loss 或全局 checksum 相同通过 Gate。
- rank 和 microbatch 排列变化后通过 `T32_DISTRIBUTED`。
- 受控失败不留下成功标记、半发布 checkpoint、本次 PID/PGID 或本项目租约；未处理任何非本次运行资源。

## 8. Gate 与后续依赖

- `G1-DDP` 通过是 `G1-EXIT` 的硬前置；当前只有单卡结果时不得把 Stage 1 标为完成。
- DDP 包装方式、no_sync 边界、world-size reduction 或 gradient accumulation 逻辑变化时必须重跑本 Gate。
- 后续正式实验可改变 world size，但必须先在相同 global batch 上复用本 Gate 的等价性测试。
