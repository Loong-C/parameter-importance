# Stage 0 → Stage 1 交接说明

本文件定义 Stage 1 的静态进入边界；正式环境 ID、资产 ID、数值容差、证据路径和最终 G10 状态由 `DATA_ROOT` 下不可变的 `stage0-g10-stage1-handoff-v1` 与 `stage0-g10-readiness-v1` 给出。不得凭本文件中的名称猜测运行身份。

相关依据为 [Stage 0 总计划](../plan/Stage0/README.md)、[S0.11 分层测试与重放](../plan/Stage0/11_test_quality_and_replay.md) 和 [S0.12 最终交付](../plan/Stage0/12_delivery_and_sync.md)。

## 可直接复用

- G2 锁定环境、G3 `ready` 资产 manifest、G4 resolved config / run identity / seed / provenance；
- Pythia 14M 正式模板、固定微型 token fixture、单卡与四卡参考报告；
- 已验证的 loss numerator / effective count 全局归约、梯度累积、`no_sync`、裁剪、optimizer 与 scheduler 边界；
- canonical JSONL、TensorBoard 派生视图、checkpoint group commit、恢复与保留策略；
- G8 容量预检、GPU UUID 租约与故障处置；
- G9 fresh-process 单卡/四卡/恢复离线重放报告。

正式执行必须从 G10 handoff 中读取环境 hash、manifest ref、asset ID、GPU UUID 和允许的数值容差；不得使用物理 GPU 序号、缓存目录名称或旧 run 目录推断身份。

## Stage 1 仍须证明

Stage 0 没有实现或证明任何参数重要性公式。Stage 1 必须使用独立 oracle 证明参数 registry、loss/gradient 尺度、raw、equal-U、double estimator、训练 step 累计器和 checkpoint 中的重要性状态均正确。`fixtures/stage0/deterministic-training-v1.json` 只证明确定性训练基础设施，不能充当 estimator oracle。

## 风险与失效条件

- Stage 0 大资产只有服务器大盘上的一个权威运行副本，不等同于备份；G1-D 风险接受到期、Stage 4 启动、存储拓扑变化或磁盘异常时立即失效。
- 源码提交、锁文件、环境、GPU 拓扑、资产 manifest、训练语义、checkpoint 或任一 Gate 证据变化时，必须按回归矩阵重跑相应层级，旧 READY 不得复用。
- Stage 1 只能在 G10 为 `PASS` 且新 readiness 为 `READY` 时进入；不存在“本地通过”替代正式服务器证据的路径。
