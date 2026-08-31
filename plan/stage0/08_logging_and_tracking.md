# S0.8 日志、追踪与运行状态

## 目的

建立离线可用、机器可读、多人可复核的运行记录，使多 rank 日志不互相覆盖，恢复运行不重号，失败能够定位，并让 TensorBoard 只承担展示而不是唯一真值。

## 当前依据

- 服务器已安装 TensorBoard，但没有 W&B。
- 服务器普通域名/DNS 和外部网络曾不稳定，正式追踪不能依赖在线服务。
- 后续实验需要保存 loss、梯度范数、参数更新、显存、吞吐、重要性统计和 checkpoint 关系。
- `Agent/worklogs.md` 要求环境和实验结论都有退出状态、指标和证据路径。

## 前置条件

- G1 已固定 `runs`、`results`、`reports` 和 `operations` 的边界。
- G4 已定义 run ID、attempt、step、session 和 provenance。
- G4 只提供事件需要引用的身份/provenance 字段；事件 schema 和统一接口由本子任务定义并冻结。
- 单卡与四卡组件可尚未完成，但在各自集成验收时必须接入本子任务产出的接口，不能另建日志格式。
- 性能相关核验使用 `README.md` 4.1 节的统一测量协议。

## 实施步骤

### 1. 定义事件 schema

1. 为 run lifecycle、训练 step、验证、checkpoint、系统指标、告警和错误定义事件类型。
2. 每个事件包含全局唯一 event ID、schema version、experiment ID、run ID、attempt ID、session ID、rank、事件类型、时间和 session 内单调序号。
3. 训练事件包含 global step、microstep、样本/token 计数、loss、学习率和梯度范数。
4. 系统事件包含 GPU 显存/利用率、CPU 内存、磁盘余量和吞吐。
5. checkpoint 事件包含 checkpoint ID、step、路径、状态和 manifest 摘要。
6. 错误事件包含异常类别、最后有效 step、受影响 rank 和是否可恢复。
7. 禁止把任意对象直接字符串化进事件，所有字段都受 schema 控制。

### 2. 建立日志分层

1. JSONL 事件日志作为机器可读真值。
2. 每 rank 控制台日志分文件保存，用于通信和栈追踪诊断。
3. rank 0 聚合全局指标，其他 rank 不写同一个共享指标文件。
4. TensorBoard 事件由 rank 0 从同一指标流生成。
5. 小型最终摘要从 JSONL 派生，不能手工填写与原始日志不一致的值。
6. 运行状态标记独立于 TensorBoard 是否成功写入。

### 3. 处理并发与原子性

1. 每个 rank 只拥有自己的控制台和 rank 事件文件。
2. 全局聚合器只有一个写者，并记录其 rank 和 session。
3. 状态和摘要先写临时文件，通过 schema 后原子发布。
4. 日志 flush 周期可配置，但关键生命周期和错误事件立即 flush。
5. 进程异常退出后，已 flush 的最后有效事件必须可读。
6. 不允许多个 attempt 追加到同一个没有 session 标记的事件流。

### 4. 处理恢复运行

1. 恢复时读取目标完整 checkpoint 的 global step、canonical event 指针和父 lineage。
2. 新 attempt/session 从 checkpoint 的下一个有效 step 继续，并获得全局唯一的 session/event 身份。
3. 保留旧 session 的 raw 文件，禁止截断或重写；因此旧 session 与恢复 session 在 checkpoint 之后可以出现相同 global step。
4. 用独立、原子发布的 lineage 索引记录父 checkpoint、选中的 session 段和分叉关系；旧 session 中 checkpoint 之后的事件标为 `ORPHANED/SUPERSEDED`，但不删除。
5. canonical 视图只沿被选中的 session 段合并并保留段内全部 typed event；同一 global step 可以合法包含 system、validation、checkpoint 等多种事件。
6. 只对 rank 0/global 的 `optimizer_step` 聚合事件要求每个 global step 恰好一条、不倒退且没有无法解释的缺口；其他事件以 event ID、type、rank 和 session 区分。
7. raw 视图按 session 展示并允许可解释的重复 step；每个 canonical 事件都能反查原始 session/event。
8. 恢复原因、checkpoint ID、旧 session 结束状态和被取代的事件范围进入 provenance。

### 5. 建立 TensorBoard 规则

1. 定义稳定 tag 命名，区分 train、validation、system 和 importance。
2. 统一 step 轴，不混用 microstep、optimizer step 和 token step。
3. 对高频参数级数据只记录聚合量或外部文件引用，避免事件文件爆炸。
4. 验证 TensorBoard 能离线读取并展示关键曲线。
5. TensorBoard 写入失败时保留 JSONL 真值并产生告警，不伪造训练失败或成功。
6. 每个 session 的 TensorBoard 原始文件独立存放，不能让查看器默认同时读取互相分叉的 session。
7. canonical TensorBoard 与最终摘要只能从 canonical JSONL lineage 重新物化，或由读取器显式选择 lineage 中的 session 片段；`ORPHANED/SUPERSEDED` 尾部不得进入 canonical 曲线。

### 6. 建立运行状态与心跳

1. run 使用规范化大写状态：`CREATED`、`RUNNING`、`RESUMABLE`、`SUCCESS`、`FAILED_FINAL`、`ABORTED_FINAL`。
2. attempt/session 使用 `STARTING`、`RUNNING`、`SUCCEEDED`、`FAILED`、`ABORTED`、`STALE`。
3. session/attempt 失败后，只有存在完整 checkpoint 且恢复策略允许时，run 才转为 `RESUMABLE`；否则转为最终失败或中止。
4. 从 `RESUMABLE` 恢复时先验证 checkpoint，再创建新 attempt/session 并把 run 转回 `RUNNING`。
5. `SUCCESS`、`FAILED_FINAL`、`ABORTED_FINAL` 是 run 终态；session 的 `FAILED`/`ABORTED` 不自动等同于 run 终态。
6. `STALE` 是对 session 的运维判定，不是删除许可；只有进程不存在且心跳超时才能标记。
7. 运行期间定期更新项目级心跳，包含最后 step 和时间，不包含秘密或命令参数。
8. 终止时记录操作者、原因和最后安全 checkpoint。
9. 将所有允许/禁止的 run、attempt、session 状态转换固化为可测试矩阵。

### 7. 建立敏感信息和体积保护

1. 对 URL query、认证头、token、Cookie、私钥头和密码模式做日志扫描。
2. 在记录环境变量前使用白名单，而不是记录整个环境。
3. 对异常对象和配置值做长度限制。
4. 为 JSONL、控制台和 TensorBoard 设置滚动或分片策略。
5. 参数级大数组写入 `results`，日志只记录清单和摘要。

### 8. 验证日志开销

1. 按统一协议对最小真值日志与正式追踪做交替的成对 A/B 测试，各执行三次全新进程重复。
2. 每次先执行 10 个 warmup optimizer step，再测量 30 个 step 的吞吐、事件序列化、flush 和 TensorBoard 写入延迟。
3. 验证 JSONL/状态真值写入失败会安全中止；TensorBoard 派生写入失败只告警并保留 JSONL 真值，两者都不得无限阻塞 collective。
4. 将正式追踪开销纳入 S0.10 容量与吞吐基线。

## 产出

- 版本化事件 schema 和日志写入器；
- rank 日志、全局 JSONL、TensorBoard 和摘要的目录规范；
- run/attempt/session 状态转换矩阵、心跳和恢复日志规则；
- raw 事件、canonical lineage 索引和分叉/取代规则；
- 日志敏感信息扫描与体积守卫；
- 单卡、四卡、失败和恢复场景的日志完整性报告；
- 追踪开销报告。

## 核验标准

- 所有事件通过 schema，run/session/rank/step/sequence 字段完整。
- 四卡运行中共享指标文件只有一个写者，每 rank 诊断日志均存在且不覆盖。
- 正常运行的 rank 0/global canonical `optimizer_step` 聚合事件数与配置完全一致，每个 global step 恰好一条；同一步的其他 typed event 全部保留且 event ID 唯一。
- 从 checkpoint K 恢复后，旧 session 文件字节不变且 K 后 raw 事件被保留并标为 `ORPHANED/SUPERSEDED`；canonical lineage 从 K 接续，其 `optimizer_step` 聚合事件每个 global step 恰好一次。
- JSONL 与最终摘要的关键指标一致；canonical TensorBoard 可离线读取同一曲线，且恢复后被取代的 raw/TensorBoard 尾部不会进入该曲线。
- 受控失败时错误事件、最后有效 step、session/attempt 退出状态和 run 可恢复性均可定位。
- 敏感模式扫描为 0 命中，大型数组未进入日志或 Git。
- 正式追踪相对最小日志的吞吐开销按三组成对重复实测且不超过 10%；超过时必须先优化，仍无法满足则只可按统一协议记录获批的有范围 `APPROVED_EXCEPTION`，不能把例外写成达到原阈值。

## Gate 与后续依赖

- JSONL 真值、状态机、canonical lineage 和恢复连续性与 S0.9 共同组成 **G7 可恢复性 gate**。
- 正式追踪开销及其 `PASS/APPROVED_EXCEPTION` 状态进入 **G8-C**，不由 G7 的日志正确性结果代替。
- S0.9、S0.10 和 S0.11 依赖事件 schema 和状态机。
- TensorBoard 故障不自动否定训练数值，但 JSONL 真值、错误告警或状态机任一失败都会阻断。
