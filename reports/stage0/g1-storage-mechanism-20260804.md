# G1-B 存储机制重验（2026-08-04）

- G1-B：`PASS`
- G1-D：`PASS — TIME_BOUNDED_RISK_ACCEPTANCE`
- G1 总 gate：`PASS`
- 验证提交：`3bb0f5ace2a6832b20ba5ce67feed798d315084a`

同一提交（`3bb0f5a`）已在本机和服务器完成 CPU 测试与 Git 守卫：相关测试
本机与服务器均为 23 项通过（原子存储、生命周期/容量、Git 守卫）。服务器
13 个规定目录全部完成小型写入、读取、SHA-256、原子替换和精确清理；13/13
canary 通过、0 失败、0 残留。canary 证据哈希为
`ee24d9865a899068327f7ebdd6b763fa04ae87589cca0b77d3578acdfc0da11a`。

Stage 0 新增量按 620,000,000,000 bytes 估算，启动要求
744,000,000,000 bytes；服务器大盘复验可用 3,011,151,384,576 bytes，根盘可用
39,949,303,808 bytes（高于 10 GiB 保护线），大盘 inode 可用 233,381,501
（使用率 1%）。容量只表示分析启动预算通过，不替代 G8 的真实显存、内存、
吞吐和 checkpoint 实测。

活动 Pile 下载 `document-00009-of-00020.bin.part`（22,882,025,472 bytes）及
11 个 `.lock` 文件在 canary 前后保持不变，未读取、修改、改名或竞争；canary
只操作随机唯一的小文件。

G1-D 沿用 2026-07-19 的限时风险接受：仅覆盖 Stage 0 可再生 smoke 产物，
有效至 2026-08-18 23:59 CST 或 Stage 4 开始前（先发生者）；Stage 4/5 正式
产物不在批准范围。本轮为状态复核，不是新的风险接受；批准证据见
`reports/stage0/g1-persistence-decision-20260719.json`。进入 Stage 4 前必须
重新建立持久性决策。
