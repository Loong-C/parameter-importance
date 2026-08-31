# G1-B 存储机制核验（2026-07-19）

- G1-B：`PASS`
- G1-D：`PASS — TIME_BOUNDED_RISK_ACCEPTANCE`
- G1 总 gate：`PASS`
- 验证提交：`02f4d4376ecdf624552c70a8f066e8d1957cfa06`
- canary 实现提交：`50282e3d879f09a7ba9fc1dd2541bbe24bda00b2`

同一提交已在本机和服务器完成 CPU 测试与 Git 守卫。服务器 13 个规定目录全部完成小型写入、读取、SHA-256、原子替换和精确清理；13/13 canary 通过、0 失败、0 残留。报告哈希为 `76fa7a93488eeee5f03bcc7b96c208c4365e07b1219fb348f773a96435757fac`。

Stage 0 新增量按 620,000,000,000 bytes 估算，启动要求 744,000,000,000 bytes；服务器大盘复验可用 3,079,252,004,864 bytes。根盘复验可用 15,175,081,984 bytes，高于 10 GiB 保护线；大盘 inode 可用 233,426,895，使用率 1%。容量只表示分析启动预算通过，不替代 G8 的真实显存、内存、吞吐和 checkpoint 实测。

活动 Pile shard 7、其 `.part`、对象锁、元数据和下载进程均未读取、修改、改名或竞争。canary 只操作随机唯一的小文件。

用户已明确接受仅 Stage 0 可再生 smoke 产物的单盘丢失风险，有效至 2026-08-18 23:59 CST 或 Stage 4 开始前（先发生者）；Stage 4/5 正式产物不在批准范围。批准证据见 `reports/stage0/g1-persistence-decision-20260719.json`。因此 G1-D 与 G1 当前通过，但进入 Stage 4 前必须重新建立持久性决策。
