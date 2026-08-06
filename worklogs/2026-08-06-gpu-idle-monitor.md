# 2026-08-06 GPU 占用排查与空闲探测自动化

- 任务范围：判断服务器外部训练占用何时结束；建立定时探测 GPU 空闲与 S0.9 解锁条件的自动化。
- 当前状态：完成（外部任务已崩溃释放 GPU；探测自动化已建立并注册计划任务）
- 工作分支：`feat/stage1-cpu-evidence`

## 2026-08-06 23:30 CST — 外部训练排查与监控自动化

### 目标与范围

- 完成：识别占用 4 张允许 GPU 的外部任务并估算剩余时间；提供定时“空闲探测 + G0-G finalizer 就绪”自动化。
- 不做：不终止其他代理的训练；不做管理员级 GPU 重置/重启；不启动 S0.9 正式链。

### 实际修改

- 新增 `ops/monitor/check_gpu_idle.ps1`：SSH 只读探测 GPU 计算进程/显存、verl 进程、G0-G finalizer SUCCESS、boot_id、Xid、S0.9 链日志尾，写入 `ops/monitor/runtime/` 并在就绪时写 `S0_9_READY.flag`。
- 新增 `ops/monitor/README.md`：注册/查看/卸载说明。
- `.gitignore`：忽略 `ops/monitor/runtime/`。
- 新增本工作日志。
- 服务器或外部状态：仅只读检查；未改动服务器文件、未终止外部进程。

### 实验与验证

| 项目 | 命令/来源 | 结果 | 证据路径 |
|---|---|---|---|
| 外部任务身份 | 服务器 `ps -ef` | verl agentic RL（Qwen3-14B + Alfworld），`total_training_steps=1`、`total_epochs=1`，实验名 `gigpo_qwen3_14b_full_smoke` | 本日志 |
| 外部任务结局 | `logs_smoke_qwen3_14b.txt`、Ray driver 日志 | 崩溃：`AssertionError: max_token_len=2048 < max_seq_len=2560`；driver 于 2026-08-06T15:08:42Z 退出 | `/home/sophgo13/lcl/agentic-storage/verl-agent/logs_smoke_qwen3_14b.txt` |
| GPU 现状 | `nvidia-smi` | 4/4 卡 0 MiB、0%，无 compute apps；无 verl/ray 进程 | 本日志 |
| 自动化脚本 | `powershell -File ops/monitor/check_gpu_idle.ps1` | 初版误把历史 SUCCESS 计入（8/3、8/4 两次），误报 READY；已改为只认最新一次 finalizer 且 boot_id 等于当前 boot，修正后 `STATE=IDLE_WAIT_FINALIZER` | `ops/monitor/runtime/` |

### 判定

- 外部占用不是需要等待数小时/数天的长训练：它是 1-step smoke test，已在约 38 分钟后因配置错误自行崩溃，GPU 现已空闲。
- 对方 tmux（`verl-agent-1`）与监控进程仍存在，存在重跑可能；定时探测可捕捉再次占用与再次释放。
- S0.9 剩余唯一外部阻塞：`0000:a4:00` 的 Xid 63 row-remap 未激活，G0-G finalizer 5 次尝试均 FAILED；该事项仍需管理员处理。

### 问题、原因与风险

- 外部任务崩溃根因：`actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2048` 小于实际序列长度 2560；属对方配置问题，本项目不修复。
- 风险：对方可能修正后重跑，GPU 会再次被占；监控脚本每 30 分钟记录一次，能在下次空闲时给出就绪信号。
- 限制：计划任务使用 Interactive 登录类型，仅当前用户登录时运行；通知依赖本机可用通知渠道（BurntToast/msg），无渠道时仅保留标记文件。

### Git 与多端同步

- 本机/GitHub：`feat/stage1-cpu-evidence`（提交本日志后更新）。
- 服务器：`feat/stage1-cpu-evidence` @ `5526ce1`（本日志提交后快进）。
- `Agent/*.md` 哈希：不受影响，仍 5/5 一致。

### 下一步

- 管理员处理 `0000:a4:00`（reset 激活 row remap 或干净重启）后，root 重跑 finalizer 至 SUCCESS；监控脚本检测到 `FINALIZER_SUCCESS>0` 且 GPU 空闲时写出 `S0_9_READY.flag` 并尝试通知。
- 就绪后在服务器 `formal/run-975f83c` 重跑 `$DATA_ROOT/tmp/g7-recovery-chain-975f83c.sh` 完成 S0.9。
