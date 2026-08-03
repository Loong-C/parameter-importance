# 2026-08-03 Stage 0 GPU 恢复与 S0.3 收口

- 任务范围：恢复 sophgo13 的持久四卡路径、完成 G0-G 复验，并在不执行 S0.4–S0.12
  的前提下完成 S0.3/G2 推荐环境发布。
- 当前状态：完成
- 工作分支：`feat/local-contracts-core-stage0-9`

## 2026-08-03 13:02 CST — retry3 完成管理员服务恢复

### 目标与范围

- 恢复首次服务 finalizer 失败后建立的安全保持状态，重新启用正常运行所需的
  NVIDIA persistence、containerd、Docker 与 LXD 入口。
- 继续强制精确四 UUID 可用/四 UUID 排除合同；不重试故障卡、不清除 ECC、不把两张
  项目范围外的健康备用卡加入运行集合。
- 管理员密码只由用户在可见远程 TTY 中输入，Codex 未读取、记录或写入密码。

### 实际修改

- 修复 finalizer 的服务恢复顺序：先解除并启用全部 allocator unit，再按顺序启动，
  避免 `docker.socket` 在 `docker.service` 仍 masked 时被 systemd 拒绝。
- 为首次失败产生的安全保持状态增加严格重试恢复：只有 persistence inactive 且所有
  NVIDIA 节点精确为 root:root 0600、无 GPU client 时，才恢复驱动声明的节点权限并
  启动 persistence；后续失败仍回到安全保持。
- NVIDIA capability 节点权限按 `/proc/driver/nvidia/capabilities` 的
  `DeviceFileMinor/DeviceFileMode` 动态核对；普通节点继续要求 0666。
- 只有实际 PCI function 绑定 `nvidia-nvswitch` 驱动时才要求 Fabric Manager；本机无
  该绑定，故记录 `NOT_APPLICABLE_MASKED`，不再把控制节点本身误判为 NVSwitch fabric。
- PyTorch 2.12 返回不带 `GPU-` 前缀的 UUID，三个管理员/复验脚本改为严格解析后统一成
  canonical UUID，未放宽允许集合。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 管理员 finalizer | 哈希固定脚本，经可见 TTY `sudo` 执行 | `PASS`；run `20260803T050131Z`；全部 SHA256SUMS 复核通过 | `/var/lib/parameter-importance/stage0/g0-g-uuid-exclusion/service-finalize/finalize-20260803T050131Z.8nhBSwZk` |
| 精确设备集合 | NVML、PyTorch、模块参数、PCI/驱动映射 | 只见批准的 4 张卡；4 个排除 UUID 顺序完全一致 | `reports/stage0/g0-g-gpu-final-20260803.json` |
| 服务恢复 | `systemctl is-active/is-enabled` | persistence、containerd、Docker active/enabled；LXD 入口恢复；Fabric Manager inactive/masked 且不适用 | 同上 |
| 健康状态 | ECC、row-remapper、compute apps、当前启动 journal | 四卡不可纠正 ECC 均为 0；pending/failure 均 No；无 client；0 个关键 GPU/PCI 事件 | 同上 |

首次 finalizer 在 `docker.socket` 启动顺序处失败，但其 `safe_hold_verified=1`；retry3 从该
已验证安全状态恢复并通过。失败证据没有删除或覆盖。

## 2026-08-03 13:11 CST — 四卡 CUDA/NCCL 环境 smoke 通过

### 实际运行与结果

- 候选环境：`env-v1-aaf55d32341c60a904b7d80a3c09d69bff7b7ee500e43f42a6432f65438d9d29`。
- 每张白名单 GPU 分别完成 CUDA 向量运算与 256×256 FP32 矩阵乘；结果逐卡一致且有限。
- 4 个 NCCL rank 各绑定一张批准 GPU，对 1,048,576 个 FP32 元素连续执行 3 次
  all-reduce；每次 min/max 都为期望和 10.0。
- PyTorch 2.12.1+cu126、CUDA runtime 12.6、cuDNN 9.10.2、NCCL 2.29.3。
- 运行前后四卡 volatile/aggregate 不可纠正 ECC 均为 0，row-remap pending/failure
  均为 No；结束后无计算进程，增量内核日志无关键 GPU/PCI 错误。
- 全部证据清单 SHA-256 复核通过；结果位于
  `/home/sophgo13/cjl/storage/parameter-importance/operations/stage0/g0-g-uuid-exclusion/cuda-nccl-smoke-20260803T051156Z`。

该运行是 G0-G/G2 的最小通信 smoke，不采用 S0.7 的 20 warmup/50 measurement、三次
进程组重建、DDP/分片/梯度累积/`no_sync` 协议，因此不宣称 G6 通过。

## 2026-08-03 13:32 CST — 发布训练可用环境引用并完成 S0.3

### 实际修改

- 新增 `ops/stage0/promote_environment_after_gpu_gate.py`：严格验证 CPU candidate、
  root 管理员证据及 SHA256SUMS、CUDA/NCCL smoke、当前 boot、模块排除参数、服务状态、
  NVML/PyTorch UUID、ECC、row-remap、内核日志和逐卡即时张量，再发布不可变 qualification
  与原子推荐引用。
- 增加 Windows 可收集的纯解析/哈希回归测试，并修复两项服务器暴露的格式兼容问题：
  标准 `sha256sum` 的 `./文件名` 前缀，以及 NVIDIA `/proc` 参数的整串双引号。
- 第一次晋级尝试因 `./` 规范化缺陷在发布前失败；第二次因 `/proc` 引号解析过严在发布
  前失败，两次都确认没有生成推荐引用。
- 首份成功 qualification 的 `generated_at` 使用了证据时间。内容和 gate 正确，但字段
  时间语义不精确；该文件按不可变规则保留并标为 superseded。发布器改为分别记录
  `evidence_observed_at` 与真实 `generated_at` 后，新建资格清单并原子更新推荐引用。

### 实验与验证

| 项目 | 结果 | 证据 |
|---|---|---|
| 本机环境相关回归 | 最终 70 passed、1 skipped（仅 Windows 无 POSIX 权限语义） | `tests/test_environment_gpu_promotion.py` 等 |
| Bash/PowerShell/Python 静态入口 | `bash -n`、PowerShell parser、`compileall`、`git diff --check` 均通过 | 本轮终端记录 |
| 最终 GPU qualification | `PASS`；mode 0644、owner `sophgo13:sophgo13`、link count 1；SHA-256 `29d34128...13333c` | `/home/sophgo13/cjl/storage/parameter-importance/manifests/environment-gpu-qualifications/gpuq-v1-bf47422fb498ccbe300742a2e22d819214e0de6939dd4fc85574683584fdd10b.json` |
| 推荐环境引用 | `G2=PASS`、`training_eligible=true`；mode 0644、单链接；SHA-256 `a7946156...6dd31` | `/home/sophgo13/cjl/storage/parameter-importance/manifests/environment-recommended.json` |
| 发布后设备复核 | 精确四 UUID、不可纠正 ECC 0、无 compute client、服务状态保持正确 | `reports/stage0/g2-environment-final-20260803.json` |

### Gate 判定

- G0-C：`PASS`；本机/服务器五份 `Agent/*.md` SHA-256 再次 5/5 一致。
- G0-G：`PASS`；路径 B 持久四卡合同与当前启动健康证据通过。
- G0：`PASS`。
- G1：`PASS`；截至本日，Stage 0 可再生 smoke 单盘风险接受仍有效；不覆盖 Stage 4/5。
- G2 / S0.3：`PASS` / 完成。
- S0.4–S0.12：按用户要求暂停，未在本轮继续实施或宣称完成。

### Git 与多端同步

- 本机/GitHub 分支：`feat/local-contracts-core-stage0-9`；资格发布器提交
  `c2b19f34389c210125cfb1407cdb9c8a539ff800` 已推送。
- 服务器运行环境通过 SHA-256 固定的脚本副本执行，qualification 绑定上述提交。
- 服务器仓库仍在 `feat/stage0-infrastructure@5cc5393`，保留六个已知未跟踪的早期恢复
  文件；本轮没有覆盖、移动或删除这些文件，也不宣称 G10 三端同步完成。
- `Agent/*.md`：本机与服务器五份文件哈希全部一致。
- 四份晋级器临时副本在逐一核对绝对路径、属主和 SHA-256 后从 `DATA_ROOT/tmp` 精确
  删除；对应内容可从已推送 Git 提交恢复。管理员与 smoke 证据、两份不可变 qualification
  和推荐引用均保留。

### 问题、原因与风险

- 两张故障 GPU 仍是被持久排除的故障对象，不是已物理修复；项目训练路径不使用它们。
- 另外两张健康 GPU 作为项目范围外备用卡被排除，防止运行集合漂移。
- Stage 0 可再生 smoke 的单盘风险接受在 2026-08-18 23:59 CST 或 Stage 4 开始前
  （先发生者）失效；Stage 4/5 正式产物仍没有本批准覆盖。
- 重启前保存的 Pile `.part` 未丢失，但恢复下载需要刷新签名 URL；S0.4 已暂停，因此本轮
  未恢复下载。

### 下一步

- 按用户要求停在 S0.3；不要启动 S0.4–S0.12。
- 将来开始 S0.4 前先复核当前推荐环境、GPU qualification 和活动下载状态；开始 Stage 4
  前必须重新解决正式产物持久性决策。
