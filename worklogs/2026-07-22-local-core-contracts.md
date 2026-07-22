# 2026-07-22 本机契约冻结与核心代码建设

- 任务范围：仅在 Windows 本机实现 Stage 0–9 所需的版本化契约、CPU 核心、合成夹具和可扩展接口；不连接服务器，不下载模型或数据，不执行管理员脚本，不生成正式实验结论。
- 当前状态：本机契约与核心实现已完成验证；formal Stage、真实训练与服务器 Gate 仍未完成。
- 工作分支：`feat/local-contracts-core-stage0-9`

## 2026-07-22 11:14 CST — 审查既有 Stage 0 未跟踪产物

### 目标与范围

- 本阶段要完成什么：审查六个 GPU 管理脚本与两份授权报告，确认语法、哈希绑定和秘密边界，形成可复核的 Stage 0 收口提交。
- 不在本阶段处理什么：不通过 SSH 连接服务器，不执行任何管理员脚本，不修改服务器服务、驱动、PCI 设备或文件。

### 实际修改

- 保留六个既有运维脚本和两份授权报告，不改动管理员 Bash 脚本的历史证据常量。
- 修正 `run_admin_gpu_path_b.ps1` 中已经漂移的本地脚本 SHA-256，使包装器绑定当前 `admin_apply_gpu_path_b.sh` 的实际内容。
- 新建本工作日志，后续各实现波次持续追加。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| Bash 静态语法 | `bash -n ops/stage0/*.sh`（仅四个未跟踪管理脚本） | 4 个脚本退出码均为 0 | 本轮终端记录 |
| PowerShell 静态语法 | `System.Management.Automation.Language.Parser.ParseFile` | 2 个包装器均为 0 个解析错误 | 本轮终端记录 |
| 哈希绑定 | `Get-FileHash -Algorithm SHA256` | 重启路径包装器原本匹配；Path B 包装器发现并修复一处哈希漂移 | 本日志；对应包装器 |
| 秘密模式扫描 | 私钥、Bearer、令牌、密码字面量、签名 URL 模式 | 0 命中 | 本轮终端记录 |
| 现有测试基线 | `pytest -q` | 121 passed，2 skipped，退出码 0 | 本轮终端记录 |

### 问题、原因与风险

- 当前服务器不可连接，因此所有服务器身份、GPU 健康、NCCL、资产和多端同步 Gate 保持 `BLOCKED` 或 `NOT_RUN`。
- 两份授权报告含已获项目所有者批准记录的服务器内部硬件身份信息，但未包含密码、私钥、令牌、Cookie 或签名 URL。
- 当前测试使用全局 Python，NumPy/SciPy/Pandas/Matplotlib 版本与仓库声明漂移；下一阶段将建立仓库忽略的锁定 `.venv` 后重新验证。

### Git 与多端同步

- 本机分支/HEAD：`feat/stage0-infrastructure` / `5cc53930a3f745fbd3e9ea4e171bd0773172984a`。
- GitHub 分支/HEAD：进入提交和推送步骤后核对。
- 服务器分支/HEAD：`BLOCKED: server_unreachable`。
- `Agent/*.md` 哈希核对：本机可读；服务器侧核对为 `BLOCKED: server_unreachable`。

### 下一步

- 复验脚本哈希、运行完整本机测试和 Git 守卫，提交并推送 Stage 0 收口产物。
- 从收口提交创建 `feat/local-contracts-core-stage0-9`，建立锁定本机环境并开始契约实现。
