# Stage 0–9 本机契约与核心代码验证报告

本机适用验证已通过：最终测试集为 **320 passed、6 skipped**，退出码 0。这里的 `PASS` 只属于 `local_fixture`，不表示任何 formal Stage、服务器 Gate、GPU/NCCL 或真实训练已完成。

## 可复核身份

- 分支：`feat/local-contracts-core-stage0-9`
- 验证树父提交：`4bc2c3d794c29854009017f5831e62ff839906aa`
- resolved config：`5ed1396073dc3d3c2360c36c6070f02c5d01bb6084da7b95c3b2721495ed94d3`
- local contract freeze：`4fc9ecc25d79b1a1ac2b514aac4fed0180242ea997b9da7f3e1d4c0936b8c4ef`
- coordinate registry：`5f59b80a95ec2a011f7a97cc600c7d0ec43afa547157019bd7b149b50bb5b155`
- analysis report：`b211ae6c7f1b2da8e1bc75be8f05811969789b9d72288dac506c4aa89a1af6f3`

## 确定性重放

在 `artifacts/local-fixture/verify-a` 与 `verify-b` 两个独立目录运行统一 CLI。两次运行的 config、seed plan、registry、pipeline result、artifact、analysis report 与 Markdown 文件哈希逐项相同；其中主 artifact 文件 SHA-256 均为 `4d70cf82e187f1a0160403ac4d766bf06dee7864b4931f76a527ae649f539a5a`。运行目录由 Git 忽略，不进入提交。

## 本机验收

- 326 项收集：320 通过，6 跳过，退出码 0。
- 33 份仓库 schema 经统一 CLI 重放；格式化 JSON 被允许，BOM、重复键、NaN/Inf 和伪 schema 仍被拒绝。
- `compileall`、Git whitespace guard、秘密扫描、10 MiB 大文件守卫及 Git 资产守卫通过。
- 4 个 Bash 管理脚本和 2 个 PowerShell 包装器只做静态语法解析，未执行。
- 本机环境为 Python 3.12、Torch 2.10.0+cpu，`torch.version.cuda is None`；未安装 Transformers、Datasets、Accelerate、Safetensors 或 TensorBoard。

## 明确未运行与阻塞

- 3 项 Gloo world-size 1/2/4 测试：当前 Windows Torch wheel 报告 Gloo 可用，但没有受支持设备，标为 `BLOCKED` skip。
- 2 项 Windows 目录 symlink 权限测试与 1 项 POSIX 权限语义测试：平台能力不足，明确跳过。
- 服务器 HEAD、服务器 `Agent/*.md` 哈希及服务器 formal Gate：`BLOCKED: server_unreachable`。
- CUDA/NCCL、真实模型/数据、pilot B/M/R、正式 quadrature 阈值与真实训练结论：`NOT_RUN` 或 `UNFROZEN`。

机器可读证据见 [local-validation-20260722.json](./local-validation-20260722.json)。
