# Stage 0–9 本机验证报告（0.4.0）

- 本机适用测试：`530 passed, 9 skipped, 0 failed`，共收集 539 项。
- 跳过项：3 项 Windows symlink/POSIX 权限语义；6 项因当前 Windows Torch CPU wheel 没有可用 Gloo device。
- 确定性流水线：两个干净根逐字段、逐字节一致；同一根 fresh resume 后结果不漂移。
- 静态检查：`compileall`、全部 schema 重放、`git diff --check`、秘密模式扫描和大文件 guard 通过。
- 本机环境：Python 3.12.4、Torch 2.10.0+cpu。
- 正式状态：服务器、CUDA/NCCL、真实模型/数据资产和 formal Gate 为 `BLOCKED`；B/M/R、probe/节点预算与科学阈值保持 `UNFROZEN`。

本报告只证明本机代码、fixture、恢复和产物合同可运行，不是正式实验结论，也不会把本机 `PASS` 提升为 formal Gate `PASS`。
