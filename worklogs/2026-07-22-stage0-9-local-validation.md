# 2026-07-22 Stage 0–9 本机实现与验证工作日志

## 范围与结论边界

- 仅在 Windows 本机完成 `param_importance_nlp` 0.3.0 的合同冻结、CPU 核心、synthetic fixture 与可扩展编排。
- 未连接服务器、未运行 SSH、未执行管理员脚本、未下载模型或数据、未安装可选 Hugging Face/TensorBoard 依赖。
- 本机验证状态为 `PASS`；所有 formal Stage、服务器 Gate 和真实训练结论均保持 `BLOCKED`、`NOT_RUN` 或 `UNFROZEN`，没有宣称 Stage 完成。

## 实现波次

1. `1ab0de1`：静态审计六个 GPU 管理脚本与两份授权报告，修正一处包装器脚本哈希并推送 Stage 0 分支。
2. `9e155b0`：公共 contracts、33 份 schema、本机配置与 Windows CPU hash lock。
3. `beb5c05`：Stage 0 runtime、Stage 1 数学/registry/loss/estimator/accumulator/oracle 核心。
4. `5bcfafd`：Stage 2 固定状态 sampling、reference、paired runner、shard/reducer 恢复与 decision。
5. `35c7dd1`：Stage 3 endpoint/probe/path state、quadrature 编排、节点缓存与参考收敛。
6. `4bc2c3d`：Stage 4–9 route DAG、剪枝、消融、统计、图表与 hash-bound 报告。
7. 最终集成提交：统一 CLI、本机 fixture、证据报告与 stale freeze 更新。

以上边界提交均推送到 `origin/feat/local-contracts-core-stage0-9`。服务器分支同步未进行，原因是 `server_unreachable`。

## 环境证据

- Python：3.12；Torch：2.10.0+cpu；CUDA runtime：`None`。
- 精确 pin：32 项；直接 hash lock、普通 lock 与 wheel 文件名/大小/SHA-256 清单一一对应。
- 未安装：Transformers、Datasets、Accelerate、Safetensors、TensorBoard。
- `.venv/`、`artifacts/`、wheel、模型、数据、checkpoint 与缓存均被 Git 忽略或守卫拒绝。

## 验证结果

| 检查 | 结果 | 证据 |
|---|---:|---|
| 完整 pytest | 320 passed，6 skipped，exit 0 | 本轮终端；`reports/local/local-validation-20260722.json` |
| 测试收集 | 326 | 本轮终端 |
| 双次 fixture | 全部内容哈希相同 | `artifacts/local-fixture/verify-a`、`verify-b`（Git 忽略） |
| schema 重放 | 33/33 PASS | 统一 `contract-validate` CLI |
| compileall | PASS | `src/`、`ops/` |
| Bash 静态语法 | 4/4 PASS | 仅 `bash -n`，未执行脚本 |
| PowerShell 静态语法 | 2/2 PASS | 仅 Parser API，未执行脚本 |
| 秘密/大文件/Git 守卫 | PASS | 0 个秘密命中；0 个 >10 MiB 候选 |
| formal fail-closed | PASS | local config 返回 exit 3，缺少 Stage 0–3 freeze/decision/Gate |

## 跳过与阻塞

- Windows symlink privilege：2 项跳过。
- POSIX permission semantics：1 项跳过。
- Gloo world size 1/2/4：3 项 `BLOCKED` skip；不得外推为 CUDA/NCCL 或 formal distributed PASS。
- 服务器 HEAD、服务器 `Agent/*.md` 哈希、服务器 Gate、GPU/NCCL 与真实资产：`BLOCKED: server_unreachable`。
- 正式 B/M/R、真实 reference、默认 quadrature、probe 数、节点预算与阈值：`UNFROZEN/BLOCKED`。
