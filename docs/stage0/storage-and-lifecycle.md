# Stage 0 存储布局与生命周期

## 路径合同

服务器运行必须显式设置：

```bash
export PARAM_IMPORTANCE_DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
```

代码不会回退到用户主目录、根盘或系统 `/tmp`。`StorageLayout` 只允许解析 `Agent/server.md` 规定的 13 个子目录，并在规范化后再次检查路径仍位于 `DATA_ROOT` 内。

Git 仓库只保存代码、测试、配置、文档、工作日志、小型 manifest 和小型最终报告。数据、模型、wheel、环境、缓存、checkpoint 和大型原始运行结果只放在 `DATA_ROOT`。大型资产的服务器运行副本是权威副本，但在 G1-D 通过前不称为“已备份”。

## 目录用途

| 目录 | 唯一用途 |
|---|---|
| `datasets` | 已获取或正在受控获取的数据；只有 `ready` manifest 可作输入 |
| `models` | 固定模型、tokenizer 与初始化权重 |
| `cache` | Hugging Face、Datasets、Torch、XDG 与编译缓存 |
| `checkpoints` | 按 run ID 隔离的完整训练状态 |
| `runs` | resolved config、provenance、事件、控制台和运行状态 |
| `results` | 参数级及其他原始数值结果 |
| `reports` | 服务器侧派生的小型/大型报告 |
| `manifests` | 环境、资产、checkpoint 与结果身份 |
| `operations` | 项目租约、心跳、维护与故障记录 |
| `wheelhouse` | 已验证的 Linux 离线 wheel 集合 |
| `envs` | 不可原地破坏的版本化虚拟环境 |
| `source` | 必须留在大盘的外部源码/构建材料 |
| `tmp` | 对象专属临时文件与 Git bundle 中转 |

## 缓存和临时文件

`runtime_cache_environment()` 为每次 session 显式生成 `HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE`、`TRANSFORMERS_CACHE`、`TORCH_HOME`、`TORCH_EXTENSIONS_DIR`、`XDG_CACHE_HOME`、`TMPDIR`、`TMP` 和 `TEMP`。所有值都必须位于 `DATA_ROOT`，session 临时目录按 run/attempt/session 隔离。

运行结束只处理该 transaction/attempt manifest 记录的精确临时路径。诊断所需文件移动到对应 run 的 `diagnostics` 并更新 manifest；不得使用递归通配清理。

## Run 生命周期

新 run 通过原子目录创建，目录存在即失败。每个逻辑轨迹具有独立 `run_id`；操作者重试增加单调 `attempt_id`；连续进程组使用独立 `session_id`。独立重复创建新 run，不复用旧结果目录。

Run 状态为 `CREATED → RUNNING → RESUMABLE/SUCCESS/FAILED_FINAL/ABORTED_FINAL`。终态不可重新进入运行。attempt/session 状态为 `STARTING → RUNNING → SUCCEEDED/FAILED/ABORTED`；`STALE` 只在进程不存在且心跳超时后由运维判定，不是删除许可。完整事件和恢复 lineage 在 S0.8/S0.9 扩展。

## 空间与保留

启动预计新增 `E` 字节的工作前，大盘可用空间至少为 `E + max(0.2E, 100 GiB)`；根盘低于 10 GiB 时禁止新的项目写入。字节空间和 inode 必须同时检查。

细化规则以 `policies/storage-lifecycle.json` 为机器可读真值：

- 固定 revision 可重新获取的上游模型、数据、wheel 和缓存属于可替换资产；
- 正式 checkpoint、原始结果、人工判定和唯一运行证据属于不可再生资产；
- checkpoint 保留最新完整、最佳验证和阶段里程碑；任何删除由不可变 checkpoint ID 和引用索引驱动；
- 活动 run、活动锁、`.part` 和相关进程存在时禁止清理；
- 每次清理保存目标清单、绝对路径验证、释放空间和 tombstone。

## Gate 命令

在本机或服务器仓库运行 Git 守卫：

```bash
PYTHONPATH=src python -m param_importance_nlp git-guard --repo .
```

在服务器验证 13 个目录及 canary（会创建、原子替换并精确删除 13 个小文件）：

```bash
PYTHONPATH=src python -m param_importance_nlp storage-check \
  --data-root /home/sophgo13/cjl/storage/parameter-importance \
  --require-writable --canary
```

canary 不扫描、读取或修改数据资产、`.part`、锁或其他项目文件。

## 持久性决策

当前没有获授权的第二故障域。用户已通过
`reports/stage0/g1-persistence-decision-20260719.json` 明确接受仅 Stage 0 可再生
smoke 产物的限时单盘丢失风险，因此 G1-D 当前以
`PASS — TIME_BOUNDED_RISK_ACCEPTANCE` 满足：

1. 有效期至 2026-08-18 23:59 CST 或 Stage 4 开始前，以先发生者为准；
2. 仅覆盖可重新生成的 Stage 0 smoke checkpoint、测试事件和基础设施诊断；
3. 不覆盖 Stage 4/5 正式产物、人工判定或论文唯一证据；
4. 存储拓扑变化、磁盘/文件系统异常或用户撤回也会使批准提前失效。

进入 Stage 4 前必须获得第二故障域并完成恢复演练，或取得覆盖正式产物的新明确
决定；当前批准不得自动延展。
