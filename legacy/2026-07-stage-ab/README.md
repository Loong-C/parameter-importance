# 2026-07 Stage A/B 旧实验证据归档

本目录保存 2026-07 旧 Stage A/B、SST-2、pruning、reuse intervention 和 U-double 诊断实验的轻量复核证据。旧实验代码分支在归档完成后删除；本目录不用于恢复训练或继续旧 checkpoint。

## 来源

- 旧实验分支：`feat/nlp-pythia-stage-ab`
- 固定提交：`e8354c556dd369cf900cd59d437c1fd59e6a7d48`
- 服务器原始根：`/home/sophgo13/cjl/storage/parameter-importance`
- 服务器证据 tar SHA-256：`ca1bf9f9d2bcc3f6c6eec7d8701bf991a797526201e92c8297f6269541aa420d`
- 归档时间：2026-07-18（Asia/Shanghai）

## 内容

- `reports/`：完整综合报告、图表、表格、summary 和单元测试报告。
- `results/`：汇总结果、CSV/JSON、数值积分、估计器偏差、SST-2 和 U-double 诊断证据。
- `runs/`：逐运行配置、环境、metrics、validation、evaluation、comparison、diagnostics、结果和完成标记。
- `docs/`：旧分支中的综合结论和 U-double 诊断总结。
- `worklogs/`：旧实验的阶段日志、确定性修复、最终总结和诊断记录。
- `inventory.csv`：每个来源文件的原路径、归档路径、大小、时间和 SHA-256。
- `MANIFEST.sha256`：归档文件的最终内容哈希，不包含该 manifest 自身。

归档的 310 个来源文件共 33,638,654 bytes；新增 README 和 inventory 后仍受 40 MiB 总量及 10 MiB/文件上限约束。

## 明确排除

- `runs/*/checkpoints/` 和 `runs/*/best-validation/`；
- `.pt`、`.safetensors`、`.pth`、`.ckpt`、`.bin` 等权重、优化器和张量文件；
- `results/*/units/` 中约 6.5 GB 的逐单元原始数据；
- 隐藏失败临时目录、锁、lease 和运行控制状态；
- 旧实验源码、测试、运维脚本和 runbook；
- 数据集、基础模型、环境、wheelhouse、缓存、Pythia 源码及资产 manifest。

## 解释边界

这些文件足以复核旧实验的配置、指标、报告和结论，但不包含恢复训练所需的权重或旧代码。正在进行的全量 Pile 下载属于保留的基础资产准备，不属于本归档，也不受旧实验清理影响。
