# 2026-07-13 服务器最小闭环环境准备日志

## 工作目标

依据 `Agent/remote_access.md` 和 `plan/NLP参数重要性完整实验计划书.md`，为阶段 A+B 的最小闭环准备服务器环境、依赖、数据集、模型、Pythia 源码快照和离线验收工具，使后续可以直接开始实现和实验，不再临时下载资产。

## 已完成内容

### 1. 网络与存储复测

- 确认目标服务器公网 HTTPS/443 已开放。
- 确认普通域名访问仍因系统上游 DNS `172.26.160.1` 被内网访问策略阻断而失败。
- 未修改共享服务器 DNS、`/etc/hosts`、路由、防火墙或实验室电脑 `start.cmd`。
- 实测服务器直连 Xet 单路约 1.35 MiB/s，四路聚合约 1.66 MiB/s；lab-pc 到服务器 SCP 约 1.45 MiB/s。
- 确认大盘 `/home/sophgo13/cjl/storage` 为 ext4、可读写，尚有约 3.1 TB 可用空间。

### 2. 下载决策

- 不下载约 602 GB 全量 Pythia 预分词数据。
- 固定下载官方 `document-00000-of-00020.bin`（30,000,000,000 bytes）和完整 `document.idx`（1,757,184,042 bytes）。
- 固定数据 revision：`4647773ea142ab1ff5694602fa104bbf49088408`。
- 固定 shard0 SHA-256：`1ce355bd2683627d0ff689f8578115cf3df84bd1edf3410e6aca9705d31fc6ea`。
- 固定 idx SHA-256：`1d9fdd760295eb2007a4874440b27c559ca722239fa2814aa8a2ee6724b7852f`。

### 3. 安全混合下载链路

- 已实现 lab-pc 仅获取公开签名 URL、服务器从 Xet/CDN 直接下载的混合链路。
- 签名 URL 只经 SSH stdin 传递，不写入命令参数、文件或日志。
- 服务器使用命令级公网 DNS 交叉解析及 `curl --resolve`，下载到对象专属 `.part`。
- 已实现 URL 刷新、断点续传、HTTP 206、Content-Range、目标大小、SHA-256 和对象元数据检查。
- 已在 lab-pc 注册 `CjlPilePrefix` 用户级计划任务，下载任务不依赖当前 SSH 会话。
- 最近一次确认时 shard0 `.part` 已开始增长，说明真实 Xet 数据链路工作正常。

### 4. Python 与环境准备工具

- 已在 lab-pc 的 `C:\Users\cjl\Apps\Python312` 安装用户级 Python 3.12.10，未修改系统 Python 和全局 PATH。
- 已确认核心版本真实存在：PyTorch `2.12.1+cu126`、Transformers `4.57.6`、Datasets `4.8.5`。
- 已实现 Linux CPython 3.12 wheelhouse 的解析、精确锁定、下载、SHA-256 manifest、tar 打包、SCP 和服务器 `--no-index` 安装脚本。
- 首次解析发现只指定 `manylinux_2_28` 会漏掉 Matplotlib；脚本已修正为同时接受 `manylinux_2_28`、`manylinux_2_17` 和 `manylinux2014`。

### 5. 模型、数据集和验收工具

- 已实现固定 revision 的最小资产缓存脚本，范围包括：
  - Pythia 160M deduped step0；
  - Pythia 160M deduped step512；
  - Pythia 14M step0；
  - GLUE/SST-2 全部分片；
  - WikiText-103 raw 全部分片；
  - 官方 Pythia 源码固定 commit 快照。
- 已实现资产大小、SHA-256、revision 和文件清单 manifest。
- 已实现 Pile mmap idx 解析与前 `512 × 1024` 个 2049-token 样本覆盖验证。
- 已实现独立 mmap reader 与官方 `utils/batch_viewer.py` reader 对 step 0、1、511 的数组和 SHA-256 对比。
- 已实现 4 卡 NCCL/BF16 all-reduce、160M BF16 forward/backward、本地模型加载、SST-2/WikiText 离线读取和单 token verbalizer 验收。

## 当前阶段

- 准备阶段：进行中。
- Pile shard0+idx：后台下载中。
- wheelhouse：脚本已修正，等待重新执行。
- 模型与小数据资产：下载脚本已准备，等待执行。
- 最终离线验收：等待上述三类资产到齐。

## 遗留问题与后续动作

1. 重新运行修正后的 wheelhouse 解析与传输，生成 `environment/requirements.lock` 并在服务器创建大盘 venv。
2. 下载并传输模型、SST-2、WikiText 和官方源码快照。
3. 等待 shard0 与 idx 完成并核对固定 SHA-256。
4. 运行 idx 覆盖、官方 batch viewer、4 卡 NCCL、BF16、模型和数据集完全离线验收。
5. 生成 `prefix_coverage.json`、`batch-viewer-comparison.json`、`offline-smoke.json`、`environment.txt` 和最终 `READY` 标记。
6. 将本日志和仓库准备脚本同步到服务器仓库及 Git 远端；本轮不提交大文件、模型、数据、wheel 或签名 URL。

## 时间估计

- 从 Pile 下载启动计，正常完成总计约 7–9 小时。
- 若 Xet URL 刷新或 Range 不稳定并回退 SCP，总计约 9–12 小时。
- 主要剩余时间由约 31.8 GB Pile 前缀传输决定，其余准备与验收并行执行。

## 后续实施记录（13:03）

### 新完成

- Pile shard0 最近一次检查已达到 2,501,066,752 bytes，后台任务状态为 `Running`。
- pip 对 Linux CPython 3.12 的完整解析已成功，共生成 69 个精确版本 pin。
- 69 个 Linux wheel 已全部下载，wheelhouse tar 为 1,032,990,208 bytes，正在通过 SCP 传入服务器。
- 模型和数据资产已全部下载到 lab-pc：
  - Pythia 160M deduped step0 `model.safetensors`：649,308,728 bytes；
  - Pythia 160M deduped step512 `model.safetensors`：649,308,728 bytes；
  - Pythia 14M step0 `model.safetensors`：28,143,920 bytes；
  - SST-2 train/validation/test；
  - WikiText-103 raw train 两片及 validation/test；
  - 对应 config、tokenizer 和 special token 文件。
- 官方 Pythia 源码固定为 commit `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`；源码 tar 为 10,761,302 bytes。
- 最小资产 tar 为 1,662,750,720 bytes，正在通过 SCP 传入服务器。

### 新发现与修复

1. **PowerShell 5.1 无法解析大型 pip report**
   - 现象：`ConvertFrom-Json` 在约 1.17 MB 的 pip report 中途失败。
   - 处理：新增 `lock_from_pip_report.py`，使用 Python 标准库解析并校验核心 pin。
   - 结果：精确锁文件已生成，包含 69 个固定分发包。
2. **GitHub 前端文件域不可达**
   - 现象：lab-pc 可访问 GitHub API，但 `github.com/.../archive/...` 下载反复超时或连接重置。
   - 处理：改用同属 GitHub 官方的 `codeload.github.com` 固定 commit tar 地址。
   - 结果：源码快照下载成功，commit 不变。
3. **后台解析脚本不稳定**
   - 现象：完整 wheelhouse 脚本在解析后提前退出，wheel 下载命令单独执行成功。
   - 处理：拆出 `lab_finish_wheelhouse.ps1`，只负责对已下载 wheel 做哈希、打包、传输和服务器离线安装，缩小失败面。

### 当前遗留

- 等待两个 SCP 归档传输完成，并在服务器校验、解包和离线安装。
- 等待 Pile shard0 与 idx 下载完成。
- 资产和 venv 到齐后先做不依赖 Pile 的模型、数据和 NCCL smoke test；Pile 到齐后做最终 idx/batch viewer 验收。
