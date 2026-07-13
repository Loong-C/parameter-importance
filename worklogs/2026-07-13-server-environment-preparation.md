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

## 后续实施记录（14:01）

### 新完成

- Git 本机主分支的第一批准备脚本、环境声明、计划书和中文工作日志已形成提交 `96ec8bf80a6cf570828f40e2b143bd9def5fc1c5`，已推送至远端；同一提交已同步到服务器仓库。
- 模型、数据集和源码资产归档已传入并安装到服务器大盘：
  - 模型目录约 1.3 GiB；
  - SST-2 目录约 3.2 MiB；
  - WikiText-103 raw 目录约 301 MiB；
  - Pythia 固定源码快照解包目录约 73 MiB；
  - 资产清单位于大盘 `manifests/asset-manifest.json`。
- 服务器基础 wheelhouse 已有 70 个 wheel、约 989 MiB，且每个文件均已由 SHA-256 manifest 校验。
- 已补齐并锁定 Linux 专属 CUDA 运行时依赖；最终精确锁文件包含 88 个分发包。
- lab-pc 的最终 wheelhouse 共 88 个 wheel、3,741,623,140 bytes；其中新增 Linux CUDA/NCCL/Triton wheel 18 个、2,705,361,251 bytes。
- 已生成只含新增 18 个 wheel 的增量 tar（2,705,388,544 bytes），并注册用户级计划任务 `CjlLinuxRuntime` 后台传入服务器；不重复传输已有 70 个 wheel。
- Pile shard0 最近一次检查约为 7.8 GiB，服务器直下仍持续增长。

### 新发现与修复

1. **跨平台 pip 解析遗漏 Linux 条件依赖**
   - 现象：在 Windows 上以 `--platform` 解析出的首版报告没有包含 PyTorch 元数据中受 Linux marker 控制的依赖，服务器 `--no-index` 安装先后报告缺少 `hf-xet`、CUDA Toolkit 组件、cuDNN、NCCL、NVSHMEM 和 Triton。
   - 处理：直接检查 PyTorch wheel 元数据，并使用 PyPI、PyTorch CUDA 12.6 官方索引和 NVIDIA 官方索引，覆盖 `manylinux_2_28`、`manylinux_2_27`、`manylinux_2_18`、`manylinux_2_17` 与 `manylinux2014` 标签解析下载。
   - 结果：锁文件从 69 项扩充到 88 项，包含 `hf-xet==1.5.1` 及 18 个 Linux 运行时包；新增增量安装脚本会在服务器重新校验完整 88-wheel manifest 后才创建 venv。
2. **SHA-256 manifest 的 CRLF 文件名尾部问题**
   - 现象：lab-pc 生成的 TSV 使用 Windows 行尾，服务器 Bash 首次读取时把 `\r` 保留在 wheel 文件名末尾，造成假性的“文件不存在”。
   - 处理：服务器安装脚本在每行读取后显式剥离文件名尾部 `\r`。
   - 结果：70 个基础 wheel 的大小和 SHA-256 均通过验证。
3. **避免重复传输完整 wheelhouse**
   - 现象：完整最终 wheelhouse 已达约 3.74 GB，若重新传送会浪费已完成的约 1 GB SCP。
   - 处理：新增 `lab_finish_linux_runtime.ps1` 与 `server_install_linux_runtime.sh`，只传 18 个新增 wheel，在服务器与原 70 个合并，并用覆盖全部 88 个文件的新 manifest 重新验真。

### 当前阶段与遗留

- 小型模型、SST-2、WikiText 与官方源码：已安装，等待 venv 完成后做完全离线读取与模型计算验收。
- Python venv：新增 CUDA wheel 正在 SCP；传完后自动执行 `--no-index` 安装与 `pip check`。
- Pile：shard0 后台直下进行中；完成 30,000,000,000-byte 文件及 SHA-256 后将自动继续 1,757,184,042-byte idx。
- 最终遗留：4 卡 NCCL/BF16、160M forward/backward、单 token verbalizer、Pile idx 覆盖与官方 batch viewer 对比，以及最终 `READY` 标记。

## 后续实施记录（14:30）

### 元数据复核与最终依赖修正

- 直接读取 `torch-2.12.1+cu126` 和 `cuda_toolkit-12.6.3` wheel 的 `METADATA`，逐项核对 Linux marker 与被选择的 extras。
- 复核发现 `cuda-toolkit[cufile]` 在 Linux 上还要求 `nvidia-cufile-cu12==1.11.1.6`；Windows 主机上的跨平台 pip 解析没有应用这一条 `sys_platform == 'linux'` marker，因此 88 项集合仍少 1 项。
- 已从 NVIDIA 官方索引下载固定版本 cuFile wheel：
  - 文件：`nvidia_cufile_cu12-1.11.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl`；
  - 大小：1,142,103 bytes；
  - SHA-256：`cc23469d1c7e52ce6c1d55253273d32c565dd22068647f3aa59b3c6b005bf159`；
  - 最终精确锁文件：89 项。
- 新增 `lab_finish_cufile.ps1` 和 `server_install_cufile.sh`：等待原 18-wheel 大归档传完后，只补传这个单独 wheel，生成覆盖全部 89 个 wheel 的 SHA-256 manifest，再清理重建 venv、离线安装并运行 `pip check`。
- lab-pc 用户级计划任务 `CjlCuFile` 已启动并处于等待状态，避免与 `CjlLinuxRuntime` 的 2.7 GB SCP 争抢带宽。

### 下载恢复情况

- Pile 第一次签名 URL/SSH stdin 会话在 14:15 左右被连接重置；安全下载器没有混用对象或丢弃 `.part`。
- `CjlPilePrefix` 已按设计进入第 2 次 URL 刷新并从现有字节续传，最近一次检查 shard0 为 9,466,368,000 bytes。
- 这次恢复验证了 URL 刷新和断点续传路径确实在真实大文件下载中生效；日志没有记录完整签名 URL。

### 安装后资产复验

- 新增 `verify_asset_manifest.py`，用于对已经复制到最终大盘目录的模型、数据集和源码重新计算大小与 SHA-256，而不是只依赖解包暂存目录中的第一次校验。
- 服务器复验通过：24 个文件、合计 1,662,669,829 bytes 全部匹配。
- 复验报告已写入大盘 `manifests/asset-install-verification.json`，状态为 `ok`。
- 对全部 CUDA/NCCL/Triton wheel 的 `Requires-Dist` 再次逐条审计后，确认 cuFile 是唯一遗漏的强制依赖；其余未下载条目均属于没有选择的 `all`、`build`、`test` 等 extras。

### 当前阶段

- Git 远端和服务器仓库均已快进到 `fd6b69c413da0b3d5607388601489e7748690de3`；该提交含上一阶段详细中文日志。
- 18-wheel 增量归档正在传输，完成后预期首次安装会因尚未到达的 cuFile 报告缺包；随后 `CjlCuFile` 会自动补齐第 89 个 wheel 并完成最终安装。这是受控的两段式恢复，不需要重新传输 2.7 GB。
- Pile 与 wheelhouse 仍并行传输；最终离线计算验收尚未开始。

## 后续实施记录（15:17）

### Python 环境与核心离线验收完成

- 18 个 Linux runtime wheel 的 2,705,388,544-byte 增量归档已传完并与基础 70 个 wheel 合并。
- cuFile 跟进归档为 1,156,096 bytes，补传后服务器最终 wheelhouse 为 89 个文件。
- 最终 venv 位于大盘 `envs/parameter-importance`；`pip freeze` 包含 `nvidia-cufile-cu12==1.11.1.6`，`pip check` 返回 `No broken requirements found.`。
- 完全清空代理、设置 Hugging Face/Transformers/Datasets 离线变量后，核心验收全部通过：
  - PyTorch `2.12.1+cu126`，CUDA `12.6`，cuDNN `91002`；
  - Transformers `4.57.6`，Datasets `4.8.5`；
  - 4 卡 NCCL BF16 all-reduce：world size 4，求和结果 10.0；
  - A100-SXM4-80GB 共 8 张，驱动 `575.57.08`，BF16 支持为真；
  - Pythia 160M step0 BF16 forward/backward 有限，160M step512 和 14M step0 均能从本地 safetensors 加载；
  - SST-2 行数为 train 67,349、validation 872、test 1,821；
  - WikiText-103 raw 行数为 train 1,801,350、validation 3,760、test 4,358；
  - verbalizer ` negative` 为单 token 4016，` positive` 为单 token 2762。
- 已生成 `manifests/CORE_READY`、`core-pip-check.txt`、`nccl-smoke.txt`、`offline-smoke.json`、`offline-assets-output.txt` 和 `environment-core.txt`。

### Pile 并发写入故障、数据恢复与下载器加固

1. **故障现象**
   - 第一次 SSH 控制会话在 14:15 断开后，服务器端旧 curl 成为孤儿并继续写 `.part`；lab-pc 随即刷新 URL，启动了第二个服务器 curl。
   - 两个 curl 同时写同一 shard0 `.part`，因此并发开始后的文件尾部不能信任。
2. **保守恢复**
   - 明确终止了两组只属于 shard0 的 Bash/curl 进程，确认没有残留写入者。
   - 14:00 的只读检查已确认文件至少显示为 7.8 GiB，早于 14:15 并发开始；选择更保守的十进制 8,000,000,000 bytes 作为安全点。
   - 将 `.part` 从 10,216,787,968 bytes 截断至 8,000,000,000 bytes；丢弃不可信尾部，不尝试猜测或保留并发写入数据。
3. **永久修复**
   - 每个目标对象新增独立 `flock`，确保同一 `.part` 只有一个写入者；新进程会等待旧进程释放锁。
   - curl 改为经 config stdin 接收 URL；进程 argv 仅显示 `--config -`，不再出现签名 URL。
   - 公网 DNS 改为查询 `223.5.5.5`、`119.29.29.29`、`114.114.114.114` 三家，只采用至少被其中两家共同返回的 CDN 地址。
   - lab-pc 的自动刷新上限从 12 次提高到 100 次，保留大小、SHA-256、HTTP 206、Content-Range 和对象元数据检查。
4. **恢复验证**
   - 新下载器从 8,000,000,000-byte 安全点恢复，观察到只有 1 个真实 curl；curl argv 不含 URL。
   - 新下载器恢复后 shard0 已重新增长至 8.12 GB 以上；最终 SHA-256 未通过前不会原子改名。

### 文档与当前遗留

- `Agent/remote_access.md` 已补充单写锁、三家 DNS 任意两家交叉确认、URL 不进入进程参数，以及并发写入后的安全截断恢复规则，并同步到服务器仓库的忽略目录。
- 当前只剩 Pile shard0、完整 idx、二者固定 SHA-256、前 `512 × 1024` 样本覆盖验证和官方 batch viewer step 0/1/511 对比。

## 后续实施记录（15:46）

### 无人值守收尾链路

- 新增 lab-pc 用户级计划任务 `CjlPileSupervisor`，最长运行 18 小时：
  - 每分钟确认 shard0 与 idx 是否最终就位；
  - 若 `CjlPilePrefix` 意外退出且文件未齐，则安全重启任务；
  - 服务器对象级 `flock` 继续保证只有一个写入者。
- supervisor 在 shard0 和 idx 都完成固定大小/SHA-256 校验并原子改名后，会自动调用 `server_finalize.sh`。
- `server_finalize.sh` 新增单实例验收锁，并会自动完成：安装后资产复验、idx 覆盖、官方 batch viewer step 0/1/511 对比、`pip check`、4 卡 NCCL/BF16、模型和数据集离线验收，以及最终 `manifests/READY`。
- 当前任务已增加每小时心跳 `NLP环境准备最终验收`：在下载期间检查字节增长与单写状态；`READY` 生成后回到本任务读取报告、追加最终中文日志、提交 Git，并确认本机/服务器/Git 远端三处日志一致后停用自身。
- 自动 finalizer 与心跳配置时：
  - shard0 `.part` 为 11,086,516,224 bytes；
  - idx 尚未开始；
  - 服务器真实 curl 数为 1；
  - 服务器与 Git 远端代码提交均为 `3c011b24566dc2ecdda1c92fe3d9266c8738c4ce`。

### 剩余时间估计

- 以加固恢复后的约 1.6–1.7 MB/s 持续速度估计，shard0 还需约 3.1–3.4 小时。
- idx 约 1.76 GB，预计再需约 18–25 分钟；大小/SHA-256、覆盖、batch viewer 与重复核心验收预计 20–40 分钟。
- 正常情况下最终 `READY` 预计在 19:30–20:15（Asia/Shanghai）之间生成；若 URL/SSH 再次刷新，100 次重试与 supervisor 会自动恢复。
