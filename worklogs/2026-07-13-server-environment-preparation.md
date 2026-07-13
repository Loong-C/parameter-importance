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

## 后续实施记录（17:43）

### 第二次 SSH 断开与 supervisor 修复

- 原第 12 次控制会话在 17:14 再次发生 SSH connection reset，lab-pc 的 `CjlPilePrefix` 因旧脚本已到刷新上限而转为 `Ready`/退出码 1。
- 服务器端旧 curl 没有退出，继续持有对象锁并安全下载；shard0 从上一轮 15,394,906,112 bytes 持续增长至 21,392,900,096 bytes，没有出现第二个 curl。
- `CjlPileSupervisor` 没有及时重启控制任务，原因是它的完成状态 SSH 探测没有显式连接与保活超时，控制链路异常时可能长期卡住。
- 已给 supervisor 的状态探测和 finalizer 调用增加：
  - `ConnectTimeout=20`；
  - `ServerAliveInterval=15`；
  - `ServerAliveCountMax=2`。
- 已重新启动采用 100 次刷新上限的 `CjlPilePrefix` 与修复后的 supervisor；两项任务均为 `Running`。

### 单写锁复核

- 进程树显示：
  - 旧下载器一个父 Bash、一个执行 curl 命令替换的子 Bash，以及一个真实 curl；
  - 新控制任务一个 Bash 和一个 `flock -w 3900` 进程，仅等待旧下载器释放对象锁。
- 真实 curl 数始终为 1；新任务尚未进入 curl，不会写 `.part`。
- 复核时 shard0 已增长至 22,895,489,024 bytes，`READY` 尚未生成，idx 尚未开始。

## 后续实施记录（19:53）

### shard0 完成与 idx 接续

- shard0 已完成固定的 30,000,000,000 bytes 下载，并通过上游 SHA-256 `1ce355bd2683627d0ff689f8578115cf3df84bd1edf3410e6aca9705d31fc6ea` 后原子改名为 `document-00000-of-00020.bin`；这也从最终结果上验证了从 8,000,000,000-byte 保守安全点恢复的内容完整性。
- idx 初次接续时，三家公网 DNS 的短时响应没有形成任意两家交集，下载器按安全策略拒绝选用单一来源地址，没有写入错误对象。
- 后续复测中 `223.5.5.5` 与 `114.114.114.114` 再次返回共同地址；不记录具体 CDN IP。idx 已在第 6 次 URL 刷新中恢复，复核时 `.part` 为 227,364,864 bytes。
- 服务器复核显示恰有 1 个 idx curl 写进程，且对象锁由 1 个下载器持有；`CjlPilePrefix` 与 `CjlPileSupervisor` 均为 `Running`。

### SSH 状态探针硬超时

- 仅设置 `ConnectTimeout` 与 SSH keepalive 不能限制“连接已建立但远端命令不返回”的总时长；此前 Pile 控制脚本与 supervisor 都可能卡在 `test -f` 状态探针。
- `lab_prepare_pile_prefix.ps1` 与 `lab_supervise_pile.ps1` 现使用独立 `ssh.exe` 进程执行只读状态探针，并设置 45 秒总等待上限；超时只终止该探针，不终止服务器上持锁的真实下载器。
- `lab_hf_broker.ps1` 的长时下载 SSH 增加连接和保活参数，但不设置总时长上限，避免正常的大文件传输被误杀。
- 修复后已实际观察到 DNS 暂无双源交集时，idx 尝试能从第 1 次继续推进到第 6 次，不再永久挂起；整个日志仍未包含签名 URL 或临时 CDN IP。

### 当前阶段

- 最小闭环的 Python 环境、模型、小数据集、源码快照与核心离线验收均已完成；Pile shard0 已完成。
- 当前只剩 idx 下载及其固定哈希、前缀覆盖、官方 batch viewer 对比和最终重复离线验收；成功后由 supervisor 自动生成 `manifests/READY`。

## 最终验收记录（20:18）

### Pile 最小前缀完整到位

- `document-00000-of-00020.bin`：30,000,000,000 bytes，SHA-256 `1ce355bd2683627d0ff689f8578115cf3df84bd1edf3410e6aca9705d31fc6ea`。
- `document.idx`：1,757,184,042 bytes，SHA-256 `1d9fdd760295eb2007a4874440b27c559ca722239fa2814aa8a2ee6724b7852f`。
- 两个对象都在固定大小和固定上游哈希通过后才从 `.part` 原子改名；服务器上已无 Pile 写进程。
- `prefix_coverage.json` 解析结果：idx version 1、dtype code 8、每 token 2 bytes、146,432,000 个 sequence、1 个 document；最小闭环要求 524,288 个、每个 2,049-token 的样本，最大结束偏移为 2,148,532,224 bytes，小于 30,000,000,000-byte shard0，`covered=true`。

### 官方 batch viewer 对比

- 官方源码固定为 Pythia commit `a19eecb807ec2c79a39ebf18108816e6ffffc1d5`。
- step 0：形状 `1024 × 2049`，独立 reader 与官方 reader 的 SHA-256 均为 `7b5299e7ed772054885b16d1d61778c5f437a5fb196aeffab5dba4577977c562`。
- step 1：形状 `1024 × 2049`，两者 SHA-256 均为 `2f08a61f39b39d7fea82babb2908899434648ef0e293ba177f5e8e79a3aa6402`。
- step 511：形状 `1024 × 2049`，两者 SHA-256 均为 `24718b83391af3c41c5c87802014b436c317f7fedf404d535ef0b5819630ebfb`。
- 三个目标 step 均为 `equal=true`，`batch-viewer-comparison.json` 总状态为 `ok`。

### 完全离线环境复验

- finalizer 清空全部代理变量、设置 Hugging Face/Transformers/Datasets 离线变量后运行，没有触发网络下载。
- 安装后资产 manifest 再次通过；24 个模型、数据和源码文件共 1,662,669,829 bytes 与清单一致。
- `pip check`：`No broken requirements found.`。
- 4 卡 NCCL/BF16：`NCCL_OK world_size=4 sum=10.0`。
- `offline-smoke.json`：PyTorch `2.12.1+cu126`、CUDA `12.6`、BF16 支持为真；SST-2 行数 67,349/872/1,821，WikiText-103 raw 行数 1,801,350/3,760/4,358；` negative` 和 ` positive` 均为单 token，ID 分别是 4016 和 2762；总状态为 `ok`。
- `environment.txt`：Python `3.12.3`、Transformers `4.57.6`、Datasets `4.8.5`、cuDNN `91002`；8 张 `NVIDIA A100-SXM4-80GB`，每张 81,920 MiB，驱动 `575.57.08`。

### READY、竞态保护与阶段结论

- 最终 `manifests/READY` 于 `2026-07-13T12:17:39+00:00`（北京时间 20:17:39）生成；`prefix_coverage.json`、`batch-viewer-comparison.json`、`offline-smoke.json`、`environment.txt` 均已保存在服务器大盘 manifests 目录。
- idx 完成后，lab→server 的短状态探针仍有一次 45 秒超时，supervisor 因而重复启动了只做“对象已完成”检查的 prefix 任务。随后自动 finalizer 与本机兜底启动恰好重合；服务器 `finalize.lock` 只允许第一实例执行，第二实例以“already running”退出，没有并发验收或报告覆盖。
- `READY`、全部报告、空闲的验收锁和归零的 finalizer 进程共同确认第一实例正常完成；单行“already running”只是第二实例被正确拒绝，不是验收失败。
- **环境准备阶段已完成**：后续可以直接在服务器仓库编写代码并开始最小闭环实验，不需要再下载计划范围内的 Python 包、Pythia 160M step0/step512、14M step0、SST-2、WikiText-103、Pythia 源码或 Pile shard0+idx。
- 尚未纳入本轮范围的资产仍是其余约 570 GB Pile 分片、410M、MNLI 和 RTE；只有实验游标越过已验证前缀或研究范围明确扩展时才需要另行准备。

### Git 同步待收尾

- 本机与服务器仓库均已到提交 `0d8801e80c3bac2342b7fa101363bdfad44b0680`，且本机/服务器上一版中文日志及 `Agent/remote_access.md` 的 SHA-256 已核对一致。
- GitHub HTTPS 读请求可成功，但本轮多次写请求分别遇到 connection reset、empty reply 和 443 connect timeout；远端 `main` 暂停在 `c6b3e5011c6cbf066499e928fd7770274e8cdc41`。最终日志提交会在链路恢复后继续推送，不把这一外部网络波动误记为环境验收失败。

## 最终同步结果（20:26）

- GitHub HTTPS 写通道恢复后，待推提交已成功把远端 `main` 从 `c6b3e5011c6cbf066499e928fd7770274e8cdc41` 推进到包含最终验收记录的 `3b15b7911069a32076a177ac2ef69f25ccd158e9`；此前的多次失败均未造成半提交或远端分叉。
- 本节所在的最终日志提交继续直接推送 `main`，不创建额外分支或 PR；服务器仓库通过经过 `git bundle verify` 的 bundle 做 `--ff-only` 快进，不使用 `reset --hard`，不接触大盘数据和模型。
- 收尾验收以本机 HEAD、服务器 HEAD、GitHub `refs/heads/main` 三者完全相同，以及本机/服务器中文日志 SHA-256 完全相同为完成标准；满足后停用每小时 `nlp` 心跳。
