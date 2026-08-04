# 2026-08-03 Stage 0 剩余任务实施

- 任务范围：按 `plan/stage0/` 的依赖与 Gate 要求，从 S0.4 继续完成至 S0.12；本轮只建设和验收实验基础设施，不把 Stage 0 的 synthetic/训练 smoke 解释为参数重要性算法或科学结论。
- 当前状态：进行中
- 工作分支：`feat/stage0-completion`

## 2026-08-03 14:10 CST — 恢复 S0.4–S0.12 执行

### 目标与范围

- 本阶段要完成什么：重新核对现有代码、测试、服务器资产与 Gate 证据，从 G3 开始逐项形成 G3–G10 的当前有效证据；每个可独立复核的 Gate 完成后测试、记录、提交、推送并同步。
- 不在本阶段处理什么：不修改 SSH/反向隧道拓扑，不绕过资产 manifest 或硬件 Gate，不使用 `.part`/活动锁作为输入，不执行系统级驱动/CUDA 改造，不把本机 CPU/synthetic 结果冒充服务器 CUDA/NCCL 或正式资产证据。
- 用户决定：用户在当前任务中要求继续并完成 Stage 0 剩余任务，因此 2026-08-03 早先记录的“S0.4–S0.12 暂停”从本条开始解除。

### 实际状态

- 从干净提交 `11a3f9072b01ab122d3bd3dc947f637dcfe8d755` 创建专用分支 `feat/stage0-completion`。
- 本机与 GitHub 原功能分支起点一致；`origin/main` 仍为历史基线，不作为本轮工作分支。
- 最新封存证据保持 G0、G1、G2 为 `PASS`；S0.4–S0.12 尚无 formal Gate 通过声明。
- `lab-pc` 与 `sophgo13-via-lab` 当前均在 SSH banner 交换阶段超时。按 `Agent/remote_access.md` 未修改连接参数或启动替代拓扑。
- 本机存在预装 ToDesk 客户端，但其窗口无法由受控 Windows 应用接口可靠捕获；未尝试自动化认证、未读取连接凭据、未通过该客户端执行远端操作。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| 本机 Git 基线 | `git status -sb`、`git rev-parse HEAD`、`git ls-remote` | 工作树干净；本机/GitHub功能分支为 `11a3f907` | 本日志；当前终端记录 |
| 服务器 SSH 复核 | 两个规定别名、`BatchMode=yes`、有限连接超时 | `BLOCKED`：SSH banner 交换超时 | 本日志；当前终端记录 |
| 现有 formal 证据盘点 | `reports/stage0/`、工作日志、任务目录 | 只有 G0–G2 当前报告；G3–G10 未形成 | `reports/stage0/` |
| 本机完整测试基线 | `.venv`、`PYTHONPATH=src`、`pytest -q` | 运行中，结果待追加 | 本日志后续条目 |

### 问题、原因与风险

- S0.4 的真实模型/数据盘点以及 S0.6–S0.11 的服务器/GPU Gate 依赖 SSH 链路恢复；在链路恢复前只能完成本机代码、合同和负向测试，不能宣称相应 Gate 通过。
- G1-D 的单盘风险接受只覆盖 Stage 0 可再生 smoke，有效至 2026-08-18 23:59 CST 或 Stage 4 开始前（先发生者）；本轮不得扩展其范围。
- 当前仓库存在完整的本机 run-ready 实现，但本机验证报告明确无 formal 资格；后续必须逐 Gate 生成真实服务器证据。

### 下一步

- 完成 S0.4–S0.12 的代码/证据缺口审计。
- SSH 链路恢复后先只读复核推荐环境、GPU qualification、服务器仓库、活动下载和资产目录，再执行 S0.4。
- 在服务器不可达期间，补齐不依赖服务器的确定性、拒绝路径和报告生成缺口。

## 2026-08-03 14:42 CST — SSH 恢复、服务器只读复采与未跟踪文件封存

### 目标与范围

- 响应用户“SSH 应可连接但较慢”的信息，仅延长两条既定别名的连接等待时间；不修改 SSH 配置、跳板拓扑或反向隧道。
- 在执行 S0.4 前复采服务器仓库、存储、GPU、资产与活动下载状态，并先保存服务器仓库内已知未跟踪恢复文件。

### 实际状态

- `lab-pc` 与 `sophgo13-via-lab` 均连接成功，典型握手约 20 秒。
- 服务器仓库原状态为 `feat/stage0-infrastructure@5cc53930`，存在 6 个未跟踪的 GPU 恢复脚本/授权报告；其中 3 个与本机相同，3 个为本机已归档版本的较早版本。
- 按服务器写回流程，把 6 个文件与 `worklogs/2026-07-19-stage0-execution.md` 的封存记录提交为 `fc5a4fc8fe8630c1be2655a622b7e940ca8a44b0`（`chore: preserve server GPU recovery files`）。服务器仓库使用 repo-local Git identity；未修改全局 Git 配置。
- 生成并验证增量 bundle `server-preserve-fc5a4fc-incremental.bundle`，前置提交为 `5cc53930a3f745fbd3e9ea4e171bd0773172984a`，大小 22,911 字节；已回传本机并验证完整。
- DATA_ROOT 为 `/home/sophgo13/cjl/storage/parameter-importance`，底层 `/dev/nvme0n1` 为 ext4/rw；可用空间约 2.8 TiB、inode 使用约 1%，目录权限为 `sophgo13:sophgo13 0750`。
- 四张白名单 A100-SXM4-80GB 均可见，检查时显存使用为 0 MiB、无 compute process、ECC volatile/aggregate 为 0。
- 当前模型目录只有 Pythia 14M step0、31M-deduped step0、160M-deduped step0/step512；Pythia 410M 缺失。数据目录只有 SST-2、Pile 与 WikiText；MNLI、RTE 缺失。
- Pile 活动对象为 `document-00009-of-00020.bin.part`，大小 22,882,025,472 字节，最后写入时间为 2026-07-19 13:32:38 UTC。精确文件持有者检查为空，且没有 curl/wget/aria2/python 下载进程；因此判定下载当前停滞，不删除锁、不恢复下载，也不把 `.part` 作为资产输入。

### 实验与验证

| 项目 | 命令/配置 | 结果 | 证据路径 |
|---|---|---|---|
| SSH 复核 | 两个既定 alias，`ConnectTimeout=60` | `PASS`；两端均成功登录 | 本日志；当前终端记录 |
| 服务器未跟踪文件语法/内容检查 | `bash -n`、JSON parse、本机/服务器逐文件比较 | `PASS`；确认 6 个目标及版本差异 | `artifacts/server-untracked-audit-20260803/`（本机忽略目录） |
| 服务器保护提交 | Git staged diff、commit、bundle verify | `PASS`；提交 `fc5a4fc`，增量 bundle 验证通过 | DATA_ROOT `tmp/`；本机忽略目录 |
| Stage 0 定向回归 | 11 个资产/合同/runtime/GPU promotion 测试文件，workspace `--basetemp` | `142 passed, 7 skipped`；52.66 秒 | 本日志；当前终端记录 |

### 问题、原因与风险

- 首次完整 bundle 为 15,550,450 字节，经慢速中继传输长期停滞；未使用其不完整本机副本。随后改为以已存在共同前置提交为 prerequisite 的 22,911 字节增量 bundle，既保留提交对象又避免无意义重复传输。
- 默认 pytest 临时目录因 Windows 权限导致 setup error；改用工作区内 `--basetemp` 后定向测试通过。7 个 skip 分别来自 Windows 目录 symlink 权限和该 Windows Torch wheel 不支持所需 Gloo device，不能作为服务器 CUDA/NCCL Gate 证据。
- S0.4 正式 G3 仍为 `NOT_RUN`：缺少 410M、MNLI、RTE，31M 旧 manifest 需要 BOM 修复，Pile 当前格式/覆盖合同尚未接入 formal provider。

### 下一步

- 将服务器保护提交归并到 `feat/stage0-completion`，冲突时保留已审计的本机新版本内容，同时保留服务器提交历史和日志追加。
- 冻结 S0.4 资产矩阵、序列/目标 token 语义与最大游标预算；先补 manifest、语义审计和 Pile/GLUE 构建合同，再取得/发布正式资产。

## 2026-08-03 16:30 CST — S0.4 G3 冻结合同与安全获取入口

### 目标与范围

- 先冻结 Stage 1/2/4/5/6 实际需要的模型、tokenizer、Pile 游标与 GLUE split，证明本阶段所需 Pile 前缀后再决定下载范围。
- 建立 G3 manifest、资格准入、DATA_ROOT 逻辑布局和缺失对象获取入口；正式 Gate 必须重放 manifest、qualification、文件哈希和语义合同，不能接受 legacy `READY` 或任意单项 PASS 声明。
- 下载入口只保存稳定 source ID、固定 revision、大小和 SHA-256；运行时 URL 只在内存构造，不进入 argv、配置、报告或 manifest。

### 冻结决定

- Pile 每个原始 record 为 2049 token；运行输入为 `[0,2048)`，目标为 `[1,2049)`，attention mask 全 1，loss adapter 为 `pre-shifted-next-token-cross-entropy-v1`，禁止模型内部再次位移。
- 冻结必需游标为 `[0,1048576)`：debug 8192、train 640000、validation 64000、Stage 2 sampling universe 262144、Stage 3 probe 65536、reserve 16896。必需原始 token 为 2,148,532,224，实际目标 token 为 2,147,483,648。
- `document.idx` 前 1,048,576 项长度均为 2049，指针连续且所需最后字节仍位于完整 shard 0；因此 G3 只选择 `document.idx` 与 shard 0，既不需要也不允许为本阶段恢复整套 Pile 下载。
- Stage 6 按总计划统一使用 Pythia 410M deduped step0 作为共同 base initialization；sequence-classification head 由运行时固定 seed 初始化，不把 160M 或一个虚构的独立分类模型资产当作输入。
- GLUE raw 文件路径按各任务 asset root 平铺；上游 `sst2/`、`mnli/`、`rte/` 前缀只属于 source ID，不拼入 DATA_ROOT 下的本地相对路径。
- Pythia 160M step512 的正式身份统一为 `trained_checkpoint`，并要求绑定同仓库 step0 的 parent asset ID 与 `training_step=512`。

### 已实现合同

- 新增 G3 requirements/layout、13 个 URL-free HTTP object spec 与 13 对象下载计划；固定 410M 的 config/weights/tokenizer 以及 MNLI、RTE 全部 raw parquet。
- G3 manifest metadata 现在严格表达 model dtype 分布/最大位置、tokenizer 完整 vocab mapping 与 GLUE padding、Pile index/shard/cursor/causal mapping/reference record、GLUE task contract 与 derived lineage。
- 新增 `downloading → downloaded → verified → ready` 候选链、candidate ID、资格报告及 `resolve_qualified_asset`；G3 READY 不能再由 legacy resolver 使用，也不能绕过 qualification admission。
- 新增 Pythia mmap reader、pre-shift loss 路线，以及支持 sequential、真正有放回、确定性 DDP 无放回分片和严格恢复身份的 mmap provider adapter。
- formal 配置中的逻辑资产 ID 已改为 layout 中的精确名称；provider root/manifest refs 使用实际 `models/`、`datasets/`、`manifests/` DATA_ROOT 路径，不引入 `assets/` 别名。

### 验证与当前状态

| 项目 | 结果 |
|---|---|
| 原 G3 requirements/layout/schema/spec/asset 定向回归 | `118 passed, 1 skipped`；skip 仅为 Windows symlink 权限 |
| URL-free 下载计划与配置逻辑名回归 | `17 passed`、`34 passed` |
| G3 manifest metadata 扩展 | `155 passed, 1 skipped`；兼容回归 `48 passed` |
| Pythia mmap provider/loss 联合回归 | 新 provider `9 passed`；相关联合回归 `43 passed` |

- S0.4/G3 仍为 `IN_PROGRESS`，尚未声明 PASS。剩余工作是把本轮代码提交并以 bundle 快进服务器，下载缺失 410M/MNLI/RTE，构建三个 GLUE derived 资产，生成 13 份 manifest/qualification/离线语义证据并让 G3-S1/S2/S4/S5/S6 全部通过。
- 旧 31M 顶层 BOM manifest 保留为问题证据，不原地覆盖；新的 canonical manifest 将发布到 layout 的独立路径。

## 2026-08-03 17:22 CST — G3 真实资产闭环、网络恢复与合同纠偏（进行中）

### 代码、提交与同步

- G3 冻结控制面及安全 HTTP 获取基础提交为 `dda4fe2c3b2feb7b8763a0d54c425f45e56b62d7`；已推送 GitHub，并以已验证增量 bundle 快进服务器，三端该提交一致后精确清理本轮 bundle。
- 服务器直连公网执行 URL-free 获取入口时，首个 570-byte config 在 6 次有限重试后以 `NETWORK_ERROR` 失败；失败报告为 `$DATA_ROOT/operations/g3-download-dda4fe2.json`，artifact hash `c6dff00550ca9410d0d108a113e6b5c81b763c2157c9e73849a6c3c4d7bd584c`，报告未保存 URL。
- 新增外部字节 relay 与服务器受锁 receiver：提交 `f75cfc98d8b949c4b08eac1cdd373c10b3e23453`；单行协议修复为 `44a63177b735bd2b5ecd7d84817d2bef2348837c`。两次均已推送并以校验哈希的增量 bundle 快进服务器，随后精确删除本机与 `$DATA_ROOT/tmp` 对应 bundle。
- receiver 与原 HTTP 获取共用逐对象 advisory lock、`.part`、固定大小/SHA-256 和 no-clobber hard-link 发布；relay 进程只在内存构造 endpoint，不把 URL 放入 argv、报告或资产目录，也不在本机/lab-pc 落盘资产字节。

### 网络路径实测

- 本机到 `lab-pc`、`lab-pc` 到 `sophgo13` 两段既有 SSH alias 均通过；但 lab-pc 访问固定 Hugging Face revision 时，.NET 无代理请求 30 秒超时，`curl --config -` 45 秒超时。因此未修改 DNS/代理/SSH 拓扑，也未把历史可达性冒充当前 PASS。
- 本机访问同一固定 570-byte config 成功，内存探针得到 HTTP 200、570 bytes 和冻结 SHA-256 `d4c11e84a59c8af4d88446bba53b718f7aef740daa070ded08fd6a9a3aca4fc6`。
- 端到端 relay 首次把该 config 原子发布到 `$DATA_ROOT/models/pythia-410m-deduped-step0/config.json`；协议修复后重跑返回 `already_ready`，证明只读幂等路径有效。
- 410M `model.safetensors` 已启动串行可恢复流式传输；2026-08-03 17:16 CST 只读探针为 `339,231,645 / 1,621,370,224` bytes。连接中断后由同一 sender 等待旧 receiver 退出再按新 offset 续传，没有启动第二个写入器。该对象尚未完成时 G3 仍为 `IN_PROGRESS`。

### Pile 合同纠偏

- 复核历史 `batch-viewer-comparison` 后发现三个冻结哈希键 `0/1/511` 表示 batch index，不是 record index。服务器直接读取 shard0 实测：每批为连续 `1024 × 2049 × uint16` 原始字节，三个 SHA-256 与历史值逐项完全一致；单 record 0/1/511 的哈希均不同。
- `last_required_record_sha256` 独立表示 record `1,048,575` 的 4,098 原始字节；服务器实测仍为 `c70a2422352b10a9027d31ca5a0ed8fe9453218a7a4e1d3e5d675842c7c982c4`。
- requirements 已显式增加 `reference_batch_size=1024`，并绑定官方 reader `EleutherAI/pythia@a19eecb807ec2c79a39ebf18108816e6ffffc1d5`；语义 probe 按 batch 与最后 record 两条独立路径验证。级联后的 requirements/layout/download-plan artifact hash 分别为 `e1e7edff937cec30771e063d460f75fe68c1a66c8d7b60b055fcc4991663835c`、`22d3dfa67cd19904aa3fb3ce187d65ec473e0e92f2277cef08dd854ddbedec14`、`09768b3aa283b25784220f9ec58d07048654013b4ac4703032a543b8ba574cc9`；13 个 immutable HTTP object spec 未改。

### 本地验证与尚未关闭项

| 项目 | 结果 |
|---|---|
| G3 广泛本地回归 | 313 collected；`312 passed, 1 skipped, 0 failed`；唯一 skip 为 Windows 目录 symlink 权限 |
| relay/acquisition/plan/CLI 定向回归 | `32 passed` |
| G3 publication/materializer 定向回归 | `12 passed`；另有真实离线模型加载负载检查 |
| Pile batch 合同、requirements/layout/plan 级联 | `22 passed`，batch-vs-record 专项通过 |

- 静态审查正在补强两条 fail-closed 边界：Gate 必须独立重验 semantic evidence，而不能只信 qualification 中的哈希；GLUE derived lineage 只允许绑定冻结的三个 tokenizer 文件，不能把共享目录中的模型文件或持久 acquisition lock 纳入身份。
- 在上述修复、全部缺失对象、三个 derived GLUE、13 项 READY/qualification、五个 G3 子 Gate 和三类 formal 报告全部通过前，不声明 S0.4/G3 完成。

## 2026-08-03 18:38 CST — 慢速 SSH 断点续传完成与 G3 二次控制审计（进行中）

### 410M 权重传输结果

- 按用户要求重试既有 `lab-pc → sophgo13` SSH 链路；连接可用但握手和吞吐波动明显，未修改 SSH 配置、DNS、代理或跳板拓扑。
- 首轮 relay 在 673,520,845 bytes 后退出并遗留一个仍持锁的本轮 receiver。先只读核对 PID、父进程、完整命令和锁，再只终止该精确 receiver；确认进程退出、锁释放且 `.part` 保留后，才从冻结 offset 重启唯一 writer。
- 重启任务经历三次 `HTTP_TRANSFER_INCOMPLETE` 和一次 SSH banner timeout；每次旧 receiver 退出并释放对象锁后才重连，READY offset 依次为 673,520,845、946,116,123、1,003,467,324、1,486,513,174，没有第二个 writer，也没有把 `.part` 清零。
- 最终发布 `$DATA_ROOT/models/pythia-410m-deduped-step0/model.safetensors`：1,621,370,224 bytes，SHA-256 `25c40c86d28f75cad7f98fca8a4ed4a75485423f14df85832226b4ded66c097d`；receiver 返回 `COMPLETE status=downloaded`，隐藏 `.part` 消失。

### 二次控制审计与修复状态

- relay/receiver 正在补强不完整 HTTP 响应的 partial-byte 保留、所有 READY/COMPLETE/read/write/wait 的单调总时限、严格终态协议、端到端 requirements/layout/plan/spec/commit/route 绑定、发布前后双哈希以及集中 `.part`/锁目录。当前 410M 使用已提交旧协议完成；剩余对象只在新版整体提交并同步后传输。
- Gate/publication 已增加 Pile official-reader oracle 与 31M BOM legacy diagnostic 的原始字节、严格字段和来源提交绑定；existing READY 和 qualification-only 恢复都会在当前环境、零外连守卫下重新执行 semantic probe。该线最终定向回归为 `72 passed`。
- formal G3 runner 已改为从 13 份 qualification 的唯一时间推导稳定 `checked_at`，强制 generator/current HEAD、关键源码 clean/tracked/module-origin、三份正式产物完整来源链，并在旧 PASS 恢复时重新执行 G3、全文件哈希和语义证据；支持第一或第二份产物提交后的幂等续发。定向/邻接回归为 `14 passed`、`53 passed`。
- 新发现的重大控制面缺口是原 publisher 用同一 actor instance 连续生成 fetcher、verifier、gate 三类状态。修复方案已冻结为三个独立边界：逻辑资产 acquisition attestor、独立 verify-only CLI、gate-only materializer；报告需覆盖 `canonical-plan`、`existing-import`、`derived-build` 三种来源，三角色 instance 两两不同并由不可变 ref/hash 交接。
- 正式 runtime 正在改为只消费 Stage0.04 committed `asset_resolution`：配置只保留逻辑 asset ID；Pile 使用 manifest 精确 idx/shard 的 mmap reader；Stage 2/3 绑定冻结 sample IDs；Stage 0/1/4/5 强制 split、global-batch 和预算；GLUE derived 继续完全离线 `load_from_disk`。该线尚在实现和独立审计，未形成 PASS。

### 当前判定与下一步

- S0.4/G3 仍为 `IN_PROGRESS`。410M 权重已完成，但 tokenizer 三文件、MNLI 五文件、RTE 三文件仍待新版 relay 顺序补齐。
- 当前控制面内容哈希已更新为 requirements `591ba7f66cfc006845386a59e5661df0609b4613d7f91a527f490c4a11ddaaa9`、layout `ef347c060af25d7d2e3b1d4c774ff80dbc5ea15e5745749f24b44e0ef8b9ace6`、download-plan `002bef15cede098bb6f75deb2900e456ea3348c1f3ed2512db47729d92b496c2`；这些仍需在实现收敛提交后再次绑定最终 generator Git commit 并级联生成，17:22 节中的旧值仅是当时快照。
- 下一步依次为：完成三阶段生命周期与 formal runtime 测试；形成干净提交并按 GitHub/验证 bundle 快进服务器；顺序补齐剩余对象；执行 canonical acquisition report、独立 verify-only、gate-only materializer、13 项 READY/qualification、五个 G3 子 Gate和正式 resolution；只有全部通过后才进入 S0.5/G4。

## 2026-08-03 23:18 CST — G3 控制面本地收口与完整定向回归

### 范围与审查

- 按当前续跑要求重新完整阅读 `Agent/` 五份运维文档、`plan/general_plan.md`、
  `plan/stage0/` 全部 13 份计划和 `worklogs/` 全部现行日志；继续既有
  `feat/stage0-completion` 工作树，不覆盖或拆散 18:38 CST 之后已经形成的未提交实现。
- 完成三角色 G3 生命周期、gate-only materializer、正式 runtime resolution、Pile mmap、
  GLUE derived builder、正式 task runner 与跨 Stage 1–8 配置接线的本地审查和回归。
- 远端复采后，本机与 `origin/feat/stage0-completion` 仍为 0/0 分叉；本节没有把本机
  测试冒充服务器资产 Gate，也没有执行新的资产下载或 GPU 运行。

### 本轮修复

- `resolve_source_git_commit` 改用仓库规范允许的一次性 `safe.directory`，使受控沙箱用户
  可以读取已严格解析的 Git 源身份，且不修改全局 Git 配置。
- 两份新增 G3 lifecycle schema 补齐项目标准绝对 `$id`；schema 重放入口恢复通过。
- 测试不再引入锁文件之外、项目明确不依赖的 `jsonschema` 包，改为复用仓库自身的
  schema 源文档校验并继续由领域 validator 验证实例语义。
- 将缺失 resolution 的断言更新为实际更早、更精确的 fail-closed 错误
  `G3_RUNTIME_REF_MISSING_OR_ESCAPE`。
- 最初把 pytest `--basetemp` 放在源码仓库内，正式 runner 正确拒绝了 workspace/source
  重叠；改用独立系统临时目录后相应用例全部通过。仓库内由本轮创建的 14 个
  `.codex-stage0-*` 临时目录已逐个解析并确认位于仓库根下后精确清理。

### 验证

| 检查 | 结果 |
|---|---|
| 资产合同、获取、manifest 与 relay | `217 passed, 1 skipped`；skip 仅为 Windows 目录 symlink 权限 |
| 配置、preflight、provider 与任务 runner 邻接回归 | `90 passed` |
| 独立 acquisition/verify-only/gate 生命周期 | `16 passed` |
| G3 发布、崩溃恢复、网络阻断与 provenance 负向路径 | `13 passed` |
| 五个 G3 子 Gate 汇总与证据重放 | `24 passed` |
| 正式 runtime resolution | `16 passed, 1 skipped`；skip 仅为 Windows 文件 symlink 权限 |
| Stage 0.4 正式 runner 与 Git 来源兼容性 | `29 passed` |
| gate-only materializer CLI | `7 passed` |
| GLUE builder、Pile mmap reader/provider | `43 passed` |
| 仓库全部 schema 源文档重放 | `1 passed` |
| Python 静态编译 | `python -m compileall -q src ops tests`，退出码 0 |
| Git 大文件/禁止类型守卫 | `git-guard: PASS`；退出码 0 |
| 空白检查 | `git diff --check`，退出码 0；仅现有 Windows LF/CRLF 提示 |

上述互不重复的定向集合合计 `456 passed, 2 skipped, 0 failed`。两个 skip 都有独立
Linux/服务器补跑要求，不能计作 G3 formal PASS。

### 当前 Gate 与下一步

- G0–G2 保持既有 `PASS`；S0.4/G3 仍为 `IN_PROGRESS`。
- 本地控制面已达到可提交、可同步状态，但 requirements/layout/download-plan 的
  `generator_git_commit` 仍需在本轮实现提交后按生成器提交重新级联冻结。
- 服务器仍需补齐 410M tokenizer 三文件、MNLI 五文件和 RTE 三文件，随后依次生成
  acquisition attestation、独立 verify-only 报告、gate-only publication、13 项
  READY/qualification、G3-S1/S2/S4/S5/S6 与正式 `asset_resolution`；完成前不进入 G4。

## 2026-08-03 23:21 CST — G3 控制文件绑定实现提交

- G3 控制面实现提交：`69dbe38c1c2ead921366f665794e77a579b2f8ff`。
- requirements、layout、download plan 的 `generator_git_commit` 统一绑定该实现提交，并按
  requirements → layout → plan 的依赖顺序重新计算全部 artifact hash：
  - requirements：`3d8183c4c8a9152d3f44ca900509c2b401ba298c5af8b727f1eee0dcb61bc433`；
  - layout：`56b0dc1da9b2fa5e605afb4fd82da0a57eaa27f3cc05e5ec54803df60cab7203`；
  - download plan：`33fbd373875b3fde10074742aa04accd612109e70380989b007b6e9d00e190a8`。
- 三份文件已由严格 loader 联合重放，交叉引用、13 对象集合和生成提交均通过；下一步完成
  定向回归后形成独立冻结提交并同步服务器。
- requirements/layout/download-plan、13 个 HTTP object spec 与 relay 联合回归：
  `49 passed, 0 failed`。同时更新一条旧测试，使其断言 formal provider 不再嵌入物理
  manifest/root 路径；实际资产只允许由提交的 G3 resolution 解析。

## 2026-08-03 23:59 CST — G3–G9 收口补记与 G10 本地实现

本节按不可变 Git 提交和测试输出补记 23:21 之后的工作，不重写前述失败、进行中状态或
历史证据。当前分支为 `feat/stage0-completion`；服务器 formal Gate 尚须在最终同一提交上
重跑，因此本节只把已完成的本地实现与仍未完成的外部步骤分开记录。

### 实际修改与阶段提交

| 范围 | 提交 | 实际成果 |
|---|---|---|
| G3 控制冻结 | `83e3492b70f8dc428e9cc1f6001b82e8f8279934` | 冻结 requirements/layout/download-plan 与资产控制引用 |
| G3–G5 | `7bb9403b5de4d7137ef149839ab88d98608cee98` | 正式资产、provenance、单卡训练 Gate 与 handoff |
| G6 | `114de62a34a2f305293d03d8aa8591c3d171037f` | 四卡 NCCL、全局 loss 归约、累积、`no_sync` 与故障合同 |
| G7 logging | `828a82d2c205853ea9d1e4ad84b6836d0843192e` | canonical JSONL、lineage、TensorBoard 派生视图和开销证据 |
| G7 recovery | `d8f52ead4d265f65e88b107f0d64d681e6203686` | 单卡/四卡 fresh-process checkpoint 恢复、保留与故障拒绝 |
| 预检/Schema 修复 | `5103ba4033b3ff516e911c899d26fabc73172a46`、`8da72929c5ea74fcff56ec58c299af9cb00f55ab` | 保留 formal fail-closed 语义并统一 schema identity |
| G8 | `1ddc5ee869f93e5aa2859f8ab53528b2fdeebd65` | 14M/160M/410M 容量 envelope、36-launch 计划、租约、预检与故障处置 |
| G9 | `33924a43d48d41d0b9ea59ce005da66eb872811a` | 六层硬测试矩阵、确定性 fixture、离线 socket 审计、独立单卡/四卡/恢复重放与运行说明 |

S0.12/G10 的本地实现新增三端只读观察、逐文件 Git 交付清单、13 项服务器资产
existence/size 复核、全部 G0–G9 committed GateRecord 重放、Agent 五文件哈希、bundle 清理、
G1-D 有效期、中文动态 Worklog、Stage 1 handoff 和不可覆盖的 `READY`。该实现尚未形成
提交；只有提交、推送、服务器快进并在同一 HEAD 上完成 formal G3–G10 后才可发布 READY。

### 本地验证与退出状态

| 检查 | 结果 |
|---|---|
| G9/G7/task runner/CLI 定向回归 | `47 passed, 0 failed` |
| 完整仓库测试（86 个测试文件，源码树外四分片） | `937 passed, 10 skipped, 0 failed` |
| Windows skip 审查 | 目录/file symlink 权限、POSIX mode、Windows Torch Gloo 能力；均未计作 formal PASS，Linux G9 只允许版本化矩阵声明的 Windows-junction 不适用项 |
| G10 合成三端观察、资产 manifest、配置、schema 与 runner 回归 | `41 passed, 0 failed` |
| Python 静态编译 | `python -m compileall -q src ops tests`，退出码 0 |
| 空白检查 | `git diff --check`，退出码 0；只有现有 Windows LF/CRLF 提示 |

完整回归首次把 `--basetemp` 放入仓库时，G3 路径守卫按设计以
`STAGE0_G3_WORKSPACE_OVERLAPS_SOURCE_ROOT` 拒绝；改到 Codex 可写但位于源码树之外的专用
目录后全部通过。该失败没有被删除或改写为产品缺陷。

### 当前同步与风险

- 本机 HEAD 为 `33924a43d48d41d0b9ea59ce005da66eb872811a`；本日志与 G10 实现尚未提交。
- `origin/feat/stage0-completion` 和服务器最后已知 HEAD 仍为
  `44a63177b735bd2b5ecd7d84817d2bef2348837c`。当前不满足三端一致，G10 不能通过。
- GitHub push 与 Git bundle/源码/非敏感验证产物传输需要用户对本任务明确授权；在得到
  授权前不重试外发，也不把本地提交冒充服务器 formal 证据。
- G1-D 仍是单盘、可再生 Stage 0 smoke 产物的限时风险接受，不是备份；有效至
  `2026-08-18T23:59:00+08:00` 或更早终止条件发生时。Stage 4 前仍须第二故障域与恢复演练
  或新的明确风险决定。
- 下一步：提交 G10 本地实现；取得外发授权后推送、验证 bundle 快进服务器、同步并逐项
  核对 `Agent/` 五文件、清理精确 bundle；在最终同一提交上运行 G3→G10 formal 链并发布
  新 readiness。任何一步不一致都保持“收尾中”，不宣布 Stage 0 完成。

## 2026-08-04 07:28 CST — G10 真实仓库布局复核与 Linux 路径修正

- G10 本地实现已提交为 `021885cbbfd4c30b3ace9a0cfcd5a844496c7444`；提交前完整仓库
  四分片回归为 `945 passed, 10 skipped, 0 failed`。10 个 skip 仍仅是 Windows symlink、
  POSIX mode 与本机 Torch Gloo 能力限制，未计作 Linux formal PASS。
- 在提交后的真实 `_capture_source()` 审计中保留并定位失败：Git 索引记录
  `plan/stage0/...`，原 G10 关键源引用和三份交付文档却写成 `plan/Stage0/...`；同时仓库
  根文档的索引名称是 `Readme.md`，G9 replay runbook 使用了 `README.md`。这些引用在
  Windows 大小写不敏感文件系统上可解析，但在 Linux 服务器会导致关键源探针或链接检查
  失败，因此不能进入同步/formal 阶段。
- 已把所有相关引用改为 Git 索引的精确大小写，并增强 G10 链接验证：每个本地 Markdown
  目标除必须位于仓库且存在外，还必须与 `git ls-files -z` 中的路径逐字匹配。新增真实仓库
  回归同时校验全部关键源引用和 8 个 Stage 0/Stage 1 交接链接。
- G10 回归：`9 passed, 0 failed`；G9/G10/task runner/CLI 组合回归：
  `43 passed, 0 failed`；`python -m compileall -q src ops tests` 与
  `git diff --check` 均退出 0。修正提交形成后还须直接重跑 `_capture_source()` 与最终仓库
  inventory，随后按新 HEAD 重新生成增量 bundle。
- Linux 路径修正已提交为 `6d58b016ea38ed6da8bb02df4bb83405a34e7804`。提交后真实
  `_capture_source()` 已通过；随后 inventory 对远端基线 `44a6317..HEAD` 的范围检查继续
  保留并发现 3 个已提交的 G5 EOF 多余空行：`ops/stage0/formalize_g5.py`、
  `ops/stage0/run_g5_worker.py`、`tests/test_stage0_g5.py`。这说明只对未提交工作树运行
  `git diff --check` 不能证明整个交付范围干净。
- 已精确移除上述 3 个 EOF 空行；G5+G10 回归为 `12 passed, 0 failed`，相关 Python
  静态编译与当前工作树 `git diff --check` 均退出 0。形成新提交后必须再次用
  `44a6317..HEAD` 重跑范围检查和完整 inventory；未通过前仍不生成 bundle 或 READY。
- GitHub、服务器和 `Agent/` 外发同步仍等待本任务的明确授权；当前状态继续是“收尾中”，
  不生成或复用 `READY`。

## 2026-08-04 10:03 CST — 外发授权、首次三端快进与 G3 缺失资产诊断

- 用户已明确授权本任务将 Stage 0 分支推送 GitHub，并向 `sophgo13-via-lab` 传输完成
  Stage 0 所需的增量 bundle、源码、配置和非敏感验证产物。同步前 GitHub、服务器都在
  `44a63177b735bd2b5ecd7d84817d2bef2348837c`，服务器分支正确、工作树干净、origin 正确；
  `Agent/` 五文件集合和 SHA-256 已与本机完全相同，无需冗余覆盖。
- 本机/GitHub/服务器曾以非强推、经 SHA-256 验证的 bundle 和 `--ff-only` 同步到
  `f9a0c43d02e5d93678bf97b6947a900c841e7d29`；本次服务器及本机 bundle 均已按精确路径
  删除，可由 Git 历史重新生成。该提交上的 G0–G2 bootstrap 通过，index 为
  `evidence/stage0/bootstrap/f9a0c43d02e5d93678bf97b6947a900c841e7d29/index.json`。
- G3 acquisition 在 Python socket 离线审计下保留了一次正式失败：报告
  `operations/stage0/g3-download-f9a0c43d02e5.json` 显示 410M 配置和 1.62 GB 权重为
  `already_ready`，随后因 tokenizer 缺失触发 6 次被阻断的外连；审计记录 6 次外部 DNS
  尝试且没有下载。独立只读全清单复核确认 13 个对象只有 2 个通过，缺少 3 个 tokenizer、
  MNLI 5 个 parquet 和 RTE 3 个 parquet，共 11 个冻结对象。
- 版本化 `lab-direct` relay 使用官方 Hugging Face endpoint 的 99 字节探针在 900 秒总
  deadline 后以 `G3RelayDispatchError` 失败；服务器无残留进程、锁可获取、目标不存在、
  对象专用 `.part` 为 0 字节，因此没有把部分内容冒充正式资产，也没有并发 writer。
- lab-pc 的既有资产脚本使用 `hf-mirror.com`；不落盘内存探针从该镜像取得同一 99 字节并
  精确命中冻结 SHA-256 `6f50ab5a...fdae7ad`。为避免运行时篡改版本化 relay，已新增显式
  `official`/`hf-mirror` endpoint profile：argv 和日志只含枚举 profile 名，运行 URL 仍只
  在 lab 进程内派生，receiver 的 source commit、plan/spec/control 哈希和 no-clobber 校验
  保持不变。lab-pc 的系统 `PATH` 只有 WindowsApps alias，因此 dispatcher 还新增版本化
  `path`/`cjl-python312` 解释器 profile；只允许命名枚举，不接受任意远程命令。relay、
  acquisition 和 lifecycle 定向回归在 endpoint profile 首轮实现时为
  `42 passed, 0 failed`；加入解释器 profile 与文档后的最终回归为
  `43 passed, 0 failed`，`python -m compileall -q src ops tests` 与 `git diff --check`
  均退出 0。
- 上述源码修改形成新提交后，`f9a0c43` 的 bootstrap 自动成为旧提交证据，不得交给后续
  Gate；必须重新快进 GitHub/服务器并从新最终 HEAD 重跑 G0–G10。当前仍是“收尾中”。

## 2026-08-04 11:10 CST — 镜像中继双向管道诊断与原生字节管道修复

- endpoint/Python profile 修复提交为 `88254d2e55038e195c4ea4a03806de08d77232e8`；本机、
  GitHub、服务器已非强推/`--ff-only` 同步到该提交，增量 bundle 两端均已精确删除，
  `Agent/` 五文件集合与 SHA-256 继续完全一致。该提交的 G0–G2 bootstrap 使用错误的通用
  环境时按设计以 `STAGE0_BOOTSTRAP_G2_RUNTIME_MISMATCH` 拒绝且未发布证据；改用 G2 冻结的
  `/envs/parameter-importance-stage0-1bd963c65f75` 后通过，index 为
  `evidence/stage0/bootstrap/88254d2e55038e195c4ea4a03806de08d77232e8/index.json`。
- `hf-mirror + cjl-python312` 的 99 字节正式 duplex relay 在 900 秒 deadline 后失败；120 秒
  单尝试诊断给出安全失败码 `OVERALL_TIMEOUT`，且从未出现 `READY`。失败后无 receiver/relay
  残留，中央 `.part` 仍是 0 字节空哈希、单链接，最终目标不存在，advisory lock 可立即取得。
- lab Python 的 urllib 默认和显式无代理请求都在约 1 秒命中冻结的 99 字节与 SHA-256，故
  HTTP/镜像不是根因。最小探针定位到 Windows `ssh.exe` 被 Python 同时连接 stdin/stdout
  管道时不转发短行；相同 alias 从默认 shell 正常，本机 `cmd.exe` 原生双 SSH 管道也以
  9 字节探针正常收口。
- 已实现命名 `lab-pipe` fallback：独立 `plan-only` receiver 在锁内给出 resume offset；lab
  emitter 只输出该固定 Range，本机只做不落盘原生字节管道；正式 receiver 再次锁定并要求
  offset 精确一致，最终仍由服务器执行大小、SHA-256 和 no-clobber。任何竞争、短流或哈希
  错误都失败关闭并保留可恢复 `.part`。新增 offset、plan transcript、命令注入边界、emit
  单尝试、dispatcher 编排与完整 receiver transcript 测试后，中继单文件回归为
  `27 passed, 0 failed`。三文件组合首轮被 300 秒工具上限终止于 35 个通过点、无失败；
  拆分复核分别为 `26/26`、`5/5`、`16/16`，加入最终编排测试后以 600 秒上限完成同进程
  组合回归：`48 passed, 0 failed`（233.05 秒）。
- 提交前扩展资产状态机回归顺序执行为 `19 passed`，以及 `188 passed, 1 skipped`；skip 仍是
  既有 Windows 目录 symlink 权限能力，未计作 formal PASS。更宽的 G3 publication/formal
  组合在 600 秒上限到达时前进至 8 个通过点、无失败；60 秒 faulthandler 栈连续落在不同
  测试的 Git/source SHA 重放，而非锁死，故该次只记录为工具超时、不计作 PASS。直接相关
  的 48 项组合和 207 项资产状态机测试均已完整退出 0；最终 Linux formal 链仍须重放 G3。
- 上述新实现形成下一提交后，`88254d2` 的 bootstrap 将成为旧提交证据；须再次三端快进并
  从新 HEAD 重跑 bootstrap 与 G3–G10，当前不发布 READY。

## 2026-08-04 12:25 CST — 原生管道实测纠错与当前 GPU 硬件阻断

- `lab-pipe` 实现已提交为 `b2903883d31a2222ea259778fcb327298d3f8042`；本机、GitHub、
  服务器均以非强推/`--ff-only` 快进到该提交，增量 bundle 经两端 SHA-256 和 `git bundle
  verify` 校验后已按精确路径清理。提交后 inventory 为 900 个 tracked 文件、40,776,641
  字节、21 个关键源和 8 个交付链接；禁止类型、超限文件、疑似秘密与范围空白检查均无命中，
  `Agent/` 五文件集合和 SHA-256 继续三端一致。
- 在 `b290388` 上用 G2 冻结环境重跑 bootstrap 时，CUDA 预检不再满足旧报告中的四卡事实：
  `nvidia-smi -L` 仍列出四张白名单 A100，但物理 GPU 3（PCI `a4:00.0`）温度和 ECC 变为
  `Unknown Error`/`N/A`，CUDA 进程只能看到 3 张卡并对 `cuda:3` 返回
  `invalid device ordinal`。当前 boot ID 仍为 `04326255-b422-4f1e-8dc6-9c3cc8f0a5b9`；内核
  记录同一 GPU 的 Xid 120（GSP task exception）、119（GSP RPC timeout）和 154，恢复动作
  已变为 `GPU Reset Required`。因此本次 bootstrap 失败关闭且没有发布 `b290388` 证据；
  `f9a0c43`/`88254d2` 的旧提交 bootstrap 也不得复用。
- 2026-07-19 的 GPU 恢复/重启报告只覆盖当时维护窗口，不能自动延伸至本次新故障；用户当前
  授权覆盖 GitHub 和项目服务器文件传输，不覆盖系统 GPU reset 或 reboot。项目代码按计划
  也不得执行这两项管理员动作，因此本轮只做了只读诊断，没有复位 GPU、解绑驱动或重启服务器。
- `b290388` 的首个真实 99 字节 `lab-pipe` 冒烟在发送资产字节前失败关闭并保留失败输出：
  右侧 `cmd.exe` 管道把整条远端 receiver 命令引用成一个可执行文件名；同时镜像最终响应采用
  无 `Content-Length` 的流式传输，旧校验返回 `HTTP_CONTENT_LENGTH_INVALID`。最终资产未发布，
  没有把不完整内容视为 G3 输入。
- 已把右侧 SSH 命令改为逐 token 传递 `env`、冻结 Python 和 receiver 参数，避免整串命令被
  Windows 引用；HTTP 纠错只允许命名 `hf-mirror` profile 缺少 `Content-Length`。官方 endpoint
  继续强制长度头；镜像仍强制 200/206 状态、断点续传时的精确 `Content-Range`、固定预期字节数、
  多余字节拒绝，并由服务器最终重算冻结 SHA-256 后 no-clobber 发布。新增无长度首传/续传、
  错误 Range 和远端命令分词回归；中继单文件为 `29 passed`，与 receiver 原子发布、断点续传和
  下载计划的组合回归为 `53 passed, 0 failed`，相关 `py_compile` 与 `git diff --check` 均退出 0。
- 下一步先把本次纠错形成新提交并三端快进，再用该不可变 HEAD 重跑 99 字节真实冒烟和剩余
  10 个冻结对象。G3 非 GPU 资产闭环可以继续；bootstrap 与后续 CUDA/NCCL Gate 必须等待
  GPU 3 经新的管理员维护授权恢复并重新通过四卡只读资格检查，当前仍不发布 READY。

## 2026-08-04 12:42 CST — 99 字节实测通过与 dataset repository 类型纠错

- 原生管道纠错已提交为 `f6f64f1d4426aa1db5033b627f5c9a1d6a3ce932`，非强推到 GitHub
  并通过 4,052 字节增量 bundle 快进服务器；bundle 的 SHA-256 为
  `135c46017b40018a42609778efe1c4cb0727422767d7ac743e269f449411282c`，服务器端再次执行
  `git bundle verify` 后才 `--ff-only` 合并。本机和服务器临时 bundle 随后均按精确路径删除，
  三端 HEAD 一致且 tracked 工作树干净。
- 在该不可变提交上的 99 字节 `special_tokens_map.json` 真实 `lab-pipe` 冒烟通过：镜像 emitter
  精确输出 99 字节，dispatcher 返回 `status=PASS objects=1`，服务器 receiver 以冻结大小和
  SHA-256 原子发布。这证明逐 token 远端命令和无长度镜像响应两项修复均通过真实两段 SSH 路径。
- 随后的完整 13 项遍历只读跳过已经就绪的 config、weights 和 99 字节对象，并成功发布
  `tokenizer.json`（2,113,710 字节）与 `tokenizer_config.json`（396 字节）；到首个 MNLI train
  对象时，lab emitter 在发送字节前以 `HTTP_STATUS_404` 失败关闭，dispatcher 总体退出 1。
- 404 根因是 URL 派生把所有稳定 `huggingface/owner/repo/path` ID 都解释为 model repository；
  GLUE 实际位于 Hugging Face dataset repository，必须经过 `/datasets/owner/repo/...` namespace。
  冻结 object spec 不含单独类型字段，但同一 download plan 与 relay binding 已把目标绑定为
  `models/...` 或 `datasets/...`，且该字段受 plan hash、binding transcript 和服务器重放保护。
- 已统一修正直接 acquisition 和 lab relay：repository 类型只从冻结 `asset_root_ref` 首段派生，
  `models/` 不加 namespace，`datasets/` 加 `/datasets/`，其他根拒绝；没有修改 13 项 object ID、
  revision、大小、SHA-256 或控制文件 artifact hash，也没有把 runtime URL 写入 argv/报告。
  新增 model/dataset URL 派生、非法根和完整计划观察断言后，relay、下载计划与原子 acquisition
  组合回归为 `54 passed, 0 failed`，相关 `py_compile` 和 `git diff --check` 均退出 0。
- 下一步把 repository 类型纠错形成新提交并三端快进，再从 MNLI train 重跑；已发布 tokenizer
  将按同一服务器 SHA-256 只读识别为 ready。G3 完整离线重放前仍不声明资产 Gate 通过。

## 2026-08-04 13:03 CST — dataset 续传实测与 dispatcher 有界重试

- repository namespace 修复已提交为 `a30f3df721856eaf8df6eb92f124cbb28a383762`，非强推到
  GitHub 并以 5,732 字节增量 bundle 快进服务器；bundle SHA-256 为
  `626092c40e837b9f4d711094f5bfde4657ee76f2dc5f86c6d64925cc00127570`，服务器端
  `git bundle verify` 与 `--ff-only` 均通过后，两端临时文件已精确删除。
- 在 `a30f3df` 上只选择 download plan 中 8 个 `datasets/` 对象重跑，首个 MNLI train 已越过
  先前的即时 404，但约 654.5 秒后两段 SSH 报 `Connection reset`，dispatcher 按设计退出 1，
  没有把未完成 transcript 计作成功。随后独立 `plan-only` 只读查询也因 SSH transcript 不完整
  失败，未臆测或记录一个未经服务器确认的 offset。
- 服务器 receiver 本来已能保留固定身份 `.part`，但旧 `lab-pipe` dispatcher 每项只执行一个
  native-pipe session；外层命令失败后需要人工重启，未把已有断点能力纳入同一个整体 deadline。
  已新增每对象默认 6 次的有界 session：每次失败后都重新运行锁内 plan，输出 URL-free 的
  attempt/offset/already-ready 诊断，再从服务器确认的 offset 构造新管道；所有 session 共用原有
  monotonic 总 deadline，emitter 每个 session 仍严格单尝试，receiver 的 offset 竞争检查不变。
- 新增首次 pipe transcript 失败、第二次重新 plan 后成功的编排回归，并拒绝 0、负数、布尔值和
  非整数 attempt budget；文档新增 `--lab-pipe-max-attempts` 合同。代码形成不可变提交并三端同步
  前不用于资产发布；同步后才继续 MNLI/RTE 断点续传。
