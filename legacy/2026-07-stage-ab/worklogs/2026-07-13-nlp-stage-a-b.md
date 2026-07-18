# 2026-07-13 NLP 参数重要性阶段 A/B 实施日志

## 目标与范围

- 严格按照 `plan/NLP参数重要性完整实验计划书.md` 的停止门槛完成阶段 A 与阶段 B。
- 使用与 Pythia 完全一致的 GPT-NeoX/Pythia 架构、固定 tokenizer、step0 初始化和预打乱数据前缀。
- 实现 Microbatch-level U-statistic、raw、double sampling、actual-update 诊断、恢复和剪枝。
- 在服务器上完成 4 卡 BF16/NCCL 验证、14M 方法验证、160M step512、SST-2 两路线、统计图和报告。
- 与实验并行，以既有安全下载链路获取其余 Pythia Pile 分片；不修改已验收 shard0 和 idx。

## 启动状态（北京时间 2026-07-13 22:30）

- 本机、GitHub `origin/main` 和服务器基线提交均为 `4c91ff6356724d014da9667d2e93d6aac2f23fb8`。
- 已创建独立分支 `feat/nlp-pythia-stage-ab`。
- 服务器 8 张 A100-SXM4-80GB 均空闲；128 CPU、约 1 TiB 内存；项目 NVMe 剩余约 3.0 TB。
- 服务器 `manifests/READY`、`pip check`、4 卡 NCCL/BF16、160M forward/backward、SST-2/WikiText 离线读取均已通过。
- Pile shard0（30,000,000,000 bytes）和完整 idx（1,757,184,042 bytes）固定 SHA-256 已通过；step 0/1/511 与官方 batch viewer 一致。
- 当前 `main` 没有 NLP 实验运行代码；本次从新的 `src/param_importance_nlp` 包开始实现。
- 本机开始前已存在一个不属于本次工作的跟踪删除项：`Agents/工作日志与多端同步规范.md`。本次不暂存、不覆盖该项；实际规范从忽略目录 `Agent/工作日志与多端同步规范.md` 读取。

## 当前实施

- 已建立 Python 包、严格 YAML 配置解析、离线/随机状态/运行环境清单基础设施。
- 已并行分派三个互不重叠的工作单元：安全全量下载、重要性数学核心、Pythia 模型与 mmap 数据层。
- 尚未开始正式训练；阶段 A 核心测试未通过前不会启动阶段 B。

## 待更新

后续每次验证、服务器运行、失败恢复、产物路径、配置哈希、随机种子、Git 提交与三端同步结果均追加在本文件。

## 首轮代码实现（北京时间 2026-07-13 23:00）

- 新增 `param_importance_nlp` Python 包及阶段 A/B 配置：
  - Pythia 14M/160M 精确 GPT-NeoX 架构和固定 step0 身份校验；
  - 只读 Megatron `MMIDIDX` reader，显式绑定 shard0/idx，不扫描下载目录；
  - 手工多卡梯度聚合，microbatch mean gradient 的 FP32 `S1/S2`；
  - U-statistic、raw、signed/positive/negative、actual-update 去 decoupled weight decay；
  - AdamW、cosine/warmup、全局梯度裁剪及裁剪因子；
  - 原子 checkpoint、模型/optimizer/scheduler/RNG/data cursor/importance 恢复；
  - SST-2 A/B 单 token prompt、标签位置 loss、受控数据顺序；
  - 全局/层平衡剪枝、分布统计、估计器偏差和数值积分基础设施。
- Pythia 数据层真实只读核验：14M 为 14,067,712 参数；160M 为 162,322,944 参数；官方 step0/1/511 批 hash 与最终环境验收报告相同。
- 本机完整可收集测试（排除本机未安装的 `safetensors` checkpoint 测试）结果：`45 passed, 1 skipped`，跳过项仅因本机没有 Transformers。
- 本机核心集合另一次结果：`32 passed`；`python -m compileall -q src tests` 通过。
- 下一步：完成代码审查，形成首个里程碑提交并同步服务器，在锁定 venv 中运行全部测试和真实资产验收。

## 第二轮审查与外部评价接线（北京时间 2026-07-14 12:30）

- 只读审查确认 4 卡梯度尺度、causal shift、BF16/FP32 累加、clip factor、AdamW data update、step0 共享初始化和 SST-2 预测位置均正确。
- 已修复或加固 checkpoint manifest 验证、配置 fingerprint、world size/rank RNG、resume 独立日志段、best-validation 自包含恢复、clipped/unclipped U-statistic 持久化及 rank0 异常集体传播。
- 预训练现已接入固定 WikiText 外部评价：step0、固定间隔和 step512 均记录 loss/perplexity；结尾用同一 token 流评价公开 Pythia step512，并输出 `public_comparison.json`（perplexity ratio 门槛 1.15）。
- 验证数据、tokenizer 及公开 checkpoint 的文件大小和 SHA-256 会写入产物 manifest；验证 token 流按所需 batch 数提前停止，避免无谓装入整个 WikiText。
- 当前本机命令 `PYTHONPATH=src pytest -q --ignore=tests/test_checkpoint.py`：`51 passed, 1 skipped`；跳过项仅因本机没有 Transformers，服务器锁定环境具备完整依赖。
- 服务器只读复核：WikiText validation parquet、SST-2 三 split、公开 Pythia-160M step512 的 config/tokenizer/weights 均存在；服务器 venv 含 safetensors 与 matplotlib。
- 尚未启动正式 Stage B：double sampling、真实 4 进程等价性、完整断点恢复一致性和 Stage A 真实 provider/门槛报告仍在补齐。
- 全量 Pile 下载已完成 manifest、脚本、SHA-256 和持久任务准备，当前真实 `curl=0`；审批器要求用户在获知约 4.5–5 天、约 570 GB 后再次明确批准，因此尚未启动，不影响阶段 A。

## 下载启动与阶段 A/B 代码门槛（北京时间 2026-07-14 15:00）

- 用户明确批准全量下载后，已修复 lab SSH 环境把 `USERDOMAIN=WORKGROUP` 误当计划任务域名的问题，改用 `WindowsIdentity.GetCurrent().Name`。
- `CjlPileFull` 与 `CjlPileFullSupervisor` 均为 Running/priority 7；lab 进程 BelowNormal；服务器恰一个 curl，`nice=19`、`ionice=idle`，对象 flock 已持有。
- 首个对象 `document-00001-of-00020.bin.part` 实测约 1.506 MiB/s；裸传约 100 小时，含低优先级逐片 SHA-256 预计 4.5–5 天。
- 下载启动后再次核验 shard0/idx 的 size、mtime 和 SHA-256 均未改变；训练只读前缀与下载 `.part` 分离。
- 训练主路径已补全 clipped/unclipped U、raw、actual data update、末 100 步/下游全程 double sampling、layer/module sum/mean/count aggregates。
- 预训练已接固定 WikiText validation（固定 revision/size/SHA），并对公开 Pythia-160M step512 的 config/weights 固定 revision/size/SHA，输出 PPL ratio 1.15 gate。
- SST-2 固定 GLUE revision、三 split size/SHA 和 asset manifest；预训练后路线强制来源实验身份与 step512 checkpoint 完整性。
- 新增真实 4 进程 Gloo 等价性测试和全新进程 checkpoint-resume 测试；本机受 Windows Gloo/safetensors 限制明确 skip，待 Git 同步后在 Linux server 运行。
- Stage-A 真实 provider 现使用相同 `[0, 524288)` 总体、固定 4096 reference、uniform crop，排除 terminal zero-LR state；数值路径显式排除 AdamW decoupled weight decay并记录 LR/clip/WD 元数据。
- Stage-A 轨迹为 14M/32 step、21 checkpoint；预估轨迹 5–20 分钟、numerical 15–45 分钟、bias 1.5–5 小时，总计约 2–6 小时。
- 已实现三 seed SST-2 两路线、完整 global/layer-balanced 剪枝（含 double/random 20 masks）和阶段 A/B 报告生成器。
- 最新本机全套可运行测试在报告/剪枝合入前为 `59 passed, 4 skipped`；专项 method 测试为 `13 passed`。最终本机全集将在子任务收口后重跑。
- Git 里程碑写入曾因桌面审批器 usage limit 被拒；未绕过，等待用户在该拒绝后再次明确批准暂存/提交，再按 bundle 规范同步服务器。

## Stage A 运维收口与本机最终回归（北京时间 2026-07-14 12:30）

- 新增独立的真实 provider contract smoke：固定 reference 精确复现、同/异 seed、flatten 与 layer/module grouping、错误 checkpoint identity 拒绝、update/probe 总体不相交、数值路径排除 decoupled weight decay、有限 LR/梯度范数/clip factor 以及非零位移均有契约核验。
- Stage-A estimator-bias runner 现按 `checkpoint × M` 原子持久化；可跳过完整单元、重算损坏或部分单元，并以 config/provider/checkpoint/reference/grouping 指纹拒绝不一致恢复。正式预算门槛仍为至少 10 个 checkpoint、`M={4,8,16,32}`、每组至少 200 次重复和 4096 reference。
- 剪枝报告补齐 `u_signed`/`u_positive` 对 double 的 scalar、layer 与 module-sum Spearman；六份剪枝配置路径已与总报告配置自动核对。
- `python -m compileall -q src tests` 通过；本机 `PYTHONPATH=src pytest -q --ignore=tests/test_checkpoint.py` 为 `90 passed, 6 skipped`。六项 skip 均由本机缺少 safetensors/Transformers、Windows Gloo 限制或尚未设置服务器真实资产 smoke 环境变量引起；这些项目必须在 Linux 服务器锁定环境重跑，不计作通过证据。
- runbook 已修正 provider-smoke 的真实 CLI 与配置路径。当前仍未产生正式 Stage-A 数值产物；Git 里程碑提交与服务器同步继续等待用户在审批拒绝后明确批准。

## 计划书反向审计、训练优先与提交前收口（北京时间 2026-07-14 14:30）

- 用户已明确批准本次 Git 暂存和提交；提交仍只包含本次 NLP Stage A/B 相关路径，继续排除开始工作前已有的 `Agents/工作日志与多端同步规范.md` 删除。
- 服务器实时只读检查：正式训练进程 0、GPU compute process 0；全量下载恰一个 `curl`，`nice=19`、`ionice=idle`，检查时首个 shard 的 `.part` 为 10,044,166,144 bytes。
- 新增长训练 manager 与 lab 健康监督：预训练和监督训练只从完整且逐文件 size/SHA-256 通过的 checkpoint 恢复；方法实验、六个 controlled、六个 best-performance、六个 pruning 和三个 reuse intervention 都有注册作业、GPU 组互斥、有限重试、持久日志和 `ALERT`。
- 下载器改为扫描多 worker PID/start-time 租约；任一训练存活均会在最多约 5 秒内终止低优先级传输或校验并保留 `.part`。兼容桥会原子维护旧下载器使用的单租约，因此当前已经运行的旧进程在首次训练启动时也能让路。两组 SST-2 并发时，任一组未结束都不会恢复下载。
- 修复托管恢复器与真实 checkpoint manifest `{size, sha256}` schema 不一致的问题；测试 fixture 现使用实际 schema，错误 size/hash 会硬拒绝。
- 补齐独立 best-performance LR search：direct/pretrained 各运行 `3e-5/1e-4/3e-4`，与 controlled 使用不同实验 ID；以 validation NLL 选择。controlled 的 25%/50%/100% checkpoint 现会触发同 step validation。
- supervised 完成前会重新加载 best-validation 模型，对完整 train 和 validation 各评价一次并输出 accuracy/NLL/ECE 与 generalization gap；报告拒绝使用训练 minibatch loss 冒充完整 train 评价。matched-performance 使用同 seed、两路线都高于多数类、accuracy 差不超过 0.01 的预注册规则；不可达时记录 `not_achievable` 而不是伪造结果。
- 修复剪枝统计的伪重复：20 个 random masks 先在 seed 内聚合，所有 CI/effect size 以训练 seed 为单位且同 seed 配对；U/raw/double 与 random 的跨方法比较现能形成真实配对。
- 新增严格离线 SST-2 baseline：固定多数类、uniform random、官方 Pythia-160M-deduped step0/step512，同 prompt 评价 accuracy/NLL/ECE；官方 revision、config/weights、tokenizer、GLUE 与 prompt 均做身份和哈希验证，自产 checkpoint 不能冒充公开模型。
- 报告现直接读取 hash/step/identity 对齐的 `importance.safetensors`，输出 positive/signed 全局、层、模块统计、Gini、有效参数数、分位数、top-k、预训练到微调 overlap/Jaccard/enrichment/Spearman、训练 checkpoint 热力图、log-hist/violin/CCDF/Lorenz/top-k cumulative 和模块 share/mean 图。Pythia 的 fused QKV 保持 `attention-qkv`，不伪拆 q/k/v。
- 新增三个 seed 的独立 plan-22.4 功能复用干预：pretrain-high 在未微调/已微调模型、finetune-high、top-k 交集及 pretrain-only，覆盖 6 比例与 global/layer-balanced；每 seed 必须精确 60 行并通过 provenance/hash/因子网格 gate，普通剪枝不能替代。
- 新增 Stage-A 单测报告生成器，服务器将产出 `$DATA_ROOT/reports/unit_test_report.md` 与 JSON；任何 skip 都不算通过，真实 provider smoke 在 Stage-A trajectory 完成后单独执行。
- 最终提交前本机命令 `PYTHONPATH=src pytest -q --ignore=tests/test_checkpoint.py` 为 `117 passed, 8 skipped`，`python -m compileall -q src tests ops/train/prepare_resume_config.py` 通过。八项 skip 均来自本机缺少 safetensors/Transformers、Windows Gloo/Linux 运维工具限制或未设置服务器真实 provider 资产；服务器锁定环境必须实际执行这些门槛，不能引用本机 skip 作为通过证据。

## 服务器 Stage A 实测与数值方法 v2（UTC 2026-07-14 06:33–10:05）

- 里程碑提交 `56f46e5` 及修复提交 `813daad`、`1c98463`、`0b492c6` 已发布到 `origin/feat/nlp-pythia-stage-ab`，并通过 Git bundle 快进同步到服务器；本地和服务器均保持用户原有 `Agents/工作日志与多端同步规范.md` 删除项不被纳入提交。
- 服务器全量预检报告绑定 `813daad`：130 tests、0 failures、0 errors、0 skips；真实 provider contract、4 进程一致性和全新进程断点恢复门禁均通过。
- Stage-A 14M smoke 在提交 `1c98463` 上用 4 GPU 完成 step32，UTC 07:08:22–07:09:56，最终 checkpoint 为 `step-000032`，无告警。后台入口曾因 Windows Git bundle 未保留 executable bit 失败，已改为显式 `setsid bash` 并保留失败日志。
- 数值积分首次真实入口暴露 `python -m` 的 `__main__`/canonical dataclass 双重身份问题；两次失败及停止事件均保留。提交 `0b492c6` 固定 canonical 模块身份后，相关服务器测试 10/10 通过，正式 v1 于 UTC 07:38:05–07:52:25 完成 20 个独立路径、80 条 runtime、240 条 metrics。
- v1 的左端点结果未通过预注册层级门槛：parameter nonzero Spearman 最低 0.71049（阈值 0.70，通过），parameter top-5% overlap 最低 0.86401（阈值 0.60，通过），layer Spearman 最低 0.56667（阈值 0.90，失败）。失败只出现在 left 的 step4/6/8/10；trapezoid、Simpson、16-node Gauss 的 layer 最低值均为 1.0。`results/numerical_integration` 原始失败产物保留，Stage B 未启动，阈值未修改。
- 按计划书“排序过低则增加低频多节点积分”的允许路径，新增独立 `low_frequency_trapezoid_v2`：固定最低成本两节点梯形为 gate rule，使用从 record 65536 开始的新 update/probe 区间重新确认 20 状态，并强制引用及 SHA-256 绑定失败的 left v1 summary；Gauss reference 禁止自我门控。v2 使用新输出目录，不覆盖 v1。
- Stage-A estimator-bias 于 UTC 09:58:21 启动，GPU0 单卡、下载进程 0；检查时已原子完成 3/40 个 checkpoint×M 单元，manifest fingerprint 为 `88a6699...`，无告警。预计约 1.5–2 小时完成，之后才同步服务器代码并启动 numerical v2。
- 训练期间实验室电脑旧 `CjlPileFull` 与 `CjlPileFullSupervisor` 均保持 Disabled，服务器 curl=0。Codex heartbeat `nlp-stage-a-b` 每 15 分钟监督；全部训练结束或明确长空档后才恢复断点下载。

## Stage A 方法门禁收口（UTC 2026-07-14 11:51–12:17）

- estimator-bias v1 于 UTC 11:51:49 完成全部 40/40 个 `checkpoint × M` 原子单元、10 个 checkpoint、`M={4,8,16,32}`、每单元 200 次重复及 4096 reference；manifest fingerprint 为 `88a6699bb08b6cfce81cad91dd926168af3bb356e6dda0d8172a443d0bbb69d6`，无 ALERT。
- v1 除 `M>=8 stability` 外的原门禁均通过：layer/module U/raw bias ratio 分别为 0.02037/0.02000；U 参数 Spearman 0.49731 不低于 double 的 0.47996；U/double MSE ratio 0.91496、formula cost ratio 0.765；最大 dominance 0.01258。唯一失败是旧实现仅在参数级计算相邻 M 稳定性，最低 Spearman 0.39572，低于 0.90。
- 按计划书“参数噪声较大但 layer/top-k 稳定时可采用局部一阶方法”的预注册解释，新增独立 `scope_aware_stability_v2`：只读加载 v1 的已完成 U 均值，逐单元核验 run/checkpoint/M/repetition/reference/grouping 身份及 SHA-256，在新目录计算 parameter/layer/module 稳定性；layer/module 仍按原 0.90 作正式门禁，parameter 明确保留为描述值。v1 失败结果不得覆盖。
- scope v2 是 CPU-only，但持有训练租约、锁定 group0 且发现任何既有训练租约时拒绝启动；因此其单位文件哈希与模型分组读取不会和 Stage A/B 训练或下载并发。
- numerical-integration v2 于 UTC 12:02:58–12:17:16 完成。trapezoid 门禁全过：参数非零 Spearman 最低 0.98322（阈值 0.70）、层 Spearman 最低 0.96667（阈值 0.90）、参数 top-5% overlap 最低 0.97389（阈值 0.60）。产物保留并绑定 left v1 summary SHA-256 `dffafefe81be41249fc7ff2191bbd2d75b497c8b4b8d66f59b1665cb383cbb4e`。
- 本地 scope v2、正式 runner 分组稳定性与报告绘图专项测试为 `24 passed, 1 skipped`；skip 仅因 Windows 缺少服务端 Bash 工具。`compileall` 与 `git diff --check` 通过。下一步为提交/发布该修复、服务器全套零 skip 回归、运行 scope v2；只有 estimator 与 numerical 两项均过门禁才启动 Stage B。

## estimator positive-only v3 修复（UTC 2026-07-14 12:28–12:40）

- scope v2 绑定提交 `2523fd9`；服务器全套报告为 135 tests、0 failures、0 errors、0 skips。v2 于 UTC 12:29:46–12:32:36 完成，逐一校验 30 个 M>=8 单元及 SHA-256，未重算梯度、未覆盖 v1。
- v2 正式门禁仍失败：module signed-sum 相邻 M Spearman 最低 0.93333（通过），layer signed-sum 最低 0.58333（失败），参数级描述最低 0.39572。失败集中于 step26 的 16→32（0.66667）和 step30 的 8→16（0.58333）；全部层级 top-5%（9 组时为 top-1）重合为 1.0。Stage B 未启动。
- 只读诊断同时比较 signed/positive/absolute 与 sum/mean。signed layer mean 最低仍为 0.80，说明不能仅用层大小解释；对“已经跨 200 次平均的 signed 向量”再取正可达到 0.90，但这不等价于训练定义 `mean_r(clamp_min(U_r, 0))`，不得用作放行证据。
- 计划书 5.4、13.1、15.6 与风险表明确要求同时保存 positive-only/signed，并把 signed 正负抵消列为预期风险。因此新增独立 `signed_bias_positive_stability_v3`：保留 signed 的 bias/ranking/MSE 门禁和稳定性失败诊断，正式重算每个 repetition 的 U，持久化 `u_positive_mean = mean_r(clamp_min(U_r, 0))`，以相同 M、200 repeats、10 checkpoints 和原 0.90 layer/module 阈值验证实际训练使用的 positive-only 分数。v3 使用新 schema、fingerprint 和输出目录，不复用或覆盖 v1/v2。
- v3 专项测试为 `26 passed, 1 skipped`，覆盖 per-repetition positive 持久化、signed/positive 双稳定性表、门禁只选择 positive、schema2 身份和 manager 注册；`compileall` 与 `git diff --check` 通过。
