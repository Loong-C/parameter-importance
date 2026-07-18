# Stage B 严格确定性与梯形积分修正

## 修正前已经完成的证据

- Stage A estimator positive v3 已完成全部 40 个注册单元。参数、层、模块三个尺度的 `M>=8` 稳定性最低值分别为 `0.9299466/0.9166667/0.95`，均通过；signed cancellation 的失败诊断继续原样保留，不改名为通过证据。
- Stage A 数值积分 v1 使用左端点，一项层级 Spearman 最低仅 `0.56667 < 0.90`，正式失败产物保留。独立 v2 使用新数据区间和两节点梯形法，parameter/layer/top5 最低值为 `0.98322/0.96667/0.97389`，全部通过。v2 `summary.json` 的 SHA-256 为 `4b4d8c9b5b4fa0b2041527e9f00852ffdfcac84c547217a154694b7b6ff90d5c`。
- 首轮 Stage B 160M 已跑完 step 512，固定 WikiText perplexity 从 `63585` 降至 `420.43`，相对公开 step512 的 PPL 比率为 `0.635`。但各 rank 均记录了 memory-efficient attention 非确定性警告，所以该运行只作诊断，不能作为正式下游来源。

## 确定性 v2 运行与 GPU 故障

- v2 已固定 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，启用 `torch.use_deterministic_algorithms(..., warn_only=False)` 和 cuDNN deterministic，禁用 flash、memory-efficient、cuDNN SDPA，仅保留 math SDPA，并将 Pythia attention 固定为 eager。
- v2 运行至 step 363 后冻结；最新 hash-valid checkpoint 为 step 256。内核日志把故障定位到原物理 GPU0（UUID `GPU-6ff7389b...`、PCI `4f:00`），出现 Xid 62、120、154，含义为 GSP 崩溃并要求 GPU reset。
- 作业已安全暂停并终止；残留 D 状态只能通过重启清理。重启后故障卡已从 NVML/CUDA 枚举中消失，服务器当前只枚举 7 张卡。旧 v2 仍属于诊断结果，不能恢复后升级为正式结果。

## 梯形积分根因与 v3 修复

- 根因是训练配置和累加器仍默认使用左端点，Stage A 的梯形门禁只阻止/放行 Stage B，却没有把通过的方法落实到 Stage B 的逐步重要性积分。
- 新正式实验为 `stage-b-pretrain-v3`，实验 ID 与输出目录均为 `stage-b-160m-step512-seed1234-deterministic-trapezoid-v3`。它必须从 step 0 重新运行，不复用或覆盖 v2。
- 每个训练更新的左节点使用该步独立 microbatch U-statistic；右节点使用更新后参数上的下一独立训练 batch，按 `0.5 * left + 0.5 * right` 积分。checkpoint、验证和最终步会提前探测同一个下一 batch，但不推进数据游标，也不改变 RNG 状态；普通相邻训练步直接用下一步左节点完成上一节点，不增加额外前向/反向。
- 尚未完成的梯形左节点、U-statistic、actual data displacement、double-sampling 两半均写入 checkpoint，进程重启后可逐张量恢复。U/raw、double-sampling 和基于 AdamW 实际数据位移的 `actual_update` 全部使用梯形左右平均；纯位移大小不依赖积分规则。
- v3 启动前强制校验 Stage A v2 的 `summary.json` 哈希、完成状态、`gate_rule=trapezoid` 与 `gate_passed=true`，并把验证结果写入新运行目录。
- v3 使用 UUID 固定四卡：`GPU-180ff767...`、`GPU-d0ce0b43...`、`GPU-5c672d04...`、`GPU-e78c55cd...`，排除已消失的故障卡和重启前曾显示 `[N/A]` 的卡。它持有全局独占训练锁，运行期间禁止任何其他训练组并发。

## 验证与后续门禁

- 本地专项测试通过：梯形 U/raw/actual、double-sampling、RNG 透明、配置和运维脚本测试 `26 passed`；Windows 缺失 safetensors、Gloo 和 Linux 运维工具导致的 3 项 skip 不作为正式通过证据。
- 服务器提交 `566ace7206751cbfbe105ae4345c09014e47d177` 的完整报告为 `147 tests`，`failures/errors/skips` 均为 0；另行显式复跑分布式训练、checkpoint 真实进程恢复和 Stage A 哈希证据绑定测试，结果为 `3 passed`。报告路径为 `$DATA_ROOT/reports/unit_test_report.{md,json}`。
- v3 必须具备 `TRAINING_COMPLETE`、step512 `_SUCCESS`/manifest/hash、严格确定性环境记录、无非确定性警告、无 NaN/Inf，并通过 WikiText 下降及 PPL ratio `<=1.15`，之后才能启动 SST-2 与其余下游实验。
- 工作日志继续按 `Agent/工作日志与多端同步规范.md` 使用中文记录；用户原有的 `Agents/工作日志与多端同步规范.md` 删除不属于本任务，禁止暂存、恢复或覆盖。

## 首次 v3 启动拦截

- 首次启动在任何训练 step 和 checkpoint 产生前被 Stage A 证据门禁拦截。原因是 `metadata.numerical_integration_evidence_dir` 中的 `${DATA_ROOT}` 不经过配置 path 字段的自动展开，校验器把它当成了字面路径。
- 管理器已立即暂停，未发生 GPU 训练、产物覆盖或诊断结果冒充。修复为校验前显式调用 `os.path.expandvars`，并用包含 `${DATA_ROOT}` 的测试路径覆盖该真实配置形态；专项和服务器全套回归通过后才允许重新启动。
- 路径修复后的第二次启动同样在 step 0 前被门禁拦截：真实 Stage A runner 的完成状态是 `status=measured`，首版门禁错误地要求 `completed`。再次暂停后改为校验真实 schema，并额外要求 `state_count>=20`、三个已注册 gate 全部通过，以及 predecessor 明确保留 `left/false`；测试 fixture 改为与正式 summary 同形。

## step512 最终端点越界

- 首次完整轨迹已经执行到 step512 更新，但最终梯形右端点仍按 `data_cursor=524288` 直接取 batch；该游标恰好等于已验证数据前缀长度，四个 rank 在 checkpoint 和完成标记生成前触发 `IndexError`。因此这次 attempt 只作诊断，不能视为正式完成；最新可恢复的 hash-valid checkpoint 仍为 step256。
- 管理器一度按策略从 step256 自动恢复，发现后立即暂停，未产生新的正式完成证据。修复把 Pythia 下一 batch 定义为只在数据集边界循环：普通 checkpoint 不变，最终端点确定性回到记录0；它与 step512 左节点 batch 不同，且日志新增 source cursor、实际 cursor 和 `endpoint_wrapped=true`。未对齐或不足一个完整 batch 的边界继续 fail closed。

## step256 恢复启动脚本修复

- 最终端点修复通过服务器专项测试和零跳过全套回归后，管理器尝试从 step256 恢复，但在 worker 启动前被恢复配置路径门禁拒绝。根因是管理脚本只赋值而未导出 `DATA_ROOT`，因此 `prepare_resume_config.py` 把配置中的 `${DATA_ROOT}` 当成字面量，误判 checkpoint 不属于本次输出目录。
- 管理器另有一个退出清理缺陷：worker 注册的 `EXIT` trap 引用了函数局部变量 `start`，supervisor 稍后退出时该变量已离开作用域，在 `set -u` 下会报未绑定。修复在恢复配置生成前导出 `DATA_ROOT`，并在注册 trap 时立即固化租约 PID 和进程起始值；暂停期间无训练、无 curl，最新可恢复产物仍是 hash-valid step256。

## Stage B v3 完成与下游 GPU 门禁

- v3 已正式完成 step512。最终梯形右端点从 source cursor 524288 确定性循环到独立 cursor 0，`importance_finalized_step=512`；step512 checkpoint 的 manifest、`_SUCCESS`、大小与所有 SHA-256 均通过。恢复段环境绑定提交 `ddd2ec9efee7ef923e286b39e6fc666505e0cecb`，严格算法、`warn_only=false`、CUBLAS、cuDNN 与 eager/math SDPA 字段全部符合契约。固定 WikiText PPL 从 63585.31 降至 423.65，公开 step512 对照比率为 0.63969，质量门禁通过。
- 严格离线 SST-2 baseline 已对 872 条固定验证样本原子发布，`results.json` 与 `provenance.json` 的重算哈希和 `_SUCCESS` 一致。首次额外尝试的 `GPU-dc6...` 无法建立 CUDA context，只保留在 diagnostics；正式结果改用已完成 v3 的 `GPU-180ff...` 生成，不含 `FAILURE.json`。
- 当前仅枚举 7 张 GPU，且 `GPU-dc6...` 对 CUDA 不可用，无法构造两组互斥四卡。所有 controlled、best-performance、pruning 和 reuse manager 作业改为复用 v3 已验证的四个 UUID（单卡作业固定首个 UUID），持有全局独占锁并严格顺序执行；硬件恢复并重新通过实际 CUDA 映射门禁前禁止双组并发。

## 2026-07-16：SST-2 controlled 实验完成

- 六个 controlled 作业均严格离线、串行运行到 step3156；每项最终 checkpoint、最佳验证快照、`_SUCCESS` 与 SHA-256 均独立核验通过。所有作业使用 `integration_rule=trapezoid`，保留最终梯形端点记录；日志中未发现非确定性警告、NaN/Inf、Traceback 或 ERROR。训练期间四个已验证 GPU UUID 映射正常，服务器 `curl=0`，实验室下载任务保持禁用。
- seed1234：direct 最佳 step600，accuracy `0.7970183`、NLL `0.4193590`，最终 accuracy `0.8199541`；pretrained 最佳 step1000，accuracy `0.8188073`、NLL `0.4116327`，最终 accuracy `0.8279817`。
- seed1337：direct 最佳 step1000，accuracy `0.8176606`、NLL `0.4069731`，最终 accuracy `0.8073394`；pretrained 最佳 step789，accuracy `0.8176606`、NLL `0.3900799`，最终 accuracy `0.8222477`。
- seed2027：direct 最佳 step1000，accuracy `0.8153670`、NLL `0.4119017`，最终 accuracy `0.8165138`；pretrained 最佳 step1200，accuracy `0.8165138`、NLL `0.4081224`，最终 accuracy `0.8291284`。
- controlled 阶段至此完成。以上数值来自正式输出目录，不包含早期 GPU/CUDA 失败诊断目录，也未把任何诊断产物升级为正式结果。

## 2026-07-16：best-performance 学习率搜索进度

- 学习率搜索共六个候选，按 direct/pretrained 和 `3e-5`、`1e-4`、`3e-4` 组合串行运行；只有六项全部完成并通过哈希门禁后，才运行 `select-best-performance` 生成正式选择结果。
- `sst2-direct-bestperf-lr3e-5` 已完成：最终 step3156，最佳验证 step1700、NLL `0.4264441`，最终验证 accuracy `0.8107798`；最终 checkpoint 与最佳快照哈希通过。
- `sst2-pretrained-bestperf-lr3e-5` 已完成：最终 step3156，最佳验证 step800、NLL `0.4296466`，最终验证 accuracy `0.8405963`；最终 checkpoint 与最佳快照哈希通过。
- `sst2-direct-bestperf-lr1e-4` 已完成：最终 step3156，最佳验证 step600、NLL `0.4193590`，最终验证 accuracy `0.8199541`；最终 checkpoint 与最佳快照哈希通过。
- `sst2-pretrained-bestperf-lr1e-4` 已完成：最终 step3156，最佳验证 step1000、NLL `0.4116327`，最终验证 accuracy `0.8279817`；最终 checkpoint 与最佳快照哈希通过。
- 下一步依次运行 direct/pretrained 的 `3e-4` 候选，随后执行最佳性能选择门禁；之后才进入六个 pruning、三个 reuse intervention 和最终报告。所有后续作业继续持有全局独占锁，禁止双组并发，训练活跃时保持 `curl=0` 与实验室下载任务禁用。

## 2026-07-16：best-performance 学习率搜索完成

- `sst2-direct-bestperf-lr3e-4` 已完成：最终 step3156，最佳验证 step800、accuracy `0.8061927`、NLL `0.4173166`，最终验证 accuracy `0.8211009`；最终 checkpoint 与最佳快照哈希通过。
- `sst2-pretrained-bestperf-lr3e-4` 已完成：最终 step3156，最佳验证 step1000、accuracy `0.8038991`、NLL `0.4194457`，最终验证 accuracy `0.8073394`；最终 checkpoint 与最佳快照哈希通过。
- 六个候选全部具备正式完成标记、最终 checkpoint、最佳验证快照与独立 SHA-256 核验；日志未发现非确定性警告、NaN/Inf、Traceback 或 ERROR。候选训练期间继续严格串行，四个已验证 GPU UUID 映射正常，服务器 `curl=0`，实验室下载任务保持禁用。
- 严格离线运行 `select-best-performance` 后，正式选择结果原子发布到 `$DATA_ROOT/results/best_performance/stage-b-sst2-lr-search-seed1234`，`_SUCCESS=ok`，`selection.json` SHA-256 为 `c7dbba94240cc8167a9a10b83a507bbb3fc1d9846b8531e9fc3b0828ac866764`。
- 选择指标为验证集 NLL 最小化。direct 路线选择 `lr=3e-4`、最佳 step800、accuracy `0.8061927`、NLL `0.4173166`；pretrained 路线选择 `lr=1e-4`、最佳 step1000、accuracy `0.8188073`、NLL `0.4116327`。
- best-performance 阶段完成。下一步按 runbook 严格顺序运行六个 pruning 作业，再运行三个 reuse intervention；不得使用 pruning 结果替代功能复用干预证据。

## 2026-07-16：pruning 阶段开始

- `sst2-direct-pruning-1234` 已通过 manager 串行完成并原子发布。正式 `_SUCCESS` 声明 `record_count=594`，独立复核 `results.csv` 为 594 行；`curves.json`、`provenance.json`、`u-vs-double-rank.json`、`results.csv` 与 `results.json` 的 SHA-256 均与成功标记一致。
- provenance 正确绑定 `stage-b-sst2-direct-controlled-seed1234` 的最终 step3156 模型与重要性文件，并保留最佳验证 step600 仅作参考、`used_for_pruning=false`；没有混用 best-performance 候选或诊断产物。
- 作业日志未发现 NaN/Inf、Traceback 或 ERROR；运行期间固定使用已验证单卡 UUID `GPU-180ff767...`，服务器 `curl=0`，实验室两个下载任务保持禁用。下一项为 `sst2-direct-pruning-1337`，继续全局独占、严格串行。

## 2026-07-16：下游作业自动串行监督

- 为减少人工心跳造成的作业间空档，新增 pruning/reuse 专用正式产物验证器与实验室链式监督任务。队列固定为六个 pruning 后接三个 reuse intervention，从指定起始作业继续；每次只调用现有 manager `ensure/status`，不绕过全局独占锁、GPU UUID 门禁或训练租约。
- manager 的 pruning/reuse 完成判定已从“仅存在 `_SUCCESS`”收紧为 fail closed 验证：重算成功标记登记的全部 SHA-256，核对 `_SUCCESS`、`results.json`、`provenance.json` 的实验 ID 与 provenance ID，并要求 pruning 的 CSV/JSON/声明均恰好 594 行、reuse intervention 均恰好 60 行。
- 链式监督仅在上一作业正式验证通过且全部训练租约释放后启动下一项；任何哈希、行数、身份、ALERT、PAUSED、状态探针或 supervisor 补拉失败都会停止队列，并在实验室电脑 `training-chain/ALERT` 写入中文告警。链结束后仍需人工核验三个 reuse 产物、生成最终报告并恢复下载。
- 本地专项验证器与运维测试通过，Bash 和两个 PowerShell 脚本语法检查通过。Windows 本地全套测试因环境未安装 `safetensors` 无法完整收集，不作为正式通过证据；部署前仍须在固定服务器环境生成绑定新提交的零失败、零错误、零跳过全套报告。
- 首次部署链式任务后，实验室 Windows PowerShell 5 未进入原数组切片后的 `foreach`，任务以 0 退出并错误写入 COMPLETE，但没有调用任何下一作业 `ensure`；服务器保持 `training-active=0`、`curl=0`，未产生并发、覆盖或无门禁结果。修复改用严格模式和显式索引构造/遍历剩余队列，并要求重新专项测试、服务器全套回归与实际接管验证后才启用。

## 2026-07-16：接管审计与 pruning 阶段完成

- 接管前完整复核 `Agent/工作日志与多端同步规范.md`、`docs/stage_ab_runbook.md` 和 `worklogs/2026-07-13-nlp-stage-a-b.md`。本地、`origin/feat/nlp-pythia-stage-ab` 与服务器初始提交均为 `7e3a3d8163e0d00553e45daa1d312494d3c4ee71`；服务器 tracked 工作树干净，本地只保留用户原有的 `Agents/工作日志与多端同步规范.md` 删除，未暂存、恢复或覆盖。两个已完成同步用途的本地临时 bundle `.codex-sync-7e3a3d8.bundle` 和 `.codex-sync-93496dc.bundle` 经 `git bundle verify` 后按确切路径清理。
- 已在 `lab-pc` 终止并删除旧计划任务 `CjlNlpTrainingChain-stage-b-downstream`，没有继续使用旧自动队列。训练期间 `CjlPileFull` 与 `CjlPileFullSupervisor` 每项作业启动前均复核为 Disabled；服务器 `training-active` 在作业间为空、训练时只包含当前作业，`curl=0`。
- 当前服务器枚举 7 张 GPU；正式下游四个 UUID 映射保持为 CUDA 索引 1–4：`GPU-180ff767...`、`GPU-d0ce0b43...`、`GPU-5c672d04...`、`GPU-e78c55cd...`。所有 pruning 作业由 manager 固定使用首个正式 UUID 并持有全局独占锁，未发生并发。
- 代码回归证据：`$DATA_ROOT/reports/unit_test_report.json` 绑定提交 `7e3a3d8`，结果为 `151 tests, 0 failures, 0 errors, 0 skips`。因旧 `$DATA_ROOT/results/reproducibility/stage-ab.json` 仍绑定早期提交，按 runbook 重新执行：

  ```bash
  pytest -q tests/test_distributed_trainer_integration.py \
    tests/test_checkpoint_resume_integration.py
  python -m param_importance_nlp.experiments.run_reproducibility \
    --output "$DATA_ROOT/results/reproducibility/stage-ab.json"
  ```

  显式集成测试为 `2 passed in 22.09s`；重生成的 reproducibility 证据绑定 `7e3a3d8163e0d00553e45daa1d312494d3c4ee71`，多卡等价和断点恢复均为 true，两个子门禁各 `1 passed`、0 skip。
- 对接管时已完成的 `sst2-direct-pruning-1234` 与 `sst2-direct-pruning-1337` 重新运行 `bash ops/train/server_managed_train.sh validate-result <job>`，均为 `ok=true`、精确 594 条记录、全部登记哈希和 provenance 一致。随后严格串行完成其余四项 pruning；每项均为 attempt 1 完成，无自动重试、ALERT、PAUSED、NaN/Inf、Traceback、ERROR 或非确定性异常：

  | 作业 | UTC 运行时间 | 来源 | `results.csv` SHA-256 | `provenance.json` SHA-256 |
  |---|---|---|---|---|
  | `sst2-direct-pruning-2027` | 11:57:54–12:09:59 | direct controlled seed2027 final step3156 | `8aa466b7c8e09798e1fcd94e39f69a0de47afd315701ac1693fa5506112e5193` | `cd2fab086c53476f9e4563ffa24ef64793cbdfb7bb4c4433f67dca0aa255aab3` |
  | `sst2-pretrained-pruning-1234` | 12:11:19–12:23:23 | pretrained controlled seed1234 final step3156 | `2aff81020fbfef942c5e8a74471140d9505057d92b7c78d1c2a83bce6abe7d9c` | `48cd8e901cae84763458ad73328d736b15418cf9a33bd8fde6954956e1df861e` |
  | `sst2-pretrained-pruning-1337` | 12:24:27–12:36:39 | pretrained controlled seed1337 final step3156 | `71f929c55cec30866bb7b1a8bc999f1344d4bcd6ce2e43319254cc0b1257c017` | `496dc952245268a2b8f6a1337fe2226d3660a43a612af28e8e0b3864d309b35a` |
  | `sst2-pretrained-pruning-2027` | 12:37:34–12:49:50 | pretrained controlled seed2027 final step3156 | `35bf90eae4f2ef4f1643dd706a7a09e6b7dae2e6f35b5134197aa6c323c63f10` | `d0f785097a20f409504705b2c758865178638867a6d7969f2cf86200f999b19e` |

- 六项 pruning 的正式目录均位于 `$DATA_ROOT/runs/stage-b-sst2-{direct,pretrained}-pruning-seed{1234,1337,2027}`。最终集中复核六次 `validate-result` 全部返回 `record_count=594` 和 `ok=true`；CSV 含一行表头，因此 `wc -l` 为 595。每份 provenance 都绑定对应 controlled 路线最终 step3156 模型和重要性文件，最佳验证快照明确 `used_for_pruning=false`，未混用 best-performance、早期诊断或其他 seed。
- pruning 阶段至此完成。下一步严格按 seed 顺序运行三个 `reuse-intervention`，每项必须精确 60 条并通过 `_SUCCESS`、全部哈希、来源、模型/重要性身份与因子网格门禁；在全部干预完成前不生成最终报告、不恢复下载。

## 2026-07-16：reuse intervention、最终报告与下载恢复

- pruning 阶段日志形成提交 `2f874cbf6dc0b4a590a1aec4724969a358d204c8`，已推送到 `origin/feat/nlp-pythia-stage-ab`，并通过增量 Git bundle 将服务器仓库从 `7e3a3d8` fast-forward 到同一提交；本地和服务器临时 bundle 均按确切路径清理。该提交只修改工作日志，没有代码变更，因此不重复运行代码回归。
- 三个 plan-22.4 功能复用干预严格按 seed 串行完成，均为 attempt 1，无重试、ALERT、PAUSED、NaN/Inf、Traceback、ERROR 或非确定性异常。每项正式 CSV 含表头共 61 行，对应精确 60 条记录；manager `validate-result` 均返回 `ok=true`：

  | 作业 | UTC 运行时间 | `results.csv` SHA-256 | `provenance.json` SHA-256 |
  |---|---|---|---|
  | `reuse-intervention-1234` | 12:55:05–12:57:12 | `97d352fdabf86a5f8dbc19a0bf0dcc00bcc636178586fe144f1f50adbc698b09` | `da868a499ddb60458ef73b80dc99bff5eb5bd9ef03d1a41bb0208b03768c1310` |
  | `reuse-intervention-1337` | 12:58:39–13:00:46 | `ca9c212d6bec56c59765f5ffc389b7a966b52a894d2462925df2b43fa9a5b1f8` | `76a4412564e39065211bf0db525fae1acac504f94b13db907d30ea0ce558d491` |
  | `reuse-intervention-2027` | 13:01:50–13:03:56 | `28497ddf07012d04e1a8a47c1b41487c78f87b1b8295b31701e43b08e41d463b` | `e36461a48b741230039b740db7f8163f59d08fb9a53521acfc92e4fe53d83e50` |

- 每份 provenance 均绑定正式 Stage B trapezoid v3 预训练 step512，以及对应 seed 的 pretrained controlled 最终 step3156；模型、importance、manifest、resolved config 哈希齐全。三项 eligible 布局完全相同：49 个张量、123,568,128 个标量，identity SHA-256 为 `5c62904952f5d660c8af98e2cf740c60361d5b35e903baa1540c4c0066694fe1`。普通 pruning 未被替代为功能复用证据。
- 在九个下游结果再次集中通过 `validate-result`、服务器 tracked 工作树干净、`training-active` 不存在且 `curl=0` 后，严格离线运行：

  ```bash
  python -m param_importance_nlp.cli report \
    --config configs/report/stage_ab.yaml
  ```

  首次 SSH 前台等待 300 秒后客户端超时，但服务器报告进程仍持续高 CPU 正常运行；没有启动第二份报告。只读监控确认其分批处理 123,568,128 参数的重要性分布、分组统计和复用相关性，最终于 UTC 13:40:55 原子写出正式报告。
- 最终报告目录为 `$DATA_ROOT/reports/stage-ab-minimum-loop`，包含 48 个图文件、30 个表文件、`report.md` 和 `summary.json`；summary 登记的每个图、表和 Markdown 文件均实际存在。关键哈希：
  - `summary.json`：`53d4aa80688245ab0975949fee197a96923716a973c5cf72f8f4d7f0f5547f70`
  - `report.md`：`1504865798a34816908626c6d9e7ed6adc2d2aa53e35eb3805ed68a49ea5cadd`
  - `tables/stage_15_8_gates.csv`：`935994d600bdde2925983b0b5345a049a4932c5a3aa66bfe2e057cf67135fcb4`
  - `tables/functional_reuse_intervention.csv`：`38501d01fc1e360e18757be60b108699fc09d8994e45c3ee153419370b6d95f9`
  - `tables/functional_reuse_intervention_summary.csv`：`5552c3a70d2e4e823e5442ebefe0791c559e16840e8beabb2e3187333dc93827`
- 最终报告有效性门禁：`missing=[]`；Stage 15.8 表没有 `missing` 状态；`functional_reuse_intervention.coverage_gate_passed=true`，required/valid seeds 均为 `[1234,1337,2027]`，共 180 条记录，`ordinary_pruning_substituted=false`。经验门禁 1、2、3、4、6、7 通过；门禁 5“U-stat 与 double sampling 排序相关和剪枝效果相近或更好”正式失败，证据为 `positive pairwise rank rows=18; pruning_similar=False`。该失败按计划原样保留，未修改阈值或改写为通过；功能复用效应继续只作描述，因为计划未预注册 effect-size 通过阈值。
- 完成全部报告门禁核验后，在 `lab-pc` 删除旧监控任务 `CjlNlpTrainingChain-stage-b-downstream` 和残留的 `CjlNlpTraining-stage-b-pretrain-v3`。随后重新启用并启动 `CjlPileFull` 与 `CjlPileFullSupervisor`；两者均为 Running/Enabled。服务器复核恰一个下载 `curl`，`nice=19`、`ionice=idle`，`training-active` 不存在，下载已从原 `.part` 断点恢复。
- 正式中文总结见 `worklogs/2026-07-16-nlp-stage-ab-final-summary.md`。遗留风险为：Stage 15.8 经验门禁 5 失败；服务器仍只枚举 7 张 GPU，额外的 `GPU-dc6...` 仍不可建立 CUDA context；全量 Pile 下载尚未完成但已恢复低优先级串行运行。以上均不冒充已解决。
