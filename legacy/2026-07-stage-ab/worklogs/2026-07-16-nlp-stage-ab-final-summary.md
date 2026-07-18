# NLP Stage A/B 正式中文总结报告

## 1. 完成范围

本轮已完成 `feat/nlp-pythia-stage-ab` 的 Stage A 方法验证、正式 Stage B trapezoid v3 预训练、SST-2 离线基线、controlled 与 best-performance 实验、六项正式 pruning、三项 plan-22.4 功能复用干预，以及 `configs/report/stage_ab.yaml` 总报告。所有正式训练与评估均在服务器固定离线环境中执行；诊断运行、故障运行和 best-validation 参考均未冒充正式来源。

本轮接管的七项剩余作业已按以下顺序严格串行完成：

1. `sst2-direct-pruning-2027`
2. `sst2-pretrained-pruning-1234`
3. `sst2-pretrained-pruning-1337`
4. `sst2-pretrained-pruning-2027`
5. `reuse-intervention-1234`
6. `reuse-intervention-1337`
7. `reuse-intervention-2027`

## 2. 主要实验结果

### Stage A 方法门禁

- estimator positive-only v3 的 parameter/layer/module `M>=8` 稳定性最低值分别为 `0.9299466/0.9166667/0.95`，全部通过；signed cancellation 失败诊断继续保留。
- 数值积分 v1 左端点层级 Spearman 最低 `0.56667 < 0.90`，正式失败。独立 trapezoid v2 使用新数据区间，parameter nonzero Spearman、layer Spearman、top-5% overlap 最低分别为 `0.98322/0.96667/0.97389`，全部通过。

### Stage B 预训练

- 正式来源为从 step0 开始的 deterministic trapezoid v3；最终 step512 checkpoint 的 manifest、`_SUCCESS` 与全部文件哈希通过。
- 固定 WikiText perplexity 从 `63585.31` 降至 `423.65`；相对公开 Pythia step512 的 PPL 比率为 `0.6396877 <= 1.15`。
- strict algorithms、`warn_only=false`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、eager/math SDPA 与禁用非确定性后端均有环境证据。

### SST-2 controlled 与最佳性能

- 三 seed controlled 路线平均最佳验证 accuracy：direct `0.8100153`，pretrained `0.8176606`；各路线最低值分别为 `0.7970183` 与 `0.8165138`，均高于多数类基线 `0.5091743`。
- best-performance 以最佳验证 NLL 选择：
  - direct：`lr=3e-4`，best step800，accuracy `0.8061927`，NLL `0.4173166`；
  - pretrained：`lr=1e-4`，best step1000，accuracy `0.8188073`，NLL `0.4116327`。

### Pruning

- 六项 pruning 均通过 manager 正式验证，每项精确 594 条记录；所有 `_SUCCESS`、CSV/JSON、curves、rank 与 provenance 哈希一致。
- 每项都绑定对应 controlled 路线最终 step3156 模型和 importance；best-validation 明确 `used_for_pruning=false`，未混用 best-performance 或诊断产物。
- 报告门禁显示：
  - 高 U-stat positive 参数剪枝比低重要性剪枝造成更大损害：通过，覆盖 64 个 curve groups；
  - U-stat 区分度不低于 raw：通过，absolute gap AUC 中 `u_statistic=0.0725815`、`raw=0.0716858`；
  - U-stat 与 double sampling 的排序相关和剪枝效果“相近或更好”：失败，`positive pairwise rank rows=18; pruning_similar=False`。

### 功能复用干预

- 三个 seed 均精确 60 条，共 180 条正式记录；每份结果均通过 `_SUCCESS`、哈希、实验 ID、provenance、因子网格和来源校验。
- 干预同时核验共同的 v3 预训练 step512 与对应 pretrained controlled step3156。三 seed 使用完全一致的 eligible 布局：49 个张量、123,568,128 个标量。
- `coverage_gate_passed=true`，required/valid seeds 均为 `1234/1337/2027`，且 `ordinary_pruning_substituted=false`。
- 功能效应随 mask 来源、比例和 global/layer-balanced 分配变化；计划没有预注册 effect-size 通过阈值，因此仅作描述，不进行事后阈值放行。

## 3. 最终 Stage 15.8 门禁

| # | 门禁 | 结果 |
|---:|---|---|
| 1 | 预训练质量门槛 | 通过 |
| 2 | SST-2 两路线高于多数类基线 | 通过 |
| 3 | 高 U-stat positive 剪枝损害大于低重要性剪枝 | 通过 |
| 4 | U-stat 区分度不低于 raw | 通过 |
| 5 | U-stat 与 double sampling 排序和剪枝效果相近或更好 | **失败** |
| 6 | signed/positive-only 至少一套产生稳定功能排序 | 通过 |
| 7 | 中断恢复和多 GPU 可复现 | 通过 |

总报告没有 `missing` 输入。门禁 5 是正式经验结果，未更改阈值、未重命名或改写为通过。

## 4. 关键产物与哈希

数据根目录：`/home/sophgo13/cjl/storage/parameter-importance`

- 总报告目录：`reports/stage-ab-minimum-loop`
  - `summary.json`：`53d4aa80688245ab0975949fee197a96923716a973c5cf72f8f4d7f0f5547f70`
  - `report.md`：`1504865798a34816908626c6d9e7ed6adc2d2aa53e35eb3805ed68a49ea5cadd`
  - `tables/stage_15_8_gates.csv`：`935994d600bdde2925983b0b5345a049a4932c5a3aa66bfe2e057cf67135fcb4`
  - `tables/functional_reuse_intervention.csv`：`38501d01fc1e360e18757be60b108699fc09d8994e45c3ee153419370b6d95f9`
  - `tables/functional_reuse_intervention_summary.csv`：`5552c3a70d2e4e823e5442ebefe0791c559e16840e8beabb2e3187333dc93827`
- 报告共登记并实际生成 48 个图文件、30 个表文件，测量来源文件数为 77。
- pruning 正式目录：`runs/stage-b-sst2-{direct,pretrained}-pruning-seed{1234,1337,2027}`
- reuse 正式目录：`runs/stage-b-sst2-reuse-intervention-seed{1234,1337,2027}`

## 5. 回归、Git 与多端状态

- 服务器全套报告绑定代码提交 `7e3a3d8163e0d00553e45daa1d312494d3c4ee71`：151 tests、0 failures、0 errors、0 skips。
- 显式 Linux 集成测试重跑为 `2 passed in 22.09s`；重生成的 `results/reproducibility/stage-ab.json` 绑定同一代码提交，多 GPU 等价与 checkpoint resume 均通过。
- pruning 阶段日志里程碑提交为 `2f874cbf6dc0b4a590a1aec4724969a358d204c8`，本地、origin 和服务器已 fast-forward 一致。
- 本总结形成最终日志提交后，将再次核对本地、`origin/feat/nlp-pythia-stage-ab` 与服务器 HEAD。用户原有的 `Agents/工作日志与多端同步规范.md` 删除始终未暂存、恢复或覆盖。

## 6. 下载恢复与监控清理

- 已删除 `CjlNlpTrainingChain-stage-b-downstream` 和 `CjlNlpTraining-stage-b-pretrain-v3` 两个旧训练监控计划任务。
- `CjlPileFull` 与 `CjlPileFullSupervisor` 已重新启用并运行。
- 服务器验收时恰一个下载 `curl`，`nice=19`、`ionice=idle`，`training-active` 不存在；下载从原 `.part` 断点恢复。

## 7. 遗留风险

1. Stage 15.8 门禁 5 正式失败；这是科学结论，不是运维未完成项。
2. 服务器仍只枚举 7 张 GPU，`GPU-dc6...` 无法建立 CUDA context；恢复硬件并重新通过 UUID 映射/互斥门禁前仍不得恢复双组并发。
3. 全量 Pile 下载尚未完成，当前仅恢复为低优先级、单对象、断点续传模式。
