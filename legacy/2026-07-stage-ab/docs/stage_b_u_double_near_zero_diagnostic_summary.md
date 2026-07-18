# Stage B.5：U-stat 与 double sampling 对称剪枝诊断总结

## 结论摘要

本次诊断已完成，且没有重新训练。六个既有 SST-2 最终 checkpoint、importance 和原始 pruning 结果全部通过来源与哈希门禁，只新增了 `double_near_zero` 的 108 行评估。

最关键的结论是：现有证据不再指向“signed U-stat 估计器无法替代 double sampling”，而是指向 positive-only 变换改变了估计对象和功能排序。

- 原 Gate 5 语义复现为 `68/72`，恰好失败 4 项，正式失败状态不变。
- 只把 double 的低端从“最负”改成 near-zero 后为 `69/72`，只修复 1 项。
- signed U 与 signed double 都使用 near-zero 低端后为 `72/72`，全部满足同一条功能相近判据。
- signed U 对 double 的参数 Spearman 为 `0.962596–0.969190`，六份结果的 layer 与 module Spearman 都为 `1.0`。
- double-near-zero 自身在 64 个非零比例主指标组中全部得到正 gap，因此本次结果不支持“double sampling 本身不够好”这一解释。

因此，在当前 Pythia-160M、SST-2、三 seed 范围内，原始的“用计算更便宜的 signed U-stat 近似/替代 double sampling”重新得到强支持。尚未解决的是 signed U 的跨重复稳定性；positive-only 可以保留为操作性分数，但不能再与无偏 signed U 视为同一个估计量。

## 诊断设计

固定来源为 direct/pretrained × seed `1234/1337/2027` 六个正式训练结果。每个来源只新增一个方法：

```text
double_near_zero = 按 abs(累计 signed double score) 从小到大剪枝
```

覆盖 9 个比例 `0, 0.1%, 0.5%, 1%, 2%, 5%, 10%, 20%, 30%` 与 `global`、`layer_balanced` 两种 allocation，共 `6 × 9 × 2 = 108` 行。每次评估均从同一最终模型恢复，验证集固定为 872 个样本；eligible 参数为 49 个 tensor、123,568,128 个标量。

三组比较均以训练 seed 为聚合单位。对 accuracy，effect 为 baseline 减当前值；对 NLL，effect 为当前值减 baseline。gap 定义为 high effect 减 low/near-zero effect，正值表示剪高分参数更伤模型。沿用旧判据：

```text
U gap >= 0 且 U gap >= 0.8 × double gap
```

该判据在 B.5 中只用于诊断，不替代、不追溯修改正式 Gate 5。

## 功能结果

| 比较 | 适用组 | 通过 | 失败 | 结论 |
|---|---:|---:|---:|---|
| 原 Gate 5：positive U low 对 signed double low | 72 | 68 | 4 | 正式失败原样复现 |
| positive U low 对 double near-zero | 72 | 69 | 3 | 低端语义不对称只解释 1 项 |
| signed U near-zero 对 signed double near-zero | 72 | 72 | 0 | 对称 signed 比较全部满足 |

原语义失败条件：

| 路线 | allocation | ratio | metric | U gap | double gap | 要求的 U gap |
|---|---|---:|---|---:|---:|---:|
| direct | layer_balanced | 30% | NLL | 0.158069375 | 0.216326754 | 0.173061403 |
| pretrained | layer_balanced | 10% | NLL | 0.165199126 | 0.206511805 | 0.165209444 |
| pretrained | layer_balanced | 20% | NLL | 0.127636121 | 0.214585106 | 0.171668085 |
| pretrained | layer_balanced | 30% | NLL | 0.077034364 | 0.221317902 | 0.177054322 |

改用 double near-zero 后，direct 30% 以及 pretrained 20%/30% 仍失败，pretrained 10% 被修复。signed 对称比较没有失败项。

四种高/低端定义在所有 64 个非零比例 accuracy/NLL 组中都保持正 gap：

| 分数与低端定义 | 正 gap | 非正 gap | 平均 gap |
|---|---:|---:|---:|
| positive U / low | 64 | 0 | 0.243689749 |
| signed U / near-zero | 64 | 0 | 0.249810659 |
| signed double / most-negative | 64 | 0 | 0.249244916 |
| signed double / near-zero | 64 | 0 | 0.249282160 |

## 排序与符号分布

| 分数对 double | 参数 Spearman | layer Spearman | module Spearman |
|---|---:|---:|---:|
| positive U | 0.913501–0.935414 | 0.846154–1.0 | 0.9–1.0 |
| signed U | 0.962596–0.969190 | 1.0 | 1.0 |

最终累计 signed U 只有 `0.9016%–1.0109%` 的标量为负；signed double 的负标量比例为 `1.2191%–1.9017%`。两者的净质量/绝对质量也都接近 1：signed U 为 `0.998596–0.999275`，double 为 `0.996679–0.998398`。

但 `mean(max(U_r,0))` 不是“只改动最终为负的约 1% 参数”。它在每一步/每次重复上先截断，再累加所有瞬时正贡献。六个来源中，positive-only 总质量是 signed U 净质量的 `1.470–1.606` 倍。这说明大量本应相互抵消的正负波动被系统性保留，足以改变几乎全部参数的相对值和 layer-balanced 高比例剪枝行为。

## 对三个科学问题的更新回答

1. `mean(max(U_r,0))` 不是绝对值，而是 ReLU/单边截断；它合理地定义了一个非负操作性分数，但不再是原 signed 目标的无偏估计量。本次结果表明它不适合作为检验“signed U 是否替代 signed double”的唯一分数。
2. 预训练对 controlled SST-2 的提升有限仍是独立结论，与本诊断无冲突；B.5 没有重新训练，也没有改变 direct/pretrained 性能比较。
3. 先前不能把 U 与 double 视为功能相近，具体源于正式 Gate 5 的 positive U 与 signed double 不完全同对象、低端语义也不对称。现在的对称诊断显示：double 不是明显的坏基准，signed U 才是与它真正相近的量。

## Stage C 前必须明确的事项

- 把 signed U 恢复为科学主估计量；positive-only 仅作为明确标注的操作性 ablation，不得再称为同一无偏估计量。
- signed 比较使用“高正分数 vs near-zero”，double 使用同样语义，避免把 near-zero 与 most-negative 混为“低重要性”。
- 原 Gate 5 仍记录为失败；B.5 是独立、事后但预先固定实现的诊断证据，不能改写历史门禁。
- signed U 的 Stage A 跨重复 layer 稳定性仍是遗留风险。Stage C 正式启动前需预注册稳定化方案和门槛；在该问题解决前，可以说“功能与 double 高度相近”，但不能无条件宣称所有场景都已可替代。
- double sampling 可以考虑从“每步全量主方法”降为较小覆盖的校准基线，以兑现计算节省；具体覆盖率必须在 Stage C 计划中预注册，不能根据结果临时决定。
- 当前证据只覆盖 Pythia-160M、SST-2、三 seed；跨模型规模和任务的外推仍需 Stage C 验证。

## 门禁、产物与哈希

正式目录：

```text
/home/sophgo13/cjl/storage/parameter-importance/results/stage-b-u-double-near-zero-diagnostic-v1
```

精确行数（含 CSV header）：`results=109`、`gaps=1297`、`aggregate=433`、`pairwise=325`、`score-signs=19`。机器记录分别为 108、1296、432、324、18 行。

provenance ID：`fb3e818056fa36e82ec37ddba306e9d924d2676d054e28e37fe444bbe3b35bfa`。

| 文件 | SHA-256 |
|---|---|
| `_SUCCESS` | `cdc3eff0cdeaee155bdb18f249a6b817ebb583d6dfdde536dd0efe299e036ee6` |
| `results.csv` | `b0e48d1de39d8df48dcdff5a4134cb11a336b9b0b3f4e9136c4a48e8bd6ae3f7` |
| `results.json` | `59f98b7a2ce109ba645ea8954d39dcf81ab5ee5139fb26ec9c4d124ffdf3e952` |
| `gaps.csv` | `dcfa30250a7ab04135b363e8672a0ab336d7751058a1b9214ee5acd749551e5a` |
| `aggregate.csv` | `d9dcf94a5a0091387828729d9d0bbd2b4d203d27abf19ef148d5fee66bdc16cf` |
| `pairwise.csv` | `e9dc6b6c11594316e20dbcb0af8c377a139c711fd51455cc0040d83f47905a3f` |
| `score-signs.csv` | `77d3c9377352b22c24542808415a00f10b1b1bfe1f3d145a34511371f8b64f29` |
| `summary.json` | `e679e680067bcc6bb6546d42650d2b869f3b597005fcb42b142430903f4fd5d3` |
| `report.md` | `4a5e2975dfc66c59ecb68514095c3e3ff2d8f7bf575f2daff5c2843895a40385` |
| `provenance.json` | `488634fd744982bfa6ac3f080b318170ee338068905b36771fa530c355ddec4c` |
| `resolved-config.json` | `8725ddc463e94618184baa78f88fb849db911db25881a7e0dc6d6d3696ffcc9b` |
| `runtime-manifest.json` | `e0a7aed42119dea087d96ec0f0b503ab170482b07d5a8c96ac8ef67c7d5b947c` |

六份来源 pruning 均重新通过 `validate-result`：每份精确 594 行，`results`、`curves`、rank、provenance 哈希全部一致。诊断结果所有数值由 runner 的有限性门禁检查，无 NaN/Inf；当前正式目录没有 `FAILURE.json`。

## 回归、失败恢复与运行状态

- 实验代码提交：`78329802193d89e121a2e275d8d3ce312bbe1255`。
- 服务器零跳过报告绑定该提交：`156 tests, 0 failures, 0 errors, 0 skips`；报告 JSON SHA-256 为 `be61c39be40b6ec7307c74fa51e57ffb7f58d194adacff801f99788fad4f05ca`。
- 显式分布式与 checkpoint 恢复测试：`2 passed`；reproducibility 两项均为 true，JSON SHA-256 为 `8a8e74092c7ca3cf3acd1b076423e3b3a0ebb01a54984271ae202f1cf41b444f`。
- 首轮在第一次评估前因确定性 CUDA 环境初始化顺序 fail closed；没有正式记录。失败 staging、`FAILURE.json` 和 Traceback 保留。修复后重新提交、重新跑全套回归，第二轮独立 staging 原子发布成功。
- 完成时 manager `complete=1`、supervisor/worker/训练租约均为 0，GPU 无残留进程。
- `CjlNlpTrainingChain-stage-b-downstream` 不存在；本任务没有创建新的 Codex 定时监控。
- Pile `.part` 保留在 `20,382,468,336` bytes。`CjlPileFull` 与 `CjlPileFullSupervisor` 已重新 Enable 并处于“正在运行”；恢复时公共 DNS 双解析一致性暂时失败，任务正在按既有逻辑自动重试，核验时服务器 curl 为 0。按用户要求不继续长期监督 600GB 下载。

遗留风险只有两类：Stage C 方法学上需解决 signed 跨重复稳定性与 double 校准覆盖；基础设施上 Pile 下载依赖公共 DNS/镜像稳定性，但断点与自动重试均已保留。
