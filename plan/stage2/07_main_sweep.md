# S2.7 14M 主实验与 31M 确认实验

## 1. 子任务目的

按冻结矩阵完成固定 checkpoint 的重复抽样实验，先用 14M 建立完整 batch/M/训练阶段证据，再用 31M 进行独立规模确认。该任务只负责忠实执行和质量控制，不在运行过程中修改假设、阈值或主分析。

## 2. 前置条件

- G2.0、G2.1、G2.2、G2.3、G2.4a 和 G2.4b 全部通过；
- confirmatory matrix、sample mappings、R、Gate 和分析版本已提交；
- 每个 checkpoint 的 reference Gate 已通过；
- 当前 GPU 健康、显存、存储、inode 和运行权限重新核验；
- 没有与正式 wall-clock 口径冲突的重 I/O 或同卡任务；既有下载任务不得被停止或修改，只能选择合适运行窗口。

## 3. 实施步骤

### 3.1 生成正式运行清单

1. 从冻结矩阵展开所有 model/checkpoint/B/repetition 原子单元。
2. 对每个单元附加 M 列表、sample mapping、reference ID、runner commit 和环境 hash。
3. 为 raw 和 double 标记共享单元，避免因 M 展开而重复计算或重复计入样本量。
4. 为每个单元生成预期输出路径和完成标记。
5. 计算总单元数、每个 wave 的分母和预期 A100-hours。
6. 生成只读运行清单哈希，worker 不得自行添加单元。
7. 生成样本复用允许矩阵：同一 repetition 内方法共享与嵌套 M 必须允许；double 两个 draw-ID 半区必须互斥；不同 repetition/B 可按冻结独立 streams 偶然复用 sample ID；任何 reference/pilot/confirmatory 的 seed/draw-ID 复用均禁止。

### 3.2 执行每个 wave 的硬件与 I/O 预检

1. 按 UUID/PCI 而不是历史 index 识别候选 GPU。
2. 检查 GPU 枚举、ECC、row remap、温度、显存占用和计算进程；Xid 只使用当前用户可读日志或管理员提供的受控健康证据，不以无权限读取失败冒充“无 Xid”。
3. 检查项目盘挂载、可用空间、inode、属主和写权限。
4. 检查根盘余量，确认缓存和临时目录没有回落到根盘。
5. 核对 `HF_HOME`、`HF_DATASETS_CACHE`、`TORCH_HOME`、`XDG_CACHE_HOME` 位于 `$DATA_ROOT/cache`，`TMPDIR` 位于本 wave 的 `$DATA_ROOT/tmp/stage2/<run-id>`。
6. 检查 Pile 下载等进程是否正在造成可见 I/O 竞争，并写入 `cost_io_quiescent`。
7. 若科学计算可以运行但成本测量不具可比性，允许完成 estimator 结果，同时把该轮成本标为无效并在 S2.9 单独复测。
8. 任一 GPU 健康异常时 fail closed，不自动换到未验收设备继续同一成本 wave；不执行 reset、功率/时钟/驱动变更或 `sudo`。

### 3.3 执行首个 14M 确认性质量 wave

1. 先运行 14M 初始化 checkpoint 的一个完整 confirmatory B/M wave；其 draw IDs 与 pilot 独立，结果保留在正式分母中。
2. 校验所有 repetition 的 draw/sample 映射、重复率、状态摘要和 finite checks。
3. 校验 M=2 U/double 等价和不同 M 的 mean gradient 一致。
4. 核对完成分母、失败记录、存储增长和 profiler 字段。
5. 只审查质量指标，不根据 raw/U/double 的优劣修改后续矩阵。
6. 质量 Gate 通过后封存该 wave，并进入 14M 其余 checkpoint。

### 3.4 执行 14M 多阶段主实验

1. 按初始化、早期、中后期固定顺序运行 checkpoint waves。
2. 每个 checkpoint 内按 B 从小到大运行，便于尽早发现容量问题。
3. 每个原子单元使用冻结 repetition IDs 和相同重试上限。
4. 每个 wave 完成后生成行数、有限性、draw-ID 唯一性、sample-ID 碰撞率和状态不变报告。
5. 每个 wave 单独原子发布，避免一个 checkpoint 失败破坏其他结果。
6. 在全部 14M 单元完成前，不生成主假设支持/不支持结论。

### 3.5 执行 14M 质量审查

1. 汇总缺失、失败、重试和成功单元。
2. 验证每个 B/M 的实际样本/token 数与 manifest 一致。
3. 验证 raw/double 只计一次，U 按 M 展开正确。
4. 验证 reference 和代码/环境哈希没有在 waves 间漂移。
5. 验证 signed 负值和诊断字段没有因序列化丢失。
6. 若发现实现或数据质量错误，冻结当前结果为失败证据，修复后使用新 run/version 全量重做受影响单元；不得原地覆盖。

### 3.6 执行 31M 确认 wave

1. 重新加载 31M manifests 并执行标准 JSON、哈希和离线加载检查。
2. 重新执行最小 fixed-state 和 M=2 smoke test。
3. 按冻结的 31M 矩阵运行初始化、早期和中后期 checkpoint。
4. 保持与 14M 相同的统计单元、loss reduction、reference 规则和方法定义。
5. 若 31M 使用缩减矩阵，严格按 amendment 中的预注册单元执行。
6. 31M 结果完成前不以 14M 现象外推模型规模结论。

### 3.7 处理失败与不完整单元

1. 运行级异常写入包含阶段、异常类、最后 sample ID 和硬件状态的 FAILURE 记录。
2. 瞬时 SSH 中断不等于服务器 worker 失败；先只读核对 worker、锁和进度。
3. OOM 只允许按相同统计 microbatch、较小 forward chunk 的预注册恢复路径重试。
4. NaN/Inf、状态漂移、manifest mismatch 和 GPU Xid/ECC 不自动重试，先诊断根因。
5. 达到重试上限的单元保留为失败，并进入完成分母。
6. 若失败比例超过预注册上限，停止后续 wave，不能用已成功子集形成主结论。

### 3.8 封存正式原始结果

1. 对每个 wave 核对预期/实际单元数。
2. 核对每个输出 schema、shape、dtype 和有限性。
3. 计算大型数组、repetition 表、进度和失败文件的 SHA-256。
4. 生成 wave manifest 和全阶段 raw-results manifest。
5. 原始结果发布到 `$DATA_ROOT/results/stage2/raw/<run-id>`，写入 sealed manifest/status marker 表示逻辑封存；不通过递归 `chmod` 实现只读，也不清理既有目录。
6. 后续分析只写入 `$DATA_ROOT/results/stage2/derived/<analysis-id>` 的新唯一目录。
7. 在工作日志记录路径、大小、哈希、提交、环境、GPU 和完成状态。

## 4. 产出

- 14M 三阶段完整确认性结果；
- 31M 三阶段确认性结果；
- 每个 wave 的 repetition 表、流式参数摘要、layer/module 汇总和 profiler 数据；
- 完成/失败/重试清单；
- 样本预算、状态不变和 M=2 等价审计；
- wave manifests 和全阶段 raw-results manifest；
- 服务器大型结果目录与 Git 中的小型索引摘要。

## 5. 核验标准与 Gate

满足以下全部条件才通过 **G2.5 数据完整性 Gate**：

- 预注册成功单元全部完成，或所有失败均在完成分母中有明确记录；
- 总失败比例不超过预注册上限，且不存在集中于某个方法/模型的选择性缺失；
- 实际 B、M、有效 token、repetition 和 sample mapping 与矩阵完全一致；
- 14M 与 31M 各覆盖至少三个训练阶段；
- 无状态漂移、采样框越界、draw ID 冲突、选择性重抽、silent skip、NaN/Inf 或 signed clamp；
- 所有 M=2 检查和 mean-gradient 一致性通过；
- 原始结果已封存，哈希和 schema manifest 完整；
- 成本无效 wave 已明确标记，不混入 S2.9 主成本表。
- 样本复用符合允许矩阵；sample-ID 碰撞没有被误判为 draw 依赖，也没有选择性去重。

## 6. 后续依赖

- S2.8 只读取封存的 raw-results manifest。
- S2.9 可复用有效成本记录，并对无效 wave 做同配置复测。
- S2.10 的所有主图必须能追溯到本任务的 wave ID。
