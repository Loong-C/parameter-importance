# S0.4 模型、数据与 manifest

## 目的

为后续各阶段建立唯一、完整、离线可用的模型/tokenizer/数据资产身份，并使下载中、损坏、未验收或游标覆盖不足的文件无法被实验入口选中。

## 当前依据

- 已有 Pythia 14M step0、31M deduped step0、160M deduped step0/step512 模型目录。
- 31M 顶层 manifest 当前无法解析为合法 JSON，资产目录存在但 gate 不通过。
- Pythia 410M 统一初始化资产缺失。
- 已有 SST-2、WikiText-103 raw 和 Pile；MNLI、RTE 缺失。
- Pile shard 0–4 已有完整文件，shard 5 正由既有链路写入 `.part`；旧 `READY` 只覆盖历史最小前缀。
- 当前 `manifests` 还残留历史 `finalize.lock` 等运行痕迹，任何处理前都要先核对锁与进程。

## 前置条件

- G1 通过，资产、缓存、manifest 和临时目录边界固定。
- 下载链路只使用 `Agent` 文档允许的现有方式，不记录签名 URL 或临时 CDN 地址。
- 在决定下载范围前，先冻结后续阶段的数据预算和模型清单。

## 实施步骤

### 1. 定义资产 manifest schema

1. 为模型、tokenizer、数据集和外部源码定义不同资产类型。
2. 每项至少记录名称、来源、不可变 revision、文件相对路径、大小和 SHA-256。
3. 模型项记录架构、参数量、dtype、config 和初始化身份。
4. tokenizer 项记录 vocab 大小、特殊 token、归一化与版本身份。
5. 数据集项记录 split、样本数、字段、原始 revision 和预处理版本。
6. 流式/分片数据记录 shard 顺序、索引身份、样本游标与 token 覆盖范围。
7. 记录生成时间、生成器版本、关联 Git 提交和验收状态。
8. schema 明确区分 `downloading`、`downloaded`、`verified`、`ready` 和 `invalid`。
9. 固化状态转换职责：获取进程只能发布 `downloading/downloaded`，独立校验器只能发布 `verified/invalid`，离线语义检查和 gate 汇总器才可发布 `ready`。
10. 为每次转换记录前态、后态、检查摘要、操作者/进程身份和时间；禁止跳过 `verified` 直接标记 `ready`。
11. 对同一不可变资产身份，校验失败进入 `invalid` 后不得原地改写成成功；修复或重新获取必须产生新的候选 manifest，再重新走完整转换。

### 2. 建立可复用资产获取入口

1. 让每次获取以逻辑资产规范为输入，规范中固定来源、不可变 revision、预期文件、大小和已知哈希。
2. 对每个目标对象使用独立锁和独立 `.part`，不同资产不能共享临时文件。
3. 已有 `ready` 资产先做快速身份检查；完全匹配时幂等退出，不重复下载或改写 manifest。
4. 支持受控断点续传，并验证服务端范围响应、已有前缀和预期总大小。
5. 重试次数、退避和总超时有界；失败保留 `.part`、最后有效字节和结构化原因。
6. 完成下载后先核对大小、revision 和 SHA-256，再原子发布最终文件。
7. manifest 只在最终文件全部验证后从 `downloaded` 转为 `verified`，离线加载和语义检查通过后才转为 `ready`。
8. 提供 verify-only 模式，用于复核已有资产而不访问网络或改写文件。
9. 日志只记录稳定来源标识和 revision；签名 URL、认证信息和临时 CDN 地址不进入参数、日志或 manifest。
10. 为下载中断、错误范围响应、错误哈希、锁竞争、重复执行和原子发布建立测试。
11. 当前活动 Pile 全量下载继续使用既有受控链路；新入口只做只读状态接入和完成后验收，不在中途替换下载器。

### 3. 审计现有模型资产

1. 列出每个模型目录的预期与实际文件集合。
2. 校验每个文件的大小和 SHA-256。
3. 验证 manifest 能通过 JSON/schema 解析。
4. 对 31M 非法 manifest 保留原文件作为诊断证据，定位生成或传输问题。
5. 从已验真的目录文件和固定上游 revision 重新生成合法 manifest，采用临时文件加原子发布。
6. 在完全离线模式加载 config、tokenizer 和 safetensors。
7. 核对架构、参数量、tensor 数量和 tokenizer vocab。
8. 对 step0 资产确认它能作为统一初始化，而不是误把训练后 checkpoint 当初始化。

### 4. 准备缺失模型资产

1. 冻结 Pythia 410M 所需仓库、step/revision 和文件清单。
2. 明确只需要统一初始化、config 和 tokenizer，还是还需要官方对照 checkpoint。
3. 在下载前核对预计大小和大盘容量。
4. 下载到对象专属临时路径，不覆盖已存在目录。
5. 完成大小、上游哈希/revision 和本地 SHA-256 验证后原子发布。
6. 完成离线加载和参数量核对。
7. 将 410M 初始化身份写入 Stage 5/6 前置 gate。

### 5. 冻结预训练数据合同

1. 为 Stage 1–5 分别确定最大训练 step、global batch、sequence length 和样本游标范围。
2. 将训练计划换算为需要覆盖的样本数、token 数和 Pile shard 范围。
3. 明确使用完整 Pile 还是可证明覆盖训练游标的前缀。
4. 把 shard 顺序、索引规则和跨 shard 行为写入数据合同。
5. 若预算尚未确定，将相应 Stage 4/5 资产 gate 标记为阻塞，不用“下载中”代替合同。

### 6. 安全处理正在下载的 Pile

1. 只读核对现有下载任务、对象锁、`.part` 大小和最后更新时间。
2. 不启动第二套下载器，不修改锁、元数据或活动 `.part`。
3. 只有既有下载器完成大小、固定哈希和原子改名后，才把新 shard 纳入正式 manifest。
4. 对每个完成 shard 重新核对顺序、大小和 SHA-256。
5. 对 `document.idx` 记录固定身份并验证它覆盖数据合同所需游标。
6. 对 reader 进行边界样本与官方/参考实现对比。
7. 将完整下载状态与 Stage 0 环境 `READY` 分开，避免一个旧标记掩盖新增 shard 状态。

### 7. 审计并补齐下游数据

1. 对 SST-2 的三个 split 核对 revision、文件、字段和样本数。
2. 为 MNLI 固定 train、matched/mismatched validation/test 等实际使用 split。
3. 为 RTE 固定 train/validation/test 等实际使用 split。
4. 明确标签映射、缺失标签处理和文本字段拼接规则。
5. 固定 tokenizer revision、最大长度、截断与 padding 策略。
6. 将原始缓存与派生预处理数据分开，派生数据记录父资产和配置摘要。
7. 完成完全离线读取和最小 batch 构造。

### 8. 建立资产解析与拒绝规则

1. 训练入口只接受 manifest 中状态为 `ready` 的逻辑资产 ID。
2. 通过逻辑 ID 解析绝对路径，不在实验配置中散布硬编码路径。
3. 路径包含 `.part`、临时后缀或未完成状态时立即失败。
4. schema 解析失败、文件缺失、大小不符或 revision 不明时立即失败。
5. 大文件运行前可用快速元数据检查；首次验收和发布时必须完成全量 SHA-256。
6. 每次运行把实际选择的资产 ID 和 manifest 摘要复制到 provenance。

### 9. 验证 manifest 原子性与可追溯性

1. manifest 先写临时文件并通过 JSON/schema 校验。
2. 所有引用文件通过后再原子发布最终 manifest。
3. 故意提供截断 JSON，验证资产解析器拒绝它。
4. 故意提供缺失文件或错误大小，验证 gate 失败。
5. 验证两个 revision 相同但内容不同的资产不能获得同一 ID。
6. 保留历史 manifest，不覆盖已经用于旧实验的身份记录。
7. 在受控网络阻断或连接审计条件下执行离线加载，记录零外连尝试；若组件尝试联网则 gate 失败。

## 产出

- 模型、tokenizer、数据集和分片数据的 manifest schema；
- 资产状态转换矩阵、责任边界与转换审计记录；
- 可复用、幂等、带锁和原子发布的资产获取/复核入口；
- 现有资产审计报告；
- 修复后的 Pythia 31M manifest 及其问题记录；
- Pythia 410M 统一初始化资产与 manifest；
- Pile 数据预算、游标覆盖报告和完成 shard 清单；
- SST-2、MNLI、RTE 的固定资产与预处理 manifest；
- 资产解析器、离线加载 smoke test 和拒绝路径测试。

## 核验标准

- 每个 `ready` 资产的 revision、文件集合、大小和 SHA-256 完整，manifest 通过 schema。
- 每个 `ready` 资产都有连续的 `downloading → downloaded → verified → ready`（或既有资产的等价 verify-only）审计链；非法跳转、越权发布和 `invalid` 原地翻转均被测试拒绝。
- 模型和 tokenizer 在断网条件下加载，架构、参数量、vocab 和特殊 token 与 manifest 一致。
- Pythia 31M 非法 JSON 已被可追溯地替换为原子生成的合法 manifest，原问题有记录。
- Pile 输入范围完全覆盖对应阶段冻结的最大样本游标；只下载但未证明覆盖不能通过。
- 活动 `.part`、锁或临时元数据从未被解析为正式输入。
- SST-2、MNLI、RTE 的 split、样本数、标签映射和预处理摘要可复核。
- 所有正式资产能完全离线加载，运行期间不发生隐式网络访问。
- 资产获取入口通过断点续传、错误响应、错误哈希、锁竞争、重复执行和敏感信息脱敏测试。
- 资产解析器对非法 JSON、错误哈希、缺失文件、未知 revision 和未完成状态均明确失败。

## Gate 与后续依赖

- 本子任务形成 **G3 资产 gate**，并按阶段细分：
  - **G3-S1**：Pythia 14M 和固定调试数据；
  - **G3-S2**：Pythia 31M 和重复采样输入；
  - **G3-S4**：Pythia 160M、SST-2 和预训练游标覆盖；
  - **G3-S5**：Pythia 410M 统一初始化与正式预训练数据预算；
  - **G3-S6**：SST-2、MNLI、RTE 和统一 tokenizer/预处理。
- Stage 0 完成要求计划范围内所有 G3 子 gate 通过；任何阶段开始前还需复核其对应资产没有漂移。
