# S2.3 模型、checkpoint、数据与采样框

## 1. 子任务目的

为 Pythia 14M/31M 的多训练阶段固定实验输入，定义唯一经验目标分布 \(\mathcal F\)，并建立 reference、pilot 和 confirmatory 的独立有放回 draw streams。该任务解决当前只有 step0、31M manifest 不能被标准 JSON 解析、Pile shard 5 仍在写入等现实问题。

## 2. 前置条件

- 服务器挂载、空间、inode、权限和健康 GPU Gate 通过；
- 使用 `$DATA_ROOT` 下既有目录规范，不向根盘或仓库写大型资产；
- 资产准备脚本、manifest schema 和离线验收入口已经进入 Git；
- 既有 Pile 下载监督链的进程、锁和 `.part` 状态已只读核对。

## 3. 实施步骤

### 3.1 重新盘点现有资产

1. 列出 14M、31M 目录内的允许文件、大小、mtime 和 SHA-256。
2. 分别检查模型内 manifest 与 `$DATA_ROOT/manifests` 独立 manifest；14M 当前只有全局 asset manifest，目录内 manifest/`SHA256SUMS` 缺失必须被显式记录，可在不改权重的前提下补建或按预注册规则接受已验证的全局清单。
3. 用标准 UTF-8 JSON 解析器验证 manifest，而不是只比较文件存在性。
4. 记录 31M 两份原始 manifest 的精确路径、BOM 解析故障和原始哈希。
5. 在 `$DATA_ROOT/tmp/stage2/<asset-repair-id>` 的唯一目录生成无 BOM 规范 JSON，禁止原地编辑 manifest 或触碰权重。
6. 对规范 JSON 执行标准解析、逐文件哈希和离线加载验收。
7. 验收通过后只对两个预先列出的精确 manifest 目标做原子发布，并保留原始/规范哈希与变更记录。
8. 用发布后的 manifest 重新校验模型目录，不把“只删除 BOM”当作资产内容自动通过。
9. 在完全离线模式分别加载 config、tokenizer 和 model，验证架构、参数量、dtype 和 vocab size。

### 3.2 冻结 checkpoint 选择规则

1. 对每个模型读取可用训练 checkpoint 元数据和总训练进度。
2. 固定三个语义位置：初始化、约 1% 训练进度的早期状态、约 50% 训练进度的中后期状态。
3. 对每个语义位置选择距离目标进度最近的公开 checkpoint；并列时固定选择较早者。
4. 在查看任何 estimator 结果前冻结具体 revision/commit。
5. 记录模型仓库身份、可变标签、解析后的不可变 revision 和上游来源。
6. 若某个目标 checkpoint 不存在，不得临时换成“表现看起来更合适”的状态；先登记 amendment，再按同一规则选择替代。
7. 受网络阻塞时保留缺失状态，不使用未完成下载或临时缓存进入实验。
8. 明确 14M 非 deduped 与 31M deduped 的数据版本差异；两模型独立检验同一估计器预测，不把跨模型效应差解释成纯规模效应。

### 3.3 下载并验收缺失 checkpoint

1. 为 14M 和 31M 的每个缺失 revision 分别建立下载任务和目标路径。
2. 将临时文件写入 `$DATA_ROOT/tmp/stage2/<checkpoint-download-id>` 或 `$DATA_ROOT/cache` 下受控项目缓存，不直接覆盖最终模型目录。
3. 下载运行所需的最小文件集，避免重复权重格式和训练状态文件。
4. 对每个文件核对上游大小、下载大小和 SHA-256。
5. 生成模型目录内 manifest、独立资产 manifest 和 `SHA256SUMS`。
6. 只有全部校验与离线加载通过后，才原子发布到 `$DATA_ROOT/models` 下按不可变 revision 命名的新目录；不得覆盖现有 step0。
7. 下载任务使用独立 run ID、临时路径和锁，不复用 Pile 监督任务、锁或任务名。
8. 分别记录下载失败、外部网络阻塞和安全重试状态；不修改现有 SSH/DNS 拓扑。

### 3.4 固定数据统计单元

1. 将一个固定长度 packed sequence 定义为主统计单元。
2. 明确输入 token 数、目标 token 数、移位规则和忽略位置。
3. 确认主数据没有 padding 或不同有效 token 数；若有差异，启用加权 U 路径。
4. 记录 sequence 全局索引、来源 shard、字节范围或可重建定位信息。
5. 检查采样框中的候选 sequence 窗口彼此不重叠，且每个 sample ID 唯一映射到一个固定窗口。
6. 抽样检查相邻 sequence 的内容哈希和索引行为，识别显著内容重复或异常相关块。

### 3.5 冻结可用 Pile 范围

1. 将 Stage 2 allowlist 固定为 `prefix_coverage.json` 已验收的 `document-00000-of-00020.bin`、`document.idx` 和 sample ID `[0, 524288)`。
2. 只按 allowlist 的两个精确文件路径做读取和哈希，不递归枚举数据目录，不使用可能匹配 `.part/.meta/.lock` 的通配符。
3. 明确排除 shard 1–4（除非另做 amendment 和逐文件验收）、shard 5 `.part`、活动锁、临时元数据和任何正在写入的对象。
4. 用既有独立 reader 与官方 reader 对预注册索引点进行一致性复核。
5. 计算本阶段最大 sample ID 是否落在 `[0, 524288)` 内。
6. 生成 Stage 2 数据范围 manifest，记录精确 allowlist、最小/最大索引、数量、文件哈希和读取 schema。
7. 正式实验期间若更多 shard 完成，不自动扩大当前采样框；新增数据只能进入下一轮预注册。
8. 把验收后的 sample IDs 定义为有限经验分布 \(\mathcal F\)，主结果解释为对 \(\mathcal F\) 的条件推断；全 Pile 或自然语料总体外推只列为局限。
9. 若 reader 可稳定提供 source/document 元数据，则冻结分层字段并审计各 stream 的来源比例；若不能提供，至少按稳定 hash fold、索引间距、loss/token 分布检查局部块相关和分布漂移，并在报告中保留限制。

### 3.6 建立独立抽样 streams

1. 为 `reference_sizing`、最终 `reference_A`、最终 `reference_B`、`pilot` 和 `confirmatory` 分配互不重用的 RNG stream ID 与 seed namespace。
2. 五个 stream 都从同一冻结经验分布 \(\mathcal F\) 独立有放回抽样，从而保持 estimand 相同并满足 raw/double/U 的 iid 抽样合同。
3. 每个抽样位置生成唯一 draw ID，并保存 draw ID、stream ID、sample ID、抽样顺序和生成算法版本。
4. 允许不同 stream 或同一 stream 的不同 draw 偶然命中同一 sample ID；碰撞必须保留并与有放回抽样理论分布核对，禁止用 sample-ID 交集为零作为独立性的定义。
5. 只有 `reference_sizing` stream 可使用嵌套样本量阶梯；最终 A/B streams 的固定长度必须在其任何 sample ID 生成前冻结，不得根据 A/B 观测结果停在“刚好通过”的节点。
6. Sizing 阶梯先生成冻结的随机 draw 顺序，再使用嵌套前缀；不得直接使用 Pile 自然连续顺序构造 512、1024 等节点。
7. Pilot 只读取 pilot stream；confirmatory draw manifests 在矩阵冻结后一次生成并作逻辑封存，失败重试必须复用相同 draw IDs。
8. 为每个 stream 审计 sample-ID 碰撞率、稳定 hash fold、可用 source/document 比例、索引间距和 loss/token 分布；超出预注册容差时检查 RNG/reader，而不是选择性重抽。
9. 分别保存模型/全局 RNG 不变摘要与 sampling generator 起止状态；两类 RNG 不得混为一个“运行前后相等”检查。

### 3.7 定义 repetition 内的配对分组

1. 对每个 B 从 confirmatory stream 按冻结 seed 生成 B 个有放回独立 draw，并分别保存 draw ID 与 sample ID。
2. 允许不同 draw 偶然命中同一 sample ID；记录重复率并与有放回采样的理论范围核对，禁止事后去重或重抽。
3. 按固定 draw 顺序组成 `M_max` 个等大小基础 microbatch。
4. 用预先固定的嵌套规则合并基础 microbatch，得到 M 为 16、8、4、2 等较粗分组。
5. 将前后两个互不重叠的 draw ID 半区作为双采样的 A/B'，保证各含 B/2 个独立 draw；sample ID 因独立有放回抽样而偶然相同不视为同一个 draw。
6. raw、所有 M 的 U 和双采样使用同一总 draw 池，但公式使用的分区关系互不改变。
7. 保存每个 repetition 的 draw-to-sample-to-microbatch 映射哈希，允许重放而无需保存样本内容副本。

### 3.8 验证模型与采样组合

1. 对每个 checkpoint 生成参数注册表并验证结构相同。
2. 从每个抽样 stream 抽取最小 batch 做离线 forward/backward。
3. 检查 loss、有效 token、梯度和参数坐标均为有限值。
4. 检查不同 checkpoint 的 tokenizer、sequence schema 和 loss reduction 完全一致。
5. 记录 checkpoint × data manifest 的组合哈希，正式 run 只接受已登记组合。

## 4. 产出

- 14M/31M 多阶段 checkpoint 清单及不可变 revision；
- 修复并重新验收的 31M manifests；
- 每个 checkpoint 的文件哈希、离线加载和参数注册表报告；
- Stage 2 数据范围 manifest；
- `reference_sizing`/`pilot` draw manifests，以及最终 `reference_A/B`/`confirmatory` 的保留 seed namespaces、容量和生成 schema；最终 A/B 与 confirmatory draw manifests 分别由 S2.4 和 S2.6 在对应规模冻结后生成；
- repetition draw-to-sample-to-microbatch 映射 schema；
- checkpoint × data 组合清单。

大型模型和样本索引保存在服务器大盘；Git 只保存小型 manifest、哈希、schema 和验收摘要。

## 5. 核验标准与 Gate

仅当 14M step0、最小 fixture 和独立 draw-stream schema 可离线重放时，可标记 **G2.2-dev**；它只解锁人工/14M step0 开发 smoke，不解锁正式 reference、矩阵或 confirmatory 结果。

满足以下全部条件才通过 **G2.2 资产与采样 Gate**：

- 两个模型各至少三个 checkpoint 具有不可变 revision、完整哈希和离线加载证据；
- 31M manifest 能被标准 JSON 解析并与模型目录逐文件一致；
- 所有正式 sample ID 只来自完成且验收的数据文件；
- \(\mathcal F\) 的精确 sample-ID allowlist、概率规则和条件 estimand 已冻结；
- `reference_sizing`、最终 reference_A/B、pilot、confirmatory 的 seed namespaces 独立；已经生成的 sizing/pilot draw IDs 可重放，尚未生成的 A/B/confirmatory 有冻结 schema 与容量；sample-ID 碰撞政策已冻结；
- draw generator、seed 派生和 mapping schema 已用 fixture 验证：draw ID 唯一、有放回 sample-ID 碰撞被保留，失败重放确定；正式 repetition mappings 尚不在本 Gate 生成；
- fixture 的 B、M 和双采样两半映射可由 schema/seed 完全重建；正式 confirmatory mappings 由 G2.4b 在矩阵冻结后、读取任何确认性梯度前生成并验收；
- Pile 活动 `.part`、锁和下载任务未被读取或改变；
- checkpoint × data 组合 smoke test 无 NaN/Inf 且状态不变。

若额外训练阶段 checkpoint 尚未到齐，只能继续 S2.2 和 `G2.2-dev` 范围内的非确认性开发；不得进入 S2.6 的正式矩阵冻结。

## 6. 后续依赖

- S2.4 先只消费 `reference_sizing` 冻结固定 \(B_{\mathrm{ref}}\)，随后才生成并一次性消费全新的最终 `reference_A/B` streams。
- S2.6 只消费 `pilot` stream；不能根据 pilot 结果重定义 \(\mathcal F\) 或 seed namespace。
- S2.7 只消费预先生成并封存的 `confirmatory` repetition 映射。
- 任何资产或采样 amendment 都必须使旧组合哈希失效并重新通过本 Gate。
