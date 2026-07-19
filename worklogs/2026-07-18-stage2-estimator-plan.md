# 2026-07-18 第 2 阶段过估计与无偏估计器验证计划

- 任务范围：通读 `Agent/` 全部现行文档、`plan/general_plan.md` 和相关数学规格，结合本机、服务器、资产与旧实验只读现状，将 Stage 2 展开为细粒度、可执行、可核验、可追溯的计划。
- 当前状态：Stage 2 计划文档已完成主体编写和多轮只读审阅；没有实现 Stage 2 代码，也没有运行 reference、pilot 或正式实验。
- 工作分支：`main`

## 2026-07-18 — 计划编写与审阅

### 目标与范围

- 为每个子任务分别说明目的、前置条件、原子化步骤、产物、结果/可视化核验标准、Gate 和后续依赖。
- 主目标固定为 checkpoint 不变时、冻结经验分布上的局部梯度平方贡献 \(C^\star=\mu^2\)，主实验 `eta_eval=1`。
- 比较 raw、等总 draw 预算 double 和 microbatch U-statistic；signed U 不截断，`M=2` 必须与相同两半 double 一致。
- 不把本阶段结果外推为 AdamW 实际更新、完整参数路径贡献、数值求积或剪枝效果结论。

### 实际修改

- 新建 `plan/stage2/README.md`，记录阶段目标、边界、当前事实、依赖图、Gate 总表和退出分支。
- 新建 S2.1–S2.11 十一份子任务文档，覆盖预注册、Stage 0/1 交接、资产与抽样、reference、配对 runner、pilot、正式 sweep、统计、成本、可视化/决策和交付。
- 新建本工作日志；未修改并发任务正在编辑的 `docs/mathematics.md`、`plan/general_plan.md` 或 `plan/stage0/`、`plan/stage1/`、`plan/stage3/`。

### 只读现状核对

- 本机：Windows、Python 3.12.4、PyTorch 2.10.0 CPU-only，无 NVIDIA GPU，未安装 Transformers/Datasets/Accelerate；只适合文档、轻量 CPU 测试和 Git/SSH 调度。
- 已提交基线：本机、GitHub 与服务器 `main` 均为 `34966d0819a5229569169bfe436afe9058c0ed24`；当前共享本机工作树含多阶段并发草稿，未形成可安全提交状态。
- 服务器环境：Python 3.12.3、PyTorch 2.12.1+cu126、Transformers 4.57.6、Datasets 4.8.5、Accelerate 1.14.0，`pip check` 通过；正式 GPU 计算只能在服务器进行。
- GPU：驱动/PCI 可见设备数与 NVML/PyTorch 枚举不一致；缺失 PCI `0000:4f:00.0`，当前 `cuda:0` 映射 PCI `0000:50:00.0`，有 24 次不可纠正 ECC 和 pending row remap。当前健康四卡/DDP Gate 未通过。
- 存储：项目大盘空间充足，根盘余量很小；缓存、模型、结果和临时项必须位于 `$DATA_ROOT`。
- 模型：14M step0 与 31M deduped step0 存在；31M 两份 manifest 带 UTF-8 BOM，标准 JSON 解析失败；两个模型都缺 Stage 2 所需早期/中后期 checkpoint。
- 数据：默认只允许已验收 shard 0 前缀两个精确文件与 sample ID `[0,524288)`；shard 5 `.part` 仍由既有受控下载写入，计划禁止触碰或停止该任务。
- 同步：本机与服务器 `Agent/worklogs.md` 哈希不一致，正式进入 Gate 前必须关闭。
- 旧证据：历史 raw/U 结果存在 B/M 混杂、有限 reference、单模型短程、formula-only cost 和 positive clamp 等缺口；只转化为风险与回归用例，不作为本阶段通过证据。

### 关键审阅修正

- 把 estimand 冻结为验收 sample frame 的经验分布，所有 stream 独立有放回抽样；draw IDs 独立，sample-ID 偶然碰撞保留并审计。
- 将理论目标、无偏 bias reference、交叉 reference 和低方差 ranking reference 分开命名；有限均值平方不再承担无偏性主 Gate。
- 把 reference sizing 与最终 reference 分到独立 streams：sizing 只冻结 margin/样本量，最终 A/B 固定长度 one-shot 运行；禁止可选停时、延长或“重抽到通过”，失败即本轮阻断。
- 从 reference block mean 方差恢复单 sequence 方差时显式乘回 block size，并以单 sequence 诊断坐标验证。
- 科学等价 margin 只由预注册应用尺度决定；reference 半宽和数值误差改为独立精度 Gate，不能反向放宽 margin。
- 主方法判定冻结一个共同 `B_primary`/`M_primary`，并要求两个模型、三个训练阶段共六个主 cells 逐项通过 intersection-union；其他 B/M/scope/top-q 为 secondary。
- 为 model/layer/module endpoint 明确定义 sizing-only 的 `S_ref`、raw 预测偏差和绝对 floor；G2.3 对全部候选 B 中最严格 margin 验收，消除与 `B_primary` 的 Gate 循环。
- `B_primary/M_primary` 使用预注册的确定性扫描与 tie-breaker，只读取可运行性、最坏方差/R 和资源字段，不读取方法均值、方向或优劣。
- 唯一定义 parameter-vector NMSE 分母、one-shot reference trace 校正和正 floor；两阶段 bootstrap 同时计量 estimator repetitions 与 reference blocks。
- Stage 2 只验收并复用 `G1-EXIT` 内核；formula/provider/registry/loss/DDP 等实质变化必须返回 Stage 1。
- 成本拆为共享科学 runner、方法独立 fixed-state 和在线训练增量三种口径；方法决策只使用在线增量口径的方法独立 anchor。
- 统一 G2.0–G2.8 及 `G2.2-dev`、`G2.4a/b`、`G2.7a/b`，并收紧 `$DATA_ROOT` 路径、缓存、原子发布、sealed marker、精确 bundle 清理和 GPU 管理权限边界。

### 当前阻塞与下一步

- 当前计划完成不代表任何实验 Gate 已通过。
- 正式开工至少仍被健康四卡/DDP、`G1-EXIT`、31M manifest、缺失 checkpoint、Agent 哈希同步和现行代码缺失阻塞。
- 工作树含并发未提交成果，本次不暂存、不提交、不推送、不建立 bundle，也不改服务器文件。
- 待最终结构/链接/术语审查和文件摘要完成后，在本日志追加文档级验收结果。

## 2026-07-18 — 最终文档审查与摘要

### 审查结论

- `plan/stage2/` 共 12 份 Markdown、1,670 行、123,536 字节；11 个子任务均包含目的、前置条件、细粒度实施步骤、产出、核验标准/Gate 和后续依赖。
- 全部相对链接存在，Markdown 围栏和行内/展示数学定界符成对，无 UTF-8 BOM、尾随空白、`TODO/TBD/FIXME` 或占位文本。
- 三轮独立只读审阅分别覆盖数学、仓库/Stage 1 边界、服务器安全和最终 Gate 闭环；终审确认没有 P0/P1 阻断。
- 终审重点关闭了：有限均值平方 reference 正偏、reference 可选停时/接受性重抽、`delta_sci` 与 `B_primary` 循环、B/M 择优空间、sequence 方差 block-size 缩放、NMSE/`Var_ref` 公式、跨六个主 cells 归并、G2.2/G2.4b mapping 时序、共享/部署成本混用和 G2.8 自循环。
- 本次仅完成计划与只读事实核对；没有把文档 QA 写成任何实验 Gate 已通过，也没有提交、推送、同步或修改服务器。

### Stage 2 文件摘要

集合摘要输入为按文件名字典序排列的 `文件名<TAB>文件SHA-256`，以 UTF-8/LF 连接。

| 文件 | SHA-256 |
|---|---|
| `01_scope_hypotheses_and_preregistration.md` | `0e8b1944968205a4484f3b414294b3209a5e53b84babd105fdce267b2998e66f` |
| `02_stage1_handoff_and_fixed_state_contract.md` | `5f0d753e493a6bf6963784823450e34a7f8fac67d830a9301230bca463c0ed65` |
| `03_assets_checkpoints_and_sampling.md` | `fc365aabceb342f100098e1631b4ee60228d5bf351f58f537467a9578ef29db7` |
| `04_reference_target.md` | `ac1647bd33664f906a00aef4c422930216cea25c9b255cb7fdd022c1999d75a7` |
| `05_paired_estimator_runner.md` | `1d44637261bd3c5bcb7d288beefd941cc896a0f0a16b5d2f32440fc37fe8296f` |
| `06_pilot_and_matrix_freeze.md` | `10f2223758060c1118978332005c5a6c3a1c026eff924c899f4aa11169867c6a` |
| `07_main_sweep.md` | `fab72198833fced4b9ddaf3afc7484a06dac85683bbc46c772319295b38166c3` |
| `08_statistics_and_robustness.md` | `c1c0176c89f16e615f8f62c7684ccf9a62d9a8fbff5935df19264938b6dbc689` |
| `09_cost_and_system_validation.md` | `e7d6bb29203c82280506bafcad4f745ac83ede220879185defc9d3c1d1737a85` |
| `10_visualization_reporting_and_decision.md` | `523274baad59bbe8a2e8926f0033609f05a255df987b58da2d0eefcb6509806a` |
| `11_delivery_and_exit_gate.md` | `e76f8b68dc90684fab9323c915f4b49502105f2d4cf151a14fef4c0d0cbd64df` |
| `README.md` | `c4dcceb8a899f6a5c1b6bd34f9efaf96181a8701cba3a6e4f20294d574263b07` |
| 文件集合摘要 | `b9905dd8aea2c83ac7fd1a7bf514dd3452d2e9d0090c468d8c8cc38e7ccf84be` |

### 最终状态

- Stage 2 计划书：文档级完成并通过结构、数学、服务器边界和 Gate 终审。
- Stage 2 实验：尚未开始；当前仍受健康四卡/DDP、`G1-EXIT`、31M manifest、缺失 checkpoint、Agent 哈希同步和现行实现缺失阻断。
- Git/多端同步：未执行。共享工作树仍含其他阶段的并发未提交成果，不能在未完成整树审查时建立局部提交。
