# S1.2 代码架构与参数坐标注册表

## 1. 目的

建立最小、可测试、可替换的 Stage 1 代码骨架，使纯代数公式、模型反向、DDP 聚合、优化器集成和持久化彼此解耦。核心目标是防止训练循环细节污染估计器公式，并保证每个参数坐标在保存、恢复和跨设备比较中具有稳定身份。

## 2. 前置 Gate

- `G1-ENTRY` 已通过，或仅在本机进行不计入正式验收的骨架准备。
- `G1-CONTRACT` 已冻结字段名、损失 reduction、参数组和容差。

## 3. 模块职责

新实现至少拆分为以下职责；每项使用窄接口，不直接读取无关全局状态：

- **配置与 schema**：解析配置、校验组合、生成规范化摘要与版本。
- **资产与 manifest**：读取 revision、大小和哈希，兼容明确允许的文本编码。
- **参数 registry**：稳定映射参数名称、张量、参数组和分析标签。
- **loss adapter**：返回 loss numerator、有效计数和 mean loss。
- **gradient collector**：得到未同步、已 unscale 的 local mean gradient。
- **estimator kernels**：只接收张量与标量充分统计量，不依赖模型或 optimizer。
- **importance accumulator**：累计 signed、positive、negative mass、absolute 和基线。
- **distributed reducer**：只负责充分统计量与元数据的跨 rank 归约。
- **optimizer bridge**：设置 global mean gradient、执行 clip、step 和位移分解。
- **state serializer**：保存/恢复完整训练与统计状态。
- **reporter**：生成数值表、Gate 摘要和可视化，不参与计算真值。

## 4. 执行步骤

### 4.1 建立可安装骨架

1. 建立单一 Python package 根。
2. 建立独立的 unit、integration、CUDA 和 distributed 测试分组。
3. 建立 Stage 1 配置目录与 schema 版本常量。
4. 声明服务器锁定依赖与本机可运行的最小依赖边界。
5. 让纯估计器内核在没有 Transformers 和 CUDA 的本机环境中可导入。
6. 让 Pythia provider 只在集成入口被按需加载。
7. 建立统一的结构化错误类型，区分契约、输入、数值、资产和分布式错误。

### 4.2 实现配置闭环

1. 为每个公开字段定义类型、默认值和合法范围。
2. 为字段间互斥或依赖关系建立 fail-fast 校验。
3. 删除当前版本不支持的伪开关，而不是保留无效配置。
4. 为每个保留开关建立“开/关产生不同可观察行为”的测试。
5. 把解析后配置规范化为稳定顺序。
6. 计算规范化配置摘要哈希。
7. 在每个运行目录保存原始配置与解析后配置。
8. 配置摘要变化时生成不同 run identity，禁止覆盖旧运行。
9. 对未知字段、拼写错误和已废弃字段 fail-fast，不静默忽略或回退默认值。

### 4.3 建立参数 registry

1. 遍历所有 `requires_grad` 参数并记录 canonical name。
2. 记录形状、numel、原始 dtype、device 和参数组索引。
3. 记录参数组实际学习率与 weight decay。
4. 标记 layer、module type 和是否属于 embedding、attention、MLP、norm 或 head。
5. 检测共享参数和别名。
6. 对共享标量坐标只登记一次，并保留全部别名。
7. 把梯度为 `None` 的参数分成“契约预期无梯度”和“异常缺失梯度”。
8. 对异常缺失梯度 fail-closed，不静默填零。
9. 明确冻结参数和 buffer 的处理规则。
10. 对当前不支持的 sparse gradient 直接拒绝。
11. 固定每个张量内部的扁平化顺序。
12. 固定跨张量的 canonical 排序。
13. 生成 registry manifest 和摘要哈希。
14. 把 registry hash 绑定到 checkpoint、结果和报告。

### 4.4 建立同形状状态容器

1. 为每个 registry 条目建立 FP32 临时 `S1` 与 `S2` 槽位。
2. 为每个条目建立长期 signed、positive、negative mass 和 absolute 槽位。
3. 建立 raw、data movement、net movement 和 magnitude 槽位。
4. 为可选 actual-update 诊断建立独立命名空间。
5. 规定临时状态在单步完成后释放或复用。
6. 规定长期状态只在完整 step 后更新。
7. 规定所有保存数组的 dtype、shape 和 endianness。
8. 规定读取时的 schema/version 与 registry hash 校验。

### 4.5 建立 manifest 读取边界

1. 将文件读取与 JSON 解析分开。
2. 显式支持 UTF-8 与 UTF-8 BOM 两种已知编码。
3. 拒绝无法识别的编码。
4. 校验必需字段、类型和 revision 格式。
5. 校验文件集合不存在未声明的替代权重。
6. 校验大小和哈希后才允许 provider 使用资产。
7. 让错误信息指出具体字段或文件，不只返回“加载失败”。

## 5. 单元测试与核验步骤

1. 用参数注册顺序故意变化的等价模型验证 canonical 顺序稳定。
2. 用共享权重模型验证 alias 不造成重复计数。
3. 用冻结参数验证它不进入 eligible set。
4. 分别用预期 `grad=None` 与异常 `grad=None` fixture 验证规则与错误信息。
5. 保存并重载 registry，比较名称、形状、numel、参数组、标签和 hash。
6. 故意改变一个参数形状，验证旧状态被拒绝。
7. 故意改变参数组学习率映射，验证契约不一致被拒绝。
8. 为每个状态数组验证 shape 与 registry 完全对应。
9. 用带 BOM、无 BOM、缺字段和错误哈希 manifest 覆盖全部解析分支。
10. 对配置中的每个公开字段执行行为覆盖检查。
11. 注入未知字段、常见拼写错误和已废弃字段，验证解析器逐一拒绝并指出字段路径。

## 6. 产出

- 新 Python package、打包声明和测试分组骨架。
- 版本化配置 schema 与规范化配置摘要工具。
- 参数 registry manifest schema 和示例。
- 状态容器 schema 与序列化约定。
- 资产 manifest 读取与校验模块。
- 配置字段到行为测试的覆盖表。
- `G1-REGISTRY` 机器可读报告。

## 7. 可视化与呈现

- module/layer × eligible/frozen/excluded 参数量汇总图。
- 配置字段 × 行为测试覆盖矩阵，突出没有测试或没有可观察行为的字段。
- registry 保存前、重载后和不同入口间的 hash/shape 对照表。
- schema/version 兼容与 fail-closed 原因矩阵。

## 8. 核验标准

- 同一模型重复构建、保存/加载和不同运行入口产生完全相同的 registry 内容与 hash。
- registry 的 `numel` 总和等于 eligible 参数实际标量总数。
- 共享参数不重复累计，别名仍可追溯。
- 每个长期数组与对应参数张量 shape 完全一致，不允许隐式广播。
- 不支持的 sparse、未知梯度、schema mismatch 和资产错误必须 fail-closed。
- 所有公开配置字段都有行为测试；没有仅声明不生效的字段。
- 未知、拼写错误或已废弃配置字段全部 fail-closed。
- 本机 CPU 核心模块导入不要求 Transformers、Datasets 或 CUDA。

## 9. Gate 与后续依赖

- `G1-REGISTRY` 通过后，S1.3 才能冻结 oracle 的坐标顺序。
- registry 或 schema 变化会使既有 checkpoint、gradient dump 和 Gate 证据失效，必须生成新版本而不是原地覆盖。
- Stage 2、Stage 3 和后续剪枝都依赖同一 registry；任何参数 eligibility 变化必须回到本 Gate 重验。
