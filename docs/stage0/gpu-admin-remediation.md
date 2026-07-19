# Stage 0 GPU 管理员诊断与修复说明

## 项目范围决定（2026-07-19）

项目所有者已确认正式计算只需要四张 A100，并选择下述路径 B。项目侧同意排除故障
设备，但这不是把故障记为“已忽略”或“已修复”：`0000:4f:00.0` 与
`0000:50:00.0` 在管理员逐卡清除前均不得进入候选四卡。

当前状态为 `PROJECT_OWNER_APPROVED_ADMIN_PENDING`。管理员仍须提供并实施精确的
四卡 PCI/UUID 白名单、两张异常设备的隔离机制与处置结论，以及审批有效期/失效
条件；NVML 与 PyTorch 对该白名单的映射一致后，项目才会进行逐卡健康复验。

## 1. 请求目的与当前结论

本说明只用于解除 `G0-G` GPU 健康子 gate 的管理员协调。项目 Agent、普通操作者和训练脚本不得据此自行修改驱动、设备或系统状态。

截至 2026-07-19，本轮只读基线显示：

- PCI 与 NVIDIA 驱动目录均识别到 8 张 `NVIDIA A100-SXM4-80GB`；
- NVML 与项目 PyTorch 运行时均只枚举 7 张，缺失 PCI 设备为 `0000:4f:00.0`；
- 数字 index 已发生前移，当前可见 index 0 对应 `0000:50:00.0`，不能沿用旧的 index/UUID 映射；
- `0000:50:00.0` 有 24 次 volatile 和 24 次 aggregate 不可纠正 ECC，row-remap pending 为 `Yes`，驱动报告的恢复动作是 `Reset`；
- 自 2026-07-18 起的可读内核日志中，`0000:50:00.0` 多次出现 Xid 95（FBHUB）；本轮只读健康查询期间仍出现新记录；
- 当前无计算进程、显存占用为零，但“空闲”不等于“健康”；
- 旧的 8 卡 `READY`、旧拓扑和旧 DDP 结果已经失效。

机器可读请求见 `reports/stage0/g0-g-admin-request-20260719.json`，原始 gate 摘要见 `reports/stage0/g0-baseline-20260719.json`。在管理员完成路径 A 或路径 B 并通过复验前，`G0-G` 保持 `BLOCKED`。

### 1.1 PCI/sysfs 补充只读事实

在不调用 `nvidia-smi`、NVML、CUDA 或 PyTorch 的前提下，对两个指定设备补充读取了 PCI/sysfs 与既有内核记录：

- `0000:4f:00.0` 与 `0000:50:00.0` 都仍存在于 sysfs，`enable=1`，绑定 `/sys/bus/pci/drivers/nvidia`，模块为 `nvidia`；
- 两者当前和最大链路均为 PCIe `16.0 GT/s ×16`，电源 runtime 状态为 `active`，未发生 runtime suspend；
- 两者的 `reset_method` 都列出 `flr bus`。这只表示内核暴露的恢复能力，不是执行 reset 的授权，也不证明 reset 能解决根因；本轮没有读取 reset 触发器或写入任何 sysfs 字段；
- 两个端点可读的 AER correctable/nonfatal/fatal 总计均为 0；上游 root-port AER 汇总字段未在端点 sysfs 暴露。端点 AER 为零不能否定 GPU 内部 ECC、Xid 或 NVIDIA RM 初始化失败；
- 现有内核日志反复显示 `0000:4f:00.0` 的 `RmInitAdapter failed! (0x62:0x40:1941)` 和 `rm_init_adapter failed, device minor number 0`；
- `0000:50:00.0` 的既有记录仍为 Xid 95、`Uncontained: FBHUB`、`RST: Yes`；本次 PCI/sysfs 读取没有打开 GPU 管理或计算运行时。

因此当前证据更符合“PCI 链路和驱动绑定存在，但 NVIDIA RM 初始化/设备内部健康失败”，而不是简单的物理 PCI 链路消失。根因仍必须由管理员结合上游 root port、BMC、硬件服务记录和厂商工具确认。

## 2. 立即影响控制

1. 将本节点的 GPU 训练、CUDA、NCCL、DDP、显存和性能测试保持暂停；不得因设备当前空闲而放行。
2. 将 `0000:4f:00.0` 与 `0000:50:00.0` 视为未获准设备。管理员应在调度器或现场分配流程中阻止它们被新任务取得；项目代码不得自行修改设备可见性来伪造健康枚举。
3. 若节点承载其他使用者，管理员应先通知资源所有者、排空受影响任务并建立维护窗口。不得终止未知进程来为诊断腾卡。
4. 保存变更前的 PCI/驱动映射、ECC、row-remap、Xid、AER/BMC 平台事件和设备恢复建议。不得清除计数后把“零计数”当作修复证据。
5. 对任何可能改变驱动、内核、设备 UUID、PCI 映射或拓扑的维护，记录变更单、执行者、开始/结束时间、回退方案和最终处置结论。
6. 若健康查询继续触发 Xid、设备消失或驱动恢复建议升级，停止重复探测并按厂商/机房硬件流程升级，不用轮询压测故障设备。

## 3. 管理员诊断顺序

以下步骤只能由服务器/基础设施管理员在维护窗口内执行。项目 Agent 不执行这些动作。

### 3.1 建立维护前快照

1. 确认节点已排空，记录当前计算进程、MIG、显存、温度和调度状态。
2. 记录 8 个物理 PCI 地址、驱动目录、device minor、GPU UUID 与当前 NVML index 的映射。
3. 保存内核 NVIDIA Xid、PCIe AER、驱动绑定、BMC/平台健康和最近硬件维护记录。
4. 单独记录 `0000:4f:00.0` 为什么存在于 PCI/驱动层却不进入 NVML，以及 `0000:50:00.0` 的 Xid 95、ECC 和 row-remap 状态。

可参考的只读检查如下；管理员应按现场权限和厂商 runbook 调整，且在故障设备查询会重复触发 Xid 时立即停止：

```bash
lspci -Dnn | grep -i NVIDIA
find /proc/driver/nvidia/gpus -mindepth 1 -maxdepth 1 -type d -print
lspci -vv -s 0000:4f:00.0
readlink -f /sys/bus/pci/devices/0000:4f:00.0/driver
journalctl -k --no-pager | grep -E 'NVRM:.*Xid|AER|0000:4f:00.0|0000:50:00.0'
```

`nvidia-smi`、DCGM 或其他会打开设备的查询只应在管理员判断安全时运行；修复前不要由项目侧再次尝试。

### 3.2 诊断缺失设备 `0000:4f:00.0`

管理员需区分并记录以下原因类别：

- PCIe/SXM 链路、电源、BMC 或硬件故障；
- 驱动 probe/bind 失败、设备节点或 minor 映射异常；
- 设备进入错误/恢复状态而被 NVML 排除；
- 有意隔离，但未留下可核验的管理员白名单和处置记录；
- 其他厂商明确说明的原因。

当前已确认 PCIe Gen4 x16 链路、`enable=1` 和 `nvidia` 绑定存在，但 NVIDIA RM 初始化以 `0x62:0x40:1941` 失败。管理员应重点核对该错误签名对应的厂商处置、设备/基板/BMC 健康、上游 root-port AER 和驱动初始化链路，而不是把重新 bind 当作项目侧修复。

诊断结论必须包含根因、影响范围、是否需要硬件服务、是否会改变 UUID/PCI 拓扑，以及设备最终是恢复还是隔离。不得仅以“重启后出现”“PCI 链路为 x16”或“NVML 暂时可见”作为结案依据。

### 3.3 处置 `0000:50:00.0`

该设备同时具有不可纠正 ECC、pending row remap、Xid 95 和驱动 `Reset` 建议，默认不得进入任何候选 GPU 集合。管理员应：

1. 保存变更前计数和日志；
2. 注意该设备 PCIe 链路和端点 AER 当前正常并不能解释 GPU 内部 ECC/FBHUB Xid；依据 NVIDIA/整机厂商维护手册判断应采用受控设备恢复、节点维护、硬件检修或更换；
3. 只在节点排空、维护授权和回退条件齐备时执行管理员级恢复动作；
4. 对历史 aggregate 不可纠正 ECC 给出明确的“继续服役、永久隔离或更换”结论；
5. 完成后重新检查 row-remap pending/failure、恢复建议和观察窗口内的新 ECC/Xid。

本文不提供 reset、驱动重载或重启命令；这些动作必须走管理员既有维护流程。

## 4. 修复路径 A：恢复完整 8 卡

路径 A 的目标是恢复 PCI/驱动、NVML、PyTorch 的 `8/8/8` 一致枚举。

管理员完成设备修复、维护或更换后，必须同时满足：

1. 8 张预期设备在 PCI、驱动目录、NVML 和 PyTorch 中形成一一对应的 PCI/UUID 映射；没有缺失、重复或静默 index 前移。
2. 每张预期设备均能读取完整健康字段；查询本身不再产生 Xid 或设备消失。
3. volatile 不可纠正 ECC 为 0；row-remap 无 pending/failure；恢复建议为 `None` 或等价健康状态。
4. aggregate 不可纠正 ECC 非零的设备具有管理员书面处置结论，且只有被明确批准继续服役的设备才可进入白名单。
5. 记录起止时间的观察窗口内无新增 ECC、Xid、AER 或设备恢复事件。建议至少 15 分钟并采集首尾两份快照；厂商规范要求更长时采用更长窗口。
6. 明确选择四张健康 GPU，按 PCI/UUID 固定候选集合；不得只写数字 index。

满足以上条件并完成第 6 节复验后，才可将路径 A 标记为 `PASS`。

## 5. 修复路径 B：管理员批准稳定隔离/降配

若不能恢复完整 8 卡，管理员可以批准稳定降配，但“仍然是 8/7”本身不构成路径 B。管理员必须提供：

1. 预期物理设备清单、可用设备 PCI/UUID 白名单及其期望数量；
2. 每个被排除设备的 PCI/UUID（可取得时）、故障原因、隔离机制和最终处置；
3. 调度器或现场分配层面的强制隔离证据，不能只依赖项目进程临时设置数字 index；
4. 负责人、批准时间、适用范围、有效期、复审条件和失效条件；
5. 任何驱动/内核/硬件/拓扑变化后的自动失效与重新验收要求。

在 unresolved 状态下，`0000:4f:00.0` 和 `0000:50:00.0` 都必须排除。若管理员认为其中某设备已修复并可继续服役，必须给出独立处置结论并使其通过与路径 A 相同的健康标准。

路径 B 通过需满足：

- NVML 与 PyTorch 只枚举且完整映射到批准白名单，数量一致；
- 被排除设备不会被项目或其他常规调度重新取得；
- 白名单中候选四卡全部满足 ECC、row-remap、Xid、温度、占用和恢复状态标准；
- 报告显著标记这是管理员批准的降配基线，不得发布普通 8 卡 `READY`。

## 6. 修复后复验清单

管理员声明设备可查询后，由管理员或获明确许可的 Stage 0 操作者重新采集同一组只读字段：

### 6.1 身份与版本

- 复验时间、管理员变更单/结论引用；
- OS、内核、NVIDIA 驱动、系统 CUDA toolkit；
- 项目 Python、PyTorch 和 PyTorch CUDA runtime；
- 若内核、驱动、拓扑或 UUID 已变化，旧 G2、G5、G6、G8 GPU 证据全部标记为 `STALE`。

### 6.2 三层枚举与映射

- PCI/驱动目录数量和 PCI 地址；
- NVML index、PCI、UUID、型号和显存；
- PyTorch device count 及 PCI/UUID 映射；
- 路径 A 为严格 `8/8/8`；路径 B 为 NVML/PyTorch 与管理员白名单严格一致。

### 6.3 每卡健康

- corrected/uncorrected ECC 的 volatile 与 aggregate；
- retired pages（驱动支持时）；
- row-remap correctable/uncorrectable、pending 和 failure；
- recovery action、MIG、温度、显存、利用率和计算进程；
- 观察窗口内的 Xid/AER/设备消失与 ECC 增量。

### 6.4 候选四卡

- 以 PCI/UUID 列出四卡，不使用持久化数字 index；
- volatile 不可纠正 ECC 为 0，row-remap 无 pending/failure；
- 无活动计算进程，查询过程中无新 Xid/ECC；
- 拓扑和 NUMA 映射已记录；
- 只有 `G0-G` 复验通过后，才允许进入后续最小 CUDA、单卡和 NCCL gate。

管理员回复至少应填写机器可读请求中的 `administrator_response_required` 字段，并附可核验的维护/处置引用。项目侧不得自行替管理员签署健康结论。

## 7. 明确禁止项

在管理员授权和维护流程之外，项目 Agent、普通操作者与项目脚本不得：

- 执行 GPU reset、PCI unbind/rebind、驱动重载、节点重启或 ECC 清除；
- 使用 `sudo`、修改系统 CUDA/驱动、设备权限、MIG、持久化模式、功率或时钟；
- 终止未知或其他使用者的进程；
- 反复查询故障设备、运行 CUDA 张量、DCGM 压测、NCCL、DDP、显存测试或训练；
- 用 `CUDA_VISIBLE_DEVICES` 或旧数字 index 掩盖 8/7 枚举差异；
- 把“无计算进程”“计数未增长一次”或清零后的计数当作健康证明；
- 复用旧 `READY`、旧四卡拓扑或旧通信结果；
- 在没有书面白名单、排除结论、有效期和隔离证据时宣称路径 B 通过。

任何超出上述边界的系统动作都必须由服务器管理员独立评估、授权和执行。
