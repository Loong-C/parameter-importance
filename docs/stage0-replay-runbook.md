# Stage 0 离线重放运行说明

本说明是 S0.11/G9 的可执行交接合同。独立执行者只需要当前 Git 提交、`DATA_ROOT` 中的正式 manifest/index 和本文件；不得依赖聊天记录、未记录的物理 GPU 编号或旧 run 目录。

相关合同见 [Stage 0 总计划](../plan/stage0/README.md)、[S0.11 计划](../plan/stage0/11_test_quality_and_replay.md) 与 [项目入口](../Readme.md)。

## 1. 本机与服务器职责

- 本机 CPU：运行配置、schema、路径、manifest、run ID、seed、纯函数和确定性 fixture 测试。本机没有 CUDA 时，GPU 测试只能显示为未执行，不能记为通过。
- 服务器 CPU：验证 Linux 路径、离线资产元数据、JSONL、checkpoint 结构和完整 pytest 套件。
- 单 GPU：重放 Pythia 14M BF16 baseline 与 fresh-process resume。
- 四 GPU：重放 NCCL、DDP、分片、累积、`no_sync`、组 checkpoint 和 fresh-process resume。
- 故障层：复验损坏 checkpoint、缺 rank、非法输入、真值日志失败、容量预检和上一完整 checkpoint 保留。

正式服务器源码只能位于 `/home/sophgo13/cjl/parameter-importance`，大资产与证据只能位于 `/home/sophgo13/cjl/storage/parameter-importance`。设置：

```bash
export PARAM_IMPORTANCE_DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

推荐环境、Python、锁文件、驱动、内核、GPU 拓扑与资产 identity 必须从 G4 provenance、G3 resolution 和当前 Stage 0 index 读取，不凭名称猜测。正式 replay 使用的配置和环境 hash 会写入 plan 与报告。

## 2. 本机 CPU 测试

Windows 默认 pytest 临时目录若有 ACL 问题，应选择一个源码树之外、当前用户可写的专用空目录；不要把 `DATA_ROOT` 指向源码树。示例：

```powershell
$env:PYTHONPATH='.'
pytest -q --basetemp C:\path\to\dedicated-empty-pytest-temp
```

任何失败都必须保留退出码和 JUnit/终端证据。skip 只能按测试矩阵明确分类；硬层 skip 使 G9 失败。

## 3. 选择 GPU

GPU UUID 来自 `g4_provenance` 的四卡 `device_mapping`，不得硬编码物理编号。正式启动前必须确认：

1. UUID/PCI/rank 映射与 G4 一致；
2. ECC、温度、设备可见性和拓扑仍健康；
3. 目标 GPU 没有外部进程；
4. 项目租约可取得且资源所有者预约仍有效。

发现未知占用时退出，不终止他人进程。租约仅防止本项目重复运行，不冒充集群调度器。

## 4. 正式 G9 重放

从 G8 formalization index 取得 `g8-index-ref`，在干净提交上执行：

```bash
python ops/stage0/formalize_g9.py \
  --data-root "$PARAM_IMPORTANCE_DATA_ROOT" \
  --g8-index-ref evidence/stage0/g8-formal/<g7-gate-hash>/index.json \
  --repository /home/sophgo13/cjl/parameter-importance \
  --git-commit "$(git rev-parse HEAD)" \
  --git-branch "$(git branch --show-current)"
```

orchestrator 只把一个哈希绑定 replay plan 交给全新子进程。子进程必须从不存在的唯一输出目录开始，先运行完整 pytest，再验证源码生成的固定 token/labels fixture，最后运行单卡和四卡 baseline/interruption/resume。恢复必须创建新进程并从组 checkpoint commit 恢复，不能复用内存模型。

离线重放设置 Hugging Face offline 标志，并在 Python socket 层阻断非本机连接；loopback 仅供 `torchrun` rendezvous。每个进程生成哈希绑定审计文件。任何外连尝试、缺审计文件或不完整审计都会使 G9 失败。该审计不冒充系统防火墙，报告会明确其范围。

## 5. 输出与 evidence 定位

只凭 replay ID 或 run ID 可定位全部产物：

- plan/transcript：`evidence/stage0/g9-plans/<config-hash>/`；
- 独立 replay：`evidence/stage0/g9-replays/<replay-id>/`；
- pytest JUnit/JSON：`<replay-root>/pytest/`；
- Python 网络审计：`<replay-root>/network-audit/`；
- 单卡/四卡事件、checkpoint、lineage：`<replay-root>/recovery-replay/`；
- 机器可读与 Markdown 摘要：`independent-replay-report.json`、`independent-replay-report.md`；
- 正式任务 commit：`evidence/stage0/tasks/11-*/`；
- G9 交接 index：`evidence/stage0/g9-formal/*/index.json`。

报告中的 `run_lineage` 将每个 run ID 绑定到 Git commit、config hash、environment hash、GPU UUID、模型/tokenizer/data 资产 ID、组 checkpoint ID 和报告路径。JSON/JUnit/事件流是真值；Markdown/TensorBoard 只是可重建派生层。

## 6. 恢复与故障诊断

1. 先读 launch transcript、run status、heartbeat 与项目 launch claim，SSH 断线时不得重复启动同一 replay ID。
2. 只选择通过完整 manifest/commit 校验的 checkpoint；目录存在不等于提交完成。
3. 损坏对象、缺 rank、config/environment/world-size 不匹配必须失败关闭，并保留证据。
4. JSONL 或状态真值写入失败必须停止；TensorBoard 派生失败可从 JSONL 重建。
5. OOM、NCCL、ECC、磁盘不足或设备消失按 G8 fault report 处理，不静默缩 batch 或换卡。
6. 若 `DATA_ROOT` 故障，只能使用已授权第二故障域；同盘副本不是备份。

## 7. 环境漂移与回归触发

变化先按 [`policies/evidence-validity-and-rerun.md`](../policies/evidence-validity-and-rerun.md)
计算最小影响闭包，不把“环境变化”泛化为整层或整链失效：

- 依赖锁/Python 变化：重验实际使用该环境的层；驱动、内核、GPU 健康或拓扑变化只刷新设备层和
  实际依赖它的 GPU/容量层，不重跑本机 CPU、资产获取或无关数值结果；
- 配置 schema、training step、loss reduction、日志或 checkpoint 变化：分别重验消费相应语义的
  数值、观测或恢复单元，未受影响的原子单元沿用；
- 模型、tokenizer 或数据 manifest 变化：仅当具体内容身份或预处理合同变化时，重验消费该资产的单元；
- 只有文档、worklog、同步或下游代码变化：运行链接、结构和 consumer 回归，不占用 GPU。

正式触发矩阵位于 `configs/stage0/g9-test-matrix-v1.json`，报告必须分别说明保留、刷新、失效的 Gate
及理由；提交不同本身不属于失效理由。

## 8. 明确禁止

禁止以下操作：

- 绕过 manifest、READY/qualification 或哈希校验直接加载资产；
- 把 `.part`、下载中目录或来源不明二进制当作正式输入；
- 复用非空 replay 输出目录、覆盖旧证据或手工修改机器可读数值；
- 硬编码物理 GPU 编号、抢占未知进程或终止其他用户任务；
- 修改 SSH 拓扑、`~/.ssh/config`、跳板关系或项目规定的服务器路径；
- 使用 `sudo`、系统级安装或擅自提高全局文件描述符限制；
- 用小模型外推冒充 160M/410M G8 实测，或用 Stage 0 fixture 证明重要性算法正确。

## 9. Stage 1 交接边界

Stage 1 可直接复用环境 ID、Pythia 14M/固定调试数据资产、配置模板、seed 域、loss/gradient 语义、确定性 fixture、JSON/JUnit 报告格式和单卡/四卡恢复入口。Stage 0 只证明基础设施与 replay 可信；Stage 1 仍须独立证明参数重要性公式、估计器和训练接线的数值正确性，不能引用 G9 代替。
