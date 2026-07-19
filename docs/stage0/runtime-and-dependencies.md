# Stage 0 运行时、依赖与环境身份

## 当前结论

仓库现已具备 S0.3 的本地审计基础：严格 lock/freeze 解析、现场漂移比较、
wheelhouse TSV 清单核验和稳定 `environment_id`。这不等于 G2 已通过；服务器
候选环境的离线重建、零外连审计、缺 wheel 负向演练以及硬件恢复后的逐卡 CUDA
检查仍需独立证据。

## 运行时决策

- Python 范围为 `>=3.12,<3.13`；服务器当前基线为 3.12.3，本机为 3.12.4。
- 分布式数值真值层使用原生 PyTorch DDP/`torchrun`。Accelerate 保留为锁定工具，
  但不能隐藏 Stage 0 的梯度、归约或 `no_sync` 测试。
- Stage 0 不引入 DeepSpeed，不依赖 W&B；追踪使用结构化本地日志和 TensorBoard。
- 当前不编译自定义 CUDA 扩展。以后若进入范围，必须另建 compiler/ABI/toolkit gate。
- 环境、wheelhouse 和缓存只能位于 `DATA_ROOT`；不得原地升级既有 venv，不使用
  `sudo` 或系统 Python。

## 依赖分层

`environment/base-requirements.in` 是平台无关最小层；现有
`requirements.in` 与 `linux-only-requirements.in` 共同定义 Linux/CUDA 服务器层；
`local-dev-requirements.in` 定义 Windows CPU 开发层。精确格式、重锁条件、
wheelhouse TSV 和本机构建命令见 `environment/LOCKING.md`。

`requirements.lock` 当前包含 89 个精确分发包。审计器只接受 HTTPS index 声明
和 `name==exact-version`，明确拒绝 editable、VCS、direct URL、本地路径、marker、
通配版本、重复 pin 和未知 pip 选项。现场 `pip freeze` 使用同一规则，因此任何
未记录源码或本地 wheel 安装都会 fail closed。

重建入口固定读取 base、服务器和 Linux/CUDA 三份输入，并要求它们的 SHA-256 与
lock provenance 完全一致；调用者不能用 `--dependency-input` 替换必需输入。输入中
每个直接依赖都要在 lock 中存在，已有精确 pin 必须与 lock 相同。

## wheelhouse 审计

`parse_wheelhouse_manifest` 读取既有
`sha256<TAB>size<TAB>filename` 格式；`compare_wheelhouse_to_manifest` 核验：

1. 清单与目录文件集合完全相同；
2. 不含符号链接或非文件对象；
3. 每个文件大小和 SHA-256 一致；
4. 每个 lock pin 至少有一个同版本 wheel；
5. 清单中没有无法绑定到 lock 的额外 wheel。

覆盖检查同时验证 CPython 3.12/`cp312`、Linux x86-64 与 manylinux glibc 2.28
兼容标签；Windows/macOS、其他架构或不兼容 ABI 的 wheel 即使名称和版本匹配也失败。

这些检查只证明离线材料集合完整。服务器的无特权 network namespace 自检因
`uid_map: Operation not permitted` 不可用，因此重建器不会声称内核级断网；它使用
环境变量白名单、唯一 HOME/cache/tmp、`--no-index --require-hashes
--only-binary`，并以 `strace -f -e trace=network` 审计创建环境、安装、普通与
`--all` freeze、`pip check`、核心导入和负向安装。任何外连 `connect/send*` 记录都
使本次构建失败。凭据、代理和宿主 `PIP_*` 不会传入子进程。

安装前后都重新核对 wheelhouse 集合、大小和 SHA-256；pip 实际读取的每个 wheel
还由派生的 hash-checking requirements 绑定。候选环境的普通 freeze 必须与 89 个
应用锁完全一致，`pip freeze --all` 只允许额外出现已记录、且新旧环境版本相同的
bootstrap `pip`。

核心导入输出不只留档，还逐项与 lock provenance 比对 Python series、核心模块、
Torch CUDA runtime、cuDNN runtime 及 cuDNN/NCCL 分发包版本。实际 NCCL runtime
保持空值并显式等待 G0-G 后的四卡通信复验。

## 版本字段边界

环境 manifest 不使用含混的单个 `cuda` 字段，而是分别记录：

| 字段 | 含义 |
|---|---|
| `nvidia_driver_version` | 主机驱动版本/驱动能力 |
| `system_cuda_toolkit_version` | 系统 toolkit/`nvcc` 版本 |
| `torch_version` | PyTorch 分发包版本 |
| `torch_cuda_runtime_version` | PyTorch wheel 编译/绑定的 CUDA runtime |
| `cudnn_version` | PyTorch 实际使用的 cuDNN |
| `nccl_version` | PyTorch 实际使用的 NCCL |

`collect_runtime_versions` 不导入 PyTorch、也不初始化 CUDA；服务器采集器应把已
核验的 PyTorch/cuDNN/NCCL 字段显式传入。GPU 数量、ECC、温度和占用属于可漂移的
硬件健康报告，只在 manifest 中记录引用，不混入不可变依赖锁。

## 环境身份

`environment_id` 是 `env-v1-<SHA-256>`，由以下规范化稳定字段计算：

- 各依赖输入文件 SHA-256；
- lock 的排序后 pin 和 index 语义摘要；
- freeze 的排序后精确版本摘要；
- wheelhouse 清单的排序后文件/大小/哈希摘要（服务器环境）；
- OS、内核、架构、libc、Python 和分层运行时版本。

稳定 identity 单独写入 `DATA_ROOT/manifests/environment-identities/`；候选路径、
时间、Git 提交、全部日志/trace 的大小与 SHA-256 则写入独立、不可覆盖的
`environment-builds/<build_id>.json`。因此同一内容在新的版本化候选路径重建可得到
同一 `environment_id`，而每次构建仍有独立证据链。

当前 G0-G 仍为 `BLOCKED`，所以 CPU 导入和离线重建即使通过也只原子发布
`environment-cpu-candidate.json`，其中 `training_eligible=false`、`g2_status=BLOCKED`。
它不会更新普通训练推荐引用。只有管理员批准稳定四卡 PCI/UUID 白名单，并完成逐卡
CUDA 健康复验与实际 NCCL runtime 采集后，才能形成完整 G2。

重建入口只接受 24 小时内、schema/G0-C/总状态自洽且 hostname 匹配当前服务器的
GPU 基线；它还会在不调用 NVML/CUDA runtime 的情况下重新读取
`/proc/driver/nvidia/version` 和对应 `/usr/local/cuda-<version>/bin/nvcc --version`。
任一当前安全事实与基线不一致都会在创建推荐引用前失败。

服务器入口为 `ops/stage0/rebuild_environment.py`。它只接受仓库内 lock/GPU 基线、
大盘内既有 venv/wheelhouse/清单和一个安全的唯一候选名；项目级 advisory lock 串行化
identity、build observation 和推荐引用发布。锁固定为既有批准目录中的
`DATA_ROOT/operations/stage0-environment-rebuild.lock`，不会增加布局合同外的根目录。
负向 venv 使用带 owner marker 的专属
临时树，先封存精确清理清单，再逐文件 `unlink`、逐目录 `rmdir`，不使用通配清理。

典型 CPU 审计流程为：

```python
from param_importance_nlp.environment import (
    build_environment_manifest,
    compare_freeze_to_lock,
    parse_freeze_file,
    parse_requirements_lock,
)

lock = parse_requirements_lock("environment/requirements.lock")
freeze = parse_freeze_file("pip-freeze.txt")
comparison = compare_freeze_to_lock(lock, freeze)
if not comparison.ok:
    raise RuntimeError(comparison.as_dict())
```

每次正式运行必须同时记录 `environment_id` 和当次 freeze/运行时漂移检查；只记录
历史 `environment_id` 不能证明当前进程仍处于同一环境。
