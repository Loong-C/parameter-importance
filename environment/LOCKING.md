# 依赖声明与锁定规则

## 文件职责

| 文件 | 解析目标 | 用途 |
|---|---|---|
| `base-requirements.in` | 平台无关 | 配置、schema 与 CPU 公共逻辑的最小直接依赖 |
| `requirements.in` | Linux CPython 3.12 x86-64 | 服务器 Python/CUDA 应用依赖 |
| `linux-only-requirements.in` | Linux CPython 3.12 x86-64 | CUDA、cuDNN、NCCL、Triton 等服务器专属补充 |
| `local-dev-requirements.in` | Windows CPython 3.12 | 本机 CPU 小张量测试；不安装 CUDA/NCCL |
| `requirements.lock` | Linux CPython 3.12 x86-64 | 服务器离线重建的 89 个精确分发包 pin |

输入声明和锁文件职责不同。修改任一 `.in`、Python minor、目标平台/ABI、
索引来源或解析器版本后，都必须完整重解依赖图；禁止只手改一个传递依赖。

## 严格锁格式

`requirements.lock` 只允许：

- 空行和整行注释；
- `--index-url`、`--extra-index-url` 后跟不含凭据的 HTTPS URL；
- 每行一个 `distribution==exact-version`。

editable 安装、VCS 引用、direct URL、本地 wheel/path、环境 marker、通配版本、
重复分发包和其他 pip 选项全部拒绝。锁文件的目标平台、Python、解析命令、
解析器版本、索引和生成时间必须写入同轮环境审计报告，不能从 Windows marker
语义推断 Linux/CUDA 锁。

当前锁头部同时保存了真实历史来源：2026-07-13 使用 pip 25.0.1 为 Linux CPython
3.12/cp312 生成基础解析报告，再依据 PyTorch/CUDA wheel 的 Linux metadata 补齐
运行时强制依赖；基础报告 SHA-256 和对应生成脚本提交均已记录。它不是由当前服务器
bootstrap pip 24.0 重新解析得到，二者必须作为“解析器”和“安装器”分开记录。

机器可读的 `reports/stage0/lock-provenance-20260719.json` 还把上述三份服务器依赖
输入的逐文件 SHA-256、lock SHA-256、分发包数量和核心 runtime 预期绑定在同一份
证据中。重建入口只接受这三份固定输入，不提供用任意文件替换它们的参数；每个直接
依赖必须存在于 lock，直接精确 pin 也必须与 lock 一致。任一输入改变而 provenance
未随完整重锁更新时，重建在创建候选 venv 前失败。

建议的锁生成过程是：在固定解析器版本下，以 Linux CPython 3.12、`cp312`、
`manylinux_2_28_x86_64`、`manylinux_2_17_x86_64` 和
`manylinux2014_x86_64` 为目标完整解析；保存解析报告；再由
`param_importance_nlp.environment.parse_requirements_lock` 做严格复核。

## wheelhouse 清单

权威清单沿用现有 ASCII TSV：

```text
<64位小写SHA-256>\t<字节数>\t<wheel文件名>
```

清单放在 `DATA_ROOT/manifests`，wheel 放在 `DATA_ROOT/wheelhouse`。审计必须比较
文件集合、大小和 SHA-256，并核对每个 lock pin 至少有一个同版本 wheel；任何
额外文件、符号链接、缺 wheel 或错误哈希都失败。

平台验收目标是 CPython 3.12 / `cp312` / Linux x86-64、基线兼容下限
manylinux glibc 2.28。允许匹配的 `cp312`、向后兼容 `abi3`、`py3`/`py312` wheel，
以及 `manylinux_2_5` 至 `manylinux_2_28`（含 manylinux1/2010/2014 别名）；Windows、
macOS、其他架构、比 2.28 更新的 manylinux 标签和无法解析的 wheel 一律拒绝，不能
仅因文件名中的分发包与版本前缀相同而计为覆盖。

## 本机环境

在仓库根目录使用 Git 已忽略的 `.venv/`：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r environment\local-dev-requirements.in
.venv\Scripts\python.exe -m pytest -q
```

本机 CPU PyTorch 只用于小型逻辑测试；GPU、NCCL、BF16 和性能测试必须保留
服务器专属 marker，skip 不得记作 gate 通过。本机不得下载模型或数据资产。

## 离线重建边界

服务器候选环境必须位于 `DATA_ROOT/envs` 的唯一空目录，使用隔离缓存和环境白名单、
`--no-index --require-hashes --only-binary --find-links DATA_ROOT/wheelhouse`，且保留
所有 Python/pip 子进程的连接审计和“移除一个必需 wheel 后因该精确 pin 明确失败”
的负向证据。无特权 network namespace 不可用时必须如实标记为连接审计模式，不能
宣称内核级断网。不得原地修改现有 venv，不得使用系统根盘缓存，也不得通过未登记的
本地缓存或在线索引回退。

核心导入还必须把 Python implementation/series、Torch/Transformers/Datasets/
Accelerate/TensorBoard 模块版本、PyTorch CUDA runtime、cuDNN runtime，以及
cuDNN/NCCL 分发包版本逐项与 lock provenance 比对。G0-G 放行前不调用 NCCL GPU
runtime；该字段必须明确标为 `DEFERRED_UNTIL_G0_G`，不能用分发包版本冒充运行时值。
