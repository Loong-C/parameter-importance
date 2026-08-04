# Stage 0 资产与 manifest 合同

本合同覆盖模型、tokenizer、数据集和外部源码在进入实验前的身份、状态、验真与发布规则。实现位于 `src/param_importance_nlp/assets.py`，机器可读约束位于 `schemas/stage0-asset-manifest-v1.json`。

## 不可变身份

`asset_id` 是规范 JSON 身份对象的 SHA-256。身份对象包含：

- `asset_type`、逻辑 `name`、稳定 `source` 和固定 `revision`；
- 按相对路径排序后的文件路径、字节数、SHA-256 与可选角色；
- 与类型对应的 `metadata`。

状态、时间、actor、本地绝对路径和审计摘要不参与身份。文件或语义元数据发生变化都会产生新 ID。`revision` 必须是固定提交、digest、版本或快照标识；空值以及 `unknown`、`latest`、`main`、`master` 等泛化值（大小写不敏感）均被拒绝。

每种资产都必须提供最低元数据：

| 类型 | 必填元数据 |
| --- | --- |
| `model` | `architecture`、正整数 `parameter_count`、`dtype`、`initialization_id` |
| `tokenizer` | `tokenizer_class`、正整数 `vocab_size`、对象 `special_tokens`、`normalization` |
| `dataset` | 非空 `splits`；每个 split 含非负 `sample_count` 与非空唯一 `fields`；另含 `preprocessing_version` |
| `source` | `source_kind`、`license` |

允许附加类型专属字段，但它们也会进入身份哈希。

## 状态、角色与证据

合法状态路径如下：

```text
fetcher                 verifier                 gate
null -> downloading -> downloaded -> verified -> ready
                            |           |           |
                            +-----------+-----------+-> invalid
```

- `fetcher` 只能创建候选并执行 `downloading -> downloaded`。
- `verifier` 只能执行 `downloaded -> verified`，或把 `downloaded/verified` 置为 `invalid`。
- `gate` 只能执行 `verified -> ready`，或在后续审计中执行 `ready -> invalid`。
- `invalid` 是终态，不能翻回成功状态。

每个 `state_history` 事件必须同时记录 `actor_role` 和 `evidence_ref` 字段。进入 `verified`、`ready` 或 `invalid` 时，`evidence_ref` 必须是非空稳定引用，不能是带查询参数的临时或签名 URL。库会校验历史连续性、角色权限以及顶层 `state`。

已经发布为 `ready` 的路径永久不可原地覆盖。若后续审计判定失效，先生成 `ready -> invalid` 记录，再把该记录发布到一个新 manifest 路径；原 READY 证据保持不变。

## JSON、文件与路径安全

manifest 必须是严格 UTF-8 JSON，不允许 BOM、重复键、`NaN`、`Infinity` 或未知顶层字段。规范写入采用键排序、紧凑分隔符和结尾换行。

资产文件路径必须是规范化相对 POSIX 路径。绝对路径、`..`、反斜杠、Windows drive/ADS 语法、大小写碰撞，以及含 `.part`、`.partial`、`.lock`、`.tmp`、`tmp`、`.temp`、`temp` 标记的组件都会在文件访问前被拒绝。验真和 READY 解析还会拒绝符号链接、目录、缺失文件、越界解析、大小不符和 SHA-256 不符。

## 构建、验真与转换

```python
from pathlib import Path

from param_importance_nlp.assets import (
    AssetActorRole,
    AssetFile,
    AssetState,
    build_manifest,
    transition_manifest,
    verify_only,
)
from param_importance_nlp.atomic import sha256_file

asset_root = Path("/approved/data-root/models/example-step0")
weights = asset_root / "model.safetensors"
manifest = build_manifest(
    asset_type="model",
    name="example-step0",
    source="huggingface:organization/example",
    revision="0123456789abcdef",
    files=[AssetFile(
        path="model.safetensors",
        size_bytes=weights.stat().st_size,
        sha256=sha256_file(weights),
        role="weights",
    )],
    actor="asset-fetcher",
    actor_role=AssetActorRole.FETCHER,
    evidence_ref="reports/fetch-start.json",
    generator_version="stage0-assets/1",
    metadata={
        "architecture": "ExampleLM",
        "parameter_count": 1000000,
        "dtype": "float32",
        "initialization_id": "seed:1337",
    },
)
manifest = transition_manifest(
    manifest,
    AssetState.DOWNLOADED,
    actor="asset-fetcher",
    actor_role=AssetActorRole.FETCHER,
    evidence_ref="reports/fetch-complete.json",
    summary="all final objects published",
)

# 纯只读完整验真，不改变 manifest。
report = verify_only(manifest, asset_root)
assert report["ok"]

manifest = transition_manifest(
    manifest,
    AssetState.VERIFIED,
    actor="asset-verifier",
    actor_role=AssetActorRole.VERIFIER,
    evidence_ref="reports/verification.json",
    summary="size and SHA-256 verified",
)
manifest = transition_manifest(
    manifest,
    AssetState.READY,
    actor="asset-gate",
    actor_role=AssetActorRole.GATE,
    evidence_ref="reports/gate.json",
    summary="offline semantic checks passed",
)
```

`verify_only` 只接受 `downloaded`、`verified` 或 `ready`，对每个最终文件完整读取并核对大小和 SHA-256，但不写文件、不改状态、不发布 manifest。

## 受控原子发布与 CAS

发布必须显式给出已经存在的批准根目录 `manifest_root`。目标必须位于该根内，目标父目录必须预先创建，并且从根到父目录的链上不能有符号链接或 junction；发布函数不会替调用者自动创建目录。

```python
from param_importance_nlp.assets import publish_manifest_atomic

manifest_root = Path("/approved/data-root/manifests")  # 由部署流程预先创建
target = manifest_root / "example-step0.json"
publish_manifest_atomic(target, manifest, manifest_root=manifest_root)
```

默认发布为原子 no-clobber：目标存在即失败。候选尚未 READY 且确需沿同一路径推进状态时，替换必须提供旧文件 SHA-256 作为 compare-and-swap 预期值：

```python
expected = sha256_file(target)
publish_manifest_atomic(
    target,
    advanced_manifest,
    manifest_root=manifest_root,
    allow_replace=True,
    expected_previous_sha256=expected,
)
```

实现会持有每目标持久 advisory lock，并在锁内重新读取 digest。并发写入或过期 digest 会失败；替换还必须保持相同 `asset_id` 和不可变字段，并完整保留旧历史后只追加合法转换。`ready` 与 `invalid` 路径都不能原地替换。

## READY 运行时解析

训练和评估入口只应使用 `resolve_ready_asset`：

```python
from param_importance_nlp.assets import resolve_ready_asset

asset = resolve_ready_asset(
    "/approved/data-root/manifests/example-step0.json",
    "/approved/data-root/models/example-step0",
)
weights = asset.path_for("model.safetensors")
```

非 `ready` 状态立即失败。READY 解析始终重新检查安全路径、普通文件、大小和完整 SHA-256；不存在关闭哈希验证的运行时旁路。

## 旧资产迁移

带 BOM、缺少固定 revision、类型化元数据、大小/SHA-256、角色/证据或连续状态历史的旧清单都不是 v1 manifest。迁移时保留旧文件作为诊断证据，对最终对象重新执行完整验真，生成新的 v1 候选，并发布到新路径；不得覆盖旧证据或把旧 READY 标记直接沿用为当前 gate 结果。

## G3 受控流式中继

当服务器不能直接取得冻结对象时，使用 `ops/stage0/dispatch_g3_relay_via_lab.py` 经文档规定的 SSH alias 中继。dispatcher 只接受 download plan 中的对象 ID、命名 endpoint profile 和命名 lab-pc Python profile；不得把 URL、签名参数或任意远程命令作为参数写入日志或证据。

- `--endpoint-profile official` 是默认源；已核验镜像路径时可显式选择 `--endpoint-profile hf-mirror`。两者都在 lab-pc 进程内根据同一固定 revision 和对象 ID 派生 URL。
- Hugging Face 的 model/dataset repository namespace 不从文件扩展名猜测：它只由同一哈希绑定 download-plan entry 的 `asset_root_ref` 首段派生；`models/` 对应模型仓库，`datasets/` 对应 dataset 仓库的 `/datasets/` namespace，其他根失败关闭。
- `--lab-python-profile path` 使用远端 `PATH`；受控 lab-pc 的用户级 Python 3.12 使用 `--lab-python-profile cjl-python312`。profile 是版本化枚举，不接受任意解释器命令。
- 若 Windows `ssh.exe` 在 Python 双向 stdin/stdout 管道下不能转发 `READY`，使用 `--relay-process lab-pipe`。dispatcher 先以独立 `plan-only` 会话在对象锁内取得精确 resume offset，再用本机 `cmd.exe` 原生字节管道连接 lab emitter 与服务器 receiver；receiver 同时要求锁内 offset 未漂移，竞争或状态变化会失败关闭。
- `lab-pipe` 每项默认最多执行 6 个有界 session；连接或 transcript 未完整收口时，每次都重新运行锁内 plan、记录不含 URL 的 offset 后再续传。`--lab-pipe-max-attempts` 只能设置正整数，所有 session 共用同一个整体 monotonic deadline。
- relay 不在本机或 lab-pc 落盘对象字节；服务器 receiver 在对象专用锁内复用 `.part`，并在固定字节数和 SHA-256 全部通过后 no-clobber 发布最终文件。
- 每次中继后仍须运行离线 acquisition、独立 verification 和 READY materialization；中继成功本身不是 G3 Gate PASS。

示例：

```powershell
python ops/stage0/dispatch_g3_relay_via_lab.py `
  --source-root . `
  --requirements configs/stage0/g3-asset-requirements-v1.json `
  --layout configs/stage0/g3-asset-layout-v1.json `
  --plan configs/stage0/g3-download-plan-v1.json `
  --relay-process lab-pipe `
  --endpoint-profile hf-mirror `
  --lab-python-profile cjl-python312 `
  --object-id huggingface/EleutherAI/pythia-410m-deduped/config.json
```
