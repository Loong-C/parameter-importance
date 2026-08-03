# Stage 0 最终交付与 G10 运行说明

本说明执行 [S0.12 最终交付计划](../plan/stage0/12_delivery_and_sync.md)，并把结果交给 [Stage 1 静态交接说明](stage1-handoff.md)。任何一步不满足时保持“收尾中”，不得创建或复用 `READY`。

## 不可颠倒的顺序

1. 在本机完成全部审查、测试、中文 Worklog 和最终实现提交，确认工作树干净。
2. 经用户明确授权后，非强制推送目标分支到 GitHub；重新读取远端 branch HEAD。
3. 验证服务器仓库干净，从只含目标提交关系的 Git bundle 做 `--ff-only` 快进；不得 merge 非快进分叉。
4. 同步 Git 忽略的 `Agent/` 五文件，只处理 `git.md`、`remote_access.md`、`server.md`、`sync.md`、`worklogs.md`，逐文件核对 SHA-256。
5. 精确删除本次本机 bundle 与 `$DATA_ROOT/tmp` 中的同名 bundle；不使用通配递归删除。
6. 在最终同一提交上重新运行 G0–G9 formal 链。源码、环境、资产或配置变化后，不得复用旧提交的 Gate。
7. 运行只读三端观察采集器；输出必须在 Git 仓库外。将观察文件传入服务器 `DATA_ROOT` 的不可变 evidence 路径并核对 SHA-256。
8. 运行 G10 formalizer。只有它重新验证 G0–G9、三端 HEAD、Agent 哈希、仓库清单、13 项资产、G1-D 有效期和 Stage 1 边界后，才原子发布新 `READY`。
9. G10 后不得再修改 Git 内容或 `Agent/`；任何后续变化都使该 readiness 失效并要求重新收尾。

服务器路径固定为：

```text
/home/sophgo13/cjl/parameter-importance
/home/sophgo13/cjl/storage/parameter-importance
```

SSH 只使用既有别名 `sophgo13-via-lab`，不得修改代理、DNS、跳板或 SSH 拓扑。

## 三端观察

在本机、且 GitHub/服务器同步和 bundle 清理均已完成后运行：

```powershell
$env:PYTHONPATH='src'
python ops/stage0/collect_g10_sync_observation.py `
  --repository D:\Personal\Code\parameter-importance `
  --branch feat/stage0-completion `
  --previous-github-head <同步前 GitHub HEAD> `
  --previous-server-head <同步前服务器 HEAD> `
  --authorization-ref <本任务明确外发授权引用> `
  --output <仓库外专用目录>\stage0-g10-sync-observation.json
```

采集器只读执行本机 Git、`git ls-remote` 与固定服务器探针；它不会 push、复制、删除或修改任一端。以下任一情况都会失败：工作树脏、HEAD/分支不一致、旧 HEAD 不是新 HEAD 祖先、强推声明、远端 URL 漂移、Agent 文件集合或哈希不一致、bundle 残留、`docs/mathematics.md` 未被保留。

观察文件传入服务器后，记录本机与服务器 SHA-256；本机临时副本只按已授权精确路径处理。观察文件是 G10 输入证据，不得手工编辑。

## 正式 G10

在服务器冻结环境中设置 `PARAM_IMPORTANCE_DATA_ROOT=/home/sophgo13/cjl/storage/parameter-importance`，使用最终 G9 index 运行：

```bash
python ops/stage0/formalize_g10.py \
  --data-root /home/sophgo13/cjl/storage/parameter-importance \
  --g9-index-ref <最终 G9 index logical ref> \
  --sync-observation-ref <DATA_ROOT 内观察 logical ref> \
  --repository /home/sophgo13/cjl/parameter-importance \
  --git-commit <最终三端共同 HEAD> \
  --git-branch feat/stage0-completion
```

成功输出必须给出 G10 index、三类 task output、最终 environment 和 readiness logical ref。随后用 `load_stage0_g10_formal_state` 再读取验证；检查 readiness 为 `READY`、G0–G10 共 21 个记录全部 `PASS`、`approved_exceptions=[]`，且 G1-D 明确显示单盘权威副本不是备份。

## 失败与恢复

- 非快进、远端新增提交、服务器脏工作树：停止，不自动 merge/rebase，不强推。
- Agent 不一致：只对五个规定文件做逐文件审查与同步；不把 `Agent/` 加入 Git。
- formal Gate 失败或 skip：保留原报告和退出状态，修复后用新路径重跑，不覆盖历史。
- readiness 路径已存在但内容不同：视为不可变证据冲突，停止；不得覆盖。
- G1-D 到期或提前终止：G10 失败；先取得第二故障域与恢复证据或新的明确风险决定。
