# 2026-07-18 旧实验归档与彻底清场

- 任务范围：将旧 Stage A/B 实验的轻量证据固化到 Git，随后清理服务器旧运行产物、lab-pc 旧训练状态和三端旧分支引用。
- 当前状态：归档构建完成，破坏性清理尚未开始。
- 工作分支：`main`

## 2026-07-18 12:44 CST — 保护基线与证据归档

### 保护范围

- 保留服务器 `datasets`、`models`、`cache`、`envs`、`wheelhouse`、`source`、`manifests`。
- 保留 lab-pc 的 `pile-full/`、`CjlPileFull`、`CjlPileFullSupervisor` 和 `Desktop/start.cmd`。
- 保留服务器 `tmp/pile-full-download/server_xet_download.sh`，SHA-256 为 `b153505597fba6b9f85cb0e68bc19ca08bd81417acffd8108c85bb9c298dcfb8`。
- Pile shard 0–4 均为 30,000,000,000 bytes；shard 5 `.part` 清理前为 12,518,366,905 bytes，锁和元数据均存在。

### 旧产物基线

- `runs`：924,209,265,260 bytes。
- `results`：6,781,213,815 bytes。
- `reports`：14,205,264 bytes。
- `operations`：89,175 apparent bytes。
- `tmp`：2,715,611,224 bytes，其中 Pile 服务目录必须保留。
- 清理前大盘可用空间：2,216,807,796,736 bytes。

### 归档内容与验证

- 服务器证据 tar SHA-256：`ca1bf9f9d2bcc3f6c6eec7d8701bf991a797526201e92c8297f6269541aa420d`。
- 归档来源文件 310 个、33,638,654 bytes：`reports` 82、`results` 54、`runs` 167、旧分支 `docs` 2、`worklogs` 5。
- 所有来源文件小于等于 10 MiB；服务器来源中无符号链接。
- 已排除 checkpoint、best-validation、权重/张量、`results/*/units` 和隐藏失败临时目录。
- 旧分支固定为 `e8354c556dd369cf900cd59d437c1fd59e6a7d48`；当前三端 `main` 在归档提交前均为 `7182b7f03009d2b30c65d92eacd91ddeb8463ac9`。

### 下一步

- 已生成逐文件 `inventory.csv` 和 `MANIFEST.sha256`；313 个归档文件合计 33,783,816 bytes，最大单文件 4,712,733 bytes，哈希、敏感信息、禁用类型和体积门检查均通过。
- 首次在服务器检出提交 `f7cfecf` 后，严格校验发现 68 个原始 CRLF 文本文件被 Git 规范化为 LF；删除操作因此暂停。随后为整个归档目录设置 `-text`，重新暂存原始字节，并要求再次完成服务器 manifest 校验后才能清场。
- 清理过程中检测到用户将 `plan/NLP参数重要性完整实验计划书.md` 删除并新建空文件 `plan/general_plan.md`；按 Git 协作规则原样纳入阶段提交，不代填、不回滚。
- 先提交、推送并同步归档；三端一致后才开始删除旧产物和旧分支引用。
