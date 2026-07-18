# 2026-07-18 Pile 状态与 Pythia-31M step0 资产准备

- 任务范围：确认全量 Pile 下载是否仍在运行；若未运行则恢复；在服务器大盘 `models/` 中准备 Pythia-31M 的初始权重。
- 当前状态：模型与 Pile 状态验收完成，Git 三端同步待完成
- 工作分支：`main`

## 2026-07-18 14:06 CST — Pile 状态核对

### 实际状态

- lab-pc 计划任务 `CjlPileFull` 与 `CjlPileFullSupervisor` 均为 `Running`，因此没有启动重复下载进程。
- 下载日志仍持续产生重试记录；当前失败点是 `cas-bridge.xethub.hf.co` 的公共 DNS 解析结果无法由两个公共解析器相互印证。
- 服务器 Pile shard 0–4 仍完整；shard 5 的 `.part` 为 `12,518,366,905` bytes，未被缩短或重建。
- 本任务没有修改 Pile 下载脚本、计划任务定义、锁、元数据或 `.part`。

### 结论

Pile 下载监督链仍在运行，但 shard 5 的字节进度暂时受外部 DNS/Xet 网络问题阻塞；监督器会继续重试，无需额外启动第二套下载。

## 2026-07-18 14:30 CST — Pythia-31M 初始权重下载

### 模型身份与文件范围

- Hugging Face 仓库：`EleutherAI/pythia-31m-deduped`。
- 请求 revision：`step0`；固定不可变 revision：`73628c85dd9d12d43c07be77ebcf10cef5fd9660`。
- `model.safetensors` 预期大小：`121,987,544` bytes；预期 SHA-256：`5dad0c31b7c19af86f5840674fa4f16a91bdf617245aae17387bdaedd3600319`。
- 只保留 Transformers 本地加载需要的 `config.json`、`generation_config.json`、`model.safetensors`、`special_tokens_map.json`、`tokenizer.json`、`tokenizer_config.json`；不下载重复的 `pytorch_model.bin` 或训练状态文件。

### 网络路径与处理

- 服务器访问 `hf-mirror.com` 会被重定向回不可达的官方站点；lab-pc 访问同一镜像 API 返回 HTTP 200。
- 使用 lab-pc 作为传输端，通过镜像下载固定 revision；镜像只承担传输，下载器同时固定 revision 并强制核对模型 LFS SHA-256。
- lab-pc 下载完成后通过现有 SSH 路径上传到服务器精确暂存目录，服务器端执行 `sha256sum -c` 后再原子移动到最终目录。

### 目标路径

- 服务器最终目录：`/home/sophgo13/cjl/storage/parameter-importance/models/pythia-31m-deduped-step0`。
- 服务器资产 manifest：`/home/sophgo13/cjl/storage/parameter-importance/manifests/pythia-31m-deduped-step0.json`。

## 验证、Git 与同步

- 服务器最终目录包含 8 个文件：6 个运行文件、`model-manifest.json` 和 `SHA256SUMS`；`sha256sum -c SHA256SUMS` 全部通过。
- `model.safetensors` 实际大小为 `121,987,544` bytes，实际 SHA-256 为 `5dad0c31b7c19af86f5840674fa4f16a91bdf617245aae17387bdaedd3600319`，与固定预期一致。
- 服务器环境离线加载通过：架构 `GPTNeoXForCausalLM`，模型类型 `gpt_neox`，76 个 safetensors tensor，参数 dtype 为 `torch.float32`，总参数量 `30,494,720`，tokenizer vocab size `50,254`。
- 最终目录权限为目录 `775`、文件 `664`，属主/属组均为 `sophgo13:sophgo13`。
- 独立资产 manifest 已保存到规定的 `manifests/` 目录；与模型目录内 manifest 的 SHA-256 均为 `1f7f812bb522ddb344d5ae10e66b94dc4e11ac9049e7eedbb737350719a24e59`。
- lab-pc 临时模型任务最终退出码为 0，日志记录完成时间为 14:37:37 CST；随后精确注销该任务并删除临时 `model-assets/`，工具根目录重新只保留 `pile-full/`。
- 清理后 `CjlPileFull` 与 `CjlPileFullSupervisor` 仍为 `Running`；Pile 日志持续更新至 14:35 之后，shard 5 `.part` 仍为 `12,518,366,905` bytes。
- 本次用户对 `docs/` 与 `plan/` 的现有修改按协作规则一并纳入阶段提交，不回滚、不拆分规避。
- 本机、GitHub 与服务器最终 HEAD、工作树状态和临时传输项清理：待完成。

## 尚未解决的问题与下一步

- 提交并推送本日志和用户已有修改，再以增量 bundle 快进服务器。
- Pile 当前外部 DNS/Xet 阻塞仍未解决；监督器将持续自动重试。
