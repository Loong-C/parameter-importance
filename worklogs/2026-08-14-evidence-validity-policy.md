# 2026-08-14 证据有效性与最小重验规则修订

- 任务范围：修订 `Agent/` 全部工作规范及仓库内与整链重跑、提交绑定和失败恢复冲突的规则；不启动任何正式实验或 GPU 重放。
- 当前状态：阶段完成
- 工作分支：`feat/stage1-cpu-evidence`

## 2026-08-14 规则模型与第一轮实现

### 目标与范围

- 取消“当前 HEAD 不同即上游证据失效”和“交付必须重新运行整条 Stage 0”的粗粒度规则。
- 建立 producer/consumer 提交分离、内容哈希、依赖闭包、时效性刷新、最近安全边界恢复和有界重试规则。
- 不修改或覆盖既有 Stage 0 运行证据，不进行正式重跑。

### 实际修改

- 重写 Git 忽略的 `Agent/git.md`、`remote_access.md`、`server.md`、`sync.md`、`worklogs.md`，统一保护用户修改、按需同步、资产复用、有界连接恢复和最小重验语义。
- 新增版本化政策 `policies/evidence-validity-and-rerun.md`，作为所有 Stage 的规则优先入口。
- 修订 Stage 0 G9/G10 runbook、计划总览和 Stage 1–3 交付条款，取消无条件三端同 HEAD 与无关整链重放。
- 细化 `configs/stage0/g9-test-matrix-v1.json`：文档/worklog/下游/G10 consumer 变化不重跑 G9 层；硬件变化不触发本机或服务器 CPU 层；logging-only 只重验 CPU/故障层。
- 将统一政策加入 G9/G10 关键源码清单，并加入回归断言。

### 验证与问题

| 项目 | 结果 | 说明 |
|---|---|---|
| `git diff --check` | 通过 | 无空白错误 |
| 第一轮 G9/G10/配置测试 | 未通过，已定位 | 测试临时根目录权限异常；新矩阵需 canonical JSON；新增政策链接尚未进入 Git 索引；runbook 链接数断言需更新 |

第一轮失败没有触发任何正式重放。已把矩阵重新编码为 canonical JSON、更新链接计数和关键文件清单；临时目录权限问题将在仓库受控的全新 `--basetemp` 下针对性复验。

### 证据身份与有效性

- 既有 G0–G9 证据的 `producer_commit` 和产物保持不变。
- 本次只改变规则、G9 回归矩阵和 G10/G9 的关键文件清单；未运行或伪造新 Gate。
- 后续若消费旧 G9，必须发布独立兼容性判定；不得修改旧索引，也不得只因本次提交变化全链重跑。

### 下一步

- 在受控临时目录运行针对性测试和规则残留扫描。
- 审查完整差异；达到稳定边界后提交、推送，并按新同步规则只在确有交付需要时同步服务器和 `Agent/`。

## 2026-08-14 机器执行与针对性回归

### 实际修改

- 新增 `evidence-reuse-attestation-v1` schema、构建/校验模块与 G9 沿用证明生成入口。
- G9 loader 现在保留自身实际 `generator_git_commit`，可以在不冒充当前提交的前提下完整校验旧 G0–G9 链。
- G10 在 G9 producer 与当前 consumer 不同时强制要求沿用证明；证明会重新计算真实 Git 差异，要求每个变化都有非失效分类和必要的针对性测试引用，并核对旧 G9 index SHA-256。未审查变化、producer 语义变化、缺失测试引用或哈希漂移均失败关闭。
- 同提交时禁止提供多余沿用证明；G10 index 分别记录 G9 producer、G10 consumer 和证明引用/哈希。

### 验证

| 项目 | 命令/配置 | 结果 | 证据 |
|---|---|---|---|
| 证据沿用、G9、G10 针对性测试 | `pytest --basetemp=.codex-tmp/policy-tests-20260814-b tests/test_evidence_reuse.py tests/test_stage0_g9.py tests/test_stage0_g10.py -k "not g10_real_repository_paths_match_git_index_case_exactly"` | 21 passed，1 deselected | 本机控制台 |

被排除的一个用例只检查 G10 关键文件是否已进入 Git 索引；新增文件在审查完成并暂存后再运行。测试过程未启动 GPU、服务器 formalizer 或任何正式重放。

暂存后该索引检查确认所有新增关键政策、schema、生成器和校验模块均已进入 Git；测试只因两份 runbook
各新增一个政策链接而发现旧的固定链接计数仍为 8，已将预期更新为 10。该失败属于测试预期维护，
不影响既有实验或触发任何重放。

最终针对性集合在受控 `--basetemp` 下为 **28 passed**，覆盖沿用证明、G9、G10、run-ready 配置、
恢复 run spec 和新 schema 自校验；`compileall` 与 `git diff --cached --check` 通过。Windows 默认 pytest
临时根目录和仓库 `.pytest_cache` 的权限告警已通过受控临时目录/禁用 cacheprovider 隔离，未把该本机
权限问题误判成实现失败。

沿用证明还会绑定所有 `validation_refs` 的 SHA-256；证明发布后若任何验证文件被替换，即使路径和 PASS 文本仍在，也会因身份不一致而拒绝复用。该行为已包含在上述 28 个回归测试中。
