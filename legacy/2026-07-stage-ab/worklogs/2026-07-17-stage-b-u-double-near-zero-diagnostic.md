# 2026-07-17 Stage B U-stat / double sampling 对称剪枝诊断

## 任务目标与边界

- 在不重新训练的前提下，复用 direct/pretrained × seed 1234/1337/2027 六个正式 SST-2 最终 checkpoint、importance 和原始 pruning 产物，补算 `double_near_zero`。
- 并列比较：原 Gate 5 语义、positive U 对 double near-zero、signed U 对 double near-zero；统计 `u_positive`、`u_signed`、`double` 的符号与质量分布。
- 原 Gate 5、阈值、失败状态和六份精确 594 行 pruning 产物保持不变；本诊断只能用于解释，不能事后改写正式门禁。
- 全量 Pile 下载继续运行；诊断占用 GPU 时必须通过 manager 租约自动暂停并保留 `.part`，作业结束后自动续传。

## 开始前多端状态

- 本地分支：`feat/nlp-pythia-stage-ab`；本地与 origin 均为 `233ab202d10f9f41f00510f023e9945380724b87`。
- 服务器仓库：`63eed98947869385f421d436e1d812e67998f771`，tracked worktree 干净；它是本地/origin 的 fast-forward 祖先，尚未同步用户已有的 `233ab20 move agent file`。
- 本任务未编辑、暂存、恢复或覆盖用户原有删除 `Agents/工作日志与多端同步规范.md`；当前规范从 `Agent/工作日志与多端同步规范.md` 读取。
- 服务器 `training-active=0`，`curl=1`。GPU UUID 映射正常枚举 7 卡；本诊断固定使用已验证的 `GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd`，不使用无法建立 CUDA context 的 `GPU-dc6...`。
- lab 旧任务 `CjlNlpTrainingChain-stage-b-downstream` 查询结果为“系统找不到指定的文件”，确认不存在；`CjlPileFull` 和 `CjlPileFullSupervisor` 均为“正在运行”。
- Pile 已完成 `document-00000` 至 `document-00003`，各 `30,000,000,000` bytes；`document-00004-of-00020.bin.part` 核验时为 `19,512,862,960` bytes。唯一 curl 的 nice 值为 `19`、状态为 `SN`，输出目标正是该 `.part`。

## 实现

- 新增 `configs/eval/stage_b_u_double_near_zero_diagnostic_v1.yaml`，严格固定六个来源及顺序。
- 新增 `param_importance_nlp.eval.run_u_double_diagnostic`：
  - fail closed 复核每份原 pruning `_SUCCESS`、精确 594 行、全部登记哈希、provenance linkage、最终模型/importance 来源；
  - 重新执行 aligned source checkpoint、best-validation、step 和 double-step 哈希门禁；
  - 只新增一个确定性方法 `double_near_zero = smallest(abs(cumulative signed double))`，覆盖 9 个 ratio × 2 个 allocation × 6 个来源，正式结果必须精确 108 行；
  - 当前重算 baseline 必须与原 pruning baseline 逐指标一致；
  - 训练 seed 是聚合单位，不能把 mask 或评估行冒充独立 seed；
  - 输出 `results`、`gaps`、`aggregate`、`pairwise`、`score-signs`、`summary`、中文 `report.md`、provenance、resolved config、runtime manifest 和 `_SUCCESS`，所有正式文件均登记 SHA-256；
  - 原子发布且拒绝覆盖已有输出，失败 staging 保留 `FAILURE.json`。
- CLI 新增 `u-double-diagnostic`；server manager 新增 `stage-b-u-double-diagnostic-v1`，固定单 GPU、全局独占和训练租约。
- 下游校验器新增 `u_double_diagnostic` 类型，强制精确 108 行与全部文件哈希。
- runbook 和 manager README 明确该作业不重训、不改变 Gate 5，并必须经托管入口运行。

## 本地命令与结果

```powershell
python -m py_compile src\param_importance_nlp\eval\run_u_double_diagnostic.py src\param_importance_nlp\cli.py ops\train\validate_downstream_result.py
$env:PYTHONPATH='src'; pytest -q tests\test_u_double_diagnostic.py tests\test_downstream_result_validation.py tests\test_ops_supervision.py
git -c safe.directory=D:/Personal/Code/parameter-importance diff --check
$env:PYTHONPATH='src'; python -m compileall -q src tests ops\train
```

- 定向结果：`11 passed, 1 skipped`。唯一 skip 是本地 Windows 的 bash 运行时缺少 Linux `mktemp/awk/readlink/flock` 组合，相关脚本仍需在服务器 Linux 全套回归中零跳过验证。
- `diff --check` 与 Python 编译通过；CLI help 已出现 `u-double-diagnostic`。
- 本地全仓 `pytest -q` 在收集 `tests/test_checkpoint.py` 时因本机没有安装 `safetensors` 中止，同时有 5 个相同依赖原因的 skip；这不是正式回归证据。必须在服务器固定 venv 对将要运行的提交执行全套零跳过回归并生成绑定提交的 `unit_test_report.json`。

## 待执行

1. 仅显式暂存本任务文件，提交并推送；本地、origin、服务器 fast-forward 到同一提交。
2. 服务器检查提交、tracked worktree、`training-active=0`、唯一 curl、GPU UUID 与下载任务后，运行全套零跳过回归并绑定提交。
3. 通过 manager 串行启动正式诊断；运行中检查租约、curl 自动暂停、GPU、进度、NaN/Inf、Traceback、ERROR 和确定性异常。
4. 完成后验证精确 108 行、manifest、`_SUCCESS`、全部哈希和六个来源，记录实测结论。
5. 确认租约消失、唯一 curl 自动恢复、两个 Pile 任务仍在运行，再完成最终 Git 三端同步。

## 首轮正式尝试与确定性初始化修复

- 代码提交 `3568c296553ebdb393823fd2dd27981c53fe0094` 已推送并 fast-forward 到服务器；服务器 bash/Python 静态检查通过。
- 服务器全套报告绑定该提交：`155 tests, 0 failures, 0 errors, 0 skips`；显式分布式与 checkpoint 恢复测试为 `2 passed`。重新生成的 `results/reproducibility/stage-ab.json` 同样绑定 `3568c296...`，`multi_gpu_reproducible=true`、`resume_reproducible=true`。
- 2026-07-17 04:15 UTC 通过 manager 启动首轮。租约已发布，但旧下载实例在 22 秒后仍有一个 curl，未及时让出。按训练期间强隔离要求，立即停止并禁用 lab 的 `CjlPileFull`、`CjlPileFullSupervisor`，终止已核验目标为 Pile `.part` 的唯一 curl；`.part` 保留为 `20,382,468,336` bytes，随后确认 `curl=0`。
- attempt-01 完成六份来源哈希预检后，在第一次评估前 fail closed：`RuntimeError: CUBLAS_WORKSPACE_CONFIG must be set before the first CUDA context`。根因是新 runner 在首次 `seed_everything(..., deterministic=True)` 之前已经完成 device 解析和模型 CUDA transfer。没有执行剪枝评估、没有发布正式输出；失败 staging 的 `FAILURE.json` 与 attempt Traceback 均保留。
- 已显式停止 manager 并写入 `PAUSED`，确认 supervisor/worker/训练租约均为 0，Pile 两任务继续保持 Disabled、`curl=0`。
- 修复方案：runner 在任何 device/CUDA 操作之前先建立确定性策略；manager 的诊断分支额外显式导出 `CUBLAS_WORKSPACE_CONFIG=:4096:8` 作为双重门禁，并新增顺序与运维 wiring 测试。修复提交仍需重新跑服务器全套零跳过回归后才能重启。

## 修复后回归与正式完成

- 修复提交：`78329802193d89e121a2e275d8d3ce312bbe1255`，本地、origin、服务器均 fast-forward 到该提交后再回归。
- 服务器 `$DATA_ROOT/reports/unit_test_report.json` 绑定 `7832980`：`156 tests, 0 failures, 0 errors, 0 skips`，SHA-256 `be61c39be40b6ec7307c74fa51e57ffb7f58d194adacff801f99788fad4f05ca`。
- 显式命令 `pytest -q tests/test_distributed_trainer_integration.py tests/test_checkpoint_resume_integration.py` 为 `2 passed`；`run_reproducibility` 绑定 `7832980`，两项布尔值均为 true，JSON SHA-256 `8a8e74092c7ca3cf3acd1b076423e3b3a0ebb01a54984271ae202f1cf41b444f`。
- 重启前再次确认：三端提交一致、两端 tracked worktree 干净、正式输出不存在、`training-active=0`、`curl=0`、两个 Pile 任务 Disabled、manager 无 ALERT，目标 UUID 唯一映射。
- 第二轮环境实测 `CUDA_VISIBLE_DEVICES=GPU-180ff...`、`CUBLAS_WORKSPACE_CONFIG=:4096:8`、offline 变量齐全；来源校验后 GPU 仅出现本轮 PID，curl 全程为 0。进度严格为六来源各 18 行，最终 108/108。
- manager 于 2026-07-17 04:32:58 UTC 原子发布，`complete=1`，无 ALERT；supervisor、worker、训练租约和 GPU 进程全部退出。

## 正式结果、门禁和结论

- 输出：`$DATA_ROOT/results/stage-b-u-double-near-zero-diagnostic-v1`。
- `validate-result` 返回 `ok=true`、`record_count=108`、provenance ID `fb3e818056fa36e82ec37ddba306e9d924d2676d054e28e37fe444bbe3b35bfa`，全部 11 个登记文件哈希通过；`_SUCCESS` SHA-256 为 `cdc3eff0cdeaee155bdb18f249a6b817ebb583d6dfdde536dd0efe299e036ee6`。
- 精确机器行数：results 108、gaps 1296、aggregate 432、pairwise 324、score-signs 18。六来源各 18 行，唯一方法 `double_near_zero`，global/layer-balanced 各 54 行，九个 ratio 各 12 行。
- 六份原 pruning 再次逐一通过 manager `validate-result`，每份精确 594 行及全部哈希一致；未修改原产物。
- 原 Gate 5 语义复现 `68/72`，失败 4 项；positive U 对 double near-zero 为 `69/72`，失败 3 项；signed U 对 signed double 的 near-zero 对称比较为 `72/72`。
- signed U 对 double 的参数 Spearman 为 `0.962596–0.969190`，layer/module 六份均为 `1.0`；positive U 对应范围较低：参数 `0.913501–0.935414`、layer `0.846154–1.0`、module `0.9–1.0`。
- double-near-zero 自身 64/64 个非零比例 accuracy/NLL 组为正 gap，故不支持“double sampling 基准本身很差”。
- positive-only 总质量为 signed U 净质量的 `1.470–1.606` 倍，尽管最终 signed U 负标量仅约 `0.90%–1.01%`。逐步截断保留了本应抵消的瞬时正贡献；主要问题是 positive-only 改变了估计对象，而不是 signed U-stat 失去替代能力。
- 原 Gate 5 失败状态未改写。正式中文总结见 `docs/stage_b_u_double_near_zero_diagnostic_summary.md`。

## 下载恢复与遗留风险

- 诊断门禁全部完成后重新 Enable 并启动 `CjlPileFull`、`CjlPileFullSupervisor`，两者均显示“正在运行”；旧 `CjlNlpTrainingChain-stage-b-downstream` 仍不存在。
- `.part` 保留为 `20,382,468,336` bytes。恢复时公共 DNS 双解析未能形成一致地址，任务按既有策略持续自动重试，核验时 curl 为 0；按用户要求不再长期监督 600GB 下载。
- Stage C 前遗留的方法学风险：必须预注册 signed U 的跨重复稳定化方案与门槛；positive-only 只能作为单独 ablation；double 可考虑缩为校准基线，但覆盖率必须预先固定。
