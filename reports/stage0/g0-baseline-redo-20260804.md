# Stage 0 S0.1 重验基线报告（2026-08-04）

- 本机：`redo/from-run-ready` @ `8f1bc5b`，工作树干净，未推送 GitHub（待决策）。
- GitHub：`feat/stage0-completion` @ `8f1bc5b`，`main` @ `34966d0`。
- 服务器：`feat/stage0-completion` @ `1746903`（落后 GitHub 1 个提交），工作树干净。
- `Agent/*.md`：本机与服务器 5/5 SHA-256 一致。
- 系统/存储：Ubuntu 24.04.3 LTS、内核 6.8.0-136-generic、128 CPU、约 1007 GiB 内存；大盘 ext4 3.5T 可用 2.8T，inode 使用 1%；`DATA_ROOT` 属主 `sophgo13:sophgo13`、权限 0750。
- GPU：4 张白名单 A100 均被 `nvidia-smi` 列出，其中 GPU 3（`0000:A4:00.0`）ECC 为 N/A，且存在 PID 不可见的 compute app；结合 08-04 Xid 120/119/154 记录，判定为故障待管理员恢复。

Gate 判定：

- G0-C：`PASS`。
- G0-G：`BLOCKED`（等待管理员恢复 GPU 3 后重验）。
- G0：`BLOCKED`。

机器可读证据：`reports/stage0/g0-baseline-redo-20260804.json`。
