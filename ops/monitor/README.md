# GPU 空闲探测（S0.9 解锁监控）

目的：定时探测服务器 `sophgo13` 的 4 张允许 GPU 是否空闲，并监测 G0-G 管理员
finalizer 的**最新一次运行**是否为 `SUCCESS` 且其 `boot_id` 等于当前 boot；两者
同时满足才视为 S0.9 可重跑（`S0_9_READY.flag`）。

## 手动运行

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ops\monitor\check_gpu_idle.ps1
```

## 注册计划任务（每 30 分钟）

```powershell
$action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument '-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "D:\Personal\Code\parameter-importance\ops\monitor\check_gpu_idle.ps1"'
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration (New-TimeSpan -Days 3650)
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName 'CheckStage0GpuIdle' -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force
```

## 查看结果

- 历史：`ops/monitor/runtime/gpu_idle_history.csv`
- 最新：`ops/monitor/runtime/latest.json`
- 状态变化：`ops/monitor/runtime/state_changes.txt`
- 就绪标记：`ops/monitor/runtime/S0_9_READY.flag`（GPU 空闲且最新 finalizer 为当前 boot 的 SUCCESS）

## 卸载

```powershell
Unregister-ScheduledTask -TaskName 'CheckStage0GpuIdle' -Confirm:$false
```

运行时产物已由 `.gitignore` 忽略，不入库。
