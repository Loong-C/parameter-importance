# H1–H6、指标与 Gate 映射

| 假设 | 主证据 | 独立单位 | 质量前提 | 失败状态 |
|---|---|---|---|---|
| H1 raw 正偏 | raw bias calibration、model-total signed bias | repetition/checkpoint/model | fixed state、finite values | supported / not_supported / inconclusive |
| H2 `1/B` 衰减 | 分层 raw bias slope | repetition/checkpoint/model | candidate B 完整、独立 sizing | supported / not_supported / inconclusive |
| H3 U/double 无偏 | model-total、layer-total L1、module-total L1 signed bias | repetition/checkpoint/model | reference h_ref、numeric epsilon | supported / not_supported / inconclusive |
| H4 U 的 M 不变性 | signed mean、negative fraction、M drift | repetition/checkpoint/model | nested M、equal total budget | supported / not_supported / inconclusive |
| H5 等预算效率 | corrected parameter NMSE、variance、MSE、online cost | repetition/checkpoint/model | double floor > tau_nmse、fair budget | supported / not_supported / inconclusive |
| H6 排序恢复 | parameter Spearman、Overlap@1%、Jaccard | repetition/checkpoint/model | canonical IDs、average ties | supported / not_supported / inconclusive |

主 bias 方法资格采用六个 model×stage cells 的 intersection-union，不用事后多重
比较替代；逐 layer/tensor 的 exploratory 结果统一使用 FDR 或 simultaneous interval。
`quality_gates` 是执行证据，`hypothesis_decisions` 是科学结论，两者必须分开。
