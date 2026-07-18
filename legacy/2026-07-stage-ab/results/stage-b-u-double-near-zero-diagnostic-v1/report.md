# Stage B U-stat 与 double sampling 对称剪枝诊断

本诊断没有重新训练模型，只读取六个既有 SST-2 最终 checkpoint、importance 和原始 594 行剪枝结果，并新增 `double_near_zero` 掩码评估。原 Gate 5 的定义、阈值、失败状态和原产物均未修改。

## 三组比较

| 比较 | 适用行 | 通过 | 失败 | 全部通过 |
|---|---:|---:|---:|---|
| 原 Gate 5：positive U low 对 signed double low | 72 | 68 | 4 | 否 |
| positive U low 对 double near-zero | 72 | 69 | 3 | 否 |
| signed U near-zero 对 double near-zero | 72 | 72 | 0 | 是 |

这里的‘通过’只是把同一条旧判据应用到诊断配对上，不能替代或追溯修改正式 Gate 5。正 gap 表示剪掉高分参数比剪掉对应低分/近零参数更伤害模型。

## 各分数自身的功能区分

| 分数与低端语义 | 非零比例主指标行 | 正 gap | 非正 gap | 平均 gap |
|---|---:|---:|---:|---:|
| u_positive | 64 | 64 | 0 | 0.243689749 |
| u_signed | 64 | 64 | 0 | 0.249810659 |
| double_original | 64 | 64 | 0 | 0.249244916 |
| double_near_zero | 64 | 64 | 0 | 0.24928216 |

## 未满足诊断判据的条件

| 比较 | 路线 | allocation | ratio | metric | U gap | double gap | 所需 U gap |
|---|---|---|---:|---|---:|---:|---:|
| original_gate5_semantics | direct | layer_balanced | 0.3 | nll | 0.158069375 | 0.216326754 | 0.173061403 |
| original_gate5_semantics | pretrained | layer_balanced | 0.1 | nll | 0.165199126 | 0.206511805 | 0.165209444 |
| original_gate5_semantics | pretrained | layer_balanced | 0.2 | nll | 0.127636121 | 0.214585106 | 0.171668085 |
| original_gate5_semantics | pretrained | layer_balanced | 0.3 | nll | 0.0770343636 | 0.221317902 | 0.177054322 |
| positive_vs_double_near_zero | direct | layer_balanced | 0.3 | nll | 0.158069375 | 0.207620667 | 0.166096533 |
| positive_vs_double_near_zero | pretrained | layer_balanced | 0.2 | nll | 0.127636121 | 0.200492714 | 0.160394171 |
| positive_vs_double_near_zero | pretrained | layer_balanced | 0.3 | nll | 0.0770343636 | 0.202610505 | 0.162088404 |

## 分数符号分布

| 路线 | seed | 分数 | 正比例 | 零比例 | 负比例 | 净质量/绝对质量 |
|---|---:|---|---:|---:|---:|---:|
| direct | 1234 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| direct | 1234 | u_signed | 0.990984 | 0.000000 | 0.009016 | 0.998798 |
| direct | 1234 | double | 0.987809 | 0.000000 | 0.012191 | 0.997183 |
| direct | 1337 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| direct | 1337 | u_signed | 0.989891 | 0.000000 | 0.010109 | 0.998822 |
| direct | 1337 | double | 0.984555 | 0.000000 | 0.015445 | 0.997387 |
| direct | 2027 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| direct | 2027 | u_signed | 0.990095 | 0.000000 | 0.009905 | 0.998596 |
| direct | 2027 | double | 0.986056 | 0.000000 | 0.013944 | 0.996679 |
| pretrained | 1234 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| pretrained | 1234 | u_signed | 0.990311 | 0.000000 | 0.009689 | 0.999192 |
| pretrained | 1234 | double | 0.983640 | 0.000000 | 0.016360 | 0.998254 |
| pretrained | 1337 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| pretrained | 1337 | u_signed | 0.989892 | 0.000000 | 0.010108 | 0.999261 |
| pretrained | 1337 | double | 0.980983 | 0.000000 | 0.019017 | 0.998398 |
| pretrained | 2027 | u_positive | 1.000000 | 0.000000 | 0.000000 | 1.000000 |
| pretrained | 2027 | u_signed | 0.990207 | 0.000000 | 0.009793 | 0.999275 |
| pretrained | 2027 | double | 0.982571 | 0.000000 | 0.017429 | 0.998326 |

## 解读边界

- 若 near-zero 对称比较明显减少失败，说明原 Gate 5 的 low 端语义不对称是重要混杂因素。
- 若 `double_near_zero` 自身仍频繁出现非正 gap，说明 double sampling 在这组功能剪枝任务上也不是稳定的金标准；这只能质疑比较基准，不能单独证明 U-stat 更好。
- 若 signed 对称比较仍失败，则现有证据仍不支持用 U-stat 全面替代 double sampling。
- 本诊断没有构造逐步 `max(double, 0)`；既有 checkpoint 只保存累计 signed double，重建该量需要重放训练，超出本次无重训诊断。
