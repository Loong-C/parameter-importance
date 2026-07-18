# Pythia 160M parameter-importance Stage A/B report

Report ID: `stage-ab-minimum-loop`  
Schema: `stage-ab-report-v1`  
Measured source files: 77  
Missing/invalid inputs: 0

## Stage 15.8 gates

| # | Criterion | Status | Evidence |
|---:|---|---|---|
| 1 | 预训练质量门槛通过 | passed | public_comparison.json |
| 2 | SST-2 两条路线均高于多数类基线 | passed | route means={'direct': 0.8100152905198778, 'pretrained': 0.8176605504587157}; route minima={'direct': 0.7970183486238532, 'pretrained': 0.8165137614678899}; majority=0.5091743119266054 |
| 3 | 高 U-stat positive 参数剪枝损害大于低重要性剪枝 | passed | 64 curve groups |
| 4 | U-stat 区分度不低于 raw | passed | absolute gap AUC={'double': 0.07592371610344305, 'magnitude': 0.02878434379623394, 'movement': 0.055857685275733274, 'raw': 0.07168578218950802, 'u_statistic': 0.07258145570523883} |
| 5 | U-stat 与 double sampling 排序相关和剪枝效果相近或更好 | failed | positive pairwise rank rows=18; pruning_similar=False |
| 6 | signed 与 positive-only 至少一套产生稳定功能排序 | passed | positive curve-gap score types=['signed'] |
| 7 | 中断恢复和多 GPU 结果可复现 | passed | explicit evidence booleans |

## Stage A: estimator bias

Status: **available**.

Direct U-vs-double Spearman:

- parameter, M=None: 0.9692
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9354
- layer, M=None: 1.0000
- module, M=None: 0.9000
- parameter, M=None: 0.9647
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9250
- layer, M=None: 0.9835
- module, M=None: 0.9000
- parameter, M=None: 0.9664
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9313
- layer, M=None: 0.9890
- module, M=None: 0.9000
- parameter, M=None: 0.9666
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9207
- layer, M=None: 0.9121
- module, M=None: 0.9000
- parameter, M=None: 0.9626
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9135
- layer, M=None: 0.8462
- module, M=None: 1.0000
- parameter, M=None: 0.9650
- layer, M=None: 1.0000
- module, M=None: 1.0000
- parameter, M=None: 0.9182
- layer, M=None: 0.9011
- module, M=None: 1.0000

## Stage A: numerical integration

Status: **available**.

## Stage B: pretraining

Status: **available**.
Public-checkpoint PPL ratio: `0.639687711315972`; gate: `True`.

## Stage B: SST-2 baselines

Status: **available**.
- majority_class: accuracy=0.5091743119266054, NLL=0.6929788351143502, ECE=0.0, n=872.
- published_pythia_160m_deduped_step0: accuracy=0.481651376146789, NLL=0.7181956593049775, ECE=0.1111429726683435, n=872.
- published_pythia_160m_deduped_step512: accuracy=0.4908256880733945, NLL=1.080982594315065, ECE=0.3642443128333779, n=872.
- uniform_random_expected: accuracy=0.5, NLL=0.6931471805599453, ECE=0.0, n=872.
Published step0 comparator: `available`.

## Stage B: SST-2 routes

Status: **available**.
- direct: accuracy mean=0.8100152905198778, 95% CI=[0.7970183486238532, 0.8176605504587156], n=3.
- pretrained: accuracy mean=0.8176605504587157, 95% CI=[0.8165137614678898, 0.8188073394495413], n=3.
Fixed 25%/50%/100% comparison points available: 18/18.
Best-model full-train generalization gaps available: 6/6; minibatch training loss is not substituted.
Matched performance: **available**; accuracy tolerance=0.01; tie-break=`highest_min_accuracy;smallest_accuracy_difference;earliest_max_step;direct_step;pretrained_step`.
- seed 1234: direct step 1300 (accuracy=0.8222477064220184, nll=0.45320616101999894), pretrained step 1700 (accuracy=0.8222477064220184, nll=0.43577588862235395).
- seed 1337: direct step 1578 (accuracy=0.8314220183486238, nll=0.4103415306281606), pretrained step 1400 (accuracy=0.8348623853211009, nll=0.4351193975964817).
- seed 2027: direct step 2900 (accuracy=0.8211009174311926, nll=0.5318008384436642), pretrained step 2500 (accuracy=0.8211009174311926, nll=0.5132854263716882).

## Stage B: measured importance distribution and reuse

Status: **available**.
Global, layer/module and pretrain-to-finetune overlap/Spearman measurements are in the `importance_*` and `pretrain_finetune_*` tables. Functional reuse intervention is a separate artifact gate and is never inferred from ordinary finetuned-model pruning curves.
Distribution figures use deterministic samples of hash-verified measured positive tensors; full tensors are used for the tabulated global Gini, HHI, and effective-parameter statistics. Pythia exposes a fused `attention.query_key_value` weight, so module heatmaps report `attention-qkv` and do not fabricate separate q/k/v tensors. Module share and per-parameter mean heatmaps are both emitted.

## Stage B: functional reuse intervention

Status: **available**; exact three-seed coverage gate: `True`.
The gate verifies hash/identity-aligned step512 pretraining and final finetuning artifacts plus the complete global/layer-balanced plan-22.4 factor grid. Effect sizes remain descriptive because no post-hoc pass threshold is introduced.

## Stage B: SST-2 best-performance LR search

Status: **available**. Controlled mechanism runs are excluded from candidate selection.
- direct: lr=0.0003, best validation NLL=0.41731656773374715, accuracy=0.8061926605504587, experiment=`stage-b-sst2-direct-bestperf-lr3e-4-seed1234`.
- pretrained: lr=0.0001, best validation NLL=0.41163272321771044, accuracy=0.8188073394495413, experiment=`stage-b-sst2-pretrained-bestperf-lr1e-4-seed1234`.

## Stage B: pruning

Status: **available**.
High/low and estimator curve differences are in `tables/pruning_curve_differences.csv`; uncertainty bands use configured bootstrap resampling.

## Missing or invalid inputs

None.

## Figures

![estimator_m_stability](figures/estimator_m_stability.png)

![estimator_quality](figures/estimator_quality.png)

![functional_reuse_intervention](figures/functional_reuse_intervention.png)

![importance_layer_heatmap](figures/importance_layer_heatmap.png)

![importance_positive_ccdf](figures/importance_positive_ccdf.png)

![importance_positive_log_histogram](figures/importance_positive_log_histogram.png)

![importance_positive_lorenz](figures/importance_positive_lorenz.png)

![importance_positive_topk_cumulative](figures/importance_positive_topk_cumulative.png)

![importance_positive_violin](figures/importance_positive_violin.png)

![numerical_integration](figures/numerical_integration.png)

![pretrain_finetune_overlap](figures/pretrain_finetune_overlap.png)

![pretrain_importance_layer_positive_checkpoint_heatmap](figures/pretrain_importance_layer_positive_checkpoint_heatmap.png)

![pretrain_importance_layer_signed_checkpoint_heatmap](figures/pretrain_importance_layer_signed_checkpoint_heatmap.png)

![pretrain_importance_module_positive_checkpoint_heatmap](figures/pretrain_importance_module_positive_checkpoint_heatmap.png)

![pretrain_importance_module_positive_mean_checkpoint_heatmap](figures/pretrain_importance_module_positive_mean_checkpoint_heatmap.png)

![pretrain_importance_module_signed_checkpoint_heatmap](figures/pretrain_importance_module_signed_checkpoint_heatmap.png)

![pretrain_loss_perplexity](figures/pretrain_loss_perplexity.png)

![pruning_sst2_direct_accuracy](figures/pruning_sst2_direct_accuracy.png)

![pruning_sst2_direct_ece](figures/pruning_sst2_direct_ece.png)

![pruning_sst2_direct_nll](figures/pruning_sst2_direct_nll.png)

![pruning_sst2_pretrained_accuracy](figures/pruning_sst2_pretrained_accuracy.png)

![pruning_sst2_pretrained_ece](figures/pruning_sst2_pretrained_ece.png)

![pruning_sst2_pretrained_nll](figures/pruning_sst2_pretrained_nll.png)

![sst2_routes](figures/sst2_routes.png)


## Artifacts

- `tables/estimator_metrics_aggregate.csv`
- `tables/estimator_stability.csv`
- `tables/functional_reuse_intervention.csv`
- `tables/functional_reuse_intervention_summary.csv`
- `tables/importance_global.csv`
- `tables/importance_layer_module.csv`
- `tables/numerical_integration_aggregate.csv`
- `tables/pretrain_finetune_overlap.csv`
- `tables/pretrain_finetune_spearman.csv`
- `tables/pretrain_importance_evolution.csv`
- `tables/pretrain_training.csv`
- `tables/pretrain_validation.csv`
- `tables/pruning_aggregate.csv`
- `tables/pruning_curve_differences.csv`
- `tables/pruning_gap_auc.csv`
- `tables/pruning_normalized.csv`
- `tables/pruning_seed_coverage.csv`
- `tables/pruning_seed_differences.csv`
- `tables/sst2_baselines.csv`
- `tables/sst2_best_performance_candidates.csv`
- `tables/sst2_best_performance_selected.csv`
- `tables/sst2_comparison_positions.csv`
- `tables/sst2_generalization_gaps.csv`
- `tables/sst2_matched_performance.csv`
- `tables/sst2_paired_routes.csv`
- `tables/sst2_routes.csv`
- `tables/sst2_runs.csv`
- `tables/sst2_seed_coverage.csv`
- `tables/stage_15_8_gates.csv`
- `tables/u_double_spearman.csv`
- `figures/estimator_m_stability.png`
- `figures/estimator_m_stability.svg`
- `figures/estimator_quality.png`
- `figures/estimator_quality.svg`
- `figures/functional_reuse_intervention.png`
- `figures/functional_reuse_intervention.svg`
- `figures/importance_layer_heatmap.png`
- `figures/importance_layer_heatmap.svg`
- `figures/importance_positive_ccdf.png`
- `figures/importance_positive_ccdf.svg`
- `figures/importance_positive_log_histogram.png`
- `figures/importance_positive_log_histogram.svg`
- `figures/importance_positive_lorenz.png`
- `figures/importance_positive_lorenz.svg`
- `figures/importance_positive_topk_cumulative.png`
- `figures/importance_positive_topk_cumulative.svg`
- `figures/importance_positive_violin.png`
- `figures/importance_positive_violin.svg`
- `figures/numerical_integration.png`
- `figures/numerical_integration.svg`
- `figures/pretrain_finetune_overlap.png`
- `figures/pretrain_finetune_overlap.svg`
- `figures/pretrain_importance_layer_positive_checkpoint_heatmap.png`
- `figures/pretrain_importance_layer_positive_checkpoint_heatmap.svg`
- `figures/pretrain_importance_layer_signed_checkpoint_heatmap.png`
- `figures/pretrain_importance_layer_signed_checkpoint_heatmap.svg`
- `figures/pretrain_importance_module_positive_checkpoint_heatmap.png`
- `figures/pretrain_importance_module_positive_checkpoint_heatmap.svg`
- `figures/pretrain_importance_module_positive_mean_checkpoint_heatmap.png`
- `figures/pretrain_importance_module_positive_mean_checkpoint_heatmap.svg`
- `figures/pretrain_importance_module_signed_checkpoint_heatmap.png`
- `figures/pretrain_importance_module_signed_checkpoint_heatmap.svg`
- `figures/pretrain_loss_perplexity.png`
- `figures/pretrain_loss_perplexity.svg`
- `figures/pruning_sst2_direct_accuracy.png`
- `figures/pruning_sst2_direct_accuracy.svg`
- `figures/pruning_sst2_direct_ece.png`
- `figures/pruning_sst2_direct_ece.svg`
- `figures/pruning_sst2_direct_nll.png`
- `figures/pruning_sst2_direct_nll.svg`
- `figures/pruning_sst2_pretrained_accuracy.png`
- `figures/pruning_sst2_pretrained_accuracy.svg`
- `figures/pruning_sst2_pretrained_ece.png`
- `figures/pruning_sst2_pretrained_ece.svg`
- `figures/pruning_sst2_pretrained_nll.png`
- `figures/pruning_sst2_pretrained_nll.svg`
- `figures/sst2_routes.png`
- `figures/sst2_routes.svg`
