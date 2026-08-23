# Stage 2.1 Amendment（append-only）

```yaml
schema_version: stage2-preregistration-amendment-v1
amendment_id: REPLACE_WITH_APPEND_ONLY_ID
parent_preregistration_hash: REPLACE_WITH_64_HEX_HASH
state: DRAFT
created_before_confirmatory_draws: true
reason: REQUIRED_NON_EMPTY_JUSTIFICATION
changed_fields: []
unchanged_fields: [estimand, estimators, primary_endpoints, decision_tree]
non_posthoc_basis: REQUIRED_IMPLEMENTATION_ASSET_OR_RESOURCE_FACT
affected_gates: []
append_only: true
amendment_hash: RECOMPUTE_CANONICAL_HASH
review:
  reviewer: null
  reviewed_at: null
  decision: PENDING
```

Amendment 必须在读取任何确认性梯度/生成对应 sample mapping 前提交；不得以 pilot
结果方向、方法均值、NMSE、排序或显著性为依据。原注册和所有旧 amendment 保留，
新文件只能追加。
