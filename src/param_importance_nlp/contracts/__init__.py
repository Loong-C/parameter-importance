"""参数重要性项目 Stage 0–9 的共享合同公共入口。

该包只定义数据、身份、状态和资格规则，不依赖训练、实验或分析实现，因此其余分区
可以单向依赖这里。业务模块应从本入口导入公开类型；具体编码辅助函数仍按需从
``jsonio`` 导入，避免把 legacy 兼容误当成普通运行时路径。
"""

from .config import (
    CONFIG_SCHEMA_VERSION,
    CONFIG_SECTIONS,
    ConfigDifference,
    ResolvedConfig,
    diff_configs,
    strict_merge,
)
from .artifacts import (
    ValidatedArtifact,
    validate_estimator_decision_artifact,
    validate_path_integral_result_artifact,
    validate_path_spec_artifact,
    validate_reference_result_artifact,
)
from .errors import (
    CanonicalJSONError,
    ConfigContractError,
    ContractError,
    DependencyUnavailable,
    FormalRunRejected,
    FreezeContractError,
    GateContractError,
    IdentityContractError,
    ProvenanceContractError,
    SeedContractError,
)
from .freeze import ContractFreeze
from .identity import RunIdentity, derive_experiment_id
from .jsonio import (
    JSONScalar,
    JSONValue,
    canonical_json_bytes,
    canonical_json_hash,
    ensure_json_object,
    import_legacy_json,
    load_canonical_json,
    loads_strict_json,
    write_canonical_json,
)
from .provenance import ProvenanceRecord, ProvenanceStatus
from .readiness import (
    FormalReadiness,
    evaluate_formal_readiness,
    require_formal_readiness,
)
from .seed import (
    DEFAULT_SEED_DOMAINS,
    SEED_ALGORITHM_VERSION,
    SeedPlan,
    derive_seed,
)
from .status import (
    ContractState,
    GateRecord,
    GateStatus,
    LocalValidationRecord,
    LocalValidationStatus,
    validate_gate_id,
)

__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_SECTIONS",
    "DEFAULT_SEED_DOMAINS",
    "SEED_ALGORITHM_VERSION",
    "CanonicalJSONError",
    "ConfigContractError",
    "ConfigDifference",
    "ContractError",
    "DependencyUnavailable",
    "ContractFreeze",
    "ContractState",
    "FormalReadiness",
    "FormalRunRejected",
    "FreezeContractError",
    "GateContractError",
    "GateRecord",
    "GateStatus",
    "IdentityContractError",
    "JSONScalar",
    "JSONValue",
    "LocalValidationRecord",
    "LocalValidationStatus",
    "ProvenanceContractError",
    "ProvenanceRecord",
    "ProvenanceStatus",
    "ResolvedConfig",
    "RunIdentity",
    "SeedContractError",
    "SeedPlan",
    "ValidatedArtifact",
    "canonical_json_bytes",
    "canonical_json_hash",
    "derive_experiment_id",
    "derive_seed",
    "diff_configs",
    "ensure_json_object",
    "evaluate_formal_readiness",
    "import_legacy_json",
    "load_canonical_json",
    "loads_strict_json",
    "require_formal_readiness",
    "strict_merge",
    "validate_path_integral_result_artifact",
    "validate_estimator_decision_artifact",
    "validate_path_spec_artifact",
    "validate_reference_result_artifact",
    "validate_gate_id",
    "write_canonical_json",
]
