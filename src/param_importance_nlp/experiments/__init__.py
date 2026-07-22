"""Stage 2 固定状态估计器的可复现实验编排合同。

这里只导出稳定数据类型与纯 CPU 编排；导入本包不会访问服务器、下载模型或
导入 Hugging Face。需要 Torch 的核心估计器/求积适配器在真正执行时才加载。
"""

from .sampling import (
    CANDIDATE_BATCH_SIZES,
    CANDIDATE_MICROBATCH_COUNTS,
    MICROBATCH_SELECTION_ORDER,
    STREAM_NAMES,
    CandidateEvaluation,
    Draw,
    FormalDecisionBlocked,
    PilotObservation,
    PrimaryPairDecision,
    RepetitionMapping,
    SamplingPlan,
    SamplingUniverse,
    select_primary_pair,
)
from .stage2 import (
    COST_SEMANTICS,
    CoreEstimatorKernel,
    DeterministicShardReducer,
    EstimatorDecision,
    PairedEstimatorResult,
    PairedEstimatorRunner,
    ReducedMoments,
    ReducedSufficientStatistics,
    ReferenceResult,
    ReferenceRunner,
    ShardArtifactStore,
    Stage2FixtureStudy,
    SufficientStatisticShard,
    build_fixture_estimator_decision,
)

__all__ = [
    "CANDIDATE_BATCH_SIZES",
    "CANDIDATE_MICROBATCH_COUNTS",
    "COST_SEMANTICS",
    "CandidateEvaluation",
    "CoreEstimatorKernel",
    "DeterministicShardReducer",
    "Draw",
    "EstimatorDecision",
    "FormalDecisionBlocked",
    "MICROBATCH_SELECTION_ORDER",
    "PairedEstimatorResult",
    "PairedEstimatorRunner",
    "PilotObservation",
    "PrimaryPairDecision",
    "ReducedMoments",
    "ReducedSufficientStatistics",
    "ReferenceResult",
    "ReferenceRunner",
    "RepetitionMapping",
    "STREAM_NAMES",
    "SamplingPlan",
    "SamplingUniverse",
    "ShardArtifactStore",
    "Stage2FixtureStudy",
    "SufficientStatisticShard",
    "build_fixture_estimator_decision",
    "select_primary_pair",
]
