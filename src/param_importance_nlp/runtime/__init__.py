"""运行时层：事件、张量产物、checkpoint、归约和优化器事务。"""

from .checkpoint import (
    CheckpointCommit,
    CheckpointRetentionApplication,
    CheckpointRetentionPolicy,
    CheckpointRetentionSelection,
    CheckpointStore,
)
from .events import (
    EventRecord,
    EventSink,
    EventType,
    JsonlEventSink,
    canonical_optimizer_steps,
    read_event_stream,
)
from .gradients import GradientAttempt, GradientPhase
from .lineage import AttemptDisposition, AttemptLineageRecord, LineageStore
from .optimizer import OptimizerBridge, StepOutcome, compute_global_clip_factor
from .process_state import ProcessSnapshot, ProcessStateStore
from .reducers import (
    LocalReducer,
    Reducer,
    ReducerCapabilities,
    TorchDistributedReducer,
)
from .tensor_bundle import TensorBundle, load_tensor_bundle, publish_tensor_bundle
from .transactions import StepPhase, StepTransaction

__all__ = [
    "CheckpointCommit",
    "CheckpointRetentionApplication",
    "CheckpointRetentionPolicy",
    "CheckpointRetentionSelection",
    "CheckpointStore",
    "AttemptDisposition",
    "AttemptLineageRecord",
    "EventRecord",
    "EventSink",
    "EventType",
    "GradientAttempt",
    "GradientPhase",
    "JsonlEventSink",
    "LocalReducer",
    "LineageStore",
    "OptimizerBridge",
    "ProcessSnapshot",
    "ProcessStateStore",
    "Reducer",
    "ReducerCapabilities",
    "StepOutcome",
    "StepPhase",
    "StepTransaction",
    "TensorBundle",
    "TorchDistributedReducer",
    "canonical_optimizer_steps",
    "compute_global_clip_factor",
    "load_tensor_bundle",
    "publish_tensor_bundle",
    "read_event_stream",
]
