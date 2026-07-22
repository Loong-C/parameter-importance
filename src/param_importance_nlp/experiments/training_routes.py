"""Stage 4–6 训练 phase DAG 的可恢复执行器。

``TrainingRouteSpec`` 冻结科学 lineage，本模块负责按拓扑序真正运行每个 phase。
模型、数据和 optimizer 的构造仍由配置驱动的 ``PhaseBuilder`` 注入；runner 不
硬编码 Pythia、SST-2、MNLI、RTE 或服务器路径。每个完成 phase 先写不可变结果
对象，再写独立 commit，resume 只读取通过 hash 校验且绑定同一路线的 commit。
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Mapping, Protocol

from ..atomic import atomic_write_json, stable_json_hash
from ..contracts.jsonio import load_canonical_json
from ..runtime.training import TrainingRunResult
from .routes import TrainingPhaseSpec, TrainingRouteSpec


def _safe_id(value: str, *, field: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError(f"TRAINING_ROUTE_IDENTIFIER_INVALID:{field}")
    return value


@dataclass(frozen=True, slots=True)
class TrainingPhaseRuntime:
    """builder 返回的 phase 运行资源及父 checkpoint 消费证明。"""

    engine: "PhaseEngine"
    consumed_checkpoint_id: str | None = None
    consumed_parent_result_hash: str | None = None


class PhaseEngine(Protocol):
    """路线 runner 所需的最小 phase engine 协议。

    包装器可以在不复制训练状态机的前提下增加 profiling 或边界回调；真正的参数
    更新仍由 ``TrainingEngine`` 完成。
    """

    def run(self) -> TrainingRunResult: ...


class PhaseBuilder(Protocol):
    """从 phase/config 构造全新的执行资源；不得复用前一 phase optimizer。"""

    def __call__(
        self,
        phase: TrainingPhaseSpec,
        parent_result: Mapping[str, object] | None,
    ) -> TrainingPhaseRuntime: ...


@dataclass(frozen=True, slots=True)
class TrainingRouteResult:
    """一个 route 的阶段结果索引，不复制大型 checkpoint。"""

    route_id: str
    route_lineage_hash: str
    status: str
    phase_results: Mapping[str, Mapping[str, object]]

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "training-route-result-v1",
            "route_id": self.route_id,
            "route_lineage_hash": self.route_lineage_hash,
            "status": self.status,
            "phase_results": {
                phase_id: dict(result)
                for phase_id, result in sorted(self.phase_results.items())
            },
        }
        payload["artifact_hash"] = stable_json_hash(payload)
        return payload


class TrainingRouteRunner:
    """按冻结 DAG 执行、提交和恢复 Stage 4–6 路线。"""

    def __init__(
        self,
        route: TrainingRouteSpec,
        builder: PhaseBuilder,
        *,
        result_root: str | Path,
    ) -> None:
        self.route = route
        self.builder = builder
        self.root = Path(result_root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        _safe_id(route.route_id, field="route_id")

    def _paths(self, phase_id: str) -> tuple[Path, Path]:
        _safe_id(phase_id, field="phase_id")
        return self.objects / f"{phase_id}.json", self.commits / f"{phase_id}.json"

    def _load_committed(self, phase: TrainingPhaseSpec) -> Mapping[str, object] | None:
        object_path, commit_path = self._paths(phase.phase_id)
        if not commit_path.exists():
            return None
        commit = load_canonical_json(commit_path)
        value = load_canonical_json(object_path)
        if not isinstance(commit, dict) or not isinstance(value, dict):
            raise ValueError("TRAINING_ROUTE_PHASE_COMMIT_ROOT_INVALID")
        expected_commit_fields = {
            "schema_version", "route_lineage_hash", "phase_id", "object_sha256"
        }
        if set(commit) != expected_commit_fields:
            raise ValueError("TRAINING_ROUTE_PHASE_COMMIT_FIELDS_MISMATCH")
        if commit["schema_version"] != "training-route-phase-commit-v1":
            raise ValueError("TRAINING_ROUTE_PHASE_COMMIT_VERSION_INVALID")
        if commit["route_lineage_hash"] != self.route.lineage_hash:
            raise ValueError("TRAINING_ROUTE_PHASE_LINEAGE_MISMATCH")
        if commit["phase_id"] != phase.phase_id:
            raise ValueError("TRAINING_ROUTE_PHASE_ID_MISMATCH")
        if commit["object_sha256"] != stable_json_hash(value):
            raise ValueError("TRAINING_ROUTE_PHASE_OBJECT_HASH_MISMATCH")
        if value.get("logical_output_checkpoint_id") != phase.output_checkpoint_id:
            raise ValueError("TRAINING_ROUTE_PHASE_OUTPUT_CHECKPOINT_MISMATCH")
        if value.get("status") != "COMPLETE":
            raise ValueError("TRAINING_ROUTE_COMMITTED_PHASE_NOT_COMPLETE")
        return value

    def _publish_phase(
        self,
        phase: TrainingPhaseSpec,
        result: TrainingRunResult,
    ) -> Mapping[str, object]:
        object_path, commit_path = self._paths(phase.phase_id)
        if object_path.exists() or commit_path.exists():
            raise FileExistsError(f"TRAINING_ROUTE_PHASE_ALREADY_PUBLISHED:{phase.phase_id}")
        run_result = result.to_dict()
        value: dict[str, object] = {
            "schema_version": "training-route-phase-result-v1",
            "route_id": self.route.route_id,
            "route_lineage_hash": self.route.lineage_hash,
            "phase_id": phase.phase_id,
            "phase_type": phase.phase_type,
            "parent_phase_id": phase.parent_phase_id,
            "logical_output_checkpoint_id": phase.output_checkpoint_id,
            "physical_checkpoint_id": result.state.last_checkpoint_id,
            "status": result.status,
            "training_result": run_result,
        }
        value["artifact_hash"] = stable_json_hash(value)
        atomic_write_json(object_path, value)
        # 从磁盘重新读回并复算，之后才发布权威 commit。
        verified = load_canonical_json(object_path)
        if not isinstance(verified, dict) or stable_json_hash(verified) != stable_json_hash(value):
            raise RuntimeError("TRAINING_ROUTE_PHASE_POST_PUBLISH_HASH_DRIFT")
        atomic_write_json(
            commit_path,
            {
                "schema_version": "training-route-phase-commit-v1",
                "route_lineage_hash": self.route.lineage_hash,
                "phase_id": phase.phase_id,
                "object_sha256": stable_json_hash(verified),
            },
        )
        return verified

    def run(self, *, resume: bool = True) -> TrainingRouteResult:
        """执行所有未提交 phase；任何父节点失败都会阻断其后代。"""

        completed: dict[str, Mapping[str, object]] = {}
        for phase_id in self.route.topological_order:
            phase = self.route.phase(phase_id)
            existing = self._load_committed(phase) if resume else None
            if existing is not None:
                completed[phase_id] = existing
                continue
            parent = None if phase.parent_phase_id is None else completed.get(phase.parent_phase_id)
            if phase.parent_phase_id is not None and parent is None:
                raise RuntimeError("TRAINING_ROUTE_PARENT_NOT_COMPLETE")
            runtime = self.builder(phase, parent)
            if not isinstance(runtime, TrainingPhaseRuntime):
                raise TypeError("TRAINING_ROUTE_BUILDER_RESULT_INVALID")
            if phase.input_checkpoint_id != runtime.consumed_checkpoint_id:
                raise ValueError("TRAINING_ROUTE_CONSUMED_CHECKPOINT_MISMATCH")
            expected_parent_hash = None if parent is None else parent.get("artifact_hash")
            if expected_parent_hash != runtime.consumed_parent_result_hash:
                raise ValueError("TRAINING_ROUTE_CONSUMED_PARENT_HASH_MISMATCH")
            result = runtime.engine.run()
            if result.status != "COMPLETE":
                return TrainingRouteResult(
                    self.route.route_id,
                    self.route.lineage_hash,
                    "PARTIAL",
                    completed,
                )
            if result.state.last_checkpoint_id is None:
                raise RuntimeError("TRAINING_ROUTE_PHASE_MISSING_OUTPUT_CHECKPOINT")
            completed[phase_id] = self._publish_phase(phase, result)
        return TrainingRouteResult(
            self.route.route_id,
            self.route.lineage_hash,
            "COMPLETE",
            completed,
        )


__all__ = [
    "PhaseEngine",
    "PhaseBuilder",
    "TrainingPhaseRuntime",
    "TrainingRouteResult",
    "TrainingRouteRunner",
]
