"""Stage 3 端点、probe、持久节点缓存与参考求积的正式编排层。

这里不拥有模型或 optimizer，而是通过窄协议协调真实 adapter：端点捕获必须按
``pre -> optimizer post -> attempt commit -> replay`` 的顺序发生；probe panel 必须
与 update 及彼此互斥；节点缓存和参考级别都采用“不可变 tensor bundle + 独立
权威 commit”。

所有运行结果首先只是候选证据。即使使用 ``run_intent=formal``，也只有绑定本阶段
可接受 Gate 后才会出现 ``formal_eligible=true``。本机 fixture 可以验证数学、恢复
和故障注入，但永远不能触发该资格化分支。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import hashlib
import math
import os
from pathlib import Path
import re
from types import MappingProxyType
from typing import Callable, Hashable, Mapping, Protocol, Sequence

import numpy as np

from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.immutable import (
    freeze_json_mapping,
    thaw_json_value,
)
from param_importance_nlp.contracts.jsonio import (
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.stage23 import (
    ArtifactQualification,
    FormalExecutionEvidence,
    require_accepted_gate,
)
from param_importance_nlp.contracts.status import GateRecord
from param_importance_nlp.runtime.tensor_bundle import (
    load_tensor_bundle,
    publish_tensor_bundle,
)

from .stage3 import EndpointRecord, EndpointState, NodeCacheKey, ProbeSpec


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,159}$")


def _require_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field_name} 不是安全标识")
    return value


def _require_sha256(value: str, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _require_gate_binding(gate: GateRecord, artifact_ref: str | None) -> None:
    if artifact_ref is None or artifact_ref not in gate.evidence_refs:
        raise FormalRunRejected(
            f"FORMAL_GATE_DOES_NOT_BIND_ARTIFACT:{gate.gate_id}:"
            f"{artifact_ref or '<missing>'}"
        )


def _as_array(value: object, *, field_name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()  # type: ignore[union-attr]
    if hasattr(value, "cpu"):
        value = value.cpu()  # type: ignore[union-attr]
    if hasattr(value, "numpy"):
        value = value.numpy()  # type: ignore[union-attr]
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} 包含 NaN/Inf")
    return np.array(array, dtype=np.float64, order="C", copy=True)


def _as_vector(value: object, *, field_name: str = "vector") -> dict[str, np.ndarray]:
    if not hasattr(value, "items"):
        raise TypeError(f"{field_name} 必须是 parameter-name -> tensor mapping")
    result: dict[str, np.ndarray] = {}
    for raw_name, item in value.items():  # type: ignore[union-attr]
        name = str(raw_name)
        if not name or name in result:
            raise ValueError(f"{field_name} 包含空名称或重复参数名")
        result[name] = _as_array(item, field_name=f"{field_name}.{name}")
    if not result:
        raise ValueError(f"{field_name} 不能为空")
    return {name: result[name] for name in sorted(result)}


def _vector_digest(value: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    for name, array in _as_vector(value).items():
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _flatten(value: Mapping[str, object]) -> np.ndarray:
    vector = _as_vector(value)
    return np.concatenate([vector[name].reshape(-1) for name in sorted(vector)])


def _normalized_l1(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    lhs, rhs = _as_vector(left), _as_vector(right)
    if tuple(lhs) != tuple(rhs) or any(lhs[name].shape != rhs[name].shape for name in lhs):
        raise ValueError("参考贡献参数名或 shape 不一致")
    lhs_flat, rhs_flat = _flatten(lhs), _flatten(rhs)
    denominator = float(np.abs(rhs_flat).sum())
    numerator = float(np.abs(lhs_flat - rhs_flat).sum())
    if denominator == 0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def _state_to_dict(state: EndpointState) -> dict[str, str]:
    return {
        "artifact_id": state.artifact_id,
        "artifact_hash": state.artifact_hash,
        "parameter_hash": state.parameter_hash,
        "buffer_hash": state.buffer_hash,
        "optimizer_hash": state.optimizer_hash,
        "scheduler_hash": state.scheduler_hash,
        "scaler_hash": state.scaler_hash,
        "rng_hash": state.rng_hash,
        "data_cursor_hash": state.data_cursor_hash,
        "model_mode_hash": state.model_mode_hash,
    }


def _record_to_dict(record: EndpointRecord) -> dict[str, object]:
    return {
        "path_state_id": record.path_state_id,
        "source_run_id": record.source_run_id,
        "optimizer_step": record.optimizer_step,
        "parameter_registry_hash": record.parameter_registry_hash,
        "pre_state": _state_to_dict(record.pre_state),
        "parameter_post_state": _state_to_dict(record.parameter_post_state),
        "attempt_commit_state": _state_to_dict(record.attempt_commit_state),
        "attempt_commit_parent_hash": record.attempt_commit_parent_hash,
        "probe_buffer_snapshot_hash": record.probe_buffer_snapshot_hash,
        "full_update_delta_hash": record.full_update_delta_hash,
        "update_sample_ids": list(record.update_sample_ids),
        "replay_verified": record.replay_verified,
        "metadata": thaw_json_value(record.metadata),
        "endpoint_digest": record.digest,
    }


@dataclass(frozen=True, slots=True)
class EndpointCaptureRequest:
    """真实 optimizer transition 的静态身份与 update 数据范围。"""

    path_state_id: str
    source_run_id: str
    optimizer_step: int
    parameter_registry_hash: str
    update_sample_ids: tuple[Hashable, ...]
    execution: FormalExecutionEvidence = field(
        default_factory=lambda: FormalExecutionEvidence("local_fixture")
    )
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_identifier(self.path_state_id, field_name="path_state_id")
        _require_identifier(self.source_run_id, field_name="source_run_id")
        if isinstance(self.optimizer_step, bool) or self.optimizer_step < 0:
            raise ValueError("optimizer_step 必须是非负整数")
        _require_sha256(self.parameter_registry_hash, field_name="parameter_registry_hash")
        if not self.update_sample_ids:
            raise ValueError("update_sample_ids 不能为空")
        if self.execution.run_intent == "formal":
            self.execution.require_for_stage(3)
        object.__setattr__(self, "metadata", freeze_json_mapping(self.metadata))


class EndpointCaptureAdapter(Protocol):
    """框架相关端点 adapter 必须实现的严格时序协议。"""

    def capture_pre_state(self) -> EndpointState:
        """在 optimizer 更新前捕获参数、buffer、optimizer 与控制状态。"""

    def apply_optimizer_update(self) -> None:
        """只执行 optimizer 更新；尚不得推进 scheduler/scaler/RNG/数据游标。"""

    def capture_parameter_post_state(self) -> EndpointState:
        """紧接 optimizer 更新捕获 parameter_post_state。"""

    def advance_attempt_commit(self) -> None:
        """推进 scheduler/scaler/RNG/数据游标并发布权威 attempt commit。"""

    def capture_attempt_commit_state(self) -> EndpointState:
        """读取已经权威提交的恢复状态。"""

    def full_update_delta_hash(self) -> str:
        """返回 pre -> parameter_post 的逐参数位移 bundle 摘要。"""

    def probe_buffer_snapshot_hash(self) -> str:
        """返回 pre/post 共同 probe buffer snapshot 摘要。"""

    def verify_replay(self, record: EndpointRecord) -> bool:
        """从 pre artifact 独立重放并逐项比较 post/commit；成功后回到 commit。"""

    def restore_pre_state(self) -> None:
        """捕获失败时恢复 pre 状态；失败不得继续训练。"""


@dataclass(frozen=True, slots=True)
class CapturedEndpoint:
    record: EndpointRecord
    execution_evidence_hash: str
    qualification: ArtifactQualification
    schema_version: str = "stage3-endpoint-capture-v1"

    def __post_init__(self) -> None:
        _require_sha256(self.execution_evidence_hash, field_name="execution_evidence_hash")
        if not self.record.replay_verified:
            raise ValueError("CapturedEndpoint 必须通过独立 replay")

    @property
    def scope(self) -> str:
        return self.qualification.scope

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "record": _record_to_dict(self.record),
            "execution_evidence_hash": self.execution_evidence_hash,
            "scope": self.scope,
            "formal_eligible": self.qualification.formal_eligible,
            "qualification_gate_hash": self.qualification.qualification_gate_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def qualify(
        self,
        *,
        execution: FormalExecutionEvidence,
        gate: GateRecord,
        artifact_ref: str | None = None,
    ) -> "CapturedEndpoint":
        execution.require_for_stage(3)
        if self.scope != "formal" or execution.artifact_hash != self.execution_evidence_hash:
            raise FormalRunRejected("ENDPOINT_FORMAL_EXECUTION_EVIDENCE_MISMATCH")
        accepted = require_accepted_gate(gate, stage=3)
        _require_gate_binding(accepted, artifact_ref)
        return replace(
            self,
            qualification=ArtifactQualification.from_gate(
                scope="formal", gate=accepted, stage=3
            ),
        )


class EndpointCaptureCoordinator:
    """强制执行 pre/post/attempt-commit/replay 顺序并在失败时回滚。"""

    def capture(
        self,
        request: EndpointCaptureRequest,
        adapter: EndpointCaptureAdapter,
    ) -> CapturedEndpoint:
        if request.execution.run_intent == "formal":
            request.execution.require_for_stage(3)
        pre = adapter.capture_pre_state()
        try:
            adapter.apply_optimizer_update()
            parameter_post = adapter.capture_parameter_post_state()
            adapter.advance_attempt_commit()
            attempt_commit = adapter.capture_attempt_commit_state()
            provisional = EndpointRecord(
                path_state_id=request.path_state_id,
                source_run_id=request.source_run_id,
                optimizer_step=request.optimizer_step,
                parameter_registry_hash=request.parameter_registry_hash,
                pre_state=pre,
                parameter_post_state=parameter_post,
                attempt_commit_state=attempt_commit,
                attempt_commit_parent_hash=parameter_post.artifact_hash,
                probe_buffer_snapshot_hash=adapter.probe_buffer_snapshot_hash(),
                full_update_delta_hash=adapter.full_update_delta_hash(),
                update_sample_ids=request.update_sample_ids,
                replay_verified=False,
                metadata=request.metadata,
            )
            if adapter.verify_replay(provisional) is not True:
                raise RuntimeError("ENDPOINT_REPLAY_VERIFICATION_FAILED")
            record = replace(provisional, replay_verified=True)
        except BaseException:
            adapter.restore_pre_state()
            raise
        return CapturedEndpoint(
            record=record,
            execution_evidence_hash=request.execution.artifact_hash,
            qualification=ArtifactQualification.candidate(request.execution.run_intent),
        )


@dataclass(frozen=True, slots=True)
class ProbePanelEntry:
    role: str
    probe: ProbeSpec

    def __post_init__(self) -> None:
        if self.role not in {"pilot", "formal", "replay"}:
            raise ValueError("probe role 只能是 pilot/formal/replay")

    def to_dict(self) -> dict[str, object]:
        for sample_id in self.probe.sample_ids:
            if isinstance(sample_id, bool) or not isinstance(sample_id, (str, int)):
                raise TypeError("probe artifact 的 sample ID 只能是字符串或整数")
        return {
            "role": self.role,
            "probe_id": self.probe.probe_id,
            "sample_ids": list(self.probe.sample_ids),
            "content_hash": self.probe.content_hash,
            "loss_contract_hash": self.probe.loss_contract_hash,
            "effective_weight_unit": self.probe.effective_weight_unit,
            "metadata": thaw_json_value(self.probe.metadata),
            "probe_digest": self.probe.digest,
        }


@dataclass(frozen=True, slots=True)
class ProbePanel:
    """与一个端点绑定、彼此互斥的冻结 probe 集合。"""

    panel_id: str
    endpoint_digest: str
    entries: tuple[ProbePanelEntry, ...]
    execution_evidence_hash: str
    qualification: ArtifactQualification
    minimum_formal_probes: int = 3
    schema_version: str = "stage3-probe-panel-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.panel_id, field_name="panel_id")
        _require_sha256(self.endpoint_digest, field_name="endpoint_digest")
        _require_sha256(self.execution_evidence_hash, field_name="execution_evidence_hash")
        if not self.entries:
            raise ValueError("ProbePanel.entries 不能为空")
        ids = tuple(entry.probe.probe_id for entry in self.entries)
        if len(set(ids)) != len(ids):
            raise ValueError("ProbePanel probe_id 不能重复")
        losses = {entry.probe.loss_contract_hash for entry in self.entries}
        if len(losses) != 1:
            raise ValueError("同一 ProbePanel 必须共享唯一 loss contract")
        seen: set[Hashable] = set()
        for entry in self.entries:
            overlap = seen.intersection(entry.probe.sample_ids)
            if overlap:
                raise ValueError("ProbePanel 内不同 probe 发生统计单元重叠")
            seen.update(entry.probe.sample_ids)
        if self.minimum_formal_probes <= 0:
            raise ValueError("minimum_formal_probes 必须为正")
        formal_count = sum(entry.role == "formal" for entry in self.entries)
        if self.qualification.scope == "formal" and formal_count < self.minimum_formal_probes:
            raise FormalRunRejected("FORMAL_PROBE_PANEL_COUNT_BELOW_FROZEN_MINIMUM")

    @classmethod
    def build(
        cls,
        *,
        panel_id: str,
        endpoint: EndpointRecord,
        entries: Sequence[ProbePanelEntry],
        execution: FormalExecutionEvidence | None = None,
        minimum_formal_probes: int = 3,
    ) -> "ProbePanel":
        execution = execution or FormalExecutionEvidence("local_fixture")
        if execution.run_intent == "formal":
            execution.require_for_stage(3)
        for entry in entries:
            entry.probe.assert_independent_from(endpoint)
        return cls(
            panel_id=panel_id,
            endpoint_digest=endpoint.digest,
            entries=tuple(entries),
            execution_evidence_hash=execution.artifact_hash,
            qualification=ArtifactQualification.candidate(execution.run_intent),
            minimum_formal_probes=minimum_formal_probes,
        )

    @property
    def scope(self) -> str:
        return self.qualification.scope

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "panel_id": self.panel_id,
            "endpoint_digest": self.endpoint_digest,
            "entries": [entry.to_dict() for entry in self.entries],
            "minimum_formal_probes": self.minimum_formal_probes,
            "execution_evidence_hash": self.execution_evidence_hash,
            "scope": self.scope,
            "formal_eligible": self.qualification.formal_eligible,
            "qualification_gate_hash": self.qualification.qualification_gate_hash,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def qualify(
        self,
        *,
        execution: FormalExecutionEvidence,
        gate: GateRecord,
        artifact_ref: str | None = None,
    ) -> "ProbePanel":
        execution.require_for_stage(3)
        if self.scope != "formal" or execution.artifact_hash != self.execution_evidence_hash:
            raise FormalRunRejected("PROBE_PANEL_EXECUTION_EVIDENCE_MISMATCH")
        accepted = require_accepted_gate(gate, stage=3)
        _require_gate_binding(accepted, artifact_ref)
        return replace(
            self,
            qualification=ArtifactQualification.from_gate(
                scope="formal", gate=accepted, stage=3
            ),
        )


class NodeValueCodec(Protocol):
    """持久节点缓存的安全值 codec；实现不得使用 pickle。"""

    @property
    def codec_id(self) -> str:
        """返回会进入 commit 的稳定 codec 身份。"""

    def encode(self, value: object) -> object:
        """转换为 tensor bundle 支持的 primitive/tensor 状态树。"""

    def decode(self, value: object) -> object:
        """从安全状态树恢复调用方梯度对象。"""


def _clone_tree(value: object) -> object:
    # ``TensorMap`` 实现了 ``Mapping``；必须在普通 mapping 分支之前识别，
    # 否则防御性复制会悄悄把有序坐标容器降级成 ``dict``，丢失 registry 绑定。
    try:
        from param_importance_nlp.core.tensors import TensorMap
    except ImportError:  # pragma: no cover - 极简环境
        TensorMap = None  # type: ignore[assignment,misc]
    if TensorMap is not None and isinstance(value, TensorMap):
        return value.clone(detach=True)
    try:
        import torch
    except ImportError:  # pragma: no cover - 极简环境
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if isinstance(value, Mapping):
        return {str(key): _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return copy.deepcopy(value)


class SafeTensorTreeCodec:
    """支持 primitive/tensor tree，并可选重建绑定 registry 的 TensorMap。"""

    def __init__(self, *, registry: object | None = None) -> None:
        self.registry = registry
        registry_hash = getattr(registry, "coordinate_registry_hash", None)
        self._codec_id = f"safe-tensor-tree-v1:{registry_hash or 'unbound'}"

    @property
    def codec_id(self) -> str:
        return self._codec_id

    def encode(self, value: object) -> object:
        try:
            from param_importance_nlp.core.tensors import TensorMap
        except ImportError:  # pragma: no cover
            TensorMap = None  # type: ignore[assignment,misc]
        if TensorMap is not None and isinstance(value, TensorMap):
            return {
                "__node_value_type__": "tensor_map",
                "registry_hash": value.registry_hash,
                "values": value.to_dict(clone=True),
            }
        if isinstance(value, Mapping):
            if "__node_value_type__" in value:
                raise ValueError("普通 mapping 不得使用保留键 __node_value_type__")
            if any(not isinstance(key, str) for key in value):
                raise TypeError("NODE_CACHE_MAPPING_KEYS_MUST_BE_STRINGS")
            return {key: self.encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.encode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.encode(item) for item in value)
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None  # type: ignore[assignment]
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().clone()
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise TypeError(f"NODE_CACHE_UNSUPPORTED_VALUE:{type(value).__qualname__}")

    def decode(self, value: object) -> object:
        if isinstance(value, Mapping) and value.get("__node_value_type__") == "tensor_map":
            registry_hash = value.get("registry_hash")
            values = value.get("values")
            if not isinstance(values, Mapping):
                raise ValueError("NODE_CACHE_TENSOR_MAP_VALUES_INVALID")
            if registry_hash is not None:
                actual = getattr(self.registry, "coordinate_registry_hash", None)
                if actual != registry_hash:
                    raise FormalRunRejected("NODE_CACHE_REGISTRY_RESOLVER_REQUIRED")
            from param_importance_nlp.core.tensors import TensorMap

            return TensorMap(values, registry=self.registry)  # type: ignore[arg-type]
        if isinstance(value, Mapping):
            return {str(key): self.decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.decode(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.decode(item) for item in value)
        return _clone_tree(value)


def _tree_digest(value: object) -> str:
    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if hasattr(item, "detach"):
            item = item.detach()  # type: ignore[union-attr]
        if hasattr(item, "cpu"):
            item = item.cpu()  # type: ignore[union-attr]
        if hasattr(item, "numpy"):
            item = item.numpy()  # type: ignore[union-attr]
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            if array.dtype.hasobject:
                raise TypeError("NODE_CACHE_OBJECT_DTYPE_FORBIDDEN")
            digest.update(b"array\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(canonical_json_hash(list(array.shape)).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                if not isinstance(key, str):
                    raise TypeError("NODE_CACHE_MAPPING_KEYS_MUST_BE_STRINGS")
                digest.update(key.encode("utf-8"))
                digest.update(b"\0")
                visit(item[key])
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
            return
        if item is None or isinstance(item, (bool, int, str)):
            digest.update(canonical_json_hash(item).encode("ascii"))
            return
        if isinstance(item, float) and math.isfinite(item):
            digest.update(item.hex().encode("ascii"))
            return
        raise TypeError(f"NODE_CACHE_DIGEST_UNSUPPORTED:{type(item).__qualname__}")

    visit(value)
    return digest.hexdigest()


def _key_payload(key: NodeCacheKey) -> dict[str, object]:
    return {
        "path_unit_id": key.path_unit_id,
        "alpha_hex": key.alpha.hex(),
        "precision": key.precision,
        "parameter_registry_hash": key.parameter_registry_hash,
        "loss_contract_hash": key.loss_contract_hash,
    }


def _key_from_payload(value: Mapping[str, object]) -> NodeCacheKey:
    return NodeCacheKey(
        path_unit_id=str(value["path_unit_id"]),
        alpha=float.fromhex(str(value["alpha_hex"])),
        precision=str(value["precision"]),
        parameter_registry_hash=str(value["parameter_registry_hash"]),
        loss_contract_hash=str(value["loss_contract_hash"]),
    )


class PersistentNodeGradientCache:
    """批次原子提交、可跨进程恢复且不允许覆盖的安全节点缓存。"""

    def __init__(
        self,
        root: str | Path,
        *,
        codec: NodeValueCodec | None = None,
    ) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self.codec = codec or SafeTensorTreeCodec()
        self._index: dict[str, tuple[NodeCacheKey, str, str]] = {}
        self._reload_index()

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, NodeCacheKey) and key.digest in self._index

    def _validated_commit(self, path: Path) -> tuple[Mapping[str, object], Mapping[str, object]]:
        commit = load_canonical_json(path)
        if not isinstance(commit, Mapping):
            raise ValueError("NODE_CACHE_COMMIT_NOT_OBJECT")
        payload = {name: item for name, item in commit.items() if name != "artifact_hash"}
        if canonical_json_hash(payload) != commit.get("artifact_hash"):
            raise ValueError("NODE_CACHE_COMMIT_HASH_MISMATCH")
        if commit.get("schema_version") != "stage3-node-cache-commit-v1":
            raise ValueError("NODE_CACHE_COMMIT_SCHEMA_MISMATCH")
        if commit.get("codec_id") != self.codec.codec_id:
            raise ValueError("NODE_CACHE_CODEC_ID_MISMATCH")
        relative = Path(str(commit["object_ref"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("NODE_CACHE_OBJECT_PATH_ESCAPE")
        state, bundle = load_tensor_bundle(self.root / relative)
        if bundle.manifest_sha256 != commit["object_manifest_hash"]:
            raise ValueError("NODE_CACHE_OBJECT_MANIFEST_HASH_MISMATCH")
        if not isinstance(state, Mapping) or state.get("codec_id") != self.codec.codec_id:
            raise ValueError("NODE_CACHE_OBJECT_STATE_INVALID")
        return commit, state

    def _reload_index(self) -> None:
        index: dict[str, tuple[NodeCacheKey, str, str]] = {}
        for path in sorted(self.commits.glob("*.json")):
            commit, state = self._validated_commit(path)
            entries = commit.get("entries")
            values = state.get("values")
            if not isinstance(entries, list) or not isinstance(values, Mapping):
                raise ValueError("NODE_CACHE_ENTRIES_INVALID")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("NODE_CACHE_ENTRY_NOT_OBJECT")
                key = _key_from_payload(entry["key"])  # type: ignore[arg-type]
                key_digest = str(entry["key_digest"])
                value_digest = str(entry["value_digest"])
                if key.digest != key_digest:
                    raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
                encoded = values.get(key_digest)
                if encoded is None or _tree_digest(encoded) != value_digest:
                    raise ValueError("NODE_CACHE_VALUE_DIGEST_MISMATCH")
                previous = index.get(key_digest)
                current = (key, str(commit["object_ref"]), value_digest)
                if previous is not None and previous[2] != value_digest:
                    raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
                index[key_digest] = current
        self._index = index

    def get(self, key: NodeCacheKey) -> object:
        if not isinstance(key, NodeCacheKey):
            raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
        indexed = self._index.get(key.digest)
        if indexed is None:
            raise KeyError(key)
        _stored_key, object_ref, value_digest = indexed
        state, _bundle = load_tensor_bundle(self.root / object_ref)
        assert isinstance(state, Mapping) and isinstance(state["values"], Mapping)
        encoded = state["values"][key.digest]
        if _tree_digest(encoded) != value_digest:
            raise ValueError("NODE_CACHE_VALUE_CHANGED_AFTER_INDEX")
        return _clone_tree(self.codec.decode(encoded))

    def publish_many(self, entries: Mapping[NodeCacheKey, object]) -> int:
        if not entries:
            return 0
        prepared: list[tuple[NodeCacheKey, object, str]] = []
        for key, value in entries.items():
            if not isinstance(key, NodeCacheKey):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            encoded = self.codec.encode(value)
            value_digest = _tree_digest(encoded)
            existing = self._index.get(key.digest)
            if existing is not None:
                if existing[2] != value_digest:
                    raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
                continue
            prepared.append((key, encoded, value_digest))
        if not prepared:
            return 0
        prepared.sort(key=lambda item: item[0].digest)
        lock_path = self.root / "writer.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("STAGE3_NODE_CACHE_WRITER_ALREADY_ACTIVE") from error
        try:
            self._reload_index()
            still_new: list[tuple[NodeCacheKey, object, str]] = []
            for key, _value, digest in prepared:
                existing = self._index.get(key.digest)
                if existing is not None:
                    if existing[2] != digest:
                        raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
                    continue
                still_new.append((key, _value, digest))
            prepared = still_new
            if not prepared:
                return 0
            identity = {
                "codec_id": self.codec.codec_id,
                "entries": [
                    {"key_digest": key.digest, "value_digest": digest}
                    for key, _value, digest in prepared
                ],
            }
            batch_digest = canonical_json_hash(identity)
            object_path = self.objects / batch_digest
            state = {
                "schema_version": "stage3-node-cache-object-v1",
                "codec_id": self.codec.codec_id,
                "values": {key.digest: value for key, value, _digest in prepared},
            }
            if not object_path.exists():
                bundle = publish_tensor_bundle(object_path, state)
            else:
                restored, bundle = load_tensor_bundle(object_path)
                if not isinstance(restored, Mapping) or restored.get("codec_id") != self.codec.codec_id:
                    raise ValueError("NODE_CACHE_EXISTING_OBJECT_MISMATCH")
            commit: dict[str, object] = {
                "schema_version": "stage3-node-cache-commit-v1",
                "batch_digest": batch_digest,
                "codec_id": self.codec.codec_id,
                "entries": [
                    {
                        "key": _key_payload(key),
                        "key_digest": key.digest,
                        "value_digest": digest,
                    }
                    for key, _value, digest in prepared
                ],
                "object_ref": f"objects/{batch_digest}",
                "object_manifest_hash": bundle.manifest_sha256,
            }
            commit["artifact_hash"] = canonical_json_hash(commit)
            commit_path = self.commits / f"{batch_digest}.json"
            if commit_path.exists():
                if load_canonical_json(commit_path) != commit:
                    raise ValueError("NODE_CACHE_COMMIT_CONFLICT")
            else:
                write_canonical_json(commit_path, commit)
            self._reload_index()
        finally:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)
        return len(prepared)

    def keys(self) -> tuple[NodeCacheKey, ...]:
        return tuple(self._index[digest][0] for digest in sorted(self._index))

    def reconcile(self) -> dict[str, object]:
        self._reload_index()
        referenced = {
            str(load_canonical_json(path)["batch_digest"])  # type: ignore[index]
            for path in self.commits.glob("*.json")
        }
        objects = {path.name for path in self.objects.iterdir() if path.is_dir()}
        return {
            "committed_key_digests": sorted(self._index),
            "orphan_objects": sorted(objects - referenced),
        }

    def commit_evidence(
        self,
        keys: Sequence[NodeCacheKey],
    ) -> dict[str, object]:
        """返回一组节点键的确定性两阶段提交与恢复证据。

        证据只描述调用方明确请求的节点键，以及缓存目录当前经过严格校验的权威
        commit；它不记录“本次运行命中了几次”之类依赖恢复时机的诊断值。这样 fresh
        run 与 crash-resume 在最终节点集合相同时会得到同一个证据 hash，同时仍能
        证明每个节点都由安全 TensorBundle 对象和独立 commit 共同授权。
        """

        requested: dict[str, NodeCacheKey] = {}
        for key in keys:
            if not isinstance(key, NodeCacheKey):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            requested[key.digest] = key
        requested_digests = sorted(requested)
        reconciliation = self.reconcile()
        committed = set(reconciliation["committed_key_digests"])  # type: ignore[arg-type]
        missing = sorted(set(requested_digests) - committed)

        authoritative_commits: list[dict[str, object]] = []
        for path in sorted(self.commits.glob("*.json")):
            commit, _state = self._validated_commit(path)
            entries = commit.get("entries")
            if not isinstance(entries, list):  # _reload_index 已验证；保留窄化防线
                raise ValueError("NODE_CACHE_ENTRIES_INVALID")
            entry_digests = {
                str(entry["key_digest"])
                for entry in entries
                if isinstance(entry, Mapping)
            }
            matched = sorted(entry_digests.intersection(requested_digests))
            if not matched:
                continue
            authoritative_commits.append(
                {
                    "commit_ref": path.relative_to(self.root).as_posix(),
                    "commit_artifact_hash": commit["artifact_hash"],
                    "object_ref": commit["object_ref"],
                    "object_manifest_hash": commit["object_manifest_hash"],
                    "key_digests": matched,
                }
            )

        payload: dict[str, object] = {
            "schema_version": "stage3-node-cache-evidence-v1",
            "codec_id": self.codec.codec_id,
            "publication_protocol": (
                "immutable_tensor_bundle_then_independent_authoritative_commit"
            ),
            "requested_key_digests": requested_digests,
            "all_requested_keys_committed": not missing,
            "missing_key_digests": missing,
            "authoritative_commits": authoritative_commits,
            "reconciliation": reconciliation,
        }
        payload["evidence_hash"] = canonical_json_hash(payload)
        return payload


@dataclass(frozen=True, slots=True)
class ReferenceRuleLevel:
    family: str
    level: int
    rule: object

    def __post_init__(self) -> None:
        _require_identifier(self.family, field_name="reference family")
        if self.level < 0:
            raise ValueError("reference level 不能为负")
        _rule_hash(self.rule)

    @property
    def unique_nodes(self) -> int:
        value = getattr(self.rule, "unique_gradient_evaluations", None)
        if value is None:
            value = getattr(self.rule, "node_count", None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("reference rule 必须声明正的 unique node 数")
        return value


def _rule_hash(rule: object) -> str:
    value = getattr(rule, "artifact_hash", None)
    if not isinstance(value, str):
        raise TypeError("reference rule 必须暴露 artifact_hash")
    return _require_sha256(value, field_name="rule.artifact_hash")


def _extract_path_evaluation(result: object) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if isinstance(result, Mapping):
        contribution = _as_vector(result, field_name="reference contribution")
        return contribution, {
            "completeness_absolute_residual": None,
            "completeness_relative_residual": None,
            "completeness_l1_scaled_residual": None,
        }
    contribution = getattr(result, "signed", None)
    if contribution is None:
        contribution = getattr(result, "contributions", None)
    if contribution is None:
        raise TypeError("reference evaluator 必须返回 contribution mapping 或 PathIntegralResult")
    metrics: dict[str, object] = {}
    for name in (
        "completeness_absolute_residual",
        "completeness_relative_residual",
        "completeness_l1_scaled_residual",
        "endpoint_loss_pre",
        "endpoint_loss_post",
        "loss_drop",
        "unique_gradient_evaluations",
    ):
        value = getattr(result, name, None)
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{name} 必须有限或 null")
        metrics[name] = value
    return _as_vector(contribution), metrics


class _RefinementStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)

    def _commit_id(self, unit_id: str, family: str, level: int) -> str:
        return canonical_json_hash({"unit_id": unit_id, "family": family, "level": level})

    def publish(
        self,
        *,
        unit_id: str,
        level: ReferenceRuleLevel,
        contribution: Mapping[str, object],
        metrics: Mapping[str, object],
    ) -> None:
        lock_path = self.root / "writer.lock"
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError("STAGE3_REFERENCE_WRITER_ALREADY_ACTIVE") from error
        try:
            self._publish_unlocked(
                unit_id=unit_id,
                level=level,
                contribution=contribution,
                metrics=metrics,
            )
        finally:
            os.close(lock_fd)
            lock_path.unlink(missing_ok=True)

    def _publish_unlocked(
        self,
        *,
        unit_id: str,
        level: ReferenceRuleLevel,
        contribution: Mapping[str, object],
        metrics: Mapping[str, object],
    ) -> None:
        rule_hash = _rule_hash(level.rule)
        vector_hash = _vector_digest(contribution)
        object_id = canonical_json_hash(
            {
                "unit_id": unit_id,
                "family": level.family,
                "level": level.level,
                "rule_hash": rule_hash,
                "vector_hash": vector_hash,
                "metrics": dict(metrics),
            }
        )
        object_path = self.objects / object_id
        state = {
            "schema_version": "stage3-reference-level-object-v1",
            "contribution": {name: np.array(value, copy=True) for name, value in _as_vector(contribution).items()},
            "metrics": dict(metrics),
        }
        if not object_path.exists():
            bundle = publish_tensor_bundle(object_path, state)
        else:
            restored, bundle = load_tensor_bundle(object_path)
            if not isinstance(restored, Mapping) or _vector_digest(
                restored["contribution"]  # type: ignore[arg-type]
            ) != vector_hash:
                raise ValueError("REFERENCE_LEVEL_EXISTING_OBJECT_MISMATCH")
        commit_id = self._commit_id(unit_id, level.family, level.level)
        commit: dict[str, object] = {
            "schema_version": "stage3-reference-level-commit-v1",
            "commit_id": commit_id,
            "unit_id": unit_id,
            "family": level.family,
            "level": level.level,
            "unique_nodes": level.unique_nodes,
            "rule_hash": rule_hash,
            "contribution_hash": vector_hash,
            "object_ref": f"objects/{object_id}",
            "object_manifest_hash": bundle.manifest_sha256,
        }
        commit["artifact_hash"] = canonical_json_hash(commit)
        path = self.commits / f"{commit_id}.json"
        if path.exists():
            if load_canonical_json(path) != commit:
                raise ValueError("REFERENCE_LEVEL_COMMIT_CONFLICT")
        else:
            write_canonical_json(path, commit)

    def load_all(self, *, unit_id: str) -> dict[tuple[str, int], dict[str, object]]:
        results: dict[tuple[str, int], dict[str, object]] = {}
        for path in sorted(self.commits.glob("*.json")):
            commit = load_canonical_json(path)
            if not isinstance(commit, Mapping):
                raise ValueError("REFERENCE_LEVEL_COMMIT_NOT_OBJECT")
            payload = {name: item for name, item in commit.items() if name != "artifact_hash"}
            if canonical_json_hash(payload) != commit.get("artifact_hash"):
                raise ValueError("REFERENCE_LEVEL_COMMIT_HASH_MISMATCH")
            if commit.get("unit_id") != unit_id:
                continue
            relative = Path(str(commit["object_ref"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("REFERENCE_LEVEL_OBJECT_PATH_ESCAPE")
            state, bundle = load_tensor_bundle(self.root / relative)
            if bundle.manifest_sha256 != commit["object_manifest_hash"]:
                raise ValueError("REFERENCE_LEVEL_MANIFEST_HASH_MISMATCH")
            if not isinstance(state, Mapping):
                raise ValueError("REFERENCE_LEVEL_STATE_NOT_OBJECT")
            contribution = state["contribution"]
            if _vector_digest(contribution) != commit["contribution_hash"]:  # type: ignore[arg-type]
                raise ValueError("REFERENCE_LEVEL_VECTOR_HASH_MISMATCH")
            key = (str(commit["family"]), int(commit["level"]))
            results[key] = {
                "family": key[0],
                "level": key[1],
                "unique_nodes": int(commit["unique_nodes"]),
                "rule_hash": str(commit["rule_hash"]),
                "contribution": contribution,
                "metrics": state["metrics"],
            }
        return results


@dataclass(frozen=True, slots=True)
class ReferenceRefinementResult:
    unit_id: str
    converged: bool
    convergence_defined: bool
    status: str
    primary_family: str
    selected_level: int | None
    selected_rule_hash: str | None
    conservative_error: float | None
    within_family_errors: Mapping[str, float]
    cross_family_error: float | None
    completed_levels: tuple[Mapping[str, object], ...]
    reference_contribution: Mapping[str, object]
    scope: str
    execution_evidence_hash: str
    reasons: tuple[str, ...] = ()
    schema_version: str = "stage3-reference-refinement-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.unit_id, field_name="unit_id")
        _require_identifier(self.primary_family, field_name="primary_family")
        _require_sha256(self.execution_evidence_hash, field_name="execution_evidence_hash")
        if self.status not in {"FIXTURE_CONVERGED", "FORMAL_CANDIDATE", "REFERENCE_UNRESOLVED"}:
            raise ValueError("ReferenceRefinementResult status 不受支持")
        if type(self.convergence_defined) is not bool:
            raise TypeError("convergence_defined 必须是显式 bool")
        if self.converged and not self.convergence_defined:
            raise ValueError("未定义的 convergence 不能标记为 converged")
        if self.converged != (self.selected_rule_hash is not None):
            raise ValueError("converged 与 selected rule 不一致")
        if self.selected_rule_hash is not None:
            _require_sha256(self.selected_rule_hash, field_name="selected_rule_hash")
        if not self.converged and not self.reasons:
            raise ValueError("未收敛 reference 必须给出至少一个 reason")
        if any(not isinstance(reason, str) or not reason for reason in self.reasons):
            raise TypeError("reasons 必须是非空字符串")
        object.__setattr__(
            self,
            "reference_contribution",
            MappingProxyType(_as_vector(self.reference_contribution)),
        )

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "unit_id": self.unit_id,
            "converged": self.converged,
            "convergence_defined": self.convergence_defined,
            "status": self.status,
            "primary_family": self.primary_family,
            "selected_level": self.selected_level,
            "selected_rule_hash": self.selected_rule_hash,
            "conservative_error": self.conservative_error,
            "within_family_errors": dict(sorted(self.within_family_errors.items())),
            "cross_family_error": self.cross_family_error,
            "completed_levels": [dict(item) for item in self.completed_levels],
            "reference_contribution_hash": _vector_digest(self.reference_contribution),
            "scope": self.scope,
            "formal_eligible": False,
            "qualification_gate_hash": None,
            "execution_evidence_hash": self.execution_evidence_hash,
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}


class ReferenceRefinementRunner:
    """跨两个独立规则家族连续加密，并在 level commit 边界恢复。"""

    def run(
        self,
        *,
        unit_id: str,
        levels: Sequence[ReferenceRuleLevel],
        evaluator: Callable[[object], object],
        artifact_root: str | Path,
        tolerance: float,
        required_consecutive: int = 2,
        primary_family: str | None = None,
        execution: FormalExecutionEvidence | None = None,
        max_new_evaluations: int | None = None,
    ) -> ReferenceRefinementResult:
        _require_identifier(unit_id, field_name="unit_id")
        execution = execution or FormalExecutionEvidence("local_fixture")
        if execution.run_intent == "formal":
            execution.require_for_stage(3)
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance 必须是有限正数")
        if required_consecutive <= 0:
            raise ValueError("required_consecutive 必须为正")
        if max_new_evaluations is not None and max_new_evaluations <= 0:
            raise ValueError("max_new_evaluations 必须为正或 null")
        grouped: dict[str, list[ReferenceRuleLevel]] = {}
        for item in levels:
            grouped.setdefault(item.family, []).append(item)
        if len(grouped) != 2:
            raise ValueError("reference refinement 必须恰好使用两个独立规则家族")
        for family_levels in grouped.values():
            family_levels.sort(key=lambda item: item.level)
            if len(family_levels) < 2:
                raise ValueError("每个 reference family 至少需要两个 level")
            if tuple(item.level for item in family_levels) != tuple(
                range(family_levels[0].level, family_levels[0].level + len(family_levels))
            ):
                raise ValueError("reference family level 必须连续")
            nodes = tuple(item.unique_nodes for item in family_levels)
            if any(right <= left for left, right in zip(nodes, nodes[1:])):
                raise ValueError("reference family unique nodes 必须严格递增")
        families = tuple(sorted(grouped))
        primary = primary_family or families[0]
        if primary not in grouped:
            raise ValueError("primary_family 不存在")

        store = _RefinementStore(artifact_root)
        completed = store.load_all(unit_id=unit_id)
        new_count = 0
        max_rounds = min(len(grouped[family]) for family in families)
        # 逐 round 执行并立刻检查停止规则。若先把整条 ladder 全部跑完再分析，
        # 虽然数值正确，却会违背 reference sizing 的节点预算语义，也使恢复后的
        # 已收敛结果仍继续产生昂贵梯度。
        evaluation_streak = 0
        stop_after_round: int | None = None
        for round_index in range(max_rounds):
            for family in families:
                level = grouped[family][round_index]
                key = (family, level.level)
                if key in completed:
                    if completed[key]["rule_hash"] != _rule_hash(level.rule):
                        raise ValueError("REFERENCE_RESUME_RULE_HASH_MISMATCH")
                    continue
                if max_new_evaluations is not None and new_count >= max_new_evaluations:
                    break
                contribution, metrics = _extract_path_evaluation(evaluator(level.rule))
                store.publish(
                    unit_id=unit_id,
                    level=level,
                    contribution=contribution,
                    metrics=metrics,
                )
                completed = store.load_all(unit_id=unit_id)
                new_count += 1
            round_keys = [
                (family, grouped[family][round_index].level) for family in families
            ]
            budget_exhausted = (
                max_new_evaluations is not None
                and new_count >= max_new_evaluations
            )
            if any(key not in completed for key in round_keys):
                if budget_exhausted:
                    break
                continue
            if round_index == 0:
                if budget_exhausted:
                    break
                continue
            round_errors: list[float] = []
            for family in families:
                previous = completed[
                    (family, grouped[family][round_index - 1].level)
                ]
                current = completed[(family, grouped[family][round_index].level)]
                round_errors.append(
                    _normalized_l1(
                        previous["contribution"],  # type: ignore[arg-type]
                        current["contribution"],  # type: ignore[arg-type]
                    )
                )
            secondary = next(family for family in families if family != primary)
            primary_row = completed[(primary, grouped[primary][round_index].level)]
            secondary_row = completed[
                (secondary, grouped[secondary][round_index].level)
            ]
            round_errors.append(
                _normalized_l1(
                    secondary_row["contribution"],  # type: ignore[arg-type]
                    primary_row["contribution"],  # type: ignore[arg-type]
                )
            )
            round_conservative = max(round_errors)
            evaluation_streak = (
                evaluation_streak + 1
                if math.isfinite(round_conservative)
                and round_conservative <= tolerance
                else 0
            )
            if evaluation_streak >= required_consecutive:
                stop_after_round = round_index
                break
            if budget_exhausted:
                break

        within: dict[str, float] = {}
        reasons: list[str] = []
        cross: float | None = None
        conservative: float | None = None
        streak = 0
        selected_level: int | None = None
        selected_rule_hash: str | None = None
        selected_vector: Mapping[str, object] | None = None
        complete_rows: list[Mapping[str, object]] = []
        analysis_rounds = (
            max_rounds if stop_after_round is None else stop_after_round + 1
        )
        for round_index in range(analysis_rounds):
            keys = [(family, grouped[family][round_index].level) for family in families]
            if any(key not in completed for key in keys):
                break
            for key in keys:
                row = completed[key]
                complete_rows.append(
                    {
                        "family": row["family"],
                        "level": row["level"],
                        "unique_nodes": row["unique_nodes"],
                        "rule_hash": row["rule_hash"],
                        "contribution_hash": _vector_digest(row["contribution"]),  # type: ignore[arg-type]
                    }
                )
            if round_index == 0:
                continue
            round_errors: list[float] = []
            for family in families:
                previous = completed[(family, grouped[family][round_index - 1].level)]
                current = completed[(family, grouped[family][round_index].level)]
                observed_error = _normalized_l1(
                    previous["contribution"], current["contribution"]  # type: ignore[arg-type]
                )
                if math.isfinite(observed_error):
                    within[family] = observed_error
                    round_errors.append(observed_error)
                else:
                    reasons.append(f"{family}:zero_reference_l1_norm")
            primary_row = completed[(primary, grouped[primary][round_index].level)]
            secondary = next(family for family in families if family != primary)
            secondary_row = completed[(secondary, grouped[secondary][round_index].level)]
            observed_cross = _normalized_l1(
                secondary_row["contribution"], primary_row["contribution"]  # type: ignore[arg-type]
            )
            if math.isfinite(observed_cross):
                cross = observed_cross
            else:
                cross = None
                reasons.append("cross_family:zero_reference_l1_norm")
            conservative = (
                max((*round_errors, cross))
                if cross is not None and len(round_errors) == len(families)
                else None
            )
            streak = (
                streak + 1
                if conservative is not None and conservative <= tolerance
                else 0
            )
            if streak >= required_consecutive:
                selected_level = int(primary_row["level"])
                selected_rule_hash = str(primary_row["rule_hash"])
                selected_vector = primary_row["contribution"]  # type: ignore[assignment]
                break
        if selected_vector is None:
            # 返回最高已完成 primary level 便于诊断，但状态明确 unresolved。
            primary_rows = [
                row for (family, _level), row in completed.items() if family == primary
            ]
            if not primary_rows:
                raise RuntimeError("REFERENCE_REFINEMENT_NO_PRIMARY_RESULT")
            primary_rows.sort(key=lambda row: int(row["level"]))
            selected_vector = primary_rows[-1]["contribution"]  # type: ignore[assignment]
        converged = selected_rule_hash is not None
        status = "REFERENCE_UNRESOLVED"
        if converged:
            status = (
                "FIXTURE_CONVERGED"
                if execution.run_intent == "local_fixture"
                else "FORMAL_CANDIDATE"
            )
        elif not reasons:
            reasons.append("tolerance_not_met_or_budget_exhausted")
        return ReferenceRefinementResult(
            unit_id=unit_id,
            converged=converged,
            convergence_defined=conservative is not None,
            status=status,
            primary_family=primary,
            selected_level=selected_level,
            selected_rule_hash=selected_rule_hash,
            conservative_error=conservative,
            within_family_errors=MappingProxyType(dict(within)),
            cross_family_error=cross,
            completed_levels=tuple(complete_rows),
            reference_contribution=selected_vector,
            scope=execution.run_intent,
            execution_evidence_hash=execution.artifact_hash,
            reasons=tuple(sorted(set(reasons))),
        )


@dataclass(frozen=True, slots=True)
class QuadratureThresholds:
    max_normalized_l1_error: float
    max_completeness_absolute_residual: float
    min_spearman: float
    min_topk_overlap: float
    max_unique_nodes: int

    def __post_init__(self) -> None:
        for name in ("max_normalized_l1_error", "max_completeness_absolute_residual"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是非负有限数")
        for name in ("min_spearman", "min_topk_overlap"):
            value = float(getattr(self, name))
            lower = -1.0 if name == "min_spearman" else 0.0
            if not math.isfinite(value) or not lower <= value <= 1:
                raise ValueError(f"{name} 必须位于 [{lower:g},1]")
        if (
            isinstance(self.max_unique_nodes, bool)
            or not isinstance(self.max_unique_nodes, int)
            or self.max_unique_nodes <= 0
        ):
            raise ValueError("max_unique_nodes 必须是正整数")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_normalized_l1_error": self.max_normalized_l1_error,
            "max_completeness_absolute_residual": self.max_completeness_absolute_residual,
            "min_spearman": self.min_spearman,
            "min_topk_overlap": self.min_topk_overlap,
            "max_unique_nodes": self.max_unique_nodes,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.to_dict())


@dataclass(frozen=True, slots=True)
class QuadratureObservation:
    unit_id: str
    rule_name: str
    unique_nodes: int
    normalized_l1_error: float
    completeness_absolute_residual: float
    spearman: float
    topk_overlap: float
    wall_seconds: float

    def __post_init__(self) -> None:
        _require_identifier(self.unit_id, field_name="unit_id")
        _require_identifier(self.rule_name, field_name="rule_name")
        if self.unique_nodes <= 0:
            raise ValueError("unique_nodes 必须为正")
        numeric = (
            self.normalized_l1_error,
            self.completeness_absolute_residual,
            self.spearman,
            self.topk_overlap,
            self.wall_seconds,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("QuadratureObservation 指标必须全部有限")
        if self.normalized_l1_error < 0 or self.completeness_absolute_residual < 0:
            raise ValueError("误差与残差不能为负")
        if self.wall_seconds < 0:
            raise ValueError("wall_seconds 不能为负")
        if not -1 <= self.spearman <= 1 or not 0 <= self.topk_overlap <= 1:
            raise ValueError("spearman/topk_overlap 超出定义域")


@dataclass(frozen=True, slots=True)
class QuadratureRecommendation:
    recommendation_id: str
    status: str
    default_rule: str | None
    fallback_rule: str | None
    passing_rules: tuple[str, ...]
    required_unit_ids: tuple[str, ...]
    thresholds: QuadratureThresholds
    execution_evidence_hash: str
    qualification: ArtifactQualification
    reasons: tuple[str, ...] = ()
    schema_version: str = "stage3-quadrature-recommendation-v1"

    def __post_init__(self) -> None:
        _require_identifier(self.recommendation_id, field_name="recommendation_id")
        _require_sha256(self.execution_evidence_hash, field_name="execution_evidence_hash")
        if self.status not in {"FIXTURE_RECOMMENDATION", "FORMAL_CANDIDATE", "QUALIFIED", "BLOCKED"}:
            raise ValueError("QuadratureRecommendation status 不受支持")
        if self.status == "BLOCKED" and self.default_rule is not None:
            raise ValueError("BLOCKED recommendation 不得指定 default rule")
        if self.status != "BLOCKED" and self.default_rule is None:
            raise ValueError("非 BLOCKED recommendation 必须指定 default rule")
        if self.status == "QUALIFIED" and not self.qualification.formal_eligible:
            raise FormalRunRejected("QUALIFIED_RECOMMENDATION_REQUIRES_GATE")
        if self.qualification.formal_eligible and self.status != "QUALIFIED":
            raise FormalRunRejected("FORMAL_ELIGIBLE_RECOMMENDATION_MUST_BE_QUALIFIED")

    @property
    def scope(self) -> str:
        return self.qualification.scope

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "recommendation_id": self.recommendation_id,
            "status": self.status,
            "default_rule": self.default_rule,
            "fallback_rule": self.fallback_rule,
            "passing_rules": list(self.passing_rules),
            "required_unit_ids": list(self.required_unit_ids),
            "thresholds": self.thresholds.to_dict(),
            "thresholds_hash": self.thresholds.artifact_hash,
            "execution_evidence_hash": self.execution_evidence_hash,
            "scope": self.scope,
            "formal_eligible": self.qualification.formal_eligible,
            "qualification_gate_hash": self.qualification.qualification_gate_hash,
            "reasons": list(self.reasons),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "QuadratureRecommendation":
        """严格重建 recommendation，并复算阈值与 artifact hash。"""

        expected = {
            "schema_version",
            "recommendation_id",
            "status",
            "default_rule",
            "fallback_rule",
            "passing_rules",
            "required_unit_ids",
            "thresholds",
            "thresholds_hash",
            "execution_evidence_hash",
            "scope",
            "formal_eligible",
            "qualification_gate_hash",
            "reasons",
            "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("QUADRATURE_RECOMMENDATION_FIELDS_MISMATCH")
        thresholds_raw = value["thresholds"]
        if not isinstance(thresholds_raw, Mapping) or set(thresholds_raw) != {
            "max_normalized_l1_error",
            "max_completeness_absolute_residual",
            "min_spearman",
            "min_topk_overlap",
            "max_unique_nodes",
        }:
            raise ValueError("QUADRATURE_RECOMMENDATION_THRESHOLDS_INVALID")
        passing = value["passing_rules"]
        units = value["required_unit_ids"]
        reasons = value["reasons"]
        if not all(isinstance(item, list) for item in (passing, units, reasons)):
            raise TypeError("QUADRATURE_RECOMMENDATION_ARRAYS_INVALID")
        if not all(
            all(isinstance(child, str) for child in item)
            for item in (passing, units, reasons)
        ):
            raise TypeError("QUADRATURE_RECOMMENDATION_ARRAY_ITEM_INVALID")
        thresholds = QuadratureThresholds(**dict(thresholds_raw))  # type: ignore[arg-type]
        if value["thresholds_hash"] != thresholds.artifact_hash:
            raise ValueError("QUADRATURE_RECOMMENDATION_THRESHOLDS_HASH_MISMATCH")
        recommendation = cls(
            recommendation_id=value["recommendation_id"],  # type: ignore[arg-type]
            status=value["status"],  # type: ignore[arg-type]
            default_rule=value["default_rule"],  # type: ignore[arg-type]
            fallback_rule=value["fallback_rule"],  # type: ignore[arg-type]
            passing_rules=tuple(passing),
            required_unit_ids=tuple(units),
            thresholds=thresholds,
            execution_evidence_hash=value["execution_evidence_hash"],  # type: ignore[arg-type]
            qualification=ArtifactQualification(
                scope=value["scope"],  # type: ignore[arg-type]
                formal_eligible=value["formal_eligible"],  # type: ignore[arg-type]
                qualification_gate_hash=value["qualification_gate_hash"],  # type: ignore[arg-type]
            ),
            reasons=tuple(reasons),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != recommendation.artifact_hash:
            raise ValueError("QUADRATURE_RECOMMENDATION_HASH_MISMATCH")
        return recommendation

    def qualify(
        self,
        *,
        execution: FormalExecutionEvidence,
        gate: GateRecord,
        artifact_ref: str | None = None,
    ) -> "QuadratureRecommendation":
        execution.require_for_stage(3)
        if self.scope != "formal" or self.status != "FORMAL_CANDIDATE":
            raise FormalRunRejected("QUADRATURE_RECOMMENDATION_NOT_FORMAL_CANDIDATE")
        if execution.artifact_hash != self.execution_evidence_hash:
            raise FormalRunRejected("QUADRATURE_EXECUTION_EVIDENCE_MISMATCH")
        accepted = require_accepted_gate(gate, stage=3)
        _require_gate_binding(accepted, artifact_ref)
        return replace(
            self,
            status="QUALIFIED",
            qualification=ArtifactQualification.from_gate(
                scope="formal", gate=accepted, stage=3
            ),
        )


class QuadratureRecommendationEngine:
    """要求每条候选规则通过全部预注册单元，再按节点/时间/名称稳定选择。"""

    def recommend(
        self,
        *,
        recommendation_id: str,
        observations: Sequence[QuadratureObservation],
        required_unit_ids: Sequence[str],
        thresholds: QuadratureThresholds,
        execution: FormalExecutionEvidence | None = None,
    ) -> QuadratureRecommendation:
        execution = execution or FormalExecutionEvidence("local_fixture")
        if execution.run_intent == "formal":
            execution.require_for_stage(3)
        units = tuple(required_unit_ids)
        if not units or len(set(units)) != len(units):
            raise ValueError("required_unit_ids 必须非空且无重复")
        by_rule: dict[str, dict[str, QuadratureObservation]] = {}
        for item in observations:
            bucket = by_rule.setdefault(item.rule_name, {})
            if item.unit_id in bucket:
                raise ValueError(f"DUPLICATE_QUADRATURE_OBSERVATION:{item.rule_name}:{item.unit_id}")
            bucket[item.unit_id] = item
        passing: list[tuple[int, float, str]] = []
        reasons: list[str] = []
        for rule_name, bucket in sorted(by_rule.items()):
            missing = sorted(set(units) - set(bucket))
            if missing:
                reasons.append(f"{rule_name}:missing_units={','.join(missing)}")
                continue
            rows = [bucket[unit] for unit in units]
            passed = all(
                row.normalized_l1_error <= thresholds.max_normalized_l1_error
                and row.completeness_absolute_residual
                <= thresholds.max_completeness_absolute_residual
                and row.spearman >= thresholds.min_spearman
                and row.topk_overlap >= thresholds.min_topk_overlap
                and row.unique_nodes <= thresholds.max_unique_nodes
                for row in rows
            )
            if not passed:
                reasons.append(f"{rule_name}:threshold_failed")
                continue
            passing.append(
                (
                    max(row.unique_nodes for row in rows),
                    sum(row.wall_seconds for row in rows) / len(rows),
                    rule_name,
                )
            )
        passing.sort()
        ordered_rules = tuple(item[2] for item in passing)
        default = ordered_rules[0] if ordered_rules else None
        fallback = ordered_rules[1] if len(ordered_rules) > 1 else None
        status = "BLOCKED"
        if default is not None:
            status = (
                "FIXTURE_RECOMMENDATION"
                if execution.run_intent == "local_fixture"
                else "FORMAL_CANDIDATE"
            )
        return QuadratureRecommendation(
            recommendation_id=recommendation_id,
            status=status,
            default_rule=default,
            fallback_rule=fallback,
            passing_rules=ordered_rules,
            required_unit_ids=units,
            thresholds=thresholds,
            execution_evidence_hash=execution.artifact_hash,
            qualification=ArtifactQualification.candidate(execution.run_intent),
            reasons=tuple(reasons),
        )


__all__ = [
    "CapturedEndpoint",
    "EndpointCaptureAdapter",
    "EndpointCaptureCoordinator",
    "EndpointCaptureRequest",
    "NodeValueCodec",
    "PersistentNodeGradientCache",
    "ProbePanel",
    "ProbePanelEntry",
    "QuadratureObservation",
    "QuadratureRecommendation",
    "QuadratureRecommendationEngine",
    "QuadratureThresholds",
    "ReferenceRefinementResult",
    "ReferenceRefinementRunner",
    "ReferenceRuleLevel",
    "SafeTensorTreeCodec",
]
