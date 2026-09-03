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
from collections import OrderedDict
from dataclasses import dataclass, field, replace
import hashlib
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from types import MappingProxyType
from typing import Callable, Hashable, Mapping, Protocol, Sequence

import numpy as np

from param_importance_nlp.contracts.errors import FormalRunRejected
from param_importance_nlp.contracts.immutable import (
    freeze_json_mapping,
    thaw_json_value,
)
from param_importance_nlp.contracts.jsonio import (
    canonical_json_bytes,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.stage23 import (
    ArtifactQualification,
    FormalExecutionEvidence,
    require_accepted_gate,
)
from param_importance_nlp.contracts.status import GateRecord, GateStatus
from param_importance_nlp.runtime.tensor_bundle import (
    load_tensor_bundle,
    publish_tensor_bundle,
)
from param_importance_nlp.atomic import sha256_file

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
        scope: str | None = None,
    ) -> "ProbePanel":
        execution = execution or FormalExecutionEvidence("local_fixture")
        if execution.run_intent == "formal":
            execution.require_for_stage(3)
        for entry in entries:
            entry.probe.assert_independent_from(endpoint)
        qualification_scope = scope or execution.run_intent
        if qualification_scope not in {"local_fixture", "pilot", "formal"}:
            raise ValueError("ProbePanel.scope 不受支持")
        expected_role = "formal" if qualification_scope == "formal" else "pilot"
        if scope is not None and any(entry.role != expected_role for entry in entries):
            raise ValueError("ProbePanel entries role 与 scope 不一致")
        return cls(
            panel_id=panel_id,
            endpoint_digest=endpoint.digest,
            entries=tuple(entries),
            execution_evidence_hash=execution.artifact_hash,
            qualification=ArtifactQualification.candidate(qualification_scope),
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


def _path_is_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` is equal to, or below, ``root``."""

    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolved_cache_path(value: str | Path, *, field_name: str) -> Path:
    try:
        path = Path(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 不是有效路径") from error
    if not path.is_absolute():
        # A receipt/ref path is never interpreted relative to the cache root.  The
        # caller must either pass an absolute path or deliberately use the current
        # process directory, which is then captured by the immutable receipt.
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _canonical_cache_root_reference(
    value: object,
    *,
    base: str | Path | None,
) -> Path:
    """Resolve a receipt/evidence cache root without weakening workspace binding.

    Runtime evidence stores a workspace-relative logical ref, whereas the cache
    instance stores a canonical absolute path. Relative refs are accepted only
    with an explicit trusted workspace base and may not cross it or traverse a
    symlink. Absolute refs are canonicalized and still compared exactly by the
    caller.
    """

    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_INVALID")
    reference = Path(value)
    if reference.is_absolute():
        return reference.resolve(strict=False)
    if base is None:
        raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_BASE_REQUIRED")
    base_input = Path(base)
    if base_input.is_symlink():
        raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_BASE_SYMLINK")
    base_path = _resolved_cache_path(base_input, field_name="cache_root_base")
    if not reference.parts or any(
        part in {"", ".", ".."} for part in reference.parts
    ):
        raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_ESCAPE")
    current = base_path
    for part in reference.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_SYMLINK")
    resolved = current.resolve(strict=False)
    if not _path_is_within(resolved, base_path):
        raise ValueError("NODE_CACHE_CACHE_ROOT_REFERENCE_ESCAPE")
    return resolved


def _require_external_receipt_root(cache_root: Path, value: str | Path) -> Path:
    receipt_root = _resolved_cache_path(value, field_name="receipt_root")
    if _path_is_within(receipt_root, cache_root) or _path_is_within(
        cache_root, receipt_root
    ):
        raise ValueError("NODE_CACHE_RECEIPT_ROOT_MUST_BE_OUTSIDE_CACHE_ROOT")
    return receipt_root


def _sha256_path(path: Path, *, field_name: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{field_name} 必须是非符号链接普通文件")
    return sha256_file(path)


def _write_immutable_canonical_json(path: Path, value: Mapping[str, object]) -> None:
    """Publish a canonical JSON file without ever replacing a different artifact."""

    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"NODE_CACHE_RECEIPT_PATH_INVALID:{path.name}")
        existing = load_canonical_json(path)
        if existing != value:
            raise ValueError("NODE_CACHE_RECEIPT_IMMUTABLE_CONFLICT")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Hard-linking the fully fsynced temporary file creates the final directory
        # entry without clobbering a concurrent publisher.  A winner that already
        # exists is compared instead of replaced.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"NODE_CACHE_RECEIPT_PATH_INVALID:{path.name}")
            existing = load_canonical_json(path)
            if existing != value:
                raise ValueError("NODE_CACHE_RECEIPT_IMMUTABLE_CONFLICT")
        else:
            temporary.unlink()
            temporary = Path()
    finally:
        if temporary.name:
            temporary.unlink(missing_ok=True)
_FileIdentity = tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class _CurrentNodeIndexEntry:
    """一条从磁盘重新读取的节点索引条目及其文件身份。"""

    key: NodeCacheKey
    object_ref: str
    object_manifest_hash: str
    value_digest: str
    identity: str
    source_paths: tuple[Path, ...]
    source_identities: tuple[tuple[str, _FileIdentity], ...]


@dataclass(slots=True)
class _NodeMemoEntry:
    """已完整验证的单节点模板；模板永远不直接返回给调用方。"""

    key: NodeCacheKey
    object_ref: str
    value_digest: str
    index_identity: str
    source_identities: tuple[tuple[str, _FileIdentity], ...]
    bundle_identities: tuple[tuple[str, _FileIdentity], ...]
    generation: str
    template: object
    size_bytes: int


def _path_identity(path: Path, *, expect_directory: bool = False) -> _FileIdentity:
    """返回足以发现原地写入/替换的本地文件身份；异常时 fail closed。"""

    if path.is_symlink():
        raise ValueError(f"NODE_CACHE_PATH_SYMLINK:{path}")
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError(f"NODE_CACHE_PATH_STAT_FAILED:{path}") from error
    if expect_directory:
        if not path.is_dir():
            raise ValueError(f"NODE_CACHE_DIRECTORY_MISSING:{path}")
    elif not path.is_file():
        raise ValueError(f"NODE_CACHE_FILE_MISSING:{path}")
    return (
        int(getattr(stat, "st_dev", 0)),
        int(getattr(stat, "st_ino", 0)),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )


def _bundle_file_identities(root: Path) -> tuple[tuple[str, _FileIdentity], ...]:
    """枚举已发布 bundle 的全部文件并读取身份。

    每次命中检查都会重新枚举，因而新增/删除文件也会使缓存失效；根目录身份
    另外捕获没有文件内容的目录生命周期变化。这里不读取文件内容，哈希验证仍
    只由 ``load_tensor_bundle`` 在首次加载或失效后负责。
    """

    _path_identity(root, expect_directory=True)
    identities: list[tuple[str, _FileIdentity]] = [
        ("<root>", _path_identity(root, expect_directory=True))
    ]
    try:
        paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    except OSError as error:
        raise ValueError(f"NODE_CACHE_BUNDLE_ENUMERATION_FAILED:{root}") from error
    for path in paths:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"NODE_CACHE_PATH_SYMLINK:{relative}")
        if path.is_dir():
            # Directory mtime catches directory-level lifecycle changes while file
            # identities below catch content replacement. Include all directories so
            # a newly-created empty subdirectory cannot be silently ignored.
            identities.append((relative + "/", _path_identity(path, expect_directory=True)))
        elif path.is_file():
            identities.append((relative, _path_identity(path)))
        else:
            raise ValueError(f"NODE_CACHE_PATH_UNSUPPORTED:{relative}")
    return tuple(identities)


def _tree_nbytes(value: object, *, _seen: set[int] | None = None) -> int | None:
    """Recursively estimate template bytes, including Python object overhead.

    Tensor/array storage is counted in addition to sys.getsizeof. Mapping keys,
    containers and scalar leaves are included so a scalar-heavy tree cannot bypass
    the bound; unknown values fail closed. ``_seen`` is retained for the whole
    traversal so shared references are counted once and cycles terminate safely.
    """

    seen = _seen if _seen is not None else set()
    identity = id(value)
    if identity in seen:
        return 0
    seen.add(identity)
    try:
        total = int(sys.getsizeof(value))
    except (TypeError, ValueError):
        return None
    try:
        import torch
    except ImportError:  # pragma: no cover - 极简环境
        torch = None  # type: ignore[assignment]
    if torch is not None and isinstance(value, torch.Tensor):
        return total + int(value.numel()) * int(value.element_size())
    if isinstance(value, np.ndarray):
        return total + int(value.nbytes)
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_size = _tree_nbytes(key, _seen=seen)
            child_size = _tree_nbytes(child, _seen=seen)
            if key_size is None or child_size is None:
                return None
            total += key_size + child_size
        return total
    if isinstance(value, (list, tuple)):
        for child in value:
            child_size = _tree_nbytes(child, _seen=seen)
            if child_size is None:
                return None
            total += child_size
        return total
    if value is None or isinstance(value, (bool, int, float, str)):
        return total
    return None


def _available_memory_bytes() -> int | None:
    """Best-effort read-only host availability probe for the memo budget."""

    observed: list[int] = []
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        if page_size > 0 and available_pages > 0:
            observed.append(page_size * available_pages)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    try:
        import psutil  # type: ignore[import-not-found]

        available = int(psutil.virtual_memory().available)
        if available > 0:
            observed.append(available)
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        pass
    # A container's cgroup limit can be much smaller than host-visible RAM. Read
    # the limit/usage files when present so a bounded memo cannot reserve a whole
    # 16 GiB host budget inside a memory-constrained worker.
    for limit_path, usage_path in (
        (
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory.current"),
        ),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    ):
        try:
            limit_text = limit_path.read_text(encoding="ascii").strip()
            if limit_text == "max":
                continue
            limit = int(limit_text)
            usage = int(usage_path.read_text(encoding="ascii").strip())
            if limit > 0 and usage >= 0 and limit > usage:
                observed.append(limit - usage)
        except (OSError, UnicodeError, TypeError, ValueError):
            continue
    return min(observed) if observed else None


class PersistentNodeGradientCache:
    """批次原子提交、可跨进程恢复且不允许覆盖的安全节点缓存。

    ``get`` 的首个访问仍会完整执行 TensorBundle 的 manifest、文件集合、dtype、
    shape、内容哈希和节点值哈希验证。验证成功的单节点值可以在本进程/本实例
    内短暂复用；命中前只重新读取相关 commit/index 条目并 stat 该条目引用的
    全部文件，因此磁盘对象一旦发生原地修改、替换或生命周期变化，旧模板会被
    丢弃并回到完整验证路径。memo 不写入任何 artifact，也不跨进程共享。
    """

    _DEFAULT_MAX_MEMO_BYTES = 16 * 1024**3
    _MEMORY_SAFETY_MARGIN_BYTES = 512 * 1024**2

    def __init__(
        self,
        root: str | Path,
        *,
        codec: NodeValueCodec | None = None,
        receipt_root: str | Path | None = None,
        memoize: bool = True,
        max_memo_bytes: int | None = None,
    ) -> None:
        if type(memoize) is not bool:
            raise TypeError("memoize 必须是 bool")
        if max_memo_bytes is not None and (
            isinstance(max_memo_bytes, bool)
            or not isinstance(max_memo_bytes, int)
            or max_memo_bytes < 0
        ):
            raise ValueError("max_memo_bytes 必须是非负整数或 null")
        self.root = _resolved_cache_path(root, field_name="cache_root")
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self.codec = codec or SafeTensorTreeCodec()
        self.receipt_root = (
            _require_external_receipt_root(self.root, receipt_root)
            if receipt_root is not None
            else None
        )
        self._lock = threading.RLock()
        self._memoize = memoize
        self._max_memo_bytes = (
            self._DEFAULT_MAX_MEMO_BYTES
            if max_memo_bytes is None
            else max_memo_bytes
        )
        self._memo: OrderedDict[str, _NodeMemoEntry] = OrderedDict()
        self._memo_bytes = 0
        self._memo_enabled = False
        self._memo_generation: str | None = None
        self._memo_generation_path: Path | None = None
        self._memo_generation_identity: _FileIdentity | None = None
        self._memo_mode: str | None = None
        self._memo_allowed_key_digests: frozenset[str] | None = None
        self._memo_ephemeral_index_snapshot: tuple[tuple[str, _FileIdentity], ...] | None = None
        self._index: dict[str, tuple[NodeCacheKey, str, str]] = {}
        self._index_sources: dict[str, tuple[Path, ...]] = {}
        self._index_identities: dict[str, str] = {}
        self._index_directory_identity: _FileIdentity | None = None
        self._reload_index()
        self._try_enable_memoization()

    def __len__(self) -> int:
        with self._lock:
            return len(self._index)

    def __contains__(self, key: object) -> bool:
        with self._lock:
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

    def _commit_header(self, path: Path) -> Mapping[str, object]:
        """只验证 commit/index 外壳，不重新加载其 bundle。"""

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
        entries = commit.get("entries")
        if not isinstance(entries, list):
            raise ValueError("NODE_CACHE_ENTRIES_INVALID")
        return commit

    def _entry_identity(
        self,
        path: Path,
        commit: Mapping[str, object],
        entry: Mapping[str, object],
    ) -> str:
        try:
            commit_ref = path.relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("NODE_CACHE_COMMIT_PATH_ESCAPE") from error
        return canonical_json_hash(
            {
                "commit_ref": commit_ref,
                "commit_artifact_hash": commit["artifact_hash"],
                "entry": dict(entry),
            }
        )

    def _current_index_entry(self, key_digest: str) -> _CurrentNodeIndexEntry | None:
        """重新读取当前 key 的 commit 条目及所有来源文件身份。"""

        source_paths = self._index_sources.get(key_digest)
        if not source_paths:
            return None
        matches: list[tuple[Path, Mapping[str, object], Mapping[str, object]]] = []
        for path in source_paths:
            commit = self._commit_header(path)
            entries = commit["entries"]
            assert isinstance(entries, list)
            found = False
            for raw_entry in entries:
                if not isinstance(raw_entry, Mapping):
                    raise ValueError("NODE_CACHE_ENTRY_NOT_OBJECT")
                if str(raw_entry.get("key_digest")) != key_digest:
                    continue
                found = True
                key_payload = raw_entry.get("key")
                if not isinstance(key_payload, Mapping):
                    raise ValueError("NODE_CACHE_KEY_PAYLOAD_INVALID")
                key = _key_from_payload(key_payload)
                if key.digest != key_digest:
                    raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
                value_digest = raw_entry.get("value_digest")
                if not isinstance(value_digest, str):
                    raise ValueError("NODE_CACHE_VALUE_DIGEST_INVALID")
                matches.append((path, commit, raw_entry))
            if not found:
                raise ValueError("NODE_CACHE_INDEX_ENTRY_MISSING")
        if not matches:
            raise ValueError("NODE_CACHE_INDEX_ENTRY_MISSING")
        _, first_commit, first_entry = matches[0]
        first_key_payload = first_entry["key"]
        assert isinstance(first_key_payload, Mapping)
        key = _key_from_payload(first_key_payload)
        object_ref = str(first_commit["object_ref"])
        object_manifest_hash = str(first_commit["object_manifest_hash"])
        value_digest = str(first_entry["value_digest"])
        source_identities: list[tuple[str, _FileIdentity]] = []
        entry_identities: list[str] = []
        for path, commit, entry in matches:
            if str(commit["object_ref"]) != object_ref or str(entry["value_digest"]) != value_digest:
                raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
            entry_identities.append(self._entry_identity(path, commit, entry))
            source_identities.append(
                (
                    path.relative_to(self.root).as_posix(),
                    _path_identity(path),
                )
            )
        return _CurrentNodeIndexEntry(
            key=key,
            object_ref=object_ref,
            object_manifest_hash=object_manifest_hash,
            value_digest=value_digest,
            identity=canonical_json_hash(sorted(entry_identities)),
            source_paths=tuple(path for path, _commit, _entry in matches),
            source_identities=tuple(sorted(source_identities)),
        )

    def _reload_index(self) -> None:
        with self._lock:
            self._reload_index_unlocked()

    def _reload_index_unlocked(self) -> None:
        # Index/lifecycle changes are an explicit cache boundary. Drop every
        # template before touching disk so a failed reload can never leave a
        # previously-authorized value available to a later caller.
        if self._memo_enabled:
            # A commit/index lifecycle change means the cache is no longer the
            # sealed generation that authorized this process-local memo.
            self._disable_memoization_unlocked()
        else:
            self._memo.clear()
            self._memo_bytes = 0
        index: dict[str, tuple[NodeCacheKey, str, str]] = {}
        sources: dict[str, list[Path]] = {}
        identities: dict[str, list[str]] = {}
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
                sources.setdefault(key_digest, []).append(path)
                identities.setdefault(key_digest, []).append(
                    self._entry_identity(path, commit, entry)
                )
        self._index = index
        self._index_sources = {
            key_digest: tuple(paths)
            for key_digest, paths in sources.items()
        }
        self._index_identities = {
            key_digest: canonical_json_hash(sorted(entry_ids))
            for key_digest, entry_ids in identities.items()
        }
        self._index_directory_identity = _path_identity(self.commits, expect_directory=True)

    def _index_directory_changed(self) -> bool:
        expected = self._index_directory_identity
        if expected is None:
            return True
        try:
            return _path_identity(self.commits, expect_directory=True) != expected
        except (OSError, ValueError):
            return True

    def _disable_memoization_unlocked(self) -> None:
        self._memo.clear()
        self._memo_bytes = 0
        self._memo_enabled = False
        self._memo_generation = None
        self._memo_generation_path = None
        self._memo_generation_identity = None
        self._memo_mode = None
        self._memo_allowed_key_digests = None
        self._memo_ephemeral_index_snapshot = None

    def _activate_memoization(
        self, receipt: Mapping[str, object], path: Path, *, validate_cache: bool
    ) -> None:
        if sys.platform != "linux":
            return
        validated = self._validate_sealed_payload(receipt)
        if validated.get("cache_root_ref") != self.root.as_posix():
            raise ValueError("NODE_CACHE_CACHE_ROOT_BINDING_MISMATCH")
        if validate_cache:
            self._cache_state_matches_seal(validated, allow_missing=False)
        identity = _path_identity(path)
        generation = validated.get("receipt_hash")
        if not isinstance(generation, str):
            raise ValueError("NODE_CACHE_SEAL_HASH_MISSING")
        self._memo_generation = generation
        self._memo_generation_path = path
        self._memo_generation_identity = identity
        self._memo_mode = "sealed"
        self._memo_allowed_key_digests = None
        self._memo_ephemeral_index_snapshot = None
        self._memo_enabled = True

    def authorize_ephemeral_memoization(
        self,
        keys: Sequence[NodeCacheKey],
        evidence: Mapping[str, object],
        *,
        cache_root_base: str | Path | None = None,
    ) -> None:
        """Explicitly authorize a Linux fresh-unit memo from verified commit evidence.

        This is never called by construction or by an ordinary unsealed cache.  The
        caller must first complete two-phase publication and produce ordinary
        ``commit_evidence``.  The authorization binds the process-local memo to the
        exact requested key set and a stat identity snapshot of every commit/index
        file; any later commit/index change disables it before a hit can be served.
        """

        with self._lock:
            self._disable_memoization_unlocked()
            if not self._memoize or sys.platform != "linux":
                return
            if not isinstance(evidence, Mapping):
                raise TypeError("NODE_CACHE_EPHEMERAL_EVIDENCE_INVALID")
            if evidence.get("schema_version") != "stage3-path-node-cache-evidence-v1":
                raise ValueError("NODE_CACHE_EPHEMERAL_EVIDENCE_SCHEMA_MISMATCH")
            try:
                evidence_root = _canonical_cache_root_reference(
                    evidence.get("cache_root_ref"),
                    base=cache_root_base,
                )
            except (OSError, TypeError, ValueError) as error:
                raise ValueError(
                    "NODE_CACHE_EPHEMERAL_CACHE_ROOT_MISMATCH"
                ) from error
            if evidence_root != self.root:
                raise ValueError("NODE_CACHE_EPHEMERAL_CACHE_ROOT_MISMATCH")
            requested = sorted({
                key.digest
                for key in keys
                if isinstance(key, NodeCacheKey)
            })
            if len(requested) != len(tuple(keys)):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            if not requested:
                raise ValueError("NODE_CACHE_EPHEMERAL_KEYS_REQUIRED")
            evidence_hash = evidence.get("evidence_hash")
            evidence_body = {
                name: item for name, item in evidence.items() if name != "evidence_hash"
            }
            if (
                not isinstance(evidence_hash, str)
                or canonical_json_hash(evidence_body) != evidence_hash
            ):
                raise ValueError("NODE_CACHE_EPHEMERAL_EVIDENCE_HASH_MISMATCH")
            commit_evidence = evidence.get("commit_evidence")
            if not isinstance(commit_evidence, Mapping):
                raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_EVIDENCE_MISSING")
            commit_evidence_hash = commit_evidence.get("evidence_hash")
            commit_body = {
                name: item
                for name, item in commit_evidence.items()
                if name != "evidence_hash"
            }
            if (
                not isinstance(commit_evidence_hash, str)
                or canonical_json_hash(commit_body) != commit_evidence_hash
            ):
                raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_EVIDENCE_HASH_MISMATCH")
            if (
                commit_evidence.get("schema_version")
                != "stage3-node-cache-evidence-v1"
                or commit_evidence.get("codec_id") != self.codec.codec_id
                or commit_evidence.get("all_requested_keys_committed") is not True
                or commit_evidence.get("missing_key_digests") != []
                or commit_evidence.get("requested_key_digests") != requested
            ):
                raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_EVIDENCE_INVALID")
            authoritative = commit_evidence.get("authoritative_commits")
            if not isinstance(authoritative, list) or not authoritative:
                raise ValueError("NODE_CACHE_EPHEMERAL_AUTHORITATIVE_COMMITS_MISSING")
            covered: set[str] = set()
            records_by_digest: dict[str, list[tuple[Mapping[str, object], Mapping[str, object]]]] = {}
            for item in authoritative:
                if not isinstance(item, Mapping):
                    raise ValueError("NODE_CACHE_EPHEMERAL_AUTHORITATIVE_COMMIT_INVALID")
                commit_ref = item.get("commit_ref")
                commit_relative = Path(str(commit_ref))
                if (
                    not isinstance(commit_ref, str)
                    or commit_relative.is_absolute()
                    or commit_relative.parts[:1] != ("commits",)
                    or len(commit_relative.parts) != 2
                    or commit_relative.suffix != ".json"
                ):
                    raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_REF_INVALID")
                commit_path = self.root / commit_relative
                if commit_path.is_symlink() or not commit_path.is_file():
                    raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_MISSING")
                commit = self._commit_header(commit_path)
                for name in ("commit_artifact_hash", "object_ref", "object_manifest_hash"):
                    if item.get(name) != commit.get(
                        "artifact_hash" if name == "commit_artifact_hash" else name
                    ):
                        raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_IDENTITY_MISMATCH")
                raw_digests = item.get("key_digests")
                if not isinstance(raw_digests, list) or not raw_digests:
                    raise ValueError("NODE_CACHE_EPHEMERAL_COMMIT_KEYS_INVALID")
                for raw_digest in raw_digests:
                    digest = _require_sha256(
                        str(raw_digest), field_name="ephemeral key digest"
                    )
                    covered.add(digest)
                    records_by_digest.setdefault(digest, []).append((item, commit))
            if covered != set(requested):
                raise ValueError("NODE_CACHE_EPHEMERAL_KEY_SET_MISMATCH")
            for digest in requested:
                current = self._current_index_entry(digest)
                if current is None:
                    raise ValueError("NODE_CACHE_EPHEMERAL_INDEX_ENTRY_MISSING")
                matches = records_by_digest.get(digest, ())
                if not any(
                    current.object_ref == str(item.get("object_ref"))
                    and current.object_manifest_hash
                    == str(item.get("object_manifest_hash"))
                    and any(
                        isinstance(entry, Mapping)
                        and str(entry.get("key_digest")) == digest
                        and str(entry.get("value_digest")) == current.value_digest
                        for entry in commit.get("entries", ())
                    )
                    for item, commit in matches
                ):
                    raise ValueError("NODE_CACHE_EPHEMERAL_INDEX_IDENTITY_MISMATCH")
            snapshot = _bundle_file_identities(self.commits)
            snapshot_payload = [
                [name, list(identity)] for name, identity in snapshot
            ]
            self._memo_generation = canonical_json_hash(
                {
                    "evidence_hash": evidence_hash,
                    "requested_key_digests": requested,
                    "index_snapshot": snapshot_payload,
                }
            )
            self._memo_generation_path = None
            self._memo_generation_identity = None
            self._memo_mode = "ephemeral"
            self._memo_allowed_key_digests = frozenset(requested)
            self._memo_ephemeral_index_snapshot = snapshot
            self._memo_enabled = True

    def _try_enable_memoization(self) -> None:
        if (
            not self._memoize
            or sys.platform != "linux"
            or self.receipt_root is None
            or self.receipt_root.is_symlink()
            or not self.receipt_root.is_dir()
        ):
            return
        for path in sorted(self.receipt_root.glob("*.SEALED.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                receipt = load_canonical_json(path)
                if not isinstance(receipt, Mapping):
                    continue
                if receipt.get("cache_root_ref") != self.root.as_posix():
                    continue
                self._activate_memoization(receipt, path, validate_cache=True)
                return
            except (OSError, TypeError, ValueError):
                self._disable_memoization_unlocked()

    def _memo_generation_current(self) -> bool:
        if not self._memo_enabled or self._memo_generation is None:
            return False
        if self._memo_mode == "ephemeral":
            snapshot = self._memo_ephemeral_index_snapshot
            if snapshot is None:
                return False
            try:
                return _bundle_file_identities(self.commits) == snapshot
            except (OSError, ValueError):
                return False
        if (
            self._memo_mode != "sealed"
            or self._memo_generation_path is None
            or self._memo_generation_identity is None
        ):
            return False
        path = self._memo_generation_path
        try:
            if _path_identity(path) != self._memo_generation_identity:
                return False
            receipt = load_canonical_json(path)
            if not isinstance(receipt, Mapping):
                return False
            validated = self._validate_sealed_payload(receipt)
            return (
                validated.get("receipt_hash") == self._memo_generation
                and validated.get("cache_root_ref") == self.root.as_posix()
            )
        except (OSError, TypeError, ValueError):
            return False

    def _memo_matches(
        self,
        memo: _NodeMemoEntry,
        current: _CurrentNodeIndexEntry,
    ) -> bool:
        if (
            memo.key != current.key
            or memo.object_ref != current.object_ref
            or memo.value_digest != current.value_digest
            or memo.generation != self._memo_generation
            or memo.index_identity != current.identity
            or memo.source_identities != current.source_identities
        ):
            return False
        # Enumerating and stat'ing every file on each hit is deliberate. It keeps
        # the memo a performance hint rather than a new authority and catches
        # same-size in-place writes, atomic replacement, and bundle lifecycle drift.
        current_bundle = _bundle_file_identities(self.root / current.object_ref)
        return memo.bundle_identities == current_bundle

    def _can_remember(self, size_bytes: int) -> bool:
        if (
            not self._memo_enabled
            or not self._memoize
            or size_bytes <= 0
            or size_bytes > self._max_memo_bytes
        ):
            return False
        available = _available_memory_bytes()
        if available is not None and (
            self._memo_bytes + size_bytes
            > max(0, available - self._MEMORY_SAFETY_MARGIN_BYTES)
        ):
            return False
        return True

    def _remember(
        self,
        *,
        current: _CurrentNodeIndexEntry,
        template: object,
        bundle_identities: tuple[tuple[str, _FileIdentity], ...],
    ) -> None:
        size_bytes = _tree_nbytes(template)
        if size_bytes is None or size_bytes <= 0 or not self._memo_enabled:
            return
        digest = current.key.digest
        if (
            self._memo_mode == "ephemeral"
            and (
                self._memo_allowed_key_digests is None
                or digest not in self._memo_allowed_key_digests
            )
        ):
            return
        previous = self._memo.pop(digest, None)
        if previous is not None:
            self._memo_bytes = max(0, self._memo_bytes - previous.size_bytes)
        available = _available_memory_bytes()
        budget = self._max_memo_bytes
        if available is not None:
            budget = min(
                budget, max(0, available - self._MEMORY_SAFETY_MARGIN_BYTES)
            )
        # Evict before the admission check so a newly-small value can reclaim
        # space from old LRU entries instead of being rejected prematurely.
        while self._memo and self._memo_bytes + size_bytes > budget:
            _evicted_digest, evicted = self._memo.popitem(last=False)
            self._memo_bytes = max(0, self._memo_bytes - evicted.size_bytes)
        if size_bytes > budget:
            return
        self._memo[digest] = _NodeMemoEntry(
            key=current.key,
            object_ref=current.object_ref,
            value_digest=current.value_digest,
            index_identity=current.identity,
            source_identities=current.source_identities,
            bundle_identities=bundle_identities,
            generation=self._memo_generation or "",
            template=template,
            size_bytes=size_bytes,
        )
        self._memo_bytes += size_bytes

    def _load_current(self, current: _CurrentNodeIndexEntry) -> object:
        """完整验证当前 index 指向的 object，并返回本次调用的独立值。"""

        object_path = self.root / current.object_ref
        # Capture identities around the full load. If a writer/lifecycle actor
        # mutates the object during verification, fail closed rather than caching
        # a value whose content and recorded file identity came from different
        # moments.
        before = _bundle_file_identities(object_path)
        state, bundle = load_tensor_bundle(object_path)
        after = _bundle_file_identities(object_path)
        if before != after:
            self._memo.pop(current.key.digest, None)
            raise ValueError("NODE_CACHE_BUNDLE_CHANGED_DURING_LOAD")
        if bundle.manifest_sha256 != current.object_manifest_hash:
            raise ValueError("NODE_CACHE_OBJECT_MANIFEST_HASH_MISMATCH")
        if not isinstance(state, Mapping) or state.get("codec_id") != self.codec.codec_id:
            raise ValueError("NODE_CACHE_OBJECT_STATE_INVALID")
        values = state.get("values")
        if not isinstance(values, Mapping):
            raise ValueError("NODE_CACHE_VALUES_INVALID")
        encoded = values.get(current.key.digest)
        if encoded is None or _tree_digest(encoded) != current.value_digest:
            raise ValueError("NODE_CACHE_VALUE_CHANGED_AFTER_INDEX")
        template = self.codec.decode(encoded)
        # Do not let the complete decoded object keep occupying the budget while
        # deciding whether this one node template can be memoized. Safe codecs copy
        # tensors during decode; deleting these containers therefore cannot mutate
        # the template or alter the returned value.
        del state, values, encoded
        # ``after`` was captured before decoding, so a concurrent lifecycle change
        # during codec work is checked one more time immediately before publication.
        final_identities = _bundle_file_identities(object_path)
        if after != final_identities:
            self._memo.pop(current.key.digest, None)
            raise ValueError("NODE_CACHE_BUNDLE_CHANGED_DURING_LOAD")
        self._remember(
            current=current,
            template=template,
            bundle_identities=final_identities,
        )
        return _clone_tree(template)

    def get(self, key: NodeCacheKey) -> object:
        if not isinstance(key, NodeCacheKey):
            raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
        with self._lock:
            indexed = self._index.get(key.digest)
            if indexed is None:
                raise KeyError(key)
            if self._memo_enabled and not self._memo_generation_current():
                self._disable_memoization_unlocked()
            if self._index_directory_changed():
                self._reload_index_unlocked()
                indexed = self._index.get(key.digest)
                if indexed is None:
                    raise KeyError(key)
            current = self._current_index_entry(key.digest)
            if current is None:
                self._reload_index_unlocked()
                if key.digest not in self._index:
                    raise KeyError(key)
                current = self._current_index_entry(key.digest)
                if current is None:  # pragma: no cover - defensive race guard
                    raise KeyError(key)
            expected = (current.key, current.object_ref, current.value_digest)
            if indexed != expected or self._index_identities.get(key.digest) != current.identity:
                # A changed index is itself a lifecycle boundary. Rebuild the
                # authoritative in-memory index before deciding whether to load.
                self._reload_index_unlocked()
                indexed = self._index.get(key.digest)
                if indexed is None:
                    raise KeyError(key)
                current = self._current_index_entry(key.digest)
                if current is None:  # pragma: no cover - defensive race guard
                    raise KeyError(key)
                if indexed != (current.key, current.object_ref, current.value_digest):
                    raise ValueError("NODE_CACHE_INDEX_CHANGED_DURING_READ")
            memo = self._memo.get(key.digest)
            if memo is not None:
                try:
                    valid = self._memo_matches(memo, current)
                except (OSError, ValueError):
                    # Re-enter complete bundle verification below. This preserves
                    # the caller-visible failure reason for a malformed object and
                    # never falls back to stale in-memory data.
                    valid = False
                if valid:
                    self._memo.move_to_end(key.digest)
                    try:
                        return _clone_tree(memo.template)
                    except Exception:
                        self._memo.pop(key.digest, None)
                        self._memo_bytes = max(
                            0, self._memo_bytes - memo.size_bytes
                        )
                        raise
                self._disable_memoization_unlocked()
            return self._load_current(current)

    @property
    def memoization_max_bytes(self) -> int:
        """当前实例的进程内 memo 上限（不代表已使用量）。"""

        return self._max_memo_bytes

    @property
    def memoization_bytes(self) -> int:
        """当前 memo 模板的估计 tensor/array 字节数。"""

        with self._lock:
            return self._memo_bytes

    def clear_memoization(self) -> None:
        """清空本实例 memo；磁盘 cache/index 不受影响。"""

        with self._lock:
            self._memo.clear()
            self._memo_bytes = 0

    def clear(self) -> None:
        """生命周期兼容别名：只清空内存 memo，不删除持久 artifact。"""

        self.clear_memoization()

    def evict_memo(self, key: NodeCacheKey | None = None) -> None:
        """驱逐一个节点或整个实例的进程内 memo。"""

        with self._lock:
            if key is None:
                self._memo.clear()
                self._memo_bytes = 0
                return
            if not isinstance(key, NodeCacheKey):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            memo = self._memo.pop(key.digest, None)
            if memo is not None:
                self._memo_bytes = max(0, self._memo_bytes - memo.size_bytes)

    def close(self) -> None:
        """关闭/离开一个 cache 生命周期时清空内存 memo。

        持久对象仍由现有恢复协议管理；保持 close 后可重新 get，便于调用方在
        lifecycle transition 后显式释放内存而不改变 artifact 语义。
        """

        self.clear_memoization()

    def lifecycle_transition(self, *_args: object, **_kwargs: object) -> None:
        """供生命周期控制器调用的内存边界钩子。"""

        self.clear_memoization()

    def publish_many(self, entries: Mapping[NodeCacheKey, object]) -> int:
        if not entries:
            return 0
        prepared: list[tuple[NodeCacheKey, object, str]] = []
        for key, value in entries.items():
            if not isinstance(key, NodeCacheKey):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            encoded = self.codec.encode(value)
            value_digest = _tree_digest(encoded)
            with self._lock:
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
        with self._lock:
            return tuple(self._index[digest][0] for digest in sorted(self._index))

    def reconcile(self) -> dict[str, object]:
        with self._lock:
            self._reload_index_unlocked()
            referenced = {
                str(load_canonical_json(path)["batch_digest"])  # type: ignore[index]
                for path in self.commits.glob("*.json")
            }
            objects = {path.name for path in self.objects.iterdir() if path.is_dir()}
            return {
                "committed_key_digests": sorted(self._index),
                "orphan_objects": sorted(objects - referenced),
            }

    def _receipt_root_for(self, receipt_root: str | Path | None) -> Path:
        configured = self.receipt_root if receipt_root is None else receipt_root
        if configured is None:
            raise ValueError("NODE_CACHE_RECEIPT_ROOT_REQUIRED")
        return _require_external_receipt_root(self.root, configured)

    def _cache_inventory(self) -> dict[str, object]:
        """Return a complete, verified inventory of this unit cache.

        This is intentionally stricter than ``reconcile``.  Retention is allowed to
        remove only files in this inventory, and no lock, unknown file, symlink or
        extra directory may be mistaken for a recoverable writer artifact.
        """

        if (self.root / "writer.lock").exists() or (self.root / "writer.lock").is_symlink():
            raise ValueError("NODE_CACHE_WRITER_LOCK_PRESENT")
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("NODE_CACHE_ROOT_INVALID")
        root_children = {child.name: child for child in self.root.iterdir()}
        if set(root_children) != {"objects", "commits"}:
            unknown = sorted(set(root_children) - {"objects", "commits"})
            missing = sorted({"objects", "commits"} - set(root_children))
            raise ValueError(
                f"NODE_CACHE_UNKNOWN_ROOT_ENTRIES:missing={missing}:extra={unknown}"
            )
        for name in ("objects", "commits"):
            child = root_children[name]
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"NODE_CACHE_{name.upper()}_ROOT_INVALID")

        # Commits are direct children only.  A nested directory or a non-JSON file is
        # an unknown artifact, even when it happens to be empty or readable.
        commit_children = list(self.commits.iterdir())
        if any(child.is_dir() or child.is_symlink() for child in commit_children):
            raise ValueError("NODE_CACHE_UNKNOWN_COMMIT_ENTRY")
        commit_paths = sorted(
            (child for child in commit_children if child.is_file()),
            key=lambda item: item.name,
        )
        if len(commit_paths) != len(commit_children):
            raise ValueError("NODE_CACHE_UNKNOWN_COMMIT_ENTRY")

        commits: list[dict[str, object]] = []
        object_refs: dict[str, dict[str, object]] = {}
        key_records: dict[str, dict[str, object]] = {}
        for path in commit_paths:
            if path.suffix != ".json" or path.is_symlink():
                raise ValueError(f"NODE_CACHE_UNKNOWN_COMMIT_FILE:{path.name}")
            commit, _state = self._validated_commit(path)
            batch_digest = _require_sha256(
                str(commit.get("batch_digest")), field_name="commit.batch_digest"
            )
            if path.name != f"{batch_digest}.json":
                raise ValueError("NODE_CACHE_COMMIT_PATH_MISMATCH")
            object_ref_value = commit.get("object_ref")
            if not isinstance(object_ref_value, str):
                raise ValueError("NODE_CACHE_OBJECT_REF_INVALID")
            object_ref = Path(object_ref_value)
            if (
                object_ref.is_absolute()
                or object_ref.parts != ("objects", batch_digest)
                or ".." in object_ref.parts
            ):
                raise ValueError("NODE_CACHE_OBJECT_PATH_ESCAPE")
            object_path = self.root / object_ref
            if not _path_is_within(object_path.resolve(strict=False), self.root):
                raise ValueError("NODE_CACHE_OBJECT_PATH_ESCAPE")
            manifest_hash = _require_sha256(
                str(commit.get("object_manifest_hash")),
                field_name="commit.object_manifest_hash",
            )
            _require_sha256(
                str(commit.get("artifact_hash")), field_name="commit.artifact_hash"
            )
            if (
                batch_digest in object_refs
                and object_refs[batch_digest]["object_ref"] != object_ref_value
            ):
                raise ValueError("NODE_CACHE_OBJECT_REF_CONFLICT")
            object_refs[batch_digest] = {
                "object_ref": object_ref_value,
                "manifest_hash": manifest_hash,
            }
            entries = commit.get("entries")
            if not isinstance(entries, list):
                raise ValueError("NODE_CACHE_ENTRIES_INVALID")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise ValueError("NODE_CACHE_ENTRY_NOT_OBJECT")
                key_digest = _require_sha256(
                    str(entry.get("key_digest")), field_name="entry.key_digest"
                )
                value_digest = _require_sha256(
                    str(entry.get("value_digest")), field_name="entry.value_digest"
                )
                key_payload = entry.get("key")
                if not isinstance(key_payload, Mapping):
                    raise ValueError("NODE_CACHE_ENTRY_KEY_INVALID")
                key = _key_from_payload(key_payload)
                if key.digest != key_digest:
                    raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
                previous = key_records.get(key_digest)
                record = {
                    "key": _key_payload(key),
                    "key_digest": key_digest,
                    "value_digest": value_digest,
                }
                if previous is not None and previous != record:
                    raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
                key_records[key_digest] = record
            commits.append(
                {
                    "path": path.relative_to(self.root).as_posix(),
                    "commit": dict(commit),
                    "byte_count": path.stat().st_size,
                    "sha256": _sha256_path(path, field_name="commit"),
                }
            )

        expected_object_names = set(object_refs)
        object_children = list(self.objects.iterdir())
        if any(child.is_symlink() or not child.is_dir() for child in object_children):
            raise ValueError("NODE_CACHE_UNKNOWN_OBJECT_ENTRY")
        actual_object_names = {child.name for child in object_children}
        if actual_object_names != expected_object_names:
            raise ValueError(
                "NODE_CACHE_OBJECT_SET_MISMATCH:"
                f"missing={sorted(expected_object_names-actual_object_names)}:"
                f"extra={sorted(actual_object_names-expected_object_names)}"
            )

        objects: list[dict[str, object]] = []
        files: list[dict[str, object]] = [
            {
                "path": item["path"],
                "kind": "commit",
                "sha256": item["sha256"],
                "byte_count": item["byte_count"],
            }
            for item in commits
        ]
        for batch_digest in sorted(expected_object_names):
            object_path = self.objects / batch_digest
            nested = list(object_path.rglob("*"))
            for item in nested:
                relative = item.relative_to(self.root).as_posix()
                if item.is_symlink():
                    raise ValueError(f"NODE_CACHE_OBJECT_SYMLINK:{relative}")
                if item.is_dir() and item.relative_to(object_path).parts != ("tensors",):
                    raise ValueError(f"NODE_CACHE_UNKNOWN_OBJECT_DIRECTORY:{relative}")
                if item.is_file():
                    if not _path_is_within(item.resolve(strict=False), self.root):
                        raise ValueError("NODE_CACHE_OBJECT_PATH_ESCAPE")
                    file_record = {
                        "path": relative,
                        "kind": "object",
                        "sha256": _sha256_path(item, field_name="object"),
                        "byte_count": item.stat().st_size,
                    }
                    files.append(file_record)
            manifest_path = object_path / "manifest.json"
            if not manifest_path.is_file() or manifest_path.is_symlink():
                raise ValueError("NODE_CACHE_OBJECT_MANIFEST_MISSING")
            _state, bundle = load_tensor_bundle(object_path)
            expected_manifest_hash = str(object_refs[batch_digest]["manifest_hash"])
            if bundle.manifest_sha256 != expected_manifest_hash:
                raise ValueError("NODE_CACHE_OBJECT_MANIFEST_HASH_MISMATCH")
            object_files = [
                item for item in files if str(item["path"]).startswith(f"objects/{batch_digest}/")
            ]
            objects.append(
                {
                    "object_ref": f"objects/{batch_digest}",
                    "object_manifest_hash": expected_manifest_hash,
                    "files": sorted(object_files, key=lambda item: str(item["path"])),
                    "byte_count": sum(int(item["byte_count"]) for item in object_files),
                }
            )

        files.sort(key=lambda item: str(item["path"]))
        return {
            "commits": commits,
            "objects": objects,
            "files": files,
            "key_records": [key_records[digest] for digest in sorted(key_records)],
            "object_manifest_hashes": sorted(
                str(item["object_manifest_hash"]) for item in objects
            ),
            "commit_artifact_hashes": sorted(
                str(item["commit"]["artifact_hash"]) for item in commits  # type: ignore[index]
            ),
        }

    @staticmethod
    def _commit_evidence_payload(
        *,
        codec_id: str,
        requested_digests: Sequence[str],
        authoritative_commits: Sequence[Mapping[str, object]],
        reconciliation: Mapping[str, object],
    ) -> dict[str, object]:
        """Build the ordinary evidence body for live and sealed caches."""
        committed = set(reconciliation.get("committed_key_digests", ()))
        requested = list(requested_digests)
        payload: dict[str, object] = {
            "schema_version": "stage3-node-cache-evidence-v1",
            "codec_id": codec_id,
            "publication_protocol": (
                "immutable_tensor_bundle_then_independent_authoritative_commit"
            ),
            "requested_key_digests": requested,
            "all_requested_keys_committed": not bool(set(requested) - committed),
            "missing_key_digests": sorted(set(requested) - committed),
            "authoritative_commits": [dict(item) for item in authoritative_commits],
            "reconciliation": dict(reconciliation),
        }
        payload["evidence_hash"] = canonical_json_hash(payload)
        return payload

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

        return self._commit_evidence_payload(
            codec_id=self.codec.codec_id,
            requested_digests=requested_digests,
            authoritative_commits=authoritative_commits,
            reconciliation=reconciliation,
        )

    @staticmethod
    def _load_recovery_receipt(
        value: Mapping[str, object] | str | Path,
    ) -> tuple[dict[str, object], Path | None]:
        if isinstance(value, Mapping):
            return dict(value), None
        path = Path(value).resolve(strict=False)
        loaded = load_canonical_json(path)
        if not isinstance(loaded, Mapping):
            raise ValueError("NODE_CACHE_RECEIPT_NOT_OBJECT")
        return dict(loaded), path

    def commit_evidence_from_sealed(
        self,
        keys: Sequence[NodeCacheKey],
        sealed_receipt: Mapping[str, object] | str | Path,
        evicted_receipt: Mapping[str, object] | str | Path,
    ) -> dict[str, object]:
        """Rebuild ordinary commit evidence from a verified EVICTED fence."""
        requested: dict[str, NodeCacheKey] = {}
        for key in keys:
            if not isinstance(key, NodeCacheKey):
                raise TypeError("PersistentNodeGradientCache 只接受 NodeCacheKey")
            requested[key.digest] = key
        requested_digests = sorted(requested)
        sealed_value, sealed_path = self._load_recovery_receipt(sealed_receipt)
        evicted_value, evicted_path = self._load_recovery_receipt(evicted_receipt)
        sealed = self._validate_sealed_payload(sealed_value)
        if sealed_path is not None and sealed_path.name != str(sealed["receipt_ref"]):
            raise ValueError("NODE_CACHE_RECEIPT_REF_MISMATCH")
        evicted = self._validate_evicted_payload(
            evicted_value if evicted_path is None else evicted_path,
            sealed,
        )
        if sealed.get("cache_root_ref") != self.root.as_posix() or evicted.get("cache_root_ref") != self.root.as_posix():
            raise ValueError("NODE_CACHE_CACHE_ROOT_BINDING_MISMATCH")
        self._reload_index()
        self._cache_state_matches_seal(sealed, allow_missing=True)
        if self._cache_state_has_files(sealed) or self._index:
            raise ValueError("NODE_CACHE_EVICTED_CACHE_NOT_EMPTY")

        key_records = sealed.get("key_records")
        commits = sealed.get("commits")
        objects = sealed.get("objects")
        manifest = sealed.get("eviction_manifest")
        if not all(isinstance(item, list) for item in (key_records, commits, objects, manifest)):
            raise ValueError("NODE_CACHE_RECEIPT_COMPONENTS_INVALID")
        assert isinstance(key_records, list)
        assert isinstance(commits, list)
        assert isinstance(objects, list)
        assert isinstance(manifest, list)
        manifest_by_path: dict[str, Mapping[str, object]] = {}
        for item in manifest:
            if not isinstance(item, Mapping):
                raise ValueError("NODE_CACHE_EVICTION_FILE_INVALID")
            path = item.get("path")
            if not isinstance(path, str) or path in manifest_by_path:
                raise ValueError("NODE_CACHE_EVICTION_PATH_DUPLICATE")
            manifest_by_path[path] = item

        records: dict[str, dict[str, object]] = {}
        for item in key_records:
            if not isinstance(item, Mapping):
                raise ValueError("NODE_CACHE_KEY_RECORD_INVALID")
            key_payload = item.get("key")
            if not isinstance(key_payload, Mapping):
                raise ValueError("NODE_CACHE_KEY_RECORD_KEY_INVALID")
            key = _key_from_payload(key_payload)
            kd = _require_sha256(str(item.get("key_digest")), field_name="key_digest")
            vd = _require_sha256(str(item.get("value_digest")), field_name="value_digest")
            if key.digest != kd:
                raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
            record = {"key": _key_payload(key), "key_digest": kd, "value_digest": vd}
            if kd in records and records[kd] != record:
                raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
            records[kd] = record
        if key_records != [records[d] for d in sorted(records)]:
            raise ValueError("NODE_CACHE_KEY_RECORD_ORDER_MISMATCH")

        commit_items: list[dict[str, object]] = []
        commit_hashes: list[str] = []
        entries_by_key: dict[str, dict[str, object]] = {}
        object_by_ref: dict[str, str] = {}
        for item in commits:
            if not isinstance(item, Mapping) or not isinstance(item.get("commit"), Mapping):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_INVALID")
            path = item.get("path")
            commit = dict(item["commit"])
            if not isinstance(path, str) or Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_PATH_ESCAPE")
            if set(commit) != {"schema_version", "batch_digest", "codec_id", "entries", "object_ref", "object_manifest_hash", "artifact_hash"}:
                raise ValueError("NODE_CACHE_COMMIT_FIELDS_MISMATCH")
            if commit.get("schema_version") != "stage3-node-cache-commit-v1" or commit.get("codec_id") != self.codec.codec_id:
                raise ValueError("NODE_CACHE_CODEC_ID_MISMATCH")
            batch = _require_sha256(str(commit.get("batch_digest")), field_name="commit.batch_digest")
            if path != f"commits/{batch}.json" or commit.get("object_ref") != f"objects/{batch}":
                raise ValueError("NODE_CACHE_COMMIT_PATH_MISMATCH")
            object_hash = _require_sha256(str(commit.get("object_manifest_hash")), field_name="commit.object_manifest_hash")
            artifact = _require_sha256(str(commit.get("artifact_hash")), field_name="commit.artifact_hash")
            if canonical_json_hash({k: v for k, v in commit.items() if k != "artifact_hash"}) != artifact:
                raise ValueError("NODE_CACHE_COMMIT_HASH_MISMATCH")
            entries = commit.get("entries")
            if not isinstance(entries, list) or entries != sorted(entries, key=lambda v: str(v.get("key_digest")) if isinstance(v, Mapping) else ""):
                raise ValueError("NODE_CACHE_ENTRIES_INVALID")
            batch_entries: list[dict[str, str]] = []
            for entry in entries:
                if not isinstance(entry, Mapping) or set(entry) != {"key", "key_digest", "value_digest"}:
                    raise ValueError("NODE_CACHE_ENTRY_INVALID")
                key_payload = entry.get("key")
                if not isinstance(key_payload, Mapping):
                    raise ValueError("NODE_CACHE_ENTRY_KEY_INVALID")
                key = _key_from_payload(key_payload)
                kd = _require_sha256(str(entry.get("key_digest")), field_name="entry.key_digest")
                vd = _require_sha256(str(entry.get("value_digest")), field_name="entry.value_digest")
                if key.digest != kd:
                    raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
                record = {"key": _key_payload(key), "key_digest": kd, "value_digest": vd}
                if kd in entries_by_key and entries_by_key[kd] != record:
                    raise ValueError("NODE_CACHE_IMMUTABLE_KEY_CONFLICT")
                entries_by_key[kd] = record
                batch_entries.append({"key_digest": kd, "value_digest": vd})
            if canonical_json_hash({"codec_id": self.codec.codec_id, "entries": batch_entries}) != batch:
                raise ValueError("NODE_CACHE_BATCH_DIGEST_MISMATCH")
            m = manifest_by_path.get(path)
            if not isinstance(m, Mapping) or m.get("kind") != "commit" or m.get("sha256") != item.get("sha256") or m.get("byte_count") != item.get("byte_count"):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_BYTE_BINDING_MISMATCH")
            if batch in object_by_ref:
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_REF_DUPLICATE")
            object_by_ref[batch] = object_hash
            commit_items.append({"path": path, "commit": commit, "byte_count": item.get("byte_count"), "sha256": item.get("sha256")})
            commit_hashes.append(artifact)
        if commits != sorted(commit_items, key=lambda v: str(v["path"])):
            raise ValueError("NODE_CACHE_RECEIPT_COMMIT_ORDER_MISMATCH")
        if sorted(commit_hashes) != sealed.get("commit_artifact_hashes") or entries_by_key != records:
            raise ValueError("NODE_CACHE_RECEIPT_KEY_RECORD_SET_MISMATCH")

        object_items: list[dict[str, object]] = []
        object_hashes: list[str] = []
        object_files: set[str] = set()
        for item in objects:
            if not isinstance(item, Mapping) or not isinstance(item.get("object_ref"), str):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_INVALID")
            ref = str(item["object_ref"])
            batch = Path(ref).name
            if ref != f"objects/{batch}" or batch not in object_by_ref or object_by_ref[batch] != item.get("object_manifest_hash"):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_REF_MISMATCH")
            nested = item.get("files")
            if not isinstance(nested, list) or nested != sorted(nested, key=lambda v: str(v.get("path")) if isinstance(v, Mapping) else ""):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILES_INVALID")
            nested_paths: set[str] = set()
            nested_bytes = 0
            for file_item in nested:
                if not isinstance(file_item, Mapping):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_INVALID")
                file_path = file_item.get("path")
                if not isinstance(file_path, str) or not file_path.startswith(f"{ref}/") or file_path in nested_paths:
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_BINDING_MISMATCH")
                m = manifest_by_path.get(file_path)
                if not isinstance(m, Mapping) or m.get("kind") != "object" or dict(m) != dict(file_item):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_BINDING_MISMATCH")
                nested_paths.add(file_path)
                object_files.add(file_path)
                nested_bytes += int(file_item.get("byte_count", -1))
            if nested_bytes != item.get("byte_count"):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_BYTE_COUNT_MISMATCH")
            object_items.append({"object_ref": ref, "object_manifest_hash": item.get("object_manifest_hash"), "files": [dict(v) for v in nested], "byte_count": item.get("byte_count")})
            object_hashes.append(str(item.get("object_manifest_hash")))
        if objects != sorted(object_items, key=lambda v: str(v["object_ref"])):
            raise ValueError("NODE_CACHE_RECEIPT_OBJECT_ORDER_MISMATCH")
        if sorted(object_hashes) != sealed.get("object_manifest_hashes"):
            raise ValueError("NODE_CACHE_RECEIPT_OBJECT_HASH_SET_MISMATCH")
        expected_object_files = {str(v.get("path")) for v in manifest if isinstance(v, Mapping) and v.get("kind") == "object"}
        expected_paths = {str(v.get("path")) for v in commits if isinstance(v, Mapping)} | object_files
        if object_files != expected_object_files or set(manifest_by_path) != expected_paths:
            raise ValueError("NODE_CACHE_RECEIPT_PATH_SET_MISMATCH")

        authoritative: list[dict[str, object]] = []
        for item in commit_items:
            commit = item["commit"]
            assert isinstance(commit, Mapping)
            matched = sorted({str(v["key_digest"]) for v in commit["entries"] if isinstance(v, Mapping)}.intersection(requested_digests))
            if matched:
                authoritative.append({"commit_ref": item["path"], "commit_artifact_hash": commit["artifact_hash"], "object_ref": commit["object_ref"], "object_manifest_hash": commit["object_manifest_hash"], "key_digests": matched})
        return self._commit_evidence_payload(
            codec_id=self.codec.codec_id,
            requested_digests=requested_digests,
            authoritative_commits=authoritative,
            reconciliation={"committed_key_digests": sorted(records), "orphan_objects": []},
        )

    def _verify_downstream_raw_shard(
        self,
        raw_shard_ref: str | Path,
        raw_shard_hash: str,
    ) -> tuple[Path, str]:
        """Verify a canonical Stage3 raw shard before binding it into a seal.

        Stage3's domain ``shard_hash`` is the canonical JSON body hash, not the
        filesystem digest.  The latter is returned only as supplemental byte-level
        evidence so retention can detect any byte drift without confusing the two
        identities.
        """

        expected = _require_sha256(
            raw_shard_hash, field_name="downstream_raw_shard_hash"
        )
        path = _resolved_cache_path(
            raw_shard_ref, field_name="downstream_raw_shard_ref"
        )
        if _path_is_within(path, self.root):
            raise ValueError("NODE_CACHE_DOWNSTREAM_REF_INSIDE_CACHE_ROOT")
        raw = load_canonical_json(path)
        if not isinstance(raw, Mapping) or raw.get("schema_version") != (
            "stage3-formal-raw-shard-v1"
        ):
            raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_SCHEMA_INVALID")
        actual = raw.get("artifact_hash")
        if not isinstance(actual, str):
            raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_HASH_MISMATCH")
        _require_sha256(actual, field_name="downstream_raw_shard_artifact_hash")
        if canonical_json_hash(
            {key: item for key, item in raw.items() if key != "artifact_hash"}
        ) != actual:
            raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_HASH_MISMATCH")
        if actual != expected:
            raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_HASH_MISMATCH")
        return path, _sha256_path(path, field_name="downstream_raw_shard_ref")

    @staticmethod
    def _receipt_payload_without_hash(
        receipt: Mapping[str, object], field_name: str
    ) -> dict[str, object]:
        value = dict(receipt)
        digest = value.pop(field_name, None)
        if not isinstance(digest, str) or canonical_json_hash(value) != digest:
            raise ValueError(f"NODE_CACHE_{field_name.upper()}_MISMATCH")
        return value

    @staticmethod
    def _validate_sealed_payload(receipt: Mapping[str, object]) -> dict[str, object]:
        if receipt.get("schema_version") != "stage3-node-cache-seal-v1":
            raise ValueError("NODE_CACHE_SEAL_SCHEMA_MISMATCH")
        if receipt.get("state") != "SEALED":
            raise ValueError("NODE_CACHE_SEAL_STATE_MISMATCH")
        PersistentNodeGradientCache._receipt_payload_without_hash(
            receipt, "receipt_hash"
        )
        for field_name in ("scope", "unit_id"):
            _require_identifier(str(receipt.get(field_name)), field_name=field_name)
        for field_name in ("plan_hash", "run_config_hash"):
            _require_sha256(str(receipt.get(field_name)), field_name=field_name)
        _require_sha256(str(receipt.get("receipt_id")), field_name="receipt_id")
        receipt_ref = receipt.get("receipt_ref")
        if receipt_ref != f"{receipt['receipt_id']}.SEALED.json":
            raise ValueError("NODE_CACHE_RECEIPT_REF_MISMATCH")
        cache_root_ref = receipt.get("cache_root_ref")
        if not isinstance(cache_root_ref, str) or not cache_root_ref:
            raise ValueError("NODE_CACHE_CACHE_ROOT_REF_INVALID")
        raw_ref = receipt.get("downstream_raw_shard_ref")
        if not isinstance(raw_ref, str) or not raw_ref:
            raise ValueError("NODE_CACHE_DOWNSTREAM_REF_INVALID")
        _require_sha256(
            str(receipt.get("downstream_raw_shard_hash")),
            field_name="downstream_raw_shard_hash",
        )
        _require_sha256(
            str(receipt.get("downstream_raw_shard_file_sha256")),
            field_name="downstream_raw_shard_file_sha256",
        )
        identity = {
            "cache_root_ref": cache_root_ref,
            "scope": receipt["scope"],
            "unit_id": receipt["unit_id"],
            "plan_hash": receipt["plan_hash"],
            "run_config_hash": receipt["run_config_hash"],
            "downstream_raw_shard_ref": raw_ref,
            "downstream_raw_shard_hash": receipt["downstream_raw_shard_hash"],
        }
        if canonical_json_hash(identity) != receipt["receipt_id"]:
            raise ValueError("NODE_CACHE_RECEIPT_ID_MISMATCH")
        key_records = receipt.get("key_records")
        if not isinstance(key_records, list):
            raise ValueError("NODE_CACHE_KEY_RECORDS_INVALID")
        seen_keys: set[str] = set()
        for item in key_records:
            if not isinstance(item, Mapping):
                raise ValueError("NODE_CACHE_KEY_RECORD_INVALID")
            key_digest = _require_sha256(
                str(item.get("key_digest")), field_name="key_digest"
            )
            value_digest = _require_sha256(
                str(item.get("value_digest")), field_name="value_digest"
            )
            key_payload = item.get("key")
            if not isinstance(key_payload, Mapping):
                raise ValueError("NODE_CACHE_KEY_RECORD_KEY_INVALID")
            if _key_from_payload(key_payload).digest != key_digest:
                raise ValueError("NODE_CACHE_KEY_DIGEST_MISMATCH")
            if key_digest in seen_keys:
                raise ValueError("NODE_CACHE_KEY_RECORD_DUPLICATE")
            seen_keys.add(key_digest)
        if receipt.get("key_digests") != sorted(seen_keys):
            raise ValueError("NODE_CACHE_KEY_DIGEST_SET_MISMATCH")
        value_digests = receipt.get("value_digests")
        if not isinstance(value_digests, list) or value_digests != sorted(
            {str(item["value_digest"]) for item in key_records}
        ):
            raise ValueError("NODE_CACHE_VALUE_DIGEST_SET_MISMATCH")
        for field_name in ("commit_artifact_hashes", "object_manifest_hashes"):
            hashes = receipt.get(field_name)
            if not isinstance(hashes, list) or hashes != sorted(set(hashes)):
                raise ValueError(f"NODE_CACHE_{field_name.upper()}_INVALID")
            for digest in hashes:
                _require_sha256(str(digest), field_name=field_name)
        manifest = receipt.get("eviction_manifest")
        if not isinstance(manifest, list):
            raise ValueError("NODE_CACHE_EVICTION_MANIFEST_INVALID")
        manifest_paths: set[str] = set()
        manifest_bytes = 0
        manifest_order: list[str] = []
        for item in manifest:
            if not isinstance(item, Mapping):
                raise ValueError("NODE_CACHE_EVICTION_FILE_INVALID")
            relative_text = item.get("path")
            if not isinstance(relative_text, str):
                raise ValueError("NODE_CACHE_EVICTION_PATH_INVALID")
            relative = Path(relative_text)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in {"objects", "commits"}
            ):
                raise ValueError("NODE_CACHE_EVICTION_PATH_ESCAPE")
            if relative_text in manifest_paths:
                raise ValueError("NODE_CACHE_EVICTION_PATH_DUPLICATE")
            manifest_paths.add(relative_text)
            manifest_order.append(relative_text)
            if item.get("kind") not in {"object", "commit"}:
                raise ValueError("NODE_CACHE_EVICTION_KIND_INVALID")
            if item["kind"] == "commit" and not relative_text.startswith("commits/"):
                raise ValueError("NODE_CACHE_EVICTION_KIND_PATH_MISMATCH")
            if item["kind"] == "object" and not relative_text.startswith("objects/"):
                raise ValueError("NODE_CACHE_EVICTION_KIND_PATH_MISMATCH")
            _require_sha256(str(item.get("sha256")), field_name="eviction.sha256")
            byte_count = item.get("byte_count")
            if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
                raise ValueError("NODE_CACHE_EVICTION_BYTE_COUNT_INVALID")
            manifest_bytes += byte_count
        if manifest_order != sorted(manifest_order):
            raise ValueError("NODE_CACHE_EVICTION_MANIFEST_NOT_SORTED")
        byte_counts = receipt.get("byte_counts")
        if not isinstance(byte_counts, Mapping):
            raise ValueError("NODE_CACHE_BYTE_COUNTS_INVALID")
        if byte_counts.get("total_bytes") != manifest_bytes:
            raise ValueError("NODE_CACHE_TOTAL_BYTE_COUNT_MISMATCH")
        if byte_counts.get("files") != {
            str(item["path"]): item["byte_count"] for item in manifest
        }:
            raise ValueError("NODE_CACHE_FILE_BYTE_COUNTS_MISMATCH")
        commits = receipt.get("commits")
        objects = receipt.get("objects")
        if not isinstance(commits, list) or not isinstance(objects, list):
            raise ValueError("NODE_CACHE_RECEIPT_COMPONENTS_INVALID")
        commit_hashes: list[str] = []
        commit_bytes = 0
        commit_paths: set[str] = set()
        for item in commits:
            if not isinstance(item, Mapping) or not isinstance(item.get("commit"), Mapping):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_INVALID")
            commit_path = item.get("path")
            if (
                not isinstance(commit_path, str)
                or Path(commit_path).is_absolute()
                or ".." in Path(commit_path).parts
                or not commit_path.startswith("commits/")
            ):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_PATH_ESCAPE")
            if commit_path in commit_paths:
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_PATH_DUPLICATE")
            commit_paths.add(commit_path)
            commit = item["commit"]
            artifact_hash = _require_sha256(
                str(commit.get("artifact_hash")),
                field_name="commit.artifact_hash",
            )
            commit_file_hash = _require_sha256(
                str(item.get("sha256")), field_name="receipt.commit.sha256"
            )
            commit_hashes.append(artifact_hash)
            commit_byte_count = item.get("byte_count")
            if (
                isinstance(commit_byte_count, bool)
                or not isinstance(commit_byte_count, int)
                or commit_byte_count < 0
            ):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_BYTE_COUNT_INVALID")
            commit_manifest_item = next(
                (
                    candidate
                    for candidate in manifest
                    if isinstance(candidate, Mapping)
                    and candidate.get("path") == commit_path
                ),
                None,
            )
            if (
                not isinstance(commit_manifest_item, Mapping)
                or commit_manifest_item.get("byte_count") != commit_byte_count
                or commit_manifest_item.get("sha256") != commit_file_hash
            ):
                raise ValueError("NODE_CACHE_RECEIPT_COMMIT_BYTE_BINDING_MISMATCH")
            commit_bytes += commit_byte_count
        if sorted(commit_hashes) != receipt.get("commit_artifact_hashes"):
            raise ValueError("NODE_CACHE_RECEIPT_COMMIT_HASH_SET_MISMATCH")
        object_hashes: list[str] = []
        object_bytes = 0
        object_refs: set[str] = set()
        nested_object_paths: set[str] = set()
        for item in objects:
            if not isinstance(item, Mapping):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_INVALID")
            object_ref = item.get("object_ref")
            object_relative = Path(str(object_ref))
            if (
                not isinstance(object_ref, str)
                or object_relative.is_absolute()
                or ".." in object_relative.parts
                or len(object_relative.parts) != 2
                or object_relative.parts[0] != "objects"
            ):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_PATH_ESCAPE")
            if object_ref in object_refs:
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_REF_DUPLICATE")
            object_refs.add(object_ref)
            manifest_hash = _require_sha256(
                str(item.get("object_manifest_hash")),
                field_name="object.object_manifest_hash",
            )
            object_hashes.append(manifest_hash)
            nested_files = item.get("files")
            if not isinstance(nested_files, list):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILES_INVALID")
            object_byte_count = item.get("byte_count")
            if (
                isinstance(object_byte_count, bool)
                or not isinstance(object_byte_count, int)
                or object_byte_count < 0
            ):
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_BYTE_COUNT_INVALID")
            nested_byte_count = 0
            for nested in nested_files:
                if not isinstance(nested, Mapping):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_INVALID")
                relative_text = nested.get("path")
                if relative_text not in manifest_paths or nested.get("kind") != "object":
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_NOT_MANIFESTED")
                if not isinstance(relative_text, str) or not relative_text.startswith(
                    f"{object_ref}/"
                ):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_BINDING_MISMATCH")
                if relative_text in nested_object_paths:
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_DUPLICATE")
                nested_object_paths.add(str(relative_text))
                nested_hash = _require_sha256(
                    str(nested.get("sha256")), field_name="object.sha256"
                )
                nested_size = nested.get("byte_count")
                if (
                    isinstance(nested_size, bool)
                    or not isinstance(nested_size, int)
                    or nested_size < 0
                ):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_BYTE_COUNT_INVALID")
                object_manifest_item = next(
                    (
                        candidate
                        for candidate in manifest
                        if isinstance(candidate, Mapping)
                        and candidate.get("path") == relative_text
                    ),
                    None,
                )
                if (
                    not isinstance(object_manifest_item, Mapping)
                    or object_manifest_item.get("byte_count") != nested_size
                    or object_manifest_item.get("sha256") != nested_hash
                ):
                    raise ValueError("NODE_CACHE_RECEIPT_OBJECT_FILE_BINDING_MISMATCH")
                nested_byte_count += nested_size
            if nested_byte_count != object_byte_count:
                raise ValueError("NODE_CACHE_RECEIPT_OBJECT_BYTE_COUNT_MISMATCH")
            object_bytes += object_byte_count
        manifested_commit_paths = {
            str(item["path"])
            for item in manifest
            if isinstance(item, Mapping) and item.get("kind") == "commit"
        }
        if commit_paths != manifested_commit_paths:
            raise ValueError("NODE_CACHE_RECEIPT_COMMIT_PATH_SET_MISMATCH")
        manifested_object_paths = {
            str(item["path"])
            for item in manifest
            if isinstance(item, Mapping) and item.get("kind") == "object"
        }
        if nested_object_paths != manifested_object_paths:
            raise ValueError("NODE_CACHE_RECEIPT_OBJECT_PATH_SET_MISMATCH")
        if sorted(object_hashes) != receipt.get("object_manifest_hashes"):
            raise ValueError("NODE_CACHE_RECEIPT_OBJECT_HASH_SET_MISMATCH")
        if (
            byte_counts.get("commit_bytes") != commit_bytes
            or byte_counts.get("object_bytes") != object_bytes
        ):
            raise ValueError("NODE_CACHE_COMPONENT_BYTE_COUNT_MISMATCH")
        return dict(receipt)

    def _cache_state_matches_seal(
        self, receipt: Mapping[str, object], *, allow_missing: bool
    ) -> list[dict[str, object]]:
        """Validate current files against a sealed manifest.

        ``allow_missing`` is used only after an immutable SEALED receipt exists:
        missing listed files can then mean that a process crashed during the delete
        phase.  Every file that is still present must retain its sealed hash.
        """

        if (self.root / "writer.lock").exists() or (self.root / "writer.lock").is_symlink():
            raise ValueError("NODE_CACHE_WRITER_LOCK_PRESENT")
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("NODE_CACHE_ROOT_INVALID")
        children = {child.name: child for child in self.root.iterdir()}
        if set(children) != {"objects", "commits"}:
            raise ValueError("NODE_CACHE_UNKNOWN_ROOT_ENTRIES")
        for name in ("objects", "commits"):
            child = children[name]
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"NODE_CACHE_{name.upper()}_ROOT_INVALID")
        manifest = receipt["eviction_manifest"]
        assert isinstance(manifest, list)
        expected = {str(item["path"]): item for item in manifest if isinstance(item, Mapping)}
        object_refs = {
            str(entry["object_ref"])
            for entry in receipt.get("objects", [])
            if isinstance(entry, Mapping)
        }
        actual: dict[str, Path] = {}
        for base_name in ("objects", "commits"):
            base = self.root / base_name
            for item in base.rglob("*"):
                relative = item.relative_to(self.root).as_posix()
                if item.is_symlink():
                    raise ValueError(f"NODE_CACHE_EVICTION_SYMLINK:{relative}")
                if item.is_file():
                    if relative not in expected:
                        raise ValueError(f"NODE_CACHE_UNKNOWN_EVICTION_FILE:{relative}")
                    actual[relative] = item
                elif item.is_dir():
                    # The only valid directories are the two roots, an object
                    # directory, and (when tensors exist) its tensors directory.
                    parts = Path(relative).parts
                    valid = (
                        len(parts) == 2
                        and parts[0] == "objects"
                        and relative in object_refs
                        or len(parts) == 3
                        and parts[0] == "objects"
                        and parts[2] == "tensors"
                        and "/".join(parts[:2]) in object_refs
                    )
                    if not valid:
                        raise ValueError(f"NODE_CACHE_UNKNOWN_EVICTION_DIRECTORY:{relative}")
        if not allow_missing and set(actual) != set(expected):
            raise ValueError("NODE_CACHE_EVICTION_FILE_SET_MISMATCH")
        for relative, item in actual.items():
            expected_item = expected[relative]
            path = item.resolve(strict=False)
            if not _path_is_within(path, self.root):
                raise ValueError("NODE_CACHE_EVICTION_PATH_ESCAPE")
            if item.stat().st_size != expected_item["byte_count"]:
                raise ValueError(f"NODE_CACHE_EVICTION_BYTE_DRIFT:{relative}")
            if _sha256_path(item, field_name="eviction file") != expected_item["sha256"]:
                raise ValueError(f"NODE_CACHE_EVICTION_HASH_DRIFT:{relative}")
        return [dict(item) for item in manifest if isinstance(item, Mapping)]

    def _delete_sealed_files(self, receipt: Mapping[str, object]) -> None:
        manifest = self._cache_state_matches_seal(receipt, allow_missing=True)
        expected = {str(item["path"]): item for item in manifest}
        for relative in sorted(expected):
            path = self.root / relative
            if not path.exists():
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"NODE_CACHE_EVICTION_PATH_INVALID:{relative}")
            # Validate immediately before unlinking, so a race or tamper cannot turn
            # an immutable receipt into authorization for an unrelated file.
            item = expected[relative]
            if path.stat().st_size != item["byte_count"] or _sha256_path(
                path, field_name="eviction file"
            ) != item["sha256"]:
                raise ValueError(f"NODE_CACHE_EVICTION_HASH_DRIFT:{relative}")
            if not _path_is_within(path.resolve(strict=False), self.root):
                raise ValueError("NODE_CACHE_EVICTION_PATH_ESCAPE")
            path.unlink()
        # Remove only empty directories named by the sealed object refs.  Never use
        # recursive deletion: an unknown file must remain visible and fail closed.
        object_refs = {
            str(item["object_ref"])
            for item in receipt.get("objects", [])
            if isinstance(item, Mapping)
        }
        for relative in sorted(object_refs, reverse=True):
            object_path = self.root / relative
            if object_path.exists():
                if object_path.is_symlink() or not object_path.is_dir():
                    raise ValueError("NODE_CACHE_EVICTION_OBJECT_PATH_INVALID")
                tensors = object_path / "tensors"
                if tensors.exists():
                    if tensors.is_symlink() or not tensors.is_dir() or any(tensors.iterdir()):
                        raise ValueError("NODE_CACHE_UNKNOWN_OBJECT_ENTRY")
                    tensors.rmdir()
                if any(object_path.iterdir()):
                    raise ValueError("NODE_CACHE_UNKNOWN_OBJECT_ENTRY")
                object_path.rmdir()
        # Re-check after deletion.  This catches writer.lock and any unknown file;
        # those are never silently removed.
        self._cache_state_matches_seal(receipt, allow_missing=True)

    @staticmethod
    def _receipt_path(
        receipt_root: Path, receipt_ref: str | Path, *, suffix: str
    ) -> Path:
        candidate = Path(receipt_ref)
        path = candidate if candidate.is_absolute() else receipt_root / candidate
        path = path.resolve(strict=False)
        if not _path_is_within(path, receipt_root) or path.name != suffix:
            raise ValueError("NODE_CACHE_RECEIPT_PATH_ESCAPE")
        return path

    def seal(
        self,
        *,
        scope: str,
        unit_id: str,
        plan_hash: str,
        run_config_hash: str,
        downstream_raw_shard_ref: str | Path,
        downstream_raw_shard_hash: str,
        receipt_root: str | Path | None = None,
    ) -> dict[str, object]:
        """Write an immutable SEALED receipt after fully verifying the unit cache.

        The downstream shard is verified from bytes at ``downstream_raw_shard_ref``;
        a caller-provided boolean or an unverified logical reference is never enough.
        This method does not delete cache data.  ``seal_and_evict`` combines it with
        the second phase, while ``finalize_eviction`` is safe to call after a crash.
        """

        scope = _require_identifier(scope, field_name="scope")
        unit_id = _require_identifier(unit_id, field_name="unit_id")
        plan_hash = _require_sha256(plan_hash, field_name="plan_hash")
        run_config_hash = _require_sha256(run_config_hash, field_name="run_config_hash")
        receipt_base = self._receipt_root_for(receipt_root)
        _, raw_file_hash = self._verify_downstream_raw_shard(
            downstream_raw_shard_ref, downstream_raw_shard_hash
        )
        raw_ref = str(downstream_raw_shard_ref)
        raw_hash = _require_sha256(
            downstream_raw_shard_hash, field_name="downstream_raw_shard_hash"
        )
        identity = {
            "cache_root_ref": self.root.as_posix(),
            "scope": scope,
            "unit_id": unit_id,
            "plan_hash": plan_hash,
            "run_config_hash": run_config_hash,
            "downstream_raw_shard_ref": raw_ref,
            "downstream_raw_shard_hash": raw_hash,
        }
        receipt_id = canonical_json_hash(identity)
        sealed_name = f"{receipt_id}.SEALED.json"
        sealed_path = self._receipt_path(receipt_base, sealed_name, suffix=sealed_name)
        evicted_name = f"{receipt_id}.EVICTED.json"
        evicted_path = self._receipt_path(receipt_base, evicted_name, suffix=evicted_name)

        if sealed_path.exists() or sealed_path.is_symlink():
            existing = load_canonical_json(sealed_path)
            if not isinstance(existing, Mapping):
                raise ValueError("NODE_CACHE_SEAL_NOT_OBJECT")
            validated = self._validate_sealed_payload(existing)
            if validated.get("receipt_id") != receipt_id:
                raise ValueError("NODE_CACHE_RECEIPT_ID_MISMATCH")
            if validated.get("downstream_raw_shard_file_sha256") != raw_file_hash:
                raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_BYTE_HASH_MISMATCH")
            if evicted_path.exists() or evicted_path.is_symlink():
                self._validate_evicted_payload(evicted_path, validated)
                self._cache_state_matches_seal(validated, allow_missing=True)
                if self._cache_state_has_files(validated):
                    raise ValueError("NODE_CACHE_EVICTED_CACHE_NOT_EMPTY")
            else:
                self._cache_state_matches_seal(validated, allow_missing=True)
            try:
                self._activate_memoization(
                    validated, sealed_path, validate_cache=False
                )
            except (OSError, TypeError, ValueError):
                self._disable_memoization_unlocked()
            return dict(validated)

        inventory = self._cache_inventory()
        key_records = inventory["key_records"]
        assert isinstance(key_records, list)
        for record in key_records:
            assert isinstance(record, Mapping)
            key_payload = record["key"]
            assert isinstance(key_payload, Mapping)
            if key_payload.get("path_unit_id") != unit_id:
                raise ValueError("NODE_CACHE_UNIT_BINDING_MISMATCH")
        files = inventory["files"]
        assert isinstance(files, list)
        commit_bytes = sum(
            int(item["byte_count"])
            for item in files
            if isinstance(item, Mapping) and item.get("kind") == "commit"
        )
        object_bytes = sum(
            int(item["byte_count"])
            for item in files
            if isinstance(item, Mapping) and item.get("kind") == "object"
        )
        manifest_files = [dict(item) for item in files if isinstance(item, Mapping)]
        byte_counts = {
            "commit_bytes": commit_bytes,
            "object_bytes": object_bytes,
            "total_bytes": commit_bytes + object_bytes,
            "files": {str(item["path"]): int(item["byte_count"]) for item in manifest_files},
        }
        receipt: dict[str, object] = {
            "schema_version": "stage3-node-cache-seal-v1",
            "state": "SEALED",
            "receipt_id": receipt_id,
            "receipt_ref": sealed_name,
            "cache_root_ref": self.root.as_posix(),
            "scope": scope,
            "unit_id": unit_id,
            "plan_hash": plan_hash,
            "run_config_hash": run_config_hash,
            "downstream_raw_shard_ref": raw_ref,
            "downstream_raw_shard_hash": raw_hash,
            "downstream_raw_shard_file_sha256": raw_file_hash,
            "key_digests": sorted(str(item["key_digest"]) for item in key_records),
            "value_digests": sorted(
                {str(item["value_digest"]) for item in key_records}
            ),
            "key_records": key_records,
            "commit_artifact_hashes": inventory["commit_artifact_hashes"],
            "object_manifest_hashes": inventory["object_manifest_hashes"],
            "byte_counts": byte_counts,
            "eviction_manifest": manifest_files,
            "commits": inventory["commits"],
            "objects": inventory["objects"],
        }
        receipt["receipt_hash"] = canonical_json_hash(receipt)
        _write_immutable_canonical_json(sealed_path, receipt)
        try:
            self._activate_memoization(receipt, sealed_path, validate_cache=False)
        except (OSError, TypeError, ValueError):
            self._disable_memoization_unlocked()
        return receipt

    @staticmethod
    def _validate_evicted_payload(
        value_or_path: Mapping[str, object] | str | Path,
        sealed: Mapping[str, object],
    ) -> dict[str, object]:
        """Validate an EVICTED tombstone and its SEALED binding.

        A recovery caller may already have loaded an immutable tombstone from a
        streaming receipt. Validate that mapping directly instead of resolving
        its reference relative to the process cwd. Path callers retain the
        on-disk validator and also bind the filename to the payload.
        """
        path: Path | None = None
        if isinstance(value_or_path, Mapping):
            value = value_or_path
        else:
            path = Path(value_or_path)
            loaded = load_canonical_json(path)
            if not isinstance(loaded, Mapping):
                raise ValueError("NODE_CACHE_EVICTED_NOT_OBJECT")
            value = loaded
        if value.get("schema_version") != "stage3-node-cache-evicted-v1":
            raise ValueError("NODE_CACHE_EVICTED_SCHEMA_MISMATCH")
        if value.get("state") != "EVICTED":
            raise ValueError("NODE_CACHE_EVICTED_STATE_MISMATCH")
        PersistentNodeGradientCache._receipt_payload_without_hash(
            value, "tombstone_hash"
        )
        expected_fields = {
            "schema_version", "state", "receipt_id", "tombstone_ref",
            "sealed_receipt_ref", "sealed_receipt_hash", "cache_root_ref",
            "scope", "unit_id", "plan_hash", "run_config_hash",
            "downstream_raw_shard_ref", "downstream_raw_shard_hash",
            "downstream_raw_shard_file_sha256", "key_digests",
            "value_digests", "commit_artifact_hashes",
            "object_manifest_hashes", "byte_counts", "eviction_manifest",
            "deleted_bytes", "tombstone_hash",
        }
        if set(value) != expected_fields:
            raise ValueError("NODE_CACHE_EVICTED_FIELDS_MISMATCH")
        for field_name in (
            "receipt_id",
            "sealed_receipt_hash",
            "downstream_raw_shard_hash",
            "downstream_raw_shard_file_sha256",
            "plan_hash",
            "run_config_hash",
        ):
            _require_sha256(str(value.get(field_name)), field_name=field_name)
        if value.get("receipt_id") != sealed.get("receipt_id"):
            raise ValueError("NODE_CACHE_EVICTED_RECEIPT_BINDING_MISMATCH")
        if value.get("sealed_receipt_hash") != sealed.get("receipt_hash"):
            raise ValueError("NODE_CACHE_EVICTED_SEAL_HASH_MISMATCH")
        expected_tombstone_ref = f"{sealed['receipt_id']}.EVICTED.json"
        if value.get("tombstone_ref") != expected_tombstone_ref:
            raise ValueError("NODE_CACHE_EVICTED_TOMBSTONE_REF_MISMATCH")
        if path is not None and path.name != value.get("tombstone_ref"):
            raise ValueError("NODE_CACHE_EVICTED_TOMBSTONE_REF_MISMATCH")
        if value.get("sealed_receipt_ref") != sealed.get("receipt_ref"):
            raise ValueError("NODE_CACHE_EVICTED_SEALED_REF_MISMATCH")
        for field_name in (
            "cache_root_ref",
            "scope",
            "unit_id",
            "plan_hash",
            "run_config_hash",
            "downstream_raw_shard_ref",
            "downstream_raw_shard_hash",
            "downstream_raw_shard_file_sha256",
        ):
            if value.get(field_name) != sealed.get(field_name):
                raise ValueError("NODE_CACHE_EVICTED_BINDING_MISMATCH")
        for field_name in (
            "key_digests", "value_digests", "commit_artifact_hashes",
            "object_manifest_hashes", "byte_counts", "eviction_manifest",
        ):
            if value.get(field_name) != sealed.get(field_name):
                raise ValueError("NODE_CACHE_EVICTED_BINDING_MISMATCH")
        byte_counts = sealed.get("byte_counts")
        if (
            not isinstance(byte_counts, Mapping)
            or value.get("deleted_bytes") != byte_counts.get("total_bytes")
        ):
            raise ValueError("NODE_CACHE_EVICTED_DELETED_BYTES_MISMATCH")
        return dict(value)

    def finalize_eviction(
        self,
        receipt_ref: str | Path | Mapping[str, object] | None = None,
        *,
        receipt_root: str | Path | None = None,
        receipt_id: str | None = None,
    ) -> dict[str, object]:
        """Complete the delete phase for a SEALED receipt after interruption."""

        # Finalization is a persistent lifecycle boundary. Clear before any
        # validation/deletion so every failure path cannot expose stale templates.
        with self._lock:
            self._disable_memoization_unlocked()

        receipt_base = self._receipt_root_for(receipt_root)
        if isinstance(receipt_ref, Mapping):
            receipt_ref = receipt_ref.get("receipt_ref")  # type: ignore[assignment]
        if receipt_ref is None:
            if receipt_id is None:
                raise ValueError("NODE_CACHE_SEALED_RECEIPT_REQUIRED")
            receipt_id = _require_sha256(receipt_id, field_name="receipt_id")
            receipt_ref = f"{receipt_id}.SEALED.json"
        sealed_path = self._receipt_path(
            receipt_base, receipt_ref, suffix=Path(receipt_ref).name
        )
        if not sealed_path.name.endswith(".SEALED.json"):
            raise ValueError("NODE_CACHE_SEALED_RECEIPT_REQUIRED")
        value = load_canonical_json(sealed_path)
        if not isinstance(value, Mapping):
            raise ValueError("NODE_CACHE_SEAL_NOT_OBJECT")
        sealed = self._validate_sealed_payload(value)
        if sealed.get("cache_root_ref") != self.root.as_posix():
            raise ValueError("NODE_CACHE_CACHE_ROOT_BINDING_MISMATCH")
        _, raw_file_hash = self._verify_downstream_raw_shard(
            str(sealed["downstream_raw_shard_ref"]),
            str(sealed["downstream_raw_shard_hash"]),
        )
        if sealed.get("downstream_raw_shard_file_sha256") != raw_file_hash:
            raise ValueError("NODE_CACHE_DOWNSTREAM_RAW_SHARD_BYTE_HASH_MISMATCH")
        evicted_name = sealed_path.name.removesuffix(".SEALED.json") + ".EVICTED.json"
        evicted_path = self._receipt_path(
            receipt_base, evicted_name, suffix=evicted_name
        )
        if evicted_path.exists() or evicted_path.is_symlink():
            tombstone = self._validate_evicted_payload(evicted_path, sealed)
            self._cache_state_matches_seal(sealed, allow_missing=True)
            if self._cache_state_has_files(sealed):
                raise ValueError("NODE_CACHE_EVICTED_CACHE_NOT_EMPTY")
            with self._lock:
                self._disable_memoization_unlocked()
            return tombstone

        self._cache_state_matches_seal(sealed, allow_missing=True)
        self._delete_sealed_files(sealed)
        with self._lock:
            self._disable_memoization_unlocked()
        if self._cache_state_has_files(sealed):
            raise ValueError("NODE_CACHE_EVICTION_INCOMPLETE")
        tombstone: dict[str, object] = {
            "schema_version": "stage3-node-cache-evicted-v1",
            "state": "EVICTED",
            "receipt_id": sealed["receipt_id"],
            "tombstone_ref": evicted_name,
            "sealed_receipt_ref": sealed["receipt_ref"],
            "sealed_receipt_hash": sealed["receipt_hash"],
            "cache_root_ref": sealed["cache_root_ref"],
            "scope": sealed["scope"],
            "unit_id": sealed["unit_id"],
            "plan_hash": sealed["plan_hash"],
            "run_config_hash": sealed["run_config_hash"],
            "downstream_raw_shard_ref": sealed["downstream_raw_shard_ref"],
            "downstream_raw_shard_hash": sealed["downstream_raw_shard_hash"],
            "downstream_raw_shard_file_sha256": sealed[
                "downstream_raw_shard_file_sha256"
            ],
            "key_digests": sealed["key_digests"],
            "value_digests": sealed["value_digests"],
            "commit_artifact_hashes": sealed["commit_artifact_hashes"],
            "object_manifest_hashes": sealed["object_manifest_hashes"],
            "byte_counts": sealed["byte_counts"],
            "eviction_manifest": sealed["eviction_manifest"],
            "deleted_bytes": sealed["byte_counts"]["total_bytes"],  # type: ignore[index]
        }
        tombstone["tombstone_hash"] = canonical_json_hash(tombstone)
        _write_immutable_canonical_json(evicted_path, tombstone)
        return tombstone

    def _cache_state_has_files(self, receipt: Mapping[str, object]) -> bool:
        manifest = receipt.get("eviction_manifest")
        if not isinstance(manifest, list):
            raise ValueError("NODE_CACHE_EVICTION_MANIFEST_INVALID")
        for item in manifest:
            if isinstance(item, Mapping):
                path = self.root / str(item["path"])
                if path.exists():
                    return True
        return False

    def evict(
        self,
        receipt_ref: str | Path | Mapping[str, object] | NodeCacheKey | None = None,
        *,
        receipt_root: str | Path | None = None,
        receipt_id: str | None = None,
    ) -> dict[str, object]:
        """Alias for recovery, or evict one in-process memo entry."""

        if isinstance(receipt_ref, NodeCacheKey):
            if receipt_root is not None or receipt_id is not None:
                raise ValueError("NODE_CACHE_MEMO_EVICT_ARGS_INVALID")
            self.evict_memo(receipt_ref)
            return {"state": "MEMO_EVICTED", "key_digest": receipt_ref.digest}
        return self.finalize_eviction(
            receipt_ref, receipt_root=receipt_root, receipt_id=receipt_id
        )

    def seal_and_evict(
        self,
        *,
        scope: str,
        unit_id: str,
        plan_hash: str,
        run_config_hash: str,
        downstream_raw_shard_ref: str | Path,
        downstream_raw_shard_hash: str,
        receipt_root: str | Path | None = None,
    ) -> dict[str, object]:
        """Execute both retention phases, returning the immutable EVICTED tombstone."""

        try:
            sealed = self.seal(
                scope=scope,
                unit_id=unit_id,
                plan_hash=plan_hash,
                run_config_hash=run_config_hash,
                downstream_raw_shard_ref=downstream_raw_shard_ref,
                downstream_raw_shard_hash=downstream_raw_shard_hash,
                receipt_root=receipt_root,
            )
            return self.finalize_eviction(
                str(sealed["receipt_ref"]), receipt_root=receipt_root
            )
        finally:
            # The combined lifecycle must invalidate memo even when seal-phase
            # validation fails before finalize_eviction can establish the boundary.
            with self._lock:
                self._disable_memoization_unlocked()

    @staticmethod
    def verify_receipt(
        receipt_root_or_ref: str | Path,
        receipt_ref: str | Path | None = None,
    ) -> dict[str, object]:
        """Verify SEALED/EVICTED receipt identity without loading cache objects."""

        if receipt_ref is None:
            supplied = Path(receipt_root_or_ref)
            receipt_base = supplied.parent.resolve(strict=False)
            candidate = supplied
        else:
            receipt_base = _resolved_cache_path(
                receipt_root_or_ref, field_name="receipt_root"
            )
            candidate = Path(receipt_ref)
        candidate = (
            candidate if candidate.is_absolute() else receipt_base / candidate
        ).resolve(strict=False)
        if not _path_is_within(candidate, receipt_base):
            raise ValueError("NODE_CACHE_RECEIPT_PATH_ESCAPE")
        value = load_canonical_json(candidate)
        if not isinstance(value, Mapping):
            raise ValueError("NODE_CACHE_RECEIPT_NOT_OBJECT")
        if candidate.name.endswith(".SEALED.json"):
            sealed_value = PersistentNodeGradientCache._validate_sealed_payload(value)
            cache_root = _resolved_cache_path(
                str(sealed_value["cache_root_ref"]), field_name="cache_root_ref"
            )
            if _path_is_within(cache_root, receipt_base) or _path_is_within(
                receipt_base, cache_root
            ):
                raise ValueError("NODE_CACHE_RECEIPT_ROOT_MUST_BE_OUTSIDE_CACHE_ROOT")
            return sealed_value
        if candidate.name.endswith(".EVICTED.json"):
            sealed_name = candidate.name.removesuffix(".EVICTED.json") + ".SEALED.json"
            sealed_path = candidate.with_name(sealed_name)
            sealed = load_canonical_json(sealed_path)
            if not isinstance(sealed, Mapping):
                raise ValueError("NODE_CACHE_SEAL_NOT_OBJECT")
            sealed_value = PersistentNodeGradientCache._validate_sealed_payload(sealed)
            cache_root = _resolved_cache_path(
                str(sealed_value["cache_root_ref"]), field_name="cache_root_ref"
            )
            if _path_is_within(cache_root, receipt_base) or _path_is_within(
                receipt_base, cache_root
            ):
                raise ValueError("NODE_CACHE_RECEIPT_ROOT_MUST_BE_OUTSIDE_CACHE_ROOT")
            return PersistentNodeGradientCache._validate_evicted_payload(
                candidate, sealed_value
            )
        raise ValueError("NODE_CACHE_RECEIPT_FILENAME_INVALID")


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


@dataclass(frozen=True, slots=True)
class FormalReferenceBinding:
    """Immutable contract identity for a formal cross-family reference.

    The contribution vector is only one part of a reference.  The formal plan,
    threshold contract, tolerance and complete family ladders determine which
    numerical object was measured and at what cost.  Keeping this identity in
    a small standalone value object lets the producer and every consumer use
    exactly the same canonical hash without changing the legacy fixture result
    wire.
    """

    formal_plan_ref: str
    formal_plan_hash: str
    thresholds_hash: str
    reference_tolerance: float
    reference_ladder_hash: str
    reference_ladder_levels: Mapping[str, Sequence[int]]
    reference_ladder_nodes: Mapping[str, Sequence[int]]
    required_consecutive: int = 2
    primary_family: str = "gauss_legendre"
    schema_version: str = "stage3-formal-reference-binding-v1"

    def __post_init__(self) -> None:
        if self.schema_version != "stage3-formal-reference-binding-v1":
            raise ValueError("不支持的 FormalReferenceBinding schema")
        if (
            not isinstance(self.formal_plan_ref, str)
            or not self.formal_plan_ref
            or "?" in self.formal_plan_ref
            or "://" in self.formal_plan_ref
        ):
            raise ValueError("formal_plan_ref 必须是非空稳定 artifact ref")
        if any(
            marker in self.formal_plan_ref.casefold()
            for marker in ("fixture", "synthetic")
        ):
            raise FormalRunRejected("FORMAL_REFERENCE_PLAN_REF_FORBIDS_FIXTURE_LABEL")
        for name, value in (
            ("formal_plan_hash", self.formal_plan_hash),
            ("thresholds_hash", self.thresholds_hash),
            ("reference_ladder_hash", self.reference_ladder_hash),
        ):
            _require_sha256(value, field_name=name)
        if not math.isfinite(float(self.reference_tolerance)) or self.reference_tolerance <= 0:
            raise ValueError("reference_tolerance 必须是有限正数")
        if (
            isinstance(self.required_consecutive, bool)
            or not isinstance(self.required_consecutive, int)
            or self.required_consecutive != 2
        ):
            raise ValueError("formal reference required_consecutive 必须固定为 2")
        families = ("gauss_legendre", "composite_simpson")
        _require_identifier(self.primary_family, field_name="primary_family")
        if self.primary_family not in families:
            raise ValueError("primary_family 必须是冻结 reference family")

        def normalize(
            value: Mapping[str, Sequence[int]],
            *,
            field_name: str,
        ) -> Mapping[str, tuple[int, ...]]:
            if not isinstance(value, Mapping) or set(value) != set(families):
                raise ValueError(
                    f"{field_name} 必须精确覆盖 gauss_legendre/composite_simpson"
                )
            result: dict[str, tuple[int, ...]] = {}
            for family in families:
                raw = value[family]
                if (
                    not isinstance(raw, (list, tuple))
                    or len(raw) < 2
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, int)
                        or item <= 0
                        for item in raw
                    )
                    or tuple(raw) != tuple(sorted(set(raw)))
                ):
                    raise ValueError(
                        f"{field_name}.{family} 必须是严格递增的至少两个正整数"
                    )
                result[family] = tuple(raw)
            return MappingProxyType(result)

        object.__setattr__(
            self,
            "reference_ladder_levels",
            normalize(self.reference_ladder_levels, field_name="reference_ladder_levels"),
        )
        object.__setattr__(
            self,
            "reference_ladder_nodes",
            normalize(self.reference_ladder_nodes, field_name="reference_ladder_nodes"),
        )

    def payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "formal_plan_ref": self.formal_plan_ref,
            "formal_plan_hash": self.formal_plan_hash,
            "thresholds_hash": self.thresholds_hash,
            "reference_tolerance": float(self.reference_tolerance),
            "reference_ladder_hash": self.reference_ladder_hash,
            "reference_ladder_levels": {
                family: list(values)
                for family, values in sorted(self.reference_ladder_levels.items())
            },
            "reference_ladder_nodes": {
                family: list(values)
                for family, values in sorted(self.reference_ladder_nodes.items())
            },
            "required_consecutive": self.required_consecutive,
            "primary_family": self.primary_family,
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    @property
    def binding_hash(self) -> str:
        """Hash of the flattened identity embedded in path reference output."""

        return canonical_json_hash(
            {
                key: value
                for key, value in self.payload_dict().items()
                if key != "schema_version"
            }
        )

    def to_dict(self) -> dict[str, object]:
        return self.payload_dict() | {"artifact_hash": self.artifact_hash}

    def artifact_fields(self) -> dict[str, object]:
        """Return the flattened fields embedded in the outer reference artifact."""

        return {
            key: value
            for key, value in self.payload_dict().items()
            if key != "schema_version"
        } | {"reference_binding_hash": self.binding_hash}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "FormalReferenceBinding":
        expected = {
            "schema_version",
            "formal_plan_ref",
            "formal_plan_hash",
            "thresholds_hash",
            "reference_tolerance",
            "reference_ladder_hash",
            "reference_ladder_levels",
            "reference_ladder_nodes",
            "required_consecutive",
            "primary_family",
            "artifact_hash",
        }
        if set(value) != expected:
            raise ValueError("FormalReferenceBinding 字段集合不匹配")
        binding = cls(
            formal_plan_ref=value["formal_plan_ref"],  # type: ignore[arg-type]
            formal_plan_hash=value["formal_plan_hash"],  # type: ignore[arg-type]
            thresholds_hash=value["thresholds_hash"],  # type: ignore[arg-type]
            reference_tolerance=float(value["reference_tolerance"]),
            reference_ladder_hash=value["reference_ladder_hash"],  # type: ignore[arg-type]
            reference_ladder_levels=value["reference_ladder_levels"],  # type: ignore[arg-type]
            reference_ladder_nodes=value["reference_ladder_nodes"],  # type: ignore[arg-type]
            required_consecutive=value["required_consecutive"],  # type: ignore[arg-type]
            primary_family=value["primary_family"],  # type: ignore[arg-type]
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )
        if value["artifact_hash"] != binding.artifact_hash:
            raise ValueError("FormalReferenceBinding artifact_hash 与内容不一致")
        return binding


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
        if execution.run_intent == "formal" and required_consecutive != 2:
            raise FormalRunRejected(
                "FORMAL_REFERENCE_REQUIRED_CONSECUTIVE_MUST_BE_TWO"
            )
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
    # The first five fields are retained for wire compatibility with the
    # original fixture recommendation.  The fields below are deliberately
    # optional here: local fixture observations do not pretend to contain the
    # formal Stage 3 metric contract.  The independent stage3_gate evaluator
    # requires every one of these fields before it can qualify a formal
    # recommendation.
    max_normalized_l2_error: float | None = None
    max_normalized_linf_error: float | None = None
    max_completeness_relative_residual: float | None = None
    max_completeness_l1_scaled_residual: float | None = None
    min_active_spearman: float | None = None
    min_cosine_similarity: float | None = None
    min_sign_consistency: float | None = None
    min_topq_overlap: float | None = None
    min_topq_jaccard: float | None = None
    max_layer_quality_tv: float | None = None
    max_module_quality_tv: float | None = None
    max_reference_normalized_l1_error: float | None = None
    completeness_stability_epsilon: float | None = None
    active_set_threshold: float | None = None
    top_q_values: tuple[float, ...] | None = None
    required_strata: tuple[str, ...] | None = None
    require_worst_case: bool | None = None

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
        for name in (
            "max_normalized_l2_error",
            "max_normalized_linf_error",
            "max_completeness_relative_residual",
            "max_completeness_l1_scaled_residual",
            "max_layer_quality_tv",
            "max_module_quality_tv",
            "max_reference_normalized_l1_error",
            "active_set_threshold",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} 必须是非负有限数或 null")
        if self.completeness_stability_epsilon is not None and (
            not math.isfinite(float(self.completeness_stability_epsilon))
            or float(self.completeness_stability_epsilon) <= 0
        ):
            raise ValueError("completeness_stability_epsilon 必须是正有限数或 null")
        for name in (
            "min_active_spearman",
            "min_cosine_similarity",
            "min_sign_consistency",
            "min_topq_overlap",
            "min_topq_jaccard",
        ):
            value = getattr(self, name)
            if value is not None:
                lower = -1.0 if name in {"min_active_spearman", "min_cosine_similarity"} else 0.0
                if not math.isfinite(float(value)) or not lower <= float(value) <= 1:
                    raise ValueError(f"{name} 必须位于 [{lower:g},1] 或为 null")
        if self.top_q_values is not None:
            values = tuple(float(item) for item in self.top_q_values)
            if (
                not values
                or len(set(values)) != len(values)
                or any(not math.isfinite(item) or not 0 < item <= 1 for item in values)
            ):
                raise ValueError("top_q_values 必须是 (0,1] 内不重复的有限数列")
            object.__setattr__(self, "top_q_values", values)
        if self.required_strata is not None:
            values = tuple(self.required_strata)
            if not values or len(set(values)) != len(values) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise ValueError("required_strata 必须是非空无重复字符串数组")
            object.__setattr__(self, "required_strata", values)
        if self.require_worst_case is not None and type(self.require_worst_case) is not bool:
            raise TypeError("require_worst_case 必须是 bool 或 null")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "max_normalized_l1_error": self.max_normalized_l1_error,
            "max_completeness_absolute_residual": self.max_completeness_absolute_residual,
            "min_spearman": self.min_spearman,
            "min_topk_overlap": self.min_topk_overlap,
            "max_unique_nodes": self.max_unique_nodes,
        }
        for name in (
            "max_normalized_l2_error",
            "max_normalized_linf_error",
            "max_completeness_relative_residual",
            "max_completeness_l1_scaled_residual",
            "min_active_spearman",
            "min_cosine_similarity",
            "min_sign_consistency",
            "min_topq_overlap",
            "min_topq_jaccard",
            "max_layer_quality_tv",
            "max_module_quality_tv",
            "max_reference_normalized_l1_error",
            "completeness_stability_epsilon",
            "active_set_threshold",
            "top_q_values",
            "required_strata",
            "require_worst_case",
        ):
            current = getattr(self, name)
            if current is not None:
                value[name] = list(current) if isinstance(current, tuple) else current
        return value

    def require_formal_contract(self) -> "QuadratureThresholds":
        """Reject the legacy five-column fixture contract for formal use."""

        required = (
            "max_normalized_l2_error",
            "max_normalized_linf_error",
            "max_completeness_relative_residual",
            "max_completeness_l1_scaled_residual",
            "min_active_spearman",
            "min_cosine_similarity",
            "min_sign_consistency",
            "min_topq_overlap",
            "min_topq_jaccard",
            "max_layer_quality_tv",
            "max_module_quality_tv",
            "max_reference_normalized_l1_error",
            "completeness_stability_epsilon",
            "active_set_threshold",
            "top_q_values",
            "required_strata",
            "require_worst_case",
        )
        missing = tuple(name for name in required if getattr(self, name) is None)
        if missing:
            raise FormalRunRejected(
                "FORMAL_QUADRATURE_THRESHOLDS_INCOMPLETE:" + ",".join(missing)
            )
        if self.top_q_values != (0.001, 0.01, 0.05):
            raise FormalRunRejected("FORMAL_QUADRATURE_TOP_Q_NOT_PREREGISTERED")
        if self.required_strata != ("model", "stage", "update", "probe"):
            raise FormalRunRejected("FORMAL_QUADRATURE_STRATA_NOT_PREREGISTERED")
        if self.require_worst_case is not True:
            raise FormalRunRejected("FORMAL_QUADRATURE_WORST_CASE_REQUIRED")
        from .stage3_protocol import DEFAULT_THRESHOLDS

        observed = self.to_dict()
        for name, bound in DEFAULT_THRESHOLDS.items():
            value = observed.get(name)
            if value is None:
                raise FormalRunRejected(f"FORMAL_QUADRATURE_THRESHOLD_MISSING:{name}")
            if name.startswith("max_") and float(value) > float(bound):
                raise FormalRunRejected(f"FORMAL_QUADRATURE_THRESHOLD_WIDENED:{name}")
            if name.startswith("min_") and float(value) < float(bound):
                raise FormalRunRejected(f"FORMAL_QUADRATURE_THRESHOLD_WIDENED:{name}")
        if float(self.max_reference_normalized_l1_error) > (
            self.max_normalized_l1_error / 10.0
        ):
            raise FormalRunRejected("FORMAL_REFERENCE_ERROR_NOT_TEN_TIMES_STRICTER")
        return self

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
    normalized_l2_error: float | None = None
    normalized_linf_error: float | None = None
    completeness_relative_residual: float | None = None
    completeness_l1_scaled_residual: float | None = None
    active_spearman: float | None = None
    cosine_similarity: float | None = None
    sign_consistency: float | None = None
    topq_overlap: Mapping[float | str, float] = field(default_factory=dict)
    topq_jaccard: Mapping[float | str, float] = field(default_factory=dict)
    layer_quality_tv: float | None = None
    module_quality_tv: float | None = None
    reference_normalized_l1_error: float | None = None
    strata: Mapping[str, object] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    scope: str = "local_fixture"
    # Deprecated producer diagnostic retained for old fixture wire rows.  A
    # formal Gate derives worst cases from the complete table and never trusts
    # this flag.
    worst_case: bool | None = None
    # These fields deliberately remain optional/defaulted for older fixture
    # observations.  Formal runners must populate them from the callback
    # accounting rather than substituting ``unique_nodes`` for elapsed time.
    gradient_evaluations: int = 0
    loss_evaluations: int = 0
    forward_evaluations: int = 0
    backward_evaluations: int = 0
    peak_gpu_memory_bytes: int | None = None

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
        for name in (
            "normalized_l2_error",
            "normalized_linf_error",
            "completeness_relative_residual",
            "completeness_l1_scaled_residual",
            "layer_quality_tv",
            "module_quality_tv",
            "reference_normalized_l1_error",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} 必须是非负有限数或 null")
        for name in ("active_spearman", "cosine_similarity"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or not -1 <= float(value) <= 1):
                raise ValueError(f"{name} 必须位于 [-1,1] 或为 null")
        for name in ("sign_consistency",):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(float(value)) or not 0 <= float(value) <= 1):
                raise ValueError(f"{name} 必须位于 [0,1] 或为 null")
        for name in ("topq_overlap", "topq_jaccard"):
            raw = getattr(self, name)
            normalized = {str(key): float(value) for key, value in raw.items()}
            if any(not math.isfinite(value) or not 0 <= value <= 1 for value in normalized.values()):
                raise ValueError(f"{name} 必须只包含 [0,1] 内的有限数")
            object.__setattr__(self, name, MappingProxyType(normalized))
        if any(not isinstance(key, str) or not key for key in self.strata):
            raise ValueError("strata key 必须是非空字符串")
        object.__setattr__(self, "strata", MappingProxyType(dict(self.strata)))
        refs = tuple(self.evidence_refs)
        if any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("evidence_refs 必须是非空字符串数组")
        object.__setattr__(self, "evidence_refs", refs)
        if self.scope not in {"local_fixture", "formal"}:
            raise ValueError("QuadratureObservation scope 不受支持")
        if self.scope == "formal" and not refs:
            raise FormalRunRejected("FORMAL_QUADRATURE_OBSERVATION_EVIDENCE_REQUIRED")
        if self.worst_case is not None and type(self.worst_case) is not bool:
            raise TypeError("worst_case 必须是 bool 或 null")
        for name in (
            "gradient_evaluations",
            "loss_evaluations",
            "forward_evaluations",
            "backward_evaluations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if self.peak_gpu_memory_bytes is not None and (
            isinstance(self.peak_gpu_memory_bytes, bool)
            or not isinstance(self.peak_gpu_memory_bytes, int)
            or self.peak_gpu_memory_bytes < 0
        ):
            raise ValueError("peak_gpu_memory_bytes 必须是非负整数或 null")


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
    # Formal qualification is intentionally bound to the independent Gate
    # evaluation and to the completed provenance record, not just to G3-5.
    # ``None`` keeps legacy/local fixture wire artifacts round-trippable.
    gate_evaluation_hash: str | None = None
    provenance_hash: str | None = None
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
        if self.gate_evaluation_hash is not None:
            _require_sha256(self.gate_evaluation_hash, field_name="gate_evaluation_hash")
        if self.provenance_hash is not None:
            _require_sha256(self.provenance_hash, field_name="provenance_hash")
        if self.qualification.formal_eligible and (
            self.gate_evaluation_hash is None or self.provenance_hash is None
        ):
            raise FormalRunRejected(
                "FORMAL_RECOMMENDATION_REQUIRES_GATE_EVALUATION_AND_PROVENANCE"
            )
        if self.status != "QUALIFIED" and (
            self.gate_evaluation_hash is not None or self.provenance_hash is not None
        ):
            raise FormalRunRejected("UNQUALIFIED_RECOMMENDATION_CANNOT_CARRY_FORMAL_BINDINGS")

    @property
    def scope(self) -> str:
        return self.qualification.scope

    def payload_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
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
        if self.gate_evaluation_hash is not None:
            value["gate_evaluation_hash"] = self.gate_evaluation_hash
        if self.provenance_hash is not None:
            value["provenance_hash"] = self.provenance_hash
        return value

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
        optional = {"gate_evaluation_hash", "provenance_hash"}
        # Sets are unhashable; use explicit equality rather than putting the
        # accepted key sets inside another set.
        if set(value) != expected and set(value) != expected | optional:
            raise ValueError("QUADRATURE_RECOMMENDATION_FIELDS_MISMATCH")
        thresholds_raw = value["thresholds"]
        if not isinstance(thresholds_raw, Mapping):
            raise ValueError("QUADRATURE_RECOMMENDATION_THRESHOLDS_INVALID")
        allowed_thresholds = {
            "max_normalized_l1_error",
            "max_completeness_absolute_residual",
            "min_spearman",
            "min_topk_overlap",
            "max_unique_nodes",
            "max_normalized_l2_error",
            "max_normalized_linf_error",
            "max_completeness_relative_residual",
            "max_completeness_l1_scaled_residual",
            "min_active_spearman",
            "min_cosine_similarity",
            "min_sign_consistency",
            "min_topq_overlap",
            "min_topq_jaccard",
            "max_layer_quality_tv",
            "max_module_quality_tv",
            "max_reference_normalized_l1_error",
            "completeness_stability_epsilon",
            "active_set_threshold",
            "top_q_values",
            "required_strata",
            "require_worst_case",
        }
        if not set(thresholds_raw).issubset(allowed_thresholds):
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
        if value["formal_eligible"] is True or value["status"] == "QUALIFIED":
            # A payload containing only hashes cannot prove that the referenced
            # Gate/provenance artifacts were parsed and matched.  Qualified
            # recommendations must be reconstructed by an authority-aware
            # loader supplied with those canonical objects.
            raise FormalRunRejected(
                "QUALIFIED_RECOMMENDATION_REQUIRES_AUTHORITY_AWARE_LOADER"
            )
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
            gate_evaluation_hash=value.get("gate_evaluation_hash"),  # type: ignore[arg-type]
            provenance_hash=value.get("provenance_hash"),  # type: ignore[arg-type]
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
        gate_evaluation: object | None = None,
        provenance: object | None = None,
    ) -> "QuadratureRecommendation":
        from param_importance_nlp.contracts.provenance import ProvenanceRecord
        from param_importance_nlp.experiments.stage3_gate import Stage3GateEvaluation

        execution.require_for_stage(3)
        if self.scope != "formal" or self.status != "FORMAL_CANDIDATE":
            raise FormalRunRejected("QUADRATURE_RECOMMENDATION_NOT_FORMAL_CANDIDATE")
        if execution.artifact_hash != self.execution_evidence_hash:
            raise FormalRunRejected("QUADRATURE_EXECUTION_EVIDENCE_MISMATCH")
        self.thresholds.require_formal_contract()
        # A single G3-5 record is not sufficient authorization for a formal
        # recommendation.  The independent evaluator must first bind all six
        # G3-0..G3-5 records and a clean completed provenance record.
        required_gate_ids = {f"stage3.G3-{index}" for index in range(6)}
        observed_gate_ids = {
            item.gate_id
            for item in execution.prerequisite_gates
            if isinstance(item, GateRecord)
        }
        if not required_gate_ids.issubset(observed_gate_ids):
            raise FormalRunRejected("FORMAL_RECOMMENDATION_REQUIRES_G3_0_THROUGH_G3_5")
        if gate.gate_id != "stage3.G3-7":
            raise FormalRunRejected("FORMAL_RECOMMENDATION_REQUIRES_G3_7")
        accepted = require_accepted_gate(gate, stage=3)
        _require_gate_binding(accepted, artifact_ref)
        if not isinstance(gate_evaluation, Stage3GateEvaluation) or not isinstance(
            provenance, ProvenanceRecord
        ):
            raise FormalRunRejected(
                "FORMAL_RECOMMENDATION_REQUIRES_GATE_EVALUATION_AND_PROVENANCE"
            )
        evaluation_hash = gate_evaluation.artifact_hash
        provenance_hash = provenance.artifact_hash
        if gate_evaluation.formal_eligible is not True:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_NOT_ELIGIBLE")
        if provenance.formal_eligible is not True:
            raise FormalRunRejected("FORMAL_PROVENANCE_NOT_ELIGIBLE")
        if gate_evaluation.execution_evidence_hash != execution.artifact_hash:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_EXECUTION_MISMATCH")
        if gate_evaluation.thresholds_hash != self.thresholds.artifact_hash:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_THRESHOLDS_MISMATCH")
        if gate_evaluation.required_unit_ids != self.required_unit_ids:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_UNITS_MISMATCH")
        expected_gate_hashes = tuple(
            gate.artifact_hash
            for gate in execution.prerequisite_gates
            if gate.gate_id in required_gate_ids
        )
        expected_by_id = {
            gate.gate_id: gate
            for gate in execution.prerequisite_gates
            if gate.gate_id in required_gate_ids
        }
        if set(expected_by_id) != required_gate_ids or any(
            item.status is not GateStatus.PASS
            or item.effective_status() is not GateStatus.PASS
            for item in expected_by_id.values()
        ):
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_REQUIRES_REAL_PASS_GATES")
        expected_gate_hashes = tuple(
            expected_by_id[f"stage3.G3-{index}"].artifact_hash for index in range(6)
        )
        if gate_evaluation.gate_hashes != expected_gate_hashes:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_GATE_HASHES_MISMATCH")
        evaluated_passing = tuple(
            rule
            for rule, raw in gate_evaluation.rule_evaluations.items()
            if isinstance(raw, Mapping) and raw.get("passing") is True
        )
        if set(evaluated_passing) != set(self.passing_rules):
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_PASSING_RULES_MISMATCH")
        if gate_evaluation.provenance_hash != provenance_hash:
            raise FormalRunRejected("FORMAL_GATE_EVALUATION_PROVENANCE_MISMATCH")
        if (
            provenance.scope != "formal"
            or provenance.status.value != "COMPLETED"
            or not provenance.worktree_clean
            or not provenance.formal_eligible
            or not set(gate_evaluation.source_artifact_refs).issubset(
                set(provenance.artifact_refs)
            )
        ):
            raise FormalRunRejected("FORMAL_PROVENANCE_NOT_COMPLETED_CLEAN_BOUND")
        return replace(
            self,
            status="QUALIFIED",
            qualification=ArtifactQualification.from_gate(
                scope="formal", gate=accepted, stage=3
            ),
            gate_evaluation_hash=evaluation_hash,
            provenance_hash=provenance_hash,
        )

    @classmethod
    def from_mapping_authorized(
        cls,
        value: Mapping[str, object],
        *,
        execution: FormalExecutionEvidence,
        gate: GateRecord,
        gate_evaluation: object,
        provenance: object,
        artifact_ref: str | None = None,
    ) -> "QuadratureRecommendation":
        """Reload and qualify a candidate only with canonical authorities.

        ``from_mapping`` intentionally rejects already-qualified hash-only
        payloads.  This loader instead reconstructs an unqualified candidate
        and immediately verifies the live G3-7 Gate, independent Gate
        evaluation, completed provenance and execution evidence objects.
        """

        candidate = cls.from_mapping(value)
        return candidate.qualify(
            execution=execution,
            gate=gate,
            artifact_ref=artifact_ref,
            gate_evaluation=gate_evaluation,
            provenance=provenance,
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
            thresholds.require_formal_contract()
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
            if passed and execution.run_intent == "formal":
                # Formal recommendation cannot silently fall back to the old
                # five-column fixture contract.  The independent evaluator is
                # still the authority for six-Gate/provenance binding, but the
                # recommendation engine must at least reject incomplete rows.
                required_thresholds = (
                    "max_normalized_l2_error",
                    "max_normalized_linf_error",
                    "max_completeness_relative_residual",
                    "max_completeness_l1_scaled_residual",
                    "min_active_spearman",
                    "min_cosine_similarity",
                    "min_sign_consistency",
                    "min_topq_overlap",
                    "min_topq_jaccard",
                    "max_layer_quality_tv",
                    "max_module_quality_tv",
                    "max_reference_normalized_l1_error",
                )
                passed = all(
                    getattr(thresholds, name) is not None for name in required_thresholds
                ) and all(
                    row.normalized_l2_error is not None
                    and row.normalized_linf_error is not None
                    and row.completeness_relative_residual is not None
                    and row.completeness_l1_scaled_residual is not None
                    and row.active_spearman is not None
                    and row.cosine_similarity is not None
                    and row.sign_consistency is not None
                    and row.layer_quality_tv is not None
                    and row.module_quality_tv is not None
                    and row.reference_normalized_l1_error is not None
                    and row.topq_overlap
                    and row.topq_jaccard
                    and row.strata
                    and row.scope == "formal"
                    and row.evidence_refs
                    and row.normalized_l2_error <= float(thresholds.max_normalized_l2_error)
                    and row.normalized_linf_error <= float(thresholds.max_normalized_linf_error)
                    and row.completeness_relative_residual <= float(thresholds.max_completeness_relative_residual)
                    and row.completeness_l1_scaled_residual <= float(thresholds.max_completeness_l1_scaled_residual)
                    and row.active_spearman >= float(thresholds.min_active_spearman)
                    and row.cosine_similarity >= float(thresholds.min_cosine_similarity)
                    and row.sign_consistency >= float(thresholds.min_sign_consistency)
                    and row.layer_quality_tv <= float(thresholds.max_layer_quality_tv)
                    and row.module_quality_tv <= float(thresholds.max_module_quality_tv)
                    and row.reference_normalized_l1_error <= float(thresholds.max_reference_normalized_l1_error)
                    for row in rows
                )
                top_q_values = thresholds.top_q_values or (0.001, 0.01, 0.05)
                if passed:
                    passed = all(
                        all(str(q) in row.topq_overlap or f"{q:g}" in row.topq_overlap for q in top_q_values)
                        and all(str(q) in row.topq_jaccard or f"{q:g}" in row.topq_jaccard for q in top_q_values)
                        and all(
                            row.topq_overlap.get(str(q), row.topq_overlap.get(f"{q:g}", -math.inf))
                            >= float(thresholds.min_topq_overlap)
                            for q in top_q_values
                        )
                        and all(
                            row.topq_jaccard.get(str(q), row.topq_jaccard.get(f"{q:g}", -math.inf))
                            >= float(thresholds.min_topq_jaccard)
                            for q in top_q_values
                        )
                        for row in rows
                    )
                if passed and thresholds.required_strata:
                    passed = all(
                        all(key in row.strata for key in thresholds.required_strata)
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
    "FormalReferenceBinding",
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
