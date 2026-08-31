"""把训练引擎 observer 事件发布为 Stage 3 endpoint bundle。

observer 不接管 optimizer；它只在训练引擎已经冻结的 gradient-ready、
parameter-post、attempt-commit 三个边界调用外部状态快照器。独立 replay verifier
是强制参数，不能用“当前内存看起来相同”代替 fresh-state 重放。对象与 commit
分开发布，目录存在或 rename 均不代表可恢复。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping, TYPE_CHECKING

import numpy as np
import torch

from ..atomic import atomic_write_json, stable_json_hash
from ..contracts.jsonio import load_canonical_json
from ..runtime.tensor_bundle import load_tensor_bundle, publish_tensor_bundle
from ..runtime.training import (
    AttemptCommitEvent,
    GradientReadyEvent,
    ParameterPostEvent,
    SkippedAttemptEvent,
    TrainingStepObserver,
)
from .stage3 import EndpointRecord, EndpointState

if TYPE_CHECKING:
    from ..runtime.training import TrainingEngine


def _safe_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise ValueError("ENDPOINT_BUNDLE_ID_INVALID")
    return value


def _tensor_map_hash(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name].detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_tree_hash(value: object) -> str:
    """稳定摘要安全状态树；语义与 TensorBundle 编码前的值一致。"""

    digest = hashlib.sha256()

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"torch\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
            digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
            return
        if isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"numpy\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
            digest.update(array.tobytes(order="C"))
            return
        if isinstance(item, Mapping):
            digest.update(b"mapping{")
            for key in sorted(item, key=lambda candidate: (type(candidate).__name__, str(candidate))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
            return
        if isinstance(item, (list, tuple)):
            digest.update(b"list[" if isinstance(item, list) else b"tuple(")
            for child in item:
                visit(child)
            digest.update(b"]" if isinstance(item, list) else b")")
            return
        digest.update(
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        )
        digest.update(b"\0")

    visit(value)
    return digest.hexdigest()


def validate_endpoint_state_bundle(
    value: Mapping[str, object],
    expected: EndpointState,
) -> None:
    """复算 endpoint bundle 的全部组件摘要。

    TensorBundle manifest 只能证明“一组字节没有变化”；这里继续证明这些字节
    确实对应 :class:`EndpointState` 声明的参数、buffer、optimizer 与控制状态。
    formal consumer 不应只相信 producer 写入的局部 hash，否则一个内部自洽但
    语义伪造的 endpoint object 仍可能污染路径缓存与积分报告。

    Args:
        value: ``TrainingEngine.capture_observer_state`` 发布的安全状态树。
        expected: endpoint record 中绑定该 bundle 的组件身份。

    Raises:
        TypeError: 状态树字段类型不符合安全 bundle 合同。
        ValueError: 字段集合或任意组件摘要与 endpoint record 不一致。
    """

    required = {
        "parameters",
        "buffers",
        "optimizer",
        "scheduler",
        "scaler",
        "rng",
        "cursor",
        "training_state",
        "model_modes",
        "optimizer_gradient",
    }
    if set(value) != required:
        raise ValueError("ENDPOINT_STATE_BUNDLE_FIELDS_MISMATCH")
    parameters = value["parameters"]
    buffers = value["buffers"]
    optimizer = value["optimizer"]
    cursor = value["cursor"]
    training_state = value["training_state"]
    model_modes = value["model_modes"]
    if not isinstance(parameters, Mapping) or not isinstance(buffers, Mapping):
        raise TypeError("ENDPOINT_STATE_BUNDLE_TENSOR_MAP_INVALID")
    if not all(isinstance(item, torch.Tensor) for item in parameters.values()):
        raise TypeError("ENDPOINT_STATE_BUNDLE_PARAMETER_NOT_TENSOR")
    if not all(isinstance(item, torch.Tensor) for item in buffers.values()):
        raise TypeError("ENDPOINT_STATE_BUNDLE_BUFFER_NOT_TENSOR")
    if not isinstance(optimizer, Mapping):
        raise TypeError("ENDPOINT_STATE_BUNDLE_OPTIMIZER_INVALID")
    if not isinstance(cursor, Mapping) or not isinstance(training_state, Mapping):
        raise TypeError("ENDPOINT_STATE_BUNDLE_CURSOR_INVALID")
    if not isinstance(model_modes, Mapping) or not all(
        isinstance(name, str) and type(mode) is bool
        for name, mode in model_modes.items()
    ):
        raise TypeError("ENDPOINT_STATE_BUNDLE_MODEL_MODES_INVALID")

    actual = {
        "parameter_hash": _tensor_map_hash(parameters),  # type: ignore[arg-type]
        "buffer_hash": _tensor_map_hash(buffers),  # type: ignore[arg-type]
        "optimizer_hash": _state_tree_hash(optimizer),
        "scheduler_hash": _state_tree_hash(value["scheduler"]),
        "scaler_hash": _state_tree_hash(value["scaler"]),
        "rng_hash": _state_tree_hash(value["rng"]),
        "data_cursor_hash": _state_tree_hash(
            {"cursor": cursor, "training_state": training_state}
        ),
        "model_mode_hash": _state_tree_hash(model_modes),
    }
    mismatched = [
        name for name, digest in actual.items() if digest != getattr(expected, name)
    ]
    if mismatched:
        raise ValueError(
            "ENDPOINT_STATE_BUNDLE_COMPONENT_HASH_MISMATCH:"
            + ",".join(sorted(mismatched))
        )


def _state_dict(state: EndpointState) -> dict[str, str]:
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


def _record_dict(record: EndpointRecord) -> dict[str, object]:
    return {
        "path_state_id": record.path_state_id,
        "source_run_id": record.source_run_id,
        "optimizer_step": record.optimizer_step,
        "parameter_registry_hash": record.parameter_registry_hash,
        "pre_state": _state_dict(record.pre_state),
        "parameter_post_state": _state_dict(record.parameter_post_state),
        "attempt_commit_state": _state_dict(record.attempt_commit_state),
        "attempt_commit_parent_hash": record.attempt_commit_parent_hash,
        "probe_buffer_snapshot_hash": record.probe_buffer_snapshot_hash,
        "full_update_delta_hash": record.full_update_delta_hash,
        "update_sample_ids": list(record.update_sample_ids),
        "replay_verified": record.replay_verified,
        "metadata": dict(record.metadata),
        "endpoint_digest": record.digest,
    }


@dataclass(frozen=True, slots=True)
class EndpointBundle:
    """已提交且通过独立 replay 的 endpoint 引用。"""

    endpoint_id: str
    endpoint_digest: str
    object_ref: str
    commit_ref: str
    scope: str
    formal_eligible: bool = False
    qualification_evidence_hash: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": "endpoint-bundle-v1",
            "endpoint_id": self.endpoint_id,
            "endpoint_digest": self.endpoint_digest,
            "object_ref": self.object_ref,
            "commit_ref": self.commit_ref,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "qualification_evidence_hash": self.qualification_evidence_hash,
        }
        payload["artifact_hash"] = stable_json_hash(payload)
        return payload


StateCapture = Callable[[str, int, int], EndpointState]
ReplayVerifier = Callable[[EndpointRecord], bool]
ProbeBufferHash = Callable[[], str]


class TrainingEndpointObserver(TrainingStepObserver):
    """按选定成功 step 捕获 endpoint，并发布 replay-verified bundle。

    调用方既可传入三个旧式 callback（用于解析 fixture），也可在构造后调用
    :meth:`bind_engine` 绑定真实 :class:`TrainingEngine`。真实模式会为 pre、
    parameter-post、attempt-commit 各发布一个安全 TensorBundle；独立 verifier
    随后从这些磁盘对象恢复 pre optimizer/参数/梯度，在隔离副本上再次执行
    ``optimizer.step()``，而不是拿活动模型做表面 hash 比较。
    """

    def __init__(
        self,
        *,
        source_run_id: str,
        parameter_registry_hash: str,
        selected_steps: set[int] | frozenset[int],
        state_capture: StateCapture | None = None,
        replay_verifier: ReplayVerifier | None = None,
        probe_buffer_hash: ProbeBufferHash | None = None,
        output_root: str | Path,
        workspace_root: str | Path | None = None,
        scope: str = "local_fixture",
        formal_eligible: bool = False,
        qualification_evidence_hash: str | None = None,
    ) -> None:
        _safe_id(source_run_id)
        if len(parameter_registry_hash) != 64:
            raise ValueError("ENDPOINT_REGISTRY_HASH_INVALID")
        if scope not in {"local_fixture", "formal"}:
            raise ValueError("ENDPOINT_SCOPE_INVALID")
        if type(formal_eligible) is not bool:
            raise TypeError("ENDPOINT_FORMAL_ELIGIBILITY_NOT_BOOL")
        # formal scope 可以先发布未资格化 candidate，但 local fixture 永远不能
        # 冒充 formal。训练 task 的正式 capture plan 会要求此值为 true。
        if formal_eligible and scope != "formal":
            raise ValueError("ENDPOINT_FORMAL_ELIGIBILITY_SCOPE_MISMATCH")
        if formal_eligible and (
            not isinstance(qualification_evidence_hash, str)
            or not re.fullmatch(r"[0-9a-f]{64}", qualification_evidence_hash)
        ):
            raise ValueError("ENDPOINT_FORMAL_QUALIFICATION_EVIDENCE_REQUIRED")
        if not formal_eligible and qualification_evidence_hash is not None:
            raise ValueError("ENDPOINT_UNQUALIFIED_CANNOT_CARRY_EVIDENCE")
        if not selected_steps or any(
            isinstance(step, bool) or not isinstance(step, int) or step <= 0
            for step in selected_steps
        ):
            raise ValueError("ENDPOINT_SELECTED_STEPS_INVALID")
        callbacks = (state_capture, replay_verifier, probe_buffer_hash)
        if any(item is None for item in callbacks) and any(item is not None for item in callbacks):
            raise ValueError("ENDPOINT_CALLBACK_SET_INCOMPLETE")
        self.source_run_id = source_run_id
        self.parameter_registry_hash = parameter_registry_hash
        self.selected_steps = frozenset(selected_steps)
        self.state_capture = state_capture
        self.replay_verifier = replay_verifier
        self.probe_buffer_hash = probe_buffer_hash
        self.scope = scope
        self.formal_eligible = formal_eligible
        self.qualification_evidence_hash = qualification_evidence_hash
        self.root = Path(output_root).resolve()
        self.workspace_root = (
            None if workspace_root is None else Path(workspace_root).resolve()
        )
        if self.workspace_root is not None:
            try:
                self.root.relative_to(self.workspace_root)
            except ValueError as error:
                raise ValueError("ENDPOINT_OUTPUT_OUTSIDE_WORKSPACE") from error
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.state_bundles = self.root / "state-bundles"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)
        self.state_bundles.mkdir(parents=True, exist_ok=True)
        self._engine: TrainingEngine | None = None
        self._active_endpoint_id: str | None = None
        self._state_bundle_refs: dict[str, dict[str, str]] = {}
        self._gradient_event: GradientReadyEvent | None = None
        self._pre: EndpointState | None = None
        self._post: EndpointState | None = None
        self._delta_hash: str | None = None
        self._bundles: list[EndpointBundle] = []
        self._captured_steps: set[int] = set()
        self._discover_existing()

    def bind_engine(self, engine: "TrainingEngine") -> None:
        """绑定真实引擎；必须在第一次 observer 回调之前完成。"""

        if self.state_capture is not None:
            raise RuntimeError("ENDPOINT_CALLBACK_MODE_CANNOT_BIND_ENGINE")
        if self._engine is not None and self._engine is not engine:
            raise RuntimeError("ENDPOINT_ENGINE_ALREADY_BOUND")
        if engine.registry.coordinate_registry_hash != self.parameter_registry_hash:
            raise ValueError("ENDPOINT_ENGINE_REGISTRY_HASH_MISMATCH")
        self._engine = engine

    def _logical_ref(self, path: Path) -> str:
        if self.workspace_root is None:
            return path.as_posix()
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def _resolve_ref(self, reference: str) -> Path:
        """解析 bundle/object 引用并拒绝 workspace 逃逸。"""

        path = Path(reference)
        if self.workspace_root is None:
            return path.resolve()
        if path.is_absolute():
            raise ValueError("ENDPOINT_REFERENCE_MUST_BE_LOGICAL")
        candidate = self.workspace_root.joinpath(*path.parts).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("ENDPOINT_REFERENCE_OUTSIDE_WORKSPACE") from error
        return candidate

    def _capture_state(self, phase: str, step: int, attempt: int) -> EndpointState:
        if self.state_capture is not None:
            return self.state_capture(phase, step, attempt)
        if self._engine is None or self._active_endpoint_id is None:
            raise RuntimeError("ENDPOINT_ENGINE_NOT_BOUND")
        gradient = (
            self._gradient_event.optimizer_gradient.to_dict(clone=True)
            if phase == "pre" and self._gradient_event is not None
            else None
        )
        state = self._engine.capture_observer_state(optimizer_gradient=gradient)
        artifact_id = _safe_id(f"{self._active_endpoint_id}-{phase}")
        bundle_path = self.state_bundles / artifact_id
        if bundle_path.exists():
            restored, bundle = load_tensor_bundle(bundle_path)
            if _state_tree_hash(restored) != _state_tree_hash(state):
                raise ValueError("ENDPOINT_EXISTING_STATE_BUNDLE_DRIFT")
        else:
            bundle = publish_tensor_bundle(bundle_path, state)
        parameters = state["parameters"]
        buffers = state["buffers"]
        assert isinstance(parameters, Mapping) and isinstance(buffers, Mapping)
        self._state_bundle_refs[phase] = {
            "ref": self._logical_ref(bundle_path),
            "manifest_sha256": bundle.manifest_sha256,
        }
        return EndpointState(
            artifact_id=artifact_id,
            artifact_hash=bundle.manifest_sha256,
            parameter_hash=_tensor_map_hash(parameters),  # type: ignore[arg-type]
            buffer_hash=_tensor_map_hash(buffers),  # type: ignore[arg-type]
            optimizer_hash=_state_tree_hash(state["optimizer"]),
            scheduler_hash=_state_tree_hash(state["scheduler"]),
            scaler_hash=_state_tree_hash(state["scaler"]),
            rng_hash=_state_tree_hash(state["rng"]),
            data_cursor_hash=_state_tree_hash(
                {
                    "cursor": state["cursor"],
                    "training_state": state["training_state"],
                }
            ),
            model_mode_hash=_state_tree_hash(state["model_modes"]),
        )

    def _probe_buffer_snapshot_hash(self) -> str:
        if self.probe_buffer_hash is not None:
            return self.probe_buffer_hash()
        if self._pre is None:
            raise RuntimeError("ENDPOINT_PRE_STATE_MISSING")
        return self._pre.buffer_hash

    @property
    def bundles(self) -> tuple[EndpointBundle, ...]:
        return tuple(self._bundles)

    @property
    def captured_steps(self) -> frozenset[int]:
        """返回已经由权威 commit 发现的成功 optimizer step。"""

        return frozenset(self._captured_steps)

    def on_gradient_ready(self, event: GradientReadyEvent) -> None:
        target_step = event.global_step + 1
        if target_step not in self.selected_steps or target_step in self._captured_steps:
            return
        if self._gradient_event is not None:
            raise RuntimeError("ENDPOINT_CAPTURE_OVERLAPPING_ATTEMPT")
        self._gradient_event = event
        self._active_endpoint_id = _safe_id(
            f"{self.source_run_id}-step-{target_step:08d}"
        )
        self._state_bundle_refs = {}
        self._pre = self._capture_state("pre", event.global_step, event.attempt_index)

    def on_parameter_post(self, event: ParameterPostEvent) -> None:
        if self._gradient_event is None:
            return
        if event.transaction.attempt_index != self._gradient_event.attempt_index:
            raise RuntimeError("ENDPOINT_PARAMETER_POST_ATTEMPT_MISMATCH")
        self._post = self._capture_state(
            "parameter_post",
            event.transaction.global_step,
            event.transaction.attempt_index,
        )
        self._delta_hash = _tensor_map_hash(event.outcome.total_delta)

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None:
        if self._gradient_event is None:
            return
        if self._pre is None or self._post is None or self._delta_hash is None:
            raise RuntimeError("ENDPOINT_CAPTURE_MISSING_PRE_OR_POST")
        commit = self._capture_state(
            "attempt_commit",
            event.transaction.global_step,
            event.transaction.attempt_index,
        )
        if self._active_endpoint_id is None:
            raise RuntimeError("ENDPOINT_ACTIVE_ID_MISSING")
        endpoint_id = self._active_endpoint_id
        provisional = EndpointRecord(
            path_state_id=endpoint_id,
            source_run_id=self.source_run_id,
            optimizer_step=event.transaction.global_step + 1,
            parameter_registry_hash=self.parameter_registry_hash,
            pre_state=self._pre,
            parameter_post_state=self._post,
            attempt_commit_state=commit,
            attempt_commit_parent_hash=self._post.artifact_hash,
            probe_buffer_snapshot_hash=self._probe_buffer_snapshot_hash(),
            full_update_delta_hash=self._delta_hash,
            update_sample_ids=self._gradient_event.sample_ids,
            replay_verified=False,
            metadata={
                "attempt_index": event.transaction.attempt_index,
                "microbatch_ids": list(self._gradient_event.microbatch_ids),
            },
        )
        verified = (
            self.replay_verifier(provisional)
            if self.replay_verifier is not None
            else self._verify_persisted_replay(provisional)
        )
        if verified is not True:
            raise RuntimeError("ENDPOINT_REPLAY_VERIFICATION_FAILED")
        # EndpointRecord 是冻结 dataclass；replace 会保留字段类型与未来新增字段，
        # 避免依赖 ``__dataclass_fields__`` 这类实现细节来手工重建合同对象。
        record = replace(provisional, replay_verified=True)
        self._bundles.append(self._publish(endpoint_id, record))
        self._captured_steps.add(record.optimizer_step)
        self._clear()

    def on_skip(self, event: SkippedAttemptEvent) -> None:
        # skip 没有 parameter-post，不能生成 endpoint；若刚好选中该成功 step，
        # 下一次 attempt 仍会以相同 global_step 捕获。
        del event
        self._clear()

    def _clear(self) -> None:
        self._gradient_event = None
        self._pre = None
        self._post = None
        self._delta_hash = None
        self._active_endpoint_id = None
        self._state_bundle_refs = {}

    def _load_phase_bundle(self, phase: str) -> Mapping[str, object]:
        reference = self._state_bundle_refs.get(phase)
        if reference is None:
            raise ValueError(f"ENDPOINT_STATE_BUNDLE_REF_MISSING:{phase}")
        path = self._resolve_ref(reference["ref"])
        state, bundle = load_tensor_bundle(path)
        if bundle.manifest_sha256 != reference["manifest_sha256"]:
            raise ValueError(f"ENDPOINT_STATE_BUNDLE_HASH_MISMATCH:{phase}")
        if not isinstance(state, Mapping):
            raise TypeError(f"ENDPOINT_STATE_BUNDLE_ROOT_INVALID:{phase}")
        return state

    def _verify_persisted_replay(self, record: EndpointRecord) -> bool:
        """从已落盘 pre bundle 在隔离 optimizer 副本上重放一次更新。"""

        if self._engine is None:
            raise RuntimeError("ENDPOINT_ENGINE_NOT_BOUND")
        pre = self._load_phase_bundle("pre")
        post = self._load_phase_bundle("parameter_post")
        committed = self._load_phase_bundle("attempt_commit")
        for phase, value, state in (
            ("pre", pre, record.pre_state),
            ("parameter_post", post, record.parameter_post_state),
            ("attempt_commit", committed, record.attempt_commit_state),
        ):
            reference = self._state_bundle_refs.get(phase)
            if reference is None or reference.get("manifest_sha256") != state.artifact_hash:
                return False
            try:
                validate_endpoint_state_bundle(value, state)
            except (TypeError, ValueError):
                return False
        pre_parameters = pre.get("parameters")
        post_parameters = post.get("parameters")
        gradients = pre.get("optimizer_gradient")
        if not all(isinstance(item, Mapping) for item in (pre_parameters, post_parameters, gradients)):
            return False

        # deepcopy 产生与活动 optimizer 完全隔离的 Parameter/state；随后显式加载
        # pre bundle 中的 optimizer state 和真实已裁剪梯度，故验证不依赖当前内存
        # 参数是否“碰巧”等于 post。
        replay_optimizer = copy.deepcopy(self._engine.optimizer)
        original_to_name = {
            id(parameter): name
            for name, parameter in self._engine.named_parameters.items()
        }
        replay_parameters: dict[str, torch.Tensor] = {}
        for original_group, replay_group in zip(
            self._engine.optimizer.param_groups,
            replay_optimizer.param_groups,
            strict=True,
        ):
            for original, replay in zip(
                original_group["params"], replay_group["params"], strict=True
            ):
                name = original_to_name.get(id(original))
                if name is not None:
                    replay_parameters[name] = replay
        if set(replay_parameters) != set(self._engine.named_parameters):
            return False
        with torch.no_grad():
            for name, parameter in replay_parameters.items():
                source = pre_parameters[name]  # type: ignore[index]
                if not isinstance(source, torch.Tensor):
                    return False
                parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
        replay_optimizer.load_state_dict(pre["optimizer"])  # type: ignore[arg-type]
        for name, parameter in replay_parameters.items():
            gradient = gradients[name]  # type: ignore[index]
            if not isinstance(gradient, torch.Tensor):
                return False
            parameter.grad = gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
        replay_optimizer.step()
        for name, parameter in replay_parameters.items():
            expected = post_parameters[name]  # type: ignore[index]
            if not isinstance(expected, torch.Tensor) or not torch.equal(
                parameter.detach().cpu(), expected.detach().cpu()
            ):
                return False
        if _state_tree_hash(replay_optimizer.state_dict()) != record.parameter_post_state.optimizer_hash:
            return False
        delta = {
            name: post_parameters[name] - pre_parameters[name]  # type: ignore[operator,index]
            for name in sorted(pre_parameters)  # type: ignore[arg-type]
        }
        if _tensor_map_hash(delta) != record.full_update_delta_hash:
            return False
        return (
            _state_tree_hash(committed.get("optimizer"))
            == record.attempt_commit_state.optimizer_hash
            and record.parameter_post_state.parameter_hash
            == record.attempt_commit_state.parameter_hash
        )

    def _publish(self, endpoint_id: str, record: EndpointRecord) -> EndpointBundle:
        object_path = self.objects / f"{endpoint_id}.json"
        commit_path = self.commits / f"{endpoint_id}.json"
        if commit_path.exists():
            raise FileExistsError(f"ENDPOINT_ALREADY_COMMITTED:{endpoint_id}")
        value: dict[str, object] = {
            "schema_version": "endpoint-record-v1",
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "qualification_evidence_hash": self.qualification_evidence_hash,
            "record": _record_dict(record),
            "state_bundles": dict(self._state_bundle_refs),
        }
        value["artifact_hash"] = stable_json_hash(value)
        if object_path.exists():
            # 两阶段发布允许在 object 已写入、commit 尚未落盘时崩溃。fresh-process
            # 重试只能复用逐字节语义相同的不可变对象；内容漂移仍是硬错误。
            orphan = load_canonical_json(object_path)
            # 内存合同允许 tuple，canonical JSON 会规范化为 array；因此用正式
            # canonical 摘要比较语义，不能用 Python ``tuple != list`` 误报漂移。
            if stable_json_hash(orphan) != stable_json_hash(value):
                differing = sorted(
                    key
                    for key in set(orphan) | set(value)
                    if orphan.get(key) != value.get(key)
                ) if isinstance(orphan, Mapping) else ["root"]
                if differing == ["record"]:
                    old_record = orphan.get("record")
                    new_record = value.get("record")
                    if isinstance(old_record, Mapping) and isinstance(new_record, Mapping):
                        differing = [
                            f"record.{key}"
                            for key in sorted(set(old_record) | set(new_record))
                            if old_record.get(key) != new_record.get(key)
                        ]
                raise ValueError(
                    f"ENDPOINT_ORPHAN_OBJECT_DRIFT:{endpoint_id}:"
                    + ",".join(differing)
                )
        else:
            atomic_write_json(object_path, value)
        verified = load_canonical_json(object_path)
        if not isinstance(verified, dict):
            raise ValueError("ENDPOINT_OBJECT_ROOT_INVALID")
        object_hash = stable_json_hash(verified)
        commit: dict[str, object] = {
            "schema_version": "endpoint-commit-v1",
            "endpoint_id": endpoint_id,
            "optimizer_step": record.optimizer_step,
            "endpoint_digest": record.digest,
            "object_ref": self._logical_ref(object_path),
            "object_sha256": object_hash,
            "scope": self.scope,
            "formal_eligible": self.formal_eligible,
            "qualification_evidence_hash": self.qualification_evidence_hash,
        }
        commit["artifact_hash"] = stable_json_hash(commit)
        atomic_write_json(commit_path, commit)
        return EndpointBundle(
            endpoint_id,
            record.digest,
            self._logical_ref(object_path),
            self._logical_ref(commit_path),
            self.scope,
            self.formal_eligible,
            self.qualification_evidence_hash,
        )

    def _discover_existing(self) -> None:
        report = self.reconcile()
        invalid = report["invalid"]
        if invalid:
            raise ValueError(f"ENDPOINT_EXISTING_COMMIT_INVALID:{invalid}")
        for endpoint_id in report["valid"]:
            commit_path = self.commits / f"{endpoint_id}.json"
            commit = load_canonical_json(commit_path)
            assert isinstance(commit, dict)
            step = commit.get("optimizer_step")
            if not isinstance(step, int) or isinstance(step, bool):
                raise ValueError("ENDPOINT_COMMIT_STEP_INVALID")
            bundle = EndpointBundle(
                endpoint_id=str(commit["endpoint_id"]),
                endpoint_digest=str(commit["endpoint_digest"]),
                object_ref=str(commit["object_ref"]),
                commit_ref=self._logical_ref(commit_path),
                scope=str(commit["scope"]),
                formal_eligible=bool(commit["formal_eligible"]),
                qualification_evidence_hash=commit.get("qualification_evidence_hash"),  # type: ignore[arg-type]
            )
            if (
                bundle.scope != self.scope
                or bundle.formal_eligible != self.formal_eligible
                or bundle.qualification_evidence_hash
                != self.qualification_evidence_hash
            ):
                raise ValueError("ENDPOINT_EXISTING_COMMIT_SCOPE_DRIFT")
            self._bundles.append(bundle)
            self._captured_steps.add(step)

    def reconcile(self) -> dict[str, object]:
        valid: list[str] = []
        invalid: list[dict[str, str]] = []
        for commit_path in sorted(self.commits.glob("*.json")):
            try:
                commit = load_canonical_json(commit_path)
                if not isinstance(commit, dict):
                    raise ValueError("ENDPOINT_RECONCILE_COMMIT_ROOT_INVALID")
                declared_commit_hash = commit.get("artifact_hash")
                without_hash = {key: item for key, item in commit.items() if key != "artifact_hash"}
                if declared_commit_hash != stable_json_hash(without_hash):
                    raise ValueError("ENDPOINT_RECONCILE_COMMIT_HASH_MISMATCH")
                object_ref = commit.get("object_ref")
                if not isinstance(object_ref, str):
                    raise ValueError("ENDPOINT_RECONCILE_OBJECT_REF_INVALID")
                object_path = self._resolve_ref(object_ref)
                value = load_canonical_json(object_path)
                if not isinstance(value, dict):
                    raise ValueError("ENDPOINT_RECONCILE_ROOT_INVALID")
                if commit.get("object_sha256") != stable_json_hash(value):
                    raise ValueError("ENDPOINT_RECONCILE_HASH_MISMATCH")
                declared_object_hash = value.get("artifact_hash")
                object_body = {
                    key: item for key, item in value.items() if key != "artifact_hash"
                }
                if declared_object_hash != stable_json_hash(object_body):
                    raise ValueError("ENDPOINT_RECONCILE_OBJECT_ARTIFACT_HASH_MISMATCH")
                if commit.get("endpoint_id") != commit_path.stem:
                    raise ValueError("ENDPOINT_RECONCILE_ID_MISMATCH")
                if commit.get("endpoint_digest") != value.get("record", {}).get("endpoint_digest"):
                    raise ValueError("ENDPOINT_RECONCILE_DIGEST_MISMATCH")
                for field in (
                    "scope",
                    "formal_eligible",
                    "qualification_evidence_hash",
                ):
                    if commit.get(field) != value.get(field):
                        raise ValueError(f"ENDPOINT_RECONCILE_{field.upper()}_MISMATCH")
            except Exception as exc:
                invalid.append({"endpoint_id": commit_path.stem, "reason": str(exc)})
            else:
                valid.append(commit_path.stem)
        return {"valid": valid, "invalid": invalid}


__all__ = [
    "EndpointBundle",
    "TrainingEndpointObserver",
    "validate_endpoint_state_bundle",
]
