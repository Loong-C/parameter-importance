r"""Torch 固定状态梯度 provider 与内存冻结 sample resolver。

Stage 2 估计的是同一参数状态下的数据随机性，因此一次梯度查询
不得偷偷推进 dropout RNG、BatchNorm buffer、模型模式或 resolver cursor。
:class:`TorchFixedStateGradientProvider` 在调用前后对模型、Torch/Python/
NumPy RNG 和 resolver 摘要做完整核验。RNG 会在 ``finally`` 中回放；
若 forward 非法修改参数、buffer、模式或已有 ``.grad``，provider 会先
恢复原状态，再 fail-closed，不把污染模型交还调用方。

多个 draw 的 loss 按 ``LossBatch.effective_count`` 严格加权：

.. math::

   \bar L = \frac{\sum_d L_{d,\mathrm{numerator}}}
                   {\sum_d n_{d,\mathrm{effective}}},\qquad
   \bar g = \nabla_\theta \bar L.

因而有效 token 数不同的 Pile sequence 不会被错误当作等权 mean；
同一 sample ID 被有放回抽中多次时，也会按 draw 出现次数重复计入。
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import random
import re
from types import MappingProxyType

import numpy as np
import torch

from ..contracts.immutable import thaw_json_value
from ..contracts.jsonio import canonical_json_bytes
from ..core.registry import ParameterRegistry
from .protocols import FrozenSampleResolver, GradientBatch
from .training import ModelAdapter, TrainingMicrobatch


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _length_prefixed(digest: "hashlib._Hash", value: bytes) -> None:  # type: ignore[name-defined]
    """向摘要写入无歧义字节块。"""

    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def _tensor_bytes(value: torch.Tensor) -> bytes:
    """序列化 dense tensor 的原始字节，不经过 NumPy dtype 转换。"""

    if value.layout is not torch.strided:
        raise ValueError("FIXED_STATE_TENSOR_LAYOUT_UNSUPPORTED")
    if value.device.type == "meta":
        raise ValueError("FIXED_STATE_META_TENSOR_UNSUPPORTED")
    contiguous = value.detach().contiguous().cpu()
    # Torch 不允许 0-D tensor 直接跨 element-size view；先展平不改变
    # 字节顺序，同时也兼容空 tensor。
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")


def _update_tensor_digest(
    digest: "hashlib._Hash",  # type: ignore[name-defined]
    *,
    role: str,
    name: str,
    value: torch.Tensor,
) -> None:
    descriptor = canonical_json_bytes(
        {
            "role": role,
            "name": name,
            "dtype": str(value.dtype),
            "device": str(value.device),
            "layout": str(value.layout),
            "shape": list(value.shape),
            "stride": list(value.stride()),
        }
    )
    _length_prefixed(digest, descriptor)
    _length_prefixed(digest, _tensor_bytes(value))


def _validate_digest(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} 必须是小写 SHA-256")
    return value


def _draw_sample_id(draw: object) -> Hashable:
    sample_id = getattr(draw, "sample_id", draw)
    try:
        hash(sample_id)
    except TypeError as exc:
        raise TypeError("DRAW_SAMPLE_ID_NOT_HASHABLE") from exc
    return sample_id  # type: ignore[return-value]


def _named_parameters_with_aliases(
    module: torch.nn.Module,
) -> tuple[
    tuple[str, ...],
    dict[str, torch.nn.Parameter],
    dict[str, tuple[str, ...]],
]:
    """按模型遍历顺序合并共享 Parameter alias。"""

    try:
        named = list(module.named_parameters(remove_duplicate=False))
    except TypeError:  # pragma: no cover - 旧 Torch 兼容路径
        named = list(module.named_parameters())
    aliases_by_object: dict[int, list[str]] = {}
    parameter_by_object: dict[int, torch.nn.Parameter] = {}
    object_order: list[int] = []
    for name, parameter in named:
        if not name:
            raise ValueError("MODEL_PARAMETER_NAME_EMPTY")
        identity = id(parameter)
        if identity not in aliases_by_object:
            aliases_by_object[identity] = []
            parameter_by_object[identity] = parameter
            object_order.append(identity)
        aliases_by_object[identity].append(name)
    trainable_order = [
        identity
        for identity in object_order
        if parameter_by_object[identity].requires_grad
    ]
    if not trainable_order:
        raise ValueError("MODEL_HAS_NO_TRAINABLE_PARAMETER")
    names = tuple(aliases_by_object[identity][0] for identity in trainable_order)
    parameters = {
        aliases_by_object[identity][0]: parameter_by_object[identity]
        for identity in trainable_order
    }
    aliases = {
        aliases_by_object[identity][0]: tuple(aliases_by_object[identity][1:])
        for identity in trainable_order
    }
    return names, parameters, aliases


def _derived_registry_hash(
    names: Sequence[str],
    parameters: Mapping[str, torch.nn.Parameter],
    aliases: Mapping[str, Sequence[str]],
) -> str:
    """在未传入 ParameterRegistry 时生成纯模型坐标摘要。"""

    payload = [
        {
            "canonical_name": name,
            "aliases": list(aliases[name]),
            "shape": list(parameters[name].shape),
            "order": index,
            "eligibility": "requires_grad_dense_unique_parameter",
        }
        for index, name in enumerate(names)
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _assert_dense_unique_parameters(
    names: Sequence[str], parameters: Mapping[str, torch.nn.Parameter]
) -> None:
    """拒绝 sparse 梯度合同和不同 Parameter 的重叠 storage。"""

    spans: list[tuple[str, int, int, int, str]] = []
    for name in names:
        parameter = parameters[name]
        if parameter.layout is not torch.strided:
            raise ValueError(f"MODEL_PARAMETER_LAYOUT_UNSUPPORTED:{name}")
        if parameter.grad is not None and parameter.grad.layout is not torch.strided:
            raise ValueError(f"MODEL_EXISTING_SPARSE_GRADIENT_UNSUPPORTED:{name}")
        if parameter.numel() == 0:
            continue
        storage = parameter.untyped_storage()
        minimum = int(parameter.storage_offset())
        maximum = minimum
        for size, stride in zip(parameter.shape, parameter.stride(), strict=True):
            extent = (int(size) - 1) * int(stride)
            minimum += min(0, extent)
            maximum += max(0, extent)
        item_size = int(parameter.element_size())
        span = (
            str(parameter.device),
            int(storage.data_ptr()),
            int(storage.data_ptr()) + minimum * item_size,
            int(storage.data_ptr()) + (maximum + 1) * item_size,
            name,
        )
        for other in spans:
            if (
                span[0] == other[0]
                and span[1] == other[1]
                and max(span[2], other[2]) < min(span[3], other[3])
            ):
                raise ValueError(f"MODEL_PARAMETER_STORAGE_OVERLAP:{other[4]}:{name}")
        spans.append(span)


@dataclass(slots=True)
class _ExecutionSnapshot:
    parameters: dict[str, torch.Tensor]
    buffers: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor | None]
    modes: dict[str, bool]
    python_rng: object
    numpy_rng: tuple[object, ...]
    torch_rng: torch.Tensor
    cuda_rng: tuple[torch.Tensor, ...] | None


def _take_snapshot(module: torch.nn.Module) -> _ExecutionSnapshot:
    parameters = {
        name: value.detach().clone()
        for name, value in module.named_parameters(remove_duplicate=True)
    }
    buffers = {
        name: value.detach().clone()
        for name, value in module.named_buffers(remove_duplicate=True)
    }
    gradients = {
        name: None if value.grad is None else value.grad.detach().clone()
        for name, value in module.named_parameters(remove_duplicate=True)
    }
    modes = {name: child.training for name, child in module.named_modules()}
    cuda_rng = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # pragma: no cover - CPU CI
        cuda_rng = tuple(value.clone() for value in torch.cuda.get_rng_state_all())
    return _ExecutionSnapshot(
        parameters=parameters,
        buffers=buffers,
        gradients=gradients,
        modes=modes,
        python_rng=random.getstate(),
        numpy_rng=np.random.get_state(),
        torch_rng=torch.random.get_rng_state().clone(),
        cuda_rng=cuda_rng,
    )


def _tensor_mapping_changed(
    current: Mapping[str, torch.Tensor], expected: Mapping[str, torch.Tensor]
) -> bool:
    return set(current) != set(expected) or any(
        current[name].shape != expected[name].shape
        or current[name].dtype != expected[name].dtype
        or current[name].device != expected[name].device
        or not torch.equal(current[name].detach(), expected[name])
        for name in expected
    )


def _snapshot_drift(module: torch.nn.Module, snapshot: _ExecutionSnapshot) -> tuple[str, ...]:
    reasons: list[str] = []
    parameters = dict(module.named_parameters(remove_duplicate=True))
    buffers = dict(module.named_buffers(remove_duplicate=True))
    if _tensor_mapping_changed(parameters, snapshot.parameters):
        reasons.append("MODEL_PARAMETER_MUTATED")
    if _tensor_mapping_changed(buffers, snapshot.buffers):
        reasons.append("MODEL_BUFFER_MUTATED")
    modes = {name: child.training for name, child in module.named_modules()}
    if modes != snapshot.modes:
        reasons.append("MODEL_MODE_MUTATED")
    current_gradients = {
        name: value.grad for name, value in module.named_parameters(remove_duplicate=True)
    }
    if set(current_gradients) != set(snapshot.gradients):
        reasons.append("MODEL_GRADIENT_SET_MUTATED")
    else:
        for name, expected in snapshot.gradients.items():
            observed = current_gradients[name]
            if (expected is None) != (observed is None) or (
                expected is not None
                and observed is not None
                and (
                    observed.shape != expected.shape
                    or observed.dtype != expected.dtype
                    or observed.device != expected.device
                    or not torch.equal(observed.detach(), expected)
                )
            ):
                reasons.append("MODEL_EXISTING_GRAD_MUTATED")
                break
    return tuple(reasons)


def _restore_snapshot(module: torch.nn.Module, snapshot: _ExecutionSnapshot) -> None:
    """恢复模型和 RNG；任一恢复失败都不被吞掉。"""

    parameters = dict(module.named_parameters(remove_duplicate=True))
    buffers = dict(module.named_buffers(remove_duplicate=True))
    if set(parameters) != set(snapshot.parameters) or set(buffers) != set(snapshot.buffers):
        raise RuntimeError("MODEL_STATE_STRUCTURE_MUTATED_UNRECOVERABLE")
    with torch.no_grad():
        for name, expected in snapshot.parameters.items():
            parameters[name].copy_(expected)
        for name, expected in snapshot.buffers.items():
            buffers[name].copy_(expected)
    for name, parameter in parameters.items():
        expected_gradient = snapshot.gradients[name]
        parameter.grad = (
            None
            if expected_gradient is None
            else expected_gradient.detach().clone().to(device=parameter.device)
        )
    modules = dict(module.named_modules())
    if set(modules) != set(snapshot.modes):
        raise RuntimeError("MODEL_MODULE_STRUCTURE_MUTATED_UNRECOVERABLE")
    for name, training in snapshot.modes.items():
        modules[name].training = training
    random.setstate(snapshot.python_rng)  # type: ignore[arg-type]
    np.random.set_state(snapshot.numpy_rng)  # type: ignore[arg-type]
    torch.random.set_rng_state(snapshot.torch_rng)
    if snapshot.cuda_rng is not None:  # pragma: no cover - CPU CI
        torch.cuda.set_rng_state_all(list(snapshot.cuda_rng))


class InMemoryFrozenSampleResolver:
    """将单样本 ``TrainingMicrobatch`` 映射冻结为可重放 resolver。

    构造和 ``resolve`` 都防御性复制 tensor，因而调用方无法通过修改
    返回 batch 污染后续 draw。该实现既用于 CPU fixture，也是外部
    dataset adapter 需要满足的参考语义。
    """

    def __init__(
        self,
        samples: Mapping[Hashable, TrainingMicrobatch],
        *,
        resolver_id: str,
        loss_unit: str,
        statistical_unit: str,
        weight_unit: str,
        sampling_design: str,
        weights_exogenous: bool,
        common_mean_assumption: bool,
    ) -> None:
        if not samples or not resolver_id:
            raise ValueError("FROZEN_SAMPLE_RESOLVER_EMPTY")
        for name, value in (
            ("loss_unit", loss_unit),
            ("statistical_unit", statistical_unit),
            ("weight_unit", weight_unit),
            ("sampling_design", sampling_design),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"FROZEN_SAMPLE_RESOLVER_FIELD_EMPTY:{name}")
        if type(weights_exogenous) is not bool or type(common_mean_assumption) is not bool:
            raise TypeError("FROZEN_SAMPLE_RESOLVER_ASSUMPTION_NOT_BOOLEAN")
        copied: dict[Hashable, TrainingMicrobatch] = {}
        batch_ids: set[str] = set()
        for sample_id, batch in samples.items():
            try:
                canonical_json_bytes(sample_id)
            except (TypeError, ValueError) as exc:
                raise TypeError("FROZEN_SAMPLE_ID_NOT_CANONICAL_JSON") from exc
            if not isinstance(batch, TrainingMicrobatch):
                raise TypeError("FROZEN_SAMPLE_NOT_TRAINING_MICROBATCH")
            if batch.batch_id in batch_ids:
                raise ValueError("FROZEN_SAMPLE_BATCH_ID_DUPLICATE")
            batch_ids.add(batch.batch_id)
            copied[sample_id] = self._clone_batch(batch)
        self._samples = MappingProxyType(copied)
        self._sample_ids = tuple(copied)
        self._resolver_id = resolver_id
        self._loss_unit = loss_unit
        self._statistical_unit = statistical_unit
        self._weight_unit = weight_unit
        self._sampling_design = sampling_design
        self._weights_exogenous = weights_exogenous
        self._common_mean_assumption = common_mean_assumption

    @staticmethod
    def _clone_batch(batch: TrainingMicrobatch) -> TrainingMicrobatch:
        return TrainingMicrobatch(
            batch.batch_id,
            {name: value.detach().clone() for name, value in batch.payload.items()},
            batch.sample_ids,
            thaw_json_value(batch.metadata),
        )

    @property
    def resolver_id(self) -> str:
        return self._resolver_id

    @property
    def sample_ids(self) -> tuple[Hashable, ...]:
        return self._sample_ids

    @property
    def loss_unit(self) -> str:
        return self._loss_unit

    @property
    def statistical_unit(self) -> str:
        return self._statistical_unit

    @property
    def weight_unit(self) -> str:
        return self._weight_unit

    @property
    def sampling_design(self) -> str:
        return self._sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._common_mean_assumption

    def resolve(self, sample_id: Hashable) -> TrainingMicrobatch:
        try:
            batch = self._samples[sample_id]
        except KeyError as exc:
            raise KeyError(f"FROZEN_SAMPLE_ID_UNKNOWN:{sample_id!r}") from exc
        return self._clone_batch(batch)

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        contract = {
            "resolver_id": self.resolver_id,
            "sample_ids": list(self.sample_ids),
            "loss_unit": self.loss_unit,
            "statistical_unit": self.statistical_unit,
            "weight_unit": self.weight_unit,
            "sampling_design": self.sampling_design,
            "weights_exogenous": self.weights_exogenous,
            "common_mean_assumption": self.common_mean_assumption,
        }
        _length_prefixed(digest, canonical_json_bytes(contract))
        for sample_id in self.sample_ids:
            batch = self._samples[sample_id]
            _length_prefixed(digest, canonical_json_bytes(sample_id))
            _length_prefixed(
                digest,
                canonical_json_bytes(
                    {
                        "batch_id": batch.batch_id,
                        "sample_ids": list(batch.sample_ids),
                        "metadata": thaw_json_value(batch.metadata),
                    }
                ),
            )
            for name in sorted(batch.payload):
                _update_tensor_digest(
                    digest, role="sample_payload", name=name, value=batch.payload[name]
                )
        return digest.hexdigest()

    def assert_unchanged(self, expected_digest: str) -> None:
        _validate_digest(expected_digest, field_name="expected resolver digest")
        if self.state_digest() != expected_digest:
            raise RuntimeError("FROZEN_SAMPLE_RESOLVER_STATE_CHANGED")


class TorchFixedStateGradientProvider:
    """从 ``ModelAdapter`` 与冻结 resolver 计算固定状态加权 mean gradient。"""

    def __init__(
        self,
        model: ModelAdapter,
        resolver: FrozenSampleResolver,
        *,
        fixed_state_id: str,
        registry: ParameterRegistry | None = None,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        if not isinstance(model, ModelAdapter):
            raise TypeError("MODEL_ADAPTER_PROTOCOL_NOT_IMPLEMENTED")
        if not isinstance(resolver, FrozenSampleResolver):
            raise TypeError("FROZEN_SAMPLE_RESOLVER_PROTOCOL_NOT_IMPLEMENTED")
        if not fixed_state_id:
            raise ValueError("FIXED_STATE_ID_EMPTY")
        if output_dtype not in {torch.float32, torch.float64}:
            raise ValueError("FIXED_STATE_GRADIENT_DTYPE_UNSUPPORTED")
        for field_name in (
            "resolver_id",
            "loss_unit",
            "statistical_unit",
            "weight_unit",
            "sampling_design",
        ):
            value = getattr(resolver, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"RESOLVER_CONTRACT_FIELD_EMPTY:{field_name}")
        if type(resolver.weights_exogenous) is not bool or type(
            resolver.common_mean_assumption
        ) is not bool:
            raise TypeError("RESOLVER_WEIGHTING_ASSUMPTION_NOT_BOOLEAN")
        _validate_digest(resolver.state_digest(), field_name="resolver state_digest")

        names, model_parameters, aliases = _named_parameters_with_aliases(model.module)
        _assert_dense_unique_parameters(names, model_parameters)
        if registry is None:
            selected_names = names
            selected_parameters = model_parameters
            registry_hash = _derived_registry_hash(names, model_parameters, aliases)
        else:
            if not isinstance(registry, ParameterRegistry):
                raise TypeError("registry 必须是 ParameterRegistry")
            selected_names = registry.eligible_names
            if not selected_names:
                raise ValueError("PARAMETER_REGISTRY_ELIGIBLE_SET_EMPTY")
            selected_parameters = {}
            for name in selected_names:
                parameter = registry.parameter(name)
                if name not in model_parameters or model_parameters[name] is not parameter:
                    raise ValueError(f"PARAMETER_REGISTRY_MODEL_MISMATCH:{name}")
                selected_parameters[name] = parameter  # type: ignore[assignment]
            registry_hash = registry.coordinate_registry_hash
        devices = {parameter.device for parameter in selected_parameters.values()}
        if len(devices) != 1:
            raise ValueError("FIXED_STATE_PROVIDER_MULTI_DEVICE_UNSUPPORTED")

        self._model = model
        self._resolver = resolver
        self._fixed_state_id = fixed_state_id
        self._parameter_names = tuple(selected_names)
        self._parameters = MappingProxyType(dict(selected_parameters))
        self._registry_hash = _validate_digest(
            registry_hash, field_name="coordinate registry hash"
        )
        self._device = next(iter(devices))
        self._output_dtype = output_dtype

    @property
    def registry_hash(self) -> str:
        return self._registry_hash

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    @property
    def fixed_state_id(self) -> str:
        return self._fixed_state_id

    @property
    def statistical_unit(self) -> str:
        return self._resolver.statistical_unit

    @property
    def weight_unit(self) -> str:
        return self._resolver.weight_unit

    @property
    def sampling_design(self) -> str:
        return self._resolver.sampling_design

    @property
    def weights_exogenous(self) -> bool:
        return self._resolver.weights_exogenous

    @property
    def common_mean_assumption(self) -> bool:
        return self._resolver.common_mean_assumption

    @property
    def output_dtype(self) -> torch.dtype:
        """返回 provider 明确的梯度输出 dtype。"""

        return self._output_dtype

    @property
    def model_adapter(self) -> ModelAdapter:
        """返回绑定模型 adapter，供 Stage 3 只读路径状态上下文使用。

        调用方不得直接修改该对象；路径求值应使用
        :meth:`gradient_at_parameter_state`，由 provider 统一完成状态安装、验证和
        恢复。公开这一只读引用是为了构造与训练一致的 ParameterRegistry，而不是
        暴露一个绕过固定状态合同的后门。
        """

        return self._model

    def gradient_at_parameter_state(
        self,
        parameters: Mapping[str, torch.Tensor],
        draws: Sequence[object],
        *,
        buffers: Mapping[str, torch.Tensor] | None = None,
        model_modes: Mapping[str, bool] | None = None,
    ) -> GradientBatch:
        """在临时路径节点状态计算梯度，并无条件恢复 provider 原状态。

        ``parameters`` 必须完整覆盖 provider 的 canonical 坐标；``buffers`` 若给出，
        必须完整覆盖模型 buffer。该方法先保存 provider 原状态，再安装路径节点，
        调用常规 :meth:`gradient`（它自身仍执行一次固定状态防污染检查），最后恢复
        原状态并核对摘要。因此一次节点求值既不会推进数据 resolver，也不会让路径
        插值残留在后续 Stage 2/3 任务中。
        """

        if set(parameters) != set(self.parameter_names):
            raise ValueError("PATH_PARAMETER_STATE_NAMES_MISMATCH")
        module_parameters = dict(
            self._model.module.named_parameters(remove_duplicate=True)
        )
        if any(
            name not in module_parameters
            or tuple(parameters[name].shape) != tuple(module_parameters[name].shape)
            or not parameters[name].is_floating_point()
            for name in self.parameter_names
        ):
            raise ValueError("PATH_PARAMETER_STATE_SHAPE_OR_DTYPE_MISMATCH")
        module_buffers = dict(self._model.module.named_buffers(remove_duplicate=True))
        if buffers is not None:
            if set(buffers) != set(module_buffers) or any(
                tuple(buffers[name].shape) != tuple(module_buffers[name].shape)
                or buffers[name].dtype != module_buffers[name].dtype
                for name in module_buffers
            ):
                raise ValueError("PATH_BUFFER_STATE_MISMATCH")
        modules = dict(self._model.module.named_modules())
        if model_modes is not None and (
            set(model_modes) != set(modules)
            or any(type(value) is not bool for value in model_modes.values())
        ):
            raise ValueError("PATH_MODEL_MODE_STATE_MISMATCH")

        before_digest = self.state_digest()
        snapshot = _take_snapshot(self._model.module)
        result: GradientBatch | None = None
        error: BaseException | None = None
        try:
            with torch.no_grad():
                for name in self.parameter_names:
                    target = module_parameters[name]
                    target.copy_(
                        parameters[name].detach().to(
                            device=target.device, dtype=target.dtype
                        )
                    )
                if buffers is not None:
                    for name, target in module_buffers.items():
                        target.copy_(
                            buffers[name].detach().to(
                                device=target.device, dtype=target.dtype
                            )
                        )
            if model_modes is not None:
                for name, training in model_modes.items():
                    modules[name].training = training
            result = self.gradient(draws)
        except BaseException as exc:
            error = exc
        finally:
            _restore_snapshot(self._model.module, snapshot)
        if self.state_digest() != before_digest:
            raise RuntimeError("PATH_PROVIDER_STATE_CHANGED_AFTER_RESTORE") from error
        if error is not None:
            raise error
        assert result is not None
        return result

    def state_digest(self) -> str:
        """摘要全部参数、buffer、模式、RNG 与 resolver 状态。"""

        digest = hashlib.sha256()
        _length_prefixed(
            digest,
            canonical_json_bytes(
                {
                    "fixed_state_id": self.fixed_state_id,
                    "registry_hash": self.registry_hash,
                    "parameter_names": list(self.parameter_names),
                    "task_type": self._model.task_type,
                    "resolver_id": self._resolver.resolver_id,
                    "resolver_digest": self._resolver.state_digest(),
                    "output_dtype": str(self.output_dtype),
                }
            ),
        )
        for name, parameter in self._model.module.named_parameters(remove_duplicate=True):
            _update_tensor_digest(digest, role="parameter", name=name, value=parameter)
        for name, buffer in self._model.module.named_buffers(remove_duplicate=True):
            _update_tensor_digest(digest, role="buffer", name=name, value=buffer)
        _length_prefixed(
            digest,
            canonical_json_bytes(
                [
                    [name, child.training]
                    for name, child in self._model.module.named_modules()
                ]
            ),
        )
        _length_prefixed(digest, repr(random.getstate()).encode("ascii"))
        numpy_state = np.random.get_state()
        _length_prefixed(digest, str(numpy_state[0]).encode("ascii"))
        _length_prefixed(digest, np.asarray(numpy_state[1], dtype=np.uint32).tobytes())
        _length_prefixed(
            digest,
            f"{int(numpy_state[2])}:{int(numpy_state[3])}:{float(numpy_state[4]).hex()}".encode(
                "ascii"
            ),
        )
        _length_prefixed(digest, _tensor_bytes(torch.random.get_rng_state()))
        if torch.cuda.is_available() and torch.cuda.is_initialized():  # pragma: no cover
            for index, state in enumerate(torch.cuda.get_rng_state_all()):
                _update_tensor_digest(
                    digest, role="cuda_rng", name=str(index), value=state
                )
        return digest.hexdigest()

    def assert_unchanged(self, expected_digest: str) -> None:
        _validate_digest(expected_digest, field_name="expected provider digest")
        if self.state_digest() != expected_digest:
            raise RuntimeError("TORCH_FIXED_STATE_PROVIDER_STATE_CHANGED")

    def gradient(self, draws: Sequence[object]) -> GradientBatch:
        """
        计算 draw group 的 effective-count 加权 mean gradient。

        输入顺序和 sample-ID 碰撞被完整保留。返回梯度与参数同
        device，并显式转为 ``output_dtype``；任一 ``None``、sparse 或
        非有限梯度都会失败。
        """

        if not draws:
            raise ValueError("FIXED_STATE_DRAWS_EMPTY")
        sample_ids = tuple(_draw_sample_id(draw) for draw in draws)
        before_digest = self.state_digest()
        resolver_digest = self._resolver.state_digest()
        snapshot = _take_snapshot(self._model.module)
        result: GradientBatch | None = None
        computation_error: BaseException | None = None
        drift_reasons: tuple[str, ...] = ()
        try:
            losses = []
            total_count = 0
            observed_loss_unit: str | None = None
            with torch.enable_grad():
                for sample_id in sample_ids:
                    resolved = self._resolver.resolve(sample_id)
                    if not isinstance(resolved, TrainingMicrobatch):
                        raise TypeError("RESOLVER_RESULT_NOT_TRAINING_MICROBATCH")
                    batch = resolved.to(self._device)
                    loss = self._model.loss(batch)
                    if loss.statistical_unit != self._resolver.loss_unit:
                        raise ValueError("RESOLVER_LOSS_UNIT_MISMATCH")
                    if observed_loss_unit is None:
                        observed_loss_unit = loss.statistical_unit
                    elif observed_loss_unit != loss.statistical_unit:
                        raise ValueError("DRAW_LOSS_STATISTICAL_UNIT_MISMATCH")
                    losses.append(loss.loss_numerator)
                    total_count += loss.effective_count
                if total_count <= 0:  # LossBatch 已拒绝 0，此处保留组合边界
                    raise ValueError("FIXED_STATE_EFFECTIVE_COUNT_ZERO")
                numerator = losses[0]
                for value in losses[1:]:
                    numerator = numerator + value
                mean_loss = numerator / total_count
                gradients = torch.autograd.grad(
                    mean_loss,
                    tuple(self._parameters[name] for name in self.parameter_names),
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=False,
                )
            output: dict[str, torch.Tensor] = {}
            for name, gradient in zip(self.parameter_names, gradients, strict=True):
                if gradient.layout is not torch.strided:
                    raise ValueError(f"FIXED_STATE_SPARSE_GRADIENT:{name}")
                converted = gradient.detach().to(dtype=self.output_dtype).clone()
                if not bool(torch.isfinite(converted).all()):
                    raise ValueError(f"FIXED_STATE_NONFINITE_GRADIENT:{name}")
                output[name] = converted
            result = GradientBatch(
                gradients=output,
                statistical_weight=float(total_count),
                statistical_unit=self.statistical_unit,
                weight_unit=self.weight_unit,
                sampling_design=self.sampling_design,
                weights_exogenous=self.weights_exogenous,
                common_mean_assumption=self.common_mean_assumption,
                sample_ids=sample_ids,
                loss=float(mean_loss.detach().to(dtype=torch.float64).cpu().item()),
            )
        except BaseException as exc:  # 先恢复状态，再原样传播任务异常
            computation_error = exc
        finally:
            drift_reasons = _snapshot_drift(self._model.module, snapshot)
            _restore_snapshot(self._model.module, snapshot)

        resolver_error: BaseException | None = None
        try:
            self._resolver.assert_unchanged(resolver_digest)
        except BaseException as exc:
            resolver_error = exc
        after_digest = self.state_digest()
        if after_digest != before_digest:
            raise RuntimeError("TORCH_FIXED_STATE_DIGEST_CHANGED_AFTER_RESTORE") from (
                resolver_error or computation_error
            )
        if drift_reasons:
            raise RuntimeError(";".join(drift_reasons)) from computation_error
        if resolver_error is not None:
            raise RuntimeError("FROZEN_SAMPLE_RESOLVER_MUTATED_DURING_GRADIENT") from resolver_error
        if computation_error is not None:
            raise computation_error
        assert result is not None
        return result


__all__ = [
    "InMemoryFrozenSampleResolver",
    "TorchFixedStateGradientProvider",
]
