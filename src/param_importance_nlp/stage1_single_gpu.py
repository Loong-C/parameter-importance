"""Real Pythia-14M worker primitives for Stage 1 S1.7.

The formalizer prepares a small, hash-bound safetensors fixture from the
qualified Pile mmap files *before* it holds a GPU lease.  This fresh-process
worker then sees precisely one GPU UUID, loads the offline Pythia assets, and
performs the fixed-state and two-step training checks.  Large numerical output
is safetensors plus a strict manifest; JSON contains identities and summaries
only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

try:  # unavailable on Windows development hosts
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]

from .atomic import sha256_file
from .contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from .core.estimators import (
    double_sample_importance,
    explicit_ordered_pair_u_reference,
    raw_importance,
    weighted_u_importance,
)
from .core.registry import ParameterRegistry
from .core.sufficient_statistics import WeightedSufficientStatistics
from .core.tensors import TensorMap
from .offline_guard import finalize_offline_guard, install_from_environment
from .providers.training import DeterministicBatchCursor, TorchModelAdapter, TrainingMicrobatch
from .runtime.training import AttemptCommitEvent, GradientReadyEvent, ParameterPostEvent, SkippedAttemptEvent, TrainingEngine, TrainingRunSpec, TrainingStepObserver


TASK_ID = "stage1.07_single_gpu_pythia14m"
FIXTURE_ID = "stage1-s17-pythia14m-pile16-v1"
WORKER_PLAN_SCHEMA = "stage1-s1-7-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage1-s1-7-worker-report-v1"
ARRAY_MANIFEST_SCHEMA = "stage1-s1-7-safetensors-manifest-v1"
T32_ATOL = 1.0e-7
T32_RTOL = 1.0e-5
T32_NORMALIZED_L2_LIMIT = 1.0e-5
EXPECTED_MODEL_CONFIG_VOCAB_SIZE = 50_304
EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE = 50_277
PYTHIA_ACTIVE_DROPOUT_FIELDS = ("attention_dropout", "hidden_dropout")
ACCUMULATOR_FIELDS = (
    "actual_update_raw_importance", "data_displacement", "data_movement",
    "initial_parameters", "last_parameters", "magnitude", "negative_mass",
    "positive", "raw", "raw_clipped", "total_displacement", "total_movement",
    "weight_decay_displacement", "weight_decay_movement",
)


class Stage1S17WorkerError(RuntimeError):
    """One S1.7 real-GPU worker invariant failed."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"stage1-s1-7-tensor-map-v1\x00")
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(b"\x00")
        digest.update(str(value.dtype).encode("ascii")); digest.update(b"\x00")
        digest.update(str(tuple(value.shape)).encode("ascii")); digest.update(b"\x00")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


_ADAMW_GROUP_KEYS = frozenset({
    "params", "lr", "betas", "eps", "weight_decay", "amsgrad", "maximize",
    "foreach", "capturable", "differentiable", "fused",
    "decoupled_weight_decay",
})


def _adamw_parameter_binding(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> tuple[list[str], list[torch.nn.Parameter], Mapping[str, object]]:
    if type(optimizer) is not torch.optim.AdamW or len(optimizer.param_groups) != 1:
        raise Stage1S17WorkerError("S17_WORKER_ADAMW_GROUP_COUNT_OR_TYPE_INVALID")
    group = optimizer.param_groups[0]
    if set(group) != _ADAMW_GROUP_KEYS:
        raise Stage1S17WorkerError("S17_WORKER_ADAMW_GROUP_FIELDS_INVALID")
    named = list(module.named_parameters())
    names = [name for name, _ in named]
    parameters = [parameter for _, parameter in named]
    actual = group.get("params")
    if (
        not names
        or not isinstance(actual, list)
        or len(actual) != len(parameters)
        or any(observed is not expected for observed, expected in zip(actual, parameters, strict=True))
        or any(not parameter.requires_grad for parameter in parameters)
    ):
        raise Stage1S17WorkerError("S17_WORKER_ADAMW_PARAMETER_BINDING_INVALID")
    return names, parameters, group


def _adamw_group_wire(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, object]:
    """Freeze the exact single-group AdamW control wire used by S1.7."""

    names, _parameters, group = _adamw_parameter_binding(module, optimizer)
    expected_numbers = {"lr": 3.0e-4, "eps": 1.0e-8, "weight_decay": 0.01}
    for key, expected in expected_numbers.items():
        value = group[key]
        if type(value) is not float or not math.isfinite(value) or value != expected:
            raise Stage1S17WorkerError(f"S17_WORKER_ADAMW_NUMERIC_OPTION_INVALID:{key}")
    betas = group["betas"]
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(type(item) is not float or not math.isfinite(item) for item in betas)
        or list(betas) != [0.9, 0.999]
    ):
        raise Stage1S17WorkerError("S17_WORKER_ADAMW_BETAS_INVALID")
    expected_booleans = {
        "amsgrad": False, "maximize": False, "foreach": False,
        "capturable": False, "differentiable": False, "fused": False,
        "decoupled_weight_decay": True,
    }
    for key, expected in expected_booleans.items():
        if group[key] is not expected:
            raise Stage1S17WorkerError(f"S17_WORKER_ADAMW_BOOLEAN_OPTION_INVALID:{key}")
    return {
        "optimizer_type": "AdamW",
        "parameter_names": names,
        **expected_numbers,
        "betas": [float(item) for item in betas],
        **expected_booleans,
    }


def _optimizer_state_tensors(
    module: torch.nn.Module, optimizer: torch.optim.Optimizer,
) -> dict[str, torch.Tensor]:
    """Return the exact empty-or-complete AdamW tensor state by parameter name."""

    names, parameters, _group = _adamw_parameter_binding(module, optimizer)
    _adamw_group_wire(module, optimizer)
    if not optimizer.state:
        return {}
    expected_ids = {id(parameter) for parameter in parameters}
    if {id(parameter) for parameter in optimizer.state} != expected_ids:
        raise Stage1S17WorkerError("S17_WORKER_ADAMW_STATE_PARAMETER_SET_INVALID")
    tensors: dict[str, torch.Tensor] = {}
    for name, parameter in zip(names, parameters, strict=True):
        state = optimizer.state.get(parameter)
        if not isinstance(state, Mapping) or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise Stage1S17WorkerError(f"S17_WORKER_ADAMW_STATE_FIELDS_INVALID:{name}")
        for field in ("step", "exp_avg", "exp_avg_sq"):
            value = state[field]
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all().item()):
                raise Stage1S17WorkerError(f"S17_WORKER_ADAMW_STATE_TENSOR_INVALID:{name}:{field}")
            if field == "step":
                valid_layout = (
                    value.ndim == 0
                    and value.dtype == torch.float32
                    and value.device.type == "cpu"
                    and value.layout == torch.strided
                    and value.stride() == ()
                    and not value.requires_grad
                )
            else:
                valid_layout = (
                    value.shape == parameter.shape
                    and value.dtype == parameter.dtype
                    and value.device == parameter.device
                    and value.layout == parameter.layout
                    and value.stride() == parameter.stride()
                    and not value.requires_grad
                )
            if not valid_layout:
                raise Stage1S17WorkerError(f"S17_WORKER_ADAMW_STATE_LAYOUT_INVALID:{name}:{field}")
            tensors[f"{name}:{field}"] = value.detach()
    return tensors


def _state_components(module: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: object | None = None) -> dict[str, object]:
    if scheduler is not None:
        raise Stage1S17WorkerError("S17_WORKER_SCHEDULER_MUST_BE_NULL")
    payload: dict[str, object] = {
        "parameters_sha256": _tensor_digest({name: item.detach() for name, item in module.named_parameters()}),
        "buffers_sha256": _tensor_digest({name: item.detach() for name, item in module.named_buffers()}),
        "model_modes_sha256": canonical_json_hash({name: bool(child.training) for name, child in module.named_modules()}),
        "torch_cpu_rng_sha256": _sha_bytes(torch.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda_rng_sha256": _sha_bytes(torch.cuda.get_rng_state(0).cpu().numpy().tobytes()),
        "python_rng_sha256": _sha_bytes(repr(random.getstate()).encode("utf-8")),
        "numpy_rng_sha256": _sha_bytes(repr(np.random.get_state()).encode("utf-8")),
        "scheduler": None,
    }
    if optimizer is not None:
        optimizer_tensors = _optimizer_state_tensors(module, optimizer)
        payload["optimizer_tensors_sha256"] = _tensor_digest(optimizer_tensors)
        payload["optimizer_groups_sha256"] = canonical_json_hash(
            _adamw_group_wire(module, optimizer)
        )
    else:
        payload["optimizer_tensors_sha256"] = None; payload["optimizer_groups_sha256"] = None
    return payload


def _state_digest(module: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, scheduler: object | None = None) -> str:
    return canonical_json_hash(_state_components(module, optimizer, scheduler))


def _optimizer_tensor_values(module: torch.nn.Module, optimizer: torch.optim.Optimizer) -> dict[str, torch.Tensor]:
    """Name optimizer tensors deterministically for post-step parity checks."""

    values = {
        name: tensor.detach().cpu().clone()
        for name, tensor in _optimizer_state_tensors(module, optimizer).items()
    }
    if not values:
        raise Stage1S17WorkerError("S17_WORKER_OPTIMIZER_STATE_EMPTY_AFTER_STEP")
    return values


def _configure_determinism(fixture: Mapping[str, Any]) -> None:
    contract = _require_object(fixture.get("execution_contract"), field="fixture.execution_contract")
    expected = _require_object(contract.get("determinism"), field="fixture.determinism")
    if expected != {"model_seed": 1707, "training_seed": 2707, "deterministic_algorithms": True, "allow_tf32": False, "cublas_workspace_config": ":4096:8"}:
        raise Stage1S17WorkerError("S17_WORKER_DETERMINISM_CONTRACT_INVALID")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(expected["cublas_workspace_config"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise Stage1S17WorkerError("S17_WORKER_DETERMINISM_CONFIGURATION_FAILED")


def _rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) > 1 and fields[1].isdigit():
                    return int(fields[1]) * 1024
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(usage) * 1024 if usage else None


def _memory(phase: str) -> dict[str, int | str | None]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage1S17WorkerError("S17_WORKER_SINGLE_CUDA_DEVICE_REQUIRED")
    return {
        "phase": phase,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "cpu_rss_bytes": _rss_bytes(),
    }


def _require_object(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage1S17WorkerError(f"S17_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _validate_plan(path: Path) -> dict[str, Any]:
    value = _require_object(load_canonical_json(path), field="worker_plan")
    expected = {
        "schema_version", "task_id", "execution_commit", "approved_gpu_uuid", "physical_gpu_index",
        "fixture", "fixture_safetensors_ref", "fixture_safetensors_sha256", "model_root",
        "tokenizer_root", "cache_root", "assets", "output_ref", "run_token", "artifact_hash",
    }
    if set(value) != expected or value.get("schema_version") != WORKER_PLAN_SCHEMA:
        raise Stage1S17WorkerError("S17_WORKER_PLAN_FIELDS_INVALID")
    body = dict(value); declared = body.pop("artifact_hash")
    if declared != canonical_json_hash(body):
        raise Stage1S17WorkerError("S17_WORKER_PLAN_HASH_INVALID")
    if value["task_id"] != TASK_ID or not isinstance(value["execution_commit"], str) or len(value["execution_commit"]) != 40:
        raise Stage1S17WorkerError("S17_WORKER_PLAN_IDENTITY_INVALID")
    uuid = value["approved_gpu_uuid"]
    if not isinstance(uuid, str) or not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise Stage1S17WorkerError("S17_WORKER_CUDA_VISIBLE_DEVICES_MISMATCH")
    if isinstance(value["physical_gpu_index"], bool) or not isinstance(value["physical_gpu_index"], int) or value["physical_gpu_index"] < 0:
        raise Stage1S17WorkerError("S17_WORKER_PHYSICAL_MAPPING_INVALID")
    if not isinstance(value["run_token"], str) or len(value["run_token"]) < 16:
        raise Stage1S17WorkerError("S17_WORKER_RUN_TOKEN_INVALID")
    fixture = _require_object(value["fixture"], field="fixture")
    if fixture.get("fixture_id") != FIXTURE_ID or fixture.get("fixture_hash") != canonical_json_hash({key: item for key, item in fixture.items() if key != "fixture_hash"}):
        raise Stage1S17WorkerError("S17_WORKER_FIXTURE_HASH_INVALID")
    batching = _require_object(fixture.get("batching"), field="fixture.batching")
    if batching != {"global_batch_size": 4, "microbatch_size": 1, "accumulation_steps": 4, "world_size": 1}:
        raise Stage1S17WorkerError("S17_WORKER_BATCHING_CONTRACT_INVALID")
    records = _require_object(fixture.get("records"), field="fixture.records")
    if records != {"a": [0, 1, 2, 3], "b": [4, 5, 6, 7], "training": [[8, 9, 10, 11], [12, 13, 14, 15]]}:
        raise Stage1S17WorkerError("S17_WORKER_RECORD_FIXTURE_INVALID")
    fixture_identity = _require_object(fixture.get("asset_identity"), field="fixture.asset_identity")
    if fixture_identity != {
        "model": {"logical_name": "pythia-14m-step0", "asset_id": "11dd681a22649a451b9be53c255bb4e9f83207c3f22f75f1eec53a33b7776fd2", "revision": "56079904bb80b7f36d3b794089f146e7a4d6efae", "ready_manifest_sha256": "7d3404906f3dd00c0d0314863f706c5df01f1db1fc0e0b4cf501353b88963d1e", "parameter_count": 14067712, "config_vocab_size": EXPECTED_MODEL_CONFIG_VOCAB_SIZE},
        "tokenizer": {"logical_name": "pythia-tokenizer", "asset_id": "b5eebc43fe88687e5bf692761f1db25f91e8d6f9a8cceaa2342d2624ac1f652d", "revision": "e361f9afd54b3e7856879eead5326d36ff6f32d7", "ready_manifest_sha256": "ea59f3f8e37321208701326b2ea88b7491450a88eae870775beeff027d102794", "vocab_size": 50277},
        "pile": {"logical_name": "pile-selected-prefix", "asset_id": "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad", "revision": "4647773ea142ab1ff5694602fa104bbf49088408", "ready_manifest_sha256": "345cd0f49d35ad9543daa3f95118013c55bdd729ed87fdec3c7a7c93ae449f8b"},
    }:
        raise Stage1S17WorkerError("S17_WORKER_FIXTURE_ASSET_IDENTITY_INVALID")
    contract = _require_object(fixture.get("execution_contract"), field="fixture.execution_contract")
    expected_contract = {"model_mode": "train", "random_layer_policy": "all_pythia_dropout_probabilities_zero", "precision": {"compute": "float32", "gradient": "float32", "statistics": "float32", "reference": "float64", "amp": False}, "loss": {"task_type": "causal_lm", "reduction": "mean", "valid_tokens_per_microbatch": 2048, "ignore_index": -100}, "optimizer": {"type": "AdamW", "learning_rate": 0.0003, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "foreach": False, "fused": False}, "gradient_clip_max_norm": 1.0, "scheduler": None, "statistical_contract": {"estimator_name": "u", "statistical_unit": "microbatch_mean_gradient", "weight_unit": "effective_target_tokens", "sampling_design": "ordered_disjoint_microbatches", "weights_exogenous": True, "common_mean_assumption": True}, "determinism": {"model_seed": 1707, "training_seed": 2707, "deterministic_algorithms": True, "allow_tf32": False, "cublas_workspace_config": ":4096:8"}}
    actual_without_dropout = {key: value for key, value in contract.items() if key != "dropout_probabilities"}
    declared_dropout = contract.get("dropout_probabilities")
    if actual_without_dropout != expected_contract or not isinstance(declared_dropout, Mapping) or set(declared_dropout) != set(PYTHIA_ACTIVE_DROPOUT_FIELDS) or any(isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0.0 for value in declared_dropout.values()):
        raise Stage1S17WorkerError("S17_WORKER_EXECUTION_CONTRACT_INVALID")
    assets = _require_object(value["assets"], field="assets")
    if set(assets) != {"model", "tokenizer", "pile"}:
        raise Stage1S17WorkerError("S17_WORKER_ASSET_ROLE_SET_INVALID")
    expected_assets = {
        "model": {"parameter_count": 14067712, "config_vocab_size": EXPECTED_MODEL_CONFIG_VOCAB_SIZE, "revision": "56079904bb80b7f36d3b794089f146e7a4d6efae", "asset_id": "11dd681a22649a451b9be53c255bb4e9f83207c3f22f75f1eec53a33b7776fd2", "ready_manifest_sha256": "7d3404906f3dd00c0d0314863f706c5df01f1db1fc0e0b4cf501353b88963d1e"},
        "tokenizer": {"revision": "e361f9afd54b3e7856879eead5326d36ff6f32d7", "asset_id": "b5eebc43fe88687e5bf692761f1db25f91e8d6f9a8cceaa2342d2624ac1f652d", "ready_manifest_sha256": "ea59f3f8e37321208701326b2ea88b7491450a88eae870775beeff027d102794"},
    }
    if any(_require_object(assets.get(role), field=f"assets.{role}") != expected for role, expected in expected_assets.items()):
        raise Stage1S17WorkerError("S17_WORKER_FROZEN_ASSET_IDENTITY_INVALID")
    pile = _require_object(assets.get("pile"), field="assets.pile")
    expected_pile = {"revision": "4647773ea142ab1ff5694602fa104bbf49088408", "asset_id": "dbbfeb12bab4027b386bd97d604d8134699e96f79e309cceacff7999a55b5dad", "ready_manifest_sha256": "345cd0f49d35ad9543daa3f95118013c55bdd729ed87fdec3c7a7c93ae449f8b", "fixture_file_sha256": value["fixture_safetensors_sha256"], "hash_passes": 2}
    if any(pile.get(field) != expected for field, expected in expected_pile.items()) or set(pile) != {"revision", "asset_id", "ready_manifest_sha256", "full_hash_seconds", "qualified_resolution_hash_seconds", "dataset_rehash_seconds", "full_hash_bytes", "qualified_resolution_hashed_bytes", "dataset_rehash_bytes", "fixture_file_sha256", "hash_passes"} or any(not isinstance(pile[field], (int, float)) or isinstance(pile[field], bool) or float(pile[field]) <= 0 for field in ("full_hash_seconds", "qualified_resolution_hash_seconds", "dataset_rehash_seconds", "full_hash_bytes", "qualified_resolution_hashed_bytes", "dataset_rehash_bytes")) or int(pile["full_hash_bytes"]) != int(pile["qualified_resolution_hashed_bytes"]) + int(pile["dataset_rehash_bytes"]):
        raise Stage1S17WorkerError("S17_WORKER_FROZEN_PILE_IDENTITY_INVALID")
    for field in ("fixture_safetensors_ref", "fixture_safetensors_sha256", "model_root", "tokenizer_root", "cache_root", "output_ref"):
        if not isinstance(value[field], str) or not value[field]:
            raise Stage1S17WorkerError(f"S17_WORKER_PLAN_FIELD_INVALID:{field}")
    if not Path(value["cache_root"]).is_absolute() or Path(value["cache_root"]).name != "cache":
        raise Stage1S17WorkerError("S17_WORKER_CACHE_ROOT_INVALID")
    return value


def _device_probe(plan: Mapping[str, Any]) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage1S17WorkerError("S17_WORKER_SINGLE_CUDA_DEVICE_REQUIRED")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    if not properties.name.startswith("NVIDIA A100"):
        raise Stage1S17WorkerError(f"S17_WORKER_A100_REQUIRED:{properties.name}")
    return {
        "logical_device": 0,
        "physical_gpu_index_at_discovery": plan["physical_gpu_index"],
        "physical_gpu_uuid": plan["approved_gpu_uuid"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
    }


def _load_fixture(path: Path, plan: Mapping[str, Any]) -> dict[int, torch.Tensor]:
    if sha256_file(path) != plan["fixture_safetensors_sha256"]:
        raise Stage1S17WorkerError("S17_WORKER_FIXTURE_FILE_HASH_INVALID")
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - frozen CUDA env has dependency
        raise Stage1S17WorkerError("S17_WORKER_SAFETENSORS_UNAVAILABLE") from error
    values = load_file(str(path), device="cpu")
    expected = {f"record_{index:012d}" for index in range(16)}
    if set(values) != expected:
        raise Stage1S17WorkerError("S17_WORKER_FIXTURE_TENSOR_SET_INVALID")
    result: dict[int, torch.Tensor] = {}
    for index in range(16):
        item = values[f"record_{index:012d}"]
        if item.dtype != torch.int64 or tuple(item.shape) != (2049,) or not bool(torch.all((item >= 0) & (item < 50277))):
            raise Stage1S17WorkerError(f"S17_WORKER_FIXTURE_TENSOR_INVALID:{index}")
        result[index] = item.contiguous()
    token_hashes = _require_object(plan["fixture"].get("token_sha256"), field="fixture.token_sha256")
    for index, item in result.items():
        if token_hashes.get(str(index)) != _sha_bytes(item.numpy().tobytes(order="C")):
            raise Stage1S17WorkerError(f"S17_WORKER_FIXTURE_TOKEN_HASH_INVALID:{index}")
    return result


def _batch(tokens: Mapping[int, torch.Tensor], indexes: Sequence[int], *, label: str) -> TrainingMicrobatch:
    joined = torch.stack([tokens[index] for index in indexes], dim=0)
    input_ids, target_ids = joined[:, :-1], joined[:, 1:]
    return TrainingMicrobatch(
        f"s17:{label}",
        {"input_ids": input_ids, "target_ids": target_ids, "attention_mask": torch.ones_like(input_ids)},
        tuple(f"pile:record:{index:012d}" for index in indexes),
        {"record_indexes": list(indexes), "sequence_length": 2048},
    )


def _maps_from_gradients(registry: ParameterRegistry, gradients: Sequence[torch.Tensor | None]) -> dict[str, torch.Tensor]:
    values: dict[str, torch.Tensor] = {}
    for name, value in zip(registry.eligible_names, gradients, strict=True):
        parameter = registry.parameter(name)
        values[name] = (torch.zeros_like(parameter) if value is None else value.detach()).to(torch.float32).clone()
    if not values or not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise Stage1S17WorkerError("S17_WORKER_NONFINITE_GRADIENT")
    return values


def _gradient(adapter: TorchModelAdapter, registry: ParameterRegistry, batch: TrainingMicrobatch) -> tuple[dict[str, torch.Tensor], float, int]:
    model = adapter.module
    parameters = tuple(registry.parameter(name) for name in registry.eligible_names)
    model.zero_grad(set_to_none=True)
    loss = adapter.loss(batch.to(next(iter(parameters)).device))
    gradients = torch.autograd.grad(loss.mean_loss, parameters, allow_unused=True, retain_graph=False, create_graph=False)
    return _maps_from_gradients(registry, gradients), float(loss.mean_loss.detach().cpu().item()), int(loss.effective_count)


def _mean(samples: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if len(samples) != 4:
        raise Stage1S17WorkerError("S17_WORKER_FOUR_MICROBATCHES_REQUIRED")
    return {name: torch.stack([sample[name] for sample in samples], dim=0).mean(dim=0) for name in sorted(samples[0])}


def _max_error(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> dict[str, object]:
    if set(left) != set(right):
        raise Stage1S17WorkerError("S17_WORKER_PARAMETER_SET_DRIFT")
    worst = -1.0; name = ""; coordinate: list[int] = []; scaled_worst = -1.0; delta_sq = 0.0; reference_sq = 0.0; violations = 0; near_zero = 0
    per_tensor: dict[str, dict[str, object]] = {}
    for candidate in sorted(left):
        if left[candidate].shape != right[candidate].shape or left[candidate].dtype != torch.float32 or right[candidate].dtype != torch.float32 or left[candidate].device != right[candidate].device or not bool(torch.isfinite(left[candidate]).all()) or not bool(torch.isfinite(right[candidate]).all()):
            raise Stage1S17WorkerError(f"S17_WORKER_T32_TENSOR_INVALID:{candidate}")
        delta = (left[candidate].to(torch.float64) - right[candidate].to(torch.float64)).abs()
        reference = right[candidate].to(torch.float64).abs(); threshold = T32_ATOL + T32_RTOL * reference
        violations += int((delta > threshold).sum().item()); near_zero += int((reference <= T32_ATOL).sum().item())
        delta_sq += float(delta.square().sum().item()); reference_sq += float(reference.square().sum().item())
        scaled_worst = max(scaled_worst, float((delta / threshold).max().item()))
        value, flat = torch.max(delta.reshape(-1), dim=0)
        numeric = float(value.item())
        if numeric > worst:
            worst, name = numeric, candidate
            coordinate = [int(item) for item in torch.unravel_index(flat, delta.shape)]
        tensor_delta_sq = float(delta.square().sum().item())
        tensor_reference_sq = float(reference.square().sum().item())
        per_tensor[candidate] = {
            "max_abs_error": numeric,
            "max_scaled_error": float((delta / threshold).max().item()),
            "normalized_l2_error": math.sqrt(tensor_delta_sq) / max(math.sqrt(tensor_reference_sq), T32_ATOL),
            "near_zero_coordinates": int((reference <= T32_ATOL).sum().item()),
            "violation_count": int((delta > threshold).sum().item()),
            "within_t32": bool((delta <= threshold).all().item())
            and math.sqrt(tensor_delta_sq) / max(math.sqrt(tensor_reference_sq), T32_ATOL) <= T32_NORMALIZED_L2_LIMIT,
        }
    normalized_l2 = math.sqrt(delta_sq) / max(math.sqrt(reference_sq), T32_ATOL)
    return {"max_abs_error": worst, "parameter": name, "index": coordinate, "max_scaled_error": scaled_worst, "normalized_l2_error": normalized_l2, "near_zero_coordinates": near_zero, "violation_count": violations, "within_t32": violations == 0 and normalized_l2 <= T32_NORMALIZED_L2_LIMIT, "per_tensor": per_tensor}


def _allclose(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return bool(_max_error(left, right)["within_t32"])


def _encode_arrays(path: Path, arrays: Mapping[str, Mapping[str, torch.Tensor]]) -> dict[str, object]:
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:  # pragma: no cover
        raise Stage1S17WorkerError("S17_WORKER_SAFETENSORS_UNAVAILABLE") from error
    flat: dict[str, torch.Tensor] = {}
    maps: dict[str, list[str]] = {}
    for phase, values in sorted(arrays.items()):
        if not values:
            raise Stage1S17WorkerError(f"S17_WORKER_ARRAY_PHASE_EMPTY:{phase}")
        keys: list[str] = []
        for name, tensor in sorted(values.items()):
            if tensor.device.type != "cpu" or tensor.dtype not in {torch.float32, torch.float64} or not bool(torch.isfinite(tensor).all()):
                raise Stage1S17WorkerError(f"S17_WORKER_ARRAY_INVALID:{phase}:{name}")
            key = f"{phase}::{name}"
            flat[key] = tensor.contiguous()
            keys.append(key)
        maps[phase] = keys
    if path.exists():
        raise Stage1S17WorkerError("S17_WORKER_ARRAY_TARGET_EXISTS")
    save_file(flat, str(path), metadata={"schema_version": ARRAY_MANIFEST_SCHEMA})
    metadata: dict[str, dict[str, object]] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        if set(handle.keys()) != set(flat) or handle.metadata() != {"schema_version": ARRAY_MANIFEST_SCHEMA}:
            raise Stage1S17WorkerError("S17_WORKER_ARRAY_SERIALIZATION_DRIFT")
        for key in sorted(flat):
            item = handle.get_tensor(key)
            metadata[key] = {"dtype": str(item.dtype), "shape": list(item.shape), "sha256": _sha_bytes(item.contiguous().numpy().tobytes(order="C"))}
    body: dict[str, object] = {
        "schema_version": ARRAY_MANIFEST_SCHEMA,
        "file": path.name,
        "file_sha256": sha256_file(path),
        "file_size_bytes": path.stat().st_size,
        "maps": maps,
        "tensors": metadata,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    return body


def _active_gpt_neox_dropout(config: Mapping[str, Any]) -> dict[str, float]:
    """Return the only stochastic GPTNeoX CausalLM config fields used in forward."""

    values: dict[str, float] = {}
    for key in PYTHIA_ACTIVE_DROPOUT_FIELDS:
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise Stage1S17WorkerError(f"S17_WORKER_ACTIVE_DROPOUT_FIELD_INVALID:{key}")
        values[key] = float(value)
    return values


def _load_model(plan: Mapping[str, Any]) -> tuple[TorchModelAdapter, object]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:  # pragma: no cover - frozen CUDA env provides it
        raise Stage1S17WorkerError("S17_WORKER_TRANSFORMERS_UNAVAILABLE") from error
    tokenizer = AutoTokenizer.from_pretrained(str(plan["tokenizer_root"]), local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(str(plan["model_root"]), local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32)
    model.to("cuda:0"); model.train(True)
    metadata = _require_object(_require_object(plan["assets"], field="assets").get("model"), field="assets.model")
    expected_count = metadata.get("parameter_count")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != expected_count or metadata.get("config_vocab_size") != EXPECTED_MODEL_CONFIG_VOCAB_SIZE or int(getattr(model.config, "vocab_size", -1)) != metadata["config_vocab_size"] or len(tokenizer) != EXPECTED_TOKENIZER_RUNTIME_VOCAB_SIZE:
        raise Stage1S17WorkerError("S17_WORKER_MODEL_OR_TOKENIZER_IDENTITY_DRIFT")
    if model.__class__.__name__ != "GPTNeoXForCausalLM":
        raise Stage1S17WorkerError("S17_WORKER_MODEL_ARCHITECTURE_DRIFT")
    declared_dropout = _require_object(_require_object(plan["fixture"], field="fixture").get("execution_contract"), field="execution_contract").get("dropout_probabilities")
    observed_dropout = _active_gpt_neox_dropout(model.config.to_dict())
    if declared_dropout != {key: 0.0 for key in PYTHIA_ACTIVE_DROPOUT_FIELDS} or observed_dropout != declared_dropout or any(value != 0.0 for value in observed_dropout.values()) or model.training is not True:
        raise Stage1S17WorkerError("S17_WORKER_MODEL_RANDOM_OR_MODE_DRIFT")
    return TorchModelAdapter(model, task_type="causal_lm"), tokenizer


def _fixed_array_bundle(
    *, full: Mapping[str, torch.Tensor], online_mean: Mapping[str, torch.Tensor],
    raw: Mapping[str, torch.Tensor], double: Mapping[str, torch.Tensor],
    explicit_u: Mapping[str, torch.Tensor], streaming_u: Mapping[str, torch.Tensor],
    locals_a: Sequence[Mapping[str, torch.Tensor]], locals_b: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, Mapping[str, torch.Tensor]]:
    """Build the fixed-state safetensors phases with local A/B indices zero-based."""

    if len(locals_a) != 4 or len(locals_b) != 4:
        raise Stage1S17WorkerError("S17_WORKER_FIXED_LOCAL_ARRAY_COUNT_INVALID")
    arrays: dict[str, Mapping[str, torch.Tensor]] = {
        "full_gradient": {name: value.detach().cpu() for name, value in full.items()},
        "online_mean_gradient": {name: value.detach().cpu() for name, value in online_mean.items()},
        "raw": {name: value.detach().cpu() for name, value in raw.items()},
        "double": {name: value.detach().cpu() for name, value in double.items()},
        "explicit_u": {name: value.detach().cpu() for name, value in explicit_u.items()},
        "streaming_u": {name: value.detach().cpu() for name, value in streaming_u.items()},
    }
    for index, values in enumerate(locals_a):
        arrays[f"local_a_{index}"] = {name: value.detach().cpu() for name, value in values.items()}
    for index, values in enumerate(locals_b):
        arrays[f"local_b_{index}"] = {name: value.detach().cpu() for name, value in values.items()}
    return arrays


def _fixed_state(adapter: TorchModelAdapter, registry: ParameterRegistry, tokens: Mapping[int, torch.Tensor], optimizer: torch.optim.Optimizer) -> tuple[dict[str, object], dict[str, Mapping[str, torch.Tensor]]]:
    fixed_started = time.monotonic()
    before_components = _state_components(adapter.module, optimizer)
    before = canonical_json_hash(before_components)
    locals_a: list[dict[str, torch.Tensor]] = []
    locals_b: list[dict[str, torch.Tensor]] = []
    losses_a: list[float] = []
    losses_b: list[float] = []
    micro_resources: list[dict[str, object]] = []
    for index in range(4):
        started = time.monotonic()
        current, loss, count = _gradient(adapter, registry, _batch(tokens, [index], label=f"a-{index}"))
        if count != 2048: raise Stage1S17WorkerError("S17_WORKER_EFFECTIVE_TOKEN_COUNT_DRIFT")
        locals_a.append(current); losses_a.append(loss)
        micro_resources.append({"group": "a", "index": index, "backward_seconds": time.monotonic() - started, "memory": _memory(f"fixed_a_{index}")})
    full_started = time.monotonic()
    full, full_loss, full_count = _gradient(adapter, registry, _batch(tokens, [0, 1, 2, 3], label="a-full"))
    full_gradient_seconds = time.monotonic() - full_started
    if full_count != 8192: raise Stage1S17WorkerError("S17_WORKER_FULL_EFFECTIVE_TOKEN_COUNT_DRIFT")
    for index in range(4, 8):
        started = time.monotonic()
        current, loss, count = _gradient(adapter, registry, _batch(tokens, [index], label=f"b-{index}"))
        if count != 2048: raise Stage1S17WorkerError("S17_WORKER_EFFECTIVE_TOKEN_COUNT_DRIFT")
        locals_b.append(current); losses_b.append(loss)
        micro_resources.append({"group": "b", "index": index - 4, "backward_seconds": time.monotonic() - started, "memory": _memory(f"fixed_b_{index - 4}")})
    statistics_started = time.monotonic()
    online_mean = _mean(locals_a)
    sample_maps = [TensorMap(item, registry=registry) for item in locals_a]
    weights = [2048.0] * 4
    statistics = WeightedSufficientStatistics.from_samples(sample_maps, weights, accumulation_dtype=torch.float32, statistical_unit="microbatch_mean_gradient", weight_unit="effective_target_tokens", sampling_design="ordered_disjoint_microbatches", weights_exogenous=True, common_mean_assumption=True)
    raw = raw_importance(TensorMap(online_mean, registry=registry)).to_dict()
    explicit_u = explicit_ordered_pair_u_reference(sample_maps).to_dict()
    streaming_u = weighted_u_importance(statistics, require_unbiasedness_assumptions=True).to_dict()
    mean_b = _mean(locals_b)
    double = double_sample_importance(TensorMap(online_mean, registry=registry), TensorMap(mean_b, registry=registry)).to_dict()
    statistics_seconds = time.monotonic() - statistics_started
    after_components = _state_components(adapter.module, optimizer)
    after = canonical_json_hash(after_components)
    comparisons = {
        "full_vs_online": _max_error(full, online_mean),
        "explicit_u_vs_streaming_u": _max_error(explicit_u, streaming_u),
    }
    if not _allclose(full, online_mean) or not _allclose(explicit_u, streaming_u) or before != after:
        raise Stage1S17WorkerError("S17_WORKER_FIXED_STATE_PARITY_FAILED")
    arrays = _fixed_array_bundle(
        full=full, online_mean=online_mean, raw=raw, double=double,
        explicit_u=explicit_u, streaming_u=streaming_u,
        locals_a=locals_a, locals_b=locals_b,
    )
    return {
        "state_before_sha256": before, "state_after_sha256": after, "state_before_components": before_components, "state_after_components": after_components,
        "full_loss": full_loss, "local_losses_a": losses_a, "local_losses_b": losses_b,
        "comparisons": comparisons, "microbatch_count": 4, "effective_tokens_per_microbatch": 2048, "microbatch_resources": micro_resources,
        "fixed_wall_seconds": time.monotonic() - fixed_started, "full_gradient_seconds": full_gradient_seconds,
        "statistics_seconds": statistics_seconds,
    }, arrays


@dataclass(slots=True)
class _S17StepObserver(TrainingStepObserver):
    """Read-only step boundary capture for stats-off/on trajectory parity."""

    engine: TrainingEngine | None = None
    rows: list[dict[str, object]] | None = None
    _started: dict[int, float] | None = None
    post_parameter_maps: dict[int, dict[str, torch.Tensor]] | None = None
    post_optimizer_maps: dict[int, dict[str, torch.Tensor]] | None = None

    def __post_init__(self) -> None:
        self.rows = []
        self._started = {}
        self.post_parameter_maps = {}
        self.post_optimizer_maps = {}

    def bind(self, engine: TrainingEngine) -> None:
        self.engine = engine

    def on_gradient_ready(self, event: GradientReadyEvent) -> None:
        assert self._started is not None and self.rows is not None
        self._started[event.attempt_index] = time.monotonic()
        self.rows.append({
            "step": event.global_step + 1, "attempt_index": event.attempt_index,
            "boundary": "gradient_ready", "parameters_pre_sha256": _tensor_digest(event.parameters_pre),
            "mean_gradient_sha256": _tensor_digest(event.mean_gradient),
            "optimizer_gradient_sha256": _tensor_digest(event.optimizer_gradient),
            "global_gradient_norm": math.sqrt(sum(float(value.detach().to(torch.float64).square().sum().item()) for value in event.mean_gradient.values())),
            "microbatch_ids": list(event.microbatch_ids), "sample_ids": list(event.sample_ids),
            "memory": _memory("gradient_ready"),
        })

    def on_parameter_post(self, event: ParameterPostEvent) -> None:
        assert self.rows is not None and self.engine is not None and self.post_parameter_maps is not None and self.post_optimizer_maps is not None
        step = event.transaction.global_step + 1
        self.post_parameter_maps[step] = {name: value.detach().cpu().clone() for name, value in event.parameters_post.items()}
        self.post_optimizer_maps[step] = _optimizer_tensor_values(
            self.engine.model.module, self.engine.optimizer,
        )
        update_norm = math.sqrt(sum(float(value.detach().to(torch.float64).square().sum().item()) for value in event.outcome.total_delta.values()))
        self.rows.append({
            "step": step, "attempt_index": event.transaction.attempt_index,
            "boundary": "parameter_post", "parameters_post_sha256": _tensor_digest(event.parameters_post),
            "total_delta_sha256": _tensor_digest(event.outcome.total_delta),
            "data_delta_sha256": _tensor_digest(event.outcome.data_delta),
            "weight_decay_delta_sha256": _tensor_digest(event.outcome.weight_decay_delta),
            "learning_rates": dict(event.outcome.learning_rates), "optimizer_step_called": event.outcome.optimizer_step_called,
            "total_update_norm": update_norm,
            "memory": _memory("parameter_post"),
        })

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None:
        assert self.engine is not None and self.rows is not None and self._started is not None
        snapshot = None if self.engine.tracker is None else self.engine.tracker.snapshot().to_dict()
        self.rows.append({
            "step": event.transaction.global_step + 1, "attempt_index": event.transaction.attempt_index,
            "boundary": "attempt_commit", "control_state_hash": event.control_state_hash,
            "runtime_state_sha256": _state_digest(self.engine.model.module, self.engine.optimizer, self.engine.scheduler),
            "runtime_state_components": _state_components(self.engine.model.module, self.engine.optimizer, self.engine.scheduler),
            "cursor_state_sha256": canonical_json_hash(dict(event.cursor_state)),
            "importance_snapshot": snapshot,
            "attempt_seconds": time.monotonic() - self._started[event.transaction.attempt_index],
            "memory": _memory("attempt_commit"),
        })

    def on_skip(self, event: SkippedAttemptEvent) -> None:
        raise Stage1S17WorkerError(f"S17_WORKER_UNEXPECTED_SKIP:{event.transaction.attempt_index}")


def _validate_two_step_training_result(result: object) -> None:
    """Bind the S1.7 run to TrainingEngine's real terminal wire."""

    state = getattr(result, "state", None)
    records = getattr(result, "records", None)
    status = getattr(result, "status", None)
    global_step = getattr(state, "global_step", None)
    attempt_index = getattr(state, "attempt_index", None)
    skipped_steps = getattr(state, "skipped_steps", None)
    record_count = len(records) if isinstance(records, (list, tuple)) else None
    record_statuses = (
        [getattr(record, "status", None) for record in records]
        if isinstance(records, (list, tuple))
        else None
    )
    record_attempts = (
        [getattr(record, "attempt_index", None) for record in records]
        if isinstance(records, (list, tuple))
        else None
    )
    record_steps = (
        [getattr(record, "global_step", None) for record in records]
        if isinstance(records, (list, tuple))
        else None
    )
    effective_counts = (
        [getattr(record, "effective_count", None) for record in records]
        if isinstance(records, (list, tuple))
        else None
    )
    batch_ids = (
        [getattr(record, "batch_ids", None) for record in records]
        if isinstance(records, (list, tuple))
        else None
    )
    expected_batch_ids = [
        tuple(f"s17:train-{step}-{micro}" for micro in range(4))
        for step in range(2)
    ]
    if (
        status != "COMPLETE"
        or global_step != 2
        or attempt_index != 2
        or skipped_steps != 0
        or record_count != 2
        or record_statuses != ["COMMITTED", "COMMITTED"]
        or record_attempts != [1, 2]
        or record_steps != [1, 2]
        or effective_counts != [8192, 8192]
        or batch_ids != expected_batch_ids
    ):
        raise Stage1S17WorkerError(
            "S17_WORKER_TWO_STEP_TRAINING_FAILED:"
            f"status={status}:global_step={global_step}:attempt_index={attempt_index}:"
            f"skipped_steps={skipped_steps}:record_count={record_count}:"
            f"record_statuses={record_statuses}:record_attempts={record_attempts}:"
            f"record_steps={record_steps}:effective_counts={effective_counts}:"
            f"batch_ids={batch_ids}"
        )


def _training_run(adapter: TorchModelAdapter, tokens: Mapping[int, torch.Tensor], *, statistics: bool) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[int, dict[str, torch.Tensor]], dict[int, dict[str, torch.Tensor]]]:
    optimizer = torch.optim.AdamW(adapter.module.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False)
    spec = TrainingRunSpec(run_id=f"s17-{'on' if statistics else 'off'}", run_intent="formal", max_steps=2, max_attempts=2, importance_enabled=statistics, estimator_name="u", accumulation_dtype="float32", max_grad_norm=1.0, weights_exogenous=True, common_mean_assumption=True, requires_estimator_decision=False, metadata={"task_id": TASK_ID, "decision_exempt": True})
    batches = [[_batch(tokens, [index], label=f"train-{step}-{micro}") for micro, index in enumerate(group)] for step, group in enumerate(((8, 9, 10, 11), (12, 13, 14, 15)))]
    engine = TrainingEngine(spec=spec, model=adapter, optimizer=optimizer, cursor=DeterministicBatchCursor(batches), experiment_id=f"s17-{'on' if statistics else 'off'}")
    observer = _S17StepObserver(); observer.bind(engine); engine.register_observer(observer)
    result = engine.run()
    _validate_two_step_training_result(result)
    accumulator: dict[str, torch.Tensor] = {}
    accumulator_fields: list[str] = []
    if statistics:
        if engine.tracker is None or engine.tracker.accumulator.successful_steps != 2:
            raise Stage1S17WorkerError("S17_WORKER_ACCUMULATOR_STEP_DRIFT")
        state = engine.tracker.accumulator.state_dict()
        for name, value in state.items():
            if isinstance(value, Mapping):
                accumulator_fields.append(str(name))
                for parameter, tensor in value.items():
                    if isinstance(tensor, torch.Tensor): accumulator[f"{name}:{parameter}"] = tensor.detach().cpu().to(torch.float64)
    rows = list(observer.rows or [])
    commits = [item for item in rows if item["boundary"] == "attempt_commit"]
    if len(commits) != 2:
        raise Stage1S17WorkerError("S17_WORKER_STEP_OBSERVER_COMMIT_COUNT_INVALID")
    first_memory, second_memory = commits[0]["memory"], commits[1]["memory"]
    assert isinstance(first_memory, Mapping) and isinstance(second_memory, Mapping)
    allocated_before, allocated_after = int(first_memory["cuda_allocated_bytes"] or 0), int(second_memory["cuda_allocated_bytes"] or 0)
    reserved_before, reserved_after = int(first_memory["cuda_reserved_bytes"] or 0), int(second_memory["cuda_reserved_bytes"] or 0)
    allocated_growth_limit = max(16 * 1024**2, math.ceil(allocated_before * 0.02))
    bounded = allocated_after <= allocated_before + allocated_growth_limit
    if not bounded:
        raise Stage1S17WorkerError("S17_WORKER_TEMPORARY_STATE_UNBOUNDED")
    return {"final_state_sha256": _state_digest(adapter.module, optimizer), "records": [record.to_dict() for record in result.records], "observer_rows": rows, "accumulator_fields": sorted(accumulator_fields), "temporary_state_bounded": bounded, "temporary_state_bounds": {"allocated_before": allocated_before, "allocated_after": allocated_after, "allocated_growth_limit": allocated_growth_limit, "reserved_before": reserved_before, "reserved_after": reserved_after, "reserved_growth_explained_as_allocator_cache": max(0, reserved_after - reserved_before)}}, accumulator, dict(observer.post_parameter_maps or {}), dict(observer.post_optimizer_maps or {})


def _copy_model_state(source: TorchModelAdapter, target: TorchModelAdapter) -> None:
    target.module.load_state_dict(source.module.state_dict(), strict=True)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _training_parity(
    off: Mapping[str, object], on: Mapping[str, object],
    off_parameters: Mapping[int, Mapping[str, torch.Tensor]], on_parameters: Mapping[int, Mapping[str, torch.Tensor]],
    off_optimizer: Mapping[int, Mapping[str, torch.Tensor]], on_optimizer: Mapping[int, Mapping[str, torch.Tensor]],
) -> list[dict[str, object]]:
    """Compare every training boundary, retaining on-path accumulator summaries."""

    off_records, on_records = off["records"], on["records"]
    off_rows, on_rows = off["observer_rows"], on["observer_rows"]
    if not isinstance(off_records, list) or not isinstance(on_records, list) or not isinstance(off_rows, list) or not isinstance(on_rows, list) or len(off_records) != 2 or len(on_records) != 2 or len(off_rows) != len(on_rows):
        raise Stage1S17WorkerError("S17_WORKER_TRAINING_TRACE_SHAPE_INVALID")
    expected_boundaries = ["gradient_ready", "parameter_post", "attempt_commit"] * 2
    for trace, rows, statistics in ((off, off_rows, False), (on, on_rows, True)):
        if [row.get("boundary") if isinstance(row, Mapping) else None for row in rows] != expected_boundaries:
            raise Stage1S17WorkerError("S17_WORKER_OBSERVER_BOUNDARY_SEQUENCE_INVALID")
        if trace.get("accumulator_fields") != (list(ACCUMULATOR_FIELDS) if statistics else []):
            raise Stage1S17WorkerError("S17_WORKER_ACCUMULATOR_FIELD_WIRE_INVALID")
        commits = [row for row in rows if isinstance(row, Mapping) and row.get("boundary") == "attempt_commit"]
        for successful_steps, commit in enumerate(commits, start=1):
            snapshot = commit.get("importance_snapshot")
            if not statistics and snapshot is not None:
                raise Stage1S17WorkerError("S17_WORKER_OFF_STATISTICS_SNAPSHOT_PRESENT")
            if statistics and (
                not isinstance(snapshot, Mapping)
                or snapshot.get("successful_steps") != successful_steps
                or snapshot.get("skipped_steps") != 0
            ):
                raise Stage1S17WorkerError("S17_WORKER_ON_STATISTICS_SNAPSHOT_INVALID")
    for left, right in zip(off_rows, on_rows, strict=True):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or left.get("boundary") != right.get("boundary") or left.get("step") != right.get("step"):
            raise Stage1S17WorkerError("S17_WORKER_TRAINING_BOUNDARY_DRIFT")
        shared = set(left).intersection(right) - {"memory", "attempt_seconds", "importance_snapshot"}
        if any(left[key] != right[key] for key in shared):
            raise Stage1S17WorkerError(f"S17_WORKER_STATISTICS_BOUNDARY_DRIFT:{left.get('boundary')}")
    result: list[dict[str, object]] = []
    post_rows = [item for item in on_rows if isinstance(item, Mapping) and item.get("boundary") == "parameter_post"]
    commits = [item for item in on_rows if isinstance(item, Mapping) and item.get("boundary") == "attempt_commit"]
    if len(post_rows) != 2 or len(commits) != 2:
        raise Stage1S17WorkerError("S17_WORKER_TRAINING_BOUNDARY_COUNT_INVALID")
    for index, (left, right, post, commit) in enumerate(zip(off_records, on_records, post_rows, commits, strict=True), start=1):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping): raise Stage1S17WorkerError("S17_WORKER_RECORD_INVALID")
        # TrainingEngine's committed-record wire deliberately distinguishes
        # the no-statistics reference run from the U-estimator run.  Keeping
        # this explicit prevents a report schema from accidentally treating a
        # disabled estimator as if it had executed.
        if left.get("status") != "COMMITTED" or right.get("status") != "COMMITTED":
            raise Stage1S17WorkerError("S17_WORKER_RECORD_STATUS_INVALID")
        if left.get("estimator_name") is not None or right.get("estimator_name") != "u":
            raise Stage1S17WorkerError("S17_WORKER_RECORD_ESTIMATOR_WIRE_INVALID")
        for field in ("status", "mean_loss", "effective_count", "global_gradient_norm", "clip_factor", "parameter_post_state_hash", "attempt_commit_state_hash"):
            if left.get(field) != right.get(field): raise Stage1S17WorkerError(f"S17_WORKER_TRAINING_RECORD_DRIFT:{field}")
        if not all(_finite_number(right[field]) for field in ("mean_loss", "global_gradient_norm", "clip_factor")) or not _finite_number(post.get("total_update_norm")):
            raise Stage1S17WorkerError("S17_WORKER_TRAINING_NONFINITE")
        snapshot = commit.get("importance_snapshot")
        if not isinstance(snapshot, Mapping): raise Stage1S17WorkerError("S17_WORKER_IMPORTANCE_SNAPSHOT_MISSING")
        summaries = snapshot.get("scalar_summaries")
        expected_summaries = {"signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped", "data_movement", "net_data_movement", "total_endpoint_movement", "weight_decay_movement", "magnitude", "actual_update_raw_importance"}
        if not isinstance(summaries, Mapping) or not expected_summaries <= set(summaries):
            raise Stage1S17WorkerError("S17_WORKER_IMPORTANCE_SUMMARIES_INCOMPLETE")
        if not all(_finite_number(item) for item in summaries.values()): raise Stage1S17WorkerError("S17_WORKER_IMPORTANCE_SUMMARY_NONFINITE")
        parameter_error = _max_error(off_parameters[index], on_parameters[index])
        optimizer_error = _max_error(off_optimizer[index], on_optimizer[index])
        if parameter_error["within_t32"] is not True or optimizer_error["within_t32"] is not True:
            raise Stage1S17WorkerError(f"S17_WORKER_POST_STEP_TENSOR_PARITY_FAILED:{index}")
        result.append({"step": index, "loss": right["mean_loss"], "gradient_norm": right["global_gradient_norm"], "clip_factor": right["clip_factor"], "total_update_norm": post["total_update_norm"], "parameter_post_error": parameter_error, "optimizer_state_error": optimizer_error, "accumulator_summaries": {key: summaries[key] for key in sorted(expected_summaries)}, "status": "PASS"})
    return result


def execute_worker(plan_path: str | Path) -> dict[str, object]:
    plan_file = Path(plan_path).resolve(strict=True)
    plan = _validate_plan(plan_file)
    output = Path(plan["output_ref"])
    if not output.is_absolute() or output.parent != plan_file.parent or output.exists():
        raise Stage1S17WorkerError("S17_WORKER_OUTPUT_PATH_INVALID")
    _configure_determinism(_require_object(plan["fixture"], field="fixture"))
    guard_path = install_from_environment()
    started = time.monotonic()
    try:
        device = _device_probe(plan)
        fixture_path = plan_file.parent / str(plan["fixture_safetensors_ref"])
        tokens = _load_fixture(fixture_path, plan)
        torch.manual_seed(1707); random.seed(1707); np.random.seed(1707)
        torch.cuda.manual_seed_all(1707); torch.cuda.reset_peak_memory_stats(0)
        adapter, tokenizer = _load_model(plan)
        optimizer = torch.optim.AdamW(adapter.module.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False)
        registry = ParameterRegistry.from_model(adapter.module, optimizer)
        registry_reloaded = ParameterRegistry.from_manifest(registry.to_manifest(), parameters={name: registry.parameter(name) for name in registry.eligible_names})
        if (
            registry.coordinate_registry_hash != registry_reloaded.coordinate_registry_hash
            or registry.optimizer_contract_hash != registry_reloaded.optimizer_contract_hash
            or registry.runtime_layout_hash != registry_reloaded.runtime_layout_hash
            or sum(record.numel for record in registry.eligible_records) != sum(parameter.numel() for parameter in adapter.module.parameters() if parameter.requires_grad)
        ):
            raise Stage1S17WorkerError("S17_WORKER_REGISTRY_RELOAD_OR_NUMEL_FAILED")
        registry_manifest = registry.to_manifest()
        module_numel: dict[str, int] = {}; layer_numel: dict[str, int] = {}; shared_aliases: dict[str, list[str]] = {}
        for record in registry_manifest["records"]:
            if record["eligible"]:
                tags = record["tags"]; module = str(tags["module"]); layer = str(tags["layer"])
                module_numel[module] = module_numel.get(module, 0) + int(record["numel"])
                layer_numel[layer] = layer_numel.get(layer, 0) + int(record["numel"])
            if record["aliases"]: shared_aliases[str(record["canonical_name"])] = list(record["aliases"])
        registry_audit = {"eligible_numel": sum(int(record["numel"]) for record in registry_manifest["records"] if record["eligible"]), "model_trainable_numel": sum(parameter.numel() for parameter in adapter.module.parameters() if parameter.requires_grad), "coordinate_registry_hash": registry.coordinate_registry_hash, "reload_coordinate_registry_hash": registry_reloaded.coordinate_registry_hash, "module_numel": module_numel, "layer_numel": layer_numel, "shared_weight_aliases": shared_aliases, "shared_weight_alias_contract": "registry_from_model_remove_duplicate_false"}
        timeline = [_memory("after_model_registry")]
        fixed_started = time.monotonic()
        fixed, arrays = _fixed_state(adapter, registry, tokens, optimizer)
        fixed_seconds = time.monotonic() - fixed_started
        timeline.append(_memory("after_fixed_state"))
        # Fresh models ensure the statistics-on/off paths share only qualified
        # initialization and frozen inputs, not residual optimizer state.
        off, _ = _load_model(plan); on, _ = _load_model(plan); _copy_model_state(adapter, off); _copy_model_state(adapter, on)
        reloaded_optimizer = torch.optim.AdamW(on.module.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, foreach=False, fused=False)
        reloaded_registry = ParameterRegistry.from_model(on.module, reloaded_optimizer)
        if (
            registry.coordinate_registry_hash != reloaded_registry.coordinate_registry_hash
            or registry.optimizer_contract_hash != reloaded_registry.optimizer_contract_hash
            or registry.runtime_layout_hash != reloaded_registry.runtime_layout_hash
            or registry.to_manifest()["records"] != reloaded_registry.to_manifest()["records"]
        ):
            raise Stage1S17WorkerError("S17_WORKER_FRESH_RELOAD_REGISTRY_DRIFT")
        registry_audit.update({
            "fresh_reload_coordinate_registry_hash": reloaded_registry.coordinate_registry_hash,
            "fresh_reload_optimizer_contract_hash": reloaded_registry.optimizer_contract_hash,
            "fresh_reload_runtime_layout_hash": reloaded_registry.runtime_layout_hash,
        })
        training_started = time.monotonic()
        torch.manual_seed(2707); random.seed(2707); np.random.seed(2707); torch.cuda.manual_seed_all(2707)
        off_trace, _, off_parameters, off_optimizer = _training_run(off, tokens, statistics=False)
        torch.manual_seed(2707); random.seed(2707); np.random.seed(2707); torch.cuda.manual_seed_all(2707)
        on_trace, accumulator, on_parameters, on_optimizer = _training_run(on, tokens, statistics=True)
        training_seconds = time.monotonic() - training_started
        per_step_parity = _training_parity(off_trace, on_trace, off_parameters, on_parameters, off_optimizer, on_optimizer)
        if off_trace["final_state_sha256"] != on_trace["final_state_sha256"]:
            raise Stage1S17WorkerError("S17_WORKER_STATISTICS_PERTURBED_TRAINING")
        # Full post-step copies exist only long enough for the direct off/on
        # tensor comparison above; they are not part of the retained evidence.
        del off_parameters, on_parameters, off_optimizer, on_optimizer
        arrays["training_accumulator"] = accumulator
        retained_tensor_bytes = sum(value.numel() * value.element_size() for values in arrays.values() for value in values.values())
        array_path = plan_file.parent / "s1-7-arrays.safetensors"
        serialization_started = time.monotonic()
        arrays_manifest = _encode_arrays(array_path, arrays)
        serialization_seconds = time.monotonic() - serialization_started
        fixed_and_training_seconds = fixed_seconds + training_seconds
        timeline.append(_memory("after_training_and_dump"))
        report: dict[str, object] = {
            "schema_version": WORKER_REPORT_SCHEMA, "status": "PASS", "task_id": TASK_ID,
            "execution_commit": plan["execution_commit"], "run_token": plan["run_token"], "fixture": plan["fixture"],
            "device": device, "assets": plan["assets"], "registry": registry_manifest, "registry_audit": registry_audit,
            "fixed_state": fixed, "training": {"statistics_off": off_trace, "statistics_on": on_trace, "per_step_parity": per_step_parity, "bitwise_final_state_equal": True},
            "arrays": arrays_manifest, "resources": {"timeline": timeline, "fixed_and_training_seconds": fixed_and_training_seconds, "fixed_gradient_seconds": fixed_seconds, "training_seconds": training_seconds, "safetensors_serialization_seconds": serialization_seconds, "gradient_dump_bytes": array_path.stat().st_size, "retained_tensor_bytes": retained_tensor_bytes, "wall_seconds": time.monotonic() - started},
            "offline_guard_ref": None if guard_path is None else Path(guard_path).name,
            "tokenizer_class": tokenizer.__class__.__name__, "offline_provider_loads": {"model": 3, "tokenizer": 3, "all_inside_guard": guard_path is not None},
        }
        report["artifact_hash"] = canonical_json_hash(report)
        write_canonical_json(output, report)
        return report
    finally:
        finalize_offline_guard()


__all__ = [
    "ARRAY_MANIFEST_SCHEMA", "FIXTURE_ID", "Stage1S17WorkerError", "TASK_ID", "WORKER_PLAN_SCHEMA", "WORKER_REPORT_SCHEMA", "execute_worker",
]
