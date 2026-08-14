"""S1.6 CPU training-step integration evidence and offline replay.

The production side exercises the real registry, estimator staging, optimizer
bridge and long-horizon accumulator.  Its numerical reference is kept in the
separate :mod:`stage1_training_oracle` module, which is deliberately import
isolated from every production training component.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

import numpy as np
import torch

from .atomic import sha256_file
from .contracts.jsonio import canonical_json_hash
from .core.accumulator import ImportanceAccumulator
from .core.registry import ParameterRegistry
from .core.tensors import TensorMap
from .runtime.optimizer import OptimizerBridge
from .providers.training import InMemoryDatasetAdapter, TorchModelAdapter, TrainingMicrobatch
from .runtime.training import (
    AttemptCommitEvent,
    GradientReadyEvent,
    OnlineImportanceTracker,
    ParameterPostEvent,
    SkippedAttemptEvent,
    TrainingEngine,
    TrainingRunSpec,
)
from .stage1_training_oracle import FIXTURE_ID, build_stage1_s16_oracle


TASK_ID = "stage1.06_training_integration_and_accumulators"
GATE_ID = "G1-STEP"
REPORT_SCHEMA = "stage1-s1-6-step-report-v1"
ORACLE_SCHEMA = "stage1-s1-6-oracle-bundle-v1"
TRACE_SCHEMA = "stage1-s1-6-trace-bundle-v1"
TABLE_SCHEMA = "stage1-s1-6-comparison-table-v1"
GATE_SCHEMA = "stage1-s1-6-gate-record-v1"
VALIDATION_SCHEMA = "stage1-s1-6-validation-v1"
FROZEN_FIXTURE_HASH = "5bc3e3f87672a8311c98219fe666b146d3c9b9ffd13430d572b7691b679cca2e"
S1_5_FORMAL_INDEX_REF = "evidence/stage1/s1-5-formal/36a792b6a89045ae49c32225038a6c10d5082d2c/formal-20260815-s15-schema-r2/index.json"
S1_5_FORMAL_INDEX_SHA256 = "890970c5886821377ca0409ca910184da99063432d3ef35fd3e3713a5e5514c5"
S1_5_FORMAL_GATE_ARTIFACT_HASH = "4bee73ef5053d78f77d688f94fd1737cc744cbcf2bb4774f8916fc70e466887a"
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ROLE_NAMES = ("step_report", "oracle_bundle", "trace_bundle", "comparison_table", "gate_record")
_SGD_TRACE_FIELDS = (
    "mean_gradient", "raw_core", "u_core", "raw_score", "u_score", "data_delta",
    "weight_decay_delta", "total_delta", "actual_update_raw_importance", "parameters_post",
)
_SUMMARY_FIELDS = (
    "signed", "positive", "negative_mass", "absolute", "raw", "data_movement",
    "net_data_movement", "total_endpoint_movement", "weight_decay_movement",
    "raw_clipped", "total_movement", "net_weight_decay_movement",
    "actual_update_raw_importance", "magnitude", "parameters_final",
)
_ACCUMULATOR_FIELDS = (
    "signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped",
    "data_movement", "net_data_movement", "total_movement", "total_endpoint_movement",
    "weight_decay_movement", "net_weight_decay_movement", "actual_update_raw_importance",
    "magnitude", "initial_parameters", "last_parameters",
)
_ADAMW_FIELDS = (
    "parameter_pre", "gradient", "pre_exp_avg", "pre_exp_avg_sq", "pre_optimizer_step", "exp_avg", "exp_avg_sq", "optimizer_step",
    "data_delta", "weight_decay_delta", "total_delta", "parameter_post",
)
_REQUIREMENTS = (
    "fixed_loss_scale_and_unscale_boundary",
    "preclip_raw_and_u_timing",
    "nontrivial_clip_exactly_once",
    "single_optimizer_invocation",
    "long_term_atomic_accumulation",
    "signed_mass_identities",
    "multi_group_actual_learning_rates",
    "sgd_offline_replay",
    "sgd_training_engine_integration",
    "adamw_data_decay_total_decomposition",
    "actual_update_diagnostic_boundary",
    "skip_discards_staged_long_term_state",
    "statistics_do_not_perturb_training_path",
)


class Stage1TrainingIntegrationError(RuntimeError):
    """S1.6 evidence is malformed or cannot support G1-STEP."""


@contextmanager
def _s16_deterministic_runtime(seed: int):
    """Install a private CPU evidence RNG/runtime domain and restore its caller.

    ``TrainingEngine`` deliberately includes Python, NumPy, CPU Torch and CUDA
    generator state in its commit-control hash.  The S1.6 evidence builders
    are therefore not allowed to inherit any of those streams from a caller:
    two consecutive builds must yield identical *persisted control hashes*.
    The fixture executes on CPU.  CUDA RNG is captured/restored only when the
    caller already initialised CUDA, matching the generic engine control
    state without creating a device context merely for evidence serialization.

    A one-thread deterministic CPU section also makes the small attention
    fixture independent of a process's caller-selected intra-op setting.  All
    mutated global state is restored in ``finally`` so evidence generation
    cannot perturb a surrounding test or training process.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise Stage1TrainingIntegrationError("S1_6_RUNTIME_SEED_INVALID")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    # Do not probe an available accelerator from this CPU-only formal fixture:
    # that can initialise CUDA merely to serialize an otherwise unused RNG.
    # A process which already initialised CUDA is still preserved exactly.
    cuda_states = tuple(state.cpu() for state in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else ()
    thread_count = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        # ``fork_rng`` additionally protects Torch's CPU stream should a
        # future fixture return/raise across a nested generator operation.
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            if cuda_states:
                torch.cuda.manual_seed_all(seed)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True, warn_only=False)
            yield
    finally:
        # Restore policy before the caller resumes, even if construction or a
        # formal engine lifecycle raises halfway through an attempt.
        torch.use_deterministic_algorithms(deterministic, warn_only=deterministic_warn_only)
        torch.set_num_threads(thread_count)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))


def _s16_runtime_transaction(seed: int):
    """Decorate a public evidence trace builder with a private runtime domain."""

    def decorate(function: Any) -> Any:
        @wraps(function)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with _s16_deterministic_runtime(seed):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def _hash_role(value: Mapping[str, Any], *, field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return canonical_json_hash(body)


def _fixture_hash(value: object) -> str:
    """Standalone fixture canonicalization; do not share oracle implementation."""

    try:
        payload = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") + b"\n"
    except (TypeError, ValueError) as error:
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_FIXTURE_CANONICAL_INVALID") from error
    return hashlib.sha256(payload).hexdigest()


def _with_hash(value: Mapping[str, Any], *, field: str = "artifact_hash") -> dict[str, Any]:
    body = dict(value)
    body[field] = canonical_json_hash(body)
    return body


def _source_hashes(root: Path) -> dict[str, str]:
    paths = (
        "src/param_importance_nlp/core/accumulator.py",
        "src/param_importance_nlp/core/baselines.py",
        "src/param_importance_nlp/core/estimators.py",
        "src/param_importance_nlp/core/losses.py",
        "src/param_importance_nlp/core/registry.py",
        "src/param_importance_nlp/core/sufficient_statistics.py",
        "src/param_importance_nlp/core/tensors.py",
        "src/param_importance_nlp/providers/training.py",
        "src/param_importance_nlp/providers/optional.py",
        "src/param_importance_nlp/contracts/immutable.py",
        "src/param_importance_nlp/runtime/gradients.py",
        "src/param_importance_nlp/runtime/checkpoint.py",
        "src/param_importance_nlp/runtime/events.py",
        "src/param_importance_nlp/runtime/optimizer.py",
        "src/param_importance_nlp/runtime/reducers.py",
        "src/param_importance_nlp/runtime/training.py",
        "src/param_importance_nlp/runtime/transactions.py",
        "src/param_importance_nlp/atomic.py",
        "src/param_importance_nlp/contracts/seed.py",
        "src/param_importance_nlp/contracts/jsonio.py",
        "src/param_importance_nlp/stage1_training_integration.py",
        "src/param_importance_nlp/stage1_training_oracle.py",
        "ops/stage1/formalize_s1_6.py",
        "fixtures/stage1/stage1-s16-training-fixture-v1.json",
        "schemas/stage1/s1-6-training-fixture-v1.json",
        "schemas/stage1/s1-6-step-report-v1.json",
        "schemas/stage1/s1-6-oracle-bundle-v1.json",
        "schemas/stage1/s1-6-trace-bundle-v1.json",
        "schemas/stage1/s1-6-comparison-table-v1.json",
        "schemas/stage1/s1-6-gate-record-v1.json",
        "schemas/stage1/s1-6-formalization-index-v1.json",
        "schemas/stage1/s1-6-validation-v1.json",
        "tests/test_stage1_s16_training_integration.py",
        "tests/test_stage1_s16_handoff_and_charts.py",
        "tests/test_runtime_training_engine.py",
        "tests/test_core_estimators_and_accumulator.py",
        "tests/test_stage79_run_ready_completion.py",
    )
    result: dict[str, str] = {}
    for relative in paths:
        path = root / relative
        if not path.is_file():
            raise Stage1TrainingIntegrationError(f"S1_6_SOURCE_MISSING:{relative}")
        result[relative] = sha256_file(path)
    return result


def _wire(value: TensorMap | Mapping[str, torch.Tensor]) -> dict[str, list[float]]:
    items = value.items()
    return {
        name: [float(item) for item in tensor.detach().to(dtype=torch.float64, device="cpu").reshape(-1).tolist()]
        for name, tensor in items
    }


def _flat_scalar(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Stage1TrainingIntegrationError(f"S1_6_NUMBER_INVALID:{field}")
    result = float(value)
    if not math.isfinite(result):
        raise Stage1TrainingIntegrationError(f"S1_6_NUMBER_NONFINITE:{field}")
    return result


def _nested_close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    """Finite structural equality for persisted FP64 wire values."""

    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_close(left[key], right[key], tolerance=tolerance) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nested_close(a, b, tolerance=tolerance) for a, b in zip(left, right, strict=True))
    return left == right


def _accumulator_wire(accumulator: ImportanceAccumulator) -> dict[str, dict[str, list[float]]]:
    state = accumulator.state_dict()
    return {
        field: _wire(getattr(accumulator, field) if hasattr(accumulator, field) else state[field])
        for field in _ACCUMULATOR_FIELDS
    }


def _wire_difference(current: Mapping[str, Mapping[str, list[float]]], previous: Mapping[str, Mapping[str, list[float]]]) -> dict[str, dict[str, list[float]]]:
    return {
        field: {
            name: [left - right for left, right in zip(current[field][name], previous[field][name], strict=True)]
            for name in current[field]
        }
        for field in _ACCUMULATOR_FIELDS
    }


class _TwoParameterFixture(torch.nn.Module):
    def __init__(self, values: Mapping[str, list[float]]) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(values["left"], dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.tensor(values["right"], dtype=torch.float64))


class _AdamWFixture(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([value], dtype=torch.float64))


class _S16PairGradientInjection(torch.autograd.Function):
    """Expose fixture gradients only through a real autograd/TrainingEngine path."""

    @staticmethod
    def forward(  # type: ignore[no-untyped-def]
        context: Any, left: torch.Tensor, right: torch.Tensor, left_gradient: torch.Tensor, right_gradient: torch.Tensor
    ) -> torch.Tensor:
        context.left_shape, context.right_shape = left.shape, right.shape
        context.left_gradient, context.right_gradient = left_gradient.detach(), right_gradient.detach()
        return left.new_zeros(())

    @staticmethod
    def backward(context: Any, incoming: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, None, None]:  # type: ignore[no-untyped-def]
        # CE at the frozen zero logits contributes -1/2 to this scalar for a
        # two-sample mean loss.  Preserve the caller's GradScaler multiplier
        # so TrainingEngine's one formal unscale returns the frozen gradient.
        multiplier = -2.0 * incoming
        return (
            context.left_gradient.reshape(context.left_shape) * multiplier,
            context.right_gradient.reshape(context.right_shape) * multiplier,
            None,
            None,
        )


class _S16EngineSGDFixture(torch.nn.Module):
    """Two-coordinate classifier whose frozen input supplies each micro gradient."""

    def __init__(self, values: Mapping[str, list[float]]) -> None:
        super().__init__()
        self.left = torch.nn.Parameter(torch.tensor(values["left"], dtype=torch.float64))
        self.right = torch.nn.Parameter(torch.tensor(values["right"], dtype=torch.float64))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        injected = _S16PairGradientInjection.apply(self.left, self.right, input_ids[0, :1], input_ids[0, 1:2])
        return torch.stack((injected.expand(input_ids.shape[0]), torch.zeros_like(injected).expand(input_ids.shape[0])), dim=1)


class _S16FrozenGroupSchedule:
    """The fixture's only scheduler transition: step 1 LR -> frozen step 2 LR."""

    def __init__(self, optimizer: torch.optim.Optimizer, second: Mapping[str, Any]) -> None:
        self.optimizer = optimizer
        self.second = (float(second["group_0000"]), float(second["group_0001"]))
        self.calls = 0

    def step(self) -> None:
        self.calls += 1
        if self.calls == 1:
            self.optimizer.param_groups[0]["lr"], self.optimizer.param_groups[1]["lr"] = self.second

    def state_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "second": list(self.second)}


def _s16_cpu_scaler(contract: Mapping[str, Any]) -> torch.amp.GradScaler:
    """Create the real CPU GradScaler frozen by the fixture, never a shim."""

    required = {"device", "enabled", "init_scale", "growth_factor", "backoff_factor", "growth_interval", "single_process_found_inf", "tiny_scale_before", "tiny_scale_after", "skip_scale_before", "skip_scale_after"}
    if set(contract) != required or contract.get("device") != "cpu" or contract.get("enabled") is not True or contract.get("single_process_found_inf") != "local_is_global":
        raise Stage1TrainingIntegrationError("S1_6_CPU_SCALER_CONTRACT_INVALID")
    return torch.amp.GradScaler(
        "cpu", enabled=True, init_scale=float(contract["init_scale"]),
        growth_factor=float(contract["growth_factor"]), backoff_factor=float(contract["backoff_factor"]),
        growth_interval=int(contract["growth_interval"]),
    )


class S16TinyTransformer(torch.nn.Module):
    """A deliberately small, dropout-free causal Transformer for S1.6 only.

    This is not the S1.3 analytic fixture: it has a genuine self-attention
    block, residual/normalization and MLP path so instrumentation is exercised
    through the same multi-parameter TrainingEngine lifecycle used by a model.
    """

    def __init__(self, *, vocab_size: int = 17, hidden_size: int = 8, max_length: int = 6) -> None:
        super().__init__()
        self.token_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = torch.nn.Embedding(max_length, hidden_size)
        self.attention = torch.nn.MultiheadAttention(
            hidden_size, num_heads=2, dropout=0.0, batch_first=True
        )
        self.norm_1 = torch.nn.LayerNorm(hidden_size)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size * 2), torch.nn.GELU(),
            torch.nn.Linear(hidden_size * 2, hidden_size),
        )
        self.norm_2 = torch.nn.LayerNorm(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        sequence = input_ids.shape[1]
        positions = torch.arange(sequence, device=input_ids.device).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        causal = torch.triu(
            torch.ones((sequence, sequence), dtype=torch.bool, device=input_ids.device), diagonal=1
        )
        attended, _ = self.attention(hidden, hidden, hidden, attn_mask=causal, need_weights=False)
        hidden = self.norm_1(hidden + attended)
        hidden = self.norm_2(hidden + self.mlp(hidden))
        return self.lm_head(hidden)


def _s16_transformer_fixture(*, seed: int, steps: int, contract: Mapping[str, Any] | None = None) -> tuple[TorchModelAdapter, InMemoryDatasetAdapter]:
    """Build an explicit arithmetic S1.6 fixture without RNG initialisation."""

    architecture = None if contract is None else contract.get("architecture")
    if architecture is not None and (
        not isinstance(architecture, Mapping)
        or set(architecture) != {"vocab_size", "hidden_size", "num_heads", "max_length", "mlp_multiplier", "dropout"}
        or architecture != {"vocab_size": 17, "hidden_size": 8, "num_heads": 2, "max_length": 6, "mlp_multiplier": 2, "dropout": 0.0}
    ):
        raise Stage1TrainingIntegrationError("S1_6_TINY_ARCHITECTURE_CONTRACT_INVALID")
    # PyTorch constructors initialise parameters, but their transient RNG use
    # must not become part of this frozen fixture.  The explicit arithmetic
    # rule below overwrites every tensor and `fork_rng` restores the caller.
    with torch.random.fork_rng(devices=[]):
        module = S16TinyTransformer().to(dtype=torch.float32)
    initialization = None if contract is None else contract.get("initialization")
    if initialization is not None and initialization != {"rule": "arithmetic-v1", "scale": 0.0005, "offset_modulus": 23, "bias": -0.004}:
        raise Stage1TrainingIntegrationError("S1_6_TINY_INITIALIZATION_CONTRACT_INVALID")
    with torch.no_grad():
        offset = 0
        for parameter in module.parameters():
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
            parameter.copy_((values + float(offset % 23)) * 0.0005 - 0.004)
            offset += parameter.numel()
    plan: list[tuple[TrainingMicrobatch, ...]] = []
    frozen_micros = None if contract is None else contract.get("microbatches")
    if frozen_micros is not None:
        if not isinstance(frozen_micros, list) or len(frozen_micros) != steps:
            raise Stage1TrainingIntegrationError("S1_6_TINY_MICRO_SPLIT_INVALID")
        for step, raw_micros in enumerate(frozen_micros):
            if not isinstance(raw_micros, list) or len(raw_micros) != 2:
                raise Stage1TrainingIntegrationError("S1_6_TINY_MICRO_SPLIT_INVALID")
            micros: list[TrainingMicrobatch] = []
            for micro, raw in enumerate(raw_micros):
                if not isinstance(raw, Mapping) or set(raw) != {"batch_id", "input_ids", "labels", "sample_ids"}:
                    raise Stage1TrainingIntegrationError("S1_6_TINY_MICRO_FIELDS_INVALID")
                input_ids, labels, sample_ids = raw["input_ids"], raw["labels"], raw["sample_ids"]
                if raw["batch_id"] != f"s16-transformer-step-{step}-micro-{micro}" or not isinstance(input_ids, list) or not isinstance(labels, list) or input_ids != labels or len(input_ids) != 12 or not isinstance(sample_ids, list) or len(sample_ids) != 2:
                    raise Stage1TrainingIntegrationError("S1_6_TINY_MICRO_CONTRACT_INVALID")
                tensor = torch.tensor(input_ids, dtype=torch.long).reshape(2, 6)
                micros.append(TrainingMicrobatch(str(raw["batch_id"]), {"input_ids": tensor, "attention_mask": torch.ones_like(tensor), "labels": torch.tensor(labels, dtype=torch.long).reshape(2, 6)}, tuple(str(item) for item in sample_ids), {"fixture_id": FIXTURE_ID, "profile": "T32_SINGLE"}))
            plan.append(tuple(micros))
    else:
        for step in range(steps):
            micros = []
            for micro in range(2):
                input_ids = (torch.arange(12, dtype=torch.long).reshape(2, 6) + seed + step * 7 + micro * 13).remainder(17)
                micros.append(TrainingMicrobatch(f"s16-transformer-step-{step}-micro-{micro}", {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}, tuple(f"s16-transformer-{step}-{micro}-{index}" for index in range(2)), {"fixture_id": FIXTURE_ID, "profile": "T32_SINGLE"}))
            plan.append(tuple(micros))
    return TorchModelAdapter(module, task_type="causal_lm"), InMemoryDatasetAdapter(
        "s16-tiny-transformer-v1", tuple(plan)
    )


def _wire_state_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype), "shape": list(value.shape),
            "values": value.detach().to(dtype=torch.float64, device="cpu").reshape(-1).tolist(),
        }
    if isinstance(value, Mapping):
        return {str(key): _wire_state_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire_state_tree(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise Stage1TrainingIntegrationError(f"S1_6_STATE_TREE_UNSUPPORTED:{type(value).__name__}")


class _EngineTraceObserver:
    """Read-only boundary capture used to prove ordering without changing training."""

    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: object | None, *, scaler: object | None = None, capture_accumulator: bool = False) -> None:
        self.events: list[dict[str, Any]] = []
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scaler = scaler
        self._capture_accumulator = capture_accumulator
        self._accumulator: ImportanceAccumulator | None = None

    def bind_accumulator(self, accumulator: ImportanceAccumulator) -> None:
        if not self._capture_accumulator:
            raise RuntimeError("S1_6_OBSERVER_ACCUMULATOR_CAPTURE_DISABLED")
        self._accumulator = accumulator

    def on_gradient_ready(self, event: GradientReadyEvent) -> None:
        self.events.append({
            "boundary": "gradient_ready", "attempt_index": event.attempt_index,
            "global_step": event.global_step, "parameters_pre": _wire(event.parameters_pre),
            "mean_gradient": _wire(event.mean_gradient), "optimizer_gradient": _wire(event.optimizer_gradient),
            "loss_scale": event.loss_scale,
            "microbatch_ids": list(event.microbatch_ids),
            "sample_ids": list(event.sample_ids),
            "rng_before_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        })

    def on_parameter_post(self, event: ParameterPostEvent) -> None:
        self.events.append({
            "boundary": "parameter_post", "attempt_index": event.transaction.attempt_index,
            "global_step": event.transaction.global_step, "parameters_post": _wire(event.parameters_post),
            "total_delta": _wire(event.outcome.total_delta), "data_delta": _wire(event.outcome.data_delta),
            "weight_decay_delta": _wire(event.outcome.weight_decay_delta),
            "learning_rates": dict(event.outcome.learning_rates),
            "optimizer_state": _wire_state_tree(self._optimizer.state_dict()),
            "optimizer_step_called": event.outcome.optimizer_step_called,
        })

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None:
        payload = {
            "boundary": "attempt_commit", "attempt_index": event.transaction.attempt_index,
            "global_step": event.transaction.global_step, "control_state_hash": event.control_state_hash,
            "cursor_state": dict(event.cursor_state),
            "scheduler_state": None if self._scheduler is None else _wire_state_tree(self._scheduler.state_dict()),
            "scaler_state": None if self._scaler is None else _wire_state_tree(self._scaler.state_dict()),
            "scaler_scale_after": None if self._scaler is None else float(self._scaler.get_scale()),
            "rng_after_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        }
        if self._capture_accumulator:
            if self._accumulator is None:
                raise RuntimeError("S1_6_OBSERVER_ACCUMULATOR_NOT_BOUND")
            payload["accumulator_after"] = _accumulator_wire(self._accumulator)
        self.events.append(payload)

    def on_skip(self, event: SkippedAttemptEvent) -> None:
        self.events.append({
            "boundary": "skip", "attempt_index": event.transaction.attempt_index,
            "global_step": event.transaction.global_step, "microbatch_ids": list(event.microbatch_ids),
            "sample_ids": list(event.sample_ids), "cursor_state": dict(event.cursor_state),
            "scaler_state": None if self._scaler is None else _wire_state_tree(self._scaler.state_dict()),
            "scaler_scale_after": None if self._scaler is None else float(self._scaler.get_scale()),
            "rng_after_sha256": hashlib.sha256(torch.random.get_rng_state().numpy().tobytes()).hexdigest(),
        })


class _S16FiniteThenInfClassifier(torch.nn.Module):
    """Small real-autograd model whose second deterministic attempt is nonfinite."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.5]], dtype=torch.float64))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        injected = _S16GradientInjection.apply(
            self.weight, bool(torch.isinf(input_ids).any().item())
        )
        return torch.stack((injected.expand(input_ids.shape[0]), torch.zeros_like(injected).expand(input_ids.shape[0])), dim=1)


class _S16GradientInjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, parameter: torch.Tensor, inject_nonfinite: bool) -> torch.Tensor:  # type: ignore[no-untyped-def]
        ctx.inject_nonfinite = inject_nonfinite
        ctx.parameter_shape = parameter.shape
        return parameter.new_zeros(())

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:  # type: ignore[no-untyped-def]
        value = float("inf") if ctx.inject_nonfinite else 0.0
        return torch.full(ctx.parameter_shape, value, dtype=gradient.dtype, device=gradient.device), None


def _skip_microbatch(step: int, micro: int, value: float) -> TrainingMicrobatch:
    return TrainingMicrobatch(
        f"s16-skip-step-{step}-micro-{micro}",
        {"input_ids": torch.full((2, 1), value, dtype=torch.float64), "labels": torch.zeros(2, dtype=torch.long)},
        tuple(f"s16-skip-{step}-{micro}-{index}" for index in range(2)),
        {"fixture_id": FIXTURE_ID},
    )


@_s16_runtime_transaction(6107)
def build_s16_nonfinite_skip_trace(scaler_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute finite → nonfinite skip → finite through the production engine."""

    contract = scaler_contract or {"device": "cpu", "enabled": True, "init_scale": 8.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 1, "single_process_found_inf": "local_is_global", "tiny_scale_before": [8.0, 16.0], "tiny_scale_after": [16.0, 32.0], "skip_scale_before": [8.0, 16.0, 8.0], "skip_scale_after": [16.0, 8.0, 16.0]}
    previous_rng = torch.random.get_rng_state()
    torch.manual_seed(6107)
    module = _S16FiniteThenInfClassifier()
    model = TorchModelAdapter(module, task_type="sequence_classification")
    dataset = InMemoryDatasetAdapter("s16-nonfinite-skip-v1", (
        (_skip_microbatch(0, 0, 1.0), _skip_microbatch(0, 1, 2.0)),
        (_skip_microbatch(1, 0, float("inf")), _skip_microbatch(1, 1, float("inf"))),
        (_skip_microbatch(2, 0, 3.0), _skip_microbatch(2, 1, 4.0)),
    ))
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05, weight_decay=0.1, foreach=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = _s16_cpu_scaler(contract)
    observer = _EngineTraceObserver(optimizer, scheduler, scaler=scaler)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "s16-nonfinite-skip", "local_fixture", max_steps=2, max_attempts=3,
            importance_enabled=True, estimator_name="u", accumulation_dtype="float64",
            weights_exogenous=True, common_mean_assumption=True,
        ),
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, cursor=dataset.cursor(seed=7),
        observers=(observer,), capture_boundary_trace=True,
    )
    result = engine.run()
    lifecycle = [dict(item) for item in engine.boundary_trace]
    skip_events = [item for item in lifecycle if item["boundary"] == "09_skip_discard_and_scaler"]
    scale_before = [item["observation"]["loss_scale"] for item in lifecycle if item["boundary"] == "03_freeze_loss_scale"]
    scale_after = [event["scaler_scale_after"] for event in observer.events if event["boundary"] in {"attempt_commit", "skip"}]
    if scale_before != contract["skip_scale_before"] or scale_after != contract["skip_scale_after"]:
        raise Stage1TrainingIntegrationError("S1_6_SKIP_SCALER_SEQUENCE_INVALID")
    # Reference consumes precisely the two finite frozen batches, omitting the
    # bad middle batch.  Scaler values legitimately differ after backoff, but
    # all optimizer/model/scheduler/long-term score state must rejoin.
    reference_module = _S16FiniteThenInfClassifier()
    with torch.no_grad(): reference_module.weight.fill_(0.5)
    reference_model = TorchModelAdapter(reference_module, task_type="sequence_classification")
    reference_dataset = InMemoryDatasetAdapter("s16-nonfinite-reference-v1", ((
        _skip_microbatch(0, 0, 1.0), _skip_microbatch(0, 1, 2.0)), (
        _skip_microbatch(2, 0, 3.0), _skip_microbatch(2, 1, 4.0)),
    ))
    reference_optimizer = torch.optim.AdamW(reference_module.parameters(), lr=0.05, weight_decay=0.1, foreach=False)
    reference_scheduler = torch.optim.lr_scheduler.StepLR(reference_optimizer, step_size=1, gamma=0.9)
    reference_scaler = _s16_cpu_scaler(contract)
    reference_observer = _EngineTraceObserver(reference_optimizer, reference_scheduler, scaler=reference_scaler)
    reference_engine = TrainingEngine(
        spec=TrainingRunSpec("s16-nonfinite-reference", "local_fixture", max_steps=2, max_attempts=2, importance_enabled=True, estimator_name="u", accumulation_dtype="float64", weights_exogenous=True, common_mean_assumption=True),
        model=reference_model, optimizer=reference_optimizer, scheduler=reference_scheduler, scaler=reference_scaler, cursor=reference_dataset.cursor(seed=7), observers=(reference_observer,), capture_boundary_trace=True,
    )
    reference_result = reference_engine.run()
    if reference_engine.tracker is None: raise Stage1TrainingIntegrationError("S1_6_SKIP_REFERENCE_TRACKER_MISSING")
    reference_gradient_events = [event for event in reference_observer.events if event["boundary"] == "gradient_ready"]
    reference = {
        "records": [record.to_dict() for record in reference_result.records],
        "parameters": _wire({name: parameter.detach() for name, parameter in reference_module.named_parameters()}),
        "optimizer_state": _wire_state_tree(reference_optimizer.state_dict()),
        "scheduler_state": _wire_state_tree(reference_scheduler.state_dict()),
        "accumulator_state": _wire_state_tree(reference_engine.tracker.accumulator.state_dict()),
        "scaler_state": _wire_state_tree(reference_scaler.state_dict()),
        "next_batch_ids": list(reference_result.records[1].batch_ids),
        "next_sample_ids": list(reference_gradient_events[1]["sample_ids"]),
    }
    original_success = [record.to_dict() for record in result.records if record.status == "COMMITTED"]
    if [record["batch_ids"] for record in original_success] != [record["batch_ids"] for record in reference["records"]]:
        raise Stage1TrainingIntegrationError("S1_6_SKIP_REFERENCE_BATCH_ORDER_INVALID")
    original_gradient_events = [event for event in observer.events if event["boundary"] == "gradient_ready"]
    original_success_events = [original_gradient_events[0], original_gradient_events[1]]
    original_skip_event = next(event for event in observer.events if event["boundary"] == "skip")
    attempt_cursor_states = [
        event["cursor_state"] for event in observer.events if event["boundary"] in {"attempt_commit", "skip"}
    ]
    records_without_attempt = [
        {key: value for key, value in dict(record, attempt_index=index + 1).items() if key != "attempt_commit_state_hash"}
        for index, record in enumerate(original_success)
    ]
    reference_records_without_scaler_control = [
        {key: value for key, value in record.items() if key != "attempt_commit_state_hash"}
        for record in reference["records"]
    ]
    state_checks = {
        "parameters": _wire({name: parameter.detach() for name, parameter in module.named_parameters()}) == reference["parameters"],
        "optimizer": _wire_state_tree(optimizer.state_dict()) == reference["optimizer_state"],
        "scheduler": _wire_state_tree(scheduler.state_dict()) == reference["scheduler_state"],
        "accumulator": engine.tracker is not None and _wire_state_tree({key: value for key, value in engine.tracker.accumulator.state_dict().items() if key != "skipped_steps"}) == _wire_state_tree({key: value for key, value in reference_engine.tracker.accumulator.state_dict().items() if key != "skipped_steps"}),
        # A skip legitimately changes the GradScaler control digest.  The
        # committed model/optimizer/scheduler/score record is otherwise exact.
        "success_records": records_without_attempt == reference_records_without_scaler_control,
        "third_attempt_matches_reference_batch_and_samples": original_success_events[1]["microbatch_ids"] == reference["next_batch_ids"] and original_success_events[1]["sample_ids"] == reference["next_sample_ids"],
        "skip_batch_and_samples_saved": original_skip_event["microbatch_ids"] == ["s16-skip-step-1-micro-0", "s16-skip-step-1-micro-1"] and len(original_skip_event["sample_ids"]) == 4,
        "cursor_advances_each_attempt": [state.get("index") for state in attempt_cursor_states] == [1, 2, 3],
    }
    # The frozen fixtures are dropout-free and the engine does not consume RNG;
    # record that contract rather than implying a random advance occurred.
    rng_before = [
        item["observation"]["rng_before_sha256"]
        for item in lifecycle if item["boundary"] == "03_freeze_loss_scale"
    ]
    state_checks["rng_unchanged_by_computation"] = rng_before == [
        next(event["rng_after_sha256"] for event in observer.events if event["boundary"] == "attempt_commit" and event["attempt_index"] == 1),
        original_skip_event["rng_after_sha256"],
        next(event["rng_after_sha256"] for event in observer.events if event["boundary"] == "attempt_commit" and event["attempt_index"] == 3),
    ]
    if not all(state_checks.values()):
        raise Stage1TrainingIntegrationError("S1_6_SKIP_REFERENCE_STATE_DIVERGENCE:" + ",".join(key for key, value in state_checks.items() if not value))
    result_wire = {
        "records": [record.to_dict() for record in result.records],
        "lifecycle": lifecycle, "events": observer.events,
        "state": result.state.to_dict(), "optimizer_state": _wire_state_tree(optimizer.state_dict()),
        "scheduler_state": _wire_state_tree(scheduler.state_dict()),
        "scaler_state": _wire_state_tree(scaler.state_dict()), "scale_before": scale_before, "scale_after": scale_after,
        "accumulator_state": _wire_state_tree(engine.tracker.accumulator.state_dict()) if engine.tracker is not None else None,
        "skip_observation": skip_events[0]["observation"] if len(skip_events) == 1 else None,
        "reference": reference, "reference_comparison": state_checks,
    }
    torch.random.set_rng_state(previous_rng)
    return result_wire


@_s16_runtime_transaction(6106)
def build_s16_tiny_transformer_parity_trace(contract: Mapping[str, Any] | None = None, scaler_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run stats on/off through the real TrainingEngine and compare each boundary.

    CPU FP32 is intentionally registered as a T32 tolerance profile.  This
    deterministic CPU fixture is compared bitwise; the formal artifact records
    the tolerance separately so it cannot be mistaken for a GPU assertion.
    """

    previous_rng = torch.random.get_rng_state()
    if contract is not None and (contract.get("rng_seed") != 6106 or contract.get("cursor_seed") != 1 or contract.get("optimizer") != {"name": "AdamW", "learning_rate": 0.05, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8, "foreach": False} or contract.get("scheduler") != {"name": "StepLR", "step_size": 1, "gamma": 0.8}):
        raise Stage1TrainingIntegrationError("S1_6_TINY_RUNTIME_CONTRACT_INVALID")
    seed = 6106 if contract is None else int(contract["rng_seed"])
    cursor_seed = 1 if contract is None else int(contract["cursor_seed"])
    scale_contract = scaler_contract or {"device": "cpu", "enabled": True, "init_scale": 8.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 1, "single_process_found_inf": "local_is_global", "tiny_scale_before": [8.0, 16.0], "tiny_scale_after": [16.0, 32.0], "skip_scale_before": [8.0, 16.0, 8.0], "skip_scale_after": [16.0, 8.0, 16.0]}
    torch.manual_seed(seed)
    observed_model, observed_dataset = _s16_transformer_fixture(seed=seed, steps=2, contract=contract)
    control_model, control_dataset = _s16_transformer_fixture(seed=seed, steps=2, contract=contract)
    observed_initial = _wire({name: parameter.detach() for name, parameter in observed_model.module.named_parameters()})
    control_initial = _wire({name: parameter.detach() for name, parameter in control_model.module.named_parameters()})
    observed_optimizer = torch.optim.AdamW(
        observed_model.module.parameters(), lr=0.05, weight_decay=0.01, foreach=False
    )
    control_optimizer = torch.optim.AdamW(
        control_model.module.parameters(), lr=0.05, weight_decay=0.01, foreach=False
    )
    observed_scheduler = torch.optim.lr_scheduler.StepLR(observed_optimizer, step_size=1, gamma=0.8)
    control_scheduler = torch.optim.lr_scheduler.StepLR(control_optimizer, step_size=1, gamma=0.8)
    observed_scaler, control_scaler = _s16_cpu_scaler(scale_contract), _s16_cpu_scaler(scale_contract)
    common = dict(
        run_intent="local_fixture", max_steps=2, max_attempts=2, estimator_name="u",
        weights_exogenous=True, common_mean_assumption=True, accumulation_dtype="float32",
    )
    observer = _EngineTraceObserver(observed_optimizer, observed_scheduler, scaler=observed_scaler)
    # Constructors may advance the global generator even though their values
    # are overwritten by the arithmetic initializer; freeze the run stream at
    # the actual lifecycle boundary, not merely before construction.
    torch.manual_seed(seed)
    observed_engine = TrainingEngine(
        spec=TrainingRunSpec("s16-transformer-observed", importance_enabled=True, **common),
        model=observed_model, optimizer=observed_optimizer, scheduler=observed_scheduler, scaler=observed_scaler,
        cursor=observed_dataset.cursor(seed=cursor_seed), observers=(observer,), capture_boundary_trace=True,
    )
    observed = observed_engine.run()
    observed_boundaries = [dict(item) for item in observed_engine.boundary_trace]
    # Reset precisely before the control run: the persisted RNG trace is part
    # of its training-control parity, while statistics remain a separate state.
    torch.manual_seed(seed)
    control_observer = _EngineTraceObserver(control_optimizer, control_scheduler, scaler=control_scaler)
    control_engine = TrainingEngine(
        spec=TrainingRunSpec("s16-transformer-control", importance_enabled=False, **common),
        model=control_model, optimizer=control_optimizer, scheduler=control_scheduler, scaler=control_scaler,
        cursor=control_dataset.cursor(seed=cursor_seed), observers=(control_observer,), capture_boundary_trace=True,
    )
    control = control_engine.run()
    control_boundaries = [dict(item) for item in control_engine.boundary_trace]
    observed_before = [item["observation"]["loss_scale"] for item in observed_boundaries if item["boundary"] == "03_freeze_loss_scale"]
    control_before = [item["observation"]["loss_scale"] for item in control_boundaries if item["boundary"] == "03_freeze_loss_scale"]
    observed_after = [item["scaler_scale_after"] for item in observer.events if item["boundary"] == "attempt_commit"]
    control_after = [item["scaler_scale_after"] for item in control_observer.events if item["boundary"] == "attempt_commit"]
    if observed_before != scale_contract["tiny_scale_before"] or control_before != scale_contract["tiny_scale_before"] or observed_after != scale_contract["tiny_scale_after"] or control_after != scale_contract["tiny_scale_after"]:
        raise Stage1TrainingIntegrationError("S1_6_TINY_SCALER_SEQUENCE_INVALID")
    observed_params = _wire({name: parameter.detach() for name, parameter in observed_model.module.named_parameters()})
    control_params = _wire({name: parameter.detach() for name, parameter in control_model.module.named_parameters()})
    result_wire = {
        "profile": "T32_SINGLE", "cpu_bitwise_tolerance": 0.0,
        "initial_state_hash": canonical_json_hash(observed_initial),
        "control_initial_state_hash": canonical_json_hash(control_initial),
        "observed_records": [record.to_dict() for record in observed.records],
        "control_records": [record.to_dict() for record in control.records],
        "observed_events": observer.events, "control_events": control_observer.events,
        "observed_lifecycle": observed_boundaries, "control_lifecycle": control_boundaries,
        "observed_parameters": observed_params, "control_parameters": control_params,
        "observed_optimizer_state": _wire_state_tree(observed_optimizer.state_dict()),
        "control_optimizer_state": _wire_state_tree(control_optimizer.state_dict()),
        "observed_scheduler_state": _wire_state_tree(observed_scheduler.state_dict()),
        "control_scheduler_state": _wire_state_tree(control_scheduler.state_dict()),
        "observed_scaler_state": _wire_state_tree(observed_scaler.state_dict()),
        "control_scaler_state": _wire_state_tree(control_scaler.state_dict()),
        "observed_scale_before": observed_before, "control_scale_before": control_before,
        "observed_scale_after": observed_after, "control_scale_after": control_after,
    }
    torch.random.set_rng_state(previous_rng)
    return result_wire




def _map_from_oracle(
    value: Mapping[str, list[float]], registry: ParameterRegistry
) -> TensorMap:
    return TensorMap(
        {name: torch.tensor(items, dtype=torch.float64) for name, items in value.items()},
        registry=registry,
    )


def _load_production_fixture(root: Path) -> Mapping[str, Any]:
    path = root / "fixtures/stage1/stage1-s16-training-fixture-v1.json"
    try:
        fixture = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_FIXTURE_UNREADABLE") from error
    if not isinstance(fixture, Mapping) or fixture.get("fixture_id") != FIXTURE_ID:
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_FIXTURE_INVALID")
    body = dict(fixture)
    supplied_hash = body.pop("fixture_hash", None)
    if supplied_hash != FROZEN_FIXTURE_HASH or supplied_hash != _fixture_hash(body):
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_FIXTURE_HASH_INVALID")
    return fixture


def _sgd_engine_microbatch(step: int, micro: int, gradient: Mapping[str, Any]) -> TrainingMicrobatch:
    left, right = gradient.get("left"), gradient.get("right")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != 1 or len(right) != 1:
        raise Stage1TrainingIntegrationError("S1_6_ENGINE_SGD_MICRO_GRADIENT_INVALID")
    values = torch.tensor([[float(left[0]), float(right[0])], [float(left[0]), float(right[0])]], dtype=torch.float64)
    return TrainingMicrobatch(
        f"s16-sgd-engine-step-{step}-micro-{micro}",
        {"input_ids": values, "labels": torch.zeros(2, dtype=torch.long)},
        tuple(f"s16-sgd-engine-{step}-{micro}-{sample}" for sample in range(2)),
        {"fixture_id": FIXTURE_ID, "profile": "T64_ENGINE"},
    )


@_s16_runtime_transaction(6108)
def build_s16_sgd_training_engine_trace(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Run the frozen two-group SGD fixture through the production engine.

    The analytical score projection remains useful for the independent oracle
    table, but this trace is deliberately non-bypassable integration evidence:
    frozen micro gradients enter autograd, then the exact TrainingEngine owns
    unscale/finite/stage/optimizer/accumulator/scheduler/publication.
    """

    sgd = fixture.get("sgd")
    if not isinstance(sgd, Mapping) or not isinstance(sgd.get("initial_parameters"), Mapping) or not isinstance(sgd.get("steps"), list) or len(sgd["steps"]) != 2:
        raise Stage1TrainingIntegrationError("S1_6_ENGINE_SGD_FIXTURE_INVALID")
    steps = sgd["steps"]
    if not all(isinstance(step, Mapping) and isinstance(step.get("learning_rates"), Mapping) and isinstance(step.get("microbatch_gradients"), list) and len(step["microbatch_gradients"]) == 2 for step in steps):
        raise Stage1TrainingIntegrationError("S1_6_ENGINE_SGD_STEP_INVALID")
    initial = sgd["initial_parameters"]
    first_rates = steps[0]["learning_rates"]
    second_rates = steps[1]["learning_rates"]
    module = _S16EngineSGDFixture(initial)  # type: ignore[arg-type]
    model = TorchModelAdapter(module, task_type="sequence_classification")
    dataset = InMemoryDatasetAdapter(
        "s16-engine-sgd-v1",
        tuple(
            tuple(_sgd_engine_microbatch(step_index, micro_index, gradient) for micro_index, gradient in enumerate(raw_step["microbatch_gradients"]))
            for step_index, raw_step in enumerate(steps)
        ),
    )
    optimizer = torch.optim.SGD(
        [{"params": [module.left], "lr": float(first_rates["group_0000"]), "foreach": False}, {"params": [module.right], "lr": float(first_rates["group_0001"]), "foreach": False}],
        momentum=0.0, foreach=False,
    )
    scheduler = _S16FrozenGroupSchedule(optimizer, second_rates)
    scaler = _s16_cpu_scaler({"device": "cpu", "enabled": True, "init_scale": 8.0, "growth_factor": 2.0, "backoff_factor": 0.5, "growth_interval": 1, "single_process_found_inf": "local_is_global", "tiny_scale_before": [8.0, 16.0], "tiny_scale_after": [16.0, 32.0], "skip_scale_before": [8.0, 16.0, 8.0], "skip_scale_after": [16.0, 8.0, 16.0]})
    observer = _EngineTraceObserver(optimizer, scheduler, scaler=scaler, capture_accumulator=True)
    engine = TrainingEngine(
        spec=TrainingRunSpec(
            "s16-engine-sgd", "local_fixture", max_steps=2, max_attempts=2,
            importance_enabled=True, estimator_name="u", accumulation_dtype="float64",
            weights_exogenous=True, common_mean_assumption=True,
        ),
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler, cursor=dataset.cursor(seed=16),
        observers=(observer,), capture_boundary_trace=True,
    )
    if engine.tracker is None: raise Stage1TrainingIntegrationError("S1_6_ENGINE_SGD_TRACKER_MISSING")
    observer.bind_accumulator(engine.tracker.accumulator)
    result = engine.run()
    gradients = [event for event in observer.events if event["boundary"] == "gradient_ready"]
    posts = [event for event in observer.events if event["boundary"] == "parameter_post"]
    commits = [event for event in observer.events if event["boundary"] == "attempt_commit"]
    if len(gradients) != 2 or len(posts) != 2 or len(commits) != 2 or result.state.to_dict() != {"global_step": 2, "attempt_index": 2, "skipped_steps": 0, "event_sequence": 0, "last_checkpoint_id": None}:
        raise Stage1TrainingIntegrationError("S1_6_ENGINE_SGD_EXECUTION_NOT_COMPLETE")
    return {
        "records": [record.to_dict() for record in result.records],
        "gradient_events": gradients, "parameter_post_events": posts, "commit_events": commits,
        "lifecycle": [dict(item) for item in engine.boundary_trace],
        "optimizer_state": _wire_state_tree(optimizer.state_dict()),
        "scheduler_state": _wire_state_tree(scheduler.state_dict()),
        "accumulator_state": _wire_state_tree(engine.tracker.accumulator.state_dict()),
        "parameters_final": _wire({name: parameter.detach() for name, parameter in module.named_parameters()}),
    }


@_s16_runtime_transaction(6109)
def build_s16_clip_training_engine_trace(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise a nontrivial norm clip through the real TrainingEngine."""

    sgd, clip = fixture.get("sgd"), fixture.get("clip")
    if not isinstance(sgd, Mapping) or not isinstance(clip, Mapping):
        raise Stage1TrainingIntegrationError("S1_6_CLIP_FIXTURE_INVALID")
    initial, mean = sgd.get("initial_parameters"), clip.get("mean_gradient")
    if not isinstance(initial, Mapping) or mean != {"left": [3.0], "right": [4.0]}:
        raise Stage1TrainingIntegrationError("S1_6_CLIP_MEAN_INVALID")
    module = _S16EngineSGDFixture(initial)  # type: ignore[arg-type]
    model = TorchModelAdapter(module, task_type="sequence_classification")
    pair = (_sgd_engine_microbatch(9, 0, mean), _sgd_engine_microbatch(9, 1, mean))
    dataset = InMemoryDatasetAdapter("s16-clip-engine-v1", (pair,))
    optimizer = torch.optim.SGD(
        [{"params": [module.left], "lr": 0.1, "foreach": False}, {"params": [module.right], "lr": 0.2, "foreach": False}],
        momentum=0.0, foreach=False,
    )
    observer = _EngineTraceObserver(optimizer, None, capture_accumulator=True)
    engine = TrainingEngine(
        spec=TrainingRunSpec("s16-clip-engine", "local_fixture", max_steps=1, max_attempts=1,
            importance_enabled=True, estimator_name="u", accumulation_dtype="float64", max_grad_norm=float(clip["max_grad_norm"]),
            weights_exogenous=True, common_mean_assumption=True),
        model=model, optimizer=optimizer, cursor=dataset.cursor(seed=19), observers=(observer,), capture_boundary_trace=True,
    )
    if engine.tracker is None:
        raise Stage1TrainingIntegrationError("S1_6_CLIP_TRACKER_MISSING")
    observer.bind_accumulator(engine.tracker.accumulator)
    accumulator_before = _accumulator_wire(engine.tracker.accumulator)
    result = engine.run()
    gradient = next(event for event in observer.events if event["boundary"] == "gradient_ready")
    post = next(event for event in observer.events if event["boundary"] == "parameter_post")
    commit = next(event for event in observer.events if event["boundary"] == "attempt_commit")
    expected_factor = float(clip["max_grad_norm"]) / (float(clip["global_norm"]) + float(clip["clip_epsilon"]))
    if (
        result.records[0].clip_factor is None
        or abs(result.records[0].clip_factor - expected_factor) > 1e-15
        or result.records[0].global_gradient_norm is None
        or abs(result.records[0].global_gradient_norm - float(clip["global_norm"])) > 1e-14
    ):
        raise Stage1TrainingIntegrationError("S1_6_CLIP_ENGINE_RESULT_INVALID")
    return {
        "record": result.records[0].to_dict(), "gradient_event": gradient,
        "parameter_post_event": post, "commit_event": commit,
        "lifecycle": [dict(item) for item in engine.boundary_trace],
        "accumulator_state": _accumulator_wire(engine.tracker.accumulator),
        "accumulator_interval_delta": _wire_difference(_accumulator_wire(engine.tracker.accumulator), accumulator_before),
    }


def _production_trace(fixture: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    sgd_fixture = fixture.get("sgd")
    adamw_cases = fixture.get("adamw_cases")
    if not isinstance(sgd_fixture, Mapping) or not isinstance(adamw_cases, list) or len(adamw_cases) != 2:
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_FIXTURE_STRUCTURE_INVALID")
    initial_values = sgd_fixture.get("initial_parameters")
    steps = sgd_fixture.get("steps")
    parameter_to_group = fixture.get("parameter_to_group")
    if not isinstance(initial_values, Mapping) or not isinstance(steps, list) or len(steps) != 2 or parameter_to_group != {"left": "group_0000", "right": "group_0001"}:
        raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_SGD_FIXTURE_INVALID")
    model = _TwoParameterFixture(initial_values)  # type: ignore[arg-type]
    optimizer = torch.optim.SGD(
        [
            {"params": [model.left], "lr": 0.1, "foreach": False},
            {"params": [model.right], "lr": 0.2, "foreach": False},
        ],
        momentum=0.0,
        foreach=False,
    )
    registry = ParameterRegistry.from_model(model, optimizer)
    spec = TrainingRunSpec(
        "s1-6-production-fixture", "local_fixture", 2, 2,
        importance_enabled=True, estimator_name="u", accumulation_dtype="float64",
        weights_exogenous=True, common_mean_assumption=True,
    )
    tracker = OnlineImportanceTracker(registry, spec)
    bridge = OptimizerBridge(
        {name: registry.parameter(name) for name in registry.eligible_names}, optimizer
    )
    production: list[dict[str, Any]] = []
    # Intervals are deltas from the committed pre-step v3 state, never a
    # shorthand for a full after-state on the first successful step.
    previous_accumulator_snapshot = _accumulator_wire(tracker.accumulator)
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_SGD_STEP_INVALID")
        learning_rates = raw_step.get("learning_rates")
        raw_samples = raw_step.get("microbatch_gradients")
        if not isinstance(learning_rates, Mapping) or not isinstance(raw_samples, list) or len(raw_samples) != 2:
            raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_SGD_STEP_INVALID")
        # Apply the next frozen schedule before score formation.  This verifies
        # that each public score sees the actual current parameter-group LR.
        optimizer.param_groups[0]["lr"] = float(learning_rates["group_0000"])
        optimizer.param_groups[1]["lr"] = float(learning_rates["group_0001"])
        samples = tuple(_map_from_oracle(item, registry) for item in raw_samples)
        lrs = {"group_0000": float(learning_rates["group_0000"]), "group_0001": float(learning_rates["group_0001"])}
        main, raw, raw_clipped = tracker.stage(samples, (1.0, 1.0), lrs, clip_factor=1.0)
        mean = (samples[0] + samples[1]) / 2.0
        for name in registry.eligible_names:
            registry.parameter(name).grad = mean[name].clone()
        outcome = bridge.step()
        tracker.commit(main, raw, raw_clipped, outcome, mean_gradient=mean)
        accumulator_snapshot = _accumulator_wire(tracker.accumulator)
        interval_delta = _wire_difference(accumulator_snapshot, previous_accumulator_snapshot)
        previous_accumulator_snapshot = accumulator_snapshot
        production.append({
            "step": index + 1,
            "learning_rates": dict(lrs),
            "mean_gradient": _wire(mean),
            "raw_core": _wire(raw.core),
            "u_core": _wire(main.core),
            "raw_score": _wire(raw.score),
            "u_score": _wire(main.score),
            "data_delta": _wire(outcome.data_delta),
            "weight_decay_delta": _wire(outcome.weight_decay_delta),
            "total_delta": _wire(outcome.total_delta),
            "actual_update_raw_importance": _wire(-(TensorMap(outcome.data_delta, registry=registry) * mean)),
            "parameters_post": _wire({name: registry.parameter(name).detach() for name in registry.eligible_names}),
            "accumulator_after": accumulator_snapshot,
            "accumulator_interval_delta": interval_delta,
        })
    accumulator = tracker.accumulator
    summary = {
        "signed": _wire(accumulator.signed), "positive": _wire(accumulator.positive),
        "negative_mass": _wire(accumulator.negative_mass), "absolute": _wire(accumulator.absolute),
        "raw": _wire(accumulator.raw), "raw_clipped": _wire(accumulator.raw_clipped), "data_movement": _wire(accumulator.data_movement),
        "net_data_movement": _wire(accumulator.net_data_movement),
        "total_movement": _wire(accumulator.total_movement),
        "total_endpoint_movement": _wire(accumulator.total_endpoint_movement),
        "weight_decay_movement": _wire(accumulator.weight_decay_movement),
        "net_weight_decay_movement": _wire(accumulator.net_weight_decay_movement),
        "actual_update_raw_importance": _wire(accumulator.actual_update_raw_importance),
        "magnitude": _wire(accumulator.magnitude),
        "parameters_final": _wire({name: registry.parameter(name).detach() for name in registry.eligible_names}),
        "actual_update_raw_importance_available": accumulator.actual_update_raw_importance_available,
    }
    roundtrip = ImportanceAccumulator(
        TensorMap({name: registry.parameter(name).detach() for name in registry.eligible_names}, registry=registry),
        accumulation_dtype=torch.float64,
    )
    roundtrip.load_state_dict(accumulator.state_dict())
    summary["v3_roundtrip"] = _accumulator_wire(roundtrip)
    adamw_production: list[dict[str, Any]] = []
    for adamw_fixture in adamw_cases:
        if not isinstance(adamw_fixture, Mapping):
            raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_ADAMW_FIXTURE_INVALID")
        required = {"case_id", "initial_parameter", "learning_rate", "weight_decay", "betas", "epsilon", "gradients"}
        if set(adamw_fixture) != required or not isinstance(adamw_fixture["gradients"], list):
            raise Stage1TrainingIntegrationError("S1_6_PRODUCTION_ADAMW_FIXTURE_INVALID")
        model_adamw = _AdamWFixture(float(adamw_fixture["initial_parameter"]))
        betas = adamw_fixture["betas"]
        optimizer_adamw = torch.optim.AdamW(
            model_adamw.parameters(), lr=float(adamw_fixture["learning_rate"]),
            betas=(float(betas[0]), float(betas[1])), eps=float(adamw_fixture["epsilon"]),
            weight_decay=float(adamw_fixture["weight_decay"]), foreach=False,
        )
        registry_adamw = ParameterRegistry.from_model(model_adamw, optimizer_adamw)
        tracker_adamw = OnlineImportanceTracker(
            registry_adamw,
            TrainingRunSpec(
                f"s1-6-adamw-{adamw_fixture['case_id']}", "local_fixture", 2, 2,
                importance_enabled=True, estimator_name="u", accumulation_dtype="float64",
                weights_exogenous=True, common_mean_assumption=True,
            ),
        )
        bridge_adamw = OptimizerBridge({"weight": model_adamw.weight}, optimizer_adamw)
        previous_accumulator = _accumulator_wire(tracker_adamw.accumulator)
        for step, gradient in enumerate(adamw_fixture["gradients"], start=1):
            parameter_pre = float(model_adamw.weight.item())
            pre_state = optimizer_adamw.state.get(model_adamw.weight, {})
            pre_exp_avg = float(pre_state.get("exp_avg", torch.zeros_like(model_adamw.weight)).item())
            pre_exp_avg_sq = float(pre_state.get("exp_avg_sq", torch.zeros_like(model_adamw.weight)).item())
            pre_optimizer_step = float(pre_state.get("step", 0.0))
            samples = tuple(
                TensorMap({"weight": torch.tensor([gradient], dtype=torch.float64)}, registry=registry_adamw)
                for _ in range(2)
            )
            manifest_records = registry_adamw.to_manifest()["records"]
            group_id = manifest_records[0]["group_id"]
            if not isinstance(group_id, str):
                raise Stage1TrainingIntegrationError("S1_6_ADAMW_GROUP_ID_MISSING")
            learning_rates = {group_id: float(adamw_fixture["learning_rate"])}
            main, raw, raw_clipped = tracker_adamw.stage(samples, (1.0, 1.0), learning_rates, clip_factor=1.0)
            mean_gradient = (samples[0] + samples[1]) / 2.0
            model_adamw.weight.grad = torch.tensor([gradient], dtype=torch.float64)
            outcome = bridge_adamw.step()
            tracker_adamw.commit(main, raw, raw_clipped, outcome, mean_gradient=mean_gradient)
            state = optimizer_adamw.state[model_adamw.weight]
            accumulator_after = _accumulator_wire(tracker_adamw.accumulator)
            accumulator_interval_delta = _wire_difference(accumulator_after, previous_accumulator)
            previous_accumulator = accumulator_after
            accumulator_roundtrip = ImportanceAccumulator(
                TensorMap({"weight": torch.tensor([float(adamw_fixture["initial_parameter"])], dtype=torch.float64)}, registry=registry_adamw),
                accumulation_dtype=torch.float64,
            )
            accumulator_roundtrip.load_state_dict(tracker_adamw.accumulator.state_dict())
            adamw_production.append({
                "case_id": str(adamw_fixture["case_id"]), "step": float(step),
                "parameter_pre": parameter_pre, "gradient": float(gradient),
                "pre_exp_avg": pre_exp_avg, "pre_exp_avg_sq": pre_exp_avg_sq,
                "pre_optimizer_step": pre_optimizer_step,
                "exp_avg": float(state["exp_avg"].item()), "exp_avg_sq": float(state["exp_avg_sq"].item()),
                "optimizer_step": float(state["step"].item()),
                "data_delta": float(outcome.data_delta["weight"].item()),
                "weight_decay_delta": float(outcome.weight_decay_delta["weight"].item()),
                "total_delta": float(outcome.total_delta["weight"].item()),
                "parameter_post": float(model_adamw.weight.item()),
                "accumulator_after": accumulator_after,
                "accumulator_interval_delta": accumulator_interval_delta,
                "v3_roundtrip": _accumulator_wire(accumulator_roundtrip),
            })
    return production, summary, adamw_production


def _rows(
    production: list[Mapping[str, Any]], summary: Mapping[str, Any], oracle: Mapping[str, Any], adamw: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for actual, expected in zip(production, oracle["sgd_trace"], strict=True):
        for field in _SGD_TRACE_FIELDS:
            for coordinate in ("left", "right"):
                for index, (candidate, reference) in enumerate(zip(actual[field][coordinate], expected[field][coordinate], strict=True)):
                    error = abs(float(candidate) - float(reference))
                    rows.append({"section": "sgd_step", "step": int(actual["step"]), "field": field, "coordinate": coordinate, "index": index, "actual": float(candidate), "reference": float(reference), "absolute_error": error, "passed": error <= 1e-12})
    for field in _SUMMARY_FIELDS:
        for coordinate in ("left", "right"):
            for index, (candidate, reference) in enumerate(zip(summary[field][coordinate], oracle["sgd_summary"][field][coordinate], strict=True)):
                error = abs(float(candidate) - float(reference))
                rows.append({"section": "sgd_accumulator", "step": 2, "field": field, "coordinate": coordinate, "index": index, "actual": float(candidate), "reference": float(reference), "absolute_error": error, "passed": error <= 1e-12})
    for actual, expected in zip(adamw, oracle["adamw_trace"], strict=True):
        for field in _ADAMW_FIELDS:
            error = abs(float(actual[field]) - float(expected[field]))
            rows.append({"section": "adamw", "case_id": actual["case_id"], "step": int(actual["step"]), "field": field, "coordinate": "weight", "index": 0, "actual": float(actual[field]), "reference": float(expected[field]), "absolute_error": error, "passed": error <= 1e-12})
    if len(rows) != 118:
        raise AssertionError(f"S1_6_COMPARISON_ROW_COUNT_DRIFT:{len(rows)}")
    return rows


def build_stage1_s16_evidence(
    source_root: str | Path,
    *,
    producer_commit: str,
    scope: str = "local_fixture",
    upstream_evidence: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build deterministic S1.6 role objects; formal publication is external."""

    root = Path(source_root).resolve()
    if _COMMIT.fullmatch(producer_commit) is None or scope not in {"local_fixture", "formal"}:
        raise Stage1TrainingIntegrationError("S1_6_BUILD_ID_OR_SCOPE_INVALID")
    formal_upstream_keys = {"s1_5_index_ref", "s1_5_index_sha256", "s1_5_gate_artifact_hash", "s1_5_replay_sha256"}
    if scope == "formal":
        if not isinstance(upstream_evidence, Mapping) or set(upstream_evidence) != formal_upstream_keys:
            raise Stage1TrainingIntegrationError("S1_6_FORMAL_UPSTREAM_REQUIRED")
        if (
            upstream_evidence.get("s1_5_index_ref") != S1_5_FORMAL_INDEX_REF
            or upstream_evidence.get("s1_5_index_sha256") != S1_5_FORMAL_INDEX_SHA256
            or upstream_evidence.get("s1_5_gate_artifact_hash") != S1_5_FORMAL_GATE_ARTIFACT_HASH
            or not isinstance(upstream_evidence.get("s1_5_replay_sha256"), str)
            or _DIGEST.fullmatch(str(upstream_evidence.get("s1_5_replay_sha256"))) is None
        ):
            raise Stage1TrainingIntegrationError("S1_6_FORMAL_UPSTREAM_BINDING_INVALID")
    if scope == "local_fixture" and upstream_evidence not in (None, {}):
        raise Stage1TrainingIntegrationError("S1_6_LOCAL_UPSTREAM_FORBIDDEN")
    oracle = build_stage1_s16_oracle(root)
    fixture = _load_production_fixture(root)
    production, summary, adamw = _production_trace(fixture)
    engine_sgd = build_s16_sgd_training_engine_trace(fixture)
    clip_engine = build_s16_clip_training_engine_trace(fixture)
    rows = _rows(production, summary, oracle, adamw)
    adamw_checks = [
        {
            "case_id": str(actual["case_id"]), "step": int(actual["step"]),
            "moments_and_outcome_match_oracle": _nested_close(
                {field: actual[field] for field in _ADAMW_FIELDS},
                {field: expected[field] for field in _ADAMW_FIELDS},
            ),
            "accumulator_after_match_oracle": _nested_close(actual["accumulator_after"], expected["accumulator_after"]),
            "accumulator_interval_match_oracle": _nested_close(actual["accumulator_interval_delta"], expected["accumulator_interval_delta"]),
            "v3_roundtrip_match": actual["accumulator_after"] == actual["v3_roundtrip"],
        }
        for actual, expected in zip(adamw, oracle["adamw_trace"], strict=True)
    ]
    tiny_contract = fixture.get("tiny_transformer")
    scaler_contract = fixture.get("scaler")
    if not isinstance(tiny_contract, Mapping) or not isinstance(scaler_contract, Mapping):
        raise Stage1TrainingIntegrationError("S1_6_TINY_FIXTURE_MISSING")
    tiny_transformer = build_s16_tiny_transformer_parity_trace(tiny_contract, scaler_contract)
    skip_trace = build_s16_nonfinite_skip_trace(scaler_contract)
    source_hashes = _source_hashes(root)
    oracle_bundle = _with_hash({
        "schema_version": ORACLE_SCHEMA, "fixture_id": FIXTURE_ID,
        "independent_implementation": True, "oracle": oracle,
    }, field="oracle_hash")
    trace_bundle = _with_hash({
        "schema_version": TRACE_SCHEMA, "fixture_id": FIXTURE_ID,
        "production_sgd_trace": production, "production_sgd_summary": summary,
        "production_adamw_trace": adamw, "adamw_checks": adamw_checks,
        "production_engine_sgd_trace": engine_sgd, "clip_training_engine_trace": clip_engine, "fixture_hash": fixture["fixture_hash"],
        "tiny_transformer_trace": tiny_transformer,
        "nonfinite_skip_trace": skip_trace,
    }, field="trace_hash")
    table = _with_hash({
        "schema_version": TABLE_SCHEMA, "fixture_id": FIXTURE_ID,
        "profile": "T64_ORACLE", "rows": rows,
    }, field="table_hash")
    lifecycle_labels = (
        "01_parameters_optimizer_lr_pre", "02_clear_transient_gradients", "03_freeze_loss_scale",
        "04_local_unscaled_statistics", "05_global_mean_gradient_loss", "06_install_scaled_optimizer_gradient",
        "07_formal_unscale", "08_global_finite_decision", "10_stage_preclip_scores",
        "11_install_clipped_gradient", "12_single_optimizer_or_scaler_step", "13_freeze_score_mass_payload",
        "14_parameter_post", "15_decompose_data_decay_delta", "16_commit_all_long_term_views",
        "17_scheduler_success_counter", "18_publish_step_log_state",
    )
    lifecycle = tiny_transformer["observed_lifecycle"]
    lifecycle_chunks = [lifecycle[index : index + len(lifecycle_labels)] for index in range(0, len(lifecycle), len(lifecycle_labels))]
    control_lifecycle = tiny_transformer["control_lifecycle"]
    control_chunks = [control_lifecycle[index : index + len(lifecycle_labels)] for index in range(0, len(control_lifecycle), len(lifecycle_labels))]

    def exact_execute_lifecycle(chunks: list[list[Mapping[str, Any]]]) -> bool:
        """Verify all fixed success boundaries, not merely a present subset."""

        if len(chunks) != 2 or any(len(chunk) != len(lifecycle_labels) for chunk in chunks):
            return False
        for attempt, chunk in enumerate(chunks, start=1):
            if [item.get("boundary") for item in chunk] != list(lifecycle_labels):
                return False
            if [item.get("sequence") for item in chunk] != list(range(1 + (attempt - 1) * 17, 1 + attempt * 17)):
                return False
            if any(item.get("attempt_index") != attempt for item in chunk):
                return False
            # 17/18 are post-success-control-state and post-publication;
            # every earlier boundary observes the previous successful step.
            expected_step = [attempt - 1] * 15 + [attempt, attempt]
            if [item.get("global_step") for item in chunk] != expected_step:
                return False
        return True

    skip_labels = (
        "01_parameters_optimizer_lr_pre", "02_clear_transient_gradients", "03_freeze_loss_scale",
        "04_local_unscaled_statistics", "05_global_mean_gradient_loss", "06_install_scaled_optimizer_gradient",
        "07_formal_unscale", "08_global_finite_decision", "09_skip_discard_and_scaler",
        "17_skip_control_counter", "18_publish_step_log_state",
    )

    def exact_skip_lifecycle(value: object) -> bool:
        if not isinstance(value, list) or len(value) != 45:
            return False
        chunks = [value[:17], value[17:28], value[28:]]
        expected_labels = [list(lifecycle_labels), list(skip_labels), list(lifecycle_labels)]
        expected_lengths = [17, 11, 17]
        expected_steps = [[0] * 15 + [1, 1], [1] * 11, [1] * 15 + [2, 2]]
        cursor = 1
        for attempt, (chunk, labels, length, steps) in enumerate(
            zip(chunks, expected_labels, expected_lengths, expected_steps, strict=True), start=1
        ):
            if len(chunk) != length or [item.get("boundary") for item in chunk] != labels:
                return False
            if [item.get("sequence") for item in chunk] != list(range(cursor, cursor + length)):
                return False
            if any(item.get("attempt_index") != attempt for item in chunk):
                return False
            if [item.get("global_step") for item in chunk] != steps:
                return False
            cursor += length
        return True

    def records_equal_ignoring_statistics() -> bool:
        observed_records = tiny_transformer["observed_records"]
        control_records = tiny_transformer["control_records"]
        if (
            not isinstance(observed_records, list)
            or not isinstance(control_records, list)
            or len(observed_records) != 2
            or len(control_records) != 2
        ):
            return False
        for observed, control in zip(observed_records, control_records, strict=True):
            if not isinstance(observed, Mapping) or not isinstance(control, Mapping): return False
            left, right = dict(observed), dict(control)
            left.pop("estimator_name", None); right.pop("estimator_name", None)
            if left != right: return False
        return True
    def mass_identities(views: Mapping[str, Mapping[str, list[float]]]) -> bool:
        return all(
            abs(views["signed"][name][index] - (views["positive"][name][index] - views["negative_mass"][name][index])) <= 1e-12
            and abs(views["absolute"][name][index] - (views["positive"][name][index] + views["negative_mass"][name][index])) <= 1e-12
            and views["positive"][name][index] >= 0 and views["negative_mass"][name][index] >= 0 and views["absolute"][name][index] >= 0
            for name in ("left", "right") for index in range(len(views["signed"][name]))
        )
    signed_identities = all(mass_identities(step["accumulator_after"]) for step in production) and mass_identities(summary) and production[-1]["accumulator_after"] == summary["v3_roundtrip"]
    clip_oracle = oracle["clip_oracle"]
    clip_accumulator = clip_engine["accumulator_state"]
    clip_interval = clip_engine["accumulator_interval_delta"]
    clip_factor = float(fixture["clip"]["max_grad_norm"]) / (
        float(fixture["clip"]["global_norm"]) + float(fixture["clip"]["clip_epsilon"])
    )
    clip_once = (
        _nested_close(clip_engine["gradient_event"]["mean_gradient"], clip_oracle["mean_gradient"])
        and _nested_close(clip_engine["gradient_event"]["optimizer_gradient"], clip_oracle["optimizer_gradient"])
        and _nested_close(clip_engine["parameter_post_event"]["data_delta"], clip_oracle["data_delta"])
        and abs(float(clip_engine["record"]["global_gradient_norm"]) - float(fixture["clip"]["global_norm"])) <= 1e-14
        and abs(float(clip_engine["record"]["clip_factor"]) - clip_factor) <= 1e-14
        and abs(float(clip_oracle["clip_factor"]) - clip_factor) <= 1e-15
        and _nested_close(clip_accumulator["raw"], clip_oracle["raw_unclipped"])
        and _nested_close(clip_accumulator["raw_clipped"], clip_oracle["raw_clipped"])
        and _nested_close(clip_accumulator["signed"], clip_oracle["raw_clipped"])
        and _nested_close(clip_interval["raw"], clip_oracle["raw_unclipped"])
        and _nested_close(clip_interval["raw_clipped"], clip_oracle["raw_clipped"])
        and _nested_close(clip_interval["signed"], clip_oracle["raw_clipped"])
        and _nested_close(clip_interval["positive"], clip_oracle["raw_clipped"])
        and _nested_close(clip_interval["negative_mass"], {"left": [0.0], "right": [0.0]})
        and _nested_close(clip_interval["absolute"], clip_oracle["raw_clipped"])
    )
    requirement_values = {
        "fixed_loss_scale_and_unscale_boundary": exact_execute_lifecycle(lifecycle_chunks) and exact_execute_lifecycle(control_chunks) and [event["loss_scale"] for event in tiny_transformer["observed_events"] if event["boundary"] == "gradient_ready"] == scaler_contract["tiny_scale_before"],
        "preclip_raw_and_u_timing": all(row["passed"] for row in rows if row["field"] in {"raw_core", "u_core", "raw_score", "u_score"}) and exact_execute_lifecycle(lifecycle_chunks),
        "nontrivial_clip_exactly_once": clip_once,
        "single_optimizer_invocation": exact_execute_lifecycle(lifecycle_chunks) and exact_execute_lifecycle(control_chunks),
        "long_term_atomic_accumulation": exact_execute_lifecycle(lifecycle_chunks) and exact_execute_lifecycle(control_chunks),
        "signed_mass_identities": signed_identities,
        "multi_group_actual_learning_rates": all(item["learning_rates"] == expected["learning_rates"] for item, expected in zip(production, oracle["sgd_trace"], strict=True)),
        "sgd_offline_replay": all(row["passed"] for row in rows if row["section"] == "sgd_step"),
        "sgd_training_engine_integration": [record["status"] for record in engine_sgd["records"]] == ["COMMITTED", "COMMITTED"] and len(engine_sgd["lifecycle"]) == 34 and all(
            engine_event["mean_gradient"] == analysis["mean_gradient"]
            and engine_post["data_delta"] == analysis["data_delta"]
            and engine_post["weight_decay_delta"] == analysis["weight_decay_delta"]
            and engine_post["total_delta"] == analysis["total_delta"]
            and engine_commit["accumulator_after"] == analysis["accumulator_after"]
            for engine_event, engine_post, engine_commit, analysis in zip(
                engine_sgd["gradient_events"], engine_sgd["parameter_post_events"], engine_sgd["commit_events"], production, strict=True
            )
        ),
        "adamw_data_decay_total_decomposition": (
            len(adamw_checks) == 4
            and all(all(value is True for key, value in check.items() if key not in {"case_id", "step"}) for check in adamw_checks)
            and all(abs(item["total_delta"] - item["data_delta"] - item["weight_decay_delta"]) <= 1e-12 for item in adamw)
            and (lambda rows: bool(rows) and (
                (views := rows[-1]["accumulator_after"])
                and abs(views["data_movement"]["weight"][0] - sum(abs(row["data_delta"]) for row in rows)) <= 1e-12
                and abs(views["total_movement"]["weight"][0] - sum(abs(row["total_delta"]) for row in rows)) <= 1e-12
                and abs(views["weight_decay_movement"]["weight"][0] - sum(abs(row["weight_decay_delta"]) for row in rows)) <= 1e-12
                and abs(views["net_data_movement"]["weight"][0] - abs(sum(row["data_delta"] for row in rows))) <= 1e-12
                and abs(views["net_weight_decay_movement"]["weight"][0] - abs(sum(row["weight_decay_delta"] for row in rows))) <= 1e-12
                and abs(views["total_endpoint_movement"]["weight"][0] - abs(rows[-1]["parameter_post"] - rows[0]["parameter_pre"])) <= 1e-12
                and abs(views["actual_update_raw_importance"]["weight"][0] + sum(row["data_delta"] * row["gradient"] for row in rows)) <= 1e-12
                and abs(views["net_data_movement"]["weight"][0] - views["total_endpoint_movement"]["weight"][0]) > 1e-15
            ))([item for item in adamw if item["case_id"] == "decoupled_weight_decay"])
        ),
        "actual_update_diagnostic_boundary": summary["actual_update_raw_importance_available"] is True and all(row["passed"] for row in rows if row["field"] == "actual_update_raw_importance"),
        "skip_discards_staged_long_term_state": skip_trace["skip_observation"] == {"parameters_unchanged": True, "optimizer_unchanged": True, "accumulator_long_term_unchanged": True, "scaler_present": True} and skip_trace["scale_before"] == scaler_contract["skip_scale_before"] and skip_trace["scale_after"] == scaler_contract["skip_scale_after"] and [record["status"] for record in skip_trace["records"]] == ["COMMITTED", "SKIPPED", "COMMITTED"] and exact_skip_lifecycle(skip_trace["lifecycle"]) and isinstance(skip_trace.get("reference_comparison"), Mapping) and all(skip_trace["reference_comparison"].values()),
        "statistics_do_not_perturb_training_path": records_equal_ignoring_statistics() and tiny_transformer["observed_parameters"] == tiny_transformer["control_parameters"] and tiny_transformer["observed_optimizer_state"] == tiny_transformer["control_optimizer_state"] and tiny_transformer["observed_scheduler_state"] == tiny_transformer["control_scheduler_state"] and tiny_transformer["observed_scaler_state"] == tiny_transformer["control_scaler_state"] and tiny_transformer["observed_scale_before"] == tiny_transformer["control_scale_before"] == scaler_contract["tiny_scale_before"] and tiny_transformer["observed_scale_after"] == tiny_transformer["control_scale_after"] == scaler_contract["tiny_scale_after"] and tiny_transformer["observed_events"] == tiny_transformer["control_events"],
    }
    gate = _with_hash({
        "schema_version": GATE_SCHEMA, "status": "PASS", "gate_id": GATE_ID,
        "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "requirements": requirement_values,
    })
    report = _with_hash({
        "schema_version": REPORT_SCHEMA, "status": "PASS", "gate_id": GATE_ID,
        "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "scope": scope,
        "producer_commit": producer_commit, "upstream": dict(upstream_evidence or {}),
        "source_sha256": source_hashes, "comparison_row_count": len(rows),
        "sgd_successful_steps": 2, "adamw_successful_steps": 4,
        "skip_contract": oracle["skip_contract"], "requirement_checks": requirement_values,
        "oracle_hash": oracle_bundle["oracle_hash"], "trace_hash": trace_bundle["trace_hash"],
        "table_hash": table["table_hash"], "gate_artifact_hash": gate["artifact_hash"],
    }, field="report_hash")
    evidence = {
        "step_report": report, "oracle_bundle": oracle_bundle, "trace_bundle": trace_bundle,
        "comparison_table": table, "gate_record": gate,
    }
    validate_stage1_s16_evidence(evidence, source_root=root)
    return evidence


def validate_stage1_s16_evidence(evidence: Mapping[str, Any], *, source_root: str | Path | None = None) -> dict[str, str]:
    def exact(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Stage1TrainingIntegrationError(f"S1_6_EXACT_FIELDS_INVALID:{field}")
        return value

    if set(evidence) != set(_ROLE_NAMES) or not all(isinstance(evidence[name], Mapping) for name in _ROLE_NAMES):
        raise Stage1TrainingIntegrationError("S1_6_ROLE_SET_INVALID")
    report = evidence["step_report"]
    oracle_bundle = evidence["oracle_bundle"]
    trace = evidence["trace_bundle"]
    table = evidence["comparison_table"]
    gate = evidence["gate_record"]
    exact(report, {"schema_version", "status", "gate_id", "task_id", "fixture_id", "scope", "producer_commit", "upstream", "source_sha256", "comparison_row_count", "sgd_successful_steps", "adamw_successful_steps", "skip_contract", "requirement_checks", "oracle_hash", "trace_hash", "table_hash", "gate_artifact_hash", "report_hash"}, field="step_report")
    exact(oracle_bundle, {"schema_version", "fixture_id", "independent_implementation", "oracle", "oracle_hash"}, field="oracle_bundle")
    exact(trace, {"schema_version", "fixture_id", "fixture_hash", "production_sgd_trace", "production_sgd_summary", "production_adamw_trace", "adamw_checks", "production_engine_sgd_trace", "clip_training_engine_trace", "tiny_transformer_trace", "nonfinite_skip_trace", "trace_hash"}, field="trace_bundle")
    exact(table, {"schema_version", "fixture_id", "profile", "rows", "table_hash"}, field="comparison_table")
    exact(gate, {"schema_version", "status", "gate_id", "task_id", "fixture_id", "requirements", "artifact_hash"}, field="gate_record")
    if report.get("schema_version") != REPORT_SCHEMA or report.get("status") != "PASS" or report.get("task_id") != TASK_ID or report.get("gate_id") != GATE_ID:
        raise Stage1TrainingIntegrationError("S1_6_REPORT_CONTRACT_INVALID")
    formal_keys = {"s1_5_index_ref", "s1_5_index_sha256", "s1_5_gate_artifact_hash", "s1_5_replay_sha256"}
    if (report.get("scope") == "formal" and set(report.get("upstream", {})) != formal_keys) or (report.get("scope") == "local_fixture" and report.get("upstream") != {}):
        raise Stage1TrainingIntegrationError("S1_6_REPORT_UPSTREAM_CONTRACT_INVALID")
    if report.get("scope") == "formal" and (
        report["upstream"].get("s1_5_index_ref") != S1_5_FORMAL_INDEX_REF
        or report["upstream"].get("s1_5_index_sha256") != S1_5_FORMAL_INDEX_SHA256
        or report["upstream"].get("s1_5_gate_artifact_hash") != S1_5_FORMAL_GATE_ARTIFACT_HASH
    ):
        raise Stage1TrainingIntegrationError("S1_6_REPORT_UPSTREAM_BINDING_INVALID")
    if oracle_bundle.get("schema_version") != ORACLE_SCHEMA or oracle_bundle.get("fixture_id") != FIXTURE_ID or oracle_bundle.get("independent_implementation") is not True:
        raise Stage1TrainingIntegrationError("S1_6_ORACLE_CONTRACT_INVALID")
    if trace.get("schema_version") != TRACE_SCHEMA or trace.get("fixture_id") != FIXTURE_ID:
        raise Stage1TrainingIntegrationError("S1_6_TRACE_CONTRACT_INVALID")
    if table.get("schema_version") != TABLE_SCHEMA or table.get("fixture_id") != FIXTURE_ID or table.get("profile") != "T64_ORACLE" or not isinstance(table.get("rows"), list) or len(table["rows"]) != 118 or not all(row.get("passed") is True for row in table["rows"]):
        raise Stage1TrainingIntegrationError("S1_6_TABLE_CONTRACT_INVALID")
    for index, row in enumerate(table["rows"]):
        expected_row = {"section", "step", "field", "coordinate", "index", "actual", "reference", "absolute_error", "passed"}
        if row.get("section") == "adamw": expected_row.add("case_id")
        exact(row, expected_row, field=f"comparison_table.rows[{index}]")
        for field in ("actual", "reference", "absolute_error"):
            _flat_scalar(row[field], field=f"comparison_table.rows[{index}].{field}")
        if float(row["absolute_error"]) < 0.0 or type(row["passed"]) is not bool:
            raise Stage1TrainingIntegrationError("S1_6_TABLE_VALUE_INVALID")
    if gate.get("schema_version") != GATE_SCHEMA or gate.get("status") != "PASS" or gate.get("requirements") is None or set(gate["requirements"]) != set(_REQUIREMENTS) or not all(gate["requirements"].values()):
        raise Stage1TrainingIntegrationError("S1_6_GATE_CONTRACT_INVALID:" + ",".join(key for key, value in (gate.get("requirements") or {}).items() if value is not True))
    if _hash_role(oracle_bundle, field="oracle_hash") != oracle_bundle.get("oracle_hash") or _hash_role(trace, field="trace_hash") != trace.get("trace_hash") or _hash_role(table, field="table_hash") != table.get("table_hash") or _hash_role(gate, field="artifact_hash") != gate.get("artifact_hash") or _hash_role(report, field="report_hash") != report.get("report_hash"):
        raise Stage1TrainingIntegrationError("S1_6_ROLE_HASH_INVALID")
    if report.get("oracle_hash") != oracle_bundle.get("oracle_hash") or report.get("trace_hash") != trace.get("trace_hash") or report.get("table_hash") != table.get("table_hash") or report.get("gate_artifact_hash") != gate.get("artifact_hash"):
        raise Stage1TrainingIntegrationError("S1_6_CROSS_ROLE_BINDING_INVALID")
    if source_root is not None:
        root = Path(source_root).resolve()
        if report.get("source_sha256") != _source_hashes(root):
            raise Stage1TrainingIntegrationError("S1_6_SOURCE_BINDING_INVALID")
        if trace.get("fixture_hash") != _load_production_fixture(root).get("fixture_hash"):
            raise Stage1TrainingIntegrationError("S1_6_FIXTURE_BINDING_INVALID")
    return {"report_hash": str(report["report_hash"]), "oracle_hash": str(oracle_bundle["oracle_hash"]), "trace_hash": str(trace["trace_hash"]), "table_hash": str(table["table_hash"]), "gate_artifact_hash": str(gate["artifact_hash"])}


def replay_stage1_s16_evidence(evidence: Mapping[str, Any], *, source_root: str | Path) -> dict[str, Any]:
    """Offline replay: independently rebuild oracle then compare saved roles."""

    root = Path(source_root).resolve()
    identity = validate_stage1_s16_evidence(evidence, source_root=root)
    report = evidence["step_report"]
    rebuilt = build_stage1_s16_evidence(
        root,
        producer_commit=str(report["producer_commit"]),
        scope=str(report["scope"]),
        upstream_evidence=report.get("upstream") if isinstance(report.get("upstream"), Mapping) else None,
    )
    if evidence != rebuilt:
        raise Stage1TrainingIntegrationError("S1_6_OFFLINE_REPLAY_MISMATCH")
    replay = {
        "schema_version": "stage1-s1-6-offline-replay-v1", "status": "PASS",
        "fixture_id": FIXTURE_ID, "source_report_hash": identity["report_hash"],
        "source_oracle_hash": identity["oracle_hash"], "source_trace_hash": identity["trace_hash"],
        "source_table_hash": identity["table_hash"], "source_gate_artifact_hash": identity["gate_artifact_hash"],
        "comparison_row_count": 118,
    }
    replay["replay_hash"] = canonical_json_hash(replay)
    return replay


__all__ = [
    "FIXTURE_ID", "GATE_ID", "TASK_ID", "Stage1TrainingIntegrationError",
    "build_stage1_s16_evidence", "replay_stage1_s16_evidence", "validate_stage1_s16_evidence",
]
