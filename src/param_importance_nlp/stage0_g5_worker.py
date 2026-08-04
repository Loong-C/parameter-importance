"""Fresh-process worker used by the formal Stage 0 G5 single-GPU gate.

The worker intentionally does not publish the final ``stage0.06`` task commit.
It executes one hash-bound child run, writes a canonical report, and exits.  The
parent G5 controller is the sole writer of the aggregate GateRecord.  Keeping
that boundary prevents one successful optimizer step from being mistaken for
the multi-repeat G5 decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time
from typing import Any, Mapping, Sequence

try:  # ``resource`` is unavailable on Windows development hosts.
    import resource
except ImportError:  # pragma: no cover - exercised by Windows import checks
    resource = None  # type: ignore[assignment]

import torch

from .atomic import sha256_file
from .contracts import ResolvedConfig, ResolvedConfigV2, canonical_json_hash
from .contracts.jsonio import JSONValue, load_canonical_json, write_canonical_json
from .core.errors import NumericalError
from .core.tensors import TensorMap
from .experiments.task_runners import (
    TrainingTaskRunner,
    _TrainingResources,
    _training_resources,
)
from .lifecycle import RunDirectory
from .providers import DeterministicBatchCursor, TrainingMicrobatch
from .runtime import (
    AttemptCommitEvent,
    GradientReadyEvent,
    ParameterPostEvent,
    SkippedAttemptEvent,
    TaskExecutionRequest,
    TaskRuntimeEnvironment,
    TrainingEngine,
    TrainingStepObserver,
    load_tensor_bundle,
    publish_tensor_bundle,
)
from .runtime.checkpoint import CheckpointStore
from .storage import REQUIRED_DIRECTORIES, StorageLayout


WORKER_PLAN_SCHEMA = "stage0-g5-worker-plan-v1"
WORKER_REPORT_SCHEMA = "stage0-g5-worker-report-v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_RUN_KINDS = {
    "fp32_determinism",
    "overfit",
    "bf16",
    "memory",
    "failure_invalid_asset",
    "failure_out_of_range",
    "failure_nonfinite",
    "failure_output_collision",
    "failure_checkpoint_write",
}
_SUCCESS_KINDS = {"fp32_determinism", "overfit", "bf16", "memory"}


class Stage0G5WorkerError(RuntimeError):
    """One G5 child plan or execution violated its fail-closed contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: object) -> JSONValue:
    """Convert frozen tuple metadata back into JSON-safe containers."""

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return value  # type: ignore[return-value]


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise Stage0G5WorkerError(f"G5_WORKER_OBJECT_INVALID:{field}")
    return dict(value)


def _logical_path(root: Path, value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Stage0G5WorkerError(f"G5_WORKER_LOGICAL_PATH_INVALID:{field}")
    logical = PurePosixPath(value)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage0G5WorkerError(f"G5_WORKER_LOGICAL_PATH_ESCAPE:{field}")
    resolved = root.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise Stage0G5WorkerError(f"G5_WORKER_LOGICAL_PATH_ESCAPE:{field}") from error
    return resolved


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"stage0-g5-tensor-map-v1\x00")
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_tensor_map(values: Mapping[str, torch.Tensor]) -> bool:
    return bool(values) and all(bool(torch.isfinite(value).all()) for value in values.values())


def _rss_bytes() -> int | None:
    status = Path("/proc/self/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                fields = line.split()
                if len(fields) >= 2 and fields[1].isdigit():
                    return int(fields[1]) * 1024
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if usage <= 0:
        return None
    # Linux reports KiB; macOS reports bytes.  Formal G5 is Linux-only.
    return int(usage) * 1024


def _fd_count() -> int | None:
    root = Path("/proc/self/fd")
    try:
        return sum(1 for _ in root.iterdir()) if root.is_dir() else None
    except OSError:
        return None


def _memory_snapshot(*, phase: str, step: int | None = None) -> dict[str, JSONValue]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage0G5WorkerError("G5_WORKER_SINGLE_CUDA_DEVICE_REQUIRED")
    return {
        "phase": phase,
        "step": step,
        "cuda_allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "cuda_reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "cpu_rss_bytes": _rss_bytes(),
        "open_file_descriptors": _fd_count(),
        "data_loader_workers": 0,
    }


def _device_probe(selected_uuid: str) -> dict[str, JSONValue]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != selected_uuid:
        raise Stage0G5WorkerError(
            f"G5_WORKER_CUDA_VISIBLE_DEVICES_MISMATCH:{visible!r}:{selected_uuid!r}"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise Stage0G5WorkerError("G5_WORKER_SINGLE_CUDA_DEVICE_REQUIRED")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    return {
        "selected_physical_uuid": selected_uuid,
        "cuda_visible_devices": visible,
        "logical_device": 0,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "total_memory_bytes": int(properties.total_memory),
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


class _RepeatedDataset:
    """Repeat one real, already-qualified optimizer batch for a fixed run."""

    def __init__(
        self,
        microbatches: Sequence[TrainingMicrobatch],
        *,
        steps: int,
        source: object,
    ) -> None:
        if not microbatches or steps <= 0:
            raise Stage0G5WorkerError("G5_WORKER_FIXED_BATCH_INVALID")
        self._microbatches = tuple(microbatches)
        self._steps = steps
        self._source = source
        self.dataset_id = str(getattr(source, "dataset_id", "g5-fixed-real-batch"))

    def cursor(self, *, seed: int, rank: int = 0, world_size: int = 1) -> DeterministicBatchCursor:
        del seed
        if rank != 0 or world_size != 1:
            raise Stage0G5WorkerError("G5_WORKER_FIXED_BATCH_SINGLE_RANK_ONLY")
        return DeterministicBatchCursor(
            tuple(self._microbatches for _ in range(self._steps))
        )

    def state_digest(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": "stage0-g5-fixed-real-batch-v1",
                "dataset_id": self.dataset_id,
                "steps": self._steps,
                "batch_ids": [item.batch_id for item in self._microbatches],
                "sample_ids": [
                    sample for item in self._microbatches for sample in item.sample_ids
                ],
            }
        )

    def close(self) -> None:
        close = getattr(self._source, "close", None)
        if callable(close):
            close()


class _NonfiniteModelAdapter:
    """Inject one declared non-finite loss after exercising the real adapter."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate

    @property
    def module(self) -> torch.nn.Module:
        return getattr(self._delegate, "module")

    @property
    def task_type(self) -> str:
        return str(getattr(self._delegate, "task_type"))

    def loss(self, microbatch: TrainingMicrobatch) -> object:
        # The real forward/loss path must be reached before the controlled
        # failure is raised; this is not a configuration-only negative test.
        getattr(self._delegate, "loss")(microbatch)
        raise NumericalError("STAGE0_G5_INJECTED_NONFINITE_LOSS")


class _FailingCheckpointStore(CheckpointStore):
    def publish(
        self,
        checkpoint_id: str,
        state: Any,
        *,
        generation: int,
        metadata: dict[str, Any],
        parent_checkpoint_id: str | None = None,
    ) -> object:
        del checkpoint_id, state, generation, metadata, parent_checkpoint_id
        raise OSError("STAGE0_G5_INJECTED_CHECKPOINT_WRITE_FAILURE")


@dataclass(slots=True)
class Stage0G5Observer(TrainingStepObserver):
    """Capture selected numerical tensors and per-step CUDA telemetry."""

    capture_step: int | None
    tensor_root: Path
    _engine: TrainingEngine | None = None
    _selected_names: tuple[str, ...] = ()
    _tensors: dict[str, dict[str, torch.Tensor]] | None = None
    _step_rows: list[dict[str, JSONValue]] | None = None
    _samples: dict[int, tuple[str, ...]] | None = None

    def __post_init__(self) -> None:
        self.tensor_root = self.tensor_root.resolve()
        self._tensors = {}
        self._step_rows = []
        self._samples = {}

    def bind_engine(self, engine: TrainingEngine) -> None:
        if self._engine is not None and self._engine is not engine:
            raise Stage0G5WorkerError("G5_OBSERVER_ENGINE_REBOUND")
        self._engine = engine

    def _select(self, values: Mapping[str, torch.Tensor]) -> tuple[str, ...]:
        names = tuple(sorted(values))
        if not names:
            raise Stage0G5WorkerError("G5_OBSERVER_EMPTY_TENSOR_MAP")
        if len(names) <= 6:
            return names
        by_size = sorted(names, key=lambda name: (values[name].numel(), name))
        selected = [*by_size[:4], max(names, key=lambda name: values[name].numel()), names[-1]]
        return tuple(dict.fromkeys(selected))

    def on_gradient_ready(self, event: GradientReadyEvent) -> None:
        step = event.global_step + 1
        assert self._samples is not None
        self._samples[step] = event.sample_ids
        if step != self.capture_step:
            return
        self._selected_names = self._select(event.parameters_pre)
        assert self._tensors is not None
        self._tensors["parameters_pre"] = {
            name: event.parameters_pre[name].detach().cpu().clone()
            for name in self._selected_names
        }
        self._tensors["mean_gradient"] = {
            name: event.mean_gradient[name].detach().cpu().clone()
            for name in self._selected_names
        }
        self._tensors["optimizer_gradient"] = {
            name: event.optimizer_gradient[name].detach().cpu().clone()
            for name in self._selected_names
        }
        self._tensors["full_digests"] = {
            "parameters_pre": torch.tensor(
                list(bytes.fromhex(_tensor_digest(event.parameters_pre))), dtype=torch.uint8
            ),
            "mean_gradient": torch.tensor(
                list(bytes.fromhex(_tensor_digest(event.mean_gradient))), dtype=torch.uint8
            ),
            "optimizer_gradient": torch.tensor(
                list(bytes.fromhex(_tensor_digest(event.optimizer_gradient))), dtype=torch.uint8
            ),
        }

    def on_parameter_post(self, event: ParameterPostEvent) -> None:
        step = event.transaction.global_step + 1
        if step != self.capture_step:
            return
        if not self._selected_names:
            raise Stage0G5WorkerError("G5_OBSERVER_PRE_STATE_MISSING")
        assert self._tensors is not None
        self._tensors["parameters_post"] = {
            name: event.parameters_post[name].detach().cpu().clone()
            for name in self._selected_names
        }
        self._tensors["update"] = {
            name: event.outcome.total_delta[name].detach().cpu().clone()
            for name in self._selected_names
        }
        self._tensors["full_digests"].update(
            {
                "parameters_post": torch.tensor(
                    list(bytes.fromhex(_tensor_digest(event.parameters_post))),
                    dtype=torch.uint8,
                ),
                "update": torch.tensor(
                    list(bytes.fromhex(_tensor_digest(event.outcome.total_delta))),
                    dtype=torch.uint8,
                ),
            }
        )

    def on_attempt_commit(self, event: AttemptCommitEvent) -> None:
        if self._engine is None:
            raise Stage0G5WorkerError("G5_OBSERVER_ENGINE_NOT_BOUND")
        step = event.transaction.global_step + 1
        assert self._step_rows is not None and self._samples is not None
        rates = [float(group["lr"]) for group in self._engine.optimizer.param_groups]
        memory = _memory_snapshot(phase="attempt_commit", step=step)
        self._step_rows.append(
            {
                "step": step,
                "attempt_index": event.transaction.attempt_index,
                "sample_ids": list(self._samples.get(step, ())),
                "learning_rates": rates,
                "memory": memory,
            }
        )

    def on_skip(self, event: SkippedAttemptEvent) -> None:
        assert self._step_rows is not None
        self._step_rows.append(
            {
                "step": event.transaction.global_step,
                "attempt_index": event.transaction.attempt_index,
                "sample_ids": [],
                "learning_rates": [],
                "memory": _memory_snapshot(
                    phase="skipped_attempt", step=event.transaction.global_step
                ),
            }
        )

    @property
    def step_rows(self) -> tuple[Mapping[str, JSONValue], ...]:
        assert self._step_rows is not None
        return tuple(self._step_rows)

    def publish_tensor_capture(self) -> tuple[str | None, str | None, tuple[str, ...]]:
        if self.capture_step is None:
            return None, None, ()
        assert self._tensors is not None
        expected = {
            "parameters_pre",
            "mean_gradient",
            "optimizer_gradient",
            "parameters_post",
            "update",
            "full_digests",
        }
        if set(self._tensors) != expected:
            raise Stage0G5WorkerError("G5_OBSERVER_TENSOR_CAPTURE_INCOMPLETE")
        for phase, values in self._tensors.items():
            if phase != "full_digests" and not _finite_tensor_map(values):
                raise Stage0G5WorkerError(f"G5_OBSERVER_NONFINITE_TENSOR:{phase}")
        identity = publish_tensor_bundle(
            self.tensor_root,
            {
                "schema_version": "stage0-g5-selected-step-tensors-v1",
                "capture_step": self.capture_step,
                "selected_names": list(self._selected_names),
                "tensors": self._tensors,
            },
        )
        return self.tensor_root.as_posix(), identity.manifest_sha256, self._selected_names


def _validate_plan(root: Path, value: object) -> dict[str, Any]:
    plan = _mapping(value, field="plan")
    expected = {
        "schema_version",
        "run_id",
        "run_kind",
        "repeat_index",
        "generator_git_commit",
        "config_ref",
        "config_sha256",
        "environment_ref",
        "environment_sha256",
        "selected_gpu_uuid",
        "repeat_fixed_batch",
        "capture_tensor_step",
        "memory_warmup_steps",
        "memory_measure_steps",
        "expected_failure_code",
        "artifact_hash",
    }
    if set(plan) != expected or plan.get("schema_version") != WORKER_PLAN_SCHEMA:
        raise Stage0G5WorkerError("G5_WORKER_PLAN_FIELDS_OR_VERSION_INVALID")
    declared = plan.pop("artifact_hash")
    if declared != canonical_json_hash(plan):
        raise Stage0G5WorkerError("G5_WORKER_PLAN_HASH_MISMATCH")
    plan["artifact_hash"] = declared
    if plan["run_kind"] not in _RUN_KINDS:
        raise Stage0G5WorkerError("G5_WORKER_RUN_KIND_INVALID")
    if not isinstance(plan["run_id"], str) or not plan["run_id"]:
        raise Stage0G5WorkerError("G5_WORKER_RUN_ID_INVALID")
    if (
        isinstance(plan["repeat_index"], bool)
        or not isinstance(plan["repeat_index"], int)
        or plan["repeat_index"] < 0
    ):
        raise Stage0G5WorkerError("G5_WORKER_REPEAT_INDEX_INVALID")
    if _GIT_COMMIT_RE.fullmatch(str(plan["generator_git_commit"])) is None:
        raise Stage0G5WorkerError("G5_WORKER_GIT_COMMIT_INVALID")
    for field in ("config_sha256", "environment_sha256"):
        if _SHA256_RE.fullmatch(str(plan[field])) is None:
            raise Stage0G5WorkerError(f"G5_WORKER_DIGEST_INVALID:{field}")
    uuid = plan["selected_gpu_uuid"]
    if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
        raise Stage0G5WorkerError("G5_WORKER_GPU_UUID_INVALID")
    if type(plan["repeat_fixed_batch"]) is not bool:
        raise Stage0G5WorkerError("G5_WORKER_REPEAT_FIXED_BATCH_INVALID")
    for field in ("capture_tensor_step", "memory_warmup_steps", "memory_measure_steps"):
        value = plan[field]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            raise Stage0G5WorkerError(f"G5_WORKER_INTEGER_INVALID:{field}")
    if plan["run_kind"] in _SUCCESS_KINDS and plan["expected_failure_code"] is not None:
        raise Stage0G5WorkerError("G5_WORKER_SUCCESS_KIND_HAS_FAILURE_CODE")
    if plan["run_kind"] not in _SUCCESS_KINDS and (
        not isinstance(plan["expected_failure_code"], str)
        or not plan["expected_failure_code"]
    ):
        raise Stage0G5WorkerError("G5_WORKER_FAILURE_CODE_MISSING")
    # Resolve now so a path escape fails before CUDA/model access.
    _logical_path(root, plan["config_ref"], field="config_ref")
    _logical_path(root, plan["environment_ref"], field="environment_ref")
    return plan


def _load_request(root: Path, plan: Mapping[str, Any]) -> TaskExecutionRequest:
    config_path = _logical_path(root, plan["config_ref"], field="config_ref")
    environment_path = _logical_path(root, plan["environment_ref"], field="environment_ref")
    if sha256_file(config_path) != plan["config_sha256"]:
        raise Stage0G5WorkerError("G5_WORKER_CONFIG_FILE_HASH_MISMATCH")
    if sha256_file(environment_path) != plan["environment_sha256"]:
        raise Stage0G5WorkerError("G5_WORKER_ENVIRONMENT_FILE_HASH_MISMATCH")
    config_value = _mapping(load_canonical_json(config_path), field="config")
    environment_value = _mapping(load_canonical_json(environment_path), field="environment")
    config = ResolvedConfigV2.from_mapping(config_value)
    environment = TaskRuntimeEnvironment.from_mapping(environment_value)
    if config.task_id != "stage0.06_single_gpu_smoke" or config.run_intent != "formal":
        raise Stage0G5WorkerError("G5_WORKER_CONFIG_TASK_OR_SCOPE_INVALID")
    return TaskExecutionRequest(
        config=config,
        task=config.task_definition,
        environment=environment,
    )


def _first_real_batch(
    resources: _TrainingResources,
    request: TaskExecutionRequest,
) -> tuple[tuple[TrainingMicrobatch, ...], dict[str, JSONValue]]:
    identity = request.config.base_config.section("identity")
    assert isinstance(identity, dict)
    from .contracts import SeedPlan

    seed = SeedPlan.from_master_seed(int(identity["master_seed"]), world_size=1).seed_for(
        "sampler"
    )
    cursor = resources.dataset.cursor(seed=seed, rank=0, world_size=1)
    try:
        microbatches = tuple(cursor.next_microbatches())
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()
    if not microbatches:
        raise Stage0G5WorkerError("G5_WORKER_REAL_BATCH_EMPTY")
    base = request.config.base_config
    data = base.section("data")
    model = base.section("model")
    loss = base.section("loss")
    assert isinstance(data, dict) and isinstance(model, dict) and isinstance(loss, dict)
    vocab_size = getattr(getattr(resources.model.module, "config", None), "vocab_size", None)
    rows: list[JSONValue] = []
    all_valid = True
    for item in microbatches:
        payload = item.payload
        input_ids = payload.get("input_ids")
        target_ids = payload.get("target_ids")
        attention_mask = payload.get("attention_mask")
        valid = (
            isinstance(input_ids, torch.Tensor)
            and isinstance(target_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and input_ids.ndim == target_ids.ndim == attention_mask.ndim == 2
            and tuple(input_ids.shape) == tuple(target_ids.shape) == tuple(attention_mask.shape)
            and int(input_ids.shape[1]) == int(data["sequence_length"])
            and bool(((attention_mask == 0) | (attention_mask == 1)).all())
            and int(input_ids.min().item()) >= 0
            and isinstance(vocab_size, int)
            and int(input_ids.max().item()) < vocab_size
            and int(target_ids.max().item()) < vocab_size
        )
        all_valid = all_valid and bool(valid)
        rows.append(
            {
                "batch_id": item.batch_id,
                "sample_ids": list(item.sample_ids),
                "input_shape": list(input_ids.shape) if isinstance(input_ids, torch.Tensor) else None,
                "target_shape": list(target_ids.shape) if isinstance(target_ids, torch.Tensor) else None,
                "attention_mask_shape": (
                    list(attention_mask.shape)
                    if isinstance(attention_mask, torch.Tensor)
                    else None
                ),
                "min_token_id": int(input_ids.min().item()) if isinstance(input_ids, torch.Tensor) else None,
                "max_token_id": int(input_ids.max().item()) if isinstance(input_ids, torch.Tensor) else None,
                "metadata": _json_safe(item.metadata),
                "valid": bool(valid),
            }
        )
    boundary: dict[str, JSONValue] = {
        "sequence_length": data["sequence_length"],
        "ignore_index": loss["ignore_index"],
        "model_asset_id": model["asset_id"],
        "tokenizer_asset_id": model["tokenizer_asset_id"],
        "model_vocab_size": vocab_size,
        "microbatches": rows,
        "all_valid": all_valid,
    }
    if not all_valid:
        raise Stage0G5WorkerError("G5_WORKER_DATA_BOUNDARY_INVALID")
    return microbatches, boundary


def _output_dir(root: Path, request: TaskExecutionRequest) -> Path:
    artifacts = request.config.section("artifacts")
    assert isinstance(artifacts, dict)
    return _logical_path(root, artifacts["output_dir"], field="artifacts.output_dir")


def _success_commit_absent(root: Path, request: TaskExecutionRequest) -> bool:
    output = _output_dir(root, request)
    commits = output / "commits"
    return not commits.exists() or not any(commits.glob("*.json"))


def _run_collision_check(root: Path, plan: Mapping[str, Any]) -> dict[str, JSONValue]:
    with tempfile.TemporaryDirectory(prefix="stage0-g5-collision-", dir=root / "tmp") as raw:
        owned = Path(raw)
        for name in REQUIRED_DIRECTORIES:
            (owned / name).mkdir()
        layout = StorageLayout.from_value(owned)
        run_id = f"g5-collision-{plan['repeat_index']:04d}"
        first = RunDirectory.create(layout, run_id)
        sentinel = first.path / "diagnostics" / "sentinel.txt"
        sentinel.write_text("do-not-overwrite\n", encoding="utf-8")
        before = sha256_file(sentinel)
        try:
            RunDirectory.create(layout, run_id)
        except FileExistsError as error:
            observed = type(error).__name__
        else:  # pragma: no cover - hard safety invariant
            raise Stage0G5WorkerError("G5_WORKER_OUTPUT_COLLISION_NOT_REJECTED")
        after = sha256_file(sentinel)
        if before != after:
            raise Stage0G5WorkerError("G5_WORKER_OUTPUT_COLLISION_OVERWROTE_SENTINEL")
        return {
            "exception_class": observed,
            "exception_message": "RUN_DIRECTORY_EXISTS",
            "last_valid_step": 0,
            "safe_retry": "choose a new immutable run_id/output directory",
            "sentinel_sha256": after,
            "success_commit_absent": True,
        }


def _execute_training(
    root: Path,
    plan: Mapping[str, Any],
    request: TaskExecutionRequest,
) -> dict[str, JSONValue]:
    kind = str(plan["run_kind"])
    runner = TrainingTaskRunner(root)
    before = _memory_snapshot(phase="before_model_load")
    resources = _training_resources(request, root, rank=0, world_size=1)
    fixed_batch = bool(plan["repeat_fixed_batch"])
    boundary: dict[str, JSONValue] | None = None
    if fixed_batch:
        microbatches, boundary = _first_real_batch(resources, request)
        steps = int(request.config.section("training")["max_steps"])  # type: ignore[index]
        resources = _TrainingResources(
            resources.model,
            _RepeatedDataset(microbatches, steps=steps, source=resources.dataset),
            None,
            None,
            resources.task_name,
            resources.asset_evidence,
        )
    resources.model.module.to(torch.device("cuda:0"))
    torch.cuda.synchronize(0)
    after_load = _memory_snapshot(phase="after_model_load")
    torch.cuda.reset_peak_memory_stats(0)
    observer = Stage0G5Observer(
        plan["capture_tensor_step"],
        _output_dir(root, request) / "g5-selected-tensors",
    )
    checkpoint_factory = (
        _FailingCheckpointStore
        if kind == "failure_checkpoint_write"
        else CheckpointStore
    )
    if kind == "failure_nonfinite":
        resources = _TrainingResources(
            _NonfiniteModelAdapter(resources.model),  # type: ignore[arg-type]
            resources.dataset,
            resources.evaluation_dataset,
            resources.evaluator,
            resources.task_name,
            resources.asset_evidence,
        )
    started = time.perf_counter()
    result, engine, events_path, assets, evaluations, profiles = runner._run_training(
        request,
        resources=resources,
        observers=(observer,),
        checkpoint_store_factory=checkpoint_factory,
    )
    torch.cuda.synchronize(0)
    wall = time.perf_counter() - started
    after = _memory_snapshot(phase="after_training")
    bundle_path, bundle_hash, selected_names = observer.publish_tensor_capture()
    if bundle_path is not None:
        bundle_path = Path(bundle_path).resolve().relative_to(root).as_posix()
        restored, identity = load_tensor_bundle(root / bundle_path)
        if identity.manifest_sha256 != bundle_hash or not isinstance(restored, Mapping):
            raise Stage0G5WorkerError("G5_WORKER_TENSOR_BUNDLE_RELOAD_FAILED")
    payloads = runner._result_payloads(
        request,
        result,
        engine,
        events_path,
        assets,
        evaluations,
        profiles,
    )
    common = _mapping(next(iter(payloads.values())), field="training_payload")
    records = result.records
    finite_records = all(
        record.status == "COMMITTED"
        and record.mean_loss is not None
        and math.isfinite(record.mean_loss)
        and record.global_gradient_norm is not None
        and math.isfinite(record.global_gradient_norm)
        and record.parameter_post_state_hash is not None
        for record in records
    )
    if not finite_records:
        raise Stage0G5WorkerError("G5_WORKER_TRAINING_RECORDS_NOT_FINITE_COMMITTED")
    return {
        "phase_memory": [before, after_load, after],
        "training_wall_seconds": wall,
        "training_result": common["training_result"],
        "event_stream_ref": common["event_stream_ref"],
        "event_stream_semantic_sha256": common["event_stream_semantic_sha256"],
        "checkpoint_commits": common["checkpoint_commits"],
        "asset_evidence": common["asset_evidence"],
        "resource_profiles": common["resource_profiles"],
        "selected_tensor_bundle_ref": bundle_path,
        "selected_tensor_bundle_sha256": bundle_hash,
        "selected_tensor_names": list(selected_names),
        "data_boundary": boundary,
        "step_telemetry": [dict(item) for item in observer.step_rows],
        "failure": None,
    }


def run_stage0_g5_worker(
    *,
    data_root: str | Path,
    plan_ref: str,
    report_ref: str,
) -> dict[str, JSONValue]:
    root = Path(data_root).resolve(strict=True)
    plan_path = _logical_path(root, plan_ref, field="plan_ref")
    report_path = _logical_path(root, report_ref, field="report_ref")
    plan = _validate_plan(root, load_canonical_json(plan_path))
    if report_path.exists():
        existing = _mapping(load_canonical_json(report_path), field="existing_report")
        declared = existing.get("artifact_hash")
        body = {key: value for key, value in existing.items() if key != "artifact_hash"}
        if (
            existing.get("schema_version") != WORKER_REPORT_SCHEMA
            or declared != canonical_json_hash(body)
            or existing.get("plan_sha256") != plan["artifact_hash"]
        ):
            raise Stage0G5WorkerError("G5_WORKER_EXISTING_REPORT_DRIFT")
        return existing  # type: ignore[return-value]

    started_at = _now()
    device = _device_probe(str(plan["selected_gpu_uuid"]))
    request = _load_request(root, plan)
    expected_failure = str(plan["run_kind"]) not in _SUCCESS_KINDS
    status = "FAIL"
    execution: dict[str, JSONValue] = {
        "phase_memory": [],
        "training_wall_seconds": None,
        "training_result": None,
        "event_stream_ref": None,
        "event_stream_semantic_sha256": None,
        "checkpoint_commits": [],
        "asset_evidence": [],
        "resource_profiles": [],
        "selected_tensor_bundle_ref": None,
        "selected_tensor_bundle_sha256": None,
        "selected_tensor_names": [],
        "data_boundary": None,
        "step_telemetry": [],
        "failure": None,
    }
    if plan["run_kind"] == "failure_output_collision":
        execution["failure"] = _run_collision_check(root, plan)
        status = "EXPECTED_FAILURE_CONFIRMED"
    else:
        try:
            execution = _execute_training(root, plan, request)
        except Exception as error:
            if not expected_failure:
                raise
            message = f"{type(error).__name__}:{error}"
            expected_code = str(plan["expected_failure_code"])
            if expected_code not in message:
                raise Stage0G5WorkerError(
                    f"G5_WORKER_UNEXPECTED_FAILURE:{expected_code}:{message}"
                ) from error
            execution["failure"] = {
                "exception_class": type(error).__name__,
                "exception_message": str(error),
                "last_valid_step": (
                    1
                    if plan["run_kind"] == "failure_checkpoint_write"
                    else max(
                        (
                            int(item["step"])
                            for item in execution.get("step_telemetry", [])
                            if isinstance(item, Mapping)
                            and isinstance(item.get("step"), int)
                        ),
                        default=0,
                    )
                ),
                "safe_retry": "correct the declared cause and start a fresh immutable run",
                "sentinel_sha256": None,
                "success_commit_absent": _success_commit_absent(root, request),
            }
            if not execution["failure"]["success_commit_absent"]:  # type: ignore[index]
                raise Stage0G5WorkerError("G5_WORKER_FAILURE_PUBLISHED_SUCCESS") from error
            status = "EXPECTED_FAILURE_CONFIRMED"
        else:
            if expected_failure:
                raise Stage0G5WorkerError("G5_WORKER_EXPECTED_FAILURE_DID_NOT_OCCUR")
            status = "PASS"

    report: dict[str, JSONValue] = {
        "schema_version": WORKER_REPORT_SCHEMA,
        "run_id": plan["run_id"],
        "run_kind": plan["run_kind"],
        "repeat_index": plan["repeat_index"],
        "status": status,
        "pid": os.getpid(),
        "parent_pid": os.getppid(),
        "started_at": started_at,
        "completed_at": _now(),
        "generator_git_commit": plan["generator_git_commit"],
        "plan_ref": plan_ref,
        "plan_sha256": plan["artifact_hash"],
        "config_ref": plan["config_ref"],
        "config_sha256": plan["config_sha256"],
        "config_hash": request.config.config_hash,
        "config_full_hash": request.config.full_hash,
        "environment_ref": plan["environment_ref"],
        "environment_sha256": plan["environment_sha256"],
        "environment_hash": request.environment.environment_hash,
        "device": device,
        **execution,
    }
    report["artifact_hash"] = canonical_json_hash(report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(report_path, report)
    return report


__all__ = [
    "Stage0G5Observer",
    "Stage0G5WorkerError",
    "WORKER_PLAN_SCHEMA",
    "WORKER_REPORT_SCHEMA",
    "run_stage0_g5_worker",
]
