"""NCCL worker for the S1.9 finite→global-skip→finite contract.

This program is launched only by ``formalize_s1_9.py`` after it has acquired
the project lease for explicitly approved UUIDs.  It never selects physical
GPUs itself: torchrun sees an already UUID-isolated ``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from param_importance_nlp.core.registry import ParameterRegistry
from param_importance_nlp.core.tensors import TensorMap
from param_importance_nlp.runtime.optimizer import OptimizerBridge, compute_global_clip_factor
from param_importance_nlp.runtime.reducers import TorchDistributedReducer
from param_importance_nlp.runtime.training import OnlineImportanceTracker, TrainingRunSpec


_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_DETERMINISM_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}


class _CountingAdamW(torch.optim.AdamW):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.actual_step_calls = 0

    def step(self, closure: Any = None) -> Any:  # type: ignore[override]
        self.actual_step_calls += 1
        return super().step(closure)


class _CountingGradScaler(torch.amp.GradScaler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.unscale_calls = self.step_calls = self.update_calls = 0

    def unscale_(self, optimizer: torch.optim.Optimizer) -> Any:  # type: ignore[override]
        self.unscale_calls += 1
        return super().unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        self.step_calls += 1
        return super().step(optimizer, *args, **kwargs)

    def update(self, new_scale: float | torch.Tensor | None = None) -> None:  # type: ignore[override]
        self.update_calls += 1
        return super().update(new_scale)


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _require_pre_cuda_policy(*, cuda_visible_devices: str) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cuda_visible_devices or any(os.environ.get(name) != value for name, value in _DETERMINISM_ENV.items()):
        raise SystemExit("S1_9_DDP_PRE_CUDA_POLICY_INVALID")


def _version_wire(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def _configure_correctness_policy(*, cuda_visible_devices: str, local_rank: int, local_gpu_uuid: str) -> dict[str, Any]:
    """Set deterministic flags before set_device/NCCL and record them."""

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    nccl = getattr(torch.cuda, "nccl", None)
    summary = {
        "torch_version": str(torch.__version__),
        "cuda_runtime_version": _version_wire(torch.version.cuda),
        "cudnn_version": _version_wire(torch.backends.cudnn.version()),
        "nccl_version": _version_wire(None if nccl is None else nccl.version()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cuda_visible_devices": cuda_visible_devices,
        "local_rank": local_rank,
        "local_gpu_uuid": local_gpu_uuid,
    }
    if not (summary["deterministic_algorithms"] and summary["cudnn_deterministic"] and not summary["cudnn_benchmark"]):
        raise SystemExit("S1_9_DDP_CORRECTNESS_POLICY_INVALID")
    return summary


def _wire_tensor(value: torch.Tensor) -> dict[str, Any]:
    return {"dtype": str(value.dtype), "shape": list(value.shape), "values": [float(item) for item in value.detach().cpu().to(torch.float64).reshape(-1).tolist()]}


def _wire(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _wire_tensor(value)
    if isinstance(value, Mapping):
        return {str(key): _wire(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_wire(item) for item in value]
    if isinstance(value, list):
        return [_wire(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"S1_9_DDP_STATE_TYPE_UNSUPPORTED:{type(value).__name__}")


def _production_tracker(model: DistributedDataParallel, optimizer: _CountingAdamW) -> OnlineImportanceTracker:
    """Build the actual production tracker, including its accumulator state."""

    registry = ParameterRegistry.from_model(model.module, optimizer)
    spec = TrainingRunSpec("s19-ddp-production-tracker", "local_fixture", max_steps=2, max_attempts=3, importance_enabled=True, estimator_name="u", accumulation_dtype="float32", weights_exogenous=True, common_mean_assumption=True)
    return OnlineImportanceTracker(registry, spec)


def _learning_rates_by_group(registry: ParameterRegistry) -> dict[str, float]:
    """Return the estimator's public optimizer-group learning-rate wire."""

    learning_rates: dict[str, float] = {}
    for record in registry.eligible_records:
        if record.group_id is None or record.learning_rate is None:
            raise RuntimeError("S1_9_DDP_TRACKER_LEARNING_RATES_INVALID")
        observed = float(record.learning_rate)
        previous = learning_rates.get(record.group_id)
        if previous is not None and previous != observed:
            raise RuntimeError("S1_9_DDP_TRACKER_LEARNING_RATES_INVALID")
        learning_rates[record.group_id] = observed
    expected_groups = {record.group_id for record in registry.eligible_records}
    if None in expected_groups or set(learning_rates) != expected_groups:
        raise RuntimeError("S1_9_DDP_TRACKER_LEARNING_RATES_INVALID")
    return learning_rates


def _tracker_views(tracker: OnlineImportanceTracker) -> dict[str, Any]:
    accumulator = tracker.accumulator
    return _wire({
        "signed": accumulator.signed, "positive": accumulator.positive,
        "negative_mass": accumulator.negative_mass, "absolute": accumulator.absolute,
        "raw": accumulator.raw, "raw_clipped": accumulator.raw_clipped,
        "data_movement": accumulator.data_movement,
        "net_data_movement": accumulator.net_data_movement,
        "total_movement": accumulator.total_movement,
        "total_endpoint_movement": accumulator.total_endpoint_movement,
        "weight_decay_movement": accumulator.weight_decay_movement,
        "net_weight_decay_movement": accumulator.net_weight_decay_movement,
        "actual_update_raw_importance": accumulator.actual_update_raw_importance,
        "magnitude": accumulator.magnitude,
    })


def _state(model: DistributedDataParallel, optimizer: _CountingAdamW, scheduler: Any, scaler: _CountingGradScaler, tracker: OnlineImportanceTracker, *, cursor: int, attempt: int, skipped: int) -> dict[str, Any]:
    return _wire({
        "parameters": {name: parameter.detach().clone() for name, parameter in model.module.named_parameters()},
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
        "importance_accumulator": tracker.accumulator.state_dict(),
        "importance_views": _tracker_views(tracker), "cursor": cursor,
        "attempt": attempt, "skipped": skipped,
        "rng_cpu": torch.random.get_rng_state().clone(), "rng_cuda_local": torch.cuda.get_rng_state(),
    })


def _run_sequence(*, rank: int, model: DistributedDataParallel, attempts_spec: tuple[tuple[float, bool], ...]) -> dict[str, Any]:
    device = torch.device("cuda", rank)
    optimizer = _CountingAdamW(model.parameters(), lr=0.05, weight_decay=0.1, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = _CountingGradScaler("cuda", enabled=True, init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1)
    tracker = _production_tracker(model, optimizer)
    optimizer_bridge = OptimizerBridge(dict(model.module.named_parameters()), optimizer)
    reducer = TorchDistributedReducer(integer_device=device)
    learning_rates = _learning_rates_by_group(tracker.registry)
    cursor = attempt = skipped = successful = 0
    attempts: list[dict[str, Any]] = []
    for sequence, (frozen_sample_value, inject_here) in enumerate(attempts_spec):
        pre_fetch = _state(model, optimizer, scheduler, scaler, tracker, cursor=cursor, attempt=attempt, skipped=skipped)
        # The sample identity remains fixed across the observed and reference
        # trajectories; rank-specific values exercise real DDP reduction.
        inputs = torch.tensor([[1.0 + rank + frozen_sample_value]], dtype=torch.float32, device=device)
        target = torch.tensor([[0.25 * (rank + 1)]], dtype=torch.float32, device=device)
        cursor += 1
        post_fetch = _state(model, optimizer, scheduler, scaler, tracker, cursor=cursor, attempt=attempt, skipped=skipped)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = (model(inputs) - target).square().mean()
        scaler.scale(loss).backward()
        # Preserve the actual finite scaled gradient before the rank-local
        # fault.  All ranks can use this staged candidate below; the injected
        # path subsequently proves it is discarded rather than committed.
        staged_scaled_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.module.named_parameters() if parameter.grad is not None}
        # ``found_inf`` is recorded by the sole formal unscale_.  The local
        # injection therefore must occur while gradients are still scaled.
        if inject_here:
            next(parameter for parameter in model.parameters() if parameter.grad is not None).grad.fill_(float("inf"))
        local_nonfinite = any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters())
        flag = torch.tensor([int(local_nonfinite)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        global_skip = bool(int(flag.item()))
        if global_skip:
            # Propagate before unscale so every scaler records found_inf.
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.fill_(float("inf"))
        scaler.unscale_(optimizer)
        gradients = {name: parameter.grad.detach().clone() for name, parameter in model.module.named_parameters() if parameter.grad is not None}
        if set(gradients) != {name for name, _ in model.module.named_parameters()}:
            raise RuntimeError("S1_9_DDP_GRADIENT_SET_INVALID")
        stage_gradients = gradients if not global_skip else {name: value / float(scaler.get_scale()) for name, value in staged_scaled_gradients.items()}
        gradient_map = TensorMap(stage_gradients, registry=tracker.registry)
        _, clip_factor = compute_global_clip_factor(stage_gradients, max_norm=1.0)
        # ``stage_distributed`` is production code and is deliberately called
        # before the step.  It has no accumulator side effect; the bad attempt
        # below proves this staged score is discarded rather than committed.
        staged_main, staged_raw, staged_raw_clipped = tracker.stage_distributed((gradient_map, gradient_map.clone()), (1.0, 1.0), learning_rates, clip_factor=clip_factor, reducer=reducer, rank=rank)
        before_step = _state(model, optimizer, scheduler, scaler, tracker, cursor=cursor, attempt=attempt, skipped=skipped)
        if not global_skip:
            outcome = optimizer_bridge.step(stepper=scaler.step)
            scaler.update()
            scheduler.step()
            tracker.commit(staged_main, staged_raw, staged_raw_clipped, outcome, mean_gradient=gradient_map)
            successful += 1
            staging_disposition = "COMMITTED"
        else:
            scaler.step(optimizer); scaler.update()
            tracker.record_skip()
            skipped += 1
            staging_disposition = "DISCARDED_ON_GLOBAL_SKIP"
        attempt += 1
        optimizer.zero_grad(set_to_none=True)
        after = _state(model, optimizer, scheduler, scaler, tracker, cursor=cursor, attempt=attempt, skipped=skipped)
        attempts.append({"sequence": sequence, "local_nonfinite": local_nonfinite, "global_skip": global_skip, "global_skip_event_recorded": bool(global_skip and rank == 0), "staging_disposition": staging_disposition, "pre_fetch_state": pre_fetch, "post_fetch_state": post_fetch, "before_step_state": before_step, "post_state": after, "scale_after_attempt": float(scaler.get_scale()), "growth_tracker_after_attempt": int(scaler.state_dict().get("_growth_tracker", 0))})
    tracker.accumulator.validate_invariants()
    return {"attempts": attempts, "final_state": _state(model, optimizer, scheduler, scaler, tracker, cursor=cursor, attempt=attempt, skipped=skipped), "scaler_calls": {"unscale": scaler.unscale_calls, "step": scaler.step_calls, "update": scaler.update_calls}, "actual_optimizer_step_calls": optimizer.actual_step_calls, "successful": successful, "skipped": skipped}


def _new_model(device: torch.device) -> DistributedDataParallel:
    # ``torch.manual_seed`` also seeds every visible CUDA generator.  The
    # UUID-isolated rank must seed CPU and *only* its current CUDA device.
    torch.random.default_generator.manual_seed(8019)
    torch.cuda.manual_seed(8019)
    return DistributedDataParallel(torch.nn.Linear(1, 1, bias=False, device=device, dtype=torch.float32), device_ids=[device.index], output_device=device.index, broadcast_buffers=False)


def _cpu_gradscaler_ordering_negative_control() -> bool:
    """Exercise GradScaler's real found-inf timing on CPU as a negative case."""

    def execute(*, inject_before_unscale: bool) -> tuple[torch.Tensor, int]:
        parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        optimizer = _CountingAdamW([parameter], lr=0.1, foreach=False, fused=False)
        scaler = _CountingGradScaler("cpu", enabled=True, init_scale=8.0)
        scaler.scale(parameter.sum()).backward()
        if inject_before_unscale:
            assert parameter.grad is not None
            parameter.grad.fill_(float("inf"))
        scaler.unscale_(optimizer)
        if not inject_before_unscale:
            assert parameter.grad is not None
            parameter.grad.fill_(float("inf"))
        scaler.step(optimizer); scaler.update()
        return parameter.detach(), optimizer.actual_step_calls

    correct_parameter, correct_calls = execute(inject_before_unscale=True)
    wrong_parameter, wrong_calls = execute(inject_before_unscale=False)
    return correct_calls == 0 and torch.equal(correct_parameter, torch.tensor([1.0])) and wrong_calls == 1 and not bool(torch.isfinite(wrong_parameter).all())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-rank", type=int, default=0)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--run-token", required=True)
    parser.add_argument("--approved-gpu-uuids", required=True)
    args = parser.parse_args(argv)
    local_rank, world_size = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    approved = tuple(item for item in args.approved_gpu_uuids.split(",") if item)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if world_size != 4 or not 0 <= args.inject_rank < world_size or len(approved) != 4 or len(set(approved)) != 4 or any(_UUID.fullmatch(item) is None for item in approved) or visible != args.approved_gpu_uuids or re.fullmatch(r"[0-9a-f]{40}", args.execution_commit) is None or re.fullmatch(r"[0-9a-f]{64}", args.run_token) is None or os.environ.get("S1_9_RUN_TOKEN") != args.run_token:
        raise SystemExit("S1_9_DDP_WORLD_OR_INJECTION_INVALID")
    _require_pre_cuda_policy(cuda_visible_devices=visible)
    environment_summary = _configure_correctness_policy(cuda_visible_devices=visible, local_rank=local_rank, local_gpu_uuid=approved[local_rank])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    try:
        observed = _run_sequence(rank=local_rank, model=_new_model(torch.device("cuda", local_rank)), attempts_spec=((0.0, False), (99.0, local_rank == args.inject_rank), (1.0, False)))
        reference = _run_sequence(rank=local_rank, model=_new_model(torch.device("cuda", local_rank)), attempts_spec=((0.0, False), (1.0, False)))
        gathered_observed: list[Any] = [None] * world_size
        gathered_reference: list[Any] = [None] * world_size
        gathered_environment: list[Any] = [None] * world_size
        dist.all_gather_object(gathered_observed, observed)
        dist.all_gather_object(gathered_reference, reference)
        dist.all_gather_object(gathered_environment, environment_summary)
        if dist.get_rank() == 0:
            skip_rows = [item["attempts"][1] for item in gathered_observed]
            per_rank_skip = [bool(item["global_skip"]) for item in skip_rows]
            pre, post_fetch, post = skip_rows[0]["pre_fetch_state"], skip_rows[0]["post_fetch_state"], skip_rows[0]["post_state"]
            skip_zero_update = all(
                item["before_step_state"]["parameters"] == item["post_state"]["parameters"]
                and item["before_step_state"]["optimizer"] == item["post_state"]["optimizer"]
                and item["before_step_state"]["scheduler"] == item["post_state"]["scheduler"]
                and {key: value for key, value in item["before_step_state"]["importance_accumulator"].items() if key != "skipped_steps"} == {key: value for key, value in item["post_state"]["importance_accumulator"].items() if key != "skipped_steps"}
                and item["before_step_state"]["importance_views"] == item["post_state"]["importance_views"]
                and item["post_state"]["importance_accumulator"]["skipped_steps"] == item["before_step_state"]["importance_accumulator"]["skipped_steps"] + 1
                and item["staging_disposition"] == "DISCARDED_ON_GLOBAL_SKIP"
                for item in skip_rows
            )
            expected_cursor = all(int(item["post_fetch_state"]["cursor"]) == int(item["pre_fetch_state"]["cursor"]) + 1 and item["post_fetch_state"]["cursor"] == item["post_state"]["cursor"] for item in skip_rows)
            next_finite_parity = all(observed_item["final_state"]["parameters"] == reference_item["final_state"]["parameters"] and observed_item["final_state"]["optimizer"] == reference_item["final_state"]["optimizer"] and observed_item["final_state"]["scheduler"] == reference_item["final_state"]["scheduler"] and {key: value for key, value in observed_item["final_state"]["importance_accumulator"].items() if key != "skipped_steps"} == reference_item["final_state"]["importance_accumulator"] and observed_item["final_state"]["importance_views"] == reference_item["final_state"]["importance_views"] for observed_item, reference_item in zip(gathered_observed, gathered_reference, strict=True))
            counters = all(item["scaler_calls"] == {"unscale": 3, "step": 3, "update": 3} and item["actual_optimizer_step_calls"] == 2 and item["successful"] == 2 and item["skipped"] == 1 for item in gathered_observed)
            rank_parity = len({_canonical(item["final_state"]).hex() for item in gathered_observed}) == 1
            progression = all(item["attempts"][1]["pre_fetch_state"]["rng_cpu"] == item["attempts"][1]["post_fetch_state"]["rng_cpu"] == item["attempts"][1]["post_state"]["rng_cpu"] and item["attempts"][1]["pre_fetch_state"]["rng_cuda_local"] == item["attempts"][1]["post_fetch_state"]["rng_cuda_local"] == item["attempts"][1]["post_state"]["rng_cuda_local"] and int(item["attempts"][1]["post_state"]["attempt"]) == int(item["attempts"][1]["pre_fetch_state"]["attempt"]) + 1 and int(item["attempts"][1]["post_state"]["skipped"]) == int(item["attempts"][1]["pre_fetch_state"]["skipped"]) + 1 for item in gathered_observed)
            scaler_rank_parity = len({tuple((attempt["scale_after_attempt"], attempt["growth_tracker_after_attempt"]) for attempt in item["attempts"]) for item in gathered_observed}) == 1
            environment_invariants = [{key: value for key, value in item.items() if key not in {"local_rank", "local_gpu_uuid"}} for item in gathered_environment]
            environment_cross_rank = (len({_canonical(item).hex() for item in environment_invariants}) == 1 and [item.get("local_rank") for item in gathered_environment] == list(range(world_size)) and [item.get("local_gpu_uuid") for item in gathered_environment] == list(approved))
            global_skip_events = sum(int(attempt["global_skip_event_recorded"]) for item in gathered_observed for attempt in item["attempts"])
            checks = {"cpu_gradscaler_wrong_order_negative_control": _cpu_gradscaler_ordering_negative_control(), "all_rank_global_skip": all(per_rank_skip), "injection_is_rank_local": sum(bool(item["local_nonfinite"]) for item in skip_rows) == 1 and bool(skip_rows[args.inject_rank]["local_nonfinite"]), "scaler_unscale_step_update_once_per_attempt": counters, "skip_zero_optimizer_and_full_importance_state_update": skip_zero_update, "cursor_rng_attempt_skip_scaler_contract": expected_cursor and progression, "next_finite_parity_with_no_skip_reference": next_finite_parity, "all_rank_final_state_parity": rank_parity, "all_rank_scaler_growth_tracker_parity": scaler_rank_parity, "global_skip_event_recorded_once": global_skip_events == 1, "environment_summary_cross_rank_consistent": environment_cross_rank}
            payload = {"schema_version": "stage1-s1-9-ddp-skip-worker-v1", "status": "PASS" if all(checks.values()) else "FAIL", "execution_commit": args.execution_commit, "run_token": args.run_token, "world_size": world_size, "approved_gpu_uuids": list(approved), "cuda_visible_devices": visible, "rank_to_uuid": {str(index): uuid for index, uuid in enumerate(approved)}, "injected_rank": args.inject_rank, "checks": checks, "environment_summary": gathered_environment[0], "per_rank_environment_summary": gathered_environment, "per_rank_observed": gathered_observed, "per_rank_reference": gathered_reference}
            payload["artifact_hash"] = hashlib.sha256(_canonical(payload)).hexdigest()
            if args.output.exists():
                raise RuntimeError("S1_9_DDP_OUTPUT_ALREADY_EXISTS")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(_canonical(payload))
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
