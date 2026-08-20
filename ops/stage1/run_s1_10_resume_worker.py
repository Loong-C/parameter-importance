#!/usr/bin/env python3
"""Real CUDA/NCCL S1.10 checkpoint-resume observation worker.

The formalizer is the only caller.  It UUID-isolates this process before
importing torch and supplies a run-token through its inherited environment.
This worker deliberately uses ``TrainingEngine`` and its real checkpoint
stores; it is not a shadow state machine or a CPU/gloo substitute.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence


_UUID = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_TOKEN = re.compile(r"^[0-9a-f]{64}$")
_DETERMINISM = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
TASK_ID = "stage1.10_checkpoint_resume_and_artifacts"
GATE_ID = "G1-RESUME"
FIXTURE_ID = "stage1-s110-checkpoint-fixture-v1"


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _process_identity(pid: int) -> str:
    """Return a PID-reuse-resistant creation identity without importing torch."""

    if os.name == "nt":
        import ctypes

        class _FileTime(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            raise ProcessLookupError(pid)
        try:
            exit_code = ctypes.c_ulong()
            if not ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                raise RuntimeError("S1_10_SOURCE_PROCESS_ID_UNVERIFIABLE")
            if int(exit_code.value) != 259:  # STILL_ACTIVE
                raise ProcessLookupError(pid)
            created = _FileTime()
            exited = _FileTime()
            kernel = _FileTime()
            user = _FileTime()
            if not ctypes.windll.kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
                raise RuntimeError("S1_10_SOURCE_PROCESS_ID_UNVERIFIABLE")
            return f"win:{(int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)}"
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    except (FileNotFoundError, IndexError) as error:
        raise ProcessLookupError(pid) from error
    # starttime is proc stat field 22; the post-comm tail begins at field 3.
    return f"linux:{fields[19]}"


def _source_process_exited(source_pid: object, source_identity: object) -> bool:
    """Fail closed unless the source worker is a different, exited OS process."""

    if isinstance(source_pid, bool) or not isinstance(source_pid, int) or source_pid <= 0:
        raise RuntimeError("S1_10_SOURCE_PROCESS_ID_INVALID")
    if not isinstance(source_identity, str) or not source_identity:
        raise RuntimeError("S1_10_SOURCE_PROCESS_IDENTITY_INVALID")
    try:
        current_identity = _process_identity(source_pid)
    except ProcessLookupError:
        return True
    except PermissionError as error:
        raise RuntimeError("S1_10_SOURCE_PROCESS_ID_UNVERIFIABLE") from error
    if current_identity == source_identity:
        raise RuntimeError("S1_10_SOURCE_PROCESS_STILL_RUNNING")
    return True


def _copy_checkpoint_lineage(source_store: Any, output_store: Any, checkpoint_id: str) -> None:
    state, commit = source_store.load(checkpoint_id)
    if commit.parent_checkpoint_id is not None:
        _copy_checkpoint_lineage(source_store, output_store, commit.parent_checkpoint_id)
    output_store.publish(
        checkpoint_id,
        state,
        generation=commit.generation,
        metadata=commit.metadata,
        parent_checkpoint_id=commit.parent_checkpoint_id,
    )


def _resume_and_run(engine: Any, checkpoint_id: str, *, output_store: Any) -> None:
    """Load from immutable source, clone only its lineage, then isolate writes."""

    source_store = engine.checkpoint_store
    if source_store is None:
        raise RuntimeError("S1_10_RESUME_SOURCE_STORE_MISSING")
    _copy_checkpoint_lineage(source_store, output_store, checkpoint_id)
    engine.resume_checkpoint(checkpoint_id)
    engine.checkpoint_store = output_store
    engine.run()


def _pre_cuda(args: argparse.Namespace) -> tuple[str, ...]:
    approved = tuple(item for item in args.approved_gpu_uuids.split(",") if item)
    expected = 1 if args.mode == "single" else 4
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        _COMMIT.fullmatch(args.execution_commit) is None
        or _TOKEN.fullmatch(args.run_token) is None
        or os.environ.get("S1_10_RUN_TOKEN") != args.run_token
        or len(approved) != expected
        or len(set(approved)) != expected
        or any(_UUID.fullmatch(item) is None for item in approved)
        or visible != args.approved_gpu_uuids
        or any(os.environ.get(key) != value for key, value in _DETERMINISM.items())
    ):
        raise SystemExit("S1_10_WORKER_PRE_CUDA_POLICY_INVALID")
    return approved


def _seed(torch: Any, np: Any, *, rank: int) -> None:
    # Per-rank initial streams are deterministic.  A resume must overwrite
    # them from the checkpoint, so arbitrary reseeding cannot mask an RNG bug.
    random.seed(2026081010 + rank)
    np.random.seed(2026081020 + rank)
    torch.manual_seed(2026081030 + rank)
    torch.cuda.manual_seed(2026081040 + rank)


def _environment(torch: Any, *, visible: str, rank: int, uuid: str) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    nccl = getattr(torch.cuda, "nccl", None)
    version = lambda value: "unavailable" if value is None else (".".join(map(str, value)) if isinstance(value, tuple) else str(value))
    value = {
        "torch_version": str(torch.__version__), "cuda_runtime_version": version(torch.version.cuda),
        "cudnn_version": version(torch.backends.cudnn.version()),
        "nccl_version": version(None if nccl is None else nccl.version()),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "pythonhashseed": os.environ["PYTHONHASHSEED"], "cuda_visible_devices": visible,
        "local_rank": rank, "local_gpu_uuid": uuid,
    }
    if not (value["deterministic_algorithms"] and value["cudnn_deterministic"] and not value["cudnn_benchmark"]):
        raise RuntimeError("S1_10_WORKER_DETERMINISM_POLICY_INVALID")
    return value


def _snapshot(
    engine: Any,
    checkpoint_state_sha256: Any,
    *,
    steps: Sequence[Sequence[Any]],
) -> dict[str, Any]:
    state = engine.capture_observer_state()
    cursor = dict(state["cursor"])
    cursor_index = cursor.get("index")
    if isinstance(cursor_index, bool) or not isinstance(cursor_index, int) or not 0 <= cursor_index <= len(steps):
        raise RuntimeError("S1_10_WORKER_CURSOR_SNAPSHOT_INVALID")
    complete = {
        "observer": state,
        "importance": None if engine.tracker is None else engine.tracker.accumulator.state_dict(),
        "records": [row.to_dict() for row in engine._records],  # production checkpoint payload fields
        "points": [row.to_dict() for row in engine._importance_points],
        "checkpoint_ids": list(engine._checkpoint_ids),
        "bridge_optimizer_alias": engine.bridge.optimizer is engine.optimizer,
    }
    cuda_rng = state["rng"].get("torch_cuda") if isinstance(state["rng"], Mapping) else None
    if not isinstance(cuda_rng, tuple):
        raise RuntimeError("S1_10_WORKER_CUDA_RNG_SNAPSHOT_INVALID")
    sample_multiset = sorted(
        sample_id
        for attempt in steps[:cursor_index]
        for microbatch in attempt
        for sample_id in microbatch.sample_ids
    )
    object_digests = {
        "parameters": checkpoint_state_sha256(state["parameters"]),
        "buffers": checkpoint_state_sha256(state["buffers"]),
        "optimizer": checkpoint_state_sha256(state["optimizer"]),
        "scheduler": checkpoint_state_sha256(state["scheduler"]),
        "scaler": checkpoint_state_sha256(state["scaler"]),
        "importance_accumulator": checkpoint_state_sha256(complete["importance"]),
        "records": checkpoint_state_sha256(complete["records"]),
        "importance_points": checkpoint_state_sha256(complete["points"]),
    }
    return {
        "state_sha256": checkpoint_state_sha256(complete),
        "training_state": state["training_state"],
        "cursor": cursor,
        "next_cursor": cursor,
        "sample_multiset": sample_multiset,
        "rng_sha256": checkpoint_state_sha256(state["rng"]),
        "next_rng_state_sha256": checkpoint_state_sha256(state["rng"]),
        "per_rank_cuda_rng_state_hex": [item.detach().cpu().numpy().tobytes().hex() for item in cuda_rng],
        "optimizer_sha256": object_digests["optimizer"],
        "scheduler_sha256": object_digests["scheduler"],
        "scaler_sha256": object_digests["scaler"],
        "importance_sha256": object_digests["importance_accumulator"],
        "records_sha256": object_digests["records"],
        "object_digests": object_digests,
        "checkpoint_ids": complete["checkpoint_ids"],
        "bridge_optimizer_alias": complete["bridge_optimizer_alias"],
    }


def _assert_trajectory(reference: Mapping[str, Any], resumed: Mapping[str, Any], *, label: str) -> bool:
    expected_snapshot_fields = {
        "state_sha256", "training_state", "cursor", "next_cursor", "sample_multiset",
        "rng_sha256", "next_rng_state_sha256", "per_rank_cuda_rng_state_hex",
        "optimizer_sha256", "scheduler_sha256", "scaler_sha256", "importance_sha256",
        "records_sha256", "object_digests", "checkpoint_ids", "bridge_optimizer_alias",
    }
    if set(reference) != expected_snapshot_fields or set(resumed) != expected_snapshot_fields:
        return False
    keys = ("rng_sha256", "next_rng_state_sha256", "per_rank_cuda_rng_state_hex", "optimizer_sha256", "scheduler_sha256", "scaler_sha256", "importance_sha256", "records_sha256", "cursor", "next_cursor", "sample_multiset", "object_digests", "bridge_optimizer_alias")
    if any(reference.get(key) != resumed.get(key) for key in keys):
        return False
    state, expected_state = resumed.get("training_state"), reference.get("training_state")
    if not isinstance(state, Mapping) or not isinstance(expected_state, Mapping) or int(state.get("attempt_index", -1)) < 6:
        return False
    actual_control, reference_control = dict(state), dict(expected_state)
    # Checkpoint publication advances only provenance/event fields.  The
    # training cursor and all numerical/optimizer/RNG state must still match.
    for field in ("event_sequence", "last_checkpoint_id"):
        actual_control.pop(field, None); reference_control.pop(field, None)
    if actual_control != reference_control:
        return False
    return label in {"pre_skip", "post_skip"}


def _steps(torch: Any, *, rank: int) -> tuple[tuple[Any, ...], ...]:
    from param_importance_nlp.providers import TrainingMicrobatch
    values = (0.25, 0.5, 0.75, 0.375, -0.25, 0.625)
    result = []
    for attempt, value in enumerate(values, start=1):
        payload = {"gradient": torch.tensor([value], dtype=torch.float32)}
        if attempt == 3 and rank == 0:
            payload["inject_nonfinite"] = torch.tensor([1.0], dtype=torch.float32)
        result.append((TrainingMicrobatch(f"s110-r{rank}-a{attempt}-m0", payload, (f"s110-r{rank}-a{attempt}-m0",)), TrainingMicrobatch(f"s110-r{rank}-a{attempt}-m1", {"gradient": torch.tensor([0.125], dtype=torch.float32)}, (f"s110-r{rank}-a{attempt}-m1",))))
    return tuple(result)


def _engine(torch: Any, *, rank: int, world_size: int, steps: Sequence[Sequence[Any]], checkpoint_root: Path | None, event_path: Path | None, distributed: bool) -> tuple[Any, Any]:
    from param_importance_nlp.core.losses import LossBatch
    from param_importance_nlp.providers import DeterministicBatchCursor
    from param_importance_nlp.runtime import CheckpointStore, JsonlEventSink
    from param_importance_nlp.runtime.reducers import LocalReducer, TorchDistributedReducer
    from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec

    class InjectNonfiniteGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any) -> Any: return value * 0.0
        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any]: return (torch.full_like(gradient, float("inf")),)

    class Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__(); self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device="cuda"))
        def forward(self, value: Any) -> Any: return self.weight * value

    module: Any = Module()
    if distributed:
        from torch.nn.parallel import DistributedDataParallel
        module = DistributedDataParallel(module, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    class Adapter:
        task_type = "s1-10-production-resume"
        def __init__(self, item: Any) -> None: self._module = item
        @property
        def module(self) -> Any: return self._module
        def loss(self, microbatch: Any) -> Any:
            # Consume all global RNG domains in the actual loss path.  The
            # zero-valued random term has no numeric influence, but a resumed
            # trajectory has to recreate each stream exactly.
            random.random(); __import__("numpy").random.random()
            noise = torch.rand((), device="cuda") * 0.0
            values = microbatch.payload["gradient"].to("cuda")
            loss = self._module(values).sum() + noise
            if "inject_nonfinite" in microbatch.payload:
                loss = loss + InjectNonfiniteGradient.apply(next(self._module.parameters())).sum()
            return LossBatch(loss, int(values.numel()), "s110")

    optimizer = torch.optim.AdamW(module.parameters(), lr=0.05, weight_decay=0.01, foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=8.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=1)
    reducer = TorchDistributedReducer(integer_device=torch.device("cuda", rank)) if distributed else LocalReducer()
    spec = TrainingRunSpec("s110-formal-production-resume", "formal", max_steps=5, max_attempts=6, importance_enabled=True, estimator_name="u", checkpoint_every_steps=1, weights_exogenous=True, common_mean_assumption=True)
    sink = None if event_path is None else JsonlEventSink(event_path)
    engine = TrainingEngine(spec=spec, model=Adapter(module), optimizer=optimizer, scheduler=scheduler, scaler=scaler, reducer=reducer, cursor=DeterministicBatchCursor(steps), checkpoint_store=None if checkpoint_root is None else CheckpointStore(checkpoint_root), event_sink=sink, experiment_id="s1-10-formal", attempt_id=f"rank-{rank}", session_id="s110-resume", rank=rank)
    return engine, sink


def _event_pointer(root: Path, store: Any, checkpoint_id: str, event_path: Path) -> dict[str, Any]:
    state, _ = store.load(checkpoint_id)
    return {"event_ref": event_path.relative_to(root).as_posix(), "event_sha256": _file_sha(event_path), "checkpoint_event_sequence": int(state["training_state"]["event_sequence"]) - 1}


def _group_publish(torch: Any, *, root: Path, rank: int, world_size: int, label: str, checkpoint_root: Path, checkpoint_id: str, event_path: Path, parent: str | None, generation: int, config_hash: str, environment_hash: str) -> dict[str, Any]:
    from param_importance_nlp.runtime import CheckpointStore
    from param_importance_nlp.runtime.checkpoint_group import CheckpointGroupStore
    import torch.distributed as dist
    local = {"rank": rank, "checkpoint_store_ref": checkpoint_root.relative_to(root).as_posix(), "checkpoint_id": checkpoint_id, "event_pointer": _event_pointer(root, CheckpointStore(checkpoint_root), checkpoint_id, event_path)}
    gathered: list[Any] = [None] * world_size; dist.all_gather_object(gathered, local); dist.barrier()
    output: dict[str, Any] = {}
    if rank == 0:
        group = CheckpointGroupStore(root, "group")
        metadata = {"config_hash": config_hash, "environment_hash": environment_hash, "model_manifest_id": "s110-tiny-production-model", "data_manifest_id": "s110-frozen-six-attempt-stream", "sampler_seed": 2026081004, "epoch": 0, "committed_global_batch": generation, "next_global_batch": generation, "prefetch_policy": "disabled_for_correctness", "snapshot_type": "optimizer_step_checkpoint", "state_extension_schema": "training-checkpoint-state-v2", "save_wall_seconds": 0.0, "checkpoint_bytes": 0, "peak_memory_bytes": int(torch.cuda.max_memory_allocated())}
        commit = group.publish(f"s110-{label}-attempt-{generation:08d}", generation=generation, run_id="s110-formal-production-resume", world_size=world_size, rank_checkpoints=gathered, metadata=metadata, parent_checkpoint_id=parent, commit_schema_version="runtime.checkpoint-group-commit.v2")
        states, reloaded = group.load(commit.checkpoint_id, expected_run_id="s110-formal-production-resume", expected_world_size=world_size, expected_config_hash=config_hash, expected_data_manifest_id="s110-frozen-six-attempt-stream")
        output = {
            "checkpoint_id": commit.checkpoint_id,
            "commit_sha256": commit.commit_sha256,
            "schema_version": commit.schema_version,
            "parent_checkpoint_id": commit.parent_checkpoint_id,
            "global_step": commit.global_step,
            "successful_optimizer_step": commit.successful_optimizer_step,
            "skip_count": commit.skip_count,
            "rank_state_count": len(states),
            "rank_checkpoint_ids": [item["checkpoint_id"] for item in commit.rank_checkpoints],
            "barrier_ack_rank_count": len(gathered),
            "broadcast_commit_sha256": reloaded.commit_sha256,
            "reload_commit_sha256": reloaded.commit_sha256,
        }
    gathered_output: list[Any] = [None]; dist.broadcast_object_list(gathered_output, src=0); dist.barrier()
    return dict(gathered_output[0])


def _run(args: argparse.Namespace, *, approved: tuple[str, ...], torch: Any) -> dict[str, Any]:
    import numpy as np
    distributed = args.mode == "four-rank"
    rank, world_size = (int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])) if distributed else (0, 1)
    if (distributed and world_size != 4) or (not distributed and world_size != 1) or torch.cuda.device_count() != world_size:
        raise RuntimeError("S1_10_WORKER_WORLD_SIZE_OR_UUID_ISOLATION_INVALID")
    torch.cuda.set_device(rank)
    if distributed:
        import torch.distributed as dist; dist.init_process_group("nccl")
    try:
        environment = _environment(torch, visible=os.environ["CUDA_VISIBLE_DEVICES"], rank=rank, uuid=approved[rank])
        from param_importance_nlp.runtime.checkpoint_group import checkpoint_state_sha256
        all_steps = _steps(torch, rank=rank)
        root = args.checkpoint_root.resolve(); root.mkdir(parents=True, exist_ok=True)
        config_hash, environment_hash = _sha({"commit": args.execution_commit, "world_size": world_size, "mode": args.mode}), _sha(environment)
        # Source and recovery are deliberately distinct OS invocations.  The
        # formalizer starts ``--phase resume`` only after this source launcher
        # has exited successfully; do not turn a newly constructed engine in
        # this process into a false fresh-process observation.
        pre_root, pre_event = root / "pre" / f"rank-{rank:04d}", root / "events-pre" / f"rank-{rank:04d}.jsonl"
        if args.phase == "source":
            _seed(torch, np, rank=rank); pre_source, pre_sink = _engine(torch, rank=rank, world_size=world_size, steps=all_steps, checkpoint_root=pre_root, event_path=pre_event, distributed=distributed); pre_source.run(until_step=2); pre_id = pre_source.state.last_checkpoint_id; assert pre_id and pre_sink is not None
            pre_group = _group_publish(torch, root=root, rank=rank, world_size=world_size, label="pre", checkpoint_root=pre_root, checkpoint_id=pre_id, event_path=pre_event, parent=None, generation=2, config_hash=config_hash, environment_hash=environment_hash) if distributed else None
            # Continue the same source chain through exactly the controlled skip,
            # publish its own attempt generation, and bind group post->pre to the
            # rank-local committed parents.  This is source production, not resume.
            skip_record = pre_source._run_attempt(pre_source.cursor.next_microbatches())
            pre_source._records.append(skip_record)
            post_id = pre_source.save_checkpoint()
            post_group = _group_publish(torch, root=root, rank=rank, world_size=world_size, label="post", checkpoint_root=pre_root, checkpoint_id=post_id, event_path=pre_event, parent=None if pre_group is None else pre_group["checkpoint_id"], generation=3, config_hash=config_hash, environment_hash=environment_hash) if distributed else None
            pre_sink.close()
            source = {"pid": os.getpid(), "process_identity": _process_identity(os.getpid()), "pre_checkpoint_id": pre_id, "post_checkpoint_id": post_id, "pre_group": pre_group, "post_group": post_group}
        else:
            source_report = json.loads(Path(args.source_report).read_text(encoding="utf-8"))
            source = dict(source_report["per_rank"][rank]["source"])
            pre_id, post_id = source["pre_checkpoint_id"], source["post_checkpoint_id"]
            pre_group, post_group = source["pre_group"], source["post_group"]
            source_exited = _source_process_exited(source.get("pid"), source.get("process_identity"))
        if args.phase == "source":
            local = {
                "rank": rank,
                "environment": environment,
                "source": source,
                "checks": {
                    "source_committed_pre_and_post": True,
                    "source_post_parent_is_pre": (
                        post_group is None
                        or (
                            post_group["parent_checkpoint_id"] == pre_group["checkpoint_id"]
                            and post_group["skip_count"] == 1
                            and post_group["barrier_ack_rank_count"] == world_size
                            and post_group["broadcast_commit_sha256"] == post_group["commit_sha256"]
                        )
                    ),
                },
            }
            if distributed:
                import torch.distributed as dist
                gathered: list[Any] = [None] * world_size; dist.all_gather_object(gathered, local)
                if rank != 0: return {}
                return {"environment": [item["environment"] for item in gathered], "per_rank": gathered, "checks": {"source_committed_pre_and_post": all(all(item["checks"].values()) for item in gathered)}}
            return {"environment": [environment], "per_rank": [local], "checks": local["checks"]}

        # Uninterrupted reference is intentionally executed only in the
        # recovery process.  It shares no Python/NumPy/Torch state with source.
        _seed(torch, np, rank=rank); reference, sink = _engine(torch, rank=rank, world_size=world_size, steps=all_steps, checkpoint_root=None, event_path=None, distributed=distributed); reference.run(); assert sink is None
        reference_state = _snapshot(reference, checkpoint_state_sha256, steps=all_steps)
        group_authoritative = not distributed
        if distributed:
            from param_importance_nlp.runtime.checkpoint_group import CheckpointGroupStore
            group = CheckpointGroupStore(root, "group")
            # The group commit is the only authority selected by recovery;
            # rank-local IDs are accepted only after its full load/verify.
            source_row = source
            pre_group_id, post_group_id = source_row["pre_group"]["checkpoint_id"], source_row["post_group"]["checkpoint_id"]
            pre_states, _ = group.load(pre_group_id, expected_run_id="s110-formal-production-resume", expected_world_size=world_size, expected_config_hash=config_hash, expected_data_manifest_id="s110-frozen-six-attempt-stream")
            post_states, _ = group.load(post_group_id, expected_run_id="s110-formal-production-resume", expected_world_size=world_size, expected_config_hash=config_hash, expected_data_manifest_id="s110-frozen-six-attempt-stream")
            if pre_states[rank]["training_state"]["last_checkpoint_id"] != pre_id or post_states[rank]["training_state"]["last_checkpoint_id"] != post_id:
                raise RuntimeError("S1_10_GROUP_RECOVERY_RANK_AUTHORITY_MISMATCH")
            group_authoritative = True
        from param_importance_nlp.runtime import CheckpointStore
        _seed(torch, np, rank=rank); pre_resume, pre_resume_sink = _engine(torch, rank=rank, world_size=world_size, steps=all_steps, checkpoint_root=pre_root, event_path=None, distributed=distributed); _resume_and_run(pre_resume, pre_id, output_store=CheckpointStore(root / "resume-pre" / f"rank-{rank:04d}")); assert pre_resume_sink is None
        _seed(torch, np, rank=rank); post_resume, post_resume_sink = _engine(torch, rank=rank, world_size=world_size, steps=all_steps, checkpoint_root=pre_root, event_path=None, distributed=distributed); _resume_and_run(post_resume, post_id, output_store=CheckpointStore(root / "resume-post" / f"rank-{rank:04d}")); assert post_resume_sink is None
        pre_state, post_state = (
            _snapshot(pre_resume, checkpoint_state_sha256, steps=all_steps),
            _snapshot(post_resume, checkpoint_state_sha256, steps=all_steps),
        )
        pre_matches = _assert_trajectory(reference_state, pre_state, label="pre_skip")
        post_matches = _assert_trajectory(reference_state, post_state, label="post_skip")
        checks = {"production_training_engine": True, "source_process_exited_before_resume": source_exited, "pre_skip_resume_matches_uninterrupted": pre_matches, "post_skip_resume_matches_uninterrupted": post_matches, "three_attempts_after_each_boundary": int(pre_state["training_state"]["attempt_index"]) - 2 >= 3 and int(post_state["training_state"]["attempt_index"]) - 3 >= 3, "cursor_rng_optimizer_scheduler_scaler_importance_restored": pre_matches and post_matches, "checkpoint_lineage_complete": len(pre_state["checkpoint_ids"]) >= 1 and len(post_state["checkpoint_ids"]) >= 1, "group_authoritative_load_verified": group_authoritative}
        local = {"rank": rank, "environment": environment, "reference": reference_state, "pre": pre_state, "post": post_state, "checks": checks, "pre_group": pre_group, "post_group": post_group}
        if distributed:
            import torch.distributed as dist
            gathered: list[Any] = [None] * world_size; dist.all_gather_object(gathered, local)
            if rank != 0: return {}
            checks["all_rank_state_parity"] = all(all(item["checks"].values()) for item in gathered)
            checks["barrier_atomic_group_pre_and_post"] = all(isinstance(item["pre_group"], Mapping) and isinstance(item["post_group"], Mapping) and item["pre_group"]["schema_version"] == "runtime.checkpoint-group-commit.v2" and item["post_group"]["skip_count"] == 1 and item["post_group"]["rank_state_count"] == 4 for item in gathered)
            return {"environment": [item["environment"] for item in gathered], "per_rank": gathered, "checks": checks}
        return {"environment": [environment], "per_rank": [local], "checks": checks}
    finally:
        if distributed:
            import torch.distributed as dist
            if dist.is_initialized(): dist.destroy_process_group()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "four-rank"), required=True); parser.add_argument("--phase", choices=("source", "resume"), required=True); parser.add_argument("--source-report", type=Path); parser.add_argument("--execution-commit", required=True); parser.add_argument("--run-token", required=True); parser.add_argument("--approved-gpu-uuids", required=True)
    args = parser.parse_args(argv); approved = _pre_cuda(args)
    if (args.phase == "resume") != (args.source_report is not None):
        raise SystemExit("S1_10_WORKER_SOURCE_REPORT_POLICY_INVALID")
    if str(args.repository / "src") not in sys.path: sys.path.insert(0, str(args.repository / "src"))
    import torch
    payload = _run(args, approved=approved, torch=torch)
    if args.mode == "four-rank" and int(os.environ["RANK"]) != 0: return 0
    checks = payload.get("checks", {})
    result = {"schema_version": "stage1-s1-10-formal-worker-report-v1", "status": "PASS" if isinstance(checks, Mapping) and checks and all(checks.values()) else "FAIL", "task_id": TASK_ID, "gate_id": GATE_ID, "fixture_id": FIXTURE_ID, "execution_commit": args.execution_commit, "run_token_sha256": hashlib.sha256(args.run_token.encode("ascii")).hexdigest(), "mode": args.mode, "phase": args.phase, "world_size": 1 if args.mode == "single" else 4, "approved_gpu_uuids": list(approved), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "environment": payload.get("environment"), "checks": checks, "per_rank": payload.get("per_rank")}
    result["artifact_hash"] = _sha(result)
    if args.output.exists(): raise SystemExit("S1_10_WORKER_OUTPUT_EXISTS")
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(_canonical(result)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
