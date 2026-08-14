"""Executable configuration-to-behavior compiler owned by S1.2.

This is deliberately a component contract, not an alternate hash formatter.  It
turns the public resolved configuration into the control decisions a task
runner must obey (execution mode, data/precision policy, checkpoint contents,
and publication disposition).  It gives retained switches a testable semantic
consumer before a task runner is allowed to consume the configuration.
"""

from __future__ import annotations

from typing import Any

from .contracts import ResolvedConfig, ResolvedConfigV2


def _v1_behavior(config: ResolvedConfig) -> dict[str, Any]:
    identity = config.section("identity")
    runtime = config.section("runtime")
    data = config.section("data")
    loss = config.section("loss")
    batching = config.section("batching")
    distributed = config.section("distributed")
    precision = config.section("precision")
    optimizer = config.section("optimizer")
    logging = config.section("logging")
    checkpoint = config.section("checkpoint")
    importance = config.section("importance")
    path = config.section("path_integration")
    pruning = config.section("pruning")
    return {
        "run_mode": "formal_evidence" if identity["run_intent"] == "formal" else "local_fixture",
        "worktree_policy": "clean_required" if not runtime["allow_dirty_worktree"] else "local_dirty_allowed",
        "asset_access": "offline_only" if runtime["offline"] else "network_permitted",
        "device_plan": f"{runtime['device']}:{distributed['backend']}",
        "statistical_assumptions": (
            "unbiased_common_mean"
            if data["weights_exogenous"] and data["common_mean_assumption"]
            else "assumption_limited"
        ),
        "loss_plan": f"{loss['task_type']}:{loss['reduction']}:{loss['weighting']}",
        "batch_sync": "deferred_no_sync" if batching["no_sync"] else "synchronous",
        "precision_plan": (
            f"autocast:{precision['compute_dtype']}:path_accum={precision['path_accumulation_dtype']}"
            if precision["amp"]
            else f"float32_no_autocast:path_accum={precision['path_accumulation_dtype']}"
        ),
        "optimizer_plan": f"{optimizer['type']}:safe_unfused",
        "telemetry_plan": "jsonl_plus_tensorboard" if logging["tensorboard"] else "jsonl_only",
        "checkpoint_commit": "two_phase_required" if checkpoint["two_phase_commit"] else "invalid",
        "importance_plan": f"{importance['estimator_name']}:{importance['clip_mode']}",
        "formal_decision_policy": "required" if importance["require_decision_for_formal"] else "not_required",
        "path_plan": "enabled" if path["enabled"] else "disabled",
        "pruning_plan": f"{pruning['strategy']}:{pruning['scope']}" if pruning["enabled"] else "disabled",
        # The remaining scalar/list fields are compiled into the task input
        # envelope.  This is a component payload, not a run identity: runners
        # may consume it directly without re-reading arbitrary config fields.
        "input_envelope": {
            "identity": identity,
            "runtime": runtime,
            "model": config.section("model"),
            "data": data,
            "loss": loss,
            "batching": batching,
            "distributed": distributed,
            "precision": precision,
            "optimizer": optimizer,
            "logging": logging,
            "checkpoint": checkpoint,
            "importance": importance,
            "sampling": config.section("sampling"),
            "path_integration": path,
            "pruning": pruning,
            "analysis": config.section("analysis"),
        },
    }


def compile_config_behavior(config: ResolvedConfig | ResolvedConfigV2) -> dict[str, Any]:
    """Compile retained public settings into task-execution component behavior."""

    if isinstance(config, ResolvedConfig):
        return {"schema_version": "stage1-config-behavior-v1", "v1": _v1_behavior(config)}

    execution = config.section("execution")
    training = config.section("training")
    scheduler = config.section("scheduler")
    loader = config.section("data_loader")
    providers = config.section("providers")
    evaluation = config.section("evaluation")
    profiling = config.section("profiling")
    checkpoints = config.section("checkpoint_schedule")
    precision = config.section("precision_runtime")
    optimizer = config.section("optimizer_runtime")
    launcher = config.section("launcher")
    orchestration = config.section("orchestration")
    recovery = config.section("recovery")
    artifacts = config.section("artifacts")
    assert all(isinstance(value, dict) for value in (
        execution, training, scheduler, loader, providers, evaluation, profiling,
        checkpoints, precision, optimizer, launcher, orchestration, recovery, artifacts,
    ))
    return {
        "schema_version": "stage1-config-behavior-v1",
        "v1": _v1_behavior(config.base_config),
        "execution_action": "plan_only" if execution["dry_run"] else "execute",
        "blocked_input_action": "raise" if execution["fail_on_blocked"] else "record_blocked",
        "training_determinism": "enforced" if training["deterministic_algorithms"] else "best_effort",
        "scheduler_action": f"{scheduler['kind']}:{scheduler['warmup_steps']}:{scheduler['total_steps']}",
        "dataloader_action": {
            "worker_lifecycle": "persistent" if loader["persistent_workers"] else "ephemeral",
            "last_batch": "drop" if loader["drop_last"] else "keep",
            "cursor_commit": loader["cursor_policy"],
        },
        "provider_action": f"{providers['kind']}:{providers['task_type']}:{providers['task_name']}",
        "evaluation_action": "run_and_save" if evaluation["enabled"] and evaluation["save_predictions"] else ("run" if evaluation["enabled"] else "skip"),
        "profiling_action": {
            "enabled": profiling["enabled"],
            "captures": [
                name for name, enabled in (
                    ("memory", profiling["capture_memory"]),
                    ("throughput", profiling["capture_throughput"]),
                    ("communication", profiling["capture_communication"]),
                ) if enabled
            ],
            "synchronize": profiling["synchronize_device"],
        },
        "checkpoint_action": {
            "phase_end": checkpoints["save_on_phase_end"],
            "optimizer": checkpoints["save_optimizer"],
            "rng": checkpoints["save_rng"],
            "data_state": checkpoints["save_data_state"],
        },
        "precision_action": f"autocast={precision['autocast_enabled']};scaler={precision['grad_scaler_enabled']};found_inf_reduce={precision['global_found_inf_reduce']}",
        "optimizer_action": f"nesterov={optimizer['nesterov']};dampening={optimizer['dampening']}",
        "launcher_action": f"{launcher['kind']}:{launcher['backend']}:{launcher['init_method']}",
        "paired_action": f"{orchestration['paired_design']['enabled']}:{orchestration['paired_design']['design']}:{orchestration['paired_design']['budget_unit']}",
        "recovery_action": f"{recovery['mode']}:{recovery['safe_boundary']}",
        "publication_action": "partial" if artifacts["publish_partial"] else "complete_only",
        "input_envelope": {
            "training": training,
            "data_loader": loader,
            "providers": providers,
            "evaluation": evaluation,
            "profiling": profiling,
            "checkpoint_schedule": checkpoints,
            "precision_runtime": precision,
            "optimizer_runtime": optimizer,
            "launcher": launcher,
            "orchestration": orchestration,
            "recovery": recovery,
            "artifacts": artifacts,
        },
    }


__all__ = ["compile_config_behavior"]
