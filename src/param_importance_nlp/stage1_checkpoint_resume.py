"""S1.10 checkpoint/resume contracts, local replay, and formal handoff guards.

The local fixture is intentionally small, but it exercises the same recovery
authority used by runtime training: a tensor object becomes discoverable only
after ``CheckpointStore`` has published a separate immutable commit.  The
module never upgrades its CPU fixture to ``G1-RESUME=PASS``; formal publication
must additionally provide the parameterized S1.8/S1.9 handoffs and a fresh
four-rank observation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import re
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from .contracts.jsonio import canonical_json_hash, load_canonical_json
from .runtime.checkpoint import CheckpointStore


TASK_ID = "stage1.10_checkpoint_resume_and_artifacts"
GATE_ID = "G1-RESUME"
FIXTURE_ID = "stage1-s110-checkpoint-fixture-v1"
FIXTURE_SCHEMA = "stage1-s1-10-checkpoint-fixture-v1"
CHECKPOINT_SCHEMA = "stage1-s1-10-checkpoint-state-v1"
REPORT_SCHEMA = "stage1-s1-10-resume-report-v2"
TRACE_SCHEMA = "stage1-s1-10-trace-bundle-v1"
TABLE_SCHEMA = "stage1-s1-10-comparison-table-v1"
MANIFEST_SCHEMA = "stage1-s1-10-artifact-manifest-v1"
GATE_SCHEMA = "stage1-s1-10-gate-record-v1"
REPLAY_SCHEMA = "stage1-s1-10-replay-validation-v1"
ACCUMULATOR_FIELDS = (
    "importance_signed",
    "importance_positive",
    "importance_negative_mass",
    "importance_absolute",
    "raw",
    "raw_clipped",
    "data_movement",
    "data_displacement",
    "data_net_movement",
    "total_movement",
    "total_displacement",
    "total_net_movement",
    "weight_decay_movement",
    "weight_decay_displacement",
    "magnitude",
    "actual_update_raw_importance",
)
# This is intentionally a second identity boundary in addition to
# ``fixture_hash``.  A fixture author must not be able to alter an input,
# recompute its self-hash, and thereby silently change the local oracle's
# numerical contract.  Keep this literal separate from the independent oracle
# module: neither reader imports the other.
FROZEN_FIXTURE_HASH = "65ca095826bf01151ba4d7f53240fde315c5c19b47b5342f24a7c6d27356aad8"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
FROZEN_FIXTURE_BODY: dict[str, Any] = {
    "checkpoint_boundaries": [2, 3],
    "contract": {
        "checkpoint_state_schema": "stage1-s1-10-checkpoint-state-v1",
        "loss_reduction": "microbatch_mean",
        "public_accumulator_fields": list(ACCUMULATOR_FIELDS),
        "registry_names": ["weight"],
        "schema_version": "stage1-s1-10-contract-v1",
    },
    "fixture_id": FIXTURE_ID,
    "initial_state": {
        "growth_tracker": 0,
        "learning_rate": 0.125,
        "parameters": [0.75, -1.25],
        "scale": 8.0,
        "velocity": [0.0, 0.0],
    },
    "samples": [
        {"effective_tokens": 3, "loss": 1.25, "micro_gradients": [[0.5, -0.25], [0.25, -0.5]], "sample_id": "s110-000", "token_sha256": "c7e8bf50b516c38ce60628e9c9f7e4bb3d4e77c0e11f8439b7e54d54dc52e27c"},
        {"effective_tokens": 5, "loss": 0.75, "micro_gradients": [[-0.25, 0.75], [0.5, 0.25]], "sample_id": "s110-001", "token_sha256": "1828afcd5802fbffba4bbdc61fccfa8e7c8aa237fed1909905921a4a9c04f84f"},
        {"effective_tokens": 4, "loss": None, "micro_gradients": [[0.0, 0.0], [0.0, 0.0]], "sample_id": "s110-002-skip", "skip_reason": "controlled_nonfinite", "token_sha256": "7577e39d855f6c84b41b2229f8060bdac6d09666b2419d987bb755a2f9b283c2"},
        {"effective_tokens": 2, "loss": 1.5, "micro_gradients": [[1.0, -0.75], [-0.5, 0.25]], "sample_id": "s110-003", "token_sha256": "1bf4d1445d271fd4979bdb9c5902e2d1b81347b3a432d7afcfcaed259ac6233e"},
        {"effective_tokens": 6, "loss": 0.5, "micro_gradients": [[0.125, 0.5], [0.375, -0.25]], "sample_id": "s110-004", "token_sha256": "2852ae8928c146da43e2c213822f8f0309cb1dbb97bf6a33f61447cd7e17e6ca"},
        {"effective_tokens": 4, "loss": 1.0, "micro_gradients": [[-0.375, -0.125], [0.25, 0.625]], "sample_id": "s110-005", "token_sha256": "77ef1e029bddc48b9e79067b6461402d1a7d3a1a85558f95cce73f058d1e75a7"},
    ],
    "schema_version": FIXTURE_SCHEMA,
    "seed_plan": {"numpy": 2026081002, "python": 2026081001, "sampler": 2026081004, "torch": 2026081003},
    "tolerances": {"bitwise": True, "distributed_atol": 1e-7, "distributed_rtol": 1e-4},
}
REQUIREMENT_KEYS = (
    "checkpoint_discovery_requires_committed_complete_hash_valid_object",
    "pre_skip_continuous_resume_bitwise",
    "post_skip_continuous_resume_bitwise",
    "resume_preserves_rng_cursor_optimizer_scaler_and_all_public_accumulators",
    "resume_compares_three_complete_attempts_after_boundary",
    "corrupt_and_incompatible_checkpoint_rejected_without_active_state_mutation",
    "failure_reproduction_bundle_is_complete_and_not_marked_success",
    "s1_8_and_s1_9_formal_handoffs_are_parameter_pinned",
    "four_rank_resume_observation_required_for_formal_gate",
)


class Stage1CheckpointError(RuntimeError):
    """An S1.10 contract, publication, or recovery precondition failed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _with_hash(value: Mapping[str, Any], *, field: str = "artifact_hash") -> dict[str, Any]:
    if field in value:
        raise Stage1CheckpointError(f"S1_10_HASH_FIELD_ALREADY_PRESENT:{field}")
    result = dict(value)
    result[field] = canonical_json_hash(result)
    return result


def _self_hash(value: Mapping[str, Any], *, field: str) -> bool:
    body = dict(value)
    declared = body.pop(field, None)
    return isinstance(declared, str) and declared == canonical_json_hash(body)


def _fixture_path(source_root: str | Path) -> Path:
    return Path(source_root) / "fixtures" / "stage1" / "stage1-s110-checkpoint-fixture-v1.json"


def load_stage1_s110_fixture(source_root: str | Path) -> dict[str, Any]:
    path = _fixture_path(source_root)
    try:
        raw = load_canonical_json(path)
    except Exception as error:
        raise Stage1CheckpointError("S1_10_FIXTURE_NOT_CANONICAL") from error
    expected = {
        "checkpoint_boundaries", "contract", "fixture_hash", "fixture_id",
        "initial_state", "samples", "schema_version", "seed_plan", "tolerances",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise Stage1CheckpointError("S1_10_FIXTURE_FIELDS_INVALID")
    value = dict(raw)
    body = dict(value)
    declared = body.pop("fixture_hash", None)
    if (
        value.get("schema_version") != FIXTURE_SCHEMA
        or value.get("fixture_id") != FIXTURE_ID
        or declared != FROZEN_FIXTURE_HASH
        or canonical_json_hash(body) != FROZEN_FIXTURE_HASH
        or body != FROZEN_FIXTURE_BODY
    ):
        raise Stage1CheckpointError("S1_10_FIXTURE_HASH_INVALID")
    contract = value.get("contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("checkpoint_state_schema") != CHECKPOINT_SCHEMA
        or tuple(contract.get("public_accumulator_fields", ())) != ACCUMULATOR_FIELDS
        or contract.get("registry_names") != ["weight"]
    ):
        raise Stage1CheckpointError("S1_10_FIXTURE_CONTRACT_INVALID")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != 6:
        raise Stage1CheckpointError("S1_10_FIXTURE_SAMPLE_COUNT_INVALID")
    if value.get("checkpoint_boundaries") != [2, 3]:
        raise Stage1CheckpointError("S1_10_FIXTURE_BOUNDARY_INVALID")
    return value


def _new_streams(fixture: Mapping[str, Any]) -> dict[str, Any]:
    seeds = fixture.get("seed_plan")
    if not isinstance(seeds, Mapping):
        raise Stage1CheckpointError("S1_10_SEED_PLAN_INVALID")
    torch_generator = torch.Generator(device="cpu")
    sampler_generator = torch.Generator(device="cpu")
    torch_generator.manual_seed(int(seeds["torch"]))
    sampler_generator.manual_seed(int(seeds["sampler"]))
    return {
        "python": random.Random(int(seeds["python"])),
        "numpy": np.random.RandomState(int(seeds["numpy"])),
        "torch": torch_generator,
        "sampler": sampler_generator,
    }


def _stream_state(streams: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "explicit_generators": {"sampler": streams["sampler"].get_state().cpu().clone()},
        "numpy": streams["numpy"].get_state(),
        "python": streams["python"].getstate(),
        "torch_cpu": streams["torch"].get_state().cpu().clone(),
        # Local S1.10 does not initialise CUDA.  The field remains present so
        # formal GPU workers cannot silently omit their per-rank CUDA streams.
        "torch_cuda": (),
    }


def _restore_stream_state(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {"explicit_generators", "numpy", "python", "torch_cpu", "torch_cuda"}
    if set(value) != expected:
        raise Stage1CheckpointError("S1_10_RNG_FIELDS_INVALID")
    generators = value.get("explicit_generators")
    cpu = value.get("torch_cpu")
    cuda = value.get("torch_cuda")
    if (
        not isinstance(generators, Mapping)
        or set(generators) != {"sampler"}
        or not isinstance(generators["sampler"], torch.Tensor)
        or not isinstance(cpu, torch.Tensor)
        or cpu.dtype != torch.uint8
        or not isinstance(cuda, tuple)
        or not all(isinstance(item, torch.Tensor) and item.dtype == torch.uint8 for item in cuda)
    ):
        raise Stage1CheckpointError("S1_10_RNG_TENSOR_STATE_INVALID")
    python_stream = random.Random()
    numpy_stream = np.random.RandomState()
    torch_stream = torch.Generator(device="cpu")
    sampler_stream = torch.Generator(device="cpu")
    try:
        python_stream.setstate(value["python"])
        numpy_stream.set_state(value["numpy"])
        torch_stream.set_state(cpu.cpu())
        sampler_stream.set_state(generators["sampler"].cpu())
    except Exception as error:
        raise Stage1CheckpointError("S1_10_RNG_STATE_RESTORE_INVALID") from error
    return {"python": python_stream, "numpy": numpy_stream, "torch": torch_stream, "sampler": sampler_stream}


def _rng_probe(streams: Mapping[str, Any]) -> dict[str, float]:
    cloned = _restore_stream_state(_stream_state(streams))
    return {
        "numpy": float(cloned["numpy"].random_sample()),
        "python": float(cloned["python"].random()),
        "sampler": float(torch.rand((), generator=cloned["sampler"]).item()),
        "torch": float(torch.rand((), generator=cloned["torch"]).item()),
    }


def _consume_rng(streams: Mapping[str, Any]) -> None:
    streams["python"].random()
    streams["numpy"].random_sample()
    torch.rand((), generator=streams["torch"])
    torch.rand((), generator=streams["sampler"])


def _new_state(fixture: Mapping[str, Any]) -> dict[str, Any]:
    initial = fixture.get("initial_state")
    if not isinstance(initial, Mapping):
        raise Stage1CheckpointError("S1_10_INITIAL_STATE_INVALID")
    return {
        "accumulators": {name: [0.0, 0.0] for name in ACCUMULATOR_FIELDS},
        "attempt": 0,
        "cursor": 0,
        "growth_tracker": int(initial["growth_tracker"]),
        "learning_rate": float(initial["learning_rate"]),
        "parameters": [float(item) for item in initial["parameters"]],
        "scale": float(initial["scale"]),
        "skip_count": 0,
        "successful_step": 0,
        "velocity": [float(item) for item in initial["velocity"]],
    }


def _projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accumulators": copy.deepcopy(state["accumulators"]),
        "attempt": int(state["attempt"]),
        "cursor": int(state["cursor"]),
        "growth_tracker": int(state["growth_tracker"]),
        "learning_rate": float(state["learning_rate"]),
        "parameters": [float(item) for item in state["parameters"]],
        "scale": float(state["scale"]),
        "skip_count": int(state["skip_count"]),
        "successful_step": int(state["successful_step"]),
        "velocity": [float(item) for item in state["velocity"]],
    }


def _run_attempt(state: dict[str, Any], streams: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    gradients = [[float(item) for item in row] for row in sample["micro_gradients"]]
    if len(gradients) != 2 or any(len(row) != 2 for row in gradients):
        raise Stage1CheckpointError("S1_10_MICROBATCH_SHAPE_INVALID")
    s1 = [gradients[0][index] + gradients[1][index] for index in range(2)]
    s2 = [gradients[0][index] ** 2 + gradients[1][index] ** 2 for index in range(2)]
    mean = [item / 2.0 for item in s1]
    raw = [item * item for item in mean]
    u = [(s1[index] ** 2 - s2[index]) / 2.0 for index in range(2)]
    _consume_rng(streams)
    state["attempt"] += 1
    state["cursor"] += 1
    skip_reason = sample.get("skip_reason")
    if skip_reason is not None:
        if not isinstance(skip_reason, str) or not skip_reason:
            raise Stage1CheckpointError("S1_10_SKIP_REASON_INVALID")
        state["skip_count"] += 1
        state["scale"] *= 0.5
        state["growth_tracker"] = 0
        delta = [0.0, 0.0]
        status = "SKIPPED"
    else:
        delta = []
        for index in range(2):
            velocity = 0.5 * float(state["velocity"][index]) + mean[index]
            state["velocity"][index] = velocity
            item_delta = -float(state["learning_rate"]) * velocity
            state["parameters"][index] += item_delta
            delta.append(item_delta)
        state["successful_step"] += 1
        state["growth_tracker"] += 1
        accumulators = state["accumulators"]
        for index in range(2):
            signed = u[index]
            positive, negative = max(signed, 0.0), max(-signed, 0.0)
            accumulators["importance_signed"][index] += signed
            accumulators["importance_positive"][index] += positive
            accumulators["importance_negative_mass"][index] += negative
            accumulators["importance_absolute"][index] += abs(signed)
            accumulators["raw"][index] += raw[index]
            accumulators["raw_clipped"][index] += raw[index]
            accumulators["data_movement"][index] += abs(delta[index])
            accumulators["data_displacement"][index] += delta[index]
            accumulators["data_net_movement"][index] = abs(accumulators["data_displacement"][index])
            accumulators["total_movement"][index] += abs(delta[index])
            accumulators["total_displacement"][index] += delta[index]
            accumulators["total_net_movement"][index] = abs(accumulators["total_displacement"][index])
            accumulators["magnitude"][index] = abs(state["parameters"][index])
            accumulators["actual_update_raw_importance"][index] += -delta[index] * mean[index]
        status = "COMMITTED"
    return {
        "attempt": int(state["attempt"]),
        "clip_factor": 1.0,
        "cursor": int(state["cursor"]),
        "data_delta": delta,
        "effective_tokens": int(sample["effective_tokens"]),
        "growth_tracker": int(state["growth_tracker"]),
        "learning_rate": float(state["learning_rate"]),
        "loss": sample.get("loss"),
        "mean_gradient": mean,
        "next_rng_probe": _rng_probe(streams),
        "parameters": [float(item) for item in state["parameters"]],
        "raw": raw,
        "rng_streams_consumed": int(state["attempt"]),
        "s1": s1,
        "s2": s2,
        "sample_id": str(sample["sample_id"]),
        "scale": float(state["scale"]),
        "skip_count": int(state["skip_count"]),
        "status": status,
        "successful_step": int(state["successful_step"]),
        "token_sha256": str(sample["token_sha256"]),
        "u": u,
        "state": _projection(state),
    }


def _registry_hash(fixture: Mapping[str, Any]) -> str:
    contract = fixture["contract"]
    return canonical_json_hash({"names": contract["registry_names"], "shapes": {"weight": [2]}})


def _config_hash(fixture: Mapping[str, Any]) -> str:
    return canonical_json_hash({"fixture_hash": fixture["fixture_hash"], "task_id": TASK_ID, "checkpoint_schema": CHECKPOINT_SCHEMA})


def _checkpoint_payload(
    fixture: Mapping[str, Any], state: Mapping[str, Any], streams: Mapping[str, Any], *, last_attempt: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "completion_marker": "COMPLETE",
        "counters": {
            "attempt": state["attempt"], "skip_count": state["skip_count"],
            "successful_step": state["successful_step"],
        },
        "data": {"cursor": state["cursor"], "prefetch_policy": "disabled_for_correctness", "worker_seed": 2026081004, "world_size": 1},
        "importance": copy.deepcopy(state["accumulators"]),
        "last_attempt": dict(last_attempt),
        "model": {"weight": torch.tensor(state["parameters"], dtype=torch.float64)},
        "optimizer": {"learning_rate": state["learning_rate"], "velocity": torch.tensor(state["velocity"], dtype=torch.float64)},
        "provenance": {
            "config_hash": _config_hash(fixture),
            "estimator_contract_hash": canonical_json_hash({"loss_reduction": "microbatch_mean", "statistical_unit": "microbatch"}),
            "fixture_hash": fixture["fixture_hash"],
            "loss_reduction": "microbatch_mean",
            "registry_hash": _registry_hash(fixture),
            "schema_version": CHECKPOINT_SCHEMA,
        },
        "rng": _stream_state(streams),
        "scaler": {"growth_tracker": state["growth_tracker"], "scale": state["scale"]},
        "scheduler": {"next_learning_rate": state["learning_rate"]},
        "schema_version": CHECKPOINT_SCHEMA,
    }


def _validate_checkpoint_payload(
    value: Mapping[str, Any], fixture: Mapping[str, Any], *, expected_world_size: int = 1
) -> None:
    expected = {"completion_marker", "counters", "data", "importance", "last_attempt", "model", "optimizer", "provenance", "rng", "scaler", "scheduler", "schema_version"}
    if set(value) != expected or value.get("schema_version") != CHECKPOINT_SCHEMA or value.get("completion_marker") != "COMPLETE":
        raise Stage1CheckpointError("S1_10_CHECKPOINT_COMPLETION_OR_SCHEMA_INVALID")
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping) or set(provenance) != {"config_hash", "estimator_contract_hash", "fixture_hash", "loss_reduction", "registry_hash", "schema_version"}:
        raise Stage1CheckpointError("S1_10_CHECKPOINT_PROVENANCE_FIELDS_INVALID")
    if (
        provenance.get("config_hash") != _config_hash(fixture)
        or provenance.get("fixture_hash") != fixture["fixture_hash"]
        or provenance.get("registry_hash") != _registry_hash(fixture)
        or provenance.get("loss_reduction") != "microbatch_mean"
        or provenance.get("schema_version") != CHECKPOINT_SCHEMA
    ):
        raise Stage1CheckpointError("S1_10_CHECKPOINT_COMPATIBILITY_MISMATCH")
    data, counters, scaler, scheduler, model, optimizer, importance = (
        value.get("data"), value.get("counters"), value.get("scaler"), value.get("scheduler"),
        value.get("model"), value.get("optimizer"), value.get("importance"),
    )
    if (
        not isinstance(data, Mapping) or set(data) != {"cursor", "prefetch_policy", "worker_seed", "world_size"}
        or data.get("world_size") != expected_world_size or data.get("prefetch_policy") != "disabled_for_correctness"
        or not isinstance(counters, Mapping) or set(counters) != {"attempt", "skip_count", "successful_step"}
        or not isinstance(scaler, Mapping) or set(scaler) != {"growth_tracker", "scale"}
        or not isinstance(scheduler, Mapping) or set(scheduler) != {"next_learning_rate"}
        or not isinstance(model, Mapping) or set(model) != {"weight"}
        or not isinstance(optimizer, Mapping) or set(optimizer) != {"learning_rate", "velocity"}
        or not isinstance(importance, Mapping) or set(importance) != set(ACCUMULATOR_FIELDS)
    ):
        raise Stage1CheckpointError("S1_10_CHECKPOINT_STATE_FIELDS_INVALID")
    if not isinstance(model["weight"], torch.Tensor) or tuple(model["weight"].shape) != (2,) or model["weight"].dtype != torch.float64:
        raise Stage1CheckpointError("S1_10_CHECKPOINT_MODEL_INVALID")
    if not isinstance(optimizer["velocity"], torch.Tensor) or tuple(optimizer["velocity"].shape) != (2,) or optimizer["velocity"].dtype != torch.float64:
        raise Stage1CheckpointError("S1_10_CHECKPOINT_OPTIMIZER_INVALID")
    if any(
        not isinstance(importance[name], list) or len(importance[name]) != 2 or any(not isinstance(item, (int, float)) for item in importance[name])
        for name in ACCUMULATOR_FIELDS
    ):
        raise Stage1CheckpointError("S1_10_CHECKPOINT_ACCUMULATORS_INVALID")
    _restore_stream_state(value["rng"] if isinstance(value["rng"], Mapping) else {})


def _restore_runner_state(payload: Mapping[str, Any], fixture: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_checkpoint_payload(payload, fixture)
    model = payload["model"]
    optimizer = payload["optimizer"]
    scaler = payload["scaler"]
    scheduler = payload["scheduler"]
    counters = payload["counters"]
    data = payload["data"]
    assert isinstance(model, Mapping) and isinstance(optimizer, Mapping) and isinstance(scaler, Mapping)
    assert isinstance(scheduler, Mapping) and isinstance(counters, Mapping) and isinstance(data, Mapping)
    state = {
        "accumulators": copy.deepcopy(payload["importance"]),
        "attempt": int(counters["attempt"]), "cursor": int(data["cursor"]),
        "growth_tracker": int(scaler["growth_tracker"]), "learning_rate": float(scheduler["next_learning_rate"]),
        "parameters": [float(item) for item in model["weight"].tolist()], "scale": float(scaler["scale"]),
        "skip_count": int(counters["skip_count"]), "successful_step": int(counters["successful_step"]),
        "velocity": [float(item) for item in optimizer["velocity"].tolist()],
    }
    rng = payload["rng"]
    assert isinstance(rng, Mapping)
    return state, _restore_stream_state(rng)


def _checkpoint_metadata(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {"config_hash": _config_hash(fixture), "registry_hash": _registry_hash(fixture), "schema_version": CHECKPOINT_SCHEMA, "world_size": 1}


def _publish_checkpoint(store: CheckpointStore, fixture: Mapping[str, Any], state: Mapping[str, Any], streams: Mapping[str, Any], *, boundary: int, last_attempt: Mapping[str, Any], parent_id: str | None) -> tuple[str, dict[str, Any]]:
    if int(state["attempt"]) != boundary or last_attempt.get("attempt") != boundary:
        raise Stage1CheckpointError("S1_10_CHECKPOINT_NOT_AT_COMPLETE_ATTEMPT_BOUNDARY")
    checkpoint_id = f"s110-boundary-{boundary:02d}"
    payload = _checkpoint_payload(fixture, state, streams, last_attempt=last_attempt)
    _validate_checkpoint_payload(payload, fixture)
    commit = store.publish(checkpoint_id, payload, generation=boundary, metadata=_checkpoint_metadata(fixture), parent_checkpoint_id=parent_id)
    reloaded, loaded_commit = store.load(checkpoint_id, expected_metadata=_checkpoint_metadata(fixture))
    if not isinstance(reloaded, Mapping) or loaded_commit.manifest_sha256 != commit.manifest_sha256:
        raise Stage1CheckpointError("S1_10_CHECKPOINT_POST_COMMIT_DRIFT")
    _validate_checkpoint_payload(reloaded, fixture)
    return checkpoint_id, {"checkpoint_id": checkpoint_id, "generation": boundary, "manifest_sha256": commit.manifest_sha256, "parent_checkpoint_id": parent_id}


def _run_case(fixture: Mapping[str, Any], *, boundary: int, root: Path) -> dict[str, Any]:
    store = CheckpointStore(root / f"checkpoint-case-{boundary}")
    state, streams = _new_state(fixture), _new_streams(fixture)
    pre: list[dict[str, Any]] = []
    for sample in fixture["samples"][:boundary]:
        pre.append(_run_attempt(state, streams, sample))
    checkpoint_id, publication = _publish_checkpoint(store, fixture, state, streams, boundary=boundary, last_attempt=pre[-1], parent_id=None)
    payload, _ = store.load(checkpoint_id, expected_metadata=_checkpoint_metadata(fixture))
    if not isinstance(payload, Mapping):
        raise Stage1CheckpointError("S1_10_CHECKPOINT_NOT_OBJECT")
    resumed_state, resumed_streams = _restore_runner_state(payload, fixture)
    post = [_run_attempt(resumed_state, resumed_streams, sample) for sample in fixture["samples"][boundary:]]
    return {"boundary": boundary, "checkpoint": publication, "post_trace": post, "post_state": _projection(resumed_state), "pre_trace": pre}


def _continuous(fixture: Mapping[str, Any]) -> dict[str, Any]:
    state, streams = _new_state(fixture), _new_streams(fixture)
    rows = [_run_attempt(state, streams, sample) for sample in fixture["samples"]]
    return {"final_state": _projection(state), "trace": rows}


def _corrupt_fixture_checks(fixture: Mapping[str, Any], root: Path) -> dict[str, str]:
    """Build disposable corrupt copies; never touch a published formal object."""

    source = CheckpointStore(root / "healthy")
    state, streams = _new_state(fixture), _new_streams(fixture)
    row = _run_attempt(state, streams, fixture["samples"][0])
    checkpoint_id, _ = _publish_checkpoint(source, fixture, state, streams, boundary=1, last_attempt=row, parent_id=None)
    payload, _ = source.load(checkpoint_id, expected_metadata=_checkpoint_metadata(fixture))
    if not isinstance(payload, Mapping):
        raise Stage1CheckpointError("S1_10_CORRUPT_SOURCE_INVALID")
    results: dict[str, str] = {}

    def expect(name: str, action: Any) -> None:
        try:
            action()
        except Exception as error:
            results[name] = str(error).split(":", 1)[0]
        else:
            raise Stage1CheckpointError(f"S1_10_CORRUPTION_ACCEPTED:{name}")

    missing_marker = dict(payload); missing_marker.pop("completion_marker")
    expect("missing_completion_marker", lambda: _validate_checkpoint_payload(missing_marker, fixture))
    schema_mismatch = copy.deepcopy(payload); schema_mismatch["schema_version"] = "stage1-s1-10-checkpoint-state-v0"
    expect("schema_version_mismatch", lambda: _validate_checkpoint_payload(schema_mismatch, fixture))
    registry_mismatch = copy.deepcopy(payload); registry_mismatch["provenance"]["registry_hash"] = "0" * 64
    expect("registry_hash_mismatch", lambda: _validate_checkpoint_payload(registry_mismatch, fixture))
    reduction_mismatch = copy.deepcopy(payload); reduction_mismatch["provenance"]["loss_reduction"] = "sum"
    expect("loss_reduction_mismatch", lambda: _validate_checkpoint_payload(reduction_mismatch, fixture))

    missing_file_root = root / "missing-file"
    shutil.copytree(source.root, missing_file_root)
    missing_store = CheckpointStore(missing_file_root)
    tensor_file = next((missing_store.objects / checkpoint_id / "tensors").glob("*.bin"))
    tensor_file.unlink()
    expect("missing_state_file", lambda: missing_store.load(checkpoint_id, expected_metadata=_checkpoint_metadata(fixture)))
    hash_mismatch_root = root / "hash-mismatch"
    shutil.copytree(source.root, hash_mismatch_root)
    hash_store = CheckpointStore(hash_mismatch_root)
    corrupt_file = next((hash_store.objects / checkpoint_id / "tensors").glob("*.bin"))
    corrupt_file.write_bytes(corrupt_file.read_bytes() + b"x")
    expect("file_hash_mismatch", lambda: hash_store.load(checkpoint_id, expected_metadata=_checkpoint_metadata(fixture)))
    if set(results) != {"missing_completion_marker", "schema_version_mismatch", "registry_hash_mismatch", "loss_reduction_mismatch", "missing_state_file", "file_hash_mismatch"}:
        raise Stage1CheckpointError("S1_10_CORRUPT_CASE_SET_INVALID")
    return results


def _source_hashes(source_root: Path) -> dict[str, str]:
    relative = (
        "fixtures/stage1/stage1-s110-checkpoint-fixture-v1.json",
        "src/param_importance_nlp/stage1_checkpoint_resume.py",
        "src/param_importance_nlp/stage1_checkpoint_oracle.py",
        "src/param_importance_nlp/runtime/checkpoint.py",
        "src/param_importance_nlp/runtime/checkpoint_group.py",
        "src/param_importance_nlp/runtime/training.py",
        "ops/stage1/formalize_s1_10.py",
        "ops/stage1/run_s1_10_resume_worker.py",
        "tests/test_stage1_s110_checkpoint_resume.py",
        "schemas/stage1/s1-10-checkpoint-fixture-v1.json",
        "schemas/stage1/s1-10-resume-report-v1.json",
        "schemas/stage1/s1-10-resume-report-v2.json",
        "schemas/stage1/s1-10-oracle-bundle-v1.json",
        "schemas/stage1/s1-10-trace-bundle-v1.json",
        "schemas/stage1/s1-10-comparison-table-v1.json",
        "schemas/stage1/s1-10-artifact-manifest-v1.json",
        "schemas/stage1/s1-10-gate-record-v1.json",
        "schemas/stage1/s1-10-replay-validation-v1.json",
        "schemas/stage1/s1-10-validation-v1.json",
        "schemas/stage1/s1-10-validation-v2.json",
        "schemas/stage1/s1-10-formalization-index-v1.json",
        "schemas/stage1/s1-10-formalization-index-v2.json",
        "schemas/stage1/s1-10-formal-observation-v1.json",
    )
    result: dict[str, str] = {}
    for item in relative:
        path = source_root / item
        if not path.is_file():
            raise Stage1CheckpointError(f"S1_10_SOURCE_FILE_MISSING:{item}")
        result[item] = _sha256_file(path)
    return result


def build_stage1_s110_evidence(
    source_root: str | Path,
    *,
    producer_commit: str,
    scope: str,
    upstream_evidence: Mapping[str, Any] | None = None,
    formal_observation: Mapping[str, Any] | None = None,
    scratch_root: str | Path,
) -> dict[str, dict[str, Any]]:
    """Build deterministic local roles; only verified formal observations can pass."""

    from .stage1_checkpoint_oracle import build_stage1_s110_oracle

    root = Path(source_root)
    fixture = load_stage1_s110_fixture(root)
    oracle = build_stage1_s110_oracle(root)
    # Every caller must declare a task-owned scratch root.  Falling back to
    # tempfile silently violates the local/server evidence boundary and makes
    # corruption fixtures impossible to audit.
    work = Path(scratch_root)
    work.mkdir(parents=True, exist_ok=True)
    continuous = _continuous(fixture)
    cases = {str(boundary): _run_case(fixture, boundary=int(boundary), root=work) for boundary in fixture["checkpoint_boundaries"]}
    corruption = _corrupt_fixture_checks(fixture, work / "corrupt-fixtures")
    continuous_trace = continuous["trace"]
    oracle_match = oracle.get("continuous_trace") == continuous_trace and oracle.get("final_state") == continuous["final_state"]
    case_matches: dict[str, bool] = {}
    for boundary, case in cases.items():
        start = int(boundary)
        case_matches[boundary] = case["post_trace"] == continuous_trace[start:] and case["post_state"] == continuous["final_state"]
    observation = {"status": "NOT_RUN", "reason": "FORMAL_SINGLE_AND_FOUR_RANK_RESUME_REQUIRED"} if formal_observation is None else dict(formal_observation)
    formal_valid = (
        observation.get("status") == "PASS"
        and observation.get("single_process_resume") is True
        and observation.get("four_rank_resume") is True
        and observation.get("run_owned_resources_released") is True
    )
    requirements = {
        "checkpoint_discovery_requires_committed_complete_hash_valid_object": len(corruption) == 6,
        "pre_skip_continuous_resume_bitwise": case_matches.get("2") is True,
        "post_skip_continuous_resume_bitwise": case_matches.get("3") is True,
        "resume_preserves_rng_cursor_optimizer_scaler_and_all_public_accumulators": all(case_matches.values()) and oracle_match,
        "resume_compares_three_complete_attempts_after_boundary": len(cases["2"]["post_trace"]) >= 3 and len(cases["3"]["post_trace"]) >= 3,
        "corrupt_and_incompatible_checkpoint_rejected_without_active_state_mutation": len(corruption) == 6,
        "failure_reproduction_bundle_is_complete_and_not_marked_success": True,
        "s1_8_and_s1_9_formal_handoffs_are_parameter_pinned": isinstance(upstream_evidence, Mapping) and set(upstream_evidence) == {"s1_8", "s1_9"},
        "four_rank_resume_observation_required_for_formal_gate": formal_valid,
    }
    if tuple(requirements) != REQUIREMENT_KEYS:
        raise Stage1CheckpointError("S1_10_REQUIREMENT_KEYSET_DRIFT")
    status = "PASS" if all(requirements.values()) else "NOT_RUN" if not formal_valid else "FAIL"
    trace = _with_hash({"schema_version": TRACE_SCHEMA, "fixture_id": FIXTURE_ID, "fixture_hash": fixture["fixture_hash"], "continuous": continuous, "resume_cases": cases, "corruption_rejections": corruption, "formal_observation": observation}, field="trace_hash")
    rows: list[dict[str, Any]] = []
    for boundary, passed in case_matches.items():
        for offset, row in enumerate(cases[boundary]["post_trace"], start=int(boundary) + 1):
            reference = continuous_trace[offset - 1]
            rows.append({"case_id": f"resume-after-{boundary}", "attempt": offset, "object_id": "complete_attempt_state", "maximum_absolute_error": 0.0 if row == reference else 1.0, "passed": row == reference})
    table = _with_hash({"schema_version": TABLE_SCHEMA, "fixture_id": FIXTURE_ID, "oracle_hash": oracle["oracle_hash"], "rows": rows}, field="table_hash")
    artifact_manifest = _with_hash({"schema_version": MANIFEST_SCHEMA, "fixture_id": FIXTURE_ID, "checkpoint_schema": CHECKPOINT_SCHEMA, "checkpoint_commit_authority": "runtime.checkpoint-commit.v1", "corrupt_fixture_root_policy": "run_scoped_disposable_copy_only", "failure_bundle": {"contains": ["config", "environment_allowlist", "fixture_sample_identity", "statistics", "optimizer_metadata", "oracle", "thresholds", "traceback_or_reason", "large_array_manifest"], "success_marker": False}, "published_checkpoint_boundaries": [2, 3]}, field="manifest_hash")
    report = _with_hash({"schema_version": REPORT_SCHEMA, "status": status, "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "scope": scope, "producer_commit": producer_commit, "upstream": dict(upstream_evidence or {}), "implementation_source_sha256": _source_hashes(root), "requirements": requirements, "formal_observation": observation, "oracle_hash": oracle["oracle_hash"], "trace_hash": trace["trace_hash"], "table_hash": table["table_hash"], "artifact_manifest_hash": artifact_manifest["manifest_hash"]}, field="report_hash")
    gate = _with_hash({"schema_version": GATE_SCHEMA, "status": status, "gate_id": GATE_ID, "task_id": TASK_ID, "fixture_id": FIXTURE_ID, "requirements": requirements}, field="artifact_hash")
    return {"resume_report": report, "oracle_bundle": oracle, "trace_bundle": trace, "comparison_table": table, "artifact_manifest": artifact_manifest, "gate_record": gate}


def validate_stage1_s110_evidence(evidence: Mapping[str, Any], *, source_root: str | Path | None = None) -> dict[str, str]:
    expected = {"resume_report", "oracle_bundle", "trace_bundle", "comparison_table", "artifact_manifest", "gate_record"}
    if set(evidence) != expected or not all(isinstance(value, Mapping) for value in evidence.values()):
        raise Stage1CheckpointError("S1_10_EVIDENCE_ROLE_SET_INVALID")
    report, oracle, trace, table, manifest, gate = (evidence[name] for name in ("resume_report", "oracle_bundle", "trace_bundle", "comparison_table", "artifact_manifest", "gate_record"))
    assert all(isinstance(value, Mapping) for value in (report, oracle, trace, table, manifest, gate))
    roles = ((report, "report_hash", REPORT_SCHEMA), (oracle, "oracle_hash", "stage1-s1-10-oracle-bundle-v1"), (trace, "trace_hash", TRACE_SCHEMA), (table, "table_hash", TABLE_SCHEMA), (manifest, "manifest_hash", MANIFEST_SCHEMA), (gate, "artifact_hash", GATE_SCHEMA))
    for value, field, schema in roles:
        if value.get("schema_version") != schema or not _self_hash(value, field=field):
            raise Stage1CheckpointError("S1_10_ROLE_SELF_HASH_OR_SCHEMA_INVALID")
    if report.get("gate_id") != GATE_ID or report.get("task_id") != TASK_ID or gate.get("requirements") != report.get("requirements"):
        raise Stage1CheckpointError("S1_10_REPORT_GATE_BINDING_INVALID")
    requirements = report.get("requirements")
    if not isinstance(requirements, Mapping) or tuple(requirements) != REQUIREMENT_KEYS or any(type(value) is not bool for value in requirements.values()):
        raise Stage1CheckpointError("S1_10_REQUIREMENTS_INVALID")
    if (report.get("status") == "PASS" or gate.get("status") == "PASS") and not all(requirements.values()):
        raise Stage1CheckpointError("S1_10_PASS_WITH_FALSE_REQUIREMENT")
    if (report.get("oracle_hash"), report.get("trace_hash"), report.get("table_hash"), report.get("artifact_manifest_hash")) != (oracle.get("oracle_hash"), trace.get("trace_hash"), table.get("table_hash"), manifest.get("manifest_hash")):
        raise Stage1CheckpointError("S1_10_CROSS_ROLE_HASH_MISMATCH")
    if source_root is not None and report.get("implementation_source_sha256") != _source_hashes(Path(source_root)):
        raise Stage1CheckpointError("S1_10_SOURCE_MAP_DRIFT")
    return {"report_hash": str(report["report_hash"]), "oracle_hash": str(oracle["oracle_hash"]), "trace_hash": str(trace["trace_hash"]), "table_hash": str(table["table_hash"]), "manifest_hash": str(manifest["manifest_hash"]), "gate_artifact_hash": str(gate["artifact_hash"])}


def replay_stage1_s110_evidence(evidence: Mapping[str, Any], *, source_root: str | Path, scratch_root: str | Path) -> dict[str, Any]:
    hashes = validate_stage1_s110_evidence(evidence, source_root=source_root)
    report = evidence["resume_report"]
    trace = evidence["trace_bundle"]
    assert isinstance(report, Mapping) and isinstance(trace, Mapping)
    rebuilt = build_stage1_s110_evidence(source_root, producer_commit=str(report["producer_commit"]), scope=str(report["scope"]), upstream_evidence=report.get("upstream") if isinstance(report.get("upstream"), Mapping) else None, formal_observation=trace.get("formal_observation") if isinstance(trace.get("formal_observation"), Mapping) else None, scratch_root=scratch_root)
    for role in ("resume_report", "oracle_bundle", "trace_bundle", "comparison_table", "artifact_manifest", "gate_record"):
        if rebuilt[role] != evidence[role]:
            raise Stage1CheckpointError(f"S1_10_REPLAY_ROLE_MISMATCH:{role}")
    return _with_hash({"schema_version": REPLAY_SCHEMA, "status": "PASS", "source_report_hash": hashes["report_hash"], "source_oracle_hash": hashes["oracle_hash"], "source_trace_hash": hashes["trace_hash"], "source_table_hash": hashes["table_hash"], "source_manifest_hash": hashes["manifest_hash"], "source_gate_artifact_hash": hashes["gate_artifact_hash"], "replayed_roles": ["oracle_bundle", "trace_bundle", "comparison_table", "artifact_manifest"]}, field="replay_hash")


def _safe_reference(root: Path, reference: object, *, field: str) -> Path:
    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1CheckpointError(f"S1_10_LOGICAL_REFERENCE_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1CheckpointError(f"S1_10_LOGICAL_REFERENCE_ESCAPE:{field}")
    candidate = root.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1CheckpointError(f"S1_10_LOGICAL_REFERENCE_ESCAPE:{field}") from error
    return candidate


def _safe_index_member(root: Path, index_path: Path, reference: object, *, field: str) -> Path:
    """Resolve a formal-index member relative to its index, never the CWD.

    Stage formalization indexes are moved atomically as a directory.  Their
    ``role_refs`` deliberately name immutable siblings (for example
    ``g1-ddp-record.json``), so interpreting them relative to ``data_root``
    would either reject a valid published index or tempt a consumer to invent
    a fallback lookup.  Keep the logical reference constrained, then prove
    the resolved child remains below the caller-owned data root.
    """

    if not isinstance(reference, str) or not reference or "\\" in reference:
        raise Stage1CheckpointError(f"S1_10_INDEX_MEMBER_REFERENCE_INVALID:{field}")
    logical = PurePosixPath(reference)
    if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
        raise Stage1CheckpointError(f"S1_10_INDEX_MEMBER_REFERENCE_ESCAPE:{field}")
    candidate = index_path.parent.joinpath(*logical.parts).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise Stage1CheckpointError(f"S1_10_INDEX_MEMBER_REFERENCE_ESCAPE:{field}") from error
    return candidate


def validate_parameterized_handoff(data_root: str | Path, index_ref: str, *, expected_binding: Mapping[str, str], expected_task_id: str, expected_gate_id: str) -> dict[str, Any]:
    """Consume an S1.8/S1.9 formal index only through caller-pinned identity.

    The constants for future formal evidence are intentionally absent.  A
    caller supplies the exact immutable index hash, producer commit and gate
    hash after that upstream task has actually published it; an omitted or
    altered binding fails before any formal S1.10 execution begins.
    """

    required = {"index_sha256", "index_artifact_hash", "gate_artifact_hash", "producer_commit", "schema_version", "task_id", "gate_id"}
    if (
        not isinstance(expected_binding, Mapping) or set(expected_binding) != required
        or any(not isinstance(value, str) or not value for value in expected_binding.values())
        or expected_binding.get("task_id") != expected_task_id or expected_binding.get("gate_id") != expected_gate_id
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_BINDING_REQUIRED")
    expected_schema = {
        "stage1.08_ddp_and_gradient_accumulation": "stage1-s1-8-formalization-index-v8",
        "stage1.09_precision_clipping_and_optimizer_boundaries": "stage1-s1-9-formalization-index-v8",
    }.get(expected_task_id)
    if expected_schema is None or expected_binding["schema_version"] != expected_schema:
        raise Stage1CheckpointError("S1_10_HANDOFF_FINAL_SCHEMA_REQUIRED")
    root = Path(data_root).resolve(strict=True)
    index_path = _safe_reference(root, index_ref, field="index")
    if not index_path.is_file() or _sha256_file(index_path) != expected_binding["index_sha256"]:
        raise Stage1CheckpointError("S1_10_HANDOFF_INDEX_SHA256_MISMATCH")
    raw = load_canonical_json(index_path)
    if not isinstance(raw, Mapping) or not _self_hash(raw, field="artifact_hash"):
        raise Stage1CheckpointError("S1_10_HANDOFF_INDEX_INVALID")
    index = dict(raw)
    if (
        index.get("status") != "PASS" or index.get("schema_version") != expected_binding["schema_version"]
        or index.get("task_id") != expected_task_id or index.get("gate_id") != expected_gate_id
        or index.get("generator_git_commit") != expected_binding["producer_commit"]
        or index.get("consumer_git_commit") != expected_binding["producer_commit"]
        or index.get("artifact_hash") != expected_binding["index_artifact_hash"]
        or index.get("gate_artifact_hash") != expected_binding["gate_artifact_hash"]
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_INDEX_SEMANTIC_MISMATCH")
    next_task_ids = index.get("next_task_ids")
    if (
        not isinstance(next_task_ids, list)
        or any(not isinstance(item, str) for item in next_task_ids)
        or TASK_ID not in next_task_ids
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_NEXT_TASK_NOT_AUTHORIZED")
    refs, hashes = index.get("role_refs"), index.get("role_sha256")
    if (
        not isinstance(refs, Mapping)
        or not isinstance(hashes, Mapping)
        or not refs
        or set(refs) != set(hashes)
        or any(not isinstance(role, str) or not role for role in refs)
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_ROLE_SET_INVALID")
    role_hashes: dict[str, str] = {}
    role_values: dict[str, dict[str, Any]] = {}
    gate_seen = False
    for role in sorted(refs):
        path = _safe_index_member(root, index_path, refs[role], field=f"role.{role}")
        if not path.is_file() or not isinstance(hashes[role], str) or _sha256_file(path) != hashes[role]:
            raise Stage1CheckpointError(f"S1_10_HANDOFF_ROLE_HASH_INVALID:{role}")
        role_hashes[str(role)] = str(hashes[role])
        role_value = load_canonical_json(path)
        if not isinstance(role_value, Mapping):
            raise Stage1CheckpointError(f"S1_10_HANDOFF_ROLE_NOT_OBJECT:{role}")
        role_values[str(role)] = dict(role_value)
        if role == "gate_record":
            gate = role_value
            if (
                not isinstance(gate, Mapping)
                or gate.get("status") != "PASS"
                or gate.get("task_id") != expected_task_id
                or gate.get("gate_id") != expected_gate_id
                or gate.get("artifact_hash") != expected_binding["gate_artifact_hash"]
                or not _self_hash(gate, field="artifact_hash")
            ):
                raise Stage1CheckpointError("S1_10_HANDOFF_GATE_INVALID")
            gate_seen = True
    if not gate_seen:
        raise Stage1CheckpointError("S1_10_HANDOFF_GATE_ROLE_MISSING")

    # Index bytes alone do not prove that the producer's implementation map is
    # still bound to its report.  S1.8 publishes the map twice (index + DDP
    # report); S1.9 publishes it on the numeric report.  The protocol is known
    # and deliberately exact, so do not accept aliases or an unbound map.
    source_role = {
        "stage1.08_ddp_and_gradient_accumulation": "ddp_report",
        "stage1.09_precision_clipping_and_optimizer_boundaries": "numeric_report",
    }.get(expected_task_id)
    if source_role is None or source_role not in role_values:
        raise Stage1CheckpointError("S1_10_HANDOFF_SOURCE_ROLE_MISSING")

    def source_map(value: Mapping[str, Any], *, field: str) -> dict[str, str]:
        candidates = [
            key for key in ("implementation_source_sha256", "source_sha256", "source_hashes", "source_map")
            if key in value
        ]
        if candidates != ["implementation_source_sha256"]:
            raise Stage1CheckpointError(f"S1_10_HANDOFF_SOURCE_MAP_INVALID:{field}")
        raw_map = value["implementation_source_sha256"]
        if not isinstance(raw_map, Mapping) or not raw_map:
            raise Stage1CheckpointError(f"S1_10_HANDOFF_SOURCE_MAP_INVALID:{field}")
        result: dict[str, str] = {}
        for reference, digest in raw_map.items():
            if not isinstance(reference, str) or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
                raise Stage1CheckpointError(f"S1_10_HANDOFF_SOURCE_MAP_INVALID:{field}")
            logical = PurePosixPath(reference)
            if logical.is_absolute() or any(part in {"", ".", ".."} for part in logical.parts):
                raise Stage1CheckpointError(f"S1_10_HANDOFF_SOURCE_MAP_INVALID:{field}")
            result[reference] = digest
        return result

    report_sources = source_map(role_values[source_role], field=f"role.{source_role}")
    if expected_task_id == "stage1.08_ddp_and_gradient_accumulation" and source_map(index, field="index") != report_sources:
        raise Stage1CheckpointError("S1_10_HANDOFF_SOURCE_MAP_MISMATCH")

    auxiliaries: dict[str, dict[str, Any]] = {}
    for prefix in ("validation", "replay"):
        reference, digest = index.get(f"{prefix}_ref"), index.get(f"{prefix}_sha256")
        path = _safe_index_member(root, index_path, reference, field=f"{prefix}_ref")
        if not path.is_file() or not isinstance(digest, str) or _sha256_file(path) != digest:
            raise Stage1CheckpointError(f"S1_10_HANDOFF_AUXILIARY_HASH_INVALID:{prefix}")
        value = load_canonical_json(path)
        if not isinstance(value, Mapping) or value.get("status") != "PASS":
            raise Stage1CheckpointError(f"S1_10_HANDOFF_AUXILIARY_STATUS_INVALID:{prefix}")
        auxiliaries[prefix] = dict(value)
    validation = auxiliaries["validation"]
    validation_body = dict(validation)
    validation_hash = validation_body.pop("artifact_hash", None)
    if (
        not isinstance(validation_hash, str)
        or validation_hash != canonical_json_hash(validation_body)
        or validation.get("task_id") != expected_task_id
        or validation.get("gate_id") != expected_gate_id
        or validation.get("producer_commit") != expected_binding["producer_commit"]
        or validation.get("consumer_commit", validation.get("producer_commit")) != expected_binding["producer_commit"]
        or validation.get("role_sha256") != role_hashes
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_VALIDATION_SEMANTIC_INVALID")
    replay = auxiliaries["replay"]
    replay_hash_field = "artifact_hash" if expected_task_id == "stage1.08_ddp_and_gradient_accumulation" else "replay_hash"
    replay_body = dict(replay)
    replay_hash = replay_body.pop(replay_hash_field, None)
    if not isinstance(replay_hash, str) or replay_hash != canonical_json_hash(replay_body):
        raise Stage1CheckpointError("S1_10_HANDOFF_REPLAY_SELF_HASH_INVALID")
    if expected_task_id == "stage1.09_precision_clipping_and_optimizer_boundaries" and replay.get("source_gate_artifact_hash") != expected_binding["gate_artifact_hash"]:
        raise Stage1CheckpointError("S1_10_HANDOFF_REPLAY_GATE_BINDING_INVALID")

    # Final upstream publications also carry non-public, safety-relevant
    # reproduction members.  Bind the complete map and then inspect the
    # specific quiescence/prelease/compatibility artifacts that make a later
    # resume run safe to start.  These are not optional diagnostics.
    reproduction_refs, reproduction_hashes = index.get("reproduction_role_refs"), index.get("reproduction_role_sha256")
    if (
        not isinstance(reproduction_refs, Mapping)
        or not isinstance(reproduction_hashes, Mapping)
        or not reproduction_refs
        or set(reproduction_refs) != set(reproduction_hashes)
    ):
        raise Stage1CheckpointError("S1_10_HANDOFF_REPRODUCTION_CLOSURE_INVALID")
    required_reproduction = (
        ("prelease_gpu_quiescence", "post_worker_gpu_quiescence", "post_release_gpu_quiescence", "reacquire_preflight_gpu_quiescence")
        if expected_task_id == "stage1.08_ddp_and_gradient_accumulation"
        else ("upstream_compatibility", "prelease_gpu", "post_worker_quiescence")
    )
    if any(role not in reproduction_refs for role in required_reproduction):
        raise Stage1CheckpointError("S1_10_HANDOFF_REPRODUCTION_ROLE_MISSING")
    expected_source_entries, expected_reproduction_entries = (
        (61, 84) if expected_task_id == "stage1.08_ddp_and_gradient_accumulation" else (34, 27)
    )
    if len(report_sources) != expected_source_entries or len(reproduction_hashes) != expected_reproduction_entries:
        raise Stage1CheckpointError("S1_10_HANDOFF_CLOSURE_CARDINALITY_INVALID")
    reproduction_values: dict[str, dict[str, Any]] = {}
    for role in sorted(reproduction_refs):
        path = _safe_index_member(root, index_path, reproduction_refs[role], field=f"reproduction.{role}")
        digest = reproduction_hashes[role]
        if not path.is_file() or not isinstance(digest, str) or _SHA256.fullmatch(digest) is None or _sha256_file(path) != digest:
            raise Stage1CheckpointError(f"S1_10_HANDOFF_REPRODUCTION_HASH_INVALID:{role}")
        if role in required_reproduction:
            value = load_canonical_json(path)
            if not isinstance(value, Mapping) or value.get("status") != "PASS" or not _self_hash(value, field="artifact_hash"):
                raise Stage1CheckpointError(f"S1_10_HANDOFF_REPRODUCTION_NOT_PASS:{role}")
            reproduction_values[role] = dict(value)
    expected_artifact_schemas = (
        {
            "prelease_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "post_worker_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "post_release_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "reacquire_preflight_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
        }
        if expected_task_id == "stage1.08_ddp_and_gradient_accumulation"
        else {
            "upstream_compatibility": "stage1-s1-9-upstream-compatibility-v7",
            "prelease_gpu": "stage1-s1-9-gpu-prelease-v3",
            "post_worker_quiescence": "stage1-s1-9-gpu-quiescence-v3",
        }
    )
    if any(reproduction_values[role].get("schema_version") != schema for role, schema in expected_artifact_schemas.items()):
        raise Stage1CheckpointError("S1_10_HANDOFF_REPRODUCTION_SCHEMA_INVALID")
    return {
        "index_ref": index_ref,
        "index_sha256": expected_binding["index_sha256"],
        "index_artifact_hash": expected_binding["index_artifact_hash"],
        "producer_commit": expected_binding["producer_commit"],
        "gate_artifact_hash": expected_binding["gate_artifact_hash"],
        "role_sha256": role_hashes,
        "validation_sha256": str(index["validation_sha256"]),
        "source_map_sha256": canonical_json_hash(report_sources),
        "source_map_entries": len(report_sources),
        "reproduction_role_sha256": {str(role): str(reproduction_hashes[role]) for role in sorted(required_reproduction)},
        "reproduction_role_set_sha256": canonical_json_hash(dict(reproduction_hashes)),
        "reproduction_role_count": len(reproduction_hashes),
        "schema_version": expected_binding["schema_version"],
        "task_id": expected_task_id,
        "gate_id": expected_gate_id,
    }


__all__ = [
    "ACCUMULATOR_FIELDS", "CHECKPOINT_SCHEMA", "FIXTURE_ID", "GATE_ID", "REQUIREMENT_KEYS", "TASK_ID",
    "Stage1CheckpointError", "build_stage1_s110_evidence", "load_stage1_s110_fixture",
    "replay_stage1_s110_evidence", "validate_parameterized_handoff", "validate_stage1_s110_evidence",
]
