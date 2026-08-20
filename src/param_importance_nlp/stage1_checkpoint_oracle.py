"""Independent FP64 oracle for the S1.10 checkpoint/resume fixture.

This file deliberately imports no project production module.  It owns the
closed-form microbatch, optimizer, accumulator and RNG expectations used to
audit a serialized S1.10 trace; production checkpoint publication lives in
``stage1_checkpoint_resume.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch


FIXTURE_ID = "stage1-s110-checkpoint-fixture-v1"
FIXTURE_SCHEMA = "stage1-s1-10-checkpoint-fixture-v1"
ORACLE_SCHEMA = "stage1-s1-10-oracle-bundle-v1"
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
# The oracle owns an independently written copy of the frozen input.  Do not
# import the production loader/constants here: drift must be caught by two
# separately implemented readers, even if the fixture self-hash is recomputed.
FROZEN_FIXTURE_HASH = "65ca095826bf01151ba4d7f53240fde315c5c19b47b5342f24a7c6d27356aad8"
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


class Stage1CheckpointOracleError(RuntimeError):
    """The frozen fixture or an oracle-only calculation is invalid."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Stage1CheckpointOracleError("S1_10_FIXTURE_DUPLICATE_KEY")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise Stage1CheckpointOracleError(f"S1_10_FIXTURE_NONFINITE:{token}")


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_JSON_VALUE_INVALID") from error


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _fixture_path(source_root: str | Path) -> Path:
    return Path(source_root) / "fixtures" / "stage1" / "stage1-s110-checkpoint-fixture-v1.json"


def load_stage1_s110_fixture(source_root: str | Path) -> dict[str, Any]:
    """Read the immutable fixture without calling the production parser."""

    try:
        payload = _fixture_path(source_root).read_bytes()
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
        if _canonical_json_bytes(value) != payload:
            raise Stage1CheckpointOracleError("S1_10_FIXTURE_NOT_CANONICAL")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_UNREADABLE") from error
    expected = {
        "checkpoint_boundaries", "contract", "fixture_hash", "fixture_id",
        "initial_state", "samples", "schema_version", "seed_plan", "tolerances",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_FIELDS_INVALID")
    body = dict(value)
    declared = body.pop("fixture_hash")
    if (
        value.get("fixture_id") != FIXTURE_ID
        or value.get("schema_version") != FIXTURE_SCHEMA
        or declared != FROZEN_FIXTURE_HASH
        or _canonical_hash(body) != FROZEN_FIXTURE_HASH
        or body != FROZEN_FIXTURE_BODY
    ):
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_HASH_INVALID")
    if value.get("checkpoint_boundaries") != [2, 3]:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_BOUNDARIES_INVALID")
    contract = value.get("contract")
    if not isinstance(contract, Mapping) or tuple(contract.get("public_accumulator_fields", ())) != ACCUMULATOR_FIELDS:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_ACCUMULATOR_CONTRACT_INVALID")
    samples = value.get("samples")
    if not isinstance(samples, list) or len(samples) != 6:
        raise Stage1CheckpointOracleError("S1_10_FIXTURE_SAMPLE_COUNT_INVALID")
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping) or set(sample) - {"sample_id", "token_sha256", "effective_tokens", "loss", "micro_gradients", "skip_reason"}:
            raise Stage1CheckpointOracleError(f"S1_10_FIXTURE_SAMPLE_FIELDS_INVALID:{index}")
        gradients = sample.get("micro_gradients")
        if not isinstance(gradients, list) or len(gradients) != 2 or any(
            not isinstance(row, list) or len(row) != 2 or any(not isinstance(item, (int, float)) for item in row)
            for row in gradients
        ):
            raise Stage1CheckpointOracleError(f"S1_10_FIXTURE_MICRO_GRADIENTS_INVALID:{index}")
    return value


def _new_rng_streams(fixture: Mapping[str, Any]) -> dict[str, Any]:
    seeds = fixture["seed_plan"]
    if not isinstance(seeds, Mapping):
        raise Stage1CheckpointOracleError("S1_10_SEED_PLAN_INVALID")
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


def _rng_probe(streams: Mapping[str, Any]) -> dict[str, float]:
    """Compare the next RNG sequence from copies, never consuming state."""

    python_stream = random.Random()
    python_stream.setstate(streams["python"].getstate())
    numpy_stream = np.random.RandomState()
    numpy_stream.set_state(streams["numpy"].get_state())
    torch_stream = torch.Generator(device="cpu")
    torch_stream.set_state(streams["torch"].get_state())
    sampler_stream = torch.Generator(device="cpu")
    sampler_stream.set_state(streams["sampler"].get_state())
    return {
        "numpy": float(numpy_stream.random_sample()),
        "python": float(python_stream.random()),
        "sampler": float(torch.rand((), generator=sampler_stream).item()),
        "torch": float(torch.rand((), generator=torch_stream).item()),
    }


def _consume_rng(streams: Mapping[str, Any]) -> None:
    streams["python"].random()
    streams["numpy"].random_sample()
    torch.rand((), generator=streams["torch"])
    torch.rand((), generator=streams["sampler"])


def _new_state(fixture: Mapping[str, Any]) -> dict[str, Any]:
    initial = fixture["initial_state"]
    if not isinstance(initial, Mapping):
        raise Stage1CheckpointOracleError("S1_10_INITIAL_STATE_INVALID")
    zero = [0.0, 0.0]
    return {
        "attempt": 0,
        "successful_step": 0,
        "skip_count": 0,
        "cursor": 0,
        "parameters": [float(item) for item in initial["parameters"]],
        "velocity": [float(item) for item in initial["velocity"]],
        "learning_rate": float(initial["learning_rate"]),
        "scale": float(initial["scale"]),
        "growth_tracker": int(initial["growth_tracker"]),
        "accumulators": {name: list(zero) for name in ACCUMULATOR_FIELDS},
    }


def _state_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "accumulators": copy.deepcopy(state["accumulators"]),
        "attempt": state["attempt"],
        "cursor": state["cursor"],
        "growth_tracker": state["growth_tracker"],
        "learning_rate": state["learning_rate"],
        "parameters": list(state["parameters"]),
        "scale": state["scale"],
        "skip_count": state["skip_count"],
        "successful_step": state["successful_step"],
        "velocity": list(state["velocity"]),
    }


def _step(state: dict[str, Any], streams: Mapping[str, Any], sample: Mapping[str, Any]) -> dict[str, Any]:
    if int(state["cursor"]) >= 6:
        raise Stage1CheckpointOracleError("S1_10_CURSOR_OUT_OF_RANGE")
    gradients = [[float(item) for item in row] for row in sample["micro_gradients"]]
    s1 = [gradients[0][index] + gradients[1][index] for index in range(2)]
    s2 = [gradients[0][index] ** 2 + gradients[1][index] ** 2 for index in range(2)]
    mean = [item / 2.0 for item in s1]
    raw = [item * item for item in mean]
    u = [(s1[index] ** 2 - s2[index]) / 2.0 for index in range(2)]
    _consume_rng(streams)
    state["attempt"] += 1
    state["cursor"] += 1
    skipped = sample.get("skip_reason") is not None
    if skipped:
        state["skip_count"] += 1
        state["scale"] *= 0.5
        state["growth_tracker"] = 0
        delta = [0.0, 0.0]
        status = "SKIPPED"
    else:
        delta: list[float] = []
        for index in range(2):
            velocity = 0.5 * float(state["velocity"][index]) + mean[index]
            state["velocity"][index] = velocity
            item_delta = -float(state["learning_rate"]) * velocity
            state["parameters"][index] += item_delta
            delta.append(item_delta)
        state["successful_step"] += 1
        state["growth_tracker"] += 1
        for index in range(2):
            accumulators = state["accumulators"]
            signed = u[index]
            positive = max(signed, 0.0)
            negative = max(-signed, 0.0)
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
    row = {
        "attempt": state["attempt"],
        "clip_factor": 1.0,
        "cursor": state["cursor"],
        "data_delta": delta,
        "effective_tokens": int(sample["effective_tokens"]),
        "growth_tracker": state["growth_tracker"],
        "learning_rate": state["learning_rate"],
        "loss": sample.get("loss"),
        "mean_gradient": mean,
        "next_rng_probe": _rng_probe(streams),
        "parameters": list(state["parameters"]),
        "raw": raw,
        "rng_streams_consumed": state["attempt"],
        "s1": s1,
        "s2": s2,
        "sample_id": sample["sample_id"],
        "scale": state["scale"],
        "skip_count": state["skip_count"],
        "status": status,
        "successful_step": state["successful_step"],
        "token_sha256": sample["token_sha256"],
        "u": u,
        "state": _state_projection(state),
    }
    return row


def build_stage1_s110_oracle(source_root: str | Path) -> dict[str, Any]:
    fixture = load_stage1_s110_fixture(source_root)
    state = _new_state(fixture)
    streams = _new_rng_streams(fixture)
    trace = [_step(state, streams, sample) for sample in fixture["samples"]]
    result: dict[str, Any] = {
        "schema_version": ORACLE_SCHEMA,
        "fixture_id": FIXTURE_ID,
        "fixture_hash": fixture["fixture_hash"],
        "independent_implementation": True,
        "continuous_trace": trace,
        "final_state": _state_projection(state),
        "checkpoint_boundaries": list(fixture["checkpoint_boundaries"]),
    }
    result["oracle_hash"] = _canonical_hash(result)
    return result


__all__ = [
    "ACCUMULATOR_FIELDS",
    "FIXTURE_ID",
    "FIXTURE_SCHEMA",
    "ORACLE_SCHEMA",
    "Stage1CheckpointOracleError",
    "build_stage1_s110_oracle",
    "load_stage1_s110_fixture",
]
