"""S1.10 local contracts: checkpoint authority, replay, and resume rollback."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import pytest
import torch

from param_importance_nlp.core.losses import LossBatch
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.providers import DeterministicBatchCursor, TorchModelAdapter, TrainingMicrobatch
from param_importance_nlp.runtime import CheckpointStore, JsonlEventSink
from param_importance_nlp.runtime.checkpoint_group import GROUP_COMMIT_SCHEMA, GROUP_COMMIT_SCHEMA_V2, CheckpointGroupStore
from param_importance_nlp.runtime.checkpoint_group import checkpoint_state_sha256
from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec
from param_importance_nlp.stage1_checkpoint_oracle import build_stage1_s110_oracle, load_stage1_s110_fixture
from param_importance_nlp.stage1_checkpoint_resume import (
    REQUIREMENT_KEYS,
    Stage1CheckpointError,
    TASK_ID,
    build_stage1_s110_evidence,
    replay_stage1_s110_evidence,
    validate_parameterized_handoff,
    validate_stage1_s110_evidence,
    load_stage1_s110_fixture as load_production_s110_fixture,
)


ROOT = Path(__file__).resolve().parents[1]


def _schema_handoff(which: str) -> dict[str, object]:
    """A complete frozen S1.8/S1.9 handoff shape for schema-only roles."""

    if which == "s1_8":
        return {
            "index_ref": "published/s1-8-index.json",
            "index_sha256": "a" * 64,
            "index_artifact_hash": "b" * 64,
            "producer_commit": "0" * 40,
            "gate_artifact_hash": "c" * 64,
            "role_sha256": {
                "fixture_manifest": "a" * 64,
                "ddp_report": "b" * 64,
                "array_bundle": "c" * 64,
                "comparison_table": "d" * 64,
                "gate_record": "e" * 64,
            },
            "validation_sha256": "f" * 64,
            "source_map_sha256": "f" * 64,
            "source_map_entries": 61,
            "reproduction_role_sha256": {
                "prelease_gpu_quiescence": "f" * 64,
                "post_worker_gpu_quiescence": "f" * 64,
                "post_release_gpu_quiescence": "f" * 64,
                "reacquire_preflight_gpu_quiescence": "f" * 64,
            },
            "reproduction_role_set_sha256": "f" * 64,
            "reproduction_role_count": 84,
            "schema_version": "stage1-s1-8-formalization-index-v8",
            "task_id": "stage1.08_ddp_and_gradient_accumulation",
            "gate_id": "G1-DDP",
        }
    return {
        "index_ref": "published/s1-9-index.json",
        "index_sha256": "1" * 64,
        "index_artifact_hash": "2" * 64,
        "producer_commit": "0" * 40,
        "gate_artifact_hash": "3" * 64,
        "role_sha256": {
            "numeric_report": "1" * 64,
            "oracle_bundle": "2" * 64,
            "trace_bundle": "3" * 64,
            "comparison_table": "4" * 64,
            "gate_record": "5" * 64,
        },
        "validation_sha256": "6" * 64,
        "source_map_sha256": "6" * 64,
        "source_map_entries": 34,
        "reproduction_role_sha256": {
            "upstream_compatibility": "6" * 64,
            "prelease_gpu": "6" * 64,
            "post_worker_quiescence": "6" * 64,
        },
        "reproduction_role_set_sha256": "6" * 64,
        "reproduction_role_count": 28,
        "schema_version": "stage1-s1-9-formalization-index-v8",
        "task_id": "stage1.09_precision_clipping_and_optimizer_boundaries",
        "gate_id": "G1-NUMERIC",
    }


class _ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Keep a one-element vector rather than a 0-D scalar so the generic
        # tensor-state hash exercises the same byte-view path as real weights.
        self.weight = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))


class _InjectedNonfiniteGradient(torch.autograd.Function):
    """Keep loss wire-finite while exercising TrainingEngine's grad preflight."""

    @staticmethod
    def forward(ctx: object, value: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return value * 0.0

    @staticmethod
    def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor]:  # type: ignore[override]
        return (torch.full_like(gradient, float("inf")),)


class _ScalarAdapter:
    task_type = "s1-10-resume-rollback"

    def __init__(self, module: _ScalarModel) -> None:
        self._module = module

    @property
    def module(self) -> _ScalarModel:
        return self._module

    def loss(self, microbatch: TrainingMicrobatch) -> LossBatch:
        values = microbatch.payload["gradient"].to(self._module.weight.device)
        loss = (self._module.weight * values.sum()).sum()
        if "inject_nonfinite" in microbatch.payload:
            loss = loss + _InjectedNonfiniteGradient.apply(self._module.weight).sum()
        return LossBatch(loss, int(values.numel()), "fixture")


def _steps() -> tuple[tuple[TrainingMicrobatch, ...], ...]:
    return tuple(
        (
            TrainingMicrobatch(
                f"s110-{step}-a",
                {"gradient": torch.tensor([0.25 + step * 0.1], dtype=torch.float32)},
                (f"s110-{step}-a",),
            ),
            TrainingMicrobatch(
                f"s110-{step}-b",
                {"gradient": torch.tensor([0.5 - step * 0.05], dtype=torch.float32)},
                (f"s110-{step}-b",),
            ),
        )
        for step in range(4)
    )


def _engine(
    *,
    store: CheckpointStore | None = None,
    steps: tuple[tuple[TrainingMicrobatch, ...], ...] | None = None,
    max_steps: int = 4,
    max_attempts: int = 4,
    event_sink: object | None = None,
) -> TrainingEngine:
    model = _ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    spec = TrainingRunSpec(
        "s110-runtime-resume",
        "local_fixture",
        max_steps=max_steps,
        max_attempts=max_attempts,
        importance_enabled=True,
        estimator_name="u",
        checkpoint_every_steps=1,
        weights_exogenous=True,
        common_mean_assumption=True,
    )
    return TrainingEngine(
        spec=spec,
        model=_ScalarAdapter(model),
        optimizer=optimizer,
        scheduler=scheduler,
        cursor=DeterministicBatchCursor(_steps() if steps is None else steps),
        checkpoint_store=store,
        event_sink=event_sink,  # type: ignore[arg-type]
    )


def _engine_identity(engine: TrainingEngine) -> str:
    state = engine.capture_observer_state()
    return checkpoint_state_sha256(
        {
            "bridge_aliases_optimizer": engine.bridge.optimizer is engine.optimizer,
            "bridge_parameter_names": sorted(engine.bridge.named_parameters),
            "boundary_trace": [dict(item) for item in engine.boundary_trace],
            "checkpoint_ids": list(engine._checkpoint_ids),  # noqa: SLF001 - production rollback boundary
            "engine_state": engine.state.to_dict(),
            "importance_points": [point.to_dict() for point in engine._importance_points],  # noqa: SLF001
            "importance_state": None if engine.tracker is None else engine.tracker.accumulator.state_dict(),
            "records": [record.to_dict() for record in engine._records],  # noqa: SLF001
            "state": state,
        }
    )


def _published_state(store: CheckpointStore) -> tuple[dict[str, object], dict[str, object]]:
    source = _engine(store=store)
    assert source.run(until_step=1).status == "PAUSED"
    checkpoint_id = store.discover()[0].checkpoint_id
    state, _ = store.load(checkpoint_id)
    assert isinstance(state, dict)
    metadata = {
        "run_spec_hash": source.spec.spec_hash,
        "registry_hash": source.registry.coordinate_registry_hash,
        "optimizer_contract_hash": source.registry.optimizer_contract_hash,
        "runtime_layout_hash": source.registry.runtime_layout_hash,
        "world_size": 1,
    }
    return state, metadata


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_handoff_fixture(
    root: Path,
    *,
    task_id: str,
    gate_id: str,
    schema_version: str,
) -> tuple[str, dict[str, str]]:
    """Publish an index-shaped synthetic upstream fixture with sibling refs."""

    publication = root / "published"
    publication.mkdir()
    source_role = (
        "ddp_report"
        if task_id == "stage1.08_ddp_and_gradient_accumulation"
        else "numeric_report"
    )
    source_count, reproduction_count = (61, 84) if task_id == "stage1.08_ddp_and_gradient_accumulation" else (34, 28)
    source_map = {f"src/frozen-producer-{index}.py": "b" * 64 for index in range(source_count)}
    gate = {
        "schema_version": "synthetic-gate-v1",
        "status": "PASS",
        "task_id": task_id,
        "gate_id": gate_id,
    }
    gate["artifact_hash"] = canonical_json_hash(gate)
    role = {
        "schema_version": "synthetic-role-v1",
        "status": "PASS",
        "implementation_source_sha256": source_map,
    }
    replay_hash_field = (
        "artifact_hash"
        if task_id == "stage1.08_ddp_and_gradient_accumulation"
        else "replay_hash"
    )
    replay = {"schema_version": "synthetic-replay-v1", "status": "PASS"}
    if task_id == "stage1.09_precision_clipping_and_optimizer_boundaries":
        replay["source_gate_artifact_hash"] = gate["artifact_hash"]
    replay[replay_hash_field] = canonical_json_hash(replay)
    role_refs = {"gate_record": "gate.json", source_role: "role.json"}
    # Validation binds the actual immutable sibling identities rather than a
    # generic placeholder.  This mirrors the S1.8/S1.9 handoff contract.
    validation = {
        "schema_version": "synthetic-validation-v1",
        "status": "PASS",
        "task_id": task_id,
        "gate_id": gate_id,
        "producer_commit": "a" * 40,
        "consumer_commit": "a" * 40,
    }
    for name, value in {
        "gate.json": gate,
        "role.json": role,
        "replay.json": replay,
    }.items():
        write_canonical_json(publication / name, value)
    role_sha256 = {name: _file_sha256(publication / ref) for name, ref in role_refs.items()}
    validation["role_sha256"] = role_sha256
    validation["artifact_hash"] = canonical_json_hash(validation)
    write_canonical_json(publication / "validation.json", validation)
    required_reproduction = (
        ("prelease_gpu_quiescence", "post_worker_gpu_quiescence", "post_release_gpu_quiescence", "reacquire_preflight_gpu_quiescence")
        if task_id == "stage1.08_ddp_and_gradient_accumulation"
        else ("upstream_compatibility", "prelease_gpu", "post_worker_quiescence")
    )
    reproduction_refs = {role: f"{role}.json" for role in required_reproduction}
    for index in range(reproduction_count - len(reproduction_refs)):
        reproduction_refs[f"auxiliary_{index}"] = f"auxiliary-{index}.json"
    schema_versions = (
        {
            "prelease_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "post_worker_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "post_release_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
            "reacquire_preflight_gpu_quiescence": "stage1-s1-8-gpu-quiescence-v4",
        }
        if task_id == "stage1.08_ddp_and_gradient_accumulation"
        else {
            "upstream_compatibility": "stage1-s1-9-upstream-compatibility-v7",
            "prelease_gpu": "stage1-s1-9-gpu-prelease-v3",
            "post_worker_quiescence": "stage1-s1-9-gpu-quiescence-v3",
        }
    )
    for role_name, filename in reproduction_refs.items():
        value = {"schema_version": schema_versions.get(role_name, "synthetic-reproduction-v1"), "status": "PASS"}
        value["artifact_hash"] = canonical_json_hash(value)
        write_canonical_json(publication / filename, value)
    reproduction_sha256 = {name: _file_sha256(publication / ref) for name, ref in reproduction_refs.items()}
    index = {
        "schema_version": schema_version,
        "status": "PASS",
        "task_id": task_id,
        "gate_id": gate_id,
        "generator_git_commit": "a" * 40,
        "consumer_git_commit": "a" * 40,
        "next_task_ids": [TASK_ID],
        "role_refs": role_refs,
        "role_sha256": role_sha256,
        "reproduction_role_refs": reproduction_refs,
        "reproduction_role_sha256": reproduction_sha256,
        "gate_artifact_hash": gate["artifact_hash"],
        "validation_ref": "validation.json",
        "validation_sha256": _file_sha256(publication / "validation.json"),
        "replay_ref": "replay.json",
        "replay_sha256": _file_sha256(publication / "replay.json"),
    }
    if task_id == "stage1.08_ddp_and_gradient_accumulation":
        index["implementation_source_sha256"] = source_map
    index["artifact_hash"] = canonical_json_hash(index)
    index_path = publication / "index.json"
    write_canonical_json(index_path, index)
    return "published/index.json", {
        "index_sha256": _file_sha256(index_path),
        "index_artifact_hash": str(index["artifact_hash"]),
        "gate_artifact_hash": str(gate["artifact_hash"]),
        "producer_commit": "a" * 40,
        "schema_version": schema_version,
        "task_id": task_id,
        "gate_id": gate_id,
    }


def test_s110_fixture_oracle_evidence_and_independent_replay(tmp_path: Path) -> None:
    fixture = load_stage1_s110_fixture(ROOT)
    oracle = build_stage1_s110_oracle(ROOT)
    assert oracle["fixture_hash"] == fixture["fixture_hash"]
    assert len(oracle["continuous_trace"]) == 6
    upstream = {"s1_8": _schema_handoff("s1_8"), "s1_9": _schema_handoff("s1_9")}
    evidence = build_stage1_s110_evidence(ROOT, producer_commit="0" * 40, scope="local_cpu_fixture", upstream_evidence=upstream, scratch_root=tmp_path / "first")
    hashes = validate_stage1_s110_evidence(evidence, source_root=ROOT)
    assert evidence["gate_record"]["status"] == "NOT_RUN"
    assert tuple(evidence["gate_record"]["requirements"]) == REQUIREMENT_KEYS
    assert evidence["gate_record"]["requirements"]["pre_skip_continuous_resume_bitwise"] is True
    assert evidence["gate_record"]["requirements"]["post_skip_continuous_resume_bitwise"] is True
    assert evidence["gate_record"]["requirements"]["four_rank_resume_observation_required_for_formal_gate"] is False
    replay = replay_stage1_s110_evidence(evidence, source_root=ROOT, scratch_root=tmp_path / "replay")
    assert replay["status"] == "PASS"
    assert replay["source_gate_artifact_hash"] == hashes["gate_artifact_hash"]
    formalizer_spec = importlib.util.spec_from_file_location(
        "s110_formalizer_schema_test", ROOT / "ops" / "stage1" / "formalize_s1_10.py"
    )
    assert formalizer_spec is not None and formalizer_spec.loader is not None
    formalizer = importlib.util.module_from_spec(formalizer_spec)
    formalizer_spec.loader.exec_module(formalizer)
    formalizer._schema_validate(ROOT, {**evidence, "replay": replay})
    role_sha256 = {
        "resume_report": "1" * 64,
        "oracle_bundle": "2" * 64,
        "trace_bundle": "3" * 64,
        "comparison_table": "4" * 64,
        "artifact_manifest": "5" * 64,
        "gate_record": "6" * 64,
    }
    formal_validation = {
        "schema_version": "stage1-s1-10-validation-v2", "status": "PASS",
        "gate_id": "G1-RESUME", "task_id": TASK_ID,
        "execution_scope": "formal_server_single_and_four_rank_resume",
        "fixture_id": "stage1-s110-checkpoint-fixture-v1",
        "producer_commit": "0" * 40, "consumer_commit": "0" * 40,
        "upstream": upstream,
        "direct_checks": {
            "all_requirements_true": True,
            "formal_observation_hash_valid": True,
            "parameterized_s1_8_handoff": True,
            "parameterized_s1_9_handoff": True,
            "replay_matches": True,
        }, "role_sha256": role_sha256,
        "replay_sha256": "7" * 64, "replay_hash": replay["replay_hash"],
    }
    formal_validation["artifact_hash"] = canonical_json_hash(formal_validation)
    formal_upstream = upstream
    formal_index = {
        "schema_version": "stage1-s1-10-formalization-index-v2", "status": "PASS",
        "gate_id": "G1-RESUME", "task_id": TASK_ID,
        "fixture_id": "stage1-s110-checkpoint-fixture-v1",
        "generator_git_commit": "0" * 40, "consumer_git_commit": "0" * 40,
        "git_branch": "fixture", "checked_at": "1970-01-01T00:00:00Z",
        "upstream": formal_upstream,
        "role_refs": {"resume_report": "resume-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "artifact_manifest": "artifact-manifest.json", "gate_record": "g1-resume-record.json"},
        "role_sha256": role_sha256,
        "chart_csv_sha256": {"resume-errors.csv": "8" * 64, "state-timeline.csv": "9" * 64},
        "chart_svg_sha256": {"resume-errors.svg": "a" * 64, "state-timeline.svg": "b" * 64},
        "formal_observation_ref": "formal-observation.json", "formal_observation_sha256": "c" * 64,
        "formal_observation_artifact_hash": "d" * 64, "formal_run_token_sha256": "d" * 64,
        "formal_single_report_ref": "formal-single-report.json", "formal_single_report_sha256": "e" * 64,
        "formal_four_rank_report_ref": "formal-four-rank-report.json", "formal_four_rank_report_sha256": "f" * 64,
        "gate_artifact_hash": evidence["gate_record"]["artifact_hash"],
        "validation_ref": "validation.json", "validation_sha256": "1" * 64,
        "replay_ref": "replay-validation.json", "replay_sha256": "7" * 64,
        "replay_hash": replay["replay_hash"],
        "next_task_ids": ["stage1.11_reporting_and_exit_gate"],
    }
    formal_index["artifact_hash"] = canonical_json_hash(formal_index)
    formalizer._schema_validate(ROOT, {"validation": formal_validation, "index": formal_index, "replay": replay})
    malformed_validation = deepcopy(formal_validation)
    malformed_validation["direct_checks"]["unbound_check"] = True
    with pytest.raises(formalizer.Stage1S110FormalError):
        formalizer._schema_validate(ROOT, {"validation": malformed_validation})
    malformed_validation = deepcopy(formal_validation)
    del malformed_validation["direct_checks"]["replay_matches"]
    with pytest.raises(formalizer.Stage1S110FormalError):
        formalizer._schema_validate(ROOT, {"validation": malformed_validation})


def test_s110_schema_closure_and_nested_negative_contracts(tmp_path: Path) -> None:
    """Every nested S1.10 object is closed and evidence roles reject drift."""

    for path in sorted((ROOT / "schemas" / "stage1").glob("s1-10-*.json")):
        schema = json.loads(path.read_text(encoding="utf-8"))
        open_objects: list[str] = []

        def walk(value: object, location: str = "$") -> None:
            if isinstance(value, dict):
                if value.get("type") == "object" and value.get("additionalProperties") is not False:
                    open_objects.append(location)
                for key, child in value.items():
                    walk(child, f"{location}.{key}")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{location}[{index}]")

        walk(schema)
        assert open_objects == [], f"{path.name}: {open_objects}"

    evidence = build_stage1_s110_evidence(
        ROOT,
        producer_commit="0" * 40,
        scope="local_cpu_fixture",
        upstream_evidence={"s1_8": _schema_handoff("s1_8"), "s1_9": _schema_handoff("s1_9")},
        scratch_root=tmp_path / "source",
    )
    formalizer_spec = importlib.util.spec_from_file_location(
        "s110_schema_closure_test", ROOT / "ops" / "stage1" / "formalize_s1_10.py"
    )
    assert formalizer_spec is not None and formalizer_spec.loader is not None
    formalizer = importlib.util.module_from_spec(formalizer_spec)
    formalizer_spec.loader.exec_module(formalizer)
    formalizer._schema_validate(ROOT, evidence)
    resume_schema = json.loads(
        (ROOT / "schemas" / "stage1" / "s1-10-resume-report-v2.json").read_text(encoding="utf-8")
    )
    assert set(evidence["resume_report"]["implementation_source_sha256"]) == set(
        resume_schema["$defs"]["source_map"]["required"]
    )
    assert {
        "schemas/stage1/s1-10-resume-report-v2.json",
        "schemas/stage1/s1-10-validation-v2.json",
        "schemas/stage1/s1-10-formalization-index-v2.json",
    } <= set(evidence["resume_report"]["implementation_source_sha256"])

    mutations: list[tuple[str, object]] = []
    extra = deepcopy(evidence["trace_bundle"])
    extra["continuous"]["trace"][0]["state"]["unexpected"] = True
    mutations.append(("extra", extra))
    missing = deepcopy(evidence["trace_bundle"])
    del missing["continuous"]["trace"][0]["state"]["accumulators"]["raw"]
    mutations.append(("missing", missing))
    wrong_type = deepcopy(evidence["trace_bundle"])
    wrong_type["continuous"]["trace"][0]["parameters"][0] = "not-a-number"
    mutations.append(("type", wrong_type))
    wrong_cardinality = deepcopy(evidence["trace_bundle"])
    wrong_cardinality["resume_cases"]["3"]["post_trace"].pop()
    mutations.append(("cardinality", wrong_cardinality))
    bad_handoff = deepcopy(evidence["resume_report"])
    bad_handoff["upstream"]["s1_8"]["unbound"] = True
    mutations.append(("upstream", bad_handoff))
    source_extra = deepcopy(evidence["resume_report"])
    source_extra["implementation_source_sha256"]["src/unbound.py"] = "0" * 64
    mutations.append(("source_extra", source_extra))
    source_missing = deepcopy(evidence["resume_report"])
    del source_missing["implementation_source_sha256"]["src/param_importance_nlp/runtime/training.py"]
    mutations.append(("source_missing", source_missing))
    corruption_wrong_key = deepcopy(evidence["trace_bundle"])
    corruption_wrong_key["corruption_rejections"]["unbound_rejection"] = corruption_wrong_key["corruption_rejections"].pop("file_hash_mismatch")
    mutations.append(("corruption_wrong_key", corruption_wrong_key))
    corruption_cardinality = deepcopy(evidence["trace_bundle"])
    del corruption_cardinality["corruption_rejections"]["file_hash_mismatch"]
    mutations.append(("corruption_cardinality", corruption_cardinality))
    for label, value in mutations:
        role = "resume_report" if label in {"upstream", "source_extra", "source_missing"} else "trace_bundle"
        with pytest.raises(formalizer.Stage1S110FormalError):
            formalizer._schema_validate(ROOT, {role: value})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("tolerances", "distributed_atol"), 0.25),
        (("seed_plan", "torch"), 17),
        (("contract", "public_accumulator_fields", 0), "drifted_accumulator"),
        (("samples", 0, "effective_tokens"), 9),
        (("samples", 1, "micro_gradients", 0, 0), 0.875),
        (("samples", 2, "token_sha256"), "0" * 64),
    ],
)
def test_s110_fixture_joint_rehash_cannot_relax_frozen_inputs(
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement: object,
) -> None:
    """Every reader and the schema reject a self-consistent but altered fixture."""

    fixture = load_production_s110_fixture(ROOT)
    cursor: object = fixture
    for item in path[:-1]:
        cursor = cursor[item]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]
    body = {key: value for key, value in fixture.items() if key != "fixture_hash"}
    fixture["fixture_hash"] = canonical_json_hash(body)
    root = tmp_path / "source"
    target = root / "fixtures" / "stage1" / "stage1-s110-checkpoint-fixture-v1.json"
    target.parent.mkdir(parents=True)
    write_canonical_json(target, fixture)

    with pytest.raises(Stage1CheckpointError, match="FIXTURE_HASH_INVALID"):
        load_production_s110_fixture(root)
    with pytest.raises(Exception, match="S1_10_FIXTURE_HASH_INVALID"):
        load_stage1_s110_fixture(root)

    formalizer_spec = importlib.util.spec_from_file_location(
        "s110_fixture_schema_tamper", ROOT / "ops" / "stage1" / "formalize_s1_10.py"
    )
    assert formalizer_spec is not None and formalizer_spec.loader is not None
    formalizer = importlib.util.module_from_spec(formalizer_spec)
    formalizer_spec.loader.exec_module(formalizer)
    with pytest.raises(formalizer.Stage1S110FormalError):
        formalizer._schema_validate(ROOT, {"checkpoint_fixture": fixture})


@pytest.mark.parametrize(
    "payload",
    [
        b'{"fixture_id":"first","fixture_id":"second"}\n',
        b'{"fixture_id":NaN}\n',
        b'{ "fixture_id":"stage1-s110-checkpoint-fixture-v1" }\n',
    ],
)
def test_s110_fixture_readers_fail_closed_on_ambiguous_or_noncanonical_json(
    tmp_path: Path, payload: bytes
) -> None:
    root = tmp_path / "source"
    target = root / "fixtures" / "stage1" / "stage1-s110-checkpoint-fixture-v1.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(payload)
    with pytest.raises(Stage1CheckpointError):
        load_production_s110_fixture(root)
    with pytest.raises(Exception):
        load_stage1_s110_fixture(root)


def test_s110_parameterized_handoff_rejects_missing_or_unpinned_binding(tmp_path: Path) -> None:
    with pytest.raises(Stage1CheckpointError, match="HANDOFF_BINDING_REQUIRED"):
        validate_parameterized_handoff(
            tmp_path,
            "missing/index.json",
            expected_binding={},
            expected_task_id="stage1.08_ddp_and_gradient_accumulation",
            expected_gate_id="G1-DDP",
        )
    reference, stale_binding = _write_handoff_fixture(
        tmp_path,
        task_id="stage1.08_ddp_and_gradient_accumulation",
        gate_id="G1-DDP",
        schema_version="stage1-s1-8-formalization-index-v7",
    )
    with pytest.raises(Stage1CheckpointError, match="FINAL_SCHEMA_REQUIRED"):
        validate_parameterized_handoff(
            tmp_path,
            reference,
            expected_binding=stale_binding,
            expected_task_id="stage1.08_ddp_and_gradient_accumulation",
            expected_gate_id="G1-DDP",
        )


@pytest.mark.parametrize(
    ("task_id", "gate_id", "schema_version"),
    [
        ("stage1.08_ddp_and_gradient_accumulation", "G1-DDP", "stage1-s1-8-formalization-index-v8"),
        ("stage1.09_precision_clipping_and_optimizer_boundaries", "G1-NUMERIC", "stage1-s1-9-formalization-index-v8"),
    ],
)
def test_s110_parameterized_handoff_loads_index_siblings_and_rejects_escape(
    tmp_path: Path, task_id: str, gate_id: str, schema_version: str,
) -> None:
    reference, binding = _write_handoff_fixture(
        tmp_path,
        task_id=task_id,
        gate_id=gate_id,
        schema_version=schema_version,
    )
    loaded = validate_parameterized_handoff(
        tmp_path,
        reference,
        expected_binding=binding,
        expected_task_id=task_id,
        expected_gate_id=gate_id,
    )
    source_role = "ddp_report" if gate_id == "G1-DDP" else "numeric_report"
    assert loaded["role_sha256"] == {
        "gate_record": _file_sha256(tmp_path / "published" / "gate.json"),
        source_role: _file_sha256(tmp_path / "published" / "role.json"),
    }
    index_path = tmp_path / "published" / "index.json"
    index = {
        "schema_version": binding["schema_version"],
        "status": "PASS",
        "task_id": task_id,
        "gate_id": gate_id,
        "generator_git_commit": "a" * 40,
        "consumer_git_commit": "a" * 40,
        "next_task_ids": [TASK_ID],
        "role_refs": {"gate_record": "../escape.json"},
        "role_sha256": {"gate_record": "0" * 64},
        "gate_artifact_hash": binding["gate_artifact_hash"],
        "validation_ref": "validation.json",
        "validation_sha256": _file_sha256(tmp_path / "published" / "validation.json"),
        "replay_ref": "replay.json",
        "replay_sha256": _file_sha256(tmp_path / "published" / "replay.json"),
    }
    index["artifact_hash"] = canonical_json_hash(index)
    write_canonical_json(index_path, index)
    binding = dict(binding)
    binding["index_sha256"] = _file_sha256(index_path)
    binding["index_artifact_hash"] = str(index["artifact_hash"])
    with pytest.raises(Stage1CheckpointError, match="INDEX_MEMBER_REFERENCE_ESCAPE"):
        validate_parameterized_handoff(
            tmp_path,
            reference,
            expected_binding=binding,
            expected_task_id=task_id,
            expected_gate_id=gate_id,
        )


def test_s110_final_handoff_rejects_short_or_wrong_versioned_reproduction_closure(tmp_path: Path) -> None:
    reference, binding = _write_handoff_fixture(
        tmp_path,
        task_id="stage1.08_ddp_and_gradient_accumulation",
        gate_id="G1-DDP",
        schema_version="stage1-s1-8-formalization-index-v8",
    )
    index_path = tmp_path / "published" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    dropped = next(name for name in index["reproduction_role_refs"] if name.startswith("auxiliary_"))
    del index["reproduction_role_refs"][dropped]
    del index["reproduction_role_sha256"][dropped]
    index["artifact_hash"] = canonical_json_hash({key: value for key, value in index.items() if key != "artifact_hash"})
    write_canonical_json(index_path, index)
    binding = dict(binding)
    binding["index_sha256"] = _file_sha256(index_path)
    binding["index_artifact_hash"] = str(index["artifact_hash"])
    with pytest.raises(Stage1CheckpointError, match="CLOSURE_CARDINALITY_INVALID"):
        validate_parameterized_handoff(
            tmp_path, reference, expected_binding=binding,
            expected_task_id="stage1.08_ddp_and_gradient_accumulation", expected_gate_id="G1-DDP",
        )

    numeric_root = tmp_path / "numeric"
    numeric_root.mkdir()
    reference, binding = _write_handoff_fixture(
        numeric_root,
        task_id="stage1.09_precision_clipping_and_optimizer_boundaries",
        gate_id="G1-NUMERIC",
        schema_version="stage1-s1-9-formalization-index-v8",
    )
    index_path = numeric_root / "published" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    compatibility = numeric_root / "published" / str(index["reproduction_role_refs"]["upstream_compatibility"])
    value = json.loads(compatibility.read_text(encoding="utf-8"))
    value["schema_version"] = "stage1-s1-9-upstream-compatibility-v6"
    value["artifact_hash"] = canonical_json_hash({key: item for key, item in value.items() if key != "artifact_hash"})
    write_canonical_json(compatibility, value)
    index["reproduction_role_sha256"]["upstream_compatibility"] = _file_sha256(compatibility)
    index["artifact_hash"] = canonical_json_hash({key: item for key, item in index.items() if key != "artifact_hash"})
    write_canonical_json(index_path, index)
    binding = dict(binding)
    binding["index_sha256"] = _file_sha256(index_path)
    binding["index_artifact_hash"] = str(index["artifact_hash"])
    with pytest.raises(Stage1CheckpointError, match="REPRODUCTION_SCHEMA_INVALID"):
        validate_parameterized_handoff(
            numeric_root, reference, expected_binding=binding,
            expected_task_id="stage1.09_precision_clipping_and_optimizer_boundaries", expected_gate_id="G1-NUMERIC",
        )


def test_s110_formal_observation_requires_bound_pass_reports(tmp_path: Path) -> None:
    run_token_sha256 = "f" * 64
    for name, mode, world_size in (("single-report.json", "single", 1), ("four-rank-report.json", "four-rank", 4)):
        write_canonical_json(
            tmp_path / name,
            {
                "schema_version": "synthetic-worker-v1",
                "status": "PASS",
                "task_id": TASK_ID,
                "gate_id": "G1-RESUME",
                "fixture_id": "stage1-s110-checkpoint-fixture-v1",
                "execution_commit": "0" * 40,
                "run_token_sha256": run_token_sha256,
                "mode": mode,
                "phase": "resume",
                "world_size": world_size,
            },
        )
    observation = {
        "schema_version": "stage1-s1-10-formal-observation-v1",
        "status": "PASS",
        "task_id": TASK_ID,
        "gate_id": "G1-RESUME",
        "fixture_id": "stage1-s110-checkpoint-fixture-v1",
        "execution_commit": "0" * 40,
        "run_token_sha256": run_token_sha256,
        "single_process_resume": True,
        "four_rank_resume": True,
        "run_owned_resources_released": True,
        "single_cases": ["pre_skip", "post_skip"],
        "four_rank_cases": ["pre_skip", "post_skip"],
        "single_report_ref": "single-report.json",
        "single_report_sha256": _file_sha256(tmp_path / "single-report.json"),
        "four_rank_report_ref": "four-rank-report.json",
        "four_rank_report_sha256": _file_sha256(tmp_path / "four-rank-report.json"),
    }
    observation["artifact_hash"] = canonical_json_hash(observation)
    write_canonical_json(tmp_path / "formal-observation.json", observation)
    formalizer_spec = importlib.util.spec_from_file_location(
        "s110_formal_observation_test", ROOT / "ops" / "stage1" / "formalize_s1_10.py"
    )
    assert formalizer_spec is not None and formalizer_spec.loader is not None
    formalizer = importlib.util.module_from_spec(formalizer_spec)
    formalizer_spec.loader.exec_module(formalizer)
    assert formalizer._formal_observation(ROOT, tmp_path, "formal-observation.json", expected_commit="0" * 40)["status"] == "PASS"
    observation["execution_commit"] = "1" * 40
    observation["artifact_hash"] = canonical_json_hash({key: value for key, value in observation.items() if key != "artifact_hash"})
    write_canonical_json(tmp_path / "formal-observation.json", observation)
    with pytest.raises(formalizer.Stage1S110FormalError, match="FORMAL_OBSERVATION_NOT_PASS"):
        formalizer._formal_observation(ROOT, tmp_path, "formal-observation.json", expected_commit="0" * 40)
    observation["execution_commit"] = "0" * 40
    observation["single_report_ref"] = "four-rank-report.json"
    observation["single_report_sha256"] = _file_sha256(tmp_path / "four-rank-report.json")
    observation["artifact_hash"] = canonical_json_hash({key: value for key, value in observation.items() if key != "artifact_hash"})
    write_canonical_json(tmp_path / "formal-observation.json", observation)
    with pytest.raises(formalizer.Stage1S110FormalError, match="FORMAL_OBSERVATION_REPORT_NOT_PASS:single"):
        formalizer._formal_observation(ROOT, tmp_path, "formal-observation.json", expected_commit="0" * 40)


def test_training_resume_rejects_bad_checkpoint_before_mutating_active_engine(tmp_path: Path) -> None:
    state, metadata = _published_state(CheckpointStore(tmp_path / "source"))
    malformed = deepcopy(state)
    malformed["model"] = {"missing": torch.tensor([1.0])}
    bad_store = CheckpointStore(tmp_path / "bad-preflight")
    bad_store.publish("bad-preflight", malformed, generation=1, metadata=metadata)
    target = _engine(store=bad_store)
    assert target.run(until_step=2).status == "PAUSED"
    before = _engine_identity(target)
    with pytest.raises(ValueError, match="MODEL_KEYSET_MISMATCH"):
        target.resume_checkpoint("bad-preflight")
    assert _engine_identity(target) == before


def test_s110_source_and_resume_cpu_harness_use_distinct_exited_processes() -> None:
    """CPU coverage for the OS-process boundary used by the CUDA worker."""

    worker_spec = importlib.util.spec_from_file_location(
        "s110_worker_process_boundary_test",
        ROOT / "ops" / "stage1" / "run_s1_10_resume_worker.py",
    )
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    worker_path = ROOT / "ops" / "stage1" / "run_s1_10_resume_worker.py"
    source = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,json,os; "
                f"s=importlib.util.spec_from_file_location('source_identity', {str(worker_path)!r}); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
                "print(json.dumps({'pid': os.getpid(), 'identity': m._process_identity(os.getpid())}))"
            ),
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=30,
    )
    source_descriptor = json.loads(source.stdout)
    source_pid = int(source_descriptor["pid"])
    assert source_pid != os.getpid()
    assert worker._source_process_exited(source_pid, source_descriptor["identity"]) is True


def test_s110_source_process_identity_rejects_live_source_and_accepts_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_spec = importlib.util.spec_from_file_location(
        "s110_worker_identity_negative_test",
        ROOT / "ops" / "stage1" / "run_s1_10_resume_worker.py",
    )
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    current_identity = worker._process_identity(os.getpid())
    with pytest.raises(RuntimeError, match="SOURCE_PROCESS_STILL_RUNNING"):
        worker._source_process_exited(os.getpid(), current_identity)
    monkeypatch.setattr(worker, "_process_identity", lambda _pid: "replacement-process")
    assert worker._source_process_exited(os.getpid(), "source-process") is True


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_s110_worker_snapshot_comparison_is_closed_over_required_observations(mutation: str) -> None:
    worker_spec = importlib.util.spec_from_file_location(
        "s110_worker_snapshot_negative_test",
        ROOT / "ops" / "stage1" / "run_s1_10_resume_worker.py",
    )
    assert worker_spec is not None and worker_spec.loader is not None
    worker = importlib.util.module_from_spec(worker_spec)
    worker_spec.loader.exec_module(worker)
    snapshot = {
        "state_sha256": "a", "training_state": {"attempt_index": 6, "event_sequence": 9, "last_checkpoint_id": "c"},
        "cursor": {"index": 6}, "next_cursor": {"index": 6}, "sample_multiset": ["sample-a"],
        "rng_sha256": "b", "next_rng_state_sha256": "b", "per_rank_cuda_rng_state_hex": ["00"],
        "optimizer_sha256": "c", "scheduler_sha256": "d", "scaler_sha256": "e", "importance_sha256": "f",
        "records_sha256": "g", "object_digests": {"importance_accumulator": "f"}, "checkpoint_ids": ["c"],
        "bridge_optimizer_alias": True,
    }
    altered = deepcopy(snapshot)
    if mutation == "missing":
        altered.pop("sample_multiset")
    else:
        altered["unexpected"] = True
    assert worker._assert_trajectory(snapshot, altered, label="pre_skip") is False


def test_training_resume_rolls_back_after_mid_install_failure(tmp_path: Path) -> None:
    state, metadata = _published_state(CheckpointStore(tmp_path / "source"))
    malformed = deepcopy(state)
    # The payload passes the pre-install Mapping check but fails only after the
    # model, optimizer and scheduler have been installed, at cursor restore.
    malformed["cursor"] = {"schema_version": "batch-cursor-state-v1", "index": 999}
    malformed["training_state"] = dict(malformed["training_state"])
    malformed["training_state"]["last_checkpoint_id"] = "bad-mid-install"
    malformed["checkpoint_ids"] = ["bad-mid-install"]
    malformed["importance_trajectory_points"] = deepcopy(malformed["importance_trajectory_points"])
    malformed["importance_trajectory_points"][0]["checkpoint_id"] = "bad-mid-install"
    bad_store = CheckpointStore(tmp_path / "bad-mid-install")
    bad_store.publish("bad-mid-install", malformed, generation=1, metadata=metadata)
    target = _engine(store=bad_store)
    assert target.run(until_step=2).status == "PAUSED"
    before = _engine_identity(target)
    with pytest.raises(ValueError, match="BATCH_CURSOR_STATE_INDEX_OUT_OF_RANGE"):
        target.resume_checkpoint("bad-mid-install")
    assert _engine_identity(target) == before


@pytest.mark.parametrize(
    ("payload_ids", "label"),
    [
        (["forged-ancestor", "lineage-descendant"], "forged"),
        (["lineage-descendant"], "missing"),
        (["lineage-middle", "lineage-source", "lineage-descendant"], "reordered"),
    ],
)
def test_training_resume_rejects_hash_valid_payload_lineage_that_disagrees_with_committed_parents(
    tmp_path: Path, payload_ids: list[str], label: str
) -> None:
    """The payload list is not an authority over immutable store parents."""

    store = CheckpointStore(tmp_path / label)
    state, metadata = _published_state(store)
    ancestor = store.discover()[0].checkpoint_id
    parent_id = ancestor
    if label == "reordered":
        middle = deepcopy(state)
        middle["training_state"] = dict(middle["training_state"])
        middle["training_state"]["last_checkpoint_id"] = "lineage-middle"
        middle["checkpoint_ids"] = [ancestor, "lineage-middle"]
        middle["importance_trajectory_points"] = deepcopy(middle["importance_trajectory_points"])
        middle["importance_trajectory_points"][-1]["checkpoint_id"] = "lineage-middle"
        store.publish(
            "lineage-middle",
            middle,
            generation=2,
            metadata=metadata,
            parent_checkpoint_id=ancestor,
        )
        parent_id = "lineage-middle"
        malformed = deepcopy(middle)
    else:
        malformed = deepcopy(state)
    malformed["training_state"] = dict(malformed["training_state"])
    malformed["training_state"]["last_checkpoint_id"] = "lineage-descendant"
    malformed["checkpoint_ids"] = [ancestor if item == "lineage-source" else item for item in payload_ids]
    malformed["importance_trajectory_points"] = deepcopy(malformed["importance_trajectory_points"])
    malformed["importance_trajectory_points"][-1]["checkpoint_id"] = "lineage-descendant"
    store.publish(
        "lineage-descendant",
        malformed,
        generation=3 if label == "reordered" else 2,
        metadata=metadata,
        parent_checkpoint_id=parent_id,
    )
    target = _engine(store=store)
    before = _engine_identity(target)
    with pytest.raises(ValueError, match="PAYLOAD_LINEAGE_MISMATCH"):
        target.resume_checkpoint("lineage-descendant")
    assert _engine_identity(target) == before


def test_checkpoint_group_v2_commits_every_complete_attempt_and_keeps_commits_on_view_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A post-skip checkpoint is a later group generation at the same step."""

    steps = list(_steps())
    steps[2] = (
        TrainingMicrobatch("s110-skip-a", {"gradient": torch.tensor([0.25]), "inject_nonfinite": torch.tensor([1.0])}, ("s110-skip-a",)),
        TrainingMicrobatch("s110-skip-b", {"gradient": torch.tensor([0.125])}, ("s110-skip-b",)),
    )
    root, store_root, event_path = tmp_path / "workspace", tmp_path / "workspace" / "rank-0000", tmp_path / "workspace" / "events" / "rank-0000.jsonl"
    root.mkdir()
    sink = JsonlEventSink(event_path)
    engine = _engine(store=CheckpointStore(store_root), steps=tuple(steps[:3]), max_steps=3, max_attempts=4, event_sink=sink)
    assert engine.run().status == "DATA_EXHAUSTED"
    sink.close()
    commits = CheckpointStore(store_root).discover()
    pre, post = commits[-2:]
    assert pre.generation == 2 and post.generation == 3
    assert pre.checkpoint_id != post.checkpoint_id

    def binding(commit_id: str) -> dict[str, object]:
        state, _ = CheckpointStore(store_root).load(commit_id)
        return {
            "rank": 0,
            "checkpoint_store_ref": "rank-0000",
            "checkpoint_id": commit_id,
            "event_pointer": {
                "event_ref": "events/rank-0000.jsonl",
                "event_sha256": _file_sha256(event_path),
                "checkpoint_event_sequence": int(state["training_state"]["event_sequence"]) - 1,
            },
        }

    metadata = {
        "config_hash": "a" * 64, "environment_hash": "b" * 64,
        "model_manifest_id": "s110-tiny", "data_manifest_id": "s110-stream",
        "sampler_seed": 1, "epoch": 0, "committed_global_batch": 0,
        "next_global_batch": 0, "prefetch_policy": "disabled_for_correctness",
        "snapshot_type": "optimizer_step_checkpoint",
        "state_extension_schema": "training-checkpoint-state-v2",
        "save_wall_seconds": 0.0, "checkpoint_bytes": 0, "peak_memory_bytes": 0,
    }
    group = CheckpointGroupStore(root, "group")
    # A v2 rank payload must never silently downgrade into a legacy group
    # commit merely because the caller omitted the explicit v2 opt-in.
    with pytest.raises(ValueError, match="COMMIT_AND_TRAINING_STATE_VERSION_MISMATCH"):
        group.publish("s110-default-v1-rejects-v2", generation=3, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding(post.checkpoint_id)], metadata=metadata)
    # The no-skip v2 payload has exactly the legacy generation semantics and
    # therefore remains readable through the narrow v1 compatibility bridge.
    bridged = group.publish(
        "s110-default-v1-accepts-equivalent-v2",
        generation=2,
        run_id="s110-runtime-resume",
        world_size=1,
        rank_checkpoints=[binding(pre.checkpoint_id)],
        metadata=metadata,
    )
    assert bridged.schema_version == GROUP_COMMIT_SCHEMA
    assert group.verify(bridged.checkpoint_id).commit_sha256 == bridged.commit_sha256
    post_state, _ = CheckpointStore(store_root).load(post.checkpoint_id)
    assert isinstance(post_state, dict)
    legacy = deepcopy(post_state)
    legacy["schema_version"] = "training-checkpoint-state-v1"
    legacy.pop("checkpoint_ids")
    legacy["training_state"] = dict(legacy["training_state"])
    legacy["training_state"]["last_checkpoint_id"] = "s110-legacy-rank"
    legacy["importance_trajectory_points"] = deepcopy(legacy["importance_trajectory_points"])
    legacy["importance_trajectory_points"][-1]["checkpoint_id"] = "s110-legacy-rank"
    CheckpointStore(store_root).publish("s110-legacy-rank", legacy, generation=2, metadata={"run_spec_hash": engine.spec.spec_hash, "registry_hash": engine.registry.coordinate_registry_hash, "optimizer_contract_hash": engine.registry.optimizer_contract_hash, "runtime_layout_hash": engine.registry.runtime_layout_hash, "world_size": 1})
    legacy_commit = group.publish("s110-legacy-v1", generation=2, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding("s110-legacy-rank")], metadata=metadata)
    assert legacy_commit.schema_version == GROUP_COMMIT_SCHEMA
    with pytest.raises(ValueError, match="COMMIT_AND_TRAINING_STATE_VERSION_MISMATCH"):
        group.publish("s110-explicit-v2-rejects-v1", generation=2, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding("s110-legacy-rank")], metadata=metadata, commit_schema_version=GROUP_COMMIT_SCHEMA_V2)
    native_publish = group.publish

    def publish_v2(checkpoint_id: str, **kwargs: object) -> object:
        return native_publish(
            checkpoint_id,
            commit_schema_version=GROUP_COMMIT_SCHEMA_V2,
            **kwargs,
        )

    monkeypatch.setattr(group, "publish", publish_v2)
    first = group.publish("s110-group-pre", generation=2, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding(pre.checkpoint_id)], metadata=metadata)
    second = group.publish("s110-group-post", generation=3, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding(post.checkpoint_id)], metadata=metadata, parent_checkpoint_id=first.checkpoint_id)
    assert second.schema_version == "runtime.checkpoint-group-commit.v2"
    assert (second.global_step, second.successful_optimizer_step, second.skip_count) == (2, 2, 1)
    assert group.load(second.checkpoint_id)[1].commit_sha256 == second.commit_sha256
    forged = deepcopy(post_state)
    forged["training_state"] = dict(forged["training_state"])
    forged["training_state"]["last_checkpoint_id"] = "s110-rank-forged"
    forged["training_state"]["attempt_index"] = 4
    forged["training_state"]["skipped_steps"] = 2
    # ``pre`` is a real, hash-valid ancestor in this same store, but the
    # committed parent for the forged rank object is ``post``.  This isolates
    # the store-parent/payload disagreement from malformed IDs or bad hashes.
    forged["checkpoint_ids"] = [pre.checkpoint_id, "s110-rank-forged"]
    forged["importance_trajectory_points"] = deepcopy(forged["importance_trajectory_points"])
    forged["importance_trajectory_points"][-1]["checkpoint_id"] = "s110-rank-forged"
    CheckpointStore(store_root).publish(
        "s110-rank-forged",
        forged,
        generation=4,
        metadata={
            "run_spec_hash": engine.spec.spec_hash,
            "registry_hash": engine.registry.coordinate_registry_hash,
            "optimizer_contract_hash": engine.registry.optimizer_contract_hash,
            "runtime_layout_hash": engine.registry.runtime_layout_hash,
            "world_size": 1,
        },
        parent_checkpoint_id=post.checkpoint_id,
    )
    with pytest.raises(ValueError, match="V2_PAYLOAD_LINEAGE_MISMATCH"):
        group.publish(
            "s110-group-forged-rank-lineage",
            generation=4,
            run_id="s110-runtime-resume",
            world_size=1,
            rank_checkpoints=[binding("s110-rank-forged")],
            metadata=metadata,
            parent_checkpoint_id=second.checkpoint_id,
        )
    self_consistent = deepcopy(post_state)
    self_consistent["training_state"] = dict(self_consistent["training_state"])
    self_consistent["training_state"].update(
        {"last_checkpoint_id": "s110-rank-group-parent-mismatch", "attempt_index": 4, "skipped_steps": 2}
    )
    self_consistent["checkpoint_ids"] = [
        *list(post_state["checkpoint_ids"]),
        "s110-rank-group-parent-mismatch",
    ]
    self_consistent["importance_trajectory_points"] = deepcopy(self_consistent["importance_trajectory_points"])
    self_consistent["importance_trajectory_points"][-1]["checkpoint_id"] = "s110-rank-group-parent-mismatch"
    CheckpointStore(store_root).publish(
        "s110-rank-group-parent-mismatch",
        self_consistent,
        generation=4,
        metadata={
            "run_spec_hash": engine.spec.spec_hash,
            "registry_hash": engine.registry.coordinate_registry_hash,
            "optimizer_contract_hash": engine.registry.optimizer_contract_hash,
            "runtime_layout_hash": engine.registry.runtime_layout_hash,
            "world_size": 1,
        },
        parent_checkpoint_id=post.checkpoint_id,
    )
    with pytest.raises(ValueError, match="V2_PARENT_RANK_LINEAGE_MISMATCH"):
        group.publish(
            "s110-group-parent-rank-mismatch",
            generation=4,
            run_id="s110-runtime-resume",
            world_size=1,
            rank_checkpoints=[binding("s110-rank-group-parent-mismatch")],
            metadata=metadata,
            parent_checkpoint_id=first.checkpoint_id,
        )
    with pytest.raises(FileExistsError, match="COMMIT_EXISTS"):
        group.publish("s110-group-post", generation=3, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding(post.checkpoint_id)], metadata=metadata, parent_checkpoint_id=first.checkpoint_id)

    original_reconcile = group.reconcile
    monkeypatch.setattr(group, "reconcile", lambda: (_ for _ in ()).throw(RuntimeError("broadcast failed")))
    with pytest.raises(RuntimeError, match="broadcast failed"):
        group.publish("s110-group-post-view-failure", generation=3, run_id="s110-runtime-resume", world_size=1, rank_checkpoints=[binding(post.checkpoint_id)], metadata=metadata, parent_checkpoint_id=first.checkpoint_id)
    assert (group.commits / "s110-group-post-view-failure.json").is_file()
    monkeypatch.setattr(group, "reconcile", original_reconcile)
    assert group.verify("s110-group-post-view-failure").generation == 3
