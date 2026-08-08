"""Stage 0 S0.10 capacity protocol and schema regressions."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from param_importance_nlp.capacity import (
    ParameterTensorShape,
    build_compute_communication_envelope,
    build_parameter_state_envelope,
)
from param_importance_nlp.cli import _load_mapping, _validate_project_json_schema
from param_importance_nlp.contracts import GateRecord, GateStatus, ResolvedConfig
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.runtime import (
    TaskArtifactStore,
    TaskRuntime,
    TaskRuntimeEnvironment,
)
from param_importance_nlp.stage0_g8 import (
    _capacity_config,
    _gate_key,
    _summarize_measurements,
)


ROOT = Path(__file__).resolve().parents[1]


def _formal_template() -> ResolvedConfigV2:
    base = ResolvedConfig.resolve(
        _load_mapping(ROOT / "configs/local-fixtures/resolved-config-v1.json"),
        _load_mapping(ROOT / "configs/run-ready/layers/formal-stage1-pythia14m.yaml"),
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id="stage1.07_single_gpu_pythia14m",
        overrides=_load_mapping(
            ROOT / "configs/run-ready/v2/stage1-pythia14m-formal.yaml"
        ),
    )


def test_capacity_candidate_configs_bind_precision_scale_and_data_route() -> None:
    template = _formal_template()
    cases = (
        ("pythia-14m-step0", "g8-c-14m", 0, 1, "fp32", "debug", 4),
        ("pythia-14m-step0", "g8-c-14m", 0, 4, "bf16", "debug", 4),
        ("pythia-160m-deduped-step0", "g8-s4-160m", 4, 4, "bf16", "train", 64),
        ("pythia-410m-deduped-step0", "g8-s5-410m", 5, 4, "bf16", "train", 64),
    )
    hashes: set[str] = set()
    for model, profile, stage, world, precision, split, global_batch in cases:
        config = _capacity_config(
            repository=ROOT,
            input_refs=("evidence/upstream/commit.json",),
            template=template,
            model_id=model,
            profile_id=profile,
            candidate_stage=stage,
            world_size=world,
            precision_profile=precision,
            output_dir=f"evidence/g8/{profile}-{precision}-w{world}",
        )
        hashes.add(config.config_hash)
        assert config.task_id == "stage0.10_capacity_and_operations"
        assert config.base_config.section("model")["asset_id"] == model
        assert config.base_config.section("data")["split"] == split
        assert config.base_config.section("data")["sampling_design"] == "without_replacement_frozen_epoch"
        assert config.base_config.section("batching")["global_batch_size"] == global_batch
        assert config.base_config.section("precision")["compute_dtype"] == (
            "float32" if precision == "fp32" else "bfloat16"
        )
    assert len(hashes) == len(cases)


def test_g8_gate_key_matches_runtime_environment_ref_contract() -> None:
    assert _gate_key("stage0.G8-C") == "gate_stage0_g8_c"
    assert _gate_key("stage0.G8-S4") == "gate_stage0_g8_s4"
    assert _gate_key("stage0.G8-S5") == "gate_stage0_g8_s5"
    assert _gate_key("stage0.G8") == "gate_stage0_g8"


def test_g8_gate_record_commit_is_verifiable_by_runtime_preflight(
    tmp_path: Path,
) -> None:
    store = TaskArtifactStore(tmp_path, "evidence/stage0/tasks/10-test")
    record = GateRecord(
        gate_id="stage0.G8-C",
        stage=0,
        status=GateStatus.PASS,
        checked_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        measured={"capacity": "PASS"},
        threshold={"required": "all G8-C checks PASS"},
        evidence_refs=("evidence/stage0/g8-suite/fixture/preflight.json",),
    )
    ref = store.publish(
        task_id="stage0.10_capacity_and_operations",
        artifact_kind="gate_g8_c",
        config_hash="c" * 64,
        run_intent="formal",
        payload=record.to_dict(),
        formal_eligible=True,
    ).commit_ref
    environment = TaskRuntimeEnvironment(
        capabilities=frozenset({"server"}),
        frozen_contract_stages=frozenset({0}),
        passed_gate_ids=frozenset({"stage0.G8-C"}),
        evidence_refs={"gate_stage0_g8_c": ref},
    )
    verified, evidence = TaskRuntime(workspace_root=tmp_path)._verified_gate_ref(
        environment,
        "stage0.G8-C",
    )
    assert verified is True
    assert evidence == (ref,)


def test_g8_envelope_schemas_accept_exact_shape_controls() -> None:
    tensors = (
        ParameterTensorShape("weight", (4, 8), "float32"),
        ParameterTensorShape("bias", (4,), "float32"),
    )
    parameter = build_parameter_state_envelope(
        model_id="fixture-model",
        tensors=tensors,
        config_hash="a" * 64,
        model_manifest_id="fixture-manifest",
        checkpoint_every_steps=25,
    )
    work = build_compute_communication_envelope(
        model_id="fixture-model",
        parameter_count=36,
        world_size=4,
        microbatches_per_optimizer_step=4,
        checkpoint_every_steps=25,
        compute_dtype="bfloat16",
        sequence_length=2048,
        microbatch_size=4,
        candidate_stage=4,
    )
    for name, value in (
        ("stage0-g8-parameter-envelope-v1.json", parameter),
        ("stage0-g8-work-envelope-v1.json", work),
    ):
        schema = json.loads((ROOT / "schemas" / name).read_text(encoding="utf-8"))
        _validate_project_json_schema(schema)
        assert value["schema_version"] == {
            "stage0-g8-parameter-envelope-v1.json": "stage0.parameter-state-capacity-envelope.v1",
            "stage0-g8-work-envelope-v1.json": "stage0.compute-communication-work-envelope.v1",
        }[name]


def _report(key: str, mode: str, repeat: int) -> dict[str, object]:
    if key.endswith("-w4"):
        base_tps = 320.0
    else:
        base_tps = 100.0
    tps = base_tps if mode == "minimal" else base_tps * 0.95
    rank_reports = [{"loader_profiles": []}]
    if key == "g8-c-14m-bf16-w4" and mode == "formal" and repeat == 0:
        rank_reports = [
            {
                "loader_profiles": [
                    {"num_workers": workers, "samples_per_second": float(100 + workers * 10)}
                    for workers in (0, 1, 2, 4)
                ]
            }
            for _ in range(4)
        ]
    return {
        "effective_tokens_per_second": tps,
        "peak_memory_fraction_max": 0.5,
        "open_fds_fraction_max": 0.2,
        "rank_peak_memory_imbalance_fraction": 0.01,
        "checkpoint_pause_seconds_max": 1.0,
        "peak_memory_bytes_max": 1000,
        "checkpoint_bytes_total_max_boundary": 2000,
        "world_size": 4 if key.endswith("-w4") else 1,
        "rank_reports": rank_reports,
        "profile_id": (
            "g8-s4-160m" if key.startswith("g8-s4") else
            "g8-s5-410m" if key.startswith("g8-s5") else "g8-c-14m"
        ),
        "precision_profile": "fp32" if "fp32" in key else "bf16",
    }


def test_measurement_summary_replays_paired_overhead_scaling_and_loader_sweep() -> None:
    keys = (
        "g8-c-14m-fp32-w1",
        "g8-c-14m-fp32-w4",
        "g8-c-14m-bf16-w1",
        "g8-c-14m-bf16-w4",
        "g8-s4-160m-bf16-w4",
        "g8-s5-410m-bf16-w4",
    )
    controls = {
        "execution_keys": keys,
        "parameter_values": {
            key: {
                "peak_parameter_state_gpu_bytes": 900,
                "checkpoint_parameter_state_bytes": 500,
            }
            for key in keys
        },
    }
    reports = {
        (key, mode, repeat): _report(key, mode, repeat)
        for key in keys
        for mode in ("minimal", "formal")
        for repeat in range(3)
    }
    summary = _summarize_measurements(controls, reports)
    assert summary["status"] == "PASS"
    assert summary["precision_coverage"] == ["bf16", "fp32"]
    assert summary["formal_logging_overhead_limit"] == 0.10
    assert summary["strong_scaling"]["bf16"]["strong_scaling_efficiency"] == 0.8
    assert summary["loader_worker_saturation_choice"] == 4
    assert len(summary["estimation_error_rows"]) == 6


def test_all_g8_json_schemas_are_valid_project_schema_documents() -> None:
    schemas = sorted((ROOT / "schemas").glob("stage0-g8-*.json"))
    assert len(schemas) == 6
    for path in schemas:
        _validate_project_json_schema(json.loads(path.read_text(encoding="utf-8")))
