from __future__ import annotations

import importlib.util
import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from param_importance_nlp.stage1_precision import (
    SCALE_STATISTIC_OBJECTS,
    Stage1PrecisionError,
    _engine_skip_trace,
    _source_hashes,
    build_stage1_s19_evidence,
    replay_stage1_s19_evidence,
    run_stage1_s19_bf16_smoke,
    validate_stage1_s19_evidence,
)
import param_importance_nlp.stage1_precision as precision
from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.core.accumulator import ImportanceAccumulator
from param_importance_nlp.runtime.checkpoint import CheckpointStore
from param_importance_nlp.runtime.training import TrainingEngine, TrainingRunSpec
import param_importance_nlp.stage1_precision_oracle as precision_oracle


ROOT = Path(__file__).resolve().parents[1]
GPU_UUIDS = (
    "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd",
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
)


def _environment_summary(*, visible: str, rank: int, uuid: str) -> dict[str, object]:
    return {
        "torch_version": "2.8.0", "cuda_runtime_version": "12.8", "cudnn_version": "9100", "nccl_version": "2.27.7",
        "deterministic_algorithms": True, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "cublas_workspace_config": ":4096:8", "pythonhashseed": "0", "cuda_visible_devices": visible,
        "local_rank": rank, "local_gpu_uuid": uuid,
    }


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_s19_cpu_trace_is_not_formal_but_replays_and_covers_all_amp_objects() -> None:
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="cpu-test")
    assert evidence["numeric_report"]["status"] == "NOT_RUN"
    assert all(value for key, value in evidence["numeric_report"]["requirements"].items() if key not in {"bf16_cuda_smoke", "ddp_global_nonfinite_skip", "repeated_short_run_bitwise_deterministic"})
    validate_stage1_s19_evidence(evidence, source_root=ROOT)
    assert replay_stage1_s19_evidence(evidence, source_root=ROOT)["status"] == "PASS"
    objects = {item["object"] for item in evidence["comparison_table"]["comparison_objects"]}
    assert set(SCALE_STATISTIC_OBJECTS) <= objects
    assert {"optimizer_delta", "s1", "g1", "weighted_u_score"} <= objects
    assert all(row["n_q"] > 0 and row["n_q_rule"] and row["oracle_hash"] for item in evidence["comparison_table"]["comparison_objects"] for row in item["rows"])


def test_s19_t_amp_uses_exact_disjoint_normalized_and_near_zero_shapes() -> None:
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="t-amp-shapes")
    table = evidence["comparison_table"]
    objects = {item["object"]: item for item in table["comparison_objects"]}
    exact_zero = objects["t_amp_exact_zero_contract"]["rows"][0]
    near_zero = objects["t_amp_near_zero_contract"]["rows"][0]
    normalized = next(row for item in table["comparison_objects"] for row in item["rows"] if row["branch"] == "normalized")
    assert exact_zero["branch"] == near_zero["branch"] == "near_zero_absolute"
    assert exact_zero["original_unit_max_abs_error"] == 0.0
    assert near_zero["original_unit_max_abs_error"] > 0.0
    assert set(exact_zero).isdisjoint({"oracle_norm_inf", "scaled_threshold", "normalized_l2_limit"})
    assert {"oracle_norm_inf", "scaled_threshold", "normalized_l2_limit"} <= set(normalized)
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_t_amp_oneof")

    # Rehash the affected role and its report binding together.  The strict
    # oneOf shapes must still reject a row that mixes the two branch contracts.
    for object_id, extra_key, extra_value in (
        ("t_amp_exact_zero_contract", "oracle_norm_inf", 1.0),
        (normalized["object"], "near_zero_threshold", 1e-6),
    ):
        tampered = json.loads(json.dumps(evidence))
        changed = next(item for item in tampered["comparison_table"]["comparison_objects"] if item["object"] == object_id)
        coordinate = "value" if object_id.startswith("t_amp_") else normalized["coordinate"]
        changed_row = next(row for row in changed["rows"] if row["coordinate"] == coordinate)
        changed_row[extra_key] = extra_value
        for flat_row in tampered["comparison_table"]["rows"]:
            if flat_row["case_id"] == object_id and flat_row["field"] == changed_row["coordinate"]:
                flat_row["actual"][extra_key] = extra_value
        table_body = dict(tampered["comparison_table"])
        table_body.pop("table_hash")
        tampered["comparison_table"]["table_hash"] = canonical_json_hash(table_body)
        report_body = dict(tampered["numeric_report"])
        report_body["table_hash"] = tampered["comparison_table"]["table_hash"]
        report_body.pop("report_hash")
        tampered["numeric_report"] = {**report_body, "report_hash": canonical_json_hash(report_body)}
        with pytest.raises(Stage1PrecisionError, match="S1_9_T_AMP_"):
            validate_stage1_s19_evidence(tampered, source_root=ROOT)
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:comparison_table"):
            formal._validate_s1_9_schemas(ROOT, {"comparison_table": tampered["comparison_table"]})


def test_s19_bf16_resume_schema_binds_full_engine_state_and_accumulator_views() -> None:
    schema = json.loads((ROOT / "schemas" / "stage1" / "s1-9-single-bf16-worker-v1.json").read_text(encoding="utf-8"))
    resume = schema["$defs"]["resume"]
    state_hashes = schema["$defs"]["state_field_hashes"]
    expected_checkpoint_roles = {
        "schema_version", "model", "optimizer", "scheduler", "scaler", "cursor", "training_state", "rng", "importance", "records", "importance_trajectory_points", "checkpoint_ids", "run_spec_hash", "registry_hash", "optimizer_contract_hash", "runtime_layout_hash",
    }
    expected_accumulator_views = {
        "signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped",
        "data_movement", "net_data_movement", "total_movement", "total_endpoint_movement",
        "weight_decay_movement", "net_weight_decay_movement", "actual_update_raw_importance",
        "actual_update_raw_importance_available", "magnitude", "attempted_steps",
    }
    expected_state_hashes = expected_checkpoint_roles - {"schema_version"} | {
        f"importance_view_{name}" for name in expected_accumulator_views
    }
    production_view_names = {
        name for name, member in vars(ImportanceAccumulator).items() if isinstance(member, property)
    }
    assert expected_accumulator_views == production_view_names
    assert set(resume["properties"]["checkpoint_role_fields"]["const"]) == expected_checkpoint_roles
    assert set(state_hashes["required"]) == expected_state_hashes
    assert state_hashes["additionalProperties"] is False
    assert {"checkpoint_id", "checkpoint_manifest_sha256", "next_step_independent_engines", "accumulator_public_view_set_exact"} <= set(resume["required"])
    source = (ROOT / "src" / "param_importance_nlp" / "stage1_precision.py").read_text(encoding="utf-8")
    assert "TrainingEngine(" in source
    assert "engine.tracker.accumulator" in source
    assert all(token in source for token in ("CheckpointStore(", "engine.save_checkpoint()", "fresh.resume_checkpoint(checkpoint_id)", "corrupt_manifest.write_bytes", "omission_rejected", "record_order_rejected", "importance_views"))


def test_s19_checkpoint_store_resume_uses_a_fresh_engine_and_tracker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the same production save/resume surfaces used by BF16 S1.9.

    CPU execution is intentional here: the isolated worker remains responsible
    for BF16 arithmetic, while this unit test proves the Store → commit → fresh
    TrainingEngine.resume_checkpoint lifecycle without a GPU.
    """

    store = CheckpointStore(tmp_path / "s19-checkpoint-store")
    save_calls: list[int] = []
    resume_calls: list[tuple[int, str]] = []
    production_save = TrainingEngine.save_checkpoint
    production_resume = TrainingEngine.resume_checkpoint

    def spy_save(engine: TrainingEngine) -> str:
        save_calls.append(id(engine))
        return production_save(engine)

    def spy_resume(engine: TrainingEngine, checkpoint_id: str) -> str:
        resume_calls.append((id(engine), checkpoint_id))
        return production_resume(engine, checkpoint_id)

    monkeypatch.setattr(TrainingEngine, "save_checkpoint", spy_save)
    monkeypatch.setattr(TrainingEngine, "resume_checkpoint", spy_resume)

    def build_engine(store_override: CheckpointStore = store) -> TrainingEngine:
        module = precision._S19FiniteSkipFiniteClassifier(device=torch.device("cpu"))
        batches = tuple(
            (
                precision._engine_microbatch(sample_id=f"resume-{index}", micro=0, value=float(1 + index), device=torch.device("cpu")),
                precision._engine_microbatch(sample_id=f"resume-{index}", micro=1, value=float(2 + index), device=torch.device("cpu")),
            )
            for index in range(3)
        )
        optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, weight_decay=0.01, foreach=False, fused=False)
        return TrainingEngine(
            spec=TrainingRunSpec("s19-production-store-resume", "local_fixture", max_steps=3, max_attempts=3, importance_enabled=True, estimator_name="u", accumulation_dtype="float32", weights_exogenous=True, common_mean_assumption=True),
            model=precision.TorchModelAdapter(module, task_type="sequence_classification"), optimizer=optimizer,
            cursor=precision.InMemoryDatasetAdapter("s19-production-store-resume", batches).cursor(seed=1909), checkpoint_store=store_override,
        )

    def run_one(engine: TrainingEngine) -> dict[str, object]:
        record = engine._run_attempt(engine.cursor.next_microbatches())
        engine._records.append(record)
        return record.to_dict()

    source = build_engine()
    run_one(source); run_one(source)
    checkpoint_id = source.save_checkpoint()
    payload, commit = store.load(checkpoint_id)

    def rebased_payload(negative_id: str) -> dict[str, object]:
        result = deepcopy(dict(payload))
        state = dict(result["training_state"])
        state["last_checkpoint_id"] = negative_id
        result["training_state"] = state
        result["checkpoint_ids"] = [negative_id]
        result["importance_trajectory_points"] = []
        return result

    def live_state(engine: TrainingEngine) -> dict[str, object]:
        assert engine.tracker is not None
        return {
            "state": engine.state.to_dict(), "cursor": dict(engine.cursor.state_dict()),
            "model": precision._state_wire(engine.model.module.state_dict()),
            "optimizer": precision._state_wire(engine.optimizer.state_dict()),
            "importance": precision._state_wire(engine.tracker.accumulator.state_dict()),
        }

    def assert_rejected_before_mutation(label: str, mutate, *, corrupt_manifest: bool = False) -> None:
        negative_store = CheckpointStore(tmp_path / f"negative-{label}")
        negative_id = f"s19-production-negative-{label}"
        negative_payload = rebased_payload(negative_id)
        mutate(negative_payload)
        negative_store.publish(negative_id, negative_payload, generation=2, metadata={"run_spec_hash": source.spec.spec_hash, "registry_hash": source.registry.coordinate_registry_hash, "optimizer_contract_hash": source.registry.optimizer_contract_hash, "runtime_layout_hash": source.registry.runtime_layout_hash, "world_size": 1})
        if corrupt_manifest:
            manifest = negative_store.objects / negative_id / "manifest.json"
            bytes_before = manifest.read_bytes()
            manifest.write_bytes(bytes_before[:-1] + bytes([bytes_before[-1] ^ 1]))
        probe = build_engine(negative_store)
        before = live_state(probe)
        with pytest.raises(Exception):
            probe.resume_checkpoint(negative_id)
        assert live_state(probe) == before

    assert_rejected_before_mutation("omission", lambda item: item.pop("importance"))
    assert_rejected_before_mutation("record-order", lambda item: item.__setitem__("records", list(reversed(item["records"]))))
    assert_rejected_before_mutation("corruption", lambda _item: None, corrupt_manifest=True)

    fresh = build_engine()
    assert fresh is not source and fresh.tracker is not source.tracker
    assert fresh.tracker is not None and source.tracker is not None
    assert fresh.tracker.accumulator is not source.tracker.accumulator
    assert fresh.resume_checkpoint(checkpoint_id) == checkpoint_id
    assert save_calls == [id(source)]
    assert (id(fresh), checkpoint_id) in resume_calls
    assert len(resume_calls) == 4
    assert commit.checkpoint_id == checkpoint_id
    assert payload["checkpoint_ids"] == [checkpoint_id]
    assert fresh.state == source.state
    assert fresh.cursor.state_dict() == source.cursor.state_dict()
    assert precision._state_wire(fresh.tracker.accumulator.state_dict()) == precision._state_wire(source.tracker.accumulator.state_dict())
    assert run_one(source) == run_one(fresh)


def test_s19_formal_reproduction_copies_only_the_committed_checkpoint_store_authority(tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_checkpoint_store_reproduction")
    source = CheckpointStore(tmp_path / "source-store")
    source.publish("authoritative", {"parameter": torch.tensor([1.0])}, generation=1, metadata={})
    source.publish("unrelated-negative", {"parameter": torch.tensor([2.0])}, generation=2, metadata={})
    target = tmp_path / "published-store"
    index = formal._copy_checkpoint_store_reproduction(source.root, target, "authoritative")
    formal._validate_s1_9_schemas(ROOT, {"bf16_checkpoint_store": index})
    assert formal._verify_checkpoint_store_reproduction(target, index)
    assert (target / "commits" / "authoritative.json").is_file()
    assert (target / "objects" / "authoritative" / "manifest.json").is_file()
    assert not (target / "commits" / "unrelated-negative.json").exists()
    manifest = target / "objects" / "authoritative" / "manifest.json"
    bytes_before = manifest.read_bytes()
    manifest.write_bytes(bytes_before[:-1] + bytes([bytes_before[-1] ^ 1]))
    assert not formal._verify_checkpoint_store_reproduction(target, index)


def test_s19_checkpoint_store_reproduction_schema_rejects_deep_drift_and_path_escape(tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_checkpoint_store_reproduction_schema")
    source = CheckpointStore(tmp_path / "source-store")
    source.publish("authoritative", {"parameter": torch.tensor([1.0])}, generation=1, metadata={})
    target = tmp_path / "published-store"
    index = formal._copy_checkpoint_store_reproduction(source.root, target, "authoritative")

    deep_extra = json.loads(json.dumps(index))
    deep_extra["bundle_file_hashes"][0]["unexpected"] = True
    missing = dict(index); missing.pop("bundle_manifest_sha256")
    escaped = json.loads(json.dumps(index))
    escaped["bundle_file_hashes"].append({"ref": "objects/authoritative/tensors/../escape", "sha256": "0" * 64})
    hash_map_drift = json.loads(json.dumps(index))
    hash_map_drift["bundle_file_hashes"][0]["sha256"] = "f" * 64
    for mutated in (deep_extra, missing, escaped):
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:bf16_checkpoint_store"):
            formal._validate_s1_9_schemas(ROOT, {"bf16_checkpoint_store": mutated})
    # A syntactically valid digest still cannot drift from the committed
    # CheckpointStore bytes during production readback.
    formal._validate_s1_9_schemas(ROOT, {"bf16_checkpoint_store": hash_map_drift})
    assert not formal._verify_checkpoint_store_reproduction(target, hash_map_drift)
    with pytest.raises(formal.Stage1S19FormalError, match="REPRODUCTION_ID_INVALID"):
        formal._checkpoint_store_reproduction_index(source.root, "../escape")
    escaped_id = dict(index); escaped_id["checkpoint_id"] = "authoritative\\escape"
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:bf16_checkpoint_store"):
        formal._validate_s1_9_schemas(ROOT, {"bf16_checkpoint_store": escaped_id})


def test_s19_skip_state_is_full_and_cursor_progresses() -> None:
    trace = _engine_skip_trace(device=torch.device("cpu"))
    checks = trace["skip_full_state_checks"]
    assert all(checks.values())
    assert checks["cursor_advanced_exactly_one_frozen_batch"]
    assert checks["rng_zero_consumption_contract"]
    assert trace["scaler_calls"] == {"unscale": 3, "step": 3, "update": 3}
    assert trace["actual_optimizer_step_calls"] == 2
    assert trace["reference_comparison"]["next_finite_batch"]


def test_s19_source_closure_includes_runtime_and_formal_workers() -> None:
    hashes = _source_hashes(ROOT)
    expected = {
        "fixtures/stage1/stage1-s19-precision-fixture-v1.json",
        "src/param_importance_nlp/stage1_precision.py", "src/param_importance_nlp/stage1_precision_oracle.py",
        "src/param_importance_nlp/contracts/jsonio.py", "src/param_importance_nlp/contracts/runtime_evidence.py",
        "src/param_importance_nlp/core/accumulator.py", "src/param_importance_nlp/core/estimators.py", "src/param_importance_nlp/core/registry.py", "src/param_importance_nlp/core/sufficient_statistics.py", "src/param_importance_nlp/core/tensors.py",
        "src/param_importance_nlp/providers/training.py", "src/param_importance_nlp/runtime/gradients.py", "src/param_importance_nlp/runtime/operations.py", "src/param_importance_nlp/runtime/optimizer.py", "src/param_importance_nlp/runtime/reducers.py", "src/param_importance_nlp/runtime/task_artifacts.py", "src/param_importance_nlp/runtime/training.py",
        "ops/stage1/formalize_s1_6.py", "ops/stage1/formalize_s1_9.py", "ops/stage1/run_s1_9_single_bf16_worker.py", "ops/stage1/run_s1_9_ddp_skip_worker.py",
        "tests/test_stage1_s19_precision.py",
        "schemas/stage1/s1-9-precision-fixture-v1.json", "schemas/stage1/s1-9-numeric-report-v1.json", "schemas/stage1/s1-9-oracle-bundle-v1.json", "schemas/stage1/s1-9-trace-bundle-v1.json", "schemas/stage1/s1-9-comparison-table-v1.json", "schemas/stage1/s1-9-gate-record-v1.json", "schemas/stage1/s1-9-replay-validation-v1.json", "schemas/stage1/s1-9-validation-v1.json", "schemas/stage1/s1-9-formalization-index-v1.json", "schemas/stage1/s1-9-single-bf16-worker-v1.json", "schemas/stage1/s1-9-ddp-skip-worker-v1.json", "schemas/stage1/s1-9-bf16-checkpoint-store-reproduction-v1.json",
    }
    assert set(hashes) == expected
    assert all(len(value) == 64 for value in hashes.values())


def test_s19_frozen_fixture_runs_strict_schema_before_oracle_arithmetic() -> None:
    fixture_path = ROOT / "fixtures" / "stage1" / "stage1-s19-precision-fixture-v1.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    precision_oracle._validate_s1_9_fixture_schema(ROOT, fixture)
    assert precision_oracle.load_stage1_s19_fixture(ROOT)["fixture_id"] == precision_oracle.FIXTURE_ID
    mutations = []
    nested_unknown = json.loads(json.dumps(fixture)); nested_unknown["clip"]["unknown"] = True; mutations.append(nested_unknown)
    nested_missing = json.loads(json.dumps(fixture)); nested_missing["autocast_regression"]["microbatches"][0].pop("target"); mutations.append(nested_missing)
    cardinality = json.loads(json.dumps(fixture)); cardinality["loss_scales"].pop(); mutations.append(cardinality)
    identifier = json.loads(json.dumps(fixture)); identifier["fixture_id"] = "drift"; mutations.append(identifier)
    for mutated in mutations:
        with pytest.raises(precision_oracle.Stage1PrecisionOracleError, match="FIXTURE_SCHEMA_VALIDATION_FAILED"):
            precision_oracle._validate_s1_9_fixture_schema(ROOT, mutated)

    # A fixture self-hash only detects accidental corruption.  The v1 numeric
    # vectors themselves are frozen independently, so an attacker cannot edit
    # a scale/threshold/optimizer/gradient and simply recompute fixture_hash.
    joint_rehash_paths = (
        ("loss_scales", 0),
        ("clip", "max_norm"),
        ("adamw", "betas", 0),
        ("adamw", "gradient", 0),
    )
    for path in joint_rehash_paths:
        mutated = json.loads(json.dumps(fixture))
        cursor = mutated
        for part in path[:-1]:
            cursor = cursor[part]
        cursor[path[-1]] = float(cursor[path[-1]]) + 0.125
        body = dict(mutated)
        body.pop("fixture_hash")
        mutated["fixture_hash"] = precision_oracle._canonical_hash(body)
        with pytest.raises(precision_oracle.Stage1PrecisionOracleError, match="FIXTURE_FROZEN_BODY_HASH_MISMATCH"):
            precision_oracle._validate_frozen_fixture_body(mutated)


def test_s19_bf16_requires_isolated_cuda_worker() -> None:
    with pytest.raises(Stage1PrecisionError, match="S1_9_BF16_REQUIRES_CUDA"):
        run_stage1_s19_bf16_smoke(source_root=ROOT, device="cpu", checkpoint_dir=ROOT / ".s19-test-never-created")


def test_s19_ddp_worker_proves_gradscaler_ordering_negative_control() -> None:
    worker = _module(ROOT / "ops" / "stage1" / "run_s1_9_ddp_skip_worker.py", "s19_ddp_worker")
    assert worker._cpu_gradscaler_ordering_negative_control() is True


def test_s19_formal_helpers_reject_bad_explicit_uuid_sets() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_formalizer")
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_APPROVED_UUIDS_INVALID"):
        formal._parse_uuids(",".join(GPU_UUIDS[:3]))
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_APPROVED_UUIDS_INVALID"):
        formal._parse_uuids(",".join((GPU_UUIDS[0].upper(), *GPU_UUIDS[1:])))
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_APPROVED_UUIDS_INVALID"):
        formal._parse_uuids(",".join((GPU_UUIDS[0][:-1], *GPU_UUIDS[1:])))
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_APPROVED_UUIDS_INVALID"):
        formal._parse_uuids(",".join((GPU_UUIDS[0] + "-x", *GPU_UUIDS[1:])))
    assert formal._parse_uuids(",".join(GPU_UUIDS)) == GPU_UUIDS


def test_s19_pre_cuda_policy_and_environment_summary_are_strict_and_uuid_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_formal_environment")
    visible = ",".join(GPU_UUIDS)
    environment = {"CUDA_VISIBLE_DEVICES": visible, "S1_9_RUN_TOKEN": "a" * 64, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
    assert formal._validate_child_environment(environment, approved=GPU_UUIDS, run_token="a" * 64)["cuda_visible_devices"] == visible
    malformed = dict(environment); malformed["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    with pytest.raises(formal.Stage1S19FormalError, match="PRE_CUDA_ENVIRONMENT_POLICY_INVALID"):
        formal._validate_child_environment(malformed, approved=GPU_UUIDS, run_token="a" * 64)
    summary = _environment_summary(visible=visible, rank=0, uuid=GPU_UUIDS[0])
    assert formal._worker_environment_summary_valid(summary, cuda_visible_devices=visible, local_rank=0, local_gpu_uuid=GPU_UUIDS[0])
    bad_summary = dict(summary); bad_summary["local_gpu_uuid"] = GPU_UUIDS[0].upper()
    assert not formal._worker_environment_summary_valid(bad_summary, cuda_visible_devices=visible, local_rank=0, local_gpu_uuid=GPU_UUIDS[0])
    single = _module(ROOT / "ops" / "stage1" / "run_s1_9_single_bf16_worker.py", "s19_single_environment")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", GPU_UUIDS[0]); monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8"); monkeypatch.setenv("PYTHONHASHSEED", "0")
    single._require_pre_cuda_policy(cuda_visible_devices=GPU_UUIDS[0])
    monkeypatch.setenv("PYTHONHASHSEED", "drift")
    with pytest.raises(SystemExit, match="PRE_CUDA_POLICY_INVALID"):
        single._require_pre_cuda_policy(cuda_visible_devices=GPU_UUIDS[0])


def test_s19_charts_have_replayable_identity_and_heatmap_projections(tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_formal_charts")
    identity_title = "U identity"
    identity_rows = [("linear.weight-0", 2.0, 0.5, False), ("linear.bias-0", 0.0, 0.0, True)]
    identity = tmp_path / "identity.csv"; identity_svg = tmp_path / "identity.svg"
    identity.write_text("coordinate,unclipped,clipped,expected,identity_residual,a_q,ratio_threshold,ratio,eligible_for_ratio\nlinear.weight-0,2,0.5,0.5,0,0.9,0.00009,1,true\nlinear.bias-0,0,0,0,0,0.9,0.00009,nan,false\n", encoding="utf-8")
    identity_svg.write_text(formal._identity_svg(identity_title, identity_rows), encoding="utf-8")
    assert formal._verify_identity_chart(identity, identity_svg, title=identity_title)
    heatmap = tmp_path / "heatmap.csv"; heatmap_svg = tmp_path / "heatmap.svg"
    values = [("linear", 1e-4, 2), ("head", 2e-5, 1)]
    heatmap.write_text("module_layer,max_abs_error,tensor_count\nlinear,0.0001,2\nhead,0.00002,1\n", encoding="utf-8")
    heatmap_svg.write_text(formal._heatmap_svg("heat", values), encoding="utf-8")
    assert formal._verify_heatmap_chart(heatmap, heatmap_svg, title="heat")


def test_s19_hash_and_process_failures_are_fail_closed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_formal_process")
    with pytest.raises(formal.Stage1S19FormalError, match="HASH_FIELD_ALREADY_PRESENT"):
        formal._with_hash({"artifact_hash": "0" * 64})
    kill_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(formal, "_process_group_pids", lambda _pgid: [12])
    monkeypatch.setattr(formal, "_child_tree", lambda *_args, **_kwargs: (_ for _ in ()).throw(formal.Stage1S19ManualInterventionRequired("unknown")))
    monkeypatch.setattr(formal.os, "killpg", lambda pgid, signal: kill_calls.append((pgid, signal)), raising=False)
    with pytest.raises(formal.Stage1S19ManualInterventionRequired):
        formal._verified_kill_process_group({"pid": 12, "pgid": 12, "start_ticks": "1", "parent_pid": 1, "uid": 1, "exe": "python", "cmdline_sha256": "0" * 64, "run_token": "a" * 64}, run_token="a" * 64)
    assert not kill_calls

    class BrokenLease:
        current_path = tmp_path / "current.json"
        def release(self, *, outcome: str) -> Path:
            raise RuntimeError(outcome)
        def close(self) -> None:
            self.closed = True
    lease = BrokenLease()
    with pytest.raises(formal.Stage1S19ManualInterventionRequired, match="LEASE_RELEASE_FAILED"):
        formal._release_lease_verified(lease, outcome="FAILED")
    assert lease.closed is True


def test_s19_launcher_gone_with_verified_child_escalates_only_after_full_pgid_recheck(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_launcher_gone")
    token = "a" * 64
    root = {"pid": 12, "pgid": 12, "start_ticks": "1", "parent_pid": 1, "uid": 1000, "exe": "/usr/bin/python", "cmdline_sha256": "0" * 64, "run_token": token}
    child = {"pid": 13, "pgid": 12, "start_ticks": "2", "parent_pid": 12, "uid": 1000, "exe": "/usr/bin/python", "cmdline_sha256": "1" * 64, "run_token": token}
    pgid_reads = iter(([12, 13], [13]))
    monkeypatch.setattr(formal, "_process_group_pids", lambda _pgid: list(next(pgid_reads)))
    trees = iter(([root, child], [child]))
    monkeypatch.setattr(formal, "_child_tree", lambda *_args, **_kwargs: list(next(trees)))
    monkeypatch.setattr(formal, "_run_token_processes", lambda _token: [13])
    monotonic = iter((0.0, 21.0))
    monkeypatch.setattr(formal.time, "monotonic", lambda: next(monotonic))
    monkeypatch.setattr(formal.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(formal, "_assert_process_group_empty", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(formal, "_assert_no_run_token_processes", lambda *_args, **_kwargs: None)
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(formal.signal, "SIGTERM", 15, raising=False)
    monkeypatch.setattr(formal.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(formal.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False)
    formal._verified_kill_process_group(root, run_token=token)
    assert signals == [(12, formal.signal.SIGTERM), (12, formal.signal.SIGKILL)]


def test_s19_formal_schema_validator_accepts_real_roles_and_rejects_nested_drift() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_formal_schema")
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="schema-test")
    formal._validate_s1_9_schemas(ROOT, evidence)
    drift = dict(evidence["trace_bundle"])
    drift_clip = dict(drift["clip"]); drift_clip["nested_unexpected"] = True; drift["clip"] = drift_clip
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:trace_bundle"):
        formal._validate_s1_9_schemas(ROOT, {"trace_bundle": drift})

    # Exercise nested full-state proof fields, not just a shallow role key.
    for mutate in (
        lambda item: item["scaler"]["engine_finite_skip_finite"]["skip_pre_state"]["cursor"].__setitem__("unexpected", True),
        lambda item: item["scaler"]["engine_finite_skip_finite"]["skip_pre_state"]["long_term_accumulator"].pop("raw"),
        lambda item: item["scaler"]["engine_finite_skip_finite"]["attempt_state_snapshots"].pop(),
    ):
        nested = json.loads(json.dumps(evidence["trace_bundle"]))
        mutate(nested)
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:trace_bundle"):
            formal._validate_s1_9_schemas(ROOT, {"trace_bundle": nested})


def test_s19_new_compatibility_schemas_are_recursively_closed() -> None:
    """Every new S1.9 compatibility schema is closed through nested arrays."""

    def walk(value: object) -> list[dict[str, object]]:
        if isinstance(value, dict):
            schema_keys = {"type", "$ref", "const", "enum", "allOf", "anyOf", "properties", "items", "prefixItems", "additionalProperties", "required"}
            found = [value] if set(value) & schema_keys else []
            for child in value.values():
                found.extend(walk(child))
            return found
        if isinstance(value, list):
            return [item for child in value for item in walk(child)]
        return []

    names = ("s1-9-upstream-compatibility-v5.json", "s1-9-upstream-compatibility-v6.json", "s1-9-upstream-compatibility-v7.json", "s1-9-gpu-quiescence-v3.json", "s1-9-gpu-prelease-v3.json", "s1-9-formalization-index-v6.json", "s1-9-formalization-index-v7.json", "s1-9-formalization-index-v8.json", "s1-9-single-bf16-worker-v2.json", "s1-9-validation-v2.json", "s1-9-bf16-checkpoint-store-reproduction-v2.json")
    for name in names:
        path = ROOT / "schemas" / "stage1" / name
        for schema in walk(json.loads(path.read_text(encoding="utf-8"))):
            assert schema != {}, path.name
            if "$ref" in schema:
                continue
            kind = schema.get("type")
            is_object = kind == "object" or isinstance(kind, list) and "object" in kind
            if is_object:
                additional = schema.get("additionalProperties")
                assert additional is False, path.name
            is_array = kind == "array" or isinstance(kind, list) and "array" in kind
            if is_array:
                assert isinstance(schema.get("items"), dict) or isinstance(schema.get("prefixItems"), list), path.name

    worker_v2 = json.loads((ROOT / "schemas" / "stage1" / "s1-9-single-bf16-worker-v2.json").read_text(encoding="utf-8"))
    validation_v2 = json.loads((ROOT / "schemas" / "stage1" / "s1-9-validation-v2.json").read_text(encoding="utf-8"))
    assert worker_v2["properties"]["observation"]["properties"]["determinism"]["properties"]["allowed_nondeterministic_kernel_classes"]["items"] == {"type": "string"}
    assert validation_v2["properties"]["regression"]["properties"]["kernel_allowlist"]["items"] == {"type": "string"}


def test_s19_runtime_validator_rejects_jointly_rehashed_nested_schema_drift() -> None:
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="runtime-schema-drift")
    trace = json.loads(json.dumps(evidence["trace_bundle"]))
    trace["scaler"]["engine_finite_skip_finite"]["skip_pre_state"]["cursor"]["unexpected"] = True
    trace.pop("trace_hash"); trace["trace_hash"] = canonical_json_hash(trace)
    report = dict(evidence["numeric_report"]); report["trace_hash"] = trace["trace_hash"]; report.pop("report_hash"); report["report_hash"] = canonical_json_hash(report)
    with pytest.raises(Stage1PrecisionError, match="RUNTIME_SCHEMA_VALIDATION_FAILED"):
        validate_stage1_s19_evidence({**evidence, "numeric_report": report, "trace_bundle": trace}, source_root=ROOT)


def test_s19_gate_replay_and_validation_schema_sets_are_exact() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_exact_schema_sets")
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="exact-schema-test")
    replay = replay_stage1_s19_evidence(evidence, source_root=ROOT)
    formal._validate_s1_9_schemas(ROOT, {"numeric_report": evidence["numeric_report"], "gate_record": evidence["gate_record"], "replay": replay})
    bad_gate = dict(evidence["gate_record"]); bad_gate["requirements"] = {**bad_gate["requirements"], "unexpected": True}
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:gate_record"):
        formal._validate_s1_9_schemas(ROOT, {"gate_record": bad_gate})
    bad_replay = dict(replay); bad_replay["replayed_roles"] = ["trace_bundle", "oracle_bundle", "comparison_table"]
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:replay"):
        formal._validate_s1_9_schemas(ROOT, {"replay": bad_replay})

    digest, commit = "a" * 64, "b" * 40
    s17 = {"s1_7_index_ref": "s1-7/index.json", "s1_7_index_sha256": digest, "s1_7_index_artifact_hash": digest, "s1_7_generator_commit": commit, "s1_7_gate_artifact_hash": digest, "s1_7_role_sha256": {name: digest for name in ("fixture_manifest", "single_gpu_report", "gradient_bundle", "comparison_table", "gate_record")}}
    s18 = {"s1_8_index_ref": "s1-8/index.json", "s1_8_index_sha256": digest, "s1_8_index_artifact_hash": digest, "s1_8_generator_commit": commit, "s1_8_gate_artifact_hash": digest, "s1_8_role_sha256": {name: digest for name in ("fixture_manifest", "ddp_report", "array_bundle", "comparison_table", "gate_record")}}
    formal_checks = ("bf16_worker_self_hash", "bf16_checkpoint_binding_and_tamper_negative_control", "ddp_worker_self_hash_and_all_direct_checks", "post_worker_gpu_preflight_clean", "strict_schema_missing_and_unknown_negative_controls")
    visible = ",".join(GPU_UUIDS)
    validation = {"schema_version": "stage1-s1-9-validation-v1", "status": "PASS", "gate_id": "G1-NUMERIC", "task_id": precision.TASK_ID, "execution_scope": "formal_server_gpu_single_and_ddp_skip", "fixture_id": precision_oracle.FIXTURE_ID, "producer_commit": commit, "consumer_commit": commit, "upstream": {"s1_7": s17, "s1_8": s18}, "regression": {"bf16": {"algorithms_enabled": True, "cudnn_deterministic": True, "cudnn_benchmark": False, "allowed_nondeterministic_kernel_classes": [], "kernel_policy": "empty_pre_registered_allowlist"}, "ddp_world_size": 4, "kernel_allowlist": [], "environment": {"single": _environment_summary(visible=GPU_UUIDS[0], rank=0, uuid=GPU_UUIDS[0]), "ddp_rank0": _environment_summary(visible=visible, rank=0, uuid=GPU_UUIDS[0]), "ddp_all_ranks": [_environment_summary(visible=visible, rank=rank, uuid=uuid) for rank, uuid in enumerate(GPU_UUIDS)]}}, "direct_checks": [{"check_id": check_id, "status": "PASS", "detail": "checked"} for check_id in (*precision.REQUIREMENT_KEYS, *formal_checks)], "role_sha256": {name: digest for name in ("numeric_report", "oracle_bundle", "trace_bundle", "comparison_table", "gate_record")}, "csv_sha256": {name: digest for name in ("bf16-fp32-heatmap.csv", "clip-norm-factor.csv", "skip-zero-difference.csv", "t-amp-scale.csv", "u-single-factor-identity.csv", "u-single-factor-ratio-diagnostic.csv")}, "svg_sha256": {name: digest for name in ("bf16-fp32-heatmap.svg", "clip-norm-factor.svg", "skip-zero-difference.svg", "t-amp-scale.svg", "u-single-factor-identity.svg", "u-single-factor-ratio-diagnostic.svg")}, "replay_sha256": digest, "replay_hash": digest, "artifact_hash": digest}
    formal._validate_s1_9_schemas(ROOT, {"validation": validation})
    index_schema = json.loads((ROOT / "schemas" / "stage1" / "s1-9-formalization-index-v8.json").read_text(encoding="utf-8"))
    v7_index_schema = json.loads((ROOT / "schemas" / "stage1" / "s1-9-formalization-index-v7.json").read_text(encoding="utf-8"))
    assert index_schema["properties"]["reproduction_role_refs"]["$ref"] == "#/$defs/reproduction_refs"
    assert v7_index_schema["properties"]["reproduction_role_refs"]["$ref"] == "s1-9-formalization-index-v6.json#/properties/reproduction_role_refs"
    reproduction = {
        name: definition["const"]
        for name, definition in index_schema["$defs"]["reproduction_refs"]["properties"].items()
    }
    assert len(reproduction) == formal._S1_9_REPRODUCTION_ROLE_COUNT == 27
    assert reproduction == formal._S1_9_REPRODUCTION_ROLE_REFS
    assert reproduction["prelease_gpu"] == "prelease-gpu.json"
    index = {
        "schema_version": "stage1-s1-9-formalization-index-v8", "status": "PASS",
        "gate_id": "G1-NUMERIC", "task_id": precision.TASK_ID,
        "fixture_id": precision_oracle.FIXTURE_ID, "generator_git_commit": commit,
        "consumer_git_commit": commit, "git_branch": "main",
        "checked_at": "2026-08-20T00:00:00+00:00", "s1_7_handoff": s17,
        "s1_8_handoff": s18,
        "role_refs": {"numeric_report": "numeric-report.json", "oracle_bundle": "oracle-bundle.json", "trace_bundle": "trace-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-numeric-record.json"},
        "role_sha256": {name: digest for name in ("numeric_report", "oracle_bundle", "trace_bundle", "comparison_table", "gate_record")},
        "reproduction_role_refs": reproduction,
        "reproduction_role_sha256": {name: digest for name in reproduction},
        "gate_artifact_hash": digest,
        "csv_sha256": {name: digest for name in ("bf16-fp32-heatmap.csv", "clip-norm-factor.csv", "skip-zero-difference.csv", "t-amp-scale.csv", "u-single-factor-identity.csv", "u-single-factor-ratio-diagnostic.csv")},
        "svg_sha256": {name: digest for name in ("bf16-fp32-heatmap.svg", "clip-norm-factor.svg", "skip-zero-difference.svg", "t-amp-scale.svg", "u-single-factor-identity.svg", "u-single-factor-ratio-diagnostic.svg")},
        "validation_ref": "validation.json", "validation_sha256": digest,
        "replay_ref": "replay-validation.json", "replay_sha256": digest,
        "replay_hash": digest, "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"],
    }
    index["artifact_hash"] = canonical_json_hash(index)
    formal._validate_s1_9_schemas(ROOT, {"index": index})
    same_count_key = json.loads(json.dumps(index)); refs = same_count_key["reproduction_role_refs"]; refs["unreviewed"] = refs.pop("preflight"); same_count_key.pop("artifact_hash"); same_count_key["artifact_hash"] = canonical_json_hash(same_count_key)
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:index"):
        formal._validate_s1_9_schemas(ROOT, {"index": same_count_key})
    old_index = json.loads(json.dumps(index)); old_index["schema_version"] = "stage1-s1-9-formalization-index-v7"; old_index.pop("artifact_hash"); old_index["artifact_hash"] = canonical_json_hash(old_index)
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:index"):
        formal._validate_s1_9_schemas(ROOT, {"index": old_index})
    missing_prelease = json.loads(json.dumps(index)); missing_prelease["reproduction_role_refs"].pop("prelease_gpu"); missing_prelease.pop("artifact_hash"); missing_prelease["artifact_hash"] = canonical_json_hash(missing_prelease)
    extra_prelease = json.loads(json.dumps(index)); extra_prelease["reproduction_role_sha256"]["unreviewed"] = digest; extra_prelease.pop("artifact_hash"); extra_prelease["artifact_hash"] = canonical_json_hash(extra_prelease)
    for mutated in (missing_prelease, extra_prelease):
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:index"):
            formal._validate_s1_9_schemas(ROOT, {"index": mutated})
    validation_bad_id = dict(validation); validation_bad_id["direct_checks"] = list(validation["direct_checks"]); validation_bad_id["direct_checks"][0] = {**validation_bad_id["direct_checks"][0], "check_id": "drift"}
    validation_missing = dict(validation); validation_missing["role_sha256"] = dict(validation["role_sha256"]); validation_missing["role_sha256"].pop("gate_record")
    validation_cardinality = dict(validation); validation_cardinality["direct_checks"] = list(validation["direct_checks"][:-1])
    for mutated in (validation_bad_id, validation_missing, validation_cardinality):
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:validation"):
            formal._validate_s1_9_schemas(ROOT, {"validation": mutated})


def test_s19_ddp_worker_does_not_use_all_visible_cuda_rng_apis() -> None:
    source = (ROOT / "ops" / "stage1" / "run_s1_9_ddp_skip_worker.py").read_text(encoding="utf-8")
    assert "torch.manual_seed(" not in source
    assert "manual_seed_all" not in source
    assert "get_rng_state_all" not in source
    assert "set_rng_state_all" not in source


def test_s19_s17_handoff_uses_the_real_r11_role_set_and_gate_semantics(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "data"; bundle = root / "synthetic"; bundle.mkdir(parents=True)
    producer = "a" * 40
    gate_hash = ""
    role_names = {"fixture_manifest", "single_gpu_report", "gradient_bundle", "comparison_table", "gate_record"}
    refs: dict[str, str] = {}; hashes: dict[str, str] = {}
    for role in sorted(role_names):
        body = {"schema_version": "synthetic", "status": "PASS"}
        if role == "gate_record":
            body.update({"requirements": {"all_required": True}})
            body["artifact_hash"] = canonical_json_hash(body)
            gate_hash = str(body["artifact_hash"])
        else:
            body["artifact_hash"] = canonical_json_hash(body)
        path = bundle / f"{role}.json"; write_canonical_json(path, body)
        refs[role] = f"synthetic/{path.name}"; hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    def auxiliary(name: str, hash_field: str) -> tuple[str, str]:
        body = {"schema_version": "synthetic", "status": "PASS"}; body[hash_field] = canonical_json_hash(body)
        path = bundle / f"{name}.json"; write_canonical_json(path, body)
        return f"synthetic/{path.name}", hashlib.sha256(path.read_bytes()).hexdigest()
    replay_ref, replay_sha = auxiliary("replay", "replay_hash")
    validation_ref, validation_sha = auxiliary("validation", "artifact_hash")
    index_body = {"schema_version": "stage1-s1-7-formalization-index-v1", "status": "PASS", "gate_id": "G1-SINGLE", "task_id": "stage1.07_single_gpu_pythia14m", "generator_git_commit": producer, "consumer_git_commit": producer, "gate_artifact_hash": gate_hash, "next_task_ids": ["stage1.08_ddp_and_gradient_accumulation", precision.TASK_ID], "role_refs": refs, "role_sha256": hashes, "replay_ref": replay_ref, "replay_sha256": replay_sha, "validation_ref": validation_ref, "validation_sha256": validation_sha}
    index_body["artifact_hash"] = canonical_json_hash(index_body)
    index_path = bundle / "index.json"; write_canonical_json(index_path, index_body)
    monkeypatch.setattr(precision, "EXPECTED_S1_7_INDEX_SHA256", hashlib.sha256(index_path.read_bytes()).hexdigest())
    monkeypatch.setattr(precision, "EXPECTED_S1_7_INDEX_ARTIFACT_HASH", index_body["artifact_hash"])
    monkeypatch.setattr(precision, "EXPECTED_S1_7_PRODUCER", producer)
    monkeypatch.setattr(precision, "EXPECTED_S1_7_GATE_HASH", gate_hash)
    frozen_calls: list[set[str]] = []
    monkeypatch.setattr(precision, "_validate_s1_7_frozen_schemas", lambda values: frozen_calls.append(set(values)))
    handoff = precision.validate_s1_7_handoff(root, "synthetic/index.json")
    assert set(handoff["s1_7_role_sha256"]) == role_names
    assert frozen_calls == [{"index", *role_names, "replay", "validation"}]


def test_s19_s17_handoff_frozen_schema_validation_is_fail_closed() -> None:
    values = {
        "index": {"unknown": True},
        "fixture_manifest": {},
        "single_gpu_report": {},
        "gradient_bundle": {},
        "comparison_table": {},
        "gate_record": {},
        "replay": {},
        "validation": {},
    }
    with pytest.raises(Stage1PrecisionError, match="S1_9_S1_7_FROZEN_SCHEMA_VALIDATION_FAILED"):
        precision._validate_s1_7_frozen_schemas(values)


def test_s19_s18_handoff_requires_pinned_roles_gate_and_replay_validation_closure(tmp_path: Path) -> None:
    root = tmp_path / "data"; bundle = root / "synthetic"; bundle.mkdir(parents=True)
    producer = "b" * 40
    role_names = {"fixture_manifest", "ddp_report", "array_bundle", "comparison_table", "gate_record"}
    refs: dict[str, str] = {}; hashes: dict[str, str] = {}
    gate_hash = ""
    for role in sorted(role_names):
        body = {"schema_version": "synthetic", "status": "PASS"}
        if role == "gate_record":
            body["requirements"] = {"all_required": True}
        body["artifact_hash"] = canonical_json_hash(body)
        if role == "gate_record":
            gate_hash = str(body["artifact_hash"])
        path = bundle / f"{role}.json"; write_canonical_json(path, body)
        refs[role] = f"synthetic/{path.name}"; hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()

    def auxiliary(name: str, hash_field: str) -> tuple[str, str]:
        body = {"schema_version": "synthetic", "status": "PASS"}; body[hash_field] = canonical_json_hash(body)
        path = bundle / f"{name}.json"; write_canonical_json(path, body)
        return f"synthetic/{path.name}", hashlib.sha256(path.read_bytes()).hexdigest()

    validation_ref, validation_sha = auxiliary("validation", "artifact_hash")
    # Match the real S1.8 replay envelope, which uses ``artifact_hash``.
    replay_ref, replay_sha = auxiliary("replay", "artifact_hash")
    index = {"schema_version": "stage1-s1-8-formalization-index-v1", "status": "PASS", "gate_id": "G1-DDP", "task_id": "stage1.08_ddp_and_gradient_accumulation", "generator_git_commit": producer, "consumer_git_commit": producer, "role_refs": refs, "role_sha256": hashes, "gate_artifact_hash": gate_hash, "validation_ref": validation_ref, "validation_sha256": validation_sha, "replay_ref": replay_ref, "replay_sha256": replay_sha}
    index["artifact_hash"] = canonical_json_hash(index)
    index_path = bundle / "index.json"; write_canonical_json(index_path, index)
    binding = {"index_sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(), "index_artifact_hash": str(index["artifact_hash"]), "gate_artifact_hash": gate_hash, "producer_commit": producer, "schema_version": "stage1-s1-8-formalization-index-v1", "task_id": "stage1.08_ddp_and_gradient_accumulation", "gate_id": "G1-DDP"}
    result = precision.validate_s1_8_handoff(root, "synthetic/index.json", expected_binding=binding)
    assert set(result["s1_8_role_sha256"]) == role_names

    gate = bundle / "gate_record.json"
    altered = json.loads(gate.read_text(encoding="utf-8")); altered["requirements"] = {"all_required": False}; altered.pop("artifact_hash"); altered["artifact_hash"] = canonical_json_hash(altered); write_canonical_json(gate, altered)
    with pytest.raises(Stage1PrecisionError, match="S1_9_S1_8_ROLE_HASH_INVALID:gate_record"):
        precision.validate_s1_8_handoff(root, "synthetic/index.json", expected_binding=binding)


def test_s19_s18_source_map_requires_sha256_map_and_retains_changed_path_keys() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_source_map")
    valid = {
        "source_map": {
            "src/param_importance_nlp/runtime/optimizer.py": "a" * 64,
            "src/param_importance_nlp/runtime/training.py": "b" * 64,
        }
    }
    observed = formal._source_map(valid, field="s1_8.ddp_report")
    assert set(observed) == {"src/param_importance_nlp/runtime/optimizer.py", "src/param_importance_nlp/runtime/training.py"}
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_UPSTREAM_SOURCE_MAP_MISSING"):
        formal._source_map({"source_map": {"src/runtime.py": "not-a-sha"}}, field="s1_8.ddp_report")


def test_s19_s17_r11_no_source_map_uses_frozen_global_diff_allowlist() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_s17_global_allowlist")
    producer = precision.EXPECTED_S1_7_PRODUCER
    attestation = formal._s1_7_shared_dependency_attestation(
        ROOT,
        {"source_git_commit": producer},
        producer_commit=producer,
        changed_paths=["worklogs/2026-08-15-stage1-s17.md", "src/param_importance_nlp/runtime/optimizer.py"],
    )
    assert attestation["source_map_mode"] == "r11_no_per_file_source_map_global_diff_allowlist"
    assert set(attestation["authorized_shared_dependency_hashes"]) == {"src/param_importance_nlp/runtime/optimizer.py"}
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_S1_7_SHARED_DEPENDENCY_DRIFT_REQUIRES_RERUN"):
        formal._s1_7_shared_dependency_attestation(
            ROOT,
            {"source_git_commit": producer},
            producer_commit=producer,
            changed_paths=["src/param_importance_nlp/runtime/training.py"],
        )


def test_s19_s18_source_map_intersection_is_empty_or_revalidated_for_the_one_shared_fix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_s18_intersection")
    producer = precision.EXPECTED_S1_7_PRODUCER
    path = "src/param_importance_nlp/runtime/optimizer.py"
    historical = formal._git_file_sha256(ROOT, producer, path)
    s17 = {"s1_7_generator_commit": producer}
    s18 = {"s1_8_generator_commit": producer}
    roles = {
        "single_gpu_report": {"source_git_commit": producer},
        "ddp_report": {"source_map": {"src/param_importance_nlp/runtime/training.py": "b" * 64}},
    }
    monkeypatch.setattr(formal, "_upstream_role", lambda _root, _ref, role: roles[role])
    monkeypatch.setattr(formal, "_s1_8_v8_handoff_attestation", lambda *_args: {"index_schema_version": "stage1-s1-8-formalization-index-v8", "ddp_report_schema_version": "stage1-s1-8-ddp-report-v8", "validation_schema_version": "stage1-s1-8-validation-v8", "implementation_source_sha256": roles["ddp_report"]["source_map"], "reproduction_role_refs": {name: "x" for name in ()}, "reproduction_role_sha256": {}, "gpu_quiescence": {}, "replay_schema_version": "stage1-s1-8-replay-validation-v3", "comparison_table_schema_version": "stage1-s1-8-comparison-table-v2", "array_bundle_schema_version": "stage1-s1-8-array-bundle-v2", "array_bundle_route_refs": formal._S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS})
    monkeypatch.setattr(formal, "_validate_s1_9_schemas", lambda *_args: None)
    monkeypatch.setattr(formal, "_consumer_diff", lambda _repository, _producer: [path])
    monkeypatch.setattr(formal, "_git", lambda *_args: "c" * 40)
    empty = formal._upstream_compatibility_attestation(ROOT, tmp_path, s1_7_ref="s17/index.json", s1_7=s17, s1_8_ref="s18/index.json", s1_8=s18)
    assert empty["s1_8_affected_dependencies"] == []
    roles["ddp_report"] = {"source_map": {path: historical}}
    allowed = formal._upstream_compatibility_attestation(ROOT, tmp_path, s1_7_ref="s17/index.json", s1_7=s17, s1_8_ref="s18/index.json", s1_8=s18)
    assert allowed["s1_8_authorized_shared_drift"][path]["compatibility_revalidated"] is True
    roles["ddp_report"] = {"source_map": {path: "0" * 64}}
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_S1_8_SOURCE_MAP_PRODUCER_HASH_MISMATCH"):
        formal._upstream_compatibility_attestation(ROOT, tmp_path, s1_7_ref="s17/index.json", s1_7=s17, s1_8_ref="s18/index.json", s1_8=s18)


def test_s19_current_source_clip_compatibility_replay_covers_order_and_none() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_clip_compatibility")
    replay = formal._optimizer_clip_cpu_replay()
    assert replay["passed"] is True
    assert replay["empty"] == [0.0, 1.0]
    assert replay["none_only"] == [0.0, 1.0]
    assert replay["observed_norm"] == pytest.approx(replay["reverse_observed_norm"], abs=1e-15, rel=0.0)


def test_s19_s110_compatibility_paths_are_exact_and_reject_extra_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_s110_exact_paths")
    approved = [
        "ops/stage1/formalize_s1_10.py",
        "schemas/stage1/s1-10-validation-v1.json",
        "src/param_importance_nlp/runtime/training.py",
        "src/param_importance_nlp/runtime/checkpoint_group.py",
        "tests/test_stage1_s110_checkpoint_resume.py",
    ]
    monkeypatch.setattr(formal, "_git", lambda *_args: "\n".join(approved))
    assert formal._consumer_diff(ROOT, "0" * 40) == approved
    monkeypatch.setattr(formal, "_git", lambda *_args: "\n".join([*approved, "schemas/stage1/s1-10-unreviewed-v1.json"]))
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_UPSTREAM_CONSUMER_DIFF_UNAUTHORIZED:schemas/stage1/s1-10-unreviewed-v1.json"):
        formal._consumer_diff(ROOT, "0" * 40)


def test_s19_consumer_paths_are_exact_and_reject_s18_or_s19_prefix_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_exact_consumer_paths")
    approved = sorted(formal._S1_9_FROZEN_CONSUMER_FILES)
    monkeypatch.setattr(formal, "_git", lambda *_args: "\n".join(approved))
    assert formal._consumer_diff(ROOT, "0" * 40) == approved
    for extra in (
        "schemas/stage1/s1-9-unreviewed-v1.json",
        "ops/stage1/formalize_s1_8.py",
        "fixtures/stage1/stage1-s18-unreviewed.json",
    ):
        monkeypatch.setattr(formal, "_git", lambda *_args, _extra=extra: "\n".join([*approved, _extra]))
        with pytest.raises(formal.Stage1S19FormalError, match="S1_9_UPSTREAM_CONSUMER_DIFF_UNAUTHORIZED:" + extra):
            formal._consumer_diff(ROOT, "0" * 40)


def test_s19_clean_producer_diff_accepts_only_s110_and_s19_then_rejects_s111(tmp_path: Path) -> None:
    """Exercise the actual git diff path, not only a mocked name list."""

    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_clean_consumer_diff")
    repository = tmp_path / "clean-consumer"; repository.mkdir()
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repository), *args], check=True, text=True, capture_output=True).stdout.strip()
    git("init"); git("config", "user.email", "s19-test@example.invalid"); git("config", "user.name", "S19 test")
    git("commit", "--allow-empty", "-m", "producer")
    producer = git("rev-parse", "HEAD")
    reviewed = {
        "src/param_importance_nlp/runtime/optimizer.py",
        *formal._S1_9_FROZEN_CONSUMER_FILES,
        *formal._S1_10_FROZEN_CONSUMER_FILES,
        *formal._S1_10_SHARED_RUNTIME_FILES,
    }
    for relative in reviewed:
        path = repository / relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text("reviewed\n", encoding="utf-8")
    git("add", "."); git("commit", "-m", "s110-and-s19-only")
    assert set(formal._consumer_diff(repository, producer)) == reviewed
    drift = repository / "schemas/stage1/s1-11-formalization-index-v1.json"; drift.parent.mkdir(parents=True, exist_ok=True); drift.write_text("unreviewed\n", encoding="utf-8")
    git("add", "."); git("commit", "-m", "s111-must-not-pass")
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_UPSTREAM_CONSUMER_DIFF_UNAUTHORIZED:schemas/stage1/s1-11-formalization-index-v1.json"):
        formal._consumer_diff(repository, producer)


def test_s19_upstream_v7_freezes_s18_v8_source_array_and_reproduction_closure() -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_upstream_v8_closure")
    digest, commit = "a" * 64, "b" * 40
    v8 = json.loads((ROOT / "schemas" / "stage1" / "s1-8-formalization-index-v8.json").read_text(encoding="utf-8"))
    ddp_v8 = json.loads((ROOT / "schemas" / "stage1" / "s1-8-ddp-report-v8.json").read_text(encoding="utf-8"))
    quiescence_v4 = json.loads((ROOT / "schemas" / "stage1" / "s1-8-gpu-quiescence-v4.json").read_text(encoding="utf-8"))
    source_map = {name: digest for name in v8["$defs"]["source_map"]["required"]}
    reproduction = {name: item["const"] for name, item in v8["$defs"]["reproduction_role_refs"]["properties"].items()}
    quiescence = {
        role: {"ref": ddp_v8["$defs"][f"{role}_binding"]["properties"]["ref"]["const"], "sha256": digest}
        for role in ("prelease", "post_worker", "post_release", "reacquire_preflight")
    }
    assert len(source_map) == 61
    assert len(reproduction) == 84
    assert quiescence_v4["properties"]["schema_version"]["const"] == "stage1-s1-8-gpu-quiescence-v4"
    result = {
        "schema_version": "stage1-s1-9-upstream-compatibility-v7", "status": "PASS", "s1_7_producer": commit, "s1_8_producer": commit, "consumer_commit": commit,
        "s1_7_to_consumer_changed_paths": ["src/param_importance_nlp/runtime/optimizer.py"], "s1_8_to_consumer_changed_paths": ["src/param_importance_nlp/runtime/checkpoint_group.py", "src/param_importance_nlp/runtime/training.py"],
        "s1_7_source_attestation": {"source_map_mode": "r11_no_per_file_source_map_global_diff_allowlist", "report_source_git_commit": commit, "authorized_shared_dependencies": ["src/param_importance_nlp/runtime/optimizer.py"], "authorized_shared_dependency_hashes": {"src/param_importance_nlp/runtime/optimizer.py": {"producer_sha256": digest, "consumer_sha256": digest, "changed": True}}},
        "s1_8_source_dependencies": source_map,
        "s1_8_v8_handoff": {"index_schema_version": "stage1-s1-8-formalization-index-v8", "ddp_report_schema_version": "stage1-s1-8-ddp-report-v8", "validation_schema_version": "stage1-s1-8-validation-v8", "implementation_source_sha256": source_map, "reproduction_role_refs": reproduction, "reproduction_role_sha256": {name: digest for name in reproduction}, "gpu_quiescence": quiescence, "replay_schema_version": "stage1-s1-8-replay-validation-v3", "comparison_table_schema_version": "stage1-s1-8-comparison-table-v2", "array_bundle_schema_version": "stage1-s1-8-array-bundle-v2", "array_bundle_route_refs": formal._S1_8_ARRAY_BUNDLE_V2_ROUTE_REFS},
        "s1_7_affected_dependencies": ["src/param_importance_nlp/runtime/optimizer.py"], "s1_8_affected_dependencies": [], "s1_8_authorized_shared_drift": {}, "authorized_shared_change": "src/param_importance_nlp/runtime/optimizer.py",
        "current_source_cpu_clip_replay": formal._optimizer_clip_cpu_replay(),
        "nonproducer_runtime_attestation": {"affected_paths": ["src/param_importance_nlp/runtime/checkpoint_group.py", "src/param_importance_nlp/runtime/training.py"], "s1_8_source_map_excludes_paths": ["src/param_importance_nlp/runtime/checkpoint_group.py", "src/param_importance_nlp/runtime/training.py"], "s1_7_oracle_training_import_isolated": True, "checkpoint_group_producer_math_exclusion": True, "current_source_cpu_replays": {"src/param_importance_nlp/runtime/training.py": {"profile": "s1_9_current_source_training_checkpoint_resume_cpu", "checkpoint_id": "synthetic", "checkpoint_schema_version": "training-checkpoint-state-v2", "omission_rejected_before_mutation": True, "fresh_engine_next_step_exact": True, "fresh_engine_final_state_exact": True, "passed": True}}},
    }
    result["artifact_hash"] = canonical_json_hash(result)
    formal._validate_s1_9_schemas(ROOT, {"upstream_compatibility": result})
    for mutate in (
        lambda item: item["s1_8_source_dependencies"].__setitem__("unreviewed.py", item["s1_8_source_dependencies"].pop(next(iter(item["s1_8_source_dependencies"]))),),
        lambda item: item["s1_8_v8_handoff"]["reproduction_role_refs"].__setitem__("unreviewed", item["s1_8_v8_handoff"]["reproduction_role_refs"].pop("prelease_gpu_quiescence")),
        lambda item: item["s1_8_v8_handoff"].__setitem__("replay_schema_version", "stage1-s1-8-replay-validation-v2"),
        lambda item: item["s1_8_v8_handoff"]["array_bundle_route_refs"]["D-rank_swap"].__setitem__("artifact_ref", "unreviewed-route.safetensors"),
    ):
        altered = json.loads(json.dumps(result)); mutate(altered); altered.pop("artifact_hash"); altered["artifact_hash"] = canonical_json_hash(altered)
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:upstream_compatibility"):
            formal._validate_s1_9_schemas(ROOT, {"upstream_compatibility": altered})
    old_wire = json.loads(json.dumps(result)); old_wire["schema_version"] = "stage1-s1-9-upstream-compatibility-v6"; old_wire["s1_8_v8_handoff"]["index_schema_version"] = "stage1-s1-8-formalization-index-v7"; old_wire.pop("artifact_hash"); old_wire["artifact_hash"] = canonical_json_hash(old_wire)
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:upstream_compatibility"):
        formal._validate_s1_9_schemas(ROOT, {"upstream_compatibility": old_wire})


def test_s19_s18_pre_v8_handoffs_are_rejected_before_consumption(tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_reject_s18_pre_v8")
    index = tmp_path / "s18" / "index.json"; index.parent.mkdir()
    for schema_version in ("stage1-s1-8-formalization-index-v3", "stage1-s1-8-formalization-index-v5", "stage1-s1-8-formalization-index-v6", "stage1-s1-8-formalization-index-v7"):
        write_canonical_json(index, {"schema_version": schema_version})
        with pytest.raises(formal.Stage1S19FormalError, match="S1_9_S1_8_V8_INDEX_REQUIRED"):
            formal._s1_8_v8_handoff_attestation(tmp_path, "s18/index.json", {})


def test_s19_shared_runtime_drift_fails_closed_without_current_source_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_runtime_drift_replay")
    monkeypatch.setattr(formal, "_git", lambda *_args: 'import builtins\ndef _oracle_replay(arrays):\n    blocked = ("param_importance_nlp.core.estimators", "param_importance_nlp.runtime.training", "param_importance_nlp.stage1_single_gpu")\n    original = builtins.__import__\n    def guarded(name):\n        if name.startswith(blocked):\n            raise ImportError("isolated")\n        return original(name)\n    try:\n        builtins.__import__ = guarded\n    finally:\n        builtins.__import__ = original\n')
    monkeypatch.setattr(formal, "_s1_9_checkpoint_resume_cpu_replay", lambda _path: {"passed": False})
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_SHARED_RUNTIME_DRIFT_REPLAY_FAILED"):
        formal._nonproducer_runtime_attestation(
            ROOT,
            s1_7_producer="0" * 40,
            s1_7_report={"status": "PASS"},
            s1_8_sources={},
            changed_paths=["src/param_importance_nlp/runtime/training.py"],
            replay_root=tmp_path / "replay",
        )


def test_s19_s17_oracle_isolation_requires_ast_guard_not_comment_or_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_oracle_ast_isolation")
    monkeypatch.setattr(
        formal,
        "_git",
        lambda *_args: '# param_importance_nlp.runtime.training is supposedly isolated\n'
        'def _oracle_replay(arrays):\n'
        '    note = "param_importance_nlp.runtime.training"\n'
        '    return arrays\n',
    )
    with pytest.raises(formal.Stage1S19FormalError, match="S1_9_S1_7_ORACLE_RUNTIME_ISOLATION_UNPROVEN"):
        formal._nonproducer_runtime_attestation(
            ROOT,
            s1_7_producer="0" * 40,
            s1_7_report={"status": "PASS"},
            s1_8_sources={},
            changed_paths=["src/param_importance_nlp/runtime/training.py"],
            replay_root=tmp_path / "replay",
        )


def test_s19_current_source_training_checkpoint_resume_compatibility_replay(tmp_path: Path) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_runtime_resume_replay")
    replay = formal._s1_9_checkpoint_resume_cpu_replay(tmp_path / "replay")
    assert replay["passed"] is True
    assert replay["omission_rejected_before_mutation"] is True
    assert replay["fresh_engine_next_step_exact"] is True


def test_s19_gpu_probe_combines_inventory_recovery_and_ignores_unselected_processes(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_gpu_recovery")
    inventory = "\n".join(f"{index}, {uuid}, NVIDIA A100, 81920, 0, 0, 40, 8.0, None" for index, uuid in enumerate(GPU_UUIDS))
    unselected = "GPU-ffffffff-1111-2222-3333-444444444444, 999, unrelated"
    calls: list[list[str]] = []
    def runner(command, **_kwargs):
        calls.append(command)
        text = inventory if "--query-gpu=" in command[1] else unselected
        return subprocess.CompletedProcess(command, 0, stdout=text)
    monkeypatch.setattr(formal, "_run", runner)
    observed = formal._gpu_probe_once(GPU_UUIDS)
    assert len(calls) == 2
    assert "gpu_recovery_action" in calls[0][1]
    assert [row["recovery_action"] for row in observed["selected"]] == ["None"] * 4
    assert observed["compute_apps"] == [{"gpu_uuid": "GPU-ffffffff-1111-2222-3333-444444444444", "pid": 999, "process_name": "unrelated"}]
    violations, hard = formal._gpu_violations(observed, minimum_compute_capability=8.0, max_temperature_c=85)
    assert not violations and not hard
    reset = inventory.replace(f"{GPU_UUIDS[1]}, NVIDIA A100, 81920, 0, 0, 40, 8.0, None", f"{GPU_UUIDS[1]}, NVIDIA A100, 81920, 0, 0, 40, 8.0, Reset")
    monkeypatch.setattr(formal, "_run", lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, stdout=reset if "--query-gpu=" in command[1] else ""))
    failed = formal._gpu_probe_once(GPU_UUIDS)
    _, hard = formal._gpu_violations(failed, minimum_compute_capability=8.0, max_temperature_c=85)
    assert hard[0].startswith("S1_9_GPU_RECOVERY_ACTION_NOT_NONE")
    assert [row["uuid"] for row in failed["selected"]] == list(GPU_UUIDS)


def test_s19_execute_persists_strict_prelease_gpu_failure_before_any_lease(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_execute_prelease_failure")
    import param_importance_nlp.runtime.operations as operations

    digest, commit = "a" * 64, "b" * 40
    selected = [{"physical_index": index, "uuid": uuid, "name": "NVIDIA A100", "memory_total_mib": 81920, "memory_used_mib": 0, "utilization_percent": 0, "temperature_c": 40, "compute_capability": "8.0", "recovery_action": "None"} for index, uuid in enumerate(GPU_UUIDS)]
    def quiescence(*, reason: str, probe_error: bool = False) -> dict[str, object]:
        sample = {"sample_index": 0, "observed_at": "2026-08-20T00:00:00+00:00", "monotonic_elapsed_seconds": 0.1, "requested_uuid_order": list(GPU_UUIDS), "probe_error": reason, "exact_selected_idle": False, "consecutive_exact_idle_samples": 0, "transient_observation_count": 0} if probe_error else {"sample_index": 0, "observed_at": "2026-08-20T00:00:00+00:00", "monotonic_elapsed_seconds": 0.1, "requested_uuid_order": list(GPU_UUIDS), "selected": selected, "compute_apps": [{"gpu_uuid": GPU_UUIDS[0], "pid": 73, "process_name": "python"}], "violations": [reason], "exact_selected_idle": False, "consecutive_exact_idle_samples": 0, "transient_observation_count": 0}
        return formal._with_hash({"schema_version": "stage1-s1-9-gpu-quiescence-v3", "status": "FAILED", "phase": "prelease", "approved_gpu_uuids": list(GPU_UUIDS), "started_at": "2026-08-20T00:00:00+00:00", "minimum_compute_capability": 8.0, "max_temperature_c": 85, "timeout_seconds": 180.0, "sample_interval_seconds": 1.0, "required_consecutive_exact_idle_samples": 3, "max_transient_samples": 2, "transient_observation_count": 0, "operational_timeout_basis": dict(formal._GPU_OPERATIONAL_TIMEOUT_BASIS), "samples": [sample], "final_gpu": None if probe_error else {"selected": selected, "requested_uuid_order": list(GPU_UUIDS), "compute_apps": sample["compute_apps"]}, "failure_reason": reason})
    calls = {"lease_constructor": 0}

    class ForbiddenLease:
        def __init__(self, *_args, **_kwargs) -> None:
            calls["lease_constructor"] += 1
            raise AssertionError("prelease failure must precede lease construction")

    monkeypatch.setattr(precision, "load_stage1_s19_fixture", lambda _repo: {"fixture_hash": digest})
    monkeypatch.setattr(precision, "validate_s1_7_handoff", lambda *_args: {"s1_7_generator_commit": commit})
    monkeypatch.setattr(precision, "validate_s1_8_handoff", lambda *_args, **_kwargs: {"s1_8_generator_commit": commit})
    monkeypatch.setattr(formal, "_upstream_compatibility_attestation", lambda *_args, **_kwargs: {"artifact_hash": digest})
    monkeypatch.setattr(formal, "_capability", lambda *_args, **_kwargs: {"artifact_hash": digest})
    monkeypatch.setattr(formal, "_git", lambda _repo, *args: commit if args == ("rev-parse", "HEAD") else "")
    monkeypatch.setattr(operations, "ProjectGpuLease", ForbiddenLease)

    def run_failure(attempt_id: str, record: object) -> Path:
        monkeypatch.setattr(formal, "_gpu_quiescence", lambda *_args, **_kwargs: record)
        with pytest.raises(formal.Stage1S19FormalError, match="S1_9_GPU_PRELEASE_FAILED"):
            formal.execute(repository=ROOT, data_root=tmp_path, s1_7_index_ref="s1-7/index.json", s1_8_index_ref="s1-8/index.json", s1_8_binding={"index_sha256": digest, "index_artifact_hash": digest, "gate_artifact_hash": digest, "producer_commit": commit, "schema_version": "stage1-s1-8-formalization-index-v8", "task_id": "stage1.08_ddp_and_gradient_accumulation", "gate_id": "G1-DDP"}, gpu_capability_ref="capability.json", capability_binding={"task_id": "x", "artifact_kind": "x", "artifact_hash": digest, "config_hash": digest}, approved_gpu_uuids=GPU_UUIDS, attempt_id=attempt_id, lease_owner="test")
        return tmp_path / "tmp" / "stage1-s1-9" / commit / attempt_id / "prelease-gpu.json"

    hard_path = run_failure("prelease-hard", quiescence(reason="S1_9_GPU_COMPUTE_PROCESS_PRESENT:" + GPU_UUIDS[0] + ":73"))
    hard_evidence = json.loads(hard_path.read_text(encoding="utf-8"))
    assert hard_evidence["status"] == "FAILED"
    assert hard_evidence["quiescence"]["samples"][0]["selected"] == selected
    assert hard_evidence["quiescence"]["samples"][0]["compute_apps"][0]["pid"] == 73
    formal._validate_s1_9_schemas(ROOT, {"gpu_prelease": hard_evidence})
    timeout_path = run_failure("prelease-timeout", quiescence(reason="S1_9_GPU_PROBE_EXCEPTION:TimeoutExpired", probe_error=True))
    timeout_evidence = json.loads(timeout_path.read_text(encoding="utf-8"))
    assert timeout_evidence["quiescence"]["samples"][0]["probe_error"] == "S1_9_GPU_PROBE_EXCEPTION:TimeoutExpired"
    formal._validate_s1_9_schemas(ROOT, {"gpu_prelease": timeout_evidence})
    assert calls["lease_constructor"] == 0


def test_s19_gpu_quiescence_requires_three_exact_idle_samples_and_records_hard_failure(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    formal = _module(ROOT / "ops" / "stage1" / "formalize_s1_9.py", "s19_gpu_quiescence")
    def probe(*, busy: bool = False, selected_pid: bool = False) -> dict[str, object]:
        return {"selected": [{"physical_index": index, "uuid": uuid, "name": "NVIDIA A100", "memory_total_mib": 81920, "memory_used_mib": 1 if busy and index == 0 else 0, "utilization_percent": 0, "temperature_c": 40, "compute_capability": "8.0", "recovery_action": "None"} for index, uuid in enumerate(GPU_UUIDS)], "requested_uuid_order": list(GPU_UUIDS), "compute_apps": [{"gpu_uuid": GPU_UUIDS[0], "pid": 17, "process_name": "python"}] if selected_pid else []}

    busy, idle = probe(busy=True), probe()
    observations = iter((busy, idle, idle, idle))
    monkeypatch.setattr(formal, "_gpu_probe_once", lambda *_args, **_kwargs: next(observations))
    monkeypatch.setattr(formal.time, "sleep", lambda _seconds: None)
    quiescence = formal._gpu_quiescence(GPU_UUIDS, phase="post_worker", minimum_compute_capability=8.0, max_temperature_c=85)
    assert quiescence["status"] == "PASS"
    assert quiescence["timeout_seconds"] == 180.0
    assert quiescence["samples"][-1]["consecutive_exact_idle_samples"] == 3
    formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": quiescence})
    late_observations = iter((idle, idle, idle))
    clock = iter((0.0, 0.1, 1.1, 180.0))
    real_monotonic = formal.time.monotonic
    monkeypatch.setattr(formal, "_gpu_probe_once", lambda *_args, **_kwargs: next(late_observations))
    monkeypatch.setattr(formal.time, "monotonic", lambda: next(clock))
    late = formal._gpu_quiescence(GPU_UUIDS, phase="post_worker", minimum_compute_capability=8.0, max_temperature_c=85)
    assert late["status"] == "FAILED"
    assert late["failure_reason"] == "S1_9_GPU_QUIESCENCE_TIMEOUT"
    assert len(late["samples"]) == 3
    monkeypatch.setattr(formal.time, "monotonic", real_monotonic)
    monkeypatch.setattr(formal, "_gpu_probe_once", lambda *_args, **_kwargs: probe(selected_pid=True))
    failed = formal._gpu_quiescence(GPU_UUIDS, phase="post_worker", minimum_compute_capability=8.0, max_temperature_c=85)
    assert failed["status"] == "FAILED"
    assert failed["failure_reason"].startswith("S1_9_GPU_COMPUTE_PROCESS_PRESENT")
    assert failed["samples"][-1]["compute_apps"][0]["pid"] == 17
    formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": failed})
    transient_observations = iter((busy, busy, busy))
    monkeypatch.setattr(formal, "_gpu_probe_once", lambda *_args, **_kwargs: next(transient_observations))
    transient_limit = formal._gpu_quiescence(GPU_UUIDS, phase="post_worker", minimum_compute_capability=8.0, max_temperature_c=85)
    assert transient_limit["status"] == "FAILED"
    assert transient_limit["failure_reason"] == "S1_9_GPU_QUIESCENCE_TRANSIENT_LIMIT"
    assert transient_limit["transient_observation_count"] == 3
    assert len(transient_limit["samples"]) == 3
    formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": transient_limit})
    unproven_hard = json.loads(json.dumps(failed)); unproven_hard["samples"][-1]["compute_apps"] = [{"gpu_uuid": "GPU-ffffffff-1111-2222-3333-444444444444", "pid": 17, "process_name": "python"}]; unproven_hard.pop("artifact_hash"); unproven_hard["artifact_hash"] = canonical_json_hash(unproven_hard)
    with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
        formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": unproven_hard})

    for mutate in (
        lambda item: item["samples"][0]["selected"][0].__setitem__("unexpected", True),
        lambda item: item["samples"][0]["selected"][0].pop("recovery_action"),
    ):
        nested = json.loads(json.dumps(quiescence))
        mutate(nested)
        nested.pop("artifact_hash"); nested = formal._with_hash(nested)
        with pytest.raises(formal.Stage1S19FormalError, match="SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
            formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": nested})

    for error, reason in ((subprocess.TimeoutExpired(["nvidia-smi"], 1), "S1_9_GPU_PROBE_EXCEPTION:TimeoutExpired"), (OSError("nvidia-smi unavailable"), "S1_9_GPU_PROBE_EXCEPTION:OSError")):
        monkeypatch.setattr(formal, "_gpu_probe_once", lambda *_args, _error=error, **_kwargs: (_ for _ in ()).throw(_error))
        probe_error = formal._gpu_quiescence(GPU_UUIDS, phase="post_worker", minimum_compute_capability=8.0, max_temperature_c=85)
        assert probe_error["status"] == "FAILED"
        assert probe_error["failure_reason"] == reason
        assert probe_error["samples"][-1]["probe_error"] == reason
        formal._validate_s1_9_schemas(ROOT, {"gpu_quiescence": probe_error})


def test_s19_offline_replay_rejects_jointly_rehashed_report_and_gate() -> None:
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="replay-tamper")
    report = dict(evidence["numeric_report"]); requirements = dict(report["requirements"]); requirements["clip_factor_matches_analytic_oracle"] = False; report["requirements"] = requirements; report.pop("report_hash"); report["report_hash"] = canonical_json_hash(report)
    gate = dict(evidence["gate_record"]); gate["requirements"] = requirements; gate.pop("artifact_hash"); gate["artifact_hash"] = canonical_json_hash(gate)
    tampered = {**evidence, "numeric_report": report, "gate_record": gate}
    validate_stage1_s19_evidence(tampered, source_root=ROOT)
    with pytest.raises(Stage1PrecisionError, match="REPLAY_ROLE_MISMATCH:numeric_report"):
        replay_stage1_s19_evidence(tampered, source_root=ROOT)
