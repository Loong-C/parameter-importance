from __future__ import annotations

import importlib.util
import hashlib
import json
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
    first_ref = next(iter(deep_extra["bundle_files_sha256"]))
    deep_extra["bundle_files_sha256"][first_ref] = {"sha256": "0" * 64}
    missing = dict(index); missing.pop("bundle_manifest_sha256")
    escaped = json.loads(json.dumps(index))
    escaped["bundle_files_sha256"]["objects/authoritative/tensors/../escape"] = "0" * 64
    hash_map_drift = json.loads(json.dumps(index))
    hash_map_drift["bundle_files_sha256"][first_ref] = "f" * 64
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


def test_s19_schemas_prohibit_untyped_deep_object_maps() -> None:
    """S1.9 role schemas cannot hide semantic fields behind permissive maps."""

    def walk(value: object) -> list[dict[str, object]]:
        if isinstance(value, dict):
            found = [value]
            for child in value.values():
                found.extend(walk(child))
            return found
        if isinstance(value, list):
            return [item for child in value for item in walk(child)]
        return []

    for path in sorted((ROOT / "schemas" / "stage1").glob("s1-9-*.json")):
        for schema in walk(json.loads(path.read_text(encoding="utf-8"))):
            if "$ref" in schema:
                continue
            kind = schema.get("type")
            is_object = kind == "object" or isinstance(kind, list) and "object" in kind
            if is_object:
                additional = schema.get("additionalProperties")
                # Dynamic maps are allowed only when both key and value are
                # typed.  This is needed for the immutable checkpoint bundle
                # file-hash inventory, whose tensor filenames are content
                # addressed rather than a hidden free-form object.
                if isinstance(additional, dict):
                    assert isinstance(schema.get("propertyNames"), dict), path.name
                    assert "$ref" in additional or additional.get("type") is not None, path.name
                else:
                    assert additional is False, path.name


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
    index_schema = json.loads((ROOT / "schemas" / "stage1" / "s1-9-formalization-index-v1.json").read_text(encoding="utf-8"))
    reproduction = {
        name: definition["const"]
        for name, definition in index_schema["$defs"]["reproduction_refs"]["properties"].items()
    }
    index = {
        "schema_version": "stage1-s1-9-formalization-index-v1", "status": "PASS",
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


def test_s19_offline_replay_rejects_jointly_rehashed_report_and_gate() -> None:
    evidence = build_stage1_s19_evidence(ROOT, producer_commit="0" * 40, scope="replay-tamper")
    report = dict(evidence["numeric_report"]); requirements = dict(report["requirements"]); requirements["clip_factor_matches_analytic_oracle"] = False; report["requirements"] = requirements; report.pop("report_hash"); report["report_hash"] = canonical_json_hash(report)
    gate = dict(evidence["gate_record"]); gate["requirements"] = requirements; gate.pop("artifact_hash"); gate["artifact_hash"] = canonical_json_hash(gate)
    tampered = {**evidence, "numeric_report": report, "gate_record": gate}
    validate_stage1_s19_evidence(tampered, source_root=ROOT)
    with pytest.raises(Stage1PrecisionError, match="REPLAY_ROLE_MISMATCH:numeric_report"):
        replay_stage1_s19_evidence(tampered, source_root=ROOT)
