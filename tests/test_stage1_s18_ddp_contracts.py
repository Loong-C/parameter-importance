"""CPU regressions for the S1.8 route and FP64-replay contracts."""

from __future__ import annotations

import math
import importlib.util
import copy
import hashlib
import inspect
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import torch

from param_importance_nlp.stage1_ddp import (
    Stage1S18Error,
    build_fixture,
    learning_rate_map,
    validate_nccl_transport_environment,
    validate_worker_plan,
    validate_fixture,
)
from param_importance_nlp import stage1_ddp_oracle as oracle
from param_importance_nlp import stage1_ddp as ddp


def _formalizer() -> object:
    formalizer_path = Path("ops/stage1/formalize_s1_8.py")
    spec = importlib.util.spec_from_file_location("s18_formalizer_contract_test", formalizer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _worker_report_a(formalizer: object) -> dict[str, object]:
    manifest = formalizer._with_hash({"schema_version": "stage1-s1-8-safetensors-manifest-v1", "file": "route-A.safetensors", "file_sha256": "a" * 64, "file_size_bytes": 4, "tensors": {"a-reference/equal/raw_core/p": {"sha256": "b" * 64, "dtype": "torch.float32", "shape": [1]}}})
    def case(name: str) -> dict[str, object]:
        return {"case": name, "global_loss_numerator": 1.0, "global_loss_valid_token_count": 8, "global_mean_loss": 0.125, "global_microbatch_count": 8, "global_n1": 8, "global_n2": 8, "global_gradient_norm": 1.0, "clip_factor": 1.0, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64] * 8, "global_statistic_checksums": {}, "local_loss_numerator": 1.0, "local_effective_tokens": 8}], "global_statistic_checksums": {}, "ordinary_ddp_gradient_collectives": 1 if name == "equal" else 2, "manual_statistic_collectives": {"backend": "nccl", "operation": "SUM", "tensor_statistics": [], "tensor_all_reduce_count": 0, "scalar_statistics": [], "scalar_all_reduce_count": 0, "total_all_reduce_count": 0}, "post_parameter_checksum": "d" * 64, "pre_parameter_checksum": "e" * 64, "accumulator": None, "array_keys": [f"a-reference/{name}/raw_core/p"]}
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    return formalizer._with_hash({"schema_version": "stage1-s1-8-worker-report-v1", "status": "PASS", "task_id": "stage1.08_ddp_and_gradient_accumulation", "execution_commit": "f" * 40, "run_token": "0" * 64, "route": "A", "permutation": "identity", "execution_mode": "formal", "world_size": 1, "backend": "nccl", "nccl_transport_protocol": formalizer._nccl_transport_protocol(), "visible_gpu_uuids": [uuid], "rank_to_gpu_uuid": [uuid], "parameter_registry_hash": "1" * 64, "fixture_hash": "2" * 64, "route_layout": {"route": "A", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [case("equal"), case("weighted")], "arrays": manifest})


def _fixture() -> dict[str, object]:
    tokens = {str(index): f"{index:x}".rjust(64, "a")[-64:] for index in range(16)}
    return build_fixture(
        token_sha256=tokens, upstream_fixture_hash="b" * 64,
        gradient_design_scale=2.5, optimizer_delta_design_scale=0.02, pre_route_scale_oracle_hash="c" * 64,
        pre_route_parameter_registry_hash="d" * 64,
        pre_route_case_state_checksums={"equal": "e" * 64, "weighted": "f" * 64},
    )


def test_fixture_binds_pre_route_gradient_scale_and_distinct_second_statistics() -> None:
    fixture = _fixture()
    checked = validate_fixture(fixture)
    scales = checked["comparison_natural_scales"]
    assert scales["s2"] == 8 * scales["n_g"] ** 2
    assert scales["g2"] == sum(value * value for value in checked["cases"]["weighted"]["effective_target_tokens"]) * scales["n_g"] ** 2
    assert scales["s2"] != scales["g2"]
    assert scales["optimizer_delta"] == 0.02
    assert checked["pre_route_gradient_scale"]["case_pre_parameter_checksums"] == {"equal": "e" * 64, "weighted": "f" * 64}


def test_fixture_rejects_unbound_or_post_hoc_scale_mutation() -> None:
    fixture = _fixture()
    fixture["comparison_natural_scales"]["g2"] = 1.0
    with pytest.raises(Stage1S18Error, match="S18_FIXTURE_HASH_INVALID"):
        validate_fixture(fixture)


def test_learning_rate_map_uses_parameter_identity_for_nonfirst_multi_parameter_group() -> None:
    class TwoParameter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Parameter(torch.ones(2, 2))
            self.second = torch.nn.Parameter(torch.ones(3, 2))

    model = TwoParameter()
    optimizer = torch.optim.AdamW([{"params": [model.second, model.first], "lr": 0.125}], foreach=False)
    assert learning_rate_map(model, optimizer) == {"first": 0.125, "second": 0.125}
    class DuplicateOptimizer:
        param_groups = [{"params": [model.first], "lr": 0.1}, {"params": [model.first, model.second], "lr": 0.1}]

    with pytest.raises(Stage1S18Error, match="S18_OPTIMIZER_PARAMETER_DUPLICATE"):
        learning_rate_map(model, DuplicateOptimizer())  # type: ignore[arg-type]


def test_route_array_serialization_snapshots_validate_cpu_fp32_layout_and_scalar_shape() -> None:
    scalar = torch.tensor(1.0, dtype=torch.float32)
    noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, 1]
    snapshots = ddp._route_array_serialization_snapshots({"optimizer-state/equal/p::step": scalar, "scores/equal/raw_score/p": noncontiguous})
    assert tuple(snapshots["optimizer-state/equal/p::step"].shape) == ()
    assert snapshots["scores/equal/raw_score/p"].is_contiguous()
    assert torch.equal(snapshots["scores/equal/raw_score/p"], noncontiguous)
    assert len({tensor.untyped_storage().data_ptr() for tensor in snapshots.values()}) == len(snapshots)
    invalid_values = (
        ({"x": torch.tensor([1.0], dtype=torch.float64)}, "S18_ROUTE_ARRAY_DTYPE_INVALID:x"),
        ({"x": torch.empty(1, device="meta", dtype=torch.float32)}, "S18_ROUTE_ARRAY_DEVICE_INVALID:x"),
        ({"x": torch.sparse_coo_tensor(indices=[[0]], values=[1.0], size=[1])}, "S18_ROUTE_ARRAY_LAYOUT_INVALID:x"),
        ({"x": torch.empty(0, dtype=torch.float32)}, "S18_ROUTE_ARRAY_SHAPE_INVALID:x"),
        ({"x": torch.tensor([float("nan")], dtype=torch.float32)}, "S18_ROUTE_ARRAY_NONFINITE:x"),
        ({"x": object()}, "S18_ROUTE_ARRAY_TENSOR_INVALID:x"),
    )
    for values, marker in invalid_values:
        with pytest.raises(Stage1S18Error, match=marker):
            ddp._route_array_serialization_snapshots(values)  # type: ignore[arg-type]


def test_route_array_safetensors_roundtrip_preserves_aliased_semantic_keys(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.torch")
    from safetensors.torch import load_file

    shared = torch.tensor([0.25, -0.5], dtype=torch.float32)
    values = {
        "scores/equal/raw_score/p": shared,
        "accumulator/equal/contribution/raw/p": shared,
        "scores/equal/raw_score_clipped/p": shared,
        "accumulator/equal/contribution/raw_clipped/p": shared,
        "optimizer-state/equal/p::step": torch.tensor(1.0, dtype=torch.float32),
    }
    snapshots = ddp._route_array_serialization_snapshots(values)
    assert len({tensor.untyped_storage().data_ptr() for tensor in snapshots.values()}) == len(values)
    path = tmp_path / "route-B.safetensors"
    manifest = ddp._save_route_arrays(path, values)
    loaded = load_file(str(path), device="cpu")
    assert set(loaded) == set(values) == set(manifest["tensors"])
    assert len({tensor.untyped_storage().data_ptr() for tensor in loaded.values()}) == len(values)
    for key, source in values.items():
        assert loaded[key].dtype == torch.float32
        assert tuple(loaded[key].shape) == tuple(source.shape)
        assert torch.equal(loaded[key], source)
        assert manifest["tensors"][key]["sha256"] == hashlib.sha256(ddp._tensor_bytes(source)).hexdigest()


def test_frozen_nccl_transport_requires_p2p_disabled_and_plan_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert ddp.NCCL_TRANSPORT_PROTOCOL == {
        "schema_version": "stage1-s1-8-nccl-transport-v1",
        "qualification_basis_gate_ids": ["stage0.G6", "stage0.G10"],
        "current_cuda_capability_artifact_hash": "a536e191cd59318325289d238db727f8939767e384bfccd961ae7ca1c6a11ce4",
        "nccl_p2p_disable": "1",
        "process_group_timeout_seconds": 180,
    }
    for invalid in (None, "0"):
        if invalid is None:
            monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
        else:
            monkeypatch.setenv("NCCL_P2P_DISABLE", invalid)
        with pytest.raises(Stage1S18Error, match="S18_NCCL_P2P_ENVIRONMENT_INVALID"):
            validate_nccl_transport_environment()
    monkeypatch.setenv("NCCL_P2P_DISABLE", "1")
    assert validate_nccl_transport_environment() == ddp.NCCL_TRANSPORT_PROTOCOL
    data_root, model_root, cache_root = tmp_path / "data", tmp_path / "data" / "models" / "model", tmp_path / "data" / "cache"
    model_root.mkdir(parents=True); cache_root.mkdir()
    tokens = tmp_path / "fixture-inputs.safetensors"; tokens.write_bytes(b"fixture")
    plan = ddp.with_artifact_hash({
        "schema_version": ddp.WORKER_PLAN_SCHEMA, "task_id": ddp.TASK_ID, "execution_commit": "a" * 40,
        "route": "A", "fixture": _fixture(), "fixture_tokens_ref": "fixture-inputs.safetensors",
        "fixture_tokens_sha256": hashlib.sha256(b"fixture").hexdigest(), "data_root": str(data_root),
        "model_root": str(model_root), "cache_root": str(cache_root), "run_token": "b" * 64,
        "visible_gpu_uuids": ["GPU-00000000-1111-2222-3333-444444444444"],
        "nccl_transport_protocol": dict(ddp.NCCL_TRANSPORT_PROTOCOL), "output_dir": "route-output",
        "permutation": "identity", "execution_mode": "formal",
    })
    assert validate_worker_plan(plan, path=tmp_path / "worker-plan.json")["nccl_transport_protocol"] == ddp.NCCL_TRANSPORT_PROTOCOL
    drifted = copy.deepcopy(plan); drifted["nccl_transport_protocol"]["nccl_p2p_disable"] = "0"; drifted["artifact_hash"] = ddp.canonical_json_hash({key: value for key, value in drifted.items() if key != "artifact_hash"})
    with pytest.raises(Stage1S18Error, match="S18_WORKER_NCCL_TRANSPORT_PROTOCOL_INVALID"):
        validate_worker_plan(drifted, path=tmp_path / "worker-plan.json")
    smoke_source = Path("ops/stage1/run_s1_8_nccl_smoke.py").read_text(encoding="utf-8")
    assert "validate_nccl_transport_environment()" in smoke_source and "timeout=timedelta(seconds=int(NCCL_TRANSPORT_PROTOCOL" in smoke_source
    formalizer_source = Path("ops/stage1/formalize_s1_8.py").read_text(encoding="utf-8")
    assert formalizer_source.count('"NCCL_P2P_DISABLE": "1"') == 2


def test_rank_local_rng_never_uses_all_visible_cuda_generators(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        random=SimpleNamespace(default_generator=SimpleNamespace(manual_seed=lambda seed: calls.append(("cpu", seed)))),
        cuda=SimpleNamespace(manual_seed=lambda seed: calls.append(("current_cuda", seed))),
    )
    monkeypatch.setattr(ddp, "torch", fake_torch)
    assert ddp.seed_rank_local_generators(seed=1707, local_rank=3) == {"cpu_default_seed": 1707, "cuda_current_device_seed": 1707, "local_rank": 3}
    assert calls == [("cpu", 1707), ("current_cuda", 1707)]
    source = Path("src/param_importance_nlp/stage1_ddp.py").read_text(encoding="utf-8")
    assert "torch.manual_seed(" not in source and "manual_seed_all" not in source
    with pytest.raises(Stage1S18Error, match="S18_RANK_LOCAL_SEED_INVALID"):
        ddp.seed_rank_local_generators(seed=-1, local_rank=0)


def test_fp64_stable_norm_handles_large_and_small_values_without_square_overflow() -> None:
    large = torch.tensor([1.0e200, -1.0e200], dtype=torch.float64)
    small = torch.tensor([1.0e-200, -1.0e-200], dtype=torch.float64)
    assert math.isfinite(oracle._stable_l2((large,)))
    assert oracle._stable_l2((large,)) == pytest.approx(math.sqrt(2.0) * 1.0e200)
    assert oracle._stable_l2((small,)) == pytest.approx(math.sqrt(2.0) * 1.0e-200)


def test_comparator_requires_fp64_reference_and_reports_dtypes() -> None:
    candidate = {"p": torch.tensor([1.0, -2.0], dtype=torch.float32)}
    reference = {"p": candidate["p"].to(torch.float64)}
    check = oracle.compare_maps(candidate, reference, natural_scale=1.0, atol=1e-7, rtol=1e-4, normalized_l2_limit=1e-4)
    assert check["within_t32_distributed"] is True
    assert check["per_tensor"]["p"]["candidate_dtype"] == "torch.float32"
    assert check["per_tensor"]["p"]["reference_dtype"] == "torch.float64"
    with pytest.raises(oracle.Stage1S18OracleError, match="S18_ORACLE_COMPARE_TENSOR_INVALID"):
        oracle.compare_maps(candidate, candidate, natural_scale=1.0, atol=1e-7, rtol=1e-4, normalized_l2_limit=1e-4)


def test_rank_contract_requires_exact_partition_checksum_and_no_sync_zero() -> None:
    checksum = {"s1": "a" * 64, "s2": "b" * 64}
    row = {
        "rank_records": [
            {"rank": 0, "local_microbatch_ids": [0, 1, 2, 3], "local_gradient_checksums": ["c" * 64] * 4, "global_statistic_checksums": checksum, "local_loss_numerator": 3.0, "local_effective_tokens": 8192},
            {"rank": 1, "local_microbatch_ids": [4, 5, 6, 7], "local_gradient_checksums": ["d" * 64] * 4, "global_statistic_checksums": checksum, "local_loss_numerator": 4.0, "local_effective_tokens": 8192},
        ],
        "global_statistic_checksums": checksum,
        "ordinary_ddp_gradient_collectives": 0,
        "global_loss_valid_token_count": 16384,
        "global_loss_numerator": 7.0,
    }
    report = {"route_layout": {"rank_microbatch_ids": [[0, 1, 2, 3], [4, 5, 6, 7]]}}
    oracle._validate_rank_contract(route="C", report=report, row=row, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})
    row["ordinary_ddp_gradient_collectives"] = 1
    with pytest.raises(oracle.Stage1S18OracleError, match="S18_ORACLE_NO_SYNC_COLLECTIVE_DRIFT"):
        oracle._validate_rank_contract(route="C", report=report, row=row, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})


def test_formal_chart_exports_have_five_csv_data_projections_and_no_a_u_reference(tmp_path: Path) -> None:
    formalizer = _formalizer()
    replay = {"comparison_rows": [{"comparison": "equal:A:a_reference:raw_core", "parameter": "p", "max_abs_error": 0.0, "max_scaled_error": 0.25, "normalized_l2_error": 0.1, "candidate_dtype": "torch.float32", "reference_dtype": "torch.float64", "within_t32_distributed": True}]}
    arrays = {}
    for route in ("A", "B", "C", "D"):
        prefix = "a-reference/equal" if route == "A" else "scores/equal"
        arrays[route] = {f"{prefix}/mean_gradient/p": torch.tensor([0.5]), f"{prefix}/raw_core/p": torch.tensor([0.25]), "pre/equal/p": torch.tensor([1.0]), "post/weighted/p": torch.tensor([0.9])}
        if route != "A": arrays[route]["accumulator/weighted/cumulative/data_movement/p"] = torch.tensor([0.1])
    row = {"case": "equal", "rank_records": [{"rank": 0, "local_microbatch_ids": [0], "local_effective_tokens": 2048}], "global_microbatch_count": 8, "global_n1": 16384, "global_n2": 33554432}
    ddp = {"baseline_routes": {route: {"cases": [row]} for route in arrays}}
    csv_hashes, svg_hashes = formalizer._charts(tmp_path, replay, ddp, arrays)
    assert len(csv_hashes) == len(svg_hashes) == 5
    for svg in svg_hashes:
        text = (tmp_path / svg).read_text(encoding="utf-8")
        assert "data-source=" in text and "data-value=" in text
        assert "A U" not in text


def test_strict_fixture_and_gate_schemas_reject_nested_unknown_missing_and_cardinality() -> None:
    formalizer = _formalizer()
    fixture = _fixture()
    formalizer._validate_output_schemas(Path("."), {"fixture_manifest": fixture})
    unknown = copy.deepcopy(fixture); unknown["precision"]["surprise"] = 1  # type: ignore[index]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"fixture_manifest": unknown})
    missing = copy.deepcopy(fixture); del missing["routes"]["D"]  # type: ignore[index]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"fixture_manifest": missing})
    requirements = {name: True for name in formalizer.GATE_CHECK_IDS}
    gate = formalizer._with_hash({"schema_version": "stage1-s1-8-gate-record-v1", "status": "PASS", "gate_id": "G1-DDP", "task_id": "stage1.08_ddp_and_gradient_accumulation", "requirements": requirements})
    formalizer._validate_output_schemas(Path("."), {"gate_record": gate})
    gate_unknown = copy.deepcopy(gate); gate_unknown["requirements"]["unexpected"] = True  # type: ignore[index]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"gate_record": gate_unknown})
    gate_missing = copy.deepcopy(gate); del gate_missing["requirements"]["nccl_smoke"]  # type: ignore[index]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"gate_record": gate_missing})


def test_worker_and_array_schemas_validate_real_shape_then_reject_deep_unknown_and_case_cardinality() -> None:
    formalizer = _formalizer()
    report = _worker_report_a(formalizer)
    formalizer._validate_output_schemas(Path("."), {"worker_report": report, "safetensors_manifest": report["arrays"]})
    scalar_step = copy.deepcopy(report)
    scalar_step["arrays"]["tensors"]["optimizer-state/equal/embed_out.weight::step"] = {"sha256": "d" * 64, "dtype": "torch.float32", "shape": []}
    for row in scalar_step["cases"]:
        row["array_keys"] = [f"a-reference/{row['case']}/{field}/p" for field in ("mean_gradient", "raw_core", "raw_score", "raw_score_clipped", "data_update", "magnitude")]
    scalar_step["arrays"] = formalizer._with_hash({key: value for key, value in scalar_step["arrays"].items() if key != "artifact_hash"})
    scalar_step["artifact_hash"] = formalizer._canonical({key: value for key, value in scalar_step.items() if key != "artifact_hash"})
    formalizer._validate_output_schemas(Path("."), {"worker_report": scalar_step, "safetensors_manifest": scalar_step["arrays"]})
    formalizer._validate_worker_candidate_contract("A", scalar_step)
    bad_scalar_shape = copy.deepcopy(scalar_step); bad_scalar_shape["arrays"]["tensors"]["optimizer-state/equal/embed_out.weight::step"]["shape"] = [0]
    bad_scalar_shape["arrays"] = formalizer._with_hash({key: value for key, value in bad_scalar_shape["arrays"].items() if key != "artifact_hash"})
    bad_scalar_shape["artifact_hash"] = formalizer._canonical({key: value for key, value in bad_scalar_shape.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": bad_scalar_shape})
    bad_scalar_descriptor = copy.deepcopy(scalar_step); bad_scalar_descriptor["arrays"]["tensors"]["optimizer-state/equal/embed_out.weight::step"] = {"sha256": "d" * 64, "dtype": "torch.float64", "shape": [], "unexpected": True}
    bad_scalar_descriptor["arrays"] = formalizer._with_hash({key: value for key, value in bad_scalar_descriptor["arrays"].items() if key != "artifact_hash"})
    bad_scalar_descriptor["artifact_hash"] = formalizer._canonical({key: value for key, value in bad_scalar_descriptor.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": bad_scalar_descriptor})
    unknown = copy.deepcopy(report); unknown["cases"][0]["rank_records"][0]["intruder"] = 1
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": unknown})
    cardinality = copy.deepcopy(report); cardinality["cases"] = cardinality["cases"][:1]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": cardinality})
    transport = copy.deepcopy(report); transport["nccl_transport_protocol"]["nccl_p2p_disable"] = "0"; transport["artifact_hash"] = formalizer._canonical({key: value for key, value in transport.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": transport})
    out_of_range = copy.deepcopy(report); out_of_range["cases"][1]["ordinary_ddp_gradient_collectives"] = 3; out_of_range["artifact_hash"] = formalizer._canonical({key: value for key, value in out_of_range.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": out_of_range})


def test_unknown_process_group_member_never_signals_and_persists_manual_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    expected = {"pid": 12, "ppid": 1, "uid": 1000, "pgid": 44, "sid": 44, "start_ticks": "99", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda _expected, **_kwargs: (_ for _ in ()).throw(formalizer.Stage1S18ManualInterventionRequired("S18_PROCESS_GROUP_MEMBER_UNVERIFIABLE")))
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda _expected, **_kwargs: {"session_members": [], "token_members": []})
    monkeypatch.setattr(formalizer.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False)
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired):
        formalizer._terminate_exact(SimpleNamespace(wait=lambda timeout: None), expected, tmp_path, label="unknown-member")
    assert signals == []
    assert (tmp_path / "unknown-member-manual-intervention.json").is_file()


def test_parent_fingerprint_binds_planned_token_without_claiming_late_environment_inheritance(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); base = {"pid": 11, "ppid": 1, "uid": 1000, "pgid": 11, "sid": 11, "start_ticks": "99", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64}
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: {**base, "pid": pid})
    planned = "b" * 64
    parent = formalizer._parent_fingerprint(11, planned)
    assert parent == {**base, "planned_run_token": planned, "token_inherited_at_exec": False}
    assert "environment_run_token" not in parent
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_SHA256_INVALID:parent.planned_run_token"):
        formalizer._parent_fingerprint(11, "not-a-token")
    source = Path("ops/stage1/formalize_s1_8.py").read_text(encoding="utf-8")
    assert '"parent_fingerprint": _parent_fingerprint(os.getpid(), run_token)' in source
    assert 'os.environ["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = run_token' not in source


def test_child_fingerprint_remains_strict_for_missing_or_wrong_inherited_token(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); base = {"pid": 12, "ppid": 1, "uid": 1000, "pgid": 12, "sid": 12, "start_ticks": "99", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64}
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: {**base, "pid": pid})
    token = "b" * 64
    for environment in ([b"PATH=/usr/bin"], [b"PARAM_IMPORTANCE_S18_RUN_TOKEN=" + (b"c" * 64)]):
        monkeypatch.setattr(formalizer, "_process_environment", lambda pid, entries=environment: entries)
        with pytest.raises(ProcessLookupError):
            formalizer._fingerprint(12, token)
    monkeypatch.setattr(formalizer, "_process_environment", lambda pid: [b"PARAM_IMPORTANCE_S18_RUN_TOKEN=" + token.encode("ascii")])
    assert formalizer._fingerprint(12, token) == {**base, "environment_run_token": token}


def test_torchrun_endpoint_is_nonzero_loopback_and_never_reused() -> None:
    formalizer = _formalizer()
    command = [
        "python", "-m", "torch.distributed.run", "--rdzv-id", "a" * 64,
        "--rdzv-endpoint", "127.0.0.1:0", "--nproc_per_node", "4", "worker.py",
    ]
    first_command, first_reservation, first_endpoint = formalizer._prepare_rendezvous_command(command, run_token="a" * 64)
    second_command, second_reservation, second_endpoint = formalizer._prepare_rendezvous_command(command, run_token="a" * 64)
    try:
        assert first_reservation is not None and second_reservation is not None
        assert formalizer._is_nonzero_loopback_endpoint(first_endpoint)
        assert formalizer._is_nonzero_loopback_endpoint(second_endpoint)
        assert first_endpoint != second_endpoint
        assert first_command[first_command.index("--rdzv-endpoint") + 1] == first_endpoint
        assert second_command[second_command.index("--rdzv-endpoint") + 1] == second_endpoint
    finally:
        if first_reservation is not None:
            first_reservation.close()
        if second_reservation is not None:
            second_reservation.close()


def test_independent_worker_sessions_are_discovered_by_token_and_historical_ancestry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def fingerprint(pid: int, requested: str) -> dict[str, object]:
        assert requested == token
        values = {
            100: {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10"},
            101: {"pid": 101, "ppid": 100, "uid": 7, "pgid": 101, "sid": 201, "start_ticks": "11"},
            102: {"pid": 102, "ppid": 101, "uid": 7, "pgid": 102, "sid": 202, "start_ticks": "12"},
        }
        return {**values[pid], "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    expected = fingerprint(100, token)
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101, 102])
    monkeypatch.setattr(formalizer, "_fingerprint", fingerprint)
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: {100: [100], 201: [101], 202: [102]}[sid])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: fingerprint(pid, token)[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    audit = formalizer._audit_exact_process_group(expected, known_members={100: expected})
    assert audit["member_pids"] == [100, 101, 102]
    assert audit["ancestry_depths"] == {"100": 0, "101": 1, "102": 2}
    assert [member["sid"] for member in audit["members"]] == [100, 201, 202]
    signalled: list[int] = []
    monkeypatch.setattr(formalizer, "_signal_exact_member", lambda member, _signal: signalled.append(member["pid"]) or True)
    assert formalizer._signal_exact_tree(audit, 15) == [102, 101, 100]
    assert signalled == [102, 101, 100]


def test_session_member_stat_parser_handles_arbitrary_comm_and_rejects_malformed() -> None:
    formalizer = _formalizer()
    fields = ["S", "1", "456", "789", *(["0"] * 15), "12345"]
    parsed = formalizer._parse_session_member_stat("123 (worker ) has spaces) " + " ".join(fields), pid=123, uid=7)
    assert parsed == {"pid": 123, "uid": 7, "pgid": 456, "sid": 789, "state": "S", "start_ticks": "12345"}
    for malformed in ("123 worker) S 1 456 789", "124 (worker) S 1 456 789", "123 (worker S 1 456 789", "123 (worker) SS 1 456 789 " + " ".join(["0"] * 16)):
        with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE"):
            formalizer._parse_session_member_stat(malformed, pid=123, uid=7)


def test_counting_allreduce_hook_has_exact_runtime_ddp_annotations_and_calls_real_hook() -> None:
    from torch import distributed as dist
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
    from torch.nn.parallel import DistributedDataParallel

    state = {"calls": 0}
    bucket, result = object(), object()
    calls: list[tuple[object, object]] = []
    hook = ddp._counting_allreduce_hook(
        dist=dist,
        allreduce_hook=lambda process_group, received_bucket: calls.append((process_group, received_bucket)) or result,
    )
    signature = inspect.signature(hook)
    assert inspect.signature(allreduce_hook).return_annotation == torch.futures.Future[torch.Tensor]
    assert hook.__annotations__ == {
        "bucket": dist.GradBucket,
        "return": torch.futures.Future[torch.Tensor],
    }
    assert signature.parameters["bucket"].annotation is dist.GradBucket
    assert signature.return_annotation == torch.futures.Future[torch.Tensor]

    class Checker:
        def _log_and_throw(self, error_type: type[BaseException], message: str) -> None:
            raise error_type(message)

    DistributedDataParallel._check_comm_hook(Checker(), hook)
    assert hook(state, bucket) is result
    assert state == {"calls": 1}
    assert calls == [(None, bucket)]


def test_unknown_member_in_worker_session_blocks_without_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: base if pid == 100 else worker if pid == 101 else (_ for _ in ()).throw(ProcessLookupError(pid)))
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 999])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {"pid": pid, "uid": 7, "pgid": pid, "sid": 100 if pid == 100 else 201, "start_ticks": "10" if pid == 100 else "11", "state": "S"})
    outsider = {"pid": 999, "ppid": 77, "uid": 7, "pgid": 999, "sid": 201, "start_ticks": "11", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64}
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: outsider if pid == 999 else (_ for _ in ()).throw(ProcessLookupError(pid)))
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_CANDIDATE_ANCESTRY_DRIFT"):
        formalizer._audit_exact_process_group(base, known_members={100: base})
    assert sleeps == []


def test_new_session_member_token_scan_race_requires_exact_token_recovery(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, pgid: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pgid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, worker, candidate = member(100, 1, 100, 100, 10), member(101, 100, 101, 201, 11), member(102, 100, 102, 201, 12)
    records = {100: launcher, 101: worker, 102: candidate}
    token_passes = [[100, 101], [100, 101, 102]]
    session_passes = {100: [[100], [100]], 201: [[101, 102], [101, 102]]}
    candidate_calls = [ProcessLookupError(102), candidate, candidate]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    def fingerprint(pid: int, _token: str) -> dict[str, object]:
        if pid == 102:
            next_value = candidate_calls.pop(0)
            if isinstance(next_value, BaseException):
                raise next_value
            return next_value
        return records[pid]
    monkeypatch.setattr(formalizer, "_fingerprint", fingerprint)
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    known = {100: launcher}
    audit = formalizer._audit_exact_process_group(launcher, known_members=known)
    assert audit["member_pids"] == [100, 101, 102]
    assert audit["ancestry_depths"] == {"100": 0, "101": 1, "102": 1}
    assert known == {100: launcher}
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_two_new_same_session_members_recover_together_after_one_token_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, pgid: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pgid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, worker, first, second = member(100, 1, 100, 100, 10), member(101, 100, 101, 201, 11), member(102, 100, 102, 201, 12), member(103, 100, 103, 201, 13)
    records = {100: launcher, 101: worker, 102: first, 103: second}
    token_passes = [[100, 101], [100, 101, 102, 103]]
    session_passes = {100: [[100], [100]], 201: [[101, 102, 103], [101, 102, 103]]}
    first_calls = [ProcessLookupError(102), first, first]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    def fingerprint(pid: int, _token: str) -> dict[str, object]:
        if pid == 102:
            result = first_calls.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        return records[pid]
    monkeypatch.setattr(formalizer, "_fingerprint", fingerprint)
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert audit["member_pids"] == [100, 101, 102, 103]
    assert audit["ancestry_depths"] == {"100": 0, "101": 1, "102": 1, "103": 1}
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_two_new_same_session_members_require_both_to_recover_token(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, pgid: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pgid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, worker, first, second = member(100, 1, 100, 100, 10), member(101, 100, 101, 201, 11), member(102, 100, 102, 201, 12), member(103, 100, 103, 201, 13)
    records = {100: launcher, 101: worker, 102: first, 103: second}
    token_passes = [[100, 101], [100, 101, 102]]
    session_passes = {100: [[100], [100]], 201: [[101, 102, 103], [101, 102, 103]]}
    first_calls = [ProcessLookupError(102), first, first]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    def fingerprint(pid: int, _token: str) -> dict[str, object]:
        if pid == 102:
            result = first_calls.pop(0)
            if isinstance(result, BaseException):
                raise result
            return result
        if pid == 103:
            raise ProcessLookupError(pid)
        return records[pid]
    monkeypatch.setattr(formalizer, "_fingerprint", fingerprint)
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_PROVISIONAL_TOKEN_MISSING"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_new_session_member_never_token_recovers_is_manual_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, pgid: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pgid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, worker, candidate = member(100, 1, 100, 100, 10), member(101, 100, 101, 201, 11), member(102, 100, 102, 201, 12)
    records = {100: launcher, 101: worker, 102: candidate}
    token_passes = [[100, 101], [100, 101]]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: records[pid] if pid != 102 else (_ for _ in ()).throw(ProcessLookupError(pid)))
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 102])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_PROVISIONAL_TOKEN_MISSING"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_new_session_member_provisional_identity_drift_is_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, pgid: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pgid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, worker, candidate = member(100, 1, 100, 100, 10), member(101, 100, 101, 201, 11), member(102, 100, 102, 201, 12)
    drifted_candidate = {**candidate, "start_ticks": "13"}
    records = {100: launcher, 101: worker, 102: candidate}
    token_passes = [[100, 101], [100, 101, 102]]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: records[pid] if pid != 102 else drifted_candidate)
    monkeypatch.setattr(formalizer, "_process_identity", lambda pid: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 102])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_PROVISIONAL_IDENTITY_DRIFT"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_known_zombie_session_member_is_excluded_without_promoting_new_pid(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    exited_worker = {**base, "pid": 102, "ppid": 100, "pgid": 102, "sid": 201, "start_ticks": "12"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: base if pid == 100 else worker if pid == 101 else (_ for _ in ()).throw(ProcessLookupError(pid)))
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 102])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: (base if pid == 100 else worker if pid == 101 else exited_worker)[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "Z" if pid == 102 else "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(base, known_members={100: base, 102: exited_worker})
    assert audit["member_pids"] == [100, 101]
    assert sleeps == []


def test_never_known_zombie_session_member_is_manual_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: base if pid == 100 else worker if pid == 101 else (_ for _ in ()).throw(ProcessLookupError(pid)))
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 999])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {"pid": pid, "uid": 7, "pgid": pid, "sid": 100 if pid == 100 else 201, "start_ticks": "10" if pid == 100 else "11", "state": "Z" if pid == 999 else "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT"):
        formalizer._audit_exact_process_group(base, known_members={100: base})
    assert sleeps == []


def test_known_session_member_stat_disappearance_retries_a_fresh_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    exited_worker = {**base, "pid": 102, "ppid": 100, "pgid": 102, "sid": 201, "start_ticks": "12"}
    records = {100: base, 101: worker, 102: exited_worker}
    session_passes = {100: [[100], [100]], 201: [[101, 102], [101]]}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    def stat(pid: int) -> dict[str, object]:
        if pid == 102:
            raise formalizer._SessionMemberStatUnavailable(pid)
        return {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    monkeypatch.setattr(formalizer, "_session_member_stat", stat)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(base, known_members={100: base, 102: exited_worker})
    assert audit["member_pids"] == [100, 101]
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_never_known_session_member_stat_disappearance_is_manual_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    records = {100: base, 101: worker}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 999])
    def stat(pid: int) -> dict[str, object]:
        if pid == 999:
            raise formalizer._SessionMemberStatUnavailable(pid)
        return {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    monkeypatch.setattr(formalizer, "_session_member_stat", stat)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE"):
        formalizer._audit_exact_process_group(base, known_members={100: base})
    assert sleeps == []


def test_known_session_member_stat_disappearance_twice_is_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    base = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**base, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    exited_worker = {**base, "pid": 102, "ppid": 100, "pgid": 102, "sid": 201, "start_ticks": "12"}
    records = {100: base, 101: worker}
    session_passes = {100: [[100], [100]], 201: [[101, 102], [101, 102]]}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    def stat(pid: int) -> dict[str, object]:
        if pid == 102:
            raise formalizer._SessionMemberStatUnavailable(pid)
        return {key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    monkeypatch.setattr(formalizer, "_session_member_stat", stat)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_MEMBER_STATE_UNVERIFIABLE"):
        formalizer._audit_exact_process_group(base, known_members={100: base, 102: exited_worker})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_transient_session_member_recovery_requires_fresh_token_and_ancestry_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    def member(pid: int, parent: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": pid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher, first_worker, recovered_worker = member(100, 1, 100, 10), member(101, 100, 201, 11), member(102, 100, 202, 12)
    token_passes = [[100, 101], [100, 101, 102]]
    session_passes = {100: [[100], [100]], 201: [[101, 102], [101]], 202: [[102]]}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: token_passes.pop(0))
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _: {100: launcher, 101: first_worker, 102: recovered_worker}[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: session_passes[sid].pop(0))
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: {100: launcher, 101: first_worker, 102: recovered_worker}[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(launcher, known_members={100: launcher, 102: recovered_worker})
    assert audit["member_pids"] == [100, 101, 102]
    assert audit["ancestry_depths"] == {"100": 0, "101": 1, "102": 1}
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_known_launcher_procfs_owner_exit_requires_one_exit_confirmation_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    terminal = {"pid": 100, "uid": 0, "pgid": 100, "sid": 100, "start_ticks": "10", "state": "R"}
    assert formalizer._session_stat_identity_differences(launcher, terminal) == {
        "uid": {"expected": 7, "observed": 0},
    }
    assert formalizer._is_known_procfs_owner_exit_transition(expected=launcher, earlier=launcher, member_stat=terminal)
    assert not formalizer._is_known_procfs_owner_exit_transition(
        expected={**launcher, "uid": 0}, earlier=launcher, member_stat=terminal,
    )
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: launcher)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: terminal)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer._LauncherNaturalExitCandidate, match="S18_PROCESS_LAUNCHER_PROCFS_OWNER_EXIT_CANDIDATE:pid=100:state=R:fields=uid"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_known_zombie_procfs_owner_exit_is_excluded_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    live_worker = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    exited_worker = {**launcher, "pid": 102, "ppid": 100, "pgid": 102, "sid": 201, "start_ticks": "12"}
    records = {100: launcher, 101: live_worker, 102: exited_worker}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _token: records[pid])
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101, 102])
    monkeypatch.setattr(
        formalizer,
        "_session_member_stat",
        lambda pid: {
            key: records[pid][key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")
        } | ({"uid": 0, "state": "Z"} if pid == 102 else {"state": "S"}),
    )
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(
        launcher,
        known_members={100: launcher, 102: exited_worker},
    )
    assert audit["member_pids"] == [100, 101]
    assert sleeps == []


def test_known_worker_procfs_owner_exit_reaudits_then_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    stat_passes = [
        {"pid": 101, "uid": 0, "pgid": 101, "sid": 201, "start_ticks": "11", "state": "R"},
        {key: worker[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"},
    ]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _token: launcher if pid == 100 else worker)
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101])
    def stat(pid: int) -> dict[str, object]:
        if pid == 100:
            return {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
        return stat_passes.pop(0)
    monkeypatch.setattr(formalizer, "_session_member_stat", stat)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    audit = formalizer._audit_exact_process_group(launcher, known_members={100: launcher, 101: worker})
    assert audit["member_pids"] == [100, 101]
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_known_live_worker_procfs_owner_exit_twice_is_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    terminal = {"pid": 101, "uid": 0, "pgid": 101, "sid": 201, "start_ticks": "11", "state": "R"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _token: launcher if pid == 100 else worker)
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101])
    monkeypatch.setattr(
        formalizer,
        "_session_member_stat",
        lambda pid: ({key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}) if pid == 100 else terminal,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(
        formalizer.Stage1S18ManualInterventionRequired,
        match="S18_PROCESS_SESSION_WORKER_OWNER_EXIT_UNRESOLVED:pid=101:state=R",
    ):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher, 101: worker})
    assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_procfs_owner_exit_transition_rejects_other_known_identity_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}

    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: launcher)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])

    # No X/PGID/SID shortcut is permitted: frozen observations require only
    # R/Z plus exact PGID/SID/start ticks and an owner UID transition.
    cases = [
        (
            {"pid": 100, "uid": 7, "pgid": 0, "sid": 0, "start_ticks": "10", "state": "X"},
            "S18_PROCESS_SESSION_STAT_IDENTITY_DRIFT:pid=100:state=X:fields=pgid,sid",
        ),
        (
            {"pid": 100, "uid": 8, "pgid": 100, "sid": 100, "start_ticks": "10", "state": "R"},
            "S18_PROCESS_SESSION_STAT_IDENTITY_DRIFT:pid=100:state=R:fields=uid",
        ),
        (
            {"pid": 100, "uid": 0, "pgid": 100, "sid": 100, "start_ticks": "11", "state": "R"},
            "S18_PROCESS_SESSION_STAT_IDENTITY_DRIFT:pid=100:state=R:fields=uid,start_ticks",
        ),
    ]
    for member_stat, message in cases:
        monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid, value=member_stat: value)
        with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match=message):
            formalizer._audit_exact_process_group(launcher, known_members={100: launcher})


def test_procfs_owner_exit_transition_never_promotes_unknown_member(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    newcomer = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "start_ticks": "11"}
    exact_launcher_stat = {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    uid_zero_live = {key: newcomer[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"uid": 0, "state": "R"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda pid, _token: launcher if pid == 100 else newcomer)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100, 101])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: exact_launcher_stat if pid == 100 else uid_zero_live)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(
        formalizer.Stage1S18ManualInterventionRequired,
        match="S18_PROCESS_SESSION_STAT_IDENTITY_DRIFT:pid=101:state=R:fields=uid",
    ):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == []

    unknown_zombie = {"pid": 999, "uid": 0, "pgid": 999, "sid": 100, "start_ticks": "12", "state": "Z"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: launcher)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100, 999])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: exact_launcher_stat if pid == 100 else unknown_zombie)
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert sleeps == []


def test_known_launcher_token_loss_is_natural_exit_candidate_only_after_one_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    values: list[object] = [launcher, ProcessLookupError(100), launcher, ProcessLookupError(100)]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100])
    def fingerprint(_pid: int, _token: str) -> dict[str, object]:
        value = values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value
    monkeypatch.setattr(formalizer, "_fingerprint", fingerprint)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer._LauncherNaturalExitCandidate, match="S18_PROCESS_LAUNCHER_NATURAL_EXIT_CANDIDATE"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert values == [] and sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_worker_token_loss_and_live_launcher_identity_drift_remain_manual(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    worker = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "sid": 201, "start_ticks": "11"}
    worker_calls = [worker, ProcessLookupError(101), worker, ProcessLookupError(101)]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100, 101])
    def missing_worker(pid: int, _token: str) -> dict[str, object]:
        if pid == 100:
            return launcher
        value = worker_calls.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value
    monkeypatch.setattr(formalizer, "_fingerprint", missing_worker)
    monkeypatch.setattr(formalizer, "_session_members", lambda sid: [100] if sid == 100 else [101])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda pid: {key: (launcher if pid == 100 else worker)[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher, 101: worker})
    assert worker_calls == [] and sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]

    drifted = {**launcher, "exe": "/usr/bin/python-drift"}
    launcher_calls = [launcher, drifted]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [100])
    def live_drift(_pid: int, _token: str) -> dict[str, object]:
        return launcher_calls.pop(0)
    monkeypatch.setattr(formalizer, "_fingerprint", live_drift)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_SESSION_TOKEN_MEMBERSHIP_DRIFT"):
        formalizer._audit_exact_process_group(launcher, known_members={100: launcher})
    assert launcher_calls == [] and sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]


def test_launch_audit_allows_only_confirmed_natural_exit_race(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    fingerprint = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLookupError(100)))
    immediate_waits: list[float] = []
    immediate_exit = SimpleNamespace(poll=lambda: 0, wait=lambda *, timeout: immediate_waits.append(timeout) or 0)
    assert formalizer._audit_or_confirmed_launcher_exit(immediate_exit, fingerprint, {100: fingerprint}) is None
    assert immediate_waits == [formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS]
    delayed_waits: list[float] = []
    delayed_exit = SimpleNamespace(poll=lambda: None, wait=lambda *, timeout: delayed_waits.append(timeout) or 0)
    assert formalizer._audit_or_confirmed_launcher_exit(delayed_exit, fingerprint, {100: fingerprint}) is None
    assert delayed_waits == [formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS]
    def _still_live(*, timeout: float) -> int:
        raise subprocess.TimeoutExpired(cmd="torchrun", timeout=timeout)
    with pytest.raises(ProcessLookupError):
        formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(poll=lambda: None, wait=_still_live), fingerprint, {100: fingerprint})
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(formalizer._LauncherNaturalExitCandidate("candidate")))
    candidate_waits: list[float] = []
    assert formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(wait=lambda *, timeout: candidate_waits.append(timeout) or 0), fingerprint, {100: fingerprint}) is None
    assert candidate_waits == [formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS]
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_NATURAL_EXIT_UNCONFIRMED"):
        formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(wait=_still_live), fingerprint, {100: fingerprint})


def test_launch_initial_tree_never_accepts_launcher_exit_candidate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "c" * 64
    fingerprint = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    process = SimpleNamespace(pid=100, poll=lambda: None, wait=lambda *, timeout: 0, returncode=0)
    monkeypatch.setattr(formalizer.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: fingerprint)
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(formalizer._LauncherNaturalExitCandidate("candidate")))
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [], "token_members": []})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_INITIAL_TREE_LAUNCHER_EXIT_UNCONFIRMED"):
        formalizer._launch(repository=tmp_path, work=tmp_path, label="initial", command=[sys.executable, "worker.py"], environment={}, run_token=token, timeout_seconds=1, lease=SimpleNamespace(heartbeat=lambda: None), expected_success=True)
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    marker = formalizer._mapping(load_canonical_json(tmp_path / "initial-manual-intervention.json"), field="initial.manual")
    assert marker["reason"] == "S18_PROCESS_INITIAL_TREE_LAUNCHER_EXIT_UNCONFIRMED"


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="requires Linux /proc")
def test_real_linux_procfs_owner_exit_transition_is_known_member_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reproduce the frozen host's UID-only procfs exit transition.

    ``Popen.poll`` is deliberately avoided until after sampling because it can
    reap the child and remove `/proc/<pid>` before the kernel transition is
    observed.  The bounded repetitions make the short exit window reliable on
    the frozen server while keeping all children owned and synchronously reaped.
    """

    formalizer = _formalizer()
    captured: tuple[subprocess.Popen[bytes], dict[str, object], dict[str, object]] | None = None
    for attempt in range(128):
        token = hashlib.sha256(f"s18-procfs-owner-exit-{attempt}".encode("ascii")).hexdigest()
        environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
        process = subprocess.Popen(
            [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=environment,
            start_new_session=True,
        )
        selected = False
        try:
            expected = formalizer._fingerprint(process.pid, token)
            if expected["uid"] == 0:
                pytest.skip("procfs owner transition requires a non-root launch UID")
            assert process.stdin is not None
            process.stdin.write(b"x"); process.stdin.flush(); process.stdin.close()
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                try:
                    observed = formalizer._session_member_stat(process.pid)
                except formalizer._SessionMemberStatUnavailable:
                    break
                if formalizer._is_known_procfs_owner_exit_transition(
                    expected=expected,
                    earlier=expected,
                    member_stat=observed,
                ):
                    captured = (process, expected, observed)
                    selected = True
                    break
        finally:
            if not selected:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
                process.wait(timeout=2)
        if selected:
            break
    assert captured is not None, "frozen Linux host did not expose its attested UID-only procfs exit transition"

    process, expected, observed = captured
    token = str(expected["environment_run_token"])
    pid = int(expected["pid"])
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [pid])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: expected)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [pid])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: observed)
    sleeps: list[float] = []
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: sleeps.append(seconds))
    try:
        if observed["state"] == "R":
            assert formalizer._audit_or_confirmed_launcher_exit(
                process,
                expected,
                {pid: expected},
            ) is None
            assert sleeps == [formalizer.SESSION_MEMBER_REVALIDATION_SECONDS]
        else:
            assert observed["state"] == "Z"
            audit = formalizer._audit_exact_process_group(expected, known_members={pid: expected})
            assert audit["member_pids"] == [pid]
            process.wait(timeout=2)
        with pytest.raises(formalizer.Stage1S18ManualInterventionRequired):
            formalizer._audit_exact_process_group(expected, known_members={})
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir() or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"), reason="requires Linux /proc and pidfd")
def test_real_linux_elastic_style_child_session_is_audited_and_pidfd_terminated() -> None:
    """Exercise actual /proc ownership, not just a mocked independent SID."""

    formalizer = _formalizer(); token = "d" * 64
    child_code = "import time; time.sleep(8)"
    launcher_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}], start_new_session=True); "
        "time.sleep(8)"
    )
    environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
    launcher = subprocess.Popen([sys.executable, "-c", launcher_code], env=environment, start_new_session=True)
    known: dict[int, dict[str, object]] = {}
    try:
        expected = formalizer._fingerprint(launcher.pid, token); known[launcher.pid] = expected
        deadline, audit = time.monotonic() + 4.0, None
        while time.monotonic() < deadline:
            audit = formalizer._audit_exact_process_group(expected, known_members=known)
            for member in audit["members"]:
                known.setdefault(member["pid"], member)
            if len(audit["members"]) == 2:
                break
            time.sleep(0.05)
        assert audit is not None and len(audit["members"]) == 2
        assert len({member["sid"] for member in audit["members"]}) == 2
        assert len({member["pgid"] for member in audit["members"]}) == 2
        assert sorted(audit["ancestry_depths"].values()) == [0, 1]
        assert len(formalizer._signal_exact_tree(audit, signal.SIGTERM)) == 2
        launcher.wait(timeout=5)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and formalizer._residual_launch_tree(expected, known_members=known)["session_members"]:
            time.sleep(0.05)
        assert formalizer._residual_launch_tree(expected, known_members=known) == {"session_members": [], "token_members": []}
    finally:
        if launcher.poll() is None:
            launcher.kill(); launcher.wait(timeout=5)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir() or not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"), reason="requires Linux /proc and pidfd")
def test_real_linux_same_session_token_scan_race_recovers_child_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuine inherited-token child must pass the one bounded re-scan.

    The launcher and child share one session so the first deliberately stale
    token enumeration reproduces the r11 ordering race.  The child is never
    made known or signallable until the second complete token enumeration.
    """

    formalizer = _formalizer(); token = "e" * 64; pid_path = tmp_path / "child.pid"
    child_code = "import time; time.sleep(8)"
    launcher_code = (
        "import pathlib,subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        f"pathlib.Path({str(pid_path)!r}).write_text(str(child.pid), encoding='ascii'); "
        "time.sleep(8)"
    )
    environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
    launcher = subprocess.Popen([sys.executable, "-c", launcher_code], env=environment, start_new_session=True)
    child_pid: int | None = None
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline and not pid_path.is_file():
            time.sleep(0.01)
        assert pid_path.is_file()
        child_pid = int(pid_path.read_text(encoding="ascii"))
        expected = formalizer._fingerprint(launcher.pid, token)
        original_token_process_ids = formalizer._token_process_ids
        token_scans: list[list[int]] = []

        def stale_first_scan(run_token: str) -> list[int]:
            members = original_token_process_ids(run_token)
            if not token_scans:
                members = [pid for pid in members if pid != child_pid]
            token_scans.append(members)
            return members

        monkeypatch.setattr(formalizer, "_token_process_ids", stale_first_scan)
        audit = formalizer._audit_exact_process_group(expected, known_members={launcher.pid: expected})
        assert token_scans == [[launcher.pid], sorted([launcher.pid, child_pid])]
        assert audit["member_pids"] == sorted([launcher.pid, child_pid])
        assert audit["ancestry_depths"] == {str(launcher.pid): 0, str(child_pid): 1}
        assert len({member["sid"] for member in audit["members"]}) == 1
        assert len(formalizer._signal_exact_tree(audit, signal.SIGTERM)) == 2
        launcher.wait(timeout=5)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and formalizer._residual_launch_tree(expected, known_members={launcher.pid: expected})["session_members"]:
            time.sleep(0.05)
        assert formalizer._residual_launch_tree(expected, known_members={launcher.pid: expected}) == {"session_members": [], "token_members": []}
    finally:
        if launcher.poll() is None:
            launcher.kill(); launcher.wait(timeout=5)
        if child_pid is not None:
            try:
                formalizer._signal_exact_member(formalizer._fingerprint(child_pid, token), signal.SIGKILL)
            except (OSError, ProcessLookupError, formalizer.Stage1S18ManualInterventionRequired):
                pass


def test_process_outcome_requires_exact_endpoint_handoff_and_independent_worker_tree() -> None:
    formalizer = _formalizer(); token = "e" * 64
    def member(pid: int, *, parent: int, sid: int, start: int) -> dict[str, object]:
        return {"pid": pid, "ppid": parent, "uid": 7, "pgid": sid, "sid": sid, "start_ticks": str(start), "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    launcher = member(100, parent=1, sid=100, start=10)
    workers = [member(pid, parent=100, sid=pid, start=11) for pid in range(101, 105)]
    initial = {"pgid": 100, "sid": 100, "members": [launcher], "member_pids": [100], "ancestry_depths": {"100": 0}}
    known = {"pgid": 100, "sid": 100, "members": [launcher, *workers], "member_pids": [100, 101, 102, 103, 104], "ancestry_depths": {"100": 0, "101": 1, "102": 1, "103": 1, "104": 1}}
    outcome = formalizer._with_hash({
        "schema_version": "stage1-s1-8-process-outcome-v1", "label": "negative", "command": ["python", "-m", "torch.distributed.run", "--rdzv-id", token, "--rdzv-endpoint", "127.0.0.1:43123", "--nproc_per_node", "4", "worker.py"],
        "rendezvous_id": token, "rendezvous_endpoint": "127.0.0.1:43123", "rendezvous_handoff": {"reservation_held_to_popen": True, "single_attempt": True, "silent_retry": False}, "returncode": 1,
        "fingerprint": launcher, "initial_tree": initial, "known_tree": known, "termination_audit_ref": None, "stdout_sha256": "b" * 64, "stderr_sha256": "c" * 64, "expected_success": False, "residual_launch_tree": {"session_members": [], "token_members": []},
    })
    formalizer._validate_process_outcome_contract(outcome)
    # Route A/B use one worker (launcher + 1), C uses two (launcher + 2),
    # and smoke/D use four (launcher + 4); no route shares a hard-coded five.
    for nproc in (1, 2):
        route_outcome = copy.deepcopy(outcome)
        route_outcome["command"][route_outcome["command"].index("--nproc_per_node") + 1] = str(nproc)
        route_outcome["known_tree"]["members"] = [launcher, *workers[:nproc]]
        route_outcome["known_tree"]["member_pids"] = [100, *range(101, 101 + nproc)]
        route_outcome["known_tree"]["ancestry_depths"] = {"100": 0, **{str(pid): 1 for pid in range(101, 101 + nproc)}}
        route_outcome["artifact_hash"] = formalizer._canonical({key: value for key, value in route_outcome.items() if key != "artifact_hash"})
        formalizer._validate_process_outcome_contract(route_outcome)
    drifted = copy.deepcopy(outcome); drifted["rendezvous_handoff"]["silent_retry"] = True; drifted["artifact_hash"] = formalizer._canonical({key: value for key, value in drifted.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_PROCESS_RENDEZVOUS_OUTCOME_INVALID"):
        formalizer._validate_process_outcome_contract(drifted)


def test_non_distributed_scale_outcome_requires_one_known_process() -> None:
    formalizer = _formalizer(); token = "f" * 64
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    tree = {"pgid": 100, "sid": 100, "members": [launcher], "member_pids": [100], "ancestry_depths": {"100": 0}}
    outcome = formalizer._with_hash({
        "schema_version": "stage1-s1-8-process-outcome-v1", "label": "pre-route-scale", "command": ["python", "scale.py"],
        "rendezvous_id": token, "rendezvous_endpoint": None, "rendezvous_handoff": {"reservation_held_to_popen": False, "single_attempt": True, "silent_retry": False}, "returncode": 0,
        "fingerprint": launcher, "initial_tree": tree, "known_tree": tree, "termination_audit_ref": None, "stdout_sha256": "b" * 64, "stderr_sha256": "c" * 64, "expected_success": True, "residual_launch_tree": {"session_members": [], "token_members": []},
    })
    formalizer._validate_process_outcome_contract(outcome)
    drifted = copy.deepcopy(outcome)
    extra = {**launcher, "pid": 101, "ppid": 100, "pgid": 101, "sid": 101, "start_ticks": "11"}
    drifted["known_tree"] = {"pgid": 100, "sid": 100, "members": [launcher, extra], "member_pids": [100, 101], "ancestry_depths": {"100": 0, "101": 1}}
    drifted["artifact_hash"] = formalizer._canonical({key: value for key, value in drifted.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_PROCESS_TREE_MEMBER_COUNT_INVALID"):
        formalizer._validate_process_outcome_contract(drifted)


def test_release_failure_is_manual_not_a_close_only_success(tmp_path: Path) -> None:
    formalizer = _formalizer()
    class FailingLease:
        def __init__(self) -> None:
            self.current_path = tmp_path / "current.json"; self.current_path.write_text("held", encoding="utf-8"); self.calls = 0
        def release(self, *, outcome: str) -> Path:
            self.calls += 1
            raise RuntimeError("release-failed")
    lease = FailingLease()
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_LEASE_RELEASE_UNCONFIRMED"):
        formalizer._release_lease_transaction(lease, outcome="FAILED", work=tmp_path, label="lease")
    assert lease.calls == 2 and lease.current_path.exists()
    assert (tmp_path / "lease-lease-release-failed.json").is_file()


def test_current_resource_key_marker_is_prelease_block_and_is_never_deleted(tmp_path: Path) -> None:
    formalizer = _formalizer()
    marker = tmp_path / "operations" / "gpu-leases" / "current" / ("a" * 24 + ".json")
    marker.parent.mkdir(parents=True); marker.write_text("operator-review", encoding="utf-8")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_LEASE_CURRENT_RECORD_REQUIRES_REVIEW"):
        formalizer._require_no_current_lease_record(tmp_path, SimpleNamespace(resource_key="a" * 24))
    assert marker.read_text(encoding="utf-8") == "operator-review"


def test_execute_acquire_failure_preserves_original_error_and_never_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An acquire rejection is not lease ownership and must not be masked."""

    formalizer = _formalizer(); repository, data_root = tmp_path / "repo", tmp_path / "data"
    repository.mkdir(); data_root.mkdir()
    token_file, historical_attestation, historical_replay = tmp_path / "tokens.safetensors", tmp_path / "attestation.json", tmp_path / "replay.json"
    for path in (token_file, historical_attestation, historical_replay):
        path.write_bytes(b"test")
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    preserved_marker = tmp_path / "foreign-stale-marker.json"; preserved_marker.write_text("leave-for-review", encoding="utf-8")
    handoff = {
        "token_file": token_file,
        "historical_producer_attestation_file": historical_attestation,
        "historical_producer_attestation_sha256": formalizer._sha(historical_attestation),
        "historical_g3_replay_file": historical_replay,
        "historical_g3_replay_sha256": formalizer._sha(historical_replay),
    }
    instances: list[object] = []

    class AcquireRejectedLease:
        def __init__(self, _root: Path, _identity: object) -> None:
            self.current_path = preserved_marker; self.acquire_calls = 0; self.release_calls = 0; self.close_calls = 0
            instances.append(self)
        def acquire(self) -> None:
            self.acquire_calls += 1
            raise RuntimeError("GPU_LEASE_STALE_RECORD_REQUIRES_REVIEW")
        def release(self, *, outcome: str) -> Path:
            self.release_calls += 1
            raise AssertionError("unheld lease must never release")
        def close(self) -> None:
            self.close_calls += 1

    import param_importance_nlp.runtime.operations as operations
    monkeypatch.setattr(operations, "ProjectGpuLease", AcquireRejectedLease)
    monkeypatch.setattr(formalizer, "_git", lambda _repository, *_args: "a" * 40)
    monkeypatch.setattr(formalizer, "load_s1_7_handoff", lambda **_kwargs: handoff)
    monkeypatch.setattr(formalizer, "_load_capability", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(formalizer, "_require_prelease_cuda_hidden", lambda: {"cuda_visible_devices": "", "parent_cuda_initialization": False})
    monkeypatch.setattr(formalizer, "_frozen_model_and_cache_root", lambda *_args: (str(tmp_path), str(cache_root), {"status": "PASS"}))
    monkeypatch.setattr(formalizer, "_audit_pile_download_activity", lambda _handoff: {"active_count": 0})
    monkeypatch.setattr(formalizer, "discover_approved_gpus", lambda uuids: {"requested_uuid_order": list(uuids)})
    uuids = ["GPU-00000000-1111-2222-3333-44444444444" + str(index) for index in range(4)]
    with pytest.raises(RuntimeError, match="GPU_LEASE_STALE_RECORD_REQUIRES_REVIEW"):
        formalizer.execute(repository=repository, data_root=data_root, s1_7_index_ref="ignored", gpu_capability_ref="ignored", approved_gpu_uuids=uuids, attempt_id="acquire-failure", lease_owner="test-owner")
    assert len(instances) == 1
    lease = instances[0]
    assert lease.acquire_calls == 1 and lease.release_calls == 0 and lease.close_calls == 1
    assert preserved_marker.read_text(encoding="utf-8") == "leave-for-review"
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    failed = next((data_root / "tmp" / "stage1-s1-8" / "acquire-failure").glob("failed.json"))
    assert formalizer._mapping(load_canonical_json(failed), field="failure")["error"] == "GPU_LEASE_STALE_RECORD_REQUIRES_REVIEW"


def test_held_lease_failure_runs_exact_release_transaction(tmp_path: Path) -> None:
    formalizer = _formalizer()
    class HeldLease:
        def __init__(self) -> None:
            self.current_path = tmp_path / "held.json"; self.current_path.write_text("held", encoding="utf-8"); self.calls = 0; self.closed = 0
        def release(self, *, outcome: str) -> Path:
            self.calls += 1; assert outcome == "FAILED"; self.current_path.unlink()
            history = tmp_path / "history.json"; history.write_text("released", encoding="utf-8")
            return history
        def close(self) -> None:
            self.closed += 1
    lease = HeldLease()
    assert formalizer._finalize_failed_lease(lease, held=True, release_attempted=False, error=RuntimeError("worker-failed"), work=tmp_path) is True
    assert lease.calls == 1 and lease.closed == 0 and not lease.current_path.exists()


def test_immutable_writer_and_schema_set_are_fail_closed(tmp_path: Path) -> None:
    formalizer = _formalizer()
    target = tmp_path / "role.json"; target.write_text("{}", encoding="utf-8")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_FORMAL_IMMUTABLE_COLLISION"):
        formalizer._write(target, {"status": "PASS"})
    assert formalizer._schema_prepublication_check(Path("."), {}) is True


def test_implementation_source_map_is_exact_full_production_closure() -> None:
    formalizer = _formalizer()
    sources = formalizer._implementation_source_map(Path("."))
    assert set(sources) == set(formalizer.IMPLEMENTATION_SOURCE_FILES)
    assert len(sources) == 30
    assert "src/param_importance_nlp/runtime/optimizer.py" not in sources
    assert "src/param_importance_nlp/contracts/errors.py" in sources
    assert set(name for name in sources if name.startswith("schemas/stage1/s1-8-")) == {
        "schemas/stage1/s1-8-array-bundle-v1.json", "schemas/stage1/s1-8-comparison-table-v1.json",
        "schemas/stage1/s1-8-ddp-report-v1.json", "schemas/stage1/s1-8-fixture-manifest-v1.json",
        "schemas/stage1/s1-8-formalization-index-v1.json", "schemas/stage1/s1-8-gate-record-v1.json",
        "schemas/stage1/s1-8-replay-validation-v1.json", "schemas/stage1/s1-8-safetensors-manifest-v1.json",
        "schemas/stage1/s1-8-validation-v1.json", "schemas/stage1/s1-8-worker-report-v1.json",
    }
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in sources.values())


def test_dirty_worktree_and_nonfrozen_capability_are_prelease_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    monkeypatch.setattr(formalizer, "_git", lambda _repository, *args: " M src/changed.py" if args == ("status", "--porcelain=v1") else "")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_FORMAL_WORKTREE_NOT_CLEAN"):
        formalizer._audit_consumer_diff(Path("."))
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_CAPABILITY_REF_NOT_FROZEN"):
        formalizer._load_capability(tmp_path, "evidence/not-the-frozen-capability.json", ["GPU-a", "GPU-b", "GPU-c", "GPU-d"])


def test_route_specific_candidate_contract_forbids_a_u_and_requires_bcd_accumulator() -> None:
    formalizer = _formalizer()
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    a_fields = ("mean_gradient", "raw_core", "raw_score", "raw_score_clipped", "data_update", "magnitude")
    report_a = {"route": "A", "world_size": 1, "nccl_transport_protocol": formalizer._nccl_transport_protocol(), "visible_gpu_uuids": [uuid], "rank_to_gpu_uuid": [uuid], "route_layout": {"route": "A", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": case, "ordinary_ddp_gradient_collectives": 1 if case == "equal" else 2, "array_keys": [f"a-reference/{case}/{field}/p" for field in a_fields], "accumulator": None, "rank_records": [{"rank": 0}]} for case in ("equal", "weighted")]}
    formalizer._validate_worker_candidate_contract("A", report_a)
    for counts in ([0, 1], [1, 1], [1, 3]):
        drifted_collectives = copy.deepcopy(report_a)
        for row, value in zip(drifted_collectives["cases"], counts, strict=True):
            row["ordinary_ddp_gradient_collectives"] = value
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_ORDINARY_DDP_COLLECTIVE_CONTRACT_INVALID"):
            formalizer._validate_worker_candidate_contract("A", drifted_collectives)
    bad_a = copy.deepcopy(report_a); bad_a["cases"][0]["array_keys"].append("scores/equal/u_core/p")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_A_U_OR_ACCUMULATOR_FORBIDDEN"):
        formalizer._validate_worker_candidate_contract("A", bad_a)
    report_b = {**copy.deepcopy(report_a), "route": "B", "route_layout": {"route": "B", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": "equal", "ordinary_ddp_gradient_collectives": 0, "array_keys": ["scores/equal/u_core/p", "accumulator/equal/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0}]}, {"case": "weighted", "ordinary_ddp_gradient_collectives": 0, "array_keys": ["scores/weighted/u_core/p", "accumulator/weighted/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0}]}]}
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_FORMAL_OBJECT_INVALID"):
        formalizer._validate_worker_candidate_contract("B", report_b)
    report_b["cases"] = [report_b["cases"][0]]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_CASE_CARDINALITY_INVALID"):
        formalizer._validate_worker_candidate_contract("B", report_b)


def test_bcd_shaped_candidate_positive_has_u_stats_and_two_step_accumulator() -> None:
    formalizer = _formalizer(); report = _worker_report_a(formalizer)
    report["route"] = "B"; report["route_layout"] = {"route": "B", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}
    contribution = {name: "a" * 64 for name in ("signed", "raw", "raw_clipped", "data_update", "total_update", "weight_decay_update", "actual_update_raw_importance")}
    cumulative = {name: "b" * 64 for name in ("signed", "positive", "negative_mass", "absolute", "raw", "raw_clipped", "data_movement", "net_data_movement", "total_movement", "total_endpoint_movement", "weight_decay_movement", "net_weight_decay_movement", "actual_update_raw_importance", "magnitude")}
    for ordinal, row in enumerate(report["cases"], start=1):
        case = row["case"]; stats = ("s1", "s2") if case == "equal" else ("g1", "g2")
        row["ordinary_ddp_gradient_collectives"] = 0
        row["global_statistic_checksums"] = {name: "c" * 64 for name in stats}
        row["array_keys"] = [*(f"stats/{case}/{name}/p" for name in stats), *(f"scores/{case}/{name}/p" for name in ("mean_gradient", "raw_core", "u_core", "raw_score", "u_score", "u_score_clipped")), f"accumulator/{case}/cumulative/signed/p"]
        row["accumulator"] = {"successful_steps": ordinal, "skipped_steps": 0, "signed_identity": True, "absolute_identity": True, "contribution_checksums": contribution, "cumulative_checksums": cumulative}
    report.pop("artifact_hash"); report = formalizer._with_hash(report)
    formalizer._validate_output_schemas(Path("."), {"worker_report": report, "safetensors_manifest": report["arrays"]})
    formalizer._validate_worker_candidate_contract("B", report)


def test_bcd_candidate_contract_rejects_any_ordinary_ddp_collective() -> None:
    formalizer = _formalizer()
    for route in ("B", "C", "D"):
        world_size = formalizer.ROUTE_WORLD[route]
        uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(world_size)]
        layout = ddp.build_route_layout(route).to_dict()
        report = {
            "route": route, "world_size": world_size,
            "nccl_transport_protocol": formalizer._nccl_transport_protocol(),
            "visible_gpu_uuids": uuids, "rank_to_gpu_uuid": uuids,
            "route_layout": layout,
            "cases": [
                {"case": "equal", "ordinary_ddp_gradient_collectives": 1, "array_keys": [], "accumulator": None, "rank_records": [{"rank": index} for index in range(world_size)]},
                {"case": "weighted", "ordinary_ddp_gradient_collectives": 0, "array_keys": [], "accumulator": None, "rank_records": [{"rank": index} for index in range(world_size)]},
            ],
        }
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_ORDINARY_DDP_COLLECTIVE_CONTRACT_INVALID"):
            formalizer._validate_worker_candidate_contract(route, report)


def test_gpu_uuid_runtime_contract_is_canonical_lowercase_and_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    inventory = "\n".join(f"{index}, {uuid}, NVIDIA A100-SXM4-80GB, 81920, 0, 0, 40, 8.0" for index, uuid in enumerate(uuids))
    monkeypatch.setattr(formalizer, "_run", lambda command, timeout=30: inventory if "--query-gpu=" in command[1] else "")
    assert formalizer.discover_approved_gpus(uuids)["requested_uuid_order"] == uuids
    for invalid in (uuids[:3] + ["GPU-ABCDEF12-1111-2222-3333-444444444444"], uuids[:3] + ["GPU-1"], uuids[:3] + [uuids[3] + "x"]):
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_APPROVED_GPU_UUID_SET_INVALID"):
            formalizer.discover_approved_gpus(invalid)


def test_pile_audit_binds_ready_provenance_and_never_records_command_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); handoff = {"pile_provenance": {"logical_asset_id": "pile-selected-prefix", "ready_manifest_sha256": "a" * 64}}
    assert formalizer._audit_pile_download_activity(handoff, proc_root=tmp_path) == {"status": "PASS", "pile_logical_asset_id": "pile-selected-prefix", "pile_ready_manifest_sha256": "a" * 64, "active_count": 0, "process_fingerprints": []}
    # Generic project/Pile words can occur in the formalizer's own command
    # line and must not masquerade as the known background downloader.
    benign = tmp_path / "41"; benign.mkdir(); (benign / "cmdline").write_bytes(b"parameter-importance dry-check pile provenance")
    assert formalizer._audit_pile_download_activity(handoff, proc_root=tmp_path)["active_count"] == 0
    proc = tmp_path / "42"; proc.mkdir(); sensitive = b"bash server_xet_download.sh --token=secret-do-not-persist"; (proc / "cmdline").write_bytes(sensitive); (proc / "stat").write_text(" ".join(["x"] * 22), encoding="utf-8")
    monkeypatch.setattr(formalizer.os, "getpgid", lambda pid: 77, raising=False)
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_PILE_DOWNLOAD_ACTIVITY_PRESENT") as error:
        formalizer._audit_pile_download_activity(handoff, proc_root=tmp_path)
    assert "secret-do-not-persist" not in str(error.value)


def test_pile_handoff_requires_frozen_ready_identity_and_g3_binding() -> None:
    formalizer = _formalizer(); pile = dict(formalizer.EXPECTED_PILE_PROVENANCE)
    model = {"g3_resolution_ref": pile["g3_resolution_ref"], "g3_resolution_artifact_hash": pile["g3_resolution_artifact_hash"]}
    formalizer._validate_frozen_pile_provenance(model, pile)
    for field, replacement in (
        ("storage_kind", "wrong"),
        ("asset_id", "dbbfeb12" + "0" * 52 + "5dad"),
        ("verification_sha256", "0" * 64),
    ):
        drifted = dict(pile); drifted[field] = replacement
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_S17_PILE_PROVENANCE_INVALID"):
            formalizer._validate_frozen_pile_provenance(model, drifted)
    with_unknown = dict(pile); with_unknown["unknown"] = True
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_S17_PILE_PROVENANCE_INVALID"):
        formalizer._validate_frozen_pile_provenance(model, with_unknown)


def _real_r11_historical_g3_shaped_handoff(formalizer: object) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Construct the immutable S1.7 r11 role shape without requiring DATA_ROOT."""

    tokens = {str(index): f"{index:064x}" for index in range(16)}
    model = {
        "logical_asset_id": "pythia-14m-step0",
        "ready_manifest_sha256": formalizer.EXPECTED_MODEL_READY_SHA256,
        "g3_resolution_ref": formalizer.EXPECTED_PILE_PROVENANCE["g3_resolution_ref"],
        "g3_resolution_artifact_hash": formalizer.EXPECTED_G3_RESOLUTION_PAYLOAD_HASH,
    }
    tokenizer = {"logical_asset_id": "pythia-tokenizer", "ready_manifest_sha256": formalizer.EXPECTED_TOKENIZER_IDENTITY["ready_manifest_sha256"]}
    historical_hashes = {path: "1" * 64 for path in formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    consumer_hashes = {path: "2" * 64 for path in formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    attestation = formalizer._with_hash({
        "schema_version": "stage1-s1-7-historical-producer-attestation-v1", "status": "PASS",
        "historical_producer_commit": formalizer.HISTORICAL_G3_PRODUCER,
        "consumer_commit": formalizer.EXPECTED_S1_7_PRODUCER,
        "historical_producer_is_ancestor": True,
        "critical_source_diff": list(formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS),
        "critical_patch_sha256": formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256,
        "historical_source_sha256": historical_hashes, "consumer_source_sha256": consumer_hashes,
    })
    identity = {
        "model": {**formalizer.EXPECTED_MODEL_IDENTITY, "root": "/home/sophgo13/cjl/storage/parameter-importance/models/pythia-14m-step0"},
        "tokenizer": {**formalizer.EXPECTED_TOKENIZER_IDENTITY, "root": "/home/sophgo13/cjl/storage/parameter-importance/models/pythia-tokenizer"},
        "pile": dict(formalizer.EXPECTED_PILE_IDENTITY),
    }
    replay = {
        "schema_version": "stage1-s1-7-historical-g3-replay-v1", "status": "PASS",
        "model": model, "tokenizer": tokenizer, "pile": dict(formalizer.EXPECTED_PILE_PROVENANCE),
        "asset_identity": identity,
        "resolution_commit_artifact_hash": formalizer.EXPECTED_G3_RESOLUTION_ARTIFACT_HASH,
        "resolution_artifact_hash": formalizer.EXPECTED_G3_RESOLUTION_PAYLOAD_HASH,
        "fixture_file": "fixture-inputs.safetensors", "fixture_file_sha256": "3" * 64,
        "token_sha256": tokens, "dropout_probabilities": {"attention_dropout": 0.0, "hidden_dropout": 0.0},
        "resolve_hash_seconds": 1.0, "dataset_rehash_seconds": 2.0,
        "qualified_resolution_hashed_bytes": formalizer.EXPECTED_HISTORICAL_PILE_HASHED_BYTES,
        "dataset_rehash_bytes": formalizer.EXPECTED_HISTORICAL_PILE_HASHED_BYTES,
        "pile_hash_passes": 2,
        "network_policy": {
            "hf_hub_offline": True, "transformers_offline": True, "datasets_offline": True,
            "cuda_visible_devices": True, "cuda_is_available": False,
            "operations": ["committed-resolution-parse", "qualified-local-manifest-parse", "local-pile-mmap-hash-and-fixture-extraction"],
            "external_attempts": [],
        },
    }
    replay["replay_hash"] = formalizer._canonical(replay)
    handoff = {
        "model_provenance": model, "pile_provenance": dict(formalizer.EXPECTED_PILE_PROVENANCE),
        "fixture_assets": {"model": model, "tokenizer": tokenizer, "pile": dict(formalizer.EXPECTED_PILE_PROVENANCE)},
        "token_sha256": tokens, "token_file_sha256": "3" * 64,
        "historical_producer_attestation_ref": "historical-producer-attestation.json",
        "historical_producer_attestation_sha256": formalizer.EXPECTED_S1_7_HISTORICAL_PRODUCER_ATTESTATION_SHA256,
        "historical_g3_replay_ref": "historical-g3-replay.json",
        "historical_g3_replay_sha256": formalizer.EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256,
    }
    return handoff, attestation, replay


def test_historical_s17_g3_replay_binding_is_real_r11_shaped_and_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); handoff, attestation, replay = _real_r11_historical_g3_shaped_handoff(formalizer)
    compatibility = {
        "current_consumer_commit": "b" * 40, "historical_producer_is_ancestor": True,
        "critical_source_diff": list(formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS),
        "critical_patch_sha256": formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256,
        "consumer_source_sha256": attestation["consumer_source_sha256"],
    }
    monkeypatch.setattr(formalizer, "_current_historical_g3_compatibility", lambda repository, observed: compatibility)
    binding = formalizer._validate_s1_7_historical_g3_binding(repository=Path("."), handoff=handoff, attestation=attestation, replay=replay)
    assert binding["qualification_method"] == "s1_7_published_historical_g3_replay_consumer_binding"
    assert binding["historical_g3_replay"]["sha256"] == "69e74a2adea8cbc4539e85f09cd25f453780fb9f471906b96be3805194c1278b"
    assert binding["historical_producer_attestation"]["sha256"] == "c28bcf52bd268ce34fe56e509686c6f374bd80a0f1f6d584c6387123479e230a"
    drifted = copy.deepcopy(replay); drifted["network_policy"]["cuda_is_available"] = True; drifted["replay_hash"] = formalizer._canonical({key: value for key, value in drifted.items() if key != "replay_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_S17_HISTORICAL_G3_REPLAY_INVALID"):
        formalizer._validate_s1_7_historical_g3_binding(repository=Path("."), handoff=handoff, attestation=attestation, replay=drifted)
    drifted_attestation = copy.deepcopy(attestation); drifted_attestation["consumer_commit"] = "0" * 40; drifted_attestation["artifact_hash"] = formalizer._canonical({key: value for key, value in drifted_attestation.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_S17_HISTORICAL_G3_ATTESTATION_INVALID"):
        formalizer._validate_s1_7_historical_g3_binding(repository=Path("."), handoff=handoff, attestation=drifted_attestation, replay=replay)


def test_index_handoff_flattens_historical_g3_proof_without_losing_immutable_identities() -> None:
    formalizer = _formalizer(); handoff, attestation, replay = _real_r11_historical_g3_shaped_handoff(formalizer)
    handoff["token_file"] = Path("fixture-inputs.safetensors"); handoff["historical_producer_attestation"] = attestation; handoff["historical_g3_replay"] = replay
    handoff["historical_g3_binding"] = {
        "historical_producer_attestation": {"artifact_hash": attestation["artifact_hash"], "historical_producer_commit": formalizer.HISTORICAL_G3_PRODUCER, "critical_patch_sha256": formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256, "historical_source_sha256": attestation["historical_source_sha256"]},
        "historical_g3_replay": {"replay_hash": replay["replay_hash"]},
        "current_consumer_compatibility": {"current_consumer_commit": "b" * 40, "consumer_source_sha256": attestation["consumer_source_sha256"]},
    }
    published = formalizer._index_safe_s1_7_handoff(handoff)
    assert "historical_g3_replay" not in published and "historical_g3_binding" not in published
    assert published["historical_g3_replay_sha256"] == formalizer.EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256
    assert published["historical_g3_historical_producer_commit"] == formalizer.HISTORICAL_G3_PRODUCER


def test_prelease_parent_requires_cuda_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert formalizer._require_prelease_cuda_hidden() == {"cuda_visible_devices": "", "parent_cuda_initialization": False}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-should-not-be-visible")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_PRELEASE_CUDA_VISIBLE_DEVICES_NOT_EMPTY"):
        formalizer._require_prelease_cuda_hidden()


def test_index_schema_strictly_freezes_full_s17_handoff_historical_binding() -> None:
    formalizer = _formalizer()
    model = dict(formalizer.EXPECTED_PILE_PROVENANCE)
    model.update({
        "logical_asset_id": "pythia-14m-step0", "asset_id": formalizer.EXPECTED_MODEL_IDENTITY["asset_id"],
        "manifest_ref": "manifests/model/pythia-14m.json", "asset_root_ref": "models/pythia-14m-step0",
        "ready_manifest_sha256": formalizer.EXPECTED_MODEL_READY_SHA256, "storage_kind": None,
    })
    historical_sources = {path: "1" * 64 for path in formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS}
    handoff = {
        "index_ref": "evidence/stage1/s1-7-formal/dcc/index.json", "index_sha256": formalizer.EXPECTED_S1_7_INDEX_SHA256,
        "index_artifact_hash": formalizer.EXPECTED_S1_7_ARTIFACT_HASH, "producer_commit": formalizer.EXPECTED_S1_7_PRODUCER,
        "gate_artifact_hash": formalizer.EXPECTED_G1_SINGLE_HASH, "fixture_hash": "2" * 64,
        "token_file_sha256": "3" * 64, "model_resolution_ref": model["g3_resolution_ref"],
        "model_provenance": model, "pile_provenance": dict(formalizer.EXPECTED_PILE_PROVENANCE),
        "token_sha256": {str(index): f"{index:064x}" for index in range(16)},
        "role_refs": {"fixture_manifest": "fixture-manifest.json", "single_gpu_report": "worker-report.json", "gradient_bundle": "arrays-manifest.json", "comparison_table": "comparison-table.json", "gate_record": "g1-single-record.json"},
        "role_sha256": {key: "4" * 64 for key in ("fixture_manifest", "single_gpu_report", "gradient_bundle", "comparison_table", "gate_record")},
        "historical_producer_attestation_ref": "historical-producer-attestation.json", "historical_producer_attestation_sha256": formalizer.EXPECTED_S1_7_HISTORICAL_PRODUCER_ATTESTATION_SHA256,
        "historical_g3_replay_ref": "historical-g3-replay.json", "historical_g3_replay_sha256": formalizer.EXPECTED_S1_7_HISTORICAL_G3_REPLAY_SHA256,
        "historical_g3_attestation_artifact_hash": "5" * 64, "historical_g3_historical_producer_commit": formalizer.HISTORICAL_G3_PRODUCER,
        "historical_g3_critical_patch_sha256": formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256,
        "historical_g3_historical_source_sha256": historical_sources, "historical_g3_replay_hash": "6" * 64,
        "historical_g3_current_consumer_commit": "7" * 40, "historical_g3_current_consumer_source_sha256": historical_sources,
    }
    uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    index = formalizer._with_hash({
        "schema_version": "stage1-s1-8-formalization-index-v1", "status": "PASS", "gate_id": "G1-DDP", "task_id": "stage1.08_ddp_and_gradient_accumulation",
        "generator_git_commit": "8" * 40, "consumer_git_commit": "8" * 40,
        "gpu_capability": {"commit_ref": "commit", "object_ref": "object", "task_id": "stage0.01_baseline_and_safety", "artifact_kind": "capability_cuda", "artifact_hash": formalizer.EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH, "config_hash": "9" * 64, "source_refs": ["source"], "allowed_gpu_uuids": uuids}, "nccl_transport_protocol": formalizer._nccl_transport_protocol(),
        "implementation_source_sha256": formalizer._implementation_source_map(Path(".")), "s1_7_handoff": handoff,
        "role_refs": {"fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json", "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-ddp-record.json"},
        "role_sha256": {key: "a" * 64 for key in ("fixture_manifest", "ddp_report", "array_bundle", "comparison_table", "gate_record")},
        "reproduction_role_refs": {f"role{index}": f"file{index}" for index in range(20)}, "reproduction_role_sha256": {f"role{index}": "b" * 64 for index in range(20)},
        "gate_artifact_hash": "c" * 64, "validation_ref": "validation.json", "validation_sha256": "d" * 64, "replay_ref": "replay-validation.json", "replay_sha256": "e" * 64,
        "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"],
    })
    formalizer._validate_output_schemas(Path("."), {"index": index})
    unknown = copy.deepcopy(index); unknown["s1_7_handoff"]["unbound"] = True; unknown["artifact_hash"] = formalizer._canonical({key: value for key, value in unknown.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": unknown})
    missing = copy.deepcopy(index); del missing["s1_7_handoff"]["historical_g3_current_consumer_source_sha256"]; missing["artifact_hash"] = formalizer._canonical({key: value for key, value in missing.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": missing})
    transport = copy.deepcopy(index); transport["nccl_transport_protocol"]["process_group_timeout_seconds"] = 120; transport["artifact_hash"] = formalizer._canonical({key: value for key, value in transport.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": transport})


def test_current_critical_sources_are_exactly_zero_drift_from_dcc925_attestation() -> None:
    formalizer = _formalizer(); repository = Path(".").resolve()
    attestation = {"consumer_source_sha256": {path: formalizer._sha(repository / path) for path in formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS}}
    audit = formalizer._current_historical_g3_compatibility(repository, attestation)
    assert audit["s1_7_producer_to_current_critical_source_diff"] == []
    assert audit["critical_source_diff"] == list(formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS)
    assert audit["critical_patch_sha256"] == formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256
