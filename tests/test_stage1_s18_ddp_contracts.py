"""CPU regressions for the S1.8 route and FP64-replay contracts."""

from __future__ import annotations

import math
import importlib.util
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
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
from param_importance_nlp import stage1_ddp_scale_oracle as scale_oracle


def _formalizer() -> object:
    formalizer_path = Path("ops/stage1/formalize_s1_8.py")
    spec = importlib.util.spec_from_file_location("s18_formalizer_contract_test", formalizer_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _worker_report_a(formalizer: object) -> dict[str, object]:
    manifest = formalizer._with_hash({"schema_version": "stage1-s1-8-safetensors-manifest-v1", "file": "route-A.safetensors", "file_sha256": "a" * 64, "file_size_bytes": 4, "tensors": {"a-reference/equal/raw_core/p": {"sha256": "b" * 64, "dtype": "torch.float32", "shape": [1]}}})
    def case(name: str) -> dict[str, object]:
        return {"case": name, "global_loss_numerator": 1.0, "global_loss_valid_token_count": 8, "global_mean_loss": 0.125, "global_microbatch_count": 8, "global_n1": 8, "global_n2": 8, "global_gradient_norm": 1.0, "clip_factor": 1.0, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64], "global_statistic_checksums": {}, "local_loss_numerator": 1.0, "local_effective_tokens": 8}], "global_statistic_checksums": {}, "ordinary_ddp_gradient_collectives": 1 if name == "equal" else 2, "manual_statistic_collectives": {"backend": "nccl", "operation": "SUM", "tensor_statistics": [], "tensor_all_reduce_count": 0, "scalar_statistics": [], "scalar_all_reduce_count": 0, "total_all_reduce_count": 0}, "post_parameter_checksum": "d" * 64, "pre_parameter_checksum": "e" * 64, "accumulator": None, "array_keys": [f"a-reference/{name}/raw_core/p"]}
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


def _rehash_fixture(fixture: dict[str, object]) -> dict[str, object]:
    fixture["fixture_hash"] = ddp.canonical_json_hash(
        {key: value for key, value in fixture.items() if key != "fixture_hash"}
    )
    return fixture


def _worker_plan_with_fixture(tmp_path: Path, fixture: dict[str, object]) -> dict[str, object]:
    data_root = tmp_path / "data"
    model_root = data_root / "models" / "model"
    cache_root = data_root / "cache"
    model_root.mkdir(parents=True)
    cache_root.mkdir()
    tokens = tmp_path / "fixture-inputs.safetensors"
    tokens.write_bytes(b"fixture")
    return ddp.with_artifact_hash({
        "schema_version": ddp.WORKER_PLAN_SCHEMA,
        "task_id": ddp.TASK_ID,
        "execution_commit": "a" * 40,
        "route": "A",
        "fixture": fixture,
        "fixture_tokens_ref": tokens.name,
        "fixture_tokens_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "data_root": str(data_root),
        "model_root": str(model_root),
        "cache_root": str(cache_root),
        "run_token": "b" * 64,
        "visible_gpu_uuids": ["GPU-00000000-1111-2222-3333-444444444444"],
        "nccl_transport_protocol": dict(ddp.NCCL_TRANSPORT_PROTOCOL),
        "output_dir": "route-output",
        "permutation": "identity",
        "execution_mode": "formal",
    })


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


def test_conditioned_v3_fixture_rejects_jointly_rehashed_semantic_drift_at_runtime_and_schema(tmp_path: Path) -> None:
    """A new fixture hash cannot authorize a new numerical experiment."""

    formalizer = _formalizer()
    mutations: dict[str, tuple[object, str]] = {
        "schema_version": ("stage1-s1-8-fixture-manifest-v2", "S18_FIXTURE_SCHEMA_INVALID"),
        "fixture_id": ("different", "S18_FIXTURE_SCHEMA_INVALID"),
        "precision.atol": (2.0e-7, "S18_FIXTURE_PRECISION_CONTRACT_INVALID"),
        "optimizer.learning_rate": (4.0e-4, "S18_FIXTURE_OPTIMIZER_CONDITIONING_BINDING_INVALID"),
        "optimizer.betas": ([0.8, 0.999], "S18_FIXTURE_OPTIMIZER_CONDITIONING_BINDING_INVALID"),
        "optimizer.weight_decay": (0.02, "S18_FIXTURE_OPTIMIZER_CONDITIONING_BINDING_INVALID"),
        "optimizer.foreach": (0, "S18_FIXTURE_OPTIMIZER_CONDITIONING_BINDING_INVALID"),
        "optimizer.extra": (True, "S18_FIXTURE_OPTIMIZER_CONDITIONING_BINDING_INVALID"),
        "optimizer_conditioning.learning_rate": (4.0e-4, "S18_FIXTURE_OPTIMIZER_CONDITIONING_INVALID"),
        "gradient_clip_max_norm": (0.5, "S18_FIXTURE_CLIP_CONFIGURATION_INVALID"),
        "randomness.model_seed": (1708, "S18_FIXTURE_RANDOMNESS_CONTRACT_INVALID"),
        "randomness.dropout": ("enabled", "S18_FIXTURE_RANDOMNESS_CONTRACT_INVALID"),
        "cases.weighted": ({"label_ignore_suffixes": [0, 15, 32, 48, 64, 80, 96, 112], "effective_target_tokens": [2048, 2033, 2016, 2000, 1984, 1968, 1952, 1936], "statistics": "weighted_g1_g2_n1_n2"}, "S18_FIXTURE_CASE_STATISTICS_CONTRACT_INVALID"),
        "routes.A.route": ("B", "S18_FIXTURE_ROUTE_LAYOUT_INVALID"),
        "ddp.backend": ("gloo", "S18_FIXTURE_DDP_CONTRACT_INVALID"),
        "upstream_s1_7_fixture_hash": ("not-a-sha256", "S18_UPSTREAM_S1_7_FIXTURE_HASH_INVALID"),
    }
    for name, (value, marker) in mutations.items():
        drifted = copy.deepcopy(_fixture())
        if name == "fixture_id":
            drifted["fixture_id"] = value
        elif name == "schema_version":
            drifted["schema_version"] = value
        elif name == "gradient_clip_max_norm":
            drifted["gradient_clip_max_norm"] = value
        elif name == "upstream_s1_7_fixture_hash":
            drifted["upstream_s1_7_fixture_hash"] = value
        elif name == "cases.weighted":
            drifted["cases"]["weighted"] = value  # type: ignore[index]
        else:
            location = name.split(".")
            target: object = drifted
            for key in location[:-1]:
                target = target[key]  # type: ignore[index]
            target[location[-1]] = value  # type: ignore[index]
        _rehash_fixture(drifted)
        with pytest.raises(Stage1S18Error, match=marker):
            validate_fixture(drifted)
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:fixture_manifest"):
            formalizer._validate_output_schemas(Path("."), {"fixture_manifest": drifted})

    # These use the actual worker-plan validation entrance, rather than only
    # unit-testing the fixture helper.  Their self hashes are recomputed too.
    for name in ("precision.atol", "optimizer.learning_rate"):
        drifted = copy.deepcopy(_fixture())
        parent, child = name.split(".")
        drifted[parent][child] = 2.0e-7 if parent == "precision" else 4.0e-4  # type: ignore[index]
        _rehash_fixture(drifted)
        plan = _worker_plan_with_fixture(tmp_path / name.replace(".", "-"), drifted)
        with pytest.raises(Stage1S18Error):
            validate_worker_plan(plan, path=tmp_path / f"{name}.json")


def test_v3_fixture_exactly_binds_lr_conditioning_and_rejects_rehashed_v2_fixture() -> None:
    """The new fixture is a distinct conditioning experiment, never a v2 rewrite."""

    formalizer = _formalizer()
    fixture = _fixture()
    assert fixture["schema_version"] == "stage1-s1-8-fixture-manifest-v3"
    assert fixture["fixture_id"] == "stage1-s1-8-pythia14m-ddp-conditioned-v3"
    assert fixture["optimizer"]["learning_rate"] == 3.0e-3
    assert fixture["optimizer_conditioning"]["learning_rate"] == 3.0e-3
    assert ddp.OPTIMIZER_CONDITIONING["learning_rate"] == 3.0e-3
    assert scale_oracle.OPTIMIZER_CONDITIONING["learning_rate"] == 3.0e-3
    assert ddp.PRE_ROUTE_SCALE_ORACLE_SCHEMA == scale_oracle.REPORT_SCHEMA == "stage1-s1-8-pre-route-gradient-scale-oracle-v3"
    assert scale_oracle.PLAN_SCHEMA == "stage1-s1-8-pre-route-scale-plan-v3"

    old = copy.deepcopy(fixture)
    old["schema_version"] = "stage1-s1-8-fixture-manifest-v2"
    old["fixture_id"] = "stage1-s1-8-pythia14m-ddp-conditioned-v2"
    old["optimizer"]["learning_rate"] = 3.0e-4
    old["optimizer_conditioning"]["learning_rate"] = 3.0e-4
    old["optimizer_conditioning"]["schema_version"] = "stage1-s1-8-optimizer-conditioning-v2"
    old["pre_route_gradient_scale"]["schema_version"] = "stage1-s1-8-pre-route-gradient-scale-oracle-v2"
    _rehash_fixture(old)
    with pytest.raises(Stage1S18Error, match="S18_FIXTURE_SCHEMA_INVALID"):
        validate_fixture(old)
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:fixture_manifest"):
        formalizer._validate_output_schemas(Path("."), {"fixture_manifest": old})


def test_conditioned_fixture_and_scale_plan_reject_r16_epsilon(tmp_path: Path) -> None:
    fixture = _fixture()
    assert fixture["schema_version"] == "stage1-s1-8-fixture-manifest-v3"
    assert fixture["optimizer"]["eps"] == 1.0e-4
    assert fixture["optimizer_conditioning"] == oracle.OPTIMIZER_CONDITIONING == scale_oracle.OPTIMIZER_CONDITIONING
    drifted_fixture = copy.deepcopy(fixture)
    drifted_fixture["optimizer"]["eps"] = 1.0e-8
    drifted_fixture["optimizer_conditioning"]["eps"] = 1.0e-8
    drifted_fixture["fixture_hash"] = ddp.canonical_json_hash({key: value for key, value in drifted_fixture.items() if key != "fixture_hash"})
    with pytest.raises(Stage1S18Error, match="S18_FIXTURE_OPTIMIZER_CONDITIONING"):
        validate_fixture(drifted_fixture)

    token_path = tmp_path / "fixture-inputs.safetensors"; token_path.write_bytes(b"fixture")
    selected = {str(index): f"{index:064x}" for index in range(8)}
    upstream = {str(index): f"{index:064x}" for index in range(16)}
    plan_body = {
        "schema_version": scale_oracle.PLAN_SCHEMA, "task_id": scale_oracle.TASK_ID,
        "execution_commit": "a" * 40, "fixture_tokens_ref": token_path.name,
        "fixture_tokens_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "selected_token_sha256": selected, "upstream_token_sha256": upstream,
        "cases": {"equal": {"label_ignore_suffixes": [0] * 8}, "weighted": {"label_ignore_suffixes": [0, 16, 32, 48, 64, 80, 96, 112]}},
        "optimizer_conditioning": copy.deepcopy(scale_oracle.OPTIMIZER_CONDITIONING),
        "model_root": "model", "output_file": "pre-route-scale.json", "run_token": "b" * 64,
        "visible_gpu_uuids": ["GPU-00000000-1111-2222-3333-444444444444"],
    }
    plan = {**plan_body, "artifact_hash": scale_oracle.canonical_json_hash(plan_body)}
    assert scale_oracle.validate_plan(plan, path=tmp_path / "scale-plan.json")["optimizer_conditioning"] == scale_oracle.OPTIMIZER_CONDITIONING
    for field, value in (("eps", 1.0e-8), ("learning_rate", 4.0e-4), ("weight_decay", 0.02), ("betas", [0.8, 0.999])):
        old_plan = copy.deepcopy(plan)
        old_plan["optimizer_conditioning"][field] = value
        old_plan["artifact_hash"] = scale_oracle.canonical_json_hash({key: value for key, value in old_plan.items() if key != "artifact_hash"})
        with pytest.raises(scale_oracle.Stage1S18ScaleOracleError, match="S18_SCALE_PLAN_OPTIMIZER_CONDITIONING_INVALID"):
            scale_oracle.validate_plan(old_plan, path=tmp_path / "scale-plan.json")

    formalizer = _formalizer()
    for field, value in (("learning_rate", 4.0e-4), ("weight_decay", 0.02), ("betas", [0.8, 0.999])):
        report = {"optimizer_conditioning": copy.deepcopy(scale_oracle.OPTIMIZER_CONDITIONING)}
        report["optimizer_conditioning"][field] = value
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_PRE_ROUTE_SCALE_OPTIMIZER_CONDITIONING_INVALID"):
            formalizer._require_pre_route_scale_conditioning(report, ddp.OPTIMIZER_CONDITIONING)

    drifted_binding = copy.deepcopy(fixture)
    drifted_binding["optimizer"]["learning_rate"] = 4.0e-4
    drifted_binding["optimizer_conditioning"]["learning_rate"] = 4.0e-4
    _rehash_fixture(drifted_binding)
    with pytest.raises(Stage1S18Error, match="S18_FIXTURE_OPTIMIZER_CONDITIONING"):
        validate_fixture(drifted_binding)


def test_adamw_step_contract_requires_a_cpu_fp32_zero_dim_strided_non_grad_scalar() -> None:
    valid = {"p": torch.tensor(1.0, dtype=torch.float32)}
    assert oracle._require_exact_step(valid, expected_step=1, field="test") == 1
    assert oracle._exact_step_check(valid, expected_step=1)["within_t32_distributed"] is True
    invalid = {
        "rank_one": torch.tensor([1.0], dtype=torch.float32),
        "sparse": torch.sparse_coo_tensor(indices=[[0]], values=[1.0], size=[1]),
        "requires_grad": torch.tensor(1.0, dtype=torch.float32, requires_grad=True),
    }
    for name, value in invalid.items():
        candidate = {name: value}
        check = oracle._exact_step_check(candidate, expected_step=1)
        assert check["within_t32_distributed"] is False
        with pytest.raises(oracle.Stage1S18OracleError, match=f"S18_ORACLE_ADAMW_PRE_STEP_INVALID:test:{name}"):
            oracle._require_exact_step(candidate, expected_step=1, field="test")


def test_fp32_execution_order_emulator_matches_adamw_one_and_two_steps() -> None:
    optimizer_config = {"learning_rate": 3.0e-3, "weight_decay": 0.01, "betas": [0.9, 0.999], "eps": 1.0e-4, "foreach": False, "fused": False}
    pre = {"p": torch.tensor([0.25, -0.5, 1.0e-3], dtype=torch.float32)}
    mean_steps = (
        {"p": torch.tensor([1.0e-8, -2.0e-6, 0.125], dtype=torch.float32)},
        {"p": torch.tensor([-2.0e-8, 1.0e-6, -0.25], dtype=torch.float32)},
    )
    expected_avg = {"p": torch.zeros_like(pre["p"])}; expected_sq = {"p": torch.zeros_like(pre["p"])}
    actual = torch.nn.Parameter(pre["p"].clone())
    actual_optimizer = torch.optim.AdamW([actual], lr=optimizer_config["learning_rate"], betas=tuple(optimizer_config["betas"]), eps=optimizer_config["eps"], weight_decay=optimizer_config["weight_decay"], foreach=False, fused=False)
    for previous_step, mean in enumerate(mean_steps):
        expected_post, expected_avg, expected_sq, expected_step = oracle._adamw_step_fp32_execution(
            pre={"p": actual.detach().clone()}, mean=mean, clip_factor=0.75,
            previous_exp_avg=expected_avg, previous_exp_avg_sq=expected_sq,
            previous_step=previous_step, optimizer=optimizer_config,
        )
        if previous_step:
            state = actual_optimizer.state[actual]
            state["step"] = torch.tensor(float(previous_step), dtype=torch.float32)
            state["exp_avg"] = expected_avg_before.clone()
            state["exp_avg_sq"] = expected_sq_before.clone()
        actual.grad = (mean["p"] * 0.75).clone()
        actual_optimizer.step()
        state = actual_optimizer.state[actual]
        assert torch.equal(actual.detach(), expected_post["p"])
        assert torch.equal(state["exp_avg"], expected_avg["p"])
        assert torch.equal(state["exp_avg_sq"], expected_sq["p"])
        assert float(state["step"].item()) == float(expected_step)
        expected_avg_before = expected_avg["p"].clone()
        expected_sq_before = expected_sq["p"].clone()


def test_route_specific_optimizer_mean_cannot_fall_back_to_b_micro_or_a_reference() -> None:
    a_arrays = {"a-reference/equal/mean_gradient/p": torch.tensor([1.0], dtype=torch.float32)}
    b_arrays = {"scores/equal/mean_gradient/p": torch.tensor([2.0], dtype=torch.float32)}
    a_mean, a_kind = oracle._route_optimizer_mean(route="A", case="equal", arrays=a_arrays)
    b_mean, b_kind = oracle._route_optimizer_mean(route="B", case="equal", arrays=b_arrays)
    assert float(a_mean["p"].item()) == 1.0 and a_kind.startswith("A_full_batch")
    assert float(b_mean["p"].item()) == 2.0 and b_kind.startswith("route_fp32")
    with pytest.raises(oracle.Stage1S18OracleError):
        oracle._route_optimizer_mean(route="B", case="equal", arrays=a_arrays)


def test_fixed_near_zero_sensitivity_fixture_requires_conditioned_epsilon_for_t32_peer() -> None:
    optimizer = {"learning_rate": 3.0e-4, "weight_decay": 0.01, "betas": [0.9, 0.999], "foreach": False, "fused": False}
    pre = {"p": torch.tensor([0.25, -0.5], dtype=torch.float32)}
    zero = {"p": torch.zeros_like(pre["p"])}
    positive = {"p": torch.tensor([1.0e-8, -1.0e-8], dtype=torch.float32)}
    negative = {"p": -positive["p"]}
    def emulate(eps: float, gradient: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        post, _, _, _ = oracle._adamw_step_fp32_execution(
            pre=pre, mean=gradient, clip_factor=1.0, previous_exp_avg=zero,
            previous_exp_avg_sq=zero, previous_step=0, optimizer={**optimizer, "eps": eps},
        )
        return post
    old = oracle.compare_peer_maps(emulate(1.0e-8, positive), emulate(1.0e-8, negative), natural_scale=1.0, atol=ddp.T32_DISTRIBUTED_ATOL, rtol=ddp.T32_DISTRIBUTED_RTOL, normalized_l2_limit=ddp.T32_DISTRIBUTED_L2_LIMIT)
    conditioned = oracle.compare_peer_maps(emulate(1.0e-4, positive), emulate(1.0e-4, negative), natural_scale=1.0, atol=ddp.T32_DISTRIBUTED_ATOL, rtol=ddp.T32_DISTRIBUTED_RTOL, normalized_l2_limit=ddp.T32_DISTRIBUTED_L2_LIMIT)
    assert old["within_t32_distributed"] is False
    assert conditioned["within_t32_distributed"] is True


def test_peer_drift_remains_a_t32_gate_failure() -> None:
    check = oracle.compare_peer_maps(
        {"p": torch.tensor([1.0], dtype=torch.float32)}, {"p": torch.tensor([1.001], dtype=torch.float32)},
        natural_scale=1.0, atol=ddp.T32_DISTRIBUTED_ATOL, rtol=ddp.T32_DISTRIBUTED_RTOL,
        normalized_l2_limit=ddp.T32_DISTRIBUTED_L2_LIMIT,
    )
    assert check["reference_kind"] == "production_peer_fp32_promoted_to_fp64_for_metric_only"
    assert check["within_t32_distributed"] is False


def test_observed_peer_and_permutation_movements_remain_direct_t32_gates() -> None:
    """No execution reference may replace the required observed-movement checks."""

    import inspect
    source = inspect.getsource(oracle.replay)
    assert "_add_observed_peer_update_checks(" in source
    reference = {
        "data_update": {"p": torch.tensor([3.0e-3], dtype=torch.float32)},
        "data_movement": {"p": torch.tensor([3.0e-3], dtype=torch.float32)},
        "total_update": {"p": torch.tensor([2.97e-3], dtype=torch.float32)},
        "total_movement": {"p": torch.tensor([2.97e-3], dtype=torch.float32)},
        "weight_decay_update": {"p": torch.tensor([-3.0e-5], dtype=torch.float32)},
        "weight_decay_movement": {"p": torch.tensor([3.0e-5], dtype=torch.float32)},
        "actual_update_raw_importance": {"p": torch.tensor([-3.0e-4], dtype=torch.float32)},
        "magnitude": {"p": torch.tensor([0.997], dtype=torch.float32)},
    }
    # This is the same slash-keyed persisted candidate object supplied to
    # replay after D identity or a D rank_swap/local_reverse run is loaded.
    rank_swap_candidate = {
        "accumulator/equal/contribution/data_update/p": reference["data_update"]["p"].clone(),
        "accumulator/equal/contribution/total_update/p": reference["total_update"]["p"].clone(),
        "accumulator/equal/contribution/weight_decay_update/p": reference["weight_decay_update"]["p"].clone(),
        "accumulator/equal/contribution/actual_update_raw_importance/p": reference["actual_update_raw_importance"]["p"].clone(),
        "accumulator/equal/cumulative/magnitude/p": reference["magnitude"]["p"].clone(),
    }
    scales = {"optimizer_delta": 3.0e-3, "score": 3.0e-4, "parameter": 1.0}
    precision = {"atol": ddp.T32_DISTRIBUTED_ATOL, "rtol": ddp.T32_DISTRIBUTED_RTOL, "normalized_l2_limit": ddp.T32_DISTRIBUTED_L2_LIMIT}
    checks: dict[str, object] = {}; rows: list[dict[str, object]] = []
    oracle._add_observed_peer_update_checks(checks, rows, case="equal", route="D", candidate_arrays=rank_swap_candidate, reference_updates=reference, scales=scales, precision=precision)
    assert all(value["within_t32_distributed"] for value in checks.values())
    # Mutate only observed D rank-swap post-pre data movement/update.  The
    # direct replay Gate—not a comparator unit test or execution oracle—fails.
    rank_swap_candidate["accumulator/equal/contribution/data_update/p"] = torch.tensor([1.0e-1], dtype=torch.float32)
    checks = {}; rows = []
    oracle._add_observed_peer_update_checks(checks, rows, case="equal", route="D", candidate_arrays=rank_swap_candidate, reference_updates=reference, scales=scales, precision=precision)
    assert checks["equal:A:D:peer_data_update"]["within_t32_distributed"] is False
    assert checks["equal:A:D:peer_data_movement"]["within_t32_distributed"] is False


def test_t32_is_unchanged_and_r17_tenfold_margin_is_analytic_not_formal_equivalence() -> None:
    """Recorded r17 scale analysis justifies v3 conditioning; it is not a formal run."""

    assert (ddp.T32_DISTRIBUTED_ATOL, ddp.T32_DISTRIBUTED_RTOL, ddp.T32_DISTRIBUTED_L2_LIMIT) == (1.0e-7, 1.0e-4, 1.0e-4)
    r17_worst_scaled_error = 4.3059988261529014e-4
    fixed_t32_relative_limit = ddp.T32_DISTRIBUTED_ATOL + ddp.T32_DISTRIBUTED_RTOL
    projected_v3_scaled_error = r17_worst_scaled_error / 10.0
    assert r17_worst_scaled_error > fixed_t32_relative_limit
    assert projected_v3_scaled_error < fixed_t32_relative_limit
    # CPU-only scalar simulation of the same scale law: fixed FP32 ULP error
    # divided by a tenfold optimizer delta produces the same tenfold margin.
    ulp_error = torch.tensor(1.1920928955078125e-7, dtype=torch.float64)
    old_delta = torch.tensor(2.7680e-4, dtype=torch.float64)
    assert torch.equal(ulp_error / (old_delta * 10.0), (ulp_error / old_delta) / 10.0)


def test_baseline_replay_precedes_permutation_and_negative_launches_with_dedicated_code() -> None:
    import inspect
    formalizer = _formalizer()
    source = inspect.getsource(formalizer.execute)
    baseline = source.index('phase = "baseline_offline_replay"')
    rank_permutation = source.index("for permutation in PERMUTATIONS")
    negative = source.index('for mode, marker in (("ordinary_sync_negative"')
    assert baseline < rank_permutation < negative
    assert "_require_baseline_replay(" in source
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_BASELINE_REPLAY_FAILED"):
        formalizer._require_baseline_replay(replay_fn=lambda **_: {"status": "FAIL"}, route_arrays={}, fixture={}, route_reports={})


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
    for count in (3, 5):
        bad_gradient_count = copy.deepcopy(row)
        bad_gradient_count["rank_records"][0]["local_gradient_checksums"] = ["c" * 64] * count
        with pytest.raises(oracle.Stage1S18OracleError, match="S18_ORACLE_RANK_GRADIENT_CHECKSUM_INVALID:equal:C:0"):
            oracle._validate_rank_contract(route="C", report=report, row=bad_gradient_count, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})
    row["ordinary_ddp_gradient_collectives"] = 1
    with pytest.raises(oracle.Stage1S18OracleError, match="S18_ORACLE_NO_SYNC_COLLECTIVE_DRIFT"):
        oracle._validate_rank_contract(route="C", report=report, row=row, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})


def test_rank_contract_route_a_requires_one_full_batch_gradient_checksum() -> None:
    ids = list(range(8))
    row = {
        "rank_records": [{"rank": 0, "local_microbatch_ids": ids, "local_gradient_checksums": ["c" * 64], "global_statistic_checksums": {}, "local_loss_numerator": 7.0, "local_effective_tokens": 16384}],
        "global_statistic_checksums": {},
        "ordinary_ddp_gradient_collectives": 1,
        "global_loss_valid_token_count": 16384,
        "global_loss_numerator": 7.0,
    }
    report = {"route_layout": {"rank_microbatch_ids": [ids]}}
    oracle._validate_rank_contract(route="A", report=report, row=row, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})
    for gradients in ([], ["c" * 64] * 2, ["G" * 64]):
        drifted = copy.deepcopy(row); drifted["rank_records"][0]["local_gradient_checksums"] = gradients
        with pytest.raises(oracle.Stage1S18OracleError, match="S18_ORACLE_RANK_GRADIENT_CHECKSUM_INVALID:equal:A:0"):
            oracle._validate_rank_contract(route="A", report=report, row=drifted, case="equal", precision={"atol": 1e-7, "rtol": 1e-4})


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
        row["array_keys"] = [f"a-reference/{row['case']}/{field}/p" for field in ("mean_gradient", "raw_core", "raw_score", "raw_score_clipped", "data_update", "data_movement", "total_update", "total_movement", "weight_decay_update", "weight_decay_movement", "actual_update_raw_importance", "magnitude")]
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


def test_initial_launcher_attestation_retries_empty_argv_but_rejects_timeout_and_identity_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "b" * 64
    stable = {"pid": 12, "ppid": 1, "uid": 1000, "pgid": 12, "sid": 12, "start_ticks": "99", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": token}
    empty = {**stable, "cmdline_sha256": formalizer._EMPTY_CMDLINE_SHA256}
    now = [0.0]
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    observations = iter([empty, stable, stable])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: next(observations))
    assert formalizer._attest_initial_launcher(12, token) == stable

    # Non-empty observations separated by the fork/exec empty window are not
    # consecutive; a fourth stable read is required before attestation.
    now[0] = 0.0
    calls = [0]
    observations = iter([stable, empty, stable, stable])
    def separated(_pid: int, _token: str) -> dict[str, object]:
        calls[0] += 1
        return next(observations)
    monkeypatch.setattr(formalizer, "_fingerprint", separated)
    assert formalizer._attest_initial_launcher(12, token) == stable
    assert calls[0] == 4

    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: empty)
    now[0] = 0.0
    with pytest.raises(formalizer._InitialLauncherAttestationFailure, match="S18_PROCESS_INITIAL_LAUNCHER_ATTESTATION_EMPTY_CMDLINE_TIMEOUT") as timed_out:
        formalizer._attest_initial_launcher(12, token)
    assert timed_out.value.expected["cmdline_sha256"] == formalizer._EMPTY_CMDLINE_SHA256

    for changed in ({**empty, "uid": 1001}, {**empty, "environment_run_token": "c" * 64}, {**empty, "pid": 13}):
        now[0] = 0.0
        observations = iter([empty, changed])
        monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token, observations=observations: next(observations))
        with pytest.raises(formalizer._InitialLauncherAttestationFailure, match="S18_PROCESS_INITIAL_LAUNCHER_(IDENTITY|POPEN_PID)_DRIFT"):
            formalizer._attest_initial_launcher(12, token)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="requires Linux /proc")
def test_real_posix_popen_initial_attestation_recovers_empty_to_stable_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real Popen child deterministically models its one empty-argv read."""

    formalizer = _formalizer(); token = hashlib.sha256(b"s18-initial-exec-start-race").hexdigest()
    environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=environment, start_new_session=True,
    )
    try:
        real_fingerprint, calls = formalizer._fingerprint, [0]
        def empty_then_stable(pid: int, run_token: str) -> dict[str, object]:
            calls[0] += 1
            observed = real_fingerprint(pid, run_token)
            return {**observed, "cmdline_sha256": formalizer._EMPTY_CMDLINE_SHA256} if calls[0] == 1 else observed
        monkeypatch.setattr(formalizer, "_fingerprint", empty_then_stable)
        attested = formalizer._attest_initial_launcher(process.pid, token)
        assert attested["pid"] == process.pid and attested["cmdline_sha256"] != formalizer._EMPTY_CMDLINE_SHA256
        assert calls[0] == 3
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=2)


def test_launch_persists_manual_marker_when_initial_launcher_argv_never_stabilizes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); token = "c" * 64
    empty = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": formalizer._EMPTY_CMDLINE_SHA256, "environment_run_token": token}
    process = SimpleNamespace(pid=100, poll=lambda: None, wait=lambda *, timeout: 0, returncode=0)
    now = [0.0]
    monkeypatch.setattr(formalizer.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: empty)
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds))
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: pytest.fail("initial tree must not run"))
    with pytest.raises(formalizer._InitialLauncherAttestationFailure, match="S18_PROCESS_INITIAL_LAUNCHER_ATTESTATION_EMPTY_CMDLINE_TIMEOUT"):
        formalizer._launch(repository=tmp_path, work=tmp_path, label="empty-launcher", command=[sys.executable, "worker.py"], environment={}, run_token=token, timeout_seconds=1, lease=SimpleNamespace(heartbeat=lambda: None), expected_success=True)
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    marker = formalizer._mapping(load_canonical_json(tmp_path / "empty-launcher-manual-intervention.json"), field="empty-launcher.manual")
    assert marker["reason"] == "S18_PROCESS_INITIAL_LAUNCHER_ATTESTATION_EMPTY_CMDLINE_TIMEOUT"
    assert marker["action"] == "NO_SIGNAL_NO_LEASE_RELEASE"


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
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [], "token_members": []})
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
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_NATURAL_EXIT_UNCONFIRMED"):
        formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(poll=lambda: None, wait=_still_live), fingerprint, {100: fingerprint})
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(formalizer._LauncherNaturalExitCandidate("candidate")))
    candidate_waits: list[float] = []
    assert formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(wait=lambda *, timeout: candidate_waits.append(timeout) or 0), fingerprint, {100: fingerprint}) is None
    assert candidate_waits == [formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS]
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_NATURAL_EXIT_UNCONFIRMED"):
        formalizer._audit_or_confirmed_launcher_exit(SimpleNamespace(wait=_still_live), fingerprint, {100: fingerprint})


def test_launch_audit_natural_exit_requires_attestation_expected_pid_and_no_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    fingerprint = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    exited = SimpleNamespace(wait=lambda *, timeout: 0)

    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLookupError(101)))
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [], "token_members": []})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_AUDIT_TARGET_INVALID"):
        formalizer._audit_or_confirmed_launcher_exit(exited, fingerprint, {100: fingerprint})

    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda *_args, **_kwargs: (_ for _ in ()).throw(ProcessLookupError(100)))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_UNATTESTED"):
        formalizer._audit_or_confirmed_launcher_exit(exited, fingerprint, {})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_UNATTESTED"):
        formalizer._audit_or_confirmed_launcher_exit(
            exited,
            fingerprint,
            {100: {**fingerprint, "environment_run_token": "c" * 64}},
        )

    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [101], "token_members": []})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_RESIDUAL"):
        formalizer._audit_or_confirmed_launcher_exit(exited, fingerprint, {100: fingerprint})


def test_token_missing_owner_exit_uses_terminal_join_only_for_exact_attested_state(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    owner_exit = {key: launcher[key] for key in ("pid", "pgid", "sid", "start_ticks")} | {"uid": 0, "state": "R"}
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [])
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: owner_exit)
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [], "token_members": []})
    waits: list[float] = []
    def delayed_owner_exit(*, timeout: float) -> int:
        waits.append(timeout)
        if len(waits) < 3:
            raise subprocess.TimeoutExpired(cmd="torchrun", timeout=timeout)
        return 0
    assert formalizer._audit_or_confirmed_launcher_exit(
        SimpleNamespace(wait=delayed_owner_exit), launcher, {100: launcher},
    ) is None
    assert formalizer.TERMINAL_PROCESS_JOIN_TIMEOUT_SECONDS == 30.0
    assert waits == [formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS] * 3

    never_wait = SimpleNamespace(wait=lambda *, timeout: pytest.fail(f"unexpected wait {timeout}"))
    live_stat = {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: live_stat)
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_TOKEN_MISSING_LIVE_OR_IDENTITY_DRIFT"):
        formalizer._audit_or_confirmed_launcher_exit(never_wait, launcher, {100: launcher})

    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: owner_exit)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100, 101])
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_OWNER_EXIT_SESSION_MEMBERS"):
        formalizer._audit_or_confirmed_launcher_exit(never_wait, launcher, {100: launcher})
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_UNATTESTED"):
        formalizer._audit_or_confirmed_launcher_exit(never_wait, launcher, {})

    drifted_owner_exit = {**owner_exit, "start_ticks": "11"}
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: drifted_owner_exit)
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_TOKEN_MISSING_LIVE_OR_IDENTITY_DRIFT"):
        formalizer._audit_or_confirmed_launcher_exit(never_wait, launcher, {100: launcher})

    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: owner_exit)
    monkeypatch.setattr(formalizer, "_residual_launch_tree", lambda *_args, **_kwargs: {"session_members": [100], "token_members": []})
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_EXIT_RESIDUAL"):
        formalizer._audit_or_confirmed_launcher_exit(
            SimpleNamespace(wait=lambda *, timeout: 0), launcher, {100: launcher},
        )


def test_token_missing_owner_exit_revalidates_each_incomplete_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    launcher = {"pid": 100, "ppid": 1, "uid": 7, "pgid": 100, "sid": 100, "start_ticks": "10", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    owner_exit = {key: launcher[key] for key in ("pid", "pgid", "sid", "start_ticks")} | {"uid": 0, "state": "R"}
    recovered_live = {key: launcher[key] for key in ("pid", "uid", "pgid", "sid", "start_ticks")} | {"state": "S"}
    states = [owner_exit, owner_exit, recovered_live]
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [])
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [100])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: states.pop(0))
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired, match="S18_PROCESS_LAUNCHER_TOKEN_MISSING_LIVE_OR_IDENTITY_DRIFT"):
        formalizer._audit_or_confirmed_launcher_exit(
            SimpleNamespace(wait=lambda *, timeout: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd="torchrun", timeout=timeout))),
            launcher,
            {100: launcher},
        )
    assert states == []


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="requires Linux /proc")
def test_real_linux_short_launcher_exit_is_confirmed_only_after_attestation() -> None:
    """A real, reaped launcher exercises the raw ProcessLookup race closure."""

    formalizer = _formalizer()
    token = hashlib.sha256(b"s18-confirmed-short-launcher-exit").hexdigest()
    environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )
    try:
        fingerprint = formalizer._fingerprint(process.pid, token)
        assert process.stdin is not None
        process.stdin.write(b"x"); process.stdin.flush(); process.stdin.close()
        assert process.wait(timeout=2) == 0
        # The process has been reaped, so `_audit_exact_process_group` reaches
        # its expected-PID ProcessLookupError branch without a mock.
        assert formalizer._audit_or_confirmed_launcher_exit(
            process,
            fingerprint,
            {int(fingerprint["pid"]): fingerprint},
        ) is None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


@pytest.mark.skipif(os.name != "posix" or not Path("/proc").is_dir(), reason="requires Linux /proc")
def test_real_linux_owner_exit_candidate_joins_teardown_longer_than_one_second(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use a real pipe-blocked launcher to prove the owner path exceeds 1s.

    The frozen UID-only owner transition is supplied as the observed procfs
    state, while ``Popen.wait`` and the >1s teardown are real: the child cannot
    exit until the timer releases its stdin pipe.  No mocked sleep can make the
    terminal-join path pass.
    """

    formalizer = _formalizer()
    token = hashlib.sha256(b"s18-owner-exit-terminal-join").hexdigest()
    environment = dict(os.environ); environment["PARAM_IMPORTANCE_S18_RUN_TOKEN"] = token
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.buffer.read(1)"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=environment,
        start_new_session=True,
    )
    timer: threading.Timer | None = None
    try:
        fingerprint = formalizer._fingerprint(process.pid, token)
        pid = int(fingerprint["pid"])
        owner_exit = {key: fingerprint[key] for key in ("pid", "pgid", "sid", "start_ticks")} | {"uid": 0, "state": "R"}
        real_token_process_ids = formalizer._token_process_ids
        real_session_members = formalizer._session_members
        monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [])
        monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [pid])
        monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: owner_exit)
        real_wait = process.wait
        def wait_then_restore_real_procfs(*, timeout: float | None = None) -> int:
            returncode = real_wait(timeout=timeout)
            monkeypatch.setattr(formalizer, "_token_process_ids", real_token_process_ids)
            monkeypatch.setattr(formalizer, "_session_members", real_session_members)
            return returncode
        monkeypatch.setattr(process, "wait", wait_then_restore_real_procfs)
        assert process.stdin is not None
        def release_pipe() -> None:
            try:
                process.stdin.write(b"x"); process.stdin.flush(); process.stdin.close()
            except (BrokenPipeError, ValueError):
                pass
        timer = threading.Timer(formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS + 0.25, release_pipe)
        started = time.monotonic(); timer.start()
        assert formalizer._audit_or_confirmed_launcher_exit(
            process,
            fingerprint,
            {pid: fingerprint},
        ) is None
        assert time.monotonic() - started > formalizer.LAUNCHER_EXIT_CONFIRMATION_TIMEOUT_SECONDS
    finally:
        if timer is not None:
            timer.cancel()
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)


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
    pid = int(expected["pid"])
    real_token_process_ids = formalizer._token_process_ids
    real_session_members = formalizer._session_members
    monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [pid])
    monkeypatch.setattr(formalizer, "_fingerprint", lambda _pid, _token: expected)
    monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [pid])
    monkeypatch.setattr(formalizer, "_session_member_stat", lambda _pid: observed)
    try:
        if observed["state"] == "R":
            # The synthetic audit snapshot above preserves the narrow procfs
            # owner-transition window until `_audit_exact_process_group`
            # recognizes it.  After the helper's bounded Popen.wait reaps the
            # real child, restore the real procfs scanners so its residual
            # gate observes production's empty post-exit state rather than
            # this deliberately stale test snapshot.
            real_wait = process.wait
            def wait_then_restore_real_procfs(*, timeout: float | None = None) -> int:
                returncode = real_wait(timeout=timeout)
                monkeypatch.setattr(formalizer, "_token_process_ids", real_token_process_ids)
                monkeypatch.setattr(formalizer, "_session_members", real_session_members)
                return returncode
            monkeypatch.setattr(process, "wait", wait_then_restore_real_procfs)
            assert formalizer._audit_or_confirmed_launcher_exit(
                process,
                expected,
                {pid: expected},
            ) is None
        else:
            assert observed["state"] == "Z"
            audit = formalizer._audit_exact_process_group(expected, known_members={pid: expected})
            assert audit["member_pids"] == [pid]
            process.wait(timeout=2)
        # The positive R path deliberately restored real procfs after reaping
        # the child.  The following independent negative instead needs the
        # original still-observable owner-transition snapshot: it proves that
        # the same transition is rejected when no prior attestation exists.
        monkeypatch.setattr(formalizer, "_token_process_ids", lambda _: [pid])
        monkeypatch.setattr(formalizer, "_session_members", lambda _sid: [pid])
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
    monkeypatch.setattr(formalizer, "require_gpu_quiescence", lambda uuids, *, work, phase: {"final_gpu": {"requested_uuid_order": list(uuids), "selected": [], "compute_apps": []}})
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


def test_recovery_action_preflight_fails_before_lease_acquire(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Reset recommendation is hardware state, not a launch-time soft failure."""

    formalizer = _formalizer(); repository, data_root = tmp_path / "repo", tmp_path / "data"
    repository.mkdir(); data_root.mkdir()
    token_file, attestation, replay = (tmp_path / "tokens.safetensors", tmp_path / "attestation.json", tmp_path / "replay.json")
    for path in (token_file, attestation, replay):
        path.write_bytes(b"test")
    cache_root = tmp_path / "cache"; cache_root.mkdir()
    handoff = {
        "token_file": token_file, "historical_producer_attestation_file": attestation,
        "historical_producer_attestation_sha256": formalizer._sha(attestation),
        "historical_g3_replay_file": replay, "historical_g3_replay_sha256": formalizer._sha(replay),
    }
    constructed: list[object] = []
    class LeaseMustNotConstruct:
        def __init__(self, *_args: object) -> None:
            constructed.append(self)
            raise AssertionError("recovery preflight must precede lease construction")
    import param_importance_nlp.runtime.operations as operations
    monkeypatch.setattr(operations, "ProjectGpuLease", LeaseMustNotConstruct)
    monkeypatch.setattr(formalizer, "_git", lambda _repository, *_args: "a" * 40)
    monkeypatch.setattr(formalizer, "load_s1_7_handoff", lambda **_kwargs: handoff)
    monkeypatch.setattr(formalizer, "_load_capability", lambda *_args, **_kwargs: {"status": "PASS"})
    monkeypatch.setattr(formalizer, "_require_prelease_cuda_hidden", lambda: {"cuda_visible_devices": "", "parent_cuda_initialization": False})
    monkeypatch.setattr(formalizer, "_frozen_model_and_cache_root", lambda *_args: (str(tmp_path), str(cache_root), {"status": "PASS"}))
    monkeypatch.setattr(formalizer, "_audit_pile_download_activity", lambda _handoff: {"active_count": 0})
    bad_uuid = "GPU-00000000-1111-2222-3333-444444444441"
    monkeypatch.setattr(formalizer, "require_gpu_quiescence", lambda _uuids, *, work, phase: (_ for _ in ()).throw(formalizer.Stage1S18FormalError("S18_GPU_RECOVERY_ACTION_NOT_NONE:" + bad_uuid + ":Reset")))
    uuids = ["GPU-00000000-1111-2222-3333-44444444444" + str(index) for index in range(4)]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_RECOVERY_ACTION_NOT_NONE"):
        formalizer.execute(repository=repository, data_root=data_root, s1_7_index_ref="ignored", gpu_capability_ref="ignored", approved_gpu_uuids=uuids, attempt_id="recovery-action-preflight", lease_owner="test-owner")
    assert constructed == []
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    failed = next((data_root / "tmp" / "stage1-s1-8" / "recovery-action-preflight").glob("failed.json"))
    assert formalizer._mapping(load_canonical_json(failed), field="failure") == {
        "schema_version": "stage1-s1-8-failure-v1", "status": "FAILED", "phase": "preflight",
        "error_type": "Stage1S18FormalError", "error": "S18_GPU_RECOVERY_ACTION_NOT_NONE:" + bad_uuid + ":Reset",
        "artifact_hash": formalizer._mapping(load_canonical_json(failed), field="failure")["artifact_hash"],
    }


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
    assert len(sources) == 48
    assert "src/param_importance_nlp/runtime/optimizer.py" not in sources
    assert "src/param_importance_nlp/contracts/errors.py" in sources
    assert set(name for name in sources if name.startswith("schemas/stage1/s1-8-")) == {
        "schemas/stage1/s1-8-array-bundle-v1.json", "schemas/stage1/s1-8-comparison-table-v1.json",
        "schemas/stage1/s1-8-ddp-report-v1.json", "schemas/stage1/s1-8-ddp-report-v2.json", "schemas/stage1/s1-8-ddp-report-v3.json", "schemas/stage1/s1-8-ddp-report-v4.json", "schemas/stage1/s1-8-ddp-report-v5.json", "schemas/stage1/s1-8-fixture-manifest-v1.json", "schemas/stage1/s1-8-fixture-manifest-v2.json", "schemas/stage1/s1-8-fixture-manifest-v3.json",
        "schemas/stage1/s1-8-formalization-index-v1.json", "schemas/stage1/s1-8-formalization-index-v2.json", "schemas/stage1/s1-8-formalization-index-v3.json", "schemas/stage1/s1-8-formalization-index-v4.json", "schemas/stage1/s1-8-formalization-index-v5.json", "schemas/stage1/s1-8-gate-record-v1.json", "schemas/stage1/s1-8-gpu-quiescence-v1.json", "schemas/stage1/s1-8-gpu-quiescence-v2.json", "schemas/stage1/s1-8-gpu-quiescence-v3.json",
        "schemas/stage1/s1-8-replay-validation-v1.json", "schemas/stage1/s1-8-replay-validation-v2.json", "schemas/stage1/s1-8-safetensors-manifest-v1.json",
        "schemas/stage1/s1-8-validation-v1.json", "schemas/stage1/s1-8-validation-v2.json", "schemas/stage1/s1-8-validation-v3.json", "schemas/stage1/s1-8-validation-v4.json", "schemas/stage1/s1-8-validation-v5.json", "schemas/stage1/s1-8-worker-report-v1.json",
    }
    assert all(len(digest) == 64 and set(digest) <= set("0123456789abcdef") for digest in sources.values())


def test_implementation_source_map_rejects_schema_byte_drift() -> None:
    formalizer = _formalizer()
    repository = Path(".").resolve()
    source_map = formalizer._implementation_source_map(repository)
    formalizer._validate_implementation_source_map(repository, source_map)
    drifted = dict(source_map)
    source = "schemas/stage1/s1-8-formalization-index-v5.json"
    drifted[source] = "0" * 64 if source_map[source] != "0" * 64 else "1" * 64
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_SOURCE_MAP_BYTE_DRIFT"):
        formalizer._validate_implementation_source_map(repository, drifted)


def test_dirty_worktree_and_nonfrozen_capability_are_prelease_rejections(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    monkeypatch.setattr(formalizer, "_git", lambda _repository, *args: " M src/changed.py" if args == ("status", "--porcelain=v1") else "")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_FORMAL_WORKTREE_NOT_CLEAN"):
        formalizer._audit_consumer_diff(Path("."))
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_CAPABILITY_REF_NOT_FROZEN"):
        formalizer._load_capability(tmp_path, "evidence/not-the-frozen-capability.json", ["GPU-a", "GPU-b", "GPU-c", "GPU-d"])


def test_consumer_diff_allows_only_the_frozen_temp_ignore_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()

    def accepted(_repository: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1"):
            return ""
        if args[:2] == ("diff", "--name-only"):
            return ".gitignore\nops/stage1/formalize_s1_8.py"
        raise AssertionError(args)

    monkeypatch.setattr(formalizer, "_git", accepted)
    assert formalizer._audit_consumer_diff(Path(".")) == (".gitignore", "ops/stage1/formalize_s1_8.py")

    def rejected(_repository: Path, *args: str) -> str:
        if args == ("status", "--porcelain=v1"):
            return ""
        if args[:2] == ("diff", "--name-only"):
            return ".gitignore\nREADME.md"
        raise AssertionError(args)

    monkeypatch.setattr(formalizer, "_git", rejected)
    with pytest.raises(formalizer.Stage1S18FormalError, match=r"S18_S17_CONSUMER_DIFF_UNAUTHORIZED:README\.md"):
        formalizer._audit_consumer_diff(Path("."))


def test_route_specific_candidate_contract_forbids_a_u_and_requires_bcd_accumulator() -> None:
    formalizer = _formalizer()
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    a_fields = ("mean_gradient", "raw_core", "raw_score", "raw_score_clipped", "data_update", "data_movement", "total_update", "total_movement", "weight_decay_update", "weight_decay_movement", "actual_update_raw_importance", "magnitude")
    report_a = {"route": "A", "world_size": 1, "nccl_transport_protocol": formalizer._nccl_transport_protocol(), "visible_gpu_uuids": [uuid], "rank_to_gpu_uuid": [uuid], "route_layout": {"route": "A", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": case, "ordinary_ddp_gradient_collectives": 1 if case == "equal" else 2, "array_keys": [f"a-reference/{case}/{field}/p" for field in a_fields], "accumulator": None, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64]}]} for case in ("equal", "weighted")]}
    formalizer._validate_worker_candidate_contract("A", report_a)
    for gradients in ([], ["c" * 64] * 2, ["G" * 64]):
        bad_checksums = copy.deepcopy(report_a)
        bad_checksums["cases"][0]["rank_records"][0]["local_gradient_checksums"] = gradients
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_LOCAL_GRADIENT_CHECKSUM_INVALID:A"):
            formalizer._validate_worker_candidate_contract("A", bad_checksums)
    for counts in ([0, 1], [1, 1], [1, 3]):
        drifted_collectives = copy.deepcopy(report_a)
        for row, value in zip(drifted_collectives["cases"], counts, strict=True):
            row["ordinary_ddp_gradient_collectives"] = value
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_ORDINARY_DDP_COLLECTIVE_CONTRACT_INVALID"):
            formalizer._validate_worker_candidate_contract("A", drifted_collectives)
    bad_a = copy.deepcopy(report_a); bad_a["cases"][0]["array_keys"].append("scores/equal/u_core/p")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_A_U_OR_ACCUMULATOR_FORBIDDEN"):
        formalizer._validate_worker_candidate_contract("A", bad_a)
    report_b = {**copy.deepcopy(report_a), "route": "B", "route_layout": {"route": "B", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": "equal", "ordinary_ddp_gradient_collectives": 0, "array_keys": ["scores/equal/u_core/p", "accumulator/equal/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64] * 8}]}, {"case": "weighted", "ordinary_ddp_gradient_collectives": 0, "array_keys": ["scores/weighted/u_core/p", "accumulator/weighted/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64] * 8}]}]}
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
        row["rank_records"][0]["local_gradient_checksums"] = ["c" * 64] * 8
        row["global_statistic_checksums"] = {name: "c" * 64 for name in stats}
        row["array_keys"] = [*(f"stats/{case}/{name}/p" for name in stats), *(f"scores/{case}/{name}/p" for name in ("mean_gradient", "raw_core", "u_core", "raw_score", "u_score", "u_score_clipped")), f"accumulator/{case}/cumulative/signed/p"]
        row["accumulator"] = {"successful_steps": ordinal, "skipped_steps": 0, "signed_identity": True, "absolute_identity": True, "contribution_checksums": contribution, "cumulative_checksums": cumulative}
    report.pop("artifact_hash"); report = formalizer._with_hash(report)
    formalizer._validate_output_schemas(Path("."), {"worker_report": report, "safetensors_manifest": report["arrays"]})
    formalizer._validate_worker_candidate_contract("B", report)
    for count in (7, 9):
        bad_gradient_count = copy.deepcopy(report)
        bad_gradient_count["cases"][0]["rank_records"][0]["local_gradient_checksums"] = ["c" * 64] * count
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_WORKER_LOCAL_GRADIENT_CHECKSUM_INVALID:B"):
            formalizer._validate_worker_candidate_contract("B", bad_gradient_count)


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
    inventory = "\n".join(f"{index}, {uuid}, NVIDIA A100-SXM4-80GB, 81920, 0, 0, 40, 8.0, None" for index, uuid in enumerate(uuids))
    monkeypatch.setattr(formalizer, "_run", lambda command, timeout=30: inventory if "--query-gpu=" in command[1] else "")
    discovered = formalizer.discover_approved_gpus(uuids)
    assert discovered["requested_uuid_order"] == uuids
    assert [row["recovery_action"] for row in discovered["selected"]] == ["None"] * 4
    for invalid in (uuids[:3] + ["GPU-ABCDEF12-1111-2222-3333-444444444444"], uuids[:3] + ["GPU-1"], uuids[:3] + [uuids[3] + "x"]):
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_APPROVED_GPU_UUID_SET_INVALID"):
            formalizer.discover_approved_gpus(invalid)


def test_combined_inventory_recovery_parser_and_command_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    combined = "\n".join(f"{index}, {uuid}, NVIDIA A100-SXM4-80GB, 81920, 0, 0, 40, 8.0, None" for index, uuid in enumerate(uuids))
    commands: list[list[str]] = []
    def run(command: list[str], timeout: int = 30) -> str:
        commands.append(command)
        return combined if command[1].startswith("--query-gpu=") else ""
    monkeypatch.setattr(formalizer, "_run", run)
    observed = formalizer._probe_approved_gpus(uuids)
    assert observed["compute_apps"] == []
    assert [row["recovery_action"] for row in observed["selected"]] == ["None"] * 4
    assert commands == [
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap,gpu_recovery_action", "--format=csv,noheader,nounits"],
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader,nounits"],
    ]
    pid_apps = f"{uuids[3]}, 991, torchrun\n"
    monkeypatch.setattr(formalizer, "_run", lambda command, timeout=30: combined if command[1].startswith("--query-gpu=") else pid_apps)
    with pytest.raises(formalizer.Stage1S18FormalError, match=f"S18_GPU_COMPUTE_PROCESS_PRESENT:{uuids[3]}:991"):
        formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase="prelease")
    for malformed in (combined.replace(", None", ", ", 1), combined.replace("8.0, None", "8.0", 1)):
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_DISCOVERY_PARSE_INVALID"):
            formalizer._parse_approved_gpu_inventory(malformed, expected_uuids=uuids)


def test_gpu_recovery_action_query_rejects_reset_empty_unknown_and_uuid_mismatch() -> None:
    formalizer = _formalizer()
    uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    good = "\n".join(f"{uuid}, None" for uuid in uuids)
    assert formalizer._parse_gpu_recovery_actions(good, expected_uuids=uuids) == {uuid: "None" for uuid in uuids}
    reset = good.replace(f"{uuids[1]}, None", f"{uuids[1]}, Reset")
    assert formalizer._parse_gpu_recovery_actions(reset, expected_uuids=uuids)[uuids[1]] == "Reset"
    for output, marker in (
        (good.replace(f"{uuids[1]}, None", f"{uuids[1]}, "), "S18_GPU_RECOVERY_ACTION_PARSE_INVALID"),
        (good.replace(uuids[1], "GPU-ffffffff-1111-2222-3333-444444444444"), "S18_GPU_RECOVERY_ACTION_UUID_MISMATCH"),
    ):
        with pytest.raises(formalizer.Stage1S18FormalError, match=marker):
            formalizer._parse_gpu_recovery_actions(output, expected_uuids=uuids)


def test_discovery_rejects_any_non_none_recovery_action(monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    for action in ("Reset", "Unknown"):
        inventory = "\n".join(f"{index}, {uuid}, NVIDIA A100-SXM4-80GB, 81920, 0, 0, 40, 8.0, {action if uuid == uuids[1] else 'None'}" for index, uuid in enumerate(uuids))
        monkeypatch.setattr(formalizer, "_run", lambda command, timeout=30, inventory=inventory: inventory if "--query-gpu=" in command[1] else "")
        with pytest.raises(formalizer.Stage1S18FormalError, match=f"S18_GPU_RECOVERY_ACTION_NOT_NONE:{uuids[1]}:{action}"):
            formalizer.discover_approved_gpus(uuids)


def _quiescence_probe(formalizer: object, uuids: list[str], *, memory_used_mib: int = 0, utilization_percent: int = 0, recovery_action: str = "None", apps: list[dict[str, object]] | None = None) -> dict[str, object]:
    return {
        "requested_uuid_order": list(uuids),
        "selected": [
            {"physical_index": index, "uuid": uuid, "name": "NVIDIA A100-SXM4-80GB", "memory_total_mib": 81920, "memory_used_mib": memory_used_mib, "utilization_percent": utilization_percent, "temperature_c": 40, "compute_capability": "8.0", "recovery_action": recovery_action}
            for index, uuid in enumerate(uuids)
        ],
        "compute_apps": [] if apps is None else apps,
    }


def test_gpu_compute_app_parser_and_quiescence_pid_recovery_are_immediate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    assert formalizer._parse_gpu_compute_apps(f"{uuids[0]}, 123, python\n", expected_uuids=uuids) == [{"gpu_uuid": uuids[0], "pid": 123, "process_name": "python"}]
    for malformed in (f"{uuids[0]}, 0, python", f"{uuids[0]}, 12", "not-a-uuid, 12, python"):
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_COMPUTE_APPS_PARSE_INVALID"):
            formalizer._parse_gpu_compute_apps(malformed, expected_uuids=uuids)

    app = {"gpu_uuid": uuids[2], "pid": 991, "process_name": "torchrun"}
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: _quiescence_probe(formalizer, uuids, apps=[app]))
    with pytest.raises(formalizer.Stage1S18FormalError, match=f"S18_GPU_COMPUTE_PROCESS_PRESENT:{uuids[2]}:991"):
        formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase="post_worker")
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    pid_record = load_canonical_json(tmp_path / "post-worker-gpu-quiescence.json")
    assert pid_record["status"] == "FAILED" and pid_record["samples"][0]["compute_apps"] == [app]
    assert formalizer._self_hash(pid_record)

    recovery_dir = tmp_path / "recovery"; recovery_dir.mkdir()
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: _quiescence_probe(formalizer, uuids, recovery_action="Reset"))
    with pytest.raises(formalizer.Stage1S18FormalError, match=f"S18_GPU_RECOVERY_ACTION_NOT_NONE:{uuids[0]}:Reset"):
        formalizer.require_gpu_quiescence(uuids, work=recovery_dir, phase="post_worker")
    recovery_record = load_canonical_json(recovery_dir / "post-worker-gpu-quiescence.json")
    assert recovery_record["status"] == "FAILED" and recovery_record["samples"][0]["selected"][0]["recovery_action"] == "Reset"
    assert formalizer._self_hash(recovery_record)


def test_gpu_quiescence_resets_then_requires_three_exact_idle_and_deadline_retains_samples(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    now = [0.0]
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    observations = iter([
        _quiescence_probe(formalizer, uuids, memory_used_mib=1),
        _quiescence_probe(formalizer, uuids), _quiescence_probe(formalizer, uuids), _quiescence_probe(formalizer, uuids),
    ])
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: next(observations))
    record = formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase="post_release")
    assert record["status"] == "PASS"
    assert record["timeout_seconds"] == 180.0
    assert record["operational_timeout_basis"] == formalizer.GPU_QUIESCENCE_OPERATIONAL_TIMEOUT_BASIS
    assert [sample["consecutive_exact_idle_samples"] for sample in record["samples"]] == [0, 1, 2, 3]
    assert [sample["transient_observation_count"] for sample in record["samples"]] == [1, 1, 1, 1]
    assert record["max_transient_samples"] == 2 and record["transient_observation_count"] == 1
    assert record["samples"][0]["violations"] == ["S18_GPU_NOT_IDLE_A100:" + uuid for uuid in uuids]
    formalizer._validate_output_schemas(Path("."), {"gpu_quiescence": record})

    timeout_dir = tmp_path / "timeout"; timeout_dir.mkdir(); now[0] = 0.0; calls = [0]
    def timeout_after_reset(_uuids: list[str]) -> dict[str, object]:
        calls[0] += 1
        if calls[0] == 4:
            now[0] = 180.01
        return _quiescence_probe(formalizer, uuids, utilization_percent=1 if calls[0] == 3 else 0)
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", timeout_after_reset)
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_QUIESCENCE_TIMEOUT"):
        formalizer.require_gpu_quiescence(uuids, work=timeout_dir, phase="reacquire_preflight")
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    timed_out = load_canonical_json(timeout_dir / "reacquire-preflight-gpu-quiescence.json")
    assert timed_out["status"] == "FAILED" and len(timed_out["samples"]) >= 2 and timed_out["failure_reason"] == "S18_GPU_QUIESCENCE_TIMEOUT"
    assert formalizer._self_hash(timed_out)

    # Completion time, not loop entry time, defines the frozen 180.0-second
    # bound.  A third sample that crosses it cannot convert to a PASS.
    crossing_dir = tmp_path / "crossing"; crossing_dir.mkdir(); now[0] = 0.0; calls = [0]
    def cross_deadline(_uuids: list[str]) -> dict[str, object]:
        calls[0] += 1
        if calls[0] == 3:
            now[0] = 180.01
        return _quiescence_probe(formalizer, uuids)
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", cross_deadline)
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_QUIESCENCE_TIMEOUT"):
        formalizer.require_gpu_quiescence(uuids, work=crossing_dir, phase="post_worker")
    crossed = load_canonical_json(crossing_dir / "post-worker-gpu-quiescence.json")
    assert crossed["status"] == "FAILED" and crossed["samples"][-1]["consecutive_exact_idle_samples"] == 3
    assert crossed["samples"][-1]["monotonic_elapsed_seconds"] == 180.01

    # Equality is explicitly admissible: completion exactly at the deadline is
    # still within the 180.0-second bound if it provides the third sample.
    equality_dir = tmp_path / "equality"; equality_dir.mkdir(); now[0] = 0.0; calls[0] = 0
    def at_deadline(_uuids: list[str]) -> dict[str, object]:
        calls[0] += 1
        if calls[0] == 3:
            now[0] = 180.0
        return _quiescence_probe(formalizer, uuids)
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", at_deadline)
    equality = formalizer.require_gpu_quiescence(uuids, work=equality_dir, phase="post_worker")
    assert equality["status"] == "PASS" and equality["samples"][-1]["monotonic_elapsed_seconds"] == 180.0


def test_gpu_quiescence_v3_bounds_two_transients_and_ninth_sample_deadline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    now = [0.0]
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))

    # With at most two non-PID transients, E,E,T,E,E,T,E,E,E is the longest
    # admissible sequence before the ninth observation establishes E,E,E.
    worst_case = [False, False, True, False, False, True, False, False, False]
    observations = iter([_quiescence_probe(formalizer, uuids, memory_used_mib=1 if transient else 0) for transient in worst_case])
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: next(observations))
    record = formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase="post_release")
    assert record["status"] == "PASS" and len(record["samples"]) == 9
    assert record["transient_observation_count"] == 2
    assert [sample["consecutive_exact_idle_samples"] for sample in record["samples"]] == [1, 2, 0, 1, 2, 0, 1, 2, 3]
    assert [sample["transient_observation_count"] for sample in record["samples"]] == [0, 0, 1, 1, 1, 2, 2, 2, 2]

    third_transient_dir = tmp_path / "third-transient"; third_transient_dir.mkdir(); now[0] = 0.0
    observations = iter([_quiescence_probe(formalizer, uuids, utilization_percent=1) for _ in range(3)])
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: next(observations))
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_QUIESCENCE_TRANSIENT_LIMIT_EXCEEDED"):
        formalizer.require_gpu_quiescence(uuids, work=third_transient_dir, phase="post_release")
    from param_importance_nlp.contracts.jsonio import load_canonical_json
    third = load_canonical_json(third_transient_dir / "post-release-gpu-quiescence.json")
    assert third["status"] == "FAILED" and third["failure_reason"] == "S18_GPU_QUIESCENCE_TRANSIENT_LIMIT_EXCEEDED"
    assert third["transient_observation_count"] == 3 and third["samples"][-1]["transient_observation_count"] == 3

    def ninth_probe(crosses: float) -> object:
        calls = [0]
        def probe(_uuids: list[str]) -> dict[str, object]:
            calls[0] += 1
            if calls[0] == 9:
                now[0] = crosses
            return _quiescence_probe(formalizer, uuids, memory_used_mib=1 if calls[0] in {3, 6} else 0)
        return probe

    crossing_dir = tmp_path / "ninth-crossing"; crossing_dir.mkdir(); now[0] = 0.0
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", ninth_probe(180.01))
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_GPU_QUIESCENCE_TIMEOUT"):
        formalizer.require_gpu_quiescence(uuids, work=crossing_dir, phase="post_release")
    crossed = load_canonical_json(crossing_dir / "post-release-gpu-quiescence.json")
    assert crossed["samples"][-1]["sample_index"] == 8 and crossed["samples"][-1]["monotonic_elapsed_seconds"] == 180.01

    equality_dir = tmp_path / "ninth-equality"; equality_dir.mkdir(); now[0] = 0.0
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", ninth_probe(180.0))
    equality = formalizer.require_gpu_quiescence(uuids, work=equality_dir, phase="post_release")
    assert equality["status"] == "PASS" and equality["samples"][-1]["sample_index"] == 8
    assert equality["samples"][-1]["monotonic_elapsed_seconds"] == 180.0


def test_gpu_quiescence_v3_accepts_r11_transient_sequence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    now = [0.0]
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    # r11 was E,T,E,E,E: the one no-PID utilization transient resets only the
    # direct-idle counter; the later three exact observations are still Gates.
    observations = iter([
        _quiescence_probe(formalizer, uuids),
        _quiescence_probe(formalizer, uuids, utilization_percent=2),
        _quiescence_probe(formalizer, uuids), _quiescence_probe(formalizer, uuids), _quiescence_probe(formalizer, uuids),
    ])
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: next(observations))
    record = formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase="post_release")
    assert record["status"] == "PASS"
    assert [sample["consecutive_exact_idle_samples"] for sample in record["samples"]] == [1, 0, 1, 2, 3]
    assert [sample["transient_observation_count"] for sample in record["samples"]] == [0, 1, 1, 1, 1]


def test_gpu_quiescence_v1_v2_remain_immutable_and_v3_is_versioned_180_seconds() -> None:
    v1 = json.loads(Path("schemas/stage1/s1-8-gpu-quiescence-v1.json").read_text(encoding="utf-8"))
    v2 = json.loads(Path("schemas/stage1/s1-8-gpu-quiescence-v2.json").read_text(encoding="utf-8"))
    v3 = json.loads(Path("schemas/stage1/s1-8-gpu-quiescence-v3.json").read_text(encoding="utf-8"))
    assert v1["properties"]["schema_version"] == {"const": "stage1-s1-8-gpu-quiescence-v1"}
    assert v1["properties"]["timeout_seconds"] == {"const": 30.0}
    assert "operational_timeout_basis" not in v1["properties"]
    assert v2["properties"]["schema_version"] == {"const": "stage1-s1-8-gpu-quiescence-v2"}
    assert v2["properties"]["timeout_seconds"] == {"const": 60.0}
    assert v2["properties"]["operational_timeout_basis"]["additionalProperties"] is False
    assert "max_transient_samples" not in v2["properties"]
    assert v3["properties"]["schema_version"] == {"const": "stage1-s1-8-gpu-quiescence-v3"}
    assert v3["properties"]["timeout_seconds"] == {"const": 180.0}
    assert v3["properties"]["max_transient_samples"] == {"const": 2}
    assert v3["properties"]["operational_timeout_basis"]["properties"]["maximum_sample_count"] == {"const": 9}


def test_gpu_quiescence_public_role_hash_and_schema_bindings_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer(); uuids = [f"GPU-{index:08x}-1111-2222-3333-444444444444" for index in range(4)]
    now = [0.0]
    monkeypatch.setattr(formalizer.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(formalizer.time, "sleep", lambda duration: now.__setitem__(0, now[0] + duration))
    probe = _quiescence_probe(formalizer, uuids)
    monkeypatch.setattr(formalizer, "_probe_approved_gpus", lambda _uuids: probe)
    records = {phase: formalizer.require_gpu_quiescence(uuids, work=tmp_path, phase=phase) for phase in formalizer.GPU_QUIESCENCE_ROLES}
    for phase, filename in formalizer.GPU_QUIESCENCE_ROLES.items():
        if phase != "reacquire_preflight":
            snapshot = {"prelease": "preflight.json", "post_worker": "post-worker-gpu.json", "post_release": "post-release-gpu.json"}[phase]
            formalizer._write(tmp_path / snapshot, formalizer._with_hash({"schema_version": "stage1-s1-8-gpu-preflight-v1", "status": "PASS", "gpu": records[phase]["final_gpu"]}))
    sources = {filename: tmp_path / filename for filename in formalizer.GPU_QUIESCENCE_ROLES.values()}
    sources.update({"preflight.json": tmp_path / "preflight.json", "post-worker-gpu.json": tmp_path / "post-worker-gpu.json", "post-release-gpu.json": tmp_path / "post-release-gpu.json"})
    bindings = {phase: {"ref": filename, "sha256": formalizer._sha(tmp_path / filename)} for phase, filename in formalizer.GPU_QUIESCENCE_ROLES.items()}
    index_refs = {phase + "_gpu_quiescence": filename for phase, filename in formalizer.GPU_QUIESCENCE_ROLES.items()}
    index_hashes = {phase + "_gpu_quiescence": formalizer._sha(tmp_path / filename) for phase, filename in formalizer.GPU_QUIESCENCE_ROLES.items()}
    ddp, validation, index = {"gpu_quiescence": bindings}, {"gpu_quiescence": bindings}, {"reproduction_role_refs": index_refs, "reproduction_role_sha256": index_hashes}
    formalizer._validate_gpu_quiescence_publication(repository=Path("."), ddp=ddp, validation=validation, index=index, source_files=sources)
    wrong_ref = copy.deepcopy(ddp); wrong_ref["gpu_quiescence"]["post_worker"]["ref"] = "wrong.json"
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_GPU_QUIESCENCE_REFERENCE_INVALID"):
        formalizer._validate_gpu_quiescence_publication(repository=Path("."), ddp=wrong_ref, validation=validation, index=index, source_files=sources)
    wrong_hash = copy.deepcopy(validation); wrong_hash["gpu_quiescence"]["post_release"]["sha256"] = "0" * 64
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_GPU_QUIESCENCE_HASH_DRIFT"):
        formalizer._validate_gpu_quiescence_publication(repository=Path("."), ddp=ddp, validation=wrong_hash, index=index, source_files=sources)
    malformed = copy.deepcopy(records["post_worker"]); malformed["unbound"] = True
    malformed["artifact_hash"] = formalizer._canonical({key: value for key, value in malformed.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
        formalizer._validate_output_schemas(Path("."), {"gpu_quiescence": malformed})
    malformed_basis = copy.deepcopy(records["prelease"]); malformed_basis["operational_timeout_basis"]["unbound"] = True
    malformed_basis["artifact_hash"] = formalizer._canonical({key: value for key, value in malformed_basis.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
        formalizer._validate_output_schemas(Path("."), {"gpu_quiescence": malformed_basis})
    missing_transient_limit = copy.deepcopy(records["prelease"]); del missing_transient_limit["max_transient_samples"]
    missing_transient_limit["artifact_hash"] = formalizer._canonical({key: value for key, value in missing_transient_limit.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
        formalizer._validate_output_schemas(Path("."), {"gpu_quiescence": missing_transient_limit})
    legacy_wire = copy.deepcopy(records["prelease"]); legacy_wire["schema_version"] = "stage1-s1-8-gpu-quiescence-v2"
    legacy_wire["artifact_hash"] = formalizer._canonical({key: value for key, value in legacy_wire.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:gpu_quiescence"):
        formalizer._validate_output_schemas(Path("."), {"gpu_quiescence": legacy_wire})


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
        "schema_version": "stage1-s1-8-formalization-index-v5", "status": "PASS", "gate_id": "G1-DDP", "task_id": "stage1.08_ddp_and_gradient_accumulation",
        "generator_git_commit": "8" * 40, "consumer_git_commit": "8" * 40,
        "fixture_schema_version": "stage1-s1-8-fixture-manifest-v3", "fixture_id": "stage1-s1-8-pythia14m-ddp-conditioned-v3",
        "gpu_capability": {"commit_ref": "commit", "object_ref": "object", "task_id": "stage0.01_baseline_and_safety", "artifact_kind": "capability_cuda", "artifact_hash": formalizer.EXPECTED_GPU_CAPABILITY_ARTIFACT_HASH, "config_hash": "9" * 64, "source_refs": ["source"], "allowed_gpu_uuids": uuids}, "nccl_transport_protocol": formalizer._nccl_transport_protocol(),
        "implementation_source_sha256": formalizer._implementation_source_map(Path(".")), "s1_7_handoff": handoff,
        "role_refs": {"fixture_manifest": "fixture-manifest.json", "ddp_report": "ddp-report.json", "array_bundle": "array-bundle.json", "comparison_table": "comparison-table.json", "gate_record": "g1-ddp-record.json"},
        "role_sha256": {key: "a" * 64 for key in ("fixture_manifest", "ddp_report", "array_bundle", "comparison_table", "gate_record")},
        "reproduction_role_refs": {role: published for role, (published, _source) in formalizer._fixed_reproduction_roles().items()}, "reproduction_role_sha256": {role: "b" * 64 for role in formalizer._fixed_reproduction_roles()},
        "gate_artifact_hash": "c" * 64, "validation_ref": "validation.json", "validation_sha256": "d" * 64, "replay_ref": "replay-validation.json", "replay_sha256": "e" * 64,
        "next_task_ids": ["stage1.10_checkpoint_resume_and_artifacts"],
    })
    formalizer._validate_output_schemas(Path("."), {"index": index})
    for legacy_version in range(1, 5):
        legacy = copy.deepcopy(index); legacy["schema_version"] = f"stage1-s1-8-formalization-index-v{legacy_version}"
        legacy["artifact_hash"] = formalizer._canonical({key: value for key, value in legacy.items() if key != "artifact_hash"})
        with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
            formalizer._validate_output_schemas(Path("."), {"index": legacy})
    unknown = copy.deepcopy(index); unknown["s1_7_handoff"]["unbound"] = True; unknown["artifact_hash"] = formalizer._canonical({key: value for key, value in unknown.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": unknown})
    missing = copy.deepcopy(index); del missing["s1_7_handoff"]["historical_g3_current_consumer_source_sha256"]; missing["artifact_hash"] = formalizer._canonical({key: value for key, value in missing.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": missing})
    transport = copy.deepcopy(index); transport["nccl_transport_protocol"]["process_group_timeout_seconds"] = 120; transport["artifact_hash"] = formalizer._canonical({key: value for key, value in transport.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": transport})
    substituted = copy.deepcopy(index); roles = substituted["reproduction_role_refs"]
    roles["run_root_b91aa61c209eb19f"] = roles["run_root_1f7b55476eefcf3b"]
    substituted["artifact_hash"] = formalizer._canonical({key: value for key, value in substituted.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": substituted})
    extra_role = copy.deepcopy(index); extra_role["reproduction_role_sha256"]["role0"] = "b" * 64
    extra_role["artifact_hash"] = formalizer._canonical({key: value for key, value in extra_role.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": extra_role})
    missing_role = copy.deepcopy(index); del missing_role["reproduction_role_sha256"]["fixture_inputs"]
    missing_role["artifact_hash"] = formalizer._canonical({key: value for key, value in missing_role.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": missing_role})
    source_substitution = copy.deepcopy(index)
    digest = source_substitution["implementation_source_sha256"].pop("ops/stage1/run_s1_8_worker.py")
    source_substitution["implementation_source_sha256"]["ops/stage1/not-an-s1-8-source.py"] = digest
    source_substitution["artifact_hash"] = formalizer._canonical({key: value for key, value in source_substitution.items() if key != "artifact_hash"})
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED:index"):
        formalizer._validate_output_schemas(Path("."), {"index": source_substitution})


def test_current_critical_sources_are_exactly_zero_drift_from_dcc925_attestation() -> None:
    formalizer = _formalizer(); repository = Path(".").resolve()
    attestation = {"consumer_source_sha256": {path: formalizer._sha(repository / path) for path in formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS}}
    audit = formalizer._current_historical_g3_compatibility(repository, attestation)
    assert audit["s1_7_producer_to_current_critical_source_diff"] == []
    assert audit["critical_source_diff"] == list(formalizer.HISTORICAL_G3_CRITICAL_SOURCE_REFS)
    assert audit["critical_patch_sha256"] == formalizer.EXPECTED_HISTORICAL_G3_PATCH_SHA256
