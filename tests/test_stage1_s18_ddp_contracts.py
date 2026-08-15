"""CPU regressions for the S1.8 route and FP64-replay contracts."""

from __future__ import annotations

import math
import importlib.util
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from param_importance_nlp.stage1_ddp import (
    Stage1S18Error,
    build_fixture,
    learning_rate_map,
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
        return {"case": name, "global_loss_numerator": 1.0, "global_loss_valid_token_count": 8, "global_mean_loss": 0.125, "global_microbatch_count": 8, "global_n1": 8, "global_n2": 8, "global_gradient_norm": 1.0, "clip_factor": 1.0, "rank_records": [{"rank": 0, "local_microbatch_ids": list(range(8)), "local_gradient_checksums": ["c" * 64] * 8, "global_statistic_checksums": {}, "local_loss_numerator": 1.0, "local_effective_tokens": 8}], "global_statistic_checksums": {}, "ordinary_ddp_gradient_collectives": 0, "manual_statistic_collectives": {"backend": "nccl", "operation": "SUM", "tensor_statistics": [], "tensor_all_reduce_count": 0, "scalar_statistics": [], "scalar_all_reduce_count": 0, "total_all_reduce_count": 0}, "post_parameter_checksum": "d" * 64, "pre_parameter_checksum": "e" * 64, "accumulator": None, "array_keys": [f"a-reference/{name}/raw_core/p"]}
    uuid = "GPU-00000000-1111-2222-3333-444444444444"
    return formalizer._with_hash({"schema_version": "stage1-s1-8-worker-report-v1", "status": "PASS", "task_id": "stage1.08_ddp_and_gradient_accumulation", "execution_commit": "f" * 40, "run_token": "0" * 64, "route": "A", "permutation": "identity", "execution_mode": "formal", "world_size": 1, "backend": "nccl", "visible_gpu_uuids": [uuid], "rank_to_gpu_uuid": [uuid], "parameter_registry_hash": "1" * 64, "fixture_hash": "2" * 64, "route_layout": {"route": "A", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [case("equal"), case("weighted")], "arrays": manifest})


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
    unknown = copy.deepcopy(report); unknown["cases"][0]["rank_records"][0]["intruder"] = 1
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": unknown})
    cardinality = copy.deepcopy(report); cardinality["cases"] = cardinality["cases"][:1]
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_SCHEMA_VALIDATION_FAILED"):
        formalizer._validate_output_schemas(Path("."), {"worker_report": cardinality})


def test_unknown_process_group_member_never_signals_and_persists_manual_marker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    formalizer = _formalizer()
    expected = {"pid": 12, "ppid": 1, "uid": 1000, "pgid": 44, "sid": 44, "start_ticks": "99", "exe": "/usr/bin/python", "cmdline_sha256": "a" * 64, "environment_run_token": "b" * 64}
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(formalizer, "_audit_exact_process_group", lambda _: (_ for _ in ()).throw(formalizer.Stage1S18ManualInterventionRequired("S18_PROCESS_GROUP_MEMBER_UNVERIFIABLE")))
    monkeypatch.setattr(formalizer, "_pgid_members", lambda _: [12, 13])
    monkeypatch.setattr(formalizer.os, "killpg", lambda pgid, sig: signals.append((pgid, sig)), raising=False)
    with pytest.raises(formalizer.Stage1S18ManualInterventionRequired):
        formalizer._terminate_exact(SimpleNamespace(wait=lambda timeout: None), expected, tmp_path, label="unknown-member")
    assert signals == []
    assert (tmp_path / "unknown-member-manual-intervention.json").is_file()


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
    report_a = {"route": "A", "world_size": 1, "visible_gpu_uuids": [uuid], "rank_to_gpu_uuid": [uuid], "route_layout": {"route": "A", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": case, "array_keys": [f"a-reference/{case}/{field}/p" for field in a_fields], "accumulator": None, "rank_records": [{"rank": 0}]} for case in ("equal", "weighted")]}
    formalizer._validate_worker_candidate_contract("A", report_a)
    bad_a = copy.deepcopy(report_a); bad_a["cases"][0]["array_keys"].append("scores/equal/u_core/p")
    with pytest.raises(formalizer.Stage1S18FormalError, match="S18_CANDIDATE_A_U_OR_ACCUMULATOR_FORBIDDEN"):
        formalizer._validate_worker_candidate_contract("A", bad_a)
    report_b = {**copy.deepcopy(report_a), "route": "B", "route_layout": {"route": "B", "world_size": 1, "rank_microbatch_ids": [list(range(8))]}, "cases": [{"case": "equal", "array_keys": ["scores/equal/u_core/p", "accumulator/equal/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0}]}, {"case": "weighted", "array_keys": ["scores/weighted/u_core/p", "accumulator/weighted/cumulative/signed/p"], "accumulator": None, "rank_records": [{"rank": 0}]}]}
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
        row["global_statistic_checksums"] = {name: "c" * 64 for name in stats}
        row["array_keys"] = [*(f"stats/{case}/{name}/p" for name in stats), *(f"scores/{case}/{name}/p" for name in ("mean_gradient", "raw_core", "u_core", "raw_score", "u_score", "u_score_clipped")), f"accumulator/{case}/cumulative/signed/p"]
        row["accumulator"] = {"successful_steps": ordinal, "skipped_steps": 0, "signed_identity": True, "absolute_identity": True, "contribution_checksums": contribution, "cumulative_checksums": cumulative}
    report.pop("artifact_hash"); report = formalizer._with_hash(report)
    formalizer._validate_output_schemas(Path("."), {"worker_report": report, "safetensors_manifest": report["arrays"]})
    formalizer._validate_worker_candidate_contract("B", report)


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
    proc = tmp_path / "42"; proc.mkdir(); sensitive = b"parameter pile --token=secret-do-not-persist"; (proc / "cmdline").write_bytes(sensitive); (proc / "stat").write_text(" ".join(["x"] * 22), encoding="utf-8")
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
