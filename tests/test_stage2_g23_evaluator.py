"""Fail-closed and content-addressing tests for the independent G2.3 evaluator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_g23_evaluator import (
    CellInput,
    CORRECTED_DELTA_BATCH_SIZES,
    CORRECTED_DELTA_SIDECAR_FIELDS,
    CORRECTED_DELTA_SCHEMA_VERSION,
    EXPECTED_CELL_IDS,
    G23Blocked,
    _CellEvidence,
    _array,
    _bootstrap_independent_bias_interval,
    _bootstrap_independent_cross_interval,
    _bootstrap_u_diagnostics,
    _bounded_moments_strict,
    _delta_sci,
    _moments_from_blocks,
    _pearson,
    _top_overlap,
    _validate_six_cell_manifest,
    evaluate_formal_g23,
)
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_formal import (
    ReferenceSizingPlan,
    _BoundedCheckpointStore,
    _BoundedMoments,
    _ReferenceShardStore,
    _moments_from_shards,
    estimate_reference_uncertainty_shards,
)
from param_importance_nlp.experiments.stage2_g23_contracts import (
    generator_boundary,
    validate_generator_boundary,
    validate_resume_prefix,
    validate_sizing_plan_contract,
    validate_weighting_contract,
)
from param_importance_nlp.experiments.stage23_task_runners import _actual_sampling_state, _reference_capacity_preflight, _sizing_delta_sci


def _six_cell_manifest_for_registry_hashes(
    registry_hashes: tuple[str, ...],
    *,
    include_map: bool = True,
) -> dict[str, object]:
    assert len(registry_hashes) == len(EXPECTED_CELL_IDS)
    rows = [
        {
            "cell_id": cell_id,
            "model_id": cell_id.split(":", 1)[0],
            "training_stage": cell_id.split(":", 1)[1],
            "checkpoint_id": f"checkpoint-{index}",
            "checkpoint_hash": f"{index + 1:064x}",
            "checkpoint_revision": f"revision-{index}",
            "registry_hash": registry_hash,
            "config_hash": f"{index + 101:064x}",
        }
        for index, (cell_id, registry_hash) in enumerate(zip(EXPECTED_CELL_IDS, registry_hashes))
    ]
    by_cell = dict(zip(EXPECTED_CELL_IDS, registry_hashes))
    body: dict[str, object] = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY",
        "scope": "formal",
        "asset_resolution_hash": "a" * 64,
        "asset_producer_commit": "b" * 40,
        "asset_execution_commit": "c" * 40,
        "checkpoints": rows,
        "data": {"data_range_hash": "d" * 64},
        "data_range_hash": "d" * 64,
        "registry_hash": (
            next(iter(set(registry_hashes)))
            if len(set(registry_hashes)) == 1
            else canonical_json_hash(by_cell)
        ),
    }
    if include_map:
        body["registry_hashes_by_cell"] = by_cell
    body["manifest_hash"] = canonical_json_hash(body)
    return body


_PROVENANCE_SOURCE_PATHS = (
    "src/param_importance_nlp/experiments/stage2_formal.py",
    "src/param_importance_nlp/experiments/stage23_task_runners.py",
    "src/param_importance_nlp/experiments/stage2_g23_evaluator.py",
    "ops/stage2/evaluate_s204_g23.py",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _make_provenance_repo(tmp_path: Path, source_root: Path) -> tuple[Path, str, dict[str, object]]:
    repo = tmp_path / "producer-repo"
    for relative in _PROVENANCE_SOURCE_PATHS:
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, target)
    (repo / "tracked-marker.txt").write_text("clean\n", encoding="utf-8")
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "g23-test@example.invalid")
    _git(repo, "config", "user.name", "G23 test")
    _git(repo, "add", "--all")
    _git(repo, "commit", "--quiet", "-m", "producer fixture")
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    source_bytes = []
    for relative in _PROVENANCE_SOURCE_PATHS:
        source_bytes.append(
            {
                "path": relative,
                "sha256": hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
                "git_blob": _git(repo, "hash-object", "--", relative),
            }
        )
    provenance: dict[str, object] = {
        "schema_version": "stage2-reference-producer-provenance-v2",
        "repository_root_name": repo.name,
        "head_commit": head,
        "head_tree": tree,
        "tracked_clean": True,
        "source_bytes": source_bytes,
    }
    provenance["provenance_hash"] = canonical_json_hash(
        {
            "head_commit": head,
            "head_tree": tree,
            "tracked_clean": True,
            "source_bytes": source_bytes,
        }
    )
    return repo, head, provenance


def test_six_cell_manifest_binds_model_specific_registry_hashes() -> None:
    model_specific = ("1" * 64,) * 3 + ("2" * 64,) * 3

    rows = _validate_six_cell_manifest(
        _six_cell_manifest_for_registry_hashes(model_specific)
    )

    assert tuple(row["cell_id"] for row in rows) == EXPECTED_CELL_IDS
    assert tuple(row["registry_hash"] for row in rows) == model_specific


@pytest.mark.parametrize(
    ("tamper", "expected_error"),
    (
        ("map", "REGISTRY_ROW_MAP_MISMATCH"),
        ("row", "REGISTRY_ROW_MAP_MISMATCH"),
        ("top", "REGISTRY_DIGEST_MISMATCH"),
    ),
)
def test_six_cell_manifest_registry_binding_tamper_is_rejected(
    tamper: str,
    expected_error: str,
) -> None:
    model_specific = ("1" * 64,) * 3 + ("2" * 64,) * 3
    manifest = _six_cell_manifest_for_registry_hashes(model_specific)
    if tamper == "map":
        assert isinstance(manifest["registry_hashes_by_cell"], dict)
        manifest["registry_hashes_by_cell"][EXPECTED_CELL_IDS[0]] = "3" * 64  # type: ignore[index]
    elif tamper == "row":
        assert isinstance(manifest["checkpoints"], list)
        manifest["checkpoints"][0]["registry_hash"] = "3" * 64  # type: ignore[index]
    else:
        manifest["registry_hash"] = "3" * 64
    manifest["manifest_hash"] = canonical_json_hash(
        {key: item for key, item in manifest.items() if key != "manifest_hash"}
    )

    with pytest.raises(G23Blocked, match=expected_error):
        _validate_six_cell_manifest(manifest)


def test_six_cell_manifest_legacy_common_registry_form_is_explicit() -> None:
    common = ("1" * 64,) * 6
    legacy = _six_cell_manifest_for_registry_hashes(common, include_map=False)
    assert tuple(row["registry_hash"] for row in _validate_six_cell_manifest(legacy)) == common

    model_specific_without_map = _six_cell_manifest_for_registry_hashes(
        ("1" * 64,) * 3 + ("2" * 64,) * 3,
        include_map=False,
    )
    with pytest.raises(G23Blocked, match="LEGACY_COMMON_REGISTRY_REQUIRED"):
        _validate_six_cell_manifest(model_specific_without_map)


def test_evaluator_and_producer_repositories_are_separately_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PASS keeps producer 8dade and the running evaluator lineage distinct."""

    import param_importance_nlp.experiments.stage2_g23_evaluator as evaluator

    module_path = Path(evaluator.__file__).resolve()
    evaluator_repo = module_path.parents[3]
    evaluator_head, evaluator_source_sha256 = evaluator._validate_evaluator_provenance(
        evaluator_repo, module_path=module_path
    )
    producer_repo, producer_head, producer_provenance = _make_provenance_repo(
        tmp_path, evaluator_repo
    )
    assert producer_head != evaluator_head
    producer_convergence = {
        "producer_provenance": producer_provenance,
        "stage2_reference_producer_commit": producer_head,
    }
    assert evaluator._validate_producer_provenance(
        producer_convergence, repo_root=producer_repo
    ) == producer_head

    observed: dict[str, object] = {}

    def fake_prepare(
        root: Path, source: CellInput, *, repo_root: Path | None = None
    ) -> _CellEvidence:
        observed["producer_repo"] = repo_root
        evaluator._validate_producer_provenance(
            producer_convergence, repo_root=repo_root
        )
        index = EXPECTED_CELL_IDS.index(source.cell_id)
        evidence = _CellEvidence(source, workspace_root=root)
        evidence.identities.update(
            {
                "cell_id": source.cell_id,
                "result_hash": f"{index + 1:064x}",
                "config_hash": f"{index + 11:064x}",
                "checkpoint_hash": f"{index + 21:064x}",
                "one_shot_result_hash": f"{index + 31:064x}",
                "stream_a_draw_hash": f"{index + 41:064x}",
                "stream_b_draw_hash": f"{index + 51:064x}",
                "bundle_manifest_hash": f"{index + 61:064x}",
                "sizing_plan_hash": f"{index + 71:064x}",
                "reference_id": f"reference-{index}",
                "one_shot_plan_hash": f"{index + 81:064x}",
                "six_cell_manifest_hash": "f" * 64,
                "producer_commit": producer_head,
            }
        )
        return evidence

    def fake_evaluate(
        evidence: _CellEvidence,
        *,
        evaluator_commit: str,
        evaluator_source_sha256: str,
    ) -> dict[str, object]:
        observed["evaluator_commit"] = evaluator_commit
        observed["evaluator_source_sha256"] = evaluator_source_sha256
        return {
            "cell_id": evidence.source.cell_id,
            "status": "PASS",
            "identities": dict(evidence.identities),
            "metrics": {},
            "reasons": [],
        }

    monkeypatch.setattr(evaluator, "_prepare_cell", fake_prepare)
    monkeypatch.setattr(evaluator, "_evaluate_cell", fake_evaluate)
    cells = [CellInput(cell_id, f"unused/{index}.json") for index, cell_id in enumerate(EXPECTED_CELL_IDS)]
    result = evaluator.evaluate_formal_g23(
        tmp_path,
        cells,
        output_root=tmp_path / "attempts",
        repo_root=producer_repo,
    )
    assert result["status"] == "PASS"
    assert result["formal_eligible"] is True
    calculator = result["calculator"]
    assert isinstance(calculator, dict)
    assert calculator["producer_commit"] == producer_head
    assert calculator["evaluator_commit"] == evaluator_head
    assert calculator["source_sha256"] == evaluator_source_sha256
    assert observed == {
        "producer_repo": producer_repo.resolve(),
        "evaluator_commit": evaluator_head,
        "evaluator_source_sha256": evaluator_source_sha256,
    }

    with pytest.raises(G23Blocked, match="REPOSITORY_HEAD_OR_CLEAN_STATE_DRIFT"):
        evaluator.evaluate_formal_g23(
            tmp_path,
            cells,
            output_root=tmp_path / "swapped",
            repo_root=evaluator_repo,
        )

    dirty_repo, _dirty_head, dirty_provenance = _make_provenance_repo(
        tmp_path / "dirty", evaluator_repo
    )
    (dirty_repo / "tracked-marker.txt").write_text("dirty\n", encoding="utf-8")
    dirty_convergence = {
        "producer_provenance": dirty_provenance,
        "stage2_reference_producer_commit": _dirty_head,
    }
    with pytest.raises(G23Blocked, match="REPOSITORY_HEAD_OR_CLEAN_STATE_DRIFT"):
        evaluator._validate_producer_provenance(
            dirty_convergence, repo_root=dirty_repo
        )

    tampered_repo, tampered_head, tampered_provenance = _make_provenance_repo(
        tmp_path / "tampered", evaluator_repo
    )
    tampered_path = tampered_repo / evaluator._EVALUATOR_SOURCE_RELATIVE
    tampered_path.write_bytes(tampered_path.read_bytes() + b"\n# tampered\n")
    tampered_convergence = {
        "producer_provenance": tampered_provenance,
        "stage2_reference_producer_commit": tampered_head,
    }
    with pytest.raises(G23Blocked, match="SOURCE_DRIFT"):
        evaluator._validate_producer_provenance(
            tampered_convergence, repo_root=tampered_repo
        )

    evaluator_copy, _copy_head, _copy_provenance = _make_provenance_repo(
        tmp_path / "evaluator-copy", evaluator_repo
    )
    copy_module = evaluator_copy / evaluator._EVALUATOR_SOURCE_RELATIVE
    copy_module.write_bytes(copy_module.read_bytes() + b"\n# evaluator tampered\n")
    with pytest.raises(G23Blocked, match="TRACKED_FILES_NOT_CLEAN"):
        evaluator._validate_evaluator_provenance(
            evaluator_copy, module_path=copy_module
        )


def test_missing_cells_are_not_a_formal_decision(tmp_path: Path) -> None:
    result = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    assert result["status"] == "NOT_RUN"
    assert result["formal_eligible"] is False
    assert result["complete_cell_count"] == 0
    assert (tmp_path / "attempts" / "g2.3-attempts" / result["artifact_hash"] / "evaluation.json").is_file()


def test_partial_cell_set_is_blocked_and_does_not_lock_next_attempt(tmp_path: Path) -> None:
    cells = [CellInput("cell-0", "runs/cell-0/task-run-result.json")]
    first = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert first["status"] == "BLOCKED"
    # A later complete set gets a different content address; no stale partial
    # attempt can be overwritten or treated as the current formal decision.
    second = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    assert first["artifact_hash"] != second["artifact_hash"]
    lines = (tmp_path / "attempts" / "g2.3-attempts.jsonl").read_text(encoding="utf-8").splitlines()
    assert first["artifact_hash"] in lines and second["artifact_hash"] in lines


def test_nonfinite_raw_array_is_rejected_closed() -> None:
    with pytest.raises(ValueError, match="NON_FINITE"):
        _array([1.0, float("nan")], "diagnostic")


def test_boundary_metrics_use_inclusive_preregistered_comparisons() -> None:
    left = np.asarray([1.0, 2.0, 3.0, 4.0])
    right = np.asarray([1.0, 2.0, 3.0, 4.0])
    assert _pearson(left, right) >= 0.995
    assert _top_overlap(left, right, 0.01) >= 0.98


def test_weighted_u_hand_calculation_is_recomputed_from_raw_blocks() -> None:
    blocks = [
        {"p": np.asarray([1.0])},
        {"p": np.asarray([3.0])},
    ]
    moments = _moments_from_blocks(blocks, [1.0, 2.0], "hand")
    # n1=3, n2=5, g1=7, g2=37 => (49-37)/(9-5)=3.
    from param_importance_nlp.experiments.stage2_g23_evaluator import _u_from_moments

    assert np.array_equal(_u_from_moments(moments, "hand.u")["p"], np.asarray([3.0]))
    assumptions = validate_weighting_contract({
        "statistical_unit": "sequence",
        "weight_unit": "tokens",
        "sampling_design": "with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    })
    assert assumptions["weights_exogenous"] is True
    with pytest.raises(ValueError, match="WEIGHTED_U_ASSUMPTIONS_NOT_DECLARED"):
        validate_weighting_contract({
            **assumptions,
            "common_mean_assumption": False,
        })


def test_content_addressed_shard_dedup_and_weighted_jackknife(tmp_path: Path) -> None:
    store = _ReferenceShardStore(tmp_path / "sizing")
    vector = {"p": np.asarray([1.0, 2.0])}
    first = store.publish(vector, 1.0)
    duplicate = store.publish(vector, 1.0)
    assert first == duplicate
    refs = [first, store.publish({"p": np.asarray([2.0, 4.0])}, 2.0), store.publish({"p": np.asarray([3.0, 6.0])}, 3.0)]
    assumptions = {
        "statistical_unit": "sequence",
        "weight_unit": "tokens",
        "sampling_design": "with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    moments = _moments_from_shards(store, refs, assumptions)
    assert moments.count == 3 and moments.n1 == 6.0
    uncertainty = estimate_reference_uncertainty_shards(store, refs, refs, assumptions)
    assert uncertainty.block_count_a == 3 and uncertainty.block_count_b == 3
    assert all(np.all(np.isfinite(value)) for value in uncertainty.bias_variance.values())


def test_independent_ab_bootstrap_and_endpoint_h_ref_are_block_bootstrapped() -> None:
    blocks_a = [{"p": np.asarray([value, value + 1.0])} for value in (1.0, 2.0, 3.0)]
    blocks_b = [{"p": np.asarray([value, value + 1.0])} for value in (1.5, 2.5, 3.5)]
    weights = [1.0, 2.0, 1.0]
    cross_low, cross_high = _bootstrap_independent_cross_interval(blocks_a, weights, blocks_b, weights, "hand.cross")
    bias_low, bias_high = _bootstrap_independent_bias_interval(blocks_a, weights, blocks_b, weights, "hand.bias")
    center = {"p": np.asarray([2.0, 3.0])}
    h_ref, model_half, layer_q95, module_q95 = _bootstrap_u_diagnostics(
        blocks_a,
        blocks_b,
        weights,
        weights,
        center,
        {"layer0": ["p"]},
        {"module0": ["p"]},
    )
    assert np.all(cross_low <= cross_high) and np.all(bias_low <= bias_high)
    assert np.isfinite(h_ref) and np.isfinite(model_half) and np.isfinite(layer_q95) and np.isfinite(module_q95)


def test_sampling_rng_state_is_replayable_from_frozen_manifest() -> None:
    sampling = SamplingPlan(
        universe=SamplingUniverse("hand-universe", (0, 1, 2, 3)),
        stream_seeds={"reference_sizing": 7, "reference_A": 11, "reference_B": 13, "pilot": 17, "confirmatory": 19},
    )
    state = _actual_sampling_state(sampling, "reference_A", 3)
    assert state["stream"] == "reference_A" and state["count"] == 3
    assert state["state_before_sha256"] != state["state_after_sha256"]
    assert state == generator_boundary(sampling, "reference_A", 3)
    validate_generator_boundary(state, sampling=sampling, stream="reference_A", count=3, field="hand.rng")


def test_canonical_sizing_and_resume_validators_are_strict() -> None:
    plan = {
        "schema_version": "stage2-reference-sizing-plan-v1",
        "candidate_sample_counts": [2, 4, 8],
        "block_size": 2,
        "required_consecutive": 1,
    }
    assert validate_sizing_plan_contract(plan, selected_sample_count=4) == (2, 4, 8)
    with pytest.raises(ValueError, match="ADJACENT_DOUBLING_REQUIRED"):
        validate_sizing_plan_contract({**plan, "candidate_sample_counts": [2, 6]})
    first = [{"shard_hash": "a"}]
    second = [{"shard_hash": "a"}, {"shard_hash": "b"}]
    validate_resume_prefix(first, [], second, [], field="hand.resume")
    with pytest.raises(ValueError, match="PREFIX_DRIFT_A"):
        validate_resume_prefix(first, [], [{"shard_hash": "x"}], [], field="hand.resume")


def test_sizing_delta_formula_uses_noise_or_signal_floor_at_boundary() -> None:
    assert _sizing_delta_sci(100.0, 2.0) == pytest.approx(1.0)
    assert _sizing_delta_sci(100.0, 30.0) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        _sizing_delta_sci(float("nan"), 1.0)


def test_corrected_delta_uses_convergence_selected_node_and_legacy_raw_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected node is authoritative in convergence, not the sizing plan."""

    import param_importance_nlp.experiments.stage2_g23_evaluator as evaluator

    evidence = _CellEvidence(
        CellInput("pythia-14m:initialization", "runs/task-result.json"),
        workspace_root=tmp_path,
    )
    sizing_root = tmp_path / "sizing"
    store = _ReferenceShardStore(sizing_root)
    refs = [
        store.publish({"p": np.asarray([float(index), float(index + 1)])}, 1.0)
        for index in range(4)
    ]
    assumptions = {
        "statistical_unit": "sequence",
        "weight_unit": "tokens",
        "sampling_design": "with_replacement",
        "weights_exogenous": True,
        "common_mean_assumption": True,
    }
    states = [
        {
            "a": {"g1": {}, "g2": {}},
            "b": {"g1": {}, "g2": {}},
            "last_bias": {},
            "blocks_a": [],
            "blocks_b": [],
            "moment_hash_mode": "shard_refs-v1",
            "sizing_stream": True,
            "weighting_assumptions": assumptions,
            "processed_block_pairs": count,
            "last_bias_block_pairs": count,
            "shard_refs_a": refs[:count],
            "shard_refs_b": [],
        }
        for count in (2, 4)
    ]
    evidence.sizing_root = sizing_root
    evidence.sizing_states = states
    mean = {"p": np.asarray([2.0, 3.0])}
    sigma2 = {"p": np.asarray([4.0, 5.0])}
    monkeypatch.setattr(
        evaluator,
        "_sizing_sequence_variance",
        lambda *_args: (mean, sigma2),
    )
    registry_hash = "1" * 64
    registry = {
        "schema_version": "stage2-parameter-registry-artifact-v1",
        "status": "READY",
        "registry_hash": registry_hash,
        "parameter_groups": {"p": {"layer": "layer0", "module": "module0"}},
    }
    registry["artifact_hash"] = canonical_json_hash(registry)
    evidence.identities["registry_hash"] = registry_hash
    evidence.identities["producer_commit"] = "5" * 40
    floors = {
        "tau_model": 1.0e-12,
        "tau_layer": 1.0e-12,
        "tau_module": 1.0e-12,
        "tau_coord": 1.0e-12,
        "tau_nmse": 1.0e-12,
    }
    formula_hash = "2" * 64
    preregistration = {
        "equivalence_and_precision": {"absolute_floors": floors},
    }
    evidence.external_payloads = {"preregistration": preregistration}
    plan = {
        "schema_version": "stage2-reference-sizing-plan-v1",
        "reference_id": "reference-test",
        "candidate_sample_counts": [2, 4],
        "block_size": 1,
        "required_consecutive": 1,
        "convergence_tolerance": 0.02,
    }
    plan["artifact_hash"] = canonical_json_hash(plan)
    nodes = []
    for count, state in zip((2, 4), states):
        nodes.append(
            {
                "sample_count": count,
                "state_digest": evaluator._ReferenceSnapshotStore._state_digest(state),
                "shard_refs_hash": canonical_json_hash(
                    [
                        {
                            "shard_hash": ref.get("shard_hash"),
                            "manifest_hash": ref.get("manifest_hash"),
                            "weight": ref.get("weight"),
                        }
                        for ref in state["shard_refs_a"]
                    ]
                ),
                "mean_hash": evaluator._vector_digest(mean),
                "sequence_variance_hash": evaluator._vector_digest(sigma2),
            }
        )
    old_delta_tables = {
        endpoint: {"2": 0.45, "4": 0.225}
        for endpoint in ("model_total", "layer", "module")
    }
    old_signal_tables = {
        endpoint: {"2": 13.0, "4": 13.0}
        for endpoint in ("model_total", "layer", "module")
    }
    old_noise_tables = {
        endpoint: {"2": 4.5, "4": 2.25}
        for endpoint in ("model_total", "layer", "module")
    }
    source_body = {
        "schema_version": "stage2-reference-delta-sci-v2",
        "source_kind": "reference_sizing_raw_shards",
        "formula_contract_hash": formula_hash,
        "formula_version": "stage2-reference-sizing-margin-v1",
        "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
        "absolute_floors": floors,
        "reference_id": plan["reference_id"],
        "sizing_result_hash": "3" * 64,
        "sizing_plan_hash": plan["artifact_hash"],
        "candidate_sample_counts": [2, 4],
        "delta_sci_by_endpoint": old_delta_tables,
        "signal_scale_by_endpoint": old_signal_tables,
        "noise_scale_by_endpoint": old_noise_tables,
        "sizing_nodes": nodes,
        "registry_hash": registry_hash,
    }
    source_body["artifact_hash"] = canonical_json_hash(source_body)
    source_ref = "sizing/derived/delta-sci.json"
    write_canonical_json(tmp_path / source_ref, source_body)
    source_link = {
        **source_body,
        "source_ref": source_ref,
        "source_hash": source_body["artifact_hash"],
        "source_artifact_hash": source_body["artifact_hash"],
    }
    evidence.convergence = SimpleNamespace(
        payload={
            "candidate_delta_sci": source_link,
            "sizing_plan": plan,
            "sizing_plan_artifact_hash": plan["artifact_hash"],
            "formula_contract_hash": formula_hash,
            "sizing_result_hash": "3" * 64,
            "parameter_registry_artifact": registry,
            # This is intentionally omitted from ``plan`` to prove that the
            # convergence report is the authoritative selected-node source.
            "selected_sample_count_per_stream": 4,
        }
    )

    _, minimum, sidecar = _delta_sci(
        evidence, [2, 4], evaluator_commit="4" * 40
    )
    assert tuple(sorted(minimum)) == ("layer", "model_total", "module")
    assert sidecar["schema_version"] == CORRECTED_DELTA_SCHEMA_VERSION
    assert sidecar["candidate_sample_counts"] == [2, 4]
    assert sidecar["delta_sci_batch_sizes"] == list(CORRECTED_DELTA_BATCH_SIZES)
    assert sidecar["selected_sample_count_per_stream"] == 4
    assert sidecar["source_producer_table_mode"] == "sizing_nodes_legacy"
    assert sidecar["correction_reason"] == (
        "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes"
    )
    for endpoint in ("model_total", "layer", "module"):
        assert set(sidecar["delta_sci_by_endpoint"][endpoint]) == {  # type: ignore[index]
            str(batch_size) for batch_size in CORRECTED_DELTA_BATCH_SIZES
        }
    expected_hash = canonical_json_hash(
        {key: value for key, value in sidecar.items() if key != "artifact_hash"}
    )
    assert set(sidecar) == CORRECTED_DELTA_SIDECAR_FIELDS
    assert sidecar["artifact_hash"] == expected_hash

    cell = {
        "cell_id": evidence.source.cell_id,
        "status": "PASS",
        "identities": {},
        "metrics": {},
        "_corrected_delta_sci": dict(sidecar),
    }
    evaluator._publish_corrected_delta_sidecars(
        tmp_path, tmp_path / "evidence" / "g23-attempt", [cell]
    )
    sidecar_ref = cell["identities"]["corrected_delta_sci_ref"]  # type: ignore[index]
    assert sidecar_ref == (
        "evidence/g23-attempt/g2.3-corrected-delta-sci/"
        f"{sidecar['artifact_hash']}.json"
    )
    assert (tmp_path / sidecar_ref).is_file()
    assert cell["identities"]["corrected_delta_sci_hash"] == sidecar["artifact_hash"]  # type: ignore[index]
    assert cell["metrics"]["corrected_delta_sci_ref"] == sidecar_ref  # type: ignore[index]

    future_source = dict(source_body)
    future_source.update(
        {
            "delta_sci_batch_sizes": list(CORRECTED_DELTA_BATCH_SIZES),
            "selected_sample_count_per_stream": 4,
            "delta_sci_by_endpoint": sidecar["delta_sci_by_endpoint"],
            "signal_scale_by_endpoint": sidecar["signal_scale_by_endpoint"],
            "noise_scale_by_endpoint": sidecar["noise_scale_by_endpoint"],
        }
    )
    future_source["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in future_source.items() if key != "artifact_hash"}
    )
    write_canonical_json(tmp_path / source_ref, future_source)
    evidence.convergence.payload["candidate_delta_sci"] = {
        **future_source,
        "source_ref": source_ref,
        "source_hash": future_source["artifact_hash"],
        "source_artifact_hash": future_source["artifact_hash"],
    }
    _, _, verified_sidecar = _delta_sci(
        evidence, [2, 4], evaluator_commit="4" * 40
    )
    assert verified_sidecar["source_producer_table_mode"] == "candidate_batch_sizes"
    assert verified_sidecar["correction_reason"] == (
        "producer_delta_sci_candidate_batch_sizes_verified_by_evaluator"
    )

    future_tampered = dict(future_source)
    future_tampered["delta_sci_batch_sizes"] = [131072, 262144]
    future_tampered["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in future_tampered.items() if key != "artifact_hash"}
    )
    write_canonical_json(tmp_path / source_ref, future_tampered)
    evidence.convergence.payload["candidate_delta_sci"] = {
        **future_tampered,
        "source_ref": source_ref,
        "source_hash": future_tampered["artifact_hash"],
        "source_artifact_hash": future_tampered["artifact_hash"],
    }
    with pytest.raises(G23Blocked, match="PRODUCER_BATCH_DOMAIN_BINDING_MISMATCH"):
        _delta_sci(evidence, [2, 4], evaluator_commit="4" * 40)

    tampered_source = dict(source_body)
    tampered_delta = {
        endpoint: dict(values)
        for endpoint, values in old_delta_tables.items()
    }
    tampered_delta["model_total"]["2"] = 0.44
    tampered_source["delta_sci_by_endpoint"] = tampered_delta
    tampered_source["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in tampered_source.items() if key != "artifact_hash"}
    )
    write_canonical_json(tmp_path / source_ref, tampered_source)
    evidence.convergence.payload["candidate_delta_sci"] = {
        **tampered_source,
        "source_ref": source_ref,
        "source_hash": tampered_source["artifact_hash"],
        "source_artifact_hash": tampered_source["artifact_hash"],
    }
    with pytest.raises(G23Blocked, match="LEGACY_PRODUCER_FORMULA_RECOMPUTE_MISMATCH"):
        _delta_sci(evidence, [2, 4], evaluator_commit="4" * 40)


def test_corrected_delta_uses_bounded_checkpoint_and_rejects_state_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounded r23 moments are reloaded, then legacy values are rechecked."""

    import param_importance_nlp.experiments.stage2_g23_evaluator as evaluator

    evidence = _CellEvidence(
        CellInput("pythia-14m:initialization", "runs/task-result.json"),
        workspace_root=tmp_path,
    )
    sizing_root = tmp_path / "sizing"
    moments_by_count: dict[int, _BoundedMoments] = {}
    for count in (2, 4):
        moments = _BoundedMoments(include_higher=False)
        for index in range(count):
            moments.update_vector(
                {"p": np.asarray([float(index), float(index + 1)])}, 1.0
            )
        moments_by_count[count] = moments
    empty = _BoundedMoments(include_higher=False).to_state()
    states = [
        {
            "a": moments_by_count[count].to_state(),
            "b": empty,
            "processed_block_pairs": count,
            "sizing_stream": True,
            "weighting_assumptions": {
                "statistical_unit": "sequence",
                "weight_unit": "tokens",
                "sampling_design": "with_replacement",
                "weights_exogenous": True,
                "common_mean_assumption": True,
            },
        }
        for count in (2, 4)
    ]
    _BoundedCheckpointStore(sizing_root).publish(
        4,
        {
            "selected_sample_count_per_stream": 4,
            "candidate_states": {
                str(count): {"a": state["a"], "b": state["b"]}
                for count, state in zip((2, 4), states)
            },
            "a": states[-1]["a"],
            "b": states[-1]["b"],
        },
    )
    evidence.sizing_root = sizing_root
    evidence.sizing_states = states
    sigma2 = {"p": np.asarray([4.0, 5.0])}
    monkeypatch.setattr(
        evaluator,
        "estimate_sequence_variance_bounded",
        lambda *_args, **_kwargs: sigma2,
    )
    registry_hash = "6" * 64
    registry = {
        "schema_version": "stage2-parameter-registry-artifact-v1",
        "status": "READY",
        "registry_hash": registry_hash,
        "parameter_groups": {"p": {"layer": "layer0", "module": "module0"}},
    }
    registry["artifact_hash"] = canonical_json_hash(registry)
    evidence.identities.update({
        "registry_hash": registry_hash,
        "producer_commit": "7" * 40,
    })
    floors = {
        "tau_model": 1.0e-12,
        "tau_layer": 1.0e-12,
        "tau_module": 1.0e-12,
        "tau_coord": 1.0e-12,
        "tau_nmse": 1.0e-12,
    }
    formula_hash = "8" * 64
    evidence.external_payloads = {
        "preregistration": {
            "equivalence_and_precision": {"absolute_floors": floors}
        }
    }
    plan = {
        "schema_version": "stage2-reference-sizing-plan-v1",
        "reference_id": "bounded-reference-test",
        "candidate_sample_counts": [2, 4],
        "block_size": 1,
        "required_consecutive": 1,
        "convergence_tolerance": 0.02,
    }
    plan["artifact_hash"] = canonical_json_hash(plan)
    nodes = [
        {
            "sample_count": count,
            "state_digest": canonical_json_hash({
                "checkpoint_schema": _BoundedCheckpointStore.schema_version,
                "plan_hash": plan["artifact_hash"],
                "sample_count": count,
                "moments_hash": evaluator._bounded_moments_digest(moments_by_count[count]),
            }),
            "shard_refs_hash": canonical_json_hash([]),
            "mean_hash": evaluator._vector_digest(
                moments_by_count[count].mean()
            ),
            "sequence_variance_hash": evaluator._vector_digest(sigma2),
        }
        for count in (2, 4)
    ]
    old_delta = {endpoint: {} for endpoint in ("model_total", "layer", "module")}
    old_signal = {endpoint: {} for endpoint in ("model_total", "layer", "module")}
    old_noise = {endpoint: {} for endpoint in ("model_total", "layer", "module")}
    for count, moments in moments_by_count.items():
        mean = moments.mean()
        signal = float(sum(np.sum(np.square(value)) for value in mean.values()))
        noise = float(sum(np.sum(value) for value in sigma2.values())) / count
        for endpoint in old_delta:
            old_signal[endpoint][str(count)] = signal
            old_noise[endpoint][str(count)] = noise
            old_delta[endpoint][str(count)] = max(0.10 * noise, 0.01 * signal)
    source_body = {
        "schema_version": "stage2-reference-delta-sci-v2",
        "source_kind": "reference_sizing_bounded_online",
        "formula_contract_hash": formula_hash,
        "formula_version": "stage2-reference-sizing-margin-v1",
        "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
        "absolute_floors": floors,
        "reference_id": plan["reference_id"],
        "sizing_result_hash": "9" * 64,
        "sizing_plan_hash": plan["artifact_hash"],
        "candidate_sample_counts": [2, 4],
        "delta_sci_by_endpoint": old_delta,
        "signal_scale_by_endpoint": old_signal,
        "noise_scale_by_endpoint": old_noise,
        "sizing_nodes": nodes,
        "registry_hash": registry_hash,
    }
    source_body["artifact_hash"] = canonical_json_hash(source_body)
    source_ref = "sizing/derived/delta-sci.json"
    write_canonical_json(tmp_path / source_ref, source_body)
    source_link = {
        **source_body,
        "source_ref": source_ref,
        "source_hash": source_body["artifact_hash"],
        "source_artifact_hash": source_body["artifact_hash"],
    }
    evidence.convergence = SimpleNamespace(
        payload={
            "candidate_delta_sci": source_link,
            "sizing_plan": plan,
            "sizing_plan_artifact_hash": plan["artifact_hash"],
            "formula_contract_hash": formula_hash,
            "sizing_result_hash": "9" * 64,
            "parameter_registry_artifact": registry,
            "selected_sample_count_per_stream": 4,
        }
    )
    _, _, sidecar = _delta_sci(
        evidence, [2, 4], evaluator_commit="a" * 40
    )
    assert sidecar["source_producer_table_mode"] == "sizing_nodes_legacy"
    assert sidecar["selected_sample_count_per_stream"] == 4
    assert set(sidecar["delta_sci_by_endpoint"]["model_total"]) == {  # type: ignore[index]
        str(batch_size) for batch_size in CORRECTED_DELTA_BATCH_SIZES
    }

    tampered_state = dict(states[-1])
    tampered_a = dict(tampered_state["a"])
    tampered_a["n2"] = float(tampered_a["n2"]) + 1.0  # type: ignore[arg-type]
    tampered_state["a"] = tampered_a
    evidence.sizing_states[-1] = tampered_state
    with pytest.raises(G23Blocked, match="SIZING_NODE_BINDING_MISMATCH"):
        _delta_sci(evidence, [2, 4], evaluator_commit="a" * 40)


def test_bounded_final_moments_missing_higher_fields_are_blocked() -> None:
    moments = _BoundedMoments(include_higher=False)
    moments.update_vector({"p": np.asarray([1.0, 2.0])}, 1.0)

    with pytest.raises(G23Blocked, match="HIGHER_MOMENTS_REQUIRED"):
        _bounded_moments_strict(
            moments.to_state(), "bounded.final", require_higher=True
        )


def test_capacity_preflight_uses_full_14m_and_31m_counts(tmp_path: Path) -> None:
    class _Provider:
        pass

    plan = ReferenceSizingPlan(
        reference_id="capacity-test",
        candidate_sample_counts=(2, 4),
        block_size=1,
        convergence_tolerance=0.02,
        required_consecutive=1,
    )
    reports = [
        _reference_capacity_preflight(_Provider(), plan, tmp_path, model_manifest={"parameter_count": count})
        for count in (14_000_000, 31_000_000)
    ]
    assert [item["parameter_count"] for item in reports] == [14_000_000, 31_000_000]
    for item, count in zip(reports, (14_000_000, 31_000_000)):
        assert item["single_copy_shard_bytes"] == 4 * 2 * count * 8
        assert item["snapshot_moment_bytes"] == 4 * 4 * count * 8
        assert item["disk_ok"] is True and item["ram_ok"] is True


def test_attempt_json_is_hash_bound_and_tamper_detected(tmp_path: Path) -> None:
    result = evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
    path = tmp_path / "attempts" / "g2.3-attempts" / result["artifact_hash"] / "evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = "PASS"
    path.write_text(json.dumps(payload), encoding="utf-8")
    # Re-evaluation does not trust an existing attempt and must detect the
    # content-address collision rather than silently accepting a tampered file.
    with pytest.raises(RuntimeError, match="CONTENT_ADDRESS_COLLISION"):
        evaluate_formal_g23(tmp_path, [], output_root=tmp_path / "attempts")
