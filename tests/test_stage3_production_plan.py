"""Contract tests for the real Stage 3 production unit index."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from param_importance_nlp.cli import _validate_known_artifact, main
from param_importance_nlp.contracts import FormalExecutionEvidence, GateRecord, GateStatus
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage3_production_plan import (
    build_production_unit_index,
    load_production_unit_index,
    write_production_unit_index,
)
from param_importance_nlp.experiments.stage3_protocol import DEFAULT_CANDIDATE_RULES


def _h(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stage3_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _state(name: str, *, parameter: str, buffer: str, optimizer: str) -> dict[str, str]:
    return {
        "artifact_id": name,
        "artifact_hash": _h(f"artifact:{name}"),
        "parameter_hash": parameter,
        "buffer_hash": buffer,
        "optimizer_hash": optimizer,
        "scheduler_hash": _h(f"scheduler:{name}"),
        "scaler_hash": _h(f"scaler:{name}"),
        "rng_hash": _h(f"rng:{name}"),
        "data_cursor_hash": _h(f"cursor:{name}"),
        "model_mode_hash": _h(f"mode:{name}"),
    }


def _write_endpoint(root: Path, *, model: str, seed: int, stage: str, ordinal: int, formal: bool) -> tuple[Path, str]:
    endpoint_id = f"real-{model}-seed{seed}-{stage}-{ordinal}"
    endpoint_root = root / "endpoints"
    objects = endpoint_root / "objects"
    commits = endpoint_root / "commits"
    objects.mkdir(parents=True, exist_ok=True)
    commits.mkdir(parents=True, exist_ok=True)
    buffer_hash = _h(f"buffer:{endpoint_id}")
    pre = _state(f"{endpoint_id}-pre", parameter=_h(f"pre:{endpoint_id}"), buffer=buffer_hash, optimizer=_h(f"optimizer-pre:{endpoint_id}"))
    post = _state(f"{endpoint_id}-post", parameter=_h(f"post:{endpoint_id}"), buffer=buffer_hash, optimizer=_h(f"optimizer-post:{endpoint_id}"))
    attempt = _state(f"{endpoint_id}-commit", parameter=post["parameter_hash"], buffer=buffer_hash, optimizer=post["optimizer_hash"])
    metadata = {"model": model, "stage": stage, "seed": seed}
    record = {
        "path_state_id": endpoint_id,
        "source_run_id": f"real-run-{model}-seed{seed}",
        "optimizer_step": ordinal,
        "parameter_registry_hash": _h(f"registry:{model}"),
        "pre_state": pre,
        "parameter_post_state": post,
        "attempt_commit_state": attempt,
        "attempt_commit_parent_hash": post["artifact_hash"],
        "probe_buffer_snapshot_hash": buffer_hash,
        "full_update_delta_hash": _h(f"delta:{endpoint_id}"),
        "update_sample_ids": [f"update:{model}:{seed}:{stage}:{ordinal}"],
        "replay_verified": True,
        "metadata": metadata,
    }
    record["endpoint_digest"] = _stage3_hash(
        {
            "path_state_id": record["path_state_id"],
            "source_run_id": record["source_run_id"],
            "optimizer_step": record["optimizer_step"],
            "parameter_registry_hash": record["parameter_registry_hash"],
            "pre_state": _stage3_hash(pre),
            "parameter_post_state": _stage3_hash(post),
            "attempt_commit_state": _stage3_hash(attempt),
            "attempt_commit_parent_hash": record["attempt_commit_parent_hash"],
            "probe_buffer_snapshot_hash": record["probe_buffer_snapshot_hash"],
            "full_update_delta_hash": record["full_update_delta_hash"],
            "update_sample_ids": record["update_sample_ids"],
            "replay_verified": True,
        }
    )
    evidence = _h("formal-evidence") if formal else None
    object_body = {
        "schema_version": "endpoint-record-v1",
        "scope": "formal" if formal else "pilot",
        "formal_eligible": formal,
        "qualification_evidence_hash": evidence,
        "record": record,
        "state_bundles": {
            "pre": {"ref": f"state-bundles/{endpoint_id}-pre", "manifest_sha256": pre["artifact_hash"]},
            "parameter_post": {"ref": f"state-bundles/{endpoint_id}-post", "manifest_sha256": post["artifact_hash"]},
            "attempt_commit": {"ref": f"state-bundles/{endpoint_id}-commit", "manifest_sha256": attempt["artifact_hash"]},
        },
    }
    object_value = object_body | {"artifact_hash": canonical_json_hash(object_body)}
    object_path = objects / f"{endpoint_id}.json"
    write_canonical_json(object_path, object_value)
    commit_body = {
        "schema_version": "endpoint-commit-v1",
        "endpoint_id": endpoint_id,
        "optimizer_step": ordinal,
        "endpoint_digest": record["endpoint_digest"],
        "object_ref": f"objects/{endpoint_id}.json",
        "object_sha256": canonical_json_hash(object_value),
        "scope": "formal" if formal else "pilot",
        "formal_eligible": formal,
        "qualification_evidence_hash": evidence,
    }
    commit_value = commit_body | {"artifact_hash": canonical_json_hash(commit_body)}
    commit_path = commits / f"{endpoint_id}.json"
    write_canonical_json(commit_path, commit_value)
    return commit_path, str(record["endpoint_digest"])


def _write_probe(root: Path, *, endpoint_digest: str, index: int, role: str, formal: bool) -> Path:
    directory = root / "probes"
    directory.mkdir(parents=True, exist_ok=True)
    evidence = _h("formal-evidence") if formal else _h("pilot-evidence")
    entries = [
        {
            "role": role,
            "probe_id": f"real-probe-{index}-{probe}",
            "sample_ids": [f"probe-sample:{index}:{probe}"],
            "content_hash": _h(f"content:{index}:{probe}"),
            "loss_contract_hash": _h("loss-contract"),
            "effective_weight_unit": "sample",
            "metadata": {"source": "ready-data"},
        }
        for probe in range(3 if formal else 2)
    ]
    body = {
        "schema_version": "stage3-probe-plan-v1",
        "panel_id": f"real-panel-{index}",
        "endpoint_digest": endpoint_digest,
        "entries": entries,
        "minimum_formal_probes": 3 if formal else 1,
        "execution_evidence_hash": evidence,
        "scope": "formal" if formal else "pilot",
        "formal_eligible": formal,
    }
    path = directory / f"panel-{index}.json"
    write_canonical_json(path, body | {"artifact_hash": canonical_json_hash(body)})
    return path


def _build_sources(tmp_path: Path, *, formal: bool) -> tuple[Path, Path, int]:
    endpoint_root = tmp_path / ("formal" if formal else "pilot")
    endpoint_dir = endpoint_root / "endpoints"
    probe_dir = endpoint_root / "probes"
    specs: list[tuple[str, int, str, int]] = []
    if formal:
        for model, seeds, count in (("14M", (4301, 4302), 4), ("31M", (5301,), 3)):
            for seed in seeds:
                for stage in ("early", "middle", "late"):
                    for ordinal in range(1, count + 1):
                        specs.append((model, seed, stage, ordinal))
    else:
        for stage in ("early", "middle", "late"):
            for ordinal in range(1, 3):
                specs.append(("14M", 0, stage, ordinal))
    for index, (model, seed, stage, ordinal) in enumerate(specs):
        _commit, digest = _write_endpoint(tmp_path / ("formal" if formal else "pilot"), model=model, seed=seed, stage=stage, ordinal=ordinal, formal=formal)
        _write_probe(tmp_path / ("formal" if formal else "pilot"), endpoint_digest=digest, index=index, role="formal" if formal else "pilot", formal=formal)
    return endpoint_dir, probe_dir, len(specs)


def test_build_pilot_index_is_exactly_12_and_hash_bound(tmp_path: Path) -> None:
    endpoint_dir, probe_dir, endpoint_count = _build_sources(tmp_path, formal=False)
    index = build_production_unit_index(endpoint_dir, probe_dir, scope="pilot", workspace_root=tmp_path)
    assert endpoint_count == 6
    assert index.endpoint_count == 6
    assert index.unit_count == 12
    assert all(unit.model == "14M" and unit.scope == "pilot" for unit in index.units)
    assert all(unit.endpoint_hash and unit.probe_hash and unit.path_unit_id.startswith("path-unit-") for unit in index.units)
    output = tmp_path / "index.json"
    write_production_unit_index(output, index)
    assert json.loads(output.read_text(encoding="utf-8"))["artifact_hash"] == index.artifact_hash


def test_build_formal_index_is_exactly_99_and_has_two_models(tmp_path: Path) -> None:
    endpoint_dir, probe_dir, endpoint_count = _build_sources(tmp_path, formal=True)
    index = build_production_unit_index(endpoint_dir, probe_dir, scope="formal", workspace_root=tmp_path)
    assert endpoint_count == 33
    assert index.endpoint_count == 33
    assert index.unit_count == 99
    assert index.formal_eligible is True
    assert {unit.model for unit in index.units} == {"14M", "31M"}
    assert all(unit.scope == "formal" for unit in index.units)


def test_build_formal_index_rejects_placeholder_seed_matrix(tmp_path: Path) -> None:
    endpoint_dir = tmp_path / "bad-formal" / "endpoints"
    probe_dir = tmp_path / "bad-formal" / "probes"
    specs: list[tuple[str, int, str, int]] = []
    for model, seeds, count in (("14M", (0, 1), 4), ("31M", (0,), 3)):
        for seed in seeds:
            for stage in ("early", "middle", "late"):
                for ordinal in range(1, count + 1):
                    specs.append((model, seed, stage, ordinal))
    for index, (model, seed, stage, ordinal) in enumerate(specs):
        _commit, digest = _write_endpoint(
            tmp_path / "bad-formal",
            model=model,
            seed=seed,
            stage=stage,
            ordinal=ordinal,
            formal=True,
        )
        _write_probe(
            tmp_path / "bad-formal",
            endpoint_digest=digest,
            index=index,
            role="formal",
            formal=True,
        )
    with pytest.raises(ValueError, match="FORMAL_SEED_COVERAGE_INVALID"):
        build_production_unit_index(
            endpoint_dir,
            probe_dir,
            scope="formal",
            workspace_root=tmp_path,
        )


def test_formal_index_roundtrip_derives_strata_and_rejects_id_drift(tmp_path: Path) -> None:
    endpoint_dir, probe_dir, _ = _build_sources(tmp_path, formal=True)
    index = build_production_unit_index(
        endpoint_dir, probe_dir, scope="formal", workspace_root=tmp_path
    )
    output = tmp_path / "formal-unit-index.json"
    write_production_unit_index(output, index)
    loaded = load_production_unit_index(output, expected_scope="formal")
    assert loaded.to_dict() == index.to_dict()
    assert len(loaded.unit_strata()) == 99
    assert set(loaded.unit_strata()) == {
        unit.path_unit_id for unit in loaded.units
    }

    tampered = json.loads(output.read_text(encoding="utf-8"))
    tampered["units"][0]["path_unit_id"] = "path-unit-tampered"
    tampered["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in tampered.items() if key != "artifact_hash"}
    )
    write_canonical_json(output, tampered)
    with pytest.raises(ValueError, match="PATH_UNIT_ID_HASH_MISMATCH"):
        load_production_unit_index(output, expected_scope="formal")


def test_plan_identity_is_scope_separated_and_exactly_bound_to_index(tmp_path: Path) -> None:
    endpoint_dir, probe_dir, _ = _build_sources(tmp_path, formal=False)
    index = build_production_unit_index(
        endpoint_dir, probe_dir, scope="pilot", workspace_root=tmp_path
    )
    index_path = tmp_path / "pilot-unit-index.json"
    write_production_unit_index(index_path, index)
    assert load_production_unit_index(index_path, expected_scope="pilot").unit_count == 12
    with pytest.raises(ValueError, match="UNIT_INDEX_SCOPE_MISMATCH"):
        load_production_unit_index(index_path, expected_scope="formal")

    body: dict[str, object] = {
        "schema_version": "stage3-formal-pilot-plan-v1",
        "plan_id": "pilot-plan-bound",
        "scope": "formal",
        "candidate_rules": list(DEFAULT_CANDIDATE_RULES),
        "required_unit_ids": [unit.path_unit_id for unit in index.units],
        "unit_strata": index.unit_strata(),
        "plan_kind": "pilot",
        "production_unit_index_scope": "pilot",
        "production_unit_index_ref": "pilot-unit-index.json",
        "production_unit_index_hash": index.artifact_hash,
        "thresholds": {
            "max_normalized_l1_error": 0.01,
            "max_normalized_l2_error": 0.01,
            "max_normalized_linf_error": 0.01,
            "max_completeness_absolute_residual": 0.01,
            "max_completeness_relative_residual": 0.01,
            "max_completeness_l1_scaled_residual": 0.01,
            "min_spearman": 0.99,
            "min_active_spearman": 0.99,
            "min_cosine_similarity": 0.99,
            "min_sign_consistency": 0.99,
            "min_topk_overlap": 0.95,
            "min_topq_overlap": 0.95,
            "min_topq_jaccard": 0.95,
            "max_layer_quality_tv": 0.01,
            "max_module_quality_tv": 0.01,
            "max_reference_normalized_l1_error": 0.001,
            "completeness_stability_epsilon": 1.0e-12,
            "active_set_threshold": 0.0,
            "max_unique_nodes": 16,
            "top_q_values": [0.001, 0.01, 0.05],
            "required_strata": ["model", "stage", "update", "probe"],
            "require_worst_case": True,
        },
        "execution_evidence_hash": _h("formal-evidence"),
        "formal_eligible": True,
    }
    plan = body | {"artifact_hash": canonical_json_hash(body)}
    assert _validate_known_artifact(plan)[0] == "stage3_formal_pilot_plan"

    wrong_kind = dict(plan)
    wrong_kind["plan_kind"] = "matrix"
    wrong_kind["production_unit_index_scope"] = "formal"
    wrong_kind["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in wrong_kind.items() if key != "artifact_hash"}
    )
    with pytest.raises(ValueError, match="UNIT_COVERAGE_INVALID"):
        _validate_known_artifact(wrong_kind)

    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_h("contract-freeze"),
        asset_manifest_hashes=(_h("asset-manifest"),),
        prerequisite_gates=(
            GateRecord(
                "stage0.G10",
                0,
                GateStatus.PASS,
                "2026-08-28T00:00:00Z",
                evidence_refs=("evidence/stage0-g10.json",),
            ),
        ),
    )
    evidence_path = tmp_path / "formal-evidence.json"
    source_path = tmp_path / "pilot-plan-source.json"
    output_path = tmp_path / "pilot-plan.json"
    write_canonical_json(evidence_path, evidence.to_dict())
    source = {
        "plan_id": "pilot-plan-builder-bound",
        "candidate_rules": list(DEFAULT_CANDIDATE_RULES),
        "required_unit_ids": [unit.path_unit_id for unit in index.units],
        "unit_strata": index.unit_strata(),
        "plan_kind": "pilot",
        "production_unit_index_scope": "pilot",
        "production_unit_index_ref": index_path.name,
        "production_unit_index_hash": index.artifact_hash,
        "thresholds": body["thresholds"],
    }
    incomplete_source_path = tmp_path / "pilot-plan-source-incomplete.json"
    incomplete_output_path = tmp_path / "pilot-plan-incomplete.json"
    write_canonical_json(
        incomplete_source_path,
        {**source, "candidate_rules": ["midpoint", "trapezoid", "simpson"]},
    )
    assert main(
        [
            "artifact",
            "quadrature-pilot-plan-build",
            "--spec",
            str(incomplete_source_path),
            "--formal-execution-evidence",
            str(evidence_path),
            "--output",
            str(incomplete_output_path),
        ]
    ) != 0
    assert not incomplete_output_path.exists()
    write_canonical_json(source_path, source)
    assert main(
        [
            "artifact",
            "quadrature-pilot-plan-build",
            "--spec",
            str(source_path),
            "--formal-execution-evidence",
            str(evidence_path),
            "--output",
            str(output_path),
        ]
    ) == 0
    built = load_canonical_json(output_path)
    assert built["plan_kind"] == "pilot"
    assert built["production_unit_index_scope"] == "pilot"
    assert built["candidate_rules"] == list(DEFAULT_CANDIDATE_RULES)
    assert len(built["required_unit_ids"]) == 12

    formal_endpoint_dir, formal_probe_dir, _ = _build_sources(tmp_path, formal=True)
    formal_index = build_production_unit_index(
        formal_endpoint_dir,
        formal_probe_dir,
        scope="formal",
        workspace_root=tmp_path,
    )
    formal_index_path = tmp_path / "formal-unit-index.json"
    write_production_unit_index(formal_index_path, formal_index)
    formal_source_path = tmp_path / "matrix-plan-source.json"
    formal_output_path = tmp_path / "matrix-plan.json"
    formal_source = {
        "plan_id": "matrix-plan-builder-bound",
        "candidate_rules": list(DEFAULT_CANDIDATE_RULES),
        "required_unit_ids": [unit.path_unit_id for unit in formal_index.units],
        "unit_strata": formal_index.unit_strata(),
        "plan_kind": "matrix",
        "production_unit_index_scope": "formal",
        "production_unit_index_ref": formal_index_path.name,
        "production_unit_index_hash": formal_index.artifact_hash,
        "thresholds": body["thresholds"],
    }
    write_canonical_json(formal_source_path, formal_source)
    assert main(
        [
            "artifact",
            "quadrature-pilot-plan-build",
            "--spec",
            str(formal_source_path),
            "--formal-execution-evidence",
            str(evidence_path),
            "--output",
            str(formal_output_path),
        ]
    ) == 0
    built_formal = load_canonical_json(formal_output_path)
    assert built_formal["plan_kind"] == "matrix"
    assert built_formal["production_unit_index_scope"] == "formal"
    assert built_formal["candidate_rules"] == list(DEFAULT_CANDIDATE_RULES)
    assert len(built_formal["required_unit_ids"]) == 99


def test_rejects_fixture_missing_metadata_duplicate_and_incomplete_coverage(tmp_path: Path) -> None:
    endpoint_dir, probe_dir, _ = _build_sources(tmp_path, formal=False)
    first = next((endpoint_dir / "commits").glob("*.json"))
    payload = json.loads(first.read_text(encoding="utf-8"))
    payload["scope"] = "local_fixture"
    payload["artifact_hash"] = canonical_json_hash({key: value for key, value in payload.items() if key != "artifact_hash"})
    write_canonical_json(first, payload)
    with pytest.raises(ValueError, match="FIXTURE_OR_SYNTHETIC_REJECTED"):
        build_production_unit_index(endpoint_dir, probe_dir, scope="pilot", workspace_root=tmp_path)

    with pytest.raises(ValueError, match="FIXTURE_OR_SYNTHETIC_REJECTED"):
        build_production_unit_index(endpoint_dir / "commits" , probe_dir, scope="formal", workspace_root=tmp_path)
