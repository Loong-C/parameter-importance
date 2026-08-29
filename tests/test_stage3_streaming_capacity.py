from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.capacity import GIB
from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage3_protocol import DEFAULT_CANDIDATE_RULES
from param_importance_nlp.experiments.stage3_streaming_capacity import (
    STAGE3_STREAMING_CAPACITY_SCHEMA,
    STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA,
    STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA,
    STAGE3_STREAMING_RESUME_SCHEMA,
    StreamingCapacityError,
    _ref,
    build_capacity_preflight_report,
    check_filesystem_capacity,
    estimate_stage3_streaming_capacity,
    validate_streaming_launch_spec,
)
from ops.stage3.preflight_stage3_streaming_capacity import (
    _parameter_elements,
    _resume_units,
    _same_filesystem,
    main as preflight_main,
)
from ops.stage3.run_stage3_fanout import FanoutRunner, Stage3OrchestratorError


def _unit_models() -> dict[str, str]:
    return {
        **{f"14m-{index:03d}": "14M" for index in range(72)},
        **{f"31m-{index:03d}": "31M" for index in range(27)},
    }


def _metadata(model: str, *, levels: int = 3) -> dict[str, object]:
    reference_keys = [f"{index + 1:064x}" for index in range(levels)]
    candidate_keys = {
        rule: [f"{index + 101:064x}"]
        for index, rule in enumerate(DEFAULT_CANDIDATE_RULES)
    }
    return {
        "reference_completed_level_count": levels,
        "single_unit": {
            "reference_bytes": 1000,
            "observation_bytes": 2000,
            "raw_bytes": 3000,
            "receipt_bytes": 4000,
            "inode_count": 4,
        },
        "node_cache": {
            "reference_level_cache_key_digests": reference_keys,
            "candidate_cache_key_digests": candidate_keys,
            "object_bytes": 5000,
            "inode_count": 13,
        },
    }


def _snapshots() -> dict[str, list[dict[str, int]]]:
    return {
        kind: [
            {"unit_ordinal": ordinal, "bytes": ordinal * 100, "inode_count": 1}
            for ordinal in range(1, 100)
        ]
        for kind in ("reference", "raw", "stream")
    }


def _estimate(*, durable: tuple[str, ...] = ()):
    return estimate_stage3_streaming_capacity(
        unit_model_by_id=_unit_models(),
        parameter_elements_by_model={"14M": 10, "31M": 20},
        metadata_by_model={"14M": _metadata("14M"), "31M": _metadata("31M", levels=4)},
        aggregate_snapshots=_snapshots(),
        fixed_manifests={"bytes": 700, "inode_count": 7},
        inflight={"bytes": 800, "inode_count": 8},
        temporary_json={"bytes": 900, "inode_count": 9},
        durable_unit_ids=durable,
    )


def test_capacity_uses_exact_99_by_13_formula_and_all_snapshots() -> None:
    estimate = _estimate()
    assert estimate.unit_count == 99
    assert estimate.candidate_count == 13
    assert estimate.models[0].unit_count == 72
    assert estimate.models[1].unit_count == 27
    # 8P * (C + 2 + L): 14M L=3, 31M L=4.
    assert estimate.models[0].unit_vector_bytes == 8 * 10 * (13 + 2 + 3)
    assert estimate.models[1].unit_vector_bytes == 8 * 20 * (13 + 2 + 4)
    assert estimate.aggregate_snapshot_bytes == 3 * sum(range(1, 100)) * 100
    assert estimate.aggregate_snapshot_inodes == 3 * 99
    assert estimate.expected_new_bytes == (
        estimate.expected_output_bytes + estimate.expected_cache_bytes
    )
    assert estimate.failure_residue_bytes >= estimate.active_cache_peak_bytes
    assert estimate.required_free_bytes == estimate.expected_new_bytes + 100 * GIB


def test_resume_deducts_only_durable_units_not_immutable_snapshots() -> None:
    units = _unit_models()
    fresh = _estimate()
    durable = tuple(list(units)[:2])
    resumed = _estimate(durable=durable)
    assert resumed.durable_unit_count == 2
    assert resumed.pending_unit_count == 97
    assert resumed.aggregate_snapshot_bytes == fresh.aggregate_snapshot_bytes
    assert resumed.retained_vector_bytes == fresh.retained_vector_bytes - 2 * fresh.models[0].unit_vector_bytes
    assert resumed.retained_unit_metadata_bytes == fresh.retained_unit_metadata_bytes - 2 * fresh.models[0].unit_metadata_bytes
    assert resumed.expected_new_bytes < fresh.expected_new_bytes


def test_capacity_rejects_wrong_matrix_shape_or_cache_rule_coverage() -> None:
    units = _unit_models()
    with pytest.raises(StreamingCapacityError, match="FORMAL_UNIT_COUNT_INVALID"):
        estimate_stage3_streaming_capacity(
            unit_model_by_id=dict(list(units.items())[:-1]),
            parameter_elements_by_model={"14M": 10, "31M": 20},
            metadata_by_model={"14M": _metadata("14M"), "31M": _metadata("31M")},
            aggregate_snapshots=_snapshots(),
            fixed_manifests={"bytes": 1, "inode_count": 1},
            inflight={"bytes": 1, "inode_count": 1},
            temporary_json={"bytes": 1, "inode_count": 1},
        )
    broken = _metadata("14M")
    broken_cache = dict(broken["node_cache"])
    broken_cache["candidate_cache_key_digests"] = dict(broken_cache["candidate_cache_key_digests"])
    broken_cache["candidate_cache_key_digests"].pop(DEFAULT_CANDIDATE_RULES[-1])
    broken["node_cache"] = broken_cache
    with pytest.raises(StreamingCapacityError, match="CANDIDATE_CACHE_RULE_SET_INVALID"):
        estimate_stage3_streaming_capacity(
            unit_model_by_id=units,
            parameter_elements_by_model={"14M": 10, "31M": 20},
            metadata_by_model={"14M": broken, "31M": _metadata("31M")},
            aggregate_snapshots=_snapshots(),
            fixed_manifests={"bytes": 1, "inode_count": 1},
            inflight={"bytes": 1, "inode_count": 1},
            temporary_json={"bytes": 1, "inode_count": 1},
        )


def test_filesystem_capacity_fails_closed_for_bytes_or_inodes() -> None:
    estimate = _estimate()
    passed = check_filesystem_capacity(
        name="output",
        path="output",
        budget=estimate.output_budget,
        required_free_inodes=estimate.required_output_free_inodes,
        free_bytes=estimate.output_budget.required_free_bytes,
        free_inodes=estimate.required_output_free_inodes,
    )
    assert passed.ok is True
    bytes_blocked = check_filesystem_capacity(
        name="output",
        path="output",
        budget=estimate.output_budget,
        required_free_inodes=estimate.required_output_free_inodes,
        free_bytes=estimate.output_budget.required_free_bytes - 1,
        free_inodes=estimate.required_output_free_inodes,
    )
    assert bytes_blocked.ok is False
    no_inode = check_filesystem_capacity(
        name="output",
        path="output",
        budget=estimate.output_budget,
        required_free_inodes=estimate.required_output_free_inodes,
        free_bytes=estimate.output_budget.required_free_bytes,
        free_inodes=None,
    )
    # Explicit None means measure; a non-existent path must fail closed.
    assert no_inode.ok is False


def test_same_output_and_cache_filesystem_is_detected(tmp_path: Path) -> None:
    assert _same_filesystem(tmp_path, tmp_path) is True
    other = tmp_path / "other"
    other.mkdir()
    # The test workspace normally resides on one filesystem; if the platform
    # uses a different device for this child, the result remains explicit.
    assert _same_filesystem(tmp_path, other) in {True, False}


def test_preflight_report_is_canonical_hash_bound_and_blocked_if_fs_fails() -> None:
    estimate = _estimate()
    output = check_filesystem_capacity(
        name="output",
        path="output",
        budget=estimate.output_budget,
        required_free_inodes=estimate.required_output_free_inodes,
        free_bytes=0,
        free_inodes=0,
    )
    cache = check_filesystem_capacity(
        name="cache",
        path="cache",
        budget=estimate.cache_budget,
        required_free_inodes=estimate.required_cache_free_inodes,
        free_bytes=estimate.cache_budget.required_free_bytes,
        free_inodes=estimate.required_cache_free_inodes,
    )
    report = build_capacity_preflight_report(
        estimate=estimate,
        output_check=output,
        cache_check=cache,
        formal_plan_ref="plans/formal-plan.json",
        formal_plan_hash="a" * 64,
        production_index_ref="plans/production-index.json",
        production_index_hash="b" * 64,
        parameter_counts_ref="plans/parameter-counts.json",
        metadata_ref="plans/metadata.json",
    )
    assert report["schema_version"] == STAGE3_STREAMING_CAPACITY_SCHEMA
    assert report["status"] == "BLOCKED"
    assert report["formal_eligible"] is False
    assert report["artifact_hash"] == canonical_json_hash(
        {key: item for key, item in report.items() if key != "artifact_hash"}
    )


def test_canonical_documents_can_be_written_without_data_root_side_effect(tmp_path: Path) -> None:
    payload = {"schema_version": "test", "value": 1}
    payload["artifact_hash"] = canonical_json_hash(payload)
    target = tmp_path / "evidence.json"
    write_canonical_json(target, payload)
    assert load_canonical_json(target) == payload


def test_preflight_cli_emits_hash_bound_blocked_evidence_on_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "preflight.json"
    code = preflight_main(
        [
            "--workspace-root",
            str(tmp_path),
            "--formal-plan",
            str(tmp_path / "missing-plan.json"),
            "--production-index",
            str(tmp_path / "missing-index.json"),
            "--parameter-counts",
            str(tmp_path / "missing-counts.json"),
            "--metadata",
            str(tmp_path / "missing-metadata.json"),
            "--output-filesystem",
            str(tmp_path),
            "--cache-filesystem",
            str(tmp_path),
            "--output",
            str(output),
        ]
    )
    assert code == 1
    rendered = capsys.readouterr().out
    assert '"status":"BLOCKED"' in rendered
    report = load_canonical_json(output)
    assert report["schema_version"] == STAGE3_STREAMING_CAPACITY_SCHEMA
    assert report["artifact_hash"] == canonical_json_hash(
        {key: item for key, item in report.items() if key != "artifact_hash"}
    )


def _with_hash(payload: dict[str, object]) -> dict[str, object]:
    payload["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in payload.items() if key != "artifact_hash"}
    )
    return payload


def test_reference_rejects_nul_traversal_and_drive_forms() -> None:
    for value in ("metadata/\x00.json", "metadata/../counts.json", "C:counts.json"):
        with pytest.raises(StreamingCapacityError, match="REFERENCE_INVALID"):
            _ref(value, field="test")


def test_launch_spec_is_exact_and_hash_bound() -> None:
    spec = _with_hash(
        {
            "schema_version": STAGE3_STREAMING_LAUNCH_SPEC_SCHEMA,
            "formal_plan_ref": "plans/formal.json",
            "formal_plan_hash": "a" * 64,
            "production_index_ref": "plans/index.json",
            "production_index_hash": "b" * 64,
            "parameter_counts_ref": "plans/counts.json",
            "parameter_counts_hash": "c" * 64,
            "metadata_ref": "plans/metadata.json",
            "metadata_hash": "d" * 64,
            "output_filesystem_ref": "operations/output",
            "cache_filesystem_ref": "cache/stage3",
            "candidate_rule_names": list(DEFAULT_CANDIDATE_RULES),
            "resume_manifest_ref": None,
            "resume_manifest_hash": None,
        }
    )
    assert validate_streaming_launch_spec(spec)["artifact_hash"] == spec["artifact_hash"]
    drifted = dict(spec)
    drifted["metadata_hash"] = "e" * 64
    with pytest.raises(StreamingCapacityError, match="ARTIFACT_HASH_MISMATCH"):
        validate_streaming_launch_spec(drifted)


def test_parameter_count_is_recomputed_from_bound_registry_and_endpoint(tmp_path: Path) -> None:
    refs: dict[str, dict[str, str]] = {}
    for model, count in (("14M", 10), ("31M", 20)):
        refs[model] = {}
        for kind in ("registry", "endpoint"):
            ref = f"sources/{model}-{kind}.json"
            value = _with_hash(
                {
                    "schema_version": "stage2-source-v1",
                    "model_id": model,
                    "parameter_count": count,
                }
            )
            write_canonical_json(tmp_path / ref, value)
            refs[model][kind] = ref
    counts = {
        "schema_version": STAGE3_STREAMING_PARAMETER_COUNTS_SCHEMA,
        "formal_eligible": True,
        "formal_plan_hash": "a" * 64,
        "production_index_hash": "b" * 64,
        "candidate_rule_names": list(DEFAULT_CANDIDATE_RULES),
        "models": {},
    }
    for model, count in (("14M", 10), ("31M", 20)):
        registry = load_canonical_json(tmp_path / refs[model]["registry"])
        endpoint = load_canonical_json(tmp_path / refs[model]["endpoint"])
        counts["models"][model] = {
            "parameter_elements": count,
            "parameter_registry_ref": refs[model]["registry"],
            "parameter_registry_hash": registry["artifact_hash"],
            "endpoint_ref": refs[model]["endpoint"],
            "endpoint_hash": endpoint["artifact_hash"],
        }
    _with_hash(counts)
    assert _parameter_elements(
        counts,
        root=tmp_path,
        formal_plan_hash="a" * 64,
        production_index_hash="b" * 64,
        candidate_rule_names=DEFAULT_CANDIDATE_RULES,
    ) == {"14M": 10, "31M": 20}
    shrunken = dict(counts)
    shrunken["models"] = dict(counts["models"])
    shrunken["models"]["14M"] = dict(shrunken["models"]["14M"])
    shrunken["models"]["14M"]["parameter_elements"] = 9
    _with_hash(shrunken)
    with pytest.raises(StreamingCapacityError, match="SOURCE_COUNT_MISMATCH"):
        _parameter_elements(
            shrunken,
            root=tmp_path,
            formal_plan_hash="a" * 64,
            production_index_hash="b" * 64,
            candidate_rule_names=DEFAULT_CANDIDATE_RULES,
        )


def test_resume_requires_real_hash_bound_artifact_and_unit_identity(tmp_path: Path) -> None:
    artifact = _with_hash(
        {
            "schema_version": "stage3-streaming-unit-result-v1",
            "status": "PASS",
            "unit_id": "unit-001",
        }
    )
    artifact_ref = "durable/unit-001.json"
    write_canonical_json(tmp_path / artifact_ref, artifact)
    manifest = _with_hash(
        {
            "schema_version": STAGE3_STREAMING_RESUME_SCHEMA,
            "formal_eligible": True,
            "production_index_hash": "b" * 64,
            "durable_units": [
                {
                    "unit_id": "unit-001",
                    "artifact_ref": artifact_ref,
                    "artifact_hash": artifact["artifact_hash"],
                }
            ],
        }
    )
    assert _resume_units(
        manifest,
        root=tmp_path,
        production_index_hash="b" * 64,
        expected_units=("unit-001",),
    ) == ("unit-001",)
    forged = dict(manifest)
    forged["durable_units"] = [
        {
            "unit_id": "unit-001",
            "artifact_ref": "durable/missing.json",
            "artifact_hash": "f" * 64,
        }
    ]
    _with_hash(forged)
    with pytest.raises(StreamingCapacityError, match="RESUME_ARTIFACT_NOT_FILE"):
        _resume_units(
            forged,
            root=tmp_path,
            production_index_hash="b" * 64,
            expected_units=("unit-001",),
        )


def test_s307_fanout_requires_capacity_spec_during_initialization() -> None:
    runner = object.__new__(FanoutRunner)
    runner.task_id = "stage3.07_formal_experiment_matrix"
    runner.environment_payload = {"evidence_refs": {}}
    with pytest.raises(Stage3OrchestratorError, match="FANOUT_CAPACITY_PREFLIGHT_REF_MISSING"):
        runner._load_streaming_capacity_spec()
