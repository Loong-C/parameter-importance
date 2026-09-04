"""Regression tests for append-only Stage 3 execution-evidence lineage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import param_importance_nlp.experiments.stage23_task_runners as task_runners
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
)
from param_importance_nlp.contracts.jsonio import canonical_json_hash
from param_importance_nlp.contracts.jsonio import load_canonical_json, write_canonical_json
from param_importance_nlp.experiments.stage23_task_runners import (
    _load_stage3_streaming_unit_receipt,
    _persist_stage3_streaming_eviction_receipt,
    _persist_stage3_streaming_unit_receipt,
    _publish_stage3_streaming_aggregate,
    _resume_stage3_formal_matrix_from_receipt,
    _stage3_execution_is_append_only_extension,
    _stage3_execution_matches_or_extends_plan,
    _stage3_streaming_receipt_root,
)
from param_importance_nlp.experiments.stage3_protocol import DEFAULT_CANDIDATE_RULES
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.task_runtime import TaskBlockedError


def _hash(label: str) -> str:
    return canonical_json_hash({"label": label})


def _gate(index: int, *, label: str | None = None) -> GateRecord:
    return GateRecord(
        gate_id=f"stage3.G3-{index}",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:00:00Z",
        measured={"identity": label or f"g{index}"},
        threshold={"required": True},
        evidence_refs=(f"evidence/stage3/g3-{index}.json",),
    )


def _execution(*gates: GateRecord, metadata: str = "same") -> FormalExecutionEvidence:
    return FormalExecutionEvidence(
        "formal",
        contract_freeze_hash=_hash("contract"),
        asset_manifest_hashes=(_hash("model"), _hash("data")),
        prerequisite_gates=gates,
        metadata={"run_family": metadata},
    )


def test_plan_execution_allows_only_content_bound_gate_extension(tmp_path) -> None:
    base = _execution(*(_gate(index) for index in range(4)))
    current = _execution(*(_gate(index) for index in range(5)))
    assert _stage3_execution_is_append_only_extension(base, current)

    execution_commit = TaskArtifactStore(tmp_path, "authority/base-execution").publish(
        task_id="stage3.formal_execution_authority",
        artifact_kind="formal_execution_evidence",
        config_hash=_hash("base-config"),
        run_intent="formal",
        payload=base.to_dict(),
        formal_eligible=True,
    )
    plan_commit = TaskArtifactStore(tmp_path, "authority/pilot-plan").publish(
        task_id="stage3.formal_plan_authority",
        artifact_kind="stage3_formal_plan",
        config_hash=_hash("plan-config"),
        run_intent="formal",
        payload={"schema_version": "test-stage3-plan-v1"},
        formal_eligible=True,
        source_refs=(execution_commit.commit_ref,),
    )
    request = SimpleNamespace(
        environment=SimpleNamespace(evidence_refs={})
    )
    assert _stage3_execution_matches_or_extends_plan(
        request,  # type: ignore[arg-type]
        tmp_path,
        current=current,
        declared_hash=base.artifact_hash,
        plan_ref=plan_commit.commit_ref,
        plan_kind="pilot",
    )

    changed_old_gate = _execution(
        _gate(0),
        _gate(1, label="changed"),
        _gate(2),
        _gate(3),
        _gate(4),
    )
    assert not _stage3_execution_is_append_only_extension(base, changed_old_gate)
    changed_metadata = _execution(
        *(_gate(index) for index in range(5)), metadata="drifted"
    )
    assert not _stage3_execution_is_append_only_extension(base, changed_metadata)


def test_plan_execution_rejects_bare_hash_without_original_commit(tmp_path) -> None:
    base = _execution(*(_gate(index) for index in range(4)))
    current = _execution(*(_gate(index) for index in range(5)))
    plan_commit = TaskArtifactStore(tmp_path, "authority/unbound-plan").publish(
        task_id="stage3.formal_plan_authority",
        artifact_kind="stage3_formal_plan",
        config_hash=_hash("unbound-plan-config"),
        run_intent="formal",
        payload={"schema_version": "test-stage3-plan-v1"},
        formal_eligible=True,
    )
    request = SimpleNamespace(
        environment=SimpleNamespace(evidence_refs={})
    )
    assert not _stage3_execution_matches_or_extends_plan(
        request,  # type: ignore[arg-type]
        tmp_path,
        current=current,
        declared_hash=base.artifact_hash,
        plan_ref=plan_commit.commit_ref,
        plan_kind="pilot",
    )


def test_streaming_receipt_root_is_workspace_relative_and_hash_bound(tmp_path) -> None:
    execution_hash = _hash("execution")
    binding_hash = _hash("binding")
    request = SimpleNamespace(
        config=SimpleNamespace(
            base_config=SimpleNamespace(section=lambda _name: {})
        )
    )
    store = TaskArtifactStore(tmp_path, "task-output")
    receipt_root = _stage3_streaming_receipt_root(
        request,  # type: ignore[arg-type]
        tmp_path,
        store,
        execution_evidence_hash=execution_hash,
        reference_binding_hash=binding_hash,
    )
    assert receipt_root.is_relative_to(tmp_path)
    assert receipt_root.parts[-3:] == (
        "stage3-streaming-receipts",
        execution_hash,
        binding_hash,
    )


def test_streaming_receipt_root_rejects_escaping_runtime_cache(tmp_path) -> None:
    request = SimpleNamespace(
        config=SimpleNamespace(
            base_config=SimpleNamespace(
                section=lambda _name: {"cache_root": "../outside"}
            )
        )
    )
    store = TaskArtifactStore(tmp_path, "task-output")
    try:
        _stage3_streaming_receipt_root(
            request,  # type: ignore[arg-type]
            tmp_path,
            store,
            execution_evidence_hash=_hash("execution"),
            reference_binding_hash=_hash("binding"),
        )
    except ValueError as error:
        assert "PATH_ESCAPE" in str(error)
    else:  # pragma: no cover - assertion gives a clearer failure than pytest.raises
        raise AssertionError("escaping runtime.cache_root must be rejected")


def test_streaming_unit_receipt_rejects_hash_and_identity_drift(tmp_path) -> None:
    root = tmp_path
    receipt_root = root / "resume" / "stage3-streaming-receipts" / "execution" / "binding"
    required_units = ("unit-0", "unit-1")
    execution_hash = _hash("execution")
    plan_hash = _hash("plan")
    index_hash = _hash("index")
    binding_hash = _hash("binding")
    path_hash = _hash("path")
    common = {
        "execution_evidence_hash": execution_hash,
        "formal_plan_hash": plan_hash,
        "production_unit_index_hash": index_hash,
        "reference_binding_hash": binding_hash,
        "path_identity_hash": path_hash,
        "reference_artifact_hash": _hash("reference"),
    }

    def artifact(path, schema):
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema_version": schema}
        body["artifact_hash"] = canonical_json_hash(body)
        write_canonical_json(path, body)
        return body["artifact_hash"]

    refs = {
        "reference_shard_ref": "refs/reference.json",
        "reference_aggregate_ref": "refs/reference-aggregate.json",
        "observation_ledger_ref": "refs/observation.json",
        "raw_shard_ref": "refs/raw.json",
        "raw_aggregate_ref": "refs/raw-aggregate.json",
    }
    hashes = {
        "reference_shard_hash": artifact(root / refs["reference_shard_ref"], "stage3-reference-shard-v1"),
        "reference_aggregate_hash": artifact(root / refs["reference_aggregate_ref"], "stage3-reference-aggregate-v1"),
        "observation_artifact_hash": artifact(root / refs["observation_ledger_ref"], "stage3-quadrature-observation-v1"),
        "raw_shard_hash": artifact(root / refs["raw_shard_ref"], "stage3-formal-raw-shard-v1"),
        "raw_aggregate_hash": artifact(root / refs["raw_aggregate_ref"], "stage3-formal-raw-aggregate-v1"),
    }
    receipt, receipt_ref = _persist_stage3_streaming_unit_receipt(
        root=root,
        receipt_root=receipt_root,
        unit_id="unit-0",
        required_unit_ids=required_units,
        candidate_names=DEFAULT_CANDIDATE_RULES,
        formal_plan_ref="plans/matrix.json",
        production_unit_index_ref="plans/index.json",
        node_cache_seal_ref="receipts/cache.SEALED.json",
        node_cache_evidence_hash=_hash("evidence"),
        node_cache_seal_hash=_hash("seal"),
        raw_bundle_ref="refs/raw-bundle",
        raw_bundle_manifest_hash=_hash("bundle"),
        **common,
        **refs,
        **hashes,
    )
    loaded = _load_stage3_streaming_unit_receipt(
        root=root,
        receipt_root=receipt_root,
        unit_id="unit-0",
        required_unit_ids=required_units,
        candidate_names=DEFAULT_CANDIDATE_RULES,
        formal_plan_ref="plans/matrix.json",
        formal_plan_hash=plan_hash,
        production_unit_index_ref="plans/index.json",
        production_unit_index_hash=index_hash,
        execution_evidence_hash=execution_hash,
        reference_binding_hash=binding_hash,
        path_identity_hash=path_hash,
    )
    assert loaded is not None and loaded[0]["artifact_hash"] == receipt["artifact_hash"]

    drifted = dict(receipt)
    drifted["reference_aggregate_hash"] = _hash("different-aggregate")
    drifted["artifact_hash"] = canonical_json_hash(
        {key: value for key, value in drifted.items() if key != "artifact_hash"}
    )
    receipt_path = root / receipt_ref
    write_canonical_json(receipt_path, drifted)
    try:
        _load_stage3_streaming_unit_receipt(
            root=root,
            receipt_root=receipt_root,
            unit_id="unit-0",
            required_unit_ids=required_units,
            candidate_names=DEFAULT_CANDIDATE_RULES,
            formal_plan_ref="plans/matrix.json",
            formal_plan_hash=plan_hash,
            production_unit_index_ref="plans/index.json",
            production_unit_index_hash=index_hash,
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
            path_identity_hash=path_hash,
        )
    except ValueError as error:
        assert "HASH_MISMATCH" in str(error)
    else:  # pragma: no cover
        raise AssertionError("receipt hash drift must fail closed")


def test_streaming_aggregate_rejects_non_frozen_candidate_set(tmp_path) -> None:
    try:
        _publish_stage3_streaming_aggregate(
            root=tmp_path,
            receipt_root=tmp_path / "receipts",
            required_unit_ids=("unit-0",),
            candidate_names=("left",),
            execution_evidence_hash=_hash("execution"),
            reference_binding_hash=_hash("binding"),
            formal_plan_ref="plan.json",
            formal_plan_hash=_hash("plan"),
            production_unit_index_ref="index.json",
            production_unit_index_hash=_hash("index"),
            reference_aggregate_ref="reference.json",
            reference_aggregate={},
            raw_aggregate_ref="raw.json",
            raw_aggregate={},
            observation_complete_unit_ids=(),
        )
    except ValueError as error:
        assert "CANDIDATE_RULE_SET_MISMATCH" in str(error)
    else:  # pragma: no cover
        raise AssertionError("non-frozen candidate rule sets must fail closed")


def test_streaming_aggregate_blocks_first_98_and_passes_99(tmp_path, monkeypatch) -> None:
    """Every durable unit fence blocks until the exact 99-way intersection."""

    root = tmp_path
    receipt_root = root / "resume" / "streaming"
    required_units = tuple(f"unit-{index:02d}" for index in range(99))
    execution_hash = _hash("execution")
    binding_hash = _hash("binding")
    plan_hash = _hash("plan")
    index_hash = _hash("index")
    candidate_names = DEFAULT_CANDIDATE_RULES
    source_hashes = {}
    for unit_id in required_units:
        source_hashes[unit_id] = {
            "path": _hash(f"path:{unit_id}"),
            "reference": _hash(f"reference:{unit_id}"),
            "raw": _hash(f"raw:{unit_id}"),
            "bundle": _hash(f"bundle:{unit_id}"),
            "evidence": _hash(f"evidence:{unit_id}"),
            "seal": _hash(f"seal:{unit_id}"),
            "eviction": _hash(f"eviction:{unit_id}"),
        }

    def write_source(unit_id: str, kind: str, schema: str) -> str:
        path = root / "refs" / f"{unit_id}-{kind}.json"
        body = {"schema_version": schema}
        body["artifact_hash"] = canonical_json_hash(body)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_canonical_json(path, body)
        return path.relative_to(root).as_posix()

    source_refs = {}
    for unit_id in required_units:
        source_refs[unit_id] = {
            "reference": write_source(unit_id, "reference", "stage3-reference-shard-v1"),
            "observation": write_source(unit_id, "observation", "stage3-quadrature-observation-v1"),
            "raw": write_source(unit_id, "raw", "stage3-formal-raw-shard-v1"),
        }
        source_hashes[unit_id]["reference"] = load_canonical_json(
            root / source_refs[unit_id]["reference"]
        )["artifact_hash"]
        source_hashes[unit_id]["raw"] = load_canonical_json(
            root / source_refs[unit_id]["raw"]
        )["artifact_hash"]

    seal_hashes = {unit_id: source_hashes[unit_id]["seal"] for unit_id in required_units}
    eviction_hashes = {unit_id: source_hashes[unit_id]["eviction"] for unit_id in required_units}

    def fake_verify(receipt_root_or_ref, receipt_ref=None):
        name = Path(receipt_ref if receipt_ref is not None else receipt_root_or_ref).name
        if name.endswith(".SEALED.json"):
            unit_id = name.removesuffix(".SEALED.json")
            return {
                "state": "SEALED",
                "receipt_hash": seal_hashes[unit_id],
                "unit_id": unit_id,
                "downstream_raw_shard_hash": source_hashes[unit_id]["raw"],
            }
        if name.endswith(".EVICTED.json"):
            unit_id = name.removesuffix(".EVICTED.json")
            return {"state": "EVICTED", "tombstone_hash": eviction_hashes[unit_id]}
        raise AssertionError(name)

    monkeypatch.setattr(
        task_runners.PersistentNodeGradientCache,
        "verify_receipt",
        staticmethod(fake_verify),
    )

    for index in range(99):
        complete = required_units[: index + 1]
        reference_entries = {
            unit_id: {
                "reference_artifact_hash": source_hashes[unit_id]["reference"],
                "path_identity_hash": source_hashes[unit_id]["path"],
            }
            for unit_id in complete
        }
        reference_aggregate = {
            "schema_version": "stage3-reference-aggregate-v1",
            "reference_scope": "matrix",
            "execution_evidence_hash": execution_hash,
            "reference_binding_hash": binding_hash,
            "required_unit_ids": list(required_units),
            "complete_unit_ids": list(complete),
            "missing_unit_ids": list(required_units[index + 1 :]),
            "unit_references": reference_entries,
        }
        reference_aggregate["artifact_hash"] = canonical_json_hash(reference_aggregate)
        reference_aggregate_path = root / "refs" / f"reference-aggregate-{index:02d}.json"
        write_canonical_json(reference_aggregate_path, reference_aggregate)
        reference_aggregate_ref = reference_aggregate_path.relative_to(root).as_posix()

        raw_entries = {
            unit_id: {
                "shard_ref": source_refs[unit_id]["raw"],
                "shard_hash": source_hashes[unit_id]["raw"],
                "bundle_ref": f"refs/{unit_id}-bundle",
                "bundle_manifest_hash": source_hashes[unit_id]["bundle"],
                "path_identity_hash": source_hashes[unit_id]["path"],
            }
            for unit_id in complete
        }
        raw_aggregate = {
            "schema_version": "stage3-formal-raw-aggregate-v1",
            "execution_evidence_hash": execution_hash,
            "reference_binding_hash": binding_hash,
            "required_unit_ids": list(required_units),
            "candidate_rule_names": sorted(candidate_names),
            "complete_unit_ids": list(complete),
            "missing_unit_ids": list(required_units[index + 1 :]),
            "unit_shards": raw_entries,
        }
        raw_aggregate["artifact_hash"] = canonical_json_hash(raw_aggregate)
        raw_aggregate_path = root / "refs" / f"raw-aggregate-{index:02d}.json"
        write_canonical_json(raw_aggregate_path, raw_aggregate)
        raw_aggregate_ref = raw_aggregate_path.relative_to(root).as_posix()

        unit_id = required_units[index]
        hashes = source_hashes[unit_id]
        reference_hash = hashes["reference"]
        reference_shard_hash = load_canonical_json(
            root / source_refs[unit_id]["reference"]
        )["artifact_hash"]
        observation_hash = load_canonical_json(
            root / source_refs[unit_id]["observation"]
        )["artifact_hash"]
        raw_shard_hash = load_canonical_json(
            root / source_refs[unit_id]["raw"]
        )["artifact_hash"]
        _persist_stage3_streaming_unit_receipt(
            root=root,
            receipt_root=receipt_root,
            unit_id=unit_id,
            required_unit_ids=required_units,
            candidate_names=candidate_names,
            execution_evidence_hash=execution_hash,
            formal_plan_ref="plans/matrix.json",
            formal_plan_hash=plan_hash,
            production_unit_index_ref="plans/index.json",
            production_unit_index_hash=index_hash,
            reference_binding_hash=binding_hash,
            path_identity_hash=hashes["path"],
            reference_artifact_hash=reference_hash,
            reference_shard_ref=source_refs[unit_id]["reference"],
            reference_shard_hash=reference_shard_hash,
            reference_aggregate_ref=reference_aggregate_ref,
            reference_aggregate_hash=reference_aggregate["artifact_hash"],
            observation_ledger_ref=source_refs[unit_id]["observation"],
            observation_artifact_hash=observation_hash,
            raw_shard_ref=source_refs[unit_id]["raw"],
            raw_shard_hash=raw_shard_hash,
            raw_bundle_ref=f"refs/{unit_id}-bundle",
            raw_bundle_manifest_hash=hashes["bundle"],
            raw_aggregate_ref=raw_aggregate_ref,
            raw_aggregate_hash=raw_aggregate["artifact_hash"],
            node_cache_evidence_hash=hashes["evidence"],
            node_cache_seal_ref=f"receipts/{unit_id}.SEALED.json",
            node_cache_seal_hash=hashes["seal"],
        )
        _persist_stage3_streaming_eviction_receipt(
            root=root,
            receipt_root=receipt_root,
            unit_id=unit_id,
            unit_receipt_ref=(receipt_root / "units" / f"{unit_id}.json").relative_to(root).as_posix(),
            unit_receipt_hash=load_canonical_json(
                receipt_root / "units" / f"{unit_id}.json"
            )["artifact_hash"],
            seal_ref=f"receipts/{unit_id}.SEALED.json",
            seal_hash=hashes["seal"],
            cache_root_ref=f"cache/{unit_id}",
            eviction_ref=f"receipts/{unit_id}.EVICTED.json",
            eviction_hash=hashes["eviction"],
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
        )
        if index not in (97, 98):
            continue
        aggregate, _aggregate_ref = _publish_stage3_streaming_aggregate(
            root=root,
            receipt_root=receipt_root,
            required_unit_ids=required_units,
            candidate_names=candidate_names,
            execution_evidence_hash=execution_hash,
            reference_binding_hash=binding_hash,
            formal_plan_ref="plans/matrix.json",
            formal_plan_hash=plan_hash,
            production_unit_index_ref="plans/index.json",
            production_unit_index_hash=index_hash,
            reference_aggregate_ref=reference_aggregate_ref,
            reference_aggregate=reference_aggregate,
            raw_aggregate_ref=raw_aggregate_ref,
            raw_aggregate=raw_aggregate,
            observation_complete_unit_ids=complete,
        )
        assert aggregate["committed_unit_ids"] == list(complete)
        assert aggregate["missing_unit_ids"] == list(required_units[index + 1 :])


def test_matrix_unit_reference_nonconvergence_runs_zero_candidates(tmp_path, monkeypatch) -> None:
    execution_hash = _hash("execution")
    binding_hash = _hash("binding")
    plan_hash = _hash("plan")
    threshold_hash = _hash("thresholds")
    plan_ref = "plans/matrix.json"
    thresholds = SimpleNamespace(artifact_hash=threshold_hash)
    production_index = SimpleNamespace(
        artifact_hash=_hash("production-index"),
        unit=lambda unit_id: SimpleNamespace(unit_id=unit_id),
    )
    gate = GateRecord(
        gate_id="stage3.G3-5",
        stage=3,
        status=GateStatus.PASS,
        checked_at="2026-08-28T00:00:00Z",
        measured={},
        threshold={},
        evidence_refs=("evidence/stage3/g35/pilot-freeze.json",),
    )
    context = SimpleNamespace(
        unit_id="unit-00",
        path=SimpleNamespace(identity_hash=_hash("path")),
        execution=SimpleNamespace(
            run_intent="formal",
            artifact_hash=execution_hash,
            prerequisite_gates=(gate,),
        ),
    )
    request = SimpleNamespace(
        config=SimpleNamespace(
            run_intent="formal",
            config_hash=_hash("config"),
            base_config=SimpleNamespace(section=lambda _name: {}),
        )
    )
    inputs = SimpleNamespace(
        references=("inputs/predecessor.json",),
        binding_hash=_hash("inputs"),
        payload=lambda name: {"thresholds_hash": threshold_hash}
        if name == "threshold_freeze"
        else {},
    )
    candidate_calls = []
    monkeypatch.setattr(task_runners, "_predecessor_context", lambda *_args: inputs)
    monkeypatch.setattr(task_runners, "_fixture_path_context", lambda *_args: context)
    monkeypatch.setattr(
        task_runners,
        "_formal_stage3_matrix_plan",
        lambda *_args: (
            thresholds,
            DEFAULT_CANDIDATE_RULES,
            ("unit-00",),
            {"unit-00": {"model": "14M", "stage": "early", "update": "u0", "probe": "p0"}},
            plan_ref,
        ),
    )
    monkeypatch.setattr(
        task_runners,
        "_load_formal_document_ref",
        lambda *_args, **_kwargs: {"schema_version": "stage3-formal-pilot-plan-v1"},
    )
    monkeypatch.setattr(
        task_runners,
        "_load_stage3_production_index",
        lambda *_args, **_kwargs: (production_index, "plans/index.json"),
    )
    monkeypatch.setattr(
        task_runners,
        "_stage3_formal_reference_binding",
        lambda *_args, **_kwargs: {
            "formal_plan_hash": plan_hash,
            "reference_binding_hash": binding_hash,
        },
    )
    monkeypatch.setattr(
        task_runners,
        "_stage3_reference_refinement",
        lambda *_args, **_kwargs: (
            SimpleNamespace(converged=False, status="NOT_CONVERGED"),
            (),
        ),
    )
    monkeypatch.setattr(
        task_runners,
        "_quadrature_observations",
        lambda *_args, **_kwargs: candidate_calls.append(True),
    )
    try:
        task_runners._run_stage3_formal_matrix_shard(
            request,  # type: ignore[arg-type]
            tmp_path,
            TaskArtifactStore(tmp_path, "task-output"),
        )
    except TaskBlockedError as error:
        assert len(error.blockers) == 1
        assert error.blockers[0].requirement == "stage3.07_reference_convergence"
    else:  # pragma: no cover
        raise AssertionError("non-converged reference must block before candidates")
    assert candidate_calls == []


def test_matrix_resume_after_raw_skips_candidate_evaluation(tmp_path, monkeypatch) -> None:
    execution_hash = _hash("resume-execution")
    binding_hash = _hash("resume-binding")
    plan_hash = _hash("resume-plan")
    required_units = ("unit-00",)
    candidate_names = DEFAULT_CANDIDATE_RULES
    receipt_root = tmp_path / "resume" / "streaming"
    raw_hash = _hash("resume-raw")
    bundle_hash = _hash("resume-bundle")
    seal_hash = _hash("resume-seal")
    reference = {
        "schema_version": "stage3-task-path-integral-reference-v1",
        "refinement": {"conservative_error": 0.0},
        "path_identity_hash": _hash("resume-path"),
    }
    reference_path = tmp_path / "reference" / "unit-00.json"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(reference_path, {"reference": reference})
    receipt = {
        "unit_id": "unit-00",
        "reference_shard_ref": reference_path.relative_to(tmp_path).as_posix(),
        "reference_artifact_hash": canonical_json_hash(reference),
        "reference_aggregate_ref": "reference/aggregate.json",
        "reference_aggregate_hash": _hash("resume-reference-aggregate"),
        "observation_ledger_ref": "observation/unit-00.json",
        "raw_shard_ref": "raw/unit-00.json",
        "raw_shard_hash": raw_hash,
        "raw_bundle_ref": "raw/unit-00-bundle",
        "raw_bundle_manifest_hash": bundle_hash,
        "raw_aggregate_ref": "raw/aggregate.json",
        "raw_aggregate_hash": _hash("resume-raw-aggregate"),
        "node_cache_seal_ref": "receipts/unit-00.SEALED.json",
        "node_cache_seal_hash": seal_hash,
        "node_cache_evidence_hash": _hash("resume-evidence"),
        "artifact_hash": _hash("resume-unit-receipt"),
    }
    raw_shard = {
        "artifact_hash": raw_hash,
        "bundle_ref": receipt["raw_bundle_ref"],
        "bundle_manifest_hash": bundle_hash,
    }
    raw_aggregate = {
        "artifact_hash": _hash("resume-raw-aggregate"),
        "complete_unit_ids": list(required_units),
        "missing_unit_ids": [],
        "unit_shards": {"unit-00": {"shard_hash": raw_hash}},
    }
    reference_aggregate = {"artifact_hash": _hash("resume-reference-aggregate")}
    observation_rows = tuple(
        SimpleNamespace(
            unit_id="unit-00",
            rule_name=name,
            completeness_absolute_residual=0.0,
            completeness_relative_residual=0.0,
            completeness_l1_scaled_residual=0.0,
        )
        for name in candidate_names
    )
    context = SimpleNamespace(
        unit_id="unit-00",
        path=SimpleNamespace(identity_hash=reference["path_identity_hash"]),
        execution=SimpleNamespace(run_intent="formal", artifact_hash=execution_hash),
        node_cache_root_ref="cache/unit-00",
        node_cache=SimpleNamespace(
            evict=lambda *_args, **_kwargs: {
                "tombstone_ref": "receipts/unit-00.EVICTED.json",
                "tombstone_hash": _hash("resume-eviction"),
            }
        ),
    )
    request = SimpleNamespace(
        task=SimpleNamespace(
            task_id="stage3.07_formal_experiment_matrix",
            formal_eligibility=SimpleNamespace(required_gate_ids=()),
        ),
        config=SimpleNamespace(run_intent="formal"),
    )
    inputs = SimpleNamespace(references=("inputs/predecessor.json",), binding_hash=_hash("inputs"))
    thresholds = SimpleNamespace(artifact_hash=_hash("resume-thresholds"))
    production_index = SimpleNamespace(artifact_hash=_hash("resume-index"))
    reference_binding = {
        "reference_binding_hash": binding_hash,
        "formal_plan_hash": plan_hash,
        "reference_ladder_hash": _hash("ladder"),
        "reference_ladder_levels": {},
        "reference_ladder_nodes": {},
        "reference_tolerance": 0.001,
        "required_consecutive": 2,
        "primary_family": "gauss_legendre",
    }
    monkeypatch.setattr(
        task_runners,
        "_publish_stage3_reference_aggregate",
        lambda *_args, **_kwargs: (reference_aggregate, "reference/aggregate.json"),
    )
    monkeypatch.setattr(
        task_runners,
        "_publish_stage3_raw_aggregate_metadata",
        lambda **_kwargs: (raw_aggregate, "raw/aggregate.json"),
    )
    raw_shard_loads = []

    def fake_load_raw_shard(**kwargs):
        raw_shard_loads.append(kwargs["shard_ref"])
        return raw_shard, {}, None

    monkeypatch.setattr(task_runners, "_load_stage3_raw_shard", fake_load_raw_shard)
    monkeypatch.setattr(
        task_runners,
        "_load_stage3_observation_aggregate",
        lambda *_args, **_kwargs: (observation_rows, ("unit-00",), ()),
    )
    monkeypatch.setattr(
        task_runners,
        "_observation_payload",
        lambda _item: {},
    )
    monkeypatch.setattr(
        task_runners,
        "_persist_stage3_streaming_eviction_receipt",
        lambda **_kwargs: (
            {"artifact_hash": _hash("resume-eviction-receipt")},
            "resume/streaming/evictions/unit-00.json",
        ),
    )
    monkeypatch.setattr(
        task_runners,
        "_publish_stage3_streaming_aggregate",
        lambda **_kwargs: (
            {
                "required_unit_ids": list(required_units),
                "candidate_rule_names": list(candidate_names),
                "reference_complete_unit_ids": list(required_units),
                "raw_complete_unit_ids": list(required_units),
                "observation_complete_unit_ids": list(required_units),
                "receipt_complete_unit_ids": list(required_units),
                "sealed_unit_ids": list(required_units),
                "evicted_unit_ids": list(required_units),
                "committed_unit_ids": list(required_units),
                "missing_unit_ids": [],
                "artifact_hash": _hash("resume-streaming-aggregate"),
            },
            "resume/streaming/aggregate.json",
        ),
    )
    monkeypatch.setattr(
        task_runners.PersistentNodeGradientCache,
        "verify_receipt",
        staticmethod(
            lambda *_args, **_kwargs: {
                "state": "SEALED",
                "receipt_hash": seal_hash,
                "unit_id": "unit-00",
                "downstream_raw_shard_hash": raw_hash,
            }
        ),
    )
    monkeypatch.setattr(
        task_runners,
        "_gate_candidate",
        lambda _request: {"schema_version": "test-gate"},
    )
    candidate_calls = []
    monkeypatch.setattr(
        task_runners,
        "_quadrature_observations",
        lambda *_args, **_kwargs: candidate_calls.append(True),
    )
    payloads, _refs = _resume_stage3_formal_matrix_from_receipt(
        request,  # type: ignore[arg-type]
        tmp_path,
        TaskArtifactStore(tmp_path, "task-output"),
        inputs=inputs,
        context=context,
        thresholds=thresholds,
        candidate_names=candidate_names,
        required_units=required_units,
        unit_strata={"unit-00": {}},
        plan_ref="plans/matrix.json",
        production_index=production_index,
        production_index_ref="plans/index.json",
        reference_binding=reference_binding,
        receipt_root=receipt_root,
        receipt=receipt,
    )
    assert candidate_calls == []
    assert raw_shard_loads == [receipt["raw_shard_ref"]]
    assert tuple(payloads) == (
        "formal_path_results",
        "completeness_report",
    )
    assert payloads["formal_path_results"]["streaming_coverage"]["committed_unit_ids"] == [
        "unit-00"
    ]
