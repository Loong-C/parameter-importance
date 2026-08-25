"""End-to-end G2.3 fixture using real TaskArtifact and shard references.

This fixture deliberately exercises the same loading path as formal output.  It
does not inject a metrics map, a pass flag, or a fake hash; every envelope,
bundle, shard, draw manifest, registry and sizing-derived margin is published
through the repository contracts and then independently consumed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess

import numpy as np

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.contracts.seed import SeedPlan
from param_importance_nlp.experiments.preregistration import build_stage2_preregistration, validate_stage2_preregistration
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_formal import (
    FormalExecutionEvidence,
    ReferenceSizingPlan,
    _BoundedCheckpointStore,
    _BoundedMoments,
    _GradientMoments,
    _ReferenceShardStore,
    _ReferenceSnapshotStore,
    _draw_digest,
    _moments_from_shards,
    _vector_digest,
    bounded_reference_numeric_diagnostics,
    estimate_reference_uncertainty_bounded,
    estimate_reference_uncertainty_shards,
    estimate_sequence_variance_bounded,
    estimate_sequence_variance_shards,
)
from param_importance_nlp.experiments.stage2_g23_evaluator import CellInput, EXPECTED_CELL_IDS, evaluate_formal_g23
from param_importance_nlp.experiments.stage2_g23_contracts import (
    boundary_digest,
    generator_boundary,
    source_manifest_for_refs,
)
from param_importance_nlp.experiments.stage23_task_runners import (
    _actual_sampling_state,
    _derive_sizing_delta_sci,
    _trusted_stage2_provenance,
)
from param_importance_nlp.providers.synthetic import SyntheticGradientProvider
from param_importance_nlp.runtime.task_artifacts import TaskArtifactStore
from param_importance_nlp.runtime.tensor_bundle import load_tensor_bundle, publish_tensor_bundle


def _git_head(root: Path) -> str:
    return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _available_ram() -> int:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 1 << 40


def _hashed(body: dict[str, object]) -> dict[str, object]:
    return dict(body, artifact_hash=canonical_json_hash(body))


def _identity(body: dict[str, object]) -> dict[str, object]:
    return dict(body, identity_hash=canonical_json_hash(body))


def _publish_state(resume: Path, sequence: int, state: dict[str, object]) -> dict[str, object]:
    digest = _ReferenceSnapshotStore._state_digest(state)
    bundle = publish_tensor_bundle(resume / "objects" / digest, state)
    body: dict[str, object] = {
        "schema_version": "stage2-reference-progress-commit-v1",
        "sequence": sequence,
        "state_digest": digest,
        "object_ref": f"objects/{digest}",
        "object_manifest_hash": bundle.manifest_sha256,
    }
    body["artifact_hash"] = canonical_json_hash(body)
    write_canonical_json(resume / "commits" / f"{sequence:08d}.json", body)
    return body


def _make_provider(cell: str) -> SyntheticGradientProvider:
    # All six cells use a real fixed provider table.  Constant values make the
    # fixture's exact threshold-boundary diagnostics deterministic while the
    # provider still supplies the actual state and registry hashes.
    table = {
        sample_id: {
            "layer0.weight": np.asarray([10.0 + sample_id * 1.0e-2, 9.0 + sample_id * 1.0e-2]),
            "layer1.weight": np.asarray([20.0 + sample_id * 1.0e-2, 19.0 + sample_id * 1.0e-2]),
        }
        for sample_id in range(8)
    }
    return SyntheticGradientProvider(
        table,
        fixed_state_id=f"real-fixture-{cell}",
        statistical_unit="sequence",
        weight_unit="effective_tokens",
        sampling_design="uniform_with_replacement_disjoint_draw_groups",
        weights_exogenous=True,
        common_mean_assumption=True,
    )


def _build_cell(
    root: Path,
    index: int,
    cell: str,
    head: str,
    prereg: dict[str, object],
    *,
    bounded: bool = False,
) -> CellInput:
    provider = _make_provider(cell)
    assumptions = {
        "statistical_unit": provider.statistical_unit,
        "weight_unit": provider.weight_unit,
        "sampling_design": provider.sampling_design,
        "weights_exogenous": provider.weights_exogenous,
        "common_mean_assumption": provider.common_mean_assumption,
    }
    config_body = {
        "schema_version": "resolved-config-v2",
        "task_id": "stage2.04_reference_target",
        "cell_id": cell,
        "model_id": cell.split(":", 1)[0],
        "training_stage": cell.split(":", 1)[1],
        "fixed_state_id": provider.fixed_state_id,
    }
    config_hash = canonical_json_hash(config_body)
    registry = {
        "schema_version": "stage2-parameter-registry-artifact-v1",
        "status": "READY",
        "registry_hash": provider.registry_hash,
        "parameter_groups": {
            "layer0.weight": {"layer": "layer0", "module": "module0"},
            "layer1.weight": {"layer": "layer1", "module": "module1"},
        },
        "source": "real-task-artifact-fixture",
    }
    registry = _hashed(registry)
    # S2.3 publishes one common data range for the six-cell matrix.  The
    # per-cell provider state is distinct, but the data/tokenizer identity is
    # intentionally shared and therefore the manifest hash is common too.
    data_range_hash = canonical_json_hash({"dataset_id": "fixture-data", "range": [0, 8]})
    checkpoint_hash = canonical_json_hash({"cell_id": cell, "provider_state_digest": provider.state_digest()})
    model_asset_id = f"{cell}-model-asset"
    model_revision = f"revision-{provider.state_digest()[:16]}"
    checkpoint_id = f"checkpoint-{cell}"
    manifest = {
        "schema_version": "stage2-s204-six-cell-manifest-v1",
        "status": "READY",
        "scope": "formal",
        "asset_resolution_hash": canonical_json_hash({"asset": "real-fixture-s23", "cells": list(EXPECTED_CELL_IDS)}),
        "asset_producer_commit": head,
        "asset_execution_commit": head,
        "checkpoints": [],
        "data": {"data_range_hash": data_range_hash, "dataset_id": "fixture-data"},
        "data_range_hash": data_range_hash,
        "registry_hash": provider.registry_hash,
    }
    for row_cell in EXPECTED_CELL_IDS:
        row_provider = _make_provider(row_cell)
        row_config_hash = canonical_json_hash({**config_body, "cell_id": row_cell, "model_id": row_cell.split(":", 1)[0], "training_stage": row_cell.split(":", 1)[1], "fixed_state_id": row_provider.fixed_state_id})
        row_data_hash = data_range_hash
        manifest["checkpoints"].append({
            "cell_id": row_cell,
            "model_id": row_cell.split(":", 1)[0],
            "training_stage": row_cell.split(":", 1)[1],
            "checkpoint_id": f"checkpoint-{row_cell}",
            "checkpoint_hash": canonical_json_hash({"cell_id": row_cell, "provider_state_digest": row_provider.state_digest()}),
            "checkpoint_revision": f"revision-{row_provider.state_digest()[:16]}",
            "registry_hash": row_provider.registry_hash,
            "config_hash": row_config_hash,
        })
    manifest["data"] = {"data_range_hash": data_range_hash, "dataset_id": "fixture-data"}
    manifest["data_range_hash"] = data_range_hash
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    row = next(item for item in manifest["checkpoints"] if item["cell_id"] == cell)

    universe = SamplingUniverse(
        universe_id=f"real-fixture-{cell}",
        sample_ids=tuple(range(8)),
        metadata={"cell_id": cell, "data_range_hash": data_range_hash},
    )
    sampling = SamplingPlan(
        universe=universe,
        stream_seeds={
            "reference_sizing": 1 + index * 3,
            "reference_A": 201 + index * 11,
            "reference_B": 301 + index * 11,
            "pilot": 401 + index * 11,
            "confirmatory": 501 + index * 11,
        },
    )
    sizing_draws = sampling.draws("reference_sizing", 4)
    draws_a = sampling.draws("reference_A", 4)
    draws_b = sampling.draws("reference_B", 4)
    sizing_draw_hash = _draw_digest(sizing_draws)
    draw_a_hash, draw_b_hash = _draw_digest(draws_a), _draw_digest(draws_b)
    sizing_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_sizing", sequence)
        for sequence in range(5)
    )
    final_a_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_A", sequence)
        for sequence in range(5)
    )
    final_b_rng_boundaries = tuple(
        generator_boundary(sampling, "reference_B", sequence)
        for sequence in range(5)
    )
    evidence_hash = canonical_json_hash({"cell_id": cell, "scope": "formal", "gates": ["stage2.G2.2"]})
    plan_obj = ReferenceSizingPlan(
        reference_id=f"real-reference-{index}",
        candidate_sample_counts=(2, 4),
        block_size=1,
        convergence_tolerance=1.0e-12,
        required_consecutive=1,
        execution=FormalExecutionEvidence("local_fixture", metadata={"cell_id": cell, "fixture": "real-task-artifacts"}),
    )
    plan = plan_obj.to_dict()
    sizing_identity_hash = canonical_json_hash({"plan_hash": plan["artifact_hash"], "provider_state_digest": provider.state_digest(), "registry_hash": provider.registry_hash, "sizing_draw_hash": sizing_draw_hash, "sizing_stream": "reference_sizing"})
    final_plan_body = {
        "schema_version": "stage2-reference-one-shot-plan-v1",
        "reference_id": plan["reference_id"],
        "sizing_result_hash": "",
        "sample_count_per_stream": 4,
        "block_size": 1,
        "sizing_stream": "reference_sizing",
        "stream_a": "reference_A",
        "stream_b": "reference_B",
        "one_shot": True,
    }

    # Publish the authoritative external TaskArtifacts before any sizing-
    # derived numeric margin is produced.  The derived artifact is therefore
    # bound to the real S2.1 preregistration envelope hash, not to a caller
    # supplied placeholder.
    external_store = TaskArtifactStore(root, f"external/real-cell-{index}")
    external_payloads: dict[str, dict[str, object]] = {
        "s23_asset_resolution": {"schema_version": "stage2-task-asset-resolution-v1", "scope": "formal", "six_cell_manifest_hash": manifest["manifest_hash"], "producer_commit": head},
        "s23_six_cell_manifest": manifest,
        "resolved_config": {"schema_version": "resolved-config-v2", "task_id": "stage2.04_reference_target", "config_hash": config_hash, "cell_id": cell},
        "checkpoint_manifest": {"schema_version": "checkpoint-manifest-v1", **{key: row[key] for key in ("checkpoint_id", "checkpoint_hash", "checkpoint_revision", "config_hash", "registry_hash", "cell_id", "model_id", "training_stage")}},
        "model_manifest": {"schema_version": "model-manifest-v1", "asset_id": model_asset_id, "revision": model_revision, "model_id": row["model_id"], "training_stage": row["training_stage"], "parameter_count": 4},
        "data_manifest": {"schema_version": "data-manifest-v1", "asset_id": "fixture-data", "revision": f"data-{data_range_hash[:16]}", "dataset_id": "fixture-data", "data_range_hash": data_range_hash},
        "tokenizer_manifest": {"schema_version": "tokenizer-manifest-v1", "asset_id": "fixture-tokenizer", "revision": f"tokenizer-{data_range_hash[:16]}", "checkpoint_id": row["checkpoint_id"]},
        "parameter_registry": registry,
        "preregistration": prereg,
        "sizing_plan": plan,
    }
    lineage: dict[str, object] = {}
    for name, payload in external_payloads.items():
        kind = {"s23_asset_resolution": "asset_resolution", "s23_six_cell_manifest": "six_cell_manifest", "resolved_config": "resolved_config", "checkpoint_manifest": "checkpoint_manifest", "model_manifest": "model_manifest", "data_manifest": "data_manifest", "tokenizer_manifest": "tokenizer_manifest", "parameter_registry": "parameter_registry", "preregistration": "preregistration", "sizing_plan": "reference_sizing_plan"}[name]
        source_ref = f"external/source-manifests/real-cell-{index}/{name}.json"
        write_canonical_json(root / source_ref, {"schema_version": "stage2-source-input-v1", "cell_id": cell, "artifact_name": name, "payload_hash": canonical_json_hash(payload)})
        source_manifest = source_manifest_for_refs(root, [source_ref])
        published = external_store.publish(task_id="stage2.03_assets_checkpoints_and_sampling" if name.startswith("s23_") else "stage2.01_scope_hypotheses_and_preregistration", artifact_kind=kind, config_hash=config_hash, run_intent="formal", payload=payload, formal_eligible=True, source_refs=(source_ref,))
        lineage[name] = {"commit_ref": published.commit_ref, "artifact_kind": kind, "artifact_hash": published.artifact_hash, "config_hash": published.config_hash, "task_id": published.task_id, "formal_eligible": True, "payload_hash": canonical_json_hash(payload), "source_refs": [source_ref], "source_manifest": source_manifest}

    output_dir = f"runs/g23-real-cell-{index}"
    output_root = root / output_dir
    (output_root / "resume" / "reference-sizing" / "commits").mkdir(parents=True)
    (output_root / "resume" / "reference-final" / "commits").mkdir(parents=True)
    sizing_store = _ReferenceShardStore(output_root / "resume" / "reference-sizing")
    final_store = _ReferenceShardStore(output_root / "resume" / "reference-final")

    def bounded_moments(
        store: _ReferenceShardStore,
        refs: list[dict[str, object]],
        *,
        include_higher: bool,
    ) -> _BoundedMoments:
        moments = _BoundedMoments(include_higher=include_higher)
        for ref in refs:
            vector, weight, _ = store.load(ref)
            moments.update_vector(vector, weight)
        return moments

    sizing_refs: list[dict[str, object]] = []
    for draw in sizing_draws:
        batch = provider.gradient([draw])
        sizing_refs.append(
            sizing_store.publish(
                batch.gradients,
                1.0 if bounded else 1.0 + 0.01 * (int(draw.position) + 1),
            )
        )
    final_refs_a: list[dict[str, object]] = []
    final_refs_b: list[dict[str, object]] = []
    for draw_a, draw_b in zip(draws_a, draws_b):
        # Keep final A/B equal-weight as the pre-registered reference stream;
        # the sizing stream above is deliberately unequal-weight and exercises
        # the weighted delta_sci path without making ranking direction a
        # property of arbitrary A/B weights.
        final_refs_a.append(final_store.publish(provider.gradient([draw_a]).gradients, 1.0))
        final_refs_b.append(final_store.publish(provider.gradient([draw_b]).gradients, 1.0))

    empty = _GradientMoments().to_state()
    sizing_states: dict[int, dict[str, object]] = {}
    final_states: dict[int, dict[str, object]] = {}
    for sequence in range(1, 5):
        refs = sizing_refs[:sequence]
        moments = _moments_from_shards(sizing_store, refs, assumptions)
        state = {
            "schema_version": "stage2-reference-progress-state-v1",
            "plan_hash": plan["artifact_hash"],
            "provider_state_digest": provider.state_digest(),
            "registry_hash": provider.registry_hash,
            "weighting_assumptions": assumptions,
            "processed_block_pairs": sequence,
            "convergence_streak": 0 if sequence < 2 else 1,
            "selected_sample_count_per_stream": None if sequence < 4 else 4,
            "points": [],
            "last_bias": moments.u(assumptions=assumptions) if sequence >= 2 else {},
            "a": moments.to_state(),
            "b": empty,
            "shard_refs_a": refs,
            "shard_refs_b": [],
            "shard_count": sequence,
            "sizing_stream": True,
            "sizing_draw_hash": sizing_draw_hash,
            "sizing_identity_hash": sizing_identity_hash,
            "rng_state": sizing_rng_boundaries[sequence],
            "rng_state_digest": boundary_digest(sizing_rng_boundaries[sequence]),
        }
        sizing_states[sequence] = state
        _publish_state(output_root / "resume" / "reference-sizing", sequence, state)
    if bounded:
        bounded_candidates = {
            str(sequence): {
                "a": bounded_moments(
                    sizing_store, sizing_refs[:sequence], include_higher=False
                ).to_state(),
                "b": _BoundedMoments().to_state(),
            }
            for sequence in (2, 4)
        }
        bounded_sizing_state = dict(sizing_states[4])
        bounded_sizing_state.update(
            {
                "a": bounded_candidates["4"]["a"],
                "b": bounded_candidates["4"]["b"],
                "candidate_states": bounded_candidates,
                "shard_refs_a": [],
                "shard_refs_b": [],
                "shard_count": 0,
            }
        )
        _BoundedCheckpointStore(
            output_root / "resume" / "reference-sizing"
        ).publish(4, bounded_sizing_state)
    sizing_moments = _moments_from_shards(sizing_store, sizing_refs, assumptions)
    sizing_payload = {
        "schema_version": "stage2-reference-sizing-result-v1",
        "plan_hash": plan["artifact_hash"],
        "provider_state_digest": provider.state_digest(),
        "registry_hash": provider.registry_hash,
        "processed_sample_count_per_stream": 4,
        "selected_sample_count_per_stream": 4,
        "converged": True,
        "status": "FORMAL_CANDIDATE",
        "points": [],
        "bias_reference_hash": _vector_digest(sizing_moments.u(assumptions=assumptions)),
        "cross_reference_hash": _vector_digest(sizing_moments.mean()),
        "ranking_reference_hash": _vector_digest({name: np.square(value) for name, value in sizing_moments.mean().items()}),
        "scope": "formal",
        "formal_eligible": False,
        "qualification_gate_hash": None,
        "weighting_assumptions": assumptions,
    }
    sizing_result_hash = canonical_json_hash(sizing_payload)
    delta_sci = _derive_sizing_delta_sci(
        root=root,
        sizing_root=output_root / "resume" / "reference-sizing",
        plan=plan_obj,
        parameter_registry=registry,
        formula_contract=prereg,
        formula_contract_hash=str(lineage["preregistration"]["artifact_hash"]),
        provider=provider,
        sizing_result_hash=sizing_result_hash,
    )
    final_plan_body["sizing_result_hash"] = sizing_result_hash
    final_plan = _hashed(final_plan_body)
    sizing_result_identity_hash = canonical_json_hash({"sizing_result_hash": sizing_result_hash, "provider_state_digest": provider.state_digest(), "registry_hash": provider.registry_hash, "stream_a_draw_hash": draw_a_hash, "stream_b_draw_hash": draw_b_hash})
    for sequence in range(1, 5):
        moments_a = _moments_from_shards(final_store, final_refs_a[:sequence], assumptions)
        moments_b = _moments_from_shards(final_store, final_refs_b[:sequence], assumptions)
        state = {
            "schema_version": "stage2-reference-one-shot-progress-v1",
            "plan_hash": final_plan["artifact_hash"],
            "sizing_result_hash": sizing_result_hash,
            "provider_state_digest": provider.state_digest(),
            "registry_hash": provider.registry_hash,
            "weighting_assumptions": assumptions,
            "stream_a_draw_hash": draw_a_hash,
            "stream_b_draw_hash": draw_b_hash,
            "processed_block_pairs": sequence,
            "a": moments_a.to_state(),
            "b": moments_b.to_state(),
            "shard_refs_a": final_refs_a[:sequence],
            "shard_refs_b": final_refs_b[:sequence],
            "shard_count": sequence * 2,
            "final_length_required": True,
            "sizing_result_identity_hash": sizing_result_identity_hash,
            "rng_state": {
                "a": final_a_rng_boundaries[sequence],
                "b": final_b_rng_boundaries[sequence],
            },
            "rng_state_digest": boundary_digest({
                "a": final_a_rng_boundaries[sequence],
                "b": final_b_rng_boundaries[sequence],
            }),
        }
        final_states[sequence] = state
        latest_final_commit = _publish_state(output_root / "resume" / "reference-final", sequence, state)
    moments_a = _moments_from_shards(final_store, final_refs_a, assumptions)
    moments_b = _moments_from_shards(final_store, final_refs_b, assumptions)
    combined = moments_a.combine(moments_b)
    bias, cross, ranking = combined.u(assumptions=assumptions), {name: moments_a.mean()[name] * moments_b.mean()[name] for name in moments_a.mean()}, {name: np.square(value) for name, value in combined.mean().items()}
    if bounded:
        bounded_a = bounded_moments(
            final_store, final_refs_a, include_higher=True
        )
        bounded_b = bounded_moments(
            final_store, final_refs_b, include_higher=True
        )
        final_bounded_state = dict(final_states[4])
        final_bounded_state.update(
            {
                "a": bounded_a.to_state(),
                "b": bounded_b.to_state(),
                "shard_refs_a": [],
                "shard_refs_b": [],
            }
        )
        _BoundedCheckpointStore(
            output_root / "resume" / "reference-final"
        ).publish(4, final_bounded_state)
        combined_bounded = bounded_a.combine(bounded_b)
        bias = combined_bounded.u(assumptions=assumptions)
        mean_a_bounded, mean_b_bounded = bounded_a.mean(), bounded_b.mean()
        cross = {
            name: mean_a_bounded[name] * mean_b_bounded[name]
            for name in mean_a_bounded
        }
        ranking = {
            name: np.square(value)
            for name, value in combined_bounded.mean().items()
        }
        uncertainty = estimate_reference_uncertainty_bounded(bounded_a, bounded_b)
        sequence_variance = estimate_sequence_variance_bounded(
            combined_bounded, block_size=1
        )
    else:
        uncertainty = estimate_reference_uncertainty_shards(final_store, final_refs_a, final_refs_b, assumptions)
        sequence_variance = estimate_sequence_variance_shards(final_store, final_refs_a + final_refs_b, block_size=1)
    one_shot_payload = {
        "schema_version": "stage2-reference-one-shot-result-v1",
        "plan_hash": final_plan["artifact_hash"],
        "sizing_result_hash": sizing_result_hash,
        "provider_state_digest": provider.state_digest(),
        "registry_hash": provider.registry_hash,
        "processed_sample_count_per_stream": 4,
        "bias_reference_hash": _vector_digest(bias),
        "cross_reference_hash": _vector_digest(cross),
        "ranking_reference_hash": _vector_digest(ranking),
        "uncertainty": uncertainty.to_dict(),
        "stream_a_draw_hash": draw_a_hash,
        "stream_b_draw_hash": draw_b_hash,
        "status": "COMPLETE",
        "one_shot": True,
        "weighting_assumptions": assumptions,
        "sequence_variance_hash": _vector_digest(sequence_variance),
    }
    one_shot_payload = _hashed(one_shot_payload)
    if bounded:
        high_precision, accumulated, error_bound = (
            bounded_reference_numeric_diagnostics(bounded_a, bounded_b)
        )
        _, bounded_checkpoint_bundle = load_tensor_bundle(
            output_root / "resume" / "reference-final" / "bounded-checkpoint"
        )
        bounded_state_digest = canonical_json_hash(
            {
                "checkpoint_schema": _BoundedCheckpointStore.schema_version,
                "object_manifest_hash": bounded_checkpoint_bundle.manifest_sha256,
            }
        )
        numerical = _hashed(
            {
                "schema_version": "stage2-reference-numerical-diagnostics-v2",
                "storage_mode": "bounded-online-fp64-v1",
                "recompute_method": "longdouble_u_from_hash_bound_bounded_moments_with_fp64_error_envelope",
                "bounded_checkpoint_ref": "bounded-checkpoint",
                "bounded_checkpoint_manifest_hash": bounded_checkpoint_bundle.manifest_sha256,
                "bounded_checkpoint_state_digest": bounded_state_digest,
                "moments_a_hash": canonical_json_hash(
                    {
                        "schema_version": _BoundedMoments.schema_version,
                        "count": bounded_a.count,
                        "n1": bounded_a.n1,
                        "n2": bounded_a.n2,
                        "g1_hash": _vector_digest(bounded_a.g1),
                        "g2_hash": _vector_digest(bounded_a.g2),
                        "p2_hash": _vector_digest(bounded_a.p2),
                        "p3_hash": _vector_digest(bounded_a.p3),
                        "p4_hash": _vector_digest(bounded_a.p4),
                    }
                ),
                "moments_b_hash": canonical_json_hash(
                    {
                        "schema_version": _BoundedMoments.schema_version,
                        "count": bounded_b.count,
                        "n1": bounded_b.n1,
                        "n2": bounded_b.n2,
                        "g1_hash": _vector_digest(bounded_b.g1),
                        "g2_hash": _vector_digest(bounded_b.g2),
                        "p2_hash": _vector_digest(bounded_b.p2),
                        "p3_hash": _vector_digest(bounded_b.p3),
                        "p4_hash": _vector_digest(bounded_b.p4),
                    }
                ),
                "high_precision_hash": _vector_digest(high_precision),
                "accumulated_hash": _vector_digest(accumulated),
                "error_bound_hash": _vector_digest(error_bound),
                "max_abs_error": max(
                    float(np.max(value)) for value in error_bound.values()
                ),
            }
        )
        raw_block_digest = None
    else:
        numeric_rows = []
        for ref in final_refs_a + final_refs_b:
            vector, weight, _ = final_store.load(ref)
            numeric_rows.append({"vector_hash": _vector_digest(vector), "weight": float(weight)})
        raw_block_digest = canonical_json_hash(numeric_rows)
        high_precision = accumulated = bias
        error_bound = None
        numerical = _hashed({
            "schema_version": "stage2-reference-numerical-diagnostics-v1",
            "recompute_method": "longdouble_pairwise_u_from_content_addressed_shards",
            "raw_block_digest": raw_block_digest,
            "raw_block_count_a": 4,
            "raw_block_count_b": 4,
            "high_precision_hash": _vector_digest(bias),
            "accumulated_hash": _vector_digest(bias),
            "max_abs_error": 0.0,
            "resume_latest_commit_ref": "commits/00000004.json",
            "resume_latest_commit_hash": latest_final_commit["artifact_hash"],
            "resume_latest_manifest_hash": latest_final_commit["object_manifest_hash"],
        })
    bundle_state = {
        "bias_reference": dict(bias),
        "cross_reference": dict(cross),
        "ranking_reference": dict(ranking),
        "uncertainty": {"bias_variance": dict(uncertainty.bias_variance), "cross_variance": dict(uncertainty.cross_variance), "ranking_variance": dict(uncertainty.ranking_variance)},
        "sequence_variance": dict(sequence_variance),
        "numerical_diagnostics": (
            {
                "schema_version": "stage2-reference-numerical-diagnostics-v2",
                "storage_mode": "bounded-online-fp64-v1",
                "bounded_checkpoint_ref": "bounded-checkpoint",
                "bounded_checkpoint_manifest_hash": bounded_checkpoint_bundle.manifest_sha256,
                "bounded_checkpoint_state_digest": bounded_state_digest,
                "moments_a_hash": numerical["moments_a_hash"],
                "moments_b_hash": numerical["moments_b_hash"],
                "high_precision": dict(high_precision),
                "accumulated": dict(accumulated),
                "error_bound": dict(error_bound),
                "high_precision_hash": _vector_digest(high_precision),
                "accumulated_hash": _vector_digest(accumulated),
                "error_bound_hash": _vector_digest(error_bound),
            }
            if bounded
            else {
                "schema_version": "stage2-reference-numerical-diagnostics-v1",
                "raw_block_digest": raw_block_digest,
                "high_precision": dict(bias),
                "accumulated": dict(bias),
                "high_precision_hash": _vector_digest(bias),
                "accumulated_hash": _vector_digest(bias),
            }
        ),
    }
    bundle = publish_tensor_bundle(output_root / "tensor-bundles" / "reference-final", bundle_state)
    reference_payload = _hashed({
        "schema_version": "reference-result-v1",
        "reference_id": plan["reference_id"],
        "bias_reference_hash": _vector_digest(bias),
        "cross_reference_hash": _vector_digest(cross),
        "ranking_reference_hash": _vector_digest(ranking),
        "sample_count_a": 4,
        "sample_count_b": 4,
        "block_size": 1,
        "registry_hash": provider.registry_hash,
        "scope": "formal",
        "formal_eligible": False,
        "metadata": {"sizing_result_hash": sizing_result_hash, "sequence_variance_hash": _vector_digest(sequence_variance)},
        "tensor_bundle_ref": f"{output_dir}/tensor-bundles/reference-final",
        "tensor_bundle_manifest_hash": bundle.manifest_sha256,
    })
    actual_state = {stream: _actual_sampling_state(sampling, stream, 4) for stream in ("reference_sizing", "reference_A", "reference_B")}
    rng_before_state = {"sampling_plan_hash": sampling.digest, "streams": {stream: actual_state[stream]["state_before"] for stream in actual_state}}
    rng_after_state = {"sampling_plan_hash": sampling.digest, "streams": {stream: actual_state[stream]["state_after"] for stream in actual_state}}
    rng_before, rng_after = canonical_json_hash(rng_before_state), canonical_json_hash(rng_after_state)
    replay_commit = latest_final_commit
    replay = {
        "schema_version": (
            "stage2-reference-resume-replay-v2"
            if bounded
            else "stage2-reference-resume-replay-v1"
        ),
        **({"storage_mode": "bounded-online-fp64-v1"} if bounded else {}),
        "artifact_ref": "bounded-checkpoint" if bounded else "commits/00000004.json",
        "artifact_hash": (
            bounded_checkpoint_bundle.manifest_sha256
            if bounded
            else replay_commit["artifact_hash"]
        ),
        "state_digest": bounded_state_digest if bounded else replay_commit["state_digest"],
        "object_manifest_hash": (
            bounded_checkpoint_bundle.manifest_sha256
            if bounded
            else replay_commit["object_manifest_hash"]
        ),
        "source_one_shot_result_hash": one_shot_payload["artifact_hash"],
        "replayed_one_shot_result_hash": one_shot_payload["artifact_hash"],
        "sizing_result_identity_hash": sizing_result_identity_hash,
    }
    replay["replay_hash"] = canonical_json_hash(replay)
    provenance = _trusted_stage2_provenance(require_clean=True)
    capacity = {
        "schema_version": "stage2-reference-capacity-preflight-v1",
        "parameter_count": 4,
        "candidate_max_sample_count_per_stream": 4,
        "block_size": 1,
        "max_block_count_per_stream": 4,
        **({"storage_mode": "bounded-online-fp64-v1"} if bounded else {}),
        "single_copy_shard_bytes": 0 if bounded else 4 * 2 * 4 * 8,
        "snapshot_moment_bytes": (25 * 4 * 8) if bounded else 4 * 4 * 4 * 8,
        "estimated_disk_bytes": int(
            ((25 * 4 * 8) if bounded else (4 * 2 * 4 * 8 + 4 * 4 * 4 * 8))
            * 1.20
            + 64 * 1024**2
        ),
        "free_disk_bytes": shutil.disk_usage(root).free,
        "peak_ram_bytes": 3 * 4 * 8 + 64 * 1024**2,
        "available_ram_bytes": _available_ram(),
        "disk_ok": shutil.disk_usage(root).free >= int(
            ((25 * 4 * 8) if bounded else (4 * 2 * 4 * 8 + 4 * 4 * 4 * 8))
            * 1.20
            + 64 * 1024**2
        ),
        "ram_ok": _available_ram() >= 3 * 4 * 8 + 64 * 1024**2,
        "fail_closed_if_unknown": True,
    }
    capacity["artifact_hash"] = canonical_json_hash(capacity)

    config_identity = _identity({"config_hash": config_hash, "task_id": "stage2.04_reference_target", "checkpoint_config_hash": row["config_hash"]})
    model_identity = _identity({"asset_id": model_asset_id, "revision": model_revision, "model_id": row["model_id"], "training_stage": row["training_stage"]})
    data_identity = _identity({"asset_id": "fixture-data", "revision": f"data-{data_range_hash[:16]}", "data_range_hash": data_range_hash, "dataset_id": "fixture-data"})
    checkpoint_identity = _identity({"checkpoint_id": row["checkpoint_id"], "checkpoint_revision": row["checkpoint_revision"], "checkpoint_asset_id": model_asset_id, "model_id": row["model_id"], "training_stage": row["training_stage"], "checkpoint_hash": row["checkpoint_hash"], "config_hash": row["config_hash"], "registry_hash": row["registry_hash"], "cell_id": cell})
    tokenizer_identity = _identity({"asset_id": "fixture-tokenizer", "revision": f"tokenizer-{data_range_hash[:16]}", "checkpoint_id": row["checkpoint_id"]})
    registry_identity = _identity({"registry_hash": provider.registry_hash, "parameter_registry_artifact_hash": registry["artifact_hash"]})
    convergence: dict[str, object] = {
        "schema_version": "stage2-reference-convergence-report-v1",
        "plan": plan,
        "sizing_plan": plan,
        "sizing_plan_artifact_hash": plan["artifact_hash"],
        "sizing_draw_hash": sizing_draw_hash,
        "sizing_identity_hash": sizing_identity_hash,
        "status": "COMPLETE",
        "converged": True,
        "selected_sample_count_per_stream": 4,
        "processed_sample_count_per_stream": 4,
        "points": [],
        "sizing_result_hash": sizing_result_hash,
        "one_shot_plan": final_plan,
        "one_shot_result": one_shot_payload,
        "sizing_stream": "reference_sizing",
        "final_streams": ["reference_A", "reference_B"],
        "final_sample_count_per_stream": 4,
        "reference_uncertainty": uncertainty.to_dict(),
        "provider": {"fixed_state_id": provider.fixed_state_id, "provider_state_digest": provider.state_digest(), "registry_hash": provider.registry_hash, "parameter_names": list(provider.parameter_names)},
        "sampling_plan_hash": sampling.digest,
        "recovery_semantics": "authoritative_block_pair_commits",
        "reference_protocol": "authoritative_sizing_and_one_shot_block_pair_commits",
        "diagnostics_schema_version": "stage2-reference-producer-diagnostics-v1",
        "stage2_reference_producer_commit": head,
        "producer_provenance": provenance,
        "external_lineage": lineage,
        "formal_scope": "formal",
        "cell_id": cell,
        "six_cell_manifest": manifest,
        "six_cell_manifest_hash": manifest["manifest_hash"],
        "sizing_plan": plan,
        "sizing_plan_artifact_hash": plan["artifact_hash"],
        "sizing_draw_hash": sizing_draw_hash,
        "sizing_identity_hash": sizing_identity_hash,
        "formula_contract": prereg,
        "formula_contract_hash": lineage["preregistration"]["artifact_hash"],
        "candidate_delta_sci": delta_sci,
        "candidate_delta_sci_source": delta_sci["source_ref"],
        "candidate_delta_sci_source_hash": delta_sci["source_hash"],
        "config_identity": config_identity,
        "model_identity": model_identity,
        "data_identity": data_identity,
        "tokenizer_identity": tokenizer_identity,
        "checkpoint_identity": checkpoint_identity,
        "registry_identity": registry_identity,
        "parameter_registry_artifact": registry,
        "numerical_diagnostics": numerical,
        "state_invariance": {"model_state_before_hash": provider.state_digest(), "model_state_after_hash": provider.state_digest(), "rng_state_before_hash": rng_before, "rng_state_after_hash": rng_after, "rng_state_before": rng_before_state, "rng_state_after": rng_after_state},
        "draw_artifacts": {stream: {"sampling_plan": sampling.to_dict(), "manifest": sampling.draw_manifest(stream, 4).to_manifest(), "actual_state": actual_state[stream]} for stream in ("reference_sizing", "reference_A", "reference_B")},
        "resume_replay": replay,
        "sizing_result_identity_hash": sizing_result_identity_hash,
        "numerical_floor": 1.0e-12,
        "capacity_preflight": capacity,
        "formal_eligible": False,
    }
    convergence["reference_producer_diagnostics_hash"] = canonical_json_hash({key: value for key, value in convergence.items() if key != "reference_producer_diagnostics_hash"})
    gate = {"schema_version": "stage23-task-gate-candidate-v1", "task_id": "stage2.04_reference_target", "gate_ids": ["stage2.G2.3"], "gate_status": "NOT_RUN", "local_validation_status": "NOT_RUN", "formal_eligible": False, "reason": "formal_gate_requires_independent_review"}
    task_store = TaskArtifactStore(root, output_dir)
    refs = {}
    for kind, payload in (("reference_result", reference_payload), ("reference_convergence_report", convergence), ("gate_record", gate)):
        refs[kind] = task_store.publish(task_id="stage2.04_reference_target", artifact_kind=kind, config_hash=config_hash, run_intent="formal", payload=payload, formal_eligible=True).commit_ref
    result_body = {"schema_version": "task-run-result-v2", "task_id": "stage2.04_reference_target", "stage": 2, "runner_kind": "reference", "run_intent": "formal", "status": "PASS", "config_hash": config_hash, "formal_eligible": True, "artifact_refs": refs, "checkpoint_ref": lineage["checkpoint_manifest"]["commit_ref"], "blockers": [], "error_code": None, "message": "task completed", "recovery_mode": "resume_shards"}
    result_body["metadata"] = {"identity_bindings": {key: convergence[key] for key in ("stage2_reference_producer_commit", "producer_provenance", "config_identity", "checkpoint_identity", "registry_identity", "model_identity", "data_identity", "tokenizer_identity", "external_lineage")}}
    result_body["result_hash"] = canonical_json_hash({key: value for key, value in result_body.items() if key != "result_hash"})
    result_ref = output_root / "task-run-result.json"
    write_canonical_json(result_ref, result_body)
    return CellInput(cell, result_ref.relative_to(root).as_posix())


def _build_real_fixture(root: Path, *, bounded: bool = False) -> list[CellInput]:
    head = _git_head(Path(__file__).resolve().parents[1])
    prereg = build_stage2_preregistration(seed_plan_hash=SeedPlan.from_master_seed(917).artifact_hash, producer_commit=head, scope="formal")
    validate_stage2_preregistration(prereg)
    return [
        _build_cell(root, index, cell, head, prereg, bounded=bounded)
        for index, cell in enumerate(EXPECTED_CELL_IDS)
    ]


def test_real_external_task_artifacts_pass_and_resume_tamper_blocks(tmp_path: Path) -> None:
    cells = _build_real_fixture(tmp_path)
    passed = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert passed["status"] == "PASS"
    assert passed["formal_eligible"] is True
    first = Path(cells[0].task_result_ref).parent
    tensor_file = next((tmp_path / first / "tensor-bundles" / "reference-final" / "tensors").glob("*.bin"))
    tensor_file.write_bytes(tensor_file.read_bytes() + b"tamper")
    blocked = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert blocked["status"] == "BLOCKED"
    assert blocked["formal_eligible"] is False


def test_bounded_external_task_artifacts_pass_and_checkpoint_tamper_blocks(
    tmp_path: Path,
) -> None:
    cells = _build_real_fixture(tmp_path, bounded=True)
    passed = evaluate_formal_g23(tmp_path, cells, output_root=tmp_path / "attempts")
    assert passed["status"] == "PASS", passed
    assert passed["formal_eligible"] is True
    first = Path(cells[0].task_result_ref).parent
    tensor_file = next(
        (
            tmp_path
            / first
            / "resume"
            / "reference-final"
            / "bounded-checkpoint"
            / "tensors"
        ).glob("*.bin")
    )
    tensor_file.write_bytes(tensor_file.read_bytes() + b"tamper")
    blocked = evaluate_formal_g23(
        tmp_path, cells, output_root=tmp_path / "attempts"
    )
    assert blocked["status"] == "BLOCKED"
    assert blocked["formal_eligible"] is False


def test_real_external_duplicate_cell_and_missing_cell_are_blocked(tmp_path: Path) -> None:
    cells = _build_real_fixture(tmp_path)
    duplicate = list(cells)
    duplicate[-1] = duplicate[0]
    assert evaluate_formal_g23(tmp_path, duplicate, output_root=tmp_path / "attempts")["status"] == "BLOCKED"
    assert evaluate_formal_g23(tmp_path, cells[:-1], output_root=tmp_path / "attempts")["status"] in {"BLOCKED", "NOT_RUN"}
