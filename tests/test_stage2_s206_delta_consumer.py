from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_s206_delta_consumer import (
    CorrectedDeltaRejected,
    load_bound_corrected_delta,
)
from param_importance_nlp.experiments.stage2_s206_formal import (
    ANCHOR_IDS,
    S206PreparationBlocked,
    build_global_pilot_mapping,
    reduce_blinded_pilot,
)
from param_importance_nlp.experiments.sampling import SamplingPlan, SamplingUniverse
from param_importance_nlp.experiments.stage2_pilot import CostSemantics


CELL_IDS = (
    "pythia-14m:initialization",
    "pythia-14m:early",
    "pythia-14m:mid_late",
    "pythia-31m-deduped:initialization",
    "pythia-31m-deduped:early",
    "pythia-31m-deduped:mid_late",
)
TARGET = CELL_IDS[0]
CONFIG_HASH = "1" * 64
RESULT_HASH = "2" * 64
SOURCE_COMMIT = "3" * 40
EVALUATOR_COMMIT = "4" * 40
SOURCE_SHA = "5" * 64


def _nodes() -> list[dict[str, object]]:
    return [
        {
            "sample_count": count,
            "state_digest": f"{count:064x}",
            "shard_refs_hash": "6" * 64,
            "mean_hash": "7" * 64,
            "sequence_variance_hash": "8" * 64,
        }
        for count in (131072, 262144)
    ]


def _source() -> dict[str, object]:
    old_table = {
        endpoint: {"131072": 1.0, "262144": 1.0}
        for endpoint in ("model_total", "layer", "module")
    }
    value: dict[str, object] = {
        "schema_version": "stage2-reference-delta-sci-v2",
        "source_kind": "reference_sizing_bounded_online",
        "formula_contract_hash": "9" * 64,
        "formula_version": "stage2-reference-sizing-margin-v1",
        "formula": "delta_sci=max(0.10*Delta,0.01*S); a=mu_sizing^2; d=sigma_squared_over_B",
        "absolute_floors": {
            "tau_model": 1e-12,
            "tau_layer": 1e-12,
            "tau_module": 1e-12,
            "tau_coord": 1e-12,
            "tau_nmse": 1e-12,
        },
        "reference_id": "reference-r23",
        "sizing_result_hash": "a" * 64,
        "sizing_plan_hash": "b" * 64,
        "registry_hash": "c" * 64,
        "candidate_sample_counts": [131072, 262144],
        # The legacy producer's tables are intentionally keyed by sizing
        # nodes and do not carry the evaluator's selected-node binding.
        "delta_sci_by_endpoint": old_table,
        "signal_scale_by_endpoint": old_table,
        "noise_scale_by_endpoint": old_table,
        "sizing_nodes": _nodes(),
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _sidecar(source_ref: str, source: dict[str, object]) -> dict[str, object]:
    signal = {
        endpoint: {str(batch): 2.0 for batch in (32, 64, 128, 256)}
        for endpoint in ("model_total", "layer", "module")
    }
    noise = {
        # 0.7 deliberately exercises a non-decimal-friendly binary float;
        # powers-of-two B scaling must still be verified without Decimal
        # string artifacts.
        endpoint: {str(batch): 0.7 / batch for batch in (32, 64, 128, 256)}
        for endpoint in ("model_total", "layer", "module")
    }
    delta = {
        endpoint: {
            key: max(0.10 * values[key], 0.01 * signal[endpoint][key])
            for key in values
        }
        for endpoint, values in noise.items()
    }
    value: dict[str, object] = {
        "schema_version": "stage2-g23-corrected-delta-sci-v1",
        "source_producer_schema_version": source["schema_version"],
        "source_producer_ref": source_ref,
        "source_producer_artifact_hash": source["artifact_hash"],
        "source_producer_table_mode": "sizing_nodes_legacy",
        "source_producer_commit": SOURCE_COMMIT,
        "evaluator_commit": EVALUATOR_COMMIT,
        "evaluator_source_sha256": SOURCE_SHA,
        "formula_contract_hash": source["formula_contract_hash"],
        "formula_version": source["formula_version"],
        "formula": source["formula"],
        "absolute_floors": source["absolute_floors"],
        "reference_id": source["reference_id"],
        "sizing_result_hash": source["sizing_result_hash"],
        "sizing_plan_hash": source["sizing_plan_hash"],
        "registry_hash": source["registry_hash"],
        "candidate_sample_counts": [131072, 262144],
        "delta_sci_batch_sizes": [32, 64, 128, 256],
        "selected_sample_count_per_stream": 262144,
        "delta_sci_by_endpoint": delta,
        "signal_scale_by_endpoint": signal,
        "noise_scale_by_endpoint": noise,
        "sizing_nodes": source["sizing_nodes"],
        "correction_reason": "producer_delta_sci_keyed_by_sizing_nodes; S2.6_requires_candidate_batch_sizes",
    }
    value["artifact_hash"] = canonical_json_hash(value)
    return value


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], dict[str, object]]:
    source = _source()
    source_ref = "evidence/run/producer/delta-sci.json"
    write_canonical_json(tmp_path / source_ref, source)
    sidecar_ref = "evidence/run/g2.3-corrected-delta-sci/PLACEHOLDER.json"
    sidecar = _sidecar(source_ref, source)
    sidecar_ref = sidecar_ref.replace("PLACEHOLDER", str(sidecar["artifact_hash"]))
    write_canonical_json(tmp_path / sidecar_ref, sidecar)
    cells: list[dict[str, object]] = []
    for cell_id in CELL_IDS:
        cells.append(
            {
                "cell_id": cell_id,
                "status": "PASS",
                "formal_eligible": True,
                "identities": {
                    "cell_id": cell_id,
                    "config_hash": CONFIG_HASH,
                    "result_hash": RESULT_HASH,
                    "producer_commit": SOURCE_COMMIT,
                    "sizing_plan_hash": source["sizing_plan_hash"],
                    "sizing_result_hash": source["sizing_result_hash"],
                    "reference_id": source["reference_id"],
                    "registry_hash": source["registry_hash"],
                    "corrected_delta_sci_hash": sidecar["artifact_hash"],
                    "corrected_delta_sci_ref": sidecar_ref,
                },
                "metrics": {
                    "corrected_delta_sci_hash": sidecar["artifact_hash"],
                    "corrected_delta_sci_ref": sidecar_ref,
                    "corrected_delta_sci_batch_sizes": [32, 64, 128, 256],
                    "delta_sci_source": "g23_output_derived_corrected_sidecar",
                },
            }
        )
    evaluation: dict[str, object] = {
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "gate_id": "stage2.G2.3",
        "status": "PASS",
        "formal_eligible": True,
        "required_cell_count": 6,
        "complete_cell_count": 6,
        "expected_cell_ids": list(CELL_IDS),
        "cells": cells,
        "calculator": {
            "producer_commit": SOURCE_COMMIT,
            "evaluator_commit": EVALUATOR_COMMIT,
            "source_sha256": SOURCE_SHA,
            "source_schema": "stage2-g23-reference-evaluation-v1",
        },
        "thresholds": {},
        "reasons": [],
    }
    evaluation["artifact_hash"] = canonical_json_hash(evaluation)
    evaluation_ref = "evidence/run/g2.3-attempts/eval/evaluation.json"
    write_canonical_json(tmp_path / evaluation_ref, evaluation)
    return tmp_path, evaluation, sidecar


def test_real_bounded_legacy_shape_consumes_only_corrected_four_b_table(tmp_path: Path) -> None:
    root, _evaluation, sidecar = _fixture(tmp_path)
    binding = load_bound_corrected_delta(
        root,
        g23_evaluation_ref="evidence/run/g2.3-attempts/eval/evaluation.json",
        cell_id=TARGET,
        expected_config_hash=CONFIG_HASH,
        expected_result_hash=RESULT_HASH,
        expected_sizing_plan_hash="b" * 64,
        expected_sizing_result_hash="a" * 64,
        expected_reference_id="reference-r23",
        expected_registry_hash="c" * 64,
    )
    assert binding.artifact_hash == sidecar["artifact_hash"]
    assert binding.delta_for(32) == {"bias": 0.02, "nmse": 0.02, "rank": 0.02}
    assert set(binding.delta_sci_by_endpoint["model_total"]) == {"32", "64", "128", "256"}


def test_tampered_sidecar_hash_and_unknown_field_fail_closed(tmp_path: Path) -> None:
    root, _evaluation, sidecar = _fixture(tmp_path)
    sidecar_path = root / "evidence/run/g2.3-corrected-delta-sci" / f"{sidecar['artifact_hash']}.json"
    tampered = deepcopy(sidecar)
    tampered["delta_sci_by_endpoint"]["model_total"]["32"] = 0.021  # type: ignore[index]
    write_canonical_json(sidecar_path, tampered)
    with pytest.raises(CorrectedDeltaRejected, match="BOUND_HASH_MISMATCH|HASH_MISMATCH"):
        load_bound_corrected_delta(
            root,
            g23_evaluation_ref="evidence/run/g2.3-attempts/eval/evaluation.json",
            cell_id=TARGET,
            expected_config_hash=CONFIG_HASH,
            expected_result_hash=RESULT_HASH,
        )

    restored = dict(sidecar)
    restored["unknown"] = True
    restored["artifact_hash"] = canonical_json_hash({key: value for key, value in restored.items() if key != "artifact_hash"})
    write_canonical_json(sidecar_path, restored)
    with pytest.raises(CorrectedDeltaRejected, match="BOUND_HASH_MISMATCH|UNKNOWN_OR_MISSING_FIELDS"):
        load_bound_corrected_delta(
            root,
            g23_evaluation_ref="evidence/run/g2.3-attempts/eval/evaluation.json",
            cell_id=TARGET,
            expected_config_hash=CONFIG_HASH,
            expected_result_hash=RESULT_HASH,
        )


def test_cell_result_identity_and_old_sizing_table_cannot_be_substituted(tmp_path: Path) -> None:
    root, evaluation, sidecar = _fixture(tmp_path)
    mutated = deepcopy(evaluation)
    mutated["cells"][0]["identities"]["result_hash"] = "f" * 64  # type: ignore[index]
    mutated["artifact_hash"] = canonical_json_hash({key: value for key, value in mutated.items() if key != "artifact_hash"})
    write_canonical_json(root / "evidence/run/g2.3-attempts/eval/evaluation.json", mutated)
    with pytest.raises(CorrectedDeltaRejected, match="IDENTITY_MISMATCH"):
        load_bound_corrected_delta(
            root,
            g23_evaluation_ref="evidence/run/g2.3-attempts/eval/evaluation.json",
            cell_id=TARGET,
            expected_config_hash=CONFIG_HASH,
            expected_result_hash=RESULT_HASH,
        )

    # A correctly hashed old producer table is not an admissible S2.6 input;
    # the loader requires the evaluator binding and corrected sidecar schema.
    old_ref = "evidence/run/producer/old-table.json"
    write_canonical_json(root / old_ref, {"delta_sci_by_endpoint": {"model_total": {"131072": 1.0}}})
    assert sidecar["source_producer_ref"] != old_ref


def _pilot_mapping():
    sampling = SamplingPlan(
        SamplingUniverse("s206-delta-test", tuple(range(1024))),
        {"reference_sizing": 1, "reference_A": 2, "reference_B": 3, "pilot": 4, "confirmatory": 5},
    )
    return build_global_pilot_mapping(sampling)


def _pilot_rows():
    from tests.test_stage2_s206_formal_preparation import _costs, _measurements

    return _pilot_mapping(), _measurements(), _costs()


def test_six_cell_bindings_are_canonical_and_not_broadcast() -> None:
    mapping, measurements, costs = _pilot_rows()
    report = reduce_blinded_pilot(mapping, measurements, cost_semantics=costs)
    bindings = report.to_dict()["corrected_delta_sci_bindings"]
    assert [item["cell_id"] for item in bindings] == [item.replace(".", ":", 1) for item in ANCHOR_IDS]
    assert len({item["corrected_delta_sci_hash"] for item in bindings}) == 6
    assert len({item["corrected_delta_sci_ref"] for item in bindings}) == 6
    assert len({item["config_hash"] for item in bindings}) == 6
    assert len({item["result_hash"] for item in bindings}) == 6


def test_six_cell_binding_swap_duplicate_and_missing_fail_closed() -> None:
    mapping, measurements, costs = _pilot_rows()
    swapped = list(measurements)
    swapped[0] = replace(
        swapped[0],
        corrected_delta_sci_hash=swapped[20].corrected_delta_sci_hash,
        corrected_delta_sci_ref=swapped[20].corrected_delta_sci_ref,
        corrected_delta_sci_config_hash=swapped[20].corrected_delta_sci_config_hash,
        corrected_delta_sci_result_hash=swapped[20].corrected_delta_sci_result_hash,
    )
    with pytest.raises(S206PreparationBlocked, match="[Bb]INDING|identity"):
        reduce_blinded_pilot(mapping, swapped, cost_semantics=costs)

    report = reduce_blinded_pilot(mapping, measurements, cost_semantics=costs)
    duplicate = list(report.corrected_delta_sci_bindings)
    duplicate[1] = dict(duplicate[0])
    with pytest.raises(ValueError, match="binding|identity"):
        replace(report, corrected_delta_sci_bindings=tuple(duplicate))
    missing = duplicate[:5]
    with pytest.raises(ValueError, match="count"):
        replace(report, corrected_delta_sci_bindings=tuple(missing))
