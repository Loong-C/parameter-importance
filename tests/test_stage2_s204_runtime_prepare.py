from __future__ import annotations

from pathlib import Path

import pytest

from ops.stage2.materialize_s204 import (
    EXPECTED_CELL_IDS,
    S204MaterializationError,
    _validate_delta_sci_artifact,
    publish_per_cell_delta_sci_plans,
)
from ops.stage2.prepare_s204_formal import _validate_adapter_gate
from param_importance_nlp.contracts import load_canonical_json
from param_importance_nlp.runtime import TaskArtifactStore


def _s21_formula_refs(root: Path) -> dict[str, str]:
    refs: dict[str, str] = {}
    for kind in ("preregistration", "hypothesis_contract"):
        payload = {
            "schema_version": f"stage2-{kind.replace('_', '-')}-v1",
            "scope": "formal",
            "equivalence_and_precision": {
                "scientific_margin_formula": "max(0.10*Delta_c_e(B),0.01*S_c_e)",
                "absolute_floors": {"tau_model": 1e-12, "tau_layer": 1e-12, "tau_module": 1e-12},
            },
        }
        refs[kind] = TaskArtifactStore(root, f"s21/{kind}").publish(
            task_id="stage2.01_scope_hypotheses_and_preregistration",
            artifact_kind=kind,
            config_hash="a" * 64,
            run_intent="formal",
            payload=payload,
            formal_eligible=True,
        ).commit_ref
    return refs


def test_cell_id_and_path_projection_are_reversible() -> None:
    from param_importance_nlp.experiments.stage2_s204_ids import (
        canonical_cell_id,
        cell_id_from_path_component,
        cell_path_component,
    )

    assert canonical_cell_id("pythia-14m", "early") == "pythia-14m:early"
    assert cell_path_component("pythia-14m:early") == "pythia-14m__early"
    assert cell_id_from_path_component("pythia-14m__early") == "pythia-14m:early"
    with pytest.raises(ValueError):
        cell_id_from_path_component("pythia-14m-early")


def test_pre_sizing_delta_plan_has_no_numeric_margin(tmp_path: Path) -> None:
    refs = publish_per_cell_delta_sci_plans(tmp_path, s21_refs=_s21_formula_refs(tmp_path))
    assert tuple(refs) == EXPECTED_CELL_IDS
    payload = load_canonical_json(tmp_path / refs[EXPECTED_CELL_IDS[0]])
    assert payload["schema_version"] == "stage2-reference-delta-sci-plan-v1"
    assert "delta_sci_by_B" not in payload
    with pytest.raises(S204MaterializationError, match="REFERENCE_DELTA_SCI_NUMERIC_REQUIRED"):
        _validate_delta_sci_artifact(
            tmp_path,
            refs[EXPECTED_CELL_IDS[0]],
            cell_id=EXPECTED_CELL_IDS[0],
        )


def test_missing_formal_g22_commit_blocks_prepare(tmp_path: Path) -> None:
    with pytest.raises(S204MaterializationError, match="S204_FORMAL_ADAPTER_COMMIT_REQUIRED"):
        _validate_adapter_gate(
            tmp_path,
            gate_id="stage2.G2.2",
            gate_ref="evidence/stage2/s204/formal-adapters/g2-2-r7/commits/gate_record.json",
            expected_refs=("evidence/stage2/s204/asset-resolution.json",),
        )
