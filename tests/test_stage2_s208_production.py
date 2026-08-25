from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, write_canonical_json
from param_importance_nlp.experiments.stage2_formal import _vector_digest
from param_importance_nlp.experiments.stage2_s208_production import (
    S208ProductionBlocked,
    load_s208_reference_bundle,
)
from param_importance_nlp.runtime.tensor_bundle import publish_tensor_bundle


CELLS = (
    "pythia-14m:initialization", "pythia-14m:early", "pythia-14m:mid_late",
    "pythia-31m-deduped:initialization", "pythia-31m-deduped:early", "pythia-31m-deduped:mid_late",
)


def _hashed(body: dict[str, object]) -> dict[str, object]:
    body["artifact_hash"] = canonical_json_hash({key: value for key, value in body.items() if key != "artifact_hash"})
    return body


def _fixture(root: Path, *, bounded: bool = False) -> tuple[Path, Path]:
    refs = root / "refs"
    bundles = root / "bundles"
    refs.mkdir()
    bundles.mkdir()
    rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    for index, cell in enumerate(CELLS):
        values = {"a": np.asarray([1.0 + index, 2.0 + index, 3.0 + index], dtype=np.float64)}
        uncertainty_vectors = {
            name: {"a": np.zeros(3, dtype=np.float64)}
            for name in ("bias_variance", "cross_variance", "ranking_variance")
        }
        uncertainty_metadata = _hashed({
            "schema_version": "stage2-reference-uncertainty-v1",
            "estimator": "block_u_delete_one_jackknife",
            "confidence_level": 0.95,
            "block_count_a": 3,
            "block_count_b": 3,
            "bias_variance_hash": _vector_digest(uncertainty_vectors["bias_variance"]),
            "cross_variance_hash": _vector_digest(uncertainty_vectors["cross_variance"]),
            "ranking_variance_hash": _vector_digest(uncertainty_vectors["ranking_variance"]),
            "trace_bias_variance": 0.0,
            "bias_half_width_l2": 0.0,
        })
        sequence_variance = {"a": np.asarray([0.5, 0.5, 0.5], dtype=np.float64)}
        state: dict[str, object] = {
            "coordinate_ids": ["a[0]", "a[1]", "a[2]"],
            "bias_reference": values,
            "cross_reference": values,
            "ranking_reference": values,
        }
        if bounded:
            state.update({"uncertainty": uncertainty_vectors, "sequence_variance": sequence_variance})
        else:
            state["reference_blocks"] = {
                name: [values, values, values]
                for name in ("bias", "cross", "ranking")
            }
        bundle_path = bundles / f"cell-{index}"
        bundle = publish_tensor_bundle(bundle_path, state)
        candidate_body: dict[str, object] = {
            "schema_version": "reference-result-v1",
            "reference_id": f"ref-{index}",
            "bias_reference_hash": _vector_digest(values),
            "cross_reference_hash": _vector_digest(values),
            "ranking_reference_hash": _vector_digest(values),
            "tensor_bundle_ref": f"bundles/cell-{index}",
            "tensor_bundle_manifest_hash": bundle.manifest_sha256,
            "scope": "formal",
            "formal_eligible": False,
        }
        if bounded:
            candidate_body["metadata"] = {
                "uncertainty": uncertainty_metadata,
                "sequence_variance_hash": _vector_digest(sequence_variance),
            }
        candidate = _hashed(candidate_body)
        ref_path = refs / f"cell-{index}.json"
        write_canonical_json(ref_path, candidate)
        rows.append({"cell_id": cell, "reference_ref": ref_path.relative_to(root).as_posix(), "reference_hash": candidate["artifact_hash"]})
        gate_rows.append({"cell_id": cell, "status": "PASS", "formal_eligible": True})
    bundle_manifest = _hashed({
        "schema_version": "stage2-s208-reference-bundle-v1",
        "scope": "formal",
        "formal_eligible": True,
        "cells": rows,
    })
    bundle_ref = root / "reference-bundle.json"
    write_canonical_json(bundle_ref, bundle_manifest)
    gate = _hashed({
        "schema_version": "stage2-g23-reference-evaluation-v1",
        "gate_id": "stage2.G2.3",
        "status": "PASS",
        "formal_eligible": True,
        "cells": gate_rows,
    })
    gate_ref = root / "g23.json"
    write_canonical_json(gate_ref, gate)
    return bundle_ref, gate_ref


def test_loader_requires_paths_and_independent_g23(tmp_path: Path) -> None:
    bundle, gate = _fixture(tmp_path)
    loaded = load_s208_reference_bundle(tmp_path, bundle.relative_to(tmp_path), gate.relative_to(tmp_path), memmap_root=tmp_path / "memmap")
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    assert loaded["lineage"]["g23_gate_hash"] == gate_payload["artifact_hash"]
    assert loaded["cells"]
    assert all(isinstance(item["vectors"]["bias"], np.memmap) for item in loaded["cells"].values())
    with pytest.raises(S208ProductionBlocked, match="PATH_INPUTS_REQUIRED"):
        load_s208_reference_bundle(tmp_path, {}, gate.relative_to(tmp_path))  # type: ignore[arg-type]


def test_loader_rejects_candidate_self_qualification(tmp_path: Path) -> None:
    bundle, gate = _fixture(tmp_path)
    raw = dict(json.loads((tmp_path / "refs" / "cell-0.json").read_text(encoding="utf-8")))
    raw["formal_eligible"] = True
    raw["artifact_hash"] = canonical_json_hash({key: value for key, value in raw.items() if key != "artifact_hash"})
    write_canonical_json(tmp_path / "refs" / "cell-0.json", raw)
    with pytest.raises(S208ProductionBlocked, match="CANDIDATE_MUST_NOT_SELF_QUALIFY"):
        load_s208_reference_bundle(tmp_path, bundle.relative_to(tmp_path), gate.relative_to(tmp_path), memmap_root=tmp_path / "memmap")


def test_loader_accepts_hash_bound_bounded_uncertainty_and_sequence_variance(tmp_path: Path) -> None:
    bundle, gate = _fixture(tmp_path, bounded=True)
    loaded = load_s208_reference_bundle(
        tmp_path,
        bundle.relative_to(tmp_path),
        gate.relative_to(tmp_path),
        memmap_root=tmp_path / "memmap",
    )
    for reference in loaded["cells"].values():
        assert reference["reference_uncertainty_mode"] == "independent_reference_variance_combination"
        assert set(reference["reference_variances"]) == {"bias", "cross", "ranking"}
        assert isinstance(reference["sequence_variance"], np.memmap)
        assert "reference_blocks" not in reference


@pytest.mark.parametrize("field", ["bias_variance", "sequence_variance"])
def test_loader_rejects_tampered_bounded_vectors(tmp_path: Path, field: str) -> None:
    bundle, gate = _fixture(tmp_path, bounded=True)
    bundle_path = tmp_path / "bundles" / "cell-0"
    state, _identity = __import__(
        "param_importance_nlp.runtime.tensor_bundle", fromlist=["load_tensor_bundle"]
    ).load_tensor_bundle(bundle_path)
    if field == "sequence_variance":
        state["sequence_variance"]["a"][0] += 1.0
    else:
        state["uncertainty"][field]["a"][0] += 1.0
    # Publish the changed bytes under a fresh bundle identity but keep the
    # candidate's authoritative vector hashes unchanged.
    import shutil

    shutil.rmtree(bundle_path)
    changed = publish_tensor_bundle(bundle_path, state)
    candidate_path = tmp_path / "refs" / "cell-0.json"
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    candidate["tensor_bundle_manifest_hash"] = changed.manifest_sha256
    candidate["artifact_hash"] = canonical_json_hash({key: value for key, value in candidate.items() if key != "artifact_hash"})
    write_canonical_json(candidate_path, candidate)
    bundle_manifest = json.loads(bundle.read_text(encoding="utf-8"))
    bundle_manifest["cells"][0]["reference_hash"] = candidate["artifact_hash"]
    bundle_manifest["artifact_hash"] = canonical_json_hash({key: value for key, value in bundle_manifest.items() if key != "artifact_hash"})
    write_canonical_json(bundle, bundle_manifest)
    with pytest.raises(S208ProductionBlocked, match="VARIANCE_HASH_MISMATCH"):
        load_s208_reference_bundle(
            tmp_path,
            bundle.relative_to(tmp_path),
            gate.relative_to(tmp_path),
            memmap_root=tmp_path / "memmap",
        )
