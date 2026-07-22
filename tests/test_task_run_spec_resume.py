from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from param_importance_nlp.contracts import load_canonical_json
from param_importance_nlp.contracts.config_v2 import load_resolved_config_compatible
from param_importance_nlp.runtime import TaskRunSpec


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/local-fixtures/resolved-config-v1.json"


def _config(*, resume_ref: str | None):
    value = load_canonical_json(BASE)
    assert isinstance(value, dict)
    return load_resolved_config_compatible(
        deepcopy(value),
        task_id="stage0.06_single_gpu_smoke",
        overrides={
            "training": {"max_steps": 1},
            "recovery": {"resume_ref": resume_ref},
        },
    )


def test_task_run_spec_roundtrip_freezes_explicit_resume_reference() -> None:
    fresh = TaskRunSpec.from_config(_config(resume_ref=None))
    assert fresh.resume_ref is None
    assert TaskRunSpec.from_mapping(fresh.to_dict()) == fresh

    resumed = TaskRunSpec.from_config(
        _config(resume_ref="runs/stage0/checkpoints/rank-0000")
    )
    assert resumed.resume_ref == "runs/stage0/checkpoints/rank-0000"
    assert TaskRunSpec.from_mapping(resumed.to_dict()) == resumed
    assert resumed.to_dict()["spec_hash"] != fresh.to_dict()["spec_hash"]
