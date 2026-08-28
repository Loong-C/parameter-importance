from __future__ import annotations

from pathlib import Path

from param_importance_nlp.contracts import canonical_json_hash, load_canonical_json


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports" / "stage3" / "g3-0-prerequisite-audit.json"


def test_g30_audit_is_hash_bound_and_fail_closed() -> None:
    value = load_canonical_json(AUDIT)
    assert isinstance(value, dict)
    declared = value["artifact_hash"]
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    assert canonical_json_hash(body) == declared
    assert value["gate_id"] == "stage3.G3-0"
    assert value["status"] == "BLOCKED"
    assert value["formal_eligible"] is False
    assert value["stage2_delivery"]["formal_g27_g28_found"] is False
    assert value["stage2_delivery"]["s27"]["formal_eligible"] is False
    assert value["stage2_delivery"]["s28"]["formal_eligible"] is False
    assert value["stage2_delivery"]["s211"]["formal_eligible"] is False
    assert value["gate_effect"]["formal_stage3_server_experiments_authorized"] is False
    assert "Do not relabel direct-unvalidated artifacts as formal." in value[
        "prohibited_shortcuts"
    ]
