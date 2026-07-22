from __future__ import annotations

import hashlib
from pathlib import Path

from ops.local_contracts import build_local_contract_freeze_set, main
from param_importance_nlp.contracts import (
    ContractFreeze,
    canonical_json_hash,
    load_canonical_json,
)


REPOSITORY = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/local-fixtures/resolved-config-v1.json")
FROZEN_AT = "2026-07-22T00:00:00Z"


def test_local_contract_freezes_are_deterministic_and_never_formal() -> None:
    first = build_local_contract_freeze_set(
        repository=REPOSITORY,
        config_path=CONFIG,
        frozen_at=FROZEN_AT,
    )
    second = build_local_contract_freeze_set(
        repository=REPOSITORY,
        config_path=CONFIG,
        frozen_at=FROZEN_AT,
    )

    assert first == second
    assert first["scope"] == "local_fixture"
    assert first["formal_eligible"] is False
    assert len(first["freezes"]) == 10
    freezes = [ContractFreeze.from_mapping(item) for item in first["freezes"]]
    assert [freeze.stage for freeze in freezes] == list(range(10))
    assert all(not freeze.formal_eligible for freeze in freezes)

    payload = dict(first)
    artifact_hash = payload.pop("artifact_hash")
    assert artifact_hash == canonical_json_hash(payload)
    stage1 = freezes[1]
    expected_math_hash = hashlib.sha256(
        (REPOSITORY / "docs/mathematics.md").read_bytes()
    ).hexdigest()
    assert stage1.source_hashes["docs/mathematics.md"] == expected_math_hash


def test_local_contract_freeze_cli_writes_canonical_artifact(tmp_path: Path) -> None:
    output = tmp_path / "freezes.json"
    assert (
        main(
            [
                "--repository",
                str(REPOSITORY),
                "--config",
                str(CONFIG),
                "--output",
                str(output),
                "--frozen-at",
                FROZEN_AT,
            ]
        )
        == 0
    )
    loaded = load_canonical_json(output)
    expected = build_local_contract_freeze_set(
        repository=REPOSITORY,
        config_path=CONFIG,
        frozen_at=FROZEN_AT,
    )
    assert loaded == expected
