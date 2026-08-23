from pathlib import Path

import pytest

from param_importance_nlp.contracts.stage2_authorization import (
    AUTHORIZATION_REF,
    EXCLUDED_PCI,
    USER_AUTHORIZATION,
    Stage2AuthorizationError,
    load_stage2_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_authorization_amendment_is_scoped_and_hash_bound() -> None:
    value = load_stage2_authorization(ROOT)
    assert value["user_authorization_original"] == USER_AUTHORIZATION
    assert value["single_copy_accepted"] is True
    assert value["excluded_pci_bus_ids"] == [EXCLUDED_PCI]
    assert value["scope"] == ["reproducible_stage0_artifacts", "reproducible_stage2_artifacts"]
    assert "non_reproducible_human_evidence" in value["excluded_non_reproducible_evidence"]


def test_formal_authorization_rejects_tampered_gpu_exclusion(tmp_path: Path) -> None:
    target = tmp_path / AUTHORIZATION_REF
    target.parent.mkdir(parents=True)
    target.write_text((ROOT / AUTHORIZATION_REF).read_text(encoding="utf-8"), encoding="utf-8")
    text = target.read_text(encoding="utf-8").replace("0000:50:00.0", "0000:53:00.0")
    target.write_text(text, encoding="utf-8")
    with pytest.raises(Stage2AuthorizationError):
        load_stage2_authorization(tmp_path)
