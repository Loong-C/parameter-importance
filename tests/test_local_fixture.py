"""本机 Stage 0--9 fixture 的重现性与 formal 隔离测试。"""

from __future__ import annotations

from pathlib import Path

import pytest

from param_importance_nlp.contracts.jsonio import canonical_json_hash, load_canonical_json
from param_importance_nlp.experiments.sampling import FormalDecisionBlocked
from param_importance_nlp.experiments.stage2 import EstimatorDecision
from param_importance_nlp.local_fixture import LocalFixtureContractError, run_local_fixture


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_CONFIG = REPOSITORY_ROOT / "configs" / "local-fixtures" / "resolved-config-v1.json"


def test_local_fixture_is_byte_reproducible_and_never_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个独立目录必须得到逐字节相同产物，fixture decision 不能进 formal。"""

    monkeypatch.chdir(tmp_path)
    fixture_root = tmp_path / "artifacts" / "local-fixture"
    first_dir = fixture_root / "run-a"
    second_dir = fixture_root / "run-b"
    first = run_local_fixture(config_path=FIXTURE_CONFIG, output_dir=first_dir)
    second = run_local_fixture(config_path=FIXTURE_CONFIG, output_dir=second_dir)

    stable_summary_fields = {
        "config_hash",
        "seed_plan_hash",
        "registry_hash",
        "result_hash",
        "artifact_hash",
        "report_hash",
        "artifact_file_hash",
        "report_file_hash",
        "markdown_file_hash",
    }
    assert {key: first[key] for key in stable_summary_fields} == {
        key: second[key] for key in stable_summary_fields
    }
    for filename in (
        "local-fixture-result.json",
        "analysis-report.json",
        "local-fixture-report.md",
    ):
        assert (first_dir / filename).read_bytes() == (second_dir / filename).read_bytes()

    artifact = load_canonical_json(first_dir / "local-fixture-result.json")
    assert isinstance(artifact, dict)
    assert artifact["scope"] == "local_fixture"
    assert artifact["run_intent"] == "local_fixture"
    assert artifact["formal_eligible"] is False
    assert artifact["local_validation"]["local_validation_status"] == "PASS"
    assert artifact["formal_readiness"]["gate_status"] == "BLOCKED"
    assert artifact["formal_readiness"]["reason"] == "server_unreachable"
    assert artifact["stage2"]["formal_state"] == "UNFROZEN"
    assert artifact["stage3"]["formal_state"] == "UNFROZEN"
    assert artifact["stage2"]["estimator_decision"]["scope"] == "local_fixture"
    assert artifact["stage3"]["quadrature_decision"]["formal_eligible"] is False

    payload = dict(artifact)
    stored_hash = payload.pop("artifact_hash")
    assert canonical_json_hash(payload) == stored_hash == first["artifact_hash"]
    assert str(first_dir.resolve()) not in (first_dir / "local-fixture-result.json").read_text(
        encoding="utf-8"
    )

    # ``require_formal`` 是 Stage 4 在线训练消费 Stage 2 decision 前的硬门。
    # 从 fixture 结果完整重载后仍必须拒绝，证明不能靠重新序列化绕过 scope。
    fixture_decision = EstimatorDecision.from_mapping(
        artifact["stage2"]["estimator_decision"]
    )
    assert fixture_decision.formal_eligible is False
    with pytest.raises(FormalDecisionBlocked, match="正式训练必须读取"):
        fixture_decision.require_formal()


def test_local_fixture_refuses_formal_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使配置正确，也不能把 fixture 直接发布进命名明确的正式结果区。"""

    monkeypatch.chdir(tmp_path)
    with pytest.raises(
        LocalFixtureContractError,
        match="LOCAL_FIXTURE_FORMAL_OUTPUT_FORBIDDEN",
    ):
        run_local_fixture(
            config_path=FIXTURE_CONFIG,
            output_dir=tmp_path
            / "artifacts"
            / "local-fixture"
            / "formal-v1"
            / "attempt-1",
        )


def test_local_fixture_refuses_output_outside_configured_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任意本机目录也不构成授权，唯一 allowlist 来自 runtime.output_root。"""

    monkeypatch.chdir(tmp_path)
    with pytest.raises(LocalFixtureContractError, match="LOCAL_FIXTURE_OUTPUT_OUTSIDE_ROOT"):
        run_local_fixture(
            config_path=FIXTURE_CONFIG,
            output_dir=tmp_path / "other-local-directory",
        )


def test_local_fixture_refuses_symlink_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """根内路径若通过 symlink 指向根外，解析后必须被拒绝。"""

    monkeypatch.chdir(tmp_path)
    fixture_root = tmp_path / "artifacts" / "local-fixture"
    outside = tmp_path / "outside"
    fixture_root.mkdir(parents=True)
    outside.mkdir()
    escape_link = fixture_root / "escape-link"
    try:
        escape_link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"当前 Windows 权限不允许创建目录 symlink：{error}")

    with pytest.raises(LocalFixtureContractError, match="LOCAL_FIXTURE_SYMLINK_ESCAPE"):
        run_local_fixture(
            config_path=FIXTURE_CONFIG,
            output_dir=escape_link / "attempt-1",
        )
