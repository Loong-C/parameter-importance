"""Stage 0 gate aggregation must fail closed and preserve exception semantics."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from param_importance_nlp.stage0_gate import (
    Stage0CheckClass,
    Stage0CheckStatus,
    Stage0EvidenceRef,
    Stage0ExceptionApproval,
    Stage0GateCheck,
    Stage0GateEvidenceError,
    Stage0GateReport,
    Stage0GateStatus,
)


ROOT = Path(__file__).resolve().parents[1]


def _check(
    status: Stage0CheckStatus = Stage0CheckStatus.PASS,
    *,
    check_class: Stage0CheckClass = Stage0CheckClass.CORRECTNESS,
    exception_eligible: bool = False,
    approval: Stage0ExceptionApproval | None = None,
) -> Stage0GateCheck:
    return Stage0GateCheck(
        check_id="stage0.g4.config",
        check_class=check_class,
        status=status,
        summary="strict configuration contract",
        exception_eligible=exception_eligible,
        measurements={"assertions": 12},
        evidence_refs=("reports/stage0/g4-config.json",),
        approval=approval,
    )


def _report(*checks: Stage0GateCheck) -> Stage0GateReport:
    return Stage0GateReport(
        gate_id="stage0.G4",
        generated_at="2026-08-03T16:00:00Z",
        generator_git_commit="a" * 40,
        environment_id="server-cuda-formal-v1",
        checks=checks,
        input_evidence=(
            Stage0EvidenceRef(
                "reports/stage0/g3-resolution.json",
                "b" * 64,
                "stage0-g3-resolution-audit-v1",
            ),
        ),
        config_hashes={"single-gpu-fp32": "c" * 64},
    )


def test_pass_report_roundtrips_and_hash_is_bound() -> None:
    report = _report(_check())
    assert report.status is Stage0GateStatus.PASS
    restored = Stage0GateReport.from_mapping(report.to_dict())
    assert restored == report
    assert restored.artifact_hash == report.artifact_hash

    tampered = report.to_dict()
    tampered["checks"][0]["measurements"]["assertions"] = 11  # type: ignore[index]
    with pytest.raises(Stage0GateEvidenceError, match="artifact_hash"):
        Stage0GateReport.from_mapping(tampered)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (Stage0CheckStatus.FAIL, Stage0GateStatus.FAIL),
        (Stage0CheckStatus.BLOCKED, Stage0GateStatus.BLOCKED),
    ],
)
def test_failure_and_blocker_cannot_be_reported_as_pass(
    status: Stage0CheckStatus, expected: Stage0GateStatus
) -> None:
    report = _report(_check(status))
    assert report.status is expected
    with pytest.raises(Stage0GateEvidenceError, match="disagrees"):
        replace(report, status=Stage0GateStatus.PASS)


def test_only_approved_performance_or_capacity_exception_is_conditional() -> None:
    approval = Stage0ExceptionApproval(
        approval_ref="reports/stage0/approvals/g8-throughput.json",
        approval_sha256="d" * 64,
        approved_by="project-owner",
        approved_at="2026-08-03T16:00:00Z",
        expires_at="2026-08-18T15:59:00Z",
        scope="G8 throughput threshold only",
    )
    report = _report(
        _check(
            Stage0CheckStatus.APPROVED_EXCEPTION,
            check_class=Stage0CheckClass.PERFORMANCE,
            exception_eligible=True,
            approval=approval,
        )
    )
    assert report.status is Stage0GateStatus.CONDITIONALLY_ACCEPTED

    with pytest.raises(Stage0GateEvidenceError, match="only performance/capacity"):
        _check(
            check_class=Stage0CheckClass.CORRECTNESS,
            exception_eligible=True,
        )
    with pytest.raises(Stage0GateEvidenceError, match="requires eligibility"):
        _check(
            Stage0CheckStatus.APPROVED_EXCEPTION,
            check_class=Stage0CheckClass.PERFORMANCE,
            exception_eligible=True,
        )


def test_schema_is_strict_and_matches_wire_enums() -> None:
    schema = json.loads(
        (ROOT / "schemas/stage0-gate-report-v1.json").read_text(encoding="utf-8")
    )
    assert schema["additionalProperties"] is False
    assert schema["properties"]["schema_version"] == {
        "const": "stage0-gate-report-v1"
    }
    assert schema["$defs"]["check"]["additionalProperties"] is False
    assert set(schema["properties"]["status"]["enum"]) == {
        item.value for item in Stage0GateStatus
    }
