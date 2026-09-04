#!/usr/bin/env python3
"""Read-only audit of the formal S3.07 -> S3.08 handoff.

The long-running S3.07 fan-out may have been launched by a historical runner
that also published a candidate ``gate_record``.  G3-6 is an independent
post-S3.08 decision, so this command deliberately selects only
``formal_path_results`` and ``completeness_report`` as S3.08 predecessors.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    _repository_root = Path(__file__).resolve().parents[2]
    for _candidate in (_repository_root, _repository_root / "src"):
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))

from ops.stage3.run_stage3_fanout import FanoutRunner
from ops.stage3.run_stage3_formal import Stage3OrchestratorError, _load_json, _resolve_ref
from param_importance_nlp.contracts import FormalExecutionEvidence, ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import canonical_json_bytes, canonical_json_hash
from param_importance_nlp.contracts.task_catalog import DEFAULT_TASK_CATALOG, TaskCatalog
from param_importance_nlp.experiments.stage3_g36_publisher import _load_streaming_coverage
from param_importance_nlp.runtime import TaskRuntimeEnvironment
from param_importance_nlp.runtime.task_artifacts import (
    LoadedTaskArtifact,
    load_committed_task_artifact,
)


SCHEMA_VERSION = "stage3-s307-handoff-audit-v1"
S307_TASK = "stage3.07_formal_experiment_matrix"
S308_TASK = "stage3.08_error_analysis_and_stability"
S308_OUTPUT_KINDS = ("formal_path_results", "completeness_report")
HISTORICAL_NON_AUTHORITY_KINDS = frozenset({"gate_record"})


class Stage3HandoffError(ValueError):
    """Raised when the durable handoff is incomplete or inconsistent."""


def _fail(code: str, detail: object | None = None) -> Stage3HandoffError:
    suffix = "" if detail is None else f":{detail}"
    return Stage3HandoffError(f"{code}{suffix}")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail("S307_HANDOFF_MAPPING_REQUIRED", field)
    return value


def _authoritative_s308_refs(result: Mapping[str, object]) -> tuple[dict[str, str], dict[str, str]]:
    """Separate the two scientific outputs from an old candidate gate."""

    raw_refs = _mapping(result.get("artifact_refs"), field="artifact_refs")
    missing = sorted(set(S308_OUTPUT_KINDS) - set(raw_refs))
    extras = sorted(set(raw_refs) - set(S308_OUTPUT_KINDS))
    forbidden = sorted(set(extras) - HISTORICAL_NON_AUTHORITY_KINDS)
    if missing:
        raise _fail("S307_HANDOFF_OUTPUT_MISSING", ",".join(missing))
    if forbidden:
        raise _fail("S307_HANDOFF_OUTPUT_UNEXPECTED", ",".join(forbidden))

    authoritative: dict[str, str] = {}
    quarantined: dict[str, str] = {}
    for kind, target in ((name, authoritative) for name in S308_OUTPUT_KINDS):
        value = raw_refs.get(kind)
        if not isinstance(value, str) or not value:
            raise _fail("S307_HANDOFF_OUTPUT_REF_INVALID", kind)
        target[kind] = value
    for kind in extras:
        value = raw_refs.get(kind)
        if not isinstance(value, str) or not value:
            raise _fail("S307_HANDOFF_OUTPUT_REF_INVALID", kind)
        quarantined[kind] = value
    return authoritative, quarantined


def _effective_step(step: Mapping[str, Any], attempt: Mapping[str, Any]) -> dict[str, Any]:
    effective = dict(step)
    action = attempt.get("action")
    if action == step.get("action"):
        return effective
    if step.get("action") == "run" and action == "resume":
        effective["action"] = "resume"
        effective["config_ref"] = step["retry_config_ref"]
        effective["config_hash"] = step["retry_config_hash"]
        return effective
    raise _fail("S307_HANDOFF_ATTEMPT_ACTION_INVALID", step.get("step_id"))


def _historical_s307_catalog() -> TaskCatalog:
    """Reconstruct only the output declaration used by the live old runner."""

    current = DEFAULT_TASK_CATALOG.get(S307_TASK)
    historical = replace(
        current,
        artifact_kinds=(*current.artifact_kinds, "gate_record"),
        output_artifacts=(
            *current.output_artifacts,
            replace(current.output_artifacts[-1], artifact_kind="gate_record"),
        ),
    )
    return TaskCatalog(
        tuple(
            historical if task.task_id == S307_TASK else task
            for task in DEFAULT_TASK_CATALOG.tasks
        )
    )


def _audit_completed_steps(runner: FanoutRunner) -> tuple[list[Mapping[str, Any]], bool]:
    cursor = int(runner.state["next_step"])
    attempts = runner.state.get("attempts")
    if not isinstance(attempts, list):
        raise _fail("S307_HANDOFF_ATTEMPTS_INVALID")
    successes = [
        item
        for item in attempts
        if isinstance(item, Mapping)
        and item.get("status") in {"PASS", "BLOCKED"}
        and isinstance(item.get("result_hash"), str)
    ]
    expected_ids = [str(step["step_id"]) for step in runner.steps[:cursor]]
    observed_ids = [str(item.get("step_id")) for item in successes]
    if observed_ids != expected_ids:
        raise _fail("S307_HANDOFF_SUCCESS_SEQUENCE_INVALID")

    audited: list[Mapping[str, Any]] = []
    for index, step in enumerate(runner.steps):
        phases = runner.state["unit_phases"].get(step["unit_id"])
        expected_phases = sorted(step["completes_phases"]) if index < cursor else []
        if phases != expected_phases:
            raise _fail("S307_HANDOFF_PHASE_STATE_INVALID", step["step_id"])
        if index >= cursor:
            continue
        result_path = _resolve_ref(
            step["result_ref"],
            roots=(runner.data_root,),
            field=f"{step['step_id']}.result_ref",
        )
        if not result_path.is_file():
            raise _fail("S307_HANDOFF_COMPLETED_RESULT_MISSING", step["step_id"])
        effective = _effective_step(step, successes[index])
        parsed = runner._verify_result(effective, result_path)
        if parsed.get("result_hash") != successes[index].get("result_hash"):
            raise _fail("S307_HANDOFF_RESULT_HASH_DRIFT", step["step_id"])
        audited.append(parsed)

    inflight_result = False
    if cursor < len(runner.steps):
        candidate = _resolve_ref(
            runner.steps[cursor]["result_ref"],
            roots=(runner.data_root,),
            field=f"{runner.steps[cursor]['step_id']}.result_ref",
        )
        inflight_result = candidate.is_file()
    return audited, inflight_result


def _load_output(
    root: Path,
    reference: str,
    *,
    kind: str,
    config_hash: str,
) -> LoadedTaskArtifact:
    try:
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("S307_HANDOFF_COMMIT_INVALID", f"{kind}:{error}") from error
    if (
        loaded.identity.task_id != S307_TASK
        or loaded.identity.artifact_kind != kind
        or loaded.identity.config_hash != config_hash
    ):
        raise _fail("S307_HANDOFF_COMMIT_IDENTITY_INVALID", kind)
    return loaded


def _load_controls(arguments: argparse.Namespace) -> Mapping[str, Any]:
    s307_environment = TaskRuntimeEnvironment.from_mapping(_load_json(arguments.s307_environment))
    s308_environment = TaskRuntimeEnvironment.from_mapping(_load_json(arguments.s308_environment))
    s308_config = ResolvedConfigV2.from_mapping(_load_json(arguments.s308_config))
    if s308_config.task_id != S308_TASK or s308_config.run_intent != "formal":
        raise _fail("S307_HANDOFF_S308_CONFIG_INVALID")
    orchestration = _mapping(
        s308_config.section("orchestration"), field="s308.orchestration"
    )
    input_refs = orchestration.get("input_result_refs")
    if (
        not isinstance(input_refs, list)
        or len(input_refs) != 2
        or any(not isinstance(item, str) or not item for item in input_refs)
        or tuple(Path(item).name for item in input_refs)
        != ("formal_path_results.json", "completeness_report.json")
    ):
        raise _fail("S307_HANDOFF_S308_INPUT_SET_INVALID")

    raw_plan_ref = s307_environment.evidence_refs.get("formal_stage3_matrix_plan")
    promoted_plan_ref = s308_environment.evidence_refs.get("formal_stage3_matrix_plan")
    base_execution_ref = s307_environment.evidence_refs.get("formal_execution")
    if (
        not isinstance(raw_plan_ref, str)
        or not isinstance(promoted_plan_ref, str)
        or not isinstance(base_execution_ref, str)
        or s308_environment.evidence_refs.get("formal_execution") != base_execution_ref
    ):
        raise _fail("S307_HANDOFF_CONTROL_REFS_INVALID")

    raw_plan_path = _resolve_ref(
        raw_plan_ref, roots=(arguments.data_root,), field="raw_plan_ref"
    )
    raw_plan = _mapping(_load_json(raw_plan_path), field="raw_plan")
    raw_plan_hash = raw_plan.get("artifact_hash")
    if raw_plan_hash != canonical_json_hash(
        {key: item for key, item in raw_plan.items() if key != "artifact_hash"}
    ):
        raise _fail("S307_HANDOFF_RAW_PLAN_HASH_INVALID")
    promoted = _load_output_authority(
        arguments.data_root,
        promoted_plan_ref,
        task_id="stage3.formal_plan_authority",
        kinds={"formal_plan", "stage3_formal_plan"},
    )
    if (
        promoted.identity.config_hash != s308_config.config_hash
        or raw_plan_ref not in promoted.source_refs
        or dict(promoted.payload) != dict(raw_plan)
    ):
        raise _fail("S307_HANDOFF_PROMOTED_PLAN_MISMATCH")
    execution_artifact = _load_output_authority(
        arguments.data_root,
        base_execution_ref,
        task_id=None,
        kinds={"formal_execution_evidence", "execution_evidence"},
    )
    execution = FormalExecutionEvidence.from_mapping(dict(execution_artifact.payload))
    execution.require_for_stage(3)
    if raw_plan.get("execution_evidence_hash") != execution.artifact_hash:
        raise _fail("S307_HANDOFF_PLAN_EXECUTION_MISMATCH")

    units = raw_plan.get("required_unit_ids")
    rules = raw_plan.get("candidate_rules")
    if (
        not isinstance(units, list)
        or len(units) != 99
        or len(set(units)) != 99
        or any(not isinstance(item, str) or not item for item in units)
        or not isinstance(rules, list)
        or len(rules) != 13
        or len(set(rules)) != len(rules)
        or any(not isinstance(item, str) or not item for item in rules)
    ):
        raise _fail("S307_HANDOFF_PLAN_COVERAGE_INVALID")
    return {
        "s308_config_hash": s308_config.config_hash,
        "s308_input_refs": tuple(input_refs),
        "raw_plan": raw_plan,
        "raw_plan_ref": raw_plan_ref,
        "promoted_plan_ref": promoted_plan_ref,
        "promoted_plan_artifact_hash": promoted.identity.artifact_hash,
        "base_execution_ref": base_execution_ref,
        "base_execution_hash": execution.artifact_hash,
        "units": tuple(units),
        "rules": tuple(rules),
    }


def _load_output_authority(
    root: Path,
    reference: str,
    *,
    task_id: str | None,
    kinds: set[str],
) -> LoadedTaskArtifact:
    try:
        loaded = load_committed_task_artifact(root, reference, require_formal=True)
    except (OSError, TypeError, ValueError) as error:
        raise _fail("S307_HANDOFF_AUTHORITY_INVALID", reference) from error
    if loaded.identity.artifact_kind not in kinds or (
        task_id is not None and loaded.identity.task_id != task_id
    ):
        raise _fail("S307_HANDOFF_AUTHORITY_IDENTITY_INVALID", reference)
    return loaded


def _validate_status(runner: FanoutRunner) -> None:
    status = _mapping(_load_json(runner.status_path), field="fanout_status")
    rows = status.get("units")
    if (
        status.get("schema_version") != "stage3-unit-status-v1"
        or status.get("config_hash") != runner.run_config_hash
        or status.get("unit_index_hash") != runner.manifest["unit_index_hash"]
        or not isinstance(rows, list)
        or len(rows) != len(runner.units)
    ):
        raise _fail("S307_HANDOFF_STATUS_INVALID")
    expected_ids = [unit.unit_id for unit in runner.units]
    if [row.get("unit_id") for row in rows if isinstance(row, Mapping)] != expected_ids:
        raise _fail("S307_HANDOFF_STATUS_UNIT_ORDER_INVALID")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or row.get("status") != "PASS"
            or row.get("completed_phases") != ["observation", "reference"]
            or row.get("fanout_manifest_hash") != runner.manifest_hash
        ):
            raise _fail("S307_HANDOFF_STATUS_ROW_INVALID")


def _validate_final(
    runner: FanoutRunner,
    final_result: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> Mapping[str, Any]:
    _validate_status(runner)
    authoritative, quarantined = _authoritative_s308_refs(final_result)
    if tuple(authoritative.values()) != controls["s308_input_refs"]:
        raise _fail("S307_HANDOFF_S308_INPUT_REF_DRIFT")
    final_config_hash = str(runner.steps[-1]["config_hash"])
    path_artifact = _load_output(
        runner.data_root,
        authoritative["formal_path_results"],
        kind="formal_path_results",
        config_hash=final_config_hash,
    )
    completeness_artifact = _load_output(
        runner.data_root,
        authoritative["completeness_report"],
        kind="completeness_report",
        config_hash=final_config_hash,
    )
    non_authority: dict[str, Mapping[str, str]] = {}
    for kind, ref in quarantined.items():
        loaded = _load_output(
            runner.data_root, ref, kind=kind, config_hash=final_config_hash
        )
        non_authority[kind] = {
            "ref": ref,
            "artifact_hash": loaded.identity.artifact_hash,
            "role": "historical_candidate_only_not_g3_6_authority",
        }

    formal = _mapping(path_artifact.payload, field="formal_path_results")
    completeness = _mapping(
        completeness_artifact.payload, field="completeness_report"
    )
    units = controls["units"]
    rules = controls["rules"]
    progress = _mapping(formal.get("observation_progress"), field="observation_progress")
    coverage = _mapping(formal.get("streaming_coverage"), field="streaming_coverage")
    observations = formal.get("observations")
    if (
        formal.get("schema_version") != "stage3-task-path-results-v1"
        or formal.get("scope") != "formal"
        or formal.get("formal_eligible") is not False
        or formal.get("formal_gate_status") != "PENDING_G3_6_AND_G3_7"
        or formal.get("selected_rule") is not None
        or formal.get("quadrature_recommendation") is not None
        or formal.get("execution_evidence_hash") != controls["base_execution_hash"]
        or formal.get("formal_plan_ref") != controls["raw_plan_ref"]
        or formal.get("formal_plan_hash") != controls["raw_plan"]["artifact_hash"]
        or formal.get("production_unit_index_ref")
        != controls["raw_plan"].get("production_unit_index_ref")
        or formal.get("production_unit_index_hash")
        != controls["raw_plan"].get("production_unit_index_hash")
        or progress.get("complete_unit_ids") != list(units)
        or progress.get("missing_unit_ids") != []
        or progress.get("observed_unit_count") != len(units)
        or progress.get("required_unit_count") != len(units)
        or coverage.get("required_unit_ids") != list(units)
        or coverage.get("candidate_rule_names") != list(rules)
        or coverage.get("committed_unit_ids") != list(units)
        or coverage.get("missing_unit_ids") != []
        or formal.get("raw_aggregate_complete_unit_ids") != list(units)
        or formal.get("raw_aggregate_missing_unit_ids") != []
        or not isinstance(observations, list)
        or len(observations) != len(units) * len(rules)
    ):
        raise _fail("S307_HANDOFF_FORMAL_RESULT_INVALID")
    if (
        completeness.get("schema_version")
        != "stage3-task-completeness-report-v1"
        or completeness.get("active_unit_id") != runner.steps[-1]["unit_id"]
        or completeness.get("selected_rule") is not None
        or completeness.get("defined") is not True
    ):
        raise _fail("S307_HANDOFF_COMPLETENESS_INVALID")

    streaming_ref = formal.get("streaming_aggregate_ref")
    streaming_hash = formal.get("streaming_aggregate_hash")
    binding_hash = formal.get("reference_binding_hash")
    if not all(
        isinstance(item, str) and item
        for item in (streaming_ref, streaming_hash, binding_hash)
    ):
        raise _fail("S307_HANDOFF_STREAMING_BINDING_INVALID")
    _aggregate, verified_streaming_hash = _load_streaming_coverage(
        runner.data_root,
        streaming_ref,
        expected_units=units,
        expected_rules=rules,
        expected_execution_hash=controls["base_execution_hash"],
        expected_plan_ref=controls["raw_plan_ref"],
        expected_plan_hash=controls["raw_plan"]["artifact_hash"],
        expected_index_ref=controls["raw_plan"].get("production_unit_index_ref"),
        expected_index_hash=controls["raw_plan"].get("production_unit_index_hash"),
        expected_binding_hash=binding_hash,
        declared_hash=streaming_hash,
    )
    return {
        "authoritative_s308_predecessors": {
            kind: {
                "ref": authoritative[kind],
                "artifact_hash": (
                    path_artifact.identity.artifact_hash
                    if kind == "formal_path_results"
                    else completeness_artifact.identity.artifact_hash
                ),
            }
            for kind in S308_OUTPUT_KINDS
        },
        "quarantined_non_authority_outputs": non_authority,
        "g36_streaming_inputs": {
            "streaming_coverage_ref": streaming_ref,
            "streaming_coverage_hash": verified_streaming_hash,
            "streaming_formal_plan_ref": controls["raw_plan_ref"],
            "formal_plan_ref": controls["promoted_plan_ref"],
            "execution_evidence_ref": controls["base_execution_ref"],
        },
    }


def run_audit(arguments: argparse.Namespace) -> Mapping[str, Any]:
    data_root = arguments.data_root.resolve()
    manifest = _mapping(_load_json(arguments.manifest), field="fanout_manifest")
    state_dir = _resolve_ref(manifest.get("state_dir"), roots=(data_root,), field="state_dir")
    if not state_dir.is_dir() or not (state_dir / "fanout-state.json").is_file():
        raise _fail("S307_HANDOFF_STATE_MISSING")
    runner = FanoutRunner(
        manifest,
        workspace_root=arguments.workspace_root,
        data_root=data_root,
        environment=arguments.s307_environment,
        python_executable=str(arguments.python_executable),
        config_catalog=_historical_s307_catalog(),
    )
    if runner.task_id != S307_TASK or len(runner.steps) != 99:
        raise _fail("S307_HANDOFF_MANIFEST_IDENTITY_INVALID")
    controls = _load_controls(arguments)
    if tuple(runner.unit_ids) != controls["units"]:
        raise _fail("S307_HANDOFF_MANIFEST_PLAN_BINDING_INVALID")
    audited, inflight_result = _audit_completed_steps(runner)
    cursor = int(runner.state["next_step"])
    if cursor < len(runner.steps) and runner.status_path.exists():
        raise _fail("S307_HANDOFF_PREMATURE_STATUS")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "COMPLETE" if cursor == len(runner.steps) else "IN_PROGRESS",
        "formal_eligible": cursor == len(runner.steps),
        "task_id": runner.task_id,
        "manifest_hash": runner.manifest_hash,
        "state_hash": runner.state["state_hash"],
        "completed_unit_count": cursor,
        "required_unit_count": len(runner.steps),
        "next_step_id": None if cursor == len(runner.steps) else runner.steps[cursor]["step_id"],
        "inflight_result_present": inflight_result,
        "audited_result_hashes": [str(item["result_hash"]) for item in audited],
        "s308_target": {
            "config_hash": controls["s308_config_hash"],
            "input_result_refs": list(controls["s308_input_refs"]),
            "promoted_plan_ref": controls["promoted_plan_ref"],
            "promoted_plan_artifact_hash": controls["promoted_plan_artifact_hash"],
            "base_execution_ref": controls["base_execution_ref"],
            "base_execution_hash": controls["base_execution_hash"],
        },
    }
    if cursor == len(runner.steps):
        report.update(_validate_final(runner, audited[-1], controls))
    report["audit_hash"] = canonical_json_hash(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--s307-environment", type=Path, required=True)
    parser.add_argument("--s308-config", type=Path, required=True)
    parser.add_argument("--s308-environment", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = run_audit(_parser().parse_args(argv))
    except (OSError, TypeError, ValueError, Stage3OrchestratorError) as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "formal_eligible": False,
            "reason": f"{type(error).__name__}:{error}",
        }
        report["audit_hash"] = canonical_json_hash(report)
    print(canonical_json_bytes(report).decode("utf-8"), end="")
    if report["status"] == "COMPLETE":
        return 0
    return 3 if report["status"] == "IN_PROGRESS" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "SCHEMA_VERSION",
    "Stage3HandoffError",
    "_authoritative_s308_refs",
    "main",
    "run_audit",
]
