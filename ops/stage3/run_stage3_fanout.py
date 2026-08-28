"""Execute the immutable per-unit waves required by Stage3.05 and Stage3.07.

The numbered task runner deliberately blocks after publishing one durable shard
until the whole production index is covered.  This control-plane wrapper turns
those expected BLOCKED boundaries into an explicit, hash-bound schedule.  It
never manufactures scientific payloads: every completed phase is backed by a
strictly parsed ``TaskRunResult`` from the real task CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from ops.stage3.run_stage3_formal import (
    InstanceLock,
    Stage3OrchestratorError,
    _canonical_hash,
    _fail,
    _forbidden_ref,
    _load_json,
    _now,
    _resolve_ref,
    _write_atomic,
    load_unit_index,
)


SCHEMA_VERSION = "stage3-fanout-manifest-v1"
STATE_SCHEMA_VERSION = "stage3-fanout-state-v1"
STATUS_SCHEMA_VERSION = "stage3-unit-status-v1"
SUPPORTED_TASKS = {
    "stage3.05_reference_integral_and_precision": "pilot",
    "stage3.06_pilot_and_threshold_freeze": "pilot",
    "stage3.07_formal_experiment_matrix": "formal",
}
PHASES = {"reference", "observation"}
PARTIAL_REQUIREMENTS = {
    "stage3.05_reference_integral_and_precision": {
        "stage3.05_reference_coverage"
    },
    "stage3.06_pilot_and_threshold_freeze": {"stage3.06_pilot_coverage"},
    "stage3.07_formal_experiment_matrix": {
        "stage3.07_reference_coverage",
        "stage3.07_matrix_coverage",
    },
}


def _strict_fields(value: Mapping[str, Any], expected: set[str], code: str) -> None:
    if set(value) != expected:
        raise _fail(code, sorted(set(value) ^ expected))


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail("FANOUT_STRING_REQUIRED", field)
    return value


def _hash(value: object, field: str) -> str:
    text = _string(value, field)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise _fail("FANOUT_HASH_INVALID", field)
    return text


class FanoutRunner:
    """Validate and execute one exact Stage 3 fan-out schedule."""

    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        workspace_root: Path,
        data_root: Path,
        environment: Path,
        python_executable: str | None = None,
    ) -> None:
        self.manifest = dict(manifest)
        self.workspace_root = workspace_root.resolve()
        self.data_root = data_root.resolve()
        self.environment = environment.resolve()
        self.python_executable = python_executable or sys.executable
        self._validate_manifest()
        self.task_id = str(self.manifest["task_id"])
        self.scope = str(self.manifest["scope"])
        self.manifest_hash = str(self.manifest["manifest_hash"])
        self.run_config_hash = str(self.manifest["run_config_hash"])
        self.unit_index_path = _resolve_ref(
            self.manifest["unit_index_ref"],
            roots=(self.data_root, self.workspace_root),
            field="unit_index_ref",
        )
        index_hash, units = load_unit_index(self.unit_index_path, scope=self.scope)
        if index_hash != self.manifest["unit_index_hash"]:
            raise _fail("FANOUT_UNIT_INDEX_HASH_DRIFT")
        self.units = units
        self.unit_ids = tuple(item.unit_id for item in units)
        self.state_dir = _resolve_ref(
            self.manifest["state_dir"], roots=(self.data_root,), field="state_dir"
        )
        self.status_path = _resolve_ref(
            self.manifest["status_ref"], roots=(self.data_root,), field="status_ref"
        )
        self.state_path = self.state_dir / "fanout-state.json"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.steps = tuple(dict(item) for item in self.manifest["steps"])
        self._validate_schedule()
        self.state = self._load_or_initialize_state()

    def _validate_manifest(self) -> None:
        expected = {
            "schema_version",
            "task_id",
            "scope",
            "run_config_hash",
            "unit_index_ref",
            "unit_index_hash",
            "state_dir",
            "status_ref",
            "steps",
            "manifest_hash",
        }
        _strict_fields(self.manifest, expected, "FANOUT_MANIFEST_FIELDS_INVALID")
        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise _fail("FANOUT_MANIFEST_SCHEMA_INVALID")
        task_id = self.manifest.get("task_id")
        if task_id not in SUPPORTED_TASKS:
            raise _fail("FANOUT_TASK_UNSUPPORTED", task_id)
        if self.manifest.get("scope") != SUPPORTED_TASKS[task_id]:
            raise _fail("FANOUT_SCOPE_TASK_MISMATCH")
        _hash(self.manifest.get("run_config_hash"), "run_config_hash")
        _hash(self.manifest.get("unit_index_hash"), "unit_index_hash")
        steps = self.manifest.get("steps")
        if not isinstance(steps, list) or not steps:
            raise _fail("FANOUT_STEPS_REQUIRED")
        for field in ("unit_index_ref", "state_dir", "status_ref"):
            value = _string(self.manifest.get(field), field)
            _forbidden_ref(value, field)
        declared = _hash(self.manifest.get("manifest_hash"), "manifest_hash")
        calculated = _canonical_hash(
            {key: value for key, value in self.manifest.items() if key != "manifest_hash"}
        )
        if declared != calculated:
            raise _fail("FANOUT_MANIFEST_HASH_INVALID")
        if not self.environment.is_file():
            raise _fail("FANOUT_ENVIRONMENT_MISSING", self.environment)
        environment = _load_json(self.environment)
        supplied = environment.get("environment_hash")
        if supplied != _canonical_hash(
            {key: value for key, value in environment.items() if key != "environment_hash"}
        ):
            raise _fail("FANOUT_ENVIRONMENT_HASH_INVALID")

    def _validate_schedule(self) -> None:
        expected_fields = {
            "step_id",
            "unit_id",
            "completes_phases",
            "action",
            "config_ref",
            "config_hash",
            "retry_config_ref",
            "retry_config_hash",
            "result_ref",
            "command",
            "expected_status",
            "expected_blocker_requirements",
        }
        coverage: dict[str, set[str]] = {unit_id: set() for unit_id in self.unit_ids}
        step_ids: set[str] = set()
        for index, step in enumerate(self.steps):
            _strict_fields(step, expected_fields, "FANOUT_STEP_FIELDS_INVALID")
            step_id = _string(step.get("step_id"), f"steps[{index}].step_id")
            if step_id in step_ids:
                raise _fail("FANOUT_STEP_ID_DUPLICATE", step_id)
            step_ids.add(step_id)
            unit_id = step.get("unit_id")
            if unit_id not in coverage:
                raise _fail("FANOUT_STEP_UNIT_UNKNOWN", unit_id)
            phases = step.get("completes_phases")
            if (
                not isinstance(phases, list)
                or not phases
                or any(phase not in PHASES for phase in phases)
                or len(phases) != len(set(phases))
            ):
                raise _fail("FANOUT_STEP_PHASES_INVALID", step_id)
            overlap = coverage[str(unit_id)].intersection(phases)
            if overlap:
                raise _fail("FANOUT_PHASE_COVERAGE_DUPLICATE", f"{unit_id}:{sorted(overlap)}")
            coverage[str(unit_id)].update(str(phase) for phase in phases)
            action = step.get("action")
            if action not in {"run", "resume"}:
                raise _fail("FANOUT_ACTION_INVALID", step_id)
            config_ref = _string(step.get("config_ref"), f"{step_id}.config_ref")
            retry_config_ref = _string(
                step.get("retry_config_ref"), f"{step_id}.retry_config_ref"
            )
            result_ref = _string(step.get("result_ref"), f"{step_id}.result_ref")
            _forbidden_ref(config_ref, f"{step_id}.config_ref")
            _forbidden_ref(retry_config_ref, f"{step_id}.retry_config_ref")
            _forbidden_ref(result_ref, f"{step_id}.result_ref")
            config_hash = _hash(step.get("config_hash"), f"{step_id}.config_hash")
            retry_config_hash = _hash(
                step.get("retry_config_hash"), f"{step_id}.retry_config_hash"
            )
            config_path = _resolve_ref(
                config_ref,
                roots=(self.data_root, self.workspace_root),
                field=f"{step_id}.config_ref",
            )
            config = _load_json(config_path)
            try:
                from param_importance_nlp.contracts import ResolvedConfigV2

                resolved_config = ResolvedConfigV2.from_mapping(config)
            except Exception as error:
                raise _fail("FANOUT_CONFIG_CONTRACT_INVALID", step_id) from error
            if (
                resolved_config.task_id != self.task_id
                or resolved_config.config_hash != config_hash
            ):
                raise _fail("FANOUT_CONFIG_IDENTITY_INVALID", step_id)
            if resolved_config.run_intent != "formal":
                raise _fail("FANOUT_CONFIG_NOT_FORMAL", step_id)
            recovery = config.get("recovery")
            if not isinstance(recovery, Mapping):
                raise _fail("FANOUT_CONFIG_RECOVERY_INVALID", step_id)
            resume_ref = recovery.get("resume_ref")
            if (action == "run" and resume_ref is not None) or (
                action == "resume" and not isinstance(resume_ref, str)
            ):
                raise _fail("FANOUT_ACTION_RECOVERY_MISMATCH", step_id)
            retry_path = _resolve_ref(
                retry_config_ref,
                roots=(self.data_root, self.workspace_root),
                field=f"{step_id}.retry_config_ref",
            )
            try:
                retry_config = ResolvedConfigV2.from_mapping(_load_json(retry_path))
            except Exception as error:
                raise _fail("FANOUT_RETRY_CONFIG_CONTRACT_INVALID", step_id) from error
            retry_recovery = retry_config.section("recovery")
            if (
                retry_config.task_id != self.task_id
                or retry_config.config_hash != retry_config_hash
                or retry_config.run_intent != "formal"
                or not isinstance(retry_recovery, Mapping)
                or not isinstance(retry_recovery.get("resume_ref"), str)
            ):
                raise _fail("FANOUT_RETRY_CONFIG_IDENTITY_INVALID", step_id)
            orchestration = resolved_config.section("orchestration")
            if not isinstance(orchestration, Mapping):
                raise _fail("FANOUT_ORCHESTRATION_INVALID", step_id)
            if retry_config.section("orchestration") != orchestration:
                raise _fail("FANOUT_RETRY_CONFIG_ROUTE_DRIFT", step_id)
            route_spec_ref = orchestration.get("route_spec_ref")
            if not isinstance(route_spec_ref, str) or not route_spec_ref:
                raise _fail("FANOUT_SELECTOR_REF_REQUIRED", step_id)
            selector_path = _resolve_ref(
                route_spec_ref,
                roots=(self.data_root, self.workspace_root),
                field=f"{step_id}.route_spec_ref",
            )
            selector = _load_json(selector_path)
            if set(selector) != {
                "schema_version",
                "scope",
                "unit_index_hash",
                "active_unit_id",
                "artifact_hash",
            }:
                raise _fail("FANOUT_SELECTOR_FIELDS_INVALID", step_id)
            if (
                selector.get("schema_version") != "stage3-path-unit-selector-v1"
                or selector.get("scope") != self.scope
                or selector.get("unit_index_hash") != self.manifest["unit_index_hash"]
                or selector.get("active_unit_id") != unit_id
                or selector.get("artifact_hash")
                != _canonical_hash(
                    {
                        key: value
                        for key, value in selector.items()
                        if key != "artifact_hash"
                    }
                )
            ):
                raise _fail("FANOUT_ACTIVE_UNIT_MISMATCH", step_id)
            command = step.get("command")
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(token, str) or not token for token in command)
            ):
                raise _fail("FANOUT_COMMAND_INVALID", step_id)
            required_placeholders = {"{action}", "{config}", "{environment}", "{result}"}
            joined = "\n".join(command)
            if any(item not in joined for item in required_placeholders):
                raise _fail("FANOUT_COMMAND_PLACEHOLDER_MISSING", step_id)
            status = step.get("expected_status")
            requirements = step.get("expected_blocker_requirements")
            if (
                status not in {"PASS", "BLOCKED"}
                or not isinstance(requirements, list)
                or any(not isinstance(item, str) or not item for item in requirements)
                or (status == "PASS" and requirements)
                or (status == "BLOCKED" and not requirements)
            ):
                raise _fail("FANOUT_EXPECTED_RESULT_INVALID", step_id)
            if status == "BLOCKED" and (
                len(requirements) != 1
                or requirements[0] not in PARTIAL_REQUIREMENTS[self.task_id]
            ):
                raise _fail("FANOUT_PARTIAL_BOUNDARY_NOT_AUTHORIZED", step_id)
            if self.task_id.startswith("stage3.07"):
                requirement = requirements[0] if requirements else None
                if requirement == "stage3.07_reference_coverage" and phases != [
                    "reference"
                ]:
                    raise _fail("FANOUT_REFERENCE_PHASE_BOUNDARY_INVALID", step_id)
                if requirement == "stage3.07_matrix_coverage" and "observation" not in phases:
                    raise _fail("FANOUT_OBSERVATION_PHASE_BOUNDARY_INVALID", step_id)
                if status == "PASS" and "observation" not in phases:
                    raise _fail("FANOUT_FINAL_OBSERVATION_MISSING", step_id)
        if self.task_id.startswith("stage3.05"):
            expected_phases = {"reference"}
        elif self.task_id.startswith("stage3.06"):
            expected_phases = {"observation"}
        else:
            expected_phases = PHASES
        for unit_id, phases in coverage.items():
            if phases != expected_phases:
                raise _fail("FANOUT_PHASE_COVERAGE_INCOMPLETE", f"{unit_id}:{sorted(phases)}")
        if self.steps[-1].get("expected_status") != "PASS":
            raise _fail("FANOUT_FINAL_STEP_MUST_PASS")
        if any(step.get("expected_status") == "PASS" for step in self.steps[:-1]):
            raise _fail("FANOUT_EARLY_PASS_INVALID")

    def _new_state(self) -> dict[str, Any]:
        now = _now()
        value: dict[str, Any] = {
            "schema_version": STATE_SCHEMA_VERSION,
            "manifest_hash": self.manifest_hash,
            "task_id": self.task_id,
            "scope": self.scope,
            "next_step": 0,
            "unit_phases": {unit_id: [] for unit_id in self.unit_ids},
            "attempts": [],
            "created_at": now,
            "updated_at": now,
        }
        value["state_hash"] = _canonical_hash(value)
        return value

    def _load_or_initialize_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            value = self._new_state()
            _write_atomic(self.state_path, value)
            return value
        value = dict(_load_json(self.state_path))
        declared = value.get("state_hash")
        if declared != _canonical_hash(
            {key: item for key, item in value.items() if key != "state_hash"}
        ):
            raise _fail("FANOUT_STATE_HASH_INVALID")
        if (
            value.get("schema_version") != STATE_SCHEMA_VERSION
            or value.get("manifest_hash") != self.manifest_hash
            or value.get("task_id") != self.task_id
            or value.get("scope") != self.scope
        ):
            raise _fail("FANOUT_STATE_IDENTITY_DRIFT")
        next_step = value.get("next_step")
        if isinstance(next_step, bool) or not isinstance(next_step, int) or not 0 <= next_step <= len(self.steps):
            raise _fail("FANOUT_STATE_CURSOR_INVALID")
        phases = value.get("unit_phases")
        if not isinstance(phases, Mapping) or set(phases) != set(self.unit_ids):
            raise _fail("FANOUT_STATE_COVERAGE_INVALID")
        return value

    def _save_state(self) -> None:
        self.state["updated_at"] = _now()
        self.state["state_hash"] = _canonical_hash(
            {key: item for key, item in self.state.items() if key != "state_hash"}
        )
        _write_atomic(self.state_path, self.state)

    def _command(self, step: Mapping[str, Any], config: Path, result: Path) -> list[str]:
        substitutions = {
            "{python}": self.python_executable,
            "{action}": str(step["action"]),
            "{config}": str(config),
            "{environment}": str(self.environment),
            "{result}": str(result),
            "{workspace_root}": str(self.workspace_root),
            "{data_root}": str(self.data_root),
        }
        command: list[str] = []
        for raw in step["command"]:
            token = str(raw)
            for key, value in substitutions.items():
                token = token.replace(key, value)
            if "{" in token or "}" in token:
                raise _fail("FANOUT_COMMAND_UNKNOWN_PLACEHOLDER", step["step_id"])
            command.append(token)
        return command

    def _needs_retry_config(self, step: Mapping[str, Any]) -> bool:
        if step["action"] == "resume":
            return False
        if any(
            item.get("step_id") == step["step_id"] and item.get("status") == "FAIL"
            for item in self.state["attempts"]
        ):
            return True
        retry = _load_json(
            _resolve_ref(
                step["retry_config_ref"],
                roots=(self.data_root, self.workspace_root),
                field=f"{step['step_id']}.retry_config_ref",
            )
        )
        recovery = retry.get("recovery")
        resume_ref = recovery.get("resume_ref") if isinstance(recovery, Mapping) else None
        if not isinstance(resume_ref, str):
            return False
        resume_root = _resolve_ref(
            resume_ref,
            roots=(self.data_root, self.workspace_root),
            field=f"{step['step_id']}.resume_ref",
        )
        return resume_root.is_dir() and any(path.is_file() for path in resume_root.rglob("*"))

    def _verify_result(self, step: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
        try:
            from param_importance_nlp.runtime import TaskRunResult

            parsed = TaskRunResult.from_mapping(_load_json(path))
            value = parsed.to_dict()
        except Exception as error:
            raise _fail("FANOUT_RESULT_INVALID", step["step_id"]) from error
        if (
            value.get("task_id") != self.task_id
            or value.get("run_intent") != "formal"
            or value.get("config_hash") != step["config_hash"]
            or value.get("status") != step["expected_status"]
        ):
            raise _fail("FANOUT_RESULT_IDENTITY_INVALID", step["step_id"])
        blockers = value.get("blockers")
        requirements = (
            sorted(item.get("requirement") for item in blockers)
            if isinstance(blockers, list) and all(isinstance(item, Mapping) for item in blockers)
            else []
        )
        if requirements != sorted(step["expected_blocker_requirements"]):
            raise _fail("FANOUT_BLOCKER_BOUNDARY_INVALID", step["step_id"])
        if value["status"] == "BLOCKED" and any(
            item.get("code") != "ASSET_UNAVAILABLE"
            or item.get("retryable") is not True
            for item in blockers
        ):
            raise _fail("FANOUT_BLOCKER_NOT_RETRYABLE_COVERAGE", step["step_id"])
        if value["status"] == "PASS" and value.get("formal_eligible") is not True:
            raise _fail("FANOUT_FINAL_RESULT_NOT_FORMAL", step["step_id"])
        return value

    def _record_success(self, step: Mapping[str, Any], result: Mapping[str, Any], *, recovered: bool) -> None:
        unit_id = str(step["unit_id"])
        phases = self.state["unit_phases"]
        completed = list(phases[unit_id])
        completed.extend(str(item) for item in step["completes_phases"])
        phases[unit_id] = sorted(set(completed))
        self.state["attempts"].append(
            {
                "step_id": step["step_id"],
                "unit_id": unit_id,
                "action": step["action"],
                "status": result["status"],
                "result_hash": result["result_hash"],
                "recovered_from_result": recovered,
                "recorded_at": _now(),
            }
        )
        self.state["next_step"] = int(self.state["next_step"]) + 1
        self._save_state()

    def _write_status(self) -> None:
        if self.task_id.startswith("stage3.05"):
            expected = ["reference"]
        elif self.task_id.startswith("stage3.06"):
            expected = ["observation"]
        else:
            expected = sorted(PHASES)
        rows = []
        for unit in self.units:
            phases = sorted(self.state["unit_phases"][unit.unit_id])
            if phases != expected:
                raise _fail("FANOUT_STATUS_PHASE_INCOMPLETE", unit.unit_id)
            rows.append(
                {
                    "unit_id": unit.unit_id,
                    "status": "PASS",
                    "completed_phases": phases,
                    "fanout_manifest_hash": self.manifest_hash,
                }
            )
        payload = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "config_hash": self.run_config_hash,
            "unit_index_hash": self.manifest["unit_index_hash"],
            "units": rows,
        }
        if self.status_path.exists():
            if _load_json(self.status_path) != payload:
                raise _fail("FANOUT_STATUS_IMMUTABLE_DRIFT")
            return
        _write_atomic(self.status_path, payload)

    def run(self, *, executor: Any = subprocess.run) -> Mapping[str, Any]:
        with InstanceLock(self.state_dir / "fanout.lock"):
            while int(self.state["next_step"]) < len(self.steps):
                step = self.steps[int(self.state["next_step"])]
                config = _resolve_ref(
                    step["config_ref"],
                    roots=(self.data_root, self.workspace_root),
                    field=f"{step['step_id']}.config_ref",
                )
                result = _resolve_ref(
                    step["result_ref"],
                    roots=(self.data_root,),
                    field=f"{step['step_id']}.result_ref",
                )
                recovered = result.is_file()
                if not recovered:
                    result.parent.mkdir(parents=True, exist_ok=True)
                    effective_step = step
                    if self._needs_retry_config(step):
                        effective_step = dict(step)
                        effective_step["action"] = "resume"
                        effective_step["config_ref"] = step["retry_config_ref"]
                        effective_step["config_hash"] = step["retry_config_hash"]
                        config = _resolve_ref(
                            effective_step["config_ref"],
                            roots=(self.data_root, self.workspace_root),
                            field=f"{step['step_id']}.retry_config_ref",
                        )
                    command = self._command(effective_step, config, result)
                    process = executor(
                        command,
                        cwd=self.workspace_root,
                        shell=False,
                        check=False,
                    )
                    expected_code = 0 if step["expected_status"] == "PASS" else 3
                    if process.returncode != expected_code:
                        self.state["attempts"].append(
                            {
                                "step_id": step["step_id"],
                                "unit_id": step["unit_id"],
                                "action": effective_step["action"],
                                "status": "FAIL",
                                "returncode": int(process.returncode),
                                "recorded_at": _now(),
                            }
                        )
                        self._save_state()
                        raise _fail(
                            "FANOUT_PROCESS_FAILED",
                            f"{step['step_id']}:{process.returncode}",
                        )
                verification_step = effective_step if not recovered else step
                parsed = self._verify_result(verification_step, result)
                self._record_success(verification_step, parsed, recovered=recovered)
            self._write_status()
            return {
                "status": "COMPLETE",
                "task_id": self.task_id,
                "scope": self.scope,
                "unit_count": len(self.units),
                "step_count": len(self.steps),
                "status_ref": str(self.manifest["status_ref"]),
                "manifest_hash": self.manifest_hash,
            }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        runner = FanoutRunner(
            _load_json(arguments.manifest.resolve()),
            workspace_root=arguments.workspace_root,
            data_root=arguments.data_root,
            environment=arguments.environment,
        )
        result = runner.run()
    except Stage3OrchestratorError as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False))
        return 3
    print(json.dumps(dict(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["FanoutRunner", "SCHEMA_VERSION", "STATE_SCHEMA_VERSION", "main"]
