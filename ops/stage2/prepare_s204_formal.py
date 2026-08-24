#!/usr/bin/env python3
"""Prepare the formal S2.4 six-cell control-plane contract.

This command consumes only already committed formal refs.  It publishes the
pre-sizing formula plans, six resolved configs, per-cell environments, and the
explicit handoff index.  It never runs a provider and never invents numeric
``delta_sci`` values; those remain a post-sizing runner artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from param_importance_nlp.contracts import FormalExecutionEvidence, canonical_json_hash, load_canonical_json  # noqa: E402
from param_importance_nlp.contracts.status import GateRecord, GateStatus  # noqa: E402
from param_importance_nlp.runtime import load_committed_task_artifact, publish_canonical_immutable  # noqa: E402

try:  # noqa: E402
    from ops.stage2.materialize_s204 import (
        EXPECTED_CELL_IDS,
        S204_S22_CANONICAL_OUTPUT_DIR,
        S204_S22_CONTROL_OUTPUT_DIR,
        S204_S22_CONFIG_REF,
        TASK_INPUTS,
        _error,
        _load_formal_source,
        _load_gpu_health_identity,
        _mapping,
        _stage1_10_task_refs,
        _stage1_task_refs,
        _validate_formal_s22_task_group,
        _safe_relative,
        _source_ref,
        _extend_formal_execution,
        build_formal_runtime_environment,
        generate_six_cell_configs,
        publish_formal_registry_artifacts,
        publish_per_cell_delta_sci_plans,
        publish_per_cell_runtime_environments,
        publish_per_cell_sizing_plans,
        publish_six_cell_manifest,
        publish_six_cell_materialization_index,
        write_six_cell_configs,
        produce_formal_s22_task_outputs,
        ensure_formal_s22_task_outputs,
    )
except ModuleNotFoundError:  # direct ``python ops/stage2/prepare...`` launch
    from materialize_s204 import (
        EXPECTED_CELL_IDS,
        S204_S22_CANONICAL_OUTPUT_DIR,
        S204_S22_CONTROL_OUTPUT_DIR,
        S204_S22_CONFIG_REF,
        TASK_INPUTS,
        _error,
        _load_formal_source,
        _load_gpu_health_identity,
        _mapping,
        _stage1_10_task_refs,
        _stage1_task_refs,
        _validate_formal_s22_task_group,
        _safe_relative,
        _source_ref,
        _extend_formal_execution,
        build_formal_runtime_environment,
        generate_six_cell_configs,
        publish_formal_registry_artifacts,
        publish_per_cell_delta_sci_plans,
        publish_per_cell_runtime_environments,
        publish_per_cell_sizing_plans,
        publish_six_cell_manifest,
        publish_six_cell_materialization_index,
        write_six_cell_configs,
        produce_formal_s22_task_outputs,
        ensure_formal_s22_task_outputs,
    )


_ADAPTER_TASKS = {
    "stage2.G2.0": ("stage2.01_scope_hypotheses_and_preregistration", "stage2.01_scope_hypotheses_and_preregistration"),
    "stage2.G2.1": ("stage2.02_stage1_handoff_and_fixed_state_contract", "stage2.02_stage1_handoff_and_fixed_state_contract"),
    "stage2.G2.2": ("stage2.03_assets_checkpoints_and_sampling", "stage2.03_assets_checkpoints_and_sampling"),
}


def _load_sources(root: Path, ref: str) -> dict[str, Any]:
    source_ref = _source_ref(ref, "sources")
    value = load_canonical_json(_safe_relative(root, source_ref, "sources"))
    if not isinstance(value, Mapping):
        raise _error("S204_SOURCES_OBJECT_REQUIRED")
    return dict(value)


def _required_ref(source: Mapping[str, Any], key: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value:
        raise _error("S204_SOURCE_REF_REQUIRED", key)
    return value


def _validate_adapter_gate(
    root: Path,
    *,
    gate_id: str,
    gate_ref: str,
    expected_refs: tuple[str, ...],
    forbidden_task_ids: tuple[str, ...] = (),
) -> str:
    task_id, _ = _ADAPTER_TASKS[gate_id]
    ref = _source_ref(gate_ref, f"{gate_id}.gate_ref")
    try:
        loaded = load_committed_task_artifact(root, ref, require_formal=True)
        gate = GateRecord.from_mapping(dict(loaded.payload))
    except Exception as error:
        raise _error("S204_FORMAL_ADAPTER_COMMIT_REQUIRED", gate_id) from error
    if loaded.identity.task_id != task_id or loaded.identity.artifact_kind != "gate_record":
        raise _error("S204_FORMAL_ADAPTER_IDENTITY_MISMATCH", gate_id)
    if gate.gate_id != gate_id or gate.status is not GateStatus.PASS:
        raise _error("S204_FORMAL_ADAPTER_NOT_PASS", gate_id)
    observed = set(loaded.source_refs) | set(gate.evidence_refs)
    # A gate can carry arbitrary stable evidence refs, so explicitly inspect
    # formal task commits when rejecting a downstream edge.  Non-task reports
    # remain valid evidence and are handled by their own validators.
    for observed_ref in observed:
        try:
            upstream = load_committed_task_artifact(root, observed_ref, require_formal=True)
        except Exception:
            continue
        if upstream.identity.task_id in forbidden_task_ids:
            raise _error("S204_FORMAL_ADAPTER_DOWNSTREAM_BINDING", gate_id)
    if not set(expected_refs).issubset(observed):
        raise _error("S204_FORMAL_ADAPTER_UPSTREAM_MISMATCH", gate_id)
    return ref


def prepare_formal_s204(
    data_root: str | Path,
    *,
    sources: Mapping[str, Any],
    output_dir: str = "evidence/stage2/s204/prepared-r7",
) -> dict[str, Any]:
    """Publish all pre-sizing S2.4 control-plane objects in one order."""

    root = Path(data_root).resolve()
    output = PurePosixPath(output_dir).as_posix()
    formal_execution_ref = _required_ref(sources, "formal_execution")
    evidence = FormalExecutionEvidence.from_mapping(
        load_canonical_json(_safe_relative(root, formal_execution_ref, "formal_execution"))
    )
    evidence.require_for_stage(2)

    predecessor_raw = sources.get("predecessor_refs")
    if not isinstance(predecessor_raw, Mapping):
        raise _error("S204_PREDECESSOR_REFS_REQUIRED")
    predecessor_refs: dict[str, dict[str, str]] = {}
    s21_task = "stage2.01_scope_hypotheses_and_preregistration"
    s22_task = "stage2.02_stage1_handoff_and_fixed_state_contract"
    s23_task = "stage2.03_assets_checkpoints_and_sampling"
    for task_id in (s21_task, s23_task):
        kinds = TASK_INPUTS[task_id]
        raw = predecessor_raw.get(task_id)
        if not isinstance(raw, Mapping) or set(raw) != set(kinds):
            raise _error("S204_PREDECESSOR_ARTIFACT_SET_INVALID", task_id)
        predecessor_refs[task_id] = {}
        for kind in kinds:
            ref = _source_ref(raw[kind], f"predecessor.{task_id}.{kind}")
            _load_formal_source(root, ref, task_id=task_id, artifact_kind=kind)
            predecessor_refs[task_id][kind] = ref
    raw_s22 = predecessor_raw.get(s22_task)
    if raw_s22 is not None:
        if not isinstance(raw_s22, Mapping) or set(raw_s22) != set(TASK_INPUTS[s22_task]):
            raise _error("S204_PREDECESSOR_ARTIFACT_SET_INVALID", s22_task)
        predecessor_refs[s22_task] = {
            kind: _source_ref(raw_s22[kind], f"predecessor.{s22_task}.{kind}")
            for kind in TASK_INPUTS[s22_task]
        }
        for kind, ref in predecessor_refs[s22_task].items():
            _load_formal_source(root, ref, task_id=s22_task, artifact_kind=kind)

    g20_ref = _validate_adapter_gate(
        root,
        gate_id="stage2.G2.0",
        gate_ref=_required_ref(sources, "g20_adapter_output"),
        expected_refs=tuple(predecessor_refs[s21_task].values()),
    )
    stage1_10_refs = dict(_stage1_10_task_refs(sources))
    stage1_11_refs = dict(_stage1_task_refs(sources))
    g21_ref = _validate_adapter_gate(
        root,
        gate_id="stage2.G2.1",
        gate_ref=_required_ref(sources, "g21_adapter_output"),
        expected_refs=tuple((*predecessor_refs[s21_task].values(), *stage1_10_refs.values(), *stage1_11_refs.values())),
        forbidden_task_ids=(s22_task,),
    )
    asset_ref = _source_ref(_required_ref(sources, "stage2_asset_resolution"), "stage2_asset_resolution")
    if asset_ref != predecessor_refs[s23_task]["asset_resolution"]:
        raise _error("S204_ASSET_REF_PREDECESSOR_MISMATCH")
    g22_ref = _validate_adapter_gate(
        root,
        gate_id="stage2.G2.2",
        gate_ref=_required_ref(sources, "g22_adapter_output"),
        expected_refs=(asset_ref, *tuple(load_committed_task_artifact(root, asset_ref, require_formal=True).source_refs)),
    )
    g22_gate = GateRecord.from_mapping(
        dict(load_committed_task_artifact(root, g22_ref, require_formal=True).payload)
    )

    # The external G2.1 handoff report is a hardware authority; it is not the
    # S2.2 TaskArtifact handoff_manifest.  Keep the two refs independent.
    g21_handoff_ref = _required_ref(sources, "g21_handoff")
    _load_gpu_health_identity(
        root,
        _source_ref(g21_handoff_ref, "g21_handoff"),
        expected_stage1_ref=_required_ref(sources, "stage1_g1_exit"),
    )
    supplied_predecessor_dir = PurePosixPath(
        str(sources.get("formal_predecessor_output_dir", S204_S22_CONTROL_OUTPUT_DIR))
    ).as_posix()
    if supplied_predecessor_dir != S204_S22_CONTROL_OUTPUT_DIR:
        raise _error("S204_S22_OUTPUT_DIR_NOT_CANONICAL")
    formal_predecessor_dir = S204_S22_CONTROL_OUTPUT_DIR
    formal_evidence_ref = formal_execution_ref
    _, evidence, formal_evidence_ref, _s22_config_ref, _s22_environment_ref = ensure_formal_s22_task_outputs(
        root,
        predecessor_refs=predecessor_refs,
        output_dir=formal_predecessor_dir,
        producer_kwargs={
            "source": sources,
            "s21_refs": predecessor_refs[s21_task],
            "g20_ref": g20_ref,
            "g20_gate": GateRecord.from_mapping(dict(load_committed_task_artifact(root, g20_ref, require_formal=True).payload)),
            "g21_ref": g21_ref,
            "g21_gate": GateRecord.from_mapping(dict(load_committed_task_artifact(root, g21_ref, require_formal=True).payload)),
            "g21_resolved_config_ref": _required_ref(sources, "g21_resolved_config"),
            "formal_execution_ref": formal_execution_ref,
            "base_config_ref": str(sources.get("base_config_ref", "configs/run-ready/layers/formal-stage2-estimator.yaml")),
            "stage0_ref": _required_ref(sources, "stage0_handoff"),
            "stage1_ref": _required_ref(sources, "stage1_g1_exit"),
            "g3_ref": _required_ref(sources, "g3_resolution"),
            "contract_refs": {
                0: _required_ref(sources, "contract_stage_0"),
                1: _required_ref(sources, "contract_stage_1"),
                2: _required_ref(sources, "contract_stage_2"),
            },
            "stage1_10_refs": stage1_10_refs,
            "stage1_11_refs": stage1_11_refs,
            "g1_ref": _required_ref(_mapping(sources.get("gate_refs"), "gate_refs"), "stage1.G1-EXIT"),
        },
    )
    evidence, formal_evidence_ref = _extend_formal_execution(
        root,
        evidence_ref=formal_evidence_ref,
        gate=g22_gate,
        destination=f"{output}/formal-execution-g22.json",
    )

    s21_refs = predecessor_refs["stage2.01_scope_hypotheses_and_preregistration"]
    delta_refs = publish_per_cell_delta_sci_plans(
        root,
        s21_refs={key: s21_refs[key] for key in ("preregistration", "hypothesis_contract")},
        output_dir=f"{output}/reference-delta-sci-plans",
    )
    sizing_refs = publish_per_cell_sizing_plans(
        root,
        formal_execution=evidence,
        output_dir=f"{output}/reference-sizing",
    )
    registry_refs = publish_formal_registry_artifacts(
        root,
        asset_manifest_ref=asset_ref,
        g22_gate_ref=g22_ref,
        output_dir=f"{output}/parameter-registry",
    )

    g3_ref = _required_ref(sources, "g3_resolution")
    # Every formal preparation must own an append-only config namespace.  The
    # old hard-coded ``.../s204/fresh`` path allowed a later retry to silently
    # reuse configs generated from an earlier G3/source projection.  A source
    # manifest may pin the namespace explicitly; otherwise derive it from the
    # preparation root so retries remain isolated by default.
    config_output_dir = PurePosixPath(
        str(
            sources.get(
                "config_output_dir",
                f"configs/generated/stage2/s204/{PurePosixPath(output).name}",
            )
        )
    ).as_posix()
    if not config_output_dir or config_output_dir.startswith("/") or "\\" in config_output_dir:
        raise _error("S204_CONFIG_OUTPUT_DIR_INVALID")
    configs = generate_six_cell_configs(
        root,
        asset_manifest_ref=asset_ref,
        predecessor_refs=predecessor_refs,
        g3_resolution_ref=g3_ref,
        output_dir=config_output_dir,
        parameter_registry_refs=registry_refs,
        delta_sci_refs=delta_refs,
        delta_phase="pre_sizing",
    )
    config_refs = write_six_cell_configs(
        configs,
        root,
        output_dir=config_output_dir,
        mode="fresh",
    )
    manifest_ref = publish_six_cell_manifest(
        root,
        asset_manifest_ref=asset_ref,
        configs=configs,
        registry_refs=registry_refs,
        output_ref=f"{output}/six-cell-manifest.json",
    )

    capability_refs = sources.get("capability_refs")
    if not isinstance(capability_refs, Mapping):
        raise _error("S204_CAPABILITY_REFS_REQUIRED")
    base_env, base_env_ref = build_formal_runtime_environment(
        root,
        formal_execution_ref=formal_evidence_ref,
        stage0_handoff_ref=_required_ref(sources, "stage0_handoff"),
        stage1_g1_exit_ref=_required_ref(sources, "stage1_g1_exit"),
        contract_freeze_ref=_required_ref(sources, "contract_freeze"),
        contract_stage_0_ref=_required_ref(sources, "contract_stage_0"),
        contract_stage_1_ref=_required_ref(sources, "contract_stage_1"),
        contract_stage_2_ref=_required_ref(sources, "contract_stage_2"),
        g3_resolution_ref=g3_ref,
        stage2_asset_resolution_ref=asset_ref,
        g21_handoff_ref=_required_ref(sources, "g21_handoff"),
        gate_refs={"stage2.G2.0": g20_ref, "stage2.G2.1": g21_ref, "stage2.G2.2": g22_ref},
        capability_refs={str(key): str(value) for key, value in capability_refs.items()},
        reference_sizing_plan_ref=sizing_refs[EXPECTED_CELL_IDS[0]],
        g22_asset_evidence_ref=sources.get("g22_asset_evidence"),
        output_ref=f"{output}/base/runtime-environment.json",
    )
    environments, environment_refs = publish_per_cell_runtime_environments(
        root,
        base_environment_ref=base_env_ref,
        sizing_refs=sizing_refs,
        registry_refs=registry_refs,
        delta_refs=delta_refs,
        delta_phase="pre_sizing",
        output_dir=f"{output}/environments",
        six_cell_manifest_ref=manifest_ref,
        configs=configs,
        config_refs=config_refs,
        s23_asset_task_ref=asset_ref,
        asset_manifest_ref=asset_ref,
        g3_resolution_ref=g3_ref,
    )
    index_ref = publish_six_cell_materialization_index(
        root,
        config_refs=config_refs,
        environment_refs=environment_refs,
        sizing_refs=sizing_refs,
        registry_refs=registry_refs,
        delta_refs=delta_refs,
        six_cell_manifest_ref=manifest_ref,
        output_ref=f"{output}/materialization-index.json",
        mode="fresh-pre-sizing",
    )
    summary: dict[str, Any] = {
        "schema_version": "stage2-s204-formal-preparation-v1",
        "status": "READY_FOR_FORMAL_EXECUTION",
        "formal_eligible": True,
        "phase": "pre_sizing",
        "formal_execution_ref": formal_evidence_ref,
        "adapter_gate_refs": {"stage2.G2.0": g20_ref, "stage2.G2.1": g21_ref, "stage2.G2.2": g22_ref},
        "predecessor_refs": predecessor_refs,
        "s22_producer_output_dir": formal_predecessor_dir,
        "config_refs": config_refs,
        "environment_refs": environment_refs,
        "sizing_refs": sizing_refs,
        "registry_refs": registry_refs,
        "delta_plan_refs": delta_refs,
        "six_cell_manifest_ref": manifest_ref,
        "materialization_index_ref": index_ref,
        "numeric_delta_policy": "v2_is_published_only_by_the_TaskRuntime_after_committed_sizing_shards",
    }
    summary["artifact_hash"] = canonical_json_hash(summary)
    summary_ref = f"{output}/preparation.json"
    publish_canonical_immutable(_safe_relative(root, summary_ref, "preparation_summary"), summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare formal S2.4 six-cell control-plane artifacts")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True, help="JSON source-ref manifest under DATA_ROOT")
    parser.add_argument("--output-dir", default="evidence/stage2/s204/prepared-r8")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        sources = _load_sources(args.data_root.resolve(), args.sources.as_posix())
        result = prepare_formal_s204(args.data_root, sources=sources, output_dir=args.output_dir)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (OSError, TypeError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"S2.4 formal preparation blocked: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "prepare_formal_s204"]
