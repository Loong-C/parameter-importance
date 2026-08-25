"""Bounded formal-r20 queue scheduling/identity tests."""

from __future__ import annotations

import json
from pathlib import Path
import os

import pytest
from param_importance_nlp.runtime import TaskArtifactStore

from ops.stage2.run_s204_r20_queue import (
    APPROVED_GPU_UUIDS,
    EXCLUDED_GPU_UUID,
    _canonical_artifact_hash,
    _candidate_sizes_from_bound_plans,
    _child_command,
    _absolute_without_resolving,
    _hash,
    _normalize_candidate_sizes,
    _parser,
    _queue_manifest,
    _SizingBinding,
    _parse_cell_config,
    _parse_cell_environment,
    lpt_order,
)


def test_lpt_order_is_deterministic_and_longest_first() -> None:
    estimates = {
        "pythia-14m-initialization": 1.0,
        "pythia-31m-initialization": 2.0,
        "pythia-14m-early": 1.0,
        "pythia-31m-early": 2.0,
        "pythia-14m-mid_late": 1.0,
        "pythia-31m-mid_late": 2.0,
    }
    order = lpt_order(estimates)
    assert order[:3] == (
        "pythia-31m-early",
        "pythia-31m-initialization",
        "pythia-31m-mid_late",
    )
    assert set(order) == set(estimates)


def test_child_command_rejects_excluded_or_unknown_gpu() -> None:
    kwargs = {
        "python": "python",
        "launcher": Path("run_s204_formal.py"),
        "execution_commit": "a" * 40,
        "cell_id": "cell",
        "cell_config": Path("cell.json"),
        "g21_evidence": Path("g21.json"),
        "asset_resolution": Path("assets.json"),
        "data_range": Path("data.json"),
        "data_root": Path("data-root"),
        "output_root": Path("output"),
        "runtime_environment": Path("environment.json"),
        "candidate_sizes": (131072, 262144),
        "heartbeat_seconds": 30.0,
    }
    for uuid in (EXCLUDED_GPU_UUID, "GPU-unknown"):
        with pytest.raises(ValueError, match="approved r20 UUID"):
            _child_command(gpu_uuid=uuid, **kwargs)

    command = _child_command(gpu_uuid=APPROVED_GPU_UUIDS[0], **kwargs)
    assert "--execution-commit" in command
    assert command[command.index("--cuda-visible-devices") + 1] == APPROVED_GPU_UUIDS[0]
    candidate_start = command.index("--candidate-sizes")
    assert command[candidate_start : candidate_start + 3] == [
        "--candidate-sizes",
        "131072",
        "262144",
    ]
    assert "--execute" in command and "--cell-id" in command


def _write_bound_sizing_inputs(
    tmp_path: Path,
    *,
    plan_sizes: tuple[int, int] = (131072, 262144),
    amendment_sizes: tuple[int, int] = (131072, 262144),
    plan_segment: tuple[int, int] = (81920, 344064),
    amendment_segment: tuple[int, int, int] = (81920, 344064, 81920),
    plan_contract_overrides: dict[str, object] | None = None,
    amendment_contract_overrides: dict[str, object] | None = None,
) -> tuple[Path, dict[str, Path]]:
    data_root = tmp_path / "data-root"
    data_root.mkdir(parents=True)
    amendment = {
        "schema_version": "stage2-reference-sizing-amendment-v1",
        "round_id": "r23",
        "amendment_id": "r23-amend-r1",
        "sizing": {
            "candidate_sample_counts": list(amendment_sizes),
            "normalized_l1_threshold": 0.02,
            "required_consecutive": 1,
            "complete_all_candidates": True,
            "optional_stopping": False,
            "block_size": 32,
            "resume_ref": None,
            "reuse_prior_sizing_prefix": False,
            "segment_start_position": amendment_segment[0],
            "segment_end_position_exclusive": amendment_segment[1],
            "prior_consumed_end_position": amendment_segment[2],
            "final_stream_segments": {
                "reference_A": {"start_position": 81920, "end_position_exclusive": 344064},
                "reference_B": {"start_position": 81920, "end_position_exclusive": 344064},
            },
        },
    }
    if amendment_contract_overrides:
        amendment["sizing"].update(amendment_contract_overrides)
    amendment["artifact_hash"] = _canonical_artifact_hash(
        {key: value for key, value in amendment.items() if key != "artifact_hash"}
    )
    amendment_path = data_root / "amendment.json"
    amendment_path.write_text(json.dumps(amendment) + "\n", encoding="utf-8")
    plan = {
        "schema_version": "stage2-reference-sizing-plan-v1",
        "candidate_sample_counts": list(plan_sizes),
        "block_size": 32,
        "convergence_tolerance": 0.02,
        "required_consecutive": 1,
        "require_terminal_convergence": True,
        "draw_start_position": plan_segment[0],
        "draw_end_position_exclusive": plan_segment[1],
        "final_stream_start_position": plan_segment[0],
        "final_stream_end_position_exclusive": plan_segment[1],
        "round_manifest_ref": "amendment.json",
    }
    if plan_contract_overrides:
        plan.update(plan_contract_overrides)
    plan["artifact_hash"] = _canonical_artifact_hash(
        {key: value for key, value in plan.items() if key != "artifact_hash"}
    )
    plan_commit_ref = TaskArtifactStore(data_root, "task-output").publish(
        task_id="cell-0",
        artifact_kind="reference_sizing_plan",
        config_hash="a" * 64,
        run_intent="formal",
        payload=plan,
        formal_eligible=True,
    ).commit_ref
    environments: dict[str, Path] = {}
    for index in range(6):
        cell_id = f"cell-{index}"
        environment_path = data_root / f"environment-{index}.json"
        environment_path.write_text(
            json.dumps({"evidence_refs": {"stage2_reference_sizing_plan": plan_commit_ref}}) + "\n",
            encoding="utf-8",
        )
        environments[cell_id] = environment_path
    return data_root, environments


def test_candidate_sizes_require_two_explicit_nodes() -> None:
    with pytest.raises(ValueError, match="exactly two"):
        _normalize_candidate_sizes((131072,))
    with pytest.raises(ValueError, match="strictly increasing"):
        _normalize_candidate_sizes((262144, 131072))


def test_queue_cli_requires_candidate_sizes() -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--execution-commit",
                "a" * 40,
                "--cell-config",
                "cell-a=a.json",
                "--g21-evidence",
                "g21.json",
                "--asset-resolution",
                "assets.json",
                "--data-range",
                "data.json",
                "--data-root",
                "data-root",
                "--output-root",
                "formal-r20-g3-v5/output",
                "--queue-root",
                "queue-root",
                "--runtime-environment",
                "cell-a=environment.json",
            ]
        )


def test_bound_plan_and_amendment_bind_candidate_sizes_and_segment(tmp_path: Path) -> None:
    data_root, environments = _write_bound_sizing_inputs(tmp_path)
    binding = _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))
    assert binding.candidate_sizes == (131072, 262144)
    assert binding.amendment_ref == "amendment.json"
    assert len(binding.amendment_artifact_hash) == 64


def test_queue_manifest_binds_candidate_sizes_and_hash_input() -> None:
    manifest = _queue_manifest(
        run_id="r23-test",
        execution_commit="a" * 40,
        sizing_binding=_SizingBinding(
            candidate_sizes=(131072, 262144),
            amendment_ref="amendment.json",
            amendment_artifact_hash="b" * 64,
            block_size=32,
            segment_start_position=81920,
            segment_end_position_exclusive=344064,
            convergence_tolerance=0.02,
            normalized_l1_threshold=0.02,
            required_consecutive=1,
            complete_all_candidates=True,
            optional_stopping=False,
            resume_ref=None,
            reuse_prior_sizing_prefix=False,
        ),
        order=("cell-a", "cell-b"),
        estimates={"cell-a": 1.0, "cell-b": 1.0},
        cells={"cell-a": Path("a.json"), "cell-b": Path("b.json")},
        python="python",
        output_root=Path("formal-r20-g3-v5/r23-test"),
    )
    assert manifest["candidate_sample_counts"] == [131072, 262144]
    assert manifest["block_size"] == 32
    assert manifest["round_amendment_ref"] == "amendment.json"
    assert manifest["round_amendment_artifact_hash"] == "b" * 64
    assert manifest["segment_start_position"] == 81920
    assert manifest["segment_end_position_exclusive"] == 344064
    assert manifest["convergence_tolerance"] == 0.02
    assert manifest["normalized_l1_threshold"] == 0.02
    assert manifest["required_consecutive"] == 1
    assert manifest["complete_all_candidates"] is True
    assert manifest["optional_stopping"] is False
    assert _hash(manifest) != _hash({key: value for key, value in manifest.items() if key != "candidate_sample_counts"})


def test_queue_manifest_has_one_block_size_key() -> None:
    source = (Path(__file__).parents[1] / "ops/stage2/run_s204_r20_queue.py").read_text(encoding="utf-8")
    assert source.count('"block_size": sizing_binding.block_size') == 1


def test_bound_plan_rejects_candidate_drift(tmp_path: Path) -> None:
    data_root, environments = _write_bound_sizing_inputs(tmp_path, plan_sizes=(32768, 65536))
    with pytest.raises(ValueError, match="candidate sizes drift"):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))


@pytest.mark.parametrize(
    ("override", "field"),
    (
        ({"convergence_tolerance": 0.01}, "convergence contract drift"),
        ({"required_consecutive": 2}, "convergence contract drift"),
        ({"require_terminal_convergence": False}, "convergence contract drift"),
    ),
)
def test_bound_plan_rejects_convergence_contract_drift(
    tmp_path: Path, override: dict[str, object], field: str
) -> None:
    data_root, environments = _write_bound_sizing_inputs(tmp_path, plan_contract_overrides=override)
    with pytest.raises(ValueError, match=field):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))


@pytest.mark.parametrize(
    "override",
    (
        {"normalized_l1_threshold": 0.01},
        {"required_consecutive": 2},
        {"complete_all_candidates": False},
        {"optional_stopping": True},
        {"block_size": 16},
        {"resume_ref": "old-plan.json"},
        {"reuse_prior_sizing_prefix": True},
    ),
)
def test_bound_amendment_rejects_scientific_contract_drift(
    tmp_path: Path, override: dict[str, object]
) -> None:
    data_root, environments = _write_bound_sizing_inputs(
        tmp_path,
        amendment_contract_overrides=override,
    )
    with pytest.raises(ValueError, match="sizing contract drift"):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))


def test_bound_amendment_rejects_candidate_or_segment_drift(tmp_path: Path) -> None:
    data_root, environments = _write_bound_sizing_inputs(tmp_path, amendment_sizes=(32768, 65536))
    with pytest.raises(ValueError, match="candidate_sample_counts"):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))

    data_root, environments = _write_bound_sizing_inputs(tmp_path / "segment", plan_segment=(0, 262144))
    with pytest.raises(ValueError, match="segment drift"):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))

    data_root, environments = _write_bound_sizing_inputs(
        tmp_path / "amendment-segment",
        amendment_segment=(81920, 344064, 16384),
    )
    with pytest.raises(ValueError, match="segment drift"):
        _candidate_sizes_from_bound_plans(data_root, environments, (131072, 262144))


def test_runtime_environment_is_bound_one_to_one_to_cell() -> None:
    values = [
        "cell-a=/tmp/a.json",
        "cell-b=/tmp/b.json",
        "cell-c=/tmp/c.json",
        "cell-d=/tmp/d.json",
        "cell-e=/tmp/e.json",
        "cell-f=/tmp/f.json",
    ]
    cells = _parse_cell_config([f"{cell}=/{cell}.json" for cell in ("cell-a", "cell-b", "cell-c", "cell-d", "cell-e", "cell-f")])
    environments = _parse_cell_environment(values, cells)
    assert set(environments) == set(cells)
    with pytest.raises(ValueError, match="one runtime environment per cell"):
        _parse_cell_environment(values[:-1], cells)


def test_python_path_keeps_venv_symlink_spelling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a venv/bin/python symlink whose target is the system interpreter;
    # this remains portable on hosts where creating symlinks needs privileges.
    venv_python = tmp_path / "venv" / "bin" / "python"
    system_python = tmp_path / "system-python"
    monkeypatch.setattr(Path, "resolve", lambda _self: system_python)
    observed = _absolute_without_resolving(venv_python)
    assert observed == os.path.abspath(os.fspath(venv_python))
    assert observed != os.path.abspath(os.fspath(system_python))
