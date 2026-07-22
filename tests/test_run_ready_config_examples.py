"""README/run-ready 声明式模板必须可解析，并在缺正式条件时 fail-closed。

这个测试文件刻意绑定仓库中每一份 ``configs/run-ready/v2`` 模板。这样新增模板时，
如果维护者忘记把它加入严格解析和 preflight 重放，测试会立即失败，而不是等到正式环境
才发现未知字段、错误 task ID 或尚未注册的 runner。
"""

from __future__ import annotations

import json
from pathlib import Path

from param_importance_nlp.cli import _load_mapping, main
from param_importance_nlp.contracts import (
    FormalExecutionEvidence,
    GateRecord,
    GateStatus,
    ResolvedConfig,
    canonical_json_hash,
    load_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2
from param_importance_nlp.contracts.jsonio import write_canonical_json
from param_importance_nlp.experiments import (
    AblationMatrix,
    AblationMatrixDeclaration,
    FormalExperimentPlan,
    SamplingPlan,
    SamplingUniverse,
    TrainingRouteSpec,
    build_default_task_runtime,
)
from param_importance_nlp.runtime import BlockerCode, TaskRunStatus


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/local-fixtures/resolved-config-v1.json"
RUN_READY = ROOT / "configs/run-ready"
LAYERS = RUN_READY / "layers"
V2 = RUN_READY / "v2"
ARTIFACT_EXAMPLES = RUN_READY / "artifacts"


# (task ID, 按顺序合并的 v1 layers, v2 override)。这里列出的文件集合必须与目录完全一致。
LOCAL_TEMPLATE = (
    "stage0.06_single_gpu_smoke",
    ("local-classification-stage0.yaml",),
    "stage0-training-smoke.yaml",
)
FORMAL_TEMPLATES = (
    (
        "stage1.07_single_gpu_pythia14m",
        ("formal-stage1-pythia14m.yaml",),
        "stage1-pythia14m-formal.yaml",
    ),
    (
        "stage2.04_reference_target",
        ("formal-stage2-estimator.yaml",),
        "stage2-reference-formal.yaml",
    ),
    (
        "stage3.07_formal_experiment_matrix",
        ("formal-stage3-path.yaml",),
        "stage3-path-formal.yaml",
    ),
    (
        "stage4.minimal_complete_loop",
        ("formal-pythia160m-stage4.yaml",),
        "stage4-pythia160m-formal.yaml",
    ),
    (
        "stage5.formal_pretraining",
        ("formal-pythia160m-stage4.yaml", "formal-pythia410m-stage5.yaml"),
        "stage5-pythia410m-formal.yaml",
    ),
    (
        "stage6.training_route_comparison",
        (
            "formal-pythia160m-stage4.yaml",
            "formal-glue-stage6.yaml",
            "formal-glue-sst2-stage6.yaml",
        ),
        "stage6-sst2-formal.yaml",
    ),
    (
        "stage6.training_route_comparison",
        (
            "formal-pythia160m-stage4.yaml",
            "formal-glue-stage6.yaml",
            "formal-glue-mnli-stage6.yaml",
        ),
        "stage6-mnli-formal.yaml",
    ),
    (
        "stage6.training_route_comparison",
        (
            "formal-pythia160m-stage4.yaml",
            "formal-glue-stage6.yaml",
            "formal-glue-rte-stage6.yaml",
        ),
        "stage6-rte-formal.yaml",
    ),
    *(
        (
            task_id,
            ("formal-pythia160m-stage4.yaml", stage_layer),
            override,
        )
        for task_id, stage_layer, override in (
            ("stage7.matrix", "formal-stage7.yaml", "stage7-matrix-formal.yaml"),
            ("stage7.evaluate", "formal-stage7.yaml", "stage7-evaluate-formal.yaml"),
            ("stage7.reduce", "formal-stage7.yaml", "stage7-reduce-formal.yaml"),
            ("stage7.report", "formal-stage7.yaml", "stage7-report-formal.yaml"),
            ("stage8.freeze", "formal-stage8.yaml", "stage8-freeze-formal.yaml"),
            ("stage8.execute", "formal-stage8.yaml", "stage8-execute-formal.yaml"),
            ("stage8.reduce", "formal-stage8.yaml", "stage8-reduce-formal.yaml"),
            ("stage8.recommend", "formal-stage8.yaml", "stage8-recommend-formal.yaml"),
            ("stage8.report", "formal-stage8.yaml", "stage8-report-formal.yaml"),
            ("stage9.ingest", "formal-stage9.yaml", "stage9-ingest-formal.yaml"),
            (
                "stage9.statistics",
                "formal-stage9.yaml",
                "stage9-statistics-formal.yaml",
            ),
            ("stage9.tables", "formal-stage9.yaml", "stage9-tables-formal.yaml"),
            ("stage9.charts", "formal-stage9.yaml", "stage9-charts-formal.yaml"),
            ("stage9.report", "formal-stage9.yaml", "stage9-report-formal.yaml"),
            ("stage9.bundle", "formal-stage9.yaml", "stage9-bundle-formal.yaml"),
            ("stage9.replay", "formal-stage9.yaml", "stage9-replay-formal.yaml"),
        )
    ),
)


def _resolve(task_id: str, layers: tuple[str, ...], overrides: str) -> ResolvedConfigV2:
    base = ResolvedConfig.resolve(
        _load_mapping(BASE),
        *(_load_mapping(LAYERS / name) for name in layers),
    )
    return ResolvedConfigV2.resolve(
        base,
        task_id=task_id,
        overrides=_load_mapping(V2 / overrides),
    )


def _fixture_decision() -> dict[str, object]:
    """生成只用于编译本地 route 示例的、明确不具 formal 资格的 decision。"""

    payload: dict[str, object] = {
        "schema_version": "estimator-decision-v1",
        "decision_id": "run-ready-example-fixture",
        "selected_estimator": "u",
        "scope": "local_fixture",
        "status": "FIXTURE_ONLY",
        "state": "UNFROZEN",
        "batch_size": 32,
        "microbatch_count": 2,
        "repetitions": 2,
        "gate_id": "stage2.G2.7b",
        "gate_status": "NOT_RUN",
        "artifact_ref": None,
        "metadata": {"formal_eligible": False},
    }
    payload["artifact_hash"] = canonical_json_hash(payload)
    return payload


def test_every_v2_template_is_bound_to_the_strict_replay_matrix() -> None:
    """防止新增 YAML 后绕过 resolve/preflight 回归测试。"""

    expected = {LOCAL_TEMPLATE[2], *(case[2] for case in FORMAL_TEMPLATES)}
    observed = {path.name for path in V2.glob("*.yaml")}
    assert observed == expected


def test_local_training_example_is_executable_fixture_config() -> None:
    config = _resolve(*LOCAL_TEMPLATE)
    assert config.run_intent == "local_fixture"
    assert config.section("providers")["task_name"] == "fixture"  # type: ignore[index]
    assert ResolvedConfigV2.from_mapping(config.to_dict()).full_hash == config.full_hash


def test_all_formal_templates_resolve_and_fail_closed_as_structured_blocked(
    tmp_path: Path,
) -> None:
    """重放所有正式模板；空本机环境只能得到 ``BLOCKED``，不能抛异常或误报 PASS。"""

    runtime = build_default_task_runtime(tmp_path)
    for task_id, layers, override in FORMAL_TEMPLATES:
        config = _resolve(task_id, layers, override)
        roundtrip = ResolvedConfigV2.from_mapping(config.to_dict())
        assert roundtrip.full_hash == config.full_hash, override
        assert config.task_id == task_id
        assert config.run_intent == "formal"

        blockers = runtime.preflight(config)
        assert blockers, override
        assert all(blocker.code is not BlockerCode.RUNNER_UNAVAILABLE for blocker in blockers)
        assert all(blocker.requirement != "runner" for blocker in blockers)

        result = runtime.execute(config)
        assert result.status is TaskRunStatus.BLOCKED, override
        assert result.formal_eligible is False
        assert result.blockers == blockers


def test_run_ready_source_examples_compile_with_their_public_schemas(
    tmp_path: Path,
) -> None:
    """实际编译仓库内 source 示例，避免 README 示例与 builder/schema 漂移。"""

    expected_examples = {
        "training-route-source.example.yaml",
        "ablation-matrix-source.example.yaml",
        "stage2-formal-experiment-plan-source.example.yaml",
        "stage3-formal-pilot-plan-source.example.yaml",
    }
    assert {path.name for path in ARTIFACT_EXAMPLES.glob("*.yaml")} == expected_examples

    decision_path = tmp_path / "decision.json"
    route_path = tmp_path / "route.json"
    declaration_path = tmp_path / "ablation-declaration.json"
    matrix_path = tmp_path / "ablation-matrix.json"
    evidence_path = tmp_path / "formal-execution-evidence.json"
    sampling_path = tmp_path / "sampling-plan.json"
    stage2_plan_path = tmp_path / "stage2-formal-experiment-plan.json"
    pilot_plan_path = tmp_path / "stage3-formal-pilot-plan.json"
    write_canonical_json(decision_path, _fixture_decision())
    evidence = FormalExecutionEvidence(
        "formal",
        contract_freeze_hash="1" * 64,
        asset_manifest_hashes=("2" * 64,),
        prerequisite_gates=(
            GateRecord(
                "stage0.G10",
                0,
                GateStatus.PASS,
                "2026-07-22T00:00:00Z",
                evidence_refs=("commits/stage0-g10.json",),
            ),
        ),
    )
    write_canonical_json(evidence_path, evidence.to_dict())
    sampling = SamplingPlan(
        SamplingUniverse("run-ready-stage2-example", tuple(range(64))),
        {
            "reference_sizing": 11,
            "reference_A": 22,
            "reference_B": 33,
            "pilot": 44,
            "confirmatory": 55,
        },
    )
    write_canonical_json(sampling_path, sampling.to_dict())

    assert main(
        [
            "artifact",
            "route-build",
            "--spec",
            str(ARTIFACT_EXAMPLES / "training-route-source.example.yaml"),
            "--decision",
            str(decision_path),
            "--output",
            str(route_path),
        ]
    ) == 0
    assert main(
        [
            "artifact",
            "stage2-experiment-plan-build",
            "--spec",
            str(
                ARTIFACT_EXAMPLES
                / "stage2-formal-experiment-plan-source.example.yaml"
            ),
            "--sampling-plan",
            str(sampling_path),
            "--formal-execution-evidence",
            str(evidence_path),
            "--source-ref",
            "runs/stage2-05/commits/paired-runner-report.json",
            "--source-ref",
            "runs/stage2-05/commits/sufficient-stat-shards.json",
            "--source-ref",
            "runs/stage2-05/commits/gate-record.json",
            "--output",
            str(stage2_plan_path),
        ]
    ) == 0
    assert main(
        [
            "artifact",
            "quadrature-pilot-plan-build",
            "--spec",
            str(
                ARTIFACT_EXAMPLES
                / "stage3-formal-pilot-plan-source.example.yaml"
            ),
            "--formal-execution-evidence",
            str(evidence_path),
            "--output",
            str(pilot_plan_path),
        ]
    ) == 0
    assert main(
        [
            "artifact",
            "ablation-matrix-build",
            "--spec",
            str(ARTIFACT_EXAMPLES / "ablation-matrix-source.example.yaml"),
            "--output",
            str(declaration_path),
            "--compiled-output",
            str(matrix_path),
        ]
    ) == 0

    assert TrainingRouteSpec.from_mapping(load_canonical_json(route_path)).run_intent == (
        "local_fixture"
    )
    declaration = AblationMatrixDeclaration.from_mapping(
        load_canonical_json(declaration_path)
    )
    matrix = AblationMatrix.from_mapping(load_canonical_json(matrix_path))
    assert declaration.compile().to_dict() == matrix.to_dict()
    pilot_plan = load_canonical_json(pilot_plan_path)
    assert pilot_plan["execution_evidence_hash"] == evidence.artifact_hash
    stage2_plan = FormalExperimentPlan.from_mapping(
        load_canonical_json(stage2_plan_path)
    )
    assert stage2_plan.sampling_plan_hash == sampling.digest
    assert stage2_plan.execution_evidence_hash == evidence.artifact_hash

    # source schema 本身也必须继续满足项目的严格 JSON Schema 元合同。
    for name in (
        "training-route-source-v1.json",
        "ablation-matrix-source-v1.json",
        "stage2-formal-experiment-plan-source-v1.json",
        "stage2-formal-experiment-plan-v1.json",
        "stage3-formal-pilot-plan-source-v1.json",
        "stage3-formal-pilot-plan-v1.json",
    ):
        schema_path = ROOT / "schemas/shared" / name
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        assert schema["additionalProperties"] is False
        assert main(["contract-validate", str(schema_path)]) == 0
