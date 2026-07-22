"""StageTaskCatalog 静态机器合同的仓库级验收测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from param_importance_nlp.contracts.task_catalog import (
    DEFAULT_TASK_CATALOG,
    RESOLVED_CONFIG_V2_SCHEMA_REF,
    TASK_OUTPUT_ARTIFACT_SCHEMA_REF,
    TASK_OUTPUT_COMMIT_SCHEMA_REF,
    TASK_OUTPUT_PAYLOAD_SCHEMA_REF,
    TaskCatalog,
    TaskCatalogError,
    TaskDefinition,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PREFIX = "https://parameter-importance.invalid/schemas/"


def _load_schema(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert value.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert value.get("type") == "object" or any(
        key in value for key in ("$ref", "allOf", "anyOf", "oneOf")
    )
    return value


def _schema_path(schema_ref: str) -> Path:
    """把项目标准 ``$id`` 映射回仓库文件；任意外部 URL 都立即失败。"""

    assert schema_ref.startswith(SCHEMA_PREFIX), schema_ref
    relative = schema_ref.removeprefix(SCHEMA_PREFIX)
    assert relative and ".." not in Path(relative).parts
    path = ROOT / "schemas" / Path(relative)
    assert path.is_file(), schema_ref
    schema = _load_schema(path)
    assert schema.get("$id") == schema_ref
    return path


def test_all_82_tasks_freeze_existing_standard_schema_refs() -> None:
    """配置、每个输入及每个输出引用都必须落到真实 canonical schema。"""

    assert len(DEFAULT_TASK_CATALOG.tasks) == 82
    observed_refs: set[str] = set()
    for task in DEFAULT_TASK_CATALOG.tasks:
        observed_refs.add(task.config_schema_ref)
        observed_refs.update(item.schema_ref for item in task.input_artifacts)
        observed_refs.update(item.schema_ref for item in task.output_artifacts)
        observed_refs.update(item.payload_schema_ref for item in task.output_artifacts)
    for schema_ref in sorted(observed_refs):
        _schema_path(schema_ref)

    assert RESOLVED_CONFIG_V2_SCHEMA_REF in observed_refs
    assert TASK_OUTPUT_ARTIFACT_SCHEMA_REF in observed_refs
    assert TASK_OUTPUT_PAYLOAD_SCHEMA_REF in observed_refs
    assert TASK_OUTPUT_COMMIT_SCHEMA_REF in observed_refs


def test_output_mapping_exactly_covers_declared_artifact_kinds() -> None:
    for task in DEFAULT_TASK_CATALOG.tasks:
        assert tuple(item.artifact_kind for item in task.output_artifacts) == task.artifact_kinds
        assert len({item.artifact_kind for item in task.output_artifacts}) == len(
            task.output_artifacts
        )
        assert {
            item.payload_schema_ref for item in task.output_artifacts
        } == {TASK_OUTPUT_PAYLOAD_SCHEMA_REF}


def test_generic_output_payload_schema_requires_machine_schema_version() -> None:
    schema = _load_schema(ROOT / "schemas/shared/task-output-payload-v1.json")
    envelope = _load_schema(ROOT / "schemas/shared/task-output-artifact-v1.json")
    assert envelope["properties"]["payload"] == {  # type: ignore[index]
        "$ref": "task-output-payload-v1.json"
    }
    assert schema["required"] == ["schema_version"]
    version = schema["properties"]["schema_version"]  # type: ignore[index]
    assert version["type"] == "string"
    assert version["minLength"] == 1
    assert version["pattern"]


def test_dependency_dag_and_required_input_contracts_are_logically_closed() -> None:
    by_id = {task.task_id: task for task in DEFAULT_TASK_CATALOG.tasks}
    roots = [task.task_id for task in DEFAULT_TASK_CATALOG.tasks if not task.predecessor_task_ids]
    assert roots == ["stage0.01_baseline_and_safety"]

    for task in DEFAULT_TASK_CATALOG.tasks:
        required_producers = {
            producer
            for contract in task.input_artifacts
            if contract.required
            for producer in contract.producer_task_ids
        }
        assert required_producers == set(task.predecessor_task_ids)
        assert bool(task.input_artifacts) == bool(task.predecessor_task_ids)
        assert task.replay_policy.requires_same_input_hashes == bool(task.input_artifacts)
        for producer_id in task.predecessor_task_ids:
            assert producer_id in by_id
            assert by_id[producer_id].stage <= task.stage
        for contract in task.input_artifacts:
            assert len(contract.producer_task_ids) == 1
            producer = by_id[contract.producer_task_ids[0]]
            assert contract.artifact_kinds == producer.artifact_kinds

    # 锁住两个不是简单线性链的关键节点，防止以后“简化”为虚假的单输入任务。
    assert set(by_id["stage4.importance_trajectory"].predecessor_task_ids) == {
        "stage4.pretrain",
        "stage4.direct_supervised",
        "stage4.finetune",
    }
    assert set(by_id["stage9.report"].predecessor_task_ids) == {
        "stage9.tables",
        "stage9.charts",
    }


def test_each_task_freezes_replay_completion_failure_and_fixture_isolation() -> None:
    for task in DEFAULT_TASK_CATALOG.tasks:
        assert task.replay_policy.requires_same_config_hash is True
        assert task.replay_policy.existing_output_policy.value == (
            "reuse_identical_reject_drift"
        )
        assert task.completion_rules
        assert task.failure_rules
        assert not set(task.completion_rules) & set(task.failure_rules)
        assert task.local_fixture.supported is True
        assert task.local_fixture.artifact_formal_eligible is False
        assert task.local_fixture.may_satisfy_formal_gate is False
        assert task.local_fixture.gate_status_on_success == "NOT_RUN"


def test_definition_and_catalog_json_schemas_cover_every_wire_field() -> None:
    definition_schema = _load_schema(ROOT / "schemas/shared/task-definition-v2.json")
    catalog_schema = _load_schema(ROOT / "schemas/shared/task-catalog-v2.json")
    definition_wire = DEFAULT_TASK_CATALOG.tasks[0].to_dict()
    catalog_wire = DEFAULT_TASK_CATALOG.to_dict()
    assert set(definition_schema["required"]) == set(definition_wire)  # type: ignore[arg-type]
    assert set(definition_schema["properties"]) == set(definition_wire)  # type: ignore[arg-type]
    assert set(catalog_schema["required"]) == set(catalog_wire)  # type: ignore[arg-type]
    assert set(catalog_schema["properties"]) == set(catalog_wire)  # type: ignore[arg-type]
    assert catalog_schema["properties"]["tasks"]["items"] == {  # type: ignore[index]
        "$ref": "task-definition-v2.json"
    }


def test_new_nested_contracts_roundtrip_and_unknown_fields_fail_closed() -> None:
    wire = DEFAULT_TASK_CATALOG.to_dict()
    assert TaskCatalog.from_mapping(wire).to_dict() == wire

    # 选择一个有输入的任务，分别向 input/output/replay 子合同注入未知字段。
    definition = DEFAULT_TASK_CATALOG.get("stage9.report").to_dict()
    mutations: list[dict[str, object]] = []
    for section, index in (("input_artifacts", 0), ("output_artifacts", 0)):
        candidate = copy.deepcopy(definition)
        candidate[section][index]["unexpected"] = True  # type: ignore[index]
        mutations.append(candidate)
    candidate = copy.deepcopy(definition)
    candidate["replay_policy"]["unexpected"] = True  # type: ignore[index]
    mutations.append(candidate)
    candidate = copy.deepcopy(definition)
    candidate["local_fixture"]["unexpected"] = True  # type: ignore[index]
    mutations.append(candidate)

    for candidate in mutations:
        with pytest.raises(TaskCatalogError):
            TaskDefinition.from_mapping(candidate)


def test_catalog_rejects_missing_predecessor_and_uncovered_input_edge() -> None:
    wire = DEFAULT_TASK_CATALOG.to_dict()
    task = copy.deepcopy(wire["tasks"][1])  # type: ignore[index]
    task["predecessor_task_ids"] = ["stage0.not_in_catalog"]
    task["input_artifacts"][0]["producer_task_ids"] = ["stage0.not_in_catalog"]  # type: ignore[index]
    tasks = copy.deepcopy(wire["tasks"])
    tasks[1] = task  # type: ignore[index]
    definitions = tuple(TaskDefinition.from_mapping(item) for item in tasks)  # type: ignore[arg-type]
    with pytest.raises(TaskCatalogError, match="目录外任务"):
        TaskCatalog(definitions)

    uncovered = copy.deepcopy(DEFAULT_TASK_CATALOG.get("stage9.report").to_dict())
    uncovered["input_artifacts"] = uncovered["input_artifacts"][:1]  # type: ignore[index]
    with pytest.raises(TaskCatalogError, match="required input artifact"):
        TaskDefinition.from_mapping(uncovered)
