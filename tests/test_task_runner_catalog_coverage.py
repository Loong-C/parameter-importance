"""证明 82 个 task 都有确定的源码执行路径，而不是落入临时脚本。"""

from __future__ import annotations

from param_importance_nlp.contracts.task_catalog import (
    DEFAULT_TASK_CATALOG,
    RunnerKind,
)
from param_importance_nlp.experiments.stage01_task_runners import (
    STAGE01_HANDLED_TASK_IDS,
)
from param_importance_nlp.experiments.stage23_task_runners import (
    _STAGE23_TASK_ORDER,
)
from param_importance_nlp.experiments.stage456_task_runners import (
    _TASK_IDS_BY_KIND as STAGE456_TASK_IDS_BY_KIND,
)
from param_importance_nlp.experiments.stage789_task_runners import (
    _HANDLED_BY_KIND as STAGE789_TASK_IDS_BY_KIND,
)


def _flatten(values):
    return {task_id for group in values.values() for task_id in group}


def test_every_catalog_task_has_exactly_one_concrete_dispatch_owner() -> None:
    """锁住专用 runner 与通用训练 runner 的完整、不重叠分工。"""

    stage01 = set(STAGE01_HANDLED_TASK_IDS)
    stage23 = set(_STAGE23_TASK_ORDER)
    stage456 = _flatten(STAGE456_TASK_IDS_BY_KIND)
    stage789 = _flatten(STAGE789_TASK_IDS_BY_KIND)
    generic_training = {
        task.task_id
        for task in DEFAULT_TASK_CATALOG.tasks
        if task.runner_kind
        in {RunnerKind.TRAINING, RunnerKind.DISTRIBUTED_TRAINING}
    }
    owners = (stage01, stage23, stage456, stage789, generic_training)

    for index, left in enumerate(owners):
        for right in owners[index + 1 :]:
            assert left.isdisjoint(right)

    catalog_ids = {task.task_id for task in DEFAULT_TASK_CATALOG.tasks}
    assert set().union(*owners) == catalog_ids
    assert generic_training == {
        "stage0.06_single_gpu_smoke",
        "stage0.07_ddp_and_gradient_semantics",
        "stage1.06_training_integration_and_accumulators",
        "stage1.07_single_gpu_pythia14m",
        "stage1.08_ddp_and_gradient_accumulation",
    }


def test_every_dispatch_owner_is_bound_to_the_catalog_runner_kind() -> None:
    """防止 task ID 仍存在、却被登记到错误生命周期 runner。"""

    expected_by_task = {
        task_id: kind
        for kind, task_ids in STAGE456_TASK_IDS_BY_KIND.items()
        for task_id in task_ids
    }
    expected_by_task.update(
        {
            task_id: kind
            for kind, task_ids in STAGE789_TASK_IDS_BY_KIND.items()
            for task_id in task_ids
        }
    )
    for task_id, runner_kind in expected_by_task.items():
        assert DEFAULT_TASK_CATALOG.get(task_id).runner_kind is runner_kind
