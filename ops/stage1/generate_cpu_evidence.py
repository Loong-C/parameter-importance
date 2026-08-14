"""Stage 1 CPU evidence generator (local_fixture).

执行不依赖 GPU/服务器即可完成的 Stage 1 任务目录 runner（stage1.01–1.06、
1.09–1.11，外加 stage1.07 的本地 fixture 执行链），并把发布的任务输出载荷
归档到 ``reports/stage1/cpu-evidence-<date>/``，同时输出一份机器可读摘要。

正式 gate 不会被本地 fixture 判定为 PASS（``gate_status=NOT_RUN``）；本脚本只
证明 CPU 侧执行链与证据生成可复现。stage1.07 在目录中要求 server/cuda 资产，
本地运行仅作为执行链 fixture，不构成 G1-SINGLE 证据。
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from param_importance_nlp.contracts import (  # noqa: E402
    JSONValue,
    ResolvedConfig,
    load_canonical_json,
    write_canonical_json,
)
from param_importance_nlp.contracts.config_v2 import ResolvedConfigV2  # noqa: E402
from param_importance_nlp.contracts.task_catalog import (  # noqa: E402
    DEFAULT_TASK_CATALOG,
    RunnerKind,
)
from param_importance_nlp.experiments import build_default_task_runtime  # noqa: E402
from param_importance_nlp.runtime import TaskArtifactStore  # noqa: E402


BASE_CONFIG = ROOT / "configs/local-fixtures/resolved-config-v1.json"

# stage1.08 为 DISTRIBUTED_TRAINING，需要 GPU/NCCL，本地不执行。
CPU_TASK_IDS = [
    "stage1.01_entry_and_contract",
    "stage1.02_architecture_and_parameter_registry",
    "stage1.03_fixtures_and_oracles",
    "stage1.04_loss_and_gradient_scale",
    "stage1.05_estimators",
    "stage1.06_training_integration_and_accumulators",
    "stage1.07_single_gpu_pythia14m",
    "stage1.09_precision_clipping_and_optimizer_boundaries",
    "stage1.10_checkpoint_resume_and_artifacts",
    "stage1.11_reporting_and_exit_gate",
]


def _base_config(task_id: str) -> ResolvedConfig:
    task = DEFAULT_TASK_CATALOG.get(task_id)
    value = deepcopy(load_canonical_json(BASE_CONFIG))
    value["identity"]["stage"] = task.stage
    value["identity"]["task"] = task_id
    value["loss"].update({"task_type": "sequence_classification", "weighting": "sample"})
    value["data"].update({"statistical_unit": "sample", "weight_unit": "sample"})
    value["model"]["architecture"] = "tiny-sequence-classifier"
    return ResolvedConfig.from_mapping(value)


def _config(task_id: str, output: str) -> ResolvedConfigV2:
    overrides: dict[str, object] = {
        "providers": {"num_labels": 3},
        "artifacts": {"output_dir": output},
    }
    task = DEFAULT_TASK_CATALOG.get(task_id)
    if task.runner_kind in {RunnerKind.TRAINING}:
        overrides.update(
            {
                "training": {"max_steps": 2},
                "evaluation": {
                    "enabled": True,
                    "split": "validation",
                    "every_steps": 1,
                    "batch_size": 2,
                    "max_batches": 1,
                    "metrics": ["loss", "accuracy"],
                },
                "checkpoint_schedule": {
                    "segments": [{"start_step": 0, "end_step": None, "every_steps": 1}]
                },
            }
        )
    return ResolvedConfigV2.resolve(
        _base_config(task_id),
        task_id=task_id,
        overrides=overrides,
    )


def _load_payload(
    workspace: Path,
    output_dir: str,
    ref: str,
) -> JSONValue:
    published = TaskArtifactStore(workspace, output_dir).load_commit(ref)
    artifact = load_canonical_json(workspace / published.object_ref)
    if not isinstance(artifact, dict):
        raise ValueError("STAGE1_EVIDENCE_ARTIFACT_NOT_OBJECT")
    payload = artifact["payload"]
    if not isinstance(payload, dict):
        raise ValueError("STAGE1_EVIDENCE_PAYLOAD_NOT_OBJECT")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default="20260806",
        help="Evidence directory suffix (YYYYMMDD).",
    )
    args = parser.parse_args()

    evidence_root = ROOT / "reports/stage1" / f"cpu-evidence-{args.date}"
    if evidence_root.exists():
        raise SystemExit(
            f"Refusing to overwrite existing evidence directory: {evidence_root}. "
            "Choose a new --date/attempt suffix."
        )
    evidence_root.mkdir(parents=True)

    summary: dict[str, JSONValue] = {
        "schema_version": "stage1-cpu-evidence-v1",
        "scope": "local_fixture",
        "formal_eligible": False,
        "generator": str(Path(__file__).name),
        "tasks": {},
    }

    with tempfile.TemporaryDirectory(prefix="stage1-cpu-evidence-") as tmp:
        workspace = Path(tmp)
        runtime = build_default_task_runtime(workspace)
        for task_id in CPU_TASK_IDS:
            output_dir = f"runs/{task_id.replace('.', '-')}"
            config = _config(task_id, output_dir)
            result = runtime.execute(config)
            task_evidence: dict[str, JSONValue] = {
                "status": result.status.value,
                "formal_eligible": bool(result.formal_eligible),
                "output_dir": output_dir,
                "artifacts": {},
            }
            for kind, ref in result.artifact_refs.items():
                payload = _load_payload(workspace, output_dir, str(ref))
                target = evidence_root / task_id / f"{kind}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                write_canonical_json(target, payload)
                task_evidence["artifacts"][str(kind)] = str(ref)
            summary["tasks"][task_id] = task_evidence

    write_canonical_json(
        evidence_root / "summary.json",
        summary,
    )
    print(f"WROTE {evidence_root}")
    for task_id, task_evidence in summary["tasks"].items():
        print(f"  {task_id}: {task_evidence['status']} artifacts={len(task_evidence['artifacts'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
