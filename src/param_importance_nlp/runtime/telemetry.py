"""训练资源采样与 JSONL→TensorBoard 可选重建。

JSONL 事件始终是机器真值；TensorBoard 只是可删除、可重建的派生视图。本模块不在
导入时加载 TensorBoard，也不把墙钟时间混入模型、梯度或重要性状态。资源采样仅在
严格 v2 ``profiling.enabled=true`` 时由 runner 显式启用。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
import tracemalloc
from typing import Iterable, Mapping

from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)
from ..providers.optional import load_tensorboard_module


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    """一段训练区间的只读资源测量。

    单位固定为秒、step/秒、统计单位/秒和 byte。无法在当前设备获得的字段使用
    ``None``，绝不以 0 伪装成有效测量。
    """

    wall_seconds: float
    process_cpu_seconds: float
    completed_steps: int
    effective_units: int
    steps_per_second: float | None
    units_per_second: float | None
    python_peak_bytes: int | None

    def to_dict(self) -> dict[str, JSONValue]:
        payload: dict[str, JSONValue] = {
            "schema_version": "training-resource-profile-v1",
            "wall_seconds": self.wall_seconds,
            "process_cpu_seconds": self.process_cpu_seconds,
            "completed_steps": self.completed_steps,
            "effective_units": self.effective_units,
            "steps_per_second": self.steps_per_second,
            "units_per_second": self.units_per_second,
            "python_peak_bytes": self.python_peak_bytes,
            "memory_scope": (
                "python_tracemalloc" if self.python_peak_bytes is not None else None
            ),
        }
        payload["profile_hash"] = canonical_json_hash(payload)
        return payload


class ResourceSampler:
    """显式生命周期的本机资源采样器。"""

    def __init__(self, *, capture_memory: bool) -> None:
        if type(capture_memory) is not bool:
            raise TypeError("capture_memory 必须是 bool")
        self.capture_memory = capture_memory
        self._started = False
        self._wall_start = 0.0
        self._cpu_start = 0.0

    def start(self) -> None:
        if self._started:
            raise RuntimeError("RESOURCE_SAMPLER_ALREADY_STARTED")
        if self.capture_memory:
            tracemalloc.start()
            tracemalloc.reset_peak()
        self._wall_start = time.perf_counter()
        self._cpu_start = time.process_time()
        self._started = True

    def stop(self, *, completed_steps: int, effective_units: int) -> ResourceProfile:
        if not self._started:
            raise RuntimeError("RESOURCE_SAMPLER_NOT_STARTED")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (completed_steps, effective_units)
        ):
            raise ValueError("RESOURCE_PROFILE_COUNTS_INVALID")
        wall = max(time.perf_counter() - self._wall_start, 0.0)
        cpu = max(time.process_time() - self._cpu_start, 0.0)
        peak: int | None = None
        if self.capture_memory:
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        self._started = False
        return ResourceProfile(
            wall_seconds=wall,
            process_cpu_seconds=cpu,
            completed_steps=completed_steps,
            effective_units=effective_units,
            steps_per_second=(completed_steps / wall if wall > 0 and completed_steps else None),
            units_per_second=(effective_units / wall if wall > 0 and effective_units else None),
            python_peak_bytes=peak,
        )


def synchronize_profile_device(device_name: str, *, enabled: bool) -> None:
    """在测量边界按配置同步设备。

    CPU 没有需要显式同步的异步 kernel，因此这里不会伪造同步动作。CUDA 仅在
    调用方明确启用且本机确实可用时同步；缺少 CUDA 会直接失败，而不是把一段
    不可比较的墙钟时间发布成正式证据。
    """

    if not enabled:
        return
    import torch

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PROFILE_CUDA_SYNCHRONIZE_WITHOUT_CUDA")
        torch.cuda.synchronize(device)


def _validate_resource_window(value: object) -> dict[str, JSONValue]:
    """复算资源窗口及其内层 profile hash。"""

    expected = {
        "schema_version",
        "repetition",
        "start_step",
        "end_step",
        "requested",
        "profile",
        "communication",
        "artifact_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("TRAINING_RESOURCE_WINDOW_FIELDS_INVALID")
    if value.get("schema_version") != "training-resource-window-v1":
        raise ValueError("TRAINING_RESOURCE_WINDOW_VERSION_INVALID")
    declared = value.get("artifact_hash")
    body = {key: item for key, item in value.items() if key != "artifact_hash"}
    if not isinstance(declared, str) or declared != canonical_json_hash(body):
        raise ValueError("TRAINING_RESOURCE_WINDOW_HASH_MISMATCH")
    profile = value.get("profile")
    if not isinstance(profile, Mapping) or profile.get("schema_version") != (
        "training-resource-profile-v1"
    ):
        raise ValueError("TRAINING_RESOURCE_PROFILE_INVALID")
    profile_declared = profile.get("profile_hash")
    profile_body = {key: item for key, item in profile.items() if key != "profile_hash"}
    if not isinstance(profile_declared, str) or profile_declared != canonical_json_hash(
        profile_body
    ):
        raise ValueError("TRAINING_RESOURCE_PROFILE_HASH_MISMATCH")
    return dict(value)  # type: ignore[return-value]


class ResourceWindowStore:
    """以不可变对象加独立 commit 保存训练测量窗口。

    ``root`` 必须已经位于仓库工作区内。对象文件名是窗口内容 hash；权威 commit
    只保存 repetition、对象引用和两层 hash。恢复时只沿 commit 发现对象，并重新
    验证工作区边界、目录布局与全部 hash，目录中孤立对象不会被当成可用结果。
    """

    def __init__(self, workspace_root: str | Path, root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.root = Path(root).resolve()
        try:
            self.root.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("TRAINING_RESOURCE_WINDOW_ROOT_OUTSIDE_WORKSPACE") from error
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)

    def commit_path(self, repetition: int) -> Path:
        if isinstance(repetition, bool) or not isinstance(repetition, int) or repetition < 0:
            raise ValueError("TRAINING_RESOURCE_WINDOW_REPETITION_INVALID")
        return self.commits / f"window-{repetition:04d}.json"

    def load(self, repetition: int) -> dict[str, JSONValue]:
        """沿权威 commit 读取并验证一个窗口。"""

        commit_path = self.commit_path(repetition)
        commit = load_canonical_json(commit_path)
        expected = {
            "schema_version",
            "repetition",
            "artifact_hash",
            "object_ref",
            "commit_hash",
        }
        if not isinstance(commit, Mapping) or set(commit) != expected:
            raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_FIELDS_INVALID")
        if commit.get("schema_version") != "training-resource-window-commit-v1":
            raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_VERSION_INVALID")
        declared = commit.get("commit_hash")
        commit_body = {key: item for key, item in commit.items() if key != "commit_hash"}
        if not isinstance(declared, str) or declared != canonical_json_hash(commit_body):
            raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_HASH_MISMATCH")
        if commit.get("repetition") != repetition:
            raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_REPETITION_MISMATCH")
        object_ref = commit.get("object_ref")
        if not isinstance(object_ref, str) or "\\" in object_ref:
            raise TypeError("TRAINING_RESOURCE_WINDOW_OBJECT_REF_INVALID")
        object_path = (self.workspace_root / object_ref).resolve()
        try:
            object_path.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("TRAINING_RESOURCE_WINDOW_OBJECT_REF_ESCAPE") from error
        if object_path.parent != self.objects:
            raise ValueError("TRAINING_RESOURCE_WINDOW_OBJECT_LAYOUT_INVALID")
        value = _validate_resource_window(load_canonical_json(object_path))
        if value.get("artifact_hash") != commit.get("artifact_hash"):
            raise ValueError("TRAINING_RESOURCE_WINDOW_COMMIT_OBJECT_MISMATCH")
        return value

    def publish(
        self,
        *,
        repetition: int,
        start_step: int,
        end_step: int,
        requested: Mapping[str, JSONValue],
        profile: ResourceProfile,
        capture_throughput: bool,
        capture_communication: bool,
    ) -> dict[str, JSONValue]:
        """发布一个窗口；同 repetition 只能幂等指向完全相同的对象。"""

        profile_payload = profile.to_dict()
        if not capture_throughput:
            profile_payload["steps_per_second"] = None
            profile_payload["units_per_second"] = None
            profile_payload["profile_hash"] = canonical_json_hash(
                {
                    key: item
                    for key, item in profile_payload.items()
                    if key != "profile_hash"
                }
            )
        communication: dict[str, JSONValue] = {
            "requested": capture_communication,
            "defined": False,
            "bytes": None,
            "reason": (
                "backend_exact_communication_counter_unavailable"
                if capture_communication
                else "not_requested"
            ),
        }
        body: dict[str, JSONValue] = {
            "schema_version": "training-resource-window-v1",
            "repetition": repetition,
            "start_step": start_step,
            "end_step": end_step,
            "requested": dict(requested),
            "profile": profile_payload,
            "communication": communication,
        }
        body["artifact_hash"] = canonical_json_hash(body)
        object_path = self.objects / f"{body['artifact_hash']}.json"
        if object_path.exists():
            if _validate_resource_window(load_canonical_json(object_path)) != body:
                raise RuntimeError("TRAINING_RESOURCE_WINDOW_OBJECT_DRIFT")
        else:
            write_canonical_json(object_path, body)

        object_ref = object_path.relative_to(self.workspace_root).as_posix()
        commit_body: dict[str, JSONValue] = {
            "schema_version": "training-resource-window-commit-v1",
            "repetition": repetition,
            "artifact_hash": str(body["artifact_hash"]),
            "object_ref": object_ref,
        }
        commit_body["commit_hash"] = canonical_json_hash(commit_body)
        commit_path = self.commit_path(repetition)
        if commit_path.exists():
            if load_canonical_json(commit_path) != commit_body:
                raise FileExistsError("TRAINING_RESOURCE_WINDOW_COMMIT_CONFLICT")
        else:
            write_canonical_json(commit_path, commit_body)
        return self.load(repetition)


def rebuild_tensorboard_from_jsonl(
    event_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> int:
    """从已提交 JSONL 事件重建 TensorBoard scalar 视图。

    只采集 payload 中的有限 int/float 标量，step 使用事件的 ``sequence``。未知
    事件字段被忽略；损坏 JSON、非 object 行或缺 sequence 会 fail-closed。

    Returns:
        写入的 scalar 数量。
    """

    tensorboard = load_tensorboard_module()
    writer_class = getattr(
        getattr(getattr(tensorboard, "summary", None), "writer", None),
        "SummaryWriter",
        None,
    )
    if writer_class is None:
        # 常规 pip 包的公开 writer 位于 torch.utils.tensorboard；延迟导入仍只在
        # 用户显式请求重建时发生。
        from torch.utils.tensorboard import SummaryWriter as writer_class  # type: ignore[assignment]

    count = 0
    writer = writer_class(log_dir=str(Path(output_dir)))
    try:
        for raw_path in event_paths:
            path = Path(raw_path)
            with path.open("r", encoding="utf-8", newline="") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"TENSORBOARD_REBUILD_JSON_INVALID:{path}:{line_number}"
                        ) from error
                    if not isinstance(event, Mapping):
                        raise ValueError("TENSORBOARD_REBUILD_EVENT_NOT_OBJECT")
                    sequence = event.get("sequence")
                    payload = event.get("payload")
                    event_type = event.get("event_type")
                    if (
                        isinstance(sequence, bool)
                        or not isinstance(sequence, int)
                        or not isinstance(payload, Mapping)
                        or not isinstance(event_type, str)
                    ):
                        raise ValueError("TENSORBOARD_REBUILD_EVENT_FIELDS_INVALID")
                    for key, value in sorted(payload.items()):
                        if isinstance(value, bool) or not isinstance(value, (int, float)):
                            continue
                        numeric = float(value)
                        if not (-float("inf") < numeric < float("inf")):
                            raise ValueError("TENSORBOARD_REBUILD_NONFINITE_SCALAR")
                        writer.add_scalar(f"{event_type}/{key}", numeric, sequence)
                        count += 1
    finally:
        writer.close()
    return count


__all__ = [
    "ResourceProfile",
    "ResourceSampler",
    "ResourceWindowStore",
    "rebuild_tensorboard_from_jsonl",
    "synchronize_profile_device",
]
