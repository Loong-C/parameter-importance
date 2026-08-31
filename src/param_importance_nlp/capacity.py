from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
from typing import Iterable


GIB = 1024**3


@dataclass(frozen=True, slots=True)
class StorageBudget:
    name: str
    expected_new_bytes: int
    safety_margin_bytes: int
    required_free_bytes: int

    @classmethod
    def from_expected(cls, name: str, expected_new_bytes: int) -> "StorageBudget":
        if expected_new_bytes < 0:
            raise ValueError("expected_new_bytes cannot be negative")
        margin = max((expected_new_bytes + 4) // 5, 100 * GIB)
        return cls(name, expected_new_bytes, margin, expected_new_bytes + margin)

    def as_dict(self) -> dict[str, str | int]:
        return asdict(self)


def check_storage_budget(path: str | Path, budget: StorageBudget) -> dict[str, int | bool]:
    usage = shutil.disk_usage(Path(path))
    return {
        "free_bytes": usage.free,
        "required_free_bytes": budget.required_free_bytes,
        "ok": usage.free >= budget.required_free_bytes,
    }


def check_launch_storage(
    *,
    data_root: str | Path,
    root_filesystem: str | Path,
    budget: StorageBudget,
    root_minimum_free_bytes: int = 10 * GIB,
) -> dict[str, int | bool | None]:
    data_usage = shutil.disk_usage(Path(data_root))
    root_usage = shutil.disk_usage(Path(root_filesystem))
    inode_free: int | None = None
    inode_total: int | None = None
    if hasattr(os, "statvfs"):
        stat = os.statvfs(Path(data_root))
        inode_free = stat.f_favail
        inode_total = stat.f_files
    data_ok = data_usage.free >= budget.required_free_bytes
    root_ok = root_usage.free >= root_minimum_free_bytes
    inode_ok = inode_free is None or inode_free > 0
    return {
        "data_free_bytes": data_usage.free,
        "data_required_free_bytes": budget.required_free_bytes,
        "root_free_bytes": root_usage.free,
        "root_minimum_free_bytes": root_minimum_free_bytes,
        "inode_free": inode_free,
        "inode_total": inode_total,
        "data_ok": data_ok,
        "root_ok": root_ok,
        "inode_ok": inode_ok,
        "ok": data_ok and root_ok and inode_ok,
    }


def estimate_checkpoint_bytes(parameter_count: int, *, safety_factor: float = 1.25) -> int:
    """Conservative BF16 model + FP32 Adam states/RNG/metadata estimate."""

    if parameter_count <= 0:
        raise ValueError("parameter_count must be positive")
    base_bytes_per_parameter = 2 + 4 + 4 + 4
    return int(parameter_count * base_bytes_per_parameter * safety_factor)


def estimate_parameter_statistics_bytes(
    parameter_count: int,
    *,
    resident_fp32_buffers: int,
    transient_fp32_buffers: int = 2,
) -> int:
    if parameter_count <= 0 or resident_fp32_buffers < 0 or transient_fp32_buffers < 0:
        raise ValueError("invalid parameter statistics estimate")
    return parameter_count * 4 * (resident_fp32_buffers + transient_fp32_buffers)


def estimate_experiment_storage(
    *,
    parameter_count: int,
    retained_checkpoints: int,
    resident_fp32_buffers: int,
    seed_count: int,
    parallel_runs: int,
    logs_and_reports_per_run: int,
) -> int:
    values: Iterable[int] = (
        estimate_checkpoint_bytes(parameter_count) * retained_checkpoints,
        estimate_parameter_statistics_bytes(
            parameter_count, resident_fp32_buffers=resident_fp32_buffers
        ),
        logs_and_reports_per_run,
    )
    per_run = sum(values)
    return per_run * seed_count * parallel_runs
