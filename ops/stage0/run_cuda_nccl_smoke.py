#!/usr/bin/env python3
"""Generate the current-boot CUDA/NCCL smoke evidence for G0-G/G2 promotion.

This runner reproduces the exact evidence layout consumed by
``promote_environment_after_gpu_gate.py``:

* per-GPU FP32 vector sum and 256x256 matmul on each of the four approved GPUs;
* four-rank NCCL all-reduce of 1,048,576 FP32 elements with three iterations;
* NVML/row-remapper/compute-app snapshots before and after;
* kernel journal delta plus a critical-pattern scan;
* a ``SUCCESS`` marker bound to the current boot and a ``SHA256SUMS`` manifest.

It is fail-closed: any identity, runtime, numeric, activity, or kernel-journal
deviation aborts before publishing a PASS marker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


EXPECTED_HOST = "sophgo13"
EXPECTED_BDFS = [
    "00000000:9C:00.0",
    "00000000:9D:00.0",
    "00000000:A0:00.0",
    "00000000:A4:00.0",
]
EXPECTED_UUIDS = [
    "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267",
    "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d",
    "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f",
    "GPU-5a81500d-5e9c-b0d7-5607-fdfdaab65ff4",
]
EXPECTED_TORCH = "2.12.1+cu126"
EXPECTED_CUDA_RUNTIME = "12.6"
EXPECTED_CUDNN = 91002
EXPECTED_NCCL = (2, 29, 3)
EXPECTED_VECTOR_SUM = 16777216.0
EXPECTED_MATMUL_VALUE = 256.0
EXPECTED_ALLREDUCE_SUM = 10.0
MESSAGE_ELEMENTS = 1048576
COLLECTIVES = 3

CRITICAL_PATTERN = re.compile(
    r"NVRM:.*Xid|RmInitAdapter|rm_init_adapter|AER:|PCIe Bus Error"
    r"|GPU has fallen off the bus|GSP.*(fail|fatal)",
    re.IGNORECASE,
)


class SmokeError(RuntimeError):
    """Fail-closed smoke validation error."""


def canonical_uuid(value: Any) -> str:
    """Return the canonical ``GPU-`` prefixed lowercase UUID."""
    if isinstance(value, bytes):
        value = value.decode("ascii")
    value = str(value or "")
    if value.startswith("GPU-"):
        value = value[4:]
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise SmokeError(f"invalid GPU UUID: {value!r}")
    return "GPU-" + value.lower()


def short_uuid(value: Any) -> str:
    """Return the UUID without the ``GPU-`` prefix (evidence storage form)."""
    return canonical_uuid(value)[4:]


def run_command(argv: list[str], timeout: int = 120) -> str:
    try:
        completed = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SmokeError(f"command failed to run: {argv!r}: {error}") from error
    if completed.returncode != 0:
        raise SmokeError(
            f"command exited {completed.returncode}: {argv!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_boot_id() -> str:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    if not re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        boot_id,
    ):
        raise SmokeError(f"invalid boot ID: {boot_id!r}")
    return boot_id


def journal_cursor() -> str:
    output = run_command(["journalctl", "-k", "-n", "0", "--show-cursor", "--no-pager"])
    match = re.search(r"^-- cursor: (.+)$", output, re.MULTILINE)
    if not match:
        raise SmokeError("journalctl did not return a cursor")
    return match.group(1).strip()


def nvml_csv() -> str:
    return run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid,"
            "ecc.errors.uncorrected.volatile.total,"
            "ecc.errors.uncorrected.aggregate.total,"
            "memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )


def row_remapper() -> str:
    return run_command(["nvidia-smi", "-q", "-d", "ROW_REMAPPER"], timeout=30)


def compute_apps() -> str:
    return run_command(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        timeout=30,
    )


def assert_no_compute_apps(label: str) -> None:
    apps = compute_apps()
    if apps.strip():
        raise SmokeError(f"compute application detected {label}:\n{apps}")


def validate_nvml_rows(csv_text: str, label: str) -> None:
    rows = [line for line in csv_text.splitlines() if line.strip()]
    if len(rows) != 4:
        raise SmokeError(f"{label}: expected 4 NVML rows, got {len(rows)}")
    for index, row in enumerate(rows):
        fields = [item.strip() for item in row.split(",")]
        if len(fields) != 8:
            raise SmokeError(f"{label}: malformed NVML row: {row!r}")
        if fields[1].upper() != EXPECTED_BDFS[index]:
            raise SmokeError(f"{label}: PCI mismatch: {fields[1]!r}")
        if canonical_uuid(fields[2]) != EXPECTED_UUIDS[index]:
            raise SmokeError(f"{label}: UUID mismatch: {fields[2]!r}")
        if fields[3] != "0" or fields[4] != "0":
            raise SmokeError(f"{label}: uncorrectable ECC nonzero: {row!r}")
        if fields[5] != "0" or fields[6] != "0":
            raise SmokeError(f"{label}: unexpected GPU activity: {row!r}")


def per_gpu_smoke() -> dict[str, Any]:
    if torch.cuda.device_count() != 4:
        raise SmokeError(f"expected 4 CUDA devices, got {torch.cuda.device_count()}")
    devices: list[dict[str, Any]] = []
    for index in range(4):
        properties = torch.cuda.get_device_properties(index)
        uuid = canonical_uuid(getattr(properties, "uuid", ""))
        if uuid != EXPECTED_UUIDS[index]:
            raise SmokeError(f"device {index} UUID mismatch: {uuid!r}")
        started = time.perf_counter()
        vector = torch.full(
            (MESSAGE_ELEMENTS,),
            16.0,
            device=f"cuda:{index}",
            dtype=torch.float32,
        )
        vector_sum = float(vector.sum().item())
        left = torch.full((256, 256), 1.0, device=f"cuda:{index}", dtype=torch.float32)
        right = torch.full((256, 256), 1.0, device=f"cuda:{index}", dtype=torch.float32)
        product = left @ right
        torch.cuda.synchronize(index)
        elapsed = time.perf_counter() - started
        matmul_min = float(product.min().item())
        matmul_max = float(product.max().item())
        if vector_sum != EXPECTED_VECTOR_SUM:
            raise SmokeError(f"device {index}: vector sum {vector_sum}")
        if matmul_min != EXPECTED_MATMUL_VALUE or matmul_max != EXPECTED_MATMUL_VALUE:
            raise SmokeError(f"device {index}: matmul range {matmul_min}/{matmul_max}")
        devices.append(
            {
                "device": index,
                "elapsed_seconds": elapsed,
                "matmul_max": matmul_max,
                "matmul_min": matmul_min,
                "name": properties.name,
                "uuid": short_uuid(uuid),
                "vector_sum": vector_sum,
            }
        )
    torch.cuda.empty_cache()
    return {
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_count": 4,
        "devices": devices,
        "nccl": list(torch.cuda.nccl.version()),
        "status": "PASS",
        "torch": torch.__version__,
    }


def _validate_expected_runtime() -> None:
    if torch.__version__ != EXPECTED_TORCH:
        raise SmokeError(f"torch version mismatch: {torch.__version__!r}")
    if torch.version.cuda != EXPECTED_CUDA_RUNTIME:
        raise SmokeError(f"CUDA runtime mismatch: {torch.version.cuda!r}")
    if torch.backends.cudnn.version() != EXPECTED_CUDNN:
        raise SmokeError(
            f"cuDNN version mismatch: {torch.backends.cudnn.version()!r}"
        )
    if tuple(torch.cuda.nccl.version()) != EXPECTED_NCCL:
        raise SmokeError(f"NCCL version mismatch: {torch.cuda.nccl.version()!r}")


def _nccl_worker(
    rank: int,
    world_size: int,
    port: int,
    result_queue: "mp.Queue[dict[str, Any]]",
) -> None:
    import torch  # noqa: E402
    import torch.distributed as dist  # noqa: E402

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)
        properties = torch.cuda.get_device_properties(rank)
        uuid = canonical_uuid(getattr(properties, "uuid", ""))
        if uuid != EXPECTED_UUIDS[rank]:
            raise SmokeError(f"rank {rank}: UUID mismatch {uuid!r}")
        started = time.perf_counter()
        tensor = torch.full(
            (MESSAGE_ELEMENTS,),
            float(rank + 1),
            device=f"cuda:{rank}",
            dtype=torch.float32,
        )
        checks: list[dict[str, float]] = []
        for iteration in range(COLLECTIVES):
            tensor.fill_(float(rank + 1))
            dist.all_reduce(tensor)
            minimum = float(tensor.min().item())
            maximum = float(tensor.max().item())
            if minimum != EXPECTED_ALLREDUCE_SUM or maximum != EXPECTED_ALLREDUCE_SUM:
                raise SmokeError(
                    f"rank {rank} iteration {iteration}: range {minimum}/{maximum}"
                )
            checks.append({"iteration": iteration, "min": minimum, "max": maximum})
        torch.cuda.synchronize(rank)
        elapsed = time.perf_counter() - started
        result_queue.put(
            {
                "rank": rank,
                "device": rank,
                "uuid": short_uuid(uuid),
                "elapsed_seconds": elapsed,
                "checks": checks,
            }
        )
    except Exception as error:  # noqa: BLE001 - propagate to the parent process
        result_queue.put({"rank": rank, "error": str(error)})
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def nccl_smoke() -> dict[str, Any]:
    world_size = 4
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = int(probe.getsockname()[1])
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_nccl_worker,
            args=(rank, world_size, port, result_queue),
        )
        for rank in range(world_size)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=600)
    if any(process.exitcode != 0 for process in processes):
        raise SmokeError(
            f"NCCL process failed: {[p.exitcode for p in processes]}"
        )
    results: list[dict[str, Any]] = []
    for _ in processes:
        results.append(result_queue.get(timeout=30))
    results.sort(key=lambda item: item["rank"])
    for item in results:
        if "error" in item:
            raise SmokeError(f"rank {item['rank']} reported: {item['error']}")
    return {
        "backend": "nccl",
        "collectives": COLLECTIVES,
        "expected_sum": EXPECTED_ALLREDUCE_SUM,
        "message_elements": MESSAGE_ELEMENTS,
        "ranks": results,
        "status": "PASS",
        "world_size": world_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    arguments = parser.parse_args(argv)

    data_root = arguments.data_root.resolve(strict=True)
    if os.uname().nodename != EXPECTED_HOST:
        raise SmokeError(f"expected host {EXPECTED_HOST}, got {os.uname().nodename}")
    boot_id = parse_boot_id()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = arguments.output_dir
    if output_root is None:
        output_root = (
            data_root
            / "operations/stage0/g0-g-uuid-exclusion"
            / f"cuda-nccl-smoke-{run_id}"
        )
    output_root = output_root.resolve()
    if not str(output_root).startswith(str(data_root)):
        raise SmokeError("output directory escapes the approved data root")
    if output_root.exists():
        raise SmokeError(f"output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)

    assert_no_compute_apps("before the smoke")
    nvml_before = nvml_csv()
    validate_nvml_rows(nvml_before, "nvml-before")
    row_before = row_remapper()
    cursor = journal_cursor()
    (output_root / "journal-cursor-start.txt").write_text(
        cursor + "\n", encoding="utf-8"
    )
    (output_root / "nvml-before.csv").write_text(nvml_before, encoding="utf-8")
    (output_root / "row-remapper-before.txt").write_text(row_before, encoding="utf-8")
    (output_root / "compute-apps-before.csv").write_text("", encoding="utf-8")

    # Importing a CUDA-enabled torch can create a tiny driver context on this
    # host.  The before-snapshot therefore must be captured by the same process
    # before any torch import, so the evidence reflects the idle baseline.
    import torch  # noqa: E402
    import torch.distributed as dist  # noqa: E402
    import torch.multiprocessing as mp  # noqa: E402

    globals()["torch"] = torch
    globals()["dist"] = dist
    globals()["mp"] = mp

    _validate_expected_runtime()

    per_gpu = per_gpu_smoke()
    nccl = nccl_smoke()

    assert_no_compute_apps("after the smoke")
    nvml_after = nvml_csv()
    validate_nvml_rows(nvml_after, "nvml-after")
    row_after = row_remapper()
    kernel_delta = run_command(
        ["journalctl", "-k", f"--after-cursor={cursor}", "--no-pager"],
        timeout=120,
    )
    critical_lines = [
        line for line in kernel_delta.splitlines() if CRITICAL_PATTERN.search(line)
    ]
    (output_root / "nvml-after.csv").write_text(nvml_after, encoding="utf-8")
    (output_root / "row-remapper-after.txt").write_text(row_after, encoding="utf-8")
    (output_root / "compute-apps-after.csv").write_text("", encoding="utf-8")
    (output_root / "kernel-delta.txt").write_text(kernel_delta, encoding="utf-8")
    (output_root / "kernel-critical.txt").write_text(
        "\n".join(critical_lines) + ("\n" if critical_lines else ""),
        encoding="utf-8",
    )
    if critical_lines:
        raise SmokeError(f"critical kernel events during smoke:\n{critical_lines}")

    write_json(output_root / "per-gpu-tensor.json", per_gpu)
    write_json(output_root / "nccl-allreduce.json", nccl)
    (output_root / "SUCCESS").write_text(
        "\n".join(
            [
                "status=PASS",
                f"run_id={run_id}",
                f"boot_id={boot_id}",
                f"evidence={output_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_names = [
        "per-gpu-tensor.json",
        "nccl-allreduce.json",
        "nvml-before.csv",
        "nvml-after.csv",
        "row-remapper-before.txt",
        "row-remapper-after.txt",
        "compute-apps-before.csv",
        "compute-apps-after.csv",
        "kernel-delta.txt",
        "kernel-critical.txt",
        "journal-cursor-start.txt",
        "SUCCESS",
    ]
    manifest_lines = []
    for name in manifest_names:
        path = output_root / name
        manifest_lines.append(f"{sha256_file(path)}  {name}")
    (output_root / "SHA256SUMS").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8"
    )

    for path in output_root.iterdir():
        path.chmod(0o444)

    print(
        json.dumps(
            {
                "status": "PASS",
                "run_id": run_id,
                "boot_id": boot_id,
                "evidence": str(output_root),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "nccl": list(torch.cuda.nccl.version()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
