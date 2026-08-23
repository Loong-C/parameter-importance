#!/usr/bin/env python3
"""Run the bounded current Stage-0 GPU smoke with an explicit GPU exclusion.

The historical Stage-0 orchestrator expects a four-device NVML view.  This
wrapper keeps its CUDA/NCCL worker and semantic checks, but performs the
hardware admission check before launching it and passes UUIDs explicitly so
the faulty PCI device can never be scheduled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_HOST = "sophgo13"
EXCLUDED_PCI = "0000:50:00.0"
EXCLUDED_UUID = "GPU-dc6cfc60-41dd-7bcf-ed09-b7deb5be342c"
ALLOWED = (
    ("0000:53:00.0", "GPU-180ff767-885a-7dc9-c8a9-921d65a01bbd"),
    ("0000:9C:00.0", "GPU-5c672d04-4f83-3cc0-80d0-0108b1b63267"),
    ("0000:9D:00.0", "GPU-e78c55cd-db97-b761-f559-dc6eae3be81d"),
    ("0000:A0:00.0", "GPU-9b2b2a3b-3547-187f-ca29-2c02624e2e4f"),
)
EXPECTED_TORCH = "2.12.1+cu126"
EXPECTED_CUDA = "12.6"
EXPECTED_CUDNN = 91002
EXPECTED_NCCL = (2, 29, 3)


class SmokeError(RuntimeError):
    pass


def canonical_pci(value: str) -> str:
    """Normalize nvidia-smi's eight-zero domain prefix to the contract form."""
    value = value.strip().upper()
    match = re.fullmatch(r"(?:[0-9A-F]{4}|[0-9A-F]{8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-9])", value)
    if not match:
        raise SmokeError(f"invalid PCI bus id: {value!r}")
    return f"0000:{match.group(1)}:{match.group(2)}.{match.group(3)}"


def command(argv: list[str], timeout: int = 60) -> str:
    completed = subprocess.run(
        argv, capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode:
        raise SmokeError(
            f"command exited {completed.returncode}: {argv!r}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    return completed.stdout


def gpu_rows() -> list[dict[str, str]]:
    text = command(
        [
            "nvidia-smi",
            "--query-gpu=index,pci.bus_id,uuid,memory.used,utilization.gpu,"
            "temperature.gpu,ecc.errors.uncorrected.volatile.total,"
            "ecc.errors.uncorrected.aggregate.total",
            "--format=csv,noheader,nounits",
        ],
        timeout=30,
    )
    rows = list(csv.reader(line for line in text.splitlines() if line.strip()))
    if len(rows) != 8 or any(len(row) != 8 for row in rows):
        raise SmokeError(f"expected the complete 8-device inventory, got {rows!r}")
    fields = (
        "index",
        "pci_bus_id",
        "uuid",
        "memory_used_mib",
        "utilization_percent",
        "temperature_c",
        "ecc_volatile_uncorrected",
        "ecc_aggregate_uncorrected",
    )
    result = [dict(zip(fields, (item.strip() for item in row))) for row in rows]
    for row in result:
        row["pci_bus_id"] = canonical_pci(row["pci_bus_id"])
    by_pci = {row["pci_bus_id"].upper(): row for row in result}
    if EXCLUDED_PCI.upper() not in by_pci:
        raise SmokeError(f"excluded PCI device missing from inventory: {by_pci!r}")
    for pci, uuid in ALLOWED:
        row = by_pci.get(pci.upper())
        if row is None or row["uuid"].lower() != uuid.lower():
            raise SmokeError(f"allowed identity mismatch for {pci}: {row!r}")
        if int(row["ecc_volatile_uncorrected"]) or int(row["ecc_aggregate_uncorrected"]):
            raise SmokeError(f"allowed GPU has uncorrected ECC: {row!r}")
    return result


def idle_inventory() -> list[dict[str, str]]:
    """Require two consecutive idle samples before admitting CUDA work."""
    latest: list[dict[str, str]] = []
    for _ in range(12):
        latest = gpu_rows()
        by_pci = {row["pci_bus_id"]: row for row in latest}
        if all(
            by_pci[pci]["memory_used_mib"] == "0"
            and by_pci[pci]["utilization_percent"] == "0"
            for pci, _ in ALLOWED
        ):
            time.sleep(2)
            confirm = gpu_rows()
            confirm_by_pci = {row["pci_bus_id"]: row for row in confirm}
            if all(
                confirm_by_pci[pci]["memory_used_mib"] == "0"
                and confirm_by_pci[pci]["utilization_percent"] == "0"
                for pci, _ in ALLOWED
            ):
                return confirm
        time.sleep(2)
    raise SmokeError(f"allowed GPUs did not become idle after 12 samples: {latest!r}")


def no_compute_apps() -> None:
    apps = command(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        timeout=30,
    ).strip()
    if apps:
        raise SmokeError(f"compute application detected before/after smoke:\n{apps}")


def canonical_uuid(value: Any) -> str:
    value = str(value or "")
    if value.startswith("GPU-"):
        value = value[4:]
    if not re.fullmatch(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        raise SmokeError(f"invalid CUDA UUID: {value!r}")
    return "GPU-" + value.lower()


SINGLE_CODE = r'''
import json, torch
if torch.__version__ != "2.12.1+cu126" or torch.version.cuda != "12.6":
    raise RuntimeError(f"runtime mismatch: {torch.__version__!r}/{torch.version.cuda!r}")
if torch.backends.cudnn.version() != 91002 or tuple(torch.cuda.nccl.version()) != (2, 29, 3):
    raise RuntimeError("cuDNN/NCCL runtime mismatch")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one visible CUDA device, got {torch.cuda.device_count()}")
p = torch.cuda.get_device_properties(0)
uuid = str(getattr(p, "uuid", ""))
if not uuid.startswith("GPU-"):
    uuid = "GPU-" + uuid
v = torch.full((1048576,), 16.0, device="cuda:0", dtype=torch.float32)
left = torch.ones((256, 256), device="cuda:0", dtype=torch.float32)
right = torch.ones((256, 256), device="cuda:0", dtype=torch.float32)
total = float(v.sum().item())
product = left @ right
torch.cuda.synchronize()
if total != 16777216.0 or float(product.min().item()) != 256.0 or float(product.max().item()) != 256.0:
    raise RuntimeError(f"numeric mismatch: {total}, {product.min().item()}, {product.max().item()}")
print(json.dumps({"status":"PASS", "device_count":1, "uuid":uuid.lower(), "vector_sum":total, "matmul":256.0}, sort_keys=True))
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve(strict=True)
    output = args.output_dir.resolve()
    if os.uname().nodename != EXPECTED_HOST:
        raise SmokeError(f"expected host {EXPECTED_HOST}, got {os.uname().nodename}")
    if output.exists() or data_root not in output.parents:
        raise SmokeError(f"output must be a new child of DATA_ROOT: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tmp = output.parent / f".{output.name}.attempt-{run_id}-{os.getpid()}"
    if tmp.exists():
        raise SmokeError(f"temporary output already exists: {tmp}")
    tmp.mkdir(parents=True)
    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    rows_before = idle_inventory()
    no_compute_apps()
    excluded = next(row for row in rows_before if row["pci_bus_id"].upper() == EXCLUDED_PCI.upper())
    if excluded["uuid"].lower() != EXCLUDED_UUID.lower():
        raise SmokeError(f"excluded identity mismatch: {excluded!r}")
    (tmp / "gpu-inventory-before.csv").write_text(
        json.dumps(rows_before, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (tmp / "excluded-gpu-health.txt").write_text(
        command(["nvidia-smi", "-q", "-i", excluded["index"], "-d", "ECC,ROW_REMAPPER"], timeout=30),
        encoding="utf-8",
    )
    single_env = env | {"CUDA_VISIBLE_DEVICES": ALLOWED[0][1]}
    single = subprocess.run(
        [sys.executable, "-c", SINGLE_CODE], env=single_env, capture_output=True, text=True, timeout=180, check=False
    )
    if single.returncode:
        raise SmokeError(f"single-GPU smoke failed:\nstdout={single.stdout}\nstderr={single.stderr}")
    (tmp / "single-gpu.json").write_text(single.stdout, encoding="utf-8")
    four_dir = tmp / "four-gpu-runner"
    four_dir.mkdir()
    runner = Path(__file__).resolve().parents[1] / "stage0" / "run_cuda_nccl_smoke.py"
    four_env = env | {"CUDA_VISIBLE_DEVICES": ",".join(uuid for _, uuid in ALLOWED)}
    four = subprocess.run(
        [sys.executable, str(runner), "--smoke-worker", "--data-root", str(data_root), "--output-dir", str(four_dir)],
        env=four_env, capture_output=True, text=True, timeout=600, check=False,
    )
    (tmp / "four-gpu-runner.stdout.log").write_text(four.stdout, encoding="utf-8")
    (tmp / "four-gpu-runner.stderr.log").write_text(four.stderr, encoding="utf-8")
    if four.returncode:
        raise SmokeError(f"four-GPU smoke failed (rc={four.returncode}); see {tmp}")
    rows_after = idle_inventory()
    no_compute_apps()
    write_json(tmp / "gpu-inventory-after.json", {"rows": rows_after})
    report = {
        "schema_version": "stage2-s202-current-gpu-smoke-v1",
        "status": "PASS",
        "run_id": run_id,
        "host": EXPECTED_HOST,
        "scope": ["single_gpu", "healthy_four_gpu", "nccl_allreduce"],
        "excluded_pci_bus_ids": [EXCLUDED_PCI],
        "excluded_device": {"index": excluded["index"], "pci_bus_id": EXCLUDED_PCI, "uuid": EXCLUDED_UUID, "scheduled": False},
        "allowed_devices": [{"pci_bus_id": pci, "uuid": uuid} for pci, uuid in ALLOWED],
        "single_gpu": {"status": "PASS", "visible_uuid": ALLOWED[0][1]},
        "healthy_four_gpu": {"status": "PASS", "runner": "ops/stage0/run_cuda_nccl_smoke.py", "semantic_checks": ["per_gpu_tensor", "nccl_allreduce"]},
        "atomic_publication": True,
    }
    write_json(tmp / "report.json", report)
    files = [path for path in tmp.rglob("*") if path.is_file() and path.name != "SHA256SUMS"]
    (tmp / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.relative_to(tmp).as_posix()}\n" for path in sorted(files)), encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps({"status": "PASS", "evidence": str(output), "report_sha256": sha256(output / "report.json"), "excluded_pci": EXCLUDED_PCI}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SmokeError as error:
        print(f"SmokeError: {error}", file=sys.stderr)
        raise SystemExit(1)
