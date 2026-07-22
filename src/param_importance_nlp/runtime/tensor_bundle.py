"""安全、可校验的张量状态包。

该模块故意不使用 pickle。状态树只能包含有限标量、字符串、列表、元组、
字典、NumPy 数组和 Torch 张量；任何未知对象都会在写入前失败。张量以原始
连续字节保存，manifest 明确记录 dtype、shape、字节数和 SHA-256。读取时先
完成文件集合与哈希验证，再构造任何活动状态，从而避免把损坏或替换后的文件
交给训练代码。

本实现是本机可验证的 ``raw-tensor-bundle.v1`` codec。未来服务器侧可在相同
接口下增加 Safetensors codec，但不能静默改变已有 bundle 的解释方式。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

import numpy as np

from ..atomic import sha256_file, stable_json_bytes, stable_json_hash
from ..contracts import DependencyUnavailable
from ._jsonio import load_canonical_json


SCHEMA_VERSION = "runtime.tensor-bundle.v1"


def _torch() -> Any:
    """延迟导入 Torch，使不需要训练核心的 Stage 0 工具仍可独立导入。"""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - 当前本机环境已安装 Torch
        raise DependencyUnavailable(
            "torch", feature="raw_tensor_bundle", install_extra="local-core"
        ) from exc
    return torch


def _write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True, slots=True)
class TensorBundle:
    """已发布张量包的只读身份。

    ``manifest_sha256`` 是规范 manifest 内容的 SHA-256；它用于 checkpoint commit
    把逻辑状态与不可变对象绑定，不能用目录时间戳替代。
    """

    path: Path
    manifest_sha256: str
    tensor_count: int


class _Encoder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.entries: list[dict[str, Any]] = []

    def encode(self, value: Any) -> Any:
        torch = _torch()
        if isinstance(value, torch.Tensor):
            return self._encode_torch(value)
        if isinstance(value, np.ndarray):
            return self._encode_numpy(value)
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("STATE_NONFINITE_FLOAT")
            return value
        if isinstance(value, list):
            return {"kind": "list", "items": [self.encode(item) for item in value]}
        if isinstance(value, tuple):
            return {"kind": "tuple", "items": [self.encode(item) for item in value]}
        if isinstance(value, dict):
            items: list[list[Any]] = []
            for key in value:
                if isinstance(key, bool) or not isinstance(key, (str, int)):
                    raise TypeError(f"STATE_UNSUPPORTED_KEY:{type(key).__name__}")
            for key in sorted(
                value,
                key=lambda candidate: (
                    0 if isinstance(candidate, str) else 1,
                    str(candidate),
                ),
            ):
                item = value[key]
                items.append([self.encode(key), self.encode(item)])
            return {"kind": "dict", "items": items}
        raise TypeError(f"STATE_UNSUPPORTED_TYPE:{type(value).__name__}")

    def _next_path(self) -> tuple[str, Path]:
        identifier = f"tensor-{len(self.entries):08d}"
        relative = f"tensors/{identifier}.bin"
        return identifier, self.root / relative

    def _encode_torch(self, tensor: Any) -> dict[str, str]:
        torch = _torch()
        if tensor.layout != torch.strided:
            raise TypeError(f"TENSOR_UNSUPPORTED_LAYOUT:{tensor.layout}")
        detached = tensor.detach().cpu().contiguous()
        payload = detached.view(torch.uint8).numpy().tobytes(order="C")
        identifier, path = self._next_path()
        _write_file(path, payload)
        relative = path.relative_to(self.root).as_posix()
        self.entries.append(
            {
                "id": identifier,
                "kind": "torch",
                "path": relative,
                "dtype": str(detached.dtype).removeprefix("torch."),
                "shape": list(detached.shape),
                "byte_order": sys.byteorder,
                "size": len(payload),
                "sha256": sha256_file(path),
            }
        )
        return {"kind": "tensor_ref", "id": identifier}

    def _encode_numpy(self, array: np.ndarray) -> dict[str, str]:
        if array.dtype.hasobject:
            raise TypeError("NUMPY_OBJECT_DTYPE_FORBIDDEN")
        contiguous = np.ascontiguousarray(array)
        payload = contiguous.tobytes(order="C")
        identifier, path = self._next_path()
        _write_file(path, payload)
        relative = path.relative_to(self.root).as_posix()
        self.entries.append(
            {
                "id": identifier,
                "kind": "numpy",
                "path": relative,
                "dtype": contiguous.dtype.str,
                "shape": list(contiguous.shape),
                "byte_order": _numpy_byte_order(contiguous.dtype),
                "size": len(payload),
                "sha256": sha256_file(path),
            }
        )
        return {"kind": "tensor_ref", "id": identifier}


def _numpy_byte_order(dtype: np.dtype[Any]) -> str:
    if dtype.itemsize == 1 or dtype.byteorder == "|":
        return "not-applicable"
    if dtype.byteorder == "<":
        return "little"
    if dtype.byteorder == ">":
        return "big"
    return sys.byteorder


def publish_tensor_bundle(path: str | Path, state: Any) -> TensorBundle:
    """以不可覆盖方式发布一个状态包。

    临时目录与目标位于同一父目录；所有张量和 manifest 写完并重新校验后才执行
    目录 rename。rename 仅表示“对象已发布”，是否可被恢复仍由上层独立的
    checkpoint commit 决定。
    """

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"BUNDLE_ALREADY_EXISTS:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        encoder = _Encoder(temporary)
        encoded_state = encoder.encode(state)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "codec": "raw-tensor-bundle.v1",
            "state": encoded_state,
            "tensors": encoder.entries,
        }
        manifest_payload = stable_json_bytes(manifest)
        _write_file(temporary / "manifest.json", manifest_payload)
        manifest_hash = stable_json_hash(manifest)
        _verify_bundle_files(temporary, manifest)
        os.replace(temporary, target)
        return TensorBundle(target, manifest_hash, len(encoder.entries))
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _strict_manifest(path: Path) -> dict[str, Any]:
    value = load_canonical_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("TENSOR_BUNDLE_SCHEMA_MISMATCH")
    if set(value) != {"schema_version", "codec", "state", "tensors"}:
        raise ValueError("TENSOR_BUNDLE_FIELDS_MISMATCH")
    if value.get("codec") != "raw-tensor-bundle.v1":
        raise ValueError("TENSOR_BUNDLE_CODEC_MISMATCH")
    if not isinstance(value.get("tensors"), list):
        raise ValueError("TENSOR_BUNDLE_TENSORS_NOT_ARRAY")
    return value


def _verify_bundle_files(root: Path, manifest: dict[str, Any]) -> None:
    expected = {"manifest.json"}
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    for entry in manifest["tensors"]:
        if not isinstance(entry, dict):
            raise ValueError("TENSOR_ENTRY_NOT_OBJECT")
        required = {
            "id",
            "kind",
            "path",
            "dtype",
            "shape",
            "byte_order",
            "size",
            "sha256",
        }
        if set(entry) != required:
            raise ValueError("TENSOR_ENTRY_FIELDS_MISMATCH")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier.startswith("tensor-"):
            raise ValueError(f"TENSOR_INVALID_ID:{identifier!r}")
        if identifier in seen_ids:
            raise ValueError(f"TENSOR_DUPLICATE_ID:{identifier}")
        seen_ids.add(identifier)
        if entry["kind"] not in {"torch", "numpy"}:
            raise ValueError(f"TENSOR_UNKNOWN_KIND:{entry['kind']}")
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"TENSOR_PATH_ESCAPE:{relative}")
        expected_relative = f"tensors/{identifier}.bin"
        if relative.as_posix() != expected_relative:
            raise ValueError(f"TENSOR_NONCANONICAL_PATH:{relative}")
        folded = relative.as_posix().casefold()
        if folded in seen_paths:
            raise ValueError(f"TENSOR_PATH_COLLISION:{relative}")
        seen_paths.add(folded)
        shape = entry["shape"]
        if not isinstance(shape, list) or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in shape
        ):
            raise ValueError(f"TENSOR_INVALID_SHAPE:{identifier}")
        size = entry["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"TENSOR_INVALID_SIZE:{identifier}")
        digest = entry["sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError(f"TENSOR_INVALID_HASH:{identifier}")
        item_count = math.prod(shape)
        if entry["kind"] == "numpy":
            try:
                dtype = np.dtype(entry["dtype"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"NUMPY_DTYPE_UNKNOWN:{entry['dtype']}") from exc
            if dtype.hasobject:
                raise ValueError("NUMPY_OBJECT_DTYPE_FORBIDDEN")
            expected_order = _numpy_byte_order(dtype)
            item_size = dtype.itemsize
        else:
            torch = _torch()
            dtype = getattr(torch, entry["dtype"], None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"TORCH_DTYPE_UNKNOWN:{entry['dtype']}")
            expected_order = sys.byteorder
            item_size = torch.empty((), dtype=dtype).element_size()
        if entry["byte_order"] != expected_order:
            raise ValueError(f"TENSOR_BYTE_ORDER_MISMATCH:{identifier}")
        if size != item_count * item_size:
            raise ValueError(f"TENSOR_DECLARED_SIZE_MISMATCH:{identifier}")
        file_path = root / relative
        expected.add(relative.as_posix())
        if file_path.is_symlink() or not file_path.is_file():
            raise ValueError(f"TENSOR_FILE_MISSING:{relative}")
        if file_path.stat().st_size != entry["size"]:
            raise ValueError(f"TENSOR_SIZE_MISMATCH:{relative}")
        if sha256_file(file_path) != entry["sha256"]:
            raise ValueError(f"TENSOR_HASH_MISMATCH:{relative}")
    actual = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    if actual != expected:
        raise ValueError(
            f"TENSOR_FILE_SET_MISMATCH:missing={sorted(expected-actual)}:extra={sorted(actual-expected)}"
        )
    # manifest 的状态树也属于不可信输入。只校验 tensor 文件而不校验引用，会让
    # 手工构造的未标记 object、未知 kind、重复字典键或孤儿 tensor 混入恢复路径。
    reference_counts = {identifier: 0 for identifier in seen_ids}
    _validate_encoded_state(manifest["state"], reference_counts=reference_counts)
    invalid_reference_counts = {
        identifier: count
        for identifier, count in sorted(reference_counts.items())
        if count != 1
    }
    if invalid_reference_counts:
        raise ValueError(
            f"TENSOR_REFERENCE_COUNT_MISMATCH:{invalid_reference_counts}"
        )


def _validate_encoded_state(
    value: Any,
    *,
    reference_counts: dict[str, int],
) -> None:
    """递归验证 encoder wire tree，并统计每个 tensor ID 的引用次数。"""

    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):  # strict JSON loader 已拒绝，保留纵深防御。
            raise ValueError("STATE_NONFINITE_FLOAT")
        return
    if not isinstance(value, dict):
        raise ValueError(f"STATE_WIRE_VALUE_UNSUPPORTED:{type(value).__name__}")
    kind = value.get("kind")
    if kind in {"list", "tuple"}:
        if set(value) != {"kind", "items"} or not isinstance(value["items"], list):
            raise ValueError(f"STATE_{str(kind).upper()}_FIELDS_MISMATCH")
        for item in value["items"]:
            _validate_encoded_state(item, reference_counts=reference_counts)
        return
    if kind == "dict":
        if set(value) != {"kind", "items"} or not isinstance(value["items"], list):
            raise ValueError("STATE_DICT_FIELDS_MISMATCH")
        seen_keys: set[tuple[type[Any], Any]] = set()
        for pair in value["items"]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("STATE_DICT_ITEM_NOT_PAIR")
            key, item = pair
            if isinstance(key, bool) or not isinstance(key, (str, int)):
                raise ValueError("STATE_DICT_KEY_UNSUPPORTED")
            identity = (type(key), key)
            if identity in seen_keys:
                raise ValueError("STATE_DICT_DUPLICATE_KEY")
            seen_keys.add(identity)
            _validate_encoded_state(item, reference_counts=reference_counts)
        return
    if kind == "tensor_ref":
        if set(value) != {"kind", "id"} or not isinstance(value["id"], str):
            raise ValueError("STATE_TENSOR_REF_FIELDS_MISMATCH")
        identifier = value["id"]
        if identifier not in reference_counts:
            raise ValueError(f"TENSOR_UNKNOWN_REF:{identifier}")
        reference_counts[identifier] += 1
        return
    raise ValueError(f"STATE_UNKNOWN_KIND:{kind}")


class _Decoder:
    def __init__(self, root: Path, entries: list[dict[str, Any]]) -> None:
        self.root = root
        self.entries = {entry["id"]: entry for entry in entries}

    def decode(self, value: Any) -> Any:
        if not isinstance(value, dict) or "kind" not in value:
            return value
        kind = value["kind"]
        if kind == "list":
            return [self.decode(item) for item in value["items"]]
        if kind == "tuple":
            return tuple(self.decode(item) for item in value["items"])
        if kind == "dict":
            return {self.decode(key): self.decode(item) for key, item in value["items"]}
        if kind == "tensor_ref":
            return self._tensor(value["id"])
        raise ValueError(f"STATE_UNKNOWN_KIND:{kind}")

    def _tensor(self, identifier: str) -> Any:
        entry = self.entries.get(identifier)
        if entry is None:
            raise ValueError(f"TENSOR_UNKNOWN_REF:{identifier}")
        payload = (self.root / entry["path"]).read_bytes()
        shape = tuple(int(item) for item in entry["shape"])
        if entry["kind"] == "numpy":
            return np.frombuffer(payload, dtype=np.dtype(entry["dtype"])).copy().reshape(shape)
        torch = _torch()
        dtype = getattr(torch, entry["dtype"], None)
        if dtype is None:
            raise ValueError(f"TORCH_DTYPE_UNKNOWN:{entry['dtype']}")
        if not payload:
            return torch.empty(shape, dtype=dtype)
        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
        return raw.view(dtype).reshape(shape)


def load_tensor_bundle(path: str | Path) -> tuple[Any, TensorBundle]:
    """严格验证并读取张量包，返回状态树与不可变身份。"""

    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"BUNDLE_INVALID_ROOT:{root}")
    manifest = _strict_manifest(root / "manifest.json")
    _verify_bundle_files(root, manifest)
    decoder = _Decoder(root, manifest["tensors"])
    state = decoder.decode(manifest["state"])
    return state, TensorBundle(root, stable_json_hash(manifest), len(manifest["tensors"]))
