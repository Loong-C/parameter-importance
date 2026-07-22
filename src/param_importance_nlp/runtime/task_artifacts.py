"""统一任务产物的不可变对象与权威 commit 存储。

Stage 0--9 的 runner 会产生许多体积很小、语义不同的 JSON 结果。它们仍共享同一
发布原语：先按内容 hash 发布不可变对象，重新读取并核验后，再发布一个稳定逻辑
路径的 commit。消费者只允许读取 commit；仅存在对象文件、临时文件或目录 rename
都不表示任务已经完成。

本存储只处理 JSON 控制面。模型、optimizer 和逐坐标张量继续使用
``CheckpointStore``/``TensorBundle``，禁止为了方便把它们塞进 JSON 或 pickle。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Mapping

from ..contracts.jsonio import (
    JSONValue,
    canonical_json_hash,
    load_canonical_json,
    write_canonical_json,
)


_KIND_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_PAYLOAD_SCHEMA_VERSION_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$"
)


def _logical_path(value: str, *, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"TASK_ARTIFACT_LOGICAL_PATH_INVALID:{field}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"TASK_ARTIFACT_PATH_ESCAPE:{field}")
    return path


def _safe_component(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}", value
    ):
        raise ValueError(f"TASK_ARTIFACT_COMPONENT_INVALID:{field}")
    return value


def _validated_payload_object(payload: object) -> dict[str, JSONValue]:
    """收窄业务 payload 为可 canonical 编码、带机器版本的 JSON object。"""

    if not isinstance(payload, Mapping):
        raise TypeError("TASK_ARTIFACT_PAYLOAD_NOT_OBJECT")
    normalized = dict(payload)
    schema_version = normalized.get("schema_version")
    if (
        not isinstance(schema_version, str)
        or _PAYLOAD_SCHEMA_VERSION_RE.fullmatch(schema_version) is None
    ):
        raise ValueError("TASK_ARTIFACT_PAYLOAD_SCHEMA_VERSION_INVALID")
    try:
        canonical_json_hash(normalized)
    except (TypeError, ValueError) as error:
        raise ValueError("TASK_ARTIFACT_PAYLOAD_NOT_CANONICAL_JSON") from error
    return normalized


@dataclass(frozen=True, slots=True)
class PublishedTaskArtifact:
    """一个已经通过 commit 发现的任务产物引用。"""

    task_id: str
    artifact_kind: str
    artifact_hash: str
    config_hash: str
    object_ref: str
    commit_ref: str
    formal_eligible: bool

    def __post_init__(self) -> None:
        _safe_component(self.task_id, field="task_id")
        if _KIND_RE.fullmatch(self.artifact_kind) is None:
            raise ValueError("TASK_ARTIFACT_KIND_INVALID")
        if _HASH_RE.fullmatch(self.artifact_hash) is None:
            raise ValueError("TASK_ARTIFACT_HASH_INVALID")
        if _HASH_RE.fullmatch(self.config_hash) is None:
            raise ValueError("TASK_ARTIFACT_CONFIG_HASH_INVALID")
        _logical_path(self.object_ref, field="object_ref")
        _logical_path(self.commit_ref, field="commit_ref")
        if type(self.formal_eligible) is not bool:
            raise TypeError("TASK_ARTIFACT_FORMAL_ELIGIBLE_NOT_BOOL")


@dataclass(frozen=True, slots=True)
class LoadedTaskArtifact:
    """从权威 commit 完整核验后得到的只读 envelope 与语义 payload。

    ``PublishedTaskArtifact`` 只描述逻辑身份，适合发布与发现接口；formal preflight
    还必须检查 envelope 的 ``run_intent``、``source_refs`` 与 payload，因此使用本
    类型显式携带这些字段。返回值中的 mapping 都是重新构造的普通 JSON object，
    调用方修改它们不会改动磁盘对象。
    """

    identity: PublishedTaskArtifact
    run_intent: str
    source_refs: tuple[str, ...]
    payload: Mapping[str, JSONValue]


def load_committed_task_artifact(
    workspace_root: str | Path,
    commit_ref: str,
    *,
    require_formal: bool = False,
) -> LoadedTaskArtifact:
    """只读加载 task commit，并验证 commit/envelope/hash/身份四重绑定。

    该函数不会创建目录，也不会把直接 object 路径当作 commit。正式 preflight
    使用 ``require_formal=True`` 时还要求 envelope 同时满足
    ``run_intent=formal`` 与 ``formal_eligible=true``；因此本机 fixture 即使 payload
    内容看起来像正式 Gate，也不能解锁 formal runner。
    """

    root = Path(workspace_root).resolve()
    logical = _logical_path(commit_ref, field="commit_ref")
    commit_path = root.joinpath(*logical.parts).resolve()
    try:
        commit_path.relative_to(root)
    except ValueError as error:
        raise ValueError("TASK_ARTIFACT_COMMIT_ESCAPE") from error
    commit = load_canonical_json(commit_path)
    commit_fields = {
        "schema_version",
        "task_id",
        "artifact_kind",
        "config_hash",
        "artifact_hash",
        "object_ref",
        "formal_eligible",
    }
    if not isinstance(commit, dict) or set(commit) != commit_fields:
        raise ValueError("TASK_ARTIFACT_COMMIT_FIELDS_INVALID")
    if commit["schema_version"] != "task-output-commit-v1":
        raise ValueError("TASK_ARTIFACT_COMMIT_VERSION_INVALID")
    if type(commit["formal_eligible"]) is not bool:
        raise TypeError("TASK_ARTIFACT_COMMIT_FORMAL_ELIGIBLE_NOT_BOOL")
    for field_name in ("config_hash", "artifact_hash"):
        value = commit[field_name]
        if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
            raise ValueError(f"TASK_ARTIFACT_COMMIT_HASH_INVALID:{field_name}")
    task_id = commit["task_id"]
    artifact_kind = commit["artifact_kind"]
    if not isinstance(task_id, str):
        raise TypeError("TASK_ARTIFACT_COMMIT_TASK_ID_NOT_STRING")
    if not isinstance(artifact_kind, str):
        raise TypeError("TASK_ARTIFACT_COMMIT_KIND_NOT_STRING")
    _safe_component(task_id, field="task_id")
    if _KIND_RE.fullmatch(artifact_kind) is None:
        raise ValueError("TASK_ARTIFACT_KIND_INVALID")

    object_ref = commit["object_ref"]
    if not isinstance(object_ref, str):
        raise TypeError("TASK_ARTIFACT_COMMIT_OBJECT_REF_NOT_STRING")
    object_logical = _logical_path(object_ref, field="object_ref")
    object_path = root.joinpath(*object_logical.parts).resolve()
    try:
        object_path.relative_to(root)
    except ValueError as error:
        raise ValueError("TASK_ARTIFACT_OBJECT_ESCAPE") from error
    # 内容寻址对象必须保留 TaskArtifactStore 冻结的目录布局；任意 JSON 文件即使
    # 恰好含有相同 payload，也不能冒充不可变对象发布。
    if (
        object_path.name != f"{commit['artifact_hash']}.json"
        or object_path.parent.name != artifact_kind
        or object_path.parent.parent.name != "objects"
    ):
        raise ValueError("TASK_ARTIFACT_OBJECT_LAYOUT_INVALID")

    envelope = load_canonical_json(object_path)
    envelope_fields = {
        "schema_version",
        "task_id",
        "artifact_kind",
        "config_hash",
        "run_intent",
        "formal_eligible",
        "source_refs",
        "payload",
        "artifact_hash",
    }
    if not isinstance(envelope, dict) or set(envelope) != envelope_fields:
        raise ValueError("TASK_ARTIFACT_OBJECT_FIELDS_INVALID")
    if envelope["schema_version"] != "task-output-artifact-v1":
        raise ValueError("TASK_ARTIFACT_OBJECT_VERSION_INVALID")
    payload_without_hash = {
        key: item for key, item in envelope.items() if key != "artifact_hash"
    }
    declared_hash = envelope["artifact_hash"]
    if not isinstance(declared_hash, str) or _HASH_RE.fullmatch(declared_hash) is None:
        raise ValueError("TASK_ARTIFACT_OBJECT_HASH_INVALID")
    if declared_hash != canonical_json_hash(payload_without_hash):
        raise ValueError("TASK_ARTIFACT_OBJECT_HASH_MISMATCH")
    for field_name in (
        "task_id",
        "artifact_kind",
        "config_hash",
        "formal_eligible",
        "artifact_hash",
    ):
        if commit[field_name] != envelope[field_name]:
            raise ValueError(f"TASK_ARTIFACT_COMMIT_OBJECT_MISMATCH:{field_name}")
    run_intent = envelope["run_intent"]
    eligible = envelope["formal_eligible"]
    if run_intent not in {"local_fixture", "formal"} or type(eligible) is not bool:
        raise ValueError("TASK_ARTIFACT_OBJECT_SCOPE_INVALID")
    if eligible != (run_intent == "formal"):
        raise ValueError("TASK_ARTIFACT_OBJECT_ELIGIBILITY_MISMATCH")
    if require_formal and (run_intent != "formal" or eligible is not True):
        raise ValueError("TASK_ARTIFACT_FORMAL_ENVELOPE_REQUIRED")
    source_refs = envelope["source_refs"]
    if not isinstance(source_refs, list):
        raise TypeError("TASK_ARTIFACT_SOURCE_REFS_NOT_ARRAY")
    normalized_sources = tuple(
        _logical_path(value, field=f"source_refs[{index}]").as_posix()
        for index, value in enumerate(source_refs)
    )
    if len(normalized_sources) != len(set(normalized_sources)):
        raise ValueError("TASK_ARTIFACT_SOURCE_REFS_DUPLICATE")
    payload = _validated_payload_object(envelope["payload"])
    identity = PublishedTaskArtifact(
        task_id=task_id,
        artifact_kind=artifact_kind,
        artifact_hash=declared_hash,
        config_hash=str(commit["config_hash"]),
        object_ref=object_logical.as_posix(),
        commit_ref=logical.as_posix(),
        formal_eligible=eligible,
    )
    return LoadedTaskArtifact(
        identity=identity,
        run_intent=str(run_intent),
        source_refs=normalized_sources,
        payload=payload,
    )


class TaskArtifactStore:
    """在 workspace 内发布、发现并核对 task artifact。

    ``output_dir`` 必须是配置中已经通过校验的 POSIX 相对逻辑路径。构造函数仍会
    再次执行路径逃逸检查，形成运行时的第二道边界。
    """

    def __init__(self, workspace_root: str | Path, output_dir: str) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        logical = _logical_path(output_dir, field="output_dir")
        self.output_dir = logical.as_posix()
        candidate = self.workspace_root.joinpath(*logical.parts).resolve()
        try:
            candidate.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("TASK_ARTIFACT_OUTPUT_ESCAPE") from error
        self.root = candidate
        self.objects = self.root / "objects"
        self.commits = self.root / "commits"
        self.objects.mkdir(parents=True, exist_ok=True)
        self.commits.mkdir(parents=True, exist_ok=True)

    def _logical_ref(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace_root).as_posix()

    def publish(
        self,
        *,
        task_id: str,
        artifact_kind: str,
        config_hash: str,
        run_intent: str,
        payload: Mapping[str, JSONValue],
        formal_eligible: bool,
        source_refs: tuple[str, ...] = (),
    ) -> PublishedTaskArtifact:
        """幂等发布一个小型 JSON 产物并返回权威 commit 引用。"""

        _safe_component(task_id, field="task_id")
        if _KIND_RE.fullmatch(artifact_kind) is None:
            raise ValueError("TASK_ARTIFACT_KIND_INVALID")
        if _HASH_RE.fullmatch(config_hash) is None:
            raise ValueError("TASK_ARTIFACT_CONFIG_HASH_INVALID")
        if run_intent not in {"local_fixture", "formal"}:
            raise ValueError("TASK_ARTIFACT_RUN_INTENT_INVALID")
        if type(formal_eligible) is not bool or formal_eligible != (run_intent == "formal"):
            raise ValueError("TASK_ARTIFACT_FORMAL_ELIGIBILITY_MISMATCH")
        normalized_payload = _validated_payload_object(payload)
        refs = tuple(_logical_path(ref, field="source_refs").as_posix() for ref in source_refs)
        if len(refs) != len(set(refs)):
            raise ValueError("TASK_ARTIFACT_SOURCE_REFS_DUPLICATE")

        body: dict[str, JSONValue] = {
            "schema_version": "task-output-artifact-v1",
            "task_id": task_id,
            "artifact_kind": artifact_kind,
            "config_hash": config_hash,
            "run_intent": run_intent,
            "formal_eligible": formal_eligible,
            "source_refs": list(refs),
            "payload": normalized_payload,
        }
        artifact_hash = canonical_json_hash(body)
        body["artifact_hash"] = artifact_hash
        object_dir = self.objects / artifact_kind
        object_path = object_dir / f"{artifact_hash}.json"
        commit_path = self.commits / f"{artifact_kind}.json"

        if object_path.exists():
            existing = load_canonical_json(object_path)
            if existing != body:
                raise RuntimeError("TASK_ARTIFACT_HASH_COLLISION_OR_OBJECT_DRIFT")
        else:
            write_canonical_json(object_path, body)
        verified = load_canonical_json(object_path)
        if not isinstance(verified, dict) or verified.get("artifact_hash") != artifact_hash:
            raise RuntimeError("TASK_ARTIFACT_POST_PUBLISH_VALIDATION_FAILED")

        commit: dict[str, JSONValue] = {
            "schema_version": "task-output-commit-v1",
            "task_id": task_id,
            "artifact_kind": artifact_kind,
            "config_hash": config_hash,
            "artifact_hash": artifact_hash,
            "object_ref": self._logical_ref(object_path),
            "formal_eligible": formal_eligible,
        }
        if commit_path.exists():
            existing_commit = load_canonical_json(commit_path)
            if existing_commit != commit:
                raise FileExistsError(
                    f"TASK_ARTIFACT_COMMIT_CONFLICT:{self._logical_ref(commit_path)}"
                )
        else:
            write_canonical_json(commit_path, commit)
        # 从 commit 再走一次完整发现路径，避免把刚写入但不可消费的引用返回给 runner。
        return self.load_commit(self._logical_ref(commit_path))

    def load_commit(self, commit_ref: str) -> PublishedTaskArtifact:
        """读取权威 commit，并验证对象存在、hash 与身份全部一致。"""

        logical = _logical_path(commit_ref, field="commit_ref")
        commit_path = self.workspace_root.joinpath(*logical.parts).resolve()
        try:
            commit_path.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("TASK_ARTIFACT_COMMIT_ESCAPE") from error
        commit = load_canonical_json(commit_path)
        if not isinstance(commit, dict) or set(commit) != {
            "schema_version",
            "task_id",
            "artifact_kind",
            "config_hash",
            "artifact_hash",
            "object_ref",
            "formal_eligible",
        }:
            raise ValueError("TASK_ARTIFACT_COMMIT_FIELDS_INVALID")
        if commit["schema_version"] != "task-output-commit-v1":
            raise ValueError("TASK_ARTIFACT_COMMIT_VERSION_INVALID")
        object_logical = _logical_path(str(commit["object_ref"]), field="object_ref")
        object_path = self.workspace_root.joinpath(*object_logical.parts).resolve()
        try:
            object_path.relative_to(self.workspace_root)
        except ValueError as error:
            raise ValueError("TASK_ARTIFACT_OBJECT_ESCAPE") from error
        value = load_canonical_json(object_path)
        if not isinstance(value, dict):
            raise ValueError("TASK_ARTIFACT_OBJECT_ROOT_INVALID")
        declared_hash = value.get("artifact_hash")
        payload_without_hash = {key: item for key, item in value.items() if key != "artifact_hash"}
        if declared_hash != canonical_json_hash(payload_without_hash):
            raise ValueError("TASK_ARTIFACT_OBJECT_HASH_MISMATCH")
        _validated_payload_object(value.get("payload"))
        for field in ("task_id", "artifact_kind", "config_hash", "formal_eligible"):
            if commit[field] != value.get(field):
                raise ValueError(f"TASK_ARTIFACT_COMMIT_OBJECT_MISMATCH:{field}")
        if commit["artifact_hash"] != declared_hash:
            raise ValueError("TASK_ARTIFACT_COMMIT_HASH_MISMATCH")
        return PublishedTaskArtifact(
            task_id=str(commit["task_id"]),
            artifact_kind=str(commit["artifact_kind"]),
            artifact_hash=str(commit["artifact_hash"]),
            config_hash=str(commit["config_hash"]),
            object_ref=object_logical.as_posix(),
            commit_ref=logical.as_posix(),
            formal_eligible=bool(commit["formal_eligible"]),
        )

    def discover_complete(
        self,
        *,
        task_id: str,
        config_hash: str,
        artifact_kinds: tuple[str, ...],
        formal_eligible: bool,
    ) -> Mapping[str, str] | None:
        """若全部期望 commit 已完整发布则返回有序引用，否则返回 ``None``。

        只要任一已经存在的 commit 身份漂移，就立即失败，而不是把它当成“尚未
        完成”覆盖掉。这样同一个输出目录不能被另一份配置静默复用。
        """

        refs: dict[str, str] = {}
        for kind in artifact_kinds:
            commit_path = self.commits / f"{kind}.json"
            if not commit_path.exists():
                return None
            published = self.load_commit(self._logical_ref(commit_path))
            if (
                published.task_id != task_id
                or published.artifact_kind != kind
                or published.config_hash != config_hash
                or published.formal_eligible != formal_eligible
            ):
                raise ValueError(f"TASK_ARTIFACT_EXISTING_COMMIT_IDENTITY_DRIFT:{kind}")
            refs[kind] = published.commit_ref
        return refs


__all__ = [
    "LoadedTaskArtifact",
    "PublishedTaskArtifact",
    "TaskArtifactStore",
    "load_committed_task_artifact",
]
