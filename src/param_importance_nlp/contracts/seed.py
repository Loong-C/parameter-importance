"""独立随机域的确定性 seed 派生合同。

一个 ``master_seed`` 只作为命名空间根，不直接喂给任何随机数生成器。模型初始化、
训练 sampler、每 rank 随机算子、数据增强、重要性采样、剪枝和 bootstrap 都通过
SHA-256 域分离得到不同 seed。Stage 2 的 ``reference_sizing``、``reference_A``、
``reference_B``、``pilot``、``confirmatory`` 五条抽样流同样是独立一级域，避免
因为“同一个 seed 加不同 offset”而在重构时发生复用。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Final

from .errors import SeedContractError
from .jsonio import JSONValue, canonical_json_bytes, canonical_json_hash


SEED_ALGORITHM_VERSION: Final = "sha256-domain-v1"
SEED_UPPER_BOUND: Final = 2**63
DOMAIN_PATTERN: Final = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][A-Za-z0-9]+)*$"
)
DEFAULT_SEED_DOMAINS: Final = (
    "model_init",
    "sampler",
    "python_runtime",
    "numpy_runtime",
    "torch_cpu_runtime",
    "data_augmentation",
    "importance_sampling",
    "pruning_mask",
    "bootstrap",
    "reference_sizing",
    "reference_A",
    "reference_B",
    "pilot",
    "confirmatory",
)


def _validate_master_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < SEED_UPPER_BOUND:
        raise SeedContractError("master_seed 必须是 [0, 2**63) 内的整数")
    return value


def _validate_domain(value: str) -> str:
    if not isinstance(value, str) or DOMAIN_PATTERN.fullmatch(value) is None:
        raise SeedContractError(
            "seed domain 必须以小写字母开头，并只包含字母、数字、点、横线或下划线"
        )
    return value


def derive_seed(master_seed: int, domain: str, *components: JSONValue) -> int:
    """从 master seed、域名和稳定组件派生一个 63-bit seed。

    ``components`` 常用于 rank、epoch 或 repetition ID；它们进入 canonical JSON，
    因而字符串 ``"1"`` 与整数 ``1`` 不会混淆。返回值小于 ``2**63``，可安全传给
    Python、NumPy 和 Torch 常见 seed API。
    """

    root = _validate_master_seed(master_seed)
    namespace = _validate_domain(domain)
    payload = {
        "algorithm": SEED_ALGORITHM_VERSION,
        "master_seed": root,
        "domain": namespace,
        "components": list(components),
    }
    digest = hashlib.sha256(canonical_json_bytes(payload)).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % SEED_UPPER_BOUND


@dataclass(frozen=True, slots=True)
class SeedPlan:
    """一次运行全部一级随机域及每 rank 训练流的冻结映射。"""

    master_seed: int
    domains: dict[str, int]
    rank_training: dict[str, int]
    algorithm: str = SEED_ALGORITHM_VERSION
    schema_version: str = "seed-plan-v1"

    def __post_init__(self) -> None:
        _validate_master_seed(self.master_seed)
        if self.algorithm != SEED_ALGORITHM_VERSION:
            raise SeedContractError(
                f"不支持的 seed algorithm：{self.algorithm!r}"
            )
        if self.schema_version != "seed-plan-v1":
            raise SeedContractError("SeedPlan.schema_version 必须是 seed-plan-v1")
        if not isinstance(self.domains, dict):
            raise SeedContractError("domains 必须是 object")
        missing = set(DEFAULT_SEED_DOMAINS) - set(self.domains)
        if missing:
            raise SeedContractError(f"SeedPlan 缺少必需域：{sorted(missing)}")
        for domain, seed in self.domains.items():
            _validate_domain(domain)
            _validate_master_seed(seed)
            expected = derive_seed(self.master_seed, domain)
            if seed != expected:
                raise SeedContractError(f"domains.{domain} 与派生算法不一致")
        if not isinstance(self.rank_training, dict) or not self.rank_training:
            raise SeedContractError("rank_training 必须至少包含 rank 0")
        expected_rank_keys = [str(index) for index in range(len(self.rank_training))]
        if sorted(self.rank_training, key=lambda item: int(item) if item.isdigit() else -1) != expected_rank_keys:
            raise SeedContractError("rank_training 的键必须是从 0 开始连续的十进制 rank")
        for rank_text, seed in self.rank_training.items():
            rank = int(rank_text)
            _validate_master_seed(seed)
            if seed != derive_seed(self.master_seed, "rank_training", rank):
                raise SeedContractError(f"rank_training.{rank} 与派生算法不一致")
        all_seeds = list(self.domains.values()) + list(self.rank_training.values())
        if len(all_seeds) != len(set(all_seeds)):
            raise SeedContractError("不同 seed 域发生碰撞，必须更换 master_seed")
        object.__setattr__(self, "domains", dict(sorted(self.domains.items())))
        object.__setattr__(
            self,
            "rank_training",
            dict(sorted(self.rank_training.items(), key=lambda item: int(item[0]))),
        )

    @classmethod
    def from_master_seed(
        cls,
        master_seed: int,
        *,
        world_size: int = 1,
        extra_domains: tuple[str, ...] = (),
    ) -> "SeedPlan":
        """生成完整 seed plan；额外域也必须有稳定且不重复的名称。"""

        root = _validate_master_seed(master_seed)
        if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
            raise SeedContractError("world_size 必须是正整数")
        domain_names = list(DEFAULT_SEED_DOMAINS)
        for domain in extra_domains:
            normalized = _validate_domain(domain)
            if normalized in domain_names or normalized == "rank_training":
                raise SeedContractError(f"重复或保留的 seed domain：{normalized}")
            domain_names.append(normalized)
        domains = {domain: derive_seed(root, domain) for domain in domain_names}
        ranks = {
            str(rank): derive_seed(root, "rank_training", rank)
            for rank in range(world_size)
        }
        return cls(master_seed=root, domains=domains, rank_training=ranks)

    def seed_for(self, domain: str, *, rank: int | None = None) -> int:
        """读取已冻结一级域；rank 流必须显式给出 rank。"""

        if domain == "rank_training":
            if rank is None or str(rank) not in self.rank_training:
                raise SeedContractError("读取 rank_training 时必须给出计划内 rank")
            return self.rank_training[str(rank)]
        if rank is not None:
            raise SeedContractError("只有 rank_training 域接受 rank 参数")
        try:
            return self.domains[domain]
        except KeyError as error:
            raise SeedContractError(f"未知或未冻结 seed domain：{domain}") from error

    def derive_subseed(self, domain: str, *components: JSONValue) -> int:
        """在已冻结一级域下派生 repetition/epoch 等二级 seed。"""

        parent = self.seed_for(domain)
        return derive_seed(parent, "substream", domain, *components)

    def payload_dict(self) -> dict[str, JSONValue]:
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "master_seed": self.master_seed,
            "domains": dict(self.domains),
            "rank_training": dict(self.rank_training),
        }

    @property
    def artifact_hash(self) -> str:
        return canonical_json_hash(self.payload_dict())

    def to_dict(self) -> dict[str, JSONValue]:
        value = self.payload_dict()
        value["artifact_hash"] = self.artifact_hash
        return value

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "SeedPlan":
        required = {
            "schema_version",
            "algorithm",
            "master_seed",
            "domains",
            "rank_training",
            "artifact_hash",
        }
        missing = required - set(value)
        extra = set(value) - required
        if missing or extra:
            raise SeedContractError(
                f"SeedPlan 字段错误：missing={sorted(missing)}, extra={sorted(extra)}"
            )
        if not isinstance(value["domains"], dict) or not isinstance(value["rank_training"], dict):
            raise SeedContractError("domains/rank_training 必须是 object")
        plan = cls(
            master_seed=value["master_seed"],
            domains=dict(value["domains"]),
            rank_training=dict(value["rank_training"]),
            algorithm=value["algorithm"],
            schema_version=value["schema_version"],
        )
        if value["artifact_hash"] != plan.artifact_hash:
            raise SeedContractError("SeedPlan.artifact_hash 与内容不一致")
        return plan
