"""共享合同层使用的稳定异常层级。

异常类本身不携带运行时对象，只表达调用方可以据此采取的恢复策略。这样 CLI、
实验编排器和测试不需要解析自然语言错误消息，也不会把“输入编码错误”误当成
“正式运行证据缺失”。错误消息仍尽量给出字段路径，方便定位配置或 artifact。
"""

from __future__ import annotations


class ContractError(ValueError):
    """所有可预期合同违例的基类。"""


class CanonicalJSONError(ContractError):
    """JSON 不是项目规定的严格 UTF-8 canonical 表示。"""


class ConfigContractError(ContractError):
    """配置字段、类型、覆盖关系或跨字段约束不成立。"""


class IdentityContractError(ContractError):
    """experiment/run/attempt/session 身份不满足稳定身份规则。"""


class SeedContractError(ContractError):
    """seed 域未知、复用或派生记录损坏。"""


class ProvenanceContractError(ContractError):
    """provenance 缺少追溯字段或包含不允许的状态。"""


class GateContractError(ContractError):
    """Gate 或本机验证记录不符合机器可读状态合同。"""


class FreezeContractError(ContractError):
    """合同冻结 artifact 的状态、哈希或依赖关系无效。"""


class FormalRunRejected(ContractError):
    """正式入口缺少冻结合同、方法决策或合格 Gate 证据。"""


class DependencyUnavailable(RuntimeError):
    """可选依赖未安装，且调用方确实进入了对应 adapter 边界。"""

    def __init__(
        self,
        dependency: str,
        *,
        feature: str,
        install_extra: str | None = None,
    ) -> None:
        self.dependency = dependency
        self.feature = feature
        self.install_extra = install_extra
        hint = "" if install_extra is None else f":install_extra={install_extra}"
        super().__init__(f"DEPENDENCY_UNAVAILABLE:{dependency}:feature={feature}{hint}")
