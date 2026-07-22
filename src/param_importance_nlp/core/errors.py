"""纯数学核心层使用的结构化异常。

核心层刻意不依赖运行时、实验编排或第三方模型框架。这里保留少量、稳定的
异常类型，使上层可以按错误类别 fail-closed，而不必解析自然语言错误消息。
"""

from __future__ import annotations


class CoreContractError(ValueError):
    """输入违反已冻结的数学或坐标契约。"""


class RegistryError(CoreContractError):
    """参数注册表存在别名、存储或 optimizer 归属歧义。"""


class TensorMapError(CoreContractError):
    """张量映射与注册表的名称、顺序、形状或数值要求不一致。"""


class NumericalError(CoreContractError):
    """计算遇到 NaN/Inf、零分母或不支持的数值边界。"""

