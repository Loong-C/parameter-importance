"""梯度数据源协议与本机合成实现。

本分区只描述“怎样取得固定状态下的梯度”，不负责选择估计器，也不负责
训练模型。真实 Hugging Face 适配器未来可以实现 :class:`GradientProvider`
协议，但导入本包本身永远不会触发 Transformers、Datasets 或网络访问。
"""

from .protocols import FixedStateGradientProvider, GradientBatch, GradientProvider
from .optional import (
    HuggingFaceDependencies,
    load_huggingface_dependencies,
    load_safetensors_module,
    load_tensorboard_module,
    require_optional_dependency,
)
from .synthetic import SyntheticGradientProvider

__all__ = [
    "FixedStateGradientProvider",
    "GradientBatch",
    "GradientProvider",
    "HuggingFaceDependencies",
    "SyntheticGradientProvider",
    "load_huggingface_dependencies",
    "load_safetensors_module",
    "load_tensorboard_module",
    "require_optional_dependency",
]
